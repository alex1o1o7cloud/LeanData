import Mathlib

namespace circle_area_ratio_l558_55824

theorem circle_area_ratio (r R : ℝ) (h : π * R^2 - π * r^2 = (3/4) * π * r^2) :
  R / r = Real.sqrt 7 / 2 :=
by
  sorry

end circle_area_ratio_l558_55824


namespace number_of_representatives_from_companyA_l558_55888

-- Define conditions
def companyA_representatives : ℕ := 120
def companyB_representatives : ℕ := 100
def total_selected : ℕ := 11

-- Define the theorem
theorem number_of_representatives_from_companyA : 120 * (11 / (120 + 100)) = 6 := by
  sorry

end number_of_representatives_from_companyA_l558_55888


namespace alice_sales_surplus_l558_55839

-- Define the constants
def adidas_cost : ℕ := 45
def nike_cost : ℕ := 60
def reebok_cost : ℕ := 35
def quota : ℕ := 1000

-- Define the quantities sold
def adidas_sold : ℕ := 6
def nike_sold : ℕ := 8
def reebok_sold : ℕ := 9

-- Calculate total sales
def total_sales : ℕ := adidas_sold * adidas_cost + nike_sold * nike_cost + reebok_sold * reebok_cost

-- Prove that Alice's total sales minus her quota is 65
theorem alice_sales_surplus : total_sales - quota = 65 := by
  -- Calculation is omitted here. Here is the mathematical fact to prove:
  sorry

end alice_sales_surplus_l558_55839


namespace proof_problem_l558_55814

variable (x : Int) (y : Int) (m : Real)

theorem proof_problem :
  ((x = -6 ∧ y = 1 ∧ m = 7.5) ∨ (x = -1 ∧ y = 2 ∧ m = 4)) ↔
  (-2 * x + 3 * y = 2 * m ∧ x - 5 * y = -11 ∧ x < 0 ∧ y > 0)
:= sorry

end proof_problem_l558_55814


namespace total_red_papers_l558_55841

-- Defining the number of red papers in one box and the number of boxes Hoseok has
def red_papers_per_box : ℕ := 2
def number_of_boxes : ℕ := 2

-- Statement to prove
theorem total_red_papers : (red_papers_per_box * number_of_boxes) = 4 := by
  sorry

end total_red_papers_l558_55841


namespace weight_of_new_person_l558_55802

theorem weight_of_new_person
  (avg_increase : ℝ)
  (num_persons : ℕ)
  (replaced_weight : ℝ)
  (weight_increase_total : ℝ)
  (W : ℝ)
  (h1 : avg_increase = 4.5)
  (h2 : num_persons = 8)
  (h3 : replaced_weight = 65)
  (h4 : weight_increase_total = 8 * 4.5)
  (h5 : W = replaced_weight + weight_increase_total) :
  W = 101 :=
by
  sorry

end weight_of_new_person_l558_55802


namespace quadratic_function_solution_l558_55842

theorem quadratic_function_solution :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, g (x + 1) - g x = 2 * x + 3 ∧ g 2 - g 6 = -40) :=
sorry

end quadratic_function_solution_l558_55842


namespace net_progress_l558_55820

-- Definitions based on conditions in the problem
def loss := 5
def gain := 9

-- Theorem: Proving the team's net progress
theorem net_progress : (gain - loss) = 4 :=
by
  -- Placeholder for proof
  sorry

end net_progress_l558_55820


namespace z_is_real_iff_m_values_z_in_third_quadrant_iff_m_interval_l558_55884

section
variable (m : ℝ)
def z : ℂ := (m^2 + 5 * m + 6) + (m^2 - 2 * m - 15) * Complex.I

theorem z_is_real_iff_m_values :
  (z m).im = 0 ↔ m = -3 ∨ m = 5 :=
by sorry

theorem z_in_third_quadrant_iff_m_interval :
  (z m).re < 0 ∧ (z m).im < 0 ↔ m ∈ Set.Ioo (-3) (-2) :=
by sorry
end

end z_is_real_iff_m_values_z_in_third_quadrant_iff_m_interval_l558_55884


namespace linear_function_incorrect_conclusion_C_l558_55881

theorem linear_function_incorrect_conclusion_C :
  ∀ (x y : ℝ), (y = -2 * x + 4) → ¬(∃ x, y = 0 ∧ (x = 0 ∧ y = 4)) := by
  sorry

end linear_function_incorrect_conclusion_C_l558_55881


namespace inequality_a_b_c_l558_55816

theorem inequality_a_b_c 
  (a b c : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_c : c > 0) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3 / 2 :=
by 
  sorry

end inequality_a_b_c_l558_55816


namespace vectors_are_coplanar_l558_55868

-- Definitions of the vectors a, b, and c.
def a (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -2)
def b : ℝ × ℝ × ℝ := (0, 1, 2)
def c : ℝ × ℝ × ℝ := (1, 0, 0)

-- The proof statement 
theorem vectors_are_coplanar (x : ℝ) 
  (h : ∃ m n : ℝ, a x = (n, m, 2 * m)) : 
  x = -1 :=
sorry

end vectors_are_coplanar_l558_55868


namespace grasshopper_flea_adjacency_l558_55800

-- Define the types of cells
inductive CellColor
| Red
| White

-- Define the infinite grid as a function from ℤ × ℤ to CellColor
def InfiniteGrid : Type := ℤ × ℤ → CellColor

-- Define the positions of the grasshopper and the flea
variables (g_start f_start : ℤ × ℤ)

-- The conditions for the grid and movement rules
axiom grid_conditions (grid : InfiniteGrid) :
  ∃ g_pos f_pos : ℤ × ℤ, 
  (g_pos = g_start ∧ f_pos = f_start) ∧
  (∀ x y : ℤ × ℤ, grid x = CellColor.Red ∨ grid x = CellColor.White) ∧
  (∀ x y : ℤ × ℤ, grid y = CellColor.Red ∨ grid y = CellColor.White)

-- Define the movement conditions for grasshopper and flea
axiom grasshopper_jumps (grid : InfiniteGrid) (start : ℤ × ℤ) :
  ∃ end_pos : ℤ × ℤ, grid end_pos = CellColor.Red ∧ ((end_pos.1 = start.1 ∨ end_pos.2 = start.2) ∧ abs (end_pos.1 - start.1) ≤ 1 ∧ abs (end_pos.2 - start.2) ≤ 1)

axiom flea_jumps (grid : InfiniteGrid) (start : ℤ × ℤ) :
  ∃ end_pos : ℤ × ℤ, grid end_pos = CellColor.White ∧ ((end_pos.1 = start.1 ∨ end_pos.2 = start.2) ∧ abs (end_pos.1 - start.1) ≤ 1 ∧ abs (end_pos.2 - start.2) ≤ 1)

-- The main theorem statement
theorem grasshopper_flea_adjacency (grid : InfiniteGrid)
    (g_start f_start : ℤ × ℤ) :
    ∃ pos1 pos2 pos3 : ℤ × ℤ,
    (pos1 = g_start ∨ pos1 = f_start) ∧ 
    (pos2 = g_start ∨ pos2 = f_start) ∧ 
    (abs (pos3.1 - g_start.1) + abs (pos3.2 - g_start.2) ≤ 1 ∧ 
    abs (pos3.1 - f_start.1) + abs (pos3.2 - f_start.2) ≤ 1) :=
sorry

end grasshopper_flea_adjacency_l558_55800


namespace world_cup_teams_count_l558_55801

/-- In the world cup inauguration event, captains and vice-captains of all the teams are invited and awarded welcome gifts. There are some teams participating in the world cup, and 14 gifts are needed for this event. If each team has a captain and a vice-captain, and thus receives 2 gifts, then the number of teams participating is 7. -/
theorem world_cup_teams_count (total_gifts : ℕ) (gifts_per_team : ℕ) (teams : ℕ) 
  (h1 : total_gifts = 14) 
  (h2 : gifts_per_team = 2) 
  (h3 : total_gifts = teams * gifts_per_team) 
: teams = 7 :=
by sorry

end world_cup_teams_count_l558_55801


namespace petya_vasya_same_result_l558_55889

theorem petya_vasya_same_result (a b : ℤ) 
  (h1 : b = a + 1)
  (h2 : (a - 1) / (b - 2) = (a + 1) / b) :
  (a / b) = 1 := 
by
  sorry

end petya_vasya_same_result_l558_55889


namespace nat_square_iff_divisibility_l558_55826

theorem nat_square_iff_divisibility (A : ℕ) :
  (∃ k : ℕ, A = k^2) ↔ (∀ n : ℕ, ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ n ∣ ((A + i) * (A + i) - A)) :=
sorry

end nat_square_iff_divisibility_l558_55826


namespace t_plus_inv_t_eq_three_l558_55850

theorem t_plus_inv_t_eq_three {t : ℝ} (h : t^2 - 3 * t + 1 = 0) (hnz : t ≠ 0) : t + 1 / t = 3 :=
sorry

end t_plus_inv_t_eq_three_l558_55850


namespace range_of_a_l558_55833

variable (x a : ℝ)
def inequality_sys := x < a ∧ x < 3
def solution_set := x < a

theorem range_of_a (h : ∀ x, inequality_sys x a → solution_set x a) : a ≤ 3 := by
  sorry

end range_of_a_l558_55833


namespace find_original_number_l558_55854

theorem find_original_number (x : ℝ)
  (h : (((x + 3) * 3 - 3) / 3) = 10) : x = 8 :=
sorry

end find_original_number_l558_55854


namespace triangle_inequality_l558_55890

-- Define the triangle angles, semiperimeter, and circumcircle radius
variables (α β γ s R : Real)

-- Define the sum of angles in a triangle
axiom angle_sum : α + β + γ = Real.pi

-- The inequality to prove
theorem triangle_inequality (h_sum : α + β + γ = Real.pi) :
  (α + β) * (β + γ) * (γ + α) ≤ 4 * (Real.pi / Real.sqrt 3)^3 * R / s := sorry

end triangle_inequality_l558_55890


namespace intersection_complement_l558_55829

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := { y | y ≥ 0 }
noncomputable def B : Set ℝ := { y | y ≥ 1 }

theorem intersection_complement :
  A ∩ (U \ B) = Ico 0 1 :=
by
  sorry

end intersection_complement_l558_55829


namespace problem_statement_l558_55863

theorem problem_statement (a b c : ℝ) (h : a^2 + b^2 - a * b = c^2) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a - c) * (b - c) ≤ 0 :=
by sorry

end problem_statement_l558_55863


namespace sum_common_ratios_l558_55886

variable (k p r : ℝ)
variable (hp : p ≠ r)

theorem sum_common_ratios (h : k * p ^ 2 - k * r ^ 2 = 2 * (k * p - k * r)) : 
  p + r = 2 := by
  have hk : k ≠ 0 := sorry -- From the nonconstancy condition
  sorry

end sum_common_ratios_l558_55886


namespace inequality_solution_l558_55827

theorem inequality_solution (x : ℝ) :
  (3 / 20 + |x - 13 / 60| < 7 / 30) ↔ (2 / 15 < x ∧ x < 3 / 10) :=
sorry

end inequality_solution_l558_55827


namespace hortense_flower_production_l558_55803

-- Define the initial conditions
def daisy_seeds : ℕ := 25
def sunflower_seeds : ℕ := 25
def daisy_germination_rate : ℚ := 0.60
def sunflower_germination_rate : ℚ := 0.80
def flower_production_rate : ℚ := 0.80

-- Prove the number of plants that produce flowers
theorem hortense_flower_production :
  (daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) * flower_production_rate = 28 :=
by sorry

end hortense_flower_production_l558_55803


namespace tutors_all_work_together_after_360_days_l558_55846

theorem tutors_all_work_together_after_360_days :
  ∀ (n : ℕ), (n > 0) → 
    (∃ k, k > 0 ∧ k = Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 9 10)) ∧ 
     k % 7 = 3) := by
  sorry

end tutors_all_work_together_after_360_days_l558_55846


namespace meaningful_expression_l558_55879

theorem meaningful_expression (x : ℝ) : (1 / Real.sqrt (x + 2) > 0) → (x > -2) := 
sorry

end meaningful_expression_l558_55879


namespace sum_of_coefficients_of_y_terms_l558_55844

theorem sum_of_coefficients_of_y_terms :
  let p := (5 * x + 3 * y + 2) * (2 * x + 6 * y + 7)
  let expanded_p := 10 * x^2 + 36 * x * y + 39 * x + 18 * y^2 + 33 * y + 14
  (36 + 18 + 33) = 87 := by
  sorry

end sum_of_coefficients_of_y_terms_l558_55844


namespace solve_inequality_l558_55869

theorem solve_inequality (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) → x ≥ -2 :=
by
  intros h
  sorry

end solve_inequality_l558_55869


namespace f_log3_54_l558_55806

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x ∧ x < 1 then 3^x else sorry

-- Definitions of the conditions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f (x)
def functional_equation (f : ℝ → ℝ) := ∀ x, f (x + 2) = -1 / f (x)

-- Hypotheses based on conditions
variable (f : ℝ → ℝ)
axiom f_is_odd : odd_function f
axiom f_is_periodic : periodic_function f 4
axiom f_functional : functional_equation f

-- Main goal
theorem f_log3_54 : f (Real.log 54 / Real.log 3) = -3 / 2 := by
  sorry

end f_log3_54_l558_55806


namespace compute_abc_l558_55830

theorem compute_abc (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 30) 
  (h2 : (1 / a + 1 / b + 1 / c + 420 / (a * b * c) = 1)) : 
  a * b * c = 450 := 
sorry

end compute_abc_l558_55830


namespace stone_123_is_12_l558_55807

/-- Definitions: 
  1. Fifteen stones arranged in a circle counted in a specific pattern: clockwise and counterclockwise.
  2. The sequence of stones enumerated from 1 to 123
  3. The repeating pattern occurs every 28 stones
-/
def stones_counted (n : Nat) : Nat :=
  if n % 28 <= 15 then (n % 28) else (28 - (n % 28) + 1)

theorem stone_123_is_12 : stones_counted 123 = 12 :=
by
  sorry

end stone_123_is_12_l558_55807


namespace find_y_l558_55898

theorem find_y (y : ℕ) (h1 : 27 = 3^3) (h2 : 3^9 = 27^y) : y = 3 := 
by 
  sorry

end find_y_l558_55898


namespace Xiao_Ming_vertical_height_increase_l558_55877

noncomputable def vertical_height_increase (slope_ratio_v slope_ratio_h : ℝ) (distance : ℝ) : ℝ :=
  let x := distance / (Real.sqrt (1 + (slope_ratio_h / slope_ratio_v)^2))
  x

theorem Xiao_Ming_vertical_height_increase
  (slope_ratio_v slope_ratio_h distance : ℝ)
  (h_ratio : slope_ratio_v = 1)
  (h_ratio2 : slope_ratio_h = 2.4)
  (h_distance : distance = 130) :
  vertical_height_increase slope_ratio_v slope_ratio_h distance = 50 :=
by
  unfold vertical_height_increase
  rw [h_ratio, h_ratio2, h_distance]
  sorry

end Xiao_Ming_vertical_height_increase_l558_55877


namespace length_of_BD_l558_55895

theorem length_of_BD (AB AC CB BD : ℝ) (h1 : AB = 10) (h2 : AC = 4 * CB) (h3 : AC = 4 * 2) (h4 : CB = 2) :
  BD = 3 :=
sorry

end length_of_BD_l558_55895


namespace golden_section_search_third_point_l558_55858

noncomputable def golden_ratio : ℝ := 0.618

theorem golden_section_search_third_point :
  let L₀ := 1000
  let U₀ := 2000
  let d₀ := U₀ - L₀
  let x₁ := U₀ - golden_ratio * d₀
  let x₂ := L₀ + golden_ratio * d₀
  let d₁ := U₀ - x₁
  let x₃ := x₁ + golden_ratio * d₁
  x₃ = 1764 :=
by
  sorry

end golden_section_search_third_point_l558_55858


namespace complex_pow_simplify_l558_55847

noncomputable def i : ℂ := Complex.I

theorem complex_pow_simplify :
  (1 + Real.sqrt 3 * Complex.I) ^ 3 * Complex.I = -8 * Complex.I :=
by
  sorry

end complex_pow_simplify_l558_55847


namespace angle_measure_l558_55866

theorem angle_measure (y : ℝ) (hyp : 45 + 3 * y + y = 180) : y = 33.75 :=
by
  sorry

end angle_measure_l558_55866


namespace goldfish_graph_discrete_points_l558_55885

theorem goldfish_graph_discrete_points : 
  ∀ n : ℤ, 1 ≤ n ∧ n ≤ 10 → ∃ C : ℤ, C = 20 * n + 10 ∧ ∀ m : ℤ, (1 ≤ m ∧ m ≤ 10 ∧ m ≠ n) → C ≠ (20 * m + 10) :=
by
  sorry

end goldfish_graph_discrete_points_l558_55885


namespace josette_additional_cost_l558_55871

def small_bottle_cost_eur : ℝ := 1.50
def large_bottle_cost_eur : ℝ := 2.40
def exchange_rate : ℝ := 1.20
def discount_10_percent : ℝ := 0.10
def discount_15_percent : ℝ := 0.15

def initial_small_bottles : ℕ := 3
def initial_large_bottles : ℕ := 2

def initial_total_cost_eur : ℝ :=
  (small_bottle_cost_eur * initial_small_bottles) +
  (large_bottle_cost_eur * initial_large_bottles)

def discounted_cost_eur_10 : ℝ :=
  initial_total_cost_eur * (1 - discount_10_percent)

def additional_bottle_cost_eur : ℝ := small_bottle_cost_eur

def new_total_cost_eur : ℝ :=
  initial_total_cost_eur + additional_bottle_cost_eur

def discounted_cost_eur_15 : ℝ :=
  new_total_cost_eur * (1 - discount_15_percent)

def cost_usd (eur_amount : ℝ) : ℝ :=
  eur_amount * exchange_rate

def discounted_cost_usd_10 : ℝ := cost_usd discounted_cost_eur_10
def discounted_cost_usd_15 : ℝ := cost_usd discounted_cost_eur_15

def additional_cost_usd : ℝ :=
  discounted_cost_usd_15 - discounted_cost_usd_10

theorem josette_additional_cost :
  additional_cost_usd = 0.972 :=
by 
  sorry

end josette_additional_cost_l558_55871


namespace find_value_of_c_l558_55811

-- Given: The transformed linear regression equation and the definition of z
theorem find_value_of_c (z : ℝ) (y : ℝ) (x : ℝ) (c : ℝ) (k : ℝ) (h1 : z = 0.4 * x + 2) (h2 : z = Real.log y) (h3 : y = c * Real.exp (k * x)) : 
  c = Real.exp 2 :=
by
  sorry

end find_value_of_c_l558_55811


namespace solve_diff_eq_for_k_ne_zero_solve_diff_eq_for_k_eq_zero_l558_55840

open Real

theorem solve_diff_eq_for_k_ne_zero (k : ℝ) (h : k ≠ 0) (f g : ℝ → ℝ) 
  (hf : ∀ x, deriv f x = g x * (f x + g x) ^ k)
  (hg : ∀ x, deriv g x = f x * (f x + g x) ^ k)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 0) :
  (∀ x, f x = (1 / 2) * ((1 / (1 - k * x)) ^ (1 / k) + (1 - k * x) ^ (1 / k)) ∧ g x = (1 / 2) * ((1 / (1 - k * x)) ^ (1 / k) - (1 - k * x) ^ (1 / k))) :=
sorry

theorem solve_diff_eq_for_k_eq_zero (f g : ℝ → ℝ) 
  (hf : ∀ x, deriv f x = g x)
  (hg : ∀ x, deriv g x = f x)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 0) :
  (∀ x, f x = cosh x ∧ g x = sinh x) :=
sorry

end solve_diff_eq_for_k_ne_zero_solve_diff_eq_for_k_eq_zero_l558_55840


namespace take_home_pay_correct_l558_55894

def jonessa_pay : ℝ := 500
def tax_deduction_percent : ℝ := 0.10
def insurance_deduction_percent : ℝ := 0.05
def pension_plan_deduction_percent : ℝ := 0.03
def union_dues_deduction_percent : ℝ := 0.02

def total_deductions : ℝ :=
  jonessa_pay * tax_deduction_percent +
  jonessa_pay * insurance_deduction_percent +
  jonessa_pay * pension_plan_deduction_percent +
  jonessa_pay * union_dues_deduction_percent

def take_home_pay : ℝ := jonessa_pay - total_deductions

theorem take_home_pay_correct : take_home_pay = 400 :=
  by
  sorry

end take_home_pay_correct_l558_55894


namespace find_a9_l558_55855

variable {a_n : ℕ → ℝ}

-- Definition of arithmetic progression
def is_arithmetic_progression (a : ℕ → ℝ) (a1 d : ℝ) := ∀ n : ℕ, a n = a1 + (n - 1) * d

-- Conditions
variables (a1 d : ℝ)
variable (h1 : a1 + (a1 + d)^2 = -3)
variable (h2 : ((a1 + a1 + 4 * d) * 5 / 2) = 10)

-- Question, needing the final statement
theorem find_a9 (a : ℕ → ℝ) (ha : is_arithmetic_progression a a1 d) : a 9 = 20 :=
by
    -- Since the theorem requires solving the statements, we use sorry to skip the proof.
    sorry

end find_a9_l558_55855


namespace find_hcf_l558_55813

-- Defining the conditions given in the problem
def hcf_of_two_numbers_is_H (A B H : ℕ) : Prop := Nat.gcd A B = H
def lcm_of_A_B (A B : ℕ) (H : ℕ) : Prop := Nat.lcm A B = H * 21 * 23
def larger_number_is_460 (A : ℕ) : Prop := A = 460

-- The propositional goal to prove that H = 20 given the above conditions
theorem find_hcf (A B H : ℕ) (hcf_cond : hcf_of_two_numbers_is_H A B H)
  (lcm_cond : lcm_of_A_B A B H) (larger_cond : larger_number_is_460 A) : H = 20 :=
sorry

end find_hcf_l558_55813


namespace exhibit_special_13_digit_integer_l558_55810

open Nat 

def thirteenDigitInteger (N : ℕ) : Prop :=
  N ≥ 10^12 ∧ N < 10^13

def isMultipleOf8192 (N : ℕ) : Prop :=
  8192 ∣ N

def hasOnlyEightOrNineDigits (N : ℕ) : Prop :=
  ∀ d ∈ digits 10 N, d = 8 ∨ d = 9

theorem exhibit_special_13_digit_integer : 
  ∃ N : ℕ, thirteenDigitInteger N ∧ isMultipleOf8192 N ∧ hasOnlyEightOrNineDigits N ∧ N = 8888888888888 := 
by
  sorry 

end exhibit_special_13_digit_integer_l558_55810


namespace calculate_expression_l558_55848

-- Define the expression x + x * (factorial x)^x
def expression (x : ℕ) : ℕ :=
  x + x * (Nat.factorial x) ^ x

-- Set the value of x
def x_value : ℕ := 3

-- State the proposition
theorem calculate_expression : expression x_value = 651 := 
by 
  -- By substitution and calculation, the proof follows.
  sorry

end calculate_expression_l558_55848


namespace find_x_l558_55845

theorem find_x
  (x : ℝ)
  (h : 0.20 * x = 0.40 * 140 + 80) :
  x = 680 := 
sorry

end find_x_l558_55845


namespace intersection_is_correct_l558_55831

def A : Set ℝ := {x | True}
def B : Set ℝ := {y | y ≥ 0}

theorem intersection_is_correct : A ∩ B = { x | x ≥ 0 } :=
by
  sorry

end intersection_is_correct_l558_55831


namespace dog_count_l558_55875

theorem dog_count 
  (long_furred : ℕ) 
  (brown : ℕ) 
  (neither : ℕ) 
  (long_furred_brown : ℕ) 
  (total : ℕ) 
  (h1 : long_furred = 29) 
  (h2 : brown = 17) 
  (h3 : neither = 8) 
  (h4 : long_furred_brown = 9)
  (h5 : total = long_furred + brown - long_furred_brown + neither) : 
  total = 45 :=
by 
  sorry

end dog_count_l558_55875


namespace chord_line_eq_l558_55872

theorem chord_line_eq (x y : ℝ) (h : x^2 + 4 * y^2 = 36) (midpoint : x = 4 ∧ y = 2) :
  x + 2 * y - 8 = 0 := 
sorry

end chord_line_eq_l558_55872


namespace number_of_ways_to_fill_grid_l558_55870

noncomputable def totalWaysToFillGrid (S : Finset ℕ) : ℕ :=
  S.card.choose 5

theorem number_of_ways_to_fill_grid : totalWaysToFillGrid ({1, 2, 3, 4, 5, 6} : Finset ℕ) = 6 :=
by
  sorry

end number_of_ways_to_fill_grid_l558_55870


namespace unique_four_digit_number_l558_55893

theorem unique_four_digit_number (a b c d : ℕ) (ha : 1 ≤ a) (hb : b ≤ 9) (hc : c ≤ 9) (hd : d ≤ 9)
  (h1 : a + b = c + d)
  (h2 : b + d = 2 * (a + c))
  (h3 : a + d = c)
  (h4 : b + c - a = 3 * d) :
  a = 1 ∧ b = 8 ∧ c = 5 ∧ d = 4 :=
by
  sorry

end unique_four_digit_number_l558_55893


namespace find_speed_of_man_in_still_water_l558_55857

noncomputable def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
  (v_m + v_s) * 3 = 42 ∧ (v_m - v_s) * 3 = 18

theorem find_speed_of_man_in_still_water (v_s : ℝ) : ∃ v_m : ℝ, speed_of_man_in_still_water v_m v_s ∧ v_m = 10 :=
by
  sorry

end find_speed_of_man_in_still_water_l558_55857


namespace three_digit_cubes_divisible_by_16_l558_55882

theorem three_digit_cubes_divisible_by_16 (n : ℤ) (x : ℤ) 
  (h_cube : x = n^3)
  (h_div : 16 ∣ x) 
  (h_3digit : 100 ≤ x ∧ x ≤ 999) : 
  x = 512 := 
by {
  sorry
}

end three_digit_cubes_divisible_by_16_l558_55882


namespace floral_shop_bouquets_total_l558_55897

theorem floral_shop_bouquets_total (sold_monday_rose : ℕ) (sold_monday_lily : ℕ) (sold_monday_orchid : ℕ)
  (price_monday_rose : ℕ) (price_monday_lily : ℕ) (price_monday_orchid : ℕ)
  (sold_tuesday_rose : ℕ) (sold_tuesday_lily : ℕ) (sold_tuesday_orchid : ℕ)
  (price_tuesday_rose : ℕ) (price_tuesday_lily : ℕ) (price_tuesday_orchid : ℕ)
  (sold_wednesday_rose : ℕ) (sold_wednesday_lily : ℕ) (sold_wednesday_orchid : ℕ)
  (price_wednesday_rose : ℕ) (price_wednesday_lily : ℕ) (price_wednesday_orchid : ℕ)
  (H1 : sold_monday_rose = 12) (H2 : sold_monday_lily = 8) (H3 : sold_monday_orchid = 6)
  (H4 : price_monday_rose = 10) (H5 : price_monday_lily = 15) (H6 : price_monday_orchid = 20)
  (H7 : sold_tuesday_rose = 3 * sold_monday_rose) (H8 : sold_tuesday_lily = 2 * sold_monday_lily)
  (H9 : sold_tuesday_orchid = sold_monday_orchid / 2) (H10 : price_tuesday_rose = 12)
  (H11 : price_tuesday_lily = 18) (H12 : price_tuesday_orchid = 22)
  (H13 : sold_wednesday_rose = sold_tuesday_rose / 3) (H14 : sold_wednesday_lily = sold_tuesday_lily / 4)
  (H15 : sold_wednesday_orchid = 2 * sold_tuesday_orchid / 3) (H16 : price_wednesday_rose = 8)
  (H17 : price_wednesday_lily = 12) (H18 : price_wednesday_orchid = 16) :
  (sold_monday_rose + sold_tuesday_rose + sold_wednesday_rose = 60) ∧
  (sold_monday_lily + sold_tuesday_lily + sold_wednesday_lily = 28) ∧
  (sold_monday_orchid + sold_tuesday_orchid + sold_wednesday_orchid = 11) ∧
  ((sold_monday_rose * price_monday_rose + sold_tuesday_rose * price_tuesday_rose + sold_wednesday_rose * price_wednesday_rose) = 648) ∧
  ((sold_monday_lily * price_monday_lily + sold_tuesday_lily * price_tuesday_lily + sold_wednesday_lily * price_wednesday_lily) = 456) ∧
  ((sold_monday_orchid * price_monday_orchid + sold_tuesday_orchid * price_tuesday_orchid + sold_wednesday_orchid * price_wednesday_orchid) = 218) ∧
  ((sold_monday_rose + sold_tuesday_rose + sold_wednesday_rose + sold_monday_lily + sold_tuesday_lily + sold_wednesday_lily + sold_monday_orchid + sold_tuesday_orchid + sold_wednesday_orchid) = 99) ∧
  ((sold_monday_rose * price_monday_rose + sold_tuesday_rose * price_tuesday_rose + sold_wednesday_rose * price_wednesday_rose + sold_monday_lily * price_monday_lily + sold_tuesday_lily * price_tuesday_lily + sold_wednesday_lily * price_wednesday_lily + sold_monday_orchid * price_monday_orchid + sold_tuesday_orchid * price_tuesday_orchid + sold_wednesday_orchid * price_wednesday_orchid) = 1322) :=
  by sorry

end floral_shop_bouquets_total_l558_55897


namespace expression_value_l558_55815

theorem expression_value (x y : ℝ) (h : x + y = -1) : 
  x^4 + 5 * x^3 * y + x^2 * y + 8 * x^2 * y^2 + x * y^2 + 5 * x * y^3 + y^4 = 1 :=
by
  sorry

end expression_value_l558_55815


namespace overall_cost_for_all_projects_l558_55834

-- Define the daily salaries including 10% taxes and insurance.
def daily_salary_entry_level_worker : ℕ := 100 + 10
def daily_salary_experienced_worker : ℕ := 130 + 13
def daily_salary_electrician : ℕ := 2 * 100 + 20
def daily_salary_plumber : ℕ := 250 + 25
def daily_salary_architect : ℕ := (35/10) * 100 + 35

-- Define the total cost for each project.
def project1_cost : ℕ :=
  daily_salary_entry_level_worker +
  daily_salary_experienced_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

def project2_cost : ℕ :=
  2 * daily_salary_experienced_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

def project3_cost : ℕ :=
  2 * daily_salary_entry_level_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

-- Define the overall cost for all three projects.
def total_cost : ℕ :=
  project1_cost + project2_cost + project3_cost

theorem overall_cost_for_all_projects :
  total_cost = 3399 :=
by
  sorry

end overall_cost_for_all_projects_l558_55834


namespace initial_population_is_9250_l558_55891

noncomputable def initial_population : ℝ :=
  let final_population := 6514
  let factor := (1.08 * 0.85 * (1.02)^5 * 0.95 * 0.9)
  final_population / factor

theorem initial_population_is_9250 : initial_population = 9250 := by
  sorry

end initial_population_is_9250_l558_55891


namespace net_profit_correct_l558_55808

-- Define the conditions
def unit_price : ℝ := 1.25
def selling_price : ℝ := 12
def num_patches : ℕ := 100

-- Define the required total cost
def total_cost : ℝ := num_patches * unit_price

-- Define the required total revenue
def total_revenue : ℝ := num_patches * selling_price

-- Define the net profit calculation
def net_profit : ℝ := total_revenue - total_cost

-- The theorem we need to prove
theorem net_profit_correct : net_profit = 1075 := by
    sorry

end net_profit_correct_l558_55808


namespace ball_first_bounce_less_than_30_l558_55819

theorem ball_first_bounce_less_than_30 (b : ℕ) :
  (243 * ((2: ℝ) / 3) ^ b < 30) ↔ (b ≥ 6) :=
sorry

end ball_first_bounce_less_than_30_l558_55819


namespace serving_calculation_correct_l558_55859

def prepared_orange_juice_servings (cans_of_concentrate : ℕ) 
                                  (oz_per_concentrate_can : ℕ) 
                                  (water_ratio : ℕ) 
                                  (oz_per_serving : ℕ) : ℕ :=
  let total_concentrate := cans_of_concentrate * oz_per_concentrate_can
  let total_water := cans_of_concentrate * water_ratio * oz_per_concentrate_can
  let total_juice := total_concentrate + total_water
  total_juice / oz_per_serving

theorem serving_calculation_correct :
  prepared_orange_juice_servings 60 5 3 6 = 200 := by
  sorry

end serving_calculation_correct_l558_55859


namespace polynomial_factorization_l558_55864

theorem polynomial_factorization (x y : ℝ) : -(2 * x - y) * (2 * x + y) = -4 * x ^ 2 + y ^ 2 :=
by sorry

end polynomial_factorization_l558_55864


namespace ways_to_reach_5_5_l558_55838

def moves_to_destination : ℕ → ℕ → ℕ
| 0, 0     => 1
| 0, j+1   => moves_to_destination 0 j
| i+1, 0   => moves_to_destination i 0
| i+1, j+1 => moves_to_destination i (j+1) + moves_to_destination (i+1) j + moves_to_destination i j

theorem ways_to_reach_5_5 : moves_to_destination 5 5 = 1573 := by
  sorry

end ways_to_reach_5_5_l558_55838


namespace annual_interest_rate_l558_55853

/-- Suppose you invested $10000, part at a certain annual interest rate and the rest at 9% annual interest.
After one year, you received $684 in interest. You invested $7200 at this rate and the rest at 9%.
What is the annual interest rate of the first investment? -/
theorem annual_interest_rate (r : ℝ) 
  (h : 7200 * r + 2800 * 0.09 = 684) : r = 0.06 :=
by
  sorry

end annual_interest_rate_l558_55853


namespace six_digit_number_condition_l558_55822

theorem six_digit_number_condition (a b c : ℕ) (h : 1 ≤ a ∧ a ≤ 9) (hb : b < 10) (hc : c < 10) : 
  ∃ k : ℕ, 100000 * a + 10000 * b + 1000 * c + 100 * (2 * a) + 10 * (2 * b) + 2 * c = 2 * k := 
by
  sorry

end six_digit_number_condition_l558_55822


namespace smallest_sum_of_4_numbers_l558_55899

noncomputable def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

noncomputable def not_relatively_prime (a b : ℕ) : Prop :=
  ¬ relatively_prime a b

noncomputable def problem_statement : Prop :=
  ∃ (V1 V2 V3 V4 : ℕ), 
  relatively_prime V1 V3 ∧ 
  relatively_prime V2 V4 ∧ 
  not_relatively_prime V1 V2 ∧ 
  not_relatively_prime V1 V4 ∧ 
  not_relatively_prime V2 V3 ∧ 
  not_relatively_prime V3 V4 ∧ 
  V1 + V2 + V3 + V4 = 60

theorem smallest_sum_of_4_numbers : problem_statement := sorry

end smallest_sum_of_4_numbers_l558_55899


namespace range_of_a_l558_55804

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end range_of_a_l558_55804


namespace solve_comb_eq_l558_55874

open Nat

def comb (n k : ℕ) : ℕ := (factorial n) / ((factorial k) * (factorial (n - k)))
def perm (n k : ℕ) : ℕ := (factorial n) / (factorial (n - k))

theorem solve_comb_eq (x : ℕ) :
  comb (x + 5) x = comb (x + 3) (x - 1) + comb (x + 3) (x - 2) + 3/4 * perm (x + 3) 3 ->
  x = 14 := 
by 
  sorry

end solve_comb_eq_l558_55874


namespace elizabeth_initial_bottles_l558_55817

theorem elizabeth_initial_bottles (B : ℕ) (H1 : B - 2 - 1 = (3 * X) → 3 * (B - 3) = 21) : B = 10 :=
by
  sorry

end elizabeth_initial_bottles_l558_55817


namespace least_number_of_marbles_divisible_by_2_3_4_5_6_7_l558_55821

theorem least_number_of_marbles_divisible_by_2_3_4_5_6_7 : 
  ∃ n : ℕ, (∀ k ∈ [2, 3, 4, 5, 6, 7], k ∣ n) ∧ n = 420 :=
  by sorry

end least_number_of_marbles_divisible_by_2_3_4_5_6_7_l558_55821


namespace relationship_y1_y2_y3_l558_55805

noncomputable def parabola_value (x m : ℝ) : ℝ := -x^2 - 4 * x + m

variable (m y1 y2 y3 : ℝ)

def point_A_on_parabola : Prop := y1 = parabola_value (-3) m
def point_B_on_parabola : Prop := y2 = parabola_value (-2) m
def point_C_on_parabola : Prop := y3 = parabola_value 1 m


theorem relationship_y1_y2_y3 (hA : point_A_on_parabola y1 m)
                              (hB : point_B_on_parabola y2 m)
                              (hC : point_C_on_parabola y3 m) :
  y2 > y1 ∧ y1 > y3 := 
  sorry

end relationship_y1_y2_y3_l558_55805


namespace non_equivalent_paintings_wheel_l558_55896

theorem non_equivalent_paintings_wheel :
  let num_sections := 7
  let num_colors := 2
  let total_paintings := num_colors ^ num_sections
  let single_color_cases := 2
  let non_single_color_paintings := total_paintings - single_color_cases
  let equivalent_rotation_count := num_sections
  (non_single_color_paintings / equivalent_rotation_count) + single_color_cases = 20 :=
by
  let num_sections := 7
  let num_colors := 2
  let total_paintings := num_colors ^ num_sections
  let single_color_cases := 2
  let non_single_color_paintings := total_paintings - single_color_cases
  let equivalent_rotation_count := num_sections
  have h1 := (non_single_color_paintings / equivalent_rotation_count) + single_color_cases
  sorry

end non_equivalent_paintings_wheel_l558_55896


namespace swimming_championship_l558_55851

theorem swimming_championship (num_swimmers : ℕ) (lanes : ℕ) (advance : ℕ) (eliminated : ℕ) (total_races : ℕ) : 
  num_swimmers = 300 → 
  lanes = 8 → 
  advance = 2 → 
  eliminated = 6 → 
  total_races = 53 :=
by
  intros
  sorry

end swimming_championship_l558_55851


namespace expand_polynomial_l558_55849

noncomputable def polynomial_expression (x : ℝ) : ℝ := -2 * (x - 3) * (x + 4) * (2 * x - 1)

theorem expand_polynomial (x : ℝ) :
  polynomial_expression x = -4 * x^3 - 2 * x^2 + 50 * x - 24 :=
sorry

end expand_polynomial_l558_55849


namespace max_non_equivalent_100_digit_numbers_l558_55812

noncomputable def maxPairwiseNonEquivalentNumbers : ℕ := 21^5

theorem max_non_equivalent_100_digit_numbers :
  (∀ (n : ℕ), 0 < n ∧ n < 100 → (∀ (digit : Fin n → Fin 2), 
  ∃ (max_num : ℕ), max_num = maxPairwiseNonEquivalentNumbers)) :=
by sorry

end max_non_equivalent_100_digit_numbers_l558_55812


namespace find_b_l558_55860

noncomputable def circle_center_radius : Prop :=
  let C := (2, 0) -- center
  let r := 2 -- radius
  C.1 = 2 ∧ C.2 = 0 ∧ r = 2

noncomputable def line (b : ℝ) : Prop :=
  ∃ M N : ℝ × ℝ, M ≠ N ∧ 
  (M.2 = M.1 + b) ∧ (N.2 = N.1 + b) -- points on the line are M = (x1, x1 + b) and N = (x2, x2 + b)

noncomputable def perpendicular_condition (M N center: ℝ × ℝ) : Prop :=
  (M.1 - center.1) * (N.1 - center.1) + (M.2 - center.2) * (N.2 - center.2) = 0 -- CM ⟂ CN

theorem find_b (b : ℝ) : 
  circle_center_radius ∧
  (∃ M N, line b ∧ perpendicular_condition M N (2, 0)) →
  b = 0 ∨ b = -4 :=
by {
  -- Proof omitted
  sorry
}

end find_b_l558_55860


namespace max_value_bx_plus_a_l558_55892

variable (a b : ℝ)

theorem max_value_bx_plus_a (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ |b * x + a| = 2 :=
by
  -- Proof goes here
  sorry

end max_value_bx_plus_a_l558_55892


namespace sufficient_not_necessary_condition_l558_55809

theorem sufficient_not_necessary_condition (a b : ℝ) (h1 : a > 1) (h2 : b > 2) : a + b > 3 :=
by
  sorry

end sufficient_not_necessary_condition_l558_55809


namespace quotient_of_fifths_l558_55861

theorem quotient_of_fifths : (2 / 5) / (1 / 5) = 2 := 
  by 
    sorry

end quotient_of_fifths_l558_55861


namespace Frank_is_14_l558_55837

variable {d e f : ℕ}

theorem Frank_is_14
  (h1 : d + e + f = 30)
  (h2 : f - 5 = d)
  (h3 : e + 2 = 3 * (d + 2) / 4) :
  f = 14 :=
sorry

end Frank_is_14_l558_55837


namespace jenna_round_trip_pay_l558_55836

-- Definitions based on conditions
def rate : ℝ := 0.40
def one_way_distance : ℝ := 400
def round_trip_distance : ℝ := 2 * one_way_distance

-- Theorem based on the question and correct answer
theorem jenna_round_trip_pay : round_trip_distance * rate = 320 := by
  sorry

end jenna_round_trip_pay_l558_55836


namespace six_letter_words_count_l558_55832

def first_letter_possibilities := 26
def second_letter_possibilities := 26
def third_letter_possibilities := 26
def fourth_letter_possibilities := 26

def number_of_six_letter_words : Nat := 
  first_letter_possibilities * 
  second_letter_possibilities * 
  third_letter_possibilities * 
  fourth_letter_possibilities

theorem six_letter_words_count : number_of_six_letter_words = 456976 := by
  sorry

end six_letter_words_count_l558_55832


namespace increasing_interval_of_f_l558_55873

def f (x : ℝ) : ℝ := (x - 1) ^ 2 - 2

theorem increasing_interval_of_f : ∀ x, 1 < x → f x > f 1 := 
sorry

end increasing_interval_of_f_l558_55873


namespace M_eq_N_l558_55876

def M : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k + 1) * Real.pi}
def N : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k - 1) * Real.pi}

theorem M_eq_N : M = N := by
  sorry

end M_eq_N_l558_55876


namespace certain_number_is_two_l558_55823

theorem certain_number_is_two (n : ℕ) 
  (h1 : 1 = 62) 
  (h2 : 363 = 3634) 
  (h3 : 3634 = n) 
  (h4 : n = 365) 
  (h5 : 36 = 2) : 
  n = 2 := 
by 
  sorry

end certain_number_is_two_l558_55823


namespace salad_cucumbers_l558_55883

theorem salad_cucumbers (c t : ℕ) 
  (h1 : c + t = 280)
  (h2 : t = 3 * c) : c = 70 :=
sorry

end salad_cucumbers_l558_55883


namespace return_trip_speed_l558_55856

theorem return_trip_speed (d xy_dist : ℝ) (s xy_speed : ℝ) (avg_speed : ℝ) (r return_speed : ℝ) :
  xy_dist = 150 →
  xy_speed = 75 →
  avg_speed = 50 →
  2 * xy_dist / ((xy_dist / xy_speed) + (xy_dist / return_speed)) = avg_speed →
  return_speed = 37.5 :=
by
  intros hxy_dist hxy_speed h_avg_speed h_avg_speed_eq
  sorry

end return_trip_speed_l558_55856


namespace work_days_difference_l558_55865

theorem work_days_difference (d_a d_b : ℕ) (H1 : d_b = 15) (H2 : d_a = d_b / 3) : 15 - d_a = 10 := by
  sorry

end work_days_difference_l558_55865


namespace geometric_sequence_div_sum_l558_55852

noncomputable def a (n : ℕ) : ℝ := sorry

noncomputable def S (n : ℕ) : ℝ := sorry

theorem geometric_sequence_div_sum 
  (h₁ : S 3 = (1 - (2 : ℝ) ^ 3) / (1 - (2 : ℝ) ^ 2) * a 1)
  (h₂ : S 2 = (1 - (2 : ℝ) ^ 2) / (1 - 2) * a 1)
  (h₃ : 8 * a 2 = a 5) : 
  S 3 / S 2 = 7 / 3 := 
by
  sorry

end geometric_sequence_div_sum_l558_55852


namespace green_folder_stickers_l558_55818

theorem green_folder_stickers (total_stickers red_sheets blue_sheets : ℕ) (red_sticker_per_sheet blue_sticker_per_sheet green_stickers_needed green_sheets : ℕ) :
  total_stickers = 60 →
  red_sticker_per_sheet = 3 →
  blue_sticker_per_sheet = 1 →
  red_sheets = 10 →
  blue_sheets = 10 →
  green_sheets = 10 →
  let red_stickers_total := red_sticker_per_sheet * red_sheets
  let blue_stickers_total := blue_sticker_per_sheet * blue_sheets
  let green_stickers_total := total_stickers - (red_stickers_total + blue_stickers_total)
  green_sticker_per_sheet = green_stickers_total / green_sheets →
  green_sticker_per_sheet = 2 := 
sorry

end green_folder_stickers_l558_55818


namespace modulo_17_residue_l558_55867

theorem modulo_17_residue : (3^4 + 6 * 49 + 8 * 137 + 7 * 34) % 17 = 5 := 
by
  sorry

end modulo_17_residue_l558_55867


namespace jerry_age_l558_55887

theorem jerry_age (M J : ℕ) (h1 : M = 4 * J - 8) (h2 : M = 24) : J = 8 :=
by
  sorry

end jerry_age_l558_55887


namespace sum_of_coordinates_of_A_l558_55862

noncomputable def point := (ℝ × ℝ)
def B : point := (2, 6)
def C : point := (4, 12)
def AC (A C : point) : ℝ := (A.1 - C.1)^2 + (A.2 - C.2)^2
def AB (A B : point) : ℝ := (A.1 - B.1)^2 + (A.2 - B.2)^2
def BC (B C : point) : ℝ := (B.1 - C.1)^2 + (B.2 - C.2)^2

theorem sum_of_coordinates_of_A :
  ∃ A : point, AC A C / AB A B = (1/3) ∧ BC B C / AB A B = (1/3) ∧ A.1 + A.2 = 24 :=
by
  sorry

end sum_of_coordinates_of_A_l558_55862


namespace aria_spent_on_cookies_in_march_l558_55825

/-- Aria purchased 4 cookies each day for the entire month of March,
    each cookie costs 19 dollars, and March has 31 days.
    Prove that the total amount Aria spent on cookies in March is 2356 dollars. -/
theorem aria_spent_on_cookies_in_march :
  (4 * 31) * 19 = 2356 := 
by 
  sorry

end aria_spent_on_cookies_in_march_l558_55825


namespace smallest_n_for_multiple_of_11_l558_55835

theorem smallest_n_for_multiple_of_11 
  (x y : ℤ) 
  (hx : x ≡ -2 [ZMOD 11]) 
  (hy : y ≡ 2 [ZMOD 11]) : 
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n ≡ 0 [ZMOD 11]) ∧ n = 7 :=
sorry

end smallest_n_for_multiple_of_11_l558_55835


namespace equal_games_per_month_l558_55828

-- Define the given conditions
def total_games : ℕ := 27
def months : ℕ := 3
def games_per_month := total_games / months

-- Proposition that needs to be proven
theorem equal_games_per_month : games_per_month = 9 := 
by
  sorry

end equal_games_per_month_l558_55828


namespace negation_of_existence_implies_universal_l558_55880

theorem negation_of_existence_implies_universal :
  ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_of_existence_implies_universal_l558_55880


namespace triangle_inequality_l558_55878

variables (a b c : ℝ)

theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) / (1 + a + b) > c / (1 + c) :=
sorry

end triangle_inequality_l558_55878


namespace find_number_l558_55843

def initial_condition (x : ℝ) : Prop :=
  ((x + 7) * 3 - 12) / 6 = -8

theorem find_number (x : ℝ) (h : initial_condition x) : x = -19 := by
  sorry

end find_number_l558_55843
