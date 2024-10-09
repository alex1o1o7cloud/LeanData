import Mathlib

namespace derivative_at_one_is_four_l2408_240841

-- Define the function y = x^2 + 2x + 1
def f (x : ℝ) := x^2 + 2*x + 1

-- State the theorem: The derivative of f at x = 1 is 4
theorem derivative_at_one_is_four : (deriv f 1) = 4 :=
by
  -- The proof is omitted here.
  sorry

end derivative_at_one_is_four_l2408_240841


namespace integer_roots_l2408_240862

noncomputable def is_quadratic_root (p q x : ℝ) : Prop :=
  x^2 + p * x + q = 0

theorem integer_roots (p q x1 x2 : ℝ)
  (hq1 : is_quadratic_root p q x1)
  (hq2 : is_quadratic_root p q x2)
  (hd : x1 ≠ x2)
  (hx : |x1 - x2| = 1)
  (hpq : |p - q| = 1) :
  (∃ (p_int q_int x1_int x2_int : ℤ), 
      p = p_int ∧ q = q_int ∧ x1 = x1_int ∧ x2 = x2_int) :=
sorry

end integer_roots_l2408_240862


namespace eight_b_plus_one_composite_l2408_240887

theorem eight_b_plus_one_composite (a b : ℕ) (h₀ : a > b)
  (h₁ : a - b = 5 * b^2 - 4 * a^2) : ∃ (n m : ℕ), 1 < n ∧ 1 < m ∧ (8 * b + 1) = n * m :=
by
  sorry

end eight_b_plus_one_composite_l2408_240887


namespace probabilityOfWearingSunglassesGivenCap_l2408_240806

-- Define the conditions as Lean constants
def peopleWearingSunglasses : ℕ := 80
def peopleWearingCaps : ℕ := 60
def probabilityOfWearingCapGivenSunglasses : ℚ := 3 / 8
def peopleWearingBoth : ℕ := (3 / 8) * 80

-- Prove the desired probability
theorem probabilityOfWearingSunglassesGivenCap : (peopleWearingBoth / peopleWearingCaps = 1 / 2) :=
by
  -- sorry is used here to skip the proof
  sorry

end probabilityOfWearingSunglassesGivenCap_l2408_240806


namespace minimum_value_l2408_240871

noncomputable def min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : ℝ :=
  (x + y + z) * (1 / (x + y) + 1 / (x + z) + 1 / (y + z))

theorem minimum_value : ∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
  (x + y + z) * (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ≥ 9 / 2 :=
by
  intro x y z hx hy hz
  sorry

end minimum_value_l2408_240871


namespace mia_12th_roll_last_is_approximately_027_l2408_240827

noncomputable def mia_probability_last_roll_on_12th : ℚ :=
  (5/6) ^ 10 * (1/6)

theorem mia_12th_roll_last_is_approximately_027 : 
  abs (mia_probability_last_roll_on_12th - 0.027) < 0.001 :=
sorry

end mia_12th_roll_last_is_approximately_027_l2408_240827


namespace leopards_arrangement_l2408_240874

theorem leopards_arrangement :
  let total_leopards := 9
  let ends_leopards := 2
  let middle_leopard := 1
  let remaining_leopards := total_leopards - ends_leopards - middle_leopard
  (2 * 1 * (Nat.factorial remaining_leopards) = 1440) := by
  sorry

end leopards_arrangement_l2408_240874


namespace john_paid_8000_l2408_240893

-- Define the variables according to the conditions
def upfront_fee : ℕ := 1000
def hourly_rate : ℕ := 100
def court_hours : ℕ := 50
def prep_hours : ℕ := 2 * court_hours
def total_hours : ℕ := court_hours + prep_hours
def total_fee : ℕ := upfront_fee + total_hours * hourly_rate
def john_share : ℕ := total_fee / 2

-- Prove that John's share is $8,000
theorem john_paid_8000 : john_share = 8000 :=
by sorry

end john_paid_8000_l2408_240893


namespace group_is_abelian_l2408_240837

variable {G : Type} [Group G]
variable (e : G)
variable (h : ∀ x : G, x * x = e)

theorem group_is_abelian (a b : G) : a * b = b * a :=
sorry

end group_is_abelian_l2408_240837


namespace probability_suitable_joint_given_physique_l2408_240866

noncomputable def total_children : ℕ := 20
noncomputable def suitable_physique : ℕ := 4
noncomputable def suitable_joint_structure : ℕ := 5
noncomputable def both_physique_and_joint : ℕ := 2

noncomputable def P (n m : ℕ) : ℚ := n / m

theorem probability_suitable_joint_given_physique :
  P both_physique_and_joint total_children / P suitable_physique total_children = 1 / 2 :=
by
  sorry

end probability_suitable_joint_given_physique_l2408_240866


namespace complex_number_solution_l2408_240861

theorem complex_number_solution (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * ⟨1, -a⟩ = (2 : ℂ)) : a = 1 :=
sorry

end complex_number_solution_l2408_240861


namespace arithmetic_seq_a7_value_l2408_240892

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ): Prop := 
  ∀ n : ℕ, a (n+1) = a n + d

theorem arithmetic_seq_a7_value
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : a 4 = 4)
  (h3 : a 3 + a 8 = 5) :
  a 7 = 1 := 
sorry

end arithmetic_seq_a7_value_l2408_240892


namespace running_time_difference_l2408_240838

theorem running_time_difference :
  ∀ (distance speed usual_speed : ℝ), 
  distance = 30 →
  usual_speed = 10 →
  speed = (distance / (usual_speed / 2)) - (distance / (usual_speed * 1.5)) →
  speed = 4 :=
by
  intros distance speed usual_speed hd hu hs
  sorry

end running_time_difference_l2408_240838


namespace pencils_distributed_per_container_l2408_240855

noncomputable def total_pencils (initial_pencils : ℕ) (additional_pencils : ℕ) : ℕ :=
  initial_pencils + additional_pencils

noncomputable def pencils_per_container (total_pencils : ℕ) (num_containers : ℕ) : ℕ :=
  total_pencils / num_containers

theorem pencils_distributed_per_container :
  let initial_pencils := 150
  let additional_pencils := 30
  let num_containers := 5
  let total := total_pencils initial_pencils additional_pencils
  let pencils_per_container := pencils_per_container total num_containers
  pencils_per_container = 36 :=
by {
  -- sorry is used to skip the proof
  -- the actual proof is not required
  sorry
}

end pencils_distributed_per_container_l2408_240855


namespace cos_theta_plus_pi_over_3_l2408_240873

theorem cos_theta_plus_pi_over_3 {θ : ℝ} (h : Real.sin (θ / 2 + π / 6) = 2 / 3) :
  Real.cos (θ + π / 3) = 1 / 9 :=
by
  sorry

end cos_theta_plus_pi_over_3_l2408_240873


namespace dot_product_expression_max_value_of_dot_product_l2408_240856

variable (x : ℝ)
variable (k : ℤ)
variable (a : ℝ × ℝ := (Real.cos x, -1 + Real.sin x))
variable (b : ℝ × ℝ := (2 * Real.cos x, Real.sin x))

theorem dot_product_expression :
  (a.1 * b.1 + a.2 * b.2) = 2 - 3 * (Real.sin x)^2 - (Real.sin x) := 
sorry

theorem max_value_of_dot_product :
  ∃ (x : ℝ), 2 - 3 * (Real.sin x)^2 - (Real.sin x) = 9 / 4 ∧ 
  (Real.sin x = -1/2 ∧ 
  (x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ x = 11 * Real.pi / 6 + 2 * k * Real.pi)) := 
sorry

end dot_product_expression_max_value_of_dot_product_l2408_240856


namespace find_rate_l2408_240832

def plan1_cost (minutes : ℕ) : ℝ :=
  if minutes <= 500 then 50 else 50 + (minutes - 500) * 0.35

def plan2_cost (minutes : ℕ) (x : ℝ) : ℝ :=
  if minutes <= 1000 then 75 else 75 + (minutes - 1000) * x

theorem find_rate (x : ℝ) :
  plan1_cost 2500 = plan2_cost 2500 x → x = 0.45 := by
  sorry

end find_rate_l2408_240832


namespace johns_father_fraction_l2408_240811

theorem johns_father_fraction (total_money : ℝ) (given_to_mother_fraction remaining_after_father : ℝ) :
  total_money = 200 →
  given_to_mother_fraction = 3 / 8 →
  remaining_after_father = 65 →
  ((total_money - given_to_mother_fraction * total_money) - remaining_after_father) / total_money
  = 3 / 10 :=
by
  intros h1 h2 h3
  sorry

end johns_father_fraction_l2408_240811


namespace gcd_ab_l2408_240885

def a := 59^7 + 1
def b := 59^7 + 59^3 + 1

theorem gcd_ab : Nat.gcd a b = 1 := by
  sorry

end gcd_ab_l2408_240885


namespace sqrt8_sub_sqrt2_eq_sqrt2_l2408_240833

theorem sqrt8_sub_sqrt2_eq_sqrt2 : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt8_sub_sqrt2_eq_sqrt2_l2408_240833


namespace prob_all_fail_prob_at_least_one_pass_l2408_240839

def prob_pass := 1 / 2
def prob_fail := 1 - prob_pass

def indep (A B C : Prop) : Prop := true -- Usually we prove independence in a detailed manner, but let's assume it's given as true.

theorem prob_all_fail (A B C : Prop) (hA : prob_pass = 1 / 2) (hB : prob_pass = 1 / 2) (hC : prob_pass = 1 / 2) 
  (indepABC : indep A B C) : (prob_fail * prob_fail * prob_fail) = 1 / 8 :=
by
  sorry

theorem prob_at_least_one_pass (A B C : Prop) (hA : prob_pass = 1 / 2) (hB : prob_pass = 1 / 2) (hC : prob_pass = 1 / 2) 
  (indepABC : indep A B C) : 1 - (prob_fail * prob_fail * prob_fail) = 7 / 8 :=
by
  sorry

end prob_all_fail_prob_at_least_one_pass_l2408_240839


namespace batsman_average_increase_l2408_240880

-- Definitions to capture the initial conditions
def runs_scored_in_17th_inning : ℕ := 74
def average_after_17_innings : ℕ := 26

-- Statement to prove the increment in average is 3 runs per inning
theorem batsman_average_increase (A : ℕ) (initial_avg : ℕ)
  (h_initial_runs : 16 * initial_avg + 74 = 17 * 26) :
  26 - initial_avg = 3 :=
by
  sorry

end batsman_average_increase_l2408_240880


namespace intersection_volume_is_zero_l2408_240850

-- Definitions of the regions
def region1 (x y z : ℝ) : Prop := |x| + |y| + |z| ≤ 2
def region2 (x y z : ℝ) : Prop := |x| + |y| + |z - 2| ≤ 1

-- Main theorem stating the volume of their intersection
theorem intersection_volume_is_zero : 
  ∀ (x y z : ℝ), region1 x y z ∧ region2 x y z → (x = 0 ∧ y = 0 ∧ z = 2) := 
sorry

end intersection_volume_is_zero_l2408_240850


namespace cost_per_use_l2408_240844

def cost : ℕ := 30
def uses_in_a_week : ℕ := 3
def weeks : ℕ := 2
def total_uses : ℕ := uses_in_a_week * weeks

theorem cost_per_use : cost / total_uses = 5 := by
  sorry

end cost_per_use_l2408_240844


namespace problem_D_l2408_240889

theorem problem_D (a b c : ℝ) (h : |a^2 + b + c| + |a + b^2 - c| ≤ 1) : a^2 + b^2 + c^2 < 100 := 
sorry

end problem_D_l2408_240889


namespace mary_flour_indeterminate_l2408_240809

theorem mary_flour_indeterminate 
  (sugar : ℕ) (flour : ℕ) (salt : ℕ) (needed_sugar_more : ℕ) 
  (h_sugar : sugar = 11) (h_flour : flour = 6)
  (h_salt : salt = 9) (h_condition : needed_sugar_more = 2) :
  ∃ (current_flour : ℕ), current_flour ≠ current_flour :=
by
  sorry

end mary_flour_indeterminate_l2408_240809


namespace weight_of_berries_l2408_240823

theorem weight_of_berries (total_weight : ℝ) (melon_weight : ℝ) : total_weight = 0.63 → melon_weight = 0.25 → total_weight - melon_weight = 0.38 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end weight_of_berries_l2408_240823


namespace find_k_l2408_240814

theorem find_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (h3 : k ≠ 1) (h4 : 1/a + 2/b = 1) : k = 18 := by
  sorry

end find_k_l2408_240814


namespace engineers_percentage_calculation_l2408_240891

noncomputable def percentageEngineers (num_marketers num_engineers num_managers total_salary: ℝ) : ℝ := 
  let num_employees := num_marketers + num_engineers + num_managers 
  if num_employees = 0 then 0 else num_engineers / num_employees * 100

theorem engineers_percentage_calculation : 
  let marketers_percentage := 0.7 
  let engineers_salary := 80000
  let average_salary := 80000
  let marketers_salary_total := 50000 * marketers_percentage 
  let managers_total_percent := 1 - marketers_percentage - x / 100
  let managers_salary := 370000 * managers_total_percent 
  marketers_salary_total + engineers_salary * x / 100 + managers_salary = average_salary -> 
  x = 22.76 
:= 
sorry

end engineers_percentage_calculation_l2408_240891


namespace find_k_l2408_240836

-- Define point type and distances
structure Point :=
(x : ℝ)
(y : ℝ)

def dist (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Condition: H is the orthocenter of triangle ABC
variable (A B C H Q : Point)
variable (H_is_orthocenter : ∀ P : Point, dist P H = dist P A + dist P B - dist A B)

-- Prove the given equation
theorem find_k :
  dist Q A + dist Q B + dist Q C = 3 * dist Q H + dist H A + dist H B + dist H C :=
sorry

end find_k_l2408_240836


namespace sum_of_consecutive_even_integers_l2408_240852

theorem sum_of_consecutive_even_integers (a : ℤ) (h : a + (a + 6) = 136) :
  a + (a + 2) + (a + 4) + (a + 6) = 272 :=
by
  sorry

end sum_of_consecutive_even_integers_l2408_240852


namespace find_m_l2408_240859

-- Define the operation a * b
def star (a b : ℝ) : ℝ := a * b + a - 2 * b

theorem find_m (m : ℝ) (h : star 3 m = 17) : m = 14 :=
by
  -- Placeholder for the proof
  sorry

end find_m_l2408_240859


namespace floor_of_neg_sqrt_frac_l2408_240800

theorem floor_of_neg_sqrt_frac :
  (Int.floor (-Real.sqrt (64 / 9)) = -3) :=
by
  sorry

end floor_of_neg_sqrt_frac_l2408_240800


namespace simplify_expr_l2408_240897

theorem simplify_expr (a : ℝ) : 2 * a * (3 * a ^ 2 - 4 * a + 3) - 3 * a ^ 2 * (2 * a - 4) = 4 * a ^ 2 + 6 * a :=
by
  sorry

end simplify_expr_l2408_240897


namespace sector_area_l2408_240834

theorem sector_area (radius : ℝ) (central_angle : ℝ) (h1 : radius = 3) (h2 : central_angle = 2 * Real.pi / 3) : 
    (1 / 2) * radius^2 * central_angle = 6 * Real.pi :=
by
  rw [h1, h2]
  sorry

end sector_area_l2408_240834


namespace circumradius_of_right_triangle_l2408_240865

theorem circumradius_of_right_triangle (a b c : ℕ) (h : a = 8 ∧ b = 15 ∧ c = 17) : 
  ∃ R : ℝ, R = 8.5 :=
by
  sorry

end circumradius_of_right_triangle_l2408_240865


namespace area_of_rhombus_with_diagonals_6_and_8_l2408_240849

theorem area_of_rhombus_with_diagonals_6_and_8 : 
  ∀ (d1 d2 : ℕ), d1 = 6 → d2 = 8 → (1 / 2 : ℝ) * d1 * d2 = 24 :=
by
  intros d1 d2 h1 h2
  sorry

end area_of_rhombus_with_diagonals_6_and_8_l2408_240849


namespace sqrt_of_4_l2408_240810

theorem sqrt_of_4 :
  ∃ x : ℝ, x^2 = 4 ∧ (x = 2 ∨ x = -2) :=
sorry

end sqrt_of_4_l2408_240810


namespace T_five_three_l2408_240898

def T (a b : ℤ) : ℤ := 4 * a + 6 * b + 2

theorem T_five_three : T 5 3 = 40 := by
  sorry

end T_five_three_l2408_240898


namespace chess_match_probability_l2408_240896

theorem chess_match_probability (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  (3 * p^3 * (1 - p) ≤ 6 * p^3 * (1 - p)^2) → (p ≤ 1/2) :=
by
  sorry

end chess_match_probability_l2408_240896


namespace bucket_problem_l2408_240843

variable (A B C : ℝ)

theorem bucket_problem :
  (A - 6 = (1 / 3) * (B + 6)) →
  (B - 6 = (1 / 2) * (A + 6)) →
  (C - 8 = (1 / 2) * (A + 8)) →
  A = 13.2 :=
by
  sorry

end bucket_problem_l2408_240843


namespace maximum_sum_of_O_and_square_l2408_240872

theorem maximum_sum_of_O_and_square 
(O square : ℕ) (h1 : (O > 0) ∧ (square > 0)) 
(h2 : (O : ℚ) / 11 < (7 : ℚ) / (square))
(h3 : (7 : ℚ) / (square) < (4 : ℚ) / 5) : 
O + square = 18 :=
sorry

end maximum_sum_of_O_and_square_l2408_240872


namespace part1_and_part2_l2408_240815

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l2408_240815


namespace prime_divisor_condition_l2408_240804

theorem prime_divisor_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hdiv : q ∣ 2^p - 1) : p ∣ q - 1 :=
  sorry

end prime_divisor_condition_l2408_240804


namespace general_formula_l2408_240868

open Nat

def a (n : ℕ) : ℚ :=
  if n = 0 then 7/6 else 0 -- Recurrence initialization with dummy else condition

-- Defining the recurrence relation as a function
lemma recurrence_relation {n : ℕ} (h : n > 0) : 
    a n = (1 / 2) * a (n - 1) + (1 / 3) := 
sorry

-- Proof of the general formula
theorem general_formula (n : ℕ) : a n = (1 / (2^n : ℚ)) + (2 / 3) :=
sorry

end general_formula_l2408_240868


namespace dogs_food_consumption_l2408_240864

theorem dogs_food_consumption :
  (let cups_per_meal_momo_fifi := 1.5
   let meals_per_day := 3
   let cups_per_meal_gigi := 2
   let cups_to_pounds := 3
   let daily_food_momo_fifi := cups_per_meal_momo_fifi * meals_per_day * 2
   let daily_food_gigi := cups_per_meal_gigi * meals_per_day
   daily_food_momo_fifi + daily_food_gigi) / cups_to_pounds = 5 :=
by
  sorry

end dogs_food_consumption_l2408_240864


namespace benny_apples_l2408_240805

theorem benny_apples (benny dan : ℕ) (total : ℕ) (H1 : dan = 9) (H2 : total = 11) (H3 : benny + dan = total) : benny = 2 :=
by
  sorry

end benny_apples_l2408_240805


namespace prime_factors_of_expression_l2408_240812

theorem prime_factors_of_expression
  (p : ℕ) (prime_p : Nat.Prime p) :
  (∀ x y : ℕ, 0 < x → 0 < y → p ∣ ((x + y)^19 - x^19 - y^19)) ↔ (p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19) :=
by
  sorry

end prime_factors_of_expression_l2408_240812


namespace monotonically_increasing_range_l2408_240851

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end monotonically_increasing_range_l2408_240851


namespace algebraic_expr_pos_int_vals_l2408_240853

noncomputable def algebraic_expr_ineq (x : ℕ) : Prop :=
  x > 0 ∧ ((x + 1)/3 - (2*x - 1)/4 ≥ (x - 3)/6)

theorem algebraic_expr_pos_int_vals : {x : ℕ | algebraic_expr_ineq x} = {1, 2, 3} :=
sorry

end algebraic_expr_pos_int_vals_l2408_240853


namespace direct_proportion_function_l2408_240875

theorem direct_proportion_function (m : ℝ) 
  (h1 : m + 1 ≠ 0) 
  (h2 : m^2 - 1 = 0) : 
  m = 1 :=
sorry

end direct_proportion_function_l2408_240875


namespace jack_second_half_time_l2408_240828

variable (time_half1 time_half2 time_jack_total time_jill_total : ℕ)

theorem jack_second_half_time (h1 : time_half1 = 19) 
                              (h2 : time_jill_total = 32) 
                              (h3 : time_jack_total + 7 = time_jill_total) :
  time_jack_total = time_half1 + time_half2 → time_half2 = 6 := by
  sorry

end jack_second_half_time_l2408_240828


namespace tom_dollars_more_than_jerry_l2408_240830

theorem tom_dollars_more_than_jerry (total_slices : ℕ)
  (jerry_slices : ℕ)
  (tom_slices : ℕ)
  (plain_cost : ℕ)
  (pineapple_additional_cost : ℕ)
  (total_cost : ℕ)
  (cost_per_slice : ℚ)
  (cost_jerry : ℚ)
  (cost_tom : ℚ)
  (jerry_ate_plain : jerry_slices = 5)
  (tom_ate_pineapple : tom_slices = 5)
  (total_slices_10 : total_slices = 10)
  (plain_cost_10 : plain_cost = 10)
  (pineapple_additional_cost_3 : pineapple_additional_cost = 3)
  (total_cost_13 : total_cost = plain_cost + pineapple_additional_cost)
  (cost_per_slice_calc : cost_per_slice = total_cost / total_slices)
  (cost_jerry_calc : cost_jerry = cost_per_slice * jerry_slices)
  (cost_tom_calc : cost_tom = cost_per_slice * tom_slices) :
  cost_tom - cost_jerry = 0 := by
  sorry

end tom_dollars_more_than_jerry_l2408_240830


namespace value_of_x_l2408_240860

theorem value_of_x (x : ℝ) : abs (4 * x - 8) ≤ 0 ↔ x = 2 :=
by {
  sorry
}

end value_of_x_l2408_240860


namespace smallest_d_for_inequality_l2408_240842

open Real

theorem smallest_d_for_inequality :
  (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → exp (x * y) + 1 * |x^2 - y^2| ≥ exp ((x + y) / 2)) ∧
  (∀ d > 0, (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → exp (x * y) + d * |x^2 - y^2| ≥ exp ((x + y) / 2)) → d ≥ 1) :=
by
  sorry

end smallest_d_for_inequality_l2408_240842


namespace downstream_speed_l2408_240894

noncomputable def upstream_speed : ℝ := 5
noncomputable def still_water_speed : ℝ := 15

theorem downstream_speed:
  ∃ (Vd : ℝ), Vd = 25 ∧ (still_water_speed = (upstream_speed + Vd) / 2) := 
sorry

end downstream_speed_l2408_240894


namespace infinite_perfect_squares_in_ap_l2408_240819

open Nat

def is_arithmetic_progression (a d : ℕ) (an : ℕ → ℕ) : Prop :=
  ∀ n, an n = a + n * d

def is_perfect_square (x : ℕ) : Prop :=
  ∃ m, m * m = x

theorem infinite_perfect_squares_in_ap (a d : ℕ) (an : ℕ → ℕ) (m : ℕ)
  (h_arith_prog : is_arithmetic_progression a d an)
  (h_initial_square : a = m * m) :
  ∃ (f : ℕ → ℕ), ∀ n, is_perfect_square (an (f n)) :=
sorry

end infinite_perfect_squares_in_ap_l2408_240819


namespace largest_sphere_radius_in_prism_l2408_240813

noncomputable def largestInscribedSphereRadius (m : ℝ) : ℝ :=
  (Real.sqrt 6 - Real.sqrt 2) / 4 * m

theorem largest_sphere_radius_in_prism (m : ℝ) (h : 0 < m) :
  ∃ r, r = largestInscribedSphereRadius m ∧ r < m/2 :=
sorry

end largest_sphere_radius_in_prism_l2408_240813


namespace largest_plot_area_l2408_240822

def plotA_area : Real := 10
def plotB_area : Real := 10 + 1
def plotC_area : Real := 9 + 1.5
def plotD_area : Real := 12
def plotE_area : Real := 11 + 1

theorem largest_plot_area :
  max (max (max (max plotA_area plotB_area) plotC_area) plotD_area) plotE_area = 12 ∧ 
  (plotD_area = 12 ∧ plotE_area = 12) := by sorry

end largest_plot_area_l2408_240822


namespace cost_per_steak_knife_l2408_240869

theorem cost_per_steak_knife :
  ∀ (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℝ),
  sets = 2 →
  knives_per_set = 4 →
  cost_per_set = 80 →
  (cost_per_set * sets) / (sets * knives_per_set) = 20 :=
by
  intros sets knives_per_set cost_per_set sets_eq knives_per_set_eq cost_per_set_eq
  rw [sets_eq, knives_per_set_eq, cost_per_set_eq]
  sorry

end cost_per_steak_knife_l2408_240869


namespace stratified_sampling_correct_l2408_240817

def num_students := 500
def num_male_students := 500
def num_female_students := 400
def ratio_male_female := num_male_students / num_female_students

def selected_male_students := 25
def selected_female_students := (selected_male_students * num_female_students) / num_male_students

theorem stratified_sampling_correct :
  selected_female_students = 20 :=
by
  sorry

end stratified_sampling_correct_l2408_240817


namespace ethan_hours_per_day_l2408_240845

-- Define the known constants
def hourly_wage : ℝ := 18
def work_days_per_week : ℕ := 5
def total_earnings : ℝ := 3600
def weeks_worked : ℕ := 5

-- Define the main theorem
theorem ethan_hours_per_day :
  (∃ hours_per_day : ℝ, 
    hours_per_day = total_earnings / (weeks_worked * work_days_per_week * hourly_wage)) →
  hours_per_day = 8 :=
by
  sorry

end ethan_hours_per_day_l2408_240845


namespace katherine_age_l2408_240888

-- Define a Lean statement equivalent to the given problem
theorem katherine_age (K M : ℕ) (h1 : M = K - 3) (h2 : M = 21) : K = 24 := sorry

end katherine_age_l2408_240888


namespace fuel_consumption_l2408_240890

def initial_volume : ℕ := 3000
def volume_jan_1 : ℕ := 180
def volume_may_1 : ℕ := 1238
def refill_volume : ℕ := 3000

theorem fuel_consumption :
  (initial_volume - volume_jan_1) + (refill_volume - volume_may_1) = 4582 := by
  sorry

end fuel_consumption_l2408_240890


namespace probability_of_exactly_three_blue_marbles_l2408_240824

-- Define the conditions
def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def total_selections : ℕ := 6
def blue_selections : ℕ := 3
def blue_probability : ℚ := 8 / 15
def red_probability : ℚ := 7 / 15
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the binomial probability formula calculation
def binomial_probability : ℚ :=
  binomial_coefficient total_selections blue_selections * (blue_probability ^ blue_selections) * (red_probability ^ (total_selections - blue_selections))

-- The hypothesis (conditions) and conclusion (the solution)
theorem probability_of_exactly_three_blue_marbles :
  binomial_probability = (3512320 / 11390625) :=
by sorry

end probability_of_exactly_three_blue_marbles_l2408_240824


namespace work_done_together_in_six_days_l2408_240829

theorem work_done_together_in_six_days (A B : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 18) :
  1 / (A + B) = 6 :=
by
  sorry

end work_done_together_in_six_days_l2408_240829


namespace max_divisions_circle_and_lines_l2408_240825

theorem max_divisions_circle_and_lines (n : ℕ) (h₁ : n = 5) : 
  let R_lines := n * (n + 1) / 2 + 1 -- Maximum regions formed by n lines
  let R_circle_lines := 2 * n       -- Additional regions formed by a circle intersecting n lines
  R_lines + R_circle_lines = 26 := by
  sorry

end max_divisions_circle_and_lines_l2408_240825


namespace find_x_l2408_240803

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l2408_240803


namespace total_distance_l2408_240802

theorem total_distance (x y : ℝ) (h1 : x * y = 18) :
  let D2 := (y - 1) * (x + 1)
  let D3 := 15
  let D_total := 18 + D2 + D3
  D_total = y * x + y - x + 32 :=
by
  let D2 := (y - 1) * (x + 1)
  let D3 := 15
  let D_total := 18 + D2 + D3
  sorry

end total_distance_l2408_240802


namespace bicycle_cost_price_l2408_240867

variable (CP_A SP_B SP_C : ℝ)

theorem bicycle_cost_price 
  (h1 : SP_B = CP_A * 1.20) 
  (h2 : SP_C = SP_B * 1.25) 
  (h3 : SP_C = 225) :
  CP_A = 150 := 
by
  sorry

end bicycle_cost_price_l2408_240867


namespace smallest_trees_in_three_types_l2408_240807

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l2408_240807


namespace value_of_a_plus_d_l2408_240883

theorem value_of_a_plus_d 
  (a b c d : ℤ)
  (h1 : a + b = 12) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) 
  : a + d = 9 := 
  sorry

end value_of_a_plus_d_l2408_240883


namespace find_angle_C_find_side_a_l2408_240801

namespace TriangleProof

-- Declare the conditions and the proof promises
variables {A B C : ℝ} {a b c S : ℝ}

-- First part: Prove angle C
theorem find_angle_C (h1 : c^2 = a^2 + b^2 - a * b) : C = 60 :=
sorry

-- Second part: Prove the value of a
theorem find_side_a (h2 : b = 2) (h3 : S = (3 * Real.sqrt 3) / 2) : a = 3 :=
sorry

end TriangleProof

end find_angle_C_find_side_a_l2408_240801


namespace coordinates_of_point_l2408_240816

noncomputable def point_on_x_axis (x : ℝ) :=
  (x, 0)

theorem coordinates_of_point (x : ℝ) (hx : abs x = 3) :
  point_on_x_axis x = (3, 0) ∨ point_on_x_axis x = (-3, 0) :=
  sorry

end coordinates_of_point_l2408_240816


namespace infinitely_many_divisible_by_100_l2408_240886

open Nat

theorem infinitely_many_divisible_by_100 : ∀ p : ℕ, ∃ n : ℕ, n = 100 * p + 6 ∧ 100 ∣ (2^n + n^2) := by
  sorry

end infinitely_many_divisible_by_100_l2408_240886


namespace shaina_chocolate_l2408_240857

-- Define the conditions
def total_chocolate : ℚ := 48 / 5
def number_of_piles : ℚ := 4

-- Define the assertion to prove
theorem shaina_chocolate : (total_chocolate / number_of_piles) = (12 / 5) := 
by 
  sorry

end shaina_chocolate_l2408_240857


namespace problem_part1_problem_part2_l2408_240848

noncomputable def f (x : ℝ) (m : ℝ) := Real.sqrt (|x + 2| + |x - 4| - m)

theorem problem_part1 (m : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0) ↔ m ≤ 6 := 
by
  sorry

theorem problem_part2 (a b : ℕ) (n : ℝ) (h1 : (0 < a) ∧ (0 < b)) (h2 : n = 6) 
  (h3 : (4 / (a + 5 * b)) + (1 / (3 * a + 2 * b)) = n) : 
  ∃ (value : ℝ), 4 * a + 7 * b = 3 / 2 := 
by
  sorry

end problem_part1_problem_part2_l2408_240848


namespace proof_problem_l2408_240870

theorem proof_problem
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2010)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2009)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2010)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2009)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2010)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2009) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -1 :=
by
  sorry

end proof_problem_l2408_240870


namespace right_triangle_other_acute_angle_l2408_240818

theorem right_triangle_other_acute_angle (A B C : ℝ) (r : A + B + C = 180) (h : A = 90) (a : B = 30) :
  C = 60 :=
sorry

end right_triangle_other_acute_angle_l2408_240818


namespace part1_part2_l2408_240840

-- Conditions: Definitions of A and B
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 - 6 * a * b - 8 * a

-- Theorem statements
theorem part1 (a b : ℝ) :  2 * A a b - B a b = -4 * a * b + 6 * b + 8 * a := sorry

theorem part2 (a : ℝ) (h : ∀ a, 2 * A a 2 - B a 2 = - 4 * a * 2 + 6 * 2 + 8 * a) : 2 = 2 := sorry

end part1_part2_l2408_240840


namespace factorize_expression_l2408_240879

theorem factorize_expression (a x : ℝ) : a * x^3 - 16 * a * x = a * x * (x + 4) * (x - 4) := by
  sorry

end factorize_expression_l2408_240879


namespace quadratic_inequality_solution_l2408_240863

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 36 * x + 318 ≤ 0 ↔ 18 - Real.sqrt 6 ≤ x ∧ x ≤ 18 + Real.sqrt 6 := by
  sorry

end quadratic_inequality_solution_l2408_240863


namespace marked_price_l2408_240878

theorem marked_price (P : ℝ)
  (h₁ : 20 / 100 = 0.20)
  (h₂ : 15 / 100 = 0.15)
  (h₃ : 5 / 100 = 0.05)
  (h₄ : 7752 = 0.80 * 0.85 * 0.95 * P)
  : P = 11998.76 := by
  sorry

end marked_price_l2408_240878


namespace insects_in_lab_l2408_240847

theorem insects_in_lab (total_legs number_of_legs_per_insect : ℕ) (h1 : total_legs = 36) (h2 : number_of_legs_per_insect = 6) : (total_legs / number_of_legs_per_insect) = 6 :=
by
  sorry

end insects_in_lab_l2408_240847


namespace negation_proposition_l2408_240821

theorem negation_proposition (l : ℝ) (h : l = 1) : 
  (¬ ∃ x : ℝ, x + l ≥ 0) = (∀ x : ℝ, x + l < 0) := by 
  sorry

end negation_proposition_l2408_240821


namespace exists_100_integers_with_distinct_pairwise_sums_l2408_240826

-- Define number of integers and the constraint limit
def num_integers : ℕ := 100
def max_value : ℕ := 25000

-- Define the predicate for all pairwise sums being different
def pairwise_different_sums (as : Fin num_integers → ℕ) : Prop :=
  ∀ i j k l : Fin num_integers, i ≠ j ∧ k ≠ l → as i + as j ≠ as k + as l

-- Main theorem statement
theorem exists_100_integers_with_distinct_pairwise_sums :
  ∃ as : Fin num_integers → ℕ, (∀ i : Fin num_integers, as i > 0 ∧ as i ≤ max_value) ∧ pairwise_different_sums as :=
sorry

end exists_100_integers_with_distinct_pairwise_sums_l2408_240826


namespace pencil_length_l2408_240820

theorem pencil_length (L : ℝ) (h1 : (1 / 8) * L + (1 / 2) * (7 / 8) * L + (7 / 2) = L) : L = 16 :=
by
  sorry

end pencil_length_l2408_240820


namespace div_condition_positive_integers_l2408_240899

theorem div_condition_positive_integers 
  (a b d : ℕ) 
  (h1 : a + b ≡ 0 [MOD d]) 
  (h2 : a * b ≡ 0 [MOD d^2]) 
  (h3 : 0 < a) 
  (h4 : 0 < b) 
  (h5 : 0 < d) : 
  d ∣ a ∧ d ∣ b :=
sorry

end div_condition_positive_integers_l2408_240899


namespace simplify_expression_l2408_240877

variable {a b c : ℝ}

-- Assuming the conditions specified in the problem
def valid_conditions (a b c : ℝ) : Prop := (1 - a * b ≠ 0) ∧ (1 + c * a ≠ 0)

theorem simplify_expression (h : valid_conditions a b c) :
  (a + b) / (1 - a * b) + (c - a) / (1 + c * a) / 
  (1 - ((a + b) / (1 - a * b) * (c - a) / (1 + c * a))) = 
  (b + c) / (1 - b * c) := 
sorry

end simplify_expression_l2408_240877


namespace find_a_from_function_property_l2408_240854

theorem find_a_from_function_property {a : ℝ} (h : ∀ (x : ℝ), (0 ≤ x → x ≤ 1 → ax ≤ 3) ∧ (0 ≤ x → x ≤ 1 → ax ≥ 3)) :
  a = 3 :=
sorry

end find_a_from_function_property_l2408_240854


namespace unit_vector_norm_diff_l2408_240881

noncomputable def sqrt42_sqrt3_div_2 : ℝ := (Real.sqrt 42 * Real.sqrt 3) / 2
noncomputable def sqrt17_div_sqrt2 : ℝ := (Real.sqrt 17) / Real.sqrt 2

theorem unit_vector_norm_diff {x1 y1 z1 x2 y2 z2 : ℝ}
  (h1 : x1^2 + y1^2 + z1^2 = 1)
  (h2 : 3*x1 + y1 + 2*z1 = sqrt42_sqrt3_div_2)
  (h3 : 2*x1 + 2*y1 + 3*z1 = sqrt17_div_sqrt2)
  (h4 : x2^2 + y2^2 + z2^2 = 1)
  (h5 : 3*x2 + y2 + 2*z2 = sqrt42_sqrt3_div_2)
  (h6 : 2*x2 + 2*y2 + 3*z2 = sqrt17_div_sqrt2)
  (h_distinct : (x1, y1, z1) ≠ (x2, y2, z2)) :
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2) = Real.sqrt 2 :=
by
  sorry

end unit_vector_norm_diff_l2408_240881


namespace find_a_l2408_240835

theorem find_a (a : ℝ) (h : (a - 1) ≠ 0) :
  (∃ x : ℝ, ((a - 1) * x^2 + x + a^2 - 1 = 0) ∧ x = 0) → a = -1 :=
by
  sorry

end find_a_l2408_240835


namespace randy_initial_money_l2408_240882

theorem randy_initial_money (X : ℕ) (h : X + 200 - 1200 = 2000) : X = 3000 :=
by {
  sorry
}

end randy_initial_money_l2408_240882


namespace fill_tank_in_18_minutes_l2408_240895

-- Define the conditions
def rate_pipe_A := 1 / 9  -- tanks per minute
def rate_pipe_B := - (1 / 18) -- tanks per minute (negative because it's emptying)

-- Define the net rate of both pipes working together
def net_rate := rate_pipe_A + rate_pipe_B

-- Define the time to fill the tank when both pipes are working
def time_to_fill_tank := 1 / net_rate

theorem fill_tank_in_18_minutes : time_to_fill_tank = 18 := 
    by
    -- Sorry to skip the actual proof
    sorry

end fill_tank_in_18_minutes_l2408_240895


namespace allison_upload_ratio_l2408_240831

theorem allison_upload_ratio :
  ∃ (x y : ℕ), (x + y = 30) ∧ (10 * x + 20 * y = 450) ∧ (x / 30 = 1 / 2) :=
by
  sorry

end allison_upload_ratio_l2408_240831


namespace find_a1_l2408_240884

-- Given an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Arithmetic sequence is monotonically increasing
def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

-- First condition: sum of first three terms
def sum_first_three_terms (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 12

-- Second condition: product of first three terms
def product_first_three_terms (a : ℕ → ℝ) : Prop :=
  a 0 * a 1 * a 2 = 48

-- Proving that a_1 = 2 given the conditions
theorem find_a1 (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : is_monotonically_increasing a)
  (h3 : sum_first_three_terms a) (h4 : product_first_three_terms a) : a 0 = 2 :=
by
  -- Proof will be filled in here
  sorry

end find_a1_l2408_240884


namespace solve_for_s_l2408_240858

theorem solve_for_s : ∃ (s t : ℚ), (8 * s + 7 * t = 160) ∧ (s = t - 3) ∧ (s = 139 / 15) := by
  sorry

end solve_for_s_l2408_240858


namespace smallest_angle_in_triangle_l2408_240846

open Real

theorem smallest_angle_in_triangle
  (a b c : ℝ)
  (h : a = (b + c) / 3)
  (triangle_inequality_1 : a + b > c)
  (triangle_inequality_2 : a + c > b)
  (triangle_inequality_3 : b + c > a) :
  ∃ A B C α β γ : ℝ, -- A, B, C are the angles opposite to sides a, b, c respectively
  0 < α ∧ α < β ∧ α < γ :=
sorry

end smallest_angle_in_triangle_l2408_240846


namespace determine_a_l2408_240808

theorem determine_a (a : ℚ) (x : ℚ) : 
  (∃ r s : ℚ, (r*x + s)^2 = a*x^2 + 18*x + 16) → 
  a = 81/16 := 
sorry

end determine_a_l2408_240808


namespace spending_together_l2408_240876

def sandwich_cost := 2
def hamburger_cost := 2
def hotdog_cost := 1
def juice_cost := 2
def selene_sandwiches := 3
def selene_juices := 1
def tanya_hamburgers := 2
def tanya_juices := 2

def selene_spending : ℕ := (selene_sandwiches * sandwich_cost) + (selene_juices * juice_cost)
def tanya_spending : ℕ := (tanya_hamburgers * hamburger_cost) + (tanya_juices * juice_cost)
def total_spending : ℕ := selene_spending + tanya_spending

theorem spending_together : total_spending = 16 :=
by
  sorry

end spending_together_l2408_240876
