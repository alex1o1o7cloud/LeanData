import Mathlib

namespace sales_fraction_l1927_192781

theorem sales_fraction (A D : ℝ) (h : D = 2 * A) : D / (11 * A + D) = 2 / 13 :=
by
  sorry

end sales_fraction_l1927_192781


namespace nonagon_line_segments_not_adjacent_l1927_192731

def nonagon_segments (n : ℕ) : ℕ :=
(n * (n - 3)) / 2

theorem nonagon_line_segments_not_adjacent (h : ∃ n, n = 9) :
  nonagon_segments 9 = 27 :=
by
  -- proof omitted
  sorry

end nonagon_line_segments_not_adjacent_l1927_192731


namespace min_value_inequality_l1927_192789

theorem min_value_inequality (p q r : ℝ) (h₀ : 0 < p) (h₁ : 0 < q) (h₂ : 0 < r) :
  ( 3 * r / (p + 2 * q) + 3 * p / (2 * r + q) + 2 * q / (p + r) ) ≥ (29 / 6) := 
sorry

end min_value_inequality_l1927_192789


namespace rectangleY_has_tileD_l1927_192787

-- Define the structure for a tile
structure Tile where
  top : Nat
  right : Nat
  bottom : Nat
  left : Nat

-- Define tiles
def TileA : Tile := { top := 6, right := 3, bottom := 5, left := 2 }
def TileB : Tile := { top := 3, right := 6, bottom := 2, left := 5 }
def TileC : Tile := { top := 5, right := 7, bottom := 1, left := 2 }
def TileD : Tile := { top := 2, right := 5, bottom := 6, left := 3 }

-- Define rectangles (positioning)
inductive Rectangle
| W | X | Y | Z

-- Define which tile is in Rectangle Y
def tileInRectangleY : Tile → Prop :=
  fun t => t = TileD

-- Statement to prove
theorem rectangleY_has_tileD : tileInRectangleY TileD :=
by
  -- The final statement to be proven, skipping the proof itself with sorry
  sorry

end rectangleY_has_tileD_l1927_192787


namespace angle_between_strips_l1927_192796

theorem angle_between_strips (w : ℝ) (a : ℝ) (angle : ℝ) (h_w : w = 1) (h_area : a = 2) :
  ∃ θ : ℝ, θ = 30 ∧ angle = θ :=
by
  sorry

end angle_between_strips_l1927_192796


namespace find_multiple_of_q_l1927_192723

-- Definitions of x and y
def x (k q : ℤ) : ℤ := 55 + k * q
def y (q : ℤ) : ℤ := 4 * q + 41

-- The proof statement
theorem find_multiple_of_q (k : ℤ) : x k 7 = y 7 → k = 2 := by
  sorry

end find_multiple_of_q_l1927_192723


namespace groups_of_four_on_plane_l1927_192740

-- Define the points in the tetrahedron
inductive Point
| vertex : Point
| midpoint : Point

noncomputable def points : List Point :=
  [Point.vertex, Point.midpoint, Point.midpoint, Point.midpoint, Point.midpoint,
   Point.vertex, Point.midpoint, Point.midpoint, Point.midpoint, Point.vertex]

-- Condition: all 10 points are either vertices or midpoints of the edges of a tetrahedron 
def points_condition : ∀ p ∈ points, p = Point.vertex ∨ p = Point.midpoint := sorry

-- Function to count unique groups of four points lying on the same plane
noncomputable def count_groups : ℕ :=
  33  -- Given as the correct answer in the problem

-- Proof problem stating the count of groups
theorem groups_of_four_on_plane : count_groups = 33 :=
by 
  sorry -- Proof omitted

end groups_of_four_on_plane_l1927_192740


namespace angle_B_max_area_triangle_l1927_192721
noncomputable section

open Real

variables {A B C a b c : ℝ}

-- Prove B = π / 3 given b sin A = √3 a cos B
theorem angle_B (h1 : b * sin A = sqrt 3 * a * cos B) : B = π / 3 :=
sorry

-- Prove if b = 2√3, the maximum area of triangle ABC is 3√3
theorem max_area_triangle (h1 : b * sin A = sqrt 3 * a * cos B) (h2 : b = 2 * sqrt 3) : 
    (1 / 2) * a * (a : ℝ) *  (sqrt 3 / 2 : ℝ) ≤ 3 * sqrt 3 :=
sorry

end angle_B_max_area_triangle_l1927_192721


namespace eval_f_l1927_192725

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem eval_f : f (f (1 / 2)) = 1 :=
by
  sorry

end eval_f_l1927_192725


namespace triangle_area_proof_l1927_192718

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  0.5 * a * c * Real.sin B

theorem triangle_area_proof (a b c : ℝ) (B : ℝ) (hB : B = 2 * Real.pi / 3) (hb : b = Real.sqrt 13) (h_sum : a + c = 4) :
  triangle_area a b c B = 3 * Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_proof_l1927_192718


namespace intersection_is_singleton_l1927_192702

namespace ProofProblem

def M : Set ℤ := {-3, -2, -1}

def N : Set ℤ := {x : ℤ | (x + 2) * (x - 3) < 0}

theorem intersection_is_singleton : M ∩ N = {-1} :=
by
  sorry

end ProofProblem

end intersection_is_singleton_l1927_192702


namespace fractional_equation_solution_l1927_192749

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) →
  (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end fractional_equation_solution_l1927_192749


namespace marks_in_physics_l1927_192755

def marks_in_english : ℝ := 74
def marks_in_mathematics : ℝ := 65
def marks_in_chemistry : ℝ := 67
def marks_in_biology : ℝ := 90
def average_marks : ℝ := 75.6
def number_of_subjects : ℕ := 5

-- We need to show that David's marks in Physics are 82.
theorem marks_in_physics : ∃ (P : ℝ), P = 82 ∧ 
  ((marks_in_english + marks_in_mathematics + P + marks_in_chemistry + marks_in_biology) / number_of_subjects = average_marks) :=
by sorry

end marks_in_physics_l1927_192755


namespace ratio_first_to_second_l1927_192782

theorem ratio_first_to_second (S F T : ℕ) 
  (hS : S = 60)
  (hT : T = F / 3)
  (hSum : F + S + T = 220) :
  F / S = 2 :=
by
  sorry

end ratio_first_to_second_l1927_192782


namespace find_u_plus_v_l1927_192730

variables (u v : ℚ)

theorem find_u_plus_v (h1 : 5 * u - 6 * v = 19) (h2 : 3 * u + 5 * v = -1) : u + v = 27 / 43 := by
  sorry

end find_u_plus_v_l1927_192730


namespace max_sum_of_arithmetic_sequence_l1927_192766

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ)
  (d : ℤ) (h_a : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 2 + a 3 = 156)
  (h2 : a 2 + a 3 + a 4 = 147) :
  ∃ n : ℕ, n = 19 ∧ (∀ m : ℕ, S m ≤ S n) :=
sorry

end max_sum_of_arithmetic_sequence_l1927_192766


namespace meat_per_deer_is_200_l1927_192799

namespace wolf_pack

def number_hunting_wolves : ℕ := 4
def number_additional_wolves : ℕ := 16
def meat_needed_per_day : ℕ := 8
def days : ℕ := 5

def total_wolves : ℕ := number_hunting_wolves + number_additional_wolves

def total_meat_needed : ℕ := total_wolves * meat_needed_per_day * days

def number_deer : ℕ := number_hunting_wolves

def meat_per_deer : ℕ := total_meat_needed / number_deer

theorem meat_per_deer_is_200 : meat_per_deer = 200 := by
  sorry

end wolf_pack

end meat_per_deer_is_200_l1927_192799


namespace non_periodic_decimal_l1927_192710

variable {a : ℕ → ℕ}

-- Condition definitions
def is_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

def constraint (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ 10 * a n

-- Theorem statement
theorem non_periodic_decimal (a : ℕ → ℕ) 
  (h_inc : is_increasing_sequence a) 
  (h_constraint : constraint a) : 
  ¬ (∃ T : ℕ, ∀ n : ℕ, a (n + T) = a n) :=
sorry

end non_periodic_decimal_l1927_192710


namespace percentage_reduction_l1927_192785

theorem percentage_reduction (original reduced : ℕ) (h₁ : original = 260) (h₂ : reduced = 195) :
  (original - reduced) / original * 100 = 25 := by
  sorry

end percentage_reduction_l1927_192785


namespace final_value_of_S_l1927_192744

theorem final_value_of_S :
  ∀ (S n : ℕ), S = 1 → n = 1 →
  (∀ S n : ℕ, ¬ n > 3 → 
    (∃ S' n' : ℕ, S' = S + 2 * n ∧ n' = n + 1 ∧ 
      (∀ S n : ℕ, n > 3 → S' = 13))) :=
by 
  intros S n hS hn
  simp [hS, hn]
  sorry

end final_value_of_S_l1927_192744


namespace correct_number_of_eggs_to_buy_l1927_192707

/-- Define the total number of eggs needed and the number of eggs given by Andrew -/
def total_eggs_needed : ℕ := 222
def eggs_given_by_andrew : ℕ := 155

/-- Define a statement asserting the correct number of eggs to buy -/
def remaining_eggs_to_buy : ℕ := total_eggs_needed - eggs_given_by_andrew

/-- The statement of the proof problem -/
theorem correct_number_of_eggs_to_buy : remaining_eggs_to_buy = 67 :=
by sorry

end correct_number_of_eggs_to_buy_l1927_192707


namespace veronica_reroll_probability_is_correct_l1927_192706

noncomputable def veronica_reroll_probability : ℚ :=
  let P := (5 : ℚ) / 54
  P

theorem veronica_reroll_probability_is_correct :
  veronica_reroll_probability = (5 : ℚ) / 54 := sorry

end veronica_reroll_probability_is_correct_l1927_192706


namespace evaluate_expression_l1927_192747

theorem evaluate_expression :
  2 - (-3) - 4 + (-5) - 6 + 7 = -3 :=
by
  sorry

end evaluate_expression_l1927_192747


namespace find_n_l1927_192779

theorem find_n (n : ℕ) (h_lcm : Nat.lcm n 16 = 48) (h_gcf : Nat.gcd n 16 = 4) : n = 12 :=
by
  sorry

end find_n_l1927_192779


namespace computer_game_cost_l1927_192773

variable (ticket_cost : ℕ := 12)
variable (num_tickets : ℕ := 3)
variable (total_spent : ℕ := 102)

theorem computer_game_cost (C : ℕ) (h : C + num_tickets * ticket_cost = total_spent) : C = 66 :=
by
  -- Proof would go here
  sorry

end computer_game_cost_l1927_192773


namespace value_of_f2009_l1927_192737

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f2009 
  (h_ineq1 : ∀ x : ℝ, f x ≤ f (x+4) + 4)
  (h_ineq2 : ∀ x : ℝ, f (x+2) ≥ f x + 2)
  (h_f1 : f 1 = 0) :
  f 2009 = 2008 :=
sorry

end value_of_f2009_l1927_192737


namespace first_discount_percentage_l1927_192798

theorem first_discount_percentage (D : ℝ) :
  (345 * (1 - D / 100) * 0.75 = 227.70) → (D = 12) :=
by
  intro cond
  sorry

end first_discount_percentage_l1927_192798


namespace sandy_age_l1927_192735

theorem sandy_age (S M : ℕ) (h1 : M = S + 18) (h2 : S * 9 = M * 7) : S = 63 := by
  sorry

end sandy_age_l1927_192735


namespace days_worked_per_week_l1927_192752

theorem days_worked_per_week (total_toys_per_week toys_produced_each_day : ℕ) 
  (h1 : total_toys_per_week = 5505)
  (h2 : toys_produced_each_day = 1101)
  : total_toys_per_week / toys_produced_each_day = 5 :=
  by
    sorry

end days_worked_per_week_l1927_192752


namespace abigail_money_loss_l1927_192705

theorem abigail_money_loss {initial spent remaining lost : ℤ} 
  (h1 : initial = 11) 
  (h2 : spent = 2) 
  (h3 : remaining = 3) 
  (h4 : lost = initial - spent - remaining) : 
  lost = 6 := sorry

end abigail_money_loss_l1927_192705


namespace polynomial_divisibility_l1927_192743

theorem polynomial_divisibility (m : ℤ) : (3 * (-2)^2 + 5 * (-2) + m = 0) ↔ (m = -2) :=
by
  sorry

end polynomial_divisibility_l1927_192743


namespace num_of_adults_l1927_192736

def students : ℕ := 22
def vans : ℕ := 3
def capacity_per_van : ℕ := 8

theorem num_of_adults : (vans * capacity_per_van) - students = 2 := by
  sorry

end num_of_adults_l1927_192736


namespace triangle_angle_range_l1927_192711

theorem triangle_angle_range (α β γ : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α = 2 * γ)
  (h3 : α ≥ β)
  (h4 : β ≥ γ) :
  45 ≤ β ∧ β ≤ 72 := 
sorry

end triangle_angle_range_l1927_192711


namespace heat_released_is_1824_l1927_192774

def ΔH_f_NH3 : ℝ := -46  -- Enthalpy of formation of NH3 in kJ/mol
def ΔH_f_H2SO4 : ℝ := -814  -- Enthalpy of formation of H2SO4 in kJ/mol
def ΔH_f_NH4SO4 : ℝ := -909  -- Enthalpy of formation of (NH4)2SO4 in kJ/mol

def ΔH_rxn : ℝ :=
  2 * ΔH_f_NH4SO4 - (2 * ΔH_f_NH3 + ΔH_f_H2SO4)  -- Reaction enthalpy change

def heat_released : ℝ := 2 * ΔH_rxn  -- Heat released for 4 moles of NH3

theorem heat_released_is_1824 : heat_released = -1824 :=
by
  -- Theorem statement for proving heat released is 1824 kJ
  sorry

end heat_released_is_1824_l1927_192774


namespace find_c_l1927_192741

noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem find_c (a b c S : ℝ) (C : ℝ) 
  (ha : a = 3) 
  (hC : C = 120) 
  (hS : S = 15 * Real.sqrt 3 / 4) 
  (hab : a * b = 15)
  (hc2 : c^2 = a^2 + b^2 - 2 * a * b * cos_deg C) :
  c = 7 :=
by 
  sorry

end find_c_l1927_192741


namespace roger_ant_l1927_192728

def expected_steps : ℚ := 11/3

theorem roger_ant (a b : ℕ) (h1 : expected_steps = a / b) (h2 : Nat.gcd a b = 1) : 100 * a + b = 1103 :=
sorry

end roger_ant_l1927_192728


namespace car_speed_is_100_l1927_192784

def avg_speed (d1 d2 t: ℕ) := (d1 + d2) / t = 80

theorem car_speed_is_100 
  (x : ℕ)
  (speed_second_hour : ℕ := 60)
  (total_time : ℕ := 2)
  (h : avg_speed x speed_second_hour total_time):
  x = 100 :=
by
  unfold avg_speed at h
  sorry

end car_speed_is_100_l1927_192784


namespace all_weights_equal_l1927_192729

theorem all_weights_equal (w : Fin 13 → ℤ) 
  (h : ∀ (i : Fin 13), ∃ (a b : Multiset (Fin 12)),
    a + b = (Finset.univ.erase i).val ∧ Multiset.card a = 6 ∧ 
    Multiset.card b = 6 ∧ Multiset.sum (a.map w) = Multiset.sum (b.map w)) :
  ∀ i j, w i = w j :=
by sorry

end all_weights_equal_l1927_192729


namespace total_cost_eq_57_l1927_192700

namespace CandyCost

-- Conditions
def cost_of_caramel : ℕ := 3
def cost_of_candy_bar : ℕ := 2 * cost_of_caramel
def cost_of_cotton_candy : ℕ := (4 * cost_of_candy_bar) / 2

-- Define the total cost calculation
def total_cost : ℕ :=
  (6 * cost_of_candy_bar) + (3 * cost_of_caramel) + cost_of_cotton_candy

-- Theorem we want to prove
theorem total_cost_eq_57 : total_cost = 57 :=
by
  sorry  -- Proof to be provided

end CandyCost

end total_cost_eq_57_l1927_192700


namespace volume_of_inscribed_cube_l1927_192797

theorem volume_of_inscribed_cube (S : ℝ) (π : ℝ) (V : ℝ) (r : ℝ) (s : ℝ) :
    S = 12 * π → 4 * π * r^2 = 12 * π → s = 2 * r → V = s^3 → V = 8 :=
by
  sorry

end volume_of_inscribed_cube_l1927_192797


namespace max_a_value_l1927_192763

theorem max_a_value : ∃ a b : ℕ, 1 < a ∧ a < b ∧
  (∀ x y : ℝ, y = -2 * x + 4033 ∧ y = |x - 1| + |x + a| + |x - b| → 
  a = 4031) := sorry

end max_a_value_l1927_192763


namespace camel_cost_is_5200_l1927_192778

-- Definitions of costs in terms of Rs.
variable (C H O E : ℕ)

-- Conditions
axiom cond1 : 10 * C = 24 * H
axiom cond2 : ∃ X : ℕ, X * H = 4 * O
axiom cond3 : 6 * O = 4 * E
axiom cond4 : 10 * E = 130000

-- Theorem to prove
theorem camel_cost_is_5200 (hC : C = 5200) : C = 5200 :=
by sorry

end camel_cost_is_5200_l1927_192778


namespace books_per_shelf_l1927_192770

theorem books_per_shelf (mystery_shelves picture_shelves total_books : ℕ) 
    (h1 : mystery_shelves = 5) (h2 : picture_shelves = 4) (h3 : total_books = 54) : 
    total_books / (mystery_shelves + picture_shelves) = 6 := 
by
  -- necessary preliminary steps and full proof will go here
  sorry

end books_per_shelf_l1927_192770


namespace total_weight_of_4_moles_of_ba_cl2_l1927_192776

-- Conditions
def atomic_weight_ba : ℝ := 137.33
def atomic_weight_cl : ℝ := 35.45
def moles_ba_cl2 : ℝ := 4

-- Molecular weight of BaCl2
def molecular_weight_ba_cl2 : ℝ := 
  atomic_weight_ba + 2 * atomic_weight_cl

-- Total weight of 4 moles of BaCl2
def total_weight : ℝ := 
  molecular_weight_ba_cl2 * moles_ba_cl2

-- Theorem stating the total weight of 4 moles of BaCl2
theorem total_weight_of_4_moles_of_ba_cl2 :
  total_weight = 832.92 :=
sorry

end total_weight_of_4_moles_of_ba_cl2_l1927_192776


namespace solve_equation_l1927_192703

theorem solve_equation (a : ℝ) : 
  {x : ℝ | x * (x + a)^3 * (5 - x) = 0} = {0, -a, 5} :=
sorry

end solve_equation_l1927_192703


namespace number_of_ears_pierced_l1927_192768

-- Definitions for the conditions
def nosePiercingPrice : ℝ := 20
def earPiercingPrice := nosePiercingPrice + 0.5 * nosePiercingPrice
def totalAmountMade : ℝ := 390
def nosesPierced : ℕ := 6
def totalFromNoses := nosesPierced * nosePiercingPrice
def totalFromEars := totalAmountMade - totalFromNoses

-- The proof statement
theorem number_of_ears_pierced : totalFromEars / earPiercingPrice = 9 := by
  sorry

end number_of_ears_pierced_l1927_192768


namespace intersection_M_N_l1927_192788

def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := U \ complement_U_N

theorem intersection_M_N : M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
by
  sorry

end intersection_M_N_l1927_192788


namespace companion_value_4164_smallest_N_satisfies_conditions_l1927_192733

-- Define relevant functions
def G (N : ℕ) : ℕ :=
  let digits := [N / 1000 % 10, N / 100 % 10, N / 10 % 10, N % 10]
  digits.sum

def P (N : ℕ) : ℕ :=
  (N / 1000 % 10) * (N / 100 % 10)

def Q (N : ℕ) : ℕ :=
  (N / 10 % 10) * (N % 10)

def companion_value (N : ℕ) : ℚ :=
  |(G N : ℤ) / ((P N : ℤ) - (Q N : ℤ))|

-- Proof problem for part (1)
theorem companion_value_4164 : companion_value 4164 = 3 / 4 := sorry

-- Proof problem for part (2)
theorem smallest_N_satisfies_conditions :
  ∀ (N : ℕ), N > 1000 ∧ N < 10000 ∧ (∀ d, N / 10^d % 10 ≠ 0) ∧ (N / 1000 % 10 + N % 10) % 9 = 0 ∧ G N = 16 ∧ companion_value N = 4 → N = 2527 := sorry

end companion_value_4164_smallest_N_satisfies_conditions_l1927_192733


namespace probability_not_sit_at_ends_l1927_192726

theorem probability_not_sit_at_ends (h1: ∀ M J: ℕ, M ≠ J → M ≠ 1 ∧ M ≠ 8 ∧ J ≠ 1 ∧ J ≠ 8) : 
  (∃ p: ℚ, p = (3 / 7)) :=
by 
  sorry

end probability_not_sit_at_ends_l1927_192726


namespace xiao_ming_correct_answers_l1927_192794

theorem xiao_ming_correct_answers :
  let prob1 := (-2 - 2) = 0
  let prob2 := (-2 - (-2)) = -4
  let prob3 := (-3 + 5 - 6) = -4
  (if prob1 then 1 else 0) + (if prob2 then 1 else 0) + (if prob3 then 1 else 0) = 1 :=
by
  sorry

end xiao_ming_correct_answers_l1927_192794


namespace graph_passes_through_fixed_point_l1927_192793

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x-1)

theorem graph_passes_through_fixed_point (a : ℝ) : f a 1 = 5 :=
by
  -- sorry is a placeholder for the proof
  sorry

end graph_passes_through_fixed_point_l1927_192793


namespace budget_percentage_l1927_192722

-- Define the given conditions
def basic_salary_per_hour : ℝ := 7.50
def commission_rate : ℝ := 0.16
def hours_worked : ℝ := 160
def total_sales : ℝ := 25000
def amount_for_insurance : ℝ := 260

-- Define the basic salary, commission, and total earnings
def basic_salary : ℝ := basic_salary_per_hour * hours_worked
def commission : ℝ := commission_rate * total_sales
def total_earnings : ℝ := basic_salary + commission
def amount_for_budget : ℝ := total_earnings - amount_for_insurance

-- Define the proof problem
theorem budget_percentage : (amount_for_budget / total_earnings) * 100 = 95 := by
  simp [basic_salary, commission, total_earnings, amount_for_budget]
  sorry

end budget_percentage_l1927_192722


namespace boys_who_did_not_bring_laptops_l1927_192771

-- Definitions based on the conditions.
def total_boys : ℕ := 20
def students_who_brought_laptops : ℕ := 25
def girls_who_brought_laptops : ℕ := 16

-- Main theorem statement.
theorem boys_who_did_not_bring_laptops : total_boys - (students_who_brought_laptops - girls_who_brought_laptops) = 11 := by
  sorry

end boys_who_did_not_bring_laptops_l1927_192771


namespace variable_value_l1927_192777

theorem variable_value (w x v : ℝ) (h1 : 5 / w + 5 / x = 5 / v) (h2 : w * x = v) (h3 : (w + x) / 2 = 0.5) : v = 0.25 :=
by
  sorry

end variable_value_l1927_192777


namespace river_width_proof_l1927_192715
noncomputable def river_width (V FR D : ℝ) : ℝ := V / (FR * D)

theorem river_width_proof :
  river_width 2933.3333333333335 33.33333333333333 4 = 22 :=
by
  simp [river_width]
  norm_num
  sorry

end river_width_proof_l1927_192715


namespace sufficient_but_not_necessary_l1927_192739

theorem sufficient_but_not_necessary (x y : ℝ) : 
  (x ≥ 2 ∧ y ≥ 2) → x + y ≥ 4 ∧ (¬ (x + y ≥ 4 → x ≥ 2 ∧ y ≥ 2)) :=
by
  sorry

end sufficient_but_not_necessary_l1927_192739


namespace trees_died_in_typhoon_imply_all_died_l1927_192709

-- Given conditions
def trees_initial := 3
def survived_trees (x : Int) := x
def died_trees (x : Int) := x + 23

-- Prove that the number of died trees is 3
theorem trees_died_in_typhoon_imply_all_died : ∀ x, 2 * survived_trees x + 23 = trees_initial → trees_initial = died_trees x := 
by
  intro x h
  sorry

end trees_died_in_typhoon_imply_all_died_l1927_192709


namespace max_value_y_l1927_192745

noncomputable def max_y (a b c d : ℝ) : ℝ :=
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2

theorem max_value_y {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = 10) : max_y a b c d = 40 := 
  sorry

end max_value_y_l1927_192745


namespace largest_n_divisible_l1927_192708

theorem largest_n_divisible (n : ℕ) (h : (n ^ 3 + 144) % (n + 12) = 0) : n ≤ 84 :=
sorry

end largest_n_divisible_l1927_192708


namespace eve_age_l1927_192786

variable (E : ℕ)

theorem eve_age (h1 : ∀ (a : ℕ), a = 9 → (E + 1) = 3 * (9 - 4)) : E = 14 := 
by
  have h2 : 9 - 4 = 5 := by norm_num
  have h3 : 3 * 5 = 15 := by norm_num
  have h4 : (E + 1) = 15 := h1 9 rfl
  linarith

end eve_age_l1927_192786


namespace chocolate_bars_in_box_l1927_192761

theorem chocolate_bars_in_box (x : ℕ) (h1 : 2 * (x - 4) = 18) : x = 13 := 
by {
  sorry
}

end chocolate_bars_in_box_l1927_192761


namespace find_a100_l1927_192727

theorem find_a100 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 2, a n = (2 * (S n)^2) / (2 * (S n) - 1))
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 100 = -2 / 39203 := 
sorry

-- Explanation of the statement:
-- 'theorem find_a100': We define a theorem to find a_100.
-- 'a : ℕ → ℝ': a is a sequence of real numbers.
-- 'S : ℕ → ℝ': S is a sequence representing the sum of the first n terms.
-- 'h1' to 'h3': Given conditions from the problem statement.
-- 'a 100 = -2 / 39203' : The statement to prove.

end find_a100_l1927_192727


namespace marble_group_size_l1927_192792

-- Define the conditions
def num_marbles : ℕ := 220
def future_people (x : ℕ) : ℕ := x + 2
def marbles_per_person (x : ℕ) : ℕ := num_marbles / x
def marbles_if_2_more (x : ℕ) : ℕ := num_marbles / future_people x

-- Statement of the theorem
theorem marble_group_size (x : ℕ) :
  (marbles_per_person x - 1 = marbles_if_2_more x) ↔ x = 20 :=
sorry

end marble_group_size_l1927_192792


namespace printer_a_time_l1927_192719

theorem printer_a_time :
  ∀ (A B : ℕ), 
  B = A + 4 → 
  A + B = 12 → 
  (480 / A = 120) :=
by 
  intros A B hB hAB
  sorry

end printer_a_time_l1927_192719


namespace John_other_trip_length_l1927_192764

theorem John_other_trip_length :
  ∀ (fuel_per_km total_fuel first_trip_length other_trip_length : ℕ),
    fuel_per_km = 5 →
    total_fuel = 250 →
    first_trip_length = 20 →
    total_fuel / fuel_per_km - first_trip_length = other_trip_length →
    other_trip_length = 30 :=
by
  intros fuel_per_km total_fuel first_trip_length other_trip_length h1 h2 h3 h4
  sorry

end John_other_trip_length_l1927_192764


namespace probability_of_x_gt_3y_is_correct_l1927_192754

noncomputable def probability_x_gt_3y : ℚ :=
  let rectangle_width := 2016
  let rectangle_height := 2017
  let triangle_height := 672 -- 2016 / 3
  let triangle_area := 1 / 2 * rectangle_width * triangle_height
  let rectangle_area := rectangle_width * rectangle_height
  triangle_area / rectangle_area

theorem probability_of_x_gt_3y_is_correct :
  probability_x_gt_3y = 336 / 2017 :=
by
  -- Proof will be filled in later
  sorry

end probability_of_x_gt_3y_is_correct_l1927_192754


namespace sum_of_odd_integers_15_to_51_l1927_192704

def odd_arithmetic_series_sum (a1 an d : ℤ) (n : ℕ) : ℤ :=
  (n * (a1 + an)) / 2

theorem sum_of_odd_integers_15_to_51 :
  odd_arithmetic_series_sum 15 51 2 19 = 627 :=
by
  sorry

end sum_of_odd_integers_15_to_51_l1927_192704


namespace lattice_points_on_hyperbola_l1927_192772

-- Define the problem
def countLatticePoints (n : ℤ) : ℕ :=
  let factoredCount := (2 + 1) * (2 + 1) * (4 + 1) -- Number of divisors of 2^2 * 3^2 * 5^4
  2 * factoredCount -- Each pair has two solutions considering positive and negative values

-- The theorem to be proven
theorem lattice_points_on_hyperbola : countLatticePoints 1800 = 90 := sorry

end lattice_points_on_hyperbola_l1927_192772


namespace domain_of_function_l1927_192720

noncomputable def is_defined (x : ℝ) : Prop :=
  (x + 4 ≥ 0) ∧ (x ≠ 0)

theorem domain_of_function :
  ∀ x : ℝ, is_defined x ↔ x ≥ -4 ∧ x ≠ 0 :=
by
  sorry

end domain_of_function_l1927_192720


namespace prime_square_sum_eq_square_iff_l1927_192738

theorem prime_square_sum_eq_square_iff (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q):
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
sorry

end prime_square_sum_eq_square_iff_l1927_192738


namespace max_value_of_f_l1927_192795

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_of_f :
  ∃ x ∈ Set.Icc (0 : ℝ) 4, ∀ y ∈ Set.Icc (0 : ℝ) 4, f y ≤ f x ∧ f x = 1 / Real.exp 1 := 
by
  sorry

end max_value_of_f_l1927_192795


namespace combined_cost_price_l1927_192767

def cost_price_A : ℕ := (120 + 60) / 2
def cost_price_B : ℕ := (200 + 100) / 2
def cost_price_C : ℕ := (300 + 180) / 2

def total_cost_price : ℕ := cost_price_A + cost_price_B + cost_price_C

theorem combined_cost_price :
  total_cost_price = 480 := by
  sorry

end combined_cost_price_l1927_192767


namespace good_numbers_l1927_192701

def is_divisor (a b : ℕ) : Prop := b % a = 0

def is_odd_prime (n : ℕ) : Prop :=
  Prime n ∧ n % 2 = 1

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, is_divisor d n → is_divisor (d + 1) (n + 1)

theorem good_numbers :
  ∀ n : ℕ, is_good n ↔ n = 1 ∨ is_odd_prime n :=
sorry

end good_numbers_l1927_192701


namespace percentage_of_material_A_in_second_solution_l1927_192769

theorem percentage_of_material_A_in_second_solution 
  (material_A_first_solution : ℝ)
  (material_B_first_solution : ℝ)
  (material_B_second_solution : ℝ)
  (material_A_mixture : ℝ)
  (percentage_first_solution_in_mixture : ℝ)
  (percentage_second_solution_in_mixture : ℝ)
  (total_mixture: ℝ)
  (hyp1 : material_A_first_solution = 20 / 100)
  (hyp2 : material_B_first_solution = 80 / 100)
  (hyp3 : material_B_second_solution = 70 / 100)
  (hyp4 : material_A_mixture = 22 / 100)
  (hyp5 : percentage_first_solution_in_mixture = 80 / 100)
  (hyp6 : percentage_second_solution_in_mixture = 20 / 100)
  (hyp7 : percentage_first_solution_in_mixture + percentage_second_solution_in_mixture = total_mixture)
  : ∃ (x : ℝ), x = 30 := by
  sorry

end percentage_of_material_A_in_second_solution_l1927_192769


namespace range_of_c_l1927_192734

theorem range_of_c (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 4 / b = 1) : ∀ c : ℝ, c < 9 → a + b > c :=
by
  sorry

end range_of_c_l1927_192734


namespace value_of_at_20_at_l1927_192756

noncomputable def left_at (x : ℝ) : ℝ := 9 - x
noncomputable def right_at (x : ℝ) : ℝ := x - 9

theorem value_of_at_20_at : right_at (left_at 20) = -20 := by
  sorry

end value_of_at_20_at_l1927_192756


namespace solve_inequality_l1927_192790

theorem solve_inequality (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 6) 
  (h3 : x ≠ 1) 
  (h4 : x ≠ 2) :
  (x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2 ∪ Set.Ioo 2 6)) → 
  ((x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2 ∪ Set.Icc 3 5))) :=
by 
  introv h
  sorry

end solve_inequality_l1927_192790


namespace product_of_ab_l1927_192732

theorem product_of_ab (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 13) : a * b = -6 :=
by
  sorry

end product_of_ab_l1927_192732


namespace functional_equation_solution_l1927_192750

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_equation_solution_l1927_192750


namespace Jane_buys_three_bagels_l1927_192780

theorem Jane_buys_three_bagels (b m c : ℕ) (h1 : b + m + c = 5) (h2 : 80 * b + 60 * m + 100 * c = 400) : b = 3 := 
sorry

end Jane_buys_three_bagels_l1927_192780


namespace smallest_b_factors_l1927_192760

theorem smallest_b_factors 
: ∃ b : ℕ, b > 0 ∧ 
    (∃ p q : ℤ, x^2 + b * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) ∧ 
    ∀ b': ℕ, (∃ p q: ℤ, x^2 + b' * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) → (b ≤ b') := 
sorry

end smallest_b_factors_l1927_192760


namespace max_value_of_exp_l1927_192716

theorem max_value_of_exp (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) : 
  a^2 * b^3 * c ≤ 27 / 16 := 
  sorry

end max_value_of_exp_l1927_192716


namespace trees_died_more_than_survived_l1927_192742

theorem trees_died_more_than_survived :
  ∀ (initial_trees survived_percent : ℕ),
    initial_trees = 25 →
    survived_percent = 40 →
    (initial_trees * survived_percent / 100) + (initial_trees - initial_trees * survived_percent / 100) -
    (initial_trees * survived_percent / 100) = 5 :=
by
  intro initial_trees survived_percent initial_trees_eq survived_percent_eq
  sorry

end trees_died_more_than_survived_l1927_192742


namespace common_factor_of_polynomial_l1927_192717

theorem common_factor_of_polynomial :
  ∀ (x : ℝ), (2 * x^2 - 8 * x) = 2 * x * (x - 4) := by
  sorry

end common_factor_of_polynomial_l1927_192717


namespace sequence_general_formula_l1927_192791

theorem sequence_general_formula :
  ∃ (a : ℕ → ℕ), 
    (a 1 = 4) ∧ 
    (∀ n : ℕ, a (n + 1) = a n + 3) ∧ 
    (∀ n : ℕ, a n = 3 * n + 1) :=
sorry

end sequence_general_formula_l1927_192791


namespace total_vertical_distance_of_rings_l1927_192775

theorem total_vertical_distance_of_rings :
  let thickness := 2
  let top_outside_diameter := 20
  let bottom_outside_diameter := 4
  let n := (top_outside_diameter - bottom_outside_diameter) / thickness + 1
  let total_distance := n * thickness
  total_distance + thickness = 76 :=
by
  sorry

end total_vertical_distance_of_rings_l1927_192775


namespace factor_squared_of_symmetric_poly_l1927_192757

theorem factor_squared_of_symmetric_poly (P : Polynomial ℤ → Polynomial ℤ → Polynomial ℤ)
  (h_symm : ∀ x y, P x y = P y x)
  (h_factor : ∀ x y, (x - y) ∣ P x y) :
  ∀ x y, (x - y) ^ 2 ∣ P x y := 
sorry

end factor_squared_of_symmetric_poly_l1927_192757


namespace solve_for_x_l1927_192748

theorem solve_for_x (x : ℤ) (h : 13 * x + 14 * x + 17 * x + 11 = 143) : x = 3 :=
by sorry

end solve_for_x_l1927_192748


namespace perfect_square_trinomial_l1927_192753

theorem perfect_square_trinomial (m : ℝ) : (∃ (a b : ℝ), (a * x + b) ^ 2 = x^2 + m * x + 16) -> (m = 8 ∨ m = -8) :=
sorry

end perfect_square_trinomial_l1927_192753


namespace evan_books_in_ten_years_l1927_192758

def E4 : ℕ := 400
def E_now : ℕ := E4 - 80
def E2 : ℕ := E_now / 2
def E10 : ℕ := 6 * E2 + 120

theorem evan_books_in_ten_years : E10 = 1080 := by
sorry

end evan_books_in_ten_years_l1927_192758


namespace original_number_is_24_l1927_192751

def number_parts (x y original_number : ℝ) : Prop :=
  7 * x + 5 * y = 146 ∧ x = 13 ∧ original_number = x + y

theorem original_number_is_24 :
  ∃ (x y original_number : ℝ), number_parts x y original_number ∧ original_number = 24 :=
by
  sorry

end original_number_is_24_l1927_192751


namespace james_chess_learning_time_l1927_192714

theorem james_chess_learning_time (R : ℝ) 
    (h1 : R + 49 * R + 100 * (R + 49 * R) = 10100) 
    : R = 2 :=
by 
  sorry

end james_chess_learning_time_l1927_192714


namespace smallest_number_condition_l1927_192712

theorem smallest_number_condition :
  ∃ n, 
  (n > 0) ∧ 
  (∀ k, k < n → (n - 3) % 12 = 0 ∧ (n - 3) % 16 = 0 ∧ (n - 3) % 18 = 0 ∧ (n - 3) % 21 = 0 ∧ (n - 3) % 28 = 0 → k = 0) ∧
  (n - 3) % 12 = 0 ∧
  (n - 3) % 16 = 0 ∧
  (n - 3) % 18 = 0 ∧
  (n - 3) % 21 = 0 ∧
  (n - 3) % 28 = 0 ∧
  n = 1011 :=
sorry

end smallest_number_condition_l1927_192712


namespace point_on_inverse_graph_and_sum_l1927_192783

-- Definitions
variable (f : ℝ → ℝ)
variable (h : f 2 = 6)

-- Theorem statement
theorem point_on_inverse_graph_and_sum (hf : ∀ x, x = 2 → 3 = (f x) / 2) :
  (6, 1 / 2) ∈ {p : ℝ × ℝ | ∃ x, p = (x, (f⁻¹ x) / 2)} ∧
  (6 + (1 / 2) = 13 / 2) :=
by
  sorry

end point_on_inverse_graph_and_sum_l1927_192783


namespace find_missing_number_l1927_192746

theorem find_missing_number (x : ℕ) :
  (6 + 16 + 8 + x) / 4 = 13 → x = 22 :=
by
  sorry

end find_missing_number_l1927_192746


namespace sum_cubes_first_39_eq_608400_l1927_192765

def sum_of_cubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

theorem sum_cubes_first_39_eq_608400 : sum_of_cubes 39 = 608400 :=
by
  sorry

end sum_cubes_first_39_eq_608400_l1927_192765


namespace sum_of_a_b_l1927_192713

-- Define the conditions in Lean
def a : ℝ := 1
def b : ℝ := 1

-- Define the proof statement
theorem sum_of_a_b : a + b = 2 := by
  sorry

end sum_of_a_b_l1927_192713


namespace sin_double_angle_l1927_192762

theorem sin_double_angle (x : ℝ) (h : Real.tan x = 1 / 3) : Real.sin (2 * x) = 3 / 5 := 
by 
  sorry

end sin_double_angle_l1927_192762


namespace hexagon_side_equalities_l1927_192759

variables {A B C D E F : Type}

-- Define the properties and conditions of the problem
noncomputable def convex_hexagon (A B C D E F : Type) : Prop :=
  True -- Since we neglect geometric properties in this abstract.

def parallel (a b : Type) : Prop := True -- placeholder for parallel condition
def equal_length (a b : Type) : Prop := True -- placeholder for length

-- Given conditions
variables (h1 : convex_hexagon A B C D E F)
variables (h2 : parallel AB DE)
variables (h3 : parallel BC FA)
variables (h4 : parallel CD FA)
variables (h5 : equal_length AB DE)

-- Statement to prove
theorem hexagon_side_equalities : equal_length BC DE ∧ equal_length CD FA := sorry

end hexagon_side_equalities_l1927_192759


namespace triangle_angle_inequality_l1927_192724

theorem triangle_angle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  4 / A + 1 / (B + C) ≥ 9 / Real.pi := by
  sorry

end triangle_angle_inequality_l1927_192724
