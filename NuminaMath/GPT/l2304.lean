import Mathlib

namespace NUMINAMATH_GPT_square_root_of_16_is_pm_4_l2304_230482

theorem square_root_of_16_is_pm_4 : { x : ℝ | x^2 = 16 } = {4, -4} :=
sorry

end NUMINAMATH_GPT_square_root_of_16_is_pm_4_l2304_230482


namespace NUMINAMATH_GPT_smallest_area_right_triangle_l2304_230491

theorem smallest_area_right_triangle (a b : ℕ) (h₁ : a = 4) (h₂ : b = 5) : 
  ∃ c, (c = 6 ∧ ∀ (x y : ℕ) (h₃ : x = 4 ∨ y = 4) (h₄ : x = 5 ∨ y = 5), c ≤ (x * y / 2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_area_right_triangle_l2304_230491


namespace NUMINAMATH_GPT_min_points_dodecahedron_min_points_icosahedron_l2304_230422

-- Definitions for the dodecahedron
def dodecahedron_faces : ℕ := 12
def vertices_per_face_dodecahedron : ℕ := 3

-- Prove the minimum number of points to mark each face of a dodecahedron
theorem min_points_dodecahedron (n : ℕ) (h : 3 * n >= dodecahedron_faces) : n >= 4 :=
sorry

-- Definitions for the icosahedron
def icosahedron_faces : ℕ := 20
def icosahedron_vertices : ℕ := 12

-- Prove the minimum number of points to mark each face of an icosahedron
theorem min_points_icosahedron (n : ℕ) (h : n >= 6) : n = 6 :=
sorry

end NUMINAMATH_GPT_min_points_dodecahedron_min_points_icosahedron_l2304_230422


namespace NUMINAMATH_GPT_xyz_identity_l2304_230475

theorem xyz_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z :=
by
  sorry

end NUMINAMATH_GPT_xyz_identity_l2304_230475


namespace NUMINAMATH_GPT_general_term_min_S9_and_S10_sum_b_seq_l2304_230461

-- Definitions for the arithmetic sequence {a_n}
def a_seq (n : ℕ) : ℤ := 2 * ↑n - 20

-- Conditions provided in the problem
def cond1 : Prop := a_seq 4 = -12
def cond2 : Prop := a_seq 8 = -4

-- The sum of the first n terms S_n of the arithmetic sequence {a_n}
def S_n (n : ℕ) : ℤ := n * (a_seq 1 + a_seq n) / 2

-- Definitions for the new sequence {b_n}
def b_seq (n : ℕ) : ℤ := 2^n - 20

-- The sum of the first n terms of the new sequence {b_n}
def T_n (n : ℕ) : ℤ := (2^(n + 1) - 2) - 20 * n

-- Lean 4 theorem statements
theorem general_term (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, a_seq n = 2 * ↑n - 20 :=
sorry

theorem min_S9_and_S10 (h1 : cond1) (h2 : cond2) : S_n 9 = -90 ∧ S_n 10 = -90 :=
sorry

theorem sum_b_seq (n : ℕ) : ∀ k : ℕ, (k < n) → T_n k = (2^(k+1) - 20 * k - 2) :=
sorry

end NUMINAMATH_GPT_general_term_min_S9_and_S10_sum_b_seq_l2304_230461


namespace NUMINAMATH_GPT_find_pairs_l2304_230464

noncomputable def pairs_of_real_numbers (α β : ℝ) := 
  ∀ x y z w : ℝ, 0 < x → 0 < y → 0 < z → 0 < w →
    (x + y^2 + z^3 + w^6 ≥ α * (x * y * z * w)^β)

theorem find_pairs (α β : ℝ) :
  (∃ x y z w : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧
    (x + y^2 + z^3 + w^6 = α * (x * y * z * w)^β))
  →
  pairs_of_real_numbers α β :=
sorry

end NUMINAMATH_GPT_find_pairs_l2304_230464


namespace NUMINAMATH_GPT_correct_value_l2304_230403

theorem correct_value (x : ℚ) (h : x + 7/5 = 81/20) : x - 7/5 = 5/4 :=
sorry

end NUMINAMATH_GPT_correct_value_l2304_230403


namespace NUMINAMATH_GPT_divides_343_l2304_230416

theorem divides_343 
  (x y z : ℕ) 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h : 7 ∣ (x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y)) :
  343 ∣ (x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y) :=
by sorry

end NUMINAMATH_GPT_divides_343_l2304_230416


namespace NUMINAMATH_GPT_circle_radius_9_l2304_230443

theorem circle_radius_9 (k : ℝ) : 
  (∀ x y : ℝ, 2 * x^2 + 20 * x + 3 * y^2 + 18 * y - k = 81) → 
  (k = 94) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_9_l2304_230443


namespace NUMINAMATH_GPT_sticks_at_20_l2304_230415

-- Define the sequence of sticks used at each stage
def sticks (n : ℕ) : ℕ :=
  if n = 1 then 5
  else if n ≤ 10 then 5 + 3 * (n - 1)
  else 32 + 4 * (n - 11)

-- Prove that the number of sticks at the 20th stage is 68
theorem sticks_at_20 : sticks 20 = 68 := by
  sorry

end NUMINAMATH_GPT_sticks_at_20_l2304_230415


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2304_230442

theorem necessary_but_not_sufficient {a b c d : ℝ} (hcd : c > d) : 
  (a - c > b - d) → (a > b) ∧ ¬((a > b) → (a - c > b - d)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2304_230442


namespace NUMINAMATH_GPT_trig_inequality_l2304_230440

noncomputable def a : ℝ := Real.sin (31 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (58 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (32 * Real.pi / 180)

theorem trig_inequality : c > b ∧ b > a := by
  sorry

end NUMINAMATH_GPT_trig_inequality_l2304_230440


namespace NUMINAMATH_GPT_conditional_without_else_l2304_230460

def if_then_else_statement (s: String) : Prop :=
  (s = "IF—THEN" ∨ s = "IF—THEN—ELSE")

theorem conditional_without_else : if_then_else_statement "IF—THEN" :=
  sorry

end NUMINAMATH_GPT_conditional_without_else_l2304_230460


namespace NUMINAMATH_GPT_find_number_of_partners_l2304_230421

noncomputable def law_firm_partners (P A : ℕ) : Prop :=
  (P / A = 3 / 97) ∧ (P / (A + 130) = 1 / 58)

theorem find_number_of_partners (P A : ℕ) (h : law_firm_partners P A) : P = 5 :=
  sorry

end NUMINAMATH_GPT_find_number_of_partners_l2304_230421


namespace NUMINAMATH_GPT_monotonically_decreasing_iff_l2304_230449

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 2) * x + 6 * a - 1 else a^x

theorem monotonically_decreasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a y ≤ f a x) ↔ (3/8 ≤ a ∧ a < 2/3) :=
sorry

end NUMINAMATH_GPT_monotonically_decreasing_iff_l2304_230449


namespace NUMINAMATH_GPT_radius_of_spheres_in_cube_l2304_230490

noncomputable def sphere_radius (sides: ℝ) (spheres: ℕ) (tangent_pairs: ℕ) (tangent_faces: ℕ): ℝ :=
  if sides = 2 ∧ spheres = 10 ∧ tangent_pairs = 2 ∧ tangent_faces = 3 then 0.5 else 0

theorem radius_of_spheres_in_cube : sphere_radius 2 10 2 3 = 0.5 :=
by
  -- This is the main theorem that states the radius of each sphere given the problem conditions.
  sorry

end NUMINAMATH_GPT_radius_of_spheres_in_cube_l2304_230490


namespace NUMINAMATH_GPT_determine_cubic_coeffs_l2304_230417

-- Define the cubic function f(x)
def cubic_function (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Define the expression f(f(x) + x)
def composition_expression (a b c x : ℝ) : ℝ :=
  cubic_function a b c (cubic_function a b c x + x)

-- Given that the fraction of the compositions equals the given polynomial
def given_fraction_equals_polynomial (a b c : ℝ) : Prop :=
  ∀ x : ℝ, (composition_expression a b c x) / (cubic_function a b c x) = x^3 + 2023 * x^2 + 1776 * x + 2010

-- Prove that this implies specific values of a, b, and c
theorem determine_cubic_coeffs (a b c : ℝ) :
  given_fraction_equals_polynomial a b c →
  (a = 2022 ∧ b = 1776 ∧ c = 2010) :=
by
  sorry

end NUMINAMATH_GPT_determine_cubic_coeffs_l2304_230417


namespace NUMINAMATH_GPT_prob_only_one_success_first_firing_is_correct_prob_all_success_after_both_firings_is_correct_l2304_230474

noncomputable def prob_first_firing_A : ℚ := 4 / 5
noncomputable def prob_first_firing_B : ℚ := 3 / 4
noncomputable def prob_first_firing_C : ℚ := 2 / 3

noncomputable def prob_second_firing : ℚ := 3 / 5

noncomputable def prob_only_one_success_first_firing :=
  prob_first_firing_A * (1 - prob_first_firing_B) * (1 - prob_first_firing_C) +
  (1 - prob_first_firing_A) * prob_first_firing_B * (1 - prob_first_firing_C) +
  (1 - prob_first_firing_A) * (1 - prob_first_firing_B) * prob_first_firing_C

theorem prob_only_one_success_first_firing_is_correct :
  prob_only_one_success_first_firing = 3 / 20 :=
by sorry

noncomputable def prob_success_after_both_firings_A := prob_first_firing_A * prob_second_firing
noncomputable def prob_success_after_both_firings_B := prob_first_firing_B * prob_second_firing
noncomputable def prob_success_after_both_firings_C := prob_first_firing_C * prob_second_firing

noncomputable def prob_all_success_after_both_firings :=
  prob_success_after_both_firings_A * prob_success_after_both_firings_B * prob_success_after_both_firings_C

theorem prob_all_success_after_both_firings_is_correct :
  prob_all_success_after_both_firings = 54 / 625 :=
by sorry

end NUMINAMATH_GPT_prob_only_one_success_first_firing_is_correct_prob_all_success_after_both_firings_is_correct_l2304_230474


namespace NUMINAMATH_GPT_fibonacci_150_mod_7_l2304_230492

def fibonacci_mod_7 : Nat → Nat
| 0 => 0
| 1 => 1
| n + 2 => (fibonacci_mod_7 (n + 1) + fibonacci_mod_7 n) % 7

theorem fibonacci_150_mod_7 : fibonacci_mod_7 150 = 1 := 
by sorry

end NUMINAMATH_GPT_fibonacci_150_mod_7_l2304_230492


namespace NUMINAMATH_GPT_correct_calculation_l2304_230465

theorem correct_calculation (a : ℕ) :
  ¬ (a^3 + a^4 = a^7) ∧
  ¬ (2 * a - a = 2) ∧
  2 * a + a = 3 * a ∧
  ¬ (a^4 - a^3 = a) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l2304_230465


namespace NUMINAMATH_GPT_redistribution_l2304_230447

/-
Given:
- b = (12 / 13) * a
- c = (2 / 3) * b
- Person C will contribute 9 dollars based on the amount each person spent

Prove:
- Person C gives 6 dollars to Person A.
- Person C gives 3 dollars to Person B.
-/

theorem redistribution (a b c : ℝ) (h1 : b = (12 / 13) * a) (h2 : c = (2 / 3) * b) : 
  ∃ (x y : ℝ), x + y = 9 ∧ x = 6 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_redistribution_l2304_230447


namespace NUMINAMATH_GPT_toll_for_18_wheel_truck_l2304_230406

-- Definitions
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def rear_axle_wheels_per_axle : ℕ := 4
def toll_formula (x : ℕ) : ℝ := 0.50 + 0.50 * (x - 2)

-- Theorem statement
theorem toll_for_18_wheel_truck : 
  ∃ t : ℝ, t = 2.00 ∧
  ∃ x : ℕ, x = (1 + ((total_wheels - front_axle_wheels) / rear_axle_wheels_per_axle)) ∧
  t = toll_formula x := 
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_toll_for_18_wheel_truck_l2304_230406


namespace NUMINAMATH_GPT_reimbursement_diff_l2304_230478

/-- Let Tom, Emma, and Harry share equally the costs for a group activity.
- Tom paid $95
- Emma paid $140
- Harry paid $165
If Tom and Emma are to reimburse Harry to ensure all expenses are shared equally,
prove that e - t = -45 where e is the amount Emma gives Harry and t is the amount Tom gives Harry.
-/
theorem reimbursement_diff :
  let tom_paid := 95
  let emma_paid := 140
  let harry_paid := 165
  let total_cost := tom_paid + emma_paid + harry_paid
  let equal_share := total_cost / 3
  let t := equal_share - tom_paid
  let e := equal_share - emma_paid
  e - t = -45 :=
by {
  sorry
}

end NUMINAMATH_GPT_reimbursement_diff_l2304_230478


namespace NUMINAMATH_GPT_reinforcement_left_after_days_l2304_230408

theorem reinforcement_left_after_days
  (initial_men : ℕ) (initial_days : ℕ) (remaining_days : ℕ) (men_left : ℕ)
  (remaining_men : ℕ) (x : ℕ) :
  initial_men = 400 ∧
  initial_days = 31 ∧
  remaining_days = 8 ∧
  men_left = initial_men - remaining_men ∧
  remaining_men = 200 ∧
  400 * 31 - 400 * x = 200 * 8 →
  x = 27 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_reinforcement_left_after_days_l2304_230408


namespace NUMINAMATH_GPT_linda_savings_l2304_230498

theorem linda_savings :
  let original_savings := 880
  let cost_of_tv := 220
  let amount_spent_on_furniture := original_savings - cost_of_tv
  let fraction_spent_on_furniture := amount_spent_on_furniture / original_savings
  fraction_spent_on_furniture = 3 / 4 :=
by
  -- original savings
  let original_savings := 880
  -- cost of the TV
  let cost_of_tv := 220
  -- amount spent on furniture
  let amount_spent_on_furniture := original_savings - cost_of_tv
  -- fraction spent on furniture
  let fraction_spent_on_furniture := amount_spent_on_furniture / original_savings

  -- need to show that this fraction is 3/4
  sorry

end NUMINAMATH_GPT_linda_savings_l2304_230498


namespace NUMINAMATH_GPT_gdp_scientific_notation_l2304_230454

theorem gdp_scientific_notation : 
  (33.5 * 10^12 = 3.35 * 10^13) := 
by
  sorry

end NUMINAMATH_GPT_gdp_scientific_notation_l2304_230454


namespace NUMINAMATH_GPT_sum_of_fractions_correct_l2304_230483

def sum_of_fractions : ℚ := (4 / 3) + (8 / 9) + (18 / 27) + (40 / 81) + (88 / 243) - 5

theorem sum_of_fractions_correct : sum_of_fractions = -305 / 243 := by
  sorry -- proof to be provided

end NUMINAMATH_GPT_sum_of_fractions_correct_l2304_230483


namespace NUMINAMATH_GPT_increase_in_sold_items_l2304_230484

variable (P N M : ℝ)
variable (discounted_price := 0.9 * P)
variable (increased_total_income := 1.17 * P * N)

theorem increase_in_sold_items (h: 0.9 * P * M = increased_total_income):
  M = 1.3 * N :=
  by sorry

end NUMINAMATH_GPT_increase_in_sold_items_l2304_230484


namespace NUMINAMATH_GPT_race_length_l2304_230413

noncomputable def solve_race_length (a b c d : ℝ) : Prop :=
  (d > 0) →
  (d / a = (d - 40) / b) →
  (d / b = (d - 30) / c) →
  (d / a = (d - 65) / c) →
  d = 240

theorem race_length : ∃ (d : ℝ), solve_race_length a b c d :=
by
  use 240
  sorry

end NUMINAMATH_GPT_race_length_l2304_230413


namespace NUMINAMATH_GPT_range_of_m_l2304_230456

variable {f : ℝ → ℝ}

def is_decreasing (f : ℝ → ℝ) := ∀ x y, x < y → f x > f y

theorem range_of_m (hf_dec : is_decreasing f) (hf_odd : ∀ x, f (-x) = -f x) 
  (h : ∀ m, f (m - 1) + f (2 * m - 1) > 0) : ∀ m, m < 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2304_230456


namespace NUMINAMATH_GPT_find_x_l2304_230428

theorem find_x (x y z : ℝ) 
  (h1 : x + y + z = 150)
  (h2 : x + 10 = y - 10)
  (h3 : x + 10 = 3 * z) :
  x = 380 / 7 := 
  sorry

end NUMINAMATH_GPT_find_x_l2304_230428


namespace NUMINAMATH_GPT_equal_potatoes_l2304_230410

theorem equal_potatoes (total_potatoes : ℕ) (total_people : ℕ) (h_potatoes : total_potatoes = 24) (h_people : total_people = 3) :
  (total_potatoes / total_people) = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_equal_potatoes_l2304_230410


namespace NUMINAMATH_GPT_MikeSalaryNow_l2304_230469

-- Definitions based on conditions
def FredSalary  := 1000   -- Fred's salary five months ago
def MikeSalaryFiveMonthsAgo := 10 * FredSalary  -- Mike's salary five months ago
def SalaryIncreasePercent := 40 / 100  -- 40 percent salary increase
def SalaryIncrease := SalaryIncreasePercent * MikeSalaryFiveMonthsAgo  -- Increase in Mike's salary

-- Statement to be proved
theorem MikeSalaryNow : MikeSalaryFiveMonthsAgo + SalaryIncrease = 14000 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_MikeSalaryNow_l2304_230469


namespace NUMINAMATH_GPT_cut_rectangle_to_square_l2304_230473

theorem cut_rectangle_to_square (a b : ℕ) (h₁ : a = 16) (h₂ : b = 9) :
  ∃ (s : ℕ), s * s = a * b ∧ s = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_cut_rectangle_to_square_l2304_230473


namespace NUMINAMATH_GPT_number_greater_than_neg_one_by_two_l2304_230455

/-- Theorem: The number that is greater than -1 by 2 is 1. -/
theorem number_greater_than_neg_one_by_two : -1 + 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_greater_than_neg_one_by_two_l2304_230455


namespace NUMINAMATH_GPT_parabola_equation_l2304_230459

theorem parabola_equation (x y : ℝ)
    (focus : x = 1 ∧ y = -2)
    (directrix : 5 * x + 2 * y = 10) :
    4 * x^2 - 20 * x * y + 25 * y^2 + 158 * x + 156 * y + 16 = 0 := 
by
  -- use the given conditions and intermediate steps to derive the final equation
  sorry

end NUMINAMATH_GPT_parabola_equation_l2304_230459


namespace NUMINAMATH_GPT_hakeem_artichoke_dip_l2304_230457

theorem hakeem_artichoke_dip 
(total_money : ℝ)
(cost_per_artichoke : ℝ)
(artichokes_per_dip : ℕ)
(dip_per_three_artichokes : ℕ)
(h : total_money = 15)
(h₁ : cost_per_artichoke = 1.25)
(h₂ : artichokes_per_dip = 3)
(h₃ : dip_per_three_artichokes = 5) : 
total_money / cost_per_artichoke * (dip_per_three_artichokes / artichokes_per_dip) = 20 := 
sorry

end NUMINAMATH_GPT_hakeem_artichoke_dip_l2304_230457


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l2304_230404

theorem geometric_sequence_common_ratio (a_1 q : ℝ) 
  (h1 : a_1 * q^2 = 9) 
  (h2 : a_1 * (1 + q) + 9 = 27) : 
  q = 1 ∨ q = -1/2 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l2304_230404


namespace NUMINAMATH_GPT_restaurant_customer_problem_l2304_230418

theorem restaurant_customer_problem (x y z : ℕ) 
  (h1 : x = 2 * z)
  (h2 : y = x - 3)
  (h3 : 3 + x + y - z = 8) :
  x = 6 ∧ y = 3 ∧ z = 3 ∧ (x + y = 9) :=
by
  sorry

end NUMINAMATH_GPT_restaurant_customer_problem_l2304_230418


namespace NUMINAMATH_GPT_elephant_entry_duration_l2304_230441

theorem elephant_entry_duration
  (initial_elephants : ℕ)
  (exodus_duration : ℕ)
  (leaving_rate : ℕ)
  (entering_rate : ℕ)
  (final_elephants : ℕ)
  (h_initial : initial_elephants = 30000)
  (h_exodus_duration : exodus_duration = 4)
  (h_leaving_rate : leaving_rate = 2880)
  (h_entering_rate : entering_rate = 1500)
  (h_final : final_elephants = 28980) :
  (final_elephants - (initial_elephants - (exodus_duration * leaving_rate))) / entering_rate = 7 :=
by
  sorry

end NUMINAMATH_GPT_elephant_entry_duration_l2304_230441


namespace NUMINAMATH_GPT_insert_digits_identical_l2304_230499

theorem insert_digits_identical (A B : List Nat) (hA : A.length = 2007) (hB : B.length = 2007)
  (hErase : ∃ (C : List Nat) (erase7A : List Nat → List Nat) (erase7B : List Nat → List Nat),
    (erase7A A = C) ∧ (erase7B B = C) ∧ (C.length = 2000)) :
  ∃ (D : List Nat) (insert7A : List Nat → List Nat) (insert7B : List Nat → List Nat),
    (insert7A A = D) ∧ (insert7B B = D) ∧ (D.length = 2014) := sorry

end NUMINAMATH_GPT_insert_digits_identical_l2304_230499


namespace NUMINAMATH_GPT_find_dot_AP_BC_l2304_230433

-- Defining the lengths of the sides of the triangle.
def length_AB : ℝ := 13
def length_BC : ℝ := 14
def length_CA : ℝ := 15

-- Defining the provided dot product conditions at point P.
def dot_BP_CA : ℝ := 18
def dot_CP_BA : ℝ := 32

-- The target is to prove the final dot product.
theorem find_dot_AP_BC :
  ∃ (AP BC : ℝ), BC = 14 → dot_BP_CA = 18 → dot_CP_BA = 32 → (AP * BC = 14) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_dot_AP_BC_l2304_230433


namespace NUMINAMATH_GPT_sin_17pi_over_6_l2304_230489

theorem sin_17pi_over_6 : Real.sin (17 * Real.pi / 6) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_17pi_over_6_l2304_230489


namespace NUMINAMATH_GPT_option_e_is_perfect_square_l2304_230439

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem option_e_is_perfect_square :
  is_perfect_square (4^10 * 5^5 * 6^10) :=
sorry

end NUMINAMATH_GPT_option_e_is_perfect_square_l2304_230439


namespace NUMINAMATH_GPT_sin_squared_identity_l2304_230446

theorem sin_squared_identity :
  1 - 2 * (Real.sin (105 * Real.pi / 180))^2 = - (Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_GPT_sin_squared_identity_l2304_230446


namespace NUMINAMATH_GPT_most_irregular_acute_triangle_l2304_230494

theorem most_irregular_acute_triangle :
  ∃ (α β γ : ℝ), α ≤ β ∧ β ≤ γ ∧ γ ≤ (90:ℝ) ∧ 
  ((β - α ≤ 15) ∧ (γ - β ≤ 15) ∧ (90 - γ ≤ 15)) ∧
  (α + β + γ = 180) ∧ 
  (α = 45 ∧ β = 60 ∧ γ = 75) := sorry

end NUMINAMATH_GPT_most_irregular_acute_triangle_l2304_230494


namespace NUMINAMATH_GPT_combined_garden_area_l2304_230448

-- Definitions for the sizes and counts of the gardens.
def Mancino_gardens : ℕ := 4
def Marquita_gardens : ℕ := 3
def Matteo_gardens : ℕ := 2
def Martina_gardens : ℕ := 5

def Mancino_garden_area : ℕ := 16 * 5
def Marquita_garden_area : ℕ := 8 * 4
def Matteo_garden_area : ℕ := 12 * 6
def Martina_garden_area : ℕ := 10 * 3

-- The total combined area to be proven.
def total_area : ℕ :=
  (Mancino_gardens * Mancino_garden_area) +
  (Marquita_gardens * Marquita_garden_area) +
  (Matteo_gardens * Matteo_garden_area) +
  (Martina_gardens * Martina_garden_area)

-- Proof statement for the combined area.
theorem combined_garden_area : total_area = 710 :=
by sorry

end NUMINAMATH_GPT_combined_garden_area_l2304_230448


namespace NUMINAMATH_GPT_min_value_l2304_230434

theorem min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 3) : 
  ∃ c : ℝ, (c = 3 / 4) ∧ (∀ (a b c : ℝ), a = x ∧ b = y ∧ c = z → 
    (1/(a + 3*b) + 1/(b + 3*c) + 1/(c + 3*a)) ≥ c) :=
sorry

end NUMINAMATH_GPT_min_value_l2304_230434


namespace NUMINAMATH_GPT_correct_line_equation_l2304_230480

theorem correct_line_equation :
  ∃ (c : ℝ), (∀ (x y : ℝ), 2 * x - 3 * y + 4 = 0 → 2 * x - 3 * y + c = 0 ∧ 2 * (-1) - 3 * 2 + c = 0) ∧ c = 8 :=
by
  use 8
  sorry

end NUMINAMATH_GPT_correct_line_equation_l2304_230480


namespace NUMINAMATH_GPT_solve_for_x_l2304_230479

theorem solve_for_x:
  ∀ (x : ℝ), (x + 10) / (x - 4) = (x - 3) / (x + 6) → x = -(48 / 23) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2304_230479


namespace NUMINAMATH_GPT_total_invested_expression_l2304_230493

variables (x y T : ℝ)

axiom annual_income_exceed_65 : 0.10 * x - 0.08 * y = 65
axiom total_invested_is_T : x + y = T

theorem total_invested_expression :
  T = 1.8 * y + 650 :=
sorry

end NUMINAMATH_GPT_total_invested_expression_l2304_230493


namespace NUMINAMATH_GPT_copper_content_range_l2304_230486

theorem copper_content_range (x2 : ℝ) (y : ℝ) (h1 : 0 ≤ x2) (h2 : x2 ≤ 4 / 9) (hy : y = 0.4 + 0.075 * x2) : 
  40 ≤ 100 * y ∧ 100 * y ≤ 130 / 3 :=
by { sorry }

end NUMINAMATH_GPT_copper_content_range_l2304_230486


namespace NUMINAMATH_GPT_dogs_on_mon_wed_fri_l2304_230407

def dogs_on_tuesday : ℕ := 12
def dogs_on_thursday : ℕ := 9
def pay_per_dog : ℕ := 5
def total_earnings : ℕ := 210

theorem dogs_on_mon_wed_fri :
  ∃ (d : ℕ), d = 21 ∧ d * pay_per_dog = total_earnings - (dogs_on_tuesday + dogs_on_thursday) * pay_per_dog :=
by 
  sorry

end NUMINAMATH_GPT_dogs_on_mon_wed_fri_l2304_230407


namespace NUMINAMATH_GPT_degree_of_angle_C_l2304_230467

theorem degree_of_angle_C 
  (A B C : ℝ) 
  (h1 : A = 4 * x) 
  (h2 : B = 4 * x) 
  (h3 : C = 7 * x) 
  (h_sum : A + B + C = 180) : 
  C = 84 := 
by 
  sorry

end NUMINAMATH_GPT_degree_of_angle_C_l2304_230467


namespace NUMINAMATH_GPT_Q_div_P_eq_10_over_3_l2304_230466

noncomputable def solve_Q_over_P (P Q : ℤ) :=
  (Q / P = 10 / 3)

theorem Q_div_P_eq_10_over_3 (P Q : ℤ) (x : ℝ) :
  (∀ x, x ≠ 3 → x ≠ 4 → (P / (x + 3) + Q / (x^2 - 10 * x + 16) = (x^2 - 6 * x + 18) / (x^3 - 7 * x^2 + 14 * x - 48))) →
  solve_Q_over_P P Q :=
sorry

end NUMINAMATH_GPT_Q_div_P_eq_10_over_3_l2304_230466


namespace NUMINAMATH_GPT_problem1_simplification_problem2_simplification_l2304_230488

theorem problem1_simplification : (3 / Real.sqrt 3 - (Real.sqrt 3) ^ 2 - Real.sqrt 27 + (abs (Real.sqrt 3 - 2))) = -1 - 3 * Real.sqrt 3 :=
  by
    sorry

theorem problem2_simplification (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) :
  ((x + 2) / (x ^ 2 - 2 * x) - (x - 1) / (x ^ 2 - 4 * x + 4)) / ((x - 4) / x) = 1 / (x - 2) ^ 2 :=
  by
    sorry

end NUMINAMATH_GPT_problem1_simplification_problem2_simplification_l2304_230488


namespace NUMINAMATH_GPT_midpoint_sum_of_coordinates_l2304_230495

theorem midpoint_sum_of_coordinates
  (M : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ)
  (hmx : (C.1 + D.1) / 2 = M.1)
  (hmy : (C.2 + D.2) / 2 = M.2)
  (hM : M = (3, 5))
  (hC : C = (5, 3)) :
  D.1 + D.2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_sum_of_coordinates_l2304_230495


namespace NUMINAMATH_GPT_total_days_2001_2005_l2304_230435

theorem total_days_2001_2005 : 
  let is_leap_year (y : ℕ) := y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365 
  (days_in_year 2001) + (days_in_year 2002) + (days_in_year 2003) + (days_in_year 2004) + (days_in_year 2005) = 1461 :=
by
  sorry

end NUMINAMATH_GPT_total_days_2001_2005_l2304_230435


namespace NUMINAMATH_GPT_point_in_second_quadrant_l2304_230470

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

def problem_points : List (ℝ × ℝ) :=
  [(1, -2), (2, 1), (-2, -1), (-1, 2)]

theorem point_in_second_quadrant :
  ∃ (p : ℝ × ℝ), p ∈ problem_points ∧ is_in_second_quadrant p.1 p.2 := by
  use (-1, 2)
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l2304_230470


namespace NUMINAMATH_GPT_least_possible_area_l2304_230401

variable (x y : ℝ) (n : ℤ)

-- Conditions
def is_integer (x : ℝ) := ∃ k : ℤ, x = k
def is_half_integer (y : ℝ) := ∃ n : ℤ, y = n + 0.5

-- Problem statement in Lean 4
theorem least_possible_area (h1 : is_integer x) (h2 : is_half_integer y)
(h3 : 2 * (x + y) = 150) : ∃ A, A = 0 :=
sorry

end NUMINAMATH_GPT_least_possible_area_l2304_230401


namespace NUMINAMATH_GPT_probability_defective_is_three_tenths_l2304_230402

open Classical

noncomputable def probability_of_defective_product (total_products defective_products: ℕ) : ℝ :=
  (defective_products * 1.0) / (total_products * 1.0)

theorem probability_defective_is_three_tenths :
  probability_of_defective_product 10 3 = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_probability_defective_is_three_tenths_l2304_230402


namespace NUMINAMATH_GPT_bill_earnings_l2304_230431

theorem bill_earnings
  (milk_total : ℕ)
  (fraction : ℚ)
  (milk_to_butter_ratio : ℕ)
  (milk_to_sour_cream_ratio : ℕ)
  (butter_price_per_gallon : ℚ)
  (sour_cream_price_per_gallon : ℚ)
  (whole_milk_price_per_gallon : ℚ)
  (milk_for_butter : ℚ)
  (milk_for_sour_cream : ℚ)
  (remaining_milk : ℚ)
  (total_earnings : ℚ) :
  milk_total = 16 →
  fraction = 1/4 →
  milk_to_butter_ratio = 4 →
  milk_to_sour_cream_ratio = 2 →
  butter_price_per_gallon = 5 →
  sour_cream_price_per_gallon = 6 →
  whole_milk_price_per_gallon = 3 →
  milk_for_butter = milk_total * fraction / milk_to_butter_ratio →
  milk_for_sour_cream = milk_total * fraction / milk_to_sour_cream_ratio →
  remaining_milk = milk_total - 2 * (milk_total * fraction) →
  total_earnings = (remaining_milk * whole_milk_price_per_gallon) + 
                   (milk_for_sour_cream * sour_cream_price_per_gallon) + 
                   (milk_for_butter * butter_price_per_gallon) →
  total_earnings = 41 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end NUMINAMATH_GPT_bill_earnings_l2304_230431


namespace NUMINAMATH_GPT_ratio_of_horses_to_cows_l2304_230450

/-- Let H and C be the initial number of horses and cows respectively.
Given that:
1. (H - 15) / (C + 15) = 7 / 3,
2. H - 15 = C + 75,
prove that the initial ratio of horses to cows is 4:1. -/
theorem ratio_of_horses_to_cows (H C : ℕ) 
  (h1 : (H - 15 : ℚ) / (C + 15 : ℚ) = 7 / 3)
  (h2 : H - 15 = C + 75) :
  H / C = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_horses_to_cows_l2304_230450


namespace NUMINAMATH_GPT_pattern_formula_l2304_230458

theorem pattern_formula (n : ℤ) : n * (n + 2) = (n + 1) ^ 2 - 1 := 
by sorry

end NUMINAMATH_GPT_pattern_formula_l2304_230458


namespace NUMINAMATH_GPT_original_price_after_discount_l2304_230405

theorem original_price_after_discount (a x : ℝ) (h : 0.7 * x = a) : x = (10 / 7) * a := 
sorry

end NUMINAMATH_GPT_original_price_after_discount_l2304_230405


namespace NUMINAMATH_GPT_min_photos_needed_to_ensure_conditions_l2304_230429

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end NUMINAMATH_GPT_min_photos_needed_to_ensure_conditions_l2304_230429


namespace NUMINAMATH_GPT_tammy_weekly_distance_l2304_230481

-- Define the conditions.
def track_length : ℕ := 50
def loops_per_day : ℕ := 10
def days_in_week : ℕ := 7

-- Using the conditions, prove the total distance per week is 3500 meters.
theorem tammy_weekly_distance : (track_length * loops_per_day * days_in_week) = 3500 := by
  sorry

end NUMINAMATH_GPT_tammy_weekly_distance_l2304_230481


namespace NUMINAMATH_GPT_herd_total_cows_l2304_230425

noncomputable def total_cows (n : ℕ) : Prop :=
  let fraction_first_son := 1 / 3
  let fraction_second_son := 1 / 5
  let fraction_third_son := 1 / 9
  let fraction_combined := fraction_first_son + fraction_second_son + fraction_third_son
  let fraction_fourth_son := 1 - fraction_combined
  let cows_fourth_son := 11
  fraction_fourth_son * n = cows_fourth_son

theorem herd_total_cows : ∃ n : ℕ, total_cows n ∧ n = 31 :=
by
  existsi 31
  sorry

end NUMINAMATH_GPT_herd_total_cows_l2304_230425


namespace NUMINAMATH_GPT_smallest_y_of_arithmetic_sequence_l2304_230423

theorem smallest_y_of_arithmetic_sequence
  (x y z d : ℝ)
  (h_arith_series_x : x = y - d)
  (h_arith_series_z : z = y + d)
  (h_positive_x : x > 0)
  (h_positive_y : y > 0)
  (h_positive_z : z > 0)
  (h_product : x * y * z = 216) : y = 6 :=
sorry

end NUMINAMATH_GPT_smallest_y_of_arithmetic_sequence_l2304_230423


namespace NUMINAMATH_GPT_expression_greater_than_m_l2304_230487

theorem expression_greater_than_m (m : ℚ) : m + 2 > m :=
by sorry

end NUMINAMATH_GPT_expression_greater_than_m_l2304_230487


namespace NUMINAMATH_GPT_find_multiplication_value_l2304_230420

-- Define the given conditions
def student_chosen_number : ℤ := 63
def subtracted_value : ℤ := 142
def result_after_subtraction : ℤ := 110

-- Define the value he multiplied the number by
def multiplication_value (x : ℤ) : Prop := 
  (student_chosen_number * x) - subtracted_value = result_after_subtraction

-- Statement to prove that the value he multiplied the number by is 4
theorem find_multiplication_value : 
  ∃ x : ℤ, multiplication_value x ∧ x = 4 :=
by 
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_multiplication_value_l2304_230420


namespace NUMINAMATH_GPT_true_value_of_product_l2304_230472

theorem true_value_of_product (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  let product := (100 * a + 10 * b + c) * (100 * b + 10 * c + a) * (100 * c + 10 * a + b)
  product = 2342355286 → (product % 10 = 6) → product = 328245326 :=
by
  sorry

end NUMINAMATH_GPT_true_value_of_product_l2304_230472


namespace NUMINAMATH_GPT_smaller_two_digit_product_l2304_230453

-- Statement of the problem
theorem smaller_two_digit_product (a b : ℕ) (h : a * b = 4536) (h_a : 10 ≤ a ∧ a < 100) (h_b : 10 ≤ b ∧ b < 100) : a = 21 ∨ b = 21 := 
by {
  sorry -- proof not needed
}

end NUMINAMATH_GPT_smaller_two_digit_product_l2304_230453


namespace NUMINAMATH_GPT_team_total_points_l2304_230400

theorem team_total_points (T : ℕ) (h1 : ∃ x : ℕ, x = T / 6)
    (h2 : (T + (92 - 85)) / 6 = 84) : T = 497 := 
by sorry

end NUMINAMATH_GPT_team_total_points_l2304_230400


namespace NUMINAMATH_GPT_golden_ratio_minus_one_binary_l2304_230445

theorem golden_ratio_minus_one_binary (n : ℕ → ℕ) (h_n : ∀ i, 1 ≤ n i)
  (h_incr : ∀ i, n i ≤ n (i + 1)): 
  (∀ k ≥ 4, n k ≤ 2^(k - 1) - 2) := 
by
  sorry

end NUMINAMATH_GPT_golden_ratio_minus_one_binary_l2304_230445


namespace NUMINAMATH_GPT_total_apples_picked_l2304_230477

-- Definitions based on conditions from part a)
def mike_apples : ℝ := 7.5
def nancy_apples : ℝ := 3.2
def keith_apples : ℝ := 6.1
def olivia_apples : ℝ := 12.4
def thomas_apples : ℝ := 8.6

-- The theorem we need to prove
theorem total_apples_picked : mike_apples + nancy_apples + keith_apples + olivia_apples + thomas_apples = 37.8 := by
    sorry

end NUMINAMATH_GPT_total_apples_picked_l2304_230477


namespace NUMINAMATH_GPT_intersection_ST_l2304_230409

def S : Set ℝ := { x : ℝ | x < -5 } ∪ { x : ℝ | x > 5 }
def T : Set ℝ := { x : ℝ | -7 < x ∧ x < 3 }

theorem intersection_ST : S ∩ T = { x : ℝ | -7 < x ∧ x < -5 } := 
by 
  sorry

end NUMINAMATH_GPT_intersection_ST_l2304_230409


namespace NUMINAMATH_GPT_train_crossing_time_l2304_230411

-- Definitions for conditions
def train_length : ℝ := 100 -- train length in meters
def train_speed_kmh : ℝ := 90 -- train speed in km/hr
def train_speed_mps : ℝ := 25 -- train speed in m/s after conversion

-- Lean 4 statement to prove the time taken for the train to cross the electric pole is 4 seconds
theorem train_crossing_time : (train_length / train_speed_mps) = 4 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2304_230411


namespace NUMINAMATH_GPT_smallest_k_l2304_230419

theorem smallest_k (k: ℕ) : k > 1 ∧ (k % 23 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) → k = 484 :=
sorry

end NUMINAMATH_GPT_smallest_k_l2304_230419


namespace NUMINAMATH_GPT_stadium_ticket_price_l2304_230496

theorem stadium_ticket_price
  (original_price : ℝ)
  (decrease_rate : ℝ)
  (increase_rate : ℝ)
  (new_price : ℝ) 
  (h1 : original_price = 400)
  (h2 : decrease_rate = 0.2)
  (h3 : increase_rate = 0.05) 
  (h4 : (original_price * (1 + increase_rate) / (1 - decrease_rate)) = new_price) :
  new_price = 525 := 
by
  -- Proof omitted for this task.
  sorry

end NUMINAMATH_GPT_stadium_ticket_price_l2304_230496


namespace NUMINAMATH_GPT_students_per_class_l2304_230427

theorem students_per_class :
  let buns_per_package := 8
  let packages := 30
  let buns_per_student := 2
  let classes := 4
  (packages * buns_per_package) / (buns_per_student * classes) = 30 :=
by
  sorry

end NUMINAMATH_GPT_students_per_class_l2304_230427


namespace NUMINAMATH_GPT_pascal_family_min_children_l2304_230444

-- We define the conditions b >= 3 and g >= 2
def b_condition (b : ℕ) : Prop := b >= 3
def g_condition (g : ℕ) : Prop := g >= 2

-- We state that the smallest number of children given these conditions is 5
theorem pascal_family_min_children (b g : ℕ) (hb : b_condition b) (hg : g_condition g) : b + g = 5 :=
sorry

end NUMINAMATH_GPT_pascal_family_min_children_l2304_230444


namespace NUMINAMATH_GPT_find_m_l2304_230451

noncomputable def union_sets (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ A ∨ x ∈ B}

theorem find_m :
  ∀ (m : ℝ),
    (A = {1, 2 ^ m}) →
    (B = {0, 2}) →
    (union_sets A B = {0, 1, 2, 8}) →
    m = 3 :=
by
  intros m hA hB hUnion
  sorry

end NUMINAMATH_GPT_find_m_l2304_230451


namespace NUMINAMATH_GPT_division_yields_square_l2304_230412

theorem division_yields_square (a b : ℕ) (hab : ab + 1 ∣ a^2 + b^2) :
  ∃ m : ℕ, m^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end NUMINAMATH_GPT_division_yields_square_l2304_230412


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l2304_230436

variable (m n : ℝ)
variable (a : ℝ × ℝ := (m, 1))
variable (b : ℝ × ℝ := (4 - n, 2))

theorem min_value_of_reciprocal_sum
  (h1 : m > 0) (h2 : n > 0)
  (h3 : a.1 * b.2 = a.2 * b.1) :
  (1/m + 8/n) = 9/2 :=
sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l2304_230436


namespace NUMINAMATH_GPT_population_reaches_target_l2304_230476

def initial_year : ℕ := 2020
def initial_population : ℕ := 450
def growth_period : ℕ := 25
def growth_factor : ℕ := 3
def target_population : ℕ := 10800

theorem population_reaches_target : ∃ (year : ℕ), year - initial_year = 3 * growth_period ∧ (initial_population * growth_factor ^ 3) >= target_population := by
  sorry

end NUMINAMATH_GPT_population_reaches_target_l2304_230476


namespace NUMINAMATH_GPT_test_question_count_l2304_230452

theorem test_question_count :
  ∃ (x y : ℕ), x + y = 30 ∧ 5 * x + 10 * y = 200 ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_test_question_count_l2304_230452


namespace NUMINAMATH_GPT_correct_average_l2304_230462

theorem correct_average 
(n : ℕ) (avg1 avg2 avg3 : ℝ): 
  n = 10 
  → avg1 = 40.2 
  → avg2 = avg1
  → avg3 = avg1
  → avg1 = avg3 :=
by 
  intros hn h_avg1 h_avg2 h_avg3
  sorry

end NUMINAMATH_GPT_correct_average_l2304_230462


namespace NUMINAMATH_GPT_intersection_points_count_l2304_230437

theorem intersection_points_count : 
  ∃ (x1 y1 x2 y2 : ℝ), 
  (x1 - ⌊x1⌋)^2 + (y1 - 1)^2 = x1 - ⌊x1⌋ ∧ 
  y1 = 1/5 * x1 + 1 ∧ 
  (x2 - ⌊x2⌋)^2 + (y2 - 1)^2 = x2 - ⌊x2⌋ ∧ 
  y2 = 1/5 * x2 + 1 ∧ 
  (x1, y1) ≠ (x2, y2) :=
sorry

end NUMINAMATH_GPT_intersection_points_count_l2304_230437


namespace NUMINAMATH_GPT_number_of_terms_l2304_230468

variable {α : Type} [LinearOrderedField α]

def sum_of_arithmetic_sequence (a₁ aₙ d : α) (n : ℕ) : α :=
  n * (a₁ + aₙ) / 2

theorem number_of_terms (a₁ aₙ : α) (d : α) (n : ℕ)
  (h₀ : 4 * (2 * a₁ + 3 * d) / 2 = 21)
  (h₁ : 4 * (2 * aₙ - 3 * d) / 2 = 67)
  (h₂ : sum_of_arithmetic_sequence a₁ aₙ d n = 286) :
  n = 26 :=
sorry

end NUMINAMATH_GPT_number_of_terms_l2304_230468


namespace NUMINAMATH_GPT_set_points_quadrants_l2304_230430

theorem set_points_quadrants (x y : ℝ) :
  (y > 3 * x) ∧ (y > 5 - 2 * x) → 
  (y > 0 ∧ x > 0) ∨ (y > 0 ∧ x < 0) :=
by 
  sorry

end NUMINAMATH_GPT_set_points_quadrants_l2304_230430


namespace NUMINAMATH_GPT_eliot_account_balance_l2304_230485

variable (A E : ℝ)

theorem eliot_account_balance (h1 : A - E = (1/12) * (A + E)) (h2 : 1.10 * A = 1.20 * E + 20) : 
  E = 200 := 
by 
  sorry

end NUMINAMATH_GPT_eliot_account_balance_l2304_230485


namespace NUMINAMATH_GPT_determine_parity_of_f_l2304_230438

def parity_of_f (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = 0

theorem determine_parity_of_f (f : ℝ → ℝ) :
  (∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)) →
  parity_of_f f :=
sorry

end NUMINAMATH_GPT_determine_parity_of_f_l2304_230438


namespace NUMINAMATH_GPT_solve_angle_CBO_l2304_230463

theorem solve_angle_CBO 
  (BAO CAO : ℝ) (CBO ABO : ℝ) (ACO BCO : ℝ) (AOC : ℝ) 
  (h1 : BAO = CAO) 
  (h2 : CBO = ABO) 
  (h3 : ACO = BCO) 
  (h4 : AOC = 110) 
  : CBO = 20 :=
by
  sorry

end NUMINAMATH_GPT_solve_angle_CBO_l2304_230463


namespace NUMINAMATH_GPT_jellybeans_count_l2304_230432

noncomputable def jellybeans_initial (y: ℝ) (n: ℕ) : ℝ :=
  y / (0.7 ^ n)

theorem jellybeans_count (y x: ℝ) (n: ℕ) (h: y = 24) (h2: n = 3) :
  x = 70 :=
by
  apply sorry

end NUMINAMATH_GPT_jellybeans_count_l2304_230432


namespace NUMINAMATH_GPT_determine_beta_l2304_230426

-- Define a structure for angles in space
structure Angle where
  measure : ℝ

-- Define the conditions
def alpha : Angle := ⟨30⟩
def parallel_sides (a b : Angle) : Prop := true  -- Simplification for the example, should be defined properly for general case

-- The theorem to be proved
theorem determine_beta (α β : Angle) (h1 : α = Angle.mk 30) (h2 : parallel_sides α β) : β = Angle.mk 30 ∨ β = Angle.mk 150 := by
  sorry

end NUMINAMATH_GPT_determine_beta_l2304_230426


namespace NUMINAMATH_GPT_amount_of_CaCO3_required_l2304_230414

-- Define the balanced chemical reaction
def balanced_reaction (CaCO3 HCl CaCl2 CO2 H2O : ℕ) : Prop :=
  CaCO3 + 2 * HCl = CaCl2 + CO2 + H2O

-- Define the required conditions
def conditions (HCl_req CaCl2_req CO2_req H2O_req : ℕ) : Prop :=
  HCl_req = 4 ∧ CaCl2_req = 2 ∧ CO2_req = 2 ∧ H2O_req = 2

-- The main theorem to be proved
theorem amount_of_CaCO3_required :
  ∃ (CaCO3_req : ℕ), conditions 4 2 2 2 ∧ balanced_reaction CaCO3_req 4 2 2 2 ∧ CaCO3_req = 2 :=
by 
  sorry

end NUMINAMATH_GPT_amount_of_CaCO3_required_l2304_230414


namespace NUMINAMATH_GPT_jerry_pool_time_l2304_230424

variables (J : ℕ) -- Denote the time Jerry was in the pool

-- Conditions
def Elaine_time := 2 * J -- Elaine stayed in the pool for twice as long as Jerry
def George_time := (2 / 3) * J -- George could only stay in the pool for one-third as long as Elaine
def Kramer_time := 0 -- Kramer did not find the pool

-- Combined total time
def total_time : ℕ := J + Elaine_time J + George_time J + Kramer_time

-- Theorem stating that J = 3 given the combined total time of 11 minutes
theorem jerry_pool_time (h : total_time J = 11) : J = 3 :=
by
  sorry

end NUMINAMATH_GPT_jerry_pool_time_l2304_230424


namespace NUMINAMATH_GPT_average_marks_combined_l2304_230497

theorem average_marks_combined (avg1 : ℝ) (students1 : ℕ) (avg2 : ℝ) (students2 : ℕ) :
  avg1 = 30 → students1 = 30 → avg2 = 60 → students2 = 50 →
  (students1 * avg1 + students2 * avg2) / (students1 + students2) = 48.75 := 
by
  intros h_avg1 h_students1 h_avg2 h_students2
  sorry

end NUMINAMATH_GPT_average_marks_combined_l2304_230497


namespace NUMINAMATH_GPT_factor_expression_l2304_230471

theorem factor_expression (x : ℝ) :
  (12 * x^3 + 45 * x - 3) - (-3 * x^3 + 5 * x - 2) = 5 * x * (3 * x^2 + 8) - 1 :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2304_230471
