import Mathlib

namespace max_unmarried_women_l2033_203329

theorem max_unmarried_women (total_people : ℕ) (frac_women : ℚ) (frac_married : ℚ)
  (h_total : total_people = 80) (h_frac_women : frac_women = 1 / 4) (h_frac_married : frac_married = 3 / 4) :
  ∃ (max_unmarried_women : ℕ), max_unmarried_women = 20 :=
by
  -- The proof will be filled here
  sorry

end max_unmarried_women_l2033_203329


namespace red_ball_probability_correct_l2033_203307

theorem red_ball_probability_correct (R B : ℕ) (hR : R = 3) (hB : B = 3) :
  (R / (R + B) : ℚ) = 1 / 2 := by
  sorry

end red_ball_probability_correct_l2033_203307


namespace elements_in_set_C_l2033_203313

-- Definitions and main theorem
variables (C D : Finset ℕ)  -- Define sets C and D as finite sets of natural numbers
open BigOperators    -- Opens notation for finite sums

-- Given conditions as premises
def condition1 (c d : ℕ) : Prop := c = 3 * d
def condition2 (C D : Finset ℕ) : Prop := (C ∪ D).card = 4500
def condition3 (C D : Finset ℕ) : Prop := (C ∩ D).card = 1200

-- Theorem statement to be proven
theorem elements_in_set_C (c d : ℕ) (h1 : condition1 c d)
  (h2 : ∀ (C D : Finset ℕ), condition2 C D)
  (h3 : ∀ (C D : Finset ℕ), condition3 C D) :
  c = 4275 :=
sorry  -- proof to be completed

end elements_in_set_C_l2033_203313


namespace line_intersects_y_axis_at_0_6_l2033_203341

theorem line_intersects_y_axis_at_0_6 : ∃ y : ℝ, 4 * y + 3 * (0 : ℝ) = 24 ∧ (0, y) = (0, 6) :=
by
  use 6
  simp
  sorry

end line_intersects_y_axis_at_0_6_l2033_203341


namespace reduced_price_l2033_203397

theorem reduced_price (P R : ℝ) (Q : ℝ) 
  (h1 : R = 0.80 * P) 
  (h2 : 600 = Q * P) 
  (h3 : 600 = (Q + 4) * R) : 
  R = 30 :=
by
  sorry

end reduced_price_l2033_203397


namespace fraction_zero_implies_a_eq_neg2_l2033_203325

theorem fraction_zero_implies_a_eq_neg2 (a : ℝ) (h : (a^2 - 4) / (a - 2) = 0) (h2 : a ≠ 2) : a = -2 :=
sorry

end fraction_zero_implies_a_eq_neg2_l2033_203325


namespace pie_eating_contest_l2033_203305

theorem pie_eating_contest :
  (7 / 8 : ℚ) - (5 / 6 : ℚ) = (1 / 24 : ℚ) :=
sorry

end pie_eating_contest_l2033_203305


namespace analytic_expression_of_f_max_min_of_f_on_interval_l2033_203388

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem analytic_expression_of_f :
  ∀ A ω φ : ℝ, (∀ x, f x = A * Real.sin (ω * x + φ)) →
  A = 2 ∧ ω = 2 ∧ φ = Real.pi / 6 :=
by
  sorry -- Placeholder for the actual proof

theorem max_min_of_f_on_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 12 → f x ≤ Real.sqrt 3) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 12 → f x ≥ 1) :=
by
  sorry -- Placeholder for the actual proof

end analytic_expression_of_f_max_min_of_f_on_interval_l2033_203388


namespace reciprocal_of_neg_three_l2033_203311

theorem reciprocal_of_neg_three : (1 / (-3 : ℝ)) = (-1 / 3) := by
  sorry

end reciprocal_of_neg_three_l2033_203311


namespace ab_value_l2033_203339

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 * b^2 + a^2 * b^3 = 20) : ab = 2 ∨ ab = -2 :=
by
  sorry

end ab_value_l2033_203339


namespace semicircle_radius_l2033_203378

theorem semicircle_radius (π : ℝ) (P : ℝ) (r : ℝ) (hπ : π ≠ 0) (hP : P = 162) (hPerimeter : P = π * r + 2 * r) : r = 162 / (π + 2) :=
by
  sorry

end semicircle_radius_l2033_203378


namespace solve_for_x_l2033_203354

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.1 * (30 + x) = 15.5 → x = 83 := by 
  sorry

end solve_for_x_l2033_203354


namespace johns_total_cost_l2033_203301

-- Definitions for the prices and quantities
def price_shirt : ℝ := 15.75
def price_tie : ℝ := 9.40
def quantity_shirts : ℕ := 3
def quantity_ties : ℕ := 2

-- Definition for the total cost calculation
def total_cost (price_shirt price_tie : ℝ) (quantity_shirts quantity_ties : ℕ) : ℝ :=
  (price_shirt * quantity_shirts) + (price_tie * quantity_ties)

-- Theorem stating the total cost calculation for John's purchase
theorem johns_total_cost : total_cost price_shirt price_tie quantity_shirts quantity_ties = 66.05 :=
by
  sorry

end johns_total_cost_l2033_203301


namespace negation_of_universal_proposition_l2033_203333

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l2033_203333


namespace Jesse_pages_left_to_read_l2033_203352

def pages_read := [10, 15, 27, 12, 19]
def total_pages_read := pages_read.sum
def fraction_read : ℚ := 1 / 3
def total_pages : ℚ := total_pages_read / fraction_read
def pages_left_to_read : ℚ := total_pages - total_pages_read

theorem Jesse_pages_left_to_read :
  pages_left_to_read = 166 := by
  sorry

end Jesse_pages_left_to_read_l2033_203352


namespace solution_set_M_minimum_value_expr_l2033_203338

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

-- Proof problem (1): Prove that the solution set M of the inequality f(x) ≥ -1 is {x | 2/3 ≤ x ≤ 6}.
theorem solution_set_M : 
  { x : ℝ | f x ≥ -1 } = { x : ℝ | 2/3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Define the requirement for t and the expression to minimize
noncomputable def t : ℝ := 6
noncomputable def expr (a b c : ℝ) : ℝ := 1 / (2 * a + b) + 1 / (2 * a + c)

-- Proof problem (2): Given t = 6 and 4a + b + c = 6, prove that the minimum value of expr is 2/3.
theorem minimum_value_expr (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = t) :
  expr a b c ≥ 2/3 :=
sorry

end solution_set_M_minimum_value_expr_l2033_203338


namespace pear_price_is_6300_l2033_203395

def price_of_pear (P : ℕ) : Prop :=
  P + (P + 2400) = 15000

theorem pear_price_is_6300 : ∃ (P : ℕ), price_of_pear P ∧ P = 6300 :=
by
  sorry

end pear_price_is_6300_l2033_203395


namespace combined_motion_properties_l2033_203348

noncomputable def y (x : ℝ) := Real.sin x + (Real.sin x) ^ 2

theorem combined_motion_properties :
  (∀ x: ℝ, - (1/4: ℝ) ≤ y x ∧ y x ≤ 2) ∧ 
  (∃ x: ℝ, y x = 2) ∧
  (∃ x: ℝ, y x = -(1/4: ℝ)) :=
by
  -- The complete proofs for these statements are omitted.
  -- This theorem specifies the required properties of the function y.
  sorry

end combined_motion_properties_l2033_203348


namespace number_exceeds_35_percent_by_245_l2033_203346

theorem number_exceeds_35_percent_by_245 : 
  ∃ (x : ℝ), (0.35 * x + 245 = x) ∧ x = 376.92 := 
by
  sorry

end number_exceeds_35_percent_by_245_l2033_203346


namespace canyon_trail_length_l2033_203356

theorem canyon_trail_length
  (a b c d e : ℝ)
  (h1 : a + b + c = 36)
  (h2 : b + c + d = 42)
  (h3 : c + d + e = 45)
  (h4 : a + d = 29) :
  a + b + c + d + e = 71 :=
by sorry

end canyon_trail_length_l2033_203356


namespace largest_four_digit_integer_congruent_to_17_mod_26_l2033_203372

theorem largest_four_digit_integer_congruent_to_17_mod_26 :
  ∃ x : ℤ, 1000 ≤ x ∧ x < 10000 ∧ x % 26 = 17 ∧ x = 9978 :=
by
  sorry

end largest_four_digit_integer_congruent_to_17_mod_26_l2033_203372


namespace square_of_binomial_example_l2033_203381

theorem square_of_binomial_example : (23^2 + 2 * 23 * 2 + 2^2 = 625) :=
by
  sorry

end square_of_binomial_example_l2033_203381


namespace flowers_per_bouquet_l2033_203361

theorem flowers_per_bouquet (total_flowers wilted_flowers : ℕ) (bouquets : ℕ) (remaining_flowers : ℕ)
    (h1 : total_flowers = 45)
    (h2 : wilted_flowers = 35)
    (h3 : bouquets = 2)
    (h4 : remaining_flowers = total_flowers - wilted_flowers)
    (h5 : bouquets * (remaining_flowers / bouquets) = remaining_flowers) :
  remaining_flowers / bouquets = 5 :=
by
  sorry

end flowers_per_bouquet_l2033_203361


namespace alice_instructors_l2033_203394

noncomputable def num_students : ℕ := 40
noncomputable def num_life_vests_Alice_has : ℕ := 20
noncomputable def percent_students_with_their_vests : ℕ := 20
noncomputable def num_additional_life_vests_needed : ℕ := 22

-- Constants based on calculated conditions
noncomputable def num_students_with_their_vests : ℕ := (percent_students_with_their_vests * num_students) / 100
noncomputable def num_students_without_their_vests : ℕ := num_students - num_students_with_their_vests
noncomputable def num_life_vests_needed_for_students : ℕ := num_students_without_their_vests - num_life_vests_Alice_has
noncomputable def num_life_vests_needed_for_instructors : ℕ := num_additional_life_vests_needed - num_life_vests_needed_for_students

theorem alice_instructors : num_life_vests_needed_for_instructors = 10 := 
by
  sorry

end alice_instructors_l2033_203394


namespace pencils_placed_by_sara_l2033_203347

theorem pencils_placed_by_sara (initial_pencils final_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : final_pencils = 215) : final_pencils - initial_pencils = 100 := by
  sorry

end pencils_placed_by_sara_l2033_203347


namespace calc_square_uncovered_area_l2033_203380

theorem calc_square_uncovered_area :
  ∀ (side_length : ℕ) (circle_diameter : ℝ) (num_circles : ℕ),
    side_length = 16 →
    circle_diameter = (16 / 3) →
    num_circles = 9 →
    (side_length ^ 2) - num_circles * (Real.pi * (circle_diameter / 2) ^ 2) = 256 - 64 * Real.pi :=
by
  intros side_length circle_diameter num_circles h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end calc_square_uncovered_area_l2033_203380


namespace arithmetic_sequence_max_value_l2033_203365

theorem arithmetic_sequence_max_value 
  (S : ℕ → ℤ)
  (k : ℕ)
  (h1 : 2 ≤ k)
  (h2 : S (k - 1) = 8)
  (h3 : S k = 0)
  (h4 : S (k + 1) = -10) :
  ∃ n, S n = 20 ∧ (∀ m, S m ≤ 20) :=
sorry

end arithmetic_sequence_max_value_l2033_203365


namespace total_bills_proof_l2033_203334

variable (a : ℝ) (total_may : ℝ) (total_june_may_june : ℝ)

-- The total bill in May is 140 yuan.
def total_bill_may (a : ℝ) := 140

-- The water bill increases by 10% in June.
def water_bill_june (a : ℝ) := 1.1 * a

-- The electricity bill in May.
def electricity_bill_may (a : ℝ) := 140 - a

-- The electricity bill increases by 20% in June.
def electricity_bill_june (a : ℝ) := (140 - a) * 1.2

-- Total electricity bills in June.
def total_electricity_june (a : ℝ) := (140 - a) + 0.2 * (140 - a)

-- Total water and electricity bills in June.
def total_water_electricity_june (a : ℝ) := 1.1 * a + 168 - 1.2 * a

-- Total water and electricity bills for May and June.
def total_water_electricity_may_june (a : ℝ) := a + (1.1 * a) + (140 - a) + ((140 - a) * 1.2)

-- When a = 40, the total water and electricity bills for May and June.
theorem total_bills_proof : ∀ a : ℝ, a = 40 → total_water_electricity_may_june a = 304 := 
by
  intros a ha
  rw [ha]
  sorry

end total_bills_proof_l2033_203334


namespace square_area_eq_1296_l2033_203360

theorem square_area_eq_1296 (x : ℝ) (side : ℝ) (h1 : side = 6 * x - 18) (h2 : side = 3 * x + 9) : side ^ 2 = 1296 := sorry

end square_area_eq_1296_l2033_203360


namespace relationship_between_abc_l2033_203306

-- Definitions based on the conditions
def a : ℕ := 3^44
def b : ℕ := 4^33
def c : ℕ := 5^22

-- The theorem to prove the relationship a > b > c
theorem relationship_between_abc : a > b ∧ b > c := by
  sorry

end relationship_between_abc_l2033_203306


namespace gain_percent_calculation_l2033_203327

noncomputable def gain_percent (C S : ℝ) : ℝ :=
  (S - C) / C * 100

theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 46 * S) : 
  gain_percent C S = 100 / 11.5 :=
by
  sorry

end gain_percent_calculation_l2033_203327


namespace cos_double_angle_l2033_203331

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l2033_203331


namespace least_pos_int_div_by_four_distinct_primes_l2033_203349

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l2033_203349


namespace quadratic_inequality_l2033_203369

-- Defining the quadratic expression
def quadratic_expr (a x : ℝ) : ℝ :=
  (a + 2) * x^2 + 2 * (a + 2) * x + 4

-- Statement to be proven
theorem quadratic_inequality {a : ℝ} :
  (∀ x : ℝ, quadratic_expr a x > 0) ↔ -2 ≤ a ∧ a < 2 :=
by
  sorry -- Proof omitted

end quadratic_inequality_l2033_203369


namespace percentage_water_in_puree_l2033_203366

/-- Given that tomato juice is 90% water and Heinz obtains 2.5 litres of tomato puree from 20 litres of tomato juice,
proves that the percentage of water in the tomato puree is 20%. -/
theorem percentage_water_in_puree (tj_volume : ℝ) (tj_water_content : ℝ) (tp_volume : ℝ) (tj_to_tp_ratio : ℝ) 
  (h1 : tj_water_content = 0.90) 
  (h2 : tj_volume = 20) 
  (h3 : tp_volume = 2.5) 
  (h4 : tj_to_tp_ratio = tj_volume / tp_volume) : 
  ((tp_volume - (1 - tj_water_content) * (tj_volume * (tp_volume / tj_volume))) / tp_volume) * 100 = 20 := 
sorry

end percentage_water_in_puree_l2033_203366


namespace find_f_of_power_function_l2033_203362

theorem find_f_of_power_function (a : ℝ) (alpha : ℝ) (f : ℝ → ℝ) 
  (h1 : 0 < a ∧ a ≠ 1) 
  (h2 : ∀ x, f x = x^alpha) 
  (h3 : ∀ x, a^(x-2) + 3 = f (2)): 
  f 2 = 4 := 
  sorry

end find_f_of_power_function_l2033_203362


namespace percentage_error_equals_l2033_203386

noncomputable def correct_fraction_calc : ℚ :=
  let num := (3/4 : ℚ) * 16 - (7/8 : ℚ) * 8
  let denom := (3/10 : ℚ) - (1/8 : ℚ)
  num / denom

noncomputable def incorrect_fraction_calc : ℚ :=
  let num := (3/4 : ℚ) * 16 - (7 / 8 : ℚ) * 8
  num * (3/5 : ℚ)

def percentage_error (correct incorrect : ℚ) : ℚ :=
  abs (correct - incorrect) / correct * 100

theorem percentage_error_equals :
  percentage_error correct_fraction_calc incorrect_fraction_calc = 89.47 :=
by
  sorry

end percentage_error_equals_l2033_203386


namespace math_proof_problem_l2033_203385

noncomputable def problem_statement : Prop :=
  let a_bound := 14
  let b_bound := 7
  let c_bound := 14
  let num_square_divisors := (a_bound / 2 + 1) * (b_bound / 2 + 1) * (c_bound / 2 + 1)
  let num_cube_divisors := (a_bound / 3 + 1) * (b_bound / 3 + 1) * (c_bound / 3 + 1)
  let num_sixth_power_divisors := (a_bound / 6 + 1) * (b_bound / 6 + 1) * (c_bound / 6 + 1)
  
  num_square_divisors + num_cube_divisors - num_sixth_power_divisors = 313

theorem math_proof_problem : problem_statement := by sorry

end math_proof_problem_l2033_203385


namespace snowman_volume_l2033_203330

theorem snowman_volume
  (r1 r2 r3 : ℝ)
  (volume : ℝ)
  (h1 : r1 = 1)
  (h2 : r2 = 4)
  (h3 : r3 = 6)
  (h_volume : volume = (4.0 / 3.0) * Real.pi * (r1 ^ 3 + r2 ^ 3 + r3 ^ 3)) :
  volume = (1124.0 / 3.0) * Real.pi :=
by
  sorry

end snowman_volume_l2033_203330


namespace min_ab_value_l2033_203310

theorem min_ab_value (a b : Real) (h_a : 1 < a) (h_b : 1 < b)
  (h_geom_seq : ∀ (x₁ x₂ x₃ : Real), x₁ = (1/4) * Real.log a → x₂ = 1/4 → x₃ = Real.log b →  x₂^2 = x₁ * x₃) : 
  a * b ≥ Real.exp 1 := by
  sorry

end min_ab_value_l2033_203310


namespace num_integer_pairs_satisfying_m_plus_n_eq_mn_l2033_203328

theorem num_integer_pairs_satisfying_m_plus_n_eq_mn : 
  ∃ (m n : ℤ), (m + n = m * n) ∧ ∀ (m n : ℤ), (m + n = m * n) → 
  (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2) :=
by
  sorry

end num_integer_pairs_satisfying_m_plus_n_eq_mn_l2033_203328


namespace intersection_of_A_and_B_l2033_203320

def setA : Set ℝ := {y | ∃ x : ℝ, y = 2 * x}
def setB : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2}

theorem intersection_of_A_and_B : setA ∩ setB = {y | y ≥ 0} :=
by
  sorry

end intersection_of_A_and_B_l2033_203320


namespace add_base_12_l2033_203326

def a_in_base_10 := 10
def b_in_base_10 := 11
def c_base := 12

theorem add_base_12 : 
  let a := 10
  let b := 11
  (3 * c_base ^ 2 + 12 * c_base + 5) + (2 * c_base ^ 2 + a * c_base + b) = 6 * c_base ^ 2 + 3 * c_base + 4 :=
by
  sorry

end add_base_12_l2033_203326


namespace batsman_average_after_12th_innings_l2033_203318

theorem batsman_average_after_12th_innings (A : ℕ) (total_runs_11 : ℕ) (total_runs_12 : ℕ ) : 
  total_runs_11 = 11 * A → 
  total_runs_12 = total_runs_11 + 55 → 
  (total_runs_12 / 12 = A + 1) → 
  (A + 1) = 44 := 
by
  intros h1 h2 h3
  sorry

end batsman_average_after_12th_innings_l2033_203318


namespace gcd_80_36_l2033_203332

theorem gcd_80_36 : Nat.gcd 80 36 = 4 := by
  -- Using the method of successive subtraction algorithm
  sorry

end gcd_80_36_l2033_203332


namespace problem1_l2033_203302

theorem problem1 (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x, 2 * |x + 3| ≥ m - 2 * |x + 7|) →
  (m ≤ 20) :=
by
  sorry

end problem1_l2033_203302


namespace trajectory_equation_line_slope_is_constant_l2033_203314

/-- Definitions for points A, B, and the moving point P -/ 
def pointA : ℝ × ℝ := (-2, 0)
def pointB : ℝ × ℝ := (2, 0)

/-- The condition that the product of the slopes is -3/4 -/
def slope_condition (P : ℝ × ℝ) : Prop :=
  let k_PA := P.2 / (P.1 + 2)
  let k_PB := P.2 / (P.1 - 2)
  k_PA * k_PB = -3 / 4

/-- The trajectory equation as a theorem to be proved -/
theorem trajectory_equation (P : ℝ × ℝ) (h : slope_condition P) : 
  P.2 ≠ 0 ∧ (P.1^2 / 4 + P.2^2 / 3 = 1) := 
sorry

/-- Additional conditions for the line l and points M, N -/ 
def line_l (k m : ℝ) (x : ℝ) : ℝ := k * x + m
def intersect_conditions (P M N : ℝ × ℝ) (k m : ℝ) : Prop :=
  (M.2 = line_l k m M.1) ∧ (N.2 = line_l k m N.1) ∧ 
  (P ≠ M ∧ P ≠ N) ∧ ((P.1 = 1) ∧ (P.2 = 3 / 2)) ∧ 
  (let k_PM := (M.2 - P.2) / (M.1 - P.1)
  let k_PN := (N.2 - P.2) / (N.1 - P.1)
  k_PM + k_PN = 0)

/-- The theorem to prove that the slope of line l is 1/2 -/
theorem line_slope_is_constant (P M N : ℝ × ℝ) (k m : ℝ) 
  (h1 : slope_condition P) 
  (h2 : intersect_conditions P M N k m) : 
  k = 1 / 2 := 
sorry

end trajectory_equation_line_slope_is_constant_l2033_203314


namespace heidi_zoe_paint_wall_l2033_203345

theorem heidi_zoe_paint_wall :
  let heidi_rate := (1 : ℚ) / 60
  let zoe_rate := (1 : ℚ) / 90
  let combined_rate := heidi_rate + zoe_rate
  let painted_fraction_15_minutes := combined_rate * 15
  painted_fraction_15_minutes = (5 : ℚ) / 12 :=
by
  sorry

end heidi_zoe_paint_wall_l2033_203345


namespace find_smallest_sphere_radius_squared_l2033_203382

noncomputable def smallest_sphere_radius_squared
  (AB CD AD BC : ℝ) (angle_ABC : ℝ) (radius_AC_squared : ℝ) : ℝ :=
if AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120 then radius_AC_squared else 0

theorem find_smallest_sphere_radius_squared
  (AB CD AD BC : ℝ) (angle_ABC : ℝ) (radius_AC_squared : ℝ) :
  (AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120) →
  radius_AC_squared = 49 :=
by
  intros h
  have h_ABCD : AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120 := h
  sorry -- The proof steps would be filled in here

end find_smallest_sphere_radius_squared_l2033_203382


namespace x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2_l2033_203375

theorem x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2 
  (x : ℤ) (p m n : ℕ) (hp : 0 < p) (hm : 0 < m) (hn : 0 < n) :
  (x^2 + x + 1) ∣ (x^(3 * p) + x^(3 * m + 1) + x^(3 * n + 2)) :=
by
  sorry

end x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2_l2033_203375


namespace range_of_a_l2033_203358

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + 1 < 0) ↔ a < 1 :=
by
  sorry

end range_of_a_l2033_203358


namespace intersection_P_Q_l2033_203396

def P (k : ℤ) (α : ℝ) : Prop := 2 * k * Real.pi ≤ α ∧ α ≤ (2 * k + 1) * Real.pi
def Q (α : ℝ) : Prop := -4 ≤ α ∧ α ≤ 4

theorem intersection_P_Q :
  (∃ k : ℤ, P k α) ∧ Q α ↔ (-4 ≤ α ∧ α ≤ -Real.pi) ∨ (0 ≤ α ∧ α ≤ Real.pi) :=
by
  sorry

end intersection_P_Q_l2033_203396


namespace bobby_gasoline_left_l2033_203324

theorem bobby_gasoline_left
  (initial_gasoline : ℕ) (supermarket_distance : ℕ) 
  (travel_distance : ℕ) (turn_back_distance : ℕ)
  (trip_fuel_efficiency : ℕ) : 
  initial_gasoline = 12 →
  supermarket_distance = 5 →
  travel_distance = 6 →
  turn_back_distance = 2 →
  trip_fuel_efficiency = 2 →
  ∃ remaining_gasoline,
    remaining_gasoline = initial_gasoline - 
    ((supermarket_distance * 2 + 
    turn_back_distance * 2 + 
    travel_distance) / trip_fuel_efficiency) ∧ 
    remaining_gasoline = 2 :=
by sorry

end bobby_gasoline_left_l2033_203324


namespace number_of_female_students_l2033_203323

theorem number_of_female_students 
  (average_all : ℝ)
  (num_males : ℝ) 
  (average_males : ℝ)
  (average_females : ℝ) 
  (h_avg_all : average_all = 88)
  (h_num_males : num_males = 15)
  (h_avg_males : average_males = 80)
  (h_avg_females : average_females = 94) :
  ∃ F : ℝ, 1200 + 94 * F = 88 * (15 + F) ∧ F = 20 :=
by
  use 20
  sorry

end number_of_female_students_l2033_203323


namespace max_value_trig_formula_l2033_203342

theorem max_value_trig_formula (x : ℝ) : ∃ (M : ℝ), M = 5 ∧ ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ M := 
sorry

end max_value_trig_formula_l2033_203342


namespace number_condition_l2033_203321

theorem number_condition (x : ℝ) (h : 45 - 3 * x^2 = 12) : x = Real.sqrt 11 ∨ x = -Real.sqrt 11 :=
sorry

end number_condition_l2033_203321


namespace set_relationship_l2033_203315

def set_M : Set ℚ := {x : ℚ | ∃ m : ℤ, x = m + 1/6}
def set_N : Set ℚ := {x : ℚ | ∃ n : ℤ, x = n/2 - 1/3}
def set_P : Set ℚ := {x : ℚ | ∃ p : ℤ, x = p/2 + 1/6}

theorem set_relationship : set_M ⊆ set_N ∧ set_N = set_P := by
  sorry

end set_relationship_l2033_203315


namespace find_principal_l2033_203359

-- Define the conditions
def simple_interest (P R T : ℕ) : ℕ :=
  (P * R * T) / 100

-- Given values
def SI : ℕ := 750
def R : ℕ := 6
def T : ℕ := 5

-- Proof statement
theorem find_principal : ∃ P : ℕ, simple_interest P R T = SI ∧ P = 2500 := by
  aesop

end find_principal_l2033_203359


namespace nancy_hourly_wage_l2033_203363

-- Definitions based on conditions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def scholarship_amount : ℕ := 3000
def loan_amount : ℕ := 2 * scholarship_amount
def nancy_contributions : ℕ := parents_contribution + scholarship_amount + loan_amount
def remaining_tuition : ℕ := tuition_per_semester - nancy_contributions
def total_working_hours : ℕ := 200

-- Theorem to prove based on the formulated problem
theorem nancy_hourly_wage :
  (remaining_tuition / total_working_hours) = 10 :=
by
  sorry

end nancy_hourly_wage_l2033_203363


namespace relay_race_total_distance_l2033_203393

theorem relay_race_total_distance
  (Sadie_speed : ℝ) (Sadie_time : ℝ) (Ariana_speed : ℝ) (Ariana_time : ℝ) (Sarah_speed : ℝ) (total_race_time : ℝ)
  (h1 : Sadie_speed = 3) (h2 : Sadie_time = 2)
  (h3 : Ariana_speed = 6) (h4 : Ariana_time = 0.5)
  (h5 : Sarah_speed = 4) (h6 : total_race_time = 4.5) :
  (Sadie_speed * Sadie_time + Ariana_speed * Ariana_time + Sarah_speed * (total_race_time - (Sadie_time + Ariana_time))) = 17 :=
by
  sorry

end relay_race_total_distance_l2033_203393


namespace gray_eyed_brunettes_l2033_203373

-- Given conditions
def total_students : ℕ := 60
def brunettes : ℕ := 35
def green_eyed_blondes : ℕ := 20
def gray_eyed_total : ℕ := 25

-- Conclude that the number of gray-eyed brunettes is 20
theorem gray_eyed_brunettes :
    (gray_eyed_total - (total_students - brunettes - green_eyed_blondes)) = 20 := by
    sorry

end gray_eyed_brunettes_l2033_203373


namespace divisibility_by_seven_l2033_203370

theorem divisibility_by_seven (n : ℤ) (b : ℤ) (a : ℤ) (h : n = 10 * a + b) 
  (hb : 0 ≤ b) (hb9 : b ≤ 9) (ha : 0 ≤ a) (d : ℤ) (hd : d = a - 2 * b) :
  (2 * n + d) % 7 = 0 ↔ n % 7 = 0 := 
by
  sorry

end divisibility_by_seven_l2033_203370


namespace profit_in_may_highest_monthly_profit_and_max_value_l2033_203316

def f (x : ℕ) : ℕ :=
  if 1 ≤ x ∧ x ≤ 6 then 12 * x + 28 else 200 - 14 * x

theorem profit_in_may :
  f 5 = 88 :=
by sorry

theorem highest_monthly_profit_and_max_value :
  ∃ x, 1 ≤ x ∧ x ≤ 12 ∧ f x = 102 :=
by sorry

end profit_in_may_highest_monthly_profit_and_max_value_l2033_203316


namespace selling_price_of_cycle_l2033_203364

theorem selling_price_of_cycle (cost_price : ℕ) (loss_percent : ℕ) (selling_price : ℕ) :
  cost_price = 1400 → loss_percent = 25 → selling_price = 1050 := by
  sorry

end selling_price_of_cycle_l2033_203364


namespace find_n_l2033_203335

-- Definitions and conditions
def painted_total_faces (n : ℕ) : ℕ := 6 * n^2
def total_faces_of_unit_cubes (n : ℕ) : ℕ := 6 * n^3
def fraction_of_red_faces (n : ℕ) : ℚ := (painted_total_faces n : ℚ) / (total_faces_of_unit_cubes n : ℚ)

-- Statement to be proven
theorem find_n (n : ℕ) (h : fraction_of_red_faces n = 1 / 4) : n = 4 :=
by
  sorry

end find_n_l2033_203335


namespace potatoes_left_l2033_203392

def p_initial : ℕ := 8
def p_eaten : ℕ := 3
def p_left : ℕ := p_initial - p_eaten

theorem potatoes_left : p_left = 5 := by
  sorry

end potatoes_left_l2033_203392


namespace triangle_is_isosceles_l2033_203399

variable {α β γ : ℝ} (quadrilateral_angles : List ℝ)

-- Conditions from the problem
axiom triangle_angle_sum : α + β + γ = 180
axiom quadrilateral_angle_sum : quadrilateral_angles.sum = 360
axiom quadrilateral_angle_conditions : ∀ (a b : ℝ), a ∈ [α, β, γ] → b ∈ [α, β, γ] → a ≠ b → (a + b ∈ quadrilateral_angles)

-- Proof statement
theorem triangle_is_isosceles : (α = β) ∨ (β = γ) ∨ (γ = α) := 
  sorry

end triangle_is_isosceles_l2033_203399


namespace zero_polynomial_is_solution_l2033_203303

noncomputable def polynomial_zero (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (p.eval x)^2 + (p.eval (1/x))^2 = (p.eval (x^2)) * (p.eval (1/(x^2))) → p = 0

theorem zero_polynomial_is_solution : ∀ p : Polynomial ℝ, (∀ x : ℝ, x ≠ 0 → (p.eval x)^2 + (p.eval (1/x))^2 = (p.eval (x^2)) * (p.eval (1/(x^2)))) → p = 0 :=
by
  sorry

end zero_polynomial_is_solution_l2033_203303


namespace arrange_x_y_z_l2033_203336

theorem arrange_x_y_z (x : ℝ) (hx : 0.9 < x ∧ x < 1) :
  let y := x^(1/x)
  let z := x^y
  x < z ∧ z < y :=
by
  let y := x^(1/x)
  let z := x^y
  have : 0.9 < x ∧ x < 1 := hx
  sorry

end arrange_x_y_z_l2033_203336


namespace whale_consumption_third_hour_l2033_203383

theorem whale_consumption_third_hour (x : ℕ) :
  (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 450) → ((x + 6) = 90) :=
by
  intro h
  sorry

end whale_consumption_third_hour_l2033_203383


namespace big_al_bananas_l2033_203322

theorem big_al_bananas (total_bananas : ℕ) (a : ℕ)
  (h : total_bananas = 150)
  (h1 : a + (a + 7) + (a + 14) + (a + 21) + (a + 28) = total_bananas) :
  a + 14 = 30 :=
by
  -- Using the given conditions to prove the statement
  sorry

end big_al_bananas_l2033_203322


namespace sum_3x_4y_l2033_203391

theorem sum_3x_4y (x y N : ℝ) (H1 : 3 * x + 4 * y = N) (H2 : 6 * x - 4 * y = 12) (H3 : x * y = 72) : 3 * x + 4 * y = 60 := 
sorry

end sum_3x_4y_l2033_203391


namespace base_b_square_l2033_203337

-- Given that 144 in base b can be written as b^2 + 4b + 4 in base 10,
-- prove that it is a perfect square if and only if b > 4

theorem base_b_square (b : ℕ) (h : b > 4) : ∃ k : ℕ, b^2 + 4 * b + 4 = k^2 := by
  sorry

end base_b_square_l2033_203337


namespace set_intersection_l2033_203308

open Set Real

theorem set_intersection (A : Set ℝ) (hA : A = {-1, 0, 1}) (B : Set ℝ) (hB : B = {y | ∃ x ∈ A, y = cos (π * x)}) :
  A ∩ B = {-1, 1} :=
by
  rw [hA, hB]
  -- remaining proof should go here
  sorry

end set_intersection_l2033_203308


namespace amount_r_has_l2033_203344

variable (p q r : ℕ)
variable (total_amount : ℕ)
variable (two_thirdsOf_pq : ℕ)

def total_money : Prop := (p + q + r = 4000)
def two_thirds_of_pq : Prop := (r = 2 * (p + q) / 3)

theorem amount_r_has : total_money p q r → two_thirds_of_pq p q r → r = 1600 := by
  intro h1 h2
  sorry

end amount_r_has_l2033_203344


namespace only_one_P_Q_l2033_203350

def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def Q (a : ℝ) : Prop := ∀ x : ℝ, x^2 - x + a = 0

theorem only_one_P_Q (a : ℝ) :
  (P a ∧ ¬ Q a) ∨ (Q a ∧ ¬ P a) ↔
  (a < 0) ∨ (1/4 < a ∧ a < 4) :=
sorry

end only_one_P_Q_l2033_203350


namespace rancher_cattle_count_l2033_203371

theorem rancher_cattle_count
  (truck_capacity : ℕ)
  (distance_to_higher_ground : ℕ)
  (truck_speed : ℕ)
  (total_transport_time : ℕ)
  (h1 : truck_capacity = 20)
  (h2 : distance_to_higher_ground = 60)
  (h3 : truck_speed = 60)
  (h4 : total_transport_time = 40):
  ∃ (number_of_cattle : ℕ), number_of_cattle = 400 :=
by {
  sorry
}

end rancher_cattle_count_l2033_203371


namespace gel_pen_price_ratio_l2033_203340

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l2033_203340


namespace overall_average_is_63_point_4_l2033_203353

theorem overall_average_is_63_point_4 : 
  ∃ (n total_marks : ℕ) (avg_marks : ℚ), 
  n = 50 ∧ 
  (∃ (marks_group1 marks_group2 marks_group3 marks_remaining : ℕ), 
    marks_group1 = 6 * 95 ∧
    marks_group2 = 4 * 0 ∧
    marks_group3 = 10 * 80 ∧
    marks_remaining = (n - 20) * 60 ∧
    total_marks = marks_group1 + marks_group2 + marks_group3 + marks_remaining) ∧ 
  avg_marks = total_marks / n ∧ 
  avg_marks = 63.4 := 
by 
  sorry

end overall_average_is_63_point_4_l2033_203353


namespace cube_edge_length_l2033_203374

theorem cube_edge_length (sum_of_edges : ℕ) (num_edges : ℕ) (h : sum_of_edges = 144) (num_edges_h : num_edges = 12) :
  sum_of_edges / num_edges = 12 :=
by
  -- The proof is skipped.
  sorry

end cube_edge_length_l2033_203374


namespace value_of_square_l2033_203379

variable (x y : ℝ)

theorem value_of_square (h1 : x * (x + y) = 30) (h2 : y * (x + y) = 60) :
  (x + y) ^ 2 = 90 := sorry

end value_of_square_l2033_203379


namespace value_of_E_l2033_203312

variable {D E F : ℕ}

theorem value_of_E (h1 : D + E + F = 16) (h2 : F + D + 1 = 16) (h3 : E - 1 = D) : E = 1 :=
sorry

end value_of_E_l2033_203312


namespace commute_time_abs_diff_l2033_203309

theorem commute_time_abs_diff (x y : ℝ) 
  (h1 : (x + y + 10 + 11 + 9)/5 = 10) 
  (h2 : (1/5 : ℝ) * ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) = 2) : 
  |x - y| = 4 :=
by
  sorry

end commute_time_abs_diff_l2033_203309


namespace frosting_cupcakes_l2033_203368

noncomputable def Cagney_rate := 1 / 20 -- cupcakes per second
noncomputable def Lacey_rate := 1 / 30 -- cupcakes per second
noncomputable def Hardy_rate := 1 / 40 -- cupcakes per second

noncomputable def combined_rate := Cagney_rate + Lacey_rate + Hardy_rate
noncomputable def total_time := 600 -- seconds (10 minutes)

theorem frosting_cupcakes :
  total_time * combined_rate = 65 := 
by 
  sorry

end frosting_cupcakes_l2033_203368


namespace investment_problem_l2033_203376

theorem investment_problem :
  ∃ (S G : ℝ), S + G = 10000 ∧ 0.06 * G = 0.05 * S + 160 ∧ S = 4000 :=
by
  sorry

end investment_problem_l2033_203376


namespace speed_difference_l2033_203343

theorem speed_difference (distance : ℝ) (time_heavy : ℝ) (time_no_traffic : ℝ) (d : distance = 200) (th : time_heavy = 5) (tn : time_no_traffic = 4) :
  (distance / time_no_traffic) - (distance / time_heavy) = 10 :=
by
  -- Proof goes here
  sorry

end speed_difference_l2033_203343


namespace double_acute_angle_is_positive_and_less_than_180_l2033_203390

variable (α : ℝ) (h : 0 < α ∧ α < π / 2)

theorem double_acute_angle_is_positive_and_less_than_180 :
  0 < 2 * α ∧ 2 * α < π :=
by
  sorry

end double_acute_angle_is_positive_and_less_than_180_l2033_203390


namespace rhombus_area_l2033_203389

theorem rhombus_area (R1 R2 : ℝ) (x y : ℝ)
  (hR1 : R1 = 15) (hR2 : R2 = 30)
  (hx : x = 15) (hy : y = 2 * x):
  (x * y / 2 = 225) :=
by 
  -- Lean 4 proof not required here
  sorry

end rhombus_area_l2033_203389


namespace ram_leela_piggy_bank_l2033_203300

theorem ram_leela_piggy_bank (final_amount future_deposits weeks: ℕ) 
  (initial_deposit common_diff: ℕ) (total_deposits : ℕ) 
  (h_total : total_deposits = (weeks * (initial_deposit + (initial_deposit + (weeks - 1) * common_diff)) / 2)) 
  (h_final : final_amount = 1478) 
  (h_weeks : weeks = 52) 
  (h_future_deposits : future_deposits = total_deposits) 
  (h_initial_deposit : initial_deposit = 1) 
  (h_common_diff : common_diff = 1) 
  : final_amount - future_deposits = 100 :=
sorry

end ram_leela_piggy_bank_l2033_203300


namespace diameter_increase_l2033_203357

theorem diameter_increase (π : ℝ) (D : ℝ) (A A' D' : ℝ)
  (hA : A = (π / 4) * D^2)
  (hA' : A' = 4 * A)
  (hA'_def : A' = (π / 4) * D'^2) :
  D' = 2 * D :=
by
  sorry

end diameter_increase_l2033_203357


namespace probability_heart_then_king_of_clubs_l2033_203351

theorem probability_heart_then_king_of_clubs : 
  let deck := 52
  let hearts := 13
  let remaining_cards := deck - 1
  let king_of_clubs := 1
  let first_card_heart_probability := (hearts : ℝ) / deck
  let second_card_king_of_clubs_probability := (king_of_clubs : ℝ) / remaining_cards
  first_card_heart_probability * second_card_king_of_clubs_probability = 1 / 204 :=
by
  sorry

end probability_heart_then_king_of_clubs_l2033_203351


namespace complex_number_solution_l2033_203367

theorem complex_number_solution
  (z : ℂ)
  (h : i * (z - 1) = 1 + i) :
  z = 2 - i :=
sorry

end complex_number_solution_l2033_203367


namespace gcd_a_b_l2033_203384

-- Define a and b
def a : ℕ := 333333
def b : ℕ := 9999999

-- Prove that gcd(a, b) = 3
theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l2033_203384


namespace teams_worked_together_days_l2033_203319

noncomputable def first_team_rate : ℝ := 1 / 12
noncomputable def second_team_rate : ℝ := 1 / 9
noncomputable def first_team_days : ℕ := 5
noncomputable def total_work : ℝ := 1
noncomputable def work_first_team_alone := first_team_rate * first_team_days

theorem teams_worked_together_days (x : ℝ) : work_first_team_alone + (first_team_rate + second_team_rate) * x = total_work → x = 3 := 
by
  sorry

end teams_worked_together_days_l2033_203319


namespace max_non_div_by_3_l2033_203317

theorem max_non_div_by_3 (s : Finset ℕ) (h_len : s.card = 7) (h_prod : 3 ∣ s.prod id) : 
  ∃ n, n ≤ 6 ∧ ∀ x ∈ s, ¬ (3 ∣ x) → n = 6 :=
sorry

end max_non_div_by_3_l2033_203317


namespace probability_of_A_l2033_203355

variable (A B : Prop)
variable (P : Prop → ℝ)

-- Given conditions
variable (h1 : P (A ∧ B) = 0.72)
variable (h2 : P (A ∧ ¬B) = 0.18)

theorem probability_of_A: P A = 0.90 := sorry

end probability_of_A_l2033_203355


namespace find_value_l2033_203377

theorem find_value (x : ℝ) (h : 0.20 * x = 80) : 0.40 * x = 160 := 
by
  sorry

end find_value_l2033_203377


namespace find_pairs_l2033_203387

def sequence_a : Nat → Int
| 0 => 0
| 1 => 0
| n+2 => 2 * sequence_a (n+1) - sequence_a n + 2

def sequence_b : Nat → Int
| 0 => 8
| 1 => 8
| n+2 => 2 * sequence_b (n+1) - sequence_b n

theorem find_pairs :
  (sequence_a 1992 = 31872 ∧ sequence_b 1992 = 31880) ∨
  (sequence_a 1992 = -31872 ∧ sequence_b 1992 = -31864) :=
sorry

end find_pairs_l2033_203387


namespace correct_multiplication_value_l2033_203304

theorem correct_multiplication_value (N : ℝ) (x : ℝ) : 
  (0.9333333333333333 = (N * x - N / 5) / (N * x)) → 
  x = 3 := 
by 
  sorry

end correct_multiplication_value_l2033_203304


namespace min_value_of_quadratic_form_l2033_203398

theorem min_value_of_quadratic_form (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : x^2 + 2 * y^2 + 3 * z^2 ≥ 1/3 :=
sorry

end min_value_of_quadratic_form_l2033_203398
