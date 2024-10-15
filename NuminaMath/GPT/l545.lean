import Mathlib

namespace NUMINAMATH_GPT_application_methods_count_l545_54515

theorem application_methods_count :
  let S := 5; -- number of students
  let U := 3; -- number of universities
  let unrestricted := U^S; -- unrestricted distribution
  let restricted_one_university_empty := (U - 1)^S * U; -- one university empty
  let restricted_two_universities_empty := 0; -- invalid scenario
  let valid_methods := unrestricted - restricted_one_university_empty - restricted_two_universities_empty;
  valid_methods - U = 144 :=
by
  let S := 5
  let U := 3
  let unrestricted := U^S
  let restricted_one_university_empty := (U - 1)^S * U
  let restricted_two_universities_empty := 0
  let valid_methods := unrestricted - restricted_one_university_empty - restricted_two_universities_empty
  have : valid_methods - U = 144 := by sorry
  exact this

end NUMINAMATH_GPT_application_methods_count_l545_54515


namespace NUMINAMATH_GPT_mike_ride_distance_l545_54598

-- Definitions from conditions
def mike_cost (m : ℕ) : ℝ := 2.50 + 0.25 * m
def annie_cost : ℝ := 2.50 + 5.00 + 0.25 * 16

-- Theorem to prove
theorem mike_ride_distance (m : ℕ) (h : mike_cost m = annie_cost) : m = 36 := by
  sorry

end NUMINAMATH_GPT_mike_ride_distance_l545_54598


namespace NUMINAMATH_GPT_total_pounds_of_peppers_l545_54546

def green_peppers : ℝ := 2.8333333333333335
def red_peppers : ℝ := 2.8333333333333335
def total_peppers : ℝ := 5.666666666666667

theorem total_pounds_of_peppers :
  green_peppers + red_peppers = total_peppers :=
by
  -- sorry: Proof is omitted
  sorry

end NUMINAMATH_GPT_total_pounds_of_peppers_l545_54546


namespace NUMINAMATH_GPT_exists_fraction_bound_infinite_no_fraction_bound_l545_54522

-- Problem 1: Statement 1
theorem exists_fraction_bound (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℤ), 0 < b ∧ (b : ℝ) ≤ Real.sqrt n + 1 ∧ Real.sqrt n ≤ (a : ℝ) / b ∧ (a : ℝ) / b ≤ Real.sqrt (n + 1) :=
sorry

-- Problem 2: Statement 2
theorem infinite_no_fraction_bound :
  ∃ᶠ n : ℕ in Filter.atTop, ¬ ∃ (a b : ℤ), 0 < b ∧ (b : ℝ) ≤ Real.sqrt n ∧ Real.sqrt n ≤ (a : ℝ) / b ∧ (a : ℝ) / b ≤ Real.sqrt (n + 1) :=
sorry

end NUMINAMATH_GPT_exists_fraction_bound_infinite_no_fraction_bound_l545_54522


namespace NUMINAMATH_GPT_sqrt_defined_iff_le_l545_54575

theorem sqrt_defined_iff_le (x : ℝ) : (∃ y : ℝ, y^2 = 4 - x) ↔ (x ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_defined_iff_le_l545_54575


namespace NUMINAMATH_GPT_least_value_x_l545_54564

theorem least_value_x (x : ℕ) (p q : ℕ) (h_prime_p : Nat.Prime p) (h_prime_q : Nat.Prime q)
  (h_distinct : p ≠ q) (h_diff : q - p = 3) (h_even_prim : x / (11 * p * q) = 2) : x = 770 := by
  sorry

end NUMINAMATH_GPT_least_value_x_l545_54564


namespace NUMINAMATH_GPT_BP_PA_ratio_l545_54506

section

variable (A B C P : Type)
variable {AC BC PA PB BP : ℕ}

-- Conditions:
-- 1. In triangle ABC, the ratio AC:CB = 2:5.
axiom AC_CB_ratio : 2 * BC = 5 * AC

-- 2. The bisector of the exterior angle at C intersects the extension of BA at P,
--    such that B is between P and A.
axiom Angle_Bisector_Theorem : PA * BC = PB * AC

theorem BP_PA_ratio (h1 : 2 * BC = 5 * AC) (h2 : PA * BC = PB * AC) :
  BP * PA = 5 * PA := sorry

end

end NUMINAMATH_GPT_BP_PA_ratio_l545_54506


namespace NUMINAMATH_GPT_interval_k_is_40_l545_54563

def total_students := 1200
def sample_size := 30

theorem interval_k_is_40 : (total_students / sample_size) = 40 :=
by
  sorry

end NUMINAMATH_GPT_interval_k_is_40_l545_54563


namespace NUMINAMATH_GPT_integer_values_b_for_three_integer_solutions_l545_54502

theorem integer_values_b_for_three_integer_solutions (b : ℤ) :
  ¬ ∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1^2 + b * x1 + 5 ≤ 0) ∧
                     (x2^2 + b * x2 + 5 ≤ 0) ∧ (x3^2 + b * x3 + 5 ≤ 0) ∧
                     (∀ x : ℤ, x1 < x ∧ x < x3 → x^2 + b * x + 5 > 0) :=
by
  sorry

end NUMINAMATH_GPT_integer_values_b_for_three_integer_solutions_l545_54502


namespace NUMINAMATH_GPT_find_percentage_of_number_l545_54567

theorem find_percentage_of_number (P : ℝ) (N : ℝ) (h1 : P * N = (4 / 5) * N - 21) (h2 : N = 140) : P * 100 = 65 := 
by 
  sorry

end NUMINAMATH_GPT_find_percentage_of_number_l545_54567


namespace NUMINAMATH_GPT_region_area_l545_54539

noncomputable def area_of_region := 
  let a := 0
  let b := Real.sqrt 2 / 2
  ∫ x in a..b, (Real.arccos x) - (Real.arcsin x)

theorem region_area : area_of_region = 2 - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_region_area_l545_54539


namespace NUMINAMATH_GPT_sin_330_eq_negative_half_l545_54540

theorem sin_330_eq_negative_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_330_eq_negative_half_l545_54540


namespace NUMINAMATH_GPT_max_geq_four_ninths_sum_min_leq_quarter_sum_l545_54512

theorem max_geq_four_ninths_sum (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_discriminant : b^2 >= 4*a*c) :
  max a (max b c) >= 4 / 9 * (a + b + c) :=
by 
  sorry

theorem min_leq_quarter_sum (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_discriminant : b^2 >= 4*a*c) :
  min a (min b c) <= 1 / 4 * (a + b + c) :=
by 
  sorry

end NUMINAMATH_GPT_max_geq_four_ninths_sum_min_leq_quarter_sum_l545_54512


namespace NUMINAMATH_GPT_circle_area_difference_l545_54556

/-- 
Prove that the area of the circle with radius r1 = 30 inches is 675π square inches greater than 
the area of the circle with radius r2 = 15 inches.
-/
theorem circle_area_difference (r1 r2 : ℝ) (h1 : r1 = 30) (h2 : r2 = 15) :
  π * r1^2 - π * r2^2 = 675 * π := 
by {
  -- Placeholders to indicate where the proof would go
  sorry 
}

end NUMINAMATH_GPT_circle_area_difference_l545_54556


namespace NUMINAMATH_GPT_segment_length_cd_l545_54587

theorem segment_length_cd
  (AB : ℝ)
  (M : ℝ)
  (N : ℝ)
  (P : ℝ)
  (C : ℝ)
  (D : ℝ)
  (h₁ : AB = 60)
  (h₂ : N = M / 2)
  (h₃ : P = (AB - M) / 2)
  (h₄ : C = N / 2)
  (h₅ : D = P / 2) :
  |C - D| = 15 :=
by
  sorry

end NUMINAMATH_GPT_segment_length_cd_l545_54587


namespace NUMINAMATH_GPT_usamo_2003_q3_l545_54573

open Real

theorem usamo_2003_q3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ( (2 * a + b + c) ^ 2 / (2 * a ^ 2 + (b + c) ^ 2)
  + (2 * b + a + c) ^ 2 / (2 * b ^ 2 + (c + a) ^ 2)
  + (2 * c + a + b) ^ 2 / (2 * c ^ 2 + (a + b) ^ 2) ) ≤ 8 := 
sorry

end NUMINAMATH_GPT_usamo_2003_q3_l545_54573


namespace NUMINAMATH_GPT_find_a_b_c_sum_l545_54521

theorem find_a_b_c_sum (a b c : ℤ)
  (h_gcd : gcd (x ^ 2 + a * x + b) (x ^ 2 + b * x + c) = x + 1)
  (h_lcm : lcm (x ^ 2 + a * x + b) (x ^ 2 + b * x + c) = x ^ 3 - 4 * x ^ 2 + x + 6) :
  a + b + c = -6 := 
sorry

end NUMINAMATH_GPT_find_a_b_c_sum_l545_54521


namespace NUMINAMATH_GPT_swapped_coefficients_have_roots_l545_54591

theorem swapped_coefficients_have_roots 
  (a b c p q r : ℝ)
  (h1 : ∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0))
  (h2 : ∀ x : ℝ, ¬ (p * x^2 + q * x + r = 0))
  (h3 : b^2 < 4 * p * c)
  (h4 : q^2 < 4 * a * r) :
  ∃ x : ℝ, a * x^2 + q * x + c = 0 ∧ ∃ y : ℝ, p * y^2 + b * y + r = 0 :=
by
  sorry

end NUMINAMATH_GPT_swapped_coefficients_have_roots_l545_54591


namespace NUMINAMATH_GPT_log_identity_l545_54503

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_identity :
    2 * log_base_10 2 + log_base_10 (5 / 8) - log_base_10 25 = -1 :=
by 
  sorry

end NUMINAMATH_GPT_log_identity_l545_54503


namespace NUMINAMATH_GPT_find_number_l545_54537

theorem find_number (x : ℝ) : 8050 * x = 80.5 → x = 0.01 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l545_54537


namespace NUMINAMATH_GPT_desired_percentage_alcohol_l545_54584

noncomputable def original_volume : ℝ := 6
noncomputable def original_percentage : ℝ := 0.40
noncomputable def added_alcohol : ℝ := 1.2
noncomputable def final_solution_volume : ℝ := original_volume + added_alcohol
noncomputable def final_alcohol_volume : ℝ := (original_percentage * original_volume) + added_alcohol
noncomputable def desired_percentage : ℝ := (final_alcohol_volume / final_solution_volume) * 100

theorem desired_percentage_alcohol :
  desired_percentage = 50 := by
  sorry

end NUMINAMATH_GPT_desired_percentage_alcohol_l545_54584


namespace NUMINAMATH_GPT_determine_b_for_inverse_function_l545_54581

theorem determine_b_for_inverse_function (b : ℝ) :
  (∀ x, (2 - 3 * (1 / (2 * x + b))) / (3 * (1 / (2 * x + b))) = x) ↔ b = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_determine_b_for_inverse_function_l545_54581


namespace NUMINAMATH_GPT_abc_inequality_l545_54552

theorem abc_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_cond : a * b + b * c + c * a = 1) :
  (a + b + c) ≥ Real.sqrt 3 ∧ (a + b + c = Real.sqrt 3 ↔ a = b ∧ b = c ∧ c = Real.sqrt 1 / Real.sqrt 3) :=
by sorry

end NUMINAMATH_GPT_abc_inequality_l545_54552


namespace NUMINAMATH_GPT_alicia_average_speed_correct_l545_54569

/-
Alicia drove 320 miles in 6 hours.
Alicia drove another 420 miles in 7 hours.
Prove Alicia's average speed for the entire journey is 56.92 miles per hour.
-/

def alicia_total_distance : ℕ := 320 + 420
def alicia_total_time : ℕ := 6 + 7
def alicia_average_speed : ℚ := alicia_total_distance / alicia_total_time

theorem alicia_average_speed_correct : alicia_average_speed = 56.92 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_alicia_average_speed_correct_l545_54569


namespace NUMINAMATH_GPT_car_speed_l545_54585

/-- 
If a tire rotates at 400 revolutions per minute, and the circumference of the tire is 6 meters, 
the speed of the car is 144 km/h.
-/
theorem car_speed (rev_per_min : ℕ) (circumference : ℝ) (speed : ℝ) :
  rev_per_min = 400 → circumference = 6 → speed = 144 :=
by
  intro h_rev h_circ
  sorry

end NUMINAMATH_GPT_car_speed_l545_54585


namespace NUMINAMATH_GPT_problem_solution_l545_54538

def lean_problem (a : ℝ) : Prop :=
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * a - 1)^x₁ > (2 * a - 1)^x₂) →
  a > 1 / 2 ∧ a ≤ 2 / 3

theorem problem_solution (a : ℝ) : lean_problem a :=
  sorry -- Proof is to be filled in

end NUMINAMATH_GPT_problem_solution_l545_54538


namespace NUMINAMATH_GPT_price_of_individual_rose_l545_54551

-- Definitions based on conditions

def price_of_dozen := 36  -- one dozen roses cost $36
def price_of_two_dozen := 50 -- two dozen roses cost $50
def total_money := 680 -- total available money
def total_roses := 317 -- total number of roses that can be purchased

-- Define the question as a theorem
theorem price_of_individual_rose : 
  ∃ (x : ℕ), (12 * (total_money / price_of_two_dozen) + 
              (total_money % price_of_two_dozen) / price_of_dozen * 12 + 
              (total_money % price_of_two_dozen % price_of_dozen) / x = total_roses) ∧ (x = 6) :=
by
  sorry

end NUMINAMATH_GPT_price_of_individual_rose_l545_54551


namespace NUMINAMATH_GPT_only_n_eq_1_solution_l545_54561

theorem only_n_eq_1_solution (n : ℕ) (h : n > 0): 
  (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_only_n_eq_1_solution_l545_54561


namespace NUMINAMATH_GPT_total_amount_paid_is_correct_l545_54531

def rate_per_kg_grapes := 98
def quantity_grapes := 15
def rate_per_kg_mangoes := 120
def quantity_mangoes := 8
def rate_per_kg_pineapples := 75
def quantity_pineapples := 5
def rate_per_kg_oranges := 60
def quantity_oranges := 10

def cost_grapes := rate_per_kg_grapes * quantity_grapes
def cost_mangoes := rate_per_kg_mangoes * quantity_mangoes
def cost_pineapples := rate_per_kg_pineapples * quantity_pineapples
def cost_oranges := rate_per_kg_oranges * quantity_oranges

def total_amount_paid := cost_grapes + cost_mangoes + cost_pineapples + cost_oranges

theorem total_amount_paid_is_correct : total_amount_paid = 3405 := by
  sorry

end NUMINAMATH_GPT_total_amount_paid_is_correct_l545_54531


namespace NUMINAMATH_GPT_fraction_of_A_eq_l545_54595

noncomputable def fraction_A (A B C T : ℕ) : ℚ :=
  A / (T - A)

theorem fraction_of_A_eq :
  ∃ (A B C T : ℕ), T = 360 ∧ A = B + 10 ∧ B = 2 * (A + C) / 7 ∧ T = A + B + C ∧ fraction_A A B C T = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_A_eq_l545_54595


namespace NUMINAMATH_GPT_find_value_l545_54544

theorem find_value (number remainder certain_value : ℕ) (h1 : number = 26)
  (h2 : certain_value / 2 = remainder) 
  (h3 : remainder = ((number + 20) * 2 / 2) - 2) :
  certain_value = 88 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l545_54544


namespace NUMINAMATH_GPT_polynomial_coefficients_l545_54548

theorem polynomial_coefficients (a_0 a_1 a_2 a_3 a_4 : ℤ) :
  (∀ x : ℤ, (x + 2)^5 = (x + 1)^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_0 = 31 ∧ a_1 = 75 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coefficients_l545_54548


namespace NUMINAMATH_GPT_tray_height_l545_54555

-- Declare the main theorem with necessary given conditions.
theorem tray_height (a b c : ℝ) (side_length : ℝ) (cut_distance : ℝ) (angle : ℝ) : 
  (side_length = 150) →
  (cut_distance = Real.sqrt 50) →
  (angle = 45) →
  a^2 + b^2 = c^2 → -- Condition from Pythagorean theorem
  a = side_length * Real.sqrt 2 / 2 - cut_distance → -- Calculation for half diagonal minus cut distance
  b = (side_length * Real.sqrt 2 / 2 - cut_distance) / 2 → -- Perpendicular from R to the side
  side_length = 150 → -- Ensure consistency of side length
  b^2 + c^2 = side_length^2 → -- Ensure we use another Pythagorean relation
  c = Real.sqrt 7350 → -- Derived c value
  c = Real.sqrt 1470 := -- Simplified form of c.
  sorry

end NUMINAMATH_GPT_tray_height_l545_54555


namespace NUMINAMATH_GPT_range_of_a_l545_54594

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ ((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0)) →
  (-2:ℝ) ≤ a ∧ a < (6 / 5:ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l545_54594


namespace NUMINAMATH_GPT_min_quotient_l545_54514

theorem min_quotient {a b : ℕ} (h₁ : 100 ≤ a) (h₂ : a ≤ 300) (h₃ : 400 ≤ b) (h₄ : b ≤ 800) (h₅ : a + b ≤ 950) : a / b = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_min_quotient_l545_54514


namespace NUMINAMATH_GPT_find_number_l545_54599

theorem find_number (x : ℤ) (h : 35 - 3 * x = 14) : x = 7 :=
by {
  sorry -- This is where the proof would go.
}

end NUMINAMATH_GPT_find_number_l545_54599


namespace NUMINAMATH_GPT_compound_propositions_l545_54532

def divides (a b : Nat) : Prop := ∃ k : Nat, b = k * a

-- Define the propositions p and q
def p : Prop := divides 6 12
def q : Prop := divides 6 24

-- Prove the compound propositions
theorem compound_propositions :
  (p ∨ q) ∧ (p ∧ q) ∧ ¬¬p :=
by
  -- We are proving three statements:
  -- 1. "p or q" is true.
  -- 2. "p and q" is true.
  -- 3. "not p" is false (which is equivalent to "¬¬p" being true).
  -- The actual proof will be constructed here.
  sorry

end NUMINAMATH_GPT_compound_propositions_l545_54532


namespace NUMINAMATH_GPT_findQuadraticFunctionAndVertex_l545_54570

noncomputable section

def quadraticFunction (x : ℝ) (b c : ℝ) : ℝ :=
  (1 / 2) * x^2 + b * x + c

theorem findQuadraticFunctionAndVertex :
  (∃ b c : ℝ, quadraticFunction 0 b c = -1 ∧ quadraticFunction 2 b c = -3) →
  (quadraticFunction x (-2) (-1) = (1 / 2) * x^2 - 2 * x - 1) ∧
  (∃ (vₓ vᵧ : ℝ), vₓ = 2 ∧ vᵧ = -3 ∧ quadraticFunction vₓ (-2) (-1) = vᵧ)  :=
by
  sorry

end NUMINAMATH_GPT_findQuadraticFunctionAndVertex_l545_54570


namespace NUMINAMATH_GPT_maximize_revenue_l545_54562

noncomputable def revenue (p : ℝ) : ℝ :=
  p * (150 - 6 * p)

theorem maximize_revenue : ∃ (p : ℝ), p = 12.5 ∧ p ≤ 30 ∧ ∀ q ≤ 30, revenue q ≤ revenue 12.5 := by 
  sorry

end NUMINAMATH_GPT_maximize_revenue_l545_54562


namespace NUMINAMATH_GPT_minimum_value_expression_l545_54504

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minimum_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  problem_statement x y z ≥ 9 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l545_54504


namespace NUMINAMATH_GPT_trip_time_difference_l545_54547

theorem trip_time_difference 
  (speed : ℕ) (dist1 dist2 : ℕ) (time_per_hour : ℕ) 
  (h_speed : speed = 60) 
  (h_dist1 : dist1 = 360) 
  (h_dist2 : dist2 = 420) 
  (h_time_per_hour : time_per_hour = 60) : 
  ((dist2 / speed - dist1 / speed) * time_per_hour) = 60 := 
by
  sorry

end NUMINAMATH_GPT_trip_time_difference_l545_54547


namespace NUMINAMATH_GPT_student_missed_number_l545_54597

theorem student_missed_number (student_sum : ℕ) (n : ℕ) (actual_sum : ℕ) : 
  student_sum = 575 → 
  actual_sum = n * (n + 1) / 2 → 
  n = 34 → 
  actual_sum - student_sum = 20 := 
by 
  sorry

end NUMINAMATH_GPT_student_missed_number_l545_54597


namespace NUMINAMATH_GPT_evaluate_expression_l545_54582

variables (x y : ℕ)

theorem evaluate_expression : x = 2 → y = 4 → y * (y - 2 * x + 1) = 4 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_evaluate_expression_l545_54582


namespace NUMINAMATH_GPT_box_surface_area_l545_54565

theorem box_surface_area (a b c : ℕ) (h1 : a * b * c = 280) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) : 
  2 * (a * b + b * c + c * a) = 262 :=
sorry

end NUMINAMATH_GPT_box_surface_area_l545_54565


namespace NUMINAMATH_GPT_has_three_zeros_iff_b_lt_neg3_l545_54568

def f (x b : ℝ) : ℝ := x^3 - b * x^2 - 4

theorem has_three_zeros_iff_b_lt_neg3 (b : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, f x₁ b = 0 ∧ f x₂ b = 0 ∧ f x₃ b = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ↔ b < -3 := 
sorry

end NUMINAMATH_GPT_has_three_zeros_iff_b_lt_neg3_l545_54568


namespace NUMINAMATH_GPT_subset_M_N_l545_54589

-- Definition of the sets
def M : Set ℝ := {-1, 1}
def N : Set ℝ := { x | 1/x < 3 }

theorem subset_M_N : M ⊆ N :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_subset_M_N_l545_54589


namespace NUMINAMATH_GPT_probability_calculation_l545_54520

def p_X := 1 / 5
def p_Y := 1 / 2
def p_Z := 5 / 8
def p_not_Z := 1 - p_Z

theorem probability_calculation : 
    (p_X * p_Y * p_not_Z) = (3 / 80) := by
    sorry

end NUMINAMATH_GPT_probability_calculation_l545_54520


namespace NUMINAMATH_GPT_simplify_fraction_l545_54505

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l545_54505


namespace NUMINAMATH_GPT_orchid_bushes_planted_tomorrow_l545_54541

theorem orchid_bushes_planted_tomorrow 
  (initial : ℕ) (planted_today : ℕ) (final : ℕ) (planted_tomorrow : ℕ) :
  initial = 47 →
  planted_today = 37 →
  final = 109 →
  planted_tomorrow = final - (initial + planted_today) →
  planted_tomorrow = 25 :=
by
  intros h_initial h_planted_today h_final h_planted_tomorrow
  rw [h_initial, h_planted_today, h_final] at h_planted_tomorrow
  exact h_planted_tomorrow


end NUMINAMATH_GPT_orchid_bushes_planted_tomorrow_l545_54541


namespace NUMINAMATH_GPT_find_2023rd_letter_in_sequence_l545_54527

def repeating_sequence : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'F', 'E', 'D', 'C', 'B', 'A']

def nth_in_repeating_sequence (n : ℕ) : Char :=
  repeating_sequence.get! (n % 13)

theorem find_2023rd_letter_in_sequence :
  nth_in_repeating_sequence 2023 = 'H' :=
by
  sorry

end NUMINAMATH_GPT_find_2023rd_letter_in_sequence_l545_54527


namespace NUMINAMATH_GPT_one_in_M_l545_54534

def M : Set ℕ := {1, 2, 3}

theorem one_in_M : 1 ∈ M := sorry

end NUMINAMATH_GPT_one_in_M_l545_54534


namespace NUMINAMATH_GPT_neg_sin_leq_1_l545_54577

theorem neg_sin_leq_1 :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by
  sorry

end NUMINAMATH_GPT_neg_sin_leq_1_l545_54577


namespace NUMINAMATH_GPT_consistency_condition_l545_54516

theorem consistency_condition (x y z a b c d : ℝ)
  (h1 : y + z = a)
  (h2 : x + y = b)
  (h3 : x + z = c)
  (h4 : x + y + z = d) : a + b + c = 2 * d :=
by sorry

end NUMINAMATH_GPT_consistency_condition_l545_54516


namespace NUMINAMATH_GPT_roy_total_pens_l545_54511

def number_of_pens (blue black red green purple : ℕ) : ℕ :=
  blue + black + red + green + purple

theorem roy_total_pens (blue black red green purple : ℕ)
  (h1 : blue = 8)
  (h2 : black = 4 * blue)
  (h3 : red = blue + black - 5)
  (h4 : green = red / 2)
  (h5 : purple = blue + green - 3) :
  number_of_pens blue black red green purple = 114 := by
  sorry

end NUMINAMATH_GPT_roy_total_pens_l545_54511


namespace NUMINAMATH_GPT_sum_of_first_20_terms_l545_54557

variable {a : ℕ → ℕ}

-- Conditions given in the problem
axiom seq_property : ∀ n, a n + 2 * a (n + 1) = 3 * n + 2
axiom arithmetic_sequence : ∀ n m, a (n + 1) - a n = a (m + 1) - a m

-- Theorem to be proved
theorem sum_of_first_20_terms (a : ℕ → ℕ) (S20 := (Finset.range 20).sum a) :
  S20 = 210 :=
  sorry

end NUMINAMATH_GPT_sum_of_first_20_terms_l545_54557


namespace NUMINAMATH_GPT_eliana_total_steps_l545_54579

-- Define the conditions given in the problem
def steps_first_day_exercise : Nat := 200
def steps_first_day_additional : Nat := 300
def steps_first_day : Nat := steps_first_day_exercise + steps_first_day_additional

def steps_second_day : Nat := 2 * steps_first_day
def steps_additional_on_third_day : Nat := 100
def steps_third_day : Nat := steps_second_day + steps_additional_on_third_day

-- Mathematical proof problem proving that the total number of steps is 2600
theorem eliana_total_steps : steps_first_day + steps_second_day + steps_third_day = 2600 := 
by
  sorry

end NUMINAMATH_GPT_eliana_total_steps_l545_54579


namespace NUMINAMATH_GPT_proof_problem_l545_54524

variable (p q : Prop)

theorem proof_problem
  (h₁ : p ∨ q)
  (h₂ : ¬p) :
  ¬p ∧ q :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l545_54524


namespace NUMINAMATH_GPT_compute_value_l545_54517

theorem compute_value : (7^2 - 6^2)^3 = 2197 := by
  sorry

end NUMINAMATH_GPT_compute_value_l545_54517


namespace NUMINAMATH_GPT_range_of_a_plus_c_l545_54507

-- Let a, b, c be the sides of the triangle opposite to angles A, B, and C respectively.
variable (a b c A B C : ℝ)

-- Given conditions
variable (h1 : b = Real.sqrt 3)
variable (h2 : (2 * c - a) / b * Real.cos B = Real.cos A)
variable (h3 : 0 < A ∧ A < Real.pi / 2)
variable (h4 : 0 < B ∧ B < Real.pi / 2)
variable (h5 : 0 < C ∧ C < Real.pi / 2)
variable (h6 : A + B + C = Real.pi)

-- The range of a + c
theorem range_of_a_plus_c (a b c A B C : ℝ) (h1 : b = Real.sqrt 3)
  (h2 : (2 * c - a) / b * Real.cos B = Real.cos A) (h3 : 0 < A ∧ A < Real.pi / 2)
  (h4 : 0 < B ∧ B < Real.pi / 2) (h5 : 0 < C ∧ C < Real.pi / 2) (h6 : A + B + C = Real.pi) :
  a + c ∈ Set.Ioc (Real.sqrt 3) (2 * Real.sqrt 3) :=
  sorry

end NUMINAMATH_GPT_range_of_a_plus_c_l545_54507


namespace NUMINAMATH_GPT_Tiffany_total_score_l545_54578

def points_per_treasure_type : Type := ℕ × ℕ × ℕ
def treasures_per_level : Type := ℕ × ℕ × ℕ

def points (bronze silver gold : ℕ) : ℕ :=
  bronze * 6 + silver * 15 + gold * 30

def treasures_level1 : treasures_per_level := (2, 3, 1)
def treasures_level2 : treasures_per_level := (3, 1, 2)
def treasures_level3 : treasures_per_level := (5, 2, 1)

def total_points (l1 l2 l3 : treasures_per_level) : ℕ :=
  let (b1, s1, g1) := l1
  let (b2, s2, g2) := l2
  let (b3, s3, g3) := l3
  points b1 s1 g1 + points b2 s2 g2 + points b3 s3 g3

theorem Tiffany_total_score :
  total_points treasures_level1 treasures_level2 treasures_level3 = 270 :=
by
  sorry

end NUMINAMATH_GPT_Tiffany_total_score_l545_54578


namespace NUMINAMATH_GPT_line_through_origin_and_conditions_l545_54518

-- Definitions:
def system_defines_line (m n p x y z : ℝ) : Prop :=
  (x / m = y / n) ∧ (y / n = z / p)

def lies_in_coordinate_plane (m n p : ℝ) : Prop :=
  (m = 0 ∨ n = 0 ∨ p = 0) ∧ ¬(m = 0 ∧ n = 0 ∧ p = 0)

def coincides_with_coordinate_axis (m n p : ℝ) : Prop :=
  (m = 0 ∧ n = 0) ∨ (m = 0 ∧ p = 0) ∨ (n = 0 ∧ p = 0)

-- Theorem statement:
theorem line_through_origin_and_conditions (m n p x y z : ℝ) :
  system_defines_line m n p x y z →
  (∀ m n p, lies_in_coordinate_plane m n p ↔ (m = 0 ∨ n = 0 ∨ p = 0) ∧ ¬(m = 0 ∧ n = 0 ∧ p = 0)) ∧
  (∀ m n p, coincides_with_coordinate_axis m n p ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ p = 0) ∨ (n = 0 ∧ p = 0)) :=
by
  sorry

end NUMINAMATH_GPT_line_through_origin_and_conditions_l545_54518


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_l545_54510

theorem arithmetic_geometric_mean (a b : ℝ) (h1 : a + b = 48) (h2 : a * b = 440) : a^2 + b^2 = 1424 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_l545_54510


namespace NUMINAMATH_GPT_problem_l545_54572

theorem problem (p q r : ℝ)
    (h1 : p * 1^2 + q * 1 + r = 5)
    (h2 : p * 2^2 + q * 2 + r = 3) :
  p + q + 2 * r = 10 := 
sorry

end NUMINAMATH_GPT_problem_l545_54572


namespace NUMINAMATH_GPT_benny_january_savings_l545_54526

theorem benny_january_savings :
  ∃ x : ℕ, x + x + 8 = 46 ∧ x = 19 :=
by
  sorry

end NUMINAMATH_GPT_benny_january_savings_l545_54526


namespace NUMINAMATH_GPT_count_ns_divisible_by_5_l545_54576

open Nat

theorem count_ns_divisible_by_5 : 
  let f (n : ℕ) := 2 * n^5 + 2 * n^4 + 3 * n^2 + 3 
  ∃ (N : ℕ), N = 19 ∧ 
  (∀ (n : ℕ), 2 ≤ n ∧ n ≤ 100 → f n % 5 = 0 → 
  (∃ (m : ℕ), 1 ≤ m ∧ m ≤ 19 ∧ n = 5 * m + 1)) :=
by
  sorry

end NUMINAMATH_GPT_count_ns_divisible_by_5_l545_54576


namespace NUMINAMATH_GPT_work_rate_sum_l545_54536

theorem work_rate_sum (A B : ℝ) (W : ℝ) (h1 : (A + B) * 4 = W) (h2 : A * 8 = W) : (A + B) * 4 = W :=
by
  -- placeholder for actual proof
  sorry

end NUMINAMATH_GPT_work_rate_sum_l545_54536


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l545_54583

variable {x y : ℝ}

theorem ratio_of_x_to_y (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l545_54583


namespace NUMINAMATH_GPT_jenny_change_l545_54542

-- Definitions for the conditions
def single_sided_cost_per_page : ℝ := 0.10
def double_sided_cost_per_page : ℝ := 0.17
def pages_per_essay : ℕ := 25
def single_sided_copies : ℕ := 5
def double_sided_copies : ℕ := 2
def pen_cost_before_tax : ℝ := 1.50
def number_of_pens : ℕ := 7
def sales_tax_rate : ℝ := 0.10
def payment_amount : ℝ := 2 * 20.00

-- Hypothesis for the total costs and calculations
noncomputable def total_single_sided_cost : ℝ := single_sided_copies * pages_per_essay * single_sided_cost_per_page
noncomputable def total_double_sided_cost : ℝ := double_sided_copies * pages_per_essay * double_sided_cost_per_page
noncomputable def total_pen_cost_before_tax : ℝ := number_of_pens * pen_cost_before_tax
noncomputable def total_sales_tax : ℝ := sales_tax_rate * total_pen_cost_before_tax
noncomputable def total_pen_cost : ℝ := total_pen_cost_before_tax + total_sales_tax
noncomputable def total_printing_cost : ℝ := total_single_sided_cost + total_double_sided_cost
noncomputable def total_cost : ℝ := total_printing_cost + total_pen_cost
noncomputable def change : ℝ := payment_amount - total_cost

-- The proof statement
theorem jenny_change : change = 7.45 := by
  sorry

end NUMINAMATH_GPT_jenny_change_l545_54542


namespace NUMINAMATH_GPT_sasha_stickers_l545_54523

variables (m n : ℕ) (t : ℝ)

-- Conditions
def conditions : Prop :=
  m < n ∧ -- Fewer coins than stickers
  m ≥ 1 ∧ -- At least one coin
  n ≥ 1 ∧ -- At least one sticker
  t > 1 ∧ -- t is greater than 1
  m * t + n = 100 ∧ -- Coin increase condition
  m + n * t = 101 -- Sticker increase condition

-- Theorem stating that the number of stickers must be 34 or 66
theorem sasha_stickers : conditions m n t → n = 34 ∨ n = 66 :=
sorry

end NUMINAMATH_GPT_sasha_stickers_l545_54523


namespace NUMINAMATH_GPT_complex_purely_imaginary_m_value_l545_54528

theorem complex_purely_imaginary_m_value (m : ℝ) :
  (m^2 - 1 = 0) ∧ (m + 1 ≠ 0) → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_complex_purely_imaginary_m_value_l545_54528


namespace NUMINAMATH_GPT_player_catches_ball_in_5_seconds_l545_54545

theorem player_catches_ball_in_5_seconds
    (s_ball : ℕ → ℝ) (s_player : ℕ → ℝ)
    (t_ball : ℕ)
    (t_player : ℕ)
    (d_player_initial : ℝ)
    (d_sideline : ℝ) :
  (∀ t, s_ball t = (4.375 * t - 0.375 * t^2)) →
  (∀ t, s_player t = (3.25 * t + 0.25 * t^2)) →
  (d_player_initial = 10) →
  (d_sideline = 23) →
  t_player = 5 →
  s_player t_player + d_player_initial = s_ball t_player ∧ s_ball t_player < d_sideline := 
by sorry

end NUMINAMATH_GPT_player_catches_ball_in_5_seconds_l545_54545


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_range_l545_54571

theorem geometric_sequence_common_ratio_range (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 < 0) 
  (h2 : ∀ n : ℕ, 0 < n → a n < a (n + 1))
  (hq : ∀ n : ℕ, a (n + 1) = a n * q) :
  0 < q ∧ q < 1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_range_l545_54571


namespace NUMINAMATH_GPT_part1_part2_l545_54508

theorem part1 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a + 2 * c = b * Real.cos C + Real.sqrt 3 * b * Real.sin C) : 
  B = 2 * Real.pi / 3 := 
sorry

theorem part2 
  (a b c : ℝ) 
  (A C : ℝ) 
  (h1 : a + 2 * c = b * Real.cos C + Real.sqrt 3 * b * Real.sin C)
  (h2 : b = 3) : 
  6 < (a + b + c) ∧ (a + b + c) ≤ 3 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l545_54508


namespace NUMINAMATH_GPT_truck_travel_distance_l545_54500

theorem truck_travel_distance
  (miles_traveled : ℕ)
  (gallons_used : ℕ)
  (new_gallons : ℕ)
  (rate : ℕ)
  (distance : ℕ) :
  (miles_traveled = 300) ∧
  (gallons_used = 10) ∧
  (new_gallons = 15) ∧
  (rate = miles_traveled / gallons_used) ∧
  (distance = rate * new_gallons)
  → distance = 450 :=
by
  sorry

end NUMINAMATH_GPT_truck_travel_distance_l545_54500


namespace NUMINAMATH_GPT_cube_splitting_odd_numbers_l545_54593

theorem cube_splitting_odd_numbers (m : ℕ) (h1 : m > 1) (h2 : ∃ k, 2 * k + 1 = 333) : m = 18 :=
sorry

end NUMINAMATH_GPT_cube_splitting_odd_numbers_l545_54593


namespace NUMINAMATH_GPT_bus_routes_arrangement_l545_54513

-- Define the lines and intersection points (stops).
def routes := Fin 10
def stops (r1 r2 : routes) : Prop := r1 ≠ r2 -- Representing intersection

-- First condition: Any subset of 9 routes will cover all stops.
def covers_all_stops (routes_subset : Finset routes) : Prop :=
  routes_subset.card = 9 → ∀ r1 r2 : routes, r1 ≠ r2 → stops r1 r2

-- Second condition: Any subset of 8 routes will miss at least one stop.
def misses_at_least_one_stop (routes_subset : Finset routes) : Prop :=
  routes_subset.card = 8 → ∃ r1 r2 : routes, r1 ≠ r2 ∧ ¬stops r1 r2

-- The theorem to prove that this arrangement is possible.
theorem bus_routes_arrangement : 
  (∃ stops_scheme : routes → routes → Prop, 
    (∀ subset_9 : Finset routes, covers_all_stops subset_9) ∧ 
    (∀ subset_8 : Finset routes, misses_at_least_one_stop subset_8)) :=
by
  sorry

end NUMINAMATH_GPT_bus_routes_arrangement_l545_54513


namespace NUMINAMATH_GPT_john_total_spent_l545_54560

def silver_ounces : ℝ := 2.5
def silver_price_per_ounce : ℝ := 25
def gold_ounces : ℝ := 3.5
def gold_price_multiplier : ℝ := 60
def platinum_ounces : ℝ := 4.5
def platinum_price_per_ounce_gbp : ℝ := 80
def palladium_ounces : ℝ := 5.5
def palladium_price_per_ounce_eur : ℝ := 100

def usd_per_gbp_monday : ℝ := 1.3
def usd_per_gbp_friday : ℝ := 1.4
def usd_per_eur_wednesday : ℝ := 1.15
def usd_per_eur_saturday : ℝ := 1.2

def discount_rate : ℝ := 0.05
def tax_rate : ℝ := 0.08

def total_amount_john_spends_usd : ℝ := 
  (silver_ounces * silver_price_per_ounce * (1 - discount_rate)) + 
  (gold_ounces * (gold_price_multiplier * silver_price_per_ounce) * (1 - discount_rate)) + 
  (((platinum_ounces * platinum_price_per_ounce_gbp) * (1 + tax_rate)) * usd_per_gbp_monday) + 
  ((palladium_ounces * palladium_price_per_ounce_eur) * usd_per_eur_wednesday)

theorem john_total_spent : total_amount_john_spends_usd = 6184.815 := by
  sorry

end NUMINAMATH_GPT_john_total_spent_l545_54560


namespace NUMINAMATH_GPT_sqrt10_parts_sqrt6_value_sqrt3_opposite_l545_54558

-- Problem 1
theorem sqrt10_parts : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 → (⌊Real.sqrt 10⌋ = 3 ∧ Real.sqrt 10 - 3 = Real.sqrt 10 - ⌊Real.sqrt 10⌋) :=
by
  sorry

-- Problem 2
theorem sqrt6_value (a b : ℝ) : a = Real.sqrt 6 - 2 ∧ b = 3 → (a + b - Real.sqrt 6 = 1) :=
by
  sorry

-- Problem 3
theorem sqrt3_opposite (x y : ℝ) : x = 13 ∧ y = Real.sqrt 3 - 1 → (-(x - y) = Real.sqrt 3 - 14) :=
by
  sorry

end NUMINAMATH_GPT_sqrt10_parts_sqrt6_value_sqrt3_opposite_l545_54558


namespace NUMINAMATH_GPT_collinear_condition_perpendicular_condition_l545_54580

namespace Vectors

-- Definitions for vectors a and b
def a : ℝ × ℝ := (4, -2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Collinear condition
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

-- Perpendicular condition
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Proof statement for collinear condition
theorem collinear_condition (x : ℝ) (h : collinear a (b x)) : x = -2 := sorry

-- Proof statement for perpendicular condition
theorem perpendicular_condition (x : ℝ) (h : perpendicular a (b x)) : x = 1 / 2 := sorry

end Vectors

end NUMINAMATH_GPT_collinear_condition_perpendicular_condition_l545_54580


namespace NUMINAMATH_GPT_rationalize_denominator_l545_54592

theorem rationalize_denominator :
  let A := 9
  let B := 7
  let C := -18
  let D := 0
  let S := 2
  let F := 111
  (A + B + C + D + S + F = 111) ∧ 
  (
    (1 / (Real.sqrt 5 + Real.sqrt 6 + 2 * Real.sqrt 2)) * 
    ((Real.sqrt 5 + Real.sqrt 6) - 2 * Real.sqrt 2) * 
    (3 - 2 * Real.sqrt 30) / 
    (3^2 - (2 * Real.sqrt 30)^2) = 
    (9 * Real.sqrt 5 + 7 * Real.sqrt 6 - 18 * Real.sqrt 2) / 111
  ) := by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l545_54592


namespace NUMINAMATH_GPT_range_of_a_l545_54566

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * x^2 + (a - 1) * x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x > 0) ↔ (1 ≤ a ∧ a < 5) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l545_54566


namespace NUMINAMATH_GPT_no_combination_of_five_coins_is_75_l545_54509

theorem no_combination_of_five_coins_is_75 :
  ∀ (a b c d e : ℕ), 
    (a + b + c + d + e = 5) →
    ∀ (v : ℤ), 
      v = a * 1 + b * 5 + c * 10 + d * 25 + e * 50 → 
      v ≠ 75 :=
by
  intro a b c d e h1 v h2
  sorry

end NUMINAMATH_GPT_no_combination_of_five_coins_is_75_l545_54509


namespace NUMINAMATH_GPT_sum_interior_angles_equal_diagonals_l545_54559

theorem sum_interior_angles_equal_diagonals (n : ℕ) (h : n = 4 ∨ n = 5) :
  (n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540 :=
by sorry

end NUMINAMATH_GPT_sum_interior_angles_equal_diagonals_l545_54559


namespace NUMINAMATH_GPT_Mater_costs_10_percent_of_Lightning_l545_54533

-- Conditions
def price_Lightning : ℕ := 140000
def price_Sally : ℕ := 42000
def price_Mater : ℕ := price_Sally / 3

-- The theorem we want to prove
theorem Mater_costs_10_percent_of_Lightning :
  (price_Mater * 100 / price_Lightning) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_Mater_costs_10_percent_of_Lightning_l545_54533


namespace NUMINAMATH_GPT_elderly_people_sampled_l545_54550

theorem elderly_people_sampled (total_population : ℕ) (children : ℕ) (elderly : ℕ) (middle_aged : ℕ) (sample_size : ℕ)
  (h1 : total_population = 1500)
  (h2 : ∃ d, children + d = elderly ∧ elderly + d = middle_aged)
  (h3 : total_population = children + elderly + middle_aged)
  (h4 : sample_size = 60) :
  elderly * (sample_size / total_population) = 20 :=
by
  -- Proof will be written here
  sorry

end NUMINAMATH_GPT_elderly_people_sampled_l545_54550


namespace NUMINAMATH_GPT_relatively_prime_sums_l545_54549

theorem relatively_prime_sums (x y : ℤ) (h : Int.gcd x y = 1) 
  : Int.gcd (x^2 + x * y + y^2) (x^2 + 3 * x * y + y^2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_relatively_prime_sums_l545_54549


namespace NUMINAMATH_GPT_negation_of_proposition_l545_54525

theorem negation_of_proposition : (¬ (∀ x : ℝ, x > 2 → x > 3)) = ∃ x > 2, x ≤ 3 := by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l545_54525


namespace NUMINAMATH_GPT_inequality_proof_l545_54535

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + b^2 + c^2 + a * b * c = 4) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 + a * b * c ≤ 4 := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l545_54535


namespace NUMINAMATH_GPT_number_of_multiples_840_in_range_l545_54543

theorem number_of_multiples_840_in_range :
  ∃ n, n = 1 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 2500 ∧ (840 ∣ x) → x = 1680 :=
by
  sorry

end NUMINAMATH_GPT_number_of_multiples_840_in_range_l545_54543


namespace NUMINAMATH_GPT_area_square_ratio_l545_54554

theorem area_square_ratio (r : ℝ) (h1 : r > 0)
  (s1 : ℝ) (hs1 : s1^2 = r^2)
  (s2 : ℝ) (hs2 : s2^2 = (4/5) * r^2) : 
  (s1^2 / s2^2) = (5 / 4) :=
by 
  sorry

end NUMINAMATH_GPT_area_square_ratio_l545_54554


namespace NUMINAMATH_GPT_area_of_triangle_ACD_l545_54519

theorem area_of_triangle_ACD :
  ∀ (AD AC height_AD height_AC : ℝ),
  AD = 6 → height_AD = 3 → AC = 3 → height_AC = 3 →
  (1 / 2 * AD * height_AD - 1 / 2 * AC * height_AC) = 4.5 :=
by
  intros AD AC height_AD height_AC hAD hheight_AD hAC hheight_AC
  sorry

end NUMINAMATH_GPT_area_of_triangle_ACD_l545_54519


namespace NUMINAMATH_GPT_prime_product_mod_32_l545_54586

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end NUMINAMATH_GPT_prime_product_mod_32_l545_54586


namespace NUMINAMATH_GPT_cookie_radius_and_area_l545_54588

def boundary_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 = 2 * x + 4 * y

theorem cookie_radius_and_area :
  (∃ r : ℝ, r = Real.sqrt 13) ∧ (∃ A : ℝ, A = 13 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_cookie_radius_and_area_l545_54588


namespace NUMINAMATH_GPT_projections_possibilities_l545_54553

-- Define the conditions: a and b are non-perpendicular skew lines, and α is a plane
variables {a b : Line} (α : Plane)

-- Non-perpendicular skew lines definition (external knowledge required for proper setup if not inbuilt)
def non_perpendicular_skew_lines (a b : Line) : Prop := sorry

-- Projections definition (external knowledge required for proper setup if not inbuilt)
def projections (a : Line) (α : Plane) : Line := sorry

-- The projections result in new conditions
def projected_parallel (a b : Line) (α : Plane) : Prop := sorry
def projected_perpendicular (a b : Line) (α : Plane) : Prop := sorry
def projected_same_line (a b : Line) (α : Plane) : Prop := sorry
def projected_line_and_point (a b : Line) (α : Plane) : Prop := sorry

-- Given the given conditions
variables (ha : non_perpendicular_skew_lines a b)

-- Prove the resultant conditions where the projections satisfy any 3 of the listed possibilities: parallel, perpendicular, line and point.
theorem projections_possibilities :
    (projected_parallel a b α ∨ projected_perpendicular a b α ∨ projected_line_and_point a b α) ∧
    ¬ projected_same_line a b α := sorry

end NUMINAMATH_GPT_projections_possibilities_l545_54553


namespace NUMINAMATH_GPT_min_value_x_plus_y_l545_54590

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = x * y) : x + y ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l545_54590


namespace NUMINAMATH_GPT_mn_condition_l545_54530

theorem mn_condition {m n : ℕ} (h : m * n = 121) : (m + 1) * (n + 1) = 144 :=
sorry

end NUMINAMATH_GPT_mn_condition_l545_54530


namespace NUMINAMATH_GPT_radius_of_smaller_base_l545_54529

theorem radius_of_smaller_base (C1 C2 : ℝ) (r : ℝ) (l : ℝ) (A : ℝ) 
    (h1 : C2 = 3 * C1) 
    (h2 : l = 3) 
    (h3 : A = 84 * Real.pi) 
    (h4 : C1 = 2 * Real.pi * r) 
    (h5 : C2 = 2 * Real.pi * (3 * r)) :
    r = 7 := 
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_radius_of_smaller_base_l545_54529


namespace NUMINAMATH_GPT_lee_can_make_cookies_l545_54574

def cookies_per_cup_of_flour (cookies : ℕ) (flour_cups : ℕ) : ℕ :=
  cookies / flour_cups

def flour_needed (sugar_cups : ℕ) (flour_to_sugar_ratio : ℕ) : ℕ :=
  sugar_cups * flour_to_sugar_ratio

def total_cookies (cookies_per_cup : ℕ) (total_flour : ℕ) : ℕ :=
  cookies_per_cup * total_flour

theorem lee_can_make_cookies
  (cookies : ℕ)
  (flour_cups : ℕ)
  (sugar_cups : ℕ)
  (flour_to_sugar_ratio : ℕ)
  (h1 : cookies = 24)
  (h2 : flour_cups = 4)
  (h3 : sugar_cups = 3)
  (h4 : flour_to_sugar_ratio = 2) :
  total_cookies (cookies_per_cup_of_flour cookies flour_cups)
    (flour_needed sugar_cups flour_to_sugar_ratio) = 36 :=
by
  sorry

end NUMINAMATH_GPT_lee_can_make_cookies_l545_54574


namespace NUMINAMATH_GPT_jennie_speed_difference_l545_54501

theorem jennie_speed_difference :
  (∀ (d t1 t2 : ℝ), (d = 200) → (t1 = 5) → (t2 = 4) → (40 = d / t1) → (50 = d / t2) → (50 - 40 = 10)) :=
by
  intros d t1 t2 h_d h_t1 h_t2 h_speed_heavy h_speed_no_traffic
  sorry

end NUMINAMATH_GPT_jennie_speed_difference_l545_54501


namespace NUMINAMATH_GPT_women_population_percentage_l545_54596

theorem women_population_percentage (W M : ℕ) (h : M = 2 * W) : (W : ℚ) / (M : ℚ) = (50 : ℚ) / 100 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_women_population_percentage_l545_54596
