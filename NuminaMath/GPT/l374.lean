import Mathlib

namespace NUMINAMATH_GPT_weight_of_green_peppers_l374_37466

-- Definitions for conditions and question
def total_weight : ℝ := 0.6666666667
def is_split_equally (x y : ℝ) : Prop := x = y

-- Theorem statement that needs to be proved
theorem weight_of_green_peppers (g r : ℝ) (h_split : is_split_equally g r) (h_total : g + r = total_weight) :
  g = 0.33333333335 :=
by sorry

end NUMINAMATH_GPT_weight_of_green_peppers_l374_37466


namespace NUMINAMATH_GPT_slope_y_intercept_product_eq_neg_five_over_two_l374_37410

theorem slope_y_intercept_product_eq_neg_five_over_two :
  let A := (0, 10)
  let B := (0, 0)
  let C := (10, 0)
  let D := ((0 + 0) / 2, (10 + 0) / 2) -- midpoint of A and B
  let slope := (D.2 - C.2) / (D.1 - C.1)
  let y_intercept := D.2
  slope * y_intercept = -5 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_slope_y_intercept_product_eq_neg_five_over_two_l374_37410


namespace NUMINAMATH_GPT_problem1_problem2_l374_37475

section
variables (x a : ℝ)

-- Problem 1: Prove \(2^{3x-1} < 2 \implies x < \frac{2}{3}\)
theorem problem1 : (2:ℝ)^(3*x-1) < 2 → x < (2:ℝ)/3 :=
by sorry

-- Problem 2: Prove \(a^{3x^2+3x-1} < a^{3x^2+3} \implies (a > 1 \implies x < \frac{4}{3}) \land (0 < a < 1 \implies x > \frac{4}{3})\) given \(a > 0\) and \(a \neq 1\)
theorem problem2 (h0 : a > 0) (h1 : a ≠ 1) :
  a^(3*x^2 + 3*x - 1) < a^(3*x^2 + 3) →
  ((1 < a → x < (4:ℝ)/3) ∧ (0 < a ∧ a < 1 → x > (4:ℝ)/3)) :=
by sorry
end

end NUMINAMATH_GPT_problem1_problem2_l374_37475


namespace NUMINAMATH_GPT_total_fortunate_numbers_is_65_largest_odd_fortunate_number_is_1995_l374_37429

-- Definition of properties required as per the given conditions
def is_fortunate_number (abcd ab cd : ℕ) : Prop :=
  abcd = 100 * ab + cd ∧
  ab ≠ cd ∧
  ab ∣ cd ∧
  cd ∣ abcd

-- Total number of fortunate numbers is 65
theorem total_fortunate_numbers_is_65 : 
  ∃ n : ℕ, n = 65 ∧ 
  ∀(abcd ab cd : ℕ), is_fortunate_number abcd ab cd → n = 65 :=
sorry

-- Largest odd fortunate number is 1995
theorem largest_odd_fortunate_number_is_1995 : 
  ∃ abcd : ℕ, abcd = 1995 ∧ 
  ∀(abcd' ab cd : ℕ), is_fortunate_number abcd' ab cd ∧ cd % 2 = 1 → abcd = 1995 :=
sorry

end NUMINAMATH_GPT_total_fortunate_numbers_is_65_largest_odd_fortunate_number_is_1995_l374_37429


namespace NUMINAMATH_GPT_pos_count_a5_eq_2_pos_pos_count_an_eq_n2_l374_37403

-- Definitions based on the problem's conditions
def a (n : Nat) : Nat := n * n

def pos_count (n : Nat) : Nat :=
  List.length (List.filter (λ m : Nat => a m < n) (List.range (n + 1)))

def pos_pos_count (n : Nat) : Nat :=
  pos_count (pos_count n)

-- Theorem statements
theorem pos_count_a5_eq_2 : pos_count 5 = 2 := 
by
  -- Proof would go here
  sorry

theorem pos_pos_count_an_eq_n2 (n : Nat) : pos_pos_count n = n * n :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_pos_count_a5_eq_2_pos_pos_count_an_eq_n2_l374_37403


namespace NUMINAMATH_GPT_complex_power_sum_l374_37487

noncomputable def z : ℂ := sorry

theorem complex_power_sum (hz : z^2 + z + 1 = 0) : 
  z^101 + z^102 + z^103 + z^104 + z^105 = -2 :=
sorry

end NUMINAMATH_GPT_complex_power_sum_l374_37487


namespace NUMINAMATH_GPT_sequence_a500_l374_37467

theorem sequence_a500 (a : ℕ → ℤ)
  (h1 : a 1 = 2010)
  (h2 : a 2 = 2011)
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n) :
  a 500 = 2177 :=
sorry

end NUMINAMATH_GPT_sequence_a500_l374_37467


namespace NUMINAMATH_GPT_total_money_together_is_l374_37440

def Sam_has : ℚ := 750.50
def Billy_has (S : ℚ) : ℚ := 4.5 * S - 345.25
def Lila_has (B S : ℚ) : ℚ := 2.25 * (B - S)
def Total_money (S B L : ℚ) : ℚ := S + B + L

theorem total_money_together_is :
  Total_money Sam_has (Billy_has Sam_has) (Lila_has (Billy_has Sam_has) Sam_has) = 8915.88 :=
by sorry

end NUMINAMATH_GPT_total_money_together_is_l374_37440


namespace NUMINAMATH_GPT_tiffany_won_lives_l374_37462
-- Step d: Lean 4 statement incorporating the conditions and the proof goal


-- Define initial lives, lives won in the hard part and the additional lives won
def initial_lives : Float := 43.0
def additional_lives : Float := 27.0
def total_lives_after_wins : Float := 84.0

open Classical

theorem tiffany_won_lives (x : Float) :
    initial_lives + x + additional_lives = total_lives_after_wins →
    x = 14.0 :=
by
  intros h
  -- This "sorry" indicates that the proof is skipped.
  sorry

end NUMINAMATH_GPT_tiffany_won_lives_l374_37462


namespace NUMINAMATH_GPT_product_of_hypotenuse_segments_eq_area_l374_37428

theorem product_of_hypotenuse_segments_eq_area (x y c t : ℝ) : 
  -- Conditions
  (c = x + y) → 
  (t = x * y) →
  -- Conclusion
  x * y = t :=
by
  intros
  sorry

end NUMINAMATH_GPT_product_of_hypotenuse_segments_eq_area_l374_37428


namespace NUMINAMATH_GPT_terry_problems_wrong_l374_37474

theorem terry_problems_wrong (R W : ℕ) 
  (h1 : R + W = 25) 
  (h2 : 4 * R - W = 85) : 
  W = 3 := 
by
  sorry

end NUMINAMATH_GPT_terry_problems_wrong_l374_37474


namespace NUMINAMATH_GPT_carousel_revolutions_l374_37425

/-- Prove that the number of revolutions a horse 4 feet from the center needs to travel the same distance
as a horse 16 feet from the center making 40 revolutions is 160 revolutions. -/
theorem carousel_revolutions (r₁ : ℕ := 16) (revolutions₁ : ℕ := 40) (r₂ : ℕ := 4) :
  (revolutions₁ * (r₁ / r₂) = 160) :=
sorry

end NUMINAMATH_GPT_carousel_revolutions_l374_37425


namespace NUMINAMATH_GPT_simplify_expression_l374_37418

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 3) :
  (3 * x ^ 2 - 2 * x - 4) / ((x + 2) * (x - 3)) - (5 + x) / ((x + 2) * (x - 3)) =
  3 * (x ^ 2 - x - 3) / ((x + 2) * (x - 3)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l374_37418


namespace NUMINAMATH_GPT_sandy_money_l374_37483

theorem sandy_money (x : ℝ) (h : 0.70 * x = 210) : x = 300 := by
sorry

end NUMINAMATH_GPT_sandy_money_l374_37483


namespace NUMINAMATH_GPT_find_length_of_side_c_find_measure_of_angle_B_l374_37404

variable {A B C a b c : ℝ}

def triangle_problem (a b c A B C : ℝ) :=
  a * Real.cos B = 3 ∧
  b * Real.cos A = 1 ∧
  A - B = Real.pi / 6 ∧
  a^2 + c^2 - b^2 - 6 * c = 0 ∧
  b^2 + c^2 - a^2 - 2 * c = 0

theorem find_length_of_side_c (h : triangle_problem a b c A B C) :
  c = 4 :=
sorry

theorem find_measure_of_angle_B (h : triangle_problem a b c A B C) :
  B = Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_find_length_of_side_c_find_measure_of_angle_B_l374_37404


namespace NUMINAMATH_GPT_like_terms_to_exponents_matching_l374_37486

theorem like_terms_to_exponents_matching (n m : ℕ) (h1 : n = 3) (h2 : m = 3) : m^n = 27 := by
  sorry

end NUMINAMATH_GPT_like_terms_to_exponents_matching_l374_37486


namespace NUMINAMATH_GPT_tim_paid_correct_amount_l374_37405

-- Define the conditions given in the problem
def mri_cost : ℝ := 1200
def doctor_hourly_rate : ℝ := 300
def doctor_time_hours : ℝ := 0.5 -- 30 minutes is half an hour
def fee_for_being_seen : ℝ := 150
def insurance_coverage_rate : ℝ := 0.80

-- Total amount Tim paid calculation
def total_cost_before_insurance : ℝ :=
  mri_cost + (doctor_hourly_rate * doctor_time_hours) + fee_for_being_seen

def insurance_coverage : ℝ :=
  total_cost_before_insurance * insurance_coverage_rate

def amount_tim_paid : ℝ :=
  total_cost_before_insurance - insurance_coverage

-- Prove that Tim paid $300
theorem tim_paid_correct_amount : amount_tim_paid = 300 :=
by
  sorry

end NUMINAMATH_GPT_tim_paid_correct_amount_l374_37405


namespace NUMINAMATH_GPT_scientific_notation_of_150000000000_l374_37497

theorem scientific_notation_of_150000000000 :
  150000000000 = 1.5 * 10^11 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_150000000000_l374_37497


namespace NUMINAMATH_GPT_tangent_line_at_2_eq_l374_37437

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

theorem tangent_line_at_2_eq :
  let x := (2 : ℝ)
  let slope := (deriv f) x
  let y := f x
  ∃ (m y₀ : ℝ), m = slope ∧ y₀ = y ∧ 
    (∀ (x y : ℝ), y = m * (x - 2) + y₀ → x - y - 4 = 0)
:= sorry

end NUMINAMATH_GPT_tangent_line_at_2_eq_l374_37437


namespace NUMINAMATH_GPT_remainder_when_x_plus_2uy_div_y_l374_37469

theorem remainder_when_x_plus_2uy_div_y (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v) (h3 : v < y) :
  (x + 2 * u * y) % y = v := 
sorry

end NUMINAMATH_GPT_remainder_when_x_plus_2uy_div_y_l374_37469


namespace NUMINAMATH_GPT_width_at_bottom_l374_37449

-- Defining the given values and conditions
def top_width : ℝ := 14
def area : ℝ := 770
def depth : ℝ := 70

-- The proof problem
theorem width_at_bottom (b : ℝ) (h : area = (1/2) * (top_width + b) * depth) : b = 8 :=
by
  sorry

end NUMINAMATH_GPT_width_at_bottom_l374_37449


namespace NUMINAMATH_GPT_milkman_cows_l374_37409

theorem milkman_cows (x : ℕ) (c : ℕ) :
  (3 * x * c = 720) ∧ (3 * x * c + 50 * c + 140 * c + 63 * c = 3250) → x = 24 :=
by
  sorry

end NUMINAMATH_GPT_milkman_cows_l374_37409


namespace NUMINAMATH_GPT_max_AMC_expression_l374_37484

theorem max_AMC_expression (A M C : ℕ) (h : A + M + C = 24) :
  A * M * C + A * M + M * C + C * A ≤ 704 :=
sorry

end NUMINAMATH_GPT_max_AMC_expression_l374_37484


namespace NUMINAMATH_GPT_milk_per_cow_per_day_l374_37401

-- Define the conditions
def num_cows := 52
def weekly_milk_production := 364000 -- ounces

-- State the theorem
theorem milk_per_cow_per_day :
  (weekly_milk_production / 7 / num_cows) = 1000 := 
by
  -- Here we would include the proof, so we use sorry as placeholder
  sorry

end NUMINAMATH_GPT_milk_per_cow_per_day_l374_37401


namespace NUMINAMATH_GPT_area_of_figure_l374_37465

def equation (x y : ℝ) : Prop := |15 * x| + |8 * y| + |120 - 15 * x - 8 * y| = 120

theorem area_of_figure : ∃ (A : ℝ), A = 60 ∧ 
  (∃ (x y : ℝ), equation x y) :=
sorry

end NUMINAMATH_GPT_area_of_figure_l374_37465


namespace NUMINAMATH_GPT_most_frequent_data_is_mode_l374_37477

def most_frequent_data_name (dataset : Type) : String := "Mode"

theorem most_frequent_data_is_mode (dataset : Type) :
  most_frequent_data_name dataset = "Mode" :=
by
  sorry

end NUMINAMATH_GPT_most_frequent_data_is_mode_l374_37477


namespace NUMINAMATH_GPT_pow_mod_eleven_l374_37488

theorem pow_mod_eleven : 
  ∀ (n : ℕ), (n ≡ 5 ^ 1 [MOD 11] → n ≡ 5 [MOD 11]) ∧ 
             (n ≡ 5 ^ 2 [MOD 11] → n ≡ 3 [MOD 11]) ∧ 
             (n ≡ 5 ^ 3 [MOD 11] → n ≡ 4 [MOD 11]) ∧ 
             (n ≡ 5 ^ 4 [MOD 11] → n ≡ 9 [MOD 11]) ∧ 
             (n ≡ 5 ^ 5 [MOD 11] → n ≡ 1 [MOD 11]) →
  5 ^ 1233 ≡ 4 [MOD 11] :=
by
  intro n h
  sorry

end NUMINAMATH_GPT_pow_mod_eleven_l374_37488


namespace NUMINAMATH_GPT_eggs_left_after_cupcakes_l374_37454

-- Definitions derived from the given conditions
def dozen := 12
def initial_eggs := 3 * dozen
def crepes_fraction := 1 / 4
def cupcakes_fraction := 2 / 3

theorem eggs_left_after_cupcakes :
  let eggs_after_crepes := initial_eggs - crepes_fraction * initial_eggs;
  let eggs_after_cupcakes := eggs_after_crepes - cupcakes_fraction * eggs_after_crepes;
  eggs_after_cupcakes = 9 := sorry

end NUMINAMATH_GPT_eggs_left_after_cupcakes_l374_37454


namespace NUMINAMATH_GPT_trapezium_height_l374_37489

-- Define the data for the trapezium
def length1 : ℝ := 20
def length2 : ℝ := 18
def area : ℝ := 285

-- Define the result we want to prove
theorem trapezium_height (h : ℝ) : (1/2) * (length1 + length2) * h = area → h = 15 := 
by
  sorry

end NUMINAMATH_GPT_trapezium_height_l374_37489


namespace NUMINAMATH_GPT_reflect_point_P_l374_37470

-- Define the point P
def P : ℝ × ℝ := (-3, 2)

-- Define the reflection across the x-axis
def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Theorem to prove the coordinates of the point P with respect to the x-axis
theorem reflect_point_P : reflect_x_axis P = (-3, -2) := by
  sorry

end NUMINAMATH_GPT_reflect_point_P_l374_37470


namespace NUMINAMATH_GPT_remainder_17_pow_63_mod_7_l374_37496

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end NUMINAMATH_GPT_remainder_17_pow_63_mod_7_l374_37496


namespace NUMINAMATH_GPT_necessary_sufficient_condition_l374_37455

theorem necessary_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_necessary_sufficient_condition_l374_37455


namespace NUMINAMATH_GPT_find_k_l374_37485

theorem find_k (k : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + A.2) = (B.1 + B.2) / 2 ∧ (A.1^2 + A.2^2 - 6 * A.1 - 4 * A.2 + 9 = 0) ∧ (B.1^2 + B.2^2 - 6 * B.1 - 4 * B.2 + 9 = 0)
     ∧ dist A B = 2 * Real.sqrt 3)
  (h3 : ∀ x y : ℝ, y = k * x + 3 → (x^2 + y^2 - 6 * x - 4 * y + 9) = 0)
  : k = 1 := sorry

end NUMINAMATH_GPT_find_k_l374_37485


namespace NUMINAMATH_GPT_ratio_of_percentage_change_l374_37432

theorem ratio_of_percentage_change
  (P U U' : ℝ)
  (h_price_decrease : U' = 4 * U)
  : (300 / 75) = 4 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_percentage_change_l374_37432


namespace NUMINAMATH_GPT_coupon_discount_l374_37430

theorem coupon_discount (total_before_coupon : ℝ) (amount_paid_per_friend : ℝ) (number_of_friends : ℕ) :
  total_before_coupon = 100 ∧ amount_paid_per_friend = 18.8 ∧ number_of_friends = 5 →
  ∃ discount_percentage : ℝ, discount_percentage = 6 :=
by
  sorry

end NUMINAMATH_GPT_coupon_discount_l374_37430


namespace NUMINAMATH_GPT_sara_oranges_l374_37433

-- Conditions
def joan_oranges : Nat := 37
def total_oranges : Nat := 47

-- Mathematically equivalent proof problem: Prove that the number of oranges picked by Sara is 10
theorem sara_oranges : total_oranges - joan_oranges = 10 :=
by
  sorry

end NUMINAMATH_GPT_sara_oranges_l374_37433


namespace NUMINAMATH_GPT_binomial_arithmetic_sequence_iff_l374_37491

open Nat

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  n.choose k 

-- Conditions
def is_arithmetic_sequence (n k : ℕ) : Prop :=
  binomial n (k-1) - 2 * binomial n k + binomial n (k+1) = 0

-- Statement to prove
theorem binomial_arithmetic_sequence_iff (u : ℕ) (u_gt2 : u > 2) :
  ∃ (n k : ℕ), (n = u^2 - 2) ∧ (k = binomial u 2 - 1 ∨ k = binomial (u+1) 2 - 1) 
  ↔ is_arithmetic_sequence n k := 
sorry

end NUMINAMATH_GPT_binomial_arithmetic_sequence_iff_l374_37491


namespace NUMINAMATH_GPT_circles_intersect_l374_37450

theorem circles_intersect
  (r : ℝ) (R : ℝ) (d : ℝ)
  (hr : r = 4)
  (hR : R = 5)
  (hd : d = 6) :
  1 < d ∧ d < r + R :=
by
  sorry

end NUMINAMATH_GPT_circles_intersect_l374_37450


namespace NUMINAMATH_GPT_min_value_ineq_l374_37434

theorem min_value_ineq (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) : 
  (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 3 / 2 := 
sorry

end NUMINAMATH_GPT_min_value_ineq_l374_37434


namespace NUMINAMATH_GPT_maximum_value_of_piecewise_function_l374_37493

noncomputable def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + 3 else 
  if 0 < x ∧ x ≤ 1 then x + 3 else 
  -x + 5

theorem maximum_value_of_piecewise_function : ∃ M, ∀ x, piecewise_function x ≤ M ∧ (∀ y, (∀ x, piecewise_function x ≤ y) → M ≤ y) := 
by
  use 4
  sorry

end NUMINAMATH_GPT_maximum_value_of_piecewise_function_l374_37493


namespace NUMINAMATH_GPT_cars_meet_after_5_hours_l374_37473

theorem cars_meet_after_5_hours :
  ∀ (t : ℝ), (40 * t + 60 * t = 500) → t = 5 := 
by
  intro t
  intro h
  sorry

end NUMINAMATH_GPT_cars_meet_after_5_hours_l374_37473


namespace NUMINAMATH_GPT_find_correct_t_l374_37453

theorem find_correct_t (t : ℝ) :
  (∃! x1 x2 x3 : ℝ, x1^2 - 4*|x1| + 3 = t ∧
                     x2^2 - 4*|x2| + 3 = t ∧
                     x3^2 - 4*|x3| + 3 = t) → t = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_correct_t_l374_37453


namespace NUMINAMATH_GPT_continuous_function_form_l374_37435

noncomputable def f (t : ℝ) : ℝ := sorry

theorem continuous_function_form (f : ℝ → ℝ) (h1 : f 0 = -1 / 2) (h2 : ∀ x y, f (x + y) ≥ f x + f y + f (x * y) + 1) :
  ∃ (a : ℝ), ∀ x, f x = 1 / 2 + a * x + (a/2) * x ^ 2 := sorry

end NUMINAMATH_GPT_continuous_function_form_l374_37435


namespace NUMINAMATH_GPT_percentage_error_computation_l374_37416

theorem percentage_error_computation (x : ℝ) (h : 0 < x) : 
  let correct_result := 8 * x
  let erroneous_result := x / 8
  let error := |correct_result - erroneous_result|
  let error_percentage := (error / correct_result) * 100
  error_percentage = 98 :=
by
  sorry

end NUMINAMATH_GPT_percentage_error_computation_l374_37416


namespace NUMINAMATH_GPT_stratified_sampling_l374_37482

theorem stratified_sampling (N : ℕ) (r1 r2 r3 : ℕ) (sample_size : ℕ) 
  (ratio_given : r1 = 5 ∧ r2 = 2 ∧ r3 = 3) 
  (total_sample_size : sample_size = 200) :
  sample_size * r3 / (r1 + r2 + r3) = 60 := 
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l374_37482


namespace NUMINAMATH_GPT_triangle_perimeter_is_correct_l374_37422

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (S : ℝ)

def triangle_perimeter (a b c : ℝ) := a + b + c

theorem triangle_perimeter_is_correct :
  c = sqrt 7 → C = π / 3 → S = 3 * sqrt 3 / 2 →
  S = (1 / 2) * a * b * sin (C) → c^2 = a^2 + b^2 - 2 * a * b * cos (C) →
  ∃ a b : ℝ, triangle_perimeter a b c = 5 + sqrt 7 :=
  by
    intros h1 h2 h3 h4 h5
    sorry

end NUMINAMATH_GPT_triangle_perimeter_is_correct_l374_37422


namespace NUMINAMATH_GPT_least_number_subtracted_l374_37441

/--
  What least number must be subtracted from 9671 so that the remaining number is divisible by 5, 7, and 11?
-/
theorem least_number_subtracted
  (x : ℕ) :
  (9671 - x) % 5 = 0 ∧ (9671 - x) % 7 = 0 ∧ (9671 - x) % 11 = 0 ↔ x = 46 :=
sorry

end NUMINAMATH_GPT_least_number_subtracted_l374_37441


namespace NUMINAMATH_GPT_true_inverse_propositions_count_l374_37415

-- Let P1, P2, P3, P4 denote the original propositions
def P1 := "Supplementary angles are congruent, and two lines are parallel."
def P2 := "If |a| = |b|, then a = b."
def P3 := "Right angles are congruent."
def P4 := "Congruent angles are vertical angles."

-- Let IP1, IP2, IP3, IP4 denote the inverse propositions
def IP1 := "Two lines are parallel, and supplementary angles are congruent."
def IP2 := "If a = b, then |a| = |b|."
def IP3 := "Congruent angles are right angles."
def IP4 := "Vertical angles are congruent angles."

-- Counting the number of true inverse propositions
def countTrueInversePropositions : ℕ :=
  let p1_inverse_true := true  -- IP1 is true
  let p2_inverse_true := true  -- IP2 is true
  let p3_inverse_true := false -- IP3 is false
  let p4_inverse_true := true  -- IP4 is true
  [p1_inverse_true, p2_inverse_true, p4_inverse_true].length

-- The statement to be proved
theorem true_inverse_propositions_count : countTrueInversePropositions = 3 := by
  sorry

end NUMINAMATH_GPT_true_inverse_propositions_count_l374_37415


namespace NUMINAMATH_GPT_intersection_empty_l374_37420

def A : Set ℝ := {x | x > -1 ∧ x ≤ 3}
def B : Set ℝ := {2, 4}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end NUMINAMATH_GPT_intersection_empty_l374_37420


namespace NUMINAMATH_GPT_binomial_20_5_l374_37427

theorem binomial_20_5 : Nat.choose 20 5 = 15504 := by
  sorry

end NUMINAMATH_GPT_binomial_20_5_l374_37427


namespace NUMINAMATH_GPT_simplify_fraction_l374_37461

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l374_37461


namespace NUMINAMATH_GPT_trigonometric_inequalities_l374_37423

noncomputable def a : ℝ := Real.sin (21 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (72 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (23 * Real.pi / 180)

-- The proof statement
theorem trigonometric_inequalities : c > a ∧ a > b :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_inequalities_l374_37423


namespace NUMINAMATH_GPT_cube_root_of_4913_has_unit_digit_7_cube_root_of_50653_is_37_cube_root_of_110592_is_48_l374_37451

theorem cube_root_of_4913_has_unit_digit_7 :
  (∃ (y : ℕ), y^3 = 4913 ∧ y % 10 = 7) :=
sorry

theorem cube_root_of_50653_is_37 :
  (∃ (y : ℕ), y = 37 ∧ y^3 = 50653) :=
sorry

theorem cube_root_of_110592_is_48 :
  (∃ (y : ℕ), y = 48 ∧ y^3 = 110592) :=
sorry

end NUMINAMATH_GPT_cube_root_of_4913_has_unit_digit_7_cube_root_of_50653_is_37_cube_root_of_110592_is_48_l374_37451


namespace NUMINAMATH_GPT_largest_4_digit_divisible_by_98_l374_37464

theorem largest_4_digit_divisible_by_98 :
  ∃ n, (n ≤ 9999 ∧ 9999 < n + 98) ∧ 98 ∣ n :=
sorry

end NUMINAMATH_GPT_largest_4_digit_divisible_by_98_l374_37464


namespace NUMINAMATH_GPT_relation_of_a_and_b_l374_37498

theorem relation_of_a_and_b (a b : ℝ) (h : 2^a + Real.log a / Real.log 2 = 4^b + 2 * Real.log b / Real.log 4) : a < 2 * b :=
sorry

end NUMINAMATH_GPT_relation_of_a_and_b_l374_37498


namespace NUMINAMATH_GPT_pizzas_bought_l374_37481

def slices_per_pizza := 8
def total_slices := 16

theorem pizzas_bought : total_slices / slices_per_pizza = 2 := by
  sorry

end NUMINAMATH_GPT_pizzas_bought_l374_37481


namespace NUMINAMATH_GPT_rice_wheat_ratio_l374_37406

theorem rice_wheat_ratio (total_shi : ℕ) (sample_size : ℕ) (wheat_in_sample : ℕ) (total_sample : ℕ) : 
  total_shi = 1512 ∧ sample_size = 216 ∧ wheat_in_sample = 27 ∧ total_sample = 1512 * (wheat_in_sample / sample_size) →
  total_sample = 189 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_rice_wheat_ratio_l374_37406


namespace NUMINAMATH_GPT_fraction_BC_AD_l374_37436

-- Defining points and segments
variables (A B C D : Point)
variable (len : Point → Point → ℝ) -- length function

-- Conditions
axiom AB_eq_3BD : len A B = 3 * len B D
axiom AC_eq_7CD : len A C = 7 * len C D
axiom B_mid_AD : 2 * len A B = len A D

-- Theorem: Proving the fraction of BC relative to AD is 2/3
theorem fraction_BC_AD : (len B C) / (len A D) = 2 / 3 :=
sorry

end NUMINAMATH_GPT_fraction_BC_AD_l374_37436


namespace NUMINAMATH_GPT_problem_statement_l374_37439

variable {x y z : ℝ}

theorem problem_statement 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z)
  (hxyz : x * y * z = 1) :
  1 / (x ^ 3 * y) + 1 / (y ^ 3 * z) + 1 / (z ^ 3 * x) ≥ x * y + y * z + z * x :=
by sorry

end NUMINAMATH_GPT_problem_statement_l374_37439


namespace NUMINAMATH_GPT_convert_base_8_to_10_l374_37417

theorem convert_base_8_to_10 :
  let n := 4532
  let b := 8
  n = 4 * b^3 + 5 * b^2 + 3 * b^1 + 2 * b^0 → 4 * 512 + 5 * 64 + 3 * 8 + 2 * 1 = 2394 :=
by
  sorry

end NUMINAMATH_GPT_convert_base_8_to_10_l374_37417


namespace NUMINAMATH_GPT_ratio_of_A_to_B_is_4_l374_37499

noncomputable def A_share : ℝ := 360
noncomputable def B_share : ℝ := 90
noncomputable def ratio_A_B : ℝ := A_share / B_share

theorem ratio_of_A_to_B_is_4 : ratio_A_B = 4 :=
by
  -- This is the proof that we are skipping
  sorry

end NUMINAMATH_GPT_ratio_of_A_to_B_is_4_l374_37499


namespace NUMINAMATH_GPT_series_sum_eq_1_div_400_l374_37495

theorem series_sum_eq_1_div_400 :
  (∑' n : ℕ, (4 * n + 2) / ((4 * n + 1)^2 * (4 * n + 5)^2)) = 1 / 400 := 
sorry

end NUMINAMATH_GPT_series_sum_eq_1_div_400_l374_37495


namespace NUMINAMATH_GPT_subset_0_in_X_l374_37445

-- Define the set X
def X : Set ℤ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Define the theorem to prove
theorem subset_0_in_X : {0} ⊆ X :=
by
  sorry

end NUMINAMATH_GPT_subset_0_in_X_l374_37445


namespace NUMINAMATH_GPT_f_2015_l374_37471

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)

axiom periodic_f : ∀ x : ℝ, f (x - 2) = -f x

axiom f_interval : ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 0) → f x = 2 ^ x

theorem f_2015 : f 2015 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_f_2015_l374_37471


namespace NUMINAMATH_GPT_fraction_value_l374_37459

theorem fraction_value
  (m n : ℕ)
  (h : m / n = 2 / 3) :
  m / (m + n) = 2 / 5 :=
sorry

end NUMINAMATH_GPT_fraction_value_l374_37459


namespace NUMINAMATH_GPT_perimeter_of_stadium_l374_37402

-- Define the length and breadth as given conditions.
def length : ℕ := 100
def breadth : ℕ := 300

-- Define the perimeter function for a rectangle.
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Prove that the perimeter of the stadium is 800 meters given the length and breadth.
theorem perimeter_of_stadium : perimeter length breadth = 800 := 
by
  -- Placeholder for the formal proof.
  sorry

end NUMINAMATH_GPT_perimeter_of_stadium_l374_37402


namespace NUMINAMATH_GPT_keiko_speed_l374_37413

theorem keiko_speed (wA wB tA tB : ℝ) (v : ℝ)
    (h1: wA = 4)
    (h2: wB = 8)
    (h3: tA = 48)
    (h4: tB = 72)
    (h5: v = (24 * π) / 60) :
    v = 2 * π / 5 :=
by
  sorry

end NUMINAMATH_GPT_keiko_speed_l374_37413


namespace NUMINAMATH_GPT_limit_equivalence_l374_37476

open Nat
open Real

variable {u : ℕ → ℝ} {L : ℝ}

def original_def (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |L - u n| ≤ ε

def def1 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ε ≤ 0 ∨ (∃ N : ℕ, ∀ n : ℕ, n < N ∨ |L - u n| ≤ ε)

def def2 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∀ n : ℕ, ∃ N : ℕ, n ≥ N → |L - u n| ≤ ε

def def3 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, |L - u n| < ε

def def4 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∃ N : ℕ, ∀ ε > 0, ∀ n ≥ N, |L - u n| ≤ ε

theorem limit_equivalence :
  original_def u L ↔ def1 u L ∧ def3 u L ∧ ¬def2 u L ∧ ¬def4 u L :=
by
  sorry

end NUMINAMATH_GPT_limit_equivalence_l374_37476


namespace NUMINAMATH_GPT_sin_gamma_plus_delta_l374_37447

theorem sin_gamma_plus_delta (γ δ : ℝ) (hγ : Complex.exp (Complex.I * γ) = (4/5 : ℂ) + (3/5 : ℂ) * Complex.I)
                             (hδ : Complex.exp (Complex.I * δ) = (-5/13 : ℂ) + (12/13 : ℂ) * Complex.I) :
  Real.sin (γ + δ) = 33 / 65 :=
by
  sorry

end NUMINAMATH_GPT_sin_gamma_plus_delta_l374_37447


namespace NUMINAMATH_GPT_incorrect_expression_D_l374_37492

noncomputable def E : ℝ := sorry
def R : ℕ := sorry
def S : ℕ := sorry
def m : ℕ := sorry
def t : ℕ := sorry

-- E is a repeating decimal
-- R is the non-repeating part of E with m digits
-- S is the repeating part of E with t digits

theorem incorrect_expression_D : ¬ (10^m * (10^t - 1) * E = S * (R - 1)) :=
sorry

end NUMINAMATH_GPT_incorrect_expression_D_l374_37492


namespace NUMINAMATH_GPT_gcd_36_60_l374_37446

theorem gcd_36_60 : Nat.gcd 36 60 = 12 := by
  sorry

end NUMINAMATH_GPT_gcd_36_60_l374_37446


namespace NUMINAMATH_GPT_cost_of_cookbook_l374_37472

def cost_of_dictionary : ℕ := 11
def cost_of_dinosaur_book : ℕ := 19
def amount_saved : ℕ := 8
def amount_needed : ℕ := 29

theorem cost_of_cookbook :
  let total_cost := amount_saved + amount_needed
  let accounted_cost := cost_of_dictionary + cost_of_dinosaur_book
  total_cost - accounted_cost = 7 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_cookbook_l374_37472


namespace NUMINAMATH_GPT_option_A_option_D_l374_37463

variable {a : ℕ → ℤ} -- The arithmetic sequence
variable {S : ℕ → ℤ} -- Sum of the first n terms
variable {a1 d : ℤ} -- First term and common difference

-- Conditions for arithmetic sequence
axiom a_n (n : ℕ) : a n = a1 + ↑(n-1) * d
axiom S_n (n : ℕ) : S n = n * a1 + (n * (n - 1) / 2) * d
axiom condition : a 4 + 2 * a 8 = a 6

theorem option_A : a 7 = 0 :=
by
  -- Proof to be done
  sorry

theorem option_D : S 13 = 0 :=
by
  -- Proof to be done
  sorry

end NUMINAMATH_GPT_option_A_option_D_l374_37463


namespace NUMINAMATH_GPT_square_rem_1_mod_9_l374_37419

theorem square_rem_1_mod_9 (N : ℤ) (h : N % 9 = 1 ∨ N % 9 = 8) : (N * N) % 9 = 1 :=
by sorry

end NUMINAMATH_GPT_square_rem_1_mod_9_l374_37419


namespace NUMINAMATH_GPT_f_divides_f_2k_plus_1_f_coprime_f_multiple_l374_37448

noncomputable def f (g n : ℕ) : ℕ := g ^ n + 1

theorem f_divides_f_2k_plus_1 (g : ℕ) (k n : ℕ) :
  f g n ∣ f g ((2 * k + 1) * n) :=
by sorry

theorem f_coprime_f_multiple (g n : ℕ) :
  Nat.Coprime (f g n) (f g (2 * n)) ∧
  Nat.Coprime (f g n) (f g (4 * n)) ∧
  Nat.Coprime (f g n) (f g (6 * n)) :=
by sorry

end NUMINAMATH_GPT_f_divides_f_2k_plus_1_f_coprime_f_multiple_l374_37448


namespace NUMINAMATH_GPT_total_detergent_is_19_l374_37468

-- Define the quantities and usage of detergent
def detergent_per_pound_cotton := 2
def detergent_per_pound_woolen := 3
def detergent_per_pound_synthetic := 1

def pounds_of_cotton := 4
def pounds_of_woolen := 3
def pounds_of_synthetic := 2

-- Define the function to calculate the total amount of detergent needed
def total_detergent_needed := 
  detergent_per_pound_cotton * pounds_of_cotton +
  detergent_per_pound_woolen * pounds_of_woolen +
  detergent_per_pound_synthetic * pounds_of_synthetic

-- The theorem to prove the total amount of detergent used
theorem total_detergent_is_19 : total_detergent_needed = 19 :=
  by { sorry }

end NUMINAMATH_GPT_total_detergent_is_19_l374_37468


namespace NUMINAMATH_GPT_actual_cost_of_article_l374_37438

theorem actual_cost_of_article (x : ℝ) (h : 0.76 * x = 760) : x = 1000 :=
by 
  sorry

end NUMINAMATH_GPT_actual_cost_of_article_l374_37438


namespace NUMINAMATH_GPT_jimmy_cards_left_l374_37407

theorem jimmy_cards_left :
  ∀ (initial_cards jimmy_cards bob_cards mary_cards : ℕ),
    initial_cards = 18 →
    bob_cards = 3 →
    mary_cards = 2 * bob_cards →
    jimmy_cards = initial_cards - bob_cards - mary_cards →
    jimmy_cards = 9 := 
by
  intros initial_cards jimmy_cards bob_cards mary_cards h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_jimmy_cards_left_l374_37407


namespace NUMINAMATH_GPT_probability_divisible_by_3_l374_37411

theorem probability_divisible_by_3 (a b c : ℕ) (h : a ∈ Finset.range 2008 ∧ b ∈ Finset.range 2008 ∧ c ∈ Finset.range 2008) :
  (∃ p : ℚ, p = 1265/2007 ∧ (abc + ac + a) % 3 = 0) :=
sorry

end NUMINAMATH_GPT_probability_divisible_by_3_l374_37411


namespace NUMINAMATH_GPT_mogs_and_mags_to_migs_l374_37426

theorem mogs_and_mags_to_migs:
  (∀ mags migs, 1 * mags = 8 * migs) ∧ 
  (∀ mogs mags, 1 * mogs = 6 * mags) → 
  10 * (6 * 8) + 6 * 8 = 528 := by 
  sorry

end NUMINAMATH_GPT_mogs_and_mags_to_migs_l374_37426


namespace NUMINAMATH_GPT_length_of_AB_l374_37431

def parabola_eq (y : ℝ) : Prop := y^2 = 8 * y

def directrix_x : ℝ := 2

def dist_to_y_axis (E : ℝ × ℝ) : ℝ := E.1

theorem length_of_AB (A B F E : ℝ × ℝ)
  (p : parabola_eq A.2) (q : parabola_eq B.2) 
  (F_focus : F.1 = 2 ∧ F.2 = 0) 
  (midpoint_E : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (E_distance_from_y_axis : dist_to_y_axis E = 3) : 
  (abs (A.1 - B.1) + abs (A.2 - B.2)) = 10 := 
sorry

end NUMINAMATH_GPT_length_of_AB_l374_37431


namespace NUMINAMATH_GPT_increasing_interval_of_y_l374_37452

noncomputable def y (x : ℝ) : ℝ := (Real.log x) / x

theorem increasing_interval_of_y :
  ∃ (a b : ℝ), 0 < a ∧ a < e ∧ (∀ x : ℝ, a < x ∧ x < e → y x < y (x + ε)) :=
sorry

end NUMINAMATH_GPT_increasing_interval_of_y_l374_37452


namespace NUMINAMATH_GPT_population_net_increase_per_day_l374_37408

theorem population_net_increase_per_day (birth_rate death_rate : ℚ) (seconds_per_day : ℕ) (net_increase : ℚ) :
  birth_rate = 7 / 2 ∧
  death_rate = 2 / 2 ∧
  seconds_per_day = 24 * 60 * 60 ∧
  net_increase = (birth_rate - death_rate) * seconds_per_day →
  net_increase = 216000 := 
by
  sorry

end NUMINAMATH_GPT_population_net_increase_per_day_l374_37408


namespace NUMINAMATH_GPT_competition_scores_order_l374_37460

theorem competition_scores_order (A B C D : ℕ) (h1 : A + B = C + D) (h2 : C + A > D + B) (h3 : B > A + D) : (B > A) ∧ (A > C) ∧ (C > D) := 
by 
  sorry

end NUMINAMATH_GPT_competition_scores_order_l374_37460


namespace NUMINAMATH_GPT_integer_solutions_of_inequality_l374_37412

theorem integer_solutions_of_inequality :
  {x : ℤ | 3 ≤ 5 - 2 * x ∧ 5 - 2 * x ≤ 9} = {-2, -1, 0, 1} :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_of_inequality_l374_37412


namespace NUMINAMATH_GPT_find_z_l374_37443

theorem find_z (z : ℝ) (h : (z^2 - 5 * z + 6) / (z - 2) + (5 * z^2 + 11 * z - 32) / (5 * z - 16) = 1) : z = 1 :=
sorry

end NUMINAMATH_GPT_find_z_l374_37443


namespace NUMINAMATH_GPT_find_x_when_y_equals_2_l374_37456

theorem find_x_when_y_equals_2 :
  ∀ (y x k : ℝ),
  (y * (Real.sqrt x + 1) = k) →
  (y = 5 → x = 1 → k = 10) →
  (y = 2 → x = 16) := by
  intros y x k h_eq h_initial h_final
  sorry

end NUMINAMATH_GPT_find_x_when_y_equals_2_l374_37456


namespace NUMINAMATH_GPT_polynomial_difference_of_squares_l374_37414

theorem polynomial_difference_of_squares:
  (∀ a b : ℝ, ¬ ∃ x1 x2 : ℝ, a^2 + (-b)^2 = (x1 - x2) * (x1 + x2)) ∧
  (∀ m n : ℝ, ¬ ∃ x1 x2 : ℝ, 5 * m^2 - 20 * m * n = (x1 - x2) * (x1 + x2)) ∧
  (∀ x y : ℝ, ¬ ∃ x1 x2 : ℝ, -x^2 - y^2 = (x1 - x2) * (x1 + x2)) →
  ∃ x1 x2 : ℝ, -x^2 + 9 = (x1 - x2) * (x1 + x2) :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_difference_of_squares_l374_37414


namespace NUMINAMATH_GPT_annie_diorama_time_l374_37478

theorem annie_diorama_time (P B : ℕ) (h1 : B = 3 * P - 5) (h2 : B = 49) : P + B = 67 :=
sorry

end NUMINAMATH_GPT_annie_diorama_time_l374_37478


namespace NUMINAMATH_GPT_reaction_produces_correct_moles_l374_37480

-- Define the variables and constants
def moles_CO2 := 2
def moles_H2O := 2
def moles_H2CO3 := moles_CO2 -- based on the balanced reaction CO2 + H2O → H2CO3

-- The theorem we need to prove
theorem reaction_produces_correct_moles :
  moles_H2CO3 = 2 :=
by
  -- Mathematical reasoning goes here
  sorry

end NUMINAMATH_GPT_reaction_produces_correct_moles_l374_37480


namespace NUMINAMATH_GPT_correct_quotient_l374_37458

theorem correct_quotient (Q : ℤ) (D : ℤ) (h1 : D = 21 * Q) (h2 : D = 12 * 35) : Q = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_quotient_l374_37458


namespace NUMINAMATH_GPT_find_z_l374_37490

noncomputable def z := {z : ℂ | ∃ i : ℂ, i^2 = -1 ∧ i * z = i - 1}

theorem find_z (i : ℂ) (hi : i^2 = -1) : ∃ z : ℂ, i * z = i - 1 ∧ z = 1 + i := by
  use 1 + i
  sorry

end NUMINAMATH_GPT_find_z_l374_37490


namespace NUMINAMATH_GPT_problem_1_problem_2_l374_37444

theorem problem_1 (x : ℝ) : (2 * x + 3)^2 = 16 ↔ x = 1/2 ∨ x = -7/2 := by
  sorry

theorem problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l374_37444


namespace NUMINAMATH_GPT_find_m_l374_37442

open Complex

theorem find_m (m : ℝ) : (re ((1 + I) / (1 - I) + m * (1 - I) / (1 + I)) = ((1 + I) / (1 - I) + m * (1 - I) / (1 + I))) → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l374_37442


namespace NUMINAMATH_GPT_probability_of_specific_event_l374_37494

noncomputable def adam_probability := 1 / 5
noncomputable def beth_probability := 2 / 9
noncomputable def jack_probability := 1 / 6
noncomputable def jill_probability := 1 / 7
noncomputable def sandy_probability := 1 / 8

theorem probability_of_specific_event :
  (1 - adam_probability) * beth_probability * (1 - jack_probability) * jill_probability * sandy_probability = 1 / 378 := by
  sorry

end NUMINAMATH_GPT_probability_of_specific_event_l374_37494


namespace NUMINAMATH_GPT_initial_number_of_men_l374_37479

theorem initial_number_of_men (M : ℕ) (F : ℕ) (h1 : F = M * 20) (h2 : (M - 100) * 10 = M * 15) : 
  M = 200 :=
  sorry

end NUMINAMATH_GPT_initial_number_of_men_l374_37479


namespace NUMINAMATH_GPT_cistern_depth_l374_37424

noncomputable def length : ℝ := 9
noncomputable def width : ℝ := 4
noncomputable def total_wet_surface_area : ℝ := 68.5

theorem cistern_depth (h : ℝ) (h_def : 68.5 = 36 + 18 * h + 8 * h) : h = 1.25 :=
by sorry

end NUMINAMATH_GPT_cistern_depth_l374_37424


namespace NUMINAMATH_GPT_hyperbola_focal_point_k_l374_37457

theorem hyperbola_focal_point_k (k : ℝ) :
  (∃ (c : ℝ), c = 2 ∧ (5 : ℝ) * 2 ^ 2 - k * 0 ^ 2 = 5) →
  k = (5 : ℝ) / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_focal_point_k_l374_37457


namespace NUMINAMATH_GPT_jony_stop_block_correct_l374_37400

-- Jony's walk parameters
def start_time : ℕ := 7 -- In hours, but it is not used directly
def start_block : ℕ := 10
def end_block : ℕ := 90
def stop_time : ℕ := 40 -- Jony stops walking after 40 minutes starting from 07:00
def speed : ℕ := 100 -- meters per minute
def block_length : ℕ := 40 -- meters

-- Function to calculate the stop block given the parameters
def stop_block (start_block end_block stop_time speed block_length : ℕ) : ℕ :=
  let total_distance := stop_time * speed
  let outbound_distance := (end_block - start_block) * block_length
  let remaining_distance := total_distance - outbound_distance
  let blocks_walked_back := remaining_distance / block_length
  end_block - blocks_walked_back

-- The statement to prove
theorem jony_stop_block_correct :
  stop_block start_block end_block stop_time speed block_length = 70 :=
by
  sorry

end NUMINAMATH_GPT_jony_stop_block_correct_l374_37400


namespace NUMINAMATH_GPT_arithmetic_sequence_a14_eq_41_l374_37421

theorem arithmetic_sequence_a14_eq_41 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) 
  (h_a2 : a 2 = 5) 
  (h_a6 : a 6 = 17) : 
  a 14 = 41 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a14_eq_41_l374_37421
