import Mathlib

namespace NUMINAMATH_GPT_inequality_solution_set_l43_4306

theorem inequality_solution_set :
  {x : ℝ | (x^2 - 4) / (x^2 - 9) > 0} = {x : ℝ | x < -3 ∨ x > 3} :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l43_4306


namespace NUMINAMATH_GPT_evaluate_f_at_5_l43_4343

def f (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 38*x^2 - 35*x - 40

theorem evaluate_f_at_5 : f 5 = 110 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_5_l43_4343


namespace NUMINAMATH_GPT_sequence_8th_term_is_sqrt23_l43_4382

noncomputable def sequence_term (n : ℕ) : ℝ := Real.sqrt (2 + 3 * (n - 1))

theorem sequence_8th_term_is_sqrt23 : sequence_term 8 = Real.sqrt 23 :=
by
  sorry

end NUMINAMATH_GPT_sequence_8th_term_is_sqrt23_l43_4382


namespace NUMINAMATH_GPT_calculate_value_l43_4392

theorem calculate_value 
  (a : Int) (b : Int) (c : Real) (d : Real)
  (h1 : a = -1)
  (h2 : b = 2)
  (h3 : c * d = 1) :
  a + b - c * d = 0 := 
by
  sorry

end NUMINAMATH_GPT_calculate_value_l43_4392


namespace NUMINAMATH_GPT_greatest_divisor_l43_4398

theorem greatest_divisor (n : ℕ) (h1 : 1657 % n = 6) (h2 : 2037 % n = 5) : n = 127 :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_l43_4398


namespace NUMINAMATH_GPT_find_price_of_100_apples_l43_4388

noncomputable def price_of_100_apples (P : ℕ) : Prop :=
  (12000 / P) - (12000 / (P + 4)) = 5

theorem find_price_of_100_apples : price_of_100_apples 96 :=
by
  sorry

end NUMINAMATH_GPT_find_price_of_100_apples_l43_4388


namespace NUMINAMATH_GPT_natalia_apartment_number_unit_digit_l43_4360

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def true_statements (n : ℕ) : Prop :=
  (n % 3 = 0 → true) ∧   -- Statement (1): divisible by 3
  (∃ k : ℕ, k^2 = n → true) ∧  -- Statement (2): square number
  (n % 2 = 1 → true) ∧   -- Statement (3): odd
  (n % 10 = 4 → true)     -- Statement (4): ends in 4

def three_out_of_four_true (n : ℕ) : Prop :=
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 = 1 ∧ n % 10 ≠ 4) ∨
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 ≠ 1 ∧ n % 10 = 4) ∨
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 ≠ n) ∧ n % 2 = 1 ∧ n % 10 = 4) ∨
  (n % 3 ≠ 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 = 1 ∧ n % 10 = 4)

theorem natalia_apartment_number_unit_digit :
  ∀ n : ℕ, two_digit_number n → three_out_of_four_true n → n % 10 = 1 :=
by sorry

end NUMINAMATH_GPT_natalia_apartment_number_unit_digit_l43_4360


namespace NUMINAMATH_GPT_tunnel_length_l43_4311

theorem tunnel_length (L L_1 L_2 v v_new t t_new : ℝ) (H1: L_1 = 6) (H2: L_2 = 12) 
  (H3: v_new = 0.8 * v) (H4: t = (L + L_1) / v) (H5: t_new = 1.5 * t)
  (H6: t_new = (L + L_2) / v_new) : 
  L = 24 :=
by
  sorry

end NUMINAMATH_GPT_tunnel_length_l43_4311


namespace NUMINAMATH_GPT_absolute_difference_volumes_l43_4355

/-- The absolute difference in volumes of the cylindrical tubes formed by Amy and Carlos' papers. -/
theorem absolute_difference_volumes :
  let h_A := 12
  let C_A := 10
  let r_A := C_A / (2 * Real.pi)
  let V_A := Real.pi * r_A^2 * h_A
  let h_C := 8
  let C_C := 14
  let r_C := C_C / (2 * Real.pi)
  let V_C := Real.pi * r_C^2 * h_C
  abs (V_C - V_A) = 92 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_absolute_difference_volumes_l43_4355


namespace NUMINAMATH_GPT_rectangle_measurement_error_l43_4367

theorem rectangle_measurement_error 
  (L W : ℝ)
  (measured_length : ℝ := 1.05 * L)
  (measured_width : ℝ := 0.96 * W)
  (actual_area : ℝ := L * W)
  (calculated_area : ℝ := measured_length * measured_width)
  (error : ℝ := calculated_area - actual_area) :
  ((error / actual_area) * 100) = 0.8 :=
sorry

end NUMINAMATH_GPT_rectangle_measurement_error_l43_4367


namespace NUMINAMATH_GPT_yvonnes_probability_l43_4342

open Classical

variables (P_X P_Y P_Z : ℝ)

theorem yvonnes_probability
  (h1 : P_X = 1/5)
  (h2 : P_Z = 5/8)
  (h3 : P_X * P_Y * (1 - P_Z) = 0.0375) :
  P_Y = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_yvonnes_probability_l43_4342


namespace NUMINAMATH_GPT_quotient_is_76_l43_4357

def original_number : ℕ := 12401
def divisor : ℕ := 163
def remainder : ℕ := 13

theorem quotient_is_76 : (original_number - remainder) / divisor = 76 :=
by
  sorry

end NUMINAMATH_GPT_quotient_is_76_l43_4357


namespace NUMINAMATH_GPT_area_in_terms_of_diagonal_l43_4332

variables (l w d : ℝ)

-- Given conditions
def length_to_width_ratio := l / w = 5 / 2
def diagonal_relation := d^2 = l^2 + w^2

-- Proving the area is kd^2 with k = 10 / 29
theorem area_in_terms_of_diagonal 
    (ratio : length_to_width_ratio l w)
    (diag_rel : diagonal_relation l w d) :
  ∃ k, k = 10 / 29 ∧ (l * w = k * d^2) :=
sorry

end NUMINAMATH_GPT_area_in_terms_of_diagonal_l43_4332


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l43_4361

theorem sufficient_but_not_necessary_condition (x : ℝ) (p : Prop) (q : Prop)
  (h₁ : p ↔ (x^2 - 1 > 0)) (h₂ : q ↔ (x < -2)) :
  (¬p → ¬q) ∧ ¬(¬q → ¬p) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l43_4361


namespace NUMINAMATH_GPT_find_f_of_7_6_l43_4304

-- Definitions from conditions
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x k : ℤ, f (x + T * (k : ℝ)) = f x

def f_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x = x

-- The periodic function f with period 4
def f : ℝ → ℝ := sorry

-- Hypothesis
axiom f_periodic : periodic_function f 4
axiom f_on_interval : f_in_interval f

-- Theorem to prove
theorem find_f_of_7_6 : f 7.6 = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_7_6_l43_4304


namespace NUMINAMATH_GPT_negation_proposition_of_cube_of_odd_is_odd_l43_4383

def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem negation_proposition_of_cube_of_odd_is_odd :
  (¬ ∀ n : ℤ, odd n → odd (n^3)) ↔ (∃ n : ℤ, odd n ∧ ¬ odd (n^3)) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_of_cube_of_odd_is_odd_l43_4383


namespace NUMINAMATH_GPT_sum_of_cubes_divisible_l43_4384

theorem sum_of_cubes_divisible (a b c : ℤ) (h : (a + b + c) % 3 = 0) : 
  (a^3 + b^3 + c^3) % 3 = 0 := 
by sorry

end NUMINAMATH_GPT_sum_of_cubes_divisible_l43_4384


namespace NUMINAMATH_GPT_trevor_brother_age_l43_4368

theorem trevor_brother_age :
  ∃ B : ℕ, Trevor_current_age = 11 ∧
           Trevor_future_age = 24 ∧
           Brother_future_age = 3 * Trevor_current_age ∧
           B = Brother_future_age - (Trevor_future_age - Trevor_current_age) :=
sorry

end NUMINAMATH_GPT_trevor_brother_age_l43_4368


namespace NUMINAMATH_GPT_combined_weight_of_olivers_bags_l43_4375

theorem combined_weight_of_olivers_bags (w_james : ℕ) (w_oliver : ℕ) (w_combined : ℕ) 
  (h1 : w_james = 18) 
  (h2 : w_oliver = w_james / 6) 
  (h3 : w_combined = 2 * w_oliver) : 
  w_combined = 6 := 
by
  sorry

end NUMINAMATH_GPT_combined_weight_of_olivers_bags_l43_4375


namespace NUMINAMATH_GPT_eval_expression_l43_4320

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 9

theorem eval_expression : 2 * f 3 + 3 * f (-3) = 147 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l43_4320


namespace NUMINAMATH_GPT_probability_red_side_first_on_third_roll_l43_4348

noncomputable def red_side_probability_first_on_third_roll : ℚ :=
  let p_non_red := 7 / 10
  let p_red := 3 / 10
  (p_non_red * p_non_red * p_red)

theorem probability_red_side_first_on_third_roll :
  red_side_probability_first_on_third_roll = 147 / 1000 := 
sorry

end NUMINAMATH_GPT_probability_red_side_first_on_third_roll_l43_4348


namespace NUMINAMATH_GPT_average_pages_per_book_deshaun_l43_4303

-- Definitions related to the conditions
def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def person_closest_percentage : ℚ := 0.75
def second_person_daily_pages : ℕ := 180

-- Derived definitions
def second_person_total_pages : ℕ := second_person_daily_pages * summer_days
def deshaun_total_pages : ℚ := second_person_total_pages / person_closest_percentage

-- The final proof statement
theorem average_pages_per_book_deshaun : 
  deshaun_total_pages / deshaun_books = 320 := 
by
  -- We would provide the proof here
  sorry

end NUMINAMATH_GPT_average_pages_per_book_deshaun_l43_4303


namespace NUMINAMATH_GPT_ratio_apples_peaches_l43_4365

theorem ratio_apples_peaches (total_fruits oranges peaches apples : ℕ)
  (h_total : total_fruits = 56)
  (h_oranges : oranges = total_fruits / 4)
  (h_peaches : peaches = oranges / 2)
  (h_apples : apples = 35) : apples / peaches = 5 := 
by
  sorry

end NUMINAMATH_GPT_ratio_apples_peaches_l43_4365


namespace NUMINAMATH_GPT_find_modulus_z_l43_4336

open Complex

noncomputable def z_w_condition1 (z w : ℂ) : Prop := abs (3 * z - w) = 17
noncomputable def z_w_condition2 (z w : ℂ) : Prop := abs (z + 3 * w) = 4
noncomputable def z_w_condition3 (z w : ℂ) : Prop := abs (z + w) = 6

theorem find_modulus_z (z w : ℂ) (h1 : z_w_condition1 z w) (h2 : z_w_condition2 z w) (h3 : z_w_condition3 z w) :
  abs z = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_modulus_z_l43_4336


namespace NUMINAMATH_GPT_solve_for_x_l43_4371

theorem solve_for_x
  (x y : ℝ)
  (h1 : x + 2 * y = 100)
  (h2 : y = 25) :
  x = 50 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l43_4371


namespace NUMINAMATH_GPT_find_evening_tickets_l43_4387

noncomputable def matinee_price : ℕ := 5
noncomputable def evening_price : ℕ := 12
noncomputable def threeD_price : ℕ := 20
noncomputable def matinee_tickets : ℕ := 200
noncomputable def threeD_tickets : ℕ := 100
noncomputable def total_revenue : ℕ := 6600

theorem find_evening_tickets (E : ℕ) (hE : total_revenue = matinee_tickets * matinee_price + E * evening_price + threeD_tickets * threeD_price) :
  E = 300 :=
by
  sorry

end NUMINAMATH_GPT_find_evening_tickets_l43_4387


namespace NUMINAMATH_GPT_math_problem_l43_4346

theorem math_problem (x y : ℝ) (h : |x - 8 * y| + (4 * y - 1)^2 = 0) : (x + 2 * y)^3 = 125 / 8 := 
sorry

end NUMINAMATH_GPT_math_problem_l43_4346


namespace NUMINAMATH_GPT_garden_ratio_l43_4314

theorem garden_ratio 
  (P : ℕ) (L : ℕ) (W : ℕ) 
  (h1 : P = 900) 
  (h2 : L = 300) 
  (h3 : P = 2 * (L + W)) : 
  L / W = 2 :=
by 
  sorry

end NUMINAMATH_GPT_garden_ratio_l43_4314


namespace NUMINAMATH_GPT_solve_congruences_l43_4352

theorem solve_congruences :
  ∃ x : ℤ, 
  x ≡ 3 [ZMOD 7] ∧ 
  x^2 ≡ 44 [ZMOD 49] ∧ 
  x^3 ≡ 111 [ZMOD 343] ∧ 
  x ≡ 17 [ZMOD 343] :=
sorry

end NUMINAMATH_GPT_solve_congruences_l43_4352


namespace NUMINAMATH_GPT_fraction_of_power_l43_4316

theorem fraction_of_power (m : ℕ) (h : m = 16^1500) : m / 8 = 2^5997 := by
  sorry

end NUMINAMATH_GPT_fraction_of_power_l43_4316


namespace NUMINAMATH_GPT_four_distinct_real_roots_l43_4337

theorem four_distinct_real_roots (m : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 - 4 * |x| + 5 - m) ∧ ∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) ↔ (1 < m ∧ m < 5) :=
by
  sorry

end NUMINAMATH_GPT_four_distinct_real_roots_l43_4337


namespace NUMINAMATH_GPT_smallest_distance_l43_4300

open Complex

variable (z w : ℂ)

def a : ℂ := -2 - 4 * I
def b : ℂ := 5 + 6 * I

-- Conditions
def cond1 : Prop := abs (z + 2 + 4 * I) = 2
def cond2 : Prop := abs (w - 5 - 6 * I) = 4

-- Problem
theorem smallest_distance (h1 : cond1 z) (h2 : cond2 w) : abs (z - w) = Real.sqrt 149 - 6 :=
sorry

end NUMINAMATH_GPT_smallest_distance_l43_4300


namespace NUMINAMATH_GPT_point_D_sum_is_ten_l43_4318

noncomputable def D_coordinates_sum_eq_ten : Prop :=
  ∃ (D : ℝ × ℝ), (5, 5) = ( (7 + D.1) / 2, (3 + D.2) / 2 ) ∧ (D.1 + D.2 = 10)

theorem point_D_sum_is_ten : D_coordinates_sum_eq_ten :=
  sorry

end NUMINAMATH_GPT_point_D_sum_is_ten_l43_4318


namespace NUMINAMATH_GPT_problem_min_ineq_range_l43_4364

theorem problem_min_ineq_range (a b : ℝ) (x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∀ x, 1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|) ∧ (1 / a + 4 / b = 9) ∧ (-7 ≤ x ∧ x ≤ 11) :=
sorry

end NUMINAMATH_GPT_problem_min_ineq_range_l43_4364


namespace NUMINAMATH_GPT_total_initial_candles_l43_4326

-- Define the conditions
def used_candles : ℕ := 32
def leftover_candles : ℕ := 12

-- State the theorem
theorem total_initial_candles : used_candles + leftover_candles = 44 := by
  sorry

end NUMINAMATH_GPT_total_initial_candles_l43_4326


namespace NUMINAMATH_GPT_exists_multiple_with_odd_digit_sum_l43_4325

theorem exists_multiple_with_odd_digit_sum (M : Nat) :
  ∃ N : Nat, N % M = 0 ∧ (Nat.digits 10 N).sum % 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_multiple_with_odd_digit_sum_l43_4325


namespace NUMINAMATH_GPT_student_weight_l43_4344

theorem student_weight (S R : ℕ) (h1 : S - 5 = 2 * R) (h2 : S + R = 116) : S = 79 :=
sorry

end NUMINAMATH_GPT_student_weight_l43_4344


namespace NUMINAMATH_GPT_total_pages_in_book_l43_4308

-- Define the conditions
def pagesDay1To5 : Nat := 5 * 25
def pagesDay6To9 : Nat := 4 * 40
def pagesLastDay : Nat := 30

-- Total calculation
def totalPages (p1 p2 pLast : Nat) : Nat := p1 + p2 + pLast

-- The proof problem statement
theorem total_pages_in_book :
  totalPages pagesDay1To5 pagesDay6To9 pagesLastDay = 315 :=
  by
    sorry

end NUMINAMATH_GPT_total_pages_in_book_l43_4308


namespace NUMINAMATH_GPT_floor_e_eq_two_l43_4341

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 :=
by
  sorry

end NUMINAMATH_GPT_floor_e_eq_two_l43_4341


namespace NUMINAMATH_GPT_right_triangle_largest_side_l43_4312

theorem right_triangle_largest_side (b d : ℕ) (h_triangle : (b - d)^2 + b^2 = (b + d)^2)
  (h_arith_seq : (b - d) < b ∧ b < (b + d))
  (h_perimeter : (b - d) + b + (b + d) = 840) :
  (b + d = 350) :=
by sorry

end NUMINAMATH_GPT_right_triangle_largest_side_l43_4312


namespace NUMINAMATH_GPT_earn_2800_probability_l43_4317

def total_outcomes : ℕ := 7 ^ 4

def favorable_outcomes : ℕ :=
  (1 * 3 * 2 * 1) * 4 -- For each combination: \$1000, \$600, \$600, \$600; \$1000, \$1000, \$400, \$400; \$800, \$800, \$600, \$600; \$800, \$800, \$800, \$400

noncomputable def probability_of_earning_2800 : ℚ := favorable_outcomes / total_outcomes

theorem earn_2800_probability : probability_of_earning_2800 = 96 / 2401 := by
  sorry

end NUMINAMATH_GPT_earn_2800_probability_l43_4317


namespace NUMINAMATH_GPT_book_discount_l43_4338

theorem book_discount (a b : ℕ) (x y : ℕ) (h1 : x = 10 * a + b) (h2 : y = 10 * b + a) (h3 : (3 / 8) * x = y) :
  x - y = 45 := 
sorry

end NUMINAMATH_GPT_book_discount_l43_4338


namespace NUMINAMATH_GPT_luke_money_last_weeks_l43_4370

theorem luke_money_last_weeks (earnings_mowing : ℕ) (earnings_weed_eating : ℕ) (weekly_spending : ℕ) 
  (h1 : earnings_mowing = 9) (h2 : earnings_weed_eating = 18) (h3 : weekly_spending = 3) :
  (earnings_mowing + earnings_weed_eating) / weekly_spending = 9 :=
by sorry

end NUMINAMATH_GPT_luke_money_last_weeks_l43_4370


namespace NUMINAMATH_GPT_photo_area_with_frame_l43_4377

-- Define the areas and dimensions given in the conditions
def paper_length : ℕ := 12
def paper_width : ℕ := 8
def frame_width : ℕ := 2

-- Define the dimensions of the photo including the frame
def photo_length_with_frame : ℕ := paper_length + 2 * frame_width
def photo_width_with_frame : ℕ := paper_width + 2 * frame_width

-- The theorem statement proving the area of the wall photo including the frame
theorem photo_area_with_frame :
  (photo_length_with_frame * photo_width_with_frame) = 192 := by
  sorry

end NUMINAMATH_GPT_photo_area_with_frame_l43_4377


namespace NUMINAMATH_GPT_expected_value_of_win_is_correct_l43_4340

noncomputable def expected_value_of_win : ℝ :=
  (1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) +
  (1 / 8) * (8 - 4) + (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) +
  (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8)

theorem expected_value_of_win_is_correct :
  expected_value_of_win = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_win_is_correct_l43_4340


namespace NUMINAMATH_GPT_minimum_value_l43_4335

noncomputable def min_value (a b c d : ℝ) : ℝ :=
(a - c) ^ 2 + (b - d) ^ 2

theorem minimum_value (a b c d : ℝ) (hab : a * b = 3) (hcd : c + 3 * d = 0) :
  min_value a b c d ≥ (18 / 5) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l43_4335


namespace NUMINAMATH_GPT_line_slope_translation_l43_4374

theorem line_slope_translation (k : ℝ) (b : ℝ) :
  (∀ x y : ℝ, y = k * x + b → y = k * (x - 3) + (b + 2)) → k = 2 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_line_slope_translation_l43_4374


namespace NUMINAMATH_GPT_bottles_from_Shop_C_l43_4315

theorem bottles_from_Shop_C (TotalBottles ShopA ShopB ShopC : ℕ) 
  (h1 : TotalBottles = 550) 
  (h2 : ShopA = 150) 
  (h3 : ShopB = 180) 
  (h4 : TotalBottles = ShopA + ShopB + ShopC) : 
  ShopC = 220 := 
by
  sorry

end NUMINAMATH_GPT_bottles_from_Shop_C_l43_4315


namespace NUMINAMATH_GPT_truth_values_l43_4301

-- Define the region D as a set
def D (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 ≤ 4

-- Define propositions p and q
def p : Prop := ∀ x y, D x y → 2 * x + y ≤ 8
def q : Prop := ∃ x y, D x y ∧ 2 * x + y ≤ -1

-- State the propositions to be proven
def prop1 : Prop := p ∨ q
def prop2 : Prop := ¬p ∨ q
def prop3 : Prop := p ∧ ¬q
def prop4 : Prop := ¬p ∧ ¬q

-- State the main theorem asserting the truth values of the propositions
theorem truth_values : ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 :=
by
  sorry

end NUMINAMATH_GPT_truth_values_l43_4301


namespace NUMINAMATH_GPT_quadratic_roots_equal_l43_4353

theorem quadratic_roots_equal {k : ℝ} (h : (2 * k) ^ 2 - 4 * 1 * (k^2 + k + 3) = 0) : k^2 + k + 3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_equal_l43_4353


namespace NUMINAMATH_GPT_travel_same_direction_time_l43_4331

variable (A B : Type) [MetricSpace A] (downstream_speed upstream_speed : ℝ)
  (H_A_downstream_speed : downstream_speed = 8)
  (H_A_upstream_speed : upstream_speed = 4)
  (H_B_downstream_speed : downstream_speed = 8)
  (H_B_upstream_speed : upstream_speed = 4)
  (H_equal_travel_time : (∃ x : ℝ, x * downstream_speed + (3 - x) * upstream_speed = 3)
                      ∧ (∃ x : ℝ, x * upstream_speed + (3 - x) * downstream_speed = 3))

theorem travel_same_direction_time (A_α_downstream B_β_upstream A_α_upstream B_β_downstream : ℝ)
  (H_travel_time : (∃ x : ℝ, x = 1) ∧ (A_α_upstream = 3 - A_α_downstream) ∧ (B_β_downstream = 3 - B_β_upstream)) :
  A_α_downstream = 1 → A_α_upstream = 3 - 1 → B_β_downstream = 1 → B_β_upstream = 3 - 1 → ∃ t, t = 1 :=
by
  sorry

end NUMINAMATH_GPT_travel_same_direction_time_l43_4331


namespace NUMINAMATH_GPT_estimated_students_in_sport_A_correct_l43_4399

noncomputable def total_students_surveyed : ℕ := 80
noncomputable def students_in_sport_A_surveyed : ℕ := 30
noncomputable def total_school_population : ℕ := 800
noncomputable def proportion_sport_A : ℚ := students_in_sport_A_surveyed / total_students_surveyed
noncomputable def estimated_students_in_sport_A : ℚ := total_school_population * proportion_sport_A

theorem estimated_students_in_sport_A_correct :
  estimated_students_in_sport_A = 300 :=
by
  sorry

end NUMINAMATH_GPT_estimated_students_in_sport_A_correct_l43_4399


namespace NUMINAMATH_GPT_harmon_high_voting_l43_4334

theorem harmon_high_voting
  (U : Finset ℝ) -- Universe of students
  (A B : Finset ℝ) -- Sets of students favoring proposals
  (hU : U.card = 215)
  (hA : A.card = 170)
  (hB : B.card = 142)
  (hAcBc : (U \ (A ∪ B)).card = 38) :
  (A ∩ B).card = 135 :=
by {
  sorry
}

end NUMINAMATH_GPT_harmon_high_voting_l43_4334


namespace NUMINAMATH_GPT_find_n_l43_4330

theorem find_n :
  ∀ (n : ℕ),
    2^200 * 2^203 + 2^163 * 2^241 + 2^126 * 2^277 = 32^n →
    n = 81 :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_find_n_l43_4330


namespace NUMINAMATH_GPT_problem_l43_4379

variable (x y : ℝ)

theorem problem (h1 : x + 3 * y = 6) (h2 : x * y = -12) : x^2 + 9 * y^2 = 108 :=
sorry

end NUMINAMATH_GPT_problem_l43_4379


namespace NUMINAMATH_GPT_factorize_eq_l43_4373

theorem factorize_eq (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2) := 
by
  sorry

end NUMINAMATH_GPT_factorize_eq_l43_4373


namespace NUMINAMATH_GPT_benny_spent_amount_l43_4350

-- Definitions based on given conditions
def initial_amount : ℕ := 79
def amount_left : ℕ := 32

-- Proof problem statement
theorem benny_spent_amount :
  initial_amount - amount_left = 47 :=
sorry

end NUMINAMATH_GPT_benny_spent_amount_l43_4350


namespace NUMINAMATH_GPT_ratio_of_other_triangle_l43_4345

noncomputable def ratioAreaOtherTriangle (m : ℝ) : ℝ := 1 / (4 * m)

theorem ratio_of_other_triangle (m : ℝ) (h : m > 0) : ratioAreaOtherTriangle m = 1 / (4 * m) :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_ratio_of_other_triangle_l43_4345


namespace NUMINAMATH_GPT_Martiza_study_time_l43_4395

theorem Martiza_study_time :
  ∀ (x : ℕ),
  (30 * x + 30 * 25 = 20 * 60) →
  x = 15 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_Martiza_study_time_l43_4395


namespace NUMINAMATH_GPT_all_div_by_25_form_no_div_by_35_l43_4376

noncomputable def exists_div_by_25 (M : ℕ) : Prop :=
∃ (M N : ℕ) (n : ℕ), M = 6 * 10 ^ (n - 1) + N ∧ M = 25 * N ∧ 4 * N = 10 ^ (n - 1)

theorem all_div_by_25_form :
  ∀ M, exists_div_by_25 M → (∃ k : ℕ, M = 625 * 10 ^ k) :=
by
  intro M
  intro h
  sorry

noncomputable def not_exists_div_by_35 (M : ℕ) : Prop :=
∀ (M N : ℕ) (n : ℕ), M ≠ 6 * 10 ^ (n - 1) + N ∨ M ≠ 35 * N

theorem no_div_by_35 :
  ∀ M, not_exists_div_by_35 M :=
by
  intro M
  intro h
  sorry

end NUMINAMATH_GPT_all_div_by_25_form_no_div_by_35_l43_4376


namespace NUMINAMATH_GPT_total_ages_l43_4396

variable (Craig_age Mother_age : ℕ)

theorem total_ages (h1 : Craig_age = 16) (h2 : Mother_age = Craig_age + 24) : Craig_age + Mother_age = 56 := by
  sorry

end NUMINAMATH_GPT_total_ages_l43_4396


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_problem_5_l43_4347

theorem problem_1 : 286 = 200 + 80 + 6 := sorry
theorem problem_2 : 7560 = 7000 + 500 + 60 := sorry
theorem problem_3 : 2048 = 2000 + 40 + 8 := sorry
theorem problem_4 : 8009 = 8000 + 9 := sorry
theorem problem_5 : 3070 = 3000 + 70 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_problem_5_l43_4347


namespace NUMINAMATH_GPT_maria_trip_distance_l43_4394

theorem maria_trip_distance
  (D : ℝ)
  (h1 : D/2 = D/8 + 210) :
  D = 560 :=
sorry

end NUMINAMATH_GPT_maria_trip_distance_l43_4394


namespace NUMINAMATH_GPT_employee_payment_correct_l43_4372

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the percentage markup for retail price
def markup_percentage : ℝ := 0.20

-- Define the retail_price based on wholesale cost and markup percentage
def retail_price : ℝ := wholesale_cost + (markup_percentage * wholesale_cost)

-- Define the employee discount percentage
def discount_percentage : ℝ := 0.20

-- Define the discount amount based on retail price and discount percentage
def discount_amount : ℝ := retail_price * discount_percentage

-- Define the final price the employee pays after applying the discount
def employee_price : ℝ := retail_price - discount_amount

-- State the theorem to prove
theorem employee_payment_correct :
  employee_price = 192 :=
  by
    sorry

end NUMINAMATH_GPT_employee_payment_correct_l43_4372


namespace NUMINAMATH_GPT_calc_correct_operation_l43_4393

theorem calc_correct_operation (a : ℕ) :
  (2 : ℕ) * a + (3 : ℕ) * a = (5 : ℕ) * a :=
by
  -- Proof
  sorry

end NUMINAMATH_GPT_calc_correct_operation_l43_4393


namespace NUMINAMATH_GPT_general_term_sequence_l43_4390

theorem general_term_sequence (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 2)
  (h2 : ∀ (m : ℕ), m ≥ 2 → a m - a (m - 1) + 1 = 0) : 
  a n = 3 - n :=
sorry

end NUMINAMATH_GPT_general_term_sequence_l43_4390


namespace NUMINAMATH_GPT_abc_inequality_l43_4385

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a^2 < 16 * b * c) (h2 : b^2 < 16 * c * a) (h3 : c^2 < 16 * a * b) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) :=
by sorry

end NUMINAMATH_GPT_abc_inequality_l43_4385


namespace NUMINAMATH_GPT_maximal_s_value_l43_4358

noncomputable def max_tiles_sum (a b c : ℕ) : ℕ := a + c

theorem maximal_s_value :
  ∃ s : ℕ, 
    ∃ a b c : ℕ, 
      4 * a + 4 * c + 5 * b = 3986000 ∧ 
      s = max_tiles_sum a b c ∧ 
      s = 996500 := 
    sorry

end NUMINAMATH_GPT_maximal_s_value_l43_4358


namespace NUMINAMATH_GPT_tan_sum_product_l43_4351

theorem tan_sum_product (tan : ℝ → ℝ) : 
  (1 + tan 23) * (1 + tan 22) = 2 + tan 23 * tan 22 := by sorry

end NUMINAMATH_GPT_tan_sum_product_l43_4351


namespace NUMINAMATH_GPT_cards_per_box_l43_4305

-- Define the conditions
def total_cards : ℕ := 75
def cards_not_in_box : ℕ := 5
def boxes_given_away : ℕ := 2
def boxes_left : ℕ := 5

-- Calculating the total number of boxes initially
def initial_boxes : ℕ := boxes_given_away + boxes_left

-- Define the number of cards in each box
def num_cards_per_box (number_of_cards : ℕ) (number_of_boxes : ℕ) : ℕ :=
  (number_of_cards - cards_not_in_box) / number_of_boxes

-- The proof problem statement
theorem cards_per_box :
  num_cards_per_box total_cards initial_boxes = 10 :=
by
  -- Proof is omitted with sorry
  sorry

end NUMINAMATH_GPT_cards_per_box_l43_4305


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_eq_17_div_8_l43_4313

theorem sum_of_reciprocals_of_roots_eq_17_div_8 :
  ∀ p q : ℝ, (p + q = 17) → (p * q = 8) → (1 / p + 1 / q = 17 / 8) :=
by
  intros p q h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_eq_17_div_8_l43_4313


namespace NUMINAMATH_GPT_shortest_side_of_triangle_with_medians_l43_4339

noncomputable def side_lengths_of_triangle_with_medians (a b c m_a m_b m_c : ℝ) : Prop :=
  m_a = 3 ∧ m_b = 4 ∧ m_c = 5 →
  a^2 = 2*b^2 + 2*c^2 - 36 ∧
  b^2 = 2*a^2 + 2*c^2 - 64 ∧
  c^2 = 2*a^2 + 2*b^2 - 100

theorem shortest_side_of_triangle_with_medians :
  ∀ (a b c : ℝ), side_lengths_of_triangle_with_medians a b c 3 4 5 → 
  min a (min b c) = c :=
sorry

end NUMINAMATH_GPT_shortest_side_of_triangle_with_medians_l43_4339


namespace NUMINAMATH_GPT_four_digit_numbers_with_property_l43_4349

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_with_property_l43_4349


namespace NUMINAMATH_GPT_opposite_quotient_l43_4356

theorem opposite_quotient {a b : ℝ} (h1 : a ≠ b) (h2 : a = -b) : a / b = -1 := 
sorry

end NUMINAMATH_GPT_opposite_quotient_l43_4356


namespace NUMINAMATH_GPT_chair_and_desk_prices_l43_4328

theorem chair_and_desk_prices (c d : ℕ) 
  (h1 : c + d = 115)
  (h2 : d - c = 45) :
  c = 35 ∧ d = 80 := 
by
  sorry

end NUMINAMATH_GPT_chair_and_desk_prices_l43_4328


namespace NUMINAMATH_GPT_number_of_integers_l43_4381

theorem number_of_integers (n : ℕ) (h₁ : 300 < n^2) (h₂ : n^2 < 1200) : ∃ k, k = 17 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integers_l43_4381


namespace NUMINAMATH_GPT_NineChaptersProblem_l43_4391

-- Conditions: Assign the given conditions to variables
variables (x y : Int)
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Proof problem: Prove that the system of equations is consistent with the given conditions
theorem NineChaptersProblem : condition1 x y ∧ condition2 x y := sorry

end NUMINAMATH_GPT_NineChaptersProblem_l43_4391


namespace NUMINAMATH_GPT_find_B_l43_4322

theorem find_B (B: ℕ) (h1: 5457062 % 2 = 0 ∧ 200 * B % 4 = 0) (h2: 5457062 % 5 = 0 ∧ B % 5 = 0) (h3: 5450062 % 8 = 0 ∧ 100 * B % 8 = 0) : B = 0 :=
sorry

end NUMINAMATH_GPT_find_B_l43_4322


namespace NUMINAMATH_GPT_find_a_l43_4378

-- Define point
structure Point where
  x : ℝ
  y : ℝ

-- Define curves
def C1 (a x : ℝ) : ℝ := a * x^3 + 1
def C2 (P : Point) : Prop := P.x^2 + P.y^2 = 5 / 2

-- Define the tangent slope function for curve C1
def tangent_slope_C1 (a x : ℝ) : ℝ := 3 * a * x^2

-- State the problem that we need to prove
theorem find_a (a x₀ y₀ : ℝ) (h1 : y₀ = C1 a x₀) (h2 : C2 ⟨x₀, y₀⟩) (h3 : y₀ = 3 * a * x₀^3) 
  (ha_pos : 0 < a) : a = 4 := 
  by
    sorry

end NUMINAMATH_GPT_find_a_l43_4378


namespace NUMINAMATH_GPT_factorization1_factorization2_l43_4327

theorem factorization1 (x y : ℝ) : 4 - 12 * (x - y) + 9 * (x - y)^2 = (2 - 3 * x + 3 * y)^2 :=
by
  sorry

theorem factorization2 (x : ℝ) (a : ℝ) : 2 * a * (x^2 + 1)^2 - 8 * a * x^2 = 2 * a * (x - 1)^2 * (x + 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorization1_factorization2_l43_4327


namespace NUMINAMATH_GPT_equal_money_distribution_l43_4310

theorem equal_money_distribution (y : ℝ) : 
  ∃ z : ℝ, z = 0.1 * (1.25 * y) ∧ (1.25 * y) - z = y + z - y :=
by
  sorry

end NUMINAMATH_GPT_equal_money_distribution_l43_4310


namespace NUMINAMATH_GPT_white_square_area_l43_4302

theorem white_square_area
  (edge_length : ℝ)
  (total_green_area : ℝ)
  (faces : ℕ)
  (green_per_face : ℝ)
  (total_surface_area : ℝ)
  (white_area_per_face : ℝ) :
  edge_length = 12 ∧ total_green_area = 432 ∧ faces = 6 ∧ total_surface_area = 864 ∧ green_per_face = total_green_area / faces ∧ white_area_per_face = total_surface_area / faces - green_per_face → white_area_per_face = 72 :=
by
  sorry

end NUMINAMATH_GPT_white_square_area_l43_4302


namespace NUMINAMATH_GPT_graph_properties_l43_4333

theorem graph_properties (x : ℝ) :
  (∃ p : ℝ × ℝ, p = (1, -7) ∧ y = -7 * x) ∧
  (x ≠ 0 → y * x < 0) ∧
  (x > 0 → y < 0) :=
by
  sorry

end NUMINAMATH_GPT_graph_properties_l43_4333


namespace NUMINAMATH_GPT_find_z_l43_4354

theorem find_z (a z : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * 25 * 45 * z) : z = 49 :=
sorry

end NUMINAMATH_GPT_find_z_l43_4354


namespace NUMINAMATH_GPT_distance_between_foci_l43_4359

-- Define the conditions
def is_asymptote (y x : ℝ) (slope intercept : ℝ) : Prop := y = slope * x + intercept

def passes_through_point (x y x0 y0 : ℝ) : Prop := x = x0 ∧ y = y0

-- The hyperbola conditions
axiom asymptote1 : ∀ x y : ℝ, is_asymptote y x 2 3
axiom asymptote2 : ∀ x y : ℝ, is_asymptote y x (-2) 5
axiom hyperbola_passes : passes_through_point 2 9 2 9

-- The proof problem statement: distance between the foci
theorem distance_between_foci : ∀ {a b c : ℝ}, ∃ c, (c^2 = 22.75 + 22.75) → 2 * c = 2 * Real.sqrt 45.5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_l43_4359


namespace NUMINAMATH_GPT_sum_of_extreme_T_l43_4362

theorem sum_of_extreme_T (B M T : ℝ) 
  (h1 : B^2 + M^2 + T^2 = 2022)
  (h2 : B + M + T = 72) :
  ∃ Tmin Tmax, Tmin + Tmax = 48 ∧ Tmin ≤ T ∧ T ≤ Tmax :=
by
  sorry

end NUMINAMATH_GPT_sum_of_extreme_T_l43_4362


namespace NUMINAMATH_GPT_find_rate_of_current_l43_4369

noncomputable def rate_of_current : ℝ := 
  let speed_still_water := 42
  let distance_downstream := 33.733333333333334
  let time_hours := 44 / 60
  (distance_downstream / time_hours) - speed_still_water

theorem find_rate_of_current : rate_of_current = 4 :=
by sorry

end NUMINAMATH_GPT_find_rate_of_current_l43_4369


namespace NUMINAMATH_GPT_multiplication_factor_l43_4323

theorem multiplication_factor 
  (avg1 : ℕ → ℕ → ℕ)
  (avg2 : ℕ → ℕ → ℕ)
  (sum1 : ℕ)
  (num1 : ℕ)
  (num2 : ℕ)
  (sum2 : ℕ)
  (factor : ℚ) :
  avg1 sum1 num1 = 7 →
  avg2 sum2 num2 = 84 →
  sum1 = 10 * 7 →
  sum2 = 10 * 84 →
  factor = sum2 / sum1 →
  factor = 12 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_factor_l43_4323


namespace NUMINAMATH_GPT_students_with_both_l43_4380

-- Define the problem conditions as given in a)
def total_students : ℕ := 50
def students_with_bike : ℕ := 28
def students_with_scooter : ℕ := 35

-- State the theorem
theorem students_with_both :
  ∃ (n : ℕ), n = 13 ∧ total_students = students_with_bike + students_with_scooter - n := by
  sorry

end NUMINAMATH_GPT_students_with_both_l43_4380


namespace NUMINAMATH_GPT_XiaoYing_minimum_water_usage_l43_4309

-- Definitions based on the problem's conditions
def first_charge_rate : ℝ := 2.8
def excess_charge_rate : ℝ := 3
def initial_threshold : ℝ := 5
def minimum_bill : ℝ := 29

-- Main statement for the proof based on the derived inequality
theorem XiaoYing_minimum_water_usage (x : ℝ) (h1 : 2.8 * initial_threshold + 3 * (x - initial_threshold) ≥ 29) : x ≥ 10 := by
  sorry

end NUMINAMATH_GPT_XiaoYing_minimum_water_usage_l43_4309


namespace NUMINAMATH_GPT_range_of_m_l43_4397

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + 5 < 4 * x - 1 ∧ x > m → x > 2) → m ≤ 2 :=
by
  intro h
  have h₁ := h 2
  sorry

end NUMINAMATH_GPT_range_of_m_l43_4397


namespace NUMINAMATH_GPT_ellipse_equation_point_M_exists_l43_4307

-- Condition: Point (1, sqrt(2)/2) lies on the ellipse
def point_lies_on_ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
    (a_gt_b : a > b) : Prop :=
  (1, Real.sqrt 2 / 2).fst^2 / a^2 + (1, Real.sqrt 2 / 2).snd^2 / b^2 = 1

-- Condition: Eccentricity of the ellipse is sqrt(2)/2
def eccentricity_condition (a b : ℝ) (c : ℝ) : Prop :=
  c / a = Real.sqrt 2 / 2 ∧ a^2 = b^2 + c^2

-- Question (I): Equation of ellipse should be (x^2 / 2 + y^2 = 1)
theorem ellipse_equation (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (a_gt_b : a > b) (h : point_lies_on_ellipse a b a_pos b_pos a_gt_b)
    (h_ecc : eccentricity_condition a b c) : a = Real.sqrt 2 ∧ b = 1 := 
sorry

-- Question (II): There exists M such that MA · MB is constant
theorem point_M_exists (a b c x0 : ℝ)
    (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b) 
    (a_val : a = Real.sqrt 2) (b_val : b = 1) 
    (h : point_lies_on_ellipse a b a_pos b_pos a_gt_b)
    (h_ecc : eccentricity_condition a b c) : 
    ∃ (M : ℝ × ℝ), M.fst = 5 / 4 ∧ M.snd = 0 ∧ -7 / 16 = -7 / 16 := 
sorry

end NUMINAMATH_GPT_ellipse_equation_point_M_exists_l43_4307


namespace NUMINAMATH_GPT_probability_red_then_white_l43_4329

-- Define the total number of balls and the probabilities
def total_balls : ℕ := 9
def red_balls : ℕ := 3
def white_balls : ℕ := 2

-- Define the probabilities
def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

-- Define the combined probability of drawing a red and then a white ball 
theorem probability_red_then_white : (prob_red * prob_white) = 2/27 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_then_white_l43_4329


namespace NUMINAMATH_GPT_directrix_of_parabola_l43_4324

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = -3 * x^2 + 6 * x - 5 → y = -35 / 18 := by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l43_4324


namespace NUMINAMATH_GPT_find_principal_l43_4389

-- Problem conditions
variables (SI : ℚ := 4016.25) 
variables (R : ℚ := 0.08) 
variables (T : ℚ := 5)

-- The simple interest formula to find Principal
noncomputable def principal (SI : ℚ) (R : ℚ) (T : ℚ) : ℚ := SI * 100 / (R * T)

-- Lean statement to prove
theorem find_principal : principal SI R T = 10040.625 := by
  sorry

end NUMINAMATH_GPT_find_principal_l43_4389


namespace NUMINAMATH_GPT_container_capacity_l43_4363

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 36 = 0.75 * C) : 
  C = 80 :=
sorry

end NUMINAMATH_GPT_container_capacity_l43_4363


namespace NUMINAMATH_GPT_range_of_a_l43_4386

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x + x^2 - 2 * a * x

theorem range_of_a (a : ℝ) (h₀ : a > 0) 
  (h₁ h₂ : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ a - f x₂ a ≥ (3/2) - 2 * Real.log 2) : 
  a ≥ (3/2) * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l43_4386


namespace NUMINAMATH_GPT_roots_quadratic_solution_l43_4366

theorem roots_quadratic_solution (α β : ℝ) (hα : α^2 - 3*α - 2 = 0) (hβ : β^2 - 3*β - 2 = 0) :
  3*α^3 + 8*β^4 = 1229 := by
  sorry

end NUMINAMATH_GPT_roots_quadratic_solution_l43_4366


namespace NUMINAMATH_GPT_rongrong_bike_speed_l43_4321

theorem rongrong_bike_speed :
  ∃ (x : ℝ), (15 / x - 15 / (4 * x) = 45 / 60) → x = 15 :=
by
  sorry

end NUMINAMATH_GPT_rongrong_bike_speed_l43_4321


namespace NUMINAMATH_GPT_total_gain_percentage_combined_l43_4319

theorem total_gain_percentage_combined :
  let CP1 := 20
  let CP2 := 35
  let CP3 := 50
  let SP1 := 25
  let SP2 := 44
  let SP3 := 65
  let totalCP := CP1 + CP2 + CP3
  let totalSP := SP1 + SP2 + SP3
  let totalGain := totalSP - totalCP
  let gainPercentage := (totalGain / totalCP) * 100
  gainPercentage = 27.62 :=
by sorry

end NUMINAMATH_GPT_total_gain_percentage_combined_l43_4319
