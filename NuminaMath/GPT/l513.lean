import Mathlib

namespace NUMINAMATH_GPT_no_3_digit_number_with_digit_sum_27_and_even_l513_51360

-- Define what it means for a number to be 3-digit
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the digit-sum function
def digitSum (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

-- Define what it means for a number to be even
def isEven (n : ℕ) : Prop := n % 2 = 0

-- State the proof problem
theorem no_3_digit_number_with_digit_sum_27_and_even :
  ∀ n : ℕ, isThreeDigit n → digitSum n = 27 → isEven n → false :=
by
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_no_3_digit_number_with_digit_sum_27_and_even_l513_51360


namespace NUMINAMATH_GPT_nth_equation_proof_l513_51313

theorem nth_equation_proof (n : ℕ) (hn : n > 0) :
  (1 : ℝ) + (1 / (n : ℝ)) - (2 / (2 * n - 1)) = (2 * n^2 + n + 1) / (n * (2 * n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_proof_l513_51313


namespace NUMINAMATH_GPT_infinite_series_sum_l513_51322

theorem infinite_series_sum :
  (∑' n : ℕ, n * (1/5)^n) = 5/16 :=
by sorry

end NUMINAMATH_GPT_infinite_series_sum_l513_51322


namespace NUMINAMATH_GPT_circle_condition_m_l513_51336

theorem circle_condition_m (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2 * x + m = 0) → m < 1 := 
by
  sorry

end NUMINAMATH_GPT_circle_condition_m_l513_51336


namespace NUMINAMATH_GPT_all_zero_l513_51341

def circle_condition (x : Fin 2007 → ℤ) : Prop :=
  ∀ i : Fin 2007, x i + x (i+1) + x (i+2) + x (i+3) + x (i+4) = 2 * (x (i+1) + x (i+2)) + 2 * (x (i+3) + x (i+4))

theorem all_zero (x : Fin 2007 → ℤ) (h : circle_condition x) : ∀ i, x i = 0 :=
sorry

end NUMINAMATH_GPT_all_zero_l513_51341


namespace NUMINAMATH_GPT_value_of_a_l513_51316

theorem value_of_a (k : ℝ) (a : ℝ) (b : ℝ) (h1 : a = k / b^2) (h2 : a = 10) (h3 : b = 24) :
  a = 40 :=
sorry

end NUMINAMATH_GPT_value_of_a_l513_51316


namespace NUMINAMATH_GPT_parallel_line_through_A_is_2x_3y_minus_15_line_with_twice_slope_angle_l513_51385

open Real

-- Conditions:
def l1 (x y : ℝ) : Prop := x - 2 * y + 3 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y - 9 = 0
def intersection_point (x y : ℝ) : Prop := l1 x y ∧ l2 x y

-- Point A is the intersection of l1 and l2
def A : ℝ × ℝ := ⟨3, 3⟩

-- Question 1
def line_parallel (x y : ℝ) (c : ℝ) : Prop := 2 * x + 3 * y + c = 0
def line_parallel_passing_through_A : Prop := line_parallel A.1 A.2 (-15)

theorem parallel_line_through_A_is_2x_3y_minus_15 : line_parallel_passing_through_A :=
sorry

-- Question 2
def slope_angle (tan_alpha : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ y, ∃ x, y ≠ 0 ∧ l x y ∧ (tan_alpha = x / y)

def required_slope (tan_alpha : ℝ) : Prop :=
  tan_alpha = 4 / 3

def line_with_slope (x y slope : ℝ) : Prop :=
  y - A.2 = slope * (x - A.1)

def line_with_required_slope : Prop := 
  line_with_slope A.1 A.2 (4 / 3)

theorem line_with_twice_slope_angle : line_with_required_slope :=
sorry

end NUMINAMATH_GPT_parallel_line_through_A_is_2x_3y_minus_15_line_with_twice_slope_angle_l513_51385


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l513_51374

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_and_evaluate_expression :
  ((x + 1) / (x^2 + 2 * x + 1)) / (1 - (2 / (x + 1))) = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l513_51374


namespace NUMINAMATH_GPT_time_left_for_nap_l513_51370

noncomputable def total_time : ℝ := 20
noncomputable def first_train_time : ℝ := 2 + 1
noncomputable def second_train_time : ℝ := 3 + 1
noncomputable def transfer_one_time : ℝ := 0.75 + 0.5
noncomputable def third_train_time : ℝ := 2 + 1
noncomputable def transfer_two_time : ℝ := 1
noncomputable def fourth_train_time : ℝ := 1
noncomputable def transfer_three_time : ℝ := 0.5
noncomputable def fifth_train_time_before_nap : ℝ := 1.5

noncomputable def total_activities_time : ℝ :=
  first_train_time +
  second_train_time +
  transfer_one_time +
  third_train_time +
  transfer_two_time +
  fourth_train_time +
  transfer_three_time +
  fifth_train_time_before_nap

theorem time_left_for_nap : total_time - total_activities_time = 4.75 := by
  sorry

end NUMINAMATH_GPT_time_left_for_nap_l513_51370


namespace NUMINAMATH_GPT_parallelepiped_vectors_l513_51306

theorem parallelepiped_vectors (x y z : ℝ)
  (h1: ∀ (AB BC CC1 AC1 : ℝ), AC1 = AB + BC + CC1)
  (h2: ∀ (AB BC CC1 AC1 : ℝ), AC1 = x * AB + 2 * y * BC + 3 * z * CC1) :
  x + y + z = 11 / 6 :=
by
  -- This is where the proof would go, but as per the instruction we'll add sorry.
  sorry

end NUMINAMATH_GPT_parallelepiped_vectors_l513_51306


namespace NUMINAMATH_GPT_number_2018_location_l513_51324

-- Define the odd square pattern as starting positions of rows
def odd_square (k : ℕ) : ℕ := (2 * k - 1) ^ 2

-- Define the conditions in terms of numbers in each row
def start_of_row (n : ℕ) : ℕ := (2 * n - 1) ^ 2 + 1

def number_at_row_column (n m : ℕ) :=
  start_of_row n + (m - 1)

theorem number_2018_location :
  number_at_row_column 44 82 = 2018 :=
by
  sorry

end NUMINAMATH_GPT_number_2018_location_l513_51324


namespace NUMINAMATH_GPT_geometric_sequence_ratios_l513_51380

theorem geometric_sequence_ratios {n : ℕ} {r : ℝ}
  (h1 : 85 = (1 - r^(2*n)) / (1 - r^2))
  (h2 : 170 = r * 85) :
  r = 2 ∧ 2*n = 8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratios_l513_51380


namespace NUMINAMATH_GPT_lottery_blanks_l513_51364

theorem lottery_blanks (P B : ℕ) (h₁ : P = 10) (h₂ : (P : ℝ) / (P + B) = 0.2857142857142857) : B = 25 := 
by
  sorry

end NUMINAMATH_GPT_lottery_blanks_l513_51364


namespace NUMINAMATH_GPT_simplify_fraction_l513_51340

theorem simplify_fraction : 
  ∃ (c d : ℤ), ((∀ m : ℤ, (6 * m + 12) / 3 = c * m + d) ∧ c = 2 ∧ d = 4) → 
  c / d = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l513_51340


namespace NUMINAMATH_GPT_sample_frequency_in_range_l513_51328

theorem sample_frequency_in_range :
  let total_capacity := 100
  let freq_0_10 := 12
  let freq_10_20 := 13
  let freq_20_30 := 24
  let freq_30_40 := 15
  (freq_0_10 + freq_10_20 + freq_20_30 + freq_30_40) / total_capacity = 0.64 :=
by
  sorry

end NUMINAMATH_GPT_sample_frequency_in_range_l513_51328


namespace NUMINAMATH_GPT_non_zero_number_is_9_l513_51333

theorem non_zero_number_is_9 (x : ℝ) (hx : x ≠ 0) (h : (x + x^2) / 2 = 5 * x) : x = 9 :=
sorry

end NUMINAMATH_GPT_non_zero_number_is_9_l513_51333


namespace NUMINAMATH_GPT_ratio_daves_bench_to_weight_l513_51338

variables (wD bM bD bC : ℝ)

def daves_weight := wD = 175
def marks_bench_press := bM = 55
def marks_comparison_to_craig := bM = bC - 50
def craigs_comparison_to_dave := bC = 0.20 * bD

theorem ratio_daves_bench_to_weight
  (h1 : daves_weight wD)
  (h2 : marks_bench_press bM)
  (h3 : marks_comparison_to_craig bM bC)
  (h4 : craigs_comparison_to_dave bC bD) :
  (bD / wD) = 3 :=
by
  rw [daves_weight] at h1
  rw [marks_bench_press] at h2
  rw [marks_comparison_to_craig] at h3
  rw [craigs_comparison_to_dave] at h4
  -- Now we have:
  -- 1. wD = 175
  -- 2. bM = 55
  -- 3. bM = bC - 50
  -- 4. bC = 0.20 * bD
  -- We proceed to solve:
  sorry

end NUMINAMATH_GPT_ratio_daves_bench_to_weight_l513_51338


namespace NUMINAMATH_GPT_optimal_room_rate_to_maximize_income_l513_51392

noncomputable def max_income (x : ℝ) : ℝ := x * (300 - 0.5 * (x - 200))

theorem optimal_room_rate_to_maximize_income :
  ∀ x, 200 ≤ x → x ≤ 800 → max_income x ≤ max_income 400 :=
by
  sorry

end NUMINAMATH_GPT_optimal_room_rate_to_maximize_income_l513_51392


namespace NUMINAMATH_GPT_circle_radius_order_l513_51379

theorem circle_radius_order (r_X r_Y r_Z : ℝ)
  (hX : r_X = π)
  (hY : 2 * π * r_Y = 8 * π)
  (hZ : π * r_Z^2 = 9 * π) :
  r_Z < r_X ∧ r_X < r_Y :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_radius_order_l513_51379


namespace NUMINAMATH_GPT_boys_went_down_the_slide_total_l513_51323

/-- Conditions -/
def a : Nat := 87
def b : Nat := 46
def c : Nat := 29

/-- The main proof problem -/
theorem boys_went_down_the_slide_total :
  a + b + c = 162 :=
by
  sorry

end NUMINAMATH_GPT_boys_went_down_the_slide_total_l513_51323


namespace NUMINAMATH_GPT_points_distance_le_sqrt5_l513_51393

theorem points_distance_le_sqrt5 :
  ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (0 ≤ (points i).1 ∧ (points i).1 ≤ 4) ∧ (0 ≤ (points i).2 ∧ (points i).2 ≤ 3)) →
  ∃ (i j : Fin 6), i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_points_distance_le_sqrt5_l513_51393


namespace NUMINAMATH_GPT_problem1_problem2_l513_51315

theorem problem1 (x : ℚ) (h : x - 2/11 = -1/3) : x = -5/33 :=
sorry

theorem problem2 : -2 - (-1/3 + 1/2) = -13/6 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l513_51315


namespace NUMINAMATH_GPT_problem_M_m_evaluation_l513_51307

theorem problem_M_m_evaluation
  (a b c d e : ℝ)
  (h : a < b)
  (h' : b < c)
  (h'' : c < d)
  (h''' : d < e)
  (h'''' : a < e) :
  (max (min a (max b c))
       (max (min a d) (max b e))) = e := 
by
  sorry

end NUMINAMATH_GPT_problem_M_m_evaluation_l513_51307


namespace NUMINAMATH_GPT_tan_half_sum_sq_l513_51303

theorem tan_half_sum_sq (a b : ℝ) : 
  3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b + 1) = 0 → 
  ∃ (x : ℝ), (x = (Real.tan (a / 2) + Real.tan (b / 2))^2) ∧ (x = 6 ∨ x = 26) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_tan_half_sum_sq_l513_51303


namespace NUMINAMATH_GPT_three_digit_integer_conditions_l513_51352

theorem three_digit_integer_conditions:
  ∃ n : ℕ, 
    n % 5 = 3 ∧ 
    n % 7 = 4 ∧ 
    n % 4 = 2 ∧
    100 ≤ n ∧ n < 1000 ∧ 
    n = 548 :=
sorry

end NUMINAMATH_GPT_three_digit_integer_conditions_l513_51352


namespace NUMINAMATH_GPT_no_line_bisected_by_P_exists_l513_51371

theorem no_line_bisected_by_P_exists (P : ℝ × ℝ) (H : ∀ x y : ℝ, (x / 3)^2 - (y / 2)^2 = 1) : 
  P ≠ (2, 1) := 
sorry

end NUMINAMATH_GPT_no_line_bisected_by_P_exists_l513_51371


namespace NUMINAMATH_GPT_volume_PABCD_l513_51319

noncomputable def volume_of_pyramid (AB BC : ℝ) (PA : ℝ) : ℝ :=
  (1 / 3) * (AB * BC) * PA

theorem volume_PABCD (AB BC : ℝ) (h_AB : AB = 10) (h_BC : BC = 5)
  (PA : ℝ) (h_PA : PA = 2 * BC) :
  volume_of_pyramid AB BC PA = 500 / 3 :=
by
  subst h_AB
  subst h_BC
  subst h_PA
  -- At this point, we assert that everything simplifies correctly.
  -- This fill in the details for the correct expressions.
  sorry

end NUMINAMATH_GPT_volume_PABCD_l513_51319


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l513_51368

theorem quadratic_inequality_solution (x : ℝ) : -3 < x ∧ x < 4 → x^2 - x - 12 < 0 := by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l513_51368


namespace NUMINAMATH_GPT_inflection_point_on_3x_l513_51339

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4 * Real.sin x - Real.cos x
noncomputable def f' (x : ℝ) : ℝ := 3 + 4 * Real.cos x + Real.sin x
noncomputable def f'' (x : ℝ) : ℝ := -4 * Real.sin x + Real.cos x

theorem inflection_point_on_3x {x0 : ℝ} (h : f'' x0 = 0) : (f x0) = 3 * x0 := by
  sorry

end NUMINAMATH_GPT_inflection_point_on_3x_l513_51339


namespace NUMINAMATH_GPT_reducibility_implies_divisibility_l513_51335

theorem reducibility_implies_divisibility
  (a b c d l k : ℤ)
  (p q : ℤ)
  (h1 : a * l + b = k * p)
  (h2 : c * l + d = k * q) :
  k ∣ (a * d - b * c) :=
sorry

end NUMINAMATH_GPT_reducibility_implies_divisibility_l513_51335


namespace NUMINAMATH_GPT_sum_of_digits_l513_51327

def digits (n : ℕ) : Prop := n ≥ 0 ∧ n < 10

def P := 1
def Q := 0
def R := 2
def S := 5
def T := 6

theorem sum_of_digits :
  digits P ∧ digits Q ∧ digits R ∧ digits S ∧ digits T ∧ 
  (10000 * P + 1000 * Q + 100 * R + 10 * S + T) * 4 = 41024 →
  P + Q + R + S + T = 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_l513_51327


namespace NUMINAMATH_GPT_distance_to_school_l513_51366

theorem distance_to_school (d : ℝ) (h1 : d / 5 + d / 25 = 1) : d = 25 / 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_school_l513_51366


namespace NUMINAMATH_GPT_never_sunday_l513_51397

theorem never_sunday (n : ℕ) (days_in_month : ℕ → ℕ) (is_leap_year : Bool) : 
  (∀ (month : ℕ), 1 ≤ month ∧ month ≤ 12 → (days_in_month month = 28 ∨ days_in_month month = 29 ∨ days_in_month month = 30 ∨ days_in_month month = 31) ∧
  (∃ (k : ℕ), k < 7 ∧ ∀ (d : ℕ), d < days_in_month month → (d % 7 = k ↔ n ≠ d))) → n = 31 := 
by
  sorry

end NUMINAMATH_GPT_never_sunday_l513_51397


namespace NUMINAMATH_GPT_line_through_P_with_intercepts_l513_51329

theorem line_through_P_with_intercepts (a b : ℝ) (P : ℝ × ℝ) (hP : P = (6, -1)) 
  (h1 : a = 3 * b) (ha : a = 1 / ((-b - 1) / 6) + 6) (hb : b = -6 * ((-b - 1) / 6) - 1) :
  (∀ x y, y = (-1 / 3) * x + 1 ∨ y = (-1 / 6) * x) :=
sorry

end NUMINAMATH_GPT_line_through_P_with_intercepts_l513_51329


namespace NUMINAMATH_GPT_vertex_sum_of_cube_l513_51301

noncomputable def cube_vertex_sum (a : Fin 8 → ℕ) : ℕ :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7

def face_sums (a : Fin 8 → ℕ) : List ℕ :=
  [
    a 0 + a 1 + a 2 + a 3, -- first face
    a 0 + a 1 + a 4 + a 5, -- second face
    a 0 + a 3 + a 4 + a 7, -- third face
    a 1 + a 2 + a 5 + a 6, -- fourth face
    a 2 + a 3 + a 6 + a 7, -- fifth face
    a 4 + a 5 + a 6 + a 7  -- sixth face
  ]

def total_face_sum (a : Fin 8 → ℕ) : ℕ :=
  List.sum (face_sums a)

theorem vertex_sum_of_cube (a : Fin 8 → ℕ) (h : total_face_sum a = 2019) :
  cube_vertex_sum a = 673 :=
sorry

end NUMINAMATH_GPT_vertex_sum_of_cube_l513_51301


namespace NUMINAMATH_GPT_geometric_arithmetic_sequence_difference_l513_51361

theorem geometric_arithmetic_sequence_difference
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (hq : q > 0)
  (ha1 : a 1 = 2)
  (ha2 : a 2 = a 1 * q)
  (ha4 : a 4 = a 1 * q ^ 3)
  (ha5 : a 5 = a 1 * q ^ 4)
  (harith : 2 * (a 4 + 2 * a 5) = 2 * a 2 + (a 4 + 2 * a 5))
  (hS : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  S 10 - S 4 = 2016 :=
by
  sorry

end NUMINAMATH_GPT_geometric_arithmetic_sequence_difference_l513_51361


namespace NUMINAMATH_GPT_black_balls_count_l513_51300

theorem black_balls_count
  (P_red P_white : ℝ)
  (Red_balls_count : ℕ)
  (h1 : P_red = 0.42)
  (h2 : P_white = 0.28)
  (h3 : Red_balls_count = 21) :
  ∃ B, B = 15 :=
by
  sorry

end NUMINAMATH_GPT_black_balls_count_l513_51300


namespace NUMINAMATH_GPT_sum_of_two_numbers_l513_51354

theorem sum_of_two_numbers (S L : ℝ) (h1 : S = 10.0) (h2 : 7 * S = 5 * L) : S + L = 24.0 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l513_51354


namespace NUMINAMATH_GPT_determine_f_36_l513_51325

def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ n, f (n + 1) > f n

def multiplicative (f : ℕ → ℕ) : Prop :=
  ∀ m n, f (m * n) = f m * f n

def special_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n, m > n → m^m = n^n → f m = n

theorem determine_f_36 (f : ℕ → ℕ)
  (H1: strictly_increasing f)
  (H2: multiplicative f)
  (H3: special_condition f)
  : f 36 = 1296 := 
sorry

end NUMINAMATH_GPT_determine_f_36_l513_51325


namespace NUMINAMATH_GPT_sum_y_coords_l513_51334

theorem sum_y_coords (h1 : ∃(y : ℝ), (0 + 3)^2 + (y - 5)^2 = 64) : 
  ∃ y1 y2 : ℝ, y1 + y2 = 10 ∧ (0, y1) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) ∧ 
                            (0, y2) ∈ ({ p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 5)^2 = 64 }) := 
by
  sorry

end NUMINAMATH_GPT_sum_y_coords_l513_51334


namespace NUMINAMATH_GPT_cube_side_length_in_cone_l513_51320

noncomputable def side_length_of_inscribed_cube (r h : ℝ) : ℝ :=
  if r = 1 ∧ h = 3 then (3 * Real.sqrt 2) / (3 + Real.sqrt 2) else 0

theorem cube_side_length_in_cone :
  side_length_of_inscribed_cube 1 3 = (3 * Real.sqrt 2) / (3 + Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_cube_side_length_in_cone_l513_51320


namespace NUMINAMATH_GPT_food_insufficiency_l513_51384

-- Given conditions
def number_of_dogs : ℕ := 5
def food_per_meal : ℚ := 3 / 4
def meals_per_day : ℕ := 3
def initial_food : ℚ := 45
def days_in_two_weeks : ℕ := 14

-- Definitions derived from conditions
def daily_food_per_dog : ℚ := food_per_meal * meals_per_day
def daily_food_for_all_dogs : ℚ := daily_food_per_dog * number_of_dogs
def total_food_in_two_weeks : ℚ := daily_food_for_all_dogs * days_in_two_weeks

-- Proof statement: proving the food consumed exceeds the initial amount
theorem food_insufficiency : total_food_in_two_weeks > initial_food :=
by {
  sorry
}

end NUMINAMATH_GPT_food_insufficiency_l513_51384


namespace NUMINAMATH_GPT_range_of_k_l513_51349

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ 0 ≤ k ∧ k < 4 := sorry

end NUMINAMATH_GPT_range_of_k_l513_51349


namespace NUMINAMATH_GPT_crocus_bulbs_count_l513_51337

theorem crocus_bulbs_count (C D : ℕ) 
  (h1 : C + D = 55) 
  (h2 : 0.35 * (C : ℝ) + 0.65 * (D : ℝ) = 29.15) :
  C = 22 :=
sorry

end NUMINAMATH_GPT_crocus_bulbs_count_l513_51337


namespace NUMINAMATH_GPT_max_abs_sum_sqrt2_l513_51378

theorem max_abs_sum_sqrt2 (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (a : ℝ), (a = |x| + |y| ∧ a ≤ 2 * Real.sqrt 2) ∧ 
             ∀ (x y : ℝ), x^2 + y^2 = 4 → (|x| + |y|) ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_abs_sum_sqrt2_l513_51378


namespace NUMINAMATH_GPT_min_students_solving_most_l513_51358

theorem min_students_solving_most (students problems : Nat) 
    (total_students : students = 10) 
    (problems_per_student : Nat → Nat) 
    (problems_per_student_property : ∀ s, s < students → problems_per_student s = 3) 
    (common_problem : ∀ s1 s2, s1 < students → s2 < students → s1 ≠ s2 → ∃ p, p < problems ∧ (∃ (solves1 solves2 : Nat → Nat), (solves1 p = 1 ∧ solves2 p = 1) ∧ s1 < students ∧ s2 < students)): 
  ∃ min_students, min_students = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_students_solving_most_l513_51358


namespace NUMINAMATH_GPT_new_paint_intensity_l513_51381

-- Definition of the given conditions
def original_paint_intensity : ℝ := 0.15
def replacement_paint_intensity : ℝ := 0.25
def fraction_replaced : ℝ := 1.5
def original_volume : ℝ := 100

-- Proof statement
theorem new_paint_intensity :
  (original_volume * original_paint_intensity + original_volume * fraction_replaced * replacement_paint_intensity) /
  (original_volume + original_volume * fraction_replaced) = 0.21 :=
by
  sorry

end NUMINAMATH_GPT_new_paint_intensity_l513_51381


namespace NUMINAMATH_GPT_parallel_line_through_point_l513_51309

theorem parallel_line_through_point (x y : ℝ) :
  (∃ (b : ℝ), (∀ (x : ℝ), y = 2 * x + b) ∧ y = 2 * 1 - 4) :=
sorry

end NUMINAMATH_GPT_parallel_line_through_point_l513_51309


namespace NUMINAMATH_GPT_segment_MN_length_l513_51343

theorem segment_MN_length
  (A B C D M N : ℝ)
  (hA : A < B)
  (hB : B < C)
  (hC : C < D)
  (hM : M = (A + C) / 2)
  (hN : N = (B + D) / 2)
  (hAD : D - A = 68)
  (hBC : C - B = 26) :
  |M - N| = 21 :=
sorry

end NUMINAMATH_GPT_segment_MN_length_l513_51343


namespace NUMINAMATH_GPT_even_sum_probability_l513_51373

-- Define the probabilities for the first wheel
def prob_even_first_wheel : ℚ := 3 / 6
def prob_odd_first_wheel : ℚ := 3 / 6

-- Define the probabilities for the second wheel
def prob_even_second_wheel : ℚ := 3 / 4
def prob_odd_second_wheel : ℚ := 1 / 4

-- Probability that the sum of the two selected numbers is even
def prob_even_sum : ℚ :=
  (prob_even_first_wheel * prob_even_second_wheel) +
  (prob_odd_first_wheel * prob_odd_second_wheel)

-- The theorem to prove
theorem even_sum_probability : prob_even_sum = 13 / 24 := by
  sorry

end NUMINAMATH_GPT_even_sum_probability_l513_51373


namespace NUMINAMATH_GPT_value_of_a_minus_b_l513_51391

variable {R : Type} [Field R]

noncomputable def f (a b x : R) : R := a * x + b
noncomputable def g (x : R) : R := -2 * x + 7
noncomputable def h (a b x : R) : R := f a b (g x)

theorem value_of_a_minus_b (a b : R) (h_inv : R → R) 
  (h_def : ∀ x, h_inv x = x + 9)
  (h_eq : ∀ x, h a b x = x - 9) : 
  a - b = 5 := by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l513_51391


namespace NUMINAMATH_GPT_inequality_proof_l513_51377

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (h : 1/a + 1/b + 1/c = a + b + c) :
  1/(2*a + b + c)^2 + 1/(2*b + c + a)^2 + 1/(2*c + a + b)^2 ≤ 3/16 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l513_51377


namespace NUMINAMATH_GPT_fraction_inequality_l513_51365

theorem fraction_inequality {a b : ℝ} (h1 : a < b) (h2 : b < 0) : (1 / a) > (1 / b) :=
by
  sorry

end NUMINAMATH_GPT_fraction_inequality_l513_51365


namespace NUMINAMATH_GPT_find_x_given_sin_interval_l513_51396

open Real

theorem find_x_given_sin_interval (x : ℝ) (h1 : sin x = -3 / 5) (h2 : π < x ∧ x < 3 / 2 * π) :
  x = π + arcsin (3 / 5) :=
sorry

end NUMINAMATH_GPT_find_x_given_sin_interval_l513_51396


namespace NUMINAMATH_GPT_bounds_for_a_l513_51312

theorem bounds_for_a (a : ℝ) (h_a : a > 0) :
  ∀ x : ℝ, 0 < x ∧ x < 17 → (3 / 4) * x = (5 / 6) * (17 - x) + a → a < (153 / 12) := 
sorry

end NUMINAMATH_GPT_bounds_for_a_l513_51312


namespace NUMINAMATH_GPT_factorize_polynomial_l513_51375

theorem factorize_polynomial (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l513_51375


namespace NUMINAMATH_GPT_floor_condition_x_l513_51305

theorem floor_condition_x (x : ℝ) (h : ⌊x * ⌊x⌋⌋ = 48) : 8 ≤ x ∧ x < 49 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_floor_condition_x_l513_51305


namespace NUMINAMATH_GPT_trig_expression_equality_l513_51362

theorem trig_expression_equality (α : ℝ) (h : Real.tan α = 1 / 2) : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -4 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_equality_l513_51362


namespace NUMINAMATH_GPT_trig_expression_value_l513_51345

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 1/2) :
  (1 + 2 * Real.sin (π - α) * Real.cos (-2 * π - α)) / 
  (Real.sin (-α) ^ 2 - Real.sin (5 * π / 2 - α) ^ 2) = -3 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l513_51345


namespace NUMINAMATH_GPT_opposite_of_neg_2023_l513_51399

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end NUMINAMATH_GPT_opposite_of_neg_2023_l513_51399


namespace NUMINAMATH_GPT_equivalence_mod_equivalence_divisible_l513_51304

theorem equivalence_mod (a b c : ℤ) :
  (∃ k : ℤ, a - b = k * c) ↔ (a % c = b % c) := by
  sorry

theorem equivalence_divisible (a b c : ℤ) :
  (a % c = b % c) ↔ (∃ k : ℤ, a - b = k * c) := by
  sorry

end NUMINAMATH_GPT_equivalence_mod_equivalence_divisible_l513_51304


namespace NUMINAMATH_GPT_intersection_in_fourth_quadrant_l513_51331

theorem intersection_in_fourth_quadrant (a : ℝ) : 
  (∃ x y : ℝ, y = -x + 1 ∧ y = x - 2 * a ∧ x > 0 ∧ y < 0) → a > 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_intersection_in_fourth_quadrant_l513_51331


namespace NUMINAMATH_GPT_max_length_CD_l513_51357

open Real

/-- Given a circle with center O and diameter AB = 20 units,
    with points C and D positioned such that C is 6 units away from A
    and D is 7 units away from B on the diameter AB,
    prove that the maximum length of the direct path from C to D is 7 units.
-/
theorem max_length_CD {A B C D : ℝ} 
    (diameter : dist A B = 20) 
    (C_pos : dist A C = 6) 
    (D_pos : dist B D = 7) : 
    dist C D = 7 :=
by
  -- Details of the proof would go here
  sorry

end NUMINAMATH_GPT_max_length_CD_l513_51357


namespace NUMINAMATH_GPT_percentage_length_more_than_breadth_l513_51386

-- Define the basic conditions
variables {C r l b : ℝ}
variable {p : ℝ}

-- Assume the conditions
def conditions (C r l b : ℝ) : Prop :=
  C = 400 ∧ r = 3 ∧ l = 20 ∧ 20 * b = 400 / 3

-- Define the statement that we want to prove
theorem percentage_length_more_than_breadth (C r l b : ℝ) (h : conditions C r l b) :
  ∃ (p : ℝ), l = b * (1 + p / 100) ∧ p = 200 :=
sorry

end NUMINAMATH_GPT_percentage_length_more_than_breadth_l513_51386


namespace NUMINAMATH_GPT_find_correct_speed_l513_51388

variables (d t : ℝ) -- Defining distance and time as real numbers

theorem find_correct_speed
  (h1 : d = 30 * (t + 5 / 60))
  (h2 : d = 50 * (t - 5 / 60)) :
  ∃ r : ℝ, r = 37.5 ∧ d = r * t :=
by 
  -- Skip the proof for now
  sorry

end NUMINAMATH_GPT_find_correct_speed_l513_51388


namespace NUMINAMATH_GPT_claire_gerbils_l513_51359

variables (G H : ℕ)

-- Claire's total pets
def total_pets : Prop := G + H = 92

-- One-quarter of the gerbils are male
def male_gerbils (G : ℕ) : ℕ := G / 4

-- One-third of the hamsters are male
def male_hamsters (H : ℕ) : ℕ := H / 3

-- Total males are 25
def total_males : Prop := male_gerbils G + male_hamsters H = 25

theorem claire_gerbils : total_pets G H → total_males G H → G = 68 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_claire_gerbils_l513_51359


namespace NUMINAMATH_GPT_ray_walks_to_high_school_7_l513_51330

theorem ray_walks_to_high_school_7
  (walks_to_park : ℕ)
  (walks_to_high_school : ℕ)
  (walks_home : ℕ)
  (trips_per_day : ℕ)
  (total_daily_blocks : ℕ) :
  walks_to_park = 4 →
  walks_home = 11 →
  trips_per_day = 3 →
  total_daily_blocks = 66 →
  3 * (walks_to_park + walks_to_high_school + walks_home) = total_daily_blocks →
  walks_to_high_school = 7 :=
by
  sorry

end NUMINAMATH_GPT_ray_walks_to_high_school_7_l513_51330


namespace NUMINAMATH_GPT_multiple_of_4_and_8_l513_51367

theorem multiple_of_4_and_8 (a b : ℤ) (h1 : ∃ k1 : ℤ, a = 4 * k1) (h2 : ∃ k2 : ℤ, b = 8 * k2) :
  (∃ k3 : ℤ, b = 4 * k3) ∧ (∃ k4 : ℤ, a - b = 4 * k4) :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_4_and_8_l513_51367


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l513_51317

open Set

variable (M N : Set ℕ)

theorem intersection_of_M_and_N :
  M = {1, 2, 4, 8, 16} →
  N = {2, 4, 6, 8} →
  M ∩ N = {2, 4, 8} :=
by
  intros hM hN
  rw [hM, hN]
  ext x
  simp
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l513_51317


namespace NUMINAMATH_GPT_train_total_distance_l513_51389

theorem train_total_distance (x : ℝ) (h1 : x > 0) 
  (h_speed_avg : 48 = ((3 * x) / (x / 8))) : 
  3 * x = 6 := 
by
  sorry

end NUMINAMATH_GPT_train_total_distance_l513_51389


namespace NUMINAMATH_GPT_wheat_field_problem_l513_51350

def equations (x F : ℕ) :=
  (6 * x - 300 = F) ∧ (5 * x + 200 = F)

theorem wheat_field_problem :
  ∃ (x F : ℕ), equations x F ∧ x = 500 ∧ F = 2700 :=
by
  sorry

end NUMINAMATH_GPT_wheat_field_problem_l513_51350


namespace NUMINAMATH_GPT_solution_set_l513_51363

open Real

noncomputable def f : ℝ → ℝ := sorry -- The function f is abstractly defined
axiom f_point : f 1 = 0 -- f passes through (1, 0)
axiom f_deriv_pos : ∀ (x : ℝ), x > 0 → x * (deriv f x) > 1 -- xf'(x) > 1 for x > 0

theorem solution_set (x : ℝ) : f x ≤ log x ↔ 0 < x ∧ x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l513_51363


namespace NUMINAMATH_GPT_blocks_for_tower_l513_51347

theorem blocks_for_tower (total_blocks : ℕ) (house_blocks : ℕ) (extra_blocks : ℕ) (tower_blocks : ℕ) 
  (h1 : total_blocks = 95) 
  (h2 : house_blocks = 20) 
  (h3 : extra_blocks = 30) 
  (h4 : tower_blocks = house_blocks + extra_blocks) : 
  tower_blocks = 50 :=
sorry

end NUMINAMATH_GPT_blocks_for_tower_l513_51347


namespace NUMINAMATH_GPT_find_f_of_9_l513_51311

theorem find_f_of_9 (α : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x ^ α)
  (h2 : f 2 = Real.sqrt 2) :
  f 9 = 3 :=
sorry

end NUMINAMATH_GPT_find_f_of_9_l513_51311


namespace NUMINAMATH_GPT_find_a_for_tangent_parallel_l513_51382

theorem find_a_for_tangent_parallel : 
  ∀ a : ℝ,
  (∀ (x y : ℝ), y = Real.log x - a * x → x = 1 → 2 * x + y - 1 = 0) →
  a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_tangent_parallel_l513_51382


namespace NUMINAMATH_GPT_symmetric_curve_eq_l513_51394

-- Definitions from the problem conditions
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y + 1) ^ 2 = 1
def line_of_symmetry (x y : ℝ) : Prop := x - y + 3 = 0

-- Problem statement derived from the translation step
theorem symmetric_curve_eq (x y : ℝ) : (x - 2) ^ 2 + (y + 1) ^ 2 = 1 ∧ x - y + 3 = 0 → (x + 4) ^ 2 + (y - 5) ^ 2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_symmetric_curve_eq_l513_51394


namespace NUMINAMATH_GPT_total_boxes_moved_l513_51398

-- Define a truck's capacity and number of trips
def truck_capacity : ℕ := 4
def trips : ℕ := 218

-- Prove that the total number of boxes is 872
theorem total_boxes_moved : truck_capacity * trips = 872 := by
  sorry

end NUMINAMATH_GPT_total_boxes_moved_l513_51398


namespace NUMINAMATH_GPT_profit_relationship_profit_range_max_profit_l513_51351

noncomputable def profit (x : ℝ) : ℝ :=
  -20 * x ^ 2 + 100 * x + 6000

theorem profit_relationship (x : ℝ) :
  profit (x) = (60 - x) * (300 + 20 * x) - 40 * (300 + 20 * x) :=
by
  sorry
  
theorem profit_range (x : ℝ) (h : 0 ≤ x ∧ x < 20) : 
  0 ≤ profit (x) :=
by
  sorry

theorem max_profit (x : ℝ) :
  (2.5 ≤ x ∧ x < 2.6) → profit (x) ≤ 6125 := 
by
  sorry  

end NUMINAMATH_GPT_profit_relationship_profit_range_max_profit_l513_51351


namespace NUMINAMATH_GPT_find_x_l513_51369

theorem find_x (x : ℚ) (h1 : 8 * x^2 + 9 * x - 2 = 0) (h2 : 16 * x^2 + 35 * x - 4 = 0) : 
  x = 1 / 8 :=
by sorry

end NUMINAMATH_GPT_find_x_l513_51369


namespace NUMINAMATH_GPT_total_water_in_bucket_l513_51318

noncomputable def initial_gallons : ℝ := 3
noncomputable def added_gallons_1 : ℝ := 6.8
noncomputable def liters_to_gallons (liters : ℝ) : ℝ := liters / 3.78541
noncomputable def quart_to_gallons (quarts : ℝ) : ℝ := quarts / 4
noncomputable def added_gallons_2 : ℝ := liters_to_gallons 10
noncomputable def added_gallons_3 : ℝ := quart_to_gallons 4

noncomputable def total_gallons : ℝ :=
  initial_gallons + added_gallons_1 + added_gallons_2 + added_gallons_3

theorem total_water_in_bucket :
  abs (total_gallons - 13.44) < 0.01 :=
by
  -- convert amounts and perform arithmetic operations
  sorry

end NUMINAMATH_GPT_total_water_in_bucket_l513_51318


namespace NUMINAMATH_GPT_roots_geom_prog_eq_neg_cbrt_c_l513_51348

theorem roots_geom_prog_eq_neg_cbrt_c {a b c : ℝ} (h : ∀ (x1 x2 x3 : ℝ), 
  (x1^3 + a * x1^2 + b * x1 + c = 0) ∧ (x2^3 + a * x2^2 + b * x2 + c = 0) ∧ (x3^3 + a * x3^2 + b * x3 + c = 0) ∧ 
  (∃ (r : ℝ), (x2 = r * x1) ∧ (x3 = r^2 * x1))) : 
  ∃ (x : ℝ), (x^3 = c) ∧ (x = - ((c) ^ (1/3))) :=
by 
  sorry

end NUMINAMATH_GPT_roots_geom_prog_eq_neg_cbrt_c_l513_51348


namespace NUMINAMATH_GPT_greatest_difference_47x_l513_51321

def is_multiple_of_4 (n : Nat) : Prop :=
  n % 4 = 0

def valid_digit (d : Nat) : Prop :=
  d < 10

theorem greatest_difference_47x :
  ∃ x y : Nat, (is_multiple_of_4 (470 + x) ∧ valid_digit x) ∧ (is_multiple_of_4 (470 + y) ∧ valid_digit y) ∧ (x < y) ∧ (y - x = 4) :=
sorry

end NUMINAMATH_GPT_greatest_difference_47x_l513_51321


namespace NUMINAMATH_GPT_determine_m_of_monotonically_increasing_function_l513_51326

theorem determine_m_of_monotonically_increasing_function 
  (m n : ℝ)
  (h : ∀ x, 12 * x ^ 2 + 2 * m * x + (m - 3) ≥ 0) :
  m = 6 := 
by 
  sorry

end NUMINAMATH_GPT_determine_m_of_monotonically_increasing_function_l513_51326


namespace NUMINAMATH_GPT_david_marks_in_physics_l513_51355

theorem david_marks_in_physics : 
  ∀ (P : ℝ), 
  let english := 72 
  let mathematics := 60 
  let chemistry := 62 
  let biology := 84 
  let average_marks := 62.6 
  let num_subjects := 5 
  let total_marks := average_marks * num_subjects 
  let known_marks := english + mathematics + chemistry + biology 
  total_marks - known_marks = P → P = 35 :=
by
  sorry

end NUMINAMATH_GPT_david_marks_in_physics_l513_51355


namespace NUMINAMATH_GPT_Ned_washed_shirts_l513_51302

-- Definitions based on conditions
def short_sleeve_shirts : ℕ := 9
def long_sleeve_shirts : ℕ := 21
def total_shirts : ℕ := short_sleeve_shirts + long_sleeve_shirts
def not_washed_shirts : ℕ := 1
def washed_shirts : ℕ := total_shirts - not_washed_shirts

-- Statement to prove
theorem Ned_washed_shirts : washed_shirts = 29 := by
  sorry

end NUMINAMATH_GPT_Ned_washed_shirts_l513_51302


namespace NUMINAMATH_GPT_sum_even_minus_sum_odd_l513_51356

theorem sum_even_minus_sum_odd :
  let x := (100 / 2) * (2 + 200)
  let y := (100 / 2) * (1 + 199)
  x - y = 100 := by
sorry

end NUMINAMATH_GPT_sum_even_minus_sum_odd_l513_51356


namespace NUMINAMATH_GPT_non_congruent_squares_on_5x5_grid_l513_51376

def is_lattice_point (x y : ℕ) : Prop := x ≤ 4 ∧ y ≤ 4

def is_square {a b c d : (ℕ × ℕ)} : Prop :=
((a.1 - b.1)^2 + (a.2 - b.2)^2 = (c.1 - d.1)^2 + (c.2 - d.2)^2) ∧ 
((c.1 - b.1)^2 + (c.2 - b.2)^2 = (a.1 - d.1)^2 + (a.2 - d.2)^2)

def number_of_non_congruent_squares : ℕ :=
  4 + -- Standard squares: 1x1, 2x2, 3x3, 4x4
  2 + -- Diagonal squares: with sides √2 and 2√2
  2   -- Diagonal sides of 1x2 and 1x3 rectangles

theorem non_congruent_squares_on_5x5_grid :
  number_of_non_congruent_squares = 8 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_non_congruent_squares_on_5x5_grid_l513_51376


namespace NUMINAMATH_GPT_tax_percentage_first_tier_l513_51332

theorem tax_percentage_first_tier
  (car_price : ℝ)
  (total_tax : ℝ)
  (first_tier_level : ℝ)
  (second_tier_rate : ℝ)
  (first_tier_tax : ℝ)
  (T : ℝ)
  (h_car_price : car_price = 30000)
  (h_total_tax : total_tax = 5500)
  (h_first_tier_level : first_tier_level = 10000)
  (h_second_tier_rate : second_tier_rate = 0.15)
  (h_first_tier_tax : first_tier_tax = (T / 100) * first_tier_level) :
  T = 25 :=
by
  sorry

end NUMINAMATH_GPT_tax_percentage_first_tier_l513_51332


namespace NUMINAMATH_GPT_product_mod_7_l513_51353

theorem product_mod_7 : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
by
  have h1 : 2021 % 7 = 6 := by sorry
  have h2 : 2022 % 7 = 0 := by sorry
  have h3 : 2023 % 7 = 1 := by sorry
  have h4 : 2024 % 7 = 2 := by sorry
  sorry

end NUMINAMATH_GPT_product_mod_7_l513_51353


namespace NUMINAMATH_GPT_solution_set_of_inequality_l513_51390

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} := sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l513_51390


namespace NUMINAMATH_GPT_find_a_l513_51314

theorem find_a (x y a : ℝ) (h1 : 4 * x + y = 8) (h2 : 3 * x - 4 * y = 5) (h3 : a * x - 3 * y = 23) : 
  a = 12.141 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l513_51314


namespace NUMINAMATH_GPT_average_price_of_cow_l513_51308

theorem average_price_of_cow (total_price_cows_and_goats rs: ℕ) (num_cows num_goats: ℕ)
    (avg_price_goat: ℕ) (total_price: total_price_cows_and_goats = 1400)
    (num_cows_eq: num_cows = 2) (num_goats_eq: num_goats = 8)
    (avg_price_goat_eq: avg_price_goat = 60) :
    let total_price_goats := avg_price_goat * num_goats
    let total_price_cows := total_price_cows_and_goats - total_price_goats
    let avg_price_cow := total_price_cows / num_cows
    avg_price_cow = 460 :=
by
  sorry

end NUMINAMATH_GPT_average_price_of_cow_l513_51308


namespace NUMINAMATH_GPT_number_of_articles_l513_51372

variables (C S N : ℝ)
noncomputable def gain : ℝ := 3 / 7

-- Cost price of 50 articles is equal to the selling price of N articles
axiom cost_price_eq_selling_price : 50 * C = N * S

-- Selling price is cost price plus gain percentage
axiom selling_price_with_gain : S = C * (1 + gain)

-- Goal: Prove that N = 35
theorem number_of_articles (h1 : 50 * C = N * C * (10 / 7)) : N = 35 := by
  sorry

end NUMINAMATH_GPT_number_of_articles_l513_51372


namespace NUMINAMATH_GPT_minimize_expression_at_9_l513_51395

noncomputable def minimize_expression (n : ℕ) : ℚ :=
  n / 3 + 27 / n

theorem minimize_expression_at_9 : minimize_expression 9 = 6 := by
  sorry

end NUMINAMATH_GPT_minimize_expression_at_9_l513_51395


namespace NUMINAMATH_GPT_inverse_proportion_function_sol_l513_51342

theorem inverse_proportion_function_sol (k m x : ℝ) (h1 : k ≠ 0) (h2 : (m - 1) * x ^ (m ^ 2 - 2) = k / x) : m = -1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_function_sol_l513_51342


namespace NUMINAMATH_GPT_smallest_n_exists_l513_51344

def connected (a b : ℕ) : Prop := -- define connection based on a picture not specified here, placeholder
sorry

def not_connected (a b : ℕ) : Prop := ¬ connected a b

def coprime (a n : ℕ) : Prop := ∀ k : ℕ, k > 1 → k ∣ a → ¬ k ∣ n

def common_divisor_greater_than_one (a n : ℕ) : Prop := ∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ n

theorem smallest_n_exists :
  ∃ n : ℕ,
  (n = 35) ∧
  ∀ (numbers : Fin 7 → ℕ),
  (∀ i j, not_connected (numbers i) (numbers j) → coprime (numbers i + numbers j) n) ∧
  (∀ i j, connected (numbers i) (numbers j) → common_divisor_greater_than_one (numbers i + numbers j) n) := 
sorry

end NUMINAMATH_GPT_smallest_n_exists_l513_51344


namespace NUMINAMATH_GPT_height_difference_between_crates_l513_51387

theorem height_difference_between_crates 
  (n : ℕ) (diameter : ℝ) 
  (height_A : ℝ) (height_B : ℝ) :
  n = 200 →
  diameter = 12 →
  height_A = n / 10 * diameter →
  height_B = n / 20 * (diameter + 6 * Real.sqrt 3) →
  height_A - height_B = 120 - 60 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_height_difference_between_crates_l513_51387


namespace NUMINAMATH_GPT_unique_solution_for_y_l513_51383

def operation (x y : ℝ) : ℝ := 4 * x - 2 * y + x^2 * y

theorem unique_solution_for_y : ∃! (y : ℝ), operation 3 y = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_solution_for_y_l513_51383


namespace NUMINAMATH_GPT_congruence_example_l513_51346

theorem congruence_example (x : ℤ) (h : 5 * x + 3 ≡ 1 [ZMOD 18]) : 3 * x + 8 ≡ 14 [ZMOD 18] :=
sorry

end NUMINAMATH_GPT_congruence_example_l513_51346


namespace NUMINAMATH_GPT_preimage_of_4_3_is_2_1_l513_51310

theorem preimage_of_4_3_is_2_1 :
  ∃ (a b : ℝ), (a + 2 * b = 4) ∧ (2 * a - b = 3) ∧ (a = 2) ∧ (b = 1) :=
by
  exists 2
  exists 1
  constructor
  { sorry }
  constructor
  { sorry }
  constructor
  { sorry }
  { sorry }


end NUMINAMATH_GPT_preimage_of_4_3_is_2_1_l513_51310
