import Mathlib

namespace NUMINAMATH_GPT_ratio_of_pentagon_side_to_rectangle_width_l713_71398

-- Definitions based on the conditions
def pentagon_perimeter : ℝ := 60
def rectangle_perimeter : ℝ := 60
def rectangle_length (w : ℝ) : ℝ := 2 * w

-- The statement to be proven
theorem ratio_of_pentagon_side_to_rectangle_width :
  ∀ w : ℝ, 2 * (rectangle_length w + w) = rectangle_perimeter → (pentagon_perimeter / 5) / w = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_pentagon_side_to_rectangle_width_l713_71398


namespace NUMINAMATH_GPT_fisher_catch_l713_71331

theorem fisher_catch (x y : ℕ) (h1 : x + y = 80)
  (h2 : ∃ a : ℕ, x = 9 * a)
  (h3 : ∃ b : ℕ, y = 11 * b) :
  x = 36 ∧ y = 44 :=
by
  sorry

end NUMINAMATH_GPT_fisher_catch_l713_71331


namespace NUMINAMATH_GPT_find_price_of_stock_A_l713_71303

-- Define conditions
def stock_investment_A (price_A : ℝ) : Prop := 
  ∃ (income_A: ℝ), income_A = 0.10 * 100

def stock_investment_B (price_B : ℝ) (investment_B : ℝ) : Prop := 
  price_B = 115.2 ∧ investment_B = 10 / 0.12

-- The main goal statement
theorem find_price_of_stock_A 
  (price_A : ℝ) (investment_B : ℝ) 
  (hA : stock_investment_A price_A) 
  (hB : stock_investment_B price_A investment_B) :
  price_A = 138.24 := 
sorry

end NUMINAMATH_GPT_find_price_of_stock_A_l713_71303


namespace NUMINAMATH_GPT_negative_integer_solution_l713_71351

theorem negative_integer_solution (x : ℤ) (h : 3 * x + 13 ≥ 0) : x = -1 :=
by
  sorry

end NUMINAMATH_GPT_negative_integer_solution_l713_71351


namespace NUMINAMATH_GPT_solve_determinant_l713_71307

-- Definitions based on the conditions
def determinant (a b c d : ℤ) : ℤ := a * d - b * c

-- The problem translated to Lean 4:
theorem solve_determinant (x : ℤ) 
  (h : determinant (x + 1) x (2 * x - 6) (2 * (x - 1)) = 10) :
  x = 2 :=
sorry -- Proof is skipped

end NUMINAMATH_GPT_solve_determinant_l713_71307


namespace NUMINAMATH_GPT_eval_sum_and_subtract_l713_71302

theorem eval_sum_and_subtract : (2345 + 3452 + 4523 + 5234) - 1234 = 14320 := by {
  -- The rest of the proof should go here, but we'll use sorry to skip it.
  sorry
}

end NUMINAMATH_GPT_eval_sum_and_subtract_l713_71302


namespace NUMINAMATH_GPT_smallest_x_l713_71373

theorem smallest_x (M x : ℕ) (h : 720 * x = M^3) : x = 300 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_l713_71373


namespace NUMINAMATH_GPT_tan_alpha_l713_71317

variable (α : ℝ)

theorem tan_alpha (h₁ : Real.sin α = -5/13) (h₂ : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) :
  Real.tan α = -5/12 :=
sorry

end NUMINAMATH_GPT_tan_alpha_l713_71317


namespace NUMINAMATH_GPT_train_speed_correct_l713_71364

-- Define the length of the train
def train_length : ℝ := 200

-- Define the time taken to cross the telegraph post
def cross_time : ℝ := 8

-- Define the expected speed of the train
def expected_speed : ℝ := 25

-- Prove that the speed of the train is as expected
theorem train_speed_correct (length time : ℝ) (h_length : length = train_length) (h_time : time = cross_time) : 
  (length / time = expected_speed) :=
by
  rw [h_length, h_time]
  sorry

end NUMINAMATH_GPT_train_speed_correct_l713_71364


namespace NUMINAMATH_GPT_value_of_x_l713_71343

theorem value_of_x (x : ℝ) (m : ℕ) (h1 : m = 31) :
  ((x ^ m) / (5 ^ m)) * ((x ^ 16) / (4 ^ 16)) = 1 / (2 * 10 ^ 31) → x = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l713_71343


namespace NUMINAMATH_GPT_union_of_A_and_B_l713_71315

open Set

variable {x : ℝ}

-- Define sets A and B based on the given conditions
def A : Set ℝ := { x | 0 < 3 - x ∧ 3 - x ≤ 2 }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- The theorem to prove
theorem union_of_A_and_B : A ∪ B = { x | 0 ≤ x ∧ x < 3 } := 
by 
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l713_71315


namespace NUMINAMATH_GPT_limit_does_not_exist_l713_71347

noncomputable def does_not_exist_limit : Prop := 
  ¬ ∃ l : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    (0 < |x| ∧ 0 < |y| ∧ |x| < δ ∧ |y| < δ) →
    |(x^2 - y^2) / (x^2 + y^2) - l| < ε

theorem limit_does_not_exist :
  does_not_exist_limit :=
sorry

end NUMINAMATH_GPT_limit_does_not_exist_l713_71347


namespace NUMINAMATH_GPT_sheila_weekly_earnings_l713_71322

theorem sheila_weekly_earnings:
  (∀(m w f : ℕ), (m = 8) → (w = 8) → (f = 8) → 
   ∀(t th : ℕ), (t = 6) → (th = 6) → 
   ∀(h : ℕ), (h = 6) → 
   (m + w + f + t + th) * h = 216) := by
  sorry

end NUMINAMATH_GPT_sheila_weekly_earnings_l713_71322


namespace NUMINAMATH_GPT_hunting_dog_catches_fox_l713_71393

theorem hunting_dog_catches_fox :
  ∀ (V_1 V_2 : ℝ) (t : ℝ),
  V_1 / V_2 = 10 ∧ 
  t * V_2 = (10 / (V_2) + t) →
  (V_1 * t) = 100 / 9 :=
by
  intros V_1 V_2 t h
  sorry

end NUMINAMATH_GPT_hunting_dog_catches_fox_l713_71393


namespace NUMINAMATH_GPT_polynomial_roots_and_coefficients_l713_71371

theorem polynomial_roots_and_coefficients 
  (a b c d e : ℝ)
  (h1 : a = 2)
  (h2 : 256 * a + 64 * b + 16 * c + 4 * d + e = 0)
  (h3 : -81 * a + 27 * b - 9 * c + 3 * d + e = 0)
  (h4 : 625 * a + 125 * b + 25 * c + 5 * d + e = 0) :
  (b + c + d) / a = 151 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_and_coefficients_l713_71371


namespace NUMINAMATH_GPT_tan_periodic_example_l713_71330

theorem tan_periodic_example : Real.tan (13 * Real.pi / 4) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_tan_periodic_example_l713_71330


namespace NUMINAMATH_GPT_count_paths_COMPUTER_l713_71350

theorem count_paths_COMPUTER : 
  let possible_paths (n : ℕ) := 2 ^ n 
  possible_paths 7 + possible_paths 7 + 1 = 257 :=
by sorry

end NUMINAMATH_GPT_count_paths_COMPUTER_l713_71350


namespace NUMINAMATH_GPT_subset_A_B_l713_71319

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 2} -- Definition of set A
def B (a : ℝ) := {x : ℝ | x > a} -- Definition of set B

theorem subset_A_B (a : ℝ) : a < 1 → A ⊆ B a :=
by
  sorry

end NUMINAMATH_GPT_subset_A_B_l713_71319


namespace NUMINAMATH_GPT_triangle_side_lengths_consecutive_l713_71384

theorem triangle_side_lengths_consecutive (n : ℕ) (a b c A : ℕ) 
  (h1 : a = n - 1) (h2 : b = n) (h3 : c = n + 1) (h4 : A = n + 2)
  (h5 : 2 * A * A = 3 * n^2 * (n^2 - 4)) :
  a = 3 ∧ b = 4 ∧ c = 5 :=
sorry

end NUMINAMATH_GPT_triangle_side_lengths_consecutive_l713_71384


namespace NUMINAMATH_GPT_exists_root_between_roots_l713_71335

theorem exists_root_between_roots 
  (a b c : ℝ) 
  (h_a : a ≠ 0) 
  (x₁ x₂ : ℝ) 
  (h₁ : a * x₁^2 + b * x₁ + c = 0) 
  (h₂ : -a * x₂^2 + b * x₂ + c = 0) 
  (hx : x₁ < x₂) :
  ∃ x₃ : ℝ, x₁ < x₃ ∧ x₃ < x₂ ∧ (a / 2) * x₃^2 + b * x₃ + c = 0 :=
by 
  sorry

end NUMINAMATH_GPT_exists_root_between_roots_l713_71335


namespace NUMINAMATH_GPT_one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1_l713_71374

-- Definition of the conditions.
variable (x : ℝ)

-- Statement of the problem in Lean.
theorem one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1 (x : ℝ) :
    (1 / 3) * (9 * x - 3) = 3 * x - 1 :=
by sorry

end NUMINAMATH_GPT_one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1_l713_71374


namespace NUMINAMATH_GPT_loss_per_metre_l713_71370

-- Definitions for given conditions
def TSP : ℕ := 15000           -- Total Selling Price
def CPM : ℕ := 40              -- Cost Price per Metre
def TMS : ℕ := 500             -- Total Metres Sold

-- Definition for the expected Loss Per Metre
def LPM : ℕ := 10

-- Statement to prove that the loss per metre is 10
theorem loss_per_metre :
  (CPM * TMS - TSP) / TMS = LPM :=
by
sorry

end NUMINAMATH_GPT_loss_per_metre_l713_71370


namespace NUMINAMATH_GPT_avocados_per_serving_l713_71359

-- Definitions for the conditions
def original_avocados : ℕ := 5
def additional_avocados : ℕ := 4
def total_avocados : ℕ := original_avocados + additional_avocados
def servings : ℕ := 3

-- Theorem stating the result
theorem avocados_per_serving : (total_avocados / servings) = 3 :=
by
  sorry

end NUMINAMATH_GPT_avocados_per_serving_l713_71359


namespace NUMINAMATH_GPT_side_length_square_l713_71310

-- Define the length and width of the rectangle
def length_rect := 10 -- cm
def width_rect := 8 -- cm

-- Define the perimeter of the rectangle
def perimeter_rect := 2 * (length_rect + width_rect)

-- Define the perimeter of the square
def perimeter_square (s : ℕ) := 4 * s

-- The theorem to prove
theorem side_length_square : ∃ s : ℕ, perimeter_rect = perimeter_square s ∧ s = 9 :=
by
  sorry

end NUMINAMATH_GPT_side_length_square_l713_71310


namespace NUMINAMATH_GPT_divisibility_by_37_l713_71304

def sum_of_segments (n : ℕ) : ℕ :=
  let rec split_and_sum (num : ℕ) (acc : ℕ) : ℕ :=
    if num < 1000 then acc + num
    else split_and_sum (num / 1000) (acc + num % 1000)
  split_and_sum n 0

theorem divisibility_by_37 (A : ℕ) : 
  (37 ∣ A) ↔ (37 ∣ sum_of_segments A) :=
sorry

end NUMINAMATH_GPT_divisibility_by_37_l713_71304


namespace NUMINAMATH_GPT_average_speed_l713_71321

-- Define the conditions given in the problem
def distance_first_hour : ℕ := 50 -- distance traveled in the first hour
def distance_second_hour : ℕ := 60 -- distance traveled in the second hour
def total_distance : ℕ := distance_first_hour + distance_second_hour -- total distance traveled

-- Define the total time
def total_time : ℕ := 2 -- total time in hours

-- The problem statement: proving the average speed
theorem average_speed : total_distance / total_time = 55 := by
  unfold total_distance total_time
  sorry

end NUMINAMATH_GPT_average_speed_l713_71321


namespace NUMINAMATH_GPT_price_reduction_equation_l713_71305

variable (x : ℝ)

theorem price_reduction_equation (h : 25 * (1 - x) ^ 2 = 16) : 25 * (1 - x) ^ 2 = 16 :=
by
  assumption

end NUMINAMATH_GPT_price_reduction_equation_l713_71305


namespace NUMINAMATH_GPT_blocks_per_friend_l713_71344

theorem blocks_per_friend (total_blocks : ℕ) (friends : ℕ) (h1 : total_blocks = 28) (h2 : friends = 4) :
  total_blocks / friends = 7 :=
by
  sorry

end NUMINAMATH_GPT_blocks_per_friend_l713_71344


namespace NUMINAMATH_GPT_sin_and_tan_sin_add_pi_over_4_and_tan_2alpha_l713_71346

variable {α : ℝ} (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2)

theorem sin_and_tan (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -3/5 ∧ Real.tan α = 3/4 :=
sorry

theorem sin_add_pi_over_4_and_tan_2alpha (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2)
  (h_sin : Real.sin α = -3/5) (h_tan : Real.tan α = 3/4) :
  Real.sin (α + π/4) = -7 * Real.sqrt 2 / 10 ∧ Real.tan (2 * α) = 24/7 :=
sorry

end NUMINAMATH_GPT_sin_and_tan_sin_add_pi_over_4_and_tan_2alpha_l713_71346


namespace NUMINAMATH_GPT_average_speed_to_SF_l713_71349

theorem average_speed_to_SF (v d : ℝ) (h1 : d ≠ 0) (h2 : v ≠ 0) :
  (2 * d / ((d / v) + (2 * d / v)) = 34) → v = 51 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_average_speed_to_SF_l713_71349


namespace NUMINAMATH_GPT_intersection_P_Q_l713_71318

def P := {x : ℝ | 1 < x ∧ x < 3}
def Q := {x : ℝ | 2 < x}

theorem intersection_P_Q :
  P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := sorry

end NUMINAMATH_GPT_intersection_P_Q_l713_71318


namespace NUMINAMATH_GPT_number_of_ways_to_take_pieces_l713_71383

theorem number_of_ways_to_take_pieces : 
  (Nat.choose 6 4) = 15 := 
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_take_pieces_l713_71383


namespace NUMINAMATH_GPT_max_distance_l713_71395

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := 
  (rho * Real.cos theta, rho * Real.sin theta)

noncomputable def curve_C (p : ℝ × ℝ) : Prop := 
  let x := p.1 
  let y := p.2 
  x^2 + y^2 - 2*y = 0

noncomputable def line_l (t : ℝ) : ℝ × ℝ := 
  (-3/5 * t + 2, 4/5 * t)

def x_axis_intersection (l : ℝ → ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := l 0 
  (x, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance {M : ℝ × ℝ} {N : ℝ × ℝ}
  (curve_c : (ℝ × ℝ) → Prop)
  (line_l : ℝ → ℝ × ℝ)
  (h1 : curve_c = curve_C)
  (h2 : line_l = line_l)
  (M_def : x_axis_intersection line_l = M)
  (hNP : curve_c N) :
  distance M N ≤ Real.sqrt 5 + 1 :=
sorry

end NUMINAMATH_GPT_max_distance_l713_71395


namespace NUMINAMATH_GPT_smaller_third_angle_l713_71328

theorem smaller_third_angle (x y : ℕ) (h₁ : x = 64) 
  (h₂ : 2 * x + (x - y) = 180) : y = 12 :=
by
  sorry

end NUMINAMATH_GPT_smaller_third_angle_l713_71328


namespace NUMINAMATH_GPT_gcd_n4_plus_16_n_plus_3_eq_1_l713_71369

theorem gcd_n4_plus_16_n_plus_3_eq_1 (n : ℕ) (h : n > 16) : gcd (n^4 + 16) (n + 3) = 1 := 
sorry

end NUMINAMATH_GPT_gcd_n4_plus_16_n_plus_3_eq_1_l713_71369


namespace NUMINAMATH_GPT_figure_perimeter_l713_71308

theorem figure_perimeter (h_segments v_segments : ℕ) (side_length : ℕ) 
  (h_count : h_segments = 16) (v_count : v_segments = 10) (side_len : side_length = 1) :
  2 * (h_segments + v_segments) * side_length = 26 :=
by
  sorry

end NUMINAMATH_GPT_figure_perimeter_l713_71308


namespace NUMINAMATH_GPT_bill_initial_amount_l713_71338

/-- Suppose Ann has $777 and Bill gives Ann $167,
    after which they both have the same amount of money. 
    Prove that Bill initially had $1111. -/
theorem bill_initial_amount (A B : ℕ) (h₁ : A = 777) (h₂ : B - 167 = A + 167) : B = 1111 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_bill_initial_amount_l713_71338


namespace NUMINAMATH_GPT_peggy_stamps_l713_71332

-- Defining the number of stamps Peggy, Ernie, and Bert have
variables (P : ℕ) (E : ℕ) (B : ℕ)

-- Given conditions
def bert_has_four_times_ernie (B : ℕ) (E : ℕ) : Prop := B = 4 * E
def ernie_has_three_times_peggy (E : ℕ) (P : ℕ) : Prop := E = 3 * P
def peggy_needs_stamps (P : ℕ) (B : ℕ) : Prop := B = P + 825

-- Question to Answer / Theorem Statement
theorem peggy_stamps (P : ℕ) (E : ℕ) (B : ℕ)
  (h1 : bert_has_four_times_ernie B E)
  (h2 : ernie_has_three_times_peggy E P)
  (h3 : peggy_needs_stamps P B) :
  P = 75 :=
sorry

end NUMINAMATH_GPT_peggy_stamps_l713_71332


namespace NUMINAMATH_GPT_time_to_cross_pole_l713_71376

-- Conditions
def train_speed_kmh : ℕ := 108
def train_length_m : ℕ := 210

-- Conversion functions
def km_per_hr_to_m_per_sec (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600

-- Theorem to be proved
theorem time_to_cross_pole : (train_length_m : ℕ) / (km_per_hr_to_m_per_sec train_speed_kmh) = 7 := by
  -- we'll use sorry here to skip the actual proof steps.
  sorry

end NUMINAMATH_GPT_time_to_cross_pole_l713_71376


namespace NUMINAMATH_GPT_percentage_increase_l713_71377

theorem percentage_increase (A B : ℝ) (y : ℝ) (h : A > B) (h1 : B > 0) (h2 : C = A + B) (h3 : C = (1 + y / 100) * B) : y = 100 * (A / B) := 
sorry

end NUMINAMATH_GPT_percentage_increase_l713_71377


namespace NUMINAMATH_GPT_total_experiments_non_adjacent_l713_71327

theorem total_experiments_non_adjacent (n_org n_inorg n_add : ℕ) 
  (h_org : n_org = 3) (h_inorg : n_inorg = 2) (h_add : n_add = 2) 
  (no_adjacent : True) : 
  (n_org + n_inorg + n_add).factorial / (n_inorg + n_add).factorial * 
  (n_inorg + n_add + 1).choose n_org = 1440 :=
by
  -- The actual proof will go here.
  sorry

end NUMINAMATH_GPT_total_experiments_non_adjacent_l713_71327


namespace NUMINAMATH_GPT_no_solution_for_parallel_lines_values_of_a_for_perpendicular_lines_l713_71353

-- Problem 1: There is no value of m that makes the lines parallel.
theorem no_solution_for_parallel_lines (m : ℝ) :
  ¬ ∃ m, (2 * m^2 + m - 3) / (m^2 - m) = 1 := sorry

-- Problem 2: The values of a that make the lines perpendicular.
theorem values_of_a_for_perpendicular_lines (a : ℝ) :
  (a = 1 ∨ a = -3) ↔ (a * (a - 1) + (1 - a) * (2 * a + 3) = 0) := sorry

end NUMINAMATH_GPT_no_solution_for_parallel_lines_values_of_a_for_perpendicular_lines_l713_71353


namespace NUMINAMATH_GPT_time_spent_answering_questions_l713_71357

theorem time_spent_answering_questions (total_questions answered_per_question_minutes unanswered_questions : ℕ) (minutes_per_hour : ℕ) :
  total_questions = 100 → unanswered_questions = 40 → answered_per_question_minutes = 2 → minutes_per_hour = 60 → 
  ((total_questions - unanswered_questions) * answered_per_question_minutes) / minutes_per_hour = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_time_spent_answering_questions_l713_71357


namespace NUMINAMATH_GPT_lower_bound_fraction_sum_l713_71396

open Real

theorem lower_bound_fraction_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  (1 / (3 * a) + 3 / b) ≥ 8 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_lower_bound_fraction_sum_l713_71396


namespace NUMINAMATH_GPT_geometric_mean_of_negatives_l713_71365

theorem geometric_mean_of_negatives :
  ∃ x : ℝ, x^2 = (-2) * (-8) ∧ (x = 4 ∨ x = -4) := by
  sorry

end NUMINAMATH_GPT_geometric_mean_of_negatives_l713_71365


namespace NUMINAMATH_GPT_identified_rectangle_perimeter_l713_71323

-- Define the side length of the square
def side_length_mm : ℕ := 75

-- Define the heights of the rectangles
variables (x y z : ℕ)

-- Define conditions
def rectangles_cut_condition (x y z : ℕ) : Prop := x + y + z = side_length_mm
def perimeter_relation_condition (x y z : ℕ) : Prop := 2 * (x + side_length_mm) = (y + side_length_mm) + (z + side_length_mm)

-- Define the perimeter of the identified rectangle
def identified_perimeter_mm (x : ℕ) := 2 * (x + side_length_mm)

-- Define conversion from mm to cm
def mm_to_cm (mm : ℕ) : ℕ := mm / 10

-- Final proof statement
theorem identified_rectangle_perimeter :
  ∃ x y z : ℕ, rectangles_cut_condition x y z ∧ perimeter_relation_condition x y z ∧ mm_to_cm (identified_perimeter_mm x) = 20 := 
sorry

end NUMINAMATH_GPT_identified_rectangle_perimeter_l713_71323


namespace NUMINAMATH_GPT_cube_volume_of_surface_area_l713_71375

theorem cube_volume_of_surface_area (S : ℝ) (V : ℝ) (a : ℝ) (h1 : S = 150) (h2 : S = 6 * a^2) (h3 : V = a^3) : V = 125 := by
  sorry

end NUMINAMATH_GPT_cube_volume_of_surface_area_l713_71375


namespace NUMINAMATH_GPT_consecutive_numbers_product_l713_71389

theorem consecutive_numbers_product (a b c d : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c + 1 = d) (h4 : a + d = 109) :
  b * c = 2970 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_consecutive_numbers_product_l713_71389


namespace NUMINAMATH_GPT_parabola_intersection_diff_l713_71312

theorem parabola_intersection_diff (a b c d : ℝ) 
  (h₁ : ∀ x y, (3 * x^2 - 2 * x + 1 = y) → (c = x ∨ a = x))
  (h₂ : ∀ x y, (-2 * x^2 + 4 * x + 1 = y) → (c = x ∨ a = x))
  (h₃ : c ≥ a) :
  c - a = 6 / 5 :=
by sorry

end NUMINAMATH_GPT_parabola_intersection_diff_l713_71312


namespace NUMINAMATH_GPT_sequence_term_500_l713_71337

theorem sequence_term_500 :
  ∃ (a : ℕ → ℤ), 
  a 1 = 1001 ∧
  a 2 = 1005 ∧
  (∀ n, 1 ≤ n → (a n + a (n+1) + a (n+2)) = 2 * n) → 
  a 500 = 1334 := 
sorry

end NUMINAMATH_GPT_sequence_term_500_l713_71337


namespace NUMINAMATH_GPT_a6_add_b6_geq_ab_a4_add_b4_l713_71356

theorem a6_add_b6_geq_ab_a4_add_b4 (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  a^6 + b^6 ≥ ab * (a^4 + b^4) :=
sorry

end NUMINAMATH_GPT_a6_add_b6_geq_ab_a4_add_b4_l713_71356


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l713_71341

   variable (x : ℝ)

   theorem simplify_and_evaluate_expression (h : x = 2 * Real.sqrt 5 - 1) :
     (1 / (x ^ 2 + 2 * x + 1) * (1 + 3 / (x - 1)) / ((x + 2) / (x ^ 2 - 1))) = Real.sqrt 5 / 10 :=
   sorry
   
end NUMINAMATH_GPT_simplify_and_evaluate_expression_l713_71341


namespace NUMINAMATH_GPT_time_after_2500_minutes_l713_71329

/-- 
To prove that adding 2500 minutes to midnight on January 1, 2011 results in 
January 2 at 5:40 PM.
-/
theorem time_after_2500_minutes :
  let minutes_in_a_day := 1440 -- 24 hours * 60 minutes
  let minutes_in_an_hour := 60
  let start_time_minutes := 0 -- Midnight January 1, 2011 as zero minutes
  let total_minutes := 2500
  let resulting_minutes := start_time_minutes + total_minutes
  let days_passed := resulting_minutes / minutes_in_a_day
  let remaining_minutes := resulting_minutes % minutes_in_a_day
  let hours := remaining_minutes / minutes_in_an_hour
  let minutes := remaining_minutes % minutes_in_an_hour
  days_passed = 1 ∧ hours = 17 ∧ minutes = 40 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_time_after_2500_minutes_l713_71329


namespace NUMINAMATH_GPT_find_third_discount_percentage_l713_71362

noncomputable def third_discount_percentage (x : ℝ) : Prop :=
  let item_price := 68
  let num_items := 3
  let first_discount := 0.15
  let second_discount := 0.10
  let total_initial_price := num_items * item_price
  let price_after_first_discount := total_initial_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount * (1 - x / 100) = 105.32

theorem find_third_discount_percentage : ∃ x : ℝ, third_discount_percentage x ∧ x = 32.5 :=
by
  sorry

end NUMINAMATH_GPT_find_third_discount_percentage_l713_71362


namespace NUMINAMATH_GPT_mink_ratio_set_free_to_total_l713_71354

-- Given conditions
def coats_needed_per_skin : ℕ := 15
def minks_bought : ℕ := 30
def babies_per_mink : ℕ := 6
def coats_made : ℕ := 7

-- Question as a proof problem
theorem mink_ratio_set_free_to_total :
  let total_minks := minks_bought * (1 + babies_per_mink)
  let minks_used := coats_made * coats_needed_per_skin
  let minks_set_free := total_minks - minks_used
  minks_set_free * 2 = total_minks :=
by
  sorry

end NUMINAMATH_GPT_mink_ratio_set_free_to_total_l713_71354


namespace NUMINAMATH_GPT_number_of_participants_l713_71386

-- Define the conditions and theorem
theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 231) : n = 22 :=
  sorry

end NUMINAMATH_GPT_number_of_participants_l713_71386


namespace NUMINAMATH_GPT_log_inequality_l713_71394

theorem log_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) : 
  Real.log b / Real.log a + Real.log a / Real.log b ≤ -2 := sorry

end NUMINAMATH_GPT_log_inequality_l713_71394


namespace NUMINAMATH_GPT_divisible_by_5_last_digit_l713_71340

theorem divisible_by_5_last_digit (B : ℕ) (h : B < 10) : (∃ k : ℕ, 5270 + B = 5 * k) ↔ B = 0 ∨ B = 5 :=
by sorry

end NUMINAMATH_GPT_divisible_by_5_last_digit_l713_71340


namespace NUMINAMATH_GPT_line_intersects_curve_l713_71399

theorem line_intersects_curve (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ax₁ + 16 = x₁^3 ∧ ax₂ + 16 = x₂^3) →
  a = 12 :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_curve_l713_71399


namespace NUMINAMATH_GPT_proposition_equivalence_l713_71380

-- Definition of propositions p and q
variables (p q : Prop)

-- Statement of the problem in Lean 4
theorem proposition_equivalence :
  (p ∨ q) → ¬(p ∧ q) ↔ (¬((p ∨ q) → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → (p ∨ q))) :=
sorry

end NUMINAMATH_GPT_proposition_equivalence_l713_71380


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l713_71306

theorem cylinder_volume_ratio (r1 r2 V1 V2 : ℝ) (h1 : 2 * Real.pi * r1 = 6) (h2 : 2 * Real.pi * r2 = 10) (hV1 : V1 = Real.pi * r1^2 * 10) (hV2 : V2 = Real.pi * r2^2 * 6) :
  V1 < V2 → (V2 / V1) = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_ratio_l713_71306


namespace NUMINAMATH_GPT_sufficient_condition_frac_ineq_inequality_transformation_problem_equivalence_l713_71378

theorem sufficient_condition_frac_ineq (x : ℝ) : (1 < x ∧ x < 2) → ( (x + 1) / (x - 1) > 2) :=
by
  -- Given that 1 < x and x < 2, we need to show (x + 1) / (x - 1) > 2
  sorry

theorem inequality_transformation (x : ℝ) : ( (x + 1) / (x - 1) > 2) ↔ ( (x - 1) * (x - 3) < 0 ) :=
by
  -- Prove that (x + 1) / (x - 1) > 2 is equivalent to (x - 1)(x - 3) < 0
  sorry

theorem problem_equivalence (x : ℝ) : ( (x + 1) / (x - 1) > 2) → (1 < x ∧ x < 3) :=
by
  -- Prove that (x + 1) / (x - 1) > 2 implies 1 < x < 3
  sorry

end NUMINAMATH_GPT_sufficient_condition_frac_ineq_inequality_transformation_problem_equivalence_l713_71378


namespace NUMINAMATH_GPT_circle_radius_l713_71358

-- Given the equation of a circle, we want to prove its radius
theorem circle_radius : ∀ (x y : ℝ), x^2 + y^2 - 6*y - 16 = 0 → (∃ r, r = 5) :=
  by
    sorry

end NUMINAMATH_GPT_circle_radius_l713_71358


namespace NUMINAMATH_GPT_coefficients_sum_l713_71360

theorem coefficients_sum :
  let A := 3
  let B := 14
  let C := 18
  let D := 19
  let E := 30
  A + B + C + D + E = 84 := by
  sorry

end NUMINAMATH_GPT_coefficients_sum_l713_71360


namespace NUMINAMATH_GPT_measure_of_angle_B_l713_71391

theorem measure_of_angle_B (A B C a b c : ℝ) (h₁ : a = A.sin) (h₂ : b = B.sin) (h₃ : c = C.sin)
  (h₄ : (b - a) / (c + a) = c / (a + b)) :
  B = 2 * π / 3 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_B_l713_71391


namespace NUMINAMATH_GPT_divides_equiv_l713_71355

theorem divides_equiv (m n : ℤ) : 
  (17 ∣ (2 * m + 3 * n)) ↔ (17 ∣ (9 * m + 5 * n)) :=
by
  sorry

end NUMINAMATH_GPT_divides_equiv_l713_71355


namespace NUMINAMATH_GPT_sequence_a_n_general_formula_and_value_sequence_b_n_general_formula_l713_71320

theorem sequence_a_n_general_formula_and_value (a : ℕ → ℕ) 
  (h1 : a 1 = 3) 
  (h10 : a 10 = 21) 
  (h_linear : ∃ (k b : ℕ), ∀ n, a n = k * n + b) :
  (∀ n, a n = 2 * n + 1) ∧ a 2005 = 4011 :=
by 
  sorry

theorem sequence_b_n_general_formula (a b : ℕ → ℕ)
  (h_seq_a : ∀ n, a n = 2 * n + 1) 
  (h_b_formed : ∀ n, b n = a (2 * n)) : 
  ∀ n, b n = 4 * n + 1 :=
by 
  sorry

end NUMINAMATH_GPT_sequence_a_n_general_formula_and_value_sequence_b_n_general_formula_l713_71320


namespace NUMINAMATH_GPT_mowing_difference_l713_71368

-- Define the number of times mowed in spring and summer
def mowedSpring : ℕ := 8
def mowedSummer : ℕ := 5

-- Prove the difference between spring and summer mowing is 3
theorem mowing_difference : mowedSpring - mowedSummer = 3 := by
  sorry

end NUMINAMATH_GPT_mowing_difference_l713_71368


namespace NUMINAMATH_GPT_single_elimination_games_l713_71309

theorem single_elimination_games (n : ℕ) (h : n = 512) : 
  ∃ g : ℕ, g = n - 1 ∧ g = 511 := 
by
  use n - 1
  sorry

end NUMINAMATH_GPT_single_elimination_games_l713_71309


namespace NUMINAMATH_GPT_ab_value_l713_71342

-- Define sets A and B
def A : Set ℝ := {-1.3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b = 0}

-- The proof statement: Given A = B, prove ab = 0.104
theorem ab_value (a b : ℝ) (h : A = B a b) : a * b = 0.104 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l713_71342


namespace NUMINAMATH_GPT_transport_cost_in_euros_l713_71385

def cost_per_kg : ℝ := 18000
def weight_g : ℝ := 300
def exchange_rate : ℝ := 0.95

theorem transport_cost_in_euros :
  (cost_per_kg * (weight_g / 1000) * exchange_rate) = 5130 :=
by sorry

end NUMINAMATH_GPT_transport_cost_in_euros_l713_71385


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_25_l713_71363

theorem smallest_four_digit_divisible_by_25 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 25 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 25 = 0 → n ≤ m := by
  -- Prove that the smallest four-digit number divisible by 25 is 1000
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_25_l713_71363


namespace NUMINAMATH_GPT_smaller_angle_at_10_oclock_l713_71382

def degreeMeasureSmallerAngleAt10 := 
  let totalDegrees := 360
  let numHours := 12
  let degreesPerHour := totalDegrees / numHours
  let hourHandPosition := 10
  let minuteHandPosition := 12
  let divisionsBetween := if hourHandPosition < minuteHandPosition then minuteHandPosition - hourHandPosition else hourHandPosition - minuteHandPosition
  degreesPerHour * divisionsBetween

theorem smaller_angle_at_10_oclock : degreeMeasureSmallerAngleAt10 = 60 :=
  by 
    let totalDegrees := 360
    let numHours := 12
    let degreesPerHour := totalDegrees / numHours
    have h1 : degreesPerHour = 30 := by norm_num
    let hourHandPosition := 10
    let minuteHandPosition := 12
    let divisionsBetween := minuteHandPosition - hourHandPosition
    have h2 : divisionsBetween = 2 := by norm_num
    show 30 * divisionsBetween = 60
    calc 
      30 * 2 = 60 := by norm_num

end NUMINAMATH_GPT_smaller_angle_at_10_oclock_l713_71382


namespace NUMINAMATH_GPT_men_entered_room_l713_71314

theorem men_entered_room (M W x : ℕ) 
  (h1 : M / W = 4 / 5) 
  (h2 : M + x = 14) 
  (h3 : 2 * (W - 3) = 24) 
  (h4 : 14 = 14) 
  (h5 : 24 = 24) : x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_men_entered_room_l713_71314


namespace NUMINAMATH_GPT_inequality_solution_l713_71316

theorem inequality_solution (x : ℝ) :
  (2 / (x^2 + 2*x + 1) + 4 / (x^2 + 8*x + 7) > 3/2) ↔
  (x < -7 ∨ (-7 < x ∧ x < -1) ∨ (-1 < x)) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l713_71316


namespace NUMINAMATH_GPT_sine_shift_l713_71387

variable (m : ℝ)

theorem sine_shift (h : Real.sin 5.1 = m) : Real.sin 365.1 = m :=
by
  sorry

end NUMINAMATH_GPT_sine_shift_l713_71387


namespace NUMINAMATH_GPT_sum_ab_l713_71348

theorem sum_ab (a b : ℕ) (h1 : 1 < b) (h2 : a ^ b < 500) (h3 : ∀ x y : ℕ, (1 < y ∧ x ^ y < 500 ∧ (x + y) % 2 = 0) → a ^ b ≥ x ^ y) (h4 : (a + b) % 2 = 0) : a + b = 24 :=
  sorry

end NUMINAMATH_GPT_sum_ab_l713_71348


namespace NUMINAMATH_GPT_winter_sales_l713_71311

theorem winter_sales (spring_sales summer_sales fall_sales : ℕ) (fall_sales_pct : ℝ) (total_sales winter_sales : ℕ) :
  spring_sales = 6 →
  summer_sales = 7 →
  fall_sales = 5 →
  fall_sales_pct = 0.20 →
  fall_sales = ⌊fall_sales_pct * total_sales⌋ →
  total_sales = spring_sales + summer_sales + fall_sales + winter_sales →
  winter_sales = 7 :=
by
  sorry

end NUMINAMATH_GPT_winter_sales_l713_71311


namespace NUMINAMATH_GPT_ellipse_focus_coordinates_l713_71379

theorem ellipse_focus_coordinates (a b c : ℝ) (x1 y1 x2 y2 : ℝ) 
  (major_axis_length : 2 * a = 20) 
  (focal_relationship : c^2 = a^2 - b^2)
  (focus1_location : x1 = 3 ∧ y1 = 4) 
  (focus_c_calculation : c = Real.sqrt (x1^2 + y1^2)) :
  (x2 = -3 ∧ y2 = -4) := by
  sorry

end NUMINAMATH_GPT_ellipse_focus_coordinates_l713_71379


namespace NUMINAMATH_GPT_combined_experience_is_correct_l713_71339

-- Define the conditions as given in the problem
def james_experience : ℕ := 40
def partner_less_years : ℕ := 10
def partner_experience : ℕ := james_experience - partner_less_years

-- The combined experience of James and his partner
def combined_experience : ℕ := james_experience + partner_experience

-- Lean statement to prove the combined experience is 70 years
theorem combined_experience_is_correct : combined_experience = 70 := by sorry

end NUMINAMATH_GPT_combined_experience_is_correct_l713_71339


namespace NUMINAMATH_GPT_initial_percentage_of_milk_l713_71392

theorem initial_percentage_of_milk (P : ℝ) :
  (P / 100) * 60 = (68 / 100) * 74.11764705882354 → P = 84 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_of_milk_l713_71392


namespace NUMINAMATH_GPT_count_and_largest_special_numbers_l713_71333

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_four_digit_number (n : ℕ) : Prop := 
  1000 ≤ n ∧ n < 10000

theorem count_and_largest_special_numbers :
  ∃ (nums : List ℕ), 
    (∀ n ∈ nums, ∃ x y : ℕ, is_prime x ∧ is_prime y ∧ 
      55 * x * y = n ∧ is_four_digit_number (n * 5))
    ∧ nums.length = 3
    ∧ nums.maximum = some 4785 :=
sorry

end NUMINAMATH_GPT_count_and_largest_special_numbers_l713_71333


namespace NUMINAMATH_GPT_find_arrays_l713_71336

-- Defines a condition where positive integers satisfy the given properties
def satisfies_conditions (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a ∣ b * c * d - 1 ∧ 
  b ∣ a * c * d - 1 ∧ 
  c ∣ a * b * d - 1 ∧ 
  d ∣ a * b * c - 1

-- The theorem that any four positive integers satisfying the conditions are either (2, 3, 7, 11) or (2, 3, 11, 13)
theorem find_arrays :
  ∀ a b c d : ℕ, satisfies_conditions a b c d → 
    (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨ 
    (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) :=
by
  intro a b c d h
  sorry

end NUMINAMATH_GPT_find_arrays_l713_71336


namespace NUMINAMATH_GPT_net_profit_is_90_l713_71352

theorem net_profit_is_90
    (cost_seeds cost_soil : ℝ)
    (num_plants : ℕ)
    (price_per_plant : ℝ)
    (h0 : cost_seeds = 2)
    (h1 : cost_soil = 8)
    (h2 : num_plants = 20)
    (h3 : price_per_plant = 5) :
    (num_plants * price_per_plant - (cost_seeds + cost_soil)) = 90 := by
  sorry

end NUMINAMATH_GPT_net_profit_is_90_l713_71352


namespace NUMINAMATH_GPT_shelves_used_l713_71361

def initial_books : ℕ := 86
def books_sold : ℕ := 37
def books_per_shelf : ℕ := 7
def remaining_books : ℕ := initial_books - books_sold
def shelves : ℕ := remaining_books / books_per_shelf

theorem shelves_used : shelves = 7 := by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_shelves_used_l713_71361


namespace NUMINAMATH_GPT_point_distance_from_origin_l713_71326

theorem point_distance_from_origin (x y m : ℝ) (h1 : |y| = 15) (h2 : (x - 2)^2 + (y - 7)^2 = 169) (h3 : x > 2) :
  m = Real.sqrt (334 + 4 * Real.sqrt 105) :=
sorry

end NUMINAMATH_GPT_point_distance_from_origin_l713_71326


namespace NUMINAMATH_GPT_g_extreme_points_product_inequality_l713_71324

noncomputable def f (a x : ℝ) : ℝ := (-x^2 + a * x - a) / Real.exp x

noncomputable def f' (a x : ℝ) : ℝ := (x^2 - (a + 2) * x + 2 * a) / Real.exp x

noncomputable def g (a x : ℝ) : ℝ := (f a x + f' a x) / (x - 1)

theorem g_extreme_points_product_inequality {a x1 x2 : ℝ} 
  (h_cond1 : a > 2)
  (h_cond2 : x1 + x2 = (a + 2) / 2)
  (h_cond3 : x1 * x2 = 1)
  (h_cond4 : x1 ≠ 1 ∧ x2 ≠ 1)
  (h_x1 : x1 ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1))
  (h_x2 : x2 ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1)) :
  g a x1 * g a x2 < 4 / Real.exp 2 :=
sorry

end NUMINAMATH_GPT_g_extreme_points_product_inequality_l713_71324


namespace NUMINAMATH_GPT_verify_probabilities_l713_71381

/-- A bag contains 2 red balls, 3 black balls, and 4 white balls, all of the same size.
    A ball is drawn from the bag at a time, and once drawn, it is not replaced. -/
def total_balls := 9
def red_balls := 2
def black_balls := 3
def white_balls := 4

/-- Calculate the probability that the first ball is black and the second ball is white. -/
def prob_first_black_second_white :=
  (black_balls / total_balls) * (white_balls / (total_balls - 1))

/-- Calculate the probability that the number of draws does not exceed 3, 
    given that drawing a red ball means stopping. -/
def prob_draws_not_exceed_3 :=
  (red_balls / total_balls) +
  ((total_balls - red_balls) / total_balls) * (red_balls / (total_balls - 1)) +
  ((total_balls - red_balls - 1) / total_balls) *
  ((total_balls - red_balls) / (total_balls - 1)) *
  (red_balls / (total_balls - 2))

/-- Theorem that verifies the probabilities based on the given conditions. -/
theorem verify_probabilities :
  prob_first_black_second_white = 1 / 6 ∧
  prob_draws_not_exceed_3 = 7 / 12 :=
by
  sorry

end NUMINAMATH_GPT_verify_probabilities_l713_71381


namespace NUMINAMATH_GPT_min_value_of_sum_l713_71390

theorem min_value_of_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2 * a + b) : a + b ≥ 2 * Real.sqrt 2 + 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_l713_71390


namespace NUMINAMATH_GPT_planar_molecules_l713_71366

structure Molecule :=
  (name : String)
  (formula : String)
  (is_planar : Bool)

def propylene : Molecule := 
  { name := "Propylene", formula := "C3H6", is_planar := False }

def vinyl_chloride : Molecule := 
  { name := "Vinyl Chloride", formula := "C2H3Cl", is_planar := True }

def benzene : Molecule := 
  { name := "Benzene", formula := "C6H6", is_planar := True }

def toluene : Molecule := 
  { name := "Toluene", formula := "C7H8", is_planar := False }

theorem planar_molecules : 
  (vinyl_chloride.is_planar = True) ∧ (benzene.is_planar = True) := 
by
  sorry

end NUMINAMATH_GPT_planar_molecules_l713_71366


namespace NUMINAMATH_GPT_int_cubed_bound_l713_71334

theorem int_cubed_bound (a : ℤ) (h : 0 < a^3 ∧ a^3 < 9) : a = 1 ∨ a = 2 :=
sorry

end NUMINAMATH_GPT_int_cubed_bound_l713_71334


namespace NUMINAMATH_GPT_max_ab_bc_cd_l713_71345

-- Definitions of nonnegative numbers and their sum condition
variables (a b c d : ℕ) 
variables (h_sum : a + b + c + d = 120)

-- The goal to prove
theorem max_ab_bc_cd : ab + bc + cd <= 3600 :=
sorry

end NUMINAMATH_GPT_max_ab_bc_cd_l713_71345


namespace NUMINAMATH_GPT_total_pencils_l713_71372

def num_boxes : ℕ := 12
def pencils_per_box : ℕ := 17

theorem total_pencils : num_boxes * pencils_per_box = 204 := by
  sorry

end NUMINAMATH_GPT_total_pencils_l713_71372


namespace NUMINAMATH_GPT_calc_quotient_l713_71388

theorem calc_quotient (a b : ℕ) (h1 : a - b = 177) (h2 : 14^2 = 196) : (a - b)^2 / 196 = 144 := 
by sorry

end NUMINAMATH_GPT_calc_quotient_l713_71388


namespace NUMINAMATH_GPT_max_k_value_l713_71325

noncomputable def f (x : ℝ) := x + x * Real.log x

theorem max_k_value : ∃ k : ℤ, (∀ x > 2, k * (x - 2) < f x) ∧ k = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_k_value_l713_71325


namespace NUMINAMATH_GPT_expr_eval_l713_71300

theorem expr_eval : 180 / 6 * 2 + 5 = 65 := by
  sorry

end NUMINAMATH_GPT_expr_eval_l713_71300


namespace NUMINAMATH_GPT_factorization_problem1_factorization_problem2_l713_71301

-- Mathematical statements
theorem factorization_problem1 (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2 := by
  sorry

theorem factorization_problem2 (a : ℝ) : 18 * a^2 - 50 = 2 * (3 * a + 5) * (3 * a - 5) := by
  sorry

end NUMINAMATH_GPT_factorization_problem1_factorization_problem2_l713_71301


namespace NUMINAMATH_GPT_fraction_of_seats_taken_l713_71397

theorem fraction_of_seats_taken : 
  ∀ (total_seats broken_fraction available_seats : ℕ), 
    total_seats = 500 → 
    broken_fraction = 1 / 10 → 
    available_seats = 250 → 
    (total_seats - available_seats - total_seats * broken_fraction) / total_seats = 2 / 5 :=
by
  intro total_seats broken_fraction available_seats
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_fraction_of_seats_taken_l713_71397


namespace NUMINAMATH_GPT_initial_money_l713_71367

def cost_of_game : Nat := 47
def cost_of_toy : Nat := 7
def number_of_toys : Nat := 3

theorem initial_money (initial_amount : Nat) (remaining_amount : Nat) :
  initial_amount = cost_of_game + remaining_amount →
  remaining_amount = number_of_toys * cost_of_toy →
  initial_amount = 68 := by
    sorry

end NUMINAMATH_GPT_initial_money_l713_71367


namespace NUMINAMATH_GPT_greatest_multiple_of_4_l713_71313

/-- 
Given x is a positive multiple of 4 and x^3 < 2000, 
prove that x is at most 12 and 
x = 12 is the greatest value that satisfies these conditions. 
-/
theorem greatest_multiple_of_4 (x : ℕ) (hx1 : x % 4 = 0) (hx2 : x^3 < 2000) : x ≤ 12 ∧ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_greatest_multiple_of_4_l713_71313
