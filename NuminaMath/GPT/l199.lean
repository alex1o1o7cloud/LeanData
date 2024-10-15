import Mathlib

namespace NUMINAMATH_GPT_no_consecutive_nat_mul_eq_25k_plus_1_l199_19974

theorem no_consecutive_nat_mul_eq_25k_plus_1 (k : ℕ) : 
  ¬ ∃ n : ℕ, n * (n + 1) = 25 * k + 1 :=
sorry

end NUMINAMATH_GPT_no_consecutive_nat_mul_eq_25k_plus_1_l199_19974


namespace NUMINAMATH_GPT_smallest_possible_l_l199_19921

theorem smallest_possible_l (a b c L : ℕ) (h1 : a * b = 7) (h2 : a * c = 27) (h3 : b * c = L) (h4 : ∃ k, a * b * c = k * k) : L = 21 := sorry

end NUMINAMATH_GPT_smallest_possible_l_l199_19921


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l199_19994

theorem hyperbola_eccentricity (a b c : ℝ) (h₁ : 2 * a = 16) (h₂ : 2 * b = 12) (h₃ : c = Real.sqrt (a^2 + b^2)) :
  (c / a) = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l199_19994


namespace NUMINAMATH_GPT_value_of_abc_l199_19952

theorem value_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + 1 / b = 5) (h2 : b + 1 / c = 2) (h3 : c + 1 / a = 3) : 
  abc = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_abc_l199_19952


namespace NUMINAMATH_GPT_groups_needed_l199_19986

theorem groups_needed (h_camper_count : 36 > 0) (h_group_limit : 12 > 0) : 
  ∃ x : ℕ, x = 36 / 12 ∧ x = 3 := by
  sorry

end NUMINAMATH_GPT_groups_needed_l199_19986


namespace NUMINAMATH_GPT_katy_books_ratio_l199_19930

theorem katy_books_ratio (J : ℕ) (H1 : 8 + J + (J - 3) = 37) : J / 8 = 2 := 
by
  sorry

end NUMINAMATH_GPT_katy_books_ratio_l199_19930


namespace NUMINAMATH_GPT_max_integer_a_l199_19998

theorem max_integer_a :
  ∀ (a: ℤ), (∀ x: ℝ, (a + 1) * x^2 - 2 * x + 3 = 0 → (a = -2 → (-12 * a - 8) ≥ 0)) → (∀ a ≤ -2, a ≠ -1) :=
by
  sorry

end NUMINAMATH_GPT_max_integer_a_l199_19998


namespace NUMINAMATH_GPT_tetrahedron_point_choice_l199_19980

-- Definitions
variables (h s1 s2 : ℝ) -- h, s1, s2 are positive real numbers
variables (A B C : ℝ)  -- A, B, C can be points in space

-- Hypothetical tetrahedron face areas and height
def height_condition (D : ℝ) : Prop := -- D is a point in space
  ∃ (D_height : ℝ), D_height = h

def area_ACD_condition (D : ℝ) : Prop := 
  ∃ (area_ACD : ℝ), area_ACD = s1

def area_BCD_condition (D : ℝ) : Prop := 
  ∃ (area_BCD : ℝ), area_BCD = s2

-- The main theorem
theorem tetrahedron_point_choice : 
  ∃ D, height_condition h D ∧ area_ACD_condition s1 D ∧ area_BCD_condition s2 D :=
sorry

end NUMINAMATH_GPT_tetrahedron_point_choice_l199_19980


namespace NUMINAMATH_GPT_quadratic_no_ten_powers_of_2_values_l199_19961

theorem quadratic_no_ten_powers_of_2_values 
  (a b : ℝ) :
  ¬ ∃ (j : ℤ), ∀ k : ℤ, j ≤ k ∧ k < j + 10 → ∃ n : ℕ, (k^2 + a * k + b) = 2 ^ n :=
by sorry

end NUMINAMATH_GPT_quadratic_no_ten_powers_of_2_values_l199_19961


namespace NUMINAMATH_GPT_circle_center_sum_l199_19938

theorem circle_center_sum (h k : ℝ) :
  (∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = x ^ 2 + y ^ 2 - 6 * x - 8 * y + 38) → h + k = 7 :=
by sorry

end NUMINAMATH_GPT_circle_center_sum_l199_19938


namespace NUMINAMATH_GPT_exponent_product_l199_19977

theorem exponent_product (a : ℝ) (m n : ℕ)
  (h1 : a^m = 2) (h2 : a^n = 5) : a^(2*m + n) = 20 :=
sorry

end NUMINAMATH_GPT_exponent_product_l199_19977


namespace NUMINAMATH_GPT_option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l199_19926

theorem option_A_correct (a : ℝ) : a ^ 2 * a ^ 3 = a ^ 5 := by {
  -- Here, we would provide the proof if required,
  -- but we are only stating the theorem.
  sorry
}

-- You may optionally add definitions of incorrect options for completeness.
theorem option_B_incorrect (a : ℝ) : ¬(a + 2 * a = 3 * a ^ 2) := by {
  sorry
}

theorem option_C_incorrect (a b : ℝ) : ¬((a * b) ^ 3 = a * b ^ 3) := by {
  sorry
}

theorem option_D_incorrect (a : ℝ) : ¬((-a ^ 3) ^ 2 = -a ^ 6) := by {
  sorry
}

end NUMINAMATH_GPT_option_A_correct_option_B_incorrect_option_C_incorrect_option_D_incorrect_l199_19926


namespace NUMINAMATH_GPT_map_distance_8_cm_l199_19904

-- Define the conditions
def scale : ℕ := 5000000
def actual_distance_km : ℕ := 400
def actual_distance_cm : ℕ := 40000000
def map_distance_cm (x : ℕ) : Prop := x * scale = actual_distance_cm

-- The theorem to be proven
theorem map_distance_8_cm : ∃ x : ℕ, map_distance_cm x ∧ x = 8 :=
by
  use 8
  unfold map_distance_cm
  norm_num
  sorry

end NUMINAMATH_GPT_map_distance_8_cm_l199_19904


namespace NUMINAMATH_GPT_fraction_computation_l199_19968

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_computation_l199_19968


namespace NUMINAMATH_GPT_find_x_l199_19917

-- Definitions from the conditions
def isPositiveMultipleOf7 (x : ℕ) : Prop := ∃ k : ℕ, x = 7 * k ∧ x > 0
def xSquaredGreaterThan150 (x : ℕ) : Prop := x^2 > 150
def xLessThan40 (x : ℕ) : Prop := x < 40

-- Main problem statement
theorem find_x (x : ℕ) (h1 : isPositiveMultipleOf7 x) (h2 : xSquaredGreaterThan150 x) (h3 : xLessThan40 x) : x = 14 :=
sorry

end NUMINAMATH_GPT_find_x_l199_19917


namespace NUMINAMATH_GPT_TV_cost_l199_19987

theorem TV_cost (savings_furniture_fraction : ℚ)
                (original_savings : ℝ)
                (spent_on_furniture : ℝ)
                (spent_on_TV : ℝ)
                (hfurniture : savings_furniture_fraction = 2/4)
                (hsavings : original_savings = 600)
                (hspent_furniture : spent_on_furniture = original_savings * savings_furniture_fraction) :
                spent_on_TV = 300 := 
sorry

end NUMINAMATH_GPT_TV_cost_l199_19987


namespace NUMINAMATH_GPT_intersection_point_of_lines_l199_19900

theorem intersection_point_of_lines :
  ∃ x y : ℝ, (x - 2 * y - 4 = 0) ∧ (x + 3 * y + 6 = 0) ∧ (x = 0) ∧ (y = -2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l199_19900


namespace NUMINAMATH_GPT_eval_polynomial_positive_root_l199_19959

theorem eval_polynomial_positive_root : 
  ∃ x : ℝ, (x^2 - 3 * x - 10 = 0 ∧ 0 < x ∧ (x^3 - 3 * x^2 - 9 * x + 7 = 12)) :=
sorry

end NUMINAMATH_GPT_eval_polynomial_positive_root_l199_19959


namespace NUMINAMATH_GPT_semicircle_radius_l199_19912

noncomputable def radius_of_semicircle (P : ℝ) (h : P = 144) : ℝ :=
  144 / (Real.pi + 2)

theorem semicircle_radius (P : ℝ) (h : P = 144) : radius_of_semicircle P h = 144 / (Real.pi + 2) :=
  by sorry

end NUMINAMATH_GPT_semicircle_radius_l199_19912


namespace NUMINAMATH_GPT_length_of_unfenced_side_l199_19936

theorem length_of_unfenced_side
  (L W : ℝ)
  (h1 : L * W = 200)
  (h2 : 2 * W + L = 50) :
  L = 10 :=
sorry

end NUMINAMATH_GPT_length_of_unfenced_side_l199_19936


namespace NUMINAMATH_GPT_jose_fewer_rocks_l199_19982

theorem jose_fewer_rocks (J : ℕ) (H1 : 80 = J + 14) (H2 : J + 20 = 86) (H3 : J < 80) : J = 66 :=
by
  -- Installation of other conditions derived from the proof
  have H_albert_collected : 86 = 80 + 6 := by rfl
  have J_def : J = 86 - 20 := by sorry
  sorry

end NUMINAMATH_GPT_jose_fewer_rocks_l199_19982


namespace NUMINAMATH_GPT_square_area_l199_19989

noncomputable def side_length_x (x : ℚ) : Prop :=
5 * x - 20 = 30 - 4 * x

noncomputable def side_length_s : ℚ :=
70 / 9

noncomputable def area_of_square : ℚ :=
(side_length_s)^2

theorem square_area (x : ℚ) (h : side_length_x x) : area_of_square = 4900 / 81 :=
sorry

end NUMINAMATH_GPT_square_area_l199_19989


namespace NUMINAMATH_GPT_velocity_at_t4_acceleration_is_constant_l199_19951

noncomputable def s (t : ℝ) : ℝ := 3 * t^2 - 3 * t + 8

def v (t : ℝ) : ℝ := 6 * t - 3

def a : ℝ := 6

theorem velocity_at_t4 : v 4 = 21 := by 
  sorry

theorem acceleration_is_constant : a = 6 := by 
  sorry

end NUMINAMATH_GPT_velocity_at_t4_acceleration_is_constant_l199_19951


namespace NUMINAMATH_GPT_abs_x_minus_2y_is_square_l199_19903

theorem abs_x_minus_2y_is_square (x y : ℕ) (h : ∃ k : ℤ, x^2 - 4 * y + 1 = (x - 2 * y) * (1 - 2 * y) * k) : ∃ m : ℕ, x - 2 * y = m ^ 2 := by
  sorry

end NUMINAMATH_GPT_abs_x_minus_2y_is_square_l199_19903


namespace NUMINAMATH_GPT_find_f_difference_l199_19906

variable {α : Type*}
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_period : ∀ x, f (x + 5) = f x)
variable (h_value : f (-2) = 2)

theorem find_f_difference : f 2012 - f 2010 = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_f_difference_l199_19906


namespace NUMINAMATH_GPT_petya_numbers_board_l199_19979

theorem petya_numbers_board (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k → k < n → (∀ d : ℕ, 4 ∣ 10 ^ d → ¬(4 ∣ k))) 
  (h3 : ∀ k : ℕ, 0 ≤ k → k < n→ (∀ d : ℕ, 7 ∣ 10 ^ d → ¬(7 ∣ (k + n - 1)))) : 
  ∃ x : ℕ, (x = 2021) := 
by
  sorry

end NUMINAMATH_GPT_petya_numbers_board_l199_19979


namespace NUMINAMATH_GPT_score_87_not_possible_l199_19962

def max_score := 15 * 6
def score (correct unanswered incorrect : ℕ) := 6 * correct + unanswered

theorem score_87_not_possible :
  ¬∃ (correct unanswered incorrect : ℕ), 
    correct + unanswered + incorrect = 15 ∧
    6 * correct + unanswered = 87 := 
sorry

end NUMINAMATH_GPT_score_87_not_possible_l199_19962


namespace NUMINAMATH_GPT_y_at_x8_l199_19901

theorem y_at_x8 (k : ℝ) (y : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_y_at_x8_l199_19901


namespace NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_is_495_l199_19909

-- Define the necessary conditions
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

def is_less_than_500 (n : ℕ) : Prop :=
  n < 500

-- Problem statement: Prove that the largest positive multiple of 15 less than 500 is 495
theorem largest_multiple_of_15_less_than_500_is_495 : 
  ∀ n : ℕ, is_multiple_of_15 n → is_less_than_500 n → n ≤ 495 := 
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_is_495_l199_19909


namespace NUMINAMATH_GPT_sufficient_condition_for_m_ge_9_l199_19943

theorem sufficient_condition_for_m_ge_9
  (x m : ℝ)
  (p : |x - 4| ≤ 6)
  (q : x ≤ 1 + m)
  (h_sufficient : ∀ x, |x - 4| ≤ 6 → x ≤ 1 + m)
  (h_not_necessary : ∃ x, ¬(|x - 4| ≤ 6) ∧ x ≤ 1 + m) :
  m ≥ 9 := 
sorry

end NUMINAMATH_GPT_sufficient_condition_for_m_ge_9_l199_19943


namespace NUMINAMATH_GPT_average_price_of_rackets_l199_19958

theorem average_price_of_rackets (total_amount : ℝ) (number_of_pairs : ℕ) (average_price : ℝ) 
  (h1 : total_amount = 588) (h2 : number_of_pairs = 60) : average_price = 9.80 :=
by
  sorry

end NUMINAMATH_GPT_average_price_of_rackets_l199_19958


namespace NUMINAMATH_GPT_average_difference_l199_19956

theorem average_difference : 
  (500 + 1000) / 2 - (100 + 500) / 2 = 450 := 
by
  sorry

end NUMINAMATH_GPT_average_difference_l199_19956


namespace NUMINAMATH_GPT_appropriate_presentation_length_l199_19933

-- Definitions and conditions
def ideal_speaking_rate : ℕ := 160
def min_minutes : ℕ := 20
def max_minutes : ℕ := 40
def appropriate_words_range (words : ℕ) : Prop :=
  words ≥ (min_minutes * ideal_speaking_rate) ∧ words ≤ (max_minutes * ideal_speaking_rate)

-- Statement to prove
theorem appropriate_presentation_length : appropriate_words_range 5000 :=
by sorry

end NUMINAMATH_GPT_appropriate_presentation_length_l199_19933


namespace NUMINAMATH_GPT_original_number_of_men_l199_19948

theorem original_number_of_men 
    (x : ℕ) 
    (h : x * 40 = (x - 5) * 60) : x = 15 := 
sorry

end NUMINAMATH_GPT_original_number_of_men_l199_19948


namespace NUMINAMATH_GPT_winning_candidate_percentage_l199_19960

theorem winning_candidate_percentage
  (majority_difference : ℕ)
  (total_valid_votes : ℕ)
  (P : ℕ)
  (h1 : majority_difference = 192)
  (h2 : total_valid_votes = 480)
  (h3 : 960 * P = 67200) : 
  P = 70 := by
  sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l199_19960


namespace NUMINAMATH_GPT_max_quarters_l199_19925

theorem max_quarters (q : ℕ) (h1 : q + q + q / 2 = 20): q ≤ 11 :=
by
  sorry

end NUMINAMATH_GPT_max_quarters_l199_19925


namespace NUMINAMATH_GPT_range_of_a_plus_b_l199_19984

noncomputable def range_of_sum_of_sides (a b : ℝ) (c : ℝ) : Prop :=
  (2 < a + b ∧ a + b ≤ 4)

theorem range_of_a_plus_b
  (a b c : ℝ)
  (h1 : (2 * (b ^ 2 - (1/2) * a * b) = b ^ 2 + 4 - a ^ 2))
  (h2 : c = 2) :
  range_of_sum_of_sides a b c :=
by
  -- Proof would go here, but it's omitted as per the instructions.
  sorry

end NUMINAMATH_GPT_range_of_a_plus_b_l199_19984


namespace NUMINAMATH_GPT_charlie_received_495_l199_19916

theorem charlie_received_495 : 
  ∃ (A B C x : ℕ), 
    A + B + C = 1105 ∧ 
    A - 10 = 11 * x ∧ 
    B - 20 = 18 * x ∧ 
    C - 15 = 24 * x ∧ 
    C = 495 := 
by
  sorry

end NUMINAMATH_GPT_charlie_received_495_l199_19916


namespace NUMINAMATH_GPT_no_four_digit_number_differs_from_reverse_by_1008_l199_19990

theorem no_four_digit_number_differs_from_reverse_by_1008 :
  ∀ a b c d : ℕ, 
  a < 10 → b < 10 → c < 10 → d < 10 → (999 * (a - d) + 90 * (b - c) ≠ 1008) :=
by
  intro a b c d ha hb hc hd h
  sorry

end NUMINAMATH_GPT_no_four_digit_number_differs_from_reverse_by_1008_l199_19990


namespace NUMINAMATH_GPT_problem_1_problem_2_l199_19949

open Set

-- First problem: when a = 2
theorem problem_1:
  ∀ (x : ℝ), 2 * x^2 - x - 1 > 0 ↔ (x < -(1 / 2) ∨ x > 1) :=
by
  sorry

-- Second problem: when a > -1
theorem problem_2 (a : ℝ) (h : a > -1) :
  ∀ (x : ℝ), 
    (if a = 0 then x - 1 > 0 else if a > 0 then  a * x ^ 2 + (1 - a) * x - 1 > 0 ↔ (x < -1 / a ∨ x > 1) 
    else a * x ^ 2 + (1 - a) * x - 1 > 0 ↔ (1 < x ∧ x < -1 / a)) :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l199_19949


namespace NUMINAMATH_GPT_maximize_profit_l199_19978

noncomputable def selling_price_to_maximize_profit (original_price selling_price : ℝ) (units units_sold_decrease : ℝ) : ℝ :=
  let x := 5
  let optimal_selling_price := selling_price + x
  optimal_selling_price

theorem maximize_profit :
  selling_price_to_maximize_profit 80 90 400 20 = 95 :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l199_19978


namespace NUMINAMATH_GPT_angle_between_clock_hands_at_7_25_l199_19939

theorem angle_between_clock_hands_at_7_25 : 
  let degrees_per_hour := 30
  let minute_hand_position := (25 / 60 * 360 : ℝ)
  let hour_hand_position := (7 * degrees_per_hour + (25 / 60 * degrees_per_hour) : ℝ)
  abs (hour_hand_position - minute_hand_position) = 72.5 
  := by
  let degrees_per_hour := 30
  let minute_hand_position := (25 / 60 * 360 : ℝ)
  let hour_hand_position := (7 * degrees_per_hour + (25 / 60 * degrees_per_hour) : ℝ)
  sorry

end NUMINAMATH_GPT_angle_between_clock_hands_at_7_25_l199_19939


namespace NUMINAMATH_GPT_base_conversion_l199_19957

theorem base_conversion (b : ℕ) (h_pos : b > 0) :
  (1 * 6 ^ 2 + 2 * 6 ^ 1 + 5 * 6 ^ 0 = 2 * b ^ 2 + 2 * b + 1) → b = 4 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_l199_19957


namespace NUMINAMATH_GPT_find_k_l199_19983

theorem find_k 
  (k : ℝ) 
  (m_eq : ∀ x : ℝ, ∃ y : ℝ, y = 3 * x + 5)
  (n_eq : ∀ x : ℝ, ∃ y : ℝ, y = k * x - 7) 
  (intersection : ∃ x y : ℝ, (y = 3 * x + 5) ∧ (y = k * x - 7) ∧ x = -4 ∧ y = -7) :
  k = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l199_19983


namespace NUMINAMATH_GPT_find_angle_C_l199_19969

open Real

theorem find_angle_C (a b C A B : ℝ) 
  (h1 : a^2 + b^2 = 6 * a * b * cos C)
  (h2 : sin C ^ 2 = 2 * sin A * sin B) :
  C = π / 3 := 
  sorry

end NUMINAMATH_GPT_find_angle_C_l199_19969


namespace NUMINAMATH_GPT_average_of_rest_equals_40_l199_19972

-- Defining the initial conditions
def total_students : ℕ := 20
def high_scorers : ℕ := 2
def low_scorers : ℕ := 3
def class_average : ℚ := 40

-- The target function to calculate the average of the rest of the students
def average_rest_students (total_students high_scorers low_scorers : ℕ) (class_average : ℚ) : ℚ :=
  let total_marks := total_students * class_average
  let high_scorer_marks := 100 * high_scorers
  let low_scorer_marks := 0 * low_scorers
  let rest_marks := total_marks - (high_scorer_marks + low_scorer_marks)
  let rest_students := total_students - high_scorers - low_scorers
  rest_marks / rest_students

-- The theorem to prove that the average of the rest of the students is 40
theorem average_of_rest_equals_40 : average_rest_students total_students high_scorers low_scorers class_average = 40 := 
by
  sorry

end NUMINAMATH_GPT_average_of_rest_equals_40_l199_19972


namespace NUMINAMATH_GPT_total_value_proof_l199_19950

def total_bills : ℕ := 126
def five_dollar_bills : ℕ := 84
def ten_dollar_bills : ℕ := total_bills - five_dollar_bills
def value_five_dollar_bills : ℕ := five_dollar_bills * 5
def value_ten_dollar_bills : ℕ := ten_dollar_bills * 10
def total_value : ℕ := value_five_dollar_bills + value_ten_dollar_bills

theorem total_value_proof : total_value = 840 := by
  unfold total_value value_five_dollar_bills value_ten_dollar_bills
  unfold five_dollar_bills ten_dollar_bills total_bills
  -- Calculation steps to show that value_five_dollar_bills + value_ten_dollar_bills = 840
  sorry

end NUMINAMATH_GPT_total_value_proof_l199_19950


namespace NUMINAMATH_GPT_tan_600_eq_neg_sqrt_3_l199_19908

theorem tan_600_eq_neg_sqrt_3 : Real.tan (600 * Real.pi / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_600_eq_neg_sqrt_3_l199_19908


namespace NUMINAMATH_GPT_number_of_handshakes_l199_19931

-- Definitions based on the conditions:
def number_of_teams : ℕ := 4
def number_of_women_per_team : ℕ := 2
def total_women : ℕ := number_of_teams * number_of_women_per_team

-- Each woman shakes hands with all others except her partner
def handshakes_per_woman : ℕ := total_women - 1 - (number_of_women_per_team - 1)

-- Calculate total handshakes, considering each handshake is counted twice
def total_handshakes : ℕ := (total_women * handshakes_per_woman) / 2

-- Statement to prove
theorem number_of_handshakes :
  total_handshakes = 24 := 
sorry

end NUMINAMATH_GPT_number_of_handshakes_l199_19931


namespace NUMINAMATH_GPT_total_days_spent_on_islands_l199_19971

-- Define the conditions and question in Lean 4
def first_expedition_A_weeks := 3
def second_expedition_A_weeks := first_expedition_A_weeks + 2
def last_expedition_A_weeks := second_expedition_A_weeks * 2

def first_expedition_B_weeks := 5
def second_expedition_B_weeks := first_expedition_B_weeks - 3
def last_expedition_B_weeks := first_expedition_B_weeks

def total_weeks_on_island_A := first_expedition_A_weeks + second_expedition_A_weeks + last_expedition_A_weeks
def total_weeks_on_island_B := first_expedition_B_weeks + second_expedition_B_weeks + last_expedition_B_weeks

def total_weeks := total_weeks_on_island_A + total_weeks_on_island_B
def total_days := total_weeks * 7

theorem total_days_spent_on_islands : total_days = 210 :=
by
  -- We skip the proof part
  sorry

end NUMINAMATH_GPT_total_days_spent_on_islands_l199_19971


namespace NUMINAMATH_GPT_Francine_not_working_days_l199_19915

-- Conditions
variables (d : ℕ) -- Number of days Francine works each week
def distance_per_day : ℕ := 140 -- Distance Francine drives each day
def total_distance_4_weeks : ℕ := 2240 -- Total distance in 4 weeks
def days_per_week : ℕ := 7 -- Days in a week

-- Proving that the number of days she does not go to work every week is 3
theorem Francine_not_working_days :
  (4 * distance_per_day * d = total_distance_4_weeks) →
  ((days_per_week - d) = 3) :=
by sorry

end NUMINAMATH_GPT_Francine_not_working_days_l199_19915


namespace NUMINAMATH_GPT_omitted_angle_of_convex_polygon_l199_19940

theorem omitted_angle_of_convex_polygon (calculated_sum : ℕ) (omitted_angle : ℕ)
    (h₁ : calculated_sum = 2583) (h₂ : omitted_angle = 2700 - 2583) :
    omitted_angle = 117 :=
by
  sorry

end NUMINAMATH_GPT_omitted_angle_of_convex_polygon_l199_19940


namespace NUMINAMATH_GPT_astronaut_revolutions_l199_19985

theorem astronaut_revolutions (n : ℤ) (R : ℝ) (hn : n > 2) :
    ∃ k : ℤ, k = n - 1 := 
sorry

end NUMINAMATH_GPT_astronaut_revolutions_l199_19985


namespace NUMINAMATH_GPT_percentage_of_180_out_of_360_equals_50_l199_19941

theorem percentage_of_180_out_of_360_equals_50 :
  (180 / 360 : ℚ) * 100 = 50 := 
sorry

end NUMINAMATH_GPT_percentage_of_180_out_of_360_equals_50_l199_19941


namespace NUMINAMATH_GPT_time_difference_halfway_point_l199_19947

theorem time_difference_halfway_point 
  (T_d : ℝ) 
  (T_s : ℝ := 2 * T_d) 
  (H_d : ℝ := T_d / 2) 
  (H_s : ℝ := T_s / 2) 
  (diff_time : ℝ := H_s - H_d) : 
  T_d = 35 →
  T_s = 2 * T_d →
  diff_time = 17.5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_time_difference_halfway_point_l199_19947


namespace NUMINAMATH_GPT_recreation_percentage_l199_19992

theorem recreation_percentage (W : ℝ) (hW : W > 0) :
  (0.40 * W) / (0.15 * W) * 100 = 267 := by
  sorry

end NUMINAMATH_GPT_recreation_percentage_l199_19992


namespace NUMINAMATH_GPT_total_tickets_correct_l199_19963

-- Let's define the conditions given in the problem
def student_tickets (adult_tickets : ℕ) := 2 * adult_tickets
def adult_tickets := 122
def total_tickets := adult_tickets + student_tickets adult_tickets

-- We now state the theorem to be proved
theorem total_tickets_correct : total_tickets = 366 :=
by 
  sorry

end NUMINAMATH_GPT_total_tickets_correct_l199_19963


namespace NUMINAMATH_GPT_onions_total_l199_19996

theorem onions_total (Sara : ℕ) (Sally : ℕ) (Fred : ℕ)
  (hSara : Sara = 4) (hSally : Sally = 5) (hFred : Fred = 9) :
  Sara + Sally + Fred = 18 :=
by
  sorry

end NUMINAMATH_GPT_onions_total_l199_19996


namespace NUMINAMATH_GPT_length_of_train_l199_19927

-- We state the conditions as definitions.
def length_of_train_equals_length_of_platform (l_train l_platform : ℝ) : Prop :=
l_train = l_platform

def speed_of_train (s : ℕ) : Prop :=
s = 216

def crossing_time (t : ℕ) : Prop :=
t = 1

-- Defining the goal according to the problem statement.
theorem length_of_train (l_train l_platform : ℝ) (s t : ℕ) 
  (h1 : length_of_train_equals_length_of_platform l_train l_platform) 
  (h2 : speed_of_train s) 
  (h3 : crossing_time t) : 
  l_train = 1800 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l199_19927


namespace NUMINAMATH_GPT_train_pass_platform_time_l199_19944

-- Define the conditions given in the problem.
def train_length : ℕ := 1200
def platform_length : ℕ := 1100
def time_to_cross_tree : ℕ := 120

-- Define the calculation for speed.
def speed := train_length / time_to_cross_tree

-- Define the combined length of train and platform.
def combined_length := train_length + platform_length

-- Define the expected time to pass the platform.
def expected_time_to_pass_platform := combined_length / speed

-- The theorem to prove.
theorem train_pass_platform_time :
  expected_time_to_pass_platform = 230 :=
by {
  -- Placeholder for the proof.
  sorry
}

end NUMINAMATH_GPT_train_pass_platform_time_l199_19944


namespace NUMINAMATH_GPT_linear_function_no_third_quadrant_l199_19967

theorem linear_function_no_third_quadrant (m : ℝ) (h : ∀ x y : ℝ, x < 0 → y < 0 → y ≠ -2 * x + 1 - m) : 
  m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_no_third_quadrant_l199_19967


namespace NUMINAMATH_GPT_number_of_ways_to_express_n_as_sum_l199_19965

noncomputable def P (n k : ℕ) : ℕ := sorry
noncomputable def Q (n k : ℕ) : ℕ := sorry

theorem number_of_ways_to_express_n_as_sum (n : ℕ) (k : ℕ) (h : k ≥ 2) : P n k = Q n k := sorry

end NUMINAMATH_GPT_number_of_ways_to_express_n_as_sum_l199_19965


namespace NUMINAMATH_GPT_geometric_series_sum_l199_19970

theorem geometric_series_sum :
  (1 / 3 - 1 / 6 + 1 / 12 - 1 / 24 + 1 / 48 - 1 / 96) = 7 / 32 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l199_19970


namespace NUMINAMATH_GPT_gcd_sum_of_cubes_l199_19905

-- Define the problem conditions
variables (n : ℕ) (h_pos : n > 27)

-- Define the goal to prove
theorem gcd_sum_of_cubes (h : n > 27) : 
  gcd (n^3 + 27) (n + 3) = n + 3 :=
by sorry

end NUMINAMATH_GPT_gcd_sum_of_cubes_l199_19905


namespace NUMINAMATH_GPT_total_trees_l199_19929

-- Definitions based on the conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3

-- Theorem stating the total number of apple trees planted by Ava and Lily
theorem total_trees : ava_trees + lily_trees = 15 := by
  -- We skip the proof for now
  sorry

end NUMINAMATH_GPT_total_trees_l199_19929


namespace NUMINAMATH_GPT_total_area_of_squares_l199_19966

-- Condition 1: Definition of the side length
def side_length (s : ℝ) : Prop := s = 12

-- Condition 2: Definition of the center of one square coinciding with the vertex of another
-- Here, we assume the positions are fixed so this condition is given
def coincide_center_vertex (s₁ s₂ : ℝ) : Prop := s₁ = s₂ 

-- The main theorem statement
theorem total_area_of_squares
  (s₁ s₂ : ℝ) 
  (h₁ : side_length s₁)
  (h₂ : side_length s₂)
  (h₃ : coincide_center_vertex s₁ s₂) :
  (2 * s₁^2) - (s₁^2 / 4) = 252 :=
by
  sorry

end NUMINAMATH_GPT_total_area_of_squares_l199_19966


namespace NUMINAMATH_GPT_neg_sub_eq_sub_l199_19945

theorem neg_sub_eq_sub (a b : ℝ) : - (a - b) = b - a := 
by
  sorry

end NUMINAMATH_GPT_neg_sub_eq_sub_l199_19945


namespace NUMINAMATH_GPT_existence_of_unusual_100_digit_numbers_l199_19973

theorem existence_of_unusual_100_digit_numbers :
  ∃ (n₁ n₂ : ℕ), 
  (n₁ = 10^100 - 1) ∧ (n₂ = 5 * 10^99 - 1) ∧ 
  (∀ x : ℕ, x = n₁ → (x^3 % 10^100 = x) ∧ (x^2 % 10^100 ≠ x)) ∧
  (∀ x : ℕ, x = n₂ → (x^3 % 10^100 = x) ∧ (x^2 % 10^100 ≠ x)) := 
sorry

end NUMINAMATH_GPT_existence_of_unusual_100_digit_numbers_l199_19973


namespace NUMINAMATH_GPT_systematic_sampling_probabilities_l199_19920

-- Define the total number of students
def total_students : ℕ := 1005

-- Define the sample size
def sample_size : ℕ := 50

-- Define the number of individuals removed
def individuals_removed : ℕ := 5

-- Define the probability of an individual being removed
def probability_removed : ℚ := individuals_removed / total_students

-- Define the probability of an individual being selected in the sample
def probability_selected : ℚ := sample_size / total_students

-- The statement we need to prove
theorem systematic_sampling_probabilities :
  probability_removed = 5 / 1005 ∧ probability_selected = 50 / 1005 :=
sorry

end NUMINAMATH_GPT_systematic_sampling_probabilities_l199_19920


namespace NUMINAMATH_GPT_circular_permutation_divisible_41_l199_19934

theorem circular_permutation_divisible_41 (N : ℤ) (a b c d e : ℤ) (h : N = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e)
  (h41 : 41 ∣ N) :
  ∀ (k : ℕ), 41 ∣ (10^((k % 5) * (4 - (k / 5))) * a + 10^((k % 5) * 3 + (k / 5) * 4) * b + 10^((k % 5) * 2 + (k / 5) * 3) * c + 10^((k % 5) + (k / 5) * 2) * d + 10^(k / 5) * e) :=
sorry

end NUMINAMATH_GPT_circular_permutation_divisible_41_l199_19934


namespace NUMINAMATH_GPT_parallelogram_area_l199_19918

theorem parallelogram_area (s : ℝ) (ratio : ℝ) (A : ℝ) :
  s = 3 → ratio = 2 * Real.sqrt 2 → A = 9 → 
  (A * ratio = 18 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l199_19918


namespace NUMINAMATH_GPT_mika_jogging_speed_l199_19955

theorem mika_jogging_speed 
  (s : ℝ)  -- Mika's constant jogging speed in meters per second.
  (r : ℝ)  -- Radius of the inner semicircle.
  (L : ℝ)  -- Length of each straight section.
  (h1 : 8 > 0) -- Overall width of the track is 8 meters.
  (h2 : (2 * L + 2 * π * (r + 8)) / s = (2 * L + 2 * π * r) / s + 48) -- Time difference equation.
  : s = π / 3 := 
sorry

end NUMINAMATH_GPT_mika_jogging_speed_l199_19955


namespace NUMINAMATH_GPT_factor_w4_minus_16_l199_19954

theorem factor_w4_minus_16 (w : ℝ) : (w^4 - 16) = (w - 2) * (w + 2) * (w^2 + 4) :=
by
    sorry

end NUMINAMATH_GPT_factor_w4_minus_16_l199_19954


namespace NUMINAMATH_GPT_payment_to_y_l199_19922

theorem payment_to_y (X Y : ℝ) (h1 : X = 1.2 * Y) (h2 : X + Y = 580) : Y = 263.64 :=
by
  sorry

end NUMINAMATH_GPT_payment_to_y_l199_19922


namespace NUMINAMATH_GPT_find_pq_l199_19991

-- Define the constants function for the given equation and form
noncomputable def quadratic_eq (p q r : ℤ) : (ℤ × ℤ × ℤ) :=
(2*p*q, p^2 + 2*p*q + q^2 + r, q*q + r)

-- Define the theorem we want to prove
theorem find_pq (p q r: ℤ) (h : quadratic_eq 2 q r = (8, -24, -56)) : pq = -12 :=
by sorry

end NUMINAMATH_GPT_find_pq_l199_19991


namespace NUMINAMATH_GPT_inverse_of_f_l199_19995

def f (x : ℝ) : ℝ := 7 - 3 * x

noncomputable def f_inv (x : ℝ) : ℝ := (7 - x) / 3

theorem inverse_of_f : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x :=
by
  intros
  sorry

end NUMINAMATH_GPT_inverse_of_f_l199_19995


namespace NUMINAMATH_GPT_common_difference_arithmetic_sequence_l199_19976

variable (n d : ℝ) (a : ℝ := 7 - 2 * d) (an : ℝ := 37) (Sn : ℝ := 198)

theorem common_difference_arithmetic_sequence :
  7 + (n - 3) * d = 37 ∧ 
  396 = n * (44 - 2 * d) ∧
  Sn = n / 2 * (a + an) →
  (∃ d : ℝ, 7 + (n - 3) * d = 37 ∧ 396 = n * (44 - 2 * d)) :=
by
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_sequence_l199_19976


namespace NUMINAMATH_GPT_birches_count_l199_19914

-- Define the problem conditions
def total_trees : ℕ := 4000
def percentage_spruces : ℕ := 10
def percentage_pines : ℕ := 13
def number_spruces : ℕ := (percentage_spruces * total_trees) / 100
def number_pines : ℕ := (percentage_pines * total_trees) / 100
def number_oaks : ℕ := number_spruces + number_pines
def number_birches : ℕ := total_trees - number_oaks - number_pines - number_spruces

-- Prove the number of birches is 2160
theorem birches_count : number_birches = 2160 := by
  sorry

end NUMINAMATH_GPT_birches_count_l199_19914


namespace NUMINAMATH_GPT_fruit_platter_has_thirty_fruits_l199_19946

-- Define the conditions
def at_least_five_apples (g_apple r_apple y_apple : ℕ) : Prop :=
  g_apple + r_apple + y_apple ≥ 5

def at_most_five_oranges (r_orange y_orange : ℕ) : Prop :=
  r_orange + y_orange ≤ 5

def kiwi_grape_constraints (g_kiwi p_grape : ℕ) : Prop :=
  g_kiwi + p_grape ≥ 8 ∧ g_kiwi + p_grape ≤ 12 ∧ g_kiwi = p_grape

def at_least_one_each_grape (g_grape p_grape : ℕ) : Prop :=
  g_grape ≥ 1 ∧ p_grape ≥ 1

-- The final statement to prove
theorem fruit_platter_has_thirty_fruits :
  ∃ (g_apple r_apple y_apple r_orange y_orange g_kiwi p_grape g_grape : ℕ),
    at_least_five_apples g_apple r_apple y_apple ∧
    at_most_five_oranges r_orange y_orange ∧
    kiwi_grape_constraints g_kiwi p_grape ∧
    at_least_one_each_grape g_grape p_grape ∧
    g_apple + r_apple + y_apple + r_orange + y_orange + g_kiwi + p_grape + g_grape = 30 :=
sorry

end NUMINAMATH_GPT_fruit_platter_has_thirty_fruits_l199_19946


namespace NUMINAMATH_GPT_factor_expression_l199_19953

theorem factor_expression (x : ℝ) : 3 * x * (x - 5) + 7 * (x - 5) - 2 * (x - 5) = (3 * x + 5) * (x - 5) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l199_19953


namespace NUMINAMATH_GPT_smallest_n_for_good_sequence_l199_19988

def is_good_sequence (a : ℕ → ℝ) : Prop :=
   (∃ (a_0 : ℕ), a 0 = a_0) ∧
   (∀ i : ℕ, a (i+1) = 2 * a i + 1 ∨ a (i+1) = a i / (a i + 2)) ∧
   (∃ k : ℕ, a k = 2014)

theorem smallest_n_for_good_sequence : 
  ∀ (a : ℕ → ℝ), is_good_sequence a → ∃ n : ℕ, a n = 2014 ∧ ∀ m : ℕ, m < n → a m ≠ 2014 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_good_sequence_l199_19988


namespace NUMINAMATH_GPT_prob_correct_l199_19923

-- Define the individual probabilities.
def prob_first_ring := 1 / 10
def prob_second_ring := 3 / 10
def prob_third_ring := 2 / 5
def prob_fourth_ring := 1 / 10

-- Define the total probability of answering within the first four rings.
def prob_answer_within_four_rings := 
  prob_first_ring + prob_second_ring + prob_third_ring + prob_fourth_ring

-- State the theorem.
theorem prob_correct : prob_answer_within_four_rings = 9 / 10 :=
by
  -- We insert a placeholder for the proof.
  sorry

end NUMINAMATH_GPT_prob_correct_l199_19923


namespace NUMINAMATH_GPT_remainder_7_pow_150_mod_12_l199_19993

theorem remainder_7_pow_150_mod_12 :
  (7^150) % 12 = 1 := sorry

end NUMINAMATH_GPT_remainder_7_pow_150_mod_12_l199_19993


namespace NUMINAMATH_GPT_Xiaokang_position_l199_19911

theorem Xiaokang_position :
  let east := 150
  let west := 100
  let total_walks := 3
  (east - west - west = -50) :=
sorry

end NUMINAMATH_GPT_Xiaokang_position_l199_19911


namespace NUMINAMATH_GPT_second_race_length_l199_19924

variable (T L : ℝ)
variable (V_A V_B V_C : ℝ)

variables (h1 : V_A * T = 100)
variables (h2 : V_B * T = 90)
variables (h3 : V_C * T = 87)
variables (h4 : L / V_B = (L - 6) / V_C)

theorem second_race_length :
  L = 180 :=
sorry

end NUMINAMATH_GPT_second_race_length_l199_19924


namespace NUMINAMATH_GPT_least_whole_number_clock_equivalent_l199_19919

theorem least_whole_number_clock_equivalent :
  ∃ h : ℕ, h > 6 ∧ h ^ 2 % 24 = h % 24 ∧ ∀ k : ℕ, k > 6 ∧ k ^ 2 % 24 = k % 24 → h ≤ k := sorry

end NUMINAMATH_GPT_least_whole_number_clock_equivalent_l199_19919


namespace NUMINAMATH_GPT_range_of_m_l199_19997

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ (m > 2 ∨ m < -4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l199_19997


namespace NUMINAMATH_GPT_div_pow_two_sub_one_l199_19942

theorem div_pow_two_sub_one {k n : ℕ} (hk : 0 < k) (hn : 0 < n) :
  (3^k ∣ 2^n - 1) ↔ (∃ m : ℕ, n = 2 * 3^(k-1) * m) :=
by
  sorry

end NUMINAMATH_GPT_div_pow_two_sub_one_l199_19942


namespace NUMINAMATH_GPT_right_triangle_perimeter_l199_19999

theorem right_triangle_perimeter 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_area : 1/2 * 30 * b = 180)
  (h_pythagorean : c^2 = 30^2 + b^2)
  : a + b + c = 42 + 2 * Real.sqrt 261 :=
sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l199_19999


namespace NUMINAMATH_GPT_krish_spent_on_sweets_l199_19928

noncomputable def initial_amount := 200.50
noncomputable def amount_per_friend := 25.20
noncomputable def remaining_amount := 114.85

noncomputable def total_given_to_friends := amount_per_friend * 2
noncomputable def amount_before_sweets := initial_amount - total_given_to_friends
noncomputable def amount_spent_on_sweets := amount_before_sweets - remaining_amount

theorem krish_spent_on_sweets : amount_spent_on_sweets = 35.25 :=
by
  sorry

end NUMINAMATH_GPT_krish_spent_on_sweets_l199_19928


namespace NUMINAMATH_GPT_ratio_of_adults_to_children_is_24_over_25_l199_19937

theorem ratio_of_adults_to_children_is_24_over_25
  (a c : ℕ) (h₁ : a ≥ 1) (h₂ : c ≥ 1) 
  (h₃ : 30 * a + 18 * c = 2340) 
  (h₄ : c % 5 = 0) :
  a = 48 ∧ c = 50 ∧ (a / c : ℚ) = 24 / 25 :=
sorry

end NUMINAMATH_GPT_ratio_of_adults_to_children_is_24_over_25_l199_19937


namespace NUMINAMATH_GPT_tennis_players_l199_19975

theorem tennis_players (total_members badminton_players neither_players both_players : ℕ)
  (h1 : total_members = 80)
  (h2 : badminton_players = 48)
  (h3 : neither_players = 7)
  (h4 : both_players = 21) :
  total_members - neither_players = badminton_players - both_players + (total_members - neither_players - badminton_players + both_players) + both_players →
  ((total_members - neither_players) - (badminton_players - both_players) - both_players) + both_players = 46 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_tennis_players_l199_19975


namespace NUMINAMATH_GPT_sum_of_drawn_vegetable_oil_and_fruits_vegetables_l199_19981

-- Definitions based on conditions
def varieties_of_grains : ℕ := 40
def varieties_of_vegetable_oil : ℕ := 10
def varieties_of_animal_products : ℕ := 30
def varieties_of_fruits_vegetables : ℕ := 20
def total_sample_size : ℕ := 20

def sampling_fraction : ℚ := total_sample_size / (varieties_of_grains + varieties_of_vegetable_oil + varieties_of_animal_products + varieties_of_fruits_vegetables)

def expected_drawn_vegetable_oil : ℚ := varieties_of_vegetable_oil * sampling_fraction
def expected_drawn_fruits_vegetables : ℚ := varieties_of_fruits_vegetables * sampling_fraction

-- The theorem to be proved
theorem sum_of_drawn_vegetable_oil_and_fruits_vegetables : 
  expected_drawn_vegetable_oil + expected_drawn_fruits_vegetables = 6 := 
by 
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_sum_of_drawn_vegetable_oil_and_fruits_vegetables_l199_19981


namespace NUMINAMATH_GPT_solve_for_x_l199_19902

theorem solve_for_x (x : ℝ) (h : x ≠ -2) :
  (4 * x) / (x + 2) - 2 / (x + 2) = 3 / (x + 2) → x = 5 / 4 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l199_19902


namespace NUMINAMATH_GPT_avg_primes_between_30_and_50_l199_19964

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ := [31, 37, 41, 43, 47]

def sum_primes : ℕ := primes_between_30_and_50.sum

def count_primes : ℕ := primes_between_30_and_50.length

def average_primes : ℚ := (sum_primes : ℚ) / (count_primes : ℚ)

theorem avg_primes_between_30_and_50 : average_primes = 39.8 := by
  sorry

end NUMINAMATH_GPT_avg_primes_between_30_and_50_l199_19964


namespace NUMINAMATH_GPT_radius_intersection_xy_plane_l199_19910

noncomputable def center_sphere : ℝ × ℝ × ℝ := (3, 3, 3)

def radius_xz_circle : ℝ := 2

def xz_center : ℝ × ℝ × ℝ := (3, 0, 3)

def xy_center : ℝ × ℝ × ℝ := (3, 3, 0)

theorem radius_intersection_xy_plane (r : ℝ) (s : ℝ) 
(h_center : center_sphere = (3, 3, 3)) 
(h_xz : xz_center = (3, 0, 3))
(h_r_xz : radius_xz_circle = 2)
(h_xy : xy_center = (3, 3, 0)):
s = 3 := 
sorry

end NUMINAMATH_GPT_radius_intersection_xy_plane_l199_19910


namespace NUMINAMATH_GPT_roots_in_interval_l199_19932

theorem roots_in_interval (f : ℝ → ℝ)
  (h : ∀ x, f x = 4 * x ^ 2 - (3 * m + 1) * x - m - 2) :
  (forall (x1 x2 : ℝ), (f x1 = 0 ∧ f x2 = 0) → -1 < x1 ∧ x1 < 2 ∧ -1 < x2 ∧ x2 < 2) ↔ -1 < m ∧ m < 12 / 7 :=
sorry

end NUMINAMATH_GPT_roots_in_interval_l199_19932


namespace NUMINAMATH_GPT_sculpt_cost_in_mxn_l199_19913

variable (usd_to_nad usd_to_mxn cost_nad cost_mxn : ℝ)

theorem sculpt_cost_in_mxn (h1 : usd_to_nad = 8) (h2 : usd_to_mxn = 20) (h3 : cost_nad = 160) : cost_mxn = 400 :=
by
  sorry

end NUMINAMATH_GPT_sculpt_cost_in_mxn_l199_19913


namespace NUMINAMATH_GPT_total_lunch_bill_l199_19935

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h_hd : cost_hotdog = 5.36) (h_sd : cost_salad = 5.10) :
  cost_hotdog + cost_salad = 10.46 :=
by
  sorry

end NUMINAMATH_GPT_total_lunch_bill_l199_19935


namespace NUMINAMATH_GPT_football_team_birthday_collision_moscow_birthday_collision_l199_19907

theorem football_team_birthday_collision (n : ℕ) (k : ℕ) (h1 : n ≥ 11) (h2 : k = 7) : 
  ∃ (d : ℕ) (p1 p2 : ℕ), p1 ≠ p2 ∧ p1 ≤ n ∧ p2 ≤ n ∧ d ≤ k :=
by sorry

theorem moscow_birthday_collision (population : ℕ) (days : ℕ) (h1 : population > 10000000) (h2 : days = 366) :
  ∃ (day : ℕ) (count : ℕ), count ≥ 10000 ∧ count ≤ population / days :=
by sorry

end NUMINAMATH_GPT_football_team_birthday_collision_moscow_birthday_collision_l199_19907
