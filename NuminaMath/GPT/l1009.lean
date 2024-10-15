import Mathlib

namespace NUMINAMATH_GPT_min_value_a_b_l1009_100980

theorem min_value_a_b (x y a b : ℝ) (h1 : 2 * x - y + 2 ≥ 0) (h2 : 8 * x - y - 4 ≤ 0) 
  (h3 : x ≥ 0) (h4 : y ≥ 0) (h5 : a > 0) (h6 : b > 0) (h7 : a * x + y = 8) : 
  a + b ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_a_b_l1009_100980


namespace NUMINAMATH_GPT_avg_speed_l1009_100944

variable (d1 d2 t1 t2 : ℕ)

-- Conditions
def distance_first_hour : ℕ := 80
def distance_second_hour : ℕ := 40
def time_first_hour : ℕ := 1
def time_second_hour : ℕ := 1

-- Ensure that total distance and total time are defined correctly from conditions
def total_distance : ℕ := distance_first_hour + distance_second_hour
def total_time : ℕ := time_first_hour + time_second_hour

-- Theorem to prove the average speed
theorem avg_speed : total_distance / total_time = 60 := by
  sorry

end NUMINAMATH_GPT_avg_speed_l1009_100944


namespace NUMINAMATH_GPT_combined_loss_percentage_l1009_100910

theorem combined_loss_percentage
  (cost_price_radio : ℕ := 8000)
  (quantity_radio : ℕ := 5)
  (discount_radio : ℚ := 0.1)
  (tax_radio : ℚ := 0.06)
  (sale_price_radio : ℕ := 7200)
  (cost_price_tv : ℕ := 20000)
  (quantity_tv : ℕ := 3)
  (discount_tv : ℚ := 0.15)
  (tax_tv : ℚ := 0.07)
  (sale_price_tv : ℕ := 18000)
  (cost_price_phone : ℕ := 15000)
  (quantity_phone : ℕ := 4)
  (discount_phone : ℚ := 0.08)
  (tax_phone : ℚ := 0.05)
  (sale_price_phone : ℕ := 14500) :
  let total_cost_price := (quantity_radio * cost_price_radio) + (quantity_tv * cost_price_tv) + (quantity_phone * cost_price_phone)
  let total_sale_price := (quantity_radio * sale_price_radio) + (quantity_tv * sale_price_tv) + (quantity_phone * sale_price_phone)
  let total_loss := total_cost_price - total_sale_price
  let loss_percentage := (total_loss * 100 : ℚ) / total_cost_price
  loss_percentage = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_combined_loss_percentage_l1009_100910


namespace NUMINAMATH_GPT_ratio_of_ages_in_two_years_l1009_100979

-- Define the constants
def son_age : ℕ := 24
def age_difference : ℕ := 26

-- Define the equations based on conditions
def man_age := son_age + age_difference
def son_future_age := son_age + 2
def man_future_age := man_age + 2

-- State the theorem for the required ratio
theorem ratio_of_ages_in_two_years : man_future_age / son_future_age = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_in_two_years_l1009_100979


namespace NUMINAMATH_GPT_remainder_of_12345678910_div_101_l1009_100956

theorem remainder_of_12345678910_div_101 :
  12345678910 % 101 = 31 :=
sorry

end NUMINAMATH_GPT_remainder_of_12345678910_div_101_l1009_100956


namespace NUMINAMATH_GPT_find_number_l1009_100961

-- Given conditions:
def sum_and_square (n : ℕ) : Prop := n^2 + n = 252
def is_factor (n d : ℕ) : Prop := d % n = 0

-- Equivalent proof problem statement
theorem find_number : ∃ n : ℕ, sum_and_square n ∧ is_factor n 180 ∧ n > 0 ∧ n = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1009_100961


namespace NUMINAMATH_GPT_ceil_sqrt_fraction_eq_neg2_l1009_100929

theorem ceil_sqrt_fraction_eq_neg2 :
  (Int.ceil (-Real.sqrt (36 / 9))) = -2 :=
by
  sorry

end NUMINAMATH_GPT_ceil_sqrt_fraction_eq_neg2_l1009_100929


namespace NUMINAMATH_GPT_prob_8th_roll_last_l1009_100992

-- Define the conditions as functions or constants
def prob_diff_rolls : ℚ := 5/6
def prob_same_roll : ℚ := 1/6

-- Define the theorem stating the probability of the 8th roll being the last roll
theorem prob_8th_roll_last : (1 : ℚ) * prob_diff_rolls^6 * prob_same_roll = 15625 / 279936 := 
sorry

end NUMINAMATH_GPT_prob_8th_roll_last_l1009_100992


namespace NUMINAMATH_GPT_correct_inequality_l1009_100943

theorem correct_inequality (x : ℝ) : (1 / (x^2 + 1)) > (1 / (x^2 + 2)) :=
by {
  -- Lean proof steps would be here, but we will use 'sorry' instead to indicate the proof is omitted.
  sorry
}

end NUMINAMATH_GPT_correct_inequality_l1009_100943


namespace NUMINAMATH_GPT_number_of_freshmen_l1009_100947

theorem number_of_freshmen (n : ℕ) : n < 450 ∧ n % 19 = 18 ∧ n % 17 = 10 → n = 265 := by
  sorry

end NUMINAMATH_GPT_number_of_freshmen_l1009_100947


namespace NUMINAMATH_GPT_arithmetic_sequence_a9_l1009_100942

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Assume arithmetic sequence: a(n) = a1 + (n-1)d
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (n : ℕ) : ℤ := a 1 + (n - 1) * d

-- Given conditions
axiom condition1 : arithmetic_sequence a d 5 + arithmetic_sequence a d 7 = 16
axiom condition2 : arithmetic_sequence a d 3 = 1

-- Prove that a₉ = 15
theorem arithmetic_sequence_a9 : arithmetic_sequence a d 9 = 15 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a9_l1009_100942


namespace NUMINAMATH_GPT_function_y_increases_when_x_gt_1_l1009_100964

theorem function_y_increases_when_x_gt_1 :
  ∀ (x : ℝ), (x > 1 → 2*x^2 > 2*(x-1)^2) :=
by
  sorry

end NUMINAMATH_GPT_function_y_increases_when_x_gt_1_l1009_100964


namespace NUMINAMATH_GPT_compare_sqrt_l1009_100900

theorem compare_sqrt : 3 * Real.sqrt 2 > Real.sqrt 17 := by
  sorry

end NUMINAMATH_GPT_compare_sqrt_l1009_100900


namespace NUMINAMATH_GPT_brick_fence_depth_l1009_100994

theorem brick_fence_depth (length height total_bricks : ℕ) 
    (h1 : length = 20) 
    (h2 : height = 5) 
    (h3 : total_bricks = 800) : 
    (total_bricks / (4 * length * height) = 2) := 
by
  sorry

end NUMINAMATH_GPT_brick_fence_depth_l1009_100994


namespace NUMINAMATH_GPT_joan_picked_apples_l1009_100914

theorem joan_picked_apples (a b c : ℕ) (h1 : b = 27) (h2 : c = 70) (h3 : c = a + b) : a = 43 :=
by
  sorry

end NUMINAMATH_GPT_joan_picked_apples_l1009_100914


namespace NUMINAMATH_GPT_problem1_problem2_l1009_100976

theorem problem1 : (-(3 / 4) - (5 / 8) + (9 / 12)) * (-24) = 15 := by
  sorry

theorem problem2 : (-1 ^ 6 + |(-2) ^ 3 - 10| - (-3) / (-1) ^ 2023) = 14 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1009_100976


namespace NUMINAMATH_GPT_minimum_value_of_f_l1009_100987

-- Define the function y = f(x)
def f (x : ℝ) : ℝ := x^2 + 8 * x + 25

-- We need to prove that the minimum value of f(x) is 9
theorem minimum_value_of_f : ∃ x : ℝ, f x = 9 ∧ ∀ y : ℝ, f y ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1009_100987


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l1009_100972

theorem arithmetic_expression_evaluation :
  (1 / 6 * -6 / (-1 / 6) * 6) = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l1009_100972


namespace NUMINAMATH_GPT_evaluate_expression_l1009_100923

theorem evaluate_expression : 8^3 + 3 * 8^2 + 3 * 8 + 1 = 729 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1009_100923


namespace NUMINAMATH_GPT_probability_two_people_between_l1009_100955

theorem probability_two_people_between (total_people : ℕ) (favorable_arrangements : ℕ) (total_arrangements : ℕ) :
  total_people = 6 ∧ favorable_arrangements = 144 ∧ total_arrangements = 720 →
  (favorable_arrangements / total_arrangements : ℚ) = 1 / 5 :=
by
  intros h
  -- We substitute the given conditions
  have ht : total_people = 6 := h.1
  have hf : favorable_arrangements = 144 := h.2.1
  have ha : total_arrangements = 720 := h.2.2
  -- We need to calculate the probability considering the favorable and total arrangements
  sorry

end NUMINAMATH_GPT_probability_two_people_between_l1009_100955


namespace NUMINAMATH_GPT_train_speed_is_60_kmph_l1009_100974

-- Define the conditions
def time_to_cross_pole_seconds : ℚ := 36
def length_of_train_meters : ℚ := 600

-- Define the conversion factors
def seconds_per_hour : ℚ := 3600
def meters_per_kilometer : ℚ := 1000

-- Convert the conditions to appropriate units
def time_to_cross_pole_hours : ℚ := time_to_cross_pole_seconds / seconds_per_hour
def length_of_train_kilometers : ℚ := length_of_train_meters / meters_per_kilometer

-- Prove that the speed of the train in km/hr is 60
theorem train_speed_is_60_kmph : 
  (length_of_train_kilometers / time_to_cross_pole_hours) = 60 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_is_60_kmph_l1009_100974


namespace NUMINAMATH_GPT_exists_n_divisible_by_5_l1009_100928

theorem exists_n_divisible_by_5 
  (a b c d m : ℤ) 
  (h_div : a * m ^ 3 + b * m ^ 2 + c * m + d ≡ 0 [ZMOD 5]) 
  (h_d_nonzero : d ≠ 0) : 
  ∃ n : ℤ, d * n ^ 3 + c * n ^ 2 + b * n + a ≡ 0 [ZMOD 5] :=
sorry

end NUMINAMATH_GPT_exists_n_divisible_by_5_l1009_100928


namespace NUMINAMATH_GPT_tallest_stack_is_b_l1009_100925

def number_of_pieces_a : ℕ := 8
def number_of_pieces_b : ℕ := 11
def number_of_pieces_c : ℕ := 6

def height_per_piece_a : ℝ := 2
def height_per_piece_b : ℝ := 1.5
def height_per_piece_c : ℝ := 2.5

def total_height_a : ℝ := number_of_pieces_a * height_per_piece_a
def total_height_b : ℝ := number_of_pieces_b * height_per_piece_b
def total_height_c : ℝ := number_of_pieces_c * height_per_piece_c

theorem tallest_stack_is_b : (total_height_b = 16.5) ∧ (total_height_b > total_height_a) ∧ (total_height_b > total_height_c) := 
by
  sorry

end NUMINAMATH_GPT_tallest_stack_is_b_l1009_100925


namespace NUMINAMATH_GPT_rate_of_mixed_oil_l1009_100981

/--
If 10 litres of an oil at Rs. 50 per litre is mixed with 5 litres of another oil at Rs. 68 per litre, 
8 litres of a third oil at Rs. 42 per litre, and 7 litres of a fourth oil at Rs. 62 per litre, 
then the rate of the mixed oil per litre is Rs. 53.67.
-/
theorem rate_of_mixed_oil :
  let cost1 := 10 * 50
  let cost2 := 5 * 68
  let cost3 := 8 * 42
  let cost4 := 7 * 62
  let total_cost := cost1 + cost2 + cost3 + cost4
  let total_volume := 10 + 5 + 8 + 7
  let rate_per_litre := total_cost / total_volume
  rate_per_litre = 53.67 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rate_of_mixed_oil_l1009_100981


namespace NUMINAMATH_GPT_sqrt_25_eq_pm_five_l1009_100915

theorem sqrt_25_eq_pm_five (x : ℝ) : x^2 = 25 ↔ x = 5 ∨ x = -5 := 
sorry

end NUMINAMATH_GPT_sqrt_25_eq_pm_five_l1009_100915


namespace NUMINAMATH_GPT_certain_number_105_l1009_100958

theorem certain_number_105 (a x : ℕ) (h0 : a = 105) (h1 : a^3 = x * 25 * 45 * 49) : x = 21 := by
  sorry

end NUMINAMATH_GPT_certain_number_105_l1009_100958


namespace NUMINAMATH_GPT_sandy_books_cost_l1009_100982

theorem sandy_books_cost :
  ∀ (x : ℕ),
  (1280 + 880) / (x + 55) = 18 → 
  x = 65 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_sandy_books_cost_l1009_100982


namespace NUMINAMATH_GPT_oplus_calculation_l1009_100926

def my_oplus (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem oplus_calculation : my_oplus 2 3 = 23 := 
by
    sorry

end NUMINAMATH_GPT_oplus_calculation_l1009_100926


namespace NUMINAMATH_GPT_roots_of_poly_l1009_100965

theorem roots_of_poly (a b c : ℂ) :
  ∀ x, x = a ∨ x = b ∨ x = c → x^4 - a*x^3 - b*x + c = 0 :=
sorry

end NUMINAMATH_GPT_roots_of_poly_l1009_100965


namespace NUMINAMATH_GPT_find_ad_l1009_100911

-- Defining the two-digit and three-digit numbers
def two_digit (a b : ℕ) : ℕ := 10 * a + b
def three_digit (a b : ℕ) : ℕ := 100 + two_digit a b

def two_digit' (c d : ℕ) : ℕ := 10 * c + d
def three_digit' (c d : ℕ) : ℕ := 100 * c + 10 * d + 1

-- The main problem
theorem find_ad (a b c d : ℕ) (h1 : three_digit a b = three_digit' c d + 15) (h2 : two_digit a b = two_digit' c d + 24) :
    two_digit a d = 32 := by
  sorry

end NUMINAMATH_GPT_find_ad_l1009_100911


namespace NUMINAMATH_GPT_no_two_adj_or_opposite_same_num_l1009_100949

theorem no_two_adj_or_opposite_same_num :
  ∃ (prob : ℚ), prob = 25 / 648 ∧ 
  ∀ (A B C D E F : ℕ), 
    (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A) ∧
    (A ≠ D ∧ B ≠ E ∧ C ≠ F) ∧ 
    (1 ≤ A ∧ A ≤ 6) ∧ (1 ≤ B ∧ B ≤ 6) ∧ (1 ≤ C ∧ C ≤ 6) ∧ 
    (1 ≤ D ∧ D ≤ 6) ∧ (1 ≤ E ∧ E ≤ 6) ∧ (1 ≤ F ∧ F ≤ 6) →
    prob = (6 * 5 * 4 * 5 * 3 * 3) / (6^6) := 
sorry

end NUMINAMATH_GPT_no_two_adj_or_opposite_same_num_l1009_100949


namespace NUMINAMATH_GPT_cos_7pi_over_6_eq_neg_sqrt3_over_2_l1009_100902

theorem cos_7pi_over_6_eq_neg_sqrt3_over_2 : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_GPT_cos_7pi_over_6_eq_neg_sqrt3_over_2_l1009_100902


namespace NUMINAMATH_GPT_problem_expression_eq_zero_l1009_100990

variable {x y : ℝ}

theorem problem_expression_eq_zero (h : x * y ≠ 0) : 
    ( ( (x^2 - 1) / x ) * ( (y^2 - 1) / y ) ) - 
    ( ( (x^2 - 1) / y ) * ( (y^2 - 1) / x ) ) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_expression_eq_zero_l1009_100990


namespace NUMINAMATH_GPT_provisions_last_for_more_days_l1009_100903

def initial_men : ℕ := 2000
def initial_days : ℕ := 65
def additional_men : ℕ := 3000
def days_used : ℕ := 15
def remaining_provisions :=
  initial_men * initial_days - initial_men * days_used
def total_men_after_reinforcement := initial_men + additional_men
def remaining_days := remaining_provisions / total_men_after_reinforcement

theorem provisions_last_for_more_days :
  remaining_days = 20 := by
  sorry

end NUMINAMATH_GPT_provisions_last_for_more_days_l1009_100903


namespace NUMINAMATH_GPT_oliver_spent_amount_l1009_100971

theorem oliver_spent_amount :
  ∀ (S : ℕ), (33 - S + 32 = 61) → S = 4 :=
by
  sorry

end NUMINAMATH_GPT_oliver_spent_amount_l1009_100971


namespace NUMINAMATH_GPT_math_olympiad_scores_l1009_100916

theorem math_olympiad_scores (a : Fin 20 → ℕ) 
  (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → j ≠ k → i ≠ k → a i < a j + a k) :
  ∀ i : Fin 20, a i > 18 := 
sorry

end NUMINAMATH_GPT_math_olympiad_scores_l1009_100916


namespace NUMINAMATH_GPT_trajectory_of_midpoint_l1009_100969

theorem trajectory_of_midpoint (A B P : ℝ × ℝ)
  (hA : A = (2, 4))
  (hB : ∃ m n : ℝ, B = (m, n) ∧ n^2 = 2 * m)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (P.2 - 2)^2 = P.1 - 1 :=
sorry

end NUMINAMATH_GPT_trajectory_of_midpoint_l1009_100969


namespace NUMINAMATH_GPT_square_of_1024_l1009_100912

theorem square_of_1024 : 1024^2 = 1048576 :=
by
  sorry

end NUMINAMATH_GPT_square_of_1024_l1009_100912


namespace NUMINAMATH_GPT_setC_not_pythagorean_l1009_100970

/-- Defining sets of numbers as options -/
def SetA := (3, 4, 5)
def SetB := (5, 12, 13)
def SetC := (7, 25, 26)
def SetD := (6, 8, 10)

/-- Function to check if a set is a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

/-- Theorem stating set C is not a Pythagorean triple -/
theorem setC_not_pythagorean :
  ¬isPythagoreanTriple 7 25 26 :=
by {
  -- This slot will be filled with the concrete proof steps in Lean.
  sorry
}

end NUMINAMATH_GPT_setC_not_pythagorean_l1009_100970


namespace NUMINAMATH_GPT_dakotas_medical_bill_l1009_100952

theorem dakotas_medical_bill :
  let days_in_hospital := 3
  let hospital_bed_cost_per_day := 900
  let specialists_rate_per_hour := 250
  let specialist_minutes_per_day := 15
  let num_specialists := 2
  let ambulance_cost := 1800

  let hospital_bed_cost := hospital_bed_cost_per_day * days_in_hospital
  let specialists_total_minutes := specialist_minutes_per_day * num_specialists
  let specialists_hours := specialists_total_minutes / 60.0
  let specialists_cost := specialists_hours * specialists_rate_per_hour

  let total_medical_bill := hospital_bed_cost + specialists_cost + ambulance_cost

  total_medical_bill = 4625 := 
by
  sorry

end NUMINAMATH_GPT_dakotas_medical_bill_l1009_100952


namespace NUMINAMATH_GPT_calc_remainder_l1009_100959

theorem calc_remainder : 
  (1 - 90 * Nat.choose 10 1 + 90^2 * Nat.choose 10 2 - 90^3 * Nat.choose 10 3 +
   90^4 * Nat.choose 10 4 - 90^5 * Nat.choose 10 5 + 90^6 * Nat.choose 10 6 -
   90^7 * Nat.choose 10 7 + 90^8 * Nat.choose 10 8 - 90^9 * Nat.choose 10 9 +
   90^10 * Nat.choose 10 10) % 88 = 1 := 
by sorry

end NUMINAMATH_GPT_calc_remainder_l1009_100959


namespace NUMINAMATH_GPT_inequality_proof_l1009_100968

theorem inequality_proof
  (a b c d e f : ℝ)
  (h1 : 1 ≤ a)
  (h2 : a ≤ b)
  (h3 : b ≤ c)
  (h4 : c ≤ d)
  (h5 : d ≤ e)
  (h6 : e ≤ f) :
  (a * f + b * e + c * d) * (a * f + b * d + c * e) ≤ (a + b^2 + c^3) * (d + e^2 + f^3) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1009_100968


namespace NUMINAMATH_GPT_simplify_fraction_l1009_100932

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (hx2 : x^2 - (1 / y) ≠ 0) (hy2 : y^2 - (1 / x) ≠ 0) :
  (x^2 - 1 / y) / (y^2 - 1 / x) = (x * (x^2 * y - 1)) / (y * (y^2 * x - 1)) :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l1009_100932


namespace NUMINAMATH_GPT_specified_time_is_30_total_constuction_cost_is_180000_l1009_100963

noncomputable def specified_time (x : ℕ) :=
  let teamA_rate := 1 / (x:ℝ)
  let teamB_rate := 2 / (3 * (x:ℝ))
  (teamA_rate + teamB_rate) * 15 + 5 * teamA_rate = 1

theorem specified_time_is_30 : specified_time 30 :=
  by 
    sorry

noncomputable def total_constuction_cost (x : ℕ) (costA : ℕ) (costB : ℕ) :=
  let teamA_rate := 1 / (x:ℝ)
  let teamB_rate := 2 / (3 * (x:ℝ))
  let total_time := 1 / (teamA_rate + teamB_rate)
  total_time * (costA + costB)

theorem total_constuction_cost_is_180000 : total_constuction_cost 30 6500 3500 = 180000 :=
  by 
    sorry

end NUMINAMATH_GPT_specified_time_is_30_total_constuction_cost_is_180000_l1009_100963


namespace NUMINAMATH_GPT_bobArrivesBefore845Prob_l1009_100941

noncomputable def probabilityBobBefore845 (totalTime: ℕ) (cutoffTime: ℕ) : ℚ :=
  let totalArea := (totalTime * totalTime) / 2
  let areaOfInterest := (cutoffTime * cutoffTime) / 2
  (areaOfInterest : ℚ) / totalArea

theorem bobArrivesBefore845Prob (totalTime: ℕ) (cutoffTime: ℕ) (ht: totalTime = 60) (hc: cutoffTime = 45) :
  probabilityBobBefore845 totalTime cutoffTime = 9 / 16 := by
  sorry

end NUMINAMATH_GPT_bobArrivesBefore845Prob_l1009_100941


namespace NUMINAMATH_GPT_initial_number_of_persons_l1009_100907

noncomputable def avg_weight_change : ℝ := 5.5
noncomputable def old_person_weight : ℝ := 68
noncomputable def new_person_weight : ℝ := 95.5
noncomputable def weight_diff : ℝ := new_person_weight - old_person_weight

theorem initial_number_of_persons (N : ℝ) 
  (h1 : avg_weight_change * N = weight_diff) : N = 5 :=
  by
  sorry

end NUMINAMATH_GPT_initial_number_of_persons_l1009_100907


namespace NUMINAMATH_GPT_complex_expression_l1009_100940

theorem complex_expression (z : ℂ) (i : ℂ) (h1 : z^2 + 1 = 0) (h2 : i^2 = -1) : 
  (z^4 + i) * (z^4 - i) = 0 :=
sorry

end NUMINAMATH_GPT_complex_expression_l1009_100940


namespace NUMINAMATH_GPT_divisible_by_bn_l1009_100927

variables {u v a b : ℤ} {n : ℕ}

theorem divisible_by_bn 
  (h1 : ∀ x : ℤ, x^2 + a*x + b = 0 → x = u ∨ x = v)
  (h2 : a^2 % b = 0) 
  (h3 : ∀ m : ℕ, m = 2 * n) : 
  ∀ n : ℕ, (u^m + v^m) % (b^n) = 0 := 
  sorry

end NUMINAMATH_GPT_divisible_by_bn_l1009_100927


namespace NUMINAMATH_GPT_rationalize_denominator_correct_l1009_100921

noncomputable def rationalize_denominator : Prop :=
  (1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2)

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_correct_l1009_100921


namespace NUMINAMATH_GPT_find_a_l1009_100986

noncomputable def f (x : ℝ) : ℝ := x^2 + 10

noncomputable def g (x : ℝ) : ℝ := x^2 - 6

theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : f (g a) = 12) :
    a = Real.sqrt (6 + Real.sqrt 2) ∨ a = Real.sqrt (6 - Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_find_a_l1009_100986


namespace NUMINAMATH_GPT_factor_polynomial_l1009_100908

theorem factor_polynomial :
  ∃ (a b c d e f : ℤ), a < d ∧
    (a * x^2 + b * x + c) * (d * x^2 + e * x + f) = x^2 - 6 * x + 9 - 64 * x^4 ∧
    (a = -8 ∧ b = 1 ∧ c = -3 ∧ d = 8 ∧ e = 1 ∧ f = -3) := by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1009_100908


namespace NUMINAMATH_GPT_number_of_pupils_wrong_entry_l1009_100985

theorem number_of_pupils_wrong_entry 
  (n : ℕ) (A : ℝ) 
  (h_wrong_entry : ∀ m, (m = 85 → n * (A + 1 / 2) = n * A + 52))
  (h_increase : ∀ m, (m = 33 → n * (A + 1 / 2) = n * A + 52)) 
  : n = 104 := 
sorry

end NUMINAMATH_GPT_number_of_pupils_wrong_entry_l1009_100985


namespace NUMINAMATH_GPT_calculate_value_l1009_100978

theorem calculate_value : 2 * (75 * 1313 - 25 * 1313) = 131300 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_value_l1009_100978


namespace NUMINAMATH_GPT_repeated_root_condition_l1009_100901

theorem repeated_root_condition (m : ℝ) : m = 10 → ∃ x, (5 * x) / (x - 2) + 1 = m / (x - 2) ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_repeated_root_condition_l1009_100901


namespace NUMINAMATH_GPT_simplify_fraction_l1009_100939

theorem simplify_fraction : 1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2 :=
by
sorry

end NUMINAMATH_GPT_simplify_fraction_l1009_100939


namespace NUMINAMATH_GPT_S_is_multiples_of_six_l1009_100975

-- Defining the problem.
def S : Set ℝ :=
  { t | ∃ n : ℤ, t = 6 * n }

-- We are given that S is non-empty
axiom S_non_empty : ∃ x, x ∈ S

-- Condition: For any x, y ∈ S, both x + y ∈ S and x - y ∈ S.
axiom S_closed_add_sub : ∀ x y, x ∈ S → y ∈ S → (x + y ∈ S ∧ x - y ∈ S)

-- The smallest positive number in S is 6.
axiom S_smallest : ∀ ε, ε > 0 → ∃ x, x ∈ S ∧ x = 6

-- The goal is to prove that S is exactly the set of all multiples of 6.
theorem S_is_multiples_of_six : ∀ t, t ∈ S ↔ ∃ n : ℤ, t = 6 * n :=
by
  sorry

end NUMINAMATH_GPT_S_is_multiples_of_six_l1009_100975


namespace NUMINAMATH_GPT_jim_total_weight_per_hour_l1009_100988

theorem jim_total_weight_per_hour :
  let hours := 8
  let gold_chest := 100
  let gold_bag := 50
  let gold_extra := 30 + 20 + 10
  let silver := 30
  let bronze := 50
  let weight_gold := 10
  let weight_silver := 5
  let weight_bronze := 2
  let total_gold := gold_chest + 2 * gold_bag + gold_extra
  let total_weight := total_gold * weight_gold + silver * weight_silver + bronze * weight_bronze
  total_weight / hours = 356.25 := by
  sorry

end NUMINAMATH_GPT_jim_total_weight_per_hour_l1009_100988


namespace NUMINAMATH_GPT_juan_faster_than_peter_l1009_100954

theorem juan_faster_than_peter (J : ℝ) :
  (Peter_speed : ℝ) = 5.0 →
  (time : ℝ) = 1.5 →
  (distance_apart : ℝ) = 19.5 →
  (J + 5.0) * time = distance_apart →
  J - 5.0 = 3 := 
by
  intros Peter_speed_eq time_eq distance_apart_eq relative_speed_eq
  sorry

end NUMINAMATH_GPT_juan_faster_than_peter_l1009_100954


namespace NUMINAMATH_GPT_find_principal_sum_l1009_100905

noncomputable def principal_sum (P R : ℝ) : ℝ := P * (R + 6) / 100 - P * R / 100

theorem find_principal_sum (P R : ℝ) (h : P * (R + 6) / 100 - P * R / 100 = 30) : P = 500 :=
by sorry

end NUMINAMATH_GPT_find_principal_sum_l1009_100905


namespace NUMINAMATH_GPT_tony_drive_time_l1009_100919

noncomputable def time_to_first_friend (d₁ d₂ t₂ : ℝ) : ℝ :=
  let v := d₂ / t₂
  d₁ / v

theorem tony_drive_time (d₁ d₂ t₂ : ℝ) (h_d₁ : d₁ = 120) (h_d₂ : d₂ = 200) (h_t₂ : t₂ = 5) : 
    time_to_first_friend d₁ d₂ t₂ = 3 := by
  rw [h_d₁, h_d₂, h_t₂]
  -- Further simplification would follow here based on the proof steps, which we are omitting
  sorry

end NUMINAMATH_GPT_tony_drive_time_l1009_100919


namespace NUMINAMATH_GPT_counterexample_to_conjecture_l1009_100989

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, 1 < m ∧ m < n → ¬(m ∣ n)

def is_power_of_two (k : ℕ) : Prop := ∃ m : ℕ, m > 0 ∧ k = 2 ^ m

theorem counterexample_to_conjecture :
  ∃ n : ℤ, n > 5 ∧ ¬ (3 ∣ n) ∧ ¬ (∃ p k : ℕ, is_prime p ∧ is_power_of_two k ∧ n = p + k) :=
sorry

end NUMINAMATH_GPT_counterexample_to_conjecture_l1009_100989


namespace NUMINAMATH_GPT_projectile_height_reach_l1009_100999

theorem projectile_height_reach (t : ℝ) (h : -16 * t^2 + 64 * t = 25) : t = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_projectile_height_reach_l1009_100999


namespace NUMINAMATH_GPT_greatest_missed_problems_l1009_100906

theorem greatest_missed_problems (total_problems : ℕ) (passing_percentage : ℝ) (missed_problems : ℕ) : 
  total_problems = 50 ∧ passing_percentage = 0.85 → missed_problems = 7 :=
by
  sorry

end NUMINAMATH_GPT_greatest_missed_problems_l1009_100906


namespace NUMINAMATH_GPT_exists_prime_divisor_in_sequence_l1009_100938

theorem exists_prime_divisor_in_sequence
  (c d : ℕ) (hc : 2 ≤ c) (hd : 2 ≤ d)
  (a : ℕ → ℕ)
  (h0 : a 1 = c)
  (hs : ∀ n, a (n+1) = a n ^ d + c) :
  ∀ (n : ℕ), 2 ≤ n →
  ∃ (p : ℕ), Prime p ∧ p ∣ a n ∧ ∀ i, 1 ≤ i ∧ i < n → ¬ p ∣ a i := sorry

end NUMINAMATH_GPT_exists_prime_divisor_in_sequence_l1009_100938


namespace NUMINAMATH_GPT_transformed_parabola_l1009_100936

theorem transformed_parabola (x : ℝ) : 
  (λ x => -x^2 + 1) (x - 2) - 2 = - (x - 2)^2 - 1 := 
by 
  sorry 

end NUMINAMATH_GPT_transformed_parabola_l1009_100936


namespace NUMINAMATH_GPT_largest_even_number_l1009_100935

theorem largest_even_number (x : ℤ) 
  (h : x + (x + 2) + (x + 4) = x + 18) : x + 4 = 10 :=
by
  sorry

end NUMINAMATH_GPT_largest_even_number_l1009_100935


namespace NUMINAMATH_GPT_scrooge_mcduck_max_box_l1009_100983

-- Define Fibonacci numbers
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

-- The problem statement: for a given positive integer k (number of coins initially),
-- the maximum box index n into which Scrooge McDuck can place a coin
-- is F_{k+2} - 1.
theorem scrooge_mcduck_max_box (k : ℕ) (h_pos : 0 < k) :
  ∃ n, n = fib (k + 2) - 1 :=
sorry

end NUMINAMATH_GPT_scrooge_mcduck_max_box_l1009_100983


namespace NUMINAMATH_GPT_number_of_white_tshirts_in_one_pack_l1009_100924

namespace TShirts

variable (W : ℕ)

noncomputable def total_white_tshirts := 2 * W
noncomputable def total_blue_tshirts := 4 * 3
noncomputable def cost_per_tshirt := 3
noncomputable def total_cost := 66

theorem number_of_white_tshirts_in_one_pack :
  2 * W * cost_per_tshirt + total_blue_tshirts * cost_per_tshirt = total_cost → W = 5 :=
by
  sorry

end TShirts

end NUMINAMATH_GPT_number_of_white_tshirts_in_one_pack_l1009_100924


namespace NUMINAMATH_GPT_marks_in_mathematics_l1009_100997

-- Definitions for the given conditions in the problem
def marks_in_english : ℝ := 86
def marks_in_physics : ℝ := 82
def marks_in_chemistry : ℝ := 87
def marks_in_biology : ℝ := 81
def average_marks : ℝ := 85
def number_of_subjects : ℕ := 5

-- Defining the total marks based on the provided conditions
def total_marks : ℝ := average_marks * number_of_subjects

-- Proving that the marks in mathematics are 89
theorem marks_in_mathematics : total_marks - (marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology) = 89 :=
by
  sorry

end NUMINAMATH_GPT_marks_in_mathematics_l1009_100997


namespace NUMINAMATH_GPT_deployment_plans_l1009_100904

/-- Given 6 volunteers and needing to select 4 to fill different positions of 
  translator, tour guide, shopping guide, and cleaner, and knowing that neither 
  supporters A nor B can work as the translator, the total number of deployment plans is 240. -/
theorem deployment_plans (volunteers : Fin 6) (A B : Fin 6) : 
  ∀ {translator tour_guide shopping_guide cleaner : Fin 6},
  A ≠ translator ∧ B ≠ translator → 
  ∃ plans : Finset (Fin 6 × Fin 6 × Fin 6 × Fin 6), plans.card = 240 :=
by 
sorry

end NUMINAMATH_GPT_deployment_plans_l1009_100904


namespace NUMINAMATH_GPT_billy_picked_36_dandelions_initially_l1009_100967

namespace Dandelions

/-- The number of dandelions Billy picked initially. -/
def billy_initial (B : ℕ) : ℕ := B

/-- The number of dandelions George picked initially. -/
def george_initial (B : ℕ) : ℕ := B / 3

/-- The additional dandelions picked by Billy and George respectively. -/
def additional_dandelions : ℕ := 10

/-- The total dandelions picked by Billy and George initially and additionally. -/
def total_dandelions (B : ℕ) : ℕ :=
  billy_initial B + additional_dandelions + george_initial B + additional_dandelions

/-- The average number of dandelions picked by both Billy and George, given as 34. -/
def average_dandelions (total : ℕ) : Prop := total / 2 = 34

/-- The main theorem stating that Billy picked 36 dandelions initially. -/
theorem billy_picked_36_dandelions_initially :
  ∀ B : ℕ, average_dandelions (total_dandelions B) ↔ B = 36 :=
by
  intro B
  sorry

end Dandelions

end NUMINAMATH_GPT_billy_picked_36_dandelions_initially_l1009_100967


namespace NUMINAMATH_GPT_find_root_power_117_l1009_100993

noncomputable def problem (a b c : ℝ) (x1 x2 : ℝ) :=
  (3 * a - b) / c * x1^2 + c * (3 * a + b) / (3 * a - b) = 0 ∧
  (3 * a - b) / c * x2^2 + c * (3 * a + b) / (3 * a - b) = 0 ∧
  x1 + x2 = 0

theorem find_root_power_117 (a b c : ℝ) (x1 x2 : ℝ) (h : problem a b c x1 x2) : 
  x1 ^ 117 + x2 ^ 117 = 0 :=
sorry

end NUMINAMATH_GPT_find_root_power_117_l1009_100993


namespace NUMINAMATH_GPT_music_library_avg_disk_space_per_hour_l1009_100960

theorem music_library_avg_disk_space_per_hour 
  (days_of_music: ℕ) (total_space_MB: ℕ) (hours_in_day: ℕ) 
  (h1: days_of_music = 15) 
  (h2: total_space_MB = 18000) 
  (h3: hours_in_day = 24) : 
  (total_space_MB / (days_of_music * hours_in_day)) = 50 := 
by
  sorry

end NUMINAMATH_GPT_music_library_avg_disk_space_per_hour_l1009_100960


namespace NUMINAMATH_GPT_maximum_area_right_triangle_hypotenuse_8_l1009_100909

theorem maximum_area_right_triangle_hypotenuse_8 :
  ∃ a b : ℝ, (a^2 + b^2 = 64) ∧ (a * b) / 2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_maximum_area_right_triangle_hypotenuse_8_l1009_100909


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1009_100991

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ ∃ x : ℝ, |x| + x^2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1009_100991


namespace NUMINAMATH_GPT_total_expenditure_is_3000_l1009_100951

/-- Define the Hall dimensions -/
def length : ℝ := 20
def width : ℝ := 15
def cost_per_square_meter : ℝ := 10

/-- Statement to prove --/
theorem total_expenditure_is_3000 
  (h_length : length = 20)
  (h_width : width = 15)
  (h_cost : cost_per_square_meter = 10) : 
  length * width * cost_per_square_meter = 3000 :=
sorry

end NUMINAMATH_GPT_total_expenditure_is_3000_l1009_100951


namespace NUMINAMATH_GPT_germination_estimate_l1009_100957

theorem germination_estimate (germination_rate : ℝ) (total_pounds : ℝ) 
  (hrate_nonneg : 0 ≤ germination_rate) (hrate_le_one : germination_rate ≤ 1) 
  (h_germination_value : germination_rate = 0.971) 
  (h_total_pounds_value : total_pounds = 1000) : 
  total_pounds * (1 - germination_rate) = 29 := 
by 
  sorry

end NUMINAMATH_GPT_germination_estimate_l1009_100957


namespace NUMINAMATH_GPT_problem1_problem2_l1009_100977

theorem problem1 : 12 - (-18) + (-7) + (-15) = 8 :=
by sorry

theorem problem2 : (-1)^7 * 2 + (-3)^2 / 9 = -1 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1009_100977


namespace NUMINAMATH_GPT_Mark_time_spent_l1009_100931

theorem Mark_time_spent :
  let parking_time := 5
  let walking_time := 3
  let long_wait_time := 30
  let short_wait_time := 10
  let long_wait_days := 2
  let short_wait_days := 3
  let work_days := 5
  (parking_time + walking_time) * work_days + 
    long_wait_time * long_wait_days + 
    short_wait_time * short_wait_days = 130 :=
by
  sorry

end NUMINAMATH_GPT_Mark_time_spent_l1009_100931


namespace NUMINAMATH_GPT_abs_triangle_inequality_l1009_100995

theorem abs_triangle_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| :=
by sorry

end NUMINAMATH_GPT_abs_triangle_inequality_l1009_100995


namespace NUMINAMATH_GPT_roots_quadratic_l1009_100973

theorem roots_quadratic (a b : ℝ) (h₁ : a + b = 6) (h₂ : a * b = 8) :
  a^2 + a^5 * b^3 + a^3 * b^5 + b^2 = 10260 :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_l1009_100973


namespace NUMINAMATH_GPT_find_numbers_l1009_100962

theorem find_numbers (A B: ℕ) (h1: A + B = 581) (h2: (Nat.lcm A B) / (Nat.gcd A B) = 240) : 
  (A = 560 ∧ B = 21) ∨ (A = 21 ∧ B = 560) :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1009_100962


namespace NUMINAMATH_GPT_hexagonal_pyramid_volume_l1009_100917

theorem hexagonal_pyramid_volume (a : ℝ) (h : a > 0) (lateral_surface_area : ℝ) (base_area : ℝ)
  (H_base_area : base_area = (3 * Real.sqrt 3 / 2) * a^2)
  (H_lateral_surface_area : lateral_surface_area = 10 * base_area) :
  (1 / 3) * base_area * (a * Real.sqrt 3 / 2) * 3 * Real.sqrt 11 = (9 * a^3 * Real.sqrt 11) / 4 :=
by sorry

end NUMINAMATH_GPT_hexagonal_pyramid_volume_l1009_100917


namespace NUMINAMATH_GPT_advertising_department_employees_l1009_100913

theorem advertising_department_employees (N S A_s x : ℕ) (hN : N = 1000) (hS : S = 80) (hA_s : A_s = 4) 
(h_stratified : x / N = A_s / S) : x = 50 :=
sorry

end NUMINAMATH_GPT_advertising_department_employees_l1009_100913


namespace NUMINAMATH_GPT_cos_double_angle_l1009_100953

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + 3 * Real.pi / 2) = 1 / 3) : 
  Real.cos (2 * α) = -7 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1009_100953


namespace NUMINAMATH_GPT_chess_piece_problem_l1009_100984

theorem chess_piece_problem
  (a b c : ℕ)
  (h1 : b = b * 2 - a)
  (h2 : c = c * 2)
  (h3 : a = a * 2 - b)
  (h4 : c = c * 2 - a + b)
  (h5 : a * 2 = 16)
  (h6 : b * 2 = 16)
  (h7 : c * 2 = 16) : 
  a = 26 ∧ b = 14 ∧ c = 8 := 
sorry

end NUMINAMATH_GPT_chess_piece_problem_l1009_100984


namespace NUMINAMATH_GPT_enclosed_area_eq_32_over_3_l1009_100922

def line (x : ℝ) : ℝ := 2 * x + 3
def parabola (x : ℝ) : ℝ := x^2

theorem enclosed_area_eq_32_over_3 :
  ∫ x in (-(1:ℝ))..(3:ℝ), (line x - parabola x) = 32 / 3 :=
by
  sorry

end NUMINAMATH_GPT_enclosed_area_eq_32_over_3_l1009_100922


namespace NUMINAMATH_GPT_compare_logs_l1009_100937

theorem compare_logs (a b c : ℝ) (h1 : a = Real.log 6 / Real.log 3)
                              (h2 : b = Real.log 8 / Real.log 4)
                              (h3 : c = Real.log 10 / Real.log 5) : 
                              a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_compare_logs_l1009_100937


namespace NUMINAMATH_GPT_vitamin_D_scientific_notation_l1009_100945

def scientific_notation (x : ℝ) (m : ℝ) (n : ℤ) : Prop :=
  x = m * 10^n

theorem vitamin_D_scientific_notation :
  scientific_notation 0.0000046 4.6 (-6) :=
by {
  sorry
}

end NUMINAMATH_GPT_vitamin_D_scientific_notation_l1009_100945


namespace NUMINAMATH_GPT_solve_for_x_l1009_100933

theorem solve_for_x (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 :=
by
  -- reduce the problem to its final steps
  sorry

end NUMINAMATH_GPT_solve_for_x_l1009_100933


namespace NUMINAMATH_GPT_f_prime_neg_one_l1009_100918

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f x = f (-x)
axiom h2 : ∀ x : ℝ, f (x + 1) - f (1 - x) = 2 * x

theorem f_prime_neg_one : f' (-1) = -1 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_f_prime_neg_one_l1009_100918


namespace NUMINAMATH_GPT_initial_number_of_students_l1009_100998

/-- 
Theorem: If the average mark of the students of a class in an exam is 90, and 2 students whose average mark is 45 are excluded, resulting in the average mark of the remaining students being 95, then the initial number of students is 20.
-/
theorem initial_number_of_students (N : ℕ) (T : ℕ)
  (h1 : T = N * 90)
  (h2 : (T - 90) / (N - 2) = 95) : 
  N = 20 :=
sorry

end NUMINAMATH_GPT_initial_number_of_students_l1009_100998


namespace NUMINAMATH_GPT_determine_exponent_l1009_100920

-- Declare variables
variables {x y : ℝ}
variable {n : ℕ}

-- Use condition that the terms are like terms
theorem determine_exponent (h : - x ^ 2 * y ^ n = 3 * y * x ^ 2) : n = 1 :=
sorry

end NUMINAMATH_GPT_determine_exponent_l1009_100920


namespace NUMINAMATH_GPT_quadratic_neq_l1009_100946

theorem quadratic_neq (m : ℝ) : (m-2) ≠ 0 ↔ m ≠ 2 :=
sorry

end NUMINAMATH_GPT_quadratic_neq_l1009_100946


namespace NUMINAMATH_GPT_hyperbola_m_value_l1009_100996

theorem hyperbola_m_value (m k : ℝ) (h₀ : k > 0) (h₁ : 0 < -m) 
  (h₂ : 2 * k = Real.sqrt (1 + m)) : 
  m = -3 := 
by {
  sorry
}

end NUMINAMATH_GPT_hyperbola_m_value_l1009_100996


namespace NUMINAMATH_GPT_find_m_l1009_100948

theorem find_m (x y m : ℝ) 
  (h1 : x + y = 8)
  (h2 : y - m * x = 7)
  (h3 : y - x = 7.5) : m = 3 := 
  sorry

end NUMINAMATH_GPT_find_m_l1009_100948


namespace NUMINAMATH_GPT_gcf_360_180_l1009_100966

theorem gcf_360_180 : Nat.gcd 360 180 = 180 :=
by
  sorry

end NUMINAMATH_GPT_gcf_360_180_l1009_100966


namespace NUMINAMATH_GPT_sum_of_coefficients_l1009_100950

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (∀ x : ℝ, (x^3 - 1) * (x + 1)^7 = a_0 + a_1 * (x + 3) + 
           a_2 * (x + 3)^2 + a_3 * (x + 3)^3 + a_4 * (x + 3)^4 + 
           a_5 * (x + 3)^5 + a_6 * (x + 3)^6 + a_7 * (x + 3)^7 + 
           a_8 * (x + 3)^8 + a_9 * (x + 3)^9 + a_10 * (x + 3)^10) →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 9 := 
by
  -- proof steps skipped
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1009_100950


namespace NUMINAMATH_GPT_angle_difference_l1009_100934

-- Define the conditions
variables (A B : ℝ) 

def is_parallelogram := A + B = 180
def smaller_angle := A = 70
def larger_angle := B = 180 - 70

-- State the theorem to be proved
theorem angle_difference (A B : ℝ) (h1 : is_parallelogram A B) (h2 : smaller_angle A) : B - A = 40 := by
  sorry

end NUMINAMATH_GPT_angle_difference_l1009_100934


namespace NUMINAMATH_GPT_expected_value_of_boy_girl_pairs_l1009_100930

noncomputable def expected_value_of_T (boys girls : ℕ) : ℚ :=
  24 * ((boys / 24) * (girls / 23) + (girls / 24) * (boys / 23))

theorem expected_value_of_boy_girl_pairs (boys girls : ℕ) (h_boys : boys = 10) (h_girls : girls = 14) :
  expected_value_of_T boys girls = 12 :=
by
  rw [h_boys, h_girls]
  norm_num
  sorry

end NUMINAMATH_GPT_expected_value_of_boy_girl_pairs_l1009_100930
