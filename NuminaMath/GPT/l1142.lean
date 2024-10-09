import Mathlib

namespace sarah_score_l1142_114251

variable (s g : ℕ)  -- Sarah's and Greg's scores are natural numbers

theorem sarah_score
  (h1 : s = g + 50)  -- Sarah's score is 50 points more than Greg's
  (h2 : (s + g) / 2 = 110)  -- Average of their scores is 110
  : s = 135 :=  -- Prove Sarah's score is 135
by
  sorry

end sarah_score_l1142_114251


namespace derivative_at_one_l1142_114213

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + x^2) : f' 1 = -2 :=
by
  sorry

end derivative_at_one_l1142_114213


namespace total_time_l1142_114211

/-- Define the different time periods in years --/
def getting_in_shape : ℕ := 2
def learning_to_climb : ℕ := 2 * getting_in_shape
def months_climbing : ℕ := 7 * 5
def climbing : ℚ := months_climbing / 12
def break_after_climbing : ℚ := 13 / 12
def diving : ℕ := 2

/-- Prove that the total time taken to achieve all goals is 12 years --/
theorem total_time : getting_in_shape + learning_to_climb + climbing + break_after_climbing + diving = 12 := by
  sorry

end total_time_l1142_114211


namespace difference_of_two_numbers_l1142_114287

-- Definitions as per conditions
def L : ℕ := 1656
def S : ℕ := 273
def quotient : ℕ := 6
def remainder : ℕ := 15

-- Statement of the proof problem
theorem difference_of_two_numbers (h1 : L = 6 * S + 15) : L - S = 1383 :=
by sorry

end difference_of_two_numbers_l1142_114287


namespace distance_covered_l1142_114292

-- Define the conditions
def speed_still_water : ℕ := 30   -- 30 kmph
def current_speed : ℕ := 6        -- 6 kmph
def time_downstream : ℕ := 24     -- 24 seconds

-- Proving the distance covered downstream
theorem distance_covered (s_still s_current t : ℕ) (h_s_still : s_still = speed_still_water) (h_s_current : s_current = current_speed) (h_t : t = time_downstream):
  (s_still + s_current) * 1000 / 3600 * t = 240 :=
by sorry

end distance_covered_l1142_114292


namespace decorations_left_to_put_up_l1142_114243

variable (S B W P C T : Nat)
variable (h₁ : S = 12)
variable (h₂ : B = 4)
variable (h₃ : W = 12)
variable (h₄ : P = 2 * W)
variable (h₅ : C = 1)
variable (h₆ : T = 83)

theorem decorations_left_to_put_up (h₁ : S = 12) (h₂ : B = 4) (h₃ : W = 12) (h₄ : P = 2 * W) (h₅ : C = 1) (h₆ : T = 83) :
  T - (S + B + W + P + C) = 30 := sorry

end decorations_left_to_put_up_l1142_114243


namespace calculate_kevin_training_time_l1142_114216

theorem calculate_kevin_training_time : 
  ∀ (laps : ℕ) 
    (track_length : ℕ) 
    (run1_distance : ℕ) 
    (run1_speed : ℕ) 
    (walk_distance : ℕ) 
    (walk_speed : Real) 
    (run2_distance : ℕ) 
    (run2_speed : ℕ) 
    (minutes : ℕ) 
    (seconds : Real),
    laps = 8 →
    track_length = 500 →
    run1_distance = 200 →
    run1_speed = 3 →
    walk_distance = 100 →
    walk_speed = 1.5 →
    run2_distance = 200 →
    run2_speed = 4 →
    minutes = 24 →
    seconds = 27 →
    (∀ (t1 t2 t3 t_total t_training : Real),
      t1 = run1_distance / run1_speed →
      t2 = walk_distance / walk_speed →
      t3 = run2_distance / run2_speed →
      t_total = t1 + t2 + t3 →
      t_training = laps * t_total →
      t_training = (minutes * 60 + seconds)) := 
by
  intros laps track_length run1_distance run1_speed walk_distance walk_speed run2_distance run2_speed minutes seconds
  intros h_laps h_track_length h_run1_distance h_run1_speed h_walk_distance h_walk_speed h_run2_distance h_run2_speed h_minutes h_seconds
  intros t1 t2 t3 t_total t_training
  intros h_t1 h_t2 h_t3 h_t_total h_t_training
  sorry

end calculate_kevin_training_time_l1142_114216


namespace correct_option_D_l1142_114215

theorem correct_option_D (a b : ℝ) : 3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by sorry

end correct_option_D_l1142_114215


namespace square_side_length_same_area_l1142_114228

theorem square_side_length_same_area (length width : ℕ) (l_eq : length = 72) (w_eq : width = 18) : 
  ∃ side_length : ℕ, side_length * side_length = length * width ∧ side_length = 36 :=
by
  sorry

end square_side_length_same_area_l1142_114228


namespace m_is_perfect_square_l1142_114223

-- Given definitions and conditions
def is_odd (k : ℤ) : Prop := ∃ n : ℤ, k = 2 * n + 1

def is_perfect_square (m : ℕ) : Prop := ∃ a : ℕ, m = a * a

theorem m_is_perfect_square (k m n : ℕ) (h1 : (2 + Real.sqrt 3) ^ k = 1 + m + n * Real.sqrt 3)
  (h2 : 0 < m) (h3 : 0 < n) (h4 : 0 < k) (h5 : is_odd k) : is_perfect_square m := 
sorry

end m_is_perfect_square_l1142_114223


namespace solve_for_x_l1142_114298

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 1200 / 17 :=
by
  sorry

end solve_for_x_l1142_114298


namespace root_expression_value_l1142_114200

variables (a b : ℝ)
noncomputable def quadratic_eq (a b : ℝ) : Prop := (a + b = 1 ∧ a * b = -1)

theorem root_expression_value (h : quadratic_eq a b) : 3 * a ^ 2 + 4 * b + (2 / a ^ 2) = 11 := sorry

end root_expression_value_l1142_114200


namespace no_valid_weights_l1142_114268

theorem no_valid_weights (w_1 w_2 w_3 w_4 : ℝ) : 
  w_1 + w_2 + w_3 = 100 → w_1 + w_2 + w_4 = 101 → w_2 + w_3 + w_4 = 102 → 
  w_1 < 90 → w_2 < 90 → w_3 < 90 → w_4 < 90 → False :=
by 
  intros h1 h2 h3 hl1 hl2 hl3 hl4
  sorry

end no_valid_weights_l1142_114268


namespace sum_of_arithmetic_sequence_l1142_114297

variable {α : Type*} [LinearOrderedField α]

noncomputable def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n, a (n + 1) - a n = d

noncomputable def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
n * (a 1 + a n) / 2

theorem sum_of_arithmetic_sequence {a : ℕ → α} {d : α}
  (h3 : a 3 * a 7 = -16)
  (h4 : a 4 + a 6 = 0)
  (ha : is_arithmetic_sequence a d) :
  ∃ (s : α), s = n * (n - 9) ∨ s = -n * (n - 9) :=
sorry

end sum_of_arithmetic_sequence_l1142_114297


namespace constructible_angles_l1142_114249

def is_constructible (θ : ℝ) : Prop :=
  -- Define that θ is constructible if it can be constructed using compass and straightedge.
  sorry

theorem constructible_angles (α : ℝ) (β : ℝ) (k n : ℤ) (hβ : is_constructible β) :
  is_constructible (k * α / 2^n + β) :=
sorry

end constructible_angles_l1142_114249


namespace KarenParagraphCount_l1142_114245

theorem KarenParagraphCount :
  ∀ (num_essays num_short_ans num_paragraphs total_time essay_time short_ans_time paragraph_time : ℕ),
    (num_essays = 2) →
    (num_short_ans = 15) →
    (total_time = 240) →
    (essay_time = 60) →
    (short_ans_time = 3) →
    (paragraph_time = 15) →
    (total_time = num_essays * essay_time + num_short_ans * short_ans_time + num_paragraphs * paragraph_time) →
    num_paragraphs = 5 :=
by
  sorry

end KarenParagraphCount_l1142_114245


namespace delta_max_success_ratio_l1142_114284

theorem delta_max_success_ratio :
  ∃ a b c d : ℕ, 
    0 < a ∧ a < b ∧ (40 * a) < (21 * b) ∧
    0 < c ∧ c < d ∧ (4 * c) < (3 * d) ∧
    b + d = 600 ∧
    (a + c) / 600 = 349 / 600 :=
by
  sorry

end delta_max_success_ratio_l1142_114284


namespace parabola_standard_equations_l1142_114204

noncomputable def parabola_focus_condition (x y : ℝ) : Prop := 
  x + 2 * y + 3 = 0

theorem parabola_standard_equations (x y : ℝ) 
  (h : parabola_focus_condition x y) :
  (y ^ 2 = -12 * x) ∨ (x ^ 2 = -6 * y) :=
by
  sorry

end parabola_standard_equations_l1142_114204


namespace points_on_line_l1142_114208

theorem points_on_line (x y : ℝ) (h : x + y = 0) : y = -x :=
by
  sorry

end points_on_line_l1142_114208


namespace max_b_c_plus_four_over_a_l1142_114255

theorem max_b_c_plus_four_over_a (a b c : ℝ) (ha : a < 0)
  (h_quad : ∀ x : ℝ, -1 < x ∧ x < 2 → (a * x^2 + b * x + c) > 0) : 
  b - c + 4 / a ≤ -4 :=
sorry

end max_b_c_plus_four_over_a_l1142_114255


namespace min_value_expression_l1142_114274

theorem min_value_expression :
  ∀ (x y z w : ℝ), x > 0 → y > 0 → z > 0 → w > 0 → x = y → x + y + z + w = 1 →
  (x + y + z) / (x * y * z * w) ≥ 1024 :=
by
  intros x y z w hx hy hz hw hxy hsum
  sorry

end min_value_expression_l1142_114274


namespace area_of_rectangular_field_l1142_114232

theorem area_of_rectangular_field (length width perimeter : ℕ) 
  (h_perimeter : perimeter = 2 * (length + width)) 
  (h_length : length = 15) 
  (h_perimeter_value : perimeter = 70) : 
  (length * width = 300) :=
by
  sorry

end area_of_rectangular_field_l1142_114232


namespace identify_quadratic_equation_l1142_114299

theorem identify_quadratic_equation :
  (∀ b c d : Prop, ∀ (f : ℕ → Prop), f 0 → ¬ f 1 → ¬ f 2 → ¬ f 3 → b ∧ ¬ c ∧ ¬ d) →
  (∀ x y : ℝ,  (x^2 + 2 = 0) = (b ∧ ¬ b → c ∧ ¬ c → d ∧ ¬ d)) :=
by
  intros;
  sorry

end identify_quadratic_equation_l1142_114299


namespace middle_group_frequency_l1142_114257

theorem middle_group_frequency (sample_size : ℕ) (num_rectangles : ℕ)
  (A_middle : ℝ) (other_area_sum : ℝ)
  (h1 : sample_size = 300)
  (h2 : num_rectangles = 9)
  (h3 : A_middle = 1 / 5 * other_area_sum)
  (h4 : other_area_sum + A_middle = 1) :
  sample_size * A_middle = 50 :=
by
  sorry

end middle_group_frequency_l1142_114257


namespace calculation_power_l1142_114246

theorem calculation_power :
  (0.125 : ℝ) ^ 2012 * (2 ^ 2012) ^ 3 = 1 :=
sorry

end calculation_power_l1142_114246


namespace product_of_fractions_l1142_114272

theorem product_of_fractions :
  (1 / 2) * (3 / 5) * (5 / 6) = 1 / 4 := 
by
  sorry

end product_of_fractions_l1142_114272


namespace quadrants_contain_points_l1142_114218

def satisfy_inequalities (x y : ℝ) : Prop :=
  y > -3 * x ∧ y > x + 2

def in_quadrant_I (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def in_quadrant_II (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem quadrants_contain_points (x y : ℝ) :
  satisfy_inequalities x y → (in_quadrant_I x y ∨ in_quadrant_II x y) :=
sorry

end quadrants_contain_points_l1142_114218


namespace find_value_l1142_114264

variable (x y a c : ℝ)

-- Conditions
def condition1 : Prop := x * y = 2 * c
def condition2 : Prop := (1 / x ^ 2) + (1 / y ^ 2) = 3 * a

-- Proof statement
theorem find_value : condition1 x y c ∧ condition2 x y a ↔ (x + y) ^ 2 = 12 * a * c ^ 2 + 4 * c := 
by 
  -- Placeholder for the actual proof
  sorry

end find_value_l1142_114264


namespace solve_eqs_l1142_114282

theorem solve_eqs (x y : ℝ) 
  (h1 : x^2 + y^2 = 2)
  (h2 : x^2 / (2 - y) + y^2 / (2 - x) = 2) :
  x = 1 ∧ y = 1 :=
by
  sorry

end solve_eqs_l1142_114282


namespace range_of_quadratic_expression_l1142_114203

theorem range_of_quadratic_expression :
  (∃ x : ℝ, y = 2 * x^2 - 4 * x + 12) ↔ (y ≥ 10) :=
by
  sorry

end range_of_quadratic_expression_l1142_114203


namespace simple_interest_rate_l1142_114262

theorem simple_interest_rate
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (h1 : SI = 400)
  (h2 : P = 800)
  (h3 : T = 2) :
  R = 25 :=
by
  sorry

end simple_interest_rate_l1142_114262


namespace vertical_angles_eq_l1142_114256

theorem vertical_angles_eq (A B : Type) (are_vertical : A = B) :
  A = B := 
by
  exact are_vertical

end vertical_angles_eq_l1142_114256


namespace diet_soda_bottles_l1142_114285

theorem diet_soda_bottles (R D : ℕ) 
  (h1 : R = 60)
  (h2 : R = D + 41) :
  D = 19 :=
by {
  sorry
}

end diet_soda_bottles_l1142_114285


namespace polynomial_remainder_l1142_114254

theorem polynomial_remainder (x : ℂ) : 
  (3 * x ^ 1010 + x ^ 1000) % (x ^ 2 + 1) * (x - 1) = 3 * x ^ 2 + 1 := 
sorry

end polynomial_remainder_l1142_114254


namespace sum_coeff_expansion_l1142_114288

theorem sum_coeff_expansion (x y : ℝ) : 
  (x + 2 * y)^4 = 81 := sorry

end sum_coeff_expansion_l1142_114288


namespace floor_sqrt_245_l1142_114280

theorem floor_sqrt_245 : (Int.floor (Real.sqrt 245)) = 15 :=
by
  sorry

end floor_sqrt_245_l1142_114280


namespace count_paths_word_l1142_114277

def move_right_or_down_paths (n : ℕ) : ℕ := 2^n

theorem count_paths_word (n : ℕ) (w : String) (start : Char) (end_ : Char) :
    w = "строка" ∧ start = 'C' ∧ end_ = 'A' ∧ n = 5 →
    move_right_or_down_paths n = 32 :=
by
  intro h
  cases h
  sorry

end count_paths_word_l1142_114277


namespace slices_left_l1142_114286

-- Conditions
def total_slices : ℕ := 16
def fraction_eaten : ℚ := 3/4
def fraction_left : ℚ := 1 - fraction_eaten

-- Proof statement
theorem slices_left : total_slices * fraction_left = 4 := by
  sorry

end slices_left_l1142_114286


namespace initial_value_calculation_l1142_114253

theorem initial_value_calculation (P : ℝ) (h1 : ∀ n : ℕ, 0 ≤ n →
                                (P:ℝ) * (1 + 1/8) ^ n = 78468.75 → n = 2) :
  P = 61952 :=
sorry

end initial_value_calculation_l1142_114253


namespace valid_documents_count_l1142_114278

-- Definitions based on the conditions
def total_papers : ℕ := 400
def invalid_percentage : ℝ := 0.40
def valid_percentage : ℝ := 1.0 - invalid_percentage

-- Question and answer formalized as a theorem
theorem valid_documents_count : total_papers * valid_percentage = 240 := by
  sorry

end valid_documents_count_l1142_114278


namespace nate_total_run_l1142_114205

def field_length := 168
def initial_run := 4 * field_length
def additional_run := 500
def total_run := initial_run + additional_run

theorem nate_total_run : total_run = 1172 := by
  sorry

end nate_total_run_l1142_114205


namespace slips_with_3_l1142_114269

-- Definitions of the conditions
def num_slips : ℕ := 15
def expected_value : ℚ := 5.4

-- Theorem statement
theorem slips_with_3 (y : ℕ) (t : ℕ := num_slips) (E : ℚ := expected_value) :
  E = (3 * y + 8 * (t - y)) / t → y = 8 :=
by
  sorry

end slips_with_3_l1142_114269


namespace reggie_free_throws_l1142_114259

namespace BasketballShootingContest

-- Define the number of points for different shots
def points (layups free_throws long_shots : ℕ) : ℕ :=
  1 * layups + 2 * free_throws + 3 * long_shots

-- Conditions given in the problem
def Reggie_points (F: ℕ) : ℕ := 
  points 3 F 1

def Brother_points : ℕ := 
  points 0 0 4

-- The given condition that Reggie loses by 2 points
theorem reggie_free_throws:
  ∃ F : ℕ, Reggie_points F + 2 = Brother_points :=
sorry

end BasketballShootingContest

end reggie_free_throws_l1142_114259


namespace smallest_sum_of_three_integers_l1142_114260

theorem smallest_sum_of_three_integers (a b c : ℕ) (h1: a ≠ b) (h2: b ≠ c) (h3: a ≠ c) (h4: a * b * c = 72) :
  a + b + c = 13 :=
sorry

end smallest_sum_of_three_integers_l1142_114260


namespace y_increase_for_x_increase_l1142_114250

theorem y_increase_for_x_increase (x y : ℝ) (h : 4 * y = 9) : 12 * y = 27 :=
by
  sorry

end y_increase_for_x_increase_l1142_114250


namespace exist_consecutive_days_20_games_l1142_114290

theorem exist_consecutive_days_20_games 
  (a : ℕ → ℕ)
  (h_daily : ∀ n, a (n + 1) - a n ≥ 1)
  (h_weekly : ∀ n, a (n + 7) - a n ≤ 12) :
  ∃ i j, i < j ∧ a j - a i = 20 := by 
  sorry

end exist_consecutive_days_20_games_l1142_114290


namespace solution_l1142_114236

theorem solution :
  ∀ (x : ℝ), x ≠ 0 → (9 * x) ^ 18 = (27 * x) ^ 9 → x = 1 / 3 :=
by
  intro x
  intro h
  intro h_eq
  sorry

end solution_l1142_114236


namespace square_not_end_with_four_identical_digits_l1142_114283

theorem square_not_end_with_four_identical_digits (n : ℕ) (d : ℕ) :
  n = d * d → ¬ (d ≠ 0 ∧ (n % 10000 = d ^ 4)) :=
by
  sorry

end square_not_end_with_four_identical_digits_l1142_114283


namespace classroom_students_count_l1142_114265

-- Definitions from the conditions
def students (C S Sh : ℕ) : Prop :=
  S = 2 * C ∧
  S = Sh + 8 ∧
  Sh = C + 19

-- Proof statement
theorem classroom_students_count (C S Sh : ℕ) 
  (h : students C S Sh) : 3 * C = 81 :=
by
  sorry

end classroom_students_count_l1142_114265


namespace motorcycle_travel_distance_l1142_114207

noncomputable def motorcycle_distance : ℝ :=
  let t : ℝ := 1 / 2  -- time in hours (30 minutes)
  let v_bus : ℝ := 90  -- speed of the bus in km/h
  let v_motorcycle : ℝ := (2 / 3) * v_bus  -- speed of the motorcycle in km/h
  v_motorcycle * t  -- calculates the distance traveled by the motorcycle in km

theorem motorcycle_travel_distance :
  motorcycle_distance = 30 := by
  sorry

end motorcycle_travel_distance_l1142_114207


namespace rate_mangoes_correct_l1142_114266

-- Define the conditions
def weight_apples : ℕ := 8
def rate_apples : ℕ := 70
def cost_apples := weight_apples * rate_apples

def total_payment : ℕ := 1145
def weight_mangoes : ℕ := 9
def cost_mangoes := total_payment - cost_apples

-- Define the rate per kg of mangoes
def rate_mangoes := cost_mangoes / weight_mangoes

-- Prove the rate per kg for mangoes
theorem rate_mangoes_correct : rate_mangoes = 65 := by
  -- all conditions and intermediate calculations already stated
  sorry

end rate_mangoes_correct_l1142_114266


namespace reciprocal_expression_equals_two_l1142_114293

theorem reciprocal_expression_equals_two (x y : ℝ) (h : x * y = 1) : 
  (x + 1 / y) * (2 * y - 1 / x) = 2 := by
  sorry

end reciprocal_expression_equals_two_l1142_114293


namespace find_a_m_l1142_114279

theorem find_a_m :
  ∃ a m : ℤ,
    (a = -2) ∧ (m = -1 ∨ m = 3) ∧ 
    (∀ x : ℝ, (a - 1) * x^2 + a * x + 1 = 0 → 
               (m^2 + m) * x^2 + 3 * m * x - 3 = 0) := sorry

end find_a_m_l1142_114279


namespace two_sum_fourth_power_square_l1142_114226

-- Define the condition
def sum_zero (x y z : ℤ) : Prop := x + y + z = 0

-- The theorem to be proven
theorem two_sum_fourth_power_square (x y z : ℤ) (h : sum_zero x y z) : ∃ k : ℤ, 2 * (x^4 + y^4 + z^4) = k^2 :=
by
  -- skipping the proof
  sorry

end two_sum_fourth_power_square_l1142_114226


namespace smallest_b_factors_l1142_114206

theorem smallest_b_factors (b p q : ℤ) (hb : b = p + q) (hpq : p * q = 2052) : b = 132 :=
sorry

end smallest_b_factors_l1142_114206


namespace find_number_of_numbers_l1142_114275

theorem find_number_of_numbers (S : ℝ) (n : ℝ) (h1 : S - 30 = 16 * n) (h2 : S = 19 * n) : n = 10 :=
by
  sorry

end find_number_of_numbers_l1142_114275


namespace cost_of_book_l1142_114270

theorem cost_of_book (cost_album : ℝ) (discount_rate : ℝ) (cost_CD : ℝ) (cost_book : ℝ) 
  (h1 : cost_album = 20)
  (h2 : discount_rate = 0.30)
  (h3 : cost_CD = cost_album * (1 - discount_rate))
  (h4 : cost_book = cost_CD + 4) :
  cost_book = 18 := by
  sorry

end cost_of_book_l1142_114270


namespace vacation_days_in_march_l1142_114214

theorem vacation_days_in_march 
  (days_worked : ℕ) 
  (days_worked_to_vacation_days : ℕ) 
  (vacation_days_left : ℕ) 
  (days_in_march : ℕ) 
  (days_in_september : ℕ)
  (h1 : days_worked = 300)
  (h2 : days_worked_to_vacation_days = 10)
  (h3 : vacation_days_left = 15)
  (h4 : days_in_september = 2 * days_in_march)
  (h5 : days_worked / days_worked_to_vacation_days - (days_in_march + days_in_september) = vacation_days_left) 
  : days_in_march = 5 := 
by
  sorry

end vacation_days_in_march_l1142_114214


namespace total_chickens_and_ducks_l1142_114252

-- Definitions based on conditions
def num_chickens : Nat := 45
def more_chickens_than_ducks : Nat := 8
def num_ducks : Nat := num_chickens - more_chickens_than_ducks

-- The proof statement
theorem total_chickens_and_ducks : num_chickens + num_ducks = 82 := by
  -- The actual proof is omitted, only the statement is required
  sorry

end total_chickens_and_ducks_l1142_114252


namespace percent_greater_than_l1142_114220

theorem percent_greater_than (M N : ℝ) (hN : N ≠ 0) : (M - N) / N * 100 = 100 * (M - N) / N :=
by sorry

end percent_greater_than_l1142_114220


namespace simplify_expression_l1142_114227

variable (q : Int) -- condition that q is an integer

theorem simplify_expression (q : Int) : 
  ((7 * q + 3) - 3 * q * 2) * 4 + (5 - 2 / 4) * (8 * q - 12) = 40 * q - 42 :=
  by
  sorry

end simplify_expression_l1142_114227


namespace correct_quadratic_opens_upwards_l1142_114225

-- Define the quadratic functions
def A (x : ℝ) : ℝ := 1 - x - 6 * x^2
def B (x : ℝ) : ℝ := -8 * x + x^2 + 1
def C (x : ℝ) : ℝ := (1 - x) * (x + 5)
def D (x : ℝ) : ℝ := 2 - (5 - x)^2

-- The theorem stating that function B is the one that opens upwards
theorem correct_quadratic_opens_upwards :
  ∃ (f : ℝ → ℝ) (h : f = B), ∀ (a b c : ℝ), f x = a * x^2 + b * x + c → a > 0 :=
sorry

end correct_quadratic_opens_upwards_l1142_114225


namespace expression_for_f_l1142_114229

noncomputable def f (x : ℝ) : ℝ := sorry

theorem expression_for_f (x : ℝ) :
  (∀ x, f (x - 1) = x^2) → f x = x^2 + 2 * x + 1 :=
by
  intro h
  sorry

end expression_for_f_l1142_114229


namespace alice_walks_miles_each_morning_l1142_114244

theorem alice_walks_miles_each_morning (x : ℕ) :
  (5 * x + 5 * 12 = 110) → x = 10 :=
by
  intro h
  -- Proof omitted
  sorry

end alice_walks_miles_each_morning_l1142_114244


namespace diagonals_in_30_sided_polygon_l1142_114291

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l1142_114291


namespace patricia_candies_final_l1142_114233

def initial_candies : ℕ := 764
def taken_candies : ℕ := 53
def back_candies_per_7_taken : ℕ := 19

theorem patricia_candies_final :
  let given_back_times := taken_candies / 7
  let total_given_back := given_back_times * back_candies_per_7_taken
  let final_candies := initial_candies - taken_candies + total_given_back
  final_candies = 844 :=
by
  sorry

end patricia_candies_final_l1142_114233


namespace find_integer_solutions_l1142_114263

theorem find_integer_solutions (n : ℕ) (h1 : ∃ b : ℤ, 8 * n - 7 = b^2) (h2 : ∃ a : ℤ, 18 * n - 35 = a^2) : 
  n = 2 ∨ n = 22 := 
sorry

end find_integer_solutions_l1142_114263


namespace find_starting_number_l1142_114241

theorem find_starting_number (k m : ℕ) (hk : 67 = (m - k) / 3 + 1) (hm : m = 300) : k = 102 := by
  sorry

end find_starting_number_l1142_114241


namespace correct_avg_weight_l1142_114209

theorem correct_avg_weight (initial_avg_weight : ℚ) (num_boys : ℕ) (misread_weight : ℚ) (correct_weight : ℚ) :
  initial_avg_weight = 58.4 → num_boys = 20 → misread_weight = 56 → correct_weight = 60 →
  (initial_avg_weight * num_boys + (correct_weight - misread_weight)) / num_boys = 58.6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Plugging in the values makes the calculation straightforward, resulting in: 
  -- (58.4 * 20 + (60 - 56)) / 20 = 58.6 
  -- thus this verification step is:
  sorry

end correct_avg_weight_l1142_114209


namespace division_problem_l1142_114267

theorem division_problem : 250 / (15 + 13 * 3 - 4) = 5 := by
  sorry

end division_problem_l1142_114267


namespace correct_option_l1142_114231

-- Conditions as definitions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (a : ℝ) : Prop := 3 * a - 2 * a = 1
def optionC (a : ℝ) : Prop := (-2 * a^2)^3 = -8 * a^6
def optionD (a : ℝ) : Prop := a^6 / a^2 = a^3

-- The statement to prove
theorem correct_option (a : ℝ) : optionC a :=
by 
  unfold optionC
  sorry

end correct_option_l1142_114231


namespace blueberry_pancakes_count_l1142_114239

-- Definitions of the conditions
def total_pancakes : ℕ := 67
def banana_pancakes : ℕ := 24
def plain_pancakes : ℕ := 23

-- Statement of the problem
theorem blueberry_pancakes_count :
  total_pancakes - banana_pancakes - plain_pancakes = 20 := by
  sorry

end blueberry_pancakes_count_l1142_114239


namespace less_than_its_reciprocal_l1142_114222

-- Define the numbers as constants
def a := -1/3
def b := -3/2
def c := 1/4
def d := 3/4
def e := 4/3 

-- Define the proposition that needs to be proved
theorem less_than_its_reciprocal (n : ℚ) :
  (n = -3/2 ∨ n = 1/4) ↔ (n < 1/n) :=
by
  sorry

end less_than_its_reciprocal_l1142_114222


namespace equal_roots_condition_l1142_114289

theorem equal_roots_condition (m : ℝ) :
  (m = 2 ∨ m = (9 + Real.sqrt 57) / 8 ∨ m = (9 - Real.sqrt 57) / 8) →
  ∃ a b c : ℝ, 
  (∀ x : ℝ, (a * x ^ 2 + b * x + c = 0) ↔
  (x * (x - 3) - (m + 2)) / ((x - 3) * (m - 2)) = x / m) ∧
  (b^2 - 4 * a * c = 0) :=
sorry

end equal_roots_condition_l1142_114289


namespace gcd_problem_l1142_114221

theorem gcd_problem : Nat.gcd 12740 220 - 10 = 10 :=
by
  sorry

end gcd_problem_l1142_114221


namespace find_x_l1142_114247

theorem find_x (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1)
  (geom_seq : (x - ⌊x⌋) * x = ⌊x⌋^2) : x = 1.618 :=
by
  sorry

end find_x_l1142_114247


namespace population_control_l1142_114295

   noncomputable def population_growth (initial_population : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
   initial_population * (1 + growth_rate / 100) ^ years

   theorem population_control {initial_population : ℝ} {threshold_population : ℝ} {growth_rate : ℝ} {years : ℕ} :
     initial_population = 1.3 ∧ threshold_population = 1.4 ∧ growth_rate = 0.74 ∧ years = 10 →
     population_growth initial_population growth_rate years < threshold_population :=
   by
     intros
     sorry
   
end population_control_l1142_114295


namespace y_coordinate_of_second_point_l1142_114219

theorem y_coordinate_of_second_point
  (m n : ℝ)
  (h₁ : m = 2 * n + 3)
  (h₂ : m + 2 = 2 * (n + 1) + 3) :
  (n + 1) = n + 1 :=
by
  -- proof to be provided
  sorry

end y_coordinate_of_second_point_l1142_114219


namespace steve_total_time_on_roads_l1142_114234

variables (d : ℝ) (v_back : ℝ) (v_to_work : ℝ)

-- Constants from the problem statement
def distance := 10 -- The distance from Steve's house to work is 10 km
def speed_back := 5 -- Steve's speed on the way back from work is 5 km/h

-- Given conditions
def speed_to_work := speed_back / 2 -- On the way back, Steve drives twice as fast as he did on the way to work

-- Define the time to get to work and back
def time_to_work := distance / speed_to_work
def time_back_home := distance / speed_back

-- Total time on roads
def total_time := time_to_work + time_back_home

-- The theorem to prove
theorem steve_total_time_on_roads : total_time = 6 := by
  -- Proof here
  sorry

end steve_total_time_on_roads_l1142_114234


namespace cost_of_weed_eater_string_l1142_114202

-- Definitions
def num_blades := 4
def cost_per_blade := 8
def total_spent := 39
def total_cost_of_blades := num_blades * cost_per_blade
def cost_of_string := total_spent - total_cost_of_blades

-- The theorem statement
theorem cost_of_weed_eater_string : cost_of_string = 7 :=
by {
  -- The proof would go here
  sorry
}

end cost_of_weed_eater_string_l1142_114202


namespace moles_KOH_combined_l1142_114296

-- Define the number of moles of KI produced
def moles_KI_produced : ℕ := 3

-- Define the molar ratio from the balanced chemical equation
def molar_ratio_KOH_NH4I_KI : ℕ := 1

-- The number of moles of KOH combined to produce the given moles of KI
theorem moles_KOH_combined (moles_KOH moles_NH4I : ℕ) (h : moles_NH4I = 3) 
  (h_produced : moles_KI_produced = 3) (ratio : molar_ratio_KOH_NH4I_KI = 1) :
  moles_KOH = 3 :=
by {
  -- Placeholder for proof, use sorry to skip proving
  sorry
}

end moles_KOH_combined_l1142_114296


namespace intersection_of_A_and_B_l1142_114248

def A : Set ℝ := { x | x > 2 ∨ x < -1 }
def B : Set ℝ := { x | (x + 1) * (4 - x) < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | x > 3 ∨ x < -1 } := sorry

end intersection_of_A_and_B_l1142_114248


namespace greatest_drop_in_price_l1142_114217

def jan_change : ℝ := -0.75
def feb_change : ℝ := 1.50
def mar_change : ℝ := -3.00
def apr_change : ℝ := 2.50
def may_change : ℝ := -0.25
def jun_change : ℝ := 0.80
def jul_change : ℝ := -2.75
def aug_change : ℝ := -1.20

theorem greatest_drop_in_price : 
  mar_change = min (min (min (min (min (min jan_change jul_change) aug_change) may_change) feb_change) apr_change) jun_change :=
by
  -- This statement is where the proof would go.
  sorry

end greatest_drop_in_price_l1142_114217


namespace p_necessary_not_sufficient_for_q_l1142_114258

def condition_p (x : ℝ) : Prop := x > 2
def condition_q (x : ℝ) : Prop := x > 3

theorem p_necessary_not_sufficient_for_q (x : ℝ) :
  (∀ (x : ℝ), condition_q x → condition_p x) ∧ ¬(∀ (x : ℝ), condition_p x → condition_q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l1142_114258


namespace avg_visitors_sundays_l1142_114273

-- Definitions
def days_in_month := 30
def avg_visitors_per_day_month := 750
def avg_visitors_other_days := 700
def sundays_in_month := 5
def other_days := days_in_month - sundays_in_month

-- Main statement to prove
theorem avg_visitors_sundays (S : ℕ) 
  (H1 : days_in_month = 30) 
  (H2 : avg_visitors_per_day_month = 750) 
  (H3 : avg_visitors_other_days = 700) 
  (H4 : sundays_in_month = 5) 
  (H5 : other_days = days_in_month - sundays_in_month) 
  :
  (sundays_in_month * S + other_days * avg_visitors_other_days) = avg_visitors_per_day_month * days_in_month 
  → S = 1000 :=
by 
  sorry

end avg_visitors_sundays_l1142_114273


namespace brain_can_always_open_door_l1142_114201

noncomputable def can_open_door (a b c n m k : ℕ) : Prop :=
∃ x y z : ℕ, a^n = x^3 ∧ b^m = y^3 ∧ c^k = z^3

theorem brain_can_always_open_door :
  ∀ (a b c n m k : ℕ), 
  ∃ x y z : ℕ, a^n = x^3 ∧ b^m = y^3 ∧ c^k = z^3 :=
by sorry

end brain_can_always_open_door_l1142_114201


namespace group_selection_l1142_114240

theorem group_selection (m k n : ℕ) (h_m : m = 6) (h_k : k = 7) 
  (groups : ℕ → ℕ) (h_groups : groups k = n) : 
  n % 10 = (m + k) % 10 :=
by
  sorry

end group_selection_l1142_114240


namespace increasing_interval_l1142_114212

noncomputable def f (x k : ℝ) : ℝ := (x^2 / 2) - k * (Real.log x)

theorem increasing_interval (k : ℝ) (h₀ : 0 < k) : 
  ∃ (a : ℝ), (a = Real.sqrt k) ∧ 
  ∀ (x : ℝ), (x > a) → (∃ ε > 0, ∀ y, (x < y) → (f y k > f x k)) :=
sorry

end increasing_interval_l1142_114212


namespace david_money_left_l1142_114271

theorem david_money_left (S : ℤ) (h1 : S - 800 = 1800 - S) : 1800 - S = 500 :=
by
  sorry

end david_money_left_l1142_114271


namespace probability_of_roots_l1142_114210

theorem probability_of_roots (k : ℝ) (h1 : 8 ≤ k) (h2 : k ≤ 13) :
  let a := k^2 - 2 * k - 35
  let b := 3 * k - 9
  let c := 2
  let discriminant := b^2 - 4 * a * c
  discriminant ≥ 0 → 
  (∃ x1 x2 : ℝ, 
    a * x1^2 + b * x1 + c = 0 ∧ 
    a * x2^2 + b * x2 + c = 0 ∧
    x1 ≤ 2 * x2) ↔ 
  ∃ p : ℝ, p = 0.6 := 
sorry

end probability_of_roots_l1142_114210


namespace simplify_expression_l1142_114237

variables (x y z : ℝ)

theorem simplify_expression (h₁ : x ≠ 2) (h₂ : y ≠ 3) (h₃ : z ≠ 4) : 
  ((x - 2) / (4 - z)) * ((y - 3) / (2 - x)) * ((z - 4) / (3 - y)) = -1 :=
by sorry

end simplify_expression_l1142_114237


namespace remainder_of_product_mod_7_l1142_114235

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l1142_114235


namespace handshakes_exchanged_l1142_114281

-- Let n be the number of couples
noncomputable def num_couples := 7

-- Total number of people at the gathering
noncomputable def total_people := num_couples * 2

-- Number of people each person shakes hands with
noncomputable def handshakes_per_person := total_people - 2

-- Total number of unique handshakes
noncomputable def total_handshakes := total_people * handshakes_per_person / 2

theorem handshakes_exchanged :
  total_handshakes = 77 :=
by
  sorry

end handshakes_exchanged_l1142_114281


namespace find_alpha_l1142_114294

theorem find_alpha (α : ℝ) (k : ℤ) 
  (h : ∃ (k : ℤ), α + 30 = k * 360 + 180) : 
  α = k * 360 + 150 :=
by 
  sorry

end find_alpha_l1142_114294


namespace fraction_of_alvin_age_l1142_114276

variable (A E F : ℚ)

-- Conditions
def edwin_older_by_six : Prop := E = A + 6
def total_age : Prop := A + E = 30.99999999
def age_relation_in_two_years : Prop := E + 2 = F * (A + 2) + 20

-- Statement to prove
theorem fraction_of_alvin_age
  (h1 : edwin_older_by_six A E)
  (h2 : total_age A E)
  (h3 : age_relation_in_two_years A E F) :
  F = 1 / 29 :=
sorry

end fraction_of_alvin_age_l1142_114276


namespace part_a_part_b_l1142_114224

-- Part (a)
theorem part_a (ABC : Type) (M: ABC) (R_a R_b R_c r : ℝ):
  ∀ (ABC : Type) (A B C : ABC) (M : ABC), 
  R_a + R_b + R_c ≥ 6 * r := sorry

-- Part (b)
theorem part_b (ABC : Type) (M: ABC) (R_a R_b R_c r : ℝ):
  ∀ (ABC : Type) (A B C : ABC) (M : ABC), 
  R_a^2 + R_b^2 + R_c^2 ≥ 12 * r^2 := sorry

end part_a_part_b_l1142_114224


namespace ratio_of_intercepts_l1142_114238

variable {c : ℝ} (non_zero_c : c ≠ 0) (u v : ℝ)
-- Condition: The first line, slope 8, y-intercept c, x-intercept (u, 0)
variable (h_u : u = -c / 8)
-- Condition: The second line, slope 4, y-intercept c, x-intercept (v, 0)
variable (h_v : v = -c / 4)

theorem ratio_of_intercepts (non_zero_c : c ≠ 0)
    (h_u : u = -c / 8) (h_v : v = -c / 4) : u / v = 1 / 2 :=
by
  sorry

end ratio_of_intercepts_l1142_114238


namespace roots_not_in_interval_l1142_114230

theorem roots_not_in_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, (a^x + a^(-x) = 2 * a) → (x < -1 ∨ x > 1) :=
by
  sorry

end roots_not_in_interval_l1142_114230


namespace find_XY_XZ_l1142_114242

open Set

variable (P Q R X Y Z : Type) [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited X] [Inhabited Y] [Inhabited Z]
variable (length : (P → P → Real) → (Q → Q → Real) → (R → R → Real) → (X → X → Real) → (Y → Y → Real) → (Z → Z → Real) )


-- Definitions based on the conditions
def similar_triangles (PQ QR PR XY XZ YZ : Real) : Prop :=
  QR / YZ = PQ / XY ∧ QR / YZ = PR / XZ

def PQ : Real := 8
def QR : Real := 16
def YZ : Real := 32

-- We need to prove (XY = 16 ∧ XZ = 32) given the conditions of similarity
theorem find_XY_XZ (XY XZ : Real) (h_sim : similar_triangles PQ QR PQ XY XZ YZ) : XY = 16 ∧ XZ = 32 :=
by
  sorry

end find_XY_XZ_l1142_114242


namespace nonneg_sets_property_l1142_114261

open Set Nat

theorem nonneg_sets_property (A : Set ℕ) :
  (∀ m n : ℕ, m + n ∈ A → m * n ∈ A) ↔
  (A = ∅ ∨ A = {0} ∨ A = {0, 1} ∨ A = {0, 1, 2} ∨ A = {0, 1, 2, 3} ∨ A = {0, 1, 2, 3, 4} ∨ A = { n | 0 ≤ n }) :=
sorry

end nonneg_sets_property_l1142_114261
