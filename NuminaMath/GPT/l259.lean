import Mathlib

namespace NUMINAMATH_GPT_estimate_probability_l259_25910

noncomputable def freq_20 : ℝ := 0.300
noncomputable def freq_50 : ℝ := 0.360
noncomputable def freq_100 : ℝ := 0.350
noncomputable def freq_300 : ℝ := 0.350
noncomputable def freq_500 : ℝ := 0.352
noncomputable def freq_1000 : ℝ := 0.351
noncomputable def freq_5000 : ℝ := 0.351

theorem estimate_probability : (|0.35 - ((freq_20 + freq_50 + freq_100 + freq_300 + freq_500 + freq_1000 + freq_5000) / 7)| < 0.01) :=
by sorry

end NUMINAMATH_GPT_estimate_probability_l259_25910


namespace NUMINAMATH_GPT_water_segment_length_l259_25956

theorem water_segment_length 
  (total_distance : ℝ)
  (find_probability : ℝ)
  (lose_probability : ℝ)
  (probability_equation : total_distance * lose_probability = 750) :
  total_distance = 2500 → 
  find_probability = 7 / 10 →
  lose_probability = 3 / 10 →
  x = 750 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_water_segment_length_l259_25956


namespace NUMINAMATH_GPT_rollo_guinea_pigs_food_l259_25903

theorem rollo_guinea_pigs_food :
  let first_food := 2
  let second_food := 2 * first_food
  let third_food := second_food + 3
  first_food + second_food + third_food = 13 :=
by
  sorry

end NUMINAMATH_GPT_rollo_guinea_pigs_food_l259_25903


namespace NUMINAMATH_GPT_polynomial_roots_product_l259_25900

theorem polynomial_roots_product (a b : ℤ)
  (h1 : ∀ (r : ℝ), r^2 - r - 2 = 0 → r^3 - a * r - b = 0) : a * b = 6 := sorry

end NUMINAMATH_GPT_polynomial_roots_product_l259_25900


namespace NUMINAMATH_GPT_retail_price_l259_25984

theorem retail_price (R : ℝ) (wholesale_price : ℝ)
  (discount_rate : ℝ) (profit_rate : ℝ)
  (selling_price : ℝ) :
  wholesale_price = 81 →
  discount_rate = 0.10 →
  profit_rate = 0.20 →
  selling_price = wholesale_price * (1 + profit_rate) →
  selling_price = R * (1 - discount_rate) →
  R = 108 := 
by 
  intros h_wholesale h_discount h_profit h_selling_price h_discounted_selling_price
  sorry

end NUMINAMATH_GPT_retail_price_l259_25984


namespace NUMINAMATH_GPT_ratio_surface_area_l259_25920

noncomputable def side_length (a : ℝ) := a
noncomputable def radius (R : ℝ) := R

theorem ratio_surface_area (a R : ℝ) (h : a^3 = (4/3) * Real.pi * R^3) : 
  (6 * a^2) / (4 * Real.pi * R^2) = (3 * (6 / Real.pi)) :=
by sorry

end NUMINAMATH_GPT_ratio_surface_area_l259_25920


namespace NUMINAMATH_GPT_billy_total_problems_solved_l259_25917

theorem billy_total_problems_solved :
  ∃ (Q : ℕ), (3 * Q = 132) ∧ ((Q) + (2 * Q) + (3 * Q) = 264) :=
by
  sorry

end NUMINAMATH_GPT_billy_total_problems_solved_l259_25917


namespace NUMINAMATH_GPT_geometric_arithmetic_sequence_l259_25911

theorem geometric_arithmetic_sequence 
  (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (q : ℝ) 
  (h0 : 0 < q) (h1 : q ≠ 1)
  (h2 : ∀ n, a_n n = a_n 1 * q ^ (n - 1)) -- a_n is a geometric sequence
  (h3 : 2 * a_n 3 * a_n 5 = a_n 4 * (a_n 3 + a_n 5)) -- a3, a5, a4 form an arithmetic sequence
  (h4 : ∀ n, S n = a_n 1 * (1 - q^n) / (1 - q)) -- S_n is the sum of the first n terms
  : S 6 / S 3 = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_arithmetic_sequence_l259_25911


namespace NUMINAMATH_GPT_expected_adjacent_black_pairs_proof_l259_25985

-- Define the modified deck conditions.
def modified_deck (n : ℕ) := n = 60
def black_cards (b : ℕ) := b = 30
def red_cards (r : ℕ) := r = 30

-- Define the expected value of pairs of adjacent black cards.
def expected_adjacent_black_pairs (n b : ℕ) : ℚ :=
  b * (b - 1) / (n - 1)

theorem expected_adjacent_black_pairs_proof :
  modified_deck 60 →
  black_cards 30 →
  red_cards 30 →
  expected_adjacent_black_pairs 60 30 = 870 / 59 :=
by intros; sorry

end NUMINAMATH_GPT_expected_adjacent_black_pairs_proof_l259_25985


namespace NUMINAMATH_GPT_find_pointA_coordinates_l259_25918

-- Define point B
def pointB : ℝ × ℝ := (4, -1)

-- Define the symmetry condition with respect to the x-axis
def symmetricWithRespectToXAxis (p₁ p₂ : ℝ × ℝ) : Prop :=
  p₁.1 = p₂.1 ∧ p₁.2 = -p₂.2

-- Theorem statement: Prove the coordinates of point A given the conditions
theorem find_pointA_coordinates :
  ∃ A : ℝ × ℝ, symmetricWithRespectToXAxis pointB A ∧ A = (4, 1) :=
by
  sorry

end NUMINAMATH_GPT_find_pointA_coordinates_l259_25918


namespace NUMINAMATH_GPT_area_of_figure_l259_25962

theorem area_of_figure : 
  ∀ (x y : ℝ), |3 * x + 4| + |4 * y - 3| ≤ 12 → area_of_rhombus = 24 := 
by
  sorry

end NUMINAMATH_GPT_area_of_figure_l259_25962


namespace NUMINAMATH_GPT_find_solns_to_eqn_l259_25946

theorem find_solns_to_eqn (x y z w : ℕ) :
  2^x * 3^y - 5^z * 7^w = 1 ↔ (x, y, z, w) = (1, 0, 0, 0) ∨ 
                                        (x, y, z, w) = (3, 0, 0, 1) ∨ 
                                        (x, y, z, w) = (1, 1, 1, 0) ∨ 
                                        (x, y, z, w) = (2, 2, 1, 1) := 
sorry -- Placeholder for the actual proof

end NUMINAMATH_GPT_find_solns_to_eqn_l259_25946


namespace NUMINAMATH_GPT_table_area_l259_25908

theorem table_area (A : ℝ) 
  (combined_area : ℝ)
  (coverage_percentage : ℝ)
  (area_two_layers : ℝ)
  (area_three_layers : ℝ)
  (combined_area_eq : combined_area = 220)
  (coverage_percentage_eq : coverage_percentage = 0.80 * A)
  (area_two_layers_eq : area_two_layers = 24)
  (area_three_layers_eq : area_three_layers = 28) :
  A = 275 :=
by
  -- Assumptions and derivations can be filled in.
  sorry

end NUMINAMATH_GPT_table_area_l259_25908


namespace NUMINAMATH_GPT_students_contribution_l259_25968

theorem students_contribution (n x : ℕ) 
  (h₁ : ∃ (k : ℕ), k * 9 = 22725)
  (h₂ : n * x = k / 9)
  : (n = 5 ∧ x = 505) ∨ (n = 25 ∧ x = 101) :=
sorry

end NUMINAMATH_GPT_students_contribution_l259_25968


namespace NUMINAMATH_GPT_sum_of_numbers_l259_25915

theorem sum_of_numbers (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : ab + bc + ca = 100) :
  a + b + c = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l259_25915


namespace NUMINAMATH_GPT_snow_fall_time_l259_25998

theorem snow_fall_time :
  (∀ rate_per_six_minutes : ℕ, rate_per_six_minutes = 1 →
    (∀ minute : ℕ, minute = 6 →
      (∀ height_in_m : ℕ, height_in_m = 1 →
        ∃ time_in_hours : ℕ, time_in_hours = 100 ))) :=
sorry

end NUMINAMATH_GPT_snow_fall_time_l259_25998


namespace NUMINAMATH_GPT_find_certain_number_l259_25906

theorem find_certain_number (x : ℝ) (h : 0.7 * x = 28) : x = 40 := 
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l259_25906


namespace NUMINAMATH_GPT_smallest_n_for_divisibility_problem_l259_25997

theorem smallest_n_for_divisibility_problem :
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n * (n + 1) ≠ 0 ∧
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ ¬ (n * (n + 1)) % k = 0) ∧
  ∀ m : ℕ, m > 0 ∧ m < n → (∀ k : ℕ, 1 ≤ k ∧ k ≤ m → (m * (m + 1)) % k ≠ 0)) → n = 4 := sorry

end NUMINAMATH_GPT_smallest_n_for_divisibility_problem_l259_25997


namespace NUMINAMATH_GPT_find_m_n_find_a_l259_25943

def quadratic_roots (x : ℝ) (m n : ℝ) : Prop := 
  x^2 + m * x - 3 = 0

theorem find_m_n {m n : ℝ} : 
  quadratic_roots (-1) m n ∧ quadratic_roots n m n → 
  m = -2 ∧ n = 3 := 
sorry

def f (x m : ℝ) : ℝ := 
  x^2 + m * x - 3

theorem find_a {a m : ℝ} (h : m = -2) : 
  f 3 m = f (2 * a - 3) m → 
  a = 1 ∨ a = 3 := 
sorry

end NUMINAMATH_GPT_find_m_n_find_a_l259_25943


namespace NUMINAMATH_GPT_babysitting_earnings_l259_25960

theorem babysitting_earnings
  (cost_video_game : ℕ)
  (cost_candy : ℕ)
  (hours_worked : ℕ)
  (amount_left : ℕ)
  (total_earned : ℕ)
  (earnings_per_hour : ℕ) :
  cost_video_game = 60 →
  cost_candy = 5 →
  hours_worked = 9 →
  amount_left = 7 →
  total_earned = cost_video_game + cost_candy + amount_left →
  earnings_per_hour = total_earned / hours_worked →
  earnings_per_hour = 8 :=
by
  intros h_game h_candy h_hours h_left h_total_earned h_earn_per_hour
  rw [h_game, h_candy] at h_total_earned
  simp at h_total_earned
  have h_total_earned : total_earned = 72 := by linarith
  rw [h_total_earned, h_hours] at h_earn_per_hour
  simp at h_earn_per_hour
  assumption

end NUMINAMATH_GPT_babysitting_earnings_l259_25960


namespace NUMINAMATH_GPT_mart_income_percentage_j_l259_25963

variables (J T M : ℝ)

-- condition: Tim's income is 40 percent less than Juan's income
def tims_income := T = 0.60 * J

-- condition: Mart's income is 40 percent more than Tim's income
def marts_income := M = 1.40 * T

-- goal: Prove that Mart's income is 84 percent of Juan's income
theorem mart_income_percentage_j (J : ℝ) (T : ℝ) (M : ℝ)
  (h1 : T = 0.60 * J) 
  (h2 : M = 1.40 * T) : 
  M = 0.84 * J := 
sorry

end NUMINAMATH_GPT_mart_income_percentage_j_l259_25963


namespace NUMINAMATH_GPT_paige_finished_problems_l259_25928

-- Define the conditions
def initial_problems : ℕ := 110
def problems_per_page : ℕ := 9
def remaining_pages : ℕ := 7

-- Define the statement we want to prove
theorem paige_finished_problems :
  initial_problems - (remaining_pages * problems_per_page) = 47 :=
by sorry

end NUMINAMATH_GPT_paige_finished_problems_l259_25928


namespace NUMINAMATH_GPT_value_of_g_at_five_l259_25902

def g (x : ℕ) : ℕ := x^2 - 2 * x

theorem value_of_g_at_five : g 5 = 15 := by
  sorry

end NUMINAMATH_GPT_value_of_g_at_five_l259_25902


namespace NUMINAMATH_GPT_fifth_house_number_is_13_l259_25924

theorem fifth_house_number_is_13 (n : ℕ) (a₁ : ℕ) (h₀ : n ≥ 5) (h₁ : (a₁ + n - 1) * n = 117) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n -> (a₁ + 2 * (i - 1)) = 2*(i-1) + a₁) : 
  (a₁ + 2 * (5 - 1)) = 13 :=
by
  sorry

end NUMINAMATH_GPT_fifth_house_number_is_13_l259_25924


namespace NUMINAMATH_GPT_decimal_equivalence_l259_25947

theorem decimal_equivalence : 4 + 3 / 10 + 9 / 1000 = 4.309 := 
by
  sorry

end NUMINAMATH_GPT_decimal_equivalence_l259_25947


namespace NUMINAMATH_GPT_time_to_cross_is_30_seconds_l259_25927

variable (length_train : ℕ) (speed_km_per_hr : ℕ) (length_bridge : ℕ)

def total_distance := length_train + length_bridge

def speed_m_per_s := (speed_km_per_hr * 1000 : ℕ) / 3600

def time_to_cross_bridge := total_distance length_train length_bridge / speed_m_per_s speed_km_per_hr

theorem time_to_cross_is_30_seconds 
  (h_train_length : length_train = 140)
  (h_train_speed : speed_km_per_hr = 45)
  (h_bridge_length : length_bridge = 235) :
  time_to_cross_bridge length_train speed_km_per_hr length_bridge = 30 :=
by
  sorry

end NUMINAMATH_GPT_time_to_cross_is_30_seconds_l259_25927


namespace NUMINAMATH_GPT_perpendicular_lines_implies_m_values_l259_25948

-- Define the equations of the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x - (m - 2) * y + 2 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y - 1 = 0

-- Define the condition of perpendicularity between lines l1 and l2
def perpendicular (m : ℝ) : Prop :=
  let a1 := (m + 2) / (m - 2)
  let a2 := -3 / m
  a1 * a2 = -1

-- The statement to be proved
theorem perpendicular_lines_implies_m_values (m : ℝ) :
  (∀ x y : ℝ, l1 m x y ∧ l2 m x y → perpendicular m) → (m = -1 ∨ m = 6) :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_implies_m_values_l259_25948


namespace NUMINAMATH_GPT_probability_of_hitting_exactly_twice_l259_25931

def P_hit_first : ℝ := 0.4
def P_hit_second : ℝ := 0.5
def P_hit_third : ℝ := 0.7

def P_hit_exactly_twice_in_three_shots : ℝ :=
  P_hit_first * P_hit_second * (1 - P_hit_third) +
  (1 - P_hit_first) * P_hit_second * P_hit_third +
  P_hit_first * (1 - P_hit_second) * P_hit_third

theorem probability_of_hitting_exactly_twice :
  P_hit_exactly_twice_in_three_shots = 0.41 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_hitting_exactly_twice_l259_25931


namespace NUMINAMATH_GPT_point_on_line_l259_25982

theorem point_on_line : ∀ (t : ℤ), 
  (∃ m : ℤ, (6 - 2) * m = 20 - 8 ∧ (10 - 6) * m = 32 - 20) →
  (∃ b : ℤ, 8 - 2 * m = b) →
  t = m * 35 + b → t = 107 :=
by
  sorry

end NUMINAMATH_GPT_point_on_line_l259_25982


namespace NUMINAMATH_GPT_perfect_squares_example_l259_25990

def isPerfectSquare (n: ℕ) : Prop := ∃ m: ℕ, m * m = n

theorem perfect_squares_example :
  let a := 10430
  let b := 3970
  let c := 2114
  let d := 386
  isPerfectSquare (a + b) ∧
  isPerfectSquare (a + c) ∧
  isPerfectSquare (a + d) ∧
  isPerfectSquare (b + c) ∧
  isPerfectSquare (b + d) ∧
  isPerfectSquare (c + d) ∧
  isPerfectSquare (a + b + c + d) :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_perfect_squares_example_l259_25990


namespace NUMINAMATH_GPT_Q_has_negative_and_potentially_positive_roots_l259_25954

def Q (x : ℝ) : ℝ := x^7 - 4 * x^6 + 2 * x^5 - 9 * x^3 + 2 * x + 16

theorem Q_has_negative_and_potentially_positive_roots :
  (∃ x : ℝ, x < 0 ∧ Q x = 0) ∧ (∃ y : ℝ, y > 0 ∧ Q y = 0 ∨ ∀ z : ℝ, Q z > 0) :=
by
  sorry

end NUMINAMATH_GPT_Q_has_negative_and_potentially_positive_roots_l259_25954


namespace NUMINAMATH_GPT_time_to_cover_escalator_l259_25970

noncomputable def escalator_speed : ℝ := 8
noncomputable def person_speed : ℝ := 2
noncomputable def escalator_length : ℝ := 160
noncomputable def combined_speed : ℝ := escalator_speed + person_speed

theorem time_to_cover_escalator :
  escalator_length / combined_speed = 16 := by
  sorry

end NUMINAMATH_GPT_time_to_cover_escalator_l259_25970


namespace NUMINAMATH_GPT_remainder_of_3_pow_45_mod_17_l259_25973

theorem remainder_of_3_pow_45_mod_17 : 3^45 % 17 = 15 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_of_3_pow_45_mod_17_l259_25973


namespace NUMINAMATH_GPT_distribute_paper_clips_l259_25967

theorem distribute_paper_clips (total_paper_clips boxes : ℕ) (h_total : total_paper_clips = 81) (h_boxes : boxes = 9) : total_paper_clips / boxes = 9 := by
  sorry

end NUMINAMATH_GPT_distribute_paper_clips_l259_25967


namespace NUMINAMATH_GPT_percentage_reduction_l259_25907

theorem percentage_reduction :
  let original := 243.75
  let reduced := 195
  let percentage := ((original - reduced) / original) * 100
  percentage = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_reduction_l259_25907


namespace NUMINAMATH_GPT_total_diagonals_in_rectangular_prism_l259_25999

-- We define the rectangular prism with its properties
structure RectangularPrism :=
  (vertices : ℕ)
  (edges : ℕ)
  (distinct_dimensions : ℕ)

-- We specify the conditions for the rectangular prism
def givenPrism : RectangularPrism :=
{
  vertices := 8,
  edges := 12,
  distinct_dimensions := 3
}

-- We assert the total number of diagonals in the rectangular prism
theorem total_diagonals_in_rectangular_prism (P : RectangularPrism) : P = givenPrism → ∃ diag, diag = 16 :=
by
  intro h
  have diag := 16
  use diag
  sorry

end NUMINAMATH_GPT_total_diagonals_in_rectangular_prism_l259_25999


namespace NUMINAMATH_GPT_probability_diff_colors_l259_25949

theorem probability_diff_colors (total_balls red_balls white_balls selected_balls : ℕ) 
  (h_total : total_balls = 4)
  (h_red : red_balls = 2)
  (h_white : white_balls = 2)
  (h_selected : selected_balls = 2) :
  (∃ P : ℚ, P = (red_balls.choose (selected_balls / 2) * white_balls.choose (selected_balls / 2)) / total_balls.choose selected_balls ∧ P = 2 / 3) :=
by 
  sorry

end NUMINAMATH_GPT_probability_diff_colors_l259_25949


namespace NUMINAMATH_GPT_factor_expression_value_l259_25929

theorem factor_expression_value :
  ∃ (k m n : ℕ), 
    k > 1 ∧ m > 1 ∧ n > 1 ∧ 
    k ≤ 60 ∧ m ≤ 35 ∧ n ≤ 20 ∧ 
    (2^k + 3^m + k^3 * m^n - n = 43) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_value_l259_25929


namespace NUMINAMATH_GPT_num_consecutive_sets_summing_to_90_l259_25959

-- Define the arithmetic sequence sum properties
theorem num_consecutive_sets_summing_to_90 : 
  ∃ n : ℕ, n ≥ 2 ∧
    ∃ (a : ℕ), 2 * a + n - 1 = 180 / n ∧
      (∃ k : ℕ, 
         k ≥ 2 ∧
         ∃ b : ℕ, 2 * b + k - 1 = 180 / k) ∧
      (∃ m : ℕ, 
         m ≥ 2 ∧ 
         ∃ c : ℕ, 2 * c + m - 1 = 180 / m) ∧
      (n = 3 ∨ n = 5 ∨ n = 9) :=
sorry

end NUMINAMATH_GPT_num_consecutive_sets_summing_to_90_l259_25959


namespace NUMINAMATH_GPT_minimum_value_of_weighted_sum_l259_25941

theorem minimum_value_of_weighted_sum 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) :
  3 * a + 6 * b + 9 * c ≥ 54 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_weighted_sum_l259_25941


namespace NUMINAMATH_GPT_rocket_travel_time_l259_25979

/-- The rocket's distance formula as an arithmetic series sum.
    We need to prove that the rocket reaches 240 km after 15 seconds
    given the conditions in the problem. -/
theorem rocket_travel_time :
  ∃ n : ℕ, (2 * n + (n * (n - 1))) / 2 = 240 ∧ n = 15 :=
by
  sorry

end NUMINAMATH_GPT_rocket_travel_time_l259_25979


namespace NUMINAMATH_GPT_rectangle_to_square_l259_25912

theorem rectangle_to_square (a b : ℝ) (h1 : b / 2 < a) (h2 : a < b) :
  ∃ (r : ℝ), r = Real.sqrt (a * b) ∧ 
    (∃ (cut1 cut2 : ℝ × ℝ), 
      cut1.1 = 0 ∧ cut1.2 = a ∧
      cut2.1 = b - r ∧ cut2.2 = r - a ∧
      ∀ t, t = (a * b) - (r ^ 2)) := sorry

end NUMINAMATH_GPT_rectangle_to_square_l259_25912


namespace NUMINAMATH_GPT_dave_pieces_l259_25966

theorem dave_pieces (boxes_bought : ℕ) (boxes_given : ℕ) (pieces_per_box : ℕ) 
  (h₁ : boxes_bought = 12) (h₂ : boxes_given = 5) (h₃ : pieces_per_box = 3) : 
  boxes_bought - boxes_given * pieces_per_box = 21 :=
by
  sorry

end NUMINAMATH_GPT_dave_pieces_l259_25966


namespace NUMINAMATH_GPT_leaves_dropped_on_fifth_day_l259_25988

theorem leaves_dropped_on_fifth_day 
  (initial_leaves : ℕ)
  (days : ℕ)
  (drops_per_day : ℕ)
  (total_dropped_four_days : ℕ)
  (leaves_dropped_fifth_day : ℕ)
  (h1 : initial_leaves = 340)
  (h2 : days = 4)
  (h3 : drops_per_day = initial_leaves / 10)
  (h4 : total_dropped_four_days = drops_per_day * days)
  (h5 : leaves_dropped_fifth_day = initial_leaves - total_dropped_four_days) :
  leaves_dropped_fifth_day = 204 :=
by
  sorry

end NUMINAMATH_GPT_leaves_dropped_on_fifth_day_l259_25988


namespace NUMINAMATH_GPT_production_difference_l259_25955

theorem production_difference (w t : ℕ) (h1 : w = 3 * t) :
  (w * t) - ((w + 6) * (t - 3)) = 3 * t + 18 :=
by
  sorry

end NUMINAMATH_GPT_production_difference_l259_25955


namespace NUMINAMATH_GPT_ratio_proof_l259_25952

theorem ratio_proof (X: ℕ) (h: 150 * 2 = 300 * X) : X = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_proof_l259_25952


namespace NUMINAMATH_GPT_least_possible_value_l259_25914

theorem least_possible_value (x : ℚ) (h1 : x > 5 / 3) (h2 : x < 9 / 2) : 
  (9 / 2 - 5 / 3 : ℚ) = 17 / 6 :=
by sorry

end NUMINAMATH_GPT_least_possible_value_l259_25914


namespace NUMINAMATH_GPT_function_quadrants_l259_25993

theorem function_quadrants (n : ℝ) (h: ∀ x : ℝ, x ≠ 0 → ((n-1)*x * x > 0)) : n > 1 :=
sorry

end NUMINAMATH_GPT_function_quadrants_l259_25993


namespace NUMINAMATH_GPT_sales_growth_correct_equation_l259_25994

theorem sales_growth_correct_equation (x : ℝ) 
(sales_24th : ℝ) (total_sales_25th_26th : ℝ) 
(h_initial : sales_24th = 5000) (h_total : total_sales_25th_26th = 30000) :
  (5000 * (1 + x)) + (5000 * (1 + x)^2) = 30000 :=
sorry

end NUMINAMATH_GPT_sales_growth_correct_equation_l259_25994


namespace NUMINAMATH_GPT_slope_and_intercept_of_given_function_l259_25969

-- Defining the form of a linear function
def linear_function (m b : ℝ) (x : ℝ) : ℝ := m * x + b

-- The given linear function
def given_function (x : ℝ) : ℝ := 3 * x + 2

-- Stating the problem as a theorem
theorem slope_and_intercept_of_given_function :
  (∀ x : ℝ, given_function x = linear_function 3 2 x) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_slope_and_intercept_of_given_function_l259_25969


namespace NUMINAMATH_GPT_work_duration_l259_25995

variable (a b c : ℕ)
variable (daysTogether daysA daysB daysC : ℕ)

theorem work_duration (H1 : daysTogether = 4)
                      (H2 : daysA = 12)
                      (H3 : daysB = 18)
                      (H4: a = 1 / 12)
                      (H5: b = 1 / 18)
                      (H6: 1 / daysTogether = 1 / daysA + 1 / daysB + 1 / daysC) :
                      daysC = 9 :=
sorry

end NUMINAMATH_GPT_work_duration_l259_25995


namespace NUMINAMATH_GPT_discount_allowed_l259_25950

-- Define the conditions
def CP : ℝ := 100 -- Cost Price (CP) is $100 for simplicity
def MP : ℝ := CP + 0.12 * CP -- Selling price marked 12% above cost price
def Loss : ℝ := 0.01 * CP -- Trader suffers a loss of 1% on CP
def SP : ℝ := CP - Loss -- Selling price after suffering the loss

-- State the equivalent proof problem in Lean
theorem discount_allowed : MP - SP = 13 := by
  sorry

end NUMINAMATH_GPT_discount_allowed_l259_25950


namespace NUMINAMATH_GPT_maximum_height_of_projectile_l259_25992

def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

theorem maximum_height_of_projectile : ∀ t : ℝ, (h t ≤ 116) :=
by sorry

end NUMINAMATH_GPT_maximum_height_of_projectile_l259_25992


namespace NUMINAMATH_GPT_man_age_difference_l259_25944

theorem man_age_difference (S M : ℕ) (h1 : S = 24) (h2 : M + 2 = 2 * (S + 2)) : M - S = 26 := by
  sorry

end NUMINAMATH_GPT_man_age_difference_l259_25944


namespace NUMINAMATH_GPT_problem_condition_l259_25989

noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry
noncomputable def x : ℤ := sorry
noncomputable def a : ℤ := 0
noncomputable def b : ℤ := -m + n

theorem problem_condition 
  (h1 : m ≠ 0)
  (h2 : n ≠ 0)
  (h3 : m ≠ n)
  (h4 : (x + m)^2 - (x^2 + n^2) = (m - n)^2) :
  x = a * m + b * n :=
sorry

end NUMINAMATH_GPT_problem_condition_l259_25989


namespace NUMINAMATH_GPT_fraction_of_students_who_say_dislike_but_actually_like_l259_25925

-- Define the conditions
def total_students : ℕ := 100
def like_dancing : ℕ := total_students / 2
def dislike_dancing : ℕ := total_students / 2

def like_dancing_honest : ℕ := (7 * like_dancing) / 10
def like_dancing_dishonest : ℕ := (3 * like_dancing) / 10

def dislike_dancing_honest : ℕ := (4 * dislike_dancing) / 5
def dislike_dancing_dishonest : ℕ := dislike_dancing / 5

-- Define the proof objective
theorem fraction_of_students_who_say_dislike_but_actually_like :
  (like_dancing_dishonest : ℚ) / (total_students - like_dancing_honest - dislike_dancing_dishonest) = 3 / 11 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_who_say_dislike_but_actually_like_l259_25925


namespace NUMINAMATH_GPT_equal_real_roots_value_l259_25964

theorem equal_real_roots_value (a c : ℝ) (ha : a ≠ 0) (h : 4 - 4 * a * (2 - c) = 0) : (1 / a) + c = 2 := 
by
  sorry

end NUMINAMATH_GPT_equal_real_roots_value_l259_25964


namespace NUMINAMATH_GPT_sphere_radius_l259_25996

theorem sphere_radius 
  (r h1 h2 : ℝ)
  (A1_eq : 5 * π = π * (r^2 - h1^2))
  (A2_eq : 8 * π = π * (r^2 - h2^2))
  (h1_h2_eq : h1 - h2 = 1) : r = 3 :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_l259_25996


namespace NUMINAMATH_GPT_side_length_percentage_error_l259_25926

variable (s s' : Real)
-- Conditions
-- s' = s * 1.06 (measured side length is 6% more than actual side length)
-- (s'^2 - s^2) / s^2 * 100% = 12.36% (percentage error in area)

theorem side_length_percentage_error 
    (h1 : s' = s * 1.06)
    (h2 : (s'^2 - s^2) / s^2 * 100 = 12.36) :
    ((s' - s) / s) * 100 = 6 := 
sorry

end NUMINAMATH_GPT_side_length_percentage_error_l259_25926


namespace NUMINAMATH_GPT_waiting_time_probability_l259_25934

theorem waiting_time_probability :
  (∀ (t : ℝ), 0 ≤ t ∧ t < 30 → (1 / 30) * (if t < 25 then 5 else 5 - (t - 25)) = 1 / 6) :=
by
  sorry

end NUMINAMATH_GPT_waiting_time_probability_l259_25934


namespace NUMINAMATH_GPT_aliyah_more_phones_l259_25923

theorem aliyah_more_phones (vivi_phones : ℕ) (phone_price : ℕ) (total_money : ℕ) (aliyah_more : ℕ) : 
  vivi_phones = 40 → 
  phone_price = 400 → 
  total_money = 36000 → 
  40 + 40 + aliyah_more = total_money / phone_price → 
  aliyah_more = 10 :=
sorry

end NUMINAMATH_GPT_aliyah_more_phones_l259_25923


namespace NUMINAMATH_GPT_value_of_b_l259_25936

theorem value_of_b (b : ℝ) :
  (∀ x : ℝ, 3 * (5 + b * x) = 18 * x + 15) → b = 6 :=
by
  intro h
  -- Proving that b = 6
  sorry

end NUMINAMATH_GPT_value_of_b_l259_25936


namespace NUMINAMATH_GPT_triangle_side_lengths_condition_l259_25904

noncomputable def f (x k : ℝ) : ℝ := (x^2 + k*x + 1) / (x^2 + x + 1)

theorem triangle_side_lengths_condition (k : ℝ) :
  (∀ x1 x2 x3 : ℝ, x1 > 0 → x2 > 0 → x3 > 0 →
    (f x1 k) + (f x2 k) > (f x3 k) ∧ (f x2 k) + (f x3 k) > (f x1 k) ∧ (f x3 k) + (f x1 k) > (f x2 k))
  ↔ (-1/2 ≤ k ∧ k ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_condition_l259_25904


namespace NUMINAMATH_GPT_complex_number_second_quadrant_l259_25942

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i * (1 + i)

-- Define a predicate to determine if a complex number is in the second quadrant
def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- The main statement
theorem complex_number_second_quadrant : is_second_quadrant z := by
  sorry

end NUMINAMATH_GPT_complex_number_second_quadrant_l259_25942


namespace NUMINAMATH_GPT_sin_alpha_neg_point_two_l259_25978

theorem sin_alpha_neg_point_two (a : ℝ) (h : Real.sin (Real.pi + a) = 0.2) : Real.sin a = -0.2 := 
by
  sorry

end NUMINAMATH_GPT_sin_alpha_neg_point_two_l259_25978


namespace NUMINAMATH_GPT_mean_of_jane_scores_l259_25976

theorem mean_of_jane_scores :
  let scores := [96, 95, 90, 87, 91, 75]
  let n := 6
  let sum_scores := 96 + 95 + 90 + 87 + 91 + 75
  let mean := sum_scores / n
  mean = 89 := by
    sorry

end NUMINAMATH_GPT_mean_of_jane_scores_l259_25976


namespace NUMINAMATH_GPT_division_by_fraction_l259_25958

theorem division_by_fraction :
  5 / (8 / 13) = 65 / 8 :=
sorry

end NUMINAMATH_GPT_division_by_fraction_l259_25958


namespace NUMINAMATH_GPT_jackson_volume_discount_l259_25916

-- Given conditions as parameters
def hotTubVolume := 40 -- gallons
def quartsPerGallon := 4 -- quarts per gallon
def bottleVolume := 1 -- quart per bottle
def bottleCost := 50 -- dollars per bottle
def totalSpent := 6400 -- dollars spent by Jackson

-- Calculation related definitions
def totalQuarts := hotTubVolume * quartsPerGallon
def totalBottles := totalQuarts / bottleVolume
def costWithoutDiscount := totalBottles * bottleCost
def discountAmount := costWithoutDiscount - totalSpent
def discountPercentage := (discountAmount / costWithoutDiscount) * 100

-- The proof problem
theorem jackson_volume_discount : discountPercentage = 20 :=
by
  sorry

end NUMINAMATH_GPT_jackson_volume_discount_l259_25916


namespace NUMINAMATH_GPT_prime_triples_l259_25953

open Nat

theorem prime_triples (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) :
    (p ∣ q^r + 1) → (q ∣ r^p + 1) → (r ∣ p^q + 1) → (p, q, r) = (2, 5, 3) ∨ (p, q, r) = (3, 2, 5) ∨ (p, q, r) = (5, 3, 2) :=
  by
  sorry

end NUMINAMATH_GPT_prime_triples_l259_25953


namespace NUMINAMATH_GPT_problem_statement_l259_25933

variable (f : ℕ → ℝ)

theorem problem_statement (hf : ∀ k : ℕ, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2)
  (h : f 4 = 25) : ∀ k : ℕ, k ≥ 4 → f k ≥ k^2 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l259_25933


namespace NUMINAMATH_GPT_rectangle_perimeter_is_3y_l259_25935

noncomputable def congruent_rectangle_perimeter (y : ℝ) (h1 : y > 0) : ℝ :=
  let side_length := 2 * y
  let center_square_side := y
  let width := (side_length - center_square_side) / 2
  let length := center_square_side
  2 * (length + width)

theorem rectangle_perimeter_is_3y (y : ℝ) (h1 : y > 0) :
  congruent_rectangle_perimeter y h1 = 3 * y :=
sorry

end NUMINAMATH_GPT_rectangle_perimeter_is_3y_l259_25935


namespace NUMINAMATH_GPT_black_square_area_l259_25972

-- Define the edge length of the cube
def edge_length := 12

-- Define the total amount of yellow paint available
def yellow_paint_area := 432

-- Define the total surface area of the cube
def total_surface_area := 6 * (edge_length * edge_length)

-- Define the area covered by yellow paint per face
def yellow_per_face := yellow_paint_area / 6

-- Define the area of one face of the cube
def face_area := edge_length * edge_length

-- State the theorem: the area of the black square on each face
theorem black_square_area : (face_area - yellow_per_face) = 72 := by
  sorry

end NUMINAMATH_GPT_black_square_area_l259_25972


namespace NUMINAMATH_GPT_income_of_A_l259_25922

theorem income_of_A (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050) 
  (h2 : (B + C) / 2 = 5250) 
  (h3 : (A + C) / 2 = 4200) : 
  A = 3000 :=
by
  sorry

end NUMINAMATH_GPT_income_of_A_l259_25922


namespace NUMINAMATH_GPT_Joe_total_time_correct_l259_25905

theorem Joe_total_time_correct :
  ∀ (distance : ℝ) (walk_rate : ℝ) (bike_rate : ℝ) (walk_time bike_time : ℝ),
    (walk_time = 9) →
    (bike_rate = 5 * walk_rate) →
    (walk_rate * walk_time = distance / 3) →
    (bike_rate * bike_time = 2 * distance / 3) →
    (walk_time + bike_time = 12.6) := 
by
  intros distance walk_rate bike_rate walk_time bike_time
  intro walk_time_cond
  intro bike_rate_cond
  intro walk_distance_cond
  intro bike_distance_cond
  sorry

end NUMINAMATH_GPT_Joe_total_time_correct_l259_25905


namespace NUMINAMATH_GPT_smallest_positive_integer_ends_in_3_divisible_by_11_l259_25901

theorem smallest_positive_integer_ends_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 10 = 3 ∧ m % 11 = 0) → n ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_ends_in_3_divisible_by_11_l259_25901


namespace NUMINAMATH_GPT_original_length_before_final_cut_l259_25919

-- Defining the initial length of the board
def initial_length : ℕ := 143

-- Defining the length after the first cut
def length_after_first_cut : ℕ := initial_length - 25

-- Defining the length after the final cut
def length_after_final_cut : ℕ := length_after_first_cut - 7

-- Stating the theorem to prove that the original length of the board before cutting the final 7 cm is 125 cm
theorem original_length_before_final_cut : initial_length - 25 + 7 = 125 :=
sorry

end NUMINAMATH_GPT_original_length_before_final_cut_l259_25919


namespace NUMINAMATH_GPT_tangent_line_perpendicular_l259_25965

theorem tangent_line_perpendicular (m : ℝ) :
  (∀ x : ℝ, y = 2 * x^2) →
  (∀ x : ℝ, (4 * x - y + m = 0) ∧ (x + 4 * y - 8 = 0) → 
  (16 + 8 * m = 0)) →
  m = -2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_perpendicular_l259_25965


namespace NUMINAMATH_GPT_exist_m_n_l259_25957

theorem exist_m_n (p : ℕ) [hp : Fact (Nat.Prime p)] (h : 5 < p) :
  ∃ m n : ℕ, (m + n < p ∧ p ∣ (2^m * 3^n - 1)) := sorry

end NUMINAMATH_GPT_exist_m_n_l259_25957


namespace NUMINAMATH_GPT_evaluate_product_eq_l259_25987

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product_eq : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 885735 := 
sorry

end NUMINAMATH_GPT_evaluate_product_eq_l259_25987


namespace NUMINAMATH_GPT_strategy_classification_l259_25932

inductive Player
| A
| B

def A_winning_strategy (n0 : Nat) : Prop :=
  n0 >= 8

def B_winning_strategy (n0 : Nat) : Prop :=
  n0 <= 5

def neither_winning_strategy (n0 : Nat) : Prop :=
  n0 = 6 ∨ n0 = 7

theorem strategy_classification (n0 : Nat) : 
  (A_winning_strategy n0 ∨ B_winning_strategy n0 ∨ neither_winning_strategy n0) := by
    sorry

end NUMINAMATH_GPT_strategy_classification_l259_25932


namespace NUMINAMATH_GPT_shortest_altitude_of_right_triangle_l259_25983

theorem shortest_altitude_of_right_triangle
  (a b c : ℝ)
  (ha : a = 9) 
  (hb : b = 12) 
  (hc : c = 15)
  (ht : a^2 + b^2 = c^2) :
  ∃ h : ℝ, (1 / 2) * c * h = (1 / 2) * a * b ∧ h = 7.2 := by
  sorry

end NUMINAMATH_GPT_shortest_altitude_of_right_triangle_l259_25983


namespace NUMINAMATH_GPT_b_spends_85_percent_l259_25951

-- Definitions based on the given conditions
def combined_salary (a_salary b_salary : ℤ) : Prop := a_salary + b_salary = 3000
def a_salary : ℤ := 2250
def a_spending_ratio : ℝ := 0.95
def a_savings : ℝ := a_salary - a_salary * a_spending_ratio
def b_savings : ℝ := a_savings

-- The goal is to prove that B spends 85% of his salary
theorem b_spends_85_percent (b_salary : ℤ) (b_spending_ratio : ℝ) :
  combined_salary a_salary b_salary →
  b_spending_ratio * b_salary = 0.85 * b_salary :=
  sorry

end NUMINAMATH_GPT_b_spends_85_percent_l259_25951


namespace NUMINAMATH_GPT_problem_x_y_z_l259_25986

theorem problem_x_y_z (x y z : ℕ) (h1 : xy + z = 47) (h2 : yz + x = 47) (h3 : xz + y = 47) : x + y + z = 48 :=
sorry

end NUMINAMATH_GPT_problem_x_y_z_l259_25986


namespace NUMINAMATH_GPT_remainder_when_divided_l259_25909

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 + x^3 + 1

-- The statement to be proved
theorem remainder_when_divided (x : ℝ) : (p 2) = 25 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_l259_25909


namespace NUMINAMATH_GPT_total_balloons_l259_25975

theorem total_balloons:
  ∀ (R1 R2 G1 G2 B1 B2 Y1 Y2 O1 O2: ℕ),
    R1 = 31 →
    R2 = 24 →
    G1 = 15 →
    G2 = 7 →
    B1 = 12 →
    B2 = 14 →
    Y1 = 18 →
    Y2 = 20 →
    O1 = 10 →
    O2 = 16 →
    (R1 + R2 = 55) ∧
    (G1 + G2 = 22) ∧
    (B1 + B2 = 26) ∧
    (Y1 + Y2 = 38) ∧
    (O1 + O2 = 26) :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_balloons_l259_25975


namespace NUMINAMATH_GPT_three_seventy_five_as_fraction_l259_25913

theorem three_seventy_five_as_fraction : (15 : ℚ) / 4 = 3.75 := by
  sorry

end NUMINAMATH_GPT_three_seventy_five_as_fraction_l259_25913


namespace NUMINAMATH_GPT_max_value_of_a_l259_25938

variable {a : ℝ}

theorem max_value_of_a (h : a > 0) : 
  (∀ x : ℝ, x > 0 → (2 * x^2 - a * x + a > 0)) ↔ a ≤ 8 := 
sorry

end NUMINAMATH_GPT_max_value_of_a_l259_25938


namespace NUMINAMATH_GPT_distance_between_centers_l259_25937

theorem distance_between_centers (r1 r2 d x : ℝ) (h1 : r1 = 10) (h2 : r2 = 6) (h3 : d = 30) :
  x = 2 * Real.sqrt 229 := 
sorry

end NUMINAMATH_GPT_distance_between_centers_l259_25937


namespace NUMINAMATH_GPT_heather_total_distance_l259_25921

-- Definitions for distances walked
def distance_car_to_entrance : ℝ := 0.33
def distance_entrance_to_rides : ℝ := 0.33
def distance_rides_to_car : ℝ := 0.08

-- Statement of the problem to be proven
theorem heather_total_distance :
  distance_car_to_entrance + distance_entrance_to_rides + distance_rides_to_car = 0.74 :=
by
  sorry

end NUMINAMATH_GPT_heather_total_distance_l259_25921


namespace NUMINAMATH_GPT_shaded_region_area_l259_25961

structure Point where
  x : ℝ
  y : ℝ

def W : Point := ⟨0, 0⟩
def X : Point := ⟨5, 0⟩
def Y : Point := ⟨5, 2⟩
def Z : Point := ⟨0, 2⟩
def Q : Point := ⟨1, 0⟩
def S : Point := ⟨5, 0.5⟩
def R : Point := ⟨0, 1⟩
def D : Point := ⟨1, 2⟩

def triangle_area (A B C : Point) : ℝ :=
  0.5 * |(A.x * B.y + B.x * C.y + C.x * A.y) - (B.x * A.y + C.x * B.y + A.x * C.y)|

theorem shaded_region_area : triangle_area R D Y = 1 := by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l259_25961


namespace NUMINAMATH_GPT_vann_teeth_cleaning_l259_25974

def numDogsCleaned (D : Nat) : Prop :=
  let dogTeethCount := 42
  let catTeethCount := 30
  let pigTeethCount := 28
  let numCats := 10
  let numPigs := 7
  let totalTeeth := 706
  dogTeethCount * D + catTeethCount * numCats + pigTeethCount * numPigs = totalTeeth

theorem vann_teeth_cleaning : numDogsCleaned 5 :=
by
  sorry

end NUMINAMATH_GPT_vann_teeth_cleaning_l259_25974


namespace NUMINAMATH_GPT_population_ratio_l259_25945

theorem population_ratio
  (P_A P_B P_C P_D P_E P_F : ℕ)
  (h1 : P_A = 8 * P_B)
  (h2 : P_B = 5 * P_C)
  (h3 : P_D = 3 * P_C)
  (h4 : P_D = P_E / 2)
  (h5 : P_F = P_A / 4) :
  P_E / P_B = 6 / 5 := by
    sorry

end NUMINAMATH_GPT_population_ratio_l259_25945


namespace NUMINAMATH_GPT_totalNameLengths_l259_25939

-- Definitions of the lengths of names
def JonathanNameLength := 8 + 10
def YoungerSisterNameLength := 5 + 10
def OlderBrotherNameLength := 6 + 10
def YoungestSisterNameLength := 4 + 15

-- Statement to prove
theorem totalNameLengths :
  JonathanNameLength + YoungerSisterNameLength + OlderBrotherNameLength + YoungestSisterNameLength = 68 :=
by
  sorry -- no proof required

end NUMINAMATH_GPT_totalNameLengths_l259_25939


namespace NUMINAMATH_GPT_percentage_increase_l259_25940

-- Conditions
variables (S_final S_initial : ℝ) (P : ℝ)
def conditions := (S_final = 3135) ∧ (S_initial = 3000) ∧
  (S_final = (S_initial + (P/100) * S_initial) - 0.05 * (S_initial + (P/100) * S_initial))

-- Statement of the problem
theorem percentage_increase (S_final S_initial : ℝ) 
  (cond : conditions S_final S_initial P) : P = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l259_25940


namespace NUMINAMATH_GPT_mark_cans_l259_25977

variable (r j m : ℕ) -- r for Rachel, j for Jaydon, m for Mark

theorem mark_cans (r j m : ℕ) 
  (h1 : j = 5 + 2 * r)
  (h2 : m = 4 * j)
  (h3 : r + j + m = 135) : 
  m = 100 :=
by
  sorry

end NUMINAMATH_GPT_mark_cans_l259_25977


namespace NUMINAMATH_GPT_cost_of_monogramming_each_backpack_l259_25981

def number_of_backpacks : ℕ := 5
def original_price_per_backpack : ℝ := 20.00
def discount_rate : ℝ := 0.20
def total_cost : ℝ := 140.00

theorem cost_of_monogramming_each_backpack : 
  (total_cost - (number_of_backpacks * (original_price_per_backpack * (1 - discount_rate)))) / number_of_backpacks = 12.00 :=
by
  sorry 

end NUMINAMATH_GPT_cost_of_monogramming_each_backpack_l259_25981


namespace NUMINAMATH_GPT_joe_anne_bill_difference_l259_25971

theorem joe_anne_bill_difference (m j a : ℝ) 
  (hm : (15 / 100) * m = 3) 
  (hj : (10 / 100) * j = 2) 
  (ha : (20 / 100) * a = 3) : 
  j - a = 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_joe_anne_bill_difference_l259_25971


namespace NUMINAMATH_GPT_required_height_for_roller_coaster_l259_25930

-- Definitions based on conditions from the problem
def initial_height : ℕ := 48
def natural_growth_rate_per_month : ℚ := 1 / 3
def upside_down_growth_rate_per_hour : ℚ := 1 / 12
def hours_per_month_hanging_upside_down : ℕ := 2
def months_in_a_year : ℕ := 12

-- Calculations needed for the proof
def annual_natural_growth := natural_growth_rate_per_month * months_in_a_year
def annual_upside_down_growth := (upside_down_growth_rate_per_hour * hours_per_month_hanging_upside_down) * months_in_a_year
def total_annual_growth := annual_natural_growth + annual_upside_down_growth
def height_next_year := initial_height + total_annual_growth

-- Statement of the required height for the roller coaster
theorem required_height_for_roller_coaster : height_next_year = 54 :=
by
  sorry

end NUMINAMATH_GPT_required_height_for_roller_coaster_l259_25930


namespace NUMINAMATH_GPT_total_pens_is_50_l259_25980

theorem total_pens_is_50
  (red : ℕ) (black : ℕ) (blue : ℕ) (green : ℕ) (purple : ℕ) (total : ℕ)
  (h1 : red = 8)
  (h2 : black = 3 / 2 * red)
  (h3 : blue = black + 5 ∧ blue = 1 / 5 * total)
  (h4 : green = blue / 2)
  (h5 : purple = 5)
  : total = red + black + blue + green + purple := sorry

end NUMINAMATH_GPT_total_pens_is_50_l259_25980


namespace NUMINAMATH_GPT_insurance_compensation_l259_25991

/-- Given the actual damage amount and the deductible percentage, 
we can compute the amount of insurance compensation. -/
theorem insurance_compensation : 
  ∀ (damage_amount : ℕ) (deductible_percent : ℕ), 
  damage_amount = 300000 → 
  deductible_percent = 1 →
  (damage_amount - (damage_amount * deductible_percent / 100)) = 297000 :=
by
  intros damage_amount deductible_percent h_damage h_deductible
  sorry

end NUMINAMATH_GPT_insurance_compensation_l259_25991
