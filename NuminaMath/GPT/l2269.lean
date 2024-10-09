import Mathlib

namespace cost_price_of_article_l2269_226948

theorem cost_price_of_article (C SP1 SP2 G1 G2 : ℝ) 
  (h_SP1 : SP1 = 160) 
  (h_SP2 : SP2 = 220) 
  (h_gain_relation : G2 = 1.05 * G1) 
  (h_G1 : G1 = SP1 - C) 
  (h_G2 : G2 = SP2 - C) : C = 1040 :=
by
  sorry

end cost_price_of_article_l2269_226948


namespace std_dev_samples_l2269_226966

def sample_A := [82, 84, 84, 86, 86, 86, 88, 88, 88, 88]
def sample_B := [84, 86, 86, 88, 88, 88, 90, 90, 90, 90]

noncomputable def std_dev (l : List ℕ) :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  let variance := (l.map (λ x => (x - mean) * (x - mean))).sum / n
  variance.sqrt

theorem std_dev_samples :
  std_dev sample_A = std_dev sample_B := 
sorry

end std_dev_samples_l2269_226966


namespace probability_of_five_3s_is_099_l2269_226998

-- Define conditions
def number_of_dice : ℕ := 15
def rolled_value : ℕ := 3
def probability_of_3 : ℚ := 1 / 8
def number_of_successes : ℕ := 5
def probability_of_not_3 : ℚ := 7 / 8

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability calculation
def probability_exactly_five_3s : ℚ :=
  binomial_coefficient number_of_dice number_of_successes *
  probability_of_3 ^ number_of_successes *
  probability_of_not_3 ^ (number_of_dice - number_of_successes)

theorem probability_of_five_3s_is_099 :
  probability_exactly_five_3s = 0.099 := by
  sorry -- Proof to be filled in later

end probability_of_five_3s_is_099_l2269_226998


namespace coefficient_m5_n5_in_expansion_l2269_226976

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Goal: prove the coefficient of m^5 n^5 in the expansion of (m+n)^{10} is 252
theorem coefficient_m5_n5_in_expansion : binomial 10 5 = 252 :=
by
  sorry

end coefficient_m5_n5_in_expansion_l2269_226976


namespace greyhound_catches_hare_l2269_226911

theorem greyhound_catches_hare {a b : ℝ} (h_speed : b < a) : ∃ t : ℝ, ∀ s : ℝ, ∃ n : ℕ, (n * t * (a - b)) > s + t * (a + b) :=
by
  sorry

end greyhound_catches_hare_l2269_226911


namespace range_of_a_fall_within_D_l2269_226917

-- Define the conditions
variable (a : ℝ) (c : ℝ)
axiom A_through : c = 9
axiom D_through : a < 0 ∧ (6, 7) ∈ { (x, y) | y = a * x ^ 2 + c }

-- Prove the range of a given the conditions
theorem range_of_a : -1/4 < a ∧ a < -1/18 := sorry

-- Define the additional condition for point P
axiom P_through : (2, 8.1) ∈ { (x, y) | y = a * x ^ 2 + c }

-- Prove that the object can fall within interval D when passing through point P
theorem fall_within_D : a = -9/40 ∧ -1/4 < a ∧ a < -1/18 := sorry

end range_of_a_fall_within_D_l2269_226917


namespace unique_real_root_iff_a_eq_3_l2269_226981

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * abs x + a^2 - 9

theorem unique_real_root_iff_a_eq_3 {a : ℝ} (hu : ∃! x : ℝ, f x a = 0) : a = 3 :=
sorry

end unique_real_root_iff_a_eq_3_l2269_226981


namespace find_sum_of_abc_l2269_226988

noncomputable def m (a b c : ℕ) : ℝ := a - b * Real.sqrt c

theorem find_sum_of_abc (a b c : ℕ) (ha : ¬ (c % 2 = 0) ∧ ∀ p : ℕ, Prime p → ¬ p * p ∣ c) 
  (hprob : ((30 - m a b c) ^ 2 / 30 ^ 2 = 0.75)) : a + b + c = 48 := 
by
  sorry

end find_sum_of_abc_l2269_226988


namespace intersecting_lines_l2269_226939

theorem intersecting_lines (m n : ℝ) : 
  (∀ x y : ℝ, y = x / 2 + n → y = mx - 1 → (x = 1 ∧ y = -2)) → 
  m = -1 ∧ n = -5 / 2 :=
by
  sorry

end intersecting_lines_l2269_226939


namespace number_of_pieces_correct_l2269_226975

-- Define the dimensions of the pan
def pan_length : ℕ := 30
def pan_width : ℕ := 24

-- Define the dimensions of each piece of brownie
def piece_length : ℕ := 3
def piece_width : ℕ := 2

-- Calculate the area of the pan
def pan_area : ℕ := pan_length * pan_width

-- Calculate the area of each piece of brownie
def piece_area : ℕ := piece_length * piece_width

-- The proof problem statement
theorem number_of_pieces_correct : (pan_area / piece_area) = 120 :=
by sorry

end number_of_pieces_correct_l2269_226975


namespace intersection_P_Q_l2269_226970

def P (y : ℝ) : Prop :=
  ∃ (x : ℝ), y = -x^2 + 2

def Q (y : ℝ) : Prop :=
  ∃ (x : ℝ), y = x

theorem intersection_P_Q :
  { y : ℝ | P y } ∩ { y : ℝ | Q y } = { y : ℝ | y ≤ 2 } :=
by
  sorry

end intersection_P_Q_l2269_226970


namespace transmit_data_time_l2269_226960

def total_chunks (blocks: ℕ) (chunks_per_block: ℕ) : ℕ := blocks * chunks_per_block

def transmit_time (total_chunks: ℕ) (chunks_per_second: ℕ) : ℕ := total_chunks / chunks_per_second

def time_in_minutes (transmit_time_seconds: ℕ) : ℕ := transmit_time_seconds / 60

theorem transmit_data_time :
  ∀ (blocks chunks_per_block chunks_per_second : ℕ),
    blocks = 150 →
    chunks_per_block = 256 →
    chunks_per_second = 200 →
    time_in_minutes (transmit_time (total_chunks blocks chunks_per_block) chunks_per_second) = 3 := by
  intros
  sorry

end transmit_data_time_l2269_226960


namespace benny_start_cards_l2269_226994

--- Benny bought 4 new cards before the dog ate half of his collection.
def new_cards : Int := 4

--- The remaining cards after the dog ate half of the collection is 34.
def remaining_cards : Int := 34

--- The total number of cards Benny had before adding the new cards and the dog ate half.
def total_before_eating := remaining_cards * 2

theorem benny_start_cards : total_before_eating - new_cards = 64 :=
sorry

end benny_start_cards_l2269_226994


namespace min_value_expression_l2269_226919

theorem min_value_expression (x y k : ℝ) (hk : 1 < k) (hx : k < x) (hy : k < y) : 
  (∀ x y, x > k → y > k → (∃ m, (m ≤ (x^2 / (y - k) + y^2 / (x - k)))) ∧ (m = 8 * k)) := sorry

end min_value_expression_l2269_226919


namespace homework_total_time_l2269_226969

theorem homework_total_time :
  ∀ (j g p : ℕ),
  j = 18 →
  g = j - 6 →
  p = 2 * g - 4 →
  j + g + p = 50 :=
by
  intros j g p h1 h2 h3
  sorry

end homework_total_time_l2269_226969


namespace probability_diagonals_intersect_inside_decagon_l2269_226968

/-- Two diagonals of a regular decagon are chosen. 
  What is the probability that their intersection lies inside the decagon?
-/
theorem probability_diagonals_intersect_inside_decagon : 
  let num_diagonals := 35
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210
  let probability := num_intersecting_pairs / num_pairs
  probability = 42 / 119 :=
by
  -- Definitions based on the conditions
  let num_diagonals := (10 * (10 - 3)) / 2
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210

  -- Simplified probability
  let probability := num_intersecting_pairs / num_pairs

  -- Sorry used to skip the proof
  sorry

end probability_diagonals_intersect_inside_decagon_l2269_226968


namespace valentines_left_l2269_226963

theorem valentines_left (initial_valentines given_away : ℕ) (h_initial : initial_valentines = 30) (h_given : given_away = 8) :
  initial_valentines - given_away = 22 :=
by {
  sorry
}

end valentines_left_l2269_226963


namespace prime_solution_exists_l2269_226945

theorem prime_solution_exists (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  p^2 + 1 = 74 * (q^2 + r^2) → (p = 31 ∧ q = 2 ∧ r = 3) :=
by
  sorry

end prime_solution_exists_l2269_226945


namespace no_n_geq_2_makes_10101n_prime_l2269_226972

theorem no_n_geq_2_makes_10101n_prime : ∀ n : ℕ, n ≥ 2 → ¬ Prime (n^4 + n^2 + 1) :=
by
  sorry

end no_n_geq_2_makes_10101n_prime_l2269_226972


namespace linear_function_not_in_second_quadrant_l2269_226944

-- Define the linear function y = x - 1.
def linear_function (x : ℝ) : ℝ := x - 1

-- Define the condition for a point to be in the second quadrant.
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- State that for any point (x, y) in the second quadrant, it does not satisfy y = x - 1.
theorem linear_function_not_in_second_quadrant {x y : ℝ} (h : in_second_quadrant x y) : linear_function x ≠ y :=
sorry

end linear_function_not_in_second_quadrant_l2269_226944


namespace C_and_D_complete_work_together_in_2_86_days_l2269_226973

def work_rate (days : ℕ) : ℚ := 1 / days

def A_rate := work_rate 4
def B_rate := work_rate 10
def D_rate := work_rate 5

noncomputable def C_rate : ℚ :=
  let combined_A_B_C_rate := A_rate + B_rate + (1 / (2 : ℚ))
  let C_rate := 1 / (20 / 3 : ℚ)  -- Solved from the equations provided in the solution
  C_rate

noncomputable def combined_C_D_rate := C_rate + D_rate

noncomputable def days_for_C_and_D_to_complete_work : ℚ :=
  1 / combined_C_D_rate

theorem C_and_D_complete_work_together_in_2_86_days :
  abs (days_for_C_and_D_to_complete_work - 2.86) < 0.01 := sorry

end C_and_D_complete_work_together_in_2_86_days_l2269_226973


namespace ratio_of_larger_to_smaller_l2269_226926

theorem ratio_of_larger_to_smaller (S L k : ℕ) 
  (hS : S = 32)
  (h_sum : S + L = 96)
  (h_multiple : L = k * S) : L / S = 2 :=
by
  sorry

end ratio_of_larger_to_smaller_l2269_226926


namespace percentage_of_water_in_dried_grapes_l2269_226982

theorem percentage_of_water_in_dried_grapes 
  (weight_fresh : ℝ) 
  (weight_dried : ℝ) 
  (percentage_water_fresh : ℝ) 
  (solid_weight : ℝ)
  (water_weight_dried : ℝ) 
  (percentage_water_dried : ℝ) 
  (H1 : weight_fresh = 30) 
  (H2 : weight_dried = 15) 
  (H3 : percentage_water_fresh = 0.60) 
  (H4 : solid_weight = weight_fresh * (1 - percentage_water_fresh)) 
  (H5 : water_weight_dried = weight_dried - solid_weight) 
  (H6 : percentage_water_dried = (water_weight_dried / weight_dried) * 100) 
  : percentage_water_dried = 20 := 
  by { sorry }

end percentage_of_water_in_dried_grapes_l2269_226982


namespace circus_dogs_ratio_l2269_226996

theorem circus_dogs_ratio :
  ∀ (x y : ℕ), 
  (x + y = 12) → (2 * x + 4 * y = 36) → (x = y) → x / y = 1 :=
by
  intros x y h1 h2 h3
  sorry

end circus_dogs_ratio_l2269_226996


namespace find_n_22_or_23_l2269_226967

theorem find_n_22_or_23 (n : ℕ) : 
  (∃ (sol_count : ℕ), sol_count = 30 ∧ (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 2 * y + 4 * z = n)) → 
  (n = 22 ∨ n = 23) := 
sorry

end find_n_22_or_23_l2269_226967


namespace find_smallest_m_l2269_226947

theorem find_smallest_m : ∃ m : ℕ, m > 0 ∧ (790 * m ≡ 1430 * m [MOD 30]) ∧ ∀ n : ℕ, n > 0 ∧ (790 * n ≡ 1430 * n [MOD 30]) → m ≤ n :=
by
  sorry

end find_smallest_m_l2269_226947


namespace contact_prob_correct_l2269_226984

-- Define the conditions.
def m : ℕ := 6
def n : ℕ := 7
variable (p : ℝ)

-- Define the probability computation.
def prob_contact : ℝ := 1 - (1 - p)^(m * n)

-- Formal statement of the problem.
theorem contact_prob_correct : prob_contact p = 1 - (1 - p)^42 := by
  sorry

end contact_prob_correct_l2269_226984


namespace problem_1_problem_2_problem_3_l2269_226952

noncomputable def f (a x : ℝ) : ℝ := a^(x-1)

theorem problem_1 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  f a 3 = 4 → a = 2 :=
sorry

theorem problem_2 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  f a (Real.log a) = 100 → (a = 100 ∨ a = 1 / 10) :=
sorry

theorem problem_3 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  (a > 1 → f a (Real.log (1 / 100)) > f a (-2.1)) ∧
  (0 < a ∧ a < 1 → f a (Real.log (1 / 100)) < f a (-2.1)) :=
sorry

end problem_1_problem_2_problem_3_l2269_226952


namespace Emily_used_10_dimes_l2269_226956

theorem Emily_used_10_dimes
  (p n d : ℕ)
  (h1 : p + n + d = 50)
  (h2 : p + 5 * n + 10 * d = 200) :
  d = 10 := by
  sorry

end Emily_used_10_dimes_l2269_226956


namespace computer_additions_per_hour_l2269_226932

theorem computer_additions_per_hour : 
  ∀ (initial_rate : ℕ) (increase_rate: ℚ) (intervals_per_hour : ℕ),
  initial_rate = 12000 → 
  increase_rate = 0.05 → 
  intervals_per_hour = 4 → 
  (12000 * 900) + (12000 * 1.05 * 900) + (12000 * 1.05^2 * 900) + (12000 * 1.05^3 * 900) = 46549350 := 
by
  intros initial_rate increase_rate intervals_per_hour h1 h2 h3
  have h4 : initial_rate = 12000 := h1
  have h5 : increase_rate = 0.05 := h2
  have h6 : intervals_per_hour = 4 := h3
  sorry

end computer_additions_per_hour_l2269_226932


namespace find_first_term_geometric_series_l2269_226936

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l2269_226936


namespace final_output_value_of_m_l2269_226905

variables (a b m : ℕ)

theorem final_output_value_of_m (h₁ : a = 2) (h₂ : b = 3) (program_logic : (a > b → m = a) ∧ (a ≤ b → m = b)) :
  m = 3 :=
by
  have h₃ : a ≤ b := by
    rw [h₁, h₂]
    exact le_of_lt (by norm_num)
  exact (program_logic.right h₃).trans h₂

end final_output_value_of_m_l2269_226905


namespace range_of_smallest_side_l2269_226997

theorem range_of_smallest_side 
  (c : ℝ) -- the perimeter of the triangle
  (a : ℝ) (b : ℝ) (A : ℝ)  -- three sides of the triangle
  (ha : 0 < a) 
  (hb : b = 2 * a) 
  (hc : a + b + A = c)
  (htriangle : a + b > A ∧ a + A > b ∧ b + A > a) 
  : 
  ∃ (l u : ℝ), l = c / 6 ∧ u = c / 4 ∧ l < a ∧ a < u 
:= sorry

end range_of_smallest_side_l2269_226997


namespace shorter_piece_is_28_l2269_226986

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x + (x + 12) = 68 → x = 28

theorem shorter_piece_is_28 (x : ℕ) : shorter_piece_length x :=
by
  intro h
  have h1 : 2 * x + 12 = 68 := by linarith
  have h2 : 2 * x = 56 := by linarith
  have h3 : x = 28 := by linarith
  exact h3

end shorter_piece_is_28_l2269_226986


namespace supercomputer_transformation_stops_l2269_226903

def transformation_rule (n : ℕ) : ℕ :=
  let A : ℕ := n / 100
  let B : ℕ := n % 100
  2 * A + 8 * B

theorem supercomputer_transformation_stops (n : ℕ) :
  let start := (10^900 - 1) / 9 -- 111...111 with 900 ones
  (n = start) → (∀ m, transformation_rule m < 100 → false) :=
by
  sorry

end supercomputer_transformation_stops_l2269_226903


namespace max_path_length_correct_l2269_226983

noncomputable def maxFlyPathLength : ℝ :=
  2 * Real.sqrt 2 + Real.sqrt 6 + 6

theorem max_path_length_correct :
  ∀ (fly_path_length : ℝ), (fly_path_length = maxFlyPathLength) :=
by
  intro fly_path_length
  sorry

end max_path_length_correct_l2269_226983


namespace min_value_of_x_sq_plus_6x_l2269_226924

theorem min_value_of_x_sq_plus_6x : ∃ x : ℝ, ∀ y : ℝ, y^2 + 6*y ≥ -9 :=
by
  sorry

end min_value_of_x_sq_plus_6x_l2269_226924


namespace fraction_identity_l2269_226935

theorem fraction_identity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 = b^2 + b * c) (h2 : b^2 = c^2 + a * c) : 
  (1 / c) = (1 / a) + (1 / b) :=
by 
  sorry

end fraction_identity_l2269_226935


namespace geometric_sequence_sum_l2269_226906

theorem geometric_sequence_sum (a : ℕ → ℝ) (S_n : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n+1) = a n * q) → -- geometric sequence condition
  a 2 = 6 → -- first condition
  6 * a 1 + a 3 = 30 → -- second condition
  (∀ n, S_n n = (if q = 2 then 3*(2^n - 1) else if q = 3 then 3^n - 1 else 0)) :=
by intros
   sorry

end geometric_sequence_sum_l2269_226906


namespace minimum_value_of_expression_l2269_226916

theorem minimum_value_of_expression 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : 3 * a + 4 * b + 2 * c = 3) : 
  (1 / (2 * a + b) + 1 / (a + 3 * c) + 1 / (4 * b + c)) = 1.5 :=
sorry

end minimum_value_of_expression_l2269_226916


namespace polar_to_cartesian_conversion_l2269_226995

noncomputable def polarToCartesian (ρ θ : ℝ) : ℝ × ℝ :=
  let x := ρ * Real.cos θ
  let y := ρ * Real.sin θ
  (x, y)

theorem polar_to_cartesian_conversion :
  polarToCartesian 4 (Real.pi / 3) = (2, 2 * Real.sqrt 3) :=
by
  sorry

end polar_to_cartesian_conversion_l2269_226995


namespace temperature_difference_l2269_226943

-- Define variables for the highest and lowest temperatures.
def highest_temp : ℤ := 18
def lowest_temp : ℤ := -2

-- Define the statement for the maximum temperature difference.
theorem temperature_difference : 
  highest_temp - lowest_temp = 20 := 
by 
  sorry

end temperature_difference_l2269_226943


namespace product_of_a_and_c_l2269_226953

theorem product_of_a_and_c (a b c : ℝ) (h1 : a + b + c = 100) (h2 : a - b = 20) (h3 : b - c = 30) : a * c = 378.07 :=
by
  sorry

end product_of_a_and_c_l2269_226953


namespace triangle_least_perimeter_l2269_226993

theorem triangle_least_perimeter (x : ℤ) (h1 : x + 27 > 34) (h2 : 34 + 27 > x) (h3 : x + 34 > 27) : 27 + 34 + x ≥ 69 :=
by
  have h1' : x > 7 := by linarith
  sorry

end triangle_least_perimeter_l2269_226993


namespace compare_f_ln_l2269_226912

variable {f : ℝ → ℝ}

theorem compare_f_ln (h : ∀ x : ℝ, deriv f x > f x) : 3 * f (Real.log 2) < 2 * f (Real.log 3) :=
by
  sorry

end compare_f_ln_l2269_226912


namespace fraction_sum_squares_eq_sixteen_l2269_226929

variables (x a y b z c : ℝ)

theorem fraction_sum_squares_eq_sixteen
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2 + y^2 / b^2 + z^2 / c^2) = 16 := 
sorry

end fraction_sum_squares_eq_sixteen_l2269_226929


namespace average_screen_time_per_player_l2269_226971

def video_point_guard : ℕ := 130
def video_shooting_guard : ℕ := 145
def video_small_forward : ℕ := 85
def video_power_forward : ℕ := 60
def video_center : ℕ := 180
def total_video_time : ℕ := 
  video_point_guard + video_shooting_guard + video_small_forward + video_power_forward + video_center
def total_video_time_minutes : ℕ := total_video_time / 60
def number_of_players : ℕ := 5

theorem average_screen_time_per_player : total_video_time_minutes / number_of_players = 2 :=
  sorry

end average_screen_time_per_player_l2269_226971


namespace best_fit_model_l2269_226901

theorem best_fit_model
  (R2_M1 R2_M2 R2_M3 R2_M4 : ℝ)
  (h1 : R2_M1 = 0.78)
  (h2 : R2_M2 = 0.85)
  (h3 : R2_M3 = 0.61)
  (h4 : R2_M4 = 0.31) :
  ∀ i, (i = 2 ∧ R2_M2 ≥ R2_M1 ∧ R2_M2 ≥ R2_M3 ∧ R2_M2 ≥ R2_M4) := 
sorry

end best_fit_model_l2269_226901


namespace tshirt_cost_l2269_226965

theorem tshirt_cost (initial_amount sweater_cost shoes_cost amount_left spent_on_tshirt : ℕ) 
  (h_initial : initial_amount = 91) 
  (h_sweater : sweater_cost = 24) 
  (h_shoes : shoes_cost = 11) 
  (h_left : amount_left = 50)
  (h_spent : spent_on_tshirt = initial_amount - amount_left - sweater_cost - shoes_cost) :
  spent_on_tshirt = 6 :=
sorry

end tshirt_cost_l2269_226965


namespace solve_inequality_l2269_226946

theorem solve_inequality (x : ℝ) : 2 * x^2 - x - 1 > 0 ↔ x < -1/2 ∨ x > 1 :=
by
  sorry

end solve_inequality_l2269_226946


namespace indigo_restaurant_average_rating_l2269_226955

theorem indigo_restaurant_average_rating :
  let n_5stars := 6
  let n_4stars := 7
  let n_3stars := 4
  let n_2stars := 1
  let total_reviews := 18
  let total_stars := n_5stars * 5 + n_4stars * 4 + n_3stars * 3 + n_2stars * 2
  (total_stars / total_reviews : ℝ) = 4 :=
by
  sorry

end indigo_restaurant_average_rating_l2269_226955


namespace smallest_value_of_k_l2269_226962

theorem smallest_value_of_k (k : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + k = 5) ↔ k >= 9 := 
sorry

end smallest_value_of_k_l2269_226962


namespace value_of_expression_l2269_226941

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end value_of_expression_l2269_226941


namespace range_of_m_l2269_226900

theorem range_of_m (α β m : ℝ)
  (h1 : 0 < α ∧ α < 1)
  (h2 : 1 < β ∧ β < 2)
  (h3 : ∀ x, x^2 - m * x + 1 = 0 ↔ (x = α ∨ x = β)) :
  2 < m ∧ m < 5 / 2 :=
sorry

end range_of_m_l2269_226900


namespace job_completion_time_l2269_226928

theorem job_completion_time
  (A C : ℝ)
  (A_rate : A = 1 / 6)
  (C_rate : C = 1 / 12)
  (B_share : 390 / 1170 = 1 / 3) :
  ∃ B : ℝ, B = 1 / 8 ∧ (B * 8 = 1) :=
by
  -- Proof omitted
  sorry

end job_completion_time_l2269_226928


namespace dividend_50100_l2269_226978

theorem dividend_50100 (D Q R : ℕ) (h1 : D = 20 * Q) (h2 : D = 10 * R) (h3 : R = 100) : 
    D * Q + R = 50100 := by
  sorry

end dividend_50100_l2269_226978


namespace leaves_falling_every_day_l2269_226907

-- Definitions of the conditions
def roof_capacity := 500 -- in pounds
def leaves_per_pound := 1000 -- number of leaves per pound
def collapse_time := 5000 -- in days

-- Function to calculate the number of leaves falling each day
def leaves_per_day (roof_capacity : Nat) (leaves_per_pound : Nat) (collapse_time : Nat) : Nat :=
  (roof_capacity * leaves_per_pound) / collapse_time

-- Theorem stating the expected result
theorem leaves_falling_every_day :
  leaves_per_day roof_capacity leaves_per_pound collapse_time = 100 :=
by
  sorry

end leaves_falling_every_day_l2269_226907


namespace price_of_cheaper_feed_l2269_226914

theorem price_of_cheaper_feed 
  (W_total : ℝ) (P_total : ℝ) (E : ℝ) (W_C : ℝ) 
  (H1 : W_total = 27) 
  (H2 : P_total = 0.26)
  (H3 : E = 0.36)
  (H4 : W_C = 14.2105263158) 
  : (W_total * P_total = W_C * C + (W_total - W_C) * E) → 
    (C = 0.17) :=
by {
  sorry
}

end price_of_cheaper_feed_l2269_226914


namespace largest_of_consecutive_odds_l2269_226927

-- Defining the six consecutive odd numbers
def consecutive_odd_numbers (a b c d e f : ℕ) : Prop :=
  (a = b + 2) ∧ (b = c + 2) ∧ (c = d + 2) ∧ (d = e + 2) ∧ (e = f + 2)

-- Defining the product condition
def product_of_odds (a b c d e f : ℕ) : Prop :=
  a * b * c * d * e * f = 135135

-- Defining the odd numbers greater than zero
def positive_odds (a b c d e f : ℕ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0) ∧
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1)

-- Theorem
theorem largest_of_consecutive_odds (a b c d e f : ℕ) 
  (h1 : consecutive_odd_numbers a b c d e f)
  (h2 : product_of_odds a b c d e f)
  (h3 : positive_odds a b c d e f) : 
  a = 13 :=
sorry

end largest_of_consecutive_odds_l2269_226927


namespace exponentiation_rule_proof_l2269_226940

-- Definitions based on conditions
def x : ℕ := 3
def a : ℕ := 4
def b : ℕ := 2

-- The rule that relates the exponents
def rule (x a b : ℕ) : ℕ := x^(a * b)

-- Proposition that we need to prove
theorem exponentiation_rule_proof : rule x a b = 6561 :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end exponentiation_rule_proof_l2269_226940


namespace polar_coordinates_equivalence_l2269_226937

theorem polar_coordinates_equivalence :
  ∀ (ρ θ1 θ2 : ℝ), θ1 = π / 3 ∧ θ2 = -5 * π / 3 →
  (ρ = 5) → 
  (ρ * Real.cos θ1 = ρ * Real.cos θ2 ∧ ρ * Real.sin θ1 = ρ * Real.sin θ2) :=
by
  sorry

end polar_coordinates_equivalence_l2269_226937


namespace ned_did_not_wash_10_items_l2269_226954

theorem ned_did_not_wash_10_items :
  let short_sleeve_shirts := 9
  let long_sleeve_shirts := 21
  let pairs_of_pants := 15
  let jackets := 8
  let total_items := short_sleeve_shirts + long_sleeve_shirts + pairs_of_pants + jackets
  let washed_items := 43
  let not_washed_Items := total_items - washed_items
  not_washed_Items = 10 := by
sorry

end ned_did_not_wash_10_items_l2269_226954


namespace kilometers_to_chains_l2269_226989

theorem kilometers_to_chains :
  (1 * 10 * 50 = 500) :=
by
  sorry

end kilometers_to_chains_l2269_226989


namespace solution_set_of_inequality_l2269_226933

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, true
axiom f_zero_eq : f 0 = 2
axiom f_derivative_ineq : ∀ x : ℝ, f x + (deriv f x) > 1

theorem solution_set_of_inequality : { x : ℝ | e^x * f x > e^x + 1 } = { x | x > 0 } :=
by
  sorry

end solution_set_of_inequality_l2269_226933


namespace arithmetic_seq_question_l2269_226957

theorem arithmetic_seq_question (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a n = a 1 + (n - 1) * d)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 := 
sorry

end arithmetic_seq_question_l2269_226957


namespace cosine_difference_l2269_226980

theorem cosine_difference (A B : ℝ) (h1 : Real.sin A + Real.sin B = 3/2) (h2 : Real.cos A + Real.cos B = 2) :
  Real.cos (A - B) = 17 / 8 :=
by
  sorry

end cosine_difference_l2269_226980


namespace f_eq_f_inv_implies_x_eq_0_l2269_226951

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1
noncomputable def f_inv (x : ℝ) : ℝ := (-1 + Real.sqrt (3 * x + 4)) / 3

theorem f_eq_f_inv_implies_x_eq_0 (x : ℝ) : f x = f_inv x → x = 0 :=
by
  sorry

end f_eq_f_inv_implies_x_eq_0_l2269_226951


namespace value_of_PQRS_l2269_226999

theorem value_of_PQRS : 
  let P := 2 * (Real.sqrt 2010 + Real.sqrt 2011)
  let Q := 3 * (-Real.sqrt 2010 - Real.sqrt 2011)
  let R := 2 * (Real.sqrt 2010 - Real.sqrt 2011)
  let S := 3 * (Real.sqrt 2011 - Real.sqrt 2010)
  P * Q * R * S = -36 :=
by
  sorry

end value_of_PQRS_l2269_226999


namespace fruit_bowl_remaining_l2269_226991

-- Define the initial conditions
def oranges : Nat := 3
def lemons : Nat := 6
def fruits_eaten : Nat := 3

-- Define the total count of fruits initially
def total_fruits : Nat := oranges + lemons

-- The goal is to prove remaining fruits == 6
theorem fruit_bowl_remaining : total_fruits - fruits_eaten = 6 := by
  sorry

end fruit_bowl_remaining_l2269_226991


namespace sin_alpha_minus_beta_l2269_226913

theorem sin_alpha_minus_beta (α β : Real) 
  (h1 : Real.sin α = 12 / 13) 
  (h2 : Real.cos β = 4 / 5)
  (hα : π / 2 ≤ α ∧ α ≤ π)
  (hβ : -π / 2 ≤ β ∧ β ≤ 0) :
  Real.sin (α - β) = 33 / 65 := 
sorry

end sin_alpha_minus_beta_l2269_226913


namespace find_m_l2269_226974

def is_good (n : ℤ) : Prop :=
  ¬ (∃ k : ℤ, |n| = k^2)

theorem find_m (m : ℤ) : (m % 4 = 3) → 
  (∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_good a ∧ is_good b ∧ is_good c ∧ (a * b * c) % 2 = 1 ∧ a + b + c = m) :=
sorry

end find_m_l2269_226974


namespace value_of_p10_l2269_226990

def p (d e f x : ℝ) : ℝ := d * x^2 + e * x + f

theorem value_of_p10 (d e f : ℝ) 
  (h1 : p d e f 3 = p d e f 4)
  (h2 : p d e f 2 = p d e f 5)
  (h3 : p d e f 0 = 2) :
  p d e f 10 = 2 :=
by
  sorry

end value_of_p10_l2269_226990


namespace complete_square_result_l2269_226987

theorem complete_square_result (x : ℝ) :
  (x^2 - 4 * x - 3 = 0) → ((x - 2) ^ 2 = 7) :=
by sorry

end complete_square_result_l2269_226987


namespace num_valid_N_l2269_226958

theorem num_valid_N : 
  ∃ n : ℕ, n = 4 ∧ ∀ (N : ℕ), (N > 0) → (∃ k : ℕ, 60 = (N+3) * k ∧ k % 2 = 0) ↔ (N = 1 ∨ N = 9 ∨ N = 17 ∨ N = 57) :=
sorry

end num_valid_N_l2269_226958


namespace closest_ratio_adults_children_l2269_226942

theorem closest_ratio_adults_children (a c : ℕ) (h1 : 30 * a + 15 * c = 2250) (h2 : a ≥ 2) (h3 : c ≥ 2) : 
  (a : ℚ) / (c : ℚ) = 1 :=
  sorry

end closest_ratio_adults_children_l2269_226942


namespace gcd_polynomial_even_multiple_of_97_l2269_226950

theorem gcd_polynomial_even_multiple_of_97 (b : ℤ) (k : ℤ) (h_b : b = 2 * 97 * k) :
  Int.gcd (3 * b^2 + 41 * b + 74) (b + 19) = 1 :=
by
  sorry

end gcd_polynomial_even_multiple_of_97_l2269_226950


namespace gcd_98_63_l2269_226904

theorem gcd_98_63 : Nat.gcd 98 63 = 7 :=
by
  sorry

end gcd_98_63_l2269_226904


namespace max_elem_one_correct_max_elem_two_correct_min_range_x_correct_average_min_eq_x_correct_l2269_226922

def max_elem_one (c : ℝ) : Prop :=
  max (-2) (max 3 c) = max 3 c

def max_elem_two (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : Prop :=
  max (3 * m) (max ((n + 3) * m) (-m * n)) = - m * n

def min_range_x (x : ℝ) : Prop :=
  min 2 (min (2 * x + 2) (4 - 2 * x)) = 2 → 0 ≤ x ∧ x ≤ 1

def average_min_eq_x : Prop :=
  ∀ (x : ℝ), (2 + (x + 1) + 2 * x) / 3 = min 2 (min (x + 1) (2 * x)) → x = 1

-- Lean 4 statements
theorem max_elem_one_correct (c : ℝ) : max_elem_one c := 
  sorry

theorem max_elem_two_correct {m n : ℝ} (h1 : m < 0) (h2 : n > 0) : max_elem_two m n h1 h2 :=
  sorry

theorem min_range_x_correct (h : min 2 (min (2 * x + 2) (4 - 2 * x)) = 2) : min_range_x x :=
  sorry

theorem average_min_eq_x_correct : average_min_eq_x :=
  sorry

end max_elem_one_correct_max_elem_two_correct_min_range_x_correct_average_min_eq_x_correct_l2269_226922


namespace point_C_velocity_l2269_226931

theorem point_C_velocity (a T R L x : ℝ) (h : a * T / (a * T - R) = (L + x) / x) :
  x = L * (a * T / R - 1) → 
  (L * (a * T / R - 1)) / T = a * L / R :=
by
  sorry

end point_C_velocity_l2269_226931


namespace no_preimage_implies_p_gt_1_l2269_226930

   noncomputable def f (x : ℝ) : ℝ :=
     -x^2 + 2 * x

   theorem no_preimage_implies_p_gt_1 (p : ℝ) (hp : ∀ x : ℝ, f x ≠ p) : p > 1 :=
   sorry
   
end no_preimage_implies_p_gt_1_l2269_226930


namespace perfect_square_trinomial_implies_possible_m_values_l2269_226959

theorem perfect_square_trinomial_implies_possible_m_values (m : ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, (x - a)^2 = x^2 - 2*m*x + 16) → (m = 4 ∨ m = -4) :=
by
  sorry

end perfect_square_trinomial_implies_possible_m_values_l2269_226959


namespace max_a3_in_arith_geo_sequences_l2269_226961

theorem max_a3_in_arith_geo_sequences
  (a1 a2 a3 : ℝ) (b1 b2 b3 : ℝ)
  (h1 : a1 + a2 + a3 = 15)
  (h2 : a2 = ((a1 + a3) / 2))
  (h3 : b1 * b2 * b3 = 27)
  (h4 : (a1 + b1) * (a3 + b3) = (a2 + b2) ^ 2)
  (h5 : a1 + b1 > 0)
  (h6 : a2 + b2 > 0)
  (h7 : a3 + b3 > 0) :
  a3 ≤ 59 := sorry

end max_a3_in_arith_geo_sequences_l2269_226961


namespace math_problem_l2269_226949

open Classical

theorem math_problem (s x y : ℝ) (h₁ : s > 0) (h₂ : x^2 + y^2 ≠ 0) (h₃ : x * s^2 < y * s^2) :
  ¬(-x^2 < -y^2) ∧ ¬(-x^2 < y^2) ∧ ¬(x^2 < -y^2) ∧ ¬(x^2 > y^2) := by
  sorry

end math_problem_l2269_226949


namespace packages_per_box_l2269_226915

theorem packages_per_box (P : ℕ) 
  (h1 : 100 * 25 = 2500) 
  (h2 : 2 * P * 250 = 2500) : 
  P = 5 := 
sorry

end packages_per_box_l2269_226915


namespace average_of_consecutive_odds_is_24_l2269_226977

theorem average_of_consecutive_odds_is_24 (a b c d : ℤ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : d = 27) 
  (h5 : b = d - 2) (h6 : c = d - 4) (h7 : a = d - 6) 
  (h8 : ∀ x : ℤ, x % 2 = 1) :
  ((a + b + c + d) / 4) = 24 :=
by {
  sorry
}

end average_of_consecutive_odds_is_24_l2269_226977


namespace balls_distribution_l2269_226923

theorem balls_distribution : 
  ∃ (n : ℕ), 
    (∀ (b1 b2 : ℕ), ∀ (h : b1 + b2 = 4), b1 ≥ 1 ∧ b2 ≥ 2 → n = 10) :=
sorry

end balls_distribution_l2269_226923


namespace equation_solution_1_equation_solution_2_equation_solution_3_l2269_226934

def system_of_equations (x y : ℝ) : Prop :=
  (x * (x^2 - 3 * y^2) = 16) ∧ (y * (3 * x^2 - y^2) = 88)

theorem equation_solution_1 :
  system_of_equations 4 2 :=
by
  -- The proof is skipped.
  sorry

theorem equation_solution_2 :
  system_of_equations (-3.7) 2.5 :=
by
  -- The proof is skipped.
  sorry

theorem equation_solution_3 :
  system_of_equations (-0.3) (-4.5) :=
by
  -- The proof is skipped.
  sorry

end equation_solution_1_equation_solution_2_equation_solution_3_l2269_226934


namespace sequence_converges_to_zero_and_N_for_epsilon_l2269_226985

theorem sequence_converges_to_zero_and_N_for_epsilon :
  (∀ ε > 0, ∃ N : ℕ, ∀ n > N, |1 / (n : ℝ) - 0| < ε) ∧ 
  (∃ N : ℕ, ∀ n > N, |1 / (n : ℝ)| < 0.001) :=
by
  sorry

end sequence_converges_to_zero_and_N_for_epsilon_l2269_226985


namespace student_made_mistake_l2269_226918

theorem student_made_mistake (AB CD MLNKT : ℕ) (h1 : 10 ≤ AB ∧ AB ≤ 99) (h2 : 10 ≤ CD ∧ CD ≤ 99) (h3 : 10000 ≤ MLNKT ∧ MLNKT < 100000) : AB * CD ≠ MLNKT :=
by {
  sorry
}

end student_made_mistake_l2269_226918


namespace pairs_m_n_l2269_226908

theorem pairs_m_n (m n : ℤ) : n ^ 2 - 3 * m * n + m - n = 0 ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ n = 1) :=
by sorry

end pairs_m_n_l2269_226908


namespace product_of_first_two_terms_l2269_226910

-- Given parameters
variables (a d : ℤ) -- a is the first term, d is the common difference

-- Conditions
def fifth_term_condition (a d : ℤ) : Prop := a + 4 * d = 11
def common_difference_condition (d : ℤ) : Prop := d = 1

-- Main statement to prove
theorem product_of_first_two_terms (a d : ℤ) (h1 : fifth_term_condition a d) (h2 : common_difference_condition d) :
  a * (a + d) = 56 :=
by
  sorry

end product_of_first_two_terms_l2269_226910


namespace plant_branches_l2269_226921

theorem plant_branches (x : ℕ) (h : 1 + x + x^2 = 91) : 1 + x + x^2 = 91 :=
by sorry

end plant_branches_l2269_226921


namespace sufficient_remedy_l2269_226902

-- Definitions based on conditions
def aspirin_relieves_headache : Prop := true
def aspirin_relieves_knee_rheumatism : Prop := true
def aspirin_causes_heart_pain : Prop := true
def aspirin_causes_stomach_pain : Prop := true

def homeopathic_relieves_heart_issues : Prop := true
def homeopathic_relieves_stomach_issues : Prop := true
def homeopathic_causes_hip_rheumatism : Prop := true

def antibiotics_cure_migraines : Prop := true
def antibiotics_cure_heart_pain : Prop := true
def antibiotics_cause_stomach_pain : Prop := true
def antibiotics_cause_knee_pain : Prop := true
def antibiotics_cause_itching : Prop := true

def cortisone_relieves_itching : Prop := true
def cortisone_relieves_knee_rheumatism : Prop := true
def cortisone_exacerbates_hip_rheumatism : Prop := true

def warm_compress_relieves_itching : Prop := true
def warm_compress_relieves_stomach_pain : Prop := true

def severe_headache_morning : Prop := true
def impaired_ability_to_think : Prop := severe_headache_morning

-- Statement of the proof problem
theorem sufficient_remedy :
  (aspirin_relieves_headache ∧ antibiotics_cure_heart_pain ∧ warm_compress_relieves_itching ∧ warm_compress_relieves_stomach_pain) →
  (impaired_ability_to_think → true) :=
by
  sorry

end sufficient_remedy_l2269_226902


namespace positive_integer_divisors_of_sum_l2269_226938

theorem positive_integer_divisors_of_sum (n : ℕ) :
  (∃ n_values : Finset ℕ, 
    (∀ n ∈ n_values, n > 0 
      ∧ (n * (n + 1)) ∣ (2 * 10 * n)) 
      ∧ n_values.card = 5) :=
by
  sorry

end positive_integer_divisors_of_sum_l2269_226938


namespace A_time_to_complete_work_l2269_226920

-- Definitions of work rates for A, B, and C.
variables (A_work B_work C_work : ℚ)

-- Conditions
axiom cond1 : A_work = 3 * B_work
axiom cond2 : B_work = 2 * C_work
axiom cond3 : A_work + B_work + C_work = 1 / 15

-- Proof statement: The time taken by A alone to do the work is 22.5 days.
theorem A_time_to_complete_work : 1 / A_work = 22.5 :=
by {
  sorry
}

end A_time_to_complete_work_l2269_226920


namespace find_x_l2269_226979

def vector (α : Type*) := α × α

def parallel (a b : vector ℝ) : Prop :=
a.1 * b.2 - a.2 * b.1 = 0

theorem find_x (x : ℝ) (a b : vector ℝ)
  (ha : a = (1, 2))
  (hb : b = (x, 4))
  (h : parallel a b) : x = 2 :=
by sorry

end find_x_l2269_226979


namespace minimum_teachers_needed_l2269_226992

theorem minimum_teachers_needed
  (math_teachers : ℕ) (physics_teachers : ℕ) (chemistry_teachers : ℕ)
  (max_subjects_per_teacher : ℕ) :
  math_teachers = 7 →
  physics_teachers = 6 →
  chemistry_teachers = 5 →
  max_subjects_per_teacher = 3 →
  ∃ t : ℕ, t = 5 ∧ (t * max_subjects_per_teacher ≥ math_teachers + physics_teachers + chemistry_teachers) :=
by
  repeat { sorry }

end minimum_teachers_needed_l2269_226992


namespace num_integer_ks_l2269_226925

theorem num_integer_ks (k : Int) :
  (∃ a b c d : Int, (2*x + a) * (x + b) = 2*x^2 - k*x + 6 ∨
                   (2*x + c) * (x + d) = 2*x^2 - k*x + 6) →
  ∃ ks : Finset Int, ks.card = 6 ∧ k ∈ ks :=
sorry

end num_integer_ks_l2269_226925


namespace factor_expression_l2269_226964

theorem factor_expression (y : ℝ) : 84 * y ^ 13 + 210 * y ^ 26 = 42 * y ^ 13 * (2 + 5 * y ^ 13) :=
by sorry

end factor_expression_l2269_226964


namespace rational_solutions_quadratic_eq_l2269_226909

theorem rational_solutions_quadratic_eq (k : ℕ) (h_pos : k > 0) :
  (∃ x : ℚ, k * x^2 + 24 * x + k = 0) ↔ (k = 8 ∨ k = 12) :=
by sorry

end rational_solutions_quadratic_eq_l2269_226909
