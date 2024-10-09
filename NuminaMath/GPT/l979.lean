import Mathlib

namespace sqrt_calculation_l979_97914

theorem sqrt_calculation :
  Real.sqrt ((2:ℝ)^4 * 3^2 * 5^2) = 60 := 
by sorry

end sqrt_calculation_l979_97914


namespace ten_fact_minus_nine_fact_l979_97997

-- Definitions corresponding to the conditions
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Condition for 9!
def nine_factorial : ℕ := 362880

-- 10! can be expressed in terms of 9!
noncomputable def ten_factorial : ℕ := 10 * nine_factorial

-- Proof statement we need to show
theorem ten_fact_minus_nine_fact : ten_factorial - nine_factorial = 3265920 :=
by
  unfold ten_factorial
  unfold nine_factorial
  sorry

end ten_fact_minus_nine_fact_l979_97997


namespace price_reduction_example_l979_97920

def original_price_per_mango (P : ℝ) : Prop :=
  (115 * P = 383.33)

def number_of_mangoes (P : ℝ) (n : ℝ) : Prop :=
  (n * P = 360)

def new_number_of_mangoes (n : ℝ) (R : ℝ) : Prop :=
  ((n + 12) * R = 360)

def percentage_reduction (P R : ℝ) (reduction : ℝ) : Prop :=
  (reduction = ((P - R) / P) * 100)

theorem price_reduction_example : 
  ∃ P R reduction, original_price_per_mango P ∧
    (∃ n, number_of_mangoes P n ∧ new_number_of_mangoes n R) ∧ 
    percentage_reduction P R reduction ∧ 
    reduction = 9.91 :=
by
  sorry

end price_reduction_example_l979_97920


namespace compound_interest_time_l979_97921

theorem compound_interest_time (P r CI : ℝ) (n : ℕ) (A : ℝ) :
  P = 16000 ∧ r = 0.15 ∧ CI = 6218 ∧ n = 1 ∧ A = P + CI →
  t = 2 :=
by
  sorry

end compound_interest_time_l979_97921


namespace ratio_of_white_marbles_l979_97992

theorem ratio_of_white_marbles (total_marbles yellow_marbles red_marbles : ℕ)
    (h1 : total_marbles = 50)
    (h2 : yellow_marbles = 12)
    (h3 : red_marbles = 7)
    (green_marbles : ℕ)
    (h4 : green_marbles = yellow_marbles - yellow_marbles / 2) :
    (total_marbles - (yellow_marbles + green_marbles + red_marbles)) / total_marbles = 1 / 2 :=
by
  sorry

end ratio_of_white_marbles_l979_97992


namespace range_of_x_l979_97930

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x

theorem range_of_x (x : ℝ) (h : f (x^2 + 2) < f (3 * x)) : 1 < x ∧ x < 2 :=
by sorry

end range_of_x_l979_97930


namespace economy_class_seats_l979_97907

-- Definitions based on the conditions
def first_class_people : ℕ := 3
def business_class_people : ℕ := 22
def economy_class_fullness (E : ℕ) : ℕ := E / 2

-- Problem statement: Proving E == 50 given the conditions
theorem economy_class_seats :
  ∃ E : ℕ,  economy_class_fullness E = first_class_people + business_class_people → E = 50 :=
by
  sorry

end economy_class_seats_l979_97907


namespace quadratic_inequality_ab_l979_97973

theorem quadratic_inequality_ab (a b : ℝ) 
  (h1 : ∀ x : ℝ, (a * x^2 + b * x + 1 > 0) ↔ -1 < x ∧ x < 1 / 3) :
  a * b = -6 :=
by
  -- Proof is omitted
  sorry

end quadratic_inequality_ab_l979_97973


namespace second_factor_of_lcm_l979_97978

theorem second_factor_of_lcm (A B : ℕ) (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) (lcm : ℕ) 
  (h1 : hcf = 20) 
  (h2 : A = 280)
  (h3 : factor1 = 13) 
  (h4 : lcm = hcf * factor1 * factor2) 
  (h5 : A = hcf * 14) : 
  factor2 = 14 :=
by 
  sorry

end second_factor_of_lcm_l979_97978


namespace rocket_soaring_time_l979_97976

theorem rocket_soaring_time 
  (avg_speed : ℝ)                      -- The average speed of the rocket
  (soar_speed : ℝ)                     -- Speed while soaring
  (plummet_distance : ℝ)               -- Distance covered during plummet
  (plummet_time : ℝ)                   -- Time of plummet
  (total_time : ℝ := plummet_time + t) -- Total time is the sum of soaring time and plummet time
  (total_distance : ℝ := soar_speed * t + plummet_distance) -- Total distance covered
  (h_avg_speed : avg_speed = total_distance / total_time)   -- Given condition for average speed
  :
  ∃ t : ℝ, t = 12 :=                   -- Prove that the soaring time is 12 seconds
by
  sorry

end rocket_soaring_time_l979_97976


namespace number_of_pages_to_copy_l979_97913

-- Definitions based on the given conditions
def total_budget : ℕ := 5000
def service_charge : ℕ := 500
def copy_cost : ℕ := 3

-- Derived definition based on the conditions
def remaining_budget : ℕ := total_budget - service_charge

-- The statement we need to prove
theorem number_of_pages_to_copy : (remaining_budget / copy_cost) = 1500 :=
by {
  sorry
}

end number_of_pages_to_copy_l979_97913


namespace count_positive_bases_for_log_1024_l979_97964

-- Define the conditions 
def is_positive_integer_log_base (b n : ℕ) : Prop := b^n = 1024 ∧ n > 0

-- State that there are exactly 4 positive integers b that satisfy the condition
theorem count_positive_bases_for_log_1024 :
  (∃ b1 b2 b3 b4 : ℕ, b1 ≠ b2 ∧ b1 ≠ b3 ∧ b1 ≠ b4 ∧ b2 ≠ b3 ∧ b2 ≠ b4 ∧ b3 ≠ b4 ∧
    (∀ b, is_positive_integer_log_base b 1 ∨ is_positive_integer_log_base b 2 ∨ is_positive_integer_log_base b 5 ∨ is_positive_integer_log_base b 10) ∧
    (is_positive_integer_log_base b1 1 ∨ is_positive_integer_log_base b1 2 ∨ is_positive_integer_log_base b1 5 ∨ is_positive_integer_log_base b1 10) ∧
    (is_positive_integer_log_base b2 1 ∨ is_positive_integer_log_base b2 2 ∨ is_positive_integer_log_base b2 5 ∨ is_positive_integer_log_base b2 10) ∧
    (is_positive_integer_log_base b3 1 ∨ is_positive_integer_log_base b3 2 ∨ is_positive_integer_log_base b3 5 ∨ is_positive_integer_log_base b3 10) ∧
    (is_positive_integer_log_base b4 1 ∨ is_positive_integer_log_base b4 2 ∨ is_positive_integer_log_base b4 5 ∨ is_positive_integer_log_base b4 10)) :=
sorry

end count_positive_bases_for_log_1024_l979_97964


namespace rectangular_prism_diagonal_l979_97919

theorem rectangular_prism_diagonal 
  (a b c : ℝ)
  (h1 : 2 * a * b + 2 * b * c + 2 * c * a = 11)
  (h2 : 4 * (a + b + c) = 24) :
  (a^2 + b^2 + c^2 = 25) :=
by {
  -- Sorry to skip the proof steps
  sorry
}

end rectangular_prism_diagonal_l979_97919


namespace billion_in_scientific_notation_l979_97929

theorem billion_in_scientific_notation :
  (4.55 * 10^9) = (4.55 * 10^9) := by
  sorry

end billion_in_scientific_notation_l979_97929


namespace unique_solution_2023_plus_2_pow_n_eq_k_sq_l979_97922

theorem unique_solution_2023_plus_2_pow_n_eq_k_sq (n k : ℕ) (h : 2023 + 2^n = k^2) :
  (n = 1 ∧ k = 45) :=
by
  sorry

end unique_solution_2023_plus_2_pow_n_eq_k_sq_l979_97922


namespace equivalent_eq_l979_97982

variable {x y : ℝ}

theorem equivalent_eq (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) ↔ (y = 6 * x / (x - 9)) :=
by
  sorry

end equivalent_eq_l979_97982


namespace two_integer_solutions_iff_m_l979_97941

def op (p q : ℝ) : ℝ := p + q - p * q

theorem two_integer_solutions_iff_m (m : ℝ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ op 2 x1 > 0 ∧ op x1 3 ≤ m ∧ op 2 x2 > 0 ∧ op x2 3 ≤ m) ↔ 3 ≤ m ∧ m < 5 :=
by
  sorry

end two_integer_solutions_iff_m_l979_97941


namespace exists_close_points_l979_97928

theorem exists_close_points (r : ℝ) (h : r > 0) (points : Fin 5 → EuclideanSpace ℝ (Fin 3)) (hf : ∀ i, dist (points i) (0 : EuclideanSpace ℝ (Fin 3)) = r) :
  ∃ i j : Fin 5, i ≠ j ∧ dist (points i) (points j) ≤ r * Real.sqrt 2 :=
by 
  sorry

end exists_close_points_l979_97928


namespace a_alone_completes_in_eight_days_l979_97925

variable (a b : Type)
variables (days_ab : ℝ) (days_a : ℝ) (days_ab_2 : ℝ)

noncomputable def days := ℝ

axiom work_together_four_days : days_ab = 4
axiom work_together_266666_days : days_ab_2 = 8 / 3

theorem a_alone_completes_in_eight_days (a b : Type) (days_ab : ℝ) (days_a : ℝ) (days_ab_2 : ℝ)
  (work_together_four_days : days_ab = 4)
  (work_together_266666_days : days_ab_2 = 8 / 3) :
  days_a = 8 :=
by
  sorry

end a_alone_completes_in_eight_days_l979_97925


namespace min_vases_required_l979_97950

theorem min_vases_required (carnations roses tulips lilies : ℕ)
  (flowers_in_A flowers_in_B flowers_in_C : ℕ) 
  (total_flowers : ℕ) 
  (h_carnations : carnations = 10) 
  (h_roses : roses = 25) 
  (h_tulips : tulips = 15) 
  (h_lilies : lilies = 20)
  (h_flowers_in_A : flowers_in_A = 4) 
  (h_flowers_in_B : flowers_in_B = 6) 
  (h_flowers_in_C : flowers_in_C = 8)
  (h_total_flowers : total_flowers = carnations + roses + tulips + lilies) :
  total_flowers = 70 → 
  (exists vases_A vases_B vases_C : ℕ, 
    vases_A = 0 ∧ 
    vases_B = 1 ∧ 
    vases_C = 8 ∧ 
    total_flowers = vases_A * flowers_in_A + vases_B * flowers_in_B + vases_C * flowers_in_C) :=
by
  intros
  sorry

end min_vases_required_l979_97950


namespace value_of_m_l979_97965

def f (x : ℚ) : ℚ := 3 * x^3 - 1 / x + 2
def g (x : ℚ) (m : ℚ) : ℚ := 2 * x^3 - 3 * x + m
def h (x : ℚ) : ℚ := x^2

theorem value_of_m : f 3 - g 3 (122 / 3) + h 3 = 5 :=
by
  sorry

end value_of_m_l979_97965


namespace melissa_total_score_l979_97971

theorem melissa_total_score (games : ℕ) (points_per_game : ℕ) 
  (h_games : games = 3) (h_points_per_game : points_per_game = 27) : 
  points_per_game * games = 81 := 
by 
  sorry

end melissa_total_score_l979_97971


namespace find_ratio_l979_97961

variable {x y z : ℝ}

theorem find_ratio
  (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (2 * x + y - z) / (3 * x - 2 * y + z) = 5 / 6 := by
  sorry

end find_ratio_l979_97961


namespace sides_of_triangle_inequality_l979_97952

theorem sides_of_triangle_inequality (a b c : ℝ) (h : a + b > c) : a + b > c := 
by 
  exact h

end sides_of_triangle_inequality_l979_97952


namespace calculate_M_minus_m_l979_97991

def total_students : ℕ := 2001
def students_studying_spanish (S : ℕ) : Prop := 1601 ≤ S ∧ S ≤ 1700
def students_studying_french (F : ℕ) : Prop := 601 ≤ F ∧ F ≤ 800
def studying_both_languages_lower_bound (S F m : ℕ) : Prop := S + F - m = total_students
def studying_both_languages_upper_bound (S F M : ℕ) : Prop := S + F - M = total_students

theorem calculate_M_minus_m :
  ∀ (S F m M : ℕ),
    students_studying_spanish S →
    students_studying_french F →
    studying_both_languages_lower_bound S F m →
    studying_both_languages_upper_bound S F M →
    S = 1601 ∨ S = 1700 →
    F = 601 ∨ F = 800 →
    M - m = 298 :=
by
  intros S F m M hs hf hl hb Hs Hf
  sorry

end calculate_M_minus_m_l979_97991


namespace a_5_is_9_l979_97959

-- Definition of the sequence sum S_n
def S : ℕ → ℕ
| n => n^2 - 1

-- Define the specific term in the sequence
def a (n : ℕ) :=
  if n = 1 then S 1 else S n - S (n - 1)

-- Theorem to prove
theorem a_5_is_9 : a 5 = 9 :=
sorry

end a_5_is_9_l979_97959


namespace scientific_notation_of_258000000_l979_97924

theorem scientific_notation_of_258000000 :
  258000000 = 2.58 * 10^8 :=
sorry

end scientific_notation_of_258000000_l979_97924


namespace product_of_integers_around_sqrt_50_l979_97966

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l979_97966


namespace black_piece_probability_l979_97960

-- Definitions based on conditions
def total_pieces : ℕ := 10 + 5
def black_pieces : ℕ := 10

-- Probability calculation
def probability_black : ℚ := black_pieces / total_pieces

-- Statement to prove
theorem black_piece_probability : probability_black = 2/3 := by
  sorry -- proof to be filled in later

end black_piece_probability_l979_97960


namespace find_third_number_l979_97967

-- Define the given conditions
def proportion_condition (x y : ℝ) : Prop :=
  (0.75 / x) = (y / 8)

-- The main statement to be proven
theorem find_third_number (x y : ℝ) (hx : x = 1.2) (h_proportion : proportion_condition x y) : y = 5 :=
by
  -- Using the assumptions and the definition provided.
  sorry

end find_third_number_l979_97967


namespace senior_high_sample_count_l979_97977

theorem senior_high_sample_count 
  (total_students : ℕ)
  (junior_high_students : ℕ)
  (senior_high_students : ℕ)
  (total_sampled_students : ℕ)
  (H1 : total_students = 1800)
  (H2 : junior_high_students = 1200)
  (H3 : senior_high_students = 600)
  (H4 : total_sampled_students = 180) :
  (senior_high_students * total_sampled_students / total_students) = 60 := 
sorry

end senior_high_sample_count_l979_97977


namespace smallest_composite_no_prime_factors_below_15_correct_l979_97923

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l979_97923


namespace grunters_at_least_4_wins_l979_97917

noncomputable def grunters_probability : ℚ :=
  let p_win := 3 / 5
  let p_loss := 2 / 5
  let p_4_wins := 5 * (p_win^4) * (p_loss)
  let p_5_wins := p_win^5
  p_4_wins + p_5_wins

theorem grunters_at_least_4_wins :
  grunters_probability = 1053 / 3125 :=
by sorry

end grunters_at_least_4_wins_l979_97917


namespace arithmetic_result_l979_97975

theorem arithmetic_result :
  (3 * 13) + (3 * 14) + (3 * 17) + 11 = 143 :=
by
  sorry

end arithmetic_result_l979_97975


namespace not_enrolled_eq_80_l979_97936

variable (total_students : ℕ)
variable (french_students : ℕ)
variable (german_students : ℕ)
variable (spanish_students : ℕ)
variable (french_and_german : ℕ)
variable (german_and_spanish : ℕ)
variable (spanish_and_french : ℕ)
variable (all_three : ℕ)

noncomputable def students_not_enrolled_in_any_language 
  (total_students french_students german_students spanish_students french_and_german german_and_spanish spanish_and_french all_three : ℕ) : ℕ :=
  total_students - (french_students + german_students + spanish_students - french_and_german - german_and_spanish - spanish_and_french + all_three)

theorem not_enrolled_eq_80 : 
  students_not_enrolled_in_any_language 180 60 50 35 20 15 10 5 = 80 :=
  by
    unfold students_not_enrolled_in_any_language
    simp
    sorry

end not_enrolled_eq_80_l979_97936


namespace sqrt_40_simplified_l979_97953

theorem sqrt_40_simplified : Real.sqrt 40 = 2 * Real.sqrt 10 := 
by
  sorry

end sqrt_40_simplified_l979_97953


namespace solution_set_of_floor_eqn_l979_97994

theorem solution_set_of_floor_eqn:
  ∀ x y : ℝ, 
  (⌊x⌋ * ⌊x⌋ + ⌊y⌋ * ⌊y⌋ = 4) ↔ 
  ((2 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 1) ∨
   (0 ≤ x ∧ x < 1 ∧ 2 ≤ y ∧ y < 3) ∨
   (-2 ≤ x ∧ x < -1 ∧ 0 ≤ y ∧ y < 1) ∨
   (0 ≤ x ∧ x < 1 ∧ -2 ≤ y ∧ y < -1)) :=
by
  sorry

end solution_set_of_floor_eqn_l979_97994


namespace star_computation_l979_97985

def star (x y : ℝ) := x * y - 3 * x + y

theorem star_computation :
  (star 5 8) - (star 8 5) = 12 := by
  sorry

end star_computation_l979_97985


namespace standard_deviation_less_than_l979_97909

theorem standard_deviation_less_than:
  ∀ (μ σ : ℝ)
  (h1 : μ = 55)
  (h2 : μ - 3 * σ > 48),
  σ < 7 / 3 :=
by
  intros μ σ h1 h2
  sorry

end standard_deviation_less_than_l979_97909


namespace hexagon_angle_in_arithmetic_progression_l979_97979

theorem hexagon_angle_in_arithmetic_progression :
  ∃ (a d : ℝ), (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) = 720) ∧ 
  (a = 120 ∨ a + d = 120 ∨ a + 2 * d = 120 ∨ a + 3 * d = 120 ∨ a + 4 * d = 120 ∨ a + 5 * d = 120) := by
  sorry

end hexagon_angle_in_arithmetic_progression_l979_97979


namespace travel_distance_of_wheel_l979_97905

theorem travel_distance_of_wheel (r : ℝ) (revolutions : ℕ) (h_r : r = 2) (h_revolutions : revolutions = 2) : 
    ∃ d : ℝ, d = 8 * Real.pi :=
by
  sorry

end travel_distance_of_wheel_l979_97905


namespace minimum_adjacent_white_pairs_l979_97958

theorem minimum_adjacent_white_pairs (total_black_cells : ℕ) (grid_size : ℕ) (total_pairs : ℕ) : 
  total_black_cells = 20 ∧ grid_size = 8 ∧ total_pairs = 112 → ∃ min_white_pairs : ℕ, min_white_pairs = 34 :=
by
  sorry

end minimum_adjacent_white_pairs_l979_97958


namespace ratio_of_percent_changes_l979_97938

noncomputable def price_decrease_ratio (original_price : ℝ) (new_price : ℝ) : ℝ :=
(original_price - new_price) / original_price * 100

noncomputable def units_increase_ratio (original_units : ℝ) (new_units : ℝ) : ℝ :=
(new_units - original_units) / original_units * 100

theorem ratio_of_percent_changes 
  (original_price new_price original_units new_units : ℝ)
  (h1 : new_price = 0.7 * original_price)
  (h2 : original_price * original_units = new_price * new_units)
  : (units_increase_ratio original_units new_units) / (price_decrease_ratio original_price new_price) = 1.4285714285714286 :=
by
  sorry

end ratio_of_percent_changes_l979_97938


namespace edward_initial_amount_l979_97900

-- Defining the conditions
def cost_books : ℕ := 6
def cost_pens : ℕ := 16
def cost_notebook : ℕ := 5
def cost_pencil_case : ℕ := 3
def amount_left : ℕ := 19

-- Mathematical statement to prove
theorem edward_initial_amount : 
  cost_books + cost_pens + cost_notebook + cost_pencil_case + amount_left = 49 :=
by
  sorry

end edward_initial_amount_l979_97900


namespace least_n_for_perfect_square_l979_97931

theorem least_n_for_perfect_square (n : ℕ) :
  (∀ m : ℕ, 2^8 + 2^11 + 2^n = m * m) → n = 12 := sorry

end least_n_for_perfect_square_l979_97931


namespace trapezoid_area_l979_97908

theorem trapezoid_area (h : ℝ) : 
  let b1 : ℝ := 4 * h + 2
  let b2 : ℝ := 5 * h
  (b1 + b2) / 2 * h = (9 * h ^ 2 + 2 * h) / 2 :=
by 
  let b1 := 4 * h + 2
  let b2 := 5 * h
  sorry

end trapezoid_area_l979_97908


namespace parabola_focus_coordinates_l979_97903

theorem parabola_focus_coordinates (x y : ℝ) (h : y = 2 * x^2) : (0, 1 / 8) = (0, 1 / 8) :=
by
  sorry

end parabola_focus_coordinates_l979_97903


namespace find_y_of_arithmetic_mean_l979_97986

theorem find_y_of_arithmetic_mean (y : ℝ) (h : (8 + 16 + 12 + 24 + 7 + y) / 6 = 12) : y = 5 :=
by
  sorry

end find_y_of_arithmetic_mean_l979_97986


namespace total_questions_in_test_l979_97939

theorem total_questions_in_test :
  ∃ x, (5 * x = total_questions) ∧ 
       (20 : ℚ) / total_questions > (60 / 100 : ℚ) ∧ 
       (20 : ℚ) / total_questions < (70 / 100 : ℚ) ∧ 
       total_questions = 30 :=
by
  sorry

end total_questions_in_test_l979_97939


namespace ab_relationship_l979_97935

theorem ab_relationship (a b : ℝ) (n : ℕ) (h1 : a^n = a + 1) (h2 : b^(2*n) = b + 3*a) (h3 : n ≥ 2) (h4 : 0 < a) (h5 : 0 < b) :
  a > b ∧ a > 1 ∧ b > 1 :=
sorry

end ab_relationship_l979_97935


namespace minimum_value_inequality_l979_97999

variable {x y z : ℝ}
variable (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)

theorem minimum_value_inequality : (x + y + z) * (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 9 / 2 :=
sorry

end minimum_value_inequality_l979_97999


namespace avg_marks_second_class_l979_97911

theorem avg_marks_second_class
  (x : ℝ)
  (avg_class1 : ℝ)
  (avg_total : ℝ)
  (n1 n2 : ℕ)
  (h1 : n1 = 30)
  (h2 : n2 = 50)
  (h3 : avg_class1 = 30)
  (h4: avg_total = 48.75)
  (h5 : (n1 * avg_class1 + n2 * x) / (n1 + n2) = avg_total) :
  x = 60 := by
  sorry

end avg_marks_second_class_l979_97911


namespace solve_quadratic_l979_97910

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = 2 + Real.sqrt 11) ∧ (x2 = 2 - (Real.sqrt 11)) ∧ 
  (∀ x : ℝ, x^2 - 4*x - 7 = 0 ↔ x = x1 ∨ x = x2) := 
sorry

end solve_quadratic_l979_97910


namespace range_of_m_three_zeros_l979_97934

theorem range_of_m_three_zeros (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (x^3 - 3*x + m = 0) ∧ (y^3 - 3*y + m = 0) ∧ (z^3 - 3*z + m = 0)) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end range_of_m_three_zeros_l979_97934


namespace family_gathering_l979_97946

theorem family_gathering : 
  ∃ (total_people oranges bananas apples : ℕ), 
    total_people = 20 ∧ 
    oranges = total_people / 2 ∧ 
    bananas = (total_people - oranges) / 2 ∧ 
    apples = total_people - oranges - bananas ∧ 
    oranges < total_people ∧ 
    total_people - oranges = 10 :=
by
  sorry

end family_gathering_l979_97946


namespace range_a_le_2_l979_97996
-- Import everything from Mathlib

-- Define the hypothesis and the conclusion in Lean 4
theorem range_a_le_2 (a : ℝ) : 
  (∀ x > 0, Real.log x + a * x + 1 - x * Real.exp (2 * x) ≤ 0) ↔ a ≤ 2 := 
sorry

end range_a_le_2_l979_97996


namespace ratio_of_increase_to_original_l979_97995

noncomputable def ratio_increase_avg_marks (T : ℝ) : ℝ :=
  let original_avg := T / 40
  let new_total := T + 20
  let new_avg := new_total / 40
  let increase_avg := new_avg - original_avg
  increase_avg / original_avg

theorem ratio_of_increase_to_original (T : ℝ) (hT : T > 0) :
  ratio_increase_avg_marks T = 20 / T :=
by
  unfold ratio_increase_avg_marks
  sorry

end ratio_of_increase_to_original_l979_97995


namespace distance_A_to_B_l979_97981

theorem distance_A_to_B (D_B D_C V_E V_F : ℝ) (h1 : D_B / 3 = V_E)
  (h2 : D_C / 4 = V_F) (h3 : V_E / V_F = 2.533333333333333)
  (h4 : D_B = 300 ∨ D_C = 300) : D_B = 570 :=
by
  -- Proof yet to be provided
  sorry

end distance_A_to_B_l979_97981


namespace stone_counting_l979_97954

theorem stone_counting (n : ℕ) (m : ℕ) : 
    10 > 0 ∧  (n ≡ 6 [MOD 20]) ∧ m = 126 → n = 6 := 
by
  sorry

end stone_counting_l979_97954


namespace line_equation_solution_l979_97970

noncomputable def line_equation (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : Prop :=
  ∃ (l : ℝ → ℝ), (l P.fst = P.snd) ∧ (∀ (x : ℝ), l x = 4 * x - 2) ∨ (∀ (x : ℝ), x = 1)

theorem line_equation_solution : line_equation (1, 2) (2, 3) (0, -5) :=
sorry

end line_equation_solution_l979_97970


namespace average_playtime_in_minutes_l979_97974

noncomputable def lena_playtime_hours : ℝ := 3.5
noncomputable def lena_playtime_minutes : ℝ := lena_playtime_hours * 60
noncomputable def brother_playtime_minutes : ℝ := 1.2 * lena_playtime_minutes + 17
noncomputable def sister_playtime_minutes : ℝ := 1.5 * brother_playtime_minutes

theorem average_playtime_in_minutes :
  (lena_playtime_minutes + brother_playtime_minutes + sister_playtime_minutes) / 3 = 294.17 :=
by
  sorry

end average_playtime_in_minutes_l979_97974


namespace nancy_water_intake_l979_97943

theorem nancy_water_intake (water_intake body_weight : ℝ) (h1 : water_intake = 54) (h2 : body_weight = 90) : 
  (water_intake / body_weight) * 100 = 60 :=
by
  -- using the conditions h1 and h2
  rw [h1, h2]
  -- skipping the proof
  sorry

end nancy_water_intake_l979_97943


namespace probability_XOXOXOX_is_one_over_thirty_five_l979_97915

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_arrangement : ℤ :=
  let total_arrangements := binomial_coefficient 7 4
  let favorable_outcomes := 1
  favorable_outcomes / total_arrangements

theorem probability_XOXOXOX_is_one_over_thirty_five :
  probability_of_arrangement = (1 : ℤ) / 35 := 
  by
  sorry

end probability_XOXOXOX_is_one_over_thirty_five_l979_97915


namespace volleyball_tournament_first_place_score_l979_97962

theorem volleyball_tournament_first_place_score :
  ∃ (a b c d : ℕ), (a + b + c + d = 18) ∧ (a < b ∧ b < c ∧ c < d) ∧ (d = 6) :=
by
  sorry

end volleyball_tournament_first_place_score_l979_97962


namespace roots_reciprocal_sum_eq_25_l979_97993

theorem roots_reciprocal_sum_eq_25 (p q r : ℝ) (hpq : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) (hroot : ∀ x, x^3 - 9*x^2 + 8*x + 2 = 0 → (x = p ∨ x = q ∨ x = r)) :
  1/p^2 + 1/q^2 + 1/r^2 = 25 :=
by sorry

end roots_reciprocal_sum_eq_25_l979_97993


namespace polynomial_properties_l979_97937

noncomputable def polynomial : Polynomial ℚ :=
  -3/8 * (Polynomial.X ^ 5) + 5/4 * (Polynomial.X ^ 3) - 15/8 * (Polynomial.X)

theorem polynomial_properties (f : Polynomial ℚ) :
  (Polynomial.degree f = 5) ∧
  (∃ q : Polynomial ℚ, f + 1 = Polynomial.X - 1 ^ 3 * q) ∧
  (∃ p : Polynomial ℚ, f - 1 = Polynomial.X + 1 ^ 3 * p) ↔
  f = polynomial :=
by sorry

end polynomial_properties_l979_97937


namespace scientific_notation_gdp_l979_97944

theorem scientific_notation_gdp :
  8837000000 = 8.837 * 10^9 := 
by
  sorry

end scientific_notation_gdp_l979_97944


namespace distribution_of_balls_into_boxes_l979_97902

noncomputable def partitions_of_6_into_4_boxes : ℕ := 9

theorem distribution_of_balls_into_boxes :
  let balls := 6
  let boxes := 4
  let ways := partitions_of_6_into_4_boxes
  ways = 9 :=
by
  let balls := 6
  let boxes := 4
  let ways := partitions_of_6_into_4_boxes
  sorry

end distribution_of_balls_into_boxes_l979_97902


namespace max_n_l979_97906

def sum_first_n_terms (S n : ℕ) (a : ℕ → ℕ) : Prop :=
  S = 2 * a n - n

theorem max_n (S : ℕ) (a : ℕ → ℕ) :
  (∀ n, sum_first_n_terms S n a) → ∀ n, (2 ^ n - 1 ≤ 10 * n) → n ≤ 5 :=
by
  sorry

end max_n_l979_97906


namespace solve_for_x_l979_97948

theorem solve_for_x (x : ℝ) (h : 0 < x) (h_property : (x / 100) * x^2 = 9) : x = 10 := by
  sorry

end solve_for_x_l979_97948


namespace more_likely_second_machine_l979_97916

variable (P_B1 : ℝ := 0.8) -- Probability that a part is from the first machine
variable (P_B2 : ℝ := 0.2) -- Probability that a part is from the second machine
variable (P_A_given_B1 : ℝ := 0.01) -- Probability that a part is defective given it is from the first machine
variable (P_A_given_B2 : ℝ := 0.05) -- Probability that a part is defective given it is from the second machine

noncomputable def P_A : ℝ :=
  P_B1 * P_A_given_B1 + P_B2 * P_A_given_B2

noncomputable def P_B1_given_A : ℝ :=
  (P_B1 * P_A_given_B1) / P_A

noncomputable def P_B2_given_A : ℝ :=
  (P_B2 * P_A_given_B2) / P_A

theorem more_likely_second_machine :
  P_B2_given_A > P_B1_given_A :=
by
  sorry

end more_likely_second_machine_l979_97916


namespace union_complement_set_l979_97998

theorem union_complement_set (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5}) 
  (hA : A = {1, 2, 3, 5}) (hB : B = {2, 4}) :
  (U \ A) ∪ B = {0, 2, 4} :=
by
  rw [Set.diff_eq, hU, hA, hB]
  simp
  sorry

end union_complement_set_l979_97998


namespace original_average_weight_l979_97942

-- Definitions from conditions
def original_team_size : ℕ := 7
def new_player1_weight : ℝ := 110
def new_player2_weight : ℝ := 60
def new_team_size := original_team_size + 2
def new_average_weight : ℝ := 106

-- Statement to prove
theorem original_average_weight (W : ℝ) :
  (7 * W + 110 + 60 = 9 * 106) → W = 112 := by
  sorry

end original_average_weight_l979_97942


namespace probability_white_second_given_red_first_l979_97983

theorem probability_white_second_given_red_first :
  let total_balls := 8
  let red_balls := 5
  let white_balls := 3
  let event_A := red_balls
  let event_B_given_A := white_balls

  (event_B_given_A * (total_balls - 1)) / (event_A * total_balls) = 3 / 7 :=
by
  sorry

end probability_white_second_given_red_first_l979_97983


namespace profit_ratio_l979_97972

-- Definitions based on conditions
-- Let A_orig and B_orig represent the original profits of stores A and B
-- after increase and decrease respectively, they become equal

variable (A_orig B_orig : ℝ)
variable (h1 : (1.2 * A_orig) = (0.9 * B_orig))

-- Prove that the original profit of store A was 75% of the profit of store B
theorem profit_ratio (h1 : 1.2 * A_orig = 0.9 * B_orig) : A_orig = 0.75 * B_orig :=
by
  -- Insert proof here
  sorry

end profit_ratio_l979_97972


namespace find_perimeter_and_sin2A_of_triangle_l979_97987

theorem find_perimeter_and_sin2A_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h_a : a = 3) (h_B : B = Real.pi / 3) (h_area : 6 * Real.sqrt 3 = 6 * Real.sqrt 3)
  (h_S : S_ABC = 6 * Real.sqrt 3) : 
  (a + b + c = 18) ∧ (Real.sin (2 * A) = (39 * Real.sqrt 3) / 98) := 
by 
  -- The proof will be placed here. Assuming a valid proof exists.
  sorry

end find_perimeter_and_sin2A_of_triangle_l979_97987


namespace smallest_number_of_marbles_l979_97956

-- Define the conditions
variables (r w b g n : ℕ)
def valid_total (r w b g n : ℕ) := r + w + b + g = n
def valid_probability_4r (r w b g n : ℕ) := r * (r-1) * (r-2) * (r-3) = 24 * (w) * (r * (r - 1) * (r - 2) / 6)
def valid_probability_1w3r (r w b g n : ℕ) := r * (r-1) * (r-2) * (r-3) = 24 * (w * b * (r * (r - 1) / 2))
def valid_probability_1w1b2r (r w b g n : ℕ) := w * b * (r * (r - 1) / 2) = w * b * g * r

theorem smallest_number_of_marbles :
  ∃ n r w b g, valid_total r w b g n ∧
  valid_probability_4r r w b g n ∧
  valid_probability_1w3r r w b g n ∧
  valid_probability_1w1b2r r w b g n ∧ 
  n = 21 :=
  sorry

end smallest_number_of_marbles_l979_97956


namespace cubic_boxes_properties_l979_97901

-- Define the lengths of the edges of the cubic boxes
def edge_length_1 : ℝ := 3
def edge_length_2 : ℝ := 5
def edge_length_3 : ℝ := 6

-- Define the volumes of the respective cubic boxes
def volume (edge_length : ℝ) : ℝ := edge_length ^ 3
def volume_1 := volume edge_length_1
def volume_2 := volume edge_length_2
def volume_3 := volume edge_length_3

-- Define the surface areas of the respective cubic boxes
def surface_area (edge_length : ℝ) : ℝ := 6 * (edge_length ^ 2)
def surface_area_1 := surface_area edge_length_1
def surface_area_2 := surface_area edge_length_2
def surface_area_3 := surface_area edge_length_3

-- Total volume and surface area calculations
def total_volume := volume_1 + volume_2 + volume_3
def total_surface_area := surface_area_1 + surface_area_2 + surface_area_3

-- Theorem statement to be proven
theorem cubic_boxes_properties :
  total_volume = 368 ∧ total_surface_area = 420 := by
  sorry

end cubic_boxes_properties_l979_97901


namespace A_inter_B_empty_l979_97957

def setA : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def setB : Set ℝ := {x | Real.log x / Real.log 4 > 1/2}

theorem A_inter_B_empty : setA ∩ setB = ∅ := by
  sorry

end A_inter_B_empty_l979_97957


namespace max_not_divisible_by_3_l979_97990

theorem max_not_divisible_by_3 (s : Finset ℕ) (h₁ : s.card = 7) (h₂ : ∃ p ∈ s, p % 3 = 0) : 
  ∃t : Finset ℕ, t.card = 6 ∧ (∀ x ∈ t, x % 3 ≠ 0) ∧ (t ⊆ s) :=
sorry

end max_not_divisible_by_3_l979_97990


namespace total_books_l979_97927

-- Given conditions
def susan_books : Nat := 600
def lidia_books : Nat := 4 * susan_books

-- The theorem to prove
theorem total_books : susan_books + lidia_books = 3000 :=
by
  unfold susan_books lidia_books
  sorry

end total_books_l979_97927


namespace max_profit_at_150_l979_97949

-- Define the conditions
def purchase_price : ℕ := 80
def total_items : ℕ := 1000
def selling_price_initial : ℕ := 100
def sales_volume_decrease : ℕ := 5

-- The profit function
def profit (x : ℕ) : ℤ :=
  (selling_price_initial + x) * (total_items - sales_volume_decrease * x) - purchase_price * total_items

-- The statement to prove: the selling price of 150 yuan/item maximizes the profit at 32500 yuan.
theorem max_profit_at_150 : profit 50 = 32500 := by
  sorry

end max_profit_at_150_l979_97949


namespace eulers_formula_convex_polyhedron_l979_97955

theorem eulers_formula_convex_polyhedron :
  ∀ (V E F T H : ℕ),
  (V - E + F = 2) →
  (F = 24) →
  (E = (3 * T + 6 * H) / 2) →
  100 * H + 10 * T + V = 240 :=
by
  intros V E F T H h1 h2 h3
  sorry

end eulers_formula_convex_polyhedron_l979_97955


namespace part_a_part_b_part_c_l979_97947

def is_frameable (n : ℕ) : Prop :=
  n = 3 ∨ n = 4 ∨ n = 6

theorem part_a : is_frameable 3 ∧ is_frameable 4 ∧ is_frameable 6 :=
  sorry

theorem part_b (n : ℕ) (h : n ≥ 7) : ¬ is_frameable n :=
  sorry

theorem part_c : ¬ is_frameable 5 :=
  sorry

end part_a_part_b_part_c_l979_97947


namespace ab_calculation_l979_97904

noncomputable def triangle_area (a b : ℝ) : ℝ :=
  (1 / 2) * (4 / a) * (4 / b)

theorem ab_calculation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : triangle_area a b = 4) : a * b = 2 :=
by
  sorry

end ab_calculation_l979_97904


namespace range_of_m_l979_97984

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ (x^3 - 3 * x + m = 0)) → (m ≥ -2 ∧ m ≤ 2) :=
sorry

end range_of_m_l979_97984


namespace prime_factor_of_sum_of_four_consecutive_integers_l979_97963

theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by 
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_l979_97963


namespace german_russian_students_l979_97968

open Nat

theorem german_russian_students (G R : ℕ) (G_cap_R : ℕ) 
  (h_total : 1500 = G + R - G_cap_R)
  (hG_lb : 1125 ≤ G) (hG_ub : G ≤ 1275)
  (hR_lb : 375 ≤ R) (hR_ub : R ≤ 525) :
  300 = (max (G_cap_R) - min (G_cap_R)) :=
by
  -- Proof would go here
  sorry

end german_russian_students_l979_97968


namespace center_circle_sum_eq_neg1_l979_97932

theorem center_circle_sum_eq_neg1 
  (h k : ℝ) 
  (h_center : ∀ x y, (x - h)^2 + (y - k)^2 = 22) 
  (circle_eq : ∀ x y, x^2 + y^2 = 4*x - 6*y + 9) : 
  h + k = -1 := 
by 
  sorry

end center_circle_sum_eq_neg1_l979_97932


namespace find_P_Q_l979_97926

noncomputable def P := 11 / 3
noncomputable def Q := -2 / 3

theorem find_P_Q :
  ∀ x : ℝ, x ≠ 7 → x ≠ -2 →
    (3 * x + 12) / (x ^ 2 - 5 * x - 14) = P / (x - 7) + Q / (x + 2) :=
by
  intros x hx1 hx2
  dsimp [P, Q]  -- Unfold the definitions of P and Q
  -- The actual proof would go here, but we are skipping it
  sorry

end find_P_Q_l979_97926


namespace question1_question2_l979_97980

-- Question 1
theorem question1 (a : ℝ) (h : a = 1 / 2) :
  let A := {x | -1 / 2 < x ∧ x < 2}
  let B := {x | 0 < x ∧ x < 1}
  A ∩ B = {x | 0 < x ∧ x < 1} :=
by
  sorry

-- Question 2
theorem question2 (a : ℝ) :
  let A := {x | a - 1 < x ∧ x < 2 * a + 1}
  let B := {x | 0 < x ∧ x < 1}
  (A ∩ B = ∅) ↔ (a ≤ -1/2 ∨ a ≥ 2) :=
by
  sorry

end question1_question2_l979_97980


namespace largest_interior_angle_l979_97989

theorem largest_interior_angle (x : ℝ) (h_ratio : (5*x + 4*x + 3*x = 360)) :
  let e1 := 3 * x
  let e2 := 4 * x
  let e3 := 5 * x
  let i1 := 180 - e1
  let i2 := 180 - e2
  let i3 := 180 - e3
  max i1 (max i2 i3) = 90 :=
sorry

end largest_interior_angle_l979_97989


namespace solve_repeating_decimals_sum_l979_97912

def repeating_decimals_sum : Prop :=
  let x := (1 : ℚ) / 3
  let y := (4 : ℚ) / 999
  let z := (5 : ℚ) / 9999
  x + y + z = 3378 / 9999

theorem solve_repeating_decimals_sum : repeating_decimals_sum := 
by 
  sorry

end solve_repeating_decimals_sum_l979_97912


namespace swimming_club_total_members_l979_97988

def valid_total_members (total : ℕ) : Prop :=
  ∃ (J S V : ℕ),
    3 * S = 2 * J ∧
    5 * V = 2 * S ∧
    total = J + S + V

theorem swimming_club_total_members :
  valid_total_members 58 := by
  sorry

end swimming_club_total_members_l979_97988


namespace roots_cubic_properties_l979_97945

theorem roots_cubic_properties (a b c : ℝ) 
    (h1 : ∀ x : ℝ, x^3 - 2 * x^2 + 3 * x - 4 = 0 → x = a ∨ x = b ∨ x = c)
    (h_sum : a + b + c = 2)
    (h_prod_sum : a * b + b * c + c * a = 3)
    (h_prod : a * b * c = 4) :
  a^3 + b^3 + c^3 = 2 := by
  sorry

end roots_cubic_properties_l979_97945


namespace max_single_painted_faces_l979_97918

theorem max_single_painted_faces (n : ℕ) (hn : n = 64) :
  ∃ max_cubes : ℕ, max_cubes = 32 := 
sorry

end max_single_painted_faces_l979_97918


namespace books_selection_l979_97969

theorem books_selection 
  (num_mystery : ℕ)
  (num_fantasy : ℕ)
  (num_biographies : ℕ)
  (Hmystery : num_mystery = 5)
  (Hfantasy : num_fantasy = 4)
  (Hbiographies : num_biographies = 6) :
  (num_mystery * num_fantasy * num_biographies = 120) :=
by
  -- Proof goes here
  sorry

end books_selection_l979_97969


namespace find_original_price_l979_97940

noncomputable def original_price_per_bottle (P : ℝ) : Prop :=
  let discounted_price := 0.80 * P
  let final_price_per_bottle := discounted_price - 2.00
  3 * final_price_per_bottle = 30

theorem find_original_price : ∃ P : ℝ, original_price_per_bottle P ∧ P = 15 :=
by
  sorry

end find_original_price_l979_97940


namespace selena_book_pages_l979_97951

variable (S : ℕ)
variable (H : ℕ)

theorem selena_book_pages (cond1 : H = S / 2 - 20) (cond2 : H = 180) : S = 400 :=
by
  sorry

end selena_book_pages_l979_97951


namespace total_selling_price_correct_l979_97933

-- Define the given conditions
def cost_price_per_metre : ℝ := 72
def loss_per_metre : ℝ := 12
def total_metres_of_cloth : ℝ := 200

-- Define the selling price per metre
def selling_price_per_metre : ℝ := cost_price_per_metre - loss_per_metre

-- Define the total selling price
def total_selling_price : ℝ := selling_price_per_metre * total_metres_of_cloth

-- The theorem we want to prove
theorem total_selling_price_correct : 
  total_selling_price = 12000 := 
by
  sorry

end total_selling_price_correct_l979_97933
