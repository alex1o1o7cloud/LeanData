import Mathlib

namespace NUMINAMATH_GPT_ratio_u_v_l2187_218715

variables {u v : ℝ}
variables (u_lt_v : u < v)
variables (h_triangle : triangle 15 12 9)
variables (inscribed_circle : is_inscribed_circle 15 12 9 u v)

theorem ratio_u_v : u / v = 1 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_u_v_l2187_218715


namespace NUMINAMATH_GPT_initial_pennies_l2187_218792

theorem initial_pennies (initial: ℕ) (h : initial + 93 = 191) : initial = 98 := by
  sorry

end NUMINAMATH_GPT_initial_pennies_l2187_218792


namespace NUMINAMATH_GPT_circumference_of_circle_x_l2187_218700

theorem circumference_of_circle_x (A_x A_y : ℝ) (r_x r_y C_x : ℝ)
  (h_area: A_x = A_y) (h_half_radius_y: r_y = 2 * 5)
  (h_area_y: A_y = Real.pi * r_y^2)
  (h_area_x: A_x = Real.pi * r_x^2)
  (h_circumference_x: C_x = 2 * Real.pi * r_x) :
  C_x = 20 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circumference_of_circle_x_l2187_218700


namespace NUMINAMATH_GPT_exponentiation_distributes_over_multiplication_l2187_218771

theorem exponentiation_distributes_over_multiplication (a b c : ℝ) : (a * b) ^ c = a ^ c * b ^ c := 
sorry

end NUMINAMATH_GPT_exponentiation_distributes_over_multiplication_l2187_218771


namespace NUMINAMATH_GPT_max_value_200_max_value_attained_l2187_218727

noncomputable def max_value (X Y Z : ℕ) : ℕ := 
  X * Y * Z + X * Y + Y * Z + Z * X

theorem max_value_200 (X Y Z : ℕ) (h : X + Y + Z = 15) : 
  max_value X Y Z ≤ 200 :=
sorry

theorem max_value_attained (X Y Z : ℕ) (h : X = 5) (h1 : Y = 5) (h2 : Z = 5) : 
  max_value X Y Z = 200 :=
sorry

end NUMINAMATH_GPT_max_value_200_max_value_attained_l2187_218727


namespace NUMINAMATH_GPT_delta_max_success_ratio_l2187_218713

theorem delta_max_success_ratio (y w x z : ℤ) (h1 : 360 + 240 = 600)
  (h2 : 0 < x ∧ x < y ∧ z < w)
  (h3 : y + w = 600)
  (h4 : (x : ℚ) / y < (200 : ℚ) / 360)
  (h5 : (z : ℚ) / w < (160 : ℚ) / 240)
  (h6 : (360 : ℚ) / 600 = 3 / 5)
  (h7 : (x + z) < 166) :
  (x + z : ℚ) / 600 ≤ 166 / 600 := 
sorry

end NUMINAMATH_GPT_delta_max_success_ratio_l2187_218713


namespace NUMINAMATH_GPT_class_average_l2187_218732

theorem class_average (n : ℕ) (h₁ : n = 100) (h₂ : 25 ≤ n) 
  (h₃ : 50 ≤ n) (h₄ : 25 * 80 + 50 * 65 + (n - 75) * 90 = 7500) :
  (25 * 80 + 50 * 65 + (n - 75) * 90) / n = 75 := 
by
  sorry

end NUMINAMATH_GPT_class_average_l2187_218732


namespace NUMINAMATH_GPT_least_integer_a_divisible_by_240_l2187_218758

theorem least_integer_a_divisible_by_240 (a : ℤ) (h1 : 240 ∣ a^3) : a ≥ 60 := by
  sorry

end NUMINAMATH_GPT_least_integer_a_divisible_by_240_l2187_218758


namespace NUMINAMATH_GPT_minimum_pieces_for_K_1997_l2187_218739

-- Definitions provided by the conditions in the problem.
def is_cube_shaped (n : ℕ) := ∃ (a : ℕ), n = a^3

def has_chocolate_coating (surface_area : ℕ) (n : ℕ) := 
  surface_area = 6 * n^2

def min_pieces (n K : ℕ) := n^3 / K

-- Expressing the proof problem in Lean 4.
theorem minimum_pieces_for_K_1997 {n : ℕ} (h_n : n = 1997) (H : ∀ (K : ℕ), K = 1997 ∧ K > 0) 
  (h_cube : is_cube_shaped n) (h_chocolate : has_chocolate_coating 6 n) :
  min_pieces 1997 1997 = 1997^3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_pieces_for_K_1997_l2187_218739


namespace NUMINAMATH_GPT_algebraic_expression_equality_l2187_218710

variable {x : ℝ}

theorem algebraic_expression_equality (h : x^2 + 3*x + 8 = 7) : 3*x^2 + 9*x - 2 = -5 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_equality_l2187_218710


namespace NUMINAMATH_GPT_CoastalAcademy_absent_percentage_l2187_218766

theorem CoastalAcademy_absent_percentage :
  ∀ (total_students boys girls : ℕ) (absent_boys_ratio absent_girls_ratio : ℚ),
    total_students = 120 →
    boys = 70 →
    girls = 50 →
    absent_boys_ratio = 1/7 →
    absent_girls_ratio = 1/5 →
    let absent_boys := absent_boys_ratio * boys
    let absent_girls := absent_girls_ratio * girls
    let total_absent := absent_boys + absent_girls
    let absent_percentage := total_absent / total_students * 100
    absent_percentage = 16.67 :=
  by
    intros total_students boys girls absent_boys_ratio absent_girls_ratio
           h1 h2 h3 h4 h5
    let absent_boys := absent_boys_ratio * boys
    let absent_girls := absent_girls_ratio * girls
    let total_absent := absent_boys + absent_girls
    let absent_percentage := total_absent / total_students * 100
    sorry

end NUMINAMATH_GPT_CoastalAcademy_absent_percentage_l2187_218766


namespace NUMINAMATH_GPT_probability_stopping_after_three_draws_l2187_218712

def draws : List (List ℕ) := [
  [2, 3, 2], [3, 2, 1], [2, 3, 0], [0, 2, 3], [1, 2, 3], [0, 2, 1], [1, 3, 2], [2, 2, 0], [0, 0, 1],
  [2, 3, 1], [1, 3, 0], [1, 3, 3], [2, 3, 1], [0, 3, 1], [3, 2, 0], [1, 2, 2], [1, 0, 3], [2, 3, 3]
]

def favorable_sequences (seqs : List (List ℕ)) : List (List ℕ) :=
  seqs.filter (λ seq => 0 ∈ seq ∧ 1 ∈ seq)

def probability_of_drawing_zhong_hua (seqs : List (List ℕ)) : ℚ :=
  (favorable_sequences seqs).length / seqs.length

theorem probability_stopping_after_three_draws :
  probability_of_drawing_zhong_hua draws = 5 / 18 := by
sorry

end NUMINAMATH_GPT_probability_stopping_after_three_draws_l2187_218712


namespace NUMINAMATH_GPT_value_of_f_at_neg_one_l2187_218738

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x^2

noncomputable def f (x : ℝ) (h : x ≠ 0) : ℝ := (2 - 3 * x^2) / x^2

theorem value_of_f_at_neg_one : f (-1) (by norm_num) = -1 := 
sorry

end NUMINAMATH_GPT_value_of_f_at_neg_one_l2187_218738


namespace NUMINAMATH_GPT_least_of_10_consecutive_odd_integers_average_154_l2187_218730

theorem least_of_10_consecutive_odd_integers_average_154 (x : ℤ)
  (h_avg : (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18)) / 10 = 154) :
  x = 145 :=
by 
  sorry

end NUMINAMATH_GPT_least_of_10_consecutive_odd_integers_average_154_l2187_218730


namespace NUMINAMATH_GPT_students_tried_out_l2187_218762

theorem students_tried_out (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ)
  (h1 : not_picked = 36) (h2 : groups = 4) (h3 : students_per_group = 7) :
  not_picked + groups * students_per_group = 64 :=
by
  sorry

end NUMINAMATH_GPT_students_tried_out_l2187_218762


namespace NUMINAMATH_GPT_min_value_ineq_l2187_218701

noncomputable def min_value (x y z : ℝ) := (1/x) + (1/y) + (1/z)

theorem min_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z ≥ 4.5 :=
sorry

end NUMINAMATH_GPT_min_value_ineq_l2187_218701


namespace NUMINAMATH_GPT_min_value_is_neg2032188_l2187_218780

noncomputable def min_expression_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_neq: x ≠ y) (h_cond: x + y + 1/x + 1/y = 2022) : ℝ :=
(x + 1/y) * (x + 1/y - 2016) + (y + 1/x) * (y + 1/x - 2016)

theorem min_value_is_neg2032188 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_neq: x ≠ y) (h_cond: x + y + 1/x + 1/y = 2022) :
  min_expression_value x y h_pos_x h_pos_y h_neq h_cond = -2032188 := 
sorry

end NUMINAMATH_GPT_min_value_is_neg2032188_l2187_218780


namespace NUMINAMATH_GPT_find_vector_l2187_218783

def line_r (t : ℝ) : ℝ × ℝ :=
  (2 + 5 * t, 3 - 2 * t)

def line_s (u : ℝ) : ℝ × ℝ :=
  (1 + 5 * u, -2 - 2 * u)

def is_projection (w1 w2 : ℝ) : Prop :=
  w1 - w2 = 3

theorem find_vector (w1 w2 : ℝ) (h_proj : is_projection w1 w2) :
  (w1, w2) = (-2, -5) :=
sorry

end NUMINAMATH_GPT_find_vector_l2187_218783


namespace NUMINAMATH_GPT_largest_five_digit_congruent_to_31_modulo_26_l2187_218706

theorem largest_five_digit_congruent_to_31_modulo_26 :
  ∃ x : ℕ, (10000 ≤ x ∧ x < 100000) ∧ x % 26 = 31 ∧ x = 99975 :=
by
  sorry

end NUMINAMATH_GPT_largest_five_digit_congruent_to_31_modulo_26_l2187_218706


namespace NUMINAMATH_GPT_Ben_more_new_shirts_than_Joe_l2187_218787

theorem Ben_more_new_shirts_than_Joe :
  ∀ (alex_shirts joe_shirts ben_shirts : ℕ),
    alex_shirts = 4 →
    joe_shirts = alex_shirts + 3 →
    ben_shirts = 15 →
    ben_shirts - joe_shirts = 8 :=
by
  intros alex_shirts joe_shirts ben_shirts
  intros h_alex h_joe h_ben
  sorry

end NUMINAMATH_GPT_Ben_more_new_shirts_than_Joe_l2187_218787


namespace NUMINAMATH_GPT_largest_by_changing_first_digit_l2187_218782

-- Define the original number
def original_number : ℝ := 0.7162534

-- Define the transformation that changes a specific digit to 8
def transform_to_8 (n : ℕ) (d : ℝ) : ℝ :=
  match n with
  | 1 => 0.8162534
  | 2 => 0.7862534
  | 3 => 0.7182534
  | 4 => 0.7168534
  | 5 => 0.7162834
  | 6 => 0.7162584
  | 7 => 0.7162538
  | _ => d

-- State the theorem
theorem largest_by_changing_first_digit :
  ∀ (n : ℕ), transform_to_8 1 original_number ≥ transform_to_8 n original_number :=
by
  sorry

end NUMINAMATH_GPT_largest_by_changing_first_digit_l2187_218782


namespace NUMINAMATH_GPT_dispatch_plans_count_l2187_218726

theorem dispatch_plans_count:
  -- conditions
  let total_athletes := 9
  let basketball_players := 5
  let soccer_players := 6
  let both_players := 2
  let only_basketball := 3
  let only_soccer := 4
  -- proof
  (both_players.choose 2 + both_players * only_basketball + both_players * only_soccer + only_basketball * only_soccer) = 28 :=
by
  sorry

end NUMINAMATH_GPT_dispatch_plans_count_l2187_218726


namespace NUMINAMATH_GPT_car_catches_truck_in_7_hours_l2187_218760

-- Definitions based on the conditions
def initial_distance := 175 -- initial distance in kilometers
def truck_speed := 40 -- speed of the truck in km/h
def car_initial_speed := 50 -- initial speed of the car in km/h
def car_speed_increase := 5 -- speed increase per hour for the car in km/h

-- The main statement to prove
theorem car_catches_truck_in_7_hours :
  ∃ n : ℕ, (n ≥ 0) ∧ 
  (car_initial_speed - truck_speed) * n + (car_speed_increase * n * (n - 1) / 2) = initial_distance :=
by
  existsi 7
  -- Check the equation for n = 7
  -- Simplify: car initial extra speed + sum of increase terms should equal initial distance
  -- (50 - 40) * 7 + 5 * 7 * 6 / 2 = 175
  -- (10) * 7 + 35 * 3 / 2 = 175
  -- 70 + 105 = 175
  sorry

end NUMINAMATH_GPT_car_catches_truck_in_7_hours_l2187_218760


namespace NUMINAMATH_GPT_minimum_dimes_l2187_218794

-- Given amounts in dollars
def value_of_dimes (n : ℕ) : ℝ := 0.10 * n
def value_of_nickels : ℝ := 0.50
def value_of_one_dollar_bill : ℝ := 1.0
def value_of_four_tens : ℝ := 40.0
def price_of_scarf : ℝ := 42.85

-- Prove the total value of the money is at least the price of the scarf implies n >= 14
theorem minimum_dimes (n : ℕ) :
  value_of_four_tens + value_of_one_dollar_bill + value_of_nickels + value_of_dimes n ≥ price_of_scarf → n ≥ 14 :=
by
  sorry

end NUMINAMATH_GPT_minimum_dimes_l2187_218794


namespace NUMINAMATH_GPT_max_value_m_l2187_218754

theorem max_value_m (m n : ℕ) (h : 8 * m + 9 * n = m * n + 6) : m ≤ 75 := 
sorry

end NUMINAMATH_GPT_max_value_m_l2187_218754


namespace NUMINAMATH_GPT_sum_of_geometric_sequence_first_9000_terms_l2187_218773

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_geometric_sequence_first_9000_terms_l2187_218773


namespace NUMINAMATH_GPT_simplify_expression_l2187_218714

theorem simplify_expression : 1 + (1 / (1 + (1 / (2 + 1)))) = 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2187_218714


namespace NUMINAMATH_GPT_bobs_sisters_mile_time_l2187_218717

theorem bobs_sisters_mile_time (bobs_current_time_minutes : ℕ) (bobs_current_time_seconds : ℕ) (improvement_percentage : ℝ) :
  bobs_current_time_minutes = 10 → bobs_current_time_seconds = 40 → improvement_percentage = 9.062499999999996 →
  bobs_sisters_time_minutes = 9 ∧ bobs_sisters_time_seconds = 42 :=
by
  -- Definitions from conditions
  let bobs_time_in_seconds := bobs_current_time_minutes * 60 + bobs_current_time_seconds
  let improvement_in_seconds := bobs_time_in_seconds * improvement_percentage / 100
  let target_time_in_seconds := bobs_time_in_seconds - improvement_in_seconds
  let bobs_sisters_time_minutes := target_time_in_seconds / 60
  let bobs_sisters_time_seconds := target_time_in_seconds % 60
  
  sorry

end NUMINAMATH_GPT_bobs_sisters_mile_time_l2187_218717


namespace NUMINAMATH_GPT_inequality_solution_set_l2187_218705

theorem inequality_solution_set (x : ℝ) : (x - 1 < 7) ∧ (3 * x + 1 ≥ -2) ↔ -1 ≤ x ∧ x < 8 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l2187_218705


namespace NUMINAMATH_GPT_binomial_coeff_sum_l2187_218796

theorem binomial_coeff_sum : 
  (Nat.choose 3 2) + (Nat.choose 4 2) + (Nat.choose 5 2) + (Nat.choose 6 2) + (Nat.choose 7 2) + (Nat.choose 8 2) = 83 := by
  sorry

end NUMINAMATH_GPT_binomial_coeff_sum_l2187_218796


namespace NUMINAMATH_GPT_bottles_recycled_l2187_218745

theorem bottles_recycled (start_bottles : ℕ) (recycle_ratio : ℕ) (answer : ℕ)
  (h_start : start_bottles = 256) (h_recycle : recycle_ratio = 4) : answer = 85 :=
sorry

end NUMINAMATH_GPT_bottles_recycled_l2187_218745


namespace NUMINAMATH_GPT_entree_cost_14_l2187_218755

theorem entree_cost_14 (D E : ℝ) (h1 : D + E = 23) (h2 : E = D + 5) : E = 14 :=
sorry

end NUMINAMATH_GPT_entree_cost_14_l2187_218755


namespace NUMINAMATH_GPT_min_x_y_l2187_218769

theorem min_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (x + 1) * (y + 1) = 9) : x + y ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_x_y_l2187_218769


namespace NUMINAMATH_GPT_value_of_6z_l2187_218748

theorem value_of_6z (x y z : ℕ) (h1 : 6 * z = 2 * x) (h2 : x + y + z = 26) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) : 6 * z = 36 :=
by
  sorry

end NUMINAMATH_GPT_value_of_6z_l2187_218748


namespace NUMINAMATH_GPT_dealer_profit_percentage_l2187_218759

noncomputable def profit_percentage (cp_total : ℝ) (cp_count : ℝ) (sp_total : ℝ) (sp_count : ℝ) : ℝ :=
  let cp_per_article := cp_total / cp_count
  let sp_per_article := sp_total / sp_count
  let profit_per_article := sp_per_article - cp_per_article
  let profit_percentage := (profit_per_article / cp_per_article) * 100
  profit_percentage

theorem dealer_profit_percentage :
  profit_percentage 25 15 38 12 = 89.99 := by
  sorry

end NUMINAMATH_GPT_dealer_profit_percentage_l2187_218759


namespace NUMINAMATH_GPT_decrease_in_average_salary_l2187_218785

-- Define the conditions
variable (I : ℕ := 20)
variable (L : ℕ := 10)
variable (initial_wage_illiterate : ℕ := 25)
variable (new_wage_illiterate : ℕ := 10)

-- Define the theorem statement
theorem decrease_in_average_salary :
  (I * (initial_wage_illiterate - new_wage_illiterate)) / (I + L) = 10 := by
  sorry

end NUMINAMATH_GPT_decrease_in_average_salary_l2187_218785


namespace NUMINAMATH_GPT_determine_k_value_l2187_218746

theorem determine_k_value : (5 ^ 1002 + 6 ^ 1001) ^ 2 - (5 ^ 1002 - 6 ^ 1001) ^ 2 = 24 * 30 ^ 1001 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_value_l2187_218746


namespace NUMINAMATH_GPT_max_value_of_expression_l2187_218721

theorem max_value_of_expression (x : Real) :
  (x^4 / (x^8 + 2 * x^6 - 3 * x^4 + 5 * x^3 + 8 * x^2 + 5 * x + 25)) ≤ (1 / 15) :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2187_218721


namespace NUMINAMATH_GPT_train_distance_in_2_hours_l2187_218788

theorem train_distance_in_2_hours :
  (∀ (t : ℕ), t = 90 → (1 / ↑t) * 7200 = 80) :=
by
  sorry

end NUMINAMATH_GPT_train_distance_in_2_hours_l2187_218788


namespace NUMINAMATH_GPT_probability_of_same_color_is_correct_l2187_218799

-- Define the parameters for balls in the bag
def green_balls : ℕ := 8
def red_balls : ℕ := 6
def blue_balls : ℕ := 1
def total_balls : ℕ := green_balls + red_balls + blue_balls

-- Define the probabilities of drawing each color
def prob_green : ℚ := green_balls / total_balls
def prob_red : ℚ := red_balls / total_balls
def prob_blue : ℚ := blue_balls / total_balls

-- Define the probability of drawing two balls of the same color
def prob_same_color : ℚ :=
  prob_green^2 + prob_red^2 + prob_blue^2

theorem probability_of_same_color_is_correct :
  prob_same_color = 101 / 225 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_same_color_is_correct_l2187_218799


namespace NUMINAMATH_GPT_evens_minus_odds_equal_40_l2187_218756

-- Define the sum of even integers from 2 to 80
def sum_evens : ℕ := (List.range' 2 40).sum

-- Define the sum of odd integers from 1 to 79
def sum_odds : ℕ := (List.range' 1 40).sum

-- Define the main theorem to prove
theorem evens_minus_odds_equal_40 : sum_evens - sum_odds = 40 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_evens_minus_odds_equal_40_l2187_218756


namespace NUMINAMATH_GPT_area_of_triangle_is_23_over_10_l2187_218757

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  1/2 * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem area_of_triangle_is_23_over_10 :
  let A : ℚ × ℚ := (3, 3)
  let B : ℚ × ℚ := (5, 3)
  let C : ℚ × ℚ := (21 / 5, 19 / 5)
  area_of_triangle A.1 A.2 B.1 B.2 C.1 C.2 = 23 / 10 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_is_23_over_10_l2187_218757


namespace NUMINAMATH_GPT_global_chess_tournament_total_games_global_chess_tournament_player_wins_l2187_218763

theorem global_chess_tournament_total_games (num_players : ℕ) (h200 : num_players = 200) :
  (num_players * (num_players - 1)) / 2 = 19900 := by
  sorry

theorem global_chess_tournament_player_wins (num_players losses : ℕ) 
  (h200 : num_players = 200) (h30 : losses = 30) :
  (num_players - 1) - losses = 169 := by
  sorry

end NUMINAMATH_GPT_global_chess_tournament_total_games_global_chess_tournament_player_wins_l2187_218763


namespace NUMINAMATH_GPT_hundreds_digit_even_l2187_218776

-- Define the given conditions
def units_digit (n : ℕ) : ℕ := n % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- The main theorem to prove
theorem hundreds_digit_even (x : ℕ) 
  (h1 : units_digit (x*x) = 9) 
  (h2 : tens_digit (x*x) = 0) : ((x*x) / 100) % 2 = 0 :=
  sorry

end NUMINAMATH_GPT_hundreds_digit_even_l2187_218776


namespace NUMINAMATH_GPT_part1_inequality_part2_inequality_l2187_218798

theorem part1_inequality (x : ℝ) : 
  (3 * x - 2) / (x - 1) > 1 ↔ x > 1 ∨ x < 1 / 2 := 
by sorry

theorem part2_inequality (x a : ℝ) : 
  x^2 - a * x - 2 * a^2 < 0 ↔ 
  (a = 0 → False) ∧ 
  (a > 0 → -a < x ∧ x < 2 * a) ∧ 
  (a < 0 → 2 * a < x ∧ x < -a) := 
by sorry

end NUMINAMATH_GPT_part1_inequality_part2_inequality_l2187_218798


namespace NUMINAMATH_GPT_gcd_9157_2695_eq_1_l2187_218703

theorem gcd_9157_2695_eq_1 : Int.gcd 9157 2695 = 1 := 
by
  sorry

end NUMINAMATH_GPT_gcd_9157_2695_eq_1_l2187_218703


namespace NUMINAMATH_GPT_johns_friends_count_l2187_218722

-- Define the conditions
def total_cost : ℕ := 12100
def cost_per_person : ℕ := 1100

-- Define the theorem to prove the number of friends John is going with
theorem johns_friends_count (total_cost cost_per_person : ℕ) (h1 : total_cost = 12100) (h2 : cost_per_person = 1100) : (total_cost / cost_per_person) - 1 = 10 := by
  -- Providing the proof is not required, so we use sorry to skip it
  sorry

end NUMINAMATH_GPT_johns_friends_count_l2187_218722


namespace NUMINAMATH_GPT_tim_fewer_apples_l2187_218778

theorem tim_fewer_apples (martha_apples : ℕ) (harry_apples : ℕ) (tim_apples : ℕ) (H1 : martha_apples = 68) (H2 : harry_apples = 19) (H3 : harry_apples * 2 = tim_apples) : martha_apples - tim_apples = 30 :=
by
  sorry

end NUMINAMATH_GPT_tim_fewer_apples_l2187_218778


namespace NUMINAMATH_GPT_quadratic_polynomial_with_conditions_l2187_218729

theorem quadratic_polynomial_with_conditions :
  ∃ (a b c : ℝ), 
  (∀ x : ℂ, x = -3 - 4 * Complex.I ∨ x = -3 + 4 * Complex.I → a * x^2 + b * x + c = 0)
  ∧ b = -10 
  ∧ a = -5/3 
  ∧ c = -125/3 := 
sorry

end NUMINAMATH_GPT_quadratic_polynomial_with_conditions_l2187_218729


namespace NUMINAMATH_GPT_imaginary_part_of_fraction_l2187_218784

theorem imaginary_part_of_fraction (i : ℂ) (hi : i * i = -1) : (1 + i) / (1 - i) = 1 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_imaginary_part_of_fraction_l2187_218784


namespace NUMINAMATH_GPT_num_positive_integers_le_500_l2187_218735

-- Define a predicate to state that a number is a perfect square
def is_square (x : ℕ) : Prop := ∃ (k : ℕ), k * k = x

-- Define the main theorem
theorem num_positive_integers_le_500 (n : ℕ) :
  (∃ (ns : Finset ℕ), (∀ x ∈ ns, x ≤ 500 ∧ is_square (21 * x)) ∧ ns.card = 4) :=
by
  sorry

end NUMINAMATH_GPT_num_positive_integers_le_500_l2187_218735


namespace NUMINAMATH_GPT_find_alpha_l2187_218791

-- Given conditions
variables (α β : ℝ)
axiom h1 : α + β = 11
axiom h2 : α * β = 24
axiom h3 : α > β

-- Theorems to prove
theorem find_alpha : α = 8 :=
  sorry

end NUMINAMATH_GPT_find_alpha_l2187_218791


namespace NUMINAMATH_GPT_sequence_formula_l2187_218781

theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_sum: ∀ n : ℕ, n ≥ 2 → S n = n^2 * a n)
  (h_a1 : a 1 = 1) : ∀ n : ℕ, n ≥ 2 → a n = 2 / (n * (n + 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_formula_l2187_218781


namespace NUMINAMATH_GPT_smallest_y_absolute_value_equation_l2187_218736

theorem smallest_y_absolute_value_equation :
  ∃ y : ℚ, (|5 * y - 9| = 55) ∧ y = -46 / 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_y_absolute_value_equation_l2187_218736


namespace NUMINAMATH_GPT_distinct_real_numbers_eq_l2187_218708

theorem distinct_real_numbers_eq (x : ℝ) :
  (x^2 - 7)^2 + 2 * x^2 = 33 → 
  (∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                    {a, b, c, d} = {x | (x^2 - 7)^2 + 2 * x^2 = 33}) :=
sorry

end NUMINAMATH_GPT_distinct_real_numbers_eq_l2187_218708


namespace NUMINAMATH_GPT_competition_participants_l2187_218711

theorem competition_participants (n : ℕ) :
    (100 < n ∧ n < 200) ∧
    (n % 4 = 2) ∧
    (n % 5 = 2) ∧
    (n % 6 = 2)
    → (n = 122 ∨ n = 182) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_competition_participants_l2187_218711


namespace NUMINAMATH_GPT_domain_f_l2187_218743

open Real Set

noncomputable def f (x : ℝ) : ℝ := log (x + 1) + (x - 2) ^ 0

theorem domain_f :
  (∃ x : ℝ, f x = f x) ↔ (∀ x, (x > -1 ∧ x ≠ 2) ↔ (x ∈ Ioo (-1 : ℝ) 2 ∨ x ∈ Ioi 2)) :=
by
  sorry

end NUMINAMATH_GPT_domain_f_l2187_218743


namespace NUMINAMATH_GPT_mixed_groups_count_l2187_218786

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end NUMINAMATH_GPT_mixed_groups_count_l2187_218786


namespace NUMINAMATH_GPT_complex_prod_eq_l2187_218765

theorem complex_prod_eq (x y z : ℂ) (h1 : x * y + 6 * y = -24) (h2 : y * z + 6 * z = -24) (h3 : z * x + 6 * x = -24) :
  x * y * z = 144 :=
by
  sorry

end NUMINAMATH_GPT_complex_prod_eq_l2187_218765


namespace NUMINAMATH_GPT_part_a_part_b_l2187_218731

-- Definition for the number of triangles when the n-gon is divided using non-intersecting diagonals
theorem part_a (n : ℕ) (h : n ≥ 3) : 
  ∃ k, k = n - 2 := 
sorry

-- Definition for the number of diagonals when the n-gon is divided using non-intersecting diagonals
theorem part_b (n : ℕ) (h : n ≥ 3) : 
  ∃ l, l = n - 3 := 
sorry

end NUMINAMATH_GPT_part_a_part_b_l2187_218731


namespace NUMINAMATH_GPT_evaluate_expression_l2187_218744

theorem evaluate_expression : 
  let a := 3 * 5 * 6
  let b := 1 / 3 + 1 / 5 + 1 / 6
  a * b = 63 := 
by
  let a := 3 * 5 * 6
  let b := 1 / 3 + 1 / 5 + 1 / 6
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2187_218744


namespace NUMINAMATH_GPT_factor_expression_l2187_218716

theorem factor_expression (x : ℝ) : 35 * x ^ 13 + 245 * x ^ 26 = 35 * x ^ 13 * (1 + 7 * x ^ 13) :=
by {
  sorry
}

end NUMINAMATH_GPT_factor_expression_l2187_218716


namespace NUMINAMATH_GPT_total_spent_two_years_l2187_218777

def home_game_price : ℕ := 60
def away_game_price : ℕ := 75
def home_playoff_price : ℕ := 120
def away_playoff_price : ℕ := 100

def this_year_home_games : ℕ := 2
def this_year_away_games : ℕ := 2
def this_year_home_playoff_games : ℕ := 1
def this_year_away_playoff_games : ℕ := 0

def last_year_home_games : ℕ := 6
def last_year_away_games : ℕ := 3
def last_year_home_playoff_games : ℕ := 1
def last_year_away_playoff_games : ℕ := 1

def calculate_total_cost : ℕ :=
  let this_year_cost := this_year_home_games * home_game_price + this_year_away_games * away_game_price + this_year_home_playoff_games * home_playoff_price + this_year_away_playoff_games * away_playoff_price
  let last_year_cost := last_year_home_games * home_game_price + last_year_away_games * away_game_price + last_year_home_playoff_games * home_playoff_price + last_year_away_playoff_games * away_playoff_price
  this_year_cost + last_year_cost

theorem total_spent_two_years : calculate_total_cost = 1195 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_two_years_l2187_218777


namespace NUMINAMATH_GPT_range_of_a_l2187_218790

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ)
  (h1 : ∀ (n : ℕ), a_seq n = if n < 6 then (1 / 2 - a) * n + 1 else a ^ (n - 5))
  (h2 : ∀ (n : ℕ), n > 0 → a_seq n > a_seq (n + 1)) :
  (1 / 2 : ℝ) < a ∧ a < (7 / 12 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2187_218790


namespace NUMINAMATH_GPT_functional_equation_to_odd_function_l2187_218774

variables (f : ℝ → ℝ)

theorem functional_equation_to_odd_function (h : ∀ x y : ℝ, f (x + y) = f x + f y) :
  f 0 = 0 ∧ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_to_odd_function_l2187_218774


namespace NUMINAMATH_GPT_profit_percentage_is_60_l2187_218737

variable (SellingPrice CostPrice : ℝ)

noncomputable def Profit : ℝ := SellingPrice - CostPrice

noncomputable def ProfitPercentage : ℝ := (Profit SellingPrice CostPrice / CostPrice) * 100

theorem profit_percentage_is_60
  (h1 : SellingPrice = 400)
  (h2 : CostPrice = 250) :
  ProfitPercentage SellingPrice CostPrice = 60 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_60_l2187_218737


namespace NUMINAMATH_GPT_find_original_number_of_men_l2187_218779

theorem find_original_number_of_men (x : ℕ) (h1 : x * 12 = (x - 6) * 14) : x = 42 :=
  sorry

end NUMINAMATH_GPT_find_original_number_of_men_l2187_218779


namespace NUMINAMATH_GPT_solve_system_equations_l2187_218793

theorem solve_system_equations (a b c x y z : ℝ) (h1 : x + y + z = 0)
(h2 : c * x + a * y + b * z = 0)
(h3 : (x + b)^2 + (y + c)^2 + (z + a)^2 = a^2 + b^2 + c^2)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
(x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = a - b ∧ y = b - c ∧ z = c - a) := 
sorry

end NUMINAMATH_GPT_solve_system_equations_l2187_218793


namespace NUMINAMATH_GPT_gas_pressure_in_final_container_l2187_218740

variable (k : ℝ) (p_initial p_second p_final : ℝ) (v_initial v_second v_final v_half : ℝ)

theorem gas_pressure_in_final_container 
  (h1 : v_initial = 3.6)
  (h2 : p_initial = 6)
  (h3 : v_second = 7.2)
  (h4 : v_final = 3.6)
  (h5 : v_half = v_second / 2)
  (h6 : p_initial * v_initial = k)
  (h7 : p_second * v_second = k)
  (h8 : p_final * v_final = k) :
  p_final = 6 := 
sorry

end NUMINAMATH_GPT_gas_pressure_in_final_container_l2187_218740


namespace NUMINAMATH_GPT_f_decreasing_f_odd_l2187_218797

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f (a + b) = f a + f b

axiom negativity (x : ℝ) (h_pos : 0 < x) : f x < 0

theorem f_decreasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2 :=
by
  intros x1 x2 h
  sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intros x
  sorry

end NUMINAMATH_GPT_f_decreasing_f_odd_l2187_218797


namespace NUMINAMATH_GPT_smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l2187_218725

theorem smallest_positive_perfect_square_divisible_by_5_and_6_is_900 :
  ∃ n : ℕ, 0 < n ∧ (n ^ 2) % 5 = 0 ∧ (n ^ 2) % 6 = 0 ∧ (n ^ 2 = 900) := by
  sorry

end NUMINAMATH_GPT_smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l2187_218725


namespace NUMINAMATH_GPT_perimeter_rectangle_l2187_218718

-- Defining the width and length of the rectangle based on the conditions
def width (a : ℝ) := a
def length (a : ℝ) := 2 * a + 1

-- Statement of the problem: proving the perimeter
theorem perimeter_rectangle (a : ℝ) :
  let W := width a
  let L := length a
  2 * W + 2 * L = 6 * a + 2 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_rectangle_l2187_218718


namespace NUMINAMATH_GPT_james_beats_per_week_l2187_218720

def beats_per_minute := 200
def hours_per_day := 2
def days_per_week := 7

def beats_per_week (beats_per_minute: ℕ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℕ :=
  (beats_per_minute * hours_per_day * 60) * days_per_week

theorem james_beats_per_week : beats_per_week beats_per_minute hours_per_day days_per_week = 168000 := by
  sorry

end NUMINAMATH_GPT_james_beats_per_week_l2187_218720


namespace NUMINAMATH_GPT_smallest_positive_integer_l2187_218709

theorem smallest_positive_integer (n : ℕ) : 
  (∃ m : ℕ, (4410 * n = m^2)) → n = 10 := 
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l2187_218709


namespace NUMINAMATH_GPT_negation_of_original_prop_l2187_218751

variable (a : ℝ)
def original_prop (x : ℝ) : Prop := x^2 + a * x + 1 < 0

theorem negation_of_original_prop :
  ¬ (∃ x : ℝ, original_prop a x) ↔ ∀ x : ℝ, ¬ original_prop a x :=
by sorry

end NUMINAMATH_GPT_negation_of_original_prop_l2187_218751


namespace NUMINAMATH_GPT_cylinder_volume_in_sphere_l2187_218761

theorem cylinder_volume_in_sphere 
  (h_c : ℝ) (d_s : ℝ) : 
  (h_c = 1) → (d_s = 2) → 
  (π * (d_s / 2)^2 * (h_c / 2) = π / 2) :=
by 
  intros h_c_eq h_s_eq
  sorry

end NUMINAMATH_GPT_cylinder_volume_in_sphere_l2187_218761


namespace NUMINAMATH_GPT_basketball_team_girls_l2187_218753

theorem basketball_team_girls (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : B + (1 / 3) * G = 18) : 
  G = 18 :=
by
  have h3 : G - (1 / 3) * G = 30 - 18 := by sorry
  have h4 : (2 / 3) * G = 12 := by sorry
  have h5 : G = 12 * (3 / 2) := by sorry
  have h6 : G = 18 := by sorry
  exact h6

end NUMINAMATH_GPT_basketball_team_girls_l2187_218753


namespace NUMINAMATH_GPT_max_extra_time_matches_l2187_218750

theorem max_extra_time_matches (number_teams : ℕ) 
    (points_win : ℕ) (points_lose : ℕ) 
    (points_win_extra : ℕ) (points_lose_extra : ℕ) 
    (total_matches_2016 : number_teams = 2016)
    (pts_win_3 : points_win = 3)
    (pts_lose_0 : points_lose = 0)
    (pts_win_extra_2 : points_win_extra = 2)
    (pts_lose_extra_1 : points_lose_extra = 1) :
    ∃ N, N = 1512 := 
by {
  sorry
}

end NUMINAMATH_GPT_max_extra_time_matches_l2187_218750


namespace NUMINAMATH_GPT_bus_capacity_l2187_218772

def seats_available_on_left := 15
def seats_available_diff := 3
def people_per_seat := 3
def back_seat_capacity := 7

theorem bus_capacity : 
  (seats_available_on_left * people_per_seat) + 
  ((seats_available_on_left - seats_available_diff) * people_per_seat) + 
  back_seat_capacity = 88 := 
by 
  sorry

end NUMINAMATH_GPT_bus_capacity_l2187_218772


namespace NUMINAMATH_GPT_books_borrowed_l2187_218767

theorem books_borrowed (initial_books : ℕ) (additional_books : ℕ) (remaining_books : ℕ) : 
  initial_books = 300 → 
  additional_books = 10 * 5 → 
  remaining_books = 210 → 
  initial_books + additional_books - remaining_books = 140 :=
by
  intros h1 h2 h3
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_books_borrowed_l2187_218767


namespace NUMINAMATH_GPT_parabola_range_m_l2187_218704

noncomputable def parabola (m : ℝ) (x : ℝ) : ℝ := x^2 - (4*m + 1)*x + (2*m - 1)

theorem parabola_range_m (m : ℝ) :
  (∀ x : ℝ, parabola m x = 0 → (1 < x ∧ x < 2) ∨ (x < 1 ∨ x > 2)) ∧
  parabola m 0 < -1/2 →
  1/6 < m ∧ m < 1/4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_range_m_l2187_218704


namespace NUMINAMATH_GPT_find_subtracted_number_l2187_218741

theorem find_subtracted_number (x y : ℤ) (h1 : x = 129) (h2 : 2 * x - y = 110) : y = 148 := by
  have hx : 2 * 129 - y = 110 := by
    rw [h1] at h2
    exact h2
  linarith

end NUMINAMATH_GPT_find_subtracted_number_l2187_218741


namespace NUMINAMATH_GPT_change_in_max_value_l2187_218702

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem change_in_max_value (a b c : ℝ) (h1 : -b^2 / (4 * (a + 1)) + c = -b^2 / (4 * a) + c + 27 / 2)
  (h2 : -b^2 / (4 * (a - 4)) + c = -b^2 / (4 * a) + c - 9) :
  -b^2 / (4 * (a - 2)) + c = -b^2 / (4 * a) + c - 27 / 4 :=
by
  sorry

end NUMINAMATH_GPT_change_in_max_value_l2187_218702


namespace NUMINAMATH_GPT_range_of_m_plus_n_l2187_218728

theorem range_of_m_plus_n (m n : ℝ)
  (tangent_condition : (∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 = 1)) :
  m + n ∈ (Set.Iic (2 - 2*Real.sqrt 2) ∪ Set.Ici (2 + 2*Real.sqrt 2)) :=
sorry

end NUMINAMATH_GPT_range_of_m_plus_n_l2187_218728


namespace NUMINAMATH_GPT_sequence_a_n_l2187_218719

-- Given conditions from the problem
variable {a : ℕ → ℕ}
variable (S : ℕ → ℕ)
variable (n : ℕ)

-- The sum of the first n terms of the sequence is given by S_n
axiom sum_Sn : ∀ n : ℕ, n > 0 → S n = 2 * n * n

-- Definition of a_n, the nth term of the sequence
def a_n (n : ℕ) : ℕ :=
  if n = 1 then
    S 1
  else
    S n - S (n - 1)

-- Prove that a_n = 4n - 2 for all n > 0.
theorem sequence_a_n (n : ℕ) (h : n > 0) : a_n S n = 4 * n - 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_n_l2187_218719


namespace NUMINAMATH_GPT_work_efficiency_ratio_l2187_218723

-- Define the problem conditions and the ratio we need to prove.
theorem work_efficiency_ratio :
  (∃ (a b : ℝ), b = 1 / 18 ∧ (a + b) = 1 / 12 ∧ (a / b) = 1 / 2) :=
by {
  -- Definitions and variables can be listed if necessary
  -- a : ℝ
  -- b : ℝ
  -- Assume conditions
  sorry
}

end NUMINAMATH_GPT_work_efficiency_ratio_l2187_218723


namespace NUMINAMATH_GPT_range_of_k_l2187_218764

theorem range_of_k (k : ℝ) : (∀ x : ℝ, 2 * k * x^2 + k * x + 1 / 2 ≥ 0) → k ∈ Set.Ioc 0 4 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_k_l2187_218764


namespace NUMINAMATH_GPT_skyscraper_anniversary_l2187_218752

theorem skyscraper_anniversary (current_year_event future_happens_year target_anniversary_year : ℕ) :
  current_year_event + future_happens_year = target_anniversary_year - 5 →
  target_anniversary_year > current_year_event →
  future_happens_year = 95 := 
by
  sorry

-- Definitions for conditions:
def current_year_event := 100
def future_happens_year := 95
def target_anniversary_year := 200

end NUMINAMATH_GPT_skyscraper_anniversary_l2187_218752


namespace NUMINAMATH_GPT_tara_had_more_l2187_218724

theorem tara_had_more (M T X : ℕ) (h1 : T = 15) (h2 : M + T = 26) (h3 : T = M + X) : X = 4 :=
by 
  sorry

end NUMINAMATH_GPT_tara_had_more_l2187_218724


namespace NUMINAMATH_GPT_zain_has_80_coins_l2187_218770

theorem zain_has_80_coins (emerie_quarters emerie_dimes emerie_nickels emerie_pennies emerie_half_dollars : ℕ)
  (h_quarters : emerie_quarters = 6) 
  (h_dimes : emerie_dimes = 7)
  (h_nickels : emerie_nickels = 5)
  (h_pennies : emerie_pennies = 10) 
  (h_half_dollars : emerie_half_dollars = 2) : 
  10 + emerie_quarters + 10 + emerie_dimes + 10 + emerie_nickels + 10 + emerie_pennies + 10 + emerie_half_dollars = 80 :=
by
  sorry

end NUMINAMATH_GPT_zain_has_80_coins_l2187_218770


namespace NUMINAMATH_GPT_total_amount_given_away_l2187_218768

variable (numGrandchildren : ℕ)
variable (cardsPerGrandchild : ℕ)
variable (amountPerCard : ℕ)

theorem total_amount_given_away (h1 : numGrandchildren = 3) (h2 : cardsPerGrandchild = 2) (h3 : amountPerCard = 80) : 
  numGrandchildren * cardsPerGrandchild * amountPerCard = 480 := by
  sorry

end NUMINAMATH_GPT_total_amount_given_away_l2187_218768


namespace NUMINAMATH_GPT_maximize_a2_b2_c2_d2_l2187_218707

theorem maximize_a2_b2_c2_d2 
  (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 85)
  (h3 : ad + bc = 187)
  (h4 : cd = 110) :
  a^2 + b^2 + c^2 + d^2 ≤ 120 :=
sorry

end NUMINAMATH_GPT_maximize_a2_b2_c2_d2_l2187_218707


namespace NUMINAMATH_GPT_find_x_for_dot_product_l2187_218734

theorem find_x_for_dot_product :
  let a : (ℝ × ℝ) := (1, -1)
  let b : (ℝ × ℝ) := (2, x)
  (a.1 * b.1 + a.2 * b.2 = 1) ↔ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_dot_product_l2187_218734


namespace NUMINAMATH_GPT_smallest_k_for_distinct_real_roots_l2187_218795

noncomputable def discriminant (a b c : ℝ) := b^2 - 4 * a * c

theorem smallest_k_for_distinct_real_roots :
  ∃ k : ℤ, (k > 0) ∧ discriminant (k : ℝ) (-3) (-9/4) > 0 ∧ (∀ m : ℤ, discriminant (m : ℝ) (-3) (-9/4) > 0 → m ≥ k) := 
by
  sorry

end NUMINAMATH_GPT_smallest_k_for_distinct_real_roots_l2187_218795


namespace NUMINAMATH_GPT_valid_three_digit_numbers_l2187_218742

   noncomputable def three_digit_num_correct (A : ℕ) : Prop :=
     (100 ≤ A ∧ A < 1000) ∧ (1000000 + A = A * A)

   theorem valid_three_digit_numbers (A : ℕ) :
     three_digit_num_correct A → (A = 625 ∨ A = 376) :=
   by
     sorry
   
end NUMINAMATH_GPT_valid_three_digit_numbers_l2187_218742


namespace NUMINAMATH_GPT_problem_solution_l2187_218749

-- Definitions based on conditions
def valid_sequence (b : Fin 7 → Nat) : Prop :=
  (∀ i j : Fin 7, i ≤ j → b i ≥ b j) ∧ 
  (∀ i : Fin 7, b i ≤ 1500) ∧ 
  (∀ i : Fin 7, (b i + i) % 3 = 0)

-- The main theorem
theorem problem_solution :
  (∃ b : Fin 7 → Nat, valid_sequence b) →
  @Nat.choose 506 7 % 1000 = 506 :=
sorry

end NUMINAMATH_GPT_problem_solution_l2187_218749


namespace NUMINAMATH_GPT_min_value_x2_y2_z2_l2187_218789

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_x2_y2_z2_l2187_218789


namespace NUMINAMATH_GPT_other_root_correct_l2187_218733

noncomputable def other_root (p : ℝ) : ℝ :=
  let a := 3
  let c := -2
  let root1 := -1
  (-c / a) / root1

theorem other_root_correct (p : ℝ) (h_eq : 3 * (-1) ^ 2 + p * (-1) = 2) : other_root p = 2 / 3 :=
  by
    unfold other_root
    sorry

end NUMINAMATH_GPT_other_root_correct_l2187_218733


namespace NUMINAMATH_GPT_alyosha_possible_l2187_218747

theorem alyosha_possible (current_date : ℕ) (day_before_yesterday_age current_year_age next_year_age : ℕ) : 
  (next_year_age = 12 ∧ day_before_yesterday_age = 9 ∧ current_year_age = 12 - 1)
  → (current_date = 1 ∧ current_year_age = 11 → (∃ bday : ℕ, bday = 31)) := 
by
  sorry

end NUMINAMATH_GPT_alyosha_possible_l2187_218747


namespace NUMINAMATH_GPT_cube_difference_l2187_218775

theorem cube_difference (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 26) : a^3 - b^3 = 124 :=
by sorry

end NUMINAMATH_GPT_cube_difference_l2187_218775
