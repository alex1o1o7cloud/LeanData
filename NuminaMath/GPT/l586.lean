import Mathlib

namespace NUMINAMATH_GPT_maximum_marks_l586_58684

noncomputable def passing_mark (M : ℝ) : ℝ := 0.35 * M

theorem maximum_marks (M : ℝ) (h1 : passing_mark M = 210) : M = 600 :=
  by
  sorry

end NUMINAMATH_GPT_maximum_marks_l586_58684


namespace NUMINAMATH_GPT_words_on_each_page_l586_58643

theorem words_on_each_page (p : ℕ) (h : 150 * p ≡ 198 [MOD 221]) : p = 93 :=
sorry

end NUMINAMATH_GPT_words_on_each_page_l586_58643


namespace NUMINAMATH_GPT_count_multiples_200_to_400_l586_58687

def count_multiples_in_range (a b n : ℕ) : ℕ :=
  (b / n) - ((a + n - 1) / n) + 1

theorem count_multiples_200_to_400 :
  count_multiples_in_range 200 400 78 = 3 :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_200_to_400_l586_58687


namespace NUMINAMATH_GPT_initially_calculated_average_weight_l586_58603

-- Define the conditions
def num_boys : ℕ := 20
def correct_average_weight : ℝ := 58.7
def misread_weight : ℝ := 56
def correct_weight : ℝ := 62
def weight_difference : ℝ := correct_weight - misread_weight

-- State the goal
theorem initially_calculated_average_weight :
  let correct_total_weight := correct_average_weight * num_boys
  let initial_total_weight := correct_total_weight - weight_difference
  let initially_calculated_weight := initial_total_weight / num_boys
  initially_calculated_weight = 58.4 :=
by
  sorry

end NUMINAMATH_GPT_initially_calculated_average_weight_l586_58603


namespace NUMINAMATH_GPT_cistern_depth_l586_58626

theorem cistern_depth (h : ℝ) :
  (6 * 4 + 2 * (h * 6) + 2 * (h * 4) = 49) → (h = 1.25) :=
by
  sorry

end NUMINAMATH_GPT_cistern_depth_l586_58626


namespace NUMINAMATH_GPT_at_least_one_nonnegative_l586_58678

theorem at_least_one_nonnegative (x : ℝ) (a b : ℝ) (h1 : a = x^2 - 1) (h2 : b = 4 * x + 5) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_nonnegative_l586_58678


namespace NUMINAMATH_GPT_javier_average_hits_per_game_l586_58640

theorem javier_average_hits_per_game (total_games_first_part : ℕ) (average_hits_first_part : ℕ) 
  (remaining_games : ℕ) (average_hits_remaining : ℕ) : 
  total_games_first_part = 20 → average_hits_first_part = 2 → 
  remaining_games = 10 → average_hits_remaining = 5 →
  (total_games_first_part * average_hits_first_part + 
  remaining_games * average_hits_remaining) /
  (total_games_first_part + remaining_games) = 3 := 
by intros h1 h2 h3 h4;
   sorry

end NUMINAMATH_GPT_javier_average_hits_per_game_l586_58640


namespace NUMINAMATH_GPT_integer_diff_of_two_squares_l586_58676

theorem integer_diff_of_two_squares (m : ℤ) : 
  (∃ x y : ℤ, m = x^2 - y^2) ↔ (∃ k : ℤ, m ≠ 4 * k + 2) := by
  sorry

end NUMINAMATH_GPT_integer_diff_of_two_squares_l586_58676


namespace NUMINAMATH_GPT_sequence_increasing_range_l586_58681

theorem sequence_increasing_range (a : ℝ) (h : ∀ n : ℕ, (n - a) ^ 2 < (n + 1 - a) ^ 2) :
  a < 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_increasing_range_l586_58681


namespace NUMINAMATH_GPT_sale_in_third_month_l586_58639

theorem sale_in_third_month (s_1 s_2 s_4 s_5 s_6 : ℝ) (avg_sale : ℝ) (h1 : s_1 = 6435) (h2 : s_2 = 6927) (h4 : s_4 = 7230) (h5 : s_5 = 6562) (h6 : s_6 = 6191) (h_avg : avg_sale = 6700) :
  ∃ s_3 : ℝ, s_1 + s_2 + s_3 + s_4 + s_5 + s_6 = 6 * avg_sale ∧ s_3 = 6855 :=
by 
  sorry

end NUMINAMATH_GPT_sale_in_third_month_l586_58639


namespace NUMINAMATH_GPT_prime_gt_3_div_24_num_form_6n_plus_minus_1_div_24_l586_58688

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_gt_3_div_24 (p : ℕ) (hp : is_prime p) (h : p > 3) : 
  24 ∣ (p^2 - 1) :=
sorry

theorem num_form_6n_plus_minus_1_div_24 (n : ℕ) : 
  24 ∣ (6 * n + 1)^2 - 1 ∧ 24 ∣ (6 * n - 1)^2 - 1 :=
sorry

end NUMINAMATH_GPT_prime_gt_3_div_24_num_form_6n_plus_minus_1_div_24_l586_58688


namespace NUMINAMATH_GPT_unique_pair_not_opposite_l586_58680

def QuantumPair (a b : String): Prop := ∃ oppositeMeanings : Bool, a ≠ b ∧ oppositeMeanings

theorem unique_pair_not_opposite :
  ∃ (a b : String), 
    (a = "increase of 2 years" ∧ b = "decrease of 2 liters") ∧ 
    (¬ QuantumPair a b) :=
by 
  sorry

end NUMINAMATH_GPT_unique_pair_not_opposite_l586_58680


namespace NUMINAMATH_GPT_range_of_f_l586_58614

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos (x / 2))^2 + 
  Real.pi * Real.arcsin (x / 2) - 
  (Real.arcsin (x / 2))^2 + 
  (Real.pi^2 / 6) * (x^2 + 2 * x + 1)

theorem range_of_f (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 2) :
  ∃ y : ℝ, (f y) = x ∧  (Real.pi^2 / 4) ≤ y ∧ y ≤ (39 * Real.pi^2 / 96) := 
sorry

end NUMINAMATH_GPT_range_of_f_l586_58614


namespace NUMINAMATH_GPT_subtraction_result_l586_58610

theorem subtraction_result :
  let x := 567.89
  let y := 123.45
  (x - y) = 444.44 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_result_l586_58610


namespace NUMINAMATH_GPT_sum_of_below_avg_l586_58661

-- Define class averages
def a1 := 75
def a2 := 85
def a3 := 90
def a4 := 65

-- Define the overall average
def avg : ℚ := (a1 + a2 + a3 + a4) / 4

-- Define a predicate indicating if a class average is below the overall average
def below_avg (a : ℚ) : Prop := a < avg

-- The theorem to prove the required sum of averages below the overall average
theorem sum_of_below_avg : a1 < avg ∧ a4 < avg → a1 + a4 = 140 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_below_avg_l586_58661


namespace NUMINAMATH_GPT_correct_calculation_l586_58601

-- Definition of the conditions
def condition1 (a : ℕ) : Prop := a^2 * a^3 = a^6
def condition2 (a : ℕ) : Prop := (a^2)^10 = a^20
def condition3 (a : ℕ) : Prop := (2 * a) * (3 * a) = 6 * a
def condition4 (a : ℕ) : Prop := a^12 / a^2 = a^6

-- The main theorem to state that condition2 is the correct calculation
theorem correct_calculation (a : ℕ) : condition2 a :=
sorry

end NUMINAMATH_GPT_correct_calculation_l586_58601


namespace NUMINAMATH_GPT_length_of_rest_of_body_l586_58679

theorem length_of_rest_of_body (h : ℝ) (legs : ℝ) (head : ℝ) (rest_of_body : ℝ) :
  h = 60 → legs = (1 / 3) * h → head = (1 / 4) * h → rest_of_body = h - (legs + head) → rest_of_body = 25 := by
  sorry

end NUMINAMATH_GPT_length_of_rest_of_body_l586_58679


namespace NUMINAMATH_GPT_surface_area_bound_l586_58658

theorem surface_area_bound
  (a b c d : ℝ)
  (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) (h4: 0 ≤ d) 
  (h_quad: a + b + c > d) : 
  2 * (a * b + b * c + c * a) ≤ (a + b + c) ^ 2 - (d ^ 2) / 3 :=
sorry

end NUMINAMATH_GPT_surface_area_bound_l586_58658


namespace NUMINAMATH_GPT_length_of_bridge_l586_58697

theorem length_of_bridge (length_train : ℕ) (speed_train_kmh : ℕ) (crossing_time_sec : ℕ)
    (h_length_train : length_train = 125)
    (h_speed_train_kmh : speed_train_kmh = 45)
    (h_crossing_time_sec : crossing_time_sec = 30) : 
    ∃ (length_bridge : ℕ), length_bridge = 250 := by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l586_58697


namespace NUMINAMATH_GPT_mod_remainder_l586_58655

theorem mod_remainder (n : ℤ) (h : n % 5 = 3) : (4 * n - 5) % 5 = 2 := by
  sorry

end NUMINAMATH_GPT_mod_remainder_l586_58655


namespace NUMINAMATH_GPT_salary_increase_l586_58652

theorem salary_increase (new_salary increase : ℝ) (h_new : new_salary = 25000) (h_inc : increase = 5000) : 
  ((increase / (new_salary - increase)) * 100) = 25 :=
by
  -- We will write the proof to satisfy the requirement, but it is currently left out as per the instructions.
  sorry

end NUMINAMATH_GPT_salary_increase_l586_58652


namespace NUMINAMATH_GPT_sequence_expression_l586_58675

theorem sequence_expression (a : ℕ → ℝ) (h_base : a 1 = 2)
  (h_rec : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * (n + 1) * a n / (a n + n)) :
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = (n + 1) * 2^(n + 1) / (2^(n + 1) - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_expression_l586_58675


namespace NUMINAMATH_GPT_pentomino_reflectional_count_l586_58671

def is_reflectional (p : Pentomino) : Prop := sorry -- Define reflectional symmetry property
def is_rotational (p : Pentomino) : Prop := sorry -- Define rotational symmetry property

theorem pentomino_reflectional_count :
  ∀ (P : Finset Pentomino),
  P.card = 15 →
  (∃ (R : Finset Pentomino), R.card = 2 ∧ (∀ p ∈ R, is_rotational p ∧ ¬ is_reflectional p)) →
  (∃ (S : Finset Pentomino), S.card = 7 ∧ (∀ p ∈ S, is_reflectional p)) :=
by
  sorry -- Proof not required as per instructions

end NUMINAMATH_GPT_pentomino_reflectional_count_l586_58671


namespace NUMINAMATH_GPT_salary_increase_l586_58656

variable (S : ℝ) (P : ℝ)

theorem salary_increase (h1 : 1.16 * S = 406) (h2 : 350 + 350 * P = 420) : P * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_salary_increase_l586_58656


namespace NUMINAMATH_GPT_ratio_mark_to_jenna_l586_58637

-- Definitions based on the given conditions
def total_problems : ℕ := 20

def problems_angela : ℕ := 9
def problems_martha : ℕ := 2
def problems_jenna : ℕ := 4 * problems_martha - 2

def problems_completed : ℕ := problems_angela + problems_martha + problems_jenna
def problems_mark : ℕ := total_problems - problems_completed

-- The proof statement based on the question and conditions
theorem ratio_mark_to_jenna :
  (problems_mark : ℚ) / problems_jenna = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_mark_to_jenna_l586_58637


namespace NUMINAMATH_GPT_velocity_volleyball_league_members_l586_58648

theorem velocity_volleyball_league_members (total_cost : ℕ) (socks_cost t_shirt_cost cost_per_member members : ℕ)
  (h_socks_cost : socks_cost = 6)
  (h_t_shirt_cost : t_shirt_cost = socks_cost + 7)
  (h_cost_per_member : cost_per_member = 2 * (socks_cost + t_shirt_cost))
  (h_total_cost : total_cost = 3510)
  (h_total_cost_eq : total_cost = cost_per_member * members) :
  members = 92 :=
by
  sorry

end NUMINAMATH_GPT_velocity_volleyball_league_members_l586_58648


namespace NUMINAMATH_GPT_remainder_23_to_2047_mod_17_l586_58623

theorem remainder_23_to_2047_mod_17 :
  23^2047 % 17 = 11 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_23_to_2047_mod_17_l586_58623


namespace NUMINAMATH_GPT_range_of_function_l586_58663

theorem range_of_function (x y z : ℝ)
  (h : x^2 + y^2 + x - y = 1) :
  ∃ a b : ℝ, (a = (3 * Real.sqrt 6 + Real.sqrt 6) / 2) ∧ (b = (-3 * Real.sqrt 2 + Real.sqrt 6) / 2) ∧
    ∀ f : ℝ, f = (x - 1) * Real.cos z + (y + 1) * Real.sin z →
              b ≤ f ∧ f ≤ a := 
by
  sorry

end NUMINAMATH_GPT_range_of_function_l586_58663


namespace NUMINAMATH_GPT_ratio_of_weights_l586_58666

variable (x y : ℝ)

theorem ratio_of_weights (h : x + y = 7 * (x - y)) (h1 : x > y) : x / y = 4 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_weights_l586_58666


namespace NUMINAMATH_GPT_bristol_to_carlisle_routes_l586_58627

-- Given conditions
def r_bb := 6
def r_bs := 3
def r_sc := 2

-- The theorem we want to prove
theorem bristol_to_carlisle_routes :
  (r_bb * r_bs * r_sc) = 36 :=
by
  sorry

end NUMINAMATH_GPT_bristol_to_carlisle_routes_l586_58627


namespace NUMINAMATH_GPT_flower_position_after_50_beats_l586_58683

-- Define the number of students
def num_students : Nat := 7

-- Define the initial position of the flower
def initial_position : Nat := 1

-- Define the number of drum beats
def drum_beats : Nat := 50

-- Theorem stating that after 50 drum beats, the flower will be with the 2nd student
theorem flower_position_after_50_beats : 
  (initial_position + (drum_beats % num_students)) % num_students = 2 := by
  -- Start the proof (this part usually would contain the actual proof logic)
  sorry

end NUMINAMATH_GPT_flower_position_after_50_beats_l586_58683


namespace NUMINAMATH_GPT_maximize_profit_l586_58625

noncomputable def profit (x a : ℝ) : ℝ :=
  19 - 24 / (x + 2) - (3 / 2) * x

theorem maximize_profit (a : ℝ) (ha : 0 < a) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ a) ∧ 
  (if a ≥ 2 then x = 2 else x = a) :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l586_58625


namespace NUMINAMATH_GPT_find_possible_sets_C_l586_58612

open Set

def A : Set ℕ := {3, 4}
def B : Set ℕ := {0, 1, 2, 3, 4}
def possible_C_sets : Set (Set ℕ) :=
  { {3, 4}, {3, 4, 0}, {3, 4, 1}, {3, 4, 2}, {3, 4, 0, 1},
    {3, 4, 0, 2}, {3, 4, 1, 2}, {0, 1, 2, 3, 4} }

theorem find_possible_sets_C :
  {C : Set ℕ | A ⊆ C ∧ C ⊆ B} = possible_C_sets :=
by
  sorry

end NUMINAMATH_GPT_find_possible_sets_C_l586_58612


namespace NUMINAMATH_GPT_Robe_savings_l586_58667

-- Define the conditions and question in Lean 4
theorem Robe_savings 
  (repair_fee : ℕ)
  (corner_light_cost : ℕ)
  (brake_disk_cost : ℕ)
  (total_remaining_savings : ℕ)
  (total_savings_before : ℕ)
  (h1 : repair_fee = 10)
  (h2 : corner_light_cost = 2 * repair_fee)
  (h3 : brake_disk_cost = 3 * corner_light_cost)
  (h4 : total_remaining_savings = 480)
  (h5 : total_savings_before = total_remaining_savings + (repair_fee + corner_light_cost + 2 * brake_disk_cost)) :
  total_savings_before = 630 :=
by
  -- Proof steps to be filled
  sorry

end NUMINAMATH_GPT_Robe_savings_l586_58667


namespace NUMINAMATH_GPT_determine_value_of_a_l586_58632

theorem determine_value_of_a (a : ℝ) (h : 1 < a) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ a → 
  1 ≤ (1 / 2 * x^2 - x + 3 / 2) ∧ (1 / 2 * x^2 - x + 3 / 2) ≤ a) →
  a = 3 :=
by
  sorry

end NUMINAMATH_GPT_determine_value_of_a_l586_58632


namespace NUMINAMATH_GPT_tennis_tournament_total_rounds_l586_58635

theorem tennis_tournament_total_rounds
  (participants : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (pairs_formation : ℕ → ℕ)
  (single_points_award : ℕ → ℕ)
  (elimination_condition : ℕ → Prop)
  (tournament_continues : ℕ → Prop)
  (progression_condition : ℕ → ℕ → ℕ)
  (group_split : Π (n : ℕ), Π (k : ℕ), (ℕ × ℕ))
  (rounds_needed : ℕ) :
  participants = 1152 →
  points_win = 1 →
  points_loss = 0 →
  pairs_formation participants ≥ 0 →
  single_points_award participants ≥ 0 →
  (∀ p, p > 1 → participants / p > 0 → tournament_continues participants) →
  (∀ m n, progression_condition m n = n - m) →
  (group_split 1152 1024 = (1024, 128)) →
  rounds_needed = 14 :=
by
  sorry

end NUMINAMATH_GPT_tennis_tournament_total_rounds_l586_58635


namespace NUMINAMATH_GPT_mean_home_runs_correct_l586_58630

-- Define the total home runs in April
def total_home_runs_April : ℕ := 5 * 4 + 6 * 4 + 8 * 2 + 10

-- Define the total home runs in May
def total_home_runs_May : ℕ := 5 * 2 + 6 * 2 + 8 * 3 + 10 * 2 + 11

-- Define the total number of top hitters/players
def total_players : ℕ := 12

-- Define the total home runs over two months
def total_home_runs : ℕ := total_home_runs_April + total_home_runs_May

-- Calculate the mean number of home runs
def mean_home_runs : ℚ := total_home_runs / total_players

-- Prove that the calculated mean is equal to the expected result
theorem mean_home_runs_correct : mean_home_runs = 12.08 := by
  sorry

end NUMINAMATH_GPT_mean_home_runs_correct_l586_58630


namespace NUMINAMATH_GPT_pattern_proof_l586_58654

theorem pattern_proof (h1 : 1 = 6) (h2 : 2 = 36) (h3 : 3 = 363) (h4 : 4 = 364) (h5 : 5 = 365) : 36 = 3636 := by
  sorry

end NUMINAMATH_GPT_pattern_proof_l586_58654


namespace NUMINAMATH_GPT_ratio_sphere_locus_l586_58653

noncomputable def sphere_locus_ratio (r : ℝ) : ℝ :=
  let F1 := 2 * Real.pi * r^2 * (1 - Real.sqrt (2 / 3))
  let F2 := Real.pi * r^2 * (2 * Real.sqrt 3 / 3)
  F1 / F2

theorem ratio_sphere_locus (r : ℝ) (h : r > 0) : sphere_locus_ratio r = Real.sqrt 3 - 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_sphere_locus_l586_58653


namespace NUMINAMATH_GPT_max_b_squared_l586_58638

theorem max_b_squared (a b : ℤ) (h : (a + b) * (a + b) + a * (a + b) + b = 0) : b^2 ≤ 81 :=
sorry

end NUMINAMATH_GPT_max_b_squared_l586_58638


namespace NUMINAMATH_GPT_bottles_from_shop_c_correct_l586_58629

-- Definitions for the given conditions
def total_bottles := 550
def bottles_from_shop_a := 150
def bottles_from_shop_b := 180

-- Definition for the bottles from Shop C
def bottles_from_shop_c := total_bottles - (bottles_from_shop_a + bottles_from_shop_b)

-- The statement to prove
theorem bottles_from_shop_c_correct : bottles_from_shop_c = 220 :=
by
  -- proof will be filled later
  sorry

end NUMINAMATH_GPT_bottles_from_shop_c_correct_l586_58629


namespace NUMINAMATH_GPT_not_perfect_square_l586_58609

theorem not_perfect_square (n : ℤ) (hn : n > 4) : ¬ (∃ k : ℕ, n^2 - 3*n = k^2) :=
sorry

end NUMINAMATH_GPT_not_perfect_square_l586_58609


namespace NUMINAMATH_GPT_sum_and_difference_repeating_decimals_l586_58695

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_9 : ℚ := 1
noncomputable def repeating_decimal_3 : ℚ := 1 / 3

theorem sum_and_difference_repeating_decimals :
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_9 + repeating_decimal_3 = 2 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_sum_and_difference_repeating_decimals_l586_58695


namespace NUMINAMATH_GPT_evaluate_sqrt_log_expression_l586_58634

noncomputable def evaluate_log_expression : ℝ :=
  let log3 (x : ℝ) := Real.log x / Real.log 3
  let log4 (x : ℝ) := Real.log x / Real.log 4
  Real.sqrt (log3 8 + log4 8)

theorem evaluate_sqrt_log_expression : evaluate_log_expression = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_sqrt_log_expression_l586_58634


namespace NUMINAMATH_GPT_triangle_angle_equality_l586_58641

theorem triangle_angle_equality (A B C : ℝ) (h : ∃ (x : ℝ), x^2 - x * (Real.cos A * Real.cos B) - Real.cos (C / 2)^2 = 0 ∧ x = 1) : A = B :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_angle_equality_l586_58641


namespace NUMINAMATH_GPT_jogged_distance_is_13_point_5_l586_58672

noncomputable def jogger_distance (x t d : ℝ) : Prop :=
  d = x * t ∧
  d = (x + 3/4) * (3 * t / 4) ∧
  d = (x - 3/4) * (t + 3)

theorem jogged_distance_is_13_point_5:
  ∃ (x t d : ℝ), jogger_distance x t d ∧ d = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_jogged_distance_is_13_point_5_l586_58672


namespace NUMINAMATH_GPT_expression_value_l586_58689

def a : ℕ := 45
def b : ℕ := 18
def c : ℕ := 10

theorem expression_value :
  (a + b)^2 - (a^2 + b^2 + c) = 1610 := by
  sorry

end NUMINAMATH_GPT_expression_value_l586_58689


namespace NUMINAMATH_GPT_sequence_integers_l586_58642

theorem sequence_integers (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 1) 
  (h3 : ∀ n ≥ 3, a n = (a (n - 1))^2 + 2 / a (n - 2)) : ∀ n, ∃ k : ℤ, a n = k :=
sorry

end NUMINAMATH_GPT_sequence_integers_l586_58642


namespace NUMINAMATH_GPT_actors_in_one_hour_l586_58636

theorem actors_in_one_hour (actors_per_set : ℕ) (minutes_per_set : ℕ) (total_minutes : ℕ) :
  actors_per_set = 5 → minutes_per_set = 15 → total_minutes = 60 →
  (total_minutes / minutes_per_set) * actors_per_set = 20 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_actors_in_one_hour_l586_58636


namespace NUMINAMATH_GPT_crayons_count_l586_58673

-- Define the initial number of crayons
def initial_crayons : ℕ := 1453

-- Define the number of crayons given away
def crayons_given_away : ℕ := 563

-- Define the number of crayons lost
def crayons_lost : ℕ := 558

-- Define the final number of crayons left
def final_crayons_left : ℕ := initial_crayons - crayons_given_away - crayons_lost

-- State that the final number of crayons left is 332
theorem crayons_count : final_crayons_left = 332 :=
by
    -- This is where the proof would go, which we're skipping with sorry
    sorry

end NUMINAMATH_GPT_crayons_count_l586_58673


namespace NUMINAMATH_GPT_solve_for_r_l586_58604

theorem solve_for_r : ∃ r : ℝ, r ≠ 4 ∧ r ≠ 5 ∧ 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 10) / (r^2 - 2*r - 15) ↔ 
  r = 2*Real.sqrt 2 ∨ r = -2*Real.sqrt 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_r_l586_58604


namespace NUMINAMATH_GPT_odd_square_minus_one_divisible_by_eight_l586_58664

theorem odd_square_minus_one_divisible_by_eight (n : ℤ) : ∃ k : ℤ, ((2 * n + 1) ^ 2 - 1) = 8 * k := 
by
  sorry

end NUMINAMATH_GPT_odd_square_minus_one_divisible_by_eight_l586_58664


namespace NUMINAMATH_GPT_course_length_l586_58650

noncomputable def timeBicycling := 12 / 60 -- hours
noncomputable def avgRateBicycling := 30 -- miles per hour
noncomputable def timeRunning := (117 - 12) / 60 -- hours
noncomputable def avgRateRunning := 8 -- miles per hour

theorem course_length : avgRateBicycling * timeBicycling + avgRateRunning * timeRunning = 20 := 
by
  sorry

end NUMINAMATH_GPT_course_length_l586_58650


namespace NUMINAMATH_GPT_total_surface_area_of_resulting_solid_is_12_square_feet_l586_58662

noncomputable def height_of_D :=
  let h_A := 1 / 4
  let h_B := 1 / 5
  let h_C := 1 / 8
  2 - (h_A + h_B + h_C)

theorem total_surface_area_of_resulting_solid_is_12_square_feet :
  let h_A := 1 / 4
  let h_B := 1 / 5
  let h_C := 1 / 8
  let h_D := 2 - (h_A + h_B + h_C)
  let top_and_bottom_area := 4 * 2
  let side_area := 2 * (h_A + h_B + h_C + h_D)
  top_and_bottom_area + side_area = 12 := by
  sorry

end NUMINAMATH_GPT_total_surface_area_of_resulting_solid_is_12_square_feet_l586_58662


namespace NUMINAMATH_GPT_cows_black_more_than_half_l586_58690

theorem cows_black_more_than_half (t b : ℕ) (h1 : t = 18) (h2 : t - 4 = b) : b - t / 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_cows_black_more_than_half_l586_58690


namespace NUMINAMATH_GPT_num_true_propositions_l586_58628

theorem num_true_propositions : 
  (∀ (a b : ℝ), a = 0 → ab = 0) ∧
  (∀ (a b : ℝ), ab ≠ 0 → a ≠ 0) ∧
  ¬ (∀ (a b : ℝ), ab = 0 → a = 0) ∧
  ¬ (∀ (a b : ℝ), a ≠ 0 → ab ≠ 0) → 
  2 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_num_true_propositions_l586_58628


namespace NUMINAMATH_GPT_exists_n_for_perfect_square_l586_58674

theorem exists_n_for_perfect_square (k : ℕ) (hk_pos : k > 0) :
  ∃ n : ℕ, n > 0 ∧ ∃ a : ℕ, a^2 = n * 2^k - 7 :=
by
  sorry

end NUMINAMATH_GPT_exists_n_for_perfect_square_l586_58674


namespace NUMINAMATH_GPT_lyssa_fewer_correct_l586_58670

-- Define the total number of items in the exam
def total_items : ℕ := 75

-- Define the number of mistakes made by Lyssa
def lyssa_mistakes : ℕ := total_items * 20 / 100  -- 20% of 75

-- Define the number of correct answers by Lyssa
def lyssa_correct : ℕ := total_items - lyssa_mistakes

-- Define the number of mistakes made by Precious
def precious_mistakes : ℕ := 12

-- Define the number of correct answers by Precious
def precious_correct : ℕ := total_items - precious_mistakes

-- Statement to prove Lyssa got 3 fewer correct answers than Precious
theorem lyssa_fewer_correct : (precious_correct - lyssa_correct) = 3 := by
  sorry

end NUMINAMATH_GPT_lyssa_fewer_correct_l586_58670


namespace NUMINAMATH_GPT_leo_total_travel_cost_l586_58602

-- Define the conditions as variables and assumptions in Lean
def cost_one_way : ℕ := 24
def working_days : ℕ := 20

-- Define the total travel cost as a function
def total_travel_cost (cost_one_way : ℕ) (working_days : ℕ) : ℕ :=
  cost_one_way * 2 * working_days

-- State the theorem to prove the total travel cost
theorem leo_total_travel_cost : total_travel_cost 24 20 = 960 :=
sorry

end NUMINAMATH_GPT_leo_total_travel_cost_l586_58602


namespace NUMINAMATH_GPT_trig_identity_l586_58649

theorem trig_identity : 4 * Real.sin (20 * Real.pi / 180) + Real.tan (20 * Real.pi / 180) = Real.sqrt 3 := 
by sorry

end NUMINAMATH_GPT_trig_identity_l586_58649


namespace NUMINAMATH_GPT_diameter_of_circle_A_l586_58669

theorem diameter_of_circle_A
  (diameter_B : ℝ)
  (r : ℝ)
  (h1 : diameter_B = 16)
  (h2 : r^2 = (r / 8)^2 * 4):
  2 * (r / 2) = 8 :=
by
  sorry

end NUMINAMATH_GPT_diameter_of_circle_A_l586_58669


namespace NUMINAMATH_GPT_tangent_lengths_l586_58624

noncomputable def internal_tangent_length (r1 r2 d : ℝ) : ℝ :=
  Real.sqrt (d^2 - (r1 + r2)^2)

noncomputable def external_tangent_length (r1 r2 d : ℝ) : ℝ :=
  Real.sqrt (d^2 - (r1 - r2)^2)

theorem tangent_lengths (r1 r2 d : ℝ) (h_r1 : r1 = 8) (h_r2 : r2 = 10) (h_d : d = 50) :
  internal_tangent_length r1 r2 d = 46.67 ∧ external_tangent_length r1 r2 d = 49.96 :=
by
  sorry

end NUMINAMATH_GPT_tangent_lengths_l586_58624


namespace NUMINAMATH_GPT_cole_drive_time_l586_58615

noncomputable def time_to_drive_to_work (D : ℝ) : ℝ :=
  D / 50

theorem cole_drive_time (D : ℝ) (h₁ : time_to_drive_to_work D + (D / 110) = 2) : time_to_drive_to_work D * 60 = 82.5 :=
by
  sorry

end NUMINAMATH_GPT_cole_drive_time_l586_58615


namespace NUMINAMATH_GPT_B_work_days_proof_l586_58657

-- Define the main variables
variables (W : ℝ) (x : ℝ) (daysA : ℝ) (daysBworked : ℝ) (daysAremaining : ℝ)

-- Given conditions from the problem
def A_work_days : ℝ := 6
def B_work_days : ℝ := x
def B_worked_days : ℝ := 10
def A_remaining_days : ℝ := 2

-- We are asked to prove this statement
theorem B_work_days_proof (h1 : daysA = A_work_days)
                           (h2 : daysBworked = B_worked_days)
                           (h3 : daysAremaining = A_remaining_days) 
                           (hx : (W/6 = (W - 10*W/x) / 2)) : x = 15 :=
by 
  -- Proof omitted
  sorry 

end NUMINAMATH_GPT_B_work_days_proof_l586_58657


namespace NUMINAMATH_GPT_problem_solution_l586_58685

variable (α : ℝ)

/-- If $\sin\alpha = 2\cos\alpha$, then the function $f(x) = 2^x - \tan\alpha$ satisfies $f(0) = -1$. -/
theorem problem_solution (h : Real.sin α = 2 * Real.cos α) : (2^0 - Real.tan α) = -1 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l586_58685


namespace NUMINAMATH_GPT_triangle_height_l586_58659

theorem triangle_height (base height area : ℝ) (h_base : base = 3) (h_area : area = 6) (h_formula : area = (1/2) * base * height) : height = 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_height_l586_58659


namespace NUMINAMATH_GPT_digits_base8_sum_l586_58608

open Nat

theorem digits_base8_sum (X Y Z : ℕ) (hX : 0 < X) (hY : 0 < Y) (hZ : 0 < Z) 
  (h_distinct : X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) (h_base8 : X < 8 ∧ Y < 8 ∧ Z < 8) 
  (h_eq : (8^2 * X + 8 * Y + Z) + (8^2 * Y + 8 * Z + X) + (8^2 * Z + 8 * X + Y) = 8^3 * X + 8^2 * X + 8 * X) : 
  Y + Z = 7 :=
by
  sorry

end NUMINAMATH_GPT_digits_base8_sum_l586_58608


namespace NUMINAMATH_GPT_geometric_sequence_term_formula_l586_58647

theorem geometric_sequence_term_formula (a n : ℕ) (a_seq : ℕ → ℕ)
  (h1 : a_seq 0 = a - 1) (h2 : a_seq 1 = a + 1) (h3 : a_seq 2 = a + 4)
  (geometric_seq : ∀ n, a_seq (n + 1) = a_seq n * ((a_seq 1) / (a_seq 0))) :
  a = 5 ∧ a_seq n = 4 * (3 / 2) ^ (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_term_formula_l586_58647


namespace NUMINAMATH_GPT_age_of_new_person_l586_58696

-- Definitions based on conditions
def initial_avg : ℕ := 15
def new_avg : ℕ := 17
def n : ℕ := 9

-- Statement to prove
theorem age_of_new_person : 
    ∃ (A : ℕ), (initial_avg * n + A) / (n + 1) = new_avg ∧ A = 35 := 
by {
    -- Proof steps would go here, but since they are not required, we add 'sorry' to skip the proof
    sorry
}

end NUMINAMATH_GPT_age_of_new_person_l586_58696


namespace NUMINAMATH_GPT_origin_in_circle_m_gt_5_l586_58646

theorem origin_in_circle_m_gt_5 (m : ℝ) : ((0 - 1)^2 + (0 + 2)^2 < m) → (m > 5) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_origin_in_circle_m_gt_5_l586_58646


namespace NUMINAMATH_GPT_odd_multiple_of_9_is_multiple_of_3_l586_58607

theorem odd_multiple_of_9_is_multiple_of_3 (n : ℕ) (h1 : n % 2 = 1) (h2 : n % 9 = 0) : n % 3 = 0 := 
by sorry

end NUMINAMATH_GPT_odd_multiple_of_9_is_multiple_of_3_l586_58607


namespace NUMINAMATH_GPT_Ceva_theorem_l586_58665

variables {A B C K L M P : Point}
variables {BK KC CL LA AM MB : ℝ}

-- Assume P is inside the triangle ABC and KP, LP, and MP intersect BC, CA, and AB at points K, L, and M respectively
-- We need to prove the ratio product property according to Ceva's theorem
theorem Ceva_theorem 
  (h1: BK / KC = b)
  (h2: CL / LA = c)
  (h3: AM / MB = a)
  (h4: (b * c * a = 1)): 
  (BK / KC) * (CL / LA) * (AM / MB) = 1 :=
sorry

end NUMINAMATH_GPT_Ceva_theorem_l586_58665


namespace NUMINAMATH_GPT_domain_g_l586_58651

noncomputable def f : ℝ → ℝ := sorry  -- f is a real-valued function

theorem domain_g:
  (∀ x, x ∈ [-2, 4] ↔ f x ∈ [-2, 4]) →  -- The domain of f(x) is [-2, 4]
  (∀ x, x ∈ [-2, 2] ↔ (f x + f (-x)) ∈ [-2, 2]) :=  -- The domain of g(x) = f(x) + f(-x) is [-2, 2]
by
  intros h
  sorry

end NUMINAMATH_GPT_domain_g_l586_58651


namespace NUMINAMATH_GPT_original_ratio_l586_58686

namespace OilBill

-- Definitions based on conditions
def JanuaryBill : ℝ := 179.99999999999991

def FebruaryBillWith30More (F : ℝ) : Prop := 
  3 * (F + 30) = 900

-- Statement of the problem proving the original ratio
theorem original_ratio (F : ℝ) (hF : FebruaryBillWith30More F) : 
  F / JanuaryBill = 3 / 2 :=
by
  -- This will contain the proof steps
  sorry

end OilBill

end NUMINAMATH_GPT_original_ratio_l586_58686


namespace NUMINAMATH_GPT_distance_between_X_and_Y_l586_58600

theorem distance_between_X_and_Y :
  ∀ (D : ℝ), 
  (10 : ℝ) * (D / (10 : ℝ) + D / (4 : ℝ)) / (10 + 4) = 142.85714285714286 → 
  D = 1000 :=
by
  intro D
  sorry

end NUMINAMATH_GPT_distance_between_X_and_Y_l586_58600


namespace NUMINAMATH_GPT_calculate_expression_l586_58605

theorem calculate_expression (a : ℝ) : 3 * a * (2 * a^2 - 4 * a) - 2 * a^2 * (3 * a + 4) = -20 * a^2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l586_58605


namespace NUMINAMATH_GPT_P_plus_Q_l586_58631

theorem P_plus_Q (P Q : ℝ) (h : (P / (x - 3) + Q * (x - 2)) = (-5 * x^2 + 18 * x + 27) / (x - 3)) : P + Q = 31 := 
by {
  sorry
}

end NUMINAMATH_GPT_P_plus_Q_l586_58631


namespace NUMINAMATH_GPT_watch_cost_price_l586_58645

theorem watch_cost_price 
  (C : ℝ)
  (h1 : 0.9 * C + 180 = 1.05 * C) :
  C = 1200 :=
sorry

end NUMINAMATH_GPT_watch_cost_price_l586_58645


namespace NUMINAMATH_GPT_pyramid_volume_l586_58693

/-- Given the vertices of a triangle and its midpoints, calculate the volume of the folded triangular pyramid. -/
theorem pyramid_volume
  (A B C : ℝ × ℝ)
  (D E F : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (24, 0))
  (hC : C = (12, 16))
  (hD : D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hE : E = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
  (hF : F = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (area_ABC : ℝ)
  (h_area : area_ABC = 192)
  : (1 / 3) * area_ABC * 8 = 512 :=
by sorry

end NUMINAMATH_GPT_pyramid_volume_l586_58693


namespace NUMINAMATH_GPT_sufficient_not_necessary_l586_58619

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2) ↔ (x + y ≥ 4) :=
by sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l586_58619


namespace NUMINAMATH_GPT_num_possibilities_for_asima_integer_l586_58633

theorem num_possibilities_for_asima_integer (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 65) :
  ∃ (n : ℕ), n = 64 :=
by
  sorry

end NUMINAMATH_GPT_num_possibilities_for_asima_integer_l586_58633


namespace NUMINAMATH_GPT_sum_over_term_is_two_l586_58668

-- Definitions of conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n / 2 * (2 * a 0 + (n - 1) * (a 1 - a 0))

def seq_sn_over_an_arithmetic (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∃ dS : ℝ, ∀ n : ℕ, (S (n + 1)) / (a (n + 1)) = (S n) / (a n) + dS

-- The theorem to prove
theorem sum_over_term_is_two (a S : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_n_terms a S)
  (h3 : seq_sn_over_an_arithmetic S a) :
  S 3 / a 3 = 2 :=
sorry

end NUMINAMATH_GPT_sum_over_term_is_two_l586_58668


namespace NUMINAMATH_GPT_smallest_gcd_of_lcm_eq_square_diff_l586_58691

theorem smallest_gcd_of_lcm_eq_square_diff (x y : ℕ) (h : Nat.lcm x y = (x - y) ^ 2) : Nat.gcd x y = 2 :=
sorry

end NUMINAMATH_GPT_smallest_gcd_of_lcm_eq_square_diff_l586_58691


namespace NUMINAMATH_GPT_cost_price_of_book_l586_58611

theorem cost_price_of_book (SP : ℝ) (rate_of_profit : ℝ) (CP : ℝ) 
  (h1 : SP = 90) 
  (h2 : rate_of_profit = 0.8) 
  (h3 : rate_of_profit = (SP - CP) / CP) : 
  CP = 50 :=
sorry

end NUMINAMATH_GPT_cost_price_of_book_l586_58611


namespace NUMINAMATH_GPT_find_angle_ACD_l586_58699

-- Define the vertices of the quadrilateral
variables {A B C D : Type*}

-- Given angles and side equality
variables (angle_DAC : ℝ) (angle_DBC : ℝ) (angle_BCD : ℝ) (eq_BC_AD : Prop)

-- The given conditions in the problem
axiom angle_DAC_is_98 : angle_DAC = 98
axiom angle_DBC_is_82 : angle_DBC = 82
axiom angle_BCD_is_70 : angle_BCD = 70
axiom BC_eq_AD : eq_BC_AD = true

-- Target angle to be proven
def angle_ACD : ℝ := 28

-- The theorem
theorem find_angle_ACD (h1 : angle_DAC = 98)
                       (h2 : angle_DBC = 82)
                       (h3 : angle_BCD = 70)
                       (h4 : eq_BC_AD) : angle_ACD = 28 := 
by
  sorry  -- Proof of the theorem

end NUMINAMATH_GPT_find_angle_ACD_l586_58699


namespace NUMINAMATH_GPT_gardener_hourly_wage_l586_58692

-- Conditions
def rose_bushes_count : Nat := 20
def cost_per_rose_bush : Nat := 150
def hours_per_day : Nat := 5
def days_worked : Nat := 4
def soil_volume : Nat := 100
def cost_per_cubic_foot_soil : Nat := 5
def total_cost : Nat := 4100

-- Theorem statement
theorem gardener_hourly_wage :
  let cost_of_rose_bushes := rose_bushes_count * cost_per_rose_bush
  let cost_of_soil := soil_volume * cost_per_cubic_foot_soil
  let total_material_cost := cost_of_rose_bushes + cost_of_soil
  let labor_cost := total_cost - total_material_cost
  let total_hours_worked := hours_per_day * days_worked
  (labor_cost / total_hours_worked) = 30 := 
by {
  -- Proof placeholder
  sorry
}

end NUMINAMATH_GPT_gardener_hourly_wage_l586_58692


namespace NUMINAMATH_GPT_total_cost_is_90_l586_58698

variable (jackets : ℕ) (shirts : ℕ) (pants : ℕ)
variable (price_jacket : ℕ) (price_shorts : ℕ) (price_pants : ℕ)

theorem total_cost_is_90 
  (h1 : jackets = 3)
  (h2 : price_jacket = 10)
  (h3 : shirts = 2)
  (h4 : price_shorts = 6)
  (h5 : pants = 4)
  (h6 : price_pants = 12) : 
  (jackets * price_jacket + shirts * price_shorts + pants * price_pants) = 90 := by 
  sorry

end NUMINAMATH_GPT_total_cost_is_90_l586_58698


namespace NUMINAMATH_GPT_min_dist_l586_58618

open Real

theorem min_dist (a b : ℝ) :
  let A := (0, -1)
  let B := (1, 3)
  let C := (2, 6)
  let D := (0, b)
  let E := (1, a + b)
  let F := (2, 2 * a + b)
  let AD_sq := (b + 1) ^ 2
  let BE_sq := (a + b - 3) ^ 2
  let CF_sq := (2 * a + b - 6) ^ 2
  AD_sq + BE_sq + CF_sq = (b + 1) ^ 2 + (a + b - 3) ^ 2 + (2 * a + b - 6) ^ 2 → 
  a = 7 / 2 ∧ b = -5 / 6 :=
sorry

end NUMINAMATH_GPT_min_dist_l586_58618


namespace NUMINAMATH_GPT_necessary_and_sufficient_conditions_l586_58613

open Real

def cubic_has_arithmetic_sequence_roots (a b c : ℝ) : Prop :=
∃ x y : ℝ,
  (x - y) * (x) * (x + y) + a * (x^2 + x - y + x + y) + b * x + c = 0 ∧
  3 * x = -a

theorem necessary_and_sufficient_conditions
  (a b c : ℝ) (h : cubic_has_arithmetic_sequence_roots a b c) :
  2 * a^3 - 9 * a * b + 27 * c = 0 ∧ a^2 - 3 * b ≥ 0 :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_conditions_l586_58613


namespace NUMINAMATH_GPT_find_y_l586_58694

theorem find_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) 
  (h : (2 * a) ^ (2 * b ^ 2) = (a ^ b + y ^ b) ^ 2) : y = 4 * a ^ 2 - a := 
sorry

end NUMINAMATH_GPT_find_y_l586_58694


namespace NUMINAMATH_GPT_incorrect_C_l586_58620

variable (D : ℝ → ℝ)

-- Definitions to encapsulate conditions
def range_D : Set ℝ := {0, 1}
def is_even := ∀ x, D x = D (-x)
def is_periodic := ∀ T > 0, ∃ p, ∀ x, D (x + p) = D x
def is_monotonic := ∀ x y, x < y → D x ≤ D y

-- The proof statement
theorem incorrect_C : ¬ is_periodic D :=
sorry

end NUMINAMATH_GPT_incorrect_C_l586_58620


namespace NUMINAMATH_GPT_benny_leftover_money_l586_58644

-- Define the conditions
def initial_money : ℕ := 67
def spent_money : ℕ := 34

-- Define the leftover money calculation
def leftover_money : ℕ := initial_money - spent_money

-- Prove that Benny had 33 dollars left over
theorem benny_leftover_money : leftover_money = 33 :=
by 
  -- Proof
  sorry

end NUMINAMATH_GPT_benny_leftover_money_l586_58644


namespace NUMINAMATH_GPT_right_triangle_acute_angle_30_l586_58622

theorem right_triangle_acute_angle_30 (α β : ℝ) (h1 : α = 60) (h2 : α + β + 90 = 180) : β = 30 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_acute_angle_30_l586_58622


namespace NUMINAMATH_GPT_average_pages_correct_l586_58616

noncomputable def total_pages : ℝ := 50 + 75 + 80 + 120 + 100 + 90 + 110 + 130
def num_books : ℝ := 8
noncomputable def average_pages : ℝ := total_pages / num_books

theorem average_pages_correct : average_pages = 94.375 :=
by
  sorry

end NUMINAMATH_GPT_average_pages_correct_l586_58616


namespace NUMINAMATH_GPT_alex_hours_per_week_l586_58621

theorem alex_hours_per_week
  (summer_earnings : ℕ)
  (summer_weeks : ℕ)
  (summer_hours_per_week : ℕ)
  (academic_year_weeks : ℕ)
  (academic_year_earnings : ℕ)
  (same_hourly_rate : Prop) :
  summer_earnings = 4000 →
  summer_weeks = 8 →
  summer_hours_per_week = 40 →
  academic_year_weeks = 32 →
  academic_year_earnings = 8000 →
  same_hourly_rate →
  (academic_year_earnings / ((summer_earnings : ℚ) / (summer_weeks * summer_hours_per_week)) / academic_year_weeks) = 20 :=
by
  sorry

end NUMINAMATH_GPT_alex_hours_per_week_l586_58621


namespace NUMINAMATH_GPT_prime_pair_perfect_square_l586_58682

theorem prime_pair_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : ∃ a : ℕ, p^2 + p * q + q^2 = a^2) : (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) := 
sorry

end NUMINAMATH_GPT_prime_pair_perfect_square_l586_58682


namespace NUMINAMATH_GPT_total_players_on_ground_l586_58660

theorem total_players_on_ground 
  (cricket_players : ℕ) (hockey_players : ℕ) (football_players : ℕ) (softball_players : ℕ)
  (hcricket : cricket_players = 16) (hhokey : hockey_players = 12) 
  (hfootball : football_players = 18) (hsoftball : softball_players = 13) :
  cricket_players + hockey_players + football_players + softball_players = 59 :=
by
  sorry

end NUMINAMATH_GPT_total_players_on_ground_l586_58660


namespace NUMINAMATH_GPT_restore_original_problem_l586_58677

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1 / X) * (Y + 1 / 2) = 43 :=
by
  rw [hX, hY]
  -- Continue the proof steps here
  sorry

end NUMINAMATH_GPT_restore_original_problem_l586_58677


namespace NUMINAMATH_GPT_kyro_percentage_paid_l586_58617

theorem kyro_percentage_paid
    (aryan_debt : ℕ) -- Aryan owes Fernanda $1200
    (kyro_debt : ℕ) -- Kyro owes Fernanda
    (aryan_debt_twice_kyro_debt : aryan_debt = 2 * kyro_debt) -- Aryan's debt is twice what Kyro owes
    (aryan_payment : ℕ) -- Aryan's payment
    (aryan_payment_percentage : aryan_payment = 60 * aryan_debt / 100) -- Aryan pays 60% of her debt
    (initial_savings : ℕ) -- Initial savings in Fernanda's account
    (final_savings : ℕ) -- Final savings in Fernanda's account
    (initial_savings_cond : initial_savings = 300) -- Fernanda's initial savings is $300
    (final_savings_cond : final_savings = 1500) -- Fernanda's final savings is $1500
    : kyro_payment = 80 * kyro_debt / 100 := -- Kyro paid 80% of her debt
by {
    sorry
}

end NUMINAMATH_GPT_kyro_percentage_paid_l586_58617


namespace NUMINAMATH_GPT_difference_divisible_by_10_l586_58606

theorem difference_divisible_by_10 : (43 ^ 43 - 17 ^ 17) % 10 = 0 := by
  sorry

end NUMINAMATH_GPT_difference_divisible_by_10_l586_58606
