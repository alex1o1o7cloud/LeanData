import Mathlib

namespace smallest_possible_n_l1184_118480

theorem smallest_possible_n (n : ℕ) (h : lcm 60 n / gcd 60 n = 60) : n = 16 :=
sorry

end smallest_possible_n_l1184_118480


namespace speed_of_second_train_l1184_118448

-- Define the given values
def length_train1 := 290.0 -- in meters
def speed_train1 := 120.0 -- in km/h
def length_train2 := 210.04 -- in meters
def crossing_time := 9.0 -- in seconds

-- Define the conversion factors and useful calculations
def meters_per_second_to_kmph (v : Float) : Float := v * 3.6
def total_distance := length_train1 + length_train2
def relative_speed_ms := total_distance / crossing_time
def relative_speed_kmph := meters_per_second_to_kmph relative_speed_ms

-- Define the proof statement
theorem speed_of_second_train : relative_speed_kmph - speed_train1 = 80.0 :=
by
  sorry

end speed_of_second_train_l1184_118448


namespace parallelogram_area_correct_l1184_118483

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_correct (base height : ℝ) (h_base : base = 30) (h_height : height = 12) : parallelogram_area base height = 360 :=
by
  rw [h_base, h_height]
  simp [parallelogram_area]
  sorry

end parallelogram_area_correct_l1184_118483


namespace line_through_points_l1184_118497

theorem line_through_points (x1 y1 x2 y2 : ℕ) (h1 : (x1, y1) = (1, 2)) (h2 : (x2, y2) = (3, 8)) : 
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m + b = 2 := 
by
  sorry

end line_through_points_l1184_118497


namespace gallon_of_water_weighs_eight_pounds_l1184_118415

theorem gallon_of_water_weighs_eight_pounds
  (pounds_per_tablespoon : ℝ := 1.5)
  (cubic_feet_per_gallon : ℝ := 7.5)
  (cost_per_tablespoon : ℝ := 0.50)
  (total_cost : ℝ := 270)
  (bathtub_capacity_cubic_feet : ℝ := 6)
  : (6 * 7.5) * pounds_per_tablespoon = 270 / cost_per_tablespoon / 1.5 :=
by
  sorry

end gallon_of_water_weighs_eight_pounds_l1184_118415


namespace find_a1_l1184_118441

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a n = a 0 + n * d

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

theorem find_a1 (d : ℝ) (h1 : a 13 = 13) (h2 : S 13 = 13) : a 0 = -11 :=
by
  sorry

end find_a1_l1184_118441


namespace extreme_point_at_one_l1184_118443

def f (a x : ℝ) : ℝ := a*x^3 + x^2 - (a+2)*x + 1
def f' (a x : ℝ) : ℝ := 3*a*x^2 + 2*x - (a+2)

theorem extreme_point_at_one (a : ℝ) :
  (f' a 1 = 0) → (a = 0) :=
by
  intro h
  have : 3 * a * 1^2 + 2 * 1 - (a + 2) = 0 := h
  sorry

end extreme_point_at_one_l1184_118443


namespace overall_average_score_l1184_118461

noncomputable def average_score (scores : List ℝ) : ℝ :=
  scores.sum / (scores.length)

theorem overall_average_score :
  let male_scores_avg := 82
  let female_scores_avg := 92
  let num_male_students := 8
  let num_female_students := 32
  let total_students := num_male_students + num_female_students
  let combined_scores_total := num_male_students * male_scores_avg + num_female_students * female_scores_avg
  average_score ([combined_scores_total]) / total_students = 90 :=
by 
  sorry

end overall_average_score_l1184_118461


namespace Blair_17th_turn_l1184_118469

/-
  Jo begins counting by saying "5". Blair then continues the sequence, each time saying a number that is 2 more than the last number Jo said. Jo increments by 1 each turn after Blair. They alternate turns.
  Prove that Blair says the number 55 on her 17th turn.
-/

def Jo_initial := 5
def increment_Jo := 1
def increment_Blair := 2

noncomputable def blair_sequence (n : ℕ) : ℕ :=
  Jo_initial + increment_Blair + (n - 1) * (increment_Jo + increment_Blair)

theorem Blair_17th_turn : blair_sequence 17 = 55 := by
    sorry

end Blair_17th_turn_l1184_118469


namespace min_inv_sum_four_l1184_118482

theorem min_inv_sum_four (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  4 ≤ (1 / a + 1 / b) := 
sorry

end min_inv_sum_four_l1184_118482


namespace red_balls_count_is_correct_l1184_118426

-- Define conditions
def total_balls : ℕ := 100
def white_balls : ℕ := 50
def green_balls : ℕ := 30
def yellow_balls : ℕ := 10
def purple_balls : ℕ := 3
def non_red_purple_prob : ℝ := 0.9

-- Define the number of red balls
def number_of_red_balls (red_balls : ℕ) : Prop :=
  total_balls - (white_balls + green_balls + yellow_balls + purple_balls) = red_balls
  
-- The proof statement
theorem red_balls_count_is_correct : number_of_red_balls 7 := by
  sorry

end red_balls_count_is_correct_l1184_118426


namespace simplify_expression1_simplify_expression2_l1184_118402

variable {a b x y : ℝ}

theorem simplify_expression1 : 3 * a - 5 * b - 2 * a + b = a - 4 * b :=
by sorry

theorem simplify_expression2 : 4 * x^2 + 5 * x * y - 2 * (2 * x^2 - x * y) = 7 * x * y :=
by sorry

end simplify_expression1_simplify_expression2_l1184_118402


namespace bob_weight_l1184_118499

theorem bob_weight (j b : ℝ) (h1 : j + b = 220) (h2 : b - 2 * j = b / 3) : b = 165 :=
  sorry

end bob_weight_l1184_118499


namespace additional_money_needed_l1184_118437

def original_num_bales : ℕ := 10
def original_cost_per_bale : ℕ := 15
def new_cost_per_bale : ℕ := 18

theorem additional_money_needed :
  (2 * original_num_bales * new_cost_per_bale) - (original_num_bales * original_cost_per_bale) = 210 :=
by
  sorry

end additional_money_needed_l1184_118437


namespace kafelnikov_served_in_first_game_l1184_118481

theorem kafelnikov_served_in_first_game (games : ℕ) (kafelnikov_wins : ℕ) (becker_wins : ℕ)
  (server_victories : ℕ) (x y : ℕ) 
  (h1 : kafelnikov_wins = 6)
  (h2 : becker_wins = 3)
  (h3 : server_victories = 5)
  (h4 : games = 9)
  (h5 : kafelnikov_wins + becker_wins = games)
  (h6 : (5 - x) + y = 5) 
  (h7 : x + y = 6):
  x = 3 :=
by
  sorry

end kafelnikov_served_in_first_game_l1184_118481


namespace three_digit_cubes_divisible_by_8_l1184_118423

theorem three_digit_cubes_divisible_by_8 : ∃ (count : ℕ), count = 2 ∧
  ∀ (n : ℤ), (100 ≤ 8 * n^3) ∧ (8 * n^3 ≤ 999) → 
  (8 * n^3 = 216 ∨ 8 * n^3 = 512) := by
  sorry

end three_digit_cubes_divisible_by_8_l1184_118423


namespace p_necessary_but_not_sufficient_for_q_l1184_118427

noncomputable def p (x : ℝ) : Prop := abs x ≤ 2
noncomputable def q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem p_necessary_but_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) := 
by 
  sorry

end p_necessary_but_not_sufficient_for_q_l1184_118427


namespace smallest_pretty_num_l1184_118479

-- Define the notion of a pretty number
def is_pretty (n : ℕ) : Prop :=
  ∃ d1 d2 : ℕ, (1 ≤ d1 ∧ d1 ≤ n) ∧ (1 ≤ d2 ∧ d2 ≤ n) ∧ d2 - d1 ∣ n ∧ (1 < d1)

-- Define the statement to prove that 160400 is the smallest pretty number greater than 401 that is a multiple of 401
theorem smallest_pretty_num (n : ℕ) (hn1 : n > 401) (hn2 : n % 401 = 0) : n = 160400 :=
  sorry

end smallest_pretty_num_l1184_118479


namespace domain_lg_sqrt_l1184_118490

def domain_of_function (x : ℝ) : Prop :=
  1 - x > 0 ∧ x + 2 > 0

theorem domain_lg_sqrt (x : ℝ) : 
  domain_of_function x ↔ -2 < x ∧ x < 1 :=
sorry

end domain_lg_sqrt_l1184_118490


namespace triangle_construction_l1184_118433

-- Define the problem statement in Lean
theorem triangle_construction (a b c : ℝ) :
  correct_sequence = [3, 1, 4, 2] :=
sorry

end triangle_construction_l1184_118433


namespace tank_capacity_l1184_118422

theorem tank_capacity (x : ℝ) (h₁ : 0.40 * x = 60) : x = 150 :=
by
  -- a suitable proof would go here
  -- since we are only interested in the statement, we place sorry in place of the proof
  sorry

end tank_capacity_l1184_118422


namespace find_value_of_a_l1184_118496

-- Define variables and constants
variable (a : ℚ)
variable (b : ℚ := 3 * a)
variable (c : ℚ := 4 * b)
variable (d : ℚ := 6 * c)
variable (total : ℚ := 186)

-- State the theorem
theorem find_value_of_a (h : a + b + c + d = total) : a = 93 / 44 := by
  sorry

end find_value_of_a_l1184_118496


namespace trigonometric_identity_l1184_118449

theorem trigonometric_identity (x : ℝ) (h : (1 + Real.sin x) / Real.cos x = -1/2) : 
  Real.cos x / (Real.sin x - 1) = 1/2 := 
sorry

end trigonometric_identity_l1184_118449


namespace original_average_l1184_118425

theorem original_average (A : ℝ)
  (h : 2 * A = 160) : A = 80 :=
by sorry

end original_average_l1184_118425


namespace molecular_weight_of_compound_l1184_118494

def atomic_weight (count : ℕ) (atomic_mass : ℝ) : ℝ :=
  count * atomic_mass

def molecular_weight (C_atom_count H_atom_count O_atom_count : ℕ)
  (C_atomic_weight H_atomic_weight O_atomic_weight : ℝ) : ℝ :=
  (atomic_weight C_atom_count C_atomic_weight) +
  (atomic_weight H_atom_count H_atomic_weight) +
  (atomic_weight O_atom_count O_atomic_weight)

theorem molecular_weight_of_compound :
  molecular_weight 3 6 1 12.01 1.008 16.00 = 58.078 :=
by
  sorry

end molecular_weight_of_compound_l1184_118494


namespace dot_product_a_b_l1184_118489

-- Definitions for unit vectors e1 and e2 with given conditions
variables (e1 e2 : ℝ × ℝ)
variables (h_norm_e1 : e1.1^2 + e1.2^2 = 1) -- e1 is a unit vector
variables (h_norm_e2 : e2.1^2 + e2.2^2 = 1) -- e2 is a unit vector
variables (h_angle : e1.1 * e2.1 + e1.2 * e2.2 = -1 / 2) -- angle between e1 and e2 is 120 degrees

-- Definitions for vectors a and b
def a : ℝ × ℝ := (e1.1 + e2.1, e1.2 + e2.2)
def b : ℝ × ℝ := (e1.1 - 3 * e2.1, e1.2 - 3 * e2.2)

-- Theorem to prove
theorem dot_product_a_b : (a e1 e2) • (b e1 e2) = -1 :=
by
  sorry

end dot_product_a_b_l1184_118489


namespace laura_owes_correct_amount_l1184_118435

def principal : ℝ := 35
def annual_rate : ℝ := 0.07
def time_years : ℝ := 1
def interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T
def total_amount_owed (P : ℝ) (I : ℝ) : ℝ := P + I

theorem laura_owes_correct_amount :
  total_amount_owed principal (interest principal annual_rate time_years) = 37.45 :=
sorry

end laura_owes_correct_amount_l1184_118435


namespace simplify_and_rationalize_l1184_118403

noncomputable def simplify_expr : ℝ :=
  1 / (1 - (1 / (Real.sqrt 5 - 2)))

theorem simplify_and_rationalize :
  simplify_expr = (1 - Real.sqrt 5) / 4 := by
  sorry

end simplify_and_rationalize_l1184_118403


namespace savings_account_after_8_weeks_l1184_118465

noncomputable def initial_amount : ℕ := 43
noncomputable def weekly_allowance : ℕ := 10
noncomputable def comic_book_cost : ℕ := 3
noncomputable def saved_per_week : ℕ := weekly_allowance - comic_book_cost
noncomputable def weeks : ℕ := 8
noncomputable def savings_in_8_weeks : ℕ := saved_per_week * weeks
noncomputable def total_piggy_bank_after_8_weeks : ℕ := initial_amount + savings_in_8_weeks

theorem savings_account_after_8_weeks : total_piggy_bank_after_8_weeks = 99 :=
by
  have h1 : saved_per_week = 7 := rfl
  have h2 : savings_in_8_weeks = 56 := rfl
  have h3 : total_piggy_bank_after_8_weeks = 99 := rfl
  exact h3

end savings_account_after_8_weeks_l1184_118465


namespace slices_with_both_toppings_l1184_118440

-- Definitions and conditions directly from the problem statement
def total_slices : ℕ := 24
def pepperoni_slices : ℕ := 15
def mushroom_slices : ℕ := 14

-- Theorem proving the number of slices with both toppings
theorem slices_with_both_toppings :
  (∃ n : ℕ, n + (pepperoni_slices - n) + (mushroom_slices - n) = total_slices) → ∃ n : ℕ, n = 5 := 
by 
  sorry

end slices_with_both_toppings_l1184_118440


namespace systematic_sampling_interval_l1184_118478

-- Definitions based on conditions
def population_size : ℕ := 1000
def sample_size : ℕ := 40

-- Theorem statement 
theorem systematic_sampling_interval :
  population_size / sample_size = 25 :=
by
  sorry

end systematic_sampling_interval_l1184_118478


namespace player_A_wins_if_n_equals_9_l1184_118418

-- Define the conditions
def drawing_game (n : ℕ) : Prop :=
  ∃ strategy : ℕ → ℕ,
    strategy 0 = 1 ∧ -- Player A always starts by drawing 1 ball
    (∀ k, 1 ≤ strategy k ∧ strategy k ≤ 3) ∧ -- Players draw between 1 and 3 balls
    ∀ b, 1 ≤ b → b ≤ 3 → (n - 1 - strategy (b - 1)) ≤ 3 → (strategy (n - 1 - (b - 1)) = n - (b - 1) - 1)

-- State the problem to prove Player A has a winning strategy if n = 9
theorem player_A_wins_if_n_equals_9 : drawing_game 9 :=
sorry

end player_A_wins_if_n_equals_9_l1184_118418


namespace original_class_strength_l1184_118406

theorem original_class_strength (T N : ℕ) (h1 : T = 40 * N) (h2 : T + 12 * 32 = 36 * (N + 12)) : N = 12 :=
by
  sorry

end original_class_strength_l1184_118406


namespace project_completion_days_l1184_118452

-- Define the work rates and the total number of days to complete the project
variables (a_rate b_rate : ℝ) (days_to_complete : ℝ)
variable (a_quit_before_completion : ℝ)

-- Define the conditions
def A_rate := 1 / 20
def B_rate := 1 / 20
def quit_before_completion := 10 

-- The total work done in the project as 1 project 
def total_work := 1

-- Define the equation representing the amount of work done by A and B
def total_days := 
  A_rate * (days_to_complete - a_quit_before_completion) + B_rate * days_to_complete

-- The theorem statement
theorem project_completion_days :
  A_rate = a_rate → 
  B_rate = b_rate → 
  quit_before_completion = a_quit_before_completion → 
  total_days = total_work → 
  days_to_complete = 15 :=
by 
  -- placeholders for the conditions
  intros h1 h2 h3 h4
  sorry

end project_completion_days_l1184_118452


namespace range_of_x_l1184_118464

-- Let p and q be propositions regarding the range of x:
def p (x : ℝ) : Prop := x^2 - 5 * x + 6 ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

-- Main theorem statement
theorem range_of_x 
  (h1 : ∀ x : ℝ, p x ∨ q x)
  (h2 : ∀ x : ℝ, ¬ q x) :
  ∀ x : ℝ, (x ≤ 0 ∨ x ≥ 4) := by
  sorry

end range_of_x_l1184_118464


namespace cost_of_10_pound_bag_is_correct_l1184_118442

noncomputable def cost_of_5_pound_bag : ℝ := 13.80
noncomputable def cost_of_25_pound_bag : ℝ := 32.25
noncomputable def min_pounds_needed : ℝ := 65
noncomputable def max_pounds_allowed : ℝ := 80
noncomputable def least_possible_cost : ℝ := 98.73

def min_cost_10_pound_bag : ℝ := 1.98

theorem cost_of_10_pound_bag_is_correct :
  ∀ (x : ℝ), (x >= min_pounds_needed / cost_of_25_pound_bag ∧ x <= max_pounds_allowed / cost_of_5_pound_bag ∧ least_possible_cost = (3 * cost_of_25_pound_bag + x)) → x = min_cost_10_pound_bag :=
by
  sorry

end cost_of_10_pound_bag_is_correct_l1184_118442


namespace Drew_older_than_Maya_by_5_l1184_118463

variable (Maya Drew Peter John Jacob : ℕ)
variable (h1 : John = 30)
variable (h2 : John = 2 * Maya)
variable (h3 : Jacob = 11)
variable (h4 : Jacob + 2 = (Peter + 2) / 2)
variable (h5 : Peter = Drew + 4)

theorem Drew_older_than_Maya_by_5 : Drew = Maya + 5 :=
by
  have Maya_age : Maya = 30 / 2 := by sorry
  have Jacob_age_in_2_years : Jacob + 2 = 13 := by sorry
  have Peter_age_in_2_years : Peter + 2 = 2 * 13 := by sorry
  have Peter_age : Peter = 26 - 2 := by sorry
  have Drew_age : Drew = Peter - 4 := by sorry
  have Drew_older_than_Maya : Drew = Maya + 5 := by sorry
  exact Drew_older_than_Maya

end Drew_older_than_Maya_by_5_l1184_118463


namespace correct_factorization_l1184_118417

variable (x y : ℝ)

theorem correct_factorization :
  x^2 - 2 * x * y + x = x * (x - 2 * y + 1) :=
by sorry

end correct_factorization_l1184_118417


namespace P_equals_neg12_l1184_118438

def P (a b : ℝ) : ℝ :=
  (2 * a + 3 * b)^2 - (2 * a + b) * (2 * a - b) - 2 * b * (3 * a + 5 * b)

lemma simplified_P (a b : ℝ) : P a b = 6 * a * b :=
  by sorry

theorem P_equals_neg12 (a b : ℝ) (h : b = -2 / a) : P a b = -12 :=
  by sorry

end P_equals_neg12_l1184_118438


namespace problem_statement_l1184_118462

theorem problem_statement (p : ℕ) (hp : Nat.Prime p) :
  ∀ n : ℕ, (∃ φn : ℕ, φn = Nat.totient n ∧ p ∣ φn ∧ (∀ a : ℕ, Nat.gcd a n = 1 → n ∣ a ^ (φn / p) - 1)) ↔ 
  (∃ q1 q2 : ℕ, q1 ≠ q2 ∧ Nat.Prime q1 ∧ Nat.Prime q2 ∧ q1 ≡ 1 [MOD p] ∧ q2 ≡ 1 [MOD p] ∧ q1 ∣ n ∧ q2 ∣ n ∨ 
  (∃ q : ℕ, Nat.Prime q ∧ q ≡ 1 [MOD p] ∧ q ∣ n ∧ p ^ 2 ∣ n)) :=
by {
  sorry
}

end problem_statement_l1184_118462


namespace rhombus_diagonal_length_l1184_118416

theorem rhombus_diagonal_length
  (area : ℝ) (d2 : ℝ) (d1 : ℝ)
  (h_area : area = 432) 
  (h_d2 : d2 = 24) :
  d1 = 36 :=
by
  sorry

end rhombus_diagonal_length_l1184_118416


namespace find_n_l1184_118429

theorem find_n
  (n : ℕ)
  (h1 : 2287 % n = r)
  (h2 : 2028 % n = r)
  (h3 : 1806 % n = r)
  (h_r_non_zero : r ≠ 0) : 
  n = 37 :=
by
  sorry

end find_n_l1184_118429


namespace operation_value_l1184_118407

-- Define the operations as per the conditions.
def star (m n : ℤ) : ℤ := n^2 - m
def hash (m k : ℤ) : ℚ := (k + 2 * m) / 3

-- State the theorem we want to prove.
theorem operation_value : hash (star 3 3) (star 2 5) = 35 / 3 :=
  by
  sorry

end operation_value_l1184_118407


namespace first_place_beats_joe_by_two_points_l1184_118445

def points (wins draws : ℕ) : ℕ := 3 * wins + draws

theorem first_place_beats_joe_by_two_points
  (joe_wins joe_draws first_place_wins first_place_draws : ℕ)
  (h1 : joe_wins = 1)
  (h2 : joe_draws = 3)
  (h3 : first_place_wins = 2)
  (h4 : first_place_draws = 2) :
  points first_place_wins first_place_draws - points joe_wins joe_draws = 2 := by
  sorry

end first_place_beats_joe_by_two_points_l1184_118445


namespace ladder_distance_l1184_118405

theorem ladder_distance (x : ℝ) (h1 : (13:ℝ) = Real.sqrt (x ^ 2 + 12 ^ 2)) : 
  x = 5 :=
by 
  sorry

end ladder_distance_l1184_118405


namespace field_trip_savings_l1184_118434

-- Define the parameters given in the conditions
def num_students : ℕ := 30
def contribution_per_student_per_week : ℕ := 2
def weeks_per_month : ℕ := 4
def num_months : ℕ := 2

-- Define the weekly savings for the class
def weekly_savings : ℕ := num_students * contribution_per_student_per_week

-- Define the total weeks in the given number of months
def total_weeks : ℕ := num_months * weeks_per_month

-- Define the total savings in the given number of months
def total_savings : ℕ := weekly_savings * total_weeks

-- Now, we state the theorem
theorem field_trip_savings : total_savings = 480 :=
by {
  -- calculations are skipped
  sorry
}

end field_trip_savings_l1184_118434


namespace anna_pizza_fraction_l1184_118471

theorem anna_pizza_fraction :
  let total_slices := 16
  let anna_eats := 2
  let shared_slices := 1
  let anna_share := shared_slices / 3
  let fraction_alone := anna_eats / total_slices
  let fraction_shared := anna_share / total_slices
  fraction_alone + fraction_shared = 7 / 48 :=
by
  sorry

end anna_pizza_fraction_l1184_118471


namespace max_self_intersection_points_13_max_self_intersection_points_1950_l1184_118466

def max_self_intersection_points (n : ℕ) : ℕ :=
if n % 2 = 1 then n * (n - 3) / 2 else n * (n - 4) / 2 + 1

theorem max_self_intersection_points_13 : max_self_intersection_points 13 = 65 :=
by sorry

theorem max_self_intersection_points_1950 : max_self_intersection_points 1950 = 1897851 :=
by sorry

end max_self_intersection_points_13_max_self_intersection_points_1950_l1184_118466


namespace least_integer_in_ratio_1_3_5_l1184_118493

theorem least_integer_in_ratio_1_3_5 (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 90) (h_ratio : a * 3 = b ∧ a * 5 = c) : a = 10 :=
sorry

end least_integer_in_ratio_1_3_5_l1184_118493


namespace number_of_customers_who_tipped_is_3_l1184_118430

-- Definitions of conditions
def charge_per_lawn : ℤ := 33
def lawns_mowed : ℤ := 16
def total_earnings : ℤ := 558
def tip_per_customer : ℤ := 10

-- Calculate intermediate values
def earnings_from_mowing : ℤ := lawns_mowed * charge_per_lawn
def earnings_from_tips : ℤ := total_earnings - earnings_from_mowing
def number_of_tips : ℤ := earnings_from_tips / tip_per_customer

-- Theorem stating our proof
theorem number_of_customers_who_tipped_is_3 : number_of_tips = 3 := by
  sorry

end number_of_customers_who_tipped_is_3_l1184_118430


namespace geo_seq_product_l1184_118400

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m / a 1

theorem geo_seq_product
  {a : ℕ → ℝ}
  (h_pos : ∀ n, a n > 0)
  (h_seq : geometric_sequence a)
  (h_roots : ∃ x y, (x*x - 10 * x + 16 = 0) ∧ (y*y - 10 * y + 16 = 0) ∧ a 1 = x ∧ a 19 = y) :
  a 8 * a 10 * a 12 = 64 := 
sorry

end geo_seq_product_l1184_118400


namespace range_of_k_l1184_118414

def f : ℝ → ℝ := sorry

axiom cond1 (a b : ℝ) : f (a + b) = f a + f b + 2 * a * b
axiom cond2 (k : ℝ) : ∀ x : ℝ, f (x + k) = f (k - x)
axiom cond3 : ∀ x y : ℝ, 1 ≤ x → x ≤ y → y ≤ 2 → f x ≤ f y

theorem range_of_k (k : ℝ) : k ≤ 1 :=
sorry

end range_of_k_l1184_118414


namespace total_adults_across_all_three_buses_l1184_118486

def total_passengers : Nat := 450
def bus_A_passengers : Nat := 120
def bus_B_passengers : Nat := 210
def bus_C_passengers : Nat := 120
def children_ratio_A : ℚ := 1/3
def children_ratio_B : ℚ := 2/5
def children_ratio_C : ℚ := 3/8

theorem total_adults_across_all_three_buses :
  let children_A := bus_A_passengers * children_ratio_A
  let children_B := bus_B_passengers * children_ratio_B
  let children_C := bus_C_passengers * children_ratio_C
  let adults_A := bus_A_passengers - children_A
  let adults_B := bus_B_passengers - children_B
  let adults_C := bus_C_passengers - children_C
  (adults_A + adults_B + adults_C) = 281 := by {
    -- The proof steps will go here
    sorry
}

end total_adults_across_all_three_buses_l1184_118486


namespace abs_negative_five_l1184_118431

theorem abs_negative_five : abs (-5) = 5 :=
by
  sorry

end abs_negative_five_l1184_118431


namespace tan_alpha_sub_60_l1184_118421

theorem tan_alpha_sub_60 
  (alpha : ℝ) 
  (h : Real.tan alpha = 4 * Real.sin (420 * Real.pi / 180)) : 
  Real.tan (alpha - 60 * Real.pi / 180) = (Real.sqrt 3) / 7 :=
by sorry

end tan_alpha_sub_60_l1184_118421


namespace period_of_f_l1184_118476

noncomputable def f (x : ℝ) : ℝ := (Real.tan (x/3)) + (Real.sin x)

theorem period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x :=
  sorry

end period_of_f_l1184_118476


namespace find_number_l1184_118484

theorem find_number (number : ℝ) : 469138 * number = 4690910862 → number = 10000.1 :=
by
  sorry

end find_number_l1184_118484


namespace second_movie_time_difference_l1184_118412

def first_movie_length := 90 -- 1 hour and 30 minutes in minutes
def popcorn_time := 10 -- Time spent making popcorn in minutes
def fries_time := 2 * popcorn_time -- Time spent making fries in minutes
def total_time := 4 * 60 -- Total time for cooking and watching movies in minutes

theorem second_movie_time_difference :
  (total_time - (popcorn_time + fries_time + first_movie_length)) - first_movie_length = 30 :=
by
  sorry

end second_movie_time_difference_l1184_118412


namespace solve_for_M_plus_N_l1184_118411

theorem solve_for_M_plus_N (M N : ℕ) (h1 : 4 * N = 588) (h2 : 4 * 63 = 7 * M) : M + N = 183 := by
  sorry

end solve_for_M_plus_N_l1184_118411


namespace john_average_score_change_l1184_118428

/-- Given John's scores on his biology exams, calculate the change in his average score after the fourth exam. -/
theorem john_average_score_change :
  let first_three_scores := [84, 88, 95]
  let fourth_score := 92
  let first_average := (84 + 88 + 95) / 3
  let new_average := (84 + 88 + 95 + 92) / 4
  new_average - first_average = 0.75 :=
by
  sorry

end john_average_score_change_l1184_118428


namespace cakes_donated_l1184_118485
-- Import necessary libraries for arithmetic operations and proofs

-- Define the conditions and required proof in Lean
theorem cakes_donated (c : ℕ) (h : 8 * c + 4 * c + 2 * c = 140) : c = 10 :=
by
  sorry

end cakes_donated_l1184_118485


namespace only_zero_sol_l1184_118409

theorem only_zero_sol (x y z t : ℤ) : x^2 + y^2 + z^2 + t^2 = 2 * x * y * z * t → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 :=
by
  sorry

end only_zero_sol_l1184_118409


namespace total_hours_correct_l1184_118454

/-- Definitions for the times each person has left to finish their homework. -/
noncomputable def Jacob_time : ℕ := 18
noncomputable def Greg_time : ℕ := Jacob_time - 6
noncomputable def Patrick_time : ℕ := 2 * Greg_time - 4

/-- Proving the total time left for Patrick, Greg, and Jacob to finish their homework. -/

theorem total_hours_correct : Jacob_time + Greg_time + Patrick_time = 50 := by
  sorry

end total_hours_correct_l1184_118454


namespace container_volume_ratio_l1184_118475

theorem container_volume_ratio (A B : ℕ) 
  (h1 : (3 / 4 : ℚ) * A = (5 / 8 : ℚ) * B) :
  (A : ℚ) / B = 5 / 6 :=
by
  admit
-- sorry

end container_volume_ratio_l1184_118475


namespace largest_prime_divisor_of_sum_of_squares_l1184_118413

def a : ℕ := 35
def b : ℕ := 84

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, Prime p ∧ p = 13 ∧ (a^2 + b^2) % p = 0 := by
  sorry

end largest_prime_divisor_of_sum_of_squares_l1184_118413


namespace island_not_Maya_l1184_118458

variable (A B : Prop)
variable (IslandMaya : Prop)
variable (Liar : Prop → Prop)
variable (TruthTeller : Prop → Prop)

-- A's statement: "We are both liars, and this island is called Maya."
axiom A_statement : Liar A ∧ Liar B ∧ IslandMaya

-- B's statement: "At least one of us is a liar, and this island is not called Maya."
axiom B_statement : (Liar A ∨ Liar B) ∧ ¬IslandMaya

theorem island_not_Maya : ¬IslandMaya := by
  sorry

end island_not_Maya_l1184_118458


namespace find_value_l1184_118439

theorem find_value (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 :=
sorry

end find_value_l1184_118439


namespace elvins_first_month_bill_l1184_118419

variable (F C : ℕ)

def total_bill_first_month := F + C
def total_bill_second_month := F + 2 * C

theorem elvins_first_month_bill :
  total_bill_first_month F C = 46 ∧
  total_bill_second_month F C = 76 ∧
  total_bill_second_month F C - total_bill_first_month F C = 30 →
  total_bill_first_month F C = 46 :=
by
  intro h
  sorry

end elvins_first_month_bill_l1184_118419


namespace max_sum_of_factors_l1184_118459

theorem max_sum_of_factors (h k : ℕ) (h_even : Even h) (prod_eq : h * k = 24) : h + k ≤ 14 :=
sorry

end max_sum_of_factors_l1184_118459


namespace domain_of_function_l1184_118404

theorem domain_of_function :
  ∀ x : ℝ, 3 * x - 2 > 0 ∧ 2 * x - 1 > 0 ↔ x > (2 / 3) := by
  intro x
  sorry

end domain_of_function_l1184_118404


namespace cost_of_book_sold_at_loss_l1184_118456

theorem cost_of_book_sold_at_loss
  (C1 C2 : ℝ)
  (total_cost : C1 + C2 = 360)
  (selling_price1 : 0.85 * C1 = 1.19 * C2) :
  C1 = 210 :=
sorry

end cost_of_book_sold_at_loss_l1184_118456


namespace hiring_manager_acceptance_l1184_118408

theorem hiring_manager_acceptance 
    (average_age : ℤ) (std_dev : ℤ) (num_ages : ℤ)
    (applicant_ages_are_int : ∀ (x : ℤ), x ≥ (average_age - std_dev) ∧ x ≤ (average_age + std_dev)) :
    (∃ k : ℤ, (average_age + k * std_dev) - (average_age - k * std_dev) + 1 = num_ages) → k = 1 :=
by 
  intros h
  sorry

end hiring_manager_acceptance_l1184_118408


namespace sum_of_reciprocals_l1184_118457

-- We state that for all non-zero real numbers x and y, if x + y = xy,
-- then the sum of their reciprocals equals 1.
theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  1/x + 1/y = 1 :=
by
  sorry

end sum_of_reciprocals_l1184_118457


namespace distance_between_wheels_l1184_118488

theorem distance_between_wheels 
  (D : ℕ) 
  (back_perimeter : ℕ) (front_perimeter : ℕ) 
  (more_revolutions : ℕ)
  (h1 : back_perimeter = 9)
  (h2 : front_perimeter = 7)
  (h3 : more_revolutions = 10)
  (h4 : D / front_perimeter = D / back_perimeter + more_revolutions) : 
  D = 315 :=
by
  sorry

end distance_between_wheels_l1184_118488


namespace temperature_43_l1184_118455

theorem temperature_43 (T W Th F : ℝ)
  (h1 : (T + W + Th) / 3 = 42)
  (h2 : (W + Th + F) / 3 = 44)
  (h3 : T = 37) : F = 43 :=
by
  sorry

end temperature_43_l1184_118455


namespace clock_angle_9_30_l1184_118453

theorem clock_angle_9_30 : 
  let hour_hand_pos := 9.5 
  let minute_hand_pos := 6 
  let degrees_per_division := 30 
  let divisions_apart := hour_hand_pos - minute_hand_pos
  let angle := divisions_apart * degrees_per_division
  angle = 105 :=
by
  sorry

end clock_angle_9_30_l1184_118453


namespace max_n_factorable_l1184_118432

theorem max_n_factorable :
  ∃ n : ℤ, (∀ A B : ℤ, 3 * A * B = 24 → 3 * B + A = n) ∧ (n = 73) :=
sorry

end max_n_factorable_l1184_118432


namespace value_of_x_l1184_118477

theorem value_of_x (x : ℤ) (h : 3 * x = (26 - x) + 26) : x = 13 :=
by
  sorry

end value_of_x_l1184_118477


namespace prove_x_value_l1184_118410

-- Definitions of the conditions
variable (x y z w : ℕ)
variable (h1 : x = y + 8)
variable (h2 : y = z + 15)
variable (h3 : z = w + 25)
variable (h4 : w = 90)

-- The goal is to prove x = 138 given the conditions
theorem prove_x_value : x = 138 := by
  sorry

end prove_x_value_l1184_118410


namespace compute_remainder_l1184_118420

/-- T is the sum of all three-digit positive integers 
  where the digits are distinct, the hundreds digit is at least 2,
  and the digit 1 is not used in any place. -/
def T : ℕ := 
  let hundreds_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 56 * 100
  let tens_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 49 * 10
  let units_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 49
  hundreds_sum + tens_sum + units_sum

/-- Theorem: Compute the remainder when T is divided by 1000. -/
theorem compute_remainder : T % 1000 = 116 := by
  sorry

end compute_remainder_l1184_118420


namespace sequence_is_geometric_l1184_118450

theorem sequence_is_geometric (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 3) 
  (h_rec : ∀ n, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  ∀ n, a n = 2 ^ (n - 1) + 1 := 
by
  sorry

end sequence_is_geometric_l1184_118450


namespace intersection_expression_value_l1184_118491

theorem intersection_expression_value
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : x₁ * y₁ = 1)
  (h₂ : x₂ * y₂ = 1)
  (h₃ : x₁ = -x₂)
  (h₄ : y₁ = -y₂) :
  x₁ * y₂ + x₂ * y₁ = -2 :=
by
  sorry

end intersection_expression_value_l1184_118491


namespace calorie_allowance_correct_l1184_118468

-- Definitions based on the problem's conditions
def daily_calorie_allowance : ℕ := 2000
def weekly_calorie_allowance : ℕ := 10500
def days_in_week : ℕ := 7

-- The statement to be proven
theorem calorie_allowance_correct :
  daily_calorie_allowance * days_in_week = weekly_calorie_allowance :=
by
  sorry

end calorie_allowance_correct_l1184_118468


namespace rectangle_ratio_l1184_118487

-- Define the width of the rectangle
def width : ℕ := 7

-- Define the area of the rectangle
def area : ℕ := 196

-- Define that the length is a multiple of the width
def length_is_multiple_of_width (l w : ℕ) : Prop := ∃ k : ℕ, l = k * w

-- Define that the ratio of the length to the width is 4:1
def ratio_is_4_to_1 (l w : ℕ) : Prop := l / w = 4

theorem rectangle_ratio (l w : ℕ) (h1 : w = width) (h2 : area = l * w) (h3 : length_is_multiple_of_width l w) : ratio_is_4_to_1 l w :=
by
  sorry

end rectangle_ratio_l1184_118487


namespace a_n3_l1184_118473

def right_angled_triangle_array (a : ℕ → ℕ → ℚ) : Prop :=
  ∀ i j, 1 ≤ j ∧ j ≤ i →
    (j = 1 → a i j = 1 / 4 + (i - 1) / 4) ∧
    (i ≥ 3 → (1 < j → a i j = a i 1 * (1 / 2)^(j - 1)))

theorem a_n3 (a : ℕ → ℕ → ℚ) (n : ℕ) (h : right_angled_triangle_array a) : a n 3 = n / 16 :=
sorry

end a_n3_l1184_118473


namespace number_of_zeros_of_f_l1184_118451

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 4 * Real.exp x - 2
else abs (2 - Real.log x / Real.log 2)

theorem number_of_zeros_of_f :
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = Real.log (1 / 2) ∧ x₂ = 4 :=
by
  sorry

end number_of_zeros_of_f_l1184_118451


namespace youtube_dislikes_l1184_118470

def initial_dislikes (likes : ℕ) : ℕ := (likes / 2) + 100

def new_dislikes (initial : ℕ) : ℕ := initial + 1000

theorem youtube_dislikes
  (likes : ℕ)
  (h_likes : likes = 3000) :
  new_dislikes (initial_dislikes likes) = 2600 :=
by
  sorry

end youtube_dislikes_l1184_118470


namespace plane_eq_of_point_and_parallel_l1184_118446

theorem plane_eq_of_point_and_parallel (A B C D : ℤ) 
  (h1 : A = 3) (h2 : B = -2) (h3 : C = 4) 
  (point : ℝ × ℝ × ℝ) (hpoint : point = (2, -3, 5))
  (h4 : 3 * (2 : ℝ) - 2 * (-3 : ℝ) + 4 * (5 : ℝ) + (D : ℝ) = 0)
  (hD : D = -32)
  (hGCD : Int.gcd (Int.natAbs 3) (Int.gcd (Int.natAbs (-2)) (Int.gcd (Int.natAbs 4) (Int.natAbs (-32)))) = 1) : 
  3 * (x : ℝ) - 2 * (y : ℝ) + 4 * (z : ℝ) - 32 = 0 :=
sorry

end plane_eq_of_point_and_parallel_l1184_118446


namespace completing_square_correctness_l1184_118460

theorem completing_square_correctness :
  (2 * x^2 - 4 * x - 7 = 0) ->
  ((x - 1)^2 = 9 / 2) :=
sorry

end completing_square_correctness_l1184_118460


namespace percentage_of_sikhs_is_10_l1184_118474

-- Definitions based on the conditions
def total_boys : ℕ := 850
def percent_muslims : ℕ := 34
def percent_hindus : ℕ := 28
def other_community_boys : ℕ := 238

-- The problem statement to prove
theorem percentage_of_sikhs_is_10 :
  ((total_boys - ((percent_muslims * total_boys / 100) + (percent_hindus * total_boys / 100) + other_community_boys))
  * 100 / total_boys) = 10 := 
by
  sorry

end percentage_of_sikhs_is_10_l1184_118474


namespace two_digit_number_l1184_118436

theorem two_digit_number (x y : ℕ) (h1 : x + y = 11) (h2 : 10 * y + x = 10 * x + y + 63) : 10 * x + y = 29 := 
by 
  sorry

end two_digit_number_l1184_118436


namespace incorrect_statement_count_l1184_118467

theorem incorrect_statement_count :
  let statements := ["Every number has a square root",
                     "The square root of a number must be positive",
                     "The square root of a^2 is a",
                     "The square root of (π - 4)^2 is π - 4",
                     "A square root cannot be negative"]
  let incorrect := [statements.get! 0, statements.get! 1, statements.get! 2, statements.get! 3]
  incorrect.length = 4 :=
by
  sorry

end incorrect_statement_count_l1184_118467


namespace sum_repeating_decimals_l1184_118444

theorem sum_repeating_decimals : (0.14 + 0.27) = (41 / 99) := by
  sorry

end sum_repeating_decimals_l1184_118444


namespace carson_gold_stars_l1184_118424

theorem carson_gold_stars (gold_stars_yesterday gold_stars_today : ℕ) (h1 : gold_stars_yesterday = 6) (h2 : gold_stars_today = 9) : 
  gold_stars_yesterday + gold_stars_today = 15 := 
by
  sorry

end carson_gold_stars_l1184_118424


namespace max_sum_11xy_3x_2012yz_l1184_118495

theorem max_sum_11xy_3x_2012yz (x y z : ℕ) (h : x + y + z = 1000) : 
  11 * x * y + 3 * x + 2012 * y * z ≤ 503000000 :=
sorry

end max_sum_11xy_3x_2012yz_l1184_118495


namespace tire_cost_l1184_118472

theorem tire_cost (total_cost : ℕ) (number_of_tires : ℕ) (cost_per_tire : ℕ) 
    (h1 : total_cost = 240) 
    (h2 : number_of_tires = 4)
    (h3 : cost_per_tire = total_cost / number_of_tires) : 
    cost_per_tire = 60 :=
sorry

end tire_cost_l1184_118472


namespace find_xy_yz_xz_l1184_118401

noncomputable def xy_yz_xz (x y z : ℝ) : ℝ := x * y + y * z + x * z

theorem find_xy_yz_xz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 + x * y + y^2 = 48) (h2 : y^2 + y * z + z^2 = 16) (h3 : z^2 + x * z + x^2 = 64) :
  xy_yz_xz x y z = 32 :=
sorry

end find_xy_yz_xz_l1184_118401


namespace one_cow_one_bag_l1184_118447

-- Definitions based on the conditions provided.
def cows : ℕ := 45
def bags : ℕ := 45
def days : ℕ := 45

-- Problem statement: Prove that one cow will eat one bag of husk in 45 days.
theorem one_cow_one_bag (h : cows * bags = bags * days) : days = 45 :=
by
  sorry

end one_cow_one_bag_l1184_118447


namespace angle_sum_around_point_l1184_118498

theorem angle_sum_around_point (y : ℝ) (h1 : 150 + y + y = 360) : y = 105 :=
by sorry

end angle_sum_around_point_l1184_118498


namespace vector_calc_l1184_118492

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Statement to prove that 2a - b = (5, 7)
theorem vector_calc : 2 • a - b = (5, 7) :=
by {
  -- Proof will be filled here
  sorry
}

end vector_calc_l1184_118492
