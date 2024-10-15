import Mathlib

namespace NUMINAMATH_GPT_solve_for_y_l2361_236137

theorem solve_for_y (y : ℝ) (h : (5 - 2 / y)^(1/3) = -3) : y = 1 / 16 := 
sorry

end NUMINAMATH_GPT_solve_for_y_l2361_236137


namespace NUMINAMATH_GPT_volume_of_second_cube_is_twosqrt2_l2361_236131

noncomputable def side_length (volume : ℝ) : ℝ :=
  volume^(1/3)

noncomputable def surface_area (side : ℝ) : ℝ :=
  6 * side^2

theorem volume_of_second_cube_is_twosqrt2
  (v1 : ℝ)
  (h1 : v1 = 1)
  (A1 := surface_area (side_length v1))
  (A2 := 2 * A1)
  (s2 := (A2 / 6)^(1/2)) :
  (s2^3 = 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_second_cube_is_twosqrt2_l2361_236131


namespace NUMINAMATH_GPT_fraction_B_A_plus_C_l2361_236188

variable (A B C : ℝ)
variable (f : ℝ)
variable (hA : A = 1 / 3 * (B + C))
variable (hB : A = B + 30)
variable (hTotal : A + B + C = 1080)
variable (hf : B = f * (A + C))

theorem fraction_B_A_plus_C :
  f = 2 / 7 :=
sorry

end NUMINAMATH_GPT_fraction_B_A_plus_C_l2361_236188


namespace NUMINAMATH_GPT_sums_of_coordinates_of_A_l2361_236127

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end NUMINAMATH_GPT_sums_of_coordinates_of_A_l2361_236127


namespace NUMINAMATH_GPT_total_words_in_poem_l2361_236107

theorem total_words_in_poem 
  (stanzas : ℕ) 
  (lines_per_stanza : ℕ) 
  (words_per_line : ℕ) 
  (h_stanzas : stanzas = 20) 
  (h_lines_per_stanza : lines_per_stanza = 10) 
  (h_words_per_line : words_per_line = 8) : 
  stanzas * lines_per_stanza * words_per_line = 1600 := 
sorry

end NUMINAMATH_GPT_total_words_in_poem_l2361_236107


namespace NUMINAMATH_GPT_curve_touch_all_Ca_l2361_236164

theorem curve_touch_all_Ca (a : ℝ) (a_pos : a > 0) (x y : ℝ) :
  ( (y - a^2)^2 = x^2 * (a^2 - x^2) ) → (y = (3 / 4) * x^2) :=
by
  sorry

end NUMINAMATH_GPT_curve_touch_all_Ca_l2361_236164


namespace NUMINAMATH_GPT_area_of_square_with_given_diagonal_l2361_236169

theorem area_of_square_with_given_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : ∃ (A : ℝ), A = 64 :=
by
  use (8 * 8)
  sorry

end NUMINAMATH_GPT_area_of_square_with_given_diagonal_l2361_236169


namespace NUMINAMATH_GPT_friends_in_group_l2361_236185

theorem friends_in_group (n : ℕ) 
  (avg_before_increase : ℝ := 800) 
  (avg_after_increase : ℝ := 850) 
  (individual_rent_increase : ℝ := 800 * 0.25) 
  (original_rent : ℝ := 800) 
  (new_rent : ℝ := 1000)
  (original_total : ℝ := avg_before_increase * n) 
  (new_total : ℝ := original_total + individual_rent_increase):
  new_total = avg_after_increase * n → 
  n = 4 :=
by
  sorry

end NUMINAMATH_GPT_friends_in_group_l2361_236185


namespace NUMINAMATH_GPT_max_bishops_correct_bishop_position_count_correct_l2361_236153

-- Define the parameters and predicates
def chessboard_size : ℕ := 2015

def max_bishops (board_size : ℕ) : ℕ := 2 * board_size - 1 - 1

def bishop_position_count (board_size : ℕ) : ℕ := 2 ^ (board_size - 1) * 2 * 2

-- State the equalities to be proved
theorem max_bishops_correct : max_bishops chessboard_size = 4028 := by
  -- proof will be here
  sorry

theorem bishop_position_count_correct : bishop_position_count chessboard_size = 2 ^ 2016 := by
  -- proof will be here
  sorry

end NUMINAMATH_GPT_max_bishops_correct_bishop_position_count_correct_l2361_236153


namespace NUMINAMATH_GPT_speed_of_stream_l2361_236177

-- Definitions of the problem's conditions
def downstream_distance := 72
def upstream_distance := 30
def downstream_time := 3
def upstream_time := 3

-- The unknowns
variables (b s : ℝ)

-- The effective speed equations based on the problem conditions
def effective_speed_downstream := b + s
def effective_speed_upstream := b - s

-- The core conditions of the problem
def condition1 : Prop := downstream_distance = effective_speed_downstream * downstream_time
def condition2 : Prop := upstream_distance = effective_speed_upstream * upstream_time

-- The problem statement transformed into a Lean theorem
theorem speed_of_stream (h1 : condition1) (h2 : condition2) : s = 7 := 
sorry

end NUMINAMATH_GPT_speed_of_stream_l2361_236177


namespace NUMINAMATH_GPT_find_q_l2361_236117

noncomputable def solution_condition (p q : ℝ) : Prop :=
  (p > 1) ∧ (q > 1) ∧ (1 / p + 1 / q = 1) ∧ (p * q = 9)

theorem find_q (p q : ℝ) (h : solution_condition p q) : 
  q = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_find_q_l2361_236117


namespace NUMINAMATH_GPT_problem1_problem2_l2361_236105

noncomputable def circle_ast (a b : ℕ) : ℕ := sorry

axiom circle_ast_self (a : ℕ) : circle_ast a a = a
axiom circle_ast_zero (a : ℕ) : circle_ast a 0 = 2 * a
axiom circle_ast_add (a b c d : ℕ) : circle_ast a b + circle_ast c d = circle_ast (a + c) (b + d)

theorem problem1 : circle_ast (2 + 3) (0 + 3) = 7 := sorry

theorem problem2 : circle_ast 1024 48 = 2000 := sorry

end NUMINAMATH_GPT_problem1_problem2_l2361_236105


namespace NUMINAMATH_GPT_average_price_of_towels_l2361_236183

theorem average_price_of_towels :
  let total_cost := 2350
  let total_towels := 10
  total_cost / total_towels = 235 :=
by
  sorry

end NUMINAMATH_GPT_average_price_of_towels_l2361_236183


namespace NUMINAMATH_GPT_sum_of_fractions_l2361_236111

-- Definition of the fractions
def frac1 : ℚ := 3/5
def frac2 : ℚ := 5/11
def frac3 : ℚ := 1/3

-- Main theorem stating that the sum of the fractions equals 229/165
theorem sum_of_fractions : frac1 + frac2 + frac3 = 229 / 165 := sorry

end NUMINAMATH_GPT_sum_of_fractions_l2361_236111


namespace NUMINAMATH_GPT_new_person_weight_l2361_236166

-- Define the given conditions as Lean definitions
def weight_increase_per_person : ℝ := 2.5
def num_people : ℕ := 8
def replaced_person_weight : ℝ := 65

-- State the theorem using the given conditions and the correct answer
theorem new_person_weight :
  (weight_increase_per_person * num_people) + replaced_person_weight = 85 :=
sorry

end NUMINAMATH_GPT_new_person_weight_l2361_236166


namespace NUMINAMATH_GPT_binomial_theorem_fifth_term_l2361_236174
-- Import the necessary library

-- Define the theorem as per the given conditions and required proof
theorem binomial_theorem_fifth_term
  (a x : ℝ) 
  (hx : x ≠ 0) 
  (ha : a ≠ 0) : 
  (Nat.choose 8 4 * (a / x)^4 * (x / a^3)^4 = 70 / a^8) :=
by
  -- Applying the binomial theorem and simplifying the expression
  rw [Nat.choose]
  sorry

end NUMINAMATH_GPT_binomial_theorem_fifth_term_l2361_236174


namespace NUMINAMATH_GPT_simplest_common_denominator_l2361_236162

theorem simplest_common_denominator (x y : ℕ) (h1 : 2 * x ≠ 0) (h2 : 4 * y^2 ≠ 0) (h3 : 5 * x * y ≠ 0) :
  ∃ d : ℕ, d = 20 * x * y^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplest_common_denominator_l2361_236162


namespace NUMINAMATH_GPT_black_cars_count_l2361_236180

-- Conditions
def red_cars : ℕ := 28
def ratio_red_black : ℚ := 3 / 8

-- Theorem statement
theorem black_cars_count :
  ∃ (black_cars : ℕ), black_cars = 75 ∧ (red_cars : ℚ) / (black_cars) = ratio_red_black :=
sorry

end NUMINAMATH_GPT_black_cars_count_l2361_236180


namespace NUMINAMATH_GPT_geom_seq_inverse_sum_l2361_236133

theorem geom_seq_inverse_sum 
  (a_2 a_3 a_4 a_5 : ℚ) 
  (h1 : a_2 * a_5 = -3 / 4) 
  (h2 : a_2 + a_3 + a_4 + a_5 = 5 / 4) :
  1 / a_2 + 1 / a_3 + 1 / a_4 + 1 / a_5 = -4 / 3 :=
sorry

end NUMINAMATH_GPT_geom_seq_inverse_sum_l2361_236133


namespace NUMINAMATH_GPT_a_101_mod_49_l2361_236157

def a (n : ℕ) : ℕ := 5 ^ n + 9 ^ n

theorem a_101_mod_49 : (a 101) % 49 = 0 :=
by
  -- proof to be filled here
  sorry

end NUMINAMATH_GPT_a_101_mod_49_l2361_236157


namespace NUMINAMATH_GPT_probability_of_observing_color_change_l2361_236119

def cycle_duration := 100
def observation_interval := 4
def change_times := [45, 50, 100]

def probability_of_change : ℚ :=
  (observation_interval * change_times.length : ℚ) / cycle_duration

theorem probability_of_observing_color_change :
  probability_of_change = 0.12 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_of_observing_color_change_l2361_236119


namespace NUMINAMATH_GPT_cryptarithm_no_solution_proof_l2361_236196

def cryptarithm_no_solution : Prop :=
  ∀ (D O N K A L E V G R : ℕ),
    D ≠ O ∧ D ≠ N ∧ D ≠ K ∧ D ≠ A ∧ D ≠ L ∧ D ≠ E ∧ D ≠ V ∧ D ≠ G ∧ D ≠ R ∧
    O ≠ N ∧ O ≠ K ∧ O ≠ A ∧ O ≠ L ∧ O ≠ E ∧ O ≠ V ∧ O ≠ G ∧ O ≠ R ∧
    N ≠ K ∧ N ≠ A ∧ N ≠ L ∧ N ≠ E ∧ N ≠ V ∧ N ≠ G ∧ N ≠ R ∧
    K ≠ A ∧ K ≠ L ∧ K ≠ E ∧ K ≠ V ∧ K ≠ G ∧ K ≠ R ∧
    A ≠ L ∧ A ≠ E ∧ A ≠ V ∧ A ≠ G ∧ A ≠ R ∧
    L ≠ E ∧ L ≠ V ∧ L ≠ G ∧ L ≠ R ∧
    E ≠ V ∧ E ≠ G ∧ E ≠ R ∧
    V ≠ G ∧ V ≠ R ∧
    G ≠ R ∧
    (D * 100 + O * 10 + N) + (O * 100 + K * 10 + A) +
    (L * 1000 + E * 100 + N * 10 + A) + (V * 10000 + O * 1000 + L * 100 + G * 10 + A) =
    A * 100000 + N * 10000 + G * 1000 + A * 100 + R * 10 + A →
    false

theorem cryptarithm_no_solution_proof : cryptarithm_no_solution :=
by sorry

end NUMINAMATH_GPT_cryptarithm_no_solution_proof_l2361_236196


namespace NUMINAMATH_GPT_carla_correct_questions_l2361_236135

theorem carla_correct_questions :
  ∀ (Drew_correct Drew_wrong Carla_wrong Total_questions Carla_correct : ℕ), 
    Drew_correct = 20 →
    Drew_wrong = 6 →
    Carla_wrong = 2 * Drew_wrong →
    Total_questions = 52 →
    Carla_correct = Total_questions - Carla_wrong →
    Carla_correct = 40 :=
by
  intros Drew_correct Drew_wrong Carla_wrong Total_questions Carla_correct
  intros h1 h2 h3 h4 h5
  subst_vars
  sorry

end NUMINAMATH_GPT_carla_correct_questions_l2361_236135


namespace NUMINAMATH_GPT_solution_set_fraction_inequality_l2361_236136

theorem solution_set_fraction_inequality (x : ℝ) : 
  (x + 1) / (x - 1) ≤ 0 ↔ -1 ≤ x ∧ x < 1 :=
sorry

end NUMINAMATH_GPT_solution_set_fraction_inequality_l2361_236136


namespace NUMINAMATH_GPT_evaluate_expression_l2361_236189

variables (a b c : ℝ)

theorem evaluate_expression (h1 : c = b - 20) (h2 : b = a + 4) (h3 : a = 2)
  (h4 : a^2 + a ≠ 0) (h5 : b^2 - 6 * b + 8 ≠ 0) (h6 : c^2 + 12 * c + 36 ≠ 0):
  (a^2 + 2 * a) / (a^2 + a) * (b^2 - 4) / (b^2 - 6 * b + 8) * (c^2 + 16 * c + 64) / (c^2 + 12 * c + 36) = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l2361_236189


namespace NUMINAMATH_GPT_additional_time_due_to_leak_is_six_l2361_236182

open Real

noncomputable def filling_time_with_leak (R L : ℝ) : ℝ := 1 / (R - L)
noncomputable def filling_time_without_leak (R : ℝ) : ℝ := 1 / R
noncomputable def additional_filling_time (R L : ℝ) : ℝ :=
  filling_time_with_leak R L - filling_time_without_leak R

theorem additional_time_due_to_leak_is_six :
  additional_filling_time 0.25 (3 / 20) = 6 := by
  sorry

end NUMINAMATH_GPT_additional_time_due_to_leak_is_six_l2361_236182


namespace NUMINAMATH_GPT_largest_among_a_b_c_l2361_236150

theorem largest_among_a_b_c (x : ℝ) (h0 : 0 < x) (h1 : x < 1)
  (a : ℝ := 2 * Real.sqrt x) 
  (b : ℝ := 1 + x) 
  (c : ℝ := 1 / (1 - x)) : c > b ∧ b > a := by
  sorry

end NUMINAMATH_GPT_largest_among_a_b_c_l2361_236150


namespace NUMINAMATH_GPT_calc_expression_l2361_236184

theorem calc_expression : 
  (abs (Real.sqrt 2 - Real.sqrt 3) + 2 * Real.cos (Real.pi / 4) - Real.sqrt 2 * Real.sqrt 6 = -Real.sqrt 3) :=
by
  -- Given that sqrt(3) > sqrt(2)
  have h1 : Real.sqrt 3 > Real.sqrt 2 := by sorry
  -- And cos(45°) = sqrt(2)/2
  have h2 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  -- Now prove the expression equivalency
  sorry

end NUMINAMATH_GPT_calc_expression_l2361_236184


namespace NUMINAMATH_GPT_area_of_square_field_l2361_236114

theorem area_of_square_field (s : ℕ) (area : ℕ) (cost_per_meter : ℕ) (total_cost : ℕ) (gate_width : ℕ) :
  (cost_per_meter = 3) →
  (total_cost = 1998) →
  (gate_width = 1) →
  (total_cost = cost_per_meter * (4 * s - 2 * gate_width)) →
  (area = s^2) →
  area = 27889 :=
by
  intros h_cost_per_meter h_total_cost h_gate_width h_cost_eq h_area_eq
  sorry

end NUMINAMATH_GPT_area_of_square_field_l2361_236114


namespace NUMINAMATH_GPT_shaded_area_of_three_circles_l2361_236199

theorem shaded_area_of_three_circles :
  (∀ (r1 r2 : ℝ), (π * r1^2 = 100 * π) → (r2 = r1 / 2) → (shaded_area = (π * r1^2) / 2 + 2 * ((π * r2^2) / 2)) → (shaded_area = 75 * π)) :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_of_three_circles_l2361_236199


namespace NUMINAMATH_GPT_john_has_25_roommates_l2361_236176

def roommates_of_bob := 10
def roommates_of_john := 2 * roommates_of_bob + 5

theorem john_has_25_roommates : roommates_of_john = 25 := 
by
  sorry

end NUMINAMATH_GPT_john_has_25_roommates_l2361_236176


namespace NUMINAMATH_GPT_tallest_player_height_correct_l2361_236120

-- Define the height of the shortest player
def shortest_player_height : ℝ := 68.25

-- Define the height difference between the tallest and shortest player
def height_difference : ℝ := 9.5

-- Define the height of the tallest player based on the conditions
def tallest_player_height : ℝ :=
  shortest_player_height + height_difference

-- Theorem statement
theorem tallest_player_height_correct : tallest_player_height = 77.75 := by
  sorry

end NUMINAMATH_GPT_tallest_player_height_correct_l2361_236120


namespace NUMINAMATH_GPT_initial_money_eq_l2361_236115

-- Definitions for the problem conditions
def spent_on_sweets : ℝ := 1.25
def spent_on_friends : ℝ := 2 * 1.20
def money_left : ℝ :=  4.85

-- Statement of the problem to prove
theorem initial_money_eq :
  spent_on_sweets + spent_on_friends + money_left = 8.50 := 
sorry

end NUMINAMATH_GPT_initial_money_eq_l2361_236115


namespace NUMINAMATH_GPT_vertical_asymptote_singleton_l2361_236172

theorem vertical_asymptote_singleton (c : ℝ) :
  (∃ x, (x^2 - 2 * x + c) = 0 ∧ ((x - 1) * (x + 3) = 0) ∧ (x ≠ 1 ∨ x ≠ -3)) 
  ↔ (c = 1 ∨ c = -15) :=
by
  sorry

end NUMINAMATH_GPT_vertical_asymptote_singleton_l2361_236172


namespace NUMINAMATH_GPT_minimize_sum_of_cubes_l2361_236116

theorem minimize_sum_of_cubes (x y : ℝ) (h : x + y = 8) : 
  (3 * x^2 - 3 * (8 - x)^2 = 0) → (x = 4) ∧ (y = 4) :=
by
  sorry

end NUMINAMATH_GPT_minimize_sum_of_cubes_l2361_236116


namespace NUMINAMATH_GPT_no_three_even_segments_with_odd_intersections_l2361_236159

open Set

def is_even_length (s : Set ℝ) : Prop :=
  ∃ a b : ℝ, s = Icc a b ∧ (b - a) % 2 = 0

def is_odd_length (s : Set ℝ) : Prop :=
  ∃ a b : ℝ, s = Icc a b ∧ (b - a) % 2 = 1

theorem no_three_even_segments_with_odd_intersections :
  ¬ ∃ (S1 S2 S3 : Set ℝ),
    (is_even_length S1) ∧
    (is_even_length S2) ∧
    (is_even_length S3) ∧
    (is_odd_length (S1 ∩ S2)) ∧
    (is_odd_length (S1 ∩ S3)) ∧
    (is_odd_length (S2 ∩ S3)) :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_no_three_even_segments_with_odd_intersections_l2361_236159


namespace NUMINAMATH_GPT_soccer_balls_with_holes_l2361_236108

-- Define the total number of soccer balls
def total_soccer_balls : ℕ := 40

-- Define the total number of basketballs
def total_basketballs : ℕ := 15

-- Define the number of basketballs with holes
def basketballs_with_holes : ℕ := 7

-- Define the total number of balls without holes
def total_balls_without_holes : ℕ := 18

-- Prove the number of soccer balls with holes given the conditions
theorem soccer_balls_with_holes : (total_soccer_balls - (total_balls_without_holes - (total_basketballs - basketballs_with_holes))) = 30 := by
  sorry

end NUMINAMATH_GPT_soccer_balls_with_holes_l2361_236108


namespace NUMINAMATH_GPT_soda_cost_90_cents_l2361_236147

theorem soda_cost_90_cents
  (b s : ℕ)
  (h1 : 3 * b + 2 * s = 360)
  (h2 : 2 * b + 4 * s = 480) :
  s = 90 :=
by
  sorry

end NUMINAMATH_GPT_soda_cost_90_cents_l2361_236147


namespace NUMINAMATH_GPT_pizzaCostPerSlice_l2361_236134

/-- Define the constants and parameters for the problem --/
def largePizzaCost : ℝ := 10.00
def numberOfSlices : ℕ := 8
def firstToppingCost : ℝ := 2.00
def secondThirdToppingCost : ℝ := 1.00
def otherToppingCost : ℝ := 0.50
def toppings : List String := ["pepperoni", "sausage", "ham", "olives", "mushrooms", "bell peppers", "pineapple"]

/-- Calculate the total number of toppings --/
def numberOfToppings : ℕ := toppings.length

/-- Calculate the total cost of the pizza including all toppings --/
noncomputable def totalPizzaCost : ℝ :=
  largePizzaCost + 
  firstToppingCost + 
  2 * secondThirdToppingCost + 
  (numberOfToppings - 3) * otherToppingCost

/-- Calculate the cost per slice --/
noncomputable def costPerSlice : ℝ := totalPizzaCost / numberOfSlices

/-- Proof statement: The cost per slice is $2.00 --/
theorem pizzaCostPerSlice : costPerSlice = 2 := by
  sorry

end NUMINAMATH_GPT_pizzaCostPerSlice_l2361_236134


namespace NUMINAMATH_GPT_leif_apples_l2361_236149

-- Definitions based on conditions
def oranges : ℕ := 24
def apples (oranges apples_diff : ℕ) := oranges - apples_diff

-- Theorem stating the problem to prove
theorem leif_apples (oranges apples_diff : ℕ) (h1 : oranges = 24) (h2 : apples_diff = 10) : apples oranges apples_diff = 14 :=
by
  -- Using the definition of apples and given conditions, prove the number of apples
  rw [h1, h2]
  -- Calculating the number of apples
  show 24 - 10 = 14
  rfl

end NUMINAMATH_GPT_leif_apples_l2361_236149


namespace NUMINAMATH_GPT_proof_problem_l2361_236151

variable (x y : ℝ)

noncomputable def condition1 : Prop := x > y
noncomputable def condition2 : Prop := x * y = 1

theorem proof_problem (hx : condition1 x y) (hy : condition2 x y) : 
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2361_236151


namespace NUMINAMATH_GPT_integer_values_of_x_in_triangle_l2361_236191

theorem integer_values_of_x_in_triangle (x : ℝ) :
  (x + 14 > 38 ∧ x + 38 > 14 ∧ 14 + 38 > x) → 
  ∃ (n : ℕ), n = 27 ∧ ∀ m : ℕ, (24 < m ∧ m < 52 ↔ (m : ℝ) > 24 ∧ (m : ℝ) < 52) :=
by {
  sorry
}

end NUMINAMATH_GPT_integer_values_of_x_in_triangle_l2361_236191


namespace NUMINAMATH_GPT_polynomial_identity_l2361_236167

theorem polynomial_identity (x : ℝ) : 
  (2 * x^2 + 5 * x + 8) * (x + 1) - (x + 1) * (x^2 - 2 * x + 50) 
  + (3 * x - 7) * (x + 1) * (x - 2) = 4 * x^3 - 2 * x^2 - 34 * x - 28 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_identity_l2361_236167


namespace NUMINAMATH_GPT_evaluate_expression_l2361_236165

theorem evaluate_expression : (8^5 / 8^2) * 2^12 = 2^21 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2361_236165


namespace NUMINAMATH_GPT_karlson_wins_with_optimal_play_l2361_236179

def game_win_optimal_play: Prop :=
  ∀ (total_moves: ℕ), 
  (total_moves % 2 = 1) 

theorem karlson_wins_with_optimal_play: game_win_optimal_play :=
by sorry

end NUMINAMATH_GPT_karlson_wins_with_optimal_play_l2361_236179


namespace NUMINAMATH_GPT_first_year_with_digit_sum_seven_l2361_236195

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_with_digit_sum_seven : ∃ y, y > 2023 ∧ sum_of_digits y = 7 ∧ ∀ z, z > 2023 ∧ z < y → sum_of_digits z ≠ 7 :=
by
  use 2032
  sorry

end NUMINAMATH_GPT_first_year_with_digit_sum_seven_l2361_236195


namespace NUMINAMATH_GPT_increase_number_correct_l2361_236102

-- Definitions for the problem
def originalNumber : ℕ := 110
def increasePercent : ℝ := 0.5

-- Statement to be proved
theorem increase_number_correct : originalNumber + (originalNumber * increasePercent) = 165 := by
  sorry

end NUMINAMATH_GPT_increase_number_correct_l2361_236102


namespace NUMINAMATH_GPT_range_of_fx_l2361_236130

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^k

theorem range_of_fx (k : ℝ) (x : ℝ) (h1 : k < -1) (h2 : x ∈ Set.Ici (0.5)) :
  Set.Icc (0 : ℝ) 2 = {y | ∃ x, f x k = y ∧ x ∈ Set.Ici 0.5} :=
sorry

end NUMINAMATH_GPT_range_of_fx_l2361_236130


namespace NUMINAMATH_GPT_james_total_toys_l2361_236103

-- Definition for the number of toy cars
def numToyCars : ℕ := 20

-- Definition for the number of toy soldiers
def numToySoldiers : ℕ := 2 * numToyCars

-- The total number of toys is the sum of toy cars and toy soldiers
def totalToys : ℕ := numToyCars + numToySoldiers

-- Statement to prove: James buys a total of 60 toys
theorem james_total_toys : totalToys = 60 := by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_james_total_toys_l2361_236103


namespace NUMINAMATH_GPT_functional_equation_solution_l2361_236148

theorem functional_equation_solution (f : ℕ → ℕ) 
  (H : ∀ a b : ℕ, f (f a + f b) = a + b) : 
  ∀ n : ℕ, f n = n := 
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l2361_236148


namespace NUMINAMATH_GPT_part1_part2_l2361_236132

def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1 (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} :=
sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2361_236132


namespace NUMINAMATH_GPT_emma_age_l2361_236187

variables (O N L E : ℕ)

def oliver_eq : Prop := O = N - 5
def nancy_eq : Prop := N = L + 6
def emma_eq : Prop := E = L + 4
def oliver_age : Prop := O = 16

theorem emma_age :
  oliver_eq O N ∧ nancy_eq N L ∧ emma_eq E L ∧ oliver_age O → E = 19 :=
by
  sorry

end NUMINAMATH_GPT_emma_age_l2361_236187


namespace NUMINAMATH_GPT_math_proof_problem_l2361_236175

noncomputable def ellipse_standard_eq (a b : ℝ) : Prop :=
  a = 4 ∧ b = 2 * Real.sqrt 3

noncomputable def conditions (e : ℝ) (vertex : ℝ × ℝ) (p q : ℝ × ℝ) : Prop :=
  e = 1 / 2
  ∧ vertex = (0, 2 * Real.sqrt 3)  -- focus of the parabola
  ∧ p = (-2, -3)
  ∧ q = (-2, 3)

noncomputable def max_area_quadrilateral (area : ℝ) : Prop :=
  area = 12 * Real.sqrt 3

theorem math_proof_problem : 
  ∃ a b p q area, ellipse_standard_eq a b ∧ conditions (1/2) (0, 2 * Real.sqrt 3) p q 
  ∧ p = (-2, -3) ∧ q = (-2, 3) → max_area_quadrilateral area := 
  sorry

end NUMINAMATH_GPT_math_proof_problem_l2361_236175


namespace NUMINAMATH_GPT_fraction_of_gasoline_used_l2361_236194

-- Define the conditions
def gasoline_per_mile := 1 / 30  -- Gallons per mile
def full_tank := 12  -- Gallons
def speed := 60  -- Miles per hour
def travel_time := 5  -- Hours

-- Total distance traveled
def distance := speed * travel_time  -- Miles

-- Gasoline used
def gasoline_used := distance * gasoline_per_mile  -- Gallons

-- Fraction of the full tank used
def fraction_used := gasoline_used / full_tank

-- The theorem to be proved
theorem fraction_of_gasoline_used :
  fraction_used = 5 / 6 :=
by sorry

end NUMINAMATH_GPT_fraction_of_gasoline_used_l2361_236194


namespace NUMINAMATH_GPT_Q_current_age_l2361_236143

-- Definitions for the current ages of P and Q
variable (P Q : ℕ)

-- Conditions
-- 1. P + Q = 100
-- 2. P = 3 * (Q - (P - Q))  (from P is thrice as old as Q was when P was as old as Q is now)

axiom age_sum : P + Q = 100
axiom age_relation : P = 3 * (Q - (P - Q))

theorem Q_current_age : Q = 40 :=
by
  sorry

end NUMINAMATH_GPT_Q_current_age_l2361_236143


namespace NUMINAMATH_GPT_max_min_values_on_circle_l2361_236160

def on_circle (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 - 4 * x - 4 * y + 7 = 0

theorem max_min_values_on_circle (x y : ℝ) (h : on_circle x y) :
  16 ≤ (x + 1) ^ 2 + (y + 2) ^ 2 ∧ (x + 1) ^ 2 + (y + 2) ^ 2 ≤ 36 :=
  sorry

end NUMINAMATH_GPT_max_min_values_on_circle_l2361_236160


namespace NUMINAMATH_GPT_A_beats_B_by_7_seconds_l2361_236106

noncomputable def speed_A : ℝ := 200 / 33
noncomputable def distance_A : ℝ := 200
noncomputable def time_A : ℝ := 33

noncomputable def distance_B : ℝ := 200
noncomputable def distance_B_at_time_A : ℝ := 165

-- B's speed is calculated at the moment A finishes the race
noncomputable def speed_B : ℝ := distance_B_at_time_A / time_A
noncomputable def time_B : ℝ := distance_B / speed_B

-- Prove that A beats B by 7 seconds
theorem A_beats_B_by_7_seconds : time_B - time_A = 7 := 
by 
  -- Proof goes here, assume all definitions and variables are correct.
  sorry

end NUMINAMATH_GPT_A_beats_B_by_7_seconds_l2361_236106


namespace NUMINAMATH_GPT_sum_of_powers_mod7_l2361_236104

theorem sum_of_powers_mod7 (k : ℕ) : (2^k + 3^k) % 7 = 0 ↔ k % 6 = 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_mod7_l2361_236104


namespace NUMINAMATH_GPT_brenda_num_cookies_per_box_l2361_236192

def numCookiesPerBox (trays : ℕ) (cookiesPerTray : ℕ) (costPerBox : ℚ) (totalSpent : ℚ) : ℚ :=
  let totalCookies := trays * cookiesPerTray
  let numBoxes := totalSpent / costPerBox
  totalCookies / numBoxes

theorem brenda_num_cookies_per_box :
  numCookiesPerBox 3 80 3.5 14 = 60 := by
  sorry

end NUMINAMATH_GPT_brenda_num_cookies_per_box_l2361_236192


namespace NUMINAMATH_GPT_system_of_equations_solution_l2361_236142

theorem system_of_equations_solution (x y : ℝ) 
  (h1 : x + 3 * y = 7) 
  (h2 : y = 2 * x) : 
  x = 1 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l2361_236142


namespace NUMINAMATH_GPT_part1_part2_l2361_236118

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4 * x + a + 3
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := b * x + 5 - 2 * b

theorem part1 (a : ℝ) : (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x a = 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
sorry

theorem part2 (b : ℝ) : (∀ x1 : ℝ, 1 ≤ x1 ∧ x1 ≤ 4 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 4 ∧ f x2 3 = g x1 b) ↔ -1 ≤ b ∧ b ≤ 1/2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2361_236118


namespace NUMINAMATH_GPT_seating_possible_l2361_236112

theorem seating_possible (n : ℕ) (guests : Fin (2 * n) → Finset (Fin (2 * n))) 
  (h1 : ∀ i, n ≤ (guests i).card)
  (h2 : ∀ i j, (i ≠ j) → i ∈ guests j → j ∈ guests i) : 
  ∃ (a b c d : Fin (2 * n)), 
    (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ d) ∧ (d ≠ a) ∧
    (a ∈ guests b) ∧ (b ∈ guests c) ∧ (c ∈ guests d) ∧ (d ∈ guests a) := 
sorry

end NUMINAMATH_GPT_seating_possible_l2361_236112


namespace NUMINAMATH_GPT_sin_neg_4_div_3_pi_l2361_236190

theorem sin_neg_4_div_3_pi : Real.sin (- (4 / 3) * Real.pi) = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_GPT_sin_neg_4_div_3_pi_l2361_236190


namespace NUMINAMATH_GPT_chair_capacity_l2361_236152

theorem chair_capacity
  (total_chairs : ℕ)
  (total_board_members : ℕ)
  (not_occupied_fraction : ℚ)
  (occupied_people_per_chair : ℕ)
  (attending_board_members : ℕ)
  (total_chairs_eq : total_chairs = 40)
  (not_occupied_fraction_eq : not_occupied_fraction = 2/5)
  (occupied_people_per_chair_eq : occupied_people_per_chair = 2)
  (attending_board_members_eq : attending_board_members = 48)
  : total_board_members = 48 := 
by
  sorry

end NUMINAMATH_GPT_chair_capacity_l2361_236152


namespace NUMINAMATH_GPT_no_such_b_exists_l2361_236168

theorem no_such_b_exists (k n : ℕ) (a : ℕ) 
  (hk : Odd k) (hn : Odd n)
  (hk_gt_one : k > 1) (hn_gt_one : n > 1) 
  (hka : k ∣ 2^a + 1) (hna : n ∣ 2^a - 1) : 
  ¬ ∃ b : ℕ, k ∣ 2^b - 1 ∧ n ∣ 2^b + 1 :=
sorry

end NUMINAMATH_GPT_no_such_b_exists_l2361_236168


namespace NUMINAMATH_GPT_calculate_first_worker_time_l2361_236128

theorem calculate_first_worker_time
    (T : ℝ)
    (h : 1/T + 1/4 = 1/2.2222222222222223) :
    T = 5 := sorry

end NUMINAMATH_GPT_calculate_first_worker_time_l2361_236128


namespace NUMINAMATH_GPT_maximum_value_of_k_minus_b_l2361_236161

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x - a * x + b

theorem maximum_value_of_k_minus_b (b : ℝ) (k : ℝ) (x : ℝ) 
  (h₀ : 0 ≤ b ∧ b ≤ 2) 
  (h₁ : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h₂ : ∀ x ∈ Set.Icc 1 (Real.exp 1), f x 1 b ≥ (k * x - x * Real.log x - 1)) :
  k - b ≤ 0 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_k_minus_b_l2361_236161


namespace NUMINAMATH_GPT_sector_area_is_2pi_l2361_236126

/-- Problem Statement: Prove that the area of a sector of a circle with radius 4 and central
    angle 45° (or π/4 radians) is 2π. -/
theorem sector_area_is_2pi (r : ℝ) (θ : ℝ) (h_r : r = 4) (h_θ : θ = π / 4) :
  (1 / 2) * θ * r^2 = 2 * π :=
by
  rw [h_r, h_θ]
  sorry

end NUMINAMATH_GPT_sector_area_is_2pi_l2361_236126


namespace NUMINAMATH_GPT_arc_length_TQ_l2361_236100

-- Definitions from the conditions
def center (O : Type) : Prop := true
def inscribedAngle (T I Q : Type) (angle : ℝ) := angle = 45
def radius (T : Type) (len : ℝ) := len = 12

-- Theorem to be proved
theorem arc_length_TQ (O T I Q : Type) (r : ℝ) (angle : ℝ) 
  (h_center : center O) 
  (h_angle : inscribedAngle T I Q angle)
  (h_radius : radius T r) :
  ∃ l : ℝ, l = 6 * Real.pi := 
sorry

end NUMINAMATH_GPT_arc_length_TQ_l2361_236100


namespace NUMINAMATH_GPT_part_I_part_II_l2361_236145

def f (x a : ℝ) := |2 * x - a| + 5 * x

theorem part_I (x : ℝ) : f x 3 ≥ 5 * x + 1 ↔ (x ≤ 1 ∨ x ≥ 2) := sorry

theorem part_II (a x : ℝ) (h : (∀ x, f x a ≤ 0 ↔ x ≤ -1)) : a = 3 := sorry

end NUMINAMATH_GPT_part_I_part_II_l2361_236145


namespace NUMINAMATH_GPT_polynomial_expansion_correct_l2361_236154

def polynomial1 (z : ℤ) : ℤ := 3 * z^3 + 4 * z^2 - 5
def polynomial2 (z : ℤ) : ℤ := 4 * z^4 - 3 * z^2 + 2
def expandedPolynomial (z : ℤ) : ℤ := 12 * z^7 + 16 * z^6 - 9 * z^5 - 32 * z^4 + 6 * z^3 + 23 * z^2 - 10

theorem polynomial_expansion_correct (z : ℤ) :
  (polynomial1 z) * (polynomial2 z) = expandedPolynomial z :=
by sorry

end NUMINAMATH_GPT_polynomial_expansion_correct_l2361_236154


namespace NUMINAMATH_GPT_george_run_speed_last_half_mile_l2361_236170

theorem george_run_speed_last_half_mile :
  ∀ (distance_school normal_speed first_segment_distance first_segment_speed second_segment_distance second_segment_speed remaining_distance)
    (today_total_time normal_total_time remaining_time : ℝ),
    distance_school = 2 →
    normal_speed = 4 →
    first_segment_distance = 3 / 4 →
    first_segment_speed = 3 →
    second_segment_distance = 3 / 4 →
    second_segment_speed = 4 →
    remaining_distance = 1 / 2 →
    normal_total_time = distance_school / normal_speed →
    today_total_time = (first_segment_distance / first_segment_speed) + (second_segment_distance / second_segment_speed) →
    normal_total_time = today_total_time + remaining_time →
    (remaining_distance / remaining_time) = 8 :=
by
  intros distance_school normal_speed first_segment_distance first_segment_speed second_segment_distance second_segment_speed remaining_distance today_total_time normal_total_time remaining_time h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end NUMINAMATH_GPT_george_run_speed_last_half_mile_l2361_236170


namespace NUMINAMATH_GPT_highway_length_proof_l2361_236129

variable (L : ℝ) (v1 v2 : ℝ) (t : ℝ)

def highway_length : Prop :=
  v1 = 55 ∧ v2 = 35 ∧ t = 1 / 15 ∧ (L / v2 - L / v1 = t) ∧ L = 6.42

theorem highway_length_proof : highway_length L 55 35 (1 / 15) := by
  sorry

end NUMINAMATH_GPT_highway_length_proof_l2361_236129


namespace NUMINAMATH_GPT_calculation_101_squared_minus_99_squared_l2361_236155

theorem calculation_101_squared_minus_99_squared : 101^2 - 99^2 = 400 :=
by
  sorry

end NUMINAMATH_GPT_calculation_101_squared_minus_99_squared_l2361_236155


namespace NUMINAMATH_GPT_temperature_difference_l2361_236109

def lowest_temp : ℝ := -15
def highest_temp : ℝ := 3

theorem temperature_difference :
  highest_temp - lowest_temp = 18 :=
by
  sorry

end NUMINAMATH_GPT_temperature_difference_l2361_236109


namespace NUMINAMATH_GPT_rhombus_area_l2361_236138

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 30) (h2 : d2 = 16) : (d1 * d2) / 2 = 240 := by
  sorry

end NUMINAMATH_GPT_rhombus_area_l2361_236138


namespace NUMINAMATH_GPT_man_speed_down_l2361_236101

variable (d : ℝ) (v : ℝ)

theorem man_speed_down (h1 : 32 > 0) (h2 : 38.4 > 0) (h3 : d > 0) (h4 : v > 0) 
  (avg_speed : 38.4 = (2 * d) / ((d / 32) + (d / v))) : v = 48 :=
sorry

end NUMINAMATH_GPT_man_speed_down_l2361_236101


namespace NUMINAMATH_GPT_playdough_cost_l2361_236141

-- Definitions of the costs and quantities
def lego_cost := 250
def sword_cost := 120
def playdough_quantity := 10
def total_paid := 1940

-- Variables representing the quantities bought
def lego_quantity := 3
def sword_quantity := 7

-- Function to calculate the total cost for lego and sword
def total_lego_cost := lego_quantity * lego_cost
def total_sword_cost := sword_quantity * sword_cost

-- Variable representing the cost of playdough
variable (P : ℝ)

-- The main statement to prove
theorem playdough_cost :
  total_lego_cost + total_sword_cost + playdough_quantity * P = total_paid → P = 35 :=
by
  sorry

end NUMINAMATH_GPT_playdough_cost_l2361_236141


namespace NUMINAMATH_GPT_school_team_profit_is_333_l2361_236113

noncomputable def candy_profit (total_bars : ℕ) (price_800_bars : ℕ) (price_400_bars : ℕ) (sold_600_bars_price : ℕ) (remaining_600_bars_price : ℕ) : ℚ :=
  let cost_800_bars := 800 / 3
  let cost_400_bars := 400 / 4
  let total_cost := cost_800_bars + cost_400_bars
  let revenue_sold_600_bars := 600 / 2
  let revenue_remaining_600_bars := (600 * 2) / 3
  let total_revenue := revenue_sold_600_bars + revenue_remaining_600_bars
  total_revenue - total_cost

theorem school_team_profit_is_333 :
  candy_profit 1200 3 4 2 2 = 333 := by
  sorry

end NUMINAMATH_GPT_school_team_profit_is_333_l2361_236113


namespace NUMINAMATH_GPT_min_value_expression_l2361_236193

theorem min_value_expression :
  ∃ x > 0, x^2 + 6 * x + 100 / x^3 = 3 * (50:ℝ)^(2/5) + 6 * (50:ℝ)^(1/5) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l2361_236193


namespace NUMINAMATH_GPT_scientific_notation_of_virus_diameter_l2361_236140

theorem scientific_notation_of_virus_diameter :
  0.000000102 = 1.02 * 10 ^ (-7) :=
  sorry

end NUMINAMATH_GPT_scientific_notation_of_virus_diameter_l2361_236140


namespace NUMINAMATH_GPT_coefficient_of_expression_l2361_236163

theorem coefficient_of_expression :
  ∀ (a b : ℝ), (∃ (c : ℝ), - (2/3) * (a * b) = c * (a * b)) :=
by
  intros a b
  use (-2/3)
  sorry

end NUMINAMATH_GPT_coefficient_of_expression_l2361_236163


namespace NUMINAMATH_GPT_line_through_point_l2361_236146

theorem line_through_point (k : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, (x = 3) ∧ (y = -2) → (2 - 3 * k * x = -4 * y)) → k = -2/3 :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_l2361_236146


namespace NUMINAMATH_GPT_toadon_population_percentage_l2361_236171

theorem toadon_population_percentage {pop_total G L T : ℕ}
    (h_total : pop_total = 80000)
    (h_gordonia : G = pop_total / 2)
    (h_lakebright : L = 16000)
    (h_total_population : pop_total = G + T + L) :
    (T * 100 / G) = 60 :=
by sorry

end NUMINAMATH_GPT_toadon_population_percentage_l2361_236171


namespace NUMINAMATH_GPT_kendra_sunday_shirts_l2361_236124

def total_shirts := 22
def shirts_weekdays := 5 * 1
def shirts_after_school := 3
def shirts_saturday := 1

theorem kendra_sunday_shirts : 
  (total_shirts - 2 * (shirts_weekdays + shirts_after_school + shirts_saturday)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_kendra_sunday_shirts_l2361_236124


namespace NUMINAMATH_GPT_MaximMethod_CorrectNumber_l2361_236156

theorem MaximMethod_CorrectNumber (x y : ℕ) (N : ℕ) (h_digit_x : 0 ≤ x ∧ x ≤ 9) (h_digit_y : 1 ≤ y ∧ y ≤ 9)
  (h_N : N = 10 * x + y)
  (h_condition : 1 / (10 * x + y : ℚ) = 1 / (x + y : ℚ) - 1 / (x * y : ℚ)) :
  N = 24 :=
sorry

end NUMINAMATH_GPT_MaximMethod_CorrectNumber_l2361_236156


namespace NUMINAMATH_GPT_fraction_to_decimal_l2361_236158

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l2361_236158


namespace NUMINAMATH_GPT_sum_of_ages_is_60_l2361_236110

theorem sum_of_ages_is_60 (A B : ℕ) (h1 : A = 2 * B) (h2 : (A + 3) + (B + 3) = 66) : A + B = 60 :=
by sorry

end NUMINAMATH_GPT_sum_of_ages_is_60_l2361_236110


namespace NUMINAMATH_GPT_find_all_functions_l2361_236144

theorem find_all_functions (f : ℕ → ℕ) : 
  (∀ a b : ℕ, 0 < a → 0 < b → f (a^2 + b^2) = f a * f b) →
  (∀ a : ℕ, 0 < a → f (a^2) = f a ^ 2) →
  (∀ n : ℕ, 0 < n → f n = 1) :=
by
  intros h1 h2 a ha
  sorry

end NUMINAMATH_GPT_find_all_functions_l2361_236144


namespace NUMINAMATH_GPT_min_num_of_teams_l2361_236122

theorem min_num_of_teams (num_athletes : ℕ) (max_team_size : ℕ) (h1 : num_athletes = 30) (h2 : max_team_size = 9) :
  ∃ (min_teams : ℕ), min_teams = 5 ∧ (∀ nal : ℕ, (nal > 0 ∧ num_athletes % nal = 0 ∧ nal ≤ max_team_size) → num_athletes / nal ≥ min_teams) :=
by
  sorry

end NUMINAMATH_GPT_min_num_of_teams_l2361_236122


namespace NUMINAMATH_GPT_license_plate_count_l2361_236178

theorem license_plate_count :
  let digits := 10
  let letters := 26
  let positions := 6
  positions * digits^5 * letters^3 = 105456000 := by
  sorry

end NUMINAMATH_GPT_license_plate_count_l2361_236178


namespace NUMINAMATH_GPT_expand_expression_l2361_236121

variable (x : ℝ)

theorem expand_expression : 5 * (x + 3) * (2 * x - 4) = 10 * x^2 + 10 * x - 60 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l2361_236121


namespace NUMINAMATH_GPT_ellipse_major_axis_min_length_l2361_236186

theorem ellipse_major_axis_min_length (a b c : ℝ) 
  (h1 : b * c = 2)
  (h2 : a^2 = b^2 + c^2) 
  : 2 * a ≥ 4 :=
sorry

end NUMINAMATH_GPT_ellipse_major_axis_min_length_l2361_236186


namespace NUMINAMATH_GPT_quadratic_has_real_root_iff_b_in_interval_l2361_236173

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end NUMINAMATH_GPT_quadratic_has_real_root_iff_b_in_interval_l2361_236173


namespace NUMINAMATH_GPT_find_quadruples_l2361_236125

open Nat

theorem find_quadruples (a b p n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n)
    (h : a^3 + b^3 = p^n) :
    (∃ k, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3 * k + 1) ∨
    (∃ k, a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3 * k + 2) ∨
    (∃ k, a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3 * k + 2) :=
sorry

end NUMINAMATH_GPT_find_quadruples_l2361_236125


namespace NUMINAMATH_GPT_village_population_l2361_236197

theorem village_population (P : ℝ) (h : 0.8 * P = 64000) : P = 80000 := by
  sorry

end NUMINAMATH_GPT_village_population_l2361_236197


namespace NUMINAMATH_GPT_number_of_bought_bottle_caps_l2361_236139

/-- Define the initial number of bottle caps and the final number of bottle caps --/
def initial_bottle_caps : ℕ := 40
def final_bottle_caps : ℕ := 47

/-- Proof that the number of bottle caps Joshua bought is equal to 7 --/
theorem number_of_bought_bottle_caps : final_bottle_caps - initial_bottle_caps = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_bought_bottle_caps_l2361_236139


namespace NUMINAMATH_GPT_anthony_pencils_l2361_236181

def initial_pencils : ℝ := 56.0  -- Condition 1
def pencils_left : ℝ := 47.0     -- Condition 2
def pencils_given : ℝ := 9.0     -- Correct Answer

theorem anthony_pencils :
  initial_pencils - pencils_left = pencils_given :=
by
  sorry

end NUMINAMATH_GPT_anthony_pencils_l2361_236181


namespace NUMINAMATH_GPT_quadratic_inequality_solution_empty_l2361_236198

theorem quadratic_inequality_solution_empty (m : ℝ) :
  (∀ x : ℝ, ((m + 1) * x^2 - m * x + m - 1 < 0) → false) →
  (m ≥ (2 * Real.sqrt 3) / 3 ∨ m ≤ -(2 * Real.sqrt 3) / 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_empty_l2361_236198


namespace NUMINAMATH_GPT_product_not_divisible_by_770_l2361_236123

theorem product_not_divisible_by_770 (a b : ℕ) (h : a + b = 770) : ¬ (a * b) % 770 = 0 :=
sorry

end NUMINAMATH_GPT_product_not_divisible_by_770_l2361_236123
