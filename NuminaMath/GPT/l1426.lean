import Mathlib

namespace NUMINAMATH_GPT_probability_red_blue_green_l1426_142676

def total_marbles : ℕ := 5 + 4 + 3 + 6
def favorable_marbles : ℕ := 5 + 4 + 3

theorem probability_red_blue_green : 
  (favorable_marbles : ℚ) / total_marbles = 2 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_probability_red_blue_green_l1426_142676


namespace NUMINAMATH_GPT_evaluate_x_squared_minus_y_squared_l1426_142651

theorem evaluate_x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 18) :
  x^2 - y^2 = -72 := 
sorry

end NUMINAMATH_GPT_evaluate_x_squared_minus_y_squared_l1426_142651


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1426_142667

theorem arithmetic_sequence_sum (a d : ℚ) (a1 : a = 1 / 2) 
(S : ℕ → ℚ) (Sn : ∀ n, S n = n * a + (n * (n - 1) / 2) * d) 
(S2_eq_a3 : S 2 = a + 2 * d) :
  ∀ n, S n = (1 / 4 : ℚ) * n^2 + (1 / 4 : ℚ) * n :=
by
  intros n
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1426_142667


namespace NUMINAMATH_GPT_arccos_neg_one_eq_pi_l1426_142692

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
  sorry

end NUMINAMATH_GPT_arccos_neg_one_eq_pi_l1426_142692


namespace NUMINAMATH_GPT_megan_pages_left_l1426_142623

theorem megan_pages_left (total_problems completed_problems problems_per_page : ℕ)
    (h_total : total_problems = 40)
    (h_completed : completed_problems = 26)
    (h_problems_per_page : problems_per_page = 7) :
    (total_problems - completed_problems) / problems_per_page = 2 :=
by
  sorry

end NUMINAMATH_GPT_megan_pages_left_l1426_142623


namespace NUMINAMATH_GPT_problem_l1426_142690

noncomputable def x : ℝ := 123.75
noncomputable def y : ℝ := 137.5
noncomputable def original_value : ℝ := 125

theorem problem (y_more : y = original_value + 0.1 * original_value) (x_less : x = y * 0.9) : y = 137.5 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1426_142690


namespace NUMINAMATH_GPT_geom_arith_sequence_l1426_142660

theorem geom_arith_sequence (a b c m n : ℝ) 
  (h1 : b^2 = a * c) 
  (h2 : m = (a + b) / 2) 
  (h3 : n = (b + c) / 2) : 
  a / m + c / n = 2 := 
by 
  sorry

end NUMINAMATH_GPT_geom_arith_sequence_l1426_142660


namespace NUMINAMATH_GPT_side_length_of_square_l1426_142624

-- Define the areas of the triangles AOR, BOP, and CRQ
def S1 := 1
def S2 := 3
def S3 := 1

-- Prove that the side length of the square OPQR is 2
theorem side_length_of_square (side_length : ℝ) : 
  S1 = 1 ∧ S2 = 3 ∧ S3 = 1 → side_length = 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_side_length_of_square_l1426_142624


namespace NUMINAMATH_GPT_initial_flowers_per_bunch_l1426_142658

theorem initial_flowers_per_bunch (x : ℕ) (h₁: 8 * x = 72) : x = 9 :=
  by
  sorry

end NUMINAMATH_GPT_initial_flowers_per_bunch_l1426_142658


namespace NUMINAMATH_GPT_num_positive_integers_l1426_142669

theorem num_positive_integers (m : ℕ) : 
  (∃ n, m^2 - 2 = n ∧ n ∣ 2002) ↔ (m = 2 ∨ m = 3 ∨ m = 4) :=
by
  sorry

end NUMINAMATH_GPT_num_positive_integers_l1426_142669


namespace NUMINAMATH_GPT_min_score_needed_l1426_142650

-- Definitions of the conditions
def current_scores : List ℤ := [88, 92, 75, 81, 68, 70]
def desired_increase : ℤ := 5
def number_of_tests := current_scores.length
def current_total : ℤ := current_scores.sum
def current_average : ℤ := current_total / number_of_tests
def desired_average : ℤ := current_average + desired_increase 
def new_number_of_tests : ℤ := number_of_tests + 1
def total_required_score : ℤ := desired_average * new_number_of_tests

-- Lean 4 statement (theorem) to prove
theorem min_score_needed : total_required_score - current_total = 114 := by
  sorry

end NUMINAMATH_GPT_min_score_needed_l1426_142650


namespace NUMINAMATH_GPT_quadratic_func_condition_l1426_142694

noncomputable def f (x b c : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_func_condition (b c : ℝ) (h : f (-3) b c = f 1 b c) :
  f 1 b c > c ∧ c > f (-1) b c :=
by
  sorry

end NUMINAMATH_GPT_quadratic_func_condition_l1426_142694


namespace NUMINAMATH_GPT_solve_for_b_l1426_142685

theorem solve_for_b (b : ℚ) (h : b - b / 4 = 5 / 2) : b = 10 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_b_l1426_142685


namespace NUMINAMATH_GPT_positive_number_is_49_l1426_142665

theorem positive_number_is_49 (a : ℝ) (x : ℝ) (h₁ : (3 - a) * (3 - a) = x) (h₂ : (2 * a + 1) * (2 * a + 1) = x) :
  x = 49 :=
sorry

end NUMINAMATH_GPT_positive_number_is_49_l1426_142665


namespace NUMINAMATH_GPT_raisin_addition_l1426_142684

theorem raisin_addition : 
  let yellow_raisins := 0.3
  let black_raisins := 0.4
  yellow_raisins + black_raisins = 0.7 := 
by
  sorry

end NUMINAMATH_GPT_raisin_addition_l1426_142684


namespace NUMINAMATH_GPT_pqr_problem_l1426_142605

noncomputable def pqr_sums_to_44 (p q r : ℝ) : Prop :=
  (p < q) ∧ (∀ x, (x < -6 ∨ |x - 20| ≤ 2) ↔ ( (x - p) * (x - q) / (x - r) ≥ 0 ))

theorem pqr_problem (p q r : ℝ) (h : pqr_sums_to_44 p q r) : p + 2*q + 3*r = 44 :=
sorry

end NUMINAMATH_GPT_pqr_problem_l1426_142605


namespace NUMINAMATH_GPT_right_triangle_primes_l1426_142636

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop := ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- State the problem
theorem right_triangle_primes
  (a b : ℕ)
  (ha : is_prime a)
  (hb : is_prime b)
  (a_gt_b : a > b)
  (a_plus_b : a + b = 90)
  (a_minus_b_prime : is_prime (a - b)) :
  b = 17 :=
sorry

end NUMINAMATH_GPT_right_triangle_primes_l1426_142636


namespace NUMINAMATH_GPT_find_line_equation_l1426_142631
noncomputable def line_equation (l : ℝ → ℝ → Prop) : Prop :=
    (∀ x y : ℝ, l x y ↔ (2 * x + y - 4 = 0) ∨ (x + y - 3 = 0))

theorem find_line_equation (l : ℝ → ℝ → Prop) :
  (l 1 2) →
  (∃ x1 : ℝ, x1 > 0 ∧ ∃ y1 : ℝ, y1 > 0 ∧ l x1 0 ∧ l 0 y1) ∧
  (∃ x2 : ℝ, x2 < 0 ∧ ∃ y2 : ℝ, y2 > 0 ∧ l x2 0 ∧ l 0 y2) ∧
  (∃ x4 : ℝ, x4 > 0 ∧ ∃ y4 : ℝ, y4 < 0 ∧ l x4 0 ∧ l 0 y4) ∧
  (∃ x_int y_int : ℝ, l x_int 0 ∧ l 0 y_int ∧ x_int + y_int = 6) →
  (line_equation l) :=
by
  sorry

end NUMINAMATH_GPT_find_line_equation_l1426_142631


namespace NUMINAMATH_GPT_jellybeans_to_buy_l1426_142644

-- Define the conditions: a minimum of 150 jellybeans and a remainder of 15 when divided by 17.
def condition (n : ℕ) : Prop :=
  n ≥ 150 ∧ n % 17 = 15

-- Define the main statement to prove: if condition holds, then n is 151
theorem jellybeans_to_buy (n : ℕ) (h : condition n) : n = 151 :=
by
  -- Proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_jellybeans_to_buy_l1426_142644


namespace NUMINAMATH_GPT_sum_of_fractions_l1426_142601

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l1426_142601


namespace NUMINAMATH_GPT_floor_sufficient_but_not_necessary_l1426_142670

theorem floor_sufficient_but_not_necessary {x y : ℝ} : 
  (∀ x y : ℝ, (⌊x⌋₊ = ⌊y⌋₊) → abs (x - y) < 1) ∧ 
  ¬ (∀ x y : ℝ, abs (x - y) < 1 → (⌊x⌋₊ = ⌊y⌋₊)) :=
by
  sorry

end NUMINAMATH_GPT_floor_sufficient_but_not_necessary_l1426_142670


namespace NUMINAMATH_GPT_journey_distance_l1426_142682

theorem journey_distance :
  ∃ D : ℝ, (D / 42 + D / 48 = 10) ∧ D = 224 :=
by
  sorry

end NUMINAMATH_GPT_journey_distance_l1426_142682


namespace NUMINAMATH_GPT_cube_traversal_count_l1426_142686

-- Defining the cube traversal problem
def cube_traversal (num_faces : ℕ) (adj_faces : ℕ) (visits : ℕ) : ℕ :=
  if (num_faces = 6 ∧ adj_faces = 4) then
    4 * 2
  else
    0

-- Theorem statement
theorem cube_traversal_count : 
  cube_traversal 6 4 1 = 8 :=
by
  -- Skipping the proof with sorry for now
  sorry

end NUMINAMATH_GPT_cube_traversal_count_l1426_142686


namespace NUMINAMATH_GPT_total_brownies_l1426_142613

theorem total_brownies (brought_to_school left_at_home : ℕ) (h1 : brought_to_school = 16) (h2 : left_at_home = 24) : 
  brought_to_school + left_at_home = 40 := 
by 
  sorry

end NUMINAMATH_GPT_total_brownies_l1426_142613


namespace NUMINAMATH_GPT_billy_apples_ratio_l1426_142637

theorem billy_apples_ratio :
  let monday := 2
  let tuesday := 2 * monday
  let wednesday := 9
  let friday := monday / 2
  let total_apples := 20
  let thursday := total_apples - (monday + tuesday + wednesday + friday)
  thursday / friday = 4 := 
by
  let monday := 2
  let tuesday := 2 * monday
  let wednesday := 9
  let friday := monday / 2
  let total_apples := 20
  let thursday := total_apples - (monday + tuesday + wednesday + friday)
  sorry

end NUMINAMATH_GPT_billy_apples_ratio_l1426_142637


namespace NUMINAMATH_GPT_scramble_language_words_count_l1426_142699

theorem scramble_language_words_count :
  let total_words (n : ℕ) := 25 ^ n
  let words_without_B (n : ℕ) := 24 ^ n
  let words_with_B (n : ℕ) := total_words n - words_without_B n
  words_with_B 1 + words_with_B 2 + words_with_B 3 + words_with_B 4 + words_with_B 5 = 1863701 :=
by
  sorry

end NUMINAMATH_GPT_scramble_language_words_count_l1426_142699


namespace NUMINAMATH_GPT_length_difference_squares_l1426_142642

theorem length_difference_squares (A B : ℝ) (hA : A^2 = 25) (hB : B^2 = 81) : B - A = 4 :=
by
  sorry

end NUMINAMATH_GPT_length_difference_squares_l1426_142642


namespace NUMINAMATH_GPT_percentage_error_in_calculated_area_l1426_142657

theorem percentage_error_in_calculated_area
  (a : ℝ)
  (measured_side_length : ℝ := 1.025 * a) :
  (measured_side_length ^ 2 - a ^ 2) / (a ^ 2) * 100 = 5.0625 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_error_in_calculated_area_l1426_142657


namespace NUMINAMATH_GPT_valid_n_values_l1426_142663

variables (n x y : ℕ)

theorem valid_n_values :
  (n * (x - 3) = y + 3) ∧ (x + n = 3 * (y - n)) →
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7) :=
by
  sorry

end NUMINAMATH_GPT_valid_n_values_l1426_142663


namespace NUMINAMATH_GPT_determine_value_of_x_l1426_142616

theorem determine_value_of_x (x y z : ℤ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y) (hyz : y ≥ z)
  (h1 : x^2 - y^2 - z^2 + x * y = 4033) 
  (h2 : x^2 + 4 * y^2 + 4 * z^2 - 4 * x * y - 3 * x * z - 3 * y * z = -3995) : 
  x = 69 := sorry

end NUMINAMATH_GPT_determine_value_of_x_l1426_142616


namespace NUMINAMATH_GPT_find_vector_v_l1426_142646

def vector3 := ℝ × ℝ × ℝ

def cross_product (u v : vector3) : vector3 :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1  - u.1   * v.2.2,
   u.1   * v.2.1 - u.2.1 * v.1)

def a : vector3 := (1, 2, 1)
def b : vector3 := (2, 0, -1)
def v : vector3 := (3, 2, 0)
def b_cross_a : vector3 := (2, 3, 4)
def a_cross_b : vector3 := (-2, 3, -4)

theorem find_vector_v :
  cross_product v a = b_cross_a ∧ cross_product v b = a_cross_b :=
sorry

end NUMINAMATH_GPT_find_vector_v_l1426_142646


namespace NUMINAMATH_GPT_eval_expression_pow_i_l1426_142643

theorem eval_expression_pow_i :
  i^(12345 : ℤ) + i^(12346 : ℤ) + i^(12347 : ℤ) + i^(12348 : ℤ) = (0 : ℂ) :=
by
  -- Since this statement doesn't need the full proof, we use sorry to leave it open 
  sorry

end NUMINAMATH_GPT_eval_expression_pow_i_l1426_142643


namespace NUMINAMATH_GPT_amy_total_soups_l1426_142648

def chicken_soup := 6
def tomato_soup := 3
def vegetable_soup := 4
def clam_chowder := 2
def french_onion_soup := 1
def minestrone_soup := 5

theorem amy_total_soups : (chicken_soup + tomato_soup + vegetable_soup + clam_chowder + french_onion_soup + minestrone_soup) = 21 := by
  sorry

end NUMINAMATH_GPT_amy_total_soups_l1426_142648


namespace NUMINAMATH_GPT_equation_of_motion_l1426_142695

section MotionLaw

variable (t s : ℝ)
variable (v : ℝ → ℝ)
variable (C : ℝ)

-- Velocity function
def velocity (t : ℝ) : ℝ := 6 * t^2 + 1

-- Displacement function (indefinite integral of velocity)
def displacement (t : ℝ) (C : ℝ) : ℝ := 2 * t^3 + t + C

-- Given condition: displacement at t = 3 is 60
axiom displacement_at_3 : displacement 3 C = 60

-- Prove that the equation of motion is s = 2t^3 + t + 3
theorem equation_of_motion :
  ∃ C, displacement t C = 2 * t^3 + t + 3 :=
by
  use 3
  sorry

end MotionLaw

end NUMINAMATH_GPT_equation_of_motion_l1426_142695


namespace NUMINAMATH_GPT_gcd_204_85_l1426_142617

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_GPT_gcd_204_85_l1426_142617


namespace NUMINAMATH_GPT_function_domain_l1426_142619

noncomputable def domain_function : Set ℝ :=
  {x : ℝ | x ≠ 8}

theorem function_domain :
  ∀ x, x ∈ domain_function ↔ x ∈ (Set.Iio 8 ∪ Set.Ioi 8) := by
  intro x
  sorry

end NUMINAMATH_GPT_function_domain_l1426_142619


namespace NUMINAMATH_GPT_catherine_pencils_per_friend_l1426_142614

theorem catherine_pencils_per_friend :
  ∀ (pencils pens given_pens : ℕ), 
  pencils = pens ∧ pens = 60 ∧ given_pens = 8 ∧ 
  (∃ remaining_items : ℕ, remaining_items = 22 ∧ 
    ∀ friends : ℕ, friends = 7 → 
    remaining_items = (pens - (given_pens * friends)) + (pencils - (given_pens * friends * (pencils / pens)))) →
  ((pencils - (given_pens * friends * (pencils / pens))) / friends) = 6 :=
by 
  sorry

end NUMINAMATH_GPT_catherine_pencils_per_friend_l1426_142614


namespace NUMINAMATH_GPT_train_length_l1426_142664

theorem train_length (L V : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 175 = V * 39) : 
  L = 150 := 
by 
  -- proof omitted 
  sorry

end NUMINAMATH_GPT_train_length_l1426_142664


namespace NUMINAMATH_GPT_sphere_surface_area_l1426_142618

variable (x y z : ℝ)

theorem sphere_surface_area :
  (x^2 + y^2 + z^2 = 1) → (4 * Real.pi) = 4 * Real.pi :=
by
  intro h
  -- The proof will be inserted here
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l1426_142618


namespace NUMINAMATH_GPT_probability_two_heads_one_tail_in_three_tosses_l1426_142697

theorem probability_two_heads_one_tail_in_three_tosses
(P : ℕ → Prop) (pr : ℤ) : 
  (∀ n, P n → pr = 1 / 2) -> 
  P 3 → pr = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_heads_one_tail_in_three_tosses_l1426_142697


namespace NUMINAMATH_GPT_estimate_sqrt_diff_l1426_142698

-- Defining approximate values for square roots
def approx_sqrt_90 : ℝ := 9.5
def approx_sqrt_88 : ℝ := 9.4

-- Main statement
theorem estimate_sqrt_diff : |(approx_sqrt_90 - approx_sqrt_88) - 0.10| < 0.01 := by
  sorry

end NUMINAMATH_GPT_estimate_sqrt_diff_l1426_142698


namespace NUMINAMATH_GPT_complement_of_M_is_correct_l1426_142633

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

-- Define the complement of M in U
def complement_M_in_U : Set ℝ := {x : ℝ | x < -1 ∨ x > 3}

-- State the theorem
theorem complement_of_M_is_correct : (U \ M) = complement_M_in_U := by sorry

end NUMINAMATH_GPT_complement_of_M_is_correct_l1426_142633


namespace NUMINAMATH_GPT_max_passengers_l1426_142602

theorem max_passengers (total_stops : ℕ) (bus_capacity : ℕ)
  (h_total_stops : total_stops = 12) 
  (h_bus_capacity : bus_capacity = 20) 
  (h_no_same_stop : ∀ (a b : ℕ), a ≠ b → (a < total_stops) → (b < total_stops) → 
    ∃ x y : ℕ, x ≠ y ∧ x < total_stops ∧ y < total_stops ∧ 
    ((x = a ∧ y ≠ a) ∨ (x ≠ b ∧ y = b))) :
  ∃ max_passengers : ℕ, max_passengers = 50 :=
  sorry

end NUMINAMATH_GPT_max_passengers_l1426_142602


namespace NUMINAMATH_GPT_train_speed_l1426_142696

theorem train_speed (v : ℝ) (h1 : 60 * 6.5 + v * 6.5 = 910) : v = 80 := 
sorry

end NUMINAMATH_GPT_train_speed_l1426_142696


namespace NUMINAMATH_GPT_concert_attendance_difference_l1426_142678

theorem concert_attendance_difference :
  let first_concert := 65899
  let second_concert := 66018
  second_concert - first_concert = 119 :=
by
  sorry

end NUMINAMATH_GPT_concert_attendance_difference_l1426_142678


namespace NUMINAMATH_GPT_quadratic_solution_unique_l1426_142640

theorem quadratic_solution_unique (b : ℝ) (hb : b ≠ 0) (hdisc : 30 * 30 - 4 * b * 10 = 0) :
  ∃ x : ℝ, bx ^ 2 + 30 * x + 10 = 0 ∧ x = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_unique_l1426_142640


namespace NUMINAMATH_GPT_min_total_cost_l1426_142620

-- Defining the variables involved
variables (x y z : ℝ)
variables (h : ℝ := 1) (V : ℝ := 4)
def base_cost (x y : ℝ) : ℝ := 200 * (x * y)
def side_cost (x y : ℝ) (h : ℝ) : ℝ := 100 * (2 * (x + y)) * h
def total_cost (x y h : ℝ) : ℝ := base_cost x y + side_cost x y h

-- The condition that volume is 4 m^3
theorem min_total_cost : 
  (∀ x y, x * y = V) → 
  ∃ x y, total_cost x y h = 1600 :=
by
  sorry

end NUMINAMATH_GPT_min_total_cost_l1426_142620


namespace NUMINAMATH_GPT_Karen_packs_piece_of_cake_days_l1426_142654

theorem Karen_packs_piece_of_cake_days 
(Total Ham_Days : ℕ) (Ham_probability Cake_probability : ℝ) 
  (H_Total : Total = 5) 
  (H_Ham_Days : Ham_Days = 3) 
  (H_Ham_probability : Ham_probability = (3 / 5)) 
  (H_Cake_probability : Ham_probability * (Cake_probability / 5) = 0.12) : 
  Cake_probability = 1 := 
by
  sorry

end NUMINAMATH_GPT_Karen_packs_piece_of_cake_days_l1426_142654


namespace NUMINAMATH_GPT_jerry_books_vs_action_figures_l1426_142635

-- Define the initial conditions as constants
def initial_books : ℕ := 7
def initial_action_figures : ℕ := 3
def added_action_figures : ℕ := 2

-- Define the total number of action figures after adding
def total_action_figures : ℕ := initial_action_figures + added_action_figures

-- The theorem we need to prove
theorem jerry_books_vs_action_figures : initial_books - total_action_figures = 2 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_jerry_books_vs_action_figures_l1426_142635


namespace NUMINAMATH_GPT_f_of_f_of_f_of_3_l1426_142638

def f (x : ℕ) : ℕ := 
  if x > 9 then x - 1 
  else x ^ 3

theorem f_of_f_of_f_of_3 : f (f (f 3)) = 25 :=
by sorry

end NUMINAMATH_GPT_f_of_f_of_f_of_3_l1426_142638


namespace NUMINAMATH_GPT_initial_number_is_nine_l1426_142668

theorem initial_number_is_nine (x : ℝ) (h : 3 * (2 * x + 13) = 93) : x = 9 :=
sorry

end NUMINAMATH_GPT_initial_number_is_nine_l1426_142668


namespace NUMINAMATH_GPT_circle_equation_l1426_142677

theorem circle_equation : ∃ (x y : ℝ), (x - 2)^2 + y^2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l1426_142677


namespace NUMINAMATH_GPT_total_volume_correct_l1426_142674

-- Definitions based on the conditions
def box_length := 30 -- in cm
def box_width := 1 -- in cm
def box_height := 1 -- in cm
def horizontal_rows := 7
def vertical_rows := 5
def floors := 3

-- The volume of a single box
def box_volume : Int := box_length * box_width * box_height

-- The total number of boxes is the product of rows and floors
def total_boxes : Int := horizontal_rows * vertical_rows * floors

-- The total volume of all the boxes
def total_volume : Int := box_volume * total_boxes

-- The statement to prove
theorem total_volume_correct : total_volume = 3150 := 
by 
  simp [box_volume, total_boxes, total_volume]
  sorry

end NUMINAMATH_GPT_total_volume_correct_l1426_142674


namespace NUMINAMATH_GPT_integers_abs_le_3_l1426_142611

theorem integers_abs_le_3 :
  {x : ℤ | |x| ≤ 3} = { -3, -2, -1, 0, 1, 2, 3 } :=
by
  sorry

end NUMINAMATH_GPT_integers_abs_le_3_l1426_142611


namespace NUMINAMATH_GPT_pentagon_area_l1426_142662

theorem pentagon_area (a b c d e : ℤ) (O : 31 * 25 = 775) (H : 12^2 + 5^2 = 13^2) 
  (rect_side_lengths : (a, b, c, d, e) = (13, 19, 20, 25, 31)) :
  775 - 1/2 * 12 * 5 = 745 := 
by
  sorry

end NUMINAMATH_GPT_pentagon_area_l1426_142662


namespace NUMINAMATH_GPT_express_in_scientific_notation_l1426_142609

theorem express_in_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), 159600 = a * 10 ^ b ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.596 ∧ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l1426_142609


namespace NUMINAMATH_GPT_larger_value_3a_plus_1_l1426_142655

theorem larger_value_3a_plus_1 {a : ℝ} (h : 8 * a^2 + 6 * a + 2 = 0) : 3 * a + 1 ≤ 3 * (-1/4 : ℝ) + 1 := 
sorry

end NUMINAMATH_GPT_larger_value_3a_plus_1_l1426_142655


namespace NUMINAMATH_GPT_apples_remaining_l1426_142610

-- Define the initial conditions
def number_of_trees := 52
def apples_on_tree_before := 9
def apples_picked := 2

-- Define the target proof: the number of apples remaining on the tree
def apples_on_tree_after := apples_on_tree_before - apples_picked

-- The statement we aim to prove
theorem apples_remaining : apples_on_tree_after = 7 := sorry

end NUMINAMATH_GPT_apples_remaining_l1426_142610


namespace NUMINAMATH_GPT_perimeter_ABCDEFG_l1426_142606

variables {Point : Type}
variables {dist : Point → Point → ℝ}  -- Distance function

-- Definitions for midpoint and equilateral triangles
def is_midpoint (M A B : Point) : Prop := dist A M = dist M B ∧ dist A B = 2 * dist A M
def is_equilateral (A B C : Point) : Prop := dist A B = dist B C ∧ dist B C = dist C A

variables {A B C D E F G : Point}  -- Points in the plane
variables (h_eq_triangle_ABC : is_equilateral A B C)
variables (h_eq_triangle_ADE : is_equilateral A D E)
variables (h_eq_triangle_EFG : is_equilateral E F G)
variables (h_midpoint_D : is_midpoint D A C)
variables (h_midpoint_G : is_midpoint G A E)
variables (h_midpoint_F : is_midpoint F D E)
variables (h_AB_length : dist A B = 6)

theorem perimeter_ABCDEFG : 
  dist A B + dist B C + dist C D + dist D E + dist E F + dist F G + dist G A = 24 :=
sorry

end NUMINAMATH_GPT_perimeter_ABCDEFG_l1426_142606


namespace NUMINAMATH_GPT_shirts_needed_for_vacation_l1426_142607

def vacation_days := 7
def same_shirt_days := 2
def different_shirts_per_day := 2
def different_shirt_days := vacation_days - same_shirt_days

theorem shirts_needed_for_vacation : different_shirt_days * different_shirts_per_day + same_shirt_days = 11 := by
  sorry

end NUMINAMATH_GPT_shirts_needed_for_vacation_l1426_142607


namespace NUMINAMATH_GPT_triangle_angle_distance_l1426_142666

noncomputable def triangle_properties (ABC P Q R: Type) (angle : ABC → ABC → ABC → ℝ) (dist : ABC → ABC → ℝ) : Prop :=
  ∀ (A B C P Q R : ABC),
    angle B P C = 45 ∧
    angle Q A C = 45 ∧
    angle B C P = 30 ∧
    angle A C Q = 30 ∧
    angle A B R = 15 ∧
    angle B A R = 15 →
    angle P R Q = 90 ∧
    dist Q R = dist P R

theorem triangle_angle_distance (ABC P Q R: Type) (angle : ABC → ABC → ABC → ℝ) (dist : ABC → ABC → ℝ) :
  triangle_properties ABC P Q R angle dist →
  ∀ (A B C P Q R : ABC),
    angle B P C = 45 ∧
    angle Q A C = 45 ∧
    angle B C P = 30 ∧
    angle A C Q = 30 ∧
    angle A B R = 15 ∧
    angle B A R = 15 →
    angle P R Q = 90 ∧
    dist Q R = dist P R :=
by intros; sorry

end NUMINAMATH_GPT_triangle_angle_distance_l1426_142666


namespace NUMINAMATH_GPT_length_of_fourth_side_in_cyclic_quadrilateral_l1426_142681

theorem length_of_fourth_side_in_cyclic_quadrilateral :
  ∀ (r a b c : ℝ), r = 300 ∧ a = 300 ∧ b = 300 ∧ c = 150 * Real.sqrt 2 →
  ∃ d : ℝ, d = 450 :=
by
  sorry

end NUMINAMATH_GPT_length_of_fourth_side_in_cyclic_quadrilateral_l1426_142681


namespace NUMINAMATH_GPT_seventh_root_of_unity_sum_l1426_142604

theorem seventh_root_of_unity_sum (z : ℂ) (h1 : z^7 = 1) (h2 : z ≠ 1) :
  z + z^2 + z^4 = (-1 + Complex.I * Real.sqrt 11) / 2 ∨ z + z^2 + z^4 = (-1 - Complex.I * Real.sqrt 11) / 2 := 
by sorry

end NUMINAMATH_GPT_seventh_root_of_unity_sum_l1426_142604


namespace NUMINAMATH_GPT_tan_alpha_sub_beta_l1426_142625

theorem tan_alpha_sub_beta (α β : ℝ) (h₁ : Real.tan α = 9) (h₂ : Real.tan β = 6) : Real.tan (α - β) = 3 / 55 := 
sorry

end NUMINAMATH_GPT_tan_alpha_sub_beta_l1426_142625


namespace NUMINAMATH_GPT_necessarily_positive_l1426_142615

theorem necessarily_positive (a b c : ℝ) (ha : 0 < a ∧ a < 2) (hb : -2 < b ∧ b < 0) (hc : 0 < c ∧ c < 3) :
  (b + c) > 0 :=
sorry

end NUMINAMATH_GPT_necessarily_positive_l1426_142615


namespace NUMINAMATH_GPT_inequality_solution_set_l1426_142656

theorem inequality_solution_set (m : ℝ) : 
  (∀ (x : ℝ), m * x^2 - (1 - m) * x + m ≥ 0) ↔ m ≥ 1/3 := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1426_142656


namespace NUMINAMATH_GPT_distinct_real_roots_of_quadratic_l1426_142649

variable (m : ℝ)

theorem distinct_real_roots_of_quadratic (h1 : 4 + 4 * m > 0) (h2 : m ≠ 0) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_of_quadratic_l1426_142649


namespace NUMINAMATH_GPT_angle_same_terminal_side_l1426_142600

theorem angle_same_terminal_side (α θ : ℝ) (hα : α = 1690) (hθ : 0 < θ) (hθ2 : θ < 360) (h_terminal_side : ∃ k : ℤ, α = k * 360 + θ) : θ = 250 :=
by
  sorry

end NUMINAMATH_GPT_angle_same_terminal_side_l1426_142600


namespace NUMINAMATH_GPT_units_digit_of_30_factorial_is_0_l1426_142621

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_30_factorial_is_0 : units_digit (factorial 30) = 0 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_30_factorial_is_0_l1426_142621


namespace NUMINAMATH_GPT_relationship_between_y1_y2_l1426_142630

theorem relationship_between_y1_y2 (b y1 y2 : ℝ) 
  (h1 : y1 = -(-2) + b) 
  (h2 : y2 = -(3) + b) : 
  y1 > y2 := 
by {
  sorry
}

end NUMINAMATH_GPT_relationship_between_y1_y2_l1426_142630


namespace NUMINAMATH_GPT_square_of_real_is_positive_or_zero_l1426_142645

def p (x : ℝ) : Prop := x^2 > 0
def q (x : ℝ) : Prop := x^2 = 0

theorem square_of_real_is_positive_or_zero (x : ℝ) : (p x ∨ q x) :=
by
  sorry

end NUMINAMATH_GPT_square_of_real_is_positive_or_zero_l1426_142645


namespace NUMINAMATH_GPT_john_trip_time_l1426_142673

theorem john_trip_time (normal_distance : ℕ) (normal_time : ℕ) (extra_distance : ℕ) 
  (double_extra_distance : ℕ) (same_speed : ℕ) 
  (h1: normal_distance = 150) 
  (h2: normal_time = 3) 
  (h3: extra_distance = 50)
  (h4: double_extra_distance = 2 * extra_distance)
  (h5: same_speed = normal_distance / normal_time) : 
  normal_time + double_extra_distance / same_speed = 5 :=
by 
  sorry

end NUMINAMATH_GPT_john_trip_time_l1426_142673


namespace NUMINAMATH_GPT_sum_series_eq_l1426_142612

open BigOperators

theorem sum_series_eq : 
  ∑ n in Finset.range 256, (1 : ℝ) / ((2 * (n + 1 : ℕ) - 3) * (2 * (n + 1 : ℕ) + 1)) = -257 / 513 := 
by 
  sorry

end NUMINAMATH_GPT_sum_series_eq_l1426_142612


namespace NUMINAMATH_GPT_linda_savings_fraction_l1426_142632

theorem linda_savings_fraction (savings tv_cost : ℝ) (h1 : savings = 960) (h2 : tv_cost = 240) : (savings - tv_cost) / savings = 3 / 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_linda_savings_fraction_l1426_142632


namespace NUMINAMATH_GPT_total_number_of_coins_l1426_142671

theorem total_number_of_coins (x : ℕ) (h : 1 * x + 5 * x + 10 * x + 50 * x + 100 * x = 332) : 5 * x = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_number_of_coins_l1426_142671


namespace NUMINAMATH_GPT_total_number_of_squares_is_13_l1426_142672

-- Define the vertices of the region
def region_condition (x y : ℕ) : Prop :=
  y ≤ x ∧ y ≤ 4 ∧ x ≤ 4

-- Define the type of squares whose vertices have integer coordinates
def square (n : ℕ) (x y : ℕ) : Prop :=
  region_condition x y ∧ region_condition (x - n) y ∧ 
  region_condition x (y - n) ∧ region_condition (x - n) (y - n)

-- Count the number of squares of each size within the region
def number_of_squares (size : ℕ) : ℕ :=
  match size with
  | 1 => 10 -- number of 1x1 squares
  | 2 => 3  -- number of 2x2 squares
  | _ => 0  -- there are no larger squares in this context

-- Prove the total number of squares is 13
theorem total_number_of_squares_is_13 : number_of_squares 1 + number_of_squares 2 = 13 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_squares_is_13_l1426_142672


namespace NUMINAMATH_GPT_trigonometric_identity_l1426_142641

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 4) :
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + (Real.sin θ ^ 2) / 4 = 21 / 68 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1426_142641


namespace NUMINAMATH_GPT_silk_dyed_amount_l1426_142634

-- Define the conditions
def yards_green : ℕ := 61921
def yards_pink : ℕ := 49500

-- Define the total calculation
def total_yards : ℕ := yards_green + yards_pink

-- State what needs to be proven: that the total yards is 111421
theorem silk_dyed_amount : total_yards = 111421 := by
  sorry

end NUMINAMATH_GPT_silk_dyed_amount_l1426_142634


namespace NUMINAMATH_GPT_television_final_price_l1426_142608

theorem television_final_price :
  let original_price := 1200
  let discount_percent := 0.30
  let tax_percent := 0.08
  let rebate := 50
  let discount := discount_percent * original_price
  let sale_price := original_price - discount
  let tax := tax_percent * sale_price
  let price_including_tax := sale_price + tax
  let final_amount := price_including_tax - rebate
  final_amount = 857.2 :=
by
{
  -- The proof would go here, but it's omitted as per instructions.
  sorry
}

end NUMINAMATH_GPT_television_final_price_l1426_142608


namespace NUMINAMATH_GPT_chess_tournament_l1426_142653

def number_of_players := 30

def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament : total_games number_of_players = 435 := by
  sorry

end NUMINAMATH_GPT_chess_tournament_l1426_142653


namespace NUMINAMATH_GPT_meaningful_fraction_l1426_142629

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_fraction_l1426_142629


namespace NUMINAMATH_GPT_gcd_f100_f101_l1426_142691

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - x + 2010

-- A statement asserting the greatest common divisor of f(100) and f(101) is 10
theorem gcd_f100_f101 : Int.gcd (f 100) (f 101) = 10 := by
  sorry

end NUMINAMATH_GPT_gcd_f100_f101_l1426_142691


namespace NUMINAMATH_GPT_total_books_l1426_142679

variable (M K G : ℕ)

-- Conditions
def Megan_books := 32
def Kelcie_books := Megan_books / 4
def Greg_books := 2 * Kelcie_books + 9

-- Theorem to prove
theorem total_books : Megan_books + Kelcie_books + Greg_books = 65 := by
  unfold Megan_books Kelcie_books Greg_books
  sorry

end NUMINAMATH_GPT_total_books_l1426_142679


namespace NUMINAMATH_GPT_sum_and_product_l1426_142659

theorem sum_and_product (c d : ℝ) (h1 : 2 * c = -8) (h2 : c^2 - d = 4) : c + d = 8 := by
  sorry

end NUMINAMATH_GPT_sum_and_product_l1426_142659


namespace NUMINAMATH_GPT_prime_number_conditions_l1426_142639

theorem prime_number_conditions :
  ∃ p n : ℕ, Prime p ∧ p = n^2 + 9 ∧ p = (n+1)^2 - 8 :=
by
  sorry

end NUMINAMATH_GPT_prime_number_conditions_l1426_142639


namespace NUMINAMATH_GPT_intersection_M_N_l1426_142683

-- Definitions of sets M and N
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- The statement to prove
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1426_142683


namespace NUMINAMATH_GPT_find_number_l1426_142622

theorem find_number (x : ℝ) (h : (((x + 45) / 2) / 2) + 45 = 85) : x = 115 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1426_142622


namespace NUMINAMATH_GPT_james_net_profit_l1426_142680

def totalCandyBarsSold (boxes : Nat) (candyBarsPerBox : Nat) : Nat :=
  boxes * candyBarsPerBox

def revenue30CandyBars (pricePerCandyBar : Real) : Real :=
  30 * pricePerCandyBar

def revenue20CandyBars (pricePerCandyBar : Real) : Real :=
  20 * pricePerCandyBar

def totalRevenue (revenue1 : Real) (revenue2 : Real) : Real :=
  revenue1 + revenue2

def costNonDiscountedBoxes (candyBars : Nat) (pricePerCandyBar : Real) : Real :=
  candyBars * pricePerCandyBar

def costDiscountedBoxes (candyBars : Nat) (pricePerCandyBar : Real) : Real :=
  candyBars * pricePerCandyBar

def totalCost (cost1 : Real) (cost2 : Real) : Real :=
  cost1 + cost2

def salesTax (totalRevenue : Real) (taxRate : Real) : Real :=
  totalRevenue * taxRate

def totalExpenses (cost : Real) (salesTax : Real) (fixedExpense : Real) : Real :=
  cost + salesTax + fixedExpense

def netProfit (totalRevenue : Real) (totalExpenses : Real) : Real :=
  totalRevenue - totalExpenses

theorem james_net_profit :
  let boxes := 5
  let candyBarsPerBox := 10
  let totalCandyBars := totalCandyBarsSold boxes candyBarsPerBox

  let priceFirst30 := 1.50
  let priceNext20 := 1.30
  let priceSubsequent := 1.10

  let revenueFirst30 := revenue30CandyBars priceFirst30
  let revenueNext20 := revenue20CandyBars priceNext20
  let totalRevenue := totalRevenue revenueFirst30 revenueNext20

  let priceNonDiscounted := 1.00
  let candyBarsNonDiscounted := 20
  let costNonDiscounted := costNonDiscountedBoxes candyBarsNonDiscounted priceNonDiscounted

  let priceDiscounted := 0.80
  let candyBarsDiscounted := 30
  let costDiscounted := costDiscountedBoxes candyBarsDiscounted priceDiscounted

  let totalCost := totalCost costNonDiscounted costDiscounted

  let taxRate := 0.07
  let salesTax := salesTax totalRevenue taxRate

  let fixedExpense := 15.0
  let totalExpenses := totalExpenses totalCost salesTax fixedExpense

  netProfit totalRevenue totalExpenses = 7.03 :=
by
  sorry

end NUMINAMATH_GPT_james_net_profit_l1426_142680


namespace NUMINAMATH_GPT_inequality_solution_l1426_142675

theorem inequality_solution (x : ℝ) (h1 : 2 * x + 1 > x + 3) (h2 : 2 * x - 4 < x) : 2 < x ∧ x < 4 := sorry

end NUMINAMATH_GPT_inequality_solution_l1426_142675


namespace NUMINAMATH_GPT_fraction_to_percentage_l1426_142689

theorem fraction_to_percentage (y : ℝ) (h : y > 0) : ((7 * y) / 20 + (3 * y) / 10) = 0.65 * y :=
by
  -- the proof steps will go here
  sorry

end NUMINAMATH_GPT_fraction_to_percentage_l1426_142689


namespace NUMINAMATH_GPT_find_principal_l1426_142626

theorem find_principal 
  (SI : ℝ) 
  (R : ℝ) 
  (T : ℝ) 
  (h_SI : SI = 4052.25) 
  (h_R : R = 9) 
  (h_T : T = 5) : 
  (SI * 100) / (R * T) = 9005 := 
by 
  rw [h_SI, h_R, h_T]
  sorry

end NUMINAMATH_GPT_find_principal_l1426_142626


namespace NUMINAMATH_GPT_eval_g_at_8_l1426_142688

def g (x : ℚ) : ℚ := (3 * x + 2) / (x - 2)

theorem eval_g_at_8 : g 8 = 13 / 3 := by
  sorry

end NUMINAMATH_GPT_eval_g_at_8_l1426_142688


namespace NUMINAMATH_GPT_surface_dots_sum_l1426_142603

-- Define the sum of dots on opposite faces of a standard die
axiom sum_opposite_faces (x y : ℕ) : x + y = 7

-- Define the large cube dimensions
def large_cube_dimension : ℕ := 3

-- Define the total number of small cubes
def num_small_cubes : ℕ := large_cube_dimension ^ 3

-- Calculate the number of faces on the surface of the large cube
def num_surface_faces : ℕ := 6 * large_cube_dimension ^ 2

-- Given the sum of opposite faces, compute the total number of dots on the surface
theorem surface_dots_sum : num_surface_faces / 2 * 7 = 189 := by
  sorry

end NUMINAMATH_GPT_surface_dots_sum_l1426_142603


namespace NUMINAMATH_GPT_dart_prob_center_square_l1426_142693

noncomputable def hexagon_prob (s : ℝ) : ℝ :=
  let square_area := s^2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  square_area / hexagon_area

theorem dart_prob_center_square (s : ℝ) : hexagon_prob s = 2 * Real.sqrt 3 / 9 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_dart_prob_center_square_l1426_142693


namespace NUMINAMATH_GPT_isosceles_triangle_base_vertex_trajectory_l1426_142627

theorem isosceles_triangle_base_vertex_trajectory :
  ∀ (x y : ℝ), 
  (∀ (A : ℝ × ℝ) (B : ℝ × ℝ), 
    A = (2, 4) ∧ B = (2, 8) ∧ 
    ((x-2)^2 + (y-4)^2 = 16)) → 
  ((x ≠ 2) ∧ (y ≠ 8) → (x-2)^2 + (y-4)^2 = 16) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_vertex_trajectory_l1426_142627


namespace NUMINAMATH_GPT_percentage_books_not_sold_is_60_percent_l1426_142647

def initial_stock : ℕ := 700
def sold_monday : ℕ := 50
def sold_tuesday : ℕ := 82
def sold_wednesday : ℕ := 60
def sold_thursday : ℕ := 48
def sold_friday : ℕ := 40

def total_sold : ℕ := sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday
def books_not_sold : ℕ := initial_stock - total_sold
def percentage_not_sold : ℚ := (books_not_sold * 100) / initial_stock

theorem percentage_books_not_sold_is_60_percent : percentage_not_sold = 60 := by
  sorry

end NUMINAMATH_GPT_percentage_books_not_sold_is_60_percent_l1426_142647


namespace NUMINAMATH_GPT_batch_preparation_l1426_142661

theorem batch_preparation (total_students cupcakes_per_student cupcakes_per_batch percent_not_attending : ℕ)
    (hlt1 : total_students = 150)
    (hlt2 : cupcakes_per_student = 3)
    (hlt3 : cupcakes_per_batch = 20)
    (hlt4 : percent_not_attending = 20)
    : (total_students * (80 / 100) * cupcakes_per_student) / cupcakes_per_batch = 18 := by
  sorry

end NUMINAMATH_GPT_batch_preparation_l1426_142661


namespace NUMINAMATH_GPT_function_identity_l1426_142628

variable (f : ℕ+ → ℕ+)

theorem function_identity (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : ∀ n : ℕ+, f n = n := sorry

end NUMINAMATH_GPT_function_identity_l1426_142628


namespace NUMINAMATH_GPT_trig_intersection_identity_l1426_142652

theorem trig_intersection_identity (x0 : ℝ) (hx0 : x0 ≠ 0) (htan : -x0 = Real.tan x0) :
  (x0^2 + 1) * (1 + Real.cos (2 * x0)) = 2 := 
sorry

end NUMINAMATH_GPT_trig_intersection_identity_l1426_142652


namespace NUMINAMATH_GPT_restore_original_salary_l1426_142687

theorem restore_original_salary (orig_salary : ℝ) (reducing_percent : ℝ) (increasing_percent : ℝ) :
  reducing_percent = 20 → increasing_percent = 25 →
  (orig_salary * (1 - reducing_percent / 100)) * (1 + increasing_percent / 100 / (1 - reducing_percent / 100)) = orig_salary
:= by
  intros
  sorry

end NUMINAMATH_GPT_restore_original_salary_l1426_142687
