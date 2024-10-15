import Mathlib

namespace NUMINAMATH_GPT_union_A_B_l934_93425

noncomputable def U := Set.univ ℝ

def A : Set ℝ := {x | x^2 - x - 2 = 0}

def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = x + 3}

theorem union_A_B : A ∪ B = { -1, 2, 5 } :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_l934_93425


namespace NUMINAMATH_GPT_remainder_when_sum_divided_by_15_l934_93412

theorem remainder_when_sum_divided_by_15 (a b c : ℕ) 
  (h1 : a % 15 = 11) 
  (h2 : b % 15 = 12) 
  (h3 : c % 15 = 13) : 
  (a + b + c) % 15 = 6 :=
  sorry

end NUMINAMATH_GPT_remainder_when_sum_divided_by_15_l934_93412


namespace NUMINAMATH_GPT_determine_a_l934_93424

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem determine_a :
  {a : ℝ | 0 < a ∧ (f (a + 1) ≤ f (2 * a^2))} = {a : ℝ | 1 ≤ a ∧ a ≤ Real.sqrt 6 / 2 } :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l934_93424


namespace NUMINAMATH_GPT_divides_n3_minus_7n_l934_93448

theorem divides_n3_minus_7n (n : ℕ) : 6 ∣ n^3 - 7 * n := 
sorry

end NUMINAMATH_GPT_divides_n3_minus_7n_l934_93448


namespace NUMINAMATH_GPT_min_fraction_value_l934_93473

theorem min_fraction_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_tangent : 2 * a + b = 2) :
  (8 * a + b) / (a * b) ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_min_fraction_value_l934_93473


namespace NUMINAMATH_GPT_prob_a_wins_match_l934_93447

-- Define the probability of A winning a single game
def prob_win_a_single_game : ℚ := 1 / 3

-- Define the probability of A winning two consecutive games
def prob_win_a_two_consec_games : ℚ := prob_win_a_single_game * prob_win_a_single_game

-- Define the probability of A winning two games with one loss in between
def prob_win_a_two_wins_one_loss_first : ℚ := prob_win_a_single_game * (1 - prob_win_a_single_game) * prob_win_a_single_game
def prob_win_a_two_wins_one_loss_second : ℚ := (1 - prob_win_a_single_game) * prob_win_a_single_game * prob_win_a_single_game

-- Define the total probability of A winning the match
def prob_a_winning_match : ℚ := prob_win_a_two_consec_games + prob_win_a_two_wins_one_loss_first + prob_win_a_two_wins_one_loss_second

-- The theorem to be proved
theorem prob_a_wins_match : prob_a_winning_match = 7 / 27 :=
by sorry

end NUMINAMATH_GPT_prob_a_wins_match_l934_93447


namespace NUMINAMATH_GPT_exists_indices_l934_93420

-- Define the sequence condition
def is_sequence_of_all_positive_integers (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, ∃ m : ℕ, a m = n) ∧ (∀ n m1 m2 : ℕ, a m1 = n ∧ a m2 = n → m1 = m2)

-- Main theorem statement
theorem exists_indices 
  (a : ℕ → ℕ) 
  (h : is_sequence_of_all_positive_integers a) :
  ∃ (ℓ m : ℕ), 1 < ℓ ∧ ℓ < m ∧ (a 0 + a m = 2 * a ℓ) :=
by
  sorry

end NUMINAMATH_GPT_exists_indices_l934_93420


namespace NUMINAMATH_GPT_compound_interest_is_correct_l934_93428

noncomputable def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

theorem compound_interest_is_correct :
  let P := 660 / (0.2 : ℝ)
  (compound_interest P 10 2) = 693 := 
by
  -- Definitions of simple_interest and compound_interest are used
  -- The problem conditions help us conclude
  let P := 660 / (0.2 : ℝ)
  have h1 : simple_interest P 10 2 = 660 := by sorry
  have h2 : compound_interest P 10 2 = 693 := by sorry
  exact h2

end NUMINAMATH_GPT_compound_interest_is_correct_l934_93428


namespace NUMINAMATH_GPT_weight_of_triangular_piece_l934_93480

noncomputable def density_factor (weight : ℝ) (area : ℝ) : ℝ :=
  weight / area

noncomputable def square_weight (side_length : ℝ) (weight : ℝ) : ℝ := weight

noncomputable def triangle_area (side_length : ℝ) : ℝ :=
  (side_length ^ 2 * Real.sqrt 3) / 4

theorem weight_of_triangular_piece :
  let side_square := 4
  let weight_square := 16
  let side_triangle := 6
  let area_square := side_square ^ 2
  let area_triangle := triangle_area side_triangle
  let density_square := density_factor weight_square area_square
  let weight_triangle := area_triangle * density_square
  abs weight_triangle - 15.59 < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_triangular_piece_l934_93480


namespace NUMINAMATH_GPT_find_x_l934_93407

namespace IntegerProblem

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := 
by
  sorry

end IntegerProblem

end NUMINAMATH_GPT_find_x_l934_93407


namespace NUMINAMATH_GPT_num_rectangles_in_5x5_grid_l934_93439

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end NUMINAMATH_GPT_num_rectangles_in_5x5_grid_l934_93439


namespace NUMINAMATH_GPT_complex_pure_imaginary_l934_93400

theorem complex_pure_imaginary (a : ℝ) 
  (h1 : a^2 + 2*a - 3 = 0) 
  (h2 : a + 3 ≠ 0) : 
  a = 1 := 
by
  sorry

end NUMINAMATH_GPT_complex_pure_imaginary_l934_93400


namespace NUMINAMATH_GPT_lowest_temperature_at_noon_l934_93492

theorem lowest_temperature_at_noon
  (L : ℤ) -- Denote lowest temperature as L
  (avg_temp : ℤ) -- Average temperature from Monday to Friday
  (max_range : ℤ) -- Maximum possible range of the temperature
  (h1 : avg_temp = 50) -- Condition 1: average temperature is 50
  (h2 : max_range = 50) -- Condition 2: maximum range is 50
  (total_temp : ℤ) -- Sum of temperatures from Monday to Friday
  (h3 : total_temp = 250) -- Sum of temperatures equals 5 * 50
  (h4 : total_temp = L + (L + 50) + (L + 50) + (L + 50) + (L + 50)) -- Sum represented in terms of L
  : L = 10 := -- Prove that L equals 10
sorry

end NUMINAMATH_GPT_lowest_temperature_at_noon_l934_93492


namespace NUMINAMATH_GPT_inequality_solution_range_l934_93462

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x ^ 2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_range_l934_93462


namespace NUMINAMATH_GPT_fraction_of_seats_sold_l934_93488

theorem fraction_of_seats_sold
  (ticket_price : ℕ) (number_of_rows : ℕ) (seats_per_row : ℕ) (total_earnings : ℕ)
  (h1 : ticket_price = 10)
  (h2 : number_of_rows = 20)
  (h3 : seats_per_row = 10)
  (h4 : total_earnings = 1500) :
  (total_earnings / ticket_price : ℕ) / (number_of_rows * seats_per_row : ℕ) = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_fraction_of_seats_sold_l934_93488


namespace NUMINAMATH_GPT_evaluate_expression_l934_93469

theorem evaluate_expression : (7^(1/4) / 7^(1/7)) = 7^(3/28) := 
by sorry

end NUMINAMATH_GPT_evaluate_expression_l934_93469


namespace NUMINAMATH_GPT_points_earned_l934_93417

-- Definitions from conditions
def points_per_enemy : ℕ := 8
def total_enemies : ℕ := 7
def enemies_not_destroyed : ℕ := 2

-- The proof statement
theorem points_earned :
  points_per_enemy * (total_enemies - enemies_not_destroyed) = 40 := 
by
  sorry

end NUMINAMATH_GPT_points_earned_l934_93417


namespace NUMINAMATH_GPT_rectangle_vertex_x_coordinate_l934_93455

theorem rectangle_vertex_x_coordinate
  (x : ℝ)
  (y1 y2 : ℝ)
  (slope : ℝ)
  (h1 : x = 1)
  (h2 : 9 = 9)
  (h3 : slope = 0.2)
  (h4 : y1 = 0)
  (h5 : y2 = 2)
  (h6 : ∀ (x : ℝ), (0.2 * x : ℝ) = 1 → x = 1) :
  x = 1 := 
by sorry

end NUMINAMATH_GPT_rectangle_vertex_x_coordinate_l934_93455


namespace NUMINAMATH_GPT_sector_properties_l934_93431

noncomputable def central_angle (l r : ℝ) : ℝ := l / r

noncomputable def sector_area (alpha r : ℝ) : ℝ := (1/2) * alpha * r^2

theorem sector_properties (l r : ℝ) (h_l : l = Real.pi) (h_r : r = 3) :
  central_angle l r = Real.pi / 3 ∧ sector_area (central_angle l r) r = 3 * Real.pi / 2 := 
  by
  sorry

end NUMINAMATH_GPT_sector_properties_l934_93431


namespace NUMINAMATH_GPT_diminished_gcd_equals_100_l934_93438

theorem diminished_gcd_equals_100 : Nat.gcd 7800 360 - 20 = 100 := by
  sorry

end NUMINAMATH_GPT_diminished_gcd_equals_100_l934_93438


namespace NUMINAMATH_GPT_Durakavalyanie_last_lesson_class_1C_l934_93486

theorem Durakavalyanie_last_lesson_class_1C :
  ∃ (class_lesson : String × Nat → String), 
  class_lesson ("1B", 1) = "Kurashenie" ∧
  (∃ (k m n : Nat), class_lesson ("1A", k) = "Durakavalyanie" ∧ class_lesson ("1B", m) = "Durakavalyanie" ∧ m > k) ∧
  class_lesson ("1A", 2) ≠ "Nizvedenie" ∧
  class_lesson ("1C", 3) = "Durakavalyanie" :=
sorry

end NUMINAMATH_GPT_Durakavalyanie_last_lesson_class_1C_l934_93486


namespace NUMINAMATH_GPT_quadratic_real_roots_l934_93458

theorem quadratic_real_roots (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 - 1 + m = 0 ∧ x2^2 + 2 * x2 - 1 + m = 0) ↔ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l934_93458


namespace NUMINAMATH_GPT_sum_of_squares_and_product_l934_93401

theorem sum_of_squares_and_product
  (x y : ℕ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_and_product_l934_93401


namespace NUMINAMATH_GPT_train_A_distance_travelled_l934_93489

/-- Let Train A and Train B start from opposite ends of a 200-mile route at the same time.
Train A has a constant speed of 20 miles per hour, and Train B has a constant speed of 200 miles / 6 hours (which is approximately 33.33 miles per hour).
Prove that Train A had traveled 75 miles when it met Train B. --/
theorem train_A_distance_travelled:
  ∀ (T : Type) (start_time : T) (distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (meeting_time : ℝ),
  distance = 200 ∧ speed_A = 20 ∧ speed_B = 33.33 ∧ meeting_time = 200 / (speed_A + speed_B) → 
  (speed_A * meeting_time = 75) :=
by
  sorry

end NUMINAMATH_GPT_train_A_distance_travelled_l934_93489


namespace NUMINAMATH_GPT_units_digit_of_large_powers_l934_93404

theorem units_digit_of_large_powers : 
  (2^1007 * 6^1008 * 14^1009) % 10 = 2 := 
  sorry

end NUMINAMATH_GPT_units_digit_of_large_powers_l934_93404


namespace NUMINAMATH_GPT_distinct_solutions_difference_l934_93495

theorem distinct_solutions_difference (r s : ℝ) (hr : (r - 5) * (r + 5) = 25 * r - 125)
  (hs : (s - 5) * (s + 5) = 25 * s - 125) (neq : r ≠ s) (hgt : r > s) : r - s = 15 := by
  sorry

end NUMINAMATH_GPT_distinct_solutions_difference_l934_93495


namespace NUMINAMATH_GPT_find_slope_l934_93484

theorem find_slope (k b x y y2 : ℝ) (h1 : y = k * x + b) (h2 : y2 = k * (x + 3) + b) (h3 : y2 - y = -2) : k = -2 / 3 := by
  sorry

end NUMINAMATH_GPT_find_slope_l934_93484


namespace NUMINAMATH_GPT_age_difference_is_12_l934_93493

noncomputable def age_difference (x : ℕ) : ℕ :=
  let older := 3 * x
  let younger := 2 * x
  older - younger

theorem age_difference_is_12 :
  ∃ x : ℕ, 3 * x + 2 * x = 60 ∧ age_difference x = 12 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_is_12_l934_93493


namespace NUMINAMATH_GPT_total_letters_l934_93409

theorem total_letters (brother_letters : ℕ) (greta_more_than_brother : ℕ) (mother_multiple : ℕ)
  (h_brother : brother_letters = 40)
  (h_greta : ∀ (brother_letters greta_letters : ℕ), greta_letters = brother_letters + greta_more_than_brother)
  (h_mother : ∀ (total_letters mother_letters : ℕ), mother_letters = mother_multiple * total_letters) :
  brother_letters + (brother_letters + greta_more_than_brother) + (mother_multiple * (brother_letters + (brother_letters + greta_more_than_brother))) = 270 :=
by
  sorry

end NUMINAMATH_GPT_total_letters_l934_93409


namespace NUMINAMATH_GPT_solve_for_y_l934_93416

theorem solve_for_y (y : ℤ) : (4 + y) / (6 + y) = (2 + y) / (3 + y) → y = 0 := by 
  sorry

end NUMINAMATH_GPT_solve_for_y_l934_93416


namespace NUMINAMATH_GPT_remaining_amount_is_1520_l934_93441

noncomputable def totalAmountToBePaid (deposit : ℝ) (depositRate : ℝ) (taxRate : ℝ) (processingFee : ℝ) : ℝ :=
  let fullPrice := deposit / depositRate
  let salesTax := taxRate * fullPrice
  let totalAdditionalExpenses := salesTax + processingFee
  (fullPrice - deposit) + totalAdditionalExpenses

theorem remaining_amount_is_1520 :
  totalAmountToBePaid 140 0.10 0.15 50 = 1520 := by
  sorry

end NUMINAMATH_GPT_remaining_amount_is_1520_l934_93441


namespace NUMINAMATH_GPT_general_formula_minimum_n_exists_l934_93436

noncomputable def a_n (n : ℕ) : ℝ := 3 * (-2)^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := 1 - (-2)^n

theorem general_formula (n : ℕ) : a_n n = 3 * (-2)^(n-1) :=
by sorry

theorem minimum_n_exists :
  (∃ n : ℕ, S_n n > 2016) ∧ (∀ m : ℕ, S_n m > 2016 → 11 ≤ m) :=
by sorry

end NUMINAMATH_GPT_general_formula_minimum_n_exists_l934_93436


namespace NUMINAMATH_GPT_dolphins_points_l934_93403

variable (S D : ℕ)

theorem dolphins_points :
  (S + D = 36) ∧ (S = D + 12) → D = 12 :=
by
  sorry

end NUMINAMATH_GPT_dolphins_points_l934_93403


namespace NUMINAMATH_GPT_honor_students_count_l934_93440

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end NUMINAMATH_GPT_honor_students_count_l934_93440


namespace NUMINAMATH_GPT_geometric_progression_solution_l934_93461

theorem geometric_progression_solution 
  (b₁ q : ℝ)
  (h₁ : b₁^3 * q^3 = 1728)
  (h₂ : b₁ * (1 + q + q^2) = 63) :
  (b₁ = 3 ∧ q = 4) ∨ (b₁ = 48 ∧ q = 1/4) :=
  sorry

end NUMINAMATH_GPT_geometric_progression_solution_l934_93461


namespace NUMINAMATH_GPT_cd_value_l934_93449

theorem cd_value (a b c d : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (ab ac bd : ℝ) 
  (h_ab : ab = 2) (h_ac : ac = 5) (h_bd : bd = 6) :
  ∃ (cd : ℝ), cd = 3 :=
by sorry

end NUMINAMATH_GPT_cd_value_l934_93449


namespace NUMINAMATH_GPT_find_a_value_l934_93413

noncomputable def a : ℝ := (384:ℝ)^(1/7)

variables (a b c : ℝ)
variables (h1 : a^2 / b = 2) (h2 : b^2 / c = 4) (h3 : c^2 / a = 6)

theorem find_a_value : a = 384^(1/7) :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l934_93413


namespace NUMINAMATH_GPT_solution_set_of_inequality_l934_93463

theorem solution_set_of_inequality (x: ℝ) : 
  (1 / x ≤ 1) ↔ (x < 0 ∨ x ≥ 1) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l934_93463


namespace NUMINAMATH_GPT_triangle_area_l934_93444

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (3, -4)

-- State that the area of the triangle is 12.5 square units
theorem triangle_area :
  let base := 6 - 1
  let height := 1 - -4
  (1 / 2) * base * height = 12.5 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l934_93444


namespace NUMINAMATH_GPT_probability_adjacent_vertices_of_octagon_l934_93481

theorem probability_adjacent_vertices_of_octagon :
  let num_vertices := 8;
  let adjacent_vertices (v1 v2 : Fin num_vertices) : Prop := 
    (v2 = (v1 + 1) % num_vertices) ∨ (v2 = (v1 - 1 + num_vertices) % num_vertices);
  let total_vertices := num_vertices - 1;
  (2 : ℚ) / total_vertices = (2 / 7 : ℚ) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_adjacent_vertices_of_octagon_l934_93481


namespace NUMINAMATH_GPT_function_increasing_intervals_l934_93451

theorem function_increasing_intervals (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x < f (x + 1)) :
  (∃ x : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, abs (y - x) < δ → f y > f x) ∨ 
  (∀ x : ℝ, ∃ ε > 0, ∀ δ > 0, ∃ y : ℝ, abs (y - x) < δ ∧ f y < f x) :=
sorry

end NUMINAMATH_GPT_function_increasing_intervals_l934_93451


namespace NUMINAMATH_GPT_maximum_fly_path_length_in_box_l934_93411

theorem maximum_fly_path_length_in_box
  (length width height : ℝ)
  (h_length : length = 1)
  (h_width : width = 1)
  (h_height : height = 2) :
  ∃ l, l = (Real.sqrt 6 + 2 * Real.sqrt 5 + Real.sqrt 2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_maximum_fly_path_length_in_box_l934_93411


namespace NUMINAMATH_GPT_art_collection_total_area_l934_93427

-- Define the dimensions and quantities of the paintings
def square_painting_side := 6
def small_painting_width := 2
def small_painting_height := 3
def large_painting_width := 10
def large_painting_height := 15

def num_square_paintings := 3
def num_small_paintings := 4
def num_large_paintings := 1

-- Define areas of individual paintings
def square_painting_area := square_painting_side * square_painting_side
def small_painting_area := small_painting_width * small_painting_height
def large_painting_area := large_painting_width * large_painting_height

-- Define the total area calculation
def total_area :=
  num_square_paintings * square_painting_area +
  num_small_paintings * small_painting_area +
  num_large_paintings * large_painting_area

-- The theorem statement
theorem art_collection_total_area : total_area = 282 := by
  sorry

end NUMINAMATH_GPT_art_collection_total_area_l934_93427


namespace NUMINAMATH_GPT_simplify_pow_prod_eq_l934_93430

noncomputable def simplify_pow_prod : ℝ :=
  (256:ℝ)^(1/4) * (625:ℝ)^(1/2)

theorem simplify_pow_prod_eq :
  simplify_pow_prod = 100 := by
  sorry

end NUMINAMATH_GPT_simplify_pow_prod_eq_l934_93430


namespace NUMINAMATH_GPT_average_age_l934_93452

def proportion (x y z : ℕ) : Prop :=  y / x = 3 ∧ z / x = 4

theorem average_age (A B C : ℕ) 
    (h1 : proportion 2 6 8)
    (h2 : A = 15)
    (h3 : B = 45)
    (h4 : C = 60) :
    (A + B + C) / 3 = 40 := 
    by
    sorry

end NUMINAMATH_GPT_average_age_l934_93452


namespace NUMINAMATH_GPT_relative_errors_are_equal_l934_93468

theorem relative_errors_are_equal :
  let e1 := 0.04
  let l1 := 20.0
  let e2 := 0.3
  let l2 := 150.0
  (e1 / l1) = (e2 / l2) :=
by
  sorry

end NUMINAMATH_GPT_relative_errors_are_equal_l934_93468


namespace NUMINAMATH_GPT_sum_of_arith_seq_l934_93471

noncomputable def f (x : ℝ) : ℝ := (x - 3)^3 + x - 1

def is_arith_seq (a : ℕ → ℝ) : Prop := 
  ∃ d ≠ 0, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_arith_seq (a : ℕ → ℝ) (h_a : is_arith_seq a)
  (h_f_sum : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) :
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_arith_seq_l934_93471


namespace NUMINAMATH_GPT_part_I_part_II_l934_93457

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.log (a * x + 1 / 2)) + (2 / (2 * x + 1))

theorem part_I (a : ℝ) (h_a_pos : a > 0) : (∀ x > 0, (1 / ((2 * x + 1) * (a * (2 * x + 1) - (2 * (a * x + 1) / 2))) ≥ 0) ↔ a ≥ 2) :=
sorry

theorem part_II : ∃ a : ℝ, (∀ x > 0, (Real.log (a * x + 1 / 2)) + (2 / (2 * x + 1)) ≥ 1) ∧ (Real.log (a * (Real.sqrt ((2 - a) / (4 * a))) + 1 / 2) + (2 / (2 * (Real.sqrt ((2 - a) / (4 * a))) + 1)) = 1) ∧ a = 1 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l934_93457


namespace NUMINAMATH_GPT_initial_winning_percentage_calc_l934_93464

variable (W : ℝ)
variable (initial_matches : ℝ := 120)
variable (additional_wins : ℝ := 70)
variable (final_matches : ℝ := 190)
variable (final_average : ℝ := 0.52)
variable (initial_wins : ℝ := 29)

noncomputable def winning_percentage_initial :=
  (initial_wins / initial_matches) * 100

theorem initial_winning_percentage_calc :
  (W = initial_wins) →
  ((W + additional_wins) / final_matches = final_average) →
  winning_percentage_initial = 24.17 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_winning_percentage_calc_l934_93464


namespace NUMINAMATH_GPT_midpoint_product_zero_l934_93479

theorem midpoint_product_zero (x y : ℝ)
  (h_midpoint_x : (2 + x) / 2 = 4)
  (h_midpoint_y : (6 + y) / 2 = 3) :
  x * y = 0 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_product_zero_l934_93479


namespace NUMINAMATH_GPT_rational_solution_exists_l934_93414

theorem rational_solution_exists (a b c : ℤ) (x₀ y₀ z₀ : ℤ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h₁ : a * x₀^2 + b * y₀^2 + c * z₀^2 = 0) (h₂ : x₀ ≠ 0 ∨ y₀ ≠ 0 ∨ z₀ ≠ 0) : 
  ∃ (x y z : ℚ), a * x^2 + b * y^2 + c * z^2 = 1 := 
sorry

end NUMINAMATH_GPT_rational_solution_exists_l934_93414


namespace NUMINAMATH_GPT_solution_set_of_inequality_l934_93419

theorem solution_set_of_inequality :
  { x : ℝ | (x - 5) / (x + 1) ≤ 0 } = { x : ℝ | -1 < x ∧ x ≤ 5 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l934_93419


namespace NUMINAMATH_GPT_oranges_thrown_away_l934_93496

theorem oranges_thrown_away (initial_oranges old_oranges_thrown new_oranges final_oranges : ℕ) 
    (h1 : initial_oranges = 34)
    (h2 : new_oranges = 13)
    (h3 : final_oranges = 27)
    (h4 : initial_oranges - old_oranges_thrown + new_oranges = final_oranges) :
    old_oranges_thrown = 20 :=
by
  sorry

end NUMINAMATH_GPT_oranges_thrown_away_l934_93496


namespace NUMINAMATH_GPT_Albaszu_machine_productivity_l934_93460

theorem Albaszu_machine_productivity (x : ℝ) 
  (h1 : 1.5 * x = 25) : x = 16 := 
by 
  sorry

end NUMINAMATH_GPT_Albaszu_machine_productivity_l934_93460


namespace NUMINAMATH_GPT_highest_score_is_96_l934_93415

theorem highest_score_is_96 :
  let standard_score := 85
  let deviations := [-9, -4, 11, -7, 0]
  let actual_scores := deviations.map (λ x => standard_score + x)
  actual_scores.maximum = 96 :=
by
  sorry

end NUMINAMATH_GPT_highest_score_is_96_l934_93415


namespace NUMINAMATH_GPT_slices_needed_l934_93426

def slices_per_sandwich : ℕ := 3
def number_of_sandwiches : ℕ := 5

theorem slices_needed : slices_per_sandwich * number_of_sandwiches = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_slices_needed_l934_93426


namespace NUMINAMATH_GPT_manuscript_typing_cost_l934_93443

theorem manuscript_typing_cost 
  (pages_total : ℕ) (pages_first_time : ℕ) (pages_revised_once : ℕ)
  (pages_revised_twice : ℕ) (rate_first_time : ℕ) (rate_revised : ℕ) 
  (cost_total : ℕ) :
  pages_total = 100 →
  pages_first_time = pages_total →
  pages_revised_once = 35 →
  pages_revised_twice = 15 →
  rate_first_time = 6 →
  rate_revised = 4 →
  cost_total = (pages_first_time * rate_first_time) +
              (pages_revised_once * rate_revised) +
              (pages_revised_twice * rate_revised * 2) →
  cost_total = 860 :=
by
  intros htot hfirst hrev1 hrev2 hr1 hr2 hcost
  sorry

end NUMINAMATH_GPT_manuscript_typing_cost_l934_93443


namespace NUMINAMATH_GPT_doris_hourly_wage_l934_93497

-- Defining the conditions from the problem
def money_needed : ℕ := 1200
def weekday_hours_per_day : ℕ := 3
def saturday_hours_per_day : ℕ := 5
def weeks_needed : ℕ := 3
def weekdays_per_week : ℕ := 5
def saturdays_per_week : ℕ := 1

-- Calculating total hours worked by Doris in 3 weeks
def total_hours (w_hours: ℕ) (s_hours: ℕ) 
    (w_days : ℕ) (s_days : ℕ) (weeks : ℕ) : ℕ := 
    (w_days * w_hours + s_days * s_hours) * weeks

-- Defining the weekly work hours
def weekly_hours := total_hours weekday_hours_per_day saturday_hours_per_day weekdays_per_week saturdays_per_week 1

-- Result of hours worked in 3 weeks
def hours_worked_in_3_weeks := weekly_hours * weeks_needed

-- Define the proof task
theorem doris_hourly_wage : 
  (money_needed : ℕ) / (hours_worked_in_3_weeks : ℕ) = 20 := by 
  sorry

end NUMINAMATH_GPT_doris_hourly_wage_l934_93497


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l934_93450

def int_divisible_by_5 (n : ℤ) := ∃ k : ℤ, n = 5 * k
def int_odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℤ, int_divisible_by_5 n → int_odd n) ↔ (∃ n : ℤ, int_divisible_by_5 n ∧ ¬ int_odd n) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l934_93450


namespace NUMINAMATH_GPT_units_digit_2_pow_2010_5_pow_1004_14_pow_1002_l934_93432

theorem units_digit_2_pow_2010_5_pow_1004_14_pow_1002 :
  (2^2010 * 5^1004 * 14^1002) % 10 = 0 := by
sorry

end NUMINAMATH_GPT_units_digit_2_pow_2010_5_pow_1004_14_pow_1002_l934_93432


namespace NUMINAMATH_GPT_find_set_M_l934_93423

variable (U M : Set ℕ)
variable [DecidableEq ℕ]

-- Universel set U is {1, 3, 5, 7}
def universal_set : Set ℕ := {1, 3, 5, 7}

-- define the complement C_U M
def complement (U M : Set ℕ) : Set ℕ := U \ M

-- M is the set to find such that complement of M in U is {5, 7}
theorem find_set_M (M : Set ℕ) (h : complement universal_set M = {5, 7}) : M = {1, 3} := by
  sorry

end NUMINAMATH_GPT_find_set_M_l934_93423


namespace NUMINAMATH_GPT_original_number_of_men_l934_93434

theorem original_number_of_men (x : ℕ) (h1 : x * 50 = (x - 10) * 60) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_men_l934_93434


namespace NUMINAMATH_GPT_green_beans_count_l934_93483

def total_beans := 572
def red_beans := (1 / 4) * total_beans
def remaining_after_red := total_beans - red_beans
def white_beans := (1 / 3) * remaining_after_red
def remaining_after_white := remaining_after_red - white_beans
def green_beans := (1 / 2) * remaining_after_white

theorem green_beans_count : green_beans = 143 := by
  sorry

end NUMINAMATH_GPT_green_beans_count_l934_93483


namespace NUMINAMATH_GPT_find_retail_price_l934_93476

-- Define the conditions
def wholesale_price : ℝ := 90
def discount_rate : ℝ := 0.10
def profit_rate : ℝ := 0.20

-- Calculate the necessary values from conditions
def profit : ℝ := profit_rate * wholesale_price
def selling_price : ℝ := wholesale_price + profit
def discount_factor : ℝ := 1 - discount_rate

-- Rewrite the main theorem statement
theorem find_retail_price : ∃ w : ℝ, discount_factor * w = selling_price → w = 120 :=
by sorry

end NUMINAMATH_GPT_find_retail_price_l934_93476


namespace NUMINAMATH_GPT_probability_all_quitters_from_same_tribe_l934_93422

noncomputable def total_ways_to_choose_quitters : ℕ := Nat.choose 18 3

noncomputable def ways_all_from_tribe (n : ℕ) : ℕ := Nat.choose n 3

noncomputable def combined_ways_same_tribe : ℕ :=
  ways_all_from_tribe 9 + ways_all_from_tribe 9

noncomputable def probability_same_tribe (total : ℕ) (same_tribe : ℕ) : ℚ :=
  same_tribe / total

theorem probability_all_quitters_from_same_tribe :
  probability_same_tribe total_ways_to_choose_quitters combined_ways_same_tribe = 7 / 34 :=
by
  sorry

end NUMINAMATH_GPT_probability_all_quitters_from_same_tribe_l934_93422


namespace NUMINAMATH_GPT_number_of_intersection_points_l934_93456

theorem number_of_intersection_points : 
  ∃! (P : ℝ × ℝ), 
    (P.1 ^ 2 + P.2 ^ 2 = 16) ∧ (P.1 = 4) := 
by
  sorry

end NUMINAMATH_GPT_number_of_intersection_points_l934_93456


namespace NUMINAMATH_GPT_mixture_price_l934_93485

-- Define constants
noncomputable def V1 (X : ℝ) : ℝ := 3.50 * X
noncomputable def V2 : ℝ := 4.30 * 6.25
noncomputable def W2 : ℝ := 6.25
noncomputable def W1 (X : ℝ) : ℝ := X

-- Define the total mixture weight condition
theorem mixture_price (X : ℝ) (P : ℝ) (h1 : W1 X + W2 = 10) (h2 : 10 * P = V1 X + V2) :
  P = 4 := by
  sorry

end NUMINAMATH_GPT_mixture_price_l934_93485


namespace NUMINAMATH_GPT_xiaoming_department_store_profit_l934_93445

theorem xiaoming_department_store_profit:
  let P₁ := 40000   -- average monthly profit in Q1
  let L₂ := -15000  -- average monthly loss in Q2
  let L₃ := -18000  -- average monthly loss in Q3
  let P₄ := 32000   -- average monthly profit in Q4
  let P_total := (P₁ * 3 + L₂ * 3 + L₃ * 3 + P₄ * 3)
  P_total = 117000 := by
  sorry

end NUMINAMATH_GPT_xiaoming_department_store_profit_l934_93445


namespace NUMINAMATH_GPT_amanda_tickets_l934_93421

theorem amanda_tickets (F : ℕ) (h : 4 * F + 32 + 28 = 80) : F = 5 :=
by
  sorry

end NUMINAMATH_GPT_amanda_tickets_l934_93421


namespace NUMINAMATH_GPT_correct_expression_l934_93478

theorem correct_expression (x : ℝ) :
  (x^3 / x^2 = x) :=
by sorry

end NUMINAMATH_GPT_correct_expression_l934_93478


namespace NUMINAMATH_GPT_persimmons_picked_l934_93435

theorem persimmons_picked : 
  ∀ (J H : ℕ), (4 * J = H - 3) → (H = 35) → (J = 8) := 
by
  intros J H hJ hH
  sorry

end NUMINAMATH_GPT_persimmons_picked_l934_93435


namespace NUMINAMATH_GPT_percentage_problem_l934_93465

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 := 
by
  sorry

end NUMINAMATH_GPT_percentage_problem_l934_93465


namespace NUMINAMATH_GPT_smallest_positive_z_l934_93470

theorem smallest_positive_z (x z : ℝ) (hx : Real.sin x = 1) (hz : Real.sin (x + z) = -1/2) : z = 2 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_z_l934_93470


namespace NUMINAMATH_GPT_cody_candy_total_l934_93429

theorem cody_candy_total
  (C_c : ℕ) (C_m : ℕ) (P_b : ℕ)
  (h1 : C_c = 7) (h2 : C_m = 3) (h3 : P_b = 8) :
  (C_c + C_m) * P_b = 80 :=
by
  sorry

end NUMINAMATH_GPT_cody_candy_total_l934_93429


namespace NUMINAMATH_GPT_science_book_pages_l934_93410

theorem science_book_pages {history_pages novel_pages science_pages: ℕ} (h1: novel_pages = history_pages / 2) (h2: science_pages = 4 * novel_pages) (h3: history_pages = 300):
  science_pages = 600 :=
by
  sorry

end NUMINAMATH_GPT_science_book_pages_l934_93410


namespace NUMINAMATH_GPT_compare_game_A_and_C_l934_93498

-- Probability definitions for coin toss
def p_heads := 2/3
def p_tails := 1/3

-- Probability of winning Game A
def prob_win_A := (p_heads^3) + (p_tails^3)

-- Probability of winning Game C
def prob_win_C := (p_heads^3 + p_tails^3)^2

-- Theorem statement to compare chances of winning Game A to Game C
theorem compare_game_A_and_C : prob_win_A - prob_win_C = 2/9 := by sorry

end NUMINAMATH_GPT_compare_game_A_and_C_l934_93498


namespace NUMINAMATH_GPT_severe_flood_probability_next_10_years_l934_93477

variable (A B C : Prop)
variable (P : Prop → ℝ)
variable (P_A : P A = 0.8)
variable (P_B : P B = 0.85)
variable (thirty_years_no_flood : ¬A)

theorem severe_flood_probability_next_10_years :
  P C = (P B - P A) / (1 - P A) := by
  sorry

end NUMINAMATH_GPT_severe_flood_probability_next_10_years_l934_93477


namespace NUMINAMATH_GPT_a6_minus_b6_divisible_by_9_l934_93487

theorem a6_minus_b6_divisible_by_9 {a b : ℤ} (h₁ : a % 3 ≠ 0) (h₂ : b % 3 ≠ 0) : (a ^ 6 - b ^ 6) % 9 = 0 := 
sorry

end NUMINAMATH_GPT_a6_minus_b6_divisible_by_9_l934_93487


namespace NUMINAMATH_GPT_probability_of_three_heads_in_eight_tosses_l934_93408

noncomputable def coin_toss_probability : ℚ :=
  let total_outcomes := 2^8
  let favorable_outcomes := Nat.choose 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_three_heads_in_eight_tosses : coin_toss_probability = 7 / 32 :=
  by
  sorry

end NUMINAMATH_GPT_probability_of_three_heads_in_eight_tosses_l934_93408


namespace NUMINAMATH_GPT_whole_milk_fat_percentage_l934_93490

def fat_in_some_milk : ℝ := 4
def percentage_less : ℝ := 0.5

theorem whole_milk_fat_percentage : ∃ (x : ℝ), fat_in_some_milk = percentage_less * x ∧ x = 8 :=
sorry

end NUMINAMATH_GPT_whole_milk_fat_percentage_l934_93490


namespace NUMINAMATH_GPT_percentage_less_than_l934_93474

theorem percentage_less_than (x y : ℝ) (P : ℝ) (h1 : y = 1.6667 * x) (h2 : x = (1 - P / 100) * y) : P = 66.67 :=
sorry

end NUMINAMATH_GPT_percentage_less_than_l934_93474


namespace NUMINAMATH_GPT_least_positive_integer_x_l934_93482

theorem least_positive_integer_x (x : ℕ) (h : x + 5683 ≡ 420 [MOD 17]) : x = 7 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_x_l934_93482


namespace NUMINAMATH_GPT_find_n_l934_93453

theorem find_n 
  (n : ℕ) 
  (b : ℕ → ℝ)
  (h₀ : b 0 = 28)
  (h₁ : b 1 = 81)
  (hn : b n = 0)
  (h_rec : ∀ j : ℕ, 1 ≤ j → j < n → b (j+1) = b (j-1) - 5 / b j)
  : n = 455 := 
sorry

end NUMINAMATH_GPT_find_n_l934_93453


namespace NUMINAMATH_GPT_students_neither_cs_nor_elec_l934_93466

theorem students_neither_cs_nor_elec
  (total_students : ℕ)
  (cs_students : ℕ)
  (elec_students : ℕ)
  (both_cs_and_elec : ℕ)
  (h_total : total_students = 150)
  (h_cs : cs_students = 90)
  (h_elec : elec_students = 60)
  (h_both : both_cs_and_elec = 20) :
  (total_students - (cs_students + elec_students - both_cs_and_elec) = 20) :=
by
  sorry

end NUMINAMATH_GPT_students_neither_cs_nor_elec_l934_93466


namespace NUMINAMATH_GPT_books_in_school_libraries_correct_l934_93475

noncomputable def booksInSchoolLibraries : ℕ :=
  let booksInPublicLibrary := 1986
  let totalBooks := 7092
  totalBooks - booksInPublicLibrary

-- Now we create a theorem to check the correctness of our definition
theorem books_in_school_libraries_correct :
  booksInSchoolLibraries = 5106 := by
  sorry -- We skip the proof, as instructed

end NUMINAMATH_GPT_books_in_school_libraries_correct_l934_93475


namespace NUMINAMATH_GPT_number_of_men_in_first_group_l934_93491

/-- The number of men in the first group that can complete a piece of work in 5 days alongside 16 boys,
    given that 13 men and 24 boys can complete the same work in 4 days, and the ratio of daily work done 
    by a man to a boy is 2:1, is 12. -/
theorem number_of_men_in_first_group
  (x : ℕ)  -- define x as the amount of work a boy can do in a day
  (m : ℕ)  -- define m as the number of men in the first group
  (h1 : ∀ (x : ℕ), 5 * (m * 2 * x + 16 * x) = 4 * (13 * 2 * x + 24 * x))
  (h2 : 2 * x = x + x) : m = 12 :=
sorry

end NUMINAMATH_GPT_number_of_men_in_first_group_l934_93491


namespace NUMINAMATH_GPT_dice_surface_sum_l934_93472

theorem dice_surface_sum :
  ∃ X : ℤ, 1 ≤ X ∧ X ≤ 6 ∧ 
  (28175 + 2 * X = 28177 ∨
   28175 + 2 * X = 28179 ∨
   28175 + 2 * X = 28181 ∨
   28175 + 2 * X = 28183 ∨
   28175 + 2 * X = 28185 ∨
   28175 + 2 * X = 28187) := sorry

end NUMINAMATH_GPT_dice_surface_sum_l934_93472


namespace NUMINAMATH_GPT_problem_r_minus_s_l934_93459

theorem problem_r_minus_s (r s : ℝ) (h1 : r ≠ s) (h2 : ∀ x : ℝ, (6 * x - 18) / (x ^ 2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) (h3 : r > s) : r - s = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_r_minus_s_l934_93459


namespace NUMINAMATH_GPT_total_artworks_created_l934_93406

theorem total_artworks_created
  (students_group1 : ℕ := 24) (students_group2 : ℕ := 12)
  (kits_total : ℕ := 48)
  (kits_per_3_students : ℕ := 3) (kits_per_2_students : ℕ := 2)
  (artwork_types : ℕ := 3)
  (paintings_group1_1 : ℕ := 12 * 2) (drawings_group1_1 : ℕ := 12 * 4) (sculptures_group1_1 : ℕ := 12 * 1)
  (paintings_group1_2 : ℕ := 12 * 1) (drawings_group1_2 : ℕ := 12 * 5) (sculptures_group1_2 : ℕ := 12 * 3)
  (paintings_group2_1 : ℕ := 4 * 3) (drawings_group2_1 : ℕ := 4 * 6) (sculptures_group2_1 : ℕ := 4 * 3)
  (paintings_group2_2 : ℕ := 8 * 4) (drawings_group2_2 : ℕ := 8 * 7) (sculptures_group2_2 : ℕ := 8 * 1)
  : (paintings_group1_1 + paintings_group1_2 + paintings_group2_1 + paintings_group2_2) +
    (drawings_group1_1 + drawings_group1_2 + drawings_group2_1 + drawings_group2_2) +
    (sculptures_group1_1 + sculptures_group1_2 + sculptures_group2_1 + sculptures_group2_2) = 336 :=
by sorry

end NUMINAMATH_GPT_total_artworks_created_l934_93406


namespace NUMINAMATH_GPT_two_digit_remainder_one_when_divided_by_4_and_17_l934_93467

-- Given the conditions
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def yields_remainder (n d r : ℕ) : Prop := n % d = r

-- Define the main problem that checks if there is only one such number
theorem two_digit_remainder_one_when_divided_by_4_and_17 :
  ∃! n : ℕ, is_two_digit n ∧ yields_remainder n 4 1 ∧ yields_remainder n 17 1 :=
sorry

end NUMINAMATH_GPT_two_digit_remainder_one_when_divided_by_4_and_17_l934_93467


namespace NUMINAMATH_GPT_mia_min_stamps_l934_93494

theorem mia_min_stamps (x y : ℕ) (hx : 5 * x + 7 * y = 37) : x + y = 7 :=
sorry

end NUMINAMATH_GPT_mia_min_stamps_l934_93494


namespace NUMINAMATH_GPT_width_of_rectangle_l934_93405

theorem width_of_rectangle (w l : ℝ) (h1 : l = 2 * w) (h2 : l * w = 1) : w = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_width_of_rectangle_l934_93405


namespace NUMINAMATH_GPT_count_valid_combinations_l934_93418

-- Define the digits condition
def is_digit (d : ℕ) : Prop := d >= 0 ∧ d <= 9

-- Define the main proof statement
theorem count_valid_combinations (a b c: ℕ) (h1 : is_digit a)(h2 : is_digit b)(h3 : is_digit c) :
    (100 * a + 10 * b + c) + (100 * c + 10 * b + a) = 1069 → 
    ∃ (abc_combinations : ℕ), abc_combinations = 8 :=
by
  sorry

end NUMINAMATH_GPT_count_valid_combinations_l934_93418


namespace NUMINAMATH_GPT_exists_divisor_between_l934_93499

theorem exists_divisor_between (n a b : ℕ) (h_n_gt_8 : n > 8) 
  (h_div1 : a ∣ n) (h_div2 : b ∣ n) (h_neq : a ≠ b) 
  (h_lt : a < b) (h_eq : n = a^2 + b) : 
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end NUMINAMATH_GPT_exists_divisor_between_l934_93499


namespace NUMINAMATH_GPT_find_x_value_l934_93442

theorem find_x_value :
  ∀ (x : ℝ), 0.3 + 0.1 + 0.4 + x = 1 → x = 0.2 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_find_x_value_l934_93442


namespace NUMINAMATH_GPT_quadratic_inverse_condition_l934_93454

theorem quadratic_inverse_condition : 
  (∀ x₁ x₂ : ℝ, (x₁ ≥ 2 ∧ x₂ ≥ 2 ∧ x₁ ≠ x₂) → (x₁^2 - 4*x₁ + 5 ≠ x₂^2 - 4*x₂ + 5)) :=
sorry

end NUMINAMATH_GPT_quadratic_inverse_condition_l934_93454


namespace NUMINAMATH_GPT_largest_integer_modulo_l934_93402

theorem largest_integer_modulo (a : ℤ) : a < 93 ∧ a % 7 = 4 ∧ (∀ b : ℤ, b < 93 ∧ b % 7 = 4 → b ≤ a) ↔ a = 88 :=
by
    sorry

end NUMINAMATH_GPT_largest_integer_modulo_l934_93402


namespace NUMINAMATH_GPT_sequence_values_induction_proof_l934_93437

def seq (a : ℕ → ℤ) := a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = a n ^ 2 - 2 * n * a n + 2

theorem sequence_values (a : ℕ → ℤ) (h : seq a) :
  a 2 = 5 ∧ a 3 = 7 ∧ a 4 = 9 :=
sorry

theorem induction_proof (a : ℕ → ℤ) (h : seq a) :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end NUMINAMATH_GPT_sequence_values_induction_proof_l934_93437


namespace NUMINAMATH_GPT_solve_for_x_l934_93446

theorem solve_for_x (x : ℝ) (h : 3 + 5 * x = 28) : x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l934_93446


namespace NUMINAMATH_GPT_mark_deposit_is_88_l934_93433

-- Definitions according to the conditions
def markDeposit := 88
def bryanDeposit (m : ℕ) := 5 * m - 40

-- The theorem we need to prove
theorem mark_deposit_is_88 : markDeposit = 88 := 
by 
  -- Since the condition states Mark deposited $88,
  -- this is trivially true.
  sorry

end NUMINAMATH_GPT_mark_deposit_is_88_l934_93433
