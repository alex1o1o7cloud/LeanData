import Mathlib

namespace work_days_together_l473_47328

theorem work_days_together (d : ℕ) (h : d * (17 / 140) = 6 / 7) : d = 17 := by
  sorry

end work_days_together_l473_47328


namespace sin_double_angle_sub_pi_over_4_l473_47304

open Real

theorem sin_double_angle_sub_pi_over_4 (x : ℝ) (h : sin x = (sqrt 5 - 1) / 2) : 
  sin (2 * (x - π / 4)) = 2 - sqrt 5 :=
by
  sorry

end sin_double_angle_sub_pi_over_4_l473_47304


namespace drunk_drivers_traffic_class_l473_47352

-- Define the variables for drunk drivers and speeders
variable (d s : ℕ)

-- Define the given conditions as hypotheses
theorem drunk_drivers_traffic_class (h1 : d + s = 45) (h2 : s = 7 * d - 3) : d = 6 := by
  sorry

end drunk_drivers_traffic_class_l473_47352


namespace guinea_pigs_food_difference_l473_47385

theorem guinea_pigs_food_difference :
  ∀ (first second third total : ℕ),
  first = 2 →
  second = first * 2 →
  total = 13 →
  first + second + third = total →
  third - second = 3 :=
by 
  intros first second third total h1 h2 h3 h4
  sorry

end guinea_pigs_food_difference_l473_47385


namespace number_added_to_x_l473_47340

theorem number_added_to_x (x : ℕ) (some_number : ℕ) (h1 : x = 3) (h2 : x + some_number = 4) : some_number = 1 := 
by
  -- Given hypotheses can be used here
  sorry

end number_added_to_x_l473_47340


namespace line_slope_l473_47374

theorem line_slope (x1 y1 x2 y2 : ℝ) (h1 : x1 = 0) (h2 : y1 = 100) (h3 : x2 = 50) (h4 : y2 = 300) :
  (y2 - y1) / (x2 - x1) = 4 :=
by sorry

end line_slope_l473_47374


namespace wendy_pictures_in_one_album_l473_47376

theorem wendy_pictures_in_one_album 
  (total_pictures : ℕ) (pictures_per_album : ℕ) (num_other_albums : ℕ)
  (h_total : total_pictures = 45) (h_pictures_per_album : pictures_per_album = 2) 
  (h_num_other_albums : num_other_albums = 9) : 
  ∃ (pictures_in_one_album : ℕ), pictures_in_one_album = 27 :=
by {
  sorry
}

end wendy_pictures_in_one_album_l473_47376


namespace impossible_to_all_minus_l473_47395

def initial_grid : List (List Int) :=
  [[1, 1, -1, 1], 
   [-1, -1, 1, 1], 
   [1, 1, 1, 1], 
   [1, -1, 1, -1]]

-- Define the operation of flipping a row
def flip_row (grid : List (List Int)) (r : Nat) : List (List Int) :=
  grid.mapIdx (fun i row => if i == r then row.map (fun x => -x) else row)

-- Define the operation of flipping a column
def flip_col (grid : List (List Int)) (c : Nat) : List (List Int) :=
  grid.map (fun row => row.mapIdx (fun j x => if j == c then -x else x))

-- Predicate to check if all elements in the grid are -1
def all_minus (grid : List (List Int)) : Prop :=
  grid.all (fun row => row.all (fun x => x = -1))

-- The main theorem
theorem impossible_to_all_minus (init : List (List Int)) (hf1 : init = initial_grid) :
  ∀ grid, (grid = init ∨ ∃ r, grid = flip_row grid r ∨ ∃ c, grid = flip_col grid c) →
  ¬ all_minus grid := by
    sorry

end impossible_to_all_minus_l473_47395


namespace y_value_when_x_is_3_l473_47346

theorem y_value_when_x_is_3 :
  (x + y = 30) → (x - y = 12) → (x * y = 189) → (x = 3) → y = 63 :=
by 
  intros h1 h2 h3 h4
  sorry

end y_value_when_x_is_3_l473_47346


namespace base_of_third_term_l473_47329

theorem base_of_third_term (x : ℝ) (some_number : ℝ) :
  625^(-x) + 25^(-2 * x) + some_number^(-4 * x) = 14 → x = 0.25 → some_number = 125 / 1744 :=
by
  intros h1 h2
  sorry

end base_of_third_term_l473_47329


namespace decrease_neg_of_odd_and_decrease_nonneg_l473_47372

-- Define the properties of the function f
variable (f : ℝ → ℝ)

-- f is odd
def odd_function : Prop := ∀ x : ℝ, f (-x) = - f x

-- f is decreasing on [0, +∞)
def decreasing_on_nonneg : Prop := ∀ x1 x2 : ℝ, (0 ≤ x1) → (0 ≤ x2) → (x1 < x2 → f x1 > f x2)

-- Goal: f is decreasing on (-∞, 0)
def decreasing_on_neg : Prop := ∀ x1 x2 : ℝ, (x1 < 0) → (x2 < 0) → (x1 < x2) → f x1 > f x2

-- The theorem to be proved
theorem decrease_neg_of_odd_and_decrease_nonneg 
  (h_odd : odd_function f) (h_decreasing_nonneg : decreasing_on_nonneg f) :
  decreasing_on_neg f :=
sorry

end decrease_neg_of_odd_and_decrease_nonneg_l473_47372


namespace not_traversable_n_62_l473_47383

theorem not_traversable_n_62 :
  ¬ (∃ (path : ℕ → ℕ), ∀ i < 62, path (i + 1) = (path i + 8) % 62 ∨ path (i + 1) = (path i + 9) % 62 ∨ path (i + 1) = (path i + 10) % 62) :=
by sorry

end not_traversable_n_62_l473_47383


namespace simple_sampling_methods_l473_47370

theorem simple_sampling_methods :
  methods_of_implementing_simple_sampling = ["lottery method", "random number table method"] :=
sorry

end simple_sampling_methods_l473_47370


namespace pool_capacity_l473_47323

variables {T : ℕ} {A B C : ℕ → ℕ}

-- Conditions
def valve_rate_A (T : ℕ) : ℕ := T / 180
def valve_rate_B (T : ℕ) := valve_rate_A T + 60
def valve_rate_C (T : ℕ) := valve_rate_A T + 75

def combined_rate (T : ℕ) := valve_rate_A T + valve_rate_B T + valve_rate_C T

-- Theorem to prove
theorem pool_capacity (T : ℕ) (h1 : combined_rate T = T / 40) : T = 16200 :=
by
  sorry

end pool_capacity_l473_47323


namespace find_y_eq_l473_47345

theorem find_y_eq (y : ℝ) : (10 - y)^2 = 4 * y^2 → (y = 10 / 3 ∨ y = -10) :=
by
  intro h
  -- The detailed proof will be provided here
  sorry

end find_y_eq_l473_47345


namespace product_evaluation_l473_47309

theorem product_evaluation : 
  (7 - 5) * (7 - 4) * (7 - 3) * (7 - 2) * (7- 1) * 7 = 5040 := 
by 
  sorry

end product_evaluation_l473_47309


namespace solve_for_a_minus_b_l473_47338

theorem solve_for_a_minus_b (a b : ℚ) 
  (h1 : 2020 * a + 2024 * b = 2030) 
  (h2 : 2022 * a + 2026 * b = 2032) : 
  a - b = -4 := 
sorry

end solve_for_a_minus_b_l473_47338


namespace quadratic_inequality_solution_l473_47307

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 3 * x + 2 < 0 ↔ 1 < x ∧ x < 2 :=
by
  sorry

end quadratic_inequality_solution_l473_47307


namespace cost_of_each_burger_l473_47333

theorem cost_of_each_burger (purchases_per_day : ℕ) (total_days : ℕ) (total_amount_spent : ℕ)
  (h1 : purchases_per_day = 4) (h2 : total_days = 30) (h3 : total_amount_spent = 1560) : 
  total_amount_spent / (purchases_per_day * total_days) = 13 :=
by
  subst h1
  subst h2
  subst h3
  sorry

end cost_of_each_burger_l473_47333


namespace inequality_solution_set_result_l473_47308

theorem inequality_solution_set_result (a b x : ℝ) :
  (∀ x, a ≤ (3/4) * x^2 - 3 * x + 4 ∧ (3/4) * x^2 - 3 * x + 4 ≤ b) ∧ 
  (∀ x, x ∈ Set.Icc a b ↔ a ≤ x ∧ x ≤ b) →
  a + b = 4 := 
by
  sorry

end inequality_solution_set_result_l473_47308


namespace total_initial_amounts_l473_47386

theorem total_initial_amounts :
  ∃ (a j t : ℝ), a = 50 ∧ t = 50 ∧ (50 + j + 50 = 187.5) :=
sorry

end total_initial_amounts_l473_47386


namespace no_cell_with_sum_2018_l473_47335

theorem no_cell_with_sum_2018 : ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 4900 → (5 * x = 2018 → false) := 
by
  intros x hx
  have h_bound : 1 ≤ x ∧ x ≤ 4900 := hx
  sorry

end no_cell_with_sum_2018_l473_47335


namespace max_value_x_sub_2z_l473_47318

theorem max_value_x_sub_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 16) :
  ∃ m, m = 4 * Real.sqrt 5 ∧ ∀ x y z, x^2 + y^2 + z^2 = 16 → x - 2 * z ≤ m :=
sorry

end max_value_x_sub_2z_l473_47318


namespace train_speed_kmph_l473_47366

noncomputable def speed_of_train
  (train_length : ℝ) (bridge_cross_time : ℝ) (total_length : ℝ) : ℝ :=
  (total_length / bridge_cross_time) * 3.6

theorem train_speed_kmph
  (train_length : ℝ := 130) 
  (bridge_cross_time : ℝ := 30) 
  (total_length : ℝ := 245) : 
  speed_of_train train_length bridge_cross_time total_length = 29.4 := by
  sorry

end train_speed_kmph_l473_47366


namespace fraction_of_male_first_class_l473_47382

theorem fraction_of_male_first_class (total_passengers : ℕ) (percent_female : ℚ) (percent_first_class : ℚ)
    (females_in_coach : ℕ) (h1 : total_passengers = 120) (h2 : percent_female = 0.45) (h3 : percent_first_class = 0.10)
    (h4 : females_in_coach = 46) :
    (((percent_first_class * total_passengers - (percent_female * total_passengers - females_in_coach)))
    / (percent_first_class * total_passengers))  = 1 / 3 := 
by
  sorry

end fraction_of_male_first_class_l473_47382


namespace value_subtracted_l473_47332

theorem value_subtracted (x y : ℤ) (h1 : (x - 5) / 7 = 7) (h2 : (x - y) / 13 = 4) : y = 2 :=
sorry

end value_subtracted_l473_47332


namespace total_fishes_l473_47321

theorem total_fishes (Will_catfish : ℕ) (Will_eels : ℕ) (Henry_multiplier : ℕ) (Henry_return_fraction : ℚ) :
  Will_catfish = 16 → Will_eels = 10 → Henry_multiplier = 3 → Henry_return_fraction = 1 / 2 →
  (Will_catfish + Will_eels) + (Henry_multiplier * Will_catfish - (Henry_multiplier * Will_catfish / 2)) = 50 := 
by
  intros h1 h2 h3 h4
  sorry

end total_fishes_l473_47321


namespace staples_left_in_stapler_l473_47393

def initial_staples : ℕ := 50
def reports_stapled : ℕ := 3 * 12
def staples_per_report : ℕ := 1
def remaining_staples : ℕ := initial_staples - (reports_stapled * staples_per_report)

theorem staples_left_in_stapler : remaining_staples = 14 :=
by
  sorry

end staples_left_in_stapler_l473_47393


namespace path_count_in_grid_l473_47378

theorem path_count_in_grid :
  let grid_width := 6
  let grid_height := 5
  let total_steps := 8
  let right_steps := 5
  let up_steps := 3
  ∃ (C : Nat), C = Nat.choose total_steps up_steps ∧ C = 56 :=
by
  sorry

end path_count_in_grid_l473_47378


namespace four_digit_composite_l473_47368

theorem four_digit_composite (abcd : ℕ) (h : 1000 ≤ abcd ∧ abcd < 10000) :
  ∃ (m n : ℕ), m ≥ 2 ∧ n ≥ 2 ∧ m * n = (abcd * 10001) :=
by
  sorry

end four_digit_composite_l473_47368


namespace problem_solution_l473_47339

theorem problem_solution :
  ∃ n : ℕ, 50 < n ∧ n < 70 ∧ n % 5 = 3 ∧ n % 7 = 2 ∧ n = 58 :=
by
  -- Lean code to prove the statement
  sorry

end problem_solution_l473_47339


namespace area_is_300_l473_47322

variable (l w : ℝ) -- Length and Width of the playground

-- Conditions
def condition1 : Prop := 2 * l + 2 * w = 80
def condition2 : Prop := l = 3 * w

-- Question and Answer
def area_of_playground : ℝ := l * w

theorem area_is_300 (h1 : condition1 l w) (h2 : condition2 l w) : area_of_playground l w = 300 := 
by
  sorry

end area_is_300_l473_47322


namespace digital_earth_concept_wrong_l473_47317

theorem digital_earth_concept_wrong :
  ∀ (A C D : Prop),
  (A → true) →
  (C → true) →
  (D → true) →
  ¬(B → true) :=
by
  sorry

end digital_earth_concept_wrong_l473_47317


namespace baron_not_boasting_l473_47367

-- Define a function to verify if a given list of digits is a palindrome
def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

-- Define a list that represents the sequence given in the solution
def sequence_19 : List ℕ :=
  [9, 18, 7, 16, 5, 14, 3, 12, 1, 10, 11, 2, 13, 4, 15, 6, 17, 8, 19]

-- Prove that the sequence forms a palindrome
theorem baron_not_boasting : is_palindrome sequence_19 :=
by {
  -- Insert actual proof steps here
  sorry
}

end baron_not_boasting_l473_47367


namespace remainder_67_pow_67_plus_67_mod_68_l473_47347

theorem remainder_67_pow_67_plus_67_mod_68 :
  (67 ^ 67 + 67) % 68 = 66 :=
by
  -- Skip the proof for now
  sorry

end remainder_67_pow_67_plus_67_mod_68_l473_47347


namespace y_is_never_perfect_square_l473_47397

theorem y_is_never_perfect_square (x : ℕ) : ¬ ∃ k : ℕ, k^2 = x^4 + 2*x^3 + 2*x^2 + 2*x + 1 :=
sorry

end y_is_never_perfect_square_l473_47397


namespace distribution_ways_l473_47357

theorem distribution_ways :
  ∃ (n : ℕ) (erasers pencils notebooks pens : ℕ),
  pencils = 4 ∧ notebooks = 2 ∧ pens = 3 ∧ 
  n = 6 := sorry

end distribution_ways_l473_47357


namespace card_sequence_probability_l473_47391

-- Conditions about the deck and card suits
def standard_deck : ℕ := 52
def diamond_count : ℕ := 13
def spade_count : ℕ := 13
def heart_count : ℕ := 13

-- Definition of the problem statement
def diamond_first_prob : ℚ := diamond_count / standard_deck
def spade_second_prob : ℚ := spade_count / (standard_deck - 1)
def heart_third_prob : ℚ := heart_count / (standard_deck - 2)

-- Theorem statement for the required probability
theorem card_sequence_probability : 
    diamond_first_prob * spade_second_prob * heart_third_prob = 13 / 780 :=
by
  sorry

end card_sequence_probability_l473_47391


namespace c_left_days_before_completion_l473_47365

-- Definitions for the given conditions
def work_done_by_a_in_one_day := 1 / 30
def work_done_by_b_in_one_day := 1 / 30
def work_done_by_c_in_one_day := 1 / 40
def total_days := 12

-- Proof problem statement (to prove that c left 8 days before the completion)
theorem c_left_days_before_completion :
  ∃ x : ℝ, 
  (12 - x) * (7 / 60) + x * (1 / 15) = 1 → 
  x = 8 := sorry

end c_left_days_before_completion_l473_47365


namespace number_of_cubes_with_icing_on_two_sides_l473_47379

def cake_cube : ℕ := 3
def smaller_cubes : ℕ := 27
def covered_faces : ℕ := 3
def layers_with_icing : ℕ := 2
def edge_cubes_per_layer_per_face : ℕ := 2

theorem number_of_cubes_with_icing_on_two_sides :
  (covered_faces * edge_cubes_per_layer_per_face * layers_with_icing) = 12 := by
  sorry

end number_of_cubes_with_icing_on_two_sides_l473_47379


namespace line_through_fixed_point_l473_47302

-- Define the arithmetic sequence condition
def arithmetic_sequence (k b : ℝ) : Prop :=
  k + b = -2

-- Define the line passing through a fixed point
def line_passes_through (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ (x = 1 ∧ y = -2)

-- The theorem stating the main problem
theorem line_through_fixed_point (k b : ℝ) (h : arithmetic_sequence k b) : line_passes_through k b :=
  sorry

end line_through_fixed_point_l473_47302


namespace find_a_plus_b_l473_47344

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 2^(a * x + b)

theorem find_a_plus_b
  (a b : ℝ)
  (h1 : f a b 2 = 1 / 2)
  (h2 : f a b (1 / 2) = 2) :
  a + b = 1 / 3 :=
sorry

end find_a_plus_b_l473_47344


namespace sum_of_roots_l473_47349

theorem sum_of_roots : (x₁ x₂ : ℝ) → (h : 2 * x₁^2 + 6 * x₁ - 1 = 0) → (h₂ : 2 * x₂^2 + 6 * x₂ - 1 = 0) → x₁ + x₂ = -3 :=
by 
  sorry

end sum_of_roots_l473_47349


namespace num_tuples_abc_l473_47348

theorem num_tuples_abc (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 2019 ≥ 10 * a) (h5 : 10 * a ≥ 100 * b) (h6 : 100 * b ≥ 1000 * c) : 
  ∃ n, n = 574 := sorry

end num_tuples_abc_l473_47348


namespace diagonal_length_of_quadrilateral_l473_47373

theorem diagonal_length_of_quadrilateral 
  (area : ℝ) (m n : ℝ) (d : ℝ) 
  (h_area : area = 210) 
  (h_m : m = 9) 
  (h_n : n = 6) 
  (h_formula : area = 0.5 * d * (m + n)) : 
  d = 28 :=
by 
  sorry

end diagonal_length_of_quadrilateral_l473_47373


namespace benny_gave_sandy_books_l473_47319

theorem benny_gave_sandy_books :
  ∀ (Benny_initial Tim_books total_books Benny_after_giving : ℕ), 
    Benny_initial = 24 → 
    Tim_books = 33 →
    total_books = 47 → 
    total_books - Tim_books = Benny_after_giving →
    Benny_initial - Benny_after_giving = 10 :=
by
  intros Benny_initial Tim_books total_books Benny_after_giving
  intros hBenny_initial hTim_books htotal_books hBooks_after
  simp [hBenny_initial, hTim_books, htotal_books, hBooks_after]
  sorry


end benny_gave_sandy_books_l473_47319


namespace sufficient_condition_for_lg_m_lt_1_l473_47364

theorem sufficient_condition_for_lg_m_lt_1 (m : ℝ) (h1 : m ∈ ({1, 2} : Set ℝ)) : Real.log m < 1 :=
sorry

end sufficient_condition_for_lg_m_lt_1_l473_47364


namespace grape_juice_amount_l473_47390

-- Definitions for the conditions
def total_weight : ℝ := 150
def orange_percentage : ℝ := 0.35
def watermelon_percentage : ℝ := 0.35

-- Theorem statement to prove the amount of grape juice
theorem grape_juice_amount : 
  (total_weight * (1 - orange_percentage - watermelon_percentage)) = 45 :=
by
  sorry

end grape_juice_amount_l473_47390


namespace initial_legos_500_l473_47315

-- Definitions and conditions from the problem
def initial_legos (x : ℕ) : Prop :=
  let used_pieces := x / 2
  let remaining_pieces := x - used_pieces
  let boxed_pieces := remaining_pieces - 5
  boxed_pieces = 245

-- Statement to be proven
theorem initial_legos_500 : initial_legos 500 :=
by
  -- Proof goes here
  sorry

end initial_legos_500_l473_47315


namespace flat_tyre_problem_l473_47392

theorem flat_tyre_problem
    (x : ℝ)
    (h1 : 0 < x)
    (h2 : 1 / x + 1 / 6 = 1 / 5.6) :
  x = 84 :=
sorry

end flat_tyre_problem_l473_47392


namespace sequence_solution_l473_47313

theorem sequence_solution (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, (2*n - 1) * a (n + 1) = (2*n + 1) * a n) : 
∀ n : ℕ, a n = 2 * n - 1 := 
by
  sorry

end sequence_solution_l473_47313


namespace probability_fx_lt_0_l473_47380

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem probability_fx_lt_0 :
  (∫ x in -Real.pi..Real.pi, if f x < 0 then 1 else 0) / (2 * Real.pi) = 2 / Real.pi :=
by sorry

end probability_fx_lt_0_l473_47380


namespace minimum_product_value_l473_47311

-- Problem conditions
def total_stones : ℕ := 40
def b_min : ℕ := 20
def b_max : ℕ := 32

-- Define the product function
def P (b : ℕ) : ℕ := b * (total_stones - b)

-- Goal: Prove the minimum value of P(b) for b in [20, 32] is 256
theorem minimum_product_value : ∃ (b : ℕ), b_min ≤ b ∧ b ≤ b_max ∧ P b = 256 := by
  sorry

end minimum_product_value_l473_47311


namespace original_total_thumbtacks_l473_47351

-- Conditions
def num_cans : ℕ := 3
def num_boards_tested : ℕ := 120
def thumbtacks_per_board : ℕ := 3
def thumbtacks_remaining_per_can : ℕ := 30

-- Question
theorem original_total_thumbtacks :
  (num_cans * num_boards_tested * thumbtacks_per_board) + (num_cans * thumbtacks_remaining_per_can) = 450 :=
sorry

end original_total_thumbtacks_l473_47351


namespace three_digit_number_possibilities_l473_47398

theorem three_digit_number_possibilities (A B C : ℕ) (hA : A ≠ 0) (hC : C ≠ 0) (h_diff : A - C = 5) :
  ∃ (x : ℕ), x = 100 * A + 10 * B + C ∧ (x - (100 * C + 10 * B + A) = 495) ∧ ∃ n, n = 40 :=
by
  sorry

end three_digit_number_possibilities_l473_47398


namespace train_length_is_correct_l473_47312

noncomputable def length_of_train (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_s

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 :=
by 
  -- Here, a proof would be provided, eventually using the definitions and conditions given
  sorry

end train_length_is_correct_l473_47312


namespace find_a_plus_b_l473_47343

theorem find_a_plus_b (a b : ℤ) (h : 2*x^3 - a*x^2 - 5*x + 5 = (2*x^2 + a*x - 1)*(x - b) + 3) : a + b = 4 :=
by {
  -- Proof omitted
  sorry
}

end find_a_plus_b_l473_47343


namespace perfect_square_polynomial_l473_47334

theorem perfect_square_polynomial (m : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x : ℝ, x^2 + 2*(m-3)*x + 25 = f x * f x) ↔ (m = 8 ∨ m = -2) :=
by
  sorry

end perfect_square_polynomial_l473_47334


namespace x_varies_inversely_l473_47362

theorem x_varies_inversely (y: ℝ) (x: ℝ): (∃ k: ℝ, (∀ y: ℝ, x = k / y ^ 2) ∧ (1 = k / 3 ^ 2)) → x = 0.5625 :=
by
  sorry

end x_varies_inversely_l473_47362


namespace transmitter_finding_probability_l473_47396

/-- 
  A license plate in the country Kerrania consists of 4 digits followed by two letters.
  The letters A, B, and C are used only by government vehicles while the letters D through Z are used by non-government vehicles.
  Kerrania's intelligence agency has recently captured a message from the country Gonzalia indicating that an electronic transmitter 
  has been installed in a Kerrania government vehicle with a license plate starting with 79. 
  In addition, the message reveals that the last three digits of the license plate form a palindromic sequence (meaning that they are 
  the same forward and backward), and the second digit is either a 3 or a 5. 
  If it takes the police 10 minutes to inspect each vehicle, what is the probability that the police will find the transmitter 
  within 3 hours, considering the additional restrictions on the possible license plate combinations?
-/
theorem transmitter_finding_probability :
  0.1 = 18 / 180 :=
by
  sorry

end transmitter_finding_probability_l473_47396


namespace integer_square_mod_4_l473_47369

theorem integer_square_mod_4 (N : ℤ) : (N^2 % 4 = 0) ∨ (N^2 % 4 = 1) :=
by sorry

end integer_square_mod_4_l473_47369


namespace curve_is_circle_l473_47358

noncomputable def curve_eqn_polar (r θ : ℝ) : Prop :=
  r = 1 / (Real.sin θ + Real.cos θ)

theorem curve_is_circle : ∀ r θ, curve_eqn_polar r θ →
  ∃ x y : ℝ, r = Real.sqrt (x^2 + y^2) ∧ 
  x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ 
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2 :=
by
  sorry

end curve_is_circle_l473_47358


namespace length_PT_30_l473_47324

noncomputable def length_PT (PQ QR : ℝ) (angle_QRT : ℝ) (T_on_RS : Prop) : ℝ := 
  if h : PQ = 30 ∧ QR = 15 ∧ angle_QRT = 75 then 30 else 0

theorem length_PT_30 (PQ QR : ℝ) (angle_QRT : ℝ) (T_on_RS : Prop) :
  PQ = 30 → QR = 15 → angle_QRT = 75 → length_PT PQ QR angle_QRT T_on_RS = 30 :=
sorry

end length_PT_30_l473_47324


namespace A_subset_B_l473_47388

def A (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 ≤ 5 / 4

def B (x y : ℝ) (a : ℝ) : Prop :=
  abs (x - 1) + 2 * abs (y - 2) ≤ a

theorem A_subset_B (a : ℝ) (h : a ≥ 5 / 2) : 
  ∀ x y : ℝ, A x y → B x y a := 
sorry

end A_subset_B_l473_47388


namespace quadratic_roots_l473_47303

theorem quadratic_roots (x : ℝ) : 
  (x^2 + 3 * x - 1 = 0) ↔ (x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) := 
by
  sorry

end quadratic_roots_l473_47303


namespace average_of_25_results_is_24_l473_47387

theorem average_of_25_results_is_24 
  (first12_sum : ℕ)
  (last12_sum : ℕ)
  (result13 : ℕ)
  (n1 n2 n3 : ℕ)
  (h1 : n1 = 12)
  (h2 : n2 = 12)
  (h3 : n3 = 25)
  (avg_first12 : first12_sum = 14 * n1)
  (avg_last12 : last12_sum = 17 * n2)
  (res_13 : result13 = 228) :
  (first12_sum + last12_sum + result13) / n3 = 24 :=
by
  sorry

end average_of_25_results_is_24_l473_47387


namespace closest_perfect_square_l473_47384

theorem closest_perfect_square (n : ℕ) (h1 : n = 325) : 
    ∃ m : ℕ, m^2 = 324 ∧ 
    (∀ k : ℕ, (k^2 ≤ n ∨ k^2 ≥ n) → (k = 18 ∨ k^2 > 361 ∨ k^2 < 289)) := 
by
  sorry

end closest_perfect_square_l473_47384


namespace prob_board_251_l473_47306

noncomputable def probability_boarding_bus_251 (r1 r2 : ℕ) : ℚ :=
  let interval_152 := r1
  let interval_251 := r2
  let total_area := interval_152 * interval_251
  let triangle_area := 1 / 2 * interval_152 * interval_152
  triangle_area / total_area

theorem prob_board_251 : probability_boarding_bus_251 5 7 = 5 / 14 := by
  sorry

end prob_board_251_l473_47306


namespace find_d_l473_47371

theorem find_d (d : ℝ) (h : ∃ (x y : ℝ), 3 * x + 5 * y + d = 0 ∧ x = -d / 3 ∧ y = -d / 5 ∧ -d / 3 + (-d / 5) = 15) : d = -225 / 8 :=
by 
  sorry

end find_d_l473_47371


namespace rectangle_side_greater_than_12_l473_47355

theorem rectangle_side_greater_than_12 
  (a b : ℝ) (h₁ : a ≠ b) (h₂ : a * b = 6 * (a + b)) : a > 12 ∨ b > 12 := 
by
  sorry

end rectangle_side_greater_than_12_l473_47355


namespace slices_remaining_l473_47394

theorem slices_remaining (large_pizza_slices : ℕ) (xl_pizza_slices : ℕ) (large_pizza_ordered : ℕ) (xl_pizza_ordered : ℕ) (mary_eats_large : ℕ) (mary_eats_xl : ℕ) :
  large_pizza_slices = 8 →
  xl_pizza_slices = 12 →
  large_pizza_ordered = 1 →
  xl_pizza_ordered = 1 →
  mary_eats_large = 7 →
  mary_eats_xl = 3 →
  (large_pizza_slices * large_pizza_ordered - mary_eats_large + xl_pizza_slices * xl_pizza_ordered - mary_eats_xl) = 10 := 
by
  intros
  sorry

end slices_remaining_l473_47394


namespace number_of_solution_pairs_l473_47336

theorem number_of_solution_pairs : 
  ∃ n, (∀ x y : ℕ, 4 * x + 7 * y = 548 → (x > 0 ∧ y > 0) → n = 19) :=
sorry

end number_of_solution_pairs_l473_47336


namespace ab_plus_cd_eq_12_l473_47316

theorem ab_plus_cd_eq_12 (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = -1) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 5) :
  a * b + c * d = 12 := by
  sorry

end ab_plus_cd_eq_12_l473_47316


namespace emmalyn_earnings_l473_47310

theorem emmalyn_earnings :
  let rate_per_meter := 0.20
  let number_of_fences := 50
  let length_of_each_fence := 500
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * rate_per_meter
  total_income = 5000 :=
by
  sorry

end emmalyn_earnings_l473_47310


namespace point_A_coordinates_l473_47363

variable {a : ℝ}
variable {f : ℝ → ℝ}

theorem point_A_coordinates (h1 : a > 0) (h2 : a ≠ 1) (hf : ∀ x, f x = a^(x - 1)) :
  f 1 = 1 :=
by
  sorry

end point_A_coordinates_l473_47363


namespace determine_k_circle_l473_47337

theorem determine_k_circle (k : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 14*y - k = 0) ∧ ((∀ x y : ℝ, (x + 4)^2 + (y + 7)^2 = 25) ↔ k = -40) :=
by
  sorry

end determine_k_circle_l473_47337


namespace Adam_current_money_is_8_l473_47354

variable (Adam_initial : ℕ) (spent_on_game : ℕ) (allowance : ℕ)

def money_left_after_spending (initial : ℕ) (spent : ℕ) := initial - spent
def current_money (money_left : ℕ) (allowance : ℕ) := money_left + allowance

theorem Adam_current_money_is_8 
    (h1 : Adam_initial = 5)
    (h2 : spent_on_game = 2)
    (h3 : allowance = 5) :
    current_money (money_left_after_spending Adam_initial spent_on_game) allowance = 8 := 
by sorry

end Adam_current_money_is_8_l473_47354


namespace absolute_value_solution_l473_47361

theorem absolute_value_solution (m : ℤ) (h : abs m = abs (-7)) : m = 7 ∨ m = -7 := by
  sorry

end absolute_value_solution_l473_47361


namespace value_of_g_at_2_l473_47330

def g (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

theorem value_of_g_at_2 : g 2 = 11 := 
by
  sorry

end value_of_g_at_2_l473_47330


namespace number_of_packages_l473_47320

theorem number_of_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 56) (h2 : tshirts_per_package = 2) : 
  (total_tshirts / tshirts_per_package) = 28 := 
  by
    sorry

end number_of_packages_l473_47320


namespace find_c_plus_one_div_b_l473_47350

-- Assume that a, b, and c are positive real numbers such that the given conditions hold.
variables (a b c : ℝ)
variables (habc : a * b * c = 1)
variables (hac : a + 1 / c = 7)
variables (hba : b + 1 / a = 11)

-- The goal is to show that c + 1 / b = 5 / 19.
theorem find_c_plus_one_div_b : c + 1 / b = 5 / 19 :=
by 
  sorry

end find_c_plus_one_div_b_l473_47350


namespace determine_b_l473_47356

-- Define the problem conditions
variable (n b : ℝ)
variable (h_pos_b : b > 0)
variable (h_eq : ∀ x : ℝ, (x + n) ^ 2 + 16 = x^2 + b * x + 88)

-- State that we want to prove that b equals 12 * sqrt(2)
theorem determine_b : b = 12 * Real.sqrt 2 :=
by
  sorry

end determine_b_l473_47356


namespace cat_food_sufficiency_l473_47300

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l473_47300


namespace unplanted_fraction_l473_47360

theorem unplanted_fraction (a b hypotenuse : ℕ) (side_length_P : ℚ) 
                          (h1 : a = 5) (h2 : b = 12) (h3 : hypotenuse = 13)
                          (h4 : side_length_P = 5 / 3) : 
                          (side_length_P * side_length_P) / ((a * b) / 2) = 5 / 54 :=
by
  sorry

end unplanted_fraction_l473_47360


namespace symmetric_line_equation_l473_47389

theorem symmetric_line_equation (x y : ℝ) :
  let line_original := x - 2 * y + 1 = 0
  let line_symmetry := x = 1
  let line_symmetric := x + 2 * y - 3 = 0
  ∀ (x y : ℝ), (2 - x - 2 * y + 1 = 0) ↔ (x + 2 * y - 3 = 0) := by
sorry

end symmetric_line_equation_l473_47389


namespace sum_of_4n_pos_integers_l473_47375

theorem sum_of_4n_pos_integers (n : ℕ) (Sn : ℕ → ℕ)
  (hSn : ∀ k, Sn k = k * (k + 1) / 2)
  (h_condition : Sn (3 * n) - Sn n = 150) :
  Sn (4 * n) = 300 :=
by {
  sorry
}

end sum_of_4n_pos_integers_l473_47375


namespace radius_of_sphere_is_two_sqrt_46_l473_47341

theorem radius_of_sphere_is_two_sqrt_46
  (a b c : ℝ)
  (s : ℝ)
  (h1 : 4 * (a + b + c) = 160)
  (h2 : 2 * (a * b + b * c + c * a) = 864)
  (h3 : s = Real.sqrt ((a^2 + b^2 + c^2) / 4)) :
  s = 2 * Real.sqrt 46 :=
by
  -- proof placeholder
  sorry

end radius_of_sphere_is_two_sqrt_46_l473_47341


namespace minimize_x_l473_47314

theorem minimize_x (x y : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : x + y^2 = x * y) : x ≥ 3 :=
sorry

end minimize_x_l473_47314


namespace pull_ups_per_time_l473_47305

theorem pull_ups_per_time (pull_ups_week : ℕ) (times_day : ℕ) (days_week : ℕ)
  (h1 : pull_ups_week = 70) (h2 : times_day = 5) (h3 : days_week = 7) :
  pull_ups_week / (times_day * days_week) = 2 := by
  sorry

end pull_ups_per_time_l473_47305


namespace long_side_length_l473_47399

variable {a b d : ℝ}

theorem long_side_length (h1 : a / b = 2 * (b / d)) (h2 : a = 4) (hd : d = Real.sqrt (a^2 + b^2)) :
  b = Real.sqrt (2 + 4 * Real.sqrt 17) :=
sorry

end long_side_length_l473_47399


namespace books_difference_l473_47342

theorem books_difference (bobby_books : ℕ) (kristi_books : ℕ) (h1 : bobby_books = 142) (h2 : kristi_books = 78) : bobby_books - kristi_books = 64 :=
by {
  -- Placeholder for the proof
  sorry
}

end books_difference_l473_47342


namespace solve_graph_equation_l473_47326

/- Problem:
Solve for the graph of the equation x^2(x+y+2)=y^2(x+y+2)
Given condition: equation x^2(x+y+2)=y^2(x+y+2)
Conclusion: Three lines that do not all pass through a common point
The final answer should be formally proven.
-/

theorem solve_graph_equation (x y : ℝ) :
  (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (∃ a b c d : ℝ,  (a = -x - 2 ∧ b = -x ∧ c = x ∧ (a ≠ b ∧ a ≠ c ∧ b ≠ c)) ∧
   (d = 0) ∧ ¬ ∀ p q r : ℝ, p = q ∧ q = r ∧ r = p) :=
by
  sorry

end solve_graph_equation_l473_47326


namespace calculate_total_shaded_area_l473_47301

theorem calculate_total_shaded_area
(smaller_square_side larger_square_side smaller_circle_radius larger_circle_radius : ℝ)
(h1 : smaller_square_side = 6)
(h2 : larger_square_side = 12)
(h3 : smaller_circle_radius = 3)
(h4 : larger_circle_radius = 6) :
  (smaller_square_side^2 - π * smaller_circle_radius^2) + 
  (larger_square_side^2 - π * larger_circle_radius^2) = 180 - 45 * π :=
by
  sorry

end calculate_total_shaded_area_l473_47301


namespace min_value_of_expression_l473_47353

open Real

theorem min_value_of_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 :=
by
  sorry

end min_value_of_expression_l473_47353


namespace age_difference_l473_47327

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 16) : A - C = 16 :=
sorry

end age_difference_l473_47327


namespace range_of_a_l473_47331

theorem range_of_a :
  (∃ a : ℝ, (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0)
    ↔ a ≤ -2 ∨ a = 1) := 
sorry

end range_of_a_l473_47331


namespace find_max_sum_of_squares_l473_47359

open Real

theorem find_max_sum_of_squares 
  (a b c d : ℝ)
  (h1 : a + b = 17)
  (h2 : ab + c + d = 98)
  (h3 : ad + bc = 176)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 770 :=
sorry

end find_max_sum_of_squares_l473_47359


namespace find_p_tilde_one_l473_47381

noncomputable def p (x : ℝ) : ℝ :=
  let r : ℝ := -1 / 9
  let s : ℝ := 1
  x^2 - (r + s) * x + (r * s)

theorem find_p_tilde_one : p 1 = 0 := by
  sorry

end find_p_tilde_one_l473_47381


namespace ratio_of_ages_ten_years_ago_l473_47325

theorem ratio_of_ages_ten_years_ago (A T : ℕ) 
    (h1: A = 30) 
    (h2: T = A - 15) : 
    (A - 10) / (T - 10) = 4 :=
by
  sorry

end ratio_of_ages_ten_years_ago_l473_47325


namespace abs_eq_condition_l473_47377

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l473_47377
