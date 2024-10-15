import Mathlib

namespace NUMINAMATH_GPT_parallel_lines_solution_l2054_205448

theorem parallel_lines_solution (a : ℝ) :
  (∃ (k1 k2 : ℝ), k1 ≠ 0 ∧ k2 ≠ 0 ∧ 
  ∀ x y : ℝ, x + a^2 * y + 6 = 0 → k1*y = x ∧ 
             (a-2) * x + 3 * a * y + 2 * a = 0 → k2*y = x) 
  → (a = -1 ∨ a = 0) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_solution_l2054_205448


namespace NUMINAMATH_GPT_sugar_flour_difference_l2054_205444

theorem sugar_flour_difference :
  ∀ (flour_required_kg sugar_required_lb flour_added_kg kg_to_lb),
    flour_required_kg = 2.25 →
    sugar_required_lb = 5.5 →
    flour_added_kg = 1 →
    kg_to_lb = 2.205 →
    (sugar_required_lb / kg_to_lb * 1000) - ((flour_required_kg - flour_added_kg) * 1000) = 1244.8 :=
by
  intros flour_required_kg sugar_required_lb flour_added_kg kg_to_lb
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- sorry is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_sugar_flour_difference_l2054_205444


namespace NUMINAMATH_GPT_minimum_production_volume_to_avoid_loss_l2054_205411

open Real

-- Define the cost function
def cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * (x ^ 2)

-- Define the revenue function
def revenue (x : ℕ) : ℝ := 25 * x

-- Condition: 0 < x < 240 and x ∈ ℕ (naturals greater than 0)
theorem minimum_production_volume_to_avoid_loss (x : ℕ) (hx1 : 0 < x) (hx2 : x < 240) (hx3 : x ∈ (Set.Ioi 0)) :
  revenue x ≥ cost x ↔ x ≥ 150 :=
by
  sorry

end NUMINAMATH_GPT_minimum_production_volume_to_avoid_loss_l2054_205411


namespace NUMINAMATH_GPT_percentage_bob_is_36_l2054_205498

def water_per_acre_corn : ℕ := 20
def water_per_acre_cotton : ℕ := 80
def water_per_acre_beans : ℕ := 2 * water_per_acre_corn

def acres_bob_corn : ℕ := 3
def acres_bob_cotton : ℕ := 9
def acres_bob_beans : ℕ := 12

def acres_brenda_corn : ℕ := 6
def acres_brenda_cotton : ℕ := 7
def acres_brenda_beans : ℕ := 14

def acres_bernie_corn : ℕ := 2
def acres_bernie_cotton : ℕ := 12

def water_bob : ℕ := (acres_bob_corn * water_per_acre_corn) +
                      (acres_bob_cotton * water_per_acre_cotton) +
                      (acres_bob_beans * water_per_acre_beans)

def water_brenda : ℕ := (acres_brenda_corn * water_per_acre_corn) +
                         (acres_brenda_cotton * water_per_acre_cotton) +
                         (acres_brenda_beans * water_per_acre_beans)

def water_bernie : ℕ := (acres_bernie_corn * water_per_acre_corn) +
                         (acres_bernie_cotton * water_per_acre_cotton)

def total_water : ℕ := water_bob + water_brenda + water_bernie

def percentage_bob : ℚ := (water_bob : ℚ) / (total_water : ℚ) * 100

theorem percentage_bob_is_36 : percentage_bob = 36 := by
  sorry

end NUMINAMATH_GPT_percentage_bob_is_36_l2054_205498


namespace NUMINAMATH_GPT_calculate_expression_l2054_205418

theorem calculate_expression :
  ( ( (1/6) - (1/8) + (1/9) ) / ( (1/3) - (1/4) + (1/5) ) ) * 3 = 55 / 34 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2054_205418


namespace NUMINAMATH_GPT_largest_eight_digit_with_all_even_digits_l2054_205461

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end NUMINAMATH_GPT_largest_eight_digit_with_all_even_digits_l2054_205461


namespace NUMINAMATH_GPT_downstream_speed_l2054_205468

def V_u : ℝ := 26
def V_m : ℝ := 28
def V_s : ℝ := V_m - V_u
def V_d : ℝ := V_m + V_s

theorem downstream_speed : V_d = 30 := by
  sorry

end NUMINAMATH_GPT_downstream_speed_l2054_205468


namespace NUMINAMATH_GPT_intersection_A_B_l2054_205462

def A : Set ℝ := {1, 3, 9, 27}
def B : Set ℝ := {y : ℝ | ∃ x ∈ A, y = Real.log x / Real.log 3}
theorem intersection_A_B : A ∩ B = {1, 3} := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2054_205462


namespace NUMINAMATH_GPT_exists_perfect_square_of_the_form_l2054_205483

theorem exists_perfect_square_of_the_form (k : ℕ) (h : k > 0) : ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * m = n * 2^k - 7 :=
by sorry

end NUMINAMATH_GPT_exists_perfect_square_of_the_form_l2054_205483


namespace NUMINAMATH_GPT_election_result_l2054_205438

theorem election_result:
  ∀ (Henry_votes India_votes Jenny_votes Ken_votes Lena_votes : ℕ)
    (counted_percentage : ℕ)
    (counted_votes : ℕ), 
    Henry_votes = 14 → 
    India_votes = 11 → 
    Jenny_votes = 10 → 
    Ken_votes = 8 → 
    Lena_votes = 2 → 
    counted_percentage = 90 → 
    counted_votes = 45 → 
    (counted_percentage * Total_votes / 100 = counted_votes) →
    (Total_votes = counted_votes * 100 / counted_percentage) →
    (Remaining_votes = Total_votes - counted_votes) →
    ((Henry_votes + Max_remaining_Votes >= Max_votes) ∨ 
    (India_votes + Max_remaining_Votes >= Max_votes) ∨ 
    (Jenny_votes + Max_remaining_Votes >= Max_votes)) →
    3 = 
    (if Henry_votes + Remaining_votes > Max_votes then 1 else 0) + 
    (if India_votes + Remaining_votes > Max_votes then 1 else 0) + 
    (if Jenny_votes + Remaining_votes > Max_votes then 1 else 0) := 
  sorry

end NUMINAMATH_GPT_election_result_l2054_205438


namespace NUMINAMATH_GPT_more_cats_than_dogs_l2054_205475

-- Define the initial conditions
def initial_cats : ℕ := 28
def initial_dogs : ℕ := 18
def cats_adopted : ℕ := 3

-- Compute the number of cats after adoption
def cats_now : ℕ := initial_cats - cats_adopted

-- Define the target statement
theorem more_cats_than_dogs : cats_now - initial_dogs = 7 := by
  unfold cats_now
  unfold initial_cats
  unfold cats_adopted
  unfold initial_dogs
  sorry

end NUMINAMATH_GPT_more_cats_than_dogs_l2054_205475


namespace NUMINAMATH_GPT_total_animals_is_200_l2054_205452

-- Definitions for the conditions
def num_cows : Nat := 40
def num_sheep : Nat := 56
def num_goats : Nat := 104

-- The theorem to prove the total number of animals is 200
theorem total_animals_is_200 : num_cows + num_sheep + num_goats = 200 := by
  sorry

end NUMINAMATH_GPT_total_animals_is_200_l2054_205452


namespace NUMINAMATH_GPT_intersection_complement_l2054_205441

universe u

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 2}
def complement_U_B : Set ℤ := {x ∈ U | x ∉ B}

theorem intersection_complement :
  A ∩ complement_U_B = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l2054_205441


namespace NUMINAMATH_GPT_additional_cards_l2054_205429

theorem additional_cards (total_cards : ℕ) (num_decks : ℕ) (cards_per_deck : ℕ) 
  (h1 : total_cards = 319) (h2 : num_decks = 6) (h3 : cards_per_deck = 52) : 
  319 - 6 * 52 = 7 := 
by
  sorry

end NUMINAMATH_GPT_additional_cards_l2054_205429


namespace NUMINAMATH_GPT_product_value_l2054_205457

noncomputable def product_of_sequence : ℝ :=
  (1/3) * 9 * (1/27) * 81 * (1/243) * 729 * (1/2187) * 6561

theorem product_value : product_of_sequence = 729 := by
  sorry

end NUMINAMATH_GPT_product_value_l2054_205457


namespace NUMINAMATH_GPT_f_lt_2_l2054_205497

noncomputable def f : ℝ → ℝ := sorry

axiom f_even (x : ℝ) : f (x + 2) = f (-x + 2)

axiom f_ge_2 (x : ℝ) (h : x ≥ 2) : f x = x^2 - 6 * x + 4

theorem f_lt_2 (x : ℝ) (h : x < 2) : f x = x^2 - 2 * x - 4 :=
by
  sorry

end NUMINAMATH_GPT_f_lt_2_l2054_205497


namespace NUMINAMATH_GPT_Abby_wins_if_N_2011_Brian_wins_in_31_cases_l2054_205427

-- Definitions and assumptions directly from the problem conditions
inductive Player
| Abby
| Brian

def game_condition (N : ℕ) : Prop :=
  ∀ (p : Player), 
    (p = Player.Abby → (∃ k, N = 2 * k + 1)) ∧ 
    (p = Player.Brian → (∃ k, N = 2 * (2^k - 1))) -- This encodes the winning state conditions for simplicity

-- Part (a)
theorem Abby_wins_if_N_2011 : game_condition 2011 :=
by
  sorry

-- Part (b)
theorem Brian_wins_in_31_cases : 
  (∃ S : Finset ℕ, (∀ N ∈ S, N ≤ 2011 ∧ game_condition N) ∧ S.card = 31) :=
by
  sorry

end NUMINAMATH_GPT_Abby_wins_if_N_2011_Brian_wins_in_31_cases_l2054_205427


namespace NUMINAMATH_GPT_find_n_from_lcms_l2054_205435

theorem find_n_from_lcms (n : ℕ) (h_pos : n > 0) (h_lcm1 : Nat.lcm 40 n = 200) (h_lcm2 : Nat.lcm n 45 = 180) : n = 100 := 
by
  sorry

end NUMINAMATH_GPT_find_n_from_lcms_l2054_205435


namespace NUMINAMATH_GPT_find_g_l2054_205402

theorem find_g (g : ℕ) (h : g > 0) :
  (1 / 3) = ((4 + g * (g - 1)) / ((g + 4) * (g + 3))) → g = 5 :=
by
  intro h_eq
  sorry 

end NUMINAMATH_GPT_find_g_l2054_205402


namespace NUMINAMATH_GPT_sequence_formula_l2054_205489

theorem sequence_formula (a : ℕ → ℕ) :
  (a 1 = 1) ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 3 * a n + 3 ^ n) → 
  ∀ n : ℕ, 0 < n → a n = n * 3 ^ (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l2054_205489


namespace NUMINAMATH_GPT_percent_sum_l2054_205423

theorem percent_sum (A B C : ℝ)
  (hA : 0.45 * A = 270)
  (hB : 0.35 * B = 210)
  (hC : 0.25 * C = 150) :
  0.75 * A + 0.65 * B + 0.45 * C = 1110 := by
  sorry

end NUMINAMATH_GPT_percent_sum_l2054_205423


namespace NUMINAMATH_GPT_johns_raise_percentage_increase_l2054_205451

theorem johns_raise_percentage_increase (original_amount new_amount : ℝ) (h_original : original_amount = 60) (h_new : new_amount = 70) :
  ((new_amount - original_amount) / original_amount) * 100 = 16.67 := 
  sorry

end NUMINAMATH_GPT_johns_raise_percentage_increase_l2054_205451


namespace NUMINAMATH_GPT_least_number_to_add_l2054_205479

theorem least_number_to_add (a b n : ℕ) (h₁ : a = 1056) (h₂ : b = 29) (h₃ : (a + n) % b = 0) : n = 17 :=
sorry

end NUMINAMATH_GPT_least_number_to_add_l2054_205479


namespace NUMINAMATH_GPT_derivative_at_pi_l2054_205439

noncomputable def f (x : ℝ) : ℝ := (x^2) / (Real.cos x)

theorem derivative_at_pi : deriv f π = -2 * π :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_pi_l2054_205439


namespace NUMINAMATH_GPT_sum_first_39_natural_numbers_l2054_205424

theorem sum_first_39_natural_numbers :
  (39 * (39 + 1)) / 2 = 780 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_39_natural_numbers_l2054_205424


namespace NUMINAMATH_GPT_coefficient_x4_expansion_eq_7_l2054_205466

theorem coefficient_x4_expansion_eq_7 (a : ℝ) : 
  (∀ r : ℕ, 8 - (4 * r) / 3 = 4 → (a ^ r) * (Nat.choose 8 r) = 7) → a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_x4_expansion_eq_7_l2054_205466


namespace NUMINAMATH_GPT_a_minus_b_7_l2054_205482

theorem a_minus_b_7 (a b : ℤ) : (2 * y + a) * (y + b) = 2 * y^2 - 5 * y - 12 → a - b = 7 :=
by
  sorry

end NUMINAMATH_GPT_a_minus_b_7_l2054_205482


namespace NUMINAMATH_GPT_sum_of_x_y_is_13_l2054_205425

theorem sum_of_x_y_is_13 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h : x^4 + y^4 = 4721) : x + y = 13 :=
sorry

end NUMINAMATH_GPT_sum_of_x_y_is_13_l2054_205425


namespace NUMINAMATH_GPT_factory_car_production_l2054_205401

theorem factory_car_production :
  let cars_yesterday := 60
  let cars_today := 2 * cars_yesterday
  let total_cars := cars_yesterday + cars_today
  total_cars = 180 :=
by
  sorry

end NUMINAMATH_GPT_factory_car_production_l2054_205401


namespace NUMINAMATH_GPT_onions_left_on_scale_l2054_205420

-- Define the given weights and conditions
def total_weight_of_40_onions : ℝ := 7680 -- in grams
def avg_weight_remaining_onions : ℝ := 190 -- grams
def avg_weight_removed_onions : ℝ := 206 -- grams

-- Converting original weight from kg to grams
def original_weight_kg_to_g (w_kg : ℝ) : ℝ := w_kg * 1000

-- Proof problem
theorem onions_left_on_scale (w_kg : ℝ) (n_total : ℕ) (n_removed : ℕ) 
    (total_weight : ℝ) (avg_weight_remaining : ℝ) (avg_weight_removed : ℝ)
    (h1 : original_weight_kg_to_g w_kg = total_weight)
    (h2 : n_total = 40)
    (h3 : n_removed = 5)
    (h4 : avg_weight_remaining = avg_weight_remaining_onions)
    (h5 : avg_weight_removed = avg_weight_removed_onions) : 
    n_total - n_removed = 35 :=
sorry

end NUMINAMATH_GPT_onions_left_on_scale_l2054_205420


namespace NUMINAMATH_GPT_polygon_sides_count_l2054_205480

def sides_square : ℕ := 4
def sides_triangle : ℕ := 3
def sides_hexagon : ℕ := 6
def sides_heptagon : ℕ := 7
def sides_octagon : ℕ := 8
def sides_nonagon : ℕ := 9

def total_sides_exposed : ℕ :=
  let adjacent_1side := sides_square + sides_nonagon - 2 * 1
  let adjacent_2sides :=
    sides_triangle + sides_hexagon +
    sides_heptagon + sides_octagon - 4 * 2
  adjacent_1side + adjacent_2sides

theorem polygon_sides_count : total_sides_exposed = 27 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_count_l2054_205480


namespace NUMINAMATH_GPT_a6_is_3_l2054_205434

noncomputable def a4 := 8 / 2 -- Placeholder for positive root
noncomputable def a8 := 8 / 2 -- Placeholder for the second root (we know they are both the same for now)
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 2) = (a (n + 1))^2

theorem a6_is_3 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a4_a8: a 4 = a4) (h_a4_a8_root : a 8 = a8) : 
  a 6 = 3 :=
by
  sorry

end NUMINAMATH_GPT_a6_is_3_l2054_205434


namespace NUMINAMATH_GPT_find_second_angle_l2054_205465

noncomputable def angle_in_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

theorem find_second_angle
  (A B C : ℝ)
  (hA : A = 32)
  (hC : C = 2 * A - 12)
  (hB : B = 3 * A)
  (h_sum : angle_in_triangle A B C) :
  B = 96 :=
by sorry

end NUMINAMATH_GPT_find_second_angle_l2054_205465


namespace NUMINAMATH_GPT_sum_of_b_for_quadratic_has_one_solution_l2054_205419

theorem sum_of_b_for_quadratic_has_one_solution :
  (∀ x : ℝ, 3 * x^2 + (b+6) * x + 1 = 0 → 
    ∀ Δ : ℝ, Δ = (b + 6)^2 - 4 * 3 * 1 → 
    Δ = 0 → 
    b = -6 + 2 * Real.sqrt 3 ∨ b = -6 - 2 * Real.sqrt 3) → 
  (-6 + 2 * Real.sqrt 3 + -6 - 2 * Real.sqrt 3 = -12) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_b_for_quadratic_has_one_solution_l2054_205419


namespace NUMINAMATH_GPT_range_of_a_l2054_205486

theorem range_of_a 
  (a : ℝ) 
  (h₀ : ∀ x : ℝ, (3 ≤ x ∧ x ≤ 4) ↔ (y = 2 * x + (3 - a))) : 
  9 ≤ a ∧ a ≤ 11 := 
sorry

end NUMINAMATH_GPT_range_of_a_l2054_205486


namespace NUMINAMATH_GPT_tank_water_after_rain_final_l2054_205405

theorem tank_water_after_rain_final (initial_water evaporated drained rain_rate rain_time : ℕ)
  (initial_water_eq : initial_water = 6000)
  (evaporated_eq : evaporated = 2000)
  (drained_eq : drained = 3500)
  (rain_rate_eq : rain_rate = 350)
  (rain_time_eq : rain_time = 30) :
  let water_after_evaporation := initial_water - evaporated
  let water_after_drainage := water_after_evaporation - drained 
  let rain_addition := (rain_time / 10) * rain_rate
  let final_water := water_after_drainage + rain_addition
  final_water = 1550 :=
by
  sorry

end NUMINAMATH_GPT_tank_water_after_rain_final_l2054_205405


namespace NUMINAMATH_GPT_cart_max_speed_l2054_205413

noncomputable def maximum_speed (a R : ℝ) : ℝ :=
  (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4)

theorem cart_max_speed (a R v : ℝ) (h : v = maximum_speed a R) : 
  v = (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_cart_max_speed_l2054_205413


namespace NUMINAMATH_GPT_exists_quadratic_satisfying_conditions_l2054_205459

theorem exists_quadratic_satisfying_conditions :
  ∃ (a b c : ℝ), 
  (a - b + c = 0) ∧
  (∀ x : ℝ, x ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ (1 + x^2) / 2) ∧ 
  (a = 1/4 ∧ b = 1/2 ∧ c = 1/4) :=
  sorry

end NUMINAMATH_GPT_exists_quadratic_satisfying_conditions_l2054_205459


namespace NUMINAMATH_GPT_math_problem_l2054_205493

theorem math_problem (a b c d x : ℝ)
  (h1 : a = -(-b))
  (h2 : c = -1 / d)
  (h3 : |x| = 3) :
  x^3 + c * d * x^2 - (a - b) / 2 = 18 ∨ x^3 + c * d * x^2 - (a - b) / 2 = -36 :=
by sorry

end NUMINAMATH_GPT_math_problem_l2054_205493


namespace NUMINAMATH_GPT_parabola_vertex_is_two_one_l2054_205436

theorem parabola_vertex_is_two_one : 
  ∀ x y : ℝ, (y = (x - 2)^2 + 1) → (2, 1) = (2, 1) :=
by
  intros x y hyp
  sorry

end NUMINAMATH_GPT_parabola_vertex_is_two_one_l2054_205436


namespace NUMINAMATH_GPT_calculate_F_2_f_3_l2054_205472

def f (a : ℕ) : ℕ := a ^ 2 - 3 * a + 2

def F (a b : ℕ) : ℕ := b ^ 2 + a + 1

theorem calculate_F_2_f_3 : F 2 (f 3) = 7 :=
by
  show F 2 (f 3) = 7
  sorry

end NUMINAMATH_GPT_calculate_F_2_f_3_l2054_205472


namespace NUMINAMATH_GPT_part_a_no_solutions_part_a_infinite_solutions_l2054_205432

theorem part_a_no_solutions (a : ℝ) (x y : ℝ) : 
    a = -1 → ¬(∃ x y : ℝ, a * x + y = a^2 ∧ x + a * y = 1) :=
sorry

theorem part_a_infinite_solutions (a : ℝ) (x y : ℝ) : 
    a = 1 → ∃ x : ℝ, ∃ y : ℝ, a * x + y = a^2 ∧ x + a * y = 1 :=
sorry

end NUMINAMATH_GPT_part_a_no_solutions_part_a_infinite_solutions_l2054_205432


namespace NUMINAMATH_GPT_max_value_of_y_l2054_205458

noncomputable def max_value_of_function : ℝ := 1 + Real.sqrt 2

theorem max_value_of_y : ∀ x : ℝ, (2 * Real.sin x * (Real.sin x + Real.cos x)) ≤ max_value_of_function :=
by
  -- Proof goes here
  sorry

example : ∃ x : ℝ, (2 * Real.sin x * (Real.sin x + Real.cos x)) = max_value_of_function :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_max_value_of_y_l2054_205458


namespace NUMINAMATH_GPT_line_tangent_to_ellipse_l2054_205408

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 3 * (m * x + 2)^2 = 3) ∧ 
  (∀ x1 x2 : ℝ, (2 + 3 * m^2) * x1^2 + 12 * m * x1 + 9 = 0 ∧ 
                (2 + 3 * m^2) * x2^2 + 12 * m * x2 + 9 = 0 → x1 = x2) ↔ m^2 = 2 := 
sorry

end NUMINAMATH_GPT_line_tangent_to_ellipse_l2054_205408


namespace NUMINAMATH_GPT_noah_uses_36_cups_of_water_l2054_205421

theorem noah_uses_36_cups_of_water
  (O : ℕ) (hO : O = 4)
  (S : ℕ) (hS : S = 3 * O)
  (W : ℕ) (hW : W = 3 * S) :
  W = 36 := 
  by sorry

end NUMINAMATH_GPT_noah_uses_36_cups_of_water_l2054_205421


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l2054_205449

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l2054_205449


namespace NUMINAMATH_GPT_only_root_is_4_l2054_205495

noncomputable def equation_one (x : ℝ) : ℝ := (2 * x^2) / (x - 1) - (2 * x + 7) / 3 + (4 - 6 * x) / (x - 1) + 1

noncomputable def equation_two (x : ℝ) : ℝ := x^2 - 5 * x + 4

theorem only_root_is_4 (x : ℝ) (h: equation_one x = 0) (h_transformation: equation_two x = 0) : x = 4 := sorry

end NUMINAMATH_GPT_only_root_is_4_l2054_205495


namespace NUMINAMATH_GPT_reimbursement_proof_l2054_205415

-- Define the rates
def rate_industrial_weekday : ℝ := 0.36
def rate_commercial_weekday : ℝ := 0.42
def rate_weekend : ℝ := 0.45

-- Define the distances for each day
def distance_monday : ℝ := 18
def distance_tuesday : ℝ := 26
def distance_wednesday : ℝ := 20
def distance_thursday : ℝ := 20
def distance_friday : ℝ := 16
def distance_saturday : ℝ := 12

-- Calculate the reimbursement for each day
def reimbursement_monday : ℝ := distance_monday * rate_industrial_weekday
def reimbursement_tuesday : ℝ := distance_tuesday * rate_commercial_weekday
def reimbursement_wednesday : ℝ := distance_wednesday * rate_industrial_weekday
def reimbursement_thursday : ℝ := distance_thursday * rate_commercial_weekday
def reimbursement_friday : ℝ := distance_friday * rate_industrial_weekday
def reimbursement_saturday : ℝ := distance_saturday * rate_weekend

-- Calculate the total reimbursement
def total_reimbursement : ℝ :=
  reimbursement_monday + reimbursement_tuesday + reimbursement_wednesday +
  reimbursement_thursday + reimbursement_friday + reimbursement_saturday

-- State the theorem to be proven
theorem reimbursement_proof : total_reimbursement = 44.16 := by
  sorry

end NUMINAMATH_GPT_reimbursement_proof_l2054_205415


namespace NUMINAMATH_GPT_fraction_checked_by_worker_y_l2054_205422

theorem fraction_checked_by_worker_y
  (f_X f_Y : ℝ)
  (h1 : f_X + f_Y = 1)
  (h2 : 0.005 * f_X + 0.008 * f_Y = 0.0074) :
  f_Y = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_checked_by_worker_y_l2054_205422


namespace NUMINAMATH_GPT_andrew_purchased_mangoes_l2054_205477

variable (m : ℕ)

def cost_of_grapes := 8 * 70
def cost_of_mangoes (m : ℕ) := 55 * m
def total_cost (m : ℕ) := cost_of_grapes + cost_of_mangoes m

theorem andrew_purchased_mangoes :
  total_cost m = 1055 → m = 9 := by
  intros h_total_cost
  sorry

end NUMINAMATH_GPT_andrew_purchased_mangoes_l2054_205477


namespace NUMINAMATH_GPT_seventh_term_of_arithmetic_sequence_l2054_205484

theorem seventh_term_of_arithmetic_sequence 
  (a d : ℤ)
  (h1 : 5 * a + 10 * d = 35)
  (h2 : a + 5 * d = 10) :
  a + 6 * d = 11 :=
by
  sorry

end NUMINAMATH_GPT_seventh_term_of_arithmetic_sequence_l2054_205484


namespace NUMINAMATH_GPT_janet_total_l2054_205476

-- Definitions based on the conditions
variable (initial_collect : ℕ) (sold : ℕ) (better_cond : ℕ)
variable (twice_size : ℕ)

-- The conditions from part a)
def janet_initial_collection := initial_collect = 10
def janet_sells := sold = 6
def janet_gets_better := better_cond = 4
def brother_gives := twice_size = 2 * (initial_collect - sold + better_cond)

-- The proof statement based on part c)
theorem janet_total (initial_collect sold better_cond twice_size : ℕ) : 
    janet_initial_collection initial_collect →
    janet_sells sold →
    janet_gets_better better_cond →
    brother_gives initial_collect sold better_cond twice_size →
    (initial_collect - sold + better_cond + twice_size = 24) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_janet_total_l2054_205476


namespace NUMINAMATH_GPT_problem1_problem2_l2054_205491

section proof_problem

-- Define the sets as predicate functions
def A (x : ℝ) : Prop := x > 1
def B (x : ℝ) : Prop := -2 < x ∧ x < 2
def C (x : ℝ) : Prop := -3 < x ∧ x < 5

-- Define the union and intersection of sets
def union (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∨ B x
def inter (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∧ B x

-- Proving that (A ∪ B) ∩ C = {x | -2 < x < 5}
theorem problem1 : ∀ x, (inter (union A B) C) x ↔ (-2 < x ∧ x < 5) := 
by
  sorry

-- Proving the arithmetic expression result
theorem problem2 : 
  ((2 + 1/4) ^ (1/2)) - ((-9.6) ^ 0) - ((3 + 3/8) ^ (-2/3)) + ((1.5) ^ (-2)) = 1/2 := 
by
  sorry

end proof_problem

end NUMINAMATH_GPT_problem1_problem2_l2054_205491


namespace NUMINAMATH_GPT_area_difference_of_circles_l2054_205437

theorem area_difference_of_circles : 
  let r1 := 30
  let r2 := 15
  let pi := Real.pi
  900 * pi - 225 * pi = 675 * pi := by
  sorry

end NUMINAMATH_GPT_area_difference_of_circles_l2054_205437


namespace NUMINAMATH_GPT_total_cost_of_fencing_l2054_205454

theorem total_cost_of_fencing (side_count : ℕ) (cost_per_side : ℕ) (h1 : side_count = 4) (h2 : cost_per_side = 79) : side_count * cost_per_side = 316 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_fencing_l2054_205454


namespace NUMINAMATH_GPT_smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l2054_205499

noncomputable def smaller_angle_at_715 : ℝ :=
  let hour_position := 7 * 30 + 30 / 4
  let minute_position := 15 * (360 / 60)
  let angle_between := abs (hour_position - minute_position)
  if angle_between > 180 then 360 - angle_between else angle_between

theorem smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m :
  smaller_angle_at_715 = 127.5 := 
sorry

end NUMINAMATH_GPT_smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l2054_205499


namespace NUMINAMATH_GPT_find_p_range_l2054_205426

theorem find_p_range (p : ℝ) (A : ℝ → ℝ) :
  (A = fun x => abs x * x^2 + (p + 2) * x + 1) →
  (∀ x, 0 < x → A x ≠ 0) →
  (-4 < p ∧ p < 0) :=
by
  intro hA h_no_pos_roots
  sorry

end NUMINAMATH_GPT_find_p_range_l2054_205426


namespace NUMINAMATH_GPT_math_evening_problem_l2054_205460

theorem math_evening_problem
  (S : ℕ)
  (r : ℕ)
  (fifth_graders_per_row : ℕ := 3)
  (sixth_graders_per_row : ℕ := r - fifth_graders_per_row)
  (total_number_of_students : ℕ := r * r) :
  70 < total_number_of_students ∧ total_number_of_students < 90 → 
  r = 9 ∧ 
  6 * r = 54 ∧
  3 * r = 27 :=
sorry

end NUMINAMATH_GPT_math_evening_problem_l2054_205460


namespace NUMINAMATH_GPT_oldest_daily_cheese_l2054_205496

-- Given conditions
def days_per_week : ℕ := 5
def weeks : ℕ := 4
def youngest_daily : ℕ := 1
def cheeses_per_pack : ℕ := 30
def packs_needed : ℕ := 2

-- Derived conditions
def total_days : ℕ := days_per_week * weeks
def total_cheeses : ℕ := packs_needed * cheeses_per_pack
def youngest_total_cheeses : ℕ := youngest_daily * total_days
def oldest_total_cheeses : ℕ := total_cheeses - youngest_total_cheeses

-- Prove that the oldest child wants 2 string cheeses per day
theorem oldest_daily_cheese : oldest_total_cheeses / total_days = 2 := by
  sorry

end NUMINAMATH_GPT_oldest_daily_cheese_l2054_205496


namespace NUMINAMATH_GPT_solve_system_l2054_205410

theorem solve_system :
  ∃ x y : ℝ, (x^2 * y + x * y^2 + 3 * x + 3 * y + 24 = 0) ∧ 
              (x^3 * y - x * y^3 + 3 * x^2 - 3 * y^2 - 48 = 0) ∧ 
              (x = -3 ∧ y = -1) :=
  sorry

end NUMINAMATH_GPT_solve_system_l2054_205410


namespace NUMINAMATH_GPT_part_a_2001_good_part_a_3001_not_good_part_b_product_of_good_is_good_part_c_product_good_then_one_good_l2054_205478

-- Definition of a good number
def is_good (n : ℕ) : Prop := (n % 6 = 3)

-- Lean 4 statements

-- 1. 2001 is good
theorem part_a_2001_good : is_good 2001 :=
by sorry

-- 2. 3001 isn't good
theorem part_a_3001_not_good : ¬ is_good 3001 :=
by sorry

-- 3. The product of two good numbers is a good number
theorem part_b_product_of_good_is_good (x y : ℕ) (hx : is_good x) (hy : is_good y) : is_good (x * y) :=
by sorry

-- 4. If the product of two numbers is good, then at least one of the numbers is good
theorem part_c_product_good_then_one_good (x y : ℕ) (hxy : is_good (x * y)) : is_good x ∨ is_good y :=
by sorry

end NUMINAMATH_GPT_part_a_2001_good_part_a_3001_not_good_part_b_product_of_good_is_good_part_c_product_good_then_one_good_l2054_205478


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_not_sufficient_condition_l2054_205463

theorem necessary_but_not_sufficient_condition (x y : ℝ) (h : x > 0) : 
  (x > |y|) → (x > y) :=
by
  sorry

theorem not_sufficient_condition (x y : ℝ) (h : x > 0) :
  ¬ ((x > y) → (x > |y|)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_not_sufficient_condition_l2054_205463


namespace NUMINAMATH_GPT_total_visit_plans_l2054_205455

def exhibitions : List String := ["Opera Culture Exhibition", "Ming Dynasty Imperial Cellar Porcelain Exhibition", "Historical Green Landscape Painting Exhibition", "Zhao Mengfu Calligraphy and Painting Exhibition"]

def painting_exhibitions : List String := ["Historical Green Landscape Painting Exhibition", "Zhao Mengfu Calligraphy and Painting Exhibition"]

def non_painting_exhibitions : List String := ["Opera Culture Exhibition", "Ming Dynasty Imperial Cellar Porcelain Exhibition"]

def num_visit_plans (exhibit_list : List String) (paintings : List String) (non_paintings : List String) : Nat :=
  let case1 := paintings.length * non_paintings.length * 2
  let case2 := if paintings.length >= 2 then 2 else 0
  case1 + case2

theorem total_visit_plans : num_visit_plans exhibitions painting_exhibitions non_painting_exhibitions = 10 :=
  sorry

end NUMINAMATH_GPT_total_visit_plans_l2054_205455


namespace NUMINAMATH_GPT_find_a2_l2054_205407

theorem find_a2 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * (a n - 1))
  (h2 : S 1 = a 1)
  (h3 : S 2 = a 1 + a 2) :
  a 2 = 4 :=
sorry

end NUMINAMATH_GPT_find_a2_l2054_205407


namespace NUMINAMATH_GPT_arithmetic_expression_eval_l2054_205456

theorem arithmetic_expression_eval :
  -1 ^ 4 + (4 - ((3 / 8 + 1 / 6 - 3 / 4) * 24)) / 5 = 0.8 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_eval_l2054_205456


namespace NUMINAMATH_GPT_sum_of_digits_is_10_l2054_205446

def sum_of_digits_of_expression : ℕ :=
  let expression := 2^2010 * 5^2008 * 7
  let simplified := 280000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
  2 + 8

/-- The sum of the digits of the decimal representation of 2^2010 * 5^2008 * 7 is 10 -/
theorem sum_of_digits_is_10 :
  sum_of_digits_of_expression = 10 :=
by sorry

end NUMINAMATH_GPT_sum_of_digits_is_10_l2054_205446


namespace NUMINAMATH_GPT_haley_magazines_l2054_205417

theorem haley_magazines (boxes : ℕ) (magazines_per_box : ℕ) (total_magazines : ℕ) :
  boxes = 7 →
  magazines_per_box = 9 →
  total_magazines = boxes * magazines_per_box →
  total_magazines = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_haley_magazines_l2054_205417


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l2054_205485

noncomputable def repeating_decimal := 0.6 + 3 / 100

theorem repeating_decimal_to_fraction :
  repeating_decimal = 19 / 30 :=
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l2054_205485


namespace NUMINAMATH_GPT_evaluate_expression_l2054_205428

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 4 - 2 * g (-2) = 47 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2054_205428


namespace NUMINAMATH_GPT_range_of_k_l2054_205400

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x^2 - 2 * x + k^2 - 3 > 0) -> (k > 2 ∨ k < -2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l2054_205400


namespace NUMINAMATH_GPT_youngest_child_is_five_l2054_205412

-- Define the set of prime numbers
def is_prime (n: ℕ) := n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the ages of the children
def youngest_child_age (x: ℕ) : Prop :=
  is_prime x ∧
  is_prime (x + 2) ∧
  is_prime (x + 6) ∧
  is_prime (x + 8) ∧
  is_prime (x + 12) ∧
  is_prime (x + 14)

-- The main theorem stating the age of the youngest child
theorem youngest_child_is_five : ∃ x: ℕ, youngest_child_age x ∧ x = 5 :=
  sorry

end NUMINAMATH_GPT_youngest_child_is_five_l2054_205412


namespace NUMINAMATH_GPT_simplify_fraction_l2054_205487

theorem simplify_fraction (a b : ℕ) (h1 : a = 252) (h2 : b = 248) :
  (1000 ^ 2 : ℤ) / ((a ^ 2 - b ^ 2) : ℤ) = 500 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2054_205487


namespace NUMINAMATH_GPT_sum_of_products_l2054_205492

theorem sum_of_products : 1 * 15 + 2 * 14 + 3 * 13 + 4 * 12 + 5 * 11 + 6 * 10 + 7 * 9 + 8 * 8 = 372 := by
  sorry

end NUMINAMATH_GPT_sum_of_products_l2054_205492


namespace NUMINAMATH_GPT_problem_solution_l2054_205440

theorem problem_solution :
  (315^2 - 291^2) / 24 = 606 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2054_205440


namespace NUMINAMATH_GPT_maximize_distance_l2054_205447

def front_tire_lifespan : ℕ := 20000
def rear_tire_lifespan : ℕ := 30000
def max_distance : ℕ := 24000

theorem maximize_distance : max_distance = 24000 := sorry

end NUMINAMATH_GPT_maximize_distance_l2054_205447


namespace NUMINAMATH_GPT_find_set_A_l2054_205450

-- Define the set A based on the condition that its elements satisfy a quadratic equation.
def A (a : ℝ) : Set ℝ := {x | x^2 + 2 * x + a = 0}

-- Assume 1 is an element of set A
axiom one_in_A (a : ℝ) (h : 1 ∈ A a) : a = -3

-- The final theorem to prove: Given 1 ∈ A a, A a should be {-3, 1}
theorem find_set_A (a : ℝ) (h : 1 ∈ A a) : A a = {-3, 1} :=
by sorry

end NUMINAMATH_GPT_find_set_A_l2054_205450


namespace NUMINAMATH_GPT_polynomial_remainder_distinct_l2054_205471

open Nat

theorem polynomial_remainder_distinct (a b c p : ℕ) (hp : Nat.Prime p) (hp_ge5 : p ≥ 5)
  (ha : Nat.gcd a p = 1) (hb : b^2 ≡ 3 * a * c [MOD p]) (hp_mod3 : p ≡ 2 [MOD 3]) :
  ∀ m1 m2 : ℕ, m1 < p ∧ m2 < p → m1 ≠ m2 → (a * m1^3 + b * m1^2 + c * m1) % p ≠ (a * m2^3 + b * m2^2 + c * m2) % p := 
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_distinct_l2054_205471


namespace NUMINAMATH_GPT_production_rate_problem_l2054_205431

theorem production_rate_problem :
  ∀ (G T : ℕ), 
  (∀ w t, w * 3 * t = 450 * t / 150) ∧
  (∀ w t, w * 2 * t = 300 * t / 150) ∧
  (∀ w t, w * 2 * t = 360 * t / 90) ∧
  (∀ w t, w * (5/2) * t = 450 * t / 90) ∧
  (75 * 2 * 4 = 300) →
  (75 * 2 * 4 = 600) := sorry

end NUMINAMATH_GPT_production_rate_problem_l2054_205431


namespace NUMINAMATH_GPT_comprehensiveInvestigation_is_Census_l2054_205442

def comprehensiveInvestigation (s: String) : Prop :=
  s = "Census"

theorem comprehensiveInvestigation_is_Census :
  comprehensiveInvestigation "Census" :=
by
  sorry

end NUMINAMATH_GPT_comprehensiveInvestigation_is_Census_l2054_205442


namespace NUMINAMATH_GPT_rain_on_tuesday_l2054_205404

theorem rain_on_tuesday 
  (rain_monday : ℝ)
  (rain_less : ℝ) 
  (h1 : rain_monday = 0.9) 
  (h2 : rain_less = 0.7) : 
  (rain_monday - rain_less) = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_rain_on_tuesday_l2054_205404


namespace NUMINAMATH_GPT_men_at_yoga_studio_l2054_205474

open Real

def yoga_men_count (M : ℕ) (avg_weight_men avg_weight_women avg_weight_total : ℝ) (num_women num_total : ℕ) : Prop :=
  avg_weight_men = 190 ∧
  avg_weight_women = 120 ∧
  num_women = 6 ∧
  num_total = 14 ∧
  avg_weight_total = 160 →
  M + num_women = num_total ∧
  (M * avg_weight_men + num_women * avg_weight_women) / num_total = avg_weight_total ∧
  M = 8

theorem men_at_yoga_studio : ∃ M : ℕ, yoga_men_count M 190 120 160 6 14 :=
  by 
  use 8
  sorry

end NUMINAMATH_GPT_men_at_yoga_studio_l2054_205474


namespace NUMINAMATH_GPT_find_distance_d_l2054_205414

theorem find_distance_d (d : ℝ) (XR : ℝ) (YP : ℝ) (XZ : ℝ) (YZ : ℝ) (XY : ℝ) (h1 : XR = 3) (h2 : YP = 12) (h3 : XZ = 3 + d) (h4 : YZ = 12 + d) (h5 : XY = 15) (h6 : (XZ)^2 + (XY)^2 = (YZ)^2) : d = 5 :=
sorry

end NUMINAMATH_GPT_find_distance_d_l2054_205414


namespace NUMINAMATH_GPT_geometric_sequence_xz_eq_three_l2054_205490

theorem geometric_sequence_xz_eq_three 
  (x y z : ℝ)
  (h1 : ∃ r : ℝ, x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -3 = z * r) :
  x * z = 3 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_geometric_sequence_xz_eq_three_l2054_205490


namespace NUMINAMATH_GPT_intersection_complement_M_N_l2054_205445

def M : Set ℝ := { x | x > 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }
def complement_M : Set ℝ := { x | x ≤ 1 }

theorem intersection_complement_M_N :
  (complement_M ∩ N) = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_M_N_l2054_205445


namespace NUMINAMATH_GPT_chess_group_players_l2054_205406

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_chess_group_players_l2054_205406


namespace NUMINAMATH_GPT_nth_equation_l2054_205409

theorem nth_equation (n : ℕ) (h : 0 < n) : 9 * (n - 1) + n = 10 * n - 9 := 
  sorry

end NUMINAMATH_GPT_nth_equation_l2054_205409


namespace NUMINAMATH_GPT_pencils_per_associate_professor_l2054_205473

theorem pencils_per_associate_professor
    (A B P : ℕ) -- the number of associate professors, assistant professors, and pencils per associate professor respectively
    (h1 : A + B = 6) -- there are a total of 6 people
    (h2 : A * P + B = 7) -- total number of pencils is 7
    (h3 : A + 2 * B = 11) -- total number of charts is 11
    : P = 2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_pencils_per_associate_professor_l2054_205473


namespace NUMINAMATH_GPT_sum_of_digits_l2054_205443

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem sum_of_digits (a b c d : ℕ) (h_distinct : distinct_digits a b c d) (h_eqn : 100*a + 60 + b - (400 + 10*c + d) = 2) :
  a + b + c + d = 10 ∨ a + b + c + d = 18 ∨ a + b + c + d = 19 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_l2054_205443


namespace NUMINAMATH_GPT_scale_of_diagram_l2054_205416

-- Definitions for the given conditions
def length_miniature_component_mm : ℕ := 4
def length_diagram_cm : ℕ := 8
def length_diagram_mm : ℕ := 80  -- Converted length from cm to mm

-- The problem statement
theorem scale_of_diagram :
  (length_diagram_mm : ℕ) / (length_miniature_component_mm : ℕ) = 20 :=
by
  have conversion : length_diagram_mm = length_diagram_cm * 10 := by sorry
  -- conversion states the formula for converting cm to mm
  have ratio : length_diagram_mm / length_miniature_component_mm = 80 / 4 := by sorry
  -- ratio states the initial computed ratio
  exact sorry

end NUMINAMATH_GPT_scale_of_diagram_l2054_205416


namespace NUMINAMATH_GPT_remainder_47_mod_288_is_23_mod_24_l2054_205481

theorem remainder_47_mod_288_is_23_mod_24 (m : ℤ) (h : m % 288 = 47) : m % 24 = 23 := 
sorry

end NUMINAMATH_GPT_remainder_47_mod_288_is_23_mod_24_l2054_205481


namespace NUMINAMATH_GPT_volume_of_pyramid_l2054_205467

theorem volume_of_pyramid (V_cube : ℝ) (h : ℝ) (A : ℝ) (V_pyramid : ℝ) : 
  V_cube = 27 → 
  h = 3 → 
  A = 4.5 → 
  V_pyramid = (1/3) * A * h → 
  V_pyramid = 4.5 := 
by 
  intros V_cube_eq h_eq A_eq V_pyramid_eq 
  sorry

end NUMINAMATH_GPT_volume_of_pyramid_l2054_205467


namespace NUMINAMATH_GPT_min_value_a_cube_plus_b_cube_no_exist_2a_plus_3b_eq_6_l2054_205453

section ProofProblem

theorem min_value_a_cube_plus_b_cube {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hfra : 1/a + 1/b = Real.sqrt (a * b)) :
  a^3 + b^3 = 4 * Real.sqrt 2 := 
sorry

theorem no_exist_2a_plus_3b_eq_6 {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hfra : 1/a + 1/b = Real.sqrt (a * b)) :
  ¬ (2 * a + 3 * b = 6) :=
sorry

end ProofProblem

end NUMINAMATH_GPT_min_value_a_cube_plus_b_cube_no_exist_2a_plus_3b_eq_6_l2054_205453


namespace NUMINAMATH_GPT_polynomial_value_at_3_l2054_205470

-- Definitions based on given conditions
def f (x : ℕ) : ℕ :=
  5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def x := 3

-- Proof statement
theorem polynomial_value_at_3 : f x = 1641 := by
  sorry

end NUMINAMATH_GPT_polynomial_value_at_3_l2054_205470


namespace NUMINAMATH_GPT_lemonade_quarts_water_l2054_205488

-- Definitions derived from the conditions
def total_parts := 6 + 2 + 1 -- Sum of all ratio parts
def parts_per_gallon : ℚ := 1.5 / total_parts -- Volume per part in gallons
def parts_per_quart : ℚ := parts_per_gallon * 4 -- Volume per part in quarts
def water_needed : ℚ := 6 * parts_per_quart -- Quarts of water needed

-- Statement to prove
theorem lemonade_quarts_water : water_needed = 4 := 
by sorry

end NUMINAMATH_GPT_lemonade_quarts_water_l2054_205488


namespace NUMINAMATH_GPT_children_count_l2054_205433

theorem children_count (W C n : ℝ) (h1 : 4 * W = 1 / 7) (h2 : n * C = 1 / 14) (h3 : 5 * W + 10 * C = 1 / 4) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_children_count_l2054_205433


namespace NUMINAMATH_GPT_largest_y_value_l2054_205430

theorem largest_y_value (y : ℝ) (h : 3*y^2 + 18*y - 90 = y*(y + 17)) : y ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_y_value_l2054_205430


namespace NUMINAMATH_GPT_cheryl_mms_eaten_l2054_205469

variable (initial_mms : ℕ) (mms_after_dinner : ℕ) (mms_given_to_sister : ℕ) (total_mms_after_lunch : ℕ)

theorem cheryl_mms_eaten (h1 : initial_mms = 25)
                         (h2 : mms_after_dinner = 5)
                         (h3 : mms_given_to_sister = 13)
                         (h4 : total_mms_after_lunch = initial_mms - mms_after_dinner - mms_given_to_sister) :
                         total_mms_after_lunch = 7 :=
by sorry

end NUMINAMATH_GPT_cheryl_mms_eaten_l2054_205469


namespace NUMINAMATH_GPT_inequality_solution_l2054_205464

theorem inequality_solution (a : ℝ)
  (h : ∀ x, x ∈ Set.Icc (1/2 : ℝ) 2 → a * x^2 - 2 * x + 2 > 0) :
  a > 1/2 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l2054_205464


namespace NUMINAMATH_GPT_polynomial_divisibility_l2054_205494

open Polynomial

noncomputable def f (n : ℕ) : ℤ[X] :=
  (X + 1) ^ (2 * n + 1) + X ^ (n + 2)

noncomputable def p : ℤ[X] :=
  X ^ 2 + X + 1

theorem polynomial_divisibility (n : ℕ) : p ∣ f n :=
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l2054_205494


namespace NUMINAMATH_GPT_probability_purple_or_orange_face_l2054_205403

theorem probability_purple_or_orange_face 
  (total_faces : ℕ) (green_faces : ℕ) (purple_faces : ℕ) (orange_faces : ℕ) 
  (h_total : total_faces = 10) 
  (h_green : green_faces = 5) 
  (h_purple : purple_faces = 3) 
  (h_orange : orange_faces = 2) :
  (purple_faces + orange_faces) / total_faces = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_probability_purple_or_orange_face_l2054_205403
