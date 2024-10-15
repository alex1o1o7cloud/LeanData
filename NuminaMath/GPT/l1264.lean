import Mathlib

namespace NUMINAMATH_GPT_range_of_m_l1264_126441

variable {m x x1 x2 y1 y2 : ℝ}

noncomputable def linear_function (m x : ℝ) : ℝ := (m - 2) * x + (2 + m)

theorem range_of_m (h1 : x1 < x2) (h2 : y1 = linear_function m x1) (h3 : y2 = linear_function m x2) (h4 : y1 > y2) : m < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1264_126441


namespace NUMINAMATH_GPT_find_value_of_x2_plus_y2_l1264_126406

theorem find_value_of_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 + y^2 - 4 * x * y + 24 ≤ 10 * x - 1) : x^2 + y^2 = 125 := 
sorry

end NUMINAMATH_GPT_find_value_of_x2_plus_y2_l1264_126406


namespace NUMINAMATH_GPT_total_payment_correct_l1264_126472

def rate_per_kg_grapes := 68
def quantity_grapes := 7
def rate_per_kg_mangoes := 48
def quantity_mangoes := 9

def cost_grapes := rate_per_kg_grapes * quantity_grapes
def cost_mangoes := rate_per_kg_mangoes * quantity_mangoes

def total_amount_paid := cost_grapes + cost_mangoes

theorem total_payment_correct :
  total_amount_paid = 908 := by
  sorry

end NUMINAMATH_GPT_total_payment_correct_l1264_126472


namespace NUMINAMATH_GPT_henrietta_paint_needed_l1264_126477

theorem henrietta_paint_needed :
  let living_room_area := 600
  let num_bedrooms := 3
  let bedroom_area := 400
  let paint_coverage_per_gallon := 600
  let total_area := living_room_area + (num_bedrooms * bedroom_area)
  total_area / paint_coverage_per_gallon = 3 :=
by
  -- Proof should be completed here.
  sorry

end NUMINAMATH_GPT_henrietta_paint_needed_l1264_126477


namespace NUMINAMATH_GPT_cistern_length_l1264_126475

theorem cistern_length
  (L W D A : ℝ)
  (hW : W = 4)
  (hD : D = 1.25)
  (hA : A = 49)
  (hWetSurface : A = L * W + 2 * L * D) :
  L = 7.54 := by
  sorry

end NUMINAMATH_GPT_cistern_length_l1264_126475


namespace NUMINAMATH_GPT_cannot_cover_completely_with_dominoes_l1264_126414

theorem cannot_cover_completely_with_dominoes :
  ¬ (∃ f : Fin 5 × Fin 3 → Fin 5 × Fin 3, 
      (∀ p q, f p = f q → p = q) ∧ 
      (∀ p, ∃ q, f q = p) ∧ 
      (∀ p, (f p).1 = p.1 + 1 ∨ (f p).2 = p.2 + 1)) := 
sorry

end NUMINAMATH_GPT_cannot_cover_completely_with_dominoes_l1264_126414


namespace NUMINAMATH_GPT_joan_and_karl_sofas_l1264_126462

variable (J K : ℝ)

theorem joan_and_karl_sofas (hJ : J = 230) (hSum : J + K = 600) :
  2 * J - K = 90 :=
by
  sorry

end NUMINAMATH_GPT_joan_and_karl_sofas_l1264_126462


namespace NUMINAMATH_GPT_unique_fraction_increased_by_20_percent_l1264_126461

def relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem unique_fraction_increased_by_20_percent (x y : ℕ) (h1 : relatively_prime x y) (h2 : x > 0) (h3 : y > 0) :
  (∃! (x y : ℕ), relatively_prime x y ∧ (x > 0) ∧ (y > 0) ∧ (x + 2) * y = 6 * (y + 2) * x) :=
sorry

end NUMINAMATH_GPT_unique_fraction_increased_by_20_percent_l1264_126461


namespace NUMINAMATH_GPT_first_cat_blue_eyed_kittens_l1264_126457

variable (B : ℕ)
variable (C1 : 35 * (B + 17) = 100 * (B + 4))

theorem first_cat_blue_eyed_kittens : B = 3 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_first_cat_blue_eyed_kittens_l1264_126457


namespace NUMINAMATH_GPT_num_balls_total_l1264_126423

theorem num_balls_total (m : ℕ) (h1 : 6 < m) (h2 : (6 : ℝ) / (m : ℝ) = 0.3) : m = 20 :=
by
  sorry

end NUMINAMATH_GPT_num_balls_total_l1264_126423


namespace NUMINAMATH_GPT_area_bounded_region_l1264_126473

theorem area_bounded_region (x y : ℝ) (h : y^2 + 2*x*y + 30*|x| = 300) : 
  ∃ A, A = 900 := 
sorry

end NUMINAMATH_GPT_area_bounded_region_l1264_126473


namespace NUMINAMATH_GPT_option_A_is_quadratic_l1264_126415

def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0

-- Given options
def option_A_equation (x : ℝ) : Prop :=
  x^2 - 2 = 0

def option_B_equation (x y : ℝ) : Prop :=
  x + 2 * y = 3

def option_C_equation (x : ℝ) : Prop :=
  x - 1/x = 1

def option_D_equation (x y : ℝ) : Prop :=
  x^2 + x = y + 1

-- Prove that option A is a quadratic equation
theorem option_A_is_quadratic (x : ℝ) : is_quadratic_equation 1 0 (-2) :=
by
  sorry

end NUMINAMATH_GPT_option_A_is_quadratic_l1264_126415


namespace NUMINAMATH_GPT_minimum_detectors_required_l1264_126464

/-- There is a cube with each face divided into 4 identical square cells, making a total of 24 cells.
Oleg wants to mark 8 cells with invisible ink such that no two marked cells share a side.
Rustem wants to place detectors in the cells so that all marked cells can be identified. -/
def minimum_detectors_to_identify_all_marked_cells (total_cells: ℕ) (marked_cells: ℕ) 
  (cells_per_face: ℕ) (faces: ℕ) : ℕ :=
  if total_cells = faces * cells_per_face ∧ marked_cells = 8 then 16 else 0

theorem minimum_detectors_required :
  minimum_detectors_to_identify_all_marked_cells 24 8 4 6 = 16 :=
by
  sorry

end NUMINAMATH_GPT_minimum_detectors_required_l1264_126464


namespace NUMINAMATH_GPT_coats_collected_elem_schools_correct_l1264_126478

-- Conditions
def total_coats_collected : ℕ := 9437
def coats_collected_high_schools : ℕ := 6922

-- Definition to find coats collected from elementary schools
def coats_collected_elementary_schools : ℕ := total_coats_collected - coats_collected_high_schools

-- Theorem statement
theorem coats_collected_elem_schools_correct : 
  coats_collected_elementary_schools = 2515 := sorry

end NUMINAMATH_GPT_coats_collected_elem_schools_correct_l1264_126478


namespace NUMINAMATH_GPT_repeated_root_and_m_value_l1264_126492

theorem repeated_root_and_m_value :
  (∃ x m : ℝ, (x = 2 ∨ x = -2) ∧ 
              (m / (x ^ 2 - 4) + 2 / (x + 2) = 1 / (x - 2)) ∧ 
              (m = 4 ∨ m = 8)) :=
sorry

end NUMINAMATH_GPT_repeated_root_and_m_value_l1264_126492


namespace NUMINAMATH_GPT_eval_composed_function_l1264_126451

noncomputable def f (x : ℝ) := 3 * x^2 - 4
noncomputable def k (x : ℝ) := 5 * x^3 + 2

theorem eval_composed_function :
  f (k 2) = 5288 := 
by
  sorry

end NUMINAMATH_GPT_eval_composed_function_l1264_126451


namespace NUMINAMATH_GPT_initial_amount_l1264_126417

-- Define the given conditions
def amount_spent : ℕ := 16
def amount_left : ℕ := 2

-- Define the statement that we want to prove
theorem initial_amount : amount_spent + amount_left = 18 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_l1264_126417


namespace NUMINAMATH_GPT_decrement_from_observation_l1264_126454

theorem decrement_from_observation 
  (n : ℕ) (mean_original mean_updated : ℚ)
  (h1 : n = 50)
  (h2 : mean_original = 200)
  (h3 : mean_updated = 194)
  : (mean_original - mean_updated) = 6 :=
by
  sorry

end NUMINAMATH_GPT_decrement_from_observation_l1264_126454


namespace NUMINAMATH_GPT_mail_distribution_l1264_126409

theorem mail_distribution (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : total_mail / total_houses = 6 := by
  sorry

end NUMINAMATH_GPT_mail_distribution_l1264_126409


namespace NUMINAMATH_GPT_tangent_lines_through_point_l1264_126484

theorem tangent_lines_through_point (x y : ℝ) :
  (x^2 + y^2 + 2*x - 2*y + 1 = 0) ∧ (x = -2 ∨ (15*x + 8*y - 10 = 0)) ↔ 
  (x = -2 ∨ (15*x + 8*y - 10 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_lines_through_point_l1264_126484


namespace NUMINAMATH_GPT_correct_conditions_for_cubic_eq_single_root_l1264_126442

noncomputable def hasSingleRealRoot (a b : ℝ) : Prop :=
  let f := λ x : ℝ => x^3 - a * x + b
  let f' := λ x : ℝ => 3 * x^2 - a
  ∀ (x y : ℝ), f' x = 0 → f' y = 0 → x = y

theorem correct_conditions_for_cubic_eq_single_root :
  (hasSingleRealRoot 0 2) ∧ 
  (hasSingleRealRoot (-3) 2) ∧ 
  (hasSingleRealRoot 3 (-3)) :=
  by 
    sorry

end NUMINAMATH_GPT_correct_conditions_for_cubic_eq_single_root_l1264_126442


namespace NUMINAMATH_GPT_evaluate_expression_l1264_126465

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 5) :
  3 * x^4 + 2 * y^2 + 10 = 8 * 37 + 7 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1264_126465


namespace NUMINAMATH_GPT_share_of_each_person_l1264_126431

theorem share_of_each_person (total_length : ℕ) (h1 : total_length = 12) (h2 : total_length % 2 = 0)
  : total_length / 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_share_of_each_person_l1264_126431


namespace NUMINAMATH_GPT_volume_of_one_slice_l1264_126446

theorem volume_of_one_slice
  (circumference : ℝ)
  (c : circumference = 18 * Real.pi):
  ∃ V, V = 162 * Real.pi :=
by sorry

end NUMINAMATH_GPT_volume_of_one_slice_l1264_126446


namespace NUMINAMATH_GPT_jackson_entertainment_cost_l1264_126499

def price_computer_game : ℕ := 66
def price_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3
def total_entertainment_cost : ℕ := price_computer_game + number_of_movie_tickets * price_movie_ticket

theorem jackson_entertainment_cost : total_entertainment_cost = 102 := by
  sorry

end NUMINAMATH_GPT_jackson_entertainment_cost_l1264_126499


namespace NUMINAMATH_GPT_jason_average_messages_l1264_126440

theorem jason_average_messages : 
    let monday := 220
    let tuesday := monday / 2
    let wednesday := 50
    let thursday := 50
    let friday := 50
    let total_messages := monday + tuesday + wednesday + thursday + friday
    let average_messages := total_messages / 5
    average_messages = 96 :=
by
  let monday := 220
  let tuesday := monday / 2
  let wednesday := 50
  let thursday := 50
  let friday := 50
  let total_messages := monday + tuesday + wednesday + thursday + friday
  let average_messages := total_messages / 5
  have h : average_messages = 96 := sorry
  exact h

end NUMINAMATH_GPT_jason_average_messages_l1264_126440


namespace NUMINAMATH_GPT_bus_routes_theorem_l1264_126418

open Function

def bus_routes_exist : Prop :=
  ∃ (routes : Fin 10 → Set (Fin 10)), 
  (∀ (s : Finset (Fin 10)), (s.card = 8) → ∃ (stop : Fin 10), ∀ i ∈ s, stop ∉ routes i) ∧
  (∀ (s : Finset (Fin 10)), (s.card = 9) → ∀ (stop : Fin 10), ∃ i ∈ s, stop ∈ routes i)

theorem bus_routes_theorem : bus_routes_exist :=
sorry

end NUMINAMATH_GPT_bus_routes_theorem_l1264_126418


namespace NUMINAMATH_GPT_total_interest_received_l1264_126432

def principal_B := 5000
def principal_C := 3000
def rate := 9
def time_B := 2
def time_C := 4
def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℕ := P * R * T / 100

theorem total_interest_received :
  let SI_B := simple_interest principal_B rate time_B
  let SI_C := simple_interest principal_C rate time_C
  SI_B + SI_C = 1980 := 
by
  sorry

end NUMINAMATH_GPT_total_interest_received_l1264_126432


namespace NUMINAMATH_GPT_product_cubed_roots_l1264_126491

-- Given conditions
def cbrt (x : ℝ) : ℝ := x^(1/3)
def expr : ℝ := cbrt (1 + 27) * cbrt (1 + cbrt 27) * cbrt 9

-- Main statement to prove
theorem product_cubed_roots : expr = cbrt 1008 :=
by sorry

end NUMINAMATH_GPT_product_cubed_roots_l1264_126491


namespace NUMINAMATH_GPT_smallest_pos_int_y_satisfies_congruence_l1264_126424

theorem smallest_pos_int_y_satisfies_congruence :
  ∃ y : ℕ, (y > 0) ∧ (26 * y + 8) % 16 = 4 ∧ ∀ z : ℕ, (z > 0) ∧ (26 * z + 8) % 16 = 4 → y ≤ z :=
sorry

end NUMINAMATH_GPT_smallest_pos_int_y_satisfies_congruence_l1264_126424


namespace NUMINAMATH_GPT_calculate_length_of_bridge_l1264_126493

/-- Define the conditions based on given problem -/
def length_of_bridge (speed1 speed2 : ℕ) (length1 length2 : ℕ) (time : ℕ) : ℕ :=
    let distance_covered_train1 := speed1 * time
    let bridge_length_train1 := distance_covered_train1 - length1
    let distance_covered_train2 := speed2 * time
    let bridge_length_train2 := distance_covered_train2 - length2
    max bridge_length_train1 bridge_length_train2

/-- Given conditions -/
def speed_train1 := 15 -- in m/s
def length_train1 := 130 -- in meters
def speed_train2 := 20 -- in m/s
def length_train2 := 90 -- in meters
def crossing_time := 30 -- in seconds

theorem calculate_length_of_bridge : length_of_bridge speed_train1 speed_train2 length_train1 length_train2 crossing_time = 510 :=
by
  -- omitted proof
  sorry

end NUMINAMATH_GPT_calculate_length_of_bridge_l1264_126493


namespace NUMINAMATH_GPT_sum_real_roots_eq_neg4_l1264_126488

-- Define the equation condition
def equation_condition (x : ℝ) : Prop :=
  (2 * x / (x^2 + 5 * x + 3) + 3 * x / (x^2 + x + 3) = 1)

-- Define the statement that sums the real roots
theorem sum_real_roots_eq_neg4 : 
  ∃ S : ℝ, (∀ x : ℝ, equation_condition x → x = -1 ∨ x = -3) ∧ (S = -4) :=
sorry

end NUMINAMATH_GPT_sum_real_roots_eq_neg4_l1264_126488


namespace NUMINAMATH_GPT_not_prime_for_some_n_l1264_126403

theorem not_prime_for_some_n (a : ℕ) (h : 1 < a) : ∃ n : ℕ, ¬ Nat.Prime (2^(2^n) + a) := 
sorry

end NUMINAMATH_GPT_not_prime_for_some_n_l1264_126403


namespace NUMINAMATH_GPT_finite_decimal_representation_nat_numbers_l1264_126496

theorem finite_decimal_representation_nat_numbers (n : ℕ) : 
  (∀ k : ℕ, k < n → (∃ u v : ℕ, (k + 1 = 2^u ∨ k + 1 = 5^v) ∨ (k - 1 = 2^u ∨ k -1  = 5^v))) ↔ 
  (n = 2 ∨ n = 3 ∨ n = 6) :=
by sorry

end NUMINAMATH_GPT_finite_decimal_representation_nat_numbers_l1264_126496


namespace NUMINAMATH_GPT_total_pens_l1264_126444

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end NUMINAMATH_GPT_total_pens_l1264_126444


namespace NUMINAMATH_GPT_range_of_a_l1264_126436

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (2 * x - a > 0 ∧ 3 * x - 4 < 5) -> False) ↔ (a ≥ 6) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1264_126436


namespace NUMINAMATH_GPT_train_crosses_post_in_25_2_seconds_l1264_126447

noncomputable def train_crossing_time (speed_kmph : ℝ) (length_m : ℝ) : ℝ :=
  length_m / (speed_kmph * 1000 / 3600)

theorem train_crosses_post_in_25_2_seconds :
  train_crossing_time 40 280.0224 = 25.2 :=
by 
  sorry

end NUMINAMATH_GPT_train_crosses_post_in_25_2_seconds_l1264_126447


namespace NUMINAMATH_GPT_max_min_value_l1264_126476

def f (x t : ℝ) : ℝ := x^2 - 2 * t * x + t

theorem max_min_value : 
  ∀ t : ℝ, (-1 ≤ t ∧ t ≤ 1) →
  (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → f x t ≥ -t^2 + t) →
  (∃ t : ℝ, (-1 ≤ t ∧ t ≤ 1) ∧ ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → f x t ≥ -t^2 + t ∧ -t^2 + t = 1/4) :=
sorry

end NUMINAMATH_GPT_max_min_value_l1264_126476


namespace NUMINAMATH_GPT_solve_for_question_mark_l1264_126430

/-- Prove that the number that should replace "?" in the equation 
    300 * 2 + (12 + ?) * (1 / 8) = 602 is equal to 4. -/
theorem solve_for_question_mark : 
  ∃ (x : ℕ), 300 * 2 + (12 + x) * (1 / 8) = 602 ∧ x = 4 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_question_mark_l1264_126430


namespace NUMINAMATH_GPT_find_abc_l1264_126453

theorem find_abc (a b c : ℝ) (h1 : a * (b + c) = 198) (h2 : b * (c + a) = 210) (h3 : c * (a + b) = 222) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) : 
  a * b * c = 1069 :=
by
  sorry

end NUMINAMATH_GPT_find_abc_l1264_126453


namespace NUMINAMATH_GPT_loss_percentage_grinder_l1264_126429

-- Conditions
def CP_grinder : ℝ := 15000
def CP_mobile : ℝ := 8000
def profit_mobile : ℝ := 0.10
def total_profit : ℝ := 200

-- Theorem to prove the loss percentage on the grinder
theorem loss_percentage_grinder : 
  ( (CP_grinder - (23200 - (CP_mobile * (1 + profit_mobile)))) / CP_grinder ) * 100 = 4 :=
by
  sorry

end NUMINAMATH_GPT_loss_percentage_grinder_l1264_126429


namespace NUMINAMATH_GPT_cristina_catches_up_l1264_126485

theorem cristina_catches_up
  (t : ℝ)
  (cristina_speed : ℝ := 5)
  (nicky_speed : ℝ := 3)
  (nicky_head_start : ℝ := 54)
  (distance_cristina : ℝ := cristina_speed * t)
  (distance_nicky : ℝ := nicky_head_start + nicky_speed * t) :
  distance_cristina = distance_nicky → t = 27 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_cristina_catches_up_l1264_126485


namespace NUMINAMATH_GPT_total_people_l1264_126407

-- Define the conditions as constants
def B : ℕ := 50
def S : ℕ := 70
def B_inter_S : ℕ := 20

-- Total number of people in the group
theorem total_people : B + S - B_inter_S = 100 := by
  sorry

end NUMINAMATH_GPT_total_people_l1264_126407


namespace NUMINAMATH_GPT_midpoint_trajectory_l1264_126483

theorem midpoint_trajectory (x y : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (8, 0) ∧ (B.1, B.2) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 } ∧ 
   ∃ P : ℝ × ℝ, P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ P = (x, y)) → (x - 4)^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_GPT_midpoint_trajectory_l1264_126483


namespace NUMINAMATH_GPT_probability_white_first_red_second_l1264_126459

theorem probability_white_first_red_second :
  let total_marbles := 10
  let white_marbles := 6
  let red_marbles := 4
  let prob_white_first := white_marbles / total_marbles
  let prob_red_second_given_white_first := red_marbles / (total_marbles - 1)
  let prob_combined := prob_white_first * prob_red_second_given_white_first
  prob_combined = 4 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_white_first_red_second_l1264_126459


namespace NUMINAMATH_GPT_interior_and_exterior_angles_of_regular_dodecagon_l1264_126470

-- Definition of a regular dodecagon
def regular_dodecagon_sides : ℕ := 12

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Measure of one interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Measure of one exterior angle of a regular polygon (180 degrees supplementary to interior angle)
def one_exterior_angle (n : ℕ) : ℕ := 180 - one_interior_angle n

-- The theorem to prove
theorem interior_and_exterior_angles_of_regular_dodecagon :
  one_interior_angle regular_dodecagon_sides = 150 ∧ one_exterior_angle regular_dodecagon_sides = 30 :=
by
  sorry

end NUMINAMATH_GPT_interior_and_exterior_angles_of_regular_dodecagon_l1264_126470


namespace NUMINAMATH_GPT_general_term_of_sequence_l1264_126455

theorem general_term_of_sequence (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n, a (n + 1) = a n + n + 1) :
  ∀ n, a n = (n^2 + n + 2) / 2 :=
by 
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l1264_126455


namespace NUMINAMATH_GPT_find_f2_l1264_126438

namespace ProofProblem

-- Define the polynomial function f
def f (x a b : ℤ) : ℤ := x^5 + a * x^3 + b * x - 8

-- Conditions given in the problem
axiom f_neg2 : ∃ a b : ℤ, f (-2) a b = 10

-- Define the theorem statement
theorem find_f2 : ∃ a b : ℤ, f 2 a b = -26 :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_find_f2_l1264_126438


namespace NUMINAMATH_GPT_smallest_k_l1264_126487

def v_seq (v : ℕ → ℝ) : Prop :=
  v 0 = 1/8 ∧ ∀ k, v (k + 1) = 3 * v k - 3 * (v k)^2

noncomputable def limit_M : ℝ := 0.5

theorem smallest_k 
  (v : ℕ → ℝ)
  (hv : v_seq v) :
  ∃ k : ℕ, |v k - limit_M| ≤ 1 / 2 ^ 500 ∧ ∀ n < k, ¬ (|v n - limit_M| ≤ 1 / 2 ^ 500) := 
sorry

end NUMINAMATH_GPT_smallest_k_l1264_126487


namespace NUMINAMATH_GPT_terminating_decimals_count_l1264_126498

theorem terminating_decimals_count :
  (∀ m : ℤ, 1 ≤ m ∧ m ≤ 999 → ∃ k : ℕ, (m : ℝ) / 1000 = k / (2 ^ 3 * 5 ^ 3)) :=
by
  sorry

end NUMINAMATH_GPT_terminating_decimals_count_l1264_126498


namespace NUMINAMATH_GPT_answer_choices_l1264_126489

theorem answer_choices (n : ℕ) (h : (n + 1) ^ 4 = 625) : n = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_answer_choices_l1264_126489


namespace NUMINAMATH_GPT_vision_statistics_l1264_126400

noncomputable def average (values : List ℝ) : ℝ := (List.sum values) / (List.length values)

noncomputable def variance (values : List ℝ) : ℝ :=
  let mean := average values
  (List.sum (values.map (λ x => (x - mean) ^ 2))) / (List.length values)

def classA_visions : List ℝ := [4.3, 5.1, 4.6, 4.1, 4.9]
def classB_visions : List ℝ := [5.1, 4.9, 4.0, 4.0, 4.5]

theorem vision_statistics :
  average classA_visions = 4.6 ∧
  average classB_visions = 4.5 ∧
  variance classA_visions = 0.136 ∧
  (let count := List.length classB_visions
   let total := count.choose 2
   let favorable := 3  -- (5.1, 4.5), (5.1, 4.9), (4.9, 4.5)
   7 / 10 = 1 - (favorable / total)) :=
by
  sorry

end NUMINAMATH_GPT_vision_statistics_l1264_126400


namespace NUMINAMATH_GPT_amount_of_sugar_l1264_126416

-- Let ratio_sugar_flour be the ratio of sugar to flour.
def ratio_sugar_flour : ℕ := 10

-- Let flour be the amount of flour used in ounces.
def flour : ℕ := 5

-- Let sugar be the amount of sugar used in ounces.
def sugar (ratio_sugar_flour : ℕ) (flour : ℕ) : ℕ := ratio_sugar_flour * flour

-- The proof goal: given the conditions, prove that the amount of sugar used is 50 ounces.
theorem amount_of_sugar (h_ratio : ratio_sugar_flour = 10) (h_flour : flour = 5) : sugar ratio_sugar_flour flour = 50 :=
by
  -- Proof omitted.
  sorry
 
end NUMINAMATH_GPT_amount_of_sugar_l1264_126416


namespace NUMINAMATH_GPT_cone_angle_60_degrees_l1264_126404

theorem cone_angle_60_degrees (r : ℝ) (h : ℝ) (θ : ℝ) 
  (arc_len : θ = 60) 
  (slant_height : h = r) : θ = 60 :=
sorry

end NUMINAMATH_GPT_cone_angle_60_degrees_l1264_126404


namespace NUMINAMATH_GPT_find_pairs_of_nonneg_ints_l1264_126428

theorem find_pairs_of_nonneg_ints (m n : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n + 1) - 1) ↔ (m, n) = (9, 3) ∨ (m, n) = (6, 3) ∨ (m, n) = (9, 5) ∨ (m, n) = (54, 5) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_of_nonneg_ints_l1264_126428


namespace NUMINAMATH_GPT_ratio_of_doctors_lawyers_engineers_l1264_126467

variables (d l e : ℕ)

-- Conditions
def average_age_per_group (d l e : ℕ) : Prop :=
  (40 * d + 55 * l + 35 * e) = 45 * (d + l + e)

-- Theorem
theorem ratio_of_doctors_lawyers_engineers
  (h : average_age_per_group d l e) :
  l = d + 2 * e :=
by sorry

end NUMINAMATH_GPT_ratio_of_doctors_lawyers_engineers_l1264_126467


namespace NUMINAMATH_GPT_notebook_cost_l1264_126445

theorem notebook_cost :
  let mean_expenditure := 500
  let daily_expenditures := [450, 600, 400, 500, 550, 300]
  let cost_earphone := 620
  let cost_pen := 30
  let total_days := 7
  let total_expenditure := mean_expenditure * total_days
  let sum_other_days := daily_expenditures.sum
  let expenditure_friday := total_expenditure - sum_other_days
  let cost_notebook := expenditure_friday - (cost_earphone + cost_pen)
  cost_notebook = 50 := by
  sorry

end NUMINAMATH_GPT_notebook_cost_l1264_126445


namespace NUMINAMATH_GPT_mindy_tax_rate_l1264_126408

variables (M : ℝ) -- Mork's income
variables (r : ℝ) -- Mindy's tax rate

-- Conditions
def Mork_tax_rate := 0.45 -- 45% tax rate
def Mindx_income := 4 * M -- Mindy earned 4 times as much as Mork
def combined_tax_rate := 0.21 -- Combined tax rate is 21%

-- Equation derived from the conditions
def combined_tax_rate_eq := (0.45 * M + 4 * M * r) / (M + 4 * M) = 0.21

theorem mindy_tax_rate : combined_tax_rate_eq M r → r = 0.15 :=
by
  intros conditional_eq
  sorry

end NUMINAMATH_GPT_mindy_tax_rate_l1264_126408


namespace NUMINAMATH_GPT_probability_of_sine_inequality_l1264_126481

open Set Real

noncomputable def probability_sine_inequality (x : ℝ) : Prop :=
  ∃ (μ : MeasureTheory.Measure ℝ), μ (Ioc (-3) 3) = 1 ∧
    μ {x | sin (π / 6 * x) ≥ 1 / 2} = 1 / 3

theorem probability_of_sine_inequality : probability_sine_inequality x :=
by
  sorry

end NUMINAMATH_GPT_probability_of_sine_inequality_l1264_126481


namespace NUMINAMATH_GPT_inversely_proportional_value_l1264_126434

theorem inversely_proportional_value (a b k : ℝ) (h1 : a * b = k) (h2 : a = 40) (h3 : b = 8) :
  ∃ a' : ℝ, a' * 10 = k ∧ a' = 32 :=
by {
  use 32,
  sorry
}

end NUMINAMATH_GPT_inversely_proportional_value_l1264_126434


namespace NUMINAMATH_GPT_hourly_wage_increase_l1264_126494

variables (W W' H H' : ℝ)

theorem hourly_wage_increase :
  H' = (2/3) * H →
  W * H = W' * H' →
  W' = (3/2) * W :=
by
  intros h_eq income_eq
  rw [h_eq] at income_eq
  sorry

end NUMINAMATH_GPT_hourly_wage_increase_l1264_126494


namespace NUMINAMATH_GPT_negation_of_proposition_exists_negation_of_proposition_l1264_126474

theorem negation_of_proposition : 
  (∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) ↔ ¬(∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) :=
by
  sorry

theorem exists_negation_of_proposition : 
  (¬(∀ x : ℝ, 2^x - 2*x - 2 ≥ 0)) ↔ ∃ x : ℝ, 2^x - 2*x - 2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_exists_negation_of_proposition_l1264_126474


namespace NUMINAMATH_GPT_average_mb_per_hour_l1264_126482

theorem average_mb_per_hour
  (days : ℕ)
  (original_space  : ℕ)
  (compression_rate : ℝ)
  (total_hours : ℕ := days * 24)
  (effective_space : ℝ := original_space * (1 - compression_rate))
  (space_per_hour : ℝ := effective_space / total_hours) :
  days = 20 ∧ original_space = 25000 ∧ compression_rate = 0.10 → 
  (Int.floor (space_per_hour + 0.5)) = 47 := by
  intros
  sorry

end NUMINAMATH_GPT_average_mb_per_hour_l1264_126482


namespace NUMINAMATH_GPT_base_6_four_digit_odd_final_digit_l1264_126469

-- Definition of the conditions
def four_digit_number (n b : ℕ) : Prop :=
  b^3 ≤ n ∧ n < b^4

def odd_digit (n b : ℕ) : Prop :=
  (n % b) % 2 = 1

-- Problem statement
theorem base_6_four_digit_odd_final_digit :
  four_digit_number 350 6 ∧ odd_digit 350 6 := by
  sorry

end NUMINAMATH_GPT_base_6_four_digit_odd_final_digit_l1264_126469


namespace NUMINAMATH_GPT_min_value_b1_b2_l1264_126471

noncomputable def seq (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b n + 2017) / (1 + b (n + 1))

theorem min_value_b1_b2 (b : ℕ → ℕ)
  (h_pos : ∀ n, b n > 0)
  (h_seq : seq b) :
  b 1 + b 2 = 2018 := sorry

end NUMINAMATH_GPT_min_value_b1_b2_l1264_126471


namespace NUMINAMATH_GPT_x_power6_y_power6_l1264_126433

theorem x_power6_y_power6 (x y a b : ℝ) (h1 : x + y = a) (h2 : x * y = b) :
  x^6 + y^6 = a^6 - 6 * a^4 * b + 9 * a^2 * b^2 - 2 * b^3 :=
sorry

end NUMINAMATH_GPT_x_power6_y_power6_l1264_126433


namespace NUMINAMATH_GPT_sin_A_over_1_minus_cos_A_l1264_126456

variable {a b c : ℝ} -- Side lengths of the triangle
variable {A B C : ℝ} -- Angles opposite to the sides

theorem sin_A_over_1_minus_cos_A 
  (h_area : 0.5 * b * c * Real.sin A = a^2 - (b - c)^2) :
  Real.sin A / (1 - Real.cos A) = 3 :=
sorry

end NUMINAMATH_GPT_sin_A_over_1_minus_cos_A_l1264_126456


namespace NUMINAMATH_GPT_min_value_expression_l1264_126480

theorem min_value_expression (x : ℝ) (hx : x > 0) : x + 4/x ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1264_126480


namespace NUMINAMATH_GPT_vector_calculation_l1264_126458

def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (1, -1)
def vec_result : ℝ × ℝ := (3 * vec_a.fst - 2 * vec_b.fst, 3 * vec_a.snd - 2 * vec_b.snd)
def target_vec : ℝ × ℝ := (1, 5)

theorem vector_calculation :
  vec_result = target_vec :=
sorry

end NUMINAMATH_GPT_vector_calculation_l1264_126458


namespace NUMINAMATH_GPT_tom_ate_one_pound_of_carrots_l1264_126411

noncomputable def calories_from_carrots (C : ℝ) : ℝ := 51 * C
noncomputable def calories_from_broccoli (C : ℝ) : ℝ := (51 / 3) * (2 * C)
noncomputable def total_calories (C : ℝ) : ℝ :=
  calories_from_carrots C + calories_from_broccoli C

theorem tom_ate_one_pound_of_carrots :
  ∃ C : ℝ, total_calories C = 85 ∧ C = 1 :=
by
  use 1
  simp [total_calories, calories_from_carrots, calories_from_broccoli]
  sorry

end NUMINAMATH_GPT_tom_ate_one_pound_of_carrots_l1264_126411


namespace NUMINAMATH_GPT_total_distance_proof_l1264_126405

-- Define the conditions
def amoli_speed : ℕ := 42      -- Amoli's speed in miles per hour
def amoli_time : ℕ := 3        -- Amoli's driving time in hours
def anayet_speed : ℕ := 61     -- Anayet's speed in miles per hour
def anayet_time : ℕ := 2       -- Anayet's driving time in hours
def remaining_distance : ℕ := 121  -- Remaining distance to be traveled in miles

-- Total distance calculation
def total_distance : ℕ :=
  amoli_speed * amoli_time + anayet_speed * anayet_time + remaining_distance

-- The theorem to prove
theorem total_distance_proof : total_distance = 369 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_distance_proof_l1264_126405


namespace NUMINAMATH_GPT_minimum_square_side_length_l1264_126449

theorem minimum_square_side_length (s : ℝ) (h1 : s^2 ≥ 625) (h2 : ∃ (t : ℝ), t = s / 2) : s = 25 :=
by
  sorry

end NUMINAMATH_GPT_minimum_square_side_length_l1264_126449


namespace NUMINAMATH_GPT_black_and_blue_lines_l1264_126402

-- Definition of given conditions
def grid_size : ℕ := 50
def total_points : ℕ := grid_size * grid_size
def blue_points : ℕ := 1510
def blue_edge_points : ℕ := 110
def red_segments : ℕ := 947
def corner_points : ℕ := 4

-- Calculations based on conditions
def red_points : ℕ := total_points - blue_points

def edge_points (size : ℕ) : ℕ := (size - 1) * 4
def non_corner_edge_points (edge : ℕ) : ℕ := edge - corner_points

-- Math translation
noncomputable def internal_red_points : ℕ := red_points - corner_points - (edge_points grid_size - blue_edge_points)
noncomputable def connections_from_red_points : ℕ :=
  corner_points * 2 + (non_corner_edge_points (edge_points grid_size) - blue_edge_points) * 3 + internal_red_points * 4

noncomputable def adjusted_red_lines : ℕ := red_segments * 2
noncomputable def black_lines : ℕ := connections_from_red_points - adjusted_red_lines

def total_lines (size : ℕ) : ℕ := (size - 1) * size + (size - 1) * size
noncomputable def blue_lines : ℕ := total_lines grid_size - red_segments - black_lines

-- The theorem to be proven
theorem black_and_blue_lines :
  (black_lines = 1972) ∧ (blue_lines = 1981) :=
by
  sorry

end NUMINAMATH_GPT_black_and_blue_lines_l1264_126402


namespace NUMINAMATH_GPT_find_n_l1264_126401

theorem find_n : ∃ n : ℕ, 50^4 + 43^4 + 36^4 + 6^4 = n^4 := by
  sorry

end NUMINAMATH_GPT_find_n_l1264_126401


namespace NUMINAMATH_GPT_correct_understanding_of_philosophy_l1264_126495

-- Define the conditions based on the problem statement
def philosophy_from_life_and_practice : Prop :=
  -- Philosophy originates from people's lives and practice.
  sorry
  
def philosophy_affects_lives : Prop :=
  -- Philosophy consciously or unconsciously affects people's lives, learning, and work
  sorry

def philosophical_knowledge_requires_learning : Prop :=
  true

def philosophy_not_just_summary : Prop :=
  true

-- Given conditions 1, 2, 3 (as negation of 3 in original problem), and 4 (as negation of 4 in original problem),
-- We need to prove the correct understanding (which is combination ①②) is correct.
theorem correct_understanding_of_philosophy :
  philosophy_from_life_and_practice →
  philosophy_affects_lives →
  philosophical_knowledge_requires_learning →
  philosophy_not_just_summary →
  (philosophy_from_life_and_practice ∧ philosophy_affects_lives) :=
by
  intros
  apply And.intro
  · assumption
  · assumption

end NUMINAMATH_GPT_correct_understanding_of_philosophy_l1264_126495


namespace NUMINAMATH_GPT_small_branches_count_l1264_126466

theorem small_branches_count (x : ℕ) (h : x^2 + x + 1 = 91) : x = 9 := 
  sorry

end NUMINAMATH_GPT_small_branches_count_l1264_126466


namespace NUMINAMATH_GPT_solution_inequality_l1264_126427

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function_at (f : ℝ → ℝ) (x : ℝ) : Prop := f (2 + x) = f (2 - x)
def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x < y → x ∈ s → y ∈ s → f x < f y

-- Main statement
theorem solution_inequality 
  (h1 : ∀ x, is_even_function_at f x)
  (h2 : is_increasing_on f {x : ℝ | x ≤ 2}) :
  (∀ a : ℝ, (a > -1) ∧ (a ≠ 0) ↔ f (a^2 + 3*a + 2) < f (a^2 - a + 2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_inequality_l1264_126427


namespace NUMINAMATH_GPT_average_age_l1264_126435
open Nat

def age_to_months (years : ℕ) (months : ℕ) : ℕ := years * 12 + months

theorem average_age :
  let age1 := age_to_months 14 9
  let age2 := age_to_months 15 1
  let age3 := age_to_months 14 8
  let total_months := age1 + age2 + age3
  let avg_months := total_months / 3
  let avg_years := avg_months / 12
  let avg_remaining_months := avg_months % 12
  avg_years = 14 ∧ avg_remaining_months = 10 := by
  sorry

end NUMINAMATH_GPT_average_age_l1264_126435


namespace NUMINAMATH_GPT_Ron_spends_15_dollars_l1264_126486

theorem Ron_spends_15_dollars (cost_per_bar : ℝ) (sections_per_bar : ℕ) (num_scouts : ℕ) (s'mores_per_scout : ℕ) :
  cost_per_bar = 1.50 ∧ sections_per_bar = 3 ∧ num_scouts = 15 ∧ s'mores_per_scout = 2 →
  cost_per_bar * (num_scouts * s'mores_per_scout / sections_per_bar) = 15 :=
by
  sorry

end NUMINAMATH_GPT_Ron_spends_15_dollars_l1264_126486


namespace NUMINAMATH_GPT_felipe_building_time_l1264_126479

theorem felipe_building_time
  (F E : ℕ)
  (combined_time_without_breaks : ℕ)
  (felipe_time_fraction : F = E / 2)
  (combined_time_condition : F + E = 90)
  (felipe_break : ℕ)
  (emilio_break : ℕ)
  (felipe_break_is_6_months : felipe_break = 6)
  (emilio_break_is_double_felipe : emilio_break = 2 * felipe_break) :
  F + felipe_break = 36 := by
  sorry

end NUMINAMATH_GPT_felipe_building_time_l1264_126479


namespace NUMINAMATH_GPT_find_sum_of_angles_l1264_126443

-- Given conditions
def angleP := 34
def angleQ := 76
def angleR := 28

-- Proposition to prove
theorem find_sum_of_angles (x z : ℝ) (h1 : x + z = 138) : x + z = 138 :=
by
  have angleP := 34
  have angleQ := 76
  have angleR := 28
  exact h1

end NUMINAMATH_GPT_find_sum_of_angles_l1264_126443


namespace NUMINAMATH_GPT_divide_milk_in_half_l1264_126419

theorem divide_milk_in_half (bucket : ℕ) (a : ℕ) (b : ℕ) (a_liters : a = 5) (b_liters : b = 7) (bucket_liters : bucket = 12) :
  ∃ x y : ℕ, x = 6 ∧ y = 6 ∧ x + y = bucket := by
  sorry

end NUMINAMATH_GPT_divide_milk_in_half_l1264_126419


namespace NUMINAMATH_GPT_clown_balloons_l1264_126410

theorem clown_balloons 
  (initial_balloons : ℕ := 123) 
  (additional_balloons : ℕ := 53) 
  (given_away_balloons : ℕ := 27) : 
  initial_balloons + additional_balloons - given_away_balloons = 149 := 
by 
  sorry

end NUMINAMATH_GPT_clown_balloons_l1264_126410


namespace NUMINAMATH_GPT_find_y_l1264_126439

-- Define the atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of the compound C6HyO7
def molecular_weight : ℝ := 192

-- Define the contribution of Carbon and Oxygen
def contribution_C : ℝ := 6 * atomic_weight_C
def contribution_O : ℝ := 7 * atomic_weight_O

-- The proof statement
theorem find_y (y : ℕ) :
  molecular_weight = contribution_C + y * atomic_weight_H + contribution_O → y = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1264_126439


namespace NUMINAMATH_GPT_quadratic_complete_the_square_l1264_126497

theorem quadratic_complete_the_square :
  ∃ b c : ℝ, (∀ x : ℝ, x^2 + 1500 * x + 1500 = (x + b) ^ 2 + c)
      ∧ b = 750
      ∧ c = -748 * 750
      ∧ c / b = -748 := 
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_complete_the_square_l1264_126497


namespace NUMINAMATH_GPT_fluffy_striped_or_spotted_cats_l1264_126463

theorem fluffy_striped_or_spotted_cats (total_cats : ℕ) (striped_fraction : ℚ) (spotted_fraction : ℚ)
    (fluffy_striped_fraction : ℚ) (fluffy_spotted_fraction : ℚ) (striped_spotted_fraction : ℚ) :
    total_cats = 180 ∧ striped_fraction = 1/2 ∧ spotted_fraction = 1/3 ∧
    fluffy_striped_fraction = 1/8 ∧ fluffy_spotted_fraction = 3/7 →
    striped_spotted_fraction = 36 :=
by
    sorry

end NUMINAMATH_GPT_fluffy_striped_or_spotted_cats_l1264_126463


namespace NUMINAMATH_GPT_original_wattage_l1264_126468

theorem original_wattage (W : ℝ) (h1 : 143 = 1.30 * W) : W = 110 := 
by
  sorry

end NUMINAMATH_GPT_original_wattage_l1264_126468


namespace NUMINAMATH_GPT_quadratic_ineq_solution_l1264_126452

theorem quadratic_ineq_solution (a b : ℝ) 
  (h_solution_set : ∀ x, (ax^2 + bx - 1 > 0) ↔ (1 / 3 < x ∧ x < 1))
  (h_roots : (a / 3 + b = -1 / a) ∧ (a / 3 = -1 / a)) 
  (h_a_neg : a < 0) : a + b = 1 := 
sorry 

end NUMINAMATH_GPT_quadratic_ineq_solution_l1264_126452


namespace NUMINAMATH_GPT_wire_length_is_180_l1264_126490

def wire_problem (length1 length2 : ℕ) (h1 : length1 = 106) (h2 : length2 = 74) (h3 : length1 = length2 + 32) : Prop :=
  (length1 + length2 = 180)

-- Use the definition as an assumption to write the theorem.
theorem wire_length_is_180 (length1 length2 : ℕ) 
  (h1 : length1 = 106) 
  (h2 : length2 = 74) 
  (h3 : length1 = length2 + 32) : 
  length1 + length2 = 180 :=
by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_wire_length_is_180_l1264_126490


namespace NUMINAMATH_GPT_cost_of_playing_cards_l1264_126420

theorem cost_of_playing_cards 
  (allowance_each : ℕ)
  (combined_allowance : ℕ)
  (sticker_box_cost : ℕ)
  (number_of_sticker_packs : ℕ)
  (number_of_packs_Dora_got : ℕ)
  (cost_of_playing_cards : ℕ)
  (h1 : allowance_each = 9)
  (h2 : combined_allowance = allowance_each * 2)
  (h3 : sticker_box_cost = 2)
  (h4 : number_of_packs_Dora_got = 2)
  (h5 : number_of_sticker_packs = number_of_packs_Dora_got * 2)
  (h6 : combined_allowance - number_of_sticker_packs * sticker_box_cost = cost_of_playing_cards) :
  cost_of_playing_cards = 10 :=
sorry

end NUMINAMATH_GPT_cost_of_playing_cards_l1264_126420


namespace NUMINAMATH_GPT_total_problems_l1264_126425

theorem total_problems (C W : ℕ) (h1 : 3 * C + 5 * W = 110) (h2 : C = 20) : C + W = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_problems_l1264_126425


namespace NUMINAMATH_GPT_print_time_l1264_126448

-- Conditions
def printer_pages_per_minute : ℕ := 25
def total_pages : ℕ := 350

-- Theorem
theorem print_time :
  (total_pages / printer_pages_per_minute : ℕ) = 14 :=
by sorry

end NUMINAMATH_GPT_print_time_l1264_126448


namespace NUMINAMATH_GPT_chores_per_week_l1264_126422

theorem chores_per_week :
  ∀ (cookie_per_chore : ℕ) 
    (total_money : ℕ) 
    (cost_per_pack : ℕ) 
    (cookies_per_pack : ℕ) 
    (weeks : ℕ)
    (chores_per_week : ℕ),
  cookie_per_chore = 3 →
  total_money = 15 →
  cost_per_pack = 3 →
  cookies_per_pack = 24 →
  weeks = 10 →
  chores_per_week = (total_money / cost_per_pack * cookies_per_pack / weeks) / cookie_per_chore →
  chores_per_week = 4 :=
by
  intros cookie_per_chore total_money cost_per_pack cookies_per_pack weeks chores_per_week
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_chores_per_week_l1264_126422


namespace NUMINAMATH_GPT_labor_budget_constraint_l1264_126437

-- Define the conditions
def wage_per_carpenter : ℕ := 50
def wage_per_mason : ℕ := 40
def labor_budget : ℕ := 2000
def num_carpenters (x : ℕ) := x
def num_masons (y : ℕ) := y

-- The proof statement
theorem labor_budget_constraint (x y : ℕ) 
    (hx : wage_per_carpenter * num_carpenters x + wage_per_mason * num_masons y ≤ labor_budget) : 
    5 * x + 4 * y ≤ 200 := 
by sorry

end NUMINAMATH_GPT_labor_budget_constraint_l1264_126437


namespace NUMINAMATH_GPT_triangle_area_l1264_126450

structure Point where
  x : ℝ
  y : ℝ

def area_triangle (A B C : Point) : ℝ := 
  0.5 * (B.x - A.x) * (C.y - A.y)

theorem triangle_area :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨8, 15⟩
  let C : Point := ⟨8, 0⟩
  area_triangle A B C = 60 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1264_126450


namespace NUMINAMATH_GPT_find_x_l1264_126426

def diamond (x y : ℤ) : ℤ := 3 * x - y^2

theorem find_x (x : ℤ) (h : diamond x 7 = 20) : x = 23 :=
sorry

end NUMINAMATH_GPT_find_x_l1264_126426


namespace NUMINAMATH_GPT_fraction_sum_ratio_l1264_126460

theorem fraction_sum_ratio :
  let A := (Finset.range 1002).sum (λ k => 1 / ((2 * k + 1) * (2 * k + 2)))
  let B := (Finset.range 1002).sum (λ k => 1 / ((1003 + k) * (2004 - k)))
  (A / B) = (3007 / 2) :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_ratio_l1264_126460


namespace NUMINAMATH_GPT_rectangular_field_area_l1264_126413

theorem rectangular_field_area (w l A : ℝ) 
  (h1 : l = 3 * w)
  (h2 : 2 * (w + l) = 80) :
  A = w * l → A = 300 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l1264_126413


namespace NUMINAMATH_GPT_remainder_of_122_div_20_l1264_126421

theorem remainder_of_122_div_20 :
  (∃ (q r : ℕ), 122 = 20 * q + r ∧ r < 20 ∧ q = 6) →
  r = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_122_div_20_l1264_126421


namespace NUMINAMATH_GPT_bob_total_investment_l1264_126412

variable (x : ℝ) -- the amount invested at 14%

noncomputable def total_investment_amount : ℝ :=
  let interest18 := 7000 * 0.18
  let interest14 := x * 0.14
  let total_interest := 3360
  let total_investment := 7000 + x
  total_investment

theorem bob_total_investment (h : 7000 * 0.18 + x * 0.14 = 3360) :
  total_investment_amount x = 22000 := by
  sorry

end NUMINAMATH_GPT_bob_total_investment_l1264_126412
