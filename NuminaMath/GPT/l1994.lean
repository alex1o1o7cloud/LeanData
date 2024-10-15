import Mathlib

namespace NUMINAMATH_GPT_parts_from_blanks_9_parts_from_blanks_14_blanks_needed_for_40_parts_l1994_199415

theorem parts_from_blanks_9 : ∀ (produced_parts : ℕ), produced_parts = 13 :=
by
  sorry

theorem parts_from_blanks_14 : ∀ (produced_parts : ℕ), produced_parts = 20 :=
by
  sorry

theorem blanks_needed_for_40_parts : ∀ (required_blanks : ℕ), required_blanks = 27 :=
by
  sorry

end NUMINAMATH_GPT_parts_from_blanks_9_parts_from_blanks_14_blanks_needed_for_40_parts_l1994_199415


namespace NUMINAMATH_GPT_sheets_in_backpack_l1994_199455

-- Definitions for the conditions
def total_sheets := 91
def desk_sheets := 50

-- Theorem statement with the goal
theorem sheets_in_backpack (total_sheets : ℕ) (desk_sheets : ℕ) (h1 : total_sheets = 91) (h2 : desk_sheets = 50) : 
  ∃ backpack_sheets : ℕ, backpack_sheets = total_sheets - desk_sheets ∧ backpack_sheets = 41 :=
by
  -- The proof is omitted here
  sorry

end NUMINAMATH_GPT_sheets_in_backpack_l1994_199455


namespace NUMINAMATH_GPT_log_equation_positive_x_l1994_199496

theorem log_equation_positive_x (x : ℝ) (hx : 0 < x) (hx1 : x ≠ 1) : 
  (Real.log x / Real.log 2) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 2 :=
by sorry

end NUMINAMATH_GPT_log_equation_positive_x_l1994_199496


namespace NUMINAMATH_GPT_football_game_initial_population_l1994_199429

theorem football_game_initial_population (B G : ℕ) (h1 : G = 240)
  (h2 : (3 / 4 : ℚ) * B + (7 / 8 : ℚ) * G = 480) : B + G = 600 :=
sorry

end NUMINAMATH_GPT_football_game_initial_population_l1994_199429


namespace NUMINAMATH_GPT_total_pokemon_cards_l1994_199480

-- Definitions based on conditions
def dozen := 12
def amount_per_person := 9 * dozen
def num_people := 4

-- Proposition to prove
theorem total_pokemon_cards :
  num_people * amount_per_person = 432 :=
by sorry

end NUMINAMATH_GPT_total_pokemon_cards_l1994_199480


namespace NUMINAMATH_GPT_trees_still_left_l1994_199433

theorem trees_still_left 
  (initial_trees : ℕ) 
  (trees_died : ℕ) 
  (trees_cut : ℕ) 
  (initial_trees_eq : initial_trees = 86) 
  (trees_died_eq : trees_died = 15) 
  (trees_cut_eq : trees_cut = 23) 
  : initial_trees - (trees_died + trees_cut) = 48 :=
by
  sorry

end NUMINAMATH_GPT_trees_still_left_l1994_199433


namespace NUMINAMATH_GPT_divisibility_by_30_l1994_199495

theorem divisibility_by_30 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_3 : p ≥ 3) : 30 ∣ (p^3 - 1) ↔ p % 15 = 1 := 
  sorry

end NUMINAMATH_GPT_divisibility_by_30_l1994_199495


namespace NUMINAMATH_GPT_sqrt_sixteen_l1994_199422

theorem sqrt_sixteen : ∃ x : ℝ, x^2 = 16 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sixteen_l1994_199422


namespace NUMINAMATH_GPT_range_of_p_l1994_199434

noncomputable def f (x p : ℝ) : ℝ := x - p/x + p/2

theorem range_of_p (p : ℝ) :
  (∀ x : ℝ, 1 < x → (1 + p / x^2) > 0) → p ≥ -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_p_l1994_199434


namespace NUMINAMATH_GPT_total_time_spent_l1994_199435

def chess_game_duration_hours : ℕ := 20
def chess_game_duration_minutes : ℕ := 15
def additional_analysis_time : ℕ := 22
def total_expected_time : ℕ := 1237

theorem total_time_spent : 
  (chess_game_duration_hours * 60 + chess_game_duration_minutes + additional_analysis_time) = total_expected_time :=
  by
    sorry

end NUMINAMATH_GPT_total_time_spent_l1994_199435


namespace NUMINAMATH_GPT_range_of_a_l1994_199452

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1994_199452


namespace NUMINAMATH_GPT_steven_ships_boxes_l1994_199497

-- Translate the conditions into Lean definitions and state the theorem
def truck_weight_limit : ℕ := 2000
def truck_count : ℕ := 3
def pair_weight : ℕ := 10 + 40
def boxes_per_pair : ℕ := 2

theorem steven_ships_boxes :
  ((truck_weight_limit / pair_weight) * boxes_per_pair * truck_count) = 240 := by
  sorry

end NUMINAMATH_GPT_steven_ships_boxes_l1994_199497


namespace NUMINAMATH_GPT_length_of_major_axis_l1994_199424

def ellipse_length_major_axis (a b : ℝ) : ℝ := 2 * a

theorem length_of_major_axis : ellipse_length_major_axis 4 1 = 8 :=
by
  unfold ellipse_length_major_axis
  norm_num

end NUMINAMATH_GPT_length_of_major_axis_l1994_199424


namespace NUMINAMATH_GPT_number_and_its_square_root_l1994_199472

theorem number_and_its_square_root (x : ℝ) (h : x + 10 * Real.sqrt x = 39) : x = 9 :=
sorry

end NUMINAMATH_GPT_number_and_its_square_root_l1994_199472


namespace NUMINAMATH_GPT_abs_sum_of_factors_of_quadratic_l1994_199402

variable (h b c d : ℤ)

theorem abs_sum_of_factors_of_quadratic :
  (∀ x : ℤ, 6 * x * x + x - 12 = (h * x + b) * (c * x + d)) →
  (|h| + |b| + |c| + |d| = 12) :=
by
  sorry

end NUMINAMATH_GPT_abs_sum_of_factors_of_quadratic_l1994_199402


namespace NUMINAMATH_GPT_extra_men_needed_l1994_199484

theorem extra_men_needed
  (total_length : ℕ) (total_days : ℕ) (initial_men : ℕ)
  (completed_days : ℕ) (completed_work : ℕ) (remaining_work : ℕ)
  (remaining_days : ℕ) (total_man_days_needed : ℕ)
  (number_of_men_needed : ℕ) (extra_men_needed : ℕ)
  (h1 : total_length = 10)
  (h2 : total_days = 60)
  (h3 : initial_men = 30)
  (h4 : completed_days = 20)
  (h5 : completed_work = 2)
  (h6 : remaining_work = total_length - completed_work)
  (h7 : remaining_days = total_days - completed_days)
  (h8 : total_man_days_needed = remaining_work * (completed_days * initial_men) / completed_work)
  (h9 : number_of_men_needed = total_man_days_needed / remaining_days)
  (h10 : extra_men_needed = number_of_men_needed - initial_men)
  : extra_men_needed = 30 :=
by sorry

end NUMINAMATH_GPT_extra_men_needed_l1994_199484


namespace NUMINAMATH_GPT_solve_inequality_l1994_199432

-- We will define the conditions and corresponding solution sets
def solution_set (a x : ℝ) : Prop :=
  (a < -1 ∧ (x > -a ∨ x < 1)) ∨
  (a = -1 ∧ x ≠ 1) ∨
  (a > -1 ∧ (x < -a ∨ x > 1))

theorem solve_inequality (a x : ℝ) :
  (x - 1) * (x + a) > 0 ↔ solution_set a x :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1994_199432


namespace NUMINAMATH_GPT_A_can_complete_work_in_4_days_l1994_199469

-- Definitions based on conditions
def work_done_in_one_day (days : ℕ) : ℚ := 1 / days

def combined_work_done_in_two_days (a b c : ℕ) : ℚ :=
  work_done_in_one_day a + work_done_in_one_day b + work_done_in_one_day c

-- Theorem statement based on the problem
theorem A_can_complete_work_in_4_days (A B C : ℕ) 
  (hB : B = 8) (hC : C = 8) 
  (h_combined : combined_work_done_in_two_days A B C = work_done_in_one_day 2) :
  A = 4 :=
sorry

end NUMINAMATH_GPT_A_can_complete_work_in_4_days_l1994_199469


namespace NUMINAMATH_GPT_domain_of_log_function_l1994_199405

theorem domain_of_log_function (x : ℝ) :
  (5 - x > 0) ∧ (x - 2 > 0) ∧ (x - 2 ≠ 1) ↔ (2 < x ∧ x < 3) ∨ (3 < x ∧ x < 5) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_log_function_l1994_199405


namespace NUMINAMATH_GPT_inequality_proof_l1994_199428

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
    (b^2 / a + a^2 / b) ≥ (a + b) := 
    sorry

end NUMINAMATH_GPT_inequality_proof_l1994_199428


namespace NUMINAMATH_GPT_perpendicular_graphs_solve_a_l1994_199438

theorem perpendicular_graphs_solve_a (a : ℝ) : 
  (∀ x y : ℝ, 2 * y + x + 3 = 0 → 3 * y + a * x + 2 = 0 → 
  ∀ m1 m2 : ℝ, (y = m1 * x + b1 → m1 = -1 / 2) →
  (y = m2 * x + b2 → m2 = -a / 3) →
  m1 * m2 = -1) → a = -6 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_graphs_solve_a_l1994_199438


namespace NUMINAMATH_GPT_ferris_wheel_seats_l1994_199404

def number_of_people_per_seat := 6
def total_number_of_people := 84

def number_of_seats := total_number_of_people / number_of_people_per_seat

theorem ferris_wheel_seats : number_of_seats = 14 := by
  sorry

end NUMINAMATH_GPT_ferris_wheel_seats_l1994_199404


namespace NUMINAMATH_GPT_find_k_range_l1994_199498

theorem find_k_range (k : ℝ) : 
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ -6 < k ∧ k < -2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_range_l1994_199498


namespace NUMINAMATH_GPT_intersection_is_2_l1994_199442

noncomputable def M : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def N : Set ℝ := {x | x^2 ≥ 2 * x}
noncomputable def intersection : Set ℝ := M ∩ N

theorem intersection_is_2 : intersection = {2} := by
  sorry

end NUMINAMATH_GPT_intersection_is_2_l1994_199442


namespace NUMINAMATH_GPT_solution_set_l1994_199425

/-- Definition: integer solutions (a, b, c) with c ≤ 94 that satisfy the equation -/
def int_solutions (a b c : ℤ) : Prop :=
  c ≤ 94 ∧ (a + Real.sqrt c)^2 + (b + Real.sqrt c)^2 = 60 + 20 * Real.sqrt c

/-- Proposition: The integer solutions (a, b, c) that satisfy the equation are exactly these -/
theorem solution_set :
  { (a, b, c) : ℤ × ℤ × ℤ  | int_solutions a b c } =
  { (3, 7, 41), (4, 6, 44), (5, 5, 45), (6, 4, 44), (7, 3, 41) } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1994_199425


namespace NUMINAMATH_GPT_max_acute_angles_l1994_199436

theorem max_acute_angles (n : ℕ) : 
  ∃ k : ℕ, k ≤ (2 * n / 3) + 1 :=
sorry

end NUMINAMATH_GPT_max_acute_angles_l1994_199436


namespace NUMINAMATH_GPT_mary_money_left_l1994_199459

theorem mary_money_left (p : ℝ) : 50 - (4 * p + 2 * p + 4 * p) = 50 - 10 * p := 
by 
  sorry

end NUMINAMATH_GPT_mary_money_left_l1994_199459


namespace NUMINAMATH_GPT_graph_of_transformed_function_l1994_199418

theorem graph_of_transformed_function
  (f : ℝ → ℝ)
  (hf : f⁻¹ 1 = 0) :
  f (1 - 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_graph_of_transformed_function_l1994_199418


namespace NUMINAMATH_GPT_max_value_A_l1994_199448

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x * Real.sin (x + Real.pi / 6)

theorem max_value_A (A : ℝ) (hA : A = Real.pi / 6) : 
  ∀ x : ℝ, f x ≤ f A :=
sorry

end NUMINAMATH_GPT_max_value_A_l1994_199448


namespace NUMINAMATH_GPT_series_sum_eq_one_sixth_l1994_199490

noncomputable def a (n : ℕ) : ℝ := 2^n / (7^(2^n) + 1)

theorem series_sum_eq_one_sixth :
  (∑' (n : ℕ), a n) = 1 / 6 :=
sorry

end NUMINAMATH_GPT_series_sum_eq_one_sixth_l1994_199490


namespace NUMINAMATH_GPT_best_fitting_model_is_model1_l1994_199406

noncomputable def model1_R2 : ℝ := 0.98
noncomputable def model2_R2 : ℝ := 0.80
noncomputable def model3_R2 : ℝ := 0.54
noncomputable def model4_R2 : ℝ := 0.35

theorem best_fitting_model_is_model1 :
  model1_R2 > model2_R2 ∧ model1_R2 > model3_R2 ∧ model1_R2 > model4_R2 :=
by
  sorry

end NUMINAMATH_GPT_best_fitting_model_is_model1_l1994_199406


namespace NUMINAMATH_GPT_no_real_values_of_p_for_equal_roots_l1994_199426

theorem no_real_values_of_p_for_equal_roots (p : ℝ) : ¬ ∃ (p : ℝ), (p^2 - 2*p + 5 = 0) :=
by sorry

end NUMINAMATH_GPT_no_real_values_of_p_for_equal_roots_l1994_199426


namespace NUMINAMATH_GPT_cos_omega_x_3_zeros_interval_l1994_199449

theorem cos_omega_x_3_zeros_interval (ω : ℝ) (hω : ω > 0)
  (h3_zeros : ∃ a b c : ℝ, (0 ≤ a ∧ a ≤ 2 * Real.pi) ∧
    (0 ≤ b ∧ b ≤ 2 * Real.pi ∧ b ≠ a) ∧
    (0 ≤ c ∧ c ≤ 2 * Real.pi ∧ c ≠ a ∧ c ≠ b) ∧
    (∀ x : ℝ, (0 ≤ x ∧ x ≤ 2 * Real.pi) →
      (Real.cos (ω * x) - 1 = 0 ↔ x = a ∨ x = b ∨ x = c))) :
  2 ≤ ω ∧ ω < 3 :=
sorry

end NUMINAMATH_GPT_cos_omega_x_3_zeros_interval_l1994_199449


namespace NUMINAMATH_GPT_twelve_percent_greater_than_80_l1994_199475

theorem twelve_percent_greater_than_80 (x : ℝ) (h : x = 80 + 0.12 * 80) : x = 89.6 :=
by
  sorry

end NUMINAMATH_GPT_twelve_percent_greater_than_80_l1994_199475


namespace NUMINAMATH_GPT_carson_total_seed_fertilizer_l1994_199420

-- Definitions based on the conditions
variable (F S : ℝ)
variable (h_seed : S = 45)
variable (h_relation : S = 3 * F)

-- Theorem stating the total amount of seed and fertilizer used
theorem carson_total_seed_fertilizer : S + F = 60 := by
  -- Use the given conditions to relate and calculate the total
  sorry

end NUMINAMATH_GPT_carson_total_seed_fertilizer_l1994_199420


namespace NUMINAMATH_GPT_triangle_base_l1994_199488

theorem triangle_base (h : ℝ) (A : ℝ) (b : ℝ) (h_eq : h = 10) (A_eq : A = 46) (area_eq : A = (b * h) / 2) : b = 9.2 :=
by
  -- sorry to be replaced with the actual proof
  sorry

end NUMINAMATH_GPT_triangle_base_l1994_199488


namespace NUMINAMATH_GPT_inequality_solution_minimum_value_l1994_199407

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem inequality_solution :
  {x : ℝ | f x > 7} = {x | x > 4 ∨ x < -3} :=
by
  sorry

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : ∀ x, f x ≥ m + n) :
  m + n = 3 →
  (m^2 + n^2 ≥ 9 / 2 ∧ (m = 3 / 2 ∧ n = 3 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_minimum_value_l1994_199407


namespace NUMINAMATH_GPT_average_fruits_per_basket_is_correct_l1994_199427

noncomputable def average_fruits_per_basket : ℕ :=
  let basket_A := 15
  let basket_B := 30
  let basket_C := 20
  let basket_D := 25
  let basket_E := 35
  let total_fruits := basket_A + basket_B + basket_C + basket_D + basket_E
  let number_of_baskets := 5
  total_fruits / number_of_baskets

theorem average_fruits_per_basket_is_correct : average_fruits_per_basket = 25 := by
  unfold average_fruits_per_basket
  rfl

end NUMINAMATH_GPT_average_fruits_per_basket_is_correct_l1994_199427


namespace NUMINAMATH_GPT_smallest_n_condition_l1994_199491

theorem smallest_n_condition (n : ℕ) : (4 * n) ∣ (n^2) ∧ (5 * n) ∣ (u^3) → n = 100 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_condition_l1994_199491


namespace NUMINAMATH_GPT_exp_increasing_a_lt_zero_l1994_199423

theorem exp_increasing_a_lt_zero (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → (1 - a) ^ x1 < (1 - a) ^ x2) : a < 0 := 
sorry

end NUMINAMATH_GPT_exp_increasing_a_lt_zero_l1994_199423


namespace NUMINAMATH_GPT_evaluate_expression_at_x_eq_2_l1994_199450

theorem evaluate_expression_at_x_eq_2 : (3 * 2 + 4)^2 - 10 * 2 = 80 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_x_eq_2_l1994_199450


namespace NUMINAMATH_GPT_find_probabilities_l1994_199411

theorem find_probabilities (p_1 p_3 : ℝ)
  (h1 : p_1 + 0.15 + p_3 + 0.25 + 0.35 = 1)
  (h2 : p_3 = 4 * p_1) :
  p_1 = 0.05 ∧ p_3 = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_find_probabilities_l1994_199411


namespace NUMINAMATH_GPT_infinite_triangular_pairs_l1994_199440

theorem infinite_triangular_pairs : ∃ (a_i b_i : ℕ → ℕ), (∀ m : ℕ, ∃ n : ℕ, m = n * (n + 1) / 2 ↔ ∃ k : ℕ, a_i k * m + b_i k = k * (k + 1) / 2) ∧ ∀ j : ℕ, ∃ k : ℕ, k > j :=
by {
  sorry
}

end NUMINAMATH_GPT_infinite_triangular_pairs_l1994_199440


namespace NUMINAMATH_GPT_right_triangle_side_length_l1994_199454

theorem right_triangle_side_length (c a b : ℕ) (h1 : c = 5) (h2 : a = 3) (h3 : c^2 = a^2 + b^2) : b = 4 :=
  by
  sorry

end NUMINAMATH_GPT_right_triangle_side_length_l1994_199454


namespace NUMINAMATH_GPT_alex_total_earnings_l1994_199493

def total_earnings (hours_w1 hours_w2 wage : ℕ) : ℕ :=
  (hours_w1 + hours_w2) * wage

theorem alex_total_earnings
  (hours_w1 hours_w2 wage : ℕ)
  (h1 : hours_w1 = 28)
  (h2 : hours_w2 = hours_w1 - 10)
  (h3 : wage * 10 = 80) :
  total_earnings hours_w1 hours_w2 wage = 368 :=
by
  sorry

end NUMINAMATH_GPT_alex_total_earnings_l1994_199493


namespace NUMINAMATH_GPT_intersection_points_l1994_199414

theorem intersection_points (g : ℝ → ℝ) (hg_inv : Function.Injective g) : 
  ∃ n, n = 3 ∧ ∀ x, g (x^3) = g (x^5) ↔ x = 0 ∨ x = 1 ∨ x = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_points_l1994_199414


namespace NUMINAMATH_GPT_total_books_on_shelves_l1994_199473

def num_shelves : ℕ := 520
def books_per_shelf : ℝ := 37.5

theorem total_books_on_shelves : num_shelves * books_per_shelf = 19500 :=
by
  sorry

end NUMINAMATH_GPT_total_books_on_shelves_l1994_199473


namespace NUMINAMATH_GPT_interval_proof_l1994_199416

noncomputable def valid_interval (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → (5 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y

theorem interval_proof : ∀ x : ℝ, valid_interval x ↔ (0 ≤ x ∧ x < 4) :=
by
  sorry

end NUMINAMATH_GPT_interval_proof_l1994_199416


namespace NUMINAMATH_GPT_total_invested_amount_l1994_199412

theorem total_invested_amount :
  ∃ (A B : ℝ), (A = 3000 ∧ B = 5000 ∧ 
  0.085 * A + 0.064 * B = 575 ∧ A + B = 8000)
  ∨ 
  (A = 5000 ∧ B = 3000 ∧ 
  0.085 * A + 0.064 * B = 575 ∧ A + B = 8000) :=
sorry

end NUMINAMATH_GPT_total_invested_amount_l1994_199412


namespace NUMINAMATH_GPT_polynomial_expansion_identity_l1994_199463

theorem polynomial_expansion_identity
  (a a1 a3 a4 a5 : ℝ)
  (h : (a - x)^5 = a + a1 * x + 80 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) :
  a + a1 + 80 + a3 + a4 + a5 = 1 := 
sorry

end NUMINAMATH_GPT_polynomial_expansion_identity_l1994_199463


namespace NUMINAMATH_GPT_measure_of_angle_is_135_l1994_199417

noncomputable def degree_measure_of_angle (x : ℝ) : Prop :=
  (x = 3 * (180 - x)) ∧ (2 * x + (180 - x) = 180) -- Combining all conditions

theorem measure_of_angle_is_135 (x : ℝ) (h : degree_measure_of_angle x) : x = 135 :=
by sorry

end NUMINAMATH_GPT_measure_of_angle_is_135_l1994_199417


namespace NUMINAMATH_GPT_percentage_of_ll_watchers_l1994_199410

theorem percentage_of_ll_watchers 
  (T : ℕ) 
  (IS : ℕ) 
  (ME : ℕ) 
  (E2 : ℕ) 
  (A3 : ℕ) 
  (total_residents : T = 600)
  (is_watchers : IS = 210)
  (me_watchers : ME = 300)
  (e2_watchers : E2 = 108)
  (a3_watchers : A3 = 21)
  (at_least_one_show : IS + (by sorry) + ME - E2 + A3 = T) :
  ∃ x : ℕ, (x * 100 / T) = 115 :=
by sorry

end NUMINAMATH_GPT_percentage_of_ll_watchers_l1994_199410


namespace NUMINAMATH_GPT_warehouse_rental_comparison_purchase_vs_rent_comparison_l1994_199430

-- Define the necessary constants and conditions
def monthly_cost_first : ℕ := 50000
def monthly_cost_second : ℕ := 10000
def moving_cost : ℕ := 70000
def months_in_year : ℕ := 12
def purchase_cost : ℕ := 2000000
def duration_installments : ℕ := 3 * 12 -- 3 years in months
def worst_case_prob : ℕ := 50

-- Question (a)
theorem warehouse_rental_comparison
  (annual_cost_first : ℕ := monthly_cost_first * months_in_year)
  (cost_second_4months : ℕ := monthly_cost_second * 4)
  (cost_switching : ℕ := moving_cost)
  (cost_first_8months : ℕ := monthly_cost_first * 8)
  (worst_case_cost_second : ℕ := cost_second_4months + cost_first_8months + cost_switching) :
  annual_cost_first > worst_case_cost_second :=
by
  sorry

-- Question (b)
theorem purchase_vs_rent_comparison
  (total_rent_cost_4years : ℕ := 4 * annual_cost_first + worst_case_cost_second)
  (total_purchase_cost : ℕ := purchase_cost) :
  total_rent_cost_4years > total_purchase_cost :=
by
  sorry

end NUMINAMATH_GPT_warehouse_rental_comparison_purchase_vs_rent_comparison_l1994_199430


namespace NUMINAMATH_GPT_probability_sin_cos_in_range_l1994_199461

noncomputable def probability_sin_cos_interval : ℝ :=
  let interval_length := (Real.pi / 2 + Real.pi / 6)
  let valid_length := (Real.pi / 2 - 0)
  valid_length / interval_length

theorem probability_sin_cos_in_range :
  probability_sin_cos_interval = 3 / 4 :=
sorry

end NUMINAMATH_GPT_probability_sin_cos_in_range_l1994_199461


namespace NUMINAMATH_GPT_ratio_population_XZ_l1994_199464

variable (Population : Type) [Field Population]
variable (Z : Population) -- Population of City Z
variable (Y : Population) -- Population of City Y
variable (X : Population) -- Population of City X

-- Conditions
def population_Y : Y = 2 * Z := sorry
def population_X : X = 7 * Y := sorry

-- Theorem stating the ratio of populations
theorem ratio_population_XZ : (X / Z) = 14 := by
  -- The proof will use the conditions population_Y and population_X
  sorry

end NUMINAMATH_GPT_ratio_population_XZ_l1994_199464


namespace NUMINAMATH_GPT_triangle_is_acute_l1994_199466

-- Define the condition that the angles have a ratio of 2:3:4
def angle_ratio_cond (a b c : ℝ) : Prop :=
  a / b = 2 / 3 ∧ b / c = 3 / 4

-- Define the sum of the angles in a triangle
def angle_sum_cond (a b c : ℝ) : Prop :=
  a + b + c = 180

-- The proof problem stating that triangle with angles in ratio 2:3:4 is acute
theorem triangle_is_acute (a b c : ℝ) (h_ratio : angle_ratio_cond a b c) (h_sum : angle_sum_cond a b c) : 
  a < 90 ∧ b < 90 ∧ c < 90 := 
by
  sorry

end NUMINAMATH_GPT_triangle_is_acute_l1994_199466


namespace NUMINAMATH_GPT_no_such_function_l1994_199481

theorem no_such_function :
  ¬ (∃ f : ℕ → ℕ, ∀ n ≥ 2, f (f (n - 1)) = f (n + 1) - f (n)) :=
sorry

end NUMINAMATH_GPT_no_such_function_l1994_199481


namespace NUMINAMATH_GPT_remainder_when_divided_by_15_l1994_199400

def N (k : ℤ) : ℤ := 35 * k + 25

theorem remainder_when_divided_by_15 (k : ℤ) : (N k) % 15 = 10 := 
by 
  -- proof would go here
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_15_l1994_199400


namespace NUMINAMATH_GPT_smallest_x_undefined_l1994_199486

theorem smallest_x_undefined : ∃ x : ℝ, (10 * x^2 - 90 * x + 20 = 0) ∧ x = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_smallest_x_undefined_l1994_199486


namespace NUMINAMATH_GPT_find_triangle_areas_l1994_199465

variables (A B C D : Point)
variables (S_ABC S_ACD S_ABD S_BCD : ℝ)

def quadrilateral_area (S_ABC S_ACD S_ABD S_BCD : ℝ) : Prop :=
  S_ABC + S_ACD + S_ABD + S_BCD = 25

def conditions (S_ABC S_ACD S_ABD S_BCD : ℝ) : Prop :=
  (S_ABC = 2 * S_BCD) ∧ (S_ABD = 3 * S_ACD)

theorem find_triangle_areas
  (S_ABC S_ACD S_ABD S_BCD : ℝ) :
  quadrilateral_area S_ABC S_ACD S_ABD S_BCD →
  conditions S_ABC S_ACD S_ABD S_BCD →
  S_ABC = 10 ∧ S_ACD = 5 ∧ S_ABD = 15 ∧ S_BCD = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_triangle_areas_l1994_199465


namespace NUMINAMATH_GPT_find_lawn_length_l1994_199431

theorem find_lawn_length
  (width_lawn : ℕ)
  (road_width : ℕ)
  (cost_total : ℕ)
  (cost_per_sqm : ℕ)
  (total_area_roads : ℕ)
  (area_roads_length : ℕ)
  (area_roads_breadth : ℕ)
  (length_lawn : ℕ) :
  width_lawn = 60 →
  road_width = 10 →
  cost_total = 3600 →
  cost_per_sqm = 3 →
  total_area_roads = cost_total / cost_per_sqm →
  area_roads_length = road_width * length_lawn →
  area_roads_breadth = road_width * (width_lawn - road_width) →
  total_area_roads = area_roads_length + area_roads_breadth →
  length_lawn = 70 :=
by
  intros h_width_lawn h_road_width h_cost_total h_cost_per_sqm h_total_area_roads h_area_roads_length h_area_roads_breadth h_total_area_roads_eq
  sorry

end NUMINAMATH_GPT_find_lawn_length_l1994_199431


namespace NUMINAMATH_GPT_trigonometric_identity_l1994_199446

theorem trigonometric_identity (α x : ℝ) (h₁ : 5 * Real.cos α = x) (h₂ : x ^ 2 + 16 = 25) (h₃ : α > Real.pi / 2 ∧ α < Real.pi):
  x = -3 ∧ Real.tan α = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1994_199446


namespace NUMINAMATH_GPT_D_time_to_complete_job_l1994_199421

-- Let A_rate be the rate at which A works (jobs per hour)
-- Let D_rate be the rate at which D works (jobs per hour)
def A_rate : ℚ := 1 / 3
def combined_rate : ℚ := 1 / 2

-- We need to prove that D_rate, the rate at which D works alone, is 1/6 jobs per hour
def D_rate := 1 / 6

-- And thus, that D can complete the job in 6 hours
theorem D_time_to_complete_job :
  (A_rate + D_rate = combined_rate) → (1 / D_rate) = 6 :=
by
  sorry

end NUMINAMATH_GPT_D_time_to_complete_job_l1994_199421


namespace NUMINAMATH_GPT_election_problem_l1994_199403

theorem election_problem :
  ∃ (n : ℕ), n = (10 * 9) * Nat.choose 8 3 :=
  by
  use 5040
  sorry

end NUMINAMATH_GPT_election_problem_l1994_199403


namespace NUMINAMATH_GPT_cone_height_l1994_199458

theorem cone_height (h : ℝ) (r : ℝ) 
  (volume_eq : (1/3) * π * r^2 * h = 19683 * π) 
  (isosceles_right_triangle : h = r) : 
  h = 39.0 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_cone_height_l1994_199458


namespace NUMINAMATH_GPT_unique_real_solution_k_l1994_199477

theorem unique_real_solution_k (k : ℝ) :
  ∃! x : ℝ, (3 * x + 8) * (x - 6) = -62 + k * x ↔ k = -10 + 12 * Real.sqrt 1.5 ∨ k = -10 - 12 * Real.sqrt 1.5 := by
  sorry

end NUMINAMATH_GPT_unique_real_solution_k_l1994_199477


namespace NUMINAMATH_GPT_quadratic_roots_real_distinct_l1994_199457

theorem quadratic_roots_real_distinct (k : ℝ) (h : k < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + x1 + k - 1 = 0) ∧ (x2^2 + x2 + k - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_real_distinct_l1994_199457


namespace NUMINAMATH_GPT_circumference_of_minor_arc_l1994_199437

-- Given:
-- 1. Three points (D, E, F) are on a circle with radius 25
-- 2. The angle ∠EFD = 120°

-- We need to prove that the length of the minor arc DE is 50π / 3
theorem circumference_of_minor_arc 
  (D E F : Point) 
  (r : ℝ) (h : r = 25) 
  (angleEFD : ℝ) 
  (hAngle : angleEFD = 120) 
  (circumference : ℝ) 
  (hCircumference : circumference = 2 * Real.pi * r) :
  arc_length_DE = 50 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_circumference_of_minor_arc_l1994_199437


namespace NUMINAMATH_GPT_zip_code_relationship_l1994_199439

theorem zip_code_relationship (A B C D E : ℕ) 
(h1 : A + B + C + D + E = 10) 
(h2 : C = 0) 
(h3 : D = 2 * A) 
(h4 : D + E = 8) : 
A + B = 2 :=
sorry

end NUMINAMATH_GPT_zip_code_relationship_l1994_199439


namespace NUMINAMATH_GPT_f_identically_zero_l1994_199453

open Real

-- Define the function f and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Given conditions
axiom func_eqn (a b : ℝ) : f (a * b) = a * f b + b * f a 
axiom func_bounded (x : ℝ) : |f x| ≤ 1

-- Goal: Prove that f is identically zero
theorem f_identically_zero : ∀ x : ℝ, f x = 0 := 
by
  sorry

end NUMINAMATH_GPT_f_identically_zero_l1994_199453


namespace NUMINAMATH_GPT_calc_a8_l1994_199478

variable {a : ℕ+ → ℕ}

-- Conditions
axiom recur_relation : ∀ (p q : ℕ+), a (p + q) = a p * a q
axiom initial_condition : a 2 = 2

-- Proof statement
theorem calc_a8 : a 8 = 16 := by
  sorry

end NUMINAMATH_GPT_calc_a8_l1994_199478


namespace NUMINAMATH_GPT_concert_duration_is_805_l1994_199451

def hours_to_minutes (hours : ℕ) : ℕ :=
  hours * 60

def total_duration (hours : ℕ) (extra_minutes : ℕ) : ℕ :=
  hours_to_minutes hours + extra_minutes

theorem concert_duration_is_805 : total_duration 13 25 = 805 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_concert_duration_is_805_l1994_199451


namespace NUMINAMATH_GPT_floor_length_is_twelve_l1994_199408

-- Definitions based on the conditions
def floor_width := 10
def strip_width := 3
def rug_area := 24

-- Problem statement
theorem floor_length_is_twelve (L : ℕ) 
  (h1 : rug_area = (L - 2 * strip_width) * (floor_width - 2 * strip_width)) :
  L = 12 := 
sorry

end NUMINAMATH_GPT_floor_length_is_twelve_l1994_199408


namespace NUMINAMATH_GPT_f_analytical_expression_l1994_199445

noncomputable def f (x : ℝ) : ℝ := (2^(x + 1) - 2^(-x)) / 3

theorem f_analytical_expression :
  ∀ x : ℝ, f (-x) + 2 * f x = 2^x :=
by
  sorry

end NUMINAMATH_GPT_f_analytical_expression_l1994_199445


namespace NUMINAMATH_GPT_solve_quartic_equation_l1994_199460

theorem solve_quartic_equation :
  (∃ x : ℝ, x > 0 ∧ 
    (1 / 3) * (4 * x ^ 2 - 3) = (x ^ 2 - 60 * x - 12) * (x ^ 2 + 30 * x + 6) ∧ 
    ∃ y1 y2 : ℝ, y1 + y2 = 60 ∧ (x^2 - 60 * x - 12 = 0)) → 
    x = 30 + Real.sqrt 912 :=
sorry

end NUMINAMATH_GPT_solve_quartic_equation_l1994_199460


namespace NUMINAMATH_GPT_max_pasture_area_maximization_l1994_199476

noncomputable def max_side_length (fence_cost_per_foot : ℕ) (total_cost : ℕ) : ℕ :=
  let total_length := total_cost / fence_cost_per_foot
  let x := total_length / 4
  2 * x

theorem max_pasture_area_maximization :
  max_side_length 8 1920 = 120 :=
by
  sorry

end NUMINAMATH_GPT_max_pasture_area_maximization_l1994_199476


namespace NUMINAMATH_GPT_quadratic_coefficient_a_l1994_199489

theorem quadratic_coefficient_a (a b c : ℝ) :
  (2 = 9 * a - 3 * b + c) ∧
  (2 = 9 * a + 3 * b + c) ∧
  (-6 = 4 * a + 2 * b + c) →
  a = 8 / 5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_coefficient_a_l1994_199489


namespace NUMINAMATH_GPT_no_distinct_ordered_pairs_l1994_199413

theorem no_distinct_ordered_pairs (x y : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) :
  (x^2 * y^2)^2 - 14 * x^2 * y^2 + 49 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_no_distinct_ordered_pairs_l1994_199413


namespace NUMINAMATH_GPT_friend_wants_to_take_5_marbles_l1994_199468

theorem friend_wants_to_take_5_marbles
  (total_marbles : ℝ)
  (clear_marbles : ℝ)
  (black_marbles : ℝ)
  (other_marbles : ℝ)
  (friend_marbles : ℝ)
  (h1 : clear_marbles = 0.4 * total_marbles)
  (h2 : black_marbles = 0.2 * total_marbles)
  (h3 : other_marbles = total_marbles - clear_marbles - black_marbles)
  (h4 : friend_marbles = 2)
  (friend_total_marbles : ℝ)
  (h5 : friend_marbles = 0.4 * friend_total_marbles) :
  friend_total_marbles = 5 := by
  sorry

end NUMINAMATH_GPT_friend_wants_to_take_5_marbles_l1994_199468


namespace NUMINAMATH_GPT_abc_eq_bc_l1994_199471

theorem abc_eq_bc (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) 
(h : 4 * a * b * c * (a + b + c) = (a + b)^2 * (a + c)^2) :
  a * (a + b + c) = b * c :=
by 
  sorry

end NUMINAMATH_GPT_abc_eq_bc_l1994_199471


namespace NUMINAMATH_GPT_triangle_area_l1994_199487

/-
A triangle with side lengths in the ratio 4:5:6 is inscribed in a circle of radius 5.
We need to prove that the area of the triangle is 250/9.
-/

theorem triangle_area (x : ℝ) (r : ℝ) (h_r : r = 5) (h_ratio : 6 * x = 2 * r) :
  (1 / 2) * (4 * x) * (5 * x) = 250 / 9 := by 
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_triangle_area_l1994_199487


namespace NUMINAMATH_GPT_ratio_of_points_l1994_199401

theorem ratio_of_points (B J S : ℕ) 
  (h1 : B = J + 20) 
  (h2 : B + J + S = 160) 
  (h3 : B = 45) : 
  B / S = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_ratio_of_points_l1994_199401


namespace NUMINAMATH_GPT_joe_has_more_shirts_l1994_199443

theorem joe_has_more_shirts (alex_shirts : ℕ) (ben_shirts : ℕ) (ben_joe_diff : ℕ)
  (h_a : alex_shirts = 4)
  (h_b : ben_shirts = 15)
  (h_bj : ben_shirts = joe_shirts + ben_joe_diff)
  (h_bj_diff : ben_joe_diff = 8) :
  joe_shirts - alex_shirts = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_joe_has_more_shirts_l1994_199443


namespace NUMINAMATH_GPT_non_negative_solutions_l1994_199499

theorem non_negative_solutions (x : ℕ) (h : 1 + x ≥ 2 * x - 1) : x = 0 ∨ x = 1 ∨ x = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_non_negative_solutions_l1994_199499


namespace NUMINAMATH_GPT_tank_fill_time_with_leak_l1994_199492

theorem tank_fill_time_with_leak 
  (pump_fill_time : ℕ) (leak_empty_time : ℕ) (effective_fill_time : ℕ)
  (hp : pump_fill_time = 5)
  (hl : leak_empty_time = 10)
  (he : effective_fill_time = 10) : effective_fill_time = 10 :=
by
  sorry

end NUMINAMATH_GPT_tank_fill_time_with_leak_l1994_199492


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l1994_199485

noncomputable def triangle_hypotenuse (a b c : ℝ) : Prop :=
(a + b + c = 40) ∧
(a * b = 48) ∧
(a^2 + b^2 = c^2) ∧
(c = 18.8)

theorem right_triangle_hypotenuse :
  ∃ (a b c : ℝ), triangle_hypotenuse a b c :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l1994_199485


namespace NUMINAMATH_GPT_ratio_arithmetic_sequence_last_digit_l1994_199479

def is_ratio_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, n > 0 → (a (n + 2) * a n) = (a (n + 1) ^ 2) * d

theorem ratio_arithmetic_sequence_last_digit :
  ∃ a : ℕ → ℕ, is_ratio_arithmetic_sequence a 1 ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 2 ∧
  (a 2009 / a 2006) % 10 = 6 :=
sorry

end NUMINAMATH_GPT_ratio_arithmetic_sequence_last_digit_l1994_199479


namespace NUMINAMATH_GPT_keystone_arch_larger_angle_l1994_199467

def isosceles_trapezoid_larger_angle (n : ℕ) : Prop :=
  n = 10 → ∃ (x : ℝ), x = 99

theorem keystone_arch_larger_angle :
  isosceles_trapezoid_larger_angle 10 :=
by
  sorry

end NUMINAMATH_GPT_keystone_arch_larger_angle_l1994_199467


namespace NUMINAMATH_GPT_odds_against_horse_C_winning_l1994_199456

theorem odds_against_horse_C_winning (odds_A : ℚ) (odds_B : ℚ) (odds_C : ℚ) 
  (cond1 : odds_A = 5 / 2) 
  (cond2 : odds_B = 3 / 1) 
  (race_condition : odds_C = 1 - ((2 / (5 + 2)) + (1 / (3 + 1))))
  : odds_C / (1 - odds_C) = 15 / 13 := 
sorry

end NUMINAMATH_GPT_odds_against_horse_C_winning_l1994_199456


namespace NUMINAMATH_GPT_compute_XY_l1994_199444

theorem compute_XY (BC AC AB : ℝ) (hBC : BC = 30) (hAC : AC = 50) (hAB : AB = 60) :
  let XA := (BC * AB) / AC 
  let AY := (BC * AC) / AB
  let XY := XA + AY
  XY = 61 :=
by
  sorry

end NUMINAMATH_GPT_compute_XY_l1994_199444


namespace NUMINAMATH_GPT_min_removed_numbers_l1994_199470

theorem min_removed_numbers : 
  ∃ S : Finset ℤ, 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 1982) ∧ 
    (∀ a b c : ℤ, a ∈ S → b ∈ S → c ∈ S → c ≠ a * b) ∧
    ∀ T : Finset ℤ, 
      ((∀ y ∈ T, 1 ≤ y ∧ y ≤ 1982) ∧ 
       (∀ p q r : ℤ, p ∈ T → q ∈ T → r ∈ T → r ≠ p * q) → 
       T.card ≥ 1982 - 43) :=
sorry

end NUMINAMATH_GPT_min_removed_numbers_l1994_199470


namespace NUMINAMATH_GPT_A_investment_is_correct_l1994_199482

-- Definitions based on the given conditions
def B_investment : ℝ := 8000
def C_investment : ℝ := 10000
def P_B : ℝ := 1000
def diff_P_A_P_C : ℝ := 500

-- Main statement we need to prove
theorem A_investment_is_correct (A_investment : ℝ) 
  (h1 : B_investment = 8000) 
  (h2 : C_investment = 10000)
  (h3 : P_B = 1000)
  (h4 : diff_P_A_P_C = 500)
  (h5 : A_investment = B_investment * (P_B / 1000) * 1.5) :
  A_investment = 12000 :=
sorry

end NUMINAMATH_GPT_A_investment_is_correct_l1994_199482


namespace NUMINAMATH_GPT_quadratic_root_value_l1994_199447

theorem quadratic_root_value (m : ℝ) :
  ∃ m, (∀ x, x^2 - m * x - 3 = 0 → x = -2) → m = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_value_l1994_199447


namespace NUMINAMATH_GPT_smallest_positive_real_x_l1994_199474

theorem smallest_positive_real_x :
  ∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧ x = 89 / 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_real_x_l1994_199474


namespace NUMINAMATH_GPT_good_set_exists_l1994_199441

def is_good_set (A : List ℕ) : Prop :=
  ∀ i ∈ A, i > 0 ∧ ∀ j ∈ A, i ≠ j → i ^ 2015 % (List.prod (A.erase i)) = 0

theorem good_set_exists (n : ℕ) (h : 3 ≤ n ∧ n ≤ 2015) : 
  ∃ A : List ℕ, A.length = n ∧ ∀ (a : ℕ), a ∈ A → a > 0 ∧ is_good_set A :=
sorry

end NUMINAMATH_GPT_good_set_exists_l1994_199441


namespace NUMINAMATH_GPT_log_mult_l1994_199483

theorem log_mult : 
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 :=
by 
  sorry

end NUMINAMATH_GPT_log_mult_l1994_199483


namespace NUMINAMATH_GPT_John_lost_socks_l1994_199409

theorem John_lost_socks (initial_socks remaining_socks : ℕ) (H1 : initial_socks = 20) (H2 : remaining_socks = 14) : initial_socks - remaining_socks = 6 :=
by
-- Proof steps can be skipped
sorry

end NUMINAMATH_GPT_John_lost_socks_l1994_199409


namespace NUMINAMATH_GPT_find_x_and_y_l1994_199494

variables (x y : ℝ)

def arithmetic_mean_condition : Prop := (8 + 15 + x + y + 22 + 30) / 6 = 15
def relationship_condition : Prop := y = x + 6

theorem find_x_and_y (h1 : arithmetic_mean_condition x y) (h2 : relationship_condition x y) : 
  x = 4.5 ∧ y = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_find_x_and_y_l1994_199494


namespace NUMINAMATH_GPT_problem_l1994_199419

-- Condition that defines s and t
def s : ℤ := 4
def t : ℤ := 3

theorem problem (s t : ℤ) (h_s : s = 4) (h_t : t = 3) : s - 2 * t = -2 := by
  sorry

end NUMINAMATH_GPT_problem_l1994_199419


namespace NUMINAMATH_GPT_inequality_f_solution_minimum_g_greater_than_f_l1994_199462

noncomputable def f (x : ℝ) := abs (x - 2) - abs (x + 1)

theorem inequality_f_solution : {x : ℝ | f x > 1} = {x | x < 0} :=
sorry

noncomputable def g (a x : ℝ) := (a * x^2 - x + 1) / x

theorem minimum_g_greater_than_f (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 0 < x → g a x > f x) ↔ 1 ≤ a :=
sorry

end NUMINAMATH_GPT_inequality_f_solution_minimum_g_greater_than_f_l1994_199462
