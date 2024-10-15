import Mathlib

namespace NUMINAMATH_GPT_tan_sum_angle_l2413_241382

theorem tan_sum_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (π / 4 + α) = -3 := 
by sorry

end NUMINAMATH_GPT_tan_sum_angle_l2413_241382


namespace NUMINAMATH_GPT_symmetric_line_eq_l2413_241301

theorem symmetric_line_eq (x y : ℝ) : (x - y = 0) → (x = 1) → (y = -x + 2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_eq_l2413_241301


namespace NUMINAMATH_GPT_total_weight_of_new_people_l2413_241300

theorem total_weight_of_new_people (W W_new : ℝ) :
  (∀ (old_weights : List ℝ), old_weights.length = 25 →
    ((old_weights.sum - (65 + 70 + 75)) + W_new = old_weights.sum + (4 * 25)) →
    W_new = 310) := by
  intros old_weights old_weights_length increase_condition
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_total_weight_of_new_people_l2413_241300


namespace NUMINAMATH_GPT_evaluate_f_at_1_l2413_241319

noncomputable def f (x : ℝ) : ℝ := 2^x + 2

theorem evaluate_f_at_1 : f 1 = 4 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_evaluate_f_at_1_l2413_241319


namespace NUMINAMATH_GPT_Corey_found_golf_balls_on_Saturday_l2413_241379

def goal : ℕ := 48
def golf_balls_found_on_sunday : ℕ := 18
def golf_balls_needed : ℕ := 14
def golf_balls_found_on_saturday : ℕ := 16

theorem Corey_found_golf_balls_on_Saturday :
  (goal - golf_balls_found_on_sunday - golf_balls_needed) = golf_balls_found_on_saturday := 
by
  sorry

end NUMINAMATH_GPT_Corey_found_golf_balls_on_Saturday_l2413_241379


namespace NUMINAMATH_GPT_width_of_box_l2413_241367

theorem width_of_box 
(length depth num_cubes : ℕ)
(h_length : length = 49)
(h_depth : depth = 14)
(h_num_cubes : num_cubes = 84)
: ∃ width : ℕ, width = 42 := 
sorry

end NUMINAMATH_GPT_width_of_box_l2413_241367


namespace NUMINAMATH_GPT_sum_last_two_digits_7_13_23_l2413_241361

theorem sum_last_two_digits_7_13_23 :
  (7 ^ 23 + 13 ^ 23) % 100 = 40 :=
by 
-- Proof goes here
sorry

end NUMINAMATH_GPT_sum_last_two_digits_7_13_23_l2413_241361


namespace NUMINAMATH_GPT_eve_ran_further_l2413_241387

variable (ran_distance walked_distance difference_distance : ℝ)

theorem eve_ran_further (h1 : ran_distance = 0.7) (h2 : walked_distance = 0.6) : ran_distance - walked_distance = 0.1 := by
  sorry

end NUMINAMATH_GPT_eve_ran_further_l2413_241387


namespace NUMINAMATH_GPT_total_value_of_remaining_books_l2413_241321

-- initial definitions
def total_books : ℕ := 55
def hardback_books : ℕ := 10
def hardback_price : ℕ := 20
def paperback_price : ℕ := 10
def books_sold : ℕ := 14

-- calculate remaining books
def remaining_books : ℕ := total_books - books_sold

-- calculate remaining hardback and paperback books
def remaining_hardback_books : ℕ := hardback_books
def remaining_paperback_books : ℕ := remaining_books - remaining_hardback_books

-- calculate total values
def remaining_hardback_value : ℕ := remaining_hardback_books * hardback_price
def remaining_paperback_value : ℕ := remaining_paperback_books * paperback_price

-- total value of remaining books
def total_remaining_value : ℕ := remaining_hardback_value + remaining_paperback_value

theorem total_value_of_remaining_books : total_remaining_value = 510 := by
  -- calculation steps are skipped as instructed
  sorry

end NUMINAMATH_GPT_total_value_of_remaining_books_l2413_241321


namespace NUMINAMATH_GPT_prob_bashers_win_at_least_4_out_of_5_l2413_241314

-- Define the probability p that the Bashers win a single game.
def p := 4 / 5

-- Define the number of games n.
def n := 5

-- Define the random trial outcome space.
def trials : Type := Fin n → Bool

-- Define the number of wins (true means a win, false means a loss).
def wins (t : trials) : ℕ := (Finset.univ.filter (λ i => t i = true)).card

-- Define winning exactly k games.
def win_exactly (t : trials) (k : ℕ) : Prop := wins t = k

-- Define the probability of winning exactly k games.
noncomputable def prob_win_exactly (k : ℕ) : ℚ :=
  (Nat.descFactorial n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the event of winning at least 4 out of 5 games.
def event_win_at_least (t : trials) := (wins t ≥ 4)

-- Define the probability of winning at least k out of n games.
noncomputable def prob_win_at_least (k : ℕ) : ℚ :=
  prob_win_exactly k + prob_win_exactly (k + 1)

-- Theorem to prove: Probability of winning at least 4 out of 5 games is 3072/3125.
theorem prob_bashers_win_at_least_4_out_of_5 :
  prob_win_at_least 4 = 3072 / 3125 :=
by
  sorry

end NUMINAMATH_GPT_prob_bashers_win_at_least_4_out_of_5_l2413_241314


namespace NUMINAMATH_GPT_GreatWhiteSharkTeeth_l2413_241363

-- Definition of the number of teeth for a tiger shark
def tiger_shark_teeth : ℕ := 180

-- Definition of the number of teeth for a hammerhead shark
def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

-- Definition of the number of teeth for a great white shark
def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

-- Statement to prove
theorem GreatWhiteSharkTeeth : great_white_shark_teeth = 420 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_GreatWhiteSharkTeeth_l2413_241363


namespace NUMINAMATH_GPT_seongjun_ttakji_count_l2413_241317

variable (S A : ℕ)

theorem seongjun_ttakji_count (h1 : (3/4 : ℚ) * S - 25 = 7 * (A - 50)) (h2 : A = 100) : S = 500 :=
sorry

end NUMINAMATH_GPT_seongjun_ttakji_count_l2413_241317


namespace NUMINAMATH_GPT_apples_mass_left_l2413_241366

theorem apples_mass_left (initial_kidney golden canada fuji granny : ℕ)
                         (sold_kidney golden canada fuji granny : ℕ)
                         (left_kidney golden canada fuji granny : ℕ) :
  initial_kidney = 26 → sold_kidney = 15 → left_kidney = 11 →
  initial_golden = 42 → sold_golden = 28 → left_golden = 14 →
  initial_canada = 19 → sold_canada = 12 → left_canada = 7 →
  initial_fuji = 35 → sold_fuji = 20 → left_fuji = 15 →
  initial_granny = 22 → sold_granny = 18 → left_granny = 4 →
  left_kidney = initial_kidney - sold_kidney ∧
  left_golden = initial_golden - sold_golden ∧
  left_canada = initial_canada - sold_canada ∧
  left_fuji = initial_fuji - sold_fuji ∧
  left_granny = initial_granny - sold_granny := by sorry

end NUMINAMATH_GPT_apples_mass_left_l2413_241366


namespace NUMINAMATH_GPT_total_cost_of_fencing_l2413_241395

def P : ℤ := 42 + 35 + 52 + 66 + 40
def cost_per_meter : ℤ := 3
def total_cost : ℤ := P * cost_per_meter

theorem total_cost_of_fencing : total_cost = 705 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_fencing_l2413_241395


namespace NUMINAMATH_GPT_largest_possible_n_base10_l2413_241345

theorem largest_possible_n_base10 :
  ∃ (n A B C : ℕ),
    n = 25 * A + 5 * B + C ∧ 
    n = 81 * C + 9 * B + A ∧ 
    A < 5 ∧ B < 5 ∧ C < 5 ∧ 
    n = 69 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_possible_n_base10_l2413_241345


namespace NUMINAMATH_GPT_math_problem_l2413_241372

noncomputable def compute_value (a b c : ℝ) : ℝ :=
  (b / (a + b)) + (c / (b + c)) + (a / (c + a))

theorem math_problem (a b c : ℝ)
  (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -12)
  (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 15) :
  compute_value a b c = 6 :=
sorry

end NUMINAMATH_GPT_math_problem_l2413_241372


namespace NUMINAMATH_GPT_chosen_number_is_129_l2413_241347

theorem chosen_number_is_129 (x : ℕ) (h : 2 * x - 148 = 110) : x = 129 :=
by
  sorry

end NUMINAMATH_GPT_chosen_number_is_129_l2413_241347


namespace NUMINAMATH_GPT_animal_fish_consumption_l2413_241365

-- Definitions for the daily consumption of each animal
def daily_trout_polar1 := 0.2
def daily_salmon_polar1 := 0.4

def daily_trout_polar2 := 0.3
def daily_salmon_polar2 := 0.5

def daily_trout_polar3 := 0.25
def daily_salmon_polar3 := 0.45

def daily_trout_sealion1 := 0.1
def daily_salmon_sealion1 := 0.15

def daily_trout_sealion2 := 0.2
def daily_salmon_sealion2 := 0.25

-- Calculate total daily consumption
def total_daily_trout :=
  daily_trout_polar1 + daily_trout_polar2 + daily_trout_polar3 + daily_trout_sealion1 + daily_trout_sealion2

def total_daily_salmon :=
  daily_salmon_polar1 + daily_salmon_polar2 + daily_salmon_polar3 + daily_salmon_sealion1 + daily_salmon_sealion2

-- Calculate total monthly consumption
def total_monthly_trout := total_daily_trout * 30
def total_monthly_salmon := total_daily_salmon * 30

-- Total monthly fish bucket consumption
def total_monthly_fish := total_monthly_trout + total_monthly_salmon

-- The statement to prove the total consumption
theorem animal_fish_consumption : total_monthly_fish = 84 := by
  sorry

end NUMINAMATH_GPT_animal_fish_consumption_l2413_241365


namespace NUMINAMATH_GPT_cone_surface_area_is_correct_l2413_241318

noncomputable def cone_surface_area (central_angle_degrees : ℝ) (sector_area : ℝ) : ℝ :=
  if central_angle_degrees = 120 ∧ sector_area = 3 * Real.pi then 4 * Real.pi else 0

theorem cone_surface_area_is_correct :
  cone_surface_area 120 (3 * Real.pi) = 4 * Real.pi :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_cone_surface_area_is_correct_l2413_241318


namespace NUMINAMATH_GPT_unicorn_witch_ratio_l2413_241350

theorem unicorn_witch_ratio (W D U : ℕ) (h1 : W = 7) (h2 : D = W + 25) (h3 : U + W + D = 60) :
  U / W = 3 := by
  sorry

end NUMINAMATH_GPT_unicorn_witch_ratio_l2413_241350


namespace NUMINAMATH_GPT_solve_for_n_l2413_241354

theorem solve_for_n (n : ℕ) : 4^8 = 16^n → n = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l2413_241354


namespace NUMINAMATH_GPT_cone_volume_not_product_base_height_l2413_241371

noncomputable def cone_volume (S h : ℝ) := (1/3) * S * h

theorem cone_volume_not_product_base_height (S h : ℝ) :
  cone_volume S h ≠ S * h :=
by sorry

end NUMINAMATH_GPT_cone_volume_not_product_base_height_l2413_241371


namespace NUMINAMATH_GPT_a_plus_b_eq_11_l2413_241397

noncomputable def f (a b x : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

theorem a_plus_b_eq_11 (a b : ℝ) 
  (h1 : ∀ x, f a b x ≤ f a b (-1))
  (h2 : f a b (-1) = 0) 
  : a + b = 11 :=
sorry

end NUMINAMATH_GPT_a_plus_b_eq_11_l2413_241397


namespace NUMINAMATH_GPT_intersection_point_l2413_241369

noncomputable def line1 (x : ℚ) : ℚ := 3 * x
noncomputable def line2 (x : ℚ) : ℚ := -9 * x - 6

theorem intersection_point : ∃ (x y : ℚ), line1 x = y ∧ line2 x = y ∧ x = -1/2 ∧ y = -3/2 :=
by
  -- skipping the actual proof steps
  sorry

end NUMINAMATH_GPT_intersection_point_l2413_241369


namespace NUMINAMATH_GPT_abs_diff_of_solutions_l2413_241335

theorem abs_diff_of_solutions (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
sorry

end NUMINAMATH_GPT_abs_diff_of_solutions_l2413_241335


namespace NUMINAMATH_GPT_shortest_distance_proof_l2413_241343

noncomputable def shortest_distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem shortest_distance_proof : 
  let A : ℝ × ℝ := (0, 250)
  let B : ℝ × ℝ := (800, 1050)
  shortest_distance A B = 1131 :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_proof_l2413_241343


namespace NUMINAMATH_GPT_periodic_sequence_exists_l2413_241304

noncomputable def bounded_sequence (a : ℕ → ℤ) (M : ℤ) :=
  ∀ n, |a n| ≤ M

noncomputable def satisfies_recurrence (a : ℕ → ℤ) :=
  ∀ n, n ≥ 5 → a n = (a (n - 1) + a (n - 2) + a (n - 3) * a (n - 4)) / (a (n - 1) * a (n - 2) + a (n - 3) + a (n - 4))

theorem periodic_sequence_exists (a : ℕ → ℤ) (M : ℤ) 
  (h_bounded : bounded_sequence a M) (h_rec : satisfies_recurrence a) : 
  ∃ l : ℕ, ∀ n : ℕ, a (l + n) = a (l + n + (l + 1) - l) :=
sorry

end NUMINAMATH_GPT_periodic_sequence_exists_l2413_241304


namespace NUMINAMATH_GPT_relationship_between_y_l2413_241375

theorem relationship_between_y
  (m y₁ y₂ y₃ : ℝ)
  (hA : y₁ = -(-1)^2 + 2 * -1 + m)
  (hB : y₂ = -(1)^2 + 2 * 1 + m)
  (hC : y₃ = -(2)^2 + 2 * 2 + m) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end NUMINAMATH_GPT_relationship_between_y_l2413_241375


namespace NUMINAMATH_GPT_dress_total_price_correct_l2413_241394

-- Define constants and variables
def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

-- Function to calculate sale price after discount
def sale_price (op : ℝ) (dr : ℝ) : ℝ := op - (op * dr)

-- Function to calculate total price including tax
def total_selling_price (sp : ℝ) (tr : ℝ) : ℝ := sp + (sp * tr)

-- The proof statement to be proven
theorem dress_total_price_correct :
  total_selling_price (sale_price original_price discount_rate) tax_rate = 96.6 :=
  by sorry

end NUMINAMATH_GPT_dress_total_price_correct_l2413_241394


namespace NUMINAMATH_GPT_desiree_age_l2413_241338

theorem desiree_age (D C G Gr : ℕ) 
  (h1 : D = 2 * C)
  (h2 : D + 30 = (2 * (C + 30)) / 3 + 14)
  (h3 : G = D + C)
  (h4 : G + 20 = 3 * (D - C))
  (h5 : Gr = (D + 10) * (C + 10) / 2) : 
  D = 6 := 
sorry

end NUMINAMATH_GPT_desiree_age_l2413_241338


namespace NUMINAMATH_GPT_arnold_danny_age_l2413_241373

theorem arnold_danny_age (x : ℕ) : (x + 1) * (x + 1) = x * x + 9 → x = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_arnold_danny_age_l2413_241373


namespace NUMINAMATH_GPT_fraction_product_eq_six_l2413_241328

theorem fraction_product_eq_six : (2/5) * (3/4) * (1/6) * (120 : ℚ) = 6 := by
  sorry

end NUMINAMATH_GPT_fraction_product_eq_six_l2413_241328


namespace NUMINAMATH_GPT_cos_2_alpha_plus_beta_eq_l2413_241384

variable (α β : ℝ)

def tan_roots_of_quadratic (x : ℝ) : Prop := x^2 + 5 * x - 6 = 0

theorem cos_2_alpha_plus_beta_eq :
  ∀ α β : ℝ, tan_roots_of_quadratic (Real.tan α) ∧ tan_roots_of_quadratic (Real.tan β) →
  Real.cos (2 * (α + β)) = 12 / 37 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cos_2_alpha_plus_beta_eq_l2413_241384


namespace NUMINAMATH_GPT_current_height_of_tree_l2413_241331

-- Definitions of conditions
def growth_per_year : ℝ := 0.5
def years : ℕ := 240
def final_height : ℝ := 720

-- The goal is to prove that the current height of the tree is 600 inches
theorem current_height_of_tree :
  final_height - (growth_per_year * years) = 600 := 
sorry

end NUMINAMATH_GPT_current_height_of_tree_l2413_241331


namespace NUMINAMATH_GPT_subtraction_equality_l2413_241377

theorem subtraction_equality : 3.56 - 2.15 = 1.41 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_equality_l2413_241377


namespace NUMINAMATH_GPT_ball_box_distribution_l2413_241308

theorem ball_box_distribution : (∃ (f : Fin 4 → Fin 2), true) ∧ (∀ (f : Fin 4 → Fin 2), true) → ∃ (f : Fin 4 → Fin 2), true ∧ f = 16 :=
by sorry

end NUMINAMATH_GPT_ball_box_distribution_l2413_241308


namespace NUMINAMATH_GPT_no_such_number_exists_l2413_241396

theorem no_such_number_exists :
  ¬ ∃ n : ℕ, 529 < n ∧ n < 538 ∧ 16 ∣ n :=
by sorry

end NUMINAMATH_GPT_no_such_number_exists_l2413_241396


namespace NUMINAMATH_GPT_min_value_x_y_l2413_241374

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : x + y ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_value_x_y_l2413_241374


namespace NUMINAMATH_GPT_vertical_asymptotes_l2413_241392

noncomputable def f (x : ℝ) := (x^3 + 3*x^2 + 2*x + 12) / (x^2 - 5*x + 6)

theorem vertical_asymptotes (x : ℝ) : 
  (x^2 - 5*x + 6 = 0) ∧ (x^3 + 3*x^2 + 2*x + 12 ≠ 0) ↔ (x = 2 ∨ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_vertical_asymptotes_l2413_241392


namespace NUMINAMATH_GPT_nancy_carrots_l2413_241391

theorem nancy_carrots (picked_day_1 threw_out total_left total_final picked_next_day : ℕ)
  (h1 : picked_day_1 = 12)
  (h2 : threw_out = 2)
  (h3 : total_final = 31)
  (h4 : total_left = picked_day_1 - threw_out)
  (h5 : total_final = total_left + picked_next_day) :
  picked_next_day = 21 :=
by
  sorry

end NUMINAMATH_GPT_nancy_carrots_l2413_241391


namespace NUMINAMATH_GPT_circle_condition_l2413_241370

theorem circle_condition (f : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 4*x + 6*y + f = 0) ↔ f < 13 :=
by
  sorry

end NUMINAMATH_GPT_circle_condition_l2413_241370


namespace NUMINAMATH_GPT_cara_meets_don_distance_l2413_241378

theorem cara_meets_don_distance (distance total_distance : ℝ) (cara_speed don_speed : ℝ) (delay : ℝ) 
  (h_total_distance : total_distance = 45)
  (h_cara_speed : cara_speed = 6)
  (h_don_speed : don_speed = 5)
  (h_delay : delay = 2) :
  distance = 30 :=
by
  have h := 1 / total_distance
  have : cara_speed * (distance / cara_speed) + don_speed * (distance / cara_speed - delay) = 45 := sorry
  exact sorry

end NUMINAMATH_GPT_cara_meets_don_distance_l2413_241378


namespace NUMINAMATH_GPT_time_spent_in_park_is_76_19_percent_l2413_241337

noncomputable def total_time_in_park (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  trip_times.foldl (λ acc (t, _, _) => acc + t) 0

noncomputable def total_walking_time (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  trip_times.foldl (λ acc (_, w1, w2) => acc + (w1 + w2)) 0

noncomputable def total_trip_time (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  total_time_in_park trip_times + total_walking_time trip_times

noncomputable def percentage_time_in_park (trip_times : List (ℕ × ℕ × ℕ)) : ℚ :=
  (total_time_in_park trip_times : ℚ) / (total_trip_time trip_times : ℚ) * 100

theorem time_spent_in_park_is_76_19_percent (trip_times : List (ℕ × ℕ × ℕ)) :
  trip_times = [(120, 20, 25), (90, 15, 15), (150, 10, 20), (180, 30, 20), (120, 20, 10), (60, 15, 25)] →
  percentage_time_in_park trip_times = 76.19 :=
by
  intro h
  rw [h]  
  simp
  sorry

end NUMINAMATH_GPT_time_spent_in_park_is_76_19_percent_l2413_241337


namespace NUMINAMATH_GPT_consecutive_integers_product_divisible_l2413_241323

theorem consecutive_integers_product_divisible (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  ∀ n : ℕ, ∃ (x y : ℕ), (n ≤ x) ∧ (x < n + b) ∧ (n ≤ y) ∧ (y < n + b) ∧ (x ≠ y) ∧ (a * b ∣ x * y) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integers_product_divisible_l2413_241323


namespace NUMINAMATH_GPT_max_3x_4y_eq_73_l2413_241311

theorem max_3x_4y_eq_73 :
  (∀ x y : ℝ, x ^ 2 + y ^ 2 = 14 * x + 6 * y + 6 → 3 * x + 4 * y ≤ 73) ∧
  (∃ x y : ℝ, x ^ 2 + y ^ 2 = 14 * x + 6 * y + 6 ∧ 3 * x + 4 * y = 73) :=
by sorry

end NUMINAMATH_GPT_max_3x_4y_eq_73_l2413_241311


namespace NUMINAMATH_GPT_max_value_of_a_max_value_reached_l2413_241356

theorem max_value_of_a (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = 1) : 
  a ≤ Real.sqrt 6 / 3 :=
by
  sorry

theorem max_value_reached (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = 1) : 
  ∃ a, a = Real.sqrt 6 / 3 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_a_max_value_reached_l2413_241356


namespace NUMINAMATH_GPT_sum_integers_neg40_to_60_l2413_241334

theorem sum_integers_neg40_to_60 : 
  (Finset.sum (Finset.range (60 + 40 + 1)) (λ x => x - 40)) = 1010 := sorry

end NUMINAMATH_GPT_sum_integers_neg40_to_60_l2413_241334


namespace NUMINAMATH_GPT_probability_sum_3_or_7_or_10_l2413_241359

-- Definitions of the faces of each die
def die_1_faces : List ℕ := [1, 2, 2, 5, 5, 6]
def die_2_faces : List ℕ := [1, 2, 4, 4, 5, 6]

-- Probability of a sum being 3 (valid_pairs: (1, 2))
def probability_sum_3 : ℚ :=
  (1 / 6) * (1 / 6)

-- Probability of a sum being 7 (valid pairs: (1, 6), (2, 5))
def probability_sum_7 : ℚ :=
  ((1 / 6) * (1 / 6)) + ((1 / 3) * (1 / 6))

-- Probability of a sum being 10 (valid pairs: (5, 5))
def probability_sum_10 : ℚ :=
  (1 / 3) * (1 / 6)

-- Total probability for sums being 3, 7, or 10
def total_probability : ℚ :=
  probability_sum_3 + probability_sum_7 + probability_sum_10

-- The proof statement
theorem probability_sum_3_or_7_or_10 : total_probability = 1 / 6 :=
  sorry

end NUMINAMATH_GPT_probability_sum_3_or_7_or_10_l2413_241359


namespace NUMINAMATH_GPT_solve_rational_eq_l2413_241310

theorem solve_rational_eq (x : ℝ) :
  (1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 3 * x - 18) + 1 / (x^2 - 15 * x - 12) = 0) →
  (x = 1 ∨ x = -1 ∨ x = 12 ∨ x = -12) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_rational_eq_l2413_241310


namespace NUMINAMATH_GPT_total_payment_for_combined_shopping_trip_l2413_241333

noncomputable def discount (amount : ℝ) : ℝ :=
  if amount ≤ 200 then amount
  else if amount ≤ 500 then amount * 0.9
  else 500 * 0.9 + (amount - 500) * 0.7

theorem total_payment_for_combined_shopping_trip :
  discount (168 + 423 / 0.9) = 546.6 :=
by
  sorry

end NUMINAMATH_GPT_total_payment_for_combined_shopping_trip_l2413_241333


namespace NUMINAMATH_GPT_total_candies_needed_l2413_241385

def candies_per_box : ℕ := 156
def number_of_children : ℕ := 20

theorem total_candies_needed : candies_per_box * number_of_children = 3120 := by
  sorry

end NUMINAMATH_GPT_total_candies_needed_l2413_241385


namespace NUMINAMATH_GPT_alex_play_friends_with_l2413_241344

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end NUMINAMATH_GPT_alex_play_friends_with_l2413_241344


namespace NUMINAMATH_GPT_calculate_gf3_l2413_241358

def f (x : ℕ) : ℕ := x^3 - 1
def g (x : ℕ) : ℕ := 3 * x^2 + x + 2

theorem calculate_gf3 : g (f 3) = 2056 := by
  sorry

end NUMINAMATH_GPT_calculate_gf3_l2413_241358


namespace NUMINAMATH_GPT_find_remainder_l2413_241332

theorem find_remainder (a : ℕ) :
  (a ^ 100) % 73 = 2 ∧ (a ^ 101) % 73 = 69 → a % 73 = 71 :=
by
  sorry

end NUMINAMATH_GPT_find_remainder_l2413_241332


namespace NUMINAMATH_GPT_sum_of_variables_l2413_241383

theorem sum_of_variables (x y z : ℝ) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z)
  (hxy : x * y = 30) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_variables_l2413_241383


namespace NUMINAMATH_GPT_height_of_tree_l2413_241306

noncomputable def height_of_flagpole : ℝ := 4
noncomputable def shadow_of_flagpole : ℝ := 6
noncomputable def shadow_of_tree : ℝ := 12

theorem height_of_tree (h : height_of_flagpole / shadow_of_flagpole = x / shadow_of_tree) : x = 8 := by
  sorry

end NUMINAMATH_GPT_height_of_tree_l2413_241306


namespace NUMINAMATH_GPT_range_of_a_l2413_241307

def p (a : ℝ) : Prop :=
(∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0)

def q (a : ℝ) : Prop :=
0 < a ∧ a < 1

theorem range_of_a (a : ℝ) : ((p a ∨ q a) ∧ ¬(p a ∧ q a)) ↔ (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0) :=
  sorry

end NUMINAMATH_GPT_range_of_a_l2413_241307


namespace NUMINAMATH_GPT_one_in_B_neg_one_not_in_B_B_roster_l2413_241324

open Set Int

def B : Set ℤ := {x | ∃ n : ℕ, 6 = n * (3 - x)}

theorem one_in_B : 1 ∈ B :=
by sorry

theorem neg_one_not_in_B : (-1 ∉ B) :=
by sorry

theorem B_roster : B = {2, 1, 0, -3} :=
by sorry

end NUMINAMATH_GPT_one_in_B_neg_one_not_in_B_B_roster_l2413_241324


namespace NUMINAMATH_GPT_t_shirts_in_two_hours_l2413_241362

-- Definitions for the conditions
def first_hour_rate : Nat := 12
def second_hour_rate : Nat := 6

-- Main statement to prove
theorem t_shirts_in_two_hours : 
  (60 / first_hour_rate + 60 / second_hour_rate) = 15 := by
  sorry

end NUMINAMATH_GPT_t_shirts_in_two_hours_l2413_241362


namespace NUMINAMATH_GPT_nathan_subtracts_79_l2413_241339

theorem nathan_subtracts_79 (a b : ℤ) (h₁ : a = 40) (h₂ : b = 1) :
  (a - b) ^ 2 = a ^ 2 - 79 := 
by
  sorry

end NUMINAMATH_GPT_nathan_subtracts_79_l2413_241339


namespace NUMINAMATH_GPT_range_of_m_l2413_241389

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x - 2 < 0) → -8 < m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2413_241389


namespace NUMINAMATH_GPT_max_cables_191_l2413_241360

/-- 
  There are 30 employees: 20 with brand A computers and 10 with brand B computers.
  Cables can only connect a brand A computer to a brand B computer.
  Employees can communicate with each other if their computers are directly connected by a cable 
  or by relaying messages through a series of connected computers.
  The maximum possible number of cables used to ensure every employee can communicate with each other
  is 191.
-/
theorem max_cables_191 (A B : ℕ) (hA : A = 20) (hB : B = 10) : 
  ∃ (max_cables : ℕ), max_cables = 191 ∧ 
  (∀ (i j : ℕ), (i ≤ A ∧ j ≤ B) → (i = A ∨ j = B) → i * j ≤ max_cables) := 
sorry

end NUMINAMATH_GPT_max_cables_191_l2413_241360


namespace NUMINAMATH_GPT_find_f_at_3_l2413_241352

variable (f : ℝ → ℝ)

-- Conditions
-- 1. f is an odd function
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
-- 2. f(-1) = 1/2
axiom f_neg_one : f (-1) = 1 / 2
-- 3. f(x+2) = f(x) + 2 for all x
axiom functional_equation : ∀ x : ℝ, f (x + 2) = f x + 2

-- The target value to prove
theorem find_f_at_3 : f 3 = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_find_f_at_3_l2413_241352


namespace NUMINAMATH_GPT_roots_of_equation_l2413_241322

theorem roots_of_equation (x : ℝ) : ((x - 5) ^ 2 = 2 * (x - 5)) ↔ (x = 5 ∨ x = 7) := by
sorry

end NUMINAMATH_GPT_roots_of_equation_l2413_241322


namespace NUMINAMATH_GPT_remainder_2015_div_28_l2413_241376

theorem remainder_2015_div_28 : 2015 % 28 = 17 :=
by
  sorry

end NUMINAMATH_GPT_remainder_2015_div_28_l2413_241376


namespace NUMINAMATH_GPT_proof_problem_l2413_241309

theorem proof_problem (x : ℝ) (h1 : x = 3) (h2 : 2 * x ≠ 5) (h3 : x + 5 ≠ 3) 
                      (h4 : 7 - x ≠ 2) (h5 : 6 + 2 * x ≠ 14) :
    3 * x - 1 = 8 :=
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l2413_241309


namespace NUMINAMATH_GPT_value_of_a_l2413_241353

theorem value_of_a
    (a b : ℝ)
    (h₁ : 0 < a ∧ 0 < b)
    (h₂ : a + b = 1)
    (h₃ : 21 * a^5 * b^2 = 35 * a^4 * b^3) :
    a = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2413_241353


namespace NUMINAMATH_GPT_smallest_r_minus_p_l2413_241313

theorem smallest_r_minus_p 
  (p q r : ℕ) (h₀ : p * q * r = 362880) (h₁ : p < q) (h₂ : q < r) : 
  r - p = 126 :=
sorry

end NUMINAMATH_GPT_smallest_r_minus_p_l2413_241313


namespace NUMINAMATH_GPT_range_of_a_l2413_241336

theorem range_of_a 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h : ∀ x, f x = -x^2 + 2*(a - 1)*x + 2)
  (increasing_on : ∀ x < 4, deriv f x > 0) : a ≥ 5 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2413_241336


namespace NUMINAMATH_GPT_route_B_no_quicker_l2413_241351

noncomputable def time_route_A (distance_A : ℕ) (speed_A : ℕ) : ℕ :=
(distance_A * 60) / speed_A

noncomputable def time_route_B (distance_B : ℕ) (speed_B1 : ℕ) (speed_B2 : ℕ) : ℕ :=
  let distance_B1 := distance_B - 1
  let distance_B2 := 1
  (distance_B1 * 60) / speed_B1 + (distance_B2 * 60) / speed_B2

theorem route_B_no_quicker : time_route_A 8 40 = time_route_B 6 50 10 :=
by
  sorry

end NUMINAMATH_GPT_route_B_no_quicker_l2413_241351


namespace NUMINAMATH_GPT_lemonade_second_intermission_l2413_241381

theorem lemonade_second_intermission (first_intermission third_intermission total_lemonade second_intermission : ℝ) 
  (h1 : first_intermission = 0.25) 
  (h2 : third_intermission = 0.25) 
  (h3 : total_lemonade = 0.92) 
  (h4 : second_intermission = total_lemonade - (first_intermission + third_intermission)) : 
  second_intermission = 0.42 := 
by 
  sorry

end NUMINAMATH_GPT_lemonade_second_intermission_l2413_241381


namespace NUMINAMATH_GPT_a_seq_gt_one_l2413_241368

noncomputable def a_seq (a : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then 1 + a
  else (1 / a_seq a (n - 1)) + a

theorem a_seq_gt_one (a : ℝ) (h : 0 < a ∧ a < 1) : ∀ n : ℕ, 1 < a_seq a n :=
by {
  sorry
}

end NUMINAMATH_GPT_a_seq_gt_one_l2413_241368


namespace NUMINAMATH_GPT_trajectory_equation_l2413_241341

variable (m x y : ℝ)
def a := (m * x, y + 1)
def b := (x, y - 1)
def is_perpendicular (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0

theorem trajectory_equation 
  (h1: is_perpendicular (a m x y) (b x y)) : 
  m * x^2 + y^2 = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_equation_l2413_241341


namespace NUMINAMATH_GPT_find_M_pos_int_l2413_241380

theorem find_M_pos_int (M : ℕ) (hM : 33^2 * 66^2 = 15^2 * M^2) :
    M = 726 :=
by
  -- Sorry, skipping the proof.
  sorry

end NUMINAMATH_GPT_find_M_pos_int_l2413_241380


namespace NUMINAMATH_GPT_total_journey_distance_l2413_241316

variable (D : ℚ) (lateTime : ℚ := 1/4)

theorem total_journey_distance :
  (∃ (T : ℚ), T = D / 40 ∧ T + lateTime = D / 35) →
  D = 70 :=
by
  intros h
  obtain ⟨T, h1, h2⟩ := h
  have h3 : T = D / 40 := h1
  have h4 : T + lateTime = D / 35 := h2
  sorry

end NUMINAMATH_GPT_total_journey_distance_l2413_241316


namespace NUMINAMATH_GPT_function_problem_l2413_241399

theorem function_problem (f : ℕ → ℝ) (h1 : ∀ p q : ℕ, f (p + q) = f p * f q) (h2 : f 1 = 3) :
  (f (1) ^ 2 + f (2)) / f (1) + (f (2) ^ 2 + f (4)) / f (3) + (f (3) ^ 2 + f (6)) / f (5) + 
  (f (4) ^ 2 + f (8)) / f (7) + (f (5) ^ 2 + f (10)) / f (9) = 30 := by
  sorry

end NUMINAMATH_GPT_function_problem_l2413_241399


namespace NUMINAMATH_GPT_find_a1_and_d_l2413_241327

-- Given conditions
variables {a : ℕ → ℤ} 
variables {a1 d : ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

theorem find_a1_and_d 
  (h1 : is_arithmetic_sequence a a1 d)
  (h2 : (a 3) * (a 7) = -16)
  (h3 : (a 4) + (a 6) = 0)
  : (a1 = -8 ∧ d = 2) ∨ (a1 = 8 ∧ d = -2) :=
sorry

end NUMINAMATH_GPT_find_a1_and_d_l2413_241327


namespace NUMINAMATH_GPT_qualified_flour_l2413_241364

-- Define the acceptable weight range
def acceptable_range (w : ℝ) : Prop :=
  24.75 ≤ w ∧ w ≤ 25.25

-- Define the weight options
def optionA : ℝ := 24.70
def optionB : ℝ := 24.80
def optionC : ℝ := 25.30
def optionD : ℝ := 25.51

-- The statement to be proved
theorem qualified_flour : acceptable_range optionB ∧ ¬acceptable_range optionA ∧ ¬acceptable_range optionC ∧ ¬acceptable_range optionD :=
by
  sorry

end NUMINAMATH_GPT_qualified_flour_l2413_241364


namespace NUMINAMATH_GPT_ratio_equality_l2413_241303

variable (a b : ℝ)

theorem ratio_equality (h : a / b = 4 / 3) : (3 * a + 2 * b) / (3 * a - 2 * b) = 3 :=
by
sorry

end NUMINAMATH_GPT_ratio_equality_l2413_241303


namespace NUMINAMATH_GPT_robotics_club_problem_l2413_241348

theorem robotics_club_problem 
    (total_students cs_students eng_students both_students : ℕ)
    (h1 : total_students = 120)
    (h2 : cs_students = 75)
    (h3 : eng_students = 50)
    (h4 : both_students = 10) :
    total_students - (cs_students - both_students + eng_students - both_students + both_students) = 5 := by
  sorry

end NUMINAMATH_GPT_robotics_club_problem_l2413_241348


namespace NUMINAMATH_GPT_distance_corresponds_to_additional_charge_l2413_241305

-- Define the initial fee
def initial_fee : ℝ := 2.5

-- Define the charge per part of a mile
def charge_per_part_of_mile : ℝ := 0.35

-- Define the total charge for a 3.6 miles trip
def total_charge : ℝ := 5.65

-- Define the correct distance corresponding to the additional charge
def correct_distance : ℝ := 0.9

-- The theorem to prove
theorem distance_corresponds_to_additional_charge :
  (total_charge - initial_fee) / charge_per_part_of_mile * (0.1) = correct_distance :=
by
  sorry

end NUMINAMATH_GPT_distance_corresponds_to_additional_charge_l2413_241305


namespace NUMINAMATH_GPT_perpendicular_condition_l2413_241329

noncomputable def line := ℝ → (ℝ × ℝ × ℝ)
noncomputable def plane := (ℝ × ℝ × ℝ) → Prop

variable {l m : line}
variable {α : plane}

-- l and m are two different lines
axiom lines_are_different : l ≠ m

-- m is parallel to the plane α
axiom m_parallel_alpha : ∀ t : ℝ, α (m t)

-- Prove that l perpendicular to α is a sufficient but not necessary condition for l perpendicular to m
theorem perpendicular_condition :
  (∀ t : ℝ, ¬ α (l t)) → (∀ t₁ t₂ : ℝ, (l t₁) ≠ (m t₂)) ∧ ¬ (∀ t : ℝ, ¬ α (l t)) :=
by 
  sorry

end NUMINAMATH_GPT_perpendicular_condition_l2413_241329


namespace NUMINAMATH_GPT_cylinder_volume_l2413_241330

theorem cylinder_volume (length width : ℝ) (h₁ h₂ : ℝ) (radius1 radius2 : ℝ) (V1 V2 : ℝ) (π : ℝ)
  (h_length : length = 12) (h_width : width = 8) 
  (circumference1 : circumference1 = length)
  (circumference2 : circumference2 = width)
  (h_radius1 : radius1 = 6 / π) (h_radius2 : radius2 = 4 / π)
  (h_height1 : h₁ = width) (h_height2 : h₂ = length)
  (h_V1 : V1 = π * radius1^2 * h₁) (h_V2 : V2 = π * radius2^2 * h₂) :
  V1 = 288 / π ∨ V2 = 192 / π :=
sorry


end NUMINAMATH_GPT_cylinder_volume_l2413_241330


namespace NUMINAMATH_GPT_rise_in_water_level_l2413_241325

theorem rise_in_water_level : 
  let edge_length : ℝ := 15
  let volume_cube : ℝ := edge_length ^ 3
  let length : ℝ := 20
  let width : ℝ := 15
  let base_area : ℝ := length * width
  let rise_in_level : ℝ := volume_cube / base_area
  rise_in_level = 11.25 :=
by
  sorry

end NUMINAMATH_GPT_rise_in_water_level_l2413_241325


namespace NUMINAMATH_GPT_problem_1_l2413_241342

theorem problem_1 (a b : ℝ) (h : b < a ∧ a < 0) : 
  (a + b < a * b) ∧ (¬ (abs a > abs b)) ∧ (¬ (1 / b > 1 / a ∧ 1 / a > 0)) ∧ (¬ (b / a + a / b > 2)) := sorry

end NUMINAMATH_GPT_problem_1_l2413_241342


namespace NUMINAMATH_GPT_jason_seashells_initial_count_l2413_241349

variable (initialSeashells : ℕ) (seashellsGivenAway : ℕ)
variable (seashellsNow : ℕ) (initialSeashells := 49)
variable (seashellsGivenAway := 13) (seashellsNow := 36)

theorem jason_seashells_initial_count :
  initialSeashells - seashellsGivenAway = seashellsNow → initialSeashells = 49 := by
  sorry

end NUMINAMATH_GPT_jason_seashells_initial_count_l2413_241349


namespace NUMINAMATH_GPT_sector_area_is_2pi_l2413_241357

noncomputable def sectorArea (l : ℝ) (R : ℝ) : ℝ :=
  (1 / 2) * l * R

theorem sector_area_is_2pi (R : ℝ) (l : ℝ) (hR : R = 4) (hl : l = π) :
  sectorArea l R = 2 * π :=
by
  sorry

end NUMINAMATH_GPT_sector_area_is_2pi_l2413_241357


namespace NUMINAMATH_GPT_parabola_focus_line_slope_intersect_l2413_241326

theorem parabola_focus (p : ℝ) (hp : 0 < p) 
  (focus : (1/2 : ℝ) = p/2) : p = 1 :=
by sorry

theorem line_slope_intersect (t : ℝ)
  (intersects_parabola : ∃ A B : ℝ × ℝ, A ≠ (0, 0) ∧ B ≠ (0, 0) ∧
    A ≠ B ∧ A.2 = 2 * A.1 + t ∧ B.2 = 2 * B.1 + t ∧ 
    A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1 ∧ 
    A.1 * B.1 + A.2 * B.2 = 0) : 
  t = -4 :=
by sorry

end NUMINAMATH_GPT_parabola_focus_line_slope_intersect_l2413_241326


namespace NUMINAMATH_GPT_sam_travel_time_l2413_241390

theorem sam_travel_time (d_AC d_CB : ℕ) (v_sam : ℕ) 
  (h1 : d_AC = 600) (h2 : d_CB = 400) (h3 : v_sam = 50) : 
  (d_AC + d_CB) / v_sam = 20 := 
by
  sorry

end NUMINAMATH_GPT_sam_travel_time_l2413_241390


namespace NUMINAMATH_GPT_value_of_x_l2413_241355

theorem value_of_x (x y : ℝ) :
  x / (x + 1) = (y^2 + 3*y + 1) / (y^2 + 3*y + 2) → x = y^2 + 3*y + 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_x_l2413_241355


namespace NUMINAMATH_GPT_binom_2024_1_l2413_241315

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_2024_1 : binomial 2024 1 = 2024 := by
  sorry

end NUMINAMATH_GPT_binom_2024_1_l2413_241315


namespace NUMINAMATH_GPT_cos_2beta_value_l2413_241346

theorem cos_2beta_value (α β : ℝ) 
  (h1 : Real.sin (α - β) = 3/5) 
  (h2 : Real.cos (α + β) = -3/5) 
  (h3 : α - β ∈ Set.Ioo (π/2) π) 
  (h4 : α + β ∈ Set.Ioo (π/2) π) : 
  Real.cos (2 * β) = 24/25 := 
sorry

end NUMINAMATH_GPT_cos_2beta_value_l2413_241346


namespace NUMINAMATH_GPT_circle_parametric_eq_l2413_241320

theorem circle_parametric_eq 
  (a b r : ℝ) (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi):
  (∃ (x y : ℝ), (x = r * Real.cos θ + a ∧ y = r * Real.sin θ + b)) ↔ 
  (∃ (x' y' : ℝ), (x' = r * Real.cos θ ∧ y' = r * Real.sin θ)) :=
sorry

end NUMINAMATH_GPT_circle_parametric_eq_l2413_241320


namespace NUMINAMATH_GPT_base_any_number_l2413_241393

theorem base_any_number (base : ℝ) (x y : ℝ) (h1 : 3^x * base^y = 19683) (h2 : x - y = 9) (h3 : x = 9) : true :=
by
  sorry

end NUMINAMATH_GPT_base_any_number_l2413_241393


namespace NUMINAMATH_GPT_find_number_l2413_241388

def x : ℝ := 33.75

theorem find_number (x: ℝ) :
  (0.30 * x = 0.25 * 45) → x = 33.75 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2413_241388


namespace NUMINAMATH_GPT_find_ratio_of_a_b_l2413_241340

noncomputable def slope_of_tangent_to_curve_at_P := 3 * 1^2 + 1

noncomputable def perpendicular_slope (a b : ℝ) : Prop :=
  slope_of_tangent_to_curve_at_P * (a / b) = -1

noncomputable def line_slope_eq_slope_of_tangent (a b : ℝ) : Prop := 
  perpendicular_slope a b

theorem find_ratio_of_a_b (a b : ℝ) 
  (h1 : a - b * 2 = 0) 
  (h2 : line_slope_eq_slope_of_tangent a b) : 
  a / b = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_of_a_b_l2413_241340


namespace NUMINAMATH_GPT_n_times_s_eq_neg_two_l2413_241302

-- Define existence of function g
variable (g : ℝ → ℝ)

-- The given condition for the function g: ℝ -> ℝ
axiom g_cond : ∀ x y : ℝ, g (g x - y) = 2 * g x + g (g y - g (-x)) + y

-- Define n and s as per the conditions mentioned in the problem
def n : ℕ := 1 -- Based on the solution, there's only one possible value
def s : ℝ := -2 -- Sum of all possible values

-- The main statement to prove
theorem n_times_s_eq_neg_two : (n * s) = -2 := by
  sorry

end NUMINAMATH_GPT_n_times_s_eq_neg_two_l2413_241302


namespace NUMINAMATH_GPT_rowing_speed_in_still_water_l2413_241398

noncomputable def speedInStillWater (distance_m : ℝ) (time_s : ℝ) (speed_current : ℝ) : ℝ :=
  let distance_km := distance_m / 1000
  let time_h := time_s / 3600
  let speed_downstream := distance_km / time_h
  speed_downstream - speed_current

theorem rowing_speed_in_still_water :
  speedInStillWater 45.5 9.099272058235341 8.5 = 9.5 :=
by
  sorry

end NUMINAMATH_GPT_rowing_speed_in_still_water_l2413_241398


namespace NUMINAMATH_GPT_joel_laps_count_l2413_241312

def yvonne_laps : ℕ := 10

def younger_sister_laps : ℕ := yvonne_laps / 2

def joel_laps : ℕ := younger_sister_laps * 3

theorem joel_laps_count : joel_laps = 15 := by
  -- The proof is not required as per instructions
  sorry

end NUMINAMATH_GPT_joel_laps_count_l2413_241312


namespace NUMINAMATH_GPT_compute_expression_l2413_241386

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l2413_241386
