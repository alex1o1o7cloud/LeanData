import Mathlib

namespace ferry_journey_difference_l734_73413

theorem ferry_journey_difference
  (time_P : ℝ) (speed_P : ℝ) (mult_Q : ℝ) (speed_diff : ℝ)
  (dist_P : ℝ := time_P * speed_P)
  (dist_Q : ℝ := mult_Q * dist_P)
  (speed_Q : ℝ := speed_P + speed_diff)
  (time_Q : ℝ := dist_Q / speed_Q) :
  time_P = 3 ∧ speed_P = 6 ∧ mult_Q = 3 ∧ speed_diff = 3 → time_Q - time_P = 3 := by
  sorry

end ferry_journey_difference_l734_73413


namespace no_integer_solution_for_large_n_l734_73491

theorem no_integer_solution_for_large_n (n : ℕ) (m : ℤ) (h : n ≥ 11) : ¬(m^2 + 2 * 3^n = m * (2^(n+1) - 1)) :=
sorry

end no_integer_solution_for_large_n_l734_73491


namespace find_single_digit_A_l734_73417

theorem find_single_digit_A (A : ℕ) (h1 : A < 10) (h2 : (11 * A)^2 = 5929) : A = 7 := 
sorry

end find_single_digit_A_l734_73417


namespace parallel_lines_iff_a_eq_2_l734_73439

-- Define line equations
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - a + 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y - 2 = 0

-- Prove that a = 2 is necessary and sufficient for the lines to be parallel.
theorem parallel_lines_iff_a_eq_2 (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → ∃ u v : ℝ, l2 a u v → x = u ∧ y = v) ↔ (a = 2) :=
by {
  sorry
}

end parallel_lines_iff_a_eq_2_l734_73439


namespace range_of_independent_variable_l734_73441

theorem range_of_independent_variable (x : ℝ) : (1 - x > 0) → x < 1 :=
by
  sorry

end range_of_independent_variable_l734_73441


namespace algebraic_identity_l734_73442

theorem algebraic_identity (a : ℚ) (h : a + a⁻¹ = 3) : a^2 + a⁻¹^2 = 7 := 
  sorry

end algebraic_identity_l734_73442


namespace calculate_selling_prices_l734_73460

noncomputable def selling_prices
  (cost1 cost2 cost3 : ℝ) (profit1 profit2 profit3 : ℝ) : ℝ × ℝ × ℝ :=
  let selling_price1 := cost1 + (profit1 / 100) * cost1
  let selling_price2 := cost2 + (profit2 / 100) * cost2
  let selling_price3 := cost3 + (profit3 / 100) * cost3
  (selling_price1, selling_price2, selling_price3)

theorem calculate_selling_prices :
  selling_prices 500 750 1000 20 25 30 = (600, 937.5, 1300) :=
by
  sorry

end calculate_selling_prices_l734_73460


namespace no_intersection_points_of_polar_graphs_l734_73451

theorem no_intersection_points_of_polar_graphs :
  let c1_center := (3 / 2, 0)
  let r1 := 3 / 2
  let c2_center := (0, 3)
  let r2 := 3
  let distance_between_centers := Real.sqrt ((3 / 2 - 0) ^ 2 + (0 - 3) ^ 2)
  distance_between_centers > r1 + r2 :=
by
  sorry

end no_intersection_points_of_polar_graphs_l734_73451


namespace train_speed_correct_l734_73405

def length_of_train := 280 -- in meters
def time_to_pass_tree := 16 -- in seconds
def speed_of_train := 63 -- in km/hr

theorem train_speed_correct :
  (length_of_train / time_to_pass_tree) * (3600 / 1000) = speed_of_train :=
sorry

end train_speed_correct_l734_73405


namespace estimate_yellow_balls_l734_73434

theorem estimate_yellow_balls (m : ℕ) (h1: (5 : ℝ) / (5 + m) = 0.2) : m = 20 :=
  sorry

end estimate_yellow_balls_l734_73434


namespace geom_series_ratio_l734_73467

noncomputable def geomSeries (a q : ℝ) (n : ℕ) : ℝ :=
a * ((1 - q ^ n) / (1 - q))

theorem geom_series_ratio (a1 q : ℝ) (h : 8 * a1 * q + a1 * q^4 = 0) :
  (geomSeries a1 q 5) / (geomSeries a1 q 2) = -11 :=
sorry

end geom_series_ratio_l734_73467


namespace ratio_rocks_eaten_to_collected_l734_73456

def rocks_collected : ℕ := 10
def rocks_left : ℕ := 7
def rocks_spit_out : ℕ := 2

theorem ratio_rocks_eaten_to_collected : 
  (rocks_collected - rocks_left + rocks_spit_out) * 2 = rocks_collected := 
by 
  sorry

end ratio_rocks_eaten_to_collected_l734_73456


namespace greatest_value_of_sum_l734_73482

theorem greatest_value_of_sum (x y : ℝ) (h₁ : x^2 + y^2 = 100) (h₂ : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
by
  sorry

end greatest_value_of_sum_l734_73482


namespace quadratic_function_proof_l734_73420

noncomputable def quadratic_function_condition (a b c : ℝ) :=
  ∀ x : ℝ, ((-3 ≤ x ∧ x ≤ 1) → (a * x^2 + b * x + c) ≤ 0) ∧
           ((x < -3 ∨ 1 < x) → (a * x^2 + b * x + c) > 0) ∧
           (a * 2^2 + b * 2 + c) = 5

theorem quadratic_function_proof (a b c : ℝ) (m : ℝ)
  (h : quadratic_function_condition a b c) :
  (a = 1 ∧ b = 2 ∧ c = -3) ∧ (m ≥ -7/9 ↔ ∃ x : ℝ, a * x^2 + b * x + c = 9 * m + 3) :=
by
  sorry

end quadratic_function_proof_l734_73420


namespace fraction_of_repeating_decimal_l734_73465

theorem fraction_of_repeating_decimal :
  ∃ (f : ℚ), f = 0.73 ∧ f = 73 / 99 := by
  sorry

end fraction_of_repeating_decimal_l734_73465


namespace sum_of_octahedron_faces_l734_73480

theorem sum_of_octahedron_faces (n : ℕ) :
  n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 8 * n + 28 :=
by
  sorry

end sum_of_octahedron_faces_l734_73480


namespace gcd_of_polynomial_and_multiple_of_12600_l734_73487

theorem gcd_of_polynomial_and_multiple_of_12600 (x : ℕ) (h : 12600 ∣ x) : gcd ((5 * x + 7) * (11 * x + 3) * (17 * x + 8) * (4 * x + 5)) x = 840 := by
  sorry

end gcd_of_polynomial_and_multiple_of_12600_l734_73487


namespace find_side_AB_l734_73447

theorem find_side_AB 
  (B C : ℝ) (BC : ℝ) (hB : B = 45) (hC : C = 45) (hBC : BC = 10) : 
  ∃ AB : ℝ, AB = 5 * Real.sqrt 2 :=
by
  -- We add 'sorry' here to indicate that the proof is not provided.
  sorry

end find_side_AB_l734_73447


namespace range_of_a_l734_73496

variable (a : ℝ)

theorem range_of_a (ha : a ≥ 1/4) : ¬ ∃ x : ℝ, a * x^2 + x + 1 < 0 := sorry

end range_of_a_l734_73496


namespace profit_is_correct_l734_73430

-- Define the constants for expenses
def cost_of_lemons : ℕ := 10
def cost_of_sugar : ℕ := 5
def cost_of_cups : ℕ := 3

-- Define the cost per cup of lemonade
def price_per_cup : ℕ := 4

-- Define the number of cups sold
def cups_sold : ℕ := 21

-- Define the total revenue
def total_revenue : ℕ := cups_sold * price_per_cup

-- Define the total expenses
def total_expenses : ℕ := cost_of_lemons + cost_of_sugar + cost_of_cups

-- Define the profit
def profit : ℕ := total_revenue - total_expenses

-- The theorem stating the profit
theorem profit_is_correct : profit = 66 := by
  sorry

end profit_is_correct_l734_73430


namespace invalid_inverse_statement_l734_73436

/- Define the statements and their inverses -/

/-- Statement A: Vertical angles are equal. -/
def statement_A : Prop := ∀ {α β : ℝ}, α ≠ β → α = β

/-- Inverse of Statement A: If two angles are equal, then they are vertical angles. -/
def inverse_A : Prop := ∀ {α β : ℝ}, α = β → α ≠ β

/-- Statement B: If |a| = |b|, then a = b. -/
def statement_B (a b : ℝ) : Prop := abs a = abs b → a = b

/-- Inverse of Statement B: If a = b, then |a| = |b|. -/
def inverse_B (a b : ℝ) : Prop := a = b → abs a = abs b

/-- Statement C: If two lines are parallel, then the alternate interior angles are equal. -/
def statement_C (l1 l2 : Prop) : Prop := l1 → l2

/-- Inverse of Statement C: If the alternate interior angles are equal, then the two lines are parallel. -/
def inverse_C (l1 l2 : Prop) : Prop := l2 → l1

/-- Statement D: If a^2 = b^2, then a = b. -/
def statement_D (a b : ℝ) : Prop := a^2 = b^2 → a = b

/-- Inverse of Statement D: If a = b, then a^2 = b^2. -/
def inverse_D (a b : ℝ) : Prop := a = b → a^2 = b^2

/-- The statement that does not have a valid inverse among A, B, C, and D is statement A. -/
theorem invalid_inverse_statement : ¬inverse_A :=
by
sorry

end invalid_inverse_statement_l734_73436


namespace computer_production_per_month_l734_73404

def days : ℕ := 28
def hours_per_day : ℕ := 24
def intervals_per_hour : ℕ := 2
def computers_per_interval : ℕ := 3

theorem computer_production_per_month : 
  (days * hours_per_day * intervals_per_hour * computers_per_interval = 4032) :=
by sorry

end computer_production_per_month_l734_73404


namespace certain_number_is_18_l734_73484

theorem certain_number_is_18 (p q : ℚ) (h₁ : 3 / p = 8) (h₂ : p - q = 0.20833333333333334) : 3 / q = 18 :=
sorry

end certain_number_is_18_l734_73484


namespace f_of_3_l734_73424

def f (x : ℕ) : ℤ :=
  if x = 0 then sorry else 2 * (x - 1) - 1  -- Define an appropriate value for f(0) later

theorem f_of_3 : f 3 = 3 := by
  sorry

end f_of_3_l734_73424


namespace factorize_poly1_l734_73483

variable (a : ℝ)

theorem factorize_poly1 : a^4 + 2 * a^3 + 1 = (a + 1) * (a^3 + a^2 - a + 1) := 
sorry

end factorize_poly1_l734_73483


namespace solve_quadratic_eq_l734_73412

theorem solve_quadratic_eq (a b x : ℝ) :
  12 * a * b * x^2 - (16 * a^2 - 9 * b^2) * x - 12 * a * b = 0 ↔ (x = 4 * a / (3 * b)) ∨ (x = -3 * b / (4 * a)) :=
by
  sorry

end solve_quadratic_eq_l734_73412


namespace coins_in_second_stack_l734_73469

theorem coins_in_second_stack (total_coins : ℕ) (stack1_coins : ℕ) (stack2_coins : ℕ) 
  (H1 : total_coins = 12) (H2 : stack1_coins = 4) : stack2_coins = 8 :=
by
  -- The proof is omitted.
  sorry

end coins_in_second_stack_l734_73469


namespace grace_mowing_hours_l734_73445

-- Definitions for conditions
def earnings_mowing (x : ℕ) : ℕ := 6 * x
def earnings_weeds : ℕ := 11 * 9
def earnings_mulch : ℕ := 9 * 10
def total_september_earnings (x : ℕ) : ℕ := earnings_mowing x + earnings_weeds + earnings_mulch

-- Proof statement (with the total earnings of 567 specified)
theorem grace_mowing_hours (x : ℕ) (h : total_september_earnings x = 567) : x = 63 := by
  sorry

end grace_mowing_hours_l734_73445


namespace intersection_A_B_is_1_and_2_l734_73407

def A : Set ℝ := {x | x ^ 2 - 3 * x - 4 < 0}
def B : Set ℝ := {-2, -1, 1, 2, 4}

theorem intersection_A_B_is_1_and_2 : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_is_1_and_2_l734_73407


namespace distance_each_player_runs_l734_73463

-- Definitions based on conditions
def length : ℝ := 100
def width : ℝ := 50
def laps : ℝ := 6

def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

def total_distance (l w laps : ℝ) : ℝ := laps * perimeter l w

-- Theorem statement
theorem distance_each_player_runs :
  total_distance length width laps = 1800 := 
by 
  sorry

end distance_each_player_runs_l734_73463


namespace scientific_notation_equivalence_l734_73488

/-- The scientific notation for 20.26 thousand hectares in square meters is equal to 2.026 × 10^9. -/
theorem scientific_notation_equivalence :
  (20.26 * 10^3 * 10^4) = 2.026 * 10^9 := 
sorry

end scientific_notation_equivalence_l734_73488


namespace power_mod_five_l734_73440

theorem power_mod_five (n : ℕ) (hn : n ≡ 0 [MOD 4]): (3^2000 ≡ 1 [MOD 5]) :=
by 
  sorry

end power_mod_five_l734_73440


namespace cubic_polynomial_roots_l734_73435

theorem cubic_polynomial_roots (a : ℚ) :
  (x^3 - 6*x^2 + a*x - 6 = 0) ∧ (x = 3) → (x = 1 ∨ x = 2 ∨ x = 3) :=
by
  sorry

end cubic_polynomial_roots_l734_73435


namespace hyperbola_distance_property_l734_73416

theorem hyperbola_distance_property (P : ℝ × ℝ)
  (hP_on_hyperbola : (P.1 ^ 2 / 16) - (P.2 ^ 2 / 9) = 1)
  (h_dist_15 : dist P (5, 0) = 15) :
  dist P (-5, 0) = 7 ∨ dist P (-5, 0) = 23 := 
sorry

end hyperbola_distance_property_l734_73416


namespace learn_at_least_537_words_l734_73489

theorem learn_at_least_537_words (total_words : ℕ) (guess_percentage : ℝ) (required_percentage : ℝ) :
  total_words = 600 → guess_percentage = 0.05 → required_percentage = 0.90 → 
  ∀ (words_learned : ℕ), words_learned ≥ 537 → 
  (words_learned + guess_percentage * (total_words - words_learned)) / total_words ≥ required_percentage :=
by
  intros h_total_words h_guess_percentage h_required_percentage words_learned h_words_learned
  sorry

end learn_at_least_537_words_l734_73489


namespace purely_imaginary_complex_number_l734_73402

theorem purely_imaginary_complex_number (m : ℝ) :
  (m^2 - 2 * m - 3 = 0) ∧ (m^2 - 4 * m + 3 ≠ 0) → m = -1 :=
by
  sorry

end purely_imaginary_complex_number_l734_73402


namespace miguel_run_time_before_ariana_catches_up_l734_73481

theorem miguel_run_time_before_ariana_catches_up
  (head_start : ℕ := 20)
  (ariana_speed : ℕ := 6)
  (miguel_speed : ℕ := 4)
  (head_start_distance : ℕ := miguel_speed * head_start)
  (t_catchup : ℕ := (head_start_distance) / (ariana_speed - miguel_speed))
  (total_time : ℕ := t_catchup + head_start) :
  total_time = 60 := sorry

end miguel_run_time_before_ariana_catches_up_l734_73481


namespace sharon_trip_distance_l734_73449

theorem sharon_trip_distance
  (x : ℝ)
  (usual_speed : ℝ := x / 180)
  (reduced_speed : ℝ := usual_speed - 1/3)
  (time_before_storm : ℝ := (x / 3) / usual_speed)
  (time_during_storm : ℝ := (2 * x / 3) / reduced_speed)
  (total_trip_time : ℝ := 276)
  (h : time_before_storm + time_during_storm = total_trip_time) :
  x = 135 :=
sorry

end sharon_trip_distance_l734_73449


namespace quadratic_rewrite_ab_value_l734_73499

theorem quadratic_rewrite_ab_value:
  ∃ a b c : ℤ, (∀ x: ℝ, 16*x^2 + 40*x + 18 = (a*x + b)^2 + c) ∧ a * b = 20 :=
by
  -- We'll add the definitions derived from conditions here
  sorry

end quadratic_rewrite_ab_value_l734_73499


namespace total_oranges_picked_l734_73423

theorem total_oranges_picked (mary_oranges : Nat) (jason_oranges : Nat) (hmary : mary_oranges = 122) (hjason : jason_oranges = 105) : mary_oranges + jason_oranges = 227 := by
  sorry

end total_oranges_picked_l734_73423


namespace spelling_bee_students_count_l734_73444

theorem spelling_bee_students_count (x : ℕ) (h1 : x / 2 * 1 / 4 * 2 = 30) : x = 240 :=
by
  sorry

end spelling_bee_students_count_l734_73444


namespace combined_total_time_l734_73409

def jerry_time : ℕ := 3
def elaine_time : ℕ := 2 * jerry_time
def george_time : ℕ := elaine_time / 3
def kramer_time : ℕ := 0
def total_time : ℕ := jerry_time + elaine_time + george_time + kramer_time

theorem combined_total_time : total_time = 11 := by
  unfold total_time jerry_time elaine_time george_time kramer_time
  rfl

end combined_total_time_l734_73409


namespace calories_needed_l734_73475

def calories_per_orange : ℕ := 80
def cost_per_orange : ℝ := 1.2
def initial_amount : ℝ := 10
def remaining_amount : ℝ := 4

theorem calories_needed : calories_per_orange * (initial_amount - remaining_amount) / cost_per_orange = 400 := 
by 
  sorry

end calories_needed_l734_73475


namespace spinner_probabilities_l734_73478

theorem spinner_probabilities (pA pB pC pD : ℚ) (h1 : pA = 1/4) (h2 : pB = 1/3) (h3 : pA + pB + pC + pD = 1) :
  pC + pD = 5/12 :=
by
  -- Here you would construct the proof (left as sorry for this example)
  sorry

end spinner_probabilities_l734_73478


namespace new_sample_variance_l734_73415

-- Definitions based on conditions
def sample_size (original : Nat) : Prop := original = 7
def sample_average (original : ℝ) : Prop := original = 5
def sample_variance (original : ℝ) : Prop := original = 2
def new_data_point (point : ℝ) : Prop := point = 5

-- Statement to be proved
theorem new_sample_variance (original_size : Nat) (original_avg : ℝ) (original_var : ℝ) (new_point : ℝ) 
  (h₁ : sample_size original_size) 
  (h₂ : sample_average original_avg) 
  (h₃ : sample_variance original_var) 
  (h₄ : new_data_point new_point) : 
  (8 * original_var + 0) / 8 = 7 / 4 := 
by 
  sorry

end new_sample_variance_l734_73415


namespace hours_of_rain_l734_73493

def totalHours : ℕ := 9
def noRainHours : ℕ := 5
def rainHours : ℕ := totalHours - noRainHours

theorem hours_of_rain : rainHours = 4 := by
  sorry

end hours_of_rain_l734_73493


namespace prove_monotonic_increasing_range_l734_73408

open Real

noncomputable def problem_statement : Prop :=
  ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (∀ x > 0, (a^x + (1 + a)^x) ≤ (a^(x+1) + (1 + a)^(x+1))) ∧
  (a ≥ (sqrt 5 - 1) / 2 ∧ a < 1)

theorem prove_monotonic_increasing_range : problem_statement := sorry

end prove_monotonic_increasing_range_l734_73408


namespace angle_CBE_minimal_l734_73458

theorem angle_CBE_minimal
    (ABC ABD DBE: ℝ)
    (h1: ABC = 40)
    (h2: ABD = 28)
    (h3: DBE = 10) : 
    CBE = 2 :=
by
  sorry

end angle_CBE_minimal_l734_73458


namespace min_value_expression_l734_73492

theorem min_value_expression (x : ℝ) (h : x > 3) : x + 4 / (x - 3) ≥ 7 :=
sorry

end min_value_expression_l734_73492


namespace calculate_expression_l734_73419

theorem calculate_expression :
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 10.5 = 10.5 :=
by
  sorry

end calculate_expression_l734_73419


namespace sum_of_integers_with_even_product_l734_73473

theorem sum_of_integers_with_even_product (a b : ℤ) (h : ∃ k, a * b = 2 * k) : 
∃ k1 k2, a = 2 * k1 ∨ a = 2 * k1 + 1 ∧ (a + b = 2 * k2 ∨ a + b = 2 * k2 + 1) :=
by
  sorry

end sum_of_integers_with_even_product_l734_73473


namespace water_usage_l734_73486

noncomputable def litres_per_household_per_month (total_litres : ℕ) (number_of_households : ℕ) : ℕ :=
  total_litres / number_of_households

theorem water_usage : litres_per_household_per_month 2000 10 = 200 :=
by
  sorry

end water_usage_l734_73486


namespace delta_value_l734_73428

theorem delta_value (Δ : ℤ) (h : 4 * -3 = Δ - 3) : Δ = -9 :=
sorry

end delta_value_l734_73428


namespace average_speed_l734_73479

theorem average_speed (d1 d2 d3 d4 d5 t: ℕ) 
  (h1: d1 = 120) 
  (h2: d2 = 70) 
  (h3: d3 = 90) 
  (h4: d4 = 110) 
  (h5: d5 = 80) 
  (total_time: t = 5): 
  (d1 + d2 + d3 + d4 + d5) / t = 94 := 
by 
  -- proof will go here
  sorry

end average_speed_l734_73479


namespace inequality_for_positive_reals_l734_73421

variable (a b c : ℝ)

theorem inequality_for_positive_reals (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by
  sorry

end inequality_for_positive_reals_l734_73421


namespace distance_between_parallel_lines_l734_73437

theorem distance_between_parallel_lines
  (line1 : ∀ (x y : ℝ), 3*x - 2*y - 1 = 0)
  (line2 : ∀ (x y : ℝ), 3*x - 2*y + 1 = 0) :
  ∃ d : ℝ, d = (2 * Real.sqrt 13) / 13 :=
by
  sorry

end distance_between_parallel_lines_l734_73437


namespace cube_truncation_edges_l734_73459

-- Define the initial condition: a cube
def initial_cube_edges : ℕ := 12

-- Define the condition of each corner being cut off
def corners_cut (corners : ℕ) (edges_added : ℕ) : ℕ :=
  corners * edges_added

-- Define the proof problem
theorem cube_truncation_edges : initial_cube_edges + corners_cut 8 3 = 36 := by
  sorry

end cube_truncation_edges_l734_73459


namespace valid_factorizations_of_1870_l734_73462

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_valid_factor1 (n : ℕ) : Prop := 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ n = p1 * p2

def is_valid_factor2 (n : ℕ) : Prop := 
  ∃ (p k : ℕ), is_prime p ∧ (k = 4 ∨ k = 6 ∨ k = 8 ∨ k = 9) ∧ n = p * k

theorem valid_factorizations_of_1870 : 
  ∃ a b : ℕ, a * b = 1870 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ 
  ((is_valid_factor1 a ∧ is_valid_factor2 b) ∨ (is_valid_factor1 b ∧ is_valid_factor2 a)) ∧ 
  (a = 34 ∧ b = 55 ∨ a = 55 ∧ b = 34) ∧ 
  (¬∃ x y : ℕ, x * y = 1870 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ 
  ((is_valid_factor1 x ∧ is_valid_factor2 y) ∨ (is_valid_factor1 y ∧ is_valid_factor2 x)) ∧ 
  (x ≠ 34 ∨ y ≠ 55 ∨ x ≠ 55 ∨ y ≠ 34)) :=
sorry

end valid_factorizations_of_1870_l734_73462


namespace ratio_of_b_to_a_is_4_l734_73457

theorem ratio_of_b_to_a_is_4 (b a : ℚ) (h1 : b = 4 * a) (h2 : b = 15 - 4 * a) : a = 15 / 8 := by
  sorry

end ratio_of_b_to_a_is_4_l734_73457


namespace John_reads_50_pages_per_hour_l734_73429

noncomputable def pages_per_hour (reads_daily hours : ℕ) (total_pages total_weeks : ℕ) : ℕ :=
  let days := total_weeks * 7
  let pages_per_day := total_pages / days
  pages_per_day / reads_daily

theorem John_reads_50_pages_per_hour :
  pages_per_hour 2 2800 4 = 50 := by
  sorry

end John_reads_50_pages_per_hour_l734_73429


namespace ratio_of_sums_l734_73425

open Nat

def sum_multiples_of_3 (n : Nat) : Nat :=
  let m := n / 3
  m * (3 + 3 * m) / 2

def sum_first_n_integers (n : Nat) : Nat :=
  n * (n + 1) / 2

theorem ratio_of_sums :
  (sum_multiples_of_3 600) / (sum_first_n_integers 300) = 4 / 3 :=
by
  sorry

end ratio_of_sums_l734_73425


namespace tan_identity_proof_l734_73468

theorem tan_identity_proof :
  (1 - Real.tan (100 * Real.pi / 180)) * (1 - Real.tan (35 * Real.pi / 180)) = 2 :=
by
  have tan_135 : Real.tan (135 * Real.pi / 180) = -1 := by sorry -- This needs a separate proof.
  have tan_sum_formula : ∀ A B : ℝ, Real.tan (A + B) = (Real.tan A + Real.tan B) / (1 - Real.tan A * Real.tan B) := by sorry -- This needs a deeper exploration
  sorry -- Main proof to be filled

end tan_identity_proof_l734_73468


namespace find_c_l734_73448

theorem find_c (c d : ℝ) (h1 : c < 0) (h2 : d > 0)
    (max_min_condition : ∀ x, c * Real.cos (d * x) ≤ 3 ∧ c * Real.cos (d * x) ≥ -3) :
    c = -3 :=
by
  -- The statement says if c < 0, d > 0, and given the cosine function hitting max 3 and min -3, then c = -3.
  sorry

end find_c_l734_73448


namespace systematic_sampling_example_l734_73450

theorem systematic_sampling_example : 
  ∃ (a : ℕ → ℕ), (∀ i : ℕ, 5 ≤ i ∧ i ≤ 5 → a i = 5 + 10 * (i - 1)) ∧ 
  ∀ i : ℕ, 1 ≤ i ∧ i < 6 → a i - a (i - 1) = a (i + 1) - a i :=
sorry

end systematic_sampling_example_l734_73450


namespace no_possible_numbering_for_equal_sidesum_l734_73472

theorem no_possible_numbering_for_equal_sidesum (O : Point) (A : Fin 10 → Point) 
  (side_numbers : (Fin 10) → ℕ) (segment_numbers : (Fin 10) → ℕ) : 
  ¬ ∃ (side_segment_sum_equal : Fin 10 → ℕ) (sum_equal : ℕ),
    (∀ i, side_segment_sum_equal i = side_numbers i + segment_numbers i) ∧ 
    (∀ i, side_segment_sum_equal i = sum_equal) := 
sorry

end no_possible_numbering_for_equal_sidesum_l734_73472


namespace part_a_part_b_part_c_part_d_l734_73410

-- define the partitions function
def P (k l n : ℕ) : ℕ := sorry

-- Part (a) statement
theorem part_a (k l n : ℕ) :
  P k l n - P k (l - 1) n = P (k - 1) l (n - l) :=
sorry

-- Part (b) statement
theorem part_b (k l n : ℕ) :
  P k l n - P (k - 1) l n = P k (l - 1) (n - k) :=
sorry

-- Part (c) statement
theorem part_c (k l n : ℕ) :
  P k l n = P l k n :=
sorry

-- Part (d) statement
theorem part_d (k l n : ℕ) :
  P k l n = P k l (k * l - n) :=
sorry

end part_a_part_b_part_c_part_d_l734_73410


namespace max_positive_root_satisfies_range_l734_73490

noncomputable def max_positive_root_in_range (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 1.5) (hd : |d| ≤ 1) : Prop :=
  ∃ s : ℝ, 2.5 ≤ s ∧ s < 3 ∧ ∃ x : ℝ, x > 0 ∧ x^3 + b * x^2 + c * x + d = 0

theorem max_positive_root_satisfies_range (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 1.5) (hd : |d| ≤ 1) :
  max_positive_root_in_range b c d hb hc hd := sorry

end max_positive_root_satisfies_range_l734_73490


namespace total_discount_is_15_l734_73474

structure Item :=
  (price : ℝ)      -- Regular price
  (discount_rate : ℝ) -- Discount rate in decimal form

def t_shirt : Item := {price := 25, discount_rate := 0.3}
def jeans : Item := {price := 75, discount_rate := 0.1}

def discount (item : Item) : ℝ :=
  item.discount_rate * item.price

def total_discount (items : List Item) : ℝ :=
  items.map discount |>.sum

theorem total_discount_is_15 :
  total_discount [t_shirt, jeans] = 15 := by
  sorry

end total_discount_is_15_l734_73474


namespace avg_median_max_k_m_r_s_t_l734_73432

theorem avg_median_max_k_m_r_s_t (
  k m r s t : ℕ 
) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
  (h5 : 5 * 16 = k + m + r + s + t)
  (h6 : r = 17) : 
  t = 42 :=
by
  sorry

end avg_median_max_k_m_r_s_t_l734_73432


namespace last_number_in_first_set_l734_73497

variables (x y : ℕ)

def mean (a b c d e : ℕ) : ℕ :=
  (a + b + c + d + e) / 5

theorem last_number_in_first_set :
  (mean 28 x 42 78 y = 90) ∧ (mean 128 255 511 1023 x = 423) → y = 104 :=
by 
  sorry

end last_number_in_first_set_l734_73497


namespace measure_of_angle_x_in_triangle_l734_73422

theorem measure_of_angle_x_in_triangle
  (x : ℝ)
  (h1 : x + 2 * x + 45 = 180) :
  x = 45 :=
sorry

end measure_of_angle_x_in_triangle_l734_73422


namespace problem_ab_cd_l734_73470

theorem problem_ab_cd
    (a b c d : ℝ)
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (habcd : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h1 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2012)
    (h2 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2012) :
  (ab)^2012 - (cd)^2012 = -2012 := 
sorry

end problem_ab_cd_l734_73470


namespace find_primes_l734_73401

open Int

theorem find_primes (p x y : ℕ) (hp : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  p ^ x = y ^ 3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
by
  sorry

end find_primes_l734_73401


namespace x_value_l734_73476

theorem x_value (x : ℤ) (h : x = (2009^2 - 2009) / 2009) : x = 2008 := by
  sorry

end x_value_l734_73476


namespace proof_problem_l734_73485

-- Define the rates of P and Q
def P_rate : ℚ := 1/3
def Q_rate : ℚ := 1/18

-- Define the time they work together
def combined_time : ℚ := 2

-- Define the job completion rates
def combined_rate (P_rate Q_rate : ℚ) : ℚ := P_rate + Q_rate

-- Define the job completed together in given time
def job_completed_together (rate time : ℚ) : ℚ := rate * time

-- Define the remaining job
def remaining_job (total_job completed_job : ℚ) : ℚ := total_job - completed_job

-- Define the time required for P to complete the remaining job
def time_for_P (P_rate remaining_job : ℚ) : ℚ := remaining_job / P_rate

-- Define the total job as 1
def total_job : ℚ := 1

-- Correct answer in minutes
def correct_answer_in_minutes (time_in_hours : ℚ) : ℚ := time_in_hours * 60

-- Problem statement
theorem proof_problem : 
  correct_answer_in_minutes (time_for_P P_rate (remaining_job total_job 
    (job_completed_together (combined_rate P_rate Q_rate) combined_time))) = 40 := 
by
  sorry

end proof_problem_l734_73485


namespace earliest_meeting_time_l734_73403

theorem earliest_meeting_time
    (charlie_lap : ℕ := 5)
    (ben_lap : ℕ := 8)
    (laura_lap_effective : ℕ := 11) :
    lcm (lcm charlie_lap ben_lap) laura_lap_effective = 440 := by
  sorry

end earliest_meeting_time_l734_73403


namespace guise_hot_dogs_l734_73477

theorem guise_hot_dogs (x : ℤ) (h1 : x + (x + 2) + (x + 4) = 36) : x = 10 :=
by
  sorry

end guise_hot_dogs_l734_73477


namespace vector_magnitude_positive_l734_73454

variable {V : Type} [NormedAddCommGroup V] [NormedSpace ℝ V]

variables (a b : V)

-- Given: 
-- a is any non-zero vector
-- b is a unit vector
theorem vector_magnitude_positive (ha : a ≠ 0) (hb : ‖b‖ = 1) : ‖a‖ > 0 := 
sorry

end vector_magnitude_positive_l734_73454


namespace greatest_perimeter_triangle_l734_73426

theorem greatest_perimeter_triangle :
  ∃ (x : ℕ), (x > (16 / 5)) ∧ (x < (16 / 3)) ∧ ((x = 4 ∨ x = 5) → 4 * x + x + 16 = 41) :=
by
  sorry

end greatest_perimeter_triangle_l734_73426


namespace complex_abs_sum_eq_1_or_3_l734_73433

open Complex

theorem complex_abs_sum_eq_1_or_3 (a b c : ℂ) (ha : abs a = 1) (hb : abs b = 1) (hc : abs c = 1) 
  (h : a^3/(b^2 * c) + b^3/(a^2 * c) + c^3/(a^2 * b) = 1) : abs (a + b + c) = 1 ∨ abs (a + b + c) = 3 := 
by
  sorry

end complex_abs_sum_eq_1_or_3_l734_73433


namespace triangle_similarity_length_RY_l734_73452

theorem triangle_similarity_length_RY
  (P Q R X Y Z : Type)
  [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (PQ : ℝ) (XY : ℝ) (RY_length : ℝ)
  (h1 : PQ = 10)
  (h2 : XY = 6)
  (h3 : ∀ (PR QR PX QX RZ : ℝ) (angle_PY_RZ : ℝ),
    PR + RY_length = PX ∧
    QR + RY_length = QX ∧ 
    angle_PY_RZ = 120 ∧
    PR > 0 ∧ QR > 0 ∧ RY_length > 0)
  (h4 : XY / PQ = RY_length / (PQ + RY_length)) :
  RY_length = 15 := by
  sorry

end triangle_similarity_length_RY_l734_73452


namespace two_digit_number_reverse_sum_eq_99_l734_73461

theorem two_digit_number_reverse_sum_eq_99 :
  ∃ a b : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ ((10 * a + b) - (10 * b + a) = 5 * (a + b))
  ∧ (10 * a + b) + (10 * b + a) = 99 := 
by
  sorry

end two_digit_number_reverse_sum_eq_99_l734_73461


namespace area_difference_depends_only_on_bw_l734_73411

variable (b w n : ℕ)
variable (hb : b ≥ 2)
variable (hw : w ≥ 2)
variable (hn : n = b + w)

/-- Given conditions: 
1. \(b \geq 2\) 
2. \(w \geq 2\) 
3. \(n = b + w\)
4. There are \(2b\) identical black rods and \(2w\) identical white rods, each of side length 1. 
5. These rods form a regular \(2n\)-gon with parallel sides of the same color.
6. A convex \(2b\)-gon \(B\) is formed by translating the black rods. 
7. A convex \(2w\) A convex \(2w\)-gon \(W\) is formed by translating the white rods. 
Prove that the difference of the areas of \(B\) and \(W\) depends only on the numbers \(b\) and \(w\). -/
theorem area_difference_depends_only_on_bw :
  ∀ (A B W : ℝ), A - B = 2 * (b - w) :=
sorry

end area_difference_depends_only_on_bw_l734_73411


namespace remaining_soup_can_feed_adults_l734_73400

-- Define initial conditions
def cans_per_soup_for_children : ℕ := 6
def cans_per_soup_for_adults : ℕ := 4
def initial_cans : ℕ := 8
def children_to_feed : ℕ := 24

-- Define the problem statement in Lean
theorem remaining_soup_can_feed_adults :
  (initial_cans - (children_to_feed / cans_per_soup_for_children)) * cans_per_soup_for_adults = 16 := by
  sorry

end remaining_soup_can_feed_adults_l734_73400


namespace gcd_lcm_find_other_number_l734_73414

theorem gcd_lcm_find_other_number {a b : ℕ} (h_gcd : Nat.gcd a b = 36) (h_lcm : Nat.lcm a b = 8820) (h_a : a = 360) : b = 882 :=
by
  sorry

end gcd_lcm_find_other_number_l734_73414


namespace angle_in_third_quadrant_l734_73427

theorem angle_in_third_quadrant
  (α : ℝ)
  (k : ℤ)
  (h : (π / 2) + 2 * (↑k) * π < α ∧ α < π + 2 * (↑k) * π) :
  π + 2 * (↑k) * π < (π / 2) + α ∧ (π / 2) + α < (3 * π / 2) + 2 * (↑k) * π :=
by
  sorry

end angle_in_third_quadrant_l734_73427


namespace tens_digit_8_pow_2023_l734_73443

theorem tens_digit_8_pow_2023 : (8 ^ 2023 % 100) / 10 % 10 = 1 := 
sorry

end tens_digit_8_pow_2023_l734_73443


namespace prob1_prob2_l734_73453

theorem prob1:
  (6 * (Real.tan (30 * Real.pi / 180))^2 - Real.sqrt 3 * Real.sin (60 * Real.pi / 180) - 2 * Real.sin (45 * Real.pi / 180)) = (1 / 2 - Real.sqrt 2) :=
sorry

theorem prob2:
  ((Real.sqrt 2 / 2) * Real.cos (45 * Real.pi / 180) - (Real.tan (40 * Real.pi / 180) + 1)^0 + Real.sqrt (1 / 4) + Real.sin (30 * Real.pi / 180)) = (1 / 2) :=
sorry

end prob1_prob2_l734_73453


namespace car_cleaning_ratio_l734_73455

theorem car_cleaning_ratio
    (outside_cleaning_time : ℕ)
    (total_cleaning_time : ℕ)
    (h1 : outside_cleaning_time = 80)
    (h2 : total_cleaning_time = 100) :
    (total_cleaning_time - outside_cleaning_time) / outside_cleaning_time = 1 / 4 :=
by
  sorry

end car_cleaning_ratio_l734_73455


namespace sqrt_expression_eq_twelve_l734_73466

theorem sqrt_expression_eq_twelve : Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt 27) = 12 := 
sorry

end sqrt_expression_eq_twelve_l734_73466


namespace least_number_to_subtract_l734_73495

theorem least_number_to_subtract (x : ℕ) :
  1439 - x ≡ 3 [MOD 5] ∧ 
  1439 - x ≡ 3 [MOD 11] ∧ 
  1439 - x ≡ 3 [MOD 13] ↔ 
  x = 9 :=
by sorry

end least_number_to_subtract_l734_73495


namespace range_of_c_over_a_l734_73446

theorem range_of_c_over_a (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + b + c = 0) : -2 < c / a ∧ c / a < -1 :=
by {
  sorry
}

end range_of_c_over_a_l734_73446


namespace marbles_left_calculation_l734_73494

/-- A magician starts with 20 red marbles and 30 blue marbles.
    He removes 3 red marbles and 12 blue marbles. We need to 
    prove that he has 35 marbles left in total. -/
theorem marbles_left_calculation (initial_red : ℕ) (initial_blue : ℕ) (removed_red : ℕ) 
    (removed_blue : ℕ) (H1 : initial_red = 20) (H2 : initial_blue = 30) 
    (H3 : removed_red = 3) (H4 : removed_blue = 4 * removed_red) :
    (initial_red - removed_red) + (initial_blue - removed_blue) = 35 :=
by
   -- sorry to skip the proof
   sorry

end marbles_left_calculation_l734_73494


namespace factorize_expr1_factorize_expr2_l734_73471

-- Define the expressions
def expr1 (m x y : ℝ) : ℝ := 3 * m * x - 6 * m * y
def expr2 (x : ℝ) : ℝ := 1 - 25 * x^2

-- Define the factorized forms
def factorized_expr1 (m x y : ℝ) : ℝ := 3 * m * (x - 2 * y)
def factorized_expr2 (x : ℝ) : ℝ := (1 + 5 * x) * (1 - 5 * x)

-- Proof problems
theorem factorize_expr1 (m x y : ℝ) : expr1 m x y = factorized_expr1 m x y := sorry
theorem factorize_expr2 (x : ℝ) : expr2 x = factorized_expr2 x := sorry

end factorize_expr1_factorize_expr2_l734_73471


namespace find_x_l734_73498

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -1)
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

theorem find_x
  (h : dot_product vector_a (vector_b x) = 0) :
  x = 2 :=
by
  sorry

end find_x_l734_73498


namespace amount_of_H2O_formed_l734_73431

-- Define the balanced chemical equation as a relation
def balanced_equation : Prop :=
  ∀ (naoh hcl nacl h2o : ℕ), 
    (naoh + hcl = nacl + h2o)

-- Define the reaction of 2 moles of NaOH and 2 moles of HCl
def reaction (naoh hcl : ℕ) : ℕ :=
  if (naoh = 2) ∧ (hcl = 2) then 2 else 0

theorem amount_of_H2O_formed :
  balanced_equation →
  reaction 2 2 = 2 :=
by
  sorry

end amount_of_H2O_formed_l734_73431


namespace cat_weights_ratio_l734_73464

variable (meg_cat_weight : ℕ) (anne_extra_weight : ℕ) (meg_cat_weight := 20) (anne_extra_weight := 8)

/-- The ratio of the weight of Meg's cat to the weight of Anne's cat -/
theorem cat_weights_ratio : (meg_cat_weight / Nat.gcd meg_cat_weight (meg_cat_weight + anne_extra_weight)) 
                            = 5 ∧ ((meg_cat_weight + anne_extra_weight) / Nat.gcd meg_cat_weight (meg_cat_weight + anne_extra_weight)) 
                            = 7 := by
  sorry

end cat_weights_ratio_l734_73464


namespace probability_point_between_C_and_D_l734_73406

theorem probability_point_between_C_and_D :
  ∀ (A B C D E : ℝ), A < B ∧ C < D ∧
  (B - A = 4 * (D - A)) ∧ (B - A = 4 * (B - E)) ∧
  (D - A = C - D) ∧ (C - D = E - C) ∧ (E - C = B - E) →
  (B - A ≠ 0) → 
  (C - D) / (B - A) = 1 / 4 :=
by
  intros A B C D E hAB hNonZero
  sorry

end probability_point_between_C_and_D_l734_73406


namespace inequality_negatives_l734_73418

theorem inequality_negatives (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : a^2 > b^2 :=
sorry

end inequality_negatives_l734_73418


namespace intersection_A_B_l734_73438

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | x < 2 }

theorem intersection_A_B : A ∩ B = { x | -1 ≤ x ∧ x < 2 } := 
by 
  sorry

end intersection_A_B_l734_73438
