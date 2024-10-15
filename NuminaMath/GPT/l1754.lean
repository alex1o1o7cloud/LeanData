import Mathlib

namespace NUMINAMATH_GPT_intersection_at_one_point_l1754_175497

theorem intersection_at_one_point (m : ℝ) :
  (∃ x : ℝ, (m - 4) * x^2 - 2 * m * x - m - 6 = 0 ∧
            ∀ x' : ℝ, (m - 4) * x'^2 - 2 * m * x' - m - 6 = 0 → x' = x) ↔
  m = -4 ∨ m = 3 ∨ m = 4 := 
by
  sorry

end NUMINAMATH_GPT_intersection_at_one_point_l1754_175497


namespace NUMINAMATH_GPT_lcm_5_6_8_18_l1754_175427

/-- The least common multiple of the numbers 5, 6, 8, and 18 is 360. -/
theorem lcm_5_6_8_18 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 18) = 360 := by
  sorry

end NUMINAMATH_GPT_lcm_5_6_8_18_l1754_175427


namespace NUMINAMATH_GPT_no_such_constant_l1754_175430

noncomputable def f : ℚ → ℚ := sorry

theorem no_such_constant (h : ∀ x y : ℚ, ∃ k : ℤ, f (x + y) - f x - f y = k) :
  ¬ ∃ c : ℚ, ∀ x : ℚ, ∃ k : ℤ, f x - c * x = k := 
sorry

end NUMINAMATH_GPT_no_such_constant_l1754_175430


namespace NUMINAMATH_GPT_percent_democrats_is_60_l1754_175457
-- Import the necessary library

-- Define the problem conditions
variables (D R : ℝ)
variables (h1 : D + R = 100)
variables (h2 : 0.70 * D + 0.20 * R = 50)

-- State the theorem to be proved
theorem percent_democrats_is_60 (D R : ℝ) (h1 : D + R = 100) (h2 : 0.70 * D + 0.20 * R = 50) : D = 60 :=
by
  sorry

end NUMINAMATH_GPT_percent_democrats_is_60_l1754_175457


namespace NUMINAMATH_GPT_problem_statement_l1754_175491

theorem problem_statement (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) : y * (y - 1) ≤ x^2 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1754_175491


namespace NUMINAMATH_GPT_minimum_value_of_y_exists_l1754_175434

theorem minimum_value_of_y_exists :
  ∃ (y : ℝ), (∀ (x : ℝ), (y + x) = (y - x)^2 + 3 * (y - x) + 3) ∧ y = -1/2 :=
by sorry

end NUMINAMATH_GPT_minimum_value_of_y_exists_l1754_175434


namespace NUMINAMATH_GPT_square_perimeter_l1754_175458

theorem square_perimeter (a : ℝ) (side : ℝ) (perimeter : ℝ) (h1 : a = 144) (h2 : side = Real.sqrt a) (h3 : perimeter = 4 * side) : perimeter = 48 := by
  sorry

end NUMINAMATH_GPT_square_perimeter_l1754_175458


namespace NUMINAMATH_GPT_average_increase_l1754_175446

-- Define the conditions as Lean definitions
def runs_in_17th_inning : ℕ := 50
def average_after_17th_inning : ℕ := 18

-- The condition about the average increase can be written as follows
theorem average_increase 
  (initial_average: ℕ) -- The batsman's average after the 16th inning
  (h1: runs_in_17th_inning = 50)
  (h2: average_after_17th_inning = 18)
  (h3: 16 * initial_average + runs_in_17th_inning = 17 * average_after_17th_inning) :
  average_after_17th_inning - initial_average = 2 := 
sorry

end NUMINAMATH_GPT_average_increase_l1754_175446


namespace NUMINAMATH_GPT_find_x_minus_y_l1754_175482

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_x_minus_y_l1754_175482


namespace NUMINAMATH_GPT_average_speed_l1754_175418

theorem average_speed (d1 d2 : ℝ) (t1 t2 : ℝ) (h1 : d1 = 90) (h2 : d2 = 75) (ht1 : t1 = 1) (ht2 : t2 = 1) :
  (d1 + d2) / (t1 + t2) = 82.5 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l1754_175418


namespace NUMINAMATH_GPT_poly_roots_arith_progression_l1754_175441

theorem poly_roots_arith_progression (a b c : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, -- There exist roots x₁, x₂, x₃
    (x₁ + x₃ = 2 * x₂) ∧ -- Roots form an arithmetic progression
    (x₁ * x₂ * x₃ = -c) ∧ -- Roots satisfy polynomial's product condition
    (x₁ + x₂ + x₃ = -a) ∧ -- Roots satisfy polynomial's sum condition
    ((x₁ * x₂) + (x₂ * x₃) + (x₃ * x₁) = b)) -- Roots satisfy polynomial's sum of products condition
  → (2 * a^3 / 27 - a * b / 3 + c = 0) := 
sorry -- proof is not required

end NUMINAMATH_GPT_poly_roots_arith_progression_l1754_175441


namespace NUMINAMATH_GPT_graph_passes_through_fixed_point_l1754_175412

theorem graph_passes_through_fixed_point (a : ℝ) : (0, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a ^ x + 1) } :=
sorry

end NUMINAMATH_GPT_graph_passes_through_fixed_point_l1754_175412


namespace NUMINAMATH_GPT_length_of_each_movie_l1754_175492

-- Defining the amount of time Grandpa Lou watched movies on Tuesday in minutes
def time_tuesday : ℕ := 4 * 60 + 30   -- 4 hours and 30 minutes

-- Defining the number of movies watched on Tuesday
def movies_tuesday (x : ℕ) : Prop := time_tuesday / x = 90

-- Defining the total number of movies watched in both days
def total_movies_two_days (x : ℕ) : Prop := x + 2 * x = 9

theorem length_of_each_movie (x : ℕ) (h₁ : total_movies_two_days x) (h₂ : movies_tuesday x) : time_tuesday / x = 90 :=
by
  -- Given the conditions, we can prove the statement:
  sorry

end NUMINAMATH_GPT_length_of_each_movie_l1754_175492


namespace NUMINAMATH_GPT_part3_l1754_175471

noncomputable def f (x a : ℝ) : ℝ := x^2 - (2*a + 1)*x + a * Real.log x

theorem part3 (a : ℝ) : 
  (∀ x > 1, f x a > 0) ↔ a ∈ Set.Iic 0 := 
sorry

end NUMINAMATH_GPT_part3_l1754_175471


namespace NUMINAMATH_GPT_ac_plus_bd_eq_neg_10_l1754_175451

theorem ac_plus_bd_eq_neg_10 (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 3)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 6) :
  a * c + b * d = -10 :=
by
  sorry

end NUMINAMATH_GPT_ac_plus_bd_eq_neg_10_l1754_175451


namespace NUMINAMATH_GPT_x_pow_4_plus_inv_x_pow_4_l1754_175437

theorem x_pow_4_plus_inv_x_pow_4 (x : ℝ) (h : x^2 - 15 * x + 1 = 0) : x^4 + (1 / x^4) = 49727 :=
by
  sorry

end NUMINAMATH_GPT_x_pow_4_plus_inv_x_pow_4_l1754_175437


namespace NUMINAMATH_GPT_anna_pays_total_l1754_175478

-- Define the conditions
def daily_rental_cost : ℝ := 35
def cost_per_mile : ℝ := 0.25
def rental_days : ℝ := 3
def miles_driven : ℝ := 300

-- Define the total cost function
def total_cost (daily_rental_cost cost_per_mile rental_days miles_driven : ℝ) : ℝ :=
  (daily_rental_cost * rental_days) + (cost_per_mile * miles_driven)

-- The statement to be proved
theorem anna_pays_total : total_cost daily_rental_cost cost_per_mile rental_days miles_driven = 180 :=
by
  sorry

end NUMINAMATH_GPT_anna_pays_total_l1754_175478


namespace NUMINAMATH_GPT_prob_zhang_nings_wins_2_1_correct_prob_ξ_minus_2_correct_prob_ξ_minus_1_correct_prob_ξ_1_correct_prob_ξ_2_correct_expected_value_ξ_correct_l1754_175442

noncomputable def prob_zhang_nings_wins_2_1 :=
  2 * 0.4 * 0.6 * 0.6 = 0.288

theorem prob_zhang_nings_wins_2_1_correct : prob_zhang_nings_wins_2_1 := sorry

def prob_ξ_minus_2 := 0.4 * 0.4 = 0.16
def prob_ξ_minus_1 := 2 * 0.4 * 0.6 * 0.4 = 0.192
def prob_ξ_1 := 2 * 0.4 * 0.6 * 0.6 = 0.288
def prob_ξ_2 := 0.6 * 0.6 = 0.36

theorem prob_ξ_minus_2_correct : prob_ξ_minus_2 := sorry
theorem prob_ξ_minus_1_correct : prob_ξ_minus_1 := sorry
theorem prob_ξ_1_correct : prob_ξ_1 := sorry
theorem prob_ξ_2_correct : prob_ξ_2 := sorry

noncomputable def expected_value_ξ :=
  (-2 * 0.16) + (-1 * 0.192) + (1 * 0.288) + (2 * 0.36) = 0.496

theorem expected_value_ξ_correct : expected_value_ξ := sorry

end NUMINAMATH_GPT_prob_zhang_nings_wins_2_1_correct_prob_ξ_minus_2_correct_prob_ξ_minus_1_correct_prob_ξ_1_correct_prob_ξ_2_correct_expected_value_ξ_correct_l1754_175442


namespace NUMINAMATH_GPT_find_numbers_l1754_175498

theorem find_numbers (x y : ℝ) (r : ℝ) (d : ℝ) 
  (h_geom_x : x = 5 * r) 
  (h_geom_y : y = 5 * r^2)
  (h_arith_1 : y = x + d) 
  (h_arith_2 : 15 = y + d) : 
  x + y = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1754_175498


namespace NUMINAMATH_GPT_smallest_b_value_l1754_175410

theorem smallest_b_value (a b c : ℕ) (h0 : a > 0) (h1 : b > 0) (h2 : c > 0)
  (h3 : (31 : ℚ) / 72 = (a : ℚ) / 8 + (b : ℚ) / 9 - c) :
  b = 5 :=
sorry

end NUMINAMATH_GPT_smallest_b_value_l1754_175410


namespace NUMINAMATH_GPT_min_abs_sum_of_x1_x2_l1754_175495

open Real

theorem min_abs_sum_of_x1_x2 (x1 x2 : ℝ) (h : 1 / ((2 + sin x1) * (2 + sin (2 * x2))) = 1) : 
  abs (x1 + x2) = π / 4 :=
sorry

end NUMINAMATH_GPT_min_abs_sum_of_x1_x2_l1754_175495


namespace NUMINAMATH_GPT_dennis_initial_money_l1754_175453

theorem dennis_initial_money :
  let cost_of_shirts := 27
  let change_bills := 2 * 10
  let change_coins := 3
  let total_change := change_bills + change_coins
  cost_of_shirts + total_change = 50 :=
by
  let cost_of_shirts := 27
  let change_bills := 2 * 10
  let change_coins := 3
  let total_change := change_bills + change_coins
  show cost_of_shirts + total_change = 50
  sorry

end NUMINAMATH_GPT_dennis_initial_money_l1754_175453


namespace NUMINAMATH_GPT_number_of_integers_with_abs_val_conditions_l1754_175469

theorem number_of_integers_with_abs_val_conditions : 
  (∃ n : ℕ, n = 8) :=
by sorry

end NUMINAMATH_GPT_number_of_integers_with_abs_val_conditions_l1754_175469


namespace NUMINAMATH_GPT_greatest_integer_difference_l1754_175422

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x) (hx2 : x < 6) (hy : 6 < y) (hy2 : y < 10) :
  ∃ d : ℤ, d = y - x ∧ d = 5 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_difference_l1754_175422


namespace NUMINAMATH_GPT_find_b_of_roots_condition_l1754_175465

theorem find_b_of_roots_condition
  (α β : ℝ)
  (h1 : α * β = -1)
  (h2 : α + β = -b)
  (h3 : α * β - 2 * α - 2 * β = -11) :
  b = -5 := 
  sorry

end NUMINAMATH_GPT_find_b_of_roots_condition_l1754_175465


namespace NUMINAMATH_GPT_union_of_sets_l1754_175462

def set_M : Set ℕ := {0, 1, 3}
def set_N : Set ℕ := {x | ∃ (a : ℕ), a ∈ set_M ∧ x = 3 * a}

theorem union_of_sets :
  set_M ∪ set_N = {0, 1, 3, 9} :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l1754_175462


namespace NUMINAMATH_GPT_gcd_765432_654321_l1754_175486

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_765432_654321_l1754_175486


namespace NUMINAMATH_GPT_find_S_2013_l1754_175401

variable {a : ℕ → ℤ} -- the arithmetic sequence
variable {S : ℕ → ℤ} -- the sum of the first n terms

-- Conditions
axiom a1_eq_neg2011 : a 1 = -2011
axiom sum_sequence : ∀ n, S n = (n * (a 1 + a n)) / 2
axiom condition_eq : (S 2012 / 2012) - (S 2011 / 2011) = 1

-- The Lean statement to prove that S 2013 = 2013
theorem find_S_2013 : S 2013 = 2013 := by
  sorry

end NUMINAMATH_GPT_find_S_2013_l1754_175401


namespace NUMINAMATH_GPT_faster_train_speed_correct_l1754_175415

noncomputable def speed_of_faster_train (V_s_kmph : ℝ) (length_faster_train_m : ℝ) (time_s : ℝ) : ℝ :=
  let V_s_mps := V_s_kmph * (1000 / 3600)
  let V_r_mps := length_faster_train_m / time_s
  let V_f_mps := V_r_mps - V_s_mps
  V_f_mps * (3600 / 1000)

theorem faster_train_speed_correct : 
  speed_of_faster_train 36 90.0072 4 = 45.00648 := 
by
  sorry

end NUMINAMATH_GPT_faster_train_speed_correct_l1754_175415


namespace NUMINAMATH_GPT_Ki_tae_pencils_l1754_175426

theorem Ki_tae_pencils (P B : ℤ) (h1 : P + B = 12) (h2 : 1000 * P + 1300 * B = 15000) : P = 2 :=
sorry

end NUMINAMATH_GPT_Ki_tae_pencils_l1754_175426


namespace NUMINAMATH_GPT_green_peaches_per_basket_l1754_175463

-- Definitions based on given conditions
def total_peaches : ℕ := 10
def red_peaches_per_basket : ℕ := 4

-- Theorem statement based on the question and correct answer
theorem green_peaches_per_basket :
  (total_peaches - red_peaches_per_basket) = 6 := 
by
  sorry

end NUMINAMATH_GPT_green_peaches_per_basket_l1754_175463


namespace NUMINAMATH_GPT_tomatoes_ruined_and_discarded_l1754_175431

theorem tomatoes_ruined_and_discarded 
  (W : ℝ)
  (C : ℝ)
  (P : ℝ)
  (S : ℝ)
  (profit_percentage : ℝ)
  (initial_cost : C = 0.80 * W)
  (remaining_tomatoes : S = 0.9956)
  (desired_profit : profit_percentage = 0.12)
  (final_cost : 0.896 = 0.80 + 0.096) :
  0.9956 * (1 - P / 100) = 0.896 :=
by
  sorry

end NUMINAMATH_GPT_tomatoes_ruined_and_discarded_l1754_175431


namespace NUMINAMATH_GPT_combined_volleyball_percentage_l1754_175468

theorem combined_volleyball_percentage (students_north: ℕ) (students_south: ℕ)
(percent_volleyball_north percent_volleyball_south: ℚ)
(H1: students_north = 1800) (H2: percent_volleyball_north = 0.25)
(H3: students_south = 2700) (H4: percent_volleyball_south = 0.35):
  (((students_north * percent_volleyball_north) + (students_south * percent_volleyball_south))
  / (students_north + students_south) * 100) = 31 := 
  sorry

end NUMINAMATH_GPT_combined_volleyball_percentage_l1754_175468


namespace NUMINAMATH_GPT_difference_in_combined_area_l1754_175419

-- Define the dimensions of the two rectangular sheets of paper
def paper1_length : ℝ := 11
def paper1_width : ℝ := 17
def paper2_length : ℝ := 8.5
def paper2_width : ℝ := 11

-- Define the areas of one side of each sheet
def area1 : ℝ := paper1_length * paper1_width -- 187
def area2 : ℝ := paper2_length * paper2_width -- 93.5

-- Define the combined areas of front and back of each sheet
def combined_area1 : ℝ := 2 * area1 -- 374
def combined_area2 : ℝ := 2 * area2 -- 187

-- Prove that the difference in combined area is 187
theorem difference_in_combined_area : combined_area1 - combined_area2 = 187 :=
by 
  -- Using the definitions above to simplify the goal
  sorry

end NUMINAMATH_GPT_difference_in_combined_area_l1754_175419


namespace NUMINAMATH_GPT_max_gcd_is_2_l1754_175423

-- Define the sequence
def a (n : ℕ) : ℕ := 101 + (n + 1)^2 + 3 * n

-- Define the gcd of consecutive terms
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_is_2 : ∀ n : ℕ, n > 0 → d n = 2 :=
by
  intros n hn
  dsimp [d]
  sorry

end NUMINAMATH_GPT_max_gcd_is_2_l1754_175423


namespace NUMINAMATH_GPT_odd_function_property_l1754_175459

theorem odd_function_property {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = - f x) (h2 : ∀ x, f (1 + x) = f (-x)) (h3 : f (-1 / 3) = 1 / 3) : f (5 / 3) = 1 / 3 := 
sorry

end NUMINAMATH_GPT_odd_function_property_l1754_175459


namespace NUMINAMATH_GPT_select_best_player_l1754_175490

theorem select_best_player : 
  (average_A = 9.6 ∧ variance_A = 0.25) ∧ 
  (average_B = 9.5 ∧ variance_B = 0.27) ∧ 
  (average_C = 9.5 ∧ variance_C = 0.30) ∧ 
  (average_D = 9.6 ∧ variance_D = 0.23) → 
  best_player = D := 
by 
  sorry

end NUMINAMATH_GPT_select_best_player_l1754_175490


namespace NUMINAMATH_GPT_crayons_left_l1754_175447

def initial_crayons : ℕ := 253
def lost_or_given_away_crayons : ℕ := 70
def remaining_crayons : ℕ := 183

theorem crayons_left (initial_crayons : ℕ) (lost_or_given_away_crayons : ℕ) (remaining_crayons : ℕ) :
  initial_crayons - lost_or_given_away_crayons = remaining_crayons :=
by {
  sorry
}

end NUMINAMATH_GPT_crayons_left_l1754_175447


namespace NUMINAMATH_GPT_students_per_bus_l1754_175499

theorem students_per_bus
  (total_students : ℕ)
  (buses : ℕ)
  (students_in_cars : ℕ)
  (h1 : total_students = 375)
  (h2 : buses = 7)
  (h3 : students_in_cars = 4) :
  (total_students - students_in_cars) / buses = 53 :=
by
  sorry

end NUMINAMATH_GPT_students_per_bus_l1754_175499


namespace NUMINAMATH_GPT_combine_like_terms_l1754_175404

theorem combine_like_terms (a : ℝ) : 2 * a + 3 * a = 5 * a := 
by sorry

end NUMINAMATH_GPT_combine_like_terms_l1754_175404


namespace NUMINAMATH_GPT_fraction_decomposition_l1754_175444

theorem fraction_decomposition :
  ∃ (A B : ℚ), 
  (A = 27 / 10) ∧ (B = -11 / 10) ∧ 
  (∀ x : ℚ, 
    7 * x - 13 = A * (3 * x - 4) + B * (x + 2)) := 
  sorry

end NUMINAMATH_GPT_fraction_decomposition_l1754_175444


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1754_175488

variable (A : Set ℝ)
variable (B : Set ℝ)
variable (C : Set ℝ)

theorem intersection_of_A_and_B (hA : A = { x | -1 < x ∧ x < 3 })
                                (hB : B = { -1, 1, 2 })
                                (hC : C = { 1, 2 }) :
  A ∩ B = C := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1754_175488


namespace NUMINAMATH_GPT_simplify_fraction_144_1008_l1754_175435

theorem simplify_fraction_144_1008 :
  (144 : ℤ) / (1008 : ℤ) = (1 : ℤ) / (7 : ℤ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_144_1008_l1754_175435


namespace NUMINAMATH_GPT_initial_men_l1754_175472

variable (P M : ℕ) -- P represents the provisions and M represents the initial number of men.

-- Conditons
def provision_lasts_20_days : Prop := P / (M * 20) = P / ((M + 200) * 15)

-- The proof problem
theorem initial_men (h : provision_lasts_20_days P M) : M = 600 :=
sorry

end NUMINAMATH_GPT_initial_men_l1754_175472


namespace NUMINAMATH_GPT_algebraic_expression_l1754_175470

-- Definition for the problem expressed in Lean
def number_one_less_than_three_times (a : ℝ) : ℝ :=
  3 * a - 1

-- Theorem stating the proof problem
theorem algebraic_expression (a : ℝ) : number_one_less_than_three_times a = 3 * a - 1 :=
by
  -- Proof steps would go here; omitted as per instructions
  sorry

end NUMINAMATH_GPT_algebraic_expression_l1754_175470


namespace NUMINAMATH_GPT_enjoyable_gameplay_l1754_175428

theorem enjoyable_gameplay (total_hours : ℕ) (boring_percentage : ℕ) (expansion_hours : ℕ)
  (h_total : total_hours = 100)
  (h_boring : boring_percentage = 80)
  (h_expansion : expansion_hours = 30) :
  ((1 - boring_percentage / 100) * total_hours + expansion_hours) = 50 := 
by
  sorry

end NUMINAMATH_GPT_enjoyable_gameplay_l1754_175428


namespace NUMINAMATH_GPT_kohen_apples_l1754_175445

theorem kohen_apples (B : ℕ) (h1 : 300 * B = 4 * 750) : B = 10 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_kohen_apples_l1754_175445


namespace NUMINAMATH_GPT_permutation_value_l1754_175438

theorem permutation_value : ∀ (n r : ℕ), n = 5 → r = 3 → (n.choose r) * r.factorial = 60 := 
by
  intros n r hn hr 
  rw [hn, hr]
  -- We use the permutation formula A_{n}^{r} = n! / (n-r)!
  -- A_{5}^{3} = 5! / 2!
  -- Simplifies to 5 * 4 * 3 = 60.
  sorry

end NUMINAMATH_GPT_permutation_value_l1754_175438


namespace NUMINAMATH_GPT_mean_value_of_pentagon_angles_l1754_175474

theorem mean_value_of_pentagon_angles : 
  let n := 5 
  let interior_angle_sum := (n - 2) * 180 
  mean_angle = interior_angle_sum / n :=
  sorry

end NUMINAMATH_GPT_mean_value_of_pentagon_angles_l1754_175474


namespace NUMINAMATH_GPT_division_remainder_l1754_175496

theorem division_remainder (dividend divisor quotient : ℕ) (h_dividend : dividend = 131) (h_divisor : divisor = 14) (h_quotient : quotient = 9) :
  ∃ remainder : ℕ, dividend = divisor * quotient + remainder ∧ remainder = 5 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_l1754_175496


namespace NUMINAMATH_GPT_growth_pattern_equation_l1754_175409

theorem growth_pattern_equation (x : ℕ) :
  1 + x + x^2 = 73 :=
sorry

end NUMINAMATH_GPT_growth_pattern_equation_l1754_175409


namespace NUMINAMATH_GPT_total_work_completed_in_days_l1754_175448

-- Define the number of days Amit can complete the work
def amit_days : ℕ := 15

-- Define the number of days Ananthu can complete the work
def ananthu_days : ℕ := 90

-- Define the number of days Amit worked
def amit_work_days : ℕ := 3

-- Calculate the amount of work Amit can do in one day
def amit_work_day_rate : ℚ := 1 / amit_days

-- Calculate the amount of work Ananthu can do in one day
def ananthu_work_day_rate : ℚ := 1 / ananthu_days

-- Calculate the total work completed
theorem total_work_completed_in_days :
  amit_work_days * amit_work_day_rate + (1 - amit_work_days * amit_work_day_rate) / ananthu_work_day_rate = 75 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_work_completed_in_days_l1754_175448


namespace NUMINAMATH_GPT_factor_expr_l1754_175456

def expr1 (x : ℝ) := 16 * x^6 + 49 * x^4 - 9
def expr2 (x : ℝ) := 4 * x^6 - 14 * x^4 - 9

theorem factor_expr (x : ℝ) :
  (expr1 x - expr2 x) = 3 * x^4 * (4 * x^2 + 21) := 
by
  sorry

end NUMINAMATH_GPT_factor_expr_l1754_175456


namespace NUMINAMATH_GPT_task_completion_time_l1754_175436

theorem task_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ t : ℝ, t = (a * b) / (a + b) := 
sorry

end NUMINAMATH_GPT_task_completion_time_l1754_175436


namespace NUMINAMATH_GPT_intersection_M_N_l1754_175408

def M : Set ℝ := { x : ℝ | x^2 > 4 }
def N : Set ℝ := { x : ℝ | x = -3 ∨ x = -2 ∨ x = 2 ∨ x = 3 ∨ x = 4 }

theorem intersection_M_N : M ∩ N = { x : ℝ | x = -3 ∨ x = 3 ∨ x = 4 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1754_175408


namespace NUMINAMATH_GPT_pills_supply_duration_l1754_175464

open Nat

-- Definitions based on conditions
def one_third_pill_every_three_days : ℕ := 1 / 3 * 3
def pills_in_bottle : ℕ := 90
def days_per_pill : ℕ := 9
def days_per_month : ℕ := 30

-- The Lean statement to prove the question == answer given conditions
theorem pills_supply_duration : (pills_in_bottle * days_per_pill) / days_per_month = 27 := by
  sorry

end NUMINAMATH_GPT_pills_supply_duration_l1754_175464


namespace NUMINAMATH_GPT_skill_of_passing_through_walls_l1754_175476

theorem skill_of_passing_through_walls (k n : ℕ) (h : k = 8) (h_eq : k * Real.sqrt (k / (k * k - 1)) = Real.sqrt (k * k / (k * k - 1))) : n = k * k - 1 :=
by sorry

end NUMINAMATH_GPT_skill_of_passing_through_walls_l1754_175476


namespace NUMINAMATH_GPT_general_solution_of_diff_eq_l1754_175417

theorem general_solution_of_diff_eq {C1 C2 : ℝ} (y : ℝ → ℝ) (x : ℝ) :
  (∀ x, y x = C1 * Real.exp (-x) + C2 * Real.exp (-2 * x) + x^2 - 5 * x - 2) →
  (∀ x, (deriv (deriv y)) x + 3 * (deriv y) x + 2 * y x = 2 * x^2 - 4 * x - 17) :=
by
  intro hy
  sorry

end NUMINAMATH_GPT_general_solution_of_diff_eq_l1754_175417


namespace NUMINAMATH_GPT_time_for_a_and_b_together_l1754_175489

variable (R_a R_b : ℝ)
variable (T_ab : ℝ)

-- Given conditions
def condition_1 : Prop := R_a = 3 * R_b
def condition_2 : Prop := R_a * 28 = 1  -- '1' denotes the entire work

-- Proof goal
theorem time_for_a_and_b_together (h1 : condition_1 R_a R_b) (h2 : condition_2 R_a) : T_ab = 21 := 
by
  sorry

end NUMINAMATH_GPT_time_for_a_and_b_together_l1754_175489


namespace NUMINAMATH_GPT_percentage_of_good_fruits_l1754_175454

theorem percentage_of_good_fruits (total_oranges : ℕ) (total_bananas : ℕ) 
    (rotten_oranges_percent : ℝ) (rotten_bananas_percent : ℝ) :
    total_oranges = 600 ∧ total_bananas = 400 ∧ 
    rotten_oranges_percent = 0.15 ∧ rotten_bananas_percent = 0.03 →
    (510 + 388) / (600 + 400) * 100 = 89.8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_percentage_of_good_fruits_l1754_175454


namespace NUMINAMATH_GPT_find_arc_length_of_sector_l1754_175443

variable (s r p : ℝ)
variable (h_s : s = 4)
variable (h_r : r = 2)
variable (h_area : 2 * s = r * p)

theorem find_arc_length_of_sector 
  (h_s : s = 4) (h_r : r = 2) (h_area : 2 * s = r * p) :
  p = 4 :=
sorry

end NUMINAMATH_GPT_find_arc_length_of_sector_l1754_175443


namespace NUMINAMATH_GPT_maximum_PM_minus_PN_l1754_175483

noncomputable def x_squared_over_9_minus_y_squared_over_16_eq_1 (x y : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / 16) = 1

noncomputable def circle1 (x y : ℝ) : Prop :=
  (x + 5)^2 + y^2 = 4

noncomputable def circle2 (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 1

theorem maximum_PM_minus_PN :
  ∀ (P M N : ℝ × ℝ),
    x_squared_over_9_minus_y_squared_over_16_eq_1 P.1 P.2 →
    circle1 M.1 M.2 →
    circle2 N.1 N.2 →
    (|dist P M - dist P N| ≤ 9) := sorry

end NUMINAMATH_GPT_maximum_PM_minus_PN_l1754_175483


namespace NUMINAMATH_GPT_quadricycles_count_l1754_175479

theorem quadricycles_count (s q : ℕ) (hsq : s + q = 9) (hw : 2 * s + 4 * q = 30) : q = 6 :=
by
  sorry

end NUMINAMATH_GPT_quadricycles_count_l1754_175479


namespace NUMINAMATH_GPT_painted_cube_count_is_three_l1754_175400

-- Define the colors of the faces
inductive Color
| Yellow
| Black
| White

-- Define a Cube with painted faces
structure Cube :=
(f1 f2 f3 f4 f5 f6 : Color)

-- Define rotational symmetry (two cubes are the same under rotation)
def equivalentUpToRotation (c1 c2 : Cube) : Prop := sorry -- Symmetry function

-- Define a property that counts the correct painting configuration
def paintedCubeCount : ℕ :=
  sorry -- Function to count correctly painted and uniquely identifiable cubes

theorem painted_cube_count_is_three :
  paintedCubeCount = 3 :=
sorry

end NUMINAMATH_GPT_painted_cube_count_is_three_l1754_175400


namespace NUMINAMATH_GPT_verify_conditions_l1754_175449

-- Define the conditions as expressions
def condition_A (a : ℝ) : Prop := 2 * a * 3 * a = 6 * a
def condition_B (a b : ℝ) : Prop := 3 * a^2 * b - 3 * a * b^2 = 0
def condition_C (a : ℝ) : Prop := 6 * a / (2 * a) = 3
def condition_D (a : ℝ) : Prop := (-2 * a) ^ 3 = -6 * a^3

-- Prove which condition is correct
theorem verify_conditions (a b : ℝ) (h : a ≠ 0) : 
  ¬ condition_A a ∧ ¬ condition_B a b ∧ condition_C a ∧ ¬ condition_D a :=
by 
  sorry

end NUMINAMATH_GPT_verify_conditions_l1754_175449


namespace NUMINAMATH_GPT_man_l1754_175485

theorem man's_rowing_speed_in_still_water
  (river_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (H_river_speed : river_speed = 2)
  (H_total_time : total_time = 1)
  (H_total_distance : total_distance = 5.333333333333333) :
  ∃ (v : ℝ), 
    v = 7.333333333333333 ∧
    ∀ d,
    d = total_distance / 2 →
    d = (v - river_speed) * (total_time / 2) ∧
    d = (v + river_speed) * (total_time / 2) := 
by
  sorry

end NUMINAMATH_GPT_man_l1754_175485


namespace NUMINAMATH_GPT_price_reduction_l1754_175420

theorem price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h : original_price = 289) (h2 : final_price = 256) :
  289 * (1 - x) ^ 2 = 256 := sorry

end NUMINAMATH_GPT_price_reduction_l1754_175420


namespace NUMINAMATH_GPT_proof_min_value_a3_and_a2b2_l1754_175493

noncomputable def min_value_a3_and_a2b2 (a1 a2 a3 b1 b2 b3 : ℝ) : Prop :=
  (a1 > 0) ∧ (a2 > 0) ∧ (a3 > 0) ∧ (b1 > 0) ∧ (b2 > 0) ∧ (b3 > 0) ∧
  (a2 = a1 + b1) ∧ (a3 = a1 + 2 * b1) ∧ (b2 = b1 * a1) ∧ 
  (b3 = b1 * a1^2) ∧ (a3 = b3) ∧ 
  (a3 = 3 * Real.sqrt 6 / 2) ∧
  (a2 * b2 = 15 * Real.sqrt 6 / 8) 

theorem proof_min_value_a3_and_a2b2 : ∃ (a1 a2 a3 b1 b2 b3 : ℝ), min_value_a3_and_a2b2 a1 a2 a3 b1 b2 b3 :=
by
  use 2*Real.sqrt 6/3, 5*Real.sqrt 6/4, 3*Real.sqrt 6/2, Real.sqrt 6/4, 3/2, 3*Real.sqrt 6/2
  sorry

end NUMINAMATH_GPT_proof_min_value_a3_and_a2b2_l1754_175493


namespace NUMINAMATH_GPT_find_y_intercept_l1754_175411

def line_y_intercept (m x y : ℝ) (pt : ℝ × ℝ) : ℝ :=
  let y_intercept := pt.snd - m * pt.fst
  y_intercept

theorem find_y_intercept (m x y b : ℝ) (pt : ℝ × ℝ) (h1 : m = 2) (h2 : pt = (498, 998)) :
  line_y_intercept m x y pt = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_y_intercept_l1754_175411


namespace NUMINAMATH_GPT_fixed_point_of_line_l1754_175466

theorem fixed_point_of_line (m : ℝ) : 
  ∀ (x y : ℝ), (3 * x - 2 * y + 7 = 0) ∧ (4 * x + 5 * y - 6 = 0) → x = -1 ∧ y = 2 :=
sorry

end NUMINAMATH_GPT_fixed_point_of_line_l1754_175466


namespace NUMINAMATH_GPT_a_sequence_arithmetic_sum_of_bn_l1754_175475

   noncomputable def a (n : ℕ) : ℕ := 1 + n

   def S (n : ℕ) : ℕ := n * (n + 1) / 2

   def b (n : ℕ) : ℚ := 1 / S n

   def T (n : ℕ) : ℚ := (Finset.range n).sum b

   theorem a_sequence_arithmetic (n : ℕ) (a_n_positive : ∀ n, a n > 0)
     (a₁_is_one : a 0 = 1) :
     (a (n+1)) - a n = 1 := by
     sorry

   theorem sum_of_bn (n : ℕ) :
     T n = 2 * n / (n + 1) := by
     sorry
   
end NUMINAMATH_GPT_a_sequence_arithmetic_sum_of_bn_l1754_175475


namespace NUMINAMATH_GPT_triangle_area_is_3_max_f_l1754_175494

noncomputable def triangle_area :=
  let a : ℝ := 2
  let b : ℝ := 2 * Real.sqrt 3
  let c : ℝ := 2
  let A : ℝ := Real.pi / 3
  (1 / 2) * b * c * Real.sin A

theorem triangle_area_is_3 :
  triangle_area = 3 := by
  sorry

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.cos x * (Real.sin x * Real.cos (Real.pi / 3) + Real.cos x * Real.sin (Real.pi / 3))

theorem max_f :
  ∃ x ∈ Set.Icc 0 (Real.pi / 3), f x = 2 + Real.sqrt 3 ∧ x = Real.pi / 12 := by
  sorry

end NUMINAMATH_GPT_triangle_area_is_3_max_f_l1754_175494


namespace NUMINAMATH_GPT_cost_of_softball_l1754_175425

theorem cost_of_softball 
  (original_budget : ℕ)
  (dodgeball_cost : ℕ)
  (num_dodgeballs : ℕ)
  (increase_rate : ℚ)
  (num_softballs : ℕ)
  (new_budget : ℕ)
  (softball_cost : ℕ)
  (h0 : original_budget = num_dodgeballs * dodgeball_cost)
  (h1 : increase_rate = 0.20)
  (h2 : new_budget = original_budget + increase_rate * original_budget)
  (h3 : new_budget = num_softballs * softball_cost) :
  softball_cost = 9 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_softball_l1754_175425


namespace NUMINAMATH_GPT_xy_squares_l1754_175473

theorem xy_squares (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 := 
by 
  sorry

end NUMINAMATH_GPT_xy_squares_l1754_175473


namespace NUMINAMATH_GPT_john_new_weekly_earnings_l1754_175450

theorem john_new_weekly_earnings :
  let original_earnings : ℝ := 40
  let percentage_increase : ℝ := 37.5 / 100
  let raise_amount : ℝ := original_earnings * percentage_increase
  let new_weekly_earnings : ℝ := original_earnings + raise_amount
  new_weekly_earnings = 55 := 
by
  sorry

end NUMINAMATH_GPT_john_new_weekly_earnings_l1754_175450


namespace NUMINAMATH_GPT_cindy_dress_discount_l1754_175440

theorem cindy_dress_discount (P D : ℝ) 
  (h1 : P * (1 - D) * 1.25 = 61.2) 
  (h2 : P - 61.2 = 4.5) : D = 0.255 :=
sorry

end NUMINAMATH_GPT_cindy_dress_discount_l1754_175440


namespace NUMINAMATH_GPT_fill_bucket_time_l1754_175406

-- Problem statement:
-- Prove that the time taken to fill the bucket completely is 150 seconds
-- given that two-thirds of the bucket is filled in 100 seconds.

theorem fill_bucket_time (t : ℕ) (h : (2 / 3) * t = 100) : t = 150 :=
by
  -- Proof should be here
  sorry

end NUMINAMATH_GPT_fill_bucket_time_l1754_175406


namespace NUMINAMATH_GPT_max_xyz_squared_l1754_175455

theorem max_xyz_squared 
  (x y z : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (h1 : x * y * z = (14 - x) * (14 - y) * (14 - z)) 
  (h2 : x + y + z < 28) : 
  x^2 + y^2 + z^2 ≤ 219 :=
sorry

end NUMINAMATH_GPT_max_xyz_squared_l1754_175455


namespace NUMINAMATH_GPT_convergent_inequalities_l1754_175414

theorem convergent_inequalities (α : ℝ) (P Q : ℕ → ℤ) (h_convergent : ∀ n ≥ 1, abs (α - P n / Q n) < 1 / (2 * (Q n) ^ 2) ∨ abs (α - P (n - 1) / Q (n - 1)) < 1 / (2 * (Q (n - 1))^2))
  (h_continued_fraction : ∀ n ≥ 1, P (n-1) * Q n - P n * Q (n-1) = (-1)^(n-1)) :
  ∃ p q : ℕ, 0 < q ∧ abs (α - p / q) < 1 / (2 * q^2) :=
sorry

end NUMINAMATH_GPT_convergent_inequalities_l1754_175414


namespace NUMINAMATH_GPT_actor_A_constraints_l1754_175467

-- Definitions corresponding to the conditions.
def numberOfActors : Nat := 6
def positionConstraints : Nat := 4
def permutations (n : Nat) : Nat := Nat.factorial n

-- Lean statement for the proof problem.
theorem actor_A_constraints : 
  (positionConstraints * permutations (numberOfActors - 1)) = 480 := by
sorry

end NUMINAMATH_GPT_actor_A_constraints_l1754_175467


namespace NUMINAMATH_GPT_dimes_difference_l1754_175477

theorem dimes_difference
  (a b c d : ℕ)
  (h1 : a + b + c + d = 150)
  (h2 : 5 * a + 10 * b + 25 * c + 50 * d = 1500) :
  (b = 150 ∨ ∃ c d : ℕ, b = 0 ∧ 4 * c + 9 * d = 150) →
  ∃ b₁ b₂ : ℕ, (b₁ = 150 ∧ b₂ = 0 ∧ b₁ - b₂ = 150) :=
by
  sorry

end NUMINAMATH_GPT_dimes_difference_l1754_175477


namespace NUMINAMATH_GPT_solve_for_q_l1754_175402

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 14) (h2 : 6 * p + 5 * q = 17) : q = -1 / 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_q_l1754_175402


namespace NUMINAMATH_GPT_score_of_tenth_game_must_be_at_least_l1754_175487

variable (score_5 average_9 average_10 score_10 : ℤ)
variable (H1 : average_9 > score_5 / 5)
variable (H2 : average_10 > 18)
variable (score_6 score_7 score_8 score_9 : ℤ)
variable (H3 : score_6 = 23)
variable (H4 : score_7 = 14)
variable (H5 : score_8 = 11)
variable (H6 : score_9 = 20)
variable (H7 : average_9 = (score_5 + score_6 + score_7 + score_8 + score_9) / 9)
variable (H8 : average_10 = (score_5 + score_6 + score_7 + score_8 + score_9 + score_10) / 10)

theorem score_of_tenth_game_must_be_at_least :
  score_10 ≥ 29 :=
by
  sorry

end NUMINAMATH_GPT_score_of_tenth_game_must_be_at_least_l1754_175487


namespace NUMINAMATH_GPT_scientific_notation_of_190_million_l1754_175432

theorem scientific_notation_of_190_million : (190000000 : ℝ) = 1.9 * 10^8 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_190_million_l1754_175432


namespace NUMINAMATH_GPT_simplify_polynomial_l1754_175439

theorem simplify_polynomial :
  (3 * x ^ 5 - 2 * x ^ 3 + 5 * x ^ 2 - 8 * x + 6) + (7 * x ^ 4 + x ^ 3 - 3 * x ^ 2 + x - 9) =
  3 * x ^ 5 + 7 * x ^ 4 - x ^ 3 + 2 * x ^ 2 - 7 * x - 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1754_175439


namespace NUMINAMATH_GPT_negation_proposition_l1754_175484

theorem negation_proposition : 
  ¬(∀ x : ℝ, 0 ≤ x → 2^x > x^2) ↔ ∃ x : ℝ, 0 ≤ x ∧ 2^x ≤ x^2 := by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1754_175484


namespace NUMINAMATH_GPT_certain_number_equals_l1754_175403

theorem certain_number_equals (p q : ℚ) (h1 : 3 / p = 8) (h2 : 3 / q = 18) (h3 : p - q = 0.20833333333333334) : q = 1/6 := sorry

end NUMINAMATH_GPT_certain_number_equals_l1754_175403


namespace NUMINAMATH_GPT_distance_between_centers_l1754_175480

-- Define the points P, Q, R in the plane
variable (P Q R : ℝ × ℝ)

-- Define the lengths PQ, PR, and QR
variable (PQ PR QR : ℝ)
variable (is_right_triangle : ∃ (a b c : ℝ), PQ = a ∧ PR = b ∧ QR = c ∧ a^2 + b^2 = c^2)

-- Define the inradii r1, r2, r3 for triangles PQR, RST, and QUV respectively
variable (r1 r2 r3 : ℝ)

-- Assume PQ = 90, PR = 120, and QR = 150
axiom PQ_length : PQ = 90
axiom PR_length : PR = 120
axiom QR_length : QR = 150

-- Define the centers O2 and O3 of the circles C2 and C3 respectively
variable (O2 O3 : ℝ × ℝ)

-- Assume the inradius length is 30 for the initial triangle
axiom inradius_PQR : r1 = 30

-- Assume the positions of the centers of C2 and C3
axiom O2_position : O2 = (15, 75)
axiom O3_position : O3 = (70, 10)

-- Use the distance formula to express the final result
theorem distance_between_centers : ∃ n : ℕ, dist O2 O3 = Real.sqrt (10 * n) ∧ n = 725 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_centers_l1754_175480


namespace NUMINAMATH_GPT_no_sum_of_cubes_eq_2002_l1754_175433

theorem no_sum_of_cubes_eq_2002 :
  ¬ ∃ (a b c : ℕ), (a ^ 3 + b ^ 3 + c ^ 3 = 2002) :=
sorry

end NUMINAMATH_GPT_no_sum_of_cubes_eq_2002_l1754_175433


namespace NUMINAMATH_GPT_eighth_day_of_april_2000_is_saturday_l1754_175460

noncomputable def april_2000_eight_day_is_saturday : Prop :=
  (∃ n : ℕ, (1 ≤ n ∧ n ≤ 7) ∧
            ((n + 0 * 7) = 2 ∨ (n + 1 * 7) = 2 ∨ (n + 2 * 7) = 2 ∨
             (n + 3 * 7) = 2 ∨ (n + 4 * 7) = 2) ∧
            ((n + 0 * 7) % 2 = 0 ∨ (n + 1 * 7) % 2 = 0 ∨
             (n + 2 * 7) % 2 = 0 ∨ (n + 3 * 7) % 2 = 0 ∨
             (n + 4 * 7) % 2 = 0) ∧
            (∃ k : ℕ, k ≤ 4 ∧ (n + k * 7 = 8))) ∧
            (8 % 7) = 1 ∧ (1 ≠ 0)

theorem eighth_day_of_april_2000_is_saturday :
  april_2000_eight_day_is_saturday := 
sorry

end NUMINAMATH_GPT_eighth_day_of_april_2000_is_saturday_l1754_175460


namespace NUMINAMATH_GPT_floor_neg_seven_fourths_l1754_175407

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end NUMINAMATH_GPT_floor_neg_seven_fourths_l1754_175407


namespace NUMINAMATH_GPT_probability_same_flavor_l1754_175424

theorem probability_same_flavor (num_flavors : ℕ) (num_bags : ℕ) (h1 : num_flavors = 4) (h2 : num_bags = 2) :
  let total_outcomes := num_flavors ^ num_bags
  let favorable_outcomes := num_flavors
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_same_flavor_l1754_175424


namespace NUMINAMATH_GPT_john_remaining_income_l1754_175416

/-- 
  Mr. John's monthly income is $2000, and he spends 5% of his income on public transport.
  Prove that after deducting his monthly transport fare, his remaining income is $1900.
-/
theorem john_remaining_income : 
  let income := 2000 
  let transport_percent := 5 
  let transport_fare := income * transport_percent / 100 
  income - transport_fare = 1900 := 
by 
  let income := 2000 
  let transport_percent := 5 
  let transport_fare := income * transport_percent / 100 
  have transport_fare_eq : transport_fare = 100 := by sorry
  have remaining_income_eq : income - transport_fare = 1900 := by sorry
  exact remaining_income_eq

end NUMINAMATH_GPT_john_remaining_income_l1754_175416


namespace NUMINAMATH_GPT_average_cd_l1754_175421

theorem average_cd (c d: ℝ) (h: (4 + 6 + 8 + c + d) / 5 = 18) : (c + d) / 2 = 36 :=
by sorry

end NUMINAMATH_GPT_average_cd_l1754_175421


namespace NUMINAMATH_GPT_coprime_integer_pairs_sum_285_l1754_175461

theorem coprime_integer_pairs_sum_285 : 
  (∃ s : Finset (ℕ × ℕ), 
    ∀ p ∈ s, p.1 + p.2 = 285 ∧ Nat.gcd p.1 p.2 = 1 ∧ s.card = 72) := sorry

end NUMINAMATH_GPT_coprime_integer_pairs_sum_285_l1754_175461


namespace NUMINAMATH_GPT_distance_between_trains_l1754_175481

def speed_train1 : ℝ := 11 -- Speed of the first train in mph
def speed_train2 : ℝ := 31 -- Speed of the second train in mph
def time_travelled : ℝ := 8 -- Time in hours

theorem distance_between_trains : 
  (speed_train2 * time_travelled) - (speed_train1 * time_travelled) = 160 := by
  sorry

end NUMINAMATH_GPT_distance_between_trains_l1754_175481


namespace NUMINAMATH_GPT_total_amount_distributed_l1754_175405

theorem total_amount_distributed (A : ℝ) :
  (∀ A, (A / 14 = A / 18 + 80) → A = 5040) :=
by
  sorry

end NUMINAMATH_GPT_total_amount_distributed_l1754_175405


namespace NUMINAMATH_GPT_unit_squares_in_50th_ring_l1754_175429

-- Definitions from the conditions
def unit_squares_in_first_ring : ℕ := 12

def unit_squares_in_nth_ring (n : ℕ) : ℕ :=
  32 * n - 16

-- Prove the specific instance for the 50th ring
theorem unit_squares_in_50th_ring : unit_squares_in_nth_ring 50 = 1584 :=
by
  sorry

end NUMINAMATH_GPT_unit_squares_in_50th_ring_l1754_175429


namespace NUMINAMATH_GPT_average_incorrect_answers_is_correct_l1754_175413

-- Definitions
def total_items : ℕ := 60
def liza_correct_answers : ℕ := (90 * total_items) / 100
def rose_correct_answers : ℕ := liza_correct_answers + 2
def max_correct_answers : ℕ := liza_correct_answers - 5

def liza_incorrect_answers : ℕ := total_items - liza_correct_answers
def rose_incorrect_answers : ℕ := total_items - rose_correct_answers
def max_incorrect_answers : ℕ := total_items - max_correct_answers

def average_incorrect_answers : ℚ :=
  (liza_incorrect_answers + rose_incorrect_answers + max_incorrect_answers) / 3

-- Theorem statement
theorem average_incorrect_answers_is_correct : average_incorrect_answers = 7 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_average_incorrect_answers_is_correct_l1754_175413


namespace NUMINAMATH_GPT_find_missing_number_l1754_175452

theorem find_missing_number (x : ℝ) (h : 1 / ((1 / 0.03) + (1 / x)) = 0.02775) : abs (x - 0.370) < 0.001 := by
  sorry

end NUMINAMATH_GPT_find_missing_number_l1754_175452
