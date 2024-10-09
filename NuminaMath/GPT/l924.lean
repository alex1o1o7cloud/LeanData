import Mathlib

namespace quadratic_eq_has_two_distinct_real_roots_l924_92493

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Problem statement: Prove that the quadratic equation x^2 + 3x - 2 = 0 has two distinct real roots
theorem quadratic_eq_has_two_distinct_real_roots :
  discriminant 1 3 (-2) > 0 :=
by
  -- Proof goes here
  sorry

end quadratic_eq_has_two_distinct_real_roots_l924_92493


namespace double_inequality_solution_l924_92412

open Set

theorem double_inequality_solution (x : ℝ) :
  -1 < (x^2 - 16 * x + 24) / (x^2 - 4 * x + 8) ∧
  (x^2 - 16 * x + 24) / (x^2 - 4 * x + 8) < 1 ↔
  x ∈ Ioo (3 / 2) 4 ∪ Ioi 8 :=
by
  sorry

end double_inequality_solution_l924_92412


namespace simplify_expression_l924_92468

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 1) (h₂ : a ≠ 1 / 2) :
    1 - 1 / (1 - a / (1 - a)) = -a / (1 - 2 * a) := by
  sorry

end simplify_expression_l924_92468


namespace option_C_is_correct_l924_92466

theorem option_C_is_correct :
  (-3 - (-2) ≠ -5) ∧
  (-|(-1:ℝ)/3| + 1 ≠ 4/3) ∧
  (4 - 4 / 2 = 2) ∧
  (3^2 / 6 * (1/6) ≠ 9) :=
by
  -- Proof omitted
  sorry

end option_C_is_correct_l924_92466


namespace range_of_a_l924_92443

noncomputable def f (a x : ℝ) : ℝ := (Real.log (x^2 - a * x + 5)) / (Real.log a)

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (ha0 : 0 < a) (ha1 : a ≠ 1) 
  (hx₁x₂ : x₁ < x₂) (hx₂ : x₂ ≤ a / 2) 
  (hf : (f a x₂ - f a x₁ < 0)) : 
  1 < a ∧ a < 2 * Real.sqrt 5 := 
sorry

end range_of_a_l924_92443


namespace coin_probability_l924_92411

theorem coin_probability :
  let PA := 3/4
  let PB := 1/2
  let PC := 1/4
  (PA * PB * (1 - PC)) = 9/32 :=
by
  sorry

end coin_probability_l924_92411


namespace tan_difference_of_angle_l924_92421

noncomputable def point_on_terminal_side (θ : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (2, 3) = (k * Real.cos θ, k * Real.sin θ)

theorem tan_difference_of_angle (θ : ℝ) (hθ : point_on_terminal_side θ) :
  Real.tan (θ - Real.pi / 4) = 1 / 5 :=
sorry

end tan_difference_of_angle_l924_92421


namespace points_difference_l924_92482

theorem points_difference :
  let points_td := 7
  let points_epc := 1
  let points_fg := 3
  
  let touchdowns_BG := 6
  let epc_BG := 4
  let fg_BG := 2
  
  let touchdowns_CF := 8
  let epc_CF := 6
  let fg_CF := 3
  
  let total_BG := touchdowns_BG * points_td + epc_BG * points_epc + fg_BG * points_fg
  let total_CF := touchdowns_CF * points_td + epc_CF * points_epc + fg_CF * points_fg
  
  total_CF - total_BG = 19 := by
  sorry

end points_difference_l924_92482


namespace max_frac_sum_l924_92460

theorem max_frac_sum {n : ℕ} (h_n : n > 1) :
  ∀ (a b c d : ℕ), (a + c ≤ n) ∧ (b > 0) ∧ (d > 0) ∧
  (a * d + b * c < b * d) → 
  ↑a / ↑b + ↑c / ↑d ≤ (1 - 1 / ( ⌊(2*n : ℝ)/3 + 1/6⌋₊ + 1) * ( ⌊(2*n : ℝ)/3 + 1/6⌋₊ * (n - ⌊(2*n : ℝ)/3 + 1/6⌋₊) + 1)) :=
by sorry

end max_frac_sum_l924_92460


namespace quadratic_function_properties_l924_92479

noncomputable def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 :=
by
  sorry

end quadratic_function_properties_l924_92479


namespace ratio_M_N_l924_92469

variables {M Q P N R : ℝ}

-- Conditions
def condition1 : M = 0.40 * Q := sorry
def condition2 : Q = 0.25 * P := sorry
def condition3 : N = 0.75 * R := sorry
def condition4 : R = 0.60 * P := sorry

-- Theorem to prove
theorem ratio_M_N : M / N = 2 / 9 := sorry

end ratio_M_N_l924_92469


namespace proof_u_g_3_l924_92430

noncomputable def u (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)

noncomputable def g (x : ℝ) : ℝ := 7 - u x

theorem proof_u_g_3 :
  u (g 3) = Real.sqrt (37 - 5 * Real.sqrt 17) :=
sorry

end proof_u_g_3_l924_92430


namespace marley_fruits_l924_92495

theorem marley_fruits 
    (louis_oranges : ℕ := 5) (louis_apples : ℕ := 3)
    (samantha_oranges : ℕ := 8) (samantha_apples : ℕ := 7)
    (marley_oranges : ℕ := 2 * louis_oranges)
    (marley_apples : ℕ := 3 * samantha_apples) :
    marley_oranges + marley_apples = 31 := by
  sorry

end marley_fruits_l924_92495


namespace value_of_N_l924_92424

theorem value_of_N (N : ℕ) (x y z w s : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
    (h_pos_z : 0 < z) (h_pos_w : 0 < w) (h_pos_s : 0 < s) (h_sum : x + y + z + w + s = N)
    (h_comb : Nat.choose N 4 = 3003) : N = 18 := 
by
  sorry

end value_of_N_l924_92424


namespace village_population_500_l924_92457

variable (n : ℝ) -- Define the variable for population increase
variable (initial_population : ℝ) -- Define the variable for the initial population

-- Conditions from the problem
def first_year_increase : Prop := initial_population * (3 : ℝ) = n
def initial_population_def : Prop := initial_population = n / 3
def second_year_increase_def := ((n / 3 + n) * (n / 100 )) = 300

-- Define the final population formula
def population_after_two_years : ℝ := (initial_population + n + 300)

theorem village_population_500 (n : ℝ) (initial_population: ℝ) :
  first_year_increase n initial_population →
  initial_population_def n initial_population →
  second_year_increase_def n →
  population_after_two_years n initial_population = 500 :=
by sorry

#check village_population_500

end village_population_500_l924_92457


namespace ratio_of_third_to_second_is_four_l924_92413

theorem ratio_of_third_to_second_is_four
  (x y z k : ℕ)
  (h1 : y = 2 * x)
  (h2 : z = k * y)
  (h3 : (x + y + z) / 3 = 165)
  (h4 : y = 90) :
  z / y = 4 :=
by
  sorry

end ratio_of_third_to_second_is_four_l924_92413


namespace ones_digit_exponent_73_l924_92416

theorem ones_digit_exponent_73 (n : ℕ) : 
  (73 ^ n) % 10 = 7 ↔ n % 4 = 3 := 
sorry

end ones_digit_exponent_73_l924_92416


namespace probability_of_event_A_l924_92432

/-- The events A and B are independent, and it is given that:
  1. P(A) > 0
  2. P(A) = 2 * P(B)
  3. P(A or B) = 8 * P(A and B)

We need to prove that P(A) = 1/3. 
-/
theorem probability_of_event_A (P_A P_B : ℝ) (hP_indep : P_A * P_B = P_A) 
  (hP_A_pos : P_A > 0) (hP_A_eq_2P_B : P_A = 2 * P_B) 
  (hP_or_eq_8P_and : P_A + P_B - P_A * P_B = 8 * P_A * P_B) : 
  P_A = 1 / 3 := 
by
  sorry

end probability_of_event_A_l924_92432


namespace positive_difference_eq_six_l924_92408

theorem positive_difference_eq_six (x y : ℝ) (h1 : x + y = 8) (h2 : x ^ 2 - y ^ 2 = 48) : |x - y| = 6 := by
  sorry

end positive_difference_eq_six_l924_92408


namespace min_value_of_quadratic_l924_92488

theorem min_value_of_quadratic (x y z : ℝ) 
  (h1 : x + 2 * y - 5 * z = 3)
  (h2 : x - 2 * y - z = -5) : 
  ∃ z' : ℝ,  x = 3 * z' - 1 ∧ y = z' + 2 ∧ (11 * z' * z' - 2 * z' + 5 = (54 : ℝ) / 11) :=
sorry

end min_value_of_quadratic_l924_92488


namespace Ryan_has_28_marbles_l924_92427

theorem Ryan_has_28_marbles :
  ∃ R : ℕ, (12 + R) - (1/4 * (12 + R)) * 2 = 20 ∧ R = 28 :=
by
  sorry

end Ryan_has_28_marbles_l924_92427


namespace ratio_of_numbers_l924_92499

theorem ratio_of_numbers (x y : ℕ) (h1 : x + y = 124) (h2 : y = 3 * x) : x / Nat.gcd x y = 1 ∧ y / Nat.gcd x y = 3 := by
  sorry

end ratio_of_numbers_l924_92499


namespace proof_6_times_15_times_5_eq_2_l924_92455

noncomputable def given_condition (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

theorem proof_6_times_15_times_5_eq_2 : 
  given_condition 6 15 5 → 6 * 15 * 5 = 2 :=
by
  sorry

end proof_6_times_15_times_5_eq_2_l924_92455


namespace abs_discriminant_inequality_l924_92436

theorem abs_discriminant_inequality 
  (a b c A B C : ℝ) 
  (ha : a ≠ 0) 
  (hA : A ≠ 0) 
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) : 
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end abs_discriminant_inequality_l924_92436


namespace intersection_is_correct_l924_92407

def M : Set ℤ := {-2, 1, 2}
def N : Set ℤ := {1, 2, 4}

theorem intersection_is_correct : M ∩ N = {1, 2} := 
by {
  sorry
}

end intersection_is_correct_l924_92407


namespace min_value_of_seq_l924_92410

theorem min_value_of_seq 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (m a₁ : ℝ)
  (h1 : ∀ n, a n + a (n + 1) = n * (-1) ^ ((n * (n + 1)) / 2))
  (h2 : m + S 2015 = -1007)
  (h3 : a₁ * m > 0) :
  ∃ x, x = (1 / a₁) + (4 / m) ∧ x = 9 :=
by
  sorry

end min_value_of_seq_l924_92410


namespace perpendicular_vectors_m_value_l924_92405

theorem perpendicular_vectors_m_value
  (a : ℝ × ℝ := (1, 2))
  (b : ℝ × ℝ)
  (h_perpendicular : (a.1 * b.1 + a.2 * b.2) = 0) :
  b = (-2, 1) :=
by
  sorry

end perpendicular_vectors_m_value_l924_92405


namespace units_digit_of_33_pow_33_mul_7_pow_7_l924_92433

theorem units_digit_of_33_pow_33_mul_7_pow_7 : (33 ^ (33 * (7 ^ 7))) % 10 = 7 := 
  sorry

end units_digit_of_33_pow_33_mul_7_pow_7_l924_92433


namespace benny_number_of_days_worked_l924_92456

-- Define the conditions
def total_hours_worked : ℕ := 18
def hours_per_day : ℕ := 3

-- Define the problem statement in Lean
theorem benny_number_of_days_worked : (total_hours_worked / hours_per_day) = 6 := 
by
  sorry

end benny_number_of_days_worked_l924_92456


namespace problem_part1_problem_part2_l924_92475

theorem problem_part1 (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : x^2 - 2 * x * y + 4 * y^2 = 11) :
  x * y = 1 :=
sorry

theorem problem_part2 (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : x^2 - 2 * x * y + 4 * y^2 = 11) :
  x^2 * y - 2 * x * y^2 = 3 :=
sorry

end problem_part1_problem_part2_l924_92475


namespace candle_duration_1_hour_per_night_l924_92496

-- Definitions based on the conditions
def burn_rate_2_hours (candles: ℕ) (nights: ℕ) : ℕ := nights / candles -- How long each candle lasts when burned for 2 hours per night

-- Given conditions provided
def nights_24 : ℕ := 24
def candles_6 : ℕ := 6

-- The duration a candle lasts when burned for 2 hours every night
def candle_duration_2_hours_per_night : ℕ := burn_rate_2_hours candles_6 nights_24 -- => 4 (not evaluated here)

-- Theorem to prove the duration a candle lasts when burned for 1 hour every night
theorem candle_duration_1_hour_per_night : candle_duration_2_hours_per_night * 2 = 8 :=
by
  sorry -- The proof is omitted, only the statement is required

-- Note: candle_duration_2_hours_per_night = 4 by the given conditions 
-- This leads to 4 * 2 = 8, which matches the required number of nights the candle lasts when burned for 1 hour per night.

end candle_duration_1_hour_per_night_l924_92496


namespace shortest_routes_l924_92484

def side_length : ℕ := 10
def refuel_distance : ℕ := 30
def num_squares_per_refuel := refuel_distance / side_length

theorem shortest_routes (A B : Type) (distance_AB : ℕ) (shortest_paths : Π (A B : Type), ℕ) : 
  shortest_paths A B = 54 := by
  sorry

end shortest_routes_l924_92484


namespace girls_try_out_l924_92481

-- Given conditions
variables (boys callBacks didNotMakeCut : ℕ)
variable (G : ℕ)

-- Define the conditions
def conditions : Prop := 
  boys = 14 ∧ 
  callBacks = 2 ∧ 
  didNotMakeCut = 21 ∧ 
  G + boys = callBacks + didNotMakeCut

-- The statement of the proof
theorem girls_try_out (h : conditions boys callBacks didNotMakeCut G) : G = 9 :=
by
  sorry

end girls_try_out_l924_92481


namespace remaining_pages_l924_92458

theorem remaining_pages (total_pages : ℕ) (science_project_percentage : ℕ) (math_homework_pages : ℕ)
  (h1 : total_pages = 120)
  (h2 : science_project_percentage = 25) 
  (h3 : math_homework_pages = 10) : 
  total_pages - (total_pages * science_project_percentage / 100) - math_homework_pages = 80 := by
  sorry

end remaining_pages_l924_92458


namespace intersection_is_correct_l924_92409

noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 2}

noncomputable def B : Set ℝ := {x | x^2 - 5 * x - 6 < 0}

theorem intersection_is_correct : A ∩ B = {x | -1 < x ∧ x < 2} := 
by { sorry }

end intersection_is_correct_l924_92409


namespace peter_pizza_fraction_l924_92428

def pizza_slices : ℕ := 16
def peter_slices_alone : ℕ := 2
def shared_slice : ℚ := 1 / 2

theorem peter_pizza_fraction :
  let fraction_alone := peter_slices_alone * (1 / pizza_slices)
  let fraction_shared := shared_slice * (1 / pizza_slices)
  let total_fraction := fraction_alone + fraction_shared
  total_fraction = 5 / 32 :=
by
  let fraction_alone := peter_slices_alone * (1 / pizza_slices)
  let fraction_shared := shared_slice * (1 / pizza_slices)
  let total_fraction := fraction_alone + fraction_shared
  sorry

end peter_pizza_fraction_l924_92428


namespace graph_two_intersecting_lines_l924_92467

theorem graph_two_intersecting_lines (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 3 * x * y ↔ x = 0 ∨ y = 0 :=
by
  -- Placeholder for the proof
  sorry

end graph_two_intersecting_lines_l924_92467


namespace rectangular_field_area_l924_92494

noncomputable def a : ℝ := 14
noncomputable def c : ℝ := 17
noncomputable def b := Real.sqrt (c^2 - a^2)
noncomputable def area := a * b

theorem rectangular_field_area : area = 14 * Real.sqrt 93 := by
  sorry

end rectangular_field_area_l924_92494


namespace cat_food_sufficiency_l924_92420

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l924_92420


namespace problem_statement_l924_92415

theorem problem_statement (x : ℤ) (h : 3 - x = -2) : x + 1 = 6 := 
by {
  -- Proof would be provided here
  sorry
}

end problem_statement_l924_92415


namespace reaction_produces_nh3_l924_92473

-- Define the Chemical Equation as a structure
structure Reaction where
  reagent1 : ℕ -- moles of NH4NO3
  reagent2 : ℕ -- moles of NaOH
  product  : ℕ -- moles of NH3

-- Given conditions
def reaction := Reaction.mk 2 2 2

-- Theorem stating that given 2 moles of NH4NO3 and 2 moles of NaOH,
-- the number of moles of NH3 formed is 2 moles.
theorem reaction_produces_nh3 (r : Reaction) (h1 : r.reagent1 = 2)
  (h2 : r.reagent2 = 2) : r.product = 2 := by
  sorry

end reaction_produces_nh3_l924_92473


namespace cost_price_approx_l924_92474

noncomputable def cost_price (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  selling_price / (1 + profit_percent / 100)

theorem cost_price_approx :
  ∀ (selling_price profit_percent : ℝ),
  selling_price = 2552.36 →
  profit_percent = 6 →
  abs (cost_price selling_price profit_percent - 2407.70) < 0.01 :=
by
  intros selling_price profit_percent h1 h2
  sorry

end cost_price_approx_l924_92474


namespace second_polygon_sides_l924_92462

theorem second_polygon_sides (s : ℝ) (n : ℝ) (h1 : 50 * 3 * s = n * s) : n = 150 := 
by
  sorry

end second_polygon_sides_l924_92462


namespace emma_prob_at_least_one_correct_l924_92403

-- Define the probability of getting a question wrong
def prob_wrong : ℚ := 4 / 5

-- Define the probability of getting all five questions wrong
def prob_all_wrong : ℚ := prob_wrong ^ 5

-- Define the probability of getting at least one question correct
def prob_at_least_one_correct : ℚ := 1 - prob_all_wrong

-- Define the main theorem to be proved
theorem emma_prob_at_least_one_correct : prob_at_least_one_correct = 2101 / 3125 := by
  sorry  -- This is where the proof would go

end emma_prob_at_least_one_correct_l924_92403


namespace tank_base_length_width_difference_l924_92487

variable (w l h : ℝ)

theorem tank_base_length_width_difference :
  (l = 5 * w) →
  (h = (1/2) * w) →
  (l * w * h = 3600) →
  (|l - w - 45.24| < 0.01) := 
by
  sorry

end tank_base_length_width_difference_l924_92487


namespace ticket_price_divisors_count_l924_92477

theorem ticket_price_divisors_count :
  ∃ (x : ℕ), (36 % x = 0) ∧ (60 % x = 0) ∧ (Nat.divisors (Nat.gcd 36 60)).card = 6 := 
by
  sorry

end ticket_price_divisors_count_l924_92477


namespace cos_4theta_l924_92450

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (4 * θ) = 17 / 32 :=
sorry

end cos_4theta_l924_92450


namespace ian_investment_percentage_change_l924_92422

theorem ian_investment_percentage_change :
  let initial_investment := 200
  let first_year_loss := 0.10
  let second_year_gain := 0.25
  let amount_after_loss := initial_investment * (1 - first_year_loss)
  let amount_after_gain := amount_after_loss * (1 + second_year_gain)
  let percentage_change := (amount_after_gain - initial_investment) / initial_investment * 100
  percentage_change = 12.5 := 
by
  sorry

end ian_investment_percentage_change_l924_92422


namespace f_at_one_f_extremes_l924_92437

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x > 0 → f x = f x
axiom f_multiplicative : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

theorem f_at_one : f 1 = 0 := sorry

theorem f_extremes (hf_sub_one_fifth : f (1 / 5) = -1) :
  ∃ c d : ℝ, (∀ x : ℝ, 1 / 25 ≤ x ∧ x ≤ 125 → c ≤ f x ∧ f x ≤ d) ∧
  c = -2 ∧ d = 3 := sorry

end f_at_one_f_extremes_l924_92437


namespace infinite_geometric_series_second_term_l924_92485

theorem infinite_geometric_series_second_term (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 16) (h3 : S = a / (1 - r)) : a * r = 3 := 
sorry

end infinite_geometric_series_second_term_l924_92485


namespace intersecting_x_value_l924_92449

theorem intersecting_x_value : 
  (∃ x y : ℝ, y = 3 * x - 17 ∧ 3 * x + y = 103) → 
  (∃ x : ℝ, x = 20) :=
by
  sorry

end intersecting_x_value_l924_92449


namespace jane_trail_mix_chocolate_chips_l924_92402

theorem jane_trail_mix_chocolate_chips (c₁ : ℝ) (c₂ : ℝ) (c₃ : ℝ) (c₄ : ℝ) (c₅ : ℝ) :
  (c₁ = 0.30) → (c₂ = 0.70) → (c₃ = 0.45) → (c₄ = 0.35) → (c₅ = 0.60) →
  c₄ = 0.35 ∧ (c₅ - c₁) * 2 = 0.40 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end jane_trail_mix_chocolate_chips_l924_92402


namespace hyperbola_asymptote_distance_l924_92401

section
open Function Real

variables (O P : ℝ × ℝ) (C : ℝ × ℝ → Prop) (M : ℝ × ℝ)
          (dist_asymptote : ℝ)

-- Conditions
def is_origin (O : ℝ × ℝ) : Prop := O = (0, 0)
def on_hyperbola (P : ℝ × ℝ) : Prop := P.1 ^ 2 / 9 - P.2 ^ 2 / 16 = 1
def unit_circle (M : ℝ × ℝ) : Prop := sqrt (M.1 ^ 2 + M.2 ^ 2) = 1
def orthogonal (O M P : ℝ × ℝ) : Prop := O.1 * P.1 + O.2 * P.2 = 0
def min_PM (dist : ℝ) : Prop := dist = 1 -- The minimum distance when |PM| is minimized

-- Proof problem
theorem hyperbola_asymptote_distance :
  is_origin O → 
  on_hyperbola P → 
  unit_circle M → 
  orthogonal O M P → 
  min_PM (sqrt ((P.1 - M.1) ^ 2 + (P.2 - M.2) ^ 2)) → 
  dist_asymptote = 12 / 5 :=
sorry
end

end hyperbola_asymptote_distance_l924_92401


namespace number_of_boys_l924_92452

theorem number_of_boys (x : ℕ) (boys girls : ℕ)
  (initialRatio : girls / boys = 5 / 6)
  (afterLeavingRatio : (girls - 20) / boys = 2 / 3) :
  boys = 120 := by
  -- Proof is omitted
  sorry

end number_of_boys_l924_92452


namespace sum_of_consecutive_page_numbers_l924_92441

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end sum_of_consecutive_page_numbers_l924_92441


namespace probability_of_selecting_cooking_l924_92459

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l924_92459


namespace triangle_inequality_proof_l924_92483

noncomputable def triangle_inequality (A B C a b c : ℝ) (hABC : A + B + C = Real.pi) : Prop :=
  Real.pi / 3 ≤ (a * A + b * B + c * C) / (a + b + c) ∧ (a * A + b * B + c * C) / (a + b + c) < Real.pi / 2

theorem triangle_inequality_proof (A B C a b c : ℝ) (hABC : A + B + C = Real.pi) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h₁: A + B + C = Real.pi) (h₂: ∀ {x y : ℝ}, A ≥ B  → a ≥ b → A * b + B * a ≤ A * a + B * b) 
  (h₃: ∀ {x y : ℝ}, x + y > 0 → A * x + B * y + C * (a + b - x - y) > 0) : 
  triangle_inequality A B C a b c hABC :=
by
  sorry

end triangle_inequality_proof_l924_92483


namespace certain_event_idiom_l924_92471

theorem certain_event_idiom : 
  ∃ (idiom : String), idiom = "Catching a turtle in a jar" ∧ 
  ∀ (option : String), 
    option = "Catching a turtle in a jar" ∨ 
    option = "Carving a boat to find a sword" ∨ 
    option = "Waiting by a tree stump for a rabbit" ∨ 
    option = "Fishing for the moon in the water" → 
    (option = idiom ↔ (option = "Catching a turtle in a jar")) := 
by
  sorry

end certain_event_idiom_l924_92471


namespace find_A_l924_92419

theorem find_A (A : ℝ) (h : 4 * A + 5 = 33) : A = 7 :=
  sorry

end find_A_l924_92419


namespace range_of_a_l924_92400

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, ∃ y ∈ Set.Ici a, y = (x^2 + 2*x + a) / (x + 1)) ↔ a ≤ 2 :=
by
  sorry

end range_of_a_l924_92400


namespace percentage_increase_l924_92489

theorem percentage_increase :
  let original_employees := 852
  let new_employees := 1065
  let increase := new_employees - original_employees
  let percentage := (increase.toFloat / original_employees.toFloat) * 100
  percentage = 25 := 
by 
  sorry

end percentage_increase_l924_92489


namespace transformation_matrix_correct_l924_92435
noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, 3],
  ![-3, 0]
]

theorem transformation_matrix_correct :
  let R : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![0, 1],
    ![-1, 0]
  ];
  let S : ℝ := 3;
  M = S • R :=
by
  sorry

end transformation_matrix_correct_l924_92435


namespace find_center_radius_l924_92486

noncomputable def circle_center_radius (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y - 6 = 0 → 
  ∃ (h k r : ℝ), (x + 1) * (x + 1) + (y - 2) * (y - 2) = r ∧ h = -1 ∧ k = 2 ∧ r = 11

theorem find_center_radius :
  circle_center_radius x y :=
sorry

end find_center_radius_l924_92486


namespace solution_set_of_inequality_l924_92426

theorem solution_set_of_inequality (x : ℝ) : 
  (3 * x - 4 > 2) → (x > 2) :=
by
  intro h
  sorry

end solution_set_of_inequality_l924_92426


namespace zhou_yu_age_at_death_l924_92434

theorem zhou_yu_age_at_death (x : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 9)
    (h₂ : ∃ age : ℕ, age = 10 * (x - 3) + x)
    (h₃ : x^2 = 10 * (x - 3) + x) :
    x^2 = 10 * (x - 3) + x :=
by
  sorry

end zhou_yu_age_at_death_l924_92434


namespace solve_modular_equation_l924_92445

theorem solve_modular_equation (x : ℤ) :
  (15 * x + 2) % 18 = 7 % 18 ↔ x % 6 = 1 % 6 := by
  sorry

end solve_modular_equation_l924_92445


namespace quarters_needed_to_buy_items_l924_92451

-- Define the costs of each item in cents
def cost_candy_bar : ℕ := 25
def cost_chocolate : ℕ := 75
def cost_juice : ℕ := 50

-- Define the quantities of each item
def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Define the value of a quarter in cents
def value_of_quarter : ℕ := 25

-- Define the total cost of the items
def total_cost : ℕ := (num_candy_bars * cost_candy_bar) + (num_chocolates * cost_chocolate) + (num_juice_packs * cost_juice)

-- Calculate the number of quarters needed
def num_quarters_needed : ℕ := total_cost / value_of_quarter

-- The theorem to prove that the number of quarters needed is 11
theorem quarters_needed_to_buy_items : num_quarters_needed = 11 := by
  -- Proof omitted
  sorry

end quarters_needed_to_buy_items_l924_92451


namespace num_ordered_pairs_eq_seven_l924_92406

theorem num_ordered_pairs_eq_seven : ∃ n, n = 7 ∧ ∀ (x y : ℕ), (x * y = 64) → (x > 0 ∧ y > 0) → n = 7 :=
by
  sorry

end num_ordered_pairs_eq_seven_l924_92406


namespace line_equation_l924_92476

open Real

theorem line_equation (x y : Real) : 
  (3 * x + 2 * y - 1 = 0) ↔ (y = (-(3 / 2)) * x + 2.5) :=
by
  sorry

end line_equation_l924_92476


namespace area_three_layers_l924_92472

def total_area_rugs : ℝ := 200
def floor_covered_area : ℝ := 140
def exactly_two_layers_area : ℝ := 24

theorem area_three_layers : (2 * (200 - 140 - 24) / 2 = 2 * 18) := 
by admit -- since we're instructed to skip the proof

end area_three_layers_l924_92472


namespace find_b_l924_92414

-- Define the conditions as hypotheses
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + b*x - 3

theorem find_b (x₁ x₂ b : ℝ) (h₁ : x₁ ≠ x₂)
  (h₂ : 3 * x₁^2 + 4 * x₁ + b = 0)
  (h₃ : 3 * x₂^2 + 4 * x₂ + b = 0)
  (h₄ : x₁^2 + x₂^2 = 34 / 9) :
  b = -3 :=
by
  -- Proof will be inserted here
  sorry

end find_b_l924_92414


namespace range_of_a_l924_92490

-- Define the function f(x) = x^2 - 3x
def f (x : ℝ) : ℝ := x^2 - 3 * x

-- Define the interval as a closed interval from -1 to 1
def interval : Set ℝ := Set.Icc (-1) (1)

-- State the main proposition
theorem range_of_a (a : ℝ) :
  (∃ x ∈ interval, -x^2 + 3 * x + a > 0) ↔ a > -2 :=
by
  sorry

end range_of_a_l924_92490


namespace pyramid_volume_is_232_l924_92478

noncomputable def pyramid_volume (length : ℝ) (width : ℝ) (slant_height : ℝ) : ℝ :=
  (1 / 3) * (length * width) * (Real.sqrt ((slant_height)^2 - ((length / 2)^2 + (width / 2)^2)))

theorem pyramid_volume_is_232 :
  pyramid_volume 5 10 15 = 232 := 
by
  sorry

end pyramid_volume_is_232_l924_92478


namespace slower_speed_l924_92446

theorem slower_speed (x : ℝ) (h_walk_faster : 12 * (100 / x) - 100 = 20) : x = 10 :=
by sorry

end slower_speed_l924_92446


namespace sin_cos_value_l924_92491

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : (Real.sin x) * (Real.cos x) = 4 / 17 := by
  sorry

end sin_cos_value_l924_92491


namespace dot_product_is_constant_l924_92444

-- Define the trajectory C as the parabola given by the equation y^2 = 4x
def trajectory (x y : ℝ) : Prop := y^2 = 4 * x

-- Prove the range of k for the line passing through point (-1, 0) and intersecting trajectory C
def valid_slope (k : ℝ) : Prop := (-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1)

-- Prove that ∀ D ≠ A, B on the parabola y^2 = 4x, and lines DA and DB intersect vertical line through (1, 0) on points P, Q, OP ⋅ OQ = 5
theorem dot_product_is_constant (D A B P Q : ℝ × ℝ) 
  (hD : trajectory D.1 D.2)
  (hA : trajectory A.1 A.2)
  (hB : trajectory B.1 B.2)
  (hDiff : D ≠ A ∧ D ≠ B)
  (hP : P = (1, (D.2 * A.2 + 4) / (D.2 + A.2))) 
  (hQ : Q = (1, (D.2 * B.2 + 4) / (D.2 + B.2))) :
  (1 + (D.2 * A.2 + 4) / (D.2 + A.2)) * (1 + (D.2 * B.2 + 4) / (D.2 + B.2)) = 5 :=
sorry

end dot_product_is_constant_l924_92444


namespace arithmetic_sequence_sum_l924_92423

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (a4_eq_3 : a 4 = 3) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
by
  sorry

end arithmetic_sequence_sum_l924_92423


namespace smallest_sum_of_five_consecutive_primes_divisible_by_three_l924_92454

-- Definition of the conditions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (a b c d e : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧
  (b = a + 1 ∨ b = a + 2) ∧ (c = b + 1 ∨ c = b + 2) ∧
  (d = c + 1 ∨ d = c + 2) ∧ (e = d + 1 ∨ e = d + 2)

theorem smallest_sum_of_five_consecutive_primes_divisible_by_three :
  ∃ a b c d e, consecutive_primes a b c d e ∧ a + b + c + d + e = 39 ∧ 39 % 3 = 0 :=
sorry

end smallest_sum_of_five_consecutive_primes_divisible_by_three_l924_92454


namespace jeans_cost_before_sales_tax_l924_92442

-- Defining conditions
def original_cost : ℝ := 49
def summer_discount : ℝ := 0.50
def wednesday_discount : ℝ := 10

-- The mathematical equivalent proof problem
theorem jeans_cost_before_sales_tax :
  let discount_price := original_cost * (1 - summer_discount)
  let wednesday_price := discount_price - wednesday_discount
  wednesday_price = 14.50 :=
by
  let discount_price := original_cost * (1 - summer_discount)
  let wednesday_price := discount_price - wednesday_discount
  sorry

end jeans_cost_before_sales_tax_l924_92442


namespace range_of_a_l924_92439

def set_A (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < 2 * a + 1}
def set_B : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) : (set_A a ∩ set_B = ∅) ↔ (a ≤ -2 ∨ (a > -2 ∧ a ≤ -1/2) ∨ a ≥ 2) := by
  sorry

end range_of_a_l924_92439


namespace total_sampled_students_l924_92464

-- Define the total number of students in each grade
def students_in_grade12 : ℕ := 700
def students_in_grade11 : ℕ := 700
def students_in_grade10 : ℕ := 800

-- Define the number of students sampled from grade 10
def sampled_from_grade10 : ℕ := 80

-- Define the total number of students in the school
def total_students : ℕ := students_in_grade12 + students_in_grade11 + students_in_grade10

-- Prove that the total number of students sampled (x) is equal to 220
theorem total_sampled_students : 
  (sampled_from_grade10 : ℚ) / (students_in_grade10 : ℚ) * (total_students : ℚ) = 220 := 
by
  sorry

end total_sampled_students_l924_92464


namespace complement_intersection_empty_l924_92418

open Set

-- Given definitions and conditions
def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

-- Complement operation with respect to U
def C_U (X : Set ℕ) : Set ℕ := U \ X

-- The proof statement to be shown
theorem complement_intersection_empty :
  (C_U A ∩ C_U B) = ∅ := by sorry

end complement_intersection_empty_l924_92418


namespace coat_price_reduction_l924_92447

variable (original_price reduction : ℝ)

theorem coat_price_reduction
  (h_orig : original_price = 500)
  (h_reduct : reduction = 350)
  : reduction / original_price * 100 = 70 := 
sorry

end coat_price_reduction_l924_92447


namespace smallest_multiple_of_7_not_particular_l924_92438

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (λ d acc => acc + d) 0

def is_particular_integer (n : ℕ) : Prop :=
  n % (sum_of_digits n) ^ 2 = 0

theorem smallest_multiple_of_7_not_particular :
  ∃ n, n > 0 ∧ n % 7 = 0 ∧ ¬ is_particular_integer n ∧ ∀ m, m > 0 ∧ m % 7 = 0 ∧ ¬ is_particular_integer m → n ≤ m :=
  by
    use 7
    sorry

end smallest_multiple_of_7_not_particular_l924_92438


namespace sum_of_values_l924_92498

def f (x : Int) : Int := Int.natAbs x - 3
def g (x : Int) : Int := -x

def fogof (x : Int) : Int := f (g (f x))

theorem sum_of_values :
  (fogof (-5)) + (fogof (-4)) + (fogof (-3)) + (fogof (-2)) + (fogof (-1)) + (fogof 0) + (fogof 1) + (fogof 2) + (fogof 3) + (fogof 4) + (fogof 5) = -17 :=
by
  sorry

end sum_of_values_l924_92498


namespace measure_of_angle_A_l924_92417

noncomputable def angle_A (angle_B : ℝ) := 3 * angle_B - 40

theorem measure_of_angle_A (x : ℝ) (angle_A_parallel_B : true) (h : ∃ k : ℝ, (k = x ∧ (angle_A x = x ∨ angle_A x + x = 180))) :
  angle_A x = 20 ∨ angle_A x = 125 :=
by
  sorry

end measure_of_angle_A_l924_92417


namespace diamond_calculation_l924_92470

def diamond (a b : ℚ) : ℚ := (a - b) / (1 + a * b)

theorem diamond_calculation : diamond 1 (diamond 2 (diamond 3 (diamond 4 5))) = 87 / 59 :=
by
  sorry

end diamond_calculation_l924_92470


namespace total_marks_l924_92461

variable (marks_in_music marks_in_maths marks_in_arts marks_in_social_studies : ℕ)

def marks_conditions : Prop :=
  marks_in_maths = marks_in_music - (1/10) * marks_in_music ∧
  marks_in_maths = marks_in_arts - 20 ∧
  marks_in_social_studies = marks_in_music + 10 ∧
  marks_in_music = 70

theorem total_marks 
  (h : marks_conditions marks_in_music marks_in_maths marks_in_arts marks_in_social_studies) :
  marks_in_music + marks_in_maths + marks_in_arts + marks_in_social_studies = 296 :=
by
  sorry

end total_marks_l924_92461


namespace power_sums_fifth_l924_92453

noncomputable def compute_power_sums (α β γ : ℂ) : ℂ :=
  α^5 + β^5 + γ^5

theorem power_sums_fifth (α β γ : ℂ)
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 10) :
  compute_power_sums α β γ = 47.2 :=
sorry

end power_sums_fifth_l924_92453


namespace remainder_when_divided_l924_92425

theorem remainder_when_divided (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_when_divided_l924_92425


namespace lines_intersect_at_point_l924_92431

noncomputable def line1 (s : ℚ) : ℚ × ℚ :=
  (1 + 2 * s, 4 - 3 * s)

noncomputable def line2 (v : ℚ) : ℚ × ℚ :=
  (3 + 3 * v, 2 - v)

theorem lines_intersect_at_point :
  ∃ s v : ℚ,
    line1 s = (15 / 7, 16 / 7) ∧
    line2 v = (15 / 7, 16 / 7) ∧
    s = 4 / 7 ∧
    v = -2 / 7 := by
  sorry

end lines_intersect_at_point_l924_92431


namespace percentage_solution_l924_92448

variable (x y : ℝ)
variable (P : ℝ)

-- Conditions
axiom cond1 : 0.20 * (x - y) = (P / 100) * (x + y)
axiom cond2 : y = (1 / 7) * x

-- Theorem statement
theorem percentage_solution : P = 15 :=
by 
  -- Sorry means skipping the proof
  sorry

end percentage_solution_l924_92448


namespace taozi_is_faster_than_xiaoxiao_l924_92463

theorem taozi_is_faster_than_xiaoxiao : 
  let taozi_speed := 210
  let xiaoxiao_distance := 500
  let xiaoxiao_time := 3
  let xiaoxiao_speed := xiaoxiao_distance / xiaoxiao_time
  taozi_speed > xiaoxiao_speed
:= by
  let taozi_speed := 210
  let xiaoxiao_distance := 500
  let xiaoxiao_time := 3
  let xiaoxiao_speed := xiaoxiao_distance / xiaoxiao_time
  sorry

end taozi_is_faster_than_xiaoxiao_l924_92463


namespace compute_f_at_2012_l924_92404

noncomputable def B := { x : ℚ | x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 2 }

noncomputable def h (x : ℚ) : ℚ := 2 - (1 / x)

noncomputable def f (x : B) : ℝ := sorry  -- As a placeholder since the definition isn't given directly

-- Main theorem
theorem compute_f_at_2012 : 
  (∀ x : B, f x + f ⟨h x, sorry⟩ = Real.log (abs (2 * (x : ℚ)))) →
  f ⟨2012, sorry⟩ = Real.log ((4024 : ℚ) / (4023 : ℚ)) :=
sorry

end compute_f_at_2012_l924_92404


namespace prime_solution_exists_l924_92480

theorem prime_solution_exists (p : ℕ) (hp : Nat.Prime p) : ∃ x y z : ℤ, x^2 + y^2 + (p:ℤ) * z = 2003 := 
by 
  sorry

end prime_solution_exists_l924_92480


namespace find_m_minus_n_l924_92465

-- Define line equations, parallelism, and perpendicularity
def line1 (x y : ℝ) : Prop := 3 * x - 6 * y + 1 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := x - m * y + 2 = 0
def line3 (x y : ℝ) (n : ℝ) : Prop := n * x + y + 3 = 0

def parallel (m1 m2 : ℝ) : Prop := m1 = m2
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_m_minus_n (m n : ℝ) (h_parallel : parallel (1/2) (1/m)) (h_perpendicular: perpendicular (1/2) (-1/n)) : m - n = 0 :=
sorry

end find_m_minus_n_l924_92465


namespace cheese_fries_cost_l924_92492

def jim_money : ℝ := 20
def cousin_money : ℝ := 10
def combined_money : ℝ := jim_money + cousin_money
def expenditure : ℝ := 0.80 * combined_money
def cheeseburger_cost : ℝ := 3
def milkshake_cost : ℝ := 5
def cheeseburgers_cost : ℝ := 2 * cheeseburger_cost
def milkshakes_cost : ℝ := 2 * milkshake_cost
def meal_cost : ℝ := cheeseburgers_cost + milkshakes_cost

theorem cheese_fries_cost :
  let cheese_fries_cost := expenditure - meal_cost 
  cheese_fries_cost = 8 := 
by
  sorry

end cheese_fries_cost_l924_92492


namespace complement_A_U_l924_92429

-- Define the universal set U and set A as given in the problem.
def U : Set ℕ := { x | x ≥ 3 }
def A : Set ℕ := { x | x * x ≥ 10 }

-- Prove that the complement of A with respect to U is {3}.
theorem complement_A_U :
  (U \ A) = {3} :=
by
  sorry

end complement_A_U_l924_92429


namespace mower_value_drop_l924_92497

theorem mower_value_drop :
  ∀ (initial_value value_six_months value_after_year : ℝ) (percentage_drop_six_months percentage_drop_next_year : ℝ),
  initial_value = 100 →
  percentage_drop_six_months = 0.25 →
  value_six_months = initial_value * (1 - percentage_drop_six_months) →
  value_after_year = 60 →
  percentage_drop_next_year = 1 - (value_after_year / value_six_months) →
  percentage_drop_next_year * 100 = 20 :=
by
  intros initial_value value_six_months value_after_year percentage_drop_six_months percentage_drop_next_year
  intros h1 h2 h3 h4 h5
  sorry

end mower_value_drop_l924_92497


namespace quadratic_polynomial_l924_92440

theorem quadratic_polynomial (x y : ℝ) (hx : x + y = 12) (hy : x * (3 * y) = 108) : 
  (t : ℝ) → t^2 - 12 * t + 36 = 0 :=
by 
  sorry

end quadratic_polynomial_l924_92440
