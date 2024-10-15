import Mathlib

namespace NUMINAMATH_GPT_find_m_l1836_183679

-- Let m be a real number such that m > 1 and
-- \sum_{n=1}^{\infty} \frac{3n+2}{m^n} = 2.
theorem find_m (m : ℝ) (h1 : m > 1) 
(h2 : ∑' n : ℕ, (3 * (n + 1) + 2) / m^(n + 1) = 2) : 
  m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_l1836_183679


namespace NUMINAMATH_GPT_penguin_fish_consumption_l1836_183682

-- Definitions based on the conditions
def initial_penguins : ℕ := 158
def total_fish_per_day : ℕ := 237
def fish_per_penguin_per_day : ℚ := 1.5

-- Lean statement for the conditional problem
theorem penguin_fish_consumption
  (P : ℕ)
  (h_initial_penguins : P = initial_penguins)
  (h_total_fish_per_day : total_fish_per_day = 237)
  (h_current_penguins : P * 2 * 3 + 129 = 1077)
  : total_fish_per_day / P = fish_per_penguin_per_day := by
  sorry

end NUMINAMATH_GPT_penguin_fish_consumption_l1836_183682


namespace NUMINAMATH_GPT_m_range_for_circle_l1836_183698

def is_circle (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * (m - 3) * x + 2 * y + 5 = 0

theorem m_range_for_circle (m : ℝ) :
  (∀ x y : ℝ, is_circle x y m) → ((m > 5) ∨ (m < 1)) :=
by 
  sorry -- Proof not required

end NUMINAMATH_GPT_m_range_for_circle_l1836_183698


namespace NUMINAMATH_GPT_bridge_length_proof_l1836_183691

open Real

def train_length : ℝ := 100
def train_speed_kmh : ℝ := 45
def crossing_time_s: ℝ := 30

noncomputable def bridge_length : ℝ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem bridge_length_proof : bridge_length = 275 := 
by
  sorry

end NUMINAMATH_GPT_bridge_length_proof_l1836_183691


namespace NUMINAMATH_GPT_exist_infinite_a_l1836_183659

theorem exist_infinite_a (n : ℕ) (a : ℕ) (h₁ : ∃ k : ℕ, k > 0 ∧ (n^6 + 3 * a = (n^2 + 3 * k)^3)) : 
  ∃ f : ℕ → ℕ, ∀ m : ℕ, (∃ k : ℕ, k > 0 ∧ f m = 9 * k^3 + 3 * n^2 * k * (n^2 + 3 * k)) :=
by 
  sorry

end NUMINAMATH_GPT_exist_infinite_a_l1836_183659


namespace NUMINAMATH_GPT_product_of_solutions_is_zero_l1836_183629

theorem product_of_solutions_is_zero :
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) -> x = 0)) -> true :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_is_zero_l1836_183629


namespace NUMINAMATH_GPT_solve_trig_equation_proof_l1836_183642

noncomputable def solve_trig_equation (θ : ℝ) : Prop :=
  2 * Real.cos θ ^ 2 - 5 * Real.cos θ + 2 = 0 ∧ (θ = 60 / 180 * Real.pi)

theorem solve_trig_equation_proof (θ : ℝ) :
  solve_trig_equation θ :=
sorry

end NUMINAMATH_GPT_solve_trig_equation_proof_l1836_183642


namespace NUMINAMATH_GPT_max_tied_teams_for_most_wins_l1836_183622

-- Definitions based on conditions
def num_teams : ℕ := 7
def total_games_played : ℕ := num_teams * (num_teams - 1) / 2

-- Proposition stating the problem and the expected answer
theorem max_tied_teams_for_most_wins : 
  (∀ (t : ℕ), t ≤ num_teams → ∃ w : ℕ, t * w = total_games_played / num_teams) → 
  t = 7 :=
by
  sorry

end NUMINAMATH_GPT_max_tied_teams_for_most_wins_l1836_183622


namespace NUMINAMATH_GPT_y_star_definition_l1836_183643

def y_star (y : Real) : Real := y - 1

theorem y_star_definition (y : Real) : (5 : Real) - y_star 5 = 1 :=
  by sorry

end NUMINAMATH_GPT_y_star_definition_l1836_183643


namespace NUMINAMATH_GPT_fraction_filled_l1836_183632

-- Definitions for the given conditions
variables (x C : ℝ) (h₁ : 20 * x / 3 = 25 * C / 5) 

-- The goal is to show that x / C = 3 / 4
theorem fraction_filled (h₁ : 20 * x / 3 = 25 * C / 5) : x / C = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_fraction_filled_l1836_183632


namespace NUMINAMATH_GPT_sqrt_product_eq_six_l1836_183648

theorem sqrt_product_eq_six (sqrt24 sqrtThreeOverTwo: ℝ)
    (h1 : sqrt24 = Real.sqrt 24)
    (h2 : sqrtThreeOverTwo = Real.sqrt (3 / 2))
    : sqrt24 * sqrtThreeOverTwo = 6 := by
  sorry

end NUMINAMATH_GPT_sqrt_product_eq_six_l1836_183648


namespace NUMINAMATH_GPT_greatest_sum_consecutive_integers_lt_500_l1836_183680

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end NUMINAMATH_GPT_greatest_sum_consecutive_integers_lt_500_l1836_183680


namespace NUMINAMATH_GPT_min_sum_nonpos_l1836_183653

theorem min_sum_nonpos (a b : ℤ) (h_nonpos_a : a ≤ 0) (h_nonpos_b : b ≤ 0) (h_prod : a * b = 144) : 
  a + b = -30 :=
sorry

end NUMINAMATH_GPT_min_sum_nonpos_l1836_183653


namespace NUMINAMATH_GPT_greatest_line_segment_length_l1836_183664

theorem greatest_line_segment_length (r : ℝ) (h : r = 4) : 
  ∃ d : ℝ, d = 2 * r ∧ d = 8 :=
by
  sorry

end NUMINAMATH_GPT_greatest_line_segment_length_l1836_183664


namespace NUMINAMATH_GPT_polynomial_solutions_l1836_183626

theorem polynomial_solutions (P : Polynomial ℝ) :
  (∀ x : ℝ, P.eval x * P.eval (x + 1) = P.eval (x^2 - x + 3)) →
  (P = 0 ∨ ∃ n : ℕ, P = (Polynomial.C 1) * (Polynomial.X^2 - 2 * Polynomial.X + 3)^n) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solutions_l1836_183626


namespace NUMINAMATH_GPT_ratio_of_efficiencies_l1836_183610

-- Definitions of efficiencies
def efficiency (time : ℕ) : ℚ := 1 / time

-- Conditions:
def E_C : ℚ := efficiency 20
def E_D : ℚ := efficiency 30
def E_A : ℚ := efficiency 18
def E_B : ℚ := 1 / 36 -- Placeholder for efficiency of B to complete the statement

-- The proof goal
theorem ratio_of_efficiencies (h1 : E_A + E_B = E_C + E_D) : E_A / E_B = 2 :=
by
  -- Placeholder to structure the format, the proof will be constructed here
  sorry

end NUMINAMATH_GPT_ratio_of_efficiencies_l1836_183610


namespace NUMINAMATH_GPT_least_number_of_stamps_is_11_l1836_183667

theorem least_number_of_stamps_is_11 (s t : ℕ) (h : 5 * s + 6 * t = 60) : s + t = 11 := 
  sorry

end NUMINAMATH_GPT_least_number_of_stamps_is_11_l1836_183667


namespace NUMINAMATH_GPT_mike_earnings_l1836_183604

theorem mike_earnings :
  let total_games := 16
  let non_working_games := 8
  let price_per_game := 7
  let working_games := total_games - non_working_games
  let earnings := working_games * price_per_game
  earnings = 56 := 
by
  sorry

end NUMINAMATH_GPT_mike_earnings_l1836_183604


namespace NUMINAMATH_GPT_sqrt_9_is_pm3_l1836_183674

theorem sqrt_9_is_pm3 : {x : ℝ | x ^ 2 = 9} = {3, -3} := sorry

end NUMINAMATH_GPT_sqrt_9_is_pm3_l1836_183674


namespace NUMINAMATH_GPT_factorize_expression_l1836_183658

-- The problem is about factorizing the expression x^3y - xy
theorem factorize_expression (x y : ℝ) : x^3 * y - x * y = x * y * (x - 1) * (x + 1) := 
by sorry

end NUMINAMATH_GPT_factorize_expression_l1836_183658


namespace NUMINAMATH_GPT_cubic_function_value_l1836_183675

noncomputable def g (x : ℝ) (p q r s : ℝ) : ℝ := p * x ^ 3 + q * x ^ 2 + r * x + s

theorem cubic_function_value (p q r s : ℝ) (h : g (-3) p q r s = -2) :
  12 * p - 6 * q + 3 * r - s = 2 :=
sorry

end NUMINAMATH_GPT_cubic_function_value_l1836_183675


namespace NUMINAMATH_GPT_shaded_area_a_length_EF_b_length_EF_c_ratio_ab_d_l1836_183696

-- (a) Prove that the area of the shaded region is 36 cm^2
theorem shaded_area_a (AB EF : ℕ) (h1 : AB = 10) (h2 : EF = 8) : (AB ^ 2) - (EF ^ 2) = 36 :=
by
  sorry

-- (b) Prove that the length of EF is 7 cm
theorem length_EF_b (AB : ℕ) (shaded_area : ℕ) (h1 : AB = 13) (h2 : shaded_area = 120)
  : ∃ EF, (AB ^ 2) - (EF ^ 2) = shaded_area ∧ EF = 7 :=
by
  sorry

-- (c) Prove that the length of EF is 9 cm
theorem length_EF_c (AB : ℕ) (h1 : AB = 18)
  : ∃ EF, (AB ^ 2) - ((1 / 4) * AB ^ 2) = (3 / 4) * AB ^ 2 ∧ EF = 9 :=
by
  sorry

-- (d) Prove that a / b = 5 / 3
theorem ratio_ab_d (a b : ℕ) (shaded_percent : ℚ) (h1 : shaded_percent = 0.64)
  : (a ^ 2) - ((0.36) * a ^ 2) = (a ^ 2) * shaded_percent ∧ (a / b) = (5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_a_length_EF_b_length_EF_c_ratio_ab_d_l1836_183696


namespace NUMINAMATH_GPT_tan_alpha_of_cos_alpha_l1836_183631

theorem tan_alpha_of_cos_alpha (α : ℝ) (hα : 0 < α ∧ α < Real.pi) (h_cos : Real.cos α = -3/5) :
  Real.tan α = -4/3 :=
sorry

end NUMINAMATH_GPT_tan_alpha_of_cos_alpha_l1836_183631


namespace NUMINAMATH_GPT_solution_interval_l1836_183657

theorem solution_interval:
  ∃ x : ℝ, (x^3 = 2^(2-x)) ∧ 1 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_interval_l1836_183657


namespace NUMINAMATH_GPT_find_g3_l1836_183672

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x^2 + b * x^3 + c * x + d

theorem find_g3 (a b c d : ℝ) (h : g (-3) a b c d = 2) : g 3 a b c d = 0 := 
by 
  sorry

end NUMINAMATH_GPT_find_g3_l1836_183672


namespace NUMINAMATH_GPT_macaroon_weight_l1836_183654

theorem macaroon_weight (bakes : ℕ) (packs : ℕ) (bags_after_eat : ℕ) (remaining_weight : ℕ) (macaroons_per_bag : ℕ) (weight_per_bag : ℕ)
  (H1 : bakes = 12) 
  (H2 : packs = 4)
  (H3 : bags_after_eat = 3)
  (H4 : remaining_weight = 45)
  (H5 : macaroons_per_bag = bakes / packs) 
  (H6 : weight_per_bag = remaining_weight / bags_after_eat) :
  ∀ (weight_per_macaroon : ℕ), weight_per_macaroon = weight_per_bag / macaroons_per_bag → weight_per_macaroon = 5 :=
by
  sorry -- Proof will come here, not required as per instructions

end NUMINAMATH_GPT_macaroon_weight_l1836_183654


namespace NUMINAMATH_GPT_task_completion_days_l1836_183628

theorem task_completion_days (a b c d : ℝ) 
    (h1 : 1/a + 1/b = 1/8)
    (h2 : 1/b + 1/c = 1/6)
    (h3 : 1/c + 1/d = 1/12) :
    1/a + 1/d = 1/24 :=
by
  sorry

end NUMINAMATH_GPT_task_completion_days_l1836_183628


namespace NUMINAMATH_GPT_sandy_marks_l1836_183613

def marks_each_correct_sum : ℕ := 3

theorem sandy_marks (x : ℕ) 
  (total_attempts : ℕ := 30)
  (correct_sums : ℕ := 23)
  (marks_per_incorrect_sum : ℕ := 2)
  (total_marks_obtained : ℕ := 55)
  (incorrect_sums : ℕ := total_attempts - correct_sums)
  (lost_marks : ℕ := incorrect_sums * marks_per_incorrect_sum) :
  (correct_sums * x - lost_marks = total_marks_obtained) -> x = marks_each_correct_sum :=
by
  sorry

end NUMINAMATH_GPT_sandy_marks_l1836_183613


namespace NUMINAMATH_GPT_find_prime_c_l1836_183652

-- Define the statement of the problem
theorem find_prime_c (c : ℕ) (hc : Nat.Prime c) (h : ∃ m : ℕ, (m > 0) ∧ (11 * c + 1 = m^2)) : c = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_prime_c_l1836_183652


namespace NUMINAMATH_GPT_no_x2_term_a_eq_1_l1836_183656

theorem no_x2_term_a_eq_1 (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a * x + 1) * (x^2 - 3 * a + 2) = x^4 + bx^3 + cx + d) →
  c = 0 →
  a = 1 :=
sorry

end NUMINAMATH_GPT_no_x2_term_a_eq_1_l1836_183656


namespace NUMINAMATH_GPT_point_outside_circle_l1836_183684

theorem point_outside_circle (a b : ℝ)
  (h_line_intersects_circle : ∃ (x1 y1 x2 y2 : ℝ), 
     x1^2 + y1^2 = 1 ∧ 
     x2^2 + y2^2 = 1 ∧ 
     a * x1 + b * y1 = 1 ∧ 
     a * x2 + b * y2 = 1 ∧ 
     (x1, y1) ≠ (x2, y2)) : 
  a^2 + b^2 > 1 :=
sorry

end NUMINAMATH_GPT_point_outside_circle_l1836_183684


namespace NUMINAMATH_GPT_transylvanian_sanity_l1836_183671

theorem transylvanian_sanity (sane : Prop) (belief : Prop) (h1 : sane) (h2 : sane → belief) : belief :=
by
  sorry

end NUMINAMATH_GPT_transylvanian_sanity_l1836_183671


namespace NUMINAMATH_GPT_dodecahedron_path_count_l1836_183637

/-- A regular dodecahedron with constraints on movement between faces. -/
def num_ways_dodecahedron_move : Nat := 810

/-- Proving the number of different ways to move from the top face to the bottom face of a regular dodecahedron via a series of adjacent faces, such that each face is visited at most once, and movement from the lower ring to the upper ring is not allowed is 810. -/
theorem dodecahedron_path_count :
  num_ways_dodecahedron_move = 810 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_dodecahedron_path_count_l1836_183637


namespace NUMINAMATH_GPT_find_b_value_l1836_183607

theorem find_b_value (f : ℝ → ℝ) (f_inv : ℝ → ℝ) (b : ℝ) :
  (∀ x, f x = 1 / (3 * x + b)) →
  (∀ x, f_inv x = (2 - 3 * x) / (3 * x)) →
  b = -3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_b_value_l1836_183607


namespace NUMINAMATH_GPT_subscription_difference_is_4000_l1836_183617

-- Given definitions
def total_subscription (A B C : ℕ) : Prop :=
  A + B + C = 50000

def subscription_B (x : ℕ) : ℕ :=
  x + 5000

def subscription_A (x y : ℕ) : ℕ :=
  x + 5000 + y

def profit_ratio (profit_C total_profit x : ℕ) : Prop :=
  (profit_C : ℚ) / total_profit = (x : ℚ) / 50000

-- Prove that A subscribed Rs. 4,000 more than B
theorem subscription_difference_is_4000 (x y : ℕ)
  (h1 : total_subscription (subscription_A x y) (subscription_B x) x)
  (h2 : profit_ratio 8400 35000 x) :
  y = 4000 :=
sorry

end NUMINAMATH_GPT_subscription_difference_is_4000_l1836_183617


namespace NUMINAMATH_GPT_nonempty_solution_set_range_l1836_183624

theorem nonempty_solution_set_range (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 := sorry

end NUMINAMATH_GPT_nonempty_solution_set_range_l1836_183624


namespace NUMINAMATH_GPT_find_range_of_f_l1836_183627

noncomputable def f (x : ℝ) : ℝ := (Real.logb (1/2) x) ^ 2 - 2 * (Real.logb (1/2) x) + 4

theorem find_range_of_f :
  ∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → 7 ≤ f x ∧ f x ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_f_l1836_183627


namespace NUMINAMATH_GPT_person_b_worked_alone_days_l1836_183641

theorem person_b_worked_alone_days :
  ∀ (x : ℕ), 
  (x / 10 + (12 - x) / 20 = 1) → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_person_b_worked_alone_days_l1836_183641


namespace NUMINAMATH_GPT_find_root_D_l1836_183689

/-- Given C and D are roots of the polynomial k x^2 + 2 x + 5 = 0, 
    and k = -1/4 and C = 10, then D must be -2. -/
theorem find_root_D 
  (k : ℚ) (C D : ℚ)
  (h1 : k = -1/4)
  (h2 : C = 10)
  (h3 : C^2 * k + 2 * C + 5 = 0)
  (h4 : D^2 * k + 2 * D + 5 = 0) : 
  D = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_root_D_l1836_183689


namespace NUMINAMATH_GPT_cheat_buying_percentage_l1836_183647

-- Definitions for the problem
def profit_margin := 0.5
def cheat_selling := 0.2

-- Prove that the cheating percentage while buying is 20%
theorem cheat_buying_percentage : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ x = 0.2 := by
  sorry

end NUMINAMATH_GPT_cheat_buying_percentage_l1836_183647


namespace NUMINAMATH_GPT_Olivia_pays_4_dollars_l1836_183606

-- Definitions based on the conditions
def quarters_chips : ℕ := 4
def quarters_soda : ℕ := 12
def conversion_rate : ℕ := 4

-- Prove that the total dollars Olivia pays is 4
theorem Olivia_pays_4_dollars (h1 : quarters_chips = 4) (h2 : quarters_soda = 12) (h3 : conversion_rate = 4) : 
  (quarters_chips + quarters_soda) / conversion_rate = 4 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_Olivia_pays_4_dollars_l1836_183606


namespace NUMINAMATH_GPT_maximum_fraction_l1836_183646

theorem maximum_fraction (a b h : ℝ) (d : ℝ) (h_d_def : d = Real.sqrt (a^2 + b^2 + h^2)) :
  (a + b + h) / d ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_maximum_fraction_l1836_183646


namespace NUMINAMATH_GPT_intersection_in_fourth_quadrant_l1836_183683

variable {a : ℝ} {x : ℝ}

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x / Real.log a
noncomputable def g (x : ℝ) (a : ℝ) := (1 - a) * x

theorem intersection_in_fourth_quadrant (h : a > 1) :
  ∃ x : ℝ, x > 0 ∧ f x a < 0 ∧ f x a = g x a :=
sorry

end NUMINAMATH_GPT_intersection_in_fourth_quadrant_l1836_183683


namespace NUMINAMATH_GPT_sandy_goal_hours_l1836_183635

def goal_liters := 3 -- The goal in liters
def liters_to_milliliters := 1000 -- Conversion rate from liters to milliliters
def goal_milliliters := goal_liters * liters_to_milliliters -- Total milliliters to drink
def drink_rate_milliliters := 500 -- Milliliters drunk every interval
def interval_hours := 2 -- Interval in hours

def sets_to_goal := goal_milliliters / drink_rate_milliliters -- The number of drink sets to reach the goal
def total_hours := sets_to_goal * interval_hours -- Total time in hours to reach the goal

theorem sandy_goal_hours : total_hours = 12 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_sandy_goal_hours_l1836_183635


namespace NUMINAMATH_GPT_simplify_expression_l1836_183666

theorem simplify_expression : 
  (((5 + 7 + 3) * 2 - 4) / 2 - (5 / 2) = 21 / 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1836_183666


namespace NUMINAMATH_GPT_bob_speed_lt_40_l1836_183697

theorem bob_speed_lt_40 (v_b v_a : ℝ) (h1 : v_a > 45) (h2 : 180 / v_a < 180 / v_b - 0.5) :
  v_b < 40 :=
by
  -- Variables and constants
  let distance := 180
  let min_speed_alice := 45
  -- Conditions
  have h_distance := distance
  have h_min_speed_alice := min_speed_alice
  have h_time_alice := (distance : ℝ) / v_a
  have h_time_bob := (distance : ℝ) / v_b
  -- Given conditions inequalities
  have ineq := h2
  have alice_min_speed := h1
  -- Now apply these facts and derived inequalities to prove bob_speed_lt_40
  sorry

end NUMINAMATH_GPT_bob_speed_lt_40_l1836_183697


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1836_183608

def f (x : ℝ) : ℝ := sorry
def f_prime (x : ℝ) : ℝ := sorry

theorem solution_set_of_inequality :
  (∀ x > 0, x^2 * f_prime x + 1 > 0) → 
  f 1 = 5 →
  { x : ℝ | 0 < x ∧ x < 1 } = { x : ℝ | 0 < x ∧ f x < 1 / x + 4 } :=
by 
  intros h1 h2 
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1836_183608


namespace NUMINAMATH_GPT_total_interest_calculation_l1836_183665

-- Define the total investment
def total_investment : ℝ := 20000

-- Define the fractional part of investment at 9 percent rate
def fraction_higher_rate : ℝ := 0.55

-- Define the investment amounts based on the fractional part
def investment_higher_rate : ℝ := fraction_higher_rate * total_investment
def investment_lower_rate : ℝ := total_investment - investment_higher_rate

-- Define interest rates
def rate_lower : ℝ := 0.06
def rate_higher : ℝ := 0.09

-- Define time period (in years)
def time_period : ℝ := 1

-- Define interest calculations
def interest_lower : ℝ := investment_lower_rate * rate_lower * time_period
def interest_higher : ℝ := investment_higher_rate * rate_higher * time_period

-- Define the total interest
def total_interest : ℝ := interest_lower + interest_higher

-- Theorem stating the total interest earned
theorem total_interest_calculation : total_interest = 1530 := by
  -- skip proof using sorry
  sorry

end NUMINAMATH_GPT_total_interest_calculation_l1836_183665


namespace NUMINAMATH_GPT_restock_quantities_correct_l1836_183688

-- Definition for the quantities of cans required
def cans_peas : ℕ := 810
def cans_carrots : ℕ := 954
def cans_corn : ℕ := 675

-- Definition for the number of cans per box, pack, and case.
def cans_per_box_peas : ℕ := 4
def cans_per_pack_carrots : ℕ := 6
def cans_per_case_corn : ℕ := 5

-- Define the expected order quantities.
def order_boxes_peas : ℕ := 203
def order_packs_carrots : ℕ := 159
def order_cases_corn : ℕ := 135

-- Proof statement for the quantities required to restock exactly.
theorem restock_quantities_correct :
  (order_boxes_peas = Nat.ceil (cans_peas / cans_per_box_peas))
  ∧ (order_packs_carrots = cans_carrots / cans_per_pack_carrots)
  ∧ (order_cases_corn = cans_corn / cans_per_case_corn) :=
by
  sorry

end NUMINAMATH_GPT_restock_quantities_correct_l1836_183688


namespace NUMINAMATH_GPT_v_2015_eq_2_l1836_183634

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 4
  | 4 => 1
  | 5 => 2
  | _ => 0  -- assuming g(x) = 0 for other values, though not used here

def v : ℕ → ℕ
| 0     => 3
| (n+1) => g (v n)

theorem v_2015_eq_2 : v 2015 = 2 :=
by
  sorry

end NUMINAMATH_GPT_v_2015_eq_2_l1836_183634


namespace NUMINAMATH_GPT__l1836_183645

def triangle (A B C : Type) : Prop :=
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

def angles_not_equal_sides_not_equal (A B C : Type) (angleB angleC : ℝ) (sideAC sideAB : ℝ) : Prop :=
  triangle A B C →
  (angleB ≠ angleC → sideAC ≠ sideAB)
  
lemma xiaoming_theorem {A B C : Type} 
  (hTriangle : triangle A B C)
  (angleB angleC : ℝ)
  (sideAC sideAB : ℝ) :
  angleB ≠ angleC → sideAC ≠ sideAB := 
sorry

end NUMINAMATH_GPT__l1836_183645


namespace NUMINAMATH_GPT_determine_y_l1836_183650

theorem determine_y (y : ℝ) (y_nonzero : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 := 
sorry

end NUMINAMATH_GPT_determine_y_l1836_183650


namespace NUMINAMATH_GPT_fabian_cards_l1836_183640

theorem fabian_cards : ∃ (g y b r : ℕ),
  (g > 0 ∧ g < 10) ∧ (y > 0 ∧ y < 10) ∧ (b > 0 ∧ b < 10) ∧ (r > 0 ∧ r < 10) ∧
  (g * y = g) ∧
  (b = r) ∧
  (b * r = 10 * g + y) ∧ 
  (g = 8) ∧
  (y = 1) ∧
  (b = 9) ∧
  (r = 9) :=
by
  sorry

end NUMINAMATH_GPT_fabian_cards_l1836_183640


namespace NUMINAMATH_GPT_correct_population_statement_l1836_183615

def correct_statement :=
  "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population."

def sample_size : ℕ := 500

def is_correct (statement : String) : Prop :=
  statement = correct_statement

theorem correct_population_statement (scores : Fin 500 → ℝ) :
  is_correct "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population." :=
by
  sorry

end NUMINAMATH_GPT_correct_population_statement_l1836_183615


namespace NUMINAMATH_GPT_eggs_at_park_l1836_183644

-- Define the number of eggs found at different locations
def eggs_at_club_house : Nat := 40
def eggs_at_town_hall : Nat := 15
def total_eggs_found : Nat := 80

-- Prove that the number of eggs found at the park is 25
theorem eggs_at_park :
  ∃ P : Nat, eggs_at_club_house + P + eggs_at_town_hall = total_eggs_found ∧ P = 25 := 
by
  sorry

end NUMINAMATH_GPT_eggs_at_park_l1836_183644


namespace NUMINAMATH_GPT_cars_in_parking_lot_l1836_183600

theorem cars_in_parking_lot (initial_cars left_cars entered_cars : ℕ) (h1 : initial_cars = 80)
(h2 : left_cars = 13) (h3 : entered_cars = left_cars + 5) : 
initial_cars - left_cars + entered_cars = 85 :=
by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_cars_in_parking_lot_l1836_183600


namespace NUMINAMATH_GPT_inequality_solution_equality_condition_l1836_183603

theorem inequality_solution (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) (h3 : b < -1 ∨ b > 0) :
  (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b :=
sorry

theorem equality_condition (a b : ℝ) :
  (1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b :=
sorry

end NUMINAMATH_GPT_inequality_solution_equality_condition_l1836_183603


namespace NUMINAMATH_GPT_inequality_problem_l1836_183681

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 / (b * (a + b)) + 2 / (c * (b + c)) + 2 / (a * (c + a))) ≥ (27 / (a + b + c)^2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l1836_183681


namespace NUMINAMATH_GPT_A_sub_B_value_l1836_183612

def A : ℕ := 1000 * 1 + 100 * 16 + 10 * 28
def B : ℕ := 355 + 245 * 3

theorem A_sub_B_value : A - B = 1790 := by
  sorry

end NUMINAMATH_GPT_A_sub_B_value_l1836_183612


namespace NUMINAMATH_GPT_find_f_of_4_l1836_183649

def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem find_f_of_4 {a b c : ℝ} (h1 : f a b c 1 = 3) (h2 : f a b c 2 = 12) (h3 : f a b c 3 = 27) :
  f a b c 4 = 48 := 
sorry

end NUMINAMATH_GPT_find_f_of_4_l1836_183649


namespace NUMINAMATH_GPT_not_divisible_67_l1836_183611

theorem not_divisible_67
  (x y : ℕ)
  (hx : ¬ (67 ∣ x))
  (hy : ¬ (67 ∣ y))
  (h : (7 * x + 32 * y) % 67 = 0)
  : (10 * x + 17 * y + 1) % 67 ≠ 0 := sorry

end NUMINAMATH_GPT_not_divisible_67_l1836_183611


namespace NUMINAMATH_GPT_find_constant_l1836_183620

-- Definitions based on the conditions provided
variable (f : ℕ → ℕ)
variable (c : ℕ)

-- Given conditions
def f_1_eq_0 : f 1 = 0 := sorry
def functional_equation (m n : ℕ) : f (m + n) = f m + f n + c * (m * n - 1) := sorry
def f_17_eq_4832 : f 17 = 4832 := sorry

-- The mathematically equivalent proof problem
theorem find_constant : c = 4 := 
sorry

end NUMINAMATH_GPT_find_constant_l1836_183620


namespace NUMINAMATH_GPT_cube_construction_possible_l1836_183609

theorem cube_construction_possible (n : ℕ) : (∃ k : ℕ, n = 12 * k) ↔ ∃ V : ℕ, (n ^ 3) = 12 * V := by
sorry

end NUMINAMATH_GPT_cube_construction_possible_l1836_183609


namespace NUMINAMATH_GPT_tan_double_angle_l1836_183625

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f_derivative_def (x : ℝ) : ℝ := 3 * f x

theorem tan_double_angle (x : ℝ) (h : f_derivative_def x = Real.cos x - Real.sin x) : 
  Real.tan (2 * x) = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l1836_183625


namespace NUMINAMATH_GPT_sum_of_positive_x_and_y_is_ten_l1836_183687

theorem sum_of_positive_x_and_y_is_ten (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x^3 + y^3 + (x + y)^3 + 30 * x * y = 2000) : 
  x + y = 10 :=
sorry

end NUMINAMATH_GPT_sum_of_positive_x_and_y_is_ten_l1836_183687


namespace NUMINAMATH_GPT_sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq_l1836_183686

open Real

theorem sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq :
  (∀ α : ℝ, sin α = cos α → ∃ k : ℤ, α = (k : ℝ) * π + π / 4) ∧
  (¬ ∀ k : ℤ, ∀ α : ℝ, α = (k : ℝ) * π + π / 4 → sin α = cos α) :=
by
  sorry

end NUMINAMATH_GPT_sin_eq_cos_is_necessary_but_not_sufficient_for_alpha_eq_l1836_183686


namespace NUMINAMATH_GPT_complex_root_problem_l1836_183601

theorem complex_root_problem (z : ℂ) :
  z^2 - 3*z = 10 - 6*Complex.I ↔
  z = 5.5 - 0.75 * Complex.I ∨
  z = -2.5 + 0.75 * Complex.I ∨
  z = 3.5 - 1.5 * Complex.I ∨
  z = -0.5 + 1.5 * Complex.I :=
sorry

end NUMINAMATH_GPT_complex_root_problem_l1836_183601


namespace NUMINAMATH_GPT_tenth_term_geometric_sequence_l1836_183699

theorem tenth_term_geometric_sequence :
  let a := (8 : ℚ)
  let r := (-2 / 3 : ℚ)
  a * r^9 = -4096 / 19683 :=
by
  sorry

end NUMINAMATH_GPT_tenth_term_geometric_sequence_l1836_183699


namespace NUMINAMATH_GPT_given_statements_l1836_183669

def addition_is_associative (x y z : ℝ) : Prop := (x + y) + z = x + (y + z)

def averaging_is_commutative (x y : ℝ) : Prop := (x + y) / 2 = (y + x) / 2

def addition_distributes_over_averaging (x y z : ℝ) : Prop := 
  x + (y + z) / 2 = (x + y + x + z) / 2

def averaging_distributes_over_addition (x y z : ℝ) : Prop := 
  (x + (y + z)) / 2 = ((x + y) / 2) + ((x + z) / 2)

def averaging_has_identity_element (x e : ℝ) : Prop := 
  (x + e) / 2 = x

theorem given_statements (x y z e : ℝ) :
  addition_is_associative x y z ∧ 
  averaging_is_commutative x y ∧ 
  addition_distributes_over_averaging x y z ∧ 
  ¬averaging_distributes_over_addition x y z ∧ 
  ¬∃ e, averaging_has_identity_element x e :=
by
  sorry

end NUMINAMATH_GPT_given_statements_l1836_183669


namespace NUMINAMATH_GPT_mark_sprint_distance_l1836_183692

theorem mark_sprint_distance (t v : ℝ) (ht : t = 24.0) (hv : v = 6.0) : 
  t * v = 144.0 := 
by
  -- This theorem is formulated with the conditions that t = 24.0 and v = 6.0,
  -- we need to prove that the resulting distance is 144.0 miles.
  sorry

end NUMINAMATH_GPT_mark_sprint_distance_l1836_183692


namespace NUMINAMATH_GPT_triangle_area_of_integral_sides_with_perimeter_8_l1836_183663

theorem triangle_area_of_integral_sides_with_perimeter_8 :
  ∃ (a b c : ℕ), a + b + c = 8 ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ 
  ∃ (area : ℝ), area = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_triangle_area_of_integral_sides_with_perimeter_8_l1836_183663


namespace NUMINAMATH_GPT_tea_set_costs_l1836_183673
noncomputable section

-- Definition for the conditions of part 1
def cost_condition1 (x y : ℝ) : Prop := x + 2 * y = 250
def cost_condition2 (x y : ℝ) : Prop := 3 * x + 4 * y = 600

-- Definition for the conditions of part 2
def cost_condition3 (a : ℝ) : ℝ := 108 * a + 60 * (80 - a)

-- Definition for the conditions of part 3
def profit (a b : ℝ) : ℝ := 30 * a + 20 * b

theorem tea_set_costs (x y : ℝ) (a : ℕ) :
  cost_condition1 x y →
  cost_condition2 x y →
  x = 100 ∧ y = 75 ∧ a ≤ 30 ∧ profit 30 50 = 1900 := by
  sorry

end NUMINAMATH_GPT_tea_set_costs_l1836_183673


namespace NUMINAMATH_GPT_no_such_function_exists_l1836_183670

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = Real.sin x :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l1836_183670


namespace NUMINAMATH_GPT_infinite_chain_resistance_l1836_183621

variables (R_0 R_X : ℝ)
def infinite_chain_resistance_condition (R_0 : ℝ) (R_X : ℝ) : Prop :=
  R_X = R_0 + (R_0 * R_X) / (R_0 + R_X)

theorem infinite_chain_resistance (R_0 : ℝ) (h : R_0 = 50) :
  ∃ R_X, infinite_chain_resistance_condition R_0 R_X ∧ R_X = (R_0 * (1 + Real.sqrt 5)) / 2 :=
  sorry

end NUMINAMATH_GPT_infinite_chain_resistance_l1836_183621


namespace NUMINAMATH_GPT_problem_220_l1836_183690

variables (x y : ℝ)

theorem problem_220 (h1 : x + y = 10) (h2 : (x * y) / (x^2) = -3 / 2) :
  x = -20 ∧ y = 30 :=
by
  sorry

end NUMINAMATH_GPT_problem_220_l1836_183690


namespace NUMINAMATH_GPT_max_mn_value_l1836_183623

noncomputable def vector_max_sum (OA OB : ℝ) (m n : ℝ) : Prop :=
  (OA * OA = 4 ∧ OB * OB = 4 ∧ OA * OB = 2) →
  ((m * OA + n * OB) * (m * OA + n * OB) = 4) →
  (m + n ≤ 2 * Real.sqrt 3 / 3)

-- Here's the statement for the maximum value problem
theorem max_mn_value {m n : ℝ} (h1 : m > 0) (h2 : n > 0) :
  vector_max_sum 2 2 m n :=
sorry

end NUMINAMATH_GPT_max_mn_value_l1836_183623


namespace NUMINAMATH_GPT_circle_area_l1836_183618

/--
Given the polar equation of a circle r = -4 * cos θ + 8 * sin θ,
prove that the area of the circle is 20π.
-/
theorem circle_area (θ : ℝ) (r : ℝ) (cos : ℝ → ℝ) (sin : ℝ → ℝ) 
  (h_eq : ∀ θ : ℝ, r = -4 * cos θ + 8 * sin θ) : 
  ∃ A : ℝ, A = 20 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_area_l1836_183618


namespace NUMINAMATH_GPT_distance_to_fourth_side_l1836_183693

-- Let s be the side length of the square.
variable (s : ℝ) (d1 d2 d3 d4 : ℝ)

-- The given conditions:
axiom h1 : d1 = 4
axiom h2 : d2 = 7
axiom h3 : d3 = 13
axiom h4 : d1 + d2 + d3 + d4 = s
axiom h5 : 0 < d4

-- The statement to prove:
theorem distance_to_fourth_side : d4 = 10 ∨ d4 = 16 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_fourth_side_l1836_183693


namespace NUMINAMATH_GPT_percentage_of_400_that_results_in_224_point_5_l1836_183636

-- Let x be the unknown percentage of 400
variable (x : ℝ)

-- Condition: x% of 400 plus 45% of 250 equals 224.5
def condition (x : ℝ) : Prop := (400 * x / 100) + (250 * 45 / 100) = 224.5

theorem percentage_of_400_that_results_in_224_point_5 : condition 28 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_percentage_of_400_that_results_in_224_point_5_l1836_183636


namespace NUMINAMATH_GPT_Delta_15_xDelta_eq_neg_15_l1836_183605

-- Definitions of the operations based on conditions
def xDelta (x : ℝ) : ℝ := 9 - x
def Delta (x : ℝ) : ℝ := x - 9

-- Statement that we need to prove
theorem Delta_15_xDelta_eq_neg_15 : Delta (xDelta 15) = -15 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_Delta_15_xDelta_eq_neg_15_l1836_183605


namespace NUMINAMATH_GPT_actual_plot_area_in_acres_l1836_183676

-- Condition Definitions
def base_cm : ℝ := 8
def height_cm : ℝ := 12
def scale_cm_to_miles : ℝ := 1  -- 1 cm = 1 mile
def miles_to_acres : ℝ := 320  -- 1 square mile = 320 acres

-- Theorem Statement
theorem actual_plot_area_in_acres (A : ℝ) :
  A = 15360 :=
by
  sorry

end NUMINAMATH_GPT_actual_plot_area_in_acres_l1836_183676


namespace NUMINAMATH_GPT_cute_pairs_count_l1836_183694

def is_cute_pair (a b : ℕ) : Prop :=
  a ≥ b / 2 + 7 ∧ b ≥ a / 2 + 7

def max_cute_pairs : Prop :=
  ∀ (ages : Finset ℕ), 
  (∀ x ∈ ages, 1 ≤ x ∧ x ≤ 100) →
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ pair ∈ pairs, is_cute_pair pair.1 pair.2) ∧
    (∀ x ∈ pairs, ∀ y ∈ pairs, x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) ∧
    pairs.card = 43)

theorem cute_pairs_count : max_cute_pairs := 
sorry

end NUMINAMATH_GPT_cute_pairs_count_l1836_183694


namespace NUMINAMATH_GPT_area_of_fifteen_sided_figure_l1836_183655

noncomputable def figure_area : ℝ :=
  let full_squares : ℝ := 6
  let num_triangles : ℝ := 10
  let triangles_to_rectangles : ℝ := num_triangles / 2
  let triangles_area : ℝ := triangles_to_rectangles
  full_squares + triangles_area

theorem area_of_fifteen_sided_figure :
  figure_area = 11 := by
  sorry

end NUMINAMATH_GPT_area_of_fifteen_sided_figure_l1836_183655


namespace NUMINAMATH_GPT_solution_set_of_abs_x_minus_1_lt_1_l1836_183616

theorem solution_set_of_abs_x_minus_1_lt_1 : {x : ℝ | |x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_abs_x_minus_1_lt_1_l1836_183616


namespace NUMINAMATH_GPT_volumes_relation_l1836_183638

-- Definitions and conditions based on the problem
variables {a b c : ℝ} (h_triangle : a > b) (h_triangle2 : b > c) (h_acute : 0 < θ ∧ θ < π)

-- The heights from vertices
variables (AD BE CF : ℝ)

-- Volumes of the tetrahedrons formed after folding
variables (V1 V2 V3 : ℝ)

-- The heights are given:
noncomputable def height_AD (BC : ℝ) (theta : ℝ) := AD
noncomputable def height_BE (CA : ℝ) (theta : ℝ) := BE
noncomputable def height_CF (AB : ℝ) (theta : ℝ) := CF

-- Using these heights and the acute nature of the triangle
noncomputable def volume_V1 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V1
noncomputable def volume_V2 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V2
noncomputable def volume_V3 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V3

-- The theorem stating the relationship between volumes
theorem volumes_relation
  (h_triangle: a > b)
  (h_triangle2: b > c)
  (h_acute: 0 < θ ∧ θ < π)
  (h_volumes: V1 > V2 ∧ V2 > V3):
  V1 > V2 ∧ V2 > V3 :=
sorry

end NUMINAMATH_GPT_volumes_relation_l1836_183638


namespace NUMINAMATH_GPT_checkerboard_sum_is_328_l1836_183639

def checkerboard_sum : Nat :=
  1 + 2 + 9 + 8 + 73 + 74 + 81 + 80

theorem checkerboard_sum_is_328 : checkerboard_sum = 328 := by
  sorry

end NUMINAMATH_GPT_checkerboard_sum_is_328_l1836_183639


namespace NUMINAMATH_GPT_ratio_of_daily_wages_l1836_183677

-- Definitions for daily wages and conditions
def daily_wage_man : ℝ := sorry
def daily_wage_woman : ℝ := sorry

axiom condition_for_men (M : ℝ) : 16 * M * 25 = 14400
axiom condition_for_women (W : ℝ) : 40 * W * 30 = 21600

-- Theorem statement for the ratio of daily wages
theorem ratio_of_daily_wages 
  (M : ℝ) (W : ℝ) 
  (hM : 16 * M * 25 = 14400) 
  (hW : 40 * W * 30 = 21600) :
  M / W = 2 := 
  sorry

end NUMINAMATH_GPT_ratio_of_daily_wages_l1836_183677


namespace NUMINAMATH_GPT_quadratic_translation_l1836_183668

theorem quadratic_translation (b c : ℝ) :
  (∀ x : ℝ, (x^2 + b * x + c = (x - 3)^2 - 2)) →
  b = 4 ∧ c = 6 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_translation_l1836_183668


namespace NUMINAMATH_GPT_three_digit_numbers_eq_11_sum_squares_l1836_183651

theorem three_digit_numbers_eq_11_sum_squares :
  ∃ (N : ℕ), 
    (N = 550 ∨ N = 803) ∧
    (∃ (a b c : ℕ), 
      N = 100 * a + 10 * b + c ∧ 
      100 * a + 10 * b + c = 11 * (a ^ 2 + b ^ 2 + c ^ 2) ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧
      0 ≤ c ∧ c ≤ 9) :=
sorry

end NUMINAMATH_GPT_three_digit_numbers_eq_11_sum_squares_l1836_183651


namespace NUMINAMATH_GPT_snow_white_seven_piles_l1836_183619

def split_pile_action (piles : List ℕ) : Prop :=
  ∃ pile1 pile2, pile1 > 0 ∧ pile2 > 0 ∧ pile1 + pile2 + 1 ∈ piles

theorem snow_white_seven_piles :
  ∃ piles : List ℕ, piles.length = 7 ∧ ∀ pile ∈ piles, pile = 3 :=
sorry

end NUMINAMATH_GPT_snow_white_seven_piles_l1836_183619


namespace NUMINAMATH_GPT_arithmetic_mean_of_sequence_beginning_at_5_l1836_183678

def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

def sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def arithmetic_mean (a d n : ℕ) : ℚ :=
  sequence_sum a d n / n

theorem arithmetic_mean_of_sequence_beginning_at_5 : 
  arithmetic_mean 5 1 60 = 34.5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_sequence_beginning_at_5_l1836_183678


namespace NUMINAMATH_GPT_intersecting_chords_second_length_l1836_183602

theorem intersecting_chords_second_length (a b : ℕ) (k : ℕ) 
  (h_a : a = 12) (h_b : b = 18) (h_ratio : k ^ 2 = (a * b) / 24) 
  (x y : ℕ) (h_x : x = 3 * k) (h_y : y = 8 * k) :
  x + y = 33 :=
by
  sorry

end NUMINAMATH_GPT_intersecting_chords_second_length_l1836_183602


namespace NUMINAMATH_GPT_discriminant_eq_M_l1836_183630

theorem discriminant_eq_M (a b c x0 : ℝ) (h1: a ≠ 0) (h2: a * x0^2 + b * x0 + c = 0) :
  (b^2 - 4 * a * c) = (2 * a * x0 + b)^2 :=
by
  sorry

end NUMINAMATH_GPT_discriminant_eq_M_l1836_183630


namespace NUMINAMATH_GPT_determine_truth_tellers_min_questions_to_determine_truth_tellers_l1836_183614

variables (n k : ℕ)
variables (h_n_pos : 0 < n) (h_k_pos : 0 < k) (h_k_le_n : k ≤ n)

theorem determine_truth_tellers (h : k % 2 = 0) : 
  ∃ m : ℕ, m = n :=
  sorry

theorem min_questions_to_determine_truth_tellers :
  ∃ m : ℕ, m = n :=
  sorry

end NUMINAMATH_GPT_determine_truth_tellers_min_questions_to_determine_truth_tellers_l1836_183614


namespace NUMINAMATH_GPT_simplify_fraction_l1836_183661

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2 + 1) = 16250 / 601 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1836_183661


namespace NUMINAMATH_GPT_surface_area_sphere_dihedral_l1836_183685

open Real

theorem surface_area_sphere_dihedral (R a : ℝ) (hR : 0 < R) (haR : 0 < a ∧ a < R) (α : ℝ) :
  2 * R^2 * arccos ((R * cos α) / sqrt (R^2 - a^2 * sin α^2)) 
  - 2 * R * a * sin α * arccos ((a * cos α) / sqrt (R^2 - a^2 * sin α^2)) = sorry :=
sorry

end NUMINAMATH_GPT_surface_area_sphere_dihedral_l1836_183685


namespace NUMINAMATH_GPT_cody_steps_away_from_goal_l1836_183660

def steps_in_week (daily_steps : ℕ) : ℕ :=
  daily_steps * 7

def total_steps_in_4_weeks (initial_steps : ℕ) : ℕ :=
  steps_in_week initial_steps +
  steps_in_week (initial_steps + 1000) +
  steps_in_week (initial_steps + 2000) +
  steps_in_week (initial_steps + 3000)

theorem cody_steps_away_from_goal :
  let goal := 100000
  let initial_daily_steps := 1000
  let total_steps := total_steps_in_4_weeks initial_daily_steps
  goal - total_steps = 30000 :=
by
  sorry

end NUMINAMATH_GPT_cody_steps_away_from_goal_l1836_183660


namespace NUMINAMATH_GPT_total_value_l1836_183662

/-- 
The total value of the item V can be determined based on the given conditions.
- The merchant paid an import tax of $109.90.
- The tax rate is 7%.
- The tax is only on the portion of the value above $1000.

Given these conditions, prove that the total value V is 2567.
-/
theorem total_value {V : ℝ} (h1 : 0.07 * (V - 1000) = 109.90) : V = 2567 :=
by
  sorry

end NUMINAMATH_GPT_total_value_l1836_183662


namespace NUMINAMATH_GPT_divisor_of_930_l1836_183695

theorem divisor_of_930 : ∃ d > 1, d ∣ 930 ∧ ∀ e, e ∣ 930 → e > 1 → d ≤ e :=
by
  sorry

end NUMINAMATH_GPT_divisor_of_930_l1836_183695


namespace NUMINAMATH_GPT_train_car_speed_ratio_l1836_183633

theorem train_car_speed_ratio
  (distance_bus : ℕ) (time_bus : ℕ) (distance_car : ℕ) (time_car : ℕ)
  (speed_bus := distance_bus / time_bus)
  (speed_train := speed_bus / (3 / 4))
  (speed_car := distance_car / time_car)
  (ratio := (speed_train : ℚ) / (speed_car : ℚ))
  (h1 : distance_bus = 480)
  (h2 : time_bus = 8)
  (h3 : distance_car = 450)
  (h4 : time_car = 6) :
  ratio = 16 / 15 :=
by
  sorry

end NUMINAMATH_GPT_train_car_speed_ratio_l1836_183633
