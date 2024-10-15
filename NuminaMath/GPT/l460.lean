import Mathlib

namespace NUMINAMATH_GPT_panthers_score_points_l460_46054

theorem panthers_score_points (C P : ℕ) (h1 : C + P = 34) (h2 : C = P + 14) : P = 10 :=
by
  sorry

end NUMINAMATH_GPT_panthers_score_points_l460_46054


namespace NUMINAMATH_GPT_entertainment_expense_percentage_l460_46001

noncomputable def salary : ℝ := 10000
noncomputable def savings : ℝ := 2000
noncomputable def food_expense_percentage : ℝ := 0.40
noncomputable def house_rent_percentage : ℝ := 0.20
noncomputable def conveyance_percentage : ℝ := 0.10

theorem entertainment_expense_percentage :
  let E := (1 - (food_expense_percentage + house_rent_percentage + conveyance_percentage) - (savings / salary))
  E = 0.10 :=
by
  sorry

end NUMINAMATH_GPT_entertainment_expense_percentage_l460_46001


namespace NUMINAMATH_GPT_find_a1_l460_46076

theorem find_a1 (a : ℕ → ℕ) (h1 : a 5 = 14) (h2 : ∀ n, a (n+1) - a n = n + 1) : a 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_l460_46076


namespace NUMINAMATH_GPT_part_I_period_part_I_monotonicity_interval_part_II_range_l460_46062

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

theorem part_I_period : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem part_I_monotonicity_interval (k : ℤ) :
  ∀ x, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + 2 * Real.pi / 3 → f (x + Real.pi) = f x := by
  sorry

theorem part_II_range :
  ∀ x, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 4 → f x ∈ Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_GPT_part_I_period_part_I_monotonicity_interval_part_II_range_l460_46062


namespace NUMINAMATH_GPT_valentines_cards_count_l460_46035

theorem valentines_cards_count (x y : ℕ) (h1 : x * y = x + y + 30) : x * y = 64 :=
by {
    sorry
}

end NUMINAMATH_GPT_valentines_cards_count_l460_46035


namespace NUMINAMATH_GPT_probability_two_balls_red_l460_46075

variables (total_balls red_balls blue_balls green_balls picked_balls : ℕ)

def probability_of_both_red
  (h_total_balls : total_balls = 8)
  (h_red_balls : red_balls = 3)
  (h_blue_balls : blue_balls = 2)
  (h_green_balls : green_balls = 3)
  (h_picked_balls : picked_balls = 2) : ℚ :=
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1))

theorem probability_two_balls_red (h_total_balls : total_balls = 8)
  (h_red_balls : red_balls = 3)
  (h_blue_balls : blue_balls = 2)
  (h_green_balls : green_balls = 3)
  (h_picked_balls : picked_balls = 2)
  (h_prob : probability_of_both_red total_balls red_balls blue_balls green_balls picked_balls 
    h_total_balls h_red_balls h_blue_balls h_green_balls h_picked_balls = 3 / 28) : 
  probability_of_both_red total_balls red_balls blue_balls green_balls picked_balls 
    h_total_balls h_red_balls h_blue_balls h_green_balls h_picked_balls = 3 / 28 := 
sorry

end NUMINAMATH_GPT_probability_two_balls_red_l460_46075


namespace NUMINAMATH_GPT_wood_not_heavier_than_brick_l460_46073

-- Define the weights of the wood and the brick
def block_weight_kg : ℝ := 8
def brick_weight_g : ℝ := 8000

-- Conversion function from kg to g
def kg_to_g (kg : ℝ) : ℝ := kg * 1000

-- State the proof problem
theorem wood_not_heavier_than_brick : ¬ (kg_to_g block_weight_kg > brick_weight_g) :=
by
  -- Begin the proof
  sorry

end NUMINAMATH_GPT_wood_not_heavier_than_brick_l460_46073


namespace NUMINAMATH_GPT_Caleb_pencils_fewer_than_twice_Candy_l460_46007

theorem Caleb_pencils_fewer_than_twice_Candy:
  ∀ (P_Caleb P_Candy: ℕ), 
    P_Candy = 9 → 
    (∃ X, P_Caleb = 2 * P_Candy - X) → 
    P_Caleb + 5 - 10 = 10 → 
    (2 * P_Candy - P_Caleb = 3) :=
by
  intros P_Caleb P_Candy hCandy hCalebLess twCalen
  sorry

end NUMINAMATH_GPT_Caleb_pencils_fewer_than_twice_Candy_l460_46007


namespace NUMINAMATH_GPT_trapezoid_diagonal_is_8sqrt5_trapezoid_leg_is_4sqrt5_l460_46089

namespace Trapezoid

def isosceles_trapezoid (AD BC : ℝ) := 
  AD = 20 ∧ BC = 12

def diagonal (AD BC : ℝ) (AC : ℝ) := 
  isosceles_trapezoid AD BC → AC = 8 * Real.sqrt 5

def leg (AD BC : ℝ) (CD : ℝ) := 
  isosceles_trapezoid AD BC → CD = 4 * Real.sqrt 5

theorem trapezoid_diagonal_is_8sqrt5 (AD BC AC : ℝ) : 
  diagonal AD BC AC :=
by
  intros
  sorry

theorem trapezoid_leg_is_4sqrt5 (AD BC CD : ℝ) : 
  leg AD BC CD :=
by
  intros
  sorry

end Trapezoid

end NUMINAMATH_GPT_trapezoid_diagonal_is_8sqrt5_trapezoid_leg_is_4sqrt5_l460_46089


namespace NUMINAMATH_GPT_carrie_money_left_l460_46077

/-- Carrie was given $91. She bought a sweater for $24, 
    a T-shirt for $6, a pair of shoes for $11,
    and a pair of jeans originally costing $30 with a 25% discount. 
    Prove that she has $27.50 left. -/
theorem carrie_money_left :
  let init_money := 91
  let sweater := 24
  let t_shirt := 6
  let shoes := 11
  let jeans := 30
  let discount := 25 / 100
  let jeans_discounted_price := jeans * (1 - discount)
  let total_cost := sweater + t_shirt + shoes + jeans_discounted_price
  let money_left := init_money - total_cost
  money_left = 27.50 :=
by
  intros
  sorry

end NUMINAMATH_GPT_carrie_money_left_l460_46077


namespace NUMINAMATH_GPT_least_possible_b_l460_46011

def is_prime (n : Nat) : Prop := Nat.Prime n

theorem least_possible_b (a b : Nat) (h1 : is_prime a) (h2 : is_prime b) (h3 : a + 2 * b = 180) (h4 : a > b) : b = 19 :=
by 
  sorry

end NUMINAMATH_GPT_least_possible_b_l460_46011


namespace NUMINAMATH_GPT_blue_pill_cost_l460_46081

theorem blue_pill_cost
  (days : ℕ)
  (total_cost : ℤ)
  (cost_diff : ℤ)
  (daily_cost : ℤ)
  (y : ℤ) : 
  days = 21 →
  total_cost = 966 →
  cost_diff = 2 →
  daily_cost = total_cost / days →
  daily_cost = 46 →
  2 * y - cost_diff = daily_cost →
  y = 24 := 
by
  intros days_eq total_cost_eq cost_diff_eq daily_cost_eq d_cost_eq daily_eq_46;
  sorry

end NUMINAMATH_GPT_blue_pill_cost_l460_46081


namespace NUMINAMATH_GPT_y_increase_when_x_increases_by_9_units_l460_46019

-- Given condition as a definition: when x increases by 3 units, y increases by 7 units.
def x_increases_y_increases (x_increase y_increase : ℕ) : Prop := 
  (x_increase = 3) → (y_increase = 7)

-- Stating the problem: when x increases by 9 units, y increases by how many units?
theorem y_increase_when_x_increases_by_9_units : 
  ∀ (x_increase y_increase : ℕ), x_increases_y_increases x_increase y_increase → ((x_increase * 3 = 9) → (y_increase * 3 = 21)) :=
by
  intros x_increase y_increase cond h
  sorry

end NUMINAMATH_GPT_y_increase_when_x_increases_by_9_units_l460_46019


namespace NUMINAMATH_GPT_max_a_value_l460_46021

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x - 1

theorem max_a_value : 
  ∀ (a : ℝ), 
  (∀ (x : ℝ), x ∈ Set.Icc (1/2) 2 → (a + 1) * x - 1 - Real.log x ≤ 0) → 
  a ≤ 1 - 2 * Real.log 2 := 
by
  sorry

end NUMINAMATH_GPT_max_a_value_l460_46021


namespace NUMINAMATH_GPT_recreation_percentage_this_week_l460_46094

variable (W : ℝ) -- David's last week wages
variable (R_last_week : ℝ) -- Recreation spending last week
variable (W_this_week : ℝ) -- This week's wages
variable (R_this_week : ℝ) -- Recreation spending this week

-- Conditions
def wages_last_week : R_last_week = 0.4 * W := sorry
def wages_this_week : W_this_week = 0.95 * W := sorry
def recreation_spending_this_week : R_this_week = 1.1875 * R_last_week := sorry

-- Theorem to prove
theorem recreation_percentage_this_week :
  (R_this_week / W_this_week) = 0.5 := sorry

end NUMINAMATH_GPT_recreation_percentage_this_week_l460_46094


namespace NUMINAMATH_GPT_oranges_to_put_back_l460_46015

variables (A O x : ℕ)

theorem oranges_to_put_back
    (h1 : 40 * A + 60 * O = 560)
    (h2 : A + O = 10)
    (h3 : (40 * A + 60 * (O - x)) / (10 - x) = 50) : x = 6 := 
sorry

end NUMINAMATH_GPT_oranges_to_put_back_l460_46015


namespace NUMINAMATH_GPT_contrapositive_statement_l460_46031

-- Condition definitions
def P (x : ℝ) := x^2 < 1
def Q (x : ℝ) := -1 < x ∧ x < 1
def not_Q (x : ℝ) := x ≤ -1 ∨ x ≥ 1
def not_P (x : ℝ) := x^2 ≥ 1

theorem contrapositive_statement (x : ℝ) : (x ≤ -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_statement_l460_46031


namespace NUMINAMATH_GPT_opposite_number_subtraction_l460_46072

variable (a b : ℝ)

theorem opposite_number_subtraction : -(a - b) = b - a := 
sorry

end NUMINAMATH_GPT_opposite_number_subtraction_l460_46072


namespace NUMINAMATH_GPT_percentage_decrease_l460_46006

theorem percentage_decrease (a b p : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (h_ratio : a / b = 4 / 5) 
    (h_x : ∃ x, x = a * 1.25)
    (h_m : ∃ m, m = b * (1 - p / 100))
    (h_mx : ∃ m x, (m / x = 0.2)) :
        (p = 80) :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l460_46006


namespace NUMINAMATH_GPT_problem_l460_46040

theorem problem (a b : ℝ) (h1 : |a - 2| + (b + 1)^2 = 0) : a - b = 3 := by
  sorry

end NUMINAMATH_GPT_problem_l460_46040


namespace NUMINAMATH_GPT_hyperbola_center_l460_46028

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (h1 : x1 = 6) (h2 : y1 = 3) (h3 : x2 = 10) (h4 : y2 = 7) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (8, 5) :=
by
  rw [h1, h2, h3, h4]
  simp
  -- Proof steps demonstrating the calculation
  -- simplify the arithmetic expressions
  sorry

end NUMINAMATH_GPT_hyperbola_center_l460_46028


namespace NUMINAMATH_GPT_find_number_of_girls_in_class_l460_46029

variable (G : ℕ)

def number_of_ways_to_select_two_boys (n : ℕ) : ℕ := Nat.choose n 2

theorem find_number_of_girls_in_class 
  (boys : ℕ := 13) 
  (ways_to_select_students : ℕ := 780) 
  (ways_to_select_two_boys : ℕ := number_of_ways_to_select_two_boys boys) :
  G * ways_to_select_two_boys = ways_to_select_students → G = 10 := 
by
  sorry

end NUMINAMATH_GPT_find_number_of_girls_in_class_l460_46029


namespace NUMINAMATH_GPT_arithmetic_sequence_binomial_l460_46022

theorem arithmetic_sequence_binomial {n k u : ℕ} (h₁ : u ≥ 3)
    (h₂ : n = u^2 - 2)
    (h₃ : k = Nat.choose u 2 - 1 ∨ k = Nat.choose (u + 1) 2 - 1)
    : (Nat.choose n (k - 1)) - 2 * (Nat.choose n k) + (Nat.choose n (k + 1)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_binomial_l460_46022


namespace NUMINAMATH_GPT_election_winner_votes_difference_l460_46002

theorem election_winner_votes_difference (V : ℝ) (h1 : 0.62 * V = 1054) : 0.24 * V = 408 :=
by
  sorry

end NUMINAMATH_GPT_election_winner_votes_difference_l460_46002


namespace NUMINAMATH_GPT_max_digit_product_l460_46043

theorem max_digit_product (N : ℕ) (digits : List ℕ) (h1 : 0 < N) (h2 : digits.sum = 23) (h3 : digits.prod < 433) : 
  digits.prod ≤ 432 :=
sorry

end NUMINAMATH_GPT_max_digit_product_l460_46043


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l460_46044

/-- In one hour, a boat goes 9 km along the stream and 5 km against the stream.
Prove that the speed of the boat in still water is 7 km/hr. -/
theorem boat_speed_in_still_water (B S : ℝ) 
  (h1 : B + S = 9) 
  (h2 : B - S = 5) : 
  B = 7 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l460_46044


namespace NUMINAMATH_GPT_calculate_fraction_l460_46042

theorem calculate_fraction (x y : ℚ) (h1 : x = 5 / 6) (h2 : y = 6 / 5) : (1 / 3) * x^8 * y^9 = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l460_46042


namespace NUMINAMATH_GPT_positive_integers_of_m_n_l460_46098

theorem positive_integers_of_m_n (m n : ℕ) (p : ℕ) (a : ℕ) (k : ℕ) (h_m_ge_2 : m ≥ 2) (h_n_ge_2 : n ≥ 2) 
  (h_prime_q : Prime (m + 1)) (h_4k_1 : m + 1 = 4 * k - 1) 
  (h_eq : (m ^ (2 ^ n - 1) - 1) / (m - 1) = m ^ n + p ^ a) : 
  (m, n) = (p - 1, 2) ∧ Prime p ∧ ∃k, p = 4 * k - 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_positive_integers_of_m_n_l460_46098


namespace NUMINAMATH_GPT_car_repair_cost_l460_46051

noncomputable def total_cost (first_mechanic_rate: ℝ) (first_mechanic_hours: ℕ) 
    (first_mechanic_days: ℕ) (second_mechanic_rate: ℝ) 
    (second_mechanic_hours: ℕ) (second_mechanic_days: ℕ) 
    (discount_first: ℝ) (discount_second: ℝ) 
    (parts_cost: ℝ) (sales_tax_rate: ℝ): ℝ :=
  let first_mechanic_cost := first_mechanic_rate * first_mechanic_hours * first_mechanic_days
  let second_mechanic_cost := second_mechanic_rate * second_mechanic_hours * second_mechanic_days
  let first_mechanic_discounted := first_mechanic_cost - (discount_first * first_mechanic_cost)
  let second_mechanic_discounted := second_mechanic_cost - (discount_second * second_mechanic_cost)
  let total_before_tax := first_mechanic_discounted + second_mechanic_discounted + parts_cost
  let sales_tax := sales_tax_rate * total_before_tax
  total_before_tax + sales_tax

theorem car_repair_cost :
  total_cost 60 8 14 75 6 10 0.15 0.10 3200 0.07 = 13869.34 := by
  sorry

end NUMINAMATH_GPT_car_repair_cost_l460_46051


namespace NUMINAMATH_GPT_vertices_of_equilateral_triangle_l460_46091

noncomputable def a : ℝ := 52 / 3
noncomputable def b : ℝ := -13 / 3 - 15 * Real.sqrt 3 / 2

theorem vertices_of_equilateral_triangle (a b : ℝ)
  (h₀ : (0, 0) = (0, 0))
  (h₁ : (a, 15) = (52 / 3, 15))
  (h₂ : (b, 41) = (-13 / 3 - 15 * Real.sqrt 3 / 2, 41)) :
  a * b = -676 / 9 := 
by
  sorry

end NUMINAMATH_GPT_vertices_of_equilateral_triangle_l460_46091


namespace NUMINAMATH_GPT_coplanar_points_l460_46050

theorem coplanar_points (a : ℝ) :
  ∀ (V : ℝ), V = 2 + a^3 → V = 0 → a = -((2:ℝ)^(1/3)) :=
by
  sorry

end NUMINAMATH_GPT_coplanar_points_l460_46050


namespace NUMINAMATH_GPT_problem_conditions_and_inequalities_l460_46041

open Real

theorem problem_conditions_and_inequalities (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + 2 * b = a * b) :
  (a + 2 * b ≥ 8) ∧ (2 * a + b ≥ 9) ∧ (a ^ 2 + 4 * b ^ 2 + 5 * a * b ≥ 72) ∧ ¬(logb 2 a + logb 2 b < 3) :=
by
  sorry

end NUMINAMATH_GPT_problem_conditions_and_inequalities_l460_46041


namespace NUMINAMATH_GPT_exponent_multiplication_l460_46013

theorem exponent_multiplication (m n : ℕ) (h : m + n = 3) : 2^m * 2^n = 8 := 
by
  sorry

end NUMINAMATH_GPT_exponent_multiplication_l460_46013


namespace NUMINAMATH_GPT_walking_speed_proof_l460_46060

-- Definitions based on the problem's conditions
def rest_time_per_period : ℕ := 5
def distance_per_rest : ℕ := 10
def total_distance : ℕ := 50
def total_time : ℕ := 320

-- The man's walking speed
def walking_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The main statement to be proved
theorem walking_speed_proof : 
  walking_speed total_distance ((total_time - ((total_distance / distance_per_rest) * rest_time_per_period)) / 60) = 10 := 
by
  sorry

end NUMINAMATH_GPT_walking_speed_proof_l460_46060


namespace NUMINAMATH_GPT_car_speed_in_kmph_l460_46061

def speed_mps : ℝ := 10  -- The speed of the car in meters per second
def conversion_factor : ℝ := 3.6  -- The conversion factor from m/s to km/h

theorem car_speed_in_kmph : speed_mps * conversion_factor = 36 := 
by
  sorry

end NUMINAMATH_GPT_car_speed_in_kmph_l460_46061


namespace NUMINAMATH_GPT_fifth_friend_paid_13_l460_46092

noncomputable def fifth_friend_payment (a b c d e : ℝ) : Prop :=
a = (1/3) * (b + c + d + e) ∧
b = (1/4) * (a + c + d + e) ∧
c = (1/5) * (a + b + d + e) ∧
a + b + c + d + e = 120 ∧
e = 13

theorem fifth_friend_paid_13 : 
  ∃ (a b c d e : ℝ), fifth_friend_payment a b c d e := 
sorry

end NUMINAMATH_GPT_fifth_friend_paid_13_l460_46092


namespace NUMINAMATH_GPT_gas_cost_problem_l460_46016

theorem gas_cost_problem (x : ℝ) (h : x / 4 - 15 = x / 7) : x = 140 :=
sorry

end NUMINAMATH_GPT_gas_cost_problem_l460_46016


namespace NUMINAMATH_GPT_cos_sin_equation_solution_l460_46037

noncomputable def solve_cos_sin_equation (x : ℝ) (n : ℤ) : Prop :=
  let lhs := (Real.cos x) / (Real.sqrt 3)
  let rhs := Real.sqrt ((1 - (Real.cos (2*x)) - 2 * (Real.sin x)^3) / (6 * Real.sin x - 2))
  (lhs = rhs) ∧ (Real.cos x ≥ 0)

theorem cos_sin_equation_solution:
  (∃ (x : ℝ) (n : ℤ), solve_cos_sin_equation x n) ↔ 
  ∃ (n : ℤ), (x = (π / 2) + 2 * π * n) ∨ (x = (π / 6) + 2 * π * n) :=
by
  sorry

end NUMINAMATH_GPT_cos_sin_equation_solution_l460_46037


namespace NUMINAMATH_GPT_find_fraction_l460_46093

theorem find_fraction (f : ℝ) (h₁ : f * 50.0 - 4 = 6) : f = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l460_46093


namespace NUMINAMATH_GPT_dropping_more_than_eating_l460_46099

theorem dropping_more_than_eating (n : ℕ) : n = 20 → (n * (n + 1)) / 2 > 10 * n := by
  intros h
  rw [h]
  sorry

end NUMINAMATH_GPT_dropping_more_than_eating_l460_46099


namespace NUMINAMATH_GPT_direct_proportion_solution_l460_46085

theorem direct_proportion_solution (m : ℝ) (h1 : m + 3 ≠ 0) (h2 : m^2 - 8 = 1) : m = 3 :=
sorry

end NUMINAMATH_GPT_direct_proportion_solution_l460_46085


namespace NUMINAMATH_GPT_parametric_to_ordinary_l460_46047

theorem parametric_to_ordinary (θ : ℝ) (x y : ℝ) : 
  x = Real.cos θ ^ 2 →
  y = 2 * Real.sin θ ^ 2 →
  (x ∈ Set.Icc 0 1) → 
  2 * x + y - 2 = 0 :=
by
  intros hx hy h_range
  sorry

end NUMINAMATH_GPT_parametric_to_ordinary_l460_46047


namespace NUMINAMATH_GPT_alex_silver_tokens_count_l460_46095

-- Conditions
def initial_red_tokens := 90
def initial_blue_tokens := 80

def red_exchange (x : ℕ) (y : ℕ) : ℕ := 90 - 3 * x + y
def blue_exchange (x : ℕ) (y : ℕ) : ℕ := 80 + 2 * x - 4 * y

-- Boundaries where exchanges stop
def red_bound (x : ℕ) (y : ℕ) : Prop := red_exchange x y < 3
def blue_bound (x : ℕ) (y : ℕ) : Prop := blue_exchange x y < 4

-- Proof statement
theorem alex_silver_tokens_count (x y : ℕ) :
    red_bound x y → blue_bound x y → (x + y) = 52 :=
    by
    sorry

end NUMINAMATH_GPT_alex_silver_tokens_count_l460_46095


namespace NUMINAMATH_GPT_brownies_maximum_l460_46033

theorem brownies_maximum (m n : ℕ) (h1 : (m - 2) * (n - 2) = 2 * (2 * m + 2 * n - 4)) :
  m * n ≤ 144 :=
sorry

end NUMINAMATH_GPT_brownies_maximum_l460_46033


namespace NUMINAMATH_GPT_cos_15_degree_l460_46004

theorem cos_15_degree : 
  let d15 := 15 * Real.pi / 180
  let d45 := 45 * Real.pi / 180
  let d30 := 30 * Real.pi / 180
  Real.cos d15 = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_GPT_cos_15_degree_l460_46004


namespace NUMINAMATH_GPT_total_spent_on_entertainment_l460_46068

def cost_of_computer_game : ℕ := 66
def cost_of_one_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3

theorem total_spent_on_entertainment : cost_of_computer_game + cost_of_one_movie_ticket * number_of_movie_tickets = 102 := 
by sorry

end NUMINAMATH_GPT_total_spent_on_entertainment_l460_46068


namespace NUMINAMATH_GPT_sides_ratio_of_arithmetic_sequence_l460_46084

theorem sides_ratio_of_arithmetic_sequence (A B C : ℝ) (a b c : ℝ) 
  (h_arith_sequence : (A = B - (B - C)) ∧ (B = C + (C - A))) 
  (h_angle_B : B = 60)  
  (h_cosine_rule : a^2 + c^2 - b^2 = 2 * a * c * (Real.cos B)) :
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :=
sorry

end NUMINAMATH_GPT_sides_ratio_of_arithmetic_sequence_l460_46084


namespace NUMINAMATH_GPT_robert_salary_loss_l460_46012

theorem robert_salary_loss (S : ℝ) : 
  let decreased_salary := S - 0.3 * S
  let increased_salary := decreased_salary + 0.3 * decreased_salary
  100 * (1 - increased_salary / S) = 9 :=
by
  let decreased_salary := S - 0.3 * S
  let increased_salary := decreased_salary + 0.3 * decreased_salary
  sorry

end NUMINAMATH_GPT_robert_salary_loss_l460_46012


namespace NUMINAMATH_GPT_planes_count_l460_46070

-- Define the conditions as given in the problem.
def total_wings : ℕ := 90
def wings_per_plane : ℕ := 2

-- Define the number of planes calculation based on conditions.
def number_of_planes : ℕ := total_wings / wings_per_plane

-- Prove that the number of planes is 45.
theorem planes_count : number_of_planes = 45 :=
by 
  -- The proof steps are omitted as specified.
  sorry

end NUMINAMATH_GPT_planes_count_l460_46070


namespace NUMINAMATH_GPT_min_value_of_x3y2z_l460_46030

noncomputable def min_value_of_polynomial (x y z : ℝ) : ℝ :=
  x^3 * y^2 * z

theorem min_value_of_x3y2z
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 1 / x + 1 / y + 1 / z = 9) :
  min_value_of_polynomial x y z = 1 / 46656 :=
sorry

end NUMINAMATH_GPT_min_value_of_x3y2z_l460_46030


namespace NUMINAMATH_GPT_find_added_number_l460_46066

def original_number : ℕ := 5
def doubled : ℕ := 2 * original_number
def resultant (added : ℕ) : ℕ := 3 * (doubled + added)
def final_result : ℕ := 57

theorem find_added_number (added : ℕ) (h : resultant added = final_result) : added = 9 :=
sorry

end NUMINAMATH_GPT_find_added_number_l460_46066


namespace NUMINAMATH_GPT_time_to_count_envelopes_l460_46074

theorem time_to_count_envelopes (r : ℕ) : (r / 10 = 1) → (r * 60 / r = 60) ∧ (r * 90 / r = 90) :=
by sorry

end NUMINAMATH_GPT_time_to_count_envelopes_l460_46074


namespace NUMINAMATH_GPT_minimal_difference_big_small_sum_l460_46000

theorem minimal_difference_big_small_sum :
  ∀ (N : ℕ), N > 0 → ∃ (S : ℕ), 
  S = (N * (N - 1) * (2 * N + 5)) / 6 :=
  by 
    sorry

end NUMINAMATH_GPT_minimal_difference_big_small_sum_l460_46000


namespace NUMINAMATH_GPT_center_of_circumcircle_lies_on_AK_l460_46071

variable {α β γ : Real} -- Angles in triangle ABC
variable (A B C L H K O : Point) -- Points in the configuration
variable (circumcircle_ABC : TriangularCircumcircle A B C) -- Circumcircle of triangle ABC

-- Definitions based on the given conditions
variable (is_angle_bisector : angle_bisector A B C L)
variable (is_height : height_from_point_to_line B A L H)
variable (intersects_circle_at_K : intersects_circumcircle A B L K circumcircle_ABC)
variable (is_circumcenter : circumcenter A B C O circumcircle_ABC)

theorem center_of_circumcircle_lies_on_AK
  (h_angle_bisector : is_angle_bisector)
  (h_height : is_height)
  (h_intersects_circle_at_K : intersects_circle_at_K)
  (h_circumcenter : is_circumcenter) 
    : lies_on_line O A K := 
sorry -- Proof is omitted

end NUMINAMATH_GPT_center_of_circumcircle_lies_on_AK_l460_46071


namespace NUMINAMATH_GPT_problem_I_problem_II_l460_46017

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x + (4 / x) - m| + m

-- Proof problem (I): When m = 0, find the minimum value of the function f(x).
theorem problem_I : ∀ x : ℝ, (f x 0) ≥ 4 := by
  sorry

-- Proof problem (II): If the function f(x) ≤ 5 for all x ∈ [1, 4], find the range of m.
theorem problem_II (m : ℝ) : (∀ x : ℝ, 1 ≤ x → x ≤ 4 → f x m ≤ 5) ↔ m ≤ 9 / 2 := by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l460_46017


namespace NUMINAMATH_GPT_find_percentage_l460_46069

theorem find_percentage (x p : ℝ) (h1 : x = 840) (h2 : 0.25 * x + 15 = p / 100 * 1500) : p = 15 := 
by
  sorry

end NUMINAMATH_GPT_find_percentage_l460_46069


namespace NUMINAMATH_GPT_greatest_possible_value_of_a_l460_46067

theorem greatest_possible_value_of_a 
  (x a : ℤ)
  (h : x^2 + a * x = -21)
  (ha_pos : 0 < a)
  (hx_int : x ∈ [-21, -7, -3, -1].toFinset): 
  a ≤ 22 := sorry

end NUMINAMATH_GPT_greatest_possible_value_of_a_l460_46067


namespace NUMINAMATH_GPT_remainder_problem_l460_46059

theorem remainder_problem (x y z : ℤ) 
  (hx : x % 15 = 11) (hy : y % 15 = 13) (hz : z % 15 = 14) : 
  (y + z - x) % 15 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_problem_l460_46059


namespace NUMINAMATH_GPT_total_distance_traveled_l460_46010

-- Definitions of distances in km
def ZX : ℝ := 4000
def XY : ℝ := 5000
def YZ : ℝ := (XY^2 - ZX^2)^(1/2)

-- Prove the total distance traveled
theorem total_distance_traveled :
  XY + YZ + ZX = 11500 := by
  have h1 : ZX = 4000 := rfl
  have h2 : XY = 5000 := rfl
  have h3 : YZ = (5000^2 - 4000^2)^(1/2) := rfl
  -- Continue the proof showing the calculation of each step
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l460_46010


namespace NUMINAMATH_GPT_minimum_value_l460_46039

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 6) :
  37.5 ≤ (9 / x + 25 / y + 49 / z) :=
sorry

end NUMINAMATH_GPT_minimum_value_l460_46039


namespace NUMINAMATH_GPT_highest_score_of_batsman_l460_46088

theorem highest_score_of_batsman
  (avg : ℕ)
  (inn : ℕ)
  (diff_high_low : ℕ)
  (sum_high_low : ℕ)
  (avg_excl : ℕ)
  (inn_excl : ℕ)
  (h_l_avg : avg = 60)
  (h_l_inn : inn = 46)
  (h_l_diff : diff_high_low = 140)
  (h_l_sum : sum_high_low = 208)
  (h_l_avg_excl : avg_excl = 58)
  (h_l_inn_excl : inn_excl = 44) :
  ∃ H L : ℕ, H = 174 :=
by
  sorry

end NUMINAMATH_GPT_highest_score_of_batsman_l460_46088


namespace NUMINAMATH_GPT_f_20_value_l460_46086

noncomputable def f (n : ℕ) : ℚ := sorry

axiom f_initial : f 1 = 3 / 2
axiom f_eq : ∀ x y : ℕ, 
  f (x + y) = (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2

theorem f_20_value : f 20 = 4305 := 
by {
  sorry 
}

end NUMINAMATH_GPT_f_20_value_l460_46086


namespace NUMINAMATH_GPT_game_promises_total_hours_l460_46057

open Real

noncomputable def total_gameplay_hours (T : ℝ) : Prop :=
  let boring_gameplay := 0.80 * T
  let enjoyable_gameplay := 0.20 * T
  let expansion_hours := 30
  (enjoyable_gameplay + expansion_hours = 50) → (T = 100)

theorem game_promises_total_hours (T : ℝ) : total_gameplay_hours T :=
  sorry

end NUMINAMATH_GPT_game_promises_total_hours_l460_46057


namespace NUMINAMATH_GPT_find_g_53_l460_46087

variable (g : ℝ → ℝ)

axiom functional_eq (x y : ℝ) : g (x * y) = y * g x
axiom g_one : g 1 = 10

theorem find_g_53 : g 53 = 530 :=
by
  sorry

end NUMINAMATH_GPT_find_g_53_l460_46087


namespace NUMINAMATH_GPT_turtle_feeding_cost_l460_46078

theorem turtle_feeding_cost (total_weight_pounds : ℝ) (food_per_half_pound : ℝ)
  (jar_food_ounces : ℝ) (jar_cost_dollars : ℝ) (total_cost : ℝ) : 
  total_weight_pounds = 30 →
  food_per_half_pound = 1 →
  jar_food_ounces = 15 →
  jar_cost_dollars = 2 →
  total_cost = 8 :=
by
  intros h_weight h_food h_jar_ounces h_jar_cost
  sorry

end NUMINAMATH_GPT_turtle_feeding_cost_l460_46078


namespace NUMINAMATH_GPT_product_of_numbers_l460_46009

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 185) : x * y = 26 :=
sorry

end NUMINAMATH_GPT_product_of_numbers_l460_46009


namespace NUMINAMATH_GPT_inscribed_sphere_radius_l460_46096

theorem inscribed_sphere_radius {V S1 S2 S3 S4 R : ℝ} :
  (1/3) * R * (S1 + S2 + S3 + S4) = V → 
  R = 3 * V / (S1 + S2 + S3 + S4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_l460_46096


namespace NUMINAMATH_GPT_even_function_l460_46082

noncomputable def f : ℝ → ℝ :=
sorry

theorem even_function (f : ℝ → ℝ) (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, -1 ≤ x ∧ x ≤ 0 → f x = x - 1) : f (1/2) = -3/2 :=
sorry

end NUMINAMATH_GPT_even_function_l460_46082


namespace NUMINAMATH_GPT_dandelion_average_l460_46045

theorem dandelion_average :
  let Billy_initial := 36
  let George_initial := Billy_initial / 3
  let Billy_total := Billy_initial + 10
  let George_total := George_initial + 10
  let total := Billy_total + George_total
  let average := total / 2
  average = 34 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_dandelion_average_l460_46045


namespace NUMINAMATH_GPT_right_triangle_condition_l460_46056

theorem right_triangle_condition (a b c : ℝ) : (a^2 = b^2 - c^2) → (∃ B : ℝ, B = 90) := 
sorry

end NUMINAMATH_GPT_right_triangle_condition_l460_46056


namespace NUMINAMATH_GPT_find_positive_real_solution_l460_46052

theorem find_positive_real_solution (x : ℝ) : 
  0 < x ∧ (1 / 2 * (4 * x^2 - 1) = (x^2 - 60 * x - 20) * (x^2 + 30 * x + 10)) ↔ 
  (x = 30 + Real.sqrt 919 ∨ x = -15 + Real.sqrt 216 ∧ 0 < -15 + Real.sqrt 216) :=
by sorry

end NUMINAMATH_GPT_find_positive_real_solution_l460_46052


namespace NUMINAMATH_GPT_solve_xy_l460_46025

variable (x y : ℝ)

-- Given conditions
def condition1 : Prop := y = (2 / 3) * x
def condition2 : Prop := 0.4 * x = (1 / 3) * y + 110

-- Statement we want to prove
theorem solve_xy (h1 : condition1 x y) (h2 : condition2 x y) : x = 618.75 ∧ y = 412.5 :=
  by sorry

end NUMINAMATH_GPT_solve_xy_l460_46025


namespace NUMINAMATH_GPT_A_B_days_together_l460_46055

variable (W : ℝ) -- total work
variable (x : ℝ) -- days A and B worked together
variable (A_B_rate : ℝ) -- combined work rate of A and B
variable (A_rate : ℝ) -- work rate of A
variable (B_days : ℝ) -- days A worked alone after B left

-- Conditions:
axiom condition1 : A_B_rate = W / 40
axiom condition2 : A_rate = W / 80
axiom condition3 : B_days = 6
axiom condition4 : (x * A_B_rate + B_days * A_rate = W)

-- We want to prove that x = 37:
theorem A_B_days_together : x = 37 :=
by
  sorry

end NUMINAMATH_GPT_A_B_days_together_l460_46055


namespace NUMINAMATH_GPT_people_in_each_column_l460_46080

theorem people_in_each_column
  (P : ℕ)
  (x : ℕ)
  (h1 : P = 16 * x)
  (h2 : P = 12 * 40) :
  x = 30 :=
sorry

end NUMINAMATH_GPT_people_in_each_column_l460_46080


namespace NUMINAMATH_GPT_smallest_norm_value_l460_46058

theorem smallest_norm_value (w : ℝ × ℝ)
  (h : ‖(w.1 + 4, w.2 + 2)‖ = 10) :
  ‖w‖ = 10 - 2*Real.sqrt 5 := sorry

end NUMINAMATH_GPT_smallest_norm_value_l460_46058


namespace NUMINAMATH_GPT_little_john_initial_money_l460_46026

theorem little_john_initial_money :
  let spent := 1.05
  let given_to_each_friend := 2 * 1.00
  let total_spent := spent + given_to_each_friend
  let left := 2.05
  total_spent + left = 5.10 :=
by
  let spent := 1.05
  let given_to_each_friend := 2 * 1.00
  let total_spent := spent + given_to_each_friend
  let left := 2.05
  show total_spent + left = 5.10
  sorry

end NUMINAMATH_GPT_little_john_initial_money_l460_46026


namespace NUMINAMATH_GPT_cost_of_fencing_field_l460_46034

def ratio (a b : ℕ) : Prop := ∃ k : ℕ, (b = k * a)

def assume_fields : Prop :=
  ∃ (x : ℚ), (ratio 3 4) ∧ (3 * 4 * x^2 = 9408) ∧ (0.25 > 0)

theorem cost_of_fencing_field :
  assume_fields → 98 = 98 := by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_field_l460_46034


namespace NUMINAMATH_GPT_triangle_angles_l460_46027

theorem triangle_angles (r_a r_b r_c R : ℝ)
    (h1 : r_a + r_b = 3 * R)
    (h2 : r_b + r_c = 2 * R) :
    ∃ (A B C : ℝ), A = 30 ∧ B = 60 ∧ C = 90 :=
sorry

end NUMINAMATH_GPT_triangle_angles_l460_46027


namespace NUMINAMATH_GPT_bus_speed_l460_46003

theorem bus_speed (d t : ℕ) (h1 : d = 201) (h2 : t = 3) : d / t = 67 :=
by sorry

end NUMINAMATH_GPT_bus_speed_l460_46003


namespace NUMINAMATH_GPT_rectangle_length_l460_46097

theorem rectangle_length (s w : ℝ) (A : ℝ) (L : ℝ) (h1 : s = 9) (h2 : w = 3) (h3 : A = s * s) (h4 : A = w * L) : L = 27 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l460_46097


namespace NUMINAMATH_GPT_thirty_six_hundredths_is_decimal_l460_46008

namespace thirty_six_hundredths

-- Define the fraction representation of thirty-six hundredths
def fraction_thirty_six_hundredths : ℚ := 36 / 100

-- The problem is to prove that this fraction is equal to 0.36 in decimal form
theorem thirty_six_hundredths_is_decimal : fraction_thirty_six_hundredths = 0.36 := 
sorry

end thirty_six_hundredths

end NUMINAMATH_GPT_thirty_six_hundredths_is_decimal_l460_46008


namespace NUMINAMATH_GPT_time_for_a_alone_l460_46005

theorem time_for_a_alone
  (b_work_time : ℕ := 20)
  (c_work_time : ℕ := 45)
  (together_work_time : ℕ := 72 / 10) :
  ∃ (a_work_time : ℕ), a_work_time = 15 :=
by
  sorry

end NUMINAMATH_GPT_time_for_a_alone_l460_46005


namespace NUMINAMATH_GPT_train_boxcars_capacity_l460_46036

theorem train_boxcars_capacity :
  let red_boxcars := 3
  let blue_boxcars := 4
  let black_boxcars := 7
  let black_capacity := 4000
  let blue_capacity := black_capacity * 2
  let red_capacity := blue_capacity * 3
  (black_boxcars * black_capacity) + (blue_boxcars * blue_capacity) + (red_boxcars * red_capacity) = 132000 := by
  sorry

end NUMINAMATH_GPT_train_boxcars_capacity_l460_46036


namespace NUMINAMATH_GPT_find_x_l460_46090

def operation_eur (x y : ℕ) : ℕ := 3 * x * y

theorem find_x (y x : ℕ) (h1 : y = 3) (h2 : operation_eur y (operation_eur x 5) = 540) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l460_46090


namespace NUMINAMATH_GPT_alice_ride_average_speed_l460_46079

theorem alice_ride_average_speed
    (d1 d2 : ℝ) 
    (s1 s2 : ℝ)
    (h_d1 : d1 = 40)
    (h_d2 : d2 = 20)
    (h_s1 : s1 = 8)
    (h_s2 : s2 = 40) :
    (d1 + d2) / (d1 / s1 + d2 / s2) = 10.909 :=
by
  simp [h_d1, h_d2, h_s1, h_s2]
  norm_num
  sorry

end NUMINAMATH_GPT_alice_ride_average_speed_l460_46079


namespace NUMINAMATH_GPT_multiple_of_5_l460_46020

theorem multiple_of_5 (a : ℤ) (h : ¬ (5 ∣ a)) : 5 ∣ (a^12 - 1) :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_5_l460_46020


namespace NUMINAMATH_GPT_probability_of_sunglasses_given_caps_l460_46038

theorem probability_of_sunglasses_given_caps
  (s c sc : ℕ) 
  (h₀ : s = 60) 
  (h₁ : c = 40)
  (h₂ : sc = 20)
  (h₃ : sc = 1 / 3 * s) : 
  (sc / c) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_sunglasses_given_caps_l460_46038


namespace NUMINAMATH_GPT_line_parallel_not_passing_through_point_l460_46046

noncomputable def point_outside_line (A B C x0 y0 : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (A * x0 + B * y0 + C = k)

theorem line_parallel_not_passing_through_point 
  (A B C x0 y0 : ℝ) (h : point_outside_line A B C x0 y0) :
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x y : ℝ, Ax + By + C + k = 0 → Ax_0 + By_0 + C + k ≠ 0) :=
sorry

end NUMINAMATH_GPT_line_parallel_not_passing_through_point_l460_46046


namespace NUMINAMATH_GPT_intersection_with_complement_l460_46064

-- Define the universal set U, sets A and B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

-- Define the complement of B with respect to U
def complement_B : Set ℕ := { x | x ∈ U ∧ x ∉ B }

-- The equivalent proof problem in Lean 4
theorem intersection_with_complement :
  A ∩ complement_B = {0, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_with_complement_l460_46064


namespace NUMINAMATH_GPT_shirley_sold_10_boxes_l460_46053

variable (cases boxes_per_case : ℕ)

-- Define the conditions
def number_of_cases := 5
def boxes_in_each_case := 2

-- Prove the total number of boxes is 10
theorem shirley_sold_10_boxes (H1 : cases = number_of_cases) (H2 : boxes_per_case = boxes_in_each_case) :
  cases * boxes_per_case = 10 := by
  sorry

end NUMINAMATH_GPT_shirley_sold_10_boxes_l460_46053


namespace NUMINAMATH_GPT_range_is_correct_l460_46065

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x

def domain : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

def range_of_function : Set ℝ := {y | ∃ x ∈ domain, quadratic_function x = y}

theorem range_is_correct : range_of_function = Set.Icc (-4) 21 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_is_correct_l460_46065


namespace NUMINAMATH_GPT_number_of_candies_bought_on_Tuesday_l460_46049

theorem number_of_candies_bought_on_Tuesday (T : ℕ) 
  (thursday_candies : ℕ := 5) 
  (friday_candies : ℕ := 2) 
  (candies_left : ℕ := 4) 
  (candies_eaten : ℕ := 6) 
  (total_initial_candies : T + thursday_candies + friday_candies = candies_left + candies_eaten) 
  : T = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_candies_bought_on_Tuesday_l460_46049


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l460_46023

theorem arithmetic_seq_sum {a_n : ℕ → ℤ} {d : ℤ} (S_n : ℕ → ℤ) :
  (∀ n : ℕ, S_n n = -(n * n)) →
  (∃ d, d = -2 ∧ ∀ n, a_n n = -2 * n + 1) :=
by
  -- Assuming that S_n is given as per the condition of the problem
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l460_46023


namespace NUMINAMATH_GPT_factorize_x4_y4_l460_46014

theorem factorize_x4_y4 (x y : ℝ) : x^4 - y^4 = (x^2 + y^2) * (x^2 - y^2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x4_y4_l460_46014


namespace NUMINAMATH_GPT_chickens_in_zoo_l460_46032

theorem chickens_in_zoo (c e : ℕ) (h_legs : 2 * c + 4 * e = 66) (h_heads : c + e = 24) : c = 15 :=
by
  sorry

end NUMINAMATH_GPT_chickens_in_zoo_l460_46032


namespace NUMINAMATH_GPT_points_above_line_l460_46048

theorem points_above_line {t : ℝ} (hP : 1 + t - 1 > 0) (hQ : t^2 + (t - 1) - 1 > 0) : t > 1 :=
by
  sorry

end NUMINAMATH_GPT_points_above_line_l460_46048


namespace NUMINAMATH_GPT_determine_sum_of_squares_l460_46083

theorem determine_sum_of_squares
  (x y z : ℝ)
  (h1 : x + y + z = 13)
  (h2 : x * y * z = 72)
  (h3 : 1/x + 1/y + 1/z = 3/4) :
  x^2 + y^2 + z^2 = 61 := 
sorry

end NUMINAMATH_GPT_determine_sum_of_squares_l460_46083


namespace NUMINAMATH_GPT_min_value_three_l460_46018

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (1 / ((1 - x) * (1 - y) * (1 - z))) +
  (1 / ((1 + x) * (1 + y) * (1 + z))) +
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)))

theorem min_value_three (x y z : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) :
  min_value_expression x y z = 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_three_l460_46018


namespace NUMINAMATH_GPT_subtraction_base_8_correct_l460_46024

def sub_in_base_8 (a b : Nat) : Nat := sorry

theorem subtraction_base_8_correct : sub_in_base_8 (sub_in_base_8 0o123 0o51) 0o15 = 0o25 :=
sorry

end NUMINAMATH_GPT_subtraction_base_8_correct_l460_46024


namespace NUMINAMATH_GPT_value_of_b_l460_46063

theorem value_of_b (b : ℝ) (h : 4 * ((3.6 * b * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) : b = 0.48 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_b_l460_46063
