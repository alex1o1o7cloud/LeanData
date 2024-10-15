import Mathlib

namespace NUMINAMATH_GPT_total_number_of_flowers_l502_50248

theorem total_number_of_flowers (pots : ℕ) (flowers_per_pot : ℕ) (h_pots : pots = 544) (h_flowers_per_pot : flowers_per_pot = 32) : 
  pots * flowers_per_pot = 17408 := by
  sorry

end NUMINAMATH_GPT_total_number_of_flowers_l502_50248


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l502_50215

variable (x : ℝ)

def condition_p := -1 ≤ x ∧ x ≤ 1
def condition_q := x ≥ -2

theorem sufficient_but_not_necessary :
  (condition_p x → condition_q x) ∧ ¬(condition_q x → condition_p x) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l502_50215


namespace NUMINAMATH_GPT_chandler_saves_weeks_l502_50243

theorem chandler_saves_weeks 
  (cost_of_bike : ℕ) 
  (grandparents_money : ℕ) 
  (aunt_money : ℕ) 
  (cousin_money : ℕ) 
  (weekly_earnings : ℕ)
  (total_birthday_money : ℕ := grandparents_money + aunt_money + cousin_money) 
  (total_money : ℕ := total_birthday_money + weekly_earnings * 24):
  (cost_of_bike = 600) → 
  (grandparents_money = 60) → 
  (aunt_money = 40) → 
  (cousin_money = 20) → 
  (weekly_earnings = 20) → 
  (total_money = cost_of_bike) → 
  24 = ((cost_of_bike - total_birthday_money) / weekly_earnings) := 
by 
  intros; 
  sorry

end NUMINAMATH_GPT_chandler_saves_weeks_l502_50243


namespace NUMINAMATH_GPT_andrew_stamps_permits_l502_50218

theorem andrew_stamps_permits (n a T r permits : ℕ)
  (h1 : n = 2)
  (h2 : a = 3)
  (h3 : T = 8)
  (h4 : r = 50)
  (h5 : permits = (T - n * a) * r) :
  permits = 100 :=
by
  rw [h1, h2, h3, h4] at h5
  norm_num at h5
  exact h5

end NUMINAMATH_GPT_andrew_stamps_permits_l502_50218


namespace NUMINAMATH_GPT_find_y_l502_50280

theorem find_y (Y : ℝ) (h : (200 + 200 / Y) * Y = 18200) : Y = 90 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l502_50280


namespace NUMINAMATH_GPT_root_expression_value_l502_50298

noncomputable def value_of_expression (p q r : ℝ) (h1 : p + q + r = 8) (h2 : pq + pr + qr = 10) (h3 : pqr = 3) : ℝ :=
  sorry

theorem root_expression_value (p q r : ℝ) (h1 : p + q + r = 8) (h2 : pq + pr + qr = 10) (h3 : pqr = 3) :
  value_of_expression p q r h1 h2 h3 = 367 / 183 :=
sorry

end NUMINAMATH_GPT_root_expression_value_l502_50298


namespace NUMINAMATH_GPT_wall_building_time_l502_50289

theorem wall_building_time
  (m1 m2 : ℕ) 
  (d1 d2 : ℝ)
  (h1 : m1 = 20)
  (h2 : d1 = 3.0)
  (h3 : m2 = 30)
  (h4 : ∃ k, m1 * d1 = k ∧ m2 * d2 = k) :
  d2 = 2.0 :=
by
  sorry

end NUMINAMATH_GPT_wall_building_time_l502_50289


namespace NUMINAMATH_GPT_chickens_count_l502_50272

-- Define conditions
def cows : Nat := 4
def sheep : Nat := 3
def bushels_per_cow : Nat := 2
def bushels_per_sheep : Nat := 2
def bushels_per_chicken : Nat := 3
def total_bushels_needed : Nat := 35

-- The main theorem to be proven
theorem chickens_count : 
  (total_bushels_needed - ((cows * bushels_per_cow) + (sheep * bushels_per_sheep))) / bushels_per_chicken = 7 :=
by
  sorry

end NUMINAMATH_GPT_chickens_count_l502_50272


namespace NUMINAMATH_GPT_age_difference_l502_50264

theorem age_difference 
  (A B : ℤ) 
  (h1 : B = 39) 
  (h2 : A + 10 = 2 * (B - 10)) :
  A - B = 9 := 
by 
  sorry

end NUMINAMATH_GPT_age_difference_l502_50264


namespace NUMINAMATH_GPT_log_eq_solution_l502_50260

open Real

noncomputable def solve_log_eq : Real :=
  let x := 62.5^(1/3)
  x

theorem log_eq_solution (x : Real) (hx : 3 * log x - 4 * log 5 = -1) :
  x = solve_log_eq :=
by
  sorry

end NUMINAMATH_GPT_log_eq_solution_l502_50260


namespace NUMINAMATH_GPT_car_speed_l502_50265

theorem car_speed (v : ℝ) (h : (1/v) * 3600 = ((1/48) * 3600) + 15) : v = 40 := 
by 
  sorry

end NUMINAMATH_GPT_car_speed_l502_50265


namespace NUMINAMATH_GPT_fractions_equivalent_under_scaling_l502_50232

theorem fractions_equivalent_under_scaling (a b d k x : ℝ) (h₀ : d ≠ 0) (h₁ : k ≠ 0) :
  (a * (k * x) + b) / (a * (k * x) + d) = (b * (k * x)) / (d * (k * x)) ↔ b = d :=
by sorry

end NUMINAMATH_GPT_fractions_equivalent_under_scaling_l502_50232


namespace NUMINAMATH_GPT_calculate_weekly_charge_l502_50282

-- Defining conditions as constraints
def daily_charge : ℕ := 30
def total_days : ℕ := 11
def total_cost : ℕ := 310

-- Defining the weekly charge
def weekly_charge : ℕ := 190

-- Prove that the weekly charge for the first week of rental is $190
theorem calculate_weekly_charge (daily_charge total_days total_cost weekly_charge: ℕ) (daily_charge_eq : daily_charge = 30) (total_days_eq : total_days = 11) (total_cost_eq : total_cost = 310) : 
  weekly_charge = 190 :=
by
  sorry

end NUMINAMATH_GPT_calculate_weekly_charge_l502_50282


namespace NUMINAMATH_GPT_polynomial_factor_pair_l502_50283

theorem polynomial_factor_pair (a b : ℝ) :
  (∃ (c d : ℝ), 3 * x^4 + a * x^3 + 48 * x^2 + b * x + 12 = (2 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 6)) →
  (a, b) = (-26.5, -40) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factor_pair_l502_50283


namespace NUMINAMATH_GPT_james_monthly_earnings_l502_50229

theorem james_monthly_earnings (initial_subscribers gifted_subscribers earnings_per_subscriber : ℕ)
  (initial_subscribers_eq : initial_subscribers = 150)
  (gifted_subscribers_eq : gifted_subscribers = 50)
  (earnings_per_subscriber_eq : earnings_per_subscriber = 9) :
  (initial_subscribers + gifted_subscribers) * earnings_per_subscriber = 1800 := by
  sorry

end NUMINAMATH_GPT_james_monthly_earnings_l502_50229


namespace NUMINAMATH_GPT_cost_of_pink_notebook_l502_50293

theorem cost_of_pink_notebook
    (total_cost : ℕ) 
    (black_cost : ℕ) 
    (green_cost : ℕ) 
    (num_green : ℕ) 
    (num_black : ℕ) 
    (num_pink : ℕ)
    (total_notebooks : ℕ)
    (h_total_cost : total_cost = 45)
    (h_black_cost : black_cost = 15) 
    (h_green_cost : green_cost = 10) 
    (h_num_green : num_green = 2) 
    (h_num_black : num_black = 1) 
    (h_num_pink : num_pink = 1)
    (h_total_notebooks : total_notebooks = 4) 
    : (total_cost - (num_green * green_cost + black_cost) = 10) :=
by
  sorry

end NUMINAMATH_GPT_cost_of_pink_notebook_l502_50293


namespace NUMINAMATH_GPT_cyclists_cannot_reach_point_B_l502_50235

def v1 := 35 -- Speed of the first cyclist in km/h
def v2 := 25 -- Speed of the second cyclist in km/h
def t := 2   -- Total time in hours
def d  := 30 -- Distance from A to B in km

-- Each cyclist does not rest simultaneously
-- Time equations based on their speed proportions

theorem cyclists_cannot_reach_point_B 
  (v1 := 35) (v2 := 25) (t := 2) (d := 30) 
  (h1 : t * (v1 * (5 / (5 + 7)) / 60) + t * (v2 * (7 / (5 + 7)) / 60) < d) : 
  False := 
sorry

end NUMINAMATH_GPT_cyclists_cannot_reach_point_B_l502_50235


namespace NUMINAMATH_GPT_min_value_expr_min_max_value_expr_max_l502_50239

noncomputable def min_value_expr (a b : ℝ) : ℝ := 
  1 / (a - b) + 4 / (b - 1)

noncomputable def max_value_expr (a b : ℝ) : ℝ :=
  a * b - b^2 - a + b

theorem min_value_expr_min (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3 * b = 5) : 
  min_value_expr a b = 25 :=
sorry

theorem max_value_expr_max (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3 * b = 5) :
  max_value_expr a b = 1 / 16 :=
sorry

end NUMINAMATH_GPT_min_value_expr_min_max_value_expr_max_l502_50239


namespace NUMINAMATH_GPT_largest_integer_y_l502_50281

theorem largest_integer_y (y : ℤ) : 
  (∃ k : ℤ, (y^2 + 3*y + 10) = k * (y - 4)) → y ≤ 42 :=
sorry

end NUMINAMATH_GPT_largest_integer_y_l502_50281


namespace NUMINAMATH_GPT_algebraic_expression_value_l502_50263

open Real

theorem algebraic_expression_value
  (θ : ℝ)
  (a := (cos θ, sin θ))
  (b := (1, -2))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  (2 * sin θ - cos θ) / (sin θ + cos θ) = 5 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l502_50263


namespace NUMINAMATH_GPT_find_number_l502_50254

theorem find_number (N p q : ℝ) 
  (h1 : N / p = 6) 
  (h2 : N / q = 18) 
  (h3 : p - q = 1 / 3) : 
  N = 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l502_50254


namespace NUMINAMATH_GPT_point_C_coordinates_line_MN_equation_area_triangle_ABC_l502_50285

-- Define the points A and B
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (7, 3)

-- Let C be an unknown point that we need to determine
variables (x y : ℝ)

-- Define the conditions given in the problem
axiom midpoint_M : (x + 5) / 2 = 0 ∧ (y + 3) / 2 = 0 -- Midpoint M lies on the y-axis
axiom midpoint_N : (x + 7) / 2 = 1 ∧ (y + 3) / 2 = 0 -- Midpoint N lies on the x-axis

-- The problem consists of proving three assertions
theorem point_C_coordinates :
  ∃ (x y : ℝ), (x, y) = (-5, -3) :=
by
  sorry

theorem line_MN_equation :
  ∃ (a b c : ℝ), a = 5 ∧ b = -2 ∧ c = -5 :=
by
  sorry

theorem area_triangle_ABC :
  ∃ (S : ℝ), S = 841 / 20 :=
by
  sorry

end NUMINAMATH_GPT_point_C_coordinates_line_MN_equation_area_triangle_ABC_l502_50285


namespace NUMINAMATH_GPT_find_k_l502_50236

-- Define the sequence and its sum
def Sn (k : ℝ) (n : ℕ) : ℝ := k + 3^n
def an (k : ℝ) (n : ℕ) : ℝ := Sn k n - (if n = 0 then 0 else Sn k (n - 1))

-- Define the condition that a sequence is geometric
def is_geometric (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = r * a n

theorem find_k (k : ℝ) :
  is_geometric (an k) (an k 1 / an k 0) → k = -1 := 
by sorry

end NUMINAMATH_GPT_find_k_l502_50236


namespace NUMINAMATH_GPT_find_prices_and_max_basketballs_l502_50295

def unit_price_condition (x : ℕ) (y : ℕ) : Prop :=
  y = 2*x - 30

def cost_ratio_condition (x : ℕ) (y : ℕ) : Prop :=
  3 * x = 2 * y - 60

def total_cost_condition (total_cost : ℕ) (num_basketballs : ℕ) (num_soccerballs : ℕ) : Prop :=
  total_cost ≤ 15500 ∧ num_basketballs + num_soccerballs = 200

theorem find_prices_and_max_basketballs
  (x y : ℕ) (total_cost : ℕ) (num_basketballs : ℕ) (num_soccerballs : ℕ)
  (h1 : unit_price_condition x y)
  (h2 : cost_ratio_condition x y)
  (h3 : total_cost_condition total_cost num_basketballs num_soccerballs)
  (h4 : total_cost = 90 * num_basketballs + 60 * num_soccerballs)
  : x = 60 ∧ y = 90 ∧ num_basketballs ≤ 116 :=
sorry

end NUMINAMATH_GPT_find_prices_and_max_basketballs_l502_50295


namespace NUMINAMATH_GPT_relationship_P_Q_l502_50214

theorem relationship_P_Q (x : ℝ) (P : ℝ) (Q : ℝ) 
  (hP : P = Real.exp x + Real.exp (-x)) 
  (hQ : Q = (Real.sin x + Real.cos x) ^ 2) : 
  P ≥ Q := 
sorry

end NUMINAMATH_GPT_relationship_P_Q_l502_50214


namespace NUMINAMATH_GPT_tan_alpha_plus_beta_tan_beta_l502_50240

variable (α β : ℝ)

-- Given conditions
def tan_condition_1 : Prop := Real.tan (Real.pi + α) = -1 / 3
def tan_condition_2 : Prop := Real.tan (α + β) = (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α)

-- Proving the results
theorem tan_alpha_plus_beta (h1 : tan_condition_1 α) (h2 : tan_condition_2 α β) : 
  Real.tan (α + β) = 5 / 16 :=
sorry

theorem tan_beta (h1 : tan_condition_1 α) (h2 : tan_condition_2 α β) :
  Real.tan β = 31 / 43 :=
sorry

end NUMINAMATH_GPT_tan_alpha_plus_beta_tan_beta_l502_50240


namespace NUMINAMATH_GPT_division_remainder_correct_l502_50296

def polynomial_div_remainder (x : ℝ) : ℝ :=
  3 * x^4 + 14 * x^3 - 50 * x^2 - 72 * x + 55

def divisor (x : ℝ) : ℝ :=
  x^2 + 8 * x - 4

theorem division_remainder_correct :
  ∀ x : ℝ, polynomial_div_remainder x % divisor x = 224 * x - 113 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_correct_l502_50296


namespace NUMINAMATH_GPT_problem_statement_l502_50252

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l502_50252


namespace NUMINAMATH_GPT_omega_not_possible_l502_50227

noncomputable def f (ω x φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem omega_not_possible (ω φ : ℝ) (h1 : ∀ x y, -π/3 ≤ x → x < y → y ≤ π/6 → f ω x φ ≤ f ω y φ)
  (h2 : f ω (π / 6) φ = f ω (4 * π / 3) φ)
  (h3 : f ω (π / 6) φ = -f ω (-π / 3) φ) :
  ω ≠ 7 / 5 :=
sorry

end NUMINAMATH_GPT_omega_not_possible_l502_50227


namespace NUMINAMATH_GPT_collinear_iff_linear_combination_l502_50246

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C : V) (k : ℝ)

theorem collinear_iff_linear_combination (O A B C : V) (k : ℝ) :
  (C = k • A + (1 - k) • B) ↔ ∃ (k' : ℝ), C - B = k' • (A - B) :=
sorry

end NUMINAMATH_GPT_collinear_iff_linear_combination_l502_50246


namespace NUMINAMATH_GPT_percentage_of_employees_driving_l502_50217

theorem percentage_of_employees_driving
  (total_employees : ℕ)
  (drivers : ℕ)
  (public_transport : ℕ)
  (H1 : total_employees = 200)
  (H2 : drivers = public_transport + 40)
  (H3 : public_transport = (total_employees - drivers) / 2) :
  (drivers:ℝ) / (total_employees:ℝ) * 100 = 46.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_percentage_of_employees_driving_l502_50217


namespace NUMINAMATH_GPT_reciprocal_sum_l502_50294

variable {x y z a b c : ℝ}

-- The function statement where we want to show the equivalence.
theorem reciprocal_sum (h1 : x ≠ y) (h2 : x ≠ z) (h3 : y ≠ z)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hxy : (x * y) / (x - y) = a)
  (hxz : (x * z) / (x - z) = b)
  (hyz : (y * z) / (y - z) = c) :
  (1/x + 1/y + 1/z) = ((1/a + 1/b + 1/c) / 2) :=
sorry

end NUMINAMATH_GPT_reciprocal_sum_l502_50294


namespace NUMINAMATH_GPT_diamond_fifteen_two_l502_50292

def diamond (a b : ℤ) : ℤ := a + (a / (b + 1))

theorem diamond_fifteen_two : diamond 15 2 = 20 := 
by 
    sorry

end NUMINAMATH_GPT_diamond_fifteen_two_l502_50292


namespace NUMINAMATH_GPT_vertex_of_parabola_l502_50237

def f (x : ℝ) : ℝ := 2 - (2*x + 1)^2

theorem vertex_of_parabola :
  (∀ x : ℝ, f x ≤ 2) ∧ (f (-1/2) = 2) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l502_50237


namespace NUMINAMATH_GPT_points_lost_calculation_l502_50284

variable (firstRound secondRound finalScore : ℕ)
variable (pointsLost : ℕ)

theorem points_lost_calculation 
  (h1 : firstRound = 40) 
  (h2 : secondRound = 50) 
  (h3 : finalScore = 86) 
  (h4 : pointsLost = firstRound + secondRound - finalScore) :
  pointsLost = 4 := 
sorry

end NUMINAMATH_GPT_points_lost_calculation_l502_50284


namespace NUMINAMATH_GPT_geometric_progression_first_term_one_l502_50291

theorem geometric_progression_first_term_one (a r : ℝ) (gp : ℕ → ℝ)
  (h_gp : ∀ n, gp n = a * r^(n - 1))
  (h_product_in_gp : ∀ i j, ∃ k, gp i * gp j = gp k) :
  a = 1 := 
sorry

end NUMINAMATH_GPT_geometric_progression_first_term_one_l502_50291


namespace NUMINAMATH_GPT_time_to_cover_escalator_l502_50261

-- Definitions for the provided conditions.
def escalator_speed : ℝ := 7
def escalator_length : ℝ := 180
def person_speed : ℝ := 2

-- Goal to prove the time taken to cover the escalator length.
theorem time_to_cover_escalator : (escalator_length / (escalator_speed + person_speed)) = 20 := by
  sorry

end NUMINAMATH_GPT_time_to_cover_escalator_l502_50261


namespace NUMINAMATH_GPT_books_remaining_after_second_day_l502_50269

variable (x a b c d : ℕ)

theorem books_remaining_after_second_day :
  let books_borrowed_first_day := a * b
  let books_borrowed_second_day := c
  let books_returned_second_day := (d * books_borrowed_first_day) / 100
  x - books_borrowed_first_day - books_borrowed_second_day + books_returned_second_day =
  x - (a * b) - c + ((d * (a * b)) / 100) :=
sorry

end NUMINAMATH_GPT_books_remaining_after_second_day_l502_50269


namespace NUMINAMATH_GPT_central_angle_of_unfolded_side_surface_l502_50249

theorem central_angle_of_unfolded_side_surface
  (radius : ℝ) (slant_height : ℝ) (arc_length : ℝ) (central_angle_deg : ℝ)
  (h_radius : radius = 1)
  (h_slant_height : slant_height = 3)
  (h_arc_length : arc_length = 2 * Real.pi) :
  central_angle_deg = 120 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_unfolded_side_surface_l502_50249


namespace NUMINAMATH_GPT_difference_between_picked_and_left_is_five_l502_50297

theorem difference_between_picked_and_left_is_five :
  let dave_sticks := 14
  let amy_sticks := 9
  let ben_sticks := 12
  let total_initial_sticks := 65
  let total_picked_up := dave_sticks + amy_sticks + ben_sticks
  let sticks_left := total_initial_sticks - total_picked_up
  total_picked_up - sticks_left = 5 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_picked_and_left_is_five_l502_50297


namespace NUMINAMATH_GPT_find_y_given_conditions_l502_50290

theorem find_y_given_conditions (x y : ℝ) (hx : x = 102) 
                                (h : x^3 * y - 3 * x^2 * y + 3 * x * y = 106200) : 
  y = 10 / 97 :=
by
  sorry

end NUMINAMATH_GPT_find_y_given_conditions_l502_50290


namespace NUMINAMATH_GPT_A_can_give_C_start_l502_50206

noncomputable def start_A_can_give_C : ℝ :=
  let start_AB := 50
  let start_BC := 157.89473684210532
  start_AB + start_BC

theorem A_can_give_C_start :
  start_A_can_give_C = 207.89473684210532 :=
by
  sorry

end NUMINAMATH_GPT_A_can_give_C_start_l502_50206


namespace NUMINAMATH_GPT_all_perfect_squares_l502_50267

theorem all_perfect_squares (a b c : ℕ) (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) 
  (h_eq : a ^ 2 + b ^ 2 + c ^ 2 = 2 * (a * b + b * c + c * a)) : 
  ∃ (k l m : ℕ), a = k ^ 2 ∧ b = l ^ 2 ∧ c = m ^ 2 :=
sorry

end NUMINAMATH_GPT_all_perfect_squares_l502_50267


namespace NUMINAMATH_GPT_find_b_l502_50270

noncomputable def f (x : ℝ) : ℝ := (x+1)^3 + (x / (x + 1))

theorem find_b (b : ℝ) (h_sum : ∃ x1 x2 : ℝ, f x1 = -x1 + b ∧ f x2 = -x2 + b ∧ x1 + x2 = -2) : b = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l502_50270


namespace NUMINAMATH_GPT_isosceles_triangle_smallest_angle_l502_50205

-- Given conditions:
-- 1. The triangle is isosceles
-- 2. One angle is 40% larger than the measure of a right angle

theorem isosceles_triangle_smallest_angle :
  ∃ (A B C : ℝ), 
  A + B + C = 180 ∧ 
  (A = B ∨ A = C ∨ B = C) ∧ 
  (∃ (large_angle : ℝ), large_angle = 90 + 0.4 * 90 ∧ (A = large_angle ∨ B = large_angle ∨ C = large_angle)) →
  (A = 27 ∨ B = 27 ∨ C = 27) := sorry

end NUMINAMATH_GPT_isosceles_triangle_smallest_angle_l502_50205


namespace NUMINAMATH_GPT_total_legs_and_hands_on_ground_is_118_l502_50299

-- Definitions based on the conditions given
def total_dogs := 20
def dogs_on_two_legs := total_dogs / 2
def dogs_on_four_legs := total_dogs / 2

def total_cats := 10
def cats_on_two_legs := total_cats / 3
def cats_on_four_legs := total_cats - cats_on_two_legs

def total_horses := 5
def horses_on_two_legs := 2
def horses_on_four_legs := total_horses - horses_on_two_legs

def total_acrobats := 6
def acrobats_on_one_hand := 4
def acrobats_on_two_hands := 2

-- Functions to calculate the number of legs/paws/hands on the ground
def dogs_legs_on_ground := (dogs_on_two_legs * 2) + (dogs_on_four_legs * 4)
def cats_legs_on_ground := (cats_on_two_legs * 2) + (cats_on_four_legs * 4)
def horses_legs_on_ground := (horses_on_two_legs * 2) + (horses_on_four_legs * 4)
def acrobats_hands_on_ground := (acrobats_on_one_hand * 1) + (acrobats_on_two_hands * 2)

-- Total legs/paws/hands on the ground
def total_legs_on_ground := dogs_legs_on_ground + cats_legs_on_ground + horses_legs_on_ground + acrobats_hands_on_ground

-- The theorem to prove
theorem total_legs_and_hands_on_ground_is_118 : total_legs_on_ground = 118 :=
by sorry

end NUMINAMATH_GPT_total_legs_and_hands_on_ground_is_118_l502_50299


namespace NUMINAMATH_GPT_conference_handshakes_l502_50259

theorem conference_handshakes (total_people : ℕ) (group1_people : ℕ) (group2_people : ℕ)
  (group1_knows_each_other : group1_people = 25)
  (group2_knows_no_one_in_group1 : group2_people = 15)
  (total_group : total_people = group1_people + group2_people)
  (total_handshakes : ℕ := group2_people * (group1_people + group2_people - 1) - group2_people * (group2_people - 1) / 2) :
  total_handshakes = 480 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_conference_handshakes_l502_50259


namespace NUMINAMATH_GPT_minimum_value_l502_50268

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (1 / x + 9 / y) = 16 :=
sorry

end NUMINAMATH_GPT_minimum_value_l502_50268


namespace NUMINAMATH_GPT_three_digit_number_divisible_by_8_and_even_tens_digit_l502_50266

theorem three_digit_number_divisible_by_8_and_even_tens_digit (d : ℕ) (hd : d % 2 = 0) (hdiv : (100 * 5 + 10 * d + 4) % 8 = 0) :
  100 * 5 + 10 * d + 4 = 544 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_divisible_by_8_and_even_tens_digit_l502_50266


namespace NUMINAMATH_GPT_Louisa_traveled_240_miles_first_day_l502_50258

noncomputable def distance_first_day (h : ℕ) := 60 * (h - 3)

theorem Louisa_traveled_240_miles_first_day :
  ∃ h : ℕ, 420 = 60 * h ∧ distance_first_day h = 240 :=
by
  sorry

end NUMINAMATH_GPT_Louisa_traveled_240_miles_first_day_l502_50258


namespace NUMINAMATH_GPT_part_a_part_b_l502_50279

theorem part_a (n : ℕ) (hn : n % 2 = 1) (h_pos : n > 0) :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n-1 ∧ ∃ f : (ℕ → ℕ), f k ≥ (n - 1) / 2 :=
sorry

theorem part_b : ∃ᶠ n in at_top, ∃ f : (ℕ → ℕ), ∀ k : ℕ, 1 ≤ k ∧ k ≤ n-1 → f k ≤ (n - 1) / 2 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l502_50279


namespace NUMINAMATH_GPT_parallel_lines_intersect_parabola_l502_50201

theorem parallel_lines_intersect_parabola {a k b c x1 x2 x3 x4 : ℝ} 
    (h₁ : x1 < x2) 
    (h₂ : x3 < x4) 
    (intersect1 : ∀ y : ℝ, y = k * x1 + b ∧ y = a * x1^2 ∧ y = k * x2 + b ∧ y = a * x2^2) 
    (intersect2 : ∀ y : ℝ, y = k * x3 + c ∧ y = a * x3^2 ∧ y = k * x4 + c ∧ y = a * x4^2) :
    (x3 - x1) = (x2 - x4) := 
by 
    sorry

end NUMINAMATH_GPT_parallel_lines_intersect_parabola_l502_50201


namespace NUMINAMATH_GPT_initial_loss_percentage_l502_50255

theorem initial_loss_percentage 
  (CP : ℝ := 250) 
  (SP : ℝ) 
  (h1 : SP + 50 = 1.10 * CP) : 
  (CP - SP) / CP * 100 = 10 := 
sorry

end NUMINAMATH_GPT_initial_loss_percentage_l502_50255


namespace NUMINAMATH_GPT_domain_of_rational_function_l502_50200

theorem domain_of_rational_function 
  (c : ℝ) 
  (h : -7 * (6 ^ 2) + 28 * c < 0) : 
  c < -9 / 7 :=
by sorry

end NUMINAMATH_GPT_domain_of_rational_function_l502_50200


namespace NUMINAMATH_GPT_roof_ratio_l502_50287

theorem roof_ratio (L W : ℝ) (h1 : L * W = 576) (h2 : L - W = 36) : L / W = 4 := 
by
  sorry

end NUMINAMATH_GPT_roof_ratio_l502_50287


namespace NUMINAMATH_GPT_area_of_square_is_correct_l502_50210

-- Define the nature of the problem setup and parameters
def radius_of_circle : ℝ := 7
def diameter_of_circle : ℝ := 2 * radius_of_circle
def side_length_of_square : ℝ := 2 * diameter_of_circle
def area_of_square : ℝ := side_length_of_square ^ 2

-- Statement of the problem to prove
theorem area_of_square_is_correct : area_of_square = 784 := by
  sorry

end NUMINAMATH_GPT_area_of_square_is_correct_l502_50210


namespace NUMINAMATH_GPT_units_produced_by_line_B_l502_50250

-- State the problem with the given conditions and prove the question equals the answer.
theorem units_produced_by_line_B (total_units : ℕ) (B : ℕ) (A C : ℕ) 
    (h1 : total_units = 13200)
    (h2 : A + B + C = total_units)
    (h3 : ∃ d : ℕ, A = B - d ∧ C = B + d) :
    B = 4400 :=
by
  sorry

end NUMINAMATH_GPT_units_produced_by_line_B_l502_50250


namespace NUMINAMATH_GPT_funfair_initial_visitors_l502_50222

theorem funfair_initial_visitors {a : ℕ} (ha1 : 50 * a - 40 > 0) (ha2 : 90 - 20 * a > 0) (ha3 : 50 * a - 40 > 90 - 20 * a) :
  (50 * a - 40 = 60) ∨ (50 * a - 40 = 110) ∨ (50 * a - 40 = 160) :=
sorry

end NUMINAMATH_GPT_funfair_initial_visitors_l502_50222


namespace NUMINAMATH_GPT_James_pays_35_l502_50241

theorem James_pays_35 (first_lesson_free : Bool) (total_lessons : Nat) (cost_per_lesson : Nat) 
  (first_x_paid_lessons_free : Nat) (every_other_remainings_free : Nat) (uncle_pays_half : Bool) :
  total_lessons = 20 → 
  first_lesson_free = true → 
  cost_per_lesson = 5 →
  first_x_paid_lessons_free = 10 →
  every_other_remainings_free = 1 → 
  uncle_pays_half = true →
  (10 * cost_per_lesson + 4 * cost_per_lesson) / 2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_James_pays_35_l502_50241


namespace NUMINAMATH_GPT_solve_linear_eq_l502_50244

theorem solve_linear_eq (x : ℝ) : (x + 1) / 3 = 0 → x = -1 := 
by 
  sorry

end NUMINAMATH_GPT_solve_linear_eq_l502_50244


namespace NUMINAMATH_GPT_circumference_of_wheels_l502_50211

-- Define the variables and conditions
variables (x y : ℝ)

def condition1 (x y : ℝ) : Prop := (120 / x) - (120 / y) = 6
def condition2 (x y : ℝ) : Prop := (4 / 5) * (120 / x) - (5 / 6) * (120 / y) = 4

-- The main theorem to prove
theorem circumference_of_wheels (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 4 ∧ y = 5 :=
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_circumference_of_wheels_l502_50211


namespace NUMINAMATH_GPT_solve_inequality_l502_50225

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := (x - 2) * (a * x + 2 * a)

-- Theorem Statement
theorem solve_inequality (f_even : ∀ x a, f x a = f (-x) a) (f_inc : ∀ x y a, 0 < x → x < y → f x a ≤ f y a) :
    ∀ a > 0, { x : ℝ | f (2 - x) a > 0 } = { x | x < 0 ∨ x > 4 } :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_solve_inequality_l502_50225


namespace NUMINAMATH_GPT_no_intersection_of_sets_l502_50278

noncomputable def A (a b x y : ℝ) :=
  a * (Real.sin x + Real.sin y) + (b - 1) * (Real.cos x + Real.cos y) = 0

noncomputable def B (a b x y : ℝ) :=
  (b + 1) * Real.sin (x + y) - a * Real.cos (x + y) = a

noncomputable def C (a b : ℝ) :=
  ∀ z : ℝ, z^2 - 2 * (a - b) * z + (a + b)^2 - 2 > 0

theorem no_intersection_of_sets (a b x y : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) (h3 : 0 < y) (h4 : y < Real.pi / 2) :
  (C a b) → ¬(∃ x y, A a b x y ∧ B a b x y) :=
by 
  sorry

end NUMINAMATH_GPT_no_intersection_of_sets_l502_50278


namespace NUMINAMATH_GPT_prove_b_value_l502_50271

theorem prove_b_value (b : ℚ) (h : b + b / 4 = 10 / 4) : b = 2 :=
sorry

end NUMINAMATH_GPT_prove_b_value_l502_50271


namespace NUMINAMATH_GPT_fundraiser_goal_l502_50216

theorem fundraiser_goal (bronze_donation silver_donation gold_donation goal : ℕ)
  (bronze_families silver_families gold_family : ℕ)
  (H_bronze_amount : bronze_families * bronze_donation = 250)
  (H_silver_amount : silver_families * silver_donation = 350)
  (H_gold_amount : gold_family * gold_donation = 100)
  (H_goal : goal = 750) :
  goal - (bronze_families * bronze_donation + silver_families * silver_donation + gold_family * gold_donation) = 50 :=
by
  sorry

end NUMINAMATH_GPT_fundraiser_goal_l502_50216


namespace NUMINAMATH_GPT_calories_per_burger_l502_50212

-- Conditions given in the problem
def burgers_per_day : Nat := 3
def days : Nat := 2
def total_calories : Nat := 120

-- Total burgers Dimitri will eat in the given period
def total_burgers := burgers_per_day * days

-- Prove that the number of calories per burger is 20
theorem calories_per_burger : total_calories / total_burgers = 20 := 
by 
  -- Skipping the proof with 'sorry' as instructed
  sorry

end NUMINAMATH_GPT_calories_per_burger_l502_50212


namespace NUMINAMATH_GPT_no_integer_solution_l502_50220

theorem no_integer_solution : ¬ ∃ (x y : ℤ), x^2 - 7 * y = 10 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_l502_50220


namespace NUMINAMATH_GPT_area_of_field_l502_50276

theorem area_of_field (w l A : ℝ) 
    (h1 : l = 2 * w + 35) 
    (h2 : 2 * (w + l) = 700) : 
    A = 25725 :=
by sorry

end NUMINAMATH_GPT_area_of_field_l502_50276


namespace NUMINAMATH_GPT_sum_of_first_40_terms_l502_50207

def a : ℕ → ℤ := sorry

def S (n : ℕ) : ℤ := (Finset.range n).sum a

theorem sum_of_first_40_terms :
  (∀ n : ℕ, a (n + 1) + (-1) ^ n * a n = n) →
  S 40 = 420 := 
sorry

end NUMINAMATH_GPT_sum_of_first_40_terms_l502_50207


namespace NUMINAMATH_GPT_max_t_for_real_root_l502_50274

theorem max_t_for_real_root (t : ℝ) (x : ℝ) 
  (h : 0 < x ∧ x < π ∧ (t+1) * Real.cos x - t * Real.sin x = t + 2) : t = -1 :=
sorry

end NUMINAMATH_GPT_max_t_for_real_root_l502_50274


namespace NUMINAMATH_GPT_sqrt_expression_value_l502_50208

variable (a b : ℝ) 

theorem sqrt_expression_value (ha : a ≠ 0) (hb : b ≠ 0) (ha_neg : a < 0) :
  Real.sqrt (-a^3) * Real.sqrt ((-b)^4) = -a * |b| * Real.sqrt (-a) := by
  sorry

end NUMINAMATH_GPT_sqrt_expression_value_l502_50208


namespace NUMINAMATH_GPT_infinite_primes_solutions_l502_50202

theorem infinite_primes_solutions :
  ∀ (P : Finset ℕ), (∀ p ∈ P, Prime p) →
  ∃ q, Prime q ∧ q ∉ P ∧ ∃ x y : ℤ, x^2 + x + 1 = q * y :=
by sorry

end NUMINAMATH_GPT_infinite_primes_solutions_l502_50202


namespace NUMINAMATH_GPT_find_k_value_for_unique_real_solution_l502_50223

noncomputable def cubic_has_exactly_one_real_solution (k : ℝ) : Prop :=
    ∃! x : ℝ, 4*x^3 + 9*x^2 + k*x + 4 = 0

theorem find_k_value_for_unique_real_solution :
  ∃ (k : ℝ), k > 0 ∧ cubic_has_exactly_one_real_solution k ∧ k = 6.75 :=
sorry

end NUMINAMATH_GPT_find_k_value_for_unique_real_solution_l502_50223


namespace NUMINAMATH_GPT_function_intersection_le_one_l502_50288

theorem function_intersection_le_one (f : ℝ → ℝ)
  (h : ∀ x t : ℝ, t ≠ 0 → t * (f (x + t) - f x) > 0) :
  ∀ a : ℝ, ∃! x : ℝ, f x = a :=
by 
sorry

end NUMINAMATH_GPT_function_intersection_le_one_l502_50288


namespace NUMINAMATH_GPT_imaginary_part_is_empty_l502_50251

def imaginary_part_empty (z : ℂ) : Prop :=
  z.im = 0

theorem imaginary_part_is_empty (z : ℂ) (h : z.im = 0) : imaginary_part_empty z :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_imaginary_part_is_empty_l502_50251


namespace NUMINAMATH_GPT_problem_a_range_l502_50277

theorem problem_a_range (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x^2 - 2 * (a - 1) * x - 2 < 0) ↔ (-1 < a ∧ a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_a_range_l502_50277


namespace NUMINAMATH_GPT_range_of_a_l502_50219

theorem range_of_a 
  (f : ℝ → ℝ)
  (h_even : ∀ x, -5 ≤ x ∧ x ≤ 5 → f x = f (-x))
  (h_decreasing : ∀ a b, 0 ≤ a ∧ a < b ∧ b ≤ 5 → f b < f a)
  (h_inequality : ∀ a, f (2 * a + 3) < f a) :
  ∀ a, -5 ≤ a ∧ a ≤ 5 → a ∈ (Set.Icc (-4) (-3) ∪ Set.Ioc (-1) 1) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l502_50219


namespace NUMINAMATH_GPT_removing_zeros_changes_value_l502_50257

noncomputable def a : ℝ := 7.0800
noncomputable def b : ℝ := 7.8

theorem removing_zeros_changes_value : a ≠ b :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_removing_zeros_changes_value_l502_50257


namespace NUMINAMATH_GPT_flag_count_l502_50209

-- Definitions of colors as a datatype
inductive Color
| red : Color
| white : Color
| blue : Color
| green : Color
| yellow : Color

open Color

-- Total number of distinct flags possible
theorem flag_count : 
  (∃ m : Color, 
   (∃ t : Color, 
    (t ≠ m ∧ 
     ∃ b : Color, 
     (b ≠ m ∧ b ≠ red ∧ b ≠ blue)))) ∧ 
  (5 * 4 * 2 = 40) := 
  sorry

end NUMINAMATH_GPT_flag_count_l502_50209


namespace NUMINAMATH_GPT_expression_evaluation_l502_50228

theorem expression_evaluation :
  (0.8 ^ 3) - ((0.5 ^ 3) / (0.8 ^ 2)) + 0.40 + (0.5 ^ 2) = 0.9666875 := 
by 
  sorry

end NUMINAMATH_GPT_expression_evaluation_l502_50228


namespace NUMINAMATH_GPT_jeans_and_shirts_l502_50238

-- Let's define the necessary variables and conditions.
variables (J S X : ℝ)

-- Given conditions
def condition1 := 3 * J + 2 * S = X
def condition2 := 2 * J + 3 * S = 61

-- Given the price of one shirt
def price_of_shirt := S = 9

-- The problem we need to prove
theorem jeans_and_shirts : condition1 J S X ∧ condition2 J S ∧ price_of_shirt S →
  X = 69 :=
by
  sorry

end NUMINAMATH_GPT_jeans_and_shirts_l502_50238


namespace NUMINAMATH_GPT_total_pages_of_book_l502_50213

theorem total_pages_of_book (P : ℝ) (h : 0.4 * P = 16) : P = 40 :=
sorry

end NUMINAMATH_GPT_total_pages_of_book_l502_50213


namespace NUMINAMATH_GPT_correct_equation_l502_50203

-- Define the daily paving distances for Team A and Team B
variables (x : ℝ) (h₀ : x > 10)

-- Assuming Team A takes the same number of days to pave 150m as Team B takes to pave 120m
def same_days_to_pave (h₁ : x - 10 > 0) : Prop :=
  (150 / x = 120 / (x - 10))

-- The theorem to be proven
theorem correct_equation (h₁ : x - 10 > 0) : 150 / x = 120 / (x - 10) :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_l502_50203


namespace NUMINAMATH_GPT_find_x_l502_50224

theorem find_x (x : ℚ) (h : |x - 1| = |x - 2|) : x = 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_x_l502_50224


namespace NUMINAMATH_GPT_pow_div_l502_50204

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end NUMINAMATH_GPT_pow_div_l502_50204


namespace NUMINAMATH_GPT_larger_box_can_carry_more_clay_l502_50256

variable {V₁ : ℝ} -- Volume of the first box
variable {V₂ : ℝ} -- Volume of the second box
variable {m₁ : ℝ} -- Mass the first box can carry
variable {m₂ : ℝ} -- Mass the second box can carry

-- Defining the dimensions of the first box.
def height₁ : ℝ := 1
def width₁ : ℝ := 2
def length₁ : ℝ := 4

-- Defining the dimensions of the second box.
def height₂ : ℝ := 3 * height₁
def width₂ : ℝ := 2 * width₁
def length₂ : ℝ := 2 * length₁

-- Volume calculation for the first box.
def volume₁ : ℝ := height₁ * width₁ * length₁

-- Volume calculation for the second box.
def volume₂ : ℝ := height₂ * width₂ * length₂

-- Condition: The first box can carry 30 grams of clay
def mass₁ : ℝ := 30

-- Given the above conditions, prove the second box can carry 360 grams of clay.
theorem larger_box_can_carry_more_clay (h₁ : volume₁ = height₁ * width₁ * length₁)
                                      (h₂ : volume₂ = height₂ * width₂ * length₂)
                                      (h₃ : mass₁ = 30)
                                      (h₄ : V₁ = volume₁)
                                      (h₅ : V₂ = volume₂) :
  m₂ = 12 * mass₁ := by
  -- Skipping the detailed proof.
  sorry

end NUMINAMATH_GPT_larger_box_can_carry_more_clay_l502_50256


namespace NUMINAMATH_GPT_chord_line_parabola_l502_50253

theorem chord_line_parabola (x1 x2 y1 y2 : ℝ) (hx1 : y1^2 = 8*x1) (hx2 : y2^2 = 8*x2)
  (hmid : (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1) : 4*(1/2*(x1 + x2)) + (1/2*(y1 + y2)) - 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_chord_line_parabola_l502_50253


namespace NUMINAMATH_GPT_ratio_of_d_to_s_l502_50234

theorem ratio_of_d_to_s (s d : ℝ) (n : ℕ) (h1 : n = 15) (h2 : (n^2 * s^2) / ((n * s + 2 * n * d)^2) = 0.75) :
  d / s = 1 / 13 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_d_to_s_l502_50234


namespace NUMINAMATH_GPT_jennifer_sweets_l502_50262

theorem jennifer_sweets :
  let green_sweets := 212
  let blue_sweets := 310
  let yellow_sweets := 502
  let total_sweets := green_sweets + blue_sweets + yellow_sweets
  let number_of_people := 4
  total_sweets / number_of_people = 256 := 
by
  sorry

end NUMINAMATH_GPT_jennifer_sweets_l502_50262


namespace NUMINAMATH_GPT_product_of_three_numbers_l502_50247

theorem product_of_three_numbers
  (x y z n : ℤ)
  (h1 : x + y + z = 165)
  (h2 : n = 7 * x)
  (h3 : n = y - 9)
  (h4 : n = z + 9) :
  x * y * z = 64328 := 
by
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_l502_50247


namespace NUMINAMATH_GPT_distinct_numbers_in_union_set_l502_50273

def first_seq_term (k : ℕ) : ℤ := 5 * ↑k - 3
def second_seq_term (m : ℕ) : ℤ := 9 * ↑m - 3

def first_seq_set : Finset ℤ := ((Finset.range 1003).image first_seq_term)
def second_seq_set : Finset ℤ := ((Finset.range 1003).image second_seq_term)

def union_set : Finset ℤ := first_seq_set ∪ second_seq_set

theorem distinct_numbers_in_union_set : union_set.card = 1895 := by
  sorry

end NUMINAMATH_GPT_distinct_numbers_in_union_set_l502_50273


namespace NUMINAMATH_GPT_no_solution_inequality_l502_50226

theorem no_solution_inequality (a : ℝ) : (¬ ∃ x : ℝ, x > 1 ∧ x < a - 1) → a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_inequality_l502_50226


namespace NUMINAMATH_GPT_solve_linear_eq_l502_50242

theorem solve_linear_eq (x : ℝ) : 3 * x - 6 = 0 ↔ x = 2 :=
sorry

end NUMINAMATH_GPT_solve_linear_eq_l502_50242


namespace NUMINAMATH_GPT_max_a_is_2_l502_50233

noncomputable def max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : ℝ :=
  2

theorem max_a_is_2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) :
  max_value_of_a a b c h1 h2 = 2 :=
sorry

end NUMINAMATH_GPT_max_a_is_2_l502_50233


namespace NUMINAMATH_GPT_Carol_weight_equals_nine_l502_50286

-- conditions in Lean definitions
def Mildred_weight : ℤ := 59
def weight_difference : ℤ := 50

-- problem statement to prove in Lean 4
theorem Carol_weight_equals_nine (Carol_weight : ℤ) :
  Mildred_weight = Carol_weight + weight_difference → Carol_weight = 9 :=
by
  sorry

end NUMINAMATH_GPT_Carol_weight_equals_nine_l502_50286


namespace NUMINAMATH_GPT_tan_neg_five_pi_over_three_l502_50231

theorem tan_neg_five_pi_over_three : Real.tan (-5 * Real.pi / 3) = Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_neg_five_pi_over_three_l502_50231


namespace NUMINAMATH_GPT_ab_range_l502_50245

variable (a b : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variable (h_eq : a * b = a + b + 8)

theorem ab_range (h : a * b = a + b + 8) : 16 ≤ a * b :=
by sorry

end NUMINAMATH_GPT_ab_range_l502_50245


namespace NUMINAMATH_GPT_am_gm_inequality_l502_50275

noncomputable def arithmetic_mean (a c : ℝ) : ℝ := (a + c) / 2

noncomputable def geometric_mean (a c : ℝ) : ℝ := Real.sqrt (a * c)

theorem am_gm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  (arithmetic_mean a c - geometric_mean a c < (c - a)^2 / (8 * a)) :=
sorry

end NUMINAMATH_GPT_am_gm_inequality_l502_50275


namespace NUMINAMATH_GPT_value_of_a_plus_b_minus_c_l502_50221

theorem value_of_a_plus_b_minus_c (a b c : ℝ) 
  (h1 : abs a = 1) 
  (h2 : abs b = 2) 
  (h3 : abs c = 3) 
  (h4 : a > b) 
  (h5 : b > c) : 
  a + b - c = 2 := 
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_minus_c_l502_50221


namespace NUMINAMATH_GPT_length_ac_l502_50230

theorem length_ac (a b c d e : ℝ) (h1 : bc = 3 * cd) (h2 : de = 7) (h3 : ab = 5) (h4 : ae = 20) :
    ac = 11 :=
by
  sorry

end NUMINAMATH_GPT_length_ac_l502_50230
