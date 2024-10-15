import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_complete_square_l1350_135004

theorem quadratic_complete_square :
  ∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l1350_135004


namespace NUMINAMATH_CALUDE_jacksons_entertainment_spending_l1350_135070

/-- The total amount Jackson spent on entertainment -/
def total_spent (computer_game_price movie_ticket_price number_of_tickets : ℕ) : ℕ :=
  computer_game_price + movie_ticket_price * number_of_tickets

/-- Theorem stating that Jackson's total entertainment spending is $102 -/
theorem jacksons_entertainment_spending :
  total_spent 66 12 3 = 102 := by
  sorry

end NUMINAMATH_CALUDE_jacksons_entertainment_spending_l1350_135070


namespace NUMINAMATH_CALUDE_basketball_games_l1350_135086

theorem basketball_games (x : ℕ) : 
  (3 * x / 4 : ℚ) = x * 3 / 4 ∧ 
  (2 * (x + 10) / 3 : ℚ) = x * 3 / 4 + 5 → 
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_games_l1350_135086


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1350_135069

theorem rationalize_denominator :
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1350_135069


namespace NUMINAMATH_CALUDE_division_problem_l1350_135023

theorem division_problem : (160 : ℝ) / (10 + 11 * 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1350_135023


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1350_135058

/-- An arithmetic sequence with common difference d and first term a₁ -/
def arithmetic_sequence (d a₁ : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

/-- Theorem: For an arithmetic sequence {aₙ} where the common difference d ≠ 0 and
    the first term a₁ ≠ 0, if a₂, a₄, a₈ form a geometric sequence,
    then (a₁ + a₅ + a₉) / (a₂ + a₃) = 3 -/
theorem arithmetic_geometric_ratio
  (d a₁ : ℝ)
  (hd : d ≠ 0)
  (ha₁ : a₁ ≠ 0)
  (h_geom : (arithmetic_sequence d a₁ 4) ^ 2 = 
            (arithmetic_sequence d a₁ 2) * (arithmetic_sequence d a₁ 8)) :
  (arithmetic_sequence d a₁ 1 + arithmetic_sequence d a₁ 5 + arithmetic_sequence d a₁ 9) /
  (arithmetic_sequence d a₁ 2 + arithmetic_sequence d a₁ 3) = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1350_135058


namespace NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_iff_product_neg_l1350_135030

theorem abs_sum_lt_sum_abs_iff_product_neg (a b : ℝ) :
  |a + b| < |a| + |b| ↔ a * b < 0 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_iff_product_neg_l1350_135030


namespace NUMINAMATH_CALUDE_fraction_product_proof_l1350_135059

theorem fraction_product_proof :
  (8 / 4) * (10 / 25) * (27 / 18) * (16 / 24) * (35 / 21) * (30 / 50) * (14 / 7) * (20 / 40) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_proof_l1350_135059


namespace NUMINAMATH_CALUDE_pentadecagon_diagonals_l1350_135083

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentadecagon is a 15-sided polygon -/
def pentadecagon_sides : ℕ := 15

/-- Theorem: The number of diagonals in a pentadecagon is 90 -/
theorem pentadecagon_diagonals : num_diagonals pentadecagon_sides = 90 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_diagonals_l1350_135083


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l1350_135039

theorem sum_of_four_numbers : 1432 + 3214 + 2143 + 4321 = 11110 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l1350_135039


namespace NUMINAMATH_CALUDE_pizza_area_increase_l1350_135094

theorem pizza_area_increase : 
  let small_diameter : ℝ := 12
  let large_diameter : ℝ := 18
  let small_area := Real.pi * (small_diameter / 2)^2
  let large_area := Real.pi * (large_diameter / 2)^2
  let area_increase := large_area - small_area
  let percent_increase := (area_increase / small_area) * 100
  percent_increase = 125 :=
by sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l1350_135094


namespace NUMINAMATH_CALUDE_P_equals_set_l1350_135081

def P : Set ℝ := {x | x^2 = 1}

theorem P_equals_set : P = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_P_equals_set_l1350_135081


namespace NUMINAMATH_CALUDE_lemonade_solution_water_amount_l1350_135029

/-- The amount of lemonade syrup in the solution -/
def lemonade_syrup : ℝ := 7

/-- The amount of solution removed and replaced with water -/
def removed_amount : ℝ := 2.1428571428571423

/-- The desired concentration of lemonade syrup after adjustment -/
def desired_concentration : ℝ := 0.20

/-- The amount of water in the original solution -/
def water_amount : ℝ := 25.857142857142854

theorem lemonade_solution_water_amount :
  (lemonade_syrup / (lemonade_syrup + water_amount + removed_amount) = desired_concentration) :=
by sorry

end NUMINAMATH_CALUDE_lemonade_solution_water_amount_l1350_135029


namespace NUMINAMATH_CALUDE_exponent_calculations_l1350_135013

theorem exponent_calculations (x m : ℝ) (hx : x ≠ 0) (hm : m ≠ 0) :
  (x^7 / x^3 * x^4 = x^8) ∧ (m * m^3 + (-m^2)^3 / m^2 = 0) := by sorry

end NUMINAMATH_CALUDE_exponent_calculations_l1350_135013


namespace NUMINAMATH_CALUDE_runner_stops_in_quarter_A_l1350_135011

def track_circumference : ℝ := 80
def distance_run : ℝ := 2000
def num_quarters : ℕ := 4

theorem runner_stops_in_quarter_A :
  ∀ (start_point : ℝ) (quarters : Fin num_quarters),
  start_point ∈ Set.Icc 0 track_circumference →
  ∃ (n : ℕ), distance_run = n * track_circumference + start_point :=
by sorry

end NUMINAMATH_CALUDE_runner_stops_in_quarter_A_l1350_135011


namespace NUMINAMATH_CALUDE_M_intersect_N_l1350_135006

def M : Set Int := {m | -3 < m ∧ m < 2}
def N : Set Int := {n | -1 ≤ n ∧ n ≤ 3}

theorem M_intersect_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1350_135006


namespace NUMINAMATH_CALUDE_commute_days_l1350_135001

theorem commute_days (x : ℕ) 
  (h1 : x > 0)
  (h2 : 2 * x = 9 + 8 + 15) : 
  x = 16 := by
sorry

end NUMINAMATH_CALUDE_commute_days_l1350_135001


namespace NUMINAMATH_CALUDE_cube_of_product_equality_l1350_135032

theorem cube_of_product_equality (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_equality_l1350_135032


namespace NUMINAMATH_CALUDE_completing_square_result_l1350_135028

theorem completing_square_result (x : ℝ) :
  x^2 - 4*x + 2 = 0 → (x - 2)^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_completing_square_result_l1350_135028


namespace NUMINAMATH_CALUDE_line_equation_l1350_135063

-- Define the points A and B
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (1, 4)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the property of having equal intercepts on both axes
def equal_intercepts (a b c : ℝ) : Prop := ∃ k : ℝ, a * k = c ∧ b * k = c ∧ k ≠ 0

-- Define the main theorem
theorem line_equation :
  ∃ (a b c : ℝ),
    -- The line passes through point A
    (a * A.1 + b * A.2 + c = 0) ∧
    -- The line is perpendicular to 2x + y - 5 = 0
    (a * 2 + b * 1 = 0) ∧
    -- The line passes through point B
    (a * B.1 + b * B.2 + c = 0) ∧
    -- The line has equal intercepts on both axes
    (equal_intercepts a b c) ∧
    -- The equation of the line is either x + y - 5 = 0 or 4x - y = 0
    ((a = 1 ∧ b = 1 ∧ c = -5) ∨ (a = 4 ∧ b = -1 ∧ c = 0)) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1350_135063


namespace NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l1350_135088

theorem log_sqrt10_1000sqrt10 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l1350_135088


namespace NUMINAMATH_CALUDE_basketball_league_games_l1350_135010

/-- The number of games played in a basketball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264. -/
theorem basketball_league_games :
  total_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_basketball_league_games_l1350_135010


namespace NUMINAMATH_CALUDE_one_third_of_390_l1350_135015

theorem one_third_of_390 : (1 / 3 : ℚ) * 390 = 130 := by sorry

end NUMINAMATH_CALUDE_one_third_of_390_l1350_135015


namespace NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l1350_135084

theorem lcm_of_ratio_numbers (a b : ℕ) (h1 : a = 21) (h2 : 4 * a = 3 * b) : 
  Nat.lcm a b = 84 := by sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l1350_135084


namespace NUMINAMATH_CALUDE_D_most_stable_l1350_135031

-- Define the variances for each person
def variance_A : ℝ := 0.56
def variance_B : ℝ := 0.60
def variance_C : ℝ := 0.50
def variance_D : ℝ := 0.45

-- Define a function to determine if one variance is more stable than another
def more_stable (v1 v2 : ℝ) : Prop := v1 < v2

-- Theorem stating that D has the most stable performance
theorem D_most_stable :
  more_stable variance_D variance_C ∧
  more_stable variance_D variance_A ∧
  more_stable variance_D variance_B :=
by sorry

end NUMINAMATH_CALUDE_D_most_stable_l1350_135031


namespace NUMINAMATH_CALUDE_total_pictures_taken_l1350_135016

def pictures_already_taken : ℕ := 28
def pictures_at_dolphin_show : ℕ := 16

theorem total_pictures_taken : 
  pictures_already_taken + pictures_at_dolphin_show = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_pictures_taken_l1350_135016


namespace NUMINAMATH_CALUDE_cosine_sine_relation_l1350_135008

theorem cosine_sine_relation (x : ℝ) :
  2 * Real.cos x + 3 * Real.sin x = 4 →
  Real.cos x = 8 / 13 ∧ Real.sin x = 12 / 13 →
  3 * Real.cos x - 2 * Real.sin x = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_sine_relation_l1350_135008


namespace NUMINAMATH_CALUDE_garden_flowers_l1350_135076

theorem garden_flowers (red_flowers : ℕ) (additional_red : ℕ) (white_flowers : ℕ) :
  red_flowers = 347 →
  additional_red = 208 →
  white_flowers = red_flowers + additional_red →
  white_flowers = 555 := by
  sorry

end NUMINAMATH_CALUDE_garden_flowers_l1350_135076


namespace NUMINAMATH_CALUDE_symmetric_periodic_function_max_period_l1350_135073

/-- A function with symmetry around x=1 and x=8, and a periodic property -/
def SymmetricPeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T > 0 ∧
  (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x : ℝ, f (1 + x) = f (1 - x)) ∧
  (∀ x : ℝ, f (8 + x) = f (8 - x)) ∧
  ∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T'

theorem symmetric_periodic_function_max_period :
  ∀ f : ℝ → ℝ, SymmetricPeriodicFunction f →
  ∃ T : ℝ, T > 0 ∧ SymmetricPeriodicFunction f ∧ T = 14 ∧
  ∀ T' : ℝ, T' > 0 → SymmetricPeriodicFunction f → T' ≤ T :=
sorry

end NUMINAMATH_CALUDE_symmetric_periodic_function_max_period_l1350_135073


namespace NUMINAMATH_CALUDE_product_of_one_plus_tans_l1350_135020

theorem product_of_one_plus_tans (α β : Real) (h : α + β = π / 4) :
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_tans_l1350_135020


namespace NUMINAMATH_CALUDE_cos_three_pi_four_plus_two_alpha_l1350_135002

theorem cos_three_pi_four_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 8 - α) = 1 / 6) : 
  Real.cos (3 * π / 4 + 2 * α) = 17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_four_plus_two_alpha_l1350_135002


namespace NUMINAMATH_CALUDE_fraction_not_on_time_l1350_135009

/-- Represents the fraction of attendees who are male -/
def male_fraction : ℚ := 3/5

/-- Represents the fraction of male attendees who arrived on time -/
def male_on_time : ℚ := 7/8

/-- Represents the fraction of female attendees who arrived on time -/
def female_on_time : ℚ := 4/5

/-- Theorem stating that the fraction of attendees who did not arrive on time is 3/20 -/
theorem fraction_not_on_time : 
  1 - (male_fraction * male_on_time + (1 - male_fraction) * female_on_time) = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_on_time_l1350_135009


namespace NUMINAMATH_CALUDE_centimeters_per_kilometer_l1350_135041

-- Define the conversion factors
def meters_per_kilometer : ℝ := 1000
def centimeters_per_meter : ℝ := 100

-- Theorem statement
theorem centimeters_per_kilometer : 
  meters_per_kilometer * centimeters_per_meter = 100000 := by
  sorry

end NUMINAMATH_CALUDE_centimeters_per_kilometer_l1350_135041


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l1350_135099

-- Define the position function
def s (t : ℝ) : ℝ := 5 * t^2

-- Define the velocity function as the derivative of the position function
def v (t : ℝ) : ℝ := 10 * t

-- Theorem stating that the instantaneous velocity at t=3 is 30
theorem instantaneous_velocity_at_3 : v 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l1350_135099


namespace NUMINAMATH_CALUDE_persimmon_count_l1350_135074

theorem persimmon_count (total : ℕ) (difference : ℕ) (persimmons : ℕ) (tangerines : ℕ) : 
  total = 129 →
  difference = 43 →
  total = persimmons + tangerines →
  persimmons + difference = tangerines →
  persimmons = 43 := by
sorry

end NUMINAMATH_CALUDE_persimmon_count_l1350_135074


namespace NUMINAMATH_CALUDE_sqrt_eight_combinable_with_sqrt_two_l1350_135090

theorem sqrt_eight_combinable_with_sqrt_two :
  ∃ (n : ℤ), Real.sqrt 8 = n * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_eight_combinable_with_sqrt_two_l1350_135090


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l1350_135077

/-- Given a paint mixture with a ratio of red:blue:white as 5:3:7,
    if 21 quarts of white paint are used, then 15 quarts of red paint should be used. -/
theorem paint_mixture_ratio (red blue white : ℚ) (h1 : red / white = 5 / 7) (h2 : white = 21) :
  red = 15 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l1350_135077


namespace NUMINAMATH_CALUDE_sqrt_40_between_6_and_7_l1350_135089

theorem sqrt_40_between_6_and_7 :
  ∃ (x : ℝ), x = Real.sqrt 40 ∧ 6 < x ∧ x < 7 :=
by
  have h1 : Real.sqrt 36 < Real.sqrt 40 ∧ Real.sqrt 40 < Real.sqrt 49 := by sorry
  sorry

end NUMINAMATH_CALUDE_sqrt_40_between_6_and_7_l1350_135089


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_unique_intersection_point_l1350_135019

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (50/17, 24/17)

/-- First line equation: 2y = 3x - 6 -/
def line1 (x y : ℚ) : Prop := 2 * y = 3 * x - 6

/-- Second line equation: x + 5y = 10 -/
def line2 (x y : ℚ) : Prop := x + 5 * y = 10

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations : 
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_unique_intersection_point_l1350_135019


namespace NUMINAMATH_CALUDE_lineman_drinks_eight_ounces_l1350_135025

/-- Represents the water consumption scenario of a football team -/
structure WaterConsumption where
  linemen_count : ℕ
  skill_players_count : ℕ
  cooler_capacity : ℕ
  skill_player_consumption : ℕ
  waiting_skill_players : ℕ

/-- Calculates the amount of water each lineman drinks -/
def lineman_consumption (wc : WaterConsumption) : ℚ :=
  let skill_players_drinking := wc.skill_players_count - wc.waiting_skill_players
  let total_skill_consumption := skill_players_drinking * wc.skill_player_consumption
  (wc.cooler_capacity - total_skill_consumption) / wc.linemen_count

/-- Theorem stating that each lineman drinks 8 ounces of water -/
theorem lineman_drinks_eight_ounces (wc : WaterConsumption) 
  (h1 : wc.linemen_count = 12)
  (h2 : wc.skill_players_count = 10)
  (h3 : wc.cooler_capacity = 126)
  (h4 : wc.skill_player_consumption = 6)
  (h5 : wc.waiting_skill_players = 5) :
  lineman_consumption wc = 8 := by
  sorry

#eval lineman_consumption {
  linemen_count := 12,
  skill_players_count := 10,
  cooler_capacity := 126,
  skill_player_consumption := 6,
  waiting_skill_players := 5
}

end NUMINAMATH_CALUDE_lineman_drinks_eight_ounces_l1350_135025


namespace NUMINAMATH_CALUDE_andy_math_problem_l1350_135048

theorem andy_math_problem (start_num end_num count : ℕ) : 
  end_num = 125 → count = 46 → end_num - start_num + 1 = count → start_num = 80 := by
  sorry

end NUMINAMATH_CALUDE_andy_math_problem_l1350_135048


namespace NUMINAMATH_CALUDE_function_difference_inequality_l1350_135061

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivative condition
variable (h : ∀ x, HasDerivAt f (f' x) x ∧ HasDerivAt g (g' x) x ∧ f' x > g' x)

-- State the theorem
theorem function_difference_inequality (x₁ x₂ : ℝ) (h_lt : x₁ < x₂) :
  f x₁ - f x₂ < g x₁ - g x₂ := by
  sorry

end NUMINAMATH_CALUDE_function_difference_inequality_l1350_135061


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1350_135062

theorem negation_of_proposition :
  (¬(∀ x y : ℝ, x > 0 ∧ y > 0 → x + y > 0)) ↔
  (∀ x y : ℝ, ¬(x > 0 ∧ y > 0) → x + y ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1350_135062


namespace NUMINAMATH_CALUDE_equal_positive_reals_from_inequalities_l1350_135098

theorem equal_positive_reals_from_inequalities 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0) (pos₄ : x₄ > 0) (pos₅ : x₅ > 0)
  (ineq₁ : (x₁^2 - x₃*x₃)*(x₂^2 - x₃*x₃) ≤ 0)
  (ineq₂ : (x₃^2 - x₁*x₁)*(x₃^2 - x₁*x₁) ≤ 0)
  (ineq₃ : (x₃^2 - x₃*x₂)*(x₁^2 - x₃*x₂) ≤ 0)
  (ineq₄ : (x₁^2 - x₁*x₃)*(x₃^2 - x₁*x₃) ≤ 0)
  (ineq₅ : (x₃^2 - x₂*x₁)*(x₁^2 - x₂*x₁) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ := by
  sorry


end NUMINAMATH_CALUDE_equal_positive_reals_from_inequalities_l1350_135098


namespace NUMINAMATH_CALUDE_expression_value_l1350_135051

theorem expression_value (x : ℝ) : x = 2 → 3 * x^2 - 4 * x + 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1350_135051


namespace NUMINAMATH_CALUDE_angle_C_measure_l1350_135005

structure Quadrilateral where
  A : Real
  B : Real
  C : Real
  D : Real
  sum_angles : A + B + C + D = 360

def adjacent_angle_ratio (q : Quadrilateral) : Prop :=
  q.A / q.B = 2 / 7

theorem angle_C_measure (q : Quadrilateral) (h : adjacent_angle_ratio q) : q.C = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l1350_135005


namespace NUMINAMATH_CALUDE_tangled_legs_scenario_l1350_135096

/-- The number of legs tangled in leashes when two dog walkers meet --/
def tangled_legs (dogs_group1 : ℕ) (dogs_group2 : ℕ) (legs_per_dog : ℕ) (walkers : ℕ) (legs_per_walker : ℕ) : ℕ :=
  (dogs_group1 + dogs_group2) * legs_per_dog + walkers * legs_per_walker

/-- Theorem stating the number of legs tangled in leashes in the given scenario --/
theorem tangled_legs_scenario : tangled_legs 5 3 4 2 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_tangled_legs_scenario_l1350_135096


namespace NUMINAMATH_CALUDE_rectangle_count_in_5x5_grid_l1350_135007

/-- The number of ways to select a rectangle in a 5x5 grid -/
def rectangleCount : ℕ := 225

/-- The number of horizontal or vertical lines in a 5x5 grid, including boundaries -/
def lineCount : ℕ := 6

theorem rectangle_count_in_5x5_grid :
  rectangleCount = (lineCount.choose 2) * (lineCount.choose 2) :=
sorry

end NUMINAMATH_CALUDE_rectangle_count_in_5x5_grid_l1350_135007


namespace NUMINAMATH_CALUDE_no_valid_operation_l1350_135071

-- Define the type for standard arithmetic operations
inductive ArithOp
  | Add
  | Sub
  | Mul
  | Div

def applyOp (op : ArithOp) (a b : Int) : Int :=
  match op with
  | ArithOp.Add => a + b
  | ArithOp.Sub => a - b
  | ArithOp.Mul => a * b
  | ArithOp.Div => a / b

theorem no_valid_operation :
  ∀ (op : ArithOp), (applyOp op 8 4) + 5 - (3 - 2) ≠ 4 := by
  sorry

#check no_valid_operation

end NUMINAMATH_CALUDE_no_valid_operation_l1350_135071


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1350_135024

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a ≥ 6 ∧
  ((a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a = 6 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1350_135024


namespace NUMINAMATH_CALUDE_cody_payment_proof_l1350_135068

def initial_purchase : ℝ := 40
def tax_rate : ℝ := 0.05
def discount : ℝ := 8
def cody_payment : ℝ := 17

theorem cody_payment_proof :
  cody_payment = (initial_purchase * (1 + tax_rate) - discount) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cody_payment_proof_l1350_135068


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1350_135017

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ [1, 2] → x^2 < 4) ↔ (∃ x : ℝ, x ∈ [1, 2] ∧ x^2 ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1350_135017


namespace NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l1350_135060

/-- The decimal representation of a real number with a single repeating digit. -/
def repeatingDecimal (digit : ℕ) : ℚ :=
  (digit : ℚ) / 9

/-- Prove that the repeating decimal 0.666... is equal to 2/3 -/
theorem repeating_six_equals_two_thirds :
  repeatingDecimal 6 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l1350_135060


namespace NUMINAMATH_CALUDE_min_value_expression_l1350_135055

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 48) :
  x^2 + 6*x*y + 9*y^2 + 4*z^2 ≥ 128 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 48 ∧ x₀^2 + 6*x₀*y₀ + 9*y₀^2 + 4*z₀^2 = 128 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1350_135055


namespace NUMINAMATH_CALUDE_min_distance_parabola_circle_l1350_135054

/-- The minimum distance between a point on the parabola y^2 = 6x and a point on the circle (x-4)^2 + y^2 = 1 is √15 - 1 -/
theorem min_distance_parabola_circle :
  ∃ (d : ℝ), d = Real.sqrt 15 - 1 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    y₁^2 = 6*x₁ →
    (x₂ - 4)^2 + y₂^2 = 1 →
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 : ℝ) ≥ d^2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parabola_circle_l1350_135054


namespace NUMINAMATH_CALUDE_M_value_l1350_135085

def M : ℕ → ℕ 
  | 0 => 0
  | n + 1 => (2*n + 2)^2 + (2*n)^2 - (2*n - 2)^2 + M n

theorem M_value : M 25 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_M_value_l1350_135085


namespace NUMINAMATH_CALUDE_cube_minimizes_edge_sum_squares_l1350_135033

/-- A parallelepiped with edges a, b, c and volume V -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  V : ℝ
  volume_eq : a * b * c = V
  positive : 0 < a ∧ 0 < b ∧ 0 < c

/-- The sum of squares of edges meeting at one vertex -/
def edge_sum_squares (p : Parallelepiped) : ℝ := p.a^2 + p.b^2 + p.c^2

/-- Theorem: The cube minimizes the sum of squares of edges among parallelepipeds of equal volume -/
theorem cube_minimizes_edge_sum_squares (V : ℝ) (hV : 0 < V) :
  ∀ p : Parallelepiped, p.V = V →
  edge_sum_squares p ≥ 3 * V^(2/3) ∧
  (edge_sum_squares p = 3 * V^(2/3) ↔ p.a = p.b ∧ p.b = p.c) :=
sorry


end NUMINAMATH_CALUDE_cube_minimizes_edge_sum_squares_l1350_135033


namespace NUMINAMATH_CALUDE_triangle_theorem_l1350_135079

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def triangle_condition (t : Triangle) : Prop :=
  t.a * Real.sin t.A * Real.sin t.B + t.b * (Real.cos t.A)^2 = 4/3 * t.a

/-- The additional condition for part 2 -/
def additional_condition (t : Triangle) : Prop :=
  t.c^2 = t.a^2 + 1/4 * t.b^2

theorem triangle_theorem (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : additional_condition t) : 
  t.b / t.a = 4/3 ∧ t.C = π/3 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l1350_135079


namespace NUMINAMATH_CALUDE_fifteen_percent_greater_l1350_135053

theorem fifteen_percent_greater : ∃ (x : ℝ), (15 / 100 * 40 = 25 / 100 * x + 2) ∧ (x = 16) := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_greater_l1350_135053


namespace NUMINAMATH_CALUDE_unit_circle_sector_angle_l1350_135052

/-- The radian measure of a central angle in a unit circle, given the area of the sector -/
def central_angle (area : ℝ) : ℝ := 2 * area

theorem unit_circle_sector_angle (area : ℝ) (h : area = 1) : 
  central_angle area = 2 := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_sector_angle_l1350_135052


namespace NUMINAMATH_CALUDE_largest_four_digit_multiple_of_48_l1350_135091

theorem largest_four_digit_multiple_of_48 : 
  (∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 48 = 0 → n ≤ 9984) ∧ 
  9984 % 48 = 0 ∧ 
  9984 ≤ 9999 ∧ 
  9984 ≥ 1000 := by
sorry

end NUMINAMATH_CALUDE_largest_four_digit_multiple_of_48_l1350_135091


namespace NUMINAMATH_CALUDE_bryans_annual_travel_time_l1350_135035

/-- Represents the time in minutes for each leg of Bryan's journey --/
structure JourneyTime where
  walkToBus : ℕ
  busRide : ℕ
  walkToJob : ℕ

/-- Calculates the total annual travel time in hours --/
def annualTravelTime (j : JourneyTime) (daysWorked : ℕ) : ℕ :=
  let totalDailyMinutes := 2 * (j.walkToBus + j.busRide + j.walkToJob)
  (totalDailyMinutes * daysWorked) / 60

/-- Theorem stating that Bryan spends 365 hours per year traveling to and from work --/
theorem bryans_annual_travel_time :
  let j : JourneyTime := { walkToBus := 5, busRide := 20, walkToJob := 5 }
  annualTravelTime j 365 = 365 := by
  sorry

end NUMINAMATH_CALUDE_bryans_annual_travel_time_l1350_135035


namespace NUMINAMATH_CALUDE_finite_zero_additions_l1350_135034

/-- Represents the state of the board at any given time -/
def BoardState := List ℕ

/-- The process of updating the board -/
def update_board (a b : ℕ) (state : BoardState) : BoardState :=
  sorry

/-- Predicate to check if we need to add two zeros -/
def need_zeros (state : BoardState) : Prop :=
  sorry

/-- The main theorem statement -/
theorem finite_zero_additions (a b : ℕ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    let state := (update_board a b)^[n] []
    ¬(need_zeros state) :=
  sorry

end NUMINAMATH_CALUDE_finite_zero_additions_l1350_135034


namespace NUMINAMATH_CALUDE_point_division_l1350_135000

/-- Given a line segment AB and a point P on AB such that AP:PB = 3:5,
    prove that P can be expressed as a linear combination of A and B with coefficients 5/8 and 3/8 respectively. -/
theorem point_division (A B P : ℝ × ℝ) : 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) →  -- P is on line segment AB
  (dist A P) / (dist P B) = 3 / 5 →                     -- AP:PB = 3:5
  P = (5/8) • A + (3/8) • B :=                          -- P = (5/8)A + (3/8)B
by sorry

end NUMINAMATH_CALUDE_point_division_l1350_135000


namespace NUMINAMATH_CALUDE_coeff_x_cubed_in_product_l1350_135018

def p (x : ℝ) : ℝ := x^5 - 4*x^3 + 3*x^2 - 2*x + 1
def q (x : ℝ) : ℝ := 3*x^3 - 2*x^2 + x + 5

theorem coeff_x_cubed_in_product (x : ℝ) :
  ∃ (a b c d e : ℝ), p x * q x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + (p 0 * q 0) ∧ c = -10 :=
sorry

end NUMINAMATH_CALUDE_coeff_x_cubed_in_product_l1350_135018


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l1350_135067

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  center : Point
  width : ℝ
  height : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a right triangle in 2D space -/
structure RightTriangle where
  vertex : Point
  leg1 : ℝ
  leg2 : ℝ

/-- Calculate the area of intersection between a rectangle, circle, and right triangle -/
def areaOfIntersection (rect : Rectangle) (circ : Circle) (tri : RightTriangle) : ℝ :=
  sorry

theorem intersection_area_theorem (rect : Rectangle) (circ : Circle) (tri : RightTriangle) :
  rect.width = 10 →
  rect.height = 4 →
  circ.radius = 4 →
  rect.center = circ.center →
  tri.leg1 = 3 →
  tri.leg2 = 3 →
  -- Assuming the triangle is positioned correctly
  areaOfIntersection rect circ tri = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l1350_135067


namespace NUMINAMATH_CALUDE_maria_workday_end_l1350_135014

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

def Time.add (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes
  let newHours := (t.hours + d.hours + totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, by sorry, by sorry⟩

def Duration.add (d1 d2 : Duration) : Duration :=
  let totalMinutes := d1.minutes + d2.minutes + (d1.hours + d2.hours) * 60
  ⟨totalMinutes / 60, totalMinutes % 60⟩

def workDay (start : Time) (workDuration : Duration) (lunchStart : Time) (lunchDuration : Duration) (breakStart : Time) (breakDuration : Duration) : Time :=
  let lunchEnd := lunchStart.add lunchDuration
  let breakEnd := breakStart.add breakDuration
  let totalBreakDuration := Duration.add lunchDuration breakDuration
  start.add (Duration.add workDuration totalBreakDuration)

theorem maria_workday_end :
  let start : Time := ⟨8, 0, by sorry, by sorry⟩
  let workDuration : Duration := ⟨8, 0⟩
  let lunchStart : Time := ⟨13, 0, by sorry, by sorry⟩
  let lunchDuration : Duration := ⟨1, 0⟩
  let breakStart : Time := ⟨15, 30, by sorry, by sorry⟩
  let breakDuration : Duration := ⟨0, 15⟩
  let endTime : Time := workDay start workDuration lunchStart lunchDuration breakStart breakDuration
  endTime = ⟨18, 0, by sorry, by sorry⟩ := by
  sorry


end NUMINAMATH_CALUDE_maria_workday_end_l1350_135014


namespace NUMINAMATH_CALUDE_sum_base4_numbers_l1350_135078

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4^i)) 0

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem sum_base4_numbers : 
  let a := [2, 0, 2]  -- 202₄
  let b := [0, 3, 3]  -- 330₄
  let c := [0, 0, 0, 1]  -- 1000₄
  let sum_base10 := base4ToBase10 a + base4ToBase10 b + base4ToBase10 c
  base10ToBase4 sum_base10 = [2, 3, 1, 2] ∧ sum_base10 = 158 := by
  sorry

end NUMINAMATH_CALUDE_sum_base4_numbers_l1350_135078


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1350_135047

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The main theorem: For a geometric sequence satisfying given conditions, a₂ + a₆ = 34 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsGeometricSequence a →
  a 3 + a 5 = 20 →
  a 4 = 8 →
  a 2 + a 6 = 34 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l1350_135047


namespace NUMINAMATH_CALUDE_exists_polygon_with_1980_degrees_l1350_135040

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- 1980 degrees is a valid sum of interior angles for some polygon -/
theorem exists_polygon_with_1980_degrees :
  ∃ (n : ℕ), sum_interior_angles n = 1980 :=
sorry

end NUMINAMATH_CALUDE_exists_polygon_with_1980_degrees_l1350_135040


namespace NUMINAMATH_CALUDE_hyperbola_tangent_intersection_product_l1350_135057

/-- The hyperbola with equation x²/4 - y² = 1 -/
def hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) - p.2^2 = 1}

/-- The asymptotes of the hyperbola -/
def asymptotes : Set (Set (ℝ × ℝ)) :=
  {{p : ℝ × ℝ | p.2 = p.1 / 2}, {p : ℝ × ℝ | p.2 = -p.1 / 2}}

/-- A line tangent to the hyperbola at point P -/
def tangent_line (P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {Q : ℝ × ℝ | ∃ t : ℝ, Q = (P.1 + t, P.2 + t * (P.2 / P.1))}

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem hyperbola_tangent_intersection_product (P : ℝ × ℝ) 
  (h_P : P ∈ hyperbola) 
  (M N : ℝ × ℝ) 
  (h_M : M ∈ (tangent_line P ∩ (⋃₀ asymptotes))) 
  (h_N : N ∈ (tangent_line P ∩ (⋃₀ asymptotes))) 
  (h_M_ne_N : M ≠ N) :
  dot_product M N = 3 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_tangent_intersection_product_l1350_135057


namespace NUMINAMATH_CALUDE_probability_of_losing_is_one_third_l1350_135027

/-- A game where a single standard die is rolled once -/
structure DieGame where
  /-- The set of all possible outcomes when rolling a standard die -/
  outcomes : Finset Nat
  /-- The set of losing outcomes -/
  losing_outcomes : Finset Nat
  /-- Assumption that outcomes are the numbers 1 to 6 -/
  outcomes_def : outcomes = Finset.range 6
  /-- Assumption that losing outcomes are 5 and 6 -/
  losing_def : losing_outcomes = {5, 6}

/-- The probability of losing in the die game -/
def probability_of_losing (game : DieGame) : ℚ :=
  (game.losing_outcomes.card : ℚ) / (game.outcomes.card : ℚ)

/-- Theorem stating that the probability of losing is 1/3 -/
theorem probability_of_losing_is_one_third (game : DieGame) :
    probability_of_losing game = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_losing_is_one_third_l1350_135027


namespace NUMINAMATH_CALUDE_trajectory_of_G_no_perpendicular_bisector_l1350_135072

-- Define the circle M
def circle_M (m n r : ℝ) (x y : ℝ) : Prop :=
  (x - m)^2 + (y - n)^2 = r^2

-- Define point N
def point_N : ℝ × ℝ := (1, 0)

-- Define the conditions for points P, Q, and G
def point_conditions (m n r : ℝ) (P Q G : ℝ × ℝ) : Prop :=
  circle_M m n r P.1 P.2 ∧
  (∃ t : ℝ, Q = point_N + t • (P - point_N)) ∧
  (∃ s : ℝ, G = (m, n) + s • (P - (m, n))) ∧
  P - point_N = 2 • (Q - point_N) ∧
  (G - Q) • (P - point_N) = 0

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Theorem 1: Trajectory of G is the ellipse C
theorem trajectory_of_G (P Q G : ℝ × ℝ) :
  point_conditions (-1) 0 4 P Q G →
  trajectory_C G.1 G.2 :=
sorry

-- Theorem 2: No positive m, n, r exist such that MN perpendicularly bisects AB
theorem no_perpendicular_bisector :
  ¬ ∃ (m n r : ℝ) (A B : ℝ × ℝ),
    m > 0 ∧ n > 0 ∧ r > 0 ∧
    circle_M m n r A.1 A.2 ∧
    circle_M m n r B.1 B.2 ∧
    trajectory_C A.1 A.2 ∧
    trajectory_C B.1 B.2 ∧
    A ≠ B ∧
    (∃ (t : ℝ), (A + B) / 2 = point_N + t • ((m, n) - point_N)) ∧
    ((m, n) - point_N) • (B - A) = 0 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_G_no_perpendicular_bisector_l1350_135072


namespace NUMINAMATH_CALUDE_unique_square_number_l1350_135056

/-- A function to convert a two-digit number to its decimal representation -/
def twoDigitToNumber (a b : ℕ) : ℕ := 10 * a + b

/-- A function to convert a three-digit number to its decimal representation -/
def threeDigitToNumber (c₁ c₂ b : ℕ) : ℕ := 100 * c₁ + 10 * c₂ + b

/-- Theorem stating that under given conditions, ccb must be 441 -/
theorem unique_square_number (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c →
  b = 1 →
  0 < a → a < 10 →
  0 ≤ c → c < 10 →
  (twoDigitToNumber a b)^2 = threeDigitToNumber c c b →
  threeDigitToNumber c c b > 300 →
  threeDigitToNumber c c b = 441 := by
sorry

end NUMINAMATH_CALUDE_unique_square_number_l1350_135056


namespace NUMINAMATH_CALUDE_tallest_tree_height_l1350_135095

theorem tallest_tree_height (h_shortest h_middle h_tallest : ℝ) : 
  h_middle = (2/3) * h_tallest →
  h_shortest = (1/2) * h_middle →
  h_shortest = 50 →
  h_tallest = 150 := by
sorry

end NUMINAMATH_CALUDE_tallest_tree_height_l1350_135095


namespace NUMINAMATH_CALUDE_min_cases_for_shirley_order_l1350_135036

/-- Represents the number of boxes of each cookie type sold -/
structure CookiesSold where
  trefoils : Nat
  samoas : Nat
  thinMints : Nat

/-- Represents the composition of each case -/
structure CaseComposition where
  trefoils : Nat
  samoas : Nat
  thinMints : Nat

/-- Calculates the minimum number of cases needed to fulfill the orders -/
def minCasesNeeded (sold : CookiesSold) (composition : CaseComposition) : Nat :=
  max
    (((sold.trefoils + composition.trefoils - 1) / composition.trefoils) : Nat)
    (max
      (((sold.samoas + composition.samoas - 1) / composition.samoas) : Nat)
      (((sold.thinMints + composition.thinMints - 1) / composition.thinMints) : Nat))

theorem min_cases_for_shirley_order :
  let sold : CookiesSold := { trefoils := 54, samoas := 36, thinMints := 48 }
  let composition : CaseComposition := { trefoils := 4, samoas := 3, thinMints := 5 }
  minCasesNeeded sold composition = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_cases_for_shirley_order_l1350_135036


namespace NUMINAMATH_CALUDE_clay_target_permutations_l1350_135049

theorem clay_target_permutations : 
  (Nat.factorial 9) / ((Nat.factorial 3) * (Nat.factorial 3) * (Nat.factorial 3)) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_clay_target_permutations_l1350_135049


namespace NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l1350_135038

def f (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x + k^2

theorem f_monotonicity_and_zeros (k : ℝ) :
  (∀ x y, x < y → k ≤ 0 → f k x < f k y) ∧
  (k > 0 → ∀ x y, (x < y ∧ y < -Real.sqrt (k/3)) ∨ (x < y ∧ x > Real.sqrt (k/3)) → f k x < f k y) ∧
  (k > 0 → ∀ x y, -Real.sqrt (k/3) < x ∧ x < y ∧ y < Real.sqrt (k/3) → f k x > f k y) ∧
  (∃ x y z, x < y ∧ y < z ∧ f k x = 0 ∧ f k y = 0 ∧ f k z = 0 ↔ 0 < k ∧ k < 4/27) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l1350_135038


namespace NUMINAMATH_CALUDE_thief_catch_time_l1350_135066

/-- The time it takes for the passenger to catch the thief -/
def catchUpTime (thief_speed : ℝ) (passenger_speed : ℝ) (bus_speed : ℝ) (stop_time : ℝ) : ℝ :=
  stop_time

theorem thief_catch_time :
  ∀ (thief_speed : ℝ),
    thief_speed > 0 →
    let passenger_speed := 2 * thief_speed
    let bus_speed := 10 * thief_speed
    let stop_time := 40
    catchUpTime thief_speed passenger_speed bus_speed stop_time = 40 :=
by
  sorry

#check thief_catch_time

end NUMINAMATH_CALUDE_thief_catch_time_l1350_135066


namespace NUMINAMATH_CALUDE_product_bounds_l1350_135082

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  x + y = a ∧ x^2 + y^2 = -a^2 + 2

-- Define the product function
def product (x y : ℝ) : ℝ := x * y

-- Theorem statement
theorem product_bounds :
  ∀ x y a : ℝ, system x y a → 
    (∃ x' y' a' : ℝ, system x' y' a' ∧ product x' y' = 1/3) ∧
    (∃ x'' y'' a'' : ℝ, system x'' y'' a'' ∧ product x'' y'' = -1) ∧
    (∀ x''' y''' a''' : ℝ, system x''' y''' a''' → 
      -1 ≤ product x''' y''' ∧ product x''' y''' ≤ 1/3) :=
sorry

end NUMINAMATH_CALUDE_product_bounds_l1350_135082


namespace NUMINAMATH_CALUDE_problem_part_1_problem_part_2_l1350_135037

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a
def g (x : ℝ) : ℝ := |x + 1| + |x - 2|

theorem problem_part_1 :
  {x : ℝ | f (-4) x ≥ g x} = {x : ℝ | x ≤ -1 - Real.sqrt 6 ∨ x ≥ 3} := by sorry

theorem problem_part_2 (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f a x ≤ g x) → a ≤ -4 := by sorry

end NUMINAMATH_CALUDE_problem_part_1_problem_part_2_l1350_135037


namespace NUMINAMATH_CALUDE_product_equals_zero_l1350_135065

theorem product_equals_zero (b : ℤ) (h : b = 3) : 
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b * (b + 1) * (b + 2) = 0 := by
sorry

end NUMINAMATH_CALUDE_product_equals_zero_l1350_135065


namespace NUMINAMATH_CALUDE_test_score_problem_l1350_135042

theorem test_score_problem (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (total_score : ℤ) :
  total_questions = 30 →
  correct_points = 3 →
  incorrect_points = 1 →
  total_score = 78 →
  ∃ (correct_answers : ℕ),
    correct_answers * correct_points - (total_questions - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 27 :=
by sorry

end NUMINAMATH_CALUDE_test_score_problem_l1350_135042


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1350_135092

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1350_135092


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_1000_l1350_135045

def hulk_jump (n : ℕ) : ℕ := 2^(n-1)

theorem hulk_jump_exceeds_1000 :
  ∃ n : ℕ, n > 0 ∧ hulk_jump n > 1000 ∧ ∀ m : ℕ, m > 0 ∧ m < n → hulk_jump m ≤ 1000 :=
by
  use 11
  sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_1000_l1350_135045


namespace NUMINAMATH_CALUDE_max_net_revenue_l1350_135021

/-- Represents the net revenue function for a movie theater --/
def net_revenue (x : ℕ) : ℤ :=
  if x ≤ 10 then 1000 * x - 5750
  else -30 * x * x + 1300 * x - 5750

/-- Theorem stating the maximum net revenue and optimal ticket price --/
theorem max_net_revenue :
  ∃ (max_revenue : ℕ) (optimal_price : ℕ),
    max_revenue = 8830 ∧
    optimal_price = 22 ∧
    (∀ (x : ℕ), x ≥ 6 → x ≤ 38 → net_revenue x ≤ net_revenue optimal_price) :=
by sorry

end NUMINAMATH_CALUDE_max_net_revenue_l1350_135021


namespace NUMINAMATH_CALUDE_problem_statement_l1350_135097

theorem problem_statement (a b c : ℝ) (h : a + b = ab ∧ ab = c) :
  (c ≠ 0 → (2*a - 3*a*b + 2*b) / (5*a + 7*a*b + 5*b) = -1/12) ∧
  (a = 3 → b + c = 6) ∧
  (c ≠ 0 → (1-a)*(1-b) = 1/a + 1/b) ∧
  (c = 4 → a^2 + b^2 = 8) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1350_135097


namespace NUMINAMATH_CALUDE_w_share_is_375_l1350_135003

/-- A structure representing the distribution of money among four individuals -/
structure MoneyDistribution where
  total : ℝ
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  proportion_w : ℝ := 1
  proportion_x : ℝ := 6
  proportion_y : ℝ := 2
  proportion_z : ℝ := 4
  sum_proportions : ℝ := proportion_w + proportion_x + proportion_y + proportion_z
  proportional_distribution :
    w / proportion_w = x / proportion_x ∧
    x / proportion_x = y / proportion_y ∧
    y / proportion_y = z / proportion_z ∧
    w + x + y + z = total
  x_exceeds_y : x = y + 1500

theorem w_share_is_375 (d : MoneyDistribution) : d.w = 375 := by
  sorry

end NUMINAMATH_CALUDE_w_share_is_375_l1350_135003


namespace NUMINAMATH_CALUDE_consecutive_terms_iff_equation_l1350_135093

/-- Sequence a_k defined by a_0 = 0, a_1 = n, and a_{k+1} = n^2 * a_k - a_{k-1} -/
def sequence_a (n : ℕ) : ℕ → ℤ
  | 0 => 0
  | 1 => n
  | (k + 2) => n^2 * sequence_a n (k + 1) - sequence_a n k

/-- Predicate to check if two integers are consecutive terms in the sequence -/
def are_consecutive_terms (n a b : ℕ) : Prop :=
  ∃ k : ℕ, sequence_a n k = a ∧ sequence_a n (k + 1) = b

theorem consecutive_terms_iff_equation (n : ℕ) (hn : n > 0) (a b : ℕ) (hab : a ≤ b) :
  are_consecutive_terms n a b ↔ a^2 + b^2 = n^2 * (a * b + 1) :=
sorry

end NUMINAMATH_CALUDE_consecutive_terms_iff_equation_l1350_135093


namespace NUMINAMATH_CALUDE_comparison_inequality_l1350_135026

theorem comparison_inequality (a b : ℝ) (h1 : a ≠ b) (h2 : a < 0) :
  a < 2 * b - b^2 / a := by
sorry

end NUMINAMATH_CALUDE_comparison_inequality_l1350_135026


namespace NUMINAMATH_CALUDE_letters_with_both_count_l1350_135050

/-- Represents the number of letters in the alphabet. -/
def total_letters : ℕ := 40

/-- Represents the number of letters with a straight line but no dot. -/
def line_only : ℕ := 24

/-- Represents the number of letters with a dot but no straight line. -/
def dot_only : ℕ := 6

/-- Represents the number of letters with both a dot and a straight line. -/
def both : ℕ := total_letters - line_only - dot_only

theorem letters_with_both_count :
  both = 10 :=
sorry

end NUMINAMATH_CALUDE_letters_with_both_count_l1350_135050


namespace NUMINAMATH_CALUDE_socks_theorem_l1350_135080

def socks_problem (initial_pairs : ℕ) : Prop :=
  let week1 := 12
  let week2 := week1 + 4
  let week3 := (week1 + week2) / 2
  let week4 := week3 - 3
  let total := 57
  initial_pairs = total - (week1 + week2 + week3 + week4)

theorem socks_theorem : ∃ (x : ℕ), socks_problem x :=
sorry

end NUMINAMATH_CALUDE_socks_theorem_l1350_135080


namespace NUMINAMATH_CALUDE_gcd_of_mersenne_numbers_l1350_135064

theorem gcd_of_mersenne_numbers : Nat.gcd (2^2048 - 1) (2^2035 - 1) = 2^13 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_mersenne_numbers_l1350_135064


namespace NUMINAMATH_CALUDE_quadratic_equation_and_inequality_l1350_135046

theorem quadratic_equation_and_inequality 
  (a b : ℝ) 
  (h1 : (a:ℝ) * (-1/2)^2 + b * (-1/2) + 2 = 0)
  (h2 : (a:ℝ) * 2^2 + b * 2 + 2 = 0) :
  (a = -2 ∧ b = 3) ∧ 
  (∀ x : ℝ, a * x^2 + b * x - 1 > 0 ↔ 1/2 < x ∧ x < 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_and_inequality_l1350_135046


namespace NUMINAMATH_CALUDE_pave_hall_l1350_135022

/-- Calculates the number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  let hall_area := hall_length * hall_width * 100
  let stone_area := stone_length * stone_width
  (hall_area / stone_area).ceil.toNat

/-- Theorem stating that 5400 stones are required to pave the given hall -/
theorem pave_hall : stones_required 36 15 2 5 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_pave_hall_l1350_135022


namespace NUMINAMATH_CALUDE_min_value_theorem_l1350_135044

theorem min_value_theorem (x y : ℝ) :
  (y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 ≥ 1/6 ∧
  ((y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 = 1/6 ↔ x = 5/2 ∧ y = 5/6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1350_135044


namespace NUMINAMATH_CALUDE_scientific_notation_159600_l1350_135012

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_159600 :
  toScientificNotation 159600 = ScientificNotation.mk 1.596 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_159600_l1350_135012


namespace NUMINAMATH_CALUDE_triangle_third_side_prime_l1350_135043

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def valid_third_side (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_prime (a b : ℕ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), is_prime c ∧ valid_third_side a b c ↔ 
  c = 5 ∨ c = 7 ∨ c = 11 ∨ c = 13 ∨ c = 17 :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_prime_l1350_135043


namespace NUMINAMATH_CALUDE_absolute_value_expression_l1350_135087

theorem absolute_value_expression : 
  let x : ℤ := -2023
  ‖‖|x| - (x + 3)‖ - (|x| - 3)‖ - (x - 3) = 4049 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l1350_135087


namespace NUMINAMATH_CALUDE_game_tie_fraction_l1350_135075

theorem game_tie_fraction (mark_wins jane_wins : ℚ) 
  (h1 : mark_wins = 5 / 12)
  (h2 : jane_wins = 1 / 4) : 
  1 - (mark_wins + jane_wins) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_game_tie_fraction_l1350_135075
