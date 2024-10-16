import Mathlib

namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l593_59373

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a + 4 = 0) → 
  (b^3 - 2*b + 4 = 0) → 
  (c^3 - 2*c + 4 = 0) → 
  (1/(a-2) + 1/(b-2) + 1/(c-2) = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l593_59373


namespace NUMINAMATH_CALUDE_tangent_values_l593_59346

/-- Two linear functions with parallel non-vertical graphs -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  hf : f = λ x => a * x + b
  hg : g = λ x => a * x + c
  ha : a ≠ 0

/-- The property that (f x)^2 is tangent to -12(g x) -/
def is_tangent_f_g (p : ParallelLinearFunctions) : Prop :=
  ∃! x, (p.f x)^2 = -12 * (p.g x)

/-- The main theorem -/
theorem tangent_values (p : ParallelLinearFunctions) 
  (h : is_tangent_f_g p) :
  ∃ A : Set ℝ, A = {0, 12} ∧ 
  ∀ a : ℝ, a ∈ A ↔ ∃! x, (p.g x)^2 = a * (p.f x) := by
  sorry

end NUMINAMATH_CALUDE_tangent_values_l593_59346


namespace NUMINAMATH_CALUDE_purchase_percentage_l593_59369

/-- Given a 25% price increase and a net difference in expenditure of 20,
    prove that the percentage of the required amount purchased is 16%. -/
theorem purchase_percentage (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 1.25 * P
  let R := (500 : ℝ) / 31.25
  let new_expenditure := new_price * (R / 100) * Q
  P * Q - new_expenditure = 20 → R = 16 := by sorry

end NUMINAMATH_CALUDE_purchase_percentage_l593_59369


namespace NUMINAMATH_CALUDE_square_ending_in_five_l593_59327

theorem square_ending_in_five (a : ℕ) :
  let n : ℕ := 10 * a + 5
  ∃ (m : ℕ), n^2 = m^2 → a % 10 = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_ending_in_five_l593_59327


namespace NUMINAMATH_CALUDE_girl_transfer_problem_l593_59365

/-- Represents the number of girls in each group before and after transfers -/
structure GirlCounts where
  initial_B : ℕ
  initial_A : ℕ
  initial_C : ℕ
  final : ℕ

/-- Represents the number of girls transferred between groups -/
structure GirlTransfers where
  from_A : ℕ
  from_B : ℕ
  from_C : ℕ

/-- The theorem statement for the girl transfer problem -/
theorem girl_transfer_problem (g : GirlCounts) (t : GirlTransfers) : 
  g.initial_A = g.initial_B + 4 →
  g.initial_B = g.initial_C + 1 →
  t.from_C = 2 →
  g.final = g.initial_A - t.from_A + t.from_C →
  g.final = g.initial_B - t.from_B + t.from_A →
  g.final = g.initial_C - t.from_C + t.from_B →
  t.from_A = 5 ∧ t.from_B = 4 := by
  sorry


end NUMINAMATH_CALUDE_girl_transfer_problem_l593_59365


namespace NUMINAMATH_CALUDE_chess_tournament_schedules_l593_59358

/-- Represents the number of players from each school -/
def num_players : ℕ := 4

/-- Represents the number of rounds in the tournament -/
def num_rounds : ℕ := 4

/-- Represents the number of games per round -/
def games_per_round : ℕ := 4

/-- Calculates the total number of games in the tournament -/
def total_games : ℕ := num_players * num_players

/-- Theorem stating the number of ways to schedule the chess tournament -/
theorem chess_tournament_schedules : 
  (num_rounds.factorial * (games_per_round.factorial ^ num_rounds)) = 7962624 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_schedules_l593_59358


namespace NUMINAMATH_CALUDE_determinant_fraction_equality_l593_59354

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the theorem
theorem determinant_fraction_equality (θ : ℝ) : 
  det (Real.sin θ) 2 (Real.cos θ) 3 = 0 →
  (3 * Real.sin θ + 2 * Real.cos θ) / (3 * Real.sin θ - Real.cos θ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_determinant_fraction_equality_l593_59354


namespace NUMINAMATH_CALUDE_clerks_count_l593_59339

/-- Represents the grocery store employee structure and salaries --/
structure GroceryStore where
  manager_salary : ℕ
  clerk_salary : ℕ
  num_managers : ℕ
  total_salary : ℕ

/-- Calculates the number of clerks in the grocery store --/
def num_clerks (store : GroceryStore) : ℕ :=
  (store.total_salary - store.manager_salary * store.num_managers) / store.clerk_salary

/-- Theorem stating that the number of clerks is 3 given the conditions --/
theorem clerks_count (store : GroceryStore) 
    (h1 : store.manager_salary = 5)
    (h2 : store.clerk_salary = 2)
    (h3 : store.num_managers = 2)
    (h4 : store.total_salary = 16) : 
  num_clerks store = 3 := by
  sorry

end NUMINAMATH_CALUDE_clerks_count_l593_59339


namespace NUMINAMATH_CALUDE_octal_arithmetic_l593_59368

/-- Represents a number in base 8 --/
def OctalNumber := Nat

/-- Addition in base 8 --/
def octal_add (a b : OctalNumber) : OctalNumber := sorry

/-- Subtraction in base 8 --/
def octal_sub (a b : OctalNumber) : OctalNumber := sorry

/-- Conversion from decimal to octal --/
def to_octal (n : Nat) : OctalNumber := sorry

/-- Theorem: In base 8, (672₈ + 156₈) - 213₈ = 645₈ --/
theorem octal_arithmetic : 
  octal_sub (octal_add (to_octal 672) (to_octal 156)) (to_octal 213) = to_octal 645 := by
  sorry

end NUMINAMATH_CALUDE_octal_arithmetic_l593_59368


namespace NUMINAMATH_CALUDE_h_3_value_l593_59378

-- Define the functions
def f (x : ℝ) : ℝ := 2 * x + 9
def g (x : ℝ) : ℝ := (f x) ^ (1/3) - 3
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_3_value : h 3 = 2 * 15^(1/3) + 3 := by sorry

end NUMINAMATH_CALUDE_h_3_value_l593_59378


namespace NUMINAMATH_CALUDE_michael_matchsticks_l593_59357

/-- The number of matchstick houses Michael creates -/
def num_houses : ℕ := 30

/-- The number of matchsticks used per house -/
def matchsticks_per_house : ℕ := 10

/-- The total number of matchsticks Michael used -/
def total_matchsticks_used : ℕ := num_houses * matchsticks_per_house

/-- Michael's original number of matchsticks -/
def original_matchsticks : ℕ := 2 * total_matchsticks_used

theorem michael_matchsticks : original_matchsticks = 600 := by
  sorry

end NUMINAMATH_CALUDE_michael_matchsticks_l593_59357


namespace NUMINAMATH_CALUDE_recurrence_sequence_a1_zero_l593_59379

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (c : ℝ) (a : ℕ → ℝ) : Prop :=
  c > 2 ∧ ∀ n : ℕ, a n = (a (n - 1))^2 - a (n - 1) ∧ a n < 1 / Real.sqrt (c * n)

/-- The main theorem stating that a₁ = 0 for any sequence satisfying the recurrence relation -/
theorem recurrence_sequence_a1_zero (c : ℝ) (a : ℕ → ℝ) (h : RecurrenceSequence c a) : a 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_a1_zero_l593_59379


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_of_six_l593_59301

theorem least_four_digit_multiple_of_six : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧  -- four-digit number
  n % 6 = 0 ∧               -- multiple of 6
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 6 = 0) → n ≤ m) ∧ -- least such number
  n = 1002 := by
sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_of_six_l593_59301


namespace NUMINAMATH_CALUDE_triangle_intersection_height_l593_59323

theorem triangle_intersection_height (t : ℝ) : 
  let A : ℝ × ℝ := (0, 8)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (8, 0)
  let T : ℝ × ℝ := ((8 - t) / 4, t)
  let U : ℝ × ℝ := (8 - t, t)
  let area_ATU : ℝ := (1 / 2) * (U.1 - T.1) * (A.2 - T.2)
  (0 ≤ t) ∧ (t ≤ 8) ∧ (area_ATU = 13.5) → t = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_intersection_height_l593_59323


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l593_59367

/-- Given a cube with a sphere inscribed within it, and another cube inscribed within the sphere, 
    this theorem relates the surface areas of the outer and inner cubes. -/
theorem inscribed_cube_surface_area 
  (outer_cube_surface_area : ℝ) 
  (inner_cube_surface_area : ℝ) 
  (h_outer : outer_cube_surface_area = 54) :
  inner_cube_surface_area = 18 :=
sorry

#check inscribed_cube_surface_area

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l593_59367


namespace NUMINAMATH_CALUDE_small_circle_radius_l593_59343

/-- Given a large circle with radius 10 meters containing seven smaller congruent circles
    that fit exactly along its diameter, prove that the radius of each smaller circle is 10/7 meters. -/
theorem small_circle_radius (R : ℝ) (n : ℕ) (r : ℝ) : 
  R = 10 → n = 7 → 2 * R = n * (2 * r) → r = 10 / 7 := by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l593_59343


namespace NUMINAMATH_CALUDE_max_intersection_points_for_circles_l593_59310

/-- The maximum number of intersection points for n circles in a plane -/
def max_intersection_points (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: Given n circles in a plane, where n ≥ 2, that intersect each other pairwise,
    the maximum number of intersection points is n(n-1). -/
theorem max_intersection_points_for_circles (n : ℕ) (h : n ≥ 2) :
  max_intersection_points n = n * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_for_circles_l593_59310


namespace NUMINAMATH_CALUDE_probability_three_red_jellybeans_l593_59381

/-- Represents the probability of selecting exactly 3 red jellybeans from a bowl -/
def probability_three_red (total : ℕ) (red : ℕ) (blue : ℕ) (white : ℕ) : ℚ :=
  let total_combinations := Nat.choose total 4
  let favorable_outcomes := Nat.choose red 3 * Nat.choose (blue + white) 1
  favorable_outcomes / total_combinations

/-- Theorem stating the probability of selecting exactly 3 red jellybeans -/
theorem probability_three_red_jellybeans :
  probability_three_red 15 6 3 6 = 4 / 91 := by
  sorry

#eval probability_three_red 15 6 3 6

end NUMINAMATH_CALUDE_probability_three_red_jellybeans_l593_59381


namespace NUMINAMATH_CALUDE_forty_percent_of_number_l593_59372

theorem forty_percent_of_number (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → (40/100 : ℝ) * N = 192 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_l593_59372


namespace NUMINAMATH_CALUDE_greatest_integer_of_a_l593_59342

def a : ℕ → ℚ
  | 0 => 1994
  | n + 1 => 1994^2 / (a n + 1)

theorem greatest_integer_of_a (n : ℕ) (h : n ≤ 998) :
  ⌊a n⌋ = 1994 - n := by sorry

end NUMINAMATH_CALUDE_greatest_integer_of_a_l593_59342


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l593_59396

theorem arithmetic_expression_equality : -6 * 5 - (-4 * -2) + (-12 * -6) / 3 = -14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l593_59396


namespace NUMINAMATH_CALUDE_two_thirds_squared_l593_59375

theorem two_thirds_squared : (2 / 3 : ℚ) ^ 2 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_squared_l593_59375


namespace NUMINAMATH_CALUDE_quadratic_sufficient_not_necessary_l593_59393

theorem quadratic_sufficient_not_necessary :
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) ∧
  ¬(∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sufficient_not_necessary_l593_59393


namespace NUMINAMATH_CALUDE_game_correct_answers_l593_59316

theorem game_correct_answers (total_questions : ℕ) (correct_reward : ℕ) (incorrect_penalty : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_reward = 7)
  (h3 : incorrect_penalty = 3) :
  ∃ (x : ℕ), x * correct_reward = (total_questions - x) * incorrect_penalty ∧ x = 15 := by
sorry

end NUMINAMATH_CALUDE_game_correct_answers_l593_59316


namespace NUMINAMATH_CALUDE_maintenance_check_time_l593_59324

/-- The initial time between maintenance checks before using the additive -/
def initial_time : ℝ := 20

/-- The new time between maintenance checks after using the additive -/
def new_time : ℝ := 25

/-- The percentage increase in time between maintenance checks -/
def percentage_increase : ℝ := 0.25

theorem maintenance_check_time : 
  initial_time * (1 + percentage_increase) = new_time :=
by sorry

end NUMINAMATH_CALUDE_maintenance_check_time_l593_59324


namespace NUMINAMATH_CALUDE_fraction_equality_l593_59338

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) : 
  18 / 7 + ((2 * q - p) / (2 * q + p)) = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l593_59338


namespace NUMINAMATH_CALUDE_not_both_perfect_squares_l593_59309

theorem not_both_perfect_squares (p q : ℕ) (hp : p > 0) (hq : q > 0) :
  ¬(∃ (a b : ℕ), p^2 + q = a^2 ∧ p + q^2 = b^2) := by
  sorry

end NUMINAMATH_CALUDE_not_both_perfect_squares_l593_59309


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l593_59328

/-- Given ratios between w, x, y, and z, prove the ratio of w to y -/
theorem ratio_w_to_y 
  (h_wx : (w : ℚ) / x = 5 / 2)
  (h_yz : (y : ℚ) / z = 4 / 1)
  (h_zx : (z : ℚ) / x = 2 / 5) :
  w / y = 25 / 16 :=
by sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l593_59328


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l593_59352

theorem contrapositive_equivalence (p q : Prop) : (p → q) → (¬q → ¬p) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l593_59352


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l593_59341

theorem arithmetic_sequence_count (a₁ aₙ d : ℤ) (n : ℕ) :
  a₁ = 165 ∧ aₙ = 35 ∧ d = -5 →
  aₙ = a₁ + (n - 1) * d →
  n = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l593_59341


namespace NUMINAMATH_CALUDE_black_cubes_removed_multiple_of_four_l593_59340

/-- Represents a cube constructed from unit cubes of two colors -/
structure ColoredCube where
  edge_length : ℕ
  black_cubes : ℕ
  white_cubes : ℕ
  adjacent_different : Bool

/-- Represents the removal of unit cubes from a ColoredCube -/
structure CubeRemoval where
  cube : ColoredCube
  removed_cubes : ℕ
  rods_affected : ℕ
  cubes_per_rod : ℕ

/-- Theorem stating that the number of black cubes removed is a multiple of 4 -/
theorem black_cubes_removed_multiple_of_four (removal : CubeRemoval) : 
  removal.cube.edge_length = 10 ∧ 
  removal.cube.black_cubes = 500 ∧ 
  removal.cube.white_cubes = 500 ∧
  removal.cube.adjacent_different = true ∧
  removal.removed_cubes = 100 ∧
  removal.rods_affected = 300 ∧
  removal.cubes_per_rod = 1 →
  ∃ (k : ℕ), (removal.removed_cubes - k) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_black_cubes_removed_multiple_of_four_l593_59340


namespace NUMINAMATH_CALUDE_g_of_neg_four_l593_59374

/-- Given a function g(x) = 5x - 2, prove that g(-4) = -22 -/
theorem g_of_neg_four (g : ℝ → ℝ) (h : ∀ x, g x = 5 * x - 2) : g (-4) = -22 := by
  sorry

end NUMINAMATH_CALUDE_g_of_neg_four_l593_59374


namespace NUMINAMATH_CALUDE_fraction_cubes_equals_729_l593_59302

theorem fraction_cubes_equals_729 : (81000 ^ 3) / (9000 ^ 3) = 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cubes_equals_729_l593_59302


namespace NUMINAMATH_CALUDE_min_toothpicks_to_remove_for_given_figure_l593_59305

/-- Represents a figure made of toothpicks forming squares and triangles. -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  num_squares : ℕ
  num_triangles : ℕ
  toothpicks_per_square : ℕ
  toothpicks_per_triangle : ℕ

/-- Calculates the minimum number of toothpicks to remove to eliminate all shapes. -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ := sorry

/-- Theorem stating the minimum number of toothpicks to remove for the given figure. -/
theorem min_toothpicks_to_remove_for_given_figure :
  let figure : ToothpickFigure := {
    total_toothpicks := 40,
    num_squares := 10,
    num_triangles := 15,
    toothpicks_per_square := 4,
    toothpicks_per_triangle := 3
  }
  min_toothpicks_to_remove figure = 10 := by sorry

end NUMINAMATH_CALUDE_min_toothpicks_to_remove_for_given_figure_l593_59305


namespace NUMINAMATH_CALUDE_chicken_egg_production_l593_59317

theorem chicken_egg_production 
  (num_chickens : ℕ) 
  (price_per_dozen : ℚ) 
  (total_revenue : ℚ) 
  (num_weeks : ℕ) 
  (h1 : num_chickens = 46)
  (h2 : price_per_dozen = 3)
  (h3 : total_revenue = 552)
  (h4 : num_weeks = 8) :
  (total_revenue / (price_per_dozen / 12) / num_weeks / num_chickens : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_egg_production_l593_59317


namespace NUMINAMATH_CALUDE_claire_pets_ratio_l593_59300

theorem claire_pets_ratio : 
  ∀ (total_pets gerbils hamsters male_gerbils male_hamsters : ℕ),
    total_pets = 92 →
    gerbils + hamsters = total_pets →
    gerbils = 68 →
    male_hamsters = hamsters / 3 →
    male_gerbils + male_hamsters = 25 →
    male_gerbils * 4 = gerbils :=
by
  sorry

end NUMINAMATH_CALUDE_claire_pets_ratio_l593_59300


namespace NUMINAMATH_CALUDE_wednesday_earnings_l593_59318

/-- Represents the earnings from selling cabbage over three days -/
structure CabbageEarnings where
  wednesday : ℝ
  friday : ℝ
  today : ℝ
  total_kg : ℝ
  price_per_kg : ℝ

/-- Theorem stating that given the conditions, Johannes earned $30 on Wednesday -/
theorem wednesday_earnings (e : CabbageEarnings) 
  (h1 : e.friday = 24)
  (h2 : e.today = 42)
  (h3 : e.total_kg = 48)
  (h4 : e.price_per_kg = 2)
  (h5 : e.wednesday + e.friday + e.today = e.total_kg * e.price_per_kg) :
  e.wednesday = 30 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_earnings_l593_59318


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l593_59336

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope 3 and x-intercept (7, 0), the y-intercept is (0, -21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := 3, x_intercept := (7, 0) }
  y_intercept l = (0, -21) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l593_59336


namespace NUMINAMATH_CALUDE_tan_half_sum_of_angles_l593_59359

theorem tan_half_sum_of_angles (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 1)
  (h2 : Real.sin a + Real.sin b = 1/2)
  (h3 : Real.tan (a - b) = 1) :
  Real.tan ((a + b) / 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_of_angles_l593_59359


namespace NUMINAMATH_CALUDE_cereal_boxes_purchased_l593_59350

/-- Given the initial price, price reduction, and total payment for monster boxes of cereal,
    prove that the number of boxes purchased is 20. -/
theorem cereal_boxes_purchased
  (initial_price : ℕ)
  (price_reduction : ℕ)
  (total_payment : ℕ)
  (h1 : initial_price = 104)
  (h2 : price_reduction = 24)
  (h3 : total_payment = 1600) :
  total_payment / (initial_price - price_reduction) = 20 :=
by sorry

end NUMINAMATH_CALUDE_cereal_boxes_purchased_l593_59350


namespace NUMINAMATH_CALUDE_fraction_equality_l593_59331

theorem fraction_equality : (18 * 3 + 12) / (6 - 4) = 33 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l593_59331


namespace NUMINAMATH_CALUDE_road_travel_cost_l593_59311

/-- Calculates the cost of traveling two intersecting roads on a rectangular lawn. -/
theorem road_travel_cost
  (lawn_length lawn_width road_width : ℕ)
  (cost_per_sqm : ℚ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 60)
  (h3 : road_width = 10)
  (h4 : cost_per_sqm = 4) :
  (((lawn_length * road_width + lawn_width * road_width - road_width * road_width) : ℚ) * cost_per_sqm) = 5200 := by
  sorry

end NUMINAMATH_CALUDE_road_travel_cost_l593_59311


namespace NUMINAMATH_CALUDE_intersection_when_m_3_intersection_equals_B_l593_59366

-- Define sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

-- Theorem for part (1)
theorem intersection_when_m_3 :
  (A ∩ B 3) = {x | 3 ≤ x ∧ x ≤ 4} ∧
  (A ∩ (B 3)ᶜ) = {x | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem intersection_equals_B (m : ℝ) :
  (A ∩ B m) = B m ↔ 1 ≤ m ∧ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_3_intersection_equals_B_l593_59366


namespace NUMINAMATH_CALUDE_cars_meeting_halfway_l593_59394

/-- Two cars meeting halfway between two points --/
theorem cars_meeting_halfway 
  (total_distance : ℝ) 
  (speed_car1 : ℝ) 
  (start_time_car1 start_time_car2 : ℕ) 
  (speed_car2 : ℝ) :
  total_distance = 600 →
  speed_car1 = 50 →
  start_time_car1 = 7 →
  start_time_car2 = 8 →
  (total_distance / 2) / speed_car1 + start_time_car1 = 
    (total_distance / 2) / speed_car2 + start_time_car2 →
  speed_car2 = 60 := by
sorry

end NUMINAMATH_CALUDE_cars_meeting_halfway_l593_59394


namespace NUMINAMATH_CALUDE_money_left_l593_59308

def initial_amount : ℚ := 200.50
def sweets_cost : ℚ := 35.25
def stickers_cost : ℚ := 10.75
def friend_gift : ℚ := 25.20
def num_friends : ℕ := 4
def charity_donation : ℚ := 15.30

theorem money_left : 
  initial_amount - (sweets_cost + stickers_cost + friend_gift * num_friends + charity_donation) = 38.40 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l593_59308


namespace NUMINAMATH_CALUDE_good_numbers_characterization_l593_59325

/-- A number n > 3 is 'good' if the set of weights {1, 2, 3, ..., n} can be divided into three piles of equal mass -/
def is_good (n : ℕ) : Prop :=
  n > 3 ∧ ∃ (a b c : Finset ℕ), a ∪ b ∪ c = Finset.range n ∧ 
    a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
    a.sum id = b.sum id ∧ b.sum id = c.sum id

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem good_numbers_characterization (n : ℕ) :
  is_good n ↔ (∃ k : ℕ, k ≥ 1 ∧ (n = 3 * k ∨ n = 3 * k + 2)) :=
sorry

end NUMINAMATH_CALUDE_good_numbers_characterization_l593_59325


namespace NUMINAMATH_CALUDE_line_through_points_l593_59395

/-- A line with slope 4 passing through points (3,5), (a,7), and (-1,b) has a = 7/2 and b = -11 -/
theorem line_through_points (a b : ℚ) : 
  (((7 - 5) / (a - 3) = 4) ∧ ((b - 5) / (-1 - 3) = 4)) → 
  (a = 7/2 ∧ b = -11) := by
sorry

end NUMINAMATH_CALUDE_line_through_points_l593_59395


namespace NUMINAMATH_CALUDE_point_D_in_fourth_quadrant_l593_59334

/-- Definition of a point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point D -/
def point_D : Point :=
  { x := 6, y := -7 }

/-- Theorem: point D is in the fourth quadrant -/
theorem point_D_in_fourth_quadrant : fourth_quadrant point_D := by
  sorry

end NUMINAMATH_CALUDE_point_D_in_fourth_quadrant_l593_59334


namespace NUMINAMATH_CALUDE_min_distance_to_i_l593_59384

theorem min_distance_to_i (z : ℂ) (h : Complex.abs (z + Complex.I * Real.sqrt 3) + Complex.abs (z - Complex.I * Real.sqrt 3) = 4) :
  ∃ (w : ℂ), Complex.abs (w + Complex.I * Real.sqrt 3) + Complex.abs (w - Complex.I * Real.sqrt 3) = 4 ∧
    Complex.abs (w - Complex.I) ≤ Complex.abs (z - Complex.I) ∧
    Complex.abs (w - Complex.I) = Real.sqrt 6 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_i_l593_59384


namespace NUMINAMATH_CALUDE_comprehensive_office_increases_profit_building_comprehensive_offices_increases_profit_l593_59386

/-- Represents a technology company -/
structure TechCompany where
  name : String
  profit : ℝ
  employeeRetention : ℝ
  productivity : ℝ
  workLifeIntegration : ℝ

/-- Represents an office environment -/
structure OfficeEnvironment where
  hasWorkSpaces : Bool
  hasLeisureSpaces : Bool
  hasLivingSpaces : Bool

/-- Function to determine if an office environment is comprehensive -/
def isComprehensiveOffice (office : OfficeEnvironment) : Bool :=
  office.hasWorkSpaces ∧ office.hasLeisureSpaces ∧ office.hasLivingSpaces

/-- Function to calculate the impact of office environment on company metrics -/
def officeImpact (company : TechCompany) (office : OfficeEnvironment) : TechCompany :=
  if isComprehensiveOffice office then
    { company with
      employeeRetention := company.employeeRetention * 1.1
      productivity := company.productivity * 1.15
      workLifeIntegration := company.workLifeIntegration * 1.2
    }
  else
    company

/-- Theorem stating that comprehensive offices increase profit -/
theorem comprehensive_office_increases_profit (company : TechCompany) (office : OfficeEnvironment) :
  isComprehensiveOffice office →
  (officeImpact company office).profit > company.profit :=
by sorry

/-- Main theorem proving that building comprehensive offices increases profit through specific factors -/
theorem building_comprehensive_offices_increases_profit (company : TechCompany) (office : OfficeEnvironment) :
  isComprehensiveOffice office →
  (∃ newCompany : TechCompany,
    newCompany = officeImpact company office ∧
    newCompany.profit > company.profit ∧
    newCompany.employeeRetention > company.employeeRetention ∧
    newCompany.productivity > company.productivity ∧
    newCompany.workLifeIntegration > company.workLifeIntegration) :=
by sorry

end NUMINAMATH_CALUDE_comprehensive_office_increases_profit_building_comprehensive_offices_increases_profit_l593_59386


namespace NUMINAMATH_CALUDE_babysitting_hourly_rate_l593_59360

/-- Calculates the hourly rate for babysitting given total expenses, hours worked, and leftover money -/
def calculate_hourly_rate (total_expenses : ℕ) (hours_worked : ℕ) (leftover : ℕ) : ℚ :=
  (total_expenses + leftover) / hours_worked

/-- Theorem: Given the problem conditions, the hourly rate for babysitting is $8 -/
theorem babysitting_hourly_rate :
  let total_expenses := 65
  let hours_worked := 9
  let leftover := 7
  calculate_hourly_rate total_expenses hours_worked leftover = 8 := by
sorry

end NUMINAMATH_CALUDE_babysitting_hourly_rate_l593_59360


namespace NUMINAMATH_CALUDE_no_integer_solution_l593_59362

theorem no_integer_solution : ¬∃ (m n : ℤ), m^2 + 1954 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l593_59362


namespace NUMINAMATH_CALUDE_triangle_circles_area_sum_l593_59337

theorem triangle_circles_area_sum : 
  ∀ (u v w : ℝ),
  u > 0 ∧ v > 0 ∧ w > 0 →
  u + v = 6 →
  u + w = 8 →
  v + w = 10 →
  π * (u^2 + v^2 + w^2) = 56 * π :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_circles_area_sum_l593_59337


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_four_l593_59376

/-- Given two vectors a and b in ℝ², where a = (2,1) and b = (x,2),
    if a + b is parallel to a - 2b, then x = 4 -/
theorem parallel_vectors_imply_x_equals_four (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, 2)
  (∃ (k : ℝ), k ≠ 0 ∧ (a.1 + b.1, a.2 + b.2) = k • (a.1 - 2*b.1, a.2 - 2*b.2)) →
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_four_l593_59376


namespace NUMINAMATH_CALUDE_trent_tears_per_three_onions_l593_59388

def tears_per_three_onions (pots : ℕ) (onions_per_pot : ℕ) (total_tears : ℕ) : ℚ :=
  (3 * total_tears : ℚ) / (pots * onions_per_pot : ℚ)

theorem trent_tears_per_three_onions :
  tears_per_three_onions 6 4 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trent_tears_per_three_onions_l593_59388


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l593_59335

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f := λ x : ℝ => a^(x - 2) + 2
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l593_59335


namespace NUMINAMATH_CALUDE_range_of_m_l593_59321

/-- The equation |(x-1)(x-3)| = m*x has four distinct real roots -/
def has_four_distinct_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (∀ (x : ℝ), |((x - 1) * (x - 3))| = m * x ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

/-- The theorem stating the range of m -/
theorem range_of_m : 
  ∀ (m : ℝ), has_four_distinct_roots m ↔ 0 < m ∧ m < 4 - 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l593_59321


namespace NUMINAMATH_CALUDE_gcf_of_180_240_300_l593_59382

theorem gcf_of_180_240_300 : Nat.gcd 180 (Nat.gcd 240 300) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_240_300_l593_59382


namespace NUMINAMATH_CALUDE_sum_a_d_equals_one_l593_59363

theorem sum_a_d_equals_one 
  (a b c d : ℤ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 5) 
  (h3 : c + d = 3) : 
  a + d = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_one_l593_59363


namespace NUMINAMATH_CALUDE_javier_donut_fundraising_l593_59303

/-- Represents the problem of calculating Javier's fundraising for a new game through donut sales. -/
theorem javier_donut_fundraising
  (dozen_cost : ℚ)
  (donut_price : ℚ)
  (donuts_per_dozen : ℕ)
  (dozens_to_sell : ℕ)
  (h1 : dozen_cost = 240 / 100)  -- $2.40 per dozen
  (h2 : donut_price = 1)         -- $1 per donut
  (h3 : donuts_per_dozen = 12)   -- 12 donuts in a dozen
  (h4 : dozens_to_sell = 10)     -- Selling 10 dozens
  : (dozens_to_sell * donuts_per_dozen * donut_price) - (dozens_to_sell * dozen_cost) = 96 := by
  sorry


end NUMINAMATH_CALUDE_javier_donut_fundraising_l593_59303


namespace NUMINAMATH_CALUDE_gcd_sum_and_sum_squares_l593_59351

theorem gcd_sum_and_sum_squares (a b : ℕ) (h : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_gcd_sum_and_sum_squares_l593_59351


namespace NUMINAMATH_CALUDE_supermarket_profit_and_discount_l593_59370

-- Define the goods
structure Good where
  cost : ℝ
  price : ℝ

-- Define the problem parameters
def good_A : Good := { cost := 22, price := 29 }
def good_B : Good := { cost := 30, price := 40 }

-- Define the theorem
theorem supermarket_profit_and_discount 
  (total_cost : ℝ) 
  (num_A : ℕ) 
  (num_B : ℕ) 
  (second_profit_increase : ℝ) :
  total_cost = 6000 ∧ 
  num_B = (num_A / 2 + 15 : ℕ) ∧
  num_A * good_A.cost + num_B * good_B.cost = total_cost →
  (num_A * (good_A.price - good_A.cost) + num_B * (good_B.price - good_B.cost) = 1950) ∧
  ∃ discount_rate : ℝ,
    discount_rate ≥ 0 ∧ 
    discount_rate ≤ 1 ∧
    num_A * (good_A.price - good_A.cost) + 3 * num_B * ((1 - discount_rate) * good_B.price - good_B.cost) = 
    1950 + second_profit_increase ∧
    discount_rate = 0.085 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_supermarket_profit_and_discount_l593_59370


namespace NUMINAMATH_CALUDE_largest_digit_change_corrects_addition_l593_59361

def original_sum : ℕ := 735 + 468 + 281
def given_result : ℕ := 1584
def correct_first_addend : ℕ := 835

theorem largest_digit_change_corrects_addition :
  (original_sum ≠ given_result) →
  (correct_first_addend + 468 + 281 = given_result) →
  ∀ (d : ℕ), d ≤ 9 →
    (d > 7 → 
      ¬∃ (a b c : ℕ), a ≤ 999 ∧ b ≤ 999 ∧ c ≤ 999 ∧
        (a + b + c = given_result) ∧
        (a = 735 + d * 100 - 700 ∨
         b = 468 + d * 100 - 400 ∨
         c = 281 + d * 100 - 200)) :=
sorry

end NUMINAMATH_CALUDE_largest_digit_change_corrects_addition_l593_59361


namespace NUMINAMATH_CALUDE_touchdown_points_l593_59387

theorem touchdown_points (total_points : ℕ) (num_touchdowns : ℕ) (points_per_touchdown : ℕ) :
  total_points = 21 →
  num_touchdowns = 3 →
  total_points = num_touchdowns * points_per_touchdown →
  points_per_touchdown = 7 := by
  sorry

end NUMINAMATH_CALUDE_touchdown_points_l593_59387


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l593_59313

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l593_59313


namespace NUMINAMATH_CALUDE_card_flip_game_l593_59385

theorem card_flip_game (n k : ℕ) (hn : Odd n) (hk : Even k) (hkn : k < n) :
  ∀ (t : ℕ), ∃ (i : ℕ), i < n ∧ Even (t * k / n + (if i < t * k % n then 1 else 0)) := by
  sorry

end NUMINAMATH_CALUDE_card_flip_game_l593_59385


namespace NUMINAMATH_CALUDE_sum_of_digits_N_l593_59332

/-- The smallest positive integer whose digits have a product of 1728 -/
def N : ℕ := sorry

/-- The product of the digits of N is 1728 -/
axiom N_digit_product : (N.digits 10).prod = 1728

/-- N is the smallest such positive integer -/
axiom N_smallest (m : ℕ) : m > 0 → (m.digits 10).prod = 1728 → m ≥ N

/-- The sum of the digits of N is 28 -/
theorem sum_of_digits_N : (N.digits 10).sum = 28 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_N_l593_59332


namespace NUMINAMATH_CALUDE_perimeter_ratio_of_similar_triangles_l593_59355

/-- Two triangles are similar -/
def SimilarTriangles (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

/-- The similarity ratio between two triangles -/
def SimilarityRatio (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : ℝ := sorry

/-- The perimeter of a triangle -/
def Perimeter (t : Set (Fin 3 → ℝ × ℝ)) : ℝ := sorry

/-- Theorem: If two triangles are similar with a ratio of 1:2, then their perimeters have the same ratio -/
theorem perimeter_ratio_of_similar_triangles (ABC DEF : Set (Fin 3 → ℝ × ℝ)) :
  SimilarTriangles ABC DEF →
  SimilarityRatio ABC DEF = 1 / 2 →
  Perimeter ABC / Perimeter DEF = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_of_similar_triangles_l593_59355


namespace NUMINAMATH_CALUDE_min_top_managers_bound_l593_59348

/-- Represents the structure of a company with its employees and order distribution system. -/
structure Company where
  total_employees : ℕ
  direct_connections : ℕ
  distribution_days : ℕ
  (total_employees_positive : total_employees > 0)
  (direct_connections_positive : direct_connections > 0)
  (distribution_days_positive : distribution_days > 0)

/-- Calculates the minimum number of top-level managers in the company. -/
def min_top_managers (c : Company) : ℕ :=
  ((c.total_employees - 1) / (c.direct_connections^(c.distribution_days + 1) - 1)) + 1

/-- Theorem stating that a company with 50,000 employees, 7 direct connections per employee, 
    and 4 distribution days has at least 28 top-level managers. -/
theorem min_top_managers_bound (c : Company) 
  (h1 : c.total_employees = 50000)
  (h2 : c.direct_connections = 7)
  (h3 : c.distribution_days = 4) :
  min_top_managers c ≥ 28 := by
  sorry

#eval min_top_managers ⟨50000, 7, 4, by norm_num, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_min_top_managers_bound_l593_59348


namespace NUMINAMATH_CALUDE_jakes_weight_l593_59349

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 15 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 132) : 
  jake_weight = 93 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l593_59349


namespace NUMINAMATH_CALUDE_miranda_savings_duration_l593_59392

def total_cost : ℕ := 260
def sister_contribution : ℕ := 50
def monthly_saving : ℕ := 70

theorem miranda_savings_duration :
  (total_cost - sister_contribution) / monthly_saving = 3 := by
  sorry

end NUMINAMATH_CALUDE_miranda_savings_duration_l593_59392


namespace NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l593_59356

/-- The volume of a cube given its side length -/
def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

/-- The surface area of a cube given its side length -/
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length ^ 2

theorem volume_of_cube_with_triple_surface_area :
  ∀ (side_length1 side_length2 : ℝ),
  side_length1 > 0 →
  side_length2 > 0 →
  cube_volume side_length1 = 8 →
  cube_surface_area side_length2 = 3 * cube_surface_area side_length1 →
  cube_volume side_length2 = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l593_59356


namespace NUMINAMATH_CALUDE_product_simplification_l593_59306

theorem product_simplification : 
  (1/3 : ℚ) * 9 * (1/27 : ℚ) * 81 * (1/243 : ℚ) * 729 * (1/2187 : ℚ) * 6561 = 81 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l593_59306


namespace NUMINAMATH_CALUDE_factors_of_M_l593_59345

def M : ℕ := 2^5 * 3^4 * 5^3 * 7^2 * 11^1

theorem factors_of_M :
  (∃ (f : ℕ → ℕ), f M = 720 ∧ (∀ d : ℕ, d ∣ M ↔ d ∈ Finset.range (f M + 1))) ∧
  (∃ (g : ℕ → ℕ), g M = 120 ∧ (∀ d : ℕ, d ∣ M ∧ Odd d ↔ d ∈ Finset.range (g M + 1))) :=
by sorry

end NUMINAMATH_CALUDE_factors_of_M_l593_59345


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l593_59377

theorem relationship_between_exponents (a b c d x y q z : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b) 
  (h3 : c^(4*y) = a^(3*z)) 
  (h4 : c^(4*y) = d) 
  (h5 : a ≠ 0) 
  (h6 : c ≠ 0) : 
  9*q*z = 8*x*y := by
sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l593_59377


namespace NUMINAMATH_CALUDE_factor_implies_q_value_l593_59347

theorem factor_implies_q_value (m q : ℤ) : 
  (∃ k : ℤ, m^2 - q*m - 24 = (m - 8) * k) → q = 5 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_q_value_l593_59347


namespace NUMINAMATH_CALUDE_seating_solution_l593_59322

/-- A seating arrangement with rows of 6 or 7 people. -/
structure SeatingArrangement where
  rows_with_7 : ℕ
  rows_with_6 : ℕ
  total_people : ℕ
  h1 : total_people = 7 * rows_with_7 + 6 * rows_with_6
  h2 : total_people = 59

/-- The solution to the seating arrangement problem. -/
theorem seating_solution (s : SeatingArrangement) : s.rows_with_7 = 5 := by
  sorry

#check seating_solution

end NUMINAMATH_CALUDE_seating_solution_l593_59322


namespace NUMINAMATH_CALUDE_bryan_pushups_l593_59320

/-- The number of push-up sets Bryan does -/
def total_sets : ℕ := 15

/-- The number of push-ups Bryan intends to do in each set -/
def pushups_per_set : ℕ := 18

/-- The number of push-ups Bryan doesn't do in the last set due to exhaustion -/
def missed_pushups : ℕ := 12

/-- The actual number of push-ups Bryan does in the last set -/
def last_set_pushups : ℕ := pushups_per_set - missed_pushups

/-- The total number of push-ups Bryan does -/
def total_pushups : ℕ := (total_sets - 1) * pushups_per_set + last_set_pushups

theorem bryan_pushups : total_pushups = 258 := by
  sorry

end NUMINAMATH_CALUDE_bryan_pushups_l593_59320


namespace NUMINAMATH_CALUDE_largest_parallelogram_perimeter_l593_59312

/-- Triangle with sides 13, 13, and 12 -/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Parallelogram formed by four copies of a triangle -/
def Parallelogram (t : Triangle) :=
  { p : ℝ // ∃ (a b c d : ℝ), 
    a + b + c + d = p ∧
    a ≤ t.side1 ∧ b ≤ t.side1 ∧ c ≤ t.side2 ∧ d ≤ t.side3 }

/-- The theorem stating the largest possible perimeter of the parallelogram -/
theorem largest_parallelogram_perimeter :
  let t : Triangle := { side1 := 13, side2 := 13, side3 := 12 }
  ∀ p : Parallelogram t, p.val ≤ 76 :=
by sorry

end NUMINAMATH_CALUDE_largest_parallelogram_perimeter_l593_59312


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l593_59391

theorem angle_terminal_side_point (α : Real) (m : Real) :
  m > 0 →
  (2 : Real) / Real.sqrt (4 + m^2) = 2 * Real.sqrt 5 / 5 →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l593_59391


namespace NUMINAMATH_CALUDE_field_width_l593_59380

/-- The width of a rectangular field satisfying specific conditions -/
theorem field_width : ∃ (W : ℝ), W = 10 ∧ 
  20 * W * 0.5 - 40 * 0.5 = 8 * 5 * 2 := by
  sorry

end NUMINAMATH_CALUDE_field_width_l593_59380


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l593_59314

theorem largest_angle_in_triangle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α + β = (7/5) * 90 →  -- Sum of two angles is 7/5 of a right angle
  β = α + 40 →  -- One angle is 40° larger than the other
  max α (max β γ) = 83 :=  -- The largest angle is 83°
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l593_59314


namespace NUMINAMATH_CALUDE_park_ant_count_l593_59344

/-- Represents the dimensions and ant densities of a rectangular park with a special corner area -/
structure ParkInfo where
  width : ℝ  -- width of the park in feet
  length : ℝ  -- length of the park in feet
  normal_density : ℝ  -- average number of ants per square inch in most of the park
  corner_side : ℝ  -- side length of the square corner patch in feet
  corner_density : ℝ  -- average number of ants per square inch in the corner patch

/-- Calculates the total number of ants in the park -/
def totalAnts (park : ParkInfo) : ℝ :=
  let inches_per_foot : ℝ := 12
  let park_area := park.width * park.length * inches_per_foot^2
  let corner_area := park.corner_side^2 * inches_per_foot^2
  let normal_area := park_area - corner_area
  normal_area * park.normal_density + corner_area * park.corner_density

/-- Theorem stating that the total number of ants in the given park is approximately 73 million -/
theorem park_ant_count :
  let park : ParkInfo := {
    width := 200,
    length := 500,
    normal_density := 5,
    corner_side := 50,
    corner_density := 8
  }
  abs (totalAnts park - 73000000) < 100000 := by
  sorry


end NUMINAMATH_CALUDE_park_ant_count_l593_59344


namespace NUMINAMATH_CALUDE_no_integer_root_trinomials_l593_59371

theorem no_integer_root_trinomials : ¬∃ (a b c : ℤ),
  (∃ (x₁ x₂ : ℤ), a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂) ∧
  (∃ (y₁ y₂ : ℤ), (a + 1) * y₁^2 + (b + 1) * y₁ + (c + 1) = 0 ∧ (a + 1) * y₂^2 + (b + 1) * y₂ + (c + 1) = 0 ∧ y₁ ≠ y₂) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_root_trinomials_l593_59371


namespace NUMINAMATH_CALUDE_vegetable_processing_plant_profit_l593_59304

/-- Represents the total net profit for the first n years -/
def f (n : ℕ) : ℚ :=
  500000 * n - (120000 * n + 40000 * n * (n - 1) / 2) - 720000

/-- Represents the annual average net profit for the first n years -/
def avg_profit (n : ℕ) : ℚ := f n / n

theorem vegetable_processing_plant_profit :
  (∀ k : ℕ, k < 3 → f k ≤ 0) ∧
  f 3 > 0 ∧
  (∀ n : ℕ, n > 0 → avg_profit n ≤ avg_profit 6) ∧
  f 6 = 1440000 := by sorry

end NUMINAMATH_CALUDE_vegetable_processing_plant_profit_l593_59304


namespace NUMINAMATH_CALUDE_ticket_price_increase_l593_59353

theorem ticket_price_increase (last_year_income : ℝ) (club_share_last_year : ℝ) 
  (club_share_this_year : ℝ) (rental_cost : ℝ) : 
  club_share_last_year = 0.1 * last_year_income →
  rental_cost = 0.9 * last_year_income →
  club_share_this_year = 0.2 →
  (((rental_cost / (1 - club_share_this_year)) / last_year_income) - 1) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_increase_l593_59353


namespace NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l593_59364

theorem max_value_fraction (x y : ℝ) : 
  (2*x + 3*y + 4) / Real.sqrt (x^2 + y^2 + 2) ≤ Real.sqrt 29 :=
by sorry

theorem max_value_achievable : 
  ∃ x y : ℝ, (2*x + 3*y + 4) / Real.sqrt (x^2 + y^2 + 2) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l593_59364


namespace NUMINAMATH_CALUDE_vikas_rank_among_boys_l593_59329

/-- Represents the ranking information of students in a class -/
structure ClassRanking where
  total_students : ℕ
  vikas_overall_rank : ℕ
  tanvi_overall_rank : ℕ
  girls_between : ℕ
  vikas_boys_top_rank : ℕ
  vikas_bottom_rank : ℕ

/-- The theorem to prove Vikas's rank among boys -/
theorem vikas_rank_among_boys (c : ClassRanking) 
  (h1 : c.vikas_overall_rank = 9)
  (h2 : c.tanvi_overall_rank = 17)
  (h3 : c.girls_between = 2)
  (h4 : c.vikas_boys_top_rank = 4)
  (h5 : c.vikas_bottom_rank = 18) :
  c.vikas_boys_top_rank = 4 := by
  sorry


end NUMINAMATH_CALUDE_vikas_rank_among_boys_l593_59329


namespace NUMINAMATH_CALUDE_total_elephants_l593_59398

theorem total_elephants (we_preserve : ℕ) (gestures : ℕ) (natures_last : ℕ) : 
  we_preserve = 70 →
  gestures = 3 * we_preserve →
  natures_last = 5 * gestures →
  we_preserve + gestures + natures_last = 1330 := by
  sorry

#check total_elephants

end NUMINAMATH_CALUDE_total_elephants_l593_59398


namespace NUMINAMATH_CALUDE_trigonometric_values_l593_59390

theorem trigonometric_values : 
  (Real.sin (30 * π / 180) = 1 / 2) ∧ 
  (Real.cos (11 * π / 4) = -Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_values_l593_59390


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l593_59333

theorem algebraic_expression_equality (x y : ℝ) (h : x - 2*y = 3) : 
  4*y + 1 - 2*x = -5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l593_59333


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l593_59397

/-- Represents a systematic sampling process -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  removed_size : ℕ
  h_pop_size : population_size = 1002
  h_sample_size : sample_size = 50
  h_removed_size : removed_size = 2

/-- The probability of an individual being selected in the systematic sampling process -/
def selection_probability (s : SystematicSampling) : ℚ :=
  s.sample_size / s.population_size

theorem systematic_sampling_probability (s : SystematicSampling) :
  selection_probability s = 50 / 1002 := by
  sorry

#eval (50 : ℚ) / 1002

end NUMINAMATH_CALUDE_systematic_sampling_probability_l593_59397


namespace NUMINAMATH_CALUDE_max_k_value_l593_59326

theorem max_k_value (k : ℝ) : (∀ x : ℝ, Real.exp x ≥ k + x) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_k_value_l593_59326


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l593_59399

theorem sum_of_two_numbers (a b : ℤ) : 
  (a = 2 * b - 43) → (min a b = 19) → (a + b = 14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l593_59399


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_sum_l593_59389

theorem smallest_n_for_integer_sum : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    ¬∃ (j : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / m = j) ∧
  n = 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_sum_l593_59389


namespace NUMINAMATH_CALUDE_problem_statement_l593_59319

theorem problem_statement : 
  let N := (Real.sqrt (Real.sqrt 6 + 3) + Real.sqrt (Real.sqrt 6 - 3)) / Real.sqrt (Real.sqrt 6 + 2) - Real.sqrt (4 - 2 * Real.sqrt 3)
  N = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l593_59319


namespace NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l593_59383

def total_plants : ℕ := 5
def selected_plants : ℕ := 3

theorem probability_of_selecting_A_and_B : 
  (Nat.choose total_plants selected_plants) > 0 → 
  (Nat.choose (total_plants - 2) (selected_plants - 2)) > 0 →
  (Nat.choose (total_plants - 2) (selected_plants - 2) : ℚ) / 
  (Nat.choose total_plants selected_plants : ℚ) = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l593_59383


namespace NUMINAMATH_CALUDE_largest_pot_cost_largest_pot_cost_is_1_92_l593_59307

/-- The cost of the largest pot given specific conditions -/
theorem largest_pot_cost (num_pots : ℕ) (total_cost : ℚ) (price_diff : ℚ) (smallest_pot_odd_cents : Bool) : ℚ :=
  let smallest_pot_cost : ℚ := (total_cost - price_diff * (num_pots * (num_pots - 1) / 2)) / num_pots
  let rounded_smallest_pot_cost : ℚ := if smallest_pot_odd_cents then ⌊smallest_pot_cost * 100⌋ / 100 else ⌈smallest_pot_cost * 100⌉ / 100
  rounded_smallest_pot_cost + price_diff * (num_pots - 1)

/-- The main theorem proving the cost of the largest pot -/
theorem largest_pot_cost_is_1_92 :
  largest_pot_cost 6 (39/5) (1/4) true = 96/50 := by
  sorry

end NUMINAMATH_CALUDE_largest_pot_cost_largest_pot_cost_is_1_92_l593_59307


namespace NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l593_59330

/-- The maximum number of stamps that can be purchased with a given budget and stamp price -/
def max_stamps (budget : ℕ) (stamp_price : ℕ) : ℕ :=
  (budget / stamp_price : ℕ)

/-- Theorem: Given a stamp price of 25 cents and a budget of 5000 cents, 
    the maximum number of stamps that can be purchased is 200 -/
theorem max_stamps_with_50_dollars :
  max_stamps 5000 25 = 200 := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l593_59330


namespace NUMINAMATH_CALUDE_box_dimensions_theorem_l593_59315

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  smallest : ℝ
  middle : ℝ
  largest : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (d : BoxDimensions) : Prop :=
  d.smallest + d.largest = 17 ∧
  d.smallest + d.middle = 13 ∧
  d.middle + d.largest = 20

/-- The theorem to prove -/
theorem box_dimensions_theorem (d : BoxDimensions) :
  satisfiesConditions d → d = BoxDimensions.mk 5 8 12 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_theorem_l593_59315
