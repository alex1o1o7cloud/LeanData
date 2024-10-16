import Mathlib

namespace NUMINAMATH_CALUDE_sport_participation_l3362_336292

theorem sport_participation (total : ℕ) (football : ℕ) (basketball : ℕ) (baseball : ℕ) (all_three : ℕ)
  (h1 : total = 427)
  (h2 : football = 128)
  (h3 : basketball = 291)
  (h4 : baseball = 318)
  (h5 : all_three = 36)
  (h6 : total = football + basketball + baseball - (football_basketball + football_baseball + basketball_baseball) + all_three)
  : football_basketball + football_baseball + basketball_baseball - 3 * all_three = 274 :=
by sorry

end NUMINAMATH_CALUDE_sport_participation_l3362_336292


namespace NUMINAMATH_CALUDE_measure_six_liters_possible_l3362_336215

/-- Represents the state of milk distribution among containers -/
structure MilkState :=
  (container : ℕ)
  (jug9 : ℕ)
  (jug5 : ℕ)
  (bucket10 : ℕ)

/-- Represents a pouring action between two containers -/
inductive PourAction
  | ContainerTo9
  | ContainerTo5
  | NineToContainer
  | NineTo10
  | NineTo5
  | FiveTo9
  | FiveTo10
  | FiveToContainer

/-- Applies a pouring action to a milk state -/
def applyAction (state : MilkState) (action : PourAction) : MilkState :=
  sorry

/-- Checks if the given sequence of actions results in 6 liters in the 10-liter bucket -/
def isValidSolution (actions : List PourAction) : Bool :=
  sorry

/-- Proves that it's possible to measure out 6 liters using given containers -/
theorem measure_six_liters_possible :
  ∃ (actions : List PourAction), isValidSolution actions = true :=
sorry

end NUMINAMATH_CALUDE_measure_six_liters_possible_l3362_336215


namespace NUMINAMATH_CALUDE_min_value_expression_l3362_336247

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^3 + b^3 + 1/a^3 + b/a ≥ 53/27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3362_336247


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3362_336287

theorem min_value_of_expression (a : ℝ) (h1 : 1 < a) (h2 : a < 3/2) :
  ∃ (m : ℝ), m = 16/9 ∧ ∀ x, 1 < x ∧ x < 3/2 → (1/(3-2*x) + 2/(x-1)) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3362_336287


namespace NUMINAMATH_CALUDE_pq_length_is_eight_l3362_336244

/-- A quadrilateral with three equal sides -/
structure ThreeEqualSidesQuadrilateral where
  -- The lengths of the four sides
  pq : ℝ
  qr : ℝ
  rs : ℝ
  sp : ℝ
  -- Three sides are equal
  three_equal : pq = qr ∧ pq = sp
  -- SR length is 16
  sr_length : rs = 16
  -- Perimeter is 40
  perimeter : pq + qr + rs + sp = 40

/-- The length of PQ in a ThreeEqualSidesQuadrilateral is 8 -/
theorem pq_length_is_eight (quad : ThreeEqualSidesQuadrilateral) : quad.pq = 8 :=
by sorry

end NUMINAMATH_CALUDE_pq_length_is_eight_l3362_336244


namespace NUMINAMATH_CALUDE_sum_of_possible_numbers_l3362_336245

/-- The original number from which we remove digits -/
def original_number : ℕ := 112277

/-- The set of all possible three-digit numbers obtained by removing three digits from the original number -/
def possible_numbers : Finset ℕ := {112, 117, 122, 127, 177, 227, 277}

/-- The theorem stating that the sum of all possible three-digit numbers is 1159 -/
theorem sum_of_possible_numbers : 
  (possible_numbers.sum id) = 1159 := by sorry

end NUMINAMATH_CALUDE_sum_of_possible_numbers_l3362_336245


namespace NUMINAMATH_CALUDE_simplify_expression_l3362_336239

theorem simplify_expression (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 4*x + 4) = |x - 2| + |x + 2| :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3362_336239


namespace NUMINAMATH_CALUDE_complex_equality_squared_l3362_336285

theorem complex_equality_squared (m n : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : m * (1 + i) = 1 + n * i) : 
  ((m + n * i) / (m - n * i))^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_squared_l3362_336285


namespace NUMINAMATH_CALUDE_hawks_win_rate_theorem_l3362_336267

/-- The minimum number of additional games needed for the Hawks to reach 90% win rate -/
def min_additional_games : ℕ := 25

/-- The initial number of games played -/
def initial_games : ℕ := 5

/-- The number of games initially won by the Hawks -/
def initial_hawks_wins : ℕ := 2

/-- The target win percentage as a fraction -/
def target_win_rate : ℚ := 9/10

theorem hawks_win_rate_theorem :
  ∀ n : ℕ, 
    (initial_hawks_wins + n : ℚ) / (initial_games + n) ≥ target_win_rate ↔ 
    n ≥ min_additional_games := by
  sorry

end NUMINAMATH_CALUDE_hawks_win_rate_theorem_l3362_336267


namespace NUMINAMATH_CALUDE_minimal_fence_posts_l3362_336288

/-- Calculates the number of fence posts required for a rectangular park --/
def fence_posts (length width post_spacing : ℕ) : ℕ :=
  let long_side_posts := length / post_spacing + 1
  let short_side_posts := width / post_spacing
  long_side_posts + 2 * short_side_posts

/-- Theorem stating the minimal number of fence posts required for the given park --/
theorem minimal_fence_posts :
  fence_posts 90 45 15 = 13 := by
  sorry

end NUMINAMATH_CALUDE_minimal_fence_posts_l3362_336288


namespace NUMINAMATH_CALUDE_machine_production_l3362_336211

/-- The number of shirts a machine can make per minute -/
def shirts_per_minute : ℕ := 8

/-- The number of minutes the machine worked -/
def minutes_worked : ℕ := 2

/-- The number of shirts made by the machine -/
def shirts_made : ℕ := shirts_per_minute * minutes_worked

theorem machine_production :
  shirts_made = 16 := by sorry

end NUMINAMATH_CALUDE_machine_production_l3362_336211


namespace NUMINAMATH_CALUDE_circle_center_on_line_l3362_336208

/-- Given a circle x^2 + y^2 + Dx + Ey = 0 with center on the line x + y = l, prove D + E = -2 -/
theorem circle_center_on_line (D E l : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + D*x + E*y = 0 ∧ x + y = l) → D + E = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_on_line_l3362_336208


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3362_336233

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, 16) and (2, -8) is 9 -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 8
  let y1 : ℝ := 16
  let x2 : ℝ := 2
  let y2 : ℝ := -8
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 9 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3362_336233


namespace NUMINAMATH_CALUDE_coefficient_x2y2_l3362_336246

/-- The coefficient of x²y² in the expansion of (x+y)⁵(c+1/c)⁸ is 700 -/
theorem coefficient_x2y2 : 
  (Finset.sum Finset.univ (fun (k : Fin 6) => 
    Nat.choose 5 k.val * Nat.choose 8 4 * k.val.choose 2)) = 700 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y2_l3362_336246


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3362_336275

theorem algebraic_expression_equality (x : ℝ) : 
  3 * x^2 - 2 * x - 1 = 2 → -9 * x^2 + 6 * x - 1 = -10 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3362_336275


namespace NUMINAMATH_CALUDE_cos_negative_seventy_nine_pi_sixths_l3362_336212

theorem cos_negative_seventy_nine_pi_sixths :
  Real.cos (-79 * Real.pi / 6) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_negative_seventy_nine_pi_sixths_l3362_336212


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_for_48_l3362_336290

/-- Represents a hexagonal grid structure --/
structure HexagonalGrid where
  toothpicks : ℕ
  small_hexagons : ℕ

/-- Calculates the minimum number of toothpicks to remove to eliminate all triangles --/
def min_toothpicks_to_remove (grid : HexagonalGrid) : ℕ :=
  sorry

/-- Theorem stating the minimum number of toothpicks to remove for a specific grid --/
theorem min_toothpicks_removal_for_48 :
  ∀ (grid : HexagonalGrid),
    grid.toothpicks = 48 →
    min_toothpicks_to_remove grid = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_min_toothpicks_removal_for_48_l3362_336290


namespace NUMINAMATH_CALUDE_divisibility_count_l3362_336220

theorem divisibility_count : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (1638 : ℤ) % (n^2 - 3) = 0) ∧ 
    (∀ n : ℕ, n > 0 ∧ (1638 : ℤ) % (n^2 - 3) = 0 → n ∈ S) ∧
    S.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_count_l3362_336220


namespace NUMINAMATH_CALUDE_probability_at_least_twice_value_l3362_336271

def single_shot_probability : ℝ := 0.6
def number_of_shots : ℕ := 3

def probability_at_least_twice : ℝ :=
  (Nat.choose number_of_shots 2) * (single_shot_probability ^ 2) * (1 - single_shot_probability) +
  (Nat.choose number_of_shots 3) * (single_shot_probability ^ 3)

theorem probability_at_least_twice_value : 
  probability_at_least_twice = 81 / 125 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_twice_value_l3362_336271


namespace NUMINAMATH_CALUDE_derivative_implies_function_l3362_336253

open Real

theorem derivative_implies_function (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, deriv f x = 1 + cos x) →
  ∃ C, ∀ x, f x = x + sin x + C :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_implies_function_l3362_336253


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3362_336279

theorem greatest_divisor_with_remainders : 
  Nat.gcd (976543 - 7) (897623 - 11) = 4 := by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3362_336279


namespace NUMINAMATH_CALUDE_two_m_minus_b_is_zero_l3362_336223

/-- A line passing through two points (1, 3) and (-1, 1) -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The line passes through the points (1, 3) and (-1, 1) -/
def line_through_points (l : Line) : Prop :=
  3 = l.m * 1 + l.b ∧ 1 = l.m * (-1) + l.b

/-- Theorem stating that 2m - b = 0 for the line passing through (1, 3) and (-1, 1) -/
theorem two_m_minus_b_is_zero (l : Line) (h : line_through_points l) : 
  2 * l.m - l.b = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_m_minus_b_is_zero_l3362_336223


namespace NUMINAMATH_CALUDE_raduzhny_population_is_900_l3362_336297

/-- The number of villages in Sunny Valley -/
def num_villages : ℕ := 10

/-- The population of Znoynoe -/
def znoynoe_population : ℕ := 1000

/-- The difference between Znoynoe's population and the average village population -/
def population_difference : ℕ := 90

/-- The maximum population difference between any village and Znoynoe -/
def max_population_difference : ℕ := 100

/-- The total population of all villages except Znoynoe -/
def other_villages_population : ℕ := (num_villages - 1) * (znoynoe_population - population_difference)

/-- The population of Raduzhny -/
def raduzhny_population : ℕ := other_villages_population / (num_villages - 1)

theorem raduzhny_population_is_900 :
  raduzhny_population = 900 :=
sorry

end NUMINAMATH_CALUDE_raduzhny_population_is_900_l3362_336297


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l3362_336256

/-- 
A quadratic function y = kx^2 + 2x + 1 intersects the x-axis at two points 
if and only if k < 1 and k ≠ 0
-/
theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 + 2 * x₁ + 1 = 0 ∧ k * x₂^2 + 2 * x₂ + 1 = 0) ↔
  (k < 1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l3362_336256


namespace NUMINAMATH_CALUDE_fraction_nonnegative_l3362_336203

theorem fraction_nonnegative (x : ℝ) : 
  (x^4 - 4*x^3 + 4*x^2) / (1 - x^3) ≥ 0 ↔ x ∈ Set.Ici 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_nonnegative_l3362_336203


namespace NUMINAMATH_CALUDE_farmer_land_calculation_l3362_336276

theorem farmer_land_calculation (total_land : ℝ) : 
  0.2 * 0.5 * 0.9 * total_land = 252 → total_land = 2800 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_calculation_l3362_336276


namespace NUMINAMATH_CALUDE_problem_solution_l3362_336291

theorem problem_solution (a b c x : ℝ) 
  (h1 : a + x^2 = 2015)
  (h2 : b + x^2 = 2016)
  (h3 : c + x^2 = 2017)
  (h4 : a * b * c = 24) :
  a / (b * c) + b / (a * c) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3362_336291


namespace NUMINAMATH_CALUDE_line_with_acute_inclination_l3362_336238

/-- Given a line passing through points A(2,1) and B(1,m) with an acute angle of inclination, 
    the value of m must be less than 1. -/
theorem line_with_acute_inclination (m : ℝ) : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (1, m)
  let slope : ℝ := (m - A.2) / (B.1 - A.1)
  (0 < slope) ∧ (slope < 1) → m < 1 := by sorry

end NUMINAMATH_CALUDE_line_with_acute_inclination_l3362_336238


namespace NUMINAMATH_CALUDE_tangent_circles_large_radius_l3362_336217

/-- Two circles of radius 2 that are externally tangent to each other and internally tangent to a larger circle -/
structure TangentCircles where
  /-- The radius of the two smaller circles -/
  small_radius : ℝ
  /-- The radius of the larger circle -/
  large_radius : ℝ
  /-- The two smaller circles are externally tangent -/
  externally_tangent : small_radius + small_radius = 2 * small_radius
  /-- The smaller circles are internally tangent to the larger circle -/
  internally_tangent : large_radius = small_radius + small_radius
  /-- The smaller circles are tangent to a horizontal line -/
  tangent_to_line : small_radius > 0

/-- The radius of the larger circle in the TangentCircles configuration is 4 -/
theorem tangent_circles_large_radius (tc : TangentCircles) (h : tc.small_radius = 2) :
  tc.large_radius = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_large_radius_l3362_336217


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l3362_336299

def blue_eggs : ℕ := 12
def pink_eggs : ℕ := 5
def golden_eggs : ℕ := 3

def blue_points : ℕ := 2
def pink_points : ℕ := 3
def golden_points : ℕ := 5

def total_people : ℕ := 4

theorem easter_egg_distribution :
  let total_points := blue_eggs * blue_points + pink_eggs * pink_points + golden_eggs * golden_points
  (total_points / total_people = 13) ∧ (total_points % total_people = 2) := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l3362_336299


namespace NUMINAMATH_CALUDE_max_a_inequality_l3362_336222

theorem max_a_inequality (a : ℝ) : 
  (∀ x : ℝ, Real.sqrt (2 * x) - a ≥ Real.sqrt (9 - 5 * x)) → 
  a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_max_a_inequality_l3362_336222


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l3362_336261

theorem set_equality_implies_sum_of_powers (a b : ℝ) :
  let A : Set ℝ := {a, a^2, a*b}
  let B : Set ℝ := {1, a, b}
  A = B → a^2004 + b^2004 = 1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l3362_336261


namespace NUMINAMATH_CALUDE_local_max_implies_a_less_than_neg_one_l3362_336282

/-- Given a real number a and a function y = e^x + ax with a local maximum point greater than zero, prove that a < -1 -/
theorem local_max_implies_a_less_than_neg_one (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ IsLocalMax (fun x => Real.exp x + a * x) x) → a < -1 :=
by sorry

end NUMINAMATH_CALUDE_local_max_implies_a_less_than_neg_one_l3362_336282


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3362_336227

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) *
  (Real.sqrt 7 / Real.sqrt 8) * (Real.sqrt 9 / Real.sqrt 10) =
  3 * Real.sqrt 1050 / 320 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3362_336227


namespace NUMINAMATH_CALUDE_sin_315_degrees_l3362_336205

theorem sin_315_degrees : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l3362_336205


namespace NUMINAMATH_CALUDE_green_eyes_count_l3362_336225

/-- The number of students with green eyes in Mrs. Jensen's preschool class -/
def green_eyes : ℕ := sorry

theorem green_eyes_count : green_eyes = 12 := by
  have total_students : ℕ := 40
  have red_hair : ℕ := 3 * green_eyes
  have both : ℕ := 8
  have neither : ℕ := 4

  have h1 : total_students = (green_eyes - both) + (red_hair - both) + both + neither := by sorry
  
  sorry

end NUMINAMATH_CALUDE_green_eyes_count_l3362_336225


namespace NUMINAMATH_CALUDE_product_of_solutions_l3362_336242

theorem product_of_solutions (x₁ x₂ : ℝ) : 
  (|20 / x₁ + 1| = 4) → 
  (|20 / x₂ + 1| = 4) → 
  (x₁ ≠ x₂) →
  (x₁ * x₂ = -80 / 3) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3362_336242


namespace NUMINAMATH_CALUDE_greatest_n_less_than_200_l3362_336269

theorem greatest_n_less_than_200 :
  ∃ (n : ℕ), n < 200 ∧ 
  (∃ (k : ℕ), n = 9 * k - 2) ∧
  (∃ (l : ℕ), n = 6 * l - 4) ∧
  (∀ (m : ℕ), m < 200 ∧ 
    (∃ (p : ℕ), m = 9 * p - 2) ∧ 
    (∃ (q : ℕ), m = 6 * q - 4) → 
    m ≤ n) ∧
  n = 194 := by
sorry

end NUMINAMATH_CALUDE_greatest_n_less_than_200_l3362_336269


namespace NUMINAMATH_CALUDE_johnson_family_seating_l3362_336258

/-- The number of ways to arrange 5 boys and 4 girls in a row of 9 chairs such that at least 2 boys are next to each other -/
def seating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - 2 * (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating that the number of seating arrangements for 5 boys and 4 girls with at least 2 boys next to each other is 357120 -/
theorem johnson_family_seating :
  seating_arrangements 5 4 = 357120 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l3362_336258


namespace NUMINAMATH_CALUDE_xyz_equals_four_l3362_336204

theorem xyz_equals_four (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 4 := by
  sorry

end NUMINAMATH_CALUDE_xyz_equals_four_l3362_336204


namespace NUMINAMATH_CALUDE_math_test_difference_l3362_336257

theorem math_test_difference (total_questions word_problems addition_subtraction_problems steve_can_answer : ℕ) :
  total_questions = 45 →
  word_problems = 17 →
  addition_subtraction_problems = 28 →
  steve_can_answer = 38 →
  total_questions - steve_can_answer = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_math_test_difference_l3362_336257


namespace NUMINAMATH_CALUDE_justin_jersey_problem_l3362_336264

/-- Represents the problem of determining the number of long-sleeved jerseys Justin bought. -/
theorem justin_jersey_problem (long_sleeve_cost stripe_cost total_cost : ℕ) 
                               (stripe_count : ℕ) (total_spent : ℕ) : 
  long_sleeve_cost = 15 →
  stripe_cost = 10 →
  stripe_count = 2 →
  total_spent = 80 →
  ∃ long_sleeve_count : ℕ, 
    long_sleeve_count * long_sleeve_cost + stripe_count * stripe_cost = total_spent ∧
    long_sleeve_count = 4 :=
by sorry

end NUMINAMATH_CALUDE_justin_jersey_problem_l3362_336264


namespace NUMINAMATH_CALUDE_red_jellybeans_count_l3362_336206

/-- The number of red jellybeans in a jar -/
def num_red_jellybeans (total blue purple orange pink yellow : ℕ) : ℕ :=
  total - (blue + purple + orange + pink + yellow)

/-- Theorem stating the number of red jellybeans in the jar -/
theorem red_jellybeans_count :
  num_red_jellybeans 237 14 26 40 7 21 = 129 := by
  sorry

end NUMINAMATH_CALUDE_red_jellybeans_count_l3362_336206


namespace NUMINAMATH_CALUDE_volleyball_tournament_l3362_336280

theorem volleyball_tournament (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_l3362_336280


namespace NUMINAMATH_CALUDE_b_work_time_l3362_336202

-- Define the work completion time for A
def a_time : ℝ := 6

-- Define the total payment for A and B
def total_payment : ℝ := 3200

-- Define the time taken with C's help
def time_with_c : ℝ := 3

-- Define C's payment
def c_payment : ℝ := 400.0000000000002

-- Define B's work completion time (to be proved)
def b_time : ℝ := 8

-- Theorem statement
theorem b_work_time : 
  1 / a_time + 1 / b_time + (c_payment / total_payment) * (1 / time_with_c) = 1 / time_with_c :=
sorry

end NUMINAMATH_CALUDE_b_work_time_l3362_336202


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3362_336228

theorem isosceles_triangle_largest_angle (α β γ : Real) :
  -- The triangle is isosceles
  (α = β) →
  -- One of the angles opposite an equal side is 50°
  α = 50 →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 80°
  γ = 80 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3362_336228


namespace NUMINAMATH_CALUDE_jessie_weight_before_jogging_l3362_336234

/-- Jessie's weight before jogging, given her current weight and weight loss -/
theorem jessie_weight_before_jogging 
  (current_weight : ℕ) 
  (weight_loss : ℕ) 
  (h1 : current_weight = 67) 
  (h2 : weight_loss = 7) : 
  current_weight + weight_loss = 74 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_before_jogging_l3362_336234


namespace NUMINAMATH_CALUDE_right_quadrilateral_area_l3362_336289

/-- A quadrilateral with right angles at B and D, diagonal AC = 3, and two sides with distinct integer lengths. -/
structure RightQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  right_angle_B : AB * BC = 0
  right_angle_D : CD * DA = 0
  diagonal_AC : AB^2 + BC^2 = 9
  distinct_integer_sides : ∃ (x y : ℕ), x ≠ y ∧ ((AB = x ∧ CD = y) ∨ (AB = x ∧ DA = y) ∨ (BC = x ∧ CD = y) ∨ (BC = x ∧ DA = y))

/-- The area of a RightQuadrilateral is √2 + √5. -/
theorem right_quadrilateral_area (q : RightQuadrilateral) : Real.sqrt 2 + Real.sqrt 5 = q.AB * q.BC / 2 + q.CD * q.DA / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_quadrilateral_area_l3362_336289


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3362_336214

-- Define a quadratic trinomial
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Theorem statement
theorem quadratic_roots_property (a b c : ℝ) (h : discriminant a b c ≥ 0) :
  ∃ (x : ℝ), ¬(∀ (y : ℝ), discriminant (a^2) (b^2) (c^2) ≥ 0) ∧
  (∀ (z : ℝ), discriminant (a^3) (b^3) (c^3) ≥ 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3362_336214


namespace NUMINAMATH_CALUDE_percentage_problem_l3362_336207

theorem percentage_problem (P : ℝ) : 
  (0.15 * P * (0.5 * 4000) = 90) → P = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3362_336207


namespace NUMINAMATH_CALUDE_cos_B_value_angle_A_value_projection_BC_BA_l3362_336231

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi
axiom side_angle_correspondence : a = 2 * Real.sin (A / 2) ∧ 
                                  b = 2 * Real.sin (B / 2) ∧ 
                                  c = 2 * Real.sin (C / 2)
axiom line_condition : 2 * a * Real.cos B - b * Real.cos C = c * Real.cos B

-- Define the specific values for a and b
axiom a_value : a = 2 * Real.sqrt 3 / 3
axiom b_value : b = 2

-- Theorem statements
theorem cos_B_value : Real.cos B = 1 / 2 := by sorry

theorem angle_A_value : A = Real.arccos (Real.sqrt 3 / 3) := by sorry

theorem projection_BC_BA : a * Real.cos B = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_cos_B_value_angle_A_value_projection_BC_BA_l3362_336231


namespace NUMINAMATH_CALUDE_cubic_equation_natural_roots_l3362_336249

theorem cubic_equation_natural_roots (p : ℝ) : 
  (∃ (x y : ℕ) (z : ℝ), 
    x ≠ y ∧ 
    (5 * x^3 - 5*(p+1)*x^2 + (71*p-1)*x + 1 = 66*p) ∧
    (5 * y^3 - 5*(p+1)*y^2 + (71*p-1)*y + 1 = 66*p) ∧
    (5 * z^3 - 5*(p+1)*z^2 + (71*p-1)*z + 1 = 66*p)) ↔ 
  p = 76 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_natural_roots_l3362_336249


namespace NUMINAMATH_CALUDE_area_circle_circumscribed_equilateral_triangle_l3362_336237

/-- The area of a circle circumscribed about an equilateral triangle with side length 15 units is 75π square units. -/
theorem area_circle_circumscribed_equilateral_triangle :
  let s : ℝ := 15  -- Side length of the equilateral triangle
  let r : ℝ := s * Real.sqrt 3 / 3  -- Radius of the circumscribed circle
  let area : ℝ := π * r^2  -- Area of the circle
  area = 75 * π := by
  sorry

end NUMINAMATH_CALUDE_area_circle_circumscribed_equilateral_triangle_l3362_336237


namespace NUMINAMATH_CALUDE_train_length_l3362_336213

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 24 → ∃ length : ℝ, 
  (abs (length - 199.92) < 0.01 ∧ length = speed * (1000 / 3600) * time) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3362_336213


namespace NUMINAMATH_CALUDE_same_terminal_side_l3362_336226

theorem same_terminal_side (k : ℤ) : 
  ∃ k : ℤ, -390 = k * 360 + 330 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l3362_336226


namespace NUMINAMATH_CALUDE_average_of_B_and_C_l3362_336229

theorem average_of_B_and_C (A B C : ℕ) : 
  A + B + C = 111 →
  (A + B) / 2 = 31 →
  (A + C) / 2 = 37 →
  (B + C) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_of_B_and_C_l3362_336229


namespace NUMINAMATH_CALUDE_at_least_one_positive_discriminant_l3362_336219

theorem at_least_one_positive_discriminant 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (4 * b^2 - 4 * a * c > 0) ∨ 
  (4 * c^2 - 4 * a * b > 0) ∨ 
  (4 * a^2 - 4 * b * c > 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_positive_discriminant_l3362_336219


namespace NUMINAMATH_CALUDE_faculty_size_l3362_336260

-- Define the number of students in each category
def numeric_methods : ℕ := 230
def automatic_control : ℕ := 423
def both_subjects : ℕ := 134

-- Define the percentage of students in these subjects compared to total faculty
def percentage : ℚ := 80 / 100

-- Theorem statement
theorem faculty_size :
  ∃ (total : ℕ), 
    (numeric_methods + automatic_control - both_subjects : ℚ) = percentage * total ∧
    total = 649 := by sorry

end NUMINAMATH_CALUDE_faculty_size_l3362_336260


namespace NUMINAMATH_CALUDE_probability_one_red_ball_l3362_336221

def num_red_balls : ℕ := 3
def num_yellow_balls : ℕ := 2
def total_balls : ℕ := num_red_balls + num_yellow_balls
def num_drawn : ℕ := 2

theorem probability_one_red_ball :
  (Nat.choose num_red_balls 1 * Nat.choose num_yellow_balls 1) / Nat.choose total_balls num_drawn = 6 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_red_ball_l3362_336221


namespace NUMINAMATH_CALUDE_smallest_a_value_l3362_336232

theorem smallest_a_value (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0)
  (h3 : ∀ x : ℤ, Real.sin (a * x + b + π / 4) = Real.sin (15 * x + π / 4)) :
  a ≥ 15 ∧ ∃ a₀ : ℝ, a₀ = 15 ∧ a₀ ≥ 0 ∧
    ∀ x : ℤ, Real.sin (a₀ * x + π / 4) = Real.sin (15 * x + π / 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3362_336232


namespace NUMINAMATH_CALUDE_delta_computation_l3362_336293

-- Define the new operation
def delta (a b : ℕ) : ℕ := a^3 - b

-- State the theorem
theorem delta_computation :
  delta (5^(delta 6 8)) (4^(delta 2 7)) = 5^624 - 4 := by
  sorry

end NUMINAMATH_CALUDE_delta_computation_l3362_336293


namespace NUMINAMATH_CALUDE_roots_sum_powers_l3362_336263

theorem roots_sum_powers (p q : ℝ) : 
  p^2 - 6*p + 10 = 0 → q^2 - 6*q + 10 = 0 → p^3 + p^4*q^2 + p^2*q^4 + p*q^3 + p^5*q^3 = 38676 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l3362_336263


namespace NUMINAMATH_CALUDE_b_cubed_is_zero_l3362_336283

theorem b_cubed_is_zero (B : Matrix (Fin 3) (Fin 3) ℝ) (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_cubed_is_zero_l3362_336283


namespace NUMINAMATH_CALUDE_tims_weekly_earnings_l3362_336240

def visitors_per_day : ℕ := 100
def days_normal : ℕ := 6
def earnings_per_visit : ℚ := 1 / 100

def total_visitors : ℕ := visitors_per_day * days_normal + 2 * (visitors_per_day * days_normal)

def total_earnings : ℚ := (total_visitors : ℚ) * earnings_per_visit

theorem tims_weekly_earnings : total_earnings = 18 := by
  sorry

end NUMINAMATH_CALUDE_tims_weekly_earnings_l3362_336240


namespace NUMINAMATH_CALUDE_ball_diameter_l3362_336277

theorem ball_diameter (h : Real) (d : Real) (r : Real) : 
  h = 2 → d = 8 → r^2 = (d/2)^2 + (r - h)^2 → 2*r = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_diameter_l3362_336277


namespace NUMINAMATH_CALUDE_not_monotonic_iff_a_in_range_l3362_336274

/-- The function f(x) = (1/3)x^3 - x^2 + ax - 5 is not monotonic on [-1, 2] iff a ∈ (-3, 1) -/
theorem not_monotonic_iff_a_in_range (a : ℝ) :
  (∃ x y, x ∈ Set.Icc (-1 : ℝ) 2 ∧ y ∈ Set.Icc (-1 : ℝ) 2 ∧ x < y ∧
    ((1/3 : ℝ) * x^3 - x^2 + a*x ≥ (1/3 : ℝ) * y^3 - y^2 + a*y)) ↔
  a ∈ Set.Ioo (-3 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_not_monotonic_iff_a_in_range_l3362_336274


namespace NUMINAMATH_CALUDE_inequality_always_holds_l3362_336241

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l3362_336241


namespace NUMINAMATH_CALUDE_janet_hires_four_warehouse_workers_l3362_336266

/-- Represents the employment scenario for Janet's company --/
structure EmploymentScenario where
  total_employees : ℕ
  managers : ℕ
  warehouse_wage : ℝ
  manager_wage : ℝ
  fica_tax_rate : ℝ
  work_days : ℕ
  work_hours : ℕ
  total_cost : ℝ

/-- Calculates the number of warehouse workers in Janet's company --/
def calculate_warehouse_workers (scenario : EmploymentScenario) : ℕ :=
  scenario.total_employees - scenario.managers

/-- Theorem stating that Janet hires 4 warehouse workers --/
theorem janet_hires_four_warehouse_workers :
  let scenario : EmploymentScenario := {
    total_employees := 6,
    managers := 2,
    warehouse_wage := 15,
    manager_wage := 20,
    fica_tax_rate := 0.1,
    work_days := 25,
    work_hours := 8,
    total_cost := 22000
  }
  calculate_warehouse_workers scenario = 4 := by
  sorry


end NUMINAMATH_CALUDE_janet_hires_four_warehouse_workers_l3362_336266


namespace NUMINAMATH_CALUDE_selling_price_l3362_336281

/-- Given an original price and a percentage increase, calculate the selling price -/
theorem selling_price (a : ℝ) : (a * (1 + 0.1)) = 1.1 * a := by sorry

end NUMINAMATH_CALUDE_selling_price_l3362_336281


namespace NUMINAMATH_CALUDE_parking_lot_tires_l3362_336236

/-- Calculates the total number of tires in a parking lot with various vehicles -/
def total_tires (cars motorcycles trucks bicycles unicycles strollers : ℕ) 
  (cars_extra_tire bicycles_flat : ℕ) (unicycles_extra : ℕ) : ℕ :=
  -- 4-wheel drive cars
  (cars * 5 + cars_extra_tire) + 
  -- Motorcycles
  (motorcycles * 4) + 
  -- 6-wheel trucks
  (trucks * 7) + 
  -- Bicycles
  (bicycles * 2 - bicycles_flat) + 
  -- Unicycles
  (unicycles + unicycles_extra) + 
  -- Baby strollers
  (strollers * 4)

/-- Theorem stating the total number of tires in the parking lot -/
theorem parking_lot_tires : 
  total_tires 30 20 10 5 3 2 4 3 1 = 323 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_tires_l3362_336236


namespace NUMINAMATH_CALUDE_original_eq_general_form_l3362_336265

/-- The original quadratic equation -/
def original_equation (x : ℝ) : ℝ := 2 * (x + 2)^2 + (x + 3) * (x - 2) + 11

/-- The general form of the quadratic equation -/
def general_form (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 13

/-- Theorem stating the equivalence of the original equation and its general form -/
theorem original_eq_general_form :
  ∀ x, original_equation x = general_form x := by sorry

end NUMINAMATH_CALUDE_original_eq_general_form_l3362_336265


namespace NUMINAMATH_CALUDE_last_digit_of_expression_l3362_336278

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_expression : last_digit (287 * 287 + 269 * 269 - 2 * 287 * 269) = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_expression_l3362_336278


namespace NUMINAMATH_CALUDE_village_population_l3362_336209

/-- Given a village population with specific demographic percentages,
    calculate the total population. -/
theorem village_population (adult_percentage : ℝ) (adult_women_percentage : ℝ)
    (adult_women_count : ℕ) :
    adult_percentage = 0.9 →
    adult_women_percentage = 0.6 →
    adult_women_count = 21600 →
    ∃ total_population : ℕ,
      total_population = 40000 ∧
      (adult_percentage * adult_women_percentage * total_population : ℝ) = adult_women_count :=
by
  sorry

end NUMINAMATH_CALUDE_village_population_l3362_336209


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3362_336272

theorem inequality_solution_set (x : ℝ) :
  (x + 5) / (x^2 + 3*x + 9) ≥ 0 ↔ x ≥ -5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3362_336272


namespace NUMINAMATH_CALUDE_rearranged_box_surface_area_l3362_336243

theorem rearranged_box_surface_area :
  let original_length : ℝ := 2
  let original_width : ℝ := 1
  let original_height : ℝ := 1
  let first_cut_height : ℝ := 1/4
  let second_cut_height : ℝ := 1/3
  let piece_A_height : ℝ := first_cut_height
  let piece_B_height : ℝ := second_cut_height
  let piece_C_height : ℝ := original_height - (piece_A_height + piece_B_height)
  let new_length : ℝ := original_width * 3
  let new_width : ℝ := original_length
  let new_height : ℝ := piece_A_height + piece_B_height + piece_C_height
  let top_bottom_area : ℝ := 2 * (new_length * new_width)
  let side_area : ℝ := 2 * (new_height * new_width)
  let front_back_area : ℝ := 2 * (new_length * new_height)
  let total_surface_area : ℝ := top_bottom_area + side_area + front_back_area
  total_surface_area = 12 := by
    sorry

end NUMINAMATH_CALUDE_rearranged_box_surface_area_l3362_336243


namespace NUMINAMATH_CALUDE_sheetrock_length_l3362_336210

/-- Represents the properties of a rectangular sheetrock -/
structure Sheetrock where
  width : ℝ
  area : ℝ

/-- Theorem stating that a sheetrock with width 5 and area 30 has length 6 -/
theorem sheetrock_length (s : Sheetrock) (h1 : s.width = 5) (h2 : s.area = 30) :
  s.area / s.width = 6 := by
  sorry


end NUMINAMATH_CALUDE_sheetrock_length_l3362_336210


namespace NUMINAMATH_CALUDE_abc_divisibility_problem_l3362_336262

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) →
    ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end NUMINAMATH_CALUDE_abc_divisibility_problem_l3362_336262


namespace NUMINAMATH_CALUDE_jean_spots_l3362_336259

/-- Represents the distribution of spots on a jaguar -/
structure JaguarSpots where
  total : ℕ
  upperTorso : ℕ
  backAndHindquarters : ℕ
  sides : ℕ

/-- Checks if the spot distribution is valid according to the given conditions -/
def isValidDistribution (spots : JaguarSpots) : Prop :=
  spots.upperTorso = spots.total / 2 ∧
  spots.backAndHindquarters = spots.total / 3 ∧
  spots.sides = spots.total - spots.upperTorso - spots.backAndHindquarters

theorem jean_spots (spots : JaguarSpots) 
  (h_valid : isValidDistribution spots) 
  (h_upperTorso : spots.upperTorso = 30) : 
  spots.sides = 10 := by
  sorry

#check jean_spots

end NUMINAMATH_CALUDE_jean_spots_l3362_336259


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3362_336273

/-- Two points symmetric with respect to the origin -/
structure SymmetricPoints where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ
  symmetric : x2 = -x1 ∧ y2 = -y1

/-- The sum of coordinates for specific symmetric points -/
def coordinateSum (p : SymmetricPoints) : ℝ := p.y1 + p.x2

/-- Theorem: For points A(1,a) and B(b,2) symmetric with respect to the origin, a + b = -3 -/
theorem symmetric_points_sum :
  ∀ (p : SymmetricPoints), p.x1 = 1 ∧ p.y2 = 2 → coordinateSum p = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3362_336273


namespace NUMINAMATH_CALUDE_min_value_and_inequality_inequality_holds_l3362_336235

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1 - x^2) / x^2

theorem min_value_and_inequality (a : ℝ) :
  (∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f a x ≥ f a x_min ∧ f a x_min = 0) ↔ a = 2 :=
sorry

theorem inequality_holds (x : ℝ) (h : x > 0) : f 2 x ≥ 1 / x - Real.exp (1 - x) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_inequality_holds_l3362_336235


namespace NUMINAMATH_CALUDE_duck_purchase_difference_l3362_336248

/-- Represents the number of ducks bought by each person -/
structure DuckPurchase where
  adelaide : ℕ
  ephraim : ℕ
  kolton : ℕ

/-- The conditions of the duck purchase problem -/
def DuckProblemConditions (d : DuckPurchase) : Prop :=
  d.adelaide = 2 * d.ephraim ∧
  d.adelaide = 30 ∧
  (d.adelaide + d.ephraim + d.kolton) / 3 = 35

/-- The theorem stating the difference between Kolton's and Ephraim's duck purchases -/
theorem duck_purchase_difference (d : DuckPurchase) :
  DuckProblemConditions d → d.kolton - d.ephraim = 45 := by
  sorry


end NUMINAMATH_CALUDE_duck_purchase_difference_l3362_336248


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l3362_336251

theorem inequality_not_always_true (a b : ℝ) (h : a > b) :
  ¬ ∀ c : ℝ, a * c > b * c :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l3362_336251


namespace NUMINAMATH_CALUDE_smallest_z_value_l3362_336270

theorem smallest_z_value (w x y z : ℕ) : 
  w^3 + x^3 + y^3 = z^3 →
  w < x ∧ x < y ∧ y < z →
  Odd w ∧ Odd x ∧ Odd y ∧ Odd z →
  (∀ a b c d : ℕ, a < b ∧ b < c ∧ c < d ∧ 
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧
    a^3 + b^3 + c^3 = d^3 → z ≤ d) →
  z = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_value_l3362_336270


namespace NUMINAMATH_CALUDE_ab_plus_one_neq_a_plus_b_l3362_336230

theorem ab_plus_one_neq_a_plus_b (a b : ℝ) : ab + 1 ≠ a + b ↔ a ≠ 1 ∧ b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_one_neq_a_plus_b_l3362_336230


namespace NUMINAMATH_CALUDE_system_solution_l3362_336298

theorem system_solution (m : ℝ) : 
  (∃ x y : ℝ, x + y = 3*m ∧ x - y = 5*m ∧ 2*x + 3*y = 10) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3362_336298


namespace NUMINAMATH_CALUDE_sin_C_value_sin_law_variation_area_inequality_l3362_336268

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.S = (t.a + t.b)^2 - t.c^2 ∧ t.a + t.b = 4

-- Theorem statements
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) : 
  Real.sin t.C = 8 / 17 := by sorry

theorem sin_law_variation (t : Triangle) : 
  (t.a^2 - t.b^2) / t.c^2 = Real.sin (t.A - t.B) / Real.sin t.C := by sorry

theorem area_inequality (t : Triangle) : 
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * Real.sqrt 3 * t.S := by sorry

end NUMINAMATH_CALUDE_sin_C_value_sin_law_variation_area_inequality_l3362_336268


namespace NUMINAMATH_CALUDE_salary_comparison_l3362_336218

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) : b = 1.25 * a := by
  sorry

end NUMINAMATH_CALUDE_salary_comparison_l3362_336218


namespace NUMINAMATH_CALUDE_common_chord_equation_l3362_336286

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the common chord line
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem common_chord_equation :
  ∀ x y : ℝ, C1 x y ∧ C2 x y → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l3362_336286


namespace NUMINAMATH_CALUDE_pot_height_problem_shorter_pot_height_l3362_336296

theorem pot_height_problem (h₁ b₁ b₂ : ℝ) (h₁_pos : 0 < h₁) (b₁_pos : 0 < b₁) (b₂_pos : 0 < b₂) :
  h₁ / b₁ = (h₁ * b₂ / b₁) / b₂ :=
by sorry

theorem shorter_pot_height (tall_pot_height tall_pot_shadow short_pot_shadow : ℝ)
  (tall_pot_height_pos : 0 < tall_pot_height)
  (tall_pot_shadow_pos : 0 < tall_pot_shadow)
  (short_pot_shadow_pos : 0 < short_pot_shadow)
  (h_tall : tall_pot_height = 40)
  (h_tall_shadow : tall_pot_shadow = 20)
  (h_short_shadow : short_pot_shadow = 10) :
  tall_pot_height * short_pot_shadow / tall_pot_shadow = 20 :=
by sorry

end NUMINAMATH_CALUDE_pot_height_problem_shorter_pot_height_l3362_336296


namespace NUMINAMATH_CALUDE_chess_tournament_ordering_l3362_336254

/-- A structure representing a chess tournament -/
structure ChessTournament (N : ℕ) where
  beats : Fin N → Fin N → Prop

/-- The tournament property described in the problem -/
def has_tournament_property {N : ℕ} (M : ℕ) (t : ChessTournament N) : Prop :=
  ∀ (players : Fin (M + 1) → Fin N),
    (∀ i : Fin M, t.beats (players i) (players (i + 1))) →
    t.beats (players 0) (players M)

/-- The theorem to be proved -/
theorem chess_tournament_ordering
  {N M : ℕ} (h_N : N > M) (h_M : M > 1)
  (t : ChessTournament N)
  (h_prop : has_tournament_property M t) :
  ∃ f : Fin N ≃ Fin N,
    ∀ a b : Fin N, (a : ℕ) ≥ (b : ℕ) + M - 1 → t.beats (f a) (f b) :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_ordering_l3362_336254


namespace NUMINAMATH_CALUDE_triangle_area_l3362_336201

/-- The area of a triangle with vertices at (2, 2), (7, 2), and (4, 9) is 17.5 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (7, 2)
  let C : ℝ × ℝ := (4, 9)
  let base := |B.1 - A.1|
  let height := |C.2 - A.2|
  let area := (1/2) * base * height
  area = 17.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3362_336201


namespace NUMINAMATH_CALUDE_fruit_salad_count_l3362_336224

/-- Given a fruit salad with red grapes, green grapes, and raspberries, 
    this theorem proves the total number of fruits in the salad. -/
theorem fruit_salad_count (red_grapes green_grapes raspberries : ℕ) : 
  red_grapes = 67 →
  red_grapes = 3 * green_grapes + 7 →
  raspberries = green_grapes - 5 →
  red_grapes + green_grapes + raspberries = 102 := by
  sorry

#check fruit_salad_count

end NUMINAMATH_CALUDE_fruit_salad_count_l3362_336224


namespace NUMINAMATH_CALUDE_conic_section_is_hyperbola_l3362_336284

/-- The conic section represented by the equation (2x-7)^2 - 4(y+3)^2 = 169 is a hyperbola. -/
theorem conic_section_is_hyperbola :
  ∃ (a b c d e f : ℝ), 
    (∀ x y : ℝ, (2*x - 7)^2 - 4*(y + 3)^2 = 169 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0) ∧
    (a > 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_conic_section_is_hyperbola_l3362_336284


namespace NUMINAMATH_CALUDE_additional_cupcakes_count_l3362_336250

-- Define the initial number of cupcakes
def initial_cupcakes : ℕ := 30

-- Define the number of cupcakes sold
def sold_cupcakes : ℕ := 9

-- Define the total number of cupcakes after making additional ones
def total_cupcakes : ℕ := 49

-- Theorem to prove
theorem additional_cupcakes_count :
  total_cupcakes - (initial_cupcakes - sold_cupcakes) = 28 :=
by sorry

end NUMINAMATH_CALUDE_additional_cupcakes_count_l3362_336250


namespace NUMINAMATH_CALUDE_radius_of_larger_circle_l3362_336294

/-- Given two identical circles touching each other from the inside of a third circle,
    prove that the radius of the larger circle is 9 when the perimeter of the triangle
    formed by connecting the three centers is 18. -/
theorem radius_of_larger_circle (r R : ℝ) : r > 0 → R > r →
  (R - r) + (R - r) + 2 * r = 18 → R = 9 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_larger_circle_l3362_336294


namespace NUMINAMATH_CALUDE_angle_triple_complement_l3362_336216

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l3362_336216


namespace NUMINAMATH_CALUDE_weight_gain_theorem_l3362_336295

def weight_gain_problem (initial_weight first_month_gain second_month_gain : ℕ) : Prop :=
  initial_weight + first_month_gain + second_month_gain = 120

theorem weight_gain_theorem : 
  weight_gain_problem 70 20 30 := by sorry

end NUMINAMATH_CALUDE_weight_gain_theorem_l3362_336295


namespace NUMINAMATH_CALUDE_negative_fifteen_inequality_l3362_336255

theorem negative_fifteen_inequality (a b : ℝ) (h : a > b) : -15 * a < -15 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_fifteen_inequality_l3362_336255


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l3362_336200

-- Problem 1
theorem problem_one : 
  Real.rpow 0.064 (-1/3) - Real.rpow (-1/8) 0 + Real.rpow 16 (3/4) + Real.rpow 0.25 (1/2) = 10 := by
  sorry

-- Problem 2
theorem problem_two :
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l3362_336200


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3362_336252

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im ((1 + 2*i) / (2 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3362_336252
