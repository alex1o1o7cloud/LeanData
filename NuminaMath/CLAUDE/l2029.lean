import Mathlib

namespace NUMINAMATH_CALUDE_sum_and_cube_sum_divisibility_l2029_202951

theorem sum_and_cube_sum_divisibility (x y : ℤ) :
  (6 ∣ (x + y)) ↔ (6 ∣ (x^3 + y^3)) := by sorry

end NUMINAMATH_CALUDE_sum_and_cube_sum_divisibility_l2029_202951


namespace NUMINAMATH_CALUDE_sarah_game_multiple_l2029_202958

/-- The game's formula to predict marriage age -/
def marriage_age_formula (name_length : ℕ) (current_age : ℕ) (multiple : ℕ) : ℕ :=
  name_length + multiple * current_age

/-- Proof that the multiple in Sarah's game is 2 -/
theorem sarah_game_multiple : ∃ (multiple : ℕ), 
  marriage_age_formula 5 9 multiple = 23 ∧ multiple = 2 :=
by sorry

end NUMINAMATH_CALUDE_sarah_game_multiple_l2029_202958


namespace NUMINAMATH_CALUDE_saree_price_calculation_l2029_202964

theorem saree_price_calculation (final_price : ℝ) : 
  final_price = 304 → 
  ∃ (original_price : ℝ), 
    original_price * (1 - 0.20) * (1 - 0.05) = final_price ∧
    original_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l2029_202964


namespace NUMINAMATH_CALUDE_stability_of_nonlinear_eq_l2029_202947

/-- The nonlinear differential equation dx/dt = 1 - x^2(t) -/
def diff_eq (x : ℝ → ℝ) : Prop :=
  ∀ t, deriv x t = 1 - (x t)^2

/-- Definition of an equilibrium point -/
def is_equilibrium_point (x : ℝ) (eq : (ℝ → ℝ) → Prop) : Prop :=
  eq (λ _ => x)

/-- Definition of asymptotic stability -/
def is_asymptotically_stable (x : ℝ) (eq : (ℝ → ℝ) → Prop) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x₀, |x₀ - x| < δ → 
    ∀ sol, eq sol → sol 0 = x₀ → ∀ t ≥ 0, |sol t - x| < ε

/-- Definition of instability -/
def is_unstable (x : ℝ) (eq : (ℝ → ℝ) → Prop) : Prop :=
  ∃ ε > 0, ∀ δ > 0, ∃ x₀, |x₀ - x| < δ ∧
    ∃ sol, eq sol ∧ sol 0 = x₀ ∧ ∃ t ≥ 0, |sol t - x| ≥ ε

/-- Theorem about the stability of the nonlinear differential equation -/
theorem stability_of_nonlinear_eq :
  (is_equilibrium_point 1 diff_eq ∧ is_equilibrium_point (-1) diff_eq) ∧
  (is_asymptotically_stable 1 diff_eq) ∧
  (is_unstable (-1) diff_eq) :=
sorry

end NUMINAMATH_CALUDE_stability_of_nonlinear_eq_l2029_202947


namespace NUMINAMATH_CALUDE_interior_angle_sum_increases_interior_angle_sum_formula_l2029_202949

/-- The sum of interior angles of a polygon with k sides -/
def interior_angle_sum (k : ℕ) : ℝ := (k - 2) * 180

/-- Theorem: The sum of interior angles increases as the number of sides increases -/
theorem interior_angle_sum_increases (k : ℕ) (h : k ≥ 3) :
  interior_angle_sum k < interior_angle_sum (k + 1) := by
  sorry

/-- Theorem: The sum of interior angles of a k-sided polygon is (k-2) * 180° -/
theorem interior_angle_sum_formula (k : ℕ) (h : k ≥ 3) :
  interior_angle_sum k = (k - 2) * 180 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_increases_interior_angle_sum_formula_l2029_202949


namespace NUMINAMATH_CALUDE_gcd_30_problem_l2029_202967

theorem gcd_30_problem (n : ℕ) : 
  70 ≤ n ∧ n ≤ 90 → Nat.gcd 30 n = 10 → n = 70 ∨ n = 80 := by sorry

end NUMINAMATH_CALUDE_gcd_30_problem_l2029_202967


namespace NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l2029_202980

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-3/10, -9/10)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = 3 * x

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := y + 3 = -7 * x

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y := by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point := by sorry

end NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l2029_202980


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2029_202999

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 27)
  (h3 : r - p = 34) : 
  (p + q) / 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2029_202999


namespace NUMINAMATH_CALUDE_triangle_equilateral_l2029_202982

/-- A triangle with side lengths a, b, and c satisfying specific conditions is equilateral. -/
theorem triangle_equilateral (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^4 = b^4 + c^4 - b^2*c^2) (h5 : b^4 = a^4 + c^4 - a^2*c^2) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l2029_202982


namespace NUMINAMATH_CALUDE_closest_point_on_line_l2029_202939

/-- The point on the line y = 2x + 3 that is closest to (2, -1) is (-6/5, 3/5) -/
theorem closest_point_on_line (x y : ℝ) : 
  y = 2 * x + 3 →  -- line equation
  (x + 6/5)^2 + (y - 3/5)^2 ≤ (x - 2)^2 + (y + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l2029_202939


namespace NUMINAMATH_CALUDE_union_with_complement_l2029_202974

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set P
def P : Set Nat := {1, 2}

-- Define set Q
def Q : Set Nat := {1, 3}

-- Theorem statement
theorem union_with_complement :
  P ∪ (U \ Q) = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_union_with_complement_l2029_202974


namespace NUMINAMATH_CALUDE_city_distance_l2029_202994

/-- The distance between Hallelujah City and San Pedro -/
def distance : ℝ := 1074

/-- The distance from San Pedro where the planes first meet -/
def first_meeting : ℝ := 437

/-- The distance from Hallelujah City where the planes meet on the return journey -/
def second_meeting : ℝ := 237

/-- The theorem stating the distance between the cities -/
theorem city_distance : 
  ∃ (v1 v2 : ℝ), v1 > v2 ∧ v1 > 0 ∧ v2 > 0 →
  first_meeting = v2 * (distance / (v1 + v2)) ∧
  second_meeting = v1 * (distance / (v1 + v2)) ∧
  distance = 1074 := by
sorry


end NUMINAMATH_CALUDE_city_distance_l2029_202994


namespace NUMINAMATH_CALUDE_total_spent_calculation_l2029_202905

-- Define the prices and quantities
def shirt_price : ℝ := 15.00
def shirt_quantity : ℕ := 4
def pants_price : ℝ := 40.00
def pants_quantity : ℕ := 2
def suit_price : ℝ := 150.00
def suit_quantity : ℕ := 1
def sweater_price : ℝ := 30.00
def sweater_quantity : ℕ := 2
def tie_price : ℝ := 20.00
def tie_quantity : ℕ := 3
def shoes_price : ℝ := 80.00
def shoes_quantity : ℕ := 1

-- Define the discount rates
def shirt_discount : ℝ := 0.20
def pants_discount : ℝ := 0.30
def suit_discount : ℝ := 0.15
def coupon_discount : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Define the theorem
theorem total_spent_calculation :
  let initial_total := shirt_price * shirt_quantity + pants_price * pants_quantity + 
                       suit_price * suit_quantity + sweater_price * sweater_quantity + 
                       tie_price * tie_quantity + shoes_price * shoes_quantity
  let discounted_shirts := shirt_price * shirt_quantity * (1 - shirt_discount)
  let discounted_pants := pants_price * pants_quantity * (1 - pants_discount)
  let discounted_suit := suit_price * suit_quantity * (1 - suit_discount)
  let discounted_total := discounted_shirts + discounted_pants + discounted_suit + 
                          sweater_price * sweater_quantity + tie_price * tie_quantity + 
                          shoes_price * shoes_quantity
  let coupon_applied := discounted_total * (1 - coupon_discount)
  let final_total := coupon_applied * (1 + sales_tax_rate)
  final_total = 407.77 := by
sorry

end NUMINAMATH_CALUDE_total_spent_calculation_l2029_202905


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2029_202921

/-- Represents the number of students in each grade and the total sample size -/
structure SchoolPopulation where
  total : Nat
  firstYear : Nat
  secondYear : Nat
  thirdYear : Nat
  sampleSize : Nat

/-- Represents the number of students sampled from each grade -/
structure StratifiedSample where
  firstYear : Nat
  secondYear : Nat
  thirdYear : Nat

/-- Function to calculate the stratified sample given a school population -/
def calculateStratifiedSample (pop : SchoolPopulation) : StratifiedSample :=
  { firstYear := pop.firstYear * pop.sampleSize / pop.total,
    secondYear := pop.secondYear * pop.sampleSize / pop.total,
    thirdYear := pop.thirdYear * pop.sampleSize / pop.total }

theorem stratified_sampling_theorem (pop : SchoolPopulation)
    (h1 : pop.total = 1000)
    (h2 : pop.firstYear = 500)
    (h3 : pop.secondYear = 300)
    (h4 : pop.thirdYear = 200)
    (h5 : pop.sampleSize = 100)
    (h6 : pop.total = pop.firstYear + pop.secondYear + pop.thirdYear) :
    let sample := calculateStratifiedSample pop
    sample.firstYear = 50 ∧ sample.secondYear = 30 ∧ sample.thirdYear = 20 :=
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2029_202921


namespace NUMINAMATH_CALUDE_profit_division_time_l2029_202925

/-- Represents the partnership problem with given conditions -/
def PartnershipProblem (initial_ratio_p initial_ratio_q initial_ratio_r : ℚ)
  (withdrawal_time : ℕ) (withdrawal_fraction : ℚ)
  (total_profit r_profit : ℚ) : Prop :=
  -- Initial ratio of shares
  initial_ratio_p + initial_ratio_q + initial_ratio_r = 1 ∧
  -- p withdraws half of the capital after two months
  withdrawal_time = 2 ∧
  withdrawal_fraction = 1/2 ∧
  -- Given total profit and r's share
  total_profit > 0 ∧
  r_profit > 0 ∧
  r_profit < total_profit

/-- Theorem stating the number of months after which the profit was divided -/
theorem profit_division_time (initial_ratio_p initial_ratio_q initial_ratio_r : ℚ)
  (withdrawal_time : ℕ) (withdrawal_fraction : ℚ)
  (total_profit r_profit : ℚ) :
  PartnershipProblem initial_ratio_p initial_ratio_q initial_ratio_r
    withdrawal_time withdrawal_fraction total_profit r_profit →
  ∃ (n : ℕ), n = 12 := by
  sorry

end NUMINAMATH_CALUDE_profit_division_time_l2029_202925


namespace NUMINAMATH_CALUDE_al_wins_probability_l2029_202942

/-- Represents the possible moves in Rock Paper Scissors -/
inductive Move
| Rock
| Paper
| Scissors

/-- The probability of Bob playing each move -/
def bobProbability : Move → ℚ
| Move.Rock => 1/3
| Move.Paper => 1/3
| Move.Scissors => 1/3

/-- Al's move is Rock -/
def alMove : Move := Move.Rock

/-- Determines if Al wins given Bob's move -/
def alWins (bobMove : Move) : Bool :=
  match bobMove with
  | Move.Scissors => true
  | _ => false

/-- The probability of Al winning -/
def probAlWins : ℚ := bobProbability Move.Scissors

theorem al_wins_probability :
  probAlWins = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_al_wins_probability_l2029_202942


namespace NUMINAMATH_CALUDE_oshea_basil_seeds_l2029_202997

/-- The number of basil seeds Oshea bought -/
def total_seeds : ℕ := sorry

/-- The number of large planters Oshea has -/
def large_planters : ℕ := 4

/-- The number of seeds each large planter can hold -/
def seeds_per_large_planter : ℕ := 20

/-- The number of seeds each small planter can hold -/
def seeds_per_small_planter : ℕ := 4

/-- The number of small planters needed to plant all the basil seeds -/
def small_planters : ℕ := 30

/-- Theorem stating that the total number of basil seeds Oshea bought is 200 -/
theorem oshea_basil_seeds : total_seeds = 200 := by sorry

end NUMINAMATH_CALUDE_oshea_basil_seeds_l2029_202997


namespace NUMINAMATH_CALUDE_twelve_percent_of_700_is_84_l2029_202935

theorem twelve_percent_of_700_is_84 : ∃ x : ℝ, (12 / 100) * x = 84 ∧ x = 700 := by
  sorry

end NUMINAMATH_CALUDE_twelve_percent_of_700_is_84_l2029_202935


namespace NUMINAMATH_CALUDE_quadratic_derivative_condition_l2029_202946

/-- Given a quadratic function f(x) = 3x² + bx + c, prove that if the derivative at x = b is 14, then b = 2 -/
theorem quadratic_derivative_condition (b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 + b * x + c
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f (b + Δx) - f b) / Δx) - 14| < ε) → 
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_derivative_condition_l2029_202946


namespace NUMINAMATH_CALUDE_total_amount_is_117_l2029_202915

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ  -- Share of x in rupees
  y : ℝ  -- Share of y in rupees
  z : ℝ  -- Share of z in rupees

/-- The conditions of the money distribution problem -/
def satisfies_conditions (d : MoneyDistribution) : Prop :=
  d.y = 27 ∧                  -- y's share is 27 rupees
  d.y = 0.45 * d.x ∧          -- y gets 45 paisa for each rupee x gets
  d.z = 0.5 * d.x             -- z gets 50 paisa for each rupee x gets

/-- The theorem stating the total amount shared -/
theorem total_amount_is_117 (d : MoneyDistribution) 
  (h : satisfies_conditions d) : 
  d.x + d.y + d.z = 117 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_117_l2029_202915


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l2029_202988

/-- The number of walnut trees in the park after planting -/
def total_trees (current_trees newly_planted_trees : ℕ) : ℕ :=
  current_trees + newly_planted_trees

/-- Theorem: The total number of walnut trees after planting is the sum of current trees and newly planted trees -/
theorem walnut_trees_after_planting 
  (current_trees : ℕ) 
  (newly_planted_trees : ℕ) :
  total_trees current_trees newly_planted_trees = current_trees + newly_planted_trees :=
by sorry

/-- Given information about the walnut trees in the park -/
def current_walnut_trees : ℕ := 22
def new_walnut_trees : ℕ := 55

/-- The total number of walnut trees after planting -/
def final_walnut_trees : ℕ := total_trees current_walnut_trees new_walnut_trees

#eval final_walnut_trees

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l2029_202988


namespace NUMINAMATH_CALUDE_cara_family_age_difference_l2029_202943

/-- The age difference between Cara's grandmother and Cara's mom -/
def age_difference (cara_age mom_age grandma_age : ℕ) : ℕ :=
  grandma_age - mom_age

theorem cara_family_age_difference :
  ∀ (cara_age mom_age grandma_age : ℕ),
    cara_age = 40 →
    mom_age = cara_age + 20 →
    grandma_age = 75 →
    age_difference cara_age mom_age grandma_age = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_cara_family_age_difference_l2029_202943


namespace NUMINAMATH_CALUDE_tournament_games_count_l2029_202913

/-- Calculates the total number of games played in a tournament given the ratio of outcomes and the number of games won. -/
def total_games (ratio_won ratio_lost ratio_tied : ℕ) (games_won : ℕ) : ℕ :=
  let games_per_ratio := games_won / ratio_won
  let games_lost := ratio_lost * games_per_ratio
  let games_tied := ratio_tied * games_per_ratio
  games_won + games_lost + games_tied

/-- Theorem stating that given a ratio of 7:4:5 for games won:lost:tied and 42 games won, the total number of games played is 96. -/
theorem tournament_games_count :
  total_games 7 4 5 42 = 96 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_count_l2029_202913


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l2029_202937

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 54)
  (h2 : 4 * (a + b + c) = 40)
  (h3 : c = a + b) :
  a^2 + b^2 + c^2 = 46 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l2029_202937


namespace NUMINAMATH_CALUDE_jen_current_age_l2029_202927

/-- Jen's age when her son was born -/
def jen_age_at_birth : ℕ := 25

/-- Relationship between Jen's age and her son's age -/
def jen_age_relation (son_age : ℕ) : ℕ := 3 * son_age - 7

/-- Theorem stating Jen's current age -/
theorem jen_current_age :
  ∃ (son_age : ℕ), jen_age_at_birth + son_age = jen_age_relation son_age ∧
                   jen_age_at_birth + son_age = 41 := by
  sorry

end NUMINAMATH_CALUDE_jen_current_age_l2029_202927


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l2029_202972

theorem quadratic_roots_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  x₁ + x₂ - x₁*x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l2029_202972


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2029_202956

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the properties of the triangle
def isIsosceles (t : Triangle) : Prop :=
  dist t.A t.B = dist t.A t.C

def pointInside (t : Triangle) : Prop :=
  -- This is a simplified condition; in reality, we'd need a more complex definition
  true

-- Define the given distances
def givenDistances (t : Triangle) : Prop :=
  dist t.A t.P = 2 ∧
  dist t.B t.P = 2 * Real.sqrt 2 ∧
  dist t.C t.P = 3

-- Theorem statement
theorem isosceles_triangle_side_length (t : Triangle) :
  isIsosceles t → pointInside t → givenDistances t →
  dist t.B t.C = 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2029_202956


namespace NUMINAMATH_CALUDE_max_shaded_area_achievable_max_area_l2029_202912

/-- Represents a rectangular picture frame made of eight identical trapezoids -/
structure PictureFrame where
  length : ℕ+
  width : ℕ+
  trapezoidArea : ℕ
  isPrime : Nat.Prime trapezoidArea

/-- Calculates the area of the shaded region in the picture frame -/
def shadedArea (frame : PictureFrame) : ℕ :=
  (frame.trapezoidArea - 1) * (3 * frame.trapezoidArea - 1)

/-- Theorem stating the maximum possible area of the shaded region -/
theorem max_shaded_area (frame : PictureFrame) :
  shadedArea frame < 2000 → shadedArea frame ≤ 1496 :=
by
  sorry

/-- Theorem proving that 1496 is achievable -/
theorem achievable_max_area :
  ∃ frame : PictureFrame, shadedArea frame = 1496 ∧ shadedArea frame < 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_max_shaded_area_achievable_max_area_l2029_202912


namespace NUMINAMATH_CALUDE_triangle_could_be_isosceles_l2029_202941

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the condition from the problem
def satisfiesCondition (t : Triangle) : Prop :=
  t.c^2 - t.a^2 + t.b^2 = (4*t.a*t.c - 2*t.b*t.c) * Real.cos t.A

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Theorem statement
theorem triangle_could_be_isosceles (t : Triangle) 
  (h : satisfiesCondition t) : 
  ∃ (t' : Triangle), satisfiesCondition t' ∧ isIsosceles t' :=
sorry

end NUMINAMATH_CALUDE_triangle_could_be_isosceles_l2029_202941


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l2029_202923

theorem absolute_value_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + y^2 = 5*x*y) : 
  |((x+y)/(x-y))| = Real.sqrt (7/3) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l2029_202923


namespace NUMINAMATH_CALUDE_expected_elderly_in_sample_l2029_202914

/-- Calculates the expected number of elderly individuals in a stratified sample -/
def expectedElderlyInSample (totalPopulation : ℕ) (elderlyPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  (elderlyPopulation * sampleSize) / totalPopulation

/-- Theorem: Expected number of elderly individuals in the sample -/
theorem expected_elderly_in_sample :
  expectedElderlyInSample 165 22 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_elderly_in_sample_l2029_202914


namespace NUMINAMATH_CALUDE_equal_sum_of_squares_l2029_202998

/-- Given a positive integer, return the sum of its digits -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The set of positive integers with at most n digits -/
def numbersWithAtMostNDigits (n : ℕ) : Set ℕ := sorry

/-- The set of positive integers with at most n digits and even digit sum -/
def evenDigitSumNumbers (n : ℕ) : Set ℕ := 
  {x ∈ numbersWithAtMostNDigits n | Even (digitSum x)}

/-- The set of positive integers with at most n digits and odd digit sum -/
def oddDigitSumNumbers (n : ℕ) : Set ℕ := 
  {x ∈ numbersWithAtMostNDigits n | Odd (digitSum x)}

/-- The sum of squares of elements in a set of natural numbers -/
def sumOfSquares (s : Set ℕ) : ℕ := sorry

theorem equal_sum_of_squares (n : ℕ) (h : n > 2) :
  sumOfSquares (evenDigitSumNumbers n) = sumOfSquares (oddDigitSumNumbers n) := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_of_squares_l2029_202998


namespace NUMINAMATH_CALUDE_first_share_interest_rate_l2029_202906

/-- Proves that the interest rate of the first type of share is 9% given the problem conditions --/
theorem first_share_interest_rate : 
  let total_investment : ℝ := 100000
  let second_share_rate : ℝ := 11
  let total_interest_rate : ℝ := 9.5
  let second_share_investment : ℝ := 25000
  let first_share_investment : ℝ := total_investment - second_share_investment
  let total_interest : ℝ := total_interest_rate / 100 * total_investment
  let second_share_interest : ℝ := second_share_rate / 100 * second_share_investment
  let first_share_interest : ℝ := total_interest - second_share_interest
  let first_share_rate : ℝ := first_share_interest / first_share_investment * 100
  first_share_rate = 9 := by
  sorry


end NUMINAMATH_CALUDE_first_share_interest_rate_l2029_202906


namespace NUMINAMATH_CALUDE_base8_12345_to_decimal_l2029_202976

/-- Converts a base-8 number to its decimal (base-10) equivalent -/
def base8_to_decimal (d1 d2 d3 d4 d5 : ℕ) : ℕ :=
  d1 * 8^4 + d2 * 8^3 + d3 * 8^2 + d4 * 8^1 + d5 * 8^0

/-- The decimal representation of 12345 in base-8 is 5349 -/
theorem base8_12345_to_decimal :
  base8_to_decimal 1 2 3 4 5 = 5349 := by
  sorry

end NUMINAMATH_CALUDE_base8_12345_to_decimal_l2029_202976


namespace NUMINAMATH_CALUDE_line_through_points_l2029_202954

def line_equation (k m x : ℝ) : ℝ := k * x + m

theorem line_through_points (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ∃ n : ℕ, b = n * a) :
  (∃ m : ℝ, line_equation k m a = a ∧ line_equation k m b = 8 * b) →
  k ∈ ({9, 15} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2029_202954


namespace NUMINAMATH_CALUDE_point_on_graph_l2029_202962

theorem point_on_graph (x y : ℝ) : 
  (x = 1 ∧ y = 4) → (y = 4 * x) := by sorry

end NUMINAMATH_CALUDE_point_on_graph_l2029_202962


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2029_202919

theorem imaginary_part_of_z (z : ℂ) : (z - 2*I) * (2 - I) = 5 → z.im = 3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2029_202919


namespace NUMINAMATH_CALUDE_solve_cricket_problem_l2029_202952

def cricket_problem (W : ℝ) : Prop :=
  let crickets_90F : ℝ := 4
  let crickets_100F : ℝ := 2 * crickets_90F
  let prop_90F : ℝ := 0.8
  let prop_100F : ℝ := 1 - prop_90F
  let total_crickets : ℝ := 72
  W * (crickets_90F * prop_90F + crickets_100F * prop_100F) = total_crickets

theorem solve_cricket_problem :
  ∃ W : ℝ, cricket_problem W ∧ W = 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_cricket_problem_l2029_202952


namespace NUMINAMATH_CALUDE_antonia_pillbox_weeks_l2029_202993

/-- Represents the number of weeks Antonia filled her pillbox -/
def weeks_filled (total_pills : ℕ) (pills_per_week : ℕ) (pills_left : ℕ) : ℕ :=
  (total_pills - pills_left) / pills_per_week

/-- Theorem stating that Antonia filled her pillbox for 2 weeks -/
theorem antonia_pillbox_weeks :
  let num_supplements : ℕ := 5
  let bottles_120 : ℕ := 3
  let bottles_30 : ℕ := 2
  let days_in_week : ℕ := 7
  let pills_left : ℕ := 350

  let total_pills : ℕ := bottles_120 * 120 + bottles_30 * 30
  let pills_per_week : ℕ := num_supplements * days_in_week

  weeks_filled total_pills pills_per_week pills_left = 2 := by
  sorry

#check antonia_pillbox_weeks

end NUMINAMATH_CALUDE_antonia_pillbox_weeks_l2029_202993


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2029_202932

theorem quadratic_roots_theorem (c : ℝ) :
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ x = (3 + Real.sqrt c) / 2 ∨ x = (3 - Real.sqrt c) / 2) →
  c = 9/5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2029_202932


namespace NUMINAMATH_CALUDE_trapezoid_equal_area_segment_l2029_202936

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midpoint_segment : ℝ
  equal_area_segment : ℝ
  midpoint_area_ratio : ℝ × ℝ
  longer_base_diff : longer_base = shorter_base + 150
  midpoint_segment_def : midpoint_segment = shorter_base + 75
  midpoint_area_ratio_def : midpoint_area_ratio = (3, 4)

/-- The main theorem about the trapezoid -/
theorem trapezoid_equal_area_segment (t : Trapezoid) :
  ⌊(t.equal_area_segment^2) / 150⌋ = 187 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_equal_area_segment_l2029_202936


namespace NUMINAMATH_CALUDE_correct_date_l2029_202924

-- Define a type for days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a type for months
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August
  | September
  | October
  | November
  | December

-- Define a structure for a date
structure Date where
  day : Nat
  month : Month
  dayOfWeek : DayOfWeek

def nextDay (d : Date) : Date := sorry
def addDays (d : Date) (n : Nat) : Date := sorry

-- The main theorem
theorem correct_date (d : Date) : 
  (nextDay d).month ≠ Month.September ∧ 
  (addDays d 7).month = Month.September ∧
  (addDays d 2).dayOfWeek ≠ DayOfWeek.Wednesday →
  d = Date.mk 25 Month.August DayOfWeek.Wednesday :=
by sorry

end NUMINAMATH_CALUDE_correct_date_l2029_202924


namespace NUMINAMATH_CALUDE_set_intersection_equality_l2029_202989

def A : Set ℝ := {x | x ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem set_intersection_equality : A ∩ B = {x | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l2029_202989


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_product_l2029_202909

theorem sqrt_sum_difference_product (a b c d : ℝ) :
  Real.sqrt 75 + Real.sqrt 27 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 8 * Real.sqrt 3 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_product_l2029_202909


namespace NUMINAMATH_CALUDE_odd_polynomial_sum_zero_main_theorem_l2029_202960

/-- Definition of an even polynomial -/
def is_even_polynomial (p : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, p y = p (-y)

/-- Definition of an odd polynomial -/
def is_odd_polynomial (p : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, p y = -p (-y)

/-- Theorem: For an odd polynomial, the sum of values at opposite points is zero -/
theorem odd_polynomial_sum_zero (A : ℝ → ℝ) (h : is_odd_polynomial A) :
  A 3 + A (-3) = 0 :=
by sorry

/-- Main theorem to be proved -/
theorem main_theorem :
  ∃ (A : ℝ → ℝ), is_odd_polynomial A ∧ A 3 + A (-3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_odd_polynomial_sum_zero_main_theorem_l2029_202960


namespace NUMINAMATH_CALUDE_limit_of_S_is_infinity_l2029_202931

def S (n : ℕ) : ℕ := (n + 1) * n / 2

theorem limit_of_S_is_infinity :
  ∀ M : ℝ, ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (S n : ℝ) > M :=
sorry

end NUMINAMATH_CALUDE_limit_of_S_is_infinity_l2029_202931


namespace NUMINAMATH_CALUDE_meeting_attendees_l2029_202975

theorem meeting_attendees (total_handshakes : ℕ) (h : total_handshakes = 66) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_handshakes ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_meeting_attendees_l2029_202975


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l2029_202973

theorem car_fuel_efficiency (x : ℝ) : x = 40 :=
  by
  have h1 : x > 0 := sorry
  have h2 : (4 / x + 4 / 20) = (8 / x) * 1.50000000000000014 := sorry
  sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l2029_202973


namespace NUMINAMATH_CALUDE_mixed_rectangles_count_even_l2029_202929

/-- Represents a tiling of an m × n grid using 2×2 and 1×3 mosaics -/
def GridTiling (m n : ℕ) : Type := Unit

/-- Counts the number of 1×2 rectangles with one cell from a 2×2 mosaic and one from a 1×3 mosaic -/
def countMixedRectangles (tiling : GridTiling m n) : ℕ := sorry

/-- Theorem stating that the count of mixed rectangles is even -/
theorem mixed_rectangles_count_even (m n : ℕ) (tiling : GridTiling m n) :
  Even (countMixedRectangles tiling) := by sorry

end NUMINAMATH_CALUDE_mixed_rectangles_count_even_l2029_202929


namespace NUMINAMATH_CALUDE_chocolates_per_student_l2029_202979

theorem chocolates_per_student (n : ℕ) :
  (∀ (students : ℕ), students * n < 288 → students ≤ 9) ∧
  (∀ (students : ℕ), students * n > 300 → students ≥ 10) →
  n = 31 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_per_student_l2029_202979


namespace NUMINAMATH_CALUDE_exists_n_sum_diff_gt_l2029_202986

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem exists_n_sum_diff_gt (m : ℕ) : 
  ∃ n : ℕ, n > 0 ∧ sum_of_digits n - sum_of_digits (n^2) > m := by sorry

end NUMINAMATH_CALUDE_exists_n_sum_diff_gt_l2029_202986


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2029_202995

theorem sum_of_roots_quadratic (b : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ + b = 0 → x₂^2 - 2*x₂ + b = 0 → x₁ + x₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2029_202995


namespace NUMINAMATH_CALUDE_concatenation_equation_solution_l2029_202953

theorem concatenation_equation_solution :
  ∃ x : ℕ, x + (10 * x + x) = 12 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_concatenation_equation_solution_l2029_202953


namespace NUMINAMATH_CALUDE_min_square_side_length_l2029_202983

theorem min_square_side_length (square_area_min : ℝ) (circle_area_min : ℝ) :
  square_area_min = 900 →
  circle_area_min = 100 →
  ∃ (s : ℝ),
    s^2 ≥ square_area_min ∧
    π * (s/2)^2 ≥ circle_area_min ∧
    ∀ (t : ℝ), (t^2 ≥ square_area_min ∧ π * (t/2)^2 ≥ circle_area_min) → s ≤ t :=
by
  sorry

#check min_square_side_length

end NUMINAMATH_CALUDE_min_square_side_length_l2029_202983


namespace NUMINAMATH_CALUDE_integer_solution_theorem_l2029_202900

theorem integer_solution_theorem (x y z w : ℤ) :
  (x * y * z / w : ℚ) + (y * z * w / x : ℚ) + (z * w * x / y : ℚ) + (w * x * y / z : ℚ) = 4 →
  ((x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 1) ∨
   (x = -1 ∧ y = -1 ∧ z = -1 ∧ w = -1) ∨
   (x = -1 ∧ y = -1 ∧ z = 1 ∧ w = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = -1 ∧ w = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = 1 ∧ w = -1) ∨
   (x = 1 ∧ y = -1 ∧ z = -1 ∧ w = 1) ∨
   (x = 1 ∧ y = -1 ∧ z = 1 ∧ w = -1) ∨
   (x = 1 ∧ y = 1 ∧ z = -1 ∧ w = -1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_theorem_l2029_202900


namespace NUMINAMATH_CALUDE_divisibility_puzzle_l2029_202966

theorem divisibility_puzzle (a : ℤ) :
  (∃! n : Fin 4, ¬ ((n = 0 → a % 2 = 0) ∧
                    (n = 1 → a % 4 = 0) ∧
                    (n = 2 → a % 12 = 0) ∧
                    (n = 3 → a % 24 = 0))) →
  ¬ (a % 24 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_puzzle_l2029_202966


namespace NUMINAMATH_CALUDE_parallelogram_area_l2029_202944

/-- Represents a parallelogram ABCD with given properties -/
structure Parallelogram where
  perimeter : ℝ
  height_BC : ℝ
  height_CD : ℝ
  perimeter_positive : perimeter > 0
  height_BC_positive : height_BC > 0
  height_CD_positive : height_CD > 0

/-- The area of the parallelogram ABCD is 280 cm² -/
theorem parallelogram_area (ABCD : Parallelogram)
  (h_perimeter : ABCD.perimeter = 75)
  (h_height_BC : ABCD.height_BC = 14)
  (h_height_CD : ABCD.height_CD = 16) :
  ∃ (area : ℝ), area = 280 ∧ (∃ (base : ℝ), base * ABCD.height_BC = area ∧ base * ABCD.height_CD = area) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2029_202944


namespace NUMINAMATH_CALUDE_base_ten_proof_l2029_202959

/-- Given that in base b, the square of 31_b is 1021_b, prove that b = 10 -/
theorem base_ten_proof (b : ℕ) (h : b > 3) : 
  (3 * b + 1)^2 = b^3 + 2 * b + 1 → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_ten_proof_l2029_202959


namespace NUMINAMATH_CALUDE_largest_square_from_string_l2029_202933

theorem largest_square_from_string (string_length : ℝ) (side_length : ℝ) : 
  string_length = 32 →
  side_length * 4 = string_length →
  side_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_from_string_l2029_202933


namespace NUMINAMATH_CALUDE_geometric_series_terms_l2029_202926

theorem geometric_series_terms (r : ℝ) (sum : ℝ) (h_r : r = 1/4) (h_sum : sum = 40) :
  let a := sum * (1 - r)
  (a * r = 7.5) ∧ (a * r^2 = 1.875) := by
sorry

end NUMINAMATH_CALUDE_geometric_series_terms_l2029_202926


namespace NUMINAMATH_CALUDE_quadratic_sum_abc_l2029_202930

/-- Given a quadratic polynomial 12x^2 - 72x + 432, prove that when written in the form a(x+b)^2 + c, 
    the sum of a, b, and c is 333. -/
theorem quadratic_sum_abc (x : ℝ) : 
  ∃ (a b c : ℝ), (12 * x^2 - 72 * x + 432 = a * (x + b)^2 + c) ∧ (a + b + c = 333) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_abc_l2029_202930


namespace NUMINAMATH_CALUDE_cosine_sum_range_inverse_tangent_sum_l2029_202961

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sum_angles : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (sine_law : a / Real.sin A = b / Real.sin B)
  (cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)

-- Part 1
theorem cosine_sum_range (t : Triangle) (h : t.B = π/3) :
  1/2 < Real.cos t.A + Real.cos t.C ∧ Real.cos t.A + Real.cos t.C ≤ 1 :=
sorry

-- Part 2
theorem inverse_tangent_sum (t : Triangle) 
  (h1 : t.b^2 = t.a * t.c) (h2 : Real.cos t.B = 4/5) :
  1 / Real.tan t.A + 1 / Real.tan t.C = 5/3 :=
sorry

end NUMINAMATH_CALUDE_cosine_sum_range_inverse_tangent_sum_l2029_202961


namespace NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l2029_202945

theorem probability_neither_red_nor_purple (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 60) (h2 : red = 15) (h3 : purple = 3) : 
  (total - (red + purple)) / total = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l2029_202945


namespace NUMINAMATH_CALUDE_distance_and_midpoint_l2029_202955

/-- Given two points in a 2D plane, calculate their distance and midpoint -/
theorem distance_and_midpoint (p1 p2 : ℝ × ℝ) : 
  p1 = (2, 3) → p2 = (5, 9) → 
  (∃ (d : ℝ), d = 3 * Real.sqrt 5 ∧ d = Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)) ∧ 
  (∃ (m : ℝ × ℝ), m = (3.5, 6) ∧ m = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_distance_and_midpoint_l2029_202955


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2029_202991

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 10 * a * x + 25 * a = a * (x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2029_202991


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l2029_202987

/-- The quadratic function f(x) = 2x^2 - 3 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 3

/-- Theorem: The quadratic function f(x) = 2x^2 - 3 has exactly two distinct real roots -/
theorem quadratic_two_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l2029_202987


namespace NUMINAMATH_CALUDE_rowing_problem_solution_l2029_202918

/-- Represents the problem of calculating the distance to a destination given rowing speeds and time. -/
def RowingProblem (stillWaterSpeed currentVelocity totalTime : ℝ) : Prop :=
  let downstreamSpeed := stillWaterSpeed + currentVelocity
  let upstreamSpeed := stillWaterSpeed - currentVelocity
  ∃ (distance : ℝ),
    distance > 0 ∧
    distance / downstreamSpeed + distance / upstreamSpeed = totalTime

/-- Theorem stating that given the specific conditions of the problem, the distance to the destination is 2.4 km. -/
theorem rowing_problem_solution :
  RowingProblem 5 1 1 →
  ∃ (distance : ℝ), distance = 2.4 := by
  sorry

#check rowing_problem_solution

end NUMINAMATH_CALUDE_rowing_problem_solution_l2029_202918


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2029_202916

-- Define the conditions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, |x - 2| + |x + 2| > m

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 4 > 0

-- Define the relationship between p and q
theorem p_necessary_not_sufficient_for_q :
  (∃ m : ℝ, p m ∧ ¬q m) ∧ (∀ m : ℝ, q m → p m) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2029_202916


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_implies_a_greater_than_four_l2029_202985

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: If point P(4-a, 2) is in the second quadrant, then a > 4 -/
theorem point_in_second_quadrant_implies_a_greater_than_four (a : ℝ) :
  SecondQuadrant ⟨4 - a, 2⟩ → a > 4 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_implies_a_greater_than_four_l2029_202985


namespace NUMINAMATH_CALUDE_min_value_theorem_l2029_202902

-- Define the lines
def l₁ (m n x y : ℝ) : Prop := m * x + y + n = 0
def l₂ (x y : ℝ) : Prop := x + y - 1 = 0
def l₃ (x y : ℝ) : Prop := 3 * x - y - 7 = 0

-- Theorem statement
theorem min_value_theorem (m n : ℝ) 
  (h1 : ∃ x y : ℝ, l₁ m n x y ∧ l₂ x y ∧ l₃ x y) 
  (h2 : m * n > 0) :
  (1 / m + 2 / n) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2029_202902


namespace NUMINAMATH_CALUDE_range_of_r_l2029_202910

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^6 + x^4 + 4*x^2 + 4

-- State the theorem
theorem range_of_r :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ r x = y) ↔ y ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_range_of_r_l2029_202910


namespace NUMINAMATH_CALUDE_movie_of_the_year_threshold_l2029_202948

theorem movie_of_the_year_threshold (total_members : ℕ) (threshold_fraction : ℚ) : 
  total_members = 795 →
  threshold_fraction = 1/4 →
  ∃ n : ℕ, n ≥ total_members * threshold_fraction ∧ 
    ∀ m : ℕ, m ≥ total_members * threshold_fraction → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_movie_of_the_year_threshold_l2029_202948


namespace NUMINAMATH_CALUDE_first_character_lines_l2029_202934

/-- The number of lines for each character in Jerry's skit script --/
structure ScriptLines where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of Jerry's skit script --/
def script_conditions (lines : ScriptLines) : Prop :=
  lines.first = lines.second + 8 ∧
  lines.third = 2 ∧
  lines.second = 6 + 3 * lines.third

/-- Theorem stating that the first character has 20 lines --/
theorem first_character_lines (lines : ScriptLines) 
  (h : script_conditions lines) : lines.first = 20 := by
  sorry

#check first_character_lines

end NUMINAMATH_CALUDE_first_character_lines_l2029_202934


namespace NUMINAMATH_CALUDE_sweater_vest_to_shirt_ratio_l2029_202922

/-- Represents Carlton's wardrobe and outfit combinations -/
structure Wardrobe where
  sweater_vests : ℕ
  button_up_shirts : ℕ
  outfits : ℕ

/-- The ratio of sweater vests to button-up shirts is 2:1 given the conditions -/
theorem sweater_vest_to_shirt_ratio (w : Wardrobe) 
  (h1 : w.button_up_shirts = 3)
  (h2 : w.outfits = 18)
  (h3 : w.outfits = w.sweater_vests * w.button_up_shirts) :
  w.sweater_vests / w.button_up_shirts = 2 := by
  sorry

#check sweater_vest_to_shirt_ratio

end NUMINAMATH_CALUDE_sweater_vest_to_shirt_ratio_l2029_202922


namespace NUMINAMATH_CALUDE_complex_addition_complex_division_complex_multiplication_division_vector_operations_vector_dot_product_all_parts_combined_all_parts_combined_proof_l2029_202970

-- (1) (3+2i)+(\sqrt{3}-2)i
theorem complex_addition : ℂ → Prop :=
  fun z ↦ (3 + 2*Complex.I) + (Real.sqrt 3 - 2)*Complex.I = 3 + Real.sqrt 3 * Complex.I

-- (2) (9+2i)/(2+i)
theorem complex_division : ℂ → Prop :=
  fun z ↦ (9 + 2*Complex.I) / (2 + Complex.I) = 4 - Complex.I

-- (3) ((-1+i)(2+i))/(i^3)
theorem complex_multiplication_division : ℂ → Prop :=
  fun z ↦ ((-1 + Complex.I) * (2 + Complex.I)) / (Complex.I^3) = -1 - 3*Complex.I

-- (4) Given vectors a⃗=(-1,2) and b⃗=(2,1), calculate 2a⃗+3b⃗ and a⃗•b⃗
theorem vector_operations (a b : ℝ × ℝ) : Prop :=
  let a := (-1, 2)
  let b := (2, 1)
  (2 • a + 3 • b = (4, 7)) ∧ (a.1 * b.1 + a.2 * b.2 = 0)

-- (5) Given vectors a⃗ and b⃗ satisfy |a⃗|=1 and a⃗•b⃗=-1, calculate a⃗•(2a⃗-b⃗)
theorem vector_dot_product (a b : ℝ × ℝ) : Prop :=
  (a.1^2 + a.2^2 = 1) →
  (a.1 * b.1 + a.2 * b.2 = -1) →
  a.1 * (2*a.1 - b.1) + a.2 * (2*a.2 - b.2) = 3

-- Proofs are omitted
theorem all_parts_combined : Prop :=
  complex_addition 0 ∧
  complex_division 0 ∧
  complex_multiplication_division 0 ∧
  vector_operations (0, 0) (0, 0) ∧
  vector_dot_product (0, 0) (0, 0)

-- Add sorry to skip the proof
theorem all_parts_combined_proof : all_parts_combined := by sorry

end NUMINAMATH_CALUDE_complex_addition_complex_division_complex_multiplication_division_vector_operations_vector_dot_product_all_parts_combined_all_parts_combined_proof_l2029_202970


namespace NUMINAMATH_CALUDE_power_of_128_over_7_l2029_202971

theorem power_of_128_over_7 : (128 : ℝ) ^ (3/7) = 8 := by sorry

end NUMINAMATH_CALUDE_power_of_128_over_7_l2029_202971


namespace NUMINAMATH_CALUDE_dress_price_difference_l2029_202907

theorem dress_price_difference (discounted_price : ℝ) (discount_rate : ℝ) (increase_rate : ℝ) : 
  discounted_price = 61.2 ∧ discount_rate = 0.15 ∧ increase_rate = 0.25 →
  (discounted_price / (1 - discount_rate) * (1 + increase_rate)) - (discounted_price / (1 - discount_rate)) = 4.5 := by
sorry

end NUMINAMATH_CALUDE_dress_price_difference_l2029_202907


namespace NUMINAMATH_CALUDE_initial_apps_equal_final_apps_l2029_202992

/-- Proves that the initial number of apps is equal to the final number of apps -/
theorem initial_apps_equal_final_apps 
  (initial_files : ℕ) 
  (final_files : ℕ) 
  (deleted_files : ℕ) 
  (final_apps : ℕ) 
  (h1 : initial_files = 21)
  (h2 : final_files = 7)
  (h3 : deleted_files = 14)
  (h4 : final_apps = 3)
  (h5 : initial_files = final_files + deleted_files) :
  initial_files - final_files = deleted_files ∧ final_apps = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_apps_equal_final_apps_l2029_202992


namespace NUMINAMATH_CALUDE_distance_between_vertices_l2029_202957

-- Define the equation
def equation (x y : ℝ) : Prop := Real.sqrt (x^2 + y^2) + |y - 2| = 4

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = -(1/12) * x^2 + 3
def parabola2 (x y : ℝ) : Prop := y = (1/4) * x^2 - 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices : 
  ∀ x y : ℝ, equation x y → 
  (parabola1 x y ∨ parabola2 x y) → 
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l2029_202957


namespace NUMINAMATH_CALUDE_function_inequality_l2029_202968

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : Differentiable ℝ f)
  (h2 : ∀ x, (x - 1) * (deriv (deriv f) x) ≤ 0) :
  f 0 + f 2 ≤ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2029_202968


namespace NUMINAMATH_CALUDE_task_assignment_count_l2029_202977

/-- Represents the number of people who can work as both English translators and software designers -/
def both_jobs : ℕ := 1

/-- Represents the total number of people -/
def total_people : ℕ := 8

/-- Represents the number of people who can work as English translators -/
def english_translators : ℕ := 5

/-- Represents the number of people who can work as software designers -/
def software_designers : ℕ := 4

/-- Represents the number of people to be selected for the task -/
def selected_people : ℕ := 5

/-- Represents the number of people to be assigned as English translators -/
def assigned_translators : ℕ := 3

/-- Represents the number of people to be assigned as software designers -/
def assigned_designers : ℕ := 2

/-- Theorem stating that the number of ways to assign tasks is 42 -/
theorem task_assignment_count : 
  (Nat.choose (english_translators - both_jobs) assigned_translators * 
   Nat.choose (software_designers - both_jobs) assigned_designers) +
  (Nat.choose (english_translators - both_jobs) (assigned_translators - 1) * 
   Nat.choose software_designers assigned_designers) +
  (Nat.choose english_translators assigned_translators * 
   Nat.choose (software_designers - both_jobs) (assigned_designers - 1)) = 42 :=
by sorry

end NUMINAMATH_CALUDE_task_assignment_count_l2029_202977


namespace NUMINAMATH_CALUDE_square_to_circle_area_ratio_l2029_202901

theorem square_to_circle_area_ratio (s : ℝ) (h : s > 0) : 
  (s^2) / (π * s^2) = 1 / π :=
by sorry

end NUMINAMATH_CALUDE_square_to_circle_area_ratio_l2029_202901


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2029_202963

/-- Given a geometric sequence {a_n} with common ratio q, 
    if the sum of the first 3 terms is 7 and the sum of the first 6 terms is 63, 
    then q = 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence with common ratio q
  (a 1 + a 2 + a 3 = 7) →       -- Sum of first 3 terms is 7
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 63) →  -- Sum of first 6 terms is 63
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2029_202963


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2029_202920

theorem trigonometric_identity : 
  Real.cos (43 * π / 180) * Real.cos (77 * π / 180) + 
  Real.sin (43 * π / 180) * Real.cos (167 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2029_202920


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l2029_202996

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, 4*x + 6*y + k = 0 → y^2 = 32*x) ↔ k = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l2029_202996


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l2029_202908

/-- Given a set of observations with an incorrect mean due to a misrecorded value,
    calculate the corrected mean. -/
def corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean + (correct_value - incorrect_value) / (n : ℚ)

/-- Theorem stating that the corrected mean for the given problem is 45.45 -/
theorem corrected_mean_problem :
  corrected_mean 100 45 20 65 = 45.45 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_problem_l2029_202908


namespace NUMINAMATH_CALUDE_cube_of_sum_and_reciprocal_l2029_202981

theorem cube_of_sum_and_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 3) :
  (a + 1/a)^3 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_sum_and_reciprocal_l2029_202981


namespace NUMINAMATH_CALUDE_nicholas_bottle_caps_l2029_202978

theorem nicholas_bottle_caps :
  ∀ (initial : ℕ),
  initial + 85 = 93 →
  initial = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_nicholas_bottle_caps_l2029_202978


namespace NUMINAMATH_CALUDE_multiples_of_seven_l2029_202969

theorem multiples_of_seven (a b : ℕ) (q : Finset ℕ) : 
  (∃ k₁ k₂ : ℕ, a = 14 * k₁ ∧ b = 14 * k₂) →  -- a and b are multiples of 14
  (∀ x ∈ q, a ≤ x ∧ x ≤ b) →  -- q is the set of consecutive integers between a and b, inclusive
  (∀ x ∈ q, x + 1 ∈ q ∨ x = b) →  -- q contains consecutive integers
  (q.filter (λ x => x % 14 = 0)).card = 14 →  -- q contains 14 multiples of 14
  (q.filter (λ x => x % 7 = 0)).card = 27 :=  -- The number of multiples of 7 in q is 27
by sorry

end NUMINAMATH_CALUDE_multiples_of_seven_l2029_202969


namespace NUMINAMATH_CALUDE_original_number_proof_l2029_202911

theorem original_number_proof (N : ℤ) : (N + 1) % 25 = 0 → N = 24 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2029_202911


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l2029_202984

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beads : ℝ
  rings : ℝ
  silver_coins : ℝ
  gold_coins : ℝ

/-- Theorem stating the percentage of gold coins in the urn --/
theorem gold_coins_percentage (u : UrnComposition) 
  (h1 : u.beads = 0.3)
  (h2 : u.rings = 0.1)
  (h3 : u.silver_coins + u.gold_coins = 0.6)
  (h4 : u.silver_coins = 0.35 * (u.silver_coins + u.gold_coins)) :
  u.gold_coins = 0.39 := by
  sorry


end NUMINAMATH_CALUDE_gold_coins_percentage_l2029_202984


namespace NUMINAMATH_CALUDE_cheese_calories_per_serving_l2029_202950

/-- Represents the number of calories in a serving of cheese -/
def calories_per_serving (total_servings : ℕ) (eaten_servings : ℕ) (remaining_calories : ℕ) : ℕ :=
  remaining_calories / (total_servings - eaten_servings)

/-- Theorem stating that the number of calories in a serving of cheese is 110 -/
theorem cheese_calories_per_serving :
  calories_per_serving 16 5 1210 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cheese_calories_per_serving_l2029_202950


namespace NUMINAMATH_CALUDE_optimal_store_strategy_l2029_202938

/-- Represents the store's inventory and pricing strategy -/
structure Store where
  total_balls : Nat
  budget : Nat
  basketball_cost : Nat
  volleyball_cost : Nat
  basketball_price_ratio : Rat
  school_basketball_revenue : Nat
  school_volleyball_revenue : Nat
  school_volleyball_count_diff : Int

/-- Represents the store's pricing and purchase strategy -/
structure Strategy where
  basketball_price : Nat
  volleyball_price : Nat
  basketball_count : Nat
  volleyball_count : Nat

/-- Checks if the strategy satisfies all constraints -/
def is_valid_strategy (store : Store) (strategy : Strategy) : Prop :=
  strategy.basketball_count + strategy.volleyball_count = store.total_balls ∧
  strategy.basketball_count * store.basketball_cost + strategy.volleyball_count * store.volleyball_cost ≤ store.budget ∧
  strategy.basketball_price = (strategy.volleyball_price : Rat) * store.basketball_price_ratio ∧
  (store.school_basketball_revenue : Rat) / strategy.basketball_price - 
    (store.school_volleyball_revenue : Rat) / strategy.volleyball_price = store.school_volleyball_count_diff

/-- Calculates the profit after price reduction -/
def profit_after_reduction (store : Store) (strategy : Strategy) : Int :=
  (strategy.basketball_price - 3 - store.basketball_cost) * strategy.basketball_count +
  (strategy.volleyball_price - 2 - store.volleyball_cost) * strategy.volleyball_count

/-- Main theorem: Proves the optimal strategy for the store -/
theorem optimal_store_strategy (store : Store) 
    (h_store : store.total_balls = 200 ∧ 
               store.budget = 5000 ∧ 
               store.basketball_cost = 30 ∧ 
               store.volleyball_cost = 24 ∧ 
               store.basketball_price_ratio = 3/2 ∧
               store.school_basketball_revenue = 1800 ∧
               store.school_volleyball_revenue = 1500 ∧
               store.school_volleyball_count_diff = 10) :
  ∃ (strategy : Strategy),
    is_valid_strategy store strategy ∧
    strategy.basketball_price = 45 ∧
    strategy.volleyball_price = 30 ∧
    strategy.basketball_count = 33 ∧
    strategy.volleyball_count = 167 ∧
    ∀ (other_strategy : Strategy),
      is_valid_strategy store other_strategy →
      profit_after_reduction store strategy ≥ profit_after_reduction store other_strategy :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_store_strategy_l2029_202938


namespace NUMINAMATH_CALUDE_oldest_child_age_l2029_202965

/-- Represents the ages of 7 children -/
def ChildrenAges := Fin 7 → ℕ

/-- The property that each child has a different age -/
def AllDifferent (ages : ChildrenAges) : Prop :=
  ∀ i j : Fin 7, i ≠ j → ages i ≠ ages j

/-- The property that the difference in age between consecutive children is 1 year -/
def ConsecutiveDifference (ages : ChildrenAges) : Prop :=
  ∀ i : Fin 6, ages (Fin.succ i) = ages i + 1

/-- The average age of the children is 8 years -/
def AverageAge (ages : ChildrenAges) : Prop :=
  (ages 0 + ages 1 + ages 2 + ages 3 + ages 4 + ages 5 + ages 6) / 7 = 8

theorem oldest_child_age
  (ages : ChildrenAges)
  (h_diff : AllDifferent ages)
  (h_cons : ConsecutiveDifference ages)
  (h_avg : AverageAge ages) :
  ages 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l2029_202965


namespace NUMINAMATH_CALUDE_roots_and_inequality_solution_set_l2029_202917

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem roots_and_inequality_solution_set 
  (a b : ℝ) 
  (h1 : f a b (-1) = 0) 
  (h2 : f a b 2 = 0) :
  {x : ℝ | a * f a b (-2*x) > 0} = Set.Ioo (-1 : ℝ) (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_roots_and_inequality_solution_set_l2029_202917


namespace NUMINAMATH_CALUDE_candy_bar_cost_l2029_202940

/-- The cost of a candy bar given initial and remaining amounts --/
theorem candy_bar_cost (initial : ℕ) (remaining : ℕ) (cost : ℕ) :
  initial = 4 →
  remaining = 3 →
  initial = remaining + cost →
  cost = 1 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l2029_202940


namespace NUMINAMATH_CALUDE_mean_median_difference_l2029_202928

/-- Represents the absence data for a class of students -/
structure AbsenceData where
  students : ℕ
  absences : List (ℕ × ℕ)  -- (days missed, number of students)

/-- Calculates the median number of days missed -/
def median (data : AbsenceData) : ℚ := sorry

/-- Calculates the mean number of days missed -/
def mean (data : AbsenceData) : ℚ := sorry

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (data : AbsenceData) : 
  data.students = 20 ∧ 
  data.absences = [(0, 4), (1, 3), (2, 7), (3, 2), (4, 2), (5, 1), (6, 1)] →
  mean data - median data = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2029_202928


namespace NUMINAMATH_CALUDE_combined_score_is_78_l2029_202903

/-- Represents the score of a player in either football or basketball -/
structure PlayerScore where
  name : String
  score : ℕ

/-- Calculates the total score of a list of players -/
def totalScore (players : List PlayerScore) : ℕ :=
  players.foldl (fun acc p => acc + p.score) 0

/-- The combined score of football and basketball games -/
theorem combined_score_is_78 (bruce michael jack sarah andy lily : PlayerScore) :
  bruce.name = "Bruce" ∧ bruce.score = 4 ∧
  michael.name = "Michael" ∧ michael.score = 2 * bruce.score ∧
  jack.name = "Jack" ∧ jack.score = bruce.score - 1 ∧
  sarah.name = "Sarah" ∧ sarah.score = jack.score / 2 ∧
  andy.name = "Andy" ∧ andy.score = 22 ∧
  lily.name = "Lily" ∧ lily.score = andy.score + 18 →
  totalScore [bruce, michael, jack, sarah, andy, lily] = 78 := by
sorry

#eval totalScore [
  {name := "Bruce", score := 4},
  {name := "Michael", score := 8},
  {name := "Jack", score := 3},
  {name := "Sarah", score := 1},
  {name := "Andy", score := 22},
  {name := "Lily", score := 40}
]

end NUMINAMATH_CALUDE_combined_score_is_78_l2029_202903


namespace NUMINAMATH_CALUDE_investment_percentage_rate_l2029_202904

/-- Given an investment scenario, prove the percentage rate of the remaining investment --/
theorem investment_percentage_rate
  (total_investment : ℝ)
  (investment_at_five_percent : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 18000)
  (h2 : investment_at_five_percent = 6000)
  (h3 : total_interest = 660)
  : (total_interest - investment_at_five_percent * 0.05) / (total_investment - investment_at_five_percent) * 100 = 3 := by
  sorry

end NUMINAMATH_CALUDE_investment_percentage_rate_l2029_202904


namespace NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l2029_202990

theorem probability_four_twos_in_five_rolls : 
  let n_rolls : ℕ := 5
  let n_sides : ℕ := 6
  let n_twos : ℕ := 4
  let p_two : ℚ := 1 / n_sides
  let p_not_two : ℚ := 1 - p_two
  Nat.choose n_rolls n_twos * p_two ^ n_twos * p_not_two ^ (n_rolls - n_twos) = 3125 / 7776 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l2029_202990
