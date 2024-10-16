import Mathlib

namespace NUMINAMATH_CALUDE_percentage_increase_l1302_130282

theorem percentage_increase (x : ℝ) (h1 : x = 62.4) (h2 : x > 52) :
  (x - 52) / 52 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1302_130282


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1302_130235

theorem fraction_to_decimal : (3 : ℚ) / 60 = (5 : ℚ) / 100 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1302_130235


namespace NUMINAMATH_CALUDE_water_distribution_l1302_130205

def total_water : ℕ := 122
def glass_5oz : ℕ := 6
def glass_8oz : ℕ := 4
def glass_7oz : ℕ := 3

def water_used : ℕ := glass_5oz * 5 + glass_8oz * 8 + glass_7oz * 7
def remaining_water : ℕ := total_water - water_used

def filled_glasses : ℕ := glass_5oz + glass_8oz + glass_7oz
def additional_4oz_glasses : ℕ := remaining_water / 4

theorem water_distribution :
  filled_glasses + additional_4oz_glasses = 22 ∧
  additional_4oz_glasses = 9 := by sorry

end NUMINAMATH_CALUDE_water_distribution_l1302_130205


namespace NUMINAMATH_CALUDE_prime_sum_floor_squared_l1302_130244

theorem prime_sum_floor_squared : ∃! (p₁ p₂ : ℕ), 
  Prime p₁ ∧ Prime p₂ ∧ p₁ ≠ p₂ ∧
  (∃ n₁ : ℕ+, 5 * p₁ = ⌊(n₁.val^2 : ℚ) / 5⌋) ∧
  (∃ n₂ : ℕ+, 5 * p₂ = ⌊(n₂.val^2 : ℚ) / 5⌋) ∧
  p₁ + p₂ = 52 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_floor_squared_l1302_130244


namespace NUMINAMATH_CALUDE_B_subset_A_iff_A_disjoint_B_iff_l1302_130267

/-- Set A defined as {x | -3 < 2x-1 < 7} -/
def A : Set ℝ := {x | -3 < 2*x-1 ∧ 2*x-1 < 7}

/-- Set B defined as {x | 2a ≤ x ≤ a+3} -/
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a+3}

/-- Theorem stating the conditions for B to be a subset of A -/
theorem B_subset_A_iff (a : ℝ) : B a ⊆ A ↔ -1/2 < a ∧ a < 1 := by sorry

/-- Theorem stating the conditions for A and B to be disjoint -/
theorem A_disjoint_B_iff (a : ℝ) : A ∩ B a = ∅ ↔ a ≤ -4 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_B_subset_A_iff_A_disjoint_B_iff_l1302_130267


namespace NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l1302_130247

theorem negation_of_cube_odd_is_odd :
  (¬ ∀ n : ℤ, Odd n → Odd (n^3)) ↔ (∃ n : ℤ, Odd n ∧ ¬Odd (n^3)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l1302_130247


namespace NUMINAMATH_CALUDE_origin_inside_ellipse_iff_k_range_l1302_130288

/-- The ellipse equation -/
def ellipse_equation (k x y : ℝ) : ℝ := k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1

/-- The origin is inside the ellipse -/
def origin_inside_ellipse (k : ℝ) : Prop := ellipse_equation k 0 0 < 0

/-- Theorem: The origin is inside the ellipse if and only if 0 < |k| < 1 -/
theorem origin_inside_ellipse_iff_k_range (k : ℝ) : 
  origin_inside_ellipse k ↔ 0 < |k| ∧ |k| < 1 := by sorry

end NUMINAMATH_CALUDE_origin_inside_ellipse_iff_k_range_l1302_130288


namespace NUMINAMATH_CALUDE_minimum_cost_theorem_l1302_130225

/-- Represents the cost and capacity of buses -/
structure BusType where
  cost : ℕ
  capacity : ℕ

/-- Represents the problem setup -/
structure BusRentalProblem where
  busA : BusType
  busB : BusType
  totalPeople : ℕ
  totalBuses : ℕ
  costOneEach : ℕ
  costTwoAThreeB : ℕ

/-- Calculates the total cost for a given number of each bus type -/
def totalCost (problem : BusRentalProblem) (numA : ℕ) : ℕ :=
  numA * problem.busA.cost + (problem.totalBuses - numA) * problem.busB.cost

/-- Calculates the total capacity for a given number of each bus type -/
def totalCapacity (problem : BusRentalProblem) (numA : ℕ) : ℕ :=
  numA * problem.busA.capacity + (problem.totalBuses - numA) * problem.busB.capacity

/-- The main theorem to prove -/
theorem minimum_cost_theorem (problem : BusRentalProblem) 
  (h1 : problem.busA.cost + problem.busB.cost = problem.costOneEach)
  (h2 : 2 * problem.busA.cost + 3 * problem.busB.cost = problem.costTwoAThreeB)
  (h3 : problem.busA.capacity = 15)
  (h4 : problem.busB.capacity = 25)
  (h5 : problem.totalPeople = 170)
  (h6 : problem.totalBuses = 8)
  (h7 : problem.costOneEach = 500)
  (h8 : problem.costTwoAThreeB = 1300) :
  ∃ (numA : ℕ), 
    numA ≤ problem.totalBuses ∧ 
    totalCapacity problem numA ≥ problem.totalPeople ∧
    totalCost problem numA = 2100 ∧
    ∀ (k : ℕ), k ≤ problem.totalBuses → 
      totalCapacity problem k ≥ problem.totalPeople → 
      totalCost problem k ≥ 2100 := by
  sorry


end NUMINAMATH_CALUDE_minimum_cost_theorem_l1302_130225


namespace NUMINAMATH_CALUDE_ohara_quadruple_example_l1302_130258

theorem ohara_quadruple_example :
  ∀ (x : ℤ), (Real.sqrt 9 + Real.sqrt 16 + 3^2 : ℝ) = x → x = 16 := by
sorry

end NUMINAMATH_CALUDE_ohara_quadruple_example_l1302_130258


namespace NUMINAMATH_CALUDE_sum_of_modified_integers_l1302_130214

theorem sum_of_modified_integers (P : ℤ) (x y : ℤ) (h : x + y = P) :
  3 * (x + 5) + 3 * (y + 5) = 3 * P + 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_modified_integers_l1302_130214


namespace NUMINAMATH_CALUDE_first_negative_term_is_14th_l1302_130268

/-- The index of the first negative term in the arithmetic sequence -/
def first_negative_term_index : ℕ := 14

/-- The first term of the arithmetic sequence -/
def a₁ : ℤ := 51

/-- The common difference of the arithmetic sequence -/
def d : ℤ := -4

/-- The general term of the arithmetic sequence -/
def aₙ (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem first_negative_term_is_14th :
  (∀ k < first_negative_term_index, aₙ k ≥ 0) ∧
  aₙ first_negative_term_index < 0 := by
  sorry

#eval aₙ first_negative_term_index

end NUMINAMATH_CALUDE_first_negative_term_is_14th_l1302_130268


namespace NUMINAMATH_CALUDE_james_played_five_rounds_l1302_130294

/-- Represents the quiz bowl scoring system and James' performance -/
structure QuizBowl where
  pointsPerCorrectAnswer : ℕ
  questionsPerRound : ℕ
  bonusPoints : ℕ
  totalPoints : ℕ
  missedQuestions : ℕ

/-- Calculates the number of rounds played given the quiz bowl parameters -/
def calculateRounds (qb : QuizBowl) : ℕ :=
  sorry

/-- Theorem stating that James played 5 rounds -/
theorem james_played_five_rounds (qb : QuizBowl) 
  (h1 : qb.pointsPerCorrectAnswer = 2)
  (h2 : qb.questionsPerRound = 5)
  (h3 : qb.bonusPoints = 4)
  (h4 : qb.totalPoints = 66)
  (h5 : qb.missedQuestions = 1) :
  calculateRounds qb = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_played_five_rounds_l1302_130294


namespace NUMINAMATH_CALUDE_union_A_complement_B_l1302_130224

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- Theorem statement
theorem union_A_complement_B : 
  A ∪ (U \ B) = Iic 1 ∪ Ioi 2 := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_l1302_130224


namespace NUMINAMATH_CALUDE_student_task_assignment_l1302_130217

/-- The number of ways to assign students to tasks under specific conditions -/
def assignment_count (n : ℕ) (m : ℕ) (k : ℕ) : ℕ :=
  Nat.choose k 1 * Nat.choose m 2 * (n - 1)^(n - 1) + Nat.choose k 2 * (n - 1)^(n - 1)

/-- Theorem stating the number of ways to assign 5 students to 4 tasks under given conditions -/
theorem student_task_assignment :
  assignment_count 4 4 3 = Nat.choose 3 1 * Nat.choose 4 2 * 3^3 + Nat.choose 3 2 * 3^3 :=
by sorry

end NUMINAMATH_CALUDE_student_task_assignment_l1302_130217


namespace NUMINAMATH_CALUDE_sum_s_1_to_321_l1302_130270

-- Define s(n) as the sum of all odd digits of n
def s (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_s_1_to_321 : 
  (Finset.range 321).sum s + s 321 = 1727 := by sorry

end NUMINAMATH_CALUDE_sum_s_1_to_321_l1302_130270


namespace NUMINAMATH_CALUDE_spider_reachable_points_l1302_130265

/-- A cube with edge length 1 -/
structure Cube where
  edge_length : ℝ
  edge_length_pos : edge_length = 1

/-- A point on the surface of a cube -/
structure CubePoint (c : Cube) where
  x : ℝ
  y : ℝ
  z : ℝ
  on_surface : (x = 0 ∨ x = c.edge_length ∨ y = 0 ∨ y = c.edge_length ∨ z = 0 ∨ z = c.edge_length) ∧
               0 ≤ x ∧ x ≤ c.edge_length ∧
               0 ≤ y ∧ y ≤ c.edge_length ∧
               0 ≤ z ∧ z ≤ c.edge_length

/-- The distance between two points on the surface of a cube -/
def surface_distance (c : Cube) (p1 p2 : CubePoint c) : ℝ :=
  sorry -- Definition of surface distance calculation

/-- The set of points reachable by the spider in 2 seconds -/
def reachable_points (c : Cube) (start : CubePoint c) : Set (CubePoint c) :=
  {p : CubePoint c | surface_distance c start p ≤ 2}

/-- Theorem: The set of points reachable by the spider in 2 seconds
    is equivalent to the set of points within 2 cm of the starting vertex -/
theorem spider_reachable_points (c : Cube) (start : CubePoint c) :
  reachable_points c start = {p : CubePoint c | surface_distance c start p ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_spider_reachable_points_l1302_130265


namespace NUMINAMATH_CALUDE_inequality_proof_l1302_130230

theorem inequality_proof (n : ℕ) : (n - 1)^(n + 1) * (n + 1)^(n - 1) < n^(2 * n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1302_130230


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1302_130298

theorem fixed_point_on_line (m : ℝ) : 
  (m + 2) * (-4/5) + (m - 3) * (4/5) + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1302_130298


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1302_130273

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 2) :
  (1 - 1 / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1302_130273


namespace NUMINAMATH_CALUDE_quadratic_sequence_problem_l1302_130257

theorem quadratic_sequence_problem (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (eq1 : y₁ + 4*y₂ + 9*y₃ + 16*y₄ + 25*y₅ = 3)
  (eq2 : 4*y₁ + 9*y₂ + 16*y₃ + 25*y₄ + 36*y₅ = 20)
  (eq3 : 9*y₁ + 16*y₂ + 25*y₃ + 36*y₄ + 49*y₅ = 150) :
  16*y₁ + 25*y₂ + 36*y₃ + 49*y₄ + 64*y₅ = 336 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_problem_l1302_130257


namespace NUMINAMATH_CALUDE_exactly_three_correct_delivery_l1302_130207

/-- The number of houses and packages -/
def n : ℕ := 5

/-- The number of correctly delivered packages -/
def k : ℕ := 3

/-- The probability of exactly k out of n packages being delivered correctly -/
def prob_correct_delivery (n k : ℕ) : ℚ :=
  (n.choose k * (n - k).factorial) / n.factorial

theorem exactly_three_correct_delivery :
  prob_correct_delivery n k = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_exactly_three_correct_delivery_l1302_130207


namespace NUMINAMATH_CALUDE_iris_blueberries_l1302_130280

/-- The number of blueberries Iris picked -/
def blueberries : ℕ := 30

/-- The number of cranberries Iris' sister picked -/
def cranberries : ℕ := 20

/-- The number of raspberries Iris' brother picked -/
def raspberries : ℕ := 10

/-- The fraction of total berries that are rotten -/
def rotten_fraction : ℚ := 1/3

/-- The fraction of fresh berries that need to be kept -/
def kept_fraction : ℚ := 1/2

/-- The number of berries they can sell -/
def sellable_berries : ℕ := 20

theorem iris_blueberries :
  blueberries = 30 ∧
  (1 - rotten_fraction) * (1 - kept_fraction) * (blueberries + cranberries + raspberries : ℚ) = sellable_berries := by
  sorry

end NUMINAMATH_CALUDE_iris_blueberries_l1302_130280


namespace NUMINAMATH_CALUDE_sock_cost_is_three_l1302_130221

/-- The cost of a uniform given the cost of socks -/
def uniform_cost (sock_cost : ℚ) : ℚ :=
  20 + 2 * 20 + (2 * 20) / 5 + sock_cost

/-- The total cost of 5 uniforms given the cost of socks -/
def total_cost (sock_cost : ℚ) : ℚ :=
  5 * uniform_cost sock_cost

theorem sock_cost_is_three :
  ∃ (sock_cost : ℚ), total_cost sock_cost = 355 ∧ sock_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_sock_cost_is_three_l1302_130221


namespace NUMINAMATH_CALUDE_range_of_a_l1302_130277

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (a + 6) + y^2 / (a - 7) = 1 ∧ 
  (∃ (b c : ℝ), (x = 0 ∧ y = b) ∨ (x = c ∧ y = 0))

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 4*x + a < 0

-- Define the theorem
theorem range_of_a : 
  (∀ a : ℝ, p a ∨ ¬(q a)) → 
  ∀ a : ℝ, a ∈ Set.Ioi (-6) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1302_130277


namespace NUMINAMATH_CALUDE_hours_until_visit_l1302_130278

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days until Joy sees her grandma -/
def days_until_visit : ℕ := 2

/-- Theorem: The number of hours until Joy sees her grandma is 48 -/
theorem hours_until_visit : days_until_visit * hours_per_day = 48 := by
  sorry

end NUMINAMATH_CALUDE_hours_until_visit_l1302_130278


namespace NUMINAMATH_CALUDE_florist_roses_l1302_130240

theorem florist_roses (initial : ℕ) : 
  (initial - 3 + 34 = 36) → initial = 5 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l1302_130240


namespace NUMINAMATH_CALUDE_unique_prime_digit_in_powers_l1302_130283

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def digit_appears_in (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k < 4 ∧ (n / (10^k)) % 10 = d

theorem unique_prime_digit_in_powers :
  ∃! p : ℕ, is_prime p ∧
    (∃ m : ℕ, is_four_digit (2^m) ∧ digit_appears_in p (2^m)) ∧
    (∃ n : ℕ, is_four_digit (5^n) ∧ digit_appears_in p (5^n)) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_digit_in_powers_l1302_130283


namespace NUMINAMATH_CALUDE_expression_value_l1302_130210

theorem expression_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 - y^2 = x + y) :
  x / y + y / x = 2 + 1 / (y^2 + y) := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1302_130210


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l1302_130233

theorem convex_polygon_sides (n : ℕ) : n > 2 → (n - 1) * 180 - 2008 < 180 ∧ 2008 < (n - 1) * 180 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l1302_130233


namespace NUMINAMATH_CALUDE_gym_cost_theorem_l1302_130271

/-- Calculates the total cost for gym memberships and personal training for one year -/
def total_gym_cost (cheap_monthly : ℝ) (cheap_signup : ℝ) (cheap_maintenance : ℝ)
                   (expensive_monthly_factor : ℝ) (expensive_signup_months : ℝ) (expensive_maintenance : ℝ)
                   (signup_discount : ℝ) (cheap_pt_base : ℝ) (cheap_pt_discount : ℝ)
                   (expensive_pt_base : ℝ) (expensive_pt_discount : ℝ) : ℝ :=
  let cheap_total := cheap_monthly * 12 + cheap_signup * (1 - signup_discount) + cheap_maintenance +
                     (cheap_pt_base * 10 + cheap_pt_base * (1 - cheap_pt_discount) * 10)
  let expensive_monthly := cheap_monthly * expensive_monthly_factor
  let expensive_total := expensive_monthly * 12 + (expensive_monthly * expensive_signup_months) * (1 - signup_discount) +
                         expensive_maintenance + (expensive_pt_base * 5 + expensive_pt_base * (1 - expensive_pt_discount) * 10)
  cheap_total + expensive_total

/-- The theorem states that the total gym cost for the given parameters is $1780.50 -/
theorem gym_cost_theorem :
  total_gym_cost 10 50 30 3 4 60 0.1 25 0.2 45 0.15 = 1780.50 := by
  sorry

end NUMINAMATH_CALUDE_gym_cost_theorem_l1302_130271


namespace NUMINAMATH_CALUDE_large_circle_radius_l1302_130213

theorem large_circle_radius (C₁ C₂ C₃ C₄ O : ℝ × ℝ) (r : ℝ) :
  -- Four unit circles externally tangent in square formation
  r = 1 ∧
  dist C₁ C₂ = 2 ∧ dist C₂ C₃ = 2 ∧ dist C₃ C₄ = 2 ∧ dist C₄ C₁ = 2 ∧
  -- Large circle internally tangent to the four unit circles
  dist O C₁ = dist O C₂ ∧ dist O C₂ = dist O C₃ ∧ dist O C₃ = dist O C₄ ∧
  dist O C₁ = dist C₁ C₃ / 2 + r →
  -- Radius of the large circle
  dist O C₁ + r = Real.sqrt 2 + 2 := by
sorry


end NUMINAMATH_CALUDE_large_circle_radius_l1302_130213


namespace NUMINAMATH_CALUDE_pyramid_side_increment_l1302_130206

/-- Represents the side length increment between pyramid levels -/
def side_increment : ℕ := 2

/-- Calculates the total number of cases in a pyramid given the side increment -/
def total_cases (x : ℕ) : ℕ :=
  1 + x^2 + (x + 1)^2 + (x + 2)^2

/-- The theorem states that for a four-level pyramid with 30 total cases,
    the side length increment between levels is 2 -/
theorem pyramid_side_increment :
  total_cases side_increment = 30 := by sorry

end NUMINAMATH_CALUDE_pyramid_side_increment_l1302_130206


namespace NUMINAMATH_CALUDE_log_equation_implies_non_square_non_cube_integer_l1302_130290

-- Define the logarithm equation
def log_equation (x : ℝ) : Prop :=
  Real.log (343 : ℝ) / Real.log (3 * x + 1) = x

-- Define what it means to be a non-square, non-cube integer
def is_non_square_non_cube_integer (x : ℝ) : Prop :=
  ∃ n : ℤ, (x : ℝ) = n ∧ ¬∃ m : ℤ, n = m^2 ∧ ¬∃ k : ℤ, n = k^3

-- The theorem statement
theorem log_equation_implies_non_square_non_cube_integer :
  ∀ x : ℝ, log_equation x → is_non_square_non_cube_integer x :=
by sorry

end NUMINAMATH_CALUDE_log_equation_implies_non_square_non_cube_integer_l1302_130290


namespace NUMINAMATH_CALUDE_sqrt_4_times_9_sqrt_49_over_36_cube_root_a_to_6_sqrt_9a_squared_l1302_130254

-- Part a
theorem sqrt_4_times_9 : Real.sqrt (4 * 9) = 6 := by sorry

-- Part b
theorem sqrt_49_over_36 : Real.sqrt (49 / 36) = 7 / 6 := by sorry

-- Part c
theorem cube_root_a_to_6 (a : ℝ) : (a^6)^(1/3 : ℝ) = a^2 := by sorry

-- Part d
theorem sqrt_9a_squared (a : ℝ) : Real.sqrt (9 * a^2) = 3 * a := by sorry

end NUMINAMATH_CALUDE_sqrt_4_times_9_sqrt_49_over_36_cube_root_a_to_6_sqrt_9a_squared_l1302_130254


namespace NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l1302_130227

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem eighth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_3 : a 3 = 3)
  (h_6 : a 6 = 24) :
  a 8 = 96 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l1302_130227


namespace NUMINAMATH_CALUDE_max_moves_less_than_500000_l1302_130220

/-- Represents the maximum number of moves for a given number of cards. -/
def max_moves (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the maximum number of moves for 1000 cards is less than 500,000. -/
theorem max_moves_less_than_500000 :
  max_moves 1000 < 500000 := by
  sorry

#eval max_moves 1000  -- This will evaluate to 499500

end NUMINAMATH_CALUDE_max_moves_less_than_500000_l1302_130220


namespace NUMINAMATH_CALUDE_exists_21_game_period_l1302_130256

-- Define the type for the sequence of cumulative games
def CumulativeGames := Nat → Nat

-- Define the properties of the sequence
def ValidSequence (a : CumulativeGames) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, a (n + 7) - a n ≤ 10) ∧
  (a 0 ≥ 1) ∧ (a 42 ≤ 60)

-- Theorem statement
theorem exists_21_game_period (a : CumulativeGames) 
  (h : ValidSequence a) : 
  ∃ k n : Nat, k + n ≤ 42 ∧ a (k + n) - a k = 21 := by
  sorry

end NUMINAMATH_CALUDE_exists_21_game_period_l1302_130256


namespace NUMINAMATH_CALUDE_solve_equation_l1302_130292

theorem solve_equation (x : ℝ) : 4 * (x - 1) - 5 * (1 + x) = 3 ↔ x = -12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1302_130292


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1302_130231

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_a3 : a 3 = 1)
  (h_a5 : a 5 = 4) :
  ∃ q : ℝ, (q = 2 ∨ q = -2) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1302_130231


namespace NUMINAMATH_CALUDE_robin_gum_count_l1302_130251

theorem robin_gum_count (initial_gum : ℝ) (additional_gum : ℝ) (total_gum : ℝ) : 
  initial_gum = 18.0 → additional_gum = 44.0 → total_gum = initial_gum + additional_gum → total_gum = 62.0 :=
by sorry

end NUMINAMATH_CALUDE_robin_gum_count_l1302_130251


namespace NUMINAMATH_CALUDE_exponential_function_max_min_sum_l1302_130269

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_max_min_sum (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ max (f a 0) (f a 1)) ∧
  (∀ x ∈ Set.Icc 0 1, f a x ≥ min (f a 0) (f a 1)) ∧
  (max (f a 0) (f a 1) + min (f a 0) (f a 1) = 3) →
  a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_exponential_function_max_min_sum_l1302_130269


namespace NUMINAMATH_CALUDE_unique_sum_of_four_smallest_divisor_squares_l1302_130236

def is_sum_of_four_smallest_divisor_squares (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d ∧
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ d ∣ n ∧
    (∀ x : ℕ, x ∣ n → x ≤ d) ∧
    n = a^2 + b^2 + c^2 + d^2

theorem unique_sum_of_four_smallest_divisor_squares : 
  ∀ n : ℕ, is_sum_of_four_smallest_divisor_squares n ↔ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_unique_sum_of_four_smallest_divisor_squares_l1302_130236


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1302_130260

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.01

/-- The atomic weight of Bromine in g/mol -/
def bromine_weight : ℝ := 79.90

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of Bromine atoms in the compound -/
def bromine_count : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ :=
  hydrogen_count * hydrogen_weight +
  bromine_count * bromine_weight +
  oxygen_count * oxygen_weight

/-- Theorem stating that the molecular weight of the compound is 128.91 g/mol -/
theorem compound_molecular_weight : molecular_weight = 128.91 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1302_130260


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1302_130223

theorem geometric_sequence_product (b : ℕ → ℝ) (q : ℝ) :
  (∀ n, b (n + 1) = q * b n) →
  ∀ n, (b n * b (n + 1) * b (n + 2)) * q^3 = (b (n + 1) * b (n + 2) * b (n + 3)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1302_130223


namespace NUMINAMATH_CALUDE_geometric_series_product_sum_limit_l1302_130279

/-- The limit of the sum of the product of corresponding terms from two geometric series --/
theorem geometric_series_product_sum_limit (a r s : ℝ) 
  (hr : |r| < 1) (hs : |s| < 1) : 
  (∑' n, a^2 * (r*s)^n) = a^2 / (1 - r*s) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_product_sum_limit_l1302_130279


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1302_130211

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/3
  let S : ℝ := ∑' n, a * r^n
  S = 3/2 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1302_130211


namespace NUMINAMATH_CALUDE_meaningful_fraction_l1302_130266

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l1302_130266


namespace NUMINAMATH_CALUDE_range_of_a_l1302_130263

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a ≥ 0) ∧
  (∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0) →
  a ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1302_130263


namespace NUMINAMATH_CALUDE_new_person_weight_l1302_130253

/-- Given a group of 8 persons, if replacing one person weighing 65 kg with a new person
    increases the average weight by 3.5 kg, then the new person weighs 93 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 93 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1302_130253


namespace NUMINAMATH_CALUDE_athlete_arrangement_count_l1302_130284

/-- The number of tracks -/
def num_tracks : ℕ := 6

/-- The number of athletes -/
def num_athletes : ℕ := 6

/-- The number of tracks athlete A cannot stand on -/
def restricted_tracks_A : ℕ := 2

/-- The number of tracks athlete B can stand on -/
def allowed_tracks_B : ℕ := 2

/-- The number of different arrangements -/
def num_arrangements : ℕ := 144

theorem athlete_arrangement_count :
  (num_tracks = num_athletes) →
  (restricted_tracks_A = 2) →
  (allowed_tracks_B = 2) →
  (num_arrangements = 144) := by
  sorry

end NUMINAMATH_CALUDE_athlete_arrangement_count_l1302_130284


namespace NUMINAMATH_CALUDE_solution_set_is_two_l1302_130293

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := log10 (2 * x + 1) + log10 x = 1

-- Theorem statement
theorem solution_set_is_two :
  ∃! x : ℝ, x > 0 ∧ 2 * x + 1 > 0 ∧ equation x := by sorry

end NUMINAMATH_CALUDE_solution_set_is_two_l1302_130293


namespace NUMINAMATH_CALUDE_rectangular_hall_dimensions_l1302_130276

theorem rectangular_hall_dimensions (length width : ℝ) (area : ℝ) : 
  width = length / 2 →
  area = length * width →
  area = 288 →
  length - width = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimensions_l1302_130276


namespace NUMINAMATH_CALUDE_cos_sin_power_relation_l1302_130252

theorem cos_sin_power_relation (x a : Real) (h : Real.cos x ^ 6 + Real.sin x ^ 6 = a) :
  Real.cos x ^ 4 + Real.sin x ^ 4 = (1 + 2 * a) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_power_relation_l1302_130252


namespace NUMINAMATH_CALUDE_composition_equation_solution_l1302_130215

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 9 * x + 6
  ∃ x : ℝ, δ (φ x) = 3 ∧ x = -5/6 := by
sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l1302_130215


namespace NUMINAMATH_CALUDE_smallest_n_square_cube_l1302_130202

/-- A number is a perfect square if it's equal to some integer squared. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2

/-- A number is a perfect cube if it's equal to some integer cubed. -/
def IsPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 3

/-- The smallest positive integer n such that 3n is a perfect square and 2n is a perfect cube is 108. -/
theorem smallest_n_square_cube : (
  ∀ n : ℕ, 
  n > 0 ∧ 
  IsPerfectSquare (3 * n) ∧ 
  IsPerfectCube (2 * n) → 
  n ≥ 108
) ∧ 
IsPerfectSquare (3 * 108) ∧ 
IsPerfectCube (2 * 108) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_cube_l1302_130202


namespace NUMINAMATH_CALUDE_distance_to_grandmas_house_l1302_130232

-- Define the car's efficiency in miles per gallon
def car_efficiency : ℝ := 20

-- Define the amount of gas needed to reach Grandma's house in gallons
def gas_needed : ℝ := 5

-- Theorem to prove the distance to Grandma's house
theorem distance_to_grandmas_house : car_efficiency * gas_needed = 100 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_grandmas_house_l1302_130232


namespace NUMINAMATH_CALUDE_dillon_luca_sum_difference_l1302_130204

def dillon_list := List.range 40

def replace_three_with_two (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

def luca_list := dillon_list.map replace_three_with_two

theorem dillon_luca_sum_difference :
  (dillon_list.sum - luca_list.sum) = 104 := by
  sorry

end NUMINAMATH_CALUDE_dillon_luca_sum_difference_l1302_130204


namespace NUMINAMATH_CALUDE_stock_price_problem_l1302_130281

theorem stock_price_problem (price_less_expensive : ℝ) (price_more_expensive : ℝ) : 
  price_more_expensive = 2 * price_less_expensive →
  14 * price_more_expensive + 26 * price_less_expensive = 2106 →
  price_more_expensive = 78 := by
sorry

end NUMINAMATH_CALUDE_stock_price_problem_l1302_130281


namespace NUMINAMATH_CALUDE_no_solution_exists_l1302_130299

def matrix (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3*y, 4],
    ![2*y, y]]

theorem no_solution_exists (y : ℝ) (h : y + 1 = 0) :
  ¬ ∃ y, (3 * y^2 - 8 * y = 5 ∧ y + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1302_130299


namespace NUMINAMATH_CALUDE_triangle_properties_l1302_130264

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def ABC : Triangle := { A := (8, 5), B := (4, -2), C := (-6, 3) }

-- Equation of a line: ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def median_to_AC (t : Triangle) : Line := sorry

def altitude_to_AB (t : Triangle) : Line := sorry

def perpendicular_bisector_BC (t : Triangle) : Line := sorry

theorem triangle_properties :
  let m := median_to_AC ABC
  let h := altitude_to_AB ABC
  let p := perpendicular_bisector_BC ABC
  m.a = 2 ∧ m.b = 1 ∧ m.c = -6 ∧
  h.a = 4 ∧ h.b = 7 ∧ h.c = 3 ∧
  p.a = 2 ∧ p.b = -1 ∧ p.c = 5/2 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1302_130264


namespace NUMINAMATH_CALUDE_library_book_increase_l1302_130226

theorem library_book_increase (N : ℕ) : 
  N > 0 ∧ 
  (N * 1.004 * 1.008 : ℝ) < 50000 →
  ⌊(N * 1.004 * 1.008 - N * 1.004 : ℝ)⌋ = 251 :=
by sorry

end NUMINAMATH_CALUDE_library_book_increase_l1302_130226


namespace NUMINAMATH_CALUDE_empty_union_l1302_130209

theorem empty_union (A : Set α) : A ∪ ∅ = A := by sorry

end NUMINAMATH_CALUDE_empty_union_l1302_130209


namespace NUMINAMATH_CALUDE_smallest_cube_ending_368_l1302_130245

theorem smallest_cube_ending_368 : 
  ∃ (n : ℕ), n > 0 ∧ n^3 ≡ 368 [MOD 1000] ∧ ∀ (m : ℕ), m > 0 ∧ m^3 ≡ 368 [MOD 1000] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_368_l1302_130245


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1302_130238

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote 3x + y = 0 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b = 3 * a) : Real.sqrt 10 = 
  Real.sqrt ((a^2 + b^2) / a^2) := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1302_130238


namespace NUMINAMATH_CALUDE_at_least_four_same_acquaintances_l1302_130295

theorem at_least_four_same_acquaintances :
  ∀ (contestants : Finset Nat) (acquaintances : Nat → Finset Nat),
    contestants.card = 90 →
    (∀ x ∈ contestants, (acquaintances x).card ≥ 60) →
    (∀ x ∈ contestants, (acquaintances x) ⊆ contestants) →
    (∀ x ∈ contestants, x ∉ acquaintances x) →
    ∃ n : Nat, ∃ s : Finset Nat, s ⊆ contestants ∧ s.card ≥ 4 ∧
      ∀ x ∈ s, (acquaintances x).card = n :=
by
  sorry


end NUMINAMATH_CALUDE_at_least_four_same_acquaintances_l1302_130295


namespace NUMINAMATH_CALUDE_average_age_decrease_l1302_130286

/-- Proves that the average age of a class decreases by 4 years when new students join --/
theorem average_age_decrease (original_strength original_average new_students new_average : ℕ) :
  original_strength = 12 →
  original_average = 40 →
  new_students = 12 →
  new_average = 32 →
  let total_age_before := original_strength * original_average
  let total_age_new := new_students * new_average
  let total_age_after := total_age_before + total_age_new
  let new_strength := original_strength + new_students
  let new_average_age := total_age_after / new_strength
  original_average - new_average_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_age_decrease_l1302_130286


namespace NUMINAMATH_CALUDE_parking_lot_cars_l1302_130212

theorem parking_lot_cars (red_cars : ℕ) (black_cars : ℕ) : 
  red_cars = 33 → 
  red_cars * 8 = black_cars * 3 → 
  black_cars = 88 := by
sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l1302_130212


namespace NUMINAMATH_CALUDE_volunteer_allocation_schemes_l1302_130200

def num_volunteers : ℕ := 5
def num_projects : ℕ := 4

theorem volunteer_allocation_schemes :
  (num_volunteers.choose 2) * (num_projects!) = 240 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_allocation_schemes_l1302_130200


namespace NUMINAMATH_CALUDE_shirt_sales_theorem_l1302_130291

/-- Represents the sales and profit data for a shirt selling business -/
structure ShirtSales where
  initial_sales : ℕ
  initial_profit : ℝ
  sales_increase : ℝ
  profit_decrease : ℝ

/-- Calculates the new sales quantity after a price reduction -/
def new_sales (data : ShirtSales) (reduction : ℝ) : ℝ :=
  data.initial_sales + data.sales_increase * reduction

/-- Calculates the new profit per piece after a price reduction -/
def new_profit_per_piece (data : ShirtSales) (reduction : ℝ) : ℝ :=
  data.initial_profit - reduction

/-- Calculates the total daily profit after a price reduction -/
def total_daily_profit (data : ShirtSales) (reduction : ℝ) : ℝ :=
  new_sales data reduction * new_profit_per_piece data reduction

/-- The main theorem about shirt sales and profit -/
theorem shirt_sales_theorem (data : ShirtSales) 
    (h1 : data.initial_sales = 20)
    (h2 : data.initial_profit = 40)
    (h3 : data.sales_increase = 2)
    (h4 : data.profit_decrease = 1) : 
    new_sales data 3 = 26 ∧ 
    ∃ x : ℝ, x = 20 ∧ total_daily_profit data x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_shirt_sales_theorem_l1302_130291


namespace NUMINAMATH_CALUDE_smallest_positive_solution_sqrt_3x_eq_5x_l1302_130237

theorem smallest_positive_solution_sqrt_3x_eq_5x :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x ∧
  ∀ (y : ℝ), y > 0 → Real.sqrt (3 * y) = 5 * y → x ≤ y ∧
  x = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_sqrt_3x_eq_5x_l1302_130237


namespace NUMINAMATH_CALUDE_ambiguous_dates_count_l1302_130289

/-- The number of months in a year -/
def num_months : ℕ := 12

/-- The maximum day number that can be confused as a month -/
def max_ambiguous_day : ℕ := 12

/-- The number of ambiguous dates in a year -/
def num_ambiguous_dates : ℕ := num_months * max_ambiguous_day - num_months

theorem ambiguous_dates_count :
  num_ambiguous_dates = 132 :=
sorry

end NUMINAMATH_CALUDE_ambiguous_dates_count_l1302_130289


namespace NUMINAMATH_CALUDE_diameter_height_ratio_l1302_130203

/-- A cylinder whose lateral surface unfolds into a square -/
structure SquareUnfoldCylinder where
  diameter : ℝ
  height : ℝ
  square_unfold : height = π * diameter

theorem diameter_height_ratio (c : SquareUnfoldCylinder) :
  c.diameter / c.height = 1 / π := by
  sorry

end NUMINAMATH_CALUDE_diameter_height_ratio_l1302_130203


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1302_130285

/-- The minimum distance from the origin to the line y = x + 2 is √2 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | p.2 = p.1 + 2}
  ∀ p ∈ line, Real.sqrt 2 ≤ Real.sqrt (p.1^2 + p.2^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1302_130285


namespace NUMINAMATH_CALUDE_largest_prime_factor_l1302_130248

theorem largest_prime_factor : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (16^4 + 2 * 16^2 + 1 - 13^4) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (16^4 + 2 * 16^2 + 1 - 13^4) → q ≤ p) ∧
  Nat.Prime 71 ∧ 
  71 ∣ (16^4 + 2 * 16^2 + 1 - 13^4) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l1302_130248


namespace NUMINAMATH_CALUDE_unique_solution_implies_negative_a_l1302_130243

theorem unique_solution_implies_negative_a (a : ℝ) :
  (∃! x : ℝ, |x^2 - 1| = a * |x - 1|) → a < 0 := by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_negative_a_l1302_130243


namespace NUMINAMATH_CALUDE_parking_lot_problem_l1302_130218

theorem parking_lot_problem :
  let total_cars : ℝ := 300
  let valid_ticket_ratio : ℝ := 0.75
  let permanent_pass_ratio : ℝ := 0.2
  let unpaid_cars : ℝ := 30
  valid_ticket_ratio * total_cars +
  permanent_pass_ratio * (valid_ticket_ratio * total_cars) +
  unpaid_cars = total_cars :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l1302_130218


namespace NUMINAMATH_CALUDE_sam_washing_pennies_l1302_130287

/-- The number of pennies Sam earned from washing clothes -/
def pennies_from_washing (total_cents : ℕ) (num_quarters : ℕ) : ℕ :=
  total_cents - (num_quarters * 25)

/-- Theorem: Given 7 quarters and a total of $1.84, Sam earned 9 pennies from washing clothes -/
theorem sam_washing_pennies :
  pennies_from_washing 184 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sam_washing_pennies_l1302_130287


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l1302_130250

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l1302_130250


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1302_130216

/-- The range of m for which the quadratic inequality (m-3)x^2 - 2mx - 8 > 0
    has a solution set that is an open interval with length between 1 and 2 -/
theorem quadratic_inequality_range (m : ℝ) : 
  (∃ a b : ℝ, 
    (∀ x : ℝ, (m - 3) * x^2 - 2 * m * x - 8 > 0 ↔ a < x ∧ x < b) ∧ 
    1 ≤ b - a ∧ b - a ≤ 2) ↔ 
  m ≤ -15 ∨ (7/3 ≤ m ∧ m ≤ 33/14) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1302_130216


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l1302_130234

theorem unique_solution_cube_equation (y : ℝ) (hy : y ≠ 0) :
  (3 * y)^5 = (9 * y)^4 ↔ y = 27 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l1302_130234


namespace NUMINAMATH_CALUDE_regular_pentagon_perimeter_l1302_130249

/-- The perimeter of a regular pentagon with side length 15 cm is 75 cm. -/
theorem regular_pentagon_perimeter :
  ∀ (side_length perimeter : ℝ),
  side_length = 15 →
  perimeter = 5 * side_length →
  perimeter = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_perimeter_l1302_130249


namespace NUMINAMATH_CALUDE_complex_fraction_eval_l1302_130219

theorem complex_fraction_eval (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 + c*d + d^2 = 0) : 
  (c^12 + d^12) / (c^3 + d^3)^4 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_eval_l1302_130219


namespace NUMINAMATH_CALUDE_pop_survey_result_l1302_130241

theorem pop_survey_result (total_surveyed : ℕ) (pop_angle : ℕ) (people_chose_pop : ℕ) : 
  total_surveyed = 540 → pop_angle = 270 → people_chose_pop = total_surveyed * pop_angle / 360 →
  people_chose_pop = 405 := by
sorry

end NUMINAMATH_CALUDE_pop_survey_result_l1302_130241


namespace NUMINAMATH_CALUDE_correct_vs_incorrect_calculation_l1302_130296

theorem correct_vs_incorrect_calculation : 
  (12 - (3 * 4)) - ((12 - 3) * 4) = -36 := by sorry

end NUMINAMATH_CALUDE_correct_vs_incorrect_calculation_l1302_130296


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l1302_130208

theorem sphere_radius_ratio (V₁ V₂ : ℝ) (r₁ r₂ : ℝ) 
  (h₁ : V₁ = (4 / 3) * π * r₁^3)
  (h₂ : V₂ = (4 / 3) * π * r₂^3)
  (h₃ : V₁ = 432 * π)
  (h₄ : V₂ = 0.25 * V₁) :
  r₂ / r₁ = 1 / Real.rpow 3 (1/3) := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l1302_130208


namespace NUMINAMATH_CALUDE_remainder_of_7n_mod_4_l1302_130259

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_7n_mod_4_l1302_130259


namespace NUMINAMATH_CALUDE_quadratic_properties_l1302_130297

variable (a b c p q : ℝ)
variable (f : ℝ → ℝ)

-- Define the quadratic function
def is_quadratic (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_properties (h : is_quadratic f a b c) (hpq : p ≠ q) :
  (f p = f q → f (p + q) = c) ∧
  (f (p + q) = c → p + q = 0 ∨ f p = f q) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1302_130297


namespace NUMINAMATH_CALUDE_oak_trees_planted_l1302_130246

/-- The number of oak trees planted by workers in a park. -/
def trees_planted (initial_trees final_trees : ℕ) : ℕ :=
  final_trees - initial_trees

/-- Theorem: Given 5 initial oak trees and 9 final oak trees, the number of trees planted is 4. -/
theorem oak_trees_planted :
  let initial_trees : ℕ := 5
  let final_trees : ℕ := 9
  trees_planted initial_trees final_trees = 4 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_planted_l1302_130246


namespace NUMINAMATH_CALUDE_exactly_two_valid_A_values_l1302_130239

/-- A function that checks if a number is divisible by 8 based on its last three digits -/
def isDivisibleBy8 (n : ℕ) : Prop :=
  n % 8 = 0

/-- A function that constructs the number 451,2A8 given A -/
def constructNumber (A : ℕ) : ℕ :=
  451200 + A * 10 + 8

/-- The main theorem stating that there are exactly 2 single-digit values of A satisfying both conditions -/
theorem exactly_two_valid_A_values :
  ∃! (S : Finset ℕ), S.card = 2 ∧ 
    (∀ A ∈ S, A < 10 ∧ 120 % A = 0 ∧ isDivisibleBy8 (constructNumber A)) ∧
    (∀ A < 10, 120 % A = 0 ∧ isDivisibleBy8 (constructNumber A) → A ∈ S) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_A_values_l1302_130239


namespace NUMINAMATH_CALUDE_polynomial_division_l1302_130272

theorem polynomial_division (x : ℝ) (h : x ≠ 0) :
  (6 * x^4 - 8 * x^3) / (-2 * x^2) = -3 * x^2 + 4 * x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l1302_130272


namespace NUMINAMATH_CALUDE_fibonacci_rabbit_problem_l1302_130262

/-- Fibonacci sequence representing the number of adult rabbit pairs -/
def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

/-- The number of adult rabbit pairs after n months -/
def adult_rabbits (n : ℕ) : ℕ := fibonacci n

theorem fibonacci_rabbit_problem :
  adult_rabbits 12 = 233 := by sorry

end NUMINAMATH_CALUDE_fibonacci_rabbit_problem_l1302_130262


namespace NUMINAMATH_CALUDE_largest_number_l1302_130274

theorem largest_number (a b c d e : ℝ) : 
  a = 24680 + 1/1357 →
  b = 24680 - 1/1357 →
  c = 24680 * (1/1357) →
  d = 24680 / (1/1357) →
  e = 24680.1357 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1302_130274


namespace NUMINAMATH_CALUDE_tangent_point_segment_difference_l1302_130229

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadrilateral where
  /-- The lengths of the four sides of the quadrilateral -/
  sides : Fin 4 → ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The semiperimeter of the quadrilateral -/
  semiperimeter : ℝ
  /-- The area of the quadrilateral -/
  area : ℝ

/-- The theorem about the difference of segments created by the point of tangency -/
theorem tangent_point_segment_difference
  (Q : CyclicQuadrilateral)
  (h1 : Q.sides 0 = 80)
  (h2 : Q.sides 1 = 100)
  (h3 : Q.sides 2 = 140)
  (h4 : Q.sides 3 = 120)
  (h5 : Q.semiperimeter = (Q.sides 0 + Q.sides 1 + Q.sides 2 + Q.sides 3) / 2)
  (h6 : Q.area = Real.sqrt ((Q.semiperimeter - Q.sides 0) *
                            (Q.semiperimeter - Q.sides 1) *
                            (Q.semiperimeter - Q.sides 2) *
                            (Q.semiperimeter - Q.sides 3)))
  (h7 : Q.inradius * Q.semiperimeter = Q.area) :
  ∃ (x y : ℝ), x + y = 140 ∧ |x - y| = 5 := by
  sorry


end NUMINAMATH_CALUDE_tangent_point_segment_difference_l1302_130229


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_l1302_130255

theorem arithmetic_sequence_sum_mod (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 3 →
  d = 5 →
  aₙ = 103 →
  0 ≤ n →
  n < 17 →
  (n : ℤ) ≡ (n * (a₁ + aₙ) / 2) [ZMOD 17] →
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_l1302_130255


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1302_130228

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  1 / a + 2 / b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2 * b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1302_130228


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l1302_130222

theorem square_difference_of_integers (x y : ℕ) 
  (h1 : x + y = 40) 
  (h2 : x - y = 14) : 
  x^2 - y^2 = 560 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l1302_130222


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l1302_130242

theorem quadratic_coefficient_sum (a k n : ℤ) : 
  (∀ x : ℤ, (3*x + 2)*(2*x - 7) = a*x^2 + k*x + n) → 
  a - n + k = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l1302_130242


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1302_130201

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 48 * y + 36 = (4 * y - 6)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1302_130201


namespace NUMINAMATH_CALUDE_kylie_daisies_l1302_130275

def initial_daisies : ℕ := 5
def sister_daisies : ℕ := 9

def total_daisies : ℕ := initial_daisies + sister_daisies

def remaining_daisies : ℕ := total_daisies / 2

theorem kylie_daisies : remaining_daisies = 7 := by sorry

end NUMINAMATH_CALUDE_kylie_daisies_l1302_130275


namespace NUMINAMATH_CALUDE_triangle_midline_lengths_l1302_130261

/-- Given a triangle with side lengths a, b, and c, the lengths of its midlines are half the lengths of the opposite sides. -/
theorem triangle_midline_lengths (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (s_a s_b s_c : ℝ),
    s_a = (1/2) * b ∧
    s_b = (1/2) * c ∧
    s_c = (1/2) * a :=
by sorry

end NUMINAMATH_CALUDE_triangle_midline_lengths_l1302_130261
