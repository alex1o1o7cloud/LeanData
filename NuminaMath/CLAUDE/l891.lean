import Mathlib

namespace NUMINAMATH_CALUDE_probability_sum_13_l891_89108

def die1 : Finset ℕ := {1, 2, 3, 7, 8, 9}
def die2 : Finset ℕ := {4, 5, 6, 10, 11, 12}

def sumTo13 : Finset (ℕ × ℕ) :=
  (die1.product die2).filter (fun p => p.1 + p.2 = 13)

theorem probability_sum_13 :
  (sumTo13.card : ℚ) / ((die1.card * die2.card) : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_13_l891_89108


namespace NUMINAMATH_CALUDE_min_sum_given_max_product_l891_89114

theorem min_sum_given_max_product (a b : ℝ) : 
  a > 0 → b > 0 → (∀ x y : ℝ, a * b * x + y ≤ 8) → a + b ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_max_product_l891_89114


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l891_89133

theorem concentric_circles_ratio (a b : ℝ) (h : a > 0) (k : b > 0) :
  (π * b^2 - π * a^2 = 4 * π * a^2) → (a / b = 1 / Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l891_89133


namespace NUMINAMATH_CALUDE_inscribed_circle_square_area_l891_89194

/-- A circle inscribed in a square with sides parallel to the axes -/
structure InscribedCircle where
  /-- The equation of the circle: x^2 + y^2 + 2x - 8y = 0 -/
  eq : ∀ (x y : ℝ), x^2 + y^2 + 2*x - 8*y = 0

/-- The area of the square that inscribes the circle -/
def squareArea (c : InscribedCircle) : ℝ := 68

/-- Theorem: The area of the square that inscribes the circle is 68 square units -/
theorem inscribed_circle_square_area (c : InscribedCircle) : squareArea c = 68 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_area_l891_89194


namespace NUMINAMATH_CALUDE_optimal_well_placement_l891_89164

/-- Three houses positioned along a straight road -/
structure Village where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The distance between adjacent houses is 50 meters -/
def house_distance : ℝ := 50

/-- A village with houses positioned at the correct intervals -/
def village : Village :=
  { A := 0,
    B := house_distance,
    C := 2 * house_distance }

/-- The sum of distances from a point to all houses -/
def total_distance (x : ℝ) : ℝ :=
  |x - village.A| + |x - village.B| + |x - village.C|

/-- The well position that minimizes the total distance -/
def optimal_well_position : ℝ := village.B

theorem optimal_well_placement :
  ∀ x : ℝ, total_distance optimal_well_position ≤ total_distance x :=
sorry

end NUMINAMATH_CALUDE_optimal_well_placement_l891_89164


namespace NUMINAMATH_CALUDE_expected_balls_original_positions_l891_89181

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 7

/-- The probability that a ball is in its original position after Chris and Silva's actions -/
def prob_original_position : ℚ := 25 / 49

/-- The expected number of balls in their original positions -/
def expected_original_positions : ℚ := num_balls * prob_original_position

/-- Theorem stating the expected number of balls in their original positions -/
theorem expected_balls_original_positions :
  expected_original_positions = 175 / 49 := by sorry

end NUMINAMATH_CALUDE_expected_balls_original_positions_l891_89181


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l891_89122

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_third : a 3 = 30)
  (h_ninth : a 9 = 60) :
  a 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l891_89122


namespace NUMINAMATH_CALUDE_polynomial_factorization_isosceles_triangle_l891_89131

-- Part 1: Polynomial factorization
theorem polynomial_factorization (x y : ℝ) :
  x^2 - 2*x*y + y^2 - 16 = (x - y + 4) * (x - y - 4) := by sorry

-- Part 2: Triangle shape determination
def is_triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem isosceles_triangle (a b c : ℝ) (h : is_triangle a b c) :
  a^2 - a*b + a*c - b*c = 0 → a = b := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_isosceles_triangle_l891_89131


namespace NUMINAMATH_CALUDE_angles_do_not_determine_triangle_uniquely_l891_89104

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  sum_angles : α + β + γ = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define a function to check if two triangles have the same angles
def SameAngles (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem statement
theorem angles_do_not_determine_triangle_uniquely :
  ∃ (t1 t2 : Triangle), SameAngles t1 t2 ∧ t1 ≠ t2 := by sorry

end NUMINAMATH_CALUDE_angles_do_not_determine_triangle_uniquely_l891_89104


namespace NUMINAMATH_CALUDE_malt_shop_shakes_l891_89180

/-- Given a malt shop scenario where:
  * Each shake uses 4 ounces of chocolate syrup
  * Each cone uses 6 ounces of chocolate syrup
  * 1 cone was sold
  * A total of 14 ounces of chocolate syrup was used
  Prove that 2 shakes were sold. -/
theorem malt_shop_shakes : 
  ∀ (shakes : ℕ), 
    (4 * shakes + 6 * 1 = 14) → shakes = 2 := by
  sorry

end NUMINAMATH_CALUDE_malt_shop_shakes_l891_89180


namespace NUMINAMATH_CALUDE_non_dividing_diagonals_count_l891_89117

/-- The number of sides in the regular polygon -/
def n : ℕ := 150

/-- The total number of diagonals in a polygon with n sides -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals that divide the polygon into two equal parts -/
def equal_dividing_diagonals (n : ℕ) : ℕ := n / 2

/-- The number of diagonals that do not divide the polygon into two equal parts -/
def non_dividing_diagonals (n : ℕ) : ℕ := total_diagonals n - equal_dividing_diagonals n

theorem non_dividing_diagonals_count :
  non_dividing_diagonals n = 10950 :=
by sorry

end NUMINAMATH_CALUDE_non_dividing_diagonals_count_l891_89117


namespace NUMINAMATH_CALUDE_volleyball_tournament_teams_l891_89119

/-- Represents a volleyball tournament -/
structure VolleyballTournament where
  teams : ℕ
  no_win_fraction : ℚ
  single_round : Bool

/-- Theorem: In a single round volleyball tournament where 20% of teams did not win a single game,
    the total number of teams must be 5. -/
theorem volleyball_tournament_teams
  (t : VolleyballTournament)
  (h1 : t.no_win_fraction = 1/5)
  (h2 : t.single_round = true)
  : t.teams = 5 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_teams_l891_89119


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l891_89161

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l891_89161


namespace NUMINAMATH_CALUDE_stratified_sampling_equal_probability_l891_89197

/-- Represents the number of individuals in each stratum -/
structure StrataSize where
  general : ℕ
  deputy : ℕ
  logistics : ℕ

/-- Represents the sample size for each stratum -/
structure StrataSample where
  general : ℕ
  deputy : ℕ
  logistics : ℕ

/-- The total population size -/
def totalPopulation (s : StrataSize) : ℕ :=
  s.general + s.deputy + s.logistics

/-- The total sample size -/
def totalSample (s : StrataSample) : ℕ :=
  s.general + s.deputy + s.logistics

/-- The probability of selection for an individual in a given stratum -/
def selectionProbability (popSize : ℕ) (sampleSize : ℕ) : ℚ :=
  sampleSize / popSize

theorem stratified_sampling_equal_probability 
  (strata : StrataSize) 
  (sample : StrataSample) 
  (h1 : totalPopulation strata = 160)
  (h2 : strata.general = 112)
  (h3 : strata.deputy = 16)
  (h4 : strata.logistics = 32)
  (h5 : totalSample sample = 20) :
  ∃ (p : ℚ), 
    selectionProbability strata.general sample.general = p ∧
    selectionProbability strata.deputy sample.deputy = p ∧
    selectionProbability strata.logistics sample.logistics = p :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_equal_probability_l891_89197


namespace NUMINAMATH_CALUDE_square_inscribed_in_circle_l891_89144

theorem square_inscribed_in_circle (r : ℝ) (S : ℝ) :
  r > 0 →
  r^2 * π = 16 * π →
  S = (2 * r)^2 / 2 →
  S = 32 :=
by sorry

end NUMINAMATH_CALUDE_square_inscribed_in_circle_l891_89144


namespace NUMINAMATH_CALUDE_zach_needs_six_dollars_l891_89129

/-- The amount of money Zach needs to earn to buy the bike -/
def money_needed (bike_cost allowance lawn_money babysit_rate babysit_hours savings : ℕ) : ℕ :=
  let total_earned := allowance + lawn_money + babysit_rate * babysit_hours
  let total_available := savings + total_earned
  if total_available ≥ bike_cost then 0
  else bike_cost - total_available

/-- Theorem stating how much more money Zach needs to earn -/
theorem zach_needs_six_dollars :
  money_needed 100 5 10 7 2 65 = 6 := by
  sorry

end NUMINAMATH_CALUDE_zach_needs_six_dollars_l891_89129


namespace NUMINAMATH_CALUDE_sally_pens_theorem_l891_89130

/-- The number of pens Sally takes home -/
def pens_taken_home (initial_pens : ℕ) (num_students : ℕ) (pens_per_student : ℕ) : ℕ :=
  let pens_given := num_students * pens_per_student
  let pens_remaining := initial_pens - pens_given
  pens_remaining / 2

theorem sally_pens_theorem :
  pens_taken_home 342 44 7 = 17 := by
  sorry

#eval pens_taken_home 342 44 7

end NUMINAMATH_CALUDE_sally_pens_theorem_l891_89130


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l891_89154

/-- Given a principal amount and an interest rate, proves that if the simple interest
    for 2 years is $600 and the compound interest for 2 years is $609,
    then the interest rate is 3% per annum. -/
theorem interest_rate_calculation (P r : ℝ) : 
  P * r * 2 = 600 →
  P * ((1 + r)^2 - 1) = 609 →
  r = 0.03 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l891_89154


namespace NUMINAMATH_CALUDE_gilbert_parsley_count_l891_89106

/-- Represents the number of herb plants Gilbert had at different stages of spring. -/
structure HerbCount where
  initial_basil : ℕ
  initial_parsley : ℕ
  initial_mint : ℕ
  final_basil : ℕ
  final_total : ℕ

/-- The conditions of Gilbert's herb garden during spring. -/
def spring_garden_conditions : HerbCount where
  initial_basil := 3
  initial_parsley := 0  -- We'll prove this is 1
  initial_mint := 2
  final_basil := 4
  final_total := 5

/-- Theorem stating that Gilbert planted 1 parsley plant initially. -/
theorem gilbert_parsley_count :
  spring_garden_conditions.initial_parsley = 1 :=
by sorry

end NUMINAMATH_CALUDE_gilbert_parsley_count_l891_89106


namespace NUMINAMATH_CALUDE_bottle_caps_problem_l891_89120

/-- The number of bottle caps left in a jar after removing some. -/
def bottle_caps_left (original : ℕ) (removed : ℕ) : ℕ :=
  original - removed

/-- Theorem stating that 40 bottle caps are left when 47 are removed from 87. -/
theorem bottle_caps_problem :
  bottle_caps_left 87 47 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_problem_l891_89120


namespace NUMINAMATH_CALUDE_square_side_length_l891_89118

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * Real.sqrt 2 = diagonal ∧ side = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l891_89118


namespace NUMINAMATH_CALUDE_lowest_dropped_score_l891_89115

theorem lowest_dropped_score (scores : Fin 4 → ℕ) 
  (avg_all : (scores 0 + scores 1 + scores 2 + scores 3) / 4 = 45)
  (avg_after_drop : ∃ i, (scores 0 + scores 1 + scores 2 + scores 3 - scores i) / 3 = 50) :
  ∃ i, scores i = 30 ∧ ∀ j, scores j ≥ scores i := by
  sorry

end NUMINAMATH_CALUDE_lowest_dropped_score_l891_89115


namespace NUMINAMATH_CALUDE_parametric_to_general_plane_equation_l891_89151

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
structure ParametricPlane where
  origin : Point3D
  dir1 : Point3D
  dir2 : Point3D

/-- Represents the equation of a plane in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a plane equation satisfies the required conditions -/
def isValidPlaneEquation (eq : PlaneEquation) : Prop :=
  eq.A > 0 ∧ Int.gcd (Int.natAbs eq.A) (Int.gcd (Int.natAbs eq.B) (Int.gcd (Int.natAbs eq.C) (Int.natAbs eq.D))) = 1

/-- The main theorem stating the equivalence of the parametric and general forms of the plane -/
theorem parametric_to_general_plane_equation 
  (plane : ParametricPlane)
  (h_plane : plane = { 
    origin := { x := 3, y := 4, z := 1 },
    dir1 := { x := 1, y := -2, z := -1 },
    dir2 := { x := 2, y := 0, z := 1 }
  }) :
  ∃ (eq : PlaneEquation), 
    isValidPlaneEquation eq ∧
    (∀ (p : Point3D), 
      (∃ (s t : ℝ), 
        p.x = plane.origin.x + s * plane.dir1.x + t * plane.dir2.x ∧
        p.y = plane.origin.y + s * plane.dir1.y + t * plane.dir2.y ∧
        p.z = plane.origin.z + s * plane.dir1.z + t * plane.dir2.z) ↔
      eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0) ∧
    eq = { A := 2, B := 3, C := -4, D := -14 } :=
  sorry

end NUMINAMATH_CALUDE_parametric_to_general_plane_equation_l891_89151


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l891_89173

def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 3]

theorem perpendicular_vectors (m : ℝ) :
  (∀ i : Fin 2, (a i) * (a i + m * (b i)) = 0) →
  m = 5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l891_89173


namespace NUMINAMATH_CALUDE_unique_solution_l891_89103

theorem unique_solution (x y z : ℝ) 
  (hx : x > 3) (hy : y > 3) (hz : z > 3)
  (h : (x + 4)^2 / (y + z - 4) + (y + 6)^2 / (z + x - 6) + (z + 8)^2 / (x + y - 8) = 48) :
  x = 11 ∧ y = 10 ∧ z = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l891_89103


namespace NUMINAMATH_CALUDE_factorial_squared_greater_than_power_l891_89162

theorem factorial_squared_greater_than_power (n : ℕ) (h : n > 2) : (n.factorial ^ 2 : ℝ) > n ^ n := by
  sorry

end NUMINAMATH_CALUDE_factorial_squared_greater_than_power_l891_89162


namespace NUMINAMATH_CALUDE_min_difference_theorem_l891_89107

noncomputable section

def f (x : ℝ) : ℝ := Real.exp (2 * x)
def g (x : ℝ) : ℝ := Real.log x + 1/2

theorem min_difference_theorem :
  ∃ (h : ℝ → ℝ), ∀ (x₁ : ℝ),
    (∃ (x₂ : ℝ), x₂ > 0 ∧ f x₁ = g x₂) ∧
    (∀ (x₂ : ℝ), x₂ > 0 → f x₁ = g x₂ → h x₁ ≤ x₂ - x₁) ∧
    (∃ (x₁ x₂ : ℝ), x₂ > 0 ∧ f x₁ = g x₂ ∧ h x₁ = x₂ - x₁) ∧
    (∀ (x : ℝ), h x = 1 + Real.log 2 / 2) :=
sorry

end

end NUMINAMATH_CALUDE_min_difference_theorem_l891_89107


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l891_89182

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := n.choose k

/-- The number of triangles with one side being a side of the decagon -/
def one_side_triangles : ℕ := n * (n - 4)

/-- The number of triangles with two sides being sides of the decagon -/
def two_side_triangles : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side being a side of the decagon) -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of a triangle having at least one side that is also a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l891_89182


namespace NUMINAMATH_CALUDE_distance_and_closest_point_theorem_l891_89134

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by a point and a direction vector -/
structure Line3D where
  point : Point3D
  direction : Vector3D

def distance_point_to_line (p : Point3D) (l : Line3D) : ℝ :=
  sorry

def closest_point_on_line (p : Point3D) (l : Line3D) : Point3D :=
  sorry

theorem distance_and_closest_point_theorem :
  let p := Point3D.mk 3 4 5
  let l := Line3D.mk (Point3D.mk 2 3 1) (Vector3D.mk 1 (-1) 2)
  distance_point_to_line p l = Real.sqrt 6 / 3 ∧
  closest_point_on_line p l = Point3D.mk (10/3) (5/3) (11/3) := by
  sorry

end NUMINAMATH_CALUDE_distance_and_closest_point_theorem_l891_89134


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l891_89160

/-- A quadratic function f(x) = ax^2 + (b-2)x + 3 where a ≠ 0 -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

theorem quadratic_function_properties (a b : ℝ) (ha : a ≠ 0) :
  /- If the solution set of f(x) > 0 is (-1, 3), then a = -1 and b = 4 -/
  (∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (a = -1 ∧ b = 4) ∧
  /- If f(1) = 2, a > 0, and b > 0, then the minimum value of 1/a + 4/b is 9 -/
  (f a b 1 = 2 ∧ a > 0 ∧ b > 0 →
   ∀ a' b', a' > 0 → b' > 0 → f a' b' 1 = 2 → 1/a' + 4/b' ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l891_89160


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l891_89163

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The value of k for which the lines y = 5x + 3 and y = (3k)x + 7 are parallel -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 7) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l891_89163


namespace NUMINAMATH_CALUDE_right_angled_triangle_l891_89170

theorem right_angled_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a * Real.cos B + b * Real.cos A = c * Real.sin A →
  a * Real.sin C = b * Real.sin B →
  b * Real.sin C = c * Real.sin A →
  c * Real.sin B = a * Real.sin C →
  A = π / 2 := by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l891_89170


namespace NUMINAMATH_CALUDE_currency_multiplication_invalid_l891_89116

-- Define currency types
inductive Currency
| Ruble
| Kopeck

-- Define a structure for money
structure Money where
  amount : ℚ
  currency : Currency

-- Define conversion rate
def conversionRate : ℚ := 100

-- Define equality for Money
def Money.eq (a b : Money) : Prop :=
  (a.currency = b.currency ∧ a.amount = b.amount) ∨
  (a.currency = Currency.Ruble ∧ b.currency = Currency.Kopeck ∧ a.amount * conversionRate = b.amount) ∨
  (a.currency = Currency.Kopeck ∧ b.currency = Currency.Ruble ∧ a.amount = b.amount * conversionRate)

-- Define multiplication for Money (this operation is not well-defined for real currencies)
def Money.mul (a b : Money) : Money :=
  { amount := a.amount * b.amount,
    currency := 
      match a.currency, b.currency with
      | Currency.Ruble, Currency.Ruble => Currency.Ruble
      | Currency.Kopeck, Currency.Kopeck => Currency.Kopeck
      | _, _ => Currency.Ruble }

-- Theorem statement
theorem currency_multiplication_invalid :
  ∃ (a b c d : Money),
    Money.eq a b ∧ Money.eq c d ∧
    ¬(Money.eq (Money.mul a c) (Money.mul b d)) := by
  sorry

end NUMINAMATH_CALUDE_currency_multiplication_invalid_l891_89116


namespace NUMINAMATH_CALUDE_trigonometric_identities_l891_89177

theorem trigonometric_identities (θ : Real) 
  (h : (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6 / 11) : 
  (Real.tan θ = 2) ∧ 
  ((5 * (Real.cos θ)^2) / (Real.sin (2*θ) + 2 * Real.sin θ * Real.cos θ - 3 * (Real.cos θ)^2) = 1) ∧ 
  (1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1/5) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l891_89177


namespace NUMINAMATH_CALUDE_candy_packing_problem_l891_89150

theorem candy_packing_problem :
  ∃! (s : Finset ℕ),
    (∀ a ∈ s, 200 ≤ a ∧ a ≤ 250) ∧
    (∀ a ∈ s, a % 10 = 6) ∧
    (∀ a ∈ s, a % 15 = 11) ∧
    s.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_candy_packing_problem_l891_89150


namespace NUMINAMATH_CALUDE_original_advertisers_from_university_a_l891_89127

/-- Represents the fraction of advertisers from University A -/
def fractionFromUniversityA : ℚ := 3/4

/-- Represents the total number of original network advertisers -/
def totalOriginalAdvertisers : ℕ := 20

/-- Represents the percentage of computer advertisers from University A -/
def percentageFromUniversityA : ℚ := 75/100

theorem original_advertisers_from_university_a :
  (↑⌊(percentageFromUniversityA * totalOriginalAdvertisers)⌋ : ℚ) / totalOriginalAdvertisers = fractionFromUniversityA :=
sorry

end NUMINAMATH_CALUDE_original_advertisers_from_university_a_l891_89127


namespace NUMINAMATH_CALUDE_expression_equivalence_l891_89155

theorem expression_equivalence (x y z : ℝ) :
  let P := x + y
  let Q := x - y
  ((P + Q + z) / (P - Q - z) - (P - Q - z) / (P + Q + z)) = 
    (4 * (x^2 + y^2 + x*z)) / ((2*y - z) * (2*x + z)) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l891_89155


namespace NUMINAMATH_CALUDE_geometric_sequence_and_curve_max_l891_89102

/-- Given real numbers a, b, c, and d forming a geometric sequence, 
    if the curve y = 3x - x^3 has a local maximum at x = b with the value c, 
    then ad = 2 -/
theorem geometric_sequence_and_curve_max (a b c d : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, 3 * b - b^3 ≥ 3 * x - x^3) →                 -- local maximum condition
  (3 * b - b^3 = c) →                                    -- value at local maximum
  a * d = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_curve_max_l891_89102


namespace NUMINAMATH_CALUDE_least_value_of_x_l891_89175

theorem least_value_of_x (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) 
  (h3 : ∃ (k : ℕ), k > 0 ∧ x = 11 * p * k ∧ Nat.Prime k ∧ Even k) : 
  x ≥ 44 ∧ ∃ (x₀ : ℕ), x₀ ≥ 44 ∧ 
    (∃ (p₀ : ℕ), Nat.Prime p₀ ∧ ∃ (k₀ : ℕ), k₀ > 0 ∧ x₀ = 11 * p₀ * k₀ ∧ Nat.Prime k₀ ∧ Even k₀) :=
by sorry

end NUMINAMATH_CALUDE_least_value_of_x_l891_89175


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l891_89179

theorem ceiling_floor_sum_zero : 
  Int.ceil (7 / 3 : ℚ) + Int.floor (-(7 / 3) : ℚ) + 
  Int.ceil (4 / 5 : ℚ) + Int.floor (-(4 / 5) : ℚ) = 0 := by
sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l891_89179


namespace NUMINAMATH_CALUDE_f_is_even_l891_89191

def f (x : ℝ) : ℝ := (x + 2)^2 + (2*x - 1)^2

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l891_89191


namespace NUMINAMATH_CALUDE_correct_ball_arrangements_l891_89126

/-- The number of ways to arrange 9 balls with 2 red, 3 yellow, and 4 white balls -/
def ballArrangements : ℕ := 2520

/-- The total number of balls -/
def totalBalls : ℕ := 9

/-- The number of red balls -/
def redBalls : ℕ := 2

/-- The number of yellow balls -/
def yellowBalls : ℕ := 3

/-- The number of white balls -/
def whiteBalls : ℕ := 4

theorem correct_ball_arrangements :
  ballArrangements = Nat.factorial totalBalls / (Nat.factorial redBalls * Nat.factorial yellowBalls * Nat.factorial whiteBalls) :=
by sorry

end NUMINAMATH_CALUDE_correct_ball_arrangements_l891_89126


namespace NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_B_l891_89149

/-- Proposition A: 0 < x < 5 -/
def prop_A (x : ℝ) : Prop := 0 < x ∧ x < 5

/-- Proposition B: |x - 2| < 3 -/
def prop_B (x : ℝ) : Prop := |x - 2| < 3

theorem A_sufficient_not_necessary_for_B :
  (∀ x : ℝ, prop_A x → prop_B x) ∧
  (∃ x : ℝ, prop_B x ∧ ¬prop_A x) := by sorry

end NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_B_l891_89149


namespace NUMINAMATH_CALUDE_jumps_before_cleaning_l891_89195

-- Define the pool characteristics
def pool_capacity : ℝ := 1200  -- in liters
def splash_out_volume : ℝ := 0.2  -- in liters (200 ml = 0.2 L)
def cleaning_threshold : ℝ := 0.8  -- 80% capacity

-- Define the number of jumps
def number_of_jumps : ℕ := 1200

-- Theorem statement
theorem jumps_before_cleaning :
  ⌊(pool_capacity - pool_capacity * cleaning_threshold) / splash_out_volume⌋ = number_of_jumps := by
  sorry

end NUMINAMATH_CALUDE_jumps_before_cleaning_l891_89195


namespace NUMINAMATH_CALUDE_fixed_point_on_parabola_l891_89166

theorem fixed_point_on_parabola (a b c : ℝ) 
  (h1 : |a| ≥ |b - c|) 
  (h2 : |b| ≥ |a + c|) 
  (h3 : |c| ≥ |a - b|) : 
  a * (-1)^2 + b * (-1) + c = 0 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_on_parabola_l891_89166


namespace NUMINAMATH_CALUDE_class_size_is_fifteen_l891_89111

-- Define the number of students
def N : ℕ := sorry

-- Define the average age function
def averageAge (numStudents : ℕ) (totalAge : ℕ) : ℚ :=
  totalAge / numStudents

-- Theorem statement
theorem class_size_is_fifteen :
  -- Conditions
  (averageAge (N - 1) (15 * (N - 1)) = 15) →
  (averageAge 4 (14 * 4) = 14) →
  (averageAge 9 (16 * 9) = 16) →
  -- Conclusion
  N = 15 := by
  sorry

end NUMINAMATH_CALUDE_class_size_is_fifteen_l891_89111


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l891_89125

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (chocolate_pies marshmallow_pies cayenne_pies peanut_pies : ℕ) : 
  total_pies = 48 →
  chocolate_pies ≥ total_pies / 2 →
  marshmallow_pies ≥ 2 * total_pies / 3 →
  cayenne_pies ≥ 3 * total_pies / 5 →
  peanut_pies ≥ total_pies / 8 →
  ∃ (pies_without_ingredients : ℕ), 
    pies_without_ingredients ≤ 16 ∧
    pies_without_ingredients + chocolate_pies + marshmallow_pies + cayenne_pies + peanut_pies ≥ total_pies :=
by sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l891_89125


namespace NUMINAMATH_CALUDE_cube_volume_in_pyramid_l891_89198

/-- Represents a pyramid with a square base and isosceles right triangle lateral faces -/
structure Pyramid where
  base_side_length : ℝ
  is_square_base : base_side_length = 2
  is_isosceles_right_triangle_faces : True

/-- Represents a cube inside the pyramid -/
structure InsideCube where
  edge_length : ℝ
  vertex_at_base_center : True
  three_vertices_touch_faces : True

/-- The volume of the cube inside the pyramid is 1 -/
theorem cube_volume_in_pyramid (p : Pyramid) (c : InsideCube) : c.edge_length ^ 3 = 1 := by
  sorry

#check cube_volume_in_pyramid

end NUMINAMATH_CALUDE_cube_volume_in_pyramid_l891_89198


namespace NUMINAMATH_CALUDE_units_digit_problem_l891_89153

theorem units_digit_problem : (8 * 25 * 983 - 8^3) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l891_89153


namespace NUMINAMATH_CALUDE_fraction_of_single_men_l891_89138

theorem fraction_of_single_men (total : ℝ) (h1 : total > 0) : 
  let women := 0.64 * total
  let men := total - women
  let married := 0.60 * total
  let married_women := 0.75 * women
  let married_men := married - married_women
  let single_men := men - married_men
  single_men / men = 2/3 := by sorry

end NUMINAMATH_CALUDE_fraction_of_single_men_l891_89138


namespace NUMINAMATH_CALUDE_custodian_jugs_l891_89171

/-- The number of cups a full jug can hold -/
def jug_capacity : ℕ := 40

/-- The number of students -/
def num_students : ℕ := 200

/-- The number of cups each student drinks per day -/
def cups_per_student : ℕ := 10

/-- Calculates the number of jugs needed to provide water for all students -/
def jugs_needed : ℕ := (num_students * cups_per_student) / jug_capacity

theorem custodian_jugs : jugs_needed = 50 := by
  sorry

end NUMINAMATH_CALUDE_custodian_jugs_l891_89171


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l891_89176

/-- Given three lines that intersect at the same point, prove the value of m -/
theorem intersection_of_three_lines (x y : ℝ) (m : ℝ) :
  (y = 4 * x + 2) ∧ 
  (y = -3 * x - 18) ∧ 
  (y = 2 * x + m) →
  m = -26 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l891_89176


namespace NUMINAMATH_CALUDE_x_value_l891_89188

theorem x_value (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((5 * x) / 7) = x) : x = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l891_89188


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l891_89190

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = q * a n

theorem common_ratio_of_geometric_sequence (a : ℕ → ℚ) 
  (h_geometric : geometric_sequence a) 
  (h_a1 : a 1 = 1/8)
  (h_a4 : a 4 = -1) :
  ∃ q : ℚ, q = -2 ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
by
  sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l891_89190


namespace NUMINAMATH_CALUDE_additional_money_needed_per_twin_l891_89148

def initial_amount : ℝ := 50
def toilet_paper_cost : ℝ := 12
def groceries_cost : ℝ := 2 * toilet_paper_cost
def remaining_after_groceries : ℝ := initial_amount - toilet_paper_cost - groceries_cost
def boot_cost : ℝ := 3 * remaining_after_groceries
def total_boot_cost : ℝ := 2 * boot_cost

theorem additional_money_needed_per_twin : 
  (total_boot_cost - remaining_after_groceries) / 2 = 35 := by sorry

end NUMINAMATH_CALUDE_additional_money_needed_per_twin_l891_89148


namespace NUMINAMATH_CALUDE_lcm_gcd_sum_reciprocal_sum_l891_89186

theorem lcm_gcd_sum_reciprocal_sum (m n : ℕ+) 
  (h_lcm : Nat.lcm m n = 210)
  (h_gcd : Nat.gcd m n = 6)
  (h_sum : m + n = 72) :
  1 / (m : ℚ) + 1 / (n : ℚ) = 2 / 35 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_sum_reciprocal_sum_l891_89186


namespace NUMINAMATH_CALUDE_optimal_selling_price_l891_89189

-- Define the parameters
def purchasePrice : ℝ := 40
def initialSellingPrice : ℝ := 50
def initialQuantitySold : ℝ := 50

-- Define the relationship between price increase and quantity decrease
def priceIncrease : ℝ → ℝ := λ x => x
def quantityDecrease : ℝ → ℝ := λ x => x

-- Define the selling price and quantity sold as functions of price increase
def sellingPrice : ℝ → ℝ := λ x => initialSellingPrice + priceIncrease x
def quantitySold : ℝ → ℝ := λ x => initialQuantitySold - quantityDecrease x

-- Define the revenue function
def revenue : ℝ → ℝ := λ x => sellingPrice x * quantitySold x

-- Define the cost function
def cost : ℝ → ℝ := λ x => purchasePrice * quantitySold x

-- Define the profit function
def profit : ℝ → ℝ := λ x => revenue x - cost x

-- State the theorem
theorem optimal_selling_price :
  ∃ x : ℝ, x = 20 ∧ sellingPrice x = 70 ∧ 
  ∀ y : ℝ, profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l891_89189


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_l891_89142

theorem negation_of_universal_quantifier (x : ℝ) :
  (¬ ∀ m : ℝ, m ∈ Set.Icc 0 1 → x + 1 / x ≥ 2^m) ↔
  (∃ m : ℝ, m ∈ Set.Icc 0 1 ∧ x + 1 / x < 2^m) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_l891_89142


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_with_given_lengths_l891_89124

/-- A quadrilateral that can be inscribed in a circle -/
structure InscribedQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  diagonal : ℝ

/-- Theorem: A quadrilateral with side lengths 15, 36, 48, 27, and diagonal 54
    can be inscribed in a circle with diameter 54 -/
theorem inscribed_quadrilateral_with_given_lengths :
  ∃ (q : InscribedQuadrilateral),
    q.a = 15 ∧
    q.b = 36 ∧
    q.c = 48 ∧
    q.d = 27 ∧
    q.diagonal = 54 ∧
    (∃ (r : ℝ), r = 54 ∧ r = q.diagonal) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_with_given_lengths_l891_89124


namespace NUMINAMATH_CALUDE_statue_weight_theorem_l891_89172

/-- The weight of a statue after three weeks of carving --/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  initial_weight * (1 - 0.3) * (1 - 0.2) * (1 - 0.25)

/-- Theorem stating the weight of the final statue --/
theorem statue_weight_theorem :
  final_statue_weight 250 = 105 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_theorem_l891_89172


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l891_89105

theorem cubic_equation_solution (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27 * x^3) 
  (h3 : a - b = x) : 
  a = (x * (1 + Real.sqrt (115 / 3))) / 2 ∨ 
  a = (x * (1 - Real.sqrt (115 / 3))) / 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l891_89105


namespace NUMINAMATH_CALUDE_fourth_root_sum_squared_l891_89135

theorem fourth_root_sum_squared : 
  (Real.rpow (7 + 3 * Real.sqrt 5) (1/4) + Real.rpow (7 - 3 * Real.sqrt 5) (1/4))^4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_sum_squared_l891_89135


namespace NUMINAMATH_CALUDE_only_set_C_forms_triangle_l891_89136

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the given sets of lengths
def set_A : (ℝ × ℝ × ℝ) := (3, 4, 8)
def set_B : (ℝ × ℝ × ℝ) := (2, 5, 2)
def set_C : (ℝ × ℝ × ℝ) := (3, 5, 6)
def set_D : (ℝ × ℝ × ℝ) := (5, 6, 11)

-- Theorem stating that only set_C can form a triangle
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A.1 set_A.2.1 set_A.2.2) ∧
  ¬(can_form_triangle set_B.1 set_B.2.1 set_B.2.2) ∧
  (can_form_triangle set_C.1 set_C.2.1 set_C.2.2) ∧
  ¬(can_form_triangle set_D.1 set_D.2.1 set_D.2.2) :=
by sorry

end NUMINAMATH_CALUDE_only_set_C_forms_triangle_l891_89136


namespace NUMINAMATH_CALUDE_largest_prime_satisfying_inequality_l891_89100

theorem largest_prime_satisfying_inequality :
  ∃ (m : ℕ), m.Prime ∧ m^2 - 11*m + 28 < 0 ∧
  ∀ (n : ℕ), n.Prime → n^2 - 11*n + 28 < 0 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_satisfying_inequality_l891_89100


namespace NUMINAMATH_CALUDE_proportionality_check_l891_89174

-- Define the concept of direct and inverse proportionality
def is_directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k / x

-- Define the equations
def eq_A (x y : ℝ) : Prop := 2*x + 3*y = 5
def eq_B (x y : ℝ) : Prop := 7*x*y = 14
def eq_C (x y : ℝ) : Prop := x = 7*y + 1
def eq_D (x y : ℝ) : Prop := 4*x + 2*y = 8
def eq_E (x y : ℝ) : Prop := x/y = 5

-- Theorem statement
theorem proportionality_check :
  (¬ ∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_A x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (¬ ∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_D x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_B x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_C x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_E x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) :=
by sorry

end NUMINAMATH_CALUDE_proportionality_check_l891_89174


namespace NUMINAMATH_CALUDE_alan_market_cost_l891_89196

/-- Calculates the total cost of Alan's market purchase including discount and tax --/
def market_cost (egg_price : ℝ) (egg_quantity : ℕ) 
                (chicken_price : ℝ) (chicken_quantity : ℕ) 
                (milk_price : ℝ) (milk_quantity : ℕ) 
                (bread_price : ℝ) (bread_quantity : ℕ) 
                (chicken_discount : ℕ → ℕ) (tax_rate : ℝ) : ℝ :=
  let egg_cost := egg_price * egg_quantity
  let chicken_cost := chicken_price * (chicken_quantity - chicken_discount chicken_quantity)
  let milk_cost := milk_price * milk_quantity
  let bread_cost := bread_price * bread_quantity
  let subtotal := egg_cost + chicken_cost + milk_cost + bread_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that Alan's market cost is $103.95 --/
theorem alan_market_cost : 
  market_cost 2 20 8 6 4 3 3.5 2 (fun n => n / 4) 0.05 = 103.95 := by
  sorry

end NUMINAMATH_CALUDE_alan_market_cost_l891_89196


namespace NUMINAMATH_CALUDE_sin_cos_identity_l891_89159

theorem sin_cos_identity : Real.sin (40 * π / 180) * Real.sin (10 * π / 180) + 
                           Real.cos (40 * π / 180) * Real.sin (80 * π / 180) = 
                           Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l891_89159


namespace NUMINAMATH_CALUDE_bill_caroline_age_difference_l891_89110

theorem bill_caroline_age_difference (bill_age caroline_age : ℕ) : 
  bill_age + caroline_age = 26 →
  bill_age = 17 →
  ∃ x : ℕ, bill_age = 2 * caroline_age - x →
  2 * caroline_age - bill_age = 1 := by
sorry

end NUMINAMATH_CALUDE_bill_caroline_age_difference_l891_89110


namespace NUMINAMATH_CALUDE_overlap_area_bound_l891_89157

open Set

-- Define the type for rectangles
structure Rectangle where
  area : ℝ

-- Define the large rectangle
def largeRectangle : Rectangle :=
  { area := 5 }

-- Define the set of smaller rectangles
def smallRectangles : Set Rectangle :=
  { r : Rectangle | r.area = 1 }

-- State the theorem
theorem overlap_area_bound (n : ℕ) (h : n = 9) :
  ∃ (r₁ r₂ : Rectangle),
    r₁ ∈ smallRectangles ∧
    r₂ ∈ smallRectangles ∧
    r₁ ≠ r₂ ∧
    (∃ (overlap : Rectangle), overlap.area ≥ 1/9) :=
sorry

end NUMINAMATH_CALUDE_overlap_area_bound_l891_89157


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l891_89128

/-- An isosceles triangle with perimeter 7 and one side length 3 has equal sides of length 3 or 2 -/
theorem isosceles_triangle_side_length (a b c : ℝ) : 
  a + b + c = 7 →  -- perimeter is 7
  (a = 3 ∨ b = 3 ∨ c = 3) →  -- one side length is 3
  ((a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)) →  -- isosceles triangle condition
  (a = 3 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (b = 3 ∧ c = 3) ∨ (b = 2 ∧ c = 2) ∨ (a = 3 ∧ c = 3) ∨ (a = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l891_89128


namespace NUMINAMATH_CALUDE_equal_cost_at_four_students_agency_a_cheaper_for_ten_students_l891_89132

/-- Represents a travel agency with a pricing strategy -/
structure Agency where
  teacherDiscount : ℝ  -- Discount for the teacher (0 means full price)
  studentDiscount : ℝ  -- Discount for students

/-- Calculates the total cost for a given number of students -/
def totalCost (a : Agency) (numStudents : ℕ) (fullPrice : ℝ) : ℝ :=
  fullPrice * (1 - a.teacherDiscount) + numStudents * fullPrice * (1 - a.studentDiscount)

/-- The full price of a ticket -/
def fullPrice : ℝ := 240

/-- Agency A's pricing strategy -/
def agencyA : Agency := ⟨0, 0.5⟩

/-- Agency B's pricing strategy -/
def agencyB : Agency := ⟨0.4, 0.4⟩

theorem equal_cost_at_four_students :
  ∃ n : ℕ, n = 4 ∧ totalCost agencyA n fullPrice = totalCost agencyB n fullPrice :=
sorry

theorem agency_a_cheaper_for_ten_students :
  totalCost agencyA 10 fullPrice < totalCost agencyB 10 fullPrice :=
sorry

end NUMINAMATH_CALUDE_equal_cost_at_four_students_agency_a_cheaper_for_ten_students_l891_89132


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l891_89158

def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2)

theorem f_derivative_at_zero : 
  (deriv f) 0 = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l891_89158


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l891_89139

universe u

def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U : 
  (U \ A) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l891_89139


namespace NUMINAMATH_CALUDE_cody_dumplings_l891_89192

/-- The number of dumplings Cody cooked -/
def dumplings_cooked : ℕ := sorry

/-- The number of dumplings Cody ate -/
def dumplings_eaten : ℕ := 7

/-- The number of dumplings Cody has left -/
def dumplings_left : ℕ := 7

theorem cody_dumplings : dumplings_cooked = dumplings_eaten + dumplings_left := by
  sorry

end NUMINAMATH_CALUDE_cody_dumplings_l891_89192


namespace NUMINAMATH_CALUDE_draw_with_replacement_l891_89199

/-- The number of items to choose from -/
def n : ℕ := 15

/-- The number of times we draw -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from n items -/
def num_lists (n k : ℕ) : ℕ := n^k

theorem draw_with_replacement :
  num_lists n k = 50625 := by
  sorry

end NUMINAMATH_CALUDE_draw_with_replacement_l891_89199


namespace NUMINAMATH_CALUDE_prob_white_after_red_is_correct_l891_89165

/-- Represents the number of balls in the box -/
def total_balls : ℕ := 20

/-- Represents the number of red balls initially -/
def red_balls : ℕ := 10

/-- Represents the number of white balls initially -/
def white_balls : ℕ := 10

/-- The probability of drawing a white ball after a red ball is drawn -/
def prob_white_after_red : ℚ := white_balls / (total_balls - 1)

theorem prob_white_after_red_is_correct : 
  prob_white_after_red = 10 / 19 := by sorry

end NUMINAMATH_CALUDE_prob_white_after_red_is_correct_l891_89165


namespace NUMINAMATH_CALUDE_tim_and_linda_mowing_time_l891_89146

/-- The time it takes for two people to complete a task together, given their individual rates -/
def combined_time (rate1 rate2 : ℚ) : ℚ :=
  1 / (rate1 + rate2)

/-- Proof that Tim and Linda can mow the lawn together in 6/7 hours -/
theorem tim_and_linda_mowing_time :
  let tim_rate : ℚ := 1 / (3/2)  -- Tim's rate: 1 lawn per 1.5 hours
  let linda_rate : ℚ := 1 / 2    -- Linda's rate: 1 lawn per 2 hours
  combined_time tim_rate linda_rate = 6/7 := by
  sorry

#eval (combined_time (1 / (3/2)) (1 / 2))

end NUMINAMATH_CALUDE_tim_and_linda_mowing_time_l891_89146


namespace NUMINAMATH_CALUDE_acid_mixture_percentage_l891_89141

theorem acid_mixture_percentage :
  ∀ (a w : ℝ),
  (a > 0) →
  (w > 0) →
  (a / (a + w + 2) = 0.3) →
  ((a + 2) / (a + w + 4) = 0.4) →
  (a / (a + w) = 0.36) :=
λ a w ha hw h1 h2 => by
  sorry

end NUMINAMATH_CALUDE_acid_mixture_percentage_l891_89141


namespace NUMINAMATH_CALUDE_real_solutions_of_equation_l891_89184

theorem real_solutions_of_equation (x : ℝ) : 
  x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by
sorry

end NUMINAMATH_CALUDE_real_solutions_of_equation_l891_89184


namespace NUMINAMATH_CALUDE_not_always_input_start_output_end_l891_89178

/-- Represents the types of boxes in a program flowchart -/
inductive FlowchartBox
  | Start
  | Input
  | Process
  | Output
  | End

/-- Represents a program flowchart as a list of boxes -/
def Flowchart := List FlowchartBox

/-- Checks if the input box immediately follows the start box -/
def inputFollowsStart (f : Flowchart) : Prop :=
  match f with
  | FlowchartBox.Start :: FlowchartBox.Input :: _ => True
  | _ => False

/-- Checks if the output box immediately precedes the end box -/
def outputPrecedesEnd (f : Flowchart) : Prop :=
  match f.reverse with
  | FlowchartBox.End :: FlowchartBox.Output :: _ => True
  | _ => False

/-- Theorem stating that it's not always true that input must follow start
    and output must precede end in a flowchart -/
theorem not_always_input_start_output_end :
  ∃ (f : Flowchart), ¬(inputFollowsStart f ∧ outputPrecedesEnd f) :=
sorry

end NUMINAMATH_CALUDE_not_always_input_start_output_end_l891_89178


namespace NUMINAMATH_CALUDE_max_k_for_no_real_roots_l891_89185

theorem max_k_for_no_real_roots (k : ℤ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_no_real_roots_l891_89185


namespace NUMINAMATH_CALUDE_acute_angle_range_l891_89143

theorem acute_angle_range (α : Real) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α < Real.cos α) : 
  α < π / 4 := by
sorry

end NUMINAMATH_CALUDE_acute_angle_range_l891_89143


namespace NUMINAMATH_CALUDE_survivor_quitters_probability_l891_89109

-- Define the total number of contestants
def total_contestants : ℕ := 20

-- Define the number of tribes
def num_tribes : ℕ := 4

-- Define the number of contestants per tribe
def contestants_per_tribe : ℕ := 5

-- Define the number of quitters
def num_quitters : ℕ := 3

-- Theorem statement
theorem survivor_quitters_probability :
  let total_ways := Nat.choose total_contestants num_quitters
  let same_tribe_ways := num_tribes * Nat.choose contestants_per_tribe num_quitters
  (same_tribe_ways : ℚ) / total_ways = 2 / 57 := by
  sorry


end NUMINAMATH_CALUDE_survivor_quitters_probability_l891_89109


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l891_89169

theorem magnitude_of_complex_fourth_power : 
  Complex.abs ((5 - 2 * Complex.I * Real.sqrt 3) ^ 4) = 1369 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l891_89169


namespace NUMINAMATH_CALUDE_negation_of_implication_negation_of_greater_than_one_l891_89101

theorem negation_of_implication (p q : Prop) :
  ¬(p → q) ↔ (p ∧ ¬q) := by sorry

theorem negation_of_greater_than_one :
  ¬(∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∃ x : ℝ, x ≤ 1 ∧ x^2 ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_negation_of_greater_than_one_l891_89101


namespace NUMINAMATH_CALUDE_marie_binders_count_l891_89137

theorem marie_binders_count :
  let notebooks_count : ℕ := 4
  let stamps_per_notebook : ℕ := 20
  let stamps_per_binder : ℕ := 50
  let kept_fraction : ℚ := 1/4
  let stamps_given_away : ℕ := 135
  ∃ binders_count : ℕ,
    (notebooks_count * stamps_per_notebook + binders_count * stamps_per_binder) * (1 - kept_fraction) = stamps_given_away ∧
    binders_count = 2 :=
by sorry

end NUMINAMATH_CALUDE_marie_binders_count_l891_89137


namespace NUMINAMATH_CALUDE_ratio_percentage_difference_l891_89167

theorem ratio_percentage_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_ratio : x / 8 = y / 7) : (y - x) / x = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_percentage_difference_l891_89167


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_l891_89145

/-- Given an isosceles triangle ABC with AB = BC = a and AC = b, 
    if ax² - √2·bx + a = 0 has two real roots with absolute difference √2,
    then ∠ABC = 120° -/
theorem isosceles_triangle_angle (a b : ℝ) (u : ℝ) : 
  a > 0 → b > 0 →
  (∃ x y : ℝ, x ≠ y ∧ 
    a * x^2 - Real.sqrt 2 * b * x + a = 0 ∧ 
    a * y^2 - Real.sqrt 2 * b * y + a = 0 ∧
    |x - y| = Real.sqrt 2) →
  b^2 = 2 * a^2 * (1 - Real.cos u) →
  u = 120 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_l891_89145


namespace NUMINAMATH_CALUDE_inequality_proof_l891_89113

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b) > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l891_89113


namespace NUMINAMATH_CALUDE_marks_theater_cost_l891_89147

/-- The cost of Mark's theater visits over a given number of weeks -/
def theater_cost (weeks : ℕ) (hours_per_visit : ℕ) (price_per_hour : ℕ) : ℕ :=
  weeks * hours_per_visit * price_per_hour

/-- Theorem: Mark's theater visits cost $90 over 6 weeks -/
theorem marks_theater_cost :
  theater_cost 6 3 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_marks_theater_cost_l891_89147


namespace NUMINAMATH_CALUDE_circle_intersection_condition_tangent_length_l891_89187

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O1 (x y r : ℝ) : Prop := (x-2)^2 + y^2 = r^2
def line (m x y : ℝ) : Prop := m*x + y - m - 1 = 0

-- Statement 1
theorem circle_intersection_condition (r : ℝ) (h : r > 0) :
  (∀ m : ℝ, ∃ x1 y1 x2 y2 : ℝ, 
    x1 ≠ x2 ∧ 
    circle_O1 x1 y1 r ∧ line m x1 y1 ∧
    circle_O1 x2 y2 r ∧ line m x2 y2) ↔ 
  r > Real.sqrt 2 :=
sorry

-- Statement 2
theorem tangent_length (A B : ℝ × ℝ) :
  (∃ t : ℝ, circle_O (A.1) (A.2) ∧ circle_O (B.1) (B.2) ∧
    (∀ x y : ℝ, circle_O x y → (x - 0)*(A.2 - 2) = (y - 2)*(A.1 - 0) ∧
                               (x - 0)*(B.2 - 2) = (y - 2)*(B.1 - 0))) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_condition_tangent_length_l891_89187


namespace NUMINAMATH_CALUDE_probability_is_one_third_l891_89112

/-- A game board consisting of an equilateral triangle divided into six smaller triangles -/
structure GameBoard where
  /-- The total number of smaller triangles in the game board -/
  total_triangles : ℕ
  /-- The number of shaded triangles in the game board -/
  shaded_triangles : ℕ
  /-- The shaded triangles are non-adjacent -/
  non_adjacent : Bool
  /-- The total number of triangles is 6 -/
  h_total : total_triangles = 6
  /-- The number of shaded triangles is 2 -/
  h_shaded : shaded_triangles = 2
  /-- The shaded triangles are indeed non-adjacent -/
  h_non_adjacent : non_adjacent = true

/-- The probability of a spinner landing in a shaded region -/
def probability (board : GameBoard) : ℚ :=
  board.shaded_triangles / board.total_triangles

/-- Theorem stating that the probability of landing in a shaded region is 1/3 -/
theorem probability_is_one_third (board : GameBoard) : probability board = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l891_89112


namespace NUMINAMATH_CALUDE_proportional_enlargement_l891_89121

/-- Given a rectangle that is enlarged proportionally, this theorem proves
    that the new height can be calculated from the original dimensions
    and the new width. -/
theorem proportional_enlargement
  (original_width original_height new_width : ℝ)
  (h_positive : original_width > 0 ∧ original_height > 0 ∧ new_width > 0)
  (h_original_width : original_width = 2)
  (h_original_height : original_height = 1.5)
  (h_new_width : new_width = 8) :
  let new_height := original_height * (new_width / original_width)
  new_height = 6 := by
sorry

end NUMINAMATH_CALUDE_proportional_enlargement_l891_89121


namespace NUMINAMATH_CALUDE_stating_perpendicular_bisector_correct_l891_89193

/-- The perpendicular bisector of a line segment. -/
def perpendicular_bisector (line_eq : ℝ → ℝ → Prop) (x_range : Set ℝ) : ℝ → ℝ → Prop :=
  fun x y => 2 * x + y - 3 = 0

/-- The original line segment equation. -/
def original_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

/-- The range of x for the original line segment. -/
def x_range : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

/-- 
Theorem stating that the perpendicular_bisector function correctly defines 
the perpendicular bisector of the line segment given by the original_line 
equation within the specified x_range.
-/
theorem perpendicular_bisector_correct : 
  perpendicular_bisector original_line x_range = 
    fun x y => 2 * x + y - 3 = 0 := by sorry

end NUMINAMATH_CALUDE_stating_perpendicular_bisector_correct_l891_89193


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l891_89183

/-- Given a quadratic function f(x) = ax² + bx + c where 2a + 3b + 6c = 0,
    there exists an x in the interval (0,1) such that f(x) = 0. -/
theorem quadratic_root_in_unit_interval (a b c : ℝ) 
  (h : 2*a + 3*b + 6*c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l891_89183


namespace NUMINAMATH_CALUDE_solve_for_a_l891_89152

theorem solve_for_a (A : Set ℝ) (a : ℝ) 
  (h1 : A = {a - 2, 2 * a^2 + 5 * a, 12})
  (h2 : -3 ∈ A) : 
  a = -3/2 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l891_89152


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l891_89140

/-- Given that y and x are inversely proportional -/
def inversely_proportional (y x : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y * x = k

/-- The theorem to prove -/
theorem inverse_proportion_problem (y₁ y₂ : ℝ) :
  inversely_proportional y₁ 4 ∧ y₁ = 30 →
  inversely_proportional y₂ 10 →
  y₂ = 12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l891_89140


namespace NUMINAMATH_CALUDE_percentage_with_diploma_l891_89123

theorem percentage_with_diploma (total : ℝ) 
  (no_diploma_but_choice : ℝ) 
  (no_choice_but_diploma_ratio : ℝ) 
  (job_of_choice : ℝ) :
  no_diploma_but_choice = 0.1 * total →
  no_choice_but_diploma_ratio = 0.15 →
  job_of_choice = 0.4 * total →
  ∃ (with_diploma : ℝ), 
    with_diploma = 0.39 * total ∧
    with_diploma = (job_of_choice - no_diploma_but_choice) + 
                   (no_choice_but_diploma_ratio * (total - job_of_choice)) :=
by sorry

end NUMINAMATH_CALUDE_percentage_with_diploma_l891_89123


namespace NUMINAMATH_CALUDE_harrys_annual_pet_feeding_cost_l891_89168

/-- Calculates the annual cost of feeding pets given the number of each type and their monthly feeding costs. -/
def annual_pet_feeding_cost (num_geckos num_iguanas num_snakes : ℕ) 
                            (gecko_cost iguana_cost snake_cost : ℕ) : ℕ :=
  12 * (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * snake_cost)

/-- Theorem stating that Harry's annual pet feeding cost is $1140. -/
theorem harrys_annual_pet_feeding_cost : 
  annual_pet_feeding_cost 3 2 4 15 5 10 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_harrys_annual_pet_feeding_cost_l891_89168


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l891_89156

theorem rectangle_area_increase (L W P : ℝ) (h : L > 0) (h' : W > 0) (h'' : P > 0) :
  (L * (1 + P)) * (W * (1 + P)) = 4 * (L * W) → P = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l891_89156
