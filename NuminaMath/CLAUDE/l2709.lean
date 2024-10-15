import Mathlib

namespace NUMINAMATH_CALUDE_unique_k_l2709_270911

theorem unique_k : ∃! (k : ℕ), k > 0 ∧ (k + 2).factorial + (k + 3).factorial = k.factorial * 1344 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_l2709_270911


namespace NUMINAMATH_CALUDE_profit_maximized_at_100_l2709_270961

/-- The profit function L(x) for annual production x (in thousand units) -/
noncomputable def L (x : ℝ) : ℝ :=
  if x < 80 then
    -1/3 * x^2 + 40 * x - 250
  else
    1200 - (x + 10000 / x)

/-- Annual fixed cost in ten thousand yuan -/
def annual_fixed_cost : ℝ := 250

/-- Price per unit in ten thousand yuan -/
def price_per_unit : ℝ := 50

theorem profit_maximized_at_100 :
  ∀ x > 0, L x ≤ L 100 :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_100_l2709_270961


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2709_270997

theorem polynomial_factorization (x : ℝ) :
  x^6 - 3*x^4 + 3*x^2 - 1 = (x-1)^3*(x+1)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2709_270997


namespace NUMINAMATH_CALUDE_simplify_fraction_l2709_270930

theorem simplify_fraction : 
  (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2709_270930


namespace NUMINAMATH_CALUDE_bus_passengers_after_four_stops_l2709_270948

/-- Represents the change in passengers at a bus stop -/
structure StopChange where
  boarding : Int
  alighting : Int

/-- Calculates the final number of passengers on a bus after a series of stops -/
def finalPassengers (initial : Int) (changes : List StopChange) : Int :=
  changes.foldl (fun acc stop => acc + stop.boarding - stop.alighting) initial

/-- Theorem stating the final number of passengers after 4 stops -/
theorem bus_passengers_after_four_stops :
  let initial := 22
  let changes := [
    { boarding := 3, alighting := 6 },
    { boarding := 8, alighting := 5 },
    { boarding := 2, alighting := 4 },
    { boarding := 1, alighting := 8 }
  ]
  finalPassengers initial changes = 13 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_after_four_stops_l2709_270948


namespace NUMINAMATH_CALUDE_numerator_smaller_than_a_l2709_270929

theorem numerator_smaller_than_a (a b n : ℕ) (h1 : a ≠ 1) (h2 : b > 0) 
  (h3 : Nat.gcd a b = 1) (h4 : (n : ℚ)⁻¹ > a / b) (h5 : a / b > (n + 1 : ℚ)⁻¹) :
  ∃ (p q : ℕ), q > 0 ∧ Nat.gcd p q = 1 ∧ 
  (a : ℚ) / b - (n + 1 : ℚ)⁻¹ = (p : ℚ) / q ∧ p < a := by
  sorry

end NUMINAMATH_CALUDE_numerator_smaller_than_a_l2709_270929


namespace NUMINAMATH_CALUDE_right_angled_constructions_l2709_270907

/-- Represents a triangle with angles in degrees -/
structure Triangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)

/-- Checks if a triangle is right-angled -/
def is_right_angled (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

/-- The basic triangle obtained from dividing a regular hexagon into 12 parts -/
def basic_triangle : Triangle :=
  { angle1 := 30, angle2 := 60, angle3 := 90 }

/-- Represents the number of basic triangles used to form a larger triangle -/
inductive TriangleComposition
  | One
  | Three
  | Four
  | Nine

/-- Function to construct a triangle from a given number of basic triangles -/
def construct_triangle (n : TriangleComposition) : Triangle :=
  sorry

/-- Theorem stating that right-angled triangles can be formed using 1, 3, 4, or 9 basic triangles -/
theorem right_angled_constructions :
  ∀ n : TriangleComposition, is_right_angled (construct_triangle n) :=
sorry

end NUMINAMATH_CALUDE_right_angled_constructions_l2709_270907


namespace NUMINAMATH_CALUDE_convex_quadrilateral_symmetric_division_l2709_270962

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  -- We don't need to define the specifics of the quadrilateral,
  -- just that it exists and is convex
  isConvex : Bool

/-- A polygon with an axis of symmetry -/
structure SymmetricPolygon where
  -- We don't need to define the specifics of the polygon,
  -- just that it exists and has an axis of symmetry
  hasSymmetryAxis : Bool

/-- A division of a quadrilateral into polygons -/
structure QuadrilateralDivision (q : ConvexQuadrilateral) where
  polygons : List SymmetricPolygon
  divisionValid : Bool  -- This would ensure the division is valid

/-- The main theorem -/
theorem convex_quadrilateral_symmetric_division 
  (q : ConvexQuadrilateral) : 
  ∃ (d : QuadrilateralDivision q), 
    d.polygons.length = 5 ∧ 
    d.divisionValid ∧ 
    ∀ p ∈ d.polygons, p.hasSymmetryAxis := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_symmetric_division_l2709_270962


namespace NUMINAMATH_CALUDE_no_intersection_and_constraint_l2709_270969

theorem no_intersection_and_constraint (a b : ℝ) : 
  ¬(∃ (x : ℤ), a * (x : ℝ) + b = 3 * (x : ℝ)^2 + 15 ∧ a^2 + b^2 ≤ 144) :=
sorry

end NUMINAMATH_CALUDE_no_intersection_and_constraint_l2709_270969


namespace NUMINAMATH_CALUDE_complex_power_2007_l2709_270996

theorem complex_power_2007 : (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I)) ^ 2007 = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_2007_l2709_270996


namespace NUMINAMATH_CALUDE_min_sum_squares_l2709_270999

theorem min_sum_squares (x y z : ℝ) (h : x + y + z = 1) : x^2 + y^2 + z^2 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2709_270999


namespace NUMINAMATH_CALUDE_intersection_point_on_line_and_plane_l2709_270937

/-- The line passing through the point (5, 2, -4) in the direction <-2, 0, -1> -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (5 - 2*t, 2, -4 - t)

/-- The plane 2x - 5y + 4z + 24 = 0 -/
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  2*x - 5*y + 4*z + 24 = 0

/-- The intersection point of the line and the plane -/
def intersection_point : ℝ × ℝ × ℝ := (3, 2, -5)

theorem intersection_point_on_line_and_plane :
  ∃ t : ℝ, line t = intersection_point ∧ plane intersection_point := by
  sorry


end NUMINAMATH_CALUDE_intersection_point_on_line_and_plane_l2709_270937


namespace NUMINAMATH_CALUDE_total_distance_driven_l2709_270953

def miles_per_gallon : ℝ := 25
def tank_capacity : ℝ := 18
def initial_gas : ℝ := 12
def first_leg_distance : ℝ := 250
def gas_purchased : ℝ := 10
def final_gas : ℝ := 3

theorem total_distance_driven : ℝ := by
  -- The total distance driven is 475 miles
  sorry

#check total_distance_driven

end NUMINAMATH_CALUDE_total_distance_driven_l2709_270953


namespace NUMINAMATH_CALUDE_prob_second_draw_black_l2709_270932

/-- The probability of drawing a black ball on the second draw without replacement -/
def second_draw_black_prob (total : ℕ) (black : ℕ) (white : ℕ) : ℚ :=
  if total = black + white ∧ black > 0 ∧ white > 0 then
    black / (total - 1)
  else
    0

theorem prob_second_draw_black :
  second_draw_black_prob 10 3 7 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_draw_black_l2709_270932


namespace NUMINAMATH_CALUDE_solid_figures_count_l2709_270926

-- Define the list of shapes
inductive Shape
  | Circle
  | Square
  | Cone
  | Cuboid
  | LineSegment
  | Sphere
  | TriangularPrism
  | RightAngledTriangle

-- Define a function to determine if a shape is solid
def isSolid (s : Shape) : Bool :=
  match s with
  | Shape.Cone => true
  | Shape.Cuboid => true
  | Shape.Sphere => true
  | Shape.TriangularPrism => true
  | _ => false

-- Define the list of shapes
def shapeList : List Shape := [
  Shape.Circle,
  Shape.Square,
  Shape.Cone,
  Shape.Cuboid,
  Shape.LineSegment,
  Shape.Sphere,
  Shape.TriangularPrism,
  Shape.RightAngledTriangle
]

-- Theorem: The number of solid figures in the list is 4
theorem solid_figures_count :
  (shapeList.filter isSolid).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_solid_figures_count_l2709_270926


namespace NUMINAMATH_CALUDE_carson_giant_slide_rides_l2709_270939

/-- Represents the number of times Carson can ride the giant slide at the carnival -/
def giant_slide_rides (total_time minutes_per_hour roller_coaster_wait tilt_a_whirl_wait giant_slide_wait : ℕ)
  (roller_coaster_rides tilt_a_whirl_rides : ℕ) : ℕ :=
  let remaining_time := total_time * minutes_per_hour -
    (roller_coaster_wait * roller_coaster_rides + tilt_a_whirl_wait * tilt_a_whirl_rides)
  remaining_time / giant_slide_wait

/-- Theorem stating the number of times Carson can ride the giant slide -/
theorem carson_giant_slide_rides :
  giant_slide_rides 4 60 30 60 15 4 1 = 4 :=
sorry

end NUMINAMATH_CALUDE_carson_giant_slide_rides_l2709_270939


namespace NUMINAMATH_CALUDE_couple_consistency_l2709_270978

-- Define the possible types of people
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a couple
structure Couple :=
  (husband : PersonType)
  (wife : PersonType)

-- Define the statement made by a person about their spouse
def makeStatement (speaker : PersonType) (spouse : PersonType) : Prop :=
  spouse ≠ PersonType.Normal

-- Define the consistency of statements with reality
def isConsistent (couple : Couple) : Prop :=
  match couple.husband, couple.wife with
  | PersonType.Knight, _ => makeStatement PersonType.Knight couple.wife
  | PersonType.Liar, _ => ¬(makeStatement PersonType.Liar couple.wife)
  | PersonType.Normal, PersonType.Knight => makeStatement PersonType.Normal couple.wife
  | PersonType.Normal, PersonType.Liar => ¬(makeStatement PersonType.Normal couple.wife)
  | PersonType.Normal, PersonType.Normal => True

-- Theorem stating that the only consistent solution is both being normal people
theorem couple_consistency :
  ∀ (couple : Couple),
    isConsistent couple ∧
    makeStatement couple.husband couple.wife ∧
    makeStatement couple.wife couple.husband →
    couple.husband = PersonType.Normal ∧
    couple.wife = PersonType.Normal :=
sorry

end NUMINAMATH_CALUDE_couple_consistency_l2709_270978


namespace NUMINAMATH_CALUDE_quadratic_roots_order_l2709_270946

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The statement of the problem -/
theorem quadratic_roots_order (b c x₁ x₂ x₃ x₄ : ℝ) :
  (∀ x, f b c x - x = 0 ↔ x = x₁ ∨ x = x₂) →
  x₂ - x₁ > 2 →
  (∀ x, f b c (f b c x) = x ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) →
  x₃ > x₄ →
  x₄ < x₁ ∧ x₁ < x₃ ∧ x₃ < x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_order_l2709_270946


namespace NUMINAMATH_CALUDE_half_vector_MN_l2709_270918

/-- Given two vectors OM and ON in ℝ², prove that half of vector MN equals (-4, 1/2) -/
theorem half_vector_MN (OM ON : ℝ × ℝ) (h1 : OM = (3, -2)) (h2 : ON = (-5, -1)) :
  (1 / 2 : ℝ) • (ON - OM) = (-4, 1/2) := by sorry

end NUMINAMATH_CALUDE_half_vector_MN_l2709_270918


namespace NUMINAMATH_CALUDE_infinite_linear_combinations_l2709_270966

/-- An infinite sequence of positive integers with strictly increasing terms. -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, 0 < a k ∧ a k < a (k + 1)

/-- The property that infinitely many elements of the sequence can be written as a linear
    combination of two earlier terms with positive integer coefficients. -/
def InfinitelyManyLinearCombinations (a : ℕ → ℕ) : Prop :=
  ∀ N, ∃ m p q x y, N < m ∧ m > p ∧ p > q ∧ 0 < x ∧ 0 < y ∧ a m = x * a p + y * a q

/-- The main theorem stating that any strictly increasing sequence of positive integers
    has infinitely many elements that can be written as a linear combination of two earlier terms. -/
theorem infinite_linear_combinations
  (a : ℕ → ℕ) (h : StrictlyIncreasingSequence a) :
  InfinitelyManyLinearCombinations a := by
  sorry

end NUMINAMATH_CALUDE_infinite_linear_combinations_l2709_270966


namespace NUMINAMATH_CALUDE_sum_of_even_positive_integers_less_than_100_l2709_270993

theorem sum_of_even_positive_integers_less_than_100 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ n > 0) (Finset.range 100)).sum id = 2450 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_positive_integers_less_than_100_l2709_270993


namespace NUMINAMATH_CALUDE_same_grade_percentage_l2709_270970

-- Define the total number of students
def total_students : ℕ := 50

-- Define the number of students who got the same grade on both tests
def same_grade_A : ℕ := 3
def same_grade_B : ℕ := 6
def same_grade_C : ℕ := 7
def same_grade_D : ℕ := 2

-- Define the total number of students who got the same grade on both tests
def total_same_grade : ℕ := same_grade_A + same_grade_B + same_grade_C + same_grade_D

-- Define the percentage of students who got the same grade on both tests
def percentage_same_grade : ℚ := (total_same_grade : ℚ) / (total_students : ℚ) * 100

-- Theorem to prove
theorem same_grade_percentage :
  percentage_same_grade = 36 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l2709_270970


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2709_270968

open Set

theorem intersection_A_complement_B (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) 
  (hA : A = {2, 4}) (hB : B = {4, 5}) : A ∩ (U \ B) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2709_270968


namespace NUMINAMATH_CALUDE_probability_of_prime_on_die_l2709_270901

/-- A standard die with six faces -/
def StandardDie := Finset.range 6

/-- The set of prime numbers on a standard die -/
def PrimeNumbersOnDie : Finset Nat := {2, 3, 5}

/-- The probability of rolling a prime number on a standard die -/
def ProbabilityOfPrime : ℚ := (PrimeNumbersOnDie.card : ℚ) / (StandardDie.card : ℚ)

/-- The given probability expression -/
def GivenProbability (a : ℕ) : ℚ := (a : ℚ) / 72

theorem probability_of_prime_on_die (a : ℕ) :
  GivenProbability a = ProbabilityOfPrime → a = 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_prime_on_die_l2709_270901


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2709_270998

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l2709_270998


namespace NUMINAMATH_CALUDE_circle_parabola_tangent_radius_l2709_270923

-- Define the parabola Γ
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the circle Ω with center (1, r) and radius r
def Ω (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - r)^2 = r^2}

-- State the theorem
theorem circle_parabola_tangent_radius :
  ∃! r : ℝ, r > 0 ∧
  (∃! p : ℝ × ℝ, p ∈ Γ ∩ Ω r) ∧
  (1, 0) ∈ Ω r ∧
  (∀ ε > 0, ∃ q : ℝ × ℝ, q.2 = -ε ∧ q ∉ Ω r) ∧
  r = 4 * Real.sqrt 3 / 9 := by
sorry

end NUMINAMATH_CALUDE_circle_parabola_tangent_radius_l2709_270923


namespace NUMINAMATH_CALUDE_alternating_draw_probability_l2709_270958

/-- The number of white balls in the box -/
def white_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 3

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of ways to choose positions for black balls -/
def total_arrangements : ℕ := Nat.choose total_balls black_balls

/-- The number of successful alternating color arrangements -/
def successful_arrangements : ℕ := Nat.choose (total_balls - 2) black_balls

/-- The probability of drawing an alternating color sequence -/
def alternating_probability : ℚ := successful_arrangements / total_arrangements

theorem alternating_draw_probability :
  alternating_probability = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_alternating_draw_probability_l2709_270958


namespace NUMINAMATH_CALUDE_square_root_of_four_l2709_270915

theorem square_root_of_four :
  {x : ℝ | x ^ 2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l2709_270915


namespace NUMINAMATH_CALUDE_magnitude_of_b_l2709_270963

/-- Given two planar vectors a and b satisfying the specified conditions, 
    the magnitude of b is 2. -/
theorem magnitude_of_b (a b : ℝ × ℝ) : 
  (‖a‖ = 1) →
  (‖a - 2 • b‖ = Real.sqrt 21) →
  (a.1 * b.1 + a.2 * b.2 = ‖a‖ * ‖b‖ * (-1/2)) →
  ‖b‖ = 2 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_b_l2709_270963


namespace NUMINAMATH_CALUDE_best_washing_effect_and_full_capacity_l2709_270925

-- Define the constants
def drum_capacity : Real := 25
def current_clothes : Real := 4.92
def current_detergent_scoops : Nat := 3
def scoop_weight : Real := 0.02
def water_per_scoop : Real := 5

-- Define the variables for additional detergent and water
def additional_detergent : Real := 0.02
def additional_water : Real := 20

-- Theorem statement
theorem best_washing_effect_and_full_capacity : 
  -- The total weight equals the drum capacity
  current_clothes + (current_detergent_scoops * scoop_weight) + additional_detergent + additional_water = drum_capacity ∧
  -- The ratio of water to detergent is correct for best washing effect
  (current_detergent_scoops * scoop_weight + additional_detergent) / 
    (additional_water + water_per_scoop * current_detergent_scoops) = 1 / water_per_scoop :=
by sorry

end NUMINAMATH_CALUDE_best_washing_effect_and_full_capacity_l2709_270925


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l2709_270949

/-- Represents a hyperbola with the given equation and foci on the y-axis -/
structure Hyperbola (k : ℝ) :=
  (equation : ∀ x y : ℝ, x^2 / (k - 3) + y^2 / (k + 3) = 1)
  (foci_on_y_axis : True)  -- We can't directly represent this condition, so we use a placeholder

/-- The range of k for a hyperbola with the given properties -/
def k_range (h : Hyperbola k) : Set ℝ :=
  {k | -3 < k ∧ k < 3}

/-- Theorem stating that for any hyperbola satisfying the given conditions, k is in the range (-3, 3) -/
theorem hyperbola_k_range (k : ℝ) (h : Hyperbola k) : k ∈ k_range h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l2709_270949


namespace NUMINAMATH_CALUDE_sqrt_seven_plus_one_bounds_l2709_270941

theorem sqrt_seven_plus_one_bounds :
  3 < Real.sqrt 7 + 1 ∧ Real.sqrt 7 + 1 < 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_plus_one_bounds_l2709_270941


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l2709_270912

theorem unique_solution_sqrt_equation :
  ∃! z : ℝ, Real.sqrt (8 + 3 * z) = 10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l2709_270912


namespace NUMINAMATH_CALUDE_min_n_for_expansion_terms_min_n_value_l2709_270956

theorem min_n_for_expansion_terms (n : ℕ) : (n + 1) ^ 2 ≥ 2021 ↔ n ≥ 44 := by sorry

theorem min_n_value : ∃ (n : ℕ), n > 0 ∧ (n + 1) ^ 2 ≥ 2021 ∧ ∀ (m : ℕ), m > 0 → (m + 1) ^ 2 ≥ 2021 → m ≥ n := by
  use 44
  sorry

end NUMINAMATH_CALUDE_min_n_for_expansion_terms_min_n_value_l2709_270956


namespace NUMINAMATH_CALUDE_missing_number_proof_l2709_270920

theorem missing_number_proof : ∃ x : ℚ, (3/4 * 60 - 8/5 * 60 + x = 12) ∧ (x = 63) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2709_270920


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2709_270984

-- Define the complex number z
variable (z : ℂ)

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  (1 + i) * z = Complex.abs (1 + Real.sqrt 3 * i) →
  z = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2709_270984


namespace NUMINAMATH_CALUDE_fourth_dog_weight_l2709_270938

theorem fourth_dog_weight (y : ℝ) :
  let dog1 : ℝ := 25
  let dog2 : ℝ := 31
  let dog3 : ℝ := 35
  let dog4 : ℝ := x
  let dog5 : ℝ := y
  (dog1 + dog2 + dog3 + dog4) / 4 = (dog1 + dog2 + dog3 + dog4 + dog5) / 5 →
  x = -91 - 5 * y :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_dog_weight_l2709_270938


namespace NUMINAMATH_CALUDE_power_24_in_terms_of_a_and_t_l2709_270982

theorem power_24_in_terms_of_a_and_t (x a t : ℝ) 
  (h1 : 2^x = a) (h2 : 3^x = t) : 24^x = a^3 * t := by
  sorry

end NUMINAMATH_CALUDE_power_24_in_terms_of_a_and_t_l2709_270982


namespace NUMINAMATH_CALUDE_equation_solutions_l2709_270980

-- Define the equation
def equation (x : ℝ) : Prop := (x ^ (1/4 : ℝ)) = 16 / (9 - (x ^ (1/4 : ℝ)))

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = 4096) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2709_270980


namespace NUMINAMATH_CALUDE_man_half_father_age_l2709_270974

theorem man_half_father_age (father_age : ℝ) (man_age : ℝ) (years_later : ℝ) : 
  father_age = 30.000000000000007 →
  man_age = (2/5) * father_age →
  man_age + years_later = (1/2) * (father_age + years_later) →
  years_later = 6 := by
sorry

end NUMINAMATH_CALUDE_man_half_father_age_l2709_270974


namespace NUMINAMATH_CALUDE_intersection_when_a_is_4_union_condition_l2709_270972

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x > 5 ∨ x < -1}

-- Theorem 1: When a = 4, A ∩ B = {x | 6 < x ≤ 7}
theorem intersection_when_a_is_4 :
  A 4 ∩ B = {x | 6 < x ∧ x ≤ 7} := by sorry

-- Theorem 2: A ∪ B = B if and only if a < -4 or a > 5
theorem union_condition (a : ℝ) :
  A a ∪ B = B ↔ a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_4_union_condition_l2709_270972


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2709_270973

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt (a - 2) - Real.sqrt (a - 3) > Real.sqrt a - Real.sqrt (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2709_270973


namespace NUMINAMATH_CALUDE_josiah_cookies_per_day_l2709_270955

/-- Proves that Josiah purchased 2 cookies each day in March given the conditions --/
theorem josiah_cookies_per_day :
  let total_spent : ℕ := 992
  let cookie_price : ℕ := 16
  let days_in_march : ℕ := 31
  (total_spent / cookie_price) / days_in_march = 2 := by
  sorry

end NUMINAMATH_CALUDE_josiah_cookies_per_day_l2709_270955


namespace NUMINAMATH_CALUDE_zeljko_distance_l2709_270964

/-- Calculates the total distance travelled given two segments of a journey -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Proves that Zeljko's total distance travelled is 20 km -/
theorem zeljko_distance : 
  let speed1 : ℝ := 30  -- km/h
  let time1  : ℝ := 20 / 60  -- 20 minutes in hours
  let speed2 : ℝ := 20  -- km/h
  let time2  : ℝ := 30 / 60  -- 30 minutes in hours
  total_distance speed1 time1 speed2 time2 = 20 := by
  sorry


end NUMINAMATH_CALUDE_zeljko_distance_l2709_270964


namespace NUMINAMATH_CALUDE_polygon_sides_l2709_270935

theorem polygon_sides (n : ℕ) (angle_sum : ℝ) : 
  (n ≥ 3) →  -- Ensure it's a polygon
  (angle_sum = 2790) →  -- Given sum of angles except one
  (∃ x : ℝ, x > 0 ∧ x < 180 ∧ 180 * (n - 2) = angle_sum + x) →  -- Existence of the missing angle
  (n = 18) :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l2709_270935


namespace NUMINAMATH_CALUDE_average_age_combined_l2709_270947

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 35 →
  let total_age_students := num_students * avg_age_students
  let total_age_parents := num_parents * avg_age_parents
  let total_people := num_students + num_parents
  let total_age := total_age_students + total_age_parents
  (total_age / total_people : ℚ) = 25.8 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l2709_270947


namespace NUMINAMATH_CALUDE_positive_number_square_sum_l2709_270985

theorem positive_number_square_sum (x : ℝ) : 
  0 < x → x < 15 → x^2 + x = 210 → x = 14 := by sorry

end NUMINAMATH_CALUDE_positive_number_square_sum_l2709_270985


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l2709_270904

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l2709_270904


namespace NUMINAMATH_CALUDE_cashier_bills_l2709_270994

theorem cashier_bills (total_bills : ℕ) (total_value : ℕ) 
  (h1 : total_bills = 126) 
  (h2 : total_value = 840) : ∃ (five_dollar_bills ten_dollar_bills : ℕ), 
  five_dollar_bills + ten_dollar_bills = total_bills ∧ 
  5 * five_dollar_bills + 10 * ten_dollar_bills = total_value ∧ 
  five_dollar_bills = 84 := by
sorry

end NUMINAMATH_CALUDE_cashier_bills_l2709_270994


namespace NUMINAMATH_CALUDE_maximum_marks_proof_l2709_270987

/-- Given a student needs 50% to pass, got 200 marks, and failed by 20 marks, prove the maximum marks are 440. -/
theorem maximum_marks_proof (passing_percentage : Real) (student_marks : Nat) (failing_margin : Nat) :
  passing_percentage = 0.5 →
  student_marks = 200 →
  failing_margin = 20 →
  ∃ (max_marks : Nat), max_marks = 440 ∧ 
    passing_percentage * max_marks = student_marks + failing_margin :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_proof_l2709_270987


namespace NUMINAMATH_CALUDE_correct_value_for_square_l2709_270903

theorem correct_value_for_square (x : ℕ) : 60 + x * 5 = 500 ↔ x = 88 :=
by sorry

end NUMINAMATH_CALUDE_correct_value_for_square_l2709_270903


namespace NUMINAMATH_CALUDE_intersection_sum_l2709_270981

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 1 = (y - 2)^2

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
  x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_sum_l2709_270981


namespace NUMINAMATH_CALUDE_trihedral_angle_properties_l2709_270905

-- Define a trihedral angle
structure TrihedralAngle where
  planeAngle1 : ℝ
  planeAngle2 : ℝ
  planeAngle3 : ℝ
  dihedralAngle1 : ℝ
  dihedralAngle2 : ℝ
  dihedralAngle3 : ℝ

-- State the theorem
theorem trihedral_angle_properties (t : TrihedralAngle) :
  t.planeAngle1 + t.planeAngle2 + t.planeAngle3 < 2 * Real.pi ∧
  t.dihedralAngle1 + t.dihedralAngle2 + t.dihedralAngle3 > Real.pi :=
by sorry

end NUMINAMATH_CALUDE_trihedral_angle_properties_l2709_270905


namespace NUMINAMATH_CALUDE_A_is_integer_l2709_270991

def n : ℤ := 8795685

def A : ℚ :=
  (((n + 4) * (n + 3) * (n + 2) * (n + 1)) - ((n - 1) * (n - 2) * (n - 3) * (n - 4))) /
  ((n + 3)^2 + (n + 1)^2 + (n - 1)^2 + (n - 3)^2)

theorem A_is_integer : ∃ (k : ℤ), A = k := by
  sorry

end NUMINAMATH_CALUDE_A_is_integer_l2709_270991


namespace NUMINAMATH_CALUDE_pentagonal_prism_coloring_l2709_270977

structure PentagonalPrism where
  vertices : Fin 10 → Point
  color : Fin 45 → Color

inductive Color
  | Red
  | Blue

def isEdge (i j : Fin 10) : Bool :=
  (i < j ∧ (i.val + 1 = j.val ∨ (i.val = 4 ∧ j.val = 0) ∨ (i.val = 9 ∧ j.val = 5))) ∨
  (j < i ∧ (j.val + 1 = i.val ∨ (j.val = 4 ∧ i.val = 0) ∨ (j.val = 9 ∧ i.val = 5)))

def isTopFaceEdge (i j : Fin 10) : Bool :=
  i < 5 ∧ j < 5 ∧ isEdge i j

def isBottomFaceEdge (i j : Fin 10) : Bool :=
  i ≥ 5 ∧ j ≥ 5 ∧ isEdge i j

def getEdgeColor (p : PentagonalPrism) (i j : Fin 10) : Color :=
  if i < j then p.color ⟨i.val * 9 + j.val - (i.val * (i.val + 1) / 2), sorry⟩
  else p.color ⟨j.val * 9 + i.val - (j.val * (j.val + 1) / 2), sorry⟩

def noMonochromaticTriangle (p : PentagonalPrism) : Prop :=
  ∀ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬(getEdgeColor p i j = getEdgeColor p j k ∧ getEdgeColor p j k = getEdgeColor p i k)

theorem pentagonal_prism_coloring (p : PentagonalPrism) 
  (h : noMonochromaticTriangle p) :
  (∀ i j : Fin 10, isTopFaceEdge i j → getEdgeColor p i j = getEdgeColor p 0 1) ∧
  (∀ i j : Fin 10, isBottomFaceEdge i j → getEdgeColor p i j = getEdgeColor p 5 6) :=
sorry

end NUMINAMATH_CALUDE_pentagonal_prism_coloring_l2709_270977


namespace NUMINAMATH_CALUDE_fraction_equality_l2709_270944

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x - 2 * y) / (2 * x + 5 * y) = 3) : 
  (2 * x - 5 * y) / (x + 2 * y) = 13 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2709_270944


namespace NUMINAMATH_CALUDE_differential_equation_holds_l2709_270986

open Real

noncomputable def y (x : ℝ) : ℝ := 1 / sqrt (sin x + x)

theorem differential_equation_holds (x : ℝ) (h : sin x + x > 0) :
  2 * sin x * (deriv y x) + y x * cos x = (y x)^3 * (x * cos x - sin x) := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_holds_l2709_270986


namespace NUMINAMATH_CALUDE_cookie_sharing_l2709_270990

theorem cookie_sharing (total_cookies : ℕ) (cookies_per_person : ℕ) (h1 : total_cookies = 24) (h2 : cookies_per_person = 4) :
  total_cookies / cookies_per_person = 6 :=
by sorry

end NUMINAMATH_CALUDE_cookie_sharing_l2709_270990


namespace NUMINAMATH_CALUDE_bacteria_growth_l2709_270913

/-- Bacteria growth problem -/
theorem bacteria_growth (initial_count : ℕ) (growth_factor : ℕ) (interval_count : ℕ) : 
  initial_count = 50 → 
  growth_factor = 3 → 
  interval_count = 5 → 
  initial_count * growth_factor ^ interval_count = 12150 := by
sorry

#eval 50 * 3 ^ 5  -- Expected output: 12150

end NUMINAMATH_CALUDE_bacteria_growth_l2709_270913


namespace NUMINAMATH_CALUDE_product_bounds_l2709_270992

theorem product_bounds (x₁ x₂ x₃ : ℝ) 
  (h_nonneg₁ : x₁ ≥ 0) (h_nonneg₂ : x₂ ≥ 0) (h_nonneg₃ : x₃ ≥ 0)
  (h_sum : x₁ + x₂ + x₃ = 1) :
  1 ≤ (x₁ + 3*x₂ + 5*x₃) * (x₁ + x₂/3 + x₃/5) ∧
  (x₁ + 3*x₂ + 5*x₃) * (x₁ + x₂/3 + x₃/5) ≤ 9/5 :=
by sorry

end NUMINAMATH_CALUDE_product_bounds_l2709_270992


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l2709_270967

theorem crazy_silly_school_series (total_movies : ℕ) (books_read : ℕ) (movies_watched : ℕ) (movies_to_watch : ℕ) :
  total_movies = 17 →
  books_read = 19 →
  movies_watched + movies_to_watch = total_movies →
  (∃ (different_books : ℕ), different_books = books_read) :=
by sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l2709_270967


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2709_270916

/-- Given a line segment with one endpoint at (7, 4) and midpoint at (5, -8),
    the sum of coordinates of the other endpoint is -17. -/
theorem midpoint_coordinate_sum :
  ∀ x y : ℝ,
  (5 : ℝ) = (7 + x) / 2 →
  (-8 : ℝ) = (4 + y) / 2 →
  x + y = -17 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2709_270916


namespace NUMINAMATH_CALUDE_aisha_has_largest_answer_l2709_270952

def starting_number : ℕ := 15

def maria_calculation (n : ℕ) : ℕ := ((n - 2) * 3) + 5

def liam_calculation (n : ℕ) : ℕ := (n * 3 - 2) + 5

def aisha_calculation (n : ℕ) : ℕ := ((n - 2) + 5) * 3

theorem aisha_has_largest_answer :
  aisha_calculation starting_number > maria_calculation starting_number ∧
  aisha_calculation starting_number > liam_calculation starting_number :=
sorry

end NUMINAMATH_CALUDE_aisha_has_largest_answer_l2709_270952


namespace NUMINAMATH_CALUDE_function_composition_equality_l2709_270908

theorem function_composition_equality (m n p q : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = m * x + n) 
  (hg : ∀ x, g x = p * x + q) : 
  (∃ x, f (g x) = g (f x)) ↔ n * (1 - p) = q * (1 - m) := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l2709_270908


namespace NUMINAMATH_CALUDE_coat_value_problem_l2709_270917

/-- Represents the problem of determining the value of a coat given to a worker --/
theorem coat_value_problem (total_pay : ℝ) (yearly_cash : ℝ) (months_worked : ℝ) 
  (partial_cash : ℝ) (h1 : total_pay = yearly_cash + coat_value) 
  (h2 : yearly_cash = 12) (h3 : months_worked = 7) (h4 : partial_cash = 5) :
  ∃ coat_value : ℝ, coat_value = 4.8 ∧ 
    (months_worked / 12) * total_pay = partial_cash + coat_value := by
  sorry


end NUMINAMATH_CALUDE_coat_value_problem_l2709_270917


namespace NUMINAMATH_CALUDE_toms_brick_cost_l2709_270983

/-- The total cost of bricks for Tom's shed -/
def total_cost (total_bricks : ℕ) (full_price : ℚ) (discount_percent : ℚ) : ℚ :=
  let half_bricks := total_bricks / 2
  let discounted_price := full_price * (1 - discount_percent)
  (half_bricks : ℚ) * discounted_price + (half_bricks : ℚ) * full_price

/-- Theorem stating the total cost for Tom's bricks -/
theorem toms_brick_cost :
  total_cost 1000 (1/2) (1/2) = 375 := by
  sorry

end NUMINAMATH_CALUDE_toms_brick_cost_l2709_270983


namespace NUMINAMATH_CALUDE_inequality_proofs_l2709_270950

theorem inequality_proofs (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l2709_270950


namespace NUMINAMATH_CALUDE_purchase_plans_theorem_l2709_270921

/-- Represents a purchasing plan for items A and B -/
structure PurchasePlan where
  a : ℕ  -- number of A items
  b : ℕ  -- number of B items

/-- Checks if a purchase plan satisfies all given conditions -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.a + p.b = 40 ∧
  p.a ≥ 3 * p.b ∧
  230 ≤ 8 * p.a + 2 * p.b ∧
  8 * p.a + 2 * p.b ≤ 266

/-- Calculates the total cost of a purchase plan -/
def totalCost (p : PurchasePlan) : ℕ :=
  8 * p.a + 2 * p.b

/-- Theorem stating the properties of valid purchase plans -/
theorem purchase_plans_theorem :
  ∃ (p1 p2 : PurchasePlan),
    isValidPlan p1 ∧
    isValidPlan p2 ∧
    p1 ≠ p2 ∧
    (∀ p, isValidPlan p → p = p1 ∨ p = p2) ∧
    (p1.a < p2.a → totalCost p1 < totalCost p2) :=
  sorry

end NUMINAMATH_CALUDE_purchase_plans_theorem_l2709_270921


namespace NUMINAMATH_CALUDE_number_puzzle_l2709_270979

theorem number_puzzle (x : ℝ) : 2 * x = 18 → x - 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2709_270979


namespace NUMINAMATH_CALUDE_x_sixth_geq_2a_minus_1_l2709_270924

theorem x_sixth_geq_2a_minus_1 (x a : ℝ) (h : x^5 - x^3 + x = a) : x^6 ≥ 2*a - 1 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_geq_2a_minus_1_l2709_270924


namespace NUMINAMATH_CALUDE_other_root_of_complex_equation_l2709_270922

theorem other_root_of_complex_equation (z : ℂ) :
  z^2 = -100 + 75*I ∧ z = 5 + 10*I → -5 - 10*I = -z :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_complex_equation_l2709_270922


namespace NUMINAMATH_CALUDE_line_equivalence_l2709_270914

/-- Given a line in the form (2, -1) · ((x, y) - (1, -3)) = 0, prove it's equivalent to y = 2x - 5 --/
theorem line_equivalence :
  ∀ (x y : ℝ), (2 : ℝ) * (x - 1) + (-1 : ℝ) * (y + 3) = 0 ↔ y = 2 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_line_equivalence_l2709_270914


namespace NUMINAMATH_CALUDE_school_basketballs_l2709_270954

/-- The number of classes that received basketballs -/
def num_classes : ℕ := 7

/-- The number of basketballs each class received -/
def basketballs_per_class : ℕ := 7

/-- The total number of basketballs bought by the school -/
def total_basketballs : ℕ := num_classes * basketballs_per_class

theorem school_basketballs : total_basketballs = 49 := by
  sorry

end NUMINAMATH_CALUDE_school_basketballs_l2709_270954


namespace NUMINAMATH_CALUDE_inequality_integer_solutions_l2709_270951

theorem inequality_integer_solutions :
  ∃! (s : Finset ℤ), (∀ x ∈ s, (x - 1 : ℚ) / 3 < 5 / 7 ∧ 5 / 7 < (x + 4 : ℚ) / 5) ∧
    s.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_integer_solutions_l2709_270951


namespace NUMINAMATH_CALUDE_jimmy_calorie_consumption_l2709_270931

def cracker_calories : ℕ := 15
def cookie_calories : ℕ := 50
def crackers_eaten : ℕ := 10
def cookies_eaten : ℕ := 7

theorem jimmy_calorie_consumption :
  cracker_calories * crackers_eaten + cookie_calories * cookies_eaten = 500 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_calorie_consumption_l2709_270931


namespace NUMINAMATH_CALUDE_angle_B_60_iff_arithmetic_progression_l2709_270945

theorem angle_B_60_iff_arithmetic_progression (A B C : ℝ) : 
  (A + B + C = 180) →  -- Sum of angles in a triangle is 180°
  (B = 60 ↔ ∃ d : ℝ, A = B - d ∧ C = B + d) :=
sorry

end NUMINAMATH_CALUDE_angle_B_60_iff_arithmetic_progression_l2709_270945


namespace NUMINAMATH_CALUDE_library_book_count_l2709_270960

/-- Given a library with shelves that each hold a fixed number of books,
    calculate the total number of books in the library. -/
def total_books (num_shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  num_shelves * books_per_shelf

/-- Theorem stating that a library with 1780 shelves, each holding 8 books,
    contains 14240 books in total. -/
theorem library_book_count : total_books 1780 8 = 14240 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l2709_270960


namespace NUMINAMATH_CALUDE_park_ticket_cost_l2709_270942

theorem park_ticket_cost (teacher_count student_count ticket_price total_budget : ℕ) :
  teacher_count = 3 →
  student_count = 9 →
  ticket_price = 22 →
  total_budget = 300 →
  (teacher_count + student_count) * ticket_price ≤ total_budget :=
by
  sorry

end NUMINAMATH_CALUDE_park_ticket_cost_l2709_270942


namespace NUMINAMATH_CALUDE_infinite_special_integers_l2709_270989

theorem infinite_special_integers : 
  ∃ f : ℕ → ℕ, Infinite {n : ℕ | ∃ m : ℕ, 
    n = m * (m + 1) + 2 ∧ 
    ∀ p : ℕ, Prime p → p ∣ (n^2 + 3) → 
      ∃ k : ℕ, k^2 < n ∧ p ∣ (k^2 + 3)} :=
sorry

end NUMINAMATH_CALUDE_infinite_special_integers_l2709_270989


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l2709_270988

theorem sum_reciprocals_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1/a + 1/b ≥ 4 ∧ ∀ M : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1/a + 1/b > M :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l2709_270988


namespace NUMINAMATH_CALUDE_n_value_for_specific_x_y_l2709_270940

theorem n_value_for_specific_x_y : ∀ (x y n : ℝ), 
  x = 3 → y = -1 → n = x - y^(x-y) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_n_value_for_specific_x_y_l2709_270940


namespace NUMINAMATH_CALUDE_nonzero_real_solution_l2709_270976

theorem nonzero_real_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 12) (h2 : y + 1 / x = 3 / 8) :
  x = 4 ∨ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_real_solution_l2709_270976


namespace NUMINAMATH_CALUDE_y_quadratic_iff_m_eq_2_y_linear_iff_m_special_l2709_270957

noncomputable def y (m : ℝ) (x : ℝ) : ℝ := (m + 3) * x^(m^2 + m - 4) + (m + 2) * x + 3

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

theorem y_quadratic_iff_m_eq_2 (m : ℝ) :
  is_quadratic (y m) ↔ m = 2 :=
sorry

theorem y_linear_iff_m_special (m : ℝ) :
  is_linear (y m) ↔ 
    m = -3 ∨ 
    m = (-1 + Real.sqrt 17) / 2 ∨ 
    m = (-1 - Real.sqrt 17) / 2 ∨
    m = (-1 + Real.sqrt 21) / 2 ∨
    m = (-1 - Real.sqrt 21) / 2 :=
sorry

end NUMINAMATH_CALUDE_y_quadratic_iff_m_eq_2_y_linear_iff_m_special_l2709_270957


namespace NUMINAMATH_CALUDE_odd_function_product_nonpositive_l2709_270919

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- For any odd function f: ℝ → ℝ, f(x)f(-x) ≤ 0 for all x ∈ ℝ -/
theorem odd_function_product_nonpositive (f : ℝ → ℝ) (h : IsOdd f) :
  ∀ x, f x * f (-x) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_product_nonpositive_l2709_270919


namespace NUMINAMATH_CALUDE_nail_triangle_impossibility_l2709_270906

/-- Given a triangle ACE on a wooden wall with nails of lengths AB = 1, CD = 2, EF = 4,
    prove that the distances between nail heads BD = √2, DF = √5, FB = √13 are impossible. -/
theorem nail_triangle_impossibility (AB CD EF BD DF FB : ℝ) :
  AB = 1 → CD = 2 → EF = 4 →
  BD = Real.sqrt 2 → DF = Real.sqrt 5 → FB = Real.sqrt 13 →
  ¬ (∃ (AC CE AE : ℝ), AC > 0 ∧ CE > 0 ∧ AE > 0 ∧
    AC + CE > AE ∧ CE + AE > AC ∧ AE + AC > CE) :=
by sorry

end NUMINAMATH_CALUDE_nail_triangle_impossibility_l2709_270906


namespace NUMINAMATH_CALUDE_drama_club_ratio_l2709_270965

theorem drama_club_ratio (girls boys : ℝ) (h : boys = 0.8 * girls) :
  girls = 1.25 * boys := by
  sorry

end NUMINAMATH_CALUDE_drama_club_ratio_l2709_270965


namespace NUMINAMATH_CALUDE_workshop_workers_l2709_270902

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := sorry

/-- Represents the average salary of all workers -/
def average_salary : ℚ := 9500

/-- Represents the number of technicians -/
def num_technicians : ℕ := 7

/-- Represents the average salary of technicians -/
def technician_salary : ℚ := 12000

/-- Represents the average salary of non-technicians -/
def non_technician_salary : ℚ := 6000

/-- Theorem stating that the total number of workers is 12 -/
theorem workshop_workers : total_workers = 12 := by sorry

end NUMINAMATH_CALUDE_workshop_workers_l2709_270902


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l2709_270936

theorem absolute_value_simplification : |(-4^2 + 6)| = 10 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l2709_270936


namespace NUMINAMATH_CALUDE_grade_assignment_count_l2709_270900

theorem grade_assignment_count :
  let num_students : ℕ := 12
  let num_grades : ℕ := 4
  num_grades ^ num_students = 16777216 :=
by sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l2709_270900


namespace NUMINAMATH_CALUDE_probiotic_diameter_scientific_notation_l2709_270928

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem probiotic_diameter_scientific_notation :
  toScientificNotation 0.00000002 = ScientificNotation.mk 2 (-8) (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_probiotic_diameter_scientific_notation_l2709_270928


namespace NUMINAMATH_CALUDE_sqrt_problem_l2709_270927

theorem sqrt_problem (h : Real.sqrt 100.4004 = 10.02) : Real.sqrt 1.004004 = 1.002 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_problem_l2709_270927


namespace NUMINAMATH_CALUDE_h_of_2_equals_2_l2709_270959

-- Define the function h
noncomputable def h : ℝ → ℝ := fun x => 
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) - 1) / (x^31 - 1)

-- Theorem statement
theorem h_of_2_equals_2 : h 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_h_of_2_equals_2_l2709_270959


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2709_270971

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 10 + (5 : ℚ) / 100 + (7 : ℚ) / 1000 + (1 : ℚ) / 1000 = (358 : ℚ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2709_270971


namespace NUMINAMATH_CALUDE_range_of_a_when_a_minus_3_in_M_range_of_a_when_interval_subset_M_l2709_270910

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x + a|
def g (x : ℝ) : ℝ := |x + 3| - x

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | f a x < g x}

-- Theorem 1
theorem range_of_a_when_a_minus_3_in_M (a : ℝ) :
  (a - 3) ∈ M a → 0 < a ∧ a < 3 := by sorry

-- Theorem 2
theorem range_of_a_when_interval_subset_M (a : ℝ) :
  Set.Icc (-1) 1 ⊆ M a → -2 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_when_a_minus_3_in_M_range_of_a_when_interval_subset_M_l2709_270910


namespace NUMINAMATH_CALUDE_box_volume_l2709_270943

/-- The volume of a box formed by cutting squares from corners of a square sheet -/
theorem box_volume (sheet_side : ℝ) (corner_cut : ℝ) : 
  sheet_side = 12 → corner_cut = 2 → 
  (sheet_side - 2 * corner_cut) * (sheet_side - 2 * corner_cut) * corner_cut = 128 :=
by
  sorry

end NUMINAMATH_CALUDE_box_volume_l2709_270943


namespace NUMINAMATH_CALUDE_expression_equality_l2709_270909

theorem expression_equality : 6 * 111 - 2 * 111 = 444 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2709_270909


namespace NUMINAMATH_CALUDE_select_student_count_l2709_270933

/-- The number of ways to select one student from a group of high school students -/
def select_student (first_year : ℕ) (second_year : ℕ) (third_year : ℕ) : ℕ :=
  first_year + second_year + third_year

/-- Theorem: Given the specified number of students in each year,
    the number of ways to select one student is 12 -/
theorem select_student_count :
  select_student 3 5 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_select_student_count_l2709_270933


namespace NUMINAMATH_CALUDE_water_problem_solution_l2709_270934

def water_problem (total_water : ℕ) (original_serving : ℕ) (serving_reduction : ℕ) : ℕ :=
  let original_servings := total_water / original_serving
  let new_servings := original_servings - serving_reduction
  total_water / new_servings

theorem water_problem_solution :
  water_problem 64 8 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_water_problem_solution_l2709_270934


namespace NUMINAMATH_CALUDE_larger_number_l2709_270975

theorem larger_number (a b : ℝ) (sum : a + b = 40) (diff : a - b = 10) : max a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l2709_270975


namespace NUMINAMATH_CALUDE_random_events_identification_l2709_270995

-- Define the types of events
inductive EventType
  | Random
  | Impossible
  | Certain

-- Define the events
structure Event :=
  (description : String)
  (eventType : EventType)

-- Define the function to check if an event is random
def isRandomEvent (e : Event) : Prop :=
  e.eventType = EventType.Random

-- Define the events
def coinEvent : Event :=
  { description := "Picking a 10 cent coin from a pocket with 50 cent, 10 cent, and 1 yuan coins"
  , eventType := EventType.Random }

def waterEvent : Event :=
  { description := "Water boiling at 90°C under standard atmospheric pressure"
  , eventType := EventType.Impossible }

def shooterEvent : Event :=
  { description := "A shooter hitting the 10-ring in one shot"
  , eventType := EventType.Random }

def diceEvent : Event :=
  { description := "Rolling two dice and the sum not exceeding 12"
  , eventType := EventType.Certain }

-- Theorem to prove
theorem random_events_identification :
  (isRandomEvent coinEvent ∧ isRandomEvent shooterEvent) ∧
  (¬isRandomEvent waterEvent ∧ ¬isRandomEvent diceEvent) :=
sorry

end NUMINAMATH_CALUDE_random_events_identification_l2709_270995
