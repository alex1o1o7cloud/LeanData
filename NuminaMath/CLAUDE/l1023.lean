import Mathlib

namespace NUMINAMATH_CALUDE_calculate_expression_l1023_102303

theorem calculate_expression : -2⁻¹ * (-8) + (2022)^0 - Real.sqrt 9 - abs (-4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1023_102303


namespace NUMINAMATH_CALUDE_omelette_combinations_l1023_102372

/-- The number of available fillings for omelettes -/
def num_fillings : ℕ := 8

/-- The number of egg choices for the omelette base -/
def num_egg_choices : ℕ := 4

/-- The total number of omelette combinations -/
def total_combinations : ℕ := 2^num_fillings * num_egg_choices

theorem omelette_combinations : total_combinations = 1024 := by
  sorry

end NUMINAMATH_CALUDE_omelette_combinations_l1023_102372


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1023_102311

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a2 : a 2 = 3) 
  (h_a7 : a 7 = 13) : 
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1023_102311


namespace NUMINAMATH_CALUDE_average_bacon_calculation_l1023_102336

-- Define the price per pound of bacon
def price_per_pound : ℝ := 6

-- Define the revenue from a half-size pig
def revenue_from_half_pig : ℝ := 60

-- Define the average amount of bacon from a pig
def average_bacon_amount : ℝ := 20

-- Theorem statement
theorem average_bacon_calculation :
  price_per_pound * (average_bacon_amount / 2) = revenue_from_half_pig :=
by sorry

end NUMINAMATH_CALUDE_average_bacon_calculation_l1023_102336


namespace NUMINAMATH_CALUDE_unique_special_sequence_l1023_102365

-- Define the sequence type
def SpecialSequence := ℕ → ℕ

-- Define the property of the sequence
def HasUniqueRepresentation (a : SpecialSequence) : Prop :=
  ∀ n : ℕ, ∃! (i j k : ℕ), n = a i + 2 * a j + 4 * a k

-- Define the strictly increasing property
def StrictlyIncreasing (a : SpecialSequence) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

-- Main theorem
theorem unique_special_sequence :
  ∃! a : SpecialSequence,
    StrictlyIncreasing a ∧
    HasUniqueRepresentation a ∧
    a 2002 = 1227132168 := by
  sorry


end NUMINAMATH_CALUDE_unique_special_sequence_l1023_102365


namespace NUMINAMATH_CALUDE_vector_equation_solution_parallel_vectors_solution_l1023_102366

/-- Given vectors in R^2 -/
def a : Fin 2 → ℚ := ![3, 2]
def b : Fin 2 → ℚ := ![-1, 2]
def c : Fin 2 → ℚ := ![4, 1]

/-- Part 1: Prove that m = 5/9 and n = 8/9 satisfy a = m*b + n*c -/
theorem vector_equation_solution :
  ∃ (m n : ℚ), (m = 5/9 ∧ n = 8/9) ∧ (∀ i : Fin 2, a i = m * b i + n * c i) :=
sorry

/-- Part 2: Prove that k = -16/13 makes (a + k*c) parallel to (2*b - a) -/
theorem parallel_vectors_solution :
  ∃ (k : ℚ), k = -16/13 ∧
  ∃ (t : ℚ), ∀ i : Fin 2, (a i + k * c i) = t * (2 * b i - a i) :=
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_parallel_vectors_solution_l1023_102366


namespace NUMINAMATH_CALUDE_special_triangle_properties_l1023_102340

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a = t.b * Real.cos t.C + (Real.sqrt 3 / 3) * t.c * Real.sin t.B ∧
  t.a = 2 ∧
  (1 / 2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.B = π / 3 ∧ t.b = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_properties_l1023_102340


namespace NUMINAMATH_CALUDE_motorcyclist_hiker_meeting_time_l1023_102395

/-- Calculates the waiting time for a motorcyclist and hiker to meet given their speeds and initial separation time. -/
theorem motorcyclist_hiker_meeting_time 
  (hiker_speed : ℝ) 
  (motorcyclist_speed : ℝ) 
  (separation_time : ℝ) 
  (hᵢ : hiker_speed = 6) 
  (mᵢ : motorcyclist_speed = 30) 
  (tᵢ : separation_time = 12 / 60) : 
  (motorcyclist_speed * separation_time) / hiker_speed = 1 := by
  sorry

#eval (60 : ℕ)  -- Expected result in minutes

end NUMINAMATH_CALUDE_motorcyclist_hiker_meeting_time_l1023_102395


namespace NUMINAMATH_CALUDE_additional_people_for_faster_mowing_l1023_102319

/-- Represents the number of people needed to mow a lawn in a given time -/
structure LawnMowing where
  people : ℕ
  hours : ℕ

/-- The work rate (people × hours) for mowing the lawn -/
def workRate (l : LawnMowing) : ℕ := l.people * l.hours

theorem additional_people_for_faster_mowing 
  (initial : LawnMowing) 
  (target : LawnMowing) 
  (h1 : initial.people = 4) 
  (h2 : initial.hours = 6) 
  (h3 : target.hours = 3) 
  (h4 : workRate initial = workRate target) : 
  target.people - initial.people = 4 := by
  sorry

end NUMINAMATH_CALUDE_additional_people_for_faster_mowing_l1023_102319


namespace NUMINAMATH_CALUDE_expression_value_at_three_l1023_102343

theorem expression_value_at_three :
  let f (x : ℝ) := (x^2 - 2*x - 8) / (x - 4)
  f 3 = 5 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l1023_102343


namespace NUMINAMATH_CALUDE_square_of_integer_l1023_102350

theorem square_of_integer (n : ℕ+) (h : ∃ (m : ℤ), m = 2 + 2 * Int.sqrt (28 * n.val^2 + 1)) :
  ∃ (k : ℤ), (2 + 2 * Int.sqrt (28 * n.val^2 + 1)) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_integer_l1023_102350


namespace NUMINAMATH_CALUDE_ant_position_100_l1023_102344

-- Define the ant's position after n steps
def ant_position (n : ℕ) : ℕ × ℕ :=
  (n * (n + 1) / 2, n * (n + 1) / 2)

-- Theorem statement
theorem ant_position_100 : ant_position 100 = (5050, 5050) := by
  sorry

end NUMINAMATH_CALUDE_ant_position_100_l1023_102344


namespace NUMINAMATH_CALUDE_condition_relationship_l1023_102305

theorem condition_relationship :
  (∀ x : ℝ, |x - 1| ≤ 1 → 2 - x ≥ 0) ∧
  (∃ x : ℝ, 2 - x ≥ 0 ∧ |x - 1| > 1) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1023_102305


namespace NUMINAMATH_CALUDE_two_integer_segments_l1023_102357

/-- A right triangle with integer leg lengths 24 and 25 -/
structure RightTriangle where
  /-- Length of the first leg -/
  de : ℕ
  /-- Length of the second leg -/
  ef : ℕ
  /-- Assumption that the first leg has length 24 -/
  de_eq : de = 24
  /-- Assumption that the second leg has length 25 -/
  ef_eq : ef = 25

/-- The number of integer-length line segments from vertex E to hypotenuse DF -/
def count_integer_segments (t : RightTriangle) : ℕ :=
  2

/-- Theorem stating that there are exactly 2 integer-length line segments 
    from vertex E to hypotenuse DF in the given right triangle -/
theorem two_integer_segments (t : RightTriangle) :
  count_integer_segments t = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_integer_segments_l1023_102357


namespace NUMINAMATH_CALUDE_inequality_theorem_l1023_102304

theorem inequality_theorem (p q r : ℝ) : 
  (∀ x : ℝ, (x - p) * (x - q) / (x - r) ≥ 0 ↔ x < -6 ∨ |x - 20| ≤ 2) →
  p < q →
  p + 2*q + 3*r = 44 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1023_102304


namespace NUMINAMATH_CALUDE_cone_base_radius_l1023_102389

/-- Given a cone whose lateral surface unfolds into a semicircle with radius 4,
    prove that the radius of the base of the cone is also 4. -/
theorem cone_base_radius (r : ℝ) (h : r = 4) : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1023_102389


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l1023_102379

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (hx : (x + 2) % 7 = 0) 
  (hy : (y - 2) % 7 = 0) : 
  ∃ n : ℕ+, (∀ m : ℕ+, (x^2 + x*y + y^2 + m) % 7 = 0 → n ≤ m) ∧ 
            (x^2 + x*y + y^2 + n) % 7 = 0 ∧ 
            n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l1023_102379


namespace NUMINAMATH_CALUDE_marching_band_weight_is_245_l1023_102354

/-- Represents the total weight carried by the Oprah Winfrey High School marching band. -/
def marching_band_weight : ℕ :=
  let trumpet_clarinet_weight := 5
  let trombone_weight := 10
  let tuba_weight := 20
  let drum_weight := 15
  let trumpet_count := 6
  let clarinet_count := 9
  let trombone_count := 8
  let tuba_count := 3
  let drum_count := 2
  (trumpet_clarinet_weight * (trumpet_count + clarinet_count)) +
  (trombone_weight * trombone_count) +
  (tuba_weight * tuba_count) +
  (drum_weight * drum_count)

/-- Theorem stating that the total weight carried by the marching band is 245 pounds. -/
theorem marching_band_weight_is_245 : marching_band_weight = 245 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_weight_is_245_l1023_102354


namespace NUMINAMATH_CALUDE_trig_inequality_l1023_102399

open Real

theorem trig_inequality (a b c : ℝ) : 
  a = sin (2 * π / 7) → 
  b = cos (12 * π / 7) → 
  c = tan (9 * π / 7) → 
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_trig_inequality_l1023_102399


namespace NUMINAMATH_CALUDE_characterize_nonnegative_quadratic_function_l1023_102301

/-- A function f: ℝ → ℝ satisfying the given conditions -/
def NonNegativeQuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≥ 0) ∧ 
  (∀ x y, f (x + y) + f (x - y) - 2 * f x - 2 * y^2 = 0)

/-- The theorem stating the form of f -/
theorem characterize_nonnegative_quadratic_function 
  (f : ℝ → ℝ) (h : NonNegativeQuadraticFunction f) : 
  ∃ a c : ℝ, (∀ x, f x = x^2 + a*x + c) ∧ a^2 - 4*c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_characterize_nonnegative_quadratic_function_l1023_102301


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1023_102368

theorem complex_number_in_third_quadrant : 
  let z : ℂ := Complex.I * (-1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1023_102368


namespace NUMINAMATH_CALUDE_school_ratio_proof_l1023_102376

/-- Represents the ratio of two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Simplifies a ratio by dividing both numerator and denominator by their GCD -/
def simplifyRatio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd, denominator := r.denominator / gcd }

theorem school_ratio_proof (total_students : ℕ) (num_girls : ℕ) 
    (h1 : total_students = 300) (h2 : num_girls = 160) : 
    simplifyRatio { numerator := num_girls, denominator := total_students - num_girls } = 
    { numerator := 8, denominator := 7 } := by
  sorry

#check school_ratio_proof

end NUMINAMATH_CALUDE_school_ratio_proof_l1023_102376


namespace NUMINAMATH_CALUDE_taller_tree_height_l1023_102314

/-- Given two trees with specific height relationships, prove the height of the taller tree -/
theorem taller_tree_height (h_shorter h_taller : ℝ) : 
  h_taller = h_shorter + 20 →  -- The top of one tree is 20 feet higher
  h_shorter / h_taller = 2 / 3 →  -- The heights are in the ratio 2:3
  h_shorter = 40 →  -- The shorter tree is 40 feet tall
  h_taller = 60 := by sorry

end NUMINAMATH_CALUDE_taller_tree_height_l1023_102314


namespace NUMINAMATH_CALUDE_freshman_class_size_l1023_102300

theorem freshman_class_size :
  ∃! n : ℕ, n < 500 ∧ n % 19 = 17 ∧ n % 18 = 9 ∧ n = 207 := by
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l1023_102300


namespace NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l1023_102371

/-- The area of a triangle with two sides of length 3 and 5, where the cosine of the angle between them is a root of 5x^2 - 7x - 6 = 0, is equal to 6. -/
theorem triangle_area_with_cosine_root : ∃ (θ : ℝ), 
  (5 * (Real.cos θ)^2 - 7 * (Real.cos θ) - 6 = 0) →
  (1/2 * 3 * 5 * Real.sin θ = 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l1023_102371


namespace NUMINAMATH_CALUDE_two_points_same_color_distance_l1023_102342

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Theorem statement
theorem two_points_same_color_distance (x : ℝ) (h : x > 0) (coloring : Coloring) :
  ∃ (p q : Point) (c : Color), coloring p = c ∧ coloring q = c ∧ 
    Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2) = x := by
  sorry

end NUMINAMATH_CALUDE_two_points_same_color_distance_l1023_102342


namespace NUMINAMATH_CALUDE_odd_function_condition_l1023_102356

def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -(f a b x)) ↔ a = 0 ∧ b = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_condition_l1023_102356


namespace NUMINAMATH_CALUDE_travelers_getting_off_subway_l1023_102328

/-- The number of stations ahead -/
def num_stations : ℕ := 10

/-- The number of travelers -/
def num_travelers : ℕ := 3

/-- The total number of ways travelers can get off at any station -/
def total_ways : ℕ := num_stations ^ num_travelers

/-- The number of ways all travelers can get off at the same station -/
def same_station_ways : ℕ := num_stations

/-- The number of ways travelers can get off without all disembarking at the same station -/
def different_station_ways : ℕ := total_ways - same_station_ways

theorem travelers_getting_off_subway :
  different_station_ways = 990 := by sorry

end NUMINAMATH_CALUDE_travelers_getting_off_subway_l1023_102328


namespace NUMINAMATH_CALUDE_function_comparison_l1023_102396

/-- A function f is strictly decreasing on the non-negative real numbers -/
def StrictlyDecreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₂ < f x₁

/-- An even function -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem function_comparison 
  (f : ℝ → ℝ) 
  (h_even : EvenFunction f)
  (h_decreasing : StrictlyDecreasingOnNonnegative f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 :=
sorry

end NUMINAMATH_CALUDE_function_comparison_l1023_102396


namespace NUMINAMATH_CALUDE_function_properties_l1023_102317

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - 5 * x^2 - b * x

-- Define the derivative of f
def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 - 10 * x - b

theorem function_properties (a b : ℝ) :
  (f_derivative a b 3 = 0) →  -- x = 3 is an extreme point
  (f a b 1 = -1) →           -- f(1) = -1
  (a = 1 ∧ b = -3) ∧         -- Part 1: a = 1 and b = -3
  (∀ x ∈ Set.Icc 2 4, f 1 (-3) x ≥ -9) ∧  -- Part 2: Minimum value on [2, 4] is -9
  (∀ x ∈ Set.Icc 2 4, f 1 (-3) x ≤ 0) ∧   -- Part 3: Maximum value on [2, 4] is 0
  (f 1 (-3) 3 = -9) ∧        -- Minimum occurs at x = 3
  (f 1 (-3) 4 = 0)           -- Maximum occurs at x = 4
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l1023_102317


namespace NUMINAMATH_CALUDE_angle_FAG_measure_l1023_102330

-- Define the structure of the geometric configuration
structure GeometricConfiguration where
  -- Triangle ABC is equilateral
  triangle_ABC_equilateral : Bool
  -- BCDFG is a regular pentagon
  pentagon_BCDFG_regular : Bool
  -- Triangle ABC and pentagon BCDFG share side BC
  shared_side_BC : Bool

-- Define the theorem
theorem angle_FAG_measure (config : GeometricConfiguration) 
  (h1 : config.triangle_ABC_equilateral = true)
  (h2 : config.pentagon_BCDFG_regular = true)
  (h3 : config.shared_side_BC = true) :
  ∃ (angle_FAG : ℝ), angle_FAG = 36 := by
  sorry

end NUMINAMATH_CALUDE_angle_FAG_measure_l1023_102330


namespace NUMINAMATH_CALUDE_sum_of_sequences_l1023_102326

-- Define the arithmetic sequence
def arithmetic_sequence (a b : ℝ) : Prop :=
  ∃ d : ℝ, a = 6 + d ∧ b = 6 + 2*d ∧ 48 = 6 + 3*d

-- Define the geometric sequence
def geometric_sequence (c d : ℝ) : Prop :=
  ∃ r : ℝ, c = 6*r ∧ d = 6*r^2 ∧ 48 = 6*r^3

theorem sum_of_sequences (a b c d : ℝ) 
  (h1 : arithmetic_sequence a b) 
  (h2 : geometric_sequence c d) : 
  a + b + c + d = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l1023_102326


namespace NUMINAMATH_CALUDE_snow_leopard_arrangements_l1023_102308

theorem snow_leopard_arrangements (n : ℕ) (h : n = 8) :
  2 * Nat.factorial (n - 2) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangements_l1023_102308


namespace NUMINAMATH_CALUDE_double_burger_cost_l1023_102385

/-- Calculates the cost of a double burger given the total spent, total number of hamburgers,
    cost of a single burger, and number of double burgers. -/
def cost_of_double_burger (total_spent : ℚ) (total_burgers : ℕ) (single_burger_cost : ℚ) (double_burgers : ℕ) : ℚ :=
  let single_burgers := total_burgers - double_burgers
  let total_single_cost := single_burgers * single_burger_cost
  let total_double_cost := total_spent - total_single_cost
  total_double_cost / double_burgers

/-- Theorem stating that the cost of a double burger is $1.50 given the specific conditions. -/
theorem double_burger_cost :
  cost_of_double_burger 64.5 50 1 29 = 1.5 := by
  sorry

#eval cost_of_double_burger 64.5 50 1 29

end NUMINAMATH_CALUDE_double_burger_cost_l1023_102385


namespace NUMINAMATH_CALUDE_blue_area_ratio_l1023_102373

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  /-- Side length of the flag -/
  side : ℝ
  /-- Width of the cross -/
  cross_width : ℝ
  /-- Assumption that the cross (including blue center) is 36% of total area -/
  cross_area_ratio : cross_width * (4 * side - cross_width) / (side * side) = 0.36

/-- Theorem stating that the blue area is 2% of the total flag area -/
theorem blue_area_ratio (flag : SquareFlag) : 
  (flag.cross_width / flag.side) ^ 2 = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_blue_area_ratio_l1023_102373


namespace NUMINAMATH_CALUDE_smallest_factor_of_36_l1023_102374

theorem smallest_factor_of_36 (a b c : ℤ) 
  (h1 : a * b * c = 36)
  (h2 : a + b + c = 4) :
  min a (min b c) = -4 :=
sorry

end NUMINAMATH_CALUDE_smallest_factor_of_36_l1023_102374


namespace NUMINAMATH_CALUDE_escalator_steps_l1023_102338

theorem escalator_steps :
  ∀ (n : ℕ) (k : ℚ),
    k > 0 →
    (18 / k) * (k + 1) = n →
    (27 / (2 * k)) * (2 * k + 1) = n →
    n = 54 := by
  sorry

end NUMINAMATH_CALUDE_escalator_steps_l1023_102338


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1023_102390

theorem complex_equation_solution (z : ℂ) :
  (1 + 2 * z) / (1 - z) = Complex.I → z = -1/5 + 3/5 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1023_102390


namespace NUMINAMATH_CALUDE_max_eggs_per_basket_l1023_102392

/-- The number of yellow Easter eggs -/
def yellow_eggs : ℕ := 16

/-- The number of green Easter eggs -/
def green_eggs : ℕ := 28

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 4

theorem max_eggs_per_basket :
  eggs_per_basket = 4 ∧
  yellow_eggs % eggs_per_basket = 0 ∧
  green_eggs % eggs_per_basket = 0 ∧
  eggs_per_basket ≥ 2 ∧
  ∀ n : ℕ, n > eggs_per_basket →
    (yellow_eggs % n ≠ 0 ∨ green_eggs % n ≠ 0 ∨ n < 2) :=
by sorry

end NUMINAMATH_CALUDE_max_eggs_per_basket_l1023_102392


namespace NUMINAMATH_CALUDE_coordinates_of_point_E_l1023_102383

/-- Given points A, B, C, D, and E in the plane, where D lies on line AB and E is on the extension of DC,
    prove that E has specific coordinates. -/
theorem coordinates_of_point_E (A B C D E : ℝ × ℝ) : 
  A = (-2, 1) →
  B = (1, 4) →
  C = (4, -3) →
  (∃ t : ℝ, D = (1 - t) • A + t • B ∧ t = 2/3) →
  (∃ s : ℝ, E = (1 + s) • D - s • C ∧ s = 5) →
  E = (-8/3, 11/3) := by
sorry

end NUMINAMATH_CALUDE_coordinates_of_point_E_l1023_102383


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1023_102335

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1023_102335


namespace NUMINAMATH_CALUDE_inequality_solution_l1023_102323

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) > 4 / x + 23 / 10) ↔ (x > -2 ∧ x < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1023_102323


namespace NUMINAMATH_CALUDE_weight_of_compound_approx_l1023_102380

/-- The atomic mass of carbon in g/mol -/
def carbon_mass : ℝ := 12.01

/-- The atomic mass of hydrogen in g/mol -/
def hydrogen_mass : ℝ := 1.008

/-- The atomic mass of oxygen in g/mol -/
def oxygen_mass : ℝ := 16.00

/-- The chemical formula of the compound -/
structure ChemicalFormula where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- The compound C6H8O7 -/
def compound : ChemicalFormula := ⟨6, 8, 7⟩

/-- Calculate the molar mass of a chemical formula -/
def molar_mass (formula : ChemicalFormula) : ℝ :=
  formula.carbon * carbon_mass + 
  formula.hydrogen * hydrogen_mass + 
  formula.oxygen * oxygen_mass

/-- The number of moles -/
def num_moles : ℝ := 3

/-- The total weight in grams -/
def total_weight : ℝ := 576

/-- Theorem stating that the weight of 3 moles of C6H8O7 is approximately 576 grams -/
theorem weight_of_compound_approx (ε : ℝ) (ε_pos : ε > 0) : 
  |num_moles * molar_mass compound - total_weight| < ε := by
  sorry

end NUMINAMATH_CALUDE_weight_of_compound_approx_l1023_102380


namespace NUMINAMATH_CALUDE_john_apple_earnings_l1023_102375

/-- Calculates the total money earned from selling apples given the plot dimensions, yield per tree, and price per apple. -/
def totalMoneyEarned (rows : ℕ) (columns : ℕ) (applesPerTree : ℕ) (pricePerApple : ℚ) : ℚ :=
  (rows * columns * applesPerTree) * pricePerApple

/-- Proves that given the specified conditions, John earns $30 from selling apples. -/
theorem john_apple_earnings :
  totalMoneyEarned 3 4 5 (1/2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_john_apple_earnings_l1023_102375


namespace NUMINAMATH_CALUDE_jason_games_last_month_l1023_102394

/-- Calculates the number of games Jason planned to attend last month -/
def games_planned_last_month (games_this_month games_missed games_attended : ℕ) : ℕ :=
  (games_attended + games_missed) - games_this_month

theorem jason_games_last_month :
  games_planned_last_month 11 16 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_jason_games_last_month_l1023_102394


namespace NUMINAMATH_CALUDE_triangle_area_l1023_102361

theorem triangle_area (a b : ℝ) (A B : ℝ) (h1 : a = 5) (h2 : b = 4) (h3 : Real.cos (A - B) = 31/32) :
  ∃ (C : ℝ), (1/2) * a * b * Real.sin C = (15 * Real.sqrt 7) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1023_102361


namespace NUMINAMATH_CALUDE_count_non_negative_numbers_l1023_102307

theorem count_non_negative_numbers : 
  let numbers : List ℝ := [-8, 2.1, 1/9, 3, 0, -2.5, 10, -1]
  (numbers.filter (λ x => x ≥ 0)).length = 5 := by
sorry

end NUMINAMATH_CALUDE_count_non_negative_numbers_l1023_102307


namespace NUMINAMATH_CALUDE_solution_set_l1023_102353

theorem solution_set (x : ℝ) :
  (x - 2) / (x - 4) ≥ 3 ∧ x ≠ 2 → 4 < x ∧ x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l1023_102353


namespace NUMINAMATH_CALUDE_salem_poem_words_l1023_102332

/-- Calculate the total number of words in a poem given the number of stanzas, lines per stanza, and words per line -/
def totalWords (stanzas : ℕ) (linesPerStanza : ℕ) (wordsPerLine : ℕ) : ℕ :=
  stanzas * linesPerStanza * wordsPerLine

/-- Theorem: The total number of words in Salem's poem is 1600 -/
theorem salem_poem_words :
  totalWords 20 10 8 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_salem_poem_words_l1023_102332


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1023_102321

/-- The area of an equilateral triangle with perimeter 3p is (√3/4) * p^2. -/
theorem equilateral_triangle_area (p : ℝ) (p_pos : p > 0) : 
  let perimeter := 3 * p
  let side_length := perimeter / 3
  let area := (Real.sqrt 3 / 4) * side_length^2
  area = (Real.sqrt 3 / 4) * p^2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1023_102321


namespace NUMINAMATH_CALUDE_solve_equation_l1023_102363

theorem solve_equation : ∀ x : ℝ, 3 * x + 4 = x + 2 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1023_102363


namespace NUMINAMATH_CALUDE_concatenated_numbers_divisible_by_45_l1023_102378

def concatenate_numbers (n : ℕ) : ℕ :=
  -- Definition of concatenating numbers from 1 to n
  sorry

theorem concatenated_numbers_divisible_by_45 :
  ∃ k : ℕ, concatenate_numbers 50 = 45 * k := by
  sorry

end NUMINAMATH_CALUDE_concatenated_numbers_divisible_by_45_l1023_102378


namespace NUMINAMATH_CALUDE_sequence_monotonicity_l1023_102388

/-- A sequence a_n is defined as a_n = -n^2 + tn, where n is a positive natural number and t is a constant real number. The sequence is monotonically decreasing. -/
theorem sequence_monotonicity (t : ℝ) : 
  (∀ n : ℕ+, ∀ m : ℕ+, n < m → (-n^2 + t * n) > (-m^2 + t * m)) → 
  t < 3 := by
sorry

end NUMINAMATH_CALUDE_sequence_monotonicity_l1023_102388


namespace NUMINAMATH_CALUDE_scooter_cost_calculation_l1023_102358

theorem scooter_cost_calculation (original_cost : ℝ) 
  (repair1_percent repair2_percent repair3_percent tax_percent : ℝ)
  (discount_percent profit_percent : ℝ) (profit : ℝ) :
  repair1_percent = 0.05 →
  repair2_percent = 0.10 →
  repair3_percent = 0.07 →
  tax_percent = 0.12 →
  discount_percent = 0.15 →
  profit_percent = 0.30 →
  profit = 2000 →
  profit = profit_percent * original_cost →
  let total_spent := original_cost * (1 + repair1_percent + repair2_percent + repair3_percent + tax_percent)
  total_spent = 1.34 * original_cost :=
by sorry

end NUMINAMATH_CALUDE_scooter_cost_calculation_l1023_102358


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1023_102333

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1023_102333


namespace NUMINAMATH_CALUDE_stating_intersection_points_count_l1023_102327

/-- 
Given a positive integer n ≥ 5 and n lines in a plane where:
- Exactly 3 lines are mutually parallel
- Any two lines that are not part of the 3 parallel lines are not parallel
- Any three lines do not intersect at a single point
This function calculates the number of intersection points.
-/
def intersectionPoints (n : ℕ) : ℕ :=
  (n^2 - n - 6) / 2

/-- 
Theorem stating that for n ≥ 5 lines in a plane satisfying the given conditions,
the number of intersection points is (n^2 - n - 6) / 2.
-/
theorem intersection_points_count (n : ℕ) (h : n ≥ 5) :
  let lines := n
  let parallel_lines := 3
  intersectionPoints n = (n^2 - n - 6) / 2 := by
  sorry

#eval intersectionPoints 5  -- Expected output: 7
#eval intersectionPoints 6  -- Expected output: 12
#eval intersectionPoints 10 -- Expected output: 42

end NUMINAMATH_CALUDE_stating_intersection_points_count_l1023_102327


namespace NUMINAMATH_CALUDE_new_person_weight_l1023_102397

/-- Proves that the weight of a new person is 77 kg when they replace a 65 kg person in a group of 8,
    causing the average weight to increase by 1.5 kg -/
theorem new_person_weight (n : ℕ) (old_weight new_weight avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : old_weight = 65)
  (h3 : avg_increase = 1.5) :
  new_weight = old_weight + n * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1023_102397


namespace NUMINAMATH_CALUDE_smallest_value_w_z_cubes_l1023_102322

theorem smallest_value_w_z_cubes (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 1)
  (h2 : Complex.abs (w^2 + z^2) = 14) :
  Complex.abs (w^3 + z^3) ≥ 41/2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_w_z_cubes_l1023_102322


namespace NUMINAMATH_CALUDE_equal_selection_probabilities_l1023_102367

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Calculate the probability of selection for a given sampling method -/
def probability_of_selection (method : SamplingMethod) (population_size : ℕ) (sample_size : ℕ) : ℚ :=
  match method with
  | SamplingMethod.SimpleRandom => sample_size / population_size
  | SamplingMethod.Systematic => sample_size / population_size
  | SamplingMethod.Stratified => sample_size / population_size

theorem equal_selection_probabilities (population_size : ℕ) (sample_size : ℕ)
    (h1 : population_size = 50)
    (h2 : sample_size = 10) :
    (probability_of_selection SamplingMethod.SimpleRandom population_size sample_size =
     probability_of_selection SamplingMethod.Systematic population_size sample_size) ∧
    (probability_of_selection SamplingMethod.SimpleRandom population_size sample_size =
     probability_of_selection SamplingMethod.Stratified population_size sample_size) ∧
    (probability_of_selection SamplingMethod.SimpleRandom population_size sample_size = 1/5) :=
  sorry

end NUMINAMATH_CALUDE_equal_selection_probabilities_l1023_102367


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_28_l1023_102393

theorem arithmetic_expression_equals_28 : 12 / 4 - 3 - 10 + 3 * 10 + 2^3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_28_l1023_102393


namespace NUMINAMATH_CALUDE_new_average_production_l1023_102398

def past_days : ℕ := 14
def past_average : ℝ := 60
def today_production : ℝ := 90

theorem new_average_production :
  let total_past_production : ℝ := past_days * past_average
  let total_production : ℝ := total_past_production + today_production
  let new_average : ℝ := total_production / (past_days + 1)
  new_average = 62 := by sorry

end NUMINAMATH_CALUDE_new_average_production_l1023_102398


namespace NUMINAMATH_CALUDE_marble_weight_proof_l1023_102347

/-- The weight of each of the two identical pieces of marble -/
def x : ℝ := 0.335

/-- The weight of the third piece of marble -/
def third_piece_weight : ℝ := 0.08

/-- The total weight of all three pieces of marble -/
def total_weight : ℝ := 0.75

theorem marble_weight_proof :
  2 * x + third_piece_weight = total_weight :=
by sorry

end NUMINAMATH_CALUDE_marble_weight_proof_l1023_102347


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_necessary_not_sufficient_l1023_102316

-- Define the propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - 5*x + 6 < 0

-- Theorem 1: Range of x when a = 1
theorem range_of_x_when_a_is_one :
  ∃ (lower upper : ℝ), lower = 2 ∧ upper = 3 ∧
  ∀ x, p x 1 ∧ q x ↔ lower < x ∧ x < upper :=
sorry

-- Theorem 2: Range of a when p is necessary but not sufficient for q
theorem range_of_a_necessary_not_sufficient :
  ∃ (lower upper : ℝ), lower = 1 ∧ upper = 2 ∧
  ∀ a, (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x) ↔ lower ≤ a ∧ a ≤ upper :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_necessary_not_sufficient_l1023_102316


namespace NUMINAMATH_CALUDE_inequality_proof_l1023_102313

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1023_102313


namespace NUMINAMATH_CALUDE_zoo_trip_average_bus_capacity_l1023_102337

theorem zoo_trip_average_bus_capacity (total_students : ℕ) (num_buses : ℕ) 
  (car1_capacity car2_capacity car3_capacity car4_capacity : ℕ) : 
  total_students = 396 →
  num_buses = 7 →
  car1_capacity = 5 →
  car2_capacity = 4 →
  car3_capacity = 3 →
  car4_capacity = 6 →
  (total_students - (car1_capacity + car2_capacity + car3_capacity + car4_capacity)) / num_buses = 54 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_average_bus_capacity_l1023_102337


namespace NUMINAMATH_CALUDE_election_votes_total_l1023_102341

theorem election_votes_total (votes_A : ℝ) (votes_B : ℝ) (votes_C : ℝ) (votes_D : ℝ) 
  (total_votes : ℝ) :
  votes_A = 0.45 * total_votes →
  votes_B = 0.25 * total_votes →
  votes_C = 0.15 * total_votes →
  votes_D = total_votes - (votes_A + votes_B + votes_C) →
  votes_A - votes_B = 800 →
  total_votes = 4000 := by
  sorry

#check election_votes_total

end NUMINAMATH_CALUDE_election_votes_total_l1023_102341


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1023_102387

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 3, 5}
def Q : Set Nat := {1, 2, 4}

theorem complement_union_theorem :
  (U \ P) ∪ Q = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1023_102387


namespace NUMINAMATH_CALUDE_basketball_team_score_l1023_102359

theorem basketball_team_score (two_pointers three_pointers free_throws : ℕ) : 
  (3 * three_pointers = 2 * (2 * two_pointers)) →
  (free_throws = 2 * two_pointers) →
  (2 * two_pointers + 3 * three_pointers + free_throws = 65) →
  free_throws = 18 := by
sorry

end NUMINAMATH_CALUDE_basketball_team_score_l1023_102359


namespace NUMINAMATH_CALUDE_largest_n_unique_k_l1023_102339

theorem largest_n_unique_k : ∃ (n : ℕ), n > 0 ∧ n = 63 ∧
  (∃! (k : ℤ), (9 : ℚ)/17 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/15) ∧
  (∀ (m : ℕ), m > n → ¬(∃! (k : ℤ), (9 : ℚ)/17 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 8/15)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_unique_k_l1023_102339


namespace NUMINAMATH_CALUDE_adam_laundry_theorem_l1023_102364

/-- Given a total number of laundry loads and the number of loads already washed,
    calculate the number of loads still to be washed. -/
def loads_remaining (total : ℕ) (washed : ℕ) : ℕ :=
  total - washed

/-- Theorem: Given 25 total loads and 6 washed loads, 19 loads remain to be washed. -/
theorem adam_laundry_theorem :
  loads_remaining 25 6 = 19 := by
  sorry

end NUMINAMATH_CALUDE_adam_laundry_theorem_l1023_102364


namespace NUMINAMATH_CALUDE_gmat_question_percentages_l1023_102331

/-- Given percentages of test takers answering questions correctly or incorrectly, 
    prove the percentage that answered both questions correctly. -/
theorem gmat_question_percentages 
  (first_correct : ℝ) 
  (second_correct : ℝ) 
  (neither_correct : ℝ) 
  (h1 : first_correct = 85) 
  (h2 : second_correct = 70) 
  (h3 : neither_correct = 5) : 
  first_correct + second_correct - (100 - neither_correct) = 60 := by
  sorry


end NUMINAMATH_CALUDE_gmat_question_percentages_l1023_102331


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l1023_102391

theorem arithmetic_geometric_mean_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_mean : (x + y) / 2 = 3 * Real.sqrt (x * y)) : 
  ∃ (n : ℤ), ∀ (m : ℤ), |x / y - n| ≤ |x / y - m| ∧ n = 34 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l1023_102391


namespace NUMINAMATH_CALUDE_expression_two_values_l1023_102386

theorem expression_two_values (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (x y : ℝ), x ≠ y ∧
    ∀ (z : ℝ), z = a / abs a + b / abs b + (a * b) / abs (a * b) → z = x ∨ z = y :=
by sorry

end NUMINAMATH_CALUDE_expression_two_values_l1023_102386


namespace NUMINAMATH_CALUDE_deck_size_l1023_102377

theorem deck_size (r b : ℕ) : 
  r > 0 → b > 0 →
  r / (r + b) = 1 / 4 →
  r / (r + b + 6) = 1 / 6 →
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l1023_102377


namespace NUMINAMATH_CALUDE_smallest_m_divisibility_l1023_102309

theorem smallest_m_divisibility : ∃! m : ℕ,
  (∀ n : ℕ, Odd n → (148^n + m * 141^n) % 2023 = 0) ∧
  (∀ k : ℕ, k < m → ∃ n : ℕ, Odd n ∧ (148^n + k * 141^n) % 2023 ≠ 0) ∧
  m = 1735 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_divisibility_l1023_102309


namespace NUMINAMATH_CALUDE_unread_messages_proof_l1023_102346

/-- The number of days it takes to read all messages -/
def days : ℕ := 7

/-- The number of messages read per day -/
def messages_read_per_day : ℕ := 20

/-- The number of new messages received per day -/
def new_messages_per_day : ℕ := 6

/-- The initial number of unread messages -/
def initial_messages : ℕ := days * (messages_read_per_day - new_messages_per_day)

theorem unread_messages_proof :
  initial_messages = 98 := by sorry

end NUMINAMATH_CALUDE_unread_messages_proof_l1023_102346


namespace NUMINAMATH_CALUDE_sqrt_511100_approx_l1023_102325

-- Define the approximation relation
def approx (x y : ℝ) : Prop := ∃ ε > 0, |x - y| < ε

-- State the theorem
theorem sqrt_511100_approx :
  approx (Real.sqrt 51.11) 7.149 →
  approx (Real.sqrt 511100) 714.9 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_511100_approx_l1023_102325


namespace NUMINAMATH_CALUDE_soccer_team_lineup_count_l1023_102320

theorem soccer_team_lineup_count :
  let total_players : ℕ := 15
  let roles : ℕ := 7
  total_players.factorial / (total_players - roles).factorial = 2541600 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_lineup_count_l1023_102320


namespace NUMINAMATH_CALUDE_problem_solution_l1023_102315

theorem problem_solution (a b : ℚ) 
  (h1 : 5 + a = 7 - b) 
  (h2 : 7 + b = 12 + a) : 
  5 - a = 13/2 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1023_102315


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l1023_102302

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 40 ∧ wrong_value = 20 ∧ correct_value = 34 ∧ corrected_mean = 36.45 →
  ∃ initial_mean : ℝ, initial_mean = 36.1 ∧
    n * corrected_mean = n * initial_mean + (correct_value - wrong_value) :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l1023_102302


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l1023_102370

def double_factorial (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

def braced_notation (x : ℕ) : ℕ := double_factorial x

theorem greatest_prime_factor_of_sum (n : ℕ) (h : n = 22) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (braced_notation n + braced_notation (n - 2)) ∧
    ∀ q : ℕ, Nat.Prime q → q ∣ (braced_notation n + braced_notation (n - 2)) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l1023_102370


namespace NUMINAMATH_CALUDE_event_probability_l1023_102382

theorem event_probability (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^4 = 65/81) →
  p = 1/3 := by
sorry

end NUMINAMATH_CALUDE_event_probability_l1023_102382


namespace NUMINAMATH_CALUDE_min_coeff_x2_and_coeff_x7_l1023_102348

/-- The function f(x) = (1+x)^m + (1+x)^n -/
def f (m n : ℕ+) (x : ℝ) : ℝ := (1 + x)^(m : ℕ) + (1 + x)^(n : ℕ)

theorem min_coeff_x2_and_coeff_x7 (m n : ℕ+) :
  (m : ℕ) + n = 19 →
  (∃ c : ℕ, c = Nat.choose m 1 + Nat.choose n 1 ∧ c = 19) →
  let coeff_x2 := Nat.choose m 2 + Nat.choose n 2
  ∃ (m' n' : ℕ+),
    (∀ k l : ℕ+, Nat.choose k 2 + Nat.choose l 2 ≥ coeff_x2) ∧
    coeff_x2 = 81 ∧
    Nat.choose m' 7 + Nat.choose n' 7 = 156 :=
sorry

end NUMINAMATH_CALUDE_min_coeff_x2_and_coeff_x7_l1023_102348


namespace NUMINAMATH_CALUDE_water_price_l1023_102318

/-- Given that six bottles of 2 liters of water cost $12, prove that the price of 1 liter of water is $1. -/
theorem water_price (bottles : ℕ) (liters_per_bottle : ℝ) (total_cost : ℝ) :
  bottles = 6 →
  liters_per_bottle = 2 →
  total_cost = 12 →
  total_cost / (bottles * liters_per_bottle) = 1 := by
  sorry

end NUMINAMATH_CALUDE_water_price_l1023_102318


namespace NUMINAMATH_CALUDE_factorization_equality_l1023_102324

theorem factorization_equality (x : ℝ) : 16 * x^3 + 8 * x^2 = 8 * x^2 * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1023_102324


namespace NUMINAMATH_CALUDE_expression_value_l1023_102306

theorem expression_value (m : ℝ) (h : m^2 - m = 1) : 
  (m - 1)^2 + (m + 1)*(m - 1) + 2022 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1023_102306


namespace NUMINAMATH_CALUDE_f_equals_g_l1023_102384

-- Define the two functions
def f (x : ℝ) : ℝ := (x ^ (1/3)) ^ 3
def g (x : ℝ) : ℝ := x

-- Theorem statement
theorem f_equals_g : ∀ (x : ℝ), f x = g x := by
  sorry

end NUMINAMATH_CALUDE_f_equals_g_l1023_102384


namespace NUMINAMATH_CALUDE_absolute_value_problem_l1023_102381

theorem absolute_value_problem (a b c : ℝ) 
  (ha : |a| = 5)
  (hb : |b| = 3)
  (hc : |c| = 6)
  (hab : |a+b| = -(a+b))
  (hac : |a+c| = a+c) :
  a - b + c = 4 ∨ a - b + c = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l1023_102381


namespace NUMINAMATH_CALUDE_escalator_ride_time_l1023_102334

/-- Represents the scenario of Clea walking on an escalator -/
structure EscalatorScenario where
  /-- Clea's walking speed on stationary escalator (units per second) -/
  walkingSpeed : ℝ
  /-- Total distance of the escalator (units) -/
  escalatorDistance : ℝ
  /-- Speed of the moving escalator (units per second) -/
  escalatorSpeed : ℝ

/-- Time taken for Clea to walk down the stationary escalator -/
def stationaryTime (scenario : EscalatorScenario) : ℝ := 70

/-- Time taken for Clea to walk down the moving escalator -/
def movingTime (scenario : EscalatorScenario) : ℝ := 30

/-- Clea's walking speed increase factor on moving escalator -/
def speedIncreaseFactor : ℝ := 1.5

/-- Theorem stating the time taken for Clea to ride the escalator without walking -/
theorem escalator_ride_time (scenario : EscalatorScenario) :
  scenario.escalatorDistance / scenario.escalatorSpeed = 84 :=
by sorry

end NUMINAMATH_CALUDE_escalator_ride_time_l1023_102334


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l1023_102355

/-- The perimeter of a square garden with area 900 square meters is 120 meters and 12000 centimeters. -/
theorem square_garden_perimeter :
  ∀ (side : ℝ), 
  side^2 = 900 →
  (4 * side = 120) ∧ (4 * side * 100 = 12000) := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l1023_102355


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l1023_102345

theorem sin_plus_two_cos_alpha (α : Real) :
  (∃ x y : Real, x = -3 ∧ y = 4 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l1023_102345


namespace NUMINAMATH_CALUDE_tangent_slopes_product_l1023_102310

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the circle where P lies
def circle_P (x y : ℝ) : Prop := x^2 + y^2 = 7

-- Define the tangent line from P(x₀, y₀) to C with slope k
def tangent_line (x₀ y₀ k x y : ℝ) : Prop := y - y₀ = k * (x - x₀)

-- Define the condition for a line to be tangent to C
def is_tangent (x₀ y₀ k : ℝ) : Prop :=
  ∃ x y, tangent_line x₀ y₀ k x y ∧ ellipse_C x y

-- Main theorem
theorem tangent_slopes_product (x₀ y₀ k₁ k₂ : ℝ) :
  circle_P x₀ y₀ →
  is_tangent x₀ y₀ k₁ →
  is_tangent x₀ y₀ k₂ →
  k₁ ≠ k₂ →
  k₁ * k₂ = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slopes_product_l1023_102310


namespace NUMINAMATH_CALUDE_triple_hash_90_l1023_102360

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.3 * N + 2

-- State the theorem
theorem triple_hash_90 : hash (hash (hash 90)) = 5.21 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_90_l1023_102360


namespace NUMINAMATH_CALUDE_friend_apple_rotations_l1023_102352

/-- Given the conditions of a juggling contest, prove the number of rotations made by each of Toby's friend's apples -/
theorem friend_apple_rotations 
  (toby_baseballs : ℕ)
  (toby_rotations_per_baseball : ℕ)
  (friend_apples : ℕ)
  (winner_total_rotations : ℕ)
  (h1 : toby_baseballs = 5)
  (h2 : toby_rotations_per_baseball = 80)
  (h3 : friend_apples = 4)
  (h4 : winner_total_rotations = 404)
  : (winner_total_rotations - toby_baseballs * toby_rotations_per_baseball) / friend_apples + toby_rotations_per_baseball = 81 := by
  sorry

end NUMINAMATH_CALUDE_friend_apple_rotations_l1023_102352


namespace NUMINAMATH_CALUDE_identical_views_imply_sphere_or_cube_l1023_102362

-- Define a type for solids
inductive Solid
| Sphere
| Cube
| Other

-- Define a property for having three identical orthographic views
def hasThreeIdenticalViews (s : Solid) : Prop :=
  match s with
  | Solid.Sphere => True
  | Solid.Cube => True
  | Solid.Other => False

-- Theorem statement
theorem identical_views_imply_sphere_or_cube (s : Solid) :
  hasThreeIdenticalViews s → (s = Solid.Sphere ∨ s = Solid.Cube) := by
  sorry

end NUMINAMATH_CALUDE_identical_views_imply_sphere_or_cube_l1023_102362


namespace NUMINAMATH_CALUDE_cuboid_volume_l1023_102349

/-- The volume of a cuboid with dimensions 4 cm, 6 cm, and 15 cm is 360 cubic centimeters. -/
theorem cuboid_volume : 
  let length : ℝ := 4
  let width : ℝ := 6
  let height : ℝ := 15
  length * width * height = 360 := by sorry

end NUMINAMATH_CALUDE_cuboid_volume_l1023_102349


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1023_102329

-- Define the function f
variable {f : ℝ → ℝ}

-- Define what it means for f to have an extreme value at x
def has_extreme_value (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x

-- Define what it means for f to be differentiable
def is_differentiable (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, DifferentiableAt ℝ f x

-- Define the proposition p
def p (f : ℝ → ℝ) (x : ℝ) : Prop :=
  has_extreme_value f x

-- Define the proposition q
def q (f : ℝ → ℝ) (x : ℝ) : Prop :=
  is_differentiable f ∧ deriv f x = 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ f : ℝ → ℝ, ∀ x : ℝ, p f x → q f x) ∧
  (∃ f : ℝ → ℝ, ∃ x : ℝ, q f x ∧ ¬p f x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1023_102329


namespace NUMINAMATH_CALUDE_particle_probability_l1023_102351

/-- Probability of reaching (0, 0) from (x, y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

/-- The probability of reaching (0, 0) from (6, 6) is 855/3^12 -/
theorem particle_probability : P 6 6 = 855 / 3^12 := by
  sorry

end NUMINAMATH_CALUDE_particle_probability_l1023_102351


namespace NUMINAMATH_CALUDE_mia_has_110_dollars_l1023_102312

def darwins_money : ℕ := 45

def mias_money : ℕ := 2 * darwins_money + 20

theorem mia_has_110_dollars : mias_money = 110 := by
  sorry

end NUMINAMATH_CALUDE_mia_has_110_dollars_l1023_102312


namespace NUMINAMATH_CALUDE_population_reaches_max_capacity_in_140_years_l1023_102369

def initial_year : ℕ := 1998
def initial_population : ℕ := 200
def total_land : ℕ := 32000
def space_per_person : ℕ := 2
def growth_period : ℕ := 20

def max_capacity : ℕ := total_land / space_per_person

def population_after_years (years : ℕ) : ℕ :=
  initial_population * 2^(years / growth_period)

theorem population_reaches_max_capacity_in_140_years :
  ∃ (y : ℕ), y = 140 ∧ 
  population_after_years y ≥ max_capacity ∧
  population_after_years (y - growth_period) < max_capacity :=
sorry

end NUMINAMATH_CALUDE_population_reaches_max_capacity_in_140_years_l1023_102369
