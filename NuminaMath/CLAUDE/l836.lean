import Mathlib

namespace NUMINAMATH_CALUDE_smallest_a_value_l836_83673

/-- The smallest possible value of a given the conditions -/
theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  a ≥ 2 * Real.pi - 17 ∧ ∃ (a₀ : ℝ), a₀ = 2 * Real.pi - 17 ∧ 
  (∃ b₀ : ℝ, 0 ≤ b₀ ∧ ∀ x : ℤ, Real.sin (a₀ * ↑x + b₀) = Real.sin (17 * ↑x)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l836_83673


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l836_83674

/-- Given a wire cut into two pieces of lengths a and b, where a forms a square
    and b forms a circle, and the perimeter of the square equals the circumference
    of the circle, prove that a/b = 1. -/
theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (4 * (a / 4) = 2 * Real.pi * (b / (2 * Real.pi))) → a / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l836_83674


namespace NUMINAMATH_CALUDE_percentage_both_correct_l836_83648

theorem percentage_both_correct (total : ℝ) (first_correct : ℝ) (second_correct : ℝ) (neither_correct : ℝ) :
  total > 0 →
  first_correct / total = 0.75 →
  second_correct / total = 0.30 →
  neither_correct / total = 0.20 →
  (first_correct + second_correct - (total - neither_correct)) / total = 0.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_both_correct_l836_83648


namespace NUMINAMATH_CALUDE_quadratic_roots_and_integer_case_l836_83683

-- Define the quadratic equation
def quadratic_equation (m : ℕ+) (x : ℝ) : ℝ :=
  m * x^2 - (3 * m + 2) * x + 6

-- Theorem statement
theorem quadratic_roots_and_integer_case (m : ℕ+) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) ∧
  ((∃ x y : ℤ, x ≠ y ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) →
   (m = 1 ∨ m = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_integer_case_l836_83683


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_three_sqrt_three_l836_83635

theorem sqrt_difference_equals_three_sqrt_three : 
  Real.sqrt 75 - Real.sqrt 12 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_three_sqrt_three_l836_83635


namespace NUMINAMATH_CALUDE_inequality_proof_l836_83610

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  (a^2 + b^2 + c^2 + d^2 ≥ 1/4) ∧ 
  (a^2/b + b^2/c + c^2/d + d^2/a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l836_83610


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l836_83667

theorem simplify_and_evaluate_expression (a : ℚ) : 
  a = -1/2 → a * (a^4 - a + 1) * (a - 2) = 59/32 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l836_83667


namespace NUMINAMATH_CALUDE_last_digit_89_base_4_l836_83692

def last_digit_base_4 (n : ℕ) : ℕ := n % 4

theorem last_digit_89_base_4 : last_digit_base_4 89 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_89_base_4_l836_83692


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l836_83611

theorem min_sum_with_reciprocal_constraint (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : 
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 9/w = 1 → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 9/b = 1 ∧ a + b = 16 :=
sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l836_83611


namespace NUMINAMATH_CALUDE_prob_odd_second_roll_l836_83636

/-- A fair die with six faces -/
structure Die :=
  (faces : Finset Nat)
  (fair : faces = {1, 2, 3, 4, 5, 6})

/-- The set of odd numbers on a die -/
def oddFaces (d : Die) : Finset Nat :=
  d.faces.filter (λ n => n % 2 = 1)

/-- Probability of an event in a finite sample space -/
def probability (event : Finset α) (sampleSpace : Finset α) : ℚ :=
  (event.card : ℚ) / (sampleSpace.card : ℚ)

theorem prob_odd_second_roll (d : Die) :
  probability (oddFaces d) d.faces = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_second_roll_l836_83636


namespace NUMINAMATH_CALUDE_max_value_quadratic_function_l836_83652

/-- Given a > 1, the maximum value of f(x) = -x^2 - 2ax + 1 on the interval [-1,1] is 2a -/
theorem max_value_quadratic_function (a : ℝ) (h : a > 1) :
  ∃ (max : ℝ), max = 2 * a ∧ ∀ x ∈ Set.Icc (-1) 1, -x^2 - 2*a*x + 1 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_function_l836_83652


namespace NUMINAMATH_CALUDE_total_chips_in_bag_l836_83637

/-- Represents the number of chips Marnie eats on the first day -/
def first_day_chips : ℕ := 10

/-- Represents the number of chips Marnie eats per day after the first day -/
def daily_chips : ℕ := 10

/-- Represents the total number of days it takes Marnie to finish the bag -/
def total_days : ℕ := 10

/-- Theorem stating that the total number of chips in the bag is 100 -/
theorem total_chips_in_bag : 
  first_day_chips + (total_days - 1) * daily_chips = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_chips_in_bag_l836_83637


namespace NUMINAMATH_CALUDE_binary_representation_of_21_l836_83687

theorem binary_representation_of_21 :
  (21 : ℕ).digits 2 = [1, 0, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_binary_representation_of_21_l836_83687


namespace NUMINAMATH_CALUDE_constant_term_of_liams_polynomial_l836_83691

/-- Represents a polynomial with degree 5 -/
structure Poly5 where
  coeffs : Fin 6 → ℝ
  monic : coeffs 5 = 1

/-- The product of two polynomials -/
def poly_product (p q : Poly5) : Fin 11 → ℝ := sorry

theorem constant_term_of_liams_polynomial 
  (serena_poly liam_poly : Poly5)
  (same_constant : serena_poly.coeffs 0 = liam_poly.coeffs 0)
  (positive_constant : serena_poly.coeffs 0 > 0)
  (same_z2_coeff : serena_poly.coeffs 2 = liam_poly.coeffs 2)
  (product : poly_product serena_poly liam_poly = 
    fun i => match i with
    | 0 => 9  | 1 => 5  | 2 => 10 | 3 => 4  | 4 => 9
    | 5 => 6  | 6 => 5  | 7 => 4  | 8 => 3  | 9 => 2
    | 10 => 1
  ) :
  liam_poly.coeffs 0 = 3 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_liams_polynomial_l836_83691


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l836_83657

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the set of x that satisfies the inequality
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (x^2 - 3*x - 3) < f 1}

-- Theorem statement
theorem solution_set_equivalence (f : ℝ → ℝ) (h : DecreasingFunction f) :
  SolutionSet f = {x | x < -1 ∨ x > 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l836_83657


namespace NUMINAMATH_CALUDE_isaac_pen_purchase_l836_83685

theorem isaac_pen_purchase : ∃ (pens : ℕ), 
  pens + (12 + 5 * pens) = 108 ∧ pens = 16 := by
  sorry

end NUMINAMATH_CALUDE_isaac_pen_purchase_l836_83685


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l836_83658

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 36 * x + c = 0) →
  (a + c = 41) →
  (a < c) →
  (a = (41 - Real.sqrt 385) / 2 ∧ c = (41 + Real.sqrt 385) / 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l836_83658


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l836_83628

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3}
def B : Set ℝ := {x : ℝ | x ≠ 0 ∧ (x - 3) / (2*x) ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l836_83628


namespace NUMINAMATH_CALUDE_total_travel_time_is_45_hours_l836_83698

/-- Represents a city with its time zone offset from New Orleans --/
structure City where
  name : String
  offset : Int

/-- Represents a flight segment with departure and arrival cities, and duration --/
structure FlightSegment where
  departure : City
  arrival : City
  duration : Nat

/-- Represents a layover with city and duration --/
structure Layover where
  city : City
  duration : Nat

/-- Calculates the total travel time considering time zone changes --/
def totalTravelTime (segments : List FlightSegment) (layovers : List Layover) : Nat :=
  sorry

/-- The cities involved in Sue's journey --/
def newOrleans : City := { name := "New Orleans", offset := 0 }
def atlanta : City := { name := "Atlanta", offset := 0 }
def chicago : City := { name := "Chicago", offset := -1 }
def newYork : City := { name := "New York", offset := 0 }
def denver : City := { name := "Denver", offset := -2 }
def sanFrancisco : City := { name := "San Francisco", offset := -3 }

/-- Sue's flight segments --/
def flightSegments : List FlightSegment := [
  { departure := newOrleans, arrival := atlanta, duration := 2 },
  { departure := atlanta, arrival := chicago, duration := 5 },
  { departure := chicago, arrival := newYork, duration := 3 },
  { departure := newYork, arrival := denver, duration := 6 },
  { departure := denver, arrival := sanFrancisco, duration := 4 }
]

/-- Sue's layovers --/
def layovers : List Layover := [
  { city := atlanta, duration := 4 },
  { city := chicago, duration := 3 },
  { city := newYork, duration := 16 },
  { city := denver, duration := 5 }
]

/-- Theorem: The total travel time from New Orleans to San Francisco is 45 hours --/
theorem total_travel_time_is_45_hours :
  totalTravelTime flightSegments layovers = 45 := by sorry

end NUMINAMATH_CALUDE_total_travel_time_is_45_hours_l836_83698


namespace NUMINAMATH_CALUDE_trajectory_area_is_8_l836_83642

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cuboid -/
structure Cuboid where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D
  a₁ : Point3D
  b₁ : Point3D
  c₁ : Point3D
  d₁ : Point3D

/-- The length of AB in the cuboid -/
def ab_length : ℝ := 6

/-- The length of BC in the cuboid -/
def bc_length : ℝ := 3

/-- A moving point P on line segment BD -/
def P (t : ℝ) : Point3D :=
  sorry

/-- A moving point Q on line segment A₁C₁ -/
def Q (t : ℝ) : Point3D :=
  sorry

/-- Point M on PQ such that PM = 2MQ -/
def M (t₁ t₂ : ℝ) : Point3D :=
  sorry

/-- The area of the trajectory of point M -/
def trajectory_area (c : Cuboid) : ℝ :=
  sorry

/-- Theorem stating that the area of the trajectory of point M is 8 -/
theorem trajectory_area_is_8 (c : Cuboid) :
  trajectory_area c = 8 :=
  sorry

end NUMINAMATH_CALUDE_trajectory_area_is_8_l836_83642


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l836_83634

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_edge := sphere_diameter / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l836_83634


namespace NUMINAMATH_CALUDE_option_A_not_sufficient_l836_83612

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def perp_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem option_A_not_sufficient
  (a b : Line3D)
  (α β : Plane3D)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : line_parallel_plane a α)
  (h4 : line_parallel_plane b β)
  (h5 : line_perp_plane a β) :
  ¬ (perp_lines a b) :=
sorry

end NUMINAMATH_CALUDE_option_A_not_sufficient_l836_83612


namespace NUMINAMATH_CALUDE_dividend_calculation_l836_83656

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 158 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l836_83656


namespace NUMINAMATH_CALUDE_shortest_return_path_length_l836_83650

/-- Represents a truncated right circular cone with given properties --/
structure TruncatedCone where
  lowerBaseCircumference : ℝ
  upperBaseCircumference : ℝ
  slopeAngle : ℝ

/-- Represents the tourist's path on the cone --/
def touristPath (cone : TruncatedCone) (upperBaseTravel : ℝ) : ℝ := sorry

/-- Theorem stating the shortest return path length --/
theorem shortest_return_path_length 
  (cone : TruncatedCone) 
  (h1 : cone.lowerBaseCircumference = 10)
  (h2 : cone.upperBaseCircumference = 9)
  (h3 : cone.slopeAngle = π / 3) -- 60 degrees in radians
  (h4 : upperBaseTravel = 3) :
  touristPath cone upperBaseTravel = (5 * Real.sqrt 3) / π :=
sorry

end NUMINAMATH_CALUDE_shortest_return_path_length_l836_83650


namespace NUMINAMATH_CALUDE_line_of_symmetry_l836_83653

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 4 = 0

-- Define the line of symmetry
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

-- Theorem statement
theorem line_of_symmetry :
  ∀ (x y : ℝ), 
    (∃ (x' y' : ℝ), circle_O x' y' ∧ circle_C x y ∧ 
      line_l ((x + x')/2) ((y + y')/2) ∧
      (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2) :=
sorry

end NUMINAMATH_CALUDE_line_of_symmetry_l836_83653


namespace NUMINAMATH_CALUDE_birds_wings_count_johns_birds_wings_l836_83641

/-- The number of wings of all birds John can buy with money from his grandparents -/
theorem birds_wings_count (money_per_grandparent : ℕ) (num_grandparents : ℕ) (cost_per_bird : ℕ) (wings_per_bird : ℕ) : ℕ :=
  by
  sorry

/-- Proof that John can buy birds with 20 wings in total -/
theorem johns_birds_wings :
  birds_wings_count 50 4 20 2 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_birds_wings_count_johns_birds_wings_l836_83641


namespace NUMINAMATH_CALUDE_triangle_angle_equality_l836_83601

theorem triangle_angle_equality (A B C : ℝ) (a b c : ℝ) :
  0 < B ∧ B < π →
  0 < a ∧ 0 < b ∧ 0 < c →
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A →
  B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_equality_l836_83601


namespace NUMINAMATH_CALUDE_only_eleven_not_sum_of_two_primes_l836_83696

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem only_eleven_not_sum_of_two_primes :
  is_sum_of_two_primes 5 ∧
  is_sum_of_two_primes 7 ∧
  is_sum_of_two_primes 9 ∧
  ¬(is_sum_of_two_primes 11) ∧
  is_sum_of_two_primes 13 :=
by sorry

end NUMINAMATH_CALUDE_only_eleven_not_sum_of_two_primes_l836_83696


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l836_83605

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 1 ≥ 0 ∧ (x - 1) / 2 < 1}
  S = {x : ℝ | -1 ≤ x ∧ x < 3} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l836_83605


namespace NUMINAMATH_CALUDE_carols_peanuts_l836_83676

theorem carols_peanuts (initial_peanuts : ℕ) :
  initial_peanuts + 5 = 7 → initial_peanuts = 2 := by
  sorry

end NUMINAMATH_CALUDE_carols_peanuts_l836_83676


namespace NUMINAMATH_CALUDE_intersection_k_value_l836_83644

/-- Given two lines that intersect at x = -15, prove the value of k -/
theorem intersection_k_value (k : ℝ) : 
  (∀ x y : ℝ, -3 * x + y = k ∧ 0.3 * x + y = 10) →
  (∃ y : ℝ, -3 * (-15) + y = k ∧ 0.3 * (-15) + y = 10) →
  k = 59.5 := by
sorry

end NUMINAMATH_CALUDE_intersection_k_value_l836_83644


namespace NUMINAMATH_CALUDE_infinitely_many_planes_through_collinear_points_l836_83689

/-- Three distinct points on a line -/
structure ThreeCollinearPoints (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V] :=
  (p₁ p₂ p₃ : V)
  (distinct : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (collinear : ∃ (t₁ t₂ : ℝ), p₃ - p₁ = t₁ • (p₂ - p₁) ∧ p₂ - p₁ = t₂ • (p₃ - p₁))

/-- A plane passing through three points -/
def Plane (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V] (p₁ p₂ p₃ : V) :=
  {x : V | ∃ (a b c : ℝ), x = a • p₁ + b • p₂ + c • p₃ ∧ a + b + c = 1}

/-- Theorem: There are infinitely many planes passing through three collinear points -/
theorem infinitely_many_planes_through_collinear_points
  {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
  (points : ThreeCollinearPoints V) :
  ∃ (planes : Set (Plane V points.p₁ points.p₂ points.p₃)), Infinite planes :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_planes_through_collinear_points_l836_83689


namespace NUMINAMATH_CALUDE_mobile_chip_transistor_count_l836_83647

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem mobile_chip_transistor_count :
  toScientificNotation 15300000000 = ScientificNotation.mk 1.53 10 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_mobile_chip_transistor_count_l836_83647


namespace NUMINAMATH_CALUDE_english_alphabet_is_set_l836_83625

-- Define the type for English alphabet letters
inductive EnglishLetter
| A | B | C | D | E | F | G | H | I | J | K | L | M
| N | O | P | Q | R | S | T | U | V | W | X | Y | Z

-- Define the properties of set elements
def isDefinite (x : Type) : Prop := sorry
def isDistinct (x : Type) : Prop := sorry
def isUnordered (x : Type) : Prop := sorry

-- Define what it means to be a valid set
def isValidSet (x : Type) : Prop :=
  isDefinite x ∧ isDistinct x ∧ isUnordered x

-- Theorem stating that the English alphabet forms a set
theorem english_alphabet_is_set :
  isValidSet EnglishLetter :=
sorry

end NUMINAMATH_CALUDE_english_alphabet_is_set_l836_83625


namespace NUMINAMATH_CALUDE_f_min_max_l836_83671

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem f_min_max :
  (∃ (x : ℝ), x ∈ domain ∧ f x = 0) ∧
  (∀ (y : ℝ), y ∈ domain → f y ≥ 0) ∧
  (∃ (z : ℝ), z ∈ domain ∧ f z = 4) ∧
  (∀ (w : ℝ), w ∈ domain → f w ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_f_min_max_l836_83671


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l836_83666

/-- Represents the number of employees in each title category -/
structure EmployeeCount where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  junior : ℕ

/-- Represents the sample size for each title category -/
structure SampleSize where
  senior : ℕ
  intermediate : ℕ
  junior : ℕ

/-- Calculates the stratified sample size for a given category -/
def stratifiedSampleSize (totalEmployees : ℕ) (categoryCount : ℕ) (sampleSize : ℕ) : ℕ :=
  (sampleSize * categoryCount) / totalEmployees

/-- Theorem: The stratified sampling results in the correct sample sizes -/
theorem stratified_sampling_correct 
  (employees : EmployeeCount) 
  (sample : SampleSize) : 
  employees.total = 150 ∧ 
  employees.senior = 15 ∧ 
  employees.intermediate = 45 ∧ 
  employees.junior = 90 ∧
  sample.senior = stratifiedSampleSize employees.total employees.senior 30 ∧
  sample.intermediate = stratifiedSampleSize employees.total employees.intermediate 30 ∧
  sample.junior = stratifiedSampleSize employees.total employees.junior 30 →
  sample.senior = 3 ∧ sample.intermediate = 9 ∧ sample.junior = 18 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l836_83666


namespace NUMINAMATH_CALUDE_school_enrollment_problem_l836_83662

theorem school_enrollment_problem (total_last_year : ℕ) 
  (xx_increase_rate yy_increase_rate : ℚ)
  (xx_to_yy yy_to_xx : ℕ)
  (xx_dropout_rate yy_dropout_rate : ℚ)
  (net_growth_diff : ℕ) :
  total_last_year = 4000 ∧
  xx_increase_rate = 7/100 ∧
  yy_increase_rate = 3/100 ∧
  xx_to_yy = 10 ∧
  yy_to_xx = 5 ∧
  xx_dropout_rate = 3/100 ∧
  yy_dropout_rate = 1/100 ∧
  net_growth_diff = 40 →
  ∃ (xx_last_year yy_last_year : ℕ),
    xx_last_year + yy_last_year = total_last_year ∧
    (xx_last_year * xx_increase_rate - xx_last_year * xx_dropout_rate - xx_to_yy) -
    (yy_last_year * yy_increase_rate - yy_last_year * yy_dropout_rate + yy_to_xx) = net_growth_diff ∧
    yy_last_year = 1750 :=
by sorry

end NUMINAMATH_CALUDE_school_enrollment_problem_l836_83662


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l836_83633

def set_A : Set ℝ := {x | |x - 1| < 3}
def set_B : Set ℝ := {x | (x - 1) / (x - 5) < 0}

theorem intersection_of_A_and_B : 
  set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l836_83633


namespace NUMINAMATH_CALUDE_forty_seventh_digit_of_1_17_l836_83697

/-- The decimal representation of 1/17 -/
def decimal_rep_1_17 : ℚ := 1 / 17

/-- The function that returns the nth digit after the decimal point in a rational number's decimal representation -/
noncomputable def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem: The 47th digit after the decimal point in the decimal representation of 1/17 is 6 -/
theorem forty_seventh_digit_of_1_17 : nth_digit_after_decimal decimal_rep_1_17 47 = 6 := by sorry

end NUMINAMATH_CALUDE_forty_seventh_digit_of_1_17_l836_83697


namespace NUMINAMATH_CALUDE_clive_change_l836_83663

/-- The amount of money Clive has to spend -/
def budget : ℚ := 10

/-- The number of olives Clive needs -/
def olives_needed : ℕ := 80

/-- The number of olives in each jar -/
def olives_per_jar : ℕ := 20

/-- The cost of one jar of olives -/
def cost_per_jar : ℚ := 3/2

/-- The change Clive will have after buying the required number of olive jars -/
def change : ℚ := budget - (↑(olives_needed / olives_per_jar) * cost_per_jar)

theorem clive_change :
  change = 4 := by sorry

end NUMINAMATH_CALUDE_clive_change_l836_83663


namespace NUMINAMATH_CALUDE_limit_function_equals_one_half_l836_83604

/-- The limit of ((1+8x)/(2+11x))^(1/(x^2+1)) as x approaches 0 is 1/2 -/
theorem limit_function_equals_one_half :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
    |(((1 + 8*x) / (2 + 11*x)) ^ (1 / (x^2 + 1))) - (1/2)| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_function_equals_one_half_l836_83604


namespace NUMINAMATH_CALUDE_students_walking_home_l836_83620

theorem students_walking_home (bus automobile skateboard bicycle : ℚ)
  (h_bus : bus = 1 / 3)
  (h_auto : automobile = 1 / 5)
  (h_skate : skateboard = 1 / 8)
  (h_bike : bicycle = 1 / 10)
  (h_total : bus + automobile + skateboard + bicycle < 1) :
  1 - (bus + automobile + skateboard + bicycle) = 29 / 120 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_l836_83620


namespace NUMINAMATH_CALUDE_highlighter_count_l836_83693

/-- The number of pink highlighters in Kaya's teacher's desk -/
def pink_highlighters : ℕ := 10

/-- The number of yellow highlighters in Kaya's teacher's desk -/
def yellow_highlighters : ℕ := 15

/-- The number of blue highlighters in Kaya's teacher's desk -/
def blue_highlighters : ℕ := 8

/-- The total number of highlighters in Kaya's teacher's desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters

theorem highlighter_count : total_highlighters = 33 := by
  sorry

end NUMINAMATH_CALUDE_highlighter_count_l836_83693


namespace NUMINAMATH_CALUDE_unique_solution_is_one_point_five_l836_83678

/-- Given that (3a+2b)x^2+ax+b=0 is a linear equation in x with a unique solution, prove that x = 1.5 -/
theorem unique_solution_is_one_point_five (a b x : ℝ) :
  ((3*a + 2*b) * x^2 + a*x + b = 0) →  -- The equation
  (∃! x, (3*a + 2*b) * x^2 + a*x + b = 0) →  -- Unique solution exists
  (∀ y, (3*a + 2*b) * y^2 + a*y + b = 0 → y = x) →  -- Linear equation condition
  x = 1.5 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_is_one_point_five_l836_83678


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l836_83677

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence a_n where a_6 + a_8 = 4, prove that a_8(a_4 + 2a_6 + a_8) = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : IsGeometricSequence a) 
    (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l836_83677


namespace NUMINAMATH_CALUDE_right_triangular_prism_volume_l836_83639

/-- The volume of a right triangular prism with base side lengths 14 and height 8 is 784 cubic units. -/
theorem right_triangular_prism_volume : 
  ∀ (base_side_length height : ℝ), 
    base_side_length = 14 → 
    height = 8 → 
    (1/2 * base_side_length * base_side_length) * height = 784 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_prism_volume_l836_83639


namespace NUMINAMATH_CALUDE_special_number_composite_l836_83638

/-- Represents the number formed by n+1 ones, followed by a 2, followed by n+1 ones -/
def special_number (n : ℕ) : ℕ :=
  (10^(n+1) - 1) / 9 * 10^(n+1) + (10^(n+1) - 1) / 9

/-- A number is composite if it has a factor between 1 and itself -/
def is_composite (m : ℕ) : Prop :=
  ∃ k, 1 < k ∧ k < m ∧ m % k = 0

/-- Theorem stating that the special number is composite for all natural numbers n -/
theorem special_number_composite (n : ℕ) : is_composite (special_number n) := by
  sorry


end NUMINAMATH_CALUDE_special_number_composite_l836_83638


namespace NUMINAMATH_CALUDE_dictation_mistakes_l836_83669

theorem dictation_mistakes (n : ℕ) (max_mistakes : ℕ) 
  (h1 : n = 30) 
  (h2 : max_mistakes = 12) : 
  ∃ k : ℕ, ∃ (s : Finset (Fin n)), s.card ≥ 3 ∧ 
  ∀ i ∈ s, ∃ f : Fin n → ℕ, f i = k ∧ f i ≤ max_mistakes :=
by sorry

end NUMINAMATH_CALUDE_dictation_mistakes_l836_83669


namespace NUMINAMATH_CALUDE_alice_favorite_number_l836_83681

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem alice_favorite_number :
  ∃! n : ℕ, 30 < n ∧ n < 150 ∧ n % 11 = 0 ∧ n % 2 ≠ 0 ∧ sum_of_digits n % 5 = 0 ∧ n = 55 := by
  sorry

end NUMINAMATH_CALUDE_alice_favorite_number_l836_83681


namespace NUMINAMATH_CALUDE_strawberry_milk_count_total_milk_sum_l836_83655

/-- The number of students who selected strawberry milk -/
def strawberry_milk_students : ℕ := sorry

/-- The number of students who selected chocolate milk -/
def chocolate_milk_students : ℕ := 2

/-- The number of students who selected regular milk -/
def regular_milk_students : ℕ := 3

/-- The total number of milks taken -/
def total_milks : ℕ := 20

/-- Theorem stating that the number of students who selected strawberry milk is 15 -/
theorem strawberry_milk_count : 
  strawberry_milk_students = 15 :=
by
  sorry

/-- Theorem stating that the total number of milks is the sum of all milk selections -/
theorem total_milk_sum :
  total_milks = chocolate_milk_students + strawberry_milk_students + regular_milk_students :=
by
  sorry

end NUMINAMATH_CALUDE_strawberry_milk_count_total_milk_sum_l836_83655


namespace NUMINAMATH_CALUDE_monomial_polynomial_multiplication_l836_83613

theorem monomial_polynomial_multiplication :
  ∀ (x y : ℝ), -3 * x * y * (4 * y - 2 * x - 1) = -12 * x * y^2 + 6 * x^2 * y + 3 * x * y := by
  sorry

end NUMINAMATH_CALUDE_monomial_polynomial_multiplication_l836_83613


namespace NUMINAMATH_CALUDE_balcony_orchestra_difference_l836_83629

/-- Represents the number of tickets sold for a theater performance -/
structure TheaterTickets where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total number of tickets sold -/
def totalTickets (t : TheaterTickets) : ℕ := t.orchestra + t.balcony

/-- Calculates the total revenue from ticket sales -/
def totalRevenue (t : TheaterTickets) : ℕ := 12 * t.orchestra + 8 * t.balcony

theorem balcony_orchestra_difference (t : TheaterTickets) :
  totalTickets t = 355 → totalRevenue t = 3320 → t.balcony - t.orchestra = 115 := by
  sorry

#check balcony_orchestra_difference

end NUMINAMATH_CALUDE_balcony_orchestra_difference_l836_83629


namespace NUMINAMATH_CALUDE_hospital_staff_count_l836_83640

theorem hospital_staff_count (doctors nurses : ℕ) (h1 : doctors * 11 = nurses * 8) (h2 : nurses = 264) : 
  doctors + nurses = 456 := by
sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l836_83640


namespace NUMINAMATH_CALUDE_simplify_absolute_value_expression_l836_83630

noncomputable def f (x : ℝ) : ℝ := |2*x + 1| - |x - 3| + |x - 6|

noncomputable def g (x : ℝ) : ℝ :=
  if x < -1/2 then -2*x + 2
  else if x < 3 then 2*x + 4
  else if x < 6 then 10
  else 2*x - 2

theorem simplify_absolute_value_expression :
  ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_expression_l836_83630


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l836_83654

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  a = 1 → b = Real.sqrt 2 → B = π / 4 → A = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l836_83654


namespace NUMINAMATH_CALUDE_problem_statement_l836_83665

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (9 / a + 1 / b ≥ 4) ∧ ((a + 3 / b) * (b + 3 / a) ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l836_83665


namespace NUMINAMATH_CALUDE_equation_solution_l836_83632

theorem equation_solution : ∃ y : ℝ, (2 / y + 3 / y / (6 / y) = 1.5) ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l836_83632


namespace NUMINAMATH_CALUDE_value_of_b_l836_83623

theorem value_of_b (a b : ℝ) (eq1 : 3 * a + 1 = 1) (eq2 : b - a = 2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l836_83623


namespace NUMINAMATH_CALUDE_intersection_distance_squared_is_396_8_l836_83668

/-- Two circles in a 2D plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- The square of the distance between intersection points of two circles -/
def intersectionPointsDistanceSquared (circles : TwoCircles) : ℝ :=
  sorry

/-- Theorem stating that the square of the distance between intersection points
    of the given circles is 396.8 -/
theorem intersection_distance_squared_is_396_8 :
  let circles : TwoCircles := {
    center1 := (0, 0),
    radius1 := 5,
    center2 := (4, -2),
    radius2 := 3
  }
  intersectionPointsDistanceSquared circles = 396.8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_is_396_8_l836_83668


namespace NUMINAMATH_CALUDE_dried_fruit_business_theorem_l836_83649

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (x : ℝ) : ℝ := -80 * x + 560

/-- Represents the daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 3) * (sales_quantity x) - 80

theorem dried_fruit_business_theorem 
  (cost_per_bag : ℝ) 
  (other_expenses : ℝ) 
  (min_price max_price : ℝ) :
  cost_per_bag = 3 →
  other_expenses = 80 →
  min_price = 3.5 →
  max_price = 5.5 →
  sales_quantity 3.5 = 280 →
  sales_quantity 5.5 = 120 →
  (∀ x, min_price ≤ x ∧ x ≤ max_price → 
    sales_quantity x = -80 * x + 560) →
  daily_profit 4 = 160 ∧
  (∀ x, min_price ≤ x ∧ x ≤ max_price → 
    daily_profit x ≤ 240) ∧
  daily_profit 5 = 240 := by
sorry

end NUMINAMATH_CALUDE_dried_fruit_business_theorem_l836_83649


namespace NUMINAMATH_CALUDE_unique_solution_l836_83643

/-- A triplet of natural numbers (a, b, c) where b and c are two-digit numbers. -/
structure Triplet where
  a : ℕ
  b : ℕ
  c : ℕ
  b_twodigit : 10 ≤ b ∧ b ≤ 99
  c_twodigit : 10 ≤ c ∧ c ≤ 99

/-- The property that a triplet (a, b, c) satisfies the equation 10^4*a + 100*b + c = (a + b + c)^3. -/
def satisfies_equation (t : Triplet) : Prop :=
  10^4 * t.a + 100 * t.b + t.c = (t.a + t.b + t.c)^3

/-- Theorem stating that (9, 11, 25) is the only triplet satisfying the equation. -/
theorem unique_solution :
  ∃! t : Triplet, satisfies_equation t ∧ t.a = 9 ∧ t.b = 11 ∧ t.c = 25 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l836_83643


namespace NUMINAMATH_CALUDE_unique_solution_for_sum_and_product_l836_83602

theorem unique_solution_for_sum_and_product (x y z : ℝ) :
  x + y + z = 38 →
  x * y * z = 2002 →
  0 < x →
  x ≤ 11 →
  z ≥ 14 →
  x = 11 ∧ y = 13 ∧ z = 14 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_sum_and_product_l836_83602


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l836_83675

theorem hot_dogs_remainder : 25197629 % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l836_83675


namespace NUMINAMATH_CALUDE_wire_around_square_field_l836_83617

theorem wire_around_square_field (area : ℝ) (wire_length : ℝ) (times_around : ℕ) : 
  area = 69696 →
  wire_length = 15840 →
  times_around = 15 →
  wire_length = times_around * (4 * Real.sqrt area) :=
by
  sorry

end NUMINAMATH_CALUDE_wire_around_square_field_l836_83617


namespace NUMINAMATH_CALUDE_legacy_gain_satisfies_conditions_l836_83646

/-- The legacy gain received by Ms. Emily Smith -/
def legacy_gain : ℝ := 46345

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The regional tax rate as a decimal -/
def regional_tax_rate : ℝ := 0.15

/-- The total amount of taxes paid -/
def total_taxes_paid : ℝ := 16800

/-- Theorem stating that the legacy gain satisfies the given conditions -/
theorem legacy_gain_satisfies_conditions :
  federal_tax_rate * legacy_gain + 
  regional_tax_rate * (legacy_gain - federal_tax_rate * legacy_gain) = 
  total_taxes_paid := by sorry

end NUMINAMATH_CALUDE_legacy_gain_satisfies_conditions_l836_83646


namespace NUMINAMATH_CALUDE_sqrt_relationship_l836_83680

theorem sqrt_relationship (h : Real.sqrt 22500 = 150) : Real.sqrt 0.0225 = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_relationship_l836_83680


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sums_l836_83688

theorem polynomial_coefficient_sums (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (x + 2)^8 = a₀ + a₁*(x + 1) + a₂*(x + 1)^2 + a₃*(x + 1)^3 + 
                        a₄*(x + 1)^4 + a₅*(x + 1)^5 + a₆*(x + 1)^6 + 
                        a₇*(x + 1)^7 + a₈*(x + 1)^8) →
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 255 ∧
   a₁ + a₃ + a₅ + a₇ = 128) := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sums_l836_83688


namespace NUMINAMATH_CALUDE_S_minimized_at_two_l836_83679

/-- The area S(a) bounded by a line and a parabola -/
noncomputable def S (a : ℝ) : ℝ :=
  (1/6) * ((a^2 - 4*a + 8) ^ (3/2))

/-- The theorem stating that S(a) is minimized when a = 2 -/
theorem S_minimized_at_two :
  ∃ (a : ℝ), 0 ≤ a ∧ a ≤ 6 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 6 → S a ≤ S x :=
by
  -- The proof goes here
  sorry

#check S_minimized_at_two

end NUMINAMATH_CALUDE_S_minimized_at_two_l836_83679


namespace NUMINAMATH_CALUDE_elliott_triangle_hypotenuse_l836_83659

-- Define a right-angle triangle
structure RightTriangle where
  base : ℝ
  height : ℝ
  hypotenuse : ℝ
  right_angle : base^2 + height^2 = hypotenuse^2

-- Theorem statement
theorem elliott_triangle_hypotenuse (t : RightTriangle) 
  (h1 : t.base = 4)
  (h2 : t.height = 3) : 
  t.hypotenuse = 5 := by
  sorry

end NUMINAMATH_CALUDE_elliott_triangle_hypotenuse_l836_83659


namespace NUMINAMATH_CALUDE_age_problem_l836_83664

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = 3 * a →
  a + b + c + d = 72 →
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l836_83664


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_x_range_l836_83695

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

theorem point_in_second_quadrant_x_range :
  ∀ x : ℝ, second_quadrant ⟨x - 2, x⟩ → 0 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_x_range_l836_83695


namespace NUMINAMATH_CALUDE_polygon_triangulation_l836_83672

/-- Theorem: For any polygon with n sides divided into k triangles, k ≥ n - 2 -/
theorem polygon_triangulation (n k : ℕ) (h_n : n ≥ 3) (h_k : k > 0) : k ≥ n - 2 := by
  sorry


end NUMINAMATH_CALUDE_polygon_triangulation_l836_83672


namespace NUMINAMATH_CALUDE_initial_mixture_amount_l836_83622

/-- Represents the problem of finding the initial amount of mixture -/
theorem initial_mixture_amount (initial_mixture : ℝ) : 
  (0.1 * initial_mixture / initial_mixture = 0.1) →  -- Initial mixture is 10% grape juice
  (0.25 * (initial_mixture + 10) = 0.1 * initial_mixture + 10) →  -- Resulting mixture is 25% grape juice
  initial_mixture = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_mixture_amount_l836_83622


namespace NUMINAMATH_CALUDE_algebraic_arithmetic_equivalence_l836_83615

theorem algebraic_arithmetic_equivalence (a b : ℕ) (h : a > b) :
  (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_arithmetic_equivalence_l836_83615


namespace NUMINAMATH_CALUDE_two_out_graph_partition_theorem_l836_83660

/-- A directed graph where each vertex has exactly two outgoing edges -/
structure TwoOutGraph (V : Type*) :=
  (edges : V → V × V)

/-- A partition of vertices into districts -/
def DistrictPartition (V : Type*) := V → Fin 1014

theorem two_out_graph_partition_theorem {V : Type*} (G : TwoOutGraph V) :
  ∃ (partition : DistrictPartition V),
    (∀ v w : V, (partition v = partition w) → 
      (G.edges v).1 ≠ w ∧ (G.edges v).2 ≠ w) ∧
    (∀ d1 d2 : Fin 1014, d1 ≠ d2 → 
      (∀ v w : V, partition v = d1 → partition w = d2 → 
        ((G.edges v).1 = w ∨ (G.edges v).2 = w) → 
        ∀ x y : V, partition x = d1 → partition y = d2 → 
          ((G.edges x).1 = y ∨ (G.edges x).2 = y) → 
          ((G.edges v).1 = w ∨ (G.edges x).1 = y) ∧ 
          ((G.edges v).2 = w ∨ (G.edges x).2 = y))) :=
sorry

end NUMINAMATH_CALUDE_two_out_graph_partition_theorem_l836_83660


namespace NUMINAMATH_CALUDE_base7_to_base10_23456_l836_83603

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- The given number in base 7 --/
def base7Number : List Nat := [6, 5, 4, 3, 2]

/-- Theorem stating that the base 10 equivalent of 23456 in base 7 is 6068 --/
theorem base7_to_base10_23456 :
  base7ToBase10 base7Number = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_23456_l836_83603


namespace NUMINAMATH_CALUDE_count_even_greater_than_20000_position_of_35214_count_divisible_by_6_l836_83645

/-- The set of available digits --/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A five-digit number formed from the available digits --/
structure FiveDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  d5 : Nat
  h1 : d1 ∈ digits
  h2 : d2 ∈ digits
  h3 : d3 ∈ digits
  h4 : d4 ∈ digits
  h5 : d5 ∈ digits
  h6 : d1 ≠ 0  -- Ensures it's a five-digit number

/-- The value of a FiveDigitNumber --/
def FiveDigitNumber.value (n : FiveDigitNumber) : Nat :=
  10000 * n.d1 + 1000 * n.d2 + 100 * n.d3 + 10 * n.d4 + n.d5

/-- The set of all valid FiveDigitNumbers --/
def allFiveDigitNumbers : Finset FiveDigitNumber := sorry

theorem count_even_greater_than_20000 :
  (allFiveDigitNumbers.filter (λ n => n.value % 2 = 0 ∧ n.value > 20000)).card = 240 := by sorry

theorem position_of_35214 :
  (allFiveDigitNumbers.filter (λ n => n.value < 35214)).card + 1 = 351 := by sorry

theorem count_divisible_by_6 :
  (allFiveDigitNumbers.filter (λ n => n.value % 6 = 0)).card = 108 := by sorry

end NUMINAMATH_CALUDE_count_even_greater_than_20000_position_of_35214_count_divisible_by_6_l836_83645


namespace NUMINAMATH_CALUDE_turtleneck_discount_theorem_l836_83631

theorem turtleneck_discount_theorem (C : ℝ) (D : ℝ) : 
  C > 0 →  -- Cost is positive
  (1.50 * C) * (1 - D / 100) = 1.125 * C → -- Equation from profit condition
  D = 25 := by
sorry

end NUMINAMATH_CALUDE_turtleneck_discount_theorem_l836_83631


namespace NUMINAMATH_CALUDE_product_of_solutions_l836_83621

theorem product_of_solutions (x : ℝ) : 
  (x^2 + 6*x - 21 = 0) → 
  (∃ α β : ℝ, (α + β = -6) ∧ (α * β = -21)) := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l836_83621


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l836_83619

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 5 * x + 4 < 0 ↔ -4/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l836_83619


namespace NUMINAMATH_CALUDE_positive_sixth_root_of_64_l836_83624

theorem positive_sixth_root_of_64 (y : ℝ) (h1 : y > 0) (h2 : y^6 = 64) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_sixth_root_of_64_l836_83624


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_l836_83600

/-- A right triangle with consecutive even number legs and hypotenuse 34 has leg sum 50 -/
theorem right_triangle_leg_sum (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 34 →  -- Hypotenuse is 34
  ∃ k : ℕ, a = 2*k ∧ b = 2*k + 2 →  -- Legs are consecutive even numbers
  a + b = 50 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_l836_83600


namespace NUMINAMATH_CALUDE_wax_calculation_l836_83627

/-- Given the total required wax and additional wax needed, calculates the amount of wax already possessed. -/
def wax_already_possessed (total_required : ℕ) (additional_needed : ℕ) : ℕ :=
  total_required - additional_needed

/-- Proves that given the specific values in the problem, the wax already possessed is 331 g. -/
theorem wax_calculation :
  let total_required : ℕ := 353
  let additional_needed : ℕ := 22
  wax_already_possessed total_required additional_needed = 331 := by
  sorry

end NUMINAMATH_CALUDE_wax_calculation_l836_83627


namespace NUMINAMATH_CALUDE_principal_calculation_l836_83608

/-- Given a principal P and an interest rate R (as a percentage),
    if the amount after 2 years is 780 and after 7 years is 1020,
    then the principal P is 684. -/
theorem principal_calculation (P R : ℚ) 
  (h1 : P + (P * R * 2) / 100 = 780)
  (h2 : P + (P * R * 7) / 100 = 1020) : 
  P = 684 := by
sorry

end NUMINAMATH_CALUDE_principal_calculation_l836_83608


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l836_83618

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = a^2 - 1 + Complex.I * (a - 1)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l836_83618


namespace NUMINAMATH_CALUDE_triangle_properties_l836_83670

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  b * Real.cos A + a * Real.cos B = 2 * c * Real.cos A →
  D.1 + D.2 = 1 →
  D.1 = 3 * D.2 →
  (D.1 * a + D.2 * b)^2 + (D.1 * c)^2 - 2 * (D.1 * a + D.2 * b) * (D.1 * c) * Real.cos A = 9 →
  A = π / 3 ∧
  (∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 0 →
    (D.1 * a' + D.2 * b')^2 + (D.1 * c')^2 - 2 * (D.1 * a' + D.2 * b') * (D.1 * c') * Real.cos A = 9 →
    1/2 * b' * c' * Real.sin A ≤ 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l836_83670


namespace NUMINAMATH_CALUDE_orange_juice_revenue_l836_83682

/-- Represents the number of trees each sister owns -/
def trees : ℕ := 110

/-- Represents the number of oranges Gabriela's trees produce per tree -/
def gabriela_oranges : ℕ := 600

/-- Represents the number of oranges Alba's trees produce per tree -/
def alba_oranges : ℕ := 400

/-- Represents the number of oranges Maricela's trees produce per tree -/
def maricela_oranges : ℕ := 500

/-- Represents the number of oranges needed to make one cup of juice -/
def oranges_per_cup : ℕ := 3

/-- Represents the price of one cup of juice in dollars -/
def price_per_cup : ℕ := 4

/-- Theorem stating that the total revenue from selling orange juice is $220,000 -/
theorem orange_juice_revenue :
  (trees * gabriela_oranges + trees * alba_oranges + trees * maricela_oranges) / oranges_per_cup * price_per_cup = 220000 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_revenue_l836_83682


namespace NUMINAMATH_CALUDE_function_always_positive_l836_83606

/-- A function satisfying the given differential inequality is always positive -/
theorem function_always_positive
  (f : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hf' : Differentiable ℝ (deriv f))
  (h : ∀ x, x * (deriv^[2] f x) + 2 * f x > x^2) :
  ∀ x, f x > 0 := by sorry

end NUMINAMATH_CALUDE_function_always_positive_l836_83606


namespace NUMINAMATH_CALUDE_difference_of_sums_l836_83690

def sum_even_up_to (n : ℕ) : ℕ :=
  (n / 2) * (2 + n)

def sum_odd_up_to (n : ℕ) : ℕ :=
  ((n + 1) / 2) ^ 2

theorem difference_of_sums : sum_even_up_to 100 - sum_odd_up_to 29 = 2325 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_sums_l836_83690


namespace NUMINAMATH_CALUDE_math_competition_unattempted_questions_l836_83694

theorem math_competition_unattempted_questions :
  ∀ (total_questions : ℕ) (correct_points incorrect_points : ℤ) (score : ℕ),
    total_questions = 20 →
    correct_points = 8 →
    incorrect_points = -5 →
    (∃ k : ℕ, score = 13 * k) →
    ∀ (correct attempted : ℕ),
      attempted ≤ total_questions →
      score = correct_points * correct + incorrect_points * (attempted - correct) →
      (total_questions - attempted = 20 ∨ total_questions - attempted = 7) :=
by sorry

end NUMINAMATH_CALUDE_math_competition_unattempted_questions_l836_83694


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l836_83626

/-- A perfect square trinomial in the form ax^2 + bx + c -/
structure PerfectSquareTrinomial (a b c : ℝ) : Prop where
  is_perfect_square : ∃ (p q : ℝ), a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, PerfectSquareTrinomial 1 (-m) 25 → m = 10 ∨ m = -10 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l836_83626


namespace NUMINAMATH_CALUDE_three_by_three_grid_paths_l836_83609

/-- The number of paths from (0,0) to (n,m) on a grid, moving only right or down -/
def grid_paths (n m : ℕ) : ℕ := Nat.choose (n + m) n

/-- Theorem: There are 20 distinct paths from the top-left to the bottom-right corner of a 3x3 grid -/
theorem three_by_three_grid_paths : grid_paths 3 3 = 20 := by sorry

end NUMINAMATH_CALUDE_three_by_three_grid_paths_l836_83609


namespace NUMINAMATH_CALUDE_rectangle_shading_theorem_l836_83699

theorem rectangle_shading_theorem :
  let r : ℝ := 1/4
  let series_sum : ℝ := r / (1 - r)
  series_sum = 1/3 := by sorry

end NUMINAMATH_CALUDE_rectangle_shading_theorem_l836_83699


namespace NUMINAMATH_CALUDE_rhombus_inscribed_circle_radius_l836_83684

theorem rhombus_inscribed_circle_radius 
  (side_length : ℝ) 
  (acute_angle : ℝ) 
  (h : side_length = 8 ∧ acute_angle = 30 * π / 180) : 
  side_length * Real.sin (acute_angle) / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_inscribed_circle_radius_l836_83684


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l836_83607

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, a * x^2 + b * x - 2 = 0 ↔ x = -2 ∨ x = -1/4) → 
  a + b = -13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l836_83607


namespace NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l836_83616

/-- Ellipse C with equation x²/4 + y²/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Line l with equation y = kx + m -/
def line_l (k m x y : ℝ) : Prop := y = k*x + m

/-- Point A is the right vertex of the ellipse -/
def point_A : ℝ × ℝ := (2, 0)

/-- Circle with diameter MN passes through point A -/
def circle_passes_through_A (M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  (x₁ - 2) * (x₂ - 2) + y₁ * y₂ = 0

theorem ellipse_intersection_fixed_point (k m : ℝ) :
  ∃ (M N : ℝ × ℝ),
    ellipse_C M.1 M.2 ∧
    ellipse_C N.1 N.2 ∧
    line_l k m M.1 M.2 ∧
    line_l k m N.1 N.2 ∧
    circle_passes_through_A M N →
    line_l k m (2/7) 0 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l836_83616


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l836_83614

theorem imaginary_part_of_z (i : ℂ) : i * i = -1 → Complex.im ((1 + i) / i) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l836_83614


namespace NUMINAMATH_CALUDE_f_7_equals_neg_2_l836_83686

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_4 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

theorem f_7_equals_neg_2 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_periodic : is_periodic_4 f) 
  (h_interval : ∀ x ∈ Set.Ioo 0 2, f x = 2 * x^2) : 
  f 7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_7_equals_neg_2_l836_83686


namespace NUMINAMATH_CALUDE_partner_c_profit_l836_83651

/-- Represents a business partner --/
inductive Partner
| A
| B
| C
| D

/-- Calculates the profit for a given partner in the first year of the business --/
def calculateProfit (totalProfit : ℕ) (partnerShares : Partner → ℕ) (dJoinTime : ℚ) (partner : Partner) : ℚ :=
  let fullYearShares := partnerShares Partner.A + partnerShares Partner.B + partnerShares Partner.C
  let dAdjustedShare := (partnerShares Partner.D : ℚ) * dJoinTime
  let totalAdjustedShares := (fullYearShares : ℚ) + dAdjustedShare
  let sharePerPart := (totalProfit : ℚ) / totalAdjustedShares
  sharePerPart * (partnerShares partner : ℚ)

/-- Theorem stating that partner C's profit is $20,250 given the problem conditions --/
theorem partner_c_profit :
  let totalProfit : ℕ := 56700
  let partnerShares : Partner → ℕ := fun
    | Partner.A => 7
    | Partner.B => 9
    | Partner.C => 10
    | Partner.D => 4
  let dJoinTime : ℚ := 1/2
  calculateProfit totalProfit partnerShares dJoinTime Partner.C = 20250 := by
  sorry


end NUMINAMATH_CALUDE_partner_c_profit_l836_83651


namespace NUMINAMATH_CALUDE_complex_number_location_l836_83661

theorem complex_number_location (z : ℂ) (h : (1 + 2*I)/z = I) : 
  z = 2/5 + 1/5*I ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l836_83661
