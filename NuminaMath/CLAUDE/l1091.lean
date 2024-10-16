import Mathlib

namespace NUMINAMATH_CALUDE_infinite_geometric_series_common_ratio_l1091_109123

theorem infinite_geometric_series_common_ratio 
  (a : ℝ) (S : ℝ) (h1 : a = 500) (h2 : S = 2500) :
  let r := 1 - a / S
  r = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_common_ratio_l1091_109123


namespace NUMINAMATH_CALUDE_stratified_sampling_most_suitable_l1091_109120

-- Define the land types
inductive LandType
  | Mountainous
  | Hilly
  | Flat
  | LowLying

-- Define the village structure
structure Village where
  landAreas : LandType → ℕ
  totalArea : ℕ
  sampleSize : ℕ

-- Define the sampling methods
inductive SamplingMethod
  | Drawing
  | RandomNumberTable
  | Systematic
  | Stratified

-- Define the suitability of a sampling method
def isSuitable (v : Village) (m : SamplingMethod) : Prop :=
  m = SamplingMethod.Stratified

-- Theorem statement
theorem stratified_sampling_most_suitable (v : Village) 
  (h1 : v.landAreas LandType.Mountainous = 8000)
  (h2 : v.landAreas LandType.Hilly = 12000)
  (h3 : v.landAreas LandType.Flat = 24000)
  (h4 : v.landAreas LandType.LowLying = 4000)
  (h5 : v.totalArea = 48000)
  (h6 : v.sampleSize = 480) :
  isSuitable v SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_suitable_l1091_109120


namespace NUMINAMATH_CALUDE_reverse_digits_square_diff_l1091_109185

/-- Given two-digit integers x and y where y is the reverse of x, and x^2 - y^2 = m^2 for some positive integer m, prove that x + y + m = 154 -/
theorem reverse_digits_square_diff (x y m : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a ∧ 0 < a ∧ a < 10 ∧ 0 < b ∧ b < 10) →  -- y is obtained by reversing the digits of x
  (x^2 - y^2 = m^2) →  -- x^2 - y^2 = m^2
  (0 < m) →  -- m is positive
  (x + y + m = 154) := by
sorry

end NUMINAMATH_CALUDE_reverse_digits_square_diff_l1091_109185


namespace NUMINAMATH_CALUDE_no_unique_solution_l1091_109192

/-- 
Theorem: The system of equations 4(3x + 4y) = 48 and kx + 12y = 30 
does not have a unique solution if and only if k = -9.
-/
theorem no_unique_solution (k : ℝ) : 
  (∀ x y : ℝ, 4*(3*x + 4*y) = 48 ∧ k*x + 12*y = 30) → 
  (¬∃! (x y : ℝ), 4*(3*x + 4*y) = 48 ∧ k*x + 12*y = 30) ↔ 
  k = -9 :=
sorry


end NUMINAMATH_CALUDE_no_unique_solution_l1091_109192


namespace NUMINAMATH_CALUDE_tan_45_degrees_l1091_109109

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l1091_109109


namespace NUMINAMATH_CALUDE_total_addresses_l1091_109169

/-- The number of commencement addresses given by each governor -/
structure GovernorAddresses where
  sandoval : ℕ
  hawkins : ℕ
  sloan : ℕ
  davenport : ℕ
  adkins : ℕ

/-- The conditions of the problem -/
def problem_conditions (g : GovernorAddresses) : Prop :=
  g.sandoval = 12 ∧
  g.hawkins = g.sandoval / 2 ∧
  g.sloan = g.sandoval + 10 ∧
  g.davenport = (g.sandoval + g.sloan) / 2 - 3 ∧
  g.adkins = g.hawkins + g.davenport + 2

/-- The theorem to be proved -/
theorem total_addresses (g : GovernorAddresses) :
  problem_conditions g →
  g.sandoval + g.hawkins + g.sloan + g.davenport + g.adkins = 70 :=
by sorry

end NUMINAMATH_CALUDE_total_addresses_l1091_109169


namespace NUMINAMATH_CALUDE_max_consecutive_semi_primes_correct_l1091_109174

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_semi_prime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p + q

def max_consecutive_semi_primes : ℕ := 5

theorem max_consecutive_semi_primes_correct :
  (∀ n : ℕ, ∃ m : ℕ, m ≥ n ∧
    (∀ k : ℕ, k < max_consecutive_semi_primes → is_semi_prime (m + k))) ∧
  (∀ n : ℕ, ¬∃ m : ℕ, 
    (∀ k : ℕ, k < max_consecutive_semi_primes + 1 → is_semi_prime (m + k))) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_semi_primes_correct_l1091_109174


namespace NUMINAMATH_CALUDE_molly_current_age_l1091_109170

/-- Represents the ages of Sandy, Molly, and Danny -/
structure Ages where
  sandy : ℕ
  molly : ℕ
  danny : ℕ

/-- The ratio of ages is 4:3:5 -/
def age_ratio (a : Ages) : Prop :=
  ∃ (x : ℕ), a.sandy = 4 * x ∧ a.molly = 3 * x ∧ a.danny = 5 * x

/-- Sandy's age after 6 years is 30 -/
def sandy_future_age (a : Ages) : Prop :=
  a.sandy + 6 = 30

/-- Theorem stating that under the given conditions, Molly's current age is 18 -/
theorem molly_current_age (a : Ages) :
  age_ratio a → sandy_future_age a → a.molly = 18 := by
  sorry


end NUMINAMATH_CALUDE_molly_current_age_l1091_109170


namespace NUMINAMATH_CALUDE_fraction_calculation_l1091_109190

theorem fraction_calculation : (3/8) / (4/9) + 1/6 = 97/96 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1091_109190


namespace NUMINAMATH_CALUDE_parcel_weight_l1091_109181

theorem parcel_weight (x y z : ℝ) 
  (h1 : x + y = 110) 
  (h2 : y + z = 140) 
  (h3 : z + x = 130) : 
  x + y + z = 190 := by
sorry

end NUMINAMATH_CALUDE_parcel_weight_l1091_109181


namespace NUMINAMATH_CALUDE_smallest_number_with_all_factors_l1091_109189

def alice_number : ℕ := 24

-- Function to check if a number has all prime factors of another number
def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n) → (p ∣ m)

theorem smallest_number_with_all_factors :
  ∃ (bob_number : ℕ), 
    bob_number > 0 ∧ 
    has_all_prime_factors alice_number bob_number ∧
    (∀ k : ℕ, k > 0 → has_all_prime_factors alice_number k → bob_number ≤ k) ∧
    bob_number = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_all_factors_l1091_109189


namespace NUMINAMATH_CALUDE_billion_scientific_notation_l1091_109163

/-- Represents 1.2 billion in decimal form -/
def billion : ℝ := 1200000000

/-- Represents 1.2 × 10^8 in scientific notation -/
def scientific_notation : ℝ := 1.2 * (10^8)

/-- Theorem stating that 1.2 billion is equal to 1.2 × 10^8 in scientific notation -/
theorem billion_scientific_notation : billion = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_billion_scientific_notation_l1091_109163


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l1091_109114

/-- Given two concentric circles with areas A1 and A1 + A2, where the larger circle
    has radius 5 and A1, A2, A1 + A2 form an arithmetic progression,
    prove that the radius of the smaller circle is 5√2/2 -/
theorem smaller_circle_radius
  (A1 A2 : ℝ)
  (h1 : A1 > 0)
  (h2 : A2 > 0)
  (h3 : (A1 + A2) = π * 5^2)
  (h4 : A2 = (A1 + (A1 + A2)) / 2)
  : ∃ (r : ℝ), r > 0 ∧ A1 = π * r^2 ∧ r = 5 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l1091_109114


namespace NUMINAMATH_CALUDE_finite_decimal_fraction_l1091_109101

theorem finite_decimal_fraction (n : ℕ) : 
  (∃ (k m : ℕ), n * (2 * n - 1) = 2^k * 5^m) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_finite_decimal_fraction_l1091_109101


namespace NUMINAMATH_CALUDE_fixed_points_of_f_squared_l1091_109151

def X := ℤ × ℤ × ℤ

def f (x : X) : X :=
  let (a, b, c) := x
  (a + b + c, a * b + b * c + c * a, a * b * c)

theorem fixed_points_of_f_squared (a b c : ℤ) :
  f (f (a, b, c)) = (a, b, c) ↔ 
    ((∃ k : ℤ, (a, b, c) = (k, 0, 0)) ∨ (a, b, c) = (-1, -1, 1)) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_squared_l1091_109151


namespace NUMINAMATH_CALUDE_dog_park_problem_l1091_109198

theorem dog_park_problem (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_eared_dogs : ℕ) :
  spotted_dogs = 15 →
  2 * spotted_dogs = total_dogs →
  5 * pointy_eared_dogs = total_dogs →
  pointy_eared_dogs = 6 := by
  sorry

end NUMINAMATH_CALUDE_dog_park_problem_l1091_109198


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l1091_109149

/-- The area of a rhombus formed by two equilateral triangles in a square -/
theorem rhombus_area_in_square (square_side : ℝ) (h_square_side : square_side = 4) :
  let triangle_height := square_side * (Real.sqrt 3) / 2
  let vertical_overlap := 2 * triangle_height - square_side
  let rhombus_area := (vertical_overlap * square_side) / 2
  rhombus_area = 8 * Real.sqrt 3 - 8 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_in_square_l1091_109149


namespace NUMINAMATH_CALUDE_opposite_of_2022_l1091_109172

-- Define the opposite of an integer
def opposite (n : ℤ) : ℤ := -n

-- Theorem stating that the opposite of 2022 is -2022
theorem opposite_of_2022 : opposite 2022 = -2022 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2022_l1091_109172


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l1091_109111

theorem factorization_of_quadratic (a : ℝ) : a^2 + 3*a = a*(a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l1091_109111


namespace NUMINAMATH_CALUDE_ship_passengers_ship_passengers_proof_l1091_109155

theorem ship_passengers : ℕ → Prop :=
  fun total_passengers =>
    (total_passengers : ℚ) = (1 / 12 + 1 / 4 + 1 / 9 + 1 / 6) * total_passengers + 42 →
    total_passengers = 108

-- Proof
theorem ship_passengers_proof : ship_passengers 108 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_ship_passengers_proof_l1091_109155


namespace NUMINAMATH_CALUDE_pole_not_perpendicular_l1091_109182

theorem pole_not_perpendicular (h : Real) (d : Real) (c : Real) 
  (h_val : h = 1.4)
  (d_val : d = 2)
  (c_val : c = 2.5) : 
  h^2 + d^2 ≠ c^2 := by
  sorry

end NUMINAMATH_CALUDE_pole_not_perpendicular_l1091_109182


namespace NUMINAMATH_CALUDE_dima_lives_on_seventh_floor_l1091_109108

/-- Represents the floor where Dima lives -/
def dimas_floor : ℕ := 7

/-- Represents the highest floor button Dima can reach -/
def max_reachable_floor : ℕ := 6

/-- The number of stories in the building -/
def building_stories : ℕ := 9

/-- Time (in seconds) it takes to descend from Dima's floor to the first floor -/
def descent_time : ℕ := 60

/-- Total time (in seconds) for the upward journey -/
def ascent_time : ℕ := 70

/-- Proposition stating that Dima lives on the 7th floor given the conditions -/
theorem dima_lives_on_seventh_floor :
  dimas_floor = 7 ∧
  max_reachable_floor = 6 ∧
  building_stories = 9 ∧
  descent_time = 60 ∧
  ascent_time = 70 ∧
  (5 * dimas_floor = 6 * max_reachable_floor + 1) :=
by sorry

end NUMINAMATH_CALUDE_dima_lives_on_seventh_floor_l1091_109108


namespace NUMINAMATH_CALUDE_median_triangle_area_l1091_109194

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (ma mb mc : ℝ)
  (area : ℝ)

-- Define the triangle formed by medians
def MedianTriangle (t : Triangle) : Triangle :=
  { a := t.ma,
    b := t.mb,
    c := t.mc,
    ma := 0,  -- We don't need these values for the median triangle
    mb := 0,
    mc := 0,
    area := 0 }  -- We'll prove this is 3/4 * t.area

-- Theorem statement
theorem median_triangle_area (t : Triangle) :
  (MedianTriangle t).area = 3/4 * t.area :=
sorry

end NUMINAMATH_CALUDE_median_triangle_area_l1091_109194


namespace NUMINAMATH_CALUDE_fifth_inequality_holds_l1091_109147

theorem fifth_inequality_holds : 
  1 + (1 : ℝ) / 2^2 + 1 / 3^2 + 1 / 4^2 + 1 / 5^2 + 1 / 6^2 < (2 * 5 + 1) / (5 + 1) :=
by sorry

end NUMINAMATH_CALUDE_fifth_inequality_holds_l1091_109147


namespace NUMINAMATH_CALUDE_combined_mixture_ratio_l1091_109140

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Combines two mixtures -/
def combine_mixtures (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk,
    water := m1.water + m2.water }

/-- Calculates the ratio of milk to water in a mixture -/
def milk_water_ratio (m : Mixture) : ℚ × ℚ :=
  (m.milk, m.water)

theorem combined_mixture_ratio :
  let m1 : Mixture := { milk := 4, water := 1 }
  let m2 : Mixture := { milk := 7, water := 3 }
  let combined := combine_mixtures m1 m2
  milk_water_ratio combined = (11, 4) := by
  sorry

end NUMINAMATH_CALUDE_combined_mixture_ratio_l1091_109140


namespace NUMINAMATH_CALUDE_inequality_not_true_l1091_109110

theorem inequality_not_true : Real.sqrt 2 + Real.sqrt 10 ≤ 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_true_l1091_109110


namespace NUMINAMATH_CALUDE_power_of_product_l1091_109156

theorem power_of_product (a b : ℝ) : (b^2 * a)^3 = a^3 * b^6 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l1091_109156


namespace NUMINAMATH_CALUDE_composite_number_l1091_109179

theorem composite_number (n : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 6 * 2^(2^(4*n)) + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_number_l1091_109179


namespace NUMINAMATH_CALUDE_exchange_candies_l1091_109154

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem exchange_candies : choose 7 5 * choose 9 5 = 2646 := by
  sorry

end NUMINAMATH_CALUDE_exchange_candies_l1091_109154


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_36_495_l1091_109129

theorem gcd_lcm_sum_36_495 : Nat.gcd 36 495 + Nat.lcm 36 495 = 1989 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_36_495_l1091_109129


namespace NUMINAMATH_CALUDE_square_rolling_octagon_l1091_109159

/-- Represents the faces of a square -/
inductive SquareFace
  | Left
  | Right
  | Top
  | Bottom

/-- Represents the rotation of a square -/
def squareRotation (n : ℕ) : ℕ := n * 135

/-- The final position of an object on a square face after rolling around an octagon -/
def finalPosition (initialFace : SquareFace) : SquareFace :=
  match (squareRotation 4) % 360 with
  | 180 => match initialFace with
    | SquareFace.Left => SquareFace.Right
    | SquareFace.Right => SquareFace.Left
    | SquareFace.Top => SquareFace.Bottom
    | SquareFace.Bottom => SquareFace.Top
  | _ => initialFace

theorem square_rolling_octagon :
  finalPosition SquareFace.Left = SquareFace.Right :=
by sorry

end NUMINAMATH_CALUDE_square_rolling_octagon_l1091_109159


namespace NUMINAMATH_CALUDE_sandwich_problem_l1091_109157

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 349/100

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 87/100

/-- The number of sodas bought -/
def num_sodas : ℕ := 4

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 1046/100

/-- The number of sandwiches bought -/
def num_sandwiches : ℕ := 2

theorem sandwich_problem :
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = total_cost :=
sorry

end NUMINAMATH_CALUDE_sandwich_problem_l1091_109157


namespace NUMINAMATH_CALUDE_button_probability_l1091_109128

theorem button_probability (initial_red : ℕ) (initial_blue : ℕ) 
  (removed_red : ℕ) (removed_blue : ℕ) :
  initial_red = 8 →
  initial_blue = 12 →
  removed_red = removed_blue →
  (initial_red + initial_blue - removed_red - removed_blue : ℚ) = 
    (5 / 8 : ℚ) * (initial_red + initial_blue : ℚ) →
  ((initial_red - removed_red : ℚ) / (initial_red + initial_blue - removed_red - removed_blue : ℚ)) *
  (removed_red : ℚ) / (removed_red + removed_blue : ℚ) = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_button_probability_l1091_109128


namespace NUMINAMATH_CALUDE_max_value_difference_l1091_109144

noncomputable def f (x : ℝ) := x^3 - 3*x^2 - x + 1

theorem max_value_difference (x₀ m : ℝ) : 
  (∀ x, f x ≤ f x₀) →  -- f attains maximum at x₀
  m ≠ x₀ →             -- m is not equal to x₀
  f x₀ = f m →         -- f(x₀) = f(m)
  |m - x₀| = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_difference_l1091_109144


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l1091_109139

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Two nested convex polygons Q1 and Q2 -/
structure NestedPolygons where
  Q1 : ConvexPolygon
  Q2 : ConvexPolygon
  m : ℕ
  h_m_ge_3 : m ≥ 3
  h_Q1_sides : Q1.sides = m
  h_Q2_sides : Q2.sides = 2 * m
  h_nested : Bool
  h_no_shared_segment : Bool
  h_both_convex : Q1.convex ∧ Q2.convex

/-- The maximum number of intersections between two nested convex polygons -/
def max_intersections (np : NestedPolygons) : ℕ := 2 * np.m^2

/-- Theorem stating the maximum number of intersections -/
theorem max_intersections_theorem (np : NestedPolygons) :
  max_intersections np = 2 * np.m^2 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l1091_109139


namespace NUMINAMATH_CALUDE_problem_solution_l1091_109161

theorem problem_solution : 
  let M : ℤ := 2007 / 3
  let N : ℤ := M / 3
  let X : ℤ := M - N
  X = 446 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1091_109161


namespace NUMINAMATH_CALUDE_total_lines_for_given_conditions_l1091_109195

/-- Given a number of intersections, crosswalks per intersection, and lines per crosswalk,
    calculate the total number of lines across all crosswalks in all intersections. -/
def total_lines (intersections : ℕ) (crosswalks_per_intersection : ℕ) (lines_per_crosswalk : ℕ) : ℕ :=
  intersections * crosswalks_per_intersection * lines_per_crosswalk

/-- Prove that for 10 intersections, each with 8 crosswalks, and each crosswalk having 30 lines,
    the total number of lines is 2400. -/
theorem total_lines_for_given_conditions :
  total_lines 10 8 30 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_total_lines_for_given_conditions_l1091_109195


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1091_109118

-- Define the set of numbers
def S : Set Nat := {n | 1 ≤ n ∧ n ≤ 9}

-- Define a type for a selection of three numbers
structure Selection :=
  (a b c : Nat)
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (inS : a ∈ S ∧ b ∈ S ∧ c ∈ S)

-- Define events
def allEven (s : Selection) : Prop := s.a % 2 = 0 ∧ s.b % 2 = 0 ∧ s.c % 2 = 0
def allOdd (s : Selection) : Prop := s.a % 2 = 1 ∧ s.b % 2 = 1 ∧ s.c % 2 = 1
def oneEvenTwoOdd (s : Selection) : Prop :=
  (s.a % 2 = 0 ∧ s.b % 2 = 1 ∧ s.c % 2 = 1) ∨
  (s.a % 2 = 1 ∧ s.b % 2 = 0 ∧ s.c % 2 = 1) ∨
  (s.a % 2 = 1 ∧ s.b % 2 = 1 ∧ s.c % 2 = 0)
def twoEvenOneOdd (s : Selection) : Prop :=
  (s.a % 2 = 0 ∧ s.b % 2 = 0 ∧ s.c % 2 = 1) ∨
  (s.a % 2 = 0 ∧ s.b % 2 = 1 ∧ s.c % 2 = 0) ∨
  (s.a % 2 = 1 ∧ s.b % 2 = 0 ∧ s.c % 2 = 0)

-- Define mutual exclusivity and complementarity
def mutuallyExclusive (e1 e2 : Selection → Prop) : Prop :=
  ∀ s : Selection, ¬(e1 s ∧ e2 s)

def complementary (e1 e2 : Selection → Prop) : Prop :=
  ∀ s : Selection, e1 s ∨ e2 s

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  (mutuallyExclusive allEven allOdd ∧ ¬complementary allEven allOdd) ∧
  (mutuallyExclusive oneEvenTwoOdd twoEvenOneOdd ∧ ¬complementary oneEvenTwoOdd twoEvenOneOdd) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1091_109118


namespace NUMINAMATH_CALUDE_metallic_sheet_volumes_l1091_109165

/-- Represents the dimensions of a rectangular sheet -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a square -/
structure SquareDimensions where
  side : ℝ

/-- Calculates the volume of open box A -/
def volume_box_a (sheet : SheetDimensions) (corner_cut : SquareDimensions) : ℝ :=
  (sheet.length - 2 * corner_cut.side) * (sheet.width - 2 * corner_cut.side) * corner_cut.side

/-- Calculates the volume of open box B -/
def volume_box_b (sheet : SheetDimensions) (corner_cut : SquareDimensions) (middle_cut : SquareDimensions) : ℝ :=
  ((sheet.length - 2 * corner_cut.side) * (sheet.width - 2 * corner_cut.side) - middle_cut.side ^ 2) * corner_cut.side

theorem metallic_sheet_volumes
  (sheet : SheetDimensions)
  (corner_cut : SquareDimensions)
  (middle_cut : SquareDimensions)
  (h1 : sheet.length = 48)
  (h2 : sheet.width = 36)
  (h3 : corner_cut.side = 8)
  (h4 : middle_cut.side = 12) :
  volume_box_a sheet corner_cut = 5120 ∧ volume_box_b sheet corner_cut middle_cut = 3968 := by
  sorry

#eval volume_box_a ⟨48, 36⟩ ⟨8⟩
#eval volume_box_b ⟨48, 36⟩ ⟨8⟩ ⟨12⟩

end NUMINAMATH_CALUDE_metallic_sheet_volumes_l1091_109165


namespace NUMINAMATH_CALUDE_alpha_range_l1091_109164

theorem alpha_range (α : Real) (h1 : 0 ≤ α) (h2 : α ≤ π) 
  (h3 : ∀ x : Real, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) :
  α ∈ Set.Icc 0 (π / 6) ∪ Set.Icc (5 * π / 6) π := by
  sorry

end NUMINAMATH_CALUDE_alpha_range_l1091_109164


namespace NUMINAMATH_CALUDE_gas_measurement_l1091_109103

/-- Represents the ratio of inches to liters per minute for liquid -/
def liquid_ratio : ℚ := 2.5 / 60

/-- Represents the movement ratio of gas compared to liquid -/
def gas_movement_ratio : ℚ := 1 / 2

/-- Represents the amount of gas that passed through the rotameter in liters -/
def gas_volume : ℚ := 192

/-- Calculates the number of inches measured for the gas phase -/
def gas_inches : ℚ := (gas_volume * liquid_ratio) / gas_movement_ratio

/-- Theorem stating that the number of inches measured for the gas phase is 4 -/
theorem gas_measurement :
  gas_inches = 4 := by sorry

end NUMINAMATH_CALUDE_gas_measurement_l1091_109103


namespace NUMINAMATH_CALUDE_pencil_distribution_remainder_l1091_109191

theorem pencil_distribution_remainder : 25197629 % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_remainder_l1091_109191


namespace NUMINAMATH_CALUDE_triangle_area_l1091_109115

/-- Given a triangle with perimeter 60 cm and inradius 2.5 cm, its area is 75 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 60 → inradius = 2.5 → area = perimeter / 2 * inradius → area = 75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1091_109115


namespace NUMINAMATH_CALUDE_dogs_sold_l1091_109171

theorem dogs_sold (cats : ℕ) (dogs : ℕ) (ratio : ℚ) : 
  ratio = 2 / 1 → cats = 16 → dogs = 8 := by
  sorry

end NUMINAMATH_CALUDE_dogs_sold_l1091_109171


namespace NUMINAMATH_CALUDE_parabola_chord_length_l1091_109137

/-- Given a parabola y^2 = 4x and a line passing through its focus intersecting 
    the parabola at points P(x₁, y₁) and Q(x₂, y₂) such that x₁ + x₂ = 6, 
    prove that the length |PQ| = 8. -/
theorem parabola_chord_length (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 4*x₁ →  -- P is on the parabola
  y₂^2 = 4*x₂ →  -- Q is on the parabola
  x₁ + x₂ = 6 →  -- Given condition
  (∃ t : ℝ, t*x₁ + (1-t)*1 = 0 ∧ t*y₁ = 0) →  -- Line PQ passes through focus (1,0)
  (∃ s : ℝ, s*x₂ + (1-s)*1 = 0 ∧ s*y₂ = 0) →  -- Line PQ passes through focus (1,0)
  ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2) = 8 :=  -- |PQ| = 8
by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l1091_109137


namespace NUMINAMATH_CALUDE_range_of_m_for_quadratic_inequality_l1091_109130

theorem range_of_m_for_quadratic_inequality (m : ℝ) : 
  m ≠ 0 → 
  (∀ x : ℝ, x ∈ Set.Icc 1 3 → m * x^2 - m * x - 1 < -m + 5) ↔ 
  m > 0 ∧ m < 6/7 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_quadratic_inequality_l1091_109130


namespace NUMINAMATH_CALUDE_circle_area_difference_l1091_109112

/-- The difference in areas between a circle with radius 25 inches and a circle with diameter 15 inches is 568.75π square inches. -/
theorem circle_area_difference : 
  let r1 : ℝ := 25
  let d2 : ℝ := 15
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 568.75 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1091_109112


namespace NUMINAMATH_CALUDE_sum_of_squares_reciprocals_l1091_109150

theorem sum_of_squares_reciprocals (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_reciprocals_l1091_109150


namespace NUMINAMATH_CALUDE_alissas_earrings_l1091_109153

/-- Proves that Alissa's total number of earrings is 36 given the problem conditions. -/
theorem alissas_earrings (pairs_bought : ℕ) (earrings_per_pair : ℕ) (alissa_initial : ℕ) : 
  pairs_bought = 12 → 
  earrings_per_pair = 2 → 
  alissa_initial + pairs_bought * earrings_per_pair / 2 = 3 * (pairs_bought * earrings_per_pair / 2) → 
  alissa_initial + pairs_bought * earrings_per_pair / 2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_alissas_earrings_l1091_109153


namespace NUMINAMATH_CALUDE_maximal_planar_iff_3n_minus_6_edges_l1091_109126

structure PlanarGraph where
  n : ℕ
  e : ℕ
  h_vertices : n ≥ 3

def is_maximal_planar (G : PlanarGraph) : Prop :=
  ∀ H : PlanarGraph, G.n = H.n → G.e ≥ H.e

theorem maximal_planar_iff_3n_minus_6_edges (G : PlanarGraph) :
  is_maximal_planar G ↔ G.e = 3 * G.n - 6 := by sorry

end NUMINAMATH_CALUDE_maximal_planar_iff_3n_minus_6_edges_l1091_109126


namespace NUMINAMATH_CALUDE_division_problem_l1091_109187

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 159 → quotient = 9 → remainder = 6 → 
  dividend = divisor * quotient + remainder → 
  divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1091_109187


namespace NUMINAMATH_CALUDE_line_through_origin_and_negative_one_l1091_109146

/-- The angle of inclination (in degrees) of a line passing through two points -/
def angleOfInclination (x1 y1 x2 y2 : ℝ) : ℝ := sorry

/-- A line passes through the origin (0, 0) and the point (-1, -1) -/
theorem line_through_origin_and_negative_one : 
  angleOfInclination 0 0 (-1) (-1) = 45 := by sorry

end NUMINAMATH_CALUDE_line_through_origin_and_negative_one_l1091_109146


namespace NUMINAMATH_CALUDE_brads_money_l1091_109152

theorem brads_money (total : ℚ) (josh_brad_ratio : ℚ) (josh_doug_ratio : ℚ)
  (h1 : total = 68)
  (h2 : josh_brad_ratio = 2)
  (h3 : josh_doug_ratio = 3/4) :
  ∃ (brad : ℚ), brad = 12 ∧ 
    ∃ (josh doug : ℚ), 
      josh = josh_brad_ratio * brad ∧
      josh = josh_doug_ratio * doug ∧
      josh + doug + brad = total :=
by sorry

end NUMINAMATH_CALUDE_brads_money_l1091_109152


namespace NUMINAMATH_CALUDE_circle_with_specific_intersection_l1091_109132

/-- A circle passing through a point with a specific chord of intersection. -/
theorem circle_with_specific_intersection (x y : ℝ) : 
  -- The equation of the circle we're proving
  x^2 + y^2 - 6*x - 4 = 0 ↔ 
  -- Conditions:
  -- 1. The circle passes through the point (0,2)
  (0^2 + 2^2 - 6*0 - 4 = 0) ∧ 
  -- 2. The chord of intersection with another circle lies on a specific line
  (∃ (t : ℝ), 
    -- Point (x,y) is on our circle
    (x^2 + y^2 - 6*x - 4 = 0) ∧ 
    -- Point (x,y) is on the other circle
    (x^2 + y^2 - x + 2*y - 3 = 0) ∧ 
    -- Point (x,y) is on the line
    (5*x + 2*y + 1 = 0) ∧ 
    -- t parameterizes all points on the line
    x = -((2*t + 1)/5) ∧ 
    y = t) :=
sorry

end NUMINAMATH_CALUDE_circle_with_specific_intersection_l1091_109132


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1091_109188

theorem fraction_evaluation : (15 : ℚ) / 45 - 2 / 9 + 1 / 4 * 8 / 3 = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1091_109188


namespace NUMINAMATH_CALUDE_trivia_team_absence_l1091_109183

theorem trivia_team_absence (total_members : ℕ) (points_per_member : ℕ) (total_score : ℕ) 
  (h1 : total_members = 14)
  (h2 : points_per_member = 5)
  (h3 : total_score = 35) :
  total_members - (total_score / points_per_member) = 7 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_absence_l1091_109183


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1091_109124

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 3/8) (h2 : x - y = 5/24) : x^2 - y^2 = 5/64 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1091_109124


namespace NUMINAMATH_CALUDE_unique_arrangement_l1091_109168

-- Define the containers and liquids as enumerated types
inductive Container : Type
| Cup : Container
| Glass : Container
| Jug : Container
| Jar : Container

inductive Liquid : Type
| Milk : Liquid
| Lemonade : Liquid
| Kvass : Liquid
| Water : Liquid

-- Define the arrangement as a function from Container to Liquid
def Arrangement := Container → Liquid

-- Define the conditions
def satisfiesConditions (arr : Arrangement) : Prop :=
  (arr Container.Cup ≠ Liquid.Water ∧ arr Container.Cup ≠ Liquid.Milk) ∧
  (∃ c, (c = Container.Jug ∨ c = Container.Jar) ∧
        arr c = Liquid.Kvass ∧
        (arr Container.Cup = Liquid.Lemonade ∨
         arr Container.Glass = Liquid.Lemonade)) ∧
  (arr Container.Jar ≠ Liquid.Lemonade ∧ arr Container.Jar ≠ Liquid.Water) ∧
  ((arr Container.Glass = Liquid.Milk ∧ arr Container.Jug = Liquid.Milk) ∨
   (arr Container.Glass = Liquid.Milk ∧ arr Container.Jar = Liquid.Milk) ∨
   (arr Container.Jug = Liquid.Milk ∧ arr Container.Jar = Liquid.Milk))

-- Define the correct arrangement
def correctArrangement : Arrangement
| Container.Cup => Liquid.Lemonade
| Container.Glass => Liquid.Water
| Container.Jug => Liquid.Milk
| Container.Jar => Liquid.Kvass

-- Theorem statement
theorem unique_arrangement :
  ∀ (arr : Arrangement), satisfiesConditions arr → arr = correctArrangement :=
by sorry

end NUMINAMATH_CALUDE_unique_arrangement_l1091_109168


namespace NUMINAMATH_CALUDE_friends_new_games_l1091_109162

theorem friends_new_games (katie_new : ℕ) (total_new : ℕ) (h1 : katie_new = 84) (h2 : total_new = 92) :
  total_new - katie_new = 8 := by
  sorry

end NUMINAMATH_CALUDE_friends_new_games_l1091_109162


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l1091_109117

theorem smallest_four_digit_mod_8 :
  ∀ n : ℕ, n ≥ 1000 ∧ n ≡ 5 [MOD 8] → n ≥ 1005 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l1091_109117


namespace NUMINAMATH_CALUDE_counterfeit_weight_equals_net_profit_l1091_109160

/-- Represents a dishonest dealer's selling strategy -/
structure DishonestDealer where
  /-- The percentage of impurities added to the product -/
  impurities : ℝ
  /-- The net profit percentage achieved by the dealer -/
  net_profit : ℝ

/-- Calculates the percentage by which the counterfeit weight is less than the real weight -/
def counterfeit_weight_percentage (dealer : DishonestDealer) : ℝ :=
  dealer.net_profit

/-- Theorem stating that under specific conditions, the counterfeit weight percentage
    equals the net profit percentage -/
theorem counterfeit_weight_equals_net_profit 
  (dealer : DishonestDealer) 
  (h1 : dealer.impurities = 35)
  (h2 : dealer.net_profit = 68.75) :
  counterfeit_weight_percentage dealer = 68.75 := by
  sorry

end NUMINAMATH_CALUDE_counterfeit_weight_equals_net_profit_l1091_109160


namespace NUMINAMATH_CALUDE_robbie_afternoon_rice_l1091_109122

/-- Represents the number of cups of rice Robbie eats at different times of the day and the fat content --/
structure RiceIntake where
  morning : ℕ
  evening : ℕ
  fat_per_cup : ℕ
  total_fat_per_week : ℕ

/-- Calculates the number of cups of rice Robbie eats in the afternoon --/
def afternoon_rice_cups (intake : RiceIntake) : ℕ :=
  (intake.total_fat_per_week - 7 * (intake.morning + intake.evening) * intake.fat_per_cup) / (7 * intake.fat_per_cup)

/-- Theorem stating that given the conditions, Robbie eats 14 cups of rice in the afternoon --/
theorem robbie_afternoon_rice 
  (intake : RiceIntake) 
  (h_morning : intake.morning = 3)
  (h_evening : intake.evening = 5)
  (h_fat_per_cup : intake.fat_per_cup = 10)
  (h_total_fat : intake.total_fat_per_week = 700) :
  afternoon_rice_cups intake = 14 := by
  sorry

end NUMINAMATH_CALUDE_robbie_afternoon_rice_l1091_109122


namespace NUMINAMATH_CALUDE_remainder_approximation_l1091_109106

/-- Given two positive real numbers satisfying certain conditions, 
    prove that the remainder of their division is approximately 15. -/
theorem remainder_approximation (L S : ℝ) (hL : L > 0) (hS : S > 0) 
    (h_diff : L - S = 1365)
    (h_approx : |L - 1542.857| < 0.001)
    (h_div : ∃ R : ℝ, R ≥ 0 ∧ L = 8 * S + R) : 
  ∃ R : ℝ, R ≥ 0 ∧ L = 8 * S + R ∧ |R - 15| < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_approximation_l1091_109106


namespace NUMINAMATH_CALUDE_jackson_charity_collection_l1091_109193

/-- Represents the problem of Jackson's charity collection --/
theorem jackson_charity_collection 
  (total_days : ℕ) 
  (goal : ℕ) 
  (monday_earning : ℕ) 
  (tuesday_earning : ℕ) 
  (houses_per_bundle : ℕ) 
  (earning_per_bundle : ℕ) 
  (h1 : total_days = 5)
  (h2 : goal = 1000)
  (h3 : monday_earning = 300)
  (h4 : tuesday_earning = 40)
  (h5 : houses_per_bundle = 4)
  (h6 : earning_per_bundle = 10) :
  ∃ (houses_per_day : ℕ), 
    houses_per_day = 88 ∧ 
    (goal - monday_earning - tuesday_earning) = 
      (total_days - 2) * houses_per_day * (earning_per_bundle / houses_per_bundle) :=
sorry

end NUMINAMATH_CALUDE_jackson_charity_collection_l1091_109193


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l1091_109175

/-- Proves that the ratio of a square pond's area to a rectangular field's area is 1:50 
    given specific dimensions -/
theorem pond_to_field_area_ratio 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (pond_side : ℝ) 
  (h1 : field_length = 80) 
  (h2 : field_width = 40) 
  (h3 : pond_side = 8) : 
  (pond_side ^ 2) / (field_length * field_width) = 1 / 50 := by
  sorry


end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l1091_109175


namespace NUMINAMATH_CALUDE_range_of_m_l1091_109166

theorem range_of_m (m : ℝ) : 
  (¬∀ (x : ℝ), m * x^2 - 2 * m * x + 1 > 0) → (m < 0 ∨ m ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1091_109166


namespace NUMINAMATH_CALUDE_sqrt2_minus_2_properties_l1091_109184

theorem sqrt2_minus_2_properties :
  let x : ℝ := Real.sqrt 2 - 2
  (- x = 2 - Real.sqrt 2) ∧ (|x| = 2 - Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_sqrt2_minus_2_properties_l1091_109184


namespace NUMINAMATH_CALUDE_jerry_payment_l1091_109196

/-- Calculates the total amount paid for Jerry's work given the following conditions:
  * Jerry's hourly rate
  * Time spent painting the house
  * Time spent fixing the kitchen counter (3 times the painting time)
  * Time spent mowing the lawn
-/
def total_amount_paid (rate : ℕ) (painting_time : ℕ) (mowing_time : ℕ) : ℕ :=
  rate * (painting_time + 3 * painting_time + mowing_time)

/-- Theorem stating that given the specific conditions of Jerry's work,
    the total amount paid is $570 -/
theorem jerry_payment : total_amount_paid 15 8 6 = 570 := by
  sorry

end NUMINAMATH_CALUDE_jerry_payment_l1091_109196


namespace NUMINAMATH_CALUDE_subset_ratio_for_ten_elements_l1091_109105

theorem subset_ratio_for_ten_elements : 
  let n : ℕ := 10
  let k : ℕ := 3
  let total_subsets : ℕ := 2^n
  let three_element_subsets : ℕ := n.choose k
  (three_element_subsets : ℚ) / total_subsets = 15 / 128 := by
  sorry

end NUMINAMATH_CALUDE_subset_ratio_for_ten_elements_l1091_109105


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1091_109180

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1091_109180


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l1091_109186

-- Define the curve equation
def curve (x y k : ℝ) : Prop := x^2 / (k + 4) + y^2 / (k - 1) = 1

-- Define what it means for the curve to be a hyperbola
def is_hyperbola (k : ℝ) : Prop := ∃ x y, curve x y k ∧ (k + 4) * (k - 1) < 0

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k → k ∈ Set.Ioo (-4 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l1091_109186


namespace NUMINAMATH_CALUDE_age_interchange_problem_l1091_109138

theorem age_interchange_problem :
  let valid_pair := λ (t n : ℕ) =>
    t > 30 ∧
    n > 0 ∧
    30 + n < 100 ∧
    t + n < 100 ∧
    (t + n) / 10 = (30 + n) % 10 ∧
    (t + n) % 10 = (30 + n) / 10
  (∃! l : List (ℕ × ℕ), l.length = 21 ∧ ∀ p ∈ l, valid_pair p.1 p.2) :=
by sorry

end NUMINAMATH_CALUDE_age_interchange_problem_l1091_109138


namespace NUMINAMATH_CALUDE_complex_number_modulus_l1091_109119

theorem complex_number_modulus (a : ℝ) (i : ℂ) : 
  a < 0 → 
  i * i = -1 → 
  Complex.abs (a * i / (1 + 2 * i)) = Real.sqrt 5 → 
  a = -5 := by
sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l1091_109119


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1091_109127

theorem solution_satisfies_system : ∃ (a b c : ℝ), 
  (a^3 + 3*a*b^2 + 3*a*c^2 - 6*a*b*c = 1) ∧
  (b^3 + 3*b*a^2 + 3*b*c^2 - 6*a*b*c = 1) ∧
  (c^3 + 3*c*a^2 + 3*c*b^2 - 6*a*b*c = 1) ∧
  (a = 1 ∧ b = 1 ∧ c = 1) := by
sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1091_109127


namespace NUMINAMATH_CALUDE_dragon_poker_partitions_l1091_109143

/-- The number of suits in the deck -/
def num_suits : ℕ := 4

/-- The target score to achieve -/
def target_score : ℕ := 2018

/-- The number of ways to partition the target score into exactly num_suits non-negative integers -/
def num_partitions : ℕ := (target_score + num_suits - 1).choose (num_suits - 1)

theorem dragon_poker_partitions :
  num_partitions = 1373734330 := by sorry

end NUMINAMATH_CALUDE_dragon_poker_partitions_l1091_109143


namespace NUMINAMATH_CALUDE_video_games_spending_l1091_109125

def total_allowance : ℚ := 50

def books_fraction : ℚ := 1/7
def video_games_fraction : ℚ := 2/7
def snacks_fraction : ℚ := 1/2
def clothes_fraction : ℚ := 3/14

def video_games_spent : ℚ := total_allowance * video_games_fraction

theorem video_games_spending :
  video_games_spent = 7.15 := by sorry

end NUMINAMATH_CALUDE_video_games_spending_l1091_109125


namespace NUMINAMATH_CALUDE_science_fair_ratio_l1091_109177

/-- Represents the number of adults and children at the science fair -/
structure Attendance where
  adults : ℕ
  children : ℕ

/-- Calculates the total fee collected given an attendance -/
def totalFee (a : Attendance) : ℕ := 30 * a.adults + 15 * a.children

/-- Calculates the ratio of adults to children -/
def ratio (a : Attendance) : ℚ := a.adults / a.children

theorem science_fair_ratio : 
  ∃ (a : Attendance), 
    a.adults ≥ 1 ∧ 
    a.children ≥ 1 ∧ 
    totalFee a = 2250 ∧ 
    ∀ (b : Attendance), 
      b.adults ≥ 1 → 
      b.children ≥ 1 → 
      totalFee b = 2250 → 
      |ratio a - 2| ≤ |ratio b - 2| := by
  sorry

end NUMINAMATH_CALUDE_science_fair_ratio_l1091_109177


namespace NUMINAMATH_CALUDE_max_planes_four_points_l1091_109173

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in three-dimensional space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Function to determine the number of planes formed by four points -/
def numPlanes (p1 p2 p3 p4 : Point3D) : ℕ := sorry

/-- Theorem stating the maximum number of planes determined by four points -/
theorem max_planes_four_points :
  ∃ (p1 p2 p3 p4 : Point3D), numPlanes p1 p2 p3 p4 = 4 ∧
  ∀ (q1 q2 q3 q4 : Point3D), numPlanes q1 q2 q3 q4 ≤ 4 := by sorry

end NUMINAMATH_CALUDE_max_planes_four_points_l1091_109173


namespace NUMINAMATH_CALUDE_M_subset_P_l1091_109133

open Set

def M : Set ℝ := {x | x > 1}
def P : Set ℝ := {x | x^2 > 1}

theorem M_subset_P : M ⊆ P := by sorry

end NUMINAMATH_CALUDE_M_subset_P_l1091_109133


namespace NUMINAMATH_CALUDE_complex_number_problem_l1091_109131

theorem complex_number_problem (α β : ℂ) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ α + β = x ∧ Complex.I * (2 * α - β) = y) →
  β = 4 + 3 * Complex.I →
  α = 2 - 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1091_109131


namespace NUMINAMATH_CALUDE_current_babysitter_rate_is_16_l1091_109167

/-- Represents the babysitting scenario with given conditions -/
structure BabysittingScenario where
  new_hourly_rate : ℕ
  scream_charge : ℕ
  hours : ℕ
  scream_count : ℕ
  cost_difference : ℕ

/-- Calculates the hourly rate of the current babysitter -/
def current_babysitter_rate (scenario : BabysittingScenario) : ℕ :=
  ((scenario.new_hourly_rate * scenario.hours + scenario.scream_charge * scenario.scream_count) + scenario.cost_difference) / scenario.hours

/-- Theorem stating that given the conditions, the current babysitter's hourly rate is $16 -/
theorem current_babysitter_rate_is_16 (scenario : BabysittingScenario) 
    (h1 : scenario.new_hourly_rate = 12)
    (h2 : scenario.scream_charge = 3)
    (h3 : scenario.hours = 6)
    (h4 : scenario.scream_count = 2)
    (h5 : scenario.cost_difference = 18) :
  current_babysitter_rate scenario = 16 := by
  sorry

#eval current_babysitter_rate { new_hourly_rate := 12, scream_charge := 3, hours := 6, scream_count := 2, cost_difference := 18 }

end NUMINAMATH_CALUDE_current_babysitter_rate_is_16_l1091_109167


namespace NUMINAMATH_CALUDE_gcd_problem_l1091_109134

-- Define the custom GCD operation
def customGCD (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the problem statement
theorem gcd_problem : customGCD (customGCD 20 16 * customGCD 18 24) (customGCD 20 16 * customGCD 18 24) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1091_109134


namespace NUMINAMATH_CALUDE_correct_operation_l1091_109100

theorem correct_operation (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1091_109100


namespace NUMINAMATH_CALUDE_complex_solutions_count_l1091_109141

/-- The equation (z^3 - 1) / (z^2 + z - 6) = 0 has exactly 3 complex solutions. -/
theorem complex_solutions_count : ∃ (S : Finset ℂ), 
  (∀ z ∈ S, (z^3 - 1) / (z^2 + z - 6) = 0) ∧ 
  (∀ z : ℂ, (z^3 - 1) / (z^2 + z - 6) = 0 → z ∈ S) ∧
  Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_solutions_count_l1091_109141


namespace NUMINAMATH_CALUDE_rectangle_containment_exists_l1091_109148

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : Nat
  height : Nat

/-- The set of all rectangles with positive integer dimensions -/
def RectangleSet : Set Rectangle :=
  {r : Rectangle | r.width > 0 ∧ r.height > 0}

/-- Predicate to check if one rectangle is contained within another -/
def contains (r1 r2 : Rectangle) : Prop :=
  r1.width ≤ r2.width ∧ r1.height ≤ r2.height

theorem rectangle_containment_exists :
  ∃ r1 r2 : Rectangle, r1 ∈ RectangleSet ∧ r2 ∈ RectangleSet ∧ r1 ≠ r2 ∧ contains r1 r2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_containment_exists_l1091_109148


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1091_109104

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1091_109104


namespace NUMINAMATH_CALUDE_caterpillar_problem_l1091_109107

/-- The number of caterpillars remaining on a tree after population changes. -/
def caterpillarsRemaining (initial : ℕ) (hatched : ℕ) (left : ℕ) : ℕ :=
  initial + hatched - left

/-- Theorem stating that given the specific numbers in the problem, 
    the result is 10 caterpillars. -/
theorem caterpillar_problem : 
  caterpillarsRemaining 14 4 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_caterpillar_problem_l1091_109107


namespace NUMINAMATH_CALUDE_student_b_speed_l1091_109176

theorem student_b_speed (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 12 →
  speed_ratio = 1.2 →
  time_difference = 1/6 →
  ∃ (speed_b : ℝ),
    distance / speed_b - time_difference = distance / (speed_ratio * speed_b) ∧
    speed_b = 12 :=
by sorry

end NUMINAMATH_CALUDE_student_b_speed_l1091_109176


namespace NUMINAMATH_CALUDE_smallest_n_for_digit_rearrangement_l1091_109178

/-- Represents a natural number as a list of its digits -/
def Digits : Type := List Nat

/-- Returns true if two lists of digits represent numbers that differ by a sequence of n ones -/
def differsBy111 (a b : Digits) (n : Nat) : Prop := sorry

/-- Returns true if two lists of digits are permutations of each other -/
def isPermutation (a b : Digits) : Prop := sorry

/-- Theorem: The smallest n for which there exist two numbers A and B,
    where B is a permutation of A's digits and A - B is n ones, is 9 -/
theorem smallest_n_for_digit_rearrangement :
  ∃ (a b : Digits),
    isPermutation a b ∧
    differsBy111 a b 9 ∧
    ∀ (n : Nat), n < 9 →
      ¬∃ (x y : Digits), isPermutation x y ∧ differsBy111 x y n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_digit_rearrangement_l1091_109178


namespace NUMINAMATH_CALUDE_even_increasing_negative_ordering_l1091_109199

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f is increasing on (-∞, 0) if f(x) < f(y) whenever x < y < 0 -/
def IncreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → y < 0 → f x < f y

theorem even_increasing_negative_ordering (f : ℝ → ℝ) 
    (h_even : IsEven f) (h_incr : IncreasingOnNegative f) : 
    f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_negative_ordering_l1091_109199


namespace NUMINAMATH_CALUDE_expansion_properties_l1091_109197

-- Define the expansion of (x-m)^7
def expansion (x m : ℝ) (a : Fin 8 → ℝ) : Prop :=
  (x - m)^7 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7

-- State the theorem
theorem expansion_properties {m : ℝ} {a : Fin 8 → ℝ} 
  (h_expansion : ∀ x, expansion x m a)
  (h_coeff : a 4 = -35) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 1) ∧
  (a 1 + a 3 + a 5 + a 7 = 26) := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l1091_109197


namespace NUMINAMATH_CALUDE_number_problem_l1091_109121

theorem number_problem : ∃ x : ℝ, (x / 6) * 12 = 13 ∧ x = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1091_109121


namespace NUMINAMATH_CALUDE_min_value_product_squares_l1091_109142

theorem min_value_product_squares (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 3) :
  x^2 * y^2 * z^2 ≥ 1/64 ∧ ∃ (a : ℝ), a > 0 ∧ a^2 * a^2 * a^2 = 1/64 ∧ 1/a + 1/a + 1/a = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_squares_l1091_109142


namespace NUMINAMATH_CALUDE_smallest_period_scaled_l1091_109135

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x - p) = f x

theorem smallest_period_scaled (f : ℝ → ℝ) (h : is_periodic f 30) :
  ∃ b : ℝ, b > 0 ∧ (∀ x, f ((x - b) / 3) = f (x / 3)) ∧
  ∀ b' : ℝ, 0 < b' ∧ (∀ x, f ((x - b') / 3) = f (x / 3)) → b ≤ b' :=
sorry

end NUMINAMATH_CALUDE_smallest_period_scaled_l1091_109135


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1091_109158

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1091_109158


namespace NUMINAMATH_CALUDE_smallest_k_for_equalization_l1091_109116

/-- Represents the state of gas cylinders -/
def CylinderState := List ℝ

/-- Represents a connection operation on cylinders -/
def Connection := List Nat

/-- Applies a single connection operation to a cylinder state -/
def applyConnection (state : CylinderState) (conn : Connection) : CylinderState :=
  sorry

/-- Checks if all pressures in a state are equal -/
def isEqualized (state : CylinderState) : Prop :=
  sorry

/-- Checks if a connection is valid (size ≤ k) -/
def isValidConnection (conn : Connection) (k : ℕ) : Prop :=
  sorry

/-- Represents a sequence of connection operations -/
def EqualizationProcess := List Connection

/-- Checks if an equalization process is valid for a given k -/
def isValidProcess (process : EqualizationProcess) (k : ℕ) : Prop :=
  sorry

/-- Applies an equalization process to a cylinder state -/
def applyProcess (state : CylinderState) (process : EqualizationProcess) : CylinderState :=
  sorry

/-- Main theorem: 5 is the smallest k that allows equalization -/
theorem smallest_k_for_equalization :
  (∀ (initial : CylinderState), initial.length = 40 →
    ∃ (process : EqualizationProcess), 
      isValidProcess process 5 ∧ 
      isEqualized (applyProcess initial process)) ∧
  (∀ k < 5, ∃ (initial : CylinderState), initial.length = 40 ∧
    ∀ (process : EqualizationProcess), 
      isValidProcess process k → 
      ¬isEqualized (applyProcess initial process)) :=
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_equalization_l1091_109116


namespace NUMINAMATH_CALUDE_min_value_z_l1091_109145

theorem min_value_z (x y : ℝ) : 3 * x^2 + 5 * y^2 + 12 * x - 10 * y + 40 ≥ 23 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l1091_109145


namespace NUMINAMATH_CALUDE_parabola_tangent_values_l1091_109136

/-- A parabola tangent to a line -/
structure ParabolaTangentToLine where
  /-- Coefficient of x^2 term in the parabola equation -/
  a : ℝ
  /-- Coefficient of x term in the parabola equation -/
  b : ℝ
  /-- The parabola y = ax^2 + bx is tangent to the line y = 2x + 4 -/
  is_tangent : ∃ (x : ℝ), a * x^2 + b * x = 2 * x + 4
  /-- The x-coordinate of the point of tangency is 1 -/
  tangent_point : ∃ (y : ℝ), a * 1^2 + b * 1 = 2 * 1 + 4 ∧ a * 1^2 + b * 1 = y

/-- The values of a and b for the parabola tangent to the line -/
theorem parabola_tangent_values (p : ParabolaTangentToLine) : p.a = 4/3 ∧ p.b = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_values_l1091_109136


namespace NUMINAMATH_CALUDE_campers_rowing_count_l1091_109113

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 15

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 17

/-- The total number of campers who went rowing that day -/
def total_campers : ℕ := morning_campers + afternoon_campers

theorem campers_rowing_count : total_campers = 32 := by
  sorry

end NUMINAMATH_CALUDE_campers_rowing_count_l1091_109113


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l1091_109102

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ 
  (∀ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z w : ℝ), (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l1091_109102
