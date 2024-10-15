import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2201_220125

theorem smallest_n_congruence (n : ℕ+) : 
  (5 * n.val ≡ 2345 [MOD 26]) ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2201_220125


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l2201_220176

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : is_periodic f 5) 
  (h_value : f 7 = 9) : 
  f 2020 - f 2018 = 9 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l2201_220176


namespace NUMINAMATH_CALUDE_cos_product_equals_one_l2201_220144

theorem cos_product_equals_one : 8 * Real.cos (4 * Real.pi / 9) * Real.cos (2 * Real.pi / 9) * Real.cos (Real.pi / 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_equals_one_l2201_220144


namespace NUMINAMATH_CALUDE_two_intersection_points_l2201_220121

/-- Define the first curve -/
def curve1 (x y : ℝ) : Prop :=
  (x + 2*y - 6) * (2*x - y + 4) = 0

/-- Define the second curve -/
def curve2 (x y : ℝ) : Prop :=
  (x - 3*y + 2) * (4*x + y - 14) = 0

/-- Define an intersection point -/
def is_intersection (x y : ℝ) : Prop :=
  curve1 x y ∧ curve2 x y

/-- The theorem stating that there are exactly two distinct intersection points -/
theorem two_intersection_points :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 :=
by
  sorry

end NUMINAMATH_CALUDE_two_intersection_points_l2201_220121


namespace NUMINAMATH_CALUDE_a_value_l2201_220184

def A (a : ℝ) : Set ℝ := {-1, a}
def B (a b : ℝ) : Set ℝ := {3^a, b}

theorem a_value (a b : ℝ) :
  A a ∪ B a b = {-1, 0, 1} → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l2201_220184


namespace NUMINAMATH_CALUDE_least_subtraction_l2201_220194

theorem least_subtraction (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((997 - y) % 5 = 3 ∧ (997 - y) % 9 = 3 ∧ (997 - y) % 11 = 3)) →
  (997 - x) % 5 = 3 ∧ (997 - x) % 9 = 3 ∧ (997 - x) % 11 = 3 →
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_l2201_220194


namespace NUMINAMATH_CALUDE_A_cubed_is_zero_l2201_220185

open Matrix

theorem A_cubed_is_zero {α : Type*} [Field α] (A : Matrix (Fin 2) (Fin 2) α) 
  (h1 : A ^ 4 = 0)
  (h2 : Matrix.trace A = 0) :
  A ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_A_cubed_is_zero_l2201_220185


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2201_220195

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) (h_sum : a 1 + a 19 = -18) :
  a 10 = -9 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2201_220195


namespace NUMINAMATH_CALUDE_min_value_z_l2201_220164

theorem min_value_z (a b : ℝ) : 
  ∃ (m : ℝ), ∀ (x y : ℝ), x^2 + y^2 ≤ 25 ∧ 2*x + y ≤ 5 → 
  x^2 + y^2 - 2*a*x - 2*b*y ≥ m ∧ m ≥ -a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l2201_220164


namespace NUMINAMATH_CALUDE_count_even_three_digit_numbers_less_than_700_l2201_220165

def valid_digits : List Nat := [1, 2, 3, 4, 5, 6]

def is_even (n : Nat) : Bool :=
  n % 2 = 0

def is_three_digit (n : Nat) : Bool :=
  100 ≤ n ∧ n < 1000

def count_valid_numbers : Nat :=
  (valid_digits.filter (· < 7)).length *
  valid_digits.length *
  (valid_digits.filter is_even).length

theorem count_even_three_digit_numbers_less_than_700 :
  count_valid_numbers = 108 := by
  sorry

end NUMINAMATH_CALUDE_count_even_three_digit_numbers_less_than_700_l2201_220165


namespace NUMINAMATH_CALUDE_decagon_equilateral_triangles_l2201_220113

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The number of distinct equilateral triangles with at least two vertices 
    from a given set of points -/
def countDistinctEquilateralTriangles (points : Set (ℝ × ℝ)) : ℕ := sorry

/-- The theorem stating the number of distinct equilateral triangles 
    in a regular decagon -/
theorem decagon_equilateral_triangles (d : RegularPolygon 10) :
  countDistinctEquilateralTriangles (Set.range d.vertices) = 90 := by sorry

end NUMINAMATH_CALUDE_decagon_equilateral_triangles_l2201_220113


namespace NUMINAMATH_CALUDE_initial_friends_correct_l2201_220108

/-- The number of friends initially playing the game -/
def initial_friends : ℕ := 2

/-- The number of new players that joined -/
def new_players : ℕ := 2

/-- The number of lives each player has -/
def lives_per_player : ℕ := 6

/-- The total number of lives after new players joined -/
def total_lives : ℕ := 24

/-- Theorem stating that the number of initial friends is correct -/
theorem initial_friends_correct : 
  lives_per_player * (initial_friends + new_players) = total_lives := by
  sorry

end NUMINAMATH_CALUDE_initial_friends_correct_l2201_220108


namespace NUMINAMATH_CALUDE_smallest_covering_rectangles_l2201_220189

/-- Represents a rectangle with width and height. -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a rectangular region to be covered. -/
structure Region where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle. -/
def rectangleArea (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the area of a region. -/
def regionArea (r : Region) : ℕ := r.width * r.height

/-- Theorem: The smallest number of 3-by-5 rectangles needed to cover a 15-by-20 region is 20. -/
theorem smallest_covering_rectangles :
  let coveringRectangle : Rectangle := { width := 3, height := 5 }
  let regionToCover : Region := { width := 15, height := 20 }
  (regionArea regionToCover) / (rectangleArea coveringRectangle) = 20 ∧
  (regionToCover.width % coveringRectangle.width = 0) ∧
  (regionToCover.height % coveringRectangle.height = 0) := by
  sorry

#check smallest_covering_rectangles

end NUMINAMATH_CALUDE_smallest_covering_rectangles_l2201_220189


namespace NUMINAMATH_CALUDE_quadratic_function_unique_form_l2201_220135

/-- Given a quadratic function f(x) = x^2 + ax + b that intersects the x-axis at (1, 0) 
    and has its axis of symmetry at x = 2, prove that f(x) = x^2 - 4x + 3 -/
theorem quadratic_function_unique_form (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^2 + a*x + b)
    (h2 : f 1 = 0)
    (h3 : -a/2 = 2) : 
  ∀ x, f x = x^2 - 4*x + 3 := by
sorry


end NUMINAMATH_CALUDE_quadratic_function_unique_form_l2201_220135


namespace NUMINAMATH_CALUDE_heptagon_angle_measure_l2201_220122

/-- In a heptagon GEOMETRY, prove that if ∠G ≅ ∠E ≅ ∠R, ∠O is supplementary to ∠Y,
    and assuming ∠M ≅ ∠T ≅ ∠R, then ∠R = 144° -/
theorem heptagon_angle_measure (GEOMETRY : Type) 
  (G E O M T R Y : GEOMETRY → ℝ) :
  (∀ x : GEOMETRY, G x = E x ∧ E x = R x) →  -- ∠G ≅ ∠E ≅ ∠R
  (∀ x : GEOMETRY, O x + Y x = 180) →        -- ∠O is supplementary to ∠Y
  (∀ x : GEOMETRY, M x = T x ∧ T x = R x) →  -- Assumption: ∠M ≅ ∠T ≅ ∠R
  (∀ x : GEOMETRY, G x + E x + O x + M x + T x + R x + Y x = 900) →  -- Sum of angles in heptagon
  (∀ x : GEOMETRY, R x = 144) :=
by
  sorry


end NUMINAMATH_CALUDE_heptagon_angle_measure_l2201_220122


namespace NUMINAMATH_CALUDE_fair_attendance_percentage_l2201_220115

/-- The percent of projected attendance that was the actual attendance --/
theorem fair_attendance_percentage (A : ℝ) (V W : ℝ) : 
  let projected_attendance := 1.25 * A
  let actual_attendance := 0.8 * A
  (actual_attendance / projected_attendance) * 100 = 64 := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_percentage_l2201_220115


namespace NUMINAMATH_CALUDE_dallas_current_age_l2201_220126

/-- Proves Dallas's current age given the relationships between family members' ages --/
theorem dallas_current_age (dallas_last_year darcy_last_year darcy_current dexter_current : ℕ) 
  (h1 : dallas_last_year = 3 * darcy_last_year)
  (h2 : darcy_current = 2 * dexter_current)
  (h3 : dexter_current = 8) :
  dallas_last_year + 1 = 46 := by
  sorry

#check dallas_current_age

end NUMINAMATH_CALUDE_dallas_current_age_l2201_220126


namespace NUMINAMATH_CALUDE_opposite_of_negative_fifth_l2201_220105

theorem opposite_of_negative_fifth : -(-(1/5 : ℚ)) = 1/5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fifth_l2201_220105


namespace NUMINAMATH_CALUDE_multiply_decimals_l2201_220193

theorem multiply_decimals : 3.6 * 0.05 = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l2201_220193


namespace NUMINAMATH_CALUDE_smallest_three_digit_prime_product_l2201_220178

theorem smallest_three_digit_prime_product : ∃ (n a b : ℕ),
  (100 ≤ n ∧ n < 1000) ∧  -- n is a three-digit positive integer
  (n = a * b * (10 * a + b)) ∧  -- n is the product of a, b, and 10a+b
  (a < 10 ∧ b < 10) ∧  -- a and b are each less than 10
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime (10 * a + b) ∧  -- a, b, and 10a+b are prime
  a ≠ b ∧ a ≠ (10 * a + b) ∧ b ≠ (10 * a + b) ∧  -- a, b, and 10a+b are distinct
  (∀ (m c d : ℕ), 
    (100 ≤ m ∧ m < 1000) →
    (m = c * d * (10 * c + d)) →
    (c < 10 ∧ d < 10) →
    Nat.Prime c → Nat.Prime d → Nat.Prime (10 * c + d) →
    c ≠ d → c ≠ (10 * c + d) → d ≠ (10 * c + d) →
    n ≤ m) ∧
  n = 138 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_prime_product_l2201_220178


namespace NUMINAMATH_CALUDE_highest_sample_number_l2201_220173

/-- Given a systematic sample from a population, calculate the highest number in the sample. -/
theorem highest_sample_number
  (total_students : Nat)
  (sample_size : Nat)
  (first_sample : Nat)
  (h1 : total_students = 54)
  (h2 : sample_size = 6)
  (h3 : first_sample = 5)
  (h4 : sample_size > 0)
  (h5 : total_students ≥ sample_size)
  : first_sample + (sample_size - 1) * (total_students / sample_size) = 50 := by
  sorry

#check highest_sample_number

end NUMINAMATH_CALUDE_highest_sample_number_l2201_220173


namespace NUMINAMATH_CALUDE_wheel_probability_l2201_220166

theorem wheel_probability (p_D p_E p_F p_G : ℚ) : 
  p_D = 3/8 → p_E = 1/4 → p_G = 1/8 → 
  p_D + p_E + p_F + p_G = 1 →
  p_F = 1/4 := by
sorry

end NUMINAMATH_CALUDE_wheel_probability_l2201_220166


namespace NUMINAMATH_CALUDE_water_purification_minimum_processes_l2201_220130

theorem water_purification_minimum_processes : ∃ n : ℕ,
  (∀ m : ℕ, m < n → (0.8 ^ m : ℝ) ≥ 0.05) ∧
  (0.8 ^ n : ℝ) < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_water_purification_minimum_processes_l2201_220130


namespace NUMINAMATH_CALUDE_smallest_with_2023_divisors_l2201_220119

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Checks if a number is divisible by another -/
def is_divisible (a b : ℕ) : Prop := sorry

theorem smallest_with_2023_divisors :
  ∃ (n m k : ℕ),
    n > 0 ∧
    num_divisors n = 2023 ∧
    n = m * 6^k ∧
    ¬ is_divisible m 6 ∧
    (∀ (n' m' k' : ℕ),
      n' > 0 →
      num_divisors n' = 2023 →
      n' = m' * 6^k' →
      ¬ is_divisible m' 6 →
      n ≤ n') ∧
    m + k = 745 :=
  sorry

end NUMINAMATH_CALUDE_smallest_with_2023_divisors_l2201_220119


namespace NUMINAMATH_CALUDE_circle_dot_product_bound_l2201_220136

theorem circle_dot_product_bound :
  ∀ (A : ℝ × ℝ),
  A.1^2 + (A.2 - 1)^2 = 1 →
  -2 ≤ (A.1 * 2 + A.2 * 0) ∧ (A.1 * 2 + A.2 * 0) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_dot_product_bound_l2201_220136


namespace NUMINAMATH_CALUDE_cone_base_diameter_l2201_220199

theorem cone_base_diameter (sphere_radius : ℝ) (cone_height : ℝ) (waste_percentage : ℝ) :
  sphere_radius = 9 →
  cone_height = 9 →
  waste_percentage = 75 →
  let cone_volume := (1 - waste_percentage / 100) * (4 / 3) * Real.pi * sphere_radius ^ 3
  let cone_base_radius := Real.sqrt (3 * cone_volume / (Real.pi * cone_height))
  2 * cone_base_radius = 9 :=
by sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l2201_220199


namespace NUMINAMATH_CALUDE_choir_arrangement_min_choir_size_l2201_220183

theorem choir_arrangement (n : ℕ) : (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 990 :=
sorry

theorem min_choir_size : ∃ (n : ℕ), n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n = 990 :=
sorry

end NUMINAMATH_CALUDE_choir_arrangement_min_choir_size_l2201_220183


namespace NUMINAMATH_CALUDE_angle_D_measure_l2201_220116

-- Define the hexagon and its angles
structure Hexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the properties of the hexagon
def is_convex_hexagon_with_properties (h : Hexagon) : Prop :=
  h.A = h.B ∧ h.B = h.C ∧  -- A, B, C are congruent
  h.D = h.E ∧ h.E = h.F ∧  -- D, E, F are congruent
  h.A + 30 = h.D ∧         -- A is 30° less than D
  h.A + h.B + h.C + h.D + h.E + h.F = 720  -- Sum of angles in a hexagon

-- Theorem statement
theorem angle_D_measure (h : Hexagon) 
  (hprop : is_convex_hexagon_with_properties h) : h.D = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l2201_220116


namespace NUMINAMATH_CALUDE_propositions_truth_l2201_220159

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := a > b → a^2 > b^2
def proposition2 (x y : ℝ) : Prop := x + y = 0 → (x = -y ∧ y = -x)
def proposition3 (x : ℝ) : Prop := x^2 < 4 → -2 < x ∧ x < 2

-- State the theorem
theorem propositions_truth : 
  (∀ x y : ℝ, x = -y ∧ y = -x → x + y = 0) ∧ 
  (∀ x : ℝ, (x ≥ 2 ∨ x ≤ -2) → x^2 ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_propositions_truth_l2201_220159


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l2201_220109

/-- A regular polygon with side length 8 units and exterior angle 72 degrees has a perimeter of 40 units. -/
theorem regular_polygon_perimeter (s : ℝ) (θ : ℝ) (n : ℕ) : 
  s = 8 → θ = 72 → θ = 360 / n → n * s = 40 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l2201_220109


namespace NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l2201_220160

theorem hua_luogeng_birthday_factorization (h : 19101112 = 1163 * 16424) :
  Nat.Prime 1163 ∧ ¬Nat.Prime 16424 := by
  sorry

end NUMINAMATH_CALUDE_hua_luogeng_birthday_factorization_l2201_220160


namespace NUMINAMATH_CALUDE_expression_simplification_l2201_220110

theorem expression_simplification (x : ℝ) :
  x - 3 * (2 + x) + 4 * (2 - x) - 5 * (1 + 3 * x) + 2 * x^2 = 2 * x^2 - 21 * x - 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2201_220110


namespace NUMINAMATH_CALUDE_units_digit_of_sum_factorials_1000_l2201_220147

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_of_sum_factorials_1000 :
  sum_of_factorials 1000 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_factorials_1000_l2201_220147


namespace NUMINAMATH_CALUDE_find_number_l2201_220196

theorem find_number : ∃ x : ℝ, 0.62 * 150 - 0.20 * x = 43 ∧ x = 250 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2201_220196


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l2201_220179

theorem complex_magnitude_equality (n : ℝ) (hn : n > 0) :
  Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 10 → n = 15 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l2201_220179


namespace NUMINAMATH_CALUDE_range_of_m_range_of_x_l2201_220104

-- Define the function f
def f (m x : ℝ) : ℝ := m * x^2 - m * x - 6 + m

-- Part 1
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x < 0) → m < 6/7 :=
sorry

-- Part 2
theorem range_of_x (x : ℝ) :
  (∀ m ∈ Set.Icc (-2) 2, f m x < 0) → -1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_x_l2201_220104


namespace NUMINAMATH_CALUDE_rent_increase_theorem_l2201_220118

theorem rent_increase_theorem (num_friends : ℕ) (initial_avg_rent : ℚ) 
  (increased_rent : ℚ) (increase_percentage : ℚ) :
  num_friends = 4 →
  initial_avg_rent = 800 →
  increased_rent = 1400 →
  increase_percentage = 20 / 100 →
  let total_rent : ℚ := initial_avg_rent * num_friends
  let new_increased_rent : ℚ := increased_rent * (1 + increase_percentage)
  let new_total_rent : ℚ := total_rent - increased_rent + new_increased_rent
  let new_avg_rent : ℚ := new_total_rent / num_friends
  new_avg_rent = 870 := by sorry

end NUMINAMATH_CALUDE_rent_increase_theorem_l2201_220118


namespace NUMINAMATH_CALUDE_third_term_not_unique_l2201_220188

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The product of the first 5 terms of a sequence equals 32 -/
def ProductEquals32 (a : ℕ → ℝ) : Prop :=
  a 1 * a 2 * a 3 * a 4 * a 5 = 32

/-- The third term of a geometric sequence cannot be uniquely determined
    given only that the product of the first 5 terms equals 32 -/
theorem third_term_not_unique (a : ℕ → ℝ) 
    (h1 : GeometricSequence a) (h2 : ProductEquals32 a) :
    ¬∃! x : ℝ, a 3 = x :=
  sorry

end NUMINAMATH_CALUDE_third_term_not_unique_l2201_220188


namespace NUMINAMATH_CALUDE_sphere_distance_to_plane_l2201_220133

/-- The distance from the center of a sphere to the plane formed by three points on its surface. -/
def distance_center_to_plane (r : ℝ) (a b c : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for a sphere of radius 13 with three points on its surface forming
    a triangle with side lengths 6, 8, and 10, the distance from the center to the plane
    containing the triangle is 12. -/
theorem sphere_distance_to_plane :
  distance_center_to_plane 13 6 8 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sphere_distance_to_plane_l2201_220133


namespace NUMINAMATH_CALUDE_distance_home_to_school_l2201_220157

/-- Represents Johnny's journey to and from school -/
structure JourneySegment where
  speed : ℝ
  time : ℝ
  distance : ℝ
  (distance_eq : distance = speed * time)

/-- Represents Johnny's complete journey -/
structure Journey where
  jog : JourneySegment
  bike : JourneySegment
  bus : JourneySegment

/-- The journey satisfies the given conditions -/
def journey_conditions (j : Journey) : Prop :=
  j.jog.speed = 5 ∧
  j.bike.speed = 10 ∧
  j.bus.speed = 30 ∧
  j.jog.time = 1 ∧
  j.bike.time = 1 ∧
  j.bus.time = 1

/-- The theorem stating the distance from home to school -/
theorem distance_home_to_school (j : Journey) 
  (h : journey_conditions j) : 
  j.bus.distance - j.bike.distance = 20 := by
  sorry


end NUMINAMATH_CALUDE_distance_home_to_school_l2201_220157


namespace NUMINAMATH_CALUDE_one_seven_two_eight_gt_one_roundness_of_1728_l2201_220120

/-- Roundness of an integer greater than 1 is the sum of exponents in its prime factorization -/
def roundness (n : ℕ) : ℕ :=
  sorry

/-- 1728 is greater than 1 -/
theorem one_seven_two_eight_gt_one : 1728 > 1 :=
  sorry

/-- The roundness of 1728 is 9 -/
theorem roundness_of_1728 : roundness 1728 = 9 :=
  sorry

end NUMINAMATH_CALUDE_one_seven_two_eight_gt_one_roundness_of_1728_l2201_220120


namespace NUMINAMATH_CALUDE_binomial_ratio_sum_l2201_220141

theorem binomial_ratio_sum (n k : ℕ+) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k+1) : ℚ) = 2/3 ∧ 
  (Nat.choose n (k+1) : ℚ) / (Nat.choose n (k+2) : ℚ) = 3/4 ∧
  (∀ m l : ℕ+, (Nat.choose m l : ℚ) / (Nat.choose m (l+1) : ℚ) = 2/3 ∧ 
               (Nat.choose m (l+1) : ℚ) / (Nat.choose m (l+2) : ℚ) = 3/4 → 
               m = n ∧ l = k) →
  n + k = 47 := by
sorry

end NUMINAMATH_CALUDE_binomial_ratio_sum_l2201_220141


namespace NUMINAMATH_CALUDE_cubic_equation_geometric_progression_solution_l2201_220190

/-- Given a cubic equation ax^3 + bx^2 + cx + d = 0 where the coefficients form
    a geometric progression with ratio q, prove that x = -q is a solution. -/
theorem cubic_equation_geometric_progression_solution
  (a b c d q : ℝ) (hq : q ≠ 0) (ha : a ≠ 0)
  (hb : b = a * q) (hc : c = a * q^2) (hd : d = a * q^3) :
  a * (-q)^3 + b * (-q)^2 + c * (-q) + d = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_geometric_progression_solution_l2201_220190


namespace NUMINAMATH_CALUDE_max_value_of_quadratic_l2201_220145

theorem max_value_of_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 1/3) :
  x * (1 - 3*x) ≤ 1/12 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_quadratic_l2201_220145


namespace NUMINAMATH_CALUDE_temperature_85_at_latest_time_l2201_220143

/-- The temperature function in Denver, CO, where t is time in hours past noon -/
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The latest time when the temperature is 85 degrees -/
def latest_85_degrees : ℝ := 9

theorem temperature_85_at_latest_time :
  temperature latest_85_degrees = 85 ∧
  ∀ t > latest_85_degrees, temperature t ≠ 85 := by
sorry

end NUMINAMATH_CALUDE_temperature_85_at_latest_time_l2201_220143


namespace NUMINAMATH_CALUDE_max_value_is_35_l2201_220168

def Digits : Finset ℕ := {1, 2, 5, 6}

def Expression (a b c d : ℕ) : ℕ := (a - b)^2 + c * d

theorem max_value_is_35 :
  ∃ (a b c d : ℕ),
    a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    Expression a b c d = 35 ∧
    ∀ (w x y z : ℕ),
      w ∈ Digits → x ∈ Digits → y ∈ Digits → z ∈ Digits →
      w ≠ x → w ≠ y → w ≠ z → x ≠ y → x ≠ z → y ≠ z →
      Expression w x y z ≤ 35 :=
by sorry

end NUMINAMATH_CALUDE_max_value_is_35_l2201_220168


namespace NUMINAMATH_CALUDE_quadratic_decreasing_condition_l2201_220114

/-- Given a quadratic function y = (x - m)^2 - 1, if it decreases as x increases
    when x ≤ 3, then m ≥ 3. -/
theorem quadratic_decreasing_condition (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 3 → 
    ((x₁ - m)^2 - 1) > ((x₂ - m)^2 - 1)) → 
  m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_condition_l2201_220114


namespace NUMINAMATH_CALUDE_operation_on_81_divided_by_3_l2201_220162

theorem operation_on_81_divided_by_3 : ∃ f : ℝ → ℝ, (f 81) / 3 = 3 := by sorry

end NUMINAMATH_CALUDE_operation_on_81_divided_by_3_l2201_220162


namespace NUMINAMATH_CALUDE_domain_of_f_l2201_220138

noncomputable def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + 5*x - 2) / (x^3 - 5*x^2 + 8*x - 4)

theorem domain_of_f :
  Set.range f = {x : ℝ | x ∈ (Set.Iio 1) ∪ (Set.Ioo 1 2) ∪ (Set.Ioo 2 4) ∪ (Set.Ioi 4)} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l2201_220138


namespace NUMINAMATH_CALUDE_product_sum_bounds_l2201_220128

theorem product_sum_bounds (x y z t : ℝ) 
  (sum_zero : x + y + z + t = 0) 
  (sum_squares_one : x^2 + y^2 + z^2 + t^2 = 1) : 
  -1 ≤ x*y + y*z + z*t + t*x ∧ x*y + y*z + z*t + t*x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_bounds_l2201_220128


namespace NUMINAMATH_CALUDE_number_transformation_l2201_220186

theorem number_transformation (x : ℝ) : (x * (5/6) / 10 + 2/3) = 3/4 * x + 3/4 := by
  sorry

end NUMINAMATH_CALUDE_number_transformation_l2201_220186


namespace NUMINAMATH_CALUDE_quadratic_completion_square_l2201_220131

theorem quadratic_completion_square (x : ℝ) :
  4 * x^2 - 8 * x - 128 = 0 →
  ∃ (r : ℝ), (x + r)^2 = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_square_l2201_220131


namespace NUMINAMATH_CALUDE_small_glass_cost_is_three_l2201_220175

/-- The cost of a small glass given Peter's purchase information -/
def small_glass_cost (total_money : ℕ) (num_small : ℕ) (num_large : ℕ) (large_cost : ℕ) (change : ℕ) : ℕ :=
  ((total_money - change) - (num_large * large_cost)) / num_small

/-- Theorem stating that the cost of a small glass is $3 given the problem conditions -/
theorem small_glass_cost_is_three :
  small_glass_cost 50 8 5 5 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_small_glass_cost_is_three_l2201_220175


namespace NUMINAMATH_CALUDE_danny_steve_time_ratio_l2201_220153

/-- The time it takes Danny to reach Steve's house, in minutes -/
def danny_time : ℝ := 35

/-- The time it takes Steve to reach Danny's house, in minutes -/
def steve_time : ℝ := 70

/-- The extra time it takes Steve to reach the halfway point compared to Danny, in minutes -/
def extra_time : ℝ := 17.5

theorem danny_steve_time_ratio :
  danny_time / steve_time = 1 / 2 ∧
  steve_time / 2 = danny_time / 2 + extra_time :=
sorry

end NUMINAMATH_CALUDE_danny_steve_time_ratio_l2201_220153


namespace NUMINAMATH_CALUDE_jelly_bean_probabilities_l2201_220152

/-- Represents the number of jelly beans of each color in the bag -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  purple : ℕ

/-- Calculates the total number of jelly beans in the bag -/
def totalJellyBeans (bag : JellyBeanBag) : ℕ :=
  bag.red + bag.green + bag.yellow + bag.blue + bag.purple

/-- The specific bag of jelly beans described in the problem -/
def ourBag : JellyBeanBag :=
  { red := 10, green := 12, yellow := 15, blue := 18, purple := 5 }

theorem jelly_bean_probabilities :
  let total := totalJellyBeans ourBag
  (ourBag.purple : ℚ) / total = 1 / 12 ∧
  ((ourBag.blue + ourBag.purple : ℚ) / total = 23 / 60) := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probabilities_l2201_220152


namespace NUMINAMATH_CALUDE_special_list_median_l2201_220140

/-- The sum of integers from 1 to n -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list where each integer n from 1 to 100 appears n times -/
def special_list : List ℕ := sorry

/-- The median of a list is the average of the middle two elements when the list has even length -/
def median (l : List ℕ) : ℚ := sorry

theorem special_list_median :
  median special_list = 71 := by sorry

end NUMINAMATH_CALUDE_special_list_median_l2201_220140


namespace NUMINAMATH_CALUDE_quadratic_vertex_ordinate_l2201_220129

theorem quadratic_vertex_ordinate (b c : ℤ) :
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ (x₁^2 + b*x₁ + c = 2017) ∧ (x₂^2 + b*x₂ + c = 2017)) →
  (-(b^2 - 4*c) / 4 = -1016064) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_ordinate_l2201_220129


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2201_220139

def A : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | -2 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2201_220139


namespace NUMINAMATH_CALUDE_coffee_cost_l2201_220169

theorem coffee_cost (sandwich_cost coffee_cost : ℕ) : 
  (3 * sandwich_cost + 2 * coffee_cost = 630) →
  (2 * sandwich_cost + 3 * coffee_cost = 690) →
  coffee_cost = 162 := by
sorry

end NUMINAMATH_CALUDE_coffee_cost_l2201_220169


namespace NUMINAMATH_CALUDE_simplify_absolute_difference_l2201_220158

theorem simplify_absolute_difference (a b : ℝ) (h : a + b < 0) :
  |a + b - 1| - |3 - a - b| = -2 := by sorry

end NUMINAMATH_CALUDE_simplify_absolute_difference_l2201_220158


namespace NUMINAMATH_CALUDE_correct_equation_l2201_220124

theorem correct_equation : (-3)^2 = 9 ∧ 
  (-2)^3 ≠ -6 ∧ 
  ¬(∀ x, x^2 = 4 → x = 2 ∨ x = -2) ∧ 
  (Real.sqrt 2)^2 ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l2201_220124


namespace NUMINAMATH_CALUDE_equation_solution_l2201_220154

theorem equation_solution : ∃ x : ℚ, (54 - x / 6 * 3 = 36) ∧ (x = 36) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2201_220154


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2201_220106

theorem sqrt_product_equality : Real.sqrt 48 * Real.sqrt 27 * Real.sqrt 8 * Real.sqrt 3 = 72 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2201_220106


namespace NUMINAMATH_CALUDE_distance_after_walk_l2201_220172

theorem distance_after_walk (west_distance : ℝ) (north_distance : ℝ) :
  west_distance = 10 →
  north_distance = 10 →
  ∃ (total_distance : ℝ), total_distance^2 = west_distance^2 + north_distance^2 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_after_walk_l2201_220172


namespace NUMINAMATH_CALUDE_uncool_parents_count_l2201_220180

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) :
  total = 35 →
  cool_dads = 18 →
  cool_moms = 22 →
  both_cool = 11 →
  total - (cool_dads + cool_moms - both_cool) = 6 :=
by sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l2201_220180


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2201_220112

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 4 + 3 * Real.sqrt 2 ∧ 
  x₂ = 4 - 3 * Real.sqrt 2 ∧ 
  x₁^2 - 8*x₁ - 2 = 0 ∧ 
  x₂^2 - 8*x₂ - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2201_220112


namespace NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l2201_220187

theorem tan_product_from_cos_sum_diff (α β : ℝ) 
  (h1 : Real.cos (α + β) = 3/5) 
  (h2 : Real.cos (α - β) = 4/5) : 
  Real.tan α * Real.tan β = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l2201_220187


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l2201_220134

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.30 * Q)
  (hQ : Q = 0.20 * P)
  (hN : N = 0.50 * P)
  (hP : P ≠ 0) :
  M / N = 3 / 25 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l2201_220134


namespace NUMINAMATH_CALUDE_max_notebooks_charlie_can_buy_l2201_220111

theorem max_notebooks_charlie_can_buy (available : ℝ) (cost_per_notebook : ℝ) 
  (h1 : available = 12) (h2 : cost_per_notebook = 1.45) : 
  ⌊available / cost_per_notebook⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_notebooks_charlie_can_buy_l2201_220111


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l2201_220149

/-- A quadratic function with coefficients a, b, and c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x, quadratic a b c (x + 4.5) = quadratic a b c (4.5 - x)) →
  quadratic a b c (-9) = 1 →
  quadratic a b c 18 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l2201_220149


namespace NUMINAMATH_CALUDE_halloween_goodie_bags_l2201_220156

theorem halloween_goodie_bags (vampire_students pumpkin_students : ℕ)
  (pack_size pack_cost individual_cost total_cost : ℕ) :
  vampire_students = 11 →
  pumpkin_students = 14 →
  pack_size = 5 →
  pack_cost = 3 →
  individual_cost = 1 →
  total_cost = 17 →
  vampire_students + pumpkin_students = 25 :=
by sorry

end NUMINAMATH_CALUDE_halloween_goodie_bags_l2201_220156


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l2201_220146

theorem parabola_point_coordinates :
  ∀ x y : ℝ,
  y = x^2 →
  |y| = |x| + 3 →
  ((x = 1 ∧ y = 4) ∨ (x = -1 ∧ y = 4)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l2201_220146


namespace NUMINAMATH_CALUDE_subtraction_result_l2201_220155

theorem subtraction_result : (3.75 : ℝ) - 1.4 = 2.35 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l2201_220155


namespace NUMINAMATH_CALUDE_green_light_probability_theorem_l2201_220142

/-- Represents the duration of traffic light colors in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of encountering a green light -/
def greenLightProbability (d : TrafficLightDuration) : ℚ :=
  d.green / (d.red + d.yellow + d.green)

/-- Theorem: The probability of encountering a green light at the given intersection is 8/15 -/
theorem green_light_probability_theorem (d : TrafficLightDuration) 
    (h1 : d.red = 30)
    (h2 : d.yellow = 5)
    (h3 : d.green = 40) : 
  greenLightProbability d = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_green_light_probability_theorem_l2201_220142


namespace NUMINAMATH_CALUDE_original_avg_age_l2201_220102

def original_class_size : ℕ := 8
def new_students_size : ℕ := 8
def new_students_avg_age : ℕ := 32
def avg_age_decrease : ℕ := 4

theorem original_avg_age (original_avg : ℕ) :
  (original_avg * original_class_size + new_students_avg_age * new_students_size) / 
  (original_class_size + new_students_size) = original_avg - avg_age_decrease →
  original_avg = 40 :=
by sorry

end NUMINAMATH_CALUDE_original_avg_age_l2201_220102


namespace NUMINAMATH_CALUDE_average_diesel_cost_approx_9_94_l2201_220150

/-- Represents the diesel purchase data for a single year -/
structure YearlyPurchase where
  litres : ℝ
  pricePerLitre : ℝ

/-- Calculates the total cost for a year including delivery fees and taxes -/
def yearlyTotalCost (purchase : YearlyPurchase) (deliveryFee : ℝ) (taxes : ℝ) : ℝ :=
  purchase.litres * purchase.pricePerLitre + deliveryFee + taxes

/-- Theorem: The average cost per litre of diesel over three years is approximately 9.94 -/
theorem average_diesel_cost_approx_9_94 
  (year1 : YearlyPurchase)
  (year2 : YearlyPurchase)
  (year3 : YearlyPurchase)
  (deliveryFee : ℝ)
  (taxes : ℝ)
  (h1 : year1.litres = 520 ∧ year1.pricePerLitre = 8.5)
  (h2 : year2.litres = 540 ∧ year2.pricePerLitre = 9)
  (h3 : year3.litres = 560 ∧ year3.pricePerLitre = 9.5)
  (h4 : deliveryFee = 200)
  (h5 : taxes = 300) :
  let totalCost := yearlyTotalCost year1 deliveryFee taxes + 
                   yearlyTotalCost year2 deliveryFee taxes + 
                   yearlyTotalCost year3 deliveryFee taxes
  let totalLitres := year1.litres + year2.litres + year3.litres
  let averageCost := totalCost / totalLitres
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |averageCost - 9.94| < ε :=
by sorry

end NUMINAMATH_CALUDE_average_diesel_cost_approx_9_94_l2201_220150


namespace NUMINAMATH_CALUDE_triangle_x_theorem_l2201_220103

/-- A function that checks if three side lengths can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of positive integer values of x for which a triangle with sides 5, 12, and x^2 exists -/
def triangle_x_values : Set ℕ+ :=
  {x : ℕ+ | is_triangle 5 12 (x.val ^ 2)}

theorem triangle_x_theorem : triangle_x_values = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_triangle_x_theorem_l2201_220103


namespace NUMINAMATH_CALUDE_square_area_increase_l2201_220171

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.5 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l2201_220171


namespace NUMINAMATH_CALUDE_number_problem_l2201_220132

theorem number_problem (x : ℚ) : (x / 6) * 12 = 10 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2201_220132


namespace NUMINAMATH_CALUDE_product_of_square_roots_l2201_220137

theorem product_of_square_roots (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (8 * q^3) * Real.sqrt (7 * q^5) = 29 * q^4 * Real.sqrt 840 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l2201_220137


namespace NUMINAMATH_CALUDE_refrigerator_transport_cost_prove_transport_cost_l2201_220198

/-- Calculates the transport cost for a refrigerator purchase --/
theorem refrigerator_transport_cost 
  (purchase_price : ℝ) 
  (discount_rate : ℝ) 
  (installation_cost : ℝ) 
  (profit_rate : ℝ) 
  (selling_price : ℝ) : ℝ :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let total_cost := purchase_price + installation_cost
  let transport_cost := (selling_price / (1 + profit_rate)) - total_cost
  transport_cost

/-- Proves that the transport cost is 4000 given the problem conditions --/
theorem prove_transport_cost : 
  refrigerator_transport_cost 15500 0.2 250 0.1 21725 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_transport_cost_prove_transport_cost_l2201_220198


namespace NUMINAMATH_CALUDE_building_height_from_shadows_l2201_220107

/-- Given a flagstaff and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height_from_shadows
  (flagstaff_height : ℝ)
  (flagstaff_shadow : ℝ)
  (building_shadow : ℝ)
  (flagstaff_height_pos : 0 < flagstaff_height)
  (flagstaff_shadow_pos : 0 < flagstaff_shadow)
  (building_shadow_pos : 0 < building_shadow)
  (h_flagstaff : flagstaff_height = 17.5)
  (h_flagstaff_shadow : flagstaff_shadow = 40.25)
  (h_building_shadow : building_shadow = 28.75) :
  flagstaff_height / flagstaff_shadow * building_shadow = 12.4375 := by
sorry

end NUMINAMATH_CALUDE_building_height_from_shadows_l2201_220107


namespace NUMINAMATH_CALUDE_point_trajectory_l2201_220182

/-- The trajectory of a point P satisfying |PA| + |PB| = 5, where A(0,0) and B(5,0) are fixed points -/
theorem point_trajectory (P : ℝ × ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (5, 0)
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 5 →
  P.2 = 0 ∧ 0 ≤ P.1 ∧ P.1 ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_point_trajectory_l2201_220182


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2201_220163

theorem complex_modulus_problem (z : ℂ) (h : (z - Complex.I) * Complex.I = 2 + 3 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2201_220163


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2201_220117

/-- Given a circular piece of paper with radius 5 inches, when a 300° sector is removed
    and the remaining sector is used to form a right circular cone,
    the circumference of the base of the cone is 25π/3 inches. -/
theorem cone_base_circumference :
  let original_radius : ℝ := 5
  let removed_angle : ℝ := 300
  let full_circle_angle : ℝ := 360
  let remaining_fraction : ℝ := (full_circle_angle - removed_angle) / full_circle_angle
  let cone_base_circumference : ℝ := 2 * π * original_radius * remaining_fraction
  cone_base_circumference = 25 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2201_220117


namespace NUMINAMATH_CALUDE_inequality_proof_l2201_220181

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2201_220181


namespace NUMINAMATH_CALUDE_odometer_sum_of_squares_l2201_220191

/-- Represents a car's odometer reading as a 3-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- Converts an OdometerReading to a natural number -/
def OdometerReading.toNat (r : OdometerReading) : Nat :=
  100 * r.hundreds + 10 * r.tens + r.ones

/-- Reverses the digits of an OdometerReading -/
def OdometerReading.reverse (r : OdometerReading) : OdometerReading where
  hundreds := r.ones
  tens := r.tens
  ones := r.hundreds
  valid := by sorry

theorem odometer_sum_of_squares (initial : OdometerReading) 
  (h1 : initial.hundreds + initial.tens + initial.ones ≤ 9)
  (h2 : ∃ (hours : Nat), 
    (OdometerReading.toNat (OdometerReading.reverse initial) - OdometerReading.toNat initial) = 75 * hours) :
  initial.hundreds^2 + initial.tens^2 + initial.ones^2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_odometer_sum_of_squares_l2201_220191


namespace NUMINAMATH_CALUDE_f_properties_l2201_220151

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * Real.log x + (a + 1) * x - 0.5 * x^2

theorem f_properties (a : ℝ) (h_a : a > 0) :
  -- Monotonicity property
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ ∧ f a x₁ < f a x₂) ∧
  (∃ (x₃ x₄ : ℝ), x₃ > 0 ∧ x₄ > 0 ∧ x₃ < x₄ ∧ f a x₃ > f a x₄) ∧
  -- Maximum value of b
  (∀ b : ℝ, (∀ x : ℝ, x > 0 → f a x ≥ -0.5 * x^2 + a * x + b) →
    b ≤ 0.5 * (1 + Real.log 2)) ∧
  (∃ b : ℝ, b = 0.5 * (1 + Real.log 2) ∧
    (∀ x : ℝ, x > 0 → f (0.5) x ≥ -0.5 * x^2 + 0.5 * x + b)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2201_220151


namespace NUMINAMATH_CALUDE_binary_101110_equals_46_l2201_220170

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_101110_equals_46 :
  binary_to_decimal [false, true, true, true, true, false, true] = 46 := by
  sorry

end NUMINAMATH_CALUDE_binary_101110_equals_46_l2201_220170


namespace NUMINAMATH_CALUDE_monomial_sum_exponent_l2201_220197

theorem monomial_sum_exponent (m n : ℤ) : 
  (∃ k : ℤ, ∃ c : ℚ, -x^(m-2) * y^3 + 2/3 * x^n * y^(2*m-3*n) = c * x^k * y^k) → 
  m^(-n : ℤ) = (1 : ℚ)/3 :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_exponent_l2201_220197


namespace NUMINAMATH_CALUDE_committee_selection_l2201_220167

theorem committee_selection (n : ℕ) : 
  (n.choose 3 = 20) → (n.choose 4 = 15) := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l2201_220167


namespace NUMINAMATH_CALUDE_range_of_f_l2201_220123

-- Define the function
def f (x : ℝ) : ℝ := |x^2 - 4| - 3*x

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -6 ≤ y ∧ y ≤ 25/4} :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2201_220123


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2201_220127

-- Define the quadratic polynomial p(x)
def p (x : ℚ) : ℚ := (12/7) * x^2 + (36/7) * x - 216/7

-- Theorem stating that p(x) satisfies the given conditions
theorem quadratic_polynomial_satisfies_conditions :
  p (-6) = 0 ∧ p 3 = 0 ∧ p 1 = -24 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2201_220127


namespace NUMINAMATH_CALUDE_five_integers_exist_l2201_220100

theorem five_integers_exist : ∃ (a b c d e : ℤ),
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  a * b * c = 8 ∧
  c * d * e = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_five_integers_exist_l2201_220100


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2201_220101

/-- A square with an inscribed circle -/
structure InscribedSquare :=
  (side : ℝ)
  (radius : ℝ)
  (h_radius : radius = side / 2)

/-- A point on the inscribed circle -/
structure CirclePoint (s : InscribedSquare) :=
  (x : ℝ)
  (y : ℝ)
  (h_on_circle : x^2 + y^2 = s.radius^2)

/-- Theorem: If a point on the inscribed circle is 1 unit from one side
    and 2 units from another side, then the area of the square is 100 -/
theorem inscribed_square_area
  (s : InscribedSquare)
  (p : CirclePoint s)
  (h_dist1 : p.x = 1 ∨ p.y = 1)
  (h_dist2 : p.x = 2 ∨ p.y = 2) :
  s.side^2 = 100 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2201_220101


namespace NUMINAMATH_CALUDE_problem_solution_l2201_220174

theorem problem_solution (x : ℝ) : 
  (x - 9)^3 / (x + 4) = 27 → (x^2 - 12*x + 15) / (x - 2) = -20.1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2201_220174


namespace NUMINAMATH_CALUDE_equation_solution_l2201_220192

theorem equation_solution : 
  ∃! x : ℝ, (x - 60) / 3 = (4 - 3*x) / 6 ∧ x = 24.8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2201_220192


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2201_220148

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ x => |2*x + 4|
  let g : ℝ → ℝ := λ x => 1 - 3*x + x^2
  let solution1 : ℝ := (5 + Real.sqrt 37) / 2
  let solution2 : ℝ := (5 - Real.sqrt 37) / 2
  (∀ x : ℝ, f x = g x ↔ x = solution1 ∨ x = solution2) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2201_220148


namespace NUMINAMATH_CALUDE_art_show_ratio_l2201_220177

/-- Given an artist who painted 153 pictures and sold 72, prove that the ratio of
    remaining pictures to sold pictures, when simplified to lowest terms, is 9:8. -/
theorem art_show_ratio :
  let total_pictures : ℕ := 153
  let sold_pictures : ℕ := 72
  let remaining_pictures : ℕ := total_pictures - sold_pictures
  let ratio := (remaining_pictures, sold_pictures)
  (ratio.1.gcd ratio.2 = 9) ∧
  (ratio.1 / ratio.1.gcd ratio.2 = 9) ∧
  (ratio.2 / ratio.1.gcd ratio.2 = 8) := by
sorry


end NUMINAMATH_CALUDE_art_show_ratio_l2201_220177


namespace NUMINAMATH_CALUDE_intersection_area_implies_m_values_l2201_220161

def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

def Line (m x y : ℝ) : Prop := x - m*y + 2 = 0

def AreaABO (m : ℝ) : ℝ := 2

theorem intersection_area_implies_m_values (m : ℝ) :
  (∃ A B : ℝ × ℝ, 
    Circle A.1 A.2 ∧ Circle B.1 B.2 ∧ 
    Line m A.1 A.2 ∧ Line m B.1 B.2 ∧
    AreaABO m = 2) →
  m = 1 ∨ m = -1 :=
sorry

end NUMINAMATH_CALUDE_intersection_area_implies_m_values_l2201_220161
