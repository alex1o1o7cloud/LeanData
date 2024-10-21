import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l389_38964

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 3) - 2 * Real.cos (2 * x) + 1

theorem triangle_side_ratio_range (A B C : ℝ) (hABC : A + B + C = Real.pi) 
  (hAcute : 0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2) 
  (hfA : f A = 0) :
  ∃ (b c : ℝ), 0 < b ∧ 0 < c ∧ 1/2 < b/c ∧ b/c < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l389_38964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_nested_evaluation_l389_38910

noncomputable def p (x y : ℝ) : ℝ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 2*y
  else if x ≥ 0 ∧ y < 0 then x^2 + y^2
  else 3*x + y

theorem p_nested_evaluation : p (p 2 (-3)) (p (-4) 1) = 290 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_nested_evaluation_l389_38910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_in_cone_quarter_l389_38952

/-- The radius of a sphere inscribed in a quarter of a cone -/
noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ :=
  a / (Real.sqrt 3 + Real.sqrt 2)

/-- Theorem: The radius of a sphere inscribed in one of the four parts of a cone,
    where the cone's axial cross-section is an equilateral triangle with side length a
    and is divided by two perpendicular planes through its axis,
    is equal to a / (√3 + √2). -/
theorem inscribed_sphere_in_cone_quarter (a : ℝ) (h : a > 0) :
  inscribed_sphere_radius a = a / (Real.sqrt 3 + Real.sqrt 2) :=
by
  -- Unfold the definition of inscribed_sphere_radius
  unfold inscribed_sphere_radius
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_in_cone_quarter_l389_38952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_approaches_two_l389_38974

/-- The eccentricity of a hyperbola with equation x²/m² - y²/(m²-4) = 1 approaches 2 for large integer m -/
theorem hyperbola_eccentricity_approaches_two (m : ℤ) (h : m.natAbs > 2) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n : ℕ, n ≥ N → 
    |Real.sqrt (2 - 4 / (m^2 : ℝ)) - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_approaches_two_l389_38974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_and_value_l389_38925

-- Define the power function as noncomputable
noncomputable def powerFunction (a : ℝ) (x : ℝ) : ℝ := x^a

-- State the theorem
theorem power_function_through_point_and_value :
  ∀ a : ℝ,
  powerFunction a 2 = Real.sqrt 2 / 2 →
  powerFunction a 8 = Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_and_value_l389_38925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l389_38914

noncomputable def f (n : ℝ) : ℝ :=
  if n > 2 then 1 / n else 2 * n^2 + 1

theorem f_range : Set.range f = {y : ℝ | y ∈ (Set.Ioo 0 (1/2)) ∪ (Set.Ici 1)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l389_38914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_international_society_theorem_l389_38939

def Member := ℕ
def Country := ℕ

structure Society :=
  (members : Finset Member)
  (countries : Finset Country)
  (memberCountry : Member → Country)
  (memberNumber : Member → ℕ)

theorem international_society_theorem (s : Society) 
  (h1 : s.members.card = 1978)
  (h2 : s.countries.card = 6)
  (h3 : ∀ m, m ∈ s.members → 1 ≤ s.memberNumber m ∧ s.memberNumber m ≤ 1978) :
  ∃ m, m ∈ s.members ∧ 
    ((∃ m1 m2, m1 ∈ s.members ∧ m2 ∈ s.members ∧ 
      s.memberCountry m = s.memberCountry m1 ∧ 
      s.memberCountry m = s.memberCountry m2 ∧ 
      s.memberNumber m = s.memberNumber m1 + s.memberNumber m2) ∨
    (∃ m1, m1 ∈ s.members ∧ 
      s.memberCountry m = s.memberCountry m1 ∧ 
      s.memberNumber m = 2 * s.memberNumber m1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_international_society_theorem_l389_38939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_treasure_signs_l389_38937

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried" -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried" -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried" -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried" -/
def signs_3 : ℕ := 3

/-- Represents that only signs under which there is no treasure are truthful -/
def truthful_signs (n : ℕ) : Prop :=
  n ≤ total_trees ∧ (∀ m : ℕ, m ≤ n → ¬(m = signs_15 ∨ m = signs_8 ∨ m = signs_4 ∨ m = signs_3))

/-- The theorem stating that the minimum number of signs under which treasures can be buried is 15 -/
theorem min_treasure_signs : ∃ (n : ℕ), n = 15 ∧ 
  (∀ m : ℕ, m < n → ¬(truthful_signs m)) ∧
  (truthful_signs n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_treasure_signs_l389_38937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l389_38958

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem no_integer_solution (n : ℕ+) : ¬ (∃ k : ℤ, (n.val^2 + 1 : ℝ) / ((floor (n.val.sqrt))^2 + 2) = k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_l389_38958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_vector_dot_product_l389_38973

def a : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (2, 3)

theorem opposite_vector_dot_product 
  (b : ℝ × ℝ) 
  (h1 : ∃ (k : ℝ), k > 0 ∧ b = (-k * a.1, -k * a.2)) 
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 3 * Real.sqrt 5) :
  (c.1 - a.1, c.2 - a.2) • (c.1 - b.1, c.2 - b.2) = -10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_vector_dot_product_l389_38973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_are_parallel_l389_38905

/-- Two circles in a plane -/
structure TwoCircles where
  circle1 : Set (ℝ × ℝ)
  circle2 : Set (ℝ × ℝ)

/-- Points of intersection between two circles -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Lines passing through intersection points -/
structure ArbitraryLines where
  line1 : Set (ℝ × ℝ)
  line2 : Set (ℝ × ℝ)

/-- Points where arbitrary lines intersect the circles -/
structure IntersectionWithCircles where
  C : ℝ × ℝ
  D : ℝ × ℝ
  C' : ℝ × ℝ
  D' : ℝ × ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ (v : ℝ × ℝ), v ≠ (0, 0) ∧ ∀ (p q : ℝ × ℝ), p ∈ l1 ∧ q ∈ l2 → ∃ (t : ℝ), q - p = t • v

/-- The main theorem -/
theorem chords_are_parallel
  (circles : TwoCircles)
  (intersectionPoints : IntersectionPoints)
  (arbitraryLines : ArbitraryLines)
  (intersectionWithCircles : IntersectionWithCircles)
  (h1 : intersectionPoints.A ∈ circles.circle1 ∧ intersectionPoints.A ∈ circles.circle2)
  (h2 : intersectionPoints.B ∈ circles.circle1 ∧ intersectionPoints.B ∈ circles.circle2)
  (h3 : intersectionWithCircles.C ∈ circles.circle1 ∧ intersectionWithCircles.D ∈ circles.circle1)
  (h4 : intersectionWithCircles.C' ∈ circles.circle2 ∧ intersectionWithCircles.D' ∈ circles.circle2)
  (h5 : intersectionPoints.A ∈ arbitraryLines.line1 ∧ intersectionPoints.B ∈ arbitraryLines.line1)
  (h6 : intersectionPoints.A ∈ arbitraryLines.line2 ∧ intersectionPoints.B ∈ arbitraryLines.line2)
  (h7 : intersectionWithCircles.C ∈ arbitraryLines.line1 ∧ intersectionWithCircles.D ∈ arbitraryLines.line2)
  (h8 : intersectionWithCircles.C' ∈ arbitraryLines.line1 ∧ intersectionWithCircles.D' ∈ arbitraryLines.line2) :
  parallel {p : ℝ × ℝ | p = intersectionWithCircles.C ∨ p = intersectionWithCircles.D}
           {p : ℝ × ℝ | p = intersectionWithCircles.C' ∨ p = intersectionWithCircles.D'} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_are_parallel_l389_38905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_common_books_l389_38940

def total_books : ℕ := 12
def books_to_read : ℕ := 6
def common_books : ℕ := 3

theorem probability_of_common_books :
  (Nat.choose total_books common_books * Nat.choose (total_books - common_books) (books_to_read - common_books) * Nat.choose (total_books - common_books) (books_to_read - common_books)) / 
  (Nat.choose total_books books_to_read * Nat.choose total_books books_to_read : ℚ) = 220 / 153 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_common_books_l389_38940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_of_interest_is_fifteen_percent_l389_38921

/-- Calculates the rate of interest given principal, simple interest, and time. -/
noncomputable def calculate_rate_of_interest (principal : ℝ) (simple_interest : ℝ) (time : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

/-- Theorem: Given the specified conditions, the rate of interest is 15%. -/
theorem rate_of_interest_is_fifteen_percent 
  (principal : ℝ) 
  (simple_interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 400) 
  (h2 : simple_interest = 120) 
  (h3 : time = 2) : 
  calculate_rate_of_interest principal simple_interest time = 15 := by
  -- Unfold the definition of calculate_rate_of_interest
  unfold calculate_rate_of_interest
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_of_interest_is_fifteen_percent_l389_38921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_20_meters_l389_38926

/-- The distance between two consecutive trees in a garden -/
noncomputable def distance_between_trees (total_length : ℝ) (num_trees : ℕ) : ℝ :=
  total_length / (num_trees - 1 : ℝ)

/-- Theorem stating that the distance between consecutive trees is 20 meters -/
theorem distance_is_20_meters (total_length : ℝ) (num_trees : ℕ) 
  (h1 : total_length = 500)
  (h2 : num_trees = 26) :
  distance_between_trees total_length num_trees = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_20_meters_l389_38926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l389_38967

/-- A polynomial with real coefficients -/
def RealPolynomial := Polynomial ℝ

/-- The statement of the problem -/
theorem polynomial_functional_equation (k : ℕ+) (P : RealPolynomial) :
  (∀ x : ℝ, P.eval (P.eval x) = (P.eval x) ^ (k : ℕ)) →
  (∃ c : ℝ, P = Polynomial.C c) ∨ (P = Polynomial.X ^ (k : ℕ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l389_38967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_difference_l389_38972

noncomputable def vector_a (θ : Real) : Fin 2 → Real := ![Real.cos θ, Real.sin θ]
def vector_b : Fin 2 → Real := ![0, -1]

theorem max_magnitude_difference :
  ∃ (max : Real), max = 2 ∧ ∀ (θ : Real),
    ‖vector_a θ - vector_b‖ ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_difference_l389_38972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l389_38983

-- Define the curves and the line
def C₁ (x y : ℝ) : Prop := x^2 + y^2/3 = 1
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ + 4 * Real.sin θ
def L (ρ θ : ℝ) : Prop := θ = Real.pi/4 ∧ ρ > 0

-- Define the intersection points
def M (ρ θ : ℝ) : Prop := C₁ (ρ * Real.cos θ) (ρ * Real.sin θ) ∧ L ρ θ
def N (ρ θ : ℝ) : Prop := C₂ ρ θ ∧ L ρ θ

-- State the theorem
theorem intersection_distance :
  ∃ (ρ_M ρ_N θ_M θ_N : ℝ),
    M ρ_M θ_M ∧ N ρ_N θ_N ∧
    |ρ_M - ρ_N| = 3 * Real.sqrt 2 - Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l389_38983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_squares_l389_38932

theorem root_sum_squares (a b c d : ℝ) : 
  (a^4 - 15 * a^2 + 56 = 0) →
  (b^4 - 15 * b^2 + 56 = 0) →
  (c^4 - 15 * c^2 + 56 = 0) →
  (d^4 - 15 * d^2 + 56 = 0) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a^2 + b^2 + c^2 + d^2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_squares_l389_38932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l389_38946

noncomputable def f (x : ℝ) : ℝ := Real.sin x * (Real.sin x - Real.sqrt 3 * Real.cos x)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 12)

theorem g_symmetry : ∀ x : ℝ, g (Real.pi / 6 - x) = g (Real.pi / 6 + x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l389_38946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_bounds_l389_38954

/-- The eccentricity of an ellipse with specific properties -/
noncomputable def ellipse_eccentricity (θ : ℝ) : ℝ :=
  3 / (6 + 2 * Real.cos θ)

/-- Theorem stating the bounds of the eccentricity -/
theorem ellipse_eccentricity_bounds :
  ∀ θ : ℝ, 3/8 ≤ ellipse_eccentricity θ ∧ ellipse_eccentricity θ ≤ 3/4 := by
  sorry

#check ellipse_eccentricity_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_bounds_l389_38954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equals_factorial_l389_38976

def sequenceA (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | k + 1 => (k + 3) * sequenceA k

theorem sequence_equals_factorial (n : ℕ) : sequenceA n = Nat.factorial (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equals_factorial_l389_38976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l389_38988

theorem problem_statement (k : ℝ) : 
  (1/2 : ℝ)^(25 : ℝ) * (1/81 : ℝ)^k = (1/18 : ℝ)^(25 : ℝ) → k = -12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l389_38988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_count_l389_38911

theorem balloon_count 
  (gold silver black blue green : ℕ) :
  gold = 141 →
  silver = (gold / 3) * 5 →
  black = silver / 2 →
  blue = black / 2 →
  green = (blue / 4) * 3 →
  gold + silver + black + blue + green = 593 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_count_l389_38911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_count_l389_38956

theorem library_book_count
  (slots_per_shelf : ℕ)
  (books_per_slot : ℕ)
  (total_shelves : ℕ)
  (empty_slots : ℕ)
  (partial_slots : ℕ)
  (books_in_partial : ℕ) :
  slots_per_shelf = 6 →
  books_per_slot = 8 →
  total_shelves = 16 →
  empty_slots = 5 →
  partial_slots = 1 →
  books_in_partial = 6 →
  (total_shelves * slots_per_shelf - empty_slots - partial_slots) * books_per_slot + books_in_partial = 726 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_count_l389_38956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l389_38907

/-- Given a triangle ABC with the following properties:
    1. cos B - cos((A+C)/2) = 0
    2. 8a = 3c where a and c are sides opposite to angles A and C respectively
    3. The altitude from AC is 12√3/7
    Prove that the measure of angle B is π/3 and the perimeter of the triangle is 18 -/
theorem triangle_properties (A B C : Real) (a b c : Real) (h : Real) : 
  A + B + C = Real.pi →
  Real.cos B - Real.cos ((A + C) / 2) = 0 →
  8 * a = 3 * c →
  h = 12 * Real.sqrt 3 / 7 →
  b * h = a * c * Real.sin B →
  B = Real.pi / 3 ∧ a + b + c = 18 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l389_38907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l389_38969

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else Real.log x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  -1 ≤ a ∧ a < 1/2 :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l389_38969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_problem_l389_38984

/-- Calculates the upstream distance given the conditions of the swimming problem -/
noncomputable def upstream_distance (speed_still : ℝ) (downstream_distance : ℝ) (time : ℝ) : ℝ :=
  let current_speed := downstream_distance / time - speed_still
  (speed_still - current_speed) * time

/-- Theorem stating the upstream distance under given conditions -/
theorem upstream_distance_problem :
  upstream_distance 11.5 51 3 = 18 := by
  -- Unfold the definition of upstream_distance
  unfold upstream_distance
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_problem_l389_38984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_l389_38935

/-- The side length of the pentagon and triangles -/
def side_length : ℝ := 1

/-- The number of triangles -/
def num_triangles : ℕ := 15

/-- The area of a regular pentagon with side length s -/
noncomputable def pentagon_area (s : ℝ) : ℝ := (1/4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) * s^2

/-- The area of an equilateral triangle with side length s -/
noncomputable def triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The region P formed by the union of the pentagon and all triangles -/
noncomputable def area_P : ℝ := pentagon_area side_length + num_triangles * triangle_area side_length

/-- The region Q, which is the smallest convex polygon containing P -/
noncomputable def area_Q : ℝ := pentagon_area (3 * side_length)

/-- The theorem to be proved -/
theorem area_difference : 
  area_Q - area_P = 2 * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) - (15 * Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_l389_38935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_cd_length_l389_38906

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  /-- Length of BD -/
  bd : ℝ
  /-- Angle DBA in radians -/
  angle_dba : ℝ
  /-- Angle BDC in radians -/
  angle_bdc : ℝ
  /-- Ratio of BC to AD -/
  ratio_bc_ad : ℝ
  /-- AD is parallel to BC -/
  ad_parallel_bc : Bool
  /-- BD equals 2 -/
  bd_eq_two : bd = 2
  /-- Angle DBA equals 30 degrees (π/6 radians) -/
  angle_dba_eq_thirty_deg : angle_dba = π/6
  /-- Angle BDC equals 60 degrees (π/3 radians) -/
  angle_bdc_eq_sixty_deg : angle_bdc = π/3
  /-- Ratio of BC to AD is 7:3 -/
  ratio_bc_ad_eq_seven_thirds : ratio_bc_ad = 7/3

/-- The length of CD in the trapezoid -/
noncomputable def length_cd (t : Trapezoid) : ℝ := 8/3

/-- Theorem stating that the length of CD in the trapezoid is 8/3 -/
theorem trapezoid_cd_length (t : Trapezoid) : length_cd t = 8/3 := by
  -- Unfold the definition of length_cd
  unfold length_cd
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_cd_length_l389_38906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l389_38953

theorem sin_cos_identity (α β : Real) 
  (h1 : Real.sin α = Real.cos β) 
  (h2 : Real.cos α = Real.sin (2 * β)) : 
  (Real.sin β)^2 + (Real.cos α)^2 = 0 ∨ (Real.sin β)^2 + (Real.cos α)^2 = 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l389_38953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_count_l389_38995

/-- Sum of digits of a positive integer -/
def starSum (x : ℕ) : ℕ := sorry

/-- Set of positive integers less than 10^6 whose digits sum to 15 -/
def S : Set ℕ := {n | starSum n = 15 ∧ n < 1000000}

/-- Number of elements in S -/
def m : ℕ := Finset.card (Finset.filter (fun n => starSum n = 15 ∧ n < 1000000) (Finset.range 1000000))

theorem digit_sum_of_count : starSum m = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_count_l389_38995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_max_a_achieved_l389_38941

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x) / x^2 - x - a / x + 2 * Real.exp 1

/-- The maximum value of a -/
noncomputable def max_a : ℝ := Real.exp 2 + 1 / Real.exp 1

/-- Theorem stating that if f has a zero point, then the maximum value of a is e^2 + 1/e -/
theorem max_a_value (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f a x = 0) →
  a ≤ max_a := by
  sorry

/-- Theorem stating that the maximum value of a is indeed achieved -/
theorem max_a_achieved :
  ∃ a : ℝ, ∃ x : ℝ, x > 0 ∧ f a x = 0 ∧ a = max_a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_max_a_achieved_l389_38941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_parabola_satisfies_conditions_l389_38978

/-- Represents a parabola with specific properties -/
structure Parabola where
  -- Equation coefficients
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  -- Conditions
  c_positive : c > 0
  gcd_one : Int.gcd a (Int.gcd b (Int.gcd c (Int.gcd d (Int.gcd e f)))) = 1
  passes_through : a * 2^2 + b * 2 * 6 + c * 6^2 + d * 2 + e * 6 + f = 0
  focus_y : ∃ (x : ℚ), a * x^2 + b * x * 4 + c * 4^2 + d * x + e * 4 + f = 0
  symmetry_parallel_x : b = 0 ∧ a = 0
  vertex_on_y : ∃ (y : ℚ), c * y^2 + e * y + f = 0

/-- The specific parabola we want to prove -/
def specificParabola : Parabola := {
  a := 0,
  b := 0,
  c := 1,
  d := -2,
  e := -8,
  f := 16,
  c_positive := by simp
  gcd_one := by simp
  passes_through := by simp
  focus_y := by sorry
  symmetry_parallel_x := by simp
  vertex_on_y := by sorry
}

/-- Theorem stating that our specific parabola satisfies all conditions -/
theorem specific_parabola_satisfies_conditions : 
  specificParabola.a = 0 ∧
  specificParabola.b = 0 ∧
  specificParabola.c = 1 ∧
  specificParabola.d = -2 ∧
  specificParabola.e = -8 ∧
  specificParabola.f = 16 := by simp [specificParabola]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_parabola_satisfies_conditions_l389_38978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_on_circle_l389_38980

/-- Circle with center (-2, 0) and radius 1 -/
def Circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

/-- Slope of the line connecting P(x,y) and M(1,2) -/
noncomputable def Slope (x y : ℝ) : ℝ := (y - 2) / (x - 1)

theorem max_slope_on_circle :
  ∃ (max : ℝ), max = (3 + Real.sqrt 3) / 4 ∧
  (∀ (x y : ℝ), Circle x y → Slope x y ≤ max) ∧
  (∃ (x y : ℝ), Circle x y ∧ Slope x y = max) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_on_circle_l389_38980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_after_discounts_l389_38965

noncomputable def list_price : ℝ := 150
noncomputable def first_discount : ℝ := 19.954259576901087
noncomputable def second_discount : ℝ := 12.55

noncomputable def price_after_first_discount : ℝ := list_price * (1 - first_discount / 100)
noncomputable def final_price : ℝ := price_after_first_discount * (1 - second_discount / 100)

theorem shirt_price_after_discounts :
  |final_price - 105| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_after_discounts_l389_38965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l389_38992

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < g x then g x + x + 4
  else g x - x

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-9/4) 0 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l389_38992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_locus_l389_38994

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The focus of the parabola x² = 4y -/
def parabolaFocus : Point :=
  ⟨0, 1⟩

/-- Theorem about the standard equation of the ellipse and the locus of point R -/
theorem ellipse_and_locus 
  (M : Ellipse) 
  (h_ecc : eccentricity M = Real.sqrt 2 / 2)
  (h_focus : M.b = parabolaFocus.y) :
  (∀ (x y : ℝ), x^2 / 2 + y^2 = 1 ↔ x^2 / M.a^2 + y^2 / M.b^2 = 1) ∧
  (∀ (P Q R : Point), 
    (P.x^2 / M.a^2 + P.y^2 / M.b^2 = 1) → 
    (Q.x^2 / M.a^2 + Q.y^2 / M.b^2 = 1) → 
    (P.x * Q.x + P.y * Q.y = 0) →
    (R.x * (P.y - Q.y) = R.y * (P.x - Q.x)) →
    R.x^2 + R.y^2 = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_locus_l389_38994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_path_length_l389_38955

/-- The length of a spiral path on a cylindrical post -/
noncomputable def spiral_path_length (post_height : ℝ) (post_circumference : ℝ) (rise_per_circuit : ℝ) : ℝ :=
  let circuits := post_height / rise_per_circuit
  let horizontal_distance := circuits * post_circumference
  Real.sqrt (post_height ^ 2 + horizontal_distance ^ 2)

/-- Theorem stating that the spiral path length is 15 feet for the given conditions -/
theorem squirrel_path_length :
  spiral_path_length 12 3 4 = 15 := by
  -- Unfold the definition of spiral_path_length
  unfold spiral_path_length
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_path_length_l389_38955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_odd_function_evaluation_l389_38993

-- Part 1
theorem expression_evaluation :
  (((Real.sqrt 121) / 2018 - 5) ^ 0) + (2 ^ (-2 : ℝ) * ((9/4 : ℝ) ^ (-1/2 : ℝ))) - (Real.log 3 / Real.log 4) * (Real.log (Real.sqrt 8) / Real.log 3) = 5/12 :=
by sorry

-- Part 2
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^(2-m)

theorem odd_function_evaluation (m : ℝ) :
  (∀ x ∈ Set.Icc (-3-m) (m^2-m), f m x = x^(2-m)) →
  (∀ x ∈ Set.Icc (-3-m) (m^2-m), f m (-x) = -(f m x)) →
  f m m = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_odd_function_evaluation_l389_38993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l389_38961

noncomputable def f (A : ℝ) (x : ℝ) : ℝ := A * Real.cos (x / 4 + Real.pi / 6)

theorem function_properties (A : ℝ) (α β : ℝ) :
  f A (Real.pi / 3) = Real.sqrt 2 →
  α ∈ Set.Icc 0 (Real.pi / 2) →
  β ∈ Set.Icc 0 (Real.pi / 2) →
  f A (4 * α + 4 * Real.pi / 3) = -30 / 17 →
  f A (4 * β - 2 * Real.pi / 3) = 8 / 5 →
  A = 2 ∧ Real.cos (α + β) = -13 / 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l389_38961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l389_38930

/-- Given vectors a, b, and c in ℝ², prove that if (a + λ*b) is parallel to c, then λ = 1/2. -/
theorem parallel_vectors_lambda (a b c : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 2) →
  b = (1, 0) →
  c = (3, 4) →
  (∃ (k : ℝ), k ≠ 0 ∧ (a.1 + lambda * b.1, a.2 + lambda * b.2) = (k * c.1, k * c.2)) →
  lambda = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l389_38930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_reflection_properties_l389_38985

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if -5 ≤ x ∧ x ≤ -2 then x + 3
  else if -2 < x ∧ x ≤ 1 then -Real.sqrt (9 - (x + 1)^2) + 1
  else if 1 < x ∧ x ≤ 4 then 2*(x - 1) - 2
  else 0  -- Undefined outside the domain

-- Define h(x) = g(-x)
noncomputable def h (x : ℝ) : ℝ := g (-x)

-- State the theorem
theorem g_reflection_properties :
  (∀ x ∈ Set.Icc 2 5, ∃ m b : ℝ, m < 0 ∧ h x = m*x + b) ∧
  (∀ x ∈ Set.Icc (-1) 2, ∃ a b r : ℝ, h x = -Real.sqrt (r^2 - (x - a)^2) + b) ∧
  (∀ x ∈ Set.Icc (-4) (-1), ∃ m b : ℝ, m > 0 ∧ h x = m*x + b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_reflection_properties_l389_38985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l389_38991

/-- Given a triangle ABC with sides AB = 8, AC = 7, and BC = 5, 
    prove that (cos((A + B)/2)/sin(C/2)) - (sin((A + B)/2)/cos(C/2)) = 0 -/
theorem triangle_trig_identity (A B C : ℝ) : 
  (8 : ℝ) = Real.sin C / Real.sin A → 
  (7 : ℝ) = Real.sin C / Real.sin B → 
  (5 : ℝ) = Real.sin A / Real.sin B → 
  (Real.cos ((A + B)/2) / Real.sin (C/2)) - (Real.sin ((A + B)/2) / Real.cos (C/2)) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l389_38991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_price_reduction_l389_38945

/-- Represents the price of wheat in Rupees per kg -/
structure WheatPrice where
  price : ℝ
  price_positive : price > 0

/-- Given a 15% reduction in wheat price, proves that if 3 kg more can be bought for 500 Rs, 
    then the reduced price is approximately 25 Rs/kg -/
theorem wheat_price_reduction (original : WheatPrice) : 
  let reduced := WheatPrice.mk (original.price * 0.85) (by 
    have h : original.price * 0.85 > 0 := by
      apply mul_pos
      · exact original.price_positive
      · norm_num
    exact h)
  (500 / reduced.price = 500 / original.price + 3) → 
  ∃ ε > 0, |reduced.price - 25| < ε := by sorry

#check wheat_price_reduction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_price_reduction_l389_38945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_of_64_l389_38913

def star (a b : ℤ) : ℚ := (a ^ 2 : ℚ) / b

def is_positive_integer (q : ℚ) : Prop := ∃ (n : ℕ), q = n

theorem count_divisors_of_64 :
  let divisors := Finset.filter (fun x => x > 0 ∧ 64 % x = 0) (Finset.range 65)
  (∀ x ∈ divisors, is_positive_integer (star 8 x)) →
  Finset.card divisors = 7 :=
by
  intro divisors h
  sorry

#eval Finset.card (Finset.filter (fun x => x > 0 ∧ 64 % x = 0) (Finset.range 65))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_of_64_l389_38913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_of_circle_l389_38951

/-- Given a circle with center (2,1) and a point (6,3) on the circle,
    the slope of the tangent line at (6,3) is -2. -/
theorem tangent_slope_of_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (2, 1) → point = (6, 3) → 
  ((-1) / ((point.2 - center.2) / (point.1 - center.1))) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_of_circle_l389_38951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_specific_trapezoid_l389_38920

/-- Represents a trapezoid ABCD with midpoints E and F -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  altitude : ℝ
  parallel : AB ≠ CD  -- Ensuring AB and CD are different (parallel sides)

/-- Calculate the area of quadrilateral EFCD in the given trapezoid -/
noncomputable def area_EFCD (t : Trapezoid) : ℝ :=
  (t.altitude / 2) * ((t.AB + t.CD) / 2 + t.CD) / 2

/-- Theorem stating the area of EFCD in the specific trapezoid -/
theorem area_EFCD_specific_trapezoid :
  let t : Trapezoid := { AB := 10, CD := 25, altitude := 15, parallel := by norm_num }
  area_EFCD t = 159.375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_specific_trapezoid_l389_38920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_ellipse_area_l389_38950

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ

/-- Checks if a point lies on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.center.x)^2 / e.semiMajorAxis^2 + (p.y - e.center.y)^2 / e.semiMinorAxis^2 = 1

/-- Calculates the area of an ellipse -/
noncomputable def ellipseArea (e : Ellipse) : ℝ :=
  Real.pi * e.semiMajorAxis * e.semiMinorAxis

/-- Theorem: The area of the specific ellipse is 42π -/
theorem specific_ellipse_area :
  ∃ (e : Ellipse),
    e.center = Point.mk 3 (-1) ∧
    e.semiMajorAxis = 10 ∧
    isOnEllipse e (Point.mk (-7) (-1)) ∧
    isOnEllipse e (Point.mk 13 (-1)) ∧
    isOnEllipse e (Point.mk 10 2) ∧
    ellipseArea e = 42 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_ellipse_area_l389_38950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_not_subsets_l389_38901

-- Define the sets A and B
def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {2, 3}

-- Define the union of A and B
def AUB : Finset ℕ := A ∪ B

-- Define the set of all subsets of AUB
def all_subsets : Finset (Finset ℕ) := Finset.powerset AUB

-- Define the set of subsets that are not subsets of AUB
def not_subsets : Finset (Finset ℕ) := all_subsets.filter (λ P => ¬(P ⊆ AUB))

-- State the theorem
theorem count_not_subsets : Finset.card not_subsets = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_not_subsets_l389_38901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l389_38979

def second_order_difference (a : ℕ → ℝ) : ℕ → ℝ := λ n ↦ a (n + 2) - 2 * a (n + 1) + a n

theorem sequence_problem (a : ℕ → ℝ) 
  (h1 : ∀ n, second_order_difference a n = 16)
  (h2 : a 63 = 10)
  (h3 : a 89 = 10) :
  a 51 = 3658 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l389_38979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_sequence_l389_38999

def is_valid_sequence (seq : List Nat) : Prop :=
  seq.length ≥ 2 ∧
  (∀ n ∈ seq, 8 ≤ n ∧ n < 64) ∧
  (∀ i, i + 1 < seq.length → seq.get! (i + 1) = seq.get! i + 8) ∧
  seq.head!.mod 8 ≠ 2 ∧ seq.head!.div 8 ≠ 2

def octal_to_decimal (oct : Nat) : Nat :=
  let digits := oct.digits 8
  digits.foldr (fun d acc => acc * 8 + d) 0

def has_two_prime_factors_diff_two (n : Nat) : Prop :=
  ∃ p q : Nat, Nat.Prime p ∧ Nat.Prime q ∧ q = p + 2 ∧ n = p * q

theorem unique_valid_sequence :
  ∃! seq : List Nat, is_valid_sequence seq ∧
    has_two_prime_factors_diff_two (octal_to_decimal (seq.foldl (fun acc d => acc * 100 + d) 0)) ∧
    seq = [33, 43] := by
  sorry

#eval octal_to_decimal (3343)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_sequence_l389_38999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_P_l389_38918

-- Define a type for the permutation of {1, 2, ..., 9}
def Permutation9 := Fin 9 → Fin 9

-- Define the function P
def P (perm : Permutation9) : ℕ :=
  let values := fun i => (perm i).val + 1
  (values 0) * (values 1) * (values 2) +
  (values 3) * (values 4) * (values 5) +
  (values 6) * (values 7) * (values 8)

-- State the theorem
theorem min_value_P :
  (∀ perm : Permutation9, Function.Bijective perm → P perm ≥ 214) ∧
  (∃ perm : Permutation9, Function.Bijective perm ∧ P perm = 214) := by
  sorry

-- Example of a permutation achieving the minimum value
def min_perm : Permutation9 := fun i =>
  match i with
  | 0 => 1  -- 2
  | 1 => 4  -- 5
  | 2 => 6  -- 7
  | 3 => 0  -- 1
  | 4 => 7  -- 8
  | 5 => 8  -- 9
  | 6 => 2  -- 3
  | 7 => 3  -- 4
  | 8 => 5  -- 6

-- Verify that min_perm achieves P = 214
example : P min_perm = 214 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_P_l389_38918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l389_38904

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- State the theorem
theorem sin_cos_product (α : ℝ) :
  (∀ x, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → f x = Real.sin x) →
  f (Real.sin α) + f (Real.cos α - 1/2) = 0 →
  Real.sin α * Real.cos α = -3/8 := by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l389_38904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_multiple_l389_38990

/-- The multiple of football ticket price to 8 movie tickets -/
def multiple : ℕ → Prop := sorry

/-- The cost of a single movie ticket in dollars -/
def movie_ticket_cost : ℕ := 30

/-- The cost of a single football ticket in dollars -/
def football_ticket_cost : ℕ → ℕ := sorry

theorem find_multiple :
  ∀ k : ℕ,
  multiple k →
  8 * movie_ticket_cost = k * football_ticket_cost k →
  8 * movie_ticket_cost + 5 * football_ticket_cost k = 840 →
  k = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_multiple_l389_38990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_flowers_bloom_thursday_l389_38997

/-- Represents the days of the week -/
inductive Day : Type
  | monday : Day
  | tuesday : Day
  | wednesday : Day
  | thursday : Day
  | friday : Day
  | saturday : Day
  | sunday : Day

/-- Represents the types of flowers -/
inductive Flower : Type
  | sunflower : Flower
  | lily : Flower
  | peony : Flower

/-- Represents whether a flower blooms on a given day -/
def blooms : Flower → Day → Prop := sorry

/-- No flower blooms for three consecutive days -/
axiom no_three_consecutive_days (f : Flower) :
  ¬∃ (d1 d2 d3 : Day), blooms f d1 ∧ blooms f d2 ∧ blooms f d3 ∧
  (d2 = Day.tuesday ∨ d2 = Day.wednesday ∨ d2 = Day.thursday ∨ d2 = Day.friday ∨ d2 = Day.saturday)

/-- At most one day when any two types of flowers do not bloom together -/
axiom at_most_one_day_two_not_bloom :
  ∀ (f1 f2 : Flower), f1 ≠ f2 →
  (∃! (d : Day), ¬(blooms f1 d ∧ blooms f2 d)) ∨
  (∀ (d : Day), blooms f1 d ∧ blooms f2 d)

/-- Sunflowers do not bloom on Tuesday, Thursday, and Sunday -/
axiom sunflower_not_bloom :
  ¬(blooms Flower.sunflower Day.tuesday) ∧
  ¬(blooms Flower.sunflower Day.thursday) ∧
  ¬(blooms Flower.sunflower Day.sunday)

/-- Lilies do not bloom on Thursday and Saturday -/
axiom lily_not_bloom :
  ¬(blooms Flower.lily Day.thursday) ∧
  ¬(blooms Flower.lily Day.saturday)

/-- Peonies do not bloom on Sunday -/
axiom peony_not_bloom :
  ¬(blooms Flower.peony Day.sunday)

/-- There is only one day when all three types of flowers bloom simultaneously -/
axiom one_day_all_bloom :
  ∃! (d : Day), blooms Flower.sunflower d ∧ blooms Flower.lily d ∧ blooms Flower.peony d

theorem all_flowers_bloom_thursday :
  ∃! (d : Day), blooms Flower.sunflower d ∧ blooms Flower.lily d ∧ blooms Flower.peony d ∧ d = Day.thursday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_flowers_bloom_thursday_l389_38997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_common_remainder_l389_38903

theorem least_number_with_common_remainder (n : ℕ) : 
  (n = 130) →
  (∀ d : ℕ, d ∈ ({6, 7, 9, 18} : Set ℕ) → n % d = 4) →
  (∀ m : ℕ, m < n → ∃ d : ℕ, d ∈ ({6, 7, 9, 18} : Set ℕ) ∧ m % d ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_common_remainder_l389_38903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_trig_identities_l389_38929

theorem unit_circle_trig_identities (α : ℝ) :
  (∃ P : ℝ × ℝ, P.1 = 4/5 ∧ P.2 = -3/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
   Real.cos α = P.1 ∧ Real.sin α = P.2) →
  (Real.cos α = 4/5 ∧ Real.tan α = -3/4 ∧ Real.sin (α + π) = 3/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_trig_identities_l389_38929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dans_initial_money_l389_38917

/-- Represents the amount of money in dollars -/
structure Money where
  amount : ℕ

/-- Dan's initial amount of money -/
def initial_money : Money := ⟨10⟩

/-- Cost of the chocolate -/
def chocolate_cost : Money := ⟨3⟩

/-- Cost of the candy bar -/
def candy_bar_cost : Money := ⟨7⟩

/-- The difference in cost between the candy bar and chocolate -/
def cost_difference : Money := ⟨4⟩

/-- Addition operation for Money -/
instance : Add Money where
  add a b := ⟨a.amount + b.amount⟩

theorem dans_initial_money :
  initial_money = chocolate_cost + candy_bar_cost ∧
  candy_bar_cost.amount = chocolate_cost.amount + cost_difference.amount :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dans_initial_money_l389_38917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l389_38986

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.sin x + 2 * x^2 > Real.cos x) ↔ (∀ x : ℝ, Real.sin x + 2 * x^2 ≤ Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l389_38986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_difference_one_l389_38968

theorem prime_power_difference_one (p q : ℕ) (r s : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : r > 1) (hs : s > 1) :
  (|Int.ofNat (p^r) - Int.ofNat (q^s)| = 1) ↔ 
  ((p = 3 ∧ q = 2 ∧ r = 2 ∧ s = 3) ∨ (p = 2 ∧ q = 3 ∧ r = 3 ∧ s = 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_difference_one_l389_38968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l389_38957

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (1/4)

-- State the theorem
theorem monotonic_increasing_interval :
  ∃ (a : ℝ), ∀ (x y : ℝ), x < a ∧ y < a ∧ x < y → f x > f y :=
by
  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l389_38957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l389_38928

open Set Real

noncomputable def f (x : ℝ) := Real.log (sin x) + Real.sqrt (cos x - Real.sqrt 2 / 2)

theorem domain_of_f :
  {x : ℝ | ∃ k : ℤ, 2 * k * π < x ∧ x ≤ 2 * k * π + π / 4} =
  {x : ℝ | sin x > 0 ∧ cos x - Real.sqrt 2 / 2 ≥ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l389_38928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_bound_smallest_constant_optimal_l389_38970

/-- The number of positive divisors of a positive integer n -/
def d (n : ℕ+) : ℕ := sorry

/-- The smallest constant c such that d(n) ≤ c√n for all positive integers n -/
noncomputable def smallest_constant : ℝ := Real.sqrt 3

theorem divisor_bound (n : ℕ+) : (d n : ℝ) ≤ smallest_constant * Real.sqrt n.val := by sorry

theorem smallest_constant_optimal (c : ℝ) : 
  (∀ n : ℕ+, (d n : ℝ) ≤ c * Real.sqrt n.val) → c ≥ smallest_constant := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_bound_smallest_constant_optimal_l389_38970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patrol_boat_problem_l389_38902

/-- The speed of the patrol boat in still water (km/h) -/
noncomputable def boat_speed : ℝ := 40

/-- The time elapsed since the raft started drifting (hours) -/
noncomputable def elapsed_time : ℝ := 1/2

/-- The speed of water flow (km/h) -/
noncomputable def water_flow_speed : ℝ := 40/3

/-- The time needed to catch up with the raft (hours) -/
noncomputable def catch_up_time : ℝ := 1/6

theorem patrol_boat_problem :
  (∀ x : ℝ, boat_speed + x = 2 * (boat_speed - x) → x = water_flow_speed) ∧
  (∀ y : ℝ, (boat_speed + water_flow_speed) * y = (y + elapsed_time) * water_flow_speed → y = catch_up_time) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patrol_boat_problem_l389_38902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l389_38923

-- Define the function f(x) = log₀.₅(x² - 4)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4) / Real.log 0.5

-- Define the domain of f
def domain : Set ℝ := {x | x < -2 ∨ x > 2}

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x ∈ domain, ∀ y ∈ domain,
    x > y → f x < f y ↔ x > 2 ∧ y > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l389_38923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selection_uses_golden_ratio_l389_38960

/-- The golden ratio -/
noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

/-- The optimal selection method developed by Hua Luogeng -/
noncomputable def optimal_selection_method : ℝ → ℝ := 
  λ x => x  -- placeholder definition

/-- Theorem: The optimal selection method uses the golden ratio -/
theorem optimal_selection_uses_golden_ratio : 
  ∃ (f : ℝ → ℝ), f = optimal_selection_method ∧ f golden_ratio = golden_ratio := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selection_uses_golden_ratio_l389_38960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_special_n_l389_38949

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The problem statement -/
theorem digit_sum_of_special_n :
  ∃ (n : ℕ), 
    n > 0 ∧ 
    (Nat.factorial (n + 2) + Nat.factorial (n + 3) = Nat.factorial n * 1320) ∧ 
    sum_of_digits n = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_special_n_l389_38949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_equality_l389_38908

/-- The limit of (n+5)(1-3n) / (2n+1)^2 as n approaches infinity is -3/4 -/
theorem limit_fraction_equality : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((n : ℝ) + 5) * (1 - 3*(n : ℝ)) / ((2*(n : ℝ) + 1)^2) - (-3/4)| < ε := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_equality_l389_38908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l389_38900

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 110 m, traveling at 72 km/h, 
    takes approximately 12.2 seconds to cross a bridge of length 134 m -/
theorem train_crossing_bridge :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |train_crossing_time 110 72 134 - 12.2| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l389_38900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l389_38947

/-- An increasing function f satisfying the given properties -/
structure SpecialFunction where
  f : ℝ → ℝ
  increasing : ∀ x y, x < y → f x < f y
  multiplicative : ∀ a b, a > 0 → b > 0 → f (a * b) = f a + f b
  f_2 : f 2 = 1

/-- The main theorem stating the properties of the special function -/
theorem special_function_properties (φ : SpecialFunction) :
  φ.f 1 = 0 ∧ φ.f 4 = 2 ∧
  Set.Ioo (-4 : ℝ) 0 ∪ Set.Ioo 0 4 = {x : ℝ | φ.f (x^2) < 2 * φ.f 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l389_38947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarters_to_dimes_ratio_l389_38987

def quarter_value : ℚ := 25 / 100
def dime_value : ℚ := 10 / 100

theorem quarters_to_dimes_ratio 
  (total_money : ℚ) 
  (num_quarters : ℕ) 
  (quarters_dimes_relation : ℕ → ℕ) :
  total_money = 1015 / 100 →
  num_quarters = 21 →
  (λ x => num_quarters - (quarters_dimes_relation x + 7)) = 0 →
  (num_quarters : ℚ) * quarter_value + 
    (quarters_dimes_relation (num_quarters - 7)) * dime_value = total_money →
  (num_quarters : ℚ) / (quarters_dimes_relation (num_quarters - 7)) = 3 / 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarters_to_dimes_ratio_l389_38987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_time_l389_38943

/-- The time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_km_hr : ℝ) : ℝ :=
  let train_speed_m_s := train_speed_km_hr * 1000 / 3600
  train_length / train_speed_m_s

/-- Theorem stating that a 500m long train traveling at 350 km/hr takes approximately 5.14 seconds to cross an electric pole -/
theorem train_crossing_approx_time :
  ∃ ε > 0, |train_crossing_time 500 350 - 5.14| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_time_l389_38943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l389_38975

-- Define the vertices of the quadrilateral
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (8, 10)
def D : ℝ × ℝ := (8, 0)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem quadrilateral_perimeter : 
  distance A B + distance B C + distance C D + distance D A = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l389_38975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_points_l389_38959

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2
def g (x : ℝ) : ℝ := x - 1

theorem min_distance_between_points (x₁ x₂ : ℝ) 
  (h₁ : x₁ ≥ 0) 
  (h₂ : x₂ > 0) 
  (h₃ : f x₁ = g x₂) : 
  ∃ d : ℝ, d = 2 ∧ ∀ y₁ y₂ : ℝ, y₁ ≥ 0 → y₂ > 0 → f y₁ = g y₂ → |y₂ - y₁| ≥ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_points_l389_38959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l389_38916

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (a - 1) * x + 4 - 2 * a
  else 1 + Real.log x / Real.log 2

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  1 < a ∧ a ≤ 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l389_38916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l389_38936

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The equality i² = -1 -/
axiom i_squared : i * i = -1

/-- The complex fraction to be simplified -/
noncomputable def complex_fraction : ℂ := (2 - i) / (1 + 4 * i)

/-- The simplified form of the complex fraction -/
noncomputable def simplified_form : ℂ := -2/17 - 9/17 * i

/-- Theorem stating that the complex fraction equals its simplified form -/
theorem complex_fraction_simplification :
  complex_fraction = simplified_form := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l389_38936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_charging_theorem_l389_38915

/-- Represents the charging characteristics of a battery -/
structure BatteryCharging where
  initial_charge_percent : ℚ
  initial_charge_time : ℚ
  additional_charge_time : ℚ

/-- Calculates the total charging time and final battery percentage -/
def calculate_charging (b : BatteryCharging) : ℚ × ℚ :=
  let charge_rate := b.initial_charge_percent / b.initial_charge_time
  let additional_charge := charge_rate * b.additional_charge_time
  let total_charge := b.initial_charge_percent + additional_charge
  let total_time := b.initial_charge_time + b.additional_charge_time
  (total_time, total_charge)

theorem battery_charging_theorem (b : BatteryCharging) 
  (h1 : b.initial_charge_percent = 20)
  (h2 : b.initial_charge_time = 60)
  (h3 : b.additional_charge_time = 120) :
  calculate_charging b = (180, 60) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_charging_theorem_l389_38915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_pyramid_volume_volume_of_given_smaller_pyramid_l389_38909

/-- Represents a right square pyramid -/
structure RightSquarePyramid where
  baseEdge : ℝ
  slantEdge : ℝ
  height : ℝ

/-- Represents a plane intersecting the pyramid parallel to its base -/
structure IntersectingPlane where
  heightFromBase : ℝ

/-- Calculates the volume of the smaller pyramid formed above the intersecting plane -/
noncomputable def volumeOfSmallerPyramid (p : RightSquarePyramid) (plane : IntersectingPlane) : ℝ :=
  let smallerPyramidHeight := p.height - plane.heightFromBase
  let topSquareEdge := p.baseEdge * (smallerPyramidHeight / p.height)
  (1/3) * (topSquareEdge^2) * smallerPyramidHeight

/-- Theorem stating the volume of the smaller pyramid -/
theorem smaller_pyramid_volume (p : RightSquarePyramid) (plane : IntersectingPlane) :
  volumeOfSmallerPyramid p plane = 
    (1/3) * ((p.baseEdge * ((p.height - plane.heightFromBase) / p.height))^2) * (p.height - plane.heightFromBase) :=
by sorry

/-- Given problem instance -/
noncomputable def givenPyramid : RightSquarePyramid :=
  { baseEdge := 10 * Real.sqrt 2
  , slantEdge := 12
  , height := Real.sqrt 44 }

def givenPlane : IntersectingPlane :=
  { heightFromBase := 4 }

/-- Theorem for the specific problem instance -/
theorem volume_of_given_smaller_pyramid :
  volumeOfSmallerPyramid givenPyramid givenPlane = 
    (1/3) * ((10 * Real.sqrt 2 * ((Real.sqrt 44 - 4) / Real.sqrt 44))^2) * (Real.sqrt 44 - 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_pyramid_volume_volume_of_given_smaller_pyramid_l389_38909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_y_axis_properties_l389_38912

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Given points A and B, prove properties about point M on y-axis -/
theorem point_on_y_axis_properties 
  (A B : Point3D) (h_A : A = ⟨3, 0, 1⟩) (h_B : B = ⟨1, 0, -3⟩) :
  (∀ y : ℝ, distance ⟨0, y, 0⟩ A = distance ⟨0, y, 0⟩ B) ∧ 
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧
    distance ⟨0, y₁, 0⟩ A = distance ⟨0, y₁, 0⟩ B ∧ 
    distance ⟨0, y₁, 0⟩ A = distance A B ∧
    distance ⟨0, y₂, 0⟩ A = distance ⟨0, y₂, 0⟩ B ∧ 
    distance ⟨0, y₂, 0⟩ A = distance A B ∧
    (y₁ = Real.sqrt 10 ∧ y₂ = -Real.sqrt 10 ∨ 
     y₁ = -Real.sqrt 10 ∧ y₂ = Real.sqrt 10)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_y_axis_properties_l389_38912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_p_value_l389_38966

/-- Probability function for a regular polygon -/
def p (n : ℕ) : ℚ :=
  if n % 2 = 0
  then 1 / (n + 1 : ℚ)
  else 2 / n

/-- The sum of p_i * p_(i+1) from i=1 to 2018 -/
def sum_p : ℚ :=
  (Finset.range 2018).sum (λ i => p i * p (i + 1))

/-- Theorem stating the value of the sum -/
theorem sum_p_value : sum_p = 1009 / 1010 := by
  sorry

#eval sum_p

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_p_value_l389_38966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l389_38922

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  ∃ (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ),
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
    (F₁.1 < 0 ∧ F₂.1 > 0) ∧
    (P.1^2 + P.2^2 = F₁.1^2 + F₁.2^2) ∧
    ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = 3 * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)) →
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 3 + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l389_38922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_offset_length_proof_l389_38931

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (q.diagonal * (q.offset1 + q.offset2)) / 2

theorem offset_length_proof (q : Quadrilateral) 
    (h1 : q.diagonal = 30)
    (h2 : q.offset2 = 6)
    (h3 : area q = 225) :
  q.offset1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_offset_length_proof_l389_38931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_schedule_exists_l389_38996

/-- Represents a player -/
inductive Player : Type
| A | B | C | D | E | F

/-- Represents a pairing of two players -/
structure Pairing where
  player1 : Player
  player2 : Player

/-- Represents a round of pairings -/
structure Round where
  pairings : List Pairing

/-- Represents a schedule of rounds -/
structure Schedule where
  rounds : List Round

/-- Checks if two pairings are equal (order-independent) -/
def Pairing.equal (p1 p2 : Pairing) : Prop :=
  (p1.player1 = p2.player1 ∧ p1.player2 = p2.player2) ∨
  (p1.player1 = p2.player2 ∧ p1.player2 = p2.player1)

/-- Checks if a pairing appears in a list of pairings -/
def Pairing.appearsIn (p : Pairing) (ps : List Pairing) : Prop :=
  ∃ q, q ∈ ps ∧ Pairing.equal p q

/-- Checks if a schedule is valid -/
def Schedule.isValid (s : Schedule) : Prop :=
  -- Each round has exactly 3 pairings
  (∀ r, r ∈ s.rounds → r.pairings.length = 3) ∧
  -- No player is paired with themselves
  (∀ r, r ∈ s.rounds → ∀ p, p ∈ r.pairings → p.player1 ≠ p.player2) ∧
  -- Each player appears exactly once in each round
  (∀ r, r ∈ s.rounds → ∀ player : Player, 
    (player ∈ (r.pairings.map Pairing.player1)) ∨ 
    (player ∈ (r.pairings.map Pairing.player2))) ∧
  -- No pairing is repeated across rounds
  (∀ r1 r2, r1 ∈ s.rounds → r2 ∈ s.rounds → r1 ≠ r2 → 
    ∀ p, p ∈ r1.pairings → ¬(Pairing.appearsIn p r2.pairings)) ∧
  -- Each player pairs with every other player exactly once
  (∀ p1 p2 : Player, p1 ≠ p2 → 
    ∃! r, r ∈ s.rounds ∧ Pairing.appearsIn ⟨p1, p2⟩ r.pairings)

/-- The main theorem: a valid schedule exists for 6 players -/
theorem valid_schedule_exists : ∃ s : Schedule, Schedule.isValid s ∧ s.rounds.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_schedule_exists_l389_38996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_maximized_l389_38927

/-- The function to be integrated -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.cos x) * (380 - x - x^2)

/-- The integral to be maximized -/
noncomputable def integral (a b : ℝ) : ℝ := ∫ x in a..b, f x

theorem integral_maximized (a b : ℝ) (h : a ≤ b) :
  integral a b ≤ integral (-20) 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_maximized_l389_38927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l389_38977

noncomputable def f (x : Real) : Real := 
  Real.sin (4 * x) + Real.cos (4 * x) + Real.sin (5 * x) + Real.cos (5 * x)

noncomputable def deg_to_rad (x : Real) : Real := x * Real.pi / 180

theorem inequality_equivalence :
  ∀ x : Real, 0 ≤ x ∧ x < 360 →
    (f (deg_to_rad x) < 0 ↔ 
      (30 < x ∧ x < 70) ∨
      (110 < x ∧ x < 150) ∨
      (180 < x ∧ x < 190) ∨
      (230 < x ∧ x < 270) ∨
      (310 < x ∧ x < 350)) :=
by sorry

#check inequality_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l389_38977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_inequality_l389_38989

/-- A sequence of four real numbers forms an arithmetic progression --/
def is_arithmetic_progression (a x y b : ℝ) : Prop :=
  ∃ d : ℝ, y - x = x - a ∧ b - y = y - x ∧ x - a = d

/-- A sequence of four real numbers forms a geometric progression --/
def is_geometric_progression (a x y b : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ x / a = q ∧ y / x = q ∧ b / y = q

theorem arithmetic_geometric_inequality (a b A₁ A₂ G₁ G₂ : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (harith : is_arithmetic_progression a A₁ A₂ b)
  (hgeom : is_geometric_progression a G₁ G₂ b) :
  A₁ * A₂ ≥ G₁ * G₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_inequality_l389_38989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_min_value_sum_min_value_is_achieved_l389_38924

-- Define the function f
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

-- Part I: Solution set of f(x + 3/2) ≥ 0
theorem solution_set_f (x : ℝ) : 
  f (x + 3/2) ≥ 0 ↔ x ∈ Set.Icc (-2 : ℝ) 2 := by sorry

-- Part II: Minimum value of 3p + 2q + r
theorem min_value_sum (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 := by sorry

-- Proof that 9/4 is the minimum value
theorem min_value_is_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ (p q r : ℝ), p > 0 ∧ q > 0 ∧ r > 0 ∧ 
    1/(3*p) + 1/(2*q) + 1/r = 4 ∧ 
    3*p + 2*q + r < 9/4 + ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_min_value_sum_min_value_is_achieved_l389_38924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flyer_distribution_probability_l389_38982

def total_mailboxes : ℕ := 10
def flyers_per_distributor : ℕ := 5
def min_mailboxes_with_flyers : ℕ := 8

def probability_at_least_8_mailboxes_with_flyers : ℚ := 1 / 2

theorem flyer_distribution_probability :
  let total_outcomes := (Nat.choose total_mailboxes flyers_per_distributor) * 
                        (Nat.choose total_mailboxes flyers_per_distributor)
  let favorable_outcomes := (Nat.choose total_mailboxes flyers_per_distributor) *
    (1 + (Nat.choose flyers_per_distributor 4) * (Nat.choose flyers_per_distributor 1) +
         (Nat.choose flyers_per_distributor 3) * (Nat.choose flyers_per_distributor 2))
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = probability_at_least_8_mailboxes_with_flyers :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flyer_distribution_probability_l389_38982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_C_l389_38942

-- Define the square and point P
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (1, 1)
def D : ℝ × ℝ := (0, 1)

-- Define distances
noncomputable def u (P : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
noncomputable def v (P : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
noncomputable def w (P : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2)

-- Define the distance from P to C
noncomputable def distPC (P : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)

-- Theorem statement
theorem max_distance_to_C :
  ∃ P : ℝ × ℝ, (u P)^2 + (w P)^2 = 2*(v P)^2 ∧
  (∀ Q : ℝ × ℝ, (u Q)^2 + (w Q)^2 = 2*(v Q)^2 → distPC P ≥ distPC Q) ∧
  distPC P = Real.sqrt 0.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_C_l389_38942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_cubic_equation_l389_38971

/-- A complex number z is a root of unity if z^n = 1 for some positive integer n. -/
def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z ^ n = 1

/-- The number of roots of unity that are also roots of z^3 + cz + d = 0 for some integers c and d is 4. -/
theorem roots_of_unity_cubic_equation :
  ∃! (S : Finset ℂ), (∀ z ∈ S, is_root_of_unity z) ∧ 
    (∃ c d : ℤ, ∀ z ∈ S, z^3 + c*z + d = 0) ∧
    S.card = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_cubic_equation_l389_38971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_factorial_sum_l389_38981

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ :=
  (List.range ((n + 1) / 2)).map (fun i => factorial (2 * i + 1)) |>.sum

theorem units_digit_of_factorial_sum :
  (sum_of_factorials 49) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_factorial_sum_l389_38981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_maximum_l389_38934

/-- Given a perimeter 2p, the area of any triangle with that perimeter is at most p^2 / (3√3),
    with equality achieved for an equilateral triangle. -/
theorem triangle_area_maximum (p : ℝ) (h : p > 0) :
  ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2 * p →
  let s := p
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area ≤ p^2 / (3 * Real.sqrt 3) ∧
  (area = p^2 / (3 * Real.sqrt 3) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_maximum_l389_38934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l389_38962

theorem product_remainder (a b c : ℕ) 
  (ha : a % 5 = 1)
  (hb : b % 5 = 2)
  (hc : c % 5 = 3) :
  (a * b * c) % 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l389_38962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_5_A_bounds_l389_38938

-- Define the type for permutations
def Permutation (n : ℕ) := Fin n → Fin n

-- Define the descending chain property
def IsDescendingChain (n : ℕ) (S : Permutation n) (start : Fin n) (len : ℕ) : Prop := sorry

-- Define G(S) as the number of descending chains in permutation S
def G (n : ℕ) (S : Permutation n) : ℕ := sorry

-- Define A(n) as the average value of G(S) for all n-point permutations
noncomputable def A (n : ℕ) : ℚ := sorry

-- Statement for A(5)
theorem A_5 : A 5 = 73 / 24 := sorry

-- Statement for the bounds of A(n) when n ≥ 6
theorem A_bounds (n : ℕ) (h : n ≥ 6) : 
  83 * n / 120 - 1 / 2 ≤ A n ∧ A n ≤ 101 * n / 120 - 1 / 2 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_5_A_bounds_l389_38938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l389_38963

/-- 
Given a triangle ABC with internal angles A, B, C and opposite sides a, b, c,
prove that if vectors m = (1, 1 - sin A) and n = (cos A, 1) are perpendicular,
and b + c = a, then A = π/2 and sin(B + C) = 1.
-/
theorem triangle_property (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (1 : ℝ) * (Real.cos A) + (1 - Real.sin A) * (1 : ℝ) = 0 ∧ 
  b + c = a →
  A = Real.pi / 2 ∧ Real.sin (B + C) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l389_38963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l389_38998

/-- Given vectors a and b in a real inner product space satisfying certain conditions,
    prove that the magnitude of their difference is 2√2. -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) 
    (h1 : ‖a‖ = 2)
    (h2 : ‖b‖ = 3)
    (h3 : a • (b - a) = 1) :
    ‖a - b‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l389_38998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_gain_is_one_mph_l389_38944

/-- Calculates the speed gained per week during baseball training -/
noncomputable def speed_gain_per_week (initial_speed : ℝ) (sessions : ℕ) (weeks_per_session : ℕ) (percent_increase : ℝ) : ℝ :=
  let total_increase := initial_speed * percent_increase
  let total_weeks := sessions * weeks_per_session
  total_increase / total_weeks

/-- Theorem stating that the speed gain per week is 1 mph under given conditions -/
theorem speed_gain_is_one_mph :
  speed_gain_per_week 80 4 4 0.2 = 1 := by
  -- Unfold the definition of speed_gain_per_week
  unfold speed_gain_per_week
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_gain_is_one_mph_l389_38944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_amount_for_yogurt_l389_38933

/-- The cost of producing yogurt batches -/
structure YogurtProduction where
  milk_cost_per_liter : ℚ
  fruit_cost_per_kg : ℚ
  milk_per_batch : ℚ
  cost_three_batches : ℚ

/-- Calculate the amount of fruit needed for one batch of yogurt -/
def fruit_per_batch (yp : YogurtProduction) : ℚ :=
  ((yp.cost_three_batches - 3 * yp.milk_cost_per_liter * yp.milk_per_batch) / 3) / yp.fruit_cost_per_kg

/-- Theorem stating that 3 kg of fruit are needed for one batch of yogurt -/
theorem fruit_amount_for_yogurt (yp : YogurtProduction)
    (h1 : yp.milk_cost_per_liter = 3/2)
    (h2 : yp.fruit_cost_per_kg = 2)
    (h3 : yp.milk_per_batch = 10)
    (h4 : yp.cost_three_batches = 63) :
    fruit_per_batch yp = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_amount_for_yogurt_l389_38933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_and_jill_meeting_point_l389_38948

/-- Represents the route details and runners' speeds --/
structure RouteData where
  totalDistance : ℚ
  uphillDistance : ℚ
  jackUphillSpeed : ℚ
  jackDownhillSpeed : ℚ
  jillUphillSpeed : ℚ
  jillDownhillSpeed : ℚ
  jackHeadStart : ℚ

/-- Calculates the meeting point of Jack and Jill --/
def meetingPoint (data : RouteData) : ℚ :=
  data.uphillDistance - 68 / 21

/-- Theorem stating that Jack and Jill meet at the calculated point --/
theorem jack_and_jill_meeting_point (data : RouteData)
  (h1 : data.totalDistance = 12)
  (h2 : data.uphillDistance = 6)
  (h3 : data.jackUphillSpeed = 14)
  (h4 : data.jackDownhillSpeed = 20)
  (h5 : data.jillUphillSpeed = 16)
  (h6 : data.jillDownhillSpeed = 24)
  (h7 : data.jackHeadStart = 1/6) :
  meetingPoint data = data.uphillDistance - 68 / 21 :=
by
  sorry

/-- Example calculation --/
def example_data : RouteData := {
  totalDistance := 12,
  uphillDistance := 6,
  jackUphillSpeed := 14,
  jackDownhillSpeed := 20,
  jillUphillSpeed := 16,
  jillDownhillSpeed := 24,
  jackHeadStart := 1/6
}

#eval meetingPoint example_data

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_and_jill_meeting_point_l389_38948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_to_hundredth_l389_38919

/-- The repeating decimal 37.837837837... -/
def repeating_decimal : ℚ := 37 + 837 / 999

/-- Rounding a rational number to the nearest hundredth -/
def round_to_hundredth (q : ℚ) : ℚ := 
  (q * 100).floor / 100 + if (q * 100 - (q * 100).floor ≥ 1/2) then 1/100 else 0

theorem round_repeating_decimal_to_hundredth : 
  round_to_hundredth repeating_decimal = 3784 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_to_hundredth_l389_38919
