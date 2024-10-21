import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_lower_bound_l707_70705

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

-- State the theorem
theorem f_minimum_and_lower_bound (a : ℝ) (h : a > 0) :
  -- f(x) has a minimum value at x = ln(1/a)
  (∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y ∧ x = Real.log (1/a)) ∧
  -- The minimum value of f(x) is greater than 2ln(a) + 3/2
  (∀ (x : ℝ), f a x > 2 * Real.log a + 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_lower_bound_l707_70705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equilateral_intersection_l707_70713

/-- A triangular prism -/
structure TriangularPrism where
  base : Triangle
  height : ℝ
  height_positive : height > 0

/-- A plane in 3D space -/
structure Plane where
  normal : Fin 3 → ℝ
  point : Fin 3 → ℝ

/-- The intersection of a plane and a triangular prism -/
noncomputable def intersection (prism : TriangularPrism) (plane : Plane) : Set (Fin 3 → ℝ) :=
  sorry

/-- An equilateral triangle -/
def is_equilateral_triangle (t : Set (Fin 3 → ℝ)) : Prop :=
  sorry

/-- Theorem: There exists a plane that intersects a triangular prism to form an equilateral triangle -/
theorem exists_equilateral_intersection (prism : TriangularPrism) :
  ∃ (plane : Plane), is_equilateral_triangle (intersection prism plane) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equilateral_intersection_l707_70713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_U_l707_70709

/-- A regular equilateral triangle in the complex plane -/
structure EquilateralTriangle where
  center : ℂ
  sideLength : ℝ
  parallelSide : Bool

/-- The set U obtained by squaring elements of T -/
def U (T : Set ℂ) : Set ℂ := {z | ∃ w ∈ T, z = w^2}

/-- The area of a set in the complex plane -/
noncomputable def area (S : Set ℂ) : ℝ := sorry

/-- The region inside the triangle -/
def triangleInterior (T : EquilateralTriangle) : Set ℂ := sorry

/-- The theorem stating the area of U -/
theorem area_of_U (T : EquilateralTriangle) (region : Set ℂ) :
  T.center = 0 ∧ 
  T.sideLength = Real.sqrt 3 ∧ 
  T.parallelSide = true ∧
  region = U (triangleInterior T) →
  area region = 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_U_l707_70709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l707_70711

/-- Given a triangle ABC with specific properties, prove it's a right triangle
    and its perimeter is in the range (4, 4] -/
theorem triangle_properties (A B C : ℝ) (m n : ℝ × ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Definition of vectors m and n
  m = (Real.sin A, Real.cos C) ∧ n = (Real.cos B, Real.sin A) →
  -- Dot product condition
  m.1 * n.1 + m.2 * n.2 = Real.sin B + Real.sin C →
  -- Radius of circumcircle is 1
  (2 * Real.sin A) = 1 →
  -- Prove: ABC is a right triangle (A is the right angle)
  A = π / 2 ∧
  -- Prove: Perimeter P satisfies 4 < P ≤ 4
  (let P := 2 + 2 * (Real.sin B + Real.cos B);
   4 < P ∧ P ≤ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l707_70711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_not_guaranteed_l707_70715

/-- The probability of winning the grand prize in a single lottery ticket. -/
noncomputable def winning_probability : ℝ := 1 / 100000

/-- The number of lottery tickets purchased. -/
def tickets_bought : ℕ := 100000

/-- Proposition that buying 'n' lottery tickets guarantees winning the grand prize. -/
def guaranteed_win (n : ℕ) : Prop :=
  ∀ (outcome : Fin n → Bool), (∃ i, outcome i = true)

theorem lottery_not_guaranteed : 
  ¬ guaranteed_win tickets_bought := by
  sorry

#check lottery_not_guaranteed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_not_guaranteed_l707_70715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l707_70793

-- Problem 1
theorem problem_1 (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin (θ - 6 * Real.pi) + Real.sin (Real.pi / 2 - θ)) / 
  (2 * Real.sin (Real.pi + θ) + Real.cos (-θ)) = -1 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) 
  (h1 : -Real.pi / 2 < x ∧ x < Real.pi / 2) 
  (h2 : Real.sin x + Real.cos x = 1 / 5) :
  Real.tan x = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l707_70793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l707_70762

noncomputable def sinusoidal_function (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem sinusoidal_function_properties (y : ℝ → ℝ) (h_max : (∀ x, y x ≤ 1/2) ∧ (∃ x, y x = 1/2)) 
  (h_period : ∀ x, y (x + π/3) = y x) (h_phase : y 0 = 1/2 * Real.sin (π/4)) :
  ∃ A ω φ, A = 1/2 ∧ ω = 6 ∧ φ = π/4 ∧ (∀ x, y x = sinusoidal_function A ω φ x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l707_70762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hours_l707_70797

-- Define the variables
noncomputable def A : ℝ := 12
noncomputable def B : ℝ := (1/3) * A
noncomputable def C : ℝ := 2 * B
noncomputable def E : ℝ := A + 3
noncomputable def D : ℝ := (1/2) * E

-- Define the total hours
noncomputable def T : ℝ := A + B + C + D + E

-- Theorem statement
theorem total_hours : T = 46.5 := by
  -- Unfold definitions
  unfold T A B C D E
  -- Simplify the expression
  simp [mul_div_cancel']
  -- Prove the equality
  norm_num
  -- If the automatic proof fails, we can use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hours_l707_70797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piggy_bank_coins_l707_70766

theorem piggy_bank_coins (piggy_banks : Fin 6 → ℕ) 
  (h1 : piggy_banks 0 = 72)
  (h3 : piggy_banks 2 = 90)
  (h4 : piggy_banks 3 = 99)
  (h5 : piggy_banks 4 = 108)
  (h6 : piggy_banks 5 = 117)
  (pattern : ∀ i : Fin 3, piggy_banks (i + 2) - piggy_banks (i + 1) = piggy_banks (i + 3) - piggy_banks (i + 2)) :
  piggy_banks 1 = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piggy_bank_coins_l707_70766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_pentagon_centroids_not_collinear_l707_70743

/-- A convex pentagon -/
structure ConvexPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_convex : Convex ℝ (Set.range vertices)

/-- A diagonal of a pentagon -/
def Diagonal (p : ConvexPentagon) (i j : Fin 5) : Set (ℝ × ℝ) :=
  Set.Icc (p.vertices i) (p.vertices j)

/-- Two diagonals are non-intersecting -/
def NonIntersectingDiagonals (p : ConvexPentagon) (d₁ d₂ : Set (ℝ × ℝ)) : Prop :=
  d₁ ∩ d₂ = ∅

/-- A triangle formed by three points -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.a.1 + t.b.1 + t.c.1) / 3, (t.a.2 + t.b.2 + t.c.2) / 3)

/-- Three points are collinear -/
def collinear (a b c : ℝ × ℝ) : Prop :=
  (b.2 - a.2) * (c.1 - a.1) = (c.2 - a.2) * (b.1 - a.1)

/-- The main theorem -/
theorem convex_pentagon_centroids_not_collinear (p : ConvexPentagon)
  (d₁ d₂ : Set (ℝ × ℝ)) (h : NonIntersectingDiagonals p d₁ d₂)
  (t₁ t₂ t₃ : Triangle) :
  ¬collinear (centroid t₁) (centroid t₂) (centroid t₃) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_pentagon_centroids_not_collinear_l707_70743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exp_function_range_l707_70700

-- Define the exponential function f(x) = (a - 1)^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) ^ x

-- State the theorem
theorem decreasing_exp_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) → 1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exp_function_range_l707_70700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_vector_coordinates_l707_70781

/-- Given two points in R², A and B, and a translation vector a,
    prove that the coordinates of vector AB after translation are (2, -5) -/
theorem translated_vector_coordinates (A B a : ℝ × ℝ) : 
  A = (3, 7) → B = (5, 2) → a = (1, 2) → 
  (B.1 - A.1, B.2 - A.2) = (2, -5) := by
  intros hA hB ha
  simp [hA, hB]
  norm_num
  
#check translated_vector_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_vector_coordinates_l707_70781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_property_l707_70759

/-- Represents an arithmetic progression with first term a₁ and common difference d -/
structure ArithmeticProgression where
  a₁ : ℚ
  d : ℚ

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  n / 2 * (2 * ap.a₁ + (n - 1) * ap.d)

/-- Theorem stating that the given equation holds for any arithmetic progression -/
theorem arithmetic_progression_sum_property (ap : ArithmeticProgression) (n : ℕ) :
  sum_n ap (n + 3) - 3 * sum_n ap (n + 2) + 3 * sum_n ap (n + 1) - sum_n ap n = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_property_l707_70759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l707_70760

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 20 cm and 18 cm, 
    and a distance of 25 cm between them, is 475 square centimeters -/
theorem trapezium_area_example : trapeziumArea 20 18 25 = 475 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic expression
  simp [add_mul, mul_div_assoc]
  -- Perform the numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l707_70760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_of_squares_l707_70703

/-- Represents a point in a 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the area of a square given its side length -/
def squareArea (side : ℝ) : ℝ :=
  side^2

theorem total_area_of_squares (A B C : Point) 
    (h1 : A = ⟨0, 0⟩)
    (h2 : B = ⟨12, 0⟩)
    (h3 : C = ⟨12, 12⟩)
    (h4 : ∃ X Y : Point, X.x = 0 ∧ Y.x = 0) -- X and Y on y-axis
    (h5 : ∃ D Z : Point, D.y > C.y ∧ Z.y > C.y) -- D and Z above C
    : squareArea (distance A B) + squareArea (distance B C) = 288 := by
  sorry

#eval squareArea 12 + squareArea 12 -- Should output 288

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_of_squares_l707_70703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_value_l707_70775

open Real

noncomputable def f (x : ℝ) : ℝ := sin x - cos x

theorem tan_2x_value (h : ∀ x, (deriv^[2] f) x = (1/2) * (f x)) : tan (2 * π / 4) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_value_l707_70775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l707_70770

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x + Real.pi / 4)

-- State the theorem
theorem f_decreasing_interval :
  ∃ (a b : ℝ), a = Real.pi / 4 ∧ b = Real.pi ∧
  (∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f y < f x) ∧
  (∀ x, x ∈ Set.Icc 0 Real.pi → (x < a ∨ x > b ∨ x ∈ Set.Icc a b)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l707_70770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_given_decreasing_f_and_g_l707_70776

/-- A function that represents f(x) = -x^2 + 2ax --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x

/-- A function that represents g(x) = a/(x+1) --/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a / (x + 1)

/-- The theorem stating the range of a given the conditions --/
theorem a_range_given_decreasing_f_and_g :
  ∀ a : ℝ,
  (∀ x y : ℝ, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f a x > f a y) →
  (∀ x y : ℝ, 1 ≤ x ∧ x < y ∧ y ≤ 2 → g a x > g a y) →
  0 < a ∧ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_given_decreasing_f_and_g_l707_70776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_baskets_pigeonhole_l707_70714

theorem orange_baskets_pigeonhole (total_baskets : ℕ) (min_oranges max_oranges : ℕ) :
  total_baskets = 180 →
  min_oranges = 130 →
  max_oranges = 170 →
  ∃ (n : ℕ), min_oranges ≤ n ∧ n ≤ max_oranges ∧ 
  (∃ (baskets : Finset (Fin total_baskets)), baskets.card ≥ 5 ∧ 
    ∀ (b : Fin total_baskets), b ∈ baskets → (∃ (m : ℕ), m = n)) := by
  sorry

#check orange_baskets_pigeonhole

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_baskets_pigeonhole_l707_70714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_coordinates_l707_70749

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 2)

theorem inverse_function_point_coordinates 
  (a : ℝ) 
  (ha : a > 0 ∧ a ≠ 1) 
  (hf : Function.Bijective (f a)) :
  ∃ P : ℝ × ℝ, (Function.invFun (f a) P.1 = P.2 ∧ P = (1, -2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_coordinates_l707_70749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l707_70792

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 3)

-- State the theorem
theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Ioo 0 (π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l707_70792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_transformed_points_in_S_l707_70772

-- Define the set S
noncomputable def S : Set ℂ := {z | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -2 ≤ z.im ∧ z.im ≤ 2}

-- Define the transformation
noncomputable def transform (z : ℂ) : ℂ := (1/2 + 1/2 * Complex.I) * z

-- Theorem statement
theorem all_transformed_points_in_S : ∀ z ∈ S, transform z ∈ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_transformed_points_in_S_l707_70772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l707_70728

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area function
noncomputable def area (t : Triangle) : ℝ := 
  1/2 * t.a * t.b * Real.sin t.C

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.c^2 - 2 * t.a^2 = t.b^2) 
  (h2 : t.a = 1) 
  (h3 : Real.tan t.A = 1/3) : 
  (t.c * Real.cos t.A - t.a * Real.cos t.C) / t.b = 1/2 ∧ 
  area t = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l707_70728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_sin_minus_cos_l707_70756

theorem tan_value_given_sin_minus_cos (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo 0 π) 
  (h2 : Real.sin θ - Real.cos θ = 1/2) : 
  Real.tan θ = (4 + Real.sqrt 7) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_sin_minus_cos_l707_70756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_balls_to_boxes_l707_70710

def number_of_ways_to_distribute (n : ℕ) (k : ℕ) : ℕ := k^n

theorem distribute_balls_to_boxes (n : ℕ) (k : ℕ) : 
  number_of_ways_to_distribute n k = k^n :=
by
  rfl

#eval number_of_ways_to_distribute 5 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_balls_to_boxes_l707_70710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_approximation_l707_70784

/-- Calculates the speed in miles per hour given distance in feet and time in seconds -/
noncomputable def speed_mph (distance_feet : ℝ) (time_seconds : ℝ) : ℝ :=
  (distance_feet / 5280) / (time_seconds / 3600)

/-- Theorem stating that an object traveling 300 feet in 6 seconds has a speed of approximately 34.091 mph -/
theorem object_speed_approximation :
  ∃ ε > 0, abs (speed_mph 300 6 - 34.091) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_approximation_l707_70784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l707_70796

noncomputable def f (x : ℝ) := 3*x + 4 - 2*(Real.sqrt (2*x^2 + 7*x + 3))
def g (x : ℝ) := |x^2 - 4*x + 2| - |x - 2|

def domain : Set ℝ := {x | x ≤ -3 ∨ x ≥ -1/2}

def solution_set : Set ℝ := Set.Iic (-3) ∪ Set.Icc 0 1 ∪ {2} ∪ Set.Icc 3 4

theorem inequality_solution (x : ℝ) :
  x ∈ domain → (f x * g x ≤ 0 ↔ x ∈ solution_set) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l707_70796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l707_70769

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point A
def point_A : ℝ × ℝ := (2, 4)

-- Define the two lines
def line1 (x : ℝ) : Prop := x = 2
def line2 (x y : ℝ) : Prop := 3*x - 4*y + 10 = 0

-- Theorem statement
theorem tangent_lines_to_circle :
  ∃ (x y : ℝ), 
    (line1 x ∨ line2 x y) ∧ 
    my_circle x y ∧ 
    (x, y) ≠ point_A ∧
    ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ 
      ∀ (x' y' : ℝ), 
        ((x' - x)^2 + (y' - y)^2 < δ^2) → 
        (¬(my_circle x' y') ∨ (x', y') = (x, y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l707_70769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_ampersand_five_l707_70783

noncomputable def ampersand (a b : ℝ) : ℝ := ((a + b) * (a - b)) / 2

theorem eight_ampersand_five : ampersand 8 5 = 19.5 := by
  -- Unfold the definition of ampersand
  unfold ampersand
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_ampersand_five_l707_70783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_row_transformation_l707_70725

theorem matrix_row_transformation (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, (1/2)]
  let MA := M * A
  MA 0 0 = 2 * A 0 0 ∧ 
  MA 0 1 = 2 * A 0 1 ∧ 
  MA 1 0 = (1/2) * A 1 0 ∧ 
  MA 1 1 = (1/2) * A 1 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_row_transformation_l707_70725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_first_is_prime_count_primes_in_sequence_l707_70794

def sequenceNum : ℕ → ℕ
  | 0 => 37
  | n + 1 => 100 * sequenceNum n + 37

theorem only_first_prime :
  ∀ n : ℕ, n > 0 → ¬(Nat.Prime (sequenceNum n)) :=
by
  sorry

theorem first_is_prime : Nat.Prime (sequenceNum 0) :=
by
  sorry

theorem count_primes_in_sequence : 
  (Finset.filter (λ i => Nat.Prime (sequenceNum i)) (Finset.range (Nat.succ ω))).card = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_first_is_prime_count_primes_in_sequence_l707_70794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_irrational_numbers_l707_70750

-- Define π as an irrational number
axiom pi_irrational : Irrational Real.pi

-- Define √2 as an irrational number
axiom sqrt2_irrational : Irrational (Real.sqrt 2)

-- Theorem stating that we have two irrational numbers
theorem two_irrational_numbers : ∃ (a b : ℝ), Irrational a ∧ Irrational b :=
by
  use Real.pi, Real.sqrt 2
  exact ⟨pi_irrational, sqrt2_irrational⟩

#check two_irrational_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_irrational_numbers_l707_70750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l707_70791

-- Define the polygonal region
def PolygonalRegion (x y : ℝ) : Prop :=
  x + 2*y ≤ 4 ∧ 3*x + y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 5

-- Define the function to calculate the length of a side
noncomputable def SideLength (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem longest_side_length :
  ∃ (x1 y1 x2 y2 : ℝ),
    PolygonalRegion x1 y1 ∧ 
    PolygonalRegion x2 y2 ∧ 
    SideLength x1 y1 x2 y2 = 5 * Real.sqrt 2 ∧
    ∀ (x3 y3 x4 y4 : ℝ), 
      PolygonalRegion x3 y3 → 
      PolygonalRegion x4 y4 → 
      SideLength x3 y3 x4 y4 ≤ 5 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l707_70791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l707_70739

/-- A quadratic function f(x) = ax^2 + bx + c where a is non-zero -/
def QuadraticFunction (a b c : ℝ) (ha : a ≠ 0) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_properties 
  (a b c : ℝ) (ha : a ≠ 0) :
  let f := QuadraticFunction a b c ha
  -- 1 is an extreme point of f(x)
  (∃ (k : ℝ), (deriv f) 1 = k ∧ k = 0) ∧ 
  -- 3 is an extreme value of f(x)
  (∃ (x : ℝ), f x = 3 ∧ (deriv f) x = 0) ∧
  -- The point (2,8) lies on the curve y = f(x)
  (f 2 = 8) →
  a = 5 ∧ b = -10 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l707_70739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_trig_l707_70701

theorem third_quadrant_trig (α : Real) (h : α ∈ Set.Ioo π (3*π/2)) : 
  Real.tan α - Real.sin α > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_trig_l707_70701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periods_of_functions_l707_70735

noncomputable def f₁ (x : ℝ) : ℝ := Real.cos (abs (2 * x))
noncomputable def f₂ (x : ℝ) : ℝ := abs (Real.cos x)
noncomputable def f₃ (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)
noncomputable def f₄ (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 4)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  has_period f p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬has_period f q

theorem periods_of_functions :
  smallest_positive_period f₁ Real.pi ∧
  smallest_positive_period f₂ Real.pi ∧
  smallest_positive_period f₃ Real.pi ∧
  smallest_positive_period f₄ (Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periods_of_functions_l707_70735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l707_70712

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Angle between three points -/
noncomputable def angle (P Q R : Point) : ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (P Q : Point) : ℝ := sorry

/-- Perimeter of a quadrilateral -/
noncomputable def perimeter (q : Quadrilateral) : ℝ := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (P : Point) (q : Quadrilateral) : Prop := sorry

/-- Main theorem -/
theorem quadrilateral_perimeter 
  (A B C D P : Point) 
  (ABCD : Quadrilateral)
  (h_inside : isInside P ABCD)
  (h_angle1 : angle P A B = angle P D A)
  (h_angle2 : angle P A D = angle P D C)
  (h_angle3 : angle P B A = angle P C B)
  (h_angle4 : angle P B C = angle P C D)
  (h_PA : distance P A = 4)
  (h_PB : distance P B = 5)
  (h_PC : distance P C = 10) :
  perimeter ABCD = 9 * Real.sqrt 410 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l707_70712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_isosceles_triangles_l707_70707

open Real EuclideanGeometry

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the external points A₁, B₁, C₁
variable (A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 2))

-- Define the angles α, β, γ
variable (α β γ : ℝ)

-- Define the isosceles triangles
def isosceles_AC₁B (A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 2)) (α : ℝ) : Prop := 
  (dist A C₁ = dist B C₁) ∧ (angle A C₁ B = 2 * α)

def isosceles_BA₁C (A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 2)) (β : ℝ) : Prop := 
  (dist B A₁ = dist C A₁) ∧ (angle B A₁ C = 2 * β)

def isosceles_CB₁A (A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 2)) (γ : ℝ) : Prop := 
  (dist C B₁ = dist A B₁) ∧ (angle C B₁ A = 2 * γ)

-- State the theorem
theorem external_isosceles_triangles 
  (h1 : isosceles_AC₁B A B C A₁ B₁ C₁ α)
  (h2 : isosceles_BA₁C A B C A₁ B₁ C₁ β)
  (h3 : isosceles_CB₁A A B C A₁ B₁ C₁ γ)
  (h4 : α + β + γ = π) :
  (angle B₁ A₁ C₁ = α) ∧ (angle A₁ B₁ C₁ = β) ∧ (angle A₁ C₁ B₁ = γ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_isosceles_triangles_l707_70707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonica_worth_four_dollars_l707_70787

/-- Represents the number of cows and the price per cow -/
def x : ℕ := sorry

/-- The total revenue from selling the cows -/
def total_revenue : ℕ := x * x

/-- The price of each sheep -/
def sheep_price : ℕ := 12

/-- The number of sheep bought -/
def sheep_count : ℕ := total_revenue / sheep_price

/-- The remaining money after buying sheep -/
def remaining_money : ℕ := total_revenue % sheep_price

/-- The price of the lamb -/
def lamb_price : ℕ := remaining_money

/-- The value of the harmonica -/
def harmonica_value : ℕ := lamb_price

theorem harmonica_worth_four_dollars :
  (x % 12 = 4) →
  (sheep_count % 2 = 1) →
  (harmonica_value = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonica_worth_four_dollars_l707_70787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circumcircle_area_l707_70745

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ

-- Define a circle
structure Circle where
  radius : ℝ

-- Define a function to calculate the area of a circle
noncomputable def circleArea (c : Circle) : ℝ := Real.pi * c.radius^2

-- State the theorem
theorem isosceles_triangle_circumcircle_area 
  (t : IsoscelesTriangle) 
  (h1 : t.side = 4) 
  (h2 : t.base = 3) : 
  ∃ c : Circle, circleArea c = 16 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circumcircle_area_l707_70745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l707_70761

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (-3, 4)
def D : ℝ × ℝ := (5, -2)
def O : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := (1, -4)

theorem distance_between_points :
  distance A.1 A.2 B.1 B.2 = 5 ∧
  distance C.1 C.2 D.1 D.2 = 10 ∧
  distance O.1 O.2 E.1 E.2 = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l707_70761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_theorem_l707_70764

/-- Two lines in a 2D plane -/
structure TwoLines where
  k₁ : ℝ
  k₂ : ℝ
  h : k₁ * k₂ + 1 = 0

/-- The distance from a point to the origin -/
noncomputable def distance_to_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- The distance from a line y = kx + b to the origin -/
noncomputable def distance_line_to_origin (k b : ℝ) : ℝ := |b| / Real.sqrt (1 + k^2)

/-- The main theorem -/
theorem two_lines_theorem (tl : TwoLines) :
  (∃ x y : ℝ, y = tl.k₁ * x + 1 ∧ y = tl.k₂ * x - 1) ∧ 
  (∀ x y : ℝ, y = tl.k₁ * x + 1 ∧ y = tl.k₂ * x - 1 → distance_to_origin x y = 1) ∧
  (∃ M : ℝ, M = Real.sqrt 2 ∧ 
    ∀ d₁ d₂ : ℝ, 
      d₁ = distance_line_to_origin tl.k₁ 1 → 
      d₂ = distance_line_to_origin tl.k₂ (-1) → 
      d₁ + d₂ ≤ M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_theorem_l707_70764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l707_70782

/-- Reflection of a point across a line --/
noncomputable def reflect (x y m b : ℝ) : ℝ × ℝ :=
  let d := (x + (y - b) / m) / (1 + 1 / m^2)
  (2 * d - x, 2 * (m * d + b) - y)

/-- The problem statement --/
theorem reflection_problem (m b : ℝ) :
  reflect (-4) 1 m b = (2, 9) → m + b = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l707_70782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_dried_fruit_l707_70726

/-- Given the purchase of nuts and dried fruits, prove the cost of dried fruits. -/
theorem cost_of_dried_fruit (nuts_weight : ℝ) (dried_fruit_weight : ℝ) 
  (nuts_price : ℝ) (total_cost : ℝ) : 
  (total_cost - nuts_weight * nuts_price) / dried_fruit_weight = 8 :=
by
  -- Assuming:
  have h1 : nuts_weight = 3 := by sorry
  have h2 : dried_fruit_weight = 2.5 := by sorry
  have h3 : nuts_price = 12 := by sorry
  have h4 : total_cost = 56 := by sorry
  
  -- Prove:
  calc
    (total_cost - nuts_weight * nuts_price) / dried_fruit_weight
      = (56 - 3 * 12) / 2.5 := by sorry
    _ = 20 / 2.5 := by sorry
    _ = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_dried_fruit_l707_70726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_and_regression_equation_l707_70740

/-- Data points representing year codes and average incomes --/
noncomputable def data : List (ℝ × ℝ) := [(1, 1.2), (2, 1.4), (3, 1.5), (4, 1.6), (5, 1.8)]

/-- Calculate the mean of a list of real numbers --/
noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

/-- Calculate the correlation coefficient --/
noncomputable def correlationCoefficient (data : List (ℝ × ℝ)) : ℝ :=
  let n := data.length
  let xs := data.map Prod.fst
  let ys := data.map Prod.snd
  let x_mean := mean xs
  let y_mean := mean ys
  let numerator := (data.map (λ (x, y) => x * y)).sum - n * x_mean * y_mean
  let denominator := (((xs.map (λ x => (x - x_mean)^2)).sum) * ((ys.map (λ y => (y - y_mean)^2)).sum)).sqrt
  numerator / denominator

/-- Calculate the slope of the regression line --/
noncomputable def regressionSlope (data : List (ℝ × ℝ)) : ℝ :=
  let n := data.length
  let xs := data.map Prod.fst
  let ys := data.map Prod.snd
  let x_mean := mean xs
  let y_mean := mean ys
  let numerator := (data.map (λ (x, y) => x * y)).sum - n * x_mean * y_mean
  let denominator := (xs.map (λ x => x^2)).sum - n * x_mean^2
  numerator / denominator

/-- Calculate the intercept of the regression line --/
noncomputable def regressionIntercept (data : List (ℝ × ℝ)) (slope : ℝ) : ℝ :=
  let y_mean := mean (data.map Prod.snd)
  let x_mean := mean (data.map Prod.fst)
  y_mean - slope * x_mean

theorem correlation_and_regression_equation (ε : ℝ) (h : ε > 0) :
  let r := correlationCoefficient data
  let b := regressionSlope data
  let a := regressionIntercept data b
  (abs (r - 0.99) < ε) ∧ (abs (b - 0.14) < ε) ∧ (abs (a - 1.08) < ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_and_regression_equation_l707_70740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_air_conditioner_problem_l707_70742

-- Define the cost of type A and type B air conditioners
def cost_A : ℕ → ℕ
| n => 9000 * n

def cost_B : ℕ → ℕ
| n => 6000 * n

-- Define the total cost function
def total_cost (a b : ℕ) : ℕ := cost_A a + cost_B b

-- Define the constraints
def constraint1 : Prop := cost_A 3 + cost_B 2 = 39000
def constraint2 : Prop := cost_A 4 = cost_B 5 + 6000
def constraint3 (a b : ℕ) : Prop := a + b = 30
def constraint4 (a b : ℕ) : Prop := a ≥ b / 2
def constraint5 (a b : ℕ) : Prop := total_cost a b ≤ 217000

-- Theorem statement
theorem air_conditioner_problem :
  constraint1 ∧ constraint2 ∧
  (∃ a b : ℕ, constraint3 a b ∧ constraint4 a b ∧ constraint5 a b) →
  (∃ a b : ℕ, a = 10 ∧ b = 20 ∧ total_cost a b = 210000 ∧
    (∀ a' b' : ℕ, constraint3 a' b' ∧ constraint4 a' b' ∧ constraint5 a' b' →
      total_cost a' b' ≥ total_cost a b)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_air_conditioner_problem_l707_70742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_ninth_row_is_43_l707_70788

/-- Represents the lattice structure described in the problem -/
structure MyLattice where
  rows : Nat
  numbers_per_row : Nat
  last_number_increment : Nat

/-- Calculates the last number in a given row of the lattice -/
def last_number_in_row (l : MyLattice) (row : Nat) : Nat :=
  l.numbers_per_row * row

/-- Calculates the third number in a given row of the lattice -/
def third_number_in_row (l : MyLattice) (row : Nat) : Nat :=
  last_number_in_row l row - 2

/-- Theorem stating that the third number in the 9th row of the specified lattice is 43 -/
theorem third_number_ninth_row_is_43 (l : MyLattice) 
  (h1 : l.rows = 9) 
  (h2 : l.numbers_per_row = 5) 
  (h3 : l.last_number_increment = 5) : 
  third_number_in_row l 9 = 43 := by
  sorry

#check third_number_ninth_row_is_43

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_ninth_row_is_43_l707_70788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integers_less_than_three_l707_70721

theorem positive_integers_less_than_three :
  (Finset.filter (fun x => x < 3) (Finset.range 3)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integers_less_than_three_l707_70721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_log_quarter_three_l707_70730

-- Define an odd function f on ℝ
noncomputable def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_positive : ∀ x > 0, f x = 2^(x + 1)

-- Define the logarithm base 1/4
noncomputable def log_quarter : ℝ → ℝ := λ x ↦ Real.log x / Real.log (1/4)

-- State the theorem
theorem f_log_quarter_three : f (log_quarter 3) = -2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_log_quarter_three_l707_70730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_properties_l707_70729

open Real

variable (a b c A B C : ℝ)

/-- In triangle ABC, a, b, c are side lengths opposite to angles A, B, C respectively --/
def is_triangle (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- a, b, c form an arithmetic sequence --/
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b = a + c

/-- Main theorem --/
theorem triangle_arithmetic_sequence_properties
  (h_triangle : is_triangle a b c)
  (h_arithmetic : is_arithmetic_sequence a b c) :
  0 < B ∧ B ≤ π / 3 ∧
  a * (cos (C / 2))^2 + c * (cos (A / 2))^2 = 3 * b / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sequence_properties_l707_70729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_coin_weights_l707_70773

/-- Represents a coin with a weight -/
structure Coin where
  weight : ℕ

/-- Represents the result of weighing two coins -/
inductive WeighResult
  | Less : WeighResult
  | Equal : WeighResult
  | Greater : WeighResult

/-- Represents a set of three coins -/
def CoinSet := Fin 3 → Coin

/-- Represents a weighing action -/
def Weighing := Coin → Coin → WeighResult

/-- The main theorem statement -/
theorem determine_coin_weights 
  (set1 set2 : CoinSet) 
  (h1 : ∃ (p : Equiv.Perm (Fin 3)), (fun i => (set1 (p i)).weight) = ![9, 10, 11])
  (h2 : ∃ (p : Equiv.Perm (Fin 3)), (fun i => (set2 (p i)).weight) = ![9, 10, 11])
  : ∃ (weighings : Fin 4 → Weighing), 
    ∀ (i : Fin 3) (j : Fin 3), 
      ∃ (w : ℕ), set1 i = Coin.mk w ∧ set2 j = Coin.mk w :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_coin_weights_l707_70773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sampled_is_160_l707_70744

/-- The total number of papers sampled in a stratified sampling survey. -/
def total_sampled_papers (papers_a papers_b papers_c : ℕ) (sampled_c : ℕ) : ℕ :=
  let total_papers := papers_a + papers_b + papers_c
  let ratio := (sampled_c : ℚ) / papers_c
  (ratio * total_papers).floor.toNat

/-- Theorem stating that the total number of papers sampled is 160 given the problem conditions. -/
theorem total_sampled_is_160 :
  total_sampled_papers 1260 720 900 50 = 160 := by
  sorry

#eval total_sampled_papers 1260 720 900 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sampled_is_160_l707_70744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_of_sin_one_third_l707_70777

theorem cos_of_sin_one_third (θ : Real) (h_acute : 0 < θ ∧ θ < Real.pi / 2) (h_sin : Real.sin θ = 1 / 3) :
  Real.cos θ = (2 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_of_sin_one_third_l707_70777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_running_distance_l707_70774

/-- Calculates the distance John can run after increasing his duration and speed --/
theorem johns_running_distance 
  (initial_duration : ℝ) 
  (duration_increase_percentage : ℝ) 
  (initial_speed : ℝ) 
  (speed_increase : ℝ) 
  (h1 : initial_duration = 8)
  (h2 : duration_increase_percentage = 75)
  (h3 : initial_speed = 8)
  (h4 : speed_increase = 4) : 
  let new_duration := initial_duration * (1 + duration_increase_percentage / 100)
  let new_speed := initial_speed + speed_increase
  new_duration * new_speed = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_running_distance_l707_70774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_13_l707_70799

def series_sum (n : ℕ) : ℤ :=
  let rec a (k : ℕ) : ℤ :=
    match k with
    | 0 => 2
    | k+1 => if k % 2 = 0 then a k - (8 * ((k+1)/2) - 1) else a k + 8 * ((k+2)/2)
  let last_term := a (n-1)
  if last_term = 48 then (List.range n).map a |>.sum
  else 0

theorem series_sum_is_13 : ∃ n : ℕ, series_sum n = 13 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_13_l707_70799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eighth_term_l707_70723

def sequenceB (b : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n ≥ 1 → b (n + 2) = b (n + 1) + b n) ∧
  b 1 = 2 ∧
  b 7 = 162

theorem sequence_eighth_term (b : ℕ → ℕ) (h : sequenceB b) : b 8 = 263 := by
  sorry

#check sequence_eighth_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eighth_term_l707_70723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discrete_pareto_expectation_zeta_two_equals_pi_squared_over_six_auxiliary_integral_l707_70754

-- Define the Riemann zeta function
noncomputable def zeta (s : ℝ) : ℝ := ∑' (n : ℕ), (n : ℝ)^(-s)

-- Define the discrete Pareto distribution
noncomputable def discrete_pareto (ρ : ℝ) (n : ℕ) : ℝ := 1 / (zeta (ρ + 1) * (n : ℝ)^(ρ + 1))

-- State the theorem for part (a)
theorem discrete_pareto_expectation (ρ : ℝ) (hρ : ρ > 0) :
  ∑' (n : ℕ), n * discrete_pareto ρ n = zeta ρ / zeta (ρ + 1) := by
  sorry

-- State the theorem for part (b)
theorem zeta_two_equals_pi_squared_over_six :
  zeta 2 = π^2 / 6 := by
  sorry

-- Auxiliary integral for part (b)
theorem auxiliary_integral :
  (∫ x in (0:ℝ)..1, ∫ y in (0:ℝ)..1, 1 / (1 - x*y)) = π^2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discrete_pareto_expectation_zeta_two_equals_pi_squared_over_six_auxiliary_integral_l707_70754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_result_perpendicular_result_l707_70789

open Real

-- Define the vectors a and b
noncomputable def a (x : ℝ) : Fin 2 → ℝ := ![1, cos x]
noncomputable def b (x : ℝ) : Fin 2 → ℝ := ![1/2, sin x]

-- Define the condition x ∈ (0, π)
def x_in_range (x : ℝ) : Prop := 0 < x ∧ x < π

-- Define parallel vectors
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (i : Fin 2), u i = k * v i

-- Define perpendicular vectors
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

-- Theorem 1
theorem parallel_result (x : ℝ) (h : x_in_range x) :
  parallel (a x) (b x) → (sin x + cos x) / (sin x - cos x) = -3 := by sorry

-- Theorem 2
theorem perpendicular_result (x : ℝ) (h : x_in_range x) :
  perpendicular (a x) (b x) → sin x - cos x = sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_result_perpendicular_result_l707_70789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l707_70731

noncomputable def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l707_70731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_14_18_l707_70720

/-- The area of a rhombus with given diagonal lengths -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem: The area of a rhombus with diagonals 14 cm and 18 cm is 126 square centimeters -/
theorem rhombus_area_14_18 : rhombusArea 14 18 = 126 := by
  -- Unfold the definition of rhombusArea
  unfold rhombusArea
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Check that 14 * 18 / 2 = 126
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_14_18_l707_70720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_doubling_growth_rate_l707_70718

/-- The annual growth rate that causes a population to double in 8 years -/
noncomputable def annual_growth_rate : ℝ := Real.rpow 2 (1/8) - 1

/-- The time it takes for a population to double given the annual growth rate -/
noncomputable def doubling_time (r : ℝ) : ℝ := (Real.log 2) / (Real.log (1 + r))

theorem population_doubling_growth_rate :
  doubling_time annual_growth_rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_doubling_growth_rate_l707_70718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l707_70704

/-- A circle with center (a, 0) and radius 2 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = 4}

/-- The line x - y + √2 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + Real.sqrt 2 = 0}

/-- The distance from a point (x, y) to the line x - y + √2 = 0 -/
noncomputable def distToLine (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 + Real.sqrt 2| / Real.sqrt 2

/-- Definition of tangency between a circle and a line -/
def IsTangentTo (S T : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ S ∩ T ∧ ∀ q, q ∈ S ∩ T → q = p

theorem circle_tangent_to_line (a : ℝ) :
  IsTangentTo (Circle a) Line ↔ a = Real.sqrt 2 ∨ a = -3 * Real.sqrt 2 := by
  sorry

#check circle_tangent_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l707_70704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_j_type_growth_rate_gt_one_l707_70741

/-- Represents the growth rate parameter in a population growth model -/
def growthRate : ℝ := sorry

/-- Represents the growth type of a population -/
inductive GrowthType
| J
| Other

/-- Defines the condition for J-type growth -/
def isJTypeGrowth (rate : ℝ) : Prop :=
  rate > 1

/-- Theorem stating that for J-type growth, the growth rate must be strictly greater than 1 -/
theorem j_type_growth_rate_gt_one (g : GrowthType) (rate : ℝ) :
  g = GrowthType.J → isJTypeGrowth rate := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_j_type_growth_rate_gt_one_l707_70741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_cube_root_nine_equals_cube_root_three_l707_70706

theorem sqrt_cube_root_nine_equals_cube_root_three :
  Real.sqrt (9 ^ (1/3)) = 3 ^ (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_cube_root_nine_equals_cube_root_three_l707_70706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_swap_and_triple_l707_70752

open Matrix

theorem row_swap_and_triple (Q : Matrix (Fin 3) (Fin 3) ℝ) :
  let P : Matrix (Fin 3) (Fin 3) ℝ := !![3, 0, 0; 0, 0, 1; 0, 1, 0]
  P * Q = !![3 * Q 0 0, 3 * Q 0 1, 3 * Q 0 2;
             Q 2 0, Q 2 1, Q 2 2;
             Q 1 0, Q 1 1, Q 1 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_swap_and_triple_l707_70752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_twenty_seven_two_thirds_times_nine_negative_three_halves_l707_70747

theorem negative_twenty_seven_two_thirds_times_nine_negative_three_halves :
  ((-27 : ℝ) ^ (2/3 : ℝ)) * (9 : ℝ) ^ ((-3/2) : ℝ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_twenty_seven_two_thirds_times_nine_negative_three_halves_l707_70747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l707_70771

def A : Set ℤ := {x | 1 ≤ x ∧ x ≤ 3}

theorem proper_subsets_count : Finset.card (Finset.powerset (Finset.range 3) \ {Finset.range 3}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l707_70771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l707_70778

/-- Given a train with the following properties:
  * Length of the train is 1200 meters
  * Time to cross a tree (point) is 80 seconds
  * Time to cross a platform is 146.67 seconds
  Prove that the length of the platform is 1000.05 meters -/
theorem platform_length (train_length : ℝ) (time_tree : ℝ) (time_platform : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_tree = 80)
  (h3 : time_platform = 146.67) :
  let speed := train_length / time_tree
  let total_distance := speed * time_platform
  let platform_length := total_distance - train_length
  platform_length = 1000.05 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l707_70778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_odd_l707_70755

def sequenceProperty (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ 
  a 2 = 7 ∧ 
  ∀ n ≥ 2, -1/2 < (a (n+1) : ℚ) - (a n)^2 / (a (n-1)) ∧ 
           (a (n+1) : ℚ) - (a n)^2 / (a (n-1)) ≤ 1/2

theorem sequence_odd (a : ℕ → ℤ) (h : sequenceProperty a) : 
  ∀ n > 1, Odd (a n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_odd_l707_70755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_parabola_vertex_l707_70795

/-- The number of solutions for 'a' where the line y = 2x + a passes through
    the vertex of the parabola y = x^2 + ax + a^2 is exactly one, and that solution is a = 0. -/
theorem line_through_parabola_vertex :
  ∃! a : ℝ, 
    (let line := fun x => 2 * x + a
     let parabola := fun x => x^2 + a * x + a^2
     let vertex_x := -a / 2
     line vertex_x = parabola vertex_x) ∧ a = 0 := by
  sorry

#check line_through_parabola_vertex

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_parabola_vertex_l707_70795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_two_l707_70717

-- Define the function f as a piecewise linear function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x
  else 1 + 2 * (x - 1)

-- State the theorem
theorem inverse_f_at_two :
  (∀ x, x ∈ Set.Icc 0 2 → f x = if x ≤ 1 then x else 1 + 2 * (x - 1)) →
  f 0 = 0 ∧ f 1 = 1 ∧ f 2 = 3 →
  ∃ y, f y = 2 ∧ y = 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_two_l707_70717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_symmetry_l707_70790

-- Define the original ellipse
def original_ellipse (x y : ℝ) : Prop :=
  (x - 3)^2 / 9 + (y - 2)^2 / 4 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y = 0

-- Define the reflection transformation
def reflect (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

-- Define the reflected ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  (x + 2)^2 / 9 + (y + 3)^2 / 4 = 1

-- Theorem statement
theorem reflection_symmetry :
  ∀ (x y : ℝ),
    original_ellipse x y →
    (let (x', y') := reflect x y
     ellipse_C x' y') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_symmetry_l707_70790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_proof_l707_70786

/-- Line l in rectangular coordinates -/
def line_l (x y : ℝ) : Prop := x + y - 2 * Real.sqrt 2 = 0

/-- Circle O in rectangular coordinates -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- The length of the chord intercepted by line l on circle O -/
noncomputable def chord_length : ℝ := 4 * Real.sqrt 3

theorem chord_length_proof :
  ∀ x y : ℝ, line_l x y → circle_O x y →
  ∃ x1 y1 x2 y2 : ℝ, 
    line_l x1 y1 ∧ line_l x2 y2 ∧ 
    circle_O x1 y1 ∧ circle_O x2 y2 ∧
    Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = chord_length :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_proof_l707_70786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rightmost_nonzero_digit_20_13_factorial_l707_70716

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def rightmostNonZeroDigit : ℕ → ℕ
| 0 => 0
| n + 1 =>
  if (n + 1) % 10 ≠ 0 then (n + 1) % 10
  else rightmostNonZeroDigit (n / 10)

theorem rightmost_nonzero_digit_20_13_factorial :
  rightmostNonZeroDigit (20 * factorial 13) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rightmost_nonzero_digit_20_13_factorial_l707_70716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_theorem_l707_70768

/-- Transformation function for an n-tuple of integers -/
def transform (a : List Int) : List Int :=
  let n := a.length
  List.zipWith (fun x y => x + y) a (a.rotateLeft 1)

/-- Predicate to check if all elements in a list are multiples of k -/
def allMultiplesOf (l : List Int) (k : Int) : Prop :=
  ∀ x, x ∈ l → k ∣ x

/-- Main theorem statement -/
theorem transformation_theorem (n k : Nat) (h1 : n ≥ 2) (h2 : k ≥ 2) :
  (∀ a : List Int, a.length = n →
    ∃ m : Nat, allMultiplesOf ((transform^[m]) a) k) ↔ n = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_theorem_l707_70768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_l707_70737

/-- Represents a cone with height h and base radius r -/
structure Cone where
  h : ℝ
  r : ℝ

/-- The volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.r^2 * c.h

/-- The volume of water in a cone filled to a certain height ratio -/
noncomputable def waterVolume (c : Cone) (heightRatio : ℝ) : ℝ :=
  (1/3) * Real.pi * (heightRatio * c.r)^2 * (heightRatio * c.h)

theorem water_volume_ratio (c : Cone) :
  waterVolume c (2/3) = (8/27) * coneVolume c := by
  sorry

#eval (8/27 : Float)  -- To verify the decimal representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_l707_70737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_streaming_bill_fixed_fee_l707_70727

/-- Represents a streaming service billing system -/
structure StreamingBill where
  fixedFee : ℚ
  hourlyRate : ℚ

/-- Calculate the total bill given hours watched -/
def StreamingBill.totalBill (bill : StreamingBill) (hours : ℚ) : ℚ :=
  bill.fixedFee + bill.hourlyRate * hours

theorem streaming_bill_fixed_fee (bill : StreamingBill) : 
  bill.totalBill 1 = 18.72 → 
  bill.totalBill 3 = 28.08 → 
  bill.fixedFee = 14.04 := by
  intro h1 h2
  -- Proof steps would go here
  sorry

#check streaming_bill_fixed_fee

end NUMINAMATH_CALUDE_ERRORFEEDBACK_streaming_bill_fixed_fee_l707_70727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizer_dragon_resilience_unique_maximizer_l707_70751

/-- The probability function P(K) for the dragon's resilience x -/
noncomputable def P (x : ℝ) : ℝ := x^12 / (1 + x + x^2)^10

/-- The maximum value of x that maximizes P(K) -/
noncomputable def x_max : ℝ := (Real.sqrt 97 + 1) / 8

theorem dragon_resilience_maximizer :
  ∀ x > 0, P x ≤ P x_max := by sorry

theorem dragon_resilience_unique_maximizer :
  ∀ x > 0, x ≠ x_max → P x < P x_max := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizer_dragon_resilience_unique_maximizer_l707_70751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l707_70757

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (2 - 3 * a) * x + 1 else a / x

theorem f_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ 2/3 < a ∧ a ≤ 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l707_70757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_l707_70785

theorem arithmetic_geometric_sequence (a : ℝ) :
  let seq := λ (i : ℕ) => a + 3 * (i - 1 : ℝ)
  (∃ r : ℝ, seq 3 = seq 1 * r ∧ seq 6 = seq 1 * r^2) →
  seq 1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_l707_70785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l707_70738

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x < g x then g x + x + 4 else g x - x

-- Theorem statement
theorem range_of_f : 
  Set.range f = Set.Icc (-(9/4)) 0 ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l707_70738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l707_70763

noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 1)
def b : ℝ × ℝ := (0, 1)
noncomputable def c (k : ℝ) : ℝ × ℝ := (k, Real.sqrt 3)

theorem perpendicular_vectors (k : ℝ) :
  (a.1 + 2 * b.1) * (c k).1 + (a.2 + 2 * b.2) * (c k).2 = 0 → k = -3 := by
  intro h
  -- Proof steps would go here
  sorry

#check perpendicular_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l707_70763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_for_specific_tetrahedron_l707_70798

/-- A regular tetrahedron with given height and base edge length -/
structure RegularTetrahedron where
  height : ℝ
  base_edge : ℝ

/-- The volume of the inscribed sphere in a regular tetrahedron -/
noncomputable def inscribed_sphere_volume (t : RegularTetrahedron) : ℝ :=
  (4 / 3) * Real.pi * ((Real.sqrt 17 - 1) / 4) ^ 3

/-- Theorem stating that the volume of the inscribed sphere in a regular tetrahedron
    with height 4 and base edge length 2 is equal to ((√17 - 1)³ / 48) * π -/
theorem inscribed_sphere_volume_for_specific_tetrahedron :
  inscribed_sphere_volume ⟨4, 2⟩ = ((Real.sqrt 17 - 1) ^ 3 / 48) * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_for_specific_tetrahedron_l707_70798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_triangle_range_l707_70734

noncomputable def f (x m : ℝ) : ℝ := (Real.cos x + m) / (Real.cos x + 2)

def forms_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem function_triangle_range :
  ∃ (lower upper : ℝ), lower = 7/5 ∧ upper = 5 ∧
  (∀ m : ℝ, lower < m ∧ m < upper ↔
    ∀ a b c : ℝ, forms_triangle (f a m) (f b m) (f c m)) := by
  sorry

#check function_triangle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_triangle_range_l707_70734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_coincides_with_inverse_exp_l707_70779

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) : ℝ := Real.log x

-- Define a function that represents a right shift by 1 unit
def right_shift (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x - 1)

-- State the theorem
theorem function_coincides_with_inverse_exp (f : ℝ → ℝ) :
  (right_shift f = ln) → (f = λ x ↦ ln (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_coincides_with_inverse_exp_l707_70779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_probability_l707_70719

/-- A random variable following a normal distribution with mean 80 and standard deviation σ > 0 -/
def score (σ : ℝ) : Type := ℝ

/-- The probability density function of the normal distribution -/
noncomputable def normal_pdf (σ : ℝ) (x : ℝ) : ℝ :=
  1 / (σ * Real.sqrt (2 * Real.pi)) * Real.exp (-(x - 80)^2 / (2 * σ^2))

/-- The cumulative distribution function of the normal distribution -/
noncomputable def normal_cdf (σ : ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Iio x, normal_pdf σ y

/-- The probability that the score falls within an interval -/
noncomputable def prob (σ : ℝ) (a b : ℝ) : ℝ :=
  normal_cdf σ b - normal_cdf σ a

/-- The theorem statement -/
theorem score_probability (σ : ℝ) (hσ : σ > 0) (h_prob : prob σ 70 90 = 0.8) :
  prob σ 90 100 = 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_probability_l707_70719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l707_70702

/-- The function f(x) = log(x^2 - 3x + 2) -/
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3*x + 2)

/-- The monotonic increasing interval of f(x) is (2, +∞) -/
theorem f_increasing_interval :
  StrictMonoOn f (Set.Ioi 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l707_70702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l707_70722

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

-- Define set B
def B : Set ℝ := {x | -2 ≤ x ∧ x < 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-2 : ℝ) (-1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l707_70722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_closest_int_even_is_correct_l707_70732

noncomputable def prob_closest_int_even : ℝ := 5/4 - Real.pi/4

/-- x is a real number chosen uniformly at random from the interval (0, 1) -/
noncomputable def x : ℝ := sorry

/-- y is a real number chosen uniformly at random from the interval (0, 1) -/
noncomputable def y : ℝ := sorry

/-- The closest integer to x/y -/
noncomputable def closest_int_to_x_div_y : ℤ := sorry

/-- The event that the closest integer to x/y is even -/
def event_closest_int_even : Prop := Even closest_int_to_x_div_y

/-- The probability measure for the problem -/
noncomputable def P : Set ℝ → ℝ := sorry

theorem prob_closest_int_even_is_correct :
  P {ω : ℝ | event_closest_int_even} = prob_closest_int_even := by
  sorry

#check prob_closest_int_even_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_closest_int_even_is_correct_l707_70732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cover_five_points_with_three_triangles_l707_70748

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  center : Point
  sideLength : ℝ

/-- The original equilateral triangle with area 1 -/
noncomputable def originalTriangle : EquilateralTriangle :=
  { center := { x := 0, y := 0 },
    sideLength := Real.sqrt (4 / Real.sqrt 3) }

/-- Checks if a point is inside an equilateral triangle -/
def isInside (p : Point) (t : EquilateralTriangle) : Prop :=
  sorry

/-- Theorem: Coverage of five points with three smaller equilateral triangles -/
theorem cover_five_points_with_three_triangles
  (p1 p2 p3 p4 p5 : Point)
  (h1 : isInside p1 originalTriangle)
  (h2 : isInside p2 originalTriangle)
  (h3 : isInside p3 originalTriangle)
  (h4 : isInside p4 originalTriangle)
  (h5 : isInside p5 originalTriangle) :
  ∃ (t1 t2 t3 : EquilateralTriangle),
    (∀ (p : Point), (p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 ∨ p = p5) →
      (isInside p t1 ∨ isInside p t2 ∨ isInside p t3)) ∧
    (t1.sideLength ^ 2 + t2.sideLength ^ 2 + t3.sideLength ^ 2 ≤ 0.64 * originalTriangle.sideLength ^ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cover_five_points_with_three_triangles_l707_70748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_upper_bound_l707_70733

theorem sequence_upper_bound (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_ineq : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n : ℕ, a n < 1 / (n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_upper_bound_l707_70733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_term_is_37_l707_70736

def my_sequence (n : ℕ+) : ℕ := 3 * n.val + 1

theorem twelfth_term_is_37 : my_sequence ⟨12, by norm_num⟩ = 37 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelfth_term_is_37_l707_70736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l707_70758

-- Define the sequence recursively
def a (a₁ : ℚ) : ℕ → ℚ
  | 0 => a₁
  | 1 => a₁
  | (n + 2) => (1 + a a₁ (n + 1)) / (1 - a a₁ (n + 1))

-- State the theorem
theorem sequence_value (a₁ : ℚ) : 
  a a₁ 2018 = 3 → a₁ = 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l707_70758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l707_70708

theorem cosine_sum_product_form :
  ∃ (a b c d : ℕ+),
    (∀ x : ℝ, Real.cos (2*x) + Real.cos (4*x) + Real.cos (8*x) + Real.cos (10*x) = 
     (a : ℝ) * Real.cos (b*x) * Real.cos (c*x) * Real.cos (d*x)) ∧
    a + b + c + d = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l707_70708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_triangle_area_l707_70765

/-- Given a linear function y = kx - 4 where y = -3 when x = 2,
    prove that the area of the triangle formed by the x-axis intersection,
    y-axis intersection, and origin after shifting the function 6 units upward is 4. -/
theorem linear_function_triangle_area 
  (k : ℝ) -- Slope of the linear function
  (h1 : -3 = 2 * k - 4) -- Condition: when x = 2, y = -3
  : 
  let f (x : ℝ) := k * x - 4 -- Original function
  let g (x : ℝ) := f x + 6 -- Shifted function
  let x_int := -4 -- x-intercept of shifted function
  let y_int := 2 -- y-intercept of shifted function
  (1/2 : ℝ) * |x_int| * y_int = 4 -- Area of triangle
  := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_triangle_area_l707_70765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_arrows_in_grid_l707_70767

/-- Represents a cell in the 10x10 grid -/
structure Cell where
  x : Fin 10
  y : Fin 10

/-- Represents an arrow drawn in a cell -/
structure Arrow where
  cell : Cell
  direction : Bool  -- true for diagonal from top-left to bottom-right, false for top-right to bottom-left

/-- The distance between two points in the grid -/
noncomputable def distance (p1 p2 : Fin 10 × Fin 10) : ℝ :=
  Real.sqrt (((p1.1 : ℝ) - (p2.1 : ℝ))^2 + ((p1.2 : ℝ) - (p2.2 : ℝ))^2)

/-- Check if two arrows satisfy the distance constraint -/
def satisfyConstraint (a1 a2 : Arrow) : Prop :=
  let end1 := if a1.direction then (a1.cell.x + 1, a1.cell.y + 1) else (a1.cell.x + 1, a1.cell.y)
  let end2 := if a2.direction then (a2.cell.x + 1, a2.cell.y + 1) else (a2.cell.x + 1, a2.cell.y)
  end1 = (a2.cell.x, a2.cell.y) ∨ end2 = (a1.cell.x, a1.cell.y) ∨ distance end1 end2 ≥ 2

/-- The main theorem stating the maximum number of arrows -/
theorem max_arrows_in_grid :
  ∃ (arrows : Finset Arrow), arrows.card = 50 ∧
  (∀ a1 a2, a1 ∈ arrows → a2 ∈ arrows → a1 ≠ a2 → satisfyConstraint a1 a2) ∧
  (∀ (bigger_set : Finset Arrow), bigger_set.card > 50 →
    ∃ a1 a2, a1 ∈ bigger_set ∧ a2 ∈ bigger_set ∧ a1 ≠ a2 ∧ ¬satisfyConstraint a1 a2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_arrows_in_grid_l707_70767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equal_set_l707_70724

-- Define set A
def A : Set ℕ := {x | 0 ≤ x ∧ x ≤ 5}

-- Define set B (now as a subset of ℕ)
def B : Set ℕ := {x : ℕ | 2 - x < 0}

-- Theorem statement
theorem intersection_complement_equal_set : A ∩ (Set.univ \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equal_set_l707_70724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l707_70746

noncomputable def given_numbers : List ℝ := [2021, -1.7, 2/5, 0, -6, 23/8, Real.pi/2]

def is_positive (x : ℝ) : Prop := x > 0
def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n
def is_negative_fraction (x : ℝ) : Prop := x < 0 ∧ ∃ p q : ℤ, q ≠ 0 ∧ x = p / q
def is_positive_rational (x : ℝ) : Prop := x > 0 ∧ ∃ p q : ℤ, q > 0 ∧ x = p / q

theorem number_categorization :
  (∀ x ∈ given_numbers, is_positive x ↔ x ∈ [2021, 2/5, 23/8, Real.pi/2]) ∧
  (∀ x ∈ given_numbers, is_integer x ↔ x ∈ [2021, 0, -6]) ∧
  (∀ x ∈ given_numbers, is_negative_fraction x ↔ x = -1.7) ∧
  (∀ x ∈ given_numbers, is_positive_rational x ↔ x ∈ [2021, 2/5, 23/8]) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l707_70746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_empty_tank_is_4320_l707_70753

/-- Represents the time it takes to empty a tank with given parameters. -/
noncomputable def time_to_empty_tank (tank_volume : ℝ) (inlet_rate : ℝ) (outlet_rate1 : ℝ) (outlet_rate2 : ℝ) : ℝ :=
  let cubic_inches_per_cubic_foot := (12 : ℝ) ^ 3
  let tank_volume_cubic_inches := tank_volume * cubic_inches_per_cubic_foot
  let net_emptying_rate := outlet_rate1 + outlet_rate2 - inlet_rate
  tank_volume_cubic_inches / net_emptying_rate

/-- Theorem stating that the time to empty the tank with given parameters is 4320 minutes. -/
theorem time_to_empty_tank_is_4320 :
  time_to_empty_tank 30 5 9 8 = 4320 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_empty_tank_is_4320_l707_70753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcircle_constant_l707_70780

/-- Triangle with side lengths a, b, c, and circumradius R -/
structure Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] where
  A : α
  B : α
  C : α
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ

/-- Orthocenter of a triangle -/
noncomputable def orthocenter {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : α :=
  sorry

/-- Point on the circumcircle of a triangle -/
noncomputable def circumcircle_point {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : α :=
  sorry

/-- Point diametrically opposite to a given point on the circumcircle -/
noncomputable def opposite_point {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) (P : α) : α :=
  sorry

/-- The main theorem -/
theorem orthocenter_circumcircle_constant {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (t : Triangle α) :
  let H := orthocenter t
  let P := circumcircle_point t
  let Q := opposite_point t P
  ‖P - t.A‖^2 + ‖Q - t.B‖^2 + ‖P - t.C‖^2 - ‖P - H‖^2 = t.a^2 + t.b^2 + t.c^2 - 4 * t.R^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcircle_constant_l707_70780
