import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l758_75818

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  y_min : ℝ
  x_max : ℝ
  y_max : ℝ

/-- The area of a rectangle --/
def Rectangle.area (r : Rectangle) : ℝ :=
  (r.x_max - r.x_min) * (r.y_max - r.y_min)

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point satisfies x > 5y --/
def satisfies_condition (p : Point) : Prop :=
  p.x > 5 * p.y

/-- The probability of a randomly chosen point in the rectangle satisfying x > 5y --/
noncomputable def probability_satisfying_condition (r : Rectangle) : ℝ :=
  200 / 20033

theorem probability_theorem (r : Rectangle) 
  (h1 : r.x_min = 0) (h2 : r.y_min = 0) (h3 : r.x_max = 3000) (h4 : r.y_max = 3005) :
  probability_satisfying_condition r = 200 / 20033 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l758_75818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_AB_ratio_l758_75867

/-- Represents the length of a path between two cities on a ring road. -/
structure PathLength (A B : Type) where
  length : ℝ
  positive : length > 0

/-- The ring road connecting cities A, B, and C. -/
class RingRoad (A B C : Type) where
  path_AC_not_B : PathLength A C
  path_BC_not_A : PathLength B C
  path_AB_not_C : PathLength A B
  path_AC_through_B : PathLength A C
  path_BC_through_A : PathLength B C
  path_AB_through_C : PathLength A B
  
  AC_condition : path_AC_not_B.length = 3 * (path_AB_not_C.length + path_BC_not_A.length)
  BC_condition : path_BC_not_A.length * 4 = path_AB_not_C.length + path_AC_not_B.length

/-- The main theorem stating the relationship between the direct and indirect paths from A to B. -/
theorem path_AB_ratio {A B C : Type} [r : RingRoad A B C] :
  (r.path_AB_through_C.length / r.path_AB_not_C.length) = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_AB_ratio_l758_75867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_not_both_on_extension_line_l758_75839

/-- Given distinct points A, B, C, D in ℝ², prove that C and D cannot both be on the extension line of AB -/
theorem points_not_both_on_extension_line (A B C D : ℝ × ℝ) (lambda mu : ℝ) : 
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (C - A : ℝ × ℝ) = lambda • (B - A) →
  (D - A : ℝ × ℝ) = mu • (B - A) →
  1 / lambda + 1 / mu = 2 →
  ¬(lambda > 1 ∧ mu > 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_not_both_on_extension_line_l758_75839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_sum_l758_75860

theorem max_power_sum (a b : ℕ) : 
  b > 1 → 
  a^b < 500 → 
  (∀ (x y : ℕ), y > 1 → x^y < 500 → x^y ≤ a^b) → 
  a + b = 24 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_sum_l758_75860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_sides_existence_l758_75865

/-- A polygon is represented by its vertices as a list of points in the plane. -/
def Polygon := List (ℝ × ℝ)

/-- A side of a polygon is a pair of consecutive vertices. -/
def Side (p : Polygon) := (ℝ × ℝ) × (ℝ × ℝ)

/-- Two sides are parallel if they have the same slope. -/
def Parallel (s1 s2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (x1, y1) := s1.1
  let (x2, y2) := s1.2
  let (x3, y3) := s2.1
  let (x4, y4) := s2.2
  (y2 - y1) * (x4 - x3) = (y4 - y3) * (x2 - x1)

/-- A polygon has the parallel sides property if each of its sides is parallel to another side. -/
def HasParallelSidesProperty (p : Polygon) : Prop :=
  ∀ s : Side p, ∃ s' : Side p, s ≠ s' ∧ Parallel s s'

/-- The main theorem stating the condition for existence of a polygon with parallel sides property. -/
theorem parallel_sides_existence (n : ℕ) (h : n ≥ 3) :
  (∃ p : Polygon, p.length = n ∧ HasParallelSidesProperty p) ↔ (Even n ∨ n ≥ 7) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_sides_existence_l758_75865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_extreme_points_l758_75826

noncomputable section

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + x - 3

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

/-- Predicate to check if f(x) has two extreme points -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0

/-- Theorem stating the range of a when f(x) has two extreme points -/
theorem range_of_a_for_two_extreme_points :
  ∀ a : ℝ, has_two_extreme_points a → a < -2 ∨ a > 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_extreme_points_l758_75826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_d_value_l758_75895

theorem quadratic_roots_d_value :
  ∀ d : ℝ,
  (∀ x : ℝ, x^2 - 3*x + d = 0 ↔ x = (3 + Real.sqrt (d - 1)) / 2 ∨ x = (3 - Real.sqrt (d - 1)) / 2) →
  d = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_d_value_l758_75895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l758_75832

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (s1 s2 angle : ℝ) : ℝ :=
  1/2 * s1 * s2 * Real.sin angle

theorem triangle_area_proof (t : Triangle) 
  (h1 : t.a = 3 * Real.sqrt 2)
  (h2 : t.c = Real.sqrt 3)
  (h3 : Real.cos t.A = Real.sqrt 3 / 3) :
  triangleArea t.b t.c t.A = 5 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l758_75832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_approximation_max_prism_volume_approximation_l758_75844

-- Constants
noncomputable def paperSideLength : ℝ := 1

-- Cone wrapping
noncomputable def coneSlantHeight (r : ℝ) : ℝ := 4 * r
noncomputable def coneRadius : ℝ := Real.sqrt 2 / (5 + Real.sqrt 2)
noncomputable def coneHeight (r : ℝ) : ℝ := Real.sqrt (15 * r^2)
noncomputable def coneVolume (r : ℝ) : ℝ := (Real.sqrt 15 / 3) * Real.pi * r^3

-- Rectangular prism wrapping
noncomputable def prismVolume (h : ℝ) : ℝ := h^3 - (3/2) * h^2 + h/2

-- Theorem statements
theorem cone_volume_approximation :
  ∃ ε > 0, |coneVolume coneRadius - 0.0435| < ε := by
  sorry

theorem max_prism_volume_approximation :
  ∃ h ε, h > 0 ∧ h < 0.5 ∧ ε > 0 ∧
  (∀ h', h' > 0 → h' < 0.5 → prismVolume h' ≤ prismVolume h) ∧
  |prismVolume h - 0.0481| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_approximation_max_prism_volume_approximation_l758_75844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l758_75884

open Set Real

noncomputable def A : Set ℝ := {x | 0 < log x / log 4 ∧ log x / log 4 < 1}
def B : Set ℝ := Iic 2

theorem intersection_A_B : A ∩ B = Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l758_75884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l758_75834

/-- A sphere. -/
structure Sphere where
  radius : ℝ
  surfaceArea : ℝ

/-- A regular tetrahedron. -/
structure RegularTetrahedron where
  edgeLength : ℝ
  circumscribedSphere : Sphere

/-- The surface area of a sphere circumscribed around a regular tetrahedron with edge length 2 is 6π. -/
theorem circumscribed_sphere_surface_area (tetrahedron : RegularTetrahedron) 
  (h : tetrahedron.edgeLength = 2) : 
  tetrahedron.circumscribedSphere.surfaceArea = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l758_75834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_satisfies_conditions_l758_75813

noncomputable def A : ℝ × ℝ × ℝ := (2, -1, 4)
noncomputable def B : ℝ × ℝ × ℝ := (-1, 2, 5)
noncomputable def P : ℝ × ℝ × ℝ := (0, 3/2, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.fst - p2.fst)^2 + (p1.snd.fst - p2.snd.fst)^2 + (p1.snd.snd - p2.snd.snd)^2)

theorem point_P_satisfies_conditions :
  P.fst = 0 ∧ P.snd.snd = 0 ∧ distance P A = distance P B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_satisfies_conditions_l758_75813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_1000_eq_188_l758_75856

/-- The number of distinguishable arrangements of flags on two poles --/
def M : ℕ := 6188  -- Actual value based on the solution

/-- Condition: There are two distinguishable flagpoles --/
axiom two_poles : ℕ
axiom two_poles_eq : two_poles = 2

/-- Condition: There are 25 flags in total --/
axiom total_flags : ℕ
axiom total_flags_eq : total_flags = 25

/-- Condition: 13 flags are identical red flags --/
axiom red_flags : ℕ
axiom red_flags_eq : red_flags = 13

/-- Condition: 12 flags are identical yellow flags --/
axiom yellow_flags : ℕ
axiom yellow_flags_eq : yellow_flags = 12

/-- Condition: Each flagpole must have at least one flag --/
axiom at_least_one_flag_per_pole : ∀ pole : Fin two_poles, ∃ flag : ℕ, flag > 0

/-- Helper function to determine if a flag is yellow --/
def is_yellow : ℕ → Prop := sorry

/-- Condition: No two yellow flags on either pole are adjacent --/
axiom no_adjacent_yellow_flags : ∀ pole : Fin two_poles, ∀ i j : ℕ, i + 1 = j → ¬(is_yellow i ∧ is_yellow j)

/-- Theorem: The remainder when M is divided by 1000 is 188 --/
theorem M_mod_1000_eq_188 : M % 1000 = 188 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_1000_eq_188_l758_75856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_exists_isosceles_triangle_no_two_right_angles_l758_75814

-- Define a triangle in Euclidean space
structure EuclideanTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_nondegenerate : A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define the angle measure in degrees
noncomputable def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

-- Theorem 1: Sum of interior angles is 180°
theorem triangle_angle_sum (t : EuclideanTriangle) :
  angle_measure t.A t.B t.C + angle_measure t.B t.C t.A + angle_measure t.C t.A t.B = 180 := by
  sorry

-- Theorem 2: Existence of isosceles triangles
theorem exists_isosceles_triangle :
  ∃ t : EuclideanTriangle, ∃ s : ℝ, 
    (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = s ∧ 
    (t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2 = s := by
  sorry

-- Theorem 3: No triangle has two right angles
theorem no_two_right_angles (t : EuclideanTriangle) :
  ¬(angle_measure t.A t.B t.C = 90 ∧ angle_measure t.B t.C t.A = 90) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_exists_isosceles_triangle_no_two_right_angles_l758_75814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cards_proof_l758_75840

def max_trading_cards (total_money : ℚ) (card_cost : ℚ) : ℕ :=
  (total_money / card_cost).floor.toNat

theorem max_cards_proof (total_money : ℚ) (card_cost : ℚ) 
  (h1 : total_money = 10) 
  (h2 : card_cost = 5/4) : 
  max_trading_cards total_money card_cost = 8 := by
  sorry

#eval max_trading_cards 10 (5/4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cards_proof_l758_75840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_l758_75894

/-- Represents a quadrilateral with a diagonal and two offsets --/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets --/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (1/2) * q.diagonal * (q.offset1 + q.offset2)

/-- Theorem: Given a quadrilateral with diagonal 28 cm, offset1 8 cm, and area 140 cm², 
    the second offset is 2 cm --/
theorem quadrilateral_offset (q : Quadrilateral) 
    (h1 : q.diagonal = 28)
    (h2 : q.offset1 = 8)
    (h3 : area q = 140) : 
  q.offset2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_l758_75894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l758_75857

-- Define the polynomials
def f (x : ℝ) : ℝ := 3*x^4 + 5*x^3 - 4*x^2 + 2*x + 1
def d (x : ℝ) : ℝ := x^2 + 2*x - 3
def q (x : ℝ) : ℝ := 3*x^2 + x
def r (x : ℝ) : ℝ := 7*x + 4

-- State the theorem
theorem polynomial_division_theorem :
  (∀ x, f x = q x * d x + r x) ∧ 
  (Polynomial.degree (Polynomial.C r) < Polynomial.degree (Polynomial.C d)) ∧
  (q (-1) + r 1 = 13) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l758_75857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bernardo_win_probability_l758_75847

/-- Bernardo's set of numbers -/
def bernardoSet : Finset ℕ := Finset.range 10

/-- Silvia's set of numbers -/
def silviaSet : Finset ℕ := Finset.range 9

/-- Type representing a 3-digit number formed by 3 distinct digits in descending order -/
structure ThreeDigitNumber where
  digits : Finset ℕ
  distinct : digits.card = 3
  descending : ∀ x y, x ∈ digits → y ∈ digits → x > y → digits.toList.indexOf x < digits.toList.indexOf y

/-- Function to create a ThreeDigitNumber from a set of 3 distinct numbers -/
noncomputable def makeThreeDigitNumber (s : Finset ℕ) (h : s.card = 3) : ThreeDigitNumber := sorry

/-- Probability that Bernardo's number is greater than Silvia's -/
def winProbability : ℚ := 9 / 14

/-- The probability measure -/
noncomputable def ℙ : (Finset ℕ → Finset ℕ → Prop) → ℚ := sorry

theorem bernardo_win_probability :
  ℙ (fun (b : Finset ℕ) (s : Finset ℕ) =>
      b ⊆ bernardoSet ∧ s ⊆ silviaSet ∧
      b.card = 3 ∧ s.card = 3 ∧
      (makeThreeDigitNumber b (by sorry)).digits.max > (makeThreeDigitNumber s (by sorry)).digits.max) =
    winProbability := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bernardo_win_probability_l758_75847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_constraint_l758_75821

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

theorem domain_range_constraint (a b : ℝ) (h1 : a < b) 
  (h2 : ∀ x, a ≤ x ∧ x ≤ b → -2 ≤ f x ∧ f x < 1) : 
  b - a ≠ 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_constraint_l758_75821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_m_values_l758_75838

/-- A line that intersects a circle -/
structure IntersectingLine where
  m : ℝ
  intersects_circle : ∃ (A B : ℝ × ℝ), 
    (A.1 - A.2 * m + 1 = 0 ∧ (A.1 - 1)^2 + A.2^2 = 4) ∧
    (B.1 - B.2 * m + 1 = 0 ∧ (B.1 - 1)^2 + B.2^2 = 4) ∧
    A ≠ B

/-- The circle with equation (x-1)^2 + y^2 = 4 -/
def Circle : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + p.2^2 = 4}

/-- The center of the circle -/
def C : ℝ × ℝ := (1, 0)

/-- The area of a triangle given three points -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

/-- The theorem stating the possible values of m -/
theorem intersecting_line_m_values (l : IntersectingLine) :
  (∃ (A B : ℝ × ℝ), A ∈ Circle ∧ B ∈ Circle ∧ A ≠ B ∧ 
    triangleArea A B C = 8/5) →
  l.m = 2 ∨ l.m = -2 ∨ l.m = 1/2 ∨ l.m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_m_values_l758_75838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_greater_than_geometric_mean_l758_75899

theorem arithmetic_mean_greater_than_geometric_mean 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_greater_than_geometric_mean_l758_75899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_distinct_powers_of_two_l758_75815

theorem polynomial_distinct_powers_of_two (n : ℕ+) :
  ∃ p : Polynomial ℤ, 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ k : ℕ, p.eval (↑i : ℤ) = 2^k) ∧
    (∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ i ≠ j → 
      p.eval (↑i : ℤ) ≠ p.eval (↑j : ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_distinct_powers_of_two_l758_75815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_monotone_increasing_l758_75853

/-- The function f(x) defined as x - a * sqrt(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.sqrt x

/-- The property that f is monotonically increasing on the interval [1, 4] --/
def is_monotone_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 4 → f a x ≤ f a y

/-- The theorem stating that the maximum value of a for which f(x) is monotonically increasing on [1, 4] is 2 --/
theorem max_a_for_monotone_increasing : 
  (∃ a_max : ℝ, 
    (∀ a : ℝ, a ≤ a_max → is_monotone_increasing_on_interval a) ∧ 
    (∀ a : ℝ, a > a_max → ¬is_monotone_increasing_on_interval a)) ∧
  (∀ a_max : ℝ, 
    ((∀ a : ℝ, a ≤ a_max → is_monotone_increasing_on_interval a) ∧ 
     (∀ a : ℝ, a > a_max → ¬is_monotone_increasing_on_interval a)) → 
    a_max = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_monotone_increasing_l758_75853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_size_l758_75861

open Set Finset

theorem min_intersection_size (X Y Z : Finset (Fin 82)) : 
  (X.card = 80) →
  (Y.card = 80) →
  ((X ∪ Y ∪ Z).card = X.card + Y.card + Z.card) →
  77 ≤ (X ∩ Y ∩ Z).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_size_l758_75861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l758_75819

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sin (ω * x) * Real.cos (ω * x) - Real.sqrt 3 / 2

theorem function_properties (ω : ℝ) (h1 : ω > 0) (h2 : Function.Periodic f π) :
  ω = 1 ∧ ∃ x₀ ∈ Set.Icc (-π/3) (π/6), ∀ x ∈ Set.Icc (-π/3) (π/6), f ω x₀ ≤ f ω x ∧ f ω x₀ = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l758_75819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_range_l758_75876

-- Define the function f(x)
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

-- Define helper functions/predicates (these are placeholders and need to be properly defined)
def SmallestPositivePeriod (f : ℝ → ℝ) : ℝ := sorry
def ExactlyTwoExtremePoints (f : ℝ → ℝ) (a b : ℝ) : Prop := sorry

-- State the theorem
theorem extreme_points_range (ω : ℝ) (h1 : ω > 0) (h2 : SmallestPositivePeriod (f ω) = Real.pi) :
  ∃ (m_lower m_upper : ℝ),
    m_lower = 5 * Real.pi / 12 ∧
    m_upper = 7 * Real.pi / 12 ∧
    ∀ m : ℝ, (ExactlyTwoExtremePoints (f ω) (-m) m) ↔ (m_lower < m ∧ m ≤ m_upper) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_range_l758_75876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_prime_l758_75824

/-- The sequence defined as 10^(4n) + 1 for n ≥ 1 -/
def a (n : ℕ) : ℕ := 10^(4*n) + 1

/-- The first term of the sequence is 10001, which equals 73 * 137 -/
axiom first_term : a 1 = 73 * 137

/-- Theorem: Every term in the sequence is not prime -/
theorem sequence_not_prime (n : ℕ) (h : n ≥ 1) : ¬ Nat.Prime (a n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_prime_l758_75824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l758_75891

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vector operations
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Theorem statement
theorem triangle_side_length (t : Triangle) 
  (h1 : vec_length (vec t.A t.B) = 4)
  (h2 : vec_length (vec t.A t.C) = 3)
  (h3 : dot_product (vec t.A t.C) (vec t.B t.C) = 1) :
  vec_length (vec t.B t.C) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l758_75891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l758_75871

theorem solve_exponential_equation :
  ∃ y : ℚ, (1/8 : ℝ)^(3*y + 6 : ℝ) = (64 : ℝ)^(3*y - 2 : ℝ) ∧ y = -2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l758_75871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_in_candidates_l758_75828

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def candidates : List ℕ := [303201, 303203, 303205, 303207, 303209]

theorem unique_prime_in_candidates : 
  ∃! n, n ∈ candidates ∧ is_prime n ∧ n = 303209 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_in_candidates_l758_75828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_sum_l758_75802

theorem decimal_to_fraction_sum (x : ℚ) (h : x = 2.52) :
  ∃ (n d : ℕ), (n : ℚ) / d = x ∧ Nat.Coprime n d ∧ n + d = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_sum_l758_75802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l758_75888

/-- An ellipse with foci at (9, 20) and (49, 55) in the xy-plane and tangent to the x-axis has a major axis of length 85. -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)),
    (∃ (k : ℝ), ∀ (p : ℝ × ℝ), p ∈ E ↔ 
      Real.sqrt ((p.1 - 9)^2 + (p.2 - 20)^2) + 
      Real.sqrt ((p.1 - 49)^2 + (p.2 - 55)^2) = k) →
    (∃ (x : ℝ), (x, 0) ∈ E) →
    (∀ (x : ℝ), (x, 0) ∉ interior E) →
    (∃ (a b : ℝ × ℝ), a ∈ E ∧ b ∈ E ∧ 
      Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 85) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l758_75888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_abs_equal_implies_product_one_l758_75887

-- Define the function f(x) = |log x|
noncomputable def f (x : ℝ) : ℝ := |Real.log x|

-- State the theorem
theorem log_abs_equal_implies_product_one (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (h : f a = f b) : a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_abs_equal_implies_product_one_l758_75887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_prob_correct_l758_75883

def basketball_prob (p : ℝ) (n : ℕ) : ℝ × ℝ × ℝ :=
  let q := 1 - p
  let exactly_two := (Nat.choose n 2 : ℝ) * p^2 * q^(n-2)
  let at_least_two := 1 - ((Nat.choose n 0 : ℝ) * q^n + (Nat.choose n 1 : ℝ) * p * q^(n-1))
  let at_most_two := (Nat.choose n 0 : ℝ) * q^n + (Nat.choose n 1 : ℝ) * p * q^(n-1) + (Nat.choose n 2 : ℝ) * p^2 * q^(n-2)
  (exactly_two, at_least_two, at_most_two)

theorem basketball_prob_correct :
  let (exactly_two, at_least_two, at_most_two) := basketball_prob 0.7 4
  abs (exactly_two - 0.2646) < 0.0001 ∧ 
  abs (at_least_two - 0.9163) < 0.0001 ∧ 
  abs (at_most_two - 0.3483) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_prob_correct_l758_75883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l758_75846

theorem coefficient_of_x (x : ℝ) : 
  let expr := 5*(x - 6) - 6*(3 - x^2 + 3*x) + 7*(4*x - 5)
  expr = 15*x + 6*x^2 - 83 := by
  simp [mul_sub, mul_add]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l758_75846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_between_235_and_240_l758_75809

theorem x_squared_between_235_and_240 (x : ℝ) : 
  ((x + 16) ^ (1/3 : ℝ) - (x - 16) ^ (1/3 : ℝ) = 4) → 
  (235 < x^2 ∧ x^2 < 240) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_between_235_and_240_l758_75809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_decrease_l758_75896

theorem tax_decrease (original_tax original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  ∃ (tax_decrease_percent : ℝ),
    let new_tax := original_tax * (1 - tax_decrease_percent / 100)
    let new_consumption := original_consumption * 1.15
    let original_revenue := original_tax * original_consumption
    let new_revenue := new_tax * new_consumption
    (new_revenue = original_revenue * 0.92) → 
    tax_decrease_percent = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_decrease_l758_75896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_x_coordinate_sum_l758_75842

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola x² = 2py -/
def Parabola (p : ℝ) : Set Point :=
  {point : Point | point.x^2 = 2 * p * point.y}

/-- Represents a line y = -2p -/
def Line (p : ℝ) : Set Point :=
  {point : Point | point.y = -2 * p}

/-- Predicate to check if a point is a tangent point of the parabola from M -/
def IsTangentPoint (p : ℝ) (T M : Point) : Prop :=
  T ∈ Parabola p ∧ 
  ∃ (m : ℝ), ∀ (P : Point), P ∈ Parabola p → 
    P.y - M.y = m * (P.x - M.x) → P = T

/-- Theorem: For a parabola x² = 2py (p > 0) and a point M on the line y = -2p,
    if tangent lines are drawn from M to the parabola touching at points A and B,
    then the x-coordinates of A, B, and M satisfy XA + XB = 2XM -/
theorem parabola_tangent_x_coordinate_sum 
  (p : ℝ) 
  (hp : p > 0)
  (M : Point)
  (hM : M ∈ Line p)
  (A B : Point)
  (hA : A ∈ Parabola p)
  (hB : B ∈ Parabola p)
  (hTangentA : IsTangentPoint p A M)
  (hTangentB : IsTangentPoint p B M) :
  A.x + B.x = 2 * M.x :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_x_coordinate_sum_l758_75842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l758_75866

/-- The solution set of a quadratic inequality -/
def SolutionSet (a b : ℝ) : Set ℝ := Set.union (Set.Iio (-1)) (Set.Ioi 3)

/-- The quadratic inequality -/
def QuadraticInequality (a b : ℝ) (x : ℝ) : Prop := a * x^2 + (b - 2) * x + 3 < 0

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, x ∈ SolutionSet a b ↔ QuadraticInequality a b x) →
  a + b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l758_75866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravitational_force_on_moon_l758_75843

-- Define the gravitational constant k
noncomputable def gravitational_constant (force : ℝ) (distance : ℝ) : ℝ :=
  force * distance^2

-- Define the inverse square law for gravitational force
noncomputable def gravitational_force (k : ℝ) (distance : ℝ) : ℝ :=
  k / distance^2

theorem gravitational_force_on_moon
  (earth_surface_distance : ℝ)
  (earth_surface_force : ℝ)
  (moon_distance : ℝ)
  (h1 : earth_surface_distance = 4000)
  (h2 : earth_surface_force = 600)
  (h3 : moon_distance = 240000)
  : gravitational_force (gravitational_constant earth_surface_force earth_surface_distance) moon_distance = 1/6 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravitational_force_on_moon_l758_75843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_check_l758_75892

theorem largest_prime_divisor_check (n : ℕ) (hn : 1000 ≤ n ∧ n ≤ 1100) :
  Nat.Prime n ↔ (∀ p : ℕ, Nat.Prime p → p ≤ 31 → ¬(p ∣ n)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_check_l758_75892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l758_75875

theorem alpha_value (α : Real) 
    (h1 : α > 0) 
    (h2 : α < Real.pi / 4) 
    (h3 : Real.tan (α + Real.pi / 4) = 2 * Real.cos (2 * α)) : 
  α = Real.arctan (2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l758_75875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l758_75850

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

-- Statement to prove
theorem axis_of_symmetry :
  ∀ x : ℝ, f (Real.pi / 6 + x) = f (Real.pi / 6 - x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l758_75850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l758_75830

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 169 + y^2 / 144 = 1

-- Define the foci
def are_foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let c := Real.sqrt (169 - 144)
  F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define the perimeter of a triangle
noncomputable def triangle_perimeter (A B C : ℝ × ℝ) : ℝ :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B + d B C + d C A

-- Theorem statement
theorem ellipse_triangle_perimeter (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  is_on_ellipse P.1 P.2 →
  are_foci F₁ F₂ →
  triangle_perimeter P F₁ F₂ = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l758_75830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_fraction_l758_75812

theorem min_trig_fraction :
  ∀ x : ℝ, (Real.sin x)^8 + (Real.cos x)^8 + 1 ≥ 2/15 * ((Real.sin x)^6 + (Real.cos x)^6 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_fraction_l758_75812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_equality_l758_75827

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2*x + a else -x - 2*a

theorem piecewise_function_equality (a : ℝ) (h : a ≠ 0) :
  f a (1 - a) = f a (1 + a) → a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_equality_l758_75827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dist_is_orthocenter_l758_75870

/-- Triangle ABC with side lengths 8, 10, and 12 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (side_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8)
  (side_BC : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 12)
  (side_CA : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 10)

/-- Predicate to check if a point is inside a triangle -/
def IsInside (P A B C : ℝ × ℝ) : Prop := sorry

/-- Point P inside the triangle -/
def PointInside (t : Triangle) := { P : ℝ × ℝ // IsInside P t.A t.B t.C }

/-- Distance from a point to a line segment -/
noncomputable def distToSide (P A B : ℝ × ℝ) : ℝ :=
  let v := (B.1 - A.1, B.2 - A.2)
  let w := (P.1 - A.1, P.2 - A.2)
  Real.sqrt (w.1^2 + w.2^2 - ((w.1 * v.1 + w.2 * v.2) / (v.1^2 + v.2^2))^2 * (v.1^2 + v.2^2))

/-- Sum of squares of distances from P to sides of the triangle -/
noncomputable def sumOfSquaredDists (t : Triangle) (P : ℝ × ℝ) : ℝ :=
  (distToSide P t.A t.B)^2 + (distToSide P t.B t.C)^2 + (distToSide P t.C t.A)^2

/-- Orthocenter of a triangle -/
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Main theorem -/
theorem min_dist_is_orthocenter (t : Triangle) :
  ∃ (P : ℝ × ℝ), IsInside P t.A t.B t.C ∧ sumOfSquaredDists t P = 77 ∧ P = orthocenter t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dist_is_orthocenter_l758_75870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_quadrilateral_ABCD_l758_75880

/-- Circle with equation x^2 + y^2 - 4x + 2y = 0 -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y = 0

/-- Point E -/
def E : ℝ × ℝ := (1, 0)

/-- Longest chord passing through E -/
def longest_chord (A C : ℝ × ℝ) : Prop :=
  circleEq A.1 A.2 ∧ circleEq C.1 C.2 ∧ 
  (∃ t : ℝ, A = (1 - t) • E + t • C) ∧
  ∀ P Q : ℝ × ℝ, circleEq P.1 P.2 → circleEq Q.1 Q.2 → 
    (∃ s : ℝ, P = (1 - s) • E + s • Q) → 
    (A.1 - C.1)^2 + (A.2 - C.2)^2 ≥ (P.1 - Q.1)^2 + (P.2 - Q.2)^2

/-- Shortest chord passing through E -/
def shortest_chord (B D : ℝ × ℝ) : Prop :=
  circleEq B.1 B.2 ∧ circleEq D.1 D.2 ∧ 
  (∃ t : ℝ, B = (1 - t) • E + t • D) ∧
  ∀ P Q : ℝ × ℝ, circleEq P.1 P.2 → circleEq Q.1 Q.2 → 
    (∃ s : ℝ, P = (1 - s) • E + s • Q) → 
    (B.1 - D.1)^2 + (B.2 - D.2)^2 ≤ (P.1 - Q.1)^2 + (P.2 - Q.2)^2

/-- Area of quadrilateral ABCD -/
noncomputable def area_ABCD (A B C D : ℝ × ℝ) : ℝ :=
  let S_ABD := (B.1 - A.1) * (D.2 - A.2) - (D.1 - A.1) * (B.2 - A.2)
  let S_BCD := (C.1 - B.1) * (D.2 - B.2) - (D.1 - B.1) * (C.2 - B.2)
  (abs S_ABD + abs S_BCD) / 2

theorem area_quadrilateral_ABCD :
  ∀ A B C D : ℝ × ℝ,
  circleEq A.1 A.2 → circleEq B.1 B.2 → circleEq C.1 C.2 → circleEq D.1 D.2 →
  longest_chord A C → shortest_chord B D →
  area_ABCD A B C D = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_quadrilateral_ABCD_l758_75880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_proof_l758_75855

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 8 = 0

/-- The circumference of the circle -/
noncomputable def circle_circumference : ℝ := 2 * Real.sqrt 2 * Real.pi

/-- Theorem stating the properties of the circle and its circumference -/
theorem circle_circumference_proof :
  ∃ (r : ℝ), r > 0 ∧
  (∀ (x y : ℝ), circle_equation x y ↔ (x - 1)^2 + (y + 3)^2 = r^2) ∧
  circle_circumference = 2 * Real.pi * r := by
  -- We know r = √2, but we'll leave the proof to be filled in later
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_proof_l758_75855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_equilateral_triangles_l758_75811

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The set of distinct equilateral triangles with at least two vertices from the regular polygon -/
noncomputable def distinctEquilateralTriangles (p : RegularPolygon 12) : Set EquilateralTriangle :=
  sorry

/-- The set of distinct equilateral triangles is finite -/
instance (p : RegularPolygon 12) : Fintype (distinctEquilateralTriangles p) :=
  sorry

/-- The count of distinct equilateral triangles is 128 -/
theorem count_distinct_equilateral_triangles (p : RegularPolygon 12) :
  Fintype.card (distinctEquilateralTriangles p) = 128 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_equilateral_triangles_l758_75811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_consumption_theorem_l758_75817

/-- Represents the relationship between coffee consumption, sleep, and extra work hours -/
structure CoffeeConsumption where
  sleep : ℝ
  extraWork : ℝ
  coffee : ℝ

/-- The constant of proportionality in the relationship -/
noncomputable def k (c : CoffeeConsumption) : ℝ := c.sleep * c.coffee / c.extraWork

theorem coffee_consumption_theorem (monday wednesday : CoffeeConsumption) 
  (hm : monday.sleep = 8 ∧ monday.extraWork = 2 ∧ monday.coffee = 4.5)
  (hw : wednesday.sleep = 4 ∧ wednesday.extraWork = 3)
  (hk : k monday = k wednesday) : 
  wednesday.coffee = 13.5 := by
  sorry

#check coffee_consumption_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_consumption_theorem_l758_75817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_theta_symmetry_l758_75831

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

noncomputable def g (θ : ℝ) (x : ℝ) : ℝ := f (2 * (x - θ))

theorem min_theta_symmetry (θ : ℝ) (h1 : θ > 0) :
  (∀ x, g θ (3 * π / 4 + x) = g θ (3 * π / 4 - x)) →
  θ ≥ π / 6 := by
  sorry

#check min_theta_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_theta_symmetry_l758_75831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_after_row_multiplication_l758_75893

def original_matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  !![1, -2, 0;
     5, -1, 2;
     0,  3, -4]

def modified_matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  !![3, -6, 0;
     5, -1, 2;
     0,  3, -4]

theorem det_after_row_multiplication :
  Matrix.det modified_matrix = 114 :=
by
  -- Calculate the determinant
  have h1 : Matrix.det modified_matrix = 3 * (-2) - (-6) * (-20) := by sorry
  -- Simplify
  have h2 : 3 * (-2) - (-6) * (-20) = -6 + 120 := by sorry
  -- Final result
  calc
    Matrix.det modified_matrix = 3 * (-2) - (-6) * (-20) := h1
    _ = -6 + 120 := h2
    _ = 114 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_after_row_multiplication_l758_75893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l758_75882

/-- Represents the possible article choices --/
inductive Article
  | The
  | An
  deriving Repr

/-- Represents the correctness of an answer --/
inductive Correctness
  | Correct
  | Incorrect
  deriving Repr

/-- Checks if the given articles match the correct answer --/
def checkAnswer (first second : Article) : Correctness :=
  match first, second with
  | .The, .An => .Correct
  | _, _ => .Incorrect

/-- Theorem stating that the correct answer is "the" for the first blank and "a" for the second --/
theorem correct_answer :
    checkAnswer Article.The Article.An = Correctness.Correct := by
  rfl

#eval checkAnswer Article.The Article.An

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l758_75882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l758_75848

/-- Circle centered at origin O -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

/-- Parabola y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Directrix of the parabola -/
def Directrix : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -1}

/-- Intersection points of circle with parabola -/
def IntersectParabola (r : ℝ) : Set (ℝ × ℝ) :=
  Circle r ∩ Parabola

/-- Intersection points of circle with directrix -/
def IntersectDirectrix (r : ℝ) : Set (ℝ × ℝ) :=
  Circle r ∩ Directrix

/-- Length of chord AB -/
noncomputable def LengthAB (r : ℝ) : ℝ :=
  let A := (1, 2)
  let B := (1, -2)
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Length of chord CD -/
noncomputable def LengthCD (r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - 1)

theorem circle_equation (r : ℝ) (h : r > 0) :
  (∃ (A B : ℝ × ℝ), A ∈ IntersectParabola r ∧ B ∈ IntersectParabola r ∧
   ∃ (C D : ℝ × ℝ), C ∈ IntersectDirectrix r ∧ D ∈ IntersectDirectrix r ∧
   LengthAB r = LengthCD r) → r^2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l758_75848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_mall_problem_l758_75863

-- Define variables
variable (a b x y : ℝ)

-- Define equations from given conditions
def equation1 (a b : ℝ) : Prop := 2 * a + b = 80
def equation2 (a b : ℝ) : Prop := 3 * a + 2 * b = 135

-- Define sales volume function
def sales_volume (x : ℝ) : ℝ := 100 - 5 * (x - 30)

-- Define profit function
def profit (x : ℝ) : ℝ := (x - 20) * sales_volume x

-- Theorem statement
theorem shopping_mall_problem :
  ∃ a b : ℝ,
    equation1 a b ∧
    equation2 a b ∧
    a = 25 ∧ 
    b = 30 ∧ 
    (∀ x, profit x = -5 * x^2 + 350 * x - 5000) ∧
    (∃ max_profit : ℝ, max_profit = profit 35 ∧ 
      ∀ x, profit x ≤ max_profit) ∧
    profit 35 = 1125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_mall_problem_l758_75863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_power_function_l758_75851

-- Define what a power function is
noncomputable def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the three functions
noncomputable def f₁ (x : ℝ) : ℝ := 1 / (x^2)
noncomputable def f₂ (x : ℝ) : ℝ := -x^2
noncomputable def f₃ (x : ℝ) : ℝ := x^2 + x

-- State the theorem
theorem one_power_function : 
  (is_power_function f₁ ∧ ¬is_power_function f₂ ∧ ¬is_power_function f₃) ∨
  (is_power_function f₂ ∧ ¬is_power_function f₁ ∧ ¬is_power_function f₃) ∨
  (is_power_function f₃ ∧ ¬is_power_function f₁ ∧ ¬is_power_function f₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_power_function_l758_75851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_lines_perpendicular_l758_75804

-- Define the polar coordinate system
structure PolarCoord where
  ρ : ℝ
  θ : ℝ

-- Define the Cartesian coordinate system
structure CartesianCoord where
  x : ℝ
  y : ℝ

-- Define the conversion from polar to Cartesian coordinates
noncomputable def polarToCartesian (p : PolarCoord) : CartesianCoord where
  x := p.ρ * Real.cos p.θ
  y := p.ρ * Real.sin p.θ

-- Define the two lines in polar coordinates
def line1 (α : ℝ) (p : PolarCoord) : Prop := p.ρ * Real.cos (p.θ - α) = 0
def line2 (α a : ℝ) (p : PolarCoord) : Prop := p.ρ * Real.sin (p.θ - α) = a

-- Define perpendicularity in Cartesian coordinates
def perpendicular (l1 l2 : CartesianCoord → Prop) : Prop :=
  ∀ (p q : CartesianCoord), l1 p → l2 q → 
    (p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) ≠ 0

-- State the theorem
theorem polar_lines_perpendicular (α a : ℝ) :
  perpendicular 
    (fun c => ∃ p, polarToCartesian p = c ∧ line1 α p)
    (fun c => ∃ p, polarToCartesian p = c ∧ line2 α a p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_lines_perpendicular_l758_75804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_to_cos_shift_l758_75833

theorem sin_to_cos_shift (x : ℝ) : 
  Real.sin (2 * (x + π / 12)) = Real.cos (2 * x - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_to_cos_shift_l758_75833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_w1_value_l758_75852

/-- A sequence of complex numbers defined by a recurrence relation -/
noncomputable def w : ℕ → ℂ
| 0 => 0  -- Arbitrary initial value, not used in the problem
| n + 1 => (Real.sqrt 2 + Complex.I) * w n

/-- The 50th term of the sequence is 1 + 3i -/
axiom w50 : w 50 = 1 + 3 * Complex.I

/-- The sum of the real and imaginary parts of the first term -/
noncomputable def sum_w1 : ℝ := (w 1).re + (w 1).im

/-- Theorem stating the value of sum_w1 -/
theorem sum_w1_value : sum_w1 = -4 / (3^(49/2) * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_w1_value_l758_75852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_geometric_progression_l758_75879

/-- Given two quadratic equations with coefficients (a₁, b₁, c₁) and (a₂, b₂, c₂),
    if their roots form a geometric progression, then (b₁/b₂)² = (c₁/c₂) * (a₁/a₂) -/
theorem quadratic_roots_geometric_progression 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (ha₁ : a₁ ≠ 0) (ha₂ : a₂ ≠ 0) 
  (hb₂ : b₂ ≠ 0) (hc₂ : c₂ ≠ 0)
  (h_quad₁ : ∀ x : ℝ, a₁ * x^2 + b₁ * x + c₁ = 0 → x ∈ Set.univ)
  (h_quad₂ : ∀ x : ℝ, a₂ * x^2 + b₂ * x + c₂ = 0 → x ∈ Set.univ)
  (h_geom_prog : ∃ r₁ r₂ r₃ r₄ : ℝ, 
    (a₁ * r₁^2 + b₁ * r₁ + c₁ = 0) ∧ 
    (a₁ * r₂^2 + b₁ * r₂ + c₁ = 0) ∧
    (a₂ * r₃^2 + b₂ * r₃ + c₂ = 0) ∧ 
    (a₂ * r₄^2 + b₂ * r₄ + c₂ = 0) ∧
    (∃ q : ℝ, q ≠ 0 ∧ r₂ = q * r₁ ∧ r₃ = q * r₂ ∧ r₄ = q * r₃)) :
  (b₁ / b₂)^2 = (c₁ / c₂) * (a₁ / a₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_geometric_progression_l758_75879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_radius_l758_75873

/-- Represents an ellipse with given major and minor axis lengths -/
structure Ellipse where
  major_axis : ℝ
  minor_axis : ℝ
  major_gt_minor : major_axis > minor_axis

/-- Represents a circle with given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Returns the distance between the center and a focus of the ellipse -/
noncomputable def Ellipse.focal_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.major_axis^2 / 4 - e.minor_axis^2 / 4)

/-- Checks if a circle is tangent to an ellipse -/
def is_tangent (e : Ellipse) (c : Circle) : Prop := sorry

/-- Checks if a circle is entirely contained within an ellipse -/
def is_contained (e : Ellipse) (c : Circle) : Prop := sorry

/-- Main theorem: The radius of the circle tangent to the ellipse and centered at its focus is 3 -/
theorem ellipse_tangent_circle_radius 
  (e : Ellipse) 
  (c : Circle) 
  (h1 : e.major_axis = 12) 
  (h2 : e.minor_axis = 6) 
  (h3 : c.center = (-e.focal_distance, 0)) 
  (h4 : is_tangent e c) 
  (h5 : is_contained e c) : 
  c.radius = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_radius_l758_75873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_attempts_for_seven_safes_min_attempts_is_minimum_l758_75825

/-- Represents the number of safes and codes -/
def n : ℕ := 7

/-- Calculates the minimum number of attempts needed to match n safes with n codes -/
def min_attempts (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that for 7 safes and codes, the minimum number of attempts is 21 -/
theorem min_attempts_for_seven_safes : 
  min_attempts n = 21 := by sorry

/-- Theorem proving that min_attempts gives the minimum number of attempts needed -/
theorem min_attempts_is_minimum (n : ℕ) : 
  ∀ k : ℕ, k < min_attempts n → ∃ arrangement : Fin n → Fin n, 
    ¬∀ i : Fin n, ∃ j : Fin n, (j.val < k + 1 ∧ arrangement j = i) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_attempts_for_seven_safes_min_attempts_is_minimum_l758_75825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_not_always_180_l758_75881

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a plane --/
def Point := ℝ × ℝ

/-- Predicate to check if a point lies on a circle --/
def lies_on (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Predicate to check if two circles intersect at a point --/
def intersect_at (c1 c2 : Circle) (p : Point) : Prop :=
  lies_on p c1 ∧ lies_on p c2

/-- The angle formed by tangents at the intersection of two circles --/
noncomputable def angle_at_intersection (c1 c2 : Circle) (p : Point) : ℝ :=
  sorry -- Definition of angle calculation

/-- Theorem: The sum of angles is not always 180° --/
theorem sum_of_angles_not_always_180 :
  ∃ (c1 c2 c3 : Circle) (P A B C : Point),
    (∀ c ∈ [c1, c2, c3], lies_on P c) ∧
    intersect_at c1 c2 A ∧
    intersect_at c2 c3 B ∧
    intersect_at c3 c1 C ∧
    A ≠ P ∧ B ≠ P ∧ C ≠ P ∧
    angle_at_intersection c1 c2 A +
    angle_at_intersection c2 c3 B +
    angle_at_intersection c3 c1 C ≠ 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_not_always_180_l758_75881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_halves_l758_75859

/-- The series term for a given n ≥ 2 -/
noncomputable def seriesTerm (n : ℕ) : ℝ :=
  (6 * n^3 - 2 * n^2 - 2 * n + 2) / (n^6 - 2 * n^5 + 2 * n^4 - 2 * n^3 + 2 * n^2 - 2 * n)

/-- The sum of the series from n = 2 to infinity -/
noncomputable def seriesSum : ℝ := ∑' n, if n ≥ 2 then seriesTerm n else 0

theorem series_sum_equals_three_halves : seriesSum = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_halves_l758_75859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_heights_inside_l758_75890

/-- Definition of an obtuse triangle -/
structure ObtuseTriangle where
  /-- The triangle has one angle greater than 90 degrees -/
  has_obtuse_angle : ∃ angle, angle > 90

/-- Definition of a height (altitude) of a triangle -/
def Height (triangle : ObtuseTriangle) := Real

/-- Predicate to check if a height is inside the triangle -/
def IsInside (h : Real) (triangle : ObtuseTriangle) : Prop := sorry

/-- Theorem: Not all heights of an obtuse triangle are inside the triangle -/
theorem not_all_heights_inside (triangle : ObtuseTriangle) :
  ∃ h : Height triangle, ¬(IsInside h triangle) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_heights_inside_l758_75890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l758_75803

theorem equation_solution : ∃ x : ℝ, 5 * 1.6 - (x * 1.4) / 1.3 = 4 ∧ 
  |x + 3.71| < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l758_75803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_flour_ratio_l758_75810

/-- Given a recipe with 50 ounces of sugar and 5 ounces of flour,
    prove that the ratio of sugar to flour is 10. -/
theorem sugar_flour_ratio : 
  (50 : ℝ) / 5 = 10 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_flour_ratio_l758_75810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_reciprocal_squares_l758_75874

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  /-- Distance from the origin to the plane -/
  distance : ℝ
  /-- The plane intersects the x-axis at this point -/
  x_intersect : ℝ
  /-- The plane intersects the y-axis at this point -/
  y_intersect : ℝ
  /-- The plane intersects the z-axis at this point -/
  z_intersect : ℝ
  /-- The intersection points are distinct from the origin -/
  distinct_from_origin : x_intersect ≠ 0 ∧ y_intersect ≠ 0 ∧ z_intersect ≠ 0

/-- The centroid of a triangle formed by the intersection points -/
noncomputable def centroid (plane : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (plane.x_intersect / 3, plane.y_intersect / 3, plane.z_intersect / 3)

/-- The main theorem -/
theorem centroid_sum_reciprocal_squares (plane : IntersectingPlane)
    (h : plane.distance = 2) :
    let (p, q, r) := centroid plane
    1 / p^2 + 1 / q^2 + 1 / r^2 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_reciprocal_squares_l758_75874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l758_75869

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

-- Define the critical points
def x₁ : ℝ := -1
def x₂ : ℝ := 1

-- Define points A and B
noncomputable def A : ℝ × ℝ := (x₁, f x₁)
noncomputable def B : ℝ × ℝ := (x₂, f x₂)

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  let (m, n) := P
  (m - x₁)*(m - x₂) + (n - f x₁)*(n - f x₂) = 4

-- Define the reflection line
def reflection_line (x y : ℝ) : Prop := y = 2*(x - 4)

-- Define the reflection of P
noncomputable def Q (P : ℝ × ℝ) : ℝ × ℝ :=
  let (m, n) := P
  let x := (8*m + n + 16) / 5
  let y := (4*m + 2*n - 32) / 5
  (x, y)

theorem main_theorem :
  (A = (-1, 0) ∧ B = (1, 4)) ∧
  ∀ P : ℝ × ℝ, P_condition P →
    let (x, y) := Q P
    (x - 8)^2 + (y + 2)^2 = 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l758_75869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l758_75877

/-- Definition of the ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 + 12 * x + 9 * y^2 - 27 * y + 36 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 15 * Real.pi / 8

/-- The eccentricity of the ellipse -/
noncomputable def ellipse_eccentricity : ℝ := Real.sqrt 5 / 3

/-- Theorem stating the area and eccentricity of the given ellipse -/
theorem ellipse_properties :
  (∃ x y : ℝ, ellipse_equation x y) →
  (∀ x y : ℝ, ellipse_equation x y →
    (ellipse_area = 15 * Real.pi / 8 ∧
     ellipse_eccentricity = Real.sqrt 5 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l758_75877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l758_75816

theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b * c / Real.sqrt (a^2 + b^2) = Real.sqrt 5 / 3 * c) →
  c = Real.sqrt (a^2 + b^2) →
  c / a = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l758_75816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_inscribed_in_prism_iff_l758_75885

/-- A prism with a given height and cross-sectional polygon -/
structure Prism where
  height : ℝ
  cross_section : Set (ℝ × ℝ)

/-- A sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- A circle inscribed in a polygon -/
def circle_inscribed_in_polygon (p : Set (ℝ × ℝ)) : Prop :=
  ∃ (c : Set (ℝ × ℝ)), ∀ side : Set (ℝ × ℝ), side ⊆ p → (∃ point, point ∈ c ∩ side)

/-- The condition for a sphere to be inscribed in a prism -/
def sphere_inscribed_in_prism (s : Sphere) (p : Prism) : Prop :=
  p.height = 2 * s.radius ∧
  circle_inscribed_in_polygon p.cross_section

/-- Theorem stating the necessary and sufficient conditions for a sphere to be inscribed in a prism -/
theorem sphere_inscribed_in_prism_iff (s : Sphere) (p : Prism) :
  sphere_inscribed_in_prism s p ↔
    (p.height = 2 * s.radius ∧
     circle_inscribed_in_polygon p.cross_section) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_inscribed_in_prism_iff_l758_75885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_temp_difference_cooling_period_l758_75835

open Real

-- Define the temperature function
noncomputable def f (t : ℝ) : ℝ :=
  10 - (Real.sqrt 3) * (cos (π * t / 12)) - sin (π * t / 12)

-- Define the time domain
def time_domain : Set ℝ := { t | 0 ≤ t ∧ t < 24 }

-- Theorem for maximum temperature difference
theorem max_temp_difference :
  ∃ (t₁ t₂ : ℝ), t₁ ∈ time_domain ∧ t₂ ∈ time_domain ∧ 
    (∀ t ∈ time_domain, f t₁ ≤ f t ∧ f t ≤ f t₂) ∧
    f t₂ - f t₁ = 4 := by
  sorry

-- Theorem for cooling period
theorem cooling_period :
  ∀ t ∈ time_domain, f t > 11 ↔ 10 < t ∧ t < 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_temp_difference_cooling_period_l758_75835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_holds_l758_75898

-- Define the left-hand side of the equation
noncomputable def lhs : ℝ := Real.sqrt (1 + Real.sqrt (3 + Real.sqrt 49))

-- Define the right-hand side of the equation
noncomputable def rhs : ℝ := (1 + Real.sqrt 49) ^ (1/3)

-- Theorem statement
theorem equation_holds : lhs = rhs := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_holds_l758_75898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l758_75841

/-- Represents a statement about model fitting and regression analysis -/
inductive ModelStatement
| ResidualPlot
| CorrelationIndex
| SumSquaredResiduals

/-- Determines if a given statement is correct -/
def is_correct (statement : ModelStatement) : Bool :=
  match statement with
  | ModelStatement.ResidualPlot => false
  | ModelStatement.CorrelationIndex => true
  | ModelStatement.SumSquaredResiduals => true

/-- The list of all statements to be evaluated -/
def all_statements : List ModelStatement :=
  [ModelStatement.ResidualPlot, ModelStatement.CorrelationIndex, ModelStatement.SumSquaredResiduals]

/-- Counts the number of correct statements in the list -/
def count_correct (statements : List ModelStatement) : Nat :=
  statements.filter is_correct |>.length

/-- Theorem stating that the number of correct statements is 2 -/
theorem correct_statements_count :
  count_correct all_statements = 2 := by
  -- Proof goes here
  sorry

#eval count_correct all_statements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l758_75841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_people_circular_arrangements_l758_75868

/-- The number of people to be seated around the table -/
def n : ℕ := 12

/-- The number of distinct circular arrangements of n people around a round table,
    where rotations are considered the same -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem stating that the number of distinct circular arrangements
    of 12 people around a round table is equal to 11! -/
theorem twelve_people_circular_arrangements :
  circularArrangements n = 39916800 := by
  -- Unfold the definition of circularArrangements
  unfold circularArrangements
  -- Simplify the expression
  simp [n]
  -- Assert that 11! = 39916800
  have h : Nat.factorial 11 = 39916800 := rfl
  -- Apply the assertion
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_people_circular_arrangements_l758_75868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l758_75820

/-- Calculates the time (in seconds) for a train to pass a stationary point -/
noncomputable def time_to_pass (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

/-- Theorem stating that a train of length 375 meters, traveling at 120 km/h, 
    takes approximately 11.25 seconds to pass a stationary point -/
theorem train_passing_time :
  let train_length := (375 : ℝ)
  let train_speed := (120 : ℝ)
  let passing_time := time_to_pass train_length train_speed
  (passing_time ≥ 11.24) ∧ (passing_time ≤ 11.26) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l758_75820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oxygen_closest_to_48_percent_l758_75845

-- Define the molar masses of elements
noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_C : ℝ := 12.01
noncomputable def molar_mass_O : ℝ := 16.00

-- Define the number of atoms of each element in CaCO3
def num_Ca : ℕ := 1
def num_C : ℕ := 1
def num_O : ℕ := 3

-- Define the molar mass of CaCO3
noncomputable def molar_mass_CaCO3 : ℝ :=
  num_Ca * molar_mass_Ca + num_C * molar_mass_C + num_O * molar_mass_O

-- Define the mass percentages of each element
noncomputable def mass_percentage_Ca : ℝ := 100 * (num_Ca * molar_mass_Ca) / molar_mass_CaCO3
noncomputable def mass_percentage_C : ℝ := 100 * (num_C * molar_mass_C) / molar_mass_CaCO3
noncomputable def mass_percentage_O : ℝ := 100 * (num_O * molar_mass_O) / molar_mass_CaCO3

-- Theorem statement
theorem oxygen_closest_to_48_percent :
  |mass_percentage_O - 48| < |mass_percentage_Ca - 48| ∧
  |mass_percentage_O - 48| < |mass_percentage_C - 48| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oxygen_closest_to_48_percent_l758_75845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equation_b_range_l758_75807

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2 - 1 / 2)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), 1)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem f_equation (x : ℝ) : f x = Real.sin (x + π / 6) := by sorry

theorem b_range (A B : ℝ) (a b : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  A + B < π →
  f A = 1 →
  a = Real.sqrt 3 →
  a / Real.sin A = b / Real.sin B →
  0 < b ∧ b ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equation_b_range_l758_75807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_difference_magnitude_l758_75862

/-- Given two perpendicular vectors a and b, prove that the magnitude of their difference is 5 -/
theorem perpendicular_vectors_difference_magnitude :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![x + 1, 2]
  let b : Fin 2 → ℝ := ![1, -2]
  (Finset.sum (Finset.range 2) (λ i => a i * b i) = 0) →  -- perpendicular condition
  ‖(λ i => a i - b i)‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_difference_magnitude_l758_75862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l758_75837

noncomputable section

-- Define the original expression
def original_expression : ℝ := 1 / (Real.sqrt 3 - 1)

-- Define the simplified expression
def simplified_expression : ℝ := (Real.sqrt 3 + 1) / 2

-- Theorem stating the equality of the original and simplified expressions
theorem rationalize_denominator : original_expression = simplified_expression := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l758_75837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_shifted_point_l758_75805

/-- Given a > 0 and a ≠ 1, f is the inverse function of log_a, then f(x) + 2 passes through (0, 3) -/
theorem inverse_log_shifted_point (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f := Real.exp ∘ (· * Real.log a)
  (f 0 + 2 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_shifted_point_l758_75805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l758_75849

/-- Definition of an ellipse passing through given points with specific properties -/
def is_valid_ellipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ 
  a = 3 * b ∧
  (3 : ℝ)^2 / a^2 + 0^2 / b^2 = 1 ∧
  6 / a^2 + 1^2 / b^2 = 1 ∧
  3 / a^2 + 2 / b^2 = 1

/-- The theorem stating that the standard equation of the ellipse is x²/9 + y²/3 = 1 -/
theorem ellipse_equation : 
  ∃ (a b : ℝ), is_valid_ellipse a b ∧ a = 3 ∧ b = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l758_75849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_768_384_l758_75858

def sum_of_divisors (n : ℕ+) : ℕ := (Finset.sum (Nat.divisors n.val) id)

def f (n : ℕ+) : ℚ := (sum_of_divisors n : ℚ) / n.val

theorem f_difference_768_384 : f ⟨768, by norm_num⟩ - f ⟨384, by norm_num⟩ = 1 / 192 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_768_384_l758_75858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_curve_intersection_l758_75836

theorem sine_curve_intersection (A ω k : ℝ) (h_A : A > 0) (h_k : k > 0) :
  (∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ π / ω ∧
    A * Real.sin (2 * ω * x₁) + k = 4 ∧
    A * Real.sin (2 * ω * x₂) + k = 4) ∧
  (∃ x₃ x₄ : ℝ, 0 ≤ x₃ ∧ x₃ < x₄ ∧ x₄ ≤ π / ω ∧
    A * Real.sin (2 * ω * x₃) + k = -2 ∧
    A * Real.sin (2 * ω * x₄) + k = -2) ∧
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    A * Real.sin (2 * ω * x₁) + k = 4 ∧
    A * Real.sin (2 * ω * x₂) + k = 4 ∧
    A * Real.sin (2 * ω * x₃) + k = -2 ∧
    A * Real.sin (2 * ω * x₄) + k = -2 ∧
    (x₂ - x₁) = (x₄ - x₃) ∧
    x₂ - x₁ > 0) →
  k = 1 ∧ A > 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_curve_intersection_l758_75836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_exceeding_ten_dollars_l758_75800

/-- Represents the day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving Repr

/-- Calculates the day of the week given the number of days since the start (Monday) -/
def getDayOfWeek (n : Nat) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

/-- Calculates the total amount in cents after n days -/
def totalAmount (n : Nat) : Nat :=
  2 * (2^n - 1)

theorem first_day_exceeding_ten_dollars :
  ∃ (n : Nat), 
    (n ≤ 21) ∧ 
    (totalAmount n > 1000) ∧ 
    (∀ (m : Nat), m < n → totalAmount m ≤ 1000) ∧
    (getDayOfWeek n = DayOfWeek.Tuesday) := by
  sorry

#eval getDayOfWeek 9
#eval totalAmount 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_exceeding_ten_dollars_l758_75800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_formation_l758_75864

-- Define the complex numbers z₁, ..., zₙ as a geometric sequence
noncomputable def geometric_sequence (z₁ : ℂ) (q : ℂ) (n : ℕ) : Fin n → ℂ :=
  fun k ↦ z₁ * q ^ (k.val - 1)

-- Define the function to calculate wₖ
noncomputable def w (z : ℂ) (h : ℝ) : ℂ := z + 1 / z + h

/-- Theorem stating that the points form an ellipse -/
theorem ellipse_formation
  (z₁ : ℂ) (q : ℂ) (n : ℕ) (h : ℝ)
  (hz₁ : Complex.abs z₁ ≠ 1)
  (hq : q ≠ 1 ∧ q ≠ -1) :
  ∃ (a b : ℝ),
    ∀ k : Fin n,
      let z := geometric_sequence z₁ q n k
      let w := w z h
      ((w.re - h) / a)^2 + (w.im / b)^2 = 1 ∧
      a^2 - b^2 = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_formation_l758_75864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clermontville_viewers_l758_75889

theorem clermontville_viewers (total_residents : ℕ) 
  (island_survival_pct : ℚ) (lovelost_lawyers_pct : ℚ) 
  (medical_emergency_pct : ℚ) (mystery_minders_pct : ℚ)
  (exactly_two_shows_pct : ℚ) (exactly_three_shows_pct : ℚ) :
  total_residents = 800 →
  island_survival_pct = 30 / 100 →
  lovelost_lawyers_pct = 35 / 100 →
  medical_emergency_pct = 45 / 100 →
  mystery_minders_pct = 25 / 100 →
  exactly_two_shows_pct = 22 / 100 →
  exactly_three_shows_pct = 8 / 100 →
  ∃ (all_four_shows : ℕ),
    all_four_shows = 168 ∧
    (island_survival_pct * ↑total_residents).floor +
    (lovelost_lawyers_pct * ↑total_residents).floor +
    (medical_emergency_pct * ↑total_residents).floor +
    (mystery_minders_pct * ↑total_residents).floor -
    (exactly_two_shows_pct * ↑total_residents).floor +
    (exactly_three_shows_pct * ↑total_residents).floor -
    all_four_shows = total_residents :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clermontville_viewers_l758_75889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_estimation_for_22cm_l758_75823

/-- Represents the linear regression model for student height estimation -/
structure HeightEstimation where
  n : ℕ  -- number of students
  x_sum : ℝ  -- sum of x values
  y_sum : ℝ  -- sum of y values
  slope : ℝ  -- slope of regression line

/-- Calculates the estimated height for a given hand span -/
noncomputable def estimate_height (model : HeightEstimation) (x : ℝ) : ℝ :=
  let x_mean : ℝ := model.x_sum / model.n
  let y_mean : ℝ := model.y_sum / model.n
  let intercept : ℝ := y_mean - model.slope * x_mean
  model.slope * x + intercept

/-- Theorem stating that the estimated height for a 22 cm hand span is 183 cm -/
theorem height_estimation_for_22cm (model : HeightEstimation) 
  (h1 : model.n = 12)
  (h2 : model.x_sum = 240)
  (h3 : model.y_sum = 2040)
  (h4 : model.slope = 6.5) :
  estimate_height model 22 = 183 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_estimation_for_22cm_l758_75823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_constant_impossible_to_equalize_l758_75886

/-- Represents the numbers in 6 sectors of a circle -/
def Sectors := Fin 6 → ℤ

/-- The sum of numbers in odd-indexed sectors -/
def odd_sum (s : Sectors) : ℤ := s 0 + s 2 + s 4

/-- The sum of numbers in even-indexed sectors -/
def even_sum (s : Sectors) : ℤ := s 1 + s 3 + s 5

/-- The difference between even and odd sums -/
def difference (s : Sectors) : ℤ := even_sum s - odd_sum s

/-- Represents the operation of adding 1 to two adjacent sectors -/
def add_to_adjacent (s : Sectors) (i : Fin 6) : Sectors :=
  fun j => if j = i ∨ j = (i + 1) % 6 then s j + 1 else s j

/-- Theorem stating that the difference remains constant after any operation -/
theorem difference_constant (s : Sectors) (i : Fin 6) :
  difference (add_to_adjacent s i) = difference s := by
  sorry

/-- Theorem stating that it's impossible to make all numbers equal unless they start equal -/
theorem impossible_to_equalize (s : Sectors) :
  (∃ (a b : Fin 6), s a ≠ s b) →
  ¬∃ (n : ℕ) (is : List (Fin 6)), (difference (is.foldl add_to_adjacent s) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_constant_impossible_to_equalize_l758_75886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_correct_l758_75822

/-- The smallest real solution to the equation 3x/(x-3) + (3x^2 - 27)/x = 18 -/
noncomputable def smallest_solution : ℝ := (15 - Real.sqrt 549) / 6

/-- The equation in question -/
def equation (x : ℝ) : Prop := 3 * x / (x - 3) + (3 * x^2 - 27) / x = 18

theorem smallest_solution_correct :
  equation smallest_solution ∧
  ∀ y : ℝ, equation y → y ≥ smallest_solution := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_correct_l758_75822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_second_job_pay_difference_l758_75878

/-- Calculates the percentage difference between two hourly wages -/
noncomputable def percentage_difference (main_wage second_wage : ℝ) : ℝ :=
  (main_wage - second_wage) / main_wage * 100

theorem james_second_job_pay_difference 
  (main_hourly_wage : ℝ) 
  (main_hours : ℝ) 
  (total_earnings : ℝ) :
  main_hourly_wage = 20 →
  main_hours = 30 →
  total_earnings = 840 →
  percentage_difference main_hourly_wage 
    ((total_earnings - main_hourly_wage * main_hours) / (main_hours / 2)) = 20 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_second_job_pay_difference_l758_75878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l758_75897

-- Define the triangle ABC
variable (A B C : ℝ)
variable (a b c : ℝ)

-- Define vectors p and q
noncomputable def p : ℝ × ℝ := (1, -Real.sqrt 3)
noncomputable def q (B : ℝ) : ℝ × ℝ := (Real.cos B, Real.sin B)

-- State the given conditions
axiom parallel_vectors (B : ℝ) : p.1 * (q B).2 = p.2 * (q B).1
axiom angle_relation : b * Real.cos C + c * Real.cos B = 2 * a * Real.sin A

-- State the theorem to be proved
theorem angle_C_measure : C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l758_75897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_NY_to_SF_l758_75854

/-- Time taken to travel from New York to San Francisco -/
noncomputable def time_NY_to_SF : ℝ := sorry

/-- Time taken to travel from New Orleans to New York -/
noncomputable def time_NO_to_NY : ℝ := (3/4) * time_NY_to_SF

/-- Layover time in New York -/
def layover_time : ℝ := 16

/-- Total travel time from New Orleans to San Francisco -/
def total_time : ℝ := 58

theorem travel_time_NY_to_SF :
  time_NO_to_NY + layover_time + time_NY_to_SF = total_time →
  time_NY_to_SF = 24 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_NY_to_SF_l758_75854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suit_price_calculation_l758_75808

/-- Given a suit with an original price, calculate the final price after a price increase and discount. -/
theorem suit_price_calculation (original_price increase_rate discount_rate : ℝ) : 
  original_price = 200 → 
  increase_rate = 0.25 → 
  discount_rate = 0.25 → 
  (original_price * (1 + increase_rate) * (1 - discount_rate)) = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suit_price_calculation_l758_75808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l758_75806

theorem largest_number (a b c d : ℝ) (h1 : a = 2/3) (h2 : b = 1) (h3 : c = -3) (h4 : d = 0) :
  max (max (max a b) c) d = b :=
by
  rw [h1, h2, h3, h4]
  simp [max_def]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l758_75806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payroll_tax_theorem_l758_75801

/-- Calculates the special municipal payroll tax -/
noncomputable def calculate_tax (payroll : ℝ) : ℝ :=
  if payroll ≤ 200000 then 0
  else 0.002 * (payroll - 200000)

/-- Theorem: A payroll of $400,000 results in a tax payment of $400 -/
theorem payroll_tax_theorem :
  calculate_tax 400000 = 400 := by
  -- Unfold the definition of calculate_tax
  unfold calculate_tax
  -- Simplify the if-then-else expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_payroll_tax_theorem_l758_75801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l758_75872

/-- Given a train with length and time to cross a point, calculate its speed -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: A train 640 meters long crossing a point in 16 seconds has a speed of 40 m/s -/
theorem train_speed_calculation :
  train_speed 640 16 = 40 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l758_75872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_integers_l758_75829

theorem count_valid_integers : 
  (Finset.filter (fun n : ℕ => 
    100 ≤ n / 4 ∧ n / 4 ≤ 999 ∧ 
    1000 ≤ 4 * n ∧ 4 * n ≤ 9999 ∧
    n % 4 = 0) 
    (Finset.range 2500)).card = 525 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_integers_l758_75829
