import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_for_max_value_4_l185_18594

theorem exists_a_for_max_value_4 : ∃ a : ℝ, 
  (∀ x : ℝ, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → Real.cos x ^ 2 + 2 * a * Real.sin x + 3 * a - 1 ≤ 4) ∧ 
  (∃ x : ℝ, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) ∧ Real.cos x ^ 2 + 2 * a * Real.sin x + 3 * a - 1 = 4) ∧
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_for_max_value_4_l185_18594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_isosceles_points_l185_18593

-- Define the parabola
def Parabola : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (1, 0)

-- Define the origin
def Origin : ℝ × ℝ := (0, 0)

-- Define an isosceles triangle
def IsIsosceles (a b c : ℝ × ℝ) : Prop :=
  (dist a b = dist a c) ∨ (dist b a = dist b c)

-- Main theorem
theorem parabola_isosceles_points :
  ∃ (S : Finset (ℝ × ℝ)), S.card = 4 ∧ 
  (∀ p ∈ S, p ∈ Parabola ∧ IsIsosceles Origin Focus p) ∧
  (∀ p ∈ Parabola, IsIsosceles Origin Focus p → p ∈ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_isosceles_points_l185_18593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_condition_l185_18532

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def areCollinear (v w : V) : Prop := ∃ (k : ℝ), v = k • w

theorem collinearity_condition
  (a b : V) -- Vectors a and b
  (h_not_collinear : ¬ areCollinear a b) -- a and b are not collinear
  (m n : ℝ) -- Real numbers m and n
  (AB AC : V) -- Vectors AB and AC
  (h_AB : AB = a + m • b) -- Definition of AB
  (h_AC : AC = n • a + b) : -- Definition of AC
  areCollinear AB AC ↔ m * n - 1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_condition_l185_18532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_expression_l185_18567

theorem simplify_sqrt_expression : Real.sqrt (73 - 40 * Real.sqrt 3) = 5 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_expression_l185_18567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l185_18505

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (16 - x^2 + 6*x)
def g (x : ℝ) : ℝ := |x - 3|

-- Define the solution set
def solution_set : Set ℝ := Set.union (Set.Ioo (-1) 1) (Set.Ioc (11/3) 8)

-- Theorem statement
theorem problem_solution :
  ∀ x : ℝ, (16 - x^2 + 6*x ≥ 0 ∧ min (f x) (g x) > (5 - x) / 2) → x ∈ solution_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l185_18505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_three_digit_number_to_sum_of_digits_l185_18562

theorem max_ratio_three_digit_number_to_sum_of_digits :
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 →
    0 ≤ b ∧ b ≤ 9 →
    0 ≤ c ∧ c ≤ 9 →
    (100 * a + 10 * b + c : ℚ) / (a + b + c) ≤ 100 ∧
    ∃ (a' : ℕ), 1 ≤ a' ∧ a' ≤ 9 ∧ (100 * a' : ℚ) / a' = 100 :=
by
  sorry

#check max_ratio_three_digit_number_to_sum_of_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_three_digit_number_to_sum_of_digits_l185_18562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l185_18521

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infiniteGeometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: For an infinite geometric series with common ratio 1/4 and sum 80, the first term is 60 -/
theorem first_term_of_geometric_series (sum r a : ℝ)
  (h_sum : sum = 80)
  (h_r : r = 1/4)
  (h_series : sum = infiniteGeometricSeriesSum a r) :
  a = 60 := by
  sorry

#check first_term_of_geometric_series

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l185_18521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l185_18595

theorem log_inequality (a x : ℝ) :
  (a > 1) →
  (Real.log (2*x - 1) > Real.log (x - 1) ↔ a > 2 ∧ x > 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l185_18595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_properties_l185_18501

-- Define N as a natural number
def N : ℕ := sorry

-- State the theorem about N's properties
theorem N_properties : Odd N ∧ ¬(3 ∣ N) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_properties_l185_18501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_I_l185_18569

open Real Set

/-- The function f(x) = x + 2cos(x) -/
noncomputable def f (x : ℝ) : ℝ := x + 2 * cos x

/-- The interval [0, π] -/
def I : Set ℝ := Icc 0 π

theorem min_value_f_on_I :
  ∃ (x : ℝ), x ∈ I ∧ f x = (5 * π / 6 - Real.sqrt 3) ∧ ∀ (y : ℝ), y ∈ I → f y ≥ f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_I_l185_18569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_multiplication_puzzle_l185_18574

theorem vertical_multiplication_puzzle :
  ∃ (multiplicand : ℕ) (multiplier : ℕ),
    multiplicand * multiplier = 2754 ∧
    multiplicand ≥ 100 ∧ multiplicand < 1000 ∧
    multiplier ≥ 10 ∧ multiplier < 100 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_multiplication_puzzle_l185_18574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l185_18512

def a : Fin 2 → ℝ
  | 0 => 1
  | 1 => 1

def b : Fin 2 → ℝ
  | 0 => -3
  | 1 => 4

theorem vector_properties :
  (∃ (cos_angle : ℝ), cos_angle = (a 0 * b 0 + a 1 * b 1) / (Real.sqrt ((a 0)^2 + (a 1)^2) * Real.sqrt ((b 0)^2 + (b 1)^2)) ∧
                       cos_angle = Real.sqrt 2 / 10) ∧
  (∀ (l : ℝ), (∃ (μ : ℝ), (a 0 - l * b 0, a 1 - l * b 1) = μ • (a 0 + l * b 0, a 1 + l * b 1)) → l = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l185_18512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_light_intersection_l185_18518

/-- The x-coordinate of the particle's position at time t -/
def x (t : ℝ) : ℝ := 3

/-- The y-coordinate of the particle's position at time t -/
noncomputable def y (t : ℝ) : ℝ := 3 + Real.sin t * Real.cos t - Real.sin t - Real.cos t

/-- The equation of the light ray -/
def lightRay (c : ℝ) (x : ℝ) : ℝ := c * x

theorem particle_light_intersection (c : ℝ) :
  (c > 0) →
  (∀ t : ℝ, y t ≠ lightRay c (x t)) ↔ 
  (c ∈ Set.Ioo 0 (3/2) ∪ Set.Ioi ((7 + 2 * Real.sqrt 2) / 6)) :=
by
  sorry

#check particle_light_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_light_intersection_l185_18518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_equation_line_n_equation_l185_18557

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the distance from a point to a line -/
noncomputable def Point.distance_to_line (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- Given point P -/
def P : Point := ⟨2, -1⟩

/-- Theorem for line m -/
theorem line_m_equation : 
  ∃ (l : Line), P.on_line l ∧ 
  ((l.a = -1/2 ∧ l.b = 1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -1)) := by
  sorry

/-- Theorem for line n -/
theorem line_n_equation :
  ∃ (l : Line), P.on_line l ∧ origin.distance_to_line l = 2 ∧
  ((l.a = 1 ∧ l.b = 0 ∧ l.c = -2) ∨ (l.a = 3 ∧ l.b = -4 ∧ l.c = -10)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_m_equation_line_n_equation_l185_18557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l185_18558

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (a b : V)

-- Define the condition that a and b are not collinear
variable (h_not_collinear : ∀ (r : ℝ), a ≠ r • b)

-- Define m and n
def m (a b : V) : V := 2 • a - 3 • b
def n (a b : V) (k : ℝ) : V := 3 • a + k • b

-- State the theorem
theorem parallel_vectors (a b : V) (h_not_collinear : ∀ (r : ℝ), a ≠ r • b) :
  ∃! k : ℝ, ∃ (l : ℝ), n a b k = l • (m a b) ∧ l ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l185_18558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l185_18531

-- Define the sets M and N
def M : Set ℝ := {x | (x - 3) * (x + 1) ≥ 0}
def N : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Icc (-2) (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l185_18531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l185_18548

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ

noncomputable def distanceBetweenPoints (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def isPointOnParabola (p : Point) (parab : Parabola) : Prop :=
  p.y^2 = 6 * p.x

def isPointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

def areLinesPerp (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem parabola_focus_distance (parab : Parabola) (F P A : Point) (l directrix AF : Line) : 
  isPointOnParabola P parab →
  isPointOnLine A directrix →
  isPointOnLine A AF →
  isPointOnLine F AF →
  areLinesPerp (Line.mk 0 A.y) directrix →
  AF.slope = -Real.sqrt 3 →
  F.x = 1.5 →
  F.y = 0 →
  directrix.slope = 0 →
  directrix.yIntercept = -1.5 →
  distanceBetweenPoints P F = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l185_18548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l185_18570

-- Define the circle E in polar coordinates
noncomputable def circle_E (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := y = Real.tan (3 * Real.pi / 4) * x

-- Define point A as the intersection of l and E
noncomputable def point_A : ℝ × ℝ := (2 * Real.sqrt 2, 3 * Real.pi / 4)

-- Define point M as the midpoint of OA
noncomputable def point_M : ℝ × ℝ := (Real.sqrt 2, 3 * Real.pi / 4)

-- State the theorem
theorem max_distance_difference : 
  ∃ (B C : ℝ × ℝ), 
    (∀ B' C' : ℝ × ℝ, 
      circle_E B'.1 B'.2 → 
      circle_E C'.1 C'.2 → 
      (∃ t : ℝ, B' = (point_M.1 + t, point_M.2 + t) ∧ 
                C' = (point_M.1 - t, point_M.2 - t)) →
      |point_M.1 - B.1| - |point_M.1 - C.1| ≥ |point_M.1 - B'.1| - |point_M.1 - C'.1|) ∧
    |point_M.1 - B.1| - |point_M.1 - C.1| = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l185_18570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_in_cone_l185_18582

/-- A right circular cone with a sphere inside --/
structure ConeWithSphere where
  base_diameter : ℝ
  is_right_circular : Prop
  cross_section_is_isosceles_right : Prop
  sphere_tangent_to_sides : Prop
  sphere_rests_on_base : Prop

/-- The volume of a sphere --/
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4/3) * Real.pi * radius^3

/-- Theorem stating the volume of the sphere inside the cone --/
theorem sphere_volume_in_cone (cone : ConeWithSphere) 
  (h : cone.base_diameter = 12) : 
  ∃ (radius : ℝ), sphere_volume radius = 288 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_in_cone_l185_18582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_probability_l185_18577

/-- The set of integers from 1 to 30 inclusive -/
def S : Finset ℕ := Finset.filter (fun n => 1 ≤ n ∧ n ≤ 30) (Finset.range 31)

/-- The set of prime numbers in S -/
def P : Finset ℕ := Finset.filter Nat.Prime S

/-- The number of elements in S -/
def s : ℕ := S.card

/-- The number of elements in P -/
def p : ℕ := P.card

/-- The probability of selecting two different prime numbers from S -/
def prob : ℚ := (p.choose 2 : ℚ) / (s.choose 2 : ℚ)

theorem prime_probability : prob = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_probability_l185_18577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l185_18500

/-- A circle with center on the line y = -4x and tangent to x + y - 1 = 0 at (3, -2) -/
structure TangentCircle where
  /-- The x-coordinate of the circle's center -/
  center_x : ℝ
  /-- The y-coordinate of the circle's center -/
  center_y : ℝ
  /-- The y-coordinate of the circle's center satisfies y = -4x -/
  center_y_eq : center_y = -4 * center_x
  /-- The circle is tangent to the line x + y - 1 = 0 at the point (3, -2) -/
  tangent_at_point : (3 : ℝ) + (-2 : ℝ) - 1 = 0

/-- The standard equation of the circle is (x - 1)² + (y + 4)² = 8 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∀ x y : ℝ, (x - 1)^2 + (y + 4)^2 = 8 ↔ 
    (x - c.center_x)^2 + (y - c.center_y)^2 = ((3 - c.center_x)^2 + (-2 - c.center_y)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l185_18500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l185_18527

theorem binomial_expansion_coefficient (a b : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x, (1 + a*x)^5 = 1 + 10*x + b*x^2 + f x) → b = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l185_18527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parabola_intersection_l185_18539

noncomputable def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y - k = 0

def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 4 * x

noncomputable def point_A (k : ℝ) : ℝ × ℝ :=
  (2, 2 * Real.sqrt 2)

noncomputable def point_B (k : ℝ) : ℝ × ℝ :=
  sorry

noncomputable def point_M (k : ℝ) : ℝ × ℝ :=
  sorry

def point_O : ℝ × ℝ :=
  (0, 0)

noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry

theorem line_parabola_intersection (k : ℝ) (h1 : k > 0) :
  line_equation k (point_A k).1 (point_A k).2 ∧
  parabola_equation (point_A k).1 (point_A k).2 ∧
  line_equation k (point_B k).1 (point_B k).2 ∧
  parabola_equation (point_B k).1 (point_B k).2 ∧
  (point_M k).1 = -1 ∧
  (point_M k).2 = (point_B k).2 ∧
  triangle_area point_O (point_B k) (point_M k) / triangle_area point_O (point_B k) (point_A k) = 1 / 2 →
  k = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parabola_intersection_l185_18539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l185_18586

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^3 else z^3

-- State the theorem
theorem f_composition_result :
  f (f (f (f (1 + Complex.I)))) = -147197952000 + 49152 * Complex.I :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l185_18586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_implies_a_value_l185_18538

theorem binomial_coefficient_implies_a_value (a : ℝ) :
  (∃ c : ℝ, c = 84 ∧ c = (2^2 * (-a)^5 * (Nat.choose 7 5))) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_implies_a_value_l185_18538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_needle_end_swap_impossible_l185_18599

-- Define a point in the plane
structure Point where
  x : ℚ × ℚ  -- Represents a + b√2
  y : ℚ × ℚ  -- Represents c + d√2

-- Define a needle
structure Needle where
  start : Point
  end_ : Point  -- Changed from 'end' to 'end_' to avoid reserved keyword

-- Define the weight of a point
def weight (p : Point) : ℚ :=
  p.x.1 + 2 * p.x.2 + p.y.1

-- Define a rotation operation
def rotate (n : Needle) (around_start : Bool) : Needle :=
  sorry  -- Actual implementation would go here

-- Theorem: It's impossible to swap the ends of the needle
theorem needle_end_swap_impossible (n : Needle) : 
  ∀ (rotations : List Bool), 
    let final_needle := rotations.foldl (λ acc b => rotate acc b) n
    final_needle.start ≠ n.end_ ∨ final_needle.end_ ≠ n.start := by
  sorry  -- Proof would go here

#check needle_end_swap_impossible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_needle_end_swap_impossible_l185_18599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_correct_l185_18520

/-- Represents the pricing strategy and profit of a retailer --/
structure RetailPricing where
  markup_percentage : ℚ
  actual_profit_percentage : ℚ
  discount_percentage : ℚ

/-- Calculates the discount percentage given markup and actual profit --/
def calculate_discount (pricing : RetailPricing) : ℚ :=
  (pricing.markup_percentage - pricing.actual_profit_percentage) / (1 + pricing.markup_percentage)

/-- Theorem stating that the calculated discount matches the given discount --/
theorem discount_calculation_correct (pricing : RetailPricing) 
  (h_markup : pricing.markup_percentage = 65/100)
  (h_profit : pricing.actual_profit_percentage = 2375/10000)
  (h_discount : pricing.discount_percentage = 1/4) :
  calculate_discount pricing = pricing.discount_percentage := by
  sorry

def main : IO Unit := do
  let pricing : RetailPricing := {
    markup_percentage := 65/100,
    actual_profit_percentage := 2375/10000,
    discount_percentage := 1/4
  }
  IO.println s!"Calculated discount: {calculate_discount pricing}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_correct_l185_18520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l185_18550

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_sum_of_factors (x y z w : ℕ+) (h : x * y * z * w = factorial 8) :
  x + y + z + w ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l185_18550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_three_digit_with_property_l185_18530

/-- A function that changes a digit in a given number at a specified position by a given amount. -/
def changeDigit (n : ℕ) (pos : ℕ) (change : ℤ) : ℕ := sorry

/-- A function that checks if a number has the property that changing any of its digits by 1 
    (either increasing or decreasing) results in a number divisible by 11. -/
def hasProperty (n : ℕ) : Prop :=
  ∀ (pos : ℕ) (change : ℤ), change.natAbs = 1 → 
    11 ∣ (changeDigit n pos change)

/-- Theorem stating that 120 is the smallest three-digit number with the specified property. -/
theorem smallest_three_digit_with_property :
  (120 ≥ 100) ∧ (hasProperty 120) ∧ (∀ n : ℕ, 100 ≤ n ∧ n < 120 → ¬(hasProperty n)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_three_digit_with_property_l185_18530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_average_age_l185_18535

/-- The average age of two women given specific conditions -/
theorem women_average_age (initial_men : ℕ) (replaced_men_age1 replaced_men_age2 : ℕ) 
  (age_increase : ℝ) : ℝ :=
  let initial_men := 6
  let replaced_men_age1 := 24
  let replaced_men_age2 := 26
  let age_increase := 3
  -- The average age of the two women
  let women_avg_age := 34

  by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_average_age_l185_18535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_collinearity_l185_18576

-- Define the tetrahedron and points
variable (O A B C M N G : EuclideanSpace ℝ (Fin 3))

-- Define the conditions
variable (h1 : ∃ t : ℝ, M = O + t • (A - O) ∧ t > 0 ∧ t < 1)
variable (h2 : M - O = 2 • (A - M))
variable (h3 : N = (B + C) / 2)

-- Define the vector OG
variable (x : ℝ)
variable (h4 : G - O = (1/3) • (A - O) + (x/4) • (B - O) + (x/4) • (C - O))

-- Define collinearity
def collinear (P Q R : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∃ t : ℝ, R - P = t • (Q - P)

-- State the theorem
theorem tetrahedron_collinearity :
  collinear M N G ↔ x = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_collinearity_l185_18576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tethered_dog_area_l185_18598

/-- The area outside a regular hexagon that can be reached by a point tethered to a vertex -/
def area_outside_hexagon (side_length : ℝ) (rope_length : ℝ) : ℝ :=
  -- Definition would involve calculating the area of sectors
  -- and subtracting the area of the hexagon
  sorry

/-- The area outside a regular hexagon reachable by a tethered point -/
theorem tethered_dog_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 → rope_length = 2 → 
  area_outside_hexagon side_length rope_length = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tethered_dog_area_l185_18598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_area_increase_percentage_l185_18583

-- Define the original and new diameters
def original_diameter : ℝ := 8
def new_diameter : ℝ := 10

-- Define a function to calculate the area of a circular cake given its diameter
noncomputable def cake_area (diameter : ℝ) : ℝ := Real.pi * (diameter / 2) ^ 2

-- Theorem statement
theorem cake_area_increase_percentage :
  (cake_area new_diameter - cake_area original_diameter) / cake_area original_diameter * 100 = 56.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_area_increase_percentage_l185_18583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l185_18596

theorem equation_solution : 
  ∃! x : ℝ, (5 : ℝ)^x * (25 : ℝ)^(2*x) = (125 : ℝ)^6 ∧ x = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l185_18596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_time_is_9_l185_18542

/-- The time it takes P to finish the job alone (in hours) -/
noncomputable def p_time : ℝ := 3

/-- The time P and Q work together (in hours) -/
noncomputable def together_time : ℝ := 2

/-- The additional time P works alone after working together (in hours) -/
noncomputable def p_additional_time : ℝ := 1/3

/-- The time it takes Q to finish the job alone (in hours) -/
noncomputable def q_time : ℝ := 9

/-- The fraction of the job completed when P and Q work together -/
noncomputable def together_work (q_time : ℝ) : ℝ := (1/p_time + 1/q_time) * together_time

/-- The fraction of the job P completes in the additional time -/
noncomputable def p_additional_work : ℝ := 1/p_time * p_additional_time

theorem q_time_is_9 : 
  together_work q_time + p_additional_work = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_time_is_9_l185_18542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_less_of_even_increasing_function_l185_18552

def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem abs_less_of_even_increasing_function
  (f : ℝ → ℝ) (a b : ℝ)
  (heven : EvenFunction f)
  (hincr : IncreasingOn f (Set.Ici 0))
  (hlt : f a < f b) :
  |a| < |b| := by
  sorry

#check abs_less_of_even_increasing_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_less_of_even_increasing_function_l185_18552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_sin_equation_l185_18536

/-- The function representing sin x after transformations -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

/-- The original sine function -/
noncomputable def original_sin (x : ℝ) : ℝ := Real.sin x

theorem transformed_sin_equation (x : ℝ) :
  f (x + Real.pi / 6) = original_sin (2 * x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_sin_equation_l185_18536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_exponential_sum_l185_18506

theorem max_value_exponential_sum (a b : ℝ) (h : a + b = 3) :
  (∀ x y : ℝ, x + y = 3 → (2:ℝ)^x + (2:ℝ)^y ≤ (2:ℝ)^a + (2:ℝ)^b) → (2:ℝ)^a + (2:ℝ)^b = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_exponential_sum_l185_18506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l185_18560

/-- Triangle ABC with sides a, b, c opposite angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : ∃ (k : ℝ), (k * t.a, k * Real.sqrt 3 * t.b) = (Real.cos t.A, Real.sin t.B))
  (h2 : t.a = Real.sqrt 7)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2) :
  t.A = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l185_18560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_optimal_cost_l185_18588

/-- Represents the specifications and cost function of a rectangular water tank. -/
structure WaterTank where
  volume : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the total cost of the water tank given its length. -/
noncomputable def totalCost (tank : WaterTank) (length : ℝ) : ℝ :=
  let width := tank.volume / (tank.depth * length)
  let wallArea := 2 * (length + width) * tank.depth
  let bottomArea := length * width
  wallArea * tank.wallCost + bottomArea * tank.bottomCost

/-- Theorem stating the minimum cost and optimal dimensions of the water tank. -/
theorem water_tank_optimal_cost (tank : WaterTank) 
  (h_volume : tank.volume = 4800)
  (h_depth : tank.depth = 3)
  (h_bottomCost : tank.bottomCost = 150)
  (h_wallCost : tank.wallCost = 120) :
  (∃ (x : ℝ), x > 0 ∧ totalCost tank x = 297600 ∧ 
    ∀ (y : ℝ), y > 0 → totalCost tank y ≥ 297600) ∧
  (∃ (x : ℝ), x = 40 ∧ totalCost tank x = 297600) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_optimal_cost_l185_18588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l185_18519

def arithmetic_progression (start : ℕ) (diff : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => start + i * diff)

theorem product_remainder (n : ℕ) :
  let nums := arithmetic_progression 3 10 n
  (nums.prod % 7 = 4) ∧ (nums.getLast? = some 93) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l185_18519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_zeros_in_Q_l185_18522

/-- Definition of R_k -/
def R (k : ℕ) : ℚ := (10^k - 1) / 9

/-- Definition of Q -/
def Q : ℚ := R 18 / R 6

/-- Function to count zeros in the decimal representation of a rational number -/
noncomputable def countZeros (q : ℚ) : ℕ := sorry

/-- Theorem stating that Q has 10 zeros in its decimal representation -/
theorem count_zeros_in_Q : countZeros Q = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_zeros_in_Q_l185_18522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_side_returns_with_triangle_l185_18529

-- Define an equilateral triangle
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_equilateral : ∀ i j : Fin 3, i ≠ j → 
    dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

-- Define a reflection of a triangle about one of its sides
def reflect (t : EquilateralTriangle) (side : Fin 3) : EquilateralTriangle := sorry

-- Define a sequence of reflections
def reflection_sequence : EquilateralTriangle → List (Fin 3) → EquilateralTriangle
  | t, [] => t
  | t, (s::ss) => reflect (reflection_sequence t ss) s

-- Define what it means for a triangle to return to its original position
def returns_to_original (t : EquilateralTriangle) (seq : List (Fin 3)) : Prop :=
  reflection_sequence t seq = t

-- Define what it means for the marked side to return to its original position
def marked_side_returns (t : EquilateralTriangle) (marked : Fin 3) (seq : List (Fin 3)) : Prop :=
  ∃ (perm : Fin 3 ≃ Fin 3), 
    (reflection_sequence t seq).vertices ∘ perm = t.vertices ∧ perm marked = marked

-- The theorem to prove
theorem marked_side_returns_with_triangle 
  (t : EquilateralTriangle) (marked : Fin 3) (seq : List (Fin 3)) :
  returns_to_original t seq → marked_side_returns t marked seq := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_side_returns_with_triangle_l185_18529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_angles_l185_18528

theorem sin_sum_of_angles (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.sin α = 5 / 13) (h4 : Real.cos β = 4 / 5) : Real.sin (α + β) = 56 / 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_angles_l185_18528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_income_l185_18592

/-- Represents the net income function in thousands of dollars -/
noncomputable def netIncome (y : ℝ) : ℝ := y - (y^2 / 100)

/-- Proves that the income maximizing net income is 50 thousand dollars -/
theorem max_net_income :
  ∃ (y : ℝ), ∀ (x : ℝ), netIncome y ≥ netIncome x ∧ y = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_income_l185_18592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_meeting_times_l185_18546

/-- The time for two vehicles to meet when traveling towards each other -/
noncomputable def time_to_meet (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  distance / (speed1 + speed2)

/-- The time for a faster vehicle to catch up to a slower vehicle -/
noncomputable def time_to_catch_up (distance : ℝ) (speed_fast : ℝ) (speed_slow : ℝ) : ℝ :=
  distance / (speed_fast - speed_slow)

theorem vehicle_meeting_times
  (distance : ℝ) (speed_motorcycle : ℝ) (speed_truck : ℝ)
  (h_distance : distance = 40)
  (h_speed_motorcycle : speed_motorcycle = 45)
  (h_speed_truck : speed_truck = 35) :
  time_to_meet distance speed_motorcycle speed_truck = 0.5 ∧
  time_to_catch_up distance speed_motorcycle speed_truck = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_meeting_times_l185_18546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_section_largest_area_l185_18591

/-- Represents a cube in 3D space -/
structure Cube where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a plane section of a cube -/
structure PlaneSection where
  cube : Cube
  area : ℝ

/-- Represents a diagonal section of a cube -/
noncomputable def DiagonalSection (c : Cube) : PlaneSection :=
  { cube := c
    area := c.side_length^2 * Real.sqrt 2 / 2 }

/-- Theorem stating that diagonal sections have the largest area -/
theorem diagonal_section_largest_area (c : Cube) (s : PlaneSection) 
    (h : s.cube = c) : s.area ≤ (DiagonalSection c).area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_section_largest_area_l185_18591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l185_18503

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x - 1)

theorem f_range :
  Set.range f = {y : ℝ | y < -1 ∨ y > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l185_18503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_sum_l185_18585

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  root_property : a 5 ^ 2 - 2 * a 5 - 6 = 0 ∧ a 7 ^ 2 - 2 * a 7 - 6 = 0

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_of_terms (seq : SpecialArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- Theorem: The sum of the first 11 terms of the special arithmetic sequence is 11 -/
theorem special_sequence_sum (seq : SpecialArithmeticSequence) : 
  sum_of_terms seq 11 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_sum_l185_18585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_path_length_l185_18580

-- Define the triangle and square
def triangle_side_length : ℚ := 3
def square_side_length : ℚ := 6

-- Define the number of rotations
def rotations_per_side : ℕ := 3
def number_of_sides : ℕ := 4

-- Define the path length for a single rotation
noncomputable def single_rotation_path_length : ℝ := 2 * Real.pi

-- Theorem statement
theorem total_path_length :
  (↑(rotations_per_side * number_of_sides) : ℝ) * single_rotation_path_length = 24 * Real.pi :=
by
  -- Proof steps would go here
  sorry

#eval rotations_per_side * number_of_sides

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_path_length_l185_18580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_T_is_42_l185_18572

def prime_digits : List ℕ := [2, 3, 5, 7]

noncomputable def T (m : ℕ) : ℚ :=
  let sum_reciprocals := (prime_digits.map (λ p => (1 : ℚ) / p)).sum
  m * (5^(m-1) : ℚ) * sum_reciprocals

theorem smallest_integer_T_is_42 :
  ∀ k : ℕ, k > 0 → (∀ i : ℕ, 0 < i ∧ i < k → ¬(T i).isInt) → (T k).isInt → k = 42 := by
  sorry

#check smallest_integer_T_is_42

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_T_is_42_l185_18572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l185_18547

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1/2 ∧ x ≤ 1 then x / (x + 2)
  else if x ≥ 0 ∧ x ≤ 1/2 then -1/2 * x + 1/4
  else 0

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (Real.pi/3 * x + 3*Real.pi/2) - 2*a + 2

theorem function_properties (a : ℝ) (h : a > 0) :
  (∀ y : ℝ, y ∈ Set.range f → 0 ≤ y ∧ y ≤ 1/3) ∧
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → x₁ < x₂ → g a x₁ < g a x₂) ∧
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → f x₁ = g a x₂ → 5/9 ≤ a ∧ a ≤ 4/5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l185_18547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_0_2571_to_hundredth_l185_18508

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The theorem states that rounding 0.2571 to the nearest hundredth results in 0.26 -/
theorem round_0_2571_to_hundredth :
  round_to_hundredth 0.2571 = 0.26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_0_2571_to_hundredth_l185_18508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_pace_calculation_l185_18571

-- Define the hiking parameters
noncomputable def total_distance : ℝ := 24 -- miles
noncomputable def total_time : ℝ := 6 -- hours
noncomputable def rest_time : ℝ := 45 / 60 -- hours (45 minutes converted to hours)

-- Calculate the actual hiking time
noncomputable def hiking_time : ℝ := total_time - rest_time

-- Define the expected average pace
noncomputable def expected_pace : ℝ := 4.57 -- miles per hour

-- Theorem to prove
theorem hiking_pace_calculation :
  abs (total_distance / hiking_time - expected_pace) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_pace_calculation_l185_18571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_trophy_implies_not_all_events_no_trophy_implies_lost_at_least_one_l185_18511

-- Define the universe of events
inductive Event : Type
  | A : Event
  | B : Event

-- Define a player
structure Player :=
  (won_event_A : Prop)
  (won_event_B : Prop)

-- Define the condition for receiving a special trophy
def receives_special_trophy (p : Player) : Prop :=
  p.won_event_A ∧ p.won_event_B

-- Theorem to prove
theorem no_trophy_implies_not_all_events (p : Player) :
  ¬(receives_special_trophy p) → ¬(p.won_event_A ∧ p.won_event_B) :=
by
  intro h
  contrapose! h
  exact h

-- The contrapositive of the original statement
theorem no_trophy_implies_lost_at_least_one (p : Player) :
  ¬(receives_special_trophy p) → (¬p.won_event_A ∨ ¬p.won_event_B) :=
by
  intro h
  contrapose! h
  simp [receives_special_trophy]
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_trophy_implies_not_all_events_no_trophy_implies_lost_at_least_one_l185_18511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_OAB_l185_18581

/-- Given complex numbers z₁ and z₂ corresponding to points A and B on the complex plane,
    with |z₁| = 4 and 4z₁² - 2z₁z₂ + z₂² = 0, prove that the area of triangle OAB is 8,
    where O is the origin. -/
theorem triangle_area_OAB (z₁ z₂ : ℂ) 
    (h₁ : Complex.abs z₁ = 4)
    (h₂ : 4 * z₁^2 - 2 * z₁ * z₂ + z₂^2 = 0) : 
  (Complex.abs (z₁.im * z₂.re - z₁.re * z₂.im)) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_OAB_l185_18581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l185_18517

theorem inequality_theorem (a : ℝ) (n : ℕ) (h_a : a > 0) (h_n : n ≥ 1) :
  a^n + a^(-n : ℝ) - 2 ≥ n^2 * (a + a^(-1 : ℝ) - 2) ∧
  (a^n + a^(-n : ℝ) - 2 = n^2 * (a + a^(-1 : ℝ) - 2) ↔ n = 1 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l185_18517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l185_18568

noncomputable def f (x a : ℝ) : ℝ := Real.sin (x + Real.pi/6) + Real.sin (x - Real.pi/6) + Real.cos x + a

def is_minimum (f : ℝ → ℝ) (m : ℝ) : Prop := ∀ x, f x ≥ m

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x > f y

def axis_of_symmetry (f : ℝ → ℝ) (l : ℝ → ℝ) : Prop :=
  ∀ x, f (l x) = f (l (-x))

theorem function_properties (a : ℝ) :
  (is_minimum (f · a) 1) →
  (a = 3) ∧
  (∀ k : ℤ, monotone_increasing_on (f · a) (Set.Icc (- 2*Real.pi/3 + 2*Real.pi*↑k) (Real.pi/3 + 2*Real.pi*↑k))) ∧
  (∀ k : ℤ, monotone_decreasing_on (f · a) (Set.Icc (- 5*Real.pi/3 + 2*Real.pi*↑k) (- 2*Real.pi/3 + 2*Real.pi*↑k))) ∧
  (axis_of_symmetry (f · a) (λ x ↦ Real.pi/3 + Real.pi*↑x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l185_18568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_positive_real_solutions_l185_18526

open Complex

theorem product_of_positive_real_solutions :
  ∃ (S : Finset ℂ), 
    (∀ z ∈ S, z^6 = -729 ∧ z.re > 0) ∧
    (∀ z, z^6 = -729 ∧ z.re > 0 → z ∈ S) ∧
    S.prod id = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_positive_real_solutions_l185_18526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vertical_and_sqrt3_slope_l185_18515

/-- The angle between a vertical line and a line with slope √3 is π/6 -/
theorem angle_between_vertical_and_sqrt3_slope :
  let vertical_line : Set (ℝ × ℝ) := {p | p.1 = -2}
  let sloped_line : Set (ℝ × ℝ) := {p | Real.sqrt 3 * p.1 - p.2 + 1 = 0}
  ∃ (angle_between : Set (ℝ × ℝ) → Set (ℝ × ℝ) → ℝ),
    angle_between vertical_line sloped_line = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vertical_and_sqrt3_slope_l185_18515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_shapes_angle_measure_l185_18533

structure InscribedShapes :=
  (circle : Set (ℝ × ℝ))
  (triangle : Set (ℝ × ℝ))
  (square1 : Set (ℝ × ℝ))
  (square2 : Set (ℝ × ℝ))

def is_inscribed (shape : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) : Prop :=
  ∀ p, p ∈ shape → p ∈ circle

def is_equilateral_triangle (triangle : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c : ℝ × ℝ, triangle = {a, b, c} ∧ 
    ∀ x y, x ∈ triangle → y ∈ triangle → x ≠ y → dist x y = dist a b

def is_square (square : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c d : ℝ × ℝ, square = {a, b, c, d} ∧
    ∀ x y, x ∈ square → y ∈ square → x ≠ y → (dist x y = dist a b ∨ dist x y = dist a c)

def shares_vertex (shape1 shape2 : Set (ℝ × ℝ)) : Prop :=
  ∃ v : ℝ × ℝ, v ∈ shape1 ∧ v ∈ shape2

noncomputable def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem inscribed_shapes_angle_measure (shapes : InscribedShapes) :
  is_inscribed shapes.triangle shapes.circle →
  is_inscribed shapes.square1 shapes.circle →
  is_inscribed shapes.square2 shapes.circle →
  is_equilateral_triangle shapes.triangle →
  is_square shapes.square1 →
  is_square shapes.square2 →
  shares_vertex shapes.triangle shapes.square1 →
  shares_vertex shapes.triangle shapes.square2 →
  ∃ p q r : ℝ × ℝ, 
    p ∈ shapes.triangle ∧ 
    q ∈ shapes.triangle ∧ 
    q ∈ shapes.square2 ∧ 
    r ∈ shapes.square2 ∧ 
    angle_measure p q r = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_shapes_angle_measure_l185_18533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l185_18513

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infiniteGeometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: For an infinite geometric series with common ratio -3/7 and sum 18, the first term is 180/7 -/
theorem first_term_of_geometric_series :
  ∃ a : ℝ, infiniteGeometricSum a (-3/7) = 18 ∧ a = 180/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l185_18513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l185_18510

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2 * Real.sin (x + Real.pi / 4) + 2 * x^2 + x) / (2 * x^2 + Real.cos x)

-- State the theorem
theorem sum_of_max_and_min_f : 
  ∃ (max_f min_f : ℝ), 
    (∀ x, f x ≤ max_f) ∧ 
    (∃ x, f x = max_f) ∧
    (∀ x, min_f ≤ f x) ∧ 
    (∃ x, f x = min_f) ∧
    max_f + min_f = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l185_18510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_sin_matrix_zero_l185_18564

/-- The determinant of the given 3x3 matrix is zero -/
theorem det_sin_matrix_zero (a b : Real) : 
  let M : Matrix (Fin 3) (Fin 3) Real := !![
    1, Real.sin (2 * a), Real.sin a;
    Real.sin (2 * a), 1, Real.sin b;
    Real.sin a, Real.sin b, 1
  ]
  Matrix.det M = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_sin_matrix_zero_l185_18564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_N_l185_18554

/-- The distance between two points in 3D space -/
noncomputable def distance3D (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

theorem distance_M_N :
  distance3D 0 1 2 (-1) 2 1 = Real.sqrt 3 := by
  sorry

#check distance_M_N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_N_l185_18554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_condition_l185_18553

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x - 2 * a * x) / x

theorem unique_integer_condition (a : ℝ) : 
  (∃! k : ℤ, f a (k : ℝ) > 1) ↔ 
  (1/4 * Real.log 2 - 1/2 ≤ a ∧ a < 1/6 * Real.log 3 - 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_condition_l185_18553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_proof_angle_from_parametric_equations_l185_18514

/-- The angle of inclination of a line with parametric equations x = -√3t and y = 1 + 3t -/
noncomputable def angle_of_inclination : ℝ := 2 * Real.pi / 3

/-- The parametric equations of the line -/
noncomputable def line_param (t : ℝ) : ℝ × ℝ := (-Real.sqrt 3 * t, 1 + 3 * t)

/-- Theorem stating that the angle of inclination is 2π/3 -/
theorem angle_of_inclination_proof :
  angle_of_inclination = 2 * Real.pi / 3 := by
  -- Proof goes here
  sorry

/-- Theorem relating the parametric equations to the angle of inclination -/
theorem angle_from_parametric_equations :
  ∃ (m : ℝ), (∀ t : ℝ, (line_param t).2 - 1 = m * (line_param t).1) ∧
             Real.tan angle_of_inclination = -m := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_proof_angle_from_parametric_equations_l185_18514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_vertex_centroid_Q_l185_18555

/-- Predicate to check if three points form a triangle -/
def IsTriangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is the centroid of a triangle -/
def IsCentroid (S A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if the sides of a triangle are seen at equal angles from a point Q -/
def EqualAnglesFromQ (Q A B C : ℝ × ℝ) : Prop := sorry

/-- Given a vertex, centroid, and point Q of a triangle, prove that a unique triangle can be constructed. -/
theorem triangle_construction_from_vertex_centroid_Q (A S Q : ℝ × ℝ) :
  ∃! (B C : ℝ × ℝ), IsTriangle A B C ∧ IsCentroid S A B C ∧ EqualAnglesFromQ Q A B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_vertex_centroid_Q_l185_18555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_solutions_l185_18523

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 0 then Real.sin (Real.pi * x^2)
  else if x ≥ 0 then Real.exp (x - 1)
  else 0  -- This case is added to make the function total

theorem f_equals_one_solutions :
  {a : ℝ | f a = 1} = {1, -Real.sqrt 2 / 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_solutions_l185_18523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l185_18559

-- Define the parametric equations of curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (3 + Real.sqrt 10 * Real.cos α, 1 + Real.sqrt 10 * Real.sin α)

-- Define the polar equation of the line
def line_polar (θ ρ : ℝ) : Prop := Real.sin θ - Real.cos θ = 1 / ρ

-- Theorem statement
theorem curve_C_properties :
  -- 1) The polar equation of curve C
  ∃ f : ℝ → ℝ, ∀ θ, f θ = 6 * Real.cos θ + 2 * Real.sin θ ∧
    ∀ ρ θ, (ρ * Real.cos θ, ρ * Real.sin θ) = curve_C θ ↔ ρ = f θ
  ∧
  -- 2) The length of the chord cut from curve C by the line
  ∃ chord_length : ℝ, chord_length = Real.sqrt 22 ∧
    ∀ θ₁ θ₂ ρ₁ ρ₂,
      line_polar θ₁ ρ₁ ∧ line_polar θ₂ ρ₂ ∧
      (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁) = curve_C θ₁ ∧
      (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂) = curve_C θ₂ →
      Real.sqrt ((ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂)^2 + (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂)^2) = chord_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l185_18559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_equidistant_l185_18587

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to a line --/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Check if a point is on a line --/
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a line is equidistant from two points --/
def isEquidistant (l : Line) (p1 p2 : Point) : Prop :=
  distancePointToLine p1 l = distancePointToLine p2 l

/-- The main theorem --/
theorem line_through_point_equidistant :
  ∃ (l1 l2 : Line),
    (isPointOnLine ⟨3, 4⟩ l1 ∧ isEquidistant l1 ⟨-2, 2⟩ ⟨4, -2⟩) ∧
    (isPointOnLine ⟨3, 4⟩ l2 ∧ isEquidistant l2 ⟨-2, 2⟩ ⟨4, -2⟩) ∧
    ((l1.a = 2 ∧ l1.b = -1 ∧ l1.c = -2) ∨ (l2.a = 2 ∧ l2.b = 3 ∧ l2.c = -18)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_equidistant_l185_18587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_from_three_lines_l185_18579

/-- A line in 3D space -/
structure Line3D where
  -- Define a line in 3D space (e.g., using a point and a direction vector)
  -- This is a placeholder definition
  dummy : Unit

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane in 3D space (e.g., using a point and a normal vector)
  -- This is a placeholder definition
  dummy : Unit

/-- The set of planes determined by three lines in 3D space -/
def planesFromThreeLines (l1 l2 l3 : Line3D) : Finset Plane3D :=
  -- Define the set of planes determined by the three lines
  -- This is a placeholder definition
  sorry

theorem max_planes_from_three_lines (l1 l2 l3 : Line3D) :
  (planesFromThreeLines l1 l2 l3).card ≤ 3 := by
  sorry

#check max_planes_from_three_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_from_three_lines_l185_18579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l185_18561

noncomputable def g (x : ℝ) : ℝ := ⌊x⌋ - x + 1

theorem g_neither_even_nor_odd :
  ¬(∀ x : ℝ, g (-x) = g x) ∧ ¬(∀ x : ℝ, g (-x) = -g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l185_18561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l185_18502

noncomputable def vector_BA : ℝ × ℝ := (Real.sqrt 3 / 2, 1 / 2)
def vector_BC : ℝ × ℝ := (0, 1)

theorem angle_between_vectors :
  Real.arccos ((vector_BA.1 * vector_BC.1 + vector_BA.2 * vector_BC.2) /
    (Real.sqrt (vector_BA.1^2 + vector_BA.2^2) * Real.sqrt (vector_BC.1^2 + vector_BC.2^2))) = π / 3 := by
  sorry

#check angle_between_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l185_18502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_hyperbola_l185_18573

/-- The equation x^2 - 16y^2 - 8x + 64 = 0 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b h k : ℝ) (ε : ℝ),
    ε = 1 ∨ ε = -1 ∧
    ∀ x y : ℝ,
      x^2 - 16*y^2 - 8*x + 64 = 0 ↔
      ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = ε ∧
      a ≠ 0 ∧ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_hyperbola_l185_18573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_A_B_l185_18556

-- Define the set A as solutions to z^3 - 8 = 0
def set_A : Set ℂ := {z : ℂ | z^3 - 8 = 0}

-- Define the set B as solutions to z^3 - 8z^2 - 8z + 64 = 0
def set_B : Set ℂ := {z : ℂ | z^3 - 8*z^2 - 8*z + 64 = 0}

-- Function to calculate the distance between two complex numbers
noncomputable def distance (z1 z2 : ℂ) : ℝ := Complex.abs (z1 - z2)

-- Theorem stating the maximum distance between points in A and B
theorem max_distance_A_B : 
  ∃ (a : ℂ) (b : ℂ), a ∈ set_A ∧ b ∈ set_B ∧
    (∀ (x : ℂ) (y : ℂ), x ∈ set_A → y ∈ set_B → 
      distance x y ≤ distance a b) ∧
    distance a b = 2 * Real.sqrt 21 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_A_B_l185_18556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_midpoint_implies_a_l185_18544

-- Define the circle C
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + a = 0

-- Define the midpoint M
def M : ℝ × ℝ := (1, 0)

-- Define the length of chord AB
def chord_length : ℝ := 3

-- Theorem statement
theorem chord_midpoint_implies_a (a : ℝ) :
  (∃ (A B : ℝ × ℝ),
    circle_equation A.1 A.2 a ∧
    circle_equation B.1 B.2 a ∧
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = chord_length^2) →
  a = 3/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_midpoint_implies_a_l185_18544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_friday_time_l185_18565

/-- Janet's weekly gym schedule -/
structure GymSchedule where
  total_time : ℚ
  days : ℕ
  mon_wed_time : ℚ
  tues_fri_equal : Bool

/-- Calculate Janet's Friday gym time -/
def friday_time (schedule : GymSchedule) : ℚ :=
  let remaining_time := schedule.total_time - 2 * schedule.mon_wed_time
  remaining_time / 2

/-- Theorem: Janet's Friday gym time is 1 hour -/
theorem janet_friday_time :
  let janet_schedule := GymSchedule.mk 5 4 (3/2) true
  friday_time janet_schedule = 1 := by
  -- Unfold the definition of friday_time
  unfold friday_time
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_friday_time_l185_18565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l185_18504

-- We'll use the existing Vector type from Mathlib
-- No need to redefine Vector, dot product, or magnitude

open Real

-- Define the angle between two vectors
noncomputable def angle {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] (v w : n) : ℝ :=
  Real.arccos ((inner v w) / (norm v * norm w))

-- The theorem to prove
theorem vector_magnitude {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] (a b : n) :
  angle a b = π / 3 →
  norm b = 1 →
  norm (a + 2 • b) = 2 * sqrt 3 →
  norm a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l185_18504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_k_value_l185_18507

theorem linear_function_k_value :
  ∀ k : ℝ,
  k ≠ 0 →
  (λ x : ℝ => k * x - 4) (-1) = -2 →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_k_value_l185_18507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l185_18525

theorem power_equation (a b : ℝ) (h1 : (30 : ℝ)^a = 2) (h2 : (30 : ℝ)^b = 7) :
  (10 : ℝ)^((2 - a - 2*b)/(3*(1 - b))) = 450/49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l185_18525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l185_18541

-- Define the function (marked as noncomputable due to dependence on real numbers)
noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 8) / (4 * x^2 + 7 * x + 3)

-- Define the denominator function (also marked as noncomputable)
noncomputable def denom (x : ℝ) : ℝ := 4 * x^2 + 7 * x + 3

-- Theorem statement
theorem vertical_asymptotes_sum :
  ∃ c d : ℝ, 
    (∀ x : ℝ, denom x = 0 ↔ x = c ∨ x = d) ∧
    c + d = -7/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l185_18541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotten_fruit_fraction_l185_18549

theorem rotten_fruit_fraction (a p : ℕ) (ha : a > 0) (hp : p > 0) : 
  (2 * a = 3 * p) →  -- Equal number of rotten apples and pears
  (2 * a : ℚ) / (a + p) = 12 / 17 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotten_fruit_fraction_l185_18549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_division_l185_18578

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to its binary representation. -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec to_bits (m : ℕ) (acc : List Bool) : List Bool :=
    if m = 0 then acc.reverse
    else to_bits (m / 2) ((m % 2 = 1) :: acc)
  to_bits n []

theorem binary_multiplication_division :
  let a := binary_to_nat [false, true, true, false, true, true]  -- 110110₂
  let b := binary_to_nat [false, true, false, true]              -- 1010₂
  let c := binary_to_nat [false, false, true]                    -- 100₂
  let result := binary_to_nat [false, true, false, false, false, false, false, true]  -- 10000010₂
  (a * b) / c = result := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_division_l185_18578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_finishes_first_l185_18590

/-- Represents the area of a garden -/
structure GardenArea where
  size : ℝ
  size_pos : size > 0

/-- Represents the speed of a lawnmower -/
structure LawnmowerSpeed where
  speed : ℝ
  speed_pos : speed > 0

/-- Represents a person with their garden area and lawnmower speed -/
structure Person where
  name : String
  garden : GardenArea
  mower : LawnmowerSpeed

/-- Calculates the time taken to mow a garden -/
noncomputable def mowingTime (p : Person) : ℝ :=
  p.garden.size / p.mower.speed

theorem danny_finishes_first (emily danny fiona : Person)
  (h1 : emily.garden.size = 3 * danny.garden.size)
  (h2 : emily.garden.size = 5 * fiona.garden.size)
  (h3 : fiona.mower.speed = 1/4 * danny.mower.speed)
  (h4 : fiona.mower.speed = 1/5 * emily.mower.speed) :
  mowingTime danny < min (mowingTime emily) (mowingTime fiona) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_finishes_first_l185_18590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l185_18543

def S (n : ℕ) : ℕ := 2^n - 1

noncomputable def a : ℕ → ℝ
  | 0 => 0  -- Adding the case for 0
  | 1 => 1
  | (n+2) => (S (n+2) : ℝ) - (S (n+1) : ℝ)

noncomputable def f (n : ℕ) : ℝ := (a n) / ((a n) * (S n) + (a 6))

theorem max_value_of_f :
  ∃ (C : ℝ), C = 1/15 ∧ ∀ (n : ℕ), f n ≤ C ∧ ∃ (n₀ : ℕ), f n₀ = C :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l185_18543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_forms_line_l185_18534

-- Define the set of complex numbers z satisfying |z+1| = |z-i|
def Z : Set ℂ := {z : ℂ | Complex.abs (z + 1) = Complex.abs (z - Complex.I)}

-- Statement: The set Z forms a line in the complex plane
theorem Z_forms_line : ∃ (a b c : ℝ) (h : (a, b) ≠ (0, 0)), 
  Z = {z : ℂ | a * z.re + b * z.im = c} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_forms_line_l185_18534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l185_18524

-- Define the universe of discourse
variable (U : Type) [Nonempty U]

-- Define the predicates
variable (citizen : U → Prop)
variable (good_driver : U → Prop)
variable (bad_driver : U → Prop)

-- Define the statements
def all_citizens_bad (U : Type) (citizen bad_driver : U → Prop) : Prop := 
  ∀ x, citizen x → bad_driver x

def some_citizens_good (U : Type) (citizen good_driver : U → Prop) : Prop := 
  ∃ x, citizen x ∧ good_driver x

-- Theorem statement
theorem negation_equivalence (U : Type) [Nonempty U] 
  (citizen good_driver bad_driver : U → Prop) :
  ¬(all_citizens_bad U citizen bad_driver) ↔ some_citizens_good U citizen good_driver :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l185_18524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l185_18575

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) / (e ^ x)

noncomputable def g (a : ℝ) : ℝ :=
  if a < 0 then (1 + a) / e
  else if a ≤ 2 then 1 / (e ^ (1 - a))
  else (a - 1) * e

theorem problem_solution :
  (∃ a : ℝ, (deriv (f a)) 0 = -1 ∧ a = 2) ∧
  (∀ a : ℝ, IsGreatest { y | ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = y } (g a)) ∧
  (∃ m : ℝ, m > 0 ∧ m = 1 ∧
    (∀ x : ℝ, x ∈ Set.Ioo 0 1 → f 0 x > f 0 (m / x)) ∧
    (∀ m' : ℝ, m' > 0 ∧ (∀ x : ℝ, x ∈ Set.Ioo 0 1 → f 0 x > f 0 (m' / x)) → m' ≥ m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l185_18575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l185_18551

/-- A right triangle with an inscribed square -/
structure RightTriangleWithInscribedSquare where
  /-- Length of the hypotenuse PQ -/
  pq : ℝ
  /-- Length of the side PR -/
  pr : ℝ
  /-- The square is inscribed in the triangle -/
  inscribed : True
  /-- PQ is the hypotenuse -/
  pq_hypotenuse : True
  /-- W lies on PR -/
  w_on_pr : True
  /-- X lies on PQ -/
  x_on_pq : True
  /-- Y and Z lie on QR -/
  yz_on_qr : True

/-- The area of the inscribed square -/
def area_of_inscribed_square (t : RightTriangleWithInscribedSquare) : ℝ := 
  t.pq * t.pr

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area (t : RightTriangleWithInscribedSquare) 
  (h1 : t.pq = 34) (h2 : t.pr = 66) : 
  ∃ s : ℝ, s * s = 2244 ∧ s * s = area_of_inscribed_square t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l185_18551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_cubic_polynomial_l185_18563

-- Define the polynomial type
def CubicPolynomial (b c : ℝ) : ℝ → ℝ := λ x ↦ x^3 + b*x^2 + c*x

-- Define munificence
noncomputable def Munificence (p : ℝ → ℝ) : ℝ :=
  ⨆ x ∈ Set.Icc (-1) 1, |p x|

theorem smallest_munificence_cubic_polynomial :
  ∃ (b c : ℝ), Munificence (CubicPolynomial b c) = 1 ∧
  ∀ (b' c' : ℝ), Munificence (CubicPolynomial b' c') ≥ 1 := by
  sorry

#check smallest_munificence_cubic_polynomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_cubic_polynomial_l185_18563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_region_existence_l185_18566

/-- Represents a circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an intersection point between two circles -/
inductive Color
  | Red
  | Blue
  | Yellow

structure IntersectionPoint where
  point : ℝ × ℝ
  color : Color

/-- Represents a region formed by circle intersections -/
structure Region where
  vertices : List IntersectionPoint

/-- The number of circles -/
def n : Nat := 2018

/-- The minimum number of yellow points required on each circle -/
def yellow_threshold : Nat := 2061

/-- Check if two circles intersect -/
def intersect (c1 c2 : Circle) : Prop := sorry

/-- Check if three circles have a common intersection point -/
def intersect_three (c1 c2 c3 : Circle) : Prop := sorry

/-- Count the number of intersections on a circle -/
def num_intersections (c : Circle) (points : List IntersectionPoint) : Nat := sorry

/-- Check if colors are alternating on a circle -/
def alternating_colors (c : Circle) (points : List IntersectionPoint) : Prop := sorry

/-- Define how a point becomes yellow -/
def yellow_definition (p : IntersectionPoint) : Prop := sorry

/-- Count yellow points on a circle -/
def count_yellow (c : Circle) (points : List IntersectionPoint) : Nat := sorry

/-- Check if all vertices of a region are yellow -/
def all_vertices_yellow (r : Region) : Prop := sorry

theorem yellow_region_existence 
  (circles : List Circle)
  (intersection_points : List IntersectionPoint)
  (regions : List Region)
  (h1 : circles.length = n)
  (h2 : ∀ c1 c2, c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → intersect c1 c2)
  (h3 : ∀ c1 c2 c3, c1 ∈ circles → c2 ∈ circles → c3 ∈ circles → ¬(intersect_three c1 c2 c3))
  (h4 : ∀ c, c ∈ circles → Even (num_intersections c intersection_points))
  (h5 : ∀ c, c ∈ circles → alternating_colors c intersection_points)
  (h6 : ∀ p, p ∈ intersection_points → yellow_definition p)
  (h7 : ∀ c, c ∈ circles → (count_yellow c intersection_points) ≥ yellow_threshold) :
  ∃ r, r ∈ regions ∧ all_vertices_yellow r :=
sorry

#check yellow_region_existence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_region_existence_l185_18566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l185_18589

-- Define the ellipse
def Ellipse (F₁ F₂ : ℝ × ℝ) (e : ℝ) :=
  {P : ℝ × ℝ | ∃ (a : ℝ), a > 0 ∧ e = (dist F₁ F₂) / (2 * a) ∧
    dist P F₁ + dist P F₂ = 2 * a}

-- Define the point P
def P_on_ellipse (F₁ F₂ : ℝ × ℝ) (e : ℝ) (P : ℝ × ℝ) :=
  P ∈ Ellipse F₁ F₂ e

-- Helper function to calculate cosine of an angle given three points
noncomputable def cos_angle (A B C : ℝ × ℝ) : ℝ :=
  let ab := dist A B
  let bc := dist B C
  let ac := dist A C
  (ab^2 + bc^2 - ac^2) / (2 * ab * bc)

-- Theorem statement
theorem ellipse_properties
  (F₁ : ℝ × ℝ) (F₂ : ℝ × ℝ) (e : ℝ) (P : ℝ × ℝ)
  (h_F₁ : F₁ = (0, -1))
  (h_F₂ : F₂ = (0, 1))
  (h_e : e = 1/2)
  (h_P : P_on_ellipse F₁ F₂ e P)
  (h_diff : dist P F₁ - dist P F₂ = 1) :
  (∀ (x y : ℝ), (x, y) ∈ Ellipse F₁ F₂ e ↔ y^2/4 + x^2/3 = 1) ∧
  (cos_angle F₁ P F₂ = 3/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l185_18589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_trailing_zeroes_l185_18584

/-- Represents the arithmetic sequence in the problem -/
def problemSequence : List Nat := List.range ((700 - 1) / 3 + 1) |>.map (fun n => 1 + 3 * n)

/-- Calculates the number of trailing zeroes in a natural number -/
def trailingZeroes (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n + 1 => if n % 10 == 9 then 0 else trailingZeroes (n / 10) + 1

/-- Theorem stating that the product of the sequence has 60 trailing zeroes -/
theorem sequence_product_trailing_zeroes :
  trailingZeroes (problemSequence.prod) = 60 := by
  sorry

#eval problemSequence -- To verify the sequence is correct
#eval trailingZeroes (problemSequence.prod) -- To verify the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_trailing_zeroes_l185_18584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_comparison_l185_18597

/-- Represents a square with its shaded area -/
structure ShadedSquare where
  totalArea : ℝ
  shadedArea : ℝ

/-- Square I: divided by diagonals, one triangle shaded -/
noncomputable def squareI : ShadedSquare :=
  { totalArea := 1
  , shadedArea := 1 / 4 }

/-- Square II: divided into four smaller squares, two shaded -/
noncomputable def squareII : ShadedSquare :=
  { totalArea := 1
  , shadedArea := 1 / 2 }

/-- Square III: divided into sixteen smaller squares, four shaded -/
noncomputable def squareIII : ShadedSquare :=
  { totalArea := 1
  , shadedArea := 1 / 4 }

theorem shaded_areas_comparison :
  squareI.shadedArea = squareIII.shadedArea ∧
  squareI.shadedArea = 1 / 4 ∧
  squareII.shadedArea = 1 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_comparison_l185_18597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_program_result_l185_18509

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Represents the nth term of an arithmetic sequence -/
def arithmeticTerm (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem computer_program_result :
  ∃ n : ℕ, 
    let a : ℕ := 5  -- Initial value of X
    let d : ℕ := 3  -- Increment of X per iteration
    arithmeticSum a d n ≥ 15000 ∧ arithmeticTerm a d n = 173 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_program_result_l185_18509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_distance_to_ceiling_l185_18537

/-- The distance from a point to the ceiling in a room --/
def distance_to_ceiling (x y z : ℝ) (ceiling_height : ℝ) : ℝ :=
  ceiling_height - z

/-- The straight line distance from the origin to a point --/
noncomputable def distance_from_origin (x y z : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 + z^2)

theorem fly_distance_to_ceiling :
  ∀ (x y z : ℝ) (ceiling_height : ℝ),
    x = 3 →
    y = 4 →
    distance_from_origin x y z = 13 →
    ceiling_height = 15 →
    distance_to_ceiling x y z ceiling_height = 3 := by
  intro x y z ceiling_height h1 h2 h3 h4
  unfold distance_to_ceiling
  unfold distance_from_origin at h3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_distance_to_ceiling_l185_18537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_solid_with_ngon_projections_l185_18516

/-- A solid in 3-dimensional space -/
def Solid : Type := Set (Fin 3 → ℝ)

/-- A parallel projection from 3D to 2D -/
def ParallelProjection : Type := (Fin 3 → ℝ) → (Fin 2 → ℝ)

/-- A convex n-gon in 2D -/
def ConvexNGon (n : ℕ) : Type := Set (Fin 2 → ℝ)

/-- The theorem statement -/
theorem exists_solid_with_ngon_projections :
  ∃ (S : Solid), ∀ (n : ℕ), n ≥ 3 → 
    ∃ (P : ParallelProjection), ∃ (G : ConvexNGon n),
      Set.image P S = G := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_solid_with_ngon_projections_l185_18516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_cost_is_168_cny_l185_18545

/-- Exchange rates and painting cost -/
structure ExchangeRates where
  usd_to_nam : ℚ
  gbp_to_usd : ℚ
  gbp_to_nam : ℚ
  usd_to_cny : ℚ
  painting_cost_nam : ℚ

/-- Calculate the cost of the painting in Chinese yuan -/
noncomputable def painting_cost_cny (rates : ExchangeRates) : ℚ :=
  rates.painting_cost_nam / rates.gbp_to_nam * rates.gbp_to_usd * rates.usd_to_cny

/-- Theorem stating that the painting costs 168 Chinese yuan -/
theorem painting_cost_is_168_cny (rates : ExchangeRates) 
    (h1 : rates.usd_to_nam = 7)
    (h2 : rates.gbp_to_usd = 14/10)
    (h3 : rates.gbp_to_nam = 49/5)
    (h4 : rates.usd_to_cny = 6)
    (h5 : rates.painting_cost_nam = 196) : 
  painting_cost_cny rates = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_cost_is_168_cny_l185_18545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l185_18540

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 1 + a * log (1 - x)

-- State the theorem
theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : 0 < a ∧ a < 1/2) 
  (h_domain : x₁ < 1 ∧ x₂ < 1)
  (h_order : x₁ < x₂) 
  (h_extreme : ∀ x, x < 1 → deriv (f a) x = 0 → x = x₁ ∨ x = x₂) :
  (f a x₁) / x₂ > (f a x₂) / x₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l185_18540
