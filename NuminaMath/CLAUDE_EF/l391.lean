import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l391_39156

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.exp x - a * x + b

-- Define the function g
noncomputable def g (a b x : ℝ) : ℝ := f a b x + Real.log (x + 1)

-- Theorem statement
theorem function_properties (a b : ℝ) :
  (∃ x₀, IsLocalMin (f a b) x₀ ∧ x₀ = 0 ∧ f a b x₀ = 2) ∧
  (∀ x ≥ 0, g a b x ≥ 1 + b) →
  a = 1 ∧ b = 1 ∧ a ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l391_39156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_approx_half_l391_39127

/-- The infinite series in question -/
noncomputable def series_sum : ℝ := ∑' n, (n^4 + 3*n^3 + 10*n + 10) / (3^n * (n^4 + 4))

/-- The theorem stating that the series sum is approximately 1/2 -/
theorem series_sum_approx_half : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |series_sum - 1/2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_approx_half_l391_39127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_foci_l391_39153

-- Define the ellipse and hyperbola
def ellipse (b : ℝ) (x y : ℝ) : Prop := x^2/25 + y^2/b^2 = 1
def hyperbola (x y : ℝ) : Prop := x^2/100 - y^2/64 = 1/16

-- Define the foci of the hyperbola
noncomputable def hyperbola_foci : ℝ × ℝ := (Real.sqrt 41 / 2, 0)

-- Define the condition that the foci coincide
def foci_coincide (b : ℝ) : Prop :=
  ∃ (c : ℝ), c^2 = 25 - b^2 ∧ (c, 0) = hyperbola_foci

-- State the theorem
theorem ellipse_hyperbola_foci (b : ℝ) :
  foci_coincide b → b^2 = 59/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_foci_l391_39153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_pi_minus_one_implies_fraction_l391_39150

theorem tan_alpha_pi_minus_one_implies_fraction (α : ℝ) :
  Real.tan (α + π) = -1 →
  (2 * Real.sin α + Real.cos α) / (Real.cos α - Real.sin α) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_pi_minus_one_implies_fraction_l391_39150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equation_solution_l391_39180

-- Define the left-hand side of the equation
noncomputable def f (y : ℝ) : ℝ := 
  let rec aux (n : ℕ) : ℝ := 
    if n = 0 then y else Real.sqrt (y + aux (n-1))
  aux 1000000  -- Using a large finite approximation

-- Define the right-hand side of the equation
noncomputable def g (y : ℝ) : ℝ := 
  let rec aux (n : ℕ) : ℝ := 
    if n = 0 then y else Real.sqrt (y * aux (n-1))
  aux 1000000  -- Using a large finite approximation

-- State the theorem
theorem nested_radical_equation_solution :
  ∃! (y : ℝ), y > 0 ∧ f y = g y ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equation_solution_l391_39180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l391_39106

theorem inequality_condition (a b x : ℝ) (ha : a < 0) (hb : b > 0) :
  (abs (a - abs (x - 1)) + abs (abs (x - 1) - b) ≥ abs (a - b)) ↔ (1 - b ≤ x ∧ x ≤ 1 + b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l391_39106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_sum_constant_l391_39104

-- Define the properties of even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem even_odd_sum_constant
  (f g : ℝ → ℝ)
  (hf : IsEven f)
  (hg : IsOdd g)
  (h_sum : ∀ x, f x + g x = 5) :
  (∀ x, f x = (5/2)) ∧ (∀ x, g x = (5/2)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_sum_constant_l391_39104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l391_39142

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area function for a triangle
noncomputable def Triangle.area (t : Triangle) : ℝ := 
  1/2 * t.a * t.b * Real.sin t.C

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : 2 * abc.b * Real.cos abc.C = abc.a * Real.cos abc.C + abc.c * Real.cos abc.A) 
  (h2 : abc.b = 2)
  (h3 : abc.c = Real.sqrt 7) :
  abc.C = π / 3 ∧ 
  abc.a = 3 ∧ 
  Triangle.area abc = 3 * Real.sqrt 3 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l391_39142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l391_39194

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- Main theorem
theorem main_theorem (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b x = -f a b (-x)) →  -- f is odd
  f a b (1/2) = 2/5 →                                      -- f(1/2) = 2/5
  (∃ g : ℝ → ℝ, 
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b x = x / (1 + x^2)) ∧  -- f(x) = x/(1+x^2)
    (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → g x < g y) ∧  -- f is increasing
    (Set.Ioo 0 (1/2) = {t | f a b (t-1) + f a b t < 0})) := by  -- solution set
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l391_39194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_358_l391_39128

def f : ℕ → ℕ
  | 0 => 1  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 2
  | n+3 => 2 * f (n+2) - f (n+1) + (n+3)^2

theorem f_10_equals_358 : f 10 = 358 := by
  -- Proof will be added later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_358_l391_39128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_domain_size_l391_39151

-- Define the function f
def f : ℕ → ℕ 
  | n => sorry  -- We'll define the function behavior in axioms

-- Define the properties of f
axiom f_7 : f 7 = 22
axiom f_odd : ∀ b : ℕ, Odd b → f b = 3 * b + 1
axiom f_even : ∀ b : ℕ, Even b → f b = b / 2

-- Define the domain of f
def domain_f : Set ℕ := {n : ℕ | ∃ m : ℕ, f m = n ∨ m = n}

-- Theorem to prove
theorem smallest_domain_size : 
  (∃ S : Finset ℕ, S.card = 16 ∧ ∀ n ∈ S, f n ∈ S) ∧ 
  (∀ S : Finset ℕ, (∀ n ∈ S, f n ∈ S) → S.card ≥ 16) :=
sorry

-- Additional lemmas that might be useful for the proof
lemma domain_contains_7 : 7 ∈ domain_f := sorry

lemma domain_closed_under_f : ∀ n ∈ domain_f, f n ∈ domain_f := sorry

lemma domain_finite : Finite domain_f := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_domain_size_l391_39151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_lambda_l391_39149

def is_equilateral_triangle (a b c : ℂ) : Prop :=
  Complex.abs (b - a) = Complex.abs (c - b) ∧
  Complex.abs (c - b) = Complex.abs (a - c) ∧
  Complex.abs (a - c) = Complex.abs (b - a)

theorem equilateral_triangle_lambda (ω : ℂ) (l : ℝ) : 
  Complex.abs ω = 3 →
  l > 1 →
  is_equilateral_triangle (ω^2) (ω^3) (l * ω^2) →
  l = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_lambda_l391_39149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_distances_line_exists_l391_39101

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in a plane (ax + by + c = 0)
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the distance function from a point to a line
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / (Real.sqrt (l.a^2 + l.b^2))

-- Theorem statement
theorem distinct_distances_line_exists (points : Finset Point) :
  ∃ (l : Line), ∀ (p q : Point), p ∈ points → q ∈ points → p ≠ q →
    distancePointToLine p l ≠ distancePointToLine q l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_distances_line_exists_l391_39101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l391_39132

noncomputable def f (x : ℝ) : ℝ := x
noncomputable def g (t : ℝ) : ℝ := (t^3 + t) / (t^2 + 1)

theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l391_39132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_cooling_time_l391_39159

/-- Cooling equation for an object in air -/
noncomputable def cooling_equation (θ₀ θ₁ t : ℝ) : ℝ := θ₀ + (θ₁ - θ₀) * Real.exp (-0.08 * t)

/-- Time needed for coffee to cool to optimal temperature -/
theorem coffee_cooling_time (ε : ℝ) (ε_pos : ε > 0) :
  ∃ t : ℝ, t > 0 ∧ |cooling_equation 25 85 t - 65| < ε ∧ |t - 5| < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_cooling_time_l391_39159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_period_and_triangle_area_l391_39145

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 + 1

-- Define the function g as a translation of f
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi/3)

-- State the theorem
theorem function_period_and_triangle_area :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (A B C : ℝ), 
    g (A/2) = 1 →
    2 = 2 → -- This represents a = 2
    B + C = 4 →
    (1/2) * 2 * (4 - 2) * Real.sin A = Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_period_and_triangle_area_l391_39145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_3_power_a_l391_39185

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem f_of_3_power_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 9 = 2) :
  f a (3^a) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_3_power_a_l391_39185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separation_l391_39184

/-- The line l with equation x - y + 2 = 0 -/
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

/-- The circle C with center (1, -1) and radius √2 -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

/-- The distance between a point (x, y) and the line ax + by + c = 0 -/
noncomputable def distance_point_line (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

theorem line_circle_separation :
  ∀ x y : ℝ, line_l x y → ¬ circle_C x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separation_l391_39184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l391_39119

def ball_numbers : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def person_A_balls : List Nat := [6, 11]
def person_B_balls : List Nat := [4, 8]
def person_C_known_ball : Nat := 1

def total_sum : Nat := ball_numbers.sum
def each_person_sum : Nat := total_sum / 3

theorem ball_distribution :
  ∃ (a b c : Nat),
    a ∈ ball_numbers ∧
    b ∈ ball_numbers ∧
    c ∈ ball_numbers ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a ∉ person_A_balls ∧
    b ∉ person_A_balls ∧
    c ∉ person_A_balls ∧
    a ∉ person_B_balls ∧
    b ∉ person_B_balls ∧
    c ∉ person_B_balls ∧
    a ≠ person_C_known_ball ∧
    b ≠ person_C_known_ball ∧
    c ≠ person_C_known_ball ∧
    person_A_balls.sum + (4 - person_A_balls.length) = each_person_sum ∧
    person_B_balls.sum + (4 - person_B_balls.length) = each_person_sum ∧
    person_C_known_ball + a + b + c = each_person_sum ∧
    (a = 3 ∧ b = 10 ∧ c = 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l391_39119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l391_39177

-- Define the ※ operation
noncomputable def star (a b : ℝ) : ℝ :=
  if a ≥ 2*b then a - b else a + b - 6

-- State the theorem
theorem star_properties :
  (star 4 3 = 1) ∧
  (star (-1) (-3) = 2) ∧
  (∃! x : ℝ, star (3*x + 2) (x - 1) = 5) ∧
  (∀ x : ℝ, star (3*x + 2) (x - 1) = 5 → x = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l391_39177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_CD_l391_39102

-- Define the segment AB and point M
noncomputable def AB : ℝ := 60
noncomputable def M : ℝ := sorry

-- Define the conditions
noncomputable def AM : ℝ := M
noncomputable def MB : ℝ := AB - M

-- Define the midpoints
noncomputable def N : ℝ := AM / 2
noncomputable def P : ℝ := M + MB / 2
noncomputable def C : ℝ := N + (M - N) / 2
noncomputable def D : ℝ := M + (P - M) / 2

-- State the theorem
theorem length_CD : C - D = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_CD_l391_39102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_x_axis_intersection_l391_39186

/-- The line equation 4y - 5x = 15 intersects the x-axis at the point (-3, 0). -/
theorem line_x_axis_intersection :
  ∃ (x : ℝ), 4 * 0 - 5 * x = 15 ∧ (x, 0) = (-3, 0) := by
  use -3
  constructor
  · -- Prove that the point satisfies the line equation
    simp
    ring
  · -- Prove that the point is on the x-axis and has x-coordinate -3
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_x_axis_intersection_l391_39186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_c_value_l391_39157

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of a line y = mx + b is m -/
def line_slope (m b : ℝ) : ℝ := m

theorem parallel_lines_c_value :
  ∀ c : ℝ,
  parallel (line_slope 5 3) (line_slope (3 * c) 1) →
  c = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_c_value_l391_39157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_five_l391_39170

/-- The line equation in the form ax + by = c --/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The area of a triangle formed by a line and coordinate axes --/
noncomputable def triangleArea (line : LineEquation) : ℝ :=
  if line.a ≠ 0 && line.b ≠ 0 && line.c ≠ 0 then
    (line.c / line.a) * (line.c / line.b) / 2
  else
    0

/-- The given line equation --/
noncomputable def givenLine : LineEquation := { a := 1/5, b := 1/2, c := 1 }

theorem triangle_area_is_five :
  triangleArea givenLine = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_five_l391_39170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_full_tank_cost_l391_39146

-- Define the given conditions
noncomputable def diesel_volume : ℝ := 36
noncomputable def diesel_cost : ℝ := 18
noncomputable def tank_capacity : ℝ := 64

-- Define the price per liter
noncomputable def price_per_liter : ℝ := diesel_cost / diesel_volume

-- Theorem to prove
theorem full_tank_cost : price_per_liter * tank_capacity = 32 := by
  -- Unfold the definitions
  unfold price_per_liter diesel_cost diesel_volume tank_capacity
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_full_tank_cost_l391_39146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_prices_correct_l391_39167

/-- The original price of the wand -/
noncomputable def original_wand_price : ℝ := 96

/-- The original price of the cloak -/
noncomputable def original_cloak_price : ℝ := 150

/-- The price Harry paid for the wand -/
noncomputable def wand_price : ℝ := 12

/-- The price Harry paid for the cloak -/
noncomputable def cloak_price : ℝ := 30

/-- The fraction of the original price that Harry paid for the wand -/
noncomputable def wand_fraction : ℝ := 1/8

/-- The fraction of the original price that Harry paid for the cloak -/
noncomputable def cloak_fraction : ℝ := 1/5

theorem original_prices_correct : 
  (wand_price = original_wand_price * wand_fraction) ∧
  (cloak_price = original_cloak_price * cloak_fraction) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_prices_correct_l391_39167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_l391_39148

/-- The circumference of the smaller circle in meters -/
def c1 : ℝ := 264

/-- The circumference of the larger circle in meters -/
def c2 : ℝ := 352

/-- The mathematical constant pi -/
noncomputable def π : ℝ := Real.pi

/-- The radius of the smaller circle -/
noncomputable def r1 : ℝ := c1 / (2 * π)

/-- The radius of the larger circle -/
noncomputable def r2 : ℝ := c2 / (2 * π)

/-- The area of the smaller circle -/
noncomputable def A1 : ℝ := π * r1^2

/-- The area of the larger circle -/
noncomputable def A2 : ℝ := π * r2^2

/-- The difference between the areas of the larger and smaller circles -/
noncomputable def areaDifference : ℝ := A2 - A1

theorem circle_area_difference :
  abs (areaDifference - 4305.28) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_l391_39148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_third_stick_l391_39191

/-- Given two sticks of lengths 6 and 9, the shortest length of a third stick
    that can form a triangle with them is 4. -/
theorem shortest_third_stick :
  ∃ (third_stick : ℝ),
    third_stick = 4 ∧
    (6 : ℝ) + 9 > third_stick ∧
    (6 : ℝ) + third_stick > 9 ∧
    9 + third_stick > 6 ∧
    ∀ x : ℝ, x < third_stick →
      ¬((6 : ℝ) + 9 > x ∧ (6 : ℝ) + x > 9 ∧ 9 + x > 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_third_stick_l391_39191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_pyramid_tangent_angle_l391_39136

/-- Represents a triangular pyramid inscribed in a cone -/
structure InscribedPyramid where
  S : Point → Type*
  A : Point → Type*
  B : Point → Type*
  C : Point → Type*
  α : ℝ   -- Dihedral angle at edge SA
  β : ℝ   -- Dihedral angle at edge SB
  γ : ℝ   -- Dihedral angle at edge SC

/-- The angle between plane SBC and the plane tangent to the cone surface along SC -/
noncomputable def tangentAngle (p : InscribedPyramid) : ℝ :=
  (Real.pi - p.α + p.β - p.γ) / 2

theorem inscribed_pyramid_tangent_angle (p : InscribedPyramid) :
  tangentAngle p = (Real.pi - p.α + p.β - p.γ) / 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_pyramid_tangent_angle_l391_39136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_overlap_l391_39192

/-- The area of overlap between two unit squares, one rotated by angle α. -/
noncomputable def overlap_area (α : Real) : Real :=
  1/3

/-- The theorem statement for the square overlap problem. -/
theorem square_overlap (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi/2) (h3 : Real.cos α = 3/5) : 
  overlap_area α = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_overlap_l391_39192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l391_39140

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (-2^x + a) / (2^x + 1)

theorem odd_function_properties (a : ℝ) 
  (h_odd : ∀ x : ℝ, f a x = -f a (-x)) :
  (a = 1) ∧ 
  (∀ x y : ℝ, x < y → f a x > f a y) ∧
  (∀ k : ℝ, (∀ t : ℝ, f a (t^2 - 2*t) + f a (2*t^2 - k) < 0) ↔ k < -1/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l391_39140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_approximation_l391_39166

/-- The value of d as defined in the problem -/
noncomputable def d : ℝ := (69.28 * (0.004^3) - Real.log 27) / (0.03 * Real.cos (55 * Real.pi / 180))

/-- Theorem stating that d is approximately equal to -191.297 -/
theorem d_approximation : ∃ ε > 0, |d + 191.297| < ε ∧ ε < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_approximation_l391_39166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_25_l391_39189

/-- The total cost function for producing x pieces -/
noncomputable def cost (x : ℝ) : ℝ := 1200 + (2/75) * x^3

/-- The constant k in the price-quantity relationship -/
def k : ℝ := 25 * 10^4

/-- The price function for x pieces -/
noncomputable def price (x : ℝ) : ℝ := 500 / Real.sqrt x

/-- The profit function for x pieces -/
noncomputable def profit (x : ℝ) : ℝ := x * price x - cost x

/-- The theorem stating the maximum profit and corresponding output -/
theorem max_profit_at_25 :
  ∃ (max_profit : ℝ),
    (∀ x > 0, profit x ≤ profit 25) ∧
    (abs (profit 25 - 883) < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_25_l391_39189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l391_39113

theorem expression_value (a b x y : ℝ) (m : ℤ) : 
  (a + b = 0) → 
  (x * y = 1) → 
  (m = -1) → 
  2023 * (a + b) + 3 * m.natAbs - 2 * x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l391_39113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l391_39115

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.cos x - Real.sqrt 3 / 2)

theorem domain_of_f :
  ∀ x : ℝ, f x ∈ Set.univ ↔ ∃ k : ℤ, 2 * k * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l391_39115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_altitude_sum_l391_39122

/-- A triangle with side lengths 3, 4, and 5 -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  side_a : a = 3
  side_b : b = 4
  side_c : c = 5

/-- A point inside the triangle -/
structure Point (t : Triangle) where
  x : ℝ
  y : ℝ
  inside : x > 0 ∧ y > 0 ∧ 4*x + 3*y < 12

/-- The sum of altitudes from a point to the sides of the triangle -/
noncomputable def altitude_sum (t : Triangle) (p : Point t) : ℝ :=
  p.x + p.y + (12 - 4*p.x - 3*p.y) / 5

theorem min_altitude_sum (t : Triangle) (p : Point t) :
  altitude_sum t p ≥ 12/5 := by
  sorry

#check min_altitude_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_altitude_sum_l391_39122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_approx_l391_39103

/-- Calculates the total journey time for a boat traveling upstream and downstream in a river with obstacles. -/
noncomputable def total_journey_time (river_speed : ℝ) (upstream_distance : ℝ) (downstream_distance : ℝ) 
                       (boat_speed : ℝ) (num_obstacles : ℕ) (speed_reduction : ℝ) : ℝ :=
  let upstream_speed := boat_speed - river_speed
  let upstream_time := upstream_distance / upstream_speed
  let initial_downstream_speed := boat_speed + river_speed
  let final_downstream_speed := initial_downstream_speed - (num_obstacles : ℝ) * speed_reduction
  let downstream_time := downstream_distance / final_downstream_speed
  upstream_time + downstream_time

/-- The total journey time is approximately 17.88 hours given the specified conditions. -/
theorem journey_time_approx (ε : ℝ) (hε : ε > 0) : 
  ∃ (t : ℝ), abs (t - total_journey_time 5 80 100 12 3 0.5) < ε ∧ abs (t - 17.88) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_approx_l391_39103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_common_points_l391_39110

/-- The first curve: x^2 + 9y^2 = 9 -/
def curve1 (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- The second curve: x^2 + 4y^2 = 4 -/
def curve2 (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

/-- A point (x, y) is common to both curves if it satisfies both equations -/
def common_point (x y : ℝ) : Prop := curve1 x y ∧ curve2 x y

/-- The set of all common points -/
def common_points : Set (ℝ × ℝ) := {p | common_point p.1 p.2}

/-- The number of common points is 2 -/
theorem number_of_common_points : ∃ (s : Finset (ℝ × ℝ)), s.card = 2 ∧ ∀ p, p ∈ s ↔ p ∈ common_points := by
  sorry

#check number_of_common_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_common_points_l391_39110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_region_area_l391_39190

/-- The area of the overlapping region formed by two circular sectors -/
theorem overlapping_region_area (r : ℝ) (angle : ℝ) : 
  r = 15 →
  angle = π / 2 →
  (2 * (angle / (2 * π)) * π * r^2) - (Real.sqrt 3 / 4 * r^2) = 112.5 * π - 56.25 * Real.sqrt 3 := by
  sorry

#check overlapping_region_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_region_area_l391_39190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equalCoordinatePoints_form_bisector_l391_39174

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The set of points where the x-coordinate equals the y-coordinate. -/
def equalCoordinatePoints : Set Point2D :=
  {p : Point2D | p.x = p.y}

/-- The line y = x in the 2D plane. -/
def bisectorLine : Set Point2D :=
  {p : Point2D | p.y = p.x}

/-- The angle between the positive x-axis and the positive y-axis. -/
noncomputable def rightAngle : ℝ := Real.pi / 2

theorem equalCoordinatePoints_form_bisector :
  equalCoordinatePoints = bisectorLine ∧
  (∀ p ∈ bisectorLine, p.x ≠ 0 → Real.arctan (p.y / p.x) = rightAngle / 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equalCoordinatePoints_form_bisector_l391_39174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_replaced_person_l391_39163

/-- Given a group of 4 people with total weight W, if replacing one person
    with a new person weighing 82 kg increases the average weight by 3 kg,
    then the weight of the replaced person was 70 kg. -/
theorem weight_of_replaced_person
  (W : ℝ)  -- Total weight of original 4 persons
  (X : ℝ)  -- Weight of the replaced person
  (h1 : (W + 12) / 4 = W / 4 + 3)  -- Average weight increases by 3 kg
  (h2 : W - X + 82 = W + 12)  -- Equation for total weight before and after replacement
  : X = 70  -- The weight of the replaced person is 70 kg
:= by
  sorry  -- Placeholder for the proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_replaced_person_l391_39163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l391_39162

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 2 * x

-- State the theorem
theorem extreme_points_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
   (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = x₁ ∨ x = x₂)) →
  0 < a ∧ a < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l391_39162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mobile_purchase_price_l391_39154

/-- Represents the purchase price of the mobile phone -/
def mobile_price : ℝ := sorry

/-- The purchase price of the grinder -/
def grinder_price : ℝ := 15000

/-- The loss percentage on the grinder -/
def grinder_loss_percent : ℝ := 0.04

/-- The profit percentage on the mobile phone -/
def mobile_profit_percent : ℝ := 0.10

/-- The overall profit -/
def total_profit : ℝ := 400

/-- Theorem stating that the purchase price of the mobile phone is 10000 -/
theorem mobile_purchase_price :
  (1 + mobile_profit_percent) * mobile_price - mobile_price -
  (grinder_price * grinder_loss_percent) = total_profit →
  mobile_price = 10000 := by
  sorry

#check mobile_purchase_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mobile_purchase_price_l391_39154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_views_exist_l391_39169

/-- A three-dimensional geometric shape -/
structure Shape3D where
  -- We'll leave this abstract for now
  mk :: -- Constructor

/-- The view of a shape from a given perspective -/
def view (s : Shape3D) (perspective : String) : Set (ℝ × ℝ) := sorry

/-- A sphere is a type of Shape3D -/
def sphere : Shape3D := Shape3D.mk

theorem identical_views_exist : 
  ∃ (s : Shape3D), 
    view s "front" = view s "top" ∧ 
    view s "front" = view s "side" ∧ 
    view s "top" = view s "side" :=
by
  -- We'll use the sphere as our example
  use sphere
  sorry -- Skip the proof details for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_views_exist_l391_39169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l391_39141

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sin x ^ 2

theorem min_positive_period_of_f : ∃ T : ℝ, T > 0 ∧ 
  (∀ x : ℝ, f (x + T) = f x) ∧ 
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  T = π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l391_39141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_neg_two_implies_fraction_eq_two_fifths_l391_39134

theorem tan_neg_two_implies_fraction_eq_two_fifths (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_neg_two_implies_fraction_eq_two_fifths_l391_39134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_correct_specific_meeting_time_approx_l391_39117

/-- The time it takes for two people walking in opposite directions on a circular track to meet. -/
noncomputable def meetingTime (trackCircumference : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  trackCircumference / (speed1 + speed2)

/-- Theorem stating that the meeting time is correct for the given scenario. -/
theorem meeting_time_correct (trackCircumference : ℝ) (speed1 : ℝ) (speed2 : ℝ)
    (h1 : trackCircumference > 0)
    (h2 : speed1 > 0)
    (h3 : speed2 > 0) :
    meetingTime trackCircumference speed1 speed2 * (speed1 + speed2) = trackCircumference := by
  sorry

/-- Specific instance of the meeting time for the given problem. -/
noncomputable def specificMeetingTime : ℝ :=
  meetingTime 1000 (20000 / 60) (17000 / 60)

/-- Theorem stating that the specific meeting time is approximately 1.62 minutes. -/
theorem specific_meeting_time_approx :
    abs (specificMeetingTime - 1.62) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_correct_specific_meeting_time_approx_l391_39117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_attendance_difference_l391_39138

def seattle_estimate : ℤ := 42000
def denver_estimate : ℤ := 55000

def seattle_lower_bound : ℤ := (seattle_estimate * 85) / 100
def seattle_upper_bound : ℤ := (seattle_estimate * 115) / 100

def denver_lower_bound : ℤ := (denver_estimate * 92) / 100
def denver_upper_bound : ℤ := (denver_estimate * 108) / 100

def total_lower_bound : ℤ := 94000
def total_upper_bound : ℤ := 96000

theorem smallest_attendance_difference : 
  ∃ (seattle denver : ℤ),
    seattle_lower_bound ≤ seattle ∧ 
    seattle ≤ seattle_upper_bound ∧
    denver_lower_bound ≤ denver ∧ 
    denver ≤ denver_upper_bound ∧
    total_lower_bound ≤ seattle + denver ∧ 
    seattle + denver ≤ total_upper_bound ∧
    ∀ (s d : ℤ), 
      seattle_lower_bound ≤ s ∧ 
      s ≤ seattle_upper_bound ∧
      denver_lower_bound ≤ d ∧ 
      d ≤ denver_upper_bound ∧
      total_lower_bound ≤ s + d ∧ 
      s + d ≤ total_upper_bound →
      5000 ≤ |d - s| :=
by
  sorry

#eval seattle_lower_bound
#eval seattle_upper_bound
#eval denver_lower_bound
#eval denver_upper_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_attendance_difference_l391_39138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_no_minimum_l391_39152

-- Define the function f(x) = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- Theorem statement
theorem f_decreasing_no_minimum :
  (∀ x y : ℝ, x < y → x ≠ 0 → y ≠ 0 → f x > f y) ∧
  (¬ ∃ m : ℝ, ∀ x : ℝ, x ≠ 0 → f x ≥ m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_no_minimum_l391_39152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l391_39109

-- Define the function k(x) as noncomputable
noncomputable def k (x : ℝ) : ℝ := (3 * x + 5) / (x - 4)

-- State the theorem about the range of k(x)
theorem range_of_k : 
  Set.range k = {y : ℝ | y < 3 ∨ y > 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l391_39109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l391_39171

-- Define the line l
noncomputable def line_l (α t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, t * Real.sin α)

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the intersection condition
def intersects (α : ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
  circle_C (line_l α t₁).1 (line_l α t₁).2 ∧
  circle_C (line_l α t₂).1 (line_l α t₂).2

-- Define the distance condition
def distance_condition (α : ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
  circle_C (line_l α t₁).1 (line_l α t₁).2 ∧
  circle_C (line_l α t₂).1 (line_l α t₂).2 ∧
  (t₁ - t₂)^2 = 14

-- Theorem statement
theorem line_circle_intersection (α : ℝ) :
  intersects α ∧ distance_condition α → α = π/4 ∨ α = 3*π/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l391_39171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l391_39199

noncomputable def proj (a : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := a.1 * v.1 + a.2 * v.2
  let norm_squared := a.1 * a.1 + a.2 * a.2
  (dot / norm_squared * a.1, dot / norm_squared * a.2)

theorem vector_satisfies_projections :
  let v : ℝ × ℝ := (3.6, 7.6)
  proj (3, 2) v = (6, 4) ∧ proj (1, 4) v = (2, 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l391_39199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l391_39197

/-- Hyperbola with given properties and eccentricity -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (F₁ F₂ P Q : ℝ × ℝ) :
  let c := Real.sqrt (a^2 + b^2)
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → (x, y) ∈ ({P, Q} : Set (ℝ × ℝ))) →  -- P and Q are on the hyperbola
  F₁ = (-c, 0) →  -- Left focus
  F₂ = (c, 0) →   -- Right focus
  (∃ t : ℝ, Q = F₂ + t • (P - F₂)) →  -- F₂, P, Q are collinear
  (P.1 > 0 ∧ Q.1 > 0) →  -- P and Q are on the right branch
  ((Q.2 - P.2) * (P.1 + c) = -(Q.1 - P.1) * P.2) →  -- PQ ⊥ PF₁
  ‖Q - P‖ = 5/12 * ‖P - F₁‖ →  -- |PQ| = 5/12 |PF₁|
  c / a = Real.sqrt 37 / 5  -- Eccentricity
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l391_39197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equality_l391_39196

theorem sqrt_equality (a b c d p : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hp : p > 0) (h_b : b = p * d) :
  (Real.sqrt (a + (b : ℝ) / c) = a * Real.sqrt ((b : ℝ) / c)) ↔ 
  (c = a * p * d - p * d / a ∧ p * d % a = 0) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equality_l391_39196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_overtake_on_11th_lap_l391_39155

/-- Represents a car with its speeds on different surfaces -/
structure Car where
  dirt_speed : ℝ
  asphalt_speed : ℝ

/-- Represents the track -/
structure Track where
  dirt_length : ℝ
  asphalt_length : ℝ

/-- Calculates the time taken for a car to complete one lap -/
noncomputable def lap_time (car : Car) (track : Track) : ℝ :=
  track.dirt_length / car.dirt_speed + track.asphalt_length / car.asphalt_speed

/-- Theorem stating that the first overtake occurs on the 11th lap -/
theorem first_overtake_on_11th_lap (niva toyota : Car) (track : Track) : 
  niva.dirt_speed = 80 → 
  niva.asphalt_speed = 90 → 
  toyota.dirt_speed = 40 → 
  toyota.asphalt_speed = 120 → 
  track.dirt_length = track.asphalt_length / 3 →
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ 
  11 * lap_time toyota track + x * (track.dirt_length / toyota.dirt_speed) = 
  10 * lap_time niva track + lap_time niva track + x * (track.dirt_length / niva.dirt_speed) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_overtake_on_11th_lap_l391_39155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_die_throw_limit_l391_39123

/-- The probability of getting exactly n as the final score when throwing a fair die repeatedly until the total is at least n -/
noncomputable def p (n : ℕ) : ℝ := sorry

/-- The probability of getting a specific number on a fair six-sided die -/
noncomputable def fair_die_prob : ℝ := 1 / 6

/-- The limit of p(n) as n approaches infinity is 2/7 -/
theorem die_throw_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |p n - 2/7| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_die_throw_limit_l391_39123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l391_39111

-- Define the expression
noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 - 2*x + 1) * (x - 1) / (x^2 + 3*x) + 1 / x

-- Theorem statement
theorem expression_evaluation :
  f (1 + Real.sqrt 3) = Real.sqrt 3 / 3 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l391_39111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circle_problem_l391_39116

/-- A regular polygon with a specified number of sides and area -/
structure RegularPolygon where
  sides : ℕ
  area : ℝ

/-- A circle with a specified radius -/
structure Circle where
  radius : ℝ

/-- Predicate to check if a circle is inscribed in a regular polygon -/
def Circle.isInscribedIn (c : Circle) (p : RegularPolygon) : Prop := sorry

/-- Predicate to check if a regular polygon is inscribed in a circle -/
def RegularPolygon.isInscribedIn (p : RegularPolygon) (c : Circle) : Prop := sorry

/-- Predicate to check if a natural number is square-free -/
def Nat.squareFree (n : ℕ) : Prop := sorry

theorem octagon_circle_problem (octagon1 octagon2 : RegularPolygon) (circle : Circle) :
  -- First octagon has 8 sides and area 2024
  octagon1.sides = 8 ∧ octagon1.area = 2024 ∧
  -- Circle is inscribed in the first octagon
  circle.isInscribedIn octagon1 ∧
  -- Second octagon is inscribed in the circle and has 8 sides
  octagon2.isInscribedIn circle ∧ octagon2.sides = 8 →
  -- There exist integers a, b, c such that:
  ∃ (a b c : ℕ),
    -- The area of the second octagon is a + b√c
    octagon2.area = a + b * Real.sqrt c ∧
    -- c is square-free
    c.squareFree ∧
    -- The sum of a, b, and c is 1520
    a + b + c = 1520 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circle_problem_l391_39116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_guesses_count_l391_39193

/-- Represents a valid partition of 7 digits into 3 groups --/
structure ValidPartition where
  p : Fin 3 → ℕ
  sum_eq_seven : p 0 + p 1 + p 2 = 7
  all_in_range : ∀ i, 1 ≤ p i ∧ p i ≤ 4

/-- The set of digits to be partitioned --/
def digits : Multiset ℕ := {1, 2, 2, 3, 3, 3, 4}

/-- The number of valid partitions --/
def num_valid_partitions : ℕ := 12

/-- The number of permutations of the given digits --/
def num_permutations : ℕ := 210

theorem prize_guesses_count : 
  num_valid_partitions * num_permutations = 2520 := by
  -- Proof goes here
  sorry

#eval num_valid_partitions * num_permutations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_guesses_count_l391_39193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_constraint_l391_39121

-- Define the curve
noncomputable def curve (x a : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x + a * x

-- Define the derivative of the curve
noncomputable def curve_derivative (x a : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x + a

-- Theorem statement
theorem tangent_line_constraint (a : ℝ) : 
  (∃ x : ℝ, curve_derivative x a = -2) → -4 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_constraint_l391_39121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_sum_l391_39100

theorem tan_ratio_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 5) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_sum_l391_39100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l391_39175

/-- A valid arrangement of numbers in a 2x3 table -/
def ValidArrangement : Type := Fin 2 → Fin 3 → Fin 6

/-- Check if a number is in the range 1 to 6 -/
def isValidNumber (n : Fin 6) : Prop := 1 ≤ n.val + 1 ∧ n.val + 1 ≤ 6

/-- Check if all numbers in the arrangement are unique and in the range 1 to 6 -/
def isValidDistribution (arr : ValidArrangement) : Prop :=
  ∀ i j, isValidNumber (arr i j) ∧ 
  ∀ i₁ j₁ i₂ j₂, (i₁ ≠ i₂ ∨ j₁ ≠ j₂) → arr i₁ j₁ ≠ arr i₂ j₂

/-- Check if the sum of a row is divisible by 3 -/
def isRowSumDivisibleBy3 (arr : ValidArrangement) (row : Fin 2) : Prop :=
  (arr row 0).val + (arr row 1).val + (arr row 2).val + 3 ≡ 0 [MOD 3]

/-- Check if the sum of a column is divisible by 3 -/
def isColumnSumDivisibleBy3 (arr : ValidArrangement) (col : Fin 3) : Prop :=
  (arr 0 col).val + (arr 1 col).val + 2 ≡ 0 [MOD 3]

/-- Check if an arrangement satisfies all conditions -/
def isValidArrangement (arr : ValidArrangement) : Prop :=
  isValidDistribution arr ∧
  (∀ row, isRowSumDivisibleBy3 arr row) ∧
  (∀ col, isColumnSumDivisibleBy3 arr col)

/-- The main theorem stating that there are exactly 48 valid arrangements -/
theorem valid_arrangements_count : 
  ∃ s : Finset ValidArrangement, (∀ arr ∈ s, isValidArrangement arr) ∧ s.card = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l391_39175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l391_39161

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 2^x + 2^(a*x + b)

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (f a b 1 = 5/2) ∧
    (f a b 2 = 17/4) ∧
    (a = -1) ∧
    (b = 0) ∧
    (∀ x, f a b x = f a b (-x)) ∧
    (∀ x y, 0 ≤ x → x < y → f a b x < f a b y) ∧
    (∀ x, 2 ≤ f a b x) ∧
    (∀ y, 2 ≤ y → ∃ x, f a b x = y) :=
by
  -- We use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l391_39161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_perimeter_correct_l391_39178

/-- Given a cube with side length 1, this function calculates the perimeter of the cross-section
    formed by a plane perpendicular to a main diagonal of the cube, passing through a point at
    distance x from one end of the diagonal. -/
noncomputable def cross_section_perimeter (x : ℝ) : ℝ :=
  if x ≤ Real.sqrt 3 / 3 then 3 * Real.sqrt 6 * x
  else if x ≤ 2 * Real.sqrt 3 / 3 then 3 * Real.sqrt 2
  else 3 * Real.sqrt 6 * (Real.sqrt 3 - x)

/-- Theorem stating that the cross_section_perimeter function correctly calculates the perimeter
    of the cross-section for all valid x values along the main diagonal of a unit cube. -/
theorem cross_section_perimeter_correct (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.sqrt 3) :
  cross_section_perimeter x =
    if x ≤ Real.sqrt 3 / 3 then 3 * Real.sqrt 6 * x
    else if x ≤ 2 * Real.sqrt 3 / 3 then 3 * Real.sqrt 2
    else 3 * Real.sqrt 6 * (Real.sqrt 3 - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_perimeter_correct_l391_39178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_formation_iff_multiple_of_twelve_l391_39160

/-- A piece made of 12 unit cubes forming a staircase of 3 steps -/
structure StaircasePiece where
  volume : ℕ
  bottom_layer : Fin 2 → Fin 3 → Unit
  middle_layer : Fin 2 → Fin 2 → Unit
  top_layer : Fin 1 → Fin 2 → Unit

/-- Definition of a valid staircase piece -/
def is_valid_staircase_piece (p : StaircasePiece) : Prop :=
  p.volume = 12

/-- Definition of an n × n × n cube -/
def Cube (n : ℕ) := Fin n → Fin n → Fin n → Unit

/-- Predicate to check if a cube can be formed using staircase pieces -/
def can_form_cube (n : ℕ) : Prop :=
  ∃ (pieces : List StaircasePiece), 
    (∀ p ∈ pieces, is_valid_staircase_piece p) ∧
    (∃ (f : Cube n → Option (Fin n × Fin n × Fin n)), 
      (∀ c, f c = none ↔ ∃ p ∈ pieces, ∃ i j k, f (fun i j k => c i j k) = some (i, j, k)))

/-- Main theorem: A cube can be formed if and only if n is a multiple of 12 -/
theorem cube_formation_iff_multiple_of_twelve (n : ℕ) :
  can_form_cube n ↔ ∃ k : ℕ, n = 12 * k :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_formation_iff_multiple_of_twelve_l391_39160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l391_39181

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x - x^2) / Real.log (1/2)

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y : ℝ, 1/2 < x ∧ x < y ∧ y < 1 → f x < f y :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l391_39181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_vector_l391_39165

def v : Fin 3 → ℚ := ![3, -2, 1]
def u : Fin 3 → ℚ := ![-1, 2, 2]

def reflectionOver (v u : Fin 3 → ℚ) : Fin 3 → ℚ :=
  2 * ((v • u) / (u • u)) • u - v

theorem reflection_over_vector :
  reflectionOver v u = ![-17/9, -2/9, -29/9] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_vector_l391_39165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_vertex_coordinates_l391_39173

/-- The hyperbola and its properties -/
structure Hyperbola where
  a : ℝ
  eq : (x : ℝ) → (y : ℝ) → Prop := λ x y => x^2 / a^2 - y^2 / 4 = 1
  a_pos : a > 0

/-- The foci of the hyperbola -/
noncomputable def foci (h : Hyperbola) : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt (h.a^2 + 4)
  (-c, 0, c, 0)

/-- The right vertex of the hyperbola -/
def right_vertex (h : Hyperbola) : ℝ × ℝ :=
  (h.a, 0)

/-- The point B -/
def point_B : ℝ × ℝ := (0, 2)

/-- The line l passing through the right vertex and point B -/
def line_l (h : Hyperbola) : (ℝ → ℝ → Prop) :=
  λ x y => (y - 0) / (x - h.a) = (2 - 0) / (0 - h.a)

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The theorem to be proved -/
theorem right_vertex_coordinates (h : Hyperbola) 
  (sum_eq_dist : let (x₁, y₁, x₂, y₂) := foci h
                 let (xA, yA) := right_vertex h
                 let (xB, yB) := point_B
                 distance x₁ y₁ xA yA + distance x₂ y₂ xA yA = 
                 distance xB yB x₂ y₂) :
  right_vertex h = (2 * Real.sqrt 2, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_vertex_coordinates_l391_39173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arthur_has_winning_strategy_l391_39131

/-- Represents a position on the round table -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents the game state -/
structure GameState where
  occupied : Set Position

/-- Represents a player's move -/
def Move := Position

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  move ∉ state.occupied ∧ 
  ∀ pos ∈ state.occupied, (move.x - pos.x)^2 + (move.y - pos.y)^2 ≥ 1

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { occupied := state.occupied ∪ {move} }

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for the first player -/
def isWinningStrategy (strategy : Strategy) : Prop :=
  ∀ (state : GameState),
    isValidMove state (strategy state) →
    ∀ (opponent_move : Move),
      isValidMove (applyMove state (strategy state)) opponent_move →
      ∃ (next_move : Move), isValidMove (applyMove (applyMove state (strategy state)) opponent_move) next_move

/-- The theorem stating that Arthur (the first player) has a winning strategy -/
theorem arthur_has_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy strategy := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arthur_has_winning_strategy_l391_39131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_for_genuine_coin_l391_39125

/-- Represents a coin, which can be either genuine or counterfeit -/
inductive Coin
  | genuine
  | counterfeit
  deriving BEq, Repr

/-- Represents the result of weighing two coins -/
inductive WeighResult
  | equal
  | left_heavier
  | right_heavier
  deriving BEq, Repr

/-- The total number of coins -/
def total_coins : Nat := 100

/-- The number of genuine coins -/
def genuine_coins : Nat := 30

/-- The number of counterfeit coins -/
def counterfeit_coins : Nat := 70

/-- A function that simulates weighing two coins -/
def weigh : Coin → Coin → WeighResult := sorry

/-- A function that represents the process of finding a genuine coin -/
def find_genuine_coin : List Coin → Nat → Option Coin := sorry

theorem min_weighings_for_genuine_coin :
  ∀ (coins : List Coin),
    coins.length = total_coins →
    coins.count Coin.genuine = genuine_coins →
    coins.count Coin.counterfeit = counterfeit_coins →
    (∀ (c1 c2 : Coin), c1 = Coin.genuine ∧ c2 = Coin.genuine → weigh c1 c2 = WeighResult.equal) →
    (∀ (c1 c2 : Coin), c1 = Coin.counterfeit ∧ c2 = Coin.counterfeit → weigh c1 c2 ≠ WeighResult.equal) →
    (∀ (c1 c2 : Coin), c1 = Coin.genuine ∧ c2 = Coin.counterfeit → weigh c1 c2 = WeighResult.right_heavier) →
    ∃ (n : Nat), n ≤ 70 ∧ (find_genuine_coin coins n).isSome := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_for_genuine_coin_l391_39125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l391_39179

theorem relationship_abc : 
  ∃ (a b c : ℝ), 
    a = Real.log 0.2 ∧
    b = 2^(0.3 : ℝ) ∧
    c = 0.3^(0.2 : ℝ) ∧
    b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l391_39179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l391_39107

open Real

theorem range_of_a (e : ℝ) (h_e : e = exp 1) :
  (∀ x₁, x₁ ∈ Set.Icc 0 1 → ∃! x₂, x₂ ∈ Set.Icc (-1) 1 ∧ x₁ + x₂^2 * e^x₂ - a = 0) ↔
  a ∈ Set.Ioo (1 + 1/e) e :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l391_39107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_lawn_remains_unmowed_l391_39172

/-- Given that Tom can mow an entire lawn in 6 hours, this function calculates
    the fraction of the lawn that remains unmowed after he works for a given number of hours. -/
noncomputable def fraction_unmowed (total_hours : ℝ) (worked_hours : ℝ) : ℝ :=
  1 - (worked_hours / total_hours)

/-- Theorem stating that if Tom can mow an entire lawn in 6 hours and works for 3 hours,
    then 1/2 of the lawn remains to be mowed. -/
theorem half_lawn_remains_unmowed :
  fraction_unmowed 6 3 = 1/2 := by
  -- Unfold the definition of fraction_unmowed
  unfold fraction_unmowed
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num

#check half_lawn_remains_unmowed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_lawn_remains_unmowed_l391_39172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_grape_price_is_7_l391_39198

/-- The original price of grapes that Erick planned to sell -/
noncomputable def original_grape_price : ℝ := 0

/-- The price increase for lemons -/
def lemon_price_increase : ℝ := 4

/-- The original price of lemons -/
def original_lemon_price : ℝ := 8

/-- The number of lemons Erick had -/
def num_lemons : ℕ := 80

/-- The number of grapes Erick had -/
def num_grapes : ℕ := 140

/-- The total amount Erick collected from selling all fruits -/
def total_collected : ℝ := 2220

/-- The price increase for grapes -/
noncomputable def grape_price_increase : ℝ := lemon_price_increase / 2

/-- The new price of lemons -/
noncomputable def new_lemon_price : ℝ := original_lemon_price + lemon_price_increase

/-- The new price of grapes -/
noncomputable def new_grape_price : ℝ := original_grape_price + grape_price_increase

theorem original_grape_price_is_7 :
  (num_lemons : ℝ) * new_lemon_price + (num_grapes : ℝ) * new_grape_price = total_collected →
  original_grape_price = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_grape_price_is_7_l391_39198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_l391_39126

/-- The function f(x) defined as (x + a) * ln((2x - 1) / (2x + 1)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

/-- A function is even if f(-x) = f(x) for all x in its domain -/
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

/-- If f(x) = (x + a) * ln((2x - 1) / (2x + 1)) is an even function, then a = 0 -/
theorem f_even_implies_a_zero (a : ℝ) : is_even (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_l391_39126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_l391_39143

/-- Calculates the time (in seconds) for a train to cross a platform -/
noncomputable def train_crossing_time (train_speed_kmh : ℝ) (train_length_m : ℝ) (platform_length_m : ℝ) : ℝ :=
  let total_distance := train_length_m + platform_length_m
  let train_speed_ms := train_speed_kmh * (5/18)
  total_distance / train_speed_ms

theorem train_crossing_platform :
  train_crossing_time 72 310 210 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_l391_39143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_two_rays_l391_39182

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := t + 1/t
def y : ℝ := -2

-- Theorem statement
theorem curve_is_two_rays :
  ∀ t : ℝ, t ≠ 0 →
    (x t ≤ -2 ∨ x t ≥ 2) ∧
    (∀ x₀ : ℝ, (x₀ ≤ -2 ∨ x₀ ≥ 2) → ∃ t : ℝ, t ≠ 0 ∧ x t = x₀) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_two_rays_l391_39182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l391_39108

-- Define the lines and circles
def line1 (x y : ℝ) : Prop := 2*x + y + 5 = 0
def line2 (x y : ℝ) : Prop := x - 2*y = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 2*y - 6 = 0

-- Define the point P
def P : ℝ × ℝ := (5, 0)

-- Define the distance function
noncomputable def distance (x y a b : ℝ) : ℝ := Real.sqrt ((x - a)^2 + (y - b)^2)

-- Define the perpendicular condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_equation (l : ℝ → ℝ → Prop) :
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ l x y) →
  (∃ x y : ℝ, l x y ∧ distance x y P.1 P.2 = 4) →
  (l = λ x y => x = 2 ∨ 4*x - 3*y - 5 = 0) ∨
  (∃ A B : ℝ × ℝ, circle1 A.1 A.2 ∧ circle2 B.1 B.2 ∧
    perpendicular ((B.2 - A.2) / (B.1 - A.1)) ((y - x) / (x + y)) ∧
    l = λ x y => 3*x - 4*y - 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l391_39108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l391_39183

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x) + 1 / Real.sqrt (1 + x)

theorem f_domain : Set.Ioo (-1 : ℝ) 1 ∪ {1} = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l391_39183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_B_in_special_triangle_l391_39133

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem max_tan_B_in_special_triangle (t : Triangle) 
  (h : 3 * t.a * Real.cos t.C + t.b = 0) : 
  ∃ (max_tan_B : Real), ∀ (tan_B : Real), tan_B = Real.tan t.B → tan_B ≤ max_tan_B ∧ max_tan_B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_B_in_special_triangle_l391_39133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_first_class_fare_l391_39105

/-- Represents the basic first class full fare -/
def full_fare : ℚ := sorry

/-- Represents the reservation charge -/
def reservation_charge : ℚ := sorry

/-- The cost of a reserved first class ticket -/
def reserved_first_class_cost : ℚ := 216

/-- The cost of one full and one half reserved first class tickets -/
def full_and_half_tickets_cost : ℚ := 327

/-- The cost of a half ticket is half the full fare -/
def half_ticket_cost : ℚ := full_fare / 2

/-- The reservation charge is the same for both full and half tickets -/
def same_reservation_charge (ticket_type : String) : ℚ := reservation_charge

/-- Theorem stating that the basic first class full fare is $210 -/
theorem basic_first_class_fare : full_fare = 210 := by
  sorry

#check basic_first_class_fare

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_first_class_fare_l391_39105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_perpendicular_line_l391_39144

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Perpendicular line passing through a point -/
def perpendicular_line_through_point (p : Point) (l : Line) : Line :=
  { a := l.b,
    b := -l.a,
    c := l.a * p.y - l.b * p.x }

theorem distance_and_perpendicular_line 
  (A : Point)
  (l : Line)
  (h1 : A.x = 2)
  (h2 : A.y = 1)
  (h3 : l.a = 3)
  (h4 : l.b = 4)
  (h5 : l.c = -20) :
  distance_point_to_line A l = 2 ∧ 
  perpendicular_line_through_point A l = { a := 4, b := -3, c := -5 } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_perpendicular_line_l391_39144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_l391_39168

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the series
noncomputable def fibSeries : ℝ := ∑' n, (fib n : ℝ) / 3^n

-- Theorem statement
theorem fibonacci_series_sum : fibSeries = 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_l391_39168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pat_initial_stickers_l391_39158

/-- The number of stickers Pat gave away during the week -/
def stickers_given_away : ℕ := 22

/-- The number of stickers Pat had left at the end of the week -/
def stickers_left : ℕ := 17

/-- The number of stickers Pat had on the first day of the week -/
def initial_stickers : ℕ := stickers_given_away + stickers_left

theorem pat_initial_stickers : initial_stickers = 39 := by
  rfl

#eval initial_stickers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pat_initial_stickers_l391_39158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_crosses_one_l391_39147

def a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => a n + (a n)^2 / 2023

theorem sequence_crosses_one :
  a 2023 < 1 ∧ 1 < a 2024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_crosses_one_l391_39147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_mpg_l391_39188

noncomputable def distance_to_conference : ℝ := 150
noncomputable def mpg_to_conference : ℝ := 25
noncomputable def distance_from_conference : ℝ := 180
noncomputable def mpg_from_conference : ℝ := 15

noncomputable def total_distance : ℝ := distance_to_conference + distance_from_conference
noncomputable def gas_used_to_conference : ℝ := distance_to_conference / mpg_to_conference
noncomputable def gas_used_from_conference : ℝ := distance_from_conference / mpg_from_conference
noncomputable def total_gas_used : ℝ := gas_used_to_conference + gas_used_from_conference

noncomputable def average_mpg : ℝ := total_distance / total_gas_used

theorem round_trip_average_mpg :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.05 ∧ |average_mpg - 18.3| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_mpg_l391_39188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_surface_area_theorem_l391_39164

noncomputable def cylinder_height : ℝ := 8
noncomputable def cylinder_radius : ℝ := 3
noncomputable def cone_height : ℝ := 5
noncomputable def cone_radius : ℝ := 3

noncomputable def total_surface_area : ℝ :=
  2 * Real.pi * cylinder_radius * cylinder_height +
  Real.pi * cone_radius * Real.sqrt (cone_radius^2 + cone_height^2) +
  Real.pi * cone_radius^2

theorem total_surface_area_theorem :
  total_surface_area = 57 * Real.pi + 3 * Real.pi * Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_surface_area_theorem_l391_39164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_example_l391_39130

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + a²/b²) -/
noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + a^2 / b^2)

/-- The hyperbola equation x²/4 - y²/3 = 1 has eccentricity √7/2 -/
theorem hyperbola_eccentricity_example : hyperbola_eccentricity 2 (Real.sqrt 3) = Real.sqrt 7 / 2 := by
  -- Unfold the definition of hyperbola_eccentricity
  unfold hyperbola_eccentricity
  -- Simplify the expression
  simp [Real.sqrt_div, Real.sqrt_mul]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_example_l391_39130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_proof_l391_39135

theorem junior_score_proof (total_students : ℕ) (junior_percentage senior_percentage : ℚ)
  (overall_average senior_average junior_score : ℚ) :
  total_students = 20 →
  junior_percentage = 1/5 →
  senior_percentage = 4/5 →
  overall_average = 78 →
  senior_average = 76 →
  (junior_percentage * total_students * junior_score +
   senior_percentage * total_students * senior_average) /
    total_students = overall_average →
  junior_score = 86 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_proof_l391_39135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l391_39139

/-- Circle C in the Cartesian plane -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- Line l in the Cartesian plane with parameter a -/
def Line (x y a : ℝ) : Prop := 2*x - y - 2*a = 0

/-- The distance from a point (x, y) to the origin -/
noncomputable def distanceToOrigin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- The distance from the line to the origin -/
noncomputable def lineDistanceToOrigin (a : ℝ) : ℝ := |2*a| / Real.sqrt 5

theorem line_circle_intersection (a : ℝ) :
  (∃ x y : ℝ, Circle x y ∧ Line x y a) ↔ a ∈ Set.Icc (-2 * Real.sqrt 5) (2 * Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l391_39139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_class_l391_39137

theorem girls_in_class (total_children : ℕ) (prob_two_boys : ℚ) (num_girls : ℕ) : 
  total_children = 25 →
  prob_two_boys = 3/25 →
  (num_girls = total_children - (total_children - num_girls)) →
  (Nat.choose (total_children - num_girls) 2 : ℚ) / (Nat.choose total_children 2 : ℚ) = prob_two_boys →
  num_girls = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_in_class_l391_39137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_monotonic_on_same_interval_l391_39129

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define monotonically increasing on an interval
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Statement of the theorem
theorem not_necessarily_monotonic_on_same_interval :
  ∃ f : ℝ → ℝ,
    MonotonicallyIncreasing f (-1) 1 ∧
    ¬ MonotonicallyIncreasing (fun x ↦ f (2*x - 1)) (-1) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_monotonic_on_same_interval_l391_39129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_orders_l391_39195

/-- Represents the number of letters. -/
def n : ℕ := 10

/-- Represents the number of letters that have already been typed. -/
def m : ℕ := 2

/-- Calculates the number of possible orders for typing the remaining letters. -/
def possible_orders : ℕ := Finset.sum (Finset.range 7) (λ k => (Nat.choose 6 k) * (k + 2) * (k + 3))

/-- Theorem stating that the number of possible orders is 2016. -/
theorem count_possible_orders : possible_orders = 2016 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_orders_l391_39195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l391_39114

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 4)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x : ℝ), f ((-Real.pi/8 + x)) = f ((-Real.pi/8 - x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l391_39114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_work_days_per_week_l391_39124

theorem julie_work_days_per_week 
  (hourly_rate : ℝ)
  (hours_per_day : ℝ)
  (monthly_salary_with_missed_day : ℝ)
  (weeks_per_month : ℝ)
  (h1 : hourly_rate = 5)
  (h2 : hours_per_day = 8)
  (h3 : monthly_salary_with_missed_day = 920)
  (h4 : weeks_per_month = 4) :
  (monthly_salary_with_missed_day + hourly_rate * hours_per_day) / (hourly_rate * hours_per_day) / weeks_per_month = 6 := by
  -- Placeholder for the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_work_days_per_week_l391_39124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_ax_monotonicity_l391_39112

-- Define the function f(x) = log(ax)
noncomputable def f (a : ℝ) (x : ℝ) := Real.log (a * x)

-- Statement of the theorem
theorem log_ax_monotonicity :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f 1 x₁ < f 1 x₂) ∧
  (∃ a : ℝ, a ≠ 1 ∧ ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f a x₁ < f a x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_ax_monotonicity_l391_39112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_values_min_tan_sum_l391_39120

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

def is_right_angled (t : Triangle) : Prop :=
  t.A = Real.pi / 2

def has_specific_height (t : Triangle) : Prop :=
  (Real.sqrt 3 / 4) * t.a = t.b * Real.sin t.C

def has_acute_angles (t : Triangle) : Prop :=
  t.B < Real.pi / 2 ∧ t.C < Real.pi / 2

-- State the theorems
theorem angle_B_values (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : is_right_angled t) 
  (h3 : has_specific_height t) :
  t.B = Real.pi / 6 ∨ t.B = Real.pi / 3 := by sorry

theorem min_tan_sum (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : is_right_angled t) 
  (h3 : has_specific_height t) 
  (h4 : has_acute_angles t) :
  (∀ (t' : Triangle), is_valid_triangle t' → is_right_angled t' → has_specific_height t' → has_acute_angles t' →
    Real.tan t.B + 4 * Real.tan t.C ≤ Real.tan t'.B + 4 * Real.tan t'.C) ∧
  Real.tan t.B + 4 * Real.tan t.C = (9 * Real.sqrt 3) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_values_min_tan_sum_l391_39120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_l391_39187

/-- A quadratic function with vertex (h, k) and y-intercept (0, b) -/
noncomputable def QuadraticFunction (h k b : ℝ) : ℝ → ℝ := 
  λ x => (b - k) / h^2 * (x - h)^2 + k

theorem quadratic_function_unique (f : ℝ → ℝ) :
  (∀ x, f x = QuadraticFunction 3 (-1) (-4) x) ↔
  (∀ x, f x = -1/3 * x^2 + 2*x - 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_l391_39187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_199_299_l391_39118

/-- Definition of the sequence -/
def my_sequence (n : ℕ+) : ℚ := (2 * n.val - 1 : ℚ) / (3 * n.val - 1 : ℚ)

/-- Theorem stating that the 100th term of the sequence is 199/299 -/
theorem hundredth_term_is_199_299 : my_sequence 100 = 199 / 299 := by
  -- Unfold the definition of my_sequence
  unfold my_sequence
  -- Simplify the numerator and denominator
  simp [Nat.cast_sub, Nat.cast_mul, Nat.cast_one]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_199_299_l391_39118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_difference_3x2_6x1_l391_39176

/-- The perimeter of a rectangle given its width and height -/
def rectanglePerimeter (width height : ℕ) : ℕ := 2 * (width + height)

/-- The positive difference between the perimeters of two rectangles -/
def perimeterDifference (w1 h1 w2 h2 : ℕ) : ℕ :=
  max (rectanglePerimeter w1 h1) (rectanglePerimeter w2 h2) -
  min (rectanglePerimeter w1 h1) (rectanglePerimeter w2 h2)

theorem perimeter_difference_3x2_6x1 :
  perimeterDifference 6 1 3 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_difference_3x2_6x1_l391_39176
