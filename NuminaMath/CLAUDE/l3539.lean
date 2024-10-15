import Mathlib

namespace NUMINAMATH_CALUDE_tank_final_volume_l3539_353984

def tank_problem (tank_capacity : ℝ) (initial_fill_ratio : ℝ) (empty_ratio : ℝ) (refill_ratio : ℝ) : ℝ :=
  let initial_volume := tank_capacity * initial_fill_ratio
  let emptied_volume := initial_volume * empty_ratio
  let remaining_volume := initial_volume - emptied_volume
  let refilled_volume := remaining_volume * refill_ratio
  remaining_volume + refilled_volume

theorem tank_final_volume :
  tank_problem 8000 (3/4) (40/100) (30/100) = 4680 := by
  sorry

end NUMINAMATH_CALUDE_tank_final_volume_l3539_353984


namespace NUMINAMATH_CALUDE_smallest_positive_number_l3539_353904

theorem smallest_positive_number : 
  let a := 8 - 2 * Real.sqrt 17
  let b := 2 * Real.sqrt 17 - 8
  let c := 25 - 7 * Real.sqrt 5
  let d := 40 - 9 * Real.sqrt 2
  let e := 9 * Real.sqrt 2 - 40
  (0 < b) ∧ 
  (a ≤ b ∨ a ≤ 0) ∧ 
  (b ≤ c ∨ c ≤ 0) ∧ 
  (b ≤ d ∨ d ≤ 0) ∧ 
  (b ≤ e ∨ e ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_number_l3539_353904


namespace NUMINAMATH_CALUDE_lines_parallel_l3539_353937

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

theorem lines_parallel : 
  let line1 : Line := ⟨-1, 0⟩
  let line2 : Line := ⟨-1, 6⟩
  parallel line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_l3539_353937


namespace NUMINAMATH_CALUDE_tangent_line_and_triangle_area_l3539_353969

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x

-- Theorem statement
theorem tangent_line_and_triangle_area :
  let P : ℝ × ℝ := (1, -1)
  -- Condition: P is on the graph of f
  (f P.1 = P.2) →
  -- Claim 1: Equation of the tangent line
  (∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ x - y - 2 = 0) ∧
  -- Claim 2: Area of the triangle
  (∃ A : ℝ, A = 2 ∧
    ∀ x₁ y₁ x₂ y₂,
      (x₁ - y₁ - 2 = 0 ∧ y₁ = 0) →
      (x₂ - y₂ - 2 = 0 ∧ x₂ = 0) →
      A = (1/2) * x₁ * (-y₂)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_triangle_area_l3539_353969


namespace NUMINAMATH_CALUDE_triangle_inequality_l3539_353976

theorem triangle_inequality (α β γ : ℝ) (h_a l_a r R : ℝ) 
  (h1 : h_a / l_a = Real.cos ((β - γ) / 2))
  (h2 : 2 * r / R = 8 * Real.sin (α / 2) * Real.sin (β / 2) * Real.sin (γ / 2)) :
  h_a / l_a ≥ Real.sqrt (2 * r / R) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3539_353976


namespace NUMINAMATH_CALUDE_equation_solution_l3539_353941

theorem equation_solution : ∃ x : ℝ, (x - 1) / (2 * x + 1) = 1 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3539_353941


namespace NUMINAMATH_CALUDE_find_m_l3539_353917

theorem find_m (A B : Set ℕ) (m : ℕ) : 
  A = {1, 3, m} → 
  B = {3, 4} → 
  A ∪ B = {1, 2, 3, 4} → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_find_m_l3539_353917


namespace NUMINAMATH_CALUDE_division_problem_l3539_353927

theorem division_problem (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3539_353927


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_50_l3539_353991

/-- The cost of paint per kg, given the coverage rate and the cost to paint a cube. -/
theorem paint_cost_per_kg (coverage_rate : ℝ) (cube_side : ℝ) (total_cost : ℝ) : ℝ :=
  let surface_area := 6 * cube_side * cube_side
  let paint_needed := surface_area / coverage_rate
  total_cost / paint_needed

/-- The cost of paint per kg is 50, given the specified conditions. -/
theorem paint_cost_is_50 : paint_cost_per_kg 20 20 6000 = 50 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_50_l3539_353991


namespace NUMINAMATH_CALUDE_quadratic_equations_solution_l3539_353992

def A (p : ℝ) : Set ℝ := {x | x^2 + p*x - 2 = 0}
def B (q r : ℝ) : Set ℝ := {x | x^2 + q*x + r = 0}

theorem quadratic_equations_solution (p q r : ℝ) :
  (A p ∪ B q r = {-2, 1, 5}) ∧
  (A p ∩ B q r = {-2}) →
  p = -1 ∧ q = -3 ∧ r = -10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solution_l3539_353992


namespace NUMINAMATH_CALUDE_expression_evaluation_l3539_353996

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x = 789 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3539_353996


namespace NUMINAMATH_CALUDE_max_ab_value_l3539_353978

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃! x, x^2 + Real.sqrt a * x - b + 1/4 = 0) → 
  ∀ c, a * b ≤ c → c ≤ 1/16 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l3539_353978


namespace NUMINAMATH_CALUDE_trig_identity_l3539_353972

theorem trig_identity (x : Real) (h : Real.tan x = -1/2) : 
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3539_353972


namespace NUMINAMATH_CALUDE_polygon_square_equal_area_l3539_353931

/-- Given a polygon with perimeter 800 cm and each side tangent to a circle of radius 100 cm,
    the side length of a square with equal area is 200 cm. -/
theorem polygon_square_equal_area (polygon_perimeter : ℝ) (circle_radius : ℝ) :
  polygon_perimeter = 800 ∧ circle_radius = 100 →
  ∃ (square_side : ℝ),
    square_side = 200 ∧
    square_side ^ 2 = (polygon_perimeter * circle_radius) / 2 := by
  sorry

end NUMINAMATH_CALUDE_polygon_square_equal_area_l3539_353931


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l3539_353958

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x - 1)

-- State the theorem
theorem f_decreasing_interval :
  ∀ x y : ℝ, x > 0 → y > 0 → 
  (Real.log (x + y) = Real.log x + Real.log y) →
  (∀ a b : ℝ, a > 1 → b > 1 → a < b → f a > f b) :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l3539_353958


namespace NUMINAMATH_CALUDE_company_profits_l3539_353970

theorem company_profits (revenue_prev : ℝ) (profit_prev : ℝ) (revenue_2009 : ℝ) (profit_2009 : ℝ) :
  revenue_2009 = 0.8 * revenue_prev →
  profit_2009 = 0.16 * revenue_2009 →
  profit_2009 = 1.28 * profit_prev →
  profit_prev = 0.1 * revenue_prev :=
by sorry

end NUMINAMATH_CALUDE_company_profits_l3539_353970


namespace NUMINAMATH_CALUDE_alices_number_l3539_353956

theorem alices_number (n : ℕ) : 
  180 ∣ n → 75 ∣ n → 900 ≤ n → n < 3000 → n = 900 ∨ n = 1800 ∨ n = 2700 := by
  sorry

end NUMINAMATH_CALUDE_alices_number_l3539_353956


namespace NUMINAMATH_CALUDE_extremum_and_minimum_l3539_353960

-- Define the function f(x) = x³ - 3ax - 1
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x - 1

-- State the theorem
theorem extremum_and_minimum (a : ℝ) :
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| ∧ |h| < ε → f a (-1 + h) ≤ f a (-1) ∨ f a (-1 + h) ≥ f a (-1)) →
  a = 1 ∧ 
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) (1 : ℝ) → f a x ≥ -3 :=
by sorry

end NUMINAMATH_CALUDE_extremum_and_minimum_l3539_353960


namespace NUMINAMATH_CALUDE_subset_difference_theorem_l3539_353974

theorem subset_difference_theorem (n k m : ℕ) (A : Finset ℕ) 
  (h1 : k ≥ 2)
  (h2 : n ≤ m)
  (h3 : m < ((2 * k - 1) * n) / k)
  (h4 : A.card = n)
  (h5 : ∀ a ∈ A, a ≤ m) :
  ∀ x : ℤ, 0 < x ∧ x < n / (k - 1) → 
    ∃ a a' : ℕ, a ∈ A ∧ a' ∈ A ∧ (a : ℤ) - (a' : ℤ) = x :=
by sorry

end NUMINAMATH_CALUDE_subset_difference_theorem_l3539_353974


namespace NUMINAMATH_CALUDE_point_on_line_l3539_353986

/-- Given two points on a line and a third point with a known y-coordinate,
    prove that the x-coordinate of the third point is -6. -/
theorem point_on_line (x : ℝ) :
  let p1 : ℝ × ℝ := (0, 8)
  let p2 : ℝ × ℝ := (-4, 0)
  let p3 : ℝ × ℝ := (x, -4)
  (p3.2 - p1.2) / (p3.1 - p1.1) = (p2.2 - p1.2) / (p2.1 - p1.1) →
  x = -6 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l3539_353986


namespace NUMINAMATH_CALUDE_solve_for_b_l3539_353946

-- Define the @ operation
def at_op (k : ℕ) (j : ℕ) : ℕ := (List.range j).foldl (λ acc i => acc * (k + i)) k

-- Define the problem parameters
def a : ℕ := 2020
def q : ℚ := 1/2

-- Theorem statement
theorem solve_for_b (b : ℕ) (h : (a : ℚ) / b = q) : b = 4040 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3539_353946


namespace NUMINAMATH_CALUDE_ball_probability_theorem_l3539_353918

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing k balls of a specific color from a bag -/
def prob_draw (bag : Bag) (color : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose color k : ℚ) / (Nat.choose (bag.white + bag.black) k)

/-- The probability of drawing all black balls from both bags -/
def prob_all_black (bagA bagB : Bag) : ℚ :=
  (prob_draw bagA bagA.black 2) * (prob_draw bagB bagB.black 2)

/-- The probability of drawing exactly one white ball from both bags -/
def prob_one_white (bagA bagB : Bag) : ℚ :=
  (prob_draw bagA bagA.black 2) * (prob_draw bagB bagB.white 1) * (prob_draw bagB bagB.black 1) +
  (prob_draw bagA bagA.white 1) * (prob_draw bagA bagA.black 1) * (prob_draw bagB bagB.black 2)

theorem ball_probability_theorem (bagA bagB : Bag) 
  (hA : bagA = ⟨2, 4⟩) (hB : bagB = ⟨1, 4⟩) : 
  prob_all_black bagA bagB = 6/25 ∧ prob_one_white bagA bagB = 12/25 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_theorem_l3539_353918


namespace NUMINAMATH_CALUDE_ellipse_area_irrational_l3539_353949

-- Define the major and minor radii as rational numbers
variable (a b : ℚ)

-- Define π as an irrational constant
noncomputable def π : ℝ := Real.pi

-- Define the area of the ellipse
noncomputable def ellipseArea (a b : ℚ) : ℝ := π * (a * b)

-- Theorem statement
theorem ellipse_area_irrational (a b : ℚ) (h1 : a > 0) (h2 : b > 0) :
  Irrational (ellipseArea a b) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_area_irrational_l3539_353949


namespace NUMINAMATH_CALUDE_polynomial_expansion_simplification_l3539_353933

theorem polynomial_expansion_simplification (x : ℝ) : 
  (x^3 - 3*x^2 + (1/2)*x - 1) * (x^2 + 3*x + 3/2) = 
  x^5 - (15/2)*x^3 - 4*x^2 - (9/4)*x - 3/2 := by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_simplification_l3539_353933


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3539_353910

theorem solve_exponential_equation :
  ∃! y : ℝ, (64 : ℝ)^(3*y) = (16 : ℝ)^(4*y - 5) :=
by
  -- The unique solution is y = -10
  use -10
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3539_353910


namespace NUMINAMATH_CALUDE_convex_polygon_24_sides_diagonals_l3539_353993

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem convex_polygon_24_sides_diagonals :
  num_diagonals 24 = 126 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_24_sides_diagonals_l3539_353993


namespace NUMINAMATH_CALUDE_characterization_of_solution_l3539_353948

/-- A real-valued function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem stating that any function satisfying the equation must be of the form ax^2 + bx -/
theorem characterization_of_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x :=
sorry

end NUMINAMATH_CALUDE_characterization_of_solution_l3539_353948


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_b_value_l3539_353990

-- Define the hyperbola equation
def is_hyperbola (x y b : ℝ) : Prop := x^2 - y^2/b^2 = 1

-- Define the asymptote equation
def is_asymptote (x y : ℝ) : Prop := y = 2*x

theorem hyperbola_asymptote_b_value (b : ℝ) :
  b > 0 →
  (∃ x y : ℝ, is_hyperbola x y b ∧ is_asymptote x y) →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_b_value_l3539_353990


namespace NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l3539_353925

theorem shaded_area_of_concentric_circles (r1 r2 r3 : ℝ) (shaded unshaded : ℝ) : 
  r1 = 4 → r2 = 5 → r3 = 6 →
  shaded + unshaded = π * (r1^2 + r2^2 + r3^2) →
  shaded = (3/7) * unshaded →
  shaded = (1617 * π) / 70 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l3539_353925


namespace NUMINAMATH_CALUDE_train_speed_l3539_353988

/-- Given a train of length 300 meters that crosses an electric pole in 20 seconds,
    prove that its speed is 15 meters per second. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 300) (h2 : crossing_time = 20) :
  train_length / crossing_time = 15 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l3539_353988


namespace NUMINAMATH_CALUDE_inequality_proof_l3539_353935

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (1/a) + (1/b) + (9/c) + (25/d) ≥ 100/(a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3539_353935


namespace NUMINAMATH_CALUDE_prime_product_minus_sum_l3539_353901

theorem prime_product_minus_sum : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ 
  p ≠ q ∧ 
  4 < p ∧ p < 18 ∧ 
  4 < q ∧ q < 18 ∧ 
  p * q - (p + q) = 119 := by
sorry

end NUMINAMATH_CALUDE_prime_product_minus_sum_l3539_353901


namespace NUMINAMATH_CALUDE_triangle_construction_exists_l3539_353915

-- Define the necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

def Line (p q : Point) : Set Point :=
  {r : Point | ∃ t : ℝ, r = Point.mk (p.x + t * (q.x - p.x)) (p.y + t * (q.y - p.y))}

def CircumscribedCircle (a b c : Point) : Set Point :=
  sorry -- Definition of circumscribed circle

def Diameter (circle : Set Point) (p q : Point) : Prop :=
  sorry -- Definition of diameter in a circle

def FirstPicturePlane : Set Point :=
  sorry -- Definition of the first picture plane

-- State the theorem
theorem triangle_construction_exists (a b d : Point) (α : ℝ) 
  (h1 : d ∈ Line a b) : 
  ∃ c : Point, 
    c ∈ FirstPicturePlane ∧ 
    d ∈ Line a b ∧ 
    Diameter (CircumscribedCircle a b c) c d := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_exists_l3539_353915


namespace NUMINAMATH_CALUDE_range_of_k_l3539_353979

-- Define set A
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}

-- Define set B
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < k + 1}

-- Define the complement of A in ℝ
def C_R_A : Set ℝ := {x | ¬(x ∈ A)}

-- Theorem statement
theorem range_of_k (k : ℝ) : 
  (C_R_A ∩ B k).Nonempty → 0 < k ∧ k < 3 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_k_l3539_353979


namespace NUMINAMATH_CALUDE_tan_sin_function_property_l3539_353940

/-- Given a function f(x) = tan x + sin x + 1, prove that if f(b) = 2, then f(-b) = 0 -/
theorem tan_sin_function_property (b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.tan x + Real.sin x + 1
  f b = 2 → f (-b) = 0 := by
sorry

end NUMINAMATH_CALUDE_tan_sin_function_property_l3539_353940


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l3539_353998

theorem estimate_sqrt_expression :
  6 < Real.sqrt 5 * (2 * Real.sqrt 5 - Real.sqrt 2) ∧
  Real.sqrt 5 * (2 * Real.sqrt 5 - Real.sqrt 2) < 7 :=
by sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l3539_353998


namespace NUMINAMATH_CALUDE_remainder_problem_l3539_353999

theorem remainder_problem (n : ℕ) 
  (h1 : n^3 % 7 = 3) 
  (h2 : n^4 % 7 = 2) : 
  n % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3539_353999


namespace NUMINAMATH_CALUDE_monic_quartic_with_given_roots_l3539_353923

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 10*x^3 + 17*x^2 + 18*x - 12

-- Theorem statement
theorem monic_quartic_with_given_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 + (-10)*x^3 + 17*x^2 + 18*x + (-12)) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- 3+√5 is a root
  p (3 + Real.sqrt 5) = 0 ∧
  -- 2-√7 is a root
  p (2 - Real.sqrt 7) = 0 :=
by sorry

end NUMINAMATH_CALUDE_monic_quartic_with_given_roots_l3539_353923


namespace NUMINAMATH_CALUDE_prime_quadruplet_l3539_353977

theorem prime_quadruplet (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
  p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧
  p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882 →
  ((p₁, p₂, p₃, p₄) = (2, 5, 19, 37) ∨ 
   (p₁, p₂, p₃, p₄) = (2, 11, 19, 31) ∨ 
   (p₁, p₂, p₃, p₄) = (2, 13, 19, 29)) :=
by sorry

end NUMINAMATH_CALUDE_prime_quadruplet_l3539_353977


namespace NUMINAMATH_CALUDE_ab_nonzero_sufficient_not_necessary_for_a_nonzero_l3539_353920

theorem ab_nonzero_sufficient_not_necessary_for_a_nonzero (a b : ℝ) :
  (∀ a b : ℝ, ab ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ ab = 0) :=
by sorry

end NUMINAMATH_CALUDE_ab_nonzero_sufficient_not_necessary_for_a_nonzero_l3539_353920


namespace NUMINAMATH_CALUDE_factorial_100_trailing_zeros_l3539_353973

-- Define a function to count trailing zeros in a factorial
def trailingZeros (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem factorial_100_trailing_zeros :
  trailingZeros 100 = 24 := by sorry

end NUMINAMATH_CALUDE_factorial_100_trailing_zeros_l3539_353973


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3539_353919

/-- Given a geometric sequence {a_n} satisfying the condition
    a_4 · a_6 + 2a_5 · a_7 + a_6 · a_8 = 36, prove that a_5 + a_7 = ±6 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_condition : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) :
  (a 5 + a 7 = 6) ∨ (a 5 + a 7 = -6) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3539_353919


namespace NUMINAMATH_CALUDE_multiple_implies_equal_l3539_353929

theorem multiple_implies_equal (a b : ℕ+) (h : ∃ k : ℕ, (a^2 + a*b + 1 : ℕ) = k * (b^2 + a*b + 1)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_multiple_implies_equal_l3539_353929


namespace NUMINAMATH_CALUDE_sandwich_cost_calculation_l3539_353971

-- Define the costs and quantities
def selling_price : ℚ := 3
def bread_cost : ℚ := 0.15
def ham_cost : ℚ := 0.25
def cheese_cost : ℚ := 0.35
def mayo_cost : ℚ := 0.10
def lettuce_cost : ℚ := 0.05
def tomato_cost : ℚ := 0.08
def packaging_cost : ℚ := 0.02

def bread_qty : ℕ := 2
def ham_qty : ℕ := 2
def cheese_qty : ℕ := 2
def mayo_qty : ℕ := 1
def lettuce_qty : ℕ := 1
def tomato_qty : ℕ := 2

def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.05

-- Define the theorem
theorem sandwich_cost_calculation :
  let ingredient_cost := bread_cost * bread_qty + ham_cost * ham_qty + cheese_cost * cheese_qty +
                         mayo_cost * mayo_qty + lettuce_cost * lettuce_qty + tomato_cost * tomato_qty
  let discount := (ham_cost * ham_qty + cheese_cost * cheese_qty) * discount_rate
  let adjusted_cost := ingredient_cost - discount + packaging_cost
  let tax := selling_price * tax_rate
  let total_cost := adjusted_cost + tax
  total_cost = 1.86 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_calculation_l3539_353971


namespace NUMINAMATH_CALUDE_periodic_decimal_is_rational_l3539_353962

/-- A real number with a periodic decimal expansion can be expressed as a rational number. -/
theorem periodic_decimal_is_rational (x : ℝ) (d : ℕ) (k : ℕ) (a b : ℕ) 
  (h1 : x = (a : ℝ) / 10^k + (b : ℝ) / (10^k * (10^d - 1)))
  (h2 : b < 10^d) :
  ∃ (p q : ℤ), x = (p : ℝ) / (q : ℝ) ∧ q ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_periodic_decimal_is_rational_l3539_353962


namespace NUMINAMATH_CALUDE_x_varies_as_z_l3539_353983

-- Define the relationships between variables
def varies_as (x y : ℝ) (n : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x = k * y^n

-- State the theorem
theorem x_varies_as_z (x y w z : ℝ) :
  varies_as x y 2 →
  varies_as y w 2 →
  varies_as w z (1/5) →
  varies_as x z (4/5) :=
sorry

end NUMINAMATH_CALUDE_x_varies_as_z_l3539_353983


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l3539_353975

/-- Represents the number of valid arrangements for n people where no two adjacent people stand -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => validArrangements (n + 1) + validArrangements (n + 2)

/-- The probability of no two adjacent people standing in a circular arrangement of n people -/
def probability (n : ℕ) : ℚ := (validArrangements n : ℚ) / (2^n : ℚ)

theorem no_adjacent_standing_probability :
  probability 10 = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l3539_353975


namespace NUMINAMATH_CALUDE_largest_unreachable_sum_eighty_eight_unreachable_l3539_353926

theorem largest_unreachable_sum : ∀ n : ℕ, n > 88 →
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 8 * a + 11 * b = n :=
by sorry

theorem eighty_eight_unreachable : ¬∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 8 * a + 11 * b = 88 :=
by sorry

end NUMINAMATH_CALUDE_largest_unreachable_sum_eighty_eight_unreachable_l3539_353926


namespace NUMINAMATH_CALUDE_skateboard_distance_l3539_353967

theorem skateboard_distance (scooter_speed : ℝ) (skateboard_speed_ratio : ℝ) (time_minutes : ℝ) :
  scooter_speed = 50 →
  skateboard_speed_ratio = 2 / 5 →
  time_minutes = 45 →
  skateboard_speed_ratio * scooter_speed * (time_minutes / 60) = 15 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_distance_l3539_353967


namespace NUMINAMATH_CALUDE_exists_all_intersecting_segment_l3539_353953

/-- A segment on a line -/
structure Segment where
  left : ℝ
  right : ℝ
  h : left < right

/-- A configuration of segments on a line -/
structure SegmentConfiguration where
  n : ℕ
  segments : Finset Segment
  total_count : segments.card = 2 * n + 1
  intersection_condition : ∀ s ∈ segments, (segments.filter (λ t => s.left < t.right ∧ t.left < s.right)).card ≥ n

/-- There exists a segment that intersects all others -/
theorem exists_all_intersecting_segment (config : SegmentConfiguration) :
  ∃ s ∈ config.segments, ∀ t ∈ config.segments, t ≠ s → s.left < t.right ∧ t.left < s.right :=
sorry

end NUMINAMATH_CALUDE_exists_all_intersecting_segment_l3539_353953


namespace NUMINAMATH_CALUDE_block_arrangement_table_height_l3539_353965

/-- The height of the table in the block arrangement problem -/
def table_height : ℝ := 36

/-- The initial length measurement in the block arrangement -/
def initial_length : ℝ := 42

/-- The final length measurement in the block arrangement -/
def final_length : ℝ := 36

/-- The difference between block width and overlap in the first arrangement -/
def width_overlap_difference : ℝ := 6

theorem block_arrangement_table_height :
  ∃ (block_length block_width overlap : ℝ),
    block_length + table_height - overlap = initial_length ∧
    block_width + table_height - block_length = final_length ∧
    block_width = overlap + width_overlap_difference ∧
    table_height = 36 := by
  sorry

#check block_arrangement_table_height

end NUMINAMATH_CALUDE_block_arrangement_table_height_l3539_353965


namespace NUMINAMATH_CALUDE_marie_erasers_l3539_353942

/-- The number of erasers Marie loses -/
def erasers_lost : ℕ := 42

/-- The number of erasers Marie ends up with -/
def erasers_left : ℕ := 53

/-- The initial number of erasers Marie had -/
def initial_erasers : ℕ := erasers_left + erasers_lost

theorem marie_erasers : initial_erasers = 95 := by sorry

end NUMINAMATH_CALUDE_marie_erasers_l3539_353942


namespace NUMINAMATH_CALUDE_three_digit_number_property_l3539_353945

theorem three_digit_number_property (A : ℕ) : 
  100 ≤ A → A < 1000 → 
  let B := 1001 * A
  (((B / 7) / 11) / 13) = A := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_property_l3539_353945


namespace NUMINAMATH_CALUDE_root_equation_value_l3539_353959

theorem root_equation_value (a : ℝ) : 
  a^2 + 2*a - 2 = 0 → 3*a^2 + 6*a + 2023 = 2029 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l3539_353959


namespace NUMINAMATH_CALUDE_distance_is_approx_7_38_l3539_353997

/-- Represents a circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  radius : ℝ
  chord_length_1 : ℝ
  chord_length_2 : ℝ
  chord_length_3 : ℝ
  parallel_line_distance : ℝ
  chord_length_1_eq : chord_length_1 = 40
  chord_length_2_eq : chord_length_2 = 36
  chord_length_3_eq : chord_length_3 = 40
  equally_spaced : True  -- Assumption that lines are equally spaced

/-- The distance between adjacent parallel lines in the given configuration -/
def distance_between_lines (c : CircleWithParallelLines) : ℝ :=
  c.parallel_line_distance

/-- Theorem stating that the distance between adjacent parallel lines is approximately 7.38 -/
theorem distance_is_approx_7_38 (c : CircleWithParallelLines) :
  ∃ ε > 0, |distance_between_lines c - 7.38| < ε :=
sorry

#check distance_is_approx_7_38

end NUMINAMATH_CALUDE_distance_is_approx_7_38_l3539_353997


namespace NUMINAMATH_CALUDE_mismatched_pens_probability_l3539_353989

def num_pens : ℕ := 3

def total_arrangements : ℕ := 6

def mismatched_arrangements : ℕ := 3

theorem mismatched_pens_probability :
  (mismatched_arrangements : ℚ) / total_arrangements = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_mismatched_pens_probability_l3539_353989


namespace NUMINAMATH_CALUDE_parabola_abc_value_l3539_353924

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_abc_value (a b c : ℝ) :
  -- Vertex condition
  (∀ x, parabola a b c x = a * (x - 4)^2 + 2) →
  -- Point (2, 0) lies on the parabola
  parabola a b c 2 = 0 →
  -- Conclusion: abc = 12
  a * b * c = 12 := by
  sorry

end NUMINAMATH_CALUDE_parabola_abc_value_l3539_353924


namespace NUMINAMATH_CALUDE_disk_arrangement_sum_l3539_353968

theorem disk_arrangement_sum (n : ℕ) (r : ℝ) :
  n = 8 →
  r > 0 →
  r = 2 - Real.sqrt 2 →
  ∃ (a b c : ℕ), 
    c = 2 ∧
    n * (π * r^2) = π * (a - b * Real.sqrt c) ∧
    a + b + c = 82 :=
sorry

end NUMINAMATH_CALUDE_disk_arrangement_sum_l3539_353968


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_four_consecutive_integers_sum_l3539_353902

theorem smallest_prime_factor_of_four_consecutive_integers_sum (n : ℤ) :
  ∃ (k : ℤ), (n - 1) + n + (n + 1) + (n + 2) = 2 * k ∧
  ∀ (p : ℕ), p < 2 → ¬(Prime p ∧ ∃ (m : ℤ), (n - 1) + n + (n + 1) + (n + 2) = p * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_four_consecutive_integers_sum_l3539_353902


namespace NUMINAMATH_CALUDE_quadratic_sequence_sum_l3539_353985

theorem quadratic_sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ + 64*x₈ = 10)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ + 81*x₈ = 40)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ + 100*x₈ = 170) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ + 121*x₈ = 400 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_sum_l3539_353985


namespace NUMINAMATH_CALUDE_function_identity_l3539_353947

theorem function_identity (f g h : ℕ → ℕ) 
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l3539_353947


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l3539_353952

theorem right_rectangular_prism_volume
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 24)
  (h_front : front_area = 18)
  (h_bottom : bottom_area = 12) :
  ∃ a b c : ℝ,
    a * b = side_area ∧
    b * c = front_area ∧
    c * a = bottom_area ∧
    a * b * c = 72 :=
by sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l3539_353952


namespace NUMINAMATH_CALUDE_incorrect_calculations_l3539_353964

theorem incorrect_calculations : 
  (¬ (4237 * 27925 = 118275855)) ∧ 
  (¬ (42971064 / 8264 = 5201)) ∧ 
  (¬ (1965^2 = 3761225)) ∧ 
  (¬ (371293^(1/5) = 23)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculations_l3539_353964


namespace NUMINAMATH_CALUDE_square_of_number_l3539_353903

theorem square_of_number (x : ℝ) : 2 * x = x / 5 + 9 → x^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_number_l3539_353903


namespace NUMINAMATH_CALUDE_pi_approx_thousandth_l3539_353922

/-- The approximation of π to the thousandth place -/
def pi_approx : ℝ := 3.142

/-- The theorem stating that the approximation of π to the thousandth place is equal to 3.142 -/
theorem pi_approx_thousandth : |π - pi_approx| < 0.0005 := by
  sorry

end NUMINAMATH_CALUDE_pi_approx_thousandth_l3539_353922


namespace NUMINAMATH_CALUDE_oak_willow_difference_l3539_353980

theorem oak_willow_difference (total_trees : ℕ) (willows : ℕ) : 
  total_trees = 83 → willows = 36 → total_trees - willows - willows = 11 := by
  sorry

end NUMINAMATH_CALUDE_oak_willow_difference_l3539_353980


namespace NUMINAMATH_CALUDE_tangent_line_at_one_two_l3539_353961

/-- The equation of the tangent line to y = -x^3 + 3x^2 at (1, 2) is y = 3x - 1 -/
theorem tangent_line_at_one_two (x : ℝ) :
  let f (x : ℝ) := -x^3 + 3*x^2
  let tangent_line (x : ℝ) := 3*x - 1
  f 1 = 2 ∧ 
  (∀ x, x ≠ 1 → (f x - f 1) / (x - 1) ≠ tangent_line x - tangent_line 1) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → 
    |(f x - f 1) / (x - 1) - (tangent_line x - tangent_line 1) / (x - 1)| < ε) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_two_l3539_353961


namespace NUMINAMATH_CALUDE_bella_current_beads_l3539_353963

/-- The number of friends Bella is making bracelets for -/
def num_friends : ℕ := 6

/-- The number of beads needed per bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of additional beads Bella needs -/
def additional_beads_needed : ℕ := 12

/-- The total number of beads Bella needs for all bracelets -/
def total_beads_needed : ℕ := num_friends * beads_per_bracelet

/-- Theorem: Bella currently has 36 beads -/
theorem bella_current_beads : 
  total_beads_needed - additional_beads_needed = 36 := by
  sorry

end NUMINAMATH_CALUDE_bella_current_beads_l3539_353963


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3539_353982

theorem expression_simplification_and_evaluation :
  let f (x : ℚ) := (x^2 - 4*x) / (x^2 - 16) / ((x^2 + 4*x) / (x^2 + 8*x + 16)) - 2*x / (x - 4)
  f (-2 : ℚ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3539_353982


namespace NUMINAMATH_CALUDE_chessboard_covering_impossible_l3539_353951

/-- Represents a chessboard with given dimensions -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a tile with given dimensions -/
structure Tile :=
  (length : Nat)
  (width : Nat)

/-- Determines if a chessboard can be covered by a given number of tiles -/
def can_cover (board : Chessboard) (tile : Tile) (num_tiles : Nat) : Prop :=
  ∃ (arrangement : Nat), 
    arrangement > 0 ∧ 
    tile.length * tile.width * num_tiles = board.rows * board.cols

/-- The main theorem stating that a 10x10 chessboard cannot be covered by 25 4x1 tiles -/
theorem chessboard_covering_impossible :
  ¬(can_cover (Chessboard.mk 10 10) (Tile.mk 4 1) 25) :=
sorry

end NUMINAMATH_CALUDE_chessboard_covering_impossible_l3539_353951


namespace NUMINAMATH_CALUDE_days_passed_before_realization_l3539_353954

/-- Represents the contractor's job scenario -/
structure JobScenario where
  totalDays : ℕ
  initialWorkers : ℕ
  workCompletedFraction : ℚ
  workersFired : ℕ
  remainingDays : ℕ

/-- Calculates the number of days passed before the contractor realized a fraction of work was done -/
def daysPassedBeforeRealization (scenario : JobScenario) : ℕ :=
  sorry

/-- The theorem stating that for the given scenario, 20 days passed before realization -/
theorem days_passed_before_realization :
  let scenario : JobScenario := {
    totalDays := 100,
    initialWorkers := 10,
    workCompletedFraction := 1/4,
    workersFired := 2,
    remainingDays := 75
  }
  daysPassedBeforeRealization scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_days_passed_before_realization_l3539_353954


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3539_353950

/-- The slope of the line -/
def m : ℚ := -1/2

/-- A point on the line -/
def p : ℝ × ℝ := (2, -3)

/-- The equation of the line in the form ax + by + c = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x + 2*y + 4 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := -4

/-- The y-intercept of the line -/
def y_intercept : ℝ := -2

/-- The area of the triangle formed by the line and coordinate axes -/
def triangle_area : ℝ := 4

theorem triangle_area_proof :
  line_equation p.1 p.2 ∧
  (∀ x y : ℝ, line_equation x y → y - p.2 = m * (x - p.1)) →
  triangle_area = (1/2) * |x_intercept| * |y_intercept| :=
sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3539_353950


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3539_353955

/-- Represents the number of cities in a group -/
def num_cities : ℕ := 8

/-- Represents the probability of a city being selected -/
def selection_probability : ℚ := 1/4

/-- Represents the number of cities drawn from the group -/
def cities_drawn : ℚ := num_cities * selection_probability

theorem stratified_sampling_theorem :
  cities_drawn = 2 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3539_353955


namespace NUMINAMATH_CALUDE_expression_value_l3539_353930

theorem expression_value : (4 - 2)^3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3539_353930


namespace NUMINAMATH_CALUDE_paint_calculation_l3539_353939

theorem paint_calculation (num_bedrooms : ℕ) (num_other_rooms : ℕ) 
  (total_cans : ℕ) (white_can_size : ℚ) :
  num_bedrooms = 3 →
  num_other_rooms = 2 * num_bedrooms →
  total_cans = 10 →
  white_can_size = 3 →
  (total_cans - num_bedrooms) * white_can_size / num_other_rooms = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l3539_353939


namespace NUMINAMATH_CALUDE_function_composition_inverse_l3539_353981

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_inverse (a b : ℝ) :
  (∀ x, h a b x = (x - 6) / 2) →
  a - b = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_function_composition_inverse_l3539_353981


namespace NUMINAMATH_CALUDE_class_average_after_exclusion_l3539_353936

/-- Proves that given a class of 10 students with an average mark of 80,
    if 5 students with an average mark of 70 are excluded,
    the average mark of the remaining students is 90. -/
theorem class_average_after_exclusion
  (total_students : ℕ)
  (total_average : ℚ)
  (excluded_students : ℕ)
  (excluded_average : ℚ)
  (h1 : total_students = 10)
  (h2 : total_average = 80)
  (h3 : excluded_students = 5)
  (h4 : excluded_average = 70) :
  let remaining_students := total_students - excluded_students
  let total_marks := total_students * total_average
  let excluded_marks := excluded_students * excluded_average
  let remaining_marks := total_marks - excluded_marks
  remaining_marks / remaining_students = 90 := by
  sorry


end NUMINAMATH_CALUDE_class_average_after_exclusion_l3539_353936


namespace NUMINAMATH_CALUDE_remaining_numbers_l3539_353966

def three_digit_numbers : ℕ := 900

def numbers_with_two_identical_nonadjacent_digits : ℕ := 81

def numbers_with_three_distinct_digits : ℕ := 648

theorem remaining_numbers :
  three_digit_numbers - (numbers_with_two_identical_nonadjacent_digits + numbers_with_three_distinct_digits) = 171 := by
  sorry

end NUMINAMATH_CALUDE_remaining_numbers_l3539_353966


namespace NUMINAMATH_CALUDE_position_relationships_l3539_353906

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the lines and planes
variable (a b : Line)
variable (α β γ : Plane)

-- State the theorem
theorem position_relationships :
  (∀ (a b : Line) (α : Plane), 
    parallel a b → subset b α → parallel_plane a α) = False ∧ 
  (∀ (a b : Line) (α : Plane), 
    parallel a b → parallel_plane a α → parallel_plane b α) = False ∧
  (∀ (a b : Line) (α β γ : Plane),
    intersect α β a → subset b γ → parallel_plane b β → subset a γ → parallel a b) = True :=
sorry

end NUMINAMATH_CALUDE_position_relationships_l3539_353906


namespace NUMINAMATH_CALUDE_distance_post_office_to_home_l3539_353995

/-- The distance Spencer walked from his house to the library -/
def distance_house_to_library : ℝ := 0.3

/-- The distance Spencer walked from the library to the post office -/
def distance_library_to_post_office : ℝ := 0.1

/-- The total distance Spencer walked -/
def total_distance : ℝ := 0.8

/-- Theorem: The distance Spencer walked from the post office back home is 0.4 miles -/
theorem distance_post_office_to_home : 
  total_distance - (distance_house_to_library + distance_library_to_post_office) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_distance_post_office_to_home_l3539_353995


namespace NUMINAMATH_CALUDE_product_of_x_values_l3539_353987

theorem product_of_x_values (x : ℝ) : 
  (|18 / x + 4| = 3) → 
  (∃ y : ℝ, y ≠ x ∧ |18 / y + 4| = 3 ∧ x * y = 324 / 7) :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l3539_353987


namespace NUMINAMATH_CALUDE_cake_area_theorem_l3539_353934

/-- Represents the dimensions of a piece of cake -/
structure PieceDimensions where
  length : ℝ
  width : ℝ

/-- Represents a cake -/
structure Cake where
  pieces : ℕ
  pieceDimensions : PieceDimensions

/-- Calculates the total area of a cake -/
def cakeArea (c : Cake) : ℝ :=
  c.pieces * (c.pieceDimensions.length * c.pieceDimensions.width)

theorem cake_area_theorem (c : Cake) 
  (h1 : c.pieces = 25)
  (h2 : c.pieceDimensions.length = 4)
  (h3 : c.pieceDimensions.width = 4) :
  cakeArea c = 400 := by
  sorry

end NUMINAMATH_CALUDE_cake_area_theorem_l3539_353934


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l3539_353944

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_2015th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_a5 : a 5 = 6) : 
  a 2015 = 2016 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l3539_353944


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3539_353932

/-- Represents the number of books of each type -/
structure BookCounts where
  chinese : Nat
  english : Nat
  math : Nat

/-- Represents the arrangement constraints -/
structure ArrangementConstraints where
  chinese_adjacent : Bool
  english_adjacent : Bool
  math_not_adjacent : Bool

/-- Calculates the number of valid book arrangements -/
def count_arrangements (counts : BookCounts) (constraints : ArrangementConstraints) : Nat :=
  sorry

/-- The main theorem stating the number of valid arrangements -/
theorem book_arrangement_count :
  let counts : BookCounts := ⟨2, 2, 3⟩
  let constraints : ArrangementConstraints := ⟨true, true, true⟩
  count_arrangements counts constraints = 48 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3539_353932


namespace NUMINAMATH_CALUDE_unique_angle_D_l3539_353938

/-- Represents a convex pentagon with equal sides -/
structure EqualSidedPentagon where
  -- Angles in degrees
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angleD : ℝ
  angleE : ℝ
  -- Conditions
  convex : angleA > 0 ∧ angleB > 0 ∧ angleC > 0 ∧ angleD > 0 ∧ angleE > 0
  sum_of_angles : angleA + angleB + angleC + angleD + angleE = 540
  angleA_is_120 : angleA = 120
  angleC_is_135 : angleC = 135

/-- The main theorem -/
theorem unique_angle_D (p : EqualSidedPentagon) : p.angleD = 90 := by
  sorry


end NUMINAMATH_CALUDE_unique_angle_D_l3539_353938


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_in_equilateral_l3539_353900

theorem right_triangle_inscribed_in_equilateral (XC BX CZ : ℝ) :
  XC = 4 →
  BX = 3 →
  CZ = 3 →
  let XZ := XC + CZ
  let XY := XZ
  let YZ := XZ
  let BC := Real.sqrt (BX^2 + XC^2 - 2 * BX * XC * Real.cos (π/3))
  let AB := Real.sqrt (BX^2 + BC^2)
  let AZ := Real.sqrt (CZ^2 + BC^2)
  AB^2 = BC^2 + AZ^2 →
  AZ = 3 := by sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_in_equilateral_l3539_353900


namespace NUMINAMATH_CALUDE_binary_to_decimal_110011_l3539_353907

/-- Converts a list of binary digits to a decimal number -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of the number we want to convert -/
def binaryNumber : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the binary number 110011₂ is equal to the decimal number 51 -/
theorem binary_to_decimal_110011 : binaryToDecimal binaryNumber = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_110011_l3539_353907


namespace NUMINAMATH_CALUDE_certain_amount_calculation_l3539_353957

theorem certain_amount_calculation (x A : ℝ) (h1 : x = 170) (h2 : 0.65 * x = 0.2 * A) : A = 552.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_calculation_l3539_353957


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_plus_x_l3539_353911

theorem factorization_of_x_squared_plus_x (x : ℝ) : x^2 + x = x * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_plus_x_l3539_353911


namespace NUMINAMATH_CALUDE_cf_length_l3539_353916

/-- A rectangle ABCD with point F such that C is on DF and B is on DE -/
structure SpecialRectangle where
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- Point E -/
  E : ℝ × ℝ
  /-- Point F -/
  F : ℝ × ℝ
  /-- ABCD is a rectangle -/
  is_rectangle : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2
  /-- AB = 8 -/
  ab_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64
  /-- BC = 6 -/
  bc_length : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 36
  /-- C is on DF -/
  c_on_df : ∃ t : ℝ, C = (1 - t) • D + t • F
  /-- B is the quarter-point of DE -/
  b_quarter_point : B = (3/4) • D + (1/4) • E
  /-- DEF is a right triangle -/
  def_right_triangle : (D.1 - E.1) * (E.1 - F.1) + (D.2 - E.2) * (E.2 - F.2) = 0

/-- The length of CF is 12 -/
theorem cf_length (rect : SpecialRectangle) : 
  (rect.C.1 - rect.F.1)^2 + (rect.C.2 - rect.F.2)^2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_cf_length_l3539_353916


namespace NUMINAMATH_CALUDE_two_pairs_satisfy_equation_l3539_353908

theorem two_pairs_satisfy_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℕ),
    (x₁ = 2 ∧ y₁ = 2 ∧ 2 * x₁^3 = y₁^4) ∧
    (x₂ = 32 ∧ y₂ = 16 ∧ 2 * x₂^3 = y₂^4) :=
by sorry

end NUMINAMATH_CALUDE_two_pairs_satisfy_equation_l3539_353908


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3539_353943

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3539_353943


namespace NUMINAMATH_CALUDE_common_divisors_2n_plus_3_and_3n_plus_2_l3539_353994

theorem common_divisors_2n_plus_3_and_3n_plus_2 (n : ℕ) :
  {d : ℕ | d ∣ (2*n + 3) ∧ d ∣ (3*n + 2)} = {d : ℕ | d = 1 ∨ (d = 5 ∧ n % 5 = 1)} := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_2n_plus_3_and_3n_plus_2_l3539_353994


namespace NUMINAMATH_CALUDE_maximal_k_for_triangle_l3539_353928

theorem maximal_k_for_triangle : ∃ (k : ℝ), k = 5 ∧ 
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → k * a * b * c > a^3 + b^3 + c^3 → 
    a + b > c ∧ b + c > a ∧ c + a > b) ∧
  (∀ (k' : ℝ), k' > k → 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ k' * a * b * c > a^3 + b^3 + c^3 ∧
      (a + b ≤ c ∨ b + c ≤ a ∨ c + a ≤ b)) :=
sorry

end NUMINAMATH_CALUDE_maximal_k_for_triangle_l3539_353928


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3539_353913

theorem complex_number_quadrant : 
  let z : ℂ := Complex.I / (1 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3539_353913


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3539_353914

theorem scientific_notation_equivalence : 
  56000000 = 5.6 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3539_353914


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3539_353905

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x > 0 ∧ ¬(1 < x ∧ x < 2)) ∧ 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3539_353905


namespace NUMINAMATH_CALUDE_interest_calculation_l3539_353909

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Interest earned calculation --/
def interest_earned (total : ℝ) (principal : ℝ) : ℝ :=
  total - principal

theorem interest_calculation (P : ℝ) (h1 : P > 0) :
  let rate : ℝ := 0.08
  let time : ℕ := 2
  let total : ℝ := 19828.80
  compound_interest P rate time = total →
  interest_earned total P = 2828.80 := by
sorry


end NUMINAMATH_CALUDE_interest_calculation_l3539_353909


namespace NUMINAMATH_CALUDE_equal_roots_condition_l3539_353921

theorem equal_roots_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (x * (x - 1) - (m^2 + 2*m + 1)) / ((x - 1) * (m^2 - 1) + 1) = x / m) →
  (∃! x : ℝ, x * (x - 1) - (m^2 + 2*m + 1) = 0) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l3539_353921


namespace NUMINAMATH_CALUDE_never_return_to_initial_l3539_353912

def transform (q : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (a, b, c, d) := q
  (a * b, b * c, c * d, d * a)

def iterate_transform (q : ℝ × ℝ × ℝ × ℝ) (n : ℕ) : ℝ × ℝ × ℝ × ℝ :=
  match n with
  | 0 => q
  | n + 1 => transform (iterate_transform q n)

theorem never_return_to_initial (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hna : a ≠ 1) (hnb : b ≠ 1) (hnc : c ≠ 1) (hnd : d ≠ 1) :
  ∀ n : ℕ, iterate_transform (a, b, c, d) n ≠ (a, b, c, d) :=
sorry

end NUMINAMATH_CALUDE_never_return_to_initial_l3539_353912
