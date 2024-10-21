import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_2_eq_14_l616_61609

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 1

-- State the theorem
theorem f_of_g_of_2_eq_14 : f (g 2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_2_eq_14_l616_61609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l616_61632

noncomputable def solution_set : Set ℝ :=
  {x | x ∈ Set.Ioc 0 (3^(-Real.sqrt (Real.log 2 / Real.log 3))) ∪ {1} ∪ Set.Ici (3^(Real.sqrt (Real.log 2 / Real.log 3)))}

theorem inequality_solution (x : ℝ) (hx : x > 0) :
  x^(Real.log x / Real.log 3) - 2 ≤ (3^(1/3))^((2 * Real.log x / Real.log 3)^2) - 2 * x^((1/3) * Real.log x / Real.log 3) ↔
  x ∈ solution_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l616_61632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_approximation_l616_61677

/-- The value of w in the given equation is approximately equal to -296.073 -/
theorem w_approximation : 
  let w : ℝ := ((69.28 * 123.57 * 0.004) - (42.67 * 3.12)) / (0.03 * 8.94 * 1.25)
  ∃ ε : ℝ, ε > 0 ∧ |w + 296.073| < ε ∧ ε < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_approximation_l616_61677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l616_61635

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 16*y + 68 = 0

-- Define the line equation
def line_equation (y : ℝ) : Prop :=
  y = 4

-- Define the area of the circle above the line
noncomputable def area_above_line : ℝ := 12 * Real.pi

-- Define the radius of the circle
noncomputable def radius_of_circle : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem circle_area_above_line :
  ∀ x y : ℝ, circle_equation x y → line_equation y →
  (∃ A : ℝ, A = area_above_line ∧ 
   A = Real.pi * radius_of_circle^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_above_line_l616_61635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l616_61656

/-- Given a hyperbola with equation x²/4 - y²/m = 1 and asymptote equations y = ±(√2/2)x, 
    the value of m is 2. -/
theorem hyperbola_m_value (m : ℝ) : 
  (∀ x y : ℝ, x^2/4 - y^2/m = 1 → y = Real.sqrt 2/2 * x ∨ y = -(Real.sqrt 2/2) * x) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l616_61656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_ellipse_equation_l616_61601

-- Define the line l: x + 2y + 6 = 0
def line_l (x y : ℝ) : Prop := x + 2*y + 6 = 0

-- Define the points F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the symmetric point F'₁
def F'₁ : ℝ × ℝ := (-3, -4)

-- Define the reflection point M on line l
def M : ℝ × ℝ := sorry

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem 1: F'₁ is symmetric to F₁ with respect to line l
theorem symmetric_point : 
  line_l F'₁.1 F'₁.2 ∧ 
  distance F₁ M = distance F'₁ M := by
  sorry

-- Theorem 2: Equation of the ellipse
theorem ellipse_equation : 
  ∀ (x y : ℝ), 
    (distance (x, y) F₁ + distance (x, y) F₂ = 2 * Real.sqrt 8) ↔ 
    (x^2 / 8 + y^2 / 7 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_ellipse_equation_l616_61601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_flight_probability_l616_61689

/-- A cube with edge length 4 -/
def large_cube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 4}

/-- The "safe" region inside the large cube -/
def safe_region : Set (Fin 3 → ℝ) :=
  {p ∈ large_cube | ∀ i, 1 < p i ∧ p i < 3}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (s : Set (Fin 3 → ℝ)) : ℝ := sorry

/-- The probability of a point being in the safe region -/
noncomputable def safe_probability : ℝ :=
  volume safe_region / volume large_cube

theorem safe_flight_probability :
  safe_probability = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_flight_probability_l616_61689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l616_61641

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the line x + y = 0 -/
noncomputable def slope_line1 : ℝ := -1

/-- The slope of the line x - a² * y = 0 -/
noncomputable def slope_line2 (a : ℝ) : ℝ := 1 / a^2

/-- The condition that a = 1 is sufficient but not necessary for perpendicularity -/
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → perpendicular slope_line1 (slope_line2 a)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ perpendicular slope_line1 (slope_line2 a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l616_61641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruits_sold_equals_90_l616_61697

/-- The number of items in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of lemons sold -/
def lemons_dozens : ℚ := 2.5

/-- The number of dozens of avocados sold -/
def avocados_dozens : ℕ := 5

/-- The total number of fruits sold -/
def total_fruits : ℕ := 90

/-- Theorem stating that the total number of fruits sold is 90 -/
theorem fruits_sold_equals_90 : 
  (lemons_dozens * (dozen : ℚ)).floor + avocados_dozens * dozen = total_fruits := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruits_sold_equals_90_l616_61697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_determinant_l616_61664

theorem cubic_roots_determinant (s p q : ℝ) (a b c : ℝ) : 
  (a^3 + s*a^2 + p*a + q = 0) → 
  (b^3 + s*b^2 + p*b + q = 0) → 
  (c^3 + s*c^2 + p*c + q = 0) → 
  Matrix.det !![a, 1, 1; 1, b, 1; 1, 1, c] = -q - s + 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_determinant_l616_61664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_22_5_deg_l616_61633

theorem cos_sin_22_5_deg : 
  (Real.cos (22.5 * π / 180))^2 - (Real.sin (22.5 * π / 180))^2 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_22_5_deg_l616_61633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_non_intersection_l616_61655

theorem polynomial_non_intersection (n : ℕ) : 
  (∀ (P Q : Polynomial ℝ) (hP : P.degree ≤ n) (hQ : Q.degree ≤ n), 
    ∃ (a b : ℝ) (k l : ℕ), k ≤ n ∧ l ≤ n ∧ 
    ∀ (x : ℝ), P.eval x + a * x^k ≠ Q.eval x + b * x^l) ↔ 
  (Even n ∨ n = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_non_intersection_l616_61655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enrique_loafer_sales_l616_61625

/-- Calculates the number of loafers sold given the commission rate, suit sales, shirt sales, prices, and total commission --/
def loafers_sold (commission_rate : ℚ) (num_suits : ℕ) (suit_price : ℚ) 
                 (num_shirts : ℕ) (shirt_price : ℚ) (loafer_price : ℚ) 
                 (total_commission : ℚ) : ℕ :=
  let suit_commission := commission_rate * (num_suits : ℚ) * suit_price
  let shirt_commission := commission_rate * (num_shirts : ℚ) * shirt_price
  let loafer_commission := total_commission - suit_commission - shirt_commission
  let loafer_commission_per_pair := commission_rate * loafer_price
  (loafer_commission / loafer_commission_per_pair).floor.toNat

theorem enrique_loafer_sales : 
  loafers_sold (15 / 100) 2 700 6 50 150 300 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enrique_loafer_sales_l616_61625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l616_61623

/-- A circle in the Cartesian plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the Cartesian plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between a point and a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  (abs (l.a * p.1 + l.b * p.2 + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- The number of intersection points between a circle and the set of points at a fixed distance from a line -/
def numIntersectionPoints (circle : Circle) (line : Line) (d : ℝ) : ℕ :=
  sorry

theorem circle_line_intersection (c : ℝ) :
  let circle : Circle := { center := (0, 0), radius := 2 }
  let line : Line := { a := 4, b := -3, c := c }
  (numIntersectionPoints circle line 1 = 4) ↔ -5 < c ∧ c < 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l616_61623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l616_61694

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 1) :
  (⨆ z, Complex.abs ((1 + 2*Complex.I)*z^2 - z^4)) = Real.sqrt 5 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l616_61694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_in_base6_l616_61695

/-- Represents a single digit in base 6 -/
def Base6Digit := Fin 6

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List Base6Digit := sorry

/-- Converts a list of base 6 digits to a natural number -/
def fromBase6 (l : List Base6Digit) : ℕ := sorry

/-- Represents the given arithmetic operation -/
def arithmeticOperation (A B : Base6Digit) : Prop :=
  ∃ (carry1 carry2 : Base6Digit),
    fromBase6 [B, A, A] + fromBase6 [⟨2, sorry⟩, ⟨3, sorry⟩, B] + fromBase6 [A, ⟨4, sorry⟩, ⟨4, sorry⟩] =
    fromBase6 [carry2, carry1, B, ⟨1, sorry⟩, ⟨3, sorry⟩, ⟨3, sorry⟩]

theorem absolute_difference_in_base6 (A B : Base6Digit) :
  arithmeticOperation A B → toBase6 (Int.natAbs (A.val - B.val)) = [⟨5, sorry⟩] := by
  sorry

#check absolute_difference_in_base6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_in_base6_l616_61695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_nonnegative_l616_61687

open MeasureTheory

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Define the domain
def domain : Set ℝ := Set.Icc (-1) 3

-- Define the condition for f(x) ≥ 0
def condition (x : ℝ) : Prop := f x ≥ 0

-- State the theorem
theorem probability_f_nonnegative :
  (volume {x ∈ domain | condition x}) / (volume domain) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_nonnegative_l616_61687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_A_is_quadratic_l616_61611

-- Define what a quadratic equation is
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the equations
noncomputable def eq_A (x : ℝ) : ℝ := x^2
noncomputable def eq_B (x : ℝ) : ℝ := x^2 - x * (x + 1)
noncomputable def eq_C (x : ℝ) : ℝ := 2 * x + 1 / x + 1
noncomputable def eq_D (x : ℝ) : ℝ := x^3 + x - 1

-- Theorem stating that only equation A is quadratic
theorem only_A_is_quadratic :
  is_quadratic eq_A ∧
  ¬is_quadratic eq_B ∧
  ¬is_quadratic eq_C ∧
  ¬is_quadratic eq_D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_A_is_quadratic_l616_61611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l616_61678

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem beta_value (α β : ℝ) :
  Real.cos α = 1 / 7 →
  det (Real.sin α) (Real.sin β) (Real.cos α) (Real.cos β) = 3 * Real.sqrt 3 / 14 →
  0 < β →
  β < α →
  α < π / 2 →
  β = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l616_61678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_polynomial_A_l616_61654

-- Define the characteristic numbers of a quadratic polynomial
def characteristic_numbers (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

-- Define a quadratic polynomial
def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Theorem statement
theorem find_polynomial_A :
  ∀ A : ℝ → ℝ,
  (∃ a b c : ℝ, ∀ x : ℝ, A x = a * x^2 + b * x + c) →
  (characteristic_numbers 1 (-4) 6 = 
   characteristic_numbers 
     (A 1 - (2 * 1^2 - 4 * 1 - 2)) 
     (A 0 - (2 * 0^2 - 4 * 0 - 2)) 
     (A 0 - (2 * 0^2 - 4 * 0 - 2))) →
  (∀ x : ℝ, A x = 3 * x^2 - 8 * x + 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_polynomial_A_l616_61654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_and_prices_l616_61660

-- Define the variables
variable (x y z u : ℚ)

-- Define the conditions as functions
def condition1 (x y z u : ℚ) : Prop := y / 4 = 6 * z
def condition2 (x y z u : ℚ) : Prop := x / 5 = 8 * z
def condition3 (x y z u : ℚ) : Prop := (x + 46) + (y - 46) / 3 = 30 * u
def condition4 (x y z u : ℚ) : Prop := (y - 46) + (x + 46) / 3 = 36 * u

-- State the theorem
theorem money_and_prices 
  (h1 : condition1 x y z u)
  (h2 : condition2 x y z u)
  (h3 : condition3 x y z u)
  (h4 : condition4 x y z u) :
  x = 520 ∧ y = 312 ∧ z = 13 ∧ u = 50/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_and_prices_l616_61660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_l616_61692

-- Define the properties of quadrilaterals
structure Quadrilateral where
  diagonals_bisect : Bool
  diagonals_perpendicular : Bool
  diagonals_equal : Bool
  is_parallelogram : Bool
  is_rhombus : Bool
  is_square : Bool
  is_rectangle : Bool

-- Define the statements
def statement1 (q : Quadrilateral) : Prop :=
  q.diagonals_bisect → q.is_parallelogram

def statement2 (q : Quadrilateral) : Prop :=
  q.diagonals_perpendicular → q.is_rhombus

def statement3 (q : Quadrilateral) : Prop :=
  q.is_parallelogram ∧ q.diagonals_perpendicular ∧ q.diagonals_equal → q.is_square

def statement4 (q : Quadrilateral) : Prop :=
  q.is_parallelogram ∧ q.diagonals_equal → q.is_rectangle

-- Theorem to prove
theorem quadrilateral_properties :
  ∃ (correct : List (Quadrilateral → Prop)),
    correct.length = 3 ∧
    (∀ s, s ∈ correct → s ∈ [statement1, statement2, statement3, statement4]) ∧
    (∀ s ∈ correct, ∀ q, s q) ∧
    (∀ s, s ∉ correct → ∃ q, ¬(s q)) := by
  sorry

#check quadrilateral_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_l616_61692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_and_reciprocal_sum_l616_61642

theorem fraction_sum_and_reciprocal_sum (a b c d e f : ℕ) : 
  a < b ∧ c < d ∧ e < f →
  a ≠ c ∧ a ≠ e ∧ c ≠ e →
  Nat.Coprime a b ∧ Nat.Coprime c d ∧ Nat.Coprime e f →
  (a : ℚ) / b + (c : ℚ) / d + (e : ℚ) / f = 1 →
  ∃ (n : ℕ), (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e = n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_and_reciprocal_sum_l616_61642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_m_range_for_inequality_l616_61657

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos x + Real.sin x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x - Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

def monotonic_increasing_interval (k : ℤ) : Set ℝ := 
  Set.Icc (-Real.pi/6 + k * Real.pi) (Real.pi/3 + k * Real.pi)

theorem f_monotonic_increasing_interval (k : ℤ) : 
  StrictMono (f ∘ (fun x => x + (-Real.pi/6 + k * Real.pi))) := by
  sorry

theorem m_range_for_inequality :
  ∀ m : ℝ, (∀ x ∈ Set.Icc (5*Real.pi/24) (5*Real.pi/12), 
    ∀ t : ℝ, m * t^2 + m * t + 3 ≥ f x) ↔ 
  m ∈ Set.Icc 0 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_m_range_for_inequality_l616_61657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_triangle_calculation_l616_61648

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ := (a * b) / (a + b)

-- Define the △ operation
noncomputable def triangle (a b : ℝ) : ℝ := (a - b) / (a / b)

-- Theorem statement
theorem otimes_triangle_calculation :
  triangle (otimes 6 4) 1.2 = 0.6 := by
  -- Unfold the definitions of otimes and triangle
  unfold otimes triangle
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_triangle_calculation_l616_61648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_is_one_l616_61600

/-- A linear function defined by its intersections with the x and y axes -/
structure LinearFunction where
  x_intercept : ℝ
  y_intercept : ℝ

/-- The product of two linear functions -/
noncomputable def product_function (f g : LinearFunction) : ℝ → ℝ :=
  λ x ↦ ((-f.y_intercept / f.x_intercept) * x + f.y_intercept) *
         ((-g.y_intercept / g.x_intercept) * x + g.y_intercept)

/-- The x-coordinate of the maximum point of the product function -/
noncomputable def max_point (f g : LinearFunction) : ℝ :=
  (f.x_intercept * g.x_intercept) / (f.x_intercept + g.x_intercept)

theorem max_point_is_one (f g : LinearFunction) 
  (hf : f.x_intercept = -2 ∧ f.y_intercept = 2)
  (hg : g.x_intercept = 2 ∧ g.y_intercept = 6) :
  max_point f g = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_is_one_l616_61600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_values_theorem_l616_61646

theorem y_values_theorem (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 54) :
  let y := (x - 3)^2 * (x + 4) / (3 * x - 4)
  (y = 7.5 ∨ y = 4.5) ∧ ∀ z, y = z → (z = 7.5 ∨ z = 4.5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_values_theorem_l616_61646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_sum_l616_61670

/-- Given a parabola y^2 = 2px (p > 0) and points A(a, 0) and A'(-a, 0) (a > 0),
    if a line through A' intersects the parabola at P and Q,
    then the sum of the slopes of AP and AQ is 0. -/
theorem parabola_slope_sum (p a : ℝ) (hp : p > 0) (ha : a > 0) : 
  ∃ (y₁ y₂ : ℝ), 
    let xp := y₁^2 / (2*p)
    let xq := y₂^2 / (2*p)
    let slope_ap := (y₁ - 0) / (xp - a)
    let slope_aq := (y₂ - 0) / (xq - a)
    (y₁ / (xp + a) = y₂ / (xq + a)) →  -- Line A'P and A'Q have the same slope
    (slope_ap + slope_aq = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_sum_l616_61670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_roots_range_f_difference_bound_l616_61649

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (1 + 2 * Real.log x) / x^2

-- Define the function g
def g (a x : ℝ) : ℝ := a * x^2 - 2 * Real.log x

-- Theorem 1: Maximum value of f
theorem f_max_value : ∃ (M : ℝ), M = 1 ∧ ∀ x > 0, f x ≤ M := by
  sorry

-- Theorem 2: Range of a
theorem g_roots_range (a : ℝ) :
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ g a x₁ = 1 ∧ g a x₂ = 1) → 0 < a ∧ a < 1 := by
  sorry

-- Theorem 3: Range of k
theorem f_difference_bound :
  ∀ k : ℝ, (∀ x₁ x₂, 1 < x₁ ∧ 1 < x₂ ∧ x₁ ≠ x₂ →
    |f x₁ - f x₂| ≥ k * |Real.log x₁ - Real.log x₂|) →
  k < 2 / Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_roots_range_f_difference_bound_l616_61649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_growing_with_square_digit_sum_l616_61624

/-- A positive integer is growing if its digits are non-increasing from left to right -/
def IsGrowing (m : ℕ) : Prop :=
  ∃ (digits : List ℕ), m = digits.foldl (λ acc d => acc * 10 + d) 0 ∧
    digits.length > 0 ∧
    ∀ i j, i < j → j < digits.length → digits.get ⟨i, by sorry⟩ ≥ digits.get ⟨j, by sorry⟩

/-- The sum of digits of a natural number -/
def SumOfDigits (m : ℕ) : ℕ :=
  (m.repr.toList.map (λ c => c.toNat - '0'.toNat)).sum

/-- Main theorem: For any natural number n, there exists a growing number
    with n digits whose sum of digits is a perfect square -/
theorem exists_growing_with_square_digit_sum (n : ℕ) :
  ∃ (m : ℕ), IsGrowing m ∧ m.repr.length = n ∧ ∃ (k : ℕ), SumOfDigits m = k^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_growing_with_square_digit_sum_l616_61624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l616_61634

-- Define a triangle with circumradius 2
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  circumradius : Real
  angle_sum : A + B + C = π
  sine_law : a / (2 * Real.sin A) = b / (2 * Real.sin B) ∧ 
             b / (2 * Real.sin B) = c / (2 * Real.sin C) ∧ 
             c / (2 * Real.sin C) = circumradius
  given_equation : (a - c) * (Real.sin A + Real.sin C) = b * (Real.sin A - Real.sin B)
  radius_is_2 : circumradius = 2

theorem triangle_properties (t : Triangle) :
  t.C = π / 3 ∧
  ∃ (p : Real), p ≤ 6 * Real.sqrt 3 ∧
    ∀ (t' : Triangle), t'.a + t'.b + t'.c ≤ p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l616_61634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_antisymmetric_at_specific_points_l616_61685

-- Define an anti-symmetric function
def antiSymmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem sum_of_antisymmetric_at_specific_points 
  (v : ℝ → ℝ) (h : antiSymmetric v) : 
  v (-2.25) + v (-1.05) + v 1.05 + v 2.25 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_antisymmetric_at_specific_points_l616_61685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_first_8_terms_l616_61674

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_8_terms : 
  geometric_sum 2 2 8 = 510 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_first_8_terms_l616_61674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_paper_l616_61645

/-- The probability of choosing paper in a single round of rock-paper-scissors -/
noncomputable def prob_paper : ℝ := 1 / 3

/-- The number of friends playing the game -/
def num_friends : ℕ := 3

/-- Theorem: The probability of all three friends choosing paper in a single round -/
theorem prob_all_paper : (prob_paper ^ num_friends : ℝ) = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_paper_l616_61645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_purchase_total_l616_61643

/-- Calculate the total amount paid for fruits with discounts and taxes -/
theorem fruit_purchase_total (grape_kg : ℝ) (grape_price : ℝ) (mango_kg : ℝ) (mango_price : ℝ)
  (grape_discount : ℝ) (mango_discount : ℝ) (grape_tax : ℝ) (mango_tax : ℝ) :
  grape_kg = 8 →
  grape_price = 70 →
  mango_kg = 9 →
  mango_price = 55 →
  grape_discount = 0.05 →
  mango_discount = 0.07 →
  grape_tax = 0.08 →
  mango_tax = 0.11 →
  (grape_kg * grape_price * (1 - grape_discount) * (1 + grape_tax) +
   mango_kg * mango_price * (1 - mango_discount) * (1 + mango_tax)) = 1085.55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_purchase_total_l616_61643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_2_l616_61699

theorem square_of_3x_plus_4_when_x_is_2 :
  ∀ x : ℝ, x = 2 → (3 * x + 4)^2 = 100 :=
by
  intro x h
  rw [h]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_2_l616_61699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_r_correct_l616_61622

/-- Given a permutation of n elements, f_r(n, k) represents the number of ways to select k elements
    such that there are at least r elements between any two selected elements. -/
def f_r (n k r : ℕ) : ℕ := Nat.choose (n - k*r + r) k

/-- Theorem stating that f_r(n, k) correctly counts the number of valid selections. -/
theorem f_r_correct (n k r : ℕ) (h : n + r ≥ k * r + k) :
  f_r n k r = (Nat.choose n k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_r_correct_l616_61622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_implies_sum_l616_61667

theorem determinant_zero_implies_sum (a b : ℝ) : 
  a ≠ b →
  Matrix.det !![2, 5, 8; 4, a, b; 4, b, a] = 0 →
  a + b = 26 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_implies_sum_l616_61667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_l616_61668

theorem triangle_angle_cosine (A B C : ℝ) (h1 : A + B + C = Real.pi)
  (h2 : A < Real.pi / 3.6) (h3 : B < 7 * Real.pi / 18) (h4 : Real.sin C = 4 / 7) :
  Real.cos C = -Real.sqrt 33 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_l616_61668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_equation_solutions_l616_61616

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem digit_sum_equation_solutions :
  ∀ n : ℕ, n > 0 → (digit_sum n * (digit_sum n - 1) = n - 1 ↔ n ∈ ({1, 13, 43, 91, 157} : Set ℕ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_equation_solutions_l616_61616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_x_minus_2_is_linear_l616_61682

/-- Represents a polynomial equation in one variable -/
structure Equation where
  coeffs : List ℝ

/-- Checks if an equation is linear in one variable -/
def is_linear_one_var (eq : Equation) : Prop :=
  eq.coeffs.length = 2 ∧ eq.coeffs.get? 1 ≠ none

/-- The specific equation πx - 2 = 0 -/
noncomputable def pi_x_minus_2 : Equation :=
  ⟨[(-2 : ℝ), Real.pi]⟩

theorem pi_x_minus_2_is_linear : is_linear_one_var pi_x_minus_2 := by
  sorry

#check pi_x_minus_2_is_linear

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_x_minus_2_is_linear_l616_61682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_time_fraction_l616_61666

/-- Given a car journey with specific parameters, prove the fraction of time needed at a new speed --/
theorem car_journey_time_fraction 
  (distance : ℝ) 
  (original_time : ℝ) 
  (required_speed : ℝ) 
  (h1 : distance = 469) 
  (h2 : original_time = 6) 
  (h3 : required_speed = 52.111111111111114) :
  distance / (required_speed * original_time) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_journey_time_fraction_l616_61666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l616_61605

-- Define the circles
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 7)^2 + (y - 5)^2 = 4

-- Define the points
def point_on_circle_M (P : ℝ × ℝ) : Prop := circle_M P.1 P.2
def point_on_circle_N (Q : ℝ × ℝ) : Prop := circle_N Q.1 Q.2
def point_on_x_axis (A : ℝ × ℝ) : Prop := A.2 = 0

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum (P Q A : ℝ × ℝ) 
  (hP : point_on_circle_M P) (hQ : point_on_circle_N Q) (hA : point_on_x_axis A) :
  ∃ (min_dist : ℝ), 
    (∀ (P' Q' A' : ℝ × ℝ), point_on_circle_M P' → point_on_circle_N Q' → point_on_x_axis A' →
      distance A' P' + distance A' Q' ≥ min_dist) ∧
    min_dist = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l616_61605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l616_61602

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 - (2/3) ^ (abs x) + 1/2

-- Define the domain
def domain : Set ℝ := {x | -Real.pi/2 ≤ x ∧ x ≤ Real.pi/2}

-- Theorem statement
theorem f_properties :
  (∀ x ∈ domain, f x < 3/2) ∧
  (∃ x ∈ domain, ∀ y ∈ domain, f x ≤ f y ∧ f x = -1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l616_61602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l616_61618

/-- Given two cones A and B with equal base radii -/
structure Cone where
  r : ℝ
  l : ℝ
  h_pos : r > 0
  h_l_pos : l > 0

/-- The side area of a cone -/
noncomputable def sideArea (c : Cone) : ℝ := Real.pi * c.r * c.l

/-- The volume of a cone -/
noncomputable def volume (c : Cone) : ℝ := (1/3) * Real.pi * c.r^2 * (Real.sqrt (c.l^2 - c.r^2))

theorem cone_volume_ratio 
  (A B : Cone)
  (h_same_r : A.r = B.r)
  (hA : sideArea A = 2 * Real.pi * A.r * A.l) -- Lateral surface of A unfolds to semicircle
  (hS : sideArea A = (2/3) * sideArea B) -- Ratio of side areas
  : volume A / volume B = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l616_61618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_equals_negative_one_l616_61615

-- Define g as a noncomputable function
noncomputable def g (a b x : ℝ) : ℝ := 1 / (2*a*x + 3*b)

-- Theorem statement
theorem inverse_g_equals_negative_one (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ x : ℝ, g a b x = -1 ∧ x = (-1 - 3*b) / (2*a) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_equals_negative_one_l616_61615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l616_61636

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem symmetry_implies_phi (φ : ℝ) :
  (∀ x, f (x + φ) = f (-x + φ)) →
  φ = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_l616_61636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_division_maximizes_probability_l616_61639

/-- Represents the total number of voters -/
def n : ℕ := sorry

/-- Represents the number of voters in the smaller district -/
def m : ℕ := sorry

/-- Represents the probability of Miraflores winning in the larger district -/
def win_prob_larger (n m : ℕ) : ℚ :=
  (n - m) / (2 * n - m)

/-- Represents the probability of Miraflores winning in the smaller district -/
def win_prob_smaller (m : ℕ) : ℚ :=
  if m = 1 then 1 else m / (2 * m)

/-- Represents the overall probability of Miraflores winning the election -/
def total_win_prob (n m : ℕ) : ℚ :=
  (win_prob_larger n m) * (win_prob_smaller m)

/-- Theorem stating that the optimal division maximizes Miraflores' winning probability -/
theorem optimal_division_maximizes_probability (n : ℕ) (h : n > 0) :
  ∀ m : ℕ, m > 0 → m ≤ 2*n → total_win_prob n 1 ≥ total_win_prob n m :=
by
  sorry

#check optimal_division_maximizes_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_division_maximizes_probability_l616_61639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_167_sequence_in_proper_fraction_under_100_l616_61627

/-- A proper fraction with denominator less than 100 -/
structure ProperFraction :=
  (numerator : ℕ)
  (denominator : ℕ)
  (h1 : denominator < 100)
  (h2 : numerator < denominator)

/-- Represents the decimal expansion of a fraction -/
def DecimalExpansion := List ℕ

/-- Check if the digits 1, 6, 7 appear consecutively in a decimal expansion -/
def has_167_sequence (d : DecimalExpansion) : Prop :=
  ∃ i, d.drop i = [1, 6, 7] ++ d.drop (i + 3)

/-- Placeholder for decimal expansion function -/
def decimal_expansion (f : ProperFraction) : DecimalExpansion :=
  sorry

/-- The main theorem to be proved -/
theorem no_167_sequence_in_proper_fraction_under_100 :
  ¬ ∃ (f : ProperFraction) (d : DecimalExpansion),
    (d = decimal_expansion f) ∧ (has_167_sequence d) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_167_sequence_in_proper_fraction_under_100_l616_61627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_inflation_l616_61663

theorem soccer_ball_inflation (total_balls : ℕ) 
  (hole_percentage : ℚ) (overinflated_percentage : ℚ) : 
  total_balls = 200 →
  hole_percentage = 60 / 100 →
  overinflated_percentage = 30 / 100 →
  (total_balls - (hole_percentage * ↑total_balls).floor - 
   (overinflated_percentage * ↑(total_balls - (hole_percentage * ↑total_balls).floor)).floor) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_inflation_l616_61663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l616_61612

noncomputable def h (x : ℝ) : ℝ := (x^3 + 11*x - 2) / (abs (x - 3) + abs (x + 1) + x)

theorem h_domain :
  {x : ℝ | h x ≠ 0} = {x : ℝ | x < -2 ∨ (-2 < x ∧ x < 2/3) ∨ x > 2/3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l616_61612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_square_l616_61652

def x : ℕ → ℕ
  | 0 => 1  -- Adding a case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => 2 * x (n + 2) + 8 * x (n + 1) - 1

theorem x_is_square : ∀ n : ℕ, ∃ y : ℕ, x n = y^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_square_l616_61652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_point_of_f_l616_61650

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 / (x^2) + Real.log x

-- State the theorem
theorem minimum_point_of_f :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ x₀ = 2 ∧ 
  (∀ (x : ℝ), x > 0 → f x ≥ f x₀) := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_point_of_f_l616_61650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_sphere_radius_formula_l616_61693

/-- A triangular prism with an equilateral base -/
structure TriangularPrism where
  /-- Side length of the equilateral base -/
  m : ℝ
  /-- The base is an equilateral triangle -/
  base_equilateral : True
  /-- Lateral edges are equal to the side of the base -/
  lateral_edges_equal_base : True
  /-- One vertex is equidistant from the base vertices -/
  vertex_equidistant : True

/-- The largest radius of a sphere that can fit inside the triangular prism -/
noncomputable def largest_inscribed_sphere_radius (prism : TriangularPrism) : ℝ :=
  (Real.sqrt 6 - Real.sqrt 2) / 4 * prism.m

/-- Theorem: The largest radius of a sphere that can fit inside the triangular prism
    is (√6 - √2) / 4 * m -/
theorem largest_inscribed_sphere_radius_formula (prism : TriangularPrism) :
  largest_inscribed_sphere_radius prism = (Real.sqrt 6 - Real.sqrt 2) / 4 * prism.m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_sphere_radius_formula_l616_61693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candied_fruit_profit_l616_61620

/-- Calculate the total profit from selling candied apples and grapes -/
theorem candied_fruit_profit
  (apple_count : ℕ)
  (grape_count : ℕ)
  (apple_price : ℚ)
  (grape_price : ℚ)
  (apple_cost : ℚ)
  (grape_cost : ℚ)
  (h1 : apple_count = 15)
  (h2 : grape_count = 12)
  (h3 : apple_price = 2)
  (h4 : grape_price = 3/2)
  (h5 : apple_cost = 6/5)
  (h6 : grape_cost = 9/10)
  : (apple_count : ℚ) * (apple_price - apple_cost) + (grape_count : ℚ) * (grape_price - grape_cost) = 96/5 := by
  sorry

#eval (15 : ℚ) * (2 - 6/5) + (12 : ℚ) * (3/2 - 9/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candied_fruit_profit_l616_61620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_l616_61653

/-- Represents a trapezoid with bases x and y, where x < y -/
structure Trapezoid where
  x : ℝ
  y : ℝ
  h_pos : 0 < x
  h_order : x < y

/-- The midline of a trapezoid -/
noncomputable def midline (t : Trapezoid) : ℝ := (t.x + t.y) / 2

/-- The area of a trapezoid -/
noncomputable def area (t : Trapezoid) (h : ℝ) : ℝ := (t.x + t.y) * h / 2

/-- The area of the smaller trapezoid formed by the midline -/
noncomputable def area_small (t : Trapezoid) (h : ℝ) : ℝ := (t.x + midline t) * h / 4

/-- The area of the larger trapezoid formed by the midline -/
noncomputable def area_large (t : Trapezoid) (h : ℝ) : ℝ := (t.y + midline t) * h / 4

theorem trapezoid_bases (t : Trapezoid) (h : ℝ) (h_pos : 0 < h) :
  midline t = 10 ∧ 
  area_small t h / area_large t h = 3 / 5 →
  t.x = 5 ∧ t.y = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_l616_61653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_20_sided_polygon_l616_61662

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  -- Additional properties of convex polygon can be added here if needed

/-- An arithmetic sequence of natural numbers with a given length, first term, and common difference -/
def ArithmeticSequence (length first diff : ℕ) : List ℕ :=
  List.range length |>.map (fun i => first + i * diff)

/-- The theorem statement -/
theorem smallest_angle_20_sided_polygon :
  ∀ (p : ConvexPolygon 20) (seq : List ℕ),
    seq.length = 20 →
    seq = ArithmeticSequence 20 (seq.head!) 1 →
    seq.sum = (20 - 2) * 180 →
    seq.Pairwise (· < ·) →
    seq.head! = 143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_20_sided_polygon_l616_61662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_positive_integer_solution_l616_61647

theorem only_positive_integer_solution : 
  ∃! (x : ℕ), x > 0 ∧ (5 * (x : ℚ) + 1) / ((x : ℚ) - 1) > 2 * (x : ℚ) + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_positive_integer_solution_l616_61647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opera_house_earnings_l616_61681

-- Define the sections
structure Section where
  rows : Nat
  price : Nat
  occupancy : Rat

-- Define the opera house
structure OperaHouse where
  sectionA : Section
  sectionB : Section
  sectionC : Section
  seatsPerRow : Nat
  convenienceFee : Nat

def operaHouse : OperaHouse := {
  sectionA := { rows := 50, price := 20, occupancy := 9/10 },
  sectionB := { rows := 60, price := 15, occupancy := 3/4 },
  sectionC := { rows := 40, price := 10, occupancy := 7/10 },
  seatsPerRow := 10,
  convenienceFee := 3
}

-- Calculate earnings for a section
def sectionEarnings (s : Section) (seatsPerRow : Nat) (convenienceFee : Nat) : ℚ :=
  (s.rows : ℚ) * (seatsPerRow : ℚ) * ((s.price : ℚ) + (convenienceFee : ℚ)) * s.occupancy

-- Calculate total earnings
def totalEarnings (oh : OperaHouse) : ℚ :=
  sectionEarnings oh.sectionA oh.seatsPerRow oh.convenienceFee +
  sectionEarnings oh.sectionB oh.seatsPerRow oh.convenienceFee +
  sectionEarnings oh.sectionC oh.seatsPerRow oh.convenienceFee

-- Theorem statement
theorem opera_house_earnings :
  totalEarnings operaHouse = 22090 := by
  sorry

#eval totalEarnings operaHouse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opera_house_earnings_l616_61681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l616_61626

/-- Calculates the length of a bridge given train parameters and crossing time. -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

/-- Proves that the length of the bridge is 235 meters given the specified conditions. -/
theorem bridge_length_calculation :
  bridge_length 140 45 30 = 235 := by
  sorry

/-- Evaluates the bridge length using rational approximations for better computability -/
def bridge_length_rat (train_length : ℚ) (train_speed_kmh : ℚ) (crossing_time_s : ℚ) : ℚ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

#eval bridge_length_rat 140 45 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l616_61626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_lattice_toothpicks_l616_61680

/-- The total number of toothpicks in a rectangular lattice with a diagonal -/
def total_toothpicks (length width : ℕ) : ℕ :=
  let vertical := (length + 1) * width
  let horizontal := (width + 1) * length
  let diagonal := Int.sqrt (length^2 + width^2)
  vertical + horizontal + diagonal.toNat

/-- Theorem: A 30x40 rectangular lattice with a diagonal uses 2520 toothpicks -/
theorem rectangle_lattice_toothpicks :
  total_toothpicks 30 40 = 2520 := by
  sorry

#eval total_toothpicks 30 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_lattice_toothpicks_l616_61680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l616_61619

/-- The minimum value of a for which there exist integers b and c and a real x 
    such that the angle between OA and OB is π/2, and x has exactly two distinct 
    solutions in (0,1), where O(0,0,0), A(a,b,c), and B(x^2,x,1) are points in 
    3D coordinate system. -/
theorem min_a_value : ∃ (a : ℕ), a = 5 ∧
  ∃ (b c : ℤ) (x : ℝ),
  (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ∧  -- Dot product condition
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 0 < x₁ ∧ x₁ < 1 ∧ 0 < x₂ ∧ x₂ < 1 ∧
   (a : ℝ) * x₁^2 + (b : ℝ) * x₁ + (c : ℝ) = 0 ∧
   (a : ℝ) * x₂^2 + (b : ℝ) * x₂ + (c : ℝ) = 0) ∧
  (∀ (a' : ℕ), a' < a → ¬(∃ (b' c' : ℤ) (x' : ℝ),
   (a' : ℝ) * x'^2 + (b' : ℝ) * x' + (c' : ℝ) = 0 ∧
   (∃ (x₁' x₂' : ℝ), x₁' ≠ x₂' ∧ 0 < x₁' ∧ x₁' < 1 ∧ 0 < x₂' ∧ x₂' < 1 ∧
   (a' : ℝ) * x₁'^2 + (b' : ℝ) * x₁' + (c' : ℝ) = 0 ∧
   (a' : ℝ) * x₂'^2 + (b' : ℝ) * x₂' + (c' : ℝ) = 0))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l616_61619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_B_l616_61607

theorem cosine_angle_B (a b : ℝ) (A : ℝ) (h1 : a = 15) (h2 : b = 10) (h3 : A = π/3) :
  let B := Real.arcsin (b * Real.sin A / a)
  Real.cos B = Real.sqrt 6 / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_B_l616_61607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tims_ratio_is_two_to_one_l616_61638

/-- Tim's daily activities -/
structure TimsSchedule where
  meditationTime : ℚ  -- Daily meditation time in hours
  weeklyReadingTime : ℚ  -- Weekly reading time in hours

/-- Calculate the ratio of daily reading time to daily meditation time -/
def readingToMeditationRatio (schedule : TimsSchedule) : ℚ :=
  (schedule.weeklyReadingTime / 7) / schedule.meditationTime

/-- Theorem: Tim's reading to meditation ratio is 2:1 -/
theorem tims_ratio_is_two_to_one (schedule : TimsSchedule)
  (h1 : schedule.meditationTime = 1)
  (h2 : schedule.weeklyReadingTime = 14) :
  readingToMeditationRatio schedule = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tims_ratio_is_two_to_one_l616_61638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l616_61610

/-- A triangle with two medians intersecting at right angles -/
structure RightAngledMedianTriangle where
  /-- The length of the first median -/
  XU : ℝ
  /-- The length of the second median -/
  YV : ℝ
  /-- The medians intersect at right angles -/
  medians_perpendicular : True

/-- The area of a triangle with two medians intersecting at right angles -/
noncomputable def area (t : RightAngledMedianTriangle) : ℝ :=
  (t.XU * t.YV) / 2

/-- Theorem: If XU = 18 and YV = 24 in a triangle with perpendicular medians, its area is 288 -/
theorem area_of_specific_triangle :
  ∀ (t : RightAngledMedianTriangle), t.XU = 18 → t.YV = 24 → area t = 288 := by
  intro t h1 h2
  unfold area
  rw [h1, h2]
  norm_num
  sorry

#check area_of_specific_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l616_61610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_neg_two_range_of_a_single_zero_l616_61631

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - (2 * a + 1) * x

-- Part I: Maximum value when a = -2
theorem max_value_when_a_neg_two :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f (-2) y ≤ f (-2) x ∧ f (-2) x = 1 := by
  sorry

-- Part II: Range of a when f has only one zero in (0, e)
theorem range_of_a_single_zero (a : ℝ) :
  a < (1/2) →
  (∃! (x : ℝ), 0 < x ∧ x < Real.exp 1 ∧ f a x = 0) →
  a < (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_neg_two_range_of_a_single_zero_l616_61631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_remainder_l616_61630

def f (n : ℕ) : ℕ := Nat.minFac (n^4 + 1)

theorem sum_f_remainder (N : ℕ) (h : N = 2014) : 
  (Finset.range N).sum f % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_remainder_l616_61630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_absolute_value_l616_61675

open Complex Filter Topology

theorem limit_absolute_value (z : ℕ → ℂ) (a : ℂ) :
  Tendsto z atTop (𝓝 a) →
  Tendsto (fun n => Complex.abs (z n)) atTop (𝓝 (Complex.abs a)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_absolute_value_l616_61675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l616_61684

-- Define the operation ⊕
noncomputable def op (a b : ℝ) : ℝ := (Real.sqrt (3 * a + 2 * b)) ^ 2

-- Theorem statement
theorem solve_equation (x : ℝ) : op 3 x = 49 → x = 20 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l616_61684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l616_61659

theorem log_inequality : ∃ (a b c : ℝ), 
  a = (Real.log 3) / (Real.log 4) ∧ 
  b = Real.log 3 ∧ 
  c = Real.sqrt 10 ∧ 
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l616_61659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_list_median_l616_61629

/-- Represents a list of integers with specific properties -/
structure SpecialList where
  elements : List Int
  mode : Int
  mean : Int
  smallest : Int
  median : Int

/-- Replace the median in the list with a new value -/
def replaceMedian (L : SpecialList) (newMedian : Int) : SpecialList :=
  { L with 
    elements := L.elements.map (fun x => if x = L.median then newMedian else x),
    median := newMedian
  }

/-- The properties of our special list -/
def specialListProperties (L : SpecialList) : Prop :=
  L.mode = 40 ∧
  L.mean = 35 ∧
  L.smallest = 20 ∧
  L.median ∈ L.elements ∧
  (replaceMedian L (L.median + 15)).mean = 40 ∧
  (replaceMedian L (L.median + 15)).median = L.median + 15 ∧
  (replaceMedian L (L.median - 10)).median = L.median - 5

/-- The theorem stating that the median of our special list must be 40 -/
theorem special_list_median (L : SpecialList) : 
  specialListProperties L → L.median = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_list_median_l616_61629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_is_real_l616_61688

/-- The function f(x) = e^x - 1 -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1

/-- The function g(x) = -x^2 + 4x - 3 -/
def g (x : ℝ) : ℝ := -x^2 + 4*x - 3

/-- Theorem stating that if f(a) = g(b) for some a, then b can be any real number -/
theorem range_of_b_is_real : ∀ (b : ℝ), ∃ (a : ℝ), f a = g b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_is_real_l616_61688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_P_l616_61669

/-- The circle with center (1, 0) and radius 5 -/
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

/-- The point P on the circle -/
def P : ℝ × ℝ := (2, -1)

/-- The equation of the line on which the shortest chord passing through P lies -/
def shortest_chord_line (x y : ℝ) : Prop := x - y - 3 = 0

/-- The theorem stating that the shortest chord passing through P lies on the line x - y - 3 = 0 -/
theorem shortest_chord_through_P :
  ∀ x y : ℝ, my_circle x y → (∃ t : ℝ, x = P.1 + t ∧ y = P.2 + t) → 
  shortest_chord_line x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_P_l616_61669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_have_three_common_tangents_l616_61686

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 9 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 1
def center2 : ℝ × ℝ := (3, 4)
def radius2 : ℝ := 4

-- Define the distance between centers
def distance_between_centers : ℝ := 5

-- Define a function to represent the number of common tangents
def number_of_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem stating that the circles have exactly 3 common tangents
theorem circles_have_three_common_tangents :
  (∀ x y : ℝ, circle1 x y ↔ (x - center1.1)^2 + (y - center1.2)^2 = radius1^2) ∧
  (∀ x y : ℝ, circle2 x y ↔ (x - center2.1)^2 + (y - center2.2)^2 = radius2^2) ∧
  distance_between_centers = radius1 + radius2 →
  ∃! n : ℕ, n = 3 ∧ n = number_of_common_tangents circle1 circle2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_have_three_common_tangents_l616_61686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l616_61661

noncomputable def original_function (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

noncomputable def shift_amount : ℝ := Real.pi / 4

noncomputable def shifted_function (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3)

theorem shift_equivalence :
  ∀ x : ℝ, original_function (x - shift_amount) = shifted_function x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_equivalence_l616_61661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_solution_l616_61613

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The polynomial 6x^2 + 2x + 1 -/
def h (x : ℝ) : ℝ := 6*x^2 + 2*x + 1

/-- Predicate to check if a function is a polynomial -/
def IsPolynomial (g : ℝ → ℝ) : Prop := sorry

theorem no_polynomial_solution :
  ¬ ∃ g : ℝ → ℝ, IsPolynomial g ∧ (∀ x, f (g x) = h x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_solution_l616_61613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_l616_61614

def product_sequence : List ℚ := [1/3, 9/1, 1/27, 81/1, 1/243, 729/1, 1/2187, 6561/1, 1/6561, 19683/1]

theorem sequence_product : (product_sequence.map (λ x => Rat.cast x : ℚ → ℝ)).prod = 729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_l616_61614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_parts_sqrt_7_parts_custom_op_result_l616_61683

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ :=
  Int.floor x

-- Define the decimal part function
noncomputable def decPart (x : ℝ) : ℝ :=
  x - Int.floor x

-- Define the custom operation
def customOp (a b : ℝ) : ℝ :=
  |a - b|

theorem sqrt_5_parts :
  intPart (Real.sqrt 5) = 2 ∧ decPart (Real.sqrt 5) = Real.sqrt 5 - 2 := by
  sorry

theorem sqrt_7_parts :
  let m := decPart (Real.sqrt 7)
  let n := intPart (Real.sqrt 7)
  m + n - Real.sqrt 7 = 0 := by
  sorry

theorem custom_op_result :
  let a := Real.sqrt 3
  let b := intPart (Real.sqrt 10)
  customOp a b + a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_5_parts_sqrt_7_parts_custom_op_result_l616_61683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_distances_l616_61673

/-- A rectangle with odd side lengths. -/
structure OddRectangle where
  a : ℕ
  b : ℕ
  a_odd : Odd a
  b_odd : Odd b

/-- A point inside a rectangle. -/
structure InnerPoint (r : OddRectangle) where
  x : ℝ
  y : ℝ
  x_bounds : 0 < x ∧ x < r.a
  y_bounds : 0 < y ∧ y < r.b

/-- The distances from an inner point to the four vertices of the rectangle. -/
noncomputable def distances (r : OddRectangle) (p : InnerPoint r) : Fin 4 → ℝ
| 0 => Real.sqrt (p.x^2 + p.y^2)
| 1 => Real.sqrt ((r.a - p.x)^2 + p.y^2)
| 2 => Real.sqrt (p.x^2 + (r.b - p.y)^2)
| 3 => Real.sqrt ((r.a - p.x)^2 + (r.b - p.y)^2)

/-- The main theorem: there is no point inside an odd rectangle with integer distances to all vertices. -/
theorem no_integer_distances (r : OddRectangle) :
  ¬∃ (p : InnerPoint r), ∀ (i : Fin 4), ∃ (n : ℤ), (distances r p i) = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_distances_l616_61673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_sequence_contains_all_combinations_l616_61617

/-- A sequence of digits -/
def Sequence := List Nat

/-- Check if a sequence contains a specific combination -/
def containsCombination (s : Sequence) (c : Sequence) : Prop :=
  ∃ i, c = (s.drop i).take c.length

/-- All possible three-digit combinations of 1, 2, and 3 -/
def allCombinations : List Sequence :=
  [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

/-- The proposed shortest sequence -/
def shortestSequence : Sequence := [1,2,3,1,2,1,3,2,1]

theorem shortest_sequence_contains_all_combinations :
  (∀ c ∈ allCombinations, containsCombination shortestSequence c) ∧
  (∀ s : Sequence, s.length < shortestSequence.length →
    ∃ c ∈ allCombinations, ¬containsCombination s c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_sequence_contains_all_combinations_l616_61617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_is_27_l616_61644

/-- Represents a star diagram with integer values at each point. -/
structure StarDiagram where
  p : Int
  q : Int
  r : Int
  s : Int
  u : Int
  a : Int
  b : Int
  c : Int
  d : Int
  e : Int

/-- The condition that the sum along each line is constant. -/
def line_sum_constant (star : StarDiagram) (S : Int) : Prop :=
  star.a + star.p + star.q + star.b = S ∧
  star.c + star.p + star.u + star.e = S ∧
  star.c + star.q + star.r + star.d = S ∧
  star.a + star.u + star.s + star.d = S ∧
  star.e + star.s + star.r + star.b = S

/-- The theorem stating that q must be 27 under the given conditions. -/
theorem q_is_27 (star : StarDiagram) (S : Int) :
  star.a = 9 ∧ star.b = 7 ∧ star.c = 3 ∧ star.d = 11 ∧ star.e = 15 ∧
  line_sum_constant star S ∧
  ({star.p, star.q, star.r, star.s, star.u} : Finset Int) = {19, 21, 23, 25, 27} →
  star.q = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_is_27_l616_61644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_triangle_points_l616_61628

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Represents the foci of an ellipse -/
structure Foci where
  f₁ : Point
  f₂ : Point

/-- Checks if a triangle formed by three points is a right triangle -/
def isRightTriangle (p₁ p₂ p₃ : Point) : Prop :=
  ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (p₁.x - p₂.x)^2 + (p₁.y - p₂.y)^2 + (p₂.x - p₃.x)^2 + (p₂.y - p₃.y)^2 =
    (p₁.x - p₃.x)^2 + (p₁.y - p₃.y)^2

/-- The main theorem -/
theorem ellipse_right_triangle_points :
  ∃! (s : Finset Point),
    let e : Ellipse := ⟨2, Real.sqrt 2⟩
    let foci : Foci := ⟨⟨-Real.sqrt 2, 0⟩, ⟨Real.sqrt 2, 0⟩⟩
    (∀ p ∈ s, isOnEllipse p e ∧ isRightTriangle p foci.f₁ foci.f₂) ∧
    s.card = 6 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_triangle_points_l616_61628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_line_equation_l616_61696

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem ellipse_line_intersection :
  ∀ (k x1 y1 x2 y2 : ℝ),
  (∀ x y, ellipse_C x y ↔ x^2/4 + y^2/3 = 1) →
  (∀ x y, line_l k x y ↔ y = k*x + 1) →
  ellipse_C x1 y1 →
  ellipse_C x2 y2 →
  line_l k x1 y1 →
  line_l k x2 y2 →
  distance x1 y1 x2 y2 = 3 * Real.sqrt 5 / 2 →
  (k = 1/2 ∨ k = -1/2) :=
by sorry

theorem line_equation :
  ∀ (x y : ℝ),
  (x - 2*y + 2 = 0 ∨ x + 2*y - 2 = 0) ↔
  (∃ k, k = 1/2 ∨ k = -1/2) ∧ (y = k*x + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_line_equation_l616_61696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l616_61671

noncomputable def my_sequence (n : ℝ) : ℝ := (8 * n^5 - 2 * n) / ((n + 1)^4 - (n - 1)^4)

theorem my_sequence_limit : 
  ∀ ε > 0, ∃ N : ℝ, ∀ n ≥ N, |my_sequence n - 4| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l616_61671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l616_61637

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (b - Real.exp (-Real.log 2 * x)) / (Real.exp (-Real.log 2 * (x-1)) + 2)

-- State the theorem
theorem odd_function_properties :
  ∃ b : ℝ, 
    (∀ x : ℝ, f b x = -f b (-x)) ∧  -- f is an odd function
    (b = 1) ∧  -- part 1
    (∀ x y : ℝ, x < y → f b x < f b y) ∧  -- part 2: f is increasing
    (∀ k : ℝ, (∃ t : ℝ, f b (t^2 - 2*t) + f b (2*t^2 - k) < 0) → k > -1/3)  -- part 3
  := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l616_61637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_inequality_l616_61676

noncomputable def f (x : ℝ) : ℝ := 1 + 1/x

theorem inverse_f_inequality (x : ℝ) :
  (∃ y > 0, f y = x) → (∃ y, Function.invFun f x = y ∧ y > 2) ↔ x ∈ Set.Ioo 1 (3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_inequality_l616_61676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orphanage_parent_assignment_l616_61640

structure Orphanage where
  V : Type
  E : V → V → Prop
  friend_or_enemy : ∀ x y : V, x ≠ y → (E x y ∨ ¬E x y)
  friend_triangle_condition : ∀ x y z w : V, 
    E x y ∧ E x z ∧ E x w → 
    ((E y z ∧ E y w ∧ ¬E z w) ∨ (E y z ∧ ¬E y w ∧ E z w) ∨ (¬E y z ∧ E y w ∧ E z w))

structure Parent (o : Orphanage) where
  P : Type
  assign : o.V → P × P
  friend_share_one : ∀ x y : o.V, o.E x y → 
    (assign x).1 = (assign y).1 ∨ (assign x).1 = (assign y).2 ∨
    (assign x).2 = (assign y).1 ∨ (assign x).2 = (assign y).2
  enemy_share_none : ∀ x y : o.V, ¬o.E x y → 
    (assign x).1 ≠ (assign y).1 ∧ (assign x).1 ≠ (assign y).2 ∧
    (assign x).2 ≠ (assign y).1 ∧ (assign x).2 ≠ (assign y).2
  no_love_triangle : ∀ p q r : P, ∀ x y z : o.V,
    (assign x = (p, q) ∧ assign y = (q, r) ∧ assign z = (r, p)) → False

theorem orphanage_parent_assignment (o : Orphanage) : 
  ∃ (p : Parent o), True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orphanage_parent_assignment_l616_61640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pentagon_angle_l616_61690

-- Define the circle
variable (C : Set (ℝ × ℝ))

-- Define the triangle
variable (T : Set (ℝ × ℝ))

-- Define the pentagon
variable (P : Set (ℝ × ℝ))

-- Define the shared vertex
variable (A : ℝ × ℝ)

-- Define predicates
def Equilateral (T : Set (ℝ × ℝ)) : Prop := sorry
def Regular (P : Set (ℝ × ℝ)) : Prop := sorry
def InscribedIn (S T : Set (ℝ × ℝ)) : Prop := sorry
def AngleMeasure (A : ℝ × ℝ) (T P : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem triangle_pentagon_angle (hT : Equilateral T) (hP : Regular P) 
  (hInscribedT : InscribedIn T C) (hInscribedP : InscribedIn P C) 
  (hShared : A ∈ T ∩ P) :
  AngleMeasure A T P = 114 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pentagon_angle_l616_61690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_theorem_l616_61672

/-- A right triangle is defined by three side lengths where one angle is a right angle. -/
structure RightTriangle (a b c : ℝ) : Prop where
  right_angle : a^2 + b^2 = c^2
  positive_a : a > 0
  positive_b : b > 0
  positive_c : c > 0

/-- The Pythagorean theorem states that in a right triangle,
    the square of the length of the hypotenuse is equal to
    the sum of squares of the lengths of the other two sides. -/
theorem pythagorean_theorem (a b c : ℝ) (h : RightTriangle a b c) :
  a^2 + b^2 = c^2 := by
  exact h.right_angle


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_theorem_l616_61672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_h_zero_point_l616_61608

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4^x + a) / (2^(x+1))

noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := 2 * f a x - a * x - b

theorem f_odd_and_h_zero_point (a b : ℝ) :
  (∀ x, f a (-x) = -f a x) →
  (∃ x ∈ Set.Icc (-1) 1, h a b x = 0) →
  a = -1 ∧ b ∈ Set.Icc (-5/2) (5/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_h_zero_point_l616_61608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l616_61651

/-- A set of points in a plane where each point is the midpoint of two other points in the set -/
def MidpointSet (S : Set (ℝ × ℝ)) : Prop :=
  ∀ p, p ∈ S → ∃ a b, a ∈ S ∧ b ∈ S ∧ p = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

/-- Theorem stating that a MidpointSet contains infinitely many points -/
theorem midpoint_set_infinite (S : Set (ℝ × ℝ)) (h : MidpointSet S) : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l616_61651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_sequence_l616_61679

def arithmetic_sequence (a₁ d : ℕ) : ℕ → ℕ := fun i => a₁ + (i - 1) * d

theorem arithmetic_mean_of_sequence :
  let a := arithmetic_sequence 5 1
  let sum := (Finset.range 49).sum (fun i => a (i + 1))
  sum / 49 = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_sequence_l616_61679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_ranking_l616_61691

/-- Represents the teams in the volleyball championship --/
inductive Team : Type
  | A | B | D | E | V | G

/-- Represents the ranking of teams --/
def Ranking := Team → Nat

/-- Checks if a ranking satisfies the given conditions --/
def valid_ranking (r : Ranking) : Prop :=
  r Team.A = r Team.B + 3 ∧
  r Team.E < r Team.D ∧
  r Team.E < r Team.B ∧
  r Team.V < r Team.G ∧
  (∀ t : Team, r t ∈ Finset.range 7 \ {0}) ∧
  (∀ t1 t2 : Team, t1 ≠ t2 → r t1 ≠ r t2)

/-- The correct ranking of teams --/
def correct_ranking : Ranking :=
  fun t => match t with
    | Team.D => 1
    | Team.E => 2
    | Team.B => 3
    | Team.V => 4
    | Team.G => 5
    | Team.A => 6

/-- Theorem stating that the correct_ranking is the only valid ranking --/
theorem unique_valid_ranking :
  valid_ranking correct_ranking ∧
  (∀ r : Ranking, valid_ranking r → r = correct_ranking) := by
  sorry

#check unique_valid_ranking

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_ranking_l616_61691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unsweepable_district_l616_61604

/-- Represents a bulldozer with its size and direction -/
structure Bulldozer where
  size : ℕ
  direction : Bool  -- true for right, false for left

/-- Represents a district with two bulldozers -/
structure District where
  left_bulldozer : Bulldozer
  right_bulldozer : Bulldozer

/-- Defines the sweeping relation between districts -/
def can_sweep (n : ℕ) (districts : Fin n → District) (i j : Fin n) : Prop :=
  sorry

/-- The main theorem statement -/
theorem exists_unsweepable_district (n : ℕ) (h : n ≥ 1) 
  (districts : Fin n → District) 
  (distinct_sizes : ∀ i j : Fin n, i ≠ j → 
    (districts i).left_bulldozer.size ≠ (districts j).left_bulldozer.size ∧ 
    (districts i).left_bulldozer.size ≠ (districts j).right_bulldozer.size ∧
    (districts i).right_bulldozer.size ≠ (districts j).left_bulldozer.size ∧
    (districts i).right_bulldozer.size ≠ (districts j).right_bulldozer.size) :
  ∃ k : Fin n, ∀ j : Fin n, j ≠ k → ¬(can_sweep n districts j k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unsweepable_district_l616_61604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_marks_combined_classes_l616_61606

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = (n1 * avg1 + n2 * avg2) / (n1 + n2) :=
by sorry

-- Example usage with the given values
example : 
  (25 : ℕ) * (40 : ℚ) + (30 : ℕ) * (60 : ℚ) / ((25 : ℕ) + (30 : ℕ)) = 2800 / 55 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_marks_combined_classes_l616_61606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_transformed_functions_l616_61658

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the two transformed functions
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 1)
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-x + 1)

-- Define symmetry with respect to the line x = 1
def symmetric_about_x_equals_1 (g h : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (1 + (x - 1)) = h (1 - (x - 1))

-- Theorem statement
theorem symmetry_of_transformed_functions (f : ℝ → ℝ) :
  symmetric_about_x_equals_1 (g f) (h f) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_transformed_functions_l616_61658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_cost_l616_61603

/-- The cost of a pencil in cents, given that:
    - 5 pens and 4 pencils cost $2.90
    - 3 pens and 6 pencils cost $2.18
-/
theorem pencil_cost : ∃ (pen_cost pencil_cost : ℚ),
  (5 * pen_cost + 4 * pencil_cost = 290) ∧
  (3 * pen_cost + 6 * pencil_cost = 218) ∧
  (pencil_cost = 110 / 9) := by
  sorry

#check pencil_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_cost_l616_61603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_one_problem_two_l616_61698

-- Problem 1
theorem problem_one (x : ℝ) 
  (h1 : Real.cos (π/4 + x) = 3/5) 
  (h2 : 17*π/12 < x ∧ x < 7*π/4) : 
  (Real.sin (2*x) + 2*(Real.sin x)^2) / (1 - Real.tan x) = -28/75 := by sorry

-- Problem 2
theorem problem_two (x₀ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = 2*Real.sqrt 3 * Real.sin x * Real.cos x + 2*(Real.cos x)^2 - 1)
  (h2 : f x₀ = 6/5)
  (h3 : x₀ ∈ Set.Icc (π/4) (π/2)) : 
  Real.cos (2*x₀) = (3 - 4*Real.sqrt 3)/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_one_problem_two_l616_61698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_asymptote_intersection_l616_61621

noncomputable section

/-- The function f(x) = (x^2 - 9x + 20) / (x^2 - 9x + 18) -/
def f (x : ℝ) : ℝ := (x^2 - 9*x + 20) / (x^2 - 9*x + 18)

/-- The set of vertical asymptotes -/
def vertical_asymptotes : Set ℝ := {3, 6}

/-- The y-value of the horizontal asymptote -/
def horizontal_asymptote : ℝ := 1

/-- Theorem: There is no intersection point of asymptotes for the function f -/
theorem no_asymptote_intersection :
  ∀ x y : ℝ, x ∈ vertical_asymptotes → y = horizontal_asymptote → ¬ (f x = y) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_asymptote_intersection_l616_61621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l616_61665

def sequence_sum : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => sequence_sum (n + 1) + (n + 2) + 2

def a : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => n + 4

theorem sum_of_first_five_terms :
  sequence_sum 5 = 23 :=
by
  -- The proof goes here
  sorry

#eval sequence_sum 5  -- This will evaluate the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l616_61665
