import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relations_l866_86676

/-- Given a triangle ABC with the specified properties, prove the relationships between angles -/
theorem triangle_angle_relations (A B C : ℝ) (AB AC BC : ℝ) : 
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Given condition
  AB * AC * Real.cos A = 3 * AB * BC * Real.cos B →
  -- Cosine of angle C
  Real.cos C = Real.sqrt 5 / 5 →
  -- Prove these two statements
  (Real.tan B = 3 * Real.tan A) ∧ (A = Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relations_l866_86676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_equation_l866_86606

/-- Given a line l₁ with equation y = x + 1, prove that the line l₂ that is
    symmetric to l₁ about point (1, 1) has the equation x - y - 1 = 0. -/
theorem symmetric_line_equation :
  ∃ (l₁ l₂ : Set (ℝ × ℝ)),
  l₁ = {(x, y) : ℝ × ℝ | y = x + 1} ∧
  l₂ = {(x, y) : ℝ × ℝ | ∀ (x' y' : ℝ), (2 - x, 2 - y) ∈ l₁ ↔ (x, y) ∈ l₂} ∧
  ∀ (x y : ℝ), (x, y) ∈ l₂ ↔ x - y - 1 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_equation_l866_86606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subtract_repeating_decimal_l866_86699

/-- Given that 0.̅6 is a repeating decimal, prove that 2 - 0.̅6 = 4/3 -/
theorem subtract_repeating_decimal :
  ∃ (x : ℚ), (∀ n : ℕ, (x * 10^n - (x * 10^n).floor = 0.6)) → 2 - x = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subtract_repeating_decimal_l866_86699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l866_86622

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

-- Define the interval
def I : Set ℝ := Set.Icc (-3) 2

-- State the theorem
theorem min_t_value (t : ℝ) : 
  (∀ x1 x2, x1 ∈ I → x2 ∈ I → |f x1 - f x2| ≤ t) ↔ t ≥ 20 := by
  sorry

#check min_t_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l866_86622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_proof_l866_86618

/-- The cubic equation whose roots are α, β, and γ -/
def cubic_equation (x : ℝ) : ℝ := x^3 - 4*x^2 + 6*x + 8

/-- P is a cubic polynomial satisfying the given conditions -/
noncomputable def P : ℝ → ℝ := sorry

/-- α is a root of the cubic equation -/
noncomputable def α : ℝ := sorry

/-- β is a root of the cubic equation -/
noncomputable def β : ℝ := sorry

/-- γ is a root of the cubic equation -/
noncomputable def γ : ℝ := sorry

theorem cubic_polynomial_proof :
  (cubic_equation α = 0) ∧
  (cubic_equation β = 0) ∧
  (cubic_equation γ = 0) ∧
  (P α = β + γ) ∧
  (P β = α + γ) ∧
  (P γ = α + β) ∧
  (P (α + β + γ) = -20) →
  P = fun x => -5/8 * x^3 + 5/2 * x^2 + 1/8 * x - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_proof_l866_86618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_cases_l866_86632

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

theorem triangle_area_cases (t : Triangle)
  (h1 : t.a + 1/t.a = 4 * Real.cos t.C)
  (h2 : t.b = 1) :
  (t.A = π/2 → area t = Real.sqrt 2 / 2) ∧
  (area t = Real.sqrt 3 / 2 → t.a = Real.sqrt 7 ∧ t.c = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_cases_l866_86632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coef_ratio_neg_three_halves_fifty_l866_86636

/-- Definition of binomial coefficient for real a and positive integer k -/
noncomputable def binomial_coef (a : ℝ) (k : ℕ+) : ℝ :=
  (Finset.range k.val).prod (fun i => (a - i) / (k.val - i))

/-- The main theorem to prove -/
theorem binomial_coef_ratio_neg_three_halves_fifty :
  binomial_coef (-3/2) 50 / binomial_coef (3/2) 50 = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coef_ratio_neg_three_halves_fifty_l866_86636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l866_86602

-- Define the propositions p and q
def p (x : ℝ) : Prop := 0 < x ∧ x < 1
def q (x a : ℝ) : Prop := (x - a) * (x - (a + 2)) ≤ 0

-- Define the sufficiency condition
def sufficient (a : ℝ) : Prop := ∀ x, p x → q x a

-- Define the not necessary condition
def not_necessary (a : ℝ) : Prop := ∃ x, q x a ∧ ¬(p x)

-- State the theorem
theorem range_of_a : 
  (∀ a, sufficient a ∧ not_necessary a) → 
  (∀ a, a ∈ Set.Icc (-1) 0 ↔ sufficient a ∧ not_necessary a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l866_86602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seokgi_has_5000_l866_86679

/-- The price of the boat in won -/
def boat_price : ℕ := sorry

/-- Seokgi's money in won -/
def seokgi_money : ℕ := sorry

/-- Ye-seul's money in won -/
def yeseul_money : ℕ := sorry

/-- Seokgi's money is short of 2,000 won to buy the boat -/
axiom seokgi_short : boat_price = seokgi_money + 2000

/-- Ye-seul's money is short of 1,500 won to buy the boat -/
axiom yeseul_short : boat_price = yeseul_money + 1500

/-- After buying the boat with their combined money, 3,500 won is left -/
axiom money_left : seokgi_money + yeseul_money = boat_price + 3500

/-- Theorem: Given the conditions, Seokgi's money is 5,000 won -/
theorem seokgi_has_5000 : seokgi_money = 5000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seokgi_has_5000_l866_86679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_conditions_l866_86667

/-- Ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

/-- Line equation -/
def is_on_line (x y m : ℝ) : Prop := y = x + m

/-- Tangent line condition -/
def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), is_on_ellipse x y ∧ is_on_line x y m ∧
  ∀ (x' y' : ℝ), is_on_ellipse x' y' ∧ is_on_line x' y' m → x' = x ∧ y' = y

/-- Intersection points with distance equal to minor axis -/
def intersects_with_minor_axis_distance (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    is_on_ellipse x₁ y₁ ∧ is_on_line x₁ y₁ m ∧
    is_on_ellipse x₂ y₂ ∧ is_on_line x₂ y₂ m ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4

theorem ellipse_intersection_conditions (m : ℝ) :
  (is_tangent m ↔ m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
  (intersects_with_minor_axis_distance m ↔ m = Real.sqrt 30 / 4 ∨ m = -Real.sqrt 30 / 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_conditions_l866_86667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_range_l866_86674

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 - 8 * x - 8 * y - 1 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x + y - 9 = 0

-- Define the center of the circle
noncomputable def center_M : ℝ × ℝ := (2, 2)

-- Define the radius of the circle
noncomputable def radius_M : ℝ := Real.sqrt (17/2)

-- Main theorem
theorem point_A_range (a : ℝ) :
  (∃ (b c : ℝ × ℝ),
    line_l a (9-a) ∧
    circle_M b.1 b.2 ∧
    circle_M c.1 c.2 ∧
    (∃ (k : ℝ), b = (center_M.1 + k * (a - center_M.1), center_M.2 + k * ((9-a) - center_M.2))) ∧
    Real.cos (π/4) * ((a - 2)^2 + (7 - a)^2) = (b.1 - c.1)^2 + (b.2 - c.2)^2) →
  3 ≤ 9 - a ∧ 9 - a ≤ 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_range_l866_86674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l866_86680

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : b * Real.sin A * Real.sin B = (3/4) * a * (Real.sin C)^2
axiom cos_A : Real.cos A = (5 * Real.sqrt 3) / 12
axiom area : (1/2) * a * c * Real.sin B = Real.sqrt 23

-- Define D as a point in ℝ × ℝ
variable (D : ℝ × ℝ)

-- Define the midpoint condition
def is_midpoint (D : ℝ × ℝ) (A C : ℝ × ℝ) : Prop :=
  D.1 = (A.1 + C.1) / 2 ∧ D.2 = (A.2 + C.2) / 2

-- State the theorem
theorem triangle_proof :
  Real.cos B = (3 * Real.sqrt 2) / 8 ∧
  ∃ (A C : ℝ × ℝ), is_midpoint D A C → Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l866_86680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_volume_calculation_l866_86690

/-- The volume of a cylindrical pool -/
noncomputable def pool_volume (diameter : ℝ) (depth : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * depth

/-- Theorem: The volume of a cylindrical pool with diameter 20 feet and depth 5 feet is 500π cubic feet -/
theorem pool_volume_calculation :
  pool_volume 20 5 = 500 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_volume_calculation_l866_86690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_in_range_l866_86627

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 4*x else 4*x - x^2

-- State the theorem
theorem f_inequality_iff_a_in_range (a : ℝ) : 
  f (2 - a^2) > f a ↔ a ∈ Set.Ioo (-2) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_in_range_l866_86627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l866_86659

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y - 6 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := Real.sqrt 11

/-- Theorem stating the equivalence between the given equation and the standard form of a circle -/
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y ↔ 
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by
  intros x y
  sorry  -- Proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l866_86659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l866_86697

noncomputable def f (x : ℝ) : ℝ := Real.cos (3 * x + Real.pi / 3) + Real.cos (3 * x - Real.pi / 3) + 2 * Real.sin (3 * x / 2) * Real.cos (3 * x / 2)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 6), f x ≤ Real.sqrt 2) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 6), f x ≥ -1) ∧
  (∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 6), f x = Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 6), f x = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l866_86697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_pi_approximation_accuracy_of_three_million_l866_86663

-- Define rounding to thousandth place
noncomputable def roundToThousandth (x : ℝ) : ℝ := 
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

-- Define scientific notation
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  normalForm : 1 ≤ |coefficient| ∧ |coefficient| < 10

-- Define accuracy place
inductive AccuracyPlace
  | Millions
  | HundredThousands
  | TenThousands
  | Thousands
  | Hundreds
  | Tens
  | Ones

-- Theorem for rounding 3.1415926
theorem round_pi_approximation :
  roundToThousandth 3.1415926 = 3.142 := by sorry

-- Theorem for accuracy of 3.0 × 10^6
theorem accuracy_of_three_million (n : ScientificNotation) 
  (h : n.coefficient = 3.0 ∧ n.exponent = 6) :
  AccuracyPlace.HundredThousands = AccuracyPlace.HundredThousands := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_pi_approximation_accuracy_of_three_million_l866_86663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_degree_three_l866_86631

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4
def g (x : ℝ) : ℝ := 3 - 2*x - 6*x^3 + 9*x^4

-- Define the combined polynomial h
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

-- Theorem stating that h has degree 3 when c = -5/9
theorem h_degree_three :
  ∃ (a b d : ℝ), (∀ x, h (-5/9) x = a + b*x + d*x^3) ∧ d ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_degree_three_l866_86631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l866_86681

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, |x - 1| < 2 → x^2 - 4*x - 5 < 0) ∧ 
  (∃ x : ℝ, x^2 - 4*x - 5 < 0 ∧ ¬(|x - 1| < 2)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l866_86681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l866_86685

/-- Given a point P(-4, m) on the terminal side of angle α, where sin α = 3/5, prove that m = 3 -/
theorem point_on_terminal_side (α : ℝ) (m : ℝ) : 
  (∃ P : ℝ × ℝ, P = (-4, m) ∧ P.1 = -4 ∧ P.2 = m) →
  Real.sin α = 3/5 →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l866_86685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_neg_one_a_value_for_max_neg_three_g_inequality_l866_86614

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * f a x

-- Theorem 1
theorem max_value_f_neg_one :
  ∃ (M : ℝ), M = -1 ∧ ∀ x > 0, f (-1) x ≤ M :=
sorry

-- Theorem 2
theorem a_value_for_max_neg_three :
  ∃ (a : ℝ), a = -Real.exp 2 ∧
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f a x ≤ -3) ∧
  (∃ x ∈ Set.Ioo 0 (Real.exp 1), f a x = -3) :=
sorry

-- Theorem 3
theorem g_inequality (a : ℝ) (h : a > 0) :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 2 * g a ((x₁ + x₂) / 2) < g a x₁ + g a x₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_neg_one_a_value_for_max_neg_three_g_inequality_l866_86614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_POI_l866_86613

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

structure Point where
  x : ℝ
  y : ℝ

def is_focus (p : Point) (a c : ℝ) : Prop := 
  (p.x = -c ∧ p.y = 0) ∨ (p.x = c ∧ p.y = 0)

noncomputable def is_incenter (i p f1 f2 : Point) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (i.x - p.x)^2 + (i.y - p.y)^2 = r^2 ∧
  (i.x - f1.x)^2 + (i.y - f1.y)^2 = r^2 ∧
  (i.x - f2.x)^2 + (i.y - f2.y)^2 = r^2

noncomputable def angle_tan (p1 p2 p3 : Point) : ℝ :=
  let v1 := (p2.x - p1.x, p2.y - p1.y)
  let v2 := (p3.x - p1.x, p3.y - p1.y)
  let cross := v1.1 * v2.2 - v1.2 * v2.1
  let dot := v1.1 * v2.1 + v1.2 * v2.2
  cross / dot

theorem max_tan_POI (a c : ℝ) (f1 f2 p i o : Point) :
  a > 0 → c > 0 → c < a →
  is_focus f1 a c →
  is_focus f2 a c →
  ellipse p.x p.y →
  first_quadrant p.x p.y →
  is_incenter i p f1 f2 →
  o.x = 0 ∧ o.y = 0 →
  ∃ (max_tan : ℝ), max_tan = Real.sqrt 6 / 12 ∧
    angle_tan o p i ≤ max_tan := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_POI_l866_86613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_f_l866_86647

/-- The function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- The closed interval [-2, 2] -/
def I : Set ℝ := Set.Icc (-2) 2

theorem min_max_f :
  (∃ (a : ℝ), a ∈ I ∧ ∀ (x : ℝ), x ∈ I → f x ≥ f a) ∧
  (∃ (b : ℝ), b ∈ I ∧ ∀ (x : ℝ), x ∈ I → f x ≤ f b) ∧
  (∃ (a : ℝ), a ∈ I ∧ f a = -3) ∧
  (∃ (b : ℝ), b ∈ I ∧ f b = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_f_l866_86647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_l866_86642

/-- The perimeter of a pentagon ABCDE with given side lengths -/
theorem pentagon_perimeter (AB BC CD DE AE : ℝ) 
  (h1 : AB = 2)
  (h2 : BC = Real.sqrt 5)
  (h3 : CD = Real.sqrt 3)
  (h4 : DE = Real.sqrt 5)
  (h5 : AE = 3) :
  AB + BC + CD + DE + AE = 5 + 2 * Real.sqrt 5 + Real.sqrt 3 := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_l866_86642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l866_86648

def sequence_a : ℕ → ℝ
  | 0 => 12  -- Add this case to handle n = 0
  | 1 => 12
  | (n + 1) => sequence_a n + 2 * n

theorem min_value_of_sequence_ratio :
  ∃ (n : ℕ), n > 0 ∧ sequence_a n / n = 6 ∧ ∀ (m : ℕ), m > 0 → sequence_a m / m ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_ratio_l866_86648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_apple_purchase_l866_86629

/-- The price of apples per kg -/
noncomputable def apple_price : ℚ := 70

/-- The amount of mangoes in kg -/
noncomputable def mango_amount : ℚ := 9

/-- The price of mangoes per kg -/
noncomputable def mango_price : ℚ := 45

/-- The total amount Tom paid -/
noncomputable def total_paid : ℚ := 965

/-- The amount of apples Tom purchased in kg -/
noncomputable def apple_amount : ℚ := (total_paid - mango_amount * mango_price) / apple_price

theorem tom_apple_purchase : apple_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_apple_purchase_l866_86629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_quadratic_many_roots_l866_86633

theorem floor_quadratic_many_roots :
  ∃ (p q : ℤ), p ≠ 0 ∧ (∃ (S : Finset ℝ), (∀ x ∈ S, ⌊x^2⌋ + p * x + q = 0) ∧ S.card > 100) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_quadratic_many_roots_l866_86633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l866_86628

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- The cube with edge length 1 -/
def unitCube : Set Point3D := sorry

/-- The point M on edge AB with AM = 1/3 -/
noncomputable def M : Point3D := ⟨1/3, 0, 0⟩

/-- The line A₁D₁ -/
noncomputable def A₁D₁ : Line3D := ⟨⟨0, 0, 1⟩, ⟨1, 0, 0⟩⟩

/-- The face ABCD of the cube -/
def faceABCD : Set Point3D := sorry

/-- Distance between a point and a line in 3D space -/
noncomputable def distPointToLine (p : Point3D) (l : Line3D) : ℝ := sorry

/-- Distance between two points in 3D space -/
noncomputable def distPointToPoint (p1 p2 : Point3D) : ℝ := sorry

/-- The locus of points P satisfying the given condition -/
noncomputable def locusP : Set Point3D :=
  {p ∈ faceABCD | (distPointToLine p A₁D₁)^2 - distPointToPoint p M = 1}

/-- Predicate to check if a set of points forms a parabola -/
def IsParabola (s : Set Point3D) : Prop := sorry

/-- Theorem stating that the locus of points P forms a parabola -/
theorem locus_is_parabola : IsParabola locusP := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l866_86628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_positive_necessary_not_sufficient_l866_86692

-- Define a geometric sequence
noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n-1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

-- Define an increasing sequence
def is_increasing (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S (n + 1) > S n

-- Theorem statement
theorem q_positive_necessary_not_sufficient :
  ∀ a₁ q : ℝ,
  (∀ n : ℕ, is_increasing (λ k => geometric_sum a₁ q k) → q > 0) ∧
  ¬(∀ a₁ : ℝ, q > 0 → ∀ n : ℕ, is_increasing (λ k => geometric_sum a₁ q k)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_positive_necessary_not_sufficient_l866_86692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l866_86641

/-- The speed of a boat in still water, given its downstream and upstream speeds -/
noncomputable def boatSpeedInStillWater (downstreamSpeed upstreamSpeed : ℝ) : ℝ :=
  (downstreamSpeed + upstreamSpeed) / 2

/-- Theorem: The speed of a boat in still water is 11 km/hr -/
theorem boat_speed_in_still_water :
  let downstreamSpeed : ℝ := 15
  let upstreamSpeed : ℝ := 7
  boatSpeedInStillWater downstreamSpeed upstreamSpeed = 11 := by
  unfold boatSpeedInStillWater
  norm_num

#eval (15 + 7) / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l866_86641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l866_86698

/-- Calculates the speed of a train given its length and time to cross a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: A train with length 200 meters that crosses a pole in 16 seconds has a speed of 45 km/h -/
theorem train_speed_calculation :
  train_speed 200 16 = 45 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the arithmetic expression
  simp
  -- Check that the result is equal to 45
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l866_86698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l866_86600

noncomputable def p : Prop := ∀ x : ℝ, (2 : ℝ)^x + (1 / (2 : ℝ)^x) > 2

noncomputable def q : Prop := ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi/2 ∧ Real.sin x + Real.cos x = 1/2

theorem problem_solution : ¬p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l866_86600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_specific_triangle_l866_86687

/-- The centroid of a triangle in the complex plane -/
noncomputable def centroid (z₁ z₂ z₃ : ℂ) : ℂ := (z₁ + z₂ + z₃) / 3

/-- Theorem: The centroid of a triangle with given vertices -/
theorem centroid_of_specific_triangle :
  let z₁ : ℂ := -11 + 3*I
  let z₂ : ℂ := 3 - 7*I
  let z₃ : ℂ := 5 + 9*I
  centroid z₁ z₂ z₃ = -1 + (5/3)*I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_of_specific_triangle_l866_86687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l866_86653

def U : Set Nat := {1,2,3,4,5,6}
def M : Set Nat := {1,2,4}

theorem complement_of_M_in_U :
  (U \ M) = {3,5,6} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l866_86653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_times_i_in_first_quadrant_l866_86625

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := 1 / (1 + i)

theorem z_times_i_in_first_quadrant : 
  let w := z * i
  0 < w.re ∧ 0 < w.im :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_times_i_in_first_quadrant_l866_86625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_pi_over_4_l866_86615

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sin x + Real.cos x) - 1/2

-- State the theorem
theorem tangent_slope_at_pi_over_4 :
  deriv f (π/4) = 1/2 := by
  -- The proof is omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_pi_over_4_l866_86615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_sale_savings_l866_86672

/-- Calculates the percentage savings when buying three jackets under a specific discount scheme. -/
theorem jacket_sale_savings (regular_price second_discount third_discount : ℝ) :
  regular_price = 80 →
  second_discount = 0.25 →
  third_discount = 0.60 →
  (let total_regular_price := 3 * regular_price
   let discounted_price := regular_price + (1 - second_discount) * regular_price + (1 - third_discount) * regular_price
   let savings := total_regular_price - discounted_price
   let savings_percentage := savings / total_regular_price * 100
   savings_percentage = 28) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_sale_savings_l866_86672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_a_range_l866_86675

-- Define the function f(x) = ax + ln x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a + 1 / x

-- Theorem 1: If f'(x₀) = (f(e) - f(1)) / (e - 1), then x₀ = e - 1
theorem x0_value (a : ℝ) :
  ∃ x₀ : ℝ, x₀ ≥ 1 ∧ f_derivative a x₀ = (f a (Real.exp 1) - f a 1) / (Real.exp 1 - 1) → x₀ = Real.exp 1 - 1 := by
  sorry

-- Theorem 2: If f(x) is monotonically decreasing on [1, +∞), then a ∈ (-∞, -1]
theorem a_range :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 1 → f_derivative a x ≤ 0) → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_value_a_range_l866_86675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_properties_l866_86603

/-- A rectangle with corners at (1, 1) and (9, 7) -/
structure Rectangle where
  corner1 : ℝ × ℝ := (1, 1)
  corner2 : ℝ × ℝ := (9, 7)

/-- The length of the diagonal of the rectangle -/
noncomputable def diagonalLength (r : Rectangle) : ℝ :=
  Real.sqrt ((r.corner2.1 - r.corner1.1)^2 + (r.corner2.2 - r.corner1.2)^2)

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ :=
  (r.corner2.1 - r.corner1.1) * (r.corner2.2 - r.corner1.2)

theorem rectangle_properties (r : Rectangle) :
  diagonalLength r = 10 ∧ area r = 48 := by
  sorry

#eval area { corner1 := (1, 1), corner2 := (9, 7) : Rectangle }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_properties_l866_86603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l866_86610

noncomputable def f (x : ℝ) : ℝ := (2 * x - 3) / (3 * x^2 - 1)

theorem f_values : 
  f (-2) = -7/11 ∧ f 0 = 3 ∧ f 1 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l866_86610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_integers_nonpositive_l866_86696

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (2 * x - 1) - a * x + a

-- Define the condition for f(x) ≤ 0 at integer x
def f_nonpositive_at_integer (a : ℝ) (x : ℤ) : Prop := f a (x : ℝ) ≤ 0

-- State the theorem
theorem exactly_two_integers_nonpositive (a : ℝ) :
  (a < 1) →
  (∃! x y : ℤ, x ≠ y ∧ f_nonpositive_at_integer a x ∧ f_nonpositive_at_integer a y) ↔
  (5 / (3 * Real.exp 2) < a ∧ a ≤ 3 / (2 * Real.exp 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_integers_nonpositive_l866_86696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l866_86673

/-- A type representing a line in the plane -/
structure Line where

/-- A type representing a point in the plane -/
structure Point where

/-- The set of all lines -/
def all_lines : Finset Line := sorry

/-- The number of lines -/
axiom num_lines : Finset.card all_lines = 120

/-- A function that returns true if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- A function that returns true if a line passes through a point -/
def passes_through (l : Line) (p : Point) : Prop := sorry

/-- The specific point B mentioned in the problem -/
def B : Point := sorry

/-- A function that returns the nth line -/
def M (n : ℕ) : Line := sorry

/-- Lines M₅ₙ are parallel to each other -/
axiom parallel_5n (n1 n2 : ℕ) : parallel (M (5 * n1)) (M (5 * n2))

/-- Lines M₅ₙ₋₄ pass through point B -/
axiom through_B (n : ℕ) : passes_through (M (5 * n - 4)) B

/-- Lines M₅ₙ₋₂ are parallel to each other but not to M₅ₙ -/
axiom parallel_5n_minus_2 (n1 n2 : ℕ) : 
  parallel (M (5 * n1 - 2)) (M (5 * n2 - 2)) ∧ 
  ¬parallel (M (5 * n1 - 2)) (M (5 * n1))

/-- A function that computes the number of intersection points -/
def num_intersections (lines : Finset Line) : ℕ := sorry

/-- The theorem to be proved -/
theorem max_intersections : 
  num_intersections all_lines = 5737 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l866_86673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_calculation_l866_86656

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Final amount calculation with simple interest -/
def final_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + simple_interest principal rate time

theorem initial_amount_calculation (amount : ℝ) (rate : ℝ) (time : ℝ) :
  amount = 900 ∧ rate = 0.05 ∧ time = 4 →
  ∃ (principal : ℝ), principal = 750 ∧ amount = final_amount principal rate time :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_calculation_l866_86656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_feeding_sequences_l866_86691

def num_pairs : ℕ := 5

def alternating_product : ℕ → ℕ → ℕ
  | 0, _ => 1
  | n + 1, m => m * (alternating_product n (m - 1))

theorem zoo_feeding_sequences :
  (alternating_product (2 * num_pairs - 1) num_pairs) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_feeding_sequences_l866_86691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l866_86669

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (7 * a n + Real.sqrt (45 * (a n)^2 - 36)) / 2

theorem sequence_properties :
  (∀ n : ℕ, ∃ k : ℤ, a n = k ∧ a n > 0) ∧
  (∀ n : ℕ, ∃ k : ℤ, (a n * a (n + 1) - 1 : ℝ) = k^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l866_86669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l866_86630

theorem count_integer_pairs : 
  let count := Finset.filter (fun p : ℕ × ℕ => 
    p.1 < p.2 ∧ 
    p.1 > 0 ∧ 
    p.2 > 0 ∧ 
    (3 : ℚ) / 2008 = 1 / (p.1 : ℚ) + 1 / (p.2 : ℚ)) (Finset.range 2009 ×ˢ Finset.range 2009)
  Finset.card count = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l866_86630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clover_percentage_l866_86624

/-- Given the following conditions:
  * June picks 200 clovers in total
  * 75% of clovers have 3 petals
  * 24% of clovers have 2 petals
  * Some percentage of clovers have 4 petals
  * June earns 554 cents
Prove that the percentage of clovers with four petals is 1% -/
theorem clover_percentage (total_clovers : ℕ) (three_petal_percent : ℚ) 
  (two_petal_percent : ℚ) (earnings : ℕ) 
  (h1 : total_clovers = 200)
  (h2 : three_petal_percent = 75 / 100)
  (h3 : two_petal_percent = 24 / 100)
  (h4 : earnings = 554)
  : (1 : ℚ) / 100 = 
    1 - (three_petal_percent + two_petal_percent) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clover_percentage_l866_86624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_range_of_a_l866_86652

-- Define the functions f and g
def f (x : ℝ) := x^3 - x
def g (x a : ℝ) := x^2 - a^2 + a

-- Define the condition for a common tangent line
def has_common_tangent (a : ℝ) : Prop :=
  ∃ (s t : ℝ), (3 * s^2 - 1 = 2 * t) ∧ (-2 * s^3 = -t^2 - a^2 + a)

-- State the theorem
theorem tangent_line_range_of_a :
  ∀ a : ℝ, has_common_tangent a →
    a ∈ Set.Icc ((1 - Real.sqrt 5) / 2) ((1 + Real.sqrt 5) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_range_of_a_l866_86652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_optimal_expense_unique_l866_86695

/-- Profit function for a manufacturer's promotional event -/
noncomputable def profit (m : ℝ) : ℝ := -(16 / (m + 1) + (m + 1)) + 29

/-- The promotional expenses that maximize profit -/
def optimal_expense : ℝ := 3

/-- The maximum profit achieved -/
def max_profit : ℝ := 21

/-- Theorem stating that the profit function is maximized at the optimal expense -/
theorem profit_maximized (m : ℝ) (h : m ≥ 0) : 
  profit m ≤ max_profit ∧ 
  profit optimal_expense = max_profit := by
  sorry

/-- Theorem stating that the optimal expense is unique -/
theorem optimal_expense_unique (m : ℝ) (h : m ≥ 0) :
  profit m = max_profit → m = optimal_expense := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_optimal_expense_unique_l866_86695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_cosine_product_l866_86611

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- Define the properties of the triangle
def IsObtuse (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

def SineOrdering (t : Triangle) : Prop :=
  Real.sin t.A < Real.sin t.B ∧ Real.sin t.B < Real.sin t.C

-- State the theorem
theorem obtuse_triangle_cosine_product (t : Triangle) 
  (h1 : IsObtuse t) (h2 : SineOrdering t) : 
  Real.cos t.A * Real.cos t.B > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_cosine_product_l866_86611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l866_86644

theorem trigonometric_equation_solution (z : ℝ) 
  (h1 : Real.cos z ≠ 0) (h2 : Real.cos (2 * z) ≠ -1) :
  (4 * (Real.sin z)^4 / (1 + Real.cos (2 * z))^2 - 2 / (Real.cos z)^2 - 1 = 0) ↔ 
  (∃ k : ℤ, z = π / 3 * (3 * k + 1) ∨ z = π / 3 * (3 * k - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l866_86644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l866_86682

noncomputable def f (x : ℝ) := Real.sin x - Real.sqrt 3 * Real.cos x

theorem f_monotone_increasing :
  ∀ x y, -π / 6 ≤ x ∧ x < y ∧ y ≤ 0 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l866_86682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l866_86688

theorem equation_solution (e : ℝ) (h : e > 0) (he : Real.exp 1 = e) :
  {x : ℝ | Real.exp x + 2 * Real.exp (-x) = 3} = {0, Real.log 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l866_86688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_percent_stock_yield_l866_86621

/-- Calculates the yield percentage of a stock given its dividend rate, face value, and market value. -/
noncomputable def stock_yield_percentage (dividend_rate : ℝ) (face_value : ℝ) (market_value : ℝ) : ℝ :=
  (dividend_rate * face_value / market_value) * 100

/-- Theorem: A 9% stock with a market value of 112.5 and an assumed face value of 100 yields 8%. -/
theorem nine_percent_stock_yield :
  let dividend_rate : ℝ := 0.09
  let face_value : ℝ := 100
  let market_value : ℝ := 112.5
  stock_yield_percentage dividend_rate face_value market_value = 8 :=
by
  -- Unfold the definition of stock_yield_percentage
  unfold stock_yield_percentage
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_percent_stock_yield_l866_86621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sets_have_perp_property_l866_86616

-- Define property ⊥
def has_perp_property (M : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ M, ∃ p₂ ∈ M, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the four sets
def M₁ : Set (ℝ × ℝ) := {p | p.2 = p.1^3 - 2*p.1^2 + 3}
def M₂ : Set (ℝ × ℝ) := {p | p.2 = Real.log (2 - p.1)}
def M₃ : Set (ℝ × ℝ) := {p | p.2 = 2 - 2^p.1}
def M₄ : Set (ℝ × ℝ) := {p | p.2 = 1 - Real.sin p.1}

-- Theorem statement
theorem all_sets_have_perp_property :
  has_perp_property M₁ ∧ has_perp_property M₂ ∧ has_perp_property M₃ ∧ has_perp_property M₄ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sets_have_perp_property_l866_86616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_is_46_l866_86612

/-- The speed of the faster train given the conditions of the problem -/
noncomputable def faster_train_speed (slower_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) : ℝ :=
  let relative_speed := 2 * train_length / passing_time
  slower_speed + relative_speed * 3600 / 1000

/-- Theorem stating that the speed of the faster train is 46 km/hr -/
theorem faster_train_speed_is_46 :
  faster_train_speed 36 18 25 = 46 := by
  -- Unfold the definition of faster_train_speed
  unfold faster_train_speed
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num
  -- QED

-- We can't use #eval for noncomputable functions, so we'll use #check instead
#check faster_train_speed 36 18 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_is_46_l866_86612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l866_86650

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- The slope of an asymptote of a hyperbola -/
noncomputable def asymptote_slope (a b : ℝ) : ℝ := b / a

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : asymptote_slope a b = 1/2) :
  eccentricity a b = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l866_86650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_returns_after_seven_steps_l866_86605

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the movement pattern
inductive MovementStep
  | ParallelToBC
  | ParallelToAB
  | ParallelToAC

-- Define the next position function
noncomputable def nextPosition (t : Triangle) (p : ℝ × ℝ) (step : MovementStep) : ℝ × ℝ :=
  sorry

-- Define the sequence of movements
def movementSequence : List MovementStep :=
  [MovementStep.ParallelToBC, MovementStep.ParallelToAB, MovementStep.ParallelToAC,
   MovementStep.ParallelToBC, MovementStep.ParallelToAB, MovementStep.ParallelToAC,
   MovementStep.ParallelToBC]

-- Define a function to check if a point is inside a triangle
noncomputable def isInside (t : Triangle) (p : ℝ × ℝ) : Prop :=
  sorry

-- Theorem statement
theorem point_returns_after_seven_steps (t : Triangle) (m : ℝ × ℝ) 
  (h : isInside t m) :
  let finalPosition := (movementSequence.foldl (nextPosition t) m)
  finalPosition = m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_returns_after_seven_steps_l866_86605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_club_enrollment_l866_86638

theorem school_club_enrollment (total chess debate science two_or_more : ℕ) 
  (h_total : total = 24)
  (h_chess : chess = 14)
  (h_debate : debate = 15)
  (h_science : science = 10)
  (h_two_or_more : two_or_more = 12)
  (h_participation : ∀ student, student ∈ Finset.range total → 
    (student ∈ Finset.range chess ∨ 
     student ∈ Finset.range debate ∨ 
     student ∈ Finset.range science)) :
  (Finset.range chess ∩ Finset.range debate ∩ Finset.range science).card = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_club_enrollment_l866_86638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_area_l866_86623

/-- The area of a cyclic quadrilateral with given diagonal and offsets -/
theorem cyclic_quadrilateral_area (d m₁ m₂ θ₁ θ₂ : ℝ) (h_d : d = 42) (h_m₁ : m₁ = 15) (h_m₂ : m₂ = 20) 
  (h_θ₁ : θ₁ = π / 6) (h_θ₂ : θ₂ = π / 4) : 
  ∃ (area : ℝ), (area ≥ 454.43 ∧ area ≤ 454.45) ∧ 
  area = (d / 2) * (m₁ * Real.sin θ₁ + m₂ * Real.sin θ₂) := by
  sorry

#check cyclic_quadrilateral_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_area_l866_86623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_application_equals_one_l866_86640

-- Define the function g
noncomputable def g : Set.Icc (-4 : ℝ) 4 → ℝ := sorry

-- State the properties of g
axiom g_lower_bound : ∀ x : Set.Icc (-4 : ℝ) 4, g x ≥ 1
axiom g_neg_four : g ⟨-4, by norm_num⟩ = 1
axiom g_zero : g ⟨0, by norm_num⟩ = 1
axiom g_four : g ⟨4, by norm_num⟩ = 3

-- State the theorem
theorem no_double_application_equals_one :
  ∀ x : Set.Icc (-4 : ℝ) 4, ∀ y : Set.Icc (-4 : ℝ) 4, g y = g x → g y ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_application_equals_one_l866_86640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_jogging_yoga_difference_l866_86607

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
  | Prime
  | Composite
  | ReRoll

/-- Probability of rolling a prime number (2, 3, 5, or 7) on an 8-sided die, excluding re-rolls -/
noncomputable def probPrime : ℝ := 4 / 7

/-- Probability of rolling a composite number (4, 6, or 8) on an 8-sided die, excluding re-rolls -/
noncomputable def probComposite : ℝ := 3 / 7

/-- Number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- Expected difference between jogging days and yoga days in a non-leap year -/
noncomputable def expectedDifference : ℝ := daysInYear * (probPrime - probComposite)

theorem alice_jogging_yoga_difference :
  ∃ ε > 0, |expectedDifference - 52.71| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_jogging_yoga_difference_l866_86607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_range_l866_86645

theorem sin_equation_range (m : ℝ) : 
  (∃ x : ℝ, Real.sin x ^ 2 + Real.sin (2 * x) = m + 2 * Real.cos x ^ 2) →
  (-1 - Real.sqrt 13) / 2 ≤ m ∧ m ≤ (-1 + Real.sqrt 13) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_range_l866_86645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_condition_l866_86655

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if sin²(A/2) = (c-b)/(2c), then the triangle is right-angled. -/
theorem right_angled_triangle_condition (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  (Real.sin (A / 2))^2 = (c - b) / (2 * c) →
  a^2 + b^2 = c^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_condition_l866_86655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l866_86664

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 11/3) (h2 : x * y = 169) : 
  (x + y) / 2 = 7 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l866_86664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallelism_l866_86608

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_parallelism 
  (a b c d : V) 
  (k : ℝ) 
  (h_not_collinear : ¬ ∃ (r : ℝ), b = r • a) 
  (h_c : c = k • a + b) 
  (h_d : d = a - b) 
  (h_parallel : ∃ (lambda : ℝ), c = lambda • d) : 
  k = -1 ∧ c = -d := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallelism_l866_86608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_value_l866_86643

/-- An arithmetic progression with 12 terms representing shadow lengths of solar terms -/
def ShadowSequence : ℕ → ℝ := sorry

/-- The 4th term of the sequence is 10.5 -/
axiom fourth_term : ShadowSequence 4 = 10.5

/-- The 10th term of the sequence is 4.5 -/
axiom tenth_term : ShadowSequence 10 = 4.5

/-- The sequence is an arithmetic progression -/
axiom is_arithmetic_progression : ∀ n : ℕ, n > 0 → n < 12 → 
  ShadowSequence (n + 1) - ShadowSequence n = ShadowSequence (n + 2) - ShadowSequence (n + 1)

/-- Theorem: The 7th term of the sequence is 7.5 -/
theorem seventh_term_value : ShadowSequence 7 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_value_l866_86643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_l866_86689

-- Define the polynomials
noncomputable def dividend : Polynomial ℚ := Polynomial.X^1004
noncomputable def divisor : Polynomial ℚ := (Polynomial.X^2 - 1) * (Polynomial.X + 2)

-- Define the remainder
noncomputable def remainder : Polynomial ℚ := Polynomial.X^2

-- Theorem statement
theorem division_remainder :
  dividend = divisor * (dividend / divisor) + remainder := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_l866_86689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_three_element_set_l866_86637

theorem subsets_of_three_element_set :
  let S : Finset ℕ := {1, 3, 4}
  Finset.card (Finset.powerset S) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_three_element_set_l866_86637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_distance_l866_86678

theorem island_distance (n : ℝ) :
  let triangle_ABC : Set (ℝ × ℝ) := sorry
  let A : ℝ × ℝ := sorry
  let B : ℝ × ℝ := sorry
  let C : ℝ × ℝ := sorry
  let AB : ℝ := 10 * n
  let angle_BAC : ℝ := 60 * (π / 180)
  let angle_ABC : ℝ := 75 * (π / 180)
  let BC : ℝ := ((B.1 - C.1)^2 + (B.2 - C.2)^2).sqrt
  triangle_ABC = {A, B, C} →
  BC = 5 * Real.sqrt 6 * n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_distance_l866_86678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_loan_monthly_payment_l866_86694

/-- Calculates the monthly payment for a car loan -/
noncomputable def calculate_monthly_payment (car_cost : ℝ) (down_payment : ℝ) (loan_duration : ℕ) (interest_rate : ℝ) : ℝ :=
  let amount_to_borrow := car_cost - down_payment
  let monthly_payment_without_interest := amount_to_borrow / (loan_duration : ℝ)
  let interest_per_month := monthly_payment_without_interest * interest_rate
  monthly_payment_without_interest + interest_per_month

/-- Theorem stating that the monthly payment for the given car loan scenario is approximately $502.08 -/
theorem car_loan_monthly_payment :
  let car_cost : ℝ := 32000
  let down_payment : ℝ := 8000
  let loan_duration : ℕ := 48
  let interest_rate : ℝ := 0.05 / 12
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
    |calculate_monthly_payment car_cost down_payment loan_duration interest_rate - 502.08| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_loan_monthly_payment_l866_86694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_less_than_zero_l866_86601

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x - 1 else -x - 1

-- State the theorem
theorem solution_set_f_less_than_zero :
  (∀ x : ℝ, f (-x) = f x) →  -- f is even
  (∀ x : ℝ, x ≥ 0 → f x = x - 1) →  -- f(x) = x - 1 for x ≥ 0
  {x : ℝ | f x < 0} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_less_than_zero_l866_86601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l866_86646

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 8 * x - 3 * y^2 + 6 * y - 17 = 0

/-- The distance between the vertices of the hyperbola -/
noncomputable def vertex_distance : ℝ := 3 * Real.sqrt 2

/-- Theorem stating that the distance between the vertices of the hyperbola
    defined by the given equation is 3√2 -/
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_equation x y → 
  ∃ a b c d : ℝ, 
    ((x - a) / b)^2 - ((y - c) / d)^2 = 1 ∧
    vertex_distance = 2 * b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l866_86646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_square_not_divides_f_and_q_divides_f_implies_qr_divides_f_l866_86657

/-- The function f(x) = (x+b)^2 - c -/
def f (b c x : ℤ) : ℤ := (x + b)^2 - c

theorem prime_square_not_divides_f_and_q_divides_f_implies_qr_divides_f 
  (b c : ℤ) (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p : ℤ) ∣ c ∧ ¬((p^2 : ℤ) ∣ c) →
  (q ≠ 2 ∧ (q : ℤ) ∣ c) →
  (∀ n : ℤ, ¬((p^2 : ℤ) ∣ f b c n)) ∧
  ((∃ n : ℤ, (q : ℤ) ∣ f b c n) → 
    (∀ r : ℕ, ∃ n' : ℤ, ((q * r : ℕ) : ℤ) ∣ f b c n')) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_square_not_divides_f_and_q_divides_f_implies_qr_divides_f_l866_86657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l866_86665

noncomputable def a : ℝ := 2^(3/10)
noncomputable def b : ℝ := (3/10)^2
noncomputable def c (x : ℝ) : ℝ := Real.log (x^2 + 3/10) / Real.log x

theorem abc_relationship (x : ℝ) (h : x > 1) : b < a ∧ a < c x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l866_86665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_other_pairs_not_equal_l866_86609

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1|

noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -1 then x + 1 else -x - 1

-- Theorem stating that f and g are equal for all real x
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry

-- Additional theorem to show that other pairs are not equal
theorem other_pairs_not_equal :
  (∃ x : ℝ, x ≠ 0 ∧ x ≠ x^2 / x) ∧
  (∃ x : ℝ, x < 0 ∧ |x| ≠ (Real.sqrt x)^2) ∧
  (∃ x : ℝ, (x + 1)^2 ≠ x^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_other_pairs_not_equal_l866_86609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l866_86661

/-- Two vectors in ℝ² parameterized by x -/
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

/-- The difference between vectors a and b -/
def diff (x : ℝ) : ℝ × ℝ := ((a x).1 - (b x).1, (a x).2 - (b x).2)

/-- The magnitude of the difference between vectors a and b -/
noncomputable def mag_diff (x : ℝ) : ℝ := Real.sqrt ((diff x).1^2 + (diff x).2^2)

/-- The dot product of vectors a and b -/
def dot_product (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem vector_problem (x : ℝ) :
  (∃ y, a y = b y → mag_diff y = 2 ∨ mag_diff y = 2 * Real.sqrt 5) ∧
  (dot_product x > 0 → x ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 3) := by
  sorry

#check vector_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l866_86661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l866_86626

/-- Proves that for an angle α in the second quadrant, if sin(π + α) = -3/5, then tan(2α) = -24/7 -/
theorem tan_double_angle_special_case (α : ℝ) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  Real.sin (π + α) = -3/5 → 
  Real.tan (2 * α) = -24/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l866_86626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_power_divisible_l866_86649

theorem floor_power_divisible (k n : ℕ+) : 
  let α : ℝ := k + 1/2 + Real.sqrt (k^2 + 1/4)
  ∃ m : ℤ, ⌊(α : ℝ)^(n : ℕ)⌋ = k * m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_power_divisible_l866_86649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_properties_l866_86683

noncomputable def α : ℝ := Real.arcsin (-Real.sqrt 5 / 5)

theorem trigonometric_properties :
  let m : ℝ := -1
  let point_A : ℝ × ℝ := (-2, m)
  Real.sin α = -Real.sqrt 5 / 5 →
  (point_A.1 = -2 ∧ point_A.2 = m) →
  (m = -1) ∧
  (Real.cos α = -2 * Real.sqrt 5 / 5) ∧
  ((Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_properties_l866_86683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_with_inscribed_circle_l866_86619

/-- A square with an inscribed circle passing through the midpoint of its diagonal -/
structure SquareWithInscribedCircle where
  /-- The side length of the square -/
  side : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The circle is inscribed in the square -/
  inscribed : side = Real.sqrt 2 * radius
  /-- The circle passes through the midpoint of the diagonal -/
  passes_midpoint : side * Real.sqrt 2 / 2 = radius

/-- The area of a square with an inscribed circle passing through the midpoint of its diagonal
    is twice the square of the circle's radius -/
theorem square_area_with_inscribed_circle (s : SquareWithInscribedCircle) :
  s.side ^ 2 = 2 * s.radius ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_with_inscribed_circle_l866_86619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_line_to_intersection_l866_86658

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle on a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an arc of a circle -/
structure Arc where
  circle : Circle
  start_angle : ℝ
  end_angle : ℝ

/-- Represents a sheet of paper with finite dimensions -/
structure Sheet where
  width : ℝ
  height : ℝ

/-- Checks if a point is on the sheet -/
def is_on_sheet (p : Point) (s : Sheet) : Prop :=
  0 ≤ p.x ∧ p.x ≤ s.width ∧ 0 ≤ p.y ∧ p.y ≤ s.height

/-- Checks if a point is on a circle -/
def is_on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Theorem: Given two arcs and a point on a sheet, a line can be constructed to an intersection point -/
theorem construct_line_to_intersection
  (s : Sheet)
  (arc1 arc2 : Arc)
  (P : Point)
  (h1 : is_on_sheet P s)
  (h2 : ∃ Q, is_on_circle Q arc1.circle ∧ is_on_circle Q arc2.circle ∧ ¬is_on_sheet Q s) :
  ∃ (line : Point → Point → Prop),
    ∃ Q, is_on_circle Q arc1.circle ∧ is_on_circle Q arc2.circle ∧ line P Q ∧
    ∀ R, is_on_sheet R s → line P R → is_on_sheet R s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_line_to_intersection_l866_86658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factor_in_C_l866_86651

def C : Set Nat := {65, 67, 68, 71, 73}

def smallest_prime_factor (n : Nat) : Nat :=
  (Nat.factors n).head!

theorem smallest_prime_factor_in_C :
  ∀ x ∈ C, smallest_prime_factor 68 ≤ smallest_prime_factor x :=
by
  intro x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factor_in_C_l866_86651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l866_86668

-- Define set M
def M : Set ℝ := {x | Real.exp (x * Real.log 2) < 1}

-- Define set N
def N : Set ℝ := {x | x^2 - x - 2 < 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo (-1 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l866_86668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_3_neg3_l866_86617

noncomputable def rect_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- undefined for (0, 0)
  (r, if θ < 0 then θ + 2*Real.pi else θ)

theorem rect_to_polar_3_neg3 :
  let (r, θ) := rect_to_polar 3 (-3)
  r = 3 * Real.sqrt 2 ∧ θ = 7 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_3_neg3_l866_86617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_sum_l866_86662

-- Define the integer part function
noncomputable def integerPart (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem integer_part_sum (x : ℝ) (h : x = 9.42) :
  integerPart x + integerPart (2 * x) + integerPart (3 * x) = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_sum_l866_86662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l866_86604

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis 5 and semi-minor axis √5 -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 / 25 + p.y^2 / 5 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the dot product of two vectors -/
def dotProduct (p1 p2 p3 : Point) : ℝ :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y)

/-- Theorem: Area of triangle F₁PF₂ is 5 -/
theorem ellipse_triangle_area
  (p f1 f2 : Point)
  (h1 : isOnEllipse p)
  (h2 : dotProduct p f1 f2 = 0)
  (h3 : distance f1 f2 = 4 * Real.sqrt 5) :
  (1/2) * distance p f1 * distance p f2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l866_86604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jed_older_than_matt_l866_86660

/-- Represents a person's age -/
structure Age where
  value : ℕ

/-- The problem setup -/
structure AgeProblem where
  jed : Age
  matt : Age
  h1 : jed.value + 10 = 25
  h2 : jed.value + matt.value = 20

/-- The theorem to prove -/
theorem jed_older_than_matt (problem : AgeProblem) :
  problem.jed.value - problem.matt.value = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jed_older_than_matt_l866_86660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l866_86684

theorem log_problem (u v w : ℝ) (hu : u ≠ 1) (hv : v ≠ 1) (hw : w ≠ 1)
  (hu_pos : u > 0) (hv_pos : v > 0) (hw_pos : w > 0)
  (h1 : Real.log (v * w) / Real.log u + Real.log w / Real.log v = 5)
  (h2 : Real.log u / Real.log v + Real.log v / Real.log w = 3) :
  Real.log u / Real.log w = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l866_86684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_visit_two_countries_l866_86634

/-- Probability of visiting exactly two countries given individual probabilities --/
theorem prob_visit_two_countries
  (p_chile p_madagascar p_japan p_egypt : ℝ)
  (h1 : 0 ≤ p_chile ∧ p_chile ≤ 1)
  (h2 : 0 ≤ p_madagascar ∧ p_madagascar ≤ 1)
  (h3 : 0 ≤ p_japan ∧ p_japan ≤ 1)
  (h4 : 0 ≤ p_egypt ∧ p_egypt ≤ 1)
  (h5 : p_chile = 0.40)
  (h6 : p_madagascar = 0.35)
  (h7 : p_japan = 0.20)
  (h8 : p_egypt = 0.15) :
  (p_chile * p_madagascar * (1 - p_japan) * (1 - p_egypt)) +
  (p_chile * p_japan * (1 - p_madagascar) * (1 - p_egypt)) +
  (p_chile * p_egypt * (1 - p_madagascar) * (1 - p_japan)) +
  (p_madagascar * p_japan * (1 - p_chile) * (1 - p_egypt)) +
  (p_madagascar * p_egypt * (1 - p_chile) * (1 - p_japan)) +
  (p_japan * p_egypt * (1 - p_chile) * (1 - p_madagascar)) = 0.2432 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_visit_two_countries_l866_86634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_n_is_zero_l866_86671

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

noncomputable def n : ℝ := 
  75 ^ (sum_of_factorials 80) + 
  25 ^ (sum_of_factorials 75) - 
  Real.log (97 ^ (sum_of_factorials 50)) + 
  Real.sin (123 ^ (sum_of_factorials 25))

theorem unit_digit_of_n_is_zero : n % 10 = 0 := by
  sorry

#eval factorial 5
#eval sum_of_factorials 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_n_is_zero_l866_86671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_one_l866_86654

theorem proposition_one (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a - b = 1) : 
  a^3 - b^3 - a*b - a^2 - b^2 = 0 := by
  -- Expand the expression
  have h3 : a^3 - b^3 - a*b - a^2 - b^2 = (a - b - 1) * (a^2 + a*b + b^2) := by ring
  -- Substitute a - b = 1
  rw [h2] at h3
  -- Simplify
  simp at h3
  -- The result follows
  exact h3

#check proposition_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_one_l866_86654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trishas_hourly_wage_l866_86693

/-- Calculates the hourly wage given the annual take-home pay, hours worked per week, 
    weeks worked per year, and the percentage of pay withheld for taxes and other deductions. -/
noncomputable def calculate_hourly_wage (annual_take_home : ℝ) (hours_per_week : ℝ) (weeks_per_year : ℝ) (withholding_percentage : ℝ) : ℝ :=
  let gross_annual_pay := annual_take_home / (1 - withholding_percentage)
  let total_hours := hours_per_week * weeks_per_year
  gross_annual_pay / total_hours

/-- Theorem stating that given the specified conditions, Trisha's hourly wage is $15. -/
theorem trishas_hourly_wage :
  let annual_take_home : ℝ := 24960
  let hours_per_week : ℝ := 40
  let weeks_per_year : ℝ := 52
  let withholding_percentage : ℝ := 0.2
  calculate_hourly_wage annual_take_home hours_per_week weeks_per_year withholding_percentage = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trishas_hourly_wage_l866_86693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l866_86639

/-- Proves that a train with given speed crossing a bridge of known length in a specific time has a certain length. -/
theorem train_length_proof (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) :
  bridge_length = 180 →
  crossing_time = 20 →
  train_speed_kmh = 77.4 →
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600;
  let total_distance : ℝ := train_speed_ms * crossing_time;
  let train_length : ℝ := total_distance - bridge_length;
  train_length = 250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l866_86639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_symmetric_point_l866_86677

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the original point P
def P : Point3D := ⟨1, 1, 1⟩

-- Define the symmetric point P' with respect to XOZ plane
def P' : Point3D := ⟨P.x, -P.y, P.z⟩

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

-- Theorem statement
theorem distance_to_symmetric_point :
  distance P P' = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_symmetric_point_l866_86677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identities_l866_86670

theorem sin_cos_identities (x : ℝ) 
  (h1 : -π/2 < x) (h2 : x < 0) (h3 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x * Real.cos x = -12/25) ∧ (Real.sin x - Real.cos x = -7/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identities_l866_86670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertices_form_parabola_l866_86666

/-- x-coordinate of the vertex of the parabola y = ax^2 + (2at)x + c -/
def vertex_x (a t : ℝ) : ℝ := -t

/-- y-coordinate of the vertex of the parabola y = ax^2 + (2at)x + c -/
def vertex_y (a c t : ℝ) : ℝ := -a * t^2 + c

/-- The set of vertices of a family of parabolas forms another parabola -/
theorem vertices_form_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  ∀ t : ℝ, ∃ x y : ℝ,
    (x, y) = (vertex_x a t, vertex_y a c t) ∧
    y = -a * x^2 + c :=
by
  intro t
  use -t, -a * t^2 + c
  constructor
  · rfl
  · ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertices_form_parabola_l866_86666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_iff_a_geq_half_l866_86635

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a - Real.log x

/-- The inequality condition from the problem -/
def inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  f a x > 1/x - Real.exp (1-x)

/-- The main theorem to be proved -/
theorem function_inequality_iff_a_geq_half :
  ∀ a : ℝ, (∀ x : ℝ, x > 1 → inequality_condition a x) ↔ a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_iff_a_geq_half_l866_86635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l866_86620

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

theorem triangle_ABC_properties
  (a b c : ℝ) (A B C : ℝ)
  (h_triangle : triangle_ABC a b c A B C)
  (h_equation : 3 * a * Real.cos A = Real.sqrt 6 * (b * Real.cos C + c * Real.cos B))
  (h_sin_B : Real.sin (Real.pi / 2 + B) = 1 / 3)
  (h_c : c = 2 * Real.sqrt 2) :
  Real.tan (2 * A) = 2 * Real.sqrt 2 ∧
  (1 / 2) * a * b * Real.sin C = (8 / 5) * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l866_86620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocal_distances_l866_86686

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t / 2, 1 + (Real.sqrt 3 * t) / 2)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, A = line_l t₁ ∧ B = line_l t₂ ∧
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧ t₁ ≠ t₂

-- Define point P on y-axis
def point_P : ℝ × ℝ := (0, 1)

-- State the theorem
theorem intersection_sum_reciprocal_distances
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  1 / dist point_P A + 1 / dist point_P B = Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocal_distances_l866_86686
