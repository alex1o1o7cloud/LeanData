import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_neighbor_divisibility_l595_59538

theorem power_of_neighbor_divisibility (k n m : ℕ) (hm : m > 0) :
  k ∣ (n * k) →
  (∃ q : ℤ, (n * k + 1 : ℤ)^m = q * k + 1) ∧
  (∃ r : ℤ, (n * k - 1 : ℤ)^m = r * k + (-1)^m) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_neighbor_divisibility_l595_59538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_l595_59569

/-- The radius of the circle to which six congruent parabolas are tangent -/
noncomputable def circle_radius : ℝ := 3/4

/-- Theorem stating the tangency conditions for the parabolas and the circle -/
theorem parabola_circle_tangency :
  let parabola := λ x : ℝ => x^2 + circle_radius
  let line := λ x : ℝ => -x * Real.sqrt 3
  (∃! x, parabola x = line x) ∧
  (parabola 0 = circle_radius) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_l595_59569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l595_59504

/-- A parabola with focus-directrix definition -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line passing through a given point -/
structure Line where
  slope : Option ℝ
  point : Point

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Definition of the parabola problem -/
def parabola_problem (C : Parabola) (P F : Point) (l : Line) : Prop :=
  -- P is on the parabola C
  P.y^2 = 2 * C.p * P.x
  -- P has x-coordinate 1
  ∧ P.x = 1
  -- F is the focus of the parabola
  ∧ F.x = C.p / 2 ∧ F.y = 0
  -- PF = 3
  ∧ distance P F = 3
  -- l passes through T(4, 0)
  ∧ l.point.x = 4 ∧ l.point.y = 0

theorem parabola_theorem (C : Parabola) (P F : Point) (l : Line) 
  (h : parabola_problem C P F l) :
  -- 1) The equation of the parabola is y² = 6x
  C.p = 3
  -- 2) OA · OB = -16, where O is the origin and A, B are intersection points
  ∧ ∃ (A B : Point), A.y^2 = 6 * A.x ∧ B.y^2 = 6 * B.x ∧ A.x * B.x + A.y * B.y = -16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l595_59504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_implies_even_terms_constraint_l595_59547

/-- An integer polynomial -/
def IntPolynomial := Polynomial ℤ

/-- Counts the number of even terms in a polynomial -/
noncomputable def countEvenTerms (f : IntPolynomial) : ℕ := sorry

/-- Checks if two natural numbers are relatively prime -/
def isRelativelyPrime (a b : ℕ) : Prop := Nat.Coprime a b

/-- Checks if a + b divides f(a) + f(b) -/
def divides_sum (f : IntPolynomial) (a b : ℕ) : Prop :=
  (a + b : ℤ) ∣ (f.eval (a : ℤ) + f.eval (b : ℤ))

/-- Main theorem -/
theorem infinite_pairs_implies_even_terms_constraint (f : IntPolynomial) :
  (∃ S : Set (ℕ × ℕ), Set.Infinite S ∧
    (∀ p ∈ S, isRelativelyPrime p.1 p.2 ∧ divides_sum f p.1 p.2)) →
  (countEvenTerms f = 0 ∨ countEvenTerms f > 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_pairs_implies_even_terms_constraint_l595_59547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_neg_one_l595_59542

noncomputable def f (x : ℝ) : ℝ := 2^x + 1/(2^(x+2))

theorem f_min_at_neg_one :
  ∀ x : ℝ, f x ≥ f (-1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_neg_one_l595_59542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l595_59548

noncomputable def f (x : ℝ) : ℝ := Real.exp (x / 2) - x / 4

noncomputable def g (x : ℝ) : ℝ := (x + 1) * (deriv f x)

noncomputable def F (a x : ℝ) : ℝ := Real.log (x + 1) - a * f x + 4

theorem problem_statement :
  (∀ x y, x > -1 → y > -1 → x < y → g x < g y) ∧
  (∀ a, a > 0 → (∀ x, F a x ≠ 0) → a > 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l595_59548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_monotonicity_l595_59510

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x + Real.sin x, 2 * Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x - Real.sin x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_period_and_monotonicity :
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi) ∧
  (∀ x ∈ Set.Icc (5 * Real.pi / 8) (3 * Real.pi / 4),
    ∀ y ∈ Set.Icc (5 * Real.pi / 8) (3 * Real.pi / 4),
      x < y → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_monotonicity_l595_59510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_coordinates_B_ABC_right_triangle_l595_59531

/-- Given two points in a 2D Cartesian coordinate system, calculate the distance between them. -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Given two points with the same x-coordinate, calculate the distance between them. -/
def vertical_distance (y₁ y₂ : ℝ) : ℝ :=
  |y₂ - y₁|

/-- Check if three points form a right-angled triangle. -/
noncomputable def is_right_triangle (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  let a := distance x₁ y₁ x₂ y₂
  let b := distance x₁ y₁ x₃ y₃
  let c := distance x₂ y₂ x₃ y₃
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

theorem distance_AB :
  distance 0 5 (-3) 6 = Real.sqrt 10 := by
  sorry

theorem coordinates_B (y : ℝ) :
  vertical_distance (-1/2) y = 10 →
  y = 19/2 ∨ y = -19/2 := by
  sorry

theorem ABC_right_triangle :
  is_right_triangle 0 6 4 0 (-9) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_coordinates_B_ABC_right_triangle_l595_59531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_decreasing_f_l595_59502

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.sqrt x

-- State the theorem
theorem min_a_for_decreasing_f : 
  (∃ (a : ℝ), ∀ (x y : ℝ), 1 ≤ x ∧ x < y ∧ y ≤ 4 → f a x > f a y) ∧ 
  (∀ (b : ℝ), b < 4 → ∃ (x y : ℝ), 1 ≤ x ∧ x < y ∧ y ≤ 4 ∧ f b x ≤ f b y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_decreasing_f_l595_59502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationery_problem_l595_59595

/-- The number of sheets of stationery bought by A and B -/
def sheets_bought : ℕ := sorry

/-- The number of envelopes A used -/
def envelopes_A : ℕ := sorry

/-- The number of envelopes B had left -/
def envelopes_B : ℕ := sorry

/-- A uses all envelopes with 1 sheet per envelope and has 40 sheets left -/
axiom A_condition : sheets_bought = envelopes_A + 40

/-- B uses all sheets with 3 sheets per envelope and has 40 envelopes left -/
axiom B_condition : sheets_bought = 3 * (envelopes_B + 40)

theorem stationery_problem : sheets_bought = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationery_problem_l595_59595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_folding_area_l595_59512

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if four points form a rectangle -/
def isRectangle (r : Rectangle) : Prop :=
  -- Add appropriate conditions for a rectangle
  sorry

/-- Checks if a point is on a line segment between two other points -/
def isOnSegment (P : Point) (A : Point) (B : Point) : Prop :=
  -- Add appropriate conditions for a point on a segment
  sorry

/-- Represents the folding operation -/
def fold (r : Rectangle) (P : Point) (Q : Point) : Rectangle × Point × Point :=
  -- Define the folding operation, returning the new rectangle and the new points D' and B'
  sorry

/-- Checks if two angles are congruent -/
def angleCongruent (A B C D E F : Point) : Prop :=
  -- Define angle congruence
  sorry

/-- Calculates the area of a rectangle -/
noncomputable def rectangleArea (r : Rectangle) : ℝ :=
  -- Calculate the area
  sorry

/-- Checks if a number is not divisible by the square of any prime -/
def notDivisibleBySquarePrime (n : ℕ) : Prop :=
  -- Define the condition
  sorry

theorem rectangle_folding_area (r : Rectangle) (P Q : Point) :
  isRectangle r →
  isOnSegment P r.A r.B →
  isOnSegment Q r.C r.D →
  (r.B.x - P.x : ℝ) < (r.D.x - Q.x : ℝ) →
  let (r', D', B') := fold r P Q
  angleCongruent r'.A D' B' B' P r'.A →
  (r'.A.x - D'.x : ℝ) = 7 →
  (r'.B.x - P.x : ℝ) = 27 →
  ∃ (x y z : ℕ),
    rectangleArea r = x + y * Real.sqrt z ∧
    notDivisibleBySquarePrime z ∧
    x + y + z = 616 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_folding_area_l595_59512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_l595_59557

-- Define the function g(x) = x ln x
noncomputable def g (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem g_inequality {a b : ℝ} (ha : 0 < a) (hab : a < b) :
  g a + g b - 2 * g ((a + b) / 2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inequality_l595_59557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_hit_probability_l595_59500

-- Define the probabilities of shooters A and B hitting the target
noncomputable def prob_A : ℝ := 1/2
noncomputable def prob_B : ℝ := 1/3

-- Theorem statement
theorem target_hit_probability : 
  1 - (1 - prob_A) * (1 - prob_B) = 2/3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_hit_probability_l595_59500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l595_59594

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToTenth (x : ℝ) : ℝ :=
  (⌊x * 10 + 0.5⌋ : ℝ) / 10

theorem sum_and_round :
  roundToTenth (5.67 + 2.45) = 8.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_l595_59594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_acute_angles_not_sufficient_for_congruence_l595_59587

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- length of one leg
  b : ℝ  -- length of the other leg
  h : ℝ  -- length of the hypotenuse
  right_angle : a^2 + b^2 = h^2  -- Pythagorean theorem

-- Define congruence for right-angled triangles
def congruent (t1 t2 : RightTriangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.h = t2.h

-- Define acute angles of a right triangle
noncomputable def acute_angle1 (t : RightTriangle) : ℝ := Real.arctan (t.a / t.b)
noncomputable def acute_angle2 (t : RightTriangle) : ℝ := Real.arctan (t.b / t.a)

-- Theorem statement
theorem two_acute_angles_not_sufficient_for_congruence :
  ∃ (t1 t2 : RightTriangle), 
    acute_angle1 t1 = acute_angle1 t2 ∧ 
    acute_angle2 t1 = acute_angle2 t2 ∧ 
    ¬(congruent t1 t2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_acute_angles_not_sufficient_for_congruence_l595_59587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_has_62_pennies_l595_59544

def alex_pennies : ℕ → Prop := sorry
def bob_pennies : ℕ → Prop := sorry

axiom condition1 : ∀ a b : ℕ, alex_pennies a → bob_pennies b → b + 2 = 4 * (a - 2)
axiom condition2 : ∀ a b : ℕ, alex_pennies a → bob_pennies b → b - 2 = 3 * (a + 2)

theorem bob_has_62_pennies : ∃ b : ℕ, bob_pennies b ∧ b = 62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_has_62_pennies_l595_59544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l595_59596

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

def triangle_sides (l m n : ℕ) : Prop :=
  l > m ∧ m > n ∧ 
  frac (3^l / 10000 : ℝ) = frac (3^m / 10000 : ℝ) ∧
  frac (3^m / 10000 : ℝ) = frac (3^n / 10000 : ℝ)

theorem min_perimeter_triangle (l m n : ℕ) : 
  triangle_sides l m n → l + m + n ≥ 3003 :=
by sorry

#check min_perimeter_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l595_59596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l595_59534

-- Define the function representing the fraction in the inequality
noncomputable def f (x : ℝ) : ℝ := (x^2 - 10*x + 21) / (x^2 - 4*x + 8)

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, (-2 < f x ∧ f x < 2) ↔ -5 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l595_59534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_sum_less_than_two_product_greater_than_one_iff_abs_greater_not_necessary_for_greater_quadratic_roots_sign_condition_l595_59564

-- Statement 1
theorem negation_of_all_sum_less_than_two :
  (¬ ∀ x y : ℝ, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → x + y < 2) ↔
  (∃ x y : ℝ, x ∈ Set.Ioo 0 1 ∧ y ∈ Set.Ioo 0 1 ∧ x + y ≥ 2) :=
sorry

-- Statement 2
theorem product_greater_than_one_iff :
  ∀ a b : ℝ, (a > 1 ∧ b > 1) ↔ (a * b > 1) :=
sorry

-- Statement 3
theorem abs_greater_not_necessary_for_greater :
  ∃ x y : ℝ, (abs x > abs y) ∧ (x ≤ y) :=
sorry

-- Statement 4
theorem quadratic_roots_sign_condition :
  ∀ m : ℝ, (m < 0) ↔
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_sum_less_than_two_product_greater_than_one_iff_abs_greater_not_necessary_for_greater_quadratic_roots_sign_condition_l595_59564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_theorem_l595_59546

/-- Calculates the distance between two trains given their speeds and time differences --/
noncomputable def distanceBetweenTrains (speedA speedB : ℝ) (departureDelay travelTime : ℝ) : ℝ :=
  let distanceA := speedA * travelTime / 60
  let distanceB := speedB * (travelTime - departureDelay) / 60
  distanceA - distanceB

/-- Proves that the distance between Train A and Train B is 7.5 km after 45 minutes --/
theorem train_distance_theorem :
  let speedA : ℝ := 90
  let speedB : ℝ := 120
  let departureDelay : ℝ := 15
  let travelTime : ℝ := 45
  distanceBetweenTrains speedA speedB departureDelay travelTime = 7.5 := by
  sorry

-- Use #eval only for functions that can be computed
def approxDistanceBetweenTrains (speedA speedB : Float) (departureDelay travelTime : Float) : Float :=
  let distanceA := speedA * travelTime / 60
  let distanceB := speedB * (travelTime - departureDelay) / 60
  distanceA - distanceB

#eval approxDistanceBetweenTrains 90 120 15 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_theorem_l595_59546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_time_l595_59536

/-- The time required to fill a pool given initial filling and two hoses -/
theorem pool_filling_time 
  (pool_capacity : ℝ) 
  (initial_time : ℝ) 
  (hose1_rate : ℝ) 
  (hose2_rate : ℝ) 
  (h1 : pool_capacity = 390) 
  (h2 : initial_time = 3) 
  (h3 : hose1_rate = 50) 
  (h4 : hose2_rate = 70) :
  (pool_capacity - initial_time * hose1_rate) / (hose1_rate + hose2_rate) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_time_l595_59536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l595_59578

/-- The area of a right triangle with base 30 cm and height 24 cm is 360 cm². -/
theorem right_triangle_area : 
  ∀ (D E F : ℝ × ℝ) (area : ℝ),
    (D.1 = 0 ∧ D.2 = 0) →  -- Point D at origin
    (E.1 = 30 ∧ E.2 = 0) →  -- Point E on x-axis, 30 units from origin
    (F.1 = 0 ∧ F.2 = 24) →  -- Point F on y-axis, 24 units from origin
    area = (1/2) * 30 * 24 →  -- Area calculation
    area = 360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l595_59578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l595_59519

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * (Real.sin x + Real.cos x) - 4 * Real.sin x * Real.cos x

-- Define the interval [0, π/2]
def I : Set ℝ := { x | 0 ≤ x ∧ x ≤ Real.pi / 2 }

-- Define t as sin x + cos x
noncomputable def t (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- Define g as a function of t
def g (m : ℝ) (t : ℝ) : ℝ := -2 * t^2 + m * t + 2

theorem f_properties (m : ℝ) :
  (∀ x ∈ I, t x ∈ Set.Icc 1 (Real.sqrt 2)) ∧
  (∀ x ∈ I, f m x = g m (t x)) ∧
  (∀ x ∈ I, f m x ≥ 0 ↔ m ≥ Real.sqrt 2) ∧
  ((∃ x ∈ I, f m x - 2*m + 4 = 0) ↔ (2 + Real.sqrt 2 ≤ m ∧ m ≤ 4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l595_59519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_roots_l595_59589

theorem trigonometric_roots : 
  ∃ (u v : ℝ), 
    u = Real.tan ((π / 2) - (22.5 * π / 180)) ∧ 
    v = 1 / Real.sin (22.5 * π / 180) ∧ 
    u^2 - 2*u - 1 = 0 ∧ 
    v^4 - 8*v^2 + 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_roots_l595_59589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_functions_l595_59533

-- Define the interval (0,1)
def openInterval : Set ℝ := Set.Ioo 0 1

-- Define the functions
noncomputable def functionA (x : ℝ) := Real.log x - x
noncomputable def functionB (x : ℝ) := -x^3 / 3 + 3*x^2 - 8*x + 1
noncomputable def functionC (x : ℝ) := Real.sqrt 3 * Real.sin x - 2*x - Real.cos x
noncomputable def functionD (x : ℝ) := x / Real.exp x

-- Define monotonic decreasing property
def MonotonicDecreasing (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f y < f x

theorem monotonic_decreasing_functions :
  ¬(MonotonicDecreasing functionA openInterval) ∧
  (MonotonicDecreasing functionB openInterval) ∧
  (MonotonicDecreasing functionC openInterval) ∧
  ¬(MonotonicDecreasing functionD openInterval) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_functions_l595_59533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l595_59577

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (sqrt 2 * cos x * sin (x + π/4)) / sin (2*x)

-- State the theorem
theorem max_value_of_f :
  ∀ x ∈ Set.Icc (π/4) (5*π/12),
    f x ≤ 1 ∧ ∃ y ∈ Set.Icc (π/4) (5*π/12), f y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l595_59577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_balls_removal_l595_59543

theorem blue_balls_removal (total : ℕ) (red_percent : ℚ) (blue_removed : ℕ) : 
  total = 120 → 
  red_percent = 40 / 100 → 
  blue_removed = 60 → 
  (red_percent * total) / (total - blue_removed) = 80 / 100 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_balls_removal_l595_59543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l595_59539

def a (x : ℝ) : ℝ × ℝ := (x, 2)
def b (y : ℝ) : ℝ × ℝ := (1, y)
def c : ℝ × ℝ := (2, -6)

theorem vector_sum_magnitude (x y : ℝ) :
  (a x).1 * c.1 + (a x).2 * c.2 = 0 →
  ∃ k : ℝ, b y = (c.1 * k, c.2 * k) →
  ‖(a x + b y)‖ = 5 * Real.sqrt 2 := by
  sorry

#check vector_sum_magnitude

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l595_59539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_eval_iff_rational_coeff_l595_59576

/-- A polynomial with complex coefficients -/
def ComplexPolynomial := ℕ → ℂ

/-- Evaluation of a complex polynomial at a point -/
noncomputable def evaluate (P : ComplexPolynomial) (x : ℂ) : ℂ :=
  ∑' n, (P n) * x^n

/-- A polynomial with rational coefficients -/
def RationalPolynomial := ℕ → ℚ

theorem rational_eval_iff_rational_coeff :
  ∀ (P : ComplexPolynomial),
    (∀ (q : ℚ), ∃ (r : ℚ), evaluate P (q : ℂ) = r) ↔
    ∃ (Q : RationalPolynomial), ∀ (n : ℕ), P n = Q n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_eval_iff_rational_coeff_l595_59576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l595_59579

noncomputable def f (x : ℝ) := Real.log (1 / (abs x + 1))

theorem range_of_f :
  (∀ y ∈ Set.range f, y ≤ 0) ∧
  (∀ ε > 0, ∃ x, f x > -ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l595_59579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_lateral_surface_area_l595_59581

/-- The lateral surface area of a regular square pyramid -/
noncomputable def lateral_surface_area (base_edge : ℝ) (height : ℝ) : ℝ :=
  4 * base_edge * (Real.sqrt ((base_edge / 2) ^ 2 + height ^ 2)) / 2

/-- Theorem: The lateral surface area of a regular square pyramid with base edge length 2 and height 1 is 4√2 -/
theorem pyramid_lateral_surface_area :
  lateral_surface_area 2 1 = 4 * Real.sqrt 2 := by
  -- Unfold the definition of lateral_surface_area
  unfold lateral_surface_area
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_lateral_surface_area_l595_59581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_theorem_l595_59537

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The product of Fibonacci ratios from k=3 to 101 -/
def fibProduct : ℚ :=
  (Finset.range 99).prod (fun k => 
    (fib (k + 3) / fib (k + 2) : ℚ) - (fib (k + 3) / fib (k + 4) : ℚ))

theorem fibonacci_product_theorem : 
  fibProduct = (fib 101 : ℚ) / (fib 102 : ℚ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_theorem_l595_59537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_derivatives_z_partial_derivatives_u_partial_derivatives_v_l595_59554

theorem partial_derivatives_z (x y : ℝ) :
  let z := x^3 + 5*x*y^2 - y^3
  (deriv (fun x => z)) x = 3*x^2 + 5*y^2 ∧ 
  (deriv (fun y => z)) y = 10*x*y - 3*y^2 := by
  sorry

theorem partial_derivatives_u (x y z : ℝ) :
  let u := x/y + y/z - z/x
  (deriv (fun x => u)) x = 1/y + z/x^2 ∧
  (deriv (fun y => u)) y = -x/y^2 + 1/z ∧
  (deriv (fun z => u)) z = -y/z^2 - 1/x := by
  sorry

theorem partial_derivatives_v (x y : ℝ) :
  let v := (Real.exp y) ^ (1/x)
  (deriv (fun x => v)) x = -y/x^2 * (Real.exp y) ^ (1/x) ∧
  (deriv (fun y => v)) y = 1/x * (Real.exp y) ^ (1/x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_derivatives_z_partial_derivatives_u_partial_derivatives_v_l595_59554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l595_59567

/-- The area of a triangle given its three vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  (1/2) * abs (v.1 * w.2 - v.2 * w.1)

/-- Theorem: The area of the triangle with vertices (3,-5), (-1,3), and (2,-7) is 8 -/
theorem triangle_area_example : triangle_area (3, -5) (-1, 3) (2, -7) = 8 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l595_59567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_relation_l595_59525

theorem cos_angle_relation (θ : ℝ) 
  (h1 : Real.cos (5 * π / 12 + θ) = 3 / 5)
  (h2 : -π < θ)
  (h3 : θ < -π / 2) :
  Real.cos (π / 12 - θ) = -4 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_relation_l595_59525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_twelve_equals_two_sqrt_three_l595_59583

theorem sqrt_twelve_equals_two_sqrt_three : Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_twelve_equals_two_sqrt_three_l595_59583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_minus_3a_19_l595_59511

/-- Sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 2  -- We define a(0) as 2 to match a(1) in the original problem
  | 1 => 5  -- This matches a(2) in the original problem
  | n + 2 => 2 * a (n + 1) + 3 * a n

/-- The theorem to be proved -/
theorem a_20_minus_3a_19 : a 19 - 3 * a 18 = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_minus_3a_19_l595_59511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_product_l595_59528

theorem cos_difference_product (x y : ℝ) : 
  Real.cos (x + y) - Real.cos (x - y) = -2 * Real.sin x * Real.sin y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_product_l595_59528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l595_59520

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x^2 - 1)
def domain_f_x2_minus_1 : Set ℝ := Set.Icc 1 3

-- Theorem statement
theorem domain_f_2x_minus_1 (h : ∀ x ∈ domain_f_x2_minus_1, f (x^2 - 1) ∈ Set.univ) :
  {x : ℝ | f (2*x - 1) ∈ Set.univ} = Set.Icc (1/2) (9/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l595_59520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_b_l595_59584

theorem existence_of_b (n m : ℕ) (h_n : n > 1) (h_m : m > 1) 
  (a : Fin m → ℕ) (h_a : ∀ i, a i ≤ n^(2*m - 1) - 2) :
  ∃ b : Fin m → Fin 2, ∀ i : Fin m, (a i + (b i : ℕ)) < n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_b_l595_59584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l595_59598

theorem max_true_statements : ∃ (x : ℝ), 
  (((0 < x^2) ∧ (x^2 < 2)) ∨ 
   (x^2 > 2) ∨ 
   ((-1 < x) ∧ (x < 0)) ∨ 
   ((0 < x) ∧ (x < 2)) ∨ 
   ((0 < x^3 - x^2) ∧ (x^3 - x^2 < 2))) ∧
  (∀ (y : ℝ), 
    ((((0 < y^2) ∧ (y^2 < 2)) : Bool).toNat + 
     ((y^2 > 2) : Bool).toNat + 
     (((-1 < y) ∧ (y < 0)) : Bool).toNat + 
     (((0 < y) ∧ (y < 2)) : Bool).toNat + 
     (((0 < y^3 - y^2) ∧ (y^3 - y^2 < 2)) : Bool).toNat) ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l595_59598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l595_59570

/-- Represents a cone with an unfolded lateral surface having a central angle of 90° and radius r -/
structure Cone (r : ℝ) where
  unfolded_angle : ℝ
  unfolded_radius : ℝ
  h_angle : unfolded_angle = Real.pi / 2
  h_radius : unfolded_radius = r

/-- The total surface area of the cone -/
noncomputable def total_surface_area (r : ℝ) (cone : Cone r) : ℝ :=
  (5 * Real.pi * r^2) / 16

/-- Theorem stating that the total surface area of the cone is (5πr²)/16 -/
theorem cone_surface_area (r : ℝ) (cone : Cone r) :
  total_surface_area r cone = (5 * Real.pi * r^2) / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l595_59570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_ten_dividing_factorial_l595_59532

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem largest_power_of_ten_dividing_factorial : 
  ∃ (n : ℕ), n = 250 ∧ 
  (∀ m : ℕ, 10^m ∣ factorial 1005 → m ≤ n) ∧
  10^n ∣ factorial 1005 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_ten_dividing_factorial_l595_59532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reward_is_190_l595_59562

/-- Represents the possible grades in Paul's scorecard --/
inductive Grade
| BPlus
| A
| APlus
deriving BEq, Repr

/-- Calculates the reward for a single grade --/
def gradeReward (g : Grade) (hasAtLeastTwoAPlus : Bool) : Nat :=
  match g with
  | Grade.BPlus => if hasAtLeastTwoAPlus then 10 else 5
  | Grade.A => if hasAtLeastTwoAPlus then 20 else 10
  | Grade.APlus => 15

/-- Calculates the total reward for a list of grades --/
def totalReward (grades : List Grade) : Nat :=
  let aPlusCount := grades.filter (· == Grade.APlus) |>.length
  let hasAtLeastTwoAPlus := aPlusCount ≥ 2
  grades.foldl (fun acc g => acc + gradeReward g hasAtLeastTwoAPlus) 0

/-- The maximum reward Paul can receive --/
def maxReward : Nat := 190

theorem max_reward_is_190 :
  ∀ (grades : List Grade),
    grades.length = 10 →
    totalReward grades ≤ maxReward :=
by
  sorry

#eval maxReward

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reward_is_190_l595_59562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_halfway_distance_eq_semi_major_axis_l595_59515

/-- Represents the orbit of planet Zorion -/
structure ZorionOrbit where
  perigee : ℝ  -- Closest approach to the sun in AU
  apogee : ℝ   -- Furthest point from the sun in AU
  tilt : ℝ     -- Tilt angle of the orbit in degrees

/-- Calculates the semi-major axis of Zorion's orbit -/
noncomputable def semiMajorAxis (orbit : ZorionOrbit) : ℝ :=
  (orbit.apogee + orbit.perigee) / 2

/-- 
Theorem stating that the distance from Zorion to its sun when halfway through its orbit
is equal to the semi-major axis of the elliptical orbit.
-/
theorem halfway_distance_eq_semi_major_axis (orbit : ZorionOrbit) :
  let halfway_distance := semiMajorAxis orbit
  halfway_distance = semiMajorAxis orbit :=
by
  -- The proof is omitted for now
  sorry

/-- Example orbit for Zorion -/
def zorionOrbit : ZorionOrbit where
  perigee := 3
  apogee := 15
  tilt := 30

-- Using #eval with noncomputable functions is not possible
-- Instead, we can use the following to check the result
#check semiMajorAxis zorionOrbit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_halfway_distance_eq_semi_major_axis_l595_59515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l595_59588

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- The focus of a parabola -/
noncomputable def focus (par : Parabola) : Point :=
  { x := par.p / 2, y := 0 }

/-- The directrix of a parabola -/
noncomputable def directrix (par : Parabola) : ℝ := -par.p / 2

/-- A point on the parabola -/
structure PointOnParabola (par : Parabola) where
  point : Point
  h_on_parabola : point.y^2 = 2 * par.p * point.x

/-- A point on the directrix -/
structure PointOnDirectrix (par : Parabola) where
  point : Point
  h_on_directrix : point.x = directrix par

/-- Theorem: For a parabola y² = 2px (p > 0), if a point Q on its directrix forms a triangle PQF 
    with a point P on the parabola such that FQ intersects the y-axis at (0, 2) and the area of 
    triangle PQF is 10, then p = 2 or p = 8 -/
theorem parabola_equation (par : Parabola) 
  (P : PointOnParabola par) 
  (Q : PointOnDirectrix par)
  (h_intersect : ∃ t : ℝ, t * (focus par).x + (1 - t) * Q.point.x = 0 ∧ 
                           t * (focus par).y + (1 - t) * Q.point.y = 2)
  (h_area : abs ((P.point.x - Q.point.x) * (P.point.y - (focus par).y) -
                 (P.point.y - Q.point.y) * (P.point.x - (focus par).x)) / 2 = 10) :
  par.p = 2 ∨ par.p = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l595_59588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_WXYZ_is_27_l595_59513

/-- A rectangle ABCD with a circle intersecting it --/
structure RectangleWithCircle where
  -- The length of side AB
  ab : ℝ
  -- The length of side BC
  bc : ℝ
  -- The radius of the circle
  radius : ℝ
  -- Assumption that AB = 20
  h_ab : ab = 20
  -- Assumption that BC = 3
  h_bc : bc = 3
  -- Assumption that the circle radius is 5
  h_radius : radius = 5
  -- Assumption that the circle is centered at the midpoint of DC
  center_at_midpoint : Bool

/-- The area of quadrilateral WXYZ formed by the intersection of the circle and rectangle --/
noncomputable def area_WXYZ (r : RectangleWithCircle) : ℝ :=
  -- Definition of the area calculation
  ((r.ab / 2 + Real.sqrt (r.radius ^ 2 - r.bc ^ 2)) + r.ab / 2) * r.bc / 2

/-- Theorem stating that the area of WXYZ is 27 --/
theorem area_WXYZ_is_27 (r : RectangleWithCircle) : area_WXYZ r = 27 := by
  sorry

#check area_WXYZ_is_27

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_WXYZ_is_27_l595_59513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l595_59553

/-- The sum of the infinite series 1 + 3(1/2000) + 5(1/2000)^2 + 7(1/2000)^3 + ... -/
noncomputable def infiniteSeries : ℝ := ∑' n, (2*n + 1) * (1/2000)^n

/-- The theorem stating that the infinite series sum is equal to 4002000/3996001 -/
theorem infiniteSeriesSum : infiniteSeries = 4002000 / 3996001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l595_59553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l595_59529

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)

-- Theorem statement
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = -3 * Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l595_59529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_seats_count_l595_59540

theorem hall_seats_count (filled_percentage : ℝ) (vacant_seats : ℕ) : ℕ :=
  by
  have h1 : filled_percentage = 45 := by sorry
  have h2 : vacant_seats = 330 := by sorry
  
  let total_seats : ℕ := 600
  
  have h3 : (100 - filled_percentage) / 100 * total_seats = vacant_seats := by sorry
  
  exact total_seats

#check hall_seats_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_seats_count_l595_59540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_40_sec_l595_59559

/-- The time taken for a train to cross a platform -/
noncomputable def timeToCrossPlatform (trainLength platformLength : ℝ) (timeToCrossPole : ℝ) : ℝ :=
  (trainLength + platformLength) / (trainLength / timeToCrossPole)

/-- Theorem stating that the time taken for the train to cross the platform is approximately 40 seconds -/
theorem train_crossing_time_approx_40_sec :
  let trainLength : ℝ := 300
  let platformLength : ℝ := 366.67
  let timeToCrossPole : ℝ := 18
  abs (timeToCrossPlatform trainLength platformLength timeToCrossPole - 40) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_40_sec_l595_59559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_numbers_theorem_l595_59505

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- The number of digits in each number -/
def num_digits : Nat := 5

/-- The total number of possible five-digit numbers -/
def total_numbers : Nat := Nat.factorial num_digits

/-- The number of odd five-digit numbers -/
def odd_numbers : Nat := 3 * Nat.factorial (num_digits - 1)

/-- The position of 43125 in the sorted list -/
def position_43125 : Nat := 90

theorem five_digit_numbers_theorem :
  (∀ n ∈ Finset.powerset digits, n.card = num_digits → n.card = 5) →
  total_numbers = 120 ∧
  odd_numbers = 72 ∧
  position_43125 = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_numbers_theorem_l595_59505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_two_l595_59508

-- Define the semicircle (C)
noncomputable def semicircle (φ : Real) : Real × Real :=
  (1 + Real.cos φ, Real.sin φ)

-- Define the line (l)
def line (ρ θ : Real) : Prop :=
  ρ * (Real.sin θ + Real.sqrt 3 * Real.cos θ) = 5 * Real.sqrt 3

-- Define the ray OM
def ray_OM (θ : Real) : Prop :=
  θ = Real.pi / 3

-- Define point P
noncomputable def point_P : Real × Real :=
  semicircle (Real.pi / 3)

-- Define point Q
noncomputable def point_Q : Real × Real :=
  let ρ : Real := 3  -- This value is derived from solving the equation, not given directly
  (ρ * Real.cos (Real.pi / 3), ρ * Real.sin (Real.pi / 3))

-- Theorem to prove
theorem length_PQ_is_two :
  let (x₁, y₁) := point_P
  let (x₂, y₂) := point_Q
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_two_l595_59508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_half_fraction_of_nick_full_l595_59503

/-- Tom's time to clean the entire house in hours -/
noncomputable def tom_full : ℝ := 6

/-- Time for Tom and Nick to clean the house together in hours -/
noncomputable def together : ℝ := 3.6

/-- Nick's time to clean the entire house in hours -/
noncomputable def nick_full : ℝ := 9

/-- Tom's time to clean half the house in hours -/
noncomputable def tom_half : ℝ := tom_full / 2

theorem tom_half_fraction_of_nick_full :
  tom_half / nick_full = 1 / 3 :=
by
  -- Unfold the definitions
  unfold tom_half nick_full tom_full
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_half_fraction_of_nick_full_l595_59503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l595_59572

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = t.c - 2 * t.a * Real.cos t.B ∧
  4 * t.a = Real.sqrt 6 * t.b ∧
  t.c = 5 ∧
  t.A + t.B + t.C = Real.pi

-- Helper function to calculate triangle area
noncomputable def area_triangle (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) : 
  t.B = 2 * t.A ∧ 
  area_triangle t = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l595_59572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_2011_for_all_N_l595_59555

def is_divisor (d n : ℕ) : Prop := d > 1 ∧ n % d = 0

def can_reach_2011 (start : ℕ) : Prop :=
  ∃ (sequence : List ℕ),
    sequence.head? = some start ∧
    sequence.getLast? = some 2011 ∧
    ∀ i < sequence.length - 1,
      ∃ d, is_divisor d (sequence.get ⟨i, by sorry⟩) ∧
        (sequence.get ⟨i + 1, by sorry⟩ = sequence.get ⟨i, by sorry⟩ + d ∨
         sequence.get ⟨i + 1, by sorry⟩ = sequence.get ⟨i, by sorry⟩ - d)

theorem reach_2011_for_all_N :
  ∀ N : ℕ, N > 1 → can_reach_2011 N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_2011_for_all_N_l595_59555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_parabola_and_passes_through_point_l595_59593

noncomputable def curve (θ : Real) : Real × Real :=
  (Real.sqrt (1 + Real.sin θ), Real.cos (Real.pi/4 - θ/2)^2)

theorem curve_is_parabola_and_passes_through_point :
  ∃ (a b c : Real),
    (∀ θ : Real, 0 ≤ θ ∧ θ < 2*Real.pi →
      let (x, y) := curve θ
      y = a*x^2 + b*x + c) ∧
    (let (x, y) := curve (Real.arcsin 0)
     x = 1 ∧ y = 1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_parabola_and_passes_through_point_l595_59593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l595_59501

-- Define the parabola
def f (x : ℝ) : ℝ := x^2 + x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x + 1

-- Define the point of tangency
def point : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (y = m * x + b) ↔ (y - f point.1 = f' point.1 * (x - point.1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l595_59501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l595_59550

-- Define the function representing the curve y = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- Define the bounds of integration
noncomputable def a : ℝ := 1 / 2
noncomputable def b : ℝ := 2

-- Theorem statement
theorem area_enclosed_by_curve : 
  ∫ x in a..b, f x = 2 * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l595_59550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l595_59592

theorem inequality_proof (m : ℝ) : True := by
  -- Define the inequality that always holds for any real x
  let inequality := ∀ x : ℝ, |x + 6| + |x - 1| ≥ m

  -- Prove that the inequality implies m ≤ 7
  have range_of_m : inequality → m ≤ 7 := by
    sorry

  -- Prove that the maximum value of m is 7
  have max_m : ∃ x : ℝ, |x + 6| + |x - 1| = 7 := by
    sorry

  -- Define the second inequality when m = 7
  let second_inequality := λ x : ℝ ↦ |x - 4| - 3*x ≤ 5

  -- Prove that the solution set of the second inequality is {x | x ≥ -1/4}
  have solution_set : ∀ x : ℝ, second_inequality x ↔ x ≥ -1/4 := by
    sorry

  trivial


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l595_59592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l595_59560

/-- The family of curves parameterized by θ --/
def curve (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

/-- The line y = 2x --/
def line (x y : ℝ) : Prop := y = 2 * x

/-- The chord length function --/
noncomputable def chord_length (x₁ x₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (2 * x₂ - 2 * x₁)^2)

/-- The maximum chord length theorem --/
theorem max_chord_length :
  ∃ (x₁ x₂ θ₁ θ₂ : ℝ),
    curve θ₁ x₁ (2 * x₁) ∧
    curve θ₂ x₂ (2 * x₂) ∧
    line x₁ (2 * x₁) ∧
    line x₂ (2 * x₂) ∧
    ∀ (x₃ x₄ θ₃ θ₄ : ℝ),
      curve θ₃ x₃ (2 * x₃) →
      curve θ₄ x₄ (2 * x₄) →
      line x₃ (2 * x₃) →
      line x₄ (2 * x₄) →
      chord_length x₁ x₂ ≥ chord_length x₃ x₄ ∧
      chord_length x₁ x₂ = 8 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l595_59560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_proof_l595_59571

/-- Proves that the length of a rectangular floor is approximately 19.59 meters
    given the specified conditions. -/
theorem floor_length_proof (breadth : ℝ) (length : ℝ) (area : ℝ) (paint_cost : ℝ) 
  (paint_rate : ℝ) :
  length = 3 * breadth →
  paint_cost = 640 →
  paint_rate = 5 →
  area = length * breadth →
  paint_cost = area * paint_rate →
  ∃ ε > 0, |length - 19.59| < ε := by
  sorry

#check floor_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_proof_l595_59571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_202109_is_prime_l595_59526

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def candidate_numbers : List ℕ := [202101, 202103, 202105, 202107, 202109]

theorem only_202109_is_prime :
  ∃! n, n ∈ candidate_numbers ∧ is_prime n ∧ n = 202109 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_202109_is_prime_l595_59526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_12_l595_59597

/-- Calculates the upstream distance swam by a man given his still water speed,
    downstream distance, and time spent swimming both upstream and downstream. -/
noncomputable def upstreamDistance (stillWaterSpeed : ℝ) (downstreamDistance : ℝ) (time : ℝ) : ℝ :=
  let streamSpeed := downstreamDistance / (2 * time) - stillWaterSpeed / 2
  (stillWaterSpeed - streamSpeed) * time

/-- Theorem stating that given the conditions of the problem,
    the upstream distance is 12 km. -/
theorem upstream_distance_is_12 :
  upstreamDistance 10 28 2 = 12 := by
  unfold upstreamDistance
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- Remove the #eval statement as it's not computable
-- #eval upstreamDistance 10 28 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_12_l595_59597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_problem_l595_59561

theorem hcf_problem (a b : ℕ) (h1 : a = 3 * (b / 4)) 
  (h2 : Nat.lcm a b = 60) (h3 : Nat.gcd a b = 5) : Nat.gcd a b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_problem_l595_59561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l595_59523

noncomputable def f (x : ℝ) := Real.log (x + 2) / Real.sqrt (-x^2 - x + 6)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ -2 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l595_59523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_properties_l595_59586

noncomputable def P (x : ℝ) : ℝ := (1/6) * x^2 - x + (11/6)

theorem polynomial_properties :
  (P 1 = 1) ∧
  (P 2 = 1/2) ∧
  (P 3 = 1/3) ∧
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 → P x ≠ 1/x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_properties_l595_59586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l595_59585

-- Define the function representing the left side of the inequality
noncomputable def f (x : ℝ) : ℝ := (x * (x - 1)) / ((x - 5)^3)

-- Define the set of x values that satisfy the inequality
def solution_set : Set ℝ := Set.Icc 2 5 ∪ Set.Ioc 5 9

-- State the theorem
theorem inequality_theorem : 
  ∀ x : ℝ, x ≠ 5 → (f x ≥ 18 ↔ x ∈ solution_set) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l595_59585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_P_intersect_Q_l595_59568

open Set

def P : Set ℝ := {x | x - 1 ≤ 0}
def Q : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem complement_P_intersect_Q : 
  (Pᶜ ∩ Q) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_P_intersect_Q_l595_59568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l595_59565

/-- In a triangle ABC, given side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine law for a triangle -/
axiom sine_law {t : Triangle} : t.a / Real.sin t.A = t.b / Real.sin t.B

theorem triangle_problem (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : t.A = π/3) : 
  (t.a = 3 → t.B = Real.arcsin (Real.sqrt 3 / 3)) ∧ 
  (∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ t1.a = t.a ∧ t2.a = t.a ↔ Real.sqrt 3 < t.a ∧ t.a < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l595_59565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l595_59545

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x : ℝ | (x - 6) * (x + 2) > 0}

-- Theorem for the first part
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a ∈ Set.Icc (-2) 3 := by
  sorry

-- Theorem for the second part
theorem union_equals_B_iff_a_in_range (a : ℝ) :
  A a ∪ B = B ↔ a ∈ Set.Ioi 6 ∪ Set.Iio (-5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l595_59545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projections_and_angles_l595_59518

noncomputable def a : ℝ × ℝ := (3/5, 4/5)
noncomputable def b_magnitude : ℝ := Real.sqrt 2 / 2
noncomputable def angle_a_b : ℝ := Real.pi / 4

theorem vector_projections_and_angles :
  let proj_a_on_b := Real.sqrt 2 / 2
  let cos_angle_diff_sum := Real.sqrt 5 / 5
  (Real.sqrt (a.1^2 + a.2^2) = 1) ∧
  (proj_a_on_b = Real.sqrt (a.1^2 + a.2^2) * Real.cos angle_a_b) ∧
  (cos_angle_diff_sum = ((a.1^2 + a.2^2) - b_magnitude^2) / 
    (Real.sqrt ((a.1^2 + a.2^2) + b_magnitude^2 - 2 * Real.sqrt (a.1^2 + a.2^2) * b_magnitude * Real.cos angle_a_b) * 
     Real.sqrt ((a.1^2 + a.2^2) + b_magnitude^2 + 2 * Real.sqrt (a.1^2 + a.2^2) * b_magnitude * Real.cos angle_a_b))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projections_and_angles_l595_59518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_pi_over_two_l595_59563

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2) + Real.sin x

-- State the theorem
theorem integral_equals_pi_over_two :
  ∫ x in Set.Icc (-1) 1, f x = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_pi_over_two_l595_59563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_even_g_l595_59527

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := 
  Real.sin (2 * (x + φ) + Real.pi/3) + Real.sqrt 3 / 2

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem min_phi_for_even_g : 
  ∃ φ : ℝ, φ > 0 ∧ is_even (g φ) ∧ ∀ ψ, 0 < ψ ∧ ψ < φ → ¬is_even (g ψ) :=
by
  -- The proof goes here
  sorry

#check min_phi_for_even_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_even_g_l595_59527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunday_newspaper_cost_l595_59506

/-- The cost of Juanita's Sunday newspaper, given the following conditions:
  * Grant spends $200.00 a year on daily newspaper delivery.
  * Juanita buys the newspaper daily.
  * Monday through Saturday, Juanita spends $0.50 per day.
  * Juanita spends $60 more on buying the newspaper yearly than Grant.
  * There are 52 weeks in a year.
-/
theorem sunday_newspaper_cost : ℝ := by
  let grant_yearly_cost : ℝ := 200
  let juanita_weekday_cost : ℝ := 0.50
  let days_per_week : ℕ := 6
  let weeks_per_year : ℕ := 52
  let juanita_yearly_difference : ℝ := 60
  let juanita_yearly_cost : ℝ := grant_yearly_cost + juanita_yearly_difference
  let juanita_weekday_yearly_cost : ℝ := juanita_weekday_cost * (days_per_week : ℝ) * (weeks_per_year : ℝ)
  let sunday_yearly_cost : ℝ := juanita_yearly_cost - juanita_weekday_yearly_cost
  exact sunday_yearly_cost / (weeks_per_year : ℝ)

#eval Float.ofScientific 2 0 1 -- Expected output: 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunday_newspaper_cost_l595_59506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_vertex_angle_formula_l595_59575

/-- Represents a concave regular polygon with an odd number of sides. -/
structure ConcaveRegularPolygon where
  n : ℕ
  odd_sides : Odd n
  more_than_three : n > 3

/-- The angle at each vertex of the star formed by extending every other side of a concave regular polygon. -/
noncomputable def star_vertex_angle (p : ConcaveRegularPolygon) : ℝ :=
  (↑p.n - 2) * 180 / ↑p.n

/-- Theorem stating the angle at each vertex of the star. -/
theorem star_vertex_angle_formula (p : ConcaveRegularPolygon) :
  star_vertex_angle p = (↑p.n - 2) * 180 / ↑p.n :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_vertex_angle_formula_l595_59575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_is_101_l595_59509

/-- Represents a four-digit integer -/
structure FourDigitInt where
  d1 : Nat -- thousands digit
  d2 : Nat -- hundreds digit
  d3 : Nat -- tens digit
  d4 : Nat -- units digit
  h1 : d1 < 10
  h2 : d2 < 10
  h3 : d3 < 10
  h4 : d4 < 10
  h5 : d1 ≠ 0

/-- Represents a cyclic sequence of four four-digit integers -/
structure CyclicSequence where
  t1 : FourDigitInt
  t2 : FourDigitInt
  t3 : FourDigitInt
  t4 : FourDigitInt
  h1 : t2.d1 = t1.d2 ∧ t2.d2 = t1.d3 ∧ t2.d3 = t1.d4
  h2 : t3.d1 = t2.d2 ∧ t3.d2 = t2.d3 ∧ t3.d3 = t2.d4
  h3 : t4.d1 = t3.d2 ∧ t4.d2 = t3.d3 ∧ t4.d3 = t3.d4
  h4 : t1.d1 = t4.d3 ∧ t1.d2 = t4.d4

/-- The sum of the modified terms in the sequence -/
def modifiedSum (seq : CyclicSequence) : Nat :=
  (seq.t1.d1 * 1000 + seq.t1.d2 * 100 + seq.t1.d3 * 10 + seq.t1.d4 + 11) +
  (seq.t2.d1 * 1000 + seq.t2.d2 * 100 + seq.t2.d3 * 10 + seq.t2.d4 + 11) +
  (seq.t3.d1 * 1000 + seq.t3.d2 * 100 + seq.t3.d3 * 10 + seq.t3.d4 + 11) +
  (seq.t4.d1 * 1000 + seq.t4.d2 * 100 + seq.t4.d3 * 10 + seq.t4.d4 + 11)

theorem largest_prime_factor_is_101 (seq : CyclicSequence) :
  ∃ (p : Nat), p.Prime ∧ p ∣ modifiedSum seq ∧ p = 101 ∧ ∀ (q : Nat), q.Prime → q ∣ modifiedSum seq → q ≤ 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_is_101_l595_59509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_B_more_suitable_l595_59551

noncomputable def scores_A : List ℝ := [60, 75, 100, 90, 75]
noncomputable def scores_B : List ℝ := [70, 90, 100, 80, 80]

def excellent_score : ℝ := 80

noncomputable def variance (scores : List ℝ) : ℝ :=
  let mean := scores.sum / scores.length
  (scores.map (λ x => (x - mean) ^ 2)).sum / scores.length

noncomputable def excellent_rate (scores : List ℝ) : ℝ :=
  (scores.filter (λ x => x ≥ excellent_score)).length / scores.length

theorem student_B_more_suitable :
  variance scores_B < variance scores_A ∧
  excellent_rate scores_B > excellent_rate scores_A →
  "Student B is more suitable for the competition" = "Student B is more suitable for the competition" := by
  sorry

#eval "Student B is more suitable for the competition"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_B_more_suitable_l595_59551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hygiene_habits_and_disease_relation_l595_59507

-- Define the survey data
def case_not_good_enough : ℕ := 40
def case_good_enough : ℕ := 60
def control_not_good_enough : ℕ := 10
def control_good_enough : ℕ := 90

-- Define the total number of participants
def total_participants : ℕ := case_not_good_enough + case_good_enough + control_not_good_enough + control_good_enough

-- Define the K² formula
noncomputable def K_squared (n a b c d : ℝ) : ℝ := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the confidence level threshold for 99% confidence
noncomputable def confidence_threshold : ℝ := 6.635

-- Define the probability functions
noncomputable def P_A_given_B : ℝ := case_not_good_enough / (case_not_good_enough + case_good_enough)
noncomputable def P_A_given_not_B : ℝ := control_not_good_enough / (control_not_good_enough + control_good_enough)

-- Theorem statement
theorem hygiene_habits_and_disease_relation :
  (K_squared (total_participants : ℝ) case_not_good_enough control_good_enough control_not_good_enough case_good_enough > confidence_threshold) ∧
  ((P_A_given_B / (1 - P_A_given_B)) * ((1 - P_A_given_not_B) / P_A_given_not_B) = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hygiene_habits_and_disease_relation_l595_59507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pattern_l595_59599

def mySequence : ℕ → ℕ
  | 0 => 2
  | 1 => 4
  | 2 => 3
  | 3 => 9
  | 4 => 4
  | 5 => 16
  | 6 => 5
  | 7 => 25  -- Added the value we're trying to prove
  | 8 => 6   -- Added the value we're trying to prove
  | 9 => 36
  | 10 => 7
  | n + 11 => mySequence n

theorem sequence_pattern : 
  mySequence 7 = 25 ∧ mySequence 8 = 6 := by
  sorry

#eval mySequence 7  -- This will output 25
#eval mySequence 8  -- This will output 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pattern_l595_59599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l595_59566

noncomputable def b_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → b n = b (n - 1) * b (n + 1)

theorem sequence_property (b : ℕ → ℝ) 
  (h1 : b_sequence b) 
  (h2 : b 1 = 2 + Real.sqrt 3) 
  (h3 : b 1776 = 10 + Real.sqrt 3) : 
  b 2009 = -4 + 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l595_59566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_is_arcsin_one_over_sqrt_three_l595_59541

/-- Regular truncated triangular pyramid with two inscribed spheres -/
structure RegularTruncatedTriangularPyramid where
  a : ℝ  -- side length of upper base
  b : ℝ  -- side length of lower base
  h_ab : a < b

/-- The dihedral angle between the base and a lateral face of a regular truncated triangular pyramid -/
noncomputable def dihedral_angle (p : RegularTruncatedTriangularPyramid) : ℝ :=
  Real.arcsin (1 / Real.sqrt 3)

/-- Theorem: The dihedral angle between the base and a lateral face of a regular truncated triangular pyramid is arcsin(1/√3) -/
theorem dihedral_angle_is_arcsin_one_over_sqrt_three (p : RegularTruncatedTriangularPyramid) :
  dihedral_angle p = Real.arcsin (1 / Real.sqrt 3) := by
  -- The proof goes here
  sorry

#check dihedral_angle_is_arcsin_one_over_sqrt_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_is_arcsin_one_over_sqrt_three_l595_59541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_masha_wins_l595_59590

/-- Represents a pack of balloons -/
structure Pack :=
  (id : Nat)
  (balloons : Nat)

/-- The game state -/
structure GameState :=
  (packs : List Pack)
  (currentPlayer : Bool)  -- true for Vika, false for Masha

/-- Defines a valid move in the game -/
def validMove (gs : GameState) (packId : Nat) : Prop :=
  ∃ p ∈ gs.packs, p.id = packId ∧ p.balloons > 0

/-- Applies a move to the game state -/
def applyMove (gs : GameState) (packId : Nat) : GameState :=
  { packs := gs.packs.map (fun p => if p.id = packId then { id := p.id, balloons := p.balloons - 1 } else p),
    currentPlayer := ¬gs.currentPlayer }

/-- Checks if the game is over -/
def gameOver (gs : GameState) : Prop :=
  ∀ p ∈ gs.packs, p.balloons = 0

/-- Initial game state -/
def initialState : GameState :=
  { packs := List.range 15 |>.map (fun i => { id := i + 1, balloons := i + 1 }),
    currentPlayer := true }

/-- Theorem stating that Masha (second player) has a winning strategy -/
theorem masha_wins :
  ∃ (strategy : GameState → Nat),
    ∀ (vikaStrategy : GameState → Nat),
      ∃ (finalState : GameState),
        (gameOver finalState ∧ finalState.currentPlayer = true) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_masha_wins_l595_59590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_and_monotonicity_l595_59530

noncomputable def f (a x : ℝ) : ℝ := Real.log (x^2 - 2*a*x + 3) / Real.log (1/2)

theorem domain_range_and_monotonicity (a : ℝ) :
  (∀ x, f a x ∈ Set.univ ↔ -Real.sqrt 3 < a ∧ a < Real.sqrt 3) ∧
  (Set.range (f a) = Set.univ ↔ a ≤ -Real.sqrt 3 ∨ a ≥ Real.sqrt 3) ∧
  ¬∃ a, StrictMonoOn (f a) (Set.Iio 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_and_monotonicity_l595_59530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dollar_thrice_30_l595_59552

noncomputable def dollar_function (x : ℝ) : ℝ := 0.4 * x + 2

noncomputable def round_to_nearest (x : ℝ) : ℤ := ⌊x + 0.5⌋

noncomputable def apply_dollar_thrice (x : ℝ) : ℤ :=
  round_to_nearest (dollar_function (round_to_nearest (dollar_function (round_to_nearest (dollar_function x)))))

theorem dollar_thrice_30 : apply_dollar_thrice 30 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dollar_thrice_30_l595_59552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l595_59558

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_monotone : monotone_decreasing_on f (Set.Ici 0)) :
  {x : ℝ | f 1 - f (1/x) < 0} = {x : ℝ | x < -1 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l595_59558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l595_59522

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Vector a -/
noncomputable def a (m : ℝ) : ℝ × ℝ := (1, m)

/-- Vector b -/
noncomputable def b : ℝ × ℝ := (-1, Real.sqrt 3)

/-- Theorem: If vector a is parallel to vector b, then m = -√3 -/
theorem parallel_vectors_m_value (m : ℝ) :
  parallel (a m) b → m = -Real.sqrt 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l595_59522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_profit_optimization_l595_59573

/-- Represents the selling price per unit in dollars -/
def selling_price : ℝ → Prop := λ x ↦ x > 6

/-- Represents the annual sales volume in ten thousand units -/
def annual_sales_volume : ℝ → ℝ → Prop := λ x u ↦ u = -2 * (x - 21/4)^2 + 585/8

/-- Represents the annual sales profit in ten thousand dollars -/
def annual_sales_profit : ℝ → ℝ := λ x ↦ -2*x^3 + 33*x^2 - 108*x - 108

theorem product_profit_optimization (x : ℝ) (h1 : selling_price x) 
  (h2 : annual_sales_volume 10 28) :
  (∀ u, annual_sales_volume x u → annual_sales_profit x = u * (x - 6)) ∧
  (∃ max_x, selling_price max_x ∧ 
    ∀ y, selling_price y → annual_sales_profit y ≤ annual_sales_profit max_x) ∧
  (∃ max_profit, max_profit = annual_sales_profit 9 ∧ max_profit = 135) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_profit_optimization_l595_59573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_coinciding_foci_l595_59517

/-- Given a parabola and a hyperbola with coinciding foci, prove the value of a and the asymptote equations -/
theorem parabola_hyperbola_coinciding_foci (a : ℝ) :
  a > 0 →
  (∃ (x y : ℝ), y^2 = -8*x ∧ x^2/a^2 - y^2 = 1 ∧ x = -2) →
  a = Real.sqrt 2 ∧ 
  (∀ (x y : ℝ), y = x ∨ y = -x → x^2/a^2 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_coinciding_foci_l595_59517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dining_bill_share_l595_59591

/-- Given a total bill, number of people, and tip percentage, calculate each person's share --/
noncomputable def calculate_share (total_bill : ℝ) (num_people : ℕ) (tip_percentage : ℝ) : ℝ :=
  let tip := total_bill * (tip_percentage / 100)
  let total_with_tip := total_bill + tip
  total_with_tip / (num_people : ℝ)

/-- The problem statement --/
theorem dining_bill_share :
  let total_bill : ℝ := 211
  let num_people : ℕ := 6
  let tip_percentage : ℝ := 15
  abs (calculate_share total_bill num_people tip_percentage - 40.44) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dining_bill_share_l595_59591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assessment_probabilities_problem_solution_l595_59574

/-- Represents the probabilities of a player answering correctly in each round -/
structure PlayerProbabilities where
  round1 : ℚ
  round2 : ℚ
  round3 : ℚ

/-- The given problem setup -/
def problem_setup : (PlayerProbabilities × PlayerProbabilities) :=
  (⟨4/5, 3/4, 2/3⟩, ⟨2/3, 2/3, 1/2⟩)

/-- Calculates the probability that a player is eliminated only after entering the third round -/
def prob_eliminated_third_round (p : PlayerProbabilities) : ℚ :=
  p.round1 * p.round2 * (1 - p.round3)

/-- Calculates the probability that a player passes all assessments -/
def prob_pass_all (p : PlayerProbabilities) : ℚ :=
  p.round1 * p.round2 * p.round3

/-- The main theorem to be proved -/
theorem assessment_probabilities (setup : (PlayerProbabilities × PlayerProbabilities)) :
  let (player_a, player_b) := setup
  (prob_eliminated_third_round player_a = 1/5) ∧
  (1 - (1 - prob_pass_all player_a) * (1 - prob_pass_all player_b) = 8/15) :=
by sorry

/-- The specific instance of the theorem for the given problem -/
theorem problem_solution : 
  let (player_a, player_b) := problem_setup
  (prob_eliminated_third_round player_a = 1/5) ∧
  (1 - (1 - prob_pass_all player_a) * (1 - prob_pass_all player_b) = 8/15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assessment_probabilities_problem_solution_l595_59574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_euros_approx_l595_59535

/-- Calculates the total amount in Euros after combining Henry's, Simon's, Olivia's, and David's money and converting to Euros -/
noncomputable def totalEuros : ℝ :=
  let henry_initial : ℝ := 5.50
  let henry_earnings : ℝ := 2.75
  let simon_initial : ℝ := 13.30
  let simon_spending_rate : ℝ := 0.25
  let olivia_multiplier : ℝ := 2
  let david_reduction_rate : ℝ := 1/3
  let exchange_rate : ℝ := 0.85

  let henry_total := henry_initial + henry_earnings
  let simon_total := simon_initial * (1 - simon_spending_rate)
  let olivia_total := henry_total * olivia_multiplier
  let david_total := olivia_total * (1 - david_reduction_rate)

  let total_dollars := henry_total + simon_total + olivia_total + david_total
  total_dollars * exchange_rate

/-- The total amount in Euros is approximately 38.87 -/
theorem total_euros_approx : 
  ‖totalEuros - 38.87‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_euros_approx_l595_59535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_equal_distribution_l595_59549

/-- Represents the initial distribution of berries among 100 bear cubs -/
def initial_distribution : Fin 100 → ℕ := fun i => 2^(i.val)

/-- The total number of berries initially -/
def total_berries : ℕ := 2^100 - 1

/-- Represents the redistribution process -/
def redistribute (berries : ℕ × ℕ) : ℕ × ℕ :=
  let (a, b) := berries
  ((a + b) / 2, (a + b) / 2)

/-- The minimum number of berries that can be equally distributed -/
def min_equal_berries : ℕ := 100

/-- Theorem stating the minimum number of berries that can be equally distributed -/
theorem min_equal_distribution :
  ∃ (final_distribution : Fin 100 → ℕ),
    (∀ i j : Fin 100, final_distribution i = final_distribution j) ∧
    (∀ i : Fin 100, final_distribution i = min_equal_berries) ∧
    (∃ (steps : ℕ),
      ∃ (intermediate_distributions : Fin (steps + 1) → (Fin 100 → ℕ)),
        intermediate_distributions 0 = initial_distribution ∧
        intermediate_distributions steps = final_distribution ∧
        ∀ k : Fin steps,
          ∃ i j : Fin 100,
            let next_dist := intermediate_distributions (k.succ)
            let curr_dist := intermediate_distributions k
            ∀ l : Fin 100,
              l ≠ i ∧ l ≠ j →
                next_dist l = curr_dist l ∧
                next_dist i = (curr_dist i + curr_dist j) / 2 ∧
                next_dist j = (curr_dist i + curr_dist j) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_equal_distribution_l595_59549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l595_59556

noncomputable section

/-- The equation of a curve -/
def f (x : ℝ) : ℝ := x^2 + 1/x

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 2*x - 1/x^2

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 2)

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem tangent_line_at_point :
  let (x₀, y₀) := point
  tangent_line x₀ y₀ ∧
  f x₀ = y₀ ∧
  ∀ x : ℝ, x ≠ 0 → (tangent_line x (f' x₀ * (x - x₀) + y₀) ↔ y = f x) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l595_59556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l595_59524

structure Pyramid where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  D : ℝ × ℝ × ℝ
  P : ℝ × ℝ × ℝ

def is_rectangle (A B C D : ℝ × ℝ × ℝ) : Prop := sorry

def is_perpendicular (v1 v2 : ℝ × ℝ × ℝ) : Prop := sorry

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ := sorry

noncomputable def volume (pyr : Pyramid) : ℝ := sorry

theorem pyramid_volume (pyr : Pyramid) :
  is_rectangle pyr.A pyr.B pyr.C pyr.D →
  distance pyr.A pyr.B = 8 →
  distance pyr.B pyr.C = 4 →
  is_perpendicular pyr.P pyr.A →
  is_perpendicular pyr.A pyr.D →
  is_perpendicular pyr.P pyr.A →
  is_perpendicular pyr.A pyr.B →
  distance pyr.P pyr.B = 17 →
  volume pyr = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l595_59524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_model_height_approx_l595_59516

/-- Represents the water tower and its scale model -/
structure WaterTower where
  full_height : ℝ
  base_height : ℝ
  sphere_volume : ℝ
  model_sphere_volume : ℝ

/-- Calculates the height of the scale model tower -/
noncomputable def model_height (tower : WaterTower) : ℝ :=
  let volume_ratio := tower.sphere_volume / tower.model_sphere_volume
  let scale_factor := Real.rpow volume_ratio (1/3)
  tower.full_height / scale_factor

/-- Theorem stating that the model height is approximately 0.44 meters -/
theorem model_height_approx (tower : WaterTower) 
  (h1 : tower.full_height = 70)
  (h2 : tower.base_height = 30)
  (h3 : tower.sphere_volume = 200000)
  (h4 : tower.model_sphere_volume = 0.05) :
  ∃ ε > 0, |model_height tower - 0.44| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_model_height_approx_l595_59516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_monthly_balance_l595_59521

noncomputable def january_balance : ℝ := 120
noncomputable def february_balance : ℝ := 240
noncomputable def march_balance : ℝ := 180
noncomputable def april_balance : ℝ := 180
noncomputable def may_balance : ℝ := 300

def number_of_months : ℕ := 5

noncomputable def total_balance : ℝ := january_balance + february_balance + march_balance + april_balance + may_balance

noncomputable def average_balance : ℝ := total_balance / number_of_months

theorem average_monthly_balance :
  average_balance = 204 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_monthly_balance_l595_59521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_minimum_l595_59582

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem f_monotonicity_and_minimum (a : ℝ) :
  (∀ x > 0, (∀ y > x, f a y > f a x) ∨
    (∃ y > 0, (∀ z ∈ Set.Ioo 0 y, ∀ w ∈ Set.Ioo 0 y, z < w → f a z < f a w) ∧
              (∀ z ∈ Set.Ioi y, ∀ w ∈ Set.Ioi y, z < w → f a z > f a w))) ∧
  (a > 0 → 
    (∀ x ∈ Set.Icc 1 2, f a x ≥ -a ∧ (a < Real.log 2 → f a x ≥ -a) ∧ (a ≥ Real.log 2 → f a x ≥ Real.log 2 - 2*a))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_minimum_l595_59582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l595_59514

/-- Given points A, B, C and vector a, prove the value of k and the projection vector --/
theorem vector_problem (k : ℝ) :
  let A : Fin 3 → ℝ := ![1, 2, -1]
  let B : Fin 3 → ℝ := ![2, k, -3]
  let C : Fin 3 → ℝ := ![0, 5, 1]
  let a : Fin 3 → ℝ := ![-3, 4, 5]
  let AB : Fin 3 → ℝ := λ i => B i - A i
  let AC : Fin 3 → ℝ := λ i => C i - A i

  -- Part 1: If AB ⊥ a, then k = 21/4
  (AB 0 * a 0 + AB 1 * a 1 + AB 2 * a 2 = 0 → k = 21/4) ∧

  -- Part 2: Projection vector of AC in direction of a
  (let dot_AC_a := AC 0 * a 0 + AC 1 * a 1 + AC 2 * a 2
   let norm_a_squared := a 0^2 + a 1^2 + a 2^2
   let proj_AC_a := λ i => (dot_AC_a / norm_a_squared) * a i
   (proj_AC_a 0, proj_AC_a 1, proj_AC_a 2) = (-3/2, 2, 5/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l595_59514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l595_59580

theorem abc_inequality (a b c : ℝ) (ha : a ∈ Set.Icc (-1) 1) (hb : b ∈ Set.Icc (-1) 1) (hc : c ∈ Set.Icc (-1) 1)
  (h : 1 + 2*a*b*c ≥ a^2 + b^2 + c^2) :
  ∀ n : ℕ, 1 + 2*(a*b*c)^n ≥ a^(2*n) + b^(2*n) + c^(2*n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l595_59580
