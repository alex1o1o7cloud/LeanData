import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_range_l881_88138

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x + 1 / x

theorem monotonicity_and_range :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f (1/2) x₁ > f (1/2) x₂) ∧
  (∀ a : ℝ, (∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x ≥ 6) → a ≥ 9/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_range_l881_88138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disco_ball_price_theorem_l881_88148

/-- Represents the price of a disco ball in dollars -/
def disco_ball_price : ℝ → Prop := λ x => True

/-- Represents Calum's budget constraint -/
def budget_constraint (x : ℝ) : Prop :=
  4 * x + 10 * (0.85 * x) + 20 * (x / 2 - 10) = 600

/-- Represents the relationship between food price and disco ball price -/
def food_price_relation (x : ℝ) : Prop :=
  ∃ y : ℝ, y = 0.85 * x

/-- Represents the relationship between decoration price and disco ball price -/
def decoration_price_relation (x : ℝ) : Prop :=
  ∃ z : ℝ, z = x / 2 - 10

/-- Theorem stating that the price of each disco ball is 800/22.5 dollars -/
theorem disco_ball_price_theorem :
  ∀ x : ℝ, disco_ball_price x →
    budget_constraint x →
    food_price_relation x →
    decoration_price_relation x →
    x = 800 / 22.5 := by
  sorry

#check disco_ball_price_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disco_ball_price_theorem_l881_88148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sine_l881_88162

theorem arithmetic_sequence_sine (a : Real) : 
  0 < a ∧ a < 2 * Real.pi →
  (∃ r : Real, Real.sin a + r = Real.sin (2 * a) ∧ Real.sin (2 * a) + r = Real.sin (3 * a)) ↔ 
  (a = Real.pi / 2 ∨ a = 3 * Real.pi / 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sine_l881_88162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_percentage_l881_88190

theorem salary_percentage (total_salary n_salary : ℝ) 
  (h1 : total_salary = 605) (h2 : n_salary = 275) :
  (((total_salary - n_salary) / n_salary) * 100) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_percentage_l881_88190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l881_88196

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Given conditions for the problem -/
def ProblemConditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  Real.sqrt 3 * t.b * Real.sin t.A = t.a * (1 + Real.cos t.B) ∧
  t.a = 1 ∧
  t.b = Real.sqrt 3

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) (h : ProblemConditions t) :
  t.B = Real.pi / 3 ∧ 
  (∃ (x y : Real), x > 0 ∧ y > 0 ∧ 
    (∀ (x' y' : Real), x' > 0 → y' > 0 → 
      x' * y' * Real.sqrt 3 / 4 ≤ x * y * Real.sqrt 3 / 4) ∧
    x * y * Real.sqrt 3 / 4 = Real.sqrt 3 / 4) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l881_88196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_M_equation_l881_88192

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define point P_0
def P_0 : ℝ × ℝ := (-1, 2)

-- Define point C
def C : ℝ × ℝ := (3, 0)

-- Define the chord AB with angle 135°
def chord_AB (x y : ℝ) : Prop := y = -x + 1

-- Theorem for the length of chord AB
theorem chord_length : 
  ∃ (A B : ℝ × ℝ), 
    chord_AB A.1 A.2 ∧ 
    chord_AB B.1 B.2 ∧ 
    P_0 ∈ Set.Icc A B ∧
    dist A B = Real.sqrt 30 := 
by sorry

-- Theorem for the equation of circle M
theorem circle_M_equation : 
  ∃ (A B : ℝ × ℝ) (M : ℝ × ℝ) (R : ℝ),
    chord_AB A.1 A.2 ∧
    chord_AB B.1 B.2 ∧
    P_0 ∈ Set.Icc A B ∧
    dist A P_0 = dist P_0 B ∧
    dist M P_0 = R ∧
    dist M C = R ∧
    (∀ (x y : ℝ), (x - M.1)^2 + (y - M.2)^2 = R^2 ↔ (x - 1/4)^2 + (y + 1/2)^2 = 125/16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_M_equation_l881_88192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_no_constraint_max_area_even_constraint_l881_88146

/-- Represents the length of the available wire for fencing -/
def wireLength : ℝ := 44

/-- Calculates the area of the rectangular fence given the length of one side -/
def fenceArea (side : ℝ) : ℝ :=
  side * (wireLength - 2 * side)

/-- Predicate to check if a real number is even -/
def isEven (x : ℝ) : Prop :=
  ∃ (n : ℤ), x = 2 * ↑n

/-- Theorem for the maximum area without constraints -/
theorem max_area_no_constraint :
  ∃ (side : ℝ), fenceArea side = 242 ∧ 
  ∀ (x : ℝ), fenceArea x ≤ fenceArea side :=
sorry

/-- Theorem for the maximum area with even side length constraint -/
theorem max_area_even_constraint :
  ∃ (side : ℝ), isEven side ∧ fenceArea side = 240 ∧
  ∀ (x : ℝ), isEven x → fenceArea x ≤ fenceArea side :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_no_constraint_max_area_even_constraint_l881_88146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_five_digit_square_cube_l881_88169

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, m < n →
    ¬((m ≥ 10000 ∧ m < 100000) ∧
      (∃ a : ℕ, m = a^2) ∧
      (∃ b : ℕ, m = b^3))) ∧
  n = 15625 := by
  -- Proof goes here
  sorry

#check least_five_digit_square_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_five_digit_square_cube_l881_88169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l881_88173

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x + y - 6 = 0

-- Define the distance function from a point (x, y) to line l
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (2*x + y - 6) / Real.sqrt 5

theorem distance_bounds :
  ∃ (d_max d_min : ℝ),
    d_max = 11 * Real.sqrt 5 / 5 ∧
    d_min = Real.sqrt 5 / 5 ∧
    (∀ x y : ℝ, curve_C x y →
      d_min ≤ distance_to_line x y ∧ distance_to_line x y ≤ d_max) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l881_88173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l881_88109

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = π) :
  (Real.sqrt (Real.sin A * Real.sin B) / Real.sin (C / 2)) + 
  (Real.sqrt (Real.sin B * Real.sin C) / Real.sin (A / 2)) + 
  (Real.sqrt (Real.sin C * Real.sin A) / Real.sin (B / 2)) ≥ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l881_88109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_formula_l881_88198

/-- The area of a square inscribed in an equilateral triangle with side length a -/
noncomputable def inscribed_square_area (a : ℝ) : ℝ := 3 * a^2 * (7 - 4 * Real.sqrt 3)

/-- Theorem: The area of a square inscribed in an equilateral triangle with side length a
    is equal to 3a²(7 - 4√3) -/
theorem inscribed_square_area_formula (a : ℝ) (h : a > 0) :
  ∃ (s : ℝ), s > 0 ∧ s^2 = inscribed_square_area a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_formula_l881_88198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_and_p_relation_l881_88106

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 8*x^2 + 10*x - 1

-- Define the roots
variable (a b c : ℝ)

-- Define p
noncomputable def p (a b c : ℝ) : ℝ := Real.sqrt a + Real.sqrt b + Real.sqrt c

-- State the theorem
theorem roots_and_p_relation 
  (ha : f a = 0) 
  (hb : f b = 0) 
  (hc : f c = 0) :
  p a b c^4 - 16*(p a b c)^2 - 8*(p a b c) = -24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_and_p_relation_l881_88106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_tangency_l881_88103

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 6 - y^2 / 3 = 1

/-- The circle equation -/
def circle_eq (x y r : ℝ) : Prop := (x - 3)^2 + y^2 = r^2

/-- The asymptotes of the hyperbola are tangent to the circle -/
def asymptotes_tangent_to_circle (r : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola x y ∧ circle_eq x y r

theorem hyperbola_circle_tangency (r : ℝ) (hr : r > 0) :
  asymptotes_tangent_to_circle r → r = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_tangency_l881_88103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_arctg_sqrt_6x_minus_1_l881_88108

open Real

theorem integral_of_arctg_sqrt_6x_minus_1 (x : ℝ) :
  HasDerivAt (λ x ↦ x * arctan (sqrt (6 * x - 1)) - (1 / 6) * sqrt (6 * x - 1))
              (arctan (sqrt (6 * x - 1)))
              x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_arctg_sqrt_6x_minus_1_l881_88108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l881_88193

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2 + 1

-- Theorem for part I
theorem part_one (a : ℝ) (h : a ≠ 0) :
  (∀ x > 0, f a x ≤ 0) ↔ a = 2 :=
sorry

-- Theorem for part II
theorem part_two (a : ℝ) (h : a ≤ -1/8) :
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → |f a x₁ - f a x₂| ≥ |x₁ - x₂| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l881_88193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hospital_staff_problem_l881_88194

/-- Given a total number of staff and a ratio of staff categories, 
    calculate the number of staff in a specific category -/
def calculate_staff_in_category (total_staff : ℕ) (ratio : List ℕ) (category_index : ℕ) : ℕ :=
  let total_parts := ratio.sum
  let staff_per_part := total_staff / total_parts
  match ratio.get? category_index with
  | some n => n * staff_per_part
  | none => 0

/-- The problem statement -/
theorem hospital_staff_problem (total_staff : ℕ) (ratio : List ℕ) :
  total_staff = 1250 →
  ratio = [4, 7, 3, 6] →
  calculate_staff_in_category total_staff ratio 1 = 437 := by
  sorry

#eval calculate_staff_in_category 1250 [4, 7, 3, 6] 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hospital_staff_problem_l881_88194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_sqrt_three_simplest_l881_88117

/-- A quadratic radical is simplest if it cannot be simplified further -/
noncomputable def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ^ 2 = x → y = x ∨ y = -x

/-- The given options for quadratic radicals -/
noncomputable def options : List ℝ := [-Real.sqrt 3, Real.sqrt (1/2), Real.sqrt 0.1, Real.sqrt 8]

/-- Theorem stating that -√3 is the simplest quadratic radical among the given options -/
theorem negative_sqrt_three_simplest :
  ∃ x ∈ options, is_simplest_quadratic_radical x ∧
  ∀ y ∈ options, is_simplest_quadratic_radical y → y = x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_sqrt_three_simplest_l881_88117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_percentage_yield_l881_88159

/-- Calculates the percentage yield of a stock given its yield and price -/
noncomputable def percentage_yield (yield : ℝ) (price : ℝ) : ℝ :=
  (yield * price) / price * 100

/-- Theorem: The percentage yield of a stock with 12% yield and price 125.00000000000001 is 12% -/
theorem stock_percentage_yield :
  let yield : ℝ := 0.12
  let price : ℝ := 125.00000000000001
  percentage_yield yield price = 12 := by
  -- Unfold the definition of percentage_yield
  unfold percentage_yield
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_percentage_yield_l881_88159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_7_l881_88156

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Theorem statement
theorem floor_of_2_7 : floor 2.7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_7_l881_88156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l881_88183

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  Real.cos C = 3/5 →
  a * b * Real.cos C = 9 →
  a + b = 8 →
  Real.sin (C + π/3) = (4 + 3 * Real.sqrt 3) / 10 ∧
  c = Real.sqrt 17 ∧
  (1/2) * a * b * Real.sin C = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l881_88183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_polar_curve_l881_88160

/-- Predicate to check if two polar curves are symmetric with respect to the polar axis -/
def is_symmetric_wrt_polar_axis (ρ₁ ρ₂ : ℝ → ℝ) : Prop :=
  ∀ θ : ℝ, ρ₁ θ = ρ₂ (-θ)

/-- Given a curve in polar coordinates ρ = cos θ + sin θ, 
    this theorem states that its symmetric curve with respect to the polar axis 
    has the equation ρ = cos θ - sin θ. -/
theorem symmetric_polar_curve 
  (ρ₁ : ℝ → ℝ) (h : ∀ θ, ρ₁ θ = Real.cos θ + Real.sin θ) :
  ∃ ρ₂ : ℝ → ℝ, (∀ θ, ρ₂ θ = Real.cos θ - Real.sin θ) ∧ 
  is_symmetric_wrt_polar_axis ρ₁ ρ₂ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_polar_curve_l881_88160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l881_88166

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := (3 * x - 4) / (x + 2)

-- State the theorem about the range of g
theorem range_of_g :
  Set.range g = {y : ℝ | y < 3 ∨ y > 3} :=
by
  sorry  -- Placeholder for the proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l881_88166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_theta_l881_88100

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + θ)

noncomputable def g (θ : ℝ) (x : ℝ) : ℝ := f θ (x + Real.pi/8)

def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

theorem smallest_positive_theta :
  ∃! θ : ℝ, θ > 0 ∧ θ ≤ Real.pi/2 ∧ is_even (g θ) ∧ θ = Real.pi/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_theta_l881_88100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_a_81_l881_88139

theorem divisibility_of_a_81 (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2) : 
  ∃ a : ℕ → ℤ, a 1 = 5 ∧ 
  (∀ n : ℕ, n > 0 → (n : ℚ) * a (n + 1) = ((n + 1) : ℚ) * a n - (p / 2 : ℚ)^4) ∧
  16 ∣ a 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_a_81_l881_88139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisectors_coplanar_l881_88119

-- Define a trihedral angle
structure TrihedralAngle where
  S : EuclideanSpace ℝ (Fin 3)
  A : EuclideanSpace ℝ (Fin 3)
  B : EuclideanSpace ℝ (Fin 3)
  C : EuclideanSpace ℝ (Fin 3)
  edges_from_S : S ≠ A ∧ S ≠ B ∧ S ≠ C
  equal_lengths : norm (S - A) = norm (S - B) ∧ norm (S - B) = norm (S - C)

-- Define bisector of a plane angle
noncomputable def bisector (p q r : EuclideanSpace ℝ (Fin 3)) : Set (EuclideanSpace ℝ (Fin 3)) :=
  sorry

-- Define the angle adjacent to a plane angle in a trihedral angle
noncomputable def adjacent_angle (p q r : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  sorry

-- Theorem statement
theorem bisectors_coplanar (t : TrihedralAngle) :
  ∃ (plane : Set (EuclideanSpace ℝ (Fin 3))),
    bisector t.S t.A t.B ⊆ plane ∧
    bisector t.S t.B t.C ⊆ plane ∧
    bisector t.S t.C t.A ⊆ plane :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisectors_coplanar_l881_88119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_tan_2x_plus_pi_over_3_l881_88136

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 3)

theorem domain_of_tan_2x_plus_pi_over_3 :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∀ k : ℤ, x ≠ k * Real.pi / 2 + Real.pi / 12} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_tan_2x_plus_pi_over_3_l881_88136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fridays_in_september_l881_88197

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving Repr, DecidableEq

/-- Represents a month -/
structure Month where
  days : Nat
  first_day : DayOfWeek

/-- Given a day of the week, returns the next day -/
def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Counts the occurrences of a specific day in a month -/
def count_day_occurrences (m : Month) (d : DayOfWeek) : Nat :=
  let rec count_aux (current_day : DayOfWeek) (days_left : Nat) (acc : Nat) : Nat :=
    if days_left = 0 then
      acc
    else if current_day = d then
      count_aux (next_day current_day) (days_left - 1) (acc + 1)
    else
      count_aux (next_day current_day) (days_left - 1) acc
  count_aux m.first_day m.days 0

theorem five_fridays_in_september
  (july : Month)
  (september : Month)
  (h1 : july.days = 31)
  (h2 : count_day_occurrences july DayOfWeek.Wednesday = 5)
  (h3 : september.days = 30) :
  count_day_occurrences september DayOfWeek.Friday = 5 :=
by
  sorry

#eval count_day_occurrences { days := 31, first_day := DayOfWeek.Friday } DayOfWeek.Friday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fridays_in_september_l881_88197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l881_88195

/-- The function f(x) = x^2 - 2ax + 1 -/
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

/-- The maximum value of f(x) on [-1, 2] -/
noncomputable def max_value (a : ℝ) : ℝ := max (5 - 4*a) (2 + 2*a)

/-- The minimum value of f(x) on [-1, 2] -/
noncomputable def min_value (a : ℝ) : ℝ := min (2 + 2*a) (min (1 - a^2) (5 - 4*a))

theorem f_extrema (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f a x ≤ max_value a) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a x = max_value a) ∧
  (∀ x ∈ Set.Icc (-1) 2, min_value a ≤ f a x) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a x = min_value a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l881_88195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_and_x_values_l881_88120

noncomputable def t_neg (a : ℝ) : ℝ := (1 - Real.sqrt (1 - 8*a + 12*a^2)) / (2*a)

noncomputable def t_pos (a : ℝ) : ℝ := (1 + Real.sqrt (1 - 8*a + 12*a^2)) / (2*a)

noncomputable def x_neg (a : ℝ) : ℝ := (t_neg a)^2

noncomputable def x_pos (a : ℝ) : ℝ := (t_pos a)^2

theorem root_and_x_values (a : ℝ) :
  (a < 0 → t_neg a ≥ 0 ∧ x_neg a = ((1 - Real.sqrt (1 - 8*a + 12*a^2)) / (2*a))^2) ∧
  ((a = 1/6 ∨ a = 1/2 ∨ a > 2/3) → 
    t_pos a ≥ 0 ∧ x_pos a = ((1 + Real.sqrt (1 - 8*a + 12*a^2)) / (2*a))^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_and_x_values_l881_88120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l881_88132

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem inverse_function_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = -1) :
  ∀ x : ℝ, (f a)⁻¹ x = (1/2:ℝ)^x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l881_88132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_finish_order_l881_88149

/-- Race parameters and runner strategies -/
structure RaceParams where
  d : ℝ  -- Half of the total race distance
  arnaldo_speed1 : ℝ := 9
  arnaldo_speed2 : ℝ := 11
  braulio_speed1 : ℝ := 9
  braulio_speed2 : ℝ := 10
  braulio_speed3 : ℝ := 11
  carlos_speed1 : ℝ := 9
  carlos_speed2 : ℝ := 11

/-- Calculate finish times for each runner -/
noncomputable def finishTimes (p : RaceParams) : ℝ × ℝ × ℝ :=
  let arnaldo_time := p.d / p.arnaldo_speed1 + p.d / p.arnaldo_speed2
  let braulio_time := (2*p.d/3) / p.braulio_speed1 + (2*p.d/3) / p.braulio_speed2 + (2*p.d/3) / p.braulio_speed3
  let carlos_time := 6 * p.d / (p.carlos_speed1 + p.carlos_speed2)
  (arnaldo_time, braulio_time, carlos_time)

/-- Theorem stating the finishing order of the runners -/
theorem race_finish_order (p : RaceParams) :
  let (tA, tB, tC) := finishTimes p
  tC < tB ∧ tB < tA := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_finish_order_l881_88149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_at_6_l881_88188

/-- A monic quintic polynomial with specific values at x = 1, 2, 3, 4, and 5 -/
noncomputable def p : ℝ → ℝ := sorry

/-- p is a monic quintic polynomial -/
axiom p_monic_quintic : ∃ a b c d e : ℝ, ∀ x, p x = x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- Values of p at specific points -/
axiom p_values : p 1 = 3 ∧ p 2 = 7 ∧ p 3 = 13 ∧ p 4 = 21 ∧ p 5 = 31

/-- Theorem: The value of p at x = 6 is 158 -/
theorem p_at_6 : p 6 = 158 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_at_6_l881_88188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_refrigerator_transport_cost_l881_88180

/-- Calculates the transport cost for a refrigerator purchase --/
theorem refrigerator_transport_cost 
  (purchase_price : ℝ) 
  (discount_percentage : ℝ) 
  (installation_cost : ℝ) 
  (selling_price : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : purchase_price = 12500)
  (h2 : discount_percentage = 0.20)
  (h3 : installation_cost = 250)
  (h4 : selling_price = 18400)
  (h5 : profit_percentage = 0.15) : 
  ∃ (transport_cost : ℝ), abs (transport_cost - 431.25) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_refrigerator_transport_cost_l881_88180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l881_88155

noncomputable def f (x φ : Real) : Real := Real.sqrt 3 * Real.sin (2 * x + φ) + Real.cos (2 * x + φ)

noncomputable def g (x : Real) : Real := f (x + Real.pi/4) (Real.pi/6)

theorem min_value_of_g :
  (∀ x, f x (Real.pi/6) = f (-Real.pi/3 - x) (Real.pi/6)) →
  (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/6), g x ≥ -1) ∧
  (∃ x ∈ Set.Icc (-Real.pi/4) (Real.pi/6), g x = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l881_88155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_coordinates_min_length_AB_vertical_line_AB_l881_88130

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - focus.1) + focus.2

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem for part 1
theorem point_A_coordinates (x y : ℝ) :
  parabola x y →
  distance x y focus.1 focus.2 = 4 →
  ((x = 3 ∧ y = 2 * Real.sqrt 3) ∨ (x = 3 ∧ y = -2 * Real.sqrt 3)) :=
by sorry

-- Theorem for part 2
theorem min_length_AB (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  parabola x₁ y₁ →
  parabola x₂ y₂ →
  line_through_focus k x₁ y₁ →
  line_through_focus k x₂ y₂ →
  x₁ ≠ x₂ →
  distance x₁ y₁ x₂ y₂ ≥ 4 :=
by sorry

-- Theorem for the case when the line is vertical (x = 1)
theorem vertical_line_AB :
  parabola 1 2 ∧ parabola 1 (-2) ∧ distance 1 2 1 (-2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_coordinates_min_length_AB_vertical_line_AB_l881_88130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gp_even_odd_sum_ratio_l881_88129

/-- Represents a geometric progression with 2n terms -/
structure GeometricProgression (α : Type*) [Field α] where
  n : ℕ
  a : α
  r : α

/-- Sum of even-indexed terms in a geometric progression -/
def sumEvenTerms {α : Type*} [Field α] (gp : GeometricProgression α) : α :=
  gp.a * gp.r * (1 - gp.r^(2 * gp.n)) / (1 - gp.r^2)

/-- Sum of odd-indexed terms in a geometric progression -/
def sumOddTerms {α : Type*} [Field α] (gp : GeometricProgression α) : α :=
  gp.a * (1 - gp.r^(2 * gp.n)) / (1 - gp.r^2)

/-- Theorem: The ratio of the sum of even-indexed terms to the sum of odd-indexed terms
    in a geometric progression is equal to its common ratio -/
theorem gp_even_odd_sum_ratio {α : Type*} [Field α] (gp : GeometricProgression α) :
  sumEvenTerms gp / sumOddTerms gp = gp.r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gp_even_odd_sum_ratio_l881_88129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_product_l881_88105

/-- Predicate to check if a list of real numbers forms an arithmetic sequence -/
def is_arithmetic_sequence (l : List ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Nat, i + 1 < l.length → l[i+1]! - l[i]! = d

/-- Predicate to check if a list of real numbers forms a geometric sequence -/
def is_geometric_sequence (l : List ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ i : Nat, i + 1 < l.length → l[i+1]! / l[i]! = r

/-- Given an arithmetic sequence (-1, a₁, a₂, -9) and a geometric sequence (-9, b₁, b₂, b₃, -1),
    prove that b₂(a₂-a₁) = 8 -/
theorem arithmetic_geometric_sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) :
  (is_arithmetic_sequence [-1, a₁, a₂, -9]) →
  (is_geometric_sequence [-9, b₁, b₂, b₃, -1]) →
  b₂ * (a₂ - a₁) = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_product_l881_88105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_m_range_l881_88172

/-- The function f(x) = mx^2 + x + m + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x + m + 2

/-- The interval (-∞, 2) -/
def interval : Set ℝ := Set.Iio 2

/-- The function is increasing on the interval -/
def is_increasing (m : ℝ) : Prop :=
  ∀ x y, x ∈ interval → y ∈ interval → x < y → f m x < f m y

/-- The range of m values for which f is increasing on the interval -/
def m_range : Set ℝ := Set.Icc (-1/4) 0

/-- Theorem: The range of m values for which f is increasing on (-∞, 2) is [-1/4, 0] -/
theorem f_increasing_m_range :
  ∀ m : ℝ, is_increasing m ↔ m ∈ m_range := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_m_range_l881_88172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_distance_difference_is_ten_l881_88123

/-- Represents the tour bus problem --/
structure TourBus where
  total_time : ℚ  -- Total tour time in hours
  destination_time : ℚ  -- Time spent at destination in hours
  destination_distance : ℚ  -- Distance to destination in miles
  speed : ℚ  -- Driving speed in miles per minute

/-- The difference between the return distance and the distance to the destination --/
def return_distance_difference (t : TourBus) : ℚ :=
  ((t.total_time - t.destination_time) * 60 / (1 / t.speed)) / 2 - t.destination_distance

/-- Theorem stating that the return distance difference is 10 miles --/
theorem return_distance_difference_is_ten :
  let t : TourBus := {
    total_time := 6,
    destination_time := 2,
    destination_distance := 55,
    speed := 1 / 2
  }
  return_distance_difference t = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_distance_difference_is_ten_l881_88123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l881_88199

theorem product_remainder (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l881_88199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_theorem_l881_88124

-- Define the point A
noncomputable def point_A (a : ℝ) : ℝ × ℝ := (-2*a, -a)

-- Define the point B
noncomputable def point_B (a : ℝ) : ℝ × ℝ := (a + 3/a, 1)

-- Define the condition for A and B being on opposite sides of x = 4
def opposite_sides (a : ℝ) : Prop :=
  (point_A a).1 < 4 ∧ (point_B a).1 > 4 ∨ (point_A a).1 > 4 ∧ (point_B a).1 < 4

-- Define the set of valid a values
def valid_a_set : Set ℝ := {a | a < -2 ∨ (0 < a ∧ a < 1) ∨ a > 3}

-- Theorem statement
theorem opposite_sides_theorem :
  ∀ a : ℝ, a ≠ 0 → (opposite_sides a ↔ a ∈ valid_a_set) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_theorem_l881_88124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_circumscribed_circle_area_l881_88170

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units -/
noncomputable def circumscribedCircleArea (sideLength : ℝ) : ℝ :=
  let height := (Real.sqrt 3 / 2) * sideLength
  let radius := (2 / 3) * height
  Real.pi * radius^2

theorem equilateral_triangle_circumscribed_circle_area :
  circumscribedCircleArea 12 = 48 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_circumscribed_circle_area_l881_88170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_squares_l881_88114

/-- The parabola y² = 4x with focus F(1, 0) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ^ 2 = 4 * p.1}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- A line passing through the focus -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - 1)}

/-- Intersection points of the line and the parabola -/
def IntersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  Parabola ∩ Line k

theorem parabola_line_intersection_sum_squares :
  ∀ k : ℝ, ∀ A B : ℝ × ℝ, A ∈ IntersectionPoints k → B ∈ IntersectionPoints k →
  A.2 ^ 2 + B.2 ^ 2 ≥ 8 ∧
  ∃ k₀ : ℝ, ∃ A₀ B₀ : ℝ × ℝ, A₀ ∈ IntersectionPoints k₀ ∧ B₀ ∈ IntersectionPoints k₀ ∧
  A₀.2 ^ 2 + B₀.2 ^ 2 = 8 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_squares_l881_88114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_when_a_is_2_range_of_a_for_necessary_not_sufficient_l881_88184

-- Define the conditions p and q
def p (a : ℝ) (x : ℝ) : Prop := x - a < 0
def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Part 1
theorem range_of_x_when_a_is_2 :
  {x : ℝ | p 2 x ∧ q x} = Set.Icc 1 2 := by sorry

-- Part 2
theorem range_of_a_for_necessary_not_sufficient :
  {a : ℝ | {x : ℝ | p a x} ⊃ {x : ℝ | q x} ∧ {x : ℝ | p a x} ≠ {x : ℝ | q x}} = Set.Ioi 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_when_a_is_2_range_of_a_for_necessary_not_sufficient_l881_88184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_properties_l881_88121

/-- The answer to the problem -/
noncomputable def answer : ℝ := 1 / Real.pi

/-- Area of an equilateral triangle with side length 2 -/
noncomputable def triangle_area : ℝ := Real.sqrt 3

/-- Radius of a circle with circumference 2 -/
noncomputable def circle_radius : ℝ := 1 / Real.pi

/-- Diagonal of a square with side length 2 -/
noncomputable def square_diagonal : ℝ := 2 * Real.sqrt 2

/-- Theorem stating the properties of the answer -/
theorem answer_properties :
  (Irrational answer ∨ answer = triangle_area) ∧
  (∃ n : ℤ, answer = 4 * n ∨ answer = circle_radius) ∧
  (answer < 3 ∨ answer = square_diagonal) ∧
  (Irrational answer) ∧
  (answer = circle_radius) ∧
  (answer < 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_properties_l881_88121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_proportions_l881_88185

theorem statue_proportions (h : ℝ) (l : ℝ) : 
  h = 2 →
  (h - l) / l = l / h →
  l = Real.sqrt 5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_proportions_l881_88185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l881_88158

def sequenceA (n : ℕ) : ℝ :=
  if n % 2 = 1 then n else 2^(n/2)

def isArithmeticSequence (f : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, f (2*n+3) - f (2*n+1) = d

def isGeometricSequence (f : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, f (2*n+2) / f (2*n) = r

theorem sequence_properties (a : ℕ → ℝ) :
  isArithmeticSequence a 2 →
  isGeometricSequence a 2 →
  a 2 + a 4 = a 1 + a 5 →
  a 7 + a 9 = a 8 →
  (∀ n, a n = sequenceA n) ∧
  (∀ m : ℕ, m > 0 → (a m * a (m+1) * a (m+2) = a m + a (m+1) + a (m+2) ↔ m = 1)) :=
by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l881_88158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_triplets_characterization_l881_88174

def is_valid_triplet (a b c : ℕ+) : Prop :=
  a ≥ b ∧ b ≥ c ∧ (1 + 1 / (a : ℝ)) * (1 + 1 / (b : ℝ)) * (1 + 1 / (c : ℝ)) = 2

def solution_set : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(⟨7, by norm_num⟩, ⟨6, by norm_num⟩, ⟨2, by norm_num⟩),
   (⟨9, by norm_num⟩, ⟨5, by norm_num⟩, ⟨2, by norm_num⟩),
   (⟨15, by norm_num⟩, ⟨4, by norm_num⟩, ⟨2, by norm_num⟩),
   (⟨8, by norm_num⟩, ⟨3, by norm_num⟩, ⟨3, by norm_num⟩),
   (⟨5, by norm_num⟩, ⟨4, by norm_num⟩, ⟨3, by norm_num⟩)}

theorem valid_triplets_characterization :
  ∀ a b c : ℕ+, is_valid_triplet a b c ↔ (a, b, c) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_triplets_characterization_l881_88174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l881_88187

noncomputable section

-- Define the hyperbola C1
def C1 (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola C2
def C2 (x y p : ℝ) : Prop := x^2 = 2 * p * y

-- Define the eccentricity of C1
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Define the distance from the focus of C2 to the asymptotes of C1
noncomputable def focus_to_asymptote_distance (a b p : ℝ) : ℝ :=
  (p / (2 * b)) / Real.sqrt ((1 / a^2) + (1 / b^2))

theorem parabola_equation (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  (eccentricity a b = 2) →
  (focus_to_asymptote_distance a b p = 2) →
  ∀ x y : ℝ, C2 x y p ↔ x^2 = 16 * y :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l881_88187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l881_88181

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem conditions
def problem_conditions (t : Triangle) : Prop :=
  -- Vector m = (c-2b, a)
  let m : ℝ × ℝ := (t.c - 2*t.b, t.a)
  -- Vector n = (cos A, cos C)
  let n : ℝ × ℝ := (Real.cos t.A, Real.cos t.C)
  -- m ⊥ n
  (m.1 * n.1 + m.2 * n.2 = 0) ∧
  -- AB · AC = 4
  (t.b * t.c = 8)

-- Theorem statement
theorem triangle_problem (t : Triangle) 
  (h : problem_conditions t) : 
  t.A = π/3 ∧ t.a ≥ 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l881_88181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fruit_pickers_xiao_yu_family_total_pickers_l881_88125

-- Define the set of people who picked passion fruits
def passion_fruit_pickers : Finset String := sorry

-- Define the set of people who picked strawberries
def strawberry_pickers : Finset String := sorry

-- Theorem stating the total number of unique fruit pickers
theorem total_fruit_pickers :
  (passion_fruit_pickers ∪ strawberry_pickers).card =
    passion_fruit_pickers.card + strawberry_pickers.card -
    (passion_fruit_pickers ∩ strawberry_pickers).card :=
by
  sorry

-- Define the specific problem instance
def xiao_yu_family_picking : Prop :=
  passion_fruit_pickers.card = 6 ∧
  strawberry_pickers.card = 4 ∧
  (passion_fruit_pickers ∩ strawberry_pickers).card = 2

-- Theorem for Xiao Yu's family fruit picking problem
theorem xiao_yu_family_total_pickers (h : xiao_yu_family_picking) :
  (passion_fruit_pickers ∪ strawberry_pickers).card = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fruit_pickers_xiao_yu_family_total_pickers_l881_88125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_function_equality_l881_88189

/-- The partition function p(n) counts the number of ways to write n as a sum of positive integers. -/
def partition_function : ℕ → ℕ := sorry

/-- The theorem states that the equation p(n) + p(n+4) = p(n+2) + p(n+3) holds if and only if n is 1, 3, or 5. -/
theorem partition_function_equality (n : ℕ) :
  (partition_function n + partition_function (n + 4) = partition_function (n + 2) + partition_function (n + 3)) ↔
  (n = 1 ∨ n = 3 ∨ n = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_function_equality_l881_88189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_unique_candies_l881_88168

/-- Represents the state of candy distribution among children -/
structure CandyDistribution where
  num_children : Nat
  initial_candies : Nat
  candies : List Nat

/-- Represents a move in the candy distribution game -/
inductive Move
  | distribute : Nat → List (Nat × Nat) → Move

/-- The result of applying a sequence of moves to a candy distribution -/
def apply_moves (init : CandyDistribution) (moves : List Move) : CandyDistribution :=
  sorry

/-- Checks if all children have unique numbers of candies -/
def all_unique (dist : CandyDistribution) : Prop :=
  sorry

/-- The main theorem stating the minimum number of moves required -/
theorem min_moves_for_unique_candies :
  ∀ (moves : List Move),
    let init := CandyDistribution.mk 100 100 (List.replicate 100 100)
    let final := apply_moves init moves
    all_unique final →
    moves.length ≥ 30 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_unique_candies_l881_88168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l881_88102

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 - a * x * Real.log x + a * x + 2

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > Real.exp 1)
  (h_extreme : ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    f_derivative a x₁ = 0 ∧ 
    f_derivative a x₂ = 0 ∧
    ∀ (x : ℝ), x ≠ x₁ ∧ x ≠ x₂ → f_derivative a x ≠ 0) :
  x₁ * x₂ < a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l881_88102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l881_88167

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular parallelepiped -/
structure RectangularParallelepiped where
  A : Point3D
  B : Point3D
  C : Point3D
  G : Point3D

/-- Represents a rectangular pyramid -/
structure RectangularPyramid where
  base : Set Point3D
  apex : Point3D

/-- Calculate the volume of a rectangular pyramid -/
noncomputable def volumeOfRectangularPyramid (pyramid : RectangularPyramid) : ℝ :=
  sorry

/-- The centroid of a rectangle -/
noncomputable def centroidOfRectangle (rectangle : Set Point3D) : Point3D :=
  sorry

theorem volume_of_specific_pyramid (parallelepiped : RectangularParallelepiped) 
  (h1 : |parallelepiped.B.x - parallelepiped.A.x| = 4)
  (h2 : |parallelepiped.C.y - parallelepiped.B.y| = 2)
  (h3 : |parallelepiped.G.z - parallelepiped.C.z| = 3) :
  let baseRectangle : Set Point3D := {parallelepiped.B, parallelepiped.C, parallelepiped.G, 
    Point3D.mk parallelepiped.B.x parallelepiped.G.y parallelepiped.B.z}
  let pyramid : RectangularPyramid := ⟨baseRectangle, centroidOfRectangle baseRectangle⟩
  volumeOfRectangularPyramid pyramid = 10 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_pyramid_l881_88167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lens_system_properties_l881_88186

/-- Represents the properties of a lens system with a point light source and its image. -/
structure LensSystem where
  x : ℝ  -- Distance of light source from lens and main optical axis
  y : ℝ  -- Distance of image from main optical axis
  isVirtual : Prop  -- Whether the image is virtual

/-- Calculates the optical power of the lens. -/
noncomputable def opticalPower (l : LensSystem) : ℝ :=
  1 / l.x - 1 / (2 * l.x)

/-- Calculates the distance between the light source and its image. -/
noncomputable def sourceImageDistance (l : LensSystem) : ℝ :=
  Real.sqrt ((l.x - l.y)^2 + (l.x - 2*l.x)^2)

/-- Theorem stating the properties of the specific lens system. -/
theorem lens_system_properties (l : LensSystem) 
    (hx : l.x = 10) (hy : l.y = 20) (hv : l.isVirtual) : 
    opticalPower l = 5 ∧ sourceImageDistance l = Real.sqrt 200 := by
  sorry

#eval "Lens system properties theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lens_system_properties_l881_88186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_production_l881_88161

/-- The amount of oranges used for orange juice production --/
def oranges_for_juice (total_production : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) : ℝ :=
  total_production * (1 - export_percentage) * juice_percentage

/-- Rounds a real number to the nearest tenth --/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem orange_juice_production : 
  let total_production : ℝ := 8
  let export_percentage : ℝ := 0.3
  let juice_percentage : ℝ := 0.6
  round_to_tenth (oranges_for_juice total_production export_percentage juice_percentage) = 3.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_production_l881_88161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_l881_88141

-- Define α as a real number representing an angle in radians
variable (α : ℝ)

-- Define the condition that α is an acute angle (0 < α < π/2)
def is_acute_angle (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define the main theorem
theorem trigonometric_equality 
  (h1 : Real.cos (15 * Real.pi / 180 + α) = 3/5) 
  (h2 : is_acute_angle α) : 
  (Real.tan (435 * Real.pi / 180 - α) + Real.sin (α - 165 * Real.pi / 180)) / 
  (Real.cos (195 * Real.pi / 180 + α) * Real.sin (105 * Real.pi / 180 + α)) = 5/36 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_l881_88141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_pair_in_large_subset_l881_88165

-- Define the set of numbers
def S : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 50) (Finset.range 51)

-- Define the property of a subset having two numbers that differ by 1
def has_consecutive_pair (A : Finset ℕ) : Prop :=
  ∃ x ∈ A, x + 1 ∈ A

-- State the theorem
theorem consecutive_pair_in_large_subset :
  ∀ A : Finset ℕ, A ⊆ S → A.card = 26 → has_consecutive_pair A :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_pair_in_large_subset_l881_88165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_OAB_properties_l881_88116

-- Define the triangle OAB
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 9)
def B : ℝ × ℝ := (6, -3)

-- Define point P
def P : ℝ × ℝ → Prop := fun p => p.1 = 14 ∧ ∃ l : ℝ, (p.1 - O.1, p.2 - O.2) = l • (B.1 - p.1, B.2 - p.2)

-- Define point Q
def Q : ℝ × ℝ → Prop := fun q => 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ q = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2)) ∧
  (q.1 * (A.1 - 14) + q.2 * (A.2 - (-7)) = 0)

-- Define point R
def R : ℝ → ℝ × ℝ := fun t => (4*t, 3*t)

theorem triangle_OAB_properties :
  ∃ (p q : ℝ × ℝ) (l : ℝ),
    P p ∧ Q q ∧
    l = -7/4 ∧ p = (14, -7) ∧ q = (4, 3) ∧
    (∀ t : ℝ, 0 ≤ t → t ≤ 1 → 
      -25/2 ≤ (R t).1 * ((A.1 - (R t).1) + (B.1 - (R t).1)) + 
              (R t).2 * ((A.2 - (R t).2) + (B.2 - (R t).2)) ∧
      (R t).1 * ((A.1 - (R t).1) + (B.1 - (R t).1)) + 
      (R t).2 * ((A.2 - (R t).2) + (B.2 - (R t).2)) ≤ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_OAB_properties_l881_88116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l881_88135

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 --/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  (|a * x₀ + b * y₀ + c|) / Real.sqrt (a^2 + b^2)

/-- The equation of a circle with center (h, k) and radius r --/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_tangent_to_line :
  let center_x : ℝ := 1
  let center_y : ℝ := 2
  let line_a : ℝ := 5
  let line_b : ℝ := -12
  let line_c : ℝ := -7
  let radius := distance_point_to_line center_x center_y line_a line_b line_c
  ∀ x y : ℝ, circle_equation x y center_x center_y radius ↔ 
    (x - center_x)^2 + (y - center_y)^2 = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l881_88135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_intersection_l881_88110

/-- The distance from the base of a cone to a plane parallel to the base,
    such that the area of the circle cut out of the inscribed sphere is
    twice the area of the circle cut out of the cone. -/
theorem cone_sphere_intersection (r m : ℝ) (hr : r > 0) (hm : m > 0) :
  let x := (m * r^2) / (2 * r^2 + m^2)
  ∃ (R : ℝ), R > 0 ∧
    (∀ y z : ℝ,
      y^2 = 2 * z^2 →
      z = r * (m - x) / m →
      y^2 = R^2 - (R - (m - x))^2 →
      R = (r^2 + m^2) / (2 * m)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_intersection_l881_88110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_sum_theorem_l881_88191

noncomputable def initial_set (n : ℕ) : Set ℝ :=
  {x | ∃ i : ℕ, 2 ≤ i ∧ i ≤ 2*n+1 ∧ x = i^3 - i}

noncomputable def operation (a b c : ℝ) : ℝ := (a*b*c) / (a*b + b*c + c*a)

def final_set (S : Set ℝ) : Prop :=
  ∃ a b : ℝ, S = {a, b} ∧ a + b > 16

theorem blackboard_sum_theorem (n : ℕ) (h : n ≥ 2) :
  ∃ S : Set ℝ, (S ⊆ initial_set n ∨
    (∃ T : Set ℝ, T ⊆ initial_set n ∧
      ∃ a b c : ℝ, a ∈ T ∧ b ∈ T ∧ c ∈ T ∧
        S = (T \ {a, b, c}) ∪ {operation a b c})) ∧
  final_set S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_sum_theorem_l881_88191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_worked_five_days_l881_88126

-- Define the work rates and remaining work time
noncomputable def x_rate : ℝ := 1 / 21
noncomputable def y_rate : ℝ := 1 / 15
noncomputable def remaining_time : ℝ := 14.000000000000002

-- Define the function to calculate y's working days
noncomputable def y_working_days : ℝ → ℝ := λ total_work =>
  (total_work - remaining_time * x_rate) / y_rate

-- Theorem statement
theorem y_worked_five_days :
  y_working_days 1 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_worked_five_days_l881_88126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_game_theorem_l881_88164

/-- Represents a division of a segment into three parts -/
structure SegmentDivision (x : ℝ) where
  part1 : ℝ
  part2 : ℝ
  part3 : ℝ
  sum_parts : part1 + part2 + part3 = x
  all_positive : part1 > 0 ∧ part2 > 0 ∧ part3 > 0

/-- Checks if three segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if two triangles can be formed from six segments -/
def can_form_two_triangles (s1 s2 s3 s4 s5 s6 : ℝ) : Prop :=
  ∃ (a b c d e f : ℝ), Finset.toSet {a, b, c, d, e, f} = Finset.toSet {s1, s2, s3, s4, s5, s6} ∧
    can_form_triangle a b c ∧ can_form_triangle d e f

theorem segment_game_theorem (k l : ℝ) (hk : k > 0) (hl : l > 0) :
  (k > l → ∃ (dk : SegmentDivision k), ∀ (dl : SegmentDivision l),
    ¬can_form_two_triangles dk.part1 dk.part2 dk.part3 dl.part1 dl.part2 dl.part3) ∧
  (k ≤ l → ∀ (dk : SegmentDivision k), ∃ (dl : SegmentDivision l),
    can_form_two_triangles dk.part1 dk.part2 dk.part3 dl.part1 dl.part2 dl.part3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_game_theorem_l881_88164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_child_share_5400_234_l881_88112

/-- Calculates the share of the second child given a total amount and a ratio --/
def second_child_share (total : ℚ) (ratio : List ℚ) : ℚ :=
  let sum_ratio := ratio.sum
  let part_value := total / sum_ratio
  if h : ratio.length > 1 then
    part_value * ratio[1]'h
  else
    0

/-- Theorem: Given $5400 and ratio 2:3:4, the second child's share is $1800 --/
theorem second_child_share_5400_234 : 
  second_child_share 5400 [2, 3, 4] = 1800 := by
  sorry

#eval second_child_share 5400 [2, 3, 4]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_child_share_5400_234_l881_88112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_mean_condition_l881_88128

theorem triangle_geometric_mean_condition (A B C : ℝ) (h_triangle : A + B + C = Real.pi) :
  (∃ D : ℝ, 0 < D ∧ D < 1 ∧
    (1 - D)^2 * Real.sin A^2 + D^2 * Real.sin B^2 + 2 * D * (1 - D) * Real.sin A * Real.sin B * Real.cos C =
    (D * Real.sin B + (1 - D) * Real.sin A)^2 * Real.sin (C/2)^2) ↔
  Real.sqrt (Real.sin A * Real.sin B) ≤ Real.sin (C/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_mean_condition_l881_88128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_absolute_difference_l881_88151

theorem minimal_absolute_difference (c d : ℕ) (h : c * d - 4 * c + 5 * d = 245) : 
  ∃ (m : ℕ), (∀ (c' d' : ℕ), c' * d' - 4 * c' + 5 * d' = 245 → 
    m ≤ Int.natAbs (c' - d')) ∧ m = Int.natAbs (c - d) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_absolute_difference_l881_88151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_equals_one_l881_88163

theorem binomial_sum_equals_one (n : ℕ) : 
  (Finset.range (n + 1)).sum (λ k => (-1)^k * (n.choose k) * (2 * n - k : ℤ)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_equals_one_l881_88163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_theorem_binomial_sum_congruence_l881_88171

def binomial_sum (n : ℕ) : ℕ := 2^n - 1

theorem congruence_theorem (a b : ℤ) (m : ℕ+) :
  a ≡ b [ZMOD m] ↔ ∃ k : ℤ, b = a + m * k := by
  sorry

theorem binomial_sum_congruence :
  let a : ℤ := binomial_sum 18
  ∀ b : ℤ, a ≡ b [ZMOD 9] → b = 2016 := by
  sorry

#check binomial_sum_congruence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_theorem_binomial_sum_congruence_l881_88171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l881_88153

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

theorem unique_solution :
  ∃! x : ℝ, 4 * x^2 - 5 * (floor x) + 8 * (frac x) = 19 ∧ 
  x = (floor x) + (frac x) ∧
  (floor x) ≤ x ∧
  x < (floor x) + 1 ∧
  0 ≤ (frac x) ∧
  (frac x) < 1 ∧
  x = 5/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l881_88153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_theorem_l881_88157

noncomputable def smallest_angle_x : ℝ := 9 * Real.pi / 180

theorem smallest_angle_theorem :
  smallest_angle_x > 0 ∧
  Real.sin (4 * smallest_angle_x) * Real.sin (6 * smallest_angle_x) = Real.cos (4 * smallest_angle_x) * Real.cos (6 * smallest_angle_x) ∧
  ∀ x : ℝ, x > 0 ∧ x < smallest_angle_x →
    Real.sin (4 * x) * Real.sin (6 * x) ≠ Real.cos (4 * x) * Real.cos (6 * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_theorem_l881_88157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_length_l881_88175

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation x^2 + y^2/b^2 = 1 -/
structure Ellipse where
  b : ℝ
  h_b : 0 < b ∧ b < 1

/-- Represents a line passing through a point -/
structure Line where
  point : Point

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if three real numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℝ) : Prop :=
  b - a = c - b

/-- Checks if a point is on the ellipse -/
def onEllipse (p : Point) (E : Ellipse) : Prop :=
  p.x^2 + p.y^2 / E.b^2 = 1

/-- The theorem to be proved -/
theorem ellipse_intersection_length 
  (E : Ellipse) 
  (F1 F2 : Point) -- Foci of the ellipse
  (l : Line) -- Line passing through F1
  (A B : Point) -- Intersection points of l and E
  (h_foci : F1 ≠ F2) -- F1 and F2 are distinct
  (h_line : l.point = F1) -- Line l passes through F1
  (h_intersect : A ≠ B ∧ onEllipse A E ∧ onEllipse B E) -- A and B are distinct points on E
  (h_arithmetic : isArithmeticSequence (distance A F2) (distance A B) (distance B F2)) -- |AF2|, |AB|, |BF2| form an arithmetic sequence
  : distance A B = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_length_l881_88175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_approximation_specific_approximation_l881_88150

/-- The function f(x, y) = x^y -/
noncomputable def f (x y : ℝ) : ℝ := x^y

/-- The second-order Taylor series expansion of f(x, y) = x^y around (1,1) -/
def taylor_expansion (x y : ℝ) : ℝ := 1 + (x - 1) + (x - 1) * (y - 1)

/-- Theorem stating that the Taylor expansion approximates f(x, y) up to second order -/
theorem taylor_approximation (x y : ℝ) :
  ∃ ε > 0, |f x y - taylor_expansion x y| < ε := by sorry

/-- Theorem stating that the Taylor expansion approximates 1.1^1.02 as 1.102 -/
theorem specific_approximation :
  abs (taylor_expansion 1.1 1.02 - 1.102) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_approximation_specific_approximation_l881_88150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l881_88101

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 9 = 0

/-- The area enclosed by the region -/
noncomputable def enclosed_area : ℝ := 4 * Real.pi

theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    enclosed_area = Real.pi * radius^2 := by
  -- Provide the center and radius
  let center : ℝ × ℝ := (2, -3)
  let radius : ℝ := 2
  
  -- Assert the existence of center and radius
  use center, radius
  
  -- Split the conjunction
  constructor
  
  -- Prove the equivalence of equations
  · sorry
  
  -- Prove the area equality
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l881_88101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_function_value_l881_88134

/-- Given a trigonometric function f(x) = sin(ωx + φ) with the following properties:
    - The terminal side of angle φ passes through point (1, -1)
    - ω > 0
    - For any two points (x₁, y₁) and (x₂, y₂) on the graph of f,
      if |f(x₁) - f(x₂)| = 2, then the minimum value of |x₁ - x₂| is π/3
    Then, f(π/2) = -√2/2 -/
theorem trig_function_value (ω φ : ℝ) (h_ω : ω > 0) 
  (h_φ : Real.tan φ = -1)
  (h_period : ∀ x₁ x₂, |Real.sin (ω * x₁ + φ) - Real.sin (ω * x₂ + φ)| = 2 → |x₁ - x₂| ≥ π / 3) :
  Real.sin (ω * (π / 2) + φ) = -Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_function_value_l881_88134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_y_l881_88177

theorem possible_values_of_y (x : ℝ) (h : x^2 + 6 * (x / (x - 3))^2 = 60) :
  let y := (x - 3)^2 * (x + 4) / (2*x - 5)
  y = 0 ∨ y = 10 ∨ y = 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_y_l881_88177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_edge_length_in_cone_l881_88154

/-- The edge length of a cube inscribed in a cone -/
noncomputable def inscribedCubeEdgeLength (θ : ℝ) : ℝ :=
  (2 * Real.sin θ) / (2 + Real.sqrt 2 * Real.tan θ)

/-- Theorem: The edge length of a cube inscribed in a cone with slant height 1 and angle θ
    between the slant height and the base is (2 * sin θ) / (2 + √2 * tan θ) -/
theorem inscribed_cube_edge_length_in_cone (θ : ℝ) :
  let cone_slant_height : ℝ := 1
  let cone_base_angle : ℝ := θ
  inscribedCubeEdgeLength θ = (2 * Real.sin θ) / (2 + Real.sqrt 2 * Real.tan θ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_edge_length_in_cone_l881_88154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_X21_X20_l881_88147

/-- X₀ is the interior of a triangle with side lengths 3, 4, and 5 -/
def X₀ : Set (Real × Real) := sorry

/-- For all positive integers n, Xₙ is the set of points within 1 unit of some point in Xₙ₋₁ -/
def X (n : Nat) : Set (Real × Real) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (Real × Real)) : Real := sorry

/-- The theorem to be proved -/
theorem area_difference_X21_X20 : 
  area (X 21 \ X 20) = 41 * Real.pi + 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_X21_X20_l881_88147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_problem_solving_l881_88179

/-- The number of problems solved by student s -/
def student_problem_count : ℕ → ℕ := sorry

/-- The number of students who solved problem p -/
def problem_student_count : ℕ → ℕ := sorry

/-- Given m > 1 students and n > 1 problems, where each student solves a different number of problems
    and each problem is solved by a different number of students, there exists a student who solved
    exactly one problem. -/
theorem olympiad_problem_solving (m n : ℕ) 
  (hm : m > 1) (hn : n > 1)
  (different_student_counts : ∀ i j, i ≠ j → student_problem_count i ≠ student_problem_count j)
  (different_problem_counts : ∀ i j, i ≠ j → problem_student_count i ≠ problem_student_count j) :
  ∃ s, student_problem_count s = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_problem_solving_l881_88179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_iff_a_ge_one_l881_88142

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - x * Real.log (a * x) - Real.exp (x - 1) / a

/-- Theorem stating the condition for f(x) to have a root in (0, +∞) -/
theorem f_has_root_iff_a_ge_one (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, x > 0 ∧ f a x = 0) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_iff_a_ge_one_l881_88142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_implies_nonnegative_a_l881_88127

theorem solution_implies_nonnegative_a (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 1 (Real.exp 1) ∧ Real.exp (a * x) ≥ 2 * Real.log x + x^2 - a * x) →
  a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_implies_nonnegative_a_l881_88127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_3_or_4_l881_88122

def is_multiple_of_3_or_4 (n : Nat) : Bool :=
  n % 3 = 0 || n % 4 = 0

def count_multiples (start finish : Nat) : Nat :=
  (List.range (finish - start + 1)).map (· + start)
    |>.filter is_multiple_of_3_or_4
    |>.length

theorem probability_multiple_3_or_4 :
  (count_multiples 1 30 : Rat) / 30 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_3_or_4_l881_88122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sasha_kolya_l881_88145

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- The race setup -/
structure Race where
  sasha : Runner
  lyosha : Runner
  kolya : Runner
  race_length : ℝ
  race_length_eq : race_length = 100
  lyosha_behind : sasha.distance - lyosha.distance = 10
  kolya_behind : lyosha.distance - kolya.distance = 10
  constant_speed : ∀ r : Runner, r.speed > 0

theorem distance_sasha_kolya (race : Race) : 
  race.sasha.distance - race.kolya.distance = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sasha_kolya_l881_88145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_distance_l881_88144

/-- A rectangular parallelepiped with given dimensions and vertices -/
structure Parallelepiped where
  x : ℝ
  y : ℝ
  z : ℝ
  P : ℝ × ℝ × ℝ
  Q : ℝ × ℝ × ℝ
  R : ℝ × ℝ × ℝ
  S : ℝ × ℝ × ℝ

/-- The perpendicular distance from a point to a plane -/
noncomputable def perpendicularDistance (p : Parallelepiped) : ℝ :=
  42 / Real.sqrt 34

/-- Theorem stating the perpendicular distance from P to plane QRS -/
theorem parallelepiped_distance (p : Parallelepiped) 
  (h1 : p.x = 5 ∧ p.y = 3 ∧ p.z = 7)
  (h2 : p.P = (0, 0, 0))
  (h3 : p.Q = (5, 0, 0))
  (h4 : p.R = (0, 3, 0))
  (h5 : p.S = (0, 0, 7)) :
  perpendicularDistance p = 42 / Real.sqrt 34 := by
  sorry

#eval Float.sqrt 34  -- This is just to demonstrate that we can use Float for approximate calculations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_distance_l881_88144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_without_y_l881_88152

theorem sum_of_coefficients_without_y (n : ℕ+) :
  let f := fun (x y : ℝ) => (4 - 3*x + 2*y)^(n : ℕ)
  (f 1 0) = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_without_y_l881_88152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relations_l881_88140

/-- Line l: ax + by - r² = 0 -/
def line_l (a b r : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y - r^2 = 0

/-- Circle C: x² + y² = r² -/
def circle_C (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2

/-- Distance from (0,0) to line l -/
noncomputable def distance_to_line (a b r : ℝ) : ℝ :=
  |r^2| / Real.sqrt (a^2 + b^2)

theorem line_circle_relations (a b r : ℝ) :
  (a^2 + b^2 = r^2 → distance_to_line a b r = r) ∧
  (a^2 + b^2 < r^2 → distance_to_line a b r > r) ∧
  (line_l a b r a b → distance_to_line a b r = r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relations_l881_88140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_classification_l881_88107

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, is_perfect_square (f (f a - b) + b * f (2*a))

def family1 (f : ℤ → ℤ) : Prop :=
  (∀ n : ℤ, Even n → f n = 0) ∧
  (∀ n : ℤ, Odd n → is_perfect_square (f n))

def family2 (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = n^2

theorem function_classification (f : ℤ → ℤ) :
  satisfies_condition f → family1 f ∨ family2 f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_classification_l881_88107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sevenFactorial_trailingZeros_base8_l881_88131

/-- The number of trailing zeros in n! when expressed in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- 7 factorial -/
def sevenFactorial : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

theorem sevenFactorial_trailingZeros_base8 :
  trailingZeros sevenFactorial 8 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sevenFactorial_trailingZeros_base8_l881_88131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l881_88104

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - x

-- Define the inequality condition
def inequality_condition (k : ℝ) : Prop :=
  ∀ x > 1, (k + 1) * (x - 1) < x * f (x - 1) + x^2

-- Theorem statement
theorem max_k_value :
  ∃ k : ℤ, k = 3 ∧ 
  inequality_condition k ∧
  ∀ m : ℤ, m > k → ¬inequality_condition m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l881_88104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_equal_area_l881_88133

structure Keyboard where
  key_side_length : ℝ
  is_positive : key_side_length > 0

def Point := ℝ × ℝ

noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

theorem triangles_equal_area (kb : Keyboard) 
  (Q A Z E S : Point)
  (h_Q : Q = (0, 2 * kb.key_side_length))
  (h_A : A = (0, kb.key_side_length))
  (h_Z : Z = (kb.key_side_length, 0))
  (h_E : E = (2 * kb.key_side_length, 2 * kb.key_side_length))
  (h_S : S = (2 * kb.key_side_length, kb.key_side_length)) :
  triangle_area Q A Z = triangle_area E S Z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_equal_area_l881_88133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outer_shape_is_regular_dodecagon_l881_88176

/-- A regular hexagon with side length a -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A square constructed on a side of the hexagon -/
structure OutwardSquare where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The shape formed by the outer vertices of the squares -/
structure OuterShape where
  vertices : Set (ℝ × ℝ)

/-- Check if a set of points forms a regular dodecagon -/
def is_regular_dodecagon (vertices : Set (ℝ × ℝ)) : Prop := sorry

/-- Calculate the area of a polygon given its vertices -/
noncomputable def area (vertices : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem to be proved -/
theorem outer_shape_is_regular_dodecagon (hexagon : RegularHexagon) 
  (squares : List OutwardSquare) 
  (outer_shape : OuterShape) : 
  (∀ s ∈ squares, s.side_length = hexagon.side_length) →
  (squares.length = 6) →
  (is_regular_dodecagon outer_shape.vertices) ∧ 
  (area outer_shape.vertices = 3 * hexagon.side_length^2 * (Real.sqrt 3 + 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outer_shape_is_regular_dodecagon_l881_88176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_g_l881_88115

-- Define the function g
def g (x : ℝ) : ℝ := |2*x - 3| + |x - 5| - |3*x - 9|

-- State the theorem
theorem sum_of_max_min_g : 
  ∃ (max_g min_g : ℝ), 
    (∀ x, x ∈ Set.Icc 3 10 → g x ≤ max_g) ∧ 
    (∃ x ∈ Set.Icc 3 10, g x = max_g) ∧
    (∀ x, x ∈ Set.Icc 3 10 → min_g ≤ g x) ∧ 
    (∃ x ∈ Set.Icc 3 10, g x = min_g) ∧
    max_g + min_g = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_g_l881_88115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_altitudes_l881_88182

/-- The angle between altitudes in a triangle -/
theorem angle_between_altitudes (α : Real) (h : 0 < α ∧ α < Real.pi) :
  let β := Real.pi - α / 2
  ∃ (f : Real → Real), f α = if α ≤ Real.pi / 2 then α else Real.pi - α :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_altitudes_l881_88182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l881_88118

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to internal angles A, B, C
  -- Conditions
  (Real.sin A = 3 * Real.sin B) →
  (C = π / 3) →
  (c = Real.sqrt 7) →
  -- Conclusions
  (a = 3 ∧ Real.sin A = (3 * Real.sqrt 21) / 14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l881_88118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_to_root_l881_88137

-- Define the equation
noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

-- Define the iteration function
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (1 + 1/x)

-- Define the sequence
noncomputable def seq : ℕ → ℝ
  | 0 => 1
  | n + 1 => g (seq n)

theorem convergence_to_root :
  ∃ (x₀ : ℝ), (f x₀ = 0) ∧ (x₀ > 0) ∧ (∀ ε > 0, ∃ N, ∀ n ≥ N, |seq n - x₀| < ε) := by
  sorry

#check convergence_to_root

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_to_root_l881_88137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l881_88178

theorem equation_solution : 
  ∃ x : ℝ, (10 : ℝ)^(x + Real.log 2) = 2000 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l881_88178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inverse_point_implies_base_l881_88143

noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

def has_inverse_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  ∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ Function.RightInverse g f ∧ g x = y

theorem log_inverse_point_implies_base (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  has_inverse_point (log a) (1/2) (Real.sqrt 2/2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inverse_point_implies_base_l881_88143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_implies_a_range_l881_88113

/-- The function f(x) defined as 1/√(ax² + 3ax + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / Real.sqrt (a * x^2 + 3 * a * x + 1)

/-- The theorem stating that if f has domain ℝ, then a is in [0, 4/9) -/
theorem f_domain_implies_a_range (a : ℝ) :
  (∀ x, f a x ∈ Set.univ) → a ∈ Set.Ici 0 ∩ Set.Iio (4/9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_implies_a_range_l881_88113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_OP_can_be_two_l881_88111

-- Define a circle with center O and radius 3
def myCircle (O : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) ≤ 3}

-- Define a point P inside the circle
def P_inside (O : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  P ∈ myCircle O ∧ P ≠ O

-- Theorem stating that OP can be 2
theorem OP_can_be_two (O : ℝ × ℝ) :
  ∃ P, P_inside O P ∧ Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_OP_can_be_two_l881_88111
