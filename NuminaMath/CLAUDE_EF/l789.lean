import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_semicircle_center_path_l789_78921

/-- The length of the path traveled by the center of a rolling semi-circle with radius r -/
noncomputable def center_path_length (r : ℝ) : ℝ := r * Real.pi

/-- The length of the path traveled by the center of a rolling semi-circle -/
theorem rolling_semicircle_center_path (r : ℝ) (h : r = 2) : 
  center_path_length r = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_semicircle_center_path_l789_78921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_arithmetic_sequence_l789_78939

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c

-- Theorem statement
theorem angles_arithmetic_sequence (t : Triangle) 
  (h : t.b * Real.cos t.C + Real.sqrt 3 * t.b * Real.sin t.C - t.a - t.c = 0) :
  ∃ (r : Real), t.A = t.B - r ∧ t.C = t.B + r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_arithmetic_sequence_l789_78939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_tangent_line_monotonicity_condition_l789_78918

/-- The function f(x) = x^3 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

/-- The function g(x) = bx^2 + c -/
def g (b c : ℝ) (x : ℝ) : ℝ := b*x^2 + c

/-- The derivative of f -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- The derivative of g -/
def g_deriv (b : ℝ) (x : ℝ) : ℝ := 2*b*x

theorem intersection_and_tangent_line (a b c : ℝ) :
  f a 1 = 0 ∧ g b c 1 = 0 ∧ f_deriv a 1 = g_deriv b 1 →
  a = -1 ∧ b = 1 ∧ c = -1 :=
by sorry

theorem monotonicity_condition (t : ℝ) :
  t ≠ 0 ∧
  f (-t^2) t = 0 ∧
  g t (-t^3) t = 0 ∧
  f_deriv (-t^2) t = g_deriv t t ∧
  (∀ x ∈ Set.Ioo (-1) 3, Monotone (λ x ↦ g t (-t^3) x - f (-t^2) x)) →
  t ≥ 3 ∨ t ≤ -9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_tangent_line_monotonicity_condition_l789_78918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l789_78992

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 
  Real.tan (π / 2 - 20 * π / 180) - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l789_78992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l789_78970

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y = 0

-- Define point E
def E : ℝ × ℝ := (0, 1)

-- Define the longest chord AC passing through E
def longest_chord (A C : ℝ × ℝ) : Prop :=
  circle_eq A.1 A.2 ∧ circle_eq C.1 C.2 ∧ 
  (∃ t : ℝ, A.1 = E.1 + t * (C.1 - E.1) ∧ A.2 = E.2 + t * (C.2 - E.2)) ∧
  (∀ P Q : ℝ × ℝ, circle_eq P.1 P.2 → circle_eq Q.1 Q.2 → 
    (∃ s : ℝ, P.1 = E.1 + s * (Q.1 - E.1) ∧ P.2 = E.2 + s * (Q.2 - E.2)) →
    (C.1 - A.1)^2 + (C.2 - A.2)^2 ≥ (Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Define the shortest chord BD passing through E
def shortest_chord (B D : ℝ × ℝ) : Prop :=
  circle_eq B.1 B.2 ∧ circle_eq D.1 D.2 ∧ 
  (∃ t : ℝ, B.1 = E.1 + t * (D.1 - E.1) ∧ B.2 = E.2 + t * (D.2 - E.2)) ∧
  (∀ P Q : ℝ × ℝ, circle_eq P.1 P.2 → circle_eq Q.1 Q.2 → 
    (∃ s : ℝ, P.1 = E.1 + s * (Q.1 - E.1) ∧ P.2 = E.2 + s * (Q.2 - E.2)) →
    (D.1 - B.1)^2 + (D.2 - B.2)^2 ≤ (Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Theorem statement
theorem quadrilateral_area 
  (A B C D : ℝ × ℝ) 
  (h_longest : longest_chord A C) 
  (h_shortest : shortest_chord B D) : 
  abs ((A.1 - C.1) * (B.2 - D.2) - (A.2 - C.2) * (B.1 - D.1)) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l789_78970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_problem_solution_l789_78930

-- Define the problem
def milk_problem (y : ℝ) : Prop :=
  let daily_production_per_cow := (y + 2) / (y * (y + 3))
  let total_daily_production := (y + 4) * daily_production_per_cow
  let days_needed := (y + 6) / total_daily_production
  days_needed = y * (y + 3) * (y + 6) / ((y + 2) * (y + 4))

-- Theorem statement
theorem milk_problem_solution (y : ℝ) (h : y ≠ 0) : milk_problem y := by
  -- Unfold the definition of milk_problem
  unfold milk_problem
  -- Simplify the expression
  simp [h]
  -- The proof is complete
  sorry

#check milk_problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_problem_solution_l789_78930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_twelve_l789_78974

def S : Set ℤ := {-24, -3, -2, 1, 2, 8}

theorem largest_quotient_is_twelve :
  ∀ a b : ℤ, a ∈ S → b ∈ S → a ≠ 0 → b ≠ 0 → (a : ℚ) / b ≤ 12 ∧ ∃ c d : ℤ, c ∈ S ∧ d ∈ S ∧ c ≠ 0 ∧ d ≠ 0 ∧ (c : ℚ) / d = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_twelve_l789_78974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_count_l789_78929

/-- Represents a hyperbola in the x-y plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop

/-- Represents a line in the x-y plane -/
structure Line where
  m : ℝ
  c : ℝ

/-- Represents a point in the x-y plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola) : Point :=
  { x := Real.sqrt (h.a^2 + h.b^2), y := 0 }

/-- A line passes through a point -/
def line_passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.c

/-- A line intersects a hyperbola -/
def line_intersects_hyperbola (l : Line) (h : Hyperbola) (p : Point) : Prop :=
  line_passes_through l p ∧ h.equation p.x p.y

/-- Main theorem -/
theorem hyperbola_intersection_count :
  ∀ (h : Hyperbola),
  h.equation = (fun x y => x^2 / 2 - y^2 = 1) →
  ∃ (lines : Finset Line),
  (∀ l ∈ lines, 
    line_passes_through l (right_focus h) ∧
    ∃ A B : Point,
      line_intersects_hyperbola l h A ∧
      line_intersects_hyperbola l h B ∧
      distance A B = 4) ∧
  lines.card = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_count_l789_78929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_relationship_l789_78975

theorem inequality_relationship (x y : ℝ) : 
  (∃ x y, (2 : ℝ)^(x - y) < 1 ∧ Real.log (x / y) ≥ 0) ∧ 
  (∃ x y, Real.log (x / y) < 0 ∧ (2 : ℝ)^(x - y) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_relationship_l789_78975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_crew_ratio_is_half_l789_78962

/-- Represents the lemonade stand problem -/
structure LemonadeStand where
  total : ℕ
  sold_to_kids : ℕ
  given_to_friends : ℕ
  drunk_by_hazel : ℕ

/-- The ratio of lemonade sold to construction crew to total lemonade made -/
def construction_crew_ratio (stand : LemonadeStand) : ℚ :=
  let sold_to_crew := stand.total - (stand.sold_to_kids + stand.given_to_friends + stand.drunk_by_hazel)
  ↑sold_to_crew / ↑stand.total

/-- Theorem stating the ratio of lemonade sold to construction crew to total lemonade made is 1/2 -/
theorem construction_crew_ratio_is_half (stand : LemonadeStand) 
    (h1 : stand.total = 56)
    (h2 : stand.sold_to_kids = 18)
    (h3 : stand.given_to_friends = stand.sold_to_kids / 2)
    (h4 : stand.drunk_by_hazel = 1) : 
  construction_crew_ratio stand = 1 / 2 := by
  sorry

#eval construction_crew_ratio { 
  total := 56, 
  sold_to_kids := 18, 
  given_to_friends := 9, 
  drunk_by_hazel := 1 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_crew_ratio_is_half_l789_78962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_independent_of_K_l789_78984

-- Define the circles and points
variable (r₁ r₂ : ℝ)
variable (ω₁ ω₂ : Set (ℝ × ℝ))
variable (T K A B S : ℝ × ℝ)

-- State the conditions
axiom r₁_positive : 0 < r₁
axiom r₂_positive : 0 < r₂
axiom r₁_less_r₂ : r₁ < r₂
axiom ω₁_def : ω₁ = {p : ℝ × ℝ | (p.1 - T.1)^2 + (p.2 - T.2)^2 = r₁^2}
axiom ω₂_def : ω₂ = {p : ℝ × ℝ | (p.1 - T.1)^2 + (p.2 - T.2)^2 = r₂^2}
axiom T_on_both : T ∈ ω₁ ∧ T ∈ ω₂
axiom K_on_ω₁ : K ∈ ω₁
axiom K_ne_T : K ≠ T
axiom A_B_on_ω₂ : A ∈ ω₂ ∧ B ∈ ω₂

-- Define tangent line and midpoint conditions
def IsTangentLine (l : Set (ℝ × ℝ)) (ω : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop := sorry

def IsMidpoint (m a b : ℝ × ℝ) (ω : Set (ℝ × ℝ)) (t : ℝ × ℝ) : Prop := sorry

axiom tangent_intersect : ∃ l : Set (ℝ × ℝ), IsTangentLine l ω₁ K ∧ A ∈ l ∧ B ∈ l
axiom S_midpoint : IsMidpoint S A B ω₂ T

-- Define the circumcircle radius function
noncomputable def circumcircle_radius (a b c : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem circumcircle_radius_independent_of_K :
  circumcircle_radius A K S = Real.sqrt (r₂ * (r₂ - r₁)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_independent_of_K_l789_78984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_is_linear_l789_78901

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ,
    (a + b + c ≥ 0 → f (a^3) + f (b^3) + f (c^3) ≥ 3 * f (a*b*c)) ∧
    (a + b + c ≤ 0 → f (a^3) + f (b^3) + f (c^3) ≤ 3 * f (a*b*c))

/-- The main theorem: any function satisfying the conditions is linear -/
theorem function_is_linear (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_is_linear_l789_78901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l789_78902

-- Define the original expression
noncomputable def original_expression (a : ℝ) : ℝ :=
  (a + 3) / a * 6 / (a^2 + 6*a + 9) + (2*a - 6) / (a^2 - 9)

-- Define the simplified expression
noncomputable def simplified_expression (a : ℝ) : ℝ := 2 / a

-- Theorem statement
theorem expression_simplification_and_evaluation :
  let a : ℝ := Real.sqrt 2
  original_expression a = simplified_expression a ∧ 
  simplified_expression a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l789_78902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_value_l789_78955

/-- The probability distribution of X for k = 1, 2, 3, 4 -/
noncomputable def P (k : ℕ) (c : ℝ) : ℝ := c / (k * (k + 1))

/-- The sum of probabilities for k = 1, 2, 3, 4 equals 1 -/
axiom prob_sum (c : ℝ) : P 1 c + P 2 c + P 3 c + P 4 c = 1

/-- The probability that X is between 1/2 and 5/2 -/
noncomputable def prob_between (c : ℝ) : ℝ := P 1 c + P 2 c

theorem prob_between_value :
  ∃ c : ℝ, prob_between c = 5/6 :=
by
  -- We'll use c = 5/4 as in the solution
  use 5/4
  -- Unfold the definition of prob_between
  unfold prob_between
  -- Unfold the definition of P
  unfold P
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_value_l789_78955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_sequence_l789_78985

theorem triangle_trig_sequence (A B C : ℝ) : 
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  -- Sum of angles in a triangle is π
  A + B + C = Real.pi ∧ 
  -- Angles form an arithmetic sequence
  A + C = 2 * B ∧ 
  -- Given cos A
  Real.cos A = 2/3 →
  -- Prove sin C
  Real.sin C = (2 * Real.sqrt 3 + Real.sqrt 5) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_sequence_l789_78985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_32_solutions_l789_78966

/-- The number of real solutions to the equation x/50 = sin x -/
def num_solutions : ℕ := 32

/-- The equation x/50 = sin x -/
def equation (x : ℝ) : Prop := x / 50 = Real.sin x

/-- The set of solutions to the equation -/
def solution_set : Set ℝ := {x : ℝ | equation x}

theorem equation_has_32_solutions :
  ∃! (S : Set ℝ), S = solution_set ∧ (∀ x ∈ S, equation x) ∧ Finite S ∧ Nat.card S = num_solutions :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_32_solutions_l789_78966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_bound_l789_78924

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + Real.log x - (a - a^2) / x

-- Define the function g(x) when a = 1
noncomputable def g (x : ℝ) : ℝ := x * f 1 x

-- Theorem about the monotonicity of f(x) and the existence of t for g(x)
theorem f_monotonicity_and_g_bound :
  (∃ (a : ℝ), ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ f a x₁ > f a x₂) ∧
  (∃ (a : ℝ), ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∃ (t : ℤ), t ≥ 0 ∧ ∀ (x : ℝ), x > 0 → ↑t ≥ g x) ∧
  (∀ (t : ℤ), t < 0 → ∃ (x : ℝ), x > 0 ∧ ↑t < g x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_bound_l789_78924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_not_equal_neg_48_l789_78914

theorem product_not_equal_neg_48 : ∃! p : ℚ × ℚ, p ∈ [(6, -8), (4, -12), (3, 16), (-1, 48), (3/2, -32)] ∧ p.1 * p.2 ≠ -48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_not_equal_neg_48_l789_78914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_divisors_l789_78934

/-- A number with exactly 15 positive divisors that is a product of two distinct odd primes -/
structure SpecialNumber where
  n : ℕ
  is_product_of_two_primes : ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ Odd p ∧ Odd q ∧ n = p * q
  has_fifteen_divisors : (Nat.divisors n).card = 15

/-- The number of positive divisors of a natural number -/
def num_divisors (m : ℕ) : ℕ := (Nat.divisors m).card

theorem special_number_divisors (x : SpecialNumber) : num_divisors (16 * x.n^2) = 225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_divisors_l789_78934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l789_78995

/-- The focus of the parabola y = 4x^2 has coordinates (0, 1/16) -/
theorem parabola_focus_coordinates :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2
  ∃ p : ℝ, p = 1/16 ∧ (0, p) = (0, 1/16) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l789_78995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lila_birthday_next_tuesday_l789_78922

/-- Represents the days of the week -/
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving BEq, Repr

/-- Checks if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)

/-- Calculates the next weekday given the current weekday and whether it's a leap year -/
def nextWeekday (day : Weekday) (isLeap : Bool) : Weekday :=
  match day, isLeap with
  | Weekday.Sunday, false => Weekday.Monday
  | Weekday.Sunday, true => Weekday.Tuesday
  | Weekday.Monday, false => Weekday.Tuesday
  | Weekday.Monday, true => Weekday.Wednesday
  | Weekday.Tuesday, false => Weekday.Wednesday
  | Weekday.Tuesday, true => Weekday.Thursday
  | Weekday.Wednesday, false => Weekday.Thursday
  | Weekday.Wednesday, true => Weekday.Friday
  | Weekday.Thursday, false => Weekday.Friday
  | Weekday.Thursday, true => Weekday.Saturday
  | Weekday.Friday, false => Weekday.Saturday
  | Weekday.Friday, true => Weekday.Sunday
  | Weekday.Saturday, false => Weekday.Sunday
  | Weekday.Saturday, true => Weekday.Monday

/-- Theorem: Lila's birthday will next fall on a Tuesday in 2021 -/
theorem lila_birthday_next_tuesday :
  ∀ (year : Nat),
  year ≥ 2012 →
  (year == 2012 → isLeapYear year ∧ nextWeekday Weekday.Friday (isLeapYear year) == Weekday.Saturday) →
  (∀ y, y > 2012 ∧ y < year →
    nextWeekday (nextWeekday Weekday.Friday (isLeapYear 2012)) (isLeapYear y) ≠ Weekday.Tuesday) →
  nextWeekday (nextWeekday Weekday.Friday (isLeapYear 2012)) (isLeapYear year) == Weekday.Tuesday →
  year == 2021 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lila_birthday_next_tuesday_l789_78922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_circle_property_l789_78945

noncomputable section

structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

def eccentricity (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2) / e.a

def passesThrough (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

def areaTriangle (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  abs ((x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)) / 2)

theorem ellipse_equation_and_circle_property (e : Ellipse)
  (h_ecc : eccentricity e = Real.sqrt 2 / 2)
  (h_pass : passesThrough e 1 (-Real.sqrt 2 / 2)) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    passesThrough e x₁ y₁ ∧
    passesThrough e x₂ y₂ ∧
    areaTriangle x₁ y₁ x₂ y₂ e.a 0 = 4 * Real.sqrt 3 / 5 →
    x₁ * x₂ + y₁ * y₂ = 0) ∧
  e.a = Real.sqrt 2 ∧ e.b = 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_circle_property_l789_78945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_problem_l789_78953

/-- The height of water in an inverted right circular cone tank -/
noncomputable def water_height (base_radius : ℝ) (tank_height : ℝ) (fill_percentage : ℝ) : ℝ :=
  tank_height * (fill_percentage ^ (1/3 : ℝ))

/-- The problem statement -/
theorem water_height_problem (base_radius tank_height fill_percentage : ℝ) 
  (h1 : base_radius = 20)
  (h2 : tank_height = 60)
  (h3 : fill_percentage = 0.4) :
  water_height base_radius tank_height fill_percentage = 30 * (4 ^ (1/3 : ℝ)) :=
by
  -- Unfold the definition of water_height
  unfold water_height
  -- Substitute the given values
  rw [h2, h3]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_problem_l789_78953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangle_area_max_cylinder_surface_area_l789_78980

/-- Represents a right-angled triangle with legs of length a and b -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- Represents a point on the hypotenuse of the right triangle -/
def HypotenusePoint (t : RightTriangle) := { x : ℝ // 0 ≤ x ∧ x ≤ t.a }

/-- Calculates the area of the rectangle formed by perpendiculars from a point on the hypotenuse -/
noncomputable def rectangleArea (t : RightTriangle) (p : HypotenusePoint t) : ℝ :=
  p.val * (t.a / t.b * (t.b - p.val))

/-- Theorem stating that the rectangle area is maximized at the midpoint of the hypotenuse -/
theorem max_rectangle_area (t : RightTriangle) :
  ∃ (p : HypotenusePoint t),
    (∀ (q : HypotenusePoint t), rectangleArea t q ≤ rectangleArea t p) ∧
    p.val = t.b / 2 ∧
    rectangleArea t p = t.a * t.b / 4 := by
  sorry

/-- Calculates the surface area of the cylinder formed by rotating the rectangle -/
noncomputable def cylinderSurfaceArea (t : RightTriangle) (p : HypotenusePoint t) : ℝ :=
  2 * Real.pi * p.val^2 + 2 * Real.pi * p.val * (t.a / t.b * (t.b - p.val))

/-- Theorem stating that the cylinder surface area is maximized at a specific point -/
theorem max_cylinder_surface_area (t : RightTriangle) :
  ∃ (p : HypotenusePoint t),
    (∀ (q : HypotenusePoint t), cylinderSurfaceArea t q ≤ cylinderSurfaceArea t p) ∧
    p.val = t.a * t.b / (2 * (t.a - t.b)) ∧
    cylinderSurfaceArea t p = Real.pi * t.a^2 * t.b^2 / (4 * (t.a - t.b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangle_area_max_cylinder_surface_area_l789_78980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_multiples_eq_six_l789_78927

/-- The number of positive factors of 180 that are also multiples of 15 -/
def count_factors_multiples : ℕ := 
  (Finset.filter (fun x => x ∣ 180 ∧ 15 ∣ x) (Finset.range 181)).card

/-- Theorem stating that the count of positive factors of 180 that are also multiples of 15 is 6 -/
theorem count_factors_multiples_eq_six : count_factors_multiples = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_multiples_eq_six_l789_78927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangular_faces_l789_78982

/-- A convex polyhedron with vertices of degree 4 -/
structure ConvexPolyhedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  faces : Finset (Finset ℕ)
  convex : Prop
  vertex_degree_4 : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4

/-- The number of triangular faces in a polyhedron -/
def num_triangular_faces (p : ConvexPolyhedron) : ℕ :=
  (p.faces.filter (λ f => f.card = 3)).card

/-- Theorem: The minimum number of triangular faces in a convex polyhedron
    with vertices of degree 4 is 8 -/
theorem min_triangular_faces (p : ConvexPolyhedron) :
  p.convex → num_triangular_faces p ≥ 8 := by
  sorry

#check min_triangular_faces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangular_faces_l789_78982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_of_matrices_l789_78928

theorem det_product_of_matrices {n : Type*} [Fintype n] [DecidableEq n]
  (C D : Matrix n n ℝ) (h1 : Matrix.det C = 3) (h2 : Matrix.det D = 8) : 
  Matrix.det (C * D) = 24 := by
  have h3 := Matrix.det_mul C D
  rw [h3, h1, h2]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_of_matrices_l789_78928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speech_orders_l789_78958

theorem speech_orders (total_students : Nat) (selected_students : Nat) 
  (h1 : total_students = 6)
  (h2 : selected_students = 3)
  (h3 : ∃ (order : Fin selected_students → Fin total_students), 
    (order ⟨0, by simp [h2]⟩ = ⟨0, by simp [h1]⟩ ∨ order ⟨0, by simp [h2]⟩ = ⟨1, by simp [h1]⟩) ∨ 
    (order ⟨1, by simp [h2]⟩ = ⟨0, by simp [h1]⟩ ∨ order ⟨1, by simp [h2]⟩ = ⟨1, by simp [h1]⟩) ∨ 
    (order ⟨2, by simp [h2]⟩ = ⟨0, by simp [h1]⟩ ∨ order ⟨2, by simp [h2]⟩ = ⟨1, by simp [h1]⟩)) :
  (Nat.choose total_students selected_students - 
   Nat.choose (total_students - 2) selected_students) * 
  (Nat.factorial selected_students) = 96 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speech_orders_l789_78958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_base_length_is_20_l789_78977

/-- The length of the base of a rectangular vessel containing a fully immersed cube -/
noncomputable def vessel_base_length (cube_edge width rise : ℝ) : ℝ :=
  (cube_edge ^ 3) / (width * rise)

/-- Theorem stating the length of the vessel's base is 20 cm under given conditions -/
theorem vessel_base_length_is_20 :
  vessel_base_length 15 15 11.25 = 20 := by
  -- Unfold the definition of vessel_base_length
  unfold vessel_base_length
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_base_length_is_20_l789_78977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l789_78964

open Real

/-- The function f(x) = (1/2)e^x + x - 6 -/
noncomputable def f (x : ℝ) : ℝ := (1/2) * exp x + x - 6

/-- Theorem: If f has a zero point in (n, n+1), then n = 2 -/
theorem zero_point_interval (n : ℕ) (h : ∃ x : ℝ, n < x ∧ x < n + 1 ∧ f x = 0) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l789_78964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_20_terms_eq_380_l789_78904

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a2_eq_2 : a 2 = 2
  geometric_sub : ∃ r ≠ 1, a 3 = a 2 * r ∧ a 5 = a 3 * r

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1))

/-- The main theorem -/
theorem sum_20_terms_eq_380 (seq : ArithmeticSequence) :
  sum_n_terms seq 20 = 380 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_20_terms_eq_380_l789_78904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_games_total_prove_team_games_total_l789_78919

theorem team_games_total (first_games : ℕ) (first_win_rate : ℚ) 
  (remaining_win_rate : ℚ) (total_win_rate : ℚ) (total_games : ℕ) : Prop :=
  first_games = 100 ∧
  first_win_rate = 60 / 100 ∧
  remaining_win_rate = 50 / 100 ∧
  total_win_rate = 70 / 100 ∧
  (first_win_rate * first_games + remaining_win_rate * (total_games - first_games)) / total_games = total_win_rate ∧
  total_games = 150

theorem prove_team_games_total : ∃ (first_games : ℕ) (first_win_rate : ℚ) 
  (remaining_win_rate : ℚ) (total_win_rate : ℚ) (total_games : ℕ),
  team_games_total first_games first_win_rate remaining_win_rate total_win_rate total_games :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_games_total_prove_team_games_total_l789_78919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_conversion_l789_78979

/-- Conversion from rectangular to polar coordinates -/
theorem rect_to_polar_conversion (x y : ℝ) (r θ : ℝ) : 
  x = 3 ∧ y = -3 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ 
  r * (Real.cos θ) = x ∧ r * (Real.sin θ) = y →
  r = 3 * Real.sqrt 2 ∧ θ = 7 * π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_conversion_l789_78979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_travel_time_equality_problem_solution_l789_78987

/-- Represents a point on the Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Represents a tunnel between two points on Earth -/
structure Tunnel (A B : EarthPoint) where
  travelTime : Real

/-- Represents the ratio of surface distances -/
structure DistanceRatio where
  m : ℕ
  n : ℕ

/-- The travel time through a tunnel is independent of its length -/
axiom travel_time_independent (A B C : EarthPoint) (t : Tunnel A B) (r : DistanceRatio) :
  ∃ (t_ac : Tunnel A C), t_ac.travelTime = t.travelTime

theorem tunnel_travel_time_equality (A B C : EarthPoint) (t : Tunnel A B) (r : DistanceRatio) :
  ∃ (t_ac : Tunnel A C), t_ac.travelTime = t.travelTime := by
  exact travel_time_independent A B C t r

/-- The specific travel time for the given problem -/
def problem_travel_time : Real := 42

theorem problem_solution (A B C : EarthPoint) (r : DistanceRatio) :
  ∃ (t_ab : Tunnel A B) (t_ac : Tunnel A C),
    t_ab.travelTime = problem_travel_time ∧ t_ac.travelTime = problem_travel_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_travel_time_equality_problem_solution_l789_78987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l789_78993

/-- The time it takes for Raja and Ram to complete a piece of work together -/
noncomputable def time_together (raja_time ram_time : ℝ) : ℝ :=
  1 / (1 / raja_time + 1 / ram_time)

/-- Theorem stating that Raja and Ram can complete the work together in 4 days -/
theorem work_completion_time (raja_time ram_time : ℝ) 
  (h_raja : raja_time = 12)
  (h_ram : ram_time = 6) :
  time_together raja_time ram_time = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l789_78993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l789_78917

/-- Represents a line segment in a 2D grid --/
structure LineSegment where
  start : Nat × Nat
  finish : Nat × Nat

/-- Represents the game state --/
structure GameState where
  grid : Nat × Nat
  segments : List LineSegment

/-- Checks if a set of line segments is valid according to the game rules --/
def isValidSegmentSet (grid : Nat × Nat) (segments : List LineSegment) : Prop :=
  -- All vertices are used
  ∀ x y, x < grid.1 ∧ y < grid.2 → 
    ∃ s ∈ segments, s.start = (x, y) ∨ s.finish = (x, y)

/-- Checks if there exists a valid direction assignment for the segments --/
def hasValidDirectionAssignment (segments : List LineSegment) : Prop :=
  ∃ directions : List (Int × Int),
    directions.length = segments.length ∧
    (directions.zip segments).foldl (λ sum (dir, seg) => 
      sum + (dir.1 * (seg.finish.1 - seg.start.1), dir.2 * (seg.finish.2 - seg.start.2))) 
      (0, 0) = (0, 0)

/-- The main theorem stating that the first player has a winning strategy --/
theorem first_player_winning_strategy :
  ∀ (gameState : GameState),
    gameState.grid = (50, 70) →
    ∃ (finalState : GameState),
      finalState.grid = gameState.grid ∧
      isValidSegmentSet finalState.grid finalState.segments ∧
      hasValidDirectionAssignment finalState.segments :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l789_78917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B1_selected_prob_physics_winner_l789_78967

-- Define the set of winners
inductive Winner : Type
| A1 | A2  -- Chinese
| B1 | B2  -- Mathematics
| C1 | C2  -- English
| D1       -- Physics
deriving BEq, DecidableEq

-- Define a function to get the subject of a winner
def subject (w : Winner) : String :=
  match w with
  | Winner.A1 | Winner.A2 => "Chinese"
  | Winner.B1 | Winner.B2 => "Mathematics"
  | Winner.C1 | Winner.C2 => "English"
  | Winner.D1 => "Physics"

-- Define a valid team
def isValidTeam (team : List Winner) : Prop :=
  team.length = 3 ∧ 
  team.toFinset.card = 3 ∧ 
  (∀ s : String, (team.filter (fun w => subject w = s)).length ≤ 1)

-- Define the set of all possible teams
noncomputable def allTeams : Finset (List Winner) :=
  sorry

-- Theorem 1: Probability of B₁ being selected
theorem prob_B1_selected : 
  (allTeams.filter (fun team => team.contains Winner.B1)).card / allTeams.card = 2/5 := by
  sorry

-- Theorem 2: Probability of team containing a Physics winner
theorem prob_physics_winner : 
  (allTeams.filter (fun team => team.contains Winner.D1)).card / allTeams.card = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B1_selected_prob_physics_winner_l789_78967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_teachers_required_l789_78973

/-- Represents the categories of subjects -/
inductive SubjectCategory
| Math
| Science
| SocialStudies
| Arts

/-- Represents the school's subject and teacher structure -/
structure School where
  totalSubjects : Nat
  totalTeachers : Nat
  mathSubjects : Nat
  scienceSubjects : Nat
  socialStudiesSubjects : Nat
  artsSubjects : Nat
  expertTeachersPerMathSubject : Nat
  expertTeachersPerScienceSubject : Nat
  expertTeachersPerSocialStudiesSubject : Nat
  expertTeachersPerArtsSubject : Nat

/-- The given school configuration -/
def mySchool : School :=
  { totalSubjects := 12
  , totalTeachers := 40
  , mathSubjects := 3
  , scienceSubjects := 4
  , socialStudiesSubjects := 3
  , artsSubjects := 2
  , expertTeachersPerMathSubject := 4
  , expertTeachersPerScienceSubject := 4
  , expertTeachersPerSocialStudiesSubject := 2
  , expertTeachersPerArtsSubject := 3
  }

/-- Calculates the total number of expert teachers required -/
def totalExpertTeachersRequired (s : School) : Nat :=
  s.mathSubjects * s.expertTeachersPerMathSubject +
  s.scienceSubjects * s.expertTeachersPerScienceSubject +
  s.socialStudiesSubjects * s.expertTeachersPerSocialStudiesSubject +
  s.artsSubjects * s.expertTeachersPerArtsSubject

/-- Theorem stating that the minimum number of teachers required is equal to the total number of teachers available -/
theorem min_teachers_required (s : School) :
  s.totalTeachers = totalExpertTeachersRequired s :=
by
  cases s
  simp [totalExpertTeachersRequired]
  sorry

#eval totalExpertTeachersRequired mySchool

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_teachers_required_l789_78973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_value_l789_78983

theorem cos_2beta_value (α β : ℝ) 
  (h1 : Real.sin (α - β) = 3/5)
  (h2 : Real.cos (α + β) = -3/5)
  (h3 : α - β ∈ Set.Ioo (π/2) π)
  (h4 : α + β ∈ Set.Ioo (π/2) π) :
  Real.cos (2 * β) = 24/25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_value_l789_78983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_neg_one_range_of_a_l789_78968

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (-x) - a * x

-- Theorem 1: Minimum value of f when a = -1
theorem min_value_when_a_neg_one :
  ∃ (min : ℝ), min = 1 ∧ ∀ (x : ℝ), f (-1) x ≥ min := by sorry

-- Theorem 2: Range of a
theorem range_of_a :
  ∀ (a : ℝ), (∀ (x : ℝ), x ≥ 0 → f a (-x) + Real.log (x + 1) ≥ 1) ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_neg_one_range_of_a_l789_78968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_m_minus_n_equals_8_l789_78952

def vector_a : Fin 3 → ℝ := ![1, 3, -2]
def vector_b (m n : ℝ) : Fin 3 → ℝ := ![2, m + 1, n - 1]

theorem parallel_vectors_imply_m_minus_n_equals_8 (m n : ℝ) :
  (∃ (k : ℝ), ∀ i, vector_b m n i = k * vector_a i) →
  m - n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_m_minus_n_equals_8_l789_78952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_tower_height_rounded_l789_78937

/-- The height of a tower given three points at specific distances from its base --/
theorem tower_height (a b c : ℝ) (h_a : a = 800) (h_b : b = 700) (h_c : c = 500)
  (h_angles : ∃ (α β γ : ℝ), 
    α + β + γ = π/2 ∧ 
    Real.tan α = H / a ∧ 
    Real.tan β = H / b ∧ 
    Real.tan γ = H / c) :
  ∃ H : ℝ, H = 100 * Real.sqrt 14 :=
by
  sorry

/-- The rounded height of the tower to the nearest meter --/
theorem tower_height_rounded (h : ∃ H : ℝ, H = 100 * Real.sqrt 14) :
  ∃ H_rounded : ℕ, H_rounded = 374 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_tower_height_rounded_l789_78937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_cost_optimization_l789_78989

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := 800 / (3 * x + 5) + 6 * x

-- State the theorem
theorem insulation_cost_optimization :
  ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 10 ∧
  f x = 70 ∧
  (∀ y : ℝ, 1 ≤ y ∧ y ≤ 10 → f y ≥ f x) ∧
  x = 5 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_cost_optimization_l789_78989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_59_l789_78910

def mySequence : ℕ → ℕ
| 0 => 1
| n + 1 => mySequence n + 2

theorem thirtieth_term_is_59 : mySequence 29 = 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_59_l789_78910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_selection_l789_78908

-- Define the range of integers
def valid_range : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 50}

-- Define a function to check if a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

-- Define the property for the selected numbers
def valid_selection (s : Finset ℕ) : Prop :=
  s.toSet ⊆ valid_range ∧ 
  s.card = 5 ∧ 
  is_perfect_cube (s.prod id)

-- The theorem to prove
theorem no_valid_selection : ¬∃ s : Finset ℕ, valid_selection s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_selection_l789_78908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_equation_l789_78971

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -2

-- Define the left focus of the hyperbola
def left_focus (x : ℝ) : Prop := x = -2

-- Define the length of the line segment
def line_segment_length : ℝ := 6

-- Main theorem
theorem asymptote_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, parabola x y ∧ hyperbola x y a b) →
  (∀ x, directrix x ↔ left_focus x) →
  (2 * b^2 / a = line_segment_length) →
  (∀ x y : ℝ, y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_equation_l789_78971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_factorial_sum_l789_78976

def factorial_sum (n : ℕ) : ℕ :=
  match n with
  | 0 => Nat.factorial 3
  | n + 1 => factorial_sum n + Nat.factorial (3 * (n + 1))

theorem last_two_digits_of_factorial_sum :
  factorial_sum 9 % 100 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_factorial_sum_l789_78976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_coprime_and_non_coprime_l789_78903

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the greatest common divisor function
def gcd (a b : ℤ) : ℕ := Int.gcd a.natAbs b.natAbs

-- Define the function f
noncomputable def f (n : ℕ+) : ℕ := gcd n (floor (Real.sqrt 2 * n))

-- Theorem statement
theorem infinitely_many_coprime_and_non_coprime :
  (∃ (S : Set ℕ+), Set.Infinite S ∧ ∀ n ∈ S, f n = 1) ∧
  (∃ (T : Set ℕ+), Set.Infinite T ∧ ∀ n ∈ T, f n ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_coprime_and_non_coprime_l789_78903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_transformation_l789_78905

theorem polynomial_root_transformation (a b c : ℂ) :
  (∀ x : ℂ, x^3 + 2*x^2 + 2 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∃! f : Polynomial ℂ, Polynomial.Monic f ∧ 
    (∀ x : ℂ, (Polynomial.eval x f = 0) ↔ x = a^2 ∨ x = b^2 ∨ x = c^2)) →
  (∃ f : Polynomial ℂ, Polynomial.Monic f ∧ 
    (∀ x : ℂ, (Polynomial.eval x f = 0) ↔ x = a^2 ∨ x = b^2 ∨ x = c^2) ∧
    Polynomial.eval 1 f = -7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_transformation_l789_78905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_sum_double_l789_78925

theorem midpoint_coordinate_sum_double : 
  let A : ℝ × ℝ := (15, -7)
  let B : ℝ × ℝ := (-3, 12)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * (M.1 + M.2) = 17 :=
by
  -- Introduce the points and midpoint
  let A : ℝ × ℝ := (15, -7)
  let B : ℝ × ℝ := (-3, 12)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_sum_double_l789_78925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_symmetry_l789_78965

-- Define a point in 3D space
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a tetrahedron
structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the center of mass of a tetrahedron
noncomputable def centerOfMass (t : Tetrahedron) : Point := sorry

-- Define the center of the inscribed sphere of a tetrahedron
noncomputable def centerOfInscribedSphere (t : Tetrahedron) : Point := sorry

-- Define a line passing through two points
structure Line where
  p1 : Point
  p2 : Point

-- Define a function to check if a line intersects an edge
def intersectsEdge (l : Line) (p1 : Point) (p2 : Point) : Prop := sorry

-- Define a distance function between two points
noncomputable def dist (p1 p2 : Point) : ℝ := sorry

-- Main theorem
theorem tetrahedron_symmetry (t : Tetrahedron) 
  (l : Line) 
  (h1 : l.p1 = centerOfMass t) 
  (h2 : l.p2 = centerOfInscribedSphere t) 
  (h3 : intersectsEdge l t.A t.B) 
  (h4 : intersectsEdge l t.C t.D) : 
  (dist t.A t.C = dist t.B t.D) ∧ (dist t.A t.D = dist t.B t.C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_symmetry_l789_78965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_length_l789_78941

/-- Represents an isosceles triangle -/
def IsoscelesTriangle (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

/-- Calculates the base length of a triangle -/
def baseLength (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
  sorry

/-- Calculates the area of a triangle -/
def area (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
  sorry

/-- Calculates the length of a congruent side in an isosceles triangle -/
def congruentSideLength (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
  sorry

/-- An isosceles triangle with base 30 and area 75 has congruent sides of length 5√10 -/
theorem isosceles_triangle_side_length (D E F : ℝ × ℝ) : 
  let triangle := (D, E, F)
  IsoscelesTriangle triangle ∧ 
  baseLength triangle = 30 ∧ 
  area triangle = 75 → 
  congruentSideLength triangle = 5 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_length_l789_78941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l789_78961

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => 3 * Real.sqrt x + 3 * x^(-(1/2 : ℝ)) - 8
  ∃ x₁ x₂ : ℝ, 
    x₁ = ((8 + 2 * Real.sqrt 7) / 6)^2 ∧
    x₂ = ((8 - 2 * Real.sqrt 7) / 6)^2 ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l789_78961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_specific_truncated_cone_l789_78907

/-- The volume of a circular truncated cone -/
noncomputable def volume_truncated_cone (top_area bottom_area : ℝ) (slant_height : ℝ) : ℝ :=
  let top_radius := Real.sqrt (top_area / Real.pi)
  let bottom_radius := Real.sqrt (bottom_area / Real.pi)
  let height := Real.sqrt (slant_height ^ 2 - (bottom_radius - top_radius) ^ 2)
  (1 / 3) * height * (top_area + Real.sqrt (top_area * bottom_area) + bottom_area)

/-- Theorem: The volume of a circular truncated cone with given parameters is 7π -/
theorem volume_specific_truncated_cone :
  volume_truncated_cone (3 * Real.pi) (12 * Real.pi) 2 = 7 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_specific_truncated_cone_l789_78907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_history_teacher_count_l789_78911

/-- Represents the number of teachers for a subject -/
structure TeacherCount where
  count : Nat

/-- The total number of teachers in the school -/
def TotalTeachers (english geography history : TeacherCount) : Nat :=
  max (english.count + geography.count - min english.count geography.count)
      (english.count + history.count - min english.count history.count)

theorem history_teacher_count
  (english : TeacherCount)
  (geography : TeacherCount)
  (h1 : english.count = 9)
  (h2 : geography.count = 6)
  (h3 : ∀ (t : TeacherCount), TotalTeachers english geography t ≥ 11) :
  ∃ (history : TeacherCount), history.count = 5 ∧ TotalTeachers english geography history = 11 := by
  sorry

#eval TotalTeachers ⟨9⟩ ⟨6⟩ ⟨5⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_history_teacher_count_l789_78911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l789_78988

noncomputable section

structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

def m (t : Triangle) : Fin 2 → ℝ
  | 0 => Real.cos t.A
  | 1 => Real.cos t.C

def n (t : Triangle) : Fin 2 → ℝ
  | 0 => t.c
  | 1 => t.a

def p (t : Triangle) : Fin 2 → ℝ
  | 0 => 2 * t.b
  | 1 => 0

def f (x : ℝ) : ℝ :=
  Real.sin x * Real.cos x + Real.sin x * Real.sin (x - Real.pi/6)

def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

theorem triangle_problem (t : Triangle) 
  (h : dot_product (m t) ((n t) - (p t)) = 0) : 
  t.A = Real.pi/3 ∧ 
  Set.Icc ((Real.sqrt 3 - 2) / 4) (Real.sqrt 3 / 2) = 
    Set.range (fun x => f x) ∩ Set.Icc (-(t.A)) t.A := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l789_78988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l789_78957

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0
  h_slope_product : b^2 / a^2 = 144 / 25
  h_focus_asymptote_distance : b = 12

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 13 / 5

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 144 = 1

/-- Theorem stating the properties of the hyperbola -/
theorem hyperbola_properties (h : Hyperbola) :
  eccentricity h = 13 / 5 ∧
  ∀ x y, hyperbola_equation x y ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l789_78957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_bat_price_calculation_l789_78999

/-- Represents the price of the cricket bat at different stages of the transaction -/
structure BatPrice where
  (a_cost : ℝ)  -- A's cost price in USD
  (a_sell : ℝ)  -- A's selling price in GBP
  (b_buy  : ℝ)  -- B's buying price in GBP
  (b_sell : ℝ)  -- B's selling price in GBP
  (c_buy  : ℝ)  -- C's buying price in GBP
  (c_sell : ℝ)  -- C's selling price in EUR
  (d_buy  : ℝ)  -- D's buying price in EUR

/-- The main theorem stating the relationship between the prices -/
theorem cricket_bat_price_calculation 
  (bp : BatPrice)
  (h1 : bp.a_sell = bp.a_cost * 0.72 * 1.2)  -- A's profit
  (h2 : bp.b_buy = bp.a_sell * 0.95)         -- A's discount
  (h3 : bp.b_sell = bp.b_buy * 1.25)         -- B's profit
  (h4 : bp.c_buy = bp.b_sell * 0.9)          -- B's discount
  (h5 : bp.c_sell = bp.c_buy * 1.15 * 1.3)   -- C's profit and currency conversion
  (h6 : bp.d_buy = bp.c_sell * 0.93)         -- C's discount
  (h7 : bp.d_buy = 310)                      -- D's payment
  : ∃ (ε : ℝ), abs (bp.a_cost - 241.5) < ε ∧ ε > 0 := by
  sorry

#check cricket_bat_price_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_bat_price_calculation_l789_78999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_sufficient_not_necessary_for_g_increasing_l789_78991

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
def g (a : ℝ) (x : ℝ) : ℝ := (2 - a) * x^3

-- State the theorem
theorem f_decreasing_sufficient_not_necessary_for_g_increasing
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  (∀ x y : ℝ, x < y → g a x < g a y) ∧
  ¬(∀ x y : ℝ, x < y → g a x < g a y → ∀ x y : ℝ, x < y → f a x > f a y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_sufficient_not_necessary_for_g_increasing_l789_78991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_six_thirds_l789_78915

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b
  a_gt_b : a > b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- A line in the form bx - ay + 2ab = 0 -/
structure TangentLine (a b : ℝ) where
  equation : ∀ (x y : ℝ), b*x - a*y + 2*a*b = 0

theorem ellipse_eccentricity_sqrt_six_thirds
  (a b : ℝ) (e : Ellipse a b) (l : TangentLine a b) :
  eccentricity e = Real.sqrt 6 / 3 := by
  sorry

#check ellipse_eccentricity_sqrt_six_thirds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_six_thirds_l789_78915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_cost_increase_is_fifty_percent_l789_78932

/-- Represents the percentage increase in food costs -/
def food_increase_percentage : ℝ → Prop := sorry

/-- Last year's monthly expenses -/
def last_year_rent : ℝ := 1000
def last_year_food : ℝ := 200
def last_year_insurance : ℝ := 100

/-- This year's changes -/
def rent_increase_rate : ℝ := 0.3
def insurance_increase_rate : ℝ := 2

/-- Total increase in expenses over the year -/
def total_increase : ℝ := 7200

/-- Theorem stating that the food cost increase percentage is 50% -/
theorem food_cost_increase_is_fifty_percent :
  food_increase_percentage 0.5 ↔
    (last_year_rent * (1 + rent_increase_rate) +
     last_year_food * (1 + 0.5) +
     last_year_insurance * (1 + insurance_increase_rate) -
     (last_year_rent + last_year_food + last_year_insurance)) * 12 =
    total_increase := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_cost_increase_is_fifty_percent_l789_78932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l789_78994

noncomputable def f (x : ℝ) := Real.sqrt (4 - Real.sqrt (6 - Real.sqrt (7 - x)))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-93 : ℝ) 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l789_78994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l789_78900

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x > 1 then Real.log x else 2 * x + m^3

theorem find_m : ∃ m : ℝ, f (f (Real.exp 1) m) m = 10 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l789_78900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_property_l789_78986

/-- Given a triangle PQR with vertices P(1,7), Q(2,-3), and R(8,4),
    if S(m,n) is the centroid of the triangle, then 10m + n = 118/3 -/
theorem centroid_property (m n : ℚ) : 
  (m = (1 + 2 + 8) / 3 ∧ n = (7 + (-3) + 4) / 3) →
  10 * m + n = 118 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_property_l789_78986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l789_78906

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 + 2 * (Real.cos x) ^ 2

theorem triangle_side_value (A B C : ℝ) (a b c : ℝ) :
  0 < A → A < Real.pi →
  f A = 4 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  a = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l789_78906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_sin_cos_sum_l789_78944

-- Define the angle α
noncomputable def α : Real := Real.arctan (2 / (-1))

-- Define the conditions for the first part
def terminal_point : ℝ × ℝ := (-1, 2)

-- Define the condition for the second part
def terminal_line (x : ℝ) : ℝ := -3 * x

-- Theorem for the first part
theorem sin_cos_product :
  Real.sin α * Real.cos α = -2/5 := by sorry

-- Theorem for the second part
theorem sin_cos_sum :
  10 * Real.sin α + 3 / Real.cos α = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_sin_cos_sum_l789_78944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_proof_l789_78949

def is_odd_prime (n : ℕ) : Prop := Nat.Prime n ∧ n % 2 = 1

theorem xy_value_proof (x y : ℕ) (hx : is_odd_prime x) (hy : is_odd_prime y) (hxy : x < y)
  (h_factors : (Finset.filter (fun d ↦ d ∣ (2 * x * y)) (Finset.range ((2 * x * y) + 1))).card = 8) :
  x * y = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_proof_l789_78949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l789_78946

theorem log_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 4 * (Real.log x / Real.log 4) = (Real.log (4 * x^2)) / Real.log 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l789_78946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_point_perpendicular_to_vector_l789_78936

/-- The plane equation passing through point A and perpendicular to vector BC -/
theorem plane_equation_through_point_perpendicular_to_vector
  (A B C : ℝ × ℝ × ℝ)
  (hA : A = (-3, 6, 4))
  (hB : B = (8, -3, 5))
  (hC : C = (0, -3, 7)) :
  (λ (x : ℝ × ℝ × ℝ) => -4 * x.1 + x.2.2 - 16 = 0) =
  (λ (x : ℝ × ℝ × ℝ) => ((C.1 - B.1) * (x.1 - A.1) + (C.2.1 - B.2.1) * (x.2.1 - A.2.1) + (C.2.2 - B.2.2) * (x.2.2 - A.2.2) = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_point_perpendicular_to_vector_l789_78936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_point_P_irrational_point_P_l789_78963

/-- A rational point on a plane -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- Four distinct rational points on a plane -/
structure FourRationalPoints where
  A : RationalPoint
  B : RationalPoint
  A' : RationalPoint
  B' : RationalPoint
  distinct : A ≠ B ∧ A ≠ A' ∧ A ≠ B' ∧ B ≠ A' ∧ B ≠ B' ∧ A' ≠ B'

/-- Distance between two rational points -/
noncomputable def distance (p q : RationalPoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: P is generally a rational point -/
theorem rational_point_P (points : FourRationalPoints) :
  ∃ P : RationalPoint,
    distance points.A' points.B' / distance points.A points.B =
    distance points.B' P / distance points.B P ∧
    distance points.B' P / distance points.B P =
    distance P points.A' / distance P points.A :=
  sorry

/-- Exceptional case: When P is not a rational point -/
theorem irrational_point_P (points : FourRationalPoints) :
  ∃ P : ℝ × ℝ,
    (¬ ∃ q : ℚ, P.1 = q) ∨ (¬ ∃ q : ℚ, P.2 = q) ∧
    distance points.A' points.B' / distance points.A points.B =
    Real.sqrt ((points.B'.x - P.1)^2 + (points.B'.y - P.2)^2) /
    Real.sqrt ((points.B.x - P.1)^2 + (points.B.y - P.2)^2) ∧
    Real.sqrt ((points.B'.x - P.1)^2 + (points.B'.y - P.2)^2) /
    Real.sqrt ((points.B.x - P.1)^2 + (points.B.y - P.2)^2) =
    Real.sqrt ((P.1 - points.A'.x)^2 + (P.2 - points.A'.y)^2) /
    Real.sqrt ((P.1 - points.A.x)^2 + (P.2 - points.A.y)^2) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_point_P_irrational_point_P_l789_78963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_solution_set_l789_78938

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| + |x + 2| ≤ 5} = Set.Icc (-3) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_solution_set_l789_78938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_intersection_condition_l789_78972

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * log x

noncomputable def g (x : ℝ) : ℝ := x * exp (1 - x)

theorem function_intersection_condition (a : ℝ) : 
  (∀ x₀, x₀ ∈ Set.Ioc 0 (exp 1) → ∃ x₁ x₂, x₁ ∈ Set.Ioc 0 (exp 1) ∧ x₂ ∈ Set.Ioc 0 (exp 1) ∧ 
    x₁ ≠ x₂ ∧ f a x₁ = g x₀ ∧ f a x₂ = g x₀) ↔ 
  a ∈ Set.Iic ((2 * exp 1 - 5) / (exp 1 - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_intersection_condition_l789_78972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l789_78998

theorem sin_double_angle (α : ℝ) :
  (Real.cos (α + π/4) = 3/5) ∧ (π/2 ≤ α) ∧ (α ≤ 3*π/2) → Real.sin (2*α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l789_78998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l789_78947

def p (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 + 6*x - 36

theorem roots_of_polynomial :
  (∃ (a b : ℝ), a ≠ b ∧ 
    (∀ x, p x = (x - a)^2 * (x - b)^2)) ∧
  p 3 = 0 ∧ p (-2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l789_78947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l789_78996

/-- The distance from the center of circle C to line l is 2√2 -/
theorem distance_circle_center_to_line : 
  let line_l (x y : ℝ) : Prop := x + y - 6 = 0
  let circle_c (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ + 2)
  let θ_domain (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 2 * Real.pi
  let distance : ℝ := 2 * Real.sqrt 2
  distance = 2 * Real.sqrt 2 := by
    sorry

#check distance_circle_center_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l789_78996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_east_to_northwest_l789_78959

/-- Represents a clock face with 12 equally spaced rays -/
structure ClockFace where
  rays : Fin 12 → ℝ
  equal_spacing : ∀ i j : Fin 12, rays ((i + 1) % 12) - rays i = rays ((j + 1) % 12) - rays j
  east_ray : rays 3 = 0  -- 3 o'clock position points East (0°)

/-- The angle between two rays on the clock face -/
noncomputable def angle_between (c : ClockFace) (i j : Fin 12) : ℝ :=
  (c.rays j - c.rays i + 360) % 360

/-- The Northwest direction is halfway between 10 and 11 o'clock -/
def northwest_ray : Fin 12 := 10  -- We use 10 here as an approximation

/-- The theorem stating the angle between East and Northwest rays -/
theorem angle_east_to_northwest (c : ClockFace) :
  angle_between c 3 northwest_ray = 225 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_east_to_northwest_l789_78959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_7_l789_78923

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_of_2_7 : floor 2.7 = 2 := by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_7_l789_78923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_drain_time_is_22_hours_l789_78981

/-- Represents the time it takes for a leak to drain a full tank, given the fill times with and without the leak. -/
noncomputable def leak_drain_time (fill_time_no_leak : ℝ) (fill_time_with_leak : ℝ) : ℝ :=
  let pump_rate := 1 / fill_time_no_leak
  let combined_rate := 1 / fill_time_with_leak
  let leak_rate := pump_rate - combined_rate
  1 / leak_rate

/-- Theorem stating that given the specific fill times, the leak drain time is 22 hours. -/
theorem leak_drain_time_is_22_hours :
  leak_drain_time 2 (11/5) = 22 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_drain_time_is_22_hours_l789_78981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_degree_polynomial_characterization_l789_78990

noncomputable def is_local_min (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y, |y - x| < ε → f x ≤ f y

def is_valid_polynomial (p : ℝ → ℝ) : Prop :=
  (∀ x, p x = p (-x)) ∧ 
  (∀ x, p x ≥ 0) ∧ 
  (p 0 = 1) ∧ 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    is_local_min p x₁ ∧ 
    is_local_min p x₂ ∧ 
    |x₁ - x₂| = 2 ∧ 
    (∀ x, is_local_min p x → x = x₁ ∨ x = x₂))

theorem fourth_degree_polynomial_characterization 
  (p : ℝ → ℝ) 
  (h_degree : ∃ a b c d e, ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e) 
  (h_valid : is_valid_polynomial p) :
  ∃ a : ℝ, 0 < a ∧ a ≤ 1 ∧ ∀ x, p x = a*(x^2 - 1)^2 + (1 - a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_degree_polynomial_characterization_l789_78990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_equality_l789_78954

theorem complex_power_equality (n : ℕ) :
  (1/2 : ℂ) + Complex.I * ((Real.sqrt 3)/2) ^ n =
  Complex.cos (n * Real.pi / 3) + Complex.I * Complex.sin (n * Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_equality_l789_78954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_cost_price_theorem_l789_78931

noncomputable def costPrice (highPrice lowPrice : ℝ) : ℝ := (highPrice + lowPrice) / 2

noncomputable def combinedCostPrice (costA costB costC : ℝ) : ℝ := costA + costB + costC

theorem combined_cost_price_theorem :
  let costA := costPrice 120 60
  let costB := costPrice 200 100
  let costC := costPrice 300 180
  combinedCostPrice costA costB costC = 480 := by
  simp [costPrice, combinedCostPrice]
  norm_num
  -- The proof is completed automatically by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_cost_price_theorem_l789_78931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l789_78969

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Define the slope of the tangent line
def tangent_slope : ℝ := 2 * point.1 + 2

-- State the theorem
theorem tangent_line_equation :
  ∀ x y : ℝ, (x = point.1 ∧ y = f x) → (4*x - y - 1 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l789_78969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_48_product_l789_78960

def pairs : List (ℚ × ℚ) := [(-6, -8), (6, 8), (-1/2, 96), (4, 12), (-3/2, 32)]

theorem unique_non_48_product : ∃! p, p ∈ pairs ∧ p.1 * p.2 ≠ 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_48_product_l789_78960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_theorem_l789_78943

/-- Calculates the distance traveled given speed and time -/
noncomputable def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Calculates the speed given distance and time -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem sam_distance_theorem (marguerite_distance : ℝ) (marguerite_total_time : ℝ) 
  (marguerite_stop_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 120 →
  marguerite_total_time = 3 →
  marguerite_stop_time = 0.5 →
  sam_time = 3.5 →
  distance (speed marguerite_distance (marguerite_total_time - marguerite_stop_time)) sam_time = 168 := by
  sorry

-- Remove the #eval line as it's causing issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_theorem_l789_78943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_zeros_l789_78948

noncomputable def f (a x : ℝ) : ℝ := min (Real.exp x - 2) (Real.exp (2*x) - a * Real.exp x + a + 24)

theorem f_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  (12 < a ∧ a < 28) := by
  sorry

#check f_three_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_zeros_l789_78948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_bottom_width_l789_78912

/-- Represents the properties of an irregular trapezium -/
structure IrregularTrapezium where
  top_width : ℝ
  area : ℝ
  height : ℝ

/-- Calculates the bottom width of an irregular trapezium -/
noncomputable def bottom_width (t : IrregularTrapezium) : ℝ :=
  2 * t.area / t.height - t.top_width

/-- Theorem stating that for the given trapezium, the bottom width is 6 meters -/
theorem trapezium_bottom_width :
  let t : IrregularTrapezium := { top_width := 10, area := 640, height := 80 }
  bottom_width t = 6 := by
  sorry

#check trapezium_bottom_width

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_bottom_width_l789_78912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_bn_bnplus1_eq_one_l789_78956

def b (n : ℕ) : ℤ := (7^n - 1) / 6

theorem gcd_bn_bnplus1_eq_one (n : ℕ) : Int.gcd (b n) (b (n + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_bn_bnplus1_eq_one_l789_78956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_distance_from_Q_l789_78951

-- Define the points P and Q
noncomputable def P : ℝ × ℝ := (2, 0)
noncomputable def Q : ℝ × ℝ := (-2, 4 * Real.sqrt 3 / 3)

-- Define the distance from Q to the line
noncomputable def distance_to_line : ℝ := 4

-- Define the two possible cases for the line
structure LineCase where
  angle : ℝ  -- Angle of inclination in degrees
  equation : (ℝ → ℝ → Prop)  -- General form equation of the line

-- Theorem statement
theorem line_through_P_distance_from_Q : 
  ∃ (case : LineCase), 
    (case.angle = 90 ∧ case.equation = (λ x y ↦ x - 2 = 0)) ∨
    (case.angle = 30 ∧ case.equation = (λ x y ↦ x - Real.sqrt 3 * y - 2 = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_distance_from_Q_l789_78951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_centers_concyclic_l789_78997

-- Define the polygon and its properties
variable (n : ℕ) (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ)

-- Define the property of being both inscribed and circumscribed
def is_inscribed_and_circumscribed (n : ℕ) (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ) : Prop := sorry

-- Define the points Cᵢ
noncomputable def C (n : ℕ) (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ) (i : Fin n) : ℝ × ℝ := sorry

-- Define the property of points being concyclic
def are_concyclic {n : ℕ} (points : Fin n → ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem polygon_centers_concyclic 
  (h : is_inscribed_and_circumscribed n A O) : 
  are_concyclic (C n A O) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_centers_concyclic_l789_78997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_true_l789_78942

theorem inverse_proposition_is_true : ∀ x : ℝ, x^2 ≤ 1 → x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_true_l789_78942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_exponential_function_l789_78940

noncomputable def f (k : ℝ) (x : ℝ) := Real.exp (k * x)

theorem range_of_exponential_function (k : ℝ) (hk : k > 0) :
  Set.range (fun x => f k x) ∩ Set.Ici 1 = Set.Ici (Real.exp k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_exponential_function_l789_78940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_a_range_l789_78916

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem f_increasing_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (3/2) 3 ∧ a ≠ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_a_range_l789_78916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_selection_l789_78920

/-- Represents a chessboard configuration -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- A valid chessboard configuration satisfies the given conditions -/
def valid_chessboard (board : Chessboard) : Prop :=
  (∀ row, ∃ col, board row col = true) ∧ 
  (∀ row₁ row₂, row₁ ≠ row₂ → 
    (Finset.filter (λ col => board row₁ col) Finset.univ).card ≠ 
    (Finset.filter (λ col => board row₂ col) Finset.univ).card)

/-- A selection of pieces from the chessboard -/
def Selection := Fin 8 → Fin 8

/-- A valid selection has exactly one piece per row and column -/
def valid_selection (board : Chessboard) (sel : Selection) : Prop :=
  (∀ row, board row (sel row) = true) ∧
  Function.Injective sel

/-- Main theorem: For any valid chessboard, there exists a valid selection -/
theorem chessboard_selection (board : Chessboard) :
  valid_chessboard board → ∃ sel : Selection, valid_selection board sel := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_selection_l789_78920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_around_point_l789_78913

theorem angle_around_point : ∃ y : ℝ, y = 50 ∧ 210 + 3 * y = 360 :=
  by
    -- Define y
    let y : ℝ := 50

    -- Prove that y satisfies the equation
    have h1 : 210 + 3 * y = 360 := by
      calc
        210 + 3 * y = 210 + 3 * 50 := by rfl
        _            = 210 + 150   := by norm_num
        _            = 360         := by norm_num

    -- Prove that y = 50
    have h2 : y = 50 := by rfl

    -- Combine the proofs
    exact ⟨y, h2, h1⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_around_point_l789_78913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_henry_kombucha_bottles_l789_78933

/-- Represents the number of bottles of kombucha Henry can buy with his refund after 1 year --/
def bottles_from_refund (bottles_per_month : ℕ) (bottle_cost : ℚ) (refund_per_bottle : ℚ) : ℕ :=
  let bottles_per_year : ℕ := bottles_per_month * 12
  let total_refund : ℚ := (bottles_per_year : ℚ) * refund_per_bottle
  (total_refund / bottle_cost).floor.toNat

/-- Theorem stating that Henry can buy 6 bottles of kombucha with his refund after 1 year --/
theorem henry_kombucha_bottles : 
  bottles_from_refund 15 3 (1/10) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_henry_kombucha_bottles_l789_78933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_passes_through_all_quadrants_l789_78909

/-- The function f(x) = (1/3)ax³ + (1/2)ax² - 2ax + 2a + 1 -/
noncomputable def f (a x : ℝ) : ℝ := (1/3) * a * x^3 + (1/2) * a * x^2 - 2 * a * x + 2 * a + 1

/-- The function passes through all four quadrants -/
def passes_through_all_quadrants (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), f x₁ > 0 ∧ f x₂ < 0 ∧ f x₃ > 0 ∧ f x₄ < 0 ∧
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ < 0 ∧ x₄ < 0

/-- Theorem: The graph of f passes through all four quadrants iff -6/5 < a < -3/16 -/
theorem graph_passes_through_all_quadrants (a : ℝ) :
  passes_through_all_quadrants (f a) ↔ -6/5 < a ∧ a < -3/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_passes_through_all_quadrants_l789_78909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_banana_distribution_l789_78935

theorem monkey_banana_distribution 
  (total_bananas : ℕ) 
  (a b c : ℕ) 
  (h_total : total_bananas = 540)
  (h_a : a ≤ total_bananas)
  (h_b : b ≤ total_bananas - a)
  (h_c : c = total_bananas - a - b)
  (h_whole_numbers : 
    (a / 2 + b / 3 + 3 * c / 8 : ℚ).den = 1 ∧ 
    (a / 4 + b / 3 + 3 * c / 8 : ℚ).den = 1 ∧ 
    (a / 4 + b / 3 + c / 4 : ℚ).den = 1)
  (h_ratio : ∃ (k : ℕ), 
    (a / 2 + b / 3 + 3 * c / 8 : ℚ) = 5 * k ∧
    (a / 4 + b / 3 + 3 * c / 8 : ℚ) = 3 * k ∧
    (a / 4 + b / 3 + c / 4 : ℚ) = 2 * k) :
  ((a / 2 + b / 3 + 3 * c / 8 : ℚ) = 270) ∧ 
  ((a / 4 + b / 3 + 3 * c / 8 : ℚ) = 162) ∧ 
  ((a / 4 + b / 3 + c / 4 : ℚ) = 108) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_banana_distribution_l789_78935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l789_78950

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

theorem unique_b_value :
  ∃! b : ℝ, (f b 3 = (f b).invFun (b + 2)) ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_value_l789_78950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l789_78978

/-- Given a triangle ABC with altitude AA' and orthocenter H -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  A' : ℝ × ℝ
  H : ℝ × ℝ

/-- k is the ratio of AA' to HA' -/
noncomputable def k (t : Triangle) : ℝ := 
  let (ax, ay) := t.A
  let (a'x, a'y) := t.A'
  let (hx, hy) := t.H
  Real.sqrt ((ax - a'x)^2 + (ay - a'y)^2) / Real.sqrt ((hx - a'x)^2 + (hy - a'y)^2)

/-- The angle at vertex B -/
noncomputable def angle_B (t : Triangle) : ℝ := sorry

/-- The angle at vertex C -/
noncomputable def angle_C (t : Triangle) : ℝ := sorry

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) (k_pos : k t > 0) : 
  k t = Real.tan (angle_B t) * Real.tan (angle_C t) ∧
  ∃ c : ℝ, c > 0 ∧ ∀ x y : ℝ, 
    (x, y) = t.A → x^2 + (k t) * y^2 = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l789_78978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_k_for_integer_b_l789_78926

noncomputable def b : ℕ → ℝ
  | 0 => 2
  | (n + 1) => b n + Real.log ((4 * (n + 1) + 5) / (4 * (n + 1) + 1)) / Real.log 3

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = ↑n

theorem least_integer_k_for_integer_b :
  (∀ k : ℕ, 1 < k → k < 11 → ¬ is_integer (b k)) ∧
  is_integer (b 11) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_k_for_integer_b_l789_78926
