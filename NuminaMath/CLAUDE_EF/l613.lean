import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_even_nor_odd_l613_61346

noncomputable def f (x : ℝ) : ℝ := ⌈x⌉ + (1/3)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_neither_even_nor_odd : ¬(is_even f) ∧ ¬(is_odd f) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_even_nor_odd_l613_61346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_and_g_monotonicity_l613_61340

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1

theorem f_upper_bound_and_g_monotonicity :
  (∃ c : ℝ, c = -1 ∧ ∀ x > 0, f x ≤ 2 * x + c ∧
    ∀ c' < c, ∃ x > 0, f x > 2 * x + c') ∧
  (∀ a > 0, ∀ x > 0, x ≠ a →
    let g := fun x => (f x - f a) / (x - a)
    (deriv g) x < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_and_g_monotonicity_l613_61340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_estate_investment_l613_61379

def total_investment : ℚ := 225000
def real_estate_ratio : ℚ := 6
def gold_ratio : ℚ := 1

theorem real_estate_investment :
  ∃ (gold : ℚ) (real_estate : ℚ),
    gold * real_estate_ratio = real_estate ∧
    gold + real_estate = total_investment ∧
    Int.floor real_estate = 192857 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_estate_investment_l613_61379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_consumption_relation_l613_61304

theorem tax_consumption_relation (original_tax : ℝ) (original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  let new_tax := 0.82 * original_tax
  let new_revenue := 0.943 * (original_tax * original_consumption)
  ∃ (increase_percentage : ℝ),
    new_tax * (original_consumption * (1 + increase_percentage / 100)) = new_revenue ∧ 
    abs (increase_percentage - 15.06) < 0.01 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_consumption_relation_l613_61304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_geometry_l613_61395

noncomputable def square_side (x : ℝ) : ℝ := 20 * Real.sqrt 5 * x

noncomputable def P (x : ℝ) : ℝ × ℝ := (square_side x, square_side x / 2)
noncomputable def Q (x : ℝ) : ℝ × ℝ := (square_side x / 2, square_side x)

def A : ℝ × ℝ := (0, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem square_geometry (x : ℝ) (h : x > 0) :
  distance A (P x) = 50 * x ∧
  distance (P x) (Q x) = 10 * Real.sqrt 10 * x ∧
  (let m := (P x).2 - (Q x).2 / ((P x).1 - (Q x).1)
   let b := (P x).2 - m * (P x).1
   (|m * A.1 - A.2 + b|) / Real.sqrt (m^2 + 1) = 15 * Real.sqrt 10 * x) ∧
  Real.sin (Real.arccos ((distance A (P x))^2 + (distance A (Q x))^2 - (distance (P x) (Q x))^2) /
    (2 * distance A (P x) * distance A (Q x))) = 3/5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_geometry_l613_61395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_of_lines_iff_discriminant_zero_l613_61303

/-- A quadratic equation in two variables represents a pair of lines if and only if its discriminant is zero. -/
theorem pair_of_lines_iff_discriminant_zero (l : ℝ) :
  (∃ (a b c d e f : ℝ), ∀ (x y : ℝ),
    l * x^2 + 4 * x * y + y^2 - 4 * x - 2 * y - 3 = 0 ↔
    (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0)) ↔
  l = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_of_lines_iff_discriminant_zero_l613_61303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_algorithms_exist_l613_61323

/-- Represents an algorithm --/
structure Algorithm where
  steps : List String
  result : Option String

/-- Characteristics of a valid algorithm --/
class ValidAlgorithm (A : Algorithm) where
  finite : A.steps.length < ω
  definite : ∀ s ∈ A.steps, s ≠ ""
  produces_output : A.result.isSome

/-- A problem type that can be solved by algorithms --/
structure ProblemType where
  name : String

/-- A function that checks if an algorithm solves a given problem type --/
def solves (A : Algorithm) (P : ProblemType) : Prop := sorry

/-- Theorem stating that there can be multiple algorithms for a problem type --/
theorem multiple_algorithms_exist (P : ProblemType) :
  ∃ (A B : Algorithm), ∃ (_ : ValidAlgorithm A) (_ : ValidAlgorithm B),
    A ≠ B ∧ solves A P ∧ solves B P := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_algorithms_exist_l613_61323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l613_61393

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop :=
  (x - 5)^2 / 7^2 - (y - 12)^2 / 10^2 = 1

/-- The focus with larger x-coordinate -/
noncomputable def focus : ℝ × ℝ := (5 + Real.sqrt 149, 12)

/-- Theorem: The focus with larger x-coordinate for the given hyperbola has coordinates (5+√149, 12) -/
theorem hyperbola_focus :
  ∀ x y : ℝ, hyperbola x y → 
  ∀ f_x f_y : ℝ, ((x - 5)^2 / 7^2 - (y - 12)^2 / 10^2 = 1 → 
  (f_x - 5)^2 - (f_y - 12)^2 = 149 → 
  f_x ≥ 5) → 
  (f_x, f_y) = focus :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l613_61393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l613_61349

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, Real.sqrt 3 * t)

-- Define the curve C
def curve_C (ρ θ : ℝ) : Prop :=
  4 * ρ^2 * Real.cos (2 * θ) - 4 * ρ * Real.sin θ + 3 = 0

-- Theorem statement
theorem intersection_distance :
  -- The polar equation of line l is θ = π/3
  (∀ (ρ : ℝ), ∃ (t : ℝ), line_l t = (ρ * Real.cos (π/3), ρ * Real.sin (π/3))) ∧
  -- The distance between the two intersection points is 3
  (∃ (ρ₁ ρ₂ : ℝ), 
    curve_C ρ₁ (π/3) ∧ 
    curve_C ρ₂ (π/3) ∧ 
    ρ₁ ≠ ρ₂ ∧ 
    |ρ₁ - ρ₂| = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l613_61349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l613_61315

noncomputable def original_function (x : ℝ) : ℝ := Real.cos (2 * x + 4 * Real.pi / 5)

noncomputable def transformed_function (x : ℝ) : ℝ := 4 * Real.cos (4 * x - Real.pi / 5)

theorem function_transformation :
  ∀ x : ℝ, transformed_function x = 4 * original_function (x / 2 - Real.pi / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l613_61315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_l613_61391

/-- A function representing f(x) = (x+a)ln((2x-1)/(2x+1)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.log ((2*x - 1) / (2*x + 1))

/-- The theorem stating that if f is even, then a must be 0 -/
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x, x > (1/2) ∨ x < -(1/2) → f a x = f a (-x)) → a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_l613_61391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_statement_l613_61351

theorem exactly_one_correct_statement : ∃! n : ℕ, n = 1 ∧
  (-- Statement 1
   (¬ ∃ x₀ : ℝ, x₀^2 - x₀ > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)) ∧
  (-- Statement 2
   ¬ (∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q))) ∧
  (-- Statement 3
   ¬ (∀ x : ℝ, (x^2 = 1 → x = 1) ↔ (x^2 = 1 → x ≠ 1))) ∧
  (-- Statement 4
   ¬ (∀ x : ℝ, (x^2 - 5*x - 6 = 0 → x = -1) ∧ ¬(x = -1 → x^2 - 5*x - 6 = 0))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_statement_l613_61351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_intersection_not_necessarily_tangent_l613_61398

-- Define a hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ

-- Define a line
structure Line where
  m : ℝ
  c : ℝ

-- Define the concept of intersection
def intersects (l : Line) (h : Hyperbola) : Prop :=
  ∃ x y : ℝ, y = l.m * x + l.c ∧ (x^2 / h.a^2) - (y^2 / h.b^2) = 1

-- Define the concept of tangency
def is_tangent (l : Line) (h : Hyperbola) : Prop :=
  ∃! p : ℝ × ℝ, intersects l h ∧ p.1^2 / h.a^2 - p.2^2 / h.b^2 = 1

-- Define the concept of being parallel to an asymptote
def parallel_to_asymptote (l : Line) (h : Hyperbola) : Prop :=
  l.m = h.b / h.a ∨ l.m = -h.b / h.a

-- Theorem statement
theorem one_intersection_not_necessarily_tangent :
  ∃ l : Line, ∃ h : Hyperbola, (∃! p : ℝ × ℝ, intersects l h) ∧ ¬(is_tangent l h) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_intersection_not_necessarily_tangent_l613_61398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_ordering_l613_61320

-- Define the set of coins
inductive Coin : Type
  | A | B | C | D | E | F
  deriving BEq, Repr

-- Define the relation "is above"
def IsAbove : Coin → Coin → Prop := sorry

-- State the problem conditions
axiom d_above_b : IsAbove Coin.D Coin.B
axiom d_above_c : IsAbove Coin.D Coin.C
axiom c_above_a : IsAbove Coin.C Coin.A
axiom a_above_f : IsAbove Coin.A Coin.F
axiom c_above_e : IsAbove Coin.C Coin.E
axiom f_above_e : IsAbove Coin.F Coin.E
axiom d_top : ∀ x : Coin, x ≠ Coin.D → IsAbove Coin.D x
axiom e_bottom : ∀ x : Coin, x ≠ Coin.E → IsAbove x Coin.E

-- Define the correct ordering
def CorrectOrder : List Coin := [Coin.D, Coin.B, Coin.C, Coin.A, Coin.F, Coin.E]

-- State the theorem
theorem coin_ordering :
  ∀ (order : List Coin),
    (∀ x : Coin, x ∈ order) ∧
    (∀ x y : Coin, x ≠ y → (IsAbove x y ↔ order.indexOf x < order.indexOf y)) →
    order = CorrectOrder := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_ordering_l613_61320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_of_symmetry_equation_l613_61334

/-- Definition of a point reflection with respect to a line -/
def reflect_point (l : Set (ℝ × ℝ)) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The equation of the line of symmetry for two points -/
theorem line_of_symmetry_equation (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) 
  (hA : A = (4, 5))
  (hB : B = (-2, 7))
  (hsym : B = reflect_point l A) : 
  l = {p : ℝ × ℝ | 3 * p.1 - p.2 + 3 = 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_of_symmetry_equation_l613_61334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l613_61318

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- Theorem: The eccentricity of the given ellipse is 1/2 -/
theorem ellipse_eccentricity : 
  ∀ (a b : ℝ), a > b ∧ b > 0 ∧ 
  (4 / a^2 = 1) ∧ 
  (1 / a^2 + (9/4) / b^2 = 1) →
  eccentricity a b = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l613_61318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_ratio_sum_l613_61361

-- Define a circle
def Circle : Type := ℝ × ℝ → Prop

-- Define points on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop := c p

-- Define an acute triangle
def AcuteTriangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define perpendicular lines
def Perpendicular (l1 l2 : ℝ × ℝ → ℝ × ℝ → Prop) : Prop := sorry

-- Define the ratio of lengths
noncomputable def LengthRatio (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem circle_triangle_ratio_sum 
  (c : Circle) 
  (A B C X Y Z D E F : ℝ × ℝ) 
  (h_acute : AcuteTriangle A B C)
  (h_on_circle : PointOnCircle c A ∧ PointOnCircle c B ∧ PointOnCircle c C ∧ 
                 PointOnCircle c X ∧ PointOnCircle c Y ∧ PointOnCircle c Z)
  (h_perp_AX : Perpendicular (λ p q ↦ p = A ∨ p = X) (λ p q ↦ p = B ∨ p = C))
  (h_perp_BY : Perpendicular (λ p q ↦ p = B ∨ p = Y) (λ p q ↦ p = A ∨ p = C))
  (h_perp_CZ : Perpendicular (λ p q ↦ p = C ∨ p = Z) (λ p q ↦ p = A ∨ p = B))
  (h_D_on_BC : sorry)
  (h_E_on_AC : sorry)
  (h_F_on_AB : sorry) :
  LengthRatio A X D + LengthRatio B Y E + LengthRatio C Z F = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_ratio_sum_l613_61361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_passes_through_point_l613_61381

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 3) + 1

theorem inverse_function_passes_through_point
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ g : ℝ → ℝ, Function.LeftInverse g (f a) ∧ Function.RightInverse g (f a) ∧ g 2 = 3 := by
  sorry

#check inverse_function_passes_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_passes_through_point_l613_61381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_PA_value_l613_61339

noncomputable def line_L (t : ℝ) : ℝ × ℝ := (2 + t, 2 - 2*t)

noncomputable def curve_C (θ : ℝ) : ℝ := 2 / Real.sqrt (1 + 3 * (Real.cos θ)^2)

noncomputable def angle_l_L : ℝ := Real.pi / 3

theorem max_PA_value :
  ∃ (max_PA : ℝ),
    max_PA = (2 * Real.sqrt 15 / 15) * (2 * Real.sqrt 2 - 6) ∧
    ∀ (θ : ℝ) (P : ℝ × ℝ),
      P.1^2 + P.2^2 / 4 = 1 →  -- P is on curve C
      ∃ (A : ℝ × ℝ),
        (∃ t, A = line_L t) ∧  -- A is on line L
        Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) ≤ max_PA :=
by sorry

#check max_PA_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_PA_value_l613_61339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_stretching_spring_l613_61311

/-- Work done by a force stretching a spring -/
theorem work_done_stretching_spring (F : ℝ) (x : ℝ) : 
  F = 10 →  -- Force in Newtons
  x = 0.02 →  -- Stretch in meters (2 cm = 0.02 m)
  (∫ t in (0 : ℝ)..x, F * (t / x)) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_stretching_spring_l613_61311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_fixed_point_l613_61368

/-- The complex number around which the rotation occurs -/
noncomputable def c : ℂ := -3 * Real.sqrt 2 - 15 * Complex.I

/-- The rotation function -/
noncomputable def f (z : ℂ) : ℂ := ((1 + Complex.I * Real.sqrt 2) * z + (3 * Real.sqrt 2 - 12 * Complex.I)) / 3

/-- Theorem stating that c is the fixed point of the rotation function f -/
theorem rotation_fixed_point : f c = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_fixed_point_l613_61368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_equal_l613_61352

/-- Represents a cell in the table -/
structure Cell where
  value : ℝ
  underlined : Bool

/-- Represents the 10x10 table -/
def Table := Fin 10 → Fin 10 → Cell

/-- Checks if a number is the largest in its row -/
def isLargestInRow (t : Table) (i j : Fin 10) : Prop :=
  ∀ k : Fin 10, (t i j).value ≥ (t i k).value

/-- Checks if a number is the smallest in its column -/
def isSmallestInColumn (t : Table) (i j : Fin 10) : Prop :=
  ∀ k : Fin 10, (t i j).value ≤ (t k j).value

/-- Checks if a cell is underlined -/
def isUnderlined (t : Table) (i j : Fin 10) : Prop :=
  (t i j).underlined

/-- Checks if all underlined numbers are underlined exactly twice -/
def allUnderlinedTwice (t : Table) : Prop :=
  ∀ i j : Fin 10, isUnderlined t i j →
    (∃! k : Fin 10, isUnderlined t i k ∧ k ≠ j) ∧
    (∃! k : Fin 10, isUnderlined t k j ∧ k ≠ i)

/-- The main theorem -/
theorem all_numbers_equal (t : Table)
  (h1 : ∀ i j : Fin 10, isLargestInRow t i j → isUnderlined t i j)
  (h2 : ∀ i j : Fin 10, isSmallestInColumn t i j → isUnderlined t i j)
  (h3 : allUnderlinedTwice t) :
  ∀ i j k l : Fin 10, (t i j).value = (t k l).value :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_equal_l613_61352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_f_range_of_t_no_solution_l613_61316

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (3^x - 3) / Real.log 10

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := f x - Real.log (3^x + 3) / Real.log 10

-- Theorem for the domain of f
theorem domain_of_f :
  ∀ x : ℝ, f x ∈ Set.univ ↔ x ∈ Set.Ioi 1 :=
by sorry

-- Theorem for the range of f
theorem range_of_f :
  Set.range f = Set.univ :=
by sorry

-- Theorem for the range of t when h(x) > t has no solution
theorem range_of_t_no_solution :
  ∀ t : ℝ, (∀ x : ℝ, h x ≤ t) ↔ t ∈ Set.Ici 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_f_range_of_t_no_solution_l613_61316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_unit_circle_l613_61355

theorem max_distance_from_unit_circle (z : ℂ) : 
  Complex.abs z = 1 → ∀ w : ℂ, Complex.abs (z - (1 + Complex.I)) ≤ Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_unit_circle_l613_61355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_selecting_A_and_B_l613_61358

theorem probability_of_selecting_A_and_B (n k total_combinations favorable_combinations : ℕ) :
  n = 5 →
  k = 3 →
  total_combinations = Nat.choose n k →
  favorable_combinations = Nat.choose (n - 2) (k - 2) →
  (favorable_combinations : ℚ) / (total_combinations : ℚ) = 3 / 10 := by
  intros hn hk htotal hfavorable
  sorry

#check probability_of_selecting_A_and_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_selecting_A_and_B_l613_61358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l613_61367

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Add a case for 0 to avoid missing cases error
  | 1 => 1
  | n + 1 => Real.sqrt (2 + a n)

theorem a_general_term : ∀ n : ℕ, n > 0 → a n = 2 * Real.cos (π / (3 * 2^(n - 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l613_61367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_addition_problem_solution_l613_61363

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition problem SIX + SIX = ELEVEN -/
def AdditionProblem (S I X E L V N : Digit) : Prop :=
  S ≠ I ∧ S ≠ X ∧ S ≠ E ∧ S ≠ L ∧ S ≠ V ∧ S ≠ N ∧
  I ≠ X ∧ I ≠ E ∧ I ≠ L ∧ I ≠ V ∧ I ≠ N ∧
  X ≠ E ∧ X ≠ L ∧ X ≠ V ∧ X ≠ N ∧
  E ≠ L ∧ E ≠ V ∧ E ≠ N ∧
  L ≠ V ∧ L ≠ N ∧
  V ≠ N ∧
  (S.val * 100 + I.val * 10 + X.val) * 2 = E.val * 100000 + L.val * 10000 + E.val * 1000 + V.val * 100 + E.val * 10 + N.val

theorem addition_problem_solution :
  ∀ (S I X E L V N : Digit),
    AdditionProblem S I X E L V N →
    S.val = 8 →
    X.val % 2 = 1 →
    I.val = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_addition_problem_solution_l613_61363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2017_l613_61306

def my_sequence (a : ℕ → ℤ) : Prop :=
  (a 1 = 2) ∧ (a 2 = 3) ∧ (∀ n, a (n + 2) = a (n + 1) - a n)

theorem sequence_2017 (a : ℕ → ℤ) (h : my_sequence a) : a 2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2017_l613_61306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l613_61389

/-- Represents a parabola with equation x = -1/4 * y^2 -/
structure Parabola where
  equation : ℝ → ℝ
  symmetric_x : Bool

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop := sorry

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Theorem stating that the directrix of the given parabola is x = 1 -/
theorem parabola_directrix (p : Parabola) 
  (h_eq : p.equation = fun y ↦ -1/4 * y^2) 
  (h_sym : p.symmetric_x = true) : 
  directrix p = fun x ↦ x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l613_61389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_implies_m_value_l613_61338

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- The circle equation -/
def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + m = 0

/-- The length of the line segment -/
noncomputable def segment_length : ℝ := 2 * Real.sqrt 5 / 5

/-- The theorem statement -/
theorem intersection_length_implies_m_value (m : ℝ) :
  (∃ x y : ℝ, line_eq x y ∧ circle_eq x y m) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    line_eq x₁ y₁ ∧ circle_eq x₁ y₁ m ∧
    line_eq x₂ y₂ ∧ circle_eq x₂ y₂ m ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = segment_length) →
  m = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_implies_m_value_l613_61338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l613_61328

/-- Represents a person with their lawn size and mower speed -/
structure Person where
  name : String
  lawn_size : ℝ
  mower_speed : ℝ

/-- Calculates the time taken to mow a lawn -/
noncomputable def mowing_time (p : Person) : ℝ := p.lawn_size / p.mower_speed

/-- Theorem stating that Beth finishes mowing first -/
theorem beth_finishes_first (andy beth carlos : Person)
  (h1 : andy.lawn_size = 3 * beth.lawn_size)
  (h2 : andy.lawn_size = 4 * carlos.lawn_size)
  (h3 : carlos.mower_speed = (1/4) * andy.mower_speed)
  (h4 : beth.mower_speed = (1/2) * andy.mower_speed)
  : mowing_time beth < mowing_time andy ∧ mowing_time beth < mowing_time carlos := by
  sorry

#check beth_finishes_first

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l613_61328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_savings_theorem_l613_61370

/-- Calculates the discounted price of a flight --/
noncomputable def discounted_price (original_price : ℝ) (discount_percent : ℝ) : ℝ :=
  original_price * (1 - discount_percent / 100)

/-- Theorem: The difference between the cheaper and more expensive discounted flight prices is $90 --/
theorem flight_savings_theorem (delta_price united_price : ℝ) 
  (delta_discount united_discount : ℝ) : 
  delta_price = 850 ∧ united_price = 1100 ∧ 
  delta_discount = 20 ∧ united_discount = 30 →
  min (discounted_price delta_price delta_discount) 
      (discounted_price united_price united_discount) = 680 ∧
  max (discounted_price delta_price delta_discount) 
      (discounted_price united_price united_discount) = 770 ∧
  max (discounted_price delta_price delta_discount) 
      (discounted_price united_price united_discount) -
  min (discounted_price delta_price delta_discount) 
      (discounted_price united_price united_discount) = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_savings_theorem_l613_61370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_outside_l613_61378

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line
def my_line (a b x y : ℝ) : Prop := a*x + b*y = 1

-- Define what it means for a point to be outside the circle
def outside_circle (a b : ℝ) : Prop := a^2 + b^2 > 1

-- State the theorem
theorem intersection_implies_outside (a b : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ 
    my_circle x₁ y₁ ∧ my_circle x₂ y₂ ∧ 
    my_line a b x₁ y₁ ∧ my_line a b x₂ y₂) →
  outside_circle a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_outside_l613_61378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_parallel_properties_l613_61372

/-- Two lines are different if they are not equal -/
def different_lines (m n : Line) : Prop := m ≠ n

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is perpendicular to another line -/
def line_perp_line (l₁ l₂ : Line) : Prop := sorry

/-- A plane is perpendicular to another plane -/
def plane_perp_plane (p₁ p₂ : Plane) : Prop := sorry

/-- A plane is parallel to another plane -/
def plane_parallel_plane (p₁ p₂ : Plane) : Prop := sorry

/-- A line is parallel to another line -/
def line_parallel_line (l₁ l₂ : Line) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry

theorem perpendicular_and_parallel_properties 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : different_lines m n)
  (h_diff_planes : different_planes α β) :
  (line_perp_line m n ∧ line_perp_plane m α ∧ line_perp_plane n β → plane_perp_plane α β) ∧
  (plane_parallel_plane α β ∧ line_perp_plane m α ∧ line_parallel_plane n β → line_perp_line m n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_and_parallel_properties_l613_61372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_path_is_straight_line_l613_61374

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  (((t.A.1 + t.B.1 + t.C.1) / 3), ((t.A.2 + t.B.2 + t.C.2) / 3))

-- Define the path of vertex C
noncomputable def C_path (t : ℝ) : ℝ × ℝ :=
  (t / Real.sqrt 2, t / Real.sqrt 2)

-- Theorem statement
theorem centroid_path_is_straight_line (A B : ℝ × ℝ) :
  ∃ (m b : ℝ), ∀ t : ℝ,
    let C := C_path t
    let triangle := Triangle.mk A B C
    centroid triangle = (m * t + b, m * t + b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_path_is_straight_line_l613_61374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_parallelepiped_volume_sum_mnp_l613_61301

/-- The volume of the set of points inside or within one unit of a rectangular parallelepiped -/
noncomputable def extended_volume (length width height : ℝ) : ℝ :=
  let box_volume := length * width * height
  let surface_area := 2 * (length * width + length * height + width * height)
  let edge_length := length + width + height
  box_volume + surface_area + (Real.pi * edge_length) + (4 * Real.pi / 3)

/-- Theorem stating the volume for a 2x3x6 rectangular parallelepiped -/
theorem extended_parallelepiped_volume :
  ∃ (m n p : ℕ), 
    m > 0 ∧ n > 0 ∧ p > 0 ∧
    Nat.Coprime n p ∧
    extended_volume 2 3 6 = (m + n * Real.pi) / p ∧
    m = 324 ∧ n = 37 ∧ p = 3 :=
by
  sorry

/-- The sum of m, n, and p is 364 -/
theorem sum_mnp : 
  ∀ (m n p : ℕ), 
    m > 0 → n > 0 → p > 0 →
    Nat.Coprime n p →
    extended_volume 2 3 6 = (m + n * Real.pi) / p →
    m = 324 → n = 37 → p = 3 →
    m + n + p = 364 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_parallelepiped_volume_sum_mnp_l613_61301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_divisible_by_seven_l613_61332

theorem count_four_digit_divisible_by_seven : ∃ n : ℕ, n = 1286 := by
  -- Define the range of four-digit numbers
  let min_four_digit : ℕ := 1000
  let max_four_digit : ℕ := 9999

  -- Define the predicate for four-digit numbers divisible by 7
  let is_valid : ℕ → Prop := λ n => 
    min_four_digit ≤ n ∧ n ≤ max_four_digit ∧ n % 7 = 0

  -- The count of numbers satisfying the predicate
  let count := (Finset.range (max_four_digit - min_four_digit + 1)).filter 
    (λ n => is_valid (n + min_four_digit)) |>.card

  -- Prove that this count equals 1286
  use count
  sorry

#eval (Finset.range 9000).filter (λ n => 1000 ≤ n + 1000 ∧ n + 1000 ≤ 9999 ∧ (n + 1000) % 7 = 0) |>.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_divisible_by_seven_l613_61332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_sqrt3x_3y_1_l613_61344

noncomputable def angle_of_inclination (a b c : ℝ) : ℝ :=
  if a = 0 then
    if b > 0 then Real.pi / 2 else -Real.pi / 2
  else if a > 0 then
    Real.arctan (-b / a)
  else
    Real.pi + Real.arctan (-b / a)

/-- Theorem stating that the angle of inclination of the line √3x + 3y + 1 = 0 is 5π/6 -/
theorem angle_of_inclination_sqrt3x_3y_1 :
  angle_of_inclination (Real.sqrt 3) 3 1 = 5 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_sqrt3x_3y_1_l613_61344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_minimum_l613_61360

theorem triangle_inequality_minimum (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b > c → b + c > a → c + a > b →
  a + b + c = 12 → 
  let M := a / (b + c - a) + 4 * b / (c + a - b) + 9 * c / (a + b - c)
  M ≥ 2.875 ∧ ∃ (a' b' c' : ℝ), M = 2.875 := by
  sorry

#check triangle_inequality_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_minimum_l613_61360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_mean_after_correction_l613_61326

/-- The actual mean of the remaining students in a class with a tabulation error. -/
theorem actual_mean_after_correction (n : ℕ) (h : n > 15) : 
  (10 : ℚ) * n - (16 : ℚ) * 15 / (n - 15) = (10 * n - 240) / (n - 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_mean_after_correction_l613_61326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l613_61335

noncomputable def f (x : ℝ) := Real.sin (2 * x)

noncomputable def g (x : ℝ) := Real.sin (4 * x - 2 * Real.pi / 3)

theorem function_transformation (x : ℝ) : 
  g x = Real.sin (4 * (x - Real.pi / 6)) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l613_61335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l613_61302

noncomputable def f (x : ℝ) : ℝ := (4 * x^4 - 20 * x^2 + 18) / (x^4 - 5 * x^2 + 4)

noncomputable def solution_set : Set ℝ :=
  Set.Ioo (-2 : ℝ) (-Real.sqrt 3) ∪
  Set.Ioo (Real.sqrt 3) 2 ∪
  Set.Ioo (-Real.sqrt 2) (-1) ∪
  Set.Ioo 1 (Real.sqrt 2)

theorem inequality_solution :
  ∀ x : ℝ, f x < 3 ↔ x ∈ solution_set := by
  sorry

#check inequality_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l613_61302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_magnitude_l613_61353

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, b.1 = t * a.1 ∧ b.2 = t * a.2

theorem collinear_vectors_magnitude (a b : ℝ × ℝ) (h : collinear a b) :
  a = (1, 2) → b.1 = -2 → ‖(3 : ℝ) • a + b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_magnitude_l613_61353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_line_segments_l613_61348

/-- Represents a dot in the grid -/
inductive Dot
| Red
| Blue

/-- Represents a line segment in the grid -/
inductive LineSegment
| Red
| Blue
| Yellow

/-- Represents the 16x16 grid -/
def Grid := Fin 16 → Fin 16 → Dot

/-- The total number of dots in the grid -/
def totalDots : Nat := 16 * 16

/-- The number of red dots -/
def redDots : Nat := 133

/-- The number of red dots on the boundary -/
def redBoundaryDots : Nat := 32

/-- The number of red dots at the corners -/
def redCornerDots : Nat := 2

/-- The number of yellow line segments -/
def yellowLineSegments : Nat := 196

/-- Function to count the number of blue line segments in the grid -/
def number_of_blue_line_segments (grid : Grid) : Nat := sorry

theorem blue_line_segments (grid : Grid) :
  (totalDots = 16 * 16) →
  (redDots = 133) →
  (redBoundaryDots = 32) →
  (redCornerDots = 2) →
  (yellowLineSegments = 196) →
  (∃ (n : Nat), n = 134 ∧ n = number_of_blue_line_segments grid) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_line_segments_l613_61348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_path_area_ratio_l613_61380

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a particle moving along the edges of a triangle -/
structure Particle where
  position : Point
  speed : ℝ

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- The region enclosed by the path of the midpoint -/
def enclosed_region (t : Triangle) (p1 p2 : Particle) : Set Point := sorry

/-- The area of a set of points -/
noncomputable def area (s : Set Point) : ℝ := sorry

/-- The theorem to be proved -/
theorem midpoint_path_area_ratio 
  (t : Triangle) 
  (p1 p2 : Particle) 
  (h1 : t.A = Point.mk (1/2) (Real.sqrt 3/2)) 
  (h2 : t.B = Point.mk 0 0) 
  (h3 : t.C = Point.mk 1 0)
  (h4 : p1.position = t.B)
  (h5 : p2.position = Point.mk (3/4) (Real.sqrt 3/4))
  (h6 : p1.speed = p2.speed)
  (h7 : is_equilateral t) :
  (area (enclosed_region t p1 p2)) / (area {p | ∃ (x : ℝ) (y : ℝ), p = Point.mk x y ∧ 
    (x = t.A.x ∧ y = t.A.y) ∨ (x = t.B.x ∧ y = t.B.y) ∨ (x = t.C.x ∧ y = t.C.y)}) = 1/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_path_area_ratio_l613_61380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l613_61394

/-- The standard equation of an ellipse given specific conditions -/
theorem ellipse_standard_equation :
  ∀ (a b : ℝ) (P : ℝ × ℝ),
  a > 0 ∧ b > 0 →  -- Ensure positive semi-axes lengths
  P.1 = 3 * Real.sqrt 2 ∧ P.2 = 4 →  -- Point P coordinates
  (∃ (c : ℝ), c > 0 ∧ c < a ∧
    Real.sqrt ((P.1 - c)^2 + P.2^2) + Real.sqrt ((P.1 + c)^2 + P.2^2) = 12) →  -- Sum of distances to foci
  P.1^2 / a^2 + P.2^2 / b^2 = 1 →  -- Point P satisfies ellipse equation
  a = 6 ∧ b = 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l613_61394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_c_gain_is_2_5_percent_l613_61310

/-- Represents a product with its weight and cost information -/
structure Product where
  name : String
  actual_weight : ℚ
  claimed_weight : ℚ
  cost : ℚ
  deriving Repr

/-- Calculates the percentage gain for a product -/
def percentage_gain (p : Product) : ℚ :=
  ((p.claimed_weight - p.actual_weight) / p.claimed_weight) * 100

/-- The problem statement -/
theorem product_c_gain_is_2_5_percent :
  let product_c : Product := {
    name := "C",
    actual_weight := 195/1,
    claimed_weight := 200/1,
    cost := 15
  }
  percentage_gain product_c = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_c_gain_is_2_5_percent_l613_61310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_to_line_l613_61362

noncomputable def line_l (t : ℝ) : ℝ × ℝ := 
  (-4 + (Real.sqrt 2 / 2) * t, -2 + (Real.sqrt 2 / 2) * t)

noncomputable def curve_C (θ : ℝ) : ℝ := 
  2 * Real.cos θ

noncomputable def point_P (θ : ℝ) : ℝ × ℝ := 
  (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ)

theorem max_distance_point_to_line :
  ∃ (d : ℝ), d = (3 * Real.sqrt 2 / 2) + 1 ∧
  ∀ (θ : ℝ), ∃ (t : ℝ), 
    Real.sqrt ((point_P θ).1 - (line_l t).1)^2 + ((point_P θ).2 - (line_l t).2)^2 ≤ d ∧
    ∃ (θ₀ : ℝ), Real.sqrt ((point_P θ₀).1 - (line_l t).1)^2 + ((point_P θ₀).2 - (line_l t).2)^2 = d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_to_line_l613_61362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_after_six_iterations_l613_61324

def sequence_sum (n : ℕ) : ℕ := 
  let a := λ i => 1 + 3 * (i - 1)
  let b := λ i => 2 + 3 * (i - 1)
  let sum_a := (Finset.range 8).sum a
  let sum_b := (Finset.range 8).sum b
  sum_a + sum_b - 1

theorem sequence_sum_after_six_iterations :
  sequence_sum 6 = 191 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_after_six_iterations_l613_61324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_72_l613_61369

/-- Represents the speed of a train given crossing times and platform length -/
noncomputable def train_speed (platform_length : ℝ) (platform_crossing_time : ℝ) (man_crossing_time : ℝ) : ℝ :=
  let train_speed_mps := platform_length / (platform_crossing_time - man_crossing_time)
  train_speed_mps * 3.6

/-- Theorem stating that the train's speed is 72 km/h given the problem conditions -/
theorem train_speed_is_72 : 
  train_speed 340 35 18 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_72_l613_61369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_Q_equation_l613_61309

/-- The circle C with equation x^2 + y^2 = 25 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 25}

/-- The point M with coordinates (-2, 3) -/
def M : ℝ × ℝ := (-2, 3)

/-- A line passing through point M -/
def line_through_M (a b : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - M.1) * b = (p.2 - M.2) * a}

/-- The set of points Q, where Q is the intersection of tangents to C at points A and B,
    and A and B are the intersections of C with a line passing through M -/
def locus_Q : Set (ℝ × ℝ) := {Q | ∃ (a b : ℝ) (A B : ℝ × ℝ),
  A ∈ C ∧ B ∈ C ∧
  A ∈ line_through_M a b ∧ B ∈ line_through_M a b ∧
  (A.1 - Q.1) * A.1 + (A.2 - Q.2) * A.2 = 0 ∧
  (B.1 - Q.1) * B.1 + (B.2 - Q.2) * B.2 = 0}

/-- The theorem stating that the locus of Q is given by the equation 2x - 3y + 25 = 0 -/
theorem locus_Q_equation : locus_Q = {Q : ℝ × ℝ | 2 * Q.1 - 3 * Q.2 + 25 = 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_Q_equation_l613_61309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_through_A_l613_61345

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point A
def point_A : ℝ × ℝ := (2, 4)

-- Define the tangent line property
def is_tangent_line (a b c : ℝ) : Prop :=
  ∀ x y, my_circle x y → (a*x + b*y + c = 0 → 
    ∀ x' y', my_circle x' y' → (x', y') ≠ (x, y) → a*x' + b*y' + c ≠ 0)

-- Define the property of a line passing through point A
def passes_through_A (a b c : ℝ) : Prop :=
  a * point_A.1 + b * point_A.2 + c = 0

-- Theorem statement
theorem tangent_lines_through_A :
  (∀ a b c, is_tangent_line a b c ∧ passes_through_A a b c →
    (a = 1 ∧ b = 0 ∧ c = -2) ∨ (a = 3 ∧ b = -4 ∧ c = 10)) ∧
  (is_tangent_line 1 0 (-2) ∧ passes_through_A 1 0 (-2)) ∧
  (is_tangent_line 3 (-4) 10 ∧ passes_through_A 3 (-4) 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_through_A_l613_61345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_zero_at_sqrt_two_l613_61343

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x) / (Real.log a) - 4 / x

-- State the theorem
theorem max_value_zero_at_sqrt_two (a : ℝ) (h1 : a > 1) :
  (∀ x ∈ Set.Icc 1 2, f a x ≤ 0) ∧ (∃ x ∈ Set.Icc 1 2, f a x = 0) ↔ a = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_zero_at_sqrt_two_l613_61343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_800_pointed_stars_l613_61350

/-- A regular n-pointed star. -/
structure RegularStar (n : ℕ) where
  /-- The number of vertices in the star. -/
  vertices : ℕ
  /-- The turning number at each vertex. -/
  m : ℕ
  /-- The turning number is less than half the number of vertices. -/
  m_lt_half : m < n / 2
  /-- The turning number and the number of vertices are coprime. -/
  coprime : Nat.Coprime m n
  /-- No three vertices are collinear. -/
  no_collinear : True
  /-- All line segments intersect at a point other than an endpoint. -/
  all_intersect : True
  /-- All angles at the vertices are congruent. -/
  congruent_angles : True
  /-- All line segments are congruent. -/
  congruent_segments : True
  /-- The path turns counterclockwise at an angle less than 180 degrees at each vertex. -/
  ccw_turn : True

/-- The number of non-similar regular 800-pointed stars. -/
def num_non_similar_800_stars : ℕ := 158

/-- Theorem stating that the number of non-similar regular 800-pointed stars is 158. -/
theorem count_800_pointed_stars :
  (Finset.filter (λ m => Nat.Coprime m 800 ∧ m < 400) (Finset.range 400)).card = num_non_similar_800_stars :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_800_pointed_stars_l613_61350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cart_length_correct_l613_61314

/-- Represents a right-angled hallway with a given width -/
structure Hallway where
  width : ℝ

/-- Represents a rectangular cart with given width and length -/
structure Cart where
  width : ℝ
  length : ℝ

/-- Checks if a cart can pass through a hallway -/
def canPass (h : Hallway) (c : Cart) : Prop :=
  c.width < h.width ∧ c.length ≤ 3 * Real.sqrt 2

/-- The maximum length of a cart that can pass through the hallway -/
noncomputable def maxCartLength (h : Hallway) (cWidth : ℝ) : ℝ := 3 * Real.sqrt 2

theorem max_cart_length_correct (h : Hallway) (cWidth : ℝ) 
  (hWidth : h.width = 1.5) (hcWidth : cWidth = 1) :
  ∀ (c : Cart), c.width = cWidth → 
    (canPass h c ↔ c.length ≤ maxCartLength h cWidth) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cart_length_correct_l613_61314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_from_three_lines_l613_61364

/-- A line in 3D space -/
structure Line3D where
  -- Placeholder for line representation
  dummy : Unit

/-- A plane in 3D space -/
structure Plane3D where
  -- Placeholder for plane representation
  dummy : Unit

/-- Determines if two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry

/-- Determines the plane formed by two intersecting lines -/
noncomputable def plane_from_lines (l1 l2 : Line3D) : Plane3D :=
  sorry

/-- The set of planes determined by three lines -/
noncomputable def planes_from_three_lines (l1 l2 l3 : Line3D) : Finset Plane3D :=
  sorry

/-- The maximum number of planes determined by three intersecting lines is 3 -/
theorem max_planes_from_three_lines (l1 l2 l3 : Line3D) 
  (h12 : intersect l1 l2) (h13 : intersect l1 l3) (h23 : intersect l2 l3) :
  (planes_from_three_lines l1 l2 l3).card ≤ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_from_three_lines_l613_61364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_rotation_vs_sphere_l613_61376

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Chord of a parabola passing through the focus -/
structure Chord (para : Parabola) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  passes_through_focus : (para.focus.1 ∈ Set.Icc P.1 Q.1) ∧ (para.focus.2 ∈ Set.Icc P.2 Q.2)

/-- Projection of a chord onto the directrix -/
def projection (para : Parabola) (c : Chord para) : ℝ × ℝ := sorry

/-- Surface area of rotation around directrix -/
noncomputable def surface_area_rotation (para : Parabola) (c : Chord para) : ℝ := sorry

/-- Surface area of sphere -/
noncomputable def surface_area_sphere (diameter : ℝ) : ℝ := sorry

/-- Main theorem -/
theorem parabola_chord_rotation_vs_sphere (para : Parabola) (c : Chord para) :
  let proj := projection para c
  let S1 := surface_area_rotation para c
  let S2 := surface_area_sphere (Real.sqrt ((proj.1 - proj.2)^2))
  S1 ≥ S2 ∧ (S1 = S2 ↔ c.P.2 = c.Q.2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_rotation_vs_sphere_l613_61376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_primes_l613_61371

def x : ℕ → ℕ → ℕ
  | a, 0 => a
  | a, n + 1 => 2 * x a n + 1

def y (a n : ℕ) : ℕ := 2^(x a n) - 1

theorem max_consecutive_primes (a : ℕ) (h : a > 0) :
  ∃ k : ℕ, k = 2 ∧
  (∀ n : ℕ, n ≤ k → Nat.Prime (y a n)) ∧
  (∀ m : ℕ, m > k → ¬(∀ n : ℕ, n ≤ m → Nat.Prime (y a n))) := by
  sorry

#check max_consecutive_primes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_primes_l613_61371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l613_61336

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := 2 / (3 * x + c)

noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem inverse_function_condition (c : ℝ) : 
  (∀ x : ℝ, f c (f_inv x) = x) → c = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l613_61336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_for_a_n_le_2021_l613_61392

def a : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 2 => if ((n + 2) % 3 = 0) then a (n + 1) else a (n + 1) + 5

theorem max_k_for_a_n_le_2021 : 
  (∀ n : ℕ, n ≤ 1211 → a n ≤ 2021) ∧ 
  (∃ n : ℕ, n > 1211 ∧ a n > 2021) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_for_a_n_le_2021_l613_61392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_delta_three_l613_61312

/-- Definition of the Delta operation -/
noncomputable def Delta (c d : ℝ) : ℝ := (c + d + 1) / (1 + c * d)

/-- Theorem stating that 4 Delta 3 equals 8/13 -/
theorem four_delta_three :
  Delta 4 3 = 8 / 13 := by
  -- Unfold the definition of Delta
  unfold Delta
  -- Simplify the expression
  simp [add_assoc, mul_comm]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_delta_three_l613_61312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l613_61300

/-- A right triangle PQR in the Cartesian plane -/
structure RightTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1 / 2) * base * height

/-- The y-coordinate of a point on line RP given its x-coordinate -/
noncomputable def yOnLineRP (x : ℝ) : ℝ := -2/3 * x + 4

/-- The area of the right part of the triangle divided by the line x = b -/
noncomputable def rightPartArea (b : ℝ) : ℝ := triangleArea (6 - b) (yOnLineRP b)

/-- The theorem stating that b = 3 divides the triangle into two equal areas -/
theorem equal_area_division (t : RightTriangle) (h1 : t.P = (0, 4)) 
    (h2 : t.Q = (0, 0)) (h3 : t.R = (6, 0)) :
    ∃ b : ℝ, b = 3 ∧ rightPartArea b = triangleArea 6 4 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l613_61300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_bag_system_l613_61396

/-- Represents the number of bags carried by the big horse -/
def x : ℕ := sorry

/-- Represents the number of bags carried by the small horse -/
def y : ℕ := sorry

/-- The condition where if the big horse gives 1 bag to the small horse, they have the same number of bags -/
axiom condition1 : x - 1 = y + 1

/-- The condition where if the small horse gives 1 bag to the big horse, the big horse has twice as many bags as the small horse -/
axiom condition2 : x + 1 = 2 * (y - 1)

/-- Theorem stating that the system of equations correctly describes the relationship between x and y -/
theorem horse_bag_system : 
  (x - 1 = y + 1) ∧ (x + 1 = 2 * (y - 1)) := by
  constructor
  · exact condition1
  · exact condition2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_bag_system_l613_61396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_train_meeting_probability_l613_61321

/-- The train's earliest arrival time in minutes after 1:30 PM -/
noncomputable def train_earliest_arrival : ℝ := 30

/-- The train's latest arrival time in minutes after 1:30 PM -/
noncomputable def train_latest_arrival : ℝ := 90

/-- John's earliest arrival time in minutes after 1:30 PM -/
noncomputable def john_earliest_arrival : ℝ := 0

/-- John's latest arrival time in minutes after 1:30 PM -/
noncomputable def john_latest_arrival : ℝ := 90

/-- The duration the train waits at the station in minutes -/
noncomputable def train_wait_time : ℝ := 15

/-- The probability of John arriving while the train is at the station -/
noncomputable def probability_john_meets_train : ℝ := 2 / 9

theorem john_train_meeting_probability :
  probability_john_meets_train = 
    (((train_latest_arrival - train_earliest_arrival) * train_wait_time) / 2) /
    ((john_latest_arrival - john_earliest_arrival) * (train_latest_arrival - train_earliest_arrival)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_train_meeting_probability_l613_61321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contractor_engagement_l613_61388

/-- Represents the number of days a contractor was engaged, given their daily pay, fine for absence, 
    number of absent days, and total earnings. -/
def days_engaged (daily_pay : ℚ) (daily_fine : ℚ) (absent_days : ℕ) (total_earnings : ℚ) : ℕ :=
  ((total_earnings + daily_fine * absent_days) / daily_pay).floor.toNat

/-- Proves that the contractor was engaged for 18 days given the problem conditions. -/
theorem contractor_engagement :
  days_engaged 25 7.5 12 360 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contractor_engagement_l613_61388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l613_61354

theorem expression_evaluation (y : ℝ) : 
  (2^(3*y + 2)) / ((1/4 : ℝ) + (1/2 : ℝ)) = (4 * 2^(3*y + 2)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l613_61354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l613_61382

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := 2 * x^2 + 6

-- Define the directrix
noncomputable def directrix : ℝ := 47/8

-- Theorem statement
theorem parabola_directrix :
  ∀ (x y : ℝ), y = parabola x → 
  ∃ (p : ℝ × ℝ), 
    (p.1 - x)^2 + (p.2 - y)^2 = (y - directrix)^2 ∧
    ∀ (q : ℝ × ℝ), q ≠ p → (q.1 - x)^2 + (q.2 - y)^2 > (y - directrix)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l613_61382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l613_61331

/-- Definition of the function g(n) -/
noncomputable def g (n : ℕ) : ℝ :=
  (6 + 4 * Real.sqrt 6) / 12 * ((2 + Real.sqrt 6) / 3) ^ n +
  (6 - 4 * Real.sqrt 6) / 12 * ((2 - Real.sqrt 6) / 3) ^ n

/-- Theorem stating the relationship between g(n+2) and g(n) -/
theorem g_relation (n : ℕ) : g (n + 2) - g n = (-1 + 6 * Real.sqrt 6) / 9 * g n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l613_61331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_closed_l613_61385

-- Define the universe set (State Duma members)
variable {U : Type}

-- Define what a faction system is
def FactionSystem (S : Set (Set U)) : Prop :=
  ∀ A B, A ∈ S → B ∈ S → (Aᶜ ∪ Bᶜ)ᶜ ∈ S

-- State the theorem
theorem union_closed {S : Set (Set U)} (h : FactionSystem S) :
  ∀ A B, A ∈ S → B ∈ S → A ∪ B ∈ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_closed_l613_61385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l613_61357

/-- The angle of the hour hand at time t (in hours) on a 12-hour clock -/
noncomputable def hour_hand_angle (t : ℝ) : ℝ := (t % 12) * 30

/-- The angle of the minute hand at time t (in hours) on a 12-hour clock -/
noncomputable def minute_hand_angle (t : ℝ) : ℝ := (t % 1) * 360

/-- The smaller angle between two angles on a circle -/
noncomputable def smaller_angle (a b : ℝ) : ℝ :=
  min (abs (a - b)) (360 - abs (a - b))

/-- The theorem stating that the smaller angle between the hands of a 12-hour clock at 3:40 pm is 130.0 degrees -/
theorem clock_angle_at_3_40 :
  smaller_angle (hour_hand_angle (15 + 40/60)) (minute_hand_angle (15 + 40/60)) = 130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l613_61357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l613_61377

theorem min_value_sin_cos (x : ℝ) (h : x ∈ Set.Icc (-π/6) (π/2)) :
  (Real.sin x + 1) * (Real.cos x + 1) ≥ (2 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l613_61377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_2b_is_sqrt3_l613_61384

noncomputable def a : ℝ × ℝ := (Real.cos (15 * Real.pi / 180), Real.sin (15 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (75 * Real.pi / 180), Real.sin (75 * Real.pi / 180))

theorem magnitude_a_minus_2b_is_sqrt3 :
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_minus_2b_is_sqrt3_l613_61384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_C_equals_two_three_power_set_of_C_l613_61317

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 6}

-- Define set C
def C : Set ℤ := {x : ℤ | (x : ℝ) ∈ A ∩ B}

-- Theorem statements
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 ≤ x ∧ x < 6} := by sorry

theorem C_equals_two_three : C = {2, 3} := by sorry

theorem power_set_of_C : 𝒫 C = {∅, {2}, {3}, {2, 3}} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_C_equals_two_three_power_set_of_C_l613_61317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l613_61313

noncomputable def m (a x : ℝ) : ℝ × ℝ := (2 * a * Real.cos x, Real.sin x)
noncomputable def n (b x : ℝ) : ℝ × ℝ := (Real.cos x, b * Real.cos x)

noncomputable def f (a b x : ℝ) : ℝ := 
  (m a x).1 * (n b x).1 + (m a x).2 * (n b x).2 - Real.sqrt 3 / 2

def is_highest_point_closest_to_y_axis (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y ∧ ∀ x' : ℝ, |x'| < |x| → f x' < y

theorem problem_solution (a b : ℝ) : 
  (∃ x, f a b x = Real.sqrt 3 / 2) → 
  is_highest_point_closest_to_y_axis (f a b) (π/12) 1 →
  (∃ φ : ℝ, φ > 0 ∧ 
    (∀ x, Real.sin x = f a b (x/2 - φ)) ∧ 
    (∀ φ' : ℝ, φ' > 0 → (∀ x, Real.sin x = f a b (x/2 - φ')) → φ' ≥ φ)) →
  (∀ x, f a b x = Real.sin (2*x + π/3)) ∧ 
  (∃! φ : ℝ, φ > 0 ∧ 
    (∀ x, Real.sin x = f a b (x/2 - φ)) ∧ 
    φ = 5*π/6) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l613_61313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_l613_61365

-- Define the given conditions
noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := a^x + x - b

-- Theorem statement
theorem zero_of_f_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 0 ∧ f x = 0 :=
by
  sorry

-- Additional lemmas that might be useful for the proof
lemma a_gt_one : a > 1 := by sorry
lemma b_lt_one : 0 < b ∧ b < 1 := by sorry

lemma f_monotone_increasing : StrictMono f := by sorry

lemma f_neg_one_lt_zero : f (-1) < 0 := by sorry
lemma f_zero_gt_zero : f 0 > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_l613_61365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_full_range_of_m_l613_61305

/-- The cube function -/
def f (x : ℝ) : ℝ := x^3

/-- Theorem stating the range of m given the conditions -/
theorem range_of_m (θ : ℝ) (h1 : 0 ≤ θ) (h2 : θ < π/4) (m : ℝ) 
  (h3 : ∀ θ, 0 ≤ θ → θ < π/4 → f (m * Real.tan θ) + f (1 - m) > 0) : 
  m < 1 := by
  sorry

/-- Corollary stating the full range of m -/
theorem full_range_of_m (m : ℝ) 
  (h : ∀ θ, 0 ≤ θ → θ < π/4 → f (m * Real.tan θ) + f (1 - m) > 0) : 
  m ∈ Set.Iio 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_full_range_of_m_l613_61305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_negative_sum_l613_61347

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x

-- State the theorem
theorem m_range_for_negative_sum (m : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f x = x^3 + Real.sin x) →
  (1 - m ∈ Set.Ioo (-1 : ℝ) 1) →
  (1 - m^2 ∈ Set.Ioo (-1 : ℝ) 1) →
  f (1 - m) + f (1 - m^2) < 0 →
  m ∈ Set.Ioo 1 (Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_negative_sum_l613_61347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l613_61322

theorem min_value_of_function (x : ℝ) : (2 : ℝ)^x + (2 : ℝ)^(2-x) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l613_61322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_l613_61387

noncomputable def first_three_scores : List ℚ := [78, 85, 92]
def fourth_score : ℚ := 95

noncomputable def average (scores : List ℚ) : ℚ :=
  (scores.sum) / scores.length

theorem average_increase :
  average (fourth_score :: first_three_scores) - average first_three_scores = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_l613_61387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_score_l613_61359

/-- Sarah's bowling score -/
def sarah_score : ℝ := sorry

/-- Greg's bowling score -/
def greg_score : ℝ := sorry

/-- Jane's bowling score -/
def jane_score : ℝ := sorry

/-- Sarah's score is 50 points more than Greg's -/
axiom sarah_greg_diff : sarah_score = greg_score + 50

/-- The average of Sarah's and Greg's scores is 110 -/
axiom sarah_greg_avg : (sarah_score + greg_score) / 2 = 110

/-- Jane's score is the average of Sarah's and Greg's scores -/
axiom jane_score_def : jane_score = (sarah_score + greg_score) / 2

/-- Theorem: Jane's score is 110 -/
theorem janes_score : jane_score = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_score_l613_61359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_line_equation_l613_61341

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3*x + 6) / x^2

noncomputable def f' (x : ℝ) : ℝ := 3/x^2 - 12/x^3

def x₀ : ℝ := 3

noncomputable def y₀ : ℝ := f x₀

noncomputable def m_normal : ℝ := -1 / (f' x₀)

noncomputable def normal_line (x : ℝ) : ℝ := m_normal * (x - x₀) + y₀

theorem normal_line_equation :
  ∀ x : ℝ, normal_line x = 9*x - 79/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_line_equation_l613_61341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l613_61390

/-- The area of a quadrilateral given its diagonal and two offsets -/
noncomputable def quadrilateral_area (d h₁ h₂ : ℝ) : ℝ := (1/2) * d * (h₁ + h₂)

/-- Theorem: The area of a quadrilateral with diagonal 20 cm and offsets 5 cm and 4 cm is 90 cm² -/
theorem quadrilateral_area_example : quadrilateral_area 20 5 4 = 90 := by
  -- Unfold the definition of quadrilateral_area
  unfold quadrilateral_area
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l613_61390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_cube_volume_l613_61319

/-- The volume of a cube after removing smaller cubes from each face -/
theorem remaining_cube_volume :
  let original_edge_length : ℝ := 3
  let small_cube_edge_length : ℝ := 1
  let num_faces : ℕ := 6
  let original_volume := original_edge_length ^ (3 : ℕ)
  let small_cube_volume := small_cube_edge_length ^ (3 : ℕ)
  let total_removed_volume := (num_faces : ℝ) * small_cube_volume
  original_volume - total_removed_volume = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_cube_volume_l613_61319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_extrema_l613_61329

theorem cos_sum_extrema :
  ∃ (max min : ℝ), max = 3/2 ∧ min = 0 ∧
  (∀ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 → x + y + z = 4*π/3 →
    Real.cos x + Real.cos y + Real.cos z ≤ max ∧ Real.cos x + Real.cos y + Real.cos z ≥ min) ∧
  (∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 4*π/3 ∧ Real.cos x + Real.cos y + Real.cos z = max) ∧
  (∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 4*π/3 ∧ Real.cos x + Real.cos y + Real.cos z = min) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_extrema_l613_61329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_calculation_l613_61337

theorem test_score_calculation (total_questions correct_points incorrect_deduction correct_answers : ℕ)
  (h1 : total_questions = 30) (h2 : correct_points = 20) 
  (h3 : incorrect_deduction = 5) (h4 : correct_answers = 19) :
  correct_answers * correct_points - (total_questions - correct_answers) * incorrect_deduction = 325 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_calculation_l613_61337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absent_student_percentage_l613_61308

theorem absent_student_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (boys_absent_fraction : ℚ) (girls_absent_fraction : ℚ) :
  total_students = 120 →
  boys = 72 →
  girls = 48 →
  boys_absent_fraction = 1 / 8 →
  girls_absent_fraction = 1 / 4 →
  (((boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students) * 100 : ℚ) = 35 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_absent_student_percentage_l613_61308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l613_61327

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_sum_of_factors (x y z w : ℕ+) 
  (h : (x : ℕ) * y * z * w = factorial 12) : 
  (x : ℕ) + y + z + w ≥ 147 ∧ 
  ∃ (a b c d : ℕ+), (a : ℕ) * b * c * d = factorial 12 ∧ (a : ℕ) + b + c + d = 147 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l613_61327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wage_theorem_l613_61307

/-- Represents the recruitment problem for Red Star Company -/
def recruitment_problem (total_workers : ℕ) (wage_A wage_B : ℕ) : Prop :=
  ∀ (workers_A : ℕ),
    let workers_B := total_workers - workers_A
    let wage := wage_A * workers_A + wage_B * workers_B
    -- Condition: workers_B ≥ 2 * workers_A
    (workers_B ≥ 2 * workers_A) →
    -- The wage when workers_A = 50 is less than or equal to any other valid configuration
    wage ≥ wage_A * 50 + wage_B * (total_workers - 50)

theorem min_wage_theorem :
  recruitment_problem 150 2000 3000 ∧
  2000 * 50 + 3000 * (150 - 50) = 400000 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wage_theorem_l613_61307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_digits_not_double_l613_61373

theorem reverse_digits_not_double (N : ℕ) (h : N ≥ 10) : 
  ∀ (a b : ℕ) (digits : List ℕ), 
  (a ≠ 0 ∧ a < 10 ∧ b < 10) → 
  (N = a * 10^(digits.length + 1) + (digits.foldl (λ acc d ↦ acc * 10 + d) 0) * 10 + b) →
  (2 * N ≠ b * 10^(digits.length + 1) + (digits.foldr (λ d acc ↦ acc * 10 + d) 0) * 10 + a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_digits_not_double_l613_61373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_at_x_equals_one_max_volume_is_one_point_eight_l613_61386

/-- Represents the dimensions and volume of a rectangular container --/
structure Container where
  x : ℝ  -- Length of the shorter side of the base
  y : ℝ  -- Length of the longer side of the base
  h : ℝ  -- Height of the container
  volume : ℝ  -- Volume of the container

/-- The total length of the steel bar used for the container frame --/
def totalLength : ℝ := 14.8

/-- The difference in length between the longer and shorter sides of the base --/
def sideDifference : ℝ := 0.5

/-- Calculates the height of the container given the shorter side length --/
noncomputable def calculateHeight (x : ℝ) : ℝ :=
  (totalLength - 2 * (2*x + sideDifference)) / 4

/-- Calculates the volume of the container given the shorter side length --/
noncomputable def calculateVolume (x : ℝ) : ℝ :=
  x * (x + sideDifference) * calculateHeight x

/-- Theorem stating that the maximum volume is achieved when x = 1 --/
theorem max_volume_at_x_equals_one :
  ∃ (c : Container),
    c.x = 1 ∧
    c.y = c.x + sideDifference ∧
    c.h = calculateHeight c.x ∧
    c.volume = calculateVolume c.x ∧
    c.volume = 1.8 ∧
    ∀ (x : ℝ), 0 < x → x < (totalLength / 4) →
      calculateVolume x ≤ c.volume :=
by sorry

/-- Corollary stating that 1.8 is indeed the maximum volume --/
theorem max_volume_is_one_point_eight :
  ∃ (v : ℝ),
    v = 1.8 ∧
    ∀ (x : ℝ), 0 < x → x < (totalLength / 4) →
      calculateVolume x ≤ v :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_at_x_equals_one_max_volume_is_one_point_eight_l613_61386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_father_age_is_45_l613_61325

/-- The age of a father and his children -/
structure FamilyAges where
  fatherAge : ℕ
  childrenAges : Fin 5 → ℕ

/-- The sum of the ages of the children -/
def sumChildrenAges (ages : FamilyAges) : ℕ :=
  (Finset.univ : Finset (Fin 5)).sum ages.childrenAges

/-- The conditions given in the problem -/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.fatherAge = sumChildrenAges ages ∧
  sumChildrenAges ages + 5 * 15 = 2 * (ages.fatherAge + 15)

/-- The theorem to be proved -/
theorem father_age_is_45 (ages : FamilyAges) (h : validFamilyAges ages) :
  ages.fatherAge = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_father_age_is_45_l613_61325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_g_solution_exists_l613_61375

-- Define vectors a and b
noncomputable def a : ℝ × ℝ := (2, Real.sqrt 3)
noncomputable def b (x : ℝ) : ℝ × ℝ := (1/2 - (Real.cos (x/2))^2, Real.sin x)

-- Define function f
noncomputable def f (x : ℝ) : ℝ := a.1 * (b x).1 + a.2 * (b x).2

-- Define function g
noncomputable def g (x : ℝ) : ℝ := f (2*x)

-- Theorem statement
theorem g_range : 
  ∀ y ∈ Set.range g, -1 ≤ y ∧ y ≤ 2 := by
  sorry

-- Theorem for the solution of g(x) - m = 0
theorem g_solution_exists (m : ℝ) :
  (-1 ≤ m ∧ m ≤ 2) ↔ ∃ x ∈ Set.Icc (0 : ℝ) (π/2), g x = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_g_solution_exists_l613_61375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_area_correct_l613_61356

/-- The region G defined by the given inequalities -/
def region_G (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≤ -p.1^2 ∧ p.2 ≥ p.1^2 - 2*p.1 + a}

/-- The largest area λ for rectangles containing region G -/
noncomputable def largest_area (a : ℝ) : ℝ :=
  if a ≥ 1/2 then 0
  else if 0 < a ∧ a < 1/2 then 1 - 2*a
  else (1 - a) * Real.sqrt (1 - 2*a)

/-- Main theorem: The largest_area function gives the correct λ -/
theorem largest_area_correct (a : ℝ) :
  ∀ (rect : Set (ℝ × ℝ)),
    (∃ x₁ x₂ y₁ y₂ : ℝ, rect = {p : ℝ × ℝ | x₁ ≤ p.1 ∧ p.1 ≤ x₂ ∧ y₁ ≤ p.2 ∧ p.2 ≤ y₂}) →
    (region_G a ⊆ rect) →
    ((x₂ - x₁) * (y₂ - y₁) ≥ largest_area a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_area_correct_l613_61356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_pole_to_line_l613_61342

/-- Given a line with polar equation ρ * sin(θ + π/4) = √2/2, 
    the distance from the pole (origin) to this line is √2/2 -/
theorem distance_from_pole_to_line (ρ θ : ℝ) : 
  ρ * Real.sin (θ + π/4) = Real.sqrt 2 / 2 → 
  (|1 * 0 + 1 * 0 - 1| : ℝ) / Real.sqrt (1^2 + 1^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_pole_to_line_l613_61342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_inside_unit_square_l613_61383

theorem square_inside_unit_square (l : ℝ) :
  (∃ (x y : ℝ), 0 ≤ x ∧ x + l ≤ 1 ∧ 0 ≤ y ∧ y + l ≤ 1 ∧
    (∀ (a b : ℝ), x ≤ a ∧ a ≤ x + l ∧ y ≤ b ∧ b ≤ y + l →
      (a - 0.5)^2 + (b - 0.5)^2 > 0)) →
  l ≤ 1/2 := by
  sorry

#check square_inside_unit_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_inside_unit_square_l613_61383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_reflection_distance_l613_61399

structure Ellipse where
  a : ℝ
  c : ℝ
  h_pos_a : 0 < a
  h_pos_c : 0 < c
  h_c_lt_a : c < a

def Ellipse.focal_distance (e : Ellipse) : ℝ := 2 * e.c

def Ellipse.major_axis (e : Ellipse) : ℝ := 2 * e.a

def Ellipse.travel_distance (e : Ellipse) : Set ℝ :=
  {4 * e.a, 2 * (e.a - e.c), 2 * (e.a + e.c)}

-- Define is_on_ellipse function
def is_on_ellipse (e : Ellipse) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  (x^2 / e.a^2) + (y^2 / (e.a^2 - e.c^2)) = 1

-- Define path_length function
noncomputable def path_length (path : ℝ → ℝ × ℝ) : ℝ := sorry

theorem ellipse_reflection_distance (e : Ellipse) :
  ∀ d : ℝ, d ∈ e.travel_distance →
    ∃ path : ℝ → ℝ × ℝ,
      (path 0 = (- e.c, 0)) ∧
      (path 1 = (- e.c, 0)) ∧
      (∃ t : ℝ, 0 < t ∧ t < 1 ∧ is_on_ellipse e (path t)) ∧
      (path_length path = d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_reflection_distance_l613_61399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_formula_l613_61330

/-- The volume of a triangular pyramid with one edge of length a and other edges of length b -/
noncomputable def triangular_pyramid_volume (a b : ℝ) : ℝ :=
  (a * b / 12) * Real.sqrt (3 * b^2 - a^2)

/-- Theorem stating the volume of a triangular pyramid with given edge lengths -/
theorem triangular_pyramid_volume_formula (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ V : ℝ, V = triangular_pyramid_volume a b ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_formula_l613_61330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_heads_probability_l613_61397

/-- The number of ways to arrange k heads in n positions without consecutive heads -/
def arrangements (n k : ℕ) : ℕ :=
  if k ≤ n + 1 then Nat.choose (n + 1 - k) k else 0

/-- The total number of favorable outcomes (sequences without consecutive heads) -/
def favorableOutcomes : ℕ :=
  (List.range 6).map (arrangements 9) |>.sum

/-- The total number of possible outcomes when flipping a coin 10 times -/
def totalOutcomes : ℕ := 2^10

/-- The probability of no two consecutive heads in 10 coin tosses -/
noncomputable def probability : ℚ := favorableOutcomes / totalOutcomes

theorem no_consecutive_heads_probability :
  probability = 9 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_heads_probability_l613_61397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_sines_l613_61333

/-- The law of sines for a triangle -/
theorem law_of_sines (a b c α β γ S R : ℝ) :
  a > 0 → b > 0 → c > 0 →
  α > 0 → β > 0 → γ > 0 →
  S > 0 → R > 0 →
  α + β + γ = π →
  S = (1/2) * a * b * Real.sin γ →
  R = a / (2 * Real.sin α) →
  a / Real.sin α = b / Real.sin β ∧
  b / Real.sin β = c / Real.sin γ ∧
  c / Real.sin γ = 2 * R ∧
  2 * R = a * b * c / (2 * S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_sines_l613_61333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l613_61366

/-- A function g with specific properties -/
noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

/-- Theorem stating the unique number not in the range of g -/
theorem unique_number_not_in_range
  (p q r s : ℝ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (hs : s ≠ 0)
  (h1 : g p q r s 13 = 13)
  (h2 : g p q r s 31 = 31)
  (h3 : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! y, ∀ x, g p q r s x ≠ y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l613_61366
