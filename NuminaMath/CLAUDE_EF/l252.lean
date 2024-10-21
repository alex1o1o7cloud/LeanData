import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_tangent_touches_circle_at_one_point_l252_25247

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 25

-- Define the point P
def point_P : ℝ × ℝ := (-1, 7)

-- Define the tangent line
def tangent_line_eq (x y : ℝ) : Prop := 3*x - 4*y + 31 = 0

-- Theorem statement
theorem tangent_line_to_circle :
  (∀ x y : ℝ, circle_eq x y → ¬tangent_line_eq x y) ∧ 
  (∃ x y : ℝ, circle_eq x y ∧ tangent_line_eq x y) ∧
  tangent_line_eq point_P.1 point_P.2 :=
sorry

-- Additional helper theorem to demonstrate the tangent property
theorem tangent_touches_circle_at_one_point :
  ∃! p : ℝ × ℝ, circle_eq p.1 p.2 ∧ tangent_line_eq p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_tangent_touches_circle_at_one_point_l252_25247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unreachable_value_l252_25235

noncomputable def f (x : ℝ) : ℝ := (2 - x) / (3 * x + 4)

theorem unreachable_value (x : ℝ) (hx : x ≠ -4/3) :
  ∀ y : ℝ, y = f x → y ≠ -1/3 :=
by
  intro y hy
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unreachable_value_l252_25235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l252_25276

/-- Proves the length of a train given its speed and time to cross a bridge -/
theorem train_length_calculation (bridge_length : ℝ) (crossing_time : ℝ) (speed_kmph : ℝ) :
  let speed_mps : ℝ := speed_kmph * (1000 / 3600)
  let train_length : ℝ := speed_mps * crossing_time - bridge_length
  bridge_length = 150 ∧ 
  crossing_time = 49.9960003199744 ∧ 
  speed_kmph = 18 →
  abs (train_length - 99.98) < 0.01 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l252_25276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_sum_of_roots_specific_equation_l252_25284

noncomputable def QuadraticFormula (a b c : ℝ) : ℝ × ℝ :=
  let d := b^2 - 4*a*c
  ((-b + Real.sqrt d) / (2*a), (-b - Real.sqrt d) / (2*a))

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let (r₁, r₂) := QuadraticFormula a b c
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_equation :
  let (r₁, r₂) := QuadraticFormula 1 (-2000) (-2001)
  r₁ + r₂ = 2000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_sum_of_roots_specific_equation_l252_25284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_center_of_symmetry_l252_25239

/-- A periodic function with given properties -/
noncomputable def f (M ω φ : ℝ) (x : ℝ) : ℝ := M * Real.sin (ω * x + φ)

/-- The main theorem -/
theorem symmetry_point (M ω φ : ℝ) (hM : M ≠ 0) (hω : ω > 0) 
  (hφ : -π/2 < φ ∧ φ < π/2) (hsym : ∀ x, f M ω φ (4*π/3 - x) = f M ω φ x) 
  (hperiod : ∀ x, f M ω φ (x + π) = f M ω φ x) : 
  ∀ x, f M ω φ (5*π/6 - x) = f M ω φ (5*π/6 + x) := by
  sorry

/-- Theorem for option C -/
theorem center_of_symmetry (M ω φ : ℝ) (hM : M ≠ 0) (hω : ω > 0) 
  (hφ : -π/2 < φ ∧ φ < π/2) (hsym : ∀ x, f M ω φ (4*π/3 - x) = f M ω φ x) 
  (hperiod : ∀ x, f M ω φ (x + π) = f M ω φ x) : 
  ∀ x, f M ω φ (5*π/6 - x) = f M ω φ (5*π/6 + x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_center_of_symmetry_l252_25239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_problem_solution_l252_25225

def work_problem (daily_wage_a daily_wage_b daily_wage_c : ℚ) 
  (days_worked_a days_worked_b days_worked_c : ℕ) : Prop :=
  daily_wage_a * days_worked_a + daily_wage_b * days_worked_b + daily_wage_c * days_worked_c = 1480 ∧
  daily_wage_a = 3 * (daily_wage_c / 5) ∧
  daily_wage_b = 4 * (daily_wage_c / 5) ∧
  days_worked_a = 6 ∧
  days_worked_b = 9 ∧
  days_worked_c = 4 ∧
  daily_wage_c = 100

theorem work_problem_solution :
  ∃ (daily_wage_a daily_wage_b daily_wage_c : ℚ)
    (days_worked_a days_worked_b days_worked_c : ℕ),
  work_problem daily_wage_a daily_wage_b daily_wage_c days_worked_a days_worked_b days_worked_c :=
by
  -- Provide the values that satisfy the problem
  use 60, 80, 100, 6, 9, 4
  -- Split the goal into individual conditions
  apply And.intro
  -- Prove the total earnings
  · simp [work_problem]
    norm_num
  -- Prove the remaining conditions
  apply And.intro
  · norm_num
  apply And.intro
  · norm_num
  apply And.intro
  · rfl
  apply And.intro
  · rfl
  apply And.intro
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_problem_solution_l252_25225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_125_l252_25245

def sequenceTerm (n : ℕ+) : ℕ := n^3

theorem fifth_term_is_125 : sequenceTerm 5 = 125 := by
  rfl

#eval sequenceTerm 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_125_l252_25245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l252_25228

/-- The set P -/
def P : Set ℝ := {x : ℝ | |x - 1| > 2}

/-- The set S parameterized by a -/
def S (a : ℝ) : Set ℝ := {x : ℝ | x^2 + (a + 1)*x + a > 0}

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ P → x ∈ S a) ∧ 
  (∃ x : ℝ, x ∈ S a ∧ x ∉ P) →
  a ∈ Set.Ioi (-3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l252_25228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_FAC_cuboid_l252_25229

/-- Cosine of angle FAC in a cuboid with dimensions a, b, and c -/
noncomputable def cos_angle_FAC (a b c : ℝ) : ℝ :=
  c^2 / (Real.sqrt (a^2 + c^2) * Real.sqrt (b^2 + c^2))

/-- Theorem: The cosine of angle FAC in a cuboid with dimensions a, b, and c 
    is equal to c² / (√(a² + c²) * √(b² + c²)) -/
theorem cos_angle_FAC_cuboid (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  cos_angle_FAC a b c = c^2 / (Real.sqrt (a^2 + c^2) * Real.sqrt (b^2 + c^2)) := by
  -- Unfold the definition of cos_angle_FAC
  unfold cos_angle_FAC
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_FAC_cuboid_l252_25229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palindromic_polynomial_close_roots_l252_25227

/-- A polynomial with complex coefficients -/
def ComplexPolynomial (n : ℕ) := Polynomial ℂ

/-- The degree of a polynomial -/
def degree (p : ComplexPolynomial n) : ℕ := p.natDegree

/-- A polynomial is palindromic if its coefficients are symmetric -/
def isPalindromic (p : ComplexPolynomial n) : Prop :=
  ∀ i : Fin (n + 1), p.coeff i = p.coeff (n - i)

/-- A polynomial has no double roots -/
def hasNoDoubleRoots (p : ComplexPolynomial n) : Prop :=
  ∀ z : ℂ, (p.eval z = 0 → p.derivative.eval z ≠ 0)

/-- The theorem statement -/
theorem palindromic_polynomial_close_roots
  (d : ℕ) (h_d : d ≥ 13) (P : ComplexPolynomial d)
  (h_palindromic : isPalindromic P) (h_no_double_roots : hasNoDoubleRoots P) :
  ∃ z1 z2 : ℂ, z1 ≠ z2 ∧
    P.eval z1 = 0 ∧
    P.eval z2 = 0 ∧
    Complex.abs (z1 - z2) < 1 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palindromic_polynomial_close_roots_l252_25227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_condition_l252_25230

/-- Definition of an ellipse with semi-major axis m and semi-minor axis n -/
noncomputable def Ellipse (m n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / m^2) + (p.2^2 / n^2) = 1}

/-- Definition of eccentricity for an ellipse -/
noncomputable def Eccentricity (m n : ℝ) : ℝ :=
  Real.sqrt (1 - (n^2 / m^2))

/-- Theorem stating that m = 5 and n = 4 are sufficient but not necessary for eccentricity 3/5 -/
theorem ellipse_eccentricity_condition (m n : ℝ) :
  (m = 5 ∧ n = 4 → Eccentricity m n = 3/5) ∧
  ¬(Eccentricity m n = 3/5 → m = 5 ∧ n = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_condition_l252_25230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_eval_at_integer_l252_25223

-- Define a monic polynomial of degree n with real coefficients
def MonicPoly (n : ℕ) := {p : Polynomial ℝ // p.Monic ∧ p.degree = n}

-- Theorem statement
theorem exists_large_eval_at_integer 
  (n : ℕ) (P : MonicPoly n) (k : Fin (n + 1) → ℤ) 
  (h_distinct : ∀ i j, i ≠ j → k i ≠ k j) :
  ∃ i : Fin (n + 1), |P.val.eval (↑(k i))| ≥ (n.factorial : ℝ) / 2^n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_eval_at_integer_l252_25223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l252_25212

/-- Calculates the length of a train given its speed and time to cross a platform of equal length -/
noncomputable def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  (speed * 1000 / 3600) * time / 2

/-- Theorem stating that a train with speed 90 km/hr crossing a platform of equal length in one minute has a length of 750 meters -/
theorem train_length_calculation :
  train_length 90 60 = 750 := by
  -- Expand the definition of train_length
  unfold train_length
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l252_25212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_zero_iff_ge_five_l252_25279

def v (n : ℕ) : ℕ := n^4 + 2*n^2 + 2

def delta : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0, f => f
  | k+1, f => λ n => delta k f (n+1) - delta k f n

theorem delta_zero_iff_ge_five (k : ℕ) :
  (∀ n, delta k v n = 0) ↔ k ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_zero_iff_ge_five_l252_25279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_extension_l252_25209

/-- Given a slope of length 1 kilometer with an initial inclination of 20°,
    prove that the base must be extended by 1 kilometer to reduce the inclination to 10°. -/
theorem slope_extension (slope_length : ℝ) (initial_angle : ℝ) (final_angle : ℝ) :
  slope_length = 1 →
  initial_angle = 20 * π / 180 →
  final_angle = 10 * π / 180 →
  slope_length * (Real.cos final_angle - Real.cos initial_angle) = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_extension_l252_25209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_positive_l252_25260

open Real

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  decreasing : ∀ x y, x < y → f x > f y
  twice_diff : DifferentiableOn ℝ (deriv f) Set.univ
  condition : ∀ x, f x / (deriv (deriv f) x) < 1 - x

/-- The main theorem to be proved -/
theorem special_function_positive (sf : SpecialFunction) : ∀ x, sf.f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_positive_l252_25260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_area_l252_25205

-- Define the constant c
variable (c : ℝ)

-- Define the condition c > 1
variable (h : c > 1)

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the line passing through (1, c) with slope m
def line (m : ℝ) (x : ℝ) : ℝ := m * (x - 1) + c

-- Define the area function
noncomputable def area (c : ℝ) (m : ℝ) : ℝ := 
  Real.sqrt ((m - 2)^2 + 4*(c - 1)) * (m^2/6 - 2*m/3 + 2*c/3)

-- Theorem statement
theorem least_area (c : ℝ) (h : c > 1) :
  ∃ (A : ℝ), A = (4/3) * (c - 1)^(3/2) ∧ 
  ∀ (m : ℝ), area c m ≥ A :=
by
  sorry

#check least_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_area_l252_25205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_movement_limit_l252_25202

/-- Represents the bug's movement pattern --/
noncomputable def bugMovement (n : ℕ) : ℝ × ℝ :=
  (5 * (1/2)^(n-1) * Real.cos (60 * (n-1) * Real.pi / 180),
   5 * (1/2)^(n-1) * Real.sin (60 * (n-1) * Real.pi / 180))

/-- The limiting point P --/
noncomputable def P : ℝ × ℝ :=
  (∑' n, (bugMovement n).1, ∑' n, (bugMovement n).2)

/-- The square of the distance between O and P --/
noncomputable def OP_squared : ℝ :=
  P.1^2 + P.2^2

/-- Theorem stating that OP² = 100/3 --/
theorem bug_movement_limit : OP_squared = 100/3 := by
  sorry

#check bug_movement_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_movement_limit_l252_25202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rv_parking_probability_l252_25270

/-- The number of parking spaces -/
def total_spaces : ℕ := 20

/-- The number of cars parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent spaces required for the RV -/
def required_spaces : ℕ := 3

/-- The probability of finding the required adjacent empty spaces -/
def probability_of_parking : ℚ := 1839 / 1938

/-- Theorem stating the probability of finding required adjacent empty spaces -/
theorem rv_parking_probability :
  let remaining_spaces := total_spaces - parked_cars
  probability_of_parking = 1 - (Nat.choose 12 5 : ℚ) / (Nat.choose total_spaces remaining_spaces) := by
  sorry

#eval probability_of_parking

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rv_parking_probability_l252_25270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l252_25252

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * Real.sin x + 1)

theorem domain_of_f :
  {x : ℝ | ∃ k : ℤ, -π/6 + 2*k*π ≤ x ∧ x ≤ 7*π/6 + 2*k*π} =
  {x : ℝ | 2 * Real.sin x + 1 ≥ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l252_25252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l252_25272

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-8*x^2 + 14*x - 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | 1/4 ≤ x ∧ x ≤ 3/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l252_25272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_geometric_series_l252_25222

/-- Given a real number x > 1, the infinite sum ∑(n=0 to ∞) 1 / (x^(3^n) - x^(-3^n)) is equal to 1 / (x - 1). -/
theorem sum_geometric_series (x : ℝ) (hx : x > 1) :
  ∑' n, 1 / (x^(3^n) - (1/x)^(3^n)) = 1 / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_geometric_series_l252_25222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_trapezoid_median_l252_25226

/-- A trapezoid formed by two equilateral triangles -/
structure EquilateralTrapezoid where
  side1 : ℝ  -- Side length of the first equilateral triangle
  side2 : ℝ  -- Side length of the second equilateral triangle

/-- The median of an equilateral trapezoid -/
noncomputable def median (t : EquilateralTrapezoid) : ℝ := (t.side1 + t.side2) / 2

/-- Theorem: The median of a trapezoid formed by equilateral triangles with sides 4 and 3 is 3.5 -/
theorem equilateral_trapezoid_median :
  ∃ (t : EquilateralTrapezoid), t.side1 = 4 ∧ t.side2 = 3 ∧ median t = 3.5 := by
  -- Construct the trapezoid
  let t : EquilateralTrapezoid := ⟨4, 3⟩
  -- Prove the existence
  use t
  -- Prove the three conjuncts
  constructor
  · rfl  -- t.side1 = 4
  constructor
  · rfl  -- t.side2 = 3
  -- Prove median t = 3.5
  unfold median
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_trapezoid_median_l252_25226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_comparison_l252_25295

/-- Probability of not selecting a bad coin when choosing one from a box of 100 coins --/
noncomputable def prob_not_bad_one : ℝ := 99 / 100

/-- Probability of not selecting a bad coin when choosing two from a box of 100 coins --/
noncomputable def prob_not_bad_two : ℝ := 98 / 100

/-- Number of boxes for method one --/
def boxes_method_one : ℕ := 10

/-- Number of boxes for method two --/
def boxes_method_two : ℕ := 5

/-- Probability of finding at least one bad coin using method one --/
noncomputable def P1 : ℝ := 1 - prob_not_bad_one ^ boxes_method_one

/-- Probability of finding at least one bad coin using method two --/
noncomputable def P2 : ℝ := 1 - prob_not_bad_two ^ boxes_method_two

theorem prob_comparison : P1 < P2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_comparison_l252_25295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_fixed_all_fixed_l252_25241

/-- Represents a player in the tournament -/
structure Player where
  room : Fin 2
  position : Fin 2023

/-- Represents the tournament setup -/
structure Tournament where
  players : Fin 4046 → Player
  challenges : Fin 2023 → Finset (Fin 2023)
  reordering : Fin 4046 → Fin 4046

/-- Defines the challenge relationship between players -/
def challenges (t : Tournament) (p q : Player) : Prop :=
  p.room ≠ q.room ∧ q.position ∈ t.challenges p.position

/-- States that the challenge relationships are preserved after reordering -/
def preserved_challenges (t : Tournament) : Prop :=
  ∀ p q, challenges t p q ↔ 
    challenges t 
      (t.players (t.reordering (p.room.val + 2023 * p.position.val)))
      (t.players (t.reordering (q.room.val + 2023 * q.position.val)))

/-- The main theorem: if one player doesn't move, no one moves -/
theorem one_fixed_all_fixed (t : Tournament) (h_challenges : preserved_challenges t) :
  (∃ p : Player, t.reordering (p.room.val + 2023 * p.position.val) = p.room.val + 2023 * p.position.val) →
  (∀ p : Player, t.reordering (p.room.val + 2023 * p.position.val) = p.room.val + 2023 * p.position.val) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_fixed_all_fixed_l252_25241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_students_l252_25282

theorem water_students (total : ℕ) (juice_percent water_percent : ℚ) (juice_count : ℕ) : ℕ :=
  let water_count : ℕ := 60
  have h1 : juice_percent = 70 / 100 := by sorry
  have h2 : water_percent = 30 / 100 := by sorry
  have h3 : juice_count = 140 := by sorry
  have h4 : juice_percent + water_percent = 1 := by sorry
  have h5 : juice_count = total * juice_percent := by sorry
  have h6 : water_count = total * water_percent := by sorry
  water_count


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_students_l252_25282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_numbers_inequality_l252_25280

theorem seven_numbers_inequality (S : Finset ℝ) (h : S.card = 7) :
  ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ 0 < (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) ≤ 1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_numbers_inequality_l252_25280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_circle_equality_l252_25293

universe u

variable {U : Type u}

def complement (X : Set U) : Set U :=
  {x : U | x ∉ X}

def operation_circle (X Y : Set U) : Set U :=
  (complement X) ∪ Y

theorem operation_circle_equality 
  (X Y Z : Set U) :
  operation_circle X (operation_circle Y Z) = 
  (complement X) ∪ (complement Y) ∪ Z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_circle_equality_l252_25293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_time_l252_25289

/-- The current time in minutes past 4:00 -/
def t : ℝ := sorry

/-- The position of the hour hand at time t -/
def hour_hand_position (t : ℝ) : ℝ := 120 + 0.5 * t

/-- The position of the minute hand at time t -/
def minute_hand_position (t : ℝ) : ℝ := 6 * t

/-- The condition that the time is between 4:00 and 5:00 -/
axiom time_range : 0 < t ∧ t < 60

/-- The condition that in 8 minutes, the minute hand will be exactly opposite to where the hour hand was 6 minutes ago -/
axiom opposite_hands : 
  |minute_hand_position (t + 8) - hour_hand_position (t - 6)| = 180

theorem exact_time : t = 45 + 3/11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_time_l252_25289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l252_25214

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Axioms based on the given conditions
axiom f_def : ∀ x : ℝ, f x = 4 * x^2 - f (-x)
axiom f'_bound : ∀ x : ℝ, x < 0 → f' x + 1/2 < 4 * x
axiom f_inequality : ∀ m : ℝ, f (m + 1) ≤ f (-m) + 4 * m + 2

-- Theorem to prove
theorem m_range : 
  ∀ m : ℝ, (f (m + 1) ≤ f (-m) + 4 * m + 2) → m ≥ -1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l252_25214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_l252_25256

/-- Circle C₁ with polar equation ρ = 4cosθ -/
noncomputable def C₁ (θ : ℝ) : ℝ := 4 * Real.cos θ

/-- Circle C₂ with polar equation ρ = 2sinθ -/
noncomputable def C₂ (θ : ℝ) : ℝ := 2 * Real.sin θ

/-- The product of the distances |OP| and |OQ| -/
noncomputable def distanceProduct (α : ℝ) : ℝ :=
  Real.sqrt ((8 + 8 * Real.cos α) * (2 + 2 * Real.sin α))

/-- Theorem stating the maximum value of |OP| · |OQ| -/
theorem max_distance_product :
  ∃ (α : ℝ), ∀ (β : ℝ), distanceProduct α ≥ distanceProduct β ∧
  distanceProduct α = 4 + 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_l252_25256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l252_25213

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x^2 + 1/(x^2 + 1/x^2)

theorem f_minimum_value (x : ℝ) (hx : x > 0) : f x ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l252_25213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l252_25203

theorem triangle_inequalities (A B C : Real) (h : A + B + C = Real.pi) :
  (Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) ≤ 1/8) ∧
  (Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2) ≤ 3 * Real.sqrt 3 / 8) ∧
  (Real.cos A + Real.cos B + Real.cos C ≤ 3/2) := by
  sorry

#check triangle_inequalities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l252_25203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_weighted_average_speed_l252_25236

/-- Calculates the weighted average speed of a train journey with four segments -/
noncomputable def weightedAverageSpeed (x : ℝ) : ℝ :=
  let d1 := x
  let d2 := 2 * x
  let d3 := 3 * x
  let d4 := 4 * x
  let v1 : ℝ := 40
  let v2 : ℝ := 20
  let v3 : ℝ := 60
  let v4 : ℝ := 30
  let totalDistance := d1 + d2 + d3 + d4
  let totalTime := d1 / v1 + d2 / v2 + d3 / v3 + d4 / v4
  totalDistance / totalTime

/-- Theorem stating that the weighted average speed is 1200 * x / 37 -/
theorem train_weighted_average_speed (x : ℝ) (h : x > 0) :
  weightedAverageSpeed x = 1200 * x / 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_weighted_average_speed_l252_25236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_tangent_line_properties_l252_25299

/-- Parabola G: y = mx² - (1-4m)x + c passing through (1,a), (-1,a), and (0,-1) -/
def G (m c a : ℝ) : ℝ → ℝ := λ x ↦ m * x^2 - (1 - 4*m) * x + c

/-- Tangent line l: y = kx + b (k ≠ 0) -/
def l (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

/-- The theorem states the properties of the parabola and the tangent line -/
theorem parabola_and_tangent_line_properties :
  ∀ (m c a k b : ℝ),
    k ≠ 0 →
    G m c a 1 = a →
    G m c a (-1) = a →
    G m c a 0 = -1 →
    l k b 0 = -3 →
    (∃ (x : ℝ), G (1/4) (-1) a x = l k b x ∧ 
      ∀ (y : ℝ), y ≠ x → G (1/4) (-1) a y ≠ l k b y) →
    (∃ (y : ℝ), 
      let p := (b + 2) / k
      let q := (b + 4) / k
      p^2 + (2 - y)^2 - q^2 - (-4 - y)^2 = -4) →
    G m c a = G (1/4) (-1) a ∧ 
    (∃ (y : ℝ), y = -1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_tangent_line_properties_l252_25299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l252_25254

def correct_answer : String := "C"

def is_correct_answer (answer : String) : Prop :=
  answer = correct_answer

theorem solution_is_correct : is_correct_answer "C" := by
  -- Unfold the definition of is_correct_answer
  unfold is_correct_answer
  -- The proof is just reflexivity, as "C" = "C"
  rfl

#check solution_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l252_25254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_percentage_bounds_l252_25232

/-- Represents the share packages in the auction --/
structure SharePackage where
  razneft : ℕ
  dvanneft : ℕ
  trineft : ℕ

/-- Represents the prices of individual shares --/
structure SharePrices where
  razneft : ℚ
  dvanneft : ℚ
  trineft : ℚ

/-- The main theorem statement --/
theorem share_percentage_bounds (sp : SharePackage) (prices : SharePrices) : 
  -- Conditions
  sp.razneft + sp.dvanneft = sp.trineft → 
  prices.dvanneft * sp.dvanneft = 1/4 * prices.razneft * sp.razneft →
  prices.razneft * sp.razneft + prices.dvanneft * sp.dvanneft = prices.trineft * sp.trineft →
  16 ≤ prices.razneft - prices.dvanneft → 
  prices.razneft - prices.dvanneft ≤ 20 →
  42 ≤ prices.trineft → 
  prices.trineft ≤ 60 →
  -- Conclusion
  (1/8 : ℚ) ≤ sp.dvanneft / (sp.razneft + sp.dvanneft + sp.trineft) ∧ 
  sp.dvanneft / (sp.razneft + sp.dvanneft + sp.trineft) ≤ (3/20 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_percentage_bounds_l252_25232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_difference_l252_25287

/-- Calculates the difference between green and red apples after delivery and sales -/
theorem apple_difference (initial_green : ℤ) (initial_red_difference : ℤ) 
  (delivered_green : ℤ) (delivered_red : ℤ) (sold_green : ℤ) (sold_red : ℤ) : 
  initial_green = 1200 → 
  initial_red_difference = 3250 → 
  delivered_green = 3600 → 
  delivered_red = 1300 → 
  sold_green = 750 → 
  sold_red = 810 → 
  (initial_green + delivered_green - sold_green) - 
  (initial_green + initial_red_difference + delivered_red - sold_red) = -890 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_difference_l252_25287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_reduction_l252_25277

/-- Proves that a specific price reduction percentage results in given sales and value increases -/
theorem tv_price_reduction (price_reduction : ℝ) 
  (h1 : (1 - price_reduction / 100) * 1.86 = 1.4508) : 
  ‖price_reduction - 21.98‖ < 0.01 := by
  sorry

#check tv_price_reduction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_reduction_l252_25277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snoring_heart_disease_relationship_l252_25248

theorem snoring_heart_disease_relationship (confidence : Real) 
  (heart_disease_patients : Finset Nat) :
  confidence = 0.99 →
  Finset.card heart_disease_patients = 100 →
  ∃ (snoring_patients : Finset Nat), 
    snoring_patients ⊆ heart_disease_patients ∧
    Finset.card snoring_patients = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snoring_heart_disease_relationship_l252_25248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_inequality_l252_25207

theorem max_lambda_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  ∃ (lambda_max : ℝ), ∀ (lambda : ℝ), (a^2 + b^2 + c^2 + lambda * Real.sqrt (a * b * c) ≤ 1) → lambda ≤ lambda_max ∧ lambda_max = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_inequality_l252_25207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_inverse_sqrt_quadratic_l252_25290

open Real

-- Define the function we're dealing with
noncomputable def f (x : ℝ) : ℝ := log (abs (x - 3 + sqrt (x^2 - 6*x + 3)))

-- State the theorem
theorem integral_of_inverse_sqrt_quadratic :
  ∀ x : ℝ, x^2 - 6*x + 3 > 0 → deriv f x = 1 / sqrt (x^2 - 6*x + 3) :=
by
  intro x h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_inverse_sqrt_quadratic_l252_25290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_as_percentage_l252_25283

/-- Represents a tax rate as a ratio of tax amount to base amount -/
structure TaxRate where
  taxAmount : ℚ
  baseAmount : ℚ
  baseAmount_pos : baseAmount > 0

/-- Converts a tax rate to a percentage -/
def toPercentage (rate : TaxRate) : ℚ :=
  (rate.taxAmount / rate.baseAmount) * 100

/-- The given tax rate of $65 per $100.00 -/
def givenTaxRate : TaxRate where
  taxAmount := 65
  baseAmount := 100
  baseAmount_pos := by norm_num

theorem tax_rate_as_percentage :
  toPercentage givenTaxRate = 65 := by
  unfold toPercentage givenTaxRate
  norm_num

#eval toPercentage givenTaxRate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_as_percentage_l252_25283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_theorem_l252_25231

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)
noncomputable def g (x : ℝ) : ℝ := x^2 + 2

-- State the theorem
theorem function_composition_theorem :
  (f 2 = 1/3) ∧
  (g 2 = 6) ∧
  (f (g 2) = 1/7) ∧
  (∀ x : ℝ, x ≠ -1 → f (g x) = 1 / (x^2 + 3)) ∧
  (∀ x : ℝ, x ≠ -1 → g (f x) = 1 / (1 + x)^2 + 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_theorem_l252_25231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_plus_pi_third_f_A_range_l252_25208

-- Define vectors m and n
noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.sqrt 3 * Real.sin (x / 4), 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos (x / 4), Real.cos (x / 4) ^ 2)

-- Define dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define f(x)
noncomputable def f (x : ℝ) : ℝ := dot_product (m x) (n x)

-- Theorem 1
theorem cos_x_plus_pi_third (x : ℝ) :
  dot_product (m x) (n x) = 2 → Real.cos (x + π / 3) = 1 / 2 := by sorry

-- Define triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  side_angle_relation : (2 * a - c) * Real.cos B = b * Real.cos C

-- Theorem 2
theorem f_A_range (t : Triangle) :
  ∃ (lb ub : ℝ), lb = 2 ∧ ub = 3 ∧ ∀ (y : ℝ), f t.A = y → lb < y ∧ y < ub := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_plus_pi_third_f_A_range_l252_25208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sumEO_equals_1350_l252_25218

/-- Sum of even digits of a natural number -/
def E (n : ℕ) : ℕ := sorry

/-- Sum of odd digits of a natural number -/
def O (n : ℕ) : ℕ := sorry

/-- The sum of E(n) + O(n) for n from 1 to 150 -/
def sumEO : ℕ := (List.range 150).map (λ n => E (n + 1) + O (n + 1)) |>.sum

theorem sumEO_equals_1350 : sumEO = 1350 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sumEO_equals_1350_l252_25218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l252_25243

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem function_range_theorem (f : ℝ → ℝ)
    (h_even : IsEven f)
    (h_incr : IncreasingOn f (Set.Ici 0))
    (h_zero : f 3 = 0) :
    {x : ℝ | f (x + 1) > 0} = Set.Ioo (-4) 2 := by
  sorry

#check function_range_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l252_25243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l252_25262

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := (n / 2 : ℝ) * (2 * a₁ + (n - 1) * d)

theorem arithmetic_sequence_problem (k : ℕ) :
  let a₁ : ℝ := 1
  let d : ℝ := 2
  S a₁ d (k + 2) - S a₁ d k = 24 → k = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l252_25262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_cube_decomposition_l252_25250

theorem cos_cube_decomposition (b₁ b₂ b₃ : ℝ) :
  (∀ θ : ℝ, (Real.cos θ) ^ 3 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ)) →
  b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_cube_decomposition_l252_25250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_duration_l252_25242

-- Define the painting rates and lunch break
variable (p : ℝ) -- Paula's painting rate (house/hour)
variable (h : ℝ) -- Combined rate of helpers (house/hour)
variable (L : ℝ) -- Lunch break duration (hours)

-- Define the conditions
def monday_condition (p h L : ℝ) : Prop := (9 - L) * (p + h) = 0.6
def tuesday_condition (h L : ℝ) : Prop := (7 - L) * h = 0.3
def wednesday_condition (p L : ℝ) : Prop := (25 - L) * p = 0.1

-- Define the theorem
theorem lunch_break_duration : 
  ∃ (p h L : ℝ), 
    monday_condition p h L ∧ 
    tuesday_condition h L ∧ 
    wednesday_condition p L ∧ 
    L = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_duration_l252_25242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_number_equality_l252_25233

theorem five_digit_number_equality : 
  (Finset.filter (fun n : ℕ => 10000 ≤ n ∧ n < 100000 ∧ n % 5 ≠ 0) (Finset.range 100000)).card = 
  (Finset.filter (fun n : ℕ => 10000 ≤ n ∧ n < 100000 ∧ (n / 10000 ≠ 5 ∧ (n / 1000) % 10 ≠ 5)) (Finset.range 100000)).card :=
by sorry

#eval (Finset.filter (fun n : ℕ => 10000 ≤ n ∧ n < 100000 ∧ n % 5 ≠ 0) (Finset.range 100000)).card
#eval (Finset.filter (fun n : ℕ => 10000 ≤ n ∧ n < 100000 ∧ (n / 10000 ≠ 5 ∧ (n / 1000) % 10 ≠ 5)) (Finset.range 100000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_number_equality_l252_25233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l252_25281

-- Define the two functions as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + 2)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

-- State the theorem
theorem sin_shift (x : ℝ) : f x = g (x + 1) := by
  -- Unfold the definitions of f and g
  unfold f g
  -- Simplify the expressions
  simp [Real.sin_add]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l252_25281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_percentage_in_mixture_l252_25271

/-- Calculates the percentage of ryegrass in a mixture of seed mixtures X and Y -/
theorem ryegrass_percentage_in_mixture 
  (x_ryegrass : ℝ) 
  (x_bluegrass : ℝ) 
  (y_ryegrass : ℝ) 
  (y_fescue : ℝ) 
  (x_weight : ℝ) :
  x_ryegrass = 40 →
  x_bluegrass = 60 →
  y_ryegrass = 25 →
  y_fescue = 75 →
  x_weight = 100/3 →
  x_ryegrass / 100 * x_weight + y_ryegrass / 100 * (100 - x_weight) = 30 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and might cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_percentage_in_mixture_l252_25271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l252_25237

theorem negation_of_sin_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l252_25237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l252_25265

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi/6) - 1

theorem f_properties :
  -- Period is π
  (∀ x, f (x + Real.pi) = f x) ∧
  -- Range on the interval [-π/6, π/4] is [-1, 2]
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f x ∈ Set.Icc (-1) 2) ∧
  (∃ x₁ ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f x₁ = -1) ∧
  (∃ x₂ ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f x₂ = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l252_25265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_15625_l252_25219

def divisorProduct (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem divisor_product_15625 (n : ℕ) :
  divisorProduct n = 15625 → n = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_15625_l252_25219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_g_l252_25269

noncomputable def f (x a : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + a)) * Real.sin x

noncomputable def g (x a b : ℝ) : ℝ := b^(x - a)

theorem fixed_point_of_g (a b : ℝ) (hb : b > 0) (hb_neq : b ≠ 1) 
  (h_even : ∀ x, f x a = f (-x) a) : 
  ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ g x a b = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_g_l252_25269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_and_die_probability_l252_25215

theorem coin_and_die_probability : 
  let coin_sides : Finset ℕ := {5, 15}
  let die_sides : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let target_sum : ℕ := 16
  let total_outcomes : ℕ := coin_sides.card * die_sides.card
  let favorable_outcomes : ℕ := (coin_sides.sum fun c => (die_sides.filter (fun d => c + d = target_sum)).card)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_and_die_probability_l252_25215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l252_25249

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (5 + (Real.sqrt 3 / 2) * t, Real.sqrt 3 + (1 / 2) * t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define point M
noncomputable def point_M : ℝ × ℝ := (5, Real.sqrt 3)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, line_l t₁ = A ∧ line_l t₂ = B ∧ 
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧ t₁ ≠ t₂

-- State the theorem
theorem intersection_product (A B : ℝ × ℝ) : 
  intersection_points A B → 
  (point_M.1 - A.1)^2 + (point_M.2 - A.2)^2 * 
  ((point_M.1 - B.1)^2 + (point_M.2 - B.2)^2) = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l252_25249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_degree_of_belief_inverse_relation_l252_25298

-- Define the Chi-square statistic
def chi_square : ℝ → ℝ := sorry

-- Define the degree of belief in the relationship between X and Y
def degree_of_belief : ℝ → ℝ := sorry

-- Theorem stating the inverse relationship between chi_square and degree_of_belief
theorem chi_square_degree_of_belief_inverse_relation :
  ∀ (χ₁ χ₂ : ℝ), χ₁ < χ₂ → degree_of_belief χ₁ < degree_of_belief χ₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_degree_of_belief_inverse_relation_l252_25298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l252_25288

def valid_first_digit (d : Nat) : Bool :=
  d ∈ [2, 3, 5, 6, 7, 8, 9]

def valid_other_digit (d : Nat) : Bool :=
  d ∈ [0, 2, 3, 5, 6, 7, 8, 9]

def is_valid_number (n : Nat) : Bool :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  valid_first_digit (n / 1000) ∧
  valid_other_digit ((n / 100) % 10) ∧
  valid_other_digit ((n / 10) % 10) ∧
  valid_other_digit (n % 10)

theorem count_valid_numbers : 
  (Finset.filter (fun n => is_valid_number n) (Finset.range 10000)).card = 3072 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l252_25288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_k_value_l252_25217

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (a 1 + a n)

theorem arithmetic_sequence_k_value
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum_2014 : sum_of_terms a 2014 > 0)
  (h_sum_2015 : sum_of_terms a 2015 < 0)
  (h_min_abs : ∃ k : ℕ, ∀ n : ℕ, n > 0 → |a n| ≥ |a k|) :
  ∃ k : ℕ, k = 1008 ∧ ∀ n : ℕ, n > 0 → |a n| ≥ |a k| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_k_value_l252_25217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_average_speed_l252_25251

/-- Represents the distance of a segment as a multiple of x -/
structure SegmentDistance (x : ℝ) where
  multiple : ℝ
  distance : ℝ := multiple * x

/-- Represents a segment of the train's journey -/
structure Segment (x : ℝ) where
  distance : SegmentDistance x
  speed : ℝ

/-- Calculates the time taken for a segment -/
noncomputable def segmentTime {x : ℝ} (seg : Segment x) : ℝ :=
  seg.distance.distance / seg.speed

/-- Represents the entire train journey -/
structure TrainJourney (x : ℝ) where
  segmentA : Segment x
  segmentB : Segment x
  segmentC : Segment x
  segmentD : Segment x
  segmentE : Segment x
  totalDistance : ℝ

/-- Calculates the total time of the journey -/
noncomputable def totalTime {x : ℝ} (journey : TrainJourney x) : ℝ :=
  segmentTime journey.segmentA +
  segmentTime journey.segmentB +
  segmentTime journey.segmentC +
  segmentTime journey.segmentD +
  segmentTime journey.segmentE

/-- Calculates the average speed of the journey -/
noncomputable def averageSpeed {x : ℝ} (journey : TrainJourney x) : ℝ :=
  journey.totalDistance / totalTime journey

theorem train_journey_average_speed (x : ℝ) (h : x > 0) :
  let journey : TrainJourney x := {
    segmentA := { distance := { multiple := 1 }, speed := 75 }
    segmentB := { distance := { multiple := 2 }, speed := 25 }
    segmentC := { distance := { multiple := 1.5 }, speed := 50 }
    segmentD := { distance := { multiple := 1 }, speed := 60 }
    segmentE := { distance := { multiple := 3 }, speed := 36 }
    totalDistance := 7.5 * x
  }
  averageSpeed journey = 7.5 * (900 / 201) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_average_speed_l252_25251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l252_25273

-- Define the new operation ⊕
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (oplus 1 x) * x - (oplus 2 x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 2 ∧
  f x = 6 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-2 : ℝ) 2 → f y ≤ 6 := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l252_25273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_theta_set_is_plane_l252_25286

/-- Cylindrical coordinate point -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying θ = 2c in cylindrical coordinates -/
def ConstantThetaSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.θ = 2 * c}

/-- Theorem: The set of points satisfying θ = 2c in cylindrical coordinates forms a plane -/
theorem constant_theta_set_is_plane (c : ℝ) :
  ∃ (a b d : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
    ConstantThetaSet c = {p : CylindricalPoint | a * p.r * Real.cos p.θ + b * p.r * Real.sin p.θ + d = 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_theta_set_is_plane_l252_25286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_b_l252_25257

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x - 2 * a

-- State the theorem
theorem smallest_positive_integer_b : 
  (∀ a ∈ Set.Ioo 1 4, ∀ x > 0, f a 11 x > 0) ∧ 
  (∀ b' ∈ Set.Ioi 0, b' < 11 → ∃ a ∈ Set.Ioo 1 4, ∃ x > 0, f a b' x ≤ 0) := by
  sorry

#check smallest_positive_integer_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_integer_b_l252_25257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_on_cylindrical_post_l252_25258

/-- The length of a string spiraling around a cylindrical post -/
theorem string_length_on_cylindrical_post 
  (circumference : ℝ) 
  (height : ℝ) 
  (num_loops : ℕ) : 
  circumference = 6 →
  height = 18 →
  num_loops = 6 →
  (num_loops : ℝ) * Real.sqrt ((height / (num_loops : ℝ))^2 + circumference^2) = 18 * Real.sqrt 5 := by
  sorry

#check string_length_on_cylindrical_post

end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_on_cylindrical_post_l252_25258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_max_value_min_value_l252_25275

-- Define the function f(x) = e^x - x + 1
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x + 1

-- Define the interval [-2, 1]
def interval : Set ℝ := Set.Icc (-2) 1

-- Statement 1: Tangent line equation at x = 2
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ (x y : ℝ), y = m * x + b ↔ (Real.exp 2 - 1) * x - y - Real.exp 2 + 1 = 0 := by
  sorry

-- Statement 2: Maximum value is e^(-2) + 3
theorem max_value :
  ∃ (x : ℝ), x ∈ interval ∧ f x = Real.exp (-2) + 3 ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x := by
  sorry

-- Statement 3: Minimum value is 2
theorem min_value :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 2 ∧ ∀ (y : ℝ), y ∈ interval → f x ≤ f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_max_value_min_value_l252_25275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_sum_l252_25294

/-- An arithmetic sequence with first term 1 and non-zero common difference -/
noncomputable def arithmetic_seq (d : ℝ) (n : ℕ) : ℝ := 1 + (n - 1 : ℝ) * d

/-- The condition that a_2, a_3, and a_6 form a geometric sequence -/
def geometric_condition (d : ℝ) : Prop := 
  (arithmetic_seq d 3) ^ 2 = (arithmetic_seq d 2) * (arithmetic_seq d 6)

/-- The sum of the first 6 terms of the arithmetic sequence -/
noncomputable def sum_first_six (d : ℝ) : ℝ := 
  6 * (arithmetic_seq d 1) + (6 * 5 / 2 : ℝ) * d

theorem arithmetic_seq_sum (d : ℝ) :
  d ≠ 0 → geometric_condition d → sum_first_six d = -24 := by
  sorry

#check arithmetic_seq_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_seq_sum_l252_25294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_difference_max_distance_difference_l252_25221

/-- The maximum value of √(x²-2x+5) - √(x²+1) is √2 -/
theorem max_value_sqrt_difference : 
  ∃ (max : ℝ), max = Real.sqrt 2 ∧ 
  ∀ (x : ℝ), Real.sqrt (x^2 - 2*x + 5) - Real.sqrt (x^2 + 1) ≤ max := by
  sorry

/-- The expression √(x²-2x+5) - √(x²+1) represents the difference between 
    distances from a point (x,0) to points (1,2) and (0,1) respectively -/
noncomputable def distance_difference (x : ℝ) : ℝ :=
  Real.sqrt ((x - 1)^2 + 2^2) - Real.sqrt (x^2 + 1^2)

/-- The maximum of the distance difference is achieved when it equals 
    the direct distance between (1,2) and (0,1) -/
theorem max_distance_difference : 
  ∃ (max : ℝ), max = Real.sqrt 2 ∧ 
  ∀ (x : ℝ), distance_difference x ≤ max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_difference_max_distance_difference_l252_25221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_direction_vector_l252_25240

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 2*x + y - 5 = 0

-- Define the line
def my_line (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Define the direction vector
def direction_vector : ℝ × ℝ := (1, 2)

theorem tangent_and_direction_vector :
  (∀ x y, my_circle x y ∧ (x, y) = P → tangent_line x y) ∧
  (∀ x y, my_line x y → ∃ t, (x, y) = (t * direction_vector.fst, t * direction_vector.snd)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_direction_vector_l252_25240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_max_volume_and_dot_product_cuboid_max_volume_cuboid_max_dot_product_l252_25278

/-- Represents a cuboid with edge lengths a, b, and c. -/
structure Cuboid where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  sphere_constraint : a^2 + b^2 + c^2 = 9

/-- The vector m in the problem. -/
noncomputable def m : Fin 3 → ℝ := ![1, 3, Real.sqrt 6]

/-- The vector n based on the cuboid's dimensions. -/
def n (cube : Cuboid) : Fin 3 → ℝ := ![cube.a, cube.b, cube.c]

/-- The main theorem stating the maximum volume and dot product. -/
theorem cuboid_max_volume_and_dot_product (cube : Cuboid) :
  (∀ other : Cuboid, cube.a * cube.b * cube.c ≥ other.a * other.b * other.c) →
  cube.a * cube.b * cube.c = 3 * Real.sqrt 3 ∧
  (∀ other : Cuboid, m • n cube ≥ m • n other) →
  m • n cube = 12 := by
  sorry

/-- The maximum volume of the cuboid is 3√3. -/
theorem cuboid_max_volume (cube : Cuboid) :
  (∀ other : Cuboid, cube.a * cube.b * cube.c ≥ other.a * other.b * other.c) →
  cube.a * cube.b * cube.c = 3 * Real.sqrt 3 := by
  sorry

/-- The maximum value of m · n is 12. -/
theorem cuboid_max_dot_product (cube : Cuboid) :
  (∀ other : Cuboid, m • n cube ≥ m • n other) →
  m • n cube = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_max_volume_and_dot_product_cuboid_max_volume_cuboid_max_dot_product_l252_25278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l252_25246

noncomputable def center : ℝ × ℝ := (-2, -3)
def radius : ℝ := 6

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def is_inside (p : ℝ × ℝ) : Prop :=
  distance center p < radius

def is_outside (p : ℝ × ℝ) : Prop :=
  distance center p > radius

theorem point_inside_circle (h : is_outside (5, -3)) : is_inside (0, -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l252_25246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l252_25261

/-- Sum of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

/-- The sum of the first 20 terms of an arithmetic sequence
    with first term 4 and common difference 3 is 650 -/
theorem arithmetic_sequence_sum :
  arithmetic_sum 4 3 20 = 650 := by
  -- Unfold the definition of arithmetic_sum
  unfold arithmetic_sum
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l252_25261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composed_number_proof_l252_25274

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToTenth (x : ℝ) : ℝ := 
  ⌊x * 10 + 0.5⌋ / 10

/-- The number composed of 10 ones, 9 tenths, and 6 hundredths -/
def composedNumber : ℝ := 10 + 0.9 + 0.06

theorem composed_number_proof : 
  composedNumber = 10.96 ∧ roundToTenth composedNumber = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composed_number_proof_l252_25274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_ratio_l252_25267

def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ := !![9/50, -15/50; -15/50, 41/50]

theorem projection_ratio {a b : ℚ} (h : projection_matrix.mulVec ![a, b] = ![a, b]) :
  b / a = -41 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_ratio_l252_25267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l252_25268

noncomputable def f (x : ℝ) : ℝ := x⁻¹ + (x⁻¹)^2 / (1 + (x⁻¹)^2)

theorem f_composition_negative_three : f (f (-3)) = -790/327 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l252_25268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_property_l252_25297

noncomputable def arithmeticGeometricSequence (a : ℝ) (q : ℝ) : ℕ → ℝ :=
  fun n => a * q^(n-1)

noncomputable def sumArithmeticGeometric (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem arithmetic_geometric_sequence_property
  (a : ℝ) (q : ℝ) (h1 : q ≠ 1) :
  let S := sumArithmeticGeometric a q
  ∃ (n : ℕ), n > 0 ∧
    (2 * S 4 = S 2 + S 3) ∧  -- S_2, S_4, S_3 form arithmetic sequence
    (arithmeticGeometricSequence a q 1 - arithmeticGeometricSequence a q 3 = 3) ∧  -- a_1 - a_3 = 3
    (q = -1/2) ∧  -- Common ratio is -1/2
    (S n = 21/8) ∧  -- Sum of n terms is 21/8
    (n = 6)  -- Number of terms is 6
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_property_l252_25297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_calculation_l252_25264

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

/-- Theorem: For a trapezium with parallel sides of lengths 22 and 18, and an area of 300,
    the distance between the parallel sides is 15. -/
theorem trapezium_height_calculation :
  ∃ h : ℝ, trapezium_area 22 18 h = 300 ∧ h = 15 := by
  use 15
  constructor
  · simp [trapezium_area]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_calculation_l252_25264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l252_25266

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem smallest_x_value (x : ℝ) : 
  (floor (x + 0.1) + floor (x + 0.2) + floor (x + 0.3) + floor (x + 0.4) + 
   floor (x + 0.5) + floor (x + 0.6) + floor (x + 0.7) + floor (x + 0.8) + 
   floor (x + 0.9) = 104) → 
  (∀ y, y < x → 
    (floor (y + 0.1) + floor (y + 0.2) + floor (y + 0.3) + floor (y + 0.4) + 
     floor (y + 0.5) + floor (y + 0.6) + floor (y + 0.7) + floor (y + 0.8) + 
     floor (y + 0.9) ≠ 104)) → 
  x = 11.5 := by
  sorry

#check smallest_x_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l252_25266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_miles_calculation_l252_25216

/-- Represents the fuel efficiency and travel distance of a car -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℚ
  city_mpg_difference : ℚ
  city_mpg : ℚ

/-- Calculates the miles per tankful in the city given the car's fuel efficiency -/
def city_miles_per_tankful (car : CarFuelEfficiency) : ℚ :=
  (car.highway_miles_per_tankful / (car.city_mpg + car.city_mpg_difference)) * car.city_mpg

/-- Theorem stating that for a car with given fuel efficiency, 
    the miles per tankful in the city is 336 -/
theorem city_miles_calculation (car : CarFuelEfficiency) 
  (h1 : car.highway_miles_per_tankful = 560)
  (h2 : car.city_mpg_difference = 6)
  (h3 : car.city_mpg = 9) :
  city_miles_per_tankful car = 336 := by
  sorry

#eval city_miles_per_tankful { highway_miles_per_tankful := 560, city_mpg_difference := 6, city_mpg := 9 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_miles_calculation_l252_25216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_equivalence_l252_25201

theorem log_inequality_equivalence (c a b : ℝ) (hc : c > 1) (ha : a > 0) (hb : b > 0) :
  (Real.log c / Real.log a > Real.log c / Real.log b) ↔ (1 / Real.log a > 1 / Real.log b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_equivalence_l252_25201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equals_polar_l252_25211

/-- Parametric equations of curve C -/
noncomputable def curve_C (t : ℝ) : ℝ × ℝ :=
  (1 + Real.sqrt 3 * t, Real.sqrt 3 - t)

/-- Polar equation of curve C -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 6) = 2

/-- Theorem stating the equivalence of parametric and polar equations -/
theorem parametric_equals_polar :
  ∀ x y ρ θ : ℝ, 
    (∃ t : ℝ, curve_C t = (x, y)) ↔ 
    (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ polar_equation ρ θ) :=
by
  sorry

#check parametric_equals_polar

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equals_polar_l252_25211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_height_is_correct_l252_25296

/-- Represents the dimensions and cost information for a room -/
structure RoomInfo where
  length : ℝ
  width : ℝ
  whitewashCost : ℝ
  doorHeight : ℝ
  doorWidth : ℝ
  windowHeight : ℝ
  windowWidth : ℝ
  windowCount : ℕ
  totalCost : ℝ

/-- Calculates the height of a room given its information -/
noncomputable def calculateRoomHeight (info : RoomInfo) : ℝ :=
  let perimeter := 2 * (info.length + info.width)
  let doorArea := info.doorHeight * info.doorWidth
  let windowArea := info.windowHeight * info.windowWidth * (info.windowCount : ℝ)
  let totalSubtractedArea := doorArea + windowArea
  (info.totalCost / info.whitewashCost + totalSubtractedArea) / perimeter

/-- Proves that the calculated room height is correct -/
theorem room_height_is_correct (info : RoomInfo) :
  info.length = 25 ∧
  info.width = 15 ∧
  info.whitewashCost = 8 ∧
  info.doorHeight = 6 ∧
  info.doorWidth = 3 ∧
  info.windowHeight = 4 ∧
  info.windowWidth = 3 ∧
  info.windowCount = 3 ∧
  info.totalCost = 7248 →
  calculateRoomHeight info = 12 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_height_is_correct_l252_25296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l252_25255

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 = 1

-- Define the distance from a point to a focus
noncomputable def distance_to_focus (x y fx fy : ℝ) : ℝ := Real.sqrt ((x - fx)^2 + (y - fy)^2)

-- Theorem statement
theorem ellipse_focus_distance 
  (x y : ℝ) 
  (f1x f1y f2x f2y : ℝ) -- Coordinates of the foci
  (h1 : is_on_ellipse x y)
  (h2 : distance_to_focus x y f1x f1y = 6) :
  distance_to_focus x y f2x f2y = 4 := by
  sorry

#check ellipse_focus_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l252_25255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l252_25244

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - 2 * (Real.sin x) ^ 2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) (Real.sqrt 2 - 1) ↔ 
    ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (3*Real.pi/8) ∧ f x = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l252_25244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l252_25204

/-- Definition of a hyperbola with parameters a and b -/
def Hyperbola (a b : ℝ) := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}

/-- Definition of the right focus of a hyperbola -/
noncomputable def RightFocus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)

/-- Definition of eccentricity of a hyperbola -/
noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- Theorem about the eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : b > a) (h2 : a > 0) :
  ∃ (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ),
    A ∈ Hyperbola a b ∧ B ∈ Hyperbola a b ∧
    A ∈ l ∧ B ∈ l ∧
    RightFocus a b ∈ l ∧
    (A.1 * B.1 + A.2 * B.2 = 0) →
    Eccentricity a b > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l252_25204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l252_25206

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- Proof that the distance between the given parallel lines is 2√13 / 13 -/
theorem distance_between_given_lines :
  let line1 := fun (x y : ℝ) => 3*x - 2*y + 1 = 0
  let line2 := fun (x y : ℝ) => 6*x - 4*y - 2 = 0
  distance_between_parallel_lines 3 (-2) 1 (-1) = 2 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l252_25206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_phi_l252_25285

/-- A function f is odd if f(-x) = -f(x) for all x in the domain of f -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = 2sin(x+φ) - cos(x) -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (x + φ) - Real.cos x

theorem odd_function_phi (φ : ℝ) : IsOdd (f φ) → φ = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_phi_l252_25285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l252_25220

-- Define the functions
noncomputable def f (x : ℝ) := 4 * Real.tan x
noncomputable def g (x : ℝ) := 6 * Real.sin x
noncomputable def h (x : ℝ) := Real.cos x

-- Define the theorem
theorem intersection_length :
  ∃ (x : ℝ) (P P₁ P₂ : ℝ × ℝ),
    0 < x ∧ x < Real.pi / 2 ∧
    f x = g x ∧
    P = (x, f x) ∧
    P₁ = (x, 0) ∧
    P₂.1 = x ∧ P₂.2 = h x ∧
    P₂.2 - P₁.2 = 2/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l252_25220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_product_shelf_life_l252_25210

-- Define the shelf life function
noncomputable def shelf_life (k b x : ℝ) : ℝ := Real.exp (k * x + b)

theorem food_product_shelf_life 
  (k b : ℝ) 
  (h1 : shelf_life k b 0 = 192) 
  (h2 : shelf_life k b 33 = 24) :
  k = -Real.log 2 / 11 ∧ 
  shelf_life k b 11 = 96 ∧ 
  shelf_life k b 22 = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_product_shelf_life_l252_25210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_value_l252_25200

theorem triangle_angle_value (A B C : ℝ) (h : 0 < A ∧ A < π) :
  (Real.sin A)^2 + Real.sqrt 2 * Real.sin B * Real.sin C = (Real.sin A)^2 - (Real.sin C)^2 →
  A = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_value_l252_25200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_line_in_plane_l252_25291

-- Define the necessary structures
structure Line3D where
  -- Placeholder for line definition
  dummy : Unit

structure Plane3D where
  -- Placeholder for plane definition
  dummy : Unit

-- Define the relationships
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry -- Definition to be implemented

def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry -- Definition to be implemented

def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry -- Definition to be implemented

def skew_lines (l1 l2 : Line3D) : Prop :=
  sorry -- Definition to be implemented

-- State the theorem
theorem line_parallel_to_plane_line_in_plane 
  (a b : Line3D) (α : Plane3D) 
  (h1 : parallel_line_plane a α) 
  (h2 : line_in_plane b α) : 
  parallel_lines a b ∨ skew_lines a b :=
by
  sorry -- Proof to be implemented


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_line_in_plane_l252_25291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l252_25253

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A color, either red or blue -/
inductive Color
  | Red
  | Blue

/-- A coloring function that assigns a color to each point in the 2x2 square -/
def Coloring := Point → Color

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The theorem statement -/
theorem same_color_unit_distance (c : Coloring) : 
  ∃ (p1 p2 : Point), p1 ≠ p2 ∧ 
    p1.x ∈ Set.Icc 0 2 ∧ p1.y ∈ Set.Icc 0 2 ∧ 
    p2.x ∈ Set.Icc 0 2 ∧ p2.y ∈ Set.Icc 0 2 ∧ 
    c p1 = c p2 ∧ distance p1 p2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l252_25253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_one_problem_two_l252_25234

-- Problem 1
theorem problem_one (m n : ℕ) (h1 : 3^m = 3) (h2 : 3^n = 2) :
  (9 : ℚ)^(m - 1 - 2*n) = 1 / 16 := by sorry

-- Problem 2
theorem problem_two (x y : ℝ) (h : 3*x - 2*y - 2 = 0) :
  (8 : ℝ)^x / ((4 : ℝ)^y * (2 : ℝ)^2) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_one_problem_two_l252_25234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_draw_three_same_color_draw_five_min_seven_points_l252_25292

def num_red_balls : ℕ := 4
def num_white_balls : ℕ := 6

def score_red_ball : ℕ := 2
def score_white_ball : ℕ := 1

def ways_to_draw_same_color (n : ℕ) : ℕ :=
  Nat.choose num_red_balls n + Nat.choose num_white_balls n

def ways_to_draw_with_score (n : ℕ) (min_score : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (fun r ↦
    if r * score_red_ball + (n - r) * score_white_ball ≥ min_score
    then Nat.choose num_red_balls r * Nat.choose num_white_balls (n - r)
    else 0)

theorem draw_three_same_color :
  ways_to_draw_same_color 3 = 24 := by sorry

theorem draw_five_min_seven_points :
  ways_to_draw_with_score 5 7 = 126 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_draw_three_same_color_draw_five_min_seven_points_l252_25292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_nine_l252_25259

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

theorem arithmetic_sequence_sum_nine (seq : ArithmeticSequence) 
  (h3 : seq.a 3 ^ 2 = seq.a 2 * seq.a 7)  -- a₃ is geometric mean of a₂ and a₇
  (h4 : seq.a 1 = 2)  -- a₁ = 2
  : sum_n seq 9 = -90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_nine_l252_25259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_segment_theorem_l252_25224

/-- Triangle with side lengths a, b, c and area S -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ

/-- Constant k depending on the triangle -/
noncomputable def k (t : Triangle) : ℝ := sorry

/-- Angle PDQ in the triangle -/
noncomputable def angle_PDQ (t : Triangle) : ℝ := sorry

/-- The length of the shortest segment dividing the triangle's area in two equal parts -/
noncomputable def shortest_segment_length (t : Triangle) : ℝ :=
  2 * Real.sqrt (t.S / k t) * Real.sin (angle_PDQ t / 2)

/-- The number of shortest segments dividing the triangle's area in two equal parts -/
def num_shortest_segments : ℕ := 3

/-- Predicate to check if a segment divides the triangle's area into two equal parts -/
def divides_area_equally (t : Triangle) (PQ : ℝ) : Prop := sorry

theorem shortest_segment_theorem (t : Triangle) :
  ∀ (PQ : ℝ), PQ ≥ shortest_segment_length t ∧
  (PQ = shortest_segment_length t ↔ divides_area_equally t PQ) ∧
  num_shortest_segments = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_segment_theorem_l252_25224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l252_25238

theorem negation_of_sin_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l252_25238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_diameter_of_specific_triangle_l252_25263

-- Define a triangle
structure Triangle where
  side : ℝ
  angle : ℝ

-- Define the diameter of the circumscribed circle
noncomputable def circumscribedCircleDiameter (t : Triangle) : ℝ :=
  t.side / Real.sin t.angle

-- Theorem statement
theorem circumscribed_circle_diameter_of_specific_triangle :
  let t : Triangle := { side := 16, angle := π / 4 }
  circumscribedCircleDiameter t = 16 * Real.sqrt 2 := by
  -- Unfold the definition of circumscribedCircleDiameter
  unfold circumscribedCircleDiameter
  -- Simplify the expression
  simp [Triangle.side, Triangle.angle]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_diameter_of_specific_triangle_l252_25263
