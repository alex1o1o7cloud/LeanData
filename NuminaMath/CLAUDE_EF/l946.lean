import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_locus_l946_94678

/-- Given a point A(0, a) and a line g represented by y = 0,
    the locus of the third vertices of equilateral triangles
    with A and a point P(λ, 0) on g is represented by two lines. -/
theorem equilateral_triangle_locus 
  (a : ℝ) : 
  ∃ (f g : ℝ → ℝ → Prop), 
    (∀ x y, f x y ↔ Real.sqrt 3 * x - y + a = 0) ∧ 
    (∀ x y, g x y ↔ Real.sqrt 3 * x + y - a = 0) ∧ 
    (∀ l : ℝ, 
      ∃ (x y : ℝ), 
        ((x - 0)^2 + (y - a)^2 = l^2 + a^2) ∧ 
        ((x - l)^2 + y^2 = l^2 + a^2) ∧ 
        (f x y ∨ g x y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_locus_l946_94678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_30_l946_94677

/-- The shaded area between a large square, four inscribed circles, and an inner square --/
noncomputable def shadedArea (largeSquareSide : ℝ) : ℝ :=
  let circleRadius := largeSquareSide / 4
  let largeSquareArea := largeSquareSide ^ 2
  let circleArea := 4 * Real.pi * circleRadius ^ 2
  let innerSquareSide := Real.sqrt (largeSquareSide ^ 2 - 2 * circleRadius ^ 2)
  let innerSquareArea := innerSquareSide ^ 2
  largeSquareArea - circleArea - innerSquareArea

/-- Theorem stating the shaded area for a square with side length 30 --/
theorem shaded_area_30 : shadedArea 30 = 112.5 - 225 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_30_l946_94677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_rolling_2_4_or_8_l946_94607

/-- A fair 8-sided die -/
structure FairDie where
  sides : Nat
  is_fair : sides = 8

/-- The set of favorable outcomes (2, 4, and 8) -/
def favorable_outcomes : Finset Nat := {2, 4, 8}

/-- The probability of rolling a number in the favorable outcomes set -/
def probability (d : FairDie) : ℚ :=
  (favorable_outcomes.card : ℚ) / d.sides

theorem probability_of_rolling_2_4_or_8 (d : FairDie) :
  probability d = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_rolling_2_4_or_8_l946_94607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l946_94697

noncomputable def a : ℝ := (5/3) ^ (1/6 : ℝ)
noncomputable def b : ℝ := (3/5) ^ (-(1/5 : ℝ))
noncomputable def c : ℝ := Real.log (2/3)

theorem size_relationship : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l946_94697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_covered_digits_correct_l946_94655

/-- Represents a four-digit integer with some digits possibly covered -/
structure PartiallyHiddenNumber where
  thousands : Option Nat
  hundreds : Option Nat
  tens : Option Nat
  ones : Nat

/-- The sum of the three partially hidden numbers -/
def total_sum : Nat := 10126

/-- The three partially hidden numbers -/
def numbers : Vector PartiallyHiddenNumber 3 :=
  ⟨[
    { thousands := some 1, hundreds := some 2, tens := some 4, ones := 3 },
    { thousands := some 2, hundreds := some 1, tens := none, ones := 7 },
    { thousands := none, hundreds := none, tens := some 2, ones := 6 }
  ], rfl⟩

/-- The covered digits -/
def covered_digits : Finset Nat := {5, 6, 7}

/-- Helper function to get the value of an Option Nat or 0 -/
def getOrZero (x : Option Nat) : Nat :=
  match x with
  | some n => n
  | none => 0

theorem covered_digits_correct :
  ∃ (a b c : Nat),
    a ∈ covered_digits ∧ b ∈ covered_digits ∧ c ∈ covered_digits ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (getOrZero (numbers.get 0).thousands * 1000 + getOrZero (numbers.get 0).hundreds * 100 + getOrZero (numbers.get 0).tens * 10 + (numbers.get 0).ones) +
    (getOrZero (numbers.get 1).thousands * 1000 + getOrZero (numbers.get 1).hundreds * 100 + a * 10 + (numbers.get 1).ones) +
    (b * 1000 + c * 100 + getOrZero (numbers.get 2).tens * 10 + (numbers.get 2).ones) = total_sum :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_covered_digits_correct_l946_94655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_square_sum_theorem_l946_94632

/-- A four-digit number represented as a pair of two-digit numbers -/
structure FourDigitNumber where
  ab : Nat
  cd : Nat
  h1 : ab ≥ 10 ∧ ab < 100
  h2 : cd ≥ 0 ∧ cd < 100

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  100 * n.ab + n.cd

/-- The property that a four-digit number satisfies the given condition -/
def satisfies_condition (n : FourDigitNumber) : Prop :=
  n.value = (n.ab + n.cd)^2

/-- The theorem stating that only three specific numbers satisfy the condition -/
theorem four_digit_square_sum_theorem :
  ∀ n : FourDigitNumber, satisfies_condition n ↔ n.value ∈ ({9801, 2025, 3025} : Set Nat) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_square_sum_theorem_l946_94632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2001_value_l946_94687

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Define for 0 to cover all natural numbers
  | n + 1 => -1 / (sequence_a n + 1)

theorem a_2001_value : sequence_a 2001 = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2001_value_l946_94687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_x_l946_94639

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the symmetry property
def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 - (x - 1)) = f (x - 1)

-- State the theorem
theorem f_negative_x (h1 : symmetric_about_one f) (h2 : ∀ x > 0, f x = x^2 - 2*x) :
  ∀ x < 0, f x = x^2 + 2*x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_x_l946_94639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expression_l946_94664

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression (x^2 - 1/x)^6
noncomputable def expression (x : ℝ) : ℝ := (x^2 - 1/x)^6

-- Theorem statement
theorem constant_term_of_expression : 
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 → 
    expression x = c + x * (expression x - c) ∧ c = (binomial 6 4 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expression_l946_94664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l946_94657

open Real

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * ω * x) - Real.cos (2 * ω * x)

-- State the theorem
theorem monotone_increasing_interval
  (ω : ℝ)
  (h_ω : ω ∈ Set.Ioo 0 1)
  (h_f_zero : f ω (π / 6) = 0) :
  ∃ (a b : ℝ),
    a = 0 ∧ b = 2 * π / 3 ∧
    StrictMonoOn (f ω) (Set.Icc a b) ∧
    ∀ c d, c ∈ Set.Icc 0 π → d ∈ Set.Icc 0 π → c < d →
      StrictMonoOn (f ω) (Set.Icc c d) → d - c ≤ b - a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l946_94657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l946_94689

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def point_B : ℝ × ℝ := (3, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_distance_theorem 
  (A : ℝ × ℝ) -- Point A
  (h1 : parabola A.1 A.2) -- A lies on the parabola
  (h2 : distance A focus = distance point_B focus) -- |AF| = |BF|
  : distance A point_B = 2 * Real.sqrt 2 := by
  sorry

#check parabola_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l946_94689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_area_l946_94670

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - 2 * (Real.sin (ω * x / 2))^2

theorem function_properties_and_triangle_area 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + 3 * Real.pi) = f ω x) :
  -- Part 1: Maximum value
  (∃ x ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, f ω x = 1) ∧ 
  (∀ x ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, f ω x ≤ 1) ∧
  -- Part 2: Minimum value
  (∃ x ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, f ω x = -Real.sqrt 3 - 1) ∧ 
  (∀ x ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, f ω x ≥ -Real.sqrt 3 - 1) ∧
  -- Part 3: Triangle area
  (∀ a b c A B C : ℝ,
    b = 2 → 
    f ω A = Real.sqrt 3 - 1 → 
    Real.sqrt 3 * a = 2 * b * Real.sin A →
    (1/2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_area_l946_94670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_integer_l946_94683

theorem exists_close_integer (a : ℝ) (h : a = (a - 1)^3) :
  ∃ N : ℤ, |a^2021 - ↑N| < (1/2)^1000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_integer_l946_94683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_and_average_value_l946_94695

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 4)) / x

theorem integral_and_average_value :
  (∫ x in (5 : ℝ)..(12 : ℝ), f x) = 2 * Real.log (5 * Real.exp 1 / 3) ∧
  (∫ x in (5 : ℝ)..(12 : ℝ), f x) / (12 - 5) = (2 / 7) * Real.log (5 * Real.exp 1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_and_average_value_l946_94695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l946_94665

theorem vector_magnitude_problem (x y : ℝ) : 
  let a : ℝ × ℝ := (x, y)
  let b : ℝ × ℝ := (-1, 2)
  (x + -1 = 1 ∧ y + 2 = 3) → 
  ‖a - (2 : ℝ) • b‖ = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l946_94665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_one_l946_94611

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Define for 0 to cover all natural numbers
  | n + 1 => 1 / (1 - sequence_a n)

theorem a_2016_equals_negative_one : sequence_a 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_one_l946_94611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l946_94619

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2*x + a) / Real.log 0.5

def q (a : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (-(5-2*a))^x₁ > (-(5-2*a))^x₂

-- Define the main theorem
theorem range_of_a : 
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → 1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l946_94619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_event_red_ball_l946_94635

/-- A pocket containing colored balls -/
structure Pocket where
  red : Nat
  white : Nat

/-- The probability of drawing at least one red ball when drawing two balls from the pocket -/
noncomputable def prob_at_least_one_red (p : Pocket) : ℝ :=
  1 - (p.white.choose 2 : ℝ) / ((p.red + p.white).choose 2 : ℝ)

/-- Theorem: Given a pocket with 2 red balls and 1 white ball, 
    the probability of drawing at least one red ball when drawing two balls is 1 -/
theorem certain_event_red_ball (p : Pocket) (h1 : p.red = 2) (h2 : p.white = 1) :
  prob_at_least_one_red p = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_event_red_ball_l946_94635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l946_94606

theorem triangle_problem (AC AB : ℝ) (BC : ℕ) (x y : ℕ) : 
  AC = 3.8 →
  AB = 0.6 →
  (BC : ℝ) < AC + AB →
  (BC : ℝ) > AC - AB →
  y < 7 →
  x < 1 + y →
  x > 5 →
  y > x - 1 →
  BC = 4 ∧ x = 6 ∧ y = 6 := by
  sorry

#check triangle_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l946_94606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l946_94675

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 8*x

/-- The focus of the parabola y^2 = 8x -/
def focus : ℝ × ℝ := (2, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: If a point on the parabola y^2 = 8x is 10 units from the focus, its coordinates are (8, ±8) -/
theorem parabola_point_coordinates (P : ParabolaPoint) :
  distance (P.x, P.y) focus = 10 → P.x = 8 ∧ (P.y = 8 ∨ P.y = -8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l946_94675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l946_94659

noncomputable def f (x : ℝ) : ℝ := (x^3 - 2*x^2 + 3*x - 4) / (x^3 - 3*x^2 - 4*x + 12)

theorem domain_of_f : 
  {x : ℝ | IsRegular (f x)} = {x : ℝ | x < 3 ∨ x > 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l946_94659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_proof_l946_94647

-- Define the variables and equations
noncomputable def x : ℝ := 100.48
noncomputable def y : ℝ := 100.70

-- Define z as a function of x and y
noncomputable def z (x y : ℝ) : ℝ := y^2 / x

-- Define the equation for w (although we don't solve for w)
def w_equation (x y z w : ℝ) : Prop := w^3 - x*w = y*z

-- Theorem statement
theorem z_value_proof :
  let z_val := z x y
  ∀ w, w_equation x y z_val w →
  |z_val - 100.92| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_proof_l946_94647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiple_with_digit_sum_l946_94623

/-- Helper function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- For any positive integer n, there exists a positive integer m 
    such that m is divisible by n and the sum of the digits of m is equal to n. -/
theorem exists_multiple_with_digit_sum (n : ℕ) (hn : n > 0) : 
  ∃ m : ℕ, m > 0 ∧ n ∣ m ∧ (sum_of_digits m = n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiple_with_digit_sum_l946_94623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snow_volume_l946_94631

noncomputable def sidewalk_length : ℝ := 30
noncomputable def sidewalk_width : ℝ := 3
noncomputable def garden_path_leg1 : ℝ := 3
noncomputable def garden_path_leg2 : ℝ := 4
noncomputable def snow_depth : ℝ := 3/4

noncomputable def sidewalk_volume : ℝ := sidewalk_length * sidewalk_width * snow_depth
noncomputable def garden_path_area : ℝ := 1/2 * garden_path_leg1 * garden_path_leg2
noncomputable def garden_path_volume : ℝ := garden_path_area * snow_depth

theorem total_snow_volume :
  sidewalk_volume + garden_path_volume = 72 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snow_volume_l946_94631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l946_94605

noncomputable def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

noncomputable def sumArithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem max_sum_arithmetic_sequence
  (a₁ : ℝ) (d : ℝ) :
  (d < 0) →  -- Decreasing sequence
  (sumArithmeticSequence a₁ d 5 = sumArithmeticSequence a₁ d 10) →
  ∀ n : ℕ, sumArithmeticSequence a₁ d n ≤ sumArithmeticSequence a₁ d 7 ∧
           sumArithmeticSequence a₁ d n ≤ sumArithmeticSequence a₁ d 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l946_94605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_square_property_l946_94667

def a : ℕ → ℤ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 12
  | 3 => 20
  | (n + 4) => 2 * a (n + 3) + 2 * a (n + 2) - a (n + 1)

theorem sequence_square_property :
  ∃ b : ℕ → ℤ, (∀ n, b n = a n) ∧
  (∀ n, ∃ k : ℤ, 1 + 4 * (b n) * (b (n + 1)) = k * k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_square_property_l946_94667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l946_94699

/-- Given ellipse equation -/
def given_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- Foci of the given ellipse -/
noncomputable def foci : Set (ℝ × ℝ) := {(Real.sqrt 5, 0), (-Real.sqrt 5, 0)}

/-- The sought ellipse passes through the point (3, -2) -/
def passes_through (f : ℝ → ℝ → Prop) : Prop := f 3 (-2)

/-- Two ellipses have the same foci -/
def same_foci (f g : ℝ → ℝ → Prop) : Prop := 
  ∃ (a b : ℝ), (∀ x y, f x y ↔ x^2 / a + y^2 / b = 1) ∧ 
               (∀ x y, g x y ↔ x^2 / a + y^2 / b = 1)

/-- The sought ellipse equation -/
def sought_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

theorem ellipse_theorem : 
  passes_through sought_ellipse ∧ 
  same_foci given_ellipse sought_ellipse := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l946_94699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brads_speed_is_12_l946_94600

/-- The speed of Brad given the conditions of the problem -/
noncomputable def brads_speed (total_distance : ℝ) (maxwells_speed : ℝ) (maxwells_distance : ℝ) : ℝ :=
  let brads_distance := total_distance - maxwells_distance
  let time := maxwells_distance / maxwells_speed
  brads_distance / time

/-- Theorem stating that Brad's speed is 12 km/h under the given conditions -/
theorem brads_speed_is_12 :
  brads_speed 72 6 24 = 12 := by
  -- Unfold the definition of brads_speed
  unfold brads_speed
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num
  -- QED

-- Note: We can't use #eval for noncomputable functions
-- Instead, we can prove that the result is equal to 12
example : brads_speed 72 6 24 = 12 := brads_speed_is_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brads_speed_is_12_l946_94600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l946_94601

theorem equation_solution (a b : ℝ) (h : a^2 + b^2 - 2*a + 6*b + 10 = 0) :
  2 * a^100 - 3 * b⁻¹ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l946_94601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_solution_set_l946_94622

theorem log_inequality_solution_set :
  {x : ℝ | Real.log (x - 1) < 1} = {x : ℝ | 1 < x ∧ x < 11} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_solution_set_l946_94622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_proof_l946_94698

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem ending_number_proof :
  ∃ (seq : List ℕ), 
    seq.length = 6 ∧ 
    seq.head? = some 2 ∧
    (∀ n ∈ seq, n > 1 ∧ is_power_of_two n) ∧
    (∀ n ∈ seq, ∀ k : ℕ, k > 1 → k % 2 = 1 → n % k ≠ 0) ∧
    seq.getLast? = some 64 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_proof_l946_94698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_B_in_triangle_l946_94625

theorem tan_B_in_triangle (A B C : ℝ) (AC BC : ℝ) :
  AC = 4 →
  BC = 3 →
  Real.cos C = 2/3 →
  Real.tan B = 4 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_B_in_triangle_l946_94625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l946_94646

-- Define the ellipse and its properties
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

-- Define a point on the ellipse
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

-- Define the foci of the ellipse
noncomputable def foci (e : Ellipse) : ℝ × ℝ := 
  let c := Real.sqrt (e.a^2 - e.b^2)
  (-c, c)

-- Define the angle between foci and a point
noncomputable def angle_foci_point (e : Ellipse) (p : PointOnEllipse e) : ℝ :=
  let (f1, f2) := foci e
  Real.arccos ((p.x - f1)^2 + p.y^2 + (p.x - f2)^2 + p.y^2 - 4*(e.a^2 - e.b^2)) / 
               (2 * Real.sqrt ((p.x - f1)^2 + p.y^2) * Real.sqrt ((p.x - f2)^2 + p.y^2))

-- Define the distance from origin to a point
noncomputable def distance_origin_point (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

-- Theorem statement
theorem ellipse_eccentricity (e : Ellipse) 
  (p : PointOnEllipse e)
  (h_angle : angle_foci_point e p = π/3)
  (h_distance : distance_origin_point (p.x, p.y) = (Real.sqrt 3 / 2) * e.a) :
  e.a^2 - e.b^2 = (e.a / 2)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l946_94646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_minus_2x_l946_94673

theorem cos_pi_third_minus_2x (x : ℝ) :
  Real.sin (2 * x + π / 6) = -1 / 3 → Real.cos (π / 3 - 2 * x) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_minus_2x_l946_94673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_15_terms_is_90_l946_94648

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  first : ℝ
  diff : ℝ

/-- The nth term of an arithmetic progression -/
noncomputable def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.first + (n - 1 : ℝ) * ap.diff

/-- The sum of the first n terms of an arithmetic progression -/
noncomputable def ArithmeticProgression.sumFirstN (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * ap.first + (n - 1 : ℝ) * ap.diff)

/-- Theorem: For an arithmetic progression where the sum of the 4th and 12th terms is 12,
    the sum of the first 15 terms is 90. -/
theorem sum_15_terms_is_90 (ap : ArithmeticProgression)
    (h : ap.nthTerm 4 + ap.nthTerm 12 = 12) :
    ap.sumFirstN 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_15_terms_is_90_l946_94648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l946_94642

noncomputable def f (x : ℝ) := Real.cos (x + Real.pi / 6)

noncomputable def g (x : ℝ) := Real.cos (2 * x + Real.pi / 6)

theorem g_decreasing_interval :
  ∀ x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12),
    ∀ y ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12),
      x < y → g y < g x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l946_94642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_given_B_l946_94654

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6 × Fin 6

-- Define event A: all three numbers are different
def A : Set Ω := {ω | ω.1 ≠ ω.2.1 ∧ ω.1 ≠ ω.2.2 ∧ ω.2.1 ≠ ω.2.2}

-- Define event B: at least one 6 appears
def B : Set Ω := {ω | ω.1 = 5 ∨ ω.2.1 = 5 ∨ ω.2.2 = 5}

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- State the theorem
theorem probability_A_given_B : P (A ∩ B) / P B = 60 / 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_given_B_l946_94654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_r_as_m_approaches_zero_l946_94643

-- Define L(m) as the x-coordinate of the left intersection point
noncomputable def L (m : ℝ) : ℝ := -Real.sqrt (6 + m)

-- Define r as a function of m
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- Theorem statement
theorem limit_of_r_as_m_approaches_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ ∧ -6 < m ∧ m < 6 →
    |r m - (1 / Real.sqrt 6)| < ε := by
  sorry

#check limit_of_r_as_m_approaches_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_r_as_m_approaches_zero_l946_94643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_even_odd_probability_l946_94641

def roll_dice (n : ℕ) : ℕ → ℕ := sorry

def is_even (n : ℕ) : Bool := sorry

def probability_equal_even_odd (num_dice : ℕ) : ℚ :=
  (Nat.choose num_dice (num_dice / 2) : ℚ) / (2 ^ num_dice : ℚ)

theorem equal_even_odd_probability :
  probability_equal_even_odd 8 = 35 / 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_even_odd_probability_l946_94641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_increase_l946_94674

/-- Eccentricity of a hyperbola with semi-axes a and b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

theorem hyperbola_eccentricity_increase
  (a b m : ℝ)
  (h1 : a > b)
  (h2 : m > 0) :
  eccentricity a b < eccentricity (a + m) (b + m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_increase_l946_94674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l946_94666

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The asymptote equation -/
def asymptote (x y : ℝ) : Prop := 3 * x - 4 * y = 0

/-- The distance formula from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

theorem distance_to_asymptote :
  ∃ (x y : ℝ), hyperbola x y ∧ asymptote x y ∧
  distance_point_to_line 3 0 3 (-4) 0 = 9/5 := by
  sorry

#check distance_to_asymptote

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l946_94666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_special_case_l946_94680

theorem cos_sum_special_case (α β : ℝ) 
  (h1 : Real.sin α - Real.sin β = 1/2) 
  (h2 : Real.cos α + Real.cos β = 1/4) : 
  Real.cos (α + β) = -27/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_special_case_l946_94680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_town_distance_on_map_l946_94630

/-- Represents the scale of a map -/
structure MapScale where
  inch : ℚ
  miles : ℚ

/-- Calculates the map distance given an actual distance and a map scale -/
def mapDistance (actualDistance : ℚ) (scale : MapScale) : ℚ :=
  (actualDistance * scale.inch) / scale.miles

theorem town_distance_on_map :
  let scale : MapScale := { inch := 1/4, miles := 8 }
  let actualDistance : ℚ := 108
  mapDistance actualDistance scale = 3375/1000 := by
  sorry

#eval (108 : ℚ) * (1/4 : ℚ) / (8 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_town_distance_on_map_l946_94630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jose_alone_time_l946_94662

/-- The time (in days) it takes Jose and Jane to complete the task together -/
noncomputable def combined_time : ℝ := 20

/-- The time (in days) it takes when Jose does half and Jane does half -/
noncomputable def half_half_time : ℝ := 45

/-- Jose's work rate (portion of task completed per day) -/
noncomputable def jose_rate : ℝ := 1 / (2 * half_half_time)

/-- Jane's work rate (portion of task completed per day) -/
noncomputable def jane_rate : ℝ := 1 / (2 * half_half_time)

/-- The combined work rate of Jose and Jane -/
noncomputable def combined_rate : ℝ := 1 / combined_time

-- Assumption that Jane is more efficient than Jose
axiom jane_more_efficient : jane_rate > jose_rate

theorem jose_alone_time : 
  jose_rate * 90 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jose_alone_time_l946_94662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_x_minus_pi_half_l946_94672

theorem cos_neg_x_minus_pi_half (x : ℝ) (h1 : x ∈ Set.Ioo (π/2) π) (h2 : Real.tan x = -4/3) :
  Real.cos (-x - π/2) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_x_minus_pi_half_l946_94672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l946_94604

/-- Given two curves in polar coordinates and a line that intersects both curves,
    prove that the distance between the intersection points is √3. -/
theorem intersection_distance (θ : ℝ) (ρ₁ ρ₂ : ℝ → ℝ) : 
  θ = 2 * Real.pi / 3 →
  ρ₁ = (fun φ => 4 * Real.sin φ) →
  ρ₂ = (fun φ => 2 * Real.sin φ) →
  |ρ₁ θ - ρ₂ θ| = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l946_94604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equations_l946_94661

/-- The circle with center (1, 2) and radius 2 -/
def Circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

/-- The point P through which the tangent line passes -/
def P : ℝ × ℝ := (-1, 5)

/-- A line passing through point P -/
def LinePassingThroughP (a b c : ℝ) : Prop :=
  a * P.1 + b * P.2 + c = 0

/-- The distance from a point (x, y) to the line ax + by + c = 0 -/
noncomputable def DistancePointToLine (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

/-- A line is tangent to the circle if it touches the circle at exactly one point -/
def IsTangentLine (a b c : ℝ) : Prop :=
  LinePassingThroughP a b c ∧
  DistancePointToLine 1 2 a b c = 2 ∧
  ∀ x y, Circle x y → (a * x + b * y + c = 0 → (x, y) = (1, 2) ∨ (x, y) = (1, 2))

theorem tangent_line_equations :
  ∀ a b c : ℝ, IsTangentLine a b c ↔ (a = 5 ∧ b = 12 ∧ c = -55) ∨ (a = 1 ∧ b = 0 ∧ c = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equations_l946_94661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l946_94650

theorem tan_theta_value (θ : ℝ) (z : ℂ) : 
  z = Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5) →
  z.re = 0 →
  z.im ≠ 0 →
  Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l946_94650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_failed_l946_94694

theorem students_failed (total : ℕ) (a_percent : ℚ) (bc_fraction : ℚ) : ℕ :=
  let a_count : ℚ := (total : ℚ) * a_percent
  let bc_count : ℚ := (total - a_count) * bc_fraction
  let failed_count : ℚ := total - a_count - bc_count
  have h : failed_count = 18 := by sorry
  18

#check students_failed

example : students_failed 32 (1/4) (1/4) = 18 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_failed_l946_94694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_get_all_threes_or_fours_l946_94693

def initial_numbers : List ℤ := [1, 2, 3, 4, 5, 6, 7]

def transform (a b : ℚ) : List ℚ := [(2 * a + b) / 3, (a + 2 * b) / 3]

def sum_invariant (xs : List ℚ) : Prop :=
  ∀ a b : ℚ, a ∈ xs → b ∈ xs → a ≠ b →
    (List.sum xs) = (List.sum (transform a b ++ (xs.filter (λ x => x ≠ a ∧ x ≠ b))))

theorem impossible_to_get_all_threes_or_fours :
  (∀ xs : List ℚ, List.length xs = 7 → (∀ x ∈ xs, x = 3) → False) ∧
  (∀ xs : List ℚ, List.length xs = 7 → (∀ x ∈ xs, x = 4) → False) :=
by
  sorry

#check impossible_to_get_all_threes_or_fours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_get_all_threes_or_fours_l946_94693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_l946_94669

def f : ℕ → ℕ
| 0 => 1  -- Adding the case for 0
| 1 => 1
| (n + 1) => f n + 2^(f n)

theorem distinct_remainders : 
  ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 3^2013 ∧ 1 ≤ j ∧ j ≤ 3^2013 ∧ i ≠ j → 
    f i % 3^2013 ≠ f j % 3^2013 :=
by
  sorry

#eval f 5  -- This line is added to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_l946_94669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_to_focus_l946_94691

/-- A parabola with equation x^2 = 4y -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- The focus of the parabola x^2 = 4y -/
def focus : ℝ × ℝ := (0, 1)

/-- Point M on the parabola -/
def M : ℝ × ℝ := (4, 4)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_M_to_focus :
  M ∈ Parabola → distance M focus = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_to_focus_l946_94691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_partition_theorem_l946_94628

/-- Represents a 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Represents a partition of the grid -/
def Partition := Fin 5 → Fin 5 → Fin 5

/-- Checks if a cell is marked -/
def is_marked (g : Grid) (i j : Fin 5) : Prop := g i j = true

/-- Counts the number of marked cells in the grid -/
def marked_count (g : Grid) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 5)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 5)) (λ j =>
      if g i j then 1 else 0))

/-- Checks if two shapes in the partition are congruent -/
def are_congruent (p : Partition) (n m : Fin 5) : Prop :=
  ∃ (f : Fin 5 × Fin 5 → Fin 5 × Fin 5),
    Function.Bijective f ∧
    ∀ (i j : Fin 5), (p i j = n) = (p (f (i, j)).1 (f (i, j)).2 = m)

/-- Main theorem -/
theorem grid_partition_theorem (g : Grid) (p : Partition) :
  marked_count g = 5 →
  (∀ n : Fin 5, ∃! (i j : Fin 5), is_marked g i j ∧ p i j = n) →
  (∀ n m : Fin 5, are_congruent p n m) →
  True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_partition_theorem_l946_94628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l946_94653

/-- The distance between two parallel lines in 2D space -/
noncomputable def distance_between_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance between lines l₁: 2x + y + 1 = 0 and l₂: 4x + 2y - 1 = 0 is 3√5 / 10 -/
theorem distance_between_given_lines :
  distance_between_lines 4 2 2 (-1) = 3 * Real.sqrt 5 / 10 := by
  sorry

#check distance_between_given_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l946_94653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l946_94684

noncomputable def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

noncomputable def TriangleArea (a b : ℝ) : ℝ := b * Real.sqrt (a^2 - b^2)

def Line (k m : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + m}

theorem ellipse_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : Eccentricity a b = Real.sqrt 3 / 3)
  (h4 : ∀ M ∈ Ellipse a b, TriangleArea a b ≤ Real.sqrt 2)
  (h5 : ∃ M ∈ Ellipse a b, TriangleArea a b = Real.sqrt 2) :
  (∃ k m : ℝ, ∀ A B : ℝ × ℝ,
    A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧
    A ∈ Line k m ∧ B ∈ Line k m ∧
    A ≠ B ∧
    (∃ t : ℝ, t ≠ 0 ∧ Line k m = {p : ℝ × ℝ | p.2 - (k * p.1 + m) = t}) ∧
    (∃ r : ℝ, (A.2 - (-a, 0).2) / (A.1 - (-a, 0).1) + r = k ∧
              k + r = (B.2 - (-a, 0).2) / (B.1 - (-a, 0).1)) →
    (a^2 = 3 ∧ b^2 = 2) ∧
    (m < -2 * Real.sqrt 6 / 3 ∨ m > 2 * Real.sqrt 6 / 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l946_94684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_is_6000_l946_94696

/-- Represents the capital of a partner -/
structure Capital where
  amount : ℚ
  positive : amount > 0

/-- Represents a business partnership -/
structure Partnership where
  a : Capital
  b : Capital
  c : Capital
  twice_a_eq_thrice_b : 2 * a.amount = 3 * b.amount
  b_eq_four_c : b.amount = 4 * c.amount
  total_profit : ℚ
  profit_is_positive : total_profit > 0

/-- Calculate the share of profit for partner B -/
def calculate_b_share (p : Partnership) : ℚ :=
  (p.b.amount / (p.a.amount + p.b.amount + p.c.amount)) * p.total_profit

/-- Theorem stating that B's share of the profit is 6000 given the conditions -/
theorem b_share_is_6000 (p : Partnership) (h : p.total_profit = 16500) :
  calculate_b_share p = 6000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_is_6000_l946_94696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l946_94617

noncomputable section

def a : ℝ × ℝ := (3, 4)

theorem vector_problem (x y : ℝ) :
  let b : ℝ × ℝ := (9, x)
  let c : ℝ × ℝ := (4, y)
  let m : ℝ × ℝ := (2 * a.1 - 9, 2 * a.2 - x)
  let n : ℝ × ℝ := (a.1 + 4, a.2 + y)
  (a.1 * x = a.2 * 9) →  -- a is parallel to b
  (a.1 * 4 + a.2 * y = 0) →  -- a is perpendicular to c
  (b = (9, 12) ∧ c = (4, -3) ∧ 
   Real.arccos ((m.1 * n.1 + m.2 * n.2) / 
    (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2))) = 3 * Real.pi / 4) :=
by
  intros b c m n h1 h2
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l946_94617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_remainders_is_98_l946_94603

-- Define a function to represent m in terms of n
def m (n : ℕ) : ℕ := 1111 * n + 123

-- Define the property of m having four consecutive increasing digits
def has_four_consecutive_increasing_digits (m : ℕ) : Prop :=
  ∃ n : ℕ, n ≤ 6 ∧ m = 1111 * n + 123

-- Define the sum of remainders
def sum_of_remainders : ℕ := Finset.sum (Finset.range 7) (fun n => (n + 11) % 23)

-- Theorem statement
theorem sum_of_remainders_is_98 :
  sum_of_remainders = 98 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_remainders_is_98_l946_94603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l946_94676

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (5 + (Real.sqrt 3 / 2) * t, Real.sqrt 3 + (1 / 2) * t)

-- Define the curve C in Cartesian coordinates
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define point M
def point_M : ℝ × ℝ := (5, Real.sqrt 3)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), 
    line_l t₁ = A ∧ 
    line_l t₂ = B ∧ 
    curve_C A.1 A.2 ∧ 
    curve_C B.1 B.2 ∧
    t₁ ≠ t₂

-- Theorem statement
theorem intersection_distance_product (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - point_M.1)^2 + (A.2 - point_M.2)^2) *
  Real.sqrt ((B.1 - point_M.1)^2 + (B.2 - point_M.2)^2) = 18 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l946_94676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_proof_l946_94668

/-- The point in the xy-plane that is equidistant from three given points -/
def equidistant_point : ℝ × ℝ × ℝ := (3.7, -0.6, 0)

/-- The first given point -/
def point1 : ℝ × ℝ × ℝ := (1, 0, -1)

/-- The second given point -/
def point2 : ℝ × ℝ × ℝ := (2, 2, 1)

/-- The third given point -/
def point3 : ℝ × ℝ × ℝ := (4, 1, -2)

/-- Calculate the square of the distance between two points in 3D space -/
def distance_squared (p q : ℝ × ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2

/-- Theorem stating that the equidistant_point is equidistant from the three given points -/
theorem equidistant_proof :
  distance_squared equidistant_point point1 = distance_squared equidistant_point point2 ∧
  distance_squared equidistant_point point1 = distance_squared equidistant_point point3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_proof_l946_94668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_count_l946_94624

theorem floor_sqrt_count : 
  (Finset.filter (fun x : ℕ => Int.floor (Real.sqrt (x : ℝ)) = 11) (Finset.range 144)).card = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_count_l946_94624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l946_94690

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos ((1/2) * x + (2/3) * Real.pi)

theorem f_increasing_interval :
  ∀ x ∈ Set.Icc ((2/3) * Real.pi) (2 * Real.pi),
    ∀ y ∈ Set.Icc ((2/3) * Real.pi) (2 * Real.pi),
      x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l946_94690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l946_94626

/-- Defines what it means for a point to be the focus of a parabola -/
def is_focus (f : ℝ × ℝ) (x y : ℝ) : Prop :=
  ∃ (d : ℝ), 
    (x - f.1)^2 + (y - f.2)^2 = (x - d)^2 ∧ 
    f.1 ≠ d ∧
    f.2 = 0

/-- The focus of the parabola x = -1/16 * y^2 is the point (-4, 0) -/
theorem parabola_focus (x y : ℝ) :
  (x = -1/16 * y^2) → (∃ (f : ℝ), f = -4 ∧ is_focus (f, 0) x y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l946_94626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_five_digit_number_l946_94660

def digits : List Nat := [0, 3, 4, 8, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  (Nat.digits 10 n).filter (λ d => d ∈ digits) = (Nat.digits 10 n) ∧
  (Nat.digits 10 n).toFinset = digits.toFinset

theorem largest_five_digit_number :
  ∀ n : Nat, is_valid_number n → n ≤ 98430 :=
by
  sorry

#check largest_five_digit_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_five_digit_number_l946_94660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_total_time_l946_94612

-- Define the total distance from home to school
noncomputable def total_distance : ℝ := 1

-- Define Joe's walking speed
noncomputable def walking_speed : ℝ := total_distance / (3 * 9)

-- Define Joe's biking speed (5 times walking speed)
noncomputable def biking_speed : ℝ := 5 * walking_speed

-- Define the walking time (given as 9 minutes)
noncomputable def walking_time : ℝ := 9

-- Define the biking time
noncomputable def biking_time : ℝ := (2 * total_distance / 3) / biking_speed

-- Theorem: The total time taken by Joe is 12.6 minutes
theorem joe_total_time : walking_time + biking_time = 12.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_total_time_l946_94612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l946_94638

theorem range_of_a (a : ℝ) : 
  (∀ θ : ℝ, θ ∈ Set.Ioo 0 (π/2) → a ≤ 1/Real.sin θ + 1/Real.cos θ) → 
  a ∈ Set.Iic (2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l946_94638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l946_94681

/-- Sum of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: The sum of the first ten terms of an arithmetic sequence 
    starting from 5 with common difference 4 is 230 -/
theorem arithmetic_sequence_sum :
  arithmetic_sum 5 4 10 = 230 := by
  -- Unfold the definition of arithmetic_sum
  unfold arithmetic_sum
  -- Simplify the expression
  simp
  -- Check that the simplified expression equals 230
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l946_94681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_sample_analysis_l946_94608

theorem oil_sample_analysis (total_samples : ℕ) 
  (heavy_oil_freq : ℚ) (light_low_sulfur_freq : ℚ) :
  total_samples = 296 →
  heavy_oil_freq = 1/8 →
  light_low_sulfur_freq = 22/37 →
  (∃ (high_sulfur_samples : ℕ), high_sulfur_samples = 142) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_sample_analysis_l946_94608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purchase_price_approximately_35_23_l946_94688

/-- Calculates the purchase price of shares given dividend rate, face value, nominal ROI, and inflation rate -/
noncomputable def calculate_purchase_price (dividend_rate : ℝ) (face_value : ℝ) (nominal_roi : ℝ) (inflation_rate : ℝ) : ℝ :=
  let actual_roi := nominal_roi - inflation_rate
  let dividend_per_share := (dividend_rate / 100) * face_value
  dividend_per_share / (actual_roi / 100)

/-- Theorem stating that the calculated purchase price is approximately 35.23 given the problem conditions -/
theorem purchase_price_approximately_35_23 :
  let dividend_rate : ℝ := 15.5
  let face_value : ℝ := 50
  let nominal_roi : ℝ := 25
  let inflation_rate : ℝ := 3
  let calculated_price := calculate_purchase_price dividend_rate face_value nominal_roi inflation_rate
  abs (calculated_price - 35.23) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purchase_price_approximately_35_23_l946_94688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_formula_l946_94644

/-- Represents an isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  a : ℝ  -- Length of the two equal sides
  b : ℝ  -- Length of the base
  r : ℝ  -- Radius of the inscribed circle
  h_positive : a > 0 ∧ b > 0 ∧ r > 0  -- Positive lengths
  h_isosceles : a > b / 2  -- Condition for triangle inequality

/-- The ratio of the area of the inscribed circle to the area of the isosceles triangle -/
noncomputable def areaRatio (t : IsoscelesTriangleWithInscribedCircle) : ℝ :=
  (Real.pi * t.r) / (t.a + t.b + t.r)

/-- Theorem stating that the area ratio formula is correct -/
theorem area_ratio_formula (t : IsoscelesTriangleWithInscribedCircle) :
  areaRatio t = (Real.pi * t.r^2) / (t.r * ((2 * t.a + t.b) / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_formula_l946_94644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_place_winnings_l946_94656

def total_participants : ℕ := 13
def contribution_per_person : ℚ := 5
def first_place_percentage : ℚ := 65 / 100
def remaining_percentage : ℚ := 35 / 100

def total_pot : ℚ := total_participants * contribution_per_person

def first_place_amount : ℚ := total_pot * first_place_percentage
def remaining_amount : ℚ := total_pot * remaining_percentage

def third_place_amount : ℚ := remaining_amount / 2

theorem third_place_winnings :
  (third_place_amount * 100).floor / 100 = 1138 / 100 := by
  -- Proof steps would go here
  sorry

#eval (third_place_amount * 100).floor / 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_place_winnings_l946_94656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_is_27_l946_94649

-- Define the prism
structure Prism where
  base_side : ℝ
  height : ℝ
  base_side_is_sqrt_9 : base_side = Real.sqrt 9
  prism_height_is_6 : height = 6

-- Define the volume calculation function
noncomputable def prism_volume (p : Prism) : ℝ :=
  (1/2) * p.base_side * p.base_side * p.height

-- Theorem statement
theorem prism_volume_is_27 (p : Prism) : prism_volume p = 27 := by
  -- Unfold the definition of prism_volume
  unfold prism_volume
  -- Use the properties of the Prism structure
  rw [p.base_side_is_sqrt_9, p.prism_height_is_6]
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- Perform the arithmetic
  norm_num
  -- QED
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_is_27_l946_94649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l946_94621

theorem cosine_inequality (α β : ℝ) (k : ℕ+) 
  (h1 : Real.cos α ≠ Real.cos β) 
  (h2 : k > 1) : 
  |((Real.cos (k * β) * Real.cos α) - (Real.cos (k * α) * Real.cos β)) / (Real.cos β - Real.cos α)| < (k : ℝ)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l946_94621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l946_94616

theorem equation_solution (θ : Real) (x : Real) :
  (Real.cos (θ / 2))^2 * x^3 + (3 * (Real.cos (θ / 2))^2 - 4) * x + Real.sin θ = 0 →
  (Real.cos (θ / 2) = 0 ∧ x = 0) ∨
  (Real.cos (θ / 2) ≠ 0 ∧ 
    (x = 2 * Real.tan (θ / 2) ∨ 
     x = -Real.tan (θ / 2) + 1 / Real.cos (θ / 2) ∨
     x = -Real.tan (θ / 2) - 1 / Real.cos (θ / 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l946_94616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l946_94609

noncomputable section

-- Define the right triangle ABC
def RightTriangleABC (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1) * (B.2 - C.2) = (B.1 - C.1) * (A.2 - B.2)

-- Define the cosine of angle C
noncomputable def CosC (A B C : ℝ × ℝ) : ℝ :=
  (C.1 - A.1) / Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)

-- Define the length of side BC
noncomputable def BC (B C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)

-- Define the length of side AC
noncomputable def AC (A C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)

-- Theorem statement
theorem triangle_side_length 
  (A B C : ℝ × ℝ) 
  (h1 : RightTriangleABC A B C) 
  (h2 : CosC A B C = 9 * Real.sqrt 130 / 130) 
  (h3 : BC B C = Real.sqrt 130) : 
  AC A C = 9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l946_94609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_explicit_l946_94614

/-- A quadratic polynomial satisfying specific conditions -/
def q : ℝ → ℝ := sorry

/-- The polynomial q satisfies q(-3) = 0 -/
axiom q_at_neg_three : q (-3) = 0

/-- The polynomial q satisfies q(2) = 0 -/
axiom q_at_two : q 2 = 0

/-- The polynomial q satisfies q(-1) = -8 -/
axiom q_at_neg_one : q (-1) = -8

/-- The polynomial q is quadratic -/
axiom q_is_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c

/-- The theorem stating that q(x) = 4/3x^2 + 4/3x - 8 -/
theorem q_explicit : ∀ x, q x = 4/3 * x^2 + 4/3 * x - 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_explicit_l946_94614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l946_94679

noncomputable section

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a parabola -/
def PointOnParabola (c : Parabola) := { p : ℝ × ℝ // p.2 = c.p * p.1^2 }

/-- Focus of a parabola -/
def focus (c : Parabola) : ℝ × ℝ := (0, c.p / 2)

/-- Tangent line to a parabola at a point -/
def tangentLine (c : Parabola) (p : PointOnParabola c) : ℝ × ℝ → Prop :=
  λ q ↦ q.2 = (p.val.1 / c.p) * q.1 - p.val.1^2 / (2 * c.p)

/-- Intersection of a line with x-axis -/
def xAxisIntersection (l : ℝ × ℝ → Prop) : ℝ × ℝ := 
  sorry

/-- Intersection of a line with y-axis -/
def yAxisIntersection (l : ℝ × ℝ → Prop) : ℝ × ℝ := 
  sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := 
  sorry

/-- Angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := 
  sorry

/-- Main theorem -/
theorem parabola_properties (c : Parabola) 
  (p : PointOnParabola c) 
  (l : ℝ × ℝ → Prop) 
  (hl : l = tangentLine c p) 
  (d : ℝ × ℝ) 
  (hd : d = xAxisIntersection l) 
  (q : ℝ × ℝ) 
  (hq : q = yAxisIntersection l) 
  (hfd : distance (focus c) d = 2) 
  (hangle : angle p.val (focus c) d = π/3) :
  (∃ (a b h : PointOnParabola c), 
    a.val = (0, 0) ∧ 
    b.val = (4, 4) ∧ 
    h.val ≠ a.val ∧ 
    h.val ≠ b.val ∧ 
    h.val = (-2, 1) ∧
    (∃ (circle : Set (ℝ × ℝ)), 
      a.val ∈ circle ∧ 
      b.val ∈ circle ∧ 
      h.val ∈ circle ∧
      (∀ (t : ℝ × ℝ → Prop), 
        (t = tangentLine c h) → 
        (∃ (s : ℝ × ℝ → Prop), 
          (∀ (x : ℝ × ℝ), x ∈ circle → s x) ∧ 
          (∀ (x : ℝ × ℝ), s x → t x))))) ∧ 
  (distance (focus c) p.val = distance (focus c) q) ∧
  c.p = 2 := by
    sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l946_94679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_GF_FC_ratio_l946_94645

noncomputable section

variable (A B C D E F G : EuclideanSpace ℝ (Fin 2))

/-- Triangle ABC with specific points D, E, F, and G -/
structure TriangleABC where
  -- D is on AB with AD:DB = 4:1
  D_on_AB : ∃ t : ℝ, D = (1 - t) • A + t • B ∧ t = 1/5
  -- E is on BC with BE:EC = 2:3
  E_on_BC : ∃ s : ℝ, E = (1 - s) • B + s • C ∧ s = 3/5
  -- F is the intersection of DE and AC
  F_on_DE : ∃ u : ℝ, F = (1 - u) • D + u • E
  F_on_AC : ∃ v : ℝ, F = (1 - v) • A + v • C
  -- G is on AC with AG:GC = 3:2
  G_on_AC : ∃ w : ℝ, G = (1 - w) • A + w • C ∧ w = 2/5

/-- The main theorem: GF:FC = 3:2 -/
theorem GF_FC_ratio (t : TriangleABC A B C D E F G) : 
  ∃ k : ℝ, (G - F) = k • (F - C) ∧ k = 3/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_GF_FC_ratio_l946_94645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_inscribed_sphere_radius_l946_94663

-- Define a tetrahedron type
structure Tetrahedron where
  altitudes : Fin 4 → ℝ
  all_altitudes_ge_one : ∀ i, altitudes i ≥ 1

-- Define the inscribed sphere radius function
noncomputable def inscribed_sphere_radius (t : Tetrahedron) : ℝ :=
  1 / (t.altitudes 0)⁻¹ + (t.altitudes 1)⁻¹ + (t.altitudes 2)⁻¹ + (t.altitudes 3)⁻¹

-- State the theorem
theorem maximal_inscribed_sphere_radius :
  (∀ t : Tetrahedron, inscribed_sphere_radius t ≥ (1 : ℝ) / 4) ∧
  (∃ t : Tetrahedron, inscribed_sphere_radius t = (1 : ℝ) / 4) := by
  sorry

#check maximal_inscribed_sphere_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_inscribed_sphere_radius_l946_94663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_99_a_100_l946_94610

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 0  -- Add a case for 0 to avoid the "missing cases" error
  | 1 => 1 / 1
  | 2 => 2 / 1
  | 3 => 1 / 2
  | 4 => 3 / 1
  | 5 => 2 / 2
  | 6 => 1 / 3
  | 7 => 4 / 1
  | 8 => 3 / 2
  | 9 => 2 / 3
  | 10 => 1 / 4
  | n + 1 => sorry  -- We don't need to define the rest of the sequence for this theorem

-- State the theorem
theorem sum_a_99_a_100 : a 99 + a 100 = 37 / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_99_a_100_l946_94610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cos_and_sin_l946_94640

theorem intersection_of_cos_and_sin (φ : ℝ) : 
  0 ≤ φ → φ < π → Real.cos (π/3) = Real.sin (2*(π/3) + φ) → φ = π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cos_and_sin_l946_94640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_8pi_l946_94615

/-- The equation of the region boundary -/
def region_boundary (x y : ℝ) : Prop :=
  x^2 + 4*y^2 = 4*|x-y| + 4*|x+y|

/-- The region enclosed by the boundary -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | region_boundary p.1 p.2}

/-- The area of the enclosed region -/
noncomputable def area : ℝ := (MeasureTheory.volume enclosed_region).toReal

/-- Theorem: The area of the region is 8π -/
theorem area_is_8pi : area = 8 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_8pi_l946_94615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_union_N_not_M_l946_94602

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- Define the complement of M in ℝ
def not_M : Set ℝ := {x | x ≤ 1}

-- Theorem for the intersection of M and N
theorem intersection_M_N : M ∩ N = Set.Ioo 1 3 := by sorry

-- Theorem for the union of N and the complement of M
theorem union_N_not_M : N ∪ not_M = Set.Iic 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_union_N_not_M_l946_94602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cosB_l946_94636

theorem right_triangle_cosB (A B C : ℝ × ℝ) : 
  (A.1 = 0 ∧ A.2 = 0) →  -- A is at origin
  (B.1 = 8 ∧ B.2 = 0) →  -- B is on x-axis, 8 units from origin
  (C.1 = 0 ∧ C.2 = 6) →  -- C is on y-axis, 6 units from origin
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 8^2 →  -- AB = 8
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 10^2 →  -- AC = 10
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 6^2 →  -- BC = 6 (right angle)
  Real.cos (Real.arctan ((C.2 - B.2) / (C.1 - B.1))) = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cosB_l946_94636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_xyz_eq_331_25_l946_94685

/-- Triangle DEF with parallel lines forming interior triangle XYZ -/
structure TriangleWithParallelLines where
  /-- Side length DE of triangle DEF -/
  de : ℝ
  /-- Side length EF of triangle DEF -/
  ef : ℝ
  /-- Side length FD of triangle DEF -/
  fd : ℝ
  /-- Length of intersection of m_D with interior of triangle DEF -/
  m_d_int : ℝ
  /-- Length of intersection of m_E with interior of triangle DEF -/
  m_e_int : ℝ
  /-- Length of intersection of m_F with interior of triangle DEF -/
  m_f_int : ℝ
  /-- m_D is parallel to EF -/
  m_d_parallel_ef : True
  /-- m_E is parallel to FD -/
  m_e_parallel_fd : True
  /-- m_F is parallel to DE -/
  m_f_parallel_de : True

/-- The perimeter of triangle XYZ formed by intersections of m_D, m_E, and m_F -/
noncomputable def perimeterXYZ (t : TriangleWithParallelLines) : ℝ :=
  (t.m_d_int / t.ef) * (t.m_e_int + t.ef) +
  (t.m_e_int / t.fd) * (t.m_f_int + t.fd) +
  (t.m_f_int / t.de) * (t.m_d_int + t.de)

/-- Theorem stating that the perimeter of triangle XYZ is 331.25 -/
theorem perimeter_xyz_eq_331_25 (t : TriangleWithParallelLines)
    (h1 : t.de = 150)
    (h2 : t.ef = 300)
    (h3 : t.fd = 250)
    (h4 : t.m_d_int = 75)
    (h5 : t.m_e_int = 125)
    (h6 : t.m_f_int = 50) :
    perimeterXYZ t = 331.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_xyz_eq_331_25_l946_94685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forty_power_mn_l946_94682

theorem forty_power_mn (m n : ℤ) (P Q : ℕ) (hP : P = 2^(m.toNat)) (hQ : Q = 5^(n.toNat)) :
  (40 : ℕ)^(m*n).toNat = P^(3*n).toNat * Q^m.toNat :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forty_power_mn_l946_94682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_calculation_l946_94613

/-- Calculates compound interest amount -/
noncomputable def compound_interest_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates simple interest amount -/
def simple_interest_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem simple_interest_time_calculation 
  (principal_simple : ℝ) 
  (rate_simple : ℝ) 
  (principal_compound : ℝ) 
  (rate_compound : ℝ) 
  (time_compound : ℝ) :
  principal_simple = 3500 →
  rate_simple = 0.06 →
  principal_compound = 4000 →
  rate_compound = 0.1 →
  time_compound = 2 →
  simple_interest_amount principal_simple rate_simple 2 = 
    (compound_interest_amount principal_compound rate_compound time_compound - principal_compound) / 2 →
  2 = simple_interest_amount principal_simple rate_simple 2 / (principal_simple * rate_simple) :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_calculation_l946_94613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxim_birth_probability_l946_94629

/-- The year Maxim starts school -/
def school_year : ℕ := 2014

/-- Maxim's age when starting school -/
def school_age : ℕ := 6

/-- The month Maxim starts school (September) -/
def school_month : ℕ := 9

/-- The day Maxim starts school -/
def school_day : ℕ := 1

/-- Function to check if a year is a leap year -/
def is_leap_year (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Function to get the number of days in a month -/
def days_in_month (year : ℕ) (month : ℕ) : ℕ :=
  if month == 2 then
    if is_leap_year year then 29 else 28
  else if month ∈ [4, 6, 9, 11] then 30
  else 31

/-- Function to calculate the number of days between two dates -/
def days_between (y1 m1 d1 y2 m2 d2 : ℕ) : ℕ :=
  sorry  -- Actual implementation would go here

/-- The probability that Maxim was born in 2008 -/
theorem maxim_birth_probability : 
  (days_between 2008 1 1 2008 8 31 : ℚ) / days_between (school_year - school_age) (school_month) (school_day + 1) school_year school_month school_day = 244 / 365 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxim_birth_probability_l946_94629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l946_94671

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_collinearity (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) :
  ‖a + b‖ = ‖a‖ - ‖b‖ → ∃ k : ℝ, b = k • a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l946_94671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l946_94651

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a / x + 3
def g (x : ℝ) : ℝ := x^3 - x^2

-- Define the theorem
theorem min_value_of_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 1/2 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1/2 ≤ x₂ ∧ x₂ ≤ 2 → f a x₁ - g x₂ ≥ 0) → 
  a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l946_94651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jan_extra_distance_l946_94618

/-- Represents the driving scenario with Ian, Han, and Jan -/
structure DrivingScenario where
  ian_time : ℝ
  ian_speed : ℝ
  han_extra_time : ℝ := 2
  han_extra_speed : ℝ := 8
  han_extra_distance : ℝ := 115
  jan_extra_time : ℝ := 3
  jan_extra_speed : ℝ := 12

/-- Calculates the extra distance Jan drove compared to Ian -/
def extra_distance_jan (scenario : DrivingScenario) : ℝ :=
  (scenario.ian_speed + scenario.jan_extra_speed) * (scenario.ian_time + scenario.jan_extra_time) -
  scenario.ian_speed * scenario.ian_time

/-- Theorem stating that Jan drove 184.5 miles more than Ian -/
theorem jan_extra_distance (scenario : DrivingScenario) :
  extra_distance_jan scenario = 184.5 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jan_extra_distance_l946_94618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l946_94633

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 8x -/
def Parabola (p : Point → Prop) : Prop :=
  ∀ pt, p pt ↔ pt.y^2 = 8 * pt.x

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The origin point (0, 0) -/
def O : Point := ⟨0, 0⟩

theorem parabola_point_distance (C : Point → Prop) (F K A : Point) :
  Parabola C →
  C A →
  K.x = -2 →  -- Directrix equation x = -2
  distance A K = Real.sqrt 2 * distance A F →
  distance O A = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l946_94633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_l946_94634

def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def set_B : Set ℝ := {x | Real.exp ((x + 1) * Real.log 2) > 4}

theorem intersection_theorem :
  set_A ∩ (Set.univ \ set_B) = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_l946_94634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_paths_l946_94637

/-- The number of non-decreasing paths from (0,0) to (n,n) staying below or on the line y = x -/
def catalan_paths (n : ℕ) : ℚ :=
  1 / (n + 1 : ℚ) * (Nat.choose (2 * n) n : ℚ)

/-- A function representing the number of valid paths (to be defined) -/
noncomputable def number_of_valid_paths (n : ℕ) : ℚ := sorry

/-- Theorem stating that catalan_paths gives the correct number of paths -/
theorem count_paths (n : ℕ) :
  catalan_paths n = number_of_valid_paths n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_paths_l946_94637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l946_94692

/-- The trajectory of point P -/
def trajectory (x y : ℝ) : Prop := y^2 - x^2 = 1 ∧ y < 0

/-- The parabola C -/
def parabola (x y : ℝ) : Prop := x^2 = 2*y

/-- The unit circle -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Tangent line to the parabola at point (a, a^2/2) -/
def tangent_line (x y a : ℝ) : Prop := y = a*x - a^2/2

theorem trajectory_of_P (x y x_A y_A x_B y_B x_M y_M : ℝ) :
  y < 0 →  -- P is below x-axis
  parabola x_A y_A →  -- A is on the parabola
  parabola x_B y_B →  -- B is on the parabola
  tangent_line x y x_A →  -- PA is tangent to parabola
  tangent_line x y x_B →  -- PB is tangent to parabola
  unit_circle x_M y_M →  -- M is on the circle
  x_M * x_A + y_M * y_A = 1 →  -- AB is tangent to circle at M
  x_M * x_B + y_M * y_B = 1 →
  trajectory x y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l946_94692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_is_one_l946_94686

def sequence_product : ℕ → ℕ
  | 0 => 7
  | n + 1 => sequence_product n + 10

theorem product_remainder_is_one : 
  (Finset.range 20).prod (fun k => sequence_product k) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_is_one_l946_94686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheryl_journey_distance_l946_94620

-- Define the journey segments
def initial_walk (speed : ℝ) (time : ℝ) : ℝ := speed * time
def second_walk (speed : ℝ) (time : ℝ) : ℝ := speed * time
def muddy_walk (speed : ℝ) (time : ℝ) (reduction : ℝ) : ℝ := speed * (1 - reduction) * time
def bicycle_ride (speed : ℝ) (time : ℝ) : ℝ := speed * time
def slow_walk (speed : ℝ) (time : ℝ) : ℝ := speed * time
def return_walk (speed : ℝ) (time : ℝ) (reduction : ℝ) : ℝ := speed * (1 - reduction) * time

-- Define the total distance function
def total_distance (
  initial_speed : ℝ)
  (initial_time : ℝ)
  (second_speed : ℝ)
  (second_time : ℝ)
  (muddy_speed : ℝ)
  (muddy_time : ℝ)
  (muddy_reduction : ℝ)
  (bicycle_speed : ℝ)
  (bicycle_time : ℝ)
  (slow_speed : ℝ)
  (slow_time : ℝ)
  (return_speed : ℝ)
  (return_time : ℝ)
  (return_reduction : ℝ) : ℝ :=
  initial_walk initial_speed initial_time +
  second_walk second_speed second_time +
  muddy_walk muddy_speed muddy_time muddy_reduction +
  bicycle_ride bicycle_speed bicycle_time +
  slow_walk slow_speed slow_time +
  return_walk return_speed return_time return_reduction

-- State the theorem
theorem cheryl_journey_distance :
  total_distance 2 3 4 2 4 1 0.5 8 2 1.5 3 3 6 0.25 = 50 := by
  sorry

#eval total_distance 2 3 4 2 4 1 0.5 8 2 1.5 3 3 6 0.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheryl_journey_distance_l946_94620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_babysitter_overtime_increase_l946_94652

/-- A babysitter's payment structure and work details for a week --/
structure BabysitterWeek where
  regularRate : ℚ
  regularHours : ℚ
  totalEarnings : ℚ
  totalHours : ℚ

/-- Calculate the percentage increase in hourly rate for overtime hours --/
noncomputable def overtimePercentageIncrease (week : BabysitterWeek) : ℚ :=
  let regularEarnings := week.regularRate * week.regularHours
  let overtimeEarnings := week.totalEarnings - regularEarnings
  let overtimeHours := week.totalHours - week.regularHours
  let overtimeRate := overtimeEarnings / overtimeHours
  let increaseAmount := overtimeRate - week.regularRate
  (increaseAmount / week.regularRate) * 100

/-- Theorem stating that the percentage increase in hourly rate for overtime hours is 75% --/
theorem babysitter_overtime_increase (week : BabysitterWeek)
    (h1 : week.regularRate = 16)
    (h2 : week.regularHours = 30)
    (h3 : week.totalEarnings = 760)
    (h4 : week.totalHours = 40) :
    overtimePercentageIncrease week = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_babysitter_overtime_increase_l946_94652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_vertex_distance_to_plane_l946_94627

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance from a point to a plane -/
noncomputable def distanceToPlane (plane : Plane) (point : Point) : ℝ :=
  (plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d) / 
  Real.sqrt (plane.a^2 + plane.b^2 + plane.c^2)

theorem cube_vertex_distance_to_plane :
  ∀ (plane : Plane),
    plane.a^2 + plane.b^2 + plane.c^2 = 1 →
    distanceToPlane plane ⟨8, 0, 0⟩ = 8 →
    distanceToPlane plane ⟨0, 8, 0⟩ = 9 →
    distanceToPlane plane ⟨0, 0, 8⟩ = 10 →
    distanceToPlane plane ⟨0, 0, 0⟩ = (27 - Real.sqrt 51) / 3 := by
  sorry

#check cube_vertex_distance_to_plane

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_vertex_distance_to_plane_l946_94627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l946_94658

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 4*x + 12)

theorem monotonic_increasing_interval_of_f :
  {x : ℝ | ∀ y, -2 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≤ f y} = Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l946_94658
