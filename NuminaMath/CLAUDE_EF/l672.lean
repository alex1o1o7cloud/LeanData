import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_exponent_in_factorial_l672_67202

theorem prime_exponent_in_factorial (p n m : ℕ) (h_prime : Nat.Prime p) (h_bound : p ^ m ≤ n ∧ n < p ^ (m + 1)) :
  (Finset.range (m + 1)).sum (fun k => (n / p ^ k : ℕ)) = (Nat.factorial n).factorization p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_exponent_in_factorial_l672_67202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_equals_1_l672_67224

def EvenFunction (f : ℤ → ℤ) : Prop := ∀ x : ℤ, f (-x) = f x

theorem f_2016_equals_1 
  (f : ℤ → ℤ) 
  (h_even : EvenFunction f)
  (h_f_1 : f 1 = 1)
  (h_f_2015 : f 2015 ≠ 1)
  (h_f_max : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)) :
  f 2016 = 1 := by
  sorry

#check f_2016_equals_1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_equals_1_l672_67224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l672_67209

noncomputable def f (x : ℝ) := Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≥ -1/2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≤ 1/4) ∧
  (f (-Real.pi/12) = -1/2) ∧
  (f (Real.pi/4) = 1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l672_67209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_eq_one_fourth_l672_67274

noncomputable def trigExpression (α : ℝ) : ℝ :=
  (Real.tan (Real.pi/4 - α) * Real.sin α * Real.cos α) /
  ((1 - Real.tan (Real.pi/4 - α)^2) * (Real.cos α^2 - Real.sin α^2))

-- Theorem statement
theorem trigExpression_eq_one_fourth :
  ∀ α : ℝ, trigExpression α = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_eq_one_fourth_l672_67274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l672_67247

noncomputable def f (φ : Real) (x : Real) : Real := 2 * Real.cos (Real.pi * x / 3 + φ)

def is_symmetry_center (φ : Real) : Prop :=
  ∀ x : Real, f φ (2 + x) + f φ (2 - x) = 2 * (f φ 2)

theorem phi_value (φ : Real) (h1 : 0 < φ) (h2 : φ < Real.pi) (h3 : is_symmetry_center φ) :
  φ = 5 * Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l672_67247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_ellipse_at_midpoint_l672_67254

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 16 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := x + 8*y - 17 = 0

/-- Midpoint condition for two points -/
def midpoint_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 2

theorem line_intersects_ellipse_at_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ 
    ellipse x₂ y₂ ∧ 
    line x₁ y₁ ∧ 
    line x₂ y₂ ∧
    midpoint_condition x₁ y₁ x₂ y₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_ellipse_at_midpoint_l672_67254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_speed_theorem_l672_67205

/-- The average speed of a monkey swinging for a given distance and time -/
noncomputable def monkey_average_speed (distance : ℝ) (time_minutes : ℝ) : ℝ :=
  distance / (time_minutes * 60)

/-- Theorem: The average speed of a monkey swinging 2160 meters in 30 minutes is 1.2 meters per second -/
theorem monkey_speed_theorem :
  monkey_average_speed 2160 30 = 1.2 := by
  -- Unfold the definition of monkey_average_speed
  unfold monkey_average_speed
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_speed_theorem_l672_67205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_problem_l672_67297

/-- Represents the tiered pricing system for electricity -/
noncomputable def electricity_price (usage : ℝ) : ℝ :=
  if usage ≤ 200 then 0.5
  else if usage ≤ 450 then 0.7
  else 1

/-- Calculates the total electricity fee for a given usage -/
noncomputable def calculate_fee (usage : ℝ) : ℝ :=
  if usage ≤ 200 then usage * 0.5
  else if usage ≤ 450 then 200 * 0.5 + (usage - 200) * 0.7
  else 200 * 0.5 + 250 * 0.7 + (usage - 450) * 1

theorem electricity_problem :
  (calculate_fee 300 = 170) ∧
  (∃ (may june : ℝ),
    may + june = 500 ∧
    calculate_fee may + calculate_fee june = 290 ∧
    june > may ∧
    may < 450 ∧
    june < 450 ∧
    may = 100 ∧
    june = 400) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_problem_l672_67297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extinction_probability_l672_67243

-- Define the probability mass function
def p : ℕ → ℝ := sorry

-- Define the expected value of X
def E (p : ℕ → ℝ) : ℝ := p 1 + 2 * p 2 + 3 * p 3

-- Define the equation for the probability of extinction
def extinction_eq (p : ℕ → ℝ) (x : ℝ) : Prop :=
  p 0 + p 1 * x + p 2 * x^2 + p 3 * x^3 = x

-- State the theorem
theorem extinction_probability (p : ℕ → ℝ) :
  (∀ i, i > 3 → p i = 0) →
  (∀ i, p i ≥ 0) →
  (p 0 + p 1 + p 2 + p 3 = 1) →
  ∃ x, x > 0 ∧ extinction_eq p x ∧ 
  (∀ y, y > 0 ∧ extinction_eq p y → x ≤ y) →
  ((E p ≤ 1 → x = 1) ∧ (E p > 1 → x < 1)) := by
  sorry

#check extinction_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extinction_probability_l672_67243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l672_67201

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (a + 1) * x + 1
noncomputable def g (x : ℝ) : ℝ := x * Real.exp x

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x + a + 1

-- Define the derivative of g
noncomputable def g_deriv (x : ℝ) : ℝ := Real.exp x + x * Real.exp x

theorem problem_solution :
  -- Part 1
  (∀ a : ℝ, f_deriv a 1 = 3 → a = 1) ∧
  -- Part 2
  (∀ m : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ≥ -2 ∧ x₂ ≥ -2 ∧ 
    g x₁ - m + 2 = 0 ∧ g x₂ - m + 2 = 0) → 
    m > -1 / Real.exp 1 + 2 ∧ m ≤ -2 / Real.exp 2 + 2) ∧
  -- Part 3
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → g_deriv x - f a x ≥ Real.exp x) → a ≤ 0)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l672_67201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_positive_a_value_l672_67266

noncomputable def a : ℕ → ℝ
| 0 => 1
| 1 => 2  -- This is derived from the recurrence relation, not given directly
| (n + 2) => 6 * a n - a (n + 1)

theorem a_positive : ∀ n : ℕ, a n > 0 :=
  sorry

theorem a_value : a 2007 = 2^2007 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_positive_a_value_l672_67266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l672_67263

noncomputable def g (a x : ℝ) : ℝ := a * (Real.cos x)^4 - 2 * Real.cos x * Real.sin x + (Real.sin x)^4

theorem g_range (a : ℝ) (ha : a > 0) :
  Set.range (g a) = Set.Icc (a - (3 - a) / 2) (a + (a + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l672_67263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l672_67271

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ w x y z : ℝ, w > 0 → x > 0 → y > 0 → z > 0 → w * x = y * z →
    ((f w)^2 + (f x)^2) / (f (y^2) + f (z^2)) = (w^2 + x^2) / (y^2 + z^2)

/-- The main theorem stating the only two functions satisfying the equation -/
theorem functional_equation_solution (f : ℝ → ℝ) (hf : SatisfiesEquation f) :
    (∀ x : ℝ, x > 0 → f x = x) ∨ (∀ x : ℝ, x > 0 → f x = 1 / x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l672_67271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l672_67268

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (0 < A) ∧ (A < Real.pi) ∧ (0 < B) ∧ (B < Real.pi) ∧ (0 < C) ∧ (C < Real.pi) ∧
  (A + B + C = Real.pi) ∧ (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Given conditions
  (Real.tan A / Real.tan B = 2 * c / b - 1) ∧
  (Real.sin (B + C) = 6 * Real.cos B * Real.sin C) →
  -- Conclusion
  b / c = Real.sqrt 6 - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l672_67268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_root_condition_l672_67246

noncomputable def f (t x : ℝ) : ℝ := 
  -(Real.cos x)^2 - 4*t*(Real.sin (x/2))*(Real.cos (x/2)) + 2*t^2 - 6*t + 2

noncomputable def g (t : ℝ) : ℝ := 
  if t < -1 then 2*t^2 - 4*t + 2
  else if t ≤ 1 then t^2 - 6*t + 1
  else 2*t^2 - 8*t + 2

theorem min_value_and_root_condition (t : ℝ) : 
  (∀ x, f t x ≥ g t) ∧ 
  ((-1 < t ∧ t < 1) → 
    (∃! k, g t = k*t) ↔ (k < -8 ∨ k > -4)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_root_condition_l672_67246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_stocks_investment_l672_67238

/-- Given a total investment and the ratio between stocks and real estate investments,
    calculate the amount invested in stocks. -/
noncomputable def stocks_investment (total : ℝ) (ratio : ℝ) : ℝ :=
  (ratio * total) / (ratio + 1)

/-- Theorem stating that given the specific conditions of Lisa's investment,
    the amount invested in stocks is $175,000. -/
theorem lisa_stocks_investment :
  let total := (200000 : ℝ)
  let ratio := (7 : ℝ)
  stocks_investment total ratio = 175000 := by
  -- Unfold the definition and simplify
  unfold stocks_investment
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_stocks_investment_l672_67238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l672_67221

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := Real.exp x * (x^3 + 3/2 * x^2 - 6*x + 2) - 2*a*Real.exp x - x

-- State the theorem
theorem min_a_value (a : ℝ) :
  (∃ x : ℝ, x ≥ -2 ∧ f a x ≤ 0) →
  a ≥ -3/4 - 1/(2*Real.exp 1) := by
  sorry

-- You can add more auxiliary lemmas or definitions here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l672_67221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_circle_radius_is_three_tenths_c_l672_67234

/-- A right-angled triangle with hypotenuse length c and three circles of radius c/5 centered at its vertices -/
structure RightTriangleWithCircles (c : ℝ) where
  -- Assume c > 0 to ensure the triangle and circles are well-defined
  c_pos : c > 0

/-- The radius of the fourth circle that touches the three given circles externally -/
noncomputable def fourth_circle_radius (c : ℝ) : ℝ := 3 * c / 10

/-- Theorem stating that the radius of the fourth circle is 3c/10 -/
theorem fourth_circle_radius_is_three_tenths_c (c : ℝ) (t : RightTriangleWithCircles c) :
  fourth_circle_radius c = 3 * c / 10 := by
  -- Unfold the definition of fourth_circle_radius
  unfold fourth_circle_radius
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_circle_radius_is_three_tenths_c_l672_67234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_focal_length_l672_67276

-- Define the polar equation
def polar_equation (ρ : ℝ) (θ : ℝ) : Prop :=
  5 * ρ^2 * Real.cos (2 * θ) + ρ^2 - 24 = 0

-- Define the focal length
noncomputable def focal_length : ℝ := 2 * Real.sqrt 10

-- Theorem statement
theorem curve_focal_length :
  ∃ (c : ℝ), c = focal_length ∧
  ∀ (ρ θ : ℝ), polar_equation ρ θ → 
  ∃ (x y : ℝ), x^2 / 4 - y^2 / 6 = 1 ∧
  c^2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_focal_length_l672_67276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_minus_dorothy_payment_l672_67242

/-- The total amount paid by all four people -/
noncomputable def total_paid : ℝ := 150 + 180 + 240 + 130

/-- The amount each person should pay for an even split -/
noncomputable def even_split : ℝ := total_paid / 4

/-- The amount Tom paid -/
def tom_paid : ℝ := 150

/-- The amount Dorothy paid -/
def dorothy_paid : ℝ := 180

/-- The amount Sammy paid -/
def sammy_paid : ℝ := 240

/-- The amount Mark paid -/
def mark_paid : ℝ := 130

/-- The amount Tom gives to Sammy -/
noncomputable def t : ℝ := even_split - tom_paid

/-- The amount Dorothy gives to Sammy -/
noncomputable def d : ℝ := max (even_split - dorothy_paid) 0

theorem tom_minus_dorothy_payment : t - d = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_minus_dorothy_payment_l672_67242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l672_67241

/-- Given a circle defined by parametric equations x = 3sin(θ) + 4cos(θ) and y = 4sin(θ) - 3cos(θ),
    where θ is the parameter, the radius of the circle is 5. -/
theorem circle_radius (θ : ℝ) :
  (3 * Real.sin θ + 4 * Real.cos θ)^2 + (4 * Real.sin θ - 3 * Real.cos θ)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l672_67241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_l672_67286

def numbers : List ℕ := [12, 13, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers :
  let sum := numbers.sum
  let count := numbers.length
  (sum : ℚ) / count = 125830.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_l672_67286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_exp_graph_rotated_exp_is_ln_neg_l672_67267

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the rotation transformation
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)

-- Define the inverse function of exp
noncomputable def lnNeg (x : ℝ) : ℝ := Real.log (-x)

-- Theorem statement
theorem rotation_of_exp_graph (x : ℝ) :
  let original_point := (x, f x)
  let rotated_point := rotate90Clockwise original_point
  rotated_point.2 = lnNeg rotated_point.1 := by
  -- Proof steps would go here
  sorry

-- Additional lemma to show the equivalence of the rotated function and y = ln(-x)
theorem rotated_exp_is_ln_neg (x : ℝ) (h : x < 0) :
  lnNeg x = Real.log (-x) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_exp_graph_rotated_exp_is_ln_neg_l672_67267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l672_67285

-- Define the ellipse and its properties
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

-- Define points on the ellipse
def Point := ℝ × ℝ

-- Define the foci
def leftFocus (e : Ellipse) : Point := (-e.a, 0)
def rightFocus (e : Ellipse) : Point := (e.a, 0)

-- Define a line passing through a point with a given slope
def Line (p : Point) (m : ℝ) := {q : Point | q.2 - p.2 = m * (q.1 - p.1)}

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Helper function for triangle area (not proven)
noncomputable def area_triangle (p q r : Point) : ℝ := sorry

-- State the theorem
theorem ellipse_properties (e : Ellipse) (A B : Point) (h_collinear : ∃ (m : ℝ), A ∈ Line (leftFocus e) m ∧ B ∈ Line (leftFocus e) m) (h_AB : distance A B = 4/3) :
  -- Part 1: Maximum value of |AF₂| · |BF₂|
  (∀ (A' B' : Point), A' ∈ Line (leftFocus e) m → B' ∈ Line (leftFocus e) m →
    distance A' B' = 4/3 →
    distance A' (rightFocus e) * distance B' (rightFocus e) ≤ 16/9) ∧
  -- Part 2: Area of triangle ABF₂ when slope is 45°
  (∃ (m : ℝ), A ∈ Line (leftFocus e) m ∧ B ∈ Line (leftFocus e) m ∧ m = 1 →
    area_triangle A B (rightFocus e) = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l672_67285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_square_l672_67235

/-- Given a unit square ABCD and a circle with radius 32/49 that passes through D
    and is tangent to AB at E, prove that DE = 8/7 -/
theorem circle_tangent_square (A B C D E : ℝ × ℝ) (O : ℝ × ℝ) : 
  -- Square ABCD is a unit square
  A = (0, 1) → B = (1, 1) → C = (1, 0) → D = (0, 0) →
  -- O is the center of the circle
  -- Circle passes through D
  dist O D = 32/49 →
  -- Circle is tangent to AB at E
  E.2 = 1 → dist O E = 32/49 →
  -- DE = 8/7
  dist D E = 8/7 := by sorry

/-- Helper function to calculate Euclidean distance -/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_square_l672_67235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_program_result_l672_67259

def update_x (x : ℕ) : ℕ := x + 2

def update_S (S x : ℕ) : ℕ := S + x

def x_value (n : ℕ) : ℕ := 3 + 2 * n

theorem computer_program_result :
  ∃ n : ℕ, x_value n = 201 ∧ n^2 + 4*n ≥ 10000 ∧ ∀ k < n, k^2 + 4*k < 10000 := by
  -- We claim that n = 99 satisfies the conditions
  use 99
  constructor
  · -- Prove x_value 99 = 201
    rfl
  constructor
  · -- Prove 99^2 + 4*99 ≥ 10000
    norm_num
  · -- Prove ∀ k < 99, k^2 + 4*k < 10000
    intro k hk
    have h : k ≤ 98 := Nat.le_of_lt_succ hk
    calc
      k^2 + 4*k ≤ 98^2 + 4*98 := by rel [h]
      _         < 10000       := by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_program_result_l672_67259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_operations_l672_67287

theorem calculator_operations (x : ℝ) (n : ℕ) (hx : x ≠ 0) :
  let f (z : ℝ) := (z^2)⁻¹
  let y := (f^[n]) x
  y = x^((-1:ℤ) * 2^n) := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_operations_l672_67287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l672_67233

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the parabola
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define point P
noncomputable def P : ℝ × ℝ := (-1, Real.sqrt 2 / 2)

-- Theorem statement
theorem ellipse_parabola_intersection :
  -- Given conditions
  (∀ x y, ellipse_C x y ↔ (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 = 8) →
  ellipse_C P.1 P.2 →
  (∃ p > 0, ∃ M N : ℝ × ℝ, 
    ellipse_C M.1 M.2 ∧ ellipse_C N.1 N.2 ∧
    parabola p M.1 M.2 ∧ parabola p N.1 N.2) →
  -- Conclusions
  (∀ x y, ellipse_C x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∃! p : ℝ, p > 0 ∧ p = 1/4 ∧
    ∀ q > 0, ∃ M N : ℝ × ℝ,
      ellipse_C M.1 M.2 ∧ ellipse_C N.1 N.2 ∧
      parabola q M.1 M.2 ∧ parabola q N.1 N.2 →
      M.1 * M.2 ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l672_67233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l672_67280

/-- A box containing 6 balls with 2 of each color (red, yellow, blue) -/
structure Box where
  balls : Finset (Fin 6)
  red_count : Nat
  yellow_count : Nat
  blue_count : Nat
  ball_count : red_count + yellow_count + blue_count = 6
  color_count : red_count = 2 ∧ yellow_count = 2 ∧ blue_count = 2

/-- A draw is a function that selects a ball from the box with equal probability -/
def Draw (box : Box) := Fin 6 → ℝ

/-- Event A: color of the ball drawn in the first draw is the same as the color of the ball drawn in the second draw -/
def EventA (box : Box) (draw1 draw2 draw3 : Draw box) : Prop := sorry

/-- Event B: color of all three drawn balls is the same -/
def EventB (box : Box) (draw1 draw2 draw3 : Draw box) : Prop := sorry

/-- The probability of event A -/
noncomputable def ProbA (box : Box) (draw1 draw2 draw3 : Draw box) : ℝ := sorry

/-- The probability of event B -/
noncomputable def ProbB (box : Box) (draw1 draw2 draw3 : Draw box) : ℝ := sorry

/-- The probability of both events A and B occurring -/
noncomputable def ProbAB (box : Box) (draw1 draw2 draw3 : Draw box) : ℝ := sorry

/-- The main theorem: P(B|A) = 1/3 -/
theorem conditional_probability_B_given_A (box : Box) (draw1 draw2 draw3 : Draw box) :
  ProbAB box draw1 draw2 draw3 / ProbA box draw1 draw2 draw3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l672_67280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l672_67232

theorem min_omega_value (ω : ℝ) (h1 : ω > 0) :
  (∀ x : ℝ, Real.sin (ω * x + π / 3) + 2 = Real.sin (ω * (x - 4 * π / 3) + π / 3) + 2) →
  ω ≥ 3 / 2 ∧ ∀ ε > 0, ∃ ω₀ : ℝ, ω₀ < 3 / 2 + ε ∧ 
    (∀ x : ℝ, Real.sin (ω₀ * x + π / 3) + 2 = Real.sin (ω₀ * (x - 4 * π / 3) + π / 3) + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l672_67232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_correct_proposition_l672_67230

-- Define the basic geometric objects
variable (L : Type) -- Type for lines
variable (P : Type) -- Type for planes

-- Define the geometric relationships
variable (parallel_line : L → L → Prop)
variable (parallel_plane : P → P → Prop)
variable (parallel_to_line : P → L → Prop)
variable (parallel_to_plane : L → P → Prop)
variable (perpendicular_to_line : L → L → Prop)
variable (perpendicular_to_plane : L → P → Prop)

-- Define the propositions
def prop1 (p1 p2 : P) (l : L) : Prop :=
  parallel_to_line p1 l → parallel_to_line p2 l → parallel_plane p1 p2

def prop2 (l1 l2 : L) (p : P) : Prop :=
  parallel_to_plane l1 p → parallel_to_plane l2 p → parallel_line l1 l2

def prop3 (l1 l2 l : L) : Prop :=
  perpendicular_to_line l1 l → perpendicular_to_line l2 l → parallel_line l1 l2

def prop4 (l1 l2 : L) (p : P) : Prop :=
  perpendicular_to_plane l1 p → perpendicular_to_plane l2 p → parallel_line l1 l2

-- Theorem stating that only one proposition is correct
theorem only_one_correct_proposition :
  (∃! i : Fin 4, match i with
    | 0 => ∀ p1 p2 l, prop1 L P parallel_plane parallel_to_line p1 p2 l
    | 1 => ∀ l1 l2 p, prop2 L P parallel_line parallel_to_plane l1 l2 p
    | 2 => ∀ l1 l2 l, prop3 L parallel_line perpendicular_to_line l1 l2 l
    | 3 => ∀ l1 l2 p, prop4 L P parallel_line perpendicular_to_plane l1 l2 p
  ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_correct_proposition_l672_67230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_A_to_circle_M_l672_67277

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 7/9

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the center of the circle M
def center_M : ℝ × ℝ := (-1, 0)

-- Define the line passing through center M and points A and B
noncomputable def line_MAB (t α : ℝ) : ℝ × ℝ := (-1 + t * Real.cos α, t * Real.sin α)

-- Define the relationship between vectors MA and MB
def vector_relation (tA tB : ℝ) : Prop := tB = 3 * tA

-- Define point A
noncomputable def point_A (tA α : ℝ) : ℝ × ℝ := line_MAB tA α

theorem min_distance_A_to_circle_M :
  ∃ (tA α : ℝ),
    let A := point_A tA α
    ∃ (tB : ℝ),
      parabola_C A.1 A.2 ∧
      parabola_C (line_MAB tB α).1 (line_MAB tB α).2 ∧
      vector_relation tA tB ∧
      (∀ (x y : ℝ), circle_M x y → Real.sqrt ((x - A.1)^2 + (y - A.2)^2) ≥ Real.sqrt 7 / 3) ∧
      (∃ (x y : ℝ), circle_M x y ∧ Real.sqrt ((x - A.1)^2 + (y - A.2)^2) = Real.sqrt 7 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_A_to_circle_M_l672_67277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_eq_one_sufficient_not_necessary_l672_67227

noncomputable section

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line mx - y = 0 -/
def slope1 (m : ℝ) : ℝ := m

/-- The slope of the line x + m^2y = 0 -/
def slope2 (m : ℝ) : ℝ := -1 / (m^2)

/-- The condition "m = 1" is sufficient but not necessary for perpendicularity -/
theorem m_eq_one_sufficient_not_necessary :
  (∀ m : ℝ, m = 1 → are_perpendicular (slope1 m) (slope2 m)) ∧
  (∃ m : ℝ, m ≠ 1 ∧ are_perpendicular (slope1 m) (slope2 m)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_eq_one_sufficient_not_necessary_l672_67227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_l672_67251

noncomputable def sequenceA (n : ℕ) : ℝ := 18 + (3 / 2) * n * (n - 1)

theorem min_value_of_sequence :
  let a := sequenceA
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 3 * n) ∧
  (∃ n : ℕ, n ≥ 1 ∧ a n / n = 9) ∧
  (∀ n : ℕ, n ≥ 1 → a n / n ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_l672_67251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l672_67298

/-- Represents the lengths of the available sticks -/
def stickLengths : List ℝ := [2, 3, 4, 5, 6]

/-- Checks if three lengths can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Calculates the area of a triangle using Heron's formula -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The maximum area of a triangle formed from the given sticks is 8 cm² -/
theorem max_triangle_area : 
  ∀ a b c : ℝ, a ∈ stickLengths → b ∈ stickLengths → c ∈ stickLengths →
  canFormTriangle a b c →
  triangleArea a b c ≤ 8 := by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l672_67298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transposition_count_invariant_l672_67207

/-- Represents a permutation of n elements -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Represents the identity permutation -/
def idPerm (n : ℕ) : Permutation n := fun i => i

/-- Applies a transposition to a permutation -/
def applyTransposition (n : ℕ) (p : Permutation n) (pair : Fin n × Fin n) : Permutation n :=
  let (i, j) := pair
  fun k => if k = i then p j else if k = j then p i else p k

/-- Counts the number of transpositions needed to sort a permutation -/
def transpositionCount (n : ℕ) (p : Permutation n) : ℕ := sorry

/-- The main theorem: the number of transpositions is invariant -/
theorem transposition_count_invariant (n : ℕ) (p : Permutation n) :
  ∀ (seq₁ seq₂ : List (Fin n × Fin n)),
    (seq₁.foldl (applyTransposition n) p = idPerm n) →
    (seq₂.foldl (applyTransposition n) p = idPerm n) →
    seq₁.length = seq₂.length :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transposition_count_invariant_l672_67207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equivalent_to_shifted_cos_l672_67270

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x) * Real.cos (ω * x) - Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sqrt 3 / 2

theorem f_equivalent_to_shifted_cos 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_zero_diff : ∀ x y, f ω x = 0 → f ω y = 0 → x ≠ y → |x - y| = π / 4) :
  ∀ x, f ω x = Real.cos (4 * x - 5 * π / 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equivalent_to_shifted_cos_l672_67270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_eleven_l672_67203

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x^2

-- State the theorem
theorem f_of_three_equals_eleven :
  f 3 = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_eleven_l672_67203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l672_67262

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi + x) * (Real.cos x - 2 * Real.sin x) + Real.sin x ^ 2

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 8)

theorem g_properties :
  (∀ x, g x = Real.sqrt 2 * Real.sin (2 * x)) ∧
  (∀ x ∈ Set.Ioo 0 (Real.pi / 4), StrictMono g) ∧
  (∀ x, g (-x) = -g x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l672_67262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l672_67244

def curve1 (x y : ℝ) : Prop := (x + y - 7) * (2 * x - 3 * y + 1) = 0

def curve2 (x y : ℝ) : Prop := (x - y - 2) * (3 * x + 2 * y - 10) = 0

def intersection_point (x y : ℝ) : Prop := curve1 x y ∧ curve2 x y

theorem intersection_count : 
  ∃ (S : Finset (ℝ × ℝ)), (∀ (p : ℝ × ℝ), p ∈ S ↔ intersection_point p.1 p.2) ∧ Finset.card S = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l672_67244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_fourth_power_unity_l672_67295

theorem solutions_of_fourth_power_unity : 
  {x : ℂ | x^4 = 1} = {1, -1, Complex.I, -Complex.I} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_fourth_power_unity_l672_67295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_numbers_with_specific_digits_l672_67248

theorem composite_numbers_with_specific_digits : 
  ∃ (S : Finset (ℕ × ℕ)), 
    (S.card ≥ 668) ∧ 
    (∀ (pair : ℕ × ℕ), pair ∈ S → 
      let (k, p) := pair
      k ≤ 2005 ∧ 
      (p = 7 ∨ p = 13) ∧ 
      ((10^2006 - 1) / 9 + 6 * 10^k) % p = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_numbers_with_specific_digits_l672_67248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_condition_l672_67215

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The radius of the circumscribed circle of a triangle. -/
noncomputable def circumradius (t : Triangle) : ℝ := sorry

/-- The semi-perimeter of a triangle. -/
noncomputable def semiperimeter (t : Triangle) : ℝ := sorry

/-- The tangent of an angle in a triangle. -/
noncomputable def tg (p : ℝ × ℝ) (t : Triangle) : ℝ := sorry

/-- A triangle is equilateral if all its sides have the same length. -/
def is_equilateral (t : Triangle) : Prop := sorry

theorem equilateral_triangle_condition (t : Triangle) :
  circumradius t * (tg t.A t + tg t.B t + tg t.C t) = 2 * semiperimeter t →
  is_equilateral t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_condition_l672_67215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l672_67239

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner
  distance_difference : ℝ

/-- Calculates the time difference between two runners -/
noncomputable def time_difference (race : Race) : ℝ :=
  race.distance_difference / race.runner_a.speed

/-- Theorem stating the time difference in the given race scenario -/
theorem race_time_difference (race : Race) 
  (h1 : race.distance = 1000)
  (h2 : race.runner_a.time = 92)
  (h3 : race.distance_difference = 80)
  (h4 : race.runner_a.speed = race.distance / race.runner_a.time) :
  ∃ ε > 0, |time_difference race - 7.36| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l672_67239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_15_gon_sum_theorem_l672_67220

/-- Regular 15-gon inscribed in a circle of radius 15 -/
def regular_15_gon (r : ℝ) : Prop := r = 15

/-- Sum of lengths of all sides and diagonals -/
noncomputable def sum_lengths (r : ℝ) : ℝ := 
  450 * (Real.sin (Real.pi / 15) + Real.sin (Real.pi / 5))

/-- Expression of sum in the form a + b√2 + c√3 + d√5 -/
def sum_expression (a b c d : ℕ) (x : ℝ) : Prop :=
  x = a + b * Real.sqrt 2 + c * Real.sqrt 3 + d * Real.sqrt 5

theorem regular_15_gon_sum_theorem (r : ℝ) (a b c d : ℕ) :
  regular_15_gon r →
  sum_expression a b c d (sum_lengths r) →
  a + b + c + d = 1800 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_15_gon_sum_theorem_l672_67220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l672_67290

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / (1 + x)

theorem tangent_line_at_zero :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (y = m * x + b) ↔ (y - f 0 = (deriv f 0) * (x - 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l672_67290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l672_67249

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define symmetry about a point
def symmetric_about (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, f x = y ↔ f (2*a - x) = 2*b - y

-- State the theorem
theorem function_symmetry (f : ℝ → ℝ) 
  (h1 : symmetric_about f (-1) 0)
  (h2 : ∀ x > 0, f x = 1 / x) :
  ∀ x < -2, f x = 1 / (2 + x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l672_67249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trajectory_and_intersection_l672_67212

open Real

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the sides a, b, c
noncomputable def side (t : Triangle) (v : Fin 3) : ℝ :=
  match v with
  | 0 => norm (t.B.1 - t.C.1, t.B.2 - t.C.2)
  | 1 => norm (t.A.1 - t.C.1, t.A.2 - t.C.2)
  | 2 => norm (t.A.1 - t.B.1, t.A.2 - t.B.2)

-- Define the arithmetic sequence property
def isArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

-- Main theorem
theorem triangle_trajectory_and_intersection 
  (t : Triangle) 
  (hB : t.B = (-1, 0)) 
  (hC : t.C = (1, 0)) 
  (hSeq : isArithmeticSequence (side t 1) (side t 0) (side t 2)) :
  (∃ (x y : ℝ), x^2/4 + y^2/3 = 1 ∧ t.A = (x, y)) ∧
  (∀ (k m : ℝ), k ≠ 0 →
    (∃ (M N : ℝ × ℝ), M ≠ N ∧
      (M.1^2/4 + M.2^2/3 = 1) ∧
      (N.1^2/4 + N.2^2/3 = 1) ∧
      (M.2 = k * M.1 + m) ∧
      (N.2 = k * N.1 + m) ∧
      (∃ (l : ℝ → ℝ), l 0 = -1/2 ∧
        (M.1 + N.1, M.2 + N.2) = (0, -1))) →
    3/2 < m ∧ m < 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trajectory_and_intersection_l672_67212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fixed_point_range_fixed_point_interval_l672_67240

-- Definition of a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Part 1
theorem no_fixed_point_range (a : ℝ) :
  (∀ x : ℝ, ¬is_fixed_point (λ x ↦ x^2 + a*x + a) x) ↔ 
  (a > 3 - 2*Real.sqrt 2 ∧ a < 3 + 2*Real.sqrt 2) :=
sorry

-- Part 2
theorem fixed_point_interval (n : ℤ) :
  (∃ x : ℝ, is_fixed_point (λ x ↦ -Real.log x + 3) x ∧ 
   x ≥ ↑n ∧ x < ↑n + 1) → n = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fixed_point_range_fixed_point_interval_l672_67240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_C_l672_67260

/-- Scaling transformation φ -/
noncomputable def φ (x y : ℝ) : ℝ × ℝ := ((1/3) * x, (1/2) * y)

/-- The unit circle -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The conic section curve C -/
def curve_C (x y : ℝ) : Prop := 
  let (x', y') := φ x y
  unit_circle x' y'

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

theorem eccentricity_of_C : 
  ∃ (a b : ℝ), (∀ (x y : ℝ), curve_C x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧ 
  eccentricity a b = Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_C_l672_67260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_bound_l672_67210

/-- Given functions f and g, prove the upper bound of a -/
theorem function_inequality_implies_a_bound 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (a : ℝ) 
  (hf : ∀ x, f x = 2 * x + a) 
  (hg : ∀ x, g x = Real.log x - 2 * x) 
  (h : ∀ x₁ x₂, x₁ ∈ Set.Icc (1/2) 2 → x₂ ∈ Set.Icc (1/2) 2 → f x₁ ≤ g x₂) : 
  a ≤ Real.log 2 - 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_bound_l672_67210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_line_l672_67219

-- Define the circle in polar coordinates
noncomputable def polar_circle (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the line in polar form
noncomputable def polar_line (θ : ℝ) : ℝ := Real.tan (θ + Real.pi / 2)

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 0)

-- Define the line in Cartesian form
def line_cartesian (x y : ℝ) : Prop := x = y

-- Statement to prove
theorem distance_from_center_to_line :
  let d := Real.sqrt 2
  ∃ (p : ℝ × ℝ), p.1 = p.2 ∧ 
    Real.sqrt ((p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2) = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_line_l672_67219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_nth_roots_of_32_l672_67282

theorem integer_nth_roots_of_32 :
  ∃! (S : Finset ℕ), S.Nonempty ∧ 
    (∀ n : ℕ, n ∈ S ↔ (n > 0 ∧ ∃ m : ℕ, m > 0 ∧ m^n = 32)) ∧
    S.card = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_nth_roots_of_32_l672_67282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_length_l672_67261

-- Define the line segment AB and points P and Q
variable (AB P Q : ℝ)

-- P divides AB in the ratio 3:5
axiom P_ratio : P / (AB - P) = 3 / 5

-- Q divides AB in the ratio 4:5
axiom Q_ratio : Q / (AB - Q) = 4 / 5

-- P and Q are on the same side of AB's midpoint
axiom same_side : P < Q ∧ Q < AB

-- Distance PQ is 3 units
axiom PQ_distance : Q - P = 3

-- Theorem to prove
theorem AB_length : AB = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_length_l672_67261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_common_tangents_theorem_l672_67273

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 3)^2 + (y - 1)^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 4

-- Define the centers of the circles
def center1 : ℝ × ℝ := (-3, 1)
def center2 : ℝ × ℝ := (4, 5)

-- Define the radius of the circles
def radius : ℝ := 2

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 65

-- Define the number of common tangents
def number_of_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := 4

-- Theorem stating the number of common tangents
theorem number_of_common_tangents_theorem :
  distance_between_centers > 2 * radius →
  (∃ n : ℕ, n = 4 ∧ n = number_of_common_tangents circle1 circle2) :=
by
  intro h
  use 4
  constructor
  · rfl
  · rfl

-- The proof is omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_common_tangents_theorem_l672_67273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_integers_sum_l672_67255

/-- Four positive integers greater than 1 with product 288000 and pairwise relatively prime -/
def SpecialIntegers : Type := 
  { nums : Fin 4 → ℕ+ // 
    (∀ i, (nums i : ℕ) > 1) ∧ 
    ((nums 0 : ℕ) * (nums 1 : ℕ) * (nums 2 : ℕ) * (nums 3 : ℕ) = 288000) ∧
    (∀ i j, i ≠ j → Nat.Coprime (nums i) (nums j)) }

/-- The sum of the special integers is 390 -/
theorem special_integers_sum (nums : SpecialIntegers) : 
  (nums.val 0 : ℕ) + (nums.val 1 : ℕ) + (nums.val 2 : ℕ) + (nums.val 3 : ℕ) = 390 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_integers_sum_l672_67255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_l672_67213

/-- Represents the set of numbers from 0 to 10 -/
def NumberSet : Set ℕ := {n | n ≤ 10}

/-- A function that takes a natural number and returns the sum of integers from 1 to that number -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the minimum number of students in the class -/
theorem min_students (student_numbers : Set ℕ) : 
  (∀ n ∈ NumberSet, n ∈ student_numbers) → 
  (∀ n ∈ student_numbers, n ∈ NumberSet) →
  33 = (sum_to_n 11) / 2 := by
  sorry

#check min_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_l672_67213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_properties_l672_67279

theorem triangle_angle_properties (A B C : Real) (a : Real) :
  -- A, B, C are interior angles of an acute triangle
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π →
  -- √3 sin A and (-cos A) are roots of x^2 - x + 2a = 0
  (Real.sqrt 3 * Real.sin A) ^ 2 - (Real.sqrt 3 * Real.sin A) + 2 * a = 0 ∧
  (-Real.cos A) ^ 2 - (-Real.cos A) + 2 * a = 0 →
  -- (1+2sin B cos B)/(cos^2 B - sin^2 B) = -3
  (1 + 2 * Real.sin B * Real.cos B) / (Real.cos B ^ 2 - Real.sin B ^ 2) = -3 →
  -- Conclusions
  A = π/3 ∧ Real.tan B = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_properties_l672_67279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kappa_grid_labeling_l672_67299

theorem kappa_grid_labeling (d : ℝ) (h_d : d > 0) :
  (∀ (n : ℕ+), ∃ (labeled : Fin n → Fin n → Bool),
    (∀ (i j : Fin n), ∀ (di dj : Fin 3),
      ¬(labeled i j ∧ labeled (i + di) (j + dj) ∧ labeled (i + 2*di) (j + 2*dj))) ∧
    ((Finset.filter (λ ij : Fin n × Fin n => labeled ij.1 ij.2) (Finset.univ)).card ≥ ⌊d * n^2⌋)) →
  d ≤ (1/2 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kappa_grid_labeling_l672_67299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_closest_l672_67257

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define membership for a point on a line
def onLine (p : Point) (l : Line) : Prop :=
  ∃ t : ℝ, p = (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2)

-- Define a tangent line to a circle
def isTangent (l : Line) (c : Circle) : Prop :=
  ∃ p : Point, onLine p l ∧ distance p c.center = c.radius ∧
    ∀ q : Point, onLine q l ∧ q ≠ p → distance q c.center > c.radius

-- Theorem statement
theorem tangent_point_closest (c : Circle) (l : Line) (p : Point) :
  isTangent l c →
  onLine p l →
  distance p c.center = c.radius →
  ∀ q : Point, onLine q l → q ≠ p → distance q c.center > distance p c.center := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_closest_l672_67257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l672_67245

theorem no_real_solutions : ¬∃ (x : ℝ), (2 : ℝ)^(2*x+3) - (2 : ℝ)^(x+4) - (2 : ℝ)^(x+1) + 16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l672_67245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_theorem_l672_67214

/-- The number of ways to arrange 5 distinct cards into 5 distinct boxes. -/
def total_arrangements : ℕ := Nat.factorial 5

/-- The number of arrangements where card 2 is in box 2 and card 4 is in box 4. -/
def invalid_case1 : ℕ := Nat.factorial 3

/-- The number of arrangements where card 2 is in box 2 but card 4 is not in box 4. -/
def invalid_case2 : ℕ := 3 * Nat.factorial 3

/-- The number of arrangements where card 2 is not in box 2 but card 4 is in box 4. -/
def invalid_case3 : ℕ := 3 * Nat.factorial 3

/-- The total number of invalid arrangements. -/
def total_invalid : ℕ := invalid_case1 + invalid_case2 + invalid_case3

/-- The number of valid arrangements satisfying the conditions. -/
def valid_arrangements : ℕ := total_arrangements - total_invalid

theorem card_arrangement_theorem : valid_arrangements = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_theorem_l672_67214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diagonal_rectangle_l672_67216

theorem min_diagonal_rectangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2*a + 2*b = 20) :
  ∃ (d : ℝ), d^2 = a^2 + b^2 ∧ d ≥ Real.sqrt 50 ∧ 
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 2*a' + 2*b' = 20 → 
    (∃ (d' : ℝ), d'^2 = a'^2 + b'^2 → d' ≥ d)) := by
  sorry

#check min_diagonal_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diagonal_rectangle_l672_67216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_parity_invariant_chip_rearrangement_impossible_l672_67293

-- Define the color of a chip
inductive Color
| Red
| Blue

-- Define a chip arrangement as a list of colors
def Arrangement := List Color

-- Define an inversion in the arrangement
def isInversion (c1 c2 : Color) : Bool :=
  match c1, c2 with
  | Color.Red, Color.Blue => true
  | _, _ => false

-- Count the number of inversions in an arrangement
def inversionCount (arr : Arrangement) : Nat :=
  arr.zip arr.tail |>.filter (fun (c1, c2) => isInversion c1 c2) |>.length

-- Define the allowed operations
inductive Operation
| Insert (c : Color) (pos : Nat)  -- Insert two chips of color c at position pos
| Remove (pos : Nat)              -- Remove two chips at position pos

-- Apply an operation to an arrangement
def applyOperation (arr : Arrangement) (op : Operation) : Arrangement :=
  match op with
  | Operation.Insert c pos => (arr.take pos) ++ [c, c] ++ (arr.drop pos)
  | Operation.Remove pos => (arr.take pos) ++ (arr.drop (pos + 2))

-- The main theorem to prove
theorem inversion_parity_invariant (initialArr finalArr : Arrangement) 
  (ops : List Operation) : 
  (inversionCount initialArr % 2 = inversionCount finalArr % 2) :=
sorry

-- The specific instance of the problem
theorem chip_rearrangement_impossible : 
  ¬∃ (ops : List Operation), 
    (ops.foldl applyOperation [Color.Red, Color.Blue]) = [Color.Blue, Color.Red] :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_parity_invariant_chip_rearrangement_impossible_l672_67293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_k_value_l672_67229

/-- Three lines in a 2D plane -/
structure ThreeLines where
  line1 : ℝ → ℝ → Prop
  line2 : ℝ → ℝ → Prop
  line3 : ℝ → ℝ → ℝ → Prop

/-- The property of three lines intersecting at a single point -/
def intersect_at_single_point (lines : ThreeLines) (k : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, lines.line1 p.1 p.2 ∧ lines.line2 p.1 p.2 ∧ lines.line3 p.1 p.2 k

/-- The main theorem -/
theorem intersection_implies_k_value (lines : ThreeLines) 
    (h1 : ∀ x y, lines.line1 x y ↔ 2*x + 3*y + 8 = 0)
    (h2 : ∀ x y, lines.line2 x y ↔ x - y - 1 = 0)
    (h3 : ∀ x y k, lines.line3 x y k ↔ x + k*y = 0)
    (h_intersect : intersect_at_single_point lines (-1/2)) :
    -1/2 = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_k_value_l672_67229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_one_thirtysecond_l672_67226

noncomputable def g (x : ℝ) : ℝ := (x^5 + 2) / 4

theorem inverse_g_at_one_thirtysecond :
  g⁻¹ (1/32) = ((-15/8) : ℝ)^(1/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_one_thirtysecond_l672_67226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_sum_l672_67200

def endpoint1 : ℝ × ℝ := (6, -2)
def midpt : ℝ × ℝ := (3, 5)

theorem endpoint_sum : 
  ∃ (x y : ℝ), 
    ((x + 6) / 2 = 3 ∧ (y + -2) / 2 = 5) ∧ 
    x + y = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_sum_l672_67200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_work_time_approx_9_l672_67252

/-- The time it takes for David to complete the work alone -/
def david_time : ℝ := 5

/-- The time it takes for David and John together to complete the work -/
def combined_time : ℝ := 3.2142857142857144

/-- The time it takes for John to complete the work alone -/
noncomputable def john_time : ℝ := (david_time * combined_time) / (david_time - combined_time)

/-- Theorem stating that John's time to complete the work is approximately 9 days -/
theorem john_work_time_approx_9 :
  |john_time - 9| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_work_time_approx_9_l672_67252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_external_tangent_y_intercept_l672_67222

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the slope of the line connecting two points -/
noncomputable def slopeBetweenPoints (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

/-- Calculates the slope of the tangent line using the tangent of the double angle formula -/
noncomputable def tangentLineSlope (slope : ℝ) : ℝ :=
  (2 * slope) / (1 - slope^2)

/-- Theorem: The y-intercept of the common external tangent with positive slope for two given circles -/
theorem common_external_tangent_y_intercept 
  (c1 c2 : Circle) 
  (h1 : c1.center = (3, 5))
  (h2 : c2.center = (16, 12))
  (h3 : c1.radius = 5)
  (h4 : c2.radius = 10) :
  let m := tangentLineSlope (slopeBetweenPoints c1.center c2.center)
  let b := 10 - m * 3
  b = 123 / 48 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_external_tangent_y_intercept_l672_67222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_power_function_l672_67289

-- Define the function f(x) = x^a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^a

-- State the theorem
theorem tangent_line_power_function (a : ℝ) :
  (∀ x, deriv (f a) x = a * x^(a - 1)) →
  deriv (f a) 1 = -4 →
  a = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_power_function_l672_67289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducible_l672_67225

/-- Given a list of distinct integers, constructs the polynomial (x-a₁)(x-a₂)...(x-aₙ)-1 -/
noncomputable def constructPolynomial (a : List Int) : Polynomial Int :=
  (a.foldr (fun aᵢ p => p * (Polynomial.X - Polynomial.C aᵢ)) 1) - 1

/-- States that the polynomial constructed from distinct integers is irreducible -/
theorem polynomial_irreducible (a : List Int) (h : a.Pairwise (· ≠ ·)) :
  Irreducible (constructPolynomial a) := by
  sorry

#check polynomial_irreducible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducible_l672_67225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l672_67288

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f)
  (h_deriv : ∀ x, deriv f x > f x) (x₁ x₂ : ℝ) (h_x : x₁ < x₂) :
  exp x₁ * f x₂ > exp x₂ * f x₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l672_67288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_astrophysics_degrees_l672_67275

/-- Represents the allocation of a budget in percentages -/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ

/-- Calculates the degrees in a circle graph for a given percentage -/
noncomputable def percentageToDegrees (percentage : ℝ) : ℝ :=
  (percentage / 100) * 360

/-- Theorem stating that the remaining sector (basic astrophysics) in the circle graph
    will be represented by 72 degrees, given the other allocations -/
theorem basic_astrophysics_degrees (budget : BudgetAllocation) 
    (h1 : budget.microphotonics = 14)
    (h2 : budget.home_electronics = 24)
    (h3 : budget.food_additives = 15)
    (h4 : budget.genetically_modified_microorganisms = 19)
    (h5 : budget.industrial_lubricants = 8) :
    percentageToDegrees (100 - (budget.microphotonics + budget.home_electronics + 
    budget.food_additives + budget.genetically_modified_microorganisms + 
    budget.industrial_lubricants)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_astrophysics_degrees_l672_67275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_chase_fox_distance_l672_67218

/-- The distance traveled by a hunting dog chasing a fox -/
theorem dog_chase_fox_distance (initial_distance : ℝ) (dog_speed : ℝ) (fox_speed : ℝ) 
  (h1 : initial_distance = 10)
  (h2 : dog_speed = 10 * fox_speed)
  (h3 : fox_speed > 0) : 
  (initial_distance * dog_speed) / (dog_speed - fox_speed) = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_chase_fox_distance_l672_67218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_iff_n_div_3_l672_67278

/-- Represents the game state -/
structure GameState where
  n : ℕ
  remaining : Finset ℕ
  first_sum : ℕ
  second_sum : ℕ

/-- Checks if a player has won -/
def has_won (sum : ℕ) : Prop := sum % 3 = 0

/-- Defines a valid move in the game -/
def valid_move (s : GameState) (m : ℕ) : Prop :=
  m ∈ s.remaining ∧ m ≤ s.n

/-- Theorem: The first player has a winning strategy if and only if n is divisible by 3 -/
theorem first_player_wins_iff_n_div_3 (n : ℕ) (h : 0 < n) :
  (∃ (strategy : GameState → ℕ), ∀ (s : GameState),
    valid_move s (strategy s) →
    has_won (s.first_sum + strategy s) ∨
    ∀ (m : ℕ), valid_move s m →
      ¬has_won (s.second_sum + m) →
      ∃ (next_strategy : GameState → ℕ), 
        valid_move {n := s.n - 1, remaining := s.remaining.erase m, first_sum := s.first_sum, second_sum := s.second_sum + m} (next_strategy {n := s.n - 1, remaining := s.remaining.erase m, first_sum := s.first_sum, second_sum := s.second_sum + m})) ↔
  n % 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_iff_n_div_3_l672_67278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l672_67292

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  h_geom : ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sum_n (g : GeometricSequence) (n : ℕ) : ℝ :=
  if g.q = 1 then n * g.a 0
  else g.a 0 * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_first_term 
  (g : GeometricSequence)
  (h1 : sum_n g 3 = g.a 1 + 10 * g.a 0)
  (h2 : g.a 4 = 9) :
  g.a 0 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l672_67292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_25_l672_67291

theorem multiples_of_25 (n : ℕ) (h1 : n > 0) (h2 : n ≤ 400) : 
  (∃ (seq : Fin 16 → ℕ), 
    (∀ i, seq i ≤ 400) ∧ 
    (∀ i, seq i % 25 = 0) ∧
    (∀ i : Fin 15, seq i.succ = seq i + 25) ∧
    (seq 0 = n)) →
  n = 25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_25_l672_67291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_of_length_15_l672_67223

mutual
  /-- Represents the number of valid sequences of length n ending with 'A' -/
  def x : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => y (n + 1)

  /-- Represents the number of valid sequences of length n ending with 'B' -/
  def y : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | (n + 2) => x n + y n
end

/-- The total number of valid sequences of length n -/
def total_sequences (n : ℕ) : ℕ := x n + y n

theorem valid_sequences_of_length_15 : total_sequences 15 = 377 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_of_length_15_l672_67223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l672_67264

theorem beta_value (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.cos α = 1/7 →
  Real.sin (α + β) = 5*Real.sqrt 3/14 →
  β = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l672_67264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_distances_l672_67236

noncomputable section

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (Real.sqrt 3 / 2) * t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := Real.sqrt (12 / (3 + Real.sin θ ^ 2))
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the point F
def F : ℝ × ℝ := (1, 0)

-- Define the intersection points A and B (existence assumed)
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- Axioms stating that A and B are on both curves
axiom A_on_C₁ : ∃ t : ℝ, C₁ t = A
axiom A_on_C₂ : ∃ θ : ℝ, C₂ θ = A
axiom B_on_C₁ : ∃ t : ℝ, C₁ t = B
axiom B_on_C₂ : ∃ θ : ℝ, C₂ θ = B

-- Function to calculate distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem to prove
theorem sum_of_reciprocal_distances :
  (1 / distance F A) + (1 / distance F B) = 4/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_distances_l672_67236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_pens_to_pencils_l672_67204

/- Define the variables -/
def num_boxes : ℕ := 15
def pencils_per_box : ℕ := 80
def pen_cost : ℚ := 5
def pencil_cost : ℚ := 4
def total_cost : ℚ := 18300

/- Define the number of pencils -/
def num_pencils : ℕ := num_boxes * pencils_per_box

/- Define the number of pens -/
noncomputable def num_pens : ℚ := (total_cost - (↑num_pencils * pencil_cost)) / pen_cost

/- Theorem statement -/
theorem ratio_of_pens_to_pencils :
  num_pens / (↑num_pencils : ℚ) = 2.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_pens_to_pencils_l672_67204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_area_ratio_is_sqrt_3_l672_67228

/-- The ratio of the surface area of a cube with side length 2 to the surface area of a regular tetrahedron formed by four of its vertices -/
noncomputable def cube_tetrahedron_area_ratio : ℝ :=
  let cube_side_length : ℝ := 2
  let cube_surface_area : ℝ := 6 * cube_side_length^2
  let tetrahedron_side_length : ℝ := cube_side_length * Real.sqrt 2
  let tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length^2
  cube_surface_area / tetrahedron_surface_area

/-- Theorem stating that the ratio of the surface areas is √3 -/
theorem cube_tetrahedron_area_ratio_is_sqrt_3 :
  cube_tetrahedron_area_ratio = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_area_ratio_is_sqrt_3_l672_67228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_is_three_l672_67296

/-- The function f(x) = x + x ln x -/
noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

/-- The inequality that must hold for all x > 1 -/
def inequality (m : ℤ) (x : ℝ) : Prop := f x - m * (x - 1) > 0

/-- The theorem stating that 3 is the maximum integer m satisfying the inequality -/
theorem max_m_is_three :
  ∃ (m : ℤ), m = 3 ∧ 
  (∀ (x : ℝ), x > 1 → inequality m x) ∧
  (∀ (n : ℤ), n > m → ∃ (y : ℝ), y > 1 ∧ ¬inequality n y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_is_three_l672_67296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_is_fifteen_l672_67256

/-- Represents the race scenario between Cynthia and David -/
structure RaceScenario where
  race_duration : ℚ
  david_distance : ℚ
  speed_difference : ℚ

/-- Calculates the difference in distance traveled between Cynthia and David -/
def distance_difference (scenario : RaceScenario) : ℚ :=
  let david_speed := scenario.david_distance / scenario.race_duration
  let cynthia_speed := david_speed + scenario.speed_difference
  (cynthia_speed - david_speed) * scenario.race_duration

/-- Theorem stating the difference in distance traveled is 15 miles -/
theorem distance_difference_is_fifteen (scenario : RaceScenario) 
  (h1 : scenario.race_duration = 5)
  (h2 : scenario.david_distance = 55)
  (h3 : scenario.speed_difference = 3) : 
  distance_difference scenario = 15 := by
  sorry

#eval distance_difference { race_duration := 5, david_distance := 55, speed_difference := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_is_fifteen_l672_67256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_80_l672_67208

def scores : List ℕ := [50, 55, 60, 65, 70, 80]

def is_integer_average (sum : ℕ) (count : ℕ) : Prop :=
  ∃ k : ℕ, sum = k * count

theorem last_score_is_80 :
  ∀ perm : List ℕ,
  perm.length = 6 →
  perm.toFinset = scores.toFinset →
  (∀ i : ℕ, i ∈ Finset.range 6 → is_integer_average (perm.take (i + 1)).sum (i + 1)) →
  perm.getLast? = some 80 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_80_l672_67208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_difference_l672_67294

-- Define the curves C₁ and C₂
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (1 + Real.cos φ, Real.sin φ)

def C₂ (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the polar coordinates
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the intersection points A and B
noncomputable def A (α : ℝ) : ℝ × ℝ := polar_to_cartesian (2 * Real.cos α) α

noncomputable def B (α : ℝ) : ℝ × ℝ := 
  polar_to_cartesian (Real.sqrt (8 / (1 + Real.sin α ^ 2))) α

-- Define the distance function
def distance_squared (p : ℝ × ℝ) : ℝ := p.1^2 + p.2^2

-- State the theorem
theorem min_distance_difference (α : ℝ) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) :
  ∃ (min : ℝ), 
    (∀ β, 0 < β ∧ β < Real.pi / 2 → 
      min ≤ distance_squared (B β) - distance_squared (A β)) ∧
    min = 8 * Real.sqrt 2 - 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_difference_l672_67294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiger_roaming_area_l672_67231

/-- The surface area a tiger can roam on a cube-shaped world -/
noncomputable def roamingArea (cubeEdgeLength : ℝ) (leashLength : ℝ) : ℝ :=
  (8 * Real.pi / 3) + 4 * Real.sqrt 3 - 4

/-- Theorem: The roaming area for a tiger on a cube with edge length 2 and leash length 2 -/
theorem tiger_roaming_area :
  roamingArea 2 2 = (8 * Real.pi / 3) + 4 * Real.sqrt 3 - 4 := by
  sorry

#check tiger_roaming_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiger_roaming_area_l672_67231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pumps_time_to_empty_l672_67250

/-- Represents the time taken for both pumps to empty the remaining water -/
noncomputable def time_to_empty (total_water : ℝ) (pump_x_rate : ℝ) (pump_y_rate : ℝ) : ℝ :=
  (total_water / 2) / (pump_x_rate + pump_y_rate)

/-- Theorem stating the time taken for both pumps to empty the remaining water -/
theorem pumps_time_to_empty (total_water : ℝ) 
  (h1 : total_water > 0)
  (h2 : total_water / 2 / 3 = pump_x_rate)
  (h3 : total_water / 20 = pump_y_rate) :
  time_to_empty total_water pump_x_rate pump_y_rate = 30 / 13 := by
  sorry

#check pumps_time_to_empty

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pumps_time_to_empty_l672_67250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_perpendicular_chords_l672_67283

theorem circle_radius_perpendicular_chords (r : ℝ) (A B C D : ℝ × ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | (x - r)^2 + y^2 = r^2}
  let AC := {(x, y) : ℝ × ℝ | ∃ t, (x, y) = (1-t) • A + t • C}
  let BD := {(x, y) : ℝ × ℝ | ∃ t, (x, y) = (1-t) • B + t • D}
  A ∈ circle → B ∈ circle → C ∈ circle → D ∈ circle →
  (∀ P ∈ AC, ∀ Q ∈ BD, (P - Q) • (C - A) = 0) →
  ‖A - B‖ = 3 →
  ‖C - D‖ = 4 →
  r = 5/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_perpendicular_chords_l672_67283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_436_to_nearest_tenth_l672_67253

-- Define the rounding function
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

-- Theorem statement
theorem round_436_to_nearest_tenth :
  roundToNearestTenth 4.36 = 4.4 := by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_436_to_nearest_tenth_l672_67253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_product_l672_67265

theorem tan_sum_product (tan15 tan30 : ℝ) : 
  Real.tan (π/4) = 1 →
  Real.tan (π/12 + π/6) = (tan15 + tan30) / (1 - tan15 * tan30) →
  (1 + tan15) * (1 + tan30) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_product_l672_67265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_20_l672_67281

def S : Set ℤ := {-15, -5, -4, 1, 5, 20}

theorem largest_quotient_is_20 :
  ∀ a b : ℤ, a ∈ S → b ∈ S → a ≠ 0 → (b : ℚ) / a ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_20_l672_67281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l672_67269

theorem problem_1 : (-49) - 91 - (-5) + (-9) = -144 := by sorry

theorem problem_2 : -4 / 36 * (-1/9) = 1/81 := by sorry

theorem problem_3 : 24 * (1/6 - 0.75 - 2/3) = -30 := by sorry

theorem problem_4 : -(2^4) - 6 / (-2) * |(-1/3)| = -15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l672_67269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_properties_l672_67206

/-- A polynomial type with two variables -/
structure BivarPolynomial (α : Type*) where
  coeff : Fin 2 → ℕ → α

/-- The degree of a polynomial -/
noncomputable def degree (p : BivarPolynomial ℤ) : ℕ :=
  sorry

/-- The coefficient of the highest degree term -/
noncomputable def leading_coeff (p : BivarPolynomial ℤ) : ℤ :=
  sorry

/-- Rearrange a polynomial in descending powers of x -/
noncomputable def rearrange_by_x (p : BivarPolynomial ℤ) : BivarPolynomial ℤ :=
  sorry

theorem polynomial_properties (n m : ℤ) :
  let p : BivarPolynomial ℤ := {
    coeff := λ i j =>
      if i = 0 ∧ j = 2 then -n
      else if i = 1 ∧ j = m + 1 then -n
      else if i = 0 ∧ j = 1 ∧ i = 1 ∧ j = 2 then 1
      else if i = 0 ∧ j = 5 then -3
      else if i = 0 ∧ j = 0 then -6
      else 0
  }
  degree p = 7 ∧ leading_coeff p = 8 →
  m = 4 ∧ n = -8 ∧
  rearrange_by_x p = {
    coeff := λ i j =>
      if i = 0 ∧ j = 5 then -3
      else if i = 0 ∧ j = 2 ∧ i = 1 ∧ j = 5 then 8
      else if i = 0 ∧ j = 1 ∧ i = 1 ∧ j = 2 then 1
      else if i = 0 ∧ j = 0 then -6
      else 0
  } :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_properties_l672_67206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_pi_half_plus_two_alpha_l672_67284

theorem cos_five_pi_half_plus_two_alpha (α : ℝ) 
  (h1 : Real.tan α = 2) 
  (h2 : α ∈ Set.Ioo 0 π) : 
  Real.cos ((5 * π) / 2 + 2 * α) = - 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_pi_half_plus_two_alpha_l672_67284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_iff_b_range_l672_67211

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x^2 - a*x
noncomputable def g (a b x : ℝ) : ℝ := b + a * Real.log (x - 1)

-- State the theorem
theorem no_common_points_iff_b_range (a b : ℝ) (h : a ≥ 1) :
  (∀ x > 1, f a x ≠ g a b x) ↔ b < 3/4 + Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_iff_b_range_l672_67211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l672_67237

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def geometric_subsequence (a : ℕ → ℝ) (i j k : ℕ) : Prop :=
  (a j)^2 = a i * a k

/-- The common ratio of a geometric sequence -/
noncomputable def geometric_ratio (a : ℕ → ℝ) (i j : ℕ) : ℝ :=
  a j / a i

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_subsequence a 2 3 6) :
  geometric_ratio a 2 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l672_67237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_problem_l672_67217

open Finset Nat

theorem circle_sum_problem (a b c d e f g h i : ℕ) : 
  a ∈ Ico 1 10 ∧ b ∈ Ico 1 10 ∧ c ∈ Ico 1 10 ∧ 
  d ∈ Ico 1 10 ∧ e ∈ Ico 1 10 ∧ f ∈ Ico 1 10 ∧ 
  g ∈ Ico 1 10 ∧ h ∈ Ico 1 10 ∧ i ∈ Ico 1 10 ∧
  ({a, b, c, d, e, f, g, h, i} : Finset ℕ).card = 9 ∧
  (∃ s : ℕ, 
    a + b + c = s ∧
    a + d + g = s ∧
    g + h + i = s ∧
    c + f + i = s ∧
    a + e + i = s ∧
    c + e + g = s ∧
    b + e + h = s ∧
    d + e + f = s ∧
    b + d + f = s) →
  a + d + g = 18 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_problem_l672_67217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_impossibility_l672_67272

theorem circle_arrangement_impossibility :
  ¬ ∃ (arr : List ℕ),
    (arr.length = 25) ∧
    (∀ n, n ∈ arr ↔ 1 ≤ n ∧ n ≤ 25) ∧
    (∀ i, i < arr.length →
      let j := (i + 1) % arr.length
      (arr[i]! = arr[j]! + 10 ∨ arr[i]! = arr[j]! - 10) ∨ 
      (arr[i]! ∣ arr[j]!) ∨ 
      (arr[j]! ∣ arr[i]!)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_impossibility_l672_67272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l672_67258

/-- Given two plane vectors a and b, prove that their difference has magnitude 1 -/
theorem vector_difference_magnitude
  (a b : ℝ × ℝ)  -- Two-dimensional real vectors
  (ha : ‖a‖ = Real.sqrt 3)  -- Magnitude of a
  (hb : ‖b‖ = 1)  -- Magnitude of b
  (hangle : Real.cos (π / 6) = Real.sqrt 3 / 2)  -- Angle between a and b is π/6
  : ‖a - b‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l672_67258
