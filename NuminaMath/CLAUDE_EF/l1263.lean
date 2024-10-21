import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_y_coordinate_l1263_126339

/-- Given a circle C with maximum radius 2 and points (2,0) and (-2,y) on the circle,
    prove that the y-coordinate of the second point must be 0. -/
theorem circle_point_y_coordinate (C : Set (ℝ × ℝ)) (y : ℝ) : 
  (∃ (center : ℝ × ℝ) (r : ℝ), 
    C = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2} ∧ 
    r ≤ 2 ∧
    (2, 0) ∈ C ∧
    (-2, y) ∈ C) →
  y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_y_coordinate_l1263_126339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_6_rational_others_l1263_126333

-- Define the numbers
def a : ℚ := -3
def b : ℚ := 0
def c : ℚ := 1/2
noncomputable def d : ℝ := Real.sqrt 6

-- Theorem stating that √6 is irrational while the others are rational
theorem irrational_sqrt_6_rational_others :
  (∃ (q : ℚ), (q : ℝ) ^ 2 = 6) ∧ 
  (∃ (q : ℚ), (q : ℝ) = a) ∧
  (∃ (q : ℚ), (q : ℝ) = b) ∧
  (∃ (q : ℚ), (q : ℝ) = c) ∧
  ¬(∃ (q : ℚ), (q : ℝ) = d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_6_rational_others_l1263_126333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1263_126317

/-- A hyperbola passing through (0, -2) with a focus at (0, -4) has the standard equation y^2/4 - x^2/12 = 1 -/
theorem hyperbola_equation (C : Set (ℝ × ℝ)) (h1 : (0, -2) ∈ C) (h2 : ∃ F ∈ C, F = (0, -4)) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ C ↔ y^2/a^2 - x^2/b^2 = 1) ∧
    a^2 = 4 ∧ b^2 = 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1263_126317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_power_and_inverse_prop_l1263_126315

-- Define a power function
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = Real.rpow x α

-- Define an inverse proportion function
def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

-- Theorem statement
theorem function_power_and_inverse_prop (f : ℝ → ℝ) :
  is_power_function f ∧ is_inverse_proportion f →
  ∀ x : ℝ, x ≠ 0 → f x = Real.rpow x (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_power_and_inverse_prop_l1263_126315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_decrease_l1263_126350

/-- Given an equilateral triangle with area 121√3 cm², prove that decreasing each side by 8 cm
    results in an area decrease of 72√3 cm². -/
theorem equilateral_triangle_area_decrease :
  ∀ (s : ℝ),
    s > 0 →
    s^2 * Real.sqrt 3 / 4 = 121 * Real.sqrt 3 →
    let s' := s - 8
    (s'^2 * Real.sqrt 3 / 4) = 121 * Real.sqrt 3 - 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_decrease_l1263_126350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_g_increasing_l1263_126305

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * log x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2/x

-- Theorem 1: Minimum value of f when a = -2
theorem f_minimum_value (x : ℝ) (hx : x > 0) : 
  f (-2) x ≥ f (-2) 1 ∧ f (-2) 1 = 1 := by
  sorry

-- Theorem 2: Condition for g to be increasing
theorem g_increasing (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → Monotone (g a)) → a ≥ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_g_increasing_l1263_126305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1263_126352

theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) 
  (h_log : Real.log b / Real.log a > 1) : (b - 1) * (b - a) > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1263_126352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mahesh_completion_time_l1263_126335

/-- Represents the time it takes for Mahesh to complete the work -/
noncomputable def mahesh_time (total_work : ℝ) (mahesh_rate : ℝ) : ℝ :=
  total_work / mahesh_rate

/-- Represents the given conditions and proves that Mahesh takes 60 days to complete the work -/
theorem mahesh_completion_time 
  (total_work : ℝ)
  (mahesh_rate : ℝ)
  (rajesh_rate : ℝ)
  (h1 : rajesh_rate = total_work / 45)
  (h2 : 20 * mahesh_rate + 30 * rajesh_rate = total_work) :
  mahesh_time total_work mahesh_rate = 60 := by
  sorry

#check mahesh_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mahesh_completion_time_l1263_126335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_in_options_l1263_126398

def Answer : Type := String

def options : List Answer := ["the other", "some", "another", "other"]

def correct_answer : Answer := "another"

theorem correct_answer_in_options : correct_answer ∈ options :=
  by simp [correct_answer, options]

#check correct_answer_in_options

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_in_options_l1263_126398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_example_l1263_126342

/-- Calculates compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time) - principal

/-- Theorem stating that the compound interest on $1200 for 1 year at 20% p.a., compounded yearly, is $240 -/
theorem compound_interest_example : 
  compound_interest 1200 0.20 1 1 = 240 := by
  -- Unfold the definition of compound_interest
  unfold compound_interest
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_example_l1263_126342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_shooting_probability_l1263_126362

theorem basketball_shooting_probability (p_first_shot p_second_given_first p_second_given_miss : ℝ) :
  p_first_shot = 3/4 →
  p_second_given_first = 3/4 →
  p_second_given_miss = 1/4 →
  p_first_shot * p_second_given_first + (1 - p_first_shot) * p_second_given_miss = 5/8 := by
  intros h1 h2 h3
  sorry

#check basketball_shooting_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_shooting_probability_l1263_126362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_length_l1263_126311

/-- The length of a straight line segment in the grid -/
def straight_length : ℝ := 2

/-- The length of a slanted line segment in the grid -/
noncomputable def slanted_length : ℝ := 2 * Real.sqrt 2

/-- The number of straight line segments in X -/
def x_straight : ℕ := 1

/-- The number of slanted line segments in X -/
def x_slanted : ℕ := 2

/-- The number of straight line segments in Y -/
def y_straight : ℕ := 1

/-- The number of slanted line segments in Y -/
def y_slanted : ℕ := 2

/-- The number of straight line segments in Z -/
def z_straight : ℕ := 2

/-- The number of slanted line segments in Z -/
def z_slanted : ℕ := 1

/-- The total length of line segments forming XYZ -/
noncomputable def total_length : ℝ :=
  (x_straight + y_straight + z_straight) * straight_length +
  (x_slanted + y_slanted + z_slanted) * slanted_length

theorem xyz_length :
  total_length = 8 + 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_length_l1263_126311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_birds_l1263_126368

def farm_birds (p q : ℕ) : Prop :=
  -- p is a prime number between 100 and 200
  Prime p ∧ 100 < p ∧ p < 200 ∧
  -- q is the smallest multiple of 7 greater than 5.25p
  ∃ n : ℕ, q = 7 * n ∧ q > (5.25 : ℝ) * (p : ℝ) ∧ ∀ m : ℕ, 7 * m > (5.25 : ℝ) * (p : ℝ) → q ≤ 7 * m

theorem total_birds (p q : ℕ) (h : farm_birds p q) :
  (7.75 : ℝ) * (p : ℝ) + (q : ℝ) = 
    (p : ℝ) +                    -- chickens
    1.5 * (p : ℝ) +              -- ducks
    ((7.5 : ℝ) - 2.25) * (p : ℝ) + -- turkeys
    (q : ℝ)                      -- pigeons
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_birds_l1263_126368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l1263_126371

def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 5}

def A : Set ℤ := {x | ∃ n : ℕ, 6 = n * (3 - x)}

def B : Set ℤ := {4, 5}

theorem intersection_complement_equality :
  (A ∩ (U \ B)) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l1263_126371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1263_126364

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  arithmetic_property : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  sum_n seq 10 = 120 → seq.a 2 + seq.a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1263_126364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l1263_126312

noncomputable def infinite_geometric_sum (θ : ℝ) : ℝ :=
  ∑' n, (Real.sin θ) ^ (2 * n)

theorem sin_double_angle (θ : ℝ) 
  (h : infinite_geometric_sum θ = 5/3) : Real.sin (2 * θ) = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l1263_126312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1263_126307

noncomputable section

def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x, x > 0 → ∃ y, f x = y

axiom f_derivative_equation : ∀ x > 0, x * (deriv f x) - f x = (x - 1) * Real.exp x

axiom f_at_one : f 1 = 0

theorem f_properties :
  (f 2 / 2 < f 3 / 3) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  (¬ ∃ M : ℝ, ∀ x > 0, f x ≤ M) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1263_126307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1263_126328

structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

def on_hyperbola (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

noncomputable def focal_length (h : Hyperbola) : ℝ :=
  4 * Real.sqrt 2

theorem hyperbola_properties (h : Hyperbola) 
  (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) 
  (h_on_hyperbola : on_hyperbola h x₀ y₀)
  (h_focal_length : focal_length h = 4 * Real.sqrt 2)
  (h_line_intersection : 3 * x₀ = x₁ + 2 * x₂ ∧ 3 * y₀ = y₁ + 2 * y₂) :
  (x₁ * x₂ - y₁ * y₂ = 9) ∧
  (∃ (max_area : ℝ), max_area = 9/2 ∧ 
    ∀ (area : ℝ), area ≤ max_area) ∧
  (∃ (fixed_point₁ fixed_point₂ : ℝ × ℝ),
    fixed_point₁ = (2 * Real.sqrt 2, 0) ∧
    fixed_point₂ = (-2 * Real.sqrt 2, 0) ∧
    -- Additional property about the circle passing through these points
    True) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1263_126328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_properties_l1263_126330

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem periodic_function_properties (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x)
  (h_equal : f ω φ (π/2) = f ω φ (2*π/3))
  (h_φ_bound : |φ| < π/2) :
  ω = 2 ∧ φ = π/3 ∧ 
  ∀ x ∈ Set.Icc (-π/3) (π/6), -Real.sqrt 3 / 2 ≤ f ω φ x ∧ f ω φ x ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_properties_l1263_126330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_CD_is_10_l1263_126389

-- Define the segment AB and its length
noncomputable def AB : ℝ := 40

-- Define point M as a real number between 0 and AB
noncomputable def M : ℝ := sorry

-- Define the other points as functions of M and AB
noncomputable def N : ℝ := M / 2
noncomputable def P : ℝ := (AB + M) / 2
noncomputable def C : ℝ := (3 * M) / 4
noncomputable def D : ℝ := (AB + 3 * M) / 4

-- State the theorem
theorem length_CD_is_10 (h1 : 0 < M) (h2 : M < AB) : D - C = 10 := by
  -- Expand the definitions of D and C
  unfold D C
  -- Simplify the expression
  simp [AB]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_CD_is_10_l1263_126389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadPyramid_volume_l1263_126356

/-- A quadrilateral pyramid with specific properties -/
structure QuadPyramid where
  base : Set (Fin 4 → ℝ × ℝ)
  apex : ℝ × ℝ × ℝ
  isSquareBase : Bool -- Placeholder for IsSquare property
  isRightTriangles : Bool -- Placeholder for IsRightTriangle property
  height : ℝ
  height_eq_one : height = 1
  dihedralAngle : ℝ
  dihedralAngle_eq_120 : dihedralAngle = 2 * Real.pi / 3

/-- The base area of a quadrilateral -/
noncomputable def baseArea (base : Set (Fin 4 → ℝ × ℝ)) : ℝ := 1 -- Placeholder implementation

/-- The volume of a pyramid -/
noncomputable def pyramidVolume (p : QuadPyramid) : ℝ :=
  (1 / 3) * baseArea p.base * p.height

/-- The theorem to be proved -/
theorem quadPyramid_volume (p : QuadPyramid) : pyramidVolume p = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadPyramid_volume_l1263_126356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l1263_126380

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + 2*a*x - 5*a)

-- Define the theorem
theorem f_increasing_implies_a_range :
  ∀ a : ℝ, (∀ x ≥ 2, StrictMono (f a)) →
  a ∈ Set.Icc (-2 : ℝ) 4 ∧ a ≠ 4 :=
by
  -- Placeholder for the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l1263_126380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1263_126320

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := Real.log (x - 3) - Real.log (5 - y) = 0
def equation2 (x y : ℝ) : Prop := (1/4 : ℝ) * (4^x)^(1/y) - 8 * (8^y)^(1/x) = 0

-- Define the domain conditions
def domain_conditions (x y : ℝ) : Prop := x > 3 ∧ y < 5 ∧ y ≠ 0

-- Theorem statement
theorem unique_solution :
  ∃! (x y : ℝ), domain_conditions x y ∧ equation1 x y ∧ equation2 x y ∧ x = 6 ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1263_126320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_payment_total_interest_paid_l1263_126309

/-- Represents a loan with annual payments -/
structure Loan where
  c : ℝ  -- Principal amount
  p : ℝ  -- Annual interest rate (in percentage)
  n : ℕ  -- Loan term in years
  a : ℝ  -- Annual payment

/-- Calculates the growth factor for one year -/
noncomputable def growthFactor (loan : Loan) : ℝ :=
  1 + loan.p / 100

/-- Theorem stating the correct annual payment amount -/
theorem annual_payment (loan : Loan) :
  loan.a = loan.c * (growthFactor loan) * ((growthFactor loan) - 1) / ((growthFactor loan)^loan.n - 1) := by
  sorry

/-- Theorem stating the total interest paid over the loan term -/
theorem total_interest_paid (loan : Loan) :
  (loan.n * (growthFactor loan)^(loan.n + 1) - (loan.n + 1) * (growthFactor loan)^loan.n + 1) / ((growthFactor loan)^loan.n - 1) =
  loan.n * loan.a - loan.c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_payment_total_interest_paid_l1263_126309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1263_126319

/-- The circle with equation x² + y² + 4y - 21 = 0 -/
def my_circle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 + 4*p.2 - 21 = 0}

/-- The point M(-3, -3) -/
def point_M : ℝ × ℝ := (-3, -3)

/-- A line passing through point M(-3, -3) -/
def line_through_M (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 + 3 = k * (p.1 + 3)}

/-- The chord length intercepted by a line on the circle -/
noncomputable def chord_length (l : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The equation of line l is either x + 2y + 9 = 0 or 2x - y + 3 = 0 -/
theorem line_equation :
  ∃ (l : Set (ℝ × ℝ)), 
    point_M ∈ l ∧ 
    chord_length l = 4 * Real.sqrt 5 ∧
    ((∀ p ∈ l, p.1 + 2*p.2 + 9 = 0) ∨ (∀ p ∈ l, 2*p.1 - p.2 + 3 = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1263_126319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_and_equilateral_same_center_implies_equilateral_l1263_126376

-- Define a structure for a triangle with altitudes
structure TriangleWithAltitudes where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

-- Define the theorem
theorem triangle_inequality_and_equilateral (t : TriangleWithAltitudes) 
  (h_ab_gt_bc : dist t.A t.B > dist t.B t.C) :
  dist t.A t.B + dist t.C t.R ≥ dist t.B t.C + dist t.A t.P := by
  sorry

-- Define a structure for a triangle with inscribed and circumscribed circles
structure TriangleWithCircles where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  I : ℝ × ℝ
  O : ℝ × ℝ

-- Define the theorem for equilateral triangle
theorem same_center_implies_equilateral (t : TriangleWithCircles) 
  (h_same_center : t.I = t.O) :
  dist t.A t.B = dist t.B t.C ∧ dist t.B t.C = dist t.C t.A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_and_equilateral_same_center_implies_equilateral_l1263_126376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_iff_a_eq_neg_one_l1263_126338

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a*y - 2 = 0

-- Define what it means for two lines to be parallel
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂, f x₁ y₁ ∧ f x₂ y₂ → 
    ∃ m : ℝ, (x₂ - x₁ = m * (y₂ - y₁)) ∧
    ∀ x y, g x y → ∃ k : ℝ, x - x₁ = k * (y - y₁)

-- State the theorem
theorem parallel_lines_iff_a_eq_neg_one (a : ℝ) :
  parallel l₁ (l₂ a) ↔ a = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_iff_a_eq_neg_one_l1263_126338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teams_combined_stats_l1263_126321

/-- Represents the weight statistics of a team --/
structure TeamStats where
  avgWeight : ℝ
  variance : ℝ
  memberRatio : ℝ

/-- Calculates the combined average weight of two teams --/
noncomputable def combinedAvgWeight (teamA teamB : TeamStats) : ℝ :=
  (teamA.avgWeight * teamA.memberRatio + teamB.avgWeight * teamB.memberRatio) / (teamA.memberRatio + teamB.memberRatio)

/-- Calculates the combined variance of two teams --/
noncomputable def combinedVariance (teamA teamB : TeamStats) (combinedAvg : ℝ) : ℝ :=
  (teamA.memberRatio * (teamA.variance + (teamA.avgWeight - combinedAvg)^2) +
   teamB.memberRatio * (teamB.variance + (teamB.avgWeight - combinedAvg)^2)) /
  (teamA.memberRatio + teamB.memberRatio)

theorem teams_combined_stats :
  let teamA : TeamStats := { avgWeight := 60, variance := 200, memberRatio := 1 }
  let teamB : TeamStats := { avgWeight := 68, variance := 300, memberRatio := 3 }
  let combinedAvg := combinedAvgWeight teamA teamB
  combinedAvg = 66 ∧ combinedVariance teamA teamB combinedAvg = 287 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teams_combined_stats_l1263_126321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_ride_percentage_increase_l1263_126375

/-- Represents a bike ride with distances for each hour -/
structure BikeRide where
  first_hour : ℚ
  second_hour : ℚ
  third_hour : ℚ

/-- Calculates the percentage increase between two values -/
def percentage_increase (a b : ℚ) : ℚ :=
  (b - a) / a * 100

/-- The bike ride satisfying the given conditions -/
def james_ride : BikeRide :=
  { first_hour := 20
    second_hour := 24
    third_hour := 30 }

theorem james_ride_percentage_increase :
  let ride := james_ride
  -- First hour condition
  ride.second_hour = ride.first_hour * (1 + 1/5) →
  -- Total distance condition
  ride.first_hour + ride.second_hour + ride.third_hour = 74 →
  -- Third hour greater than second hour
  ride.third_hour > ride.second_hour →
  -- Prove that the percentage increase is 25%
  percentage_increase ride.second_hour ride.third_hour = 25 := by
  sorry

#eval percentage_increase james_ride.second_hour james_ride.third_hour

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_ride_percentage_increase_l1263_126375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_handshakes_with_X_leaders_l1263_126332

/-- Represents a participant in the meeting -/
structure Participant where
  country : Nat
  isLeader : Bool

/-- The total number of countries -/
def numCountries : Nat := 34

/-- The set of all participants except the leader of country X -/
def S : Finset Participant := sorry

/-- The number of handshakes for each participant in S -/
def handshakes : Participant → Fin 67 := sorry

/-- The leader of country X -/
def leaderX : Participant := sorry

/-- The deputy of country X -/
def deputyX : Participant := sorry

/-- No leader shakes hands with their own deputy -/
axiom no_leader_deputy_handshake :
  ∀ p q : Participant, p.country = q.country → p.isLeader ≠ q.isLeader →
    handshakes p ≠ handshakes q

/-- Every member of S shakes hands with a different number of people -/
axiom unique_handshakes : ∀ p q : Participant, p ∈ S → q ∈ S → p ≠ q → handshakes p ≠ handshakes q

/-- The main theorem: 33 people shook hands with the leader or deputy of country X -/
theorem handshakes_with_X_leaders :
  (S.filter (fun p => handshakes p = handshakes leaderX ∨ handshakes p = handshakes deputyX)).card = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_handshakes_with_X_leaders_l1263_126332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_ball_higher_probability_l1263_126322

/-- The probability that a ball is tossed into bin k -/
noncomputable def bin_probability (k : ℕ+) : ℝ := 3^(-(k.val : ℝ))

/-- The probability that both balls land in the same bin k -/
noncomputable def same_bin_probability (k : ℕ+) : ℝ := (bin_probability k) * (bin_probability k)

/-- The sum of probabilities for both balls landing in the same bin for all k -/
noncomputable def total_same_bin_probability : ℝ := ∑' k, same_bin_probability k

/-- The probability that the blue ball is in a higher-numbered bin than the yellow ball -/
noncomputable def blue_higher_probability : ℝ := (1 - total_same_bin_probability) / 2

theorem blue_ball_higher_probability :
  blue_higher_probability = 7/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_ball_higher_probability_l1263_126322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_parameters_l1263_126308

/-- Given a sine function y = a * sin(b * x + c) with the specified properties,
    prove that a = 3, b = 1, and c = π/3 -/
theorem sine_function_parameters
  (a b c : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_max : ∀ x, a * Real.sin (b * x + c) ≤ 3)
  (h_max_achieved : ∃ x, a * Real.sin (b * x + c) = 3)
  (h_period : ∀ x, a * Real.sin (b * x + c) = a * Real.sin (b * (x + 2 * Real.pi) + c))
  (h_first_max : a * Real.sin (b * (Real.pi / 6) + c) = 3) :
  a = 3 ∧ b = 1 ∧ c = Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_parameters_l1263_126308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_day_2020_l1263_126363

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr

/-- Returns the next day of the week -/
def DayOfWeek.nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with 
  | .Sunday => .Monday
  | .Monday => .Tuesday
  | .Tuesday => .Wednesday
  | .Wednesday => .Thursday
  | .Thursday => .Friday
  | .Friday => .Saturday
  | .Saturday => .Sunday

/-- Returns the day of the week after adding a number of days -/
def addDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match (days % 7) with
  | 0 => start
  | 1 => start.nextDay
  | 2 => start.nextDay.nextDay
  | 3 => start.nextDay.nextDay.nextDay
  | 4 => start.nextDay.nextDay.nextDay.nextDay
  | 5 => start.nextDay.nextDay.nextDay.nextDay.nextDay
  | _ => start.nextDay.nextDay.nextDay.nextDay.nextDay.nextDay

theorem leap_day_2020 (h1996 : DayOfWeek) (h : h1996 = DayOfWeek.Thursday) :
  addDays h1996 8767 = DayOfWeek.Wednesday := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_day_2020_l1263_126363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shifted_even_condition_l1263_126387

noncomputable def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def cos_shifted (φ : ℝ) : ℝ → ℝ := λ x => Real.cos (x + φ)

theorem cos_shifted_even_condition (φ : ℝ) :
  (φ = 0 → is_even (cos_shifted φ)) ∧
  ¬(is_even (cos_shifted φ) → φ = 0) := by
  sorry

#check cos_shifted_even_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shifted_even_condition_l1263_126387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_angle_sum_l1263_126337

/-- A hexagon is a polygon with 6 vertices -/
structure Hexagon where
  vertices : Fin 6 → Plane

/-- The sum of interior angles of a hexagon is 360 degrees -/
def InteriorAngleSum (h : Hexagon) : ℝ := sorry

theorem hexagon_angle_sum (h : Hexagon) : 
  InteriorAngleSum h = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_angle_sum_l1263_126337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l1263_126334

/-- The curve function --/
noncomputable def f (a x : ℝ) : ℝ := a * x^2 - Real.log x

/-- The derivative of the curve function --/
noncomputable def f' (a x : ℝ) : ℝ := 2 * a * x - 1 / x

theorem curve_properties (a : ℝ) :
  (f' a 1 = 0) →
  (a = 1/2) ∧
  (∀ x > 1, StrictMono (fun x => f (1/2) x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l1263_126334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_pi_over_ten_l1263_126329

open Real MeasureTheory

/-- The volume of the solid formed by rotating the region bounded by y = x^3 and y = x^2 around the y-axis -/
noncomputable def rotationVolume : ℝ := ∫ y in Set.Icc 0 1, Real.pi * ((y^(1/3))^2 - y)

/-- The theorem stating that the volume of the solid of revolution is π/10 -/
theorem volume_is_pi_over_ten : rotationVolume = Real.pi / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_pi_over_ten_l1263_126329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cylinder_radius_l1263_126391

/-- Represents a pyramid with a specific geometry -/
structure Pyramid where
  -- Base triangle side length
  base_side : ℝ
  -- Height of the pyramid (length of BD)
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  -- Radius of the cylinder
  radius : ℝ

/-- Predicate to assert that a vertex of the pyramid lies on the cylinder surface -/
def vertex_on_cylinder_surface (p : Pyramid) (c : Cylinder) (vertex : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- Predicate to assert that the cylinder axis intersects BD and plane ABC -/
def cylinder_axis_intersects_BD_and_ABC (p : Pyramid) (c : Cylinder) : Prop :=
  sorry

/-- Main theorem: The radius of the cylinder that circumscribes the pyramid
    is either 5√6/2 or 20√3/√17 -/
theorem pyramid_cylinder_radius (p : Pyramid) (c : Cylinder)
  (h1 : p.base_side = 12)
  (h2 : p.height = 10 * Real.sqrt 3)
  (h3 : ∀ vertex, vertex_on_cylinder_surface p c vertex)
  (h4 : cylinder_axis_intersects_BD_and_ABC p c) :
  c.radius = 5 * Real.sqrt 6 / 2 ∨ c.radius = 20 * Real.sqrt 3 / Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cylinder_radius_l1263_126391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1263_126369

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 7
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the intersection points A and B
def A (ρ : ℝ) : Prop := C₁ (ρ * Real.cos (Real.pi/6)) (ρ * Real.sin (Real.pi/6)) ∧ ρ > 0
def B (ρ : ℝ) : Prop := C₂ ρ (Real.pi/6) ∧ ρ > 0

-- Theorem statement
theorem intersection_distance :
  ∃ (ρ₁ ρ₂ : ℝ), A ρ₁ ∧ B ρ₂ ∧ |ρ₁ - ρ₂| = 3 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1263_126369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_function_minimum_l1263_126349

-- Part 1: Inequality solution
def inequality_solution : Set ℝ :=
  {x : ℝ | x ≥ 3 ∨ (-1 ≤ x ∧ x < 1)}

theorem inequality_theorem :
  {x : ℝ | 4 / (x - 1) ≤ x - 1} = inequality_solution :=
by sorry

-- Part 2: Function minimum
noncomputable def f (x : ℝ) : ℝ := 2 / x + 9 / (1 - 2 * x)

theorem function_minimum :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (1/2) ∧
  ∀ (y : ℝ), y ∈ Set.Ioo 0 (1/2) → f x ≤ f y ∧
  f x = 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_function_minimum_l1263_126349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_three_zeros_l1263_126318

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (1/2) * a * x^2 + a/2

theorem g_three_zeros (a : ℝ) (ha : a > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (deriv (g a)) x₁ = 0 ∧ (deriv (g a)) x₂ = 0) →
  ∃ z₁ z₂ z₃ : ℝ, z₁ < z₂ ∧ z₂ < z₃ ∧ g a z₁ = 0 ∧ g a z₂ = 0 ∧ g a z₃ = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_three_zeros_l1263_126318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inscribed_circle_radius_l1263_126379

/-- The radius of the inscribed circle in an isosceles triangle with two sides of length 8 and base of length 10 -/
noncomputable def inscribedCircleRadius : ℝ := 5 * Real.sqrt 15 / 13

/-- Theorem: The radius of the inscribed circle in an isosceles triangle with two sides of length 8 and base of length 10 is 5√15/13 -/
theorem isosceles_triangle_inscribed_circle_radius :
  let a : ℝ := 8
  let b : ℝ := 8
  let c : ℝ := 10
  let s : ℝ := (a + b + c) / 2
  let area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = inscribedCircleRadius := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inscribed_circle_radius_l1263_126379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_line_slope_l1263_126370

-- Define the circles
def circle1 : ℝ × ℝ := (10, 90)
def circle2 : ℝ × ℝ := (15, 80)
def circle3 : ℝ × ℝ := (20, 85)
def radius : ℝ := 4

-- Define the line passing through (15, 80)
def line_point : ℝ × ℝ := (15, 80)

-- Helper function to calculate area on one side of the line (not implemented)
noncomputable def area_on_one_side (c1 c2 c3 : ℝ × ℝ) (r : ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
sorry

-- Helper function to calculate area on the other side of the line (not implemented)
noncomputable def area_on_other_side (c1 c2 c3 : ℝ × ℝ) (r : ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
sorry

-- Theorem statement
theorem equal_area_line_slope :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y - 80 = m * (x - 15) → 
      (area_on_one_side circle1 circle2 circle3 radius (λ x y ↦ y - 80 = m * (x - 15)) = 
       area_on_other_side circle1 circle2 circle3 radius (λ x y ↦ y - 80 = m * (x - 15)))) ∧
    (|m| = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_line_slope_l1263_126370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_non_perfect_square_product_l1263_126382

theorem existence_of_non_perfect_square_product (d : ℕ) 
  (h_pos : d > 0) 
  (h_neq_2 : d ≠ 2) 
  (h_neq_5 : d ≠ 5) 
  (h_neq_13 : d ≠ 13) : 
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_non_perfect_square_product_l1263_126382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1263_126381

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | x > -a}

-- Define the function g
noncomputable def g : ℝ → ℝ := fun x ↦ Real.log (x - 1)

-- Define the domain N of g
def N : Set ℝ := {x | x > 1}

-- Theorem statement
theorem range_of_a (a : ℝ) : M a ⊆ N → a ∈ Set.Iic (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1263_126381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_extrema_of_f_l1263_126310

/-- The function f(x) = 3x^3 - 9x^2 + 3 -/
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 3

/-- The first derivative of f(x) -/
def f' (x : ℝ) : ℝ := 9 * x^2 - 18 * x

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 18 * x - 18

theorem local_extrema_of_f :
  (∃ δ₁ > 0, ∀ x ∈ Set.Ioo (-δ₁) δ₁, f x ≤ f 0) ∧
  (∃ δ₂ > 0, ∀ x ∈ Set.Ioo (2 - δ₂) (2 + δ₂), f x ≥ f 2) := by
  sorry

#check local_extrema_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_extrema_of_f_l1263_126310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_plane_line_perpendicular_parallel_planes_perpendicular_intersecting_planes_l1263_126385

-- Define the basic types
variable (α β : Type) -- Planes
variable (l m : Type) -- Lines

-- Define the relations
def parallel (a b : Type) : Prop := sorry
def perpendicular (a b : Type) : Prop := sorry
def contained_in (a b : Type) : Prop := sorry
def intersect (a b c : Type) : Prop := sorry

-- State the theorems
theorem parallel_plane_line 
  (α β : Type) (l : Type)
  (h1 : parallel α β) (h2 : contained_in l α) : 
  parallel l β := by sorry

theorem perpendicular_parallel_planes 
  (α β : Type) (l : Type)
  (h1 : parallel α β) (h2 : perpendicular l α) : 
  perpendicular l β := by sorry

theorem perpendicular_intersecting_planes 
  (α β : Type) (l m : Type)
  (h1 : perpendicular α β) (h2 : intersect α β l) 
  (h3 : contained_in m α) (h4 : perpendicular m l) : 
  perpendicular m β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_plane_line_perpendicular_parallel_planes_perpendicular_intersecting_planes_l1263_126385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1263_126390

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

-- Define the point of tangency
def point : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ x + y - 2 = 0) ∧
    (m = -(deriv f point.1)) ∧
    (point.2 = m * point.1 + b) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1263_126390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_candy_bars_l1263_126345

/-- The number of candy bars Carl can buy after 4 weeks -/
def candy_bars (weekly_earnings : ℚ) (candy_cost : ℚ) (weeks : ℕ) : ℕ :=
  (weekly_earnings * weeks / candy_cost).floor.toNat

/-- Theorem stating that Carl can buy 6 candy bars after 4 weeks -/
theorem carl_candy_bars :
  candy_bars (75/100) (1/2) 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_candy_bars_l1263_126345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paintings_count_l1263_126316

/-- Checks if two numbers are consecutive in a circular sequence from 1 to 8 -/
def isConsecutive (a b : Nat) : Bool :=
  (a % 8 + 1 == b % 8) || (b % 8 + 1 == a % 8)

/-- Counts valid ways to paint the die -/
def countValidPaintings : Nat :=
  let faces := [1, 2, 3, 4, 5, 6, 7, 8]
  List.sum (List.map (fun (pair : Nat × Nat) =>
    let (a, b) := pair
    List.length (List.filter (fun c => c ≠ a && c ≠ b && !isConsecutive c a && !isConsecutive c b) faces)
  ) (List.filter (fun (pair : Nat × Nat) =>
    let (a, b) := pair
    a < b && a + b ≠ 9
  ) (List.zip faces faces)))

theorem valid_paintings_count : countValidPaintings = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paintings_count_l1263_126316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_minus_one_plus_one_l1263_126343

theorem gcd_power_minus_one_plus_one (m n : ℕ+) : 
  let d := Nat.gcd m.val n.val
  let x := 2^m.val - 1
  let y := 2^n.val + 1
  (m.val / d % 2 = 1 → Nat.gcd x y = 1) ∧
  (m.val / d % 2 = 0 → Nat.gcd x y = 2^d + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_minus_one_plus_one_l1263_126343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coach_ralph_selection_l1263_126361

/-- Given N(N+1) distinct heights, it's always possible to select 2N heights
    such that when arranged in descending order, no pair of consecutively
    ranked heights has any other height between them. -/
theorem coach_ralph_selection (N : ℕ) (h : N ≥ 2) 
  (heights : Fin (N * (N + 1)) → ℝ) (distinct : ∀ i j, i ≠ j → heights i ≠ heights j) :
  ∃ (selected : Fin (2 * N) → Fin (N * (N + 1))),
    (∀ i j, i ≠ j → selected i ≠ selected j) ∧
    (∀ i : Fin (2 * N),
      ∀ k : Fin (N * (N + 1)),
        (i.val + 1 < 2 * N) →
        heights (selected i) > heights k ∧ heights k > heights (selected ⟨i.val + 1, by sorry⟩) →
        ∃ j, selected j = k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coach_ralph_selection_l1263_126361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_6_value_l1263_126348

/-- Given a real number y such that y + 1/y = 5, T_m is defined as y^m + 1/y^m -/
noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + (1/y)^m

/-- Theorem stating that T_6 equals 12098 for y satisfying y + 1/y = 5 -/
theorem T_6_value (y : ℝ) (h : y + 1/y = 5) : T y 6 = 12098 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_6_value_l1263_126348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_side_length_correct_l1263_126392

/-- The side length of a regular tetrahedron circumscribing four unit spheres -/
noncomputable def tetrahedron_side_length : ℝ := 2 + 2 * Real.sqrt 6

/-- Represents the arrangement of four unit spheres -/
structure SpheresArrangement where
  radius : ℝ
  num_spheres : ℕ
  ground_spheres : ℕ

/-- The specific arrangement in our problem -/
def problem_arrangement : SpheresArrangement :=
  { radius := 1
  , num_spheres := 4
  , ground_spheres := 3 }

/-- Theorem stating that the calculated side length is correct for the given arrangement -/
theorem tetrahedron_side_length_correct (arr : SpheresArrangement) :
  arr = problem_arrangement →
  tetrahedron_side_length = 2 + 2 * Real.sqrt 6 := by
  sorry

#check tetrahedron_side_length_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_side_length_correct_l1263_126392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l1263_126365

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - Real.cos (x / 2))
noncomputable def g (x : ℝ) : ℝ := (3 * Real.sin x + 1) / (Real.sin x - 2)

-- Theorem for the domain of f
theorem domain_of_f : 
  ∀ x : ℝ, ∃ y : ℝ, f x = y := by
  sorry

-- Theorem for the range of g
theorem range_of_g :
  ∀ y : ℝ, (∃ x : ℝ, g x = y) ↔ -4 ≤ y ∧ y ≤ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l1263_126365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1263_126340

-- Define the ellipse E
def Ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the foci
noncomputable def Foci (a b : ℝ) (h : a > b ∧ b > 0) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 - b^2)
  ((-c, 0), (c, 0))

-- Define the theorem
theorem ellipse_properties
  (a b : ℝ) (h : a > b ∧ b > 0)
  (E : Set (ℝ × ℝ)) (hE : E = Ellipse a b h)
  (F₁ F₂ : ℝ × ℝ) (hF : (F₁, F₂) = Foci a b h)
  (A B : ℝ × ℝ) (hAB : A ∈ E ∧ B ∈ E)
  (hline : ∃ (t : ℝ), A = F₁ + t • (B - F₁))
  (hAF₁ : ‖A - F₁‖ = 3 * ‖B - F₁‖)
  (hAB : ‖A - B‖ = 4)
  (hperim : ‖A - B‖ + ‖A - F₂‖ + ‖B - F₂‖ = 16)
  (hslope : (A.2 - B.2) / (A.1 - B.1) = 1) :
  (‖A - F₂‖ = 5) ∧
  (a = 4 ∧ b = 2 * Real.sqrt 2) := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1263_126340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1263_126373

/-- The distance between two parallel lines ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₂ - c₁) / Real.sqrt (a^2 + b^2)

/-- Given the lines 5x + 12y - 7 = 0 and 5x + 12y + 6 = 0, their distance is 1 -/
theorem distance_between_given_lines :
  distance_parallel_lines 5 12 (-7) 6 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1263_126373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parameterization_l1263_126353

/-- Rational parameterization of the unit circle -/
noncomputable def circle_param (t : ℝ) : ℝ × ℝ :=
  ((t^2 - 1) / (t^2 + 1), -2*t / (t^2 + 1))

/-- The line through (1,0) with slope t -/
noncomputable def line_through_point (t : ℝ) (x : ℝ) : ℝ := t * (x - 1)

theorem circle_parameterization (t : ℝ) :
  let (x, y) := circle_param t
  (x^2 + y^2 = 1) ∧
  (y = line_through_point t x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parameterization_l1263_126353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l1263_126396

noncomputable def project_completion_time (a b : ℝ) : ℝ :=
  1 / (1 / a + 1 / b)

theorem b_completion_time (a_time : ℝ) (total_time : ℝ) (a_quit_before : ℝ) 
  (h1 : a_time = 20)
  (h2 : total_time = 21)
  (h3 : a_quit_before = 15) :
  ∃ (b_time : ℝ), 
    b_time = 30 ∧
    project_completion_time a_time b_time = total_time ∧
    (total_time - a_quit_before) * (1 / a_time + 1 / b_time) + 
      a_quit_before * (1 / b_time) = 1 :=
by
  sorry

#check b_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l1263_126396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_power_sum_l1263_126394

theorem units_digit_of_power_sum (x : ℝ) (hx : x ≠ 0) (h : x + x⁻¹ = 3) :
  ∀ n : ℕ, n ≥ 1 → (x^(2^n) + x^(-(2^n : ℤ))) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_power_sum_l1263_126394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amelia_dinner_theorem_l1263_126360

/-- Calculates the amount of money Amelia has left after buying meals -/
noncomputable def money_left (initial_amount : ℝ) (first_course : ℝ) (second_course_diff : ℝ) (dessert_percentage : ℝ) : ℝ :=
  let second_course := first_course + second_course_diff
  let dessert := dessert_percentage * second_course / 100
  initial_amount - first_course - second_course - dessert

theorem amelia_dinner_theorem :
  money_left 60 15 5 25 = 20 := by
  -- Unfold the definition of money_left
  unfold money_left
  -- Simplify the arithmetic expressions
  simp [add_assoc, mul_div_assoc]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amelia_dinner_theorem_l1263_126360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_start_l1263_126300

/-- The initial position of the particle -/
def initial_position : ℂ := 6

/-- The rotation factor for each move -/
noncomputable def ω : ℂ := Complex.exp (Complex.I * Real.pi / 6)

/-- The translation distance for each move -/
def translation : ℝ := 12

/-- The number of moves -/
def num_moves : ℕ := 120

/-- The position after n moves -/
noncomputable def position (n : ℕ) : ℂ :=
  Complex.exp (Complex.I * Real.pi / 6 * n) * initial_position + 
  translation * (1 - Complex.exp (Complex.I * Real.pi / 6 * n)) / (1 - ω)

theorem particle_returns_to_start : position num_moves = initial_position := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_start_l1263_126300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probability_l1263_126354

/-- The number of red balls in the bag -/
def n : ℕ := 2

/-- The probability of winning in a single draw -/
def P (n : ℕ) : ℚ := (n^2 - n + 2 : ℚ) / (n^2 + 3*n + 2 : ℚ)

/-- The probability of winning exactly once in three draws -/
def P3_1 (p : ℚ) : ℚ := 3 * p * (1 - p)^2

theorem ball_probability :
  /- 1. The probability of winning in a single draw -/
  P n = (n^2 - n + 2 : ℚ) / (n^2 + 3*n + 2 : ℚ) ∧
  /- 2. When n = 3, the probability of winning exactly once in three draws -/
  P3_1 (P 3) = 54 / 125 ∧
  /- 3. The probability of winning exactly once in three draws is maximized when n = 2 -/
  ∀ m : ℕ, m ≥ 2 → P3_1 (P 2) ≥ P3_1 (P m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probability_l1263_126354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_combined_function_l1263_126358

-- Define a function that is three times differentiable
def ThreeTimesDifferentiable (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  Differentiable ℝ (deriv f) ∧ 
  Differentiable ℝ (deriv (deriv f))

-- Define a function that counts distinct real zeros
noncomputable def CountDistinctRealZeros (f : ℝ → ℝ) : ℕ :=
  Nat.card { x : ℝ | f x = 0 }

-- Main theorem
theorem zeros_of_combined_function 
  (f : ℝ → ℝ) 
  (h1 : ThreeTimesDifferentiable f) 
  (h2 : CountDistinctRealZeros f ≥ 5) : 
  CountDistinctRealZeros (fun x ↦ f x + 6 * deriv f x + 12 * deriv (deriv f) x + 8 * deriv (deriv (deriv f)) x) ≥ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_combined_function_l1263_126358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_rectangle_areas_l1263_126378

theorem sum_of_rectangle_areas : ∃ (total_area : ℕ), total_area = 858 := by
  -- Define the common width
  let width : ℕ := 3

  -- Define the lengths (squares of first six odd numbers)
  let lengths : List ℕ := [1, 9, 25, 49, 81, 121]

  -- Define the function to calculate area
  let area (length : ℕ) : ℕ := width * length

  -- Calculate the sum of areas
  let total_area := (lengths.map area).sum

  -- Prove that there exists a total_area equal to 858
  use total_area
  
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_rectangle_areas_l1263_126378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_ratio_bounds_l1263_126377

noncomputable section

open Real

theorem triangle_geometric_sequence_ratio_bounds (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Sides form a geometric sequence
  ∃ q : ℝ, (0 < q ∧ q ≠ 1 ∧ b = a * q ∧ c = b * q) →
  -- The expression is strictly bounded
  (sqrt 5 - 1) / 2 < (sin A + cos A * tan C) / (sin B + cos B * tan C) ∧
  (sin A + cos A * tan C) / (sin B + cos B * tan C) < (sqrt 5 + 1) / 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_ratio_bounds_l1263_126377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_identity_l1263_126393

/-- A function f: ℝ₊ → ℝ₊ satisfying f(f(x) + y) = x + f(y) for all x, y ∈ ℝ₊ is the identity function. -/
theorem functional_equation_identity (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, 0 < x → 0 < y → f (f x + y) = x + f y) 
    (pos : ∀ x : ℝ, 0 < x → 0 < f x) : 
    ∀ x : ℝ, 0 < x → f x = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_identity_l1263_126393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_distance_points_four_points_exist_l1263_126359

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the Euclidean distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- A configuration of n points in 3D space where all pairwise distances are equal -/
structure EqualDistanceConfig where
  n : ℕ
  points : Fin n → Point3D
  all_distances_equal : ∀ (i j k l : Fin n), i ≠ j → k ≠ l → 
    distance (points i) (points j) = distance (points k) (points l)

/-- The theorem stating that the maximum number of points in an equal distance configuration is 4 -/
theorem max_equal_distance_points :
  ¬∃ (config : EqualDistanceConfig), config.n > 4 := by
  sorry

/-- The theorem stating that there exists a configuration with 4 points -/
theorem four_points_exist :
  ∃ (config : EqualDistanceConfig), config.n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_distance_points_four_points_exist_l1263_126359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antifreeze_mixture_percentage_l1263_126301

/-- Represents an antifreeze solution -/
structure Antifreeze where
  volume : ℝ
  alcohol_percentage : ℝ

/-- Calculates the total alcohol volume in a solution -/
noncomputable def alcohol_volume (solution : Antifreeze) : ℝ :=
  solution.volume * solution.alcohol_percentage / 100

/-- Theorem: The resulting alcohol percentage when mixing two antifreeze solutions -/
theorem antifreeze_mixture_percentage 
  (solution1 solution2 : Antifreeze)
  (h1 : solution1.alcohol_percentage = 10)
  (h2 : solution2.alcohol_percentage = 18)
  (h3 : solution1.volume = 7.5)
  (h4 : solution2.volume = 7.5) :
  let total_volume := solution1.volume + solution2.volume
  let total_alcohol := alcohol_volume solution1 + alcohol_volume solution2
  (total_alcohol / total_volume) * 100 = 14 := by
  sorry

#check antifreeze_mixture_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antifreeze_mixture_percentage_l1263_126301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_polygon_pairs_l1263_126326

/-- The number of pairs of regular polygons satisfying the given conditions -/
def num_polygon_pairs : ℕ :=
  Finset.card (Finset.filter (fun p : ℕ × ℕ => 
    let (r, k) := p
    k < 15 ∧ 
    (r - 2) * k * 3 = (k - 2) * r * 4)
    (Finset.product (Finset.range 100) (Finset.range 15)))

/-- Theorem stating that there are exactly 4 pairs of polygons satisfying the conditions -/
theorem four_polygon_pairs : num_polygon_pairs = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_polygon_pairs_l1263_126326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1263_126314

/-- The constant term in the expansion of (2x + 1/√x)^6 is 60 -/
theorem constant_term_expansion : ∃ (f : ℝ → ℝ), 
  (∀ x, x > 0 → f x = (2*x + x^(-(1/2:ℝ)))^6) ∧ 
  (∃ c, ∀ x, x > 0 → f x = c + x * (f x - c) / x) ∧
  c = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1263_126314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1263_126372

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The sine of an angle -/
def sin : ℝ → ℝ := Real.sin

/-- The cosine of an angle -/
def cos : ℝ → ℝ := Real.cos

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := (1/2) * t.b * t.c * sin t.A

theorem triangle_theorem (t : Triangle) 
  (h1 : t.a = 2 * t.b)
  (h2 : ∃ (k : ℝ), sin t.A = sin t.C + k ∧ sin t.C = sin t.B + k)
  (h3 : area t = (15 ^ (1/8)) / 3) :
  cos (t.B + t.C) = 1/4 ∧ t.c = 4 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1263_126372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_magnitude_l1263_126367

/-- Given two vectors a and b in ℝ², where a is parallel to b,
    prove that the magnitude of their sum is √5. -/
theorem parallel_vectors_sum_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![4, -2]
  (∃ (k : ℝ), a = k • b) →
  ‖a + b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_magnitude_l1263_126367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l1263_126304

-- Define the circles and points
structure Configuration where
  -- The two circles
  small_circle : Set (EuclideanSpace ℝ (Fin 2))
  large_circle : Set (EuclideanSpace ℝ (Fin 2))
  -- The points
  A : EuclideanSpace ℝ (Fin 2)  -- Point of internal tangency
  C : EuclideanSpace ℝ (Fin 2)  -- Point on small circle
  D : EuclideanSpace ℝ (Fin 2)  -- Point where tangent at C meets large circle
  E : EuclideanSpace ℝ (Fin 2)  -- Point where tangent at C meets large circle
  P : EuclideanSpace ℝ (Fin 2)  -- Point where AC meets large circle

-- Define the properties of the configuration
def valid_configuration (config : Configuration) : Prop :=
  -- Circles are internally tangent at A
  config.A ∈ config.small_circle ∧ config.A ∈ config.large_circle ∧
  -- C is on the small circle and different from A
  config.C ∈ config.small_circle ∧ config.C ≠ config.A ∧
  -- D and E are on the large circle
  config.D ∈ config.large_circle ∧ config.E ∈ config.large_circle ∧
  -- The line DC is tangent to the small circle at C
  True ∧  -- Placeholder for is_tangent_line
  -- P is on the large circle
  config.P ∈ config.large_circle ∧
  -- A, C, and P are collinear
  True  -- Placeholder for collinear

-- Define the circle through A, C, and E
noncomputable def circle_ACE (config : Configuration) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {x | ∃ r, ‖x - config.A‖ = r ∧ ‖x - config.C‖ = r ∧ ‖x - config.E‖ = r}

-- The theorem to be proved
theorem tangent_line_theorem (config : Configuration) 
  (h : valid_configuration config) :
  True :=  -- Placeholder for is_tangent_line (circle_ACE config) (Line.throughPoints config.P config.E) config.E
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l1263_126304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1263_126324

-- Define the hyperbola parameters
variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

-- Define the midpoint of the intersection
def P : ℝ × ℝ := (2, -1)

-- Define the slope of the line
noncomputable def slope_angle : ℝ := 135 * Real.pi / 180

-- Theorem statement
theorem hyperbola_eccentricity : 
  ∃ (A B : ℝ × ℝ),
    -- A and B are on the hyperbola
    (A.1^2 / a^2 - A.2^2 / b^2 = 1) ∧
    (B.1^2 / a^2 - B.2^2 / b^2 = 1) ∧
    -- P is the midpoint of AB
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = P ∧
    -- The line AB has the given slope
    (B.2 - A.2) / (B.1 - A.1) = Real.tan slope_angle →
    -- The eccentricity of the hyperbola is √6/2
    Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 6 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1263_126324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squares_inscribed_triangle_max_sum_squares_iff_equilateral_l1263_126313

/-- Given a circle with radius R, the sum of squares of side lengths of any inscribed triangle is at most 9R^2 -/
theorem max_sum_squares_inscribed_triangle (R : ℝ) (h : R > 0) :
  ∀ A B C : ℝ × ℝ,
    (‖A‖ = R) → (‖B‖ = R) → (‖C‖ = R) →
    ‖A - B‖^2 + ‖B - C‖^2 + ‖C - A‖^2 ≤ 9 * R^2 :=
by sorry

/-- The maximum sum of squares (9R^2) is achieved if and only if the triangle is equilateral -/
theorem max_sum_squares_iff_equilateral (R : ℝ) (h : R > 0) :
  ∀ A B C : ℝ × ℝ,
    (‖A‖ = R) → (‖B‖ = R) → (‖C‖ = R) →
    (‖A - B‖^2 + ‖B - C‖^2 + ‖C - A‖^2 = 9 * R^2) ↔
    (‖A - B‖ = ‖B - C‖ ∧ ‖B - C‖ = ‖C - A‖) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squares_inscribed_triangle_max_sum_squares_iff_equilateral_l1263_126313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_when_p_or_q_false_l1263_126347

/-- The range of a when either proposition p or q is false -/
theorem range_of_a_when_p_or_q_false : 
  ∃ (S : Set ℝ), S = Set.union (Set.Ioo (-Real.pi) (-1)) (Set.Ioo 0 1) ∧
  ∀ a : ℝ, (¬(∀ x : ℝ, a^2*x^2 + x^2 > 0) ∨ 
            ¬(∃! x : ℝ, x^2 + 2*x + 2*a ≤ 0)) → 
  a ∈ S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_when_p_or_q_false_l1263_126347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_water_tank_l1263_126386

/-- Represents the dimensions and costs of a rectangular water tank -/
structure WaterTank where
  volume : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the minimum construction cost of a water tank -/
noncomputable def minConstructionCost (tank : WaterTank) : ℝ :=
  let bottomArea := tank.volume / tank.depth
  let bottomTotalCost := bottomArea * tank.bottomCost
  let perimeterLength := 2 * (2 * Real.sqrt bottomArea)
  let wallTotalCost := perimeterLength * tank.depth * tank.wallCost
  bottomTotalCost + wallTotalCost

/-- Theorem: The minimum construction cost for the specified water tank is 297600 yuan -/
theorem min_cost_water_tank :
  let tank : WaterTank := {
    volume := 4800
    depth := 3
    bottomCost := 150
    wallCost := 120
  }
  minConstructionCost tank = 297600 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_water_tank_l1263_126386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1263_126325

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_symmetry 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_bound : |φ| < π / 2) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) 
  (h_odd_translated : ∀ x, f ω φ (x + π/3) = -f ω φ (-x + π/3)) :
  ∀ x, f ω φ (5*π/12 + x) = f ω φ (5*π/12 - x) := by
  sorry

#check function_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1263_126325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1263_126355

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides opposite to angles A, B, C respectively

-- Define vectors m and n
noncomputable def m (A B C : ℝ) : ℝ × ℝ := (Real.sin B + Real.sin C, Real.sin A - Real.sin B)
noncomputable def n (A B C : ℝ) : ℝ × ℝ := (Real.sin B - Real.sin C, Real.sin A)

-- State the theorem
theorem triangle_proof 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) -- Sum of angles in a triangle
  (h5 : (m A B C).1 * (n A B C).1 + (m A B C).2 * (n A B C).2 = 0) -- m ⊥ n
  (h6 : Real.sin A = 4/5) -- Given condition
  (h7 : a / Real.sin A = b / Real.sin B) -- Law of sines
  (h8 : a / Real.sin A = c / Real.sin C) -- Law of sines
  (h9 : a^2 = b^2 + c^2 - 2*b*c*Real.cos A) -- Law of cosines
  (h10 : b^2 = a^2 + c^2 - 2*a*c*Real.cos B) -- Law of cosines
  (h11 : c^2 = a^2 + b^2 - 2*a*b*Real.cos C) -- Law of cosines
  : C = π/3 ∧ Real.cos B = (4*Real.sqrt 3 - 3)/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1263_126355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_cube_sum_tan_plus_cot_l1263_126336

-- Define the parameter a
variable (a : ℝ)

-- Define sin θ and cos θ as functions of a
noncomputable def sin_theta (a : ℝ) : ℝ := 
  (a + Real.sqrt (a^2 - 4*a)) / 2

noncomputable def cos_theta (a : ℝ) : ℝ := 
  (a - Real.sqrt (a^2 - 4*a)) / 2

-- Axioms based on the given conditions
axiom root_condition1 (a : ℝ) : (sin_theta a) ^ 2 - a * (sin_theta a) + a = 0
axiom root_condition2 (a : ℝ) : (cos_theta a) ^ 2 - a * (cos_theta a) + a = 0

-- Theorem 1
theorem sin_cos_cube_sum (a : ℝ) : 
  (sin_theta a) ^ 3 + (cos_theta a) ^ 3 = Real.sqrt 2 - 2 := by sorry

-- Theorem 2
theorem tan_plus_cot (a : ℝ) : 
  (sin_theta a / cos_theta a) + (cos_theta a / sin_theta a) = -1 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_cube_sum_tan_plus_cot_l1263_126336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_with_45_degree_tangents_l1263_126397

/-- Given a parabola y = ax^2 where a > 0, the locus of points (u, v) from which
    tangents drawn to the parabola form an angle of 45° with each other. -/
theorem locus_of_points_with_45_degree_tangents 
  (a : ℝ) (h_a : a > 0) (u v : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    (v - a*x₁^2 = 2*a*x₁*(u - x₁))
    ∧ (v - a*x₂^2 = 2*a*x₂*(u - x₂))
    ∧ |((2*a*x₁) - (2*a*x₂)) / (1 + (2*a*x₁)*(2*a*x₂))| = 1)
  ↔ (v + 3/(4*a))^2 - u^2 = 1/(2*a^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_with_45_degree_tangents_l1263_126397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_example_l1263_126351

-- Define the factorization property
def is_factorization (f g : ℝ → ℝ) : Prop :=
  ∃ (p q : ℝ → ℝ), f = λ x ↦ (p x) * (q x)

-- State the theorem
theorem factorization_example :
  is_factorization (λ x ↦ 10*x^2 - 5*x) (λ x ↦ 5*x*(2*x - 1)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_example_l1263_126351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_not_product_l1263_126357

theorem polynomial_not_product :
  ¬ ∃ (f g : Polynomial ℝ), (fun (x y : ℝ) => x^200 * y^200 + 1) = fun (x y : ℝ) => (f.eval x) * (g.eval y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_not_product_l1263_126357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_divisor_count_l1263_126395

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem product_divisor_count (n : ℕ) :
  (divisor_count (n * (n + 2) * (n + 4)) ≤ 15) ↔ n ∈ ({1, 2, 3, 4, 5, 7, 9} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_divisor_count_l1263_126395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_prime_power_with_divisor_product_1024_l1263_126388

def is_perfect_prime_power (n : ℕ) : Prop :=
  ∃ p k, Nat.Prime p ∧ n = p ^ k

def divisor_product (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem perfect_prime_power_with_divisor_product_1024 (n : ℕ) 
  (h1 : is_perfect_prime_power n) 
  (h2 : divisor_product n = 1024) : 
  n = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_prime_power_with_divisor_product_1024_l1263_126388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_to_line_l1263_126323

/-- The maximum distance from a point to a line passing through a fixed point -/
theorem max_distance_point_to_line (P A : ℝ × ℝ) (l : ℝ → ℝ) (k : ℝ) :
  P = (-1, 3) →
  A = (2, 0) →
  (∀ x, l x = k * (x - 2)) →
  (∀ x, l x = 0 → x = 2) →
  Real.sqrt ((2 + 1)^2 + (3 - 0)^2) = 3 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_to_line_l1263_126323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_ratio_l1263_126384

-- Define the quadrilateral ABCD
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the center of the inscribed circle
variable (O : EuclideanSpace ℝ (Fin 2))

-- Define the points where the inscribed circle touches the sides
variable (E1 E2 E3 E4 : EuclideanSpace ℝ (Fin 2))

-- Define the property of being a cyclic quadrilateral
def is_cyclic_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the property of being a tangential quadrilateral
def is_tangential_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the property of a point being on a line segment
def point_on_segment (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the ratio of two line segments
noncomputable def segment_ratio (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem tangent_point_ratio 
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_tangential : is_tangential_quadrilateral A B C D)
  (h_E1 : point_on_segment E1 A B)
  (h_E2 : point_on_segment E2 B C)
  (h_E3 : point_on_segment E3 C D)
  (h_E4 : point_on_segment E4 D A)
  (h_inscribed : ∃ r : ℝ, r > 0 ∧ 
    (dist O E1 = r) ∧ (dist O E2 = r) ∧ (dist O E3 = r) ∧ (dist O E4 = r)) :
  segment_ratio A E1 B = segment_ratio D E3 C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_ratio_l1263_126384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_max_value_is_one_l1263_126327

-- Define the function f on [-1,0) ∪ (0,1]
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-1) 0 then x^3 - a*x
  else if x ∈ Set.Ioo 0 1 then -x^3 + a*x
  else 0

-- State the theorem
theorem exists_a_max_value_is_one :
  ∃ (a : ℝ), ∀ (x : ℝ), x ∈ Set.Ioo 0 1 → f a x ≤ 1 ∧ ∃ (y : ℝ), y ∈ Set.Ioo 0 1 ∧ f a y = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_max_value_is_one_l1263_126327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_earnings_value_l1263_126331

def worker_earnings (E : ℝ) (day : ℕ) : ℝ :=
  if day % 2 = 1 then E else -18

def total_earnings (E : ℝ) (n : ℕ) : ℝ :=
  (List.range n).map (worker_earnings E) |>.sum

theorem worker_earnings_value (E : ℝ) :
  total_earnings E 9 = 48 → E = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_earnings_value_l1263_126331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_divisors_of_special_number_l1263_126374

theorem distinct_divisors_of_special_number (n : ℕ) (p : Fin n → ℕ) 
  (h_n : n ≥ 1)
  (h_prime : ∀ i : Fin n, Nat.Prime (p i))
  (h_distinct : ∀ i j : Fin n, i ≠ j → p i ≠ p j)
  (h_geq_5 : ∀ i : Fin n, p i ≥ 5) :
  let N := 2^(Finset.prod Finset.univ p) + 1
  ∃ (D : Finset ℕ), D ⊆ Nat.divisors N ∧ D.card ≥ 2^(2^n) :=
by
  sorry

#check distinct_divisors_of_special_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_divisors_of_special_number_l1263_126374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l1263_126346

theorem arithmetic_calculations : 
  ((-32) / 4 * (-8) = 64) ∧ 
  (-2^2 - (1 - 0.5) / (1/5) * (2 - (-2)^2) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_calculations_l1263_126346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_l1263_126399

theorem simplify_radical (x : ℝ) : 
  x = 12 - 4 * Real.sqrt 5 → Real.sqrt x = Real.sqrt 10 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_l1263_126399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l1263_126306

-- Statement ①
def statement_1 : Prop :=
  ∀ a b : ℝ, ∃! x : ℝ, a * x + b = 0

-- Statement ②
def statement_2 : Prop :=
  ∀ p q : Prop, (p ∧ q) → (q ∨ p)

-- Statement ③
def statement_3 : Prop :=
  (∀ a b : ℤ, (Even a ∧ Even b) → Even (a + b)) →
  (∀ a b : ℤ, ¬Even (a + b) → ¬(Even a ∧ Even b))

-- Statement ④
noncomputable def statement_4 : Prop :=
  (∃ x : ℝ, Real.sin x ≤ 1) ↔ ¬(∀ x : ℝ, Real.sin x > 1)

theorem correct_statements :
  ¬statement_1 ∧ statement_2 ∧ ¬statement_3 ∧ statement_4 := by
  sorry

#check correct_statements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l1263_126306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_between_circles_l1263_126383

noncomputable section

-- Define the Sphere structure
structure Sphere where
  center : ℝ × ℝ
  radius : ℝ

-- Define necessary functions
def Sphere.internallyTangent (s1 s2 : Sphere) : Prop := sorry
def Sphere.onSphere (p : ℝ × ℝ) (s : Sphere) : Prop := sorry
def Sphere.diameter (s : Sphere) (a b : ℝ × ℝ) : Prop := sorry
def Sphere.shadedArea (s1 s2 s3 : Sphere) : ℝ := sorry

theorem shaded_area_between_circles (π : ℝ) (h_π : π > 0) : 
  let r₁ : ℝ := 2
  let r₂ : ℝ := 3
  let area_shaded := (9 * Real.arccos (2/3) / π - 4 * Real.sqrt 5 / 3 - 4) * π
  ∃ (c₁ c₂ c₃ : Sphere), 
    c₁.radius = r₁ ∧ 
    c₂.radius = r₂ ∧ 
    c₃.radius = r₂ ∧
    Sphere.internallyTangent c₁ c₂ ∧
    Sphere.internallyTangent c₁ c₃ ∧
    (∃ (a b : ℝ × ℝ), 
      Sphere.onSphere a c₁ ∧ 
      Sphere.onSphere b c₁ ∧
      Sphere.onSphere a c₂ ∧
      Sphere.onSphere b c₃ ∧
      Sphere.diameter c₁ a b) →
    area_shaded = Sphere.shadedArea c₁ c₂ c₃ :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_between_circles_l1263_126383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_individual_id_l1263_126341

def RandomNumberTable : List (List Nat) :=
  [[7816, 6572, 0802, 6314, 0702, 4369, 9728, 0198],
   [3204, 9234, 4935, 8200, 3623, 4869, 6938, 7481]]

def SelectionMethod (table : List (List Nat)) : List Nat :=
  -- Implementation of the selection method
  sorry

theorem fifth_individual_id (table : List (List Nat)) :
  table = RandomNumberTable →
  (SelectionMethod table).get? 4 = some 1 := by
  sorry

#eval RandomNumberTable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_individual_id_l1263_126341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_beta_l1263_126366

theorem sin_alpha_beta (α β : ℝ) 
  (h1 : Real.sin (π/3 + α/6) = -3/5)
  (h2 : Real.cos (π/6 + β/2) = -12/13)
  (h3 : -5*π < α ∧ α < -2*π)
  (h4 : -π/3 < β ∧ β < 5*π/3) :
  Real.sin (α/6 + β/2) = 33/65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_beta_l1263_126366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twig_cutting_l1263_126344

/-- The expected length of the remaining piece after n cuts -/
noncomputable def expected_length (n : ℕ) : ℝ :=
  (11 / 18) ^ n

/-- The probability density function for the largest piece after one cut -/
noncomputable def p (x : ℝ) : ℝ :=
  if x < 1/3 then 0
  else if x ≤ 1/2 then 6 * (3*x - 1)
  else 6 * (1 - x)

theorem twig_cutting (n : ℕ) :
  let initial_length : ℝ := 1
  let num_cuts : ℕ := 2012
  ∀ x, 0 ≤ x → x ≤ 1 →
    (∫ x in Set.Icc 0 1, x * p x) = 11/18 →
    expected_length num_cuts = (11/18) ^ num_cuts :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twig_cutting_l1263_126344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_equivalence_l1263_126302

theorem sine_angle_equivalence (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hSum : A + B + C = Real.pi) : 
  (Real.sin A > Real.sin B) ↔ (A > B) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_equivalence_l1263_126302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_abc_roots_l1263_126303

-- Define the polynomial type
def MyPolynomial (α : Type) := List α

-- Define the polynomial P(x) = x^4 - ax^3 - bx + c
def P (a b c : ℝ) : MyPolynomial ℝ := [c, -b, 0, -a, 1]

-- Define what it means for a number to be a root of a polynomial
def isRoot (p : MyPolynomial ℝ) (r : ℝ) : Prop :=
  sorry -- We'll need to implement polynomial evaluation

-- State the theorem
theorem polynomial_with_abc_roots (a b c : ℝ) :
  (∃ d : ℝ, d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
    isRoot (P a b c) a ∧
    isRoot (P a b c) b ∧
    isRoot (P a b c) c ∧
    isRoot (P a b c) d) →
  P a b c = [0, 0, 0, -a, 1] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_abc_roots_l1263_126303
