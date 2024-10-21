import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_total_score_l1183_118303

/-- Represents the score of a batsman in cricket -/
structure BatsmanScore where
  total : ℕ      -- Total score
  boundaries : ℕ  -- Number of boundaries
  sixes : ℕ       -- Number of sixes
  score_eq : total = 4 * boundaries + 6 * sixes + total / 2

/-- Theorem: Given the conditions, the batsman's total score is 120 runs -/
theorem batsman_total_score (score : BatsmanScore) 
  (h1 : score.boundaries = 3)
  (h2 : score.sixes = 8) :
  score.total = 120 := by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_total_score_l1183_118303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1183_118395

/-- Given a line l passing through point M(p/2, 0) intersecting the parabola y^2 = 2px (p > 0) at points A and B -/
def intersecting_line (p : ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (A B : ℝ × ℝ), A ∈ l ∧ B ∈ l ∧ A.2^2 = 2*p*A.1 ∧ B.2^2 = 2*p*B.1 ∧ (p/2, 0) ∈ l

/-- The dot product of OA and OB is -3, where O is the origin -/
def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = -3

/-- The equation of line l that minimizes |AM| + 4|BM| -/
def minimizing_line_equation (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (k : ℝ), k = 1 ∨ k = -1 ∧ l = {(x, y) | 4*x + k*Real.sqrt 2*y - 4 = 0}

theorem parabola_intersection_theorem (p : ℝ) (l : Set (ℝ × ℝ)) 
  (h1 : p > 0) 
  (h2 : intersecting_line p l) 
  (h3 : ∃ (A B : ℝ × ℝ), A ∈ l ∧ B ∈ l ∧ dot_product_condition A B) :
  p = 2 ∧ minimizing_line_equation l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1183_118395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_inequality_l1183_118348

theorem root_difference_inequality (m n : ℕ) (h : m ≠ n) :
  |(n : ℝ)^(1/m) - (m : ℝ)^(1/n)| > 1 / (m * n : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_inequality_l1183_118348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l1183_118331

/-- Given points A, B, C, and O in ℝ², prove that λ = 2/3 -/
theorem lambda_value (A B C O : ℝ × ℝ) (lambda : ℝ) : 
  A = (-3, 0) →
  B = (0, 2) →
  O = (0, 0) →
  C.1 < 0 ∧ C.2 > 0 → -- C is inside ∠AOB
  (C.1 - O.1)^2 + (C.2 - O.2)^2 = 8 → -- |OC⃗| = 2√2
  (C.1 - O.1) = -(C.2 - O.2) → -- ∠AOC = π/4
  C = lambda • (A - O) + (B - O) →
  lambda = 2/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l1183_118331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_r_l1183_118343

/-- Given a triangle DEF with vertices D(3, 15), E(15, 0), and F(0, r), 
    if its area is 45, then r = 11.25 -/
theorem triangle_area_implies_r (r : ℝ) : 
  let D : ℝ × ℝ := (3, 15)
  let E : ℝ × ℝ := (15, 0)
  let F : ℝ × ℝ := (0, r)
  let area := abs ((D.1 * (E.2 - F.2) + E.1 * (F.2 - D.2) + F.1 * (D.2 - E.2)) / 2)
  area = 45 → r = 11.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_r_l1183_118343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonals_and_t_value_l1183_118327

/-- Given points A, B, and C in the Cartesian coordinate system -/
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (-2, -1)

/-- Vector from O to C -/
def OC : ℝ × ℝ := C

/-- Vector from A to B -/
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

/-- Vector from A to C -/
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

/-- Dot product of two 2D vectors -/
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Length of a 2D vector -/
noncomputable def length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem parallelogram_diagonals_and_t_value :
  (length (AB + AC) = 2 * Real.sqrt 10 ∧ 
   length (AB - AC) = 4 * Real.sqrt 2) ∧
  ∃ t : ℝ, t = -11/5 ∧ dot (AB - t • OC) OC = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonals_and_t_value_l1183_118327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_intersection_range_l1183_118372

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + Real.log x) / x

-- Define the function g
def g : ℝ → ℝ := λ x ↦ 1

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ x y : ℝ, 
    y = m * x + b ↔ 
    y - f 4 1 = (deriv (f 4)) 1 * (x - 1) :=
sorry

-- Theorem for the range of a
theorem intersection_range (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x ≤ Real.exp 2 ∧ f a x = g x) ↔ 
  a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_intersection_range_l1183_118372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_section_formula_l1183_118377

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 4:1,
    prove that Q = (1/5)C + (4/5)D -/
theorem vector_section_formula (C D Q : ℝ × ℝ × ℝ) 
    (h : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D) 
    (h_ratio : ∃ k : ℝ, k > 0 ∧ 4 • (Q - C) = k • (D - Q)) :
  Q = (1/5) • C + (4/5) • D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_section_formula_l1183_118377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l1183_118309

theorem triangle_existence 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : Real.sin A = Real.sqrt 3 * Real.sin B) 
  (h2 : C = π / 6) :
  (a * c = Real.sqrt 3 → c = 1) ∧ 
  (c * Real.sin A = 3 → c = 2 * Real.sqrt 3) ∧ 
  (c ≠ Real.sqrt 3 * b) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l1183_118309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1183_118368

open Real

theorem function_transformation (f : ℝ → ℝ) : 
  (∀ x, Real.sin (x - π/4) = f (2 * (x - π/3))) → 
  (∀ x, f x = Real.sin (x/2 + π/12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1183_118368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_determines_b_l1183_118386

noncomputable def vector : Type := ℝ × ℝ

def dot_product (v w : vector) : ℝ := v.1 * w.1 + v.2 * w.2

def norm_squared (v : vector) : ℝ := dot_product v v

noncomputable def projection (v w : vector) : vector :=
  let scalar := (dot_product v w) / (norm_squared w)
  (scalar * w.1, scalar * w.2)

theorem projection_determines_b (b : ℝ) :
  projection (-6, b) (3, 2) = (-20/13 * 3, -20/13 * 2) → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_determines_b_l1183_118386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_criterion_l1183_118396

/-- The floor function, returning the greatest integer less than or equal to a given real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence converges to a limit -/
def converges (s : Sequence) (l : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |s n - l| < ε

/-- A sequence is convergent if there exists a limit to which it converges -/
def is_convergent (s : Sequence) : Prop :=
  ∃ l, converges s l

/-- Main theorem: If a sequence of reals ≥ 1 has convergent floor(x^k) for all k, then the sequence converges -/
theorem convergence_criterion (x : Sequence) 
    (h1 : ∀ n, x n ≥ 1)
    (h2 : ∀ k : ℕ, is_convergent (λ n ↦ floor ((x n) ^ k))) :
    is_convergent x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_criterion_l1183_118396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_inequality_l1183_118357

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - x
noncomputable def g (x : ℝ) : ℝ := x * Real.log x

-- State the theorems
theorem f_max_value :
  ∃ (m : ℝ), m = 0 ∧ ∀ (x : ℝ), x > -1 → f x ≤ m :=
by sorry

theorem g_inequality {a b : ℝ} (h : 0 < a ∧ a < b) :
  0 < g a + g b - 2 * g ((a + b) / 2) ∧
  g a + g b - 2 * g ((a + b) / 2) < (b - a) * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_inequality_l1183_118357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_ratio_l1183_118375

/-- The ratio of distances for an ellipse with specific properties -/
theorem ellipse_distance_ratio : 
  ∀ (M N P F : ℝ × ℝ) (k : ℝ),
  -- Ellipse equation
  (λ (p : ℝ × ℝ) => p.1^2 / 25 + p.2^2 / 9 = 1) M →
  (λ (p : ℝ × ℝ) => p.1^2 / 25 + p.2^2 / 9 = 1) N →
  -- Right focus
  F = (4, 0) →
  -- Line equation (not vertical)
  k ≠ 0 →
  (λ (p : ℝ × ℝ) => p.2 = k * (p.1 - 4)) M →
  (λ (p : ℝ × ℝ) => p.2 = k * (p.1 - 4)) N →
  -- P is on x-axis and on perpendicular bisector of MN
  P.2 = 0 →
  (P.1 - (M.1 + N.1) / 2) * k = -((M.2 + N.2) / 2) →
  -- The ratio is 2/5
  dist P F / dist M N = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_ratio_l1183_118375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1183_118359

noncomputable def f (x : ℝ) := Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4)

theorem f_properties :
  let smallestPeriod := Real.pi
  let monotonicIntervals (k : ℤ) := Set.Icc (-3 * Real.pi / 8 + k * Real.pi) (Real.pi / 8 + k * Real.pi)
  let interval := Set.Icc (-Real.pi / 8) (Real.pi / 2)
  (∀ x, f (x + smallestPeriod) = f x) ∧
  (∀ k : ℤ, ∀ x ∈ monotonicIntervals k, ∀ y ∈ monotonicIntervals k, x ≤ y → f x ≤ f y) ∧
  (∀ x ∈ interval, f x ≤ 1) ∧
  (∀ x ∈ interval, f x ≥ -1) ∧
  (f (-Real.pi / 8) = 1) ∧
  (f (Real.pi / 8) = 1) ∧
  (f (Real.pi / 2) = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1183_118359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1183_118361

/-- Definition of the sequence sum -/
def S (n : ℕ) : ℤ := 32 * n - n^2 + 1

/-- Definition of the sequence terms -/
def a : ℕ → ℤ
  | 0 => 32  -- Added case for n = 0
  | 1 => 32
  | n + 2 => 33 - 2 * (n + 2)

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → S n - S (n - 1) = a n) ∧
  (∃ k : ℕ, k = 16 ∧ ∀ n : ℕ, n ≠ k → S n ≤ S k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1183_118361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l1183_118358

noncomputable section

/-- The polar equation of curve C: ρ²(1+2sin²θ) = 3 -/
def curve_C (ρ θ : ℝ) : Prop := ρ^2 * (1 + 2 * Real.sin θ^2) = 3

/-- The line l: y - x = 6 -/
def line_l (x y : ℝ) : Prop := y - x = 6

/-- A point P on curve C -/
def point_on_C (x y : ℝ) : Prop := ∃ (ρ θ : ℝ), curve_C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

/-- The distance from a point (x, y) to line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |y - x - 6| / Real.sqrt 2

theorem min_distance_to_line :
  ∀ (x y : ℝ), point_on_C x y → distance_to_line x y ≥ 2 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l1183_118358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angles_sum_l1183_118314

/-- Curve C₁ defined by parametric equations -/
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin α, Real.sqrt 2 * Real.cos α)

/-- Line l defined by polar equation -/
noncomputable def l (θ : ℝ) : ℝ := Real.sqrt 6 / (2 * Real.cos θ + Real.sin θ)

/-- Condition for intersection of C₁ and l -/
def intersection (α θ : ℝ) : Prop :=
  C₁ α = (l θ * Real.cos θ, l θ * Real.sin θ)

/-- Theorem statement -/
theorem intersection_angles_sum :
  ∃ (θ₁ θ₂ : ℝ), 
    0 ≤ θ₁ ∧ θ₁ < 2 * Real.pi ∧
    0 ≤ θ₂ ∧ θ₂ < 2 * Real.pi ∧
    (∃ (α₁ α₂ : ℝ), intersection α₁ θ₁ ∧ intersection α₂ θ₂) ∧
    θ₁ + θ₂ = 9 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angles_sum_l1183_118314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_DE_DB_ratio_l1183_118380

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
def right_angle_ABC : Prop := sorry
def AC_equals_5 : Prop := sorry
def BC_equals_5 : Prop := sorry
def right_angle_ABD : Prop := sorry
def AD_equals_15 : Prop := sorry
def C_D_opposite_sides : Prop := sorry
def D_parallel_AC : Prop := sorry
def DE_meets_CB_extended : Prop := sorry

-- State the theorem
theorem DE_DB_ratio (h1 : right_angle_ABC)
                     (h2 : AC_equals_5)
                     (h3 : BC_equals_5)
                     (h4 : right_angle_ABD)
                     (h5 : AD_equals_15)
                     (h6 : C_D_opposite_sides)
                     (h7 : D_parallel_AC)
                     (h8 : DE_meets_CB_extended) :
  ∃ (DE DB : ℝ), DE / DB = 1 / 1 :=
by
  sorry

#check DE_DB_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_DE_DB_ratio_l1183_118380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_xyz_l1183_118302

theorem sum_of_xyz (a b : ℝ) (x y z : ℕ+) : 
  a^2 = 9/25 ∧ 
  b^2 = (3 + Real.sqrt 7)^2 / 14 ∧ 
  a < 0 ∧ 
  b > 0 ∧ 
  (a - b)^2 = (x * Real.sqrt y) / z →
  x + y + z = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_xyz_l1183_118302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_duration_l1183_118351

/-- Proves that the second part of a loan was lent for 3 years given specific conditions --/
theorem loan_duration (total : ℝ) (first_rate second_rate : ℝ) (first_time : ℝ) (second_amount : ℝ) :
  total = 2665 →
  first_rate = 0.03 →
  second_rate = 0.05 →
  first_time = 8 →
  second_amount = 1640 →
  (total - second_amount) * first_rate * first_time = second_amount * second_rate * 3 →
  3 = (total - second_amount) * first_rate * first_time / (second_amount * second_rate) := by
  intros h1 h2 h3 h4 h5 h6
  sorry

#check loan_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_duration_l1183_118351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_finishes_first_l1183_118312

/-- Represents the relative size and mowing speed for a person's lawn -/
structure LawnInfo where
  size : ℚ
  mowingSpeed : ℚ

/-- Calculates the time needed to mow a lawn -/
def mowingTime (info : LawnInfo) : ℚ := info.size / info.mowingSpeed

theorem carlos_finishes_first (andy beth carlos : LawnInfo)
  (h1 : andy.size = 3 * beth.size)
  (h2 : andy.size = 4 * carlos.size)
  (h3 : carlos.mowingSpeed = (1/4) * beth.mowingSpeed)
  (h4 : carlos.mowingSpeed = (1/6) * andy.mowingSpeed) :
  mowingTime carlos < min (mowingTime andy) (mowingTime beth) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_finishes_first_l1183_118312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_section_point_sum_l1183_118382

open Real

/-- Given a triangle ABC with points D on BC, E on CA, and F on AB,
    if DC = 2BD, CE = 2EA, and AF = 2FB, then AD + BE + CF = -1/3 * BC -/
theorem section_point_sum (A B C D E F : ℝ × ℝ) : 
  D ∈ Set.Icc B C →
  E ∈ Set.Icc C A →
  F ∈ Set.Icc A B →
  C - D = (2 : ℝ) • (D - B) →
  C - E = (2 : ℝ) • (E - A) →
  A - F = (2 : ℝ) • (F - B) →
  (A - D) + (B - E) + (C - F) = (-1/3 : ℝ) • (B - C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_section_point_sum_l1183_118382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_at_pi_div_6_g_is_even_l1183_118338

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 12)

theorem f_zero_at_pi_div_6 : f (Real.pi / 6) = 0 := by sorry

theorem g_is_even : ∀ x : ℝ, g x = g (-x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_at_pi_div_6_g_is_even_l1183_118338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_l1183_118325

noncomputable def a (n : ℕ) : ℝ := 5 * (2/5)^(2*n - 2) - 4 * (2/5)^(n - 1)

def is_max_term (x : ℕ) : Prop :=
  ∀ n, a x ≥ a n

def is_min_term (y : ℕ) : Prop :=
  ∀ n, a y ≤ a n

theorem max_min_sum :
  ∃ (x y : ℕ), is_max_term x ∧ is_min_term y ∧ x + y = 3 :=
by
  -- We claim that x = 1 and y = 2
  use 1, 2
  constructor
  · -- Prove that 1 is the max term
    intro n
    sorry
  constructor
  · -- Prove that 2 is the min term
    intro n
    sorry
  · -- Prove that 1 + 2 = 3
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_l1183_118325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1183_118347

/-- The time it takes for two people to complete a task together, given their individual rates -/
noncomputable def time_to_complete_together (time_a_half : ℝ) (time_b_third : ℝ) : ℝ :=
  1 / (1 / (2 * time_a_half) + 1 / (3 * time_b_third))

/-- Theorem stating that if A can do half the work in 70 days and B can do one-third in 35 days,
    then together they can complete the whole work in 60 days -/
theorem work_completion_time (time_a_half : ℝ) (time_b_third : ℝ)
    (h1 : time_a_half = 70)
    (h2 : time_b_third = 35) :
    time_to_complete_together time_a_half time_b_third = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1183_118347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_removed_approx_four_l1183_118384

noncomputable section

-- Define the side length of the square
def square_side : ℝ := 5

-- Define the side length of the octagon in terms of x
def octagon_side (x : ℝ) : ℝ := square_side - 2 * x

-- Define the area of one triangular piece
def triangle_area (x : ℝ) : ℝ := (1 / 2) * x^2

-- Define the total area removed
def total_area_removed (x : ℝ) : ℝ := 4 * triangle_area x

-- Theorem statement
theorem area_removed_approx_four :
  ∃ x : ℝ, 
    x * Real.sqrt 2 = octagon_side x ∧ 
    Int.floor (total_area_removed x + 0.5) = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_removed_approx_four_l1183_118384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l1183_118339

noncomputable def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let a := v.1
  let b := v.2
  let denominator := a^2 + b^2
  ![![a^2 / denominator, a * b / denominator],
    ![a * b / denominator, b^2 / denominator]]

theorem det_projection_matrix_zero :
  let v : ℝ × ℝ := (3, -2)
  let P := projection_matrix v
  Matrix.det P = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l1183_118339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_negative_pi_over_six_l1183_118362

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

-- Define symmetry about a point
def symmetric_about_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (p.1 - x) = f (p.1 + x)

-- Theorem statement
theorem f_symmetric_about_negative_pi_over_six :
  symmetric_about_point f (-Real.pi / 6, 0) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_negative_pi_over_six_l1183_118362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_X_between_neg2_and_2_l1183_118307

noncomputable section

-- Define X as a measure space
variable (X : Type) [MeasurableSpace X]

-- Define the normal distribution
def normal_distribution (μ σ : ℝ) (X : Type) [MeasurableSpace X] : Prop := sorry

-- Define the probability measure
variable (P : Measure X)

-- Axiom for X following a normal distribution with mean 0 and some standard deviation σ
axiom X_normal : ∃ (σ : ℝ), normal_distribution 0 σ X

-- Axiom for the given probability
axiom P_X_gt_2 : P {x : X | x > (2 : ℝ)} = 0.023

-- Theorem to prove
theorem P_X_between_neg2_and_2 :
  P {x : X | -2 ≤ (x : ℝ) ∧ (x : ℝ) ≤ 2} = 0.954 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_X_between_neg2_and_2_l1183_118307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l1183_118379

/-- Represents a domino with two numbers -/
structure Domino :=
  (left : ℕ)
  (right : ℕ)

/-- Represents the state of the game -/
structure GameState :=
  (chain : List Domino)
  (unused : List Domino)

/-- Checks if a domino can be placed at either end of the chain -/
def canPlace (d : Domino) (gs : GameState) : Bool :=
  match gs.chain with
  | [] => true
  | h::t => 
    let firstDomino := h
    let lastDomino := gs.chain.getLast (by sorry)  -- Using getLast with a sorry proof
    d.left = firstDomino.left ∨ d.right = firstDomino.left ∨ 
    d.left = lastDomino.right ∨ d.right = lastDomino.right

/-- Represents a player's strategy -/
def Strategy := GameState → Option Domino

/-- Represents the result of the game -/
inductive GameResult
| FirstPlayerWins
| SecondPlayerWins

/-- The main theorem stating that the first player has a winning strategy -/
theorem first_player_winning_strategy :
  ∃ (s : Strategy), ∀ (initial_state : GameState),
    (initial_state.unused.length = 28) →  -- Complete set of dominoes
    (∃ (game_result : GameResult), 
      game_result = GameResult.FirstPlayerWins) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l1183_118379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_quadrilateral_angle_sum_l1183_118371

theorem parallelogram_quadrilateral_angle_sum :
  ∀ (p1 p2 q1 q2 q3 q4 : ℝ),
  -- Parallelogram conditions
  p1 + p2 = 180 ∧
  p1 / p2 = 4 / 11 ∧
  -- Quadrilateral conditions
  q1 + q2 + q3 + q4 = 360 ∧
  q1 / q2 = 5 / 6 ∧
  q2 / q3 = 6 / 7 ∧
  q3 / q4 = 7 / 12 →
  -- Conclusion
  min p1 p2 + (max q2 q3) = 132 :=
by
  intros p1 p2 q1 q2 q3 q4 h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_quadrilateral_angle_sum_l1183_118371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_problem_l1183_118383

def parallel_vectors (v₁ v₂ : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

theorem parallel_vectors_problem (a : ℝ) 
  (h1 : a ∈ Set.Icc (π / 2) π)
  (h2 : parallel_vectors (2, -1) (Real.cos a, Real.sin a)) :
  ∃ α : ℝ, 
    (Real.tan (α + π / 4) = 1 / 3) ∧ 
    (Real.cos (5 * π / 6 - 2 * α) = -(4 + 3 * Real.sqrt 3) / 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_problem_l1183_118383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_composition_is_translation_l1183_118333

-- Define the plane as a real inner product space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A and B in the plane
variable (A B : V)

-- Define the symmetry operation
def symmetry (center : V) (point : V) : V :=
  2 • center - point

-- Define the composition of symmetries
def composedSymmetry (A B : V) (M : V) : V :=
  symmetry B (symmetry A M)

-- Theorem statement
theorem symmetry_composition_is_translation (M : V) :
  composedSymmetry A B M = M + (2 : ℝ) • (B - A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_composition_is_translation_l1183_118333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_l1183_118349

theorem theta_range (θ : Real) : 
  (∀ x : Real, (Real.cos θ) * x^2 - (4 * Real.sin θ) * x + 6 > 0) →
  (0 < θ ∧ θ < Real.pi) →
  (0 < θ ∧ θ < Real.pi/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_l1183_118349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jackson_calorie_theorem_l1183_118311

def salad_calories : ℚ := 50 + 2*50 + 30 + 60 + 210

def pizza_calories : ℚ := 600 + (1/3)*600 + (1/4)*600 + 400

def jackson_calorie_intake : ℚ := (3/8) * salad_calories + (2/7) * pizza_calories

theorem jackson_calorie_theorem : 
  ⌊jackson_calorie_intake⌋ = 554 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jackson_calorie_theorem_l1183_118311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_l1183_118353

theorem fermats_little_theorem (p : ℕ) (a : ℤ) (h : Nat.Prime p) : 
  (p : ℤ) ∣ (a^p - a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_l1183_118353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_sum_zero_l1183_118320

theorem three_sum_zero (S : Finset ℤ) (h1 : S.card = 101) (h2 : ∀ x ∈ S, |x| ≤ 99) :
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_sum_zero_l1183_118320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_length_l1183_118356

/-- Calculates the length of a goods train given its speed, platform length, and time to cross the platform. -/
theorem goods_train_length
  (speed_kmph : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (h1 : speed_kmph = 72)
  (h2 : platform_length = 250)
  (h3 : crossing_time = 26) :
  speed_kmph * (1000 / 3600) * crossing_time - platform_length = 270 := by
  sorry

#check goods_train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_length_l1183_118356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_withheld_is_twenty_percent_l1183_118378

/-- Calculates the percentage of pay withheld given job details and take-home pay -/
noncomputable def percentage_withheld (hourly_wage : ℚ) (hours_per_week : ℚ) (weeks_per_year : ℚ) (annual_take_home : ℚ) : ℚ :=
  let annual_gross := hourly_wage * hours_per_week * weeks_per_year
  let amount_withheld := annual_gross - annual_take_home
  (amount_withheld / annual_gross) * 100

/-- Theorem stating that the percentage of pay withheld is 20% given specific job details -/
theorem percentage_withheld_is_twenty_percent :
  percentage_withheld 15 40 52 24960 = 20 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval percentage_withheld 15 40 52 24960

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_withheld_is_twenty_percent_l1183_118378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_angles_correct_l1183_118366

/-- The set of angles in the third quadrant -/
def third_quadrant_angles : Set ℝ :=
  {α | ∃ k : ℤ, Real.pi + 2 * k * Real.pi < α ∧ α < 3 * Real.pi / 2 + 2 * k * Real.pi}

/-- Theorem: The set of angles in the third quadrant is correctly defined -/
theorem third_quadrant_angles_correct :
  third_quadrant_angles = {α | ∃ k : ℤ, Real.pi + 2 * k * Real.pi < α ∧ α < 3 * Real.pi / 2 + 2 * k * Real.pi} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_angles_correct_l1183_118366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CaCO3_approx_l1183_118346

noncomputable section

/-- Molar mass of Calcium in g/mol -/
def molar_mass_Ca : ℝ := 40.08

/-- Molar mass of Carbon in g/mol -/
def molar_mass_C : ℝ := 12.01

/-- Molar mass of Oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Number of Calcium atoms in CaCO3 -/
def num_Ca : ℕ := 1

/-- Number of Carbon atoms in CaCO3 -/
def num_C : ℕ := 1

/-- Number of Oxygen atoms in CaCO3 -/
def num_O : ℕ := 3

/-- Molar mass of CaCO3 in g/mol -/
noncomputable def molar_mass_CaCO3 : ℝ := molar_mass_Ca * num_Ca + molar_mass_C * num_C + molar_mass_O * num_O

/-- Mass of Oxygen in CaCO3 in g/mol -/
noncomputable def mass_O_in_CaCO3 : ℝ := molar_mass_O * num_O

/-- Mass percentage of Oxygen in CaCO3 -/
noncomputable def mass_percentage_O_in_CaCO3 : ℝ := (mass_O_in_CaCO3 / molar_mass_CaCO3) * 100

theorem mass_percentage_O_in_CaCO3_approx :
  abs (mass_percentage_O_in_CaCO3 - 47.95) < 0.01 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CaCO3_approx_l1183_118346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_point_l1183_118389

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + Real.cos (2 * x)

-- State the theorem
theorem max_value_point :
  ∃ ε > 0, ∀ x, 0 < x ∧ x < π ∧ |x - π/12| < ε → f x < f (π/12) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_point_l1183_118389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1183_118326

/-- The parabola y^2 = -8x with focus F -/
structure Parabola where
  F : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Point A on the parabola -/
structure PointOnParabola (p : Parabola) where
  A : ℝ × ℝ
  on_parabola : p.equation A.1 A.2

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_focus_distance 
  (p : Parabola) 
  (A : PointOnParabola p) 
  (h1 : p.F = (-2, 0)) 
  (h2 : A.A.1 = -6) 
  (h3 : p.equation = fun x y => y^2 = -8*x) :
  distance A.A p.F = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1183_118326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_theorem_l1183_118342

structure CircleConfiguration where
  A : ℝ  -- radius of circle A
  B : ℝ  -- radius of circle B
  C : ℝ  -- radius of circle C
  D : ℝ  -- radius of circle D
  E : ℚ  -- radius of circle E

def is_valid_configuration (config : CircleConfiguration) : Prop :=
  config.A = 10 ∧
  config.B = 4 ∧
  config.C = 2 ∧
  config.D = 2 ∧
  ∃ (p q : ℕ), config.E = p / q ∧ Nat.Coprime p q

-- Function to calculate the radius of circle E
def calculate_E_radius : ℚ :=
  7 / 5

theorem circle_configuration_theorem (config : CircleConfiguration) :
  is_valid_configuration config →
  calculate_E_radius = config.E :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_theorem_l1183_118342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_percentage_in_x_l1183_118376

/-- Represents a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
structure FinalMixture where
  x : ℝ
  y : ℝ
  ryegrass : ℝ

/-- Proves that the percentage of ryegrass in seed mixture X is approximately 40% -/
theorem ryegrass_percentage_in_x (x : SeedMixture) (y : SeedMixture) (final : FinalMixture) : 
  x.bluegrass = 60 →
  y.ryegrass = 25 →
  y.fescue = 75 →
  final.x = 46.67 →
  final.y = 100 - final.x →
  final.ryegrass = 32 →
  x.ryegrass + x.bluegrass + x.fescue = 100 →
  y.ryegrass + y.bluegrass + y.fescue = 100 →
  final.x * x.ryegrass / 100 + final.y * y.ryegrass / 100 = final.ryegrass →
  abs (x.ryegrass - 40) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryegrass_percentage_in_x_l1183_118376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l1183_118341

noncomputable def vector_projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1 * w.1 + w.2 * w.2
  ((dot_product / magnitude_squared) * w.1, (dot_product / magnitude_squared) * w.2)

theorem projection_problem (u : ℝ × ℝ) 
  (h : vector_projection (2, -2) u = (1/2, -1/2)) :
  vector_projection (5, 3) u = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l1183_118341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_l1183_118398

theorem salary_change (S : ℝ) (h : S > 0) : 
  let increased_salary := S * 1.25
  let final_salary := increased_salary * 0.75
  final_salary = S * 0.9375 := by
  sorry

#check salary_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_l1183_118398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1183_118367

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 3)

-- State the theorem
theorem min_value_of_f :
  ∀ x > 3, f x ≥ 5 ∧ ∃ x₀ > 3, f x₀ = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1183_118367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_8_and_9_l1183_118354

/-- The combined resistance of two resistors in parallel -/
noncomputable def combined_resistance (r1 r2 : ℝ) : ℝ := 1 / (1 / r1 + 1 / r2)

/-- Theorem: The combined resistance of two resistors with 8 ohms and 9 ohms in parallel is 72/17 ohms -/
theorem parallel_resistors_8_and_9 :
  combined_resistance 8 9 = 72 / 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_8_and_9_l1183_118354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_permutations_l1183_118329

theorem race_permutations (n : Nat) (h : n = 4) : n.factorial = 24 := by
  rw [h]
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_permutations_l1183_118329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l1183_118388

-- Define sets A and B
def A : Set ℝ := {x | x > 1/3}
def B : Set ℝ := {y | -2 ≤ y ∧ y ≤ 2}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_interval :
  A_intersect_B = Set.Ioc (1/3) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l1183_118388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_midpoint_theorem_l1183_118323

/-- Represents a trapezoid with bases of length a and b -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  a_ge_b : a ≥ b
  h_pos : 0 < h

/-- The area of the trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ := (t.a + t.b) * t.h / 2

/-- The area of the quadrilateral formed by midpoints -/
noncomputable def midpoint_quad_area (t : Trapezoid) : ℝ := (t.a - t.b) * t.h / 4

/-- The theorem statement -/
theorem trapezoid_midpoint_theorem (t : Trapezoid) :
  midpoint_quad_area t = trapezoid_area t / 4 → t.a = 3 * t.b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_midpoint_theorem_l1183_118323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_28_equals_reciprocal_of_one_minus_x_l1183_118316

-- Define the linear fractional transformation f₁
noncomputable def f₁ (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

-- Define the recursive function f_n
noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- Base case for n = 0
  | 1 => f₁
  | n + 1 => λ x => f₁ (f_n n x)

-- State the theorem
theorem f_28_equals_reciprocal_of_one_minus_x 
  (h : ∀ x, f_n 35 x = f_n 5 x) :
  ∀ x, f_n 28 x = 1 / (1 - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_28_equals_reciprocal_of_one_minus_x_l1183_118316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_signs_l1183_118374

/-- The distance between two signs given total distance rode, distance to first sign, and distance after second sign -/
theorem distance_between_signs
  (total_distance : ℝ)
  (distance_to_first_sign : ℝ)
  (distance_after_second_sign : ℝ)
  (h1 : total_distance = 1000)
  (h2 : distance_to_first_sign = 350)
  (h3 : distance_after_second_sign = 275) :
  total_distance - distance_after_second_sign - distance_to_first_sign = 375 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_signs_l1183_118374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_problem_l1183_118313

-- Define the ellipses
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define C₂ using a parameter for its equation
def C₂ (a b : ℝ) (x y : ℝ) : Prop := y^2 / a^2 + x^2 / b^2 = 1

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define a line passing through the origin
def line_through_origin (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Theorem statement
theorem ellipse_and_line_problem 
  (a b : ℝ) 
  (h_c₂_eq : ∀ x y, C₂ a b x y ↔ y^2 / 16 + x^2 / 4 = 1) 
  (h_same_ecc : eccentricity 2 1 = eccentricity a b) 
  (h_axis : a = 2 ∧ b = 4) 
  (A : Point) (B : Point)
  (h_A_on_C₁ : C₁ A.x A.y)
  (h_B_on_C₂ : C₂ a b B.x B.y)
  (h_OB_2OA : B.x = 2 * A.x ∧ B.y = 2 * A.y) :
  (∃ k, k = 1 ∨ k = -1) ∧ line_through_origin k A.x A.y ∧ line_through_origin k B.x B.y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_problem_l1183_118313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_f_decreasing_l1183_118306

-- Define the function f(x) = 2/x - x^m
noncomputable def f (x m : ℝ) : ℝ := 2 / x - x ^ m

-- Theorem 1: Given f(4) = -7/2, prove m = 1
theorem find_m : ∃ m : ℝ, f 4 m = -7/2 ∧ m = 1 := by sorry

-- Define the specific function f(x) = 2/x - x (with m = 1)
noncomputable def f_specific (x : ℝ) : ℝ := 2 / x - x

-- Theorem 2: Prove f(x) = 2/x - x is monotonically decreasing on (0, +∞)
theorem f_decreasing : ∀ x₁ x₂ : ℝ, 0 < x₂ → x₂ < x₁ → f_specific x₁ < f_specific x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_f_decreasing_l1183_118306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1183_118363

/-- Triangle ABC with vertices A(4,1), B(1,5), and C(-3,2) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle ABC -/
def triangle_ABC : Triangle :=
  { A := (4, 1),
    B := (1, 5),
    C := (-3, 2) }

/-- Equation of a line in general form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Equation of a circle (x - h)² + (y - k)² = r² -/
structure CircleEquation where
  h : ℝ
  k : ℝ
  r_squared : ℝ

/-- Main theorem about the properties of triangle ABC -/
theorem triangle_ABC_properties :
  let t := triangle_ABC
  ∃ (line_AB : LineEquation) (circle : CircleEquation),
    (line_AB.a = 4 ∧ line_AB.b = 3 ∧ line_AB.c = -19) ∧
    (∃ (AB BC : ℝ × ℝ), 
      (AB.1 * BC.1 + AB.2 * BC.2 = 0) ∧
      (AB = (t.B.1 - t.A.1, t.B.2 - t.A.2)) ∧
      (BC = (t.C.1 - t.B.1, t.C.2 - t.B.2))) ∧
    (circle.h = 1/2 ∧ circle.k = 3/2 ∧ circle.r_squared = 25/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1183_118363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_gcd_lcm_sum_l1183_118305

theorem polynomial_gcd_lcm_sum (a b c : ℤ) : 
  (∃ (p q : Polynomial ℤ), 
    (X^2 + a • X + b) = (X + 1) * p ∧ 
    (X^2 + b • X + c) = (X + 1) * q ∧
    (X + 1) * p * q = X^3 - 5 • X^2 + 7 • X - 3) →
  a + b + c = -8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_gcd_lcm_sum_l1183_118305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_theta_for_odd_g_min_theta_is_pi_l1183_118381

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 - Real.cos x ^ 2

/-- The shifted function g(x) with parameter θ -/
noncomputable def g (θ : ℝ) (x : ℝ) : ℝ := f (x - θ)

/-- A function is odd if f(-x) = -f(x) for all x -/
def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

/-- The theorem stating that π is the minimum positive θ for which g is odd -/
theorem min_theta_for_odd_g :
  ∀ θ > 0, is_odd (g θ) ↔ θ ≥ π ∧ ∃ n : ℕ, θ = π * n := by
  sorry

/-- The main theorem proving π is the minimum positive θ for which g is odd -/
theorem min_theta_is_pi :
  ∃! θ : ℝ, θ > 0 ∧ is_odd (g θ) ∧ ∀ φ > 0, is_odd (g φ) → θ ≤ φ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_theta_for_odd_g_min_theta_is_pi_l1183_118381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l1183_118364

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3/4) : 
  Real.tan (3 * θ) = -137/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_triple_angle_l1183_118364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_n_roots_distinct_real_l1183_118340

/-- Definition of the polynomial sequence P_n -/
def P : ℕ → (ℝ → ℝ)
  | 0 => λ x => x^2 - 2
  | n + 1 => λ x => P 0 (P n x)

/-- Theorem stating that P_n(x) = x has 2^n distinct real roots -/
theorem P_n_roots_distinct_real (n : ℕ) : 
  ∃ (S : Finset ℝ), (∀ x, x ∈ S → P n x = x) ∧ (Finset.card S = 2^n) ∧ (∀ x y, x ∈ S → y ∈ S → x ≠ y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_n_roots_distinct_real_l1183_118340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_sin_2theta_l1183_118321

theorem perpendicular_vectors_sin_2theta (θ : ℝ) : 
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (1, -1)
  (a.1 * b.1 + a.2 * b.2 = 0) → Real.sin (2 * θ) = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_sin_2theta_l1183_118321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_l1183_118301

/-- The focus of a parabola y^2 = 2px -/
noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

/-- The right focus of a hyperbola x^2/a^2 - y^2/b^2 = 1 -/
noncomputable def hyperbola_right_focus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)

/-- The statement to prove -/
theorem parabola_hyperbola_focus (p : ℝ) :
  parabola_focus p = hyperbola_right_focus 3 1 → p = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_l1183_118301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l1183_118399

def is_valid_tuple (n : ℕ) (tuple : Fin n → ℚ) : Prop :=
  (∀ i, tuple i > 0) ∧ 
  (Finset.sum Finset.univ (λ i => tuple i)).isInt ∧
  (Finset.sum Finset.univ (λ i => 1 / tuple i)).isInt

def has_infinitely_many_tuples (n : ℕ) : Prop :=
  ∀ k : ℕ, ∃ (tuples : Finset (Fin n → ℚ)),
    tuples.card > k ∧ (∀ tuple ∈ tuples, is_valid_tuple n tuple) ∧
    (∀ tuple₁ tuple₂, tuple₁ ∈ tuples → tuple₂ ∈ tuples → tuple₁ ≠ tuple₂ → ∃ i, tuple₁ i ≠ tuple₂ i)

theorem smallest_valid_n :
  (has_infinitely_many_tuples 3) ∧
  (∀ n < 3, ¬(has_infinitely_many_tuples n)) := by
  sorry

#check smallest_valid_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l1183_118399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girl_ratio_l1183_118360

/-- Represents a class of students -/
structure StudentClass where
  boys : ℕ
  girls : ℕ
  total : ℕ
  h_total : total = boys + girls
  h_size : total > 100

/-- The probability of choosing a student from a given group -/
def prob (n : ℕ) (c : StudentClass) : ℚ :=
  n / c.total

theorem girl_ratio (c : StudentClass) 
  (h : prob c.boys c = 3/4 * prob c.girls c) :
  prob c.girls c = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girl_ratio_l1183_118360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boundary_line_proof_unique_boundary_line_l1183_118370

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * (x - 1 / x)

-- Define the boundary line
def boundary_line (x : ℝ) : ℝ := x - 1

-- State the theorem
theorem boundary_line_proof (x : ℝ) (h : x ≥ 1) :
  f x ≥ boundary_line x ∧ boundary_line x ≥ g x := by
  sorry

-- Prove that this is the unique boundary line
theorem unique_boundary_line (k b : ℝ) (h : ∀ x ≥ 1, f x ≥ k * x + b ∧ k * x + b ≥ g x) :
  k = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boundary_line_proof_unique_boundary_line_l1183_118370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_color_theorem_l1183_118317

-- Define a type for countries
structure Country where
  vertices : Fin 3 → ℝ × ℝ

-- Define a type for islands
structure Island where
  countries : Finset Country
  adjacent : Country → Country → Prop
  adjacent_sym : ∀ c1 c2, adjacent c1 c2 → adjacent c2 c1
  adjacent_entire_side : ∀ c1 c2, adjacent c1 c2 → ∃ (i j : Fin 3), c1.vertices i = c2.vertices j ∧ c1.vertices (i + 1) = c2.vertices (j + 1)

-- Define a type for colorings
def Coloring (i : Island) := { c : Country // c ∈ i.countries } → Fin 3

-- The main theorem
theorem three_color_theorem (i : Island) :
  ∃ (coloring : Coloring i), ∀ (c1 c2 : { c : Country // c ∈ i.countries }), i.adjacent c1.val c2.val → coloring c1 ≠ coloring c2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_color_theorem_l1183_118317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_is_correct_l1183_118332

def P : Fin 2 → ℚ := ![1, 1]
def Q : Fin 2 → ℚ := ![2, 2]
def R : Fin 2 → ℚ := ![2, 1]

def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ := !![0, 1; 1, 0]
def scaling_factor : ℚ := 2

noncomputable def transformation_matrix : Matrix (Fin 2) (Fin 2) ℚ := scaling_factor • reflection_matrix

theorem transformation_is_correct :
  transformation_matrix = !![0, 2; 2, 0] ∧
  transformation_matrix.mulVec P = ![2, 2] ∧
  transformation_matrix.mulVec Q = ![4, 4] ∧
  transformation_matrix.mulVec R = ![2, 4] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_is_correct_l1183_118332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l1183_118397

noncomputable def f (x : ℝ) (φ : ℝ) := -2 * Real.sin (2 * x + φ)

theorem monotone_decreasing_interval 
  (φ : ℝ) 
  (h1 : |φ| < π) 
  (h2 : f (π/8) φ = -2) :
  ∀ x y, π/8 ≤ x ∧ x < y ∧ y ≤ 5*π/8 → f x φ > f y φ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l1183_118397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_ones_l1183_118336

-- Define Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define Fibonacci representation
def is_fib_rep (k : ℕ) (rep : List ℕ) : Prop :=
  k = (rep.map fib).sum ∧ rep.all (λ i => i ≥ 2) ∧ rep.Nodup

-- Define minimal Fibonacci representation
def is_minimal_fib_rep (k : ℕ) (rep : List ℕ) : Prop :=
  is_fib_rep k rep ∧ ∀ rep', is_fib_rep k rep' → rep'.length ≥ rep.length

-- Theorem statement
theorem smallest_with_eight_ones :
  ∃ (rep : List ℕ),
    is_minimal_fib_rep 1596 rep ∧
    rep.length = 8 ∧
    (∀ k < 1596, ∀ rep', is_minimal_fib_rep k rep' → rep'.length < 8) := by
  sorry

#check smallest_with_eight_ones

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_ones_l1183_118336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_best_fit_line_l1183_118304

/-- Slope of the line of best fit for four equally spaced points -/
theorem slope_of_best_fit_line (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ d : ℝ) 
  (h₁ : x₁ < x₂) (h₂ : x₂ < x₃) (h₃ : x₃ < x₄)
  (h₄ : x₂ - x₁ = d) (h₅ : x₃ - x₂ = d) (h₆ : x₄ - x₃ = d) :
  let points := [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)]
  let slope := (y₄ - y₁) / (x₄ - x₁)
  slope = (y₄ - y₁) / (x₄ - x₁) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_best_fit_line_l1183_118304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_power_theorem_l1183_118330

/-- Represents a circle with center O -/
structure Circle where
  O : Point

/-- Represents a point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The power of a point with respect to a circle -/
noncomputable def power (p : Point) (c : Circle) : ℝ := sorry

/-- Checks if a point is external to a circle -/
def is_external (p : Point) (c : Circle) : Prop := sorry

/-- Represents a line segment between two points -/
structure LineSegment where
  p : Point
  q : Point

/-- Checks if two line segments intersect -/
def intersect (l1 l2 : LineSegment) : Prop := sorry

/-- The squared distance between two points -/
noncomputable def dist_squared (p q : Point) : ℝ := sorry

theorem secant_power_theorem (O : Circle) (P A B C D Q R : Point) :
  is_external P O →
  intersect ⟨A, D⟩ ⟨B, C⟩ →
  intersect ⟨B, D⟩ ⟨A, C⟩ →
  (dist_squared P Q = power P O + power Q O) ∧
  (dist_squared P R = power P O + power R O) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_power_theorem_l1183_118330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_120_degrees_l1183_118319

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = Real.pi)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Define the sine ratio condition
def sine_ratio (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ Real.sin t.A = k ∧ Real.sin t.B = k ∧ Real.sin t.C = k * Real.sqrt 3

-- Theorem statement
theorem largest_angle_is_120_degrees (t : Triangle) (h : sine_ratio t) :
  max t.A (max t.B t.C) = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_120_degrees_l1183_118319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_squares_l1183_118322

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define the circumradius of a triangle
noncomputable def circumradius (t : Triangle) : ℝ := sorry

-- Define a point on the circumcircle
structure CircumcirclePoint (t : Triangle) where
  P : ℝ × ℝ
  on_circumcircle : sorry

-- Define the theorem
theorem constant_sum_squares (t : Triangle) (p : CircumcirclePoint t) :
  let G := centroid t
  let R := circumradius t
  let PA := Real.sqrt ((p.P.1 - t.A.1)^2 + (p.P.2 - t.A.2)^2)
  let PB := Real.sqrt ((p.P.1 - t.B.1)^2 + (p.P.2 - t.B.2)^2)
  let PC := Real.sqrt ((p.P.1 - t.C.1)^2 + (p.P.2 - t.C.2)^2)
  let PG := Real.sqrt ((p.P.1 - G.1)^2 + (p.P.2 - G.2)^2)
  PA^2 + PB^2 + PC^2 - PG^2 = 14 * R^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_squares_l1183_118322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_inscribed_triangle_l1183_118369

/-- Predicate stating that triangle ABC is isosceles -/
def IsoscelesTriangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate stating that α is the base angle of triangle ABC -/
def BaseAngle (A B C : ℝ × ℝ) (α : ℝ) : Prop := sorry

/-- Predicate stating that there's an inscribed circle in triangle ABC
    touching the sides at points A₁, B₁, C₁ -/
def InscribedCircle (A B C A₁ B₁ C₁ : ℝ × ℝ) : Prop := sorry

/-- Function to construct a triangle from three points -/
def Triangle (A B C : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Function to calculate the area ratio of two triangles -/
noncomputable def AreaRatio (T₁ T₂ : Set (ℝ × ℝ)) : ℝ := sorry

/-- Given an isosceles triangle ABC with base angle α and an inscribed circle
    touching the sides at points A₁, B₁, C₁, the ratio of the area of triangle A₁B₁C₁
    to the area of triangle ABC is cos α (1 - cos α). -/
theorem area_ratio_inscribed_triangle (A B C A₁ B₁ C₁ : ℝ × ℝ) (α : ℝ) :
  IsoscelesTriangle A B C →
  BaseAngle A B C α →
  InscribedCircle A B C A₁ B₁ C₁ →
  AreaRatio (Triangle A₁ B₁ C₁) (Triangle A B C) = Real.cos α * (1 - Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_inscribed_triangle_l1183_118369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1183_118345

/-- Definition of the ellipse C -/
def ellipse (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of isosceles right triangle condition -/
def isoscelesRightTriangle (a b : ℝ) : Prop :=
  a^2 = 2 * b^2

/-- Definition of the chord length condition -/
def chordLengthCondition (a b : ℝ) : Prop :=
  a^2 / 2 + 3 = 4 * b^2

/-- Main theorem -/
theorem ellipse_properties (a b : ℝ) :
  (∃ x y, ellipse a b x y) ∧ 
  isoscelesRightTriangle a b ∧ 
  chordLengthCondition a b →
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∃ k : ℝ, k^2 = 2 ∧ 
    (∀ x y : ℝ, y = k * (x - 1) → 
      (x^2 / 2 + y^2 = 1 → 
        ∃ x' y', x'^2 / 2 + y'^2 = 1 ∧ 
        y' = k * (x' - 1) ∧ 
        x * x' + y * y' = 0))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1183_118345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_difference_l1183_118350

/-- The parabola y^2 = 4x with focus F, intersected by a line through F with slope √3 at points A and B -/
structure ParabolaIntersection where
  -- The parabola y^2 = 4x
  parabola : ℝ → ℝ → Prop
  parabola_eq : parabola = fun x y ↦ y^2 = 4*x

  -- The focus F of the parabola
  F : ℝ × ℝ
  focus_prop : F.1 = 1 ∧ F.2 = 0

  -- The line y = √3(x - 1) passing through F
  line : ℝ → ℝ → Prop
  line_eq : line = fun x y ↦ y = Real.sqrt 3 * (x - 1)

  -- Points A and B where the line intersects the parabola
  A : ℝ × ℝ
  B : ℝ × ℝ
  A_on_parabola : parabola A.1 A.2
  A_on_line : line A.1 A.2
  B_on_parabola : parabola B.1 B.2
  B_on_line : line B.1 B.2
  A_ne_B : A ≠ B

/-- The theorem stating that ||FA|^2 - |FB|^2| = 128/9 -/
theorem parabola_intersection_distance_difference (p : ParabolaIntersection) :
  |((p.A.1 - p.F.1)^2 + (p.A.2 - p.F.2)^2) - ((p.B.1 - p.F.1)^2 + (p.B.2 - p.F.2)^2)| = 128/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_difference_l1183_118350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_k_value_l1183_118335

theorem quadratic_function_k_value (a b c k : ℤ) (g : ℤ → ℤ) : 
  (∀ x, g x = a * x^2 + b * x + c) →
  g 2 = 0 →
  20 < g 10 ∧ g 10 < 30 →
  30 < g 11 ∧ g 11 < 40 →
  7000 * k < g 200 ∧ g 200 < 7000 * (k + 1) →
  k = 5 := by
  sorry

-- Define g separately
def g (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_k_value_l1183_118335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_inner_square_l1183_118344

/-- A square with side length 4 -/
def square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 4}

/-- The inner square with points more than 1 unit away from all sides -/
def inner_square : Set (ℝ × ℝ) :=
  {p | 1 < p.1 ∧ p.1 < 3 ∧ 1 < p.2 ∧ p.2 < 3}

/-- The probability of selecting a point in the inner square -/
theorem probability_inner_square :
  (MeasureTheory.volume inner_square) / (MeasureTheory.volume square) = 1/4 := by
  sorry

#check probability_inner_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_inner_square_l1183_118344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_pills_l1183_118355

/-- The sum of an arithmetic sequence -/
def arithmetic_sequence_sum (a₁ d n : ℕ) : ℕ := 
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Joey's pill calculation -/
theorem joey_pills (weeks : ℕ) (a₁ b₁ c₁ da db dc : ℕ) 
  (ha : a₁ = 2) (hb : b₁ = 3) (hc : c₁ = 4)
  (hda : da = 1) (hdb : db = 2) (hdc : dc = 3)
  (hw : weeks = 6) :
  let days := weeks * 7
  (arithmetic_sequence_sum a₁ da days,
   arithmetic_sequence_sum b₁ db days,
   arithmetic_sequence_sum c₁ dc days) = (945, 1848, 2751) := by
  sorry

#eval arithmetic_sequence_sum 2 1 42
#eval arithmetic_sequence_sum 3 2 42
#eval arithmetic_sequence_sum 4 3 42

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_pills_l1183_118355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_homothety_l1183_118365

noncomputable def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def homothety (center : ℝ × ℝ) (ratio : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (center.1 + ratio * (p.1 - center.1), center.2 + ratio * (p.2 - center.2))

theorem locus_of_homothety (O P : ℝ × ℝ) (h_dist : distance O P = 15) :
  let ω := Circle O 6
  let M := {m : ℝ × ℝ | ∃ q ∈ ω, m = homothety P (1/3) q}
  M = Circle (P.1 + 5 * (O.1 - P.1) / 15, P.2 + 5 * (O.2 - P.2) / 15) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_homothety_l1183_118365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exponential_l1183_118315

theorem derivative_of_exponential (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  deriv (fun x : ℝ => a^x) = fun x => a^x * Real.log a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exponential_l1183_118315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_15_square_root_16_li_qiang_status_l1183_118328

-- Define propositions
def divisible_by_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k
def divisible_by_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k
def square_root_16_is_4 : Prop := 4 * 4 = 16
def square_root_16_is_neg_4 : Prop := (-4) * (-4) = 16
def is_high_school_freshman (person : String) : Prop := sorry
def is_youth_league_member (person : String) : Prop := sorry

-- Theorem statements
theorem divisibility_15 : 
  (divisible_by_3 15 ∧ divisible_by_5 15) ↔ (divisible_by_3 15 ∧ divisible_by_5 15) := by
  sorry

theorem square_root_16 :
  (square_root_16_is_4 ∨ square_root_16_is_neg_4) ↔ (square_root_16_is_4 ∨ square_root_16_is_neg_4) := by
  sorry

theorem li_qiang_status (li_qiang : String) :
  (is_high_school_freshman li_qiang ∧ is_youth_league_member li_qiang) ↔
  (is_high_school_freshman li_qiang ∧ is_youth_league_member li_qiang) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_15_square_root_16_li_qiang_status_l1183_118328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_divisible_by_27_from_8th_row_no_855_below_7th_row_l1183_118387

/-- Generates the nth row of the sequence given the initial values --/
def generate_row (n : ℕ) (a b c : ℤ) : List ℤ :=
  match n with
  | 0 => [a, b, c]
  | n+1 => 
    let prev := generate_row n a b c
    match prev with
    | [x, y, z] => [x - y, y - z, z - x]
    | _ => []  -- This case should never occur if the function is used correctly

/-- All entries in the 8th row and below are divisible by 27 --/
theorem all_divisible_by_27_from_8th_row (a b c : ℤ) (n : ℕ) (h : n ≥ 8) :
  ∀ x ∈ generate_row n a b c, 27 ∣ x := by
  sorry

/-- 855 cannot appear below the 7th row --/
theorem no_855_below_7th_row (a b c : ℤ) (n : ℕ) (h : n > 7) :
  855 ∉ generate_row n a b c := by
  sorry

#eval generate_row 7 1 2 3  -- Test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_divisible_by_27_from_8th_row_no_855_below_7th_row_l1183_118387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_labelings_l1183_118385

/-- A type representing the vertices of a regular hexagon and its center -/
inductive HexagonPoint
| A | B | C | D | E | F | G

/-- A labeling of the hexagon points with digits 1 through 7 -/
def HexagonLabeling := HexagonPoint → Fin 7

/-- The sum of labels on a line through the center -/
def lineSum (l : HexagonLabeling) (p q : HexagonPoint) : ℕ :=
  (l p).val + 1 + (l HexagonPoint.G).val + 1 + (l q).val + 1

/-- A labeling is valid if it uses each digit once and the sums on the specified lines are equal -/
def isValidLabeling (l : HexagonLabeling) : Prop :=
  Function.Injective l ∧
  lineSum l HexagonPoint.A HexagonPoint.C = lineSum l HexagonPoint.B HexagonPoint.D ∧
  lineSum l HexagonPoint.B HexagonPoint.D = lineSum l HexagonPoint.C HexagonPoint.E

/-- The main theorem stating the number of valid labelings -/
theorem number_of_valid_labelings :
  ∃ (s : Finset HexagonLabeling), (∀ l ∈ s, isValidLabeling l) ∧ s.card = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_labelings_l1183_118385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_8_sum_of_valid_B_l1183_118352

def last_three_digits (n : ℕ) : ℕ := n % 1000

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

theorem divisibility_by_8 (n : ℕ) : is_divisible_by_8 n ↔ is_divisible_by_8 (last_three_digits n) := by
  sorry

def possible_B_values : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def valid_B (b : ℕ) : Bool := (200 + 10 * b + 7) % 8 == 0

theorem sum_of_valid_B : (possible_B_values.filter valid_B).sum = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_8_sum_of_valid_B_l1183_118352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_segments_l1183_118337

-- Define the necessary types
variable (Point : Type)
variable (Circle : Type)

-- Define the necessary functions and relations
variable (angle_bisector : Point → Point → Point → Point → Prop)
variable (on_ray : Point → Point → Point → Prop)
variable (concyclic : Point → Point → Point → Point → Prop)
variable (length : Point → Point → ℝ)

-- State the theorem
theorem equality_of_segments 
  (A B C D A₁ A₂ B₁ B₂ : Point) 
  (circle1 circle2 : Circle) :
  angle_bisector A C B D →
  on_ray C A A₁ →
  on_ray C A A₂ →
  on_ray C B B₁ →
  on_ray C B B₂ →
  concyclic A₁ C B₁ D →
  concyclic A₂ C B₂ D →
  length A₁ A₂ = length B₁ B₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_of_segments_l1183_118337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1183_118324

-- Define the function f(x) = x ln(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the tangent line equation
def tangentLine (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem statement
theorem tangent_line_at_one (x : ℝ) (h : x > 0) :
  tangentLine x (f x) ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1183_118324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_least_integers_sum_l1183_118391

open Nat

-- Define the tau function
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

-- Define the property we're looking for
def has_property (n : ℕ) : Prop := tau n + tau (n + 1) = 8

-- Statement of the theorem
theorem six_least_integers_sum :
  ∃ (n₁ n₂ n₃ n₄ n₅ n₆ : ℕ),
    (∀ i, i ∈ [n₁, n₂, n₃, n₄, n₅, n₆] → has_property i) ∧
    (∀ m, m < max n₁ (max n₂ (max n₃ (max n₄ (max n₅ n₆)))) →
      m ∉ [n₁, n₂, n₃, n₄, n₅, n₆] → ¬has_property m) ∧
    (∀ S : Finset ℕ, S.card = 6 →
      (∀ n ∈ S, has_property n) →
      n₁ + n₂ + n₃ + n₄ + n₅ + n₆ ≤ S.sum id) :=
by sorry

#check six_least_integers_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_least_integers_sum_l1183_118391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_school_year_hours_l1183_118334

/-- Calculates the required weekly hours for Amy to earn a target amount during school year,
    given her summer earnings, summer work schedule, and school year duration. -/
noncomputable def required_school_year_hours (summer_earnings : ℝ) (summer_weekly_hours : ℝ) 
  (summer_weeks : ℝ) (school_year_weeks : ℝ) (school_year_target : ℝ) : ℝ :=
  let hourly_wage := summer_earnings / (summer_weekly_hours * summer_weeks)
  school_year_target / (school_year_weeks * hourly_wage)

/-- Theorem stating that Amy needs to work approximately 13 hours per week 
    during the school year to earn $3600, given her summer work details. -/
theorem amy_school_year_hours : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |required_school_year_hours 3600 40 12 36 3600 - 13| < ε :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_school_year_hours_l1183_118334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_over_expansion_iff_all_digits_less_than_ten_l1183_118373

/-- Represents a base 10 over-expansion of a positive integer -/
def BaseOverExpansion (N : ℕ+) : Type :=
  Σ' (k : ℕ) (d : Fin (k + 1) → Fin 11), 
    (d 0 ≠ 0) ∧ 
    (N = (Finset.sum (Finset.range (k + 1)) fun i => (d i : ℕ) * 10^i))

/-- Checks if a positive integer has a unique base 10 over-expansion -/
def HasUniqueOverExpansion (N : ℕ+) : Prop :=
  ∃! (_exp : BaseOverExpansion N), True

/-- Checks if all digits in the usual base 10 representation are less than 10 -/
def AllDigitsLessThanTen (N : ℕ+) : Prop :=
  ∀ d : ℕ, d ∈ (N : ℕ).digits 10 → d < 10

/-- Theorem: A positive integer has a unique base 10 over-expansion if and only if
    all its digits in the usual base 10 representation are less than 10 -/
theorem unique_over_expansion_iff_all_digits_less_than_ten (N : ℕ+) :
  HasUniqueOverExpansion N ↔ AllDigitsLessThanTen N := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_over_expansion_iff_all_digits_less_than_ten_l1183_118373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_is_universal_negation_l1183_118390

theorem negation_of_existence_is_universal_negation :
  (¬ ∃ x : ℝ, Real.tan x = 1) ↔ (∀ x : ℝ, Real.tan x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_is_universal_negation_l1183_118390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_two_elements_in_P_and_Q_l1183_118394

-- Define the sets A, B, P, and Q
def A : Set ℝ := {x | (x + 3) / (x - 4) < 0}
def B : Set ℝ := {x | |x| < 2}
def P : Set ℤ := {x : ℤ | (x : ℝ) ∈ A ∩ (Set.univ \ B)}
def Q (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ m + 1}

-- State the theorem
theorem range_of_m_for_two_elements_in_P_and_Q :
  ∃ m : ℝ, 2 ≤ m ∧ m ≤ 3 ∧
  (∃ (x y : ℤ), x ∈ P ∧ y ∈ P ∧ x ≠ y ∧ (x : ℝ) ∈ Q m ∧ (y : ℝ) ∈ Q m ∧
  ∀ z : ℤ, z ∈ P → (z : ℝ) ∈ Q m → (z = x ∨ z = y)) ∧
  ∀ m' : ℝ, (∃ (x y : ℤ), x ∈ P ∧ y ∈ P ∧ x ≠ y ∧ (x : ℝ) ∈ Q m' ∧ (y : ℝ) ∈ Q m' ∧
  ∀ z : ℤ, z ∈ P → (z : ℝ) ∈ Q m' → (z = x ∨ z = y)) → 2 ≤ m' ∧ m' ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_two_elements_in_P_and_Q_l1183_118394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_l1183_118318

/-- The series term for a given n -/
def seriesTerm (n : ℕ) : ℚ :=
  (3 * n^3 - 2 * n^2 - 2 * n + 3) / (n^6 - n^5 + n^3 - n^2 + n - 1)

/-- The series sum from n=2 to infinity -/
noncomputable def seriesSum : ℝ := ∑' n, if n ≥ 2 then seriesTerm n else 0

/-- Theorem stating that the series sum equals 1 -/
theorem series_sum_equals_one : seriesSum = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_l1183_118318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_sale_money_l1183_118310

/-- The amount of money raised by selling cookies baked by Clementine, Jake, and Tory --/
theorem cookie_sale_money (clementine_cookies jake_cookies tory_cookies : ℕ) 
  (price_per_cookie : ℚ) : 
  clementine_cookies = 72 →
  jake_cookies = 2 * clementine_cookies →
  tory_cookies = (jake_cookies + clementine_cookies) / 2 →
  price_per_cookie = 2 →
  (clementine_cookies + jake_cookies + tory_cookies) * price_per_cookie = 648 := by
  intros h1 h2 h3 h4
  -- Proof steps would go here
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_sale_money_l1183_118310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_b_highest_no_car_round_trip_l1183_118392

/-- Represents a ship with passenger data -/
structure Ship where
  roundTripPercent : ℝ
  carPercent : ℝ

/-- Calculates the percentage of passengers with round-trip tickets but no cars -/
noncomputable def noCarRoundTripPercent (s : Ship) : ℝ :=
  s.roundTripPercent * (1 - s.carPercent / 100)

/-- Theorem: Ship B has the highest percentage of passengers with round-trip tickets but no cars -/
theorem ship_b_highest_no_car_round_trip 
  (ship_a ship_b ship_c : Ship)
  (ha_round : ship_a.roundTripPercent = 30)
  (ha_car : ship_a.carPercent = 25)
  (hb_round : ship_b.roundTripPercent = 50)
  (hb_car : ship_b.carPercent = 15)
  (hc_round : ship_c.roundTripPercent = 20)
  (hc_car : ship_c.carPercent = 35) :
  noCarRoundTripPercent ship_b > noCarRoundTripPercent ship_a ∧ 
  noCarRoundTripPercent ship_b > noCarRoundTripPercent ship_c := by
  sorry

#eval "Theorem statement completed."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_b_highest_no_car_round_trip_l1183_118392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1183_118393

-- Problem 1
theorem problem_1 : (1/2 : ℝ)⁻¹ + (Real.pi - 2023)^0 - (-1)^2023 = 4 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (3*a^2)^2 - a^2 * 2*a^2 + (-2*a^2)^3 / a^2 = -a^4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1183_118393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l1183_118300

/-- Given a square with side length z divided into a smaller square with side length w
    and four congruent rectangles, the perimeter of one rectangle is 2z. -/
theorem rectangle_perimeter (z w : ℝ) (hz : z > 0) (hw : 0 < w ∧ w < z) :
  2 * ((z - w) + w) = 2 * z := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l1183_118300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1183_118308

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Defines the parabola y^2 = 8x -/
def onParabola (p : Point) : Prop :=
  p.y^2 = 8 * p.x

theorem min_distance_sum :
  let C : Point := { x := 2, y := 0 }
  let D : Point := { x := 6, y := 3 }
  (∀ Q : Point, onParabola Q → distance C Q + distance D Q ≥ 6) ∧
  (∃ Q : Point, onParabola Q ∧ distance C Q + distance D Q = 6) := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1183_118308
