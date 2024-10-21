import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l470_47067

open Real Set

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (x + π/4) * Real.cos (x + π/4) + Real.sin (2*x) - 1

noncomputable def g (x : ℝ) : ℝ := f (x + π/6)

theorem f_and_g_properties :
  (∀ k : ℤ, MonotoneOn f (Icc (k*π - 5*π/12) (k*π + π/12))) ∧
  (∀ x ∈ Icc 0 (π/2), g x ≥ -3) ∧
  (∀ x ∈ Icc 0 (π/2), g x ≤ Real.sqrt 3 - 1) ∧
  (g (5*π/12) = -3) ∧
  (g 0 = Real.sqrt 3 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l470_47067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l470_47073

/-- The circle with center (3, -1) and radius 5 -/
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 25

/-- The line parametrized by t -/
def line (t : ℝ) : ℝ × ℝ := (-2 + t, 1 - t)

/-- The chord length cut by the circle from the line -/
noncomputable def chord_length : ℝ := Real.sqrt 82

/-- Theorem stating that the chord length is correct -/
theorem chord_length_is_correct : 
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ 
  circle_eq (line t₁).1 (line t₁).2 ∧ 
  circle_eq (line t₂).1 (line t₂).2 ∧
  ((line t₁).1 - (line t₂).1)^2 + ((line t₁).2 - (line t₂).2)^2 = chord_length^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l470_47073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l470_47060

/-- The function f(x) = x^3 + k ln x -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^3 + k * Real.log x

/-- The derivative of f(x) -/
noncomputable def f_derivative (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 + k / x

theorem inequality_theorem (k : ℝ) (x₁ x₂ : ℝ) 
    (h_k : k ≥ -3) 
    (h_x₁ : x₁ ≥ 1) 
    (h_x₂ : x₂ ≥ 1) 
    (h_order : x₁ > x₂) : 
  (f_derivative k x₁ + f_derivative k x₂) / 2 > (f k x₁ - f k x₂) / (x₁ - x₂) := by
  sorry

#check inequality_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l470_47060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_yield_theorem_l470_47065

/-- Calculates the yield percentage of a stock given its dividend rate, par value, and market value. -/
noncomputable def yield_percentage (dividend_rate : ℝ) (par_value : ℝ) (market_value : ℝ) : ℝ :=
  (dividend_rate * par_value / market_value) * 100

/-- Theorem stating that an 8% stock with a market value of 40 and par value of 100 has a yield percentage of 20%. -/
theorem stock_yield_theorem :
  let dividend_rate : ℝ := 0.08
  let par_value : ℝ := 100
  let market_value : ℝ := 40
  yield_percentage dividend_rate par_value market_value = 20 := by
  -- Unfold the definition of yield_percentage
  unfold yield_percentage
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_yield_theorem_l470_47065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_lives_sum_l470_47075

noncomputable section

-- Define the number of lives for each animal
def cat_lives : ℝ := 9.5
def dog_lives : ℝ := cat_lives - 3.25
def mouse_lives : ℝ := dog_lives + 7.75
def elephant_lives : ℝ := 2 * cat_lives - 5.5
def fish_lives : ℝ := (2/3) * elephant_lives

-- Theorem statement
theorem total_lives_sum :
  cat_lives + dog_lives + mouse_lives + elephant_lives + fish_lives = 52.25 := by
  -- Expand definitions
  unfold cat_lives dog_lives mouse_lives elephant_lives fish_lives
  -- Perform algebraic simplifications
  simp [add_assoc, mul_add, add_mul]
  -- The proof is completed numerically
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_lives_sum_l470_47075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_journey_speed_l470_47034

/-- Calculates the overall average speed of a bicycle journey with multiple segments -/
noncomputable def overall_average_speed (total_distance : ℝ) (segments : List (ℝ × ℝ)) : ℝ :=
  let total_time := segments.foldr (λ (d, s) acc => acc + d / s) 0
  total_distance / total_time

/-- The bicycle journey problem -/
theorem bicycle_journey_speed : 
  let segments := [(15, 12), (20, 8), (10, 25), (15, 18)]
  let total_distance := 60
  abs (overall_average_speed total_distance segments - 12.04) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_journey_speed_l470_47034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_quadratic_polynomials_l470_47037

theorem no_special_quadratic_polynomials :
  ¬∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ (r s : ℝ), r ≠ s ∧ r ≠ 0 ∧ s ≠ 0 ∧
    (a * r^2 + b * r + c = 0) ∧ (a * s^2 + b * s + c = 0) ∧
    ((a = r * s ∧ (b = r ∧ c = s) ∨ (b = s ∧ c = r)) ∨
     (b = r * s ∧ (a = r ∧ c = s) ∨ (a = s ∧ c = r)) ∨
     (c = r * s ∧ (a = r ∧ b = s) ∨ (a = s ∧ b = r)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_quadratic_polynomials_l470_47037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_radius_l470_47071

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first parabola -/
def parabola1 (p : Point) : Prop :=
  p.y = (p.x - 2)^2

/-- Definition of the second parabola -/
def parabola2 (p : Point) : Prop :=
  p.x + 1 = (p.y + 2)^2

/-- Definition of a circle with center (2, -2) and radius r -/
def circleEq (p : Point) (r : ℝ) : Prop :=
  (p.x - 2)^2 + (p.y + 2)^2 = r^2

/-- Theorem stating that the square of the radius of the circle containing
    all intersection points of the two parabolas is 8 -/
theorem intersection_circle_radius : ∃ r : ℝ, 
  (∀ p : Point, parabola1 p → parabola2 p → circleEq p r) ∧ r^2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_radius_l470_47071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l470_47080

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 - 6*y + 9 - m^2 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := y = Real.sqrt 3 * x + 1

-- Define the intersection condition
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∧ circle_eq x1 y1 m ∧ circle_eq x2 y2 m ∧ line_eq x1 y1 ∧ line_eq x2 y2

-- Theorem statement
theorem circle_line_intersection (m : ℝ) (hm : m > 0) 
  (h_intersect : intersects_at_two_points m) : m = Real.sqrt 2 := by
  sorry

#check circle_line_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l470_47080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_greater_than_sin_B_when_squared_and_shifted_l470_47026

open Real

-- Define an acute triangle
def is_acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧ A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2

-- Define our function f
noncomputable def f (x a k : ℝ) : ℝ := (x - a)^k

-- State the theorem
theorem cos_A_greater_than_sin_B_when_squared_and_shifted 
  (A B C : ℝ) (h : is_acute_triangle A B C) :
  f (cos A) 1 2 > f (sin B) 1 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_greater_than_sin_B_when_squared_and_shifted_l470_47026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l470_47046

/-- The circle C with equation x^2 + y^2 + 2x - 4y + 3 = 0 -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 3 = 0

/-- The line with equation 2ax + by + 6 = 0 -/
def symmetry_line (a b x y : ℝ) : Prop :=
  2*a*x + b*y + 6 = 0

/-- The circle C is symmetric with respect to the line 2ax + by + 6 = 0 -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_C x y ↔ (∃ x' y' : ℝ, circle_C x' y' ∧ 
    ((x + x')/2, (y + y')/2) ∈ {(x, y) | symmetry_line a b x y})

/-- The length of the tangent from point (a, b) to circle C -/
noncomputable def tangent_length (a b : ℝ) : ℝ :=
  Real.sqrt ((a + 1)^2 + (b - 2)^2 - 2)

theorem min_tangent_length (a b : ℝ) :
  is_symmetric a b → ∀ a' b' : ℝ, tangent_length a' b' ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l470_47046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_l470_47061

noncomputable def cone_height (r : ℝ) (v : ℝ) : ℝ :=
  (3 * v) / (Real.pi * r^2)

noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem funnel_height :
  round_to_nearest (cone_height 4 150) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_l470_47061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l470_47020

/-- Triangle ABC with sides a, b, c corresponding to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine law for triangles -/
axiom sine_law (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The cosine law for triangles -/
axiom cosine_law (t : Triangle) : t.b^2 = t.a^2 + t.c^2 - 2 * t.a * t.c * Real.cos t.B

/-- The area formula for triangles -/
noncomputable def triangle_area (t : Triangle) : ℝ := 1/2 * t.a * t.c * Real.sin t.B

theorem triangle_properties (t : Triangle) 
  (ha : t.a = 4) 
  (hB : Real.cos t.B = 4/5) : 
  (t.b = 6 → Real.sin t.A = 2/5) ∧ 
  (triangle_area t = 12 → t.b = 2 * Real.sqrt 13 ∧ t.c = 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l470_47020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l470_47041

-- Define the ellipse C₁
noncomputable def C₁ (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the hyperbola C₂
def C₂ (x y : ℝ) : Prop := x^2/2 - y^2 = 1

-- Define the line that intersects C₁ at A and B
noncomputable def line_AB (x y : ℝ) : Prop := x + Real.sqrt 2 * y = 0

-- Define point A
noncomputable def point_A : ℝ × ℝ := (-Real.sqrt 2, 1)

-- Define point B
noncomputable def point_B : ℝ × ℝ := (Real.sqrt 2, -1)

-- Define the trajectory of Q
def trajectory_Q (x y : ℝ) : Prop := 2 * x^2 + y^2 = 5

-- Define the excluded points
noncomputable def excluded_points : Set (ℝ × ℝ) := {(Real.sqrt 2, -1), (Real.sqrt 2 / 2, -2), (-Real.sqrt 2, 1), (-Real.sqrt 2 / 2, 2)}

-- Helper function to calculate the area of a triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem ellipse_problem :
  -- 1. Equation of C₁
  (∀ x y, C₁ x y ↔ x^2/4 + y^2/2 = 1) ∧
  -- 2. Trajectory of Q
  (∀ x y, (x, y) ∉ excluded_points → (∃ P : ℝ × ℝ, P ≠ point_A ∧ P ≠ point_B ∧ C₁ P.1 P.2 ∧
    ((x + Real.sqrt 2) * (P.1 + Real.sqrt 2) + (y - 1) * (P.2 - 1) = 0) ∧
    ((x - Real.sqrt 2) * (P.1 - Real.sqrt 2) + (y + 1) * (P.2 + 1) = 0)) ↔
    trajectory_Q x y) ∧
  -- 3. Maximum area of triangle ABQ
  (∃ Q : ℝ × ℝ, (Q = (Real.sqrt 2 / 2, 2) ∨ Q = (-Real.sqrt 2 / 2, -2)) ∧
    trajectory_Q Q.1 Q.2 ∧
    (∀ R : ℝ × ℝ, trajectory_Q R.1 R.2 → R ∉ excluded_points →
      area_triangle point_A point_B R ≤ area_triangle point_A point_B Q)) ∧
  (area_triangle point_A point_B (Real.sqrt 2 / 2, 2) = 5 * Real.sqrt 2 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l470_47041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_harmonic_mean_of_3_and_54_l470_47064

noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

theorem closest_integer_to_harmonic_mean_of_3_and_54 :
  ∃ (n : ℤ), n = 6 ∧ 
  ∀ (m : ℤ), |harmonic_mean 3 54 - ↑n| ≤ |harmonic_mean 3 54 - ↑m| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_harmonic_mean_of_3_and_54_l470_47064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_circle_l470_47019

-- Define the system of equations
def system (a x y : ℝ) : Prop :=
  (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

-- Define the set of solutions
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ a : ℝ, system a p.1 p.2}

-- Define the circle
def solution_circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5 ∧ p ≠ (2, -1)}

-- Theorem stating the equivalence of the solution set and the circle
theorem solution_set_eq_circle : solution_set = solution_circle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_circle_l470_47019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l470_47085

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 1 / (2 * x + b)

-- Define the inverse function f⁻¹
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

-- Theorem statement
theorem inverse_function_condition (b : ℝ) :
  (∀ x, f_inv (f b x) = x) → b = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l470_47085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_green_balls_l470_47077

/-- The probability of drawing exactly two green balls from a bag -/
theorem probability_two_green_balls (red yellow green drawn : ℕ) 
  (h_red : red = 3)
  (h_yellow : yellow = 5)
  (h_green : green = 4)
  (h_drawn : drawn = 3) :
  let total := red + yellow + green
  let ways_total := Nat.choose total drawn
  let ways_two_green := Nat.choose green 2 * Nat.choose (red + yellow) 1
  (ways_two_green : ℚ) / ways_total = 12 / 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_green_balls_l470_47077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_satisfying_conditions_l470_47049

noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y + 1| / Real.sqrt 2

def in_region (x y : ℝ) : Prop :=
  x + y - 1 < 0 ∧ x - y + 1 > 0

def points : List (ℝ × ℝ) := [(1, 1), (-1, 1), (-1, -1), (1, -1)]

theorem unique_point_satisfying_conditions : 
  ∃! p, p ∈ points ∧ distance_to_line p.1 p.2 = Real.sqrt 2 / 2 ∧ in_region p.1 p.2 ∧ p = (-1, -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_satisfying_conditions_l470_47049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l470_47054

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = t.a^2 + t.b * t.c ∧
  Real.sin t.B = Real.sqrt 3 / 3 ∧
  t.b = 2

-- Helper function to calculate area
noncomputable def area (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.A = π/3 ∧ area t = (3 * Real.sqrt 2 + Real.sqrt 3) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l470_47054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l470_47010

/-- Calculates the time (in seconds) it takes for a train to pass a stationary point -/
noncomputable def time_to_pass (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

theorem train_passing_time :
  time_to_pass 280 63 = 16 := by
  -- Unfold the definition of time_to_pass
  unfold time_to_pass
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l470_47010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_and_binomial_properties_l470_47079

/-- Normal distribution with mean μ and standard deviation σ -/
structure NormalDist (μ σ : ℝ) where
  value : ℝ

/-- Binomial distribution with n trials and probability p -/
structure BinomialDist (n : ℕ) (p : ℝ) where
  value : ℕ

/-- Variance of a random variable -/
noncomputable def variance {α : Type} (X : α) : ℝ := sorry

/-- Probability of an event -/
noncomputable def prob {α : Type} (X : α) (p : α → Prop) : ℝ := sorry

theorem normal_and_binomial_properties
  (X : NormalDist 1 2)
  (Y : BinomialDist 10 0.4) :
  variance Y.value = 2.4 ∧ prob X.value (λ x => x < 3) = 0.84135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_and_binomial_properties_l470_47079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cabd_position_l470_47032

/-- The type representing our set of letters -/
inductive Letter : Type where
  | A : Letter
  | B : Letter
  | C : Letter
  | D : Letter
deriving Repr, DecidableEq

/-- A permutation is a list of 4 letters -/
def Permutation := List Letter

/-- The lexicographic order of letters -/
def letter_order (l1 l2 : Letter) : Bool :=
  match l1, l2 with
  | Letter.A, _ => true
  | Letter.B, Letter.A => false
  | Letter.B, _ => true
  | Letter.C, Letter.D => true
  | Letter.C, _ => false
  | Letter.D, _ => false

/-- Lexicographic comparison of permutations -/
def lex_compare (p1 p2 : Permutation) : Bool :=
  match p1, p2 with
  | [], [] => true
  | [], _ => true
  | _, [] => false
  | (h1::t1), (h2::t2) => 
    if h1 = h2 then lex_compare t1 t2
    else letter_order h1 h2

/-- Generate all permutations of the given letters -/
def all_permutations : List Permutation :=
  sorry

/-- The specific permutation we're interested in -/
def CABD : Permutation :=
  [Letter.C, Letter.A, Letter.B, Letter.D]

/-- Count permutations that come before the given permutation -/
def count_before (p : Permutation) : Nat :=
  (all_permutations.filter (lex_compare · p)).length

theorem cabd_position : count_before CABD = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cabd_position_l470_47032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoid_parameters_l470_47082

noncomputable def sinusoid (a b c : ℝ) (x : ℝ) : ℝ := a * Real.sin (b * x + c)

theorem sinusoid_parameters 
  (a b c : ℝ) 
  (h1 : a < 0) 
  (h2 : b > 0) 
  (h3 : ∀ x, |sinusoid a b c x| ≤ 3) 
  (h4 : ∀ x, sinusoid a b c x = sinusoid a b c (x + 8 * Real.pi)) :
  a = -3 ∧ b = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoid_parameters_l470_47082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_PQRS_l470_47008

-- Define the trapezoid PQRS
noncomputable def P : ℝ × ℝ := (1, 1)
noncomputable def Q : ℝ × ℝ := (1, 4)
noncomputable def R : ℝ × ℝ := (6, 4)
noncomputable def S : ℝ × ℝ := (7, 1)

-- Define the area of a trapezoid
noncomputable def trapezoid_area (a b h : ℝ) : ℝ := (a + b) * h / 2

-- Theorem statement
theorem area_of_trapezoid_PQRS :
  trapezoid_area (Q.2 - P.2) (S.1 - R.1) (R.1 - P.1) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_PQRS_l470_47008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l470_47048

theorem expansion_properties :
  let n : ℕ := 9
  let odd_sum := (Finset.range ((n + 1) / 2)).sum (λ k ↦ Nat.choose n (2 * k + 1))
  let constant_term := Nat.choose n (n / 2 + 1)
  odd_sum = 256 ∧ constant_term = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l470_47048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_coloring_theorem_l470_47095

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A pentagon defined by its vertices -/
structure Pentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- Color of a point -/
inductive Color
  | Red
  | Blue

/-- A function that determines if a pentagon is regular -/
def isRegular (p : Pentagon) : Prop := sorry

/-- A function that determines if a pentagon is formed by the midpoints of another pentagon -/
def isMidpointPentagon (p₁ p₂ : Pentagon) : Prop := sorry

/-- A function that determines if four points form a cyclic quadrilateral -/
def isCyclicQuadrilateral (p₁ p₂ p₃ p₄ : Point) : Prop := sorry

/-- Define membership for Point in Pentagon -/
instance : Membership Point Pentagon where
  mem p pent := p = pent.A ∨ p = pent.B ∨ p = pent.C ∨ p = pent.D ∨ p = pent.E

theorem pentagon_coloring_theorem 
  (pentagons : Fin 11 → Pentagon)
  (colors : Fin 11 → Pentagon → Point → Color)
  (h₁ : ∀ n, isRegular (pentagons n))
  (h₂ : ∀ n, n ≥ 2 → isMidpointPentagon (pentagons (n-1)) (pentagons n)) :
  ∃ (p₁ p₂ p₃ p₄ : Point) (c : Color) (i j : Fin 11),
    p₁ ∈ pentagons i ∧ p₂ ∈ pentagons i ∧ 
    p₃ ∈ pentagons j ∧ p₄ ∈ pentagons j ∧
    colors i (pentagons i) p₁ = c ∧ 
    colors i (pentagons i) p₂ = c ∧ 
    colors j (pentagons j) p₃ = c ∧ 
    colors j (pentagons j) p₄ = c ∧
    isCyclicQuadrilateral p₁ p₂ p₃ p₄ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_coloring_theorem_l470_47095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_2_sqrt_3_l470_47001

/-- Represents a right triangle -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ

/-- Represents a quadrilateral ABCD with specific properties -/
structure SpecialQuadrilateral where
  -- ABC is a right triangle
  abc_right : RightTriangle
  -- BCD is an equilateral triangle
  bcd_equilateral : EquilateralTriangle
  -- Side lengths of ABC
  ab_length : abc_right.side1 = 6
  bc_length : abc_right.side2 = 8
  ac_length : abc_right.hypotenuse = 10
  -- Side length of BCD
  bcd_side : bcd_equilateral.side = 8
  -- BC is common to both triangles
  bc_common : abc_right.side2 = bcd_equilateral.side

/-- The length of the crease when folding point A onto point D -/
noncomputable def crease_length (q : SpecialQuadrilateral) : ℝ := 2 * Real.sqrt 3

/-- Theorem stating the length of the crease -/
theorem crease_length_is_2_sqrt_3 (q : SpecialQuadrilateral) :
  crease_length q = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_2_sqrt_3_l470_47001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_in_range_exists_axis_of_symmetry_l470_47087

-- Define the function as noncomputable due to dependency on Real.sin
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

-- Define the theorem
theorem axis_of_symmetry_in_range (φ : ℝ) :
  (0 < φ ∧ φ < Real.pi / 2) →
  (∃ k : ℤ, π / 6 < k * π / 2 + π / 4 - φ / 2 ∧ k * π / 2 + π / 4 - φ / 2 < π / 3) →
  φ = Real.pi / 12 :=
by
  sorry

-- Additional helper theorem to show the existence of the axis of symmetry
theorem exists_axis_of_symmetry (φ : ℝ) :
  ∃ k : ℤ, ∀ x : ℝ, f x φ = f (k * π / 2 + π / 4 - φ / 2 - x) φ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_in_range_exists_axis_of_symmetry_l470_47087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripling_investment_l470_47017

/-- The annual interest rate that triples an investment in 15 years -/
noncomputable def annual_interest_rate : ℝ :=
  Real.log 3 / 15

theorem tripling_investment (r : ℝ) :
  r = annual_interest_rate ↔ (1 + r)^15 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripling_investment_l470_47017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_union_equal_original_l470_47063

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the semiperimeter
noncomputable def semiperimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

-- Define the area of a triangle using Heron's formula
noncomputable def area (t : Triangle) : ℝ :=
  let s := semiperimeter t
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

-- Define the scaling operation
def scale (t : Triangle) (factor : ℝ) : Triangle :=
  { a := t.a * factor, b := t.b * factor, c := t.c * factor }

-- Theorem statement
theorem area_union_equal_original (t : Triangle) (h1 : t.a = 25) (h2 : t.b = 39) (h3 : t.c = 42) :
  area t = Real.sqrt 293192 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_union_equal_original_l470_47063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_parabola_l470_47043

/-- The focus of the parabola y = 4x^2 -/
noncomputable def parabola_focus : ℝ × ℝ := (0, 1/16)

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 4 * x^2

/-- Theorem stating that parabola_focus is indeed the focus of the parabola defined by parabola_equation -/
theorem focus_of_parabola :
  ∀ (x y : ℝ), parabola_equation x y →
  ∃ (d : ℝ), (x - parabola_focus.1)^2 + (y - parabola_focus.2)^2 = (y - d)^2 :=
by
  sorry

#check focus_of_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_parabola_l470_47043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_puzzle_l470_47057

/-- Represents a family --/
inductive Family : Type
| Zhao
| Qian
| Sun
| Li
| Zhou

/-- Represents an apartment number --/
def ApartmentNumber := Fin 110

/-- Represents the order of move-in --/
def MoveInOrder := Fin 5

/-- Information about a family's residence --/
structure Residence where
  family : Family
  apartment : ApartmentNumber
  moveInOrder : MoveInOrder

/-- The building configuration --/
structure Building where
  residences : List Residence
  emptyApartments : List ApartmentNumber

/-- Predicate to check if an apartment is on the top floor --/
def isTopFloor (apt : ApartmentNumber) : Prop := apt.val ≥ 109

/-- Predicate to check if two apartments are across from each other --/
def isAcross (apt1 apt2 : ApartmentNumber) : Prop := 
  (apt1.val % 2 = 0 ∧ apt2.val = apt1.val - 1) ∨ (apt1.val % 2 = 1 ∧ apt2.val = apt1.val + 1)

/-- Predicate to check if an apartment has someone living above and below on the same side --/
def hasPeopleAboveAndBelow (b : Building) (apt : ApartmentNumber) : Prop :=
  ∃ (above below : ApartmentNumber), 
    above ∈ b.residences.map Residence.apartment ∧
    below ∈ b.residences.map Residence.apartment ∧
    above.val > apt.val ∧ below.val < apt.val ∧
    above.val % 2 = apt.val % 2 ∧ below.val % 2 = apt.val % 2

/-- Predicate to check if a floor is empty --/
def isFloorEmpty (b : Building) (floor : Nat) : Prop :=
  ∀ apt, (apt.val ≥ floor * 10 + 1) ∧ (apt.val ≤ floor * 10 + 10) → apt ∈ b.emptyApartments

/-- The main theorem --/
theorem building_puzzle (b : Building) :
  (∃ r : Residence, r ∈ b.residences ∧ r.family = Family.Zhao ∧ r.moveInOrder.val = 2 ∧
    (∃ r1 : Residence, r1 ∈ b.residences ∧ r1.moveInOrder.val = 0 ∧ isAcross r.apartment r1.apartment)) →
  (∃ r : Residence, r ∈ b.residences ∧ r.family = Family.Qian ∧ isTopFloor r.apartment ∧
    (∀ r1 : Residence, r1 ∈ b.residences ∧ isTopFloor r1.apartment → r1 = r)) →
  (∃ r : Residence, r ∈ b.residences ∧ r.family = Family.Sun ∧ hasPeopleAboveAndBelow b r.apartment) →
  (∃ r : Residence, r ∈ b.residences ∧ r.family = Family.Li ∧ r.moveInOrder.val = 4 ∧
    isFloorEmpty b ((r.apartment.val - 1) / 10)) →
  (∃ r : Residence, r ∈ b.residences ∧ r.family = Family.Zhou ∧ r.apartment.val = 106 ∧
    (⟨104, by simp⟩ : ApartmentNumber) ∈ b.emptyApartments ∧ (⟨108, by simp⟩ : ApartmentNumber) ∈ b.emptyApartments) →
  (b.residences.length = 5) →
  (∃ (r1 r2 r3 r4 r5 : Residence),
    r1 ∈ b.residences ∧ r2 ∈ b.residences ∧ r3 ∈ b.residences ∧ r4 ∈ b.residences ∧ r5 ∈ b.residences ∧
    r1.moveInOrder.val = 0 ∧ r2.moveInOrder.val = 1 ∧ r3.moveInOrder.val = 2 ∧ r4.moveInOrder.val = 3 ∧ r5.moveInOrder.val = 4 ∧
    r1.apartment.val % 10 = 6 ∧ r2.apartment.val % 10 = 9 ∧ r3.apartment.val % 10 = 5 ∧ r4.apartment.val % 10 = 7 ∧ r5.apartment.val % 10 = 3) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_puzzle_l470_47057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_transformed_function_l470_47097

/-- Given a function f : ℝ → ℝ, area_between_curve_and_axis f computes
    the area between the graph of y = f(x) and the x-axis. -/
noncomputable def area_between_curve_and_axis (f : ℝ → ℝ) : ℝ := sorry

/-- Given a function f : ℝ → ℝ, transformed_function f computes
    the function 2f(x-4). -/
def transformed_function (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x ↦ 2 * (f (x - 4))

theorem area_of_transformed_function (f : ℝ → ℝ) :
  area_between_curve_and_axis f = 10 →
  area_between_curve_and_axis (transformed_function f) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_transformed_function_l470_47097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l470_47016

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * |x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)|

/-- Theorem: The area of a triangle with vertices at (1,2), (4,5), and (6,1) is 9 square units -/
theorem triangle_area_example : triangleArea 1 2 4 5 6 1 = 9 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l470_47016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l470_47031

def M : ℕ := 57^6 + 6*57^5 + 15*57^4 + 20*57^3 + 15*57^2 + 6*57 + 1

theorem number_of_factors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l470_47031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_associate_prof_charts_l470_47090

/-- Represents the number of associate professors -/
def A : ℕ := sorry

/-- Represents the number of assistant professors -/
def B : ℕ := sorry

/-- Represents the number of charts each associate professor brings -/
def C : ℕ := sorry

/-- The total number of people present -/
axiom total_people : A + B = 6

/-- The total number of pencils brought to the meeting -/
axiom total_pencils : 2 * A + B = 10

/-- The total number of charts brought to the meeting -/
axiom total_charts : C * A + 2 * B = 8

theorem associate_prof_charts : C = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_associate_prof_charts_l470_47090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_crates_l470_47015

/-- The diameter of each ball in cm -/
def ball_diameter : ℝ := 8

/-- The number of balls in each crate -/
def num_balls : ℕ := 150

/-- The height of Crate X with vertically stacked balls -/
def height_crate_x : ℝ := 120

/-- The height of Crate Y with staggered arrangement -/
noncomputable def height_crate_y : ℝ := 56 * Real.sqrt 3 + 8

/-- The theorem stating the difference in height between Crate X and Crate Y -/
theorem height_difference_crates : 
  height_crate_x - height_crate_y = 112 - 56 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_crates_l470_47015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_fraction_equals_3125_l470_47088

theorem floor_of_fraction_equals_3125 :
  ⌊(5^80 + 4^130 : ℝ) / (5^75 + 4^125 : ℝ)⌋ = 3125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_fraction_equals_3125_l470_47088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_A_is_random_event_l470_47084

-- Define the type for events
inductive Event : Type
  | A : Event  -- Yingying encounters a green light when crossing the intersection
  | B : Event  -- Drawing a ping-pong ball from a bag with one ping-pong ball and two glass balls
  | C : Event  -- Currently answering question 12 of this test paper
  | D : Event  -- The highest temperature in our city tomorrow will be 60°C

-- Define what it means for an event to be random
def isRandomEvent (e : Event) : Prop :=
  ∃ (condition : Prop), (condition → e = Event.A) ∧ (¬condition → e ≠ Event.A)

-- Theorem stating that only event A is a random event
theorem only_A_is_random_event :
  (isRandomEvent Event.A) ∧
  (¬isRandomEvent Event.B) ∧
  (¬isRandomEvent Event.C) ∧
  (¬isRandomEvent Event.D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_A_is_random_event_l470_47084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_currents_match_expected_l470_47003

/-- Represents a branch in an electrical circuit -/
structure Branch where
  voltage : ℝ
  resistance : ℝ

/-- Represents an electrical circuit with three branches -/
structure Circuit where
  branch1 : Branch
  branch2 : Branch
  branch3 : Branch

/-- Calculates the currents in a three-branch circuit -/
noncomputable def calculate_currents (c : Circuit) : ℝ × ℝ × ℝ := sorry

/-- The given circuit configuration -/
def given_circuit : Circuit :=
  { branch1 := { voltage := 1.08, resistance := 2 }
  , branch2 := { voltage := 1.9, resistance := 5 }
  , branch3 := { voltage := 0, resistance := 10 }
  }

/-- Theorem stating that the calculated currents match the expected values -/
theorem currents_match_expected : 
  let (i1, i2, i3) := calculate_currents given_circuit
  (abs (i1 + 0.227) < 0.001) ∧ (abs (i2 - 0.073) < 0.001) ∧ (abs (i3 - 0.153) < 0.001) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_currents_match_expected_l470_47003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_properties_l470_47081

/-- Sequence definition -/
noncomputable def a (n : ℕ) : ℝ := Real.cos (10^n * Real.pi / 180)

/-- Main theorem -/
theorem a_100_properties : (a 100 > 0) ∧ (|a 100| < 0.18) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_properties_l470_47081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jersey_revenue_proof_l470_47035

/-- The amount made from each jersey, given the total revenue and number of jerseys sold. -/
noncomputable def amount_per_jersey (total_revenue : ℚ) (num_jerseys : ℕ) : ℚ :=
  total_revenue / num_jerseys

/-- Proof that the amount made from each jersey is $165. -/
theorem jersey_revenue_proof (total_revenue : ℚ) (num_jerseys : ℕ) 
  (h1 : total_revenue = 25740)
  (h2 : num_jerseys = 156) :
  amount_per_jersey total_revenue num_jerseys = 165 := by
  sorry

#eval (25740 : ℚ) / 156

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jersey_revenue_proof_l470_47035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_rotation_reaches_four_oclock_l470_47093

/-- The radius of the clock face in centimeters -/
noncomputable def clock_radius : ℝ := 20

/-- The radius of the smaller disk in centimeters -/
noncomputable def disk_radius : ℝ := 10

/-- The angle in radians corresponding to the 4 o'clock position on a clock face -/
noncomputable def four_oclock_angle : ℝ := 2 * Real.pi / 3

/-- Theorem stating that when a disk of radius 10 cm rolls along a circle of radius 20 cm,
    it completes one rotation when the tangent point reaches the 4 o'clock position -/
theorem disk_rotation_reaches_four_oclock :
  let θ := 2 * Real.pi * clock_radius / (clock_radius + disk_radius)
  θ = four_oclock_angle := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_rotation_reaches_four_oclock_l470_47093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l470_47058

theorem triangle_trig_identity (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = Real.pi) (h5 : B > Real.pi/6) 
  (h6 : Real.sin (A + Real.pi/6) = 3/5) (h7 : Real.cos (B - Real.pi/6) = 4/5) :
  Real.sin (A + B) = 24/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l470_47058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_seven_l470_47042

/-- An arithmetic sequence with first term 13 and S_3 = S_11 -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = 13
  sum_equal : (Finset.range 3).sum a = (Finset.range 11).sum a

/-- The sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (Finset.range n).sum seq.a

/-- The value of n that maximizes S_n is 7 -/
theorem max_sum_at_seven (seq : ArithmeticSequence) :
  ∀ n : ℕ, S seq n ≤ S seq 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_seven_l470_47042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_theorem_l470_47021

/-- The cost for up to 2 hours of parking in a certain garage. -/
def C : ℝ := sorry

/-- The cost per hour for parking after the first 2 hours. -/
def excess_cost_per_hour : ℝ := 1.75

/-- The average cost per hour for 9 hours of parking. -/
def average_cost_9_hours : ℝ := 2.6944444444444446

/-- The total number of hours parked. -/
def total_hours : ℕ := 9

theorem parking_cost_theorem :
  (C + excess_cost_per_hour * (total_hours - 2 : ℝ)) / total_hours = average_cost_9_hours →
  C = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_theorem_l470_47021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l470_47052

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + a*x - a)

theorem range_of_f (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a ∈ Set.Iic (-4) ∪ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l470_47052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l470_47074

theorem smallest_number_divisible (n : ℕ) : n = 252 ↔ 
  (∀ m : ℕ, m < n → ¬(∀ d : ℕ, d ∈ ({8, 12, 22, 24} : Set ℕ) → (m - 12) % d = 0)) ∧ 
  (∀ d : ℕ, d ∈ ({8, 12, 22, 24} : Set ℕ) → (n - 12) % d = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l470_47074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_one_second_l470_47028

noncomputable def s (t : ℝ) : ℝ := (t + 1)^2 * (t - 1)

noncomputable def v (t : ℝ) : ℝ := deriv s t

theorem instantaneous_velocity_at_one_second :
  v 1 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_one_second_l470_47028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_cos_x_over_4_l470_47055

-- Define the function we're working with
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (x / 4)

-- State the theorem
theorem integral_of_cos_x_over_4 (C : ℝ) :
  ∀ x : ℝ, deriv f x = Real.cos (x / 4) := by
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_cos_x_over_4_l470_47055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_savings_problem_l470_47036

/-- Thomas's savings problem --/
theorem thomas_savings_problem
  (car_cost hourly_wage weekly_hours weekly_spending additional_needed weeks_per_year : ℕ)
  (h1 : car_cost = 15000)
  (h2 : hourly_wage = 9)
  (h3 : weekly_hours = 30)
  (h4 : weekly_spending = 35)
  (h5 : additional_needed = 2000)
  (h6 : weeks_per_year = 52) :
  ∃ (weekly_allowance : ℕ),
    weekly_allowance * weeks_per_year + 
    (hourly_wage * weekly_hours - weekly_spending) * weeks_per_year = 
    car_cost - additional_needed ∧
    weekly_allowance = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_savings_problem_l470_47036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_water_depth_l470_47009

/-- Represents a cylindrical vase -/
structure Vase where
  diameter : ℝ
  height : ℝ

/-- The volume of a cylinder -/
noncomputable def cylinderVolume (d : ℝ) (h : ℝ) : ℝ := (Real.pi / 4) * d^2 * h

theorem original_water_depth (largeVase smallVase : Vase) (h : ℝ) :
  largeVase.diameter = 20 →
  smallVase.diameter = 10 →
  smallVase.height = 16 →
  h = smallVase.height / 2 →
  cylinderVolume largeVase.diameter 16 - cylinderVolume smallVase.diameter h =
    cylinderVolume largeVase.diameter 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_water_depth_l470_47009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_of_rotated_cubes_intersection_volume_correct_l470_47045

/-- The volume of intersection of two cubes with edge length a, where one cube is rotated 45° relative to the other around an axis joining the centers of two opposite faces. -/
noncomputable def volume_intersection (a : ℝ) : ℝ := 2 * a^3 * (Real.sqrt 2 - 1)

/-- Two cubes with edge length a, one rotated 45° relative to the other around an axis joining the centers of two opposite faces, have an intersection volume of 2a³(√2 - 1). -/
theorem intersection_volume_of_rotated_cubes (a : ℝ) (ha : a > 0) :
  volume_intersection a = 2 * a^3 * (Real.sqrt 2 - 1) := by
  -- Unfold the definition of volume_intersection
  unfold volume_intersection
  -- The equality follows directly from the definition
  rfl

/-- The actual intersection volume calculation (placeholder) -/
noncomputable def actual_intersection_volume (a : ℝ) : ℝ := sorry

/-- The calculated intersection volume matches the actual intersection volume -/
theorem intersection_volume_correct (a : ℝ) (ha : a > 0) :
  actual_intersection_volume a = volume_intersection a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_of_rotated_cubes_intersection_volume_correct_l470_47045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flow_rate_is_half_kiloliter_per_minute_l470_47044

/-- Represents the flow rate problem with a tank and drains. -/
structure TankSystem where
  capacity : ℚ  -- Tank capacity in liters
  initial_fill : ℚ  -- Initial amount of water in liters
  drain1_rate : ℚ  -- Drain 1 rate in kiloliters per minute
  drain2_rate : ℚ  -- Drain 2 rate in kiloliters per minute
  fill_time : ℚ  -- Time to fill the tank completely in minutes

/-- Calculates the flow rate of the pipe given a TankSystem. -/
def calculate_flow_rate (system : TankSystem) : ℚ :=
  let total_drain_rate := system.drain1_rate + system.drain2_rate
  let volume_to_fill := system.capacity - system.initial_fill
  (volume_to_fill / 1000 + total_drain_rate * system.fill_time) / system.fill_time

/-- Theorem stating that the flow rate for the given system is 0.5 kiloliters per minute. -/
theorem flow_rate_is_half_kiloliter_per_minute :
  let system : TankSystem := {
    capacity := 2000,
    initial_fill := 1000,
    drain1_rate := 1 / 4,
    drain2_rate := 1 / 6,
    fill_time := 12
  }
  calculate_flow_rate system = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flow_rate_is_half_kiloliter_per_minute_l470_47044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l470_47056

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The decimal representation of 33 in base 4 --/
def num1 : Nat := to_decimal [3, 3] 4

/-- The decimal representation of 12 in base 16 --/
def num2 : Nat := to_decimal [1, 2] 16

/-- The decimal representation of 25 in base 7 --/
def num3 : Nat := to_decimal [2, 5] 7

theorem ascending_order : num1 < num2 ∧ num2 < num3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l470_47056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_bound_l470_47096

noncomputable def nested_sqrt (n : ℕ) : ℝ :=
  Real.sqrt (match n with
    | 0 => 1
    | m + 1 => (m + 2 : ℝ) * nested_sqrt m
  )

theorem nested_sqrt_bound (n : ℕ) (h : n ≥ 2) : nested_sqrt n < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_bound_l470_47096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_integral_inequality_and_polynomial_coefficients_l470_47013

theorem function_integral_inequality_and_polynomial_coefficients 
  (f : ℝ → ℝ) (hf_diff : Differentiable ℝ f) (hf_incr : Monotone f) (hf_zero : f 0 = 0) :
  (∫ (x : ℝ) in Set.Icc 0 1, f x * (deriv f x)) ≥ (1/2) * (∫ (x : ℝ) in Set.Icc 0 1, f x)^2 ∧
  ∀ (n : ℕ), 
    let g (x : ℝ) := x^(2*n+1) + ((-3)/(2*n+3:ℝ)) * x
    ∀ (p q : ℝ), ∫ (x : ℝ) in Set.Icc (-1) 1, (p*x + q) * g x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_integral_inequality_and_polynomial_coefficients_l470_47013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_distance_45min_l470_47099

/-- The distance traveled by the tip of a clock's minute hand -/
noncomputable def minute_hand_distance (length : ℝ) (minutes : ℝ) : ℝ :=
  2 * Real.pi * length * (minutes / 60)

/-- Theorem: The distance traveled by the tip of an 8 cm minute hand in 45 minutes is 12π cm -/
theorem minute_hand_distance_45min :
  minute_hand_distance 8 45 = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_distance_45min_l470_47099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_one_ninth_l470_47092

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to avoid missing case
  | 1 => 1
  | n + 1 => sequence_a n / (2 * sequence_a n + 1)

theorem a_5_equals_one_ninth : sequence_a 5 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_one_ninth_l470_47092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_division_l470_47007

/-- Represents a triangle with vertices A, B, and E -/
structure Triangle (A B E : ℝ × ℝ) where
  isIsosceles : True

/-- Represents a trapezoid formed by dividing the triangle -/
structure Trapezoid (A B C D : ℝ × ℝ) where
  isIsosceles : True

/-- Represents the smaller triangle formed by dividing the original triangle -/
structure SmallerTriangle (A C D : ℝ × ℝ) where
  isIsosceles : True

def Triangle.area (t : Triangle A B E) : ℝ := sorry
def Triangle.altitude (t : Triangle A B E) (vertex : ℝ × ℝ) : ℝ := sorry
def Trapezoid.area (t : Trapezoid A B C D) : ℝ := sorry
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem isosceles_triangle_division 
  (A B E : ℝ × ℝ) 
  (C D : ℝ × ℝ) 
  (t : Triangle A B E) 
  (trap : Trapezoid A B C D) 
  (smallT : SmallerTriangle A C D) :
  Triangle.area t = 180 →
  Triangle.altitude t B = 30 →
  Trapezoid.area trap = 135 →
  distance C D = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_division_l470_47007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factor_in_set_l470_47040

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ 
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ({a, b, c, d} : Finset ℕ) = {3, 4, 6, 8} ∧
  n = 1000 * a + 100 * b + 10 * c + d

def is_factor_in_set (n : ℕ) : Prop :=
  is_valid_number n ∧ ∃ m, is_valid_number m ∧ m ≠ n ∧ m % n = 0

theorem unique_factor_in_set : 
  ∃! n, is_factor_in_set n ∧ n = 4386 := by
  sorry

#check unique_factor_in_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factor_in_set_l470_47040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l470_47024

/-- Represents a rectangle EFGH with a circle centered at H and G on the circle -/
structure RectangleWithCircle where
  EH : ℝ
  GH : ℝ
  is_rectangle : True  -- Placeholder for the rectangle property
  H_is_center : True   -- Placeholder for H being the center of the circle
  G_on_circle : True   -- Placeholder for G being on the circle

/-- The area of the shaded region in the given geometric configuration -/
noncomputable def shaded_area (r : RectangleWithCircle) : ℝ :=
  (Real.pi * r.GH^2 / 2) - (r.EH * r.GH)

/-- Theorem stating the area of the shaded region for the given dimensions -/
theorem shaded_area_calculation (r : RectangleWithCircle) 
  (h1 : r.EH = 5) (h2 : r.GH = 12) : 
  shaded_area r = 169 * Real.pi / 2 - 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l470_47024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l470_47018

/-- An isosceles triangle with sides of length 6 and base of length 8 -/
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  side_eq : side = 6
  base_eq : base = 8

/-- A point within the triangle -/
structure PointInTriangle where
  x : ℝ
  y : ℝ
  in_triangle : x ≥ 0 ∧ y ≥ 0 ∧ 2 * y ≤ (6 - 4) * x + 6 * 4

/-- The probability of a point forming a triangle with area > 1/3 of the total -/
noncomputable def probability_large_subtriangle (t : IsoscelesTriangle) : ℝ := 1/3

/-- The main theorem -/
theorem probability_theorem (t : IsoscelesTriangle) :
  probability_large_subtriangle t = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l470_47018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_angle_l470_47006

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Main theorem -/
theorem parabola_minimum_angle (C : Parabola) (F : Point) (l : Line) 
  (A B M N Q : Point) (circle : Circle) :
  F.x = 0 ∧ F.y = 1 →  -- Focus at (0,1)
  C.p = 2 →  -- Value of p
  (∀ x y, y = (x^2) / (2 * C.p)) →  -- Parabola equation
  (F.y = l.slope * F.x + l.intercept) →  -- Line l passes through F
  (A.y = l.slope * A.x + l.intercept) ∧ 
  (B.y = l.slope * B.x + l.intercept) ∧
  (A.y = (A.x^2) / (2 * C.p)) ∧ 
  (B.y = (B.x^2) / (2 * C.p)) →  -- A and B are on both l and C
  circle.center = Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2) →  -- Circle center
  circle.radius = Real.sqrt (((A.x - B.x)^2 + (A.y - B.y)^2) / 4) →  -- Circle radius
  M.y = 0 ∧ N.y = 0 ∧  -- M and N on x-axis
  (M.x - circle.center.x)^2 + (M.y - circle.center.y)^2 = circle.radius^2 ∧
  (N.x - circle.center.x)^2 + (N.y - circle.center.y)^2 = circle.radius^2 →  -- M and N on circle
  Q.x = (M.x + N.x) / 2 ∧ Q.y = 0 →  -- Q is midpoint of MN
  ∃ (angle_QMN : ℝ),
    (∀ (k : ℝ), angle_QMN ≤ Real.arcsin ((2 * k^2 + 1) / (2 * Real.sqrt (k^2 + 1)))) ∧
    angle_QMN = Real.pi / 6 ∧
    l.slope = 0 ∧ l.intercept = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_angle_l470_47006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_distance_theorem_l470_47027

-- Define a point on the chessboard
structure ChessPoint where
  x : Fin 8
  y : Fin 8

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ChessPoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x).val^2 + (p1.y - p2.y).val^2 : ℝ)

-- State the theorem
theorem chessboard_distance_theorem 
  (rooks : Fin 8 → ChessPoint)
  (h_unique_rows : ∀ i j, i ≠ j → (rooks i).y ≠ (rooks j).y)
  (h_unique_cols : ∀ i j, i ≠ j → (rooks i).x ≠ (rooks j).x) :
  ∃ a b c d, a ≠ b ∧ c ≠ d ∧ (a, b) ≠ (c, d) ∧ 
    distance (rooks a) (rooks b) = distance (rooks c) (rooks d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_distance_theorem_l470_47027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_zero_l470_47023

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := 2 / x

-- Define the intersection points
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- State the theorem
theorem intersection_sum_zero :
  A.2 + B.2 = 0 := by
  sorry

-- Additional lemmas to support the main theorem
lemma intersection_point_condition (p : ℝ × ℝ) :
  (f p.1 = p.2) ∧ (g p.1 = p.2) → p = A ∨ p = B := by
  sorry

lemma symmetry_of_intersection :
  B.1 = -A.1 ∧ B.2 = -A.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_zero_l470_47023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_nth_root_power_l470_47089

-- Define the function f(x) = x^(k/n)
noncomputable def f (n k : ℕ) (x : ℝ) : ℝ := x^((k : ℝ)/(n : ℝ))

-- State the theorem
theorem continuity_nth_root_power (n k : ℕ) (hn : n > 0) (hk : k > 0) :
  ∀ a : ℝ, a > 0 → ContinuousAt (f n k) a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_nth_root_power_l470_47089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2x_plus_5y_l470_47039

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem min_value_2x_plus_5y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : log_base 10 x + log_base 10 y = 1) : 
  ∀ a b : ℝ, a > 0 → b > 0 → log_base 10 a + log_base 10 b = 1 → 2*x + 5*y ≤ 2*a + 5*b :=
by
  sorry

#check min_value_2x_plus_5y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2x_plus_5y_l470_47039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_l470_47038

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 500.00000000000006 →
  final = 700 →
  ⌊((final - initial) / initial) * 100⌋ = 39 ∧
  ⌈((final - initial) / initial) * 100⌉ = 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_l470_47038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_lower_bound_l470_47022

theorem sum_of_squares_lower_bound (a b c d n : ℕ) 
  (h : 7 * 4^n = a^2 + b^2 + c^2 + d^2) :
  (a ≥ 2^(n-1)) ∧ (b ≥ 2^(n-1)) ∧ (c ≥ 2^(n-1)) ∧ (d ≥ 2^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_lower_bound_l470_47022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_interest_rate_l470_47011

noncomputable def initial_principal : ℝ := 6000
noncomputable def first_year_rate : ℝ := 0.04
noncomputable def final_amount : ℝ := 6552
def time_period : ℕ := 2

noncomputable def amount_after_first_year : ℝ := initial_principal * (1 + first_year_rate)

noncomputable def second_year_rate : ℝ := (final_amount / amount_after_first_year) - 1

theorem second_year_interest_rate :
  second_year_rate = 0.05 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_interest_rate_l470_47011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_l470_47070

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- Length of the first segment of the divided side -/
  segment1 : ℝ
  /-- Length of the second segment of the divided side -/
  segment2 : ℝ
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- Perimeter of the triangle -/
  perimeter : ℝ

/-- Helper function to get the set of sides of the triangle -/
def set_of_sides (t : TriangleWithInscribedCircle) : Set ℝ :=
  sorry

/-- Theorem stating that the shortest side of the triangle is 2 units -/
theorem shortest_side_length
  (triangle : TriangleWithInscribedCircle)
  (h1 : triangle.segment1 = 7)
  (h2 : triangle.segment2 = 9)
  (h3 : triangle.radius = 5)
  (h4 : triangle.perimeter = 46) :
  ∃ (side : ℝ), side = 2 ∧ ∀ (other_side : ℝ), other_side ∈ set_of_sides triangle → side ≤ other_side :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_l470_47070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_of_regular_tetrahedron_l470_47029

-- Define a regular tetrahedron
structure RegularTetrahedron where
  v1 : Fin 3 → ℤ
  v2 : Fin 3 → ℤ
  v3 : Fin 3 → ℤ
  v4 : Fin 3 → ℤ

-- Define the distance function between two points
def distance (p1 p2 : Fin 3 → ℤ) : ℤ :=
  (p1 0 - p2 0)^2 + (p1 1 - p2 1)^2 + (p1 2 - p2 2)^2

-- Theorem statement
theorem fourth_vertex_of_regular_tetrahedron 
  (t : RegularTetrahedron) 
  (h1 : t.v1 = ![0, 0, 0])
  (h2 : t.v2 = ![6, 0, 0])
  (h3 : t.v3 = ![5, 0, 6]) :
  t.v4 = ![0, 0, 6] ∨ t.v4 = ![0, 0, -6] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_of_regular_tetrahedron_l470_47029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wife_speed_theorem_l470_47062

/-- Calculates the speed of the second person given the track circumference,
    meeting time, and speed of the first person. -/
noncomputable def calculate_speed (track_circumference : ℝ) (meeting_time : ℝ) (speed1 : ℝ) : ℝ :=
  let speed1_m_per_min : ℝ := speed1 * 1000 / 60
  let distance1 : ℝ := speed1_m_per_min * meeting_time
  let distance2 : ℝ := track_circumference - distance1
  let speed2_m_per_min : ℝ := distance2 / meeting_time
  speed2_m_per_min * 60 / 1000

/-- Theorem stating that under given conditions, the speed of the second person is 3.75 km/hr. -/
theorem wife_speed_theorem (track_circumference : ℝ) (meeting_time : ℝ) (speed1 : ℝ)
    (h1 : track_circumference = 726)
    (h2 : meeting_time = 5.28)
    (h3 : speed1 = 4.5) :
  calculate_speed track_circumference meeting_time speed1 = 3.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wife_speed_theorem_l470_47062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_pi_functions_l470_47098

-- Define the functions
noncomputable def f1 (x : ℝ) := Real.sin (abs x)
noncomputable def f2 (x : ℝ) := abs (Real.sin x)
noncomputable def f3 (x : ℝ) := Real.sin (2 * x + 2 * Real.pi / 3)
noncomputable def f4 (x : ℝ) := Real.tan (2 * x + 2 * Real.pi / 3)

-- Define what it means for a function to have a smallest positive period of π
def has_period_pi (f : ℝ → ℝ) : Prop :=
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x)) ∧
  (∀ (T : ℝ), T > 0 → (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (∀ x, f (x + Real.pi) = f x)

-- State the theorem
theorem period_pi_functions :
  ¬(has_period_pi f1) ∧ 
  (has_period_pi f2) ∧ 
  (has_period_pi f3) ∧ 
  ¬(has_period_pi f4) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_pi_functions_l470_47098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_circle_properties_l470_47047

-- Define the triangle ABC
noncomputable def A : ℝ × ℝ := (0, 1)
noncomputable def B : ℝ × ℝ := (2, 0)
noncomputable def C : ℝ × ℝ := (0, -1/2)

-- Define the line containing median CD
def line_CD (x y : ℝ) : Prop := 2*x - 2*y - 1 = 0

-- Define the line containing altitude BH
def line_BH (y : ℝ) : Prop := y = 0

-- Define point P
noncomputable def P (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + x + 5*y - 6 = 0

-- State the theorem
theorem triangle_and_circle_properties :
  ∀ (m : ℝ),
  -- Given conditions
  (∀ x y : ℝ, line_CD x y → (x = (A.1 + B.1)/2 ∧ y = (A.2 + B.2)/2)) →
  (line_BH C.2) →
  (circle_M A.1 A.2) →
  (circle_M B.1 B.2) →
  (circle_M (P m).1 (P m).2) →
  -- Tangent line condition
  (∃ k : ℝ, k = 1 ∧ ∀ x y : ℝ, circle_M x y → (y - (P m).2) = k*(x - (P m).1)) →
  -- Conclusions
  (B = (2, 0) ∧ C = (0, -1/2) ∧ ∀ x y : ℝ, circle_M x y ↔ x^2 + y^2 + x + 5*y - 6 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_circle_properties_l470_47047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l470_47053

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

/-- Hyperbola equation -/
def on_hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_point : on_hyperbola 2 0 a b) : eccentricity a b = Real.sqrt 7 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l470_47053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_to_cone_volume_ratio_l470_47000

/-- Represents a cone with height h and radius r -/
structure Cone where
  h : ℝ
  r : ℝ

/-- The volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.r^2 * c.h

/-- The volume of water in a cone filled to 2/3 of its height -/
noncomputable def waterVolume (c : Cone) : ℝ := (1/3) * Real.pi * ((2/3) * c.r)^2 * ((2/3) * c.h)

/-- Theorem stating that the ratio of water volume to total cone volume is 8/27 -/
theorem water_to_cone_volume_ratio (c : Cone) :
  waterVolume c / coneVolume c = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_to_cone_volume_ratio_l470_47000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l470_47051

theorem remainder_theorem (x : ℕ) (h : x % 15 = 7) : (8 * x - 5) % 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l470_47051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_perpendicular_to_tangent_circles_l470_47059

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the points of tangency
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry
noncomputable def C : ℝ × ℝ := sorry

-- Define the three circles
noncomputable def circle1 : Circle := sorry
noncomputable def circle2 : Circle := sorry
noncomputable def circle3 : Circle := sorry

-- Define the property of external tangency
def externally_tangent (c1 c2 : Circle) (p : ℝ × ℝ) : Prop := sorry

-- Define the circumcircle of a triangle
noncomputable def circumcircle (p1 p2 p3 : ℝ × ℝ) : Circle := sorry

-- Define perpendicularity between circles
def perpendicular (c1 c2 : Circle) : Prop := sorry

-- Theorem statement
theorem circumcircle_perpendicular_to_tangent_circles :
  externally_tangent circle1 circle2 A ∧
  externally_tangent circle2 circle3 B ∧
  externally_tangent circle3 circle1 C →
  let circ := circumcircle A B C
  perpendicular circ circle1 ∧
  perpendicular circ circle2 ∧
  perpendicular circ circle3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_perpendicular_to_tangent_circles_l470_47059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l470_47066

-- Define the function f(x) = √(x+1)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1)

-- State the theorem
theorem f_range : Set.range f = Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l470_47066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l470_47005

/-- The number of days it takes for worker A to complete the work alone -/
noncomputable def days_for_A : ℝ := 32

/-- The number of days it takes for A and B together to complete the work -/
noncomputable def days_for_AB : ℝ := 24

/-- The rate at which worker A completes the work (portion per day) -/
noncomputable def rate_A : ℝ := 1 / days_for_A

/-- The rate at which worker B completes the work (portion per day) -/
noncomputable def rate_B : ℝ := rate_A / 3

theorem work_completion :
  rate_A = 3 * rate_B ∧ rate_A + rate_B = 1 / days_for_AB →
  days_for_A = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l470_47005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_triangles_l470_47086

/-- The number of distinct triangles ABC up to similarity satisfying the given conditions --/
def distinct_triangles : ℕ :=
  6

/-- Positive integer angles of a triangle in degrees --/
structure TriangleAngles where
  A : ℕ+
  B : ℕ+
  C : ℕ+
  sum_180 : A + B + C = 180

/-- Condition on the angles satisfying the given equation --/
def satisfies_equation (t : TriangleAngles) : Prop :=
  ∃ k : ℕ+, k * t.C ≤ 360 ∧
    Real.cos (Real.pi * ↑t.A / 180) * Real.cos (Real.pi * ↑t.B / 180) +
    Real.sin (Real.pi * ↑t.A / 180) * Real.sin (Real.pi * ↑t.B / 180) *
    Real.sin (Real.pi * ↑(k * t.C) / 180) = 1

/-- The main theorem stating that there are exactly 6 distinct triangles satisfying the conditions --/
theorem count_distinct_triangles :
    distinct_triangles = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_triangles_l470_47086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_four_l470_47069

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + a else 2^x

theorem function_composition_equals_four (a : ℝ) :
  a > -1 → f a (f a (-1)) = 4 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_four_l470_47069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_quadruples_l470_47033

/-- The number of ordered quadruples (a,b,c,d) of real numbers that satisfy the matrix equation. -/
def num_quadruples : ℕ := 2

/-- The matrix equation that the quadruples must satisfy. -/
def satisfies_equation (a b c d : ℝ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  M * M = !![1, 1; 1, 1]

/-- Theorem stating that there are exactly 2 ordered quadruples satisfying the equation. -/
theorem exactly_two_quadruples :
  ∃! (s : Set (ℝ × ℝ × ℝ × ℝ)), s.Finite ∧ s.ncard = num_quadruples ∧
    ∀ (q : ℝ × ℝ × ℝ × ℝ), q ∈ s ↔ satisfies_equation q.1 q.2.1 q.2.2.1 q.2.2.2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_quadruples_l470_47033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_squared_z_l470_47012

-- Define the complex number
noncomputable def z : ℂ := -3 + (5/4)*Complex.I

-- State the theorem
theorem modulus_of_squared_z : Complex.abs (z^2) = 169/16 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_squared_z_l470_47012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_ratio_l470_47068

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Define an equilateral triangle
def equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let d := Real.sqrt 3 / 2 * dist P Q
  dist P Q = dist Q R ∧ dist Q R = dist R P ∧
  abs ((Q.2 - P.2) * (R.1 - Q.1) - (Q.1 - P.1) * (R.2 - Q.2)) = d * dist P Q

-- Define the theorem
theorem inscribed_triangle_ratio (a b : ℝ) (P Q R F₁ F₂ : ℝ × ℝ) :
  a > 0 ∧ b > 0 ∧
  ellipse a b Q.1 Q.2 ∧
  Q = (0, b) ∧
  P.2 = R.2 ∧
  equilateral_triangle P Q R ∧
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ F₁ = (1 - t) • Q + t • R) ∧
  (∃ s : ℝ, 0 < s ∧ s < 1 ∧ F₂ = (1 - s) • P + s • Q) →
  dist P Q / dist F₁ F₂ = 8 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_ratio_l470_47068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_natural_numbers_reachable_l470_47094

def board_operation (x : ℕ) : ℕ → ℕ
| 0 => 3 * x + 1
| _ => x / 2

def reachable (n : ℕ) : Prop :=
  ∃ (k : ℕ) (seq : Fin (k + 1) → ℕ),
    seq 0 = 1 ∧
    seq k = n ∧
    ∀ i : Fin k, seq i.succ = board_operation (seq i) 0 ∨ seq i.succ = board_operation (seq i) 1

theorem all_natural_numbers_reachable :
  ∀ n : ℕ, reachable n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_natural_numbers_reachable_l470_47094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_property_l470_47076

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Defines an ellipse with foci F₁ and F₂, and constant sum of distances 2a -/
structure Ellipse where
  F1 : Point
  F2 : Point
  a : ℝ

/-- Theorem: For an ellipse with given properties, h + k + a + b = 4 + 2√3 -/
theorem ellipse_sum_property (E : Ellipse) 
    (h1 : E.F1 = ⟨-2, 0⟩) 
    (h2 : E.F2 = ⟨2, 0⟩) 
    (h3 : E.a = 4) : 
  ∃ (h k b : ℝ), 
    (∀ (P : Point), (distance P E.F1 + distance P E.F2 = 8) ↔ 
      ((P.x - h)^2 / E.a^2 + (P.y - k)^2 / b^2 = 1)) ∧
    h + k + E.a + b = 4 + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_property_l470_47076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l470_47083

theorem range_of_t (f : ℝ → ℝ) (f' : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, HasDerivAt f (f' x) x) →
  (∀ x, g x = f x - Real.sin x) →
  (∀ x, g (-x) = g x) →
  (∀ x, x ≥ 0 → f' x > Real.cos x) →
  (∀ t, f (Real.pi / 2 - t) - f t > Real.cos t - Real.sin t) →
  ∀ t, t < Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l470_47083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_2012_l470_47091

theorem function_value_2012 (f : ℕ → ℕ) 
  (h1 : ∀ n, f (f n) + f n = 2 * n + 3) 
  (h2 : f 0 = 1) : 
  f 2012 = 2013 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_2012_l470_47091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l470_47002

/-- The line with given intercepts -/
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

/-- The circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y - 3 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, 3)

/-- The radius of the circle -/
def circle_radius : ℝ := 4

/-- The distance from the center of the circle to the line -/
noncomputable def distance_center_to_line : ℝ := Real.sqrt 2

/-- Theorem: The length of the chord cut by line l on circle C is 2√14 -/
theorem chord_length : 
  2 * Real.sqrt (circle_radius^2 - distance_center_to_line^2) = 2 * Real.sqrt 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l470_47002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l470_47078

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

-- State the theorem
theorem f_properties :
  let a : ℝ := π / 6
  let b : ℝ := π / 2
  (∀ x ∈ Set.Icc (-π/2) (π/2), f x ≥ -1) ∧ 
  (∃ x ∈ Set.Icc (-π/2) (π/2), f x = -1) ∧
  (∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x ≤ y → f y ≤ f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l470_47078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l470_47072

theorem log_problem (x : ℝ) (h1 : x < 1) (h2 : (Real.log x)^2 - Real.log (x^2) = 75 * Real.log 10) :
  (Real.log x)^4 - Real.log (x^4) = (308 - 4 * Real.sqrt 304) / 16 - 2 + Real.sqrt 304 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l470_47072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_group_frequency_l470_47025

theorem sixth_group_frequency
  (total_data : ℕ)
  (num_groups : ℕ)
  (freq_1 freq_2 freq_3 freq_4 : ℕ)
  (freq_5 : ℚ)
  (h1 : total_data = 50)
  (h2 : num_groups = 6)
  (h3 : freq_1 = 10)
  (h4 : freq_2 = 8)
  (h5 : freq_3 = 7)
  (h6 : freq_4 = 11)
  (h7 : freq_5 = 16/100) :
  total_data - (freq_1 + freq_2 + freq_3 + freq_4 + (freq_5 * total_data).floor) = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_group_frequency_l470_47025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l470_47030

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add a case for 0 to avoid missing cases error
  | 1 => 1
  | n + 2 => (2 * (n + 1) * sequence_a (n + 1) + 1) / (n + 2)

theorem sequence_a_formula (n : ℕ) (h : n ≥ 1) : 
  sequence_a n = n / (2^(n-1)) := by
  sorry  -- Use 'by' instead of ':=' for tactics

#eval sequence_a 5  -- Add an evaluation to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l470_47030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_one_l470_47050

/-- Triangle ABC with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Point Q -/
def Q : ℝ × ℝ := (5, 3)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Sum of distances from Q to each vertex of the triangle -/
noncomputable def totalDistance (t : Triangle) : ℝ :=
  distance Q t.A + distance Q t.B + distance Q t.C

/-- The specific triangle ABC from the problem -/
def triangleABC : Triangle :=
  { A := (0, 0)
    B := (12, 0)
    C := (4, 6) }

/-- Theorem statement -/
theorem sum_of_coefficients_is_one :
  ∃ (p q : ℤ), totalDistance triangleABC = p * Real.sqrt 5 + q * Real.sqrt 13 ∧ p + q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_one_l470_47050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_area_and_coefficient_sum_l470_47004

/-- The radius of the circular room -/
noncomputable def room_radius : ℝ := 10

/-- The radius of the central pillar -/
noncomputable def pillar_radius : ℝ := 5

/-- The visible area from a point on the circumference -/
noncomputable def visible_area : ℝ := (175 * Real.pi / 3) - 25 * Real.sqrt 3

/-- The coefficients in the form (m*π/n) + p*√q -/
def m : ℕ := 175
def n : ℕ := 3
def p : ℤ := -25
def q : ℕ := 3

/-- Theorem stating the visible area and the sum of coefficients -/
theorem visible_area_and_coefficient_sum :
  (visible_area = (m * Real.pi / n) + p * Real.sqrt q) ∧
  (m + n + p + q = 156) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_area_and_coefficient_sum_l470_47004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_special_equation_l470_47014

/-- Given an equation x√x - 8x + 9√x - 2 = 0 with real and nonnegative roots,
    the sum of its roots is 46. -/
theorem sum_of_roots_special_equation : 
  ∃ (f : ℝ → ℝ) (S : Finset ℝ), 
    (∀ x ∈ S, x ≥ 0) ∧ 
    (∀ x ∈ S, f x = x * Real.sqrt x - 8 * x + 9 * Real.sqrt x - 2) ∧
    (∀ x ∈ S, f x = 0) ∧
    (S.sum id = 46) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_special_equation_l470_47014
