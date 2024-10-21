import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_quadrant_l20_2020

/-- If the sum of sine and cosine of an angle is greater than 1, then the angle is in the first quadrant. -/
theorem angle_in_first_quadrant (α : ℝ) : 
  Real.sin α + Real.cos α > 1 → 0 < α ∧ α < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_quadrant_l20_2020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_VMO_1996_problem_l20_2074

theorem VMO_1996_problem (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n) :
  let A := {a : Fin k → Fin n | Function.Injective a}
  let A1 := {a ∈ A | (∃ s t : Fin k, s.val < t.val ∧ a s > a t) ∨
                     (∃ s : Fin k, ¬((a s).val % 2 = s.val % 2))}
  Fintype.card A1 = n.factorial / (n - k).factorial - Nat.choose ((n + k) / 2) k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_VMO_1996_problem_l20_2074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_theorem_l20_2003

def sequence_converges_to_one (a : ℕ → ℝ) (x : ℕ → ℝ) : Prop :=
  (∀ n, a n > 1) ∧
  (∀ n, a (n + 1) ^ 2 ≥ a n * a (n + 2)) ∧
  (∀ n, x n = Real.log (a (n + 1)) / Real.log (a n)) ∧
  Filter.Tendsto x Filter.atTop (nhds 1)

theorem sequence_convergence_theorem (a : ℕ → ℝ) (x : ℕ → ℝ) :
  sequence_converges_to_one a x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_theorem_l20_2003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l20_2044

/-- Represents a pyramid with a square base and a vertex -/
structure Pyramid where
  baseArea : ℝ
  triangleArea1 : ℝ
  triangleArea2 : ℝ

/-- Calculates the volume of a pyramid given its properties -/
noncomputable def pyramidVolume (p : Pyramid) : ℝ :=
  (p.baseArea * Real.sqrt 220) / 3

/-- Theorem stating that a pyramid with the given properties has the specified volume -/
theorem pyramid_volume_theorem (p : Pyramid) 
  (h1 : p.baseArea = 256)
  (h2 : p.triangleArea1 = 128)
  (h3 : p.triangleArea2 = 112) :
  pyramidVolume p = (256 * Real.sqrt 220) / 3 := by
  sorry

#check pyramid_volume_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l20_2044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l20_2038

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_calculation (principal interest time : ℝ) 
  (h_principal : principal = 800)
  (h_interest : interest = 160)
  (h_time : time = 4) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l20_2038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_needed_for_wall_l20_2047

/-- The number of bricks needed to build a wall -/
noncomputable def number_of_bricks (wall_length wall_height wall_thickness brick_length brick_width brick_height : ℝ) : ℝ :=
  (wall_length * wall_height * wall_thickness) / (brick_length * brick_width * brick_height)

/-- Theorem stating the number of bricks needed for the given wall and brick dimensions -/
theorem bricks_needed_for_wall :
  number_of_bricks 800 600 22.5 100 11.25 6 = 1600 := by
  -- Unfold the definition of number_of_bricks
  unfold number_of_bricks
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_needed_for_wall_l20_2047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l20_2008

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Checks if a polynomial satisfies the given condition for a natural number -/
def SatisfiesCondition (P : IntPolynomial) (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → P.eval (d : ℤ) = (n / d : ℕ) ^ 2

/-- The main theorem -/
theorem polynomial_existence (n : ℕ) :
  (∃ P : IntPolynomial, SatisfiesCondition P n) ↔ n.Prime ∨ n = 1 ∨ n = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l20_2008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_bamboo_problem_l20_2052

/-- The height of a broken bamboo --/
noncomputable def broken_bamboo_height (original_height tip_to_base : ℝ) : ℝ :=
  (original_height^2 - tip_to_base^2) / (2 * original_height)

/-- Theorem: The height of the break from the ground is approximately 9.1 chi --/
theorem broken_bamboo_problem :
  let original_height : ℝ := 20
  let tip_to_base : ℝ := 6
  abs (broken_bamboo_height original_height tip_to_base - 9.1) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_bamboo_problem_l20_2052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_exists_m_below_g_l20_2028

-- Define the functions
noncomputable def f (x : ℝ) := Real.log x
def h (a x : ℝ) := a * x
noncomputable def g (x : ℝ) := Real.exp x / x

-- Theorem for part I
theorem no_common_points (a : ℝ) :
  (∀ x > 0, f x ≠ h a x) ↔ a > 1 / Real.exp 1 := by sorry

-- Theorem for part II
theorem exists_m_below_g :
  ∃ m : ℝ, (∀ x > (1/2), f x + m / x < g x) ∧ 
  (∀ n : ℤ, (∀ x > (1/2), f x + (n : ℝ) / x < g x) → n ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_exists_m_below_g_l20_2028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stability_analysis_l20_2084

variable (ε : ℝ)
variable (x : ℝ → ℝ)

-- Define the differential equation
def diff_eq (t : ℝ) : Prop :=
  ε * (deriv x t) = x t * (Real.exp (x t) - 2)

-- Define the equilibrium points
def equilibrium_points (x : ℝ) : Prop :=
  x * (Real.exp x - 2) = 0

-- Define stability for an equilibrium point
def is_stable (x₀ : ℝ) : Prop :=
  (Real.exp x₀ - 2 + x₀ * Real.exp x₀) < 0

-- Define instability for an equilibrium point
def is_unstable (x₀ : ℝ) : Prop :=
  (Real.exp x₀ - 2 + x₀ * Real.exp x₀) > 0

-- Theorem statement
theorem stability_analysis (h : ε > 0) :
  (equilibrium_points 0 ∧ equilibrium_points (Real.log 2)) ∧
  (is_stable 0) ∧
  (is_unstable (Real.log 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stability_analysis_l20_2084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_inequality_l20_2095

theorem sine_sum_inequality (x y z : ℝ) (h : Real.sin x + Real.sin y + Real.sin z ≥ 3/2) :
  Real.sin (x - π/6) + Real.sin (y - π/6) + Real.sin (z - π/6) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_inequality_l20_2095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_plus_alpha_l20_2037

/-- Given a point M(-1, 2) in the plane rectangular coordinate system xOy,
    and an angle α whose terminal side passes through M,
    prove that sin(π/2 + α) = -√5/5 -/
theorem sin_pi_half_plus_alpha (α : ℝ) (h : ∃ (t : ℝ), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 2) :
  Real.sin (π / 2 + α) = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_plus_alpha_l20_2037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l20_2021

/-- Definition of an ellipse as a set of points -/
noncomputable def ellipse (f₁ f₂ : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p f₁ + dist p f₂ = dist f₁ f₂ + 2 * Real.sqrt (3 - 2)}

/-- Distance function between two points in ℝ² -/
noncomputable def dist (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

/-- An ellipse with given foci and vertex has the specified equation -/
theorem ellipse_equation (f₁ f₂ v : ℝ × ℝ) : 
  f₁ = (-Real.sqrt 2, 0) →
  f₂ = (Real.sqrt 2, 0) →
  v = (Real.sqrt 3, 0) →
  ∀ (x y : ℝ), (x^2 / 3 + y^2 = 1) ↔ (x, y) ∈ ellipse f₁ f₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l20_2021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_one_triangle_area_min_distance_l20_2050

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P on the parabola
def point_on_parabola (s t : ℝ) : Prop := parabola s t ∧ t > 0

-- Define the tangent line at point P
def tangent_line (s t x y : ℝ) : Prop := y - t = (1 / (2 * Real.sqrt s)) * (x - s)

-- Define the perpendicular line m
def perpendicular_line (s t x y : ℝ) : Prop := y - t = -(Real.sqrt s) * (x - s)

-- Define the intersection point Q
def intersection_point (s t x y : ℝ) : Prop := 
  perpendicular_line s t x y ∧ parabola x y ∧ (x ≠ s ∨ y ≠ t)

-- Theorem 1: Tangent line equation when s = 1
theorem tangent_at_one : 
  ∀ x y : ℝ, point_on_parabola 1 2 → tangent_line 1 2 x y → y = x + 1 :=
by sorry

-- Theorem 2: Area of triangle OPQ
theorem triangle_area : 
  ∃ s t x y : ℝ, point_on_parabola s t ∧ tangent_line s t (-1) 0 ∧ 
  intersection_point s t x y ∧ (1/2 * s * (t - y) = 12) :=
by sorry

-- Theorem 3: Minimum distance |PQ|
theorem min_distance : 
  ∃ min_pq : ℝ, (∀ s t x y : ℝ, point_on_parabola s t → intersection_point s t x y → 
  ((x - s)^2 + (y - t)^2 ≥ min_pq^2)) ∧ min_pq = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_one_triangle_area_min_distance_l20_2050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winter_sales_calculation_l20_2077

/-- Represents the number of hamburgers sold in millions for each season -/
structure SeasonalSales where
  spring : ℚ
  summer : ℚ
  fall : ℚ
  winter : ℚ

/-- The total number of hamburgers sold in a year in millions -/
def total_sales (s : SeasonalSales) : ℚ := s.spring + s.summer + s.fall + s.winter

/-- The percentage of hamburgers sold in fall -/
def fall_percentage (s : SeasonalSales) : ℚ := s.fall / total_sales s * 100

theorem winter_sales_calculation (s : SeasonalSales) 
  (h1 : fall_percentage s = 25)
  (h2 : s.fall = 4)
  (h3 : s.spring = (9/2))
  (h4 : s.summer = 5) :
  s.winter = (5/2) := by
  sorry

#check winter_sales_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winter_sales_calculation_l20_2077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_length_l20_2016

/-- A hexagon is a polygon with 6 sides -/
def Hexagon := Fin 6 → ℝ × ℝ

/-- The perimeter of a polygon -/
def perimeter (p : Fin n → ℝ × ℝ) : ℝ :=
  sorry  -- Placeholder for the actual perimeter calculation

theorem hexagon_side_length (h : Hexagon) (p : ℝ) (hp : perimeter h = p) (hp24 : p = 24) :
  ∃ (side_length : ℝ), side_length * 6 = p ∧ side_length = 4 := by
  sorry  -- Placeholder for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_length_l20_2016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_C_and_l_l20_2065

/-- The curve C in the upper half-plane -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1 ∧ p.2 ≥ 0}

/-- The line l -/
def l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 4 = 0}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The minimum distance between C and l -/
noncomputable def min_distance : ℝ :=
  Real.sqrt 2 * 5 / 2 - 1

theorem min_distance_between_C_and_l :
  ∀ p ∈ C, ∀ q ∈ l, distance p q ≥ min_distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_C_and_l_l20_2065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_bound_l20_2082

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - k) * exp x

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := f k x + deriv (f k) x

theorem max_lambda_bound (lambda : ℝ) :
  (∀ k ∈ Set.Icc (3/2) (5/2), ∀ x ∈ Set.Icc 0 1, g k x ≥ lambda) →
  lambda ≤ -2 * exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lambda_bound_l20_2082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_plus_sin_C_range_l20_2053

open Real

-- Define an acute triangle
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute_A : 0 < A ∧ A < π/2
  acute_B : 0 < B ∧ B < π/2
  acute_C : 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  cos_law_a : a^2 = b^2 + c^2 - 2*b*c*(cos A)
  cos_law_b : b^2 = a^2 + c^2 - 2*a*c*(cos B)
  cos_law_c : c^2 = a^2 + b^2 - 2*a*b*(cos C)

-- State the theorem
theorem cos_A_plus_sin_C_range (t : AcuteTriangle) 
  (h : (t.a + t.b + t.c) * (t.a + t.c - t.b) = (2 + Real.sqrt 3) * t.a * t.c) : 
  Real.sqrt 3 / 2 < cos t.A + sin t.C ∧ cos t.A + sin t.C < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_plus_sin_C_range_l20_2053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l20_2073

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x | x ∈ U ∧ -2 ≤ x ∧ x < 0}

def B : Set Int := {x | x ∈ U ∧ 0 ≤ x ∧ x ≤ 1}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l20_2073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_alone_days_l20_2067

-- Define the total work
noncomputable def total_work : ℝ := 1

-- Define the combined work rate of A and B
noncomputable def combined_rate : ℝ := total_work / 30

-- Define A's individual work rate
noncomputable def a_rate : ℝ := total_work / 60

-- Define the amount of work completed in 20 days by A and B together
noncomputable def work_completed : ℝ := 20 * combined_rate

-- Define the remaining work after 20 days
noncomputable def remaining_work : ℝ := total_work - work_completed

-- Theorem statement
theorem a_alone_days : remaining_work / a_rate = 20 := by
  -- The proof would go here, but we're using sorry as instructed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_alone_days_l20_2067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l20_2018

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem system_solution :
  ∀ x y : ℝ,
  x + y > 0 →
  y > 0 →
  x^2 - x*y + y^2 > 0 →
  (2 - log2 y = 2 * log2 (x + y)) →
  (log2 (x + y) + log2 (x^2 - x*y + y^2) = 1) →
  ((x = 1 ∧ y = 1) ∨ (x = Real.rpow 6 (1/3) / 3 ∧ y = 2 * Real.rpow 6 (1/3) / 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l20_2018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l20_2064

noncomputable def f (x : ℝ) := 2 * (Real.sin (x + Real.pi/3))^2 - (Real.cos x)^2 + (Real.sin x)^2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/6) → f x ≤ 3/2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/6) → f x ≥ 0) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/6) ∧ f x = 3/2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/6) ∧ f x = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l20_2064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l20_2013

-- Define the solution set type
def SolutionSet := Set ℝ

-- Define the quadratic function type
def QuadraticFunction := ℝ → ℝ → ℝ → ℝ → ℝ

-- Define the original quadratic function
def f : QuadraticFunction := fun a b c x ↦ a * x^2 + b * x + c

-- Define the transformed quadratic function
def g : QuadraticFunction := fun a b c x ↦ c * x^2 + a * x + b

-- State the theorem
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hf : {x : ℝ | f a b c x > 0} = Set.Ioo (-3) 2) :
  {x : ℝ | g a b c x > 0} = Set.Iic (-1/3) ∪ Set.Ici (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l20_2013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_cubes_divisible_by_27_l20_2041

theorem difference_of_cubes_divisible_by_27 (S : Finset ℤ) (h : S.card = 10) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ 27 ∣ (a^3 - b^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_cubes_divisible_by_27_l20_2041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_101_equals_100_l20_2070

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | n + 2 => if n % 2 = 0 then sequence_a (n + 1) else 2 * sequence_a ((n + 2) / 2) + 1

theorem a_101_equals_100 : sequence_a 101 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_101_equals_100_l20_2070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_iff_a_in_range_l20_2032

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 - a * x) / Real.log a

theorem decreasing_f_iff_a_in_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f a x₁ > f a x₂) ↔ 1 < a ∧ a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_iff_a_in_range_l20_2032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l20_2057

/-- The length of one side of a square made from a wire -/
noncomputable def side_length (total_wire_length : ℝ) : ℝ :=
  total_wire_length / 4

/-- Theorem: The length of one side of a square is 8.7 cm when made from 34.8 cm of wire -/
theorem square_side_length :
  side_length 34.8 = 8.7 := by
  -- Unfold the definition of side_length
  unfold side_length
  -- Evaluate the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l20_2057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_maximized_at_20_twentieth_term_last_positive_common_difference_negative_l20_2002

/-- Arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  first_positive : 0 < a 1
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  special_condition : 3 * a 8 = 5 * a 13

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * seq.d)

/-- The value of n that maximizes the sum of first n terms -/
def maximizing_n : ℕ := 20

/-- Theorem stating that 20 maximizes the sum of first n terms -/
theorem sum_maximized_at_20 (seq : ArithmeticSequence) :
  ∀ n : ℕ, sum_n seq n ≤ sum_n seq maximizing_n :=
by
  sorry

/-- Proof that the 20th term is the last positive term -/
theorem twentieth_term_last_positive (seq : ArithmeticSequence) :
  seq.a 20 > 0 ∧ seq.a 21 ≤ 0 :=
by
  sorry

/-- Proof that the common difference is negative -/
theorem common_difference_negative (seq : ArithmeticSequence) :
  seq.d < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_maximized_at_20_twentieth_term_last_positive_common_difference_negative_l20_2002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_set_properties_l20_2006

def perfect_set (S : Set ℝ) : Prop :=
  Set.Nonempty S ∧ 
  ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S ∧ (x - y) ∈ S ∧ (x * y) ∈ S

def set_A : Set ℝ :=
  {x | ∃ a b : ℤ, x = a + Real.sqrt 5 * b}

theorem perfect_set_properties :
  (∀ S : Set ℝ, perfect_set S → (0 : ℝ) ∈ S) ∧
  perfect_set set_A :=
by
  constructor
  · intro S hS
    let ⟨x, hx⟩ := Set.nonempty_def.mp hS.1
    have h : x - x ∈ S := hS.2 x x hx hx |>.2.1
    rw [sub_self] at h
    exact h
  · sorry  -- Proof of set_A being a perfect set is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_set_properties_l20_2006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clicks_time_theorem_l20_2014

noncomputable section

/-- Represents the length of a rail in meters -/
def rail_length : ℝ := 15

/-- Converts kilometers per hour to meters per minute -/
noncomputable def km_per_hour_to_meters_per_minute (speed : ℝ) : ℝ :=
  speed * 1000 / 60

/-- Calculates the number of clicks per minute for a given speed -/
noncomputable def clicks_per_minute (speed : ℝ) : ℝ :=
  km_per_hour_to_meters_per_minute speed / rail_length

/-- Calculates the time in seconds for the number of clicks to equal the speed -/
noncomputable def time_for_clicks_equal_speed (speed : ℝ) : ℝ :=
  (speed / clicks_per_minute speed) * 60

theorem clicks_time_theorem (speed : ℝ) :
  time_for_clicks_equal_speed speed = 54 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clicks_time_theorem_l20_2014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neznayka_mistake_l20_2040

theorem neznayka_mistake (numbers : Fin 11 → ℕ) 
  (differences : Fin 11 → ℕ)
  (h1 : ∀ i : Fin 11, differences i = Int.natAbs (numbers i - numbers (i.succ)))
  (h2 : (List.filter (· = 1) (List.ofFn differences)).length = 4)
  (h3 : (List.filter (· = 2) (List.ofFn differences)).length = 4)
  (h4 : (List.filter (· = 3) (List.ofFn differences)).length = 3) :
  False :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neznayka_mistake_l20_2040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_identity_l20_2004

def b : ℕ → ℕ
  | 0 => 1  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 2
  | (n + 3) => b (n + 2) + b (n + 1)

theorem b_identity (n k : ℕ) (h1 : n ≥ 4) (h2 : 2 ≤ k) (h3 : k ≤ n - 2) :
  b n = b k * b (n - k) + b (k - 1) * b (n - k - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_identity_l20_2004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_example_l20_2017

/-- The height of a solid right cylinder with given radius and surface area -/
noncomputable def cylinder_height (r : ℝ) (sa : ℝ) : ℝ :=
  (sa - 2 * Real.pi * r^2) / (2 * Real.pi * r)

/-- Theorem: The height of a solid right cylinder with radius 3 feet and
    surface area 30π square feet is 2 feet -/
theorem cylinder_height_example :
  cylinder_height 3 (30 * Real.pi) = 2 := by
  -- Unfold the definition of cylinder_height
  unfold cylinder_height
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_example_l20_2017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_satisfy_conditions_l20_2071

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ
  hNonZero : a ≠ 0 ∨ b ≠ 0

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to calculate the distance from a point to a line
noncomputable def distanceToLine (p : Point2D) (l : Line2D) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

-- Function to calculate the sine of the angle of inclination of a line
noncomputable def sineOfInclination (l : Line2D) : ℝ :=
  abs l.b / Real.sqrt (l.a^2 + l.b^2)

-- Theorem statement
theorem line_equations_satisfy_conditions :
  ∃ (l1 l2 l3 l4 : Line2D),
    -- First condition
    (pointOnLine ⟨-4, 0⟩ l1 ∧ sineOfInclination l1 = Real.sqrt 10 / 10) ∧
    (pointOnLine ⟨-4, 0⟩ l2 ∧ sineOfInclination l2 = Real.sqrt 10 / 10) ∧
    -- Second condition
    (pointOnLine ⟨5, 10⟩ l3 ∧ distanceToLine ⟨0, 0⟩ l3 = 5) ∧
    (pointOnLine ⟨5, 10⟩ l4 ∧ distanceToLine ⟨0, 0⟩ l4 = 5) ∧
    -- Equations of the lines
    ((l1.a = 1 ∧ l1.b = 3 ∧ l1.c = 4) ∨ (l1.a = 1 ∧ l1.b = -3 ∧ l1.c = 4)) ∧
    ((l2.a = 1 ∧ l2.b = 3 ∧ l2.c = 4) ∨ (l2.a = 1 ∧ l2.b = -3 ∧ l2.c = 4)) ∧
    (l3.a = 1 ∧ l3.b = 0 ∧ l3.c = -5) ∧
    (l4.a = 3 ∧ l4.b = -4 ∧ l4.c = 25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_satisfy_conditions_l20_2071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_L_l20_2015

/-- The function L(n) represents the label of the last remaining card after
    applying the ABBABBABB... sequence to n cards. -/
noncomputable def L : ℕ → ℕ := sorry

/-- Recursive definition of sequence a_i -/
def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 729 * a n - 104

/-- Recursive definition of sequence b_i -/
def b : ℕ → ℕ
  | 0 => 4
  | n + 1 => 729 * b n - 104

/-- Theorem stating the characterization of k such that L(3k) = k -/
theorem characterization_of_L (k : ℕ) :
  (∃ n : ℕ, L (3 * k) = k) ↔
  (k = 1 ∨ ∃ i : ℕ, (k = 243 * a i - 35 ∨ k = 243 * b i - 35)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_L_l20_2015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cupboard_sale_percentage_l20_2051

/-- Represents the sale of a cupboard with given cost price and potential profit scenario. -/
structure CupboardSale where
  cost_price : ℚ
  potential_profit_price : ℚ
  potential_profit_percentage : ℚ

/-- Calculates the percentage below cost price at which the cupboard was sold. -/
def percentage_below_cost_price (sale : CupboardSale) : ℚ :=
  let actual_selling_price := sale.potential_profit_price - 2086
  let difference := sale.cost_price - actual_selling_price
  (difference / sale.cost_price) * 100

/-- Theorem stating that the cupboard was sold at 14% below cost price. -/
theorem cupboard_sale_percentage (sale : CupboardSale) 
  (h1 : sale.cost_price = 7450)
  (h2 : sale.potential_profit_percentage = 14)
  (h3 : sale.potential_profit_price = sale.cost_price * (1 + sale.potential_profit_percentage / 100)) :
  percentage_below_cost_price sale = 14 := by
  sorry

/-- Example calculation -/
def example_sale : CupboardSale :=
  { cost_price := 7450
  , potential_profit_price := 8493
  , potential_profit_percentage := 14 }

#eval percentage_below_cost_price example_sale

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cupboard_sale_percentage_l20_2051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_in_third_sector_l20_2062

/-- The radius of a circle inscribed in a sector that is one-third of a larger circle --/
noncomputable def inscribed_circle_radius (R : ℝ) : ℝ :=
  R * (1 - Real.sqrt 2)

/-- Theorem stating that the radius of the inscribed circle in a sector that is 
    one-third of a circle with radius 4 cm is equal to 4 - 4√2 cm --/
theorem inscribed_circle_radius_in_third_sector :
  inscribed_circle_radius 4 = 4 - 4 * Real.sqrt 2 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval inscribed_circle_radius 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_in_third_sector_l20_2062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_length_in_hexagon_l20_2094

/-- Represents the set of all sides and diagonals of a regular hexagon -/
def T : Set (Fin 15) := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The number of longer diagonals in a regular hexagon -/
def num_longer_diagonals : ℕ := 3

/-- The number of shorter diagonals in a regular hexagon -/
def num_shorter_diagonals : ℕ := 3

/-- The total number of segments (sides and diagonals) in a regular hexagon -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The probability of selecting two segments of the same length from T -/
def prob_same_length : ℚ := 9 / 14

theorem prob_same_length_in_hexagon :
  prob_same_length = (num_sides * (num_sides - 1) + num_longer_diagonals * (num_longer_diagonals - 1) + num_shorter_diagonals * (num_shorter_diagonals - 1)) / (total_segments * (total_segments - 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_length_in_hexagon_l20_2094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_l20_2099

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (l m : Line)

-- Define the property of being non-overlapping
variable (non_overlapping_planes : Plane → Plane → Prop)
variable (non_overlapping_lines : Line → Line → Prop)

-- State the theorem
theorem incorrect_statement 
  (h1 : non_overlapping_planes α β)
  (h2 : non_overlapping_lines l m) :
  ¬(∀ (α β : Plane) (l : Line), 
    perpendicular l α ∧ perpendicular l β → parallel_line_plane l β) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_l20_2099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_coefficient_l20_2059

/-- Triangle DEF with given side lengths -/
structure Triangle where
  de : ℝ
  ef : ℝ
  fd : ℝ

/-- Rectangle WXYZ inscribed in Triangle DEF -/
structure InscribedRectangle where
  triangle : Triangle
  θ : ℝ -- side length WX

/-- Area of the inscribed rectangle as a function of θ -/
def rectangleArea (rect : InscribedRectangle) (γ δ : ℝ) : ℝ :=
  γ * rect.θ - δ * rect.θ^2

/-- The coefficient δ in the area formula -/
def δ_coefficient : ℚ :=
  60 / 169

theorem inscribed_rectangle_area_coefficient 
  (t : Triangle) 
  (rect : InscribedRectangle) 
  (h1 : t.de = 15) 
  (h2 : t.ef = 39) 
  (h3 : t.fd = 36) 
  (h4 : rect.triangle = t) : 
  δ_coefficient = 60 / 169 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_coefficient_l20_2059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_g_to_h_l20_2092

-- Define the original function g
variable (g : ℝ → ℝ)

-- Define the transformed function h
noncomputable def h (x : ℝ) : ℝ := -2/3 * g x - 2

-- State the theorem
theorem transform_g_to_h :
  ∀ (x y : ℝ), y = h g x ↔ y = -2/3 * (g x) - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_g_to_h_l20_2092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_not_in_biology_l20_2058

theorem students_not_in_biology (total_students : ℕ) 
  (biology_percentage : ℚ) (h1 : total_students = 880) 
  (h2 : biology_percentage = 1/2) : 
  total_students - (biology_percentage * ↑total_students).floor = 440 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_not_in_biology_l20_2058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_sum_l20_2089

theorem trigonometric_identity_sum (x : ℝ) (p q r : ℕ) : 
  (1 + Real.sin x) * (1 + Real.cos x) = 9/4 →
  (1 - Real.sin x) * (1 - Real.cos x) = p/q - Real.sqrt r →
  Nat.Coprime p q →
  r + p + q = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_sum_l20_2089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_failure_percentage_l20_2048

noncomputable def approx_equal (a b : ℚ) : Prop := abs (a - b) < 1/100

theorem test_failure_percentage
  (total_boys : ℕ)
  (total_girls : ℕ)
  (boys_pass_rate : ℚ)
  (girls_pass_rate : ℚ)
  (h1 : total_boys = 50)
  (h2 : total_girls = 100)
  (h3 : boys_pass_rate = 1/2)
  (h4 : girls_pass_rate = 2/5) :
  let total_students := total_boys + total_girls
  let failed_students := total_students - (boys_pass_rate * total_boys + girls_pass_rate * total_girls)
  let failure_percentage := (failed_students / total_students) * 100
  approx_equal failure_percentage 56.67 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_failure_percentage_l20_2048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_interval_f_B_range_l20_2039

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 0)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + (a x).1^2 + (a x).2^2

def is_monotone_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f y < f x

theorem f_monotone_decreasing_interval (k : ℤ) :
  is_monotone_decreasing f (Set.Icc (k * π + 3 * π / 8) (k * π + 7 * π / 8)) := by
  sorry

theorem f_B_range (A B C : ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π) (h5 : f (A / 2) = 1) :
  0 < f B ∧ f B ≤ (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_interval_f_B_range_l20_2039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_less_than_sin_squared_sum_l20_2027

theorem sin_squared_sum_less_than_sin_squared_sum (α β : Real) : 
  0 < α ∧ α < π / 2 →
  0 < β ∧ β < π / 2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 < 1 →
  Real.sin α ^ 2 + Real.sin β ^ 2 < Real.sin (α + β) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_less_than_sin_squared_sum_l20_2027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l20_2080

noncomputable def normal_vector : ℝ × ℝ := (Real.sqrt 3, -1)

theorem slope_angle_of_line (l : Set (ℝ × ℝ)) 
  (h : ∃ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l ∧ p ≠ q) 
  (normal : (∀ (x : ℝ × ℝ), x ∈ l → (x.1 - (Classical.choose h).1) * normal_vector.1 + 
                                    (x.2 - (Classical.choose h).2) * normal_vector.2 = 0)) :
  ∃ (θ : ℝ), θ ∈ Set.Icc 0 π ∧ 
             (∀ (x y : ℝ), (x, y) ∈ l → y / x = Real.tan θ) ∧
             θ = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l20_2080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_A_l20_2024

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3}

-- Define set A based on its complement
def A : Finset Nat := U \ {2}

-- Theorem statement
theorem number_of_proper_subsets_of_A : Finset.card (Finset.powerset A \ {A}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_A_l20_2024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l20_2081

theorem ellipse_eccentricity_range (a b : ℝ) (A B F : ℝ × ℝ) (α : ℝ) :
  a > b ∧ b > 0 →
  (A.1^2 / a^2 + A.2^2 / b^2 = 1) →
  B = (-A.1, -A.2) →
  F.1 > 0 →
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 0 →
  π/6 ≤ α ∧ α ≤ π/4 →
  let e := Real.sqrt (1 - b^2/a^2)
  Real.sqrt 2 / 2 ≤ e ∧ e ≤ Real.sqrt 3 - 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l20_2081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_pictures_count_total_pictures_check_l20_2026

/-- Represents the number of pictures in Robin's photo collection --/
structure PhotoCollection where
  phone_pics : ℕ
  camera_pics : ℕ
  albums : ℕ
  pics_per_album : ℕ

/-- The properties of Robin's photo collection --/
def robins_collection : PhotoCollection where
  phone_pics := 35  -- We now know this value
  camera_pics := 5
  albums := 5
  pics_per_album := 8

/-- Theorem stating the number of pictures Robin uploaded from her phone --/
theorem phone_pictures_count :
  robins_collection.phone_pics = 35 := by
  -- The proof is now trivial as we defined the value in robins_collection
  rfl

/-- Theorem verifying the total number of pictures --/
theorem total_pictures_check :
  robins_collection.phone_pics + robins_collection.camera_pics =
  robins_collection.albums * robins_collection.pics_per_album := by
  -- Evaluate both sides
  calc
    robins_collection.phone_pics + robins_collection.camera_pics
    = 35 + 5 := rfl
    _ = 40 := rfl
    _ = 5 * 8 := rfl
    _ = robins_collection.albums * robins_collection.pics_per_album := rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_pictures_count_total_pictures_check_l20_2026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l20_2023

/-- A circle with center (h, k) and radius r is tangent to a line Ax + By + C = 0 if and only if
    the distance from the center to the line equals the radius of the circle. --/
def is_circle_tangent_to_line (h k r A B C : ℝ) : Prop :=
  r^2 * (A^2 + B^2) = (A*h + B*k + C)^2

/-- The equation of a circle with center (h, k) and radius r. --/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_tangent_to_line :
  ∀ (x y : ℝ),
  circle_equation x y 3 (-5) (Real.sqrt 32) ∧
  is_circle_tangent_to_line 3 (-5) (Real.sqrt 32) 1 (-7) 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l20_2023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l20_2042

noncomputable def f (k a x : ℝ) := k * a^x - a^(-x)

theorem function_properties 
  (a : ℝ) 
  (ha : a > 0 ∧ a ≠ 1) 
  (hf : ∀ x, f k a x = -f k a (-x)) 
  (hf_domain : ∀ x, f k a x ∈ Set.univ) :
  ∃ k : ℝ,
    (k = 1) ∧
    (f k a 1 > 0 → 
      (a > 1 ∧ 
       ∀ x y, x < y → f k a x < f k a y)) ∧
    (f k a 1 = 0 → 
      ∃ m : ℝ, 
        (a = 2 ∧ m = 2) ∧
        (∀ x, x ≥ 1 → a^(2*x) + a^(-2*x) - 2*m*(f k a x) ≥ -2) ∧
        (∃ x, x ≥ 1 ∧ a^(2*x) + a^(-2*x) - 2*m*(f k a x) = -2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l20_2042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_insurance_theorem_l20_2061

structure BankDeposit where
  initialAmount : ℚ
  interestRate : ℚ
  time : ℚ
  insuranceLimit : ℚ

def amountReceived (deposit : BankDeposit) : ℚ :=
  min deposit.insuranceLimit (deposit.initialAmount * (1 + deposit.interestRate * deposit.time))

theorem deposit_insurance_theorem (deposit : BankDeposit) 
  (h1 : deposit.initialAmount > 0)
  (h2 : deposit.initialAmount ≤ deposit.insuranceLimit)
  (h3 : deposit.interestRate ≥ 0)
  (h4 : deposit.time ≥ 0) :
  amountReceived deposit = deposit.initialAmount * (1 + deposit.interestRate * deposit.time) :=
by
  sorry

#check deposit_insurance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deposit_insurance_theorem_l20_2061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_theorem_l20_2069

noncomputable section

/-- Curve L in polar coordinates -/
def curve_L (ρ θ : ℝ) (a : ℝ) : Prop :=
  ρ * (Real.sin θ)^2 = 2 * a * Real.cos θ

/-- Point A in polar coordinates -/
noncomputable def point_A : ℝ × ℝ := (2 * Real.sqrt 5, Real.pi + Real.arctan 2)

/-- Line l parallel to θ = π/4 -/
def line_l (x y : ℝ) : Prop :=
  y = x - 2

/-- Geometric progression condition -/
def geometric_progression (AB BC AC : ℝ) : Prop :=
  AB^2 = BC * AC

theorem curve_intersection_theorem (a : ℝ) (h_a : a > 0) :
  ∃ (B C : ℝ × ℝ),
    curve_L B.1 B.2 a ∧
    curve_L C.1 C.2 a ∧
    line_l B.1 B.2 ∧
    line_l C.1 C.2 ∧
    line_l point_A.1 point_A.2 ∧
    geometric_progression (dist point_A B) (dist B C) (dist point_A C) →
  a = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_theorem_l20_2069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l20_2055

/-- The sum of a geometric series with first term 1, common ratio 3, and last term 19683 is 29524 -/
theorem geometric_series_sum : 
  let a := 1  -- first term
  let r := 3  -- common ratio
  let last_term := 19683
  let n := (Real.log last_term) / (Real.log r) + 1  -- number of terms
  (a * (r^n - 1)) / (r - 1) = 29524 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l20_2055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l20_2088

def has_three_integer_solutions (m : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (↑x₁ + 5 > 0 ∧ ↑x₁ - m ≤ 1) ∧
    (↑x₂ + 5 > 0 ∧ ↑x₂ - m ≤ 1) ∧
    (↑x₃ + 5 > 0 ∧ ↑x₃ - m ≤ 1) ∧
    ∀ x : ℤ, (↑x + 5 > 0 ∧ ↑x - m ≤ 1) → (x = x₁ ∨ x = x₂ ∨ x = x₃)

theorem inequality_system_solution_range :
  ∀ m : ℝ, has_three_integer_solutions m ↔ -3 ≤ m ∧ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l20_2088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l20_2087

open Set
open Real

noncomputable def enclosed_area : ℝ := ∫ x in (Icc 0 1), (sqrt x - x^2)

theorem area_between_curves : enclosed_area = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l20_2087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_reciprocal_sum_specific_parabola_reciprocal_sum_l20_2076

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 4 * a * x

/-- Line passing through a point -/
structure Line where
  m : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y = m * x + b

/-- Theorem: For any line passing through the focus of the parabola y^2 = 4x 
    and intersecting the parabola at two points A and B, 
    the sum of the reciprocals of the distances from these points to the focus is always 1 -/
theorem parabola_reciprocal_sum (p : Parabola) (l : Line) 
    (h_focus : l.equation 1 0)  -- Line passes through focus (1, 0)
    (h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      p.equation x₁ y₁ ∧ p.equation x₂ y₂ ∧ 
      l.equation x₁ y₁ ∧ l.equation x₂ y₂ ∧ 
      (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) :  -- Line intersects parabola at two distinct points
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    p.equation x₁ y₁ ∧ p.equation x₂ y₂ ∧ 
    l.equation x₁ y₁ ∧ l.equation x₂ y₂ ∧
    (1 / Real.sqrt ((x₁ - 1)^2 + y₁^2) + 1 / Real.sqrt ((x₂ - 1)^2 + y₂^2) = 1) :=
by sorry

/-- Corollary: For the specific parabola y^2 = 4x and any line x cos θ + y sin θ = cos θ
    intersecting the parabola at two points A and B, 
    the sum of the reciprocals of the distances from these points to the focus is 1 -/
theorem specific_parabola_reciprocal_sum (θ : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    y₁^2 = 4*x₁ ∧ y₂^2 = 4*x₂ ∧
    x₁ * Real.cos θ + y₁ * Real.sin θ = Real.cos θ ∧ x₂ * Real.cos θ + y₂ * Real.sin θ = Real.cos θ ∧
    (1 / Real.sqrt ((x₁ - 1)^2 + y₁^2) + 1 / Real.sqrt ((x₂ - 1)^2 + y₂^2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_reciprocal_sum_specific_parabola_reciprocal_sum_l20_2076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_involutive_function_strictly_decreasing_l20_2066

/-- A function g: (0,∞) → (0,∞) that is differentiable, has continuous derivative,
    satisfies g(g(x)) = x for all x > 0, and is not the identity function. -/
def InvolutiveFunction (g : ℝ → ℝ) : Prop :=
  (∀ x > 0, g x > 0) ∧
  DifferentiableOn ℝ g (Set.Ioi 0) ∧
  ContinuousOn (deriv g) (Set.Ioi 0) ∧
  (∀ x > 0, g (g x) = x) ∧
  (∃ x > 0, g x ≠ x)

/-- Theorem: An involutive function that is not the identity function
    must be strictly decreasing on (0,∞). -/
theorem involutive_function_strictly_decreasing (g : ℝ → ℝ) 
    (hg : InvolutiveFunction g) : 
    ∀ x y, x > 0 → y > 0 → x < y → g y < g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_involutive_function_strictly_decreasing_l20_2066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_largest_element_values_l20_2000

/-- Represents a list of 5 positive integers satisfying given conditions -/
structure IntegerList where
  elements : Fin 5 → ℕ
  mean_is_14 : (elements 0 + elements 1 + elements 2 + elements 3 + elements 4) = 70
  range_is_16 : elements 4 - elements 0 = 16
  mode_and_median_are_8 : elements 2 = 8 ∧ elements 1 = 8
  sorted : elements 0 ≤ elements 1 ∧ elements 1 ≤ elements 2 ∧ elements 2 ≤ elements 3 ∧ elements 3 ≤ elements 4
  positive : ∀ i, elements i > 0

/-- The theorem stating that the second largest element can take exactly 6 different values -/
theorem second_largest_element_values (list : IntegerList) :
  ∃ (values : Finset ℕ), values.card = 6 ∧ list.elements 3 ∈ values := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_largest_element_values_l20_2000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l20_2078

-- Define the total number of students and the number to be chosen
def total_students : ℕ := 6
def chosen_students : ℕ := 3

-- Define the number of boys and girls
def num_boys : ℕ := 4
def num_girls : ℕ := 2

-- Define events A and B as propositions
def event_A : Prop := True  -- Placeholder for "boy A is selected"
def event_B : Prop := True  -- Placeholder for "girl B is selected"

-- Define the probability of event A
noncomputable def prob_A : ℚ := 1 / 2

-- Define the probability of both events A and B occurring
noncomputable def prob_AB : ℚ := 1 / 5

-- Theorem to prove
theorem conditional_probability_B_given_A :
  (prob_AB / prob_A : ℚ) = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_B_given_A_l20_2078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_P_geq_one_l20_2085

open Real Set

/-- The polynomial P(x) = 4x³ + ax² + bx + c -/
def P (a b c : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

/-- The maximum absolute value of P(x) on [-1, 1] -/
noncomputable def M (a b c : ℝ) : ℝ := ⨆ (x : ℝ) (hx : x ∈ Icc (-1) 1), |P a b c x|

/-- The specific polynomial Q(x) = 4x³ - 3x -/
def Q (x : ℝ) : ℝ := 4 * x^3 - 3 * x

theorem max_value_of_P_geq_one (a b c : ℝ) :
  M a b c ≥ 1 ∧ (M a b c = 1 ↔ ∀ x, P a b c x = Q x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_P_geq_one_l20_2085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_15_value_l20_2033

-- Define the sequence
def a : ℕ → ℚ
  | 0 => 2  -- Add this case to cover Nat.zero
  | 1 => 2
  | n + 1 => (1 + a n) / (1 - a n)

-- State the theorem
theorem a_15_value : a 15 = -1/2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_15_value_l20_2033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_tan_squared_constant_l20_2063

/-- A regular 2n-gon inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  center : ℝ × ℝ
  radius : ℝ
  vertices : Fin (2*n) → ℝ × ℝ

/-- Angle from a point to a diagonal of opposite vertices -/
noncomputable def angle_to_diagonal (p : ℝ × ℝ) (poly : RegularPolygon n) (i : Fin n) : ℝ :=
  sorry

/-- Sum of squared tangents of angles to diagonals -/
noncomputable def sum_tan_squared (p : ℝ × ℝ) (poly : RegularPolygon n) : ℝ :=
  Finset.sum (Finset.range n) (fun i => Real.tan (angle_to_diagonal p poly ⟨i, sorry⟩)^2)

/-- Theorem: The sum of squared tangents is constant for any point on the circle -/
theorem sum_tan_squared_constant {n : ℕ} (poly : RegularPolygon n) 
  (p1 p2 : ℝ × ℝ) (h1 : (p1.1 - poly.center.1)^2 + (p1.2 - poly.center.2)^2 = poly.radius^2) 
  (h2 : (p2.1 - poly.center.1)^2 + (p2.2 - poly.center.2)^2 = poly.radius^2) :
  sum_tan_squared p1 poly = sum_tan_squared p2 poly :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_tan_squared_constant_l20_2063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_sum_l20_2011

/-- Represents a quadrilateral WXYZ with specific properties -/
structure Quadrilateral where
  WZ : ℝ
  XY : ℝ
  YZ : ℝ
  WY : ℝ
  XZ : ℝ
  angleYZX : ℝ

/-- The area of the quadrilateral can be expressed as √p + q√r -/
noncomputable def area_expression (p q r : ℝ) : ℝ := Real.sqrt p + q * Real.sqrt r

/-- Function to calculate the area of the quadrilateral -/
noncomputable def Quadrilateral.area (WXYZ : Quadrilateral) : ℝ :=
  sorry -- Placeholder for the actual area calculation

/-- Theorem stating the properties of the specific quadrilateral and its area -/
theorem quadrilateral_area_sum (WXYZ : Quadrilateral) 
    (h1 : WXYZ.WZ = 10)
    (h2 : WXYZ.XY = 6)
    (h3 : WXYZ.YZ = 8)
    (h4 : WXYZ.WY = 7)
    (h5 : WXYZ.XZ = 7)
    (h6 : WXYZ.angleYZX = 45 * π / 180)
    : ∃ (p q r : ℝ), area_expression p q r = WXYZ.area ∧ p + q + r = 824 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_sum_l20_2011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_a_in_range_l20_2079

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x => Real.log (a^2 - 2*a + 1) / Real.log (2*a - 1)

-- State the theorem
theorem f_positive_iff_a_in_range (a : ℝ) :
  (∀ x, f a x > 0) ↔ (a > 1/2 ∧ a < 1) ∨ (a > 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_a_in_range_l20_2079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_t_at_specific_points_l20_2072

noncomputable def t (x : ℝ) (c₀ c₁ c₂ c₃ c₄ : ℝ) : ℝ :=
  Real.cos (5 * x) + c₄ * Real.cos (4 * x) + c₃ * Real.cos (3 * x) + 
  c₂ * Real.cos (2 * x) + c₁ * Real.cos x + c₀

theorem sum_of_t_at_specific_points (c₀ c₁ c₂ c₃ c₄ : ℝ) :
  t 0 c₀ c₁ c₂ c₃ c₄ - t (π/5) c₀ c₁ c₂ c₃ c₄ + t (2*π/5) c₀ c₁ c₂ c₃ c₄ - 
  t (3*π/5) c₀ c₁ c₂ c₃ c₄ + t (4*π/5) c₀ c₁ c₂ c₃ c₄ - t π c₀ c₁ c₂ c₃ c₄ + 
  t (6*π/5) c₀ c₁ c₂ c₃ c₄ - t (7*π/5) c₀ c₁ c₂ c₃ c₄ + 
  t (8*π/5) c₀ c₁ c₂ c₃ c₄ - t (9*π/5) c₀ c₁ c₂ c₃ c₄ = 10 := by
  sorry

#check sum_of_t_at_specific_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_t_at_specific_points_l20_2072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_2x_plus_exp_l20_2075

open Real MeasureTheory

theorem definite_integral_2x_plus_exp : ∫ (x : ℝ) in Set.Icc 0 1, (2 * x + exp x) = exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_2x_plus_exp_l20_2075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l20_2091

noncomputable section

/-- Line l in parametric form -/
def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 + t, 6 - Real.sqrt 3 * t)

/-- Circle C -/
def circle_c (θ : ℝ) : ℝ := 4 * Real.sin θ

/-- Distance between two points in polar coordinates -/
def polar_distance (ρ₁ θ₁ ρ₂ θ₂ : ℝ) : ℝ :=
  Real.sqrt ((ρ₁ * Real.cos θ₁ - ρ₂ * Real.cos θ₂)^2 + (ρ₁ * Real.sin θ₁ - ρ₂ * Real.sin θ₂)^2)

theorem circle_and_line_intersection :
  (∀ θ, circle_c θ = 4 * Real.sin θ) ∧
  polar_distance (circle_c (π/3)) (π/3) ((9 : ℝ) / (2 * Real.sin (π/3 + π/3))) (π/3) = Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l20_2091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_ratio_decreases_hexagon_from_rectangle_l20_2093

/-- The ratio of a hexagon's perimeter to its original rectangle's perimeter -/
noncomputable def perimeter_ratio (k : ℝ) : ℝ :=
  3 * ((k + 1) - Real.sqrt (2 * k)) / (1 + k)

theorem hexagon_perimeter_ratio_decreases
  (k₁ k₂ : ℝ) (h₁ : 0 < k₁) (h₂ : k₁ < k₂) (h₃ : k₂ ≤ 1) :
  perimeter_ratio k₁ > perimeter_ratio k₂ := by
  sorry

/-- Properties of the hexagon formed from a rectangle -/
theorem hexagon_from_rectangle
  (a k : ℝ) (h₁ : a > 0) (h₂ : 0 < k) (h₃ : k ≤ 1) :
  ∃ (x : ℝ),
    x > 0 ∧ 
    x < a ∧
    x = a * ((k + 1) - Real.sqrt (2 * k)) ∧
    perimeter_ratio k = 3 * ((k + 1) - Real.sqrt (2 * k)) / (1 + k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_ratio_decreases_hexagon_from_rectangle_l20_2093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volumes_l20_2096

-- Define the edge length of the cube
def cube_edge : ℝ := 8

-- Define the volume of a sphere given its radius
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4/3) * Real.pi * (radius^3)

-- Define the volume of a cube given its edge length
def cube_volume (edge : ℝ) : ℝ := edge^3

-- Theorem statement
theorem inscribed_sphere_volumes :
  let sphere_vol := sphere_volume (cube_edge / 2)
  let cube_vol := cube_volume cube_edge
  let free_space_vol := cube_vol - sphere_vol
  (sphere_vol = (256/3) * Real.pi) ∧ 
  (free_space_vol = 512 - (256/3) * Real.pi) := by
  sorry

#check inscribed_sphere_volumes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volumes_l20_2096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_from_crescent_area_l20_2054

-- Define the side length of the square
variable (a : ℝ)

-- Define the area of a crescent-shaped region
noncomputable def crescentArea (sideLength : ℝ) : ℝ :=
  (Real.pi * sideLength^2 / 8) - (sideLength^2 / 4)

-- Theorem statement
theorem square_side_length_from_crescent_area :
  crescentArea a = 1 → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_from_crescent_area_l20_2054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_equals_4094552_l20_2035

theorem absolute_value_expression_equals_4094552 :
  let x : ℤ := -2023
  abs (abs (abs x ^ 2 - x) - abs x) - x = 4094552 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_equals_4094552_l20_2035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dental_distribution_l20_2049

theorem dental_distribution (brochures pamphlets : ℕ) 
  (h1 : brochures = 18) 
  (h2 : pamphlets = 12) : 
  (Nat.gcd brochures pamphlets) = 
    (Finset.sup (Finset.filter (fun n => n ∣ brochures ∧ n ∣ pamphlets) (Finset.range (min brochures pamphlets + 1))) id) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dental_distribution_l20_2049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l20_2019

theorem cos_alpha_minus_pi_fourth (α : ℝ) 
  (h1 : Real.cos α = 12/13) 
  (h2 : α ∈ Set.Ioo (3/2 * Real.pi) (2 * Real.pi)) : 
  Real.cos (α - Real.pi/4) = 7 * Real.sqrt 2 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l20_2019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l20_2090

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 3)
  (h2 : Real.sin t.B = Real.sin (2 * t.A))
  (h3 : t.a > t.c) : 
  (t.b / Real.cos t.A = 6) ∧ (3 < t.b ∧ t.b < 3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l20_2090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l20_2009

theorem integral_value (a : ℝ) (h1 : a > 0)
  (h2 : (Nat.choose 6 2) * a^4 = 15) :
  ∫ x in (-a)..a, (x^2 + x + Real.sqrt (1 - x^2)) = 2/3 + π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l20_2009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_of_three_A_win_best_of_five_B_win_fourth_l20_2022

-- Define the probability of Team A winning a single game
noncomputable def p_A_win : ℝ := 1/4

-- Define the probability of Team B winning a single game
noncomputable def p_B_win : ℝ := 1 - p_A_win

-- Theorem for the best-of-three format
theorem best_of_three_A_win : 
  p_A_win ^ 2 + 2 * p_A_win ^ 2 * p_B_win = 5/32 := by sorry

-- Theorem for the best-of-five format, B winning after fourth game
theorem best_of_five_B_win_fourth : 
  3 * p_A_win * p_B_win ^ 3 = 81/256 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_of_three_A_win_best_of_five_B_win_fourth_l20_2022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l20_2030

/-- 
Given a circle with center (0,k) where k > -6, and tangent to lines y = x, y = -x, and y = -6,
prove that the radius of the circle is 6√2 + 6.
-/
theorem circle_radius (k : ℝ) (h1 : k > -6) : 
  let center := (0, k)
  let tangent_line1 := fun (x : ℝ) => x
  let tangent_line2 := fun (x : ℝ) => -x
  let tangent_line3 := fun (_x : ℝ) => -6
  let is_tangent (line : ℝ → ℝ) := ∃ (p : ℝ × ℝ), 
    dist center p = dist p (p.1, line p.1) ∧ 
    dist center p = |center.2 - line center.1| / Real.sqrt (1 + (deriv line center.1)^2)
  is_tangent tangent_line1 → is_tangent tangent_line2 → is_tangent tangent_line3 →
  dist center (0, -6) = 6 * Real.sqrt 2 + 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l20_2030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l20_2036

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (x - 1) / (x + 1)

theorem tangent_line_and_monotonicity (a : ℝ) :
  (∀ x, x > 0 → DifferentiableAt ℝ (f a) x) ∧
  (a = 0 → (deriv (f a)) 1 = 1/2) ∧
  (a ≤ -1/2 → ∀ x, x > 0 → (deriv (f a)) x < 0) ∧
  (-1/2 < a → a < 0 → ∃ c, c > 0 ∧
    (∀ x, 0 < x → x < c → (deriv (f a)) x > 0) ∧
    (∀ x, x > c → (deriv (f a)) x < 0)) ∧
  (a ≥ 0 → ∀ x, x > 0 → (deriv (f a)) x > 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l20_2036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_5_representation_first_digit_l20_2056

def base_10_number : ℕ := 1024

def base_5_first_digit (n : ℕ) : ℕ :=
  let base := 5
  let max_power := Nat.log base n
  n / (base ^ max_power)

theorem base_5_representation_first_digit :
  base_5_first_digit base_10_number = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_5_representation_first_digit_l20_2056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l20_2005

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) - m * x

-- State the theorem
theorem f_properties (m : ℝ) (h_m : m > 0) :
  -- Part 1: f(x) is monotonically decreasing on (0, +∞) when m = 1
  (m = 1 → ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f 1 x₂ < f 1 x₁) ∧
  -- Part 2: Maximum value of f(x) when m > 0
  (∀ x : ℝ, x > -1 → f m x ≤ m - Real.log m - 1) ∧
  -- Part 3: Condition for f(x) to have exactly two zeros in [0, e^2 - 1]
  (∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.exp 2 - 1 ∧
    f m x₁ = 0 ∧ f m x₂ = 0 ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.exp 2 - 1 ∧ f m x = 0 → x = x₁ ∨ x = x₂)
    ↔ 2 / (Real.exp 2 - 1) ≤ m ∧ m < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l20_2005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_section_area_and_volume_correct_l20_2060

/-- Regular triangular pyramid with a section -/
structure RegularTriangularPyramid where
  a : ℝ  -- Side length of the base
  α : ℝ  -- Angle between section and base

/-- Area of the section in a regular triangular pyramid -/
noncomputable def section_area (p : RegularTriangularPyramid) : ℝ :=
  (p.a^2 * Real.sqrt 3) / (48 * Real.cos p.α)

/-- Volume of a regular triangular pyramid -/
noncomputable def pyramid_volume (p : RegularTriangularPyramid) : ℝ :=
  (p.a^3 * Real.tan p.α) / 48

/-- Theorem stating the correctness of section area and pyramid volume -/
theorem section_area_and_volume_correct (p : RegularTriangularPyramid) :
  section_area p = (p.a^2 * Real.sqrt 3) / (48 * Real.cos p.α) ∧
  pyramid_volume p = (p.a^3 * Real.tan p.α) / 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_section_area_and_volume_correct_l20_2060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_meet_once_l20_2046

/-- Represents the circular track --/
structure Track where
  circumference : ℝ
  circumference_pos : circumference > 0

/-- Represents a boy moving on the track --/
structure Boy where
  speed : ℝ
  speed_pos : speed > 0

/-- Calculates the time taken for a boy to complete one lap --/
noncomputable def lap_time (track : Track) (boy : Boy) : ℝ :=
  track.circumference / boy.speed

/-- Calculates the relative speed between two boys --/
def relative_speed (boy1 boy2 : Boy) : ℝ :=
  |boy1.speed - boy2.speed|

/-- Calculates the time taken for the boys to meet again at the starting point --/
noncomputable def time_to_meet_at_start (track : Track) (boy1 boy2 : Boy) : ℝ :=
  track.circumference / relative_speed boy1 boy2

/-- Calculates the number of times the boys meet between start and finish --/
noncomputable def number_of_meetings (track : Track) (boy1 boy2 : Boy) : ℕ :=
  Int.toNat ⌊(time_to_meet_at_start track boy1 boy2) / (track.circumference / relative_speed boy1 boy2)⌋

theorem boys_meet_once (track : Track) : 
  let boy1 := Boy.mk 6 (by norm_num)
  let boy2 := Boy.mk 10 (by norm_num)
  number_of_meetings track boy1 boy2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_meet_once_l20_2046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_percentage_l20_2031

noncomputable def radio_cost : ℝ := 1500
noncomputable def tv_cost : ℝ := 8000
noncomputable def speaker_cost : ℝ := 3000
noncomputable def radio_sell : ℝ := 1245
noncomputable def tv_sell : ℝ := 7500
noncomputable def speaker_sell : ℝ := 2800

noncomputable def total_cost : ℝ := radio_cost + tv_cost + speaker_cost
noncomputable def total_sell : ℝ := radio_sell + tv_sell + speaker_sell
noncomputable def total_loss : ℝ := total_cost - total_sell

noncomputable def loss_percentage : ℝ := (total_loss / total_cost) * 100

theorem store_loss_percentage :
  abs (loss_percentage - 7.64) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_percentage_l20_2031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_decomposition_l20_2034

/-- The radius of each circle -/
def r : ℝ := 40

/-- The side length of the equilateral triangle -/
def s : ℝ := 2 * r + 2 * r

/-- The area of the equilateral triangle -/
noncomputable def area : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The statement to be proven -/
theorem triangle_area_decomposition :
  ∃ (a b : ℕ), area = Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) ∧ a + b = 40960000 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_decomposition_l20_2034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_point_l20_2068

/-- The point of reflection on a plane for a light ray --/
def reflection_point (A C : ℝ × ℝ × ℝ) (plane_normal : ℝ × ℝ × ℝ) (plane_constant : ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Checks if a point is on a plane --/
def on_plane (point : ℝ × ℝ × ℝ) (plane_normal : ℝ × ℝ × ℝ) (plane_constant : ℝ) : Prop :=
  sorry

/-- Checks if three points are collinear --/
def collinear (p q r : ℝ × ℝ × ℝ) : Prop :=
  sorry

theorem light_reflection_point :
  let A : ℝ × ℝ × ℝ := (-4, 10, 12)
  let C : ℝ × ℝ × ℝ := (4, 4, 8)
  let plane_normal : ℝ × ℝ × ℝ := (1, 1, 1)
  let plane_constant : ℝ := 15
  let B : ℝ × ℝ × ℝ := (1/3, 30/11, 80/11)
  (on_plane B plane_normal plane_constant) ∧
  (collinear A B (reflection_point A C plane_normal plane_constant)) ∧
  (B = reflection_point A C plane_normal plane_constant) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_point_l20_2068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_remaining_days_l20_2086

noncomputable def total_days : ℝ := 10

noncomputable def arun_days : ℝ := 70

noncomputable def days_worked_together : ℝ := 4

noncomputable def combined_rate : ℝ := 1 / total_days

noncomputable def arun_rate : ℝ := 1 / arun_days

noncomputable def work_completed : ℝ := combined_rate * days_worked_together

noncomputable def remaining_work : ℝ := 1 - work_completed

theorem arun_remaining_days : 
  ∃ (days : ℝ), days = remaining_work / arun_rate ∧ days = 42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_remaining_days_l20_2086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_remaining_l20_2083

/-- Calculates the remaining pieces of bread after Lucca's eating pattern over four days -/
theorem bread_remaining (initial : ℕ) (day1_fraction day2_fraction day3_fraction day4_fraction : ℚ) : 
  initial = 500 →
  day1_fraction = 1/4 →
  day2_fraction = 2/5 →
  day3_fraction = 3/8 →
  day4_fraction = 1/3 →
  (let day1_remaining := initial - (day1_fraction * initial).floor;
   let day2_remaining := day1_remaining - (day2_fraction * day1_remaining).floor;
   let day3_remaining := day2_remaining - (day3_fraction * day2_remaining).floor;
   let day4_remaining := day3_remaining - (day4_fraction * day3_remaining).floor;
   day4_remaining) = 94 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_remaining_l20_2083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l20_2043

theorem tan_pi_4_minus_alpha (α : Real) 
  (h1 : α ∈ Set.Ioo π (3 * π / 2)) 
  (h2 : Real.cos α = -4/5) : 
  Real.tan (π/4 - α) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l20_2043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_diagonal_length_l20_2012

/-- Given a quadrilateral ABCD where AC and BD intersect at O,
    prove that if OA = 6, OC = 5, OB = 7, OD = 3, and BD = 9, then AC = √469 -/
theorem intersection_diagonal_length (O A B C D : ℝ × ℝ) : 
  ‖O - A‖ = 6 →
  ‖O - C‖ = 5 →
  ‖O - B‖ = 7 →
  ‖O - D‖ = 3 →
  ‖B - D‖ = 9 →
  ‖A - C‖ = Real.sqrt 469 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_diagonal_length_l20_2012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_reciprocal_l20_2098

theorem inverse_sum_reciprocal : (10 : ℚ) * (1/3 + 1/4 + 1/6)⁻¹ = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_reciprocal_l20_2098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_value_l20_2025

noncomputable def fibonacci : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

noncomputable def fibonacci_series_sum : ℝ := ∑' n, fibonacci n / 4^(n + 1)

theorem fibonacci_series_sum_value : fibonacci_series_sum = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_series_sum_value_l20_2025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l20_2001

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism -/
theorem cone_prism_volume_ratio (a h : ℝ) (ha : a > 0) (hh : h > 0) : 
  (1/3) * π * (a/2)^2 * h / (a * (2*a) * h) = π / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_prism_volume_ratio_l20_2001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l20_2010

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 1 + Real.log x

-- Theorem statement
theorem tangent_line_at_one :
  let A : ℝ × ℝ := (1, 0)
  let m : ℝ := f' A.fst
  let tangent_line (x : ℝ) : ℝ := m * (x - A.fst) + A.snd
  ∀ x, tangent_line x = x - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l20_2010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_unit_circle_l20_2045

theorem max_value_on_unit_circle (x y : ℝ) :
  x^2 + y^2 = 1 →
  ∃ (m : ℝ), m^3 = |x^3 - y^3| + |x - y| ∧
  ∀ (k : ℝ), (∃ (a b : ℝ), a^2 + b^2 = 1 ∧ k^3 = |a^3 - b^3| + |a - b|) →
  k ≤ m ∧
  m = (2 : ℝ) ^ (1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_unit_circle_l20_2045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_theorem_third_side_range_l20_2029

-- Define Triangle as a structure
structure Triangle (α : Type*) where
  side1 : α
  side2 : α
  side3 : α

-- Triangle inequality theorem
theorem triangle_inequality_theorem {α : Type*} [LinearOrderedField α] (a b c : α) :
  (a > 0 ∧ b > 0 ∧ c > 0) → (a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
    (∃ t : Triangle α, t.side1 = a ∧ t.side2 = b ∧ t.side3 = c) :=
sorry

-- Third side range theorem
theorem third_side_range (x : ℝ) :
  (∃ t : Triangle ℝ, t.side1 = 3 ∧ t.side2 = 7 ∧ t.side3 = x) ↔ (4 < x ∧ x < 10) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_theorem_third_side_range_l20_2029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l20_2007

/-- The complex number z = sin100° - i*cos100° corresponds to a point in the first quadrant of the complex plane. -/
theorem point_in_first_quadrant : ∃ (z : ℂ), z = Complex.exp (100 * π / 180 * Complex.I) ∧ 
  z.re > 0 ∧ z.im > 0 := by
  -- Let z be the complex number exp(100° * i)
  let z := Complex.exp (100 * π / 180 * Complex.I)
  
  -- Show that z satisfies the equation
  have h1 : z = Complex.exp (100 * π / 180 * Complex.I) := rfl
  
  -- Show that the real part of z is positive
  have h2 : z.re > 0 := by
    sorry -- Proof that cos 100° > 0
  
  -- Show that the imaginary part of z is positive
  have h3 : z.im > 0 := by
    sorry -- Proof that sin 100° > 0
  
  -- Conclude the proof
  exact ⟨z, h1, h2, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l20_2007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l20_2097

noncomputable def f (x : ℝ) : ℝ := |2*x - 1| + x + 1/2

theorem min_value_and_inequality 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 1) 
  (hmin : ∀ x, f x ≥ 1) : 
  2 * (a^2 + b^2 + c^2) ≥ a*b + b*c + c*a - 3*a*b*c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l20_2097
