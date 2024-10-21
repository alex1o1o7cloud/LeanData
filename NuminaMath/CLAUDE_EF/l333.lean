import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l333_33361

-- Define the street length in meters
noncomputable def street_length : ℝ := 595

-- Define the time taken to cross the street in minutes
noncomputable def crossing_time : ℝ := 6

-- Define the conversion factor from meters to kilometers
noncomputable def meters_to_km : ℝ := 1 / 1000

-- Define the conversion factor from minutes to hours
noncomputable def minutes_to_hours : ℝ := 1 / 60

-- State the theorem
theorem speed_calculation :
  let distance_km := street_length * meters_to_km
  let time_hours := crossing_time * minutes_to_hours
  let speed := distance_km / time_hours
  speed = 5.95 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l333_33361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_identity_l333_33309

/-- A bijective function from ℕ to ℕ -/
noncomputable def g : ℕ → ℕ := sorry

/-- A function from ℕ to ℕ satisfying certain properties -/
noncomputable def f : ℕ → ℕ := sorry

/-- For all naturals x, f applied x^2023 times to x equals x -/
axiom f_property (x : ℕ) : (f^[x^2023]) x = x

/-- For all naturals x, y such that x divides y, f(x) divides g(y) -/
axiom f_g_property (x y : ℕ) : x ∣ y → f x ∣ g y

/-- g is bijective -/
axiom g_bijective : Function.Bijective g

/-- Theorem: f(x) = x for all x in ℕ -/
theorem f_identity : ∀ x : ℕ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_identity_l333_33309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_image_of_rectangle_l333_33395

-- Define the rectangle OABC
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (0, 1)

-- Define the transformation
def u (x y : ℝ) : ℝ := x^2 - y^2
noncomputable def v (x y : ℝ) : ℝ := Real.sin (Real.pi * x * y)

-- Define the rectangle as a set of points
def rectangle : Set (ℝ × ℝ) :=
  {p | (0 ≤ p.1 ∧ p.1 ≤ 2) ∧ (0 ≤ p.2 ∧ p.2 ≤ 1)}

-- Define the transformation as a function
noncomputable def transformation (p : ℝ × ℝ) : ℝ × ℝ :=
  (u p.1 p.2, v p.1 p.2)

-- Theorem statement
theorem image_of_rectangle :
  (transformation '' rectangle) = {p | p.1 ∈ Set.Icc (-1) 4 ∧ p.2 = 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_image_of_rectangle_l333_33395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l333_33349

theorem solve_exponential_equation :
  ∃ y : ℚ, (16 : ℝ) ^ (3 * y - 4 : ℝ) = (4 : ℝ) ^ (-y - 6 : ℝ) ↔ y = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l333_33349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_101_equals_103_l333_33314

def G : ℕ → ℚ
  | 0 => 3  -- Add a case for 0 to cover all natural numbers
  | 1 => 3
  | (n + 1) => (3 * G n + 3) / 3

theorem G_101_equals_103 : G 101 = 103 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_101_equals_103_l333_33314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_line_l_properties_l333_33382

/-- Curve C in polar coordinates -/
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * (Real.cos θ + Real.sin θ)

/-- Line l in parametric form -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t / 2, 1 + (Real.sqrt 3 / 2) * t)

/-- Theorem stating the Cartesian equation of curve C and the distance between intersection points -/
theorem curve_C_and_line_l_properties :
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 ↔ ∃ θ : ℝ, x^2 + y^2 = (curve_C θ)^2 ∧ x = (curve_C θ) * Real.cos θ ∧ y = (curve_C θ) * Real.sin θ) ∧
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (∃ t₁ : ℝ, A = line_l t₁ ∧ (A.1 - 1)^2 + (A.2 - 1)^2 = 2) ∧
    (∃ t₂ : ℝ, B = line_l t₂ ∧ (B.1 - 1)^2 + (B.2 - 1)^2 = 2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_line_l_properties_l333_33382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l333_33391

def my_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 8 ∧ ∀ n ≥ 2, a (n + 1) = (4 / n : ℝ) * a n + a (n - 1)

theorem sequence_properties (a : ℕ → ℝ) (h : my_sequence a) :
  (∃ c : ℝ, c > 0 ∧ ∀ n : ℕ, n ≥ 1 → a n ≤ c * n^2) ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n ≤ 4 * n + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l333_33391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l333_33375

theorem simplify_trig_expression (x : ℝ) :
  (1 + Real.sin x + Real.cos x) / (1 - Real.sin x + Real.cos x) = Real.tan (x / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l333_33375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_results_l333_33337

def movements : List Int := [-7, 11, -6, 10, -5]

def fuel_rate : Rat := 1/2

theorem journey_results :
  (movements.sum = 3) ∧
  (movements.map (fun x => x.natAbs)).sum = 39 ∧
  ((movements.map (fun x => x.natAbs)).sum : Rat) * fuel_rate = 39/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_results_l333_33337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l333_33311

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 9*x + 18)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≠ -6 ∧ x ≠ -3}

-- Theorem stating that the domain of f is correct
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l333_33311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_expansion_of_three_elevenths_l333_33301

/-- The length of the smallest repeating block in the decimal expansion of a rational number -/
def min_repeating_block_length (p q : ℚ) : ℕ :=
  sorry

theorem decimal_expansion_of_three_elevenths (n : ℕ) : n = 2 ↔ 
  n = min_repeating_block_length (3 / 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_expansion_of_three_elevenths_l333_33301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_planes_l333_33396

-- Define the types for points, planes, and lines
variable {Point Plane Line : Type}

-- Define the operations
variable (belongs_to : Point → Plane → Prop)
variable (belongs_to_line : Point → Line → Prop)
variable (intersect : Plane → Plane → Line)
variable (line_intersect : Line → Line → Point)
variable (determine_plane : Point → Point → Point → Plane)
variable (line_through : Point → Point → Line)

-- State the theorem
theorem intersection_of_planes 
  (α β γ : Plane) (l : Line) (A B C R : Point) :
  intersect α β = l →
  belongs_to A α →
  belongs_to B α →
  belongs_to C β →
  ¬belongs_to_line C l →
  line_intersect (line_through A B) l = R →
  γ = determine_plane A B C →
  intersect β γ = line_through C R :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_planes_l333_33396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_exists_l333_33317

/-- Represents a color of a face --/
inductive Color
| White
| Triangle
| Other

/-- Represents a face of the cube --/
structure Face where
  color : Color

/-- Represents a cube --/
structure Cube where
  faces : Fin 6 → Face

/-- Represents an orientation of the cube --/
structure CubeOrientation where
  visibleFaces : Fin 3 → Fin 6

/-- The three given orientations --/
def givenOrientations : Fin 3 → CubeOrientation := sorry

/-- Checks if a cube satisfies a given orientation --/
def satisfiesOrientation (c : Cube) (o : CubeOrientation) : Prop := sorry

/-- Theorem: There exists a cube coloring that satisfies all three given orientations --/
theorem cube_coloring_exists :
  ∃ (c : Cube), ∀ (i : Fin 3), satisfiesOrientation c (givenOrientations i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_exists_l333_33317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_flowchart_statements_l333_33352

/-- Represents a statement about flowchart symbols -/
inductive FlowchartStatement
  | start_end : FlowchartStatement
  | input_output : FlowchartStatement
  | decision : FlowchartStatement
  | condition : FlowchartStatement

/-- Checks if a flowchart statement is correct -/
def is_correct (s : FlowchartStatement) : Bool :=
  match s with
  | FlowchartStatement.start_end => true
  | FlowchartStatement.input_output => true
  | FlowchartStatement.decision => true
  | FlowchartStatement.condition => false

/-- The list of all flowchart statements -/
def all_statements : List FlowchartStatement :=
  [FlowchartStatement.start_end, FlowchartStatement.input_output, 
   FlowchartStatement.decision, FlowchartStatement.condition]

/-- Counts the number of correct statements in a list -/
def count_correct (statements : List FlowchartStatement) : Nat :=
  statements.filter is_correct |>.length

theorem correct_flowchart_statements :
  count_correct all_statements = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_flowchart_statements_l333_33352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l333_33350

theorem quartic_equation_solutions :
  {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -Complex.I * Real.sqrt 2, Complex.I * Real.sqrt 2, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l333_33350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_391_l333_33384

theorem greatest_prime_factor_of_391 : 
  (Nat.factors 391).maximum? = some 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_391_l333_33384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_increasing_on_open_interval_l333_33359

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- Theorem statement
theorem sqrt_increasing_on_open_interval : 
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_increasing_on_open_interval_l333_33359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_at_zero_l333_33358

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  is_quadratic_p : ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c
  is_quadratic_q : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  horizontal_asymptote : ∀ ε > 0, ∃ M, ∀ x > M, |p x / q x + 3| < ε
  vertical_asymptote : q 3 = 0
  hole : p (-4) = 0 ∧ q (-4) = 0

/-- Theorem stating that p(0)/q(0) = 0 for the given rational function -/
theorem rational_function_value_at_zero (f : RationalFunction) : f.p 0 / f.q 0 = 0 := by
  sorry

#check rational_function_value_at_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_at_zero_l333_33358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_l333_33324

noncomputable section

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

def is_centroid (O A B C : V) : Prop :=
  O = (1/3 : ℝ) • (A + B + C)

theorem triangle_sine_ratio (O A B C : V) (a b c : ℝ) :
  is_centroid V O A B C →
  (2 : ℝ) • a • (O - A) + b • (O - B) + ((2 * Real.sqrt 3) / 3 : ℝ) • c • (O - C) = 0 →
  ∃ (k : ℝ), k > 0 ∧ 
    a = k / 2 ∧ 
    b = k ∧ 
    c = k * Real.sqrt 3 / 2 :=
sorry

#check triangle_sine_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_l333_33324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_25alpha_form_l333_33390

theorem sin_25alpha_form (α : ℝ) (h : Real.sin α = 3/5) :
  ∃ (n : ℤ), Real.sin (25 * α) = n / (5^25) ∧ ¬(5 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_25alpha_form_l333_33390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_3375_l333_33360

theorem cube_root_3375 :
  ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ 
  (∀ (k : ℕ), k > 0 → k < d → ¬ ∃ (m : ℕ), m > 0 ∧ c * (k : ℝ) ^ (1/3 : ℝ) = (3375 : ℝ) ^ (1/3 : ℝ)) ∧
  c * (d : ℝ) ^ (1/3 : ℝ) = (3375 : ℝ) ^ (1/3 : ℝ) ∧ c = 15 ∧ d = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_3375_l333_33360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l333_33362

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- State the theorem
theorem tangent_line_at_zero (x y : ℝ) :
  (∃ m : ℝ, HasDerivAt f m 0 ∧ 
   y - f 0 = m * (x - 0)) ↔ 
  x - y + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l333_33362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l333_33367

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def b : ℝ × ℝ := (-2, 2 * Real.sqrt 3)

theorem angle_between_vectors : 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l333_33367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l333_33392

noncomputable def f (x : ℝ) : ℝ := 2 * x / (x^2 - 3*x + 5)

theorem range_of_f :
  Set.range f = { y : ℝ | -2/11 ≤ y ∧ y ≤ 2 } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l333_33392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_b_time_l333_33393

/-- The time it takes for two workers to complete a task -/
structure WorkTime where
  a_alone : ℚ
  together : ℚ
  b_alone : ℚ

/-- Calculates the time it takes for worker b to complete the task alone -/
def calculate_b_alone (wt : WorkTime) : ℚ :=
  (wt.a_alone * wt.together) / (wt.a_alone - wt.together)

/-- Theorem stating that if worker a takes 16 days and both workers together take 16/3 days,
    then worker b alone takes 8 days to complete the task -/
theorem worker_b_time (wt : WorkTime) 
  (ha : wt.a_alone = 16)
  (ht : wt.together = 16/3) :
  calculate_b_alone wt = 8 := by
  sorry

#eval calculate_b_alone { a_alone := 16, together := 16/3, b_alone := 0 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_b_time_l333_33393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tamika_always_wins_l333_33378

def tamika_set : Finset ℕ := {10, 11, 12}
def carlos_set : Finset ℕ := {4, 6, 7}

def tamika_sums : Finset ℕ := {21, 22, 23}
def carlos_sums : Finset ℕ := {10, 11, 13}

def total_outcomes : ℕ := (tamika_sums.card * carlos_sums.card)

def favorable_outcomes : ℕ := (tamika_sums.card * carlos_sums.card)

theorem tamika_always_wins :
  (favorable_outcomes : ℚ) / total_outcomes = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tamika_always_wins_l333_33378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l333_33373

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - x + x^2

theorem tangent_line_equation :
  let tangent_line (x y : ℝ) := 3 * x - 2 * y + 2 * Real.log 2 - 3 = 0
  ∃ m b : ℝ,
    (∀ x, tangent_line x (m * x + b)) ∧
    tangent_line 1 (f 1) ∧
    (∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| → |h| < δ → |(f (1 + h) - f 1) / h - m| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l333_33373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_volume_inequality_l333_33323

/-- For any right circular cone, the cube of its surface area is greater than or equal to 72π times the square of its volume. -/
theorem cone_surface_area_volume_inequality (A V : ℝ) (h_A : A > 0) (h_V : V > 0) 
  (h_cone : ∃ r h : ℝ, r > 0 ∧ h > 0 ∧ A = π * r * (r + Real.sqrt (r^2 + h^2)) ∧ V = (1/3) * π * r^2 * h) :
  A^3 ≥ 72 * π * V^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_volume_inequality_l333_33323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_unique_l333_33343

/-- The line equation in parametric form -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (-2 - t, 1 + t, -4 - t)

/-- The plane equation -/
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  2 * p.1 - p.2.1 + 3 * p.2.2 + 23 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (-3, 2, -5)

theorem intersection_point_unique :
  ∃! t : ℝ, plane (line t) ∧ line t = intersection_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_unique_l333_33343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_reals_l333_33318

/-- The function f(x) defined by the given rational expression -/
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (3 * x^2 + 5 * x - 7) / (-7 * x^2 + 5 * x + c)

/-- The theorem stating the condition for f to be defined for all real numbers -/
theorem domain_of_f_is_reals (c : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f c x = y) ↔ c < -25/28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_reals_l333_33318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_equation_l333_33338

theorem solution_set_of_equation : 
  {x : ℝ | (4 : ℝ)^x - 3 * 2^(x+1) + 8 = 0} = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_equation_l333_33338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l333_33365

noncomputable def f (x : ℝ) : ℝ := 1 + Real.sqrt (Real.cos x - 1/2)

theorem domain_of_f :
  ∀ x : ℝ, (∃ k : ℤ, -π/3 + 2*k*π ≤ x ∧ x ≤ π/3 + 2*k*π) ↔ 
  (0 ≤ Real.cos x - 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l333_33365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_and_arithmetic_sequences_l333_33321

-- Define the geometric sequence and its sum
noncomputable def a (n : ℕ) : ℝ := 2^(n-1)
noncomputable def S (n : ℕ) : ℝ := (1 - 2^n) / (1 - 2)

-- Define the arithmetic sequence and its sum
noncomputable def b (n : ℕ) : ℝ := 6 - 2 * (n - 1)
noncomputable def T (n : ℕ) : ℝ := n * (12 - n)

theorem geometric_and_arithmetic_sequences :
  (S 2 = 3) ∧
  (b 2 = a 3) ∧
  (b 3 = -b 5) ∧
  (∀ n : ℕ, T n ≤ 12) :=
by
  sorry

#check geometric_and_arithmetic_sequences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_and_arithmetic_sequences_l333_33321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clayton_total_points_l333_33330

def game1_points : ℕ := 10
def game2_points : ℕ := 14
def game3_points : ℕ := 6

def first_three_games_total : ℕ := game1_points + game2_points + game3_points
def first_three_games_average : ℚ := first_three_games_total / 3

def game4_points : ℕ := (first_three_games_average.num / first_three_games_average.den).toNat

theorem clayton_total_points : 
  game1_points + game2_points + game3_points + game4_points = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clayton_total_points_l333_33330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l333_33334

theorem triangle_third_side_length 
  (a b : ℝ) (θ : ℝ) (ha : a = 10) (hb : b = 15) (hθ : θ = 150 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) ∧ c = Real.sqrt (325 + 150 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l333_33334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l333_33345

/-- The area of a quadrilateral ABCD with specific properties -/
theorem quadrilateral_area (A B C D : ℝ) : 
  A = 4 → D = 4 → B = 5 → 
  (A * D * Real.sin (60 * π / 180)) / 2 = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l333_33345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_in_square_l333_33353

/-- Predicate to check if a triangle is right isosceles -/
def IsRightIsosceles (triangle : Set ℝ × Set ℝ) : Prop := sorry

/-- Predicate to check if the hypotenuses of the triangles are opposite sides of the square -/
def HypotenusesAreOppositeSides (square_side : ℝ) (triangle1 triangle2 : Set ℝ × Set ℝ) : Prop := sorry

/-- Predicate to check if a quadrilateral is a rhombus -/
def IsRhombus (quad : Set ℝ × Set ℝ) : Prop := sorry

/-- The area of the rhombus formed by the intersection of two right isosceles triangles 
    inside a square with side length 2 -/
theorem rhombus_area_in_square (square_side : ℝ) (triangle1 triangle2 : Set ℝ × Set ℝ) 
  (rhombus : Set ℝ × Set ℝ) : ℝ :=
  by
  have h1 : square_side = 2 := by sorry
  have h2 : IsRightIsosceles triangle1 := by sorry
  have h3 : IsRightIsosceles triangle2 := by sorry
  have h4 : HypotenusesAreOppositeSides square_side triangle1 triangle2 := by sorry
  have h5 : IsRhombus rhombus := by sorry
  have h6 : rhombus = (triangle1.1 ∩ triangle2.1, triangle1.2 ∩ triangle2.2) := by sorry
  
  -- Calculate the leg length of the triangles
  have leg_length : ℝ := Real.sqrt 2
  
  -- Calculate the overlap of the triangles
  have overlap : ℝ := 4 - square_side
  
  -- Calculate the area of the rhombus
  have area : ℝ := (1 / 2) * square_side * square_side
  
  exact area

#check rhombus_area_in_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_in_square_l333_33353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_2023_is_A_l333_33348

def sequenceList : List Char := ['A', 'B', 'C', 'D', 'E', 'D', 'C', 'B', 'A', 'A', 'B', 'C', 'E', 'D', 'C', 'B', 'A']

def nth_letter (n : Nat) : Char :=
  sequenceList[n % sequenceList.length]'sorry

theorem letter_2023_is_A : nth_letter 2022 = 'A' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_2023_is_A_l333_33348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_money_needed_after_discount_l333_33313

noncomputable def calculate_additional_money_needed (initial_amount : ℝ) (additional_fraction : ℝ) (discount_percentage : ℝ) : ℝ :=
  let total_needed := initial_amount * (1 + additional_fraction)
  let discounted_amount := total_needed * (1 - discount_percentage / 100)
  discounted_amount - initial_amount

theorem additional_money_needed_after_discount :
  calculate_additional_money_needed 500 (2/5) 15 = 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_money_needed_after_discount_l333_33313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficients_divisibility_l333_33369

theorem polynomial_coefficients_divisibility 
  (p : ℕ) (hp : Nat.Prime p) 
  (f g : Polynomial ℤ) 
  (hf : ∀ (i : ℕ), (p : ℤ) ∣ (f.coeff i))
  (hg : ∀ (i : ℕ), (p : ℤ) ∣ (g.coeff i)) :
  (∀ (i : ℕ), (p : ℤ) ∣ (f.coeff i)) ∨ (∀ (i : ℕ), (p : ℤ) ∣ (g.coeff i)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficients_divisibility_l333_33369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_P_functions_l333_33315

-- Define property P
def has_property_P (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ ∀ x : ℝ, f (x + c) > f (x - c)

-- Define the four functions
def f₁ : ℝ → ℝ := λ x ↦ 3 * x - 1
def f₂ : ℝ → ℝ := λ x ↦ |x|
noncomputable def f₃ : ℝ → ℝ := λ x ↦ Real.cos x
def f₄ : ℝ → ℝ := λ x ↦ x^3 - x

-- Theorem statement
theorem property_P_functions :
  has_property_P f₁ ∧ has_property_P f₄ ∧ ¬has_property_P f₂ ∧ ¬has_property_P f₃ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_P_functions_l333_33315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l333_33376

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧
  0 < B ∧ B < Real.pi/2 ∧
  0 < C ∧ C < Real.pi/2 ∧
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0

-- State the theorem
theorem triangle_ABC_properties
  (A B C : ℝ)
  (a b c : ℝ)
  (h_triangle : triangle_ABC A B C a b c)
  (h_cos_A : Real.cos A = Real.sqrt 5 / 5)
  (h_sin_B : Real.sin B = 3 * Real.sqrt 10 / 10)
  (h_a : a = 4) :
  Real.cos (A + B) = - Real.sqrt 2 / 2 ∧
  (1/2 : ℝ) * a * c * Real.sin B = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l333_33376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_two_l333_33305

theorem trigonometric_expression_equals_two (α : ℝ) :
  Real.sin (π + α) ^ 2 - Real.cos (π + α) * Real.cos (-α) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_two_l333_33305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l333_33380

/-- Given two vectors a and b in R², and a scalar l such that c = l*a + b and c ⟂ a, prove that l = -2 -/
theorem perpendicular_vector_scalar (a b c : ℝ × ℝ) (l : ℝ) :
  a = (1, 2) →
  b = (2, 4) →
  c = (l * a.1 + b.1, l * a.2 + b.2) →
  c.1 * a.1 + c.2 * a.2 = 0 →
  l = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l333_33380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_meets_scooter_l333_33303

/-- Represents the speed of a vehicle -/
def v : ℕ → ℝ := sorry

/-- Represents the time of an event -/
noncomputable def time (h m : ℕ) : ℝ := h + m / 60

/-- Represents the distance between two vehicles at a given time -/
noncomputable def distance (t : ℝ) (v1 v2 : ℝ) : ℝ := t * (v1 - v2)

theorem bicycle_meets_scooter :
  ∀ (v1 v2 v3 v4 : ℝ),
  v1 > v2 ∧ v2 > v3 ∧ v3 > v4 ∧ v1 > 0 ∧ v2 > 0 ∧ v3 > 0 ∧ v4 > 0 →
  distance (time 12 0) v1 v3 = 0 →
  distance (time 14 0) v1 v4 = 0 →
  distance (time 16 0) v1 v2 = 0 →
  distance (time 17 0) v2 v3 = 0 →
  distance (time 18 0) v2 v4 = 0 →
  ∃ (t : ℝ), t = time 15 20 ∧ distance t v3 v4 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_meets_scooter_l333_33303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_sheep_count_test_farm_sheep_count_l333_33302

/-- Proves the number of sheep on a farm given specific conditions -/
theorem farm_sheep_count (sheep_to_horse_ratio : Rat) 
  (horse_food_per_day : ℕ) (total_horse_food : ℕ) : ℕ :=
  let sheep_count := total_horse_food / horse_food_per_day
  have h1 : sheep_to_horse_ratio = 7 / 7 := by sorry
  have h2 : horse_food_per_day = 230 := by sorry
  have h3 : total_horse_food = 12880 := by sorry
  have h4 : sheep_count = 56 := by sorry
  sheep_count

-- Remove the #eval statement as it's causing issues with compilation
-- #eval farm_sheep_count (7 / 7) 230 12880

-- Instead, we can add a test theorem to check the result
theorem test_farm_sheep_count : farm_sheep_count (7 / 7) 230 12880 = 56 := by
  -- Unfold the definition of farm_sheep_count
  unfold farm_sheep_count
  -- The result follows directly from the definition and the sorry tactics
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_sheep_count_test_farm_sheep_count_l333_33302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l333_33354

/-- Calculates the speed of a train given the conditions of the problem -/
noncomputable def calculate_train_speed (train1_length : ℝ) (train2_length : ℝ) (train2_speed : ℝ) (clearing_time : ℝ) : ℝ :=
  let total_length := train1_length + train2_length
  let relative_speed := total_length / clearing_time
  (relative_speed * 3600 / 1000) - train2_speed

/-- Theorem stating the speed of the first train under given conditions -/
theorem first_train_speed :
  let train1_length : ℝ := 120
  let train2_length : ℝ := 280
  let train2_speed : ℝ := 30
  let clearing_time : ℝ := 20
  calculate_train_speed train1_length train2_length train2_speed clearing_time = 42 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof and may cause issues
-- #eval calculate_train_speed 120 280 30 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l333_33354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_points_xy_le_6_l333_33344

/-- The set of points with positive integer coordinates satisfying xy ≤ 6 -/
def S : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 ≤ 6}

/-- The number of points in set S is 14 -/
theorem count_points_xy_le_6 : Finset.card (Finset.filter (fun p => p.1 * p.2 ≤ 6) (Finset.product (Finset.range 6) (Finset.range 6))) = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_points_xy_le_6_l333_33344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_divisors_90_l333_33342

/-- The sum of positive odd divisors of 90 is 78 -/
theorem sum_of_odd_divisors_90 : 
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors 90)).sum id = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_divisors_90_l333_33342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_results_l333_33329

/-- Represents an election with six candidates -/
structure Election :=
  (total_votes : ℕ)
  (winner_percent : ℚ)
  (runner_up_percent : ℚ)
  (third_percent : ℚ)
  (fourth_percent : ℚ)
  (fifth_percent : ℚ)
  (sixth_percent : ℚ)
  (invalid_percent : ℚ)
  (undecided_percent : ℚ)
  (winner_margin : ℕ)

/-- Theorem stating the properties of the election results -/
theorem election_results (e : Election) 
  (h1 : e.winner_percent = 35/100)
  (h2 : e.runner_up_percent = 25/100)
  (h3 : e.third_percent = 16/100)
  (h4 : e.fourth_percent = 10/100)
  (h5 : e.fifth_percent = 8/100)
  (h6 : e.sixth_percent = 6/100)
  (h7 : e.invalid_percent = 7/100)
  (h8 : e.undecided_percent = 4/100)
  (h9 : e.winner_margin = 2000)
  : e.total_votes = 22472 ∧ 
    (e.undecided_percent * ↑e.total_votes).floor = 899 := by
  sorry

#check election_results

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_results_l333_33329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_theorem_l333_33325

/-- The equation of the trajectory of point M -/
def trajectory_equation (x y : ℝ) : Prop :=
  3 * x^2 - y^2 - 8 * x + 5 = 0 ∧ x ≥ 3/2

/-- The circle C with equation x^2 + y^2 = 1 -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- The point Q(2, 0) -/
def point_Q : ℝ × ℝ := (2, 0)

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The length of the tangent from M(x, y) to circle C -/
noncomputable def tangent_length (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 - 1)

/-- The theorem stating the relationship between the tangent length and distances -/
theorem trajectory_theorem (x y : ℝ) :
  tangent_length x y = distance x y 2 0 + 1 → trajectory_equation x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_theorem_l333_33325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l333_33370

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt 2 * Real.sin x

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 Real.pi ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 Real.pi → f y ≤ f x) ∧
  f x = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l333_33370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l333_33363

noncomputable section

open Real

theorem triangle_properties (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  let A : ℝ := π/4
  let B : ℝ := Real.arccos ((a^2 + c^2 - b^2) / (2*a*c))
  let C : ℝ := π - A - B
  b^2 - a^2 = (1/2) * c^2 →
  (1/2) * a * b * sin A = 3 →
  tan C = 2 ∧ b = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l333_33363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_children_with_blue_flags_l333_33346

/-- Represents the color of a flag -/
inductive FlagColor
| Blue
| Red
deriving Repr, DecidableEq

/-- Represents the flags held by a child -/
structure ChildFlags where
  flag1 : FlagColor
  flag2 : FlagColor
deriving Repr, DecidableEq

theorem percentage_of_children_with_blue_flags
  (total_flags : ℕ)
  (children : Finset ChildFlags)
  (h_even : Even total_flags)
  (h_all_used : total_flags = 2 * children.card)
  (h_red_percentage : (children.filter (fun c => c.flag1 = FlagColor.Red ∨ c.flag2 = FlagColor.Red)).card = children.card / 2)
  (h_both_percentage : (children.filter (fun c => c.flag1 ≠ c.flag2)).card = children.card / 10)
  : (children.filter (fun c => c.flag1 = FlagColor.Blue ∨ c.flag2 = FlagColor.Blue)).card = children.card / 2 := by
  sorry

#check percentage_of_children_with_blue_flags

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_children_with_blue_flags_l333_33346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_coefficients_sum_l333_33374

-- Define x as noncomputable
noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 77 / 3) + 5 / 3)

-- State the theorem
theorem unique_coefficients_sum :
  ∃! (a b c : ℕ+),
    x^100 = 3*x^98 + 17*x^96 + 13*x^94 - 2*x^50 + (a:ℝ)*x^46 + (b:ℝ)*x^44 + (c:ℝ)*x^40 ∧
    a + b + c = 167 := by
  sorry

#check unique_coefficients_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_coefficients_sum_l333_33374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digitSum_mod_nine_l333_33355

/-- Function to create a number by repeating a digit d times -/
def repeatDigit (d : Nat) : Nat :=
  (10^d - 1) / 9 * d

/-- Sum of numbers formed by repeating digits from 2 to 9 -/
def digitSum : Nat :=
  Finset.sum (Finset.range 8) (fun i => repeatDigit (i + 2))

/-- Theorem stating that the sum is congruent to 6 modulo 9 -/
theorem digitSum_mod_nine :
  digitSum % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digitSum_mod_nine_l333_33355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l333_33328

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (1/3) * x^3 - (1/2) * (a + 4) * x^2 + (3*a + 5) * x - (2*a + 2) * Real.log x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 
  (deriv (f a)) x + (2*a + 2) / x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x > 0, g a x ≥ (2/3) * Real.log x + 3*a + 14/3) →
  a ≤ -(4 + 4 * Real.log 2) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l333_33328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l333_33366

open Set
open Real

theorem function_range (x : ℝ) (h : sin x - cos x < 0) :
  let y := (λ x => (sin x / abs (sin x)) + (cos x / abs (cos x)) + (tan x / abs (tan x)))
  (range y) = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l333_33366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_to_one_on_top_l333_33300

/-- Represents a deck of cards labeled from 1 to n -/
def Deck (n : ℕ+) := Fin n → Fin n

/-- The operation of reversing the top k cards of a deck -/
def reverse_top {n : ℕ+} (d : Deck n) (k : Fin n) : Deck n :=
  λ i ↦ if i < k then d (k - 1 - i) else d i

/-- Predicate to check if card 1 is on top of the deck -/
def one_on_top {n : ℕ+} (d : Deck n) : Prop := d 0 = 0

/-- Main theorem: For any initial deck, there exists a finite sequence of operations
    that brings card 1 to the top -/
theorem exists_sequence_to_one_on_top {n : ℕ+} (d : Deck n) :
  ∃ (seq : List (Fin n)), one_on_top (seq.foldl reverse_top d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_to_one_on_top_l333_33300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_axis_l333_33320

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

/-- The transformed function g(x) -/
noncomputable def g (x : ℝ) : ℝ := f ((x - Real.pi/3) / 2)

/-- The axis of symmetry for g(x) -/
noncomputable def axis_of_symmetry (k : ℤ) : ℝ := 2 * ↑k * Real.pi + 11 * Real.pi / 6

/-- Theorem stating that the given axis_of_symmetry is correct for g(x) -/
theorem g_symmetry_axis (k : ℤ) (t : ℝ) :
  g (axis_of_symmetry k + t) = g (axis_of_symmetry k - t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_axis_l333_33320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_correct_l333_33379

-- Define the possible article choices
inductive Article
  | A
  | The

-- Define the structure of a phrase
structure Phrase where
  article : Article
  adjective : String
  noun : String

-- Define the correct answer
def correctAnswer : Phrase × Phrase :=
  ({article := Article.A, adjective := "bluer", noun := "sky"},
   {article := Article.A, adjective := "less polluted", noun := "world"})

-- Define a function to check if a given answer is correct
def isCorrectAnswer (answer : Phrase × Phrase) : Prop :=
  answer = correctAnswer

-- Theorem stating that the correct answer is indeed correct
theorem correct_answer_is_correct :
  isCorrectAnswer correctAnswer := by
  -- The proof is trivial as it's true by definition
  rfl

-- You could add more theorems here to represent other aspects of the problem

#check correct_answer_is_correct

-- This is just a placeholder to make the file compile without errors
def main : IO Unit :=
  IO.println s!"The correct answer is option A"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_correct_l333_33379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_area_bound_golden_ratio_least_l333_33371

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- A point in the interior of a unit square -/
structure InteriorPoint where
  x : ℝ
  y : ℝ
  x_bounds : 0 < x ∧ x < 1
  y_bounds : 0 < y ∧ y < 1

/-- The area of a triangle formed by an interior point and two adjacent vertices of the unit square -/
noncomputable def triangleArea (p : InteriorPoint) (corner : Fin 4) : ℝ :=
  match corner with
  | 0 => p.x * p.y / 2
  | 1 => (1 - p.x) * p.y / 2
  | 2 => (1 - p.x) * (1 - p.y) / 2
  | 3 => p.x * (1 - p.y) / 2

theorem golden_ratio_area_bound :
  ∀ (p : InteriorPoint),
  ∀ (i j : Fin 4),
  1 / φ ≤ (triangleArea p i) / (triangleArea p j) ∧
  (triangleArea p i) / (triangleArea p j) ≤ φ := by
  sorry

theorem golden_ratio_least :
  ∀ (a : ℝ),
  (∀ (p : InteriorPoint),
   ∀ (i j : Fin 4),
   1 / a ≤ (triangleArea p i) / (triangleArea p j) ∧
   (triangleArea p i) / (triangleArea p j) ≤ a) →
  φ ≤ a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_area_bound_golden_ratio_least_l333_33371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l333_33347

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t/2, Real.sqrt 2 / 2 + Real.sqrt 3 * t / 2)

noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.cos (θ - Real.pi / 4)

noncomputable def angle_of_inclination : ℝ := Real.pi / 3  -- 60 degrees in radians

noncomputable def chord_length : ℝ := Real.sqrt 10 / 2

theorem line_and_curve_properties :
  (∀ t, line_l t = (t/2, Real.sqrt 2 / 2 + Real.sqrt 3 * t / 2)) →
  (∀ θ, curve_C θ = 2 * Real.cos (θ - Real.pi / 4)) →
  angle_of_inclination = Real.pi / 3 ∧
  chord_length = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l333_33347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l333_33399

def x : ℝ × ℝ × ℝ := (8, -7, -13)
def p : ℝ × ℝ × ℝ := (0, 1, 5)
def q : ℝ × ℝ × ℝ := (3, -1, 2)
def r : ℝ × ℝ × ℝ := (-1, 0, 1)

theorem vector_decomposition :
  x = (-4 : ℝ) • p + (3 : ℝ) • q + r := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l333_33399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_hit_targets_lower_bound_expected_hit_targets_formula_l333_33316

/-- The expected number of hit targets in a shooting range scenario -/
noncomputable def expectedHitTargets (n : ℕ) : ℝ :=
  n * (1 - (1 - 1 / (n : ℝ))^n)

/-- Theorem stating that the expected number of hit targets is at least n/2 -/
theorem expected_hit_targets_lower_bound (n : ℕ) (h : n ≥ 1) :
  expectedHitTargets n ≥ n / 2 := by
  sorry

/-- Theorem stating the exact formula for the expected number of hit targets -/
theorem expected_hit_targets_formula (n : ℕ) (h : n ≥ 1) :
  expectedHitTargets n = n * (1 - (1 - 1 / (n : ℝ))^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_hit_targets_lower_bound_expected_hit_targets_formula_l333_33316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flipflop_savings_theorem_l333_33389

/-- Represents the promotional deal on flip-flops at the 2023 Riverside County Festival -/
structure FlipFlopDeal where
  regularPrice : ℚ
  secondPairDiscount : ℚ
  thirdPairDiscount : ℚ

/-- Calculates the total cost for three pairs of flip-flops under the promotional deal -/
def totalCost (deal : FlipFlopDeal) : ℚ :=
  deal.regularPrice + 
  deal.regularPrice * (1 - deal.secondPairDiscount) + 
  deal.regularPrice * (1 - deal.thirdPairDiscount)

/-- Calculates the percentage saved on three pairs of flip-flops under the promotional deal -/
def percentageSaved (deal : FlipFlopDeal) : ℚ :=
  (3 * deal.regularPrice - totalCost deal) / (3 * deal.regularPrice) * 100

/-- Theorem stating that the percentage saved is approximately 28.33% -/
theorem flipflop_savings_theorem (deal : FlipFlopDeal) 
  (h1 : deal.regularPrice = 60)
  (h2 : deal.secondPairDiscount = 1/4)
  (h3 : deal.thirdPairDiscount = 3/5) :
  ∃ ε : ℚ, ε > 0 ∧ |percentageSaved deal - 2833/100| < ε := by
  sorry

#eval percentageSaved { regularPrice := 60, secondPairDiscount := 1/4, thirdPairDiscount := 3/5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flipflop_savings_theorem_l333_33389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_problem_l333_33385

theorem lcm_gcd_problem : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_problem_l333_33385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factorial_product_l333_33351

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def factorial_product (n : ℕ) : ℕ := Nat.factorial n * Nat.factorial (n + 1)

theorem perfect_square_factorial_product :
  is_perfect_square (factorial_product 17 / 2) ∧
  (∀ n ∈ ({18, 19, 20, 21} : Set ℕ), ¬ is_perfect_square (factorial_product n / 2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_factorial_product_l333_33351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_property_l333_33312

/-- The ellipse M with the given properties -/
structure EllipseM where
  -- Equation of the ellipse: x²/2 + y² = 1
  equation : ℝ → ℝ → Prop := fun x y => x^2 / 2 + y^2 = 1
  -- One focus is at (1, 0)
  focus : ℝ × ℝ := (1, 0)
  -- The axes of symmetry are the coordinate axes
  symmetry_axes : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop) := 
    (fun x _ => x = 0, fun _ y => y = 0)
  -- The circle condition
  circle_tangent : ∃ (b : ℝ), b > 0 ∧ 
    ∃ (p : ℝ × ℝ), (p.1 - 1)^2 + p.2^2 = b^2 ∧ p.1 - 2*Real.sqrt 2*p.2 + 2 = 0

/-- The theorem to be proved -/
theorem ellipse_intersection_property (M : EllipseM) :
  ∃ (m : ℝ), m = Real.sqrt 3 / 2 ∨ m = -Real.sqrt 3 / 2 ∧
  ∃ (A B P : ℝ × ℝ),
    (M.equation A.1 A.2 ∧ M.equation B.1 B.2 ∧ M.equation P.1 P.2) ∧
    (A.2 = A.1 + m ∧ B.2 = B.1 + m) ∧
    P.1 = A.1 + B.1 ∧ P.2 = A.2 + B.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_property_l333_33312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_type_comparison_l333_33307

/-- A structure representing a monomial with coefficients and exponents for variables a and b -/
structure Monomial where
  coeff : ℚ
  a_exp : ℕ
  b_exp : ℕ

/-- Function to check if two monomials have the same type (same exponents) -/
def same_type (m1 m2 : Monomial) : Prop :=
  m1.a_exp = m2.a_exp ∧ m1.b_exp = m2.b_exp

/-- The reference monomial -3ab^2 -/
def reference : Monomial :=
  { coeff := -3, a_exp := 1, b_exp := 2 }

/-- The list of monomials to compare -/
def monomials : List Monomial :=
  [{ coeff := -3, a_exp := 1, b_exp := 3 },   -- -3ab^3
   { coeff := 1/2, a_exp := 2, b_exp := 1 },  -- 1/2ba^2
   { coeff := 2, a_exp := 1, b_exp := 2 },    -- 2ab^2
   { coeff := 3, a_exp := 2, b_exp := 2 }]    -- 3a^2b^2

theorem monomial_type_comparison :
  ∃! m, m ∈ monomials ∧ same_type m reference :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_type_comparison_l333_33307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_half_value_l333_33394

-- Define the function f(x) = 2^x
noncomputable def f (x : ℝ) : ℝ := 2^x

-- Define the property of g being symmetric to f with respect to y = x
def symmetric_to_f (g : ℝ → ℝ) : Prop := 
  ∀ x y, g x = y ↔ f y = x

-- Theorem statement
theorem g_half_value (g : ℝ → ℝ) (h : symmetric_to_f g) : g (1/2) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_half_value_l333_33394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_of_eight_consecutive_odds_l333_33339

/-- Given 8 consecutive odd integers with an average of 414, the least of these integers is 407 -/
theorem least_of_eight_consecutive_odds (integers : List ℤ) : 
  integers.length = 8 ∧ 
  (∀ i j, i < j → integers.get! i + 2 = integers.get! j) ∧
  (∀ k, k < integers.length → integers.get! k % 2 = 1) ∧
  (integers.sum / integers.length = 414) →
  integers.get! 0 = 407 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_of_eight_consecutive_odds_l333_33339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_sum_l333_33364

noncomputable def f (x : ℝ) : ℝ := x / (x - 1) + Real.sin (Real.pi * x)

theorem range_of_floor_sum :
  ∀ x : ℝ, (⌊f x⌋ + ⌊f (2 - x)⌋ = 1) ∨ (⌊f x⌋ + ⌊f (2 - x)⌋ = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_sum_l333_33364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l333_33335

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | n + 2 => 2 * a (n + 1) + (n + 1)

def b (n : ℕ) : ℤ := a n + n + 1

theorem sequence_properties :
  (b 1 = 2 ∧ b 2 = 4) ∧
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = 2 * b n) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2^n - n - 1) := by
  sorry

#eval a 1
#eval a 2
#eval b 1
#eval b 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l333_33335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_less_than_one_l333_33308

/-- Given a circle of radius 1 and 8 points inside it, 
    there exist two points with distance less than 1 between them. -/
theorem distance_less_than_one (points : Finset (ℝ × ℝ)) : 
  (Finset.card points = 8) →
  (∀ p ∈ points, Real.sqrt ((p.1)^2 + (p.2)^2) ≤ 1) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ 
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) < 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_less_than_one_l333_33308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_swap_time_l333_33388

/-- Represents a time on a clock -/
structure ClockTime where
  hours : ℕ
  minutes : ℚ
  valid : 0 ≤ hours ∧ hours < 24 ∧ 0 ≤ minutes ∧ minutes < 60

/-- Converts a ClockTime to a fraction of hours -/
def clockTimeToHours (t : ClockTime) : ℚ :=
  t.hours + t.minutes / 60

/-- Calculates the angle of the hour hand on a clock face -/
def hourHandAngle (t : ClockTime) : ℚ :=
  30 * (clockTimeToHours t)

/-- Calculates the angle of the minute hand on a clock face -/
def minuteHandAngle (t : ClockTime) : ℚ :=
  6 * t.minutes

/-- Determines if two ClockTimes have swapped hand positions -/
def handsSwapped (t1 t2 : ClockTime) : Prop :=
  hourHandAngle t1 = minuteHandAngle t2 ∧ minuteHandAngle t1 = hourHandAngle t2

theorem clock_hands_swap_time : 
  ∃ (departure : ClockTime) (returnTime : ClockTime),
    clockTimeToHours departure ≥ 4 ∧ 
    clockTimeToHours departure < 5 ∧
    clockTimeToHours returnTime ≥ 5 ∧ 
    clockTimeToHours returnTime < 6 ∧
    handsSwapped departure returnTime ∧
    departure = ⟨4, 26 + 122 / 143, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_swap_time_l333_33388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_units_digit_l333_33381

def is_valid_a (a : ℕ) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧
  (a / 10) % 10 = 2 ∧
  (a / 100 ≠ a % 10) ∧
  (a / 100 = 1 ∨ a / 100 = 3) ∧
  (a % 10 = 1 ∨ a % 10 = 3)

def is_valid_b (b : ℕ) : Prop :=
  b ≥ 100 ∧ b < 1000 ∧
  (b / 100 ≠ (b / 10) % 10) ∧
  (b / 100 ≠ b % 10) ∧
  ((b / 10) % 10 ≠ b % 10) ∧
  (b / 100 = 4 ∨ b / 100 = 5 ∨ b / 100 = 6) ∧
  ((b / 10) % 10 = 4 ∨ (b / 10) % 10 = 5 ∨ (b / 10) % 10 = 6) ∧
  (b % 10 = 4 ∨ b % 10 = 5 ∨ b % 10 = 6)

theorem product_units_digit
  (a b : ℕ)
  (ha : is_valid_a a)
  (hb : is_valid_b b)
  (h_even_sum : Even (a + b)) :
  (a * b) % 10 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_units_digit_l333_33381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_region_l333_33356

-- Define the functions
def f₀ (x : ℝ) : ℝ := |x|
def f₁ (x : ℝ) : ℝ := |f₀ x - 1|
def f₂ (x : ℝ) : ℝ := |f₁ x - 2|

-- Define the enclosed area
noncomputable def enclosedArea : ℝ := ∫ x in Set.Ioi (-3) ∩ Set.Iic 3, f₂ x

-- Theorem statement
theorem area_of_enclosed_region : enclosedArea = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_region_l333_33356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_surface_area_relation_l333_33368

/-- A regular triangular pyramid with a lateral face inclined at 45° to the base -/
structure RegularTriangularPyramid where
  /-- The side length of the base -/
  base_side : ℝ
  /-- The volume of the pyramid -/
  volume : ℝ
  /-- The lateral face is inclined at 45° to the base -/
  h : base_side > 0 ∧ volume > 0

/-- Calculate the total surface area of a regular triangular pyramid -/
noncomputable def total_surface_area (p : RegularTriangularPyramid) : ℝ :=
  (p.base_side^2 * Real.sqrt 3 * (1 + Real.sqrt 2)) / 4

/-- The theorem stating the relationship between volume and surface area -/
theorem volume_surface_area_relation (p : RegularTriangularPyramid) :
  p.volume = 9 → total_surface_area p = 9 * Real.sqrt 3 * (1 + Real.sqrt 2) :=
by
  sorry

#check volume_surface_area_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_surface_area_relation_l333_33368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_preserving_functions_l333_33319

-- Define the domain of the functions
def Domain := {x : ℝ | x ≠ 0}

-- Define a geometric sequence
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define a geometric sequence preserving function
def IsGeometricSequencePreserving (f : ℝ → ℝ) : Prop :=
  ∀ a : ℕ → ℝ, (∀ n, a n ∈ Domain) → IsGeometricSequence a → IsGeometricSequence (fun n ↦ f (a n))

-- Define the four functions
noncomputable def f₁ : ℝ → ℝ := fun x ↦ x^2
noncomputable def f₂ : ℝ → ℝ := fun x ↦ 2^x
noncomputable def f₃ : ℝ → ℝ := fun x ↦ Real.sqrt (abs x)
noncomputable def f₄ : ℝ → ℝ := fun x ↦ Real.log (abs x)

theorem geometric_sequence_preserving_functions :
  IsGeometricSequencePreserving f₁ ∧
  IsGeometricSequencePreserving f₃ ∧
  ¬IsGeometricSequencePreserving f₂ ∧
  ¬IsGeometricSequencePreserving f₄ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_preserving_functions_l333_33319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cars_quotient_l333_33322

/-- Represents the maximum number of cars that can pass a sensor in one hour -/
def N : ℕ := 2000

/-- The length of a car in meters -/
def car_length : ℝ := 5

/-- The safety rule function: distance between cars in car lengths based on speed -/
noncomputable def safety_rule (speed : ℝ) : ℝ := ⌈speed / 10⌉

/-- The distance between cars in meters based on speed -/
noncomputable def car_distance (speed : ℝ) : ℝ := car_length * safety_rule speed

/-- The total length of a car unit (car + safety distance) in meters -/
noncomputable def unit_length (speed : ℝ) : ℝ := car_length + car_distance speed

/-- The number of car units that can pass in one hour at a given speed -/
noncomputable def units_per_hour (speed : ℝ) : ℝ := speed * 1000 / unit_length speed

/-- The maximum theoretical number of car units that can pass in one hour -/
def max_theoretical_units : ℝ := 2000

theorem max_cars_quotient :
  N / 10 = 200 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cars_quotient_l333_33322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_position_after_one_rotation_l333_33336

/-- Represents a circular object with a radius -/
structure Circle where
  radius : ℝ

/-- Represents a point on a clock face -/
inductive ClockPosition
  | o12
  | o6

/-- The state of the rolling disk -/
structure DiskState where
  position : ClockPosition
  rotations : ℕ

/-- Calculates the circumference of a circle -/
noncomputable def circumference (c : Circle) : ℝ := 2 * Real.pi * c.radius

/-- Represents the rolling of the disk around the clock face -/
def roll (clock : Circle) (disk : Circle) (initial : DiskState) : DiskState :=
  sorry

theorem disk_position_after_one_rotation 
  (clock : Circle) 
  (disk : Circle) 
  (initial : DiskState) :
  clock.radius = 30 →
  disk.radius = 15 →
  initial.position = ClockPosition.o12 →
  initial.rotations = 0 →
  (roll clock disk initial).position = ClockPosition.o6 ∧ 
  (roll clock disk initial).rotations = 1 := by
  sorry

#check disk_position_after_one_rotation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_position_after_one_rotation_l333_33336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l333_33333

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h1 : b = 3) 
  (h2 : c = 1) 
  (h3 : A = 2 * B) 
  (h4 : Real.sin A / a = Real.sin B / b) 
  (h5 : Real.cos A = (b^2 + c^2 - a^2) / (2 * b * c)) :
  a = 2 * Real.sqrt 3 ∧ 
  Real.sin (A + π/3) = (2 * Real.sqrt 2 - Real.sqrt 3) / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l333_33333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_through_origin_perpendicular_tangent_lines_l333_33304

-- Define the curve C
def f (x : ℝ) : ℝ := x^3 - x

-- Theorem for the slope of the tangent line through the origin
theorem tangent_slope_through_origin :
  ∃ (m : ℝ), m ≠ 0 ∧ f m = 0 ∧ (deriv f m = -1) :=
sorry

-- Theorem for the equations of perpendicular tangent lines
theorem perpendicular_tangent_lines :
  ∃ (x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧
    deriv f x₁ = -1/2 ∧ deriv f x₂ = -1/2 ∧
    (x₁ - 2 * f x₁ - Real.sqrt 2 = 0) ∧
    (x₂ - 2 * f x₂ + Real.sqrt 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_through_origin_perpendicular_tangent_lines_l333_33304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_l333_33327

/-- Regular octagon with perimeter 16√2 -/
structure RegularOctagon :=
  (perimeter : ℝ)
  (is_regular : perimeter = 16 * Real.sqrt 2)

/-- Point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Midpoint of a side in the octagon -/
def side_midpoint (octagon : RegularOctagon) (i : Fin 8) : Point :=
  sorry

/-- Quadrilateral formed by midpoints of every other side -/
def midpoint_quadrilateral (octagon : RegularOctagon) : Set Point :=
  {side_midpoint octagon 0, side_midpoint octagon 2, side_midpoint octagon 4, side_midpoint octagon 6}

/-- Area of a set of points forming a polygon -/
noncomputable def area (s : Set Point) : ℝ := 
  sorry

/-- Theorem stating the area of the midpoint quadrilateral -/
theorem midpoint_quadrilateral_area (octagon : RegularOctagon) :
  area (midpoint_quadrilateral octagon) = 8 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_l333_33327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_total_score_l333_33340

/-- Calculates the total score for a teacher given their test scores and weights -/
def total_score (written_score interview_score : ℝ) (written_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

/-- Theorem stating that the total score for a teacher with given scores and weights is 72 -/
theorem teacher_total_score :
  let written_score : ℝ := 80
  let interview_score : ℝ := 60
  let written_weight : ℝ := 0.6
  let interview_weight : ℝ := 0.4
  total_score written_score interview_score written_weight interview_weight = 72 := by
  -- Unfold the definition of total_score
  unfold total_score
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

#check teacher_total_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_total_score_l333_33340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_surface_sum_l333_33331

theorem dice_surface_sum (n : Nat) (X : Fin 6) : 
  n = 2012 → 
  (∃ (result : Nat), result = (n * 21 - (n - 1) * 7 + 2 * (X.val + 1)) ∧ 
    result ∈ ({28177, 28179, 28181, 28183, 28185, 28187} : Set Nat)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_surface_sum_l333_33331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_winning_percentage_l333_33306

theorem team_winning_percentage (total_games : ℕ) (first_games : ℕ) 
  (first_win_rate : ℚ) (total_win_rate : ℚ) :
  total_games = 60 →
  first_games = 30 →
  first_win_rate = 2/5 →
  total_win_rate = 3/5 →
  (((total_win_rate * total_games - first_win_rate * first_games) : ℚ) / (total_games - first_games) = 4/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_winning_percentage_l333_33306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l333_33377

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 0 (1/2) then x + 1/2
  else if x ∈ Set.Icc (1/2) 1 then 2 * (1 - x)
  else 0  -- undefined outside [0, 1]

-- Define the k-th order step function
noncomputable def f_k (k : ℕ+) (x : ℝ) : ℝ :=
  f (x - k) - k / 2

-- Define the highest and lowest points
noncomputable def P_k (k : ℕ+) : ℝ × ℝ := (k + 1/2, 1 - k/2)
noncomputable def Q_k (k : ℕ+) : ℝ × ℝ := (k + 1, -k/2)

-- Define the line L
def L : Set (ℝ × ℝ) := {p | 2 * p.1 + 4 * p.2 - 5 = 0}

-- Define the distance function from a point to a line
noncomputable def dist_point_to_line (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  |2 * p.1 + 4 * p.2 - 5| / Real.sqrt 20

-- Main theorem
theorem main_theorem :
  (∀ x, f x ≤ x ↔ x ∈ Set.Icc (2/3) 1) ∧
  (∀ k : ℕ+, P_k k ∈ L) ∧
  (∀ k : ℕ+, dist_point_to_line (Q_k k) L = 3 * Real.sqrt 5 / 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l333_33377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_fraction_roots_product_l333_33326

theorem undefined_fraction_roots_product : 
  ∃ (x y : ℝ), x^2 - 4*x - 5 = 0 ∧ y^2 - 4*y - 5 = 0 ∧ x * y = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_fraction_roots_product_l333_33326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_side_product_l333_33372

/-- Predicate stating that five points form a regular pentagon. -/
def IsRegularPentagon (A B C D E : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate stating that a circle passes through all five given points. -/
def CircumscribedCircle (A B C D E : ℝ × ℝ) : Prop :=
  sorry

/-- Function to calculate the radius of a circle passing through five points. -/
def CircleRadius (A B C D E : ℝ × ℝ) : ℝ :=
  sorry

/-- Function to calculate the length of a line segment between two points. -/
def SegmentLength (P Q : ℝ × ℝ) : ℝ :=
  sorry

/-- Given a regular pentagon ABCDE inscribed in a circle of radius 1,
    the product of the length of two adjacent sides (AB) and 
    the length of a side that skips one vertex (AC) is equal to √5. -/
theorem regular_pentagon_side_product (A B C D E : ℝ × ℝ) : 
  IsRegularPentagon A B C D E →
  CircumscribedCircle A B C D E →
  CircleRadius A B C D E = 1 →
  SegmentLength A B * SegmentLength A C = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_side_product_l333_33372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_focus_l333_33357

/-- The parabola defined by y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The focus of the parabola y^2 = 8x -/
def Focus : ℝ × ℝ := (2, 0)

/-- A point on the parabola with x-coordinate 4 -/
noncomputable def P : ℝ × ℝ := (4, Real.sqrt (8 * 4))

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_P_to_focus :
  P ∈ Parabola → distance P Focus = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_focus_l333_33357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l333_33397

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(3x+1)
def domain_f_3x_plus_1 : Set ℝ := Set.Icc 1 7

-- Define the domain of f(x)
def domain_f_x : Set ℝ := Set.Icc 4 22

-- Theorem statement
theorem domain_equivalence :
  (∀ x ∈ domain_f_3x_plus_1, f (3*x + 1) = f (3*x + 1)) →
  (∀ y ∈ domain_f_x, ∃ x ∈ domain_f_3x_plus_1, y = 3*x + 1) ∧
  (∀ x ∈ domain_f_3x_plus_1, 3*x + 1 ∈ domain_f_x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l333_33397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_assignments_for_28_points_l333_33383

def homeworkPoints : ℕ := 28
def groupSize : ℕ := 7

def assignmentsPerGroup (n : ℕ) : ℕ := n * groupSize

def totalAssignments (points : ℕ) : ℕ :=
  let fullGroups := points / groupSize
  (List.range fullGroups).map (fun i => assignmentsPerGroup (i + 1)) |>.sum

theorem minimum_assignments_for_28_points :
  totalAssignments homeworkPoints = 70 := by
  sorry

#eval totalAssignments homeworkPoints

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_assignments_for_28_points_l333_33383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_integral_l333_33387

theorem binomial_expansion_integral (a : ℝ) : 
  (3 * a^2 * (-Real.sqrt 3 / 6) = -Real.sqrt 3 / 2) → 
  (∫ x in (-2 : ℝ)..a, x^2) = 3 ∨ (∫ x in (-2 : ℝ)..a, x^2) = 7/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_integral_l333_33387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_l333_33310

theorem equation_solution_exists : ∃ x : ℝ, 
  75 * x + (18 + 12) * 6 / 4 - 11 * 8 = 2734 ∧ 
  |x - 37.03| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_l333_33310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_transitivity_l333_33332

/-- A structure representing a friendship relation on a set -/
structure FriendshipRelation (X : Type) where
  friends : X → X → Prop

/-- The property of being pairable -/
def isPairable (X : Type) (friends : X → X → Prop) (S : Set X) : Prop :=
  ∃ f : S → S, ∀ x : S, x ≠ f x ∧ friends x.1 (f x).1 ∧ f (f x) = x

theorem friendship_transitivity
  (X : Type) (friends : FriendshipRelation X)
  (h1 : ¬ isPairable X friends.friends (Set.univ : Set X))
  (h2 : ∀ A B : X, ¬ friends.friends A B →
    isPairable X friends.friends (Set.univ \ {A, B}))
  (h3 : ∀ x : X, ∃ y : X, ¬ friends.friends x y) :
  ∀ a b c : X, friends.friends a b → friends.friends b c → friends.friends a c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_transitivity_l333_33332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_PQ_l333_33386

open Real

/-- Circle C in polar coordinates -/
def circle_C (ρ θ : ℝ) : Prop := ρ = 6 * sin θ

/-- Line l in polar coordinates -/
def line_l (ρ θ : ℝ) : Prop := 2 * ρ * sin (θ - π/6) = 4 * Real.sqrt 3

/-- Ray OM -/
def ray_OM (θ : ℝ) : Prop := θ = 5*π/6

/-- Point P: intersection of C and OM -/
noncomputable def point_P : ℝ × ℝ := (3, 5*π/6)

/-- Point Q: intersection of l and OM -/
noncomputable def point_Q : ℝ × ℝ := (4, 5*π/6)

/-- Length of line segment PQ -/
noncomputable def length_PQ : ℝ := point_Q.1 - point_P.1

theorem length_of_PQ : 
  length_PQ = 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_PQ_l333_33386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_tangency_theorem_l333_33341

-- Define the basic geometric objects
structure Plane where

structure Sphere where

structure Point where

-- Define the concept of intersection and tangency
def intersect (p1 p2 : Plane) : Prop :=
  sorry

def touches (s : Sphere) (p : Plane) : Prop :=
  sorry

def touches_spheres (s1 s2 : Sphere) : Prop :=
  sorry

-- Define the bisector plane
def bisector_plane (p1 p2 : Plane) : Plane :=
  sorry

-- Define the circle of intersection
def intersection_circle (s : Sphere) (p : Plane) : Set Point :=
  sorry

-- Define the tangency points
def tangency_points (s : Sphere) (p1 p2 : Plane) : Set Point :=
  sorry

-- Define the locus of tangency points
def locus_of_tangency (given_sphere : Sphere) (p1 p2 : Plane) : Set Point :=
  sorry

-- State the theorem
theorem locus_of_tangency_theorem 
  (given_sphere : Sphere) (p1 p2 : Plane) 
  (h1 : intersect p1 p2) 
  (h2 : touches given_sphere p1) 
  (h3 : touches given_sphere p2) :
  locus_of_tangency given_sphere p1 p2 = 
    intersection_circle given_sphere (bisector_plane p1 p2) ∪ 
    tangency_points given_sphere p1 p2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_tangency_theorem_l333_33341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l333_33398

-- Define the triangle ABC
variable (A B C : ℂ)

-- Define the lengths of the sides of the triangle
noncomputable def a (A B C : ℂ) : ℝ := Complex.abs (C - B)
noncomputable def b (A B C : ℂ) : ℝ := Complex.abs (A - C)
noncomputable def c (A B C : ℂ) : ℝ := Complex.abs (B - A)

-- Define an arbitrary point P in the plane
variable (P : ℂ)

-- Define the distances from P to the vertices of the triangle
noncomputable def u (P A : ℂ) : ℝ := Complex.abs (P - A)
noncomputable def v (P B : ℂ) : ℝ := Complex.abs (P - B)
noncomputable def w (P C : ℂ) : ℝ := Complex.abs (P - C)

-- State the theorem
theorem triangle_inequality (A B C P : ℂ) : 
  (u P A / a A B C) + (v P B / b A B C) + (w P C / c A B C) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l333_33398
