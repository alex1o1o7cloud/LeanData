import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_irrational_with_rational_square_l70_7024

theorem negation_of_existence_irrational_with_rational_square :
  (¬ ∃ x : ℝ, Irrational x ∧ (∃ q : ℚ, x^2 = q)) ↔
  (∀ x : ℝ, Irrational x → Irrational (x^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_irrational_with_rational_square_l70_7024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_times_solution_l70_7027

/-- Represents the time taken by workers A, B, and C to complete a job. -/
structure WorkTimes where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the work times given the parameters a and b. -/
noncomputable def calculateWorkTimes (a b : ℝ) : WorkTimes :=
  { a := b + Real.sqrt (b * (b - a))
  , b := b - a + Real.sqrt (b * (b - a))
  , c := Real.sqrt (b * (b - a)) }

/-- Checks if the calculated work times satisfy the given conditions. -/
def isValidSolution (times : WorkTimes) (a b : ℝ) : Prop :=
  times.a = times.b + a ∧
  times.a = times.c + b ∧
  1 / times.a + 1 / times.b = 1 / times.c

theorem work_times_solution (a b : ℝ) :
  b > a →
  let times := calculateWorkTimes a b
  isValidSolution times a b ∧
  (∀ t : WorkTimes, isValidSolution t a b → t = times) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_times_solution_l70_7027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_power_2023_l70_7032

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1/2, -Real.sqrt 3/2, 0],
    ![Real.sqrt 3/2, 1/2, 0],
    ![0, 0, 1]]

theorem A_power_2023 :
  A ^ 2023 = ![![-1, 0, 0],
               ![0, -1, 0],
               ![0, 0, 1]] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_power_2023_l70_7032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_distance_positive_l70_7036

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the locus of points
def locus (A B : Point) (m n : ℝ) : Set Point :=
  {P : Point | distance P A / distance P B = m / n}

-- Theorem statement
theorem locus_characterization (A B : Point) (m n : ℝ) 
  (h_distinct : A ≠ B) (h_positive : m > 0 ∧ n > 0) :
  (m ≠ n → ∃ (C : Point) (r : ℝ), locus A B m n = {P : Point | distance P C = r}) ∧
  (m = n → locus A B m n = {P : Point | distance P A = distance P B}) := by
  sorry

-- Additional helper theorem
theorem distance_positive (p1 p2 : Point) (h : p1 ≠ p2) : distance p1 p2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_distance_positive_l70_7036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l70_7001

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (x) ^ 2 - Real.sin (x) ^ 2 + 3

-- State the theorem
theorem f_properties :
  (∀ x y, π/6 ≤ x ∧ x < y ∧ y ≤ π/2 → f x > f y) ∧
  (deriv f (π/4) = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l70_7001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l70_7035

def a : ℕ → ℚ
  | 0 => 2  -- Add a case for 0 to avoid the "missing cases" error
  | 1 => 2
  | n + 1 => a n / (3 * a n + 1)

theorem a_formula (n : ℕ) (h : n ≥ 1) : a n = 2 / (6 * n - 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l70_7035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_extra_cakes_l70_7091

/-- Given a baker who initially had 78 cakes and ended up with a total of 87 cakes,
    prove that the number of extra cakes made is 9. -/
theorem baker_extra_cakes (initial_cakes total_cakes extra_cakes : ℕ) : 
  initial_cakes = 78 → 
  total_cakes = 87 → 
  extra_cakes = total_cakes - initial_cakes →
  extra_cakes = 9 := by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

#check baker_extra_cakes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_extra_cakes_l70_7091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l70_7090

noncomputable def f (x : ℝ) := Real.log (x - 1) + 3 / (x - 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | 1 < x ∧ x ≠ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l70_7090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_edge_sum_approx_l70_7052

/-- A right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  height : ℝ

/-- Calculate the sum of the lengths of the pyramid's eight edges -/
noncomputable def edge_sum (p : RightPyramid) : ℝ :=
  4 * p.base_side + 4 * Real.sqrt (p.height^2 + 2 * (p.base_side / 2)^2)

/-- The theorem stating the sum of edge lengths for the given pyramid -/
theorem pyramid_edge_sum_approx :
  let p := RightPyramid.mk 20 15
  ⌊edge_sum p⌋₊ = 162 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_edge_sum_approx_l70_7052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l70_7074

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x-1)^2 + ax + sin(x + π/2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x - 1)^2 + a*x + Real.sin (x + Real.pi/2)

/-- If f is an even function, then a = 2 -/
theorem f_even_implies_a_eq_two :
  ∃ a, IsEven (f a) → a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l70_7074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramanujan_number_l70_7016

def hardy : ℂ := Complex.mk 2 5
def product : ℂ := Complex.mk 40 (-24)

theorem ramanujan_number : ∃ r : ℂ, r * hardy = product ∧ r = Complex.mk ((200:ℝ)/29) (-(248:ℝ)/29) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramanujan_number_l70_7016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_correct_distance_l70_7048

/-- The distance between two parallel lines with equations ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ := |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- The given line l: x - y + 1 = 0 -/
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

/-- The first parallel line: x - y + 2 = 0 -/
def line_1 (x y : ℝ) : Prop := x - y + 2 = 0

/-- The second parallel line: x - y = 0 -/
def line_2 (x y : ℝ) : Prop := x - y = 0

/-- The theorem stating that the two lines are at the correct distance from the given line -/
theorem parallel_lines_at_correct_distance :
  distance_parallel_lines 1 (-1) 1 2 = Real.sqrt 2 / 2 ∧
  distance_parallel_lines 1 (-1) 1 0 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_correct_distance_l70_7048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_finish_third_l70_7069

/-- Represents a runner in the race -/
inductive Runner
| A
| B
| C
| D
| E
deriving BEq, Repr

/-- Represents the finish order of the race -/
def FinishOrder := List Runner

/-- Checks if runner1 finished before runner2 in the given finish order -/
def finishedBefore (runner1 runner2 : Runner) (order : FinishOrder) : Prop :=
  order.indexOf runner1 < order.indexOf runner2

/-- Checks if the given finish order satisfies all race conditions -/
def validFinishOrder (order : FinishOrder) : Prop :=
  finishedBefore Runner.A Runner.B order ∧
  finishedBefore Runner.C Runner.D order ∧
  finishedBefore Runner.B Runner.E order ∧
  finishedBefore Runner.A Runner.D order ∧
  finishedBefore Runner.D Runner.B order

/-- Checks if a runner finished in third place in the given finish order -/
def finishedThird (runner : Runner) (order : FinishOrder) : Prop :=
  order.indexOf runner = 2

theorem cannot_finish_third :
  ∀ (order : FinishOrder),
    validFinishOrder order →
    ¬(finishedThird Runner.A order ∨ finishedThird Runner.E order) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_finish_third_l70_7069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l70_7022

/-- A sports competition with M events and three athletes A, B, and C. --/
structure Competition where
  M : ℕ
  p₁ : ℕ
  p₂ : ℕ
  p₃ : ℕ
  points_A : ℕ
  points_B : ℕ
  points_C : ℕ

/-- The conditions of the competition --/
def validCompetition (c : Competition) : Prop :=
  c.M > 0 ∧
  c.p₁ > c.p₂ ∧ c.p₂ > c.p₃ ∧
  c.p₁ > 0 ∧ c.p₂ > 0 ∧ c.p₃ > 0 ∧
  c.points_A = 22 ∧
  c.points_B = 9 ∧
  c.points_C = 9 ∧
  c.M * (c.p₁ + c.p₂ + c.p₃) = c.points_A + c.points_B + c.points_C

/-- The theorem to be proved --/
theorem competition_result (c : Competition) (hvalid : validCompetition c) :
  c.M = 5 ∧ ∃ (events : Fin 5 → Fin 3), events ⟨1, by norm_num⟩ = ⟨0, by norm_num⟩ ∧ events ⟨4, by norm_num⟩ = ⟨1, by norm_num⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l70_7022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_min_speed_is_60_l70_7018

/-- The distance between city A and city B in miles -/
noncomputable def distance : ℝ := 30

/-- Bob's constant speed in miles per hour -/
noncomputable def bob_speed : ℝ := 40

/-- The time difference between Bob's and Alice's departure in hours -/
noncomputable def time_difference : ℝ := 0.5

/-- The minimum constant speed Alice must exceed to arrive before Bob -/
noncomputable def alice_min_speed : ℝ := distance / (distance / bob_speed - time_difference)

theorem alice_min_speed_is_60 : alice_min_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_min_speed_is_60_l70_7018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_117_l70_7005

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 0, 5; 4, 5, -2; 1, 2, 6]

theorem det_A_eq_117 : Matrix.det A = 117 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_117_l70_7005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bin_game_purple_balls_l70_7075

theorem bin_game_purple_balls (k : ℕ+) : (
  let total_balls : ℚ := 7 + k
  let prob_green : ℚ := 7 / total_balls
  let prob_purple : ℚ := k / total_balls
  let expected_value : ℚ := prob_green * 3 + prob_purple * (-1)
  expected_value = 1
) → k = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bin_game_purple_balls_l70_7075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_production_l70_7045

-- Define the production rates and total production
def refrigerators_per_hour : ℕ := 90
def total_products : ℕ := 11250
def total_days : ℕ := 5
def hours_per_day : ℕ := 9

-- Define the coolers per hour as a natural number
def coolers_per_hour : ℕ := 160  -- We know this from the solution

-- Theorem to prove
theorem factory_production :
  (coolers_per_hour > refrigerators_per_hour) ∧
  (total_products = (refrigerators_per_hour + coolers_per_hour) * total_days * hours_per_day) →
  (coolers_per_hour - refrigerators_per_hour = 70) :=
by
  intro h
  sorry  -- Skip the proof for now

#eval coolers_per_hour - refrigerators_per_hour  -- This should evaluate to 70

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_production_l70_7045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l70_7037

theorem constant_term_binomial_expansion : 
  let a : ℝ := Real.sqrt 5 / 5
  let b : ℝ := 1
  let n : ℕ := 6
  let k : ℕ := 4
  (n.choose k) * a^(n - k) * b^k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l70_7037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charge_difference_l70_7053

/-- Represents the charge for the first hour of therapy -/
def first_hour_charge : ℝ := sorry

/-- Represents the charge for each additional hour of therapy -/
def additional_hour_charge : ℝ := sorry

/-- The total charge for 5 hours of therapy is $250 -/
axiom five_hour_charge : first_hour_charge + 4 * additional_hour_charge = 250

/-- The total charge for 2 hours of therapy is $115 -/
axiom two_hour_charge : first_hour_charge + additional_hour_charge = 115

/-- The difference between the first hour charge and the additional hour charge is $25 -/
theorem charge_difference : first_hour_charge - additional_hour_charge = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charge_difference_l70_7053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_measure_l70_7044

noncomputable def vector_BA : ℝ × ℝ := (1/2, Real.sqrt 2 / 2)
noncomputable def vector_BC : ℝ × ℝ := (Real.sqrt 3 / 2, 1/2)

theorem angle_ABC_measure : 
  let BA := vector_BA
  let BC := vector_BC
  Real.arccos ((BA.1 * BC.1 + BA.2 * BC.2) / 
    (Real.sqrt (BA.1^2 + BA.2^2) * Real.sqrt (BC.1^2 + BC.2^2))) = 
  Real.arccos ((Real.sqrt 3 + Real.sqrt 2) / (2 * Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_measure_l70_7044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_when_cos_is_sqrt3_over_2_l70_7007

open Real

-- Define the function f
noncomputable def f (α : Real) : Real :=
  (sin (π/2 - α) * cos (π/2 + α)) / cos (π + α) -
  (sin (2*π - α) * cos (π/2 - α)) / sin (π - α)

-- Theorem 1: Simplification of f(α)
theorem f_simplification (α : Real) : f α = 2 * sin α := by
  sorry

-- Theorem 2: Value of f(α) when cos(α) = √3/2
theorem f_value_when_cos_is_sqrt3_over_2 (α : Real) (h : cos α = sqrt 3 / 2) :
  f α = 1 ∨ f α = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_when_cos_is_sqrt3_over_2_l70_7007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_2F_l70_7064

-- Define F as a function of x
noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Define the substitution function
noncomputable def sub (x : ℝ) : ℝ := (2 * x - x^2) / (1 + 2 * x^2)

-- Define G as F composed with the substitution
noncomputable def G (x : ℝ) : ℝ := F (sub x)

-- Theorem statement
theorem G_equals_2F : ∀ x : ℝ, G x = 2 * F x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_2F_l70_7064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l70_7050

theorem simplify_expression (m : ℕ) : 
  (3^(m+5) - 3*(3^m)) / (3*(3^(m+4))) = 80 / 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l70_7050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_greater_than_two_l70_7081

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := ((x - 1) * Real.log x) / x

-- State the theorem
theorem sum_greater_than_two (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) (h₄ : f x₁ = f x₂) : 
  x₁ + x₂ > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_greater_than_two_l70_7081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2008_floor_l70_7014

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | (n + 2) => (n + 1) / (sequence_a (n + 1) + 1)

theorem a_2008_floor : ⌊sequence_a 2008⌋ = 2007 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2008_floor_l70_7014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cut_length_for_triangle_l70_7025

theorem smallest_cut_length_for_triangle (a b c : ℝ) (ha : a = 7) (hb : b = 24) (hc : c = 25) :
  let x := sSup {y : ℝ | (a - y) + (b - y) ≤ (c - y) ∧ y ≥ 0 ∧ y ≤ a ∧ y ≤ b ∧ y ≤ c}
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cut_length_for_triangle_l70_7025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_at_distance_two_l70_7056

noncomputable def line (x y : ℝ) : ℝ := 3 * x - 4 * y - 1

noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |3 * x - 4 * y - 1| / Real.sqrt (3^2 + (-4)^2)

theorem locus_of_points_at_distance_two (x y : ℝ) : 
  distance_to_line x y = 2 ↔ 
  (3 * x - 4 * y - 11 = 0 ∨ 3 * x - 4 * y + 9 = 0) := by
  sorry

#check locus_of_points_at_distance_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_at_distance_two_l70_7056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_equals_14_l70_7009

theorem sum_of_solutions_equals_14 :
  ∃ (S : Finset ℝ), (∀ x ∈ S, |x^2 - 14*x + 45| = 3) ∧ (S.sum id = 14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_equals_14_l70_7009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l70_7049

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.cos (α + π/4) = Real.sqrt 10 / 10) : Real.cos α = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l70_7049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l70_7020

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x)
noncomputable def g (x : ℝ) : ℝ := x - (1/2) * x^2 + (1/3) * x^3

theorem f_le_g : ∀ x > -1, f x ≤ g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l70_7020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l70_7034

noncomputable section

open Real

def f (ω : ℝ) (x : ℝ) : ℝ := (sqrt 3 * sin (ω * x) + cos (ω * x)) * cos (ω * x) - 1/2

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_properties (ω : ℝ) (h_ω : ω > 0) (h_period : has_period (f ω) (4 * π)) :
  (∀ k : ℤ, MonotoneOn (f ω) (Set.Icc (4 * k * π - 4 * π / 3) (4 * k * π + 2 * π / 3))) ∧
  (∀ a b c : ℝ, (2 * a - c) * cos b = b * cos c → b = π / 3) ∧
  (∀ A : ℝ, A ∈ Set.Ioo 0 (2 * π / 3) → f ω A ∈ Set.Ioo (1/2) 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l70_7034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_line_l70_7089

noncomputable def point_to_line_distance (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

theorem distance_origin_to_line :
  point_to_line_distance 0 0 3 4 (-15) = 3 := by
  -- Expand the definition of point_to_line_distance
  unfold point_to_line_distance
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_origin_to_line_l70_7089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_minor_axis_l70_7038

/-- An ellipse with center at origin, one focus at (2, 0), and one endpoint of semi-major axis at (5, 0) -/
structure Ellipse where
  center : ℝ × ℝ := (0, 0)
  focus : ℝ × ℝ := (2, 0)
  semi_major_endpoint : ℝ × ℝ := (5, 0)

/-- The semi-minor axis of an ellipse -/
noncomputable def semi_minor_axis (e : Ellipse) : ℝ :=
  Real.sqrt 21

theorem ellipse_semi_minor_axis (e : Ellipse) : 
  semi_minor_axis e = Real.sqrt 21 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_minor_axis_l70_7038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_eq_one_l70_7028

theorem sin_plus_cos_eq_one (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0 ∨ x = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_eq_one_l70_7028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_pokemon_card_change_l70_7026

/-- Represents the change in Sally's Pokemon card count -/
def pokemon_card_change (sold : ℕ) (received : ℕ) (bought : ℕ) : ℤ :=
  -(sold : ℤ) + (received : ℤ) + (bought : ℤ)

/-- Theorem stating that Sally's net change in Pokemon cards is 34 -/
theorem sally_pokemon_card_change :
  pokemon_card_change 27 41 20 = 34 := by
  unfold pokemon_card_change
  norm_num

#eval pokemon_card_change 27 41 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_pokemon_card_change_l70_7026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_10001_units_digit_is_3_l70_7073

noncomputable def a : ℝ := 3 + Real.sqrt 5
noncomputable def b : ℝ := 3 - Real.sqrt 5

def R : ℕ → ℝ
  | 0 => 1
  | 1 => 3
  | (n+2) => 6 * R (n+1) - 4 * R n

theorem R_10001_units_digit_is_3 : R 10001 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_10001_units_digit_is_3_l70_7073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_equals_2017_implies_x0_is_1_l70_7011

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * (2016 + log x)

-- State the theorem
theorem derivative_equals_2017_implies_x0_is_1 :
  ∀ x₀ : ℝ, x₀ > 0 → (deriv f) x₀ = 2017 → x₀ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_equals_2017_implies_x0_is_1_l70_7011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_semicircle_division_l70_7062

/-- An equilateral triangle with a semicircle inscribed on its base -/
structure EquilateralTriangleWithSemicircle where
  /-- The side length of the equilateral triangle -/
  side_length : ℝ
  /-- The semicircle's diameter is equal to the base of the triangle -/
  diameter_eq_base : ℝ
  /-- Assumption that the diameter equals the side length -/
  h_diameter : diameter_eq_base = side_length

/-- The division of the sides of an equilateral triangle by an inscribed semicircle -/
def side_division (t : EquilateralTriangleWithSemicircle) : Prop :=
  ∃ (p q : ℝ),
    p = t.side_length / 3 ∧
    q = t.side_length / 2 ∧
    p + p + p = t.side_length ∧
    q + q = t.side_length

/-- Theorem stating the division of sides in an equilateral triangle with an inscribed semicircle -/
theorem equilateral_triangle_semicircle_division (t : EquilateralTriangleWithSemicircle) :
  side_division t := by
  sorry

#check equilateral_triangle_semicircle_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_semicircle_division_l70_7062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_element_is_two_l70_7046

/-- A set of integers satisfying the given conditions -/
def ValidSet (T : Finset ℕ) : Prop :=
  (∀ x, x ∈ T → 1 ≤ x ∧ x ≤ 15) ∧ 
  (∀ x y, x ∈ T → y ∈ T → x ≠ y → Nat.gcd x y = 1) ∧
  (∀ c d, c ∈ T → d ∈ T → c < d → ¬(d % c = 0)) ∧
  T.card = 8

/-- The theorem stating that 2 is the minimum element in any valid set T -/
theorem min_element_is_two (T : Finset ℕ) (h : ValidSet T) : 
  ∃ x, x ∈ T ∧ (∀ y, y ∈ T → x ≤ y) ∧ x = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_element_is_two_l70_7046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_tangent_circles_l70_7098

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with two opposite corners -/
structure Rectangle where
  corner1 : ℝ × ℝ
  corner2 : ℝ × ℝ

/-- Predicate to check if a circle is tangent to a rectangle -/
def is_tangent (c : Circle) (r : Rectangle) : Prop := sorry

/-- Predicate to check if a point lies on a circle -/
def on_circle (point : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- Function to calculate the area of a rectangle -/
def rectangle_area (r : Rectangle) : ℝ := sorry

/-- Given three congruent circles tangent to a rectangle, with one circle's diameter known,
    prove that the area of the rectangle is 72 -/
theorem rectangle_area_with_tangent_circles
  (p q r : Circle)
  (rect : Rectangle)
  (h1 : p.radius = q.radius ∧ q.radius = r.radius)
  (h2 : q.radius = 3)
  (h3 : is_tangent p rect ∧ is_tangent q rect ∧ is_tangent r rect)
  (h4 : on_circle p.center q ∧ on_circle r.center q) :
  rectangle_area rect = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_tangent_circles_l70_7098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l70_7095

/-- Given vectors a, b, and c in ℝ², prove that if k*a - b is collinear with c, then k = -5 -/
theorem collinear_vectors_lambda (a b c : ℝ × ℝ) (h_a : a = (1, 1)) (h_b : b = (2, 3)) (h_c : c = (-7, -8)) :
  (∃ k : ℝ, ∃ m : ℝ, m ≠ 0 ∧ k • a - b = m • c) → (∃ k : ℝ, k • a - b = (-5) • a + b ∧ k = -5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l70_7095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_extension_l70_7033

noncomputable section

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def extend_side (A B C : V) (factor : ℝ) : Prop :=
  C - B = factor • (B - A)

theorem quadrilateral_extension (E F G H E' F' G' H' : V) :
  extend_side V E F E' 3 →
  extend_side V F G F' 2 →
  extend_side V G H G' 3 →
  extend_side V H E H' 2 →
  E = (1/35 : ℝ) • E' + (7/70 : ℝ) • F' + (14/35 : ℝ) • G' + (28/35 : ℝ) • H' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_extension_l70_7033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_even_function_l70_7088

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := |Real.sin (ω * x - Real.pi / 8)|

-- Define the shifted function g
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x + Real.pi / 12)

-- Theorem statement
theorem min_omega_for_even_function (ω : ℝ) :
  ω > 0 ∧ 
  (∀ x : ℝ, g ω x = g ω (-x)) →
  ∀ ω' : ℝ, ω' > 0 ∧ (∀ x : ℝ, g ω' x = g ω' (-x)) → ω' ≥ 3/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_even_function_l70_7088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_numbers_l70_7019

-- Define the range of four-digit numbers
def four_digit_range : Set ℕ := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999}

-- Theorem stating the count of four-digit numbers
theorem count_four_digit_numbers : 
  Finset.card (Finset.filter (λ n => 1000 ≤ n ∧ n ≤ 9999) (Finset.range 10000)) = 9000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_numbers_l70_7019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l70_7003

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) / (2 + 2 * Real.sin x * Real.cos x)

theorem f_properties :
  (∀ x, f (Real.pi / 2 - x) = f x) ∧
  (∀ x, f (-Real.pi / 2 - x) = -f x) ∧
  (∀ x, f x ≤ 1/2) ∧
  (∃ x, f x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l70_7003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_revenue_at_12_or_13_l70_7077

/-- Represents the revenue function for the bookstore -/
def revenue (p : ℝ) : ℝ := p * (200 - 8 * p)

/-- Theorem stating that the maximum revenue occurs at p = 12 or p = 13 -/
theorem max_revenue_at_12_or_13 :
  ∃ (p : ℝ), p ∈ ({12, 13} : Set ℝ) ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → revenue p ≥ revenue q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_revenue_at_12_or_13_l70_7077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l70_7006

noncomputable def g (x : ℝ) : ℝ := Real.log (x - Real.sqrt (1 + x^2))

theorem g_neither_even_nor_odd :
  ¬(∀ x, g (-x) = g x) ∧ ¬(∀ x, g (-x) = -g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l70_7006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_female_employees_l70_7082

/-- Prove that the total number of female employees in a company is 392, given the specified conditions. -/
theorem total_female_employees (total_employees : ℕ) (female_managers : ℕ) 
  (h_total : total_employees = 800)
  (h_female_managers : female_managers = 200)
  (h_manager_ratio : (↑(total_employees / 5) * 2 : ℚ) = total_employees)
  (h_male_manager_ratio : (↑(male_employees / 5) * 2 : ℚ) = male_employees)
  (h_non_manager_ratio : (↑male_non_managers : ℚ) / female_non_managers = 3 / 2) :
  female_employees = 392 :=
by
  -- Define auxiliary functions
  let male_employees := total_employees - female_employees
  let male_non_managers := male_employees - (male_employees / 5 * 2)
  let female_non_managers := female_employees - female_managers

  -- Proof steps would go here
  sorry

where
  female_employees : ℕ := female_managers + (total_employees - female_managers) / 5 * 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_female_employees_l70_7082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l70_7079

theorem tan_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (π + α) = -Real.sqrt 3 / 2) :
  Real.tan α = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l70_7079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_symmetric_points_l70_7065

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.exp x
def g (k : ℝ) (x : ℝ) : ℝ := k * x

-- Define symmetry about y = x
def symmetric_about_y_eq_x (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.2 ∧ P.2 = Q.1

-- State the theorem
theorem unique_symmetric_points (k : ℝ) : 
  (∃! (P Q : ℝ × ℝ), 
    P.2 = f P.1 ∧ 
    Q.2 = g k Q.1 ∧ 
    symmetric_about_y_eq_x P Q) →
  (k ≤ 0 ∨ k = 1 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_symmetric_points_l70_7065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_dividend_problem_l70_7094

/-- Compute the additional amount per share for each $0.10 above expected earnings -/
def additional_amount_per_share (expected_earnings : ℚ) (actual_earnings : ℚ) (dividend_paid : ℚ) (num_shares : ℕ) : ℚ :=
  let expected_dividend := expected_earnings / 2
  let additional_earnings := actual_earnings - expected_earnings
  let expected_total_dividend := expected_dividend * num_shares
  let additional_dividend := dividend_paid - expected_total_dividend
  let num_increments := additional_earnings / (1/10)
  additional_dividend / (num_shares * num_increments)

/-- The theorem applied to the given problem -/
theorem company_dividend_problem : 
  additional_amount_per_share (8/10) (11/10) 208 400 = 4/100 := by
  -- Unfold the definition and simplify
  unfold additional_amount_per_share
  -- Perform the calculation
  norm_num
  -- QED

#eval additional_amount_per_share (8/10) (11/10) 208 400

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_dividend_problem_l70_7094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_surrounded_by_four_circles_l70_7076

-- Define the Circle structure
structure Circle (α : Type*) where
  center : α × α
  radius : ℝ

-- Define the touching function
def touching (c1 c2 : Circle ℝ) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = c1.radius + c2.radius

-- Main theorem
theorem circle_surrounded_by_four_circles : 
  (∃ (c : Circle ℝ) (c1 c2 c3 c4 : Circle ℝ),
    c.radius = 2 ∧ 
    c1.radius = 2 + 2 * Real.sqrt 2 ∧ 
    c2.radius = 2 + 2 * Real.sqrt 2 ∧ 
    c3.radius = 2 + 2 * Real.sqrt 2 ∧ 
    c4.radius = 2 + 2 * Real.sqrt 2 ∧
    touching c c1 ∧ touching c c2 ∧ touching c c3 ∧ touching c c4 ∧
    touching c1 c2 ∧ touching c2 c3 ∧ touching c3 c4 ∧ touching c4 c1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_surrounded_by_four_circles_l70_7076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fine_is_one_percent_of_profits_l70_7023

/-- Represents MegaCorp's financial data and fine --/
structure MegaCorpFinancials where
  daily_mining_earnings : ℚ
  daily_oil_refining_earnings : ℚ
  monthly_expenses : ℚ
  fine : ℚ
  days_in_year : ℕ
  months_in_year : ℕ

/-- Calculates the percentage of annual profits that the fine represents --/
noncomputable def fine_percentage (f : MegaCorpFinancials) : ℚ :=
  let daily_earnings := f.daily_mining_earnings + f.daily_oil_refining_earnings
  let annual_earnings := daily_earnings * f.days_in_year
  let annual_expenses := f.monthly_expenses * f.months_in_year
  let annual_profits := annual_earnings - annual_expenses
  (f.fine / annual_profits) * 100

/-- Theorem stating that the fine is 1% of annual profits --/
theorem fine_is_one_percent_of_profits (f : MegaCorpFinancials) 
  (h1 : f.daily_mining_earnings = 3000000)
  (h2 : f.daily_oil_refining_earnings = 5000000)
  (h3 : f.monthly_expenses = 30000000)
  (h4 : f.fine = 25600000)
  (h5 : f.days_in_year = 365)
  (h6 : f.months_in_year = 12) :
  fine_percentage f = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fine_is_one_percent_of_profits_l70_7023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_equals_60_l70_7070

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * (5 - x) * Real.sin (Real.pi * x) - 1

-- Define the interval
def I : Set ℝ := Set.Icc 0 10

-- State the theorem
theorem sum_of_zeros_equals_60 :
  ∃ (S : Finset ℝ), (∀ x ∈ S, x ∈ I ∧ f x = 0) ∧ (S.sum id = 60) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_equals_60_l70_7070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l70_7008

/-- The equation of circle D -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 16*y + 81 = -y^2 - 12*x

/-- The center of circle D -/
def center : ℝ × ℝ := (-6, -8)

/-- The radius of circle D -/
noncomputable def radius : ℝ := Real.sqrt 19

theorem circle_properties :
  let (c, d) := center
  ∀ x y : ℝ, circle_equation x y →
    (x + 6)^2 + (y + 8)^2 = 19 ∧
    c = -6 ∧
    d = -8 ∧
    radius = Real.sqrt 19 ∧
    c + d + radius = -14 + Real.sqrt 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l70_7008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_pipe_fills_in_nine_hours_l70_7054

/-- Represents the time it takes for the faster pipe to fill the pool alone -/
noncomputable def faster_pipe_time (pipe1_rate : ℝ) (pipe2_rate : ℝ) (combined_time : ℝ) : ℝ :=
  1 / pipe1_rate

/-- Theorem stating that the faster pipe fills the pool in nine hours -/
theorem faster_pipe_fills_in_nine_hours 
  (pipe1_rate : ℝ) 
  (pipe2_rate : ℝ) 
  (combined_time : ℝ) 
  (h1 : pipe1_rate = 1.25 * pipe2_rate) 
  (h2 : combined_time = 5) 
  (h3 : (pipe1_rate + pipe2_rate) * combined_time = 1) :
  faster_pipe_time pipe1_rate pipe2_rate combined_time = 9 := by
  sorry

#check faster_pipe_fills_in_nine_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_pipe_fills_in_nine_hours_l70_7054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_cubic_equation_l70_7063

theorem product_of_roots_cubic_equation : 
  let f : ℝ → ℝ := λ x => 3 * x^3 - 9 * x^2 + x - 15
  let roots := {x : ℝ | f x = 0}
  ∃ a b c, a ∈ roots ∧ b ∈ roots ∧ c ∈ roots ∧ a * b * c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_cubic_equation_l70_7063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trisection_intersection_l70_7084

noncomputable def f (x : ℝ) := Real.log (x^2)

theorem trisection_intersection (x₁ x₂ x₃ : ℝ) :
  0 < x₁ → x₁ < x₂ →
  x₁ = 1 → x₂ = 10 →
  f x₁ < f x₂ →
  ∃ (y₁ y₂ y₃ : ℝ),
    y₁ = f x₁ ∧ y₂ = f x₂ ∧
    y₃ = (2 * y₁ + y₂) / 3 ∧
    y₃ = f x₃ →
  x₃ = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trisection_intersection_l70_7084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_7394_to_hundredth_l70_7015

noncomputable def round_to_hundredth (x : ℝ) : ℝ := 
  ⌊x * 100 + 0.5⌋ / 100

theorem round_24_7394_to_hundredth : 
  round_to_hundredth 24.7394 = 24.74 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_7394_to_hundredth_l70_7015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_side_length_l70_7067

theorem min_square_side_length (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let min_side := min a ((Real.sqrt 2 / 2) * (a + b))
  ∃ (s : ℝ), s = min_side ∧
    (∀ (x y : ℝ), x^2 + y^2 ≤ s^2 → (x ≤ a/2 ∧ y ≤ b/2)) ∧
    (∀ (t : ℝ), (∀ (x y : ℝ), x^2 + y^2 ≤ t^2 → (x ≤ a/2 ∧ y ≤ b/2)) → t ≥ s) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_side_length_l70_7067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_formation_l70_7085

-- Define the points and distances
variable (A B C D E : ℝ × ℝ)
variable (x y z : ℝ)

-- Define the conditions
def points_on_line (A B C D : ℝ × ℝ) : Prop := ∃ t₁ t₂ t₃ : ℝ, 
  B = A + t₁ • (1, 0) ∧ 
  C = A + t₂ • (1, 0) ∧ 
  D = A + t₃ • (1, 0) ∧ 
  0 < t₁ ∧ t₁ < t₂ ∧ t₂ < t₃

def distances (A B C D : ℝ × ℝ) (x y z : ℝ) : Prop :=
  ‖B - A‖ = x ∧
  ‖C - A‖ = y ∧
  ‖D - A‖ = z

def rays_intersect (B C E : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, E = B + t₁ • (C - B) ∧ E = C + t₂ • (B - C)

def positive_area (A B C E : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (E.2 - A.2) - (E.1 - A.1) * (B.2 - A.2) ≠ 0

-- Theorem statement
theorem quadrilateral_formation (A B C D E : ℝ × ℝ) (x y z : ℝ)
  (h1 : points_on_line A B C D)
  (h2 : distances A B C D x y z)
  (h3 : rays_intersect B C E)
  (h4 : positive_area A B C E) :
  x < z ∧ y < x + z ∧ y > z / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_formation_l70_7085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_percentage_is_one_sixth_l70_7068

/-- Represents the tiling of a rectangle with squares and hexagons -/
structure Tiling where
  rectangle_width : ℚ
  rectangle_height : ℚ
  num_squares : ℕ
  square_side_length : ℚ

/-- Calculates the percentage of area enclosed by hexagons in the tiling -/
def hexagon_area_percentage (t : Tiling) : ℚ :=
  let total_area := t.rectangle_width * t.rectangle_height
  let squares_area := (t.num_squares : ℚ) * (t.square_side_length ^ 2)
  let hexagons_area := total_area - squares_area
  (hexagons_area / total_area) * 100

/-- Theorem stating that for a specific tiling, the hexagon area percentage is 1/6 * 100% -/
theorem hexagon_area_percentage_is_one_sixth
  (t : Tiling)
  (h1 : t.rectangle_width = 2)
  (h2 : t.rectangle_height = 3)
  (h3 : t.num_squares = 5)
  (h4 : t.square_side_length = 1) :
  hexagon_area_percentage t = (1 / 6) * 100 := by
  sorry

#eval (1 / 6 : ℚ) * 100  -- To verify the result is approximately 16.67

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_percentage_is_one_sixth_l70_7068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l70_7010

-- Define the function f(x) = lg|x|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log 2

-- State the theorem
theorem range_of_a (a : ℝ) (h : f 1 < f a) : a > 1 ∨ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l70_7010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_at_most_one_obtuse_angle_l70_7042

-- Define Triangle as an opaque type
structure Triangle : Type where

-- Define count_obtuse_angles as an opaque function
def count_obtuse_angles : Triangle → ℕ := sorry

-- Theorem statement
theorem triangle_at_most_one_obtuse_angle :
  ∀ (T : Triangle), (count_obtuse_angles T ≥ 2) → False :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_at_most_one_obtuse_angle_l70_7042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seedling_purchase_l70_7002

-- Define the price of type A seedlings at the base
def price_A_base : ℝ → Prop := λ x ↦ 
  300 / x = 300 / (5/4 * x) + 3

-- Define the cost function
def cost : ℝ → ℝ := λ m ↦ 
  18 * m + 27 * (100 - m)

-- State the theorem
theorem seedling_purchase :
  ∃ (x : ℝ), price_A_base x ∧ 
  x = 20 ∧
  (∀ m : ℝ, 0 ≤ m ∧ m ≤ 50 → cost m ≥ 2250) ∧
  cost 50 = 2250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seedling_purchase_l70_7002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l70_7004

-- Define the given values
noncomputable def train_length : ℝ := 130
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def total_length : ℝ := 245

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Theorem statement
theorem train_crossing_time :
  let train_speed_ms := train_speed_kmh * kmh_to_ms
  let bridge_length := total_length - train_length
  let crossing_time := total_length / train_speed_ms
  ∃ ε > 0, abs (crossing_time - 19.6) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l70_7004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_outputs_l70_7017

-- Define the algorithm as a function
def algorithm (x : Int) : Int :=
  if x < 0 then x + 2 else x - 1

-- Theorem stating the correct outputs for the given inputs
theorem algorithm_outputs :
  (algorithm (-1) = 1) ∧ (algorithm 0 = -1) ∧ (algorithm 1 = 0) := by
  -- Split the conjunction into three parts
  apply And.intro
  · -- Case for x = -1
    simp [algorithm]
  · apply And.intro
    · -- Case for x = 0
      simp [algorithm]
    · -- Case for x = 1
      simp [algorithm]

-- Examples to check the algorithm
#eval algorithm (-1)
#eval algorithm 0
#eval algorithm 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_outputs_l70_7017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l70_7012

open Real

-- Define the function f(x) = ln x - x
noncomputable def f (x : ℝ) : ℝ := log x - x

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → f x₁ < f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l70_7012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unattainable_value_l70_7071

-- Define the function f(x) = (2 - x) / (3x + 4)
noncomputable def f (x : ℝ) : ℝ := (2 - x) / (3 * x + 4)

-- State the theorem
theorem unattainable_value (x : ℝ) (h : x ≠ -4/3) :
  f x ≠ -1/3 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unattainable_value_l70_7071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_hemisphere_radius_equality_l70_7093

/-- The radius of a sphere that deflates to a hemisphere with half the original volume -/
def sphere_to_hemisphere_radius (r : ℝ) : Prop :=
  let hemisphere_volume := (2/3) * Real.pi * r^3
  let sphere_volume := (4/3) * Real.pi * r^3
  hemisphere_volume = (1/2) * sphere_volume

theorem sphere_hemisphere_radius_equality (r : ℝ) (h : r = 4 * (2 : ℝ)^(1/3)) :
  sphere_to_hemisphere_radius r → r = 4 * (2 : ℝ)^(1/3) :=
by
  sorry

#check sphere_hemisphere_radius_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_hemisphere_radius_equality_l70_7093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l70_7087

/-- Represents a seating arrangement of 3 people in 6 seats -/
def SeatingArrangement := Fin 6 → Option (Fin 3)

/-- Checks if exactly two empty seats are adjacent in a seating arrangement -/
def hasExactlyTwoAdjacentEmptySeats (s : SeatingArrangement) : Prop :=
  ∃ i : Fin 5, (s i).isNone ∧ (s (i + 1)).isNone ∧
    (∀ j : Fin 5, j ≠ i → (s j).isSome ∨ (s (j + 1)).isSome)

/-- The set of all valid seating arrangements -/
def validArrangements : Set SeatingArrangement :=
  {s | (∀ i : Fin 3, ∃! j : Fin 6, s j = some i) ∧ hasExactlyTwoAdjacentEmptySeats s}

-- Add instances for Fintype and DecidablePred
instance : Fintype SeatingArrangement := by sorry

instance : DecidablePred (λ s => s ∈ validArrangements) := by sorry

theorem valid_arrangements_count :
  Finset.card (Finset.filter (λ s => s ∈ validArrangements) (Finset.univ : Finset SeatingArrangement)) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l70_7087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_seven_ninths_l70_7040

/-- The sum of the infinite series ∑(1/(n(n+3))) from n=1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' n, 1 / (n * (n + 3))

/-- Theorem: The sum of the infinite series ∑(1/(n(n+3))) from n=1 to infinity is equal to 7/9 -/
theorem infinite_series_sum_eq_seven_ninths : infinite_series_sum = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_seven_ninths_l70_7040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_calculation_l70_7086

/-- The original price of a shirt before discounts -/
def original_price (P : ℝ) : Prop := True

/-- The final price of the shirt after two 25% discounts -/
def final_price (P : ℝ) : Prop := True

theorem shirt_price_calculation (P : ℝ) :
  original_price P →
  final_price 17 →
  ∃ ε > 0, |P - 30.22| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_price_calculation_l70_7086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_go_to_sea_is_better_l70_7096

/-- Represents the decision to go to sea or not -/
inductive Decision
  | GoToSea
  | StayOnLand

/-- Calculates the expected value of a decision given the probabilities and potential profits/losses -/
def expectedValue (
  goodWeatherProb : ℝ
) (badWeatherProb : ℝ
) (profitGoodWeather : ℝ
) (lossBadWeather : ℝ
) (lossStayOnLand : ℝ
) (decision : Decision
) : ℝ :=
  match decision with
  | Decision.GoToSea => goodWeatherProb * profitGoodWeather - badWeatherProb * lossBadWeather
  | Decision.StayOnLand => -lossStayOnLand

/-- Theorem stating that going to sea is the better decision -/
theorem go_to_sea_is_better :
  let goodWeatherProb : ℝ := 0.6
  let badWeatherProb : ℝ := 0.4
  let profitGoodWeather : ℝ := 6000
  let lossBadWeather : ℝ := 8000
  let lossStayOnLand : ℝ := 1000
  expectedValue goodWeatherProb badWeatherProb profitGoodWeather lossBadWeather lossStayOnLand Decision.GoToSea >
  expectedValue goodWeatherProb badWeatherProb profitGoodWeather lossBadWeather lossStayOnLand Decision.StayOnLand :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_go_to_sea_is_better_l70_7096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_negative_six_l70_7058

theorem sum_abc_equals_negative_six (a b c : ℤ) : 
  (∃ (p q : Polynomial ℤ), 
    (X^2 + a*X + b : Polynomial ℤ) = (X + 1) * p ∧ 
    (X^2 + b*X + c : Polynomial ℤ) = (X + 1) * q ∧
    (X + 1) * p * q = X^3 - 4*X^2 + X + 6) →
  a + b + c = -6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abc_equals_negative_six_l70_7058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_golden_ratio_golden_ratio_equation_l70_7057

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Fibonacci golden ratio theorem -/
theorem fib_golden_ratio (n : ℕ) :
  (fib (n + 1) : ℝ) = φ * (fib n) + (-1 / φ) ^ n := by
  sorry

/-- φ^2 = φ + 1 -/
theorem golden_ratio_equation : φ^2 = φ + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_golden_ratio_golden_ratio_equation_l70_7057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l70_7031

/-- Curve C1 in polar coordinates -/
noncomputable def C1 (θ : ℝ) : ℝ := 4 * Real.cos θ

/-- Curve C3 in polar coordinates -/
def C3 : ℝ := 1

/-- The angle of the ray -/
noncomputable def ray_angle : ℝ := Real.pi / 3

theorem intersection_distance : 
  C1 ray_angle - C3 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l70_7031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_ratio_is_one_l70_7039

/-- The polynomial g(x) = x^2009 + 19x^2008 + 1 -/
def g (x : ℂ) : ℂ := x^2009 + 19*x^2008 + 1

/-- The roots of g(x) -/
noncomputable def s : Fin 2009 → ℂ := sorry

/-- The polynomial Q of degree 2009 -/
noncomputable def Q : Polynomial ℂ := sorry

variable (h_distinct : ∀ (i j : Fin 2009), i ≠ j → s i ≠ s j)
variable (h_roots : ∀ (i : Fin 2009), g (s i) = 0)
variable (h_Q_degree : Polynomial.degree Q = 2009)
variable (h_Q_roots : ∀ (j : Fin 2009), Q.eval (s j + (s j)⁻¹) = 0)

/-- The main theorem: Q(1) / Q(-1) = 1 -/
theorem Q_ratio_is_one : Q.eval 1 / Q.eval (-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_ratio_is_one_l70_7039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_bricks_count_l70_7043

/-- Represents the time taken by the first bricklayer to build the wall alone -/
def time1 : ℚ := 8

/-- Represents the time taken by the second bricklayer to build the wall alone -/
def time2 : ℚ := 12

/-- Represents the reduction in efficiency when working together (in bricks per hour) -/
def efficiency_reduction : ℚ := 12

/-- Represents the time taken when working together -/
def time_together : ℚ := 6

/-- Calculates the number of bricks in the wall -/
noncomputable def num_bricks : ℚ :=
  time_together * (time1⁻¹ + time2⁻¹ - efficiency_reduction)

theorem wall_bricks_count : num_bricks = 288 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_bricks_count_l70_7043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_two_l70_7078

noncomputable def f (x : ℝ) : ℝ := (2 * x - 6) / (x - 5)

theorem inverse_f_undefined_at_two :
  ∀ g : ℝ → ℝ, (∀ x ≠ 2, g (f x) = x ∧ f (g x) = x) →
  ¬∃ y : ℝ, g 2 = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_two_l70_7078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l70_7072

-- Define the constant √5
noncomputable def sqrt5 : ℝ := Real.sqrt 5

-- Define x
noncomputable def x : ℝ := (2 + sqrt5) ^ 500

-- Define n as the floor of x
noncomputable def n : ℤ := ⌊x⌋

-- Define f
noncomputable def f : ℝ := x - n

-- Theorem statement
theorem x_times_one_minus_f_equals_one : x * (1 - f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l70_7072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_final_configuration_l70_7099

/-- Represents the configuration of stones on the infinite strip --/
def Configuration := ℤ → ℕ

/-- The golden ratio, used in the weight calculation --/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Calculates the weight of a configuration --/
noncomputable def weight (c : Configuration) : ℝ :=
  ∑' (i : ℤ), (c i : ℝ) * φ ^ i

/-- Represents the two allowed actions --/
inductive StoneAction
  | move1 (n : ℤ) : StoneAction
  | move2 (n : ℤ) : StoneAction

/-- Applies an action to a configuration --/
def applyAction (c : Configuration) (a : StoneAction) : Configuration :=
  match a with
  | StoneAction.move1 n => fun i =>
      if i = n - 1 || i = n then c i - 1
      else if i = n + 1 then c i + 1
      else c i
  | StoneAction.move2 n => fun i =>
      if i = n then c i - 2
      else if i = n + 1 || i = n - 2 then c i + 1
      else c i

/-- Predicate for whether a configuration is final (no more actions possible) --/
def isFinal (c : Configuration) : Prop :=
  ∀ n : ℤ, (c n < 2 ∧ c (n - 1) = 0) ∨ (c n < 2 ∧ c (n + 1) = 0)

/-- The main theorem to be proven --/
theorem unique_final_configuration (c : Configuration) :
  ∃! (f : Configuration), isFinal f ∧
    ∃ (actions : List StoneAction), f = actions.foldl applyAction c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_final_configuration_l70_7099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leon_total_payment_l70_7060

def toy_organizer_price : ℚ := 78
def gaming_chair_price : ℚ := 83
def desk_price : ℚ := 120
def bookshelf_price : ℚ := 95

def num_toy_organizers : ℕ := 3
def num_gaming_chairs : ℕ := 2

noncomputable def delivery_fee_rate (total : ℚ) : ℚ :=
  if total ≤ 300 then 3/100
  else if total ≤ 600 then 5/100
  else 7/100

def total_before_fee : ℚ :=
  toy_organizer_price * num_toy_organizers +
  gaming_chair_price * num_gaming_chairs +
  desk_price + bookshelf_price

noncomputable def delivery_fee : ℚ := delivery_fee_rate total_before_fee * total_before_fee

noncomputable def total_amount : ℚ := total_before_fee + delivery_fee

theorem leon_total_payment :
  total_amount = 658.05 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leon_total_payment_l70_7060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_condition_uniquely_determines_plane_l70_7066

-- Define the basic types
structure Point where

structure Line where

structure Plane where

-- Define the relationships
def collinear (p q r : Point) : Prop := sorry

def pointOnLine (p : Point) (l : Line) : Prop := sorry

def lineIntersectsLine (l1 l2 : Line) : Prop := sorry

def pointDeterminesPlane (p : Point) (π : Plane) : Prop := sorry

def lineDeterminesPlane (l : Line) (π : Plane) : Prop := sorry

-- Define the conditions
def condition1 (p q r : Point) : Prop := 
  ∃ (π1 π2 : Plane), π1 ≠ π2 ∧ pointDeterminesPlane p π1 ∧ pointDeterminesPlane q π1 ∧ pointDeterminesPlane r π1 ∧
                     pointDeterminesPlane p π2 ∧ pointDeterminesPlane q π2 ∧ pointDeterminesPlane r π2

def condition2 (l : Line) (p : Point) : Prop :=
  ∃ (π1 π2 : Plane), π1 ≠ π2 ∧ lineDeterminesPlane l π1 ∧ pointDeterminesPlane p π1 ∧
                     lineDeterminesPlane l π2 ∧ pointDeterminesPlane p π2

def condition3 (l1 l2 a : Line) : Prop :=
  lineIntersectsLine l1 a ∧ lineIntersectsLine l2 a ∧
  (∃ (π1 π2 π3 : Plane), (π1 ≠ π2 ∧ π2 ≠ π3 ∧ π1 ≠ π3) ∧
                         (lineDeterminesPlane l1 π1 ∧ lineDeterminesPlane l2 π1) ∧
                         (lineDeterminesPlane l1 π2 ∧ lineDeterminesPlane l2 π2) ∧
                         (lineDeterminesPlane l1 π3 ∧ lineDeterminesPlane l2 π3))

def condition4 (l1 l2 l3 : Line) : Prop :=
  lineIntersectsLine l1 l2 ∧ lineIntersectsLine l2 l3 ∧ lineIntersectsLine l3 l1 ∧
  (∃ (π1 π2 π3 : Plane), (π1 ≠ π2 ∧ π2 ≠ π3 ∧ π1 ≠ π3) ∧
                         (lineDeterminesPlane l1 π1 ∧ lineDeterminesPlane l2 π1 ∧ lineDeterminesPlane l3 π1) ∧
                         (lineDeterminesPlane l1 π2 ∧ lineDeterminesPlane l2 π2 ∧ lineDeterminesPlane l3 π2) ∧
                         (lineDeterminesPlane l1 π3 ∧ lineDeterminesPlane l2 π3 ∧ lineDeterminesPlane l3 π3))

-- The main theorem
theorem no_condition_uniquely_determines_plane :
  (∀ p q r : Point, condition1 p q r) ∧
  (∀ l : Line, ∀ p : Point, condition2 l p) ∧
  (∀ l1 l2 a : Line, condition3 l1 l2 a) ∧
  (∀ l1 l2 l3 : Line, condition4 l1 l2 l3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_condition_uniquely_determines_plane_l70_7066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_regions_l70_7092

/-- The maximum number of regions formed by n curves where each pair intersects at most m times -/
def maxRegions (n : ℕ) (m : ℕ) : ℕ :=
  (List.range (m + 1)).map (λ k => Nat.choose n k) |>.sum

/-- The number of intersections between two sets of n curves, where each pair intersects at most m times -/
def intersectionCount (n : ℕ) (m : ℕ) : ℕ := n * n * m

/-- The total number of regions formed by two sets of n curves, where:
    - Each pair within a set intersects at most twice
    - Each pair between sets intersects at most four times -/
def totalRegions (n : ℕ) : ℕ :=
  2 * maxRegions n 2 + intersectionCount n 4 + 1

theorem parabola_regions :
  totalRegions 50 = 15053 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_regions_l70_7092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_of_roots_l70_7041

def is_root (z : ℂ) : Prop :=
  z^6 - z^4 + z^2 - 1 = 0

theorem max_imaginary_part_of_roots :
  ∃ (θ : ℝ), -π/2 ≤ θ ∧ θ ≤ π/2 ∧
  (∀ (z : ℂ), is_root z → z.im ≤ Real.sin θ) ∧
  θ = π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_of_roots_l70_7041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_identity_l70_7061

theorem sin_plus_cos_identity (θ : ℝ) (a : ℝ) 
  (h1 : θ > π / 2) (h2 : θ < 3 * π / 4) (h3 : Real.sin (2 * θ) = a) : 
  Real.sin θ + Real.cos θ = Real.sqrt (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_identity_l70_7061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_efficiency_theorem_l70_7000

/-- Represents the efficiency of a worker in completing a job -/
structure WorkerEfficiency where
  value : ℝ

/-- Represents the time taken to complete a job -/
structure Time where
  value : ℝ

/-- The efficiency of worker A -/
def efficiency_A : WorkerEfficiency := ⟨2⟩

/-- The efficiency of worker B -/
def efficiency_B : WorkerEfficiency := ⟨1⟩

/-- The time taken by workers A and B together -/
def time_together : Time := ⟨9⟩

/-- The time taken by worker B alone -/
def time_B_alone : Time := ⟨27⟩

theorem worker_efficiency_theorem :
  efficiency_A.value = 2 * efficiency_B.value →
  (efficiency_A.value + efficiency_B.value) * time_together.value = efficiency_B.value * time_B_alone.value :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_efficiency_theorem_l70_7000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_2x_at_negative_26_over_7_l70_7030

-- Define the function h as noncomputable
noncomputable def h (x : ℝ) : ℝ := (3 * ((x - 2) / 5)) - 4

-- State the theorem
theorem h_equals_2x_at_negative_26_over_7 :
  h (-26 / 7) = 2 * (-26 / 7) := by
  -- Unfold the definition of h
  unfold h
  -- Simplify the expression
  simp [mul_div_assoc, add_div, sub_div]
  -- Prove the equality
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_2x_at_negative_26_over_7_l70_7030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_unoccupied_volume_l70_7083

/-- The volume of a cube given its side length -/
noncomputable def cubeVolume (side : ℝ) : ℝ := side^3

/-- The volume of the container -/
noncomputable def containerVolume : ℝ := cubeVolume 10

/-- The volume of water in the container -/
noncomputable def waterVolume : ℝ := containerVolume / 2

/-- The volume of a single ice cube -/
noncomputable def iceCubeVolume : ℝ := cubeVolume 2

/-- The total volume of all ice cubes -/
noncomputable def totalIceVolume : ℝ := 10 * iceCubeVolume

/-- The volume occupied by water and ice -/
noncomputable def occupiedVolume : ℝ := waterVolume + totalIceVolume

/-- The unoccupied volume in the container -/
noncomputable def unoccupiedVolume : ℝ := containerVolume - occupiedVolume

theorem container_unoccupied_volume : unoccupiedVolume = 420 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_unoccupied_volume_l70_7083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_length_l70_7047

/-- The area of a quadrilateral given its diagonal and two offsets -/
noncomputable def quadrilateralArea (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) : ℝ :=
  (1 / 2) * (offset1 + offset2) * diagonal

/-- Theorem: Given a quadrilateral with diagonal 26 cm, one offset 6 cm, 
    and area 195 cm², the other offset is 9 cm -/
theorem quadrilateral_offset_length 
  (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) 
  (h1 : diagonal = 26)
  (h2 : offset1 = 6)
  (h3 : area = 195)
  (h4 : area = quadrilateralArea diagonal offset1 9) :
  ∃ (offset2 : ℝ), offset2 = 9 ∧ area = quadrilateralArea diagonal offset1 offset2 := by
  sorry

#check quadrilateral_offset_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_length_l70_7047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l70_7051

theorem problem_statement (x y : ℝ) : 
  (∃ r : ℝ, 1 - 2*x = r ∧ y - 1 - 1 = r) →  -- arithmetic sequence condition
  (∃ q : ℝ, (abs (x+1) + abs (x-1)) / (y+3) = q ∧ 
            x / (abs (x+1) + abs (x-1)) = q) →  -- geometric sequence condition
  (x + 1) * (y + 1) = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l70_7051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_c_range_correct_l70_7097

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 2*x
def g (x : ℝ) : ℝ := -x^2 + 2*x

-- Define the solution set for part (Ⅰ)
def solution_set : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1/2}

-- Define the range of c for part (Ⅱ)
def c_range : Set ℝ := {c | c ≤ -9/8}

-- Theorem for part (Ⅰ)
theorem solution_set_correct :
  ∀ x, g x ≥ f x - |x - 1| ↔ x ∈ solution_set :=
by sorry

-- Theorem for part (Ⅱ)
theorem c_range_correct :
  ∀ c, (∀ x, g x + c ≤ f x - |x - 1|) ↔ c ∈ c_range :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_c_range_correct_l70_7097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_condition_l70_7080

/-- A quadratic function f(x) = ax^2 + bx + c where a > 0 and f(-1) = 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem sufficient_but_not_necessary_condition
  (a b c : ℝ) (ha : a > 0) (hf : QuadraticFunction a b c (-1) = 0) :
  (∀ x : ℝ, b < -2 * a → QuadraticFunction a b c 2 < 0) ∧
  ¬(∀ x : ℝ, QuadraticFunction a b c 2 < 0 → b < -2 * a) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_condition_l70_7080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base12_addition_l70_7059

/-- Represents a digit in base 12 --/
inductive Base12Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C
deriving Repr

/-- Represents a number in base 12 --/
def Base12Number := List Base12Digit

/-- Convert a Base12Number to a natural number --/
def base12ToNat (n : Base12Number) : Nat := sorry

/-- Convert a natural number to a Base12Number --/
def natToBase12 (n : Nat) : Base12Number := sorry

/-- Addition of two Base12Numbers --/
def addBase12 (a b : Base12Number) : Base12Number := sorry

theorem base12_addition : 
  let a := [Base12Digit.A, Base12Digit.D4, Base12Digit.D3]
  let b := [Base12Digit.D2, Base12Digit.B, Base12Digit.D7]
  let c := [Base12Digit.D1, Base12Digit.D8, Base12Digit.D9]
  let result := [Base12Digit.D1, Base12Digit.C, Base12Digit.D0, Base12Digit.D7]
  addBase12 (addBase12 a b) c = result := by sorry

#check base12_addition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base12_addition_l70_7059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_condition_l70_7013

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define the vectors and scalars
variable (a b : V) (l m : ℝ)

-- Define the condition that a and b are non-collinear
def non_collinear (a b : V) : Prop := ∀ (k : ℝ), a ≠ k • b

-- Define the vectors AB and AC
def AB (a b : V) (l : ℝ) : V := l • a + b
def AC (a b : V) (m : ℝ) : V := a + m • b

-- Define collinearity of three points
def collinear (A B C : V) : Prop := ∃ (t : ℝ), B - A = t • (C - A) ∨ C - A = t • (B - A)

-- State the theorem
theorem collinearity_condition (h_non_collinear : non_collinear a b) :
  collinear 0 (AB a b l) (AC a b m) ↔ l * m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_condition_l70_7013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_constant_tangency_line_passes_fixed_point_l70_7029

/-- Parabola defined by y = x^2 -/
def parabola : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

/-- Line y = -1 on which point A moves -/
def line : Set (ℝ × ℝ) := {p | p.2 = -1}

/-- Given a point A on the line y = -1, returns the slopes of the two tangents from A to the parabola -/
noncomputable def tangent_slopes (a : ℝ) : ℝ × ℝ := sorry

theorem tangent_slopes_product_constant (a : ℝ) :
  let (k₁, k₂) := tangent_slopes a
  k₁ * k₂ = -4 := by sorry

/-- Given a point A on the line y = -1, returns the two points of tangency on the parabola -/
noncomputable def tangency_points (a : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

theorem tangency_line_passes_fixed_point (a : ℝ) :
  let (p, q) := tangency_points a
  ∃ t : ℝ, (1 - t) • (p : ℝ × ℝ) + t • (q : ℝ × ℝ) = (0, 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_constant_tangency_line_passes_fixed_point_l70_7029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_area_ratio_l70_7055

-- Define the side lengths of the triangles
def large_side : ℝ := 10
def small_side : ℝ := 6

-- Define the areas of the triangles and trapezoid
noncomputable def area_large : ℝ := (Real.sqrt 3 / 4) * large_side^2
noncomputable def area_small : ℝ := (Real.sqrt 3 / 4) * small_side^2
noncomputable def area_trapezoid : ℝ := area_large - area_small

-- State the theorem
theorem triangle_trapezoid_area_ratio :
  area_small / area_trapezoid = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_area_ratio_l70_7055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_qualified_chip_l70_7021

/-- The frequencies of qualified chips from five samples -/
def frequencies : List ℚ := [957/1000, 963/1000, 956/1000, 961/1000, 962/1000]

/-- The number of samples -/
def num_samples : ℕ := 5

/-- Function to calculate the average of a list of rational numbers -/
def average (l : List ℚ) : ℚ := (l.sum) / l.length

/-- Function to round a rational number to 2 decimal places -/
def round_to_hundredths (x : ℚ) : ℚ := 
  let n := (x * 100).num
  let d := (x * 100).den
  ↑(n / d) / 100

/-- Theorem stating that the probability of selecting a qualified chip is 0.96 -/
theorem probability_of_qualified_chip :
  round_to_hundredths (average frequencies) = 96/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_qualified_chip_l70_7021
