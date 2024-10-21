import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_l190_19014

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / (x + 1)

theorem monotonicity_of_f (a : ℝ) :
  (∀ x₁ x₂ : ℝ, -1 < x₁ → x₁ < x₂ → a > 0 → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂ : ℝ, -1 < x₁ → x₁ < x₂ → a < 0 → f a x₁ > f a x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_l190_19014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_completion_time_l190_19006

/-- Represents the progress of a team on a given day -/
structure DailyProgress where
  meters : ℝ
  day : ℕ

/-- Calculates the total distance dug by a team over a period of days -/
noncomputable def totalDistance (initialMeters : ℝ) (incrementFactor : ℝ) (days : ℕ) : ℝ :=
  if days = 0 then 0
  else initialMeters * (1 - incrementFactor^days) / (1 - incrementFactor)

/-- Represents the tunnel digging scenario -/
structure TunnelDigging where
  tunnelLength : ℝ
  teamAInitial : ℝ
  teamBInitial : ℝ
  teamAFactor : ℝ
  teamBFactor : ℝ

/-- Calculates the time needed to complete the tunnel -/
noncomputable def completionTime (scenario : TunnelDigging) : ℝ :=
  let fullDays := 4
  let remainingDistance := scenario.tunnelLength - 
    (totalDistance scenario.teamAInitial scenario.teamAFactor fullDays + 
     totalDistance scenario.teamBInitial scenario.teamBFactor fullDays)
  let lastDayEfficiency := scenario.teamAInitial * scenario.teamAFactor^fullDays + 
                           scenario.teamBInitial * scenario.teamBFactor^fullDays
  fullDays + remainingDistance / lastDayEfficiency

/-- The main theorem stating the completion time for the given scenario -/
theorem tunnel_completion_time : 
  let scenario := TunnelDigging.mk 300 10 10 2 1.5
  completionTime scenario = 4 + 110 / 337 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_completion_time_l190_19006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_sum_l190_19047

section EllipseTheorem

variable (a b c : ℝ)
variable (F₁ F₂ : ℝ × ℝ)

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci
def foci_condition : Prop :=
  F₁ = (-c, 0) ∧ F₂ = (c, 0) ∧ a > b ∧ b > 0

-- Define a point above x-axis
def above_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 > 0

-- Define parallel lines
def parallel (p₁ p₂ q₁ q₂ : ℝ × ℝ) : Prop :=
  (p₂.1 - p₁.1) * (q₂.2 - q₁.2) = (p₂.2 - p₁.2) * (q₂.1 - q₁.1)

-- Define intersection point
def intersect (p₁ p₂ q₁ q₂ : ℝ × ℝ) (r : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ, r = (p₁.1 + t * (p₂.1 - p₁.1), p₁.2 + t * (p₂.2 - p₁.2)) ∧
             r = (q₁.1 + s * (q₂.1 - q₁.1), q₁.2 + s * (q₂.2 - q₁.2))

-- Distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2)

theorem ellipse_constant_sum 
  (A B P : ℝ × ℝ) 
  (h₁ : is_on_ellipse a b A.1 A.2)
  (h₂ : is_on_ellipse a b B.1 B.2)
  (h₃ : above_x_axis A)
  (h₄ : above_x_axis B)
  (h₅ : foci_condition a b c F₁ F₂)
  (h₆ : parallel A F₁ B F₂)
  (h₇ : intersect A F₂ B F₁ P) :
  ∃ k : ℝ, distance P F₁ + distance P F₂ = k := by
  sorry

end EllipseTheorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_sum_l190_19047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l190_19024

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions on f
variable (h1 : ∀ x, f (-x) + f x = 0)
variable (h2 : ∀ x₁ x₂, (x₁ - x₂) * (f x₁ - f x₂) < 0)
variable (h3 : ∀ (t : ℝ) (ht : t ∈ Set.Icc 0 2) (m : ℝ), f (2 * t^2 - 4) + f (4 * m - 2 * t) ≥ f 0)

-- State the theorem
theorem max_m_value :
  ∃ m₀ : ℝ, m₀ ≤ 0 ∧ ∀ m : ℝ, (∀ t ∈ Set.Icc 0 2, f (2 * t^2 - 4) + f (4 * m - 2 * t) ≥ f 0) → m ≤ m₀ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l190_19024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruence_l190_19030

def mySequence : List ℕ := List.range 21 |>.map (λ i => 4 + 10 * i)

theorem product_congruence :
  (mySequence.prod : ℤ) % 6 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruence_l190_19030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l190_19015

/-- Proves that the speed of a train is 48 km/h, given the specified conditions. -/
theorem train_speed (distance : ℝ) (ship_speed : ℝ) (time_difference : ℝ) :
  distance = 480 →
  ship_speed = 60 →
  time_difference = 2 →
  (distance / ship_speed + time_difference) * (distance / (distance / ship_speed + time_difference)) = 48 := by
  intros h1 h2 h3
  -- Define intermediate variables
  let ship_time := distance / ship_speed
  let train_time := ship_time + time_difference
  let train_speed := distance / train_time
  -- Prove the theorem
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l190_19015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l190_19004

/-- The number of positive integer solutions (x, y, z) satisfying the given conditions -/
def count_solutions : ℕ := 336847

/-- The sum of x, y, and z is 2010 -/
def sum_constraint (x y z : ℕ) : Prop := x + y + z = 2010

/-- The order constraint x ≤ y ≤ z -/
def order_constraint (x y z : ℕ) : Prop := x ≤ y ∧ y ≤ z

/-- The set of all positive integer solutions satisfying the constraints -/
def solution_set : Set (ℕ × ℕ × ℕ) :=
  {xyz | xyz.1 > 0 ∧ xyz.2.1 > 0 ∧ xyz.2.2 > 0 ∧
         sum_constraint xyz.1 xyz.2.1 xyz.2.2 ∧
         order_constraint xyz.1 xyz.2.1 xyz.2.2}

/-- The main theorem stating that the number of solutions is equal to count_solutions -/
theorem solution_count : Nat.card solution_set = count_solutions := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l190_19004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l190_19068

theorem trig_simplification (x : ℝ) (h : Real.cos (x / 2) ≠ 0) :
  (Real.sin (2 * x) + Real.sin (3 * x)) / (1 + Real.cos (2 * x) + Real.cos (3 * x)) =
  Real.sin (5 * x / 2) / (1 + Real.cos (5 * x / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l190_19068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_F_l190_19081

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 3^(x - b)

-- Define the inverse function of f
noncomputable def f_inv (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log 3 + b

-- Define the function F
noncomputable def F (b : ℝ) (x : ℝ) : ℝ := (f_inv b x)^2 - f_inv b (x^2)

-- State the theorem
theorem range_of_F (b : ℝ) :
  (∃ x, x ∈ Set.Icc 2 4 ∧ f b x = 1) →
  Set.range (F b) = Set.Icc 2 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_F_l190_19081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_couple_functions_range_l190_19061

noncomputable def f (x : ℝ) := Real.exp (x - 2) + x - 3
noncomputable def g (a x : ℝ) := a * x - Real.log x

def couple_functions (f g : ℝ → ℝ) : Prop :=
  ∃ α β, f α = 0 ∧ g β = 0 ∧ |α - β| ≤ 1

theorem couple_functions_range (a : ℝ) :
  couple_functions f (g a) → a ∈ Set.Icc 0 (1 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_couple_functions_range_l190_19061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l190_19066

-- Define the solution set of (x-2)(x-5) ≤ 0
def solution_set : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}

-- Define the condition for the system of inequalities
def system_condition (a : ℝ) : Prop :=
  ∀ x, x ∈ solution_set ↔ (x - 2) * (x - 5) ≤ 0 ∧ x * (x - a) ≥ 0

-- Theorem statement
theorem range_of_a :
  {a : ℝ | system_condition a} = Set.Iic 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l190_19066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l190_19002

/-- The function f(x) as defined in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * (Real.cos (ω * x / 2))^2 + Real.cos (ω * x + Real.pi / 3) - 1

/-- Triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem problem_solution (ω : ℝ) (triangle : Triangle) (A B : ℝ) :
  ω > 0 →
  f ω A = 0 →
  f ω B = 0 →
  |A - B| = Real.pi / 2 →
  f ω triangle.A = -3/2 →
  triangle.c = 3 →
  1/2 * triangle.b * triangle.c * Real.sin triangle.A = 3 * Real.sqrt 3 →
  ω = 2 ∧ triangle.a = Real.sqrt 13 := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l190_19002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_division_theorem_l190_19087

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ
deriving Inhabited

/-- Represents a configuration of cubes that fill a larger cube -/
structure CubeConfiguration where
  cubes : List Cube
  total_volume : ℕ
deriving Inhabited

def volume (c : Cube) : ℕ := c.edge ^ 3

def total_volume (config : CubeConfiguration) : ℕ :=
  config.cubes.map volume |>.sum

def is_valid_configuration (config : CubeConfiguration) : Prop :=
  config.total_volume = config.total_volume ∧
  config.cubes.all (fun c => c.edge > 0 ∧ c.edge ≤ 5) ∧
  ¬(config.cubes.all (fun c => c.edge = config.cubes.head!.edge))

theorem cube_division_theorem :
  ∃ (config : CubeConfiguration),
    is_valid_configuration config ∧
    config.total_volume = 5^3 ∧
    config.cubes.length = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_division_theorem_l190_19087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_constant_dot_product_specific_line_equation_l190_19098

/-- Definition of the circle C -/
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

/-- Definition of the line l passing through A(0,1) with slope k -/
def line_l (k x y : ℝ) : Prop := y = k * x + 1

/-- Definition of a point being on the line l -/
def on_line (k x y : ℝ) : Prop := line_l k x y

/-- Definition of a point being on the circle C -/
def on_circle (x y : ℝ) : Prop := circle_C x y

/-- Definition of the dot product of two 2D vectors -/
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem line_circle_intersection (k : ℝ) :
  (∃ x y : ℝ, on_line k x y ∧ on_circle x y) ↔ 
  (4 - Real.sqrt 7) / 3 < k ∧ k < (4 + Real.sqrt 7) / 3 :=
sorry

theorem constant_dot_product (k : ℝ) 
  (h : (4 - Real.sqrt 7) / 3 < k ∧ k < (4 + Real.sqrt 7) / 3) :
  ∃ x1 y1 x2 y2 : ℝ, 
    on_line k x1 y1 ∧ on_circle x1 y1 ∧
    on_line k x2 y2 ∧ on_circle x2 y2 ∧
    dot_product x1 (y1 - 1) x2 (y2 - 1) = 7 :=
sorry

theorem specific_line_equation :
  ∃ k x1 y1 x2 y2 : ℝ,
    on_line k x1 y1 ∧ on_circle x1 y1 ∧
    on_line k x2 y2 ∧ on_circle x2 y2 ∧
    dot_product x1 y1 x2 y2 = 12 ∧
    k = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_constant_dot_product_specific_line_equation_l190_19098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_part1_range_of_a_part2_l190_19065

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 : ℝ)^x * |(2 : ℝ)^x - a| + (2 : ℝ)^(x + 1) - 3

-- Part 1: Range of f when a = 4 and x ∈ [1, 3]
theorem range_of_f_part1 :
  ∃ (y : ℝ), y ∈ Set.Icc 5 45 ↔ ∃ (x : ℝ), x ∈ Set.Icc 1 3 ∧ f 4 x = y := by
sorry

-- Part 2: Range of a when f is monotonic on ℝ
theorem range_of_a_part2 :
  ∀ (a : ℝ), (∀ (x y : ℝ), x < y → f a x < f a y) ↔ a ∈ Set.Icc (-2) 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_part1_range_of_a_part2_l190_19065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_length_l190_19040

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- angle A in radians
  a : ℝ  -- side length opposite to angle A
  c : ℝ  -- side length opposite to angle C

-- Define the properties of the specific triangle
noncomputable def special_triangle : Triangle where
  A := Real.arcsin (Real.sqrt 11 / 6)  -- This is derived from the given conditions
  a := 27
  c := 48

-- Theorem statement
theorem side_b_length (t : Triangle) (h1 : t.A = special_triangle.A) 
    (h2 : t.a = special_triangle.a) (h3 : t.c = special_triangle.c) : 
  let b := 27 * 4 * (5/6) * (2 * (5/6)^2 - 1)
  b = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_length_l190_19040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_unit_vectors_l190_19095

theorem dot_product_of_unit_vectors 
  (a b c e : ℝ × ℝ × ℝ) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = 1) 
  (hc : ‖c‖ = 1) 
  (he : ‖e‖ = 1) 
  (hab : a • b = -1/8) 
  (hac : a • c = -1/8) 
  (hbc : b • c = -1/8) 
  (hbe : b • e = -1/8) 
  (hce : c • e = -1/8) 
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ e ∧ b ≠ c ∧ b ≠ e ∧ c ≠ e) : 
  a • e = -35/34 := by
  sorry

#check dot_product_of_unit_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_unit_vectors_l190_19095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_directrix_l190_19054

/-- A parabola passing through three points with axis parallel to y-axis -/
structure Parabola where
  -- The parabola passes through these three points
  p1 : ℝ × ℝ := (0, 6)
  p2 : ℝ × ℝ := (1, 0)
  p3 : ℝ × ℝ := (4, 6)
  -- The axis is parallel to y-axis (implied by the structure)

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (2, -15/8)

/-- The directrix of a parabola -/
noncomputable def directrix (p : Parabola) : ℝ → ℝ := λ _ ↦ -17/8

/-- Theorem stating that the focus and directrix are correct for the given parabola -/
theorem parabola_focus_and_directrix (p : Parabola) :
  focus p = (2, -15/8) ∧ (∀ x, directrix p x = -17/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_directrix_l190_19054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_l190_19064

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

noncomputable def y (x : ℝ) : ℝ := f x - f (Real.exp 1 - x)

theorem three_zeros :
  ∃ (a b c : ℝ), a ∈ Set.Ioo 0 (Real.exp 1) ∧
                  b ∈ Set.Ioo 0 (Real.exp 1) ∧
                  c ∈ Set.Ioo 0 (Real.exp 1) ∧
                  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                  y a = 0 ∧ y b = 0 ∧ y c = 0 ∧
                  ∀ x ∈ Set.Ioo 0 (Real.exp 1), y x = 0 → x = a ∨ x = b ∨ x = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_l190_19064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l190_19093

/-- The time taken for two trains to pass each other completely -/
noncomputable def train_passing_time (train_length : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let total_distance := 2 * train_length
  total_distance / (relative_speed * 1000 / 3600)

/-- Theorem stating that the time taken for two trains of length 170 m each, 
    moving in opposite directions at speeds of 55 km/h and 50 km/h respectively, 
    to pass each other completely is approximately 11.66 seconds -/
theorem train_passing_time_approx :
  let result := train_passing_time 170 55 50
  abs (result - 11.66) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_passing_time 170 55 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l190_19093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l190_19083

/-- An arithmetic sequence with its first term and common difference -/
structure ArithmeticSequence where
  a1 : ℚ
  d : ℚ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a1 + (n - 1 : ℚ) * seq.d

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a1 + (n - 1 : ℚ) * seq.d)

theorem arithmetic_sequence_problem (seq : ArithmeticSequence) :
  nth_term seq 2 + sum_n_terms seq 3 = 4 ∧
  nth_term seq 3 + sum_n_terms seq 5 = 12 →
  nth_term seq 4 + sum_n_terms seq 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l190_19083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_and_range_condition_l190_19090

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x

def g (a : ℝ) (x : ℝ) : ℝ := a * x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f x - g a x

theorem minimum_distance_and_range_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≥ 0 → x₂ ≥ 0 → a = 1 → f x₁ = g 1 x₂ → |x₂ - x₁| ≥ 1) ∧
  (∀ x : ℝ, x ≥ 0 → F a x > F a (-x)) → a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_and_range_condition_l190_19090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_negative_real_parts_imply_inequalities_l190_19055

theorem cubic_roots_negative_real_parts_imply_inequalities
  (a b c d : ℝ) 
  (h : ∀ z : ℂ, a * z^3 + b * z^2 + c * z + d = 0 → Complex.re z < 0) :
  a * b > 0 ∧ b * c > a * d ∧ a * d > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_negative_real_parts_imply_inequalities_l190_19055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_reduction_time_l190_19020

/-- Represents the volume of sand in the container at time t -/
noncomputable def sand_volume (a b t : ℝ) : ℝ := a * Real.exp (-b * t)

theorem sand_reduction_time (a b : ℝ) (ha : a > 0) :
  (∃ b > 0, sand_volume a b 8 = a / 2) →
  (∃ t > 0, sand_volume a b t = a / 8 ∧ t = 24) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_reduction_time_l190_19020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pumpkin_price_proof_l190_19039

/-- Proves that the price of each regular pumpkin is $4.00 given the conditions of Peter's pumpkin sales. -/
theorem pumpkin_price_proof (jumbo_price : ℚ) (total_pumpkins : ℕ) (total_revenue : ℚ) (regular_pumpkins : ℕ) :
  jumbo_price = 9 ∧ total_pumpkins = 80 ∧ total_revenue = 395 ∧ regular_pumpkins = 65 →
  ∃ regular_price : ℚ, regular_price = 4 ∧
    regular_price * regular_pumpkins + jumbo_price * (total_pumpkins - regular_pumpkins) = total_revenue :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pumpkin_price_proof_l190_19039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_is_constant_l190_19009

-- Define the circle P
def circle_P (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 8

-- Define the trajectory of the moving circle's center
def trajectory (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line intersecting the trajectory
def intersecting_line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the condition for points A and B
def slope_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ / x₁) * (y₂ / x₂) = -1/2

-- Main theorem
theorem area_of_triangle_AOB_is_constant
  (k m x₁ y₁ x₂ y₂ : ℝ)
  (h1 : trajectory x₁ y₁)
  (h2 : trajectory x₂ y₂)
  (h3 : intersecting_line k m x₁ y₁)
  (h4 : intersecting_line k m x₂ y₂)
  (h5 : slope_product_condition x₁ y₁ x₂ y₂)
  : ∃ (S : ℝ), S = (3 * Real.sqrt 2) / 2 ∧
    S = (1/2) * Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) *
        (abs m / Real.sqrt (1 + k^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_is_constant_l190_19009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_number_l190_19099

/-- A function that returns true if a number is even, false otherwise -/
def is_even (n : ℕ) : Bool :=
  n % 2 = 0

/-- The set of balls in the urn -/
def urn : Finset ℕ :=
  Finset.range 9

/-- The set of all possible pairs of balls -/
def all_pairs : Finset (ℕ × ℕ) :=
  Finset.product urn urn

/-- The set of pairs that form even numbers -/
def even_pairs : Finset (ℕ × ℕ) :=
  all_pairs.filter (fun p => is_even (p.1 * 10 + p.2))

/-- The probability of forming an even number -/
theorem prob_even_number : 
  (Finset.card even_pairs : ℚ) / (Finset.card all_pairs : ℚ) = 4 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_number_l190_19099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l190_19072

/-- The angle between two vectors in ℝ³ --/
noncomputable def angle_between (a b : ℝ × ℝ × ℝ) : ℝ :=
  Real.arccos ((a.fst * b.fst + a.snd.fst * b.snd.fst + a.snd.snd * b.snd.snd) / 
    (Real.sqrt (a.fst^2 + a.snd.fst^2 + a.snd.snd^2) * Real.sqrt (b.fst^2 + b.snd.fst^2 + b.snd.snd^2)))

theorem angle_between_vectors :
  let a : ℝ × ℝ × ℝ := (0, 2, 1)
  let b : ℝ × ℝ × ℝ := (-1, 1, -2)
  angle_between a b = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l190_19072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_sum_even_visible_sum_not_101_l190_19057

/-- Represents a small cube with numbers from 1 to 6 on its faces -/
structure SmallCube where
  faces : Fin 6 → Nat
  face_range : ∀ i, 1 ≤ faces i ∧ faces i ≤ 6

/-- Predicate to determine if two positions in the large cube are adjacent -/
def adjacent : Fin 8 → Fin 8 → Prop := sorry

/-- Represents the large 2x2x2 cube constructed from 8 small cubes -/
structure LargeCube where
  small_cubes : Fin 8 → SmallCube
  matching_faces : ∀ i j, adjacent i j → 
    ∃ f1 f2, (small_cubes i).faces f1 = (small_cubes j).faces f2

/-- The sum of visible face numbers on the large cube -/
def visible_sum (c : LargeCube) : Nat := sorry

theorem visible_sum_even (c : LargeCube) : Even (visible_sum c) := by
  sorry

theorem visible_sum_not_101 (c : LargeCube) : visible_sum c ≠ 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_sum_even_visible_sum_not_101_l190_19057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_proof_l190_19012

-- Define the arithmetic progression
def arithmetic_progression (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Define the problem statement
theorem common_difference_proof (x y : ℤ) :
  (280 * x^2 - 61 * x * y + 3 * y^2 - 13 = 0) →
  (∃ (a₁ d : ℤ), arithmetic_progression a₁ d 4 = x ∧ 
                 arithmetic_progression a₁ d 9 = y ∧
                 (∀ (n : ℕ), n > 0 → arithmetic_progression a₁ d n ∈ Set.range Int.ofNat) ∧
                 (∀ (n : ℕ), n > 0 → arithmetic_progression a₁ d n > arithmetic_progression a₁ d (n + 1))) →
  (∃ (a₁ : ℤ), arithmetic_progression a₁ (-5) 4 = x ∧ 
               arithmetic_progression a₁ (-5) 9 = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_proof_l190_19012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l190_19048

-- Problem 1
theorem problem_1 : (9/4 : ℝ)^(1/2) + (34/100 : ℝ)^0 = 5/2 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a^(2/3) * b^(1/3))^2 * a^(1/3) / (b^(1/3) * (a^2 * b^4)^(1/3)) = a / b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l190_19048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l190_19023

open Real InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

noncomputable def angle_between (v w : V) : ℝ := arccos ((inner v w) / (norm v * norm w))

theorem vector_problem (a b : V) 
  (h_angle : angle_between a b = π / 3)
  (h_norm_a : norm a = 1)
  (h_norm_diff : norm (2 • a - b) = 2 * sqrt 3) :
  norm b = 4 ∧ angle_between b (2 • a - b) = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l190_19023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_one_by_one_tiles_23x23_l190_19036

/-- Represents a tile configuration on a square grid --/
structure TileConfiguration (n : ℕ) where
  tiles : List (ℕ × ℕ × ℕ)  -- List of (size, count, area)
  covers_all : (tiles.map (λ t => t.2.1 * t.2.2)).sum = n * n
  no_overlap : (tiles.map (λ t => t.2.1 * t.2.2)).sum ≤ n * n

/-- The minimum number of 1×1 tiles needed for a 23×23 floor --/
def min_one_by_one_tiles : ℕ := 1

/-- Theorem stating the minimum number of 1×1 tiles needed --/
theorem min_one_by_one_tiles_23x23 :
  ∀ (config : TileConfiguration 23),
    (∃ (i : ℕ), (1, i, 1) ∈ config.tiles) ∧
    (∀ (i : ℕ), (1, i, 1) ∈ config.tiles → i ≥ min_one_by_one_tiles) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_one_by_one_tiles_23x23_l190_19036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_one_after_extensions_l190_19043

/-- Extension process: replace each element k with (1, 2, ..., k) -/
def extend (seq : List Nat) : List Nat :=
  seq.bind (fun k => List.range' 1 k)

/-- The initial sequence (1, 2, ..., 9) -/
def initialSeq : List Nat := List.range' 1 9

/-- Apply the extension process n times -/
def extendNTimes (n : Nat) (seq : List Nat) : List Nat :=
  match n with
  | 0 => seq
  | n+1 => extendNTimes n (extend seq)

/-- Count the number of 1's in a sequence -/
def countOnes (seq : List Nat) : Nat :=
  seq.filter (· = 1) |>.length

theorem probability_of_one_after_extensions :
  let finalSeq := extendNTimes 2017 initialSeq
  (countOnes finalSeq : Rat) / (finalSeq.length : Rat) = 2018 / 2026 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_one_after_extensions_l190_19043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_regions_l190_19080

/-- The number of regions formed by k intersecting circles in a plane -/
def f (k : ℕ) : ℕ := k^2 - k + 2

/-- Two circles intersect at exactly two points -/
axiom two_intersection : ℕ → ℕ → Prop

/-- No three circles pass through the same point -/
axiom no_common_point : ℕ → ℕ → ℕ → Prop

/-- Theorem about the number of regions formed by intersecting circles -/
theorem circle_regions (k : ℕ) :
  (∀ i j, i ≠ j → two_intersection i j) →
  (∀ i j l, i ≠ j ∧ j ≠ l ∧ i ≠ l → no_common_point i j l) →
  f (k + 1) = f k + 2 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_regions_l190_19080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_construction_condition_l190_19026

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point A
noncomputable def A : ℝ × ℝ := sorry

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the circle
noncomputable def k : Circle := sorry

-- Theorem statement
theorem square_construction_condition (k : Circle) (A : ℝ × ℝ) :
  (∃ (B C D : ℝ × ℝ), 
    (distance A B = distance B C ∧ 
     distance B C = distance C D ∧ 
     distance C D = distance D A) ∧
    (∃ (X Y : ℝ × ℝ), X ≠ Y ∧ 
      distance X k.center = k.radius ∧ 
      distance Y k.center = k.radius ∧
      ((X = B ∨ X = C ∨ X = D) ∧ (Y = B ∨ Y = C ∨ Y = D)))) ↔
    k.radius * (Real.sqrt 2 - 1) ≤ distance A k.center ∧ 
    distance A k.center ≤ k.radius * (Real.sqrt 2 + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_construction_condition_l190_19026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l190_19092

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (2 * t.c - t.b) / t.a = Real.cos t.B / Real.cos t.A ∧
  t.a = 2 * Real.sqrt 5

-- Helper function to calculate triangle area
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.A = π / 3 ∧
  ∃ (max_area : ℝ), max_area = 5 * Real.sqrt 3 ∧
    ∀ (t' : Triangle), satisfies_conditions t' →
      triangle_area t' ≤ max_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l190_19092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_congruence_l190_19076

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), n = 994 ∧ 
  (∀ m : ℕ, m ≤ 999 ∧ m ≥ 100 ∧ (75 * m) % 450 = 300 → m ≤ n) ∧
  (75 * n) % 450 = 300 ∧ n ≤ 999 ∧ n ≥ 100 :=
by
  -- We claim that n = 994 satisfies all conditions
  use 994
  constructor
  · rfl  -- trivial: 994 = 994
  constructor
  · intro m hm
    -- We'll prove this later
    sorry
  constructor
  · -- Prove (75 * 994) % 450 = 300
    norm_num
  constructor
  · norm_num  -- 994 ≤ 999
  · norm_num  -- 994 ≥ 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_congruence_l190_19076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circles_problem_l190_19045

-- Define the circle and point types
def Circle := ℝ × ℝ × ℝ  -- center_x, center_y, radius
def Point := ℝ × ℝ

-- Define the given circles and points
noncomputable def circle1 : Circle := sorry
noncomputable def circle2 : Circle := sorry
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry
noncomputable def F : Point := sorry

-- Define the conditions
def circles_have_radius_5 (c1 c2 : Circle) : Prop := 
  c1.2 = 5 ∧ c2.2 = 5

def points_on_circles (A B C D : Point) (c1 c2 : Circle) : Prop := sorry

def B_on_CD (B C D : Point) : Prop := sorry

def angle_CAD_is_90 (C A D : Point) : Prop := sorry

def F_perpendicular_to_CD (B C D F : Point) : Prop := sorry

def BF_equals_BD (B D F : Point) : Prop := sorry

def A_F_opposite_sides (A C D F : Point) : Prop := sorry

def BC_equals_6 (B C : Point) : Prop := sorry

-- Helper function definitions
noncomputable def distance (p1 p2 : Point) : ℝ := sorry
noncomputable def area_triangle (p1 p2 p3 : Point) : ℝ := sorry

-- Define the theorem
theorem intersection_circles_problem 
  (c1 c2 : Circle) (A B C D F : Point) 
  (h1 : circles_have_radius_5 c1 c2)
  (h2 : points_on_circles A B C D c1 c2)
  (h3 : B_on_CD B C D)
  (h4 : angle_CAD_is_90 C A D)
  (h5 : F_perpendicular_to_CD B C D F)
  (h6 : BF_equals_BD B D F)
  (h7 : A_F_opposite_sides A C D F)
  (h8 : BC_equals_6 B C) :
  (distance C F = 10) ∧ (area_triangle A C F = 49) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circles_problem_l190_19045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l190_19025

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x - 1)

def domain : Set ℝ := { x | 2 ≤ x ∧ x < 5 }

theorem function_properties :
  (∃ (m : ℝ), m = 4 ∧ ∀ x ∈ domain, f x ≤ m) ∧
  (¬ ∃ (m : ℝ), ∀ x ∈ domain, m ≤ f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l190_19025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_placement_and_coverage_area_l190_19018

/-- The number of radars --/
noncomputable def n : ℕ := 5

/-- The radius of each radar's coverage area in km --/
noncomputable def r : ℝ := 13

/-- The width of the coverage ring in km --/
noncomputable def w : ℝ := 10

/-- The central angle between two adjacent radars in radians --/
noncomputable def θ : ℝ := 2 * Real.pi / n

theorem radar_placement_and_coverage_area :
  ∃ (d A : ℝ),
    -- d is the distance from the center to each radar
    d = 12 / Real.sin (θ / 2) ∧
    -- A is the area of the coverage ring
    A = 240 * Real.pi / Real.tan (θ / 2) ∧
    -- The coverage ring has the specified width
    r + w = d + r ∧
    -- The radars form a regular polygon
    2 * r * Real.sin (θ / 2) = d * Real.sin θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_placement_and_coverage_area_l190_19018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_ring_matching_l190_19060

theorem concentric_ring_matching (n : ℕ) (h : n ≥ 3) :
  (∃ (outer : Fin n → Fin n),
    Function.Bijective outer ∧
    (∀ (k : Fin n),
      ∃! (j : Fin n),
        outer (k + j) = k)) ↔
  Odd n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_ring_matching_l190_19060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_two_fifty_l190_19041

/-- Represents the outcome of rolling a fair 8-sided die -/
inductive DieRoll
| one | two | three | four | five | six | seven | eight

/-- The probability of rolling any specific number on a fair 8-sided die -/
noncomputable def rollProbability : ℝ := 1 / 8

/-- The winnings for a given roll -/
def winnings (roll : DieRoll) : ℝ :=
  match roll with
  | DieRoll.two => 2
  | DieRoll.four => 4
  | DieRoll.six => 6
  | DieRoll.eight => 8
  | _ => 0

/-- The expected value of winnings -/
noncomputable def expectedValue : ℝ :=
  rollProbability * (winnings DieRoll.one +
                     winnings DieRoll.two +
                     winnings DieRoll.three +
                     winnings DieRoll.four +
                     winnings DieRoll.five +
                     winnings DieRoll.six +
                     winnings DieRoll.seven +
                     winnings DieRoll.eight)

theorem expected_value_is_two_fifty :
  expectedValue = 2.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_two_fifty_l190_19041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l190_19016

theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  (1/2 * a * c * Real.sin (B * π / 180) = Real.sqrt 3) →
  (B = 60) →
  (a^2 + c^2 = 3 * a * c) →
  b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l190_19016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_intersection_sum_l190_19005

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in 2D space -/
structure Line where
  m : ℚ  -- slope
  b : ℚ  -- y-intercept

/-- Calculates the area of a quadrilateral given its vertices -/
def quadrilateralArea (a b c d : Point) : ℚ :=
  (1/2) * abs ((a.x * b.y - a.y * b.x) + (b.x * c.y - b.y * c.x) + 
               (c.x * d.y - c.y * d.x) + (d.x * a.y - d.y * a.x))

/-- Determines if a line divides a quadrilateral into two equal areas -/
def dividesEqualArea (l : Line) (a b c d : Point) : Prop :=
  ∃ (p : Point), p.y = l.m * p.x + l.b ∧ 
    quadrilateralArea a b p d = quadrilateralArea a p c d

/-- Theorem statement -/
theorem equal_area_intersection_sum (a b c d : Point) 
  (h1 : a.x = 0 ∧ a.y = 0)
  (h2 : b.x = 2 ∧ b.y = 3)
  (h3 : c.x = 4 ∧ c.y = 4)
  (h4 : d.x = 5 ∧ d.y = 1)
  (l : Line)
  (h5 : dividesEqualArea l a b c d)
  (p q r s : ℕ)
  (h6 : ∃ (i : Point), i.x = p/q ∧ i.y = r/s ∧ i.y = l.m * i.x + l.b)
  (h7 : Nat.Coprime p q ∧ Nat.Coprime r s) :
  p + q + r + s = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_intersection_sum_l190_19005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_together_in_eight_l190_19079

/-- The probability of three specific people sitting together in a row of eight people -/
theorem probability_three_together_in_eight (n : ℕ) (h : n = 8) :
  (Nat.factorial 6 * Nat.factorial 3) / Nat.factorial n = 1 / (9375 / 1000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_together_in_eight_l190_19079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l190_19017

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

-- Theorem stating that x = π/3 is a symmetry axis of f(x)
theorem symmetry_axis_of_f :
  ∀ (x : ℝ), f (Real.pi/3 + x) = f (Real.pi/3 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l190_19017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_m_range_l190_19013

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2 + Real.cos x

-- State the theorem
theorem f_inequality_implies_m_range (m : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f (x^2) ≥ f (1 - m*x)) →
  m ∈ Set.Icc 0 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_m_range_l190_19013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l190_19031

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_proof :
  let a := 31.5
  let b := 27.8
  let c := 10.3
  abs (triangle_area a b c - 141.65) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l190_19031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l190_19011

-- Define the curve
noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log (x + 1)

-- Define the derivative of the curve
noncomputable def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * x + a / (x + 1)

theorem tangent_line_implies_a_value (a : ℝ) :
  (curve_derivative a 0 = 3) → a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l190_19011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l190_19029

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := 
  x^(Real.log x / Real.log 13) + 
  7 * (x^(1/3))^(Real.log x / Real.log 13) - 
  7 - 
  (13^(1/3))^(2 * Real.log x / Real.log 13)

-- Define the solution set
noncomputable def solution_set : Set ℝ := 
  {x | 0 < x ∧ x ≤ 13^(-Real.sqrt (Real.log 7 / Real.log 13))} ∪ 
  {1} ∪ 
  {x | x ≥ 13^(Real.sqrt (Real.log 7 / Real.log 13))}

-- State the theorem
theorem inequality_solution :
  ∀ x > 0, f x ≤ 0 ↔ x ∈ solution_set :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l190_19029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_area_theorem_l190_19019

/-- Right square-based pyramid -/
structure RightSquarePyramid where
  baseEdge : ℝ
  lateralEdge : ℝ

/-- Calculate the total area of the four triangular faces of a right square-based pyramid -/
noncomputable def totalTriangularArea (p : RightSquarePyramid) : ℝ :=
  let h := Real.sqrt (p.lateralEdge ^ 2 - (p.baseEdge / 2) ^ 2)
  4 * (1 / 2 * p.baseEdge * h)

/-- Theorem: The total area of the four triangular faces of a right square-based pyramid
    with base edges of 8 units and lateral edges of 5 units is equal to 48 square units -/
theorem pyramid_area_theorem :
  let p : RightSquarePyramid := { baseEdge := 8, lateralEdge := 5 }
  totalTriangularArea p = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_area_theorem_l190_19019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l190_19071

theorem project_completion_time (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (1 / (1 / m + 1 / n)) = m * n / (m + n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l190_19071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_theorem_l190_19056

noncomputable section

open Real EuclideanGeometry

theorem triangle_point_theorem (A B C P : EuclideanSpace ℝ (Fin 2)) :
  ∠ B A C = π / 2 →  -- ∠B = 90°
  dist P A = 13 →
  dist P B = 5 →
  ∠ A P B = ∠ B P C →
  ∠ B P C = ∠ C P A →
  dist P C = 115 / 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_theorem_l190_19056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_eccentricity_4_l190_19062

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The equations of the asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = h.b / h.a * x ∨ y = -h.b / h.a * x}

/-- Theorem: For a hyperbola with eccentricity 4, its asymptotes are y = ±√15 x -/
theorem hyperbola_asymptotes_eccentricity_4 (h : Hyperbola) 
    (h_ecc : eccentricity h = 4) :
    asymptotes h = {(x, y) | y = Real.sqrt 15 * x ∨ y = -Real.sqrt 15 * x} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_eccentricity_4_l190_19062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_l190_19033

noncomputable def proj (a : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (u.1 * a.1 + u.2 * a.2) / (a.1^2 + a.2^2)
  (scalar * a.1, scalar * a.2)

theorem projection_line :
  ∀ (u : ℝ × ℝ), 
    proj (3, 4) u = (-3, -4) → 
    u.2 = -3/4 * u.1 - 25/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_l190_19033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_theorem_l190_19010

/-- Calculates the total distance of a round trip given uphill and downhill speeds and total time -/
noncomputable def roundTripDistance (uphillSpeed downhillSpeed totalTime : ℝ) : ℝ :=
  let oneWayDistance := (uphillSpeed * downhillSpeed * totalTime) / (2 * (uphillSpeed + downhillSpeed))
  2 * oneWayDistance

/-- Theorem stating that for the given conditions, the total distance is approximately 1066.67 km -/
theorem round_trip_distance_theorem :
  let uphillSpeed : ℝ := 50
  let downhillSpeed : ℝ := 100
  let totalTime : ℝ := 16
  abs (roundTripDistance uphillSpeed downhillSpeed totalTime - 1066.67) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval roundTripDistance 50 100 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_theorem_l190_19010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l190_19073

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  N.mulVec (![2, 3]) = ![1, -8] ∧
  N.mulVec (![4, -1]) = ![16, -5] ∧
  N = !![3.5, -2; -23/14, -11/7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l190_19073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sun_sirius_brightness_ratio_l190_19027

/-- The relationship between magnitude and brightness of celestial bodies -/
def magnitude_brightness_relation (m₁ m₂ E₁ E₂ : ℝ) : Prop :=
  m₂ - m₁ = (5/2) * Real.log (E₁ / E₂)

/-- The magnitude of the Sun -/
def sun_magnitude : ℝ := -26.7

/-- The magnitude of Sirius -/
def sirius_magnitude : ℝ := -1.45

/-- Theorem stating the ratio of brightness between the Sun and Sirius -/
theorem sun_sirius_brightness_ratio :
  ∃ (E₁ E₂ : ℝ), 
    magnitude_brightness_relation sun_magnitude sirius_magnitude E₁ E₂ →
    E₁ / E₂ = (10 : ℝ)^(10.1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sun_sirius_brightness_ratio_l190_19027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_l190_19086

/-- Represents the number of balls of each color in the urn -/
structure UrnState where
  red : ℕ
  blue : ℕ

/-- Represents one round of the operation -/
def performRound (s : UrnState) : UrnState :=
  { red := s.red, blue := s.blue }

/-- Calculates the probability of a specific sequence of four rounds -/
def sequenceProbability : ℚ :=
  (1 / 2) ^ 4 * (1 / 3) ^ 4

/-- Calculates the total number of possible sequences -/
def totalSequences : ℕ := 6

/-- The main theorem to prove -/
theorem urn_probability :
  let initial_state : UrnState := { red := 2, blue := 2 }
  let final_state : UrnState := { red := 3, blue := 3 }
  let n_rounds : ℕ := 4
  (∀ i : ℕ, i < n_rounds → 
    (performRound (performRound (performRound (performRound initial_state)))).red = 
    (performRound (performRound (performRound (performRound initial_state)))).blue) →
  (sequenceProbability * totalSequences : ℚ) = 1 / 216 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_l190_19086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_40_5_l190_19089

-- Define the trapezoid ABCD and points W and Z
structure Trapezoid :=
  (A B C D W Z : ℝ × ℝ)

-- Define the properties of the trapezoid
def IsIsoscelesTrapezoid (t : Trapezoid) : Prop :=
  -- BC is parallel to AD
  (t.B.2 - t.C.2) / (t.B.1 - t.C.1) = (t.A.2 - t.D.2) / (t.A.1 - t.D.1) ∧
  -- AB = CD
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = (t.C.1 - t.D.1)^2 + (t.C.2 - t.D.2)^2

-- Define the conditions given in the problem
def SatisfiesConditions (t : Trapezoid) : Prop :=
  -- W and Z lie on diagonal AC
  ∃ k₁ k₂ : ℝ, 0 < k₁ ∧ k₁ < k₂ ∧ k₂ < 1 ∧
    t.W = (k₁ * t.C.1 + (1 - k₁) * t.A.1, k₁ * t.C.2 + (1 - k₁) * t.A.2) ∧
    t.Z = (k₂ * t.C.1 + (1 - k₂) * t.A.1, k₂ * t.C.2 + (1 - k₂) * t.A.2) ∧
  -- ∠AWD = ∠BZC = 90°
  (t.A.1 - t.W.1) * (t.D.1 - t.W.1) + (t.A.2 - t.W.2) * (t.D.2 - t.W.2) = 0 ∧
  (t.B.1 - t.Z.1) * (t.C.1 - t.Z.1) + (t.B.2 - t.Z.2) * (t.C.2 - t.Z.2) = 0 ∧
  -- AW = 4, WZ = 2, ZC = 5
  (t.A.1 - t.W.1)^2 + (t.A.2 - t.W.2)^2 = 16 ∧
  (t.W.1 - t.Z.1)^2 + (t.W.2 - t.Z.2)^2 = 4 ∧
  (t.Z.1 - t.C.1)^2 + (t.Z.2 - t.C.2)^2 = 25

-- Define the area of the trapezoid
noncomputable def TrapezoidArea (t : Trapezoid) : ℝ :=
  let base1 := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let base2 := Real.sqrt ((t.A.1 - t.D.1)^2 + (t.A.2 - t.D.2)^2)
  let height := abs ((t.A.2 - t.B.2) * (t.C.1 - t.B.1) - (t.A.1 - t.B.1) * (t.C.2 - t.B.2)) /
                Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)
  (base1 + base2) * height / 2

-- The main theorem
theorem trapezoid_area_is_40_5 (t : Trapezoid) 
  (h1 : IsIsoscelesTrapezoid t) (h2 : SatisfiesConditions t) : 
  TrapezoidArea t = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_40_5_l190_19089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l190_19070

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then
    -x^2 + 2*(a+1)*x + 4
  else if x > 1 then
    x^a
  else
    0  -- Define a value for x ≤ 0 to make the function total

theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, 0 < x ∧ x < y → f a y < f a x) →
  -2 ≤ a ∧ a ≤ -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l190_19070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_pairs_count_l190_19028

theorem distinct_pairs_count : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      0 < p.1 ∧ p.1 < p.2 ∧ Real.sqrt 2025 = 2 * Real.sqrt (p.1 : ℝ) + Real.sqrt (p.2 : ℝ))
    (Finset.product (Finset.range 2026) (Finset.range 2026))).card ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_pairs_count_l190_19028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l190_19069

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the circle -/
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 = a^2

/-- Definition of the line intercepting the chord -/
def intercepting_line (x y : ℝ) : Prop := x - y - Real.sqrt 2 = 0

/-- Definition of the moving line -/
def moving_line (x y k : ℝ) : Prop := y = k * (x - 1)

/-- Theorem about the ellipse C and its properties -/
theorem ellipse_properties (a b : ℝ) (ha : a > b) (hb : b > 0) :
  (∃ (x y : ℝ), ellipse_C x y a b ∧ 
    (a^2 - b^2) / a^2 = 1/2 ∧
    (∃ (x₁ y₁ : ℝ), circle_eq x₁ y₁ a ∧ intercepting_line x₁ y₁ ∧
      ∃ (x₂ y₂ : ℝ), circle_eq x₂ y₂ a ∧ intercepting_line x₂ y₂ ∧
        (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4)) →
  (∀ (x y : ℝ), ellipse_C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  (∃ (m : ℝ), m = 5/4 ∧
    ∀ (k : ℝ) (hk : k ≠ 0) (x₁ y₁ x₂ y₂ : ℝ),
      ellipse_C x₁ y₁ a b ∧ moving_line x₁ y₁ k ∧
      ellipse_C x₂ y₂ a b ∧ moving_line x₂ y₂ k →
      (x₁ - m) * (x₂ - m) + y₁ * y₂ = -7/16) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l190_19069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_y_l190_19085

-- Define the function g(y)
noncomputable def g (y : ℝ) : ℝ := Real.sin (y / 5) + Real.sin (y / 7)

-- Define the maximum value of g(y)
def g_max : ℝ := 2

-- Theorem statement
theorem smallest_max_y : 
  ∀ y : ℝ, y > 0 → g y ≤ g_max → g (13230 * Real.pi / 180) = g_max ∧ 
  (∀ z : ℝ, 0 < z → z < 13230 * Real.pi / 180 → g z < g_max) := by
  sorry

#check smallest_max_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_y_l190_19085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_to_product_l190_19008

theorem cos_sum_to_product (a : ℝ) : Real.cos (3 * a) + Real.cos (5 * a) = 2 * Real.cos (4 * a) * Real.cos a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_to_product_l190_19008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_perimeter_l190_19096

/-- A polygon with 8 sides, each 12 centimeters long, has a perimeter of 96 centimeters. -/
theorem octagon_perimeter :
  ∀ (sides : List ℝ),
  (sides.length = 8) →
  (∀ side ∈ sides, side = 12) →
  (sides.sum = 96) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_perimeter_l190_19096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_relationship_l190_19058

/-- An inverse proportion function passing through (-2, 3) -/
noncomputable def inverse_prop (x : ℝ) : ℝ := -6 / x

/-- The y-coordinate when x = -3 -/
noncomputable def y₁ : ℝ := inverse_prop (-3)

/-- The y-coordinate when x = 1 -/
noncomputable def y₂ : ℝ := inverse_prop 1

/-- The y-coordinate when x = 2 -/
noncomputable def y₃ : ℝ := inverse_prop 2

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem inverse_prop_relationship : y₂ < y₃ ∧ y₃ < y₁ := by
  -- Unfold definitions
  unfold y₁ y₂ y₃ inverse_prop
  -- Simplify expressions
  simp
  -- Split into two inequalities
  apply And.intro
  -- Prove y₂ < y₃
  · norm_num
  -- Prove y₃ < y₁
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_relationship_l190_19058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_closer_to_A_l190_19077

/-- Represents the distance walked by person A in t hours -/
def distance_A (t : ℝ) : ℝ := 4 * t

/-- Represents the distance walked by person B in t hours -/
noncomputable def distance_B (t : ℝ) : ℝ := 3 * (2^t - 1)

/-- The total distance between A and B -/
def total_distance : ℝ := 100

/-- The difference in distance from the meeting point to each starting point -/
noncomputable def distance_difference (t : ℝ) : ℝ := distance_B t - distance_A t

theorem meeting_point_closer_to_A : 
  ∃ t : ℝ, t > 0 ∧ distance_A t + distance_B t = total_distance ∧ 
  distance_difference t = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_closer_to_A_l190_19077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l190_19074

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x + a else x^2

def has_minimum (f : ℝ → ℝ) : Prop :=
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x

theorem f_minimum_value (a : ℝ) (h : has_minimum (f a)) :
  (∃ (a_min : ℝ), (∀ (a' : ℝ), has_minimum (f a') → a_min ≤ a') →
    f a_min (f a_min (-2)) = 16) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l190_19074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_shared_hypotenuse_l190_19021

structure Triangle where
  side1 : Real
  side2 : Real
  hypotenuse : Real
  isRight : Prop

theorem right_triangles_shared_hypotenuse 
  (ABC : Triangle) (ABD : Triangle) 
  (h1 : ABC.isRight) (h2 : ABD.isRight)
  (h3 : ABC.hypotenuse = ABD.hypotenuse)
  (h4 : ABC.side1 = 3) (h5 : ABC.side2 = 4) (h6 : ABD.side1 = 5) :
  ABD.side2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangles_shared_hypotenuse_l190_19021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l190_19075

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 2*x - 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -1 ∨ (-1 < x ∧ x < 3) ∨ 3 < x} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l190_19075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l190_19032

/-- In a triangle ABC, if point E is between A and B, point F is between A and C, and BE = CF, then EF < BC. -/
theorem triangle_inequality (A B C E F : EuclideanSpace ℝ (Fin 2)) : 
  (∃ t₁ : ℝ, 0 < t₁ ∧ t₁ < 1 ∧ E = A + t₁ • (B - A)) →  -- E is between A and B
  (∃ t₂ : ℝ, 0 < t₂ ∧ t₂ < 1 ∧ F = A + t₂ • (C - A)) →  -- F is between A and C
  dist B E = dist C F →                                -- BE = CF
  dist E F < dist B C :=                               -- EF < BC
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l190_19032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l190_19034

def sequence_sum (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_general_term (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) :
  (∀ n, S n = sequence_sum a n) →
  S 1 = 1 →
  S 2 = 2 →
  (∀ n ≥ 2, S (n + 1) - 3 * S n + 2 * S (n - 1) = 0) →
  ∀ n : ℕ+, a n = if n = 1 then 1 else (2 : ℝ) ^ (n.val - 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l190_19034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l190_19046

/-- 
Given a triangle with sides a, b, c and opposite angles A, B, C,
prove that if A = 55°, B = 15°, and C = 110°, then c^2 - a^2 = a*b
-/
theorem triangle_side_relation (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_angles : ∃ (A B C : ℝ), A = 55 * Real.pi / 180 ∧ B = 15 * Real.pi / 180 ∧ C = 110 * Real.pi / 180 ∧
    A + B + C = Real.pi ∧
    Real.sin A / a = Real.sin B / b ∧ Real.sin B / b = Real.sin C / c) :
  c^2 - a^2 = a*b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l190_19046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_sum_of_segments_l190_19038

-- Define the points
def start_point : ℝ × ℝ := (-3, 7)
def end_point : ℝ × ℝ := (6, -5)
def intermediate_point1 : ℝ × ℝ := (0, 0)
def intermediate_point2 : ℝ × ℝ := (2, -3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem total_distance_equals_sum_of_segments :
  distance start_point intermediate_point1 +
  distance intermediate_point1 intermediate_point2 +
  distance intermediate_point2 end_point =
  Real.sqrt 58 + Real.sqrt 13 + Real.sqrt 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_equals_sum_of_segments_l190_19038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l190_19084

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the perimeter of a triangle in an ellipse -/
theorem ellipse_triangle_perimeter 
  (e : Ellipse) 
  (F₁ F₂ A B : Point) 
  (h1 : e.a > 5)
  (h2 : e.b = 5)
  (h3 : distance F₁ F₂ = 8)
  (h4 : ∀ (x y : ℝ), x^2 / e.a^2 + y^2 / 25 = 1 → 
       ∃ (p : Point), p.x = x ∧ p.y = y)
  (h5 : ∃ (t : ℝ), A.x = t * (B.x - F₁.x) + F₁.x ∧ 
                   A.y = t * (B.y - F₁.y) + F₁.y) :
  distance A B + distance A F₂ + distance B F₂ = 4 * Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l190_19084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l190_19003

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt (h.a^2 + h.b^2), 0)

/-- The left vertex of the hyperbola -/
def left_vertex (h : Hyperbola) : ℝ × ℝ :=
  (-h.a, 0)

/-- Represents a circle passing through the left vertex and intersecting the asymptote -/
structure IntersectingCircle (h : Hyperbola) where
  center : ℝ × ℝ
  passes_through_left_vertex : (center.1 + h.a)^2 + center.2^2 = (h.a + Real.sqrt (h.a^2 + h.b^2))^2
  intersects_asymptote : ∃ (p q : ℝ × ℝ), p ≠ q ∧
    (p.1 - center.1)^2 + (p.2 - center.2)^2 = (h.a + Real.sqrt (h.a^2 + h.b^2))^2 ∧
    (q.1 - center.1)^2 + (q.2 - center.2)^2 = (h.a + Real.sqrt (h.a^2 + h.b^2))^2 ∧
    p.2 / p.1 = h.b / h.a ∧ q.2 / q.1 = h.b / h.a

/-- The chord length is not less than the conjugate axis -/
def chord_length_condition (h : Hyperbola) (c : IntersectingCircle h) : Prop :=
  ∀ (p q : ℝ × ℝ), p ≠ q →
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = (h.a + Real.sqrt (h.a^2 + h.b^2))^2 →
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = (h.a + Real.sqrt (h.a^2 + h.b^2))^2 →
    p.2 / p.1 = h.b / h.a → q.2 / q.1 = h.b / h.a →
    (p.1 - q.1)^2 + (p.2 - q.2)^2 ≥ 4 * h.b^2

theorem eccentricity_range (h : Hyperbola) (c : IntersectingCircle h)
    (hc : chord_length_condition h c) :
    1 < eccentricity h ∧ eccentricity h ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l190_19003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l190_19035

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (80 - x)) + Real.sqrt (x * (5 - x))

-- State the theorem
theorem max_value_of_g :
  ∃ (x₁ : ℝ), x₁ ∈ Set.Icc 0 5 ∧
  (∀ x ∈ Set.Icc 0 5, g x ≤ g x₁) ∧
  x₁ = 80 / 17 ∧
  g x₁ = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l190_19035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_pq_is_10_l190_19094

/-- Given two real numbers p and q, their arithmetic mean is (p + q) / 2 -/
noncomputable def arithmetic_mean (p q : ℝ) : ℝ := (p + q) / 2

theorem arithmetic_mean_pq_is_10 (p q r : ℝ) 
  (h1 : arithmetic_mean q r = 20) 
  (h2 : r - p = 20) : 
  arithmetic_mean p q = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_pq_is_10_l190_19094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_social_insurance_management_leads_to_fairness_l190_19007

/-- Represents the action of announcing units in arrears with social insurance payments -/
def announce_arrears : Prop := sorry

/-- Represents the action of praising companies that pay social insurance on time -/
def praise_timely_payers : Prop := sorry

/-- Represents the state of enterprises fulfilling their social responsibilities -/
def enterprises_fulfill_responsibilities : Prop := sorry

/-- Represents the state of a perfected social security system -/
def perfected_social_security : Prop := sorry

/-- Represents the state of achieved social fairness -/
def social_fairness_achieved : Prop := sorry

/-- The main theorem representing the chain of implications from the center's actions to social fairness -/
theorem social_insurance_management_leads_to_fairness :
  (announce_arrears ∧ praise_timely_payers) →
  enterprises_fulfill_responsibilities →
  perfected_social_security →
  social_fairness_achieved :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_social_insurance_management_leads_to_fairness_l190_19007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l190_19067

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.cos x) ^ 2 + 1

-- Define the theorem
theorem function_properties :
  ∀ (α β : ℝ),
    α ∈ Set.Ioo 0 (Real.pi / 2) →
    β ∈ Set.Ioo 0 (Real.pi / 2) →
    f (α / 2 + Real.pi / 12) = 10 / 13 →
    f (β / 2 + Real.pi / 3) = 6 / 5 →
    (∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2), f x ∈ Set.Icc (-Real.sqrt 3) 2) ∧
    Real.sin (α - β) = -33 / 65 := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma f_simplified (x : ℝ) : f x = 2 * Real.sin (2 * x - Real.pi / 6) := by
  sorry

lemma range_of_f (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2)) :
  f x ∈ Set.Icc (-Real.sqrt 3) 2 := by
  sorry

lemma sin_alpha_value (α : ℝ) (hα : α ∈ Set.Ioo 0 (Real.pi / 2))
  (hf : f (α / 2 + Real.pi / 12) = 10 / 13) : Real.sin α = 5 / 13 := by
  sorry

lemma cos_beta_value (β : ℝ) (hβ : β ∈ Set.Ioo 0 (Real.pi / 2))
  (hf : f (β / 2 + Real.pi / 3) = 6 / 5) : Real.cos β = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l190_19067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_trip_distance_l190_19082

/-- Represents the distance of Philip's trips in miles -/
structure PhilipTrips where
  school_distance : ℚ
  weekly_mileage : ℚ
  school_days : ℕ
  school_trips_per_day : ℕ

/-- Calculates the one-way distance to the market -/
def market_distance (trips : PhilipTrips) : ℚ :=
  (trips.weekly_mileage - 2 * trips.school_distance * trips.school_days * trips.school_trips_per_day) / 2

/-- Theorem stating that the one-way trip to the market is 2 miles -/
theorem market_trip_distance (trips : PhilipTrips) 
  (h1 : trips.school_distance = 5/2)
  (h2 : trips.weekly_mileage = 44)
  (h3 : trips.school_days = 4)
  (h4 : trips.school_trips_per_day = 2) : 
  market_distance trips = 2 := by
  sorry

#eval market_distance { school_distance := 5/2, weekly_mileage := 44, school_days := 4, school_trips_per_day := 2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_trip_distance_l190_19082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l190_19059

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := 3 * x + 2
noncomputable def line2 (x : ℝ) : ℝ := -3 * x + 2
noncomputable def line3 : ℝ := -2

-- Define the intersection points
noncomputable def point1 : ℝ × ℝ := (0, 2)
noncomputable def point2 : ℝ × ℝ := (-4/3, -2)
noncomputable def point3 : ℝ × ℝ := (4/3, -2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem isosceles_triangle : 
  distance point1 point2 = distance point1 point3 ∧ 
  distance point1 point2 ≠ distance point2 point3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l190_19059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sequence_zero_l190_19022

/-- A sequence defined by a polynomial function with integer coefficients -/
def PolynomialSequence (f : ℤ → ℤ) : ℕ → ℤ
  | 0 => 0
  | n + 1 => f (PolynomialSequence f n)

/-- The main theorem -/
theorem polynomial_sequence_zero (f : ℤ → ℤ) (hf : ∃ p : Polynomial ℤ, ∀ x, f x = p.eval x) :
  ∀ m : ℕ, m > 0 → PolynomialSequence f m = 0 → PolynomialSequence f 1 = 0 ∨ PolynomialSequence f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sequence_zero_l190_19022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_irregular_acute_triangle_l190_19000

/-- Definition of an acute triangle -/
noncomputable def is_acute_triangle (α β γ : ℝ) : Prop :=
  0 < α ∧ α < 90 ∧ 0 < β ∧ β < 90 ∧ 0 < γ ∧ γ < 90 ∧ α + β + γ = 180

/-- Definition of angle deviations -/
noncomputable def angle_deviations (α β γ : ℝ) : ℝ :=
  min (min (abs (β - α)) (abs (γ - β))) (abs (90 - γ))

/-- Theorem: The most irregular acute triangle has angles 45°, 60°, and 75° -/
theorem most_irregular_acute_triangle :
  ∃ (α β γ : ℝ), is_acute_triangle α β γ ∧
  (∀ (a b c : ℝ), is_acute_triangle a b c → angle_deviations a b c ≤ angle_deviations α β γ) ∧
  α = 45 ∧ β = 60 ∧ γ = 75 := by
  sorry

#check most_irregular_acute_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_irregular_acute_triangle_l190_19000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mars_earth_year_ratio_approx_l190_19050

/-- The length of an Earth day in hours -/
noncomputable def earth_day_hours : ℝ := 24

/-- The length of a Mars day in hours -/
noncomputable def mars_day_hours : ℝ := earth_day_hours + 2/3

/-- The number of days in a Martian year -/
def mars_year_days : ℕ := 668

/-- The number of days in an Earth year -/
noncomputable def earth_year_days : ℝ := 365.25

/-- The ratio of a Martian year to an Earth year -/
noncomputable def mars_earth_year_ratio : ℝ :=
  (mars_year_days : ℝ) * mars_day_hours / (earth_year_days * earth_day_hours)

theorem mars_earth_year_ratio_approx :
  |mars_earth_year_ratio - 1.88| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mars_earth_year_ratio_approx_l190_19050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_l190_19042

theorem quadratic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_real_roots : b^2 - 4*a*c ≥ 0) :
  max a (max b c) ≥ (4/9) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_l190_19042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_max_value_h_l190_19051

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 1
def g (a : ℝ) (x : ℝ) : ℝ := a * |x - 1|
def h (a : ℝ) (x : ℝ) : ℝ := |f x| + g a x

-- Part I: Range of a
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x ≥ g a x) → a ≤ -2 :=
sorry

-- Part II: Maximum value of h(x) on [-2, 2]
noncomputable def max_h (a : ℝ) : ℝ :=
  if a ≥ 0 then 3*a + 3
  else if a ≥ -3 then a + 3
  else 0

theorem max_value_h (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, h a x ≤ max_h a) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, h a x = max_h a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_max_value_h_l190_19051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_l190_19049

/-- Line equation: mx - y + 2 = 0 -/
def line (m : ℝ) (x y : ℝ) : Prop := m * x - y + 2 = 0

/-- Circle equation: x^2 + y^2 - 2x - 4y + 19/4 = 0 -/
def circle' (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 19/4 = 0

/-- Proposition P: The line intersects the circle at two points -/
def P (m : ℝ) : Prop := ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ line m x₁ y₁ ∧ line m x₂ y₂ ∧ circle' x₁ y₁ ∧ circle' x₂ y₂

/-- Proposition Q: ∃x₀ ∈ [-π/6, π/4], 2sin(2x₀ + π/6) + 2cos(2x₀) ≤ m -/
def Q (m : ℝ) : Prop := ∃ x₀ : ℝ, -Real.pi/6 ≤ x₀ ∧ x₀ ≤ Real.pi/4 ∧ 
  2 * Real.sin (2*x₀ + Real.pi/6) + 2 * Real.cos (2*x₀) ≤ m

theorem intersection_theorem (m : ℝ) :
  (P m ∧ Q m ↔ 0 ≤ m ∧ m < Real.sqrt 3 / 3) ∧
  ((P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ (-Real.sqrt 3 / 3 < m ∧ m < 0) ∨ (Real.sqrt 3 / 3 ≤ m)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_l190_19049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l190_19037

/-- Represents a segment of a triathlon --/
structure TriathlonSegment where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a segment --/
noncomputable def segmentTime (segment : TriathlonSegment) : ℝ :=
  segment.distance / segment.speed

/-- Calculates the average speed for the entire triathlon --/
noncomputable def averageSpeed (segments : List TriathlonSegment) : ℝ :=
  let totalDistance := segments.foldl (fun acc s => acc + s.distance) 0
  let totalTime := segments.foldl (fun acc s => acc + segmentTime s) 0
  totalDistance / totalTime

/-- Theorem stating that the average speed of the given triathlon is approximately 7.2 km/h --/
theorem triathlon_average_speed :
  let swimming := TriathlonSegment.mk 1 2
  let biking := TriathlonSegment.mk 2 25
  let running := TriathlonSegment.mk 3 12
  let segments := [swimming, biking, running]
  abs (averageSpeed segments - 7.2) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l190_19037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l190_19078

def a (n : ℕ) : ℚ := 1 / ((n + 1 : ℕ) ^ 2 : ℚ)

def b : ℕ → ℚ
  | 0 => 1
  | n + 1 => b n * (1 - a (n + 1))

theorem b_formula (n : ℕ) : b n = (n + 2 : ℚ) / (2 * n + 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l190_19078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l190_19088

/-- The angle of inclination of a line with slope m -/
noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.arctan m

/-- The slope of a line given its angle of inclination θ -/
noncomputable def slope_from_angle (θ : ℝ) : ℝ := Real.tan θ

/-- The equation of a line passing through (x₀, y₀) with slope m -/
def line_equation (x₀ y₀ m : ℝ) (x : ℝ) : ℝ := m * (x - x₀) + y₀

theorem line_equation_proof (l : ℝ → ℝ) :
  (∃ m : ℝ, ∀ x, l x = line_equation 3 4 m x) →
  (angle_of_inclination (slope_from_angle (2 * angle_of_inclination 1)) = π/2) →
  (∀ x, l x = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l190_19088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_black_cells_l190_19091

/-- Represents a grid of unit squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a coloring of a grid -/
def Coloring (g : Grid) := Fin g.rows → Fin g.cols → Bool

/-- Counts the number of black cells in a coloring -/
def blackCellCount (g : Grid) (c : Coloring g) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin g.rows)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin g.cols)) fun j =>
      if c i j then 1 else 0)

/-- Checks if two cells are adjacent -/
def isAdjacent (i1 j1 i2 j2 : ℕ) : Bool :=
  (i1 = i2 ∧ Int.natAbs (j1 - j2) = 1) ∨ (j1 = j2 ∧ Int.natAbs (i1 - i2) = 1)

/-- Counts the number of adjacent black cells for a given cell -/
def adjacentBlackCount (g : Grid) (c : Coloring g) (i j : Fin g.rows) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin g.rows)) fun i' =>
    Finset.sum (Finset.univ : Finset (Fin g.cols)) fun j' =>
      if isAdjacent i.val j.val i'.val j'.val ∧ c i' j' then 1 else 0)

/-- Checks if a coloring is valid (each cell has at most two adjacent black cells) -/
def isValidColoring (g : Grid) (c : Coloring g) : Prop :=
  ∀ i j, adjacentBlackCount g c i j ≤ 2

/-- The main theorem -/
theorem max_black_cells (g : Grid) (h1 : g.rows = 5) (h2 : g.cols = 100) :
  (∃ c : Coloring g, isValidColoring g c ∧ blackCellCount g c = 302) ∧
  (∀ c : Coloring g, isValidColoring g c → blackCellCount g c ≤ 302) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_black_cells_l190_19091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_of_sqrt_plus_one_l190_19053

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x + 1

-- Define the inverse function f_inv
def f_inv (x : ℝ) : ℝ := (x - 1)^2

-- State the theorem
theorem inverse_function_of_sqrt_plus_one :
  ∀ x ≥ 0, ∀ y ≥ 1,
  (f x = y ↔ f_inv y = x) ∧
  (f (f_inv y) = y) ∧
  (f_inv (f x) = x) :=
by
  sorry

#check inverse_function_of_sqrt_plus_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_of_sqrt_plus_one_l190_19053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_point_of_symmetry_l190_19052

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the sum function h
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Theorem statement
theorem not_point_of_symmetry :
  ¬(∀ (t : ℝ), h (π/8 + t) = h (π/8 - t)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_point_of_symmetry_l190_19052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l190_19044

/-- The time (in days) it takes person a to complete the task independently -/
def time_a : ℝ := sorry

/-- The time (in days) it takes person b to complete the task independently -/
def time_b : ℝ := sorry

/-- The time (in days) person a works on the task before leaving -/
def time_a_works : ℝ := sorry

/-- The portion of the task completed when a leaves -/
def portion_completed : ℝ := sorry

theorem b_completion_time :
  time_b = time_a + 12 →
  time_a_works = time_b - 12 →
  portion_completed = 0.6 →
  time_b = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l190_19044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_intersect_not_perpendicular_l190_19097

/-- Two planes α and β with given normal vectors intersect but are not perpendicular -/
theorem planes_intersect_not_perpendicular :
  let n₁ : Fin 3 → ℝ := ![2, -3, 5]
  let n₂ : Fin 3 → ℝ := ![-3, 1, -4]
  let dot_product := (Finset.sum Finset.univ (λ i => n₁ i * n₂ i))
  (dot_product ≠ 0) ∧ 
  (∀ (k : ℝ), n₁ ≠ λ i => k * n₂ i) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_intersect_not_perpendicular_l190_19097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l190_19063

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 else x^2 + 2*x

-- Define the set S
def S : Set ℝ := {x | f (f x) ≤ 3}

-- State the theorem
theorem solution_set : S = Set.Iic (Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l190_19063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l190_19001

/-- A primitive cube root of unity -/
noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

/-- The polynomial we're considering -/
def f (C D : ℂ) (x : ℂ) : ℂ := x^102 + C*x + D

/-- The statement of the problem -/
theorem polynomial_divisibility (C D : ℂ) : 
  (∀ x, x^3 = 1 → f C D x = 0) ↔ (C = 0 ∧ D = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l190_19001
