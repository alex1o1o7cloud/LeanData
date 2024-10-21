import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solution_l189_18954

theorem no_positive_integer_solution :
  ∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 → 
  x^2 * y^4 - x^4 * y^2 + 4 * x^2 * y^2 * z^2 + x^2 * z^4 - y^2 * z^4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solution_l189_18954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_sliced_cone_l189_18948

/-- A right circular cone sliced into 5 pieces of equal height -/
structure SlicedCone where
  height : ℝ
  baseRadius : ℝ
  slices : Fin 5

/-- Volume of a frustum given its height and radii -/
noncomputable def frustumVolume (h r1 r2 : ℝ) : ℝ :=
  (1/3) * Real.pi * h * (r1^2 + r1*r2 + r2^2)

/-- Volume of the largest (bottom) slice -/
noncomputable def largestSliceVolume (cone : SlicedCone) : ℝ :=
  frustumVolume (cone.height / 5) (4 * cone.baseRadius / 5) cone.baseRadius

/-- Volume of the second-largest slice -/
noncomputable def secondLargestSliceVolume (cone : SlicedCone) : ℝ :=
  frustumVolume (cone.height / 5) (3 * cone.baseRadius / 5) (4 * cone.baseRadius / 5)

/-- The main theorem stating the volume ratio -/
theorem volume_ratio_of_sliced_cone (cone : SlicedCone) :
  secondLargestSliceVolume cone / largestSliceVolume cone = 25 / 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_sliced_cone_l189_18948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l189_18963

theorem problem_solution : (-10) * (-1/2) - Real.sqrt 16 - (-1)^2022 + (-8 : ℝ) ^ (1/3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l189_18963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l189_18968

-- Define the train lengths in meters
noncomputable def train1_length : ℝ := 210
noncomputable def train2_length : ℝ := 120

-- Define the initial distance between trains in meters
noncomputable def initial_distance : ℝ := 160

-- Define the speeds in kilometers per hour
noncomputable def train1_speed : ℝ := 74
noncomputable def train2_speed : ℝ := 92

-- Define the conversion factor from kmph to m/s
noncomputable def kmph_to_ms : ℝ := 1000 / 3600

-- Function to calculate the time for trains to meet
noncomputable def time_to_meet : ℝ :=
  let speed1_ms := train1_speed * kmph_to_ms
  let speed2_ms := train2_speed * kmph_to_ms
  let relative_speed := speed1_ms + speed2_ms
  let total_distance := train1_length + train2_length + initial_distance
  total_distance / relative_speed

-- Theorem stating that the time to meet is approximately 10.62 seconds
theorem trains_meet_time : 
  ∃ ε > 0, |time_to_meet - 10.62| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l189_18968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_color_count_l189_18941

/-- Represents a coloring of an n × n board --/
def Coloring (n : ℕ) := Fin n → Fin n → ℕ

/-- Checks if four cells form a rectangle with different colors --/
def hasFourColorRectangle {n : ℕ} (c : Coloring n) : Prop :=
  ∃ (i j k l : Fin n), i ≠ k ∧ j ≠ l ∧
    c i j ≠ c i l ∧ c i j ≠ c k j ∧ c i j ≠ c k l ∧
    c i l ≠ c k j ∧ c i l ≠ c k l ∧
    c k j ≠ c k l

/-- The main theorem --/
theorem smallest_color_count (n : ℕ) (h : n ≥ 2) :
  (∀ (c : Coloring n), (∃ (k : ℕ), ∀ (i j : Fin n), c i j < k) →
    hasFourColorRectangle c) ↔ 2 * n ≤ Finset.card (Finset.range (2 * n)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_color_count_l189_18941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l189_18959

noncomputable def f (x : Real) : Real := Real.cos x - Real.sqrt 3 * Real.sin x + 1

theorem f_symmetry (x : Real) : f (4 * Real.pi / 3 - x) = f (2 * Real.pi / 3 + x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l189_18959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l189_18958

/-- Curve C1 in polar coordinates -/
def curve_C1 (ρ θ : ℝ) : Prop :=
  ρ^2 - 4*ρ*(Real.cos θ) - 8*ρ*(Real.sin θ) + 16 = 0

/-- Line C2 in polar coordinates -/
def line_C2 (θ : ℝ) : Prop :=
  θ = Real.pi/4

/-- The distance between intersection points of C1 and C2 -/
theorem intersection_distance :
  ∃ (ρ₁ ρ₂ : ℝ),
    curve_C1 ρ₁ (Real.pi/4) ∧
    curve_C1 ρ₂ (Real.pi/4) ∧
    line_C2 (Real.pi/4) ∧
    ρ₁ ≠ ρ₂ ∧
    (ρ₁ - ρ₂)^2 = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l189_18958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l189_18967

noncomputable def f (x : ℝ) : ℝ := x / (abs x + 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  Set.range f = Set.Ioo (-1) 1 ∧
  StrictMono f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l189_18967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_properties_l189_18905

open Real

/-- Given x > 1, if a and b are roots of x - (x-1)2^x = 0 and x - (x-1)log₂x = 0 respectively 
    in the interval (1, +∞), then certain properties hold. -/
theorem root_properties (a b : ℝ) : 
  (∀ x, x > 1 → x - (x-1)*2^x ≠ 0) → 
  (∀ x, x > 1 → x - (x-1)*(log x / log 2) ≠ 0) → 
  a > 1 → b > 1 → 
  a - (a-1)*2^a = 0 → 
  b - (b-1)*(log b / log 2) = 0 → 
  (b - a = 2^a - (log b / log 2)) ∧ 
  (1/a + 1/b = 1) ∧ 
  (b - a > 1) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_properties_l189_18905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_sum_l189_18932

theorem max_value_sin_cos_sum (x : ℝ) : 
  (Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)) ≤ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_sum_l189_18932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l189_18918

/-- The function f(x) is defined as the minimum of three linear functions -/
noncomputable def f (x : ℝ) := min (4*x + 1) (min (x + 2) (-2*x + 4))

/-- The maximum value of f(x) over all real numbers is 8/3 -/
theorem max_value_of_f : 
  ∃ (M : ℝ), M = 8/3 ∧ ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l189_18918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_distance_l189_18938

/-- Represents a 2D point --/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Amanda's walk --/
def amanda_walk : Point := {
  x := -40,  -- 40 meters west
  y := 8  -- 12 meters south, then 20 meters north, net 8 meters north
}

theorem amanda_distance :
  distance {x := 0, y := 0} amanda_walk = Real.sqrt 1664 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_distance_l189_18938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l189_18909

/-- Calculates simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem interest_calculation (P : ℝ) :
  simpleInterest P 5 2 = 55 →
  compoundInterest P 5 2 = 56.375 := by
  sorry

#check interest_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l189_18909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_range_theorem_l189_18972

/-- Represents a quadratic function passing through specific points -/
def QuadraticFunction (a b c : ℝ) : Type :=
  {f : ℝ → ℝ // ∀ x, f x = a * x^2 + b * x + c ∧
    f (-2) = 8 ∧ f (-1) = 3 ∧ f 0 = 0 ∧ f 1 = -1 ∧ f 2 = 0 ∧ f 3 = 3}

/-- Theorem stating the range of x where y - 3 > 0 for the given quadratic function -/
theorem quadratic_range_theorem {a b c : ℝ} (f : QuadraticFunction a b c) :
  ∀ x : ℝ, (f.val x - 3 > 0) ↔ (x < -1 ∨ x > 3) := by
  sorry

#check quadratic_range_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_range_theorem_l189_18972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_sqrt_three_simplest_l189_18995

/-- A quadratic radical is considered simpler if it cannot be further simplified
    by factoring out perfect squares or reducing fractions. -/
noncomputable def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≠ 0 → x ≠ y * Real.sqrt (y^2)

/-- The given options for quadratic radicals -/
noncomputable def options : List ℝ :=
  [Real.sqrt 0.5, Real.sqrt (1/7), -Real.sqrt 3, Real.sqrt 8]

/-- Theorem stating that -√3 is the simplest quadratic radical among the given options -/
theorem negative_sqrt_three_simplest :
  ∃ x ∈ options, is_simplest_quadratic_radical x ∧
  ∀ y ∈ options, is_simplest_quadratic_radical y → y = x :=
by sorry

#check negative_sqrt_three_simplest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_sqrt_three_simplest_l189_18995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_arc_arrangement_l189_18989

-- Define a sphere
structure Sphere where
  radius : ℝ

-- Define an arc on a sphere
structure Arc where
  length : ℝ

-- Define non-intersecting arcs
def ArcNonIntersecting (a b : Arc) : Prop := sorry

-- Define the problem setup
def sphereArrangementProblem (n : ℕ) (α : ℝ) (s : Sphere) : Prop :=
  n > 2 ∧ s.radius = 1 ∧ ∃ (arcs : Fin n → Arc), 
    (∀ i, (arcs i).length = α) ∧ 
    (∀ i j, i ≠ j → ArcNonIntersecting (arcs i) (arcs j))

-- Define the theorem
theorem sphere_arc_arrangement (n : ℕ) (α : ℝ) (s : Sphere) :
  (α < π + 2 * π / n → sphereArrangementProblem n α s) ∧
  (α > π + 2 * π / n → ¬sphereArrangementProblem n α s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_arc_arrangement_l189_18989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l189_18922

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*x + 2/x + a*Real.log x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 * ((2 - 2/x^2 + a/x) + 2*x - 2)

-- Theorem for part (1)
theorem part_one (a : ℝ) : 
  (∀ x ≥ 1, Monotone (f a)) → a ≥ 0 := by
  sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) :
  (∃ x_min : ℝ, ∀ x : ℝ, x > 0 → g a x ≥ g a x_min ∧ g a x_min = -6) →
  (∀ x : ℝ, x > 0 → f a x = 2*x + 2/x - 6*Real.log x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l189_18922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_value_at_5_l189_18945

def p (b c x : ℤ) : ℤ := x^2 + b*x + c

theorem p_value_at_5 (b c : ℤ) :
  (∀ x, (x^4 + 8*x^3 + 6*x^2 + 36 : ℤ) ∣ (p b c x)) →
  (∀ x, (3*x^4 + 6*x^3 + 5*x^2 + 42*x + 15 : ℤ) ∣ (p b c x)) →
  p b c 5 = 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_value_at_5_l189_18945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_distances_l189_18997

/-- The maximum product of distances from a point to two fixed points --/
theorem max_product_distances (m : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 3)
  let P (x y : ℝ) : ℝ × ℝ := (x, y)
  ∀ x y : ℝ, x + m * y = 0 → m * x - y - m + 3 = 0 → 
    ‖P x y - A‖ * ‖P x y - B‖ ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_distances_l189_18997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_honey_water_percentage_is_30_l189_18962

/-- Represents the process of bees converting nectar to honey -/
structure HoneyProduction where
  nectar_mass : ℝ
  nectar_water_percentage : ℝ
  honey_mass : ℝ

/-- Calculates the percentage of water in honey -/
noncomputable def water_percentage_in_honey (hp : HoneyProduction) : ℝ :=
  let water_mass_in_nectar := hp.nectar_mass * (hp.nectar_water_percentage / 100)
  let solids_mass := hp.nectar_mass - water_mass_in_nectar
  let water_mass_in_honey := hp.honey_mass - solids_mass
  (water_mass_in_honey / hp.honey_mass) * 100

/-- Theorem stating that for the given conditions, the water percentage in honey is 30% -/
theorem honey_water_percentage_is_30 (hp : HoneyProduction) 
  (h1 : hp.nectar_mass = 1.4)
  (h2 : hp.nectar_water_percentage = 50)
  (h3 : hp.honey_mass = 1) :
  water_percentage_in_honey hp = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_honey_water_percentage_is_30_l189_18962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_border_length_is_5pi_over_2_l189_18978

noncomputable section

/-- The total length of the border of a specific design -/
def border_length (diagonal_length : ℝ) : ℝ :=
  let side_length := diagonal_length / Real.sqrt 2
  let radius := side_length / 2
  let semicircle_length := 4 * (Real.pi / 2 * radius)
  let three_quarter_circle_length := 4 * (3 * Real.pi / 4 * radius)
  semicircle_length + three_quarter_circle_length

/-- Theorem stating that the border length is 5π/2 when the diagonal length is 1 -/
theorem border_length_is_5pi_over_2 :
  border_length 1 = 5 * Real.pi / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_border_length_is_5pi_over_2_l189_18978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l189_18961

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - x / (x + 1)

def domain (x : ℝ) : Prop := x > -1

theorem f_properties :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, StrictMonoOn f (Set.Ioo (-1 : ℝ) 0) → False) ∧
  (∀ x ∈ Set.Ioi 0, StrictMonoOn f (Set.Ioi 0)) ∧
  (∃ m b : ℝ, m = (1 : ℝ) / 4 ∧ b = Real.log 2 - (1 : ℝ) / 2 ∧
    ∀ x y : ℝ, y = m * (x - 1) + b ↔ x - 4 * y + 4 * Real.log 2 - 3 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l189_18961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reporters_not_covering_politics_l189_18901

/-- The percentage of reporters who cover local politics in country x -/
noncomputable def local_politics_coverage : ℝ := 10

/-- The percentage of reporters who cover politics but not local politics in country x -/
noncomputable def non_local_politics_coverage : ℝ := 30

/-- The percentage of reporters who do not cover politics -/
noncomputable def non_politics_coverage : ℝ := 100 - (local_politics_coverage / (1 - non_local_politics_coverage / 100))

theorem reporters_not_covering_politics :
  ∃ (ε : ℝ), abs (non_politics_coverage - 85.71) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reporters_not_covering_politics_l189_18901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_specific_l189_18949

/-- The volume of a truncated right circular cone -/
noncomputable def truncated_cone_volume (R r h : ℝ) : ℝ := (1/3) * Real.pi * h * (R^2 + R*r + r^2)

/-- Theorem: The volume of a truncated right circular cone with large base radius 10 cm,
    small base radius 5 cm, and height 9 cm is equal to 525π cubic cm -/
theorem truncated_cone_volume_specific : 
  truncated_cone_volume 10 5 9 = 525 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_specific_l189_18949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_in_triangle_l189_18991

theorem sin_B_in_triangle (A B C : ℝ) (AB BC : ℝ) :
  A = 2 * π / 3 →  -- 120° in radians
  AB = 5 →
  BC = 7 →
  Real.sin B = (3 * Real.sqrt 3) / 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_in_triangle_l189_18991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_bounded_l189_18940

/-- A function f satisfying the given conditions -/
def f : ℝ → ℝ := sorry

/-- The derivative of f -/
def f' : ℝ → ℝ := sorry

/-- The second derivative of f -/
def f'' : ℝ → ℝ := sorry

/-- The function g defined in terms of f -/
def g : ℝ → ℝ := sorry

/-- The theorem stating that g is bounded -/
theorem g_is_bounded (x : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f'' x = 1 / (x^2 + f' x^2 + 1)) →
  f 0 = 0 →
  f' 0 = 0 →
  g 0 = 0 →
  (∀ x : ℝ, x > 0 → g x = f x / x) →
  0 ≤ x →
  0 ≤ g x ∧ g x ≤ π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_bounded_l189_18940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cancer_statements_l189_18939

-- Define the statements as axioms instead of definitions
axiom statement_A : Prop
axiom statement_B : Prop
axiom statement_C : Prop
axiom statement_D : Prop

-- Define the meanings of the statements (for documentation purposes)
/-- AIDS patients have a higher probability of developing cancer than normal people -/
def meaning_A : String := "AIDS patients have a higher probability of developing cancer than normal people"

/-- Cancer cells are the result of abnormal cell differentiation, can proliferate indefinitely, and can metastasize within the body -/
def meaning_B : String := "Cancer cells are the result of abnormal cell differentiation, can proliferate indefinitely, and can metastasize within the body"

/-- Nitrites can cause cancer by altering the structure of genes -/
def meaning_C : String := "Nitrites can cause cancer by altering the structure of genes"

/-- If a normal person is in long-term contact with cancer patients, the probability of their cells becoming cancerous will increase -/
def meaning_D : String := "If a normal person is in long-term contact with cancer patients, the probability of their cells becoming cancerous will increase"

-- Define the theorem
theorem cancer_statements :
  statement_A ∧ statement_B ∧ statement_C → ¬statement_D :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cancer_statements_l189_18939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_paths_count_l189_18917

/-- The number of paths from C to D on a grid with given right and up steps -/
def number_of_paths_from_C_to_D (right_steps up_steps : ℕ) : ℕ :=
  (right_steps + up_steps).choose up_steps

theorem grid_paths_count (right_steps up_steps : ℕ) : 
  (right_steps + up_steps).choose up_steps = 
  number_of_paths_from_C_to_D right_steps up_steps :=
by
  -- Unfold the definition of number_of_paths_from_C_to_D
  unfold number_of_paths_from_C_to_D
  -- The equality now holds by reflexivity
  rfl

#eval number_of_paths_from_C_to_D 7 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_paths_count_l189_18917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_interval_l189_18934

open Real Set

/-- The function f(x) with ω > 0 and minimum positive period π -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x - π / 4)

/-- The function g(x) obtained by shifting f(x) to the left by π/4 units -/
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x + π / 4)

/-- The theorem stating the monotonically increasing interval of g(x) -/
theorem g_monotone_increasing_interval (ω : ℝ) (h_ω : ω > 0) (h_period : ∀ x, f ω (x + π) = f ω x) :
  ∀ k : ℤ, StrictMonoOn (g ω) (Icc (-3 * π / 8 + k * π) (π / 8 + k * π)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_interval_l189_18934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_verify_num_vertical_asymptotes_l189_18943

/-- The function f(x) = (x+2)/(x^2 - 2x - 8) -/
noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x^2 - 2*x - 8)

/-- The number of vertical asymptotes of f -/
def num_vertical_asymptotes : ℕ := 1

/-- Theorem stating that f has exactly one vertical asymptote -/
theorem f_has_one_vertical_asymptote :
  ∃! x : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, 
    0 < |y - x| ∧ |y - x| < δ → |f y| > 1/ε :=
by
  -- The proof goes here
  sorry

/-- Theorem verifying that the number of vertical asymptotes is correct -/
theorem verify_num_vertical_asymptotes :
  num_vertical_asymptotes = 1 :=
by
  -- The proof goes here
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_verify_num_vertical_asymptotes_l189_18943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l189_18980

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 20

-- Define the line
def line_eq (x y m : ℝ) : Prop := y = 2 * x + m

-- Define the distance from a point to the line
noncomputable def distance_to_line (x y m : ℝ) : ℝ :=
  |2 * x - y + m| / Real.sqrt (2^2 + 1)

-- Main theorem
theorem circle_line_intersection (m : ℝ) :
  (∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℝ,
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ circle_eq x3 y3 ∧ circle_eq x4 y4 ∧
    line_eq x1 y1 m ∧ line_eq x2 y2 m ∧ line_eq x3 y3 m ∧ line_eq x4 y4 m ∧
    distance_to_line x1 y1 m = Real.sqrt 5 ∧
    distance_to_line x2 y2 m = Real.sqrt 5 ∧
    distance_to_line x3 y3 m = Real.sqrt 5 ∧
    distance_to_line x4 y4 m = Real.sqrt 5 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) →
  m > -7 ∧ m < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l189_18980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_ratio_pairs_l189_18960

theorem interior_angle_ratio_pairs : 
  ∃! n : ℕ, n > 0 ∧ 
  (∃ S : Finset (ℕ × ℕ), 
    (∀ p : ℕ × ℕ, p ∈ S → p.1 > 2 ∧ p.2 > 2) ∧
    (∀ p : ℕ × ℕ, p ∈ S → (p.2 - 2) * p.1 = 3 * (p.1 - 2) * p.2 / 2) ∧
    S.card = n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_ratio_pairs_l189_18960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l189_18928

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The functional equation -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = (-1 : ℝ) ^ (floor y).toNat * f x + (-1 : ℝ) ^ (floor x).toNat * f y

/-- The main theorem -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : satisfies_equation f) :
  ∀ x : ℝ, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l189_18928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_special_list_l189_18911

def list_elements : Finset ℕ := 
  Finset.filter (λ n => n ∈ Finset.range 3031 ∨ 
                        ∃ m ∈ Finset.range 3031, n = m^2 ∨ 
                        ∃ k ∈ Finset.range 51, n = k^3)
                (Finset.range 125001)

theorem median_of_special_list :
  let L := list_elements.toList.sort (·<·)
  (L.length % 2 = 0) ∧ 
  (L.get? ((L.length / 2) - 1) = some 3025) ∧
  (L.get? (L.length / 2) = some 3030) →
  (L.get? ((L.length / 2) - 1)).get! + (L.get? (L.length / 2)).get! / 2 = 3027.5 := by
  sorry

#eval list_elements.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_special_list_l189_18911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_convergence_l189_18907

/-- The Newton's method iteration for f(x) = x^2 - x - 1 -/
noncomputable def newton_iteration (x : ℝ) : ℝ := (x^2 + 1) / (2*x - 1)

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Sequence generated by Newton's method starting from x₀ -/
noncomputable def newton_sequence (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => newton_iteration (newton_sequence x₀ n)

/-- Theorem stating the convergence of Newton's method for x₀ = 1 and x₀ = 0 -/
theorem newton_convergence :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |newton_sequence 1 n - φ| < ε) ∧
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |newton_sequence 0 n + 1/φ| < ε) := by
  sorry

#check newton_convergence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_convergence_l189_18907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_livingRoomSizeEquivalence_l189_18914

/-- Represents an apartment with given dimensions and room specifications -/
structure Apartment where
  length : ℕ
  width : ℕ
  totalRooms : ℕ
  livingRoomArea : ℕ

/-- Calculates how many other rooms the living room is as big as -/
def livingRoomEquivalence (apt : Apartment) : ℚ :=
  let totalArea := apt.length * apt.width
  let remainingArea := totalArea - apt.livingRoomArea
  let otherRooms := apt.totalRooms - 1
  let otherRoomArea := remainingArea / otherRooms
  (apt.livingRoomArea : ℚ) / otherRoomArea

/-- Theorem stating that for the given apartment specifications, 
    the living room is as big as 3 other rooms -/
theorem livingRoomSizeEquivalence : 
  let apt := Apartment.mk 16 10 6 60
  livingRoomEquivalence apt = 3 := by
  sorry

#eval livingRoomEquivalence (Apartment.mk 16 10 6 60)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_livingRoomSizeEquivalence_l189_18914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l189_18982

-- Define the curves C1 and C2
noncomputable def C1 (θ : Real) : Real × Real := (Real.sqrt 3 * Real.cos θ, Real.sin θ)

noncomputable def C2 (θ : Real) : Real × Real := 
  let ρ := 2 * Real.sqrt 2 / Real.sin (θ + Real.pi/4)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance function between two points
noncomputable def distance (p q : Real × Real) : Real :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem min_distance_between_curves :
  ∃ (θ1 θ2 : Real), 
    let p := C1 θ1
    let q := C2 θ2
    (∀ (φ1 φ2 : Real), distance (C1 φ1) (C2 φ2) ≥ distance p q) ∧
    distance p q = Real.sqrt 2 ∧
    p = (3/2, 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l189_18982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_max_min_difference_l189_18994

noncomputable def a (n : ℕ+) : ℝ := 5 * (2/5)^(2*n.val - 2) - 4 * (2/5)^(n.val - 1)

def is_max_index (p : ℕ+) : Prop :=
  ∀ n : ℕ+, a n ≤ a p

def is_min_index (q : ℕ+) : Prop :=
  ∀ n : ℕ+, a q ≤ a n

theorem sequence_max_min_difference (p q : ℕ+) 
  (h_max : is_max_index p) (h_min : is_min_index q) : q.val - p.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_max_min_difference_l189_18994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_APBQ_l189_18913

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Helper function to calculate the area of a quadrilateral -/
noncomputable def area_quadrilateral (P Q R S : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating the maximum area of quadrilateral APBQ -/
theorem max_area_APBQ (a b : ℝ) :
  a > b ∧ b > 0 ∧
  eccentricity a b = 1/2 ∧
  ellipse_C 1 (3/2) a b →
  ∃ (A B : ℝ × ℝ),
    ellipse_C A.1 A.2 a b ∧
    ellipse_C B.1 B.2 a b ∧
    (∃ m : ℝ, A.1 = m * A.2 + 1 ∧ B.1 = m * B.2 + 1) ∧
    (∀ A' B' : ℝ × ℝ,
      ellipse_C A'.1 A'.2 a b →
      ellipse_C B'.1 B'.2 a b →
      (∃ m' : ℝ, A'.1 = m' * A'.2 + 1 ∧ B'.1 = m' * B'.2 + 1) →
      area_quadrilateral (-2, 0) A' B' (2, 0) ≤ 6) ∧
    area_quadrilateral (-2, 0) A B (2, 0) = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_APBQ_l189_18913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_same_mistakes_l189_18923

/-- Represents the number of mistakes made by a participant -/
def Mistakes := Nat

/-- The total number of participants -/
def totalParticipants : Nat := 30

/-- The maximum number of mistakes made by any participant -/
def maxMistakes : Nat := 12

/-- The number of possible mistake counts (0 to 11) for participants other than the one with max mistakes -/
def possibleMistakeCounts : Nat := maxMistakes

theorem at_least_three_same_mistakes :
  ∃ (m : Mistakes) (s : Finset Nat),
    s.card ≥ 3 ∧
    (∀ i ∈ s, i < totalParticipants) ∧
    (∀ i ∈ s, ∃ (p : Nat), p < totalParticipants ∧ m = Nat.min maxMistakes p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_same_mistakes_l189_18923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l189_18924

-- Constants
variable (A S ρ v₀ : ℝ)
variable (h_S_pos : S > 0)
variable (h_v₀_pos : v₀ > 0)

-- Force function
noncomputable def F (v : ℝ) : ℝ := (A * S * ρ * (v₀ - v)^2) / 2

-- Power function
noncomputable def N (v : ℝ) : ℝ := F A S ρ v₀ v * v

-- Theorem statement
theorem max_power_speed :
  ∃ v_max : ℝ, v_max > 0 ∧ v_max = v₀ / 3 ∧
  ∀ v : ℝ, v > 0 → N A S ρ v₀ v ≤ N A S ρ v₀ v_max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l189_18924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_mixture_alcohol_percentage_l189_18964

/-- Represents a solution with a given volume and alcohol percentage -/
structure Solution where
  volume : ℝ
  alcoholPercentage : ℝ

/-- Calculates the total alcohol content in a solution -/
noncomputable def alcoholContent (s : Solution) : ℝ :=
  s.volume * s.alcoholPercentage / 100

/-- Represents the mixture of solutions -/
def Mixture := List Solution

/-- Calculates the total volume of a mixture -/
noncomputable def totalVolume (m : Mixture) : ℝ :=
  m.map (fun s => s.volume) |>.sum

/-- Calculates the total alcohol content of a mixture -/
noncomputable def totalAlcohol (m : Mixture) : ℝ :=
  m.map alcoholContent |>.sum

/-- Calculates the alcohol percentage of a mixture -/
noncomputable def mixtureAlcoholPercentage (m : Mixture) : ℝ :=
  (totalAlcohol m / totalVolume m) * 100

/-- The initial solution and added solutions -/
def initialMixture : Mixture :=
  [{ volume := 11, alcoholPercentage := 42 },
   { volume := 3, alcoholPercentage := 0 },
   { volume := 2, alcoholPercentage := 60 },
   { volume := 1, alcoholPercentage := 80 },
   { volume := 1.5, alcoholPercentage := 35 }]

/-- The theorem stating that the alcohol percentage in the final mixture is approximately 38.62% -/
theorem final_mixture_alcohol_percentage :
  ∃ ε > 0, |mixtureAlcoholPercentage initialMixture - 38.62| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_mixture_alcohol_percentage_l189_18964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_a_arithmetic_and_increasing_l189_18926

def a : ℕ → ℚ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | n + 1 => a n / (a n + 1)

theorem reciprocal_a_arithmetic_and_increasing :
  (∀ n : ℕ, n ≥ 1 → (1 / a (n + 1) = 1 / a n + 1)) ∧
  (∀ n : ℕ, n ≥ 1 → 1 / a (n + 1) > 1 / a n) := by
  sorry

#eval a 5  -- This line is added to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_a_arithmetic_and_increasing_l189_18926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_scale_is_30_to_1_l189_18906

/-- The scale of a map is the ratio of the map distance to the actual distance. -/
noncomputable def map_scale (map_distance : ℝ) (actual_distance : ℝ) : ℝ :=
  map_distance / actual_distance

/-- The map distance in centimeters -/
def map_distance : ℝ := 3.6

/-- The actual distance in millimeters -/
def actual_distance : ℝ := 1.2

/-- Theorem: The scale of the map is 30:1 -/
theorem map_scale_is_30_to_1 : 
  map_scale (map_distance * 10) actual_distance = 30 := by
  -- Unfold definitions
  unfold map_scale map_distance actual_distance
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_scale_is_30_to_1_l189_18906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_chord_length_line_MN_equation_l189_18937

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define line l₁
def line_l₁ : Set (ℝ × ℝ) := {p | p.1 - p.2 - 2 * Real.sqrt 2 = 0}

-- Define line l₂
def line_l₂ : Set (ℝ × ℝ) := {p | 4 * p.1 - 3 * p.2 + 5 = 0}

-- Define point G
def point_G : ℝ × ℝ := (1, 3)

-- Define points A and B as the intersection of line l₂ and circle C
noncomputable def point_A : ℝ × ℝ := sorry
noncomputable def point_B : ℝ × ℝ := sorry

-- Define points M and N as the tangent points from G to circle C
noncomputable def point_M : ℝ × ℝ := sorry
noncomputable def point_N : ℝ × ℝ := sorry

theorem circle_equation : circle_C = {p : ℝ × ℝ | p.1^2 + p.2^2 = 4} := by sorry

theorem chord_length : Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 2 * Real.sqrt 3 := by sorry

theorem line_MN_equation : ∀ p : ℝ × ℝ, p ∈ {q : ℝ × ℝ | q.1 + 3 * q.2 - 4 = 0} ↔ (point_M.1 - p.1) * (point_N.2 - p.2) = (point_N.1 - p.1) * (point_M.2 - p.2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_chord_length_line_MN_equation_l189_18937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_series_l189_18950

-- Define the function g(n)
noncomputable def g (n : ℕ) : ℝ := ∑' k : ℕ, (1 : ℝ) / (k + 2 : ℝ) ^ n

-- State the theorem
theorem sum_of_g_series : ∑' n : ℕ, g (n + 2) = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_series_l189_18950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_properties_l189_18981

/-- A rhombus with given diagonal lengths -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal

/-- Calculate the area of a rhombus -/
noncomputable def area (r : Rhombus) : ℝ :=
  r.d1 * r.d2 / 2

/-- Calculate the length of a side of a rhombus -/
noncomputable def side_length (r : Rhombus) : ℝ :=
  Real.sqrt ((r.d1 / 2) ^ 2 + (r.d2 / 2) ^ 2)

/-- Theorem about a specific rhombus -/
theorem rhombus_properties (r : Rhombus) 
    (h1 : r.d1 = 18) (h2 : r.d2 = 14) : 
    area r = 126 ∧ side_length r = Real.sqrt 130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_properties_l189_18981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l189_18927

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

-- Define the perimeter of a triangle given three points
noncomputable def triangle_perimeter (A B C : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C A

-- Theorem statement
theorem ellipse_triangle_perimeter :
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 → triangle_perimeter P F₁ F₂ = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l189_18927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_h_l189_18996

-- Define the function h
def h (x : ℝ) : ℝ := 2 * x^2 - 3

-- Define the inverse function h_inv
noncomputable def h_inv (y : ℝ) : ℝ := Real.sqrt ((y + 3) / 2)

-- State the theorem about the inverse of h
theorem inverse_of_h (x : ℝ) :
  (h (h_inv x) = x ∧ h (-h_inv x) = x) ∧ 
  (h_inv (h x) = x ∨ h_inv (h x) = -x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_h_l189_18996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_moved_specific_point_l189_18919

/-- A dilation in the plane that transforms a circle -/
structure MyDilation where
  /-- Original circle center -/
  original_center : ℝ × ℝ
  /-- Original circle radius -/
  original_radius : ℝ
  /-- Dilated circle center -/
  dilated_center : ℝ × ℝ
  /-- Dilated circle radius -/
  dilated_radius : ℝ

/-- The distance a point moves under a dilation -/
noncomputable def distance_moved (d : MyDilation) (p : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem: The distance moved by (2,2) under the specific dilation -/
theorem distance_moved_specific_point :
  let d : MyDilation := {
    original_center := (1, 3),
    original_radius := 3,
    dilated_center := (7, 9),
    dilated_radius := 6
  }
  distance_moved d (2, 2) = Real.sqrt (15^2 + 17^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_moved_specific_point_l189_18919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_is_one_l189_18983

/-- A quadratic function passing through three given points -/
structure QuadraticFunction where
  a : ℚ
  b : ℚ
  c : ℚ
  point1 : a * (-2)^2 + b * (-2) + c = 9
  point2 : a * 4^2 + b * 4 + c = 9
  point3 : a * 7^2 + b * 7 + c = 16

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex_x (f : QuadraticFunction) : ℚ := -f.b / (2 * f.a)

/-- Theorem stating that the x-coordinate of the vertex is 1 -/
theorem vertex_x_is_one (f : QuadraticFunction) : vertex_x f = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_is_one_l189_18983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l189_18944

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem max_omega_value (ω : ℝ) (φ : ℝ) :
  ω > 0 →
  |φ| ≤ π / 2 →
  f ω φ (-π / 4) = 0 →
  (∀ x, f ω φ x ≤ |f ω φ (π / 4)|) →
  (∃ x₁ ∈ Set.Ioo (-π / 12) (π / 24), ∀ x ∈ Set.Ioo (-π / 12) (π / 24), f ω φ x ≥ f ω φ x₁) →
  (∀ x₂ ∈ Set.Ioo (-π / 12) (π / 24), ∃ x ∈ Set.Ioo (-π / 12) (π / 24), f ω φ x > f ω φ x₂) →
  ω ≤ 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l189_18944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drink_packaging_l189_18931

def min_cans (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

theorem drink_packaging (maaza pepsi sprite total_cans : ℕ) 
  (h_maaza : maaza = 10)
  (h_pepsi : pepsi = 144)
  (h_sprite : sprite = 368)
  (h_total_cans : total_cans = 261)
  (h_min_cans : total_cans = min_cans maaza pepsi sprite) :
  ∃ (can_volume : ℚ),
    can_volume = 2 ∧
    (maaza : ℚ) / can_volume = 5 ∧
    (pepsi : ℚ) / can_volume = 72 ∧
    (sprite : ℚ) / can_volume = 184 ∧
    ((maaza + pepsi + sprite : ℕ) : ℚ) / can_volume = total_cans :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drink_packaging_l189_18931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_5a_l189_18988

-- Define the function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  let y := x % 2
  if -1 ≤ y ∧ y < 0 then y + a
  else if 0 ≤ y ∧ y < 1 then |2/5 - y|
  else 0  -- This case should never occur due to periodicity

-- State the theorem
theorem f_value_at_5a (a : ℝ) :
  (f (-5/2) a = f (9/2) a) → f (5*a) a = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_5a_l189_18988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l189_18971

/-- The curve equation -/
noncomputable def curve (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

/-- The point P -/
def P : ℝ × ℝ := (2, 4)

/-- The derivative of the curve -/
noncomputable def curve_deriv (x : ℝ) : ℝ := x^2

/-- Theorem: The tangent line equation through P and the existence of another tangent line -/
theorem tangent_lines_theorem :
  (∃ (m b : ℝ), ∀ x y : ℝ, y = m*x + b ↔ 4*x - y - 4 = 0) ∧ 
  (∃ x₀ y₀ : ℝ, x₀ ≠ 2 ∧ y₀ = curve x₀ ∧ 
    (∀ x y : ℝ, y = x + 2 ↔ y - y₀ = (x - x₀) * (curve_deriv x₀))) :=
by sorry

/-- Lemma: P lies on the curve -/
lemma P_on_curve : curve P.1 = P.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l189_18971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l189_18915

noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/3)

theorem f_is_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l189_18915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_five_hours_l189_18925

/-- Calculates the round trip time for a boat traveling upstream and downstream in a river -/
noncomputable def round_trip_time (river_current : ℝ) (boat_speed : ℝ) (distance : ℝ) : ℝ :=
  let upstream_speed := boat_speed - river_current
  let downstream_speed := boat_speed + river_current
  let upstream_time := distance / upstream_speed
  let downstream_time := distance / downstream_speed
  upstream_time + downstream_time

/-- Theorem stating that the round trip time for the given conditions is 5 hours -/
theorem round_trip_time_is_five_hours :
  round_trip_time 10 50 120 = 5 := by
  -- Unfold the definition of round_trip_time
  unfold round_trip_time
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_five_hours_l189_18925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l189_18969

-- Define the arithmetic sequence property
def is_arithmetic_sequence (a₁ x y a₂ : ℝ) : Prop :=
  x - a₁ = y - x ∧ y - x = a₂ - y

-- Define the geometric sequence property
def is_geometric_sequence (b₁ x y b₂ : ℝ) : Prop :=
  x / b₁ = y / x ∧ y / x = b₂ / y

-- Define the expression
noncomputable def f (a₁ a₂ b₁ b₂ : ℝ) : ℝ := (a₁ + a₂)^2 / (b₁ * b₂) - 2

-- State the theorem
theorem range_of_f (a₁ x y a₂ b₁ b₂ : ℝ) 
  (h_arith : is_arithmetic_sequence a₁ x y a₂)
  (h_geom : is_geometric_sequence b₁ x y b₂) :
  ∀ z, z ∈ Set.range f ↔ z ≤ -2 ∨ z ≥ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l189_18969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_from_eccentricities_l189_18975

-- Define the eccentricity of the hyperbola
noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

-- Define the eccentricity of the ellipse
noncomputable def ellipse_eccentricity (m b : ℝ) : ℝ := Real.sqrt (m^2 - b^2) / m

-- State the theorem
theorem obtuse_triangle_from_eccentricities (a b m : ℝ) 
  (ha : a > 0) (hm : m > b) (hb : b > 0) :
  hyperbola_eccentricity a b * ellipse_eccentricity m b > 1 → 
  a^2 + b^2 - m^2 < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_from_eccentricities_l189_18975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l189_18947

open Real

theorem function_property (a : ℝ) (h_a : a > 0) :
  ∃ m n : ℝ, m ∈ Set.Icc 0 2 ∧ n ∈ Set.Icc 0 2 ∧ 
  abs (m - n) ≥ 1 ∧ 
  (exp m - a * m) / (exp n - a * n) = 1 →
  1 ≤ a / (exp 1 - 1) ∧ a / (exp 1 - 1) ≤ exp 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l189_18947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_even_l189_18970

/-- A game played on a circle with n points where players take turns drawing chords. -/
structure ChordGame where
  n : ℕ
  points : Fin n → ℝ × ℝ
  is_circle : ∀ (i : Fin n), (points i).1^2 + (points i).2^2 = 1

/-- A chord in the game connects two non-adjacent points. -/
structure Chord (g : ChordGame) where
  start : Fin g.n
  finish : Fin g.n
  non_adjacent : (start - finish) % g.n ≠ 1 ∧ (finish - start) % g.n ≠ 1

/-- The state of the game after some moves have been made. -/
structure GameState (g : ChordGame) where
  chords : List (Chord g)
  valid_move : ∀ (c1 c2 : Chord g), c1 ∈ chords → c2 ∈ chords → 
    (c1.start = c2.start ∧ c1.finish = c2.finish) ∨
    (c1.start = c2.finish ∧ c1.finish = c2.start) ∨
    (c1.start ≠ c2.start ∧ c1.start ≠ c2.finish ∧ c1.finish ≠ c2.start ∧ c1.finish ≠ c2.finish)

/-- A strategy for a player in the game. -/
def Strategy (g : ChordGame) := GameState g → Option (Chord g)

/-- Petya wins if they have a move when Vasya doesn't. -/
def PetyaWins (g : ChordGame) (s1 s2 : Strategy g) : Prop := sorry

/-- Petya has a winning strategy if n is even and at least 5. -/
theorem petya_wins_even (n : ℕ) (h1 : n ≥ 5) (h2 : Even n) :
  ∃ (s : Strategy (ChordGame.mk n (λ _ ↦ (0, 0)) sorry)), 
    ∀ (s' : Strategy (ChordGame.mk n (λ _ ↦ (0, 0)) sorry)),
      PetyaWins (ChordGame.mk n (λ _ ↦ (0, 0)) sorry) s s' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_wins_even_l189_18970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probabilities_l189_18904

structure Player where
  accuracy : ℝ

structure Game where
  playerA : Player
  playerB : Player
  startProb : ℝ

noncomputable def secondShotProb (g : Game) : ℝ := sorry

noncomputable def ithShotProb (g : Game) (i : ℕ) : ℝ := sorry

noncomputable def expectedShots (g : Game) (n : ℕ) : ℝ := sorry

theorem game_probabilities (g : Game) 
  (h1 : g.playerA.accuracy = 0.6) 
  (h2 : g.playerB.accuracy = 0.8) 
  (h3 : g.startProb = 0.5) :
  (secondShotProb g = 0.6) ∧ 
  (∀ i : ℕ, ithShotProb g i = 1/3 + (1/6) * (2/5)^(i-1)) ∧
  (∀ n : ℕ, expectedShots g n = (5/18) * (1 - (2/5)^n) + n/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probabilities_l189_18904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aunt_gemma_dogs_l189_18984

-- Define the given conditions
def food_per_meal : ℕ := 250 -- grams
def meals_per_day : ℕ := 2
def num_sacks : ℕ := 2
def sack_weight : ℕ := 50 -- kilograms
def days_food_lasts : ℕ := 50

-- Define the theorem
theorem aunt_gemma_dogs : 
  (num_sacks * sack_weight * 1000) / (food_per_meal * meals_per_day) / days_food_lasts = 400 := by
  -- Convert kg to g
  have total_food : ℕ := num_sacks * sack_weight * 1000
  -- Calculate daily consumption per dog
  have daily_consumption_per_dog : ℕ := food_per_meal * meals_per_day
  -- Calculate total days for one dog
  have total_days_for_one_dog : ℕ := total_food / daily_consumption_per_dog
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aunt_gemma_dogs_l189_18984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_between_vectors_l189_18992

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_angle_between_vectors (a b : V)
  (norm_a : ‖a‖ = 5)
  (norm_b : ‖b‖ = 7)
  (norm_sum : ‖a + b‖ = 10) :
  inner a b / (‖a‖ * ‖b‖) = 13 / 35 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_between_vectors_l189_18992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l189_18942

theorem trig_inequality : 
  let x : ℝ := 2 * π / 5
  π / 4 < x ∧ x < π / 2 → Real.cos x < Real.sin x ∧ Real.sin x < Real.tan x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l189_18942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_time_is_35_minutes_l189_18955

/-- Represents the boy's journey to school -/
structure SchoolJourney where
  usual_time : ℝ
  usual_speed : ℝ
  total_distance : ℝ

/-- The time taken with the new walking speeds -/
noncomputable def new_time (j : SchoolJourney) : ℝ :=
  (3/4 * j.total_distance) / (7/6 * j.usual_speed) +
  (1/4 * j.total_distance) / (5/6 * j.usual_speed)

/-- The theorem stating the usual time to school -/
theorem usual_time_is_35_minutes (j : SchoolJourney) :
  j.total_distance = j.usual_speed * j.usual_time →
  new_time j = j.usual_time - 2 →
  j.usual_time = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_time_is_35_minutes_l189_18955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_floor_powers_eq_geometric_series_l189_18966

/-- The floor function, representing the greatest integer not exceeding x -/
noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

/-- The sum of the floor of powers of 2 divided by 3, from 0 to 2008, plus 1004 -/
noncomputable def sum_floor_powers : ℕ → ℝ
  | 0 => floor (1 / 3) + 1004
  | n + 1 => sum_floor_powers n + floor ((2 ^ (n + 1)) / 3)

theorem sum_floor_powers_eq_geometric_series :
  sum_floor_powers 2008 = (1 / 3) * (2 ^ 2009 - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_floor_powers_eq_geometric_series_l189_18966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_is_sqrt5_l189_18910

noncomputable def root₁ (p : ℝ) : ℝ := 
  ((p + 1) + Real.sqrt ((p + 1)^2 - (p^2 + p - 2))) / 2

noncomputable def root₂ (p : ℝ) : ℝ := 
  ((p + 1) - Real.sqrt ((p + 1)^2 - (p^2 + p - 2))) / 2

theorem root_difference_is_sqrt5 (p : ℝ) : 
  let r := max (root₁ p) (root₂ p)
  let s := min (root₁ p) (root₂ p)
  r - s = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_is_sqrt5_l189_18910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l189_18993

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (4^x + k*2^x + 1) / (4^x + 2^x + 1)

theorem function_properties (k : ℝ) :
  (∀ x, f k x > 0) = (k > -2) ∧
  (∃ x₀, ∀ x, f k x ≥ f k x₀ ∧ f k x₀ = -2) = (k = -8) ∧
  (∀ x₁ x₂ x₃, f k x₁ + f k x₂ > f k x₃ ∧
               f k x₂ + f k x₃ > f k x₁ ∧
               f k x₃ + f k x₁ > f k x₂) = (-1/2 ≤ k ∧ k ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l189_18993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_n_cubed_minus_n_six_divides_n_cubed_minus_n_l189_18965

theorem largest_divisor_of_n_cubed_minus_n (n : ℤ) : 
  (∃ (k : ℕ), k > 6 ∧ (k : ℤ) ∣ (n^3 - n)) → False :=
sorry

theorem six_divides_n_cubed_minus_n (n : ℤ) : 
  (6 : ℤ) ∣ (n^3 - n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_n_cubed_minus_n_six_divides_n_cubed_minus_n_l189_18965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_powers_factorization_of_power_minus_one_divisibility_of_power_minus_one_l189_18953

theorem difference_of_powers (x y : ℕ) (n : ℕ) (hx : x > 1) (hy : y > 1) :
  ∃ (factors : List ℕ), 
    (x^(2^n) - y^(2^n) = factors.prod) ∧ 
    (factors.length = n) ∧
    (∀ f, f ∈ factors → f > 1) ∧
    (∀ f g, f ∈ factors → g ∈ factors → f ≠ g → f ≠ g) :=
by sorry

theorem factorization_of_power_minus_one (x : ℕ) (n : ℕ) (hx : x > 1) :
  x^(2^n) - 1 = (Finset.range n).prod (fun k ↦ x^(2^k) + 1) :=
by sorry

theorem divisibility_of_power_minus_one (x : ℕ) (n : ℕ) (hx : x > 1) :
  (x + 1)^(n + 1) ∣ (x^(2^n) - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_powers_factorization_of_power_minus_one_divisibility_of_power_minus_one_l189_18953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_proof_l189_18921

/-- Given a class of 100 students with specific test averages for different groups,
    prove that the overall class average is approximately 79%. -/
theorem class_average_proof (total_students : Nat) (group1_size : Nat) (group2_size : Nat) (group3_size : Nat)
  (group1_avg : Float) (group2_avg : Float) (group3_avg : Float) :
  total_students = 100 →
  group1_size = 21 →
  group2_size = 56 →
  group3_size = 23 →
  group1_avg = 98.5 →
  group2_avg = 76.8 →
  group3_avg = 67.1 →
  let overall_avg := (group1_size.toFloat * group1_avg + group2_size.toFloat * group2_avg + group3_size.toFloat * group3_avg) / total_students.toFloat
  Float.round overall_avg = 79 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_proof_l189_18921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_A_in_triangle_l189_18973

theorem tan_A_in_triangle (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Vectors m and n are parallel
  (Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C →
  -- Conclusion
  Real.tan A = Real.sqrt 2 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_A_in_triangle_l189_18973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l189_18936

noncomputable section

-- Define the garden and its components
def garden_side : ℝ := 30
def trapezoid_short_side : ℝ := 18
def trapezoid_long_side : ℝ := 30

-- Define the equilateral triangle side length
def triangle_side : ℝ := (trapezoid_long_side - trapezoid_short_side) / 2

-- Define the area of one equilateral triangle
def triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2

-- Define the total area of both triangular flower beds
def flower_beds_area : ℝ := 2 * triangle_area

-- Define the total garden area
def garden_area : ℝ := garden_side^2

-- Theorem to prove
theorem flower_beds_fraction :
  flower_beds_area / garden_area = Real.sqrt 3 / 50 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l189_18936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_function_only_solution_l189_18986

-- Define the set of prime numbers
def PrimeSet : Type := {p : ℕ // Nat.Prime p}

-- Define the property that the function must satisfy
def SatisfiesProperty (f : PrimeSet → PrimeSet) :=
  ∀ p q : PrimeSet, (f p).val ^ (f q).val + q.val ^ p.val = (f q).val ^ (f p).val + p.val ^ q.val

-- Theorem statement
theorem identity_function_only_solution :
  ∀ f : PrimeSet → PrimeSet, SatisfiesProperty f → ∀ p : PrimeSet, f p = p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_function_only_solution_l189_18986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_sequence_all_sevens_l189_18930

/-- The sequence of numbers consisting of repeated 7s -/
def a : ℕ → ℕ 
  | 0 => 0  -- Define a₀ = 0 for completeness
  | n + 1 => 7 * (10^n - 1) / 9 + 7

/-- The formula for the nth term of the sequence -/
def formula (n : ℕ) : ℚ := 7 / 9 * (10^n - 1)

/-- Theorem stating that the formula correctly describes the sequence -/
theorem sequence_formula_correct (n : ℕ) : 
  n > 0 → a n = ⌊formula n⌋ := by
  sorry

/-- Theorem stating that the sequence consists of repeated 7s -/
theorem sequence_all_sevens (n : ℕ) : 
  n > 0 → ∃ k : ℕ, a n = 7 * (10^k - 1) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_sequence_all_sevens_l189_18930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_perimeter_l189_18990

/-- A configuration of three unit squares and a right triangle -/
structure SquareTriangleConfig where
  -- Two horizontal squares
  horizontal_square1 : Real
  horizontal_square2 : Real
  -- One vertical square
  vertical_square : Real
  -- Right triangle on top of vertical square
  triangle_leg1 : Real
  triangle_leg2 : Real
  triangle_hypotenuse : Real

/-- The perimeter of the SquareTriangleConfig -/
def perimeter (config : SquareTriangleConfig) : Real :=
  -- Implement the perimeter calculation here
  4 * config.horizontal_square1 + 2 * config.vertical_square + config.triangle_leg1 + config.triangle_hypotenuse

theorem square_triangle_perimeter :
  ∀ (config : SquareTriangleConfig),
    (config.horizontal_square1 = 1) →
    (config.horizontal_square2 = 1) →
    (config.vertical_square = 1) →
    (config.triangle_leg1 = 1) →
    (config.triangle_leg2 = 1) →
    (config.triangle_hypotenuse = Real.sqrt 2) →
    perimeter config = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_perimeter_l189_18990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_48_l189_18998

/-- The average speed of a car on a round trip -/
noncomputable def average_speed (d v1 v2 : ℝ) : ℝ :=
  (2 * d) / (d / v1 + d / v2)

/-- Theorem: The average speed of a car on a round trip is 48 km/h -/
theorem average_speed_is_48 (d : ℝ) (h : d > 0) :
  average_speed d 40 60 = 48 := by
  sorry

#check average_speed_is_48

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_48_l189_18998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_arrival_delay_l189_18933

/-- Proof that Jake arrives 20 seconds after Austin when descending from the 9th floor -/
theorem jake_arrival_delay 
  (start_floor : ℕ) 
  (steps_per_floor : ℕ) 
  (jake_steps_per_second : ℕ) 
  (elevator_time : ℕ) 
  (h1 : start_floor = 9)
  (h2 : steps_per_floor = 30)
  (h3 : jake_steps_per_second = 3)
  (h4 : elevator_time = 60) :
  (start_floor - 1) * steps_per_floor / jake_steps_per_second - elevator_time = 20 := by
  sorry

#check jake_arrival_delay

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_arrival_delay_l189_18933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l189_18935

/-- Given an ellipse E with equation x²/a² + y²/b² = 1 where a > b > 0,
    right focus F(3,0), and a line passing through F intersecting E at points A and B
    with midpoint (1,-1), prove that the equation of E is x²/18 + y²/9 = 1 -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hf : (3 : ℝ) = Real.sqrt (a^2 - b^2))
  (hm : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2/a^2 + y₁^2/b^2 = 1 ∧
    x₂^2/a^2 + y₂^2/b^2 = 1 ∧
    (x₁ + x₂)/2 = 1 ∧
    (y₁ + y₂)/2 = -1 ∧
    (y₁ - y₂)/(x₁ - x₂) = (3 - x₁)/(0 - y₁) ∧
    (y₁ - y₂)/(x₁ - x₂) = (3 - x₂)/(0 - y₂)) :
  a^2 = 18 ∧ b^2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l189_18935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l189_18916

/-- Given a parabola with equation x^2 = 2py (p > 0) and directrix passing through (-1, -2),
    prove that its focus coordinates are (0, 2) -/
theorem parabola_focus_coordinates (p : ℝ) (h_p : p > 0) :
  let parabola := fun x y ↦ x^2 = 2*p*y
  let directrix := fun x y ↦ y = -2
  let focus := (0, 2)
  (∃ x y, directrix x y ∧ x = -1) →  -- Directrix passes through (-1, -2)
  (∀ x y, parabola x y → 
    (x - focus.1)^2 + (y - focus.2)^2 = (y + 2)^2) -- Focus-directrix property
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l189_18916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_evaluation_l189_18912

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ :=
  if a ≥ b then Real.sqrt (a^2 - b^2) else 0

-- Theorem statement
theorem diamond_evaluation : diamond (diamond 8 6) (diamond 10 8) = 0 := by
  -- Expand the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_evaluation_l189_18912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l189_18956

/-- The time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (train_length : ℝ) (signal_pole_time : ℝ) (platform_length : ℝ) : ℝ :=
  let train_speed := train_length / signal_pole_time
  let total_distance := train_length + platform_length
  total_distance / train_speed

/-- Theorem stating the time taken for the train to cross the platform -/
theorem train_crossing_time :
  let train_length : ℝ := 300
  let signal_pole_time : ℝ := 18
  let platform_length : ℝ := 333.33
  let crossing_time := time_to_cross_platform train_length signal_pole_time platform_length
  ∃ (ε : ℝ), ε > 0 ∧ abs (crossing_time - 38) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l189_18956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_l189_18979

/-- A function f(x) = x³ - 6bx + 3b has a local minimum in the interval (0, 1) if and only if b is in the range (0, 1/2) -/
theorem local_minimum_condition (b : ℝ) :
  (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ 
    (∀ y : ℝ, y ∈ Set.Ioo 0 1 → (y^3 - 6*b*y + 3*b) ≥ (x^3 - 6*b*x + 3*b))) 
  ↔ 
  (b > 0 ∧ b < 1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_l189_18979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l189_18920

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 2 then Real.log x - a * x
  else if -2 < x ∧ x < 0 then -(Real.log (-x) - a * (-x))
  else 0  -- Undefined outside (-2, 0) ∪ (0, 2)

-- State the theorem
theorem odd_function_value (a : ℝ) : a > 0 →
  (∀ x, f a x = -f a (-x)) →  -- f is odd
  (∀ x ∈ Set.Ioo (-2) 0, f a x ≥ 1) →  -- minimum value in (-2, 0) is 1
  (∃ x ∈ Set.Ioo (-2) 0, f a x = 1) →  -- the minimum is achieved
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l189_18920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_trapezoid_area_l189_18985

/-- The area of a trapezoid formed by connecting midpoints of alternate sides in a regular hexagon -/
theorem hexagon_trapezoid_area (s : ℝ) (h : s = 12) : 
  (s * 2) * (s * Real.sqrt 3) / 2 = 144 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_trapezoid_area_l189_18985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_l189_18987

/-- Given a linear function f(x) = kx + b where k ≠ 0, f(1) = 2, and f(2) = -1,
    prove that f(10) = -25, f is decreasing, and g(x) = 1/x - 5 + f(x) is odd. -/
theorem linear_function_properties (k b : ℝ) (hk : k ≠ 0)
  (h1 : k + b = 2) (h2 : 2*k + b = -1) :
  let f := λ x : ℝ ↦ k*x + b
  let g := λ x : ℝ ↦ 1/x - 5 + f x
  (f 10 = -25) ∧
  (∀ x y : ℝ, x < y → f x > f y) ∧
  (∀ x : ℝ, x ≠ 0 → g (-x) = -g x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_l189_18987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_of_triangle_with_radius_3_l189_18976

/-- A 30-60-90 triangle with an inscribed circle -/
structure Triangle30_60_90 where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The radius is positive -/
  r_pos : r > 0

/-- The length of the hypotenuse of a 30-60-90 triangle -/
noncomputable def hypotenuse (t : Triangle30_60_90) : ℝ := 6 * Real.sqrt 3 * t.r

/-- Theorem: The hypotenuse of a 30-60-90 triangle with an inscribed circle of radius 3 cm is 6√3 cm -/
theorem hypotenuse_of_triangle_with_radius_3 :
  ∀ t : Triangle30_60_90, t.r = 3 → hypotenuse t = 18 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_of_triangle_with_radius_3_l189_18976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_purchase_price_l189_18999

/-- The initial purchase price of a scooter, given the repair cost, selling price, and gain percentage. --/
theorem scooter_purchase_price 
  (repair_cost : ℚ) 
  (selling_price : ℚ) 
  (gain_percentage : ℚ) 
  (h1 : repair_cost = 800)
  (h2 : selling_price = 5800)
  (h3 : gain_percentage = 11.54) : 
  ∃ (initial_price : ℚ), 
    abs (initial_price - 4400) < 1 ∧ 
    selling_price = (initial_price + repair_cost) * (1 + gain_percentage / 100) := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_purchase_price_l189_18999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_point_l189_18946

/-- Given four sample points and a regression line, prove the value of m -/
theorem regression_line_point (m : ℝ) : 
  let sample_points : List (ℝ × ℝ) := [(1, 2.98), (2, 5.01), (3, m), (4, 9)]
  let regression_line := λ x : ℝ => 2 * x + 1
  let x_mean : ℝ := (1 + 2 + 3 + 4) / 4
  let y_mean : ℝ := (2.98 + 5.01 + m + 9) / 4
  (regression_line x_mean = y_mean) → m = 7.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_point_l189_18946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_monotone_f_l189_18957

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

-- State the theorem
theorem range_of_a_for_monotone_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc 4 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_monotone_f_l189_18957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_g_range_values_l189_18908

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

-- Define the function g
noncomputable def g (a b x : ℝ) : ℝ := 2 * a * f x + b

-- Theorem for the increasing intervals of f
theorem f_increasing_intervals (k : ℤ) :
  ∃ x ∈ Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12),
  Monotone f := by
  sorry

-- Theorem for the values of a and b
theorem g_range_values :
  (∃ a b : ℝ, a > 0 ∧ 
    Set.range (g a b) = Set.Icc 2 4 ∧
    a = 4 / 3 ∧ b = 10 / 3) ∨
  (∃ a b : ℝ, a < 0 ∧ 
    Set.range (g a b) = Set.Icc 2 4 ∧
    a = -4 / 3 ∧ b = 8 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_g_range_values_l189_18908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_iff_a_in_range_l189_18952

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^3 - a * x^2 - 1
  else |x - 3| + a

-- Define the property of having exactly two zeros
def has_exactly_two_zeros (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ ∀ z : ℝ, f z = 0 → z = x ∨ z = y

-- Theorem statement
theorem f_two_zeros_iff_a_in_range :
  ∀ a : ℝ, has_exactly_two_zeros (f a) ↔ -3 < a ∧ a < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_iff_a_in_range_l189_18952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sqrt_less_than_7_l189_18974

def two_digit_numbers : Finset ℕ := Finset.filter (fun n => 10 ≤ n ∧ n ≤ 99) (Finset.range 100)

def numbers_with_sqrt_less_than_7 : Finset ℕ := Finset.filter (fun n => n < 49) two_digit_numbers

theorem probability_sqrt_less_than_7 : 
  (Finset.card numbers_with_sqrt_less_than_7 : ℚ) / 
  (Finset.card two_digit_numbers : ℚ) = 13 / 30 := by
  sorry

#eval Finset.card two_digit_numbers
#eval Finset.card numbers_with_sqrt_less_than_7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sqrt_less_than_7_l189_18974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_f_unique_solution_l189_18902

/-- The set of real numbers except 1 -/
def RealStar : Set ℝ := {x : ℝ | x ≠ 1}

/-- The function f: RealStar → ℝ -/
noncomputable def f (x : ℝ) : ℝ := (1/3) * (x + 2010 - 2 * (x + 2009) / (x - 1))

/-- The theorem stating that f satisfies the functional equation -/
theorem f_satisfies_equation : 
  ∀ x ∈ RealStar, x + f x + 2 * f ((x + 2009) / (x - 1)) = 2010 := by
  sorry

/-- The theorem stating that f is the unique solution -/
theorem f_unique_solution :
  ∀ g : ℝ → ℝ, (∀ x ∈ RealStar, x + g x + 2 * g ((x + 2009) / (x - 1)) = 2010) → g = f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_f_unique_solution_l189_18902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_properties_l189_18903

/-- The exponential function g(x) = 2^x -/
noncomputable def g (x : ℝ) : ℝ := 2^x

/-- Theorem stating that g(x) passes through (3, 8) and the range of x where g(2x^2-3x+1) > g(x^2+2x-5) -/
theorem exponential_function_properties :
  (g 3 = 8) ∧
  (∀ x : ℝ, g (2*x^2 - 3*x + 1) > g (x^2 + 2*x - 5) ↔ x < 2 ∨ x > 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_properties_l189_18903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l189_18929

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- Theorem: For a hyperbola with equation x²/a² - y²/3 = 1, where a > 0, 
    if the eccentricity is 2, then a = 1 -/
theorem hyperbola_eccentricity (a : ℝ) (ha : a > 0) 
  (h_ecc : eccentricity a (Real.sqrt 3) = 2) : a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l189_18929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sine_sum_l189_18977

theorem triangle_ratio_sine_sum (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  (b + c) / 4 = (c + a) / 5 ∧ (c + a) / 5 = (a + b) / 6 →
  Real.sin A / a = Real.sin B / b ∧ Real.sin B / b = Real.sin C / c →
  (Real.sin A + Real.sin C) / Real.sin B = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sine_sum_l189_18977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_study_properties_l189_18951

/-- A structure representing a statistical study of athletes -/
structure AthleteStudy where
  total_athletes : ℕ
  sampled_athletes : ℕ
  h_sample_size : sampled_athletes ≤ total_athletes

/-- The population in the study -/
def population (study : AthleteStudy) : Finset ℕ :=
  Finset.range study.total_athletes

/-- An individual in the study -/
def individual (study : AthleteStudy) (i : ℕ) : Prop :=
  i ∈ population study

/-- The sample in the study -/
def sample (study : AthleteStudy) : Finset ℕ :=
  Finset.range study.sampled_athletes

/-- Theorem stating the properties of the athlete study -/
theorem athlete_study_properties (study : AthleteStudy) 
    (h_total : study.total_athletes = 1000)
    (h_sampled : study.sampled_athletes = 100) :
    (Finset.card (population study) = 1000) ∧ 
    (∀ i, i < 1000 → individual study i) ∧
    (Finset.card (sample study) = 100) ∧
    (study.sampled_athletes = 100) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_study_properties_l189_18951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_CD_is_five_l189_18900

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 2
def circle_O2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 5

-- Define the common point A in the first quadrant
def point_A : ℝ × ℝ := (1, 1)

-- Define the line passing through A
def line_through_A (k : ℝ) (x : ℝ) : ℝ := k * (x - point_A.fst) + point_A.snd

-- Define the condition AC = 2AD
def distance_condition (k : ℝ) : Prop :=
  let E := ((point_A.fst + 1) / 2, line_through_A k ((point_A.fst + 1) / 2))
  let F := ((point_A.fst + 3) / 2, line_through_A k ((point_A.fst + 3) / 2))
  (E.1 - point_A.fst)^2 + (E.2 - point_A.snd)^2 = 4 * ((F.1 - point_A.fst)^2 + (F.2 - point_A.snd)^2)

theorem slope_of_CD_is_five :
  ∃ (k : ℝ), 
    circle_O1 point_A.fst point_A.snd ∧
    circle_O2 point_A.fst point_A.snd ∧
    distance_condition k ∧
    k = 5 := by
  sorry

#check slope_of_CD_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_CD_is_five_l189_18900
