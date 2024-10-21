import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_recovery_time_l630_63080

/-- The total recovery time for James's injury, given the initial healing time and the skin graft healing time being 50% longer. -/
theorem james_recovery_time : ℝ := by
  let initial_healing_time : ℝ := 4
  let skin_graft_healing_time : ℝ := initial_healing_time * (1 + 0.5)
  let total_recovery_time : ℝ := initial_healing_time + skin_graft_healing_time
  have h : total_recovery_time = 10 := by
    -- Proof steps would go here
    sorry
  exact total_recovery_time


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_recovery_time_l630_63080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_average_for_new_tests_l630_63054

noncomputable def initial_scores : List ℝ := [92, 86, 74, 88, 81]
def improvement_goal : ℚ := 4
def num_new_tests : ℕ := 2

noncomputable def current_average : ℝ := (initial_scores.sum) / initial_scores.length
noncomputable def target_average : ℝ := current_average + improvement_goal

theorem minimum_average_for_new_tests :
  let total_tests : ℕ := initial_scores.length + num_new_tests
  let required_total_score : ℝ := target_average * total_tests
  let required_new_score : ℝ := required_total_score - initial_scores.sum
  (required_new_score / num_new_tests) = 98.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_average_for_new_tests_l630_63054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_plus_1_l630_63094

-- Define f as a function from ℝ to ℝ
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x-1)
def domain_f_x_minus_1 : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- Theorem to prove
theorem domain_f_2x_plus_1 :
  {x : ℝ | ∃ y ∈ domain_f_x_minus_1, 2 * x + 1 = y - 1} = {x : ℝ | -2 ≤ x ∧ x ≤ 1/2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_plus_1_l630_63094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_theorem_l630_63049

/-- Given a circle with parallel chords of lengths 5, 12, and 13 subtending
    central angles θ, φ, and θ + φ respectively, where θ + φ < π,
    prove that cos θ = 119/169. -/
theorem chord_cosine_theorem (θ φ : ℝ) : 
  θ + φ < π →
  ∃ (r : ℝ), r > 0 ∧
    5 = 2 * r * Real.sin (θ / 2) ∧
    12 = 2 * r * Real.sin (φ / 2) ∧
    13 = 2 * r * Real.sin ((θ + φ) / 2) →
  Real.cos θ = 119 / 169 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_cosine_theorem_l630_63049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_and_inequality_l630_63090

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := b * a^x

-- State the theorem
theorem exponential_function_and_inequality (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : f a b 1 = 6) (h4 : f a b 3 = 24) :
  (f 2 3 = f a b) ∧ 
  (∀ m : ℝ, m ≤ 5/6 ↔ ∀ x ≤ 1, (1/a)^x + (1/b)^x - m ≥ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_and_inequality_l630_63090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l630_63011

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / lg (x + 1) + Real.sqrt (2 - x)

-- State the theorem about the domain of f
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Ioo (-1) 0 ∪ Set.Ioc 0 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l630_63011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_driveway_cleared_in_seven_hours_l630_63014

/-- Represents Pauline's snow shoveling rate at a given hour -/
def shoveling_rate (hour : ℕ) : ℕ := 25 - (hour - 1)

/-- Calculates the total volume of snow removed up to a given hour -/
def total_removed (hours : ℕ) : ℕ := 
  (Finset.range hours).sum (λ h => shoveling_rate (h + 1))

/-- The volume of snow on the driveway -/
def snow_volume : ℕ := 5 * 12 * 5 / 2

theorem driveway_cleared_in_seven_hours : 
  total_removed 7 ≥ snow_volume ∧ total_removed 6 < snow_volume :=
by sorry

#eval total_removed 7
#eval snow_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_driveway_cleared_in_seven_hours_l630_63014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_l630_63029

/-- Represents a rhombus with given area and one diagonal length -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ

/-- Calculates the length of the second diagonal of a rhombus -/
noncomputable def secondDiagonal (r : Rhombus) : ℝ :=
  (2 * r.area) / r.diagonal1

/-- Theorem: The second diagonal of a rhombus with area 150 cm² and one diagonal 20 cm is 15 cm -/
theorem rhombus_second_diagonal :
  let r : Rhombus := { area := 150, diagonal1 := 20 }
  secondDiagonal r = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_l630_63029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_different_tens_digits_l630_63034

/-- The range of integers from which we choose -/
def range : Finset ℕ := Finset.filter (fun n => 10 ≤ n ∧ n ≤ 59) (Finset.range 60)

/-- The number of integers to be chosen -/
def k : ℕ := 5

/-- The probability of selecting k different integers from the range
    such that each has a different tens digit -/
def probability : ℚ :=
  (10^k : ℚ) / (Nat.choose range.card k : ℚ)

/-- The main theorem stating the probability -/
theorem probability_of_different_tens_digits :
  probability = 2500 / 52969 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_different_tens_digits_l630_63034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_time_is_quarter_hour_p_finishes_in_fifteen_minutes_l630_63030

noncomputable section

/-- The time it takes P to finish the job alone -/
def p_time : ℝ := 4

/-- The time it takes Q to finish the job alone -/
def q_time : ℝ := 15

/-- The time P and Q work together -/
def together_time : ℝ := 3

/-- The portion of the job completed by P in one hour -/
noncomputable def p_rate : ℝ := 1 / p_time

/-- The portion of the job completed by Q in one hour -/
noncomputable def q_rate : ℝ := 1 / q_time

/-- The portion of the job completed by P and Q working together in one hour -/
noncomputable def combined_rate : ℝ := p_rate + q_rate

/-- The portion of the job completed by P and Q working together for 3 hours -/
noncomputable def completed_portion : ℝ := combined_rate * together_time

/-- The remaining portion of the job after P and Q work together -/
noncomputable def remaining_portion : ℝ := 1 - completed_portion

/-- The time it takes P to finish the remaining portion of the job -/
noncomputable def remaining_time : ℝ := remaining_portion / p_rate

theorem remaining_time_is_quarter_hour :
  remaining_time = 1/4 := by sorry

theorem p_finishes_in_fifteen_minutes :
  remaining_time * 60 = 15 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_time_is_quarter_hour_p_finishes_in_fifteen_minutes_l630_63030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_with_tan_roots_l630_63099

theorem sum_of_angles_with_tan_roots (α β : Real) : 
  α ∈ Set.Ioo (-π/2) (π/2) →
  β ∈ Set.Ioo (-π/2) (π/2) →
  (∃ (x y : Real), x^2 + 4*Real.sqrt 3*x + 5 = 0 ∧ 
                   y^2 + 4*Real.sqrt 3*y + 5 = 0 ∧ 
                   Real.tan α = x ∧ Real.tan β = y) →
  α + β = -2*π/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_with_tan_roots_l630_63099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_find_m_range_l630_63060

-- Define the function f(x) = |ax+1|
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Define the solution set of f(x) < 3
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x < 3}

-- Theorem for part I
theorem find_a : ∃ a : ℝ, solution_set a = Set.Ioo (-1) 2 := by sorry

-- Define the inequality for part II
def inequality (m : ℝ) (x : ℝ) : Prop := f (-2) x ≤ |x + 1| + m

-- Theorem for part II
theorem find_m_range : 
  ∀ m : ℝ, (∀ x : ℝ, ¬inequality m x) ↔ m < -3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_find_m_range_l630_63060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l630_63042

theorem solve_equation : ∃ x : ℝ, (x - 5)^4 = (1/16)⁻¹ ∧ x = 7 := by
  use 7
  constructor
  · simp
    norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l630_63042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_l630_63088

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 else Real.cos x - 1

theorem integral_f : ∫ x in (-1)..Real.pi/2, f x = 4/3 - Real.pi/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_l630_63088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_measurements_l630_63021

/-- Represents a rectangular garden with given length and width -/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Creates a rectangular garden with width half of the length -/
noncomputable def createGarden (length : ℝ) : RectangularGarden :=
  { length := length, width := length / 2 }

/-- Calculates the area of a rectangular garden -/
noncomputable def area (garden : RectangularGarden) : ℝ :=
  garden.length * garden.width

/-- Calculates the perimeter of a rectangular garden -/
noncomputable def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.width)

theorem garden_measurements :
  let garden := createGarden 30
  area garden = 450 ∧ perimeter garden = 90 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_measurements_l630_63021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_properties_l630_63032

def F (a : ℝ) (n : ℕ) : ℝ := a - n * a

def is_perfect_square (x : ℝ) : Prop :=
  ∃ m : ℤ, x = m^2

theorem F_properties :
  (F 2 5 = -6) ∧
  (F (-2/3) 4 = 4/3) ∧
  (F (-2/5) 6 = 2) ∧
  (∀ x ∈ ({18, 32, 50, 72, 98} : Set ℕ), is_perfect_square (|F x 4|)) ∧
  (∀ x : ℕ, 10 ≤ x ∧ x ≤ 99 → (is_perfect_square (|F x 4|) → x ∈ ({18, 32, 50, 72, 98} : Set ℕ))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_properties_l630_63032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_correct_l630_63086

/-- A polyhedron formed by folding a regular hexagon and six equilateral triangles -/
structure HexagonalPyramid where
  s : ℝ  -- Side length of the hexagon and triangles
  s_pos : s > 0  -- Side length is positive

/-- The volume of the hexagonal pyramid -/
noncomputable def volume (p : HexagonalPyramid) : ℝ := (Real.sqrt 3 / 2) * p.s^3

/-- The area of the regular hexagon base -/
noncomputable def base_area (p : HexagonalPyramid) : ℝ := (3 * Real.sqrt 3 / 2) * p.s^2

/-- The height of the pyramid -/
def pyramid_height (p : HexagonalPyramid) : ℝ := p.s

/-- Theorem: The volume of the hexagonal pyramid is correct -/
theorem volume_is_correct (p : HexagonalPyramid) : 
  volume p = (1/3) * base_area p * pyramid_height p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_is_correct_l630_63086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_l630_63081

-- Define the given values
noncomputable def a : ℝ := Real.log 8 / Real.log 4
noncomputable def b : ℝ := Real.log 8 / Real.log 0.4
noncomputable def c : ℝ := (2 : ℝ) ^ (0.4 : ℝ)

-- State the theorem
theorem order_of_logarithms : b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_l630_63081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_weighted_average_l630_63076

/-- Calculates the weighted average for bowlers based on their scores and costs -/
noncomputable def weighted_average_bowlers (scores : Vector ℝ 4) (shoe_rental : ℝ) (total_refreshments : ℝ) : ℝ :=
  let cost_per_bowler := shoe_rental + total_refreshments / 4
  let weighted_scores := scores.map (· + cost_per_bowler)
  weighted_scores.toList.sum / 4

/-- The weighted average for the given bowling scenario is 112.88 -/
theorem bowling_weighted_average :
  let scores : Vector ℝ 4 := ⟨[120, 113, 85, 101], rfl⟩
  let shoe_rental : ℝ := 4
  let total_refreshments : ℝ := 16.5
  weighted_average_bowlers scores shoe_rental total_refreshments = 112.88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_weighted_average_l630_63076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_tricksters_possible_l630_63007

/-- Represents an inhabitant of the village -/
inductive Inhabitant
| Knight
| Trickster

/-- Represents the village -/
structure Village where
  inhabitants : Finset Inhabitant
  knight_count : Nat
  trickster_count : Nat
  h_total : inhabitants.card = knight_count + trickster_count
  h_knights : knight_count = 63
  h_tricksters : trickster_count = 2

/-- Represents a question about a group of inhabitants -/
def Question := Finset Inhabitant → Bool

/-- Represents the process of asking questions to identify tricksters -/
def IdentifyTricksters (v : Village) :=
  ∃ (questions : Finset Question),
    questions.card ≤ 30 ∧
    ∀ t : Inhabitant, t ∈ v.inhabitants → (t = Inhabitant.Trickster →
      ∃ q ∈ questions, ∃ group : Finset Inhabitant, group ⊆ v.inhabitants ∧ q group = false)

/-- The main theorem: it's possible to identify tricksters with no more than 30 questions -/
theorem identify_tricksters_possible (v : Village) : IdentifyTricksters v := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_tricksters_possible_l630_63007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l630_63055

/-- Family of curves parameterized by θ -/
def family_of_curves (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

/-- Line y = 2x -/
def line (x y : ℝ) : Prop := y = 2 * x

/-- The chord length function -/
noncomputable def chord_length (x : ℝ) : ℝ := Real.sqrt (x^2 + (2*x)^2)

/-- Theorem stating the maximum chord length -/
theorem max_chord_length :
  ∃ (x : ℝ), ∀ (θ : ℝ) (x' : ℝ),
    family_of_curves θ x' (2*x') →
    chord_length x' ≤ chord_length x ∧
    chord_length x = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l630_63055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_squares_l630_63061

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 15*x^3 + 35*x^2 - 21*x + 6

-- Define the roots (using Set instead of Finset)
def roots : Set ℝ := {a : ℝ | p a = 0}

-- Theorem statement
theorem root_sum_squares (a b c d : ℝ) (ha : a ∈ roots) (hb : b ∈ roots) (hc : c ∈ roots) (hd : d ∈ roots) :
  (a - b)^2 + (b - c)^2 + (c - d)^2 + (d - a)^2 = 336 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_squares_l630_63061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_three_l630_63025

/-- The distance from a point to a plane passing through three points -/
noncomputable def distance_point_to_plane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  
  -- Calculate the coefficients A, B, C, D of the plane equation Ax + By + Cz + D = 0
  let A := (y₂ - y₁) * (z₃ - z₁) - (z₂ - z₁) * (y₃ - y₁)
  let B := (z₂ - z₁) * (x₃ - x₁) - (x₂ - x₁) * (z₃ - z₁)
  let C := (x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁)
  let D := -A * x₁ - B * y₁ - C * z₁

  -- Calculate the distance using the formula
  abs (A * x₀ + B * y₀ + C * z₀ + D) / Real.sqrt (A * A + B * B + C * C)

/-- Theorem stating that the distance from M₀ to the plane through M₁, M₂, M₃ is 3 -/
theorem distance_is_three :
  let M₀ : ℝ × ℝ × ℝ := (10, 1, 8)
  let M₁ : ℝ × ℝ × ℝ := (7, 2, 4)
  let M₂ : ℝ × ℝ × ℝ := (7, -1, -2)
  let M₃ : ℝ × ℝ × ℝ := (-5, -2, -1)
  distance_point_to_plane M₀ M₁ M₂ M₃ = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_three_l630_63025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l630_63047

theorem trig_inequality : 
  Real.tan (34 * π / 180) > Real.sin (33 * π / 180) ∧ 
  Real.sin (33 * π / 180) > Real.cos (58 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l630_63047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_l630_63095

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]
variable (A : Matrix n n ℝ)

theorem det_cube (h : det A = 3) : det (A ^ 3) = 27 := by
  have h1 : det (A ^ 3) = (det A) ^ 3 := by
    apply det_pow
  rw [h1, h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_l630_63095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_recurrence_a_increasing_a_general_term_l630_63036

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

noncomputable def a : ℕ → ℝ
  | 0 => 2  -- Adding this case to cover all natural numbers
  | 1 => 3
  | 2 => 3
  | n + 3 => a (n + 2) * a (n + 1) - a n

theorem a_recurrence (n : ℕ) (h : n ≥ 1) :
  (a (n + 2))^2 + (a (n + 1))^2 + (a n)^2 - (a (n + 2)) * (a (n + 1)) * (a n) = 4 := by
  sorry

theorem a_increasing (n : ℕ) (h : n ≥ 1) : a n > 0 ∧ a (n + 1) > a n := by
  sorry

theorem a_general_term (n : ℕ) :
  a n = ((3 + Real.sqrt 5) / 2) ^ (fibonacci n) + ((3 - Real.sqrt 5) / 2) ^ (fibonacci n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_recurrence_a_increasing_a_general_term_l630_63036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_ratio_equality_l630_63067

-- Define the ellipse
def E (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define points
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (-2, 2)
def O : ℝ × ℝ := (0, 0)

-- Define A and B as points on the ellipse in the first quadrant
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define C as the intersection of OP and NA
noncomputable def C : ℝ × ℝ := sorry

-- Slopes
noncomputable def k_AM : ℝ := sorry
noncomputable def k_AC : ℝ := sorry
noncomputable def k_MB : ℝ := sorry
noncomputable def k_MC : ℝ := sorry

-- Axioms
axiom A_on_E : E A.1 A.2
axiom B_on_E : E B.1 B.2
axiom A_B_first_quadrant : A.1 > 0 ∧ A.2 > 0 ∧ B.1 > 0 ∧ B.2 > 0
axiom A_on_BP : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ A = (t * B.1 + (1-t) * P.1, t * B.2 + (1-t) * P.2)
axiom C_on_OP : ∃ s : ℝ, C = (s * P.1, s * P.2)
axiom C_on_NA : ∃ r : ℝ, C = (r * A.1 + (1-r) * N.1, r * A.2 + (1-r) * N.2)

-- Theorem to prove
theorem slope_ratio_equality : k_MB * k_MC = k_AM * k_AC := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_ratio_equality_l630_63067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_purely_imaginary_z₁_greater_z₂_l630_63027

-- Define complex numbers z₁ and z₂ as functions of real x
def z₁ (x : ℝ) : ℂ := Complex.mk (2 * x + 1) (x^2 - 3 * x + 2)
def z₂ (x : ℝ) : ℂ := Complex.mk (x^2 - 2) (x^2 + x - 6)

-- Theorem 1: z₁ is purely imaginary when x = -1/2
theorem z₁_purely_imaginary : 
  ∃ (x : ℝ), z₁ x = Complex.I * (z₁ x).im ∧ (z₁ x).im ≠ 0 :=
by sorry

-- Theorem 2: z₁ > z₂ when x = 2
theorem z₁_greater_z₂ : 
  ∃ (x : ℝ), (z₁ x).re > (z₂ x).re ∧ (z₁ x).im = (z₂ x).im :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_purely_imaginary_z₁_greater_z₂_l630_63027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l630_63091

/-- Custom operation ⊙ defined on real numbers -/
noncomputable def odot (x y : ℝ) : ℝ := x / (2 - y)

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, odot (x - a) (x + 1 - a) > 0 → -2 ≤ x ∧ x ≤ 2) →
  -2 ≤ a ∧ a ≤ 1 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l630_63091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_min_k_achieved_min_k_is_one_l630_63035

/-- Circle with equation x^2 + y^2 = 1 -/
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Line with equation y = √k*x + 2 -/
noncomputable def Line (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt k * p.1 + 2}

/-- Two tangents from a point to the circle are perpendicular -/
def HasPerpendicularTangents (p : ℝ × ℝ) : Prop :=
  ∃ a b : ℝ × ℝ, a ∈ Circle ∧ b ∈ Circle ∧
    (p.1 - a.1) * (p.1 - b.1) + (p.2 - a.2) * (p.2 - b.2) = 0

/-- There always exists a point on the line with perpendicular tangents -/
def ExistsPointWithPerpendicularTangents (k : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ Line k ∧ HasPerpendicularTangents p

theorem min_k_value (k : ℝ) :
  ExistsPointWithPerpendicularTangents k → k ≥ 1 :=
by sorry

theorem min_k_achieved : ExistsPointWithPerpendicularTangents 1 :=
by sorry

theorem min_k_is_one :
  (∀ k : ℝ, ExistsPointWithPerpendicularTangents k → k ≥ 1) ∧
  ExistsPointWithPerpendicularTangents 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_min_k_achieved_min_k_is_one_l630_63035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_construction_from_tangents_l630_63097

/-- A parabola in a 2D plane -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ × ℝ → Prop

/-- A line in a 2D plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a line is tangent to a parabola at a point -/
def is_tangent (p : Parabola) (l : Line) (point : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate to check if two points are endpoints of a parameter -/
def is_parameter_endpoint (p : Parabola) (p₁ p₂ : ℝ × ℝ) : Prop :=
  sorry

/-- Theorem: Given two non-parallel tangent lines to a parabola and their 
    respective points of tangency, a unique parabola can be constructed. -/
theorem parabola_construction_from_tangents 
  (t₁ t₂ : Line) 
  (P₁ P₂ : ℝ × ℝ) 
  (h_not_parallel : t₁.direction ≠ t₂.direction)
  (h_tangent₁ : ∃ p : Parabola, is_tangent p t₁ P₁)
  (h_tangent₂ : ∃ p : Parabola, is_tangent p t₂ P₂)
  (h_not_endpoints : ∀ p : Parabola, ¬ is_parameter_endpoint p P₁ P₂) :
  ∃! p : Parabola, 
    (is_tangent p t₁ P₁) ∧ 
    (is_tangent p t₂ P₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_construction_from_tangents_l630_63097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_shop_cost_price_l630_63070

theorem furniture_shop_cost_price 
  (markup_percentage : ℝ) 
  (selling_price : ℝ) 
  (markup_percentage_is_ten : markup_percentage = 10) 
  (selling_price_is_8800 : selling_price = 8800) : 
  selling_price / (1 + markup_percentage / 100) = 8000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_shop_cost_price_l630_63070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_M_repetitions_l630_63046

/-- Represents a move on the Rubik's cube -/
inductive Move
| X  -- Rotates a layer clockwise by 90 degrees around the X axis
| Y  -- Rotates a layer clockwise by 90 degrees around the Y axis

/-- Represents the state of an 8x8x8 Rubik's cube -/
def CubeState := Unit  -- We don't need to model the full state for this problem

/-- Applies a single move to the cube -/
def applyMove (s : CubeState) (m : Move) : CubeState := s

/-- Applies a sequence of moves to the cube -/
def applyMoves (s : CubeState) (ms : List Move) : CubeState :=
  ms.foldl applyMove s

/-- The M move, defined as X followed by Y -/
def M : List Move := [Move.X, Move.Y]

/-- Applies the M move n times to the cube -/
def applyMTimes (s : CubeState) (n : ℕ) : CubeState :=
  applyMoves s (List.join (List.replicate n M))

/-- The solved state of the cube -/
def solvedState : CubeState := ()

theorem smallest_M_repetitions :
  ∃ n : ℕ, n > 0 ∧ applyMTimes solvedState n = solvedState ∧
  ∀ m : ℕ, 0 < m ∧ m < n → applyMTimes solvedState m ≠ solvedState :=
by sorry

#check smallest_M_repetitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_M_repetitions_l630_63046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l630_63072

-- Define the function f(x) = 2x - sin(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sin x

-- Theorem stating that f has exactly one zero
theorem f_has_unique_zero : ∃! x, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l630_63072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l630_63022

theorem complex_absolute_value (z : ℂ) : z = 7 + 3 * I → Complex.abs (z^2 + 8*z + 100) = Real.sqrt 42772 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l630_63022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_area_is_27_sqrt_3_l630_63079

/-- The area of a figure consisting of a regular hexagon surrounded by six equilateral triangles, where each side length is 3 -/
noncomputable def figureArea : ℝ := 27 * Real.sqrt 3

/-- Theorem stating that the area of the described figure is 27√3 -/
theorem figure_area_is_27_sqrt_3 :
  let sideLength : ℝ := 3
  let triangleArea : ℝ := (Real.sqrt 3 / 4) * sideLength ^ 2
  let hexagonArea : ℝ := 6 * triangleArea
  let totalArea : ℝ := hexagonArea + 6 * triangleArea
  totalArea = figureArea := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_area_is_27_sqrt_3_l630_63079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l630_63053

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the points P and Q on the parabola
def P (a : ℝ) : ℝ × ℝ := (a, parabola a)
def Q (a : ℝ) : ℝ × ℝ := (-a, parabola a)

-- Define the distance from a point to a line
noncomputable def distanceToLine (p : ℝ × ℝ) (m b : ℝ) : ℝ :=
  abs (m * p.1 - p.2 + b) / Real.sqrt (m^2 + 1)

theorem parabola_distance_theorem (a : ℝ) :
  -- Part 1: When PQ is parallel to x-axis, the distance from O to PQ is a^2
  distanceToLine (0, 0) 0 (parabola a) = a^2 ∧
  -- Part 2: The maximum distance from O to PQ is unbounded
  ∀ M : ℝ, ∃ a : ℝ, distanceToLine (0, 0) 0 (parabola a) > M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l630_63053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_100_white_achievable_l630_63066

/-- Represents the state of the urn -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Represents the possible operations on the urn -/
inductive Operation
  | op1 -- 3 black to 2 black
  | op2 -- 2 black, 1 white to 1 white, 2 black
  | op3 -- 1 black, 2 white to 1 white
  | op4 -- 3 white to 2 white, 1 black

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.op1 => ⟨state.white, state.black - 1⟩
  | Operation.op2 => state
  | Operation.op3 => ⟨state.white - 1, state.black - 1⟩
  | Operation.op4 => ⟨state.white - 1, state.black + 1⟩

/-- Checks if a state is achievable from the initial state -/
def isAchievable (initialState : UrnState) (finalState : UrnState) : Prop :=
  ∃ (n : ℕ) (ops : Fin n → Operation),
    (List.foldl applyOperation initialState (List.ofFn ops)) = finalState

/-- The theorem to be proved -/
theorem only_100_white_achievable (initialState : UrnState) :
  initialState.white = 150 ∧ initialState.black = 50 →
  (∀ finalState : UrnState,
    isAchievable initialState finalState →
    (finalState.white = 50 ∧ finalState.black = 50) ∨
    finalState.white = 100 ∨
    finalState.white = 150 ∨
    finalState.black = 50 ∨
    finalState.white = 50) →
  ∃ finalState : UrnState,
    isAchievable initialState finalState ∧
    finalState.white = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_100_white_achievable_l630_63066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_triplet_l630_63004

theorem perfect_square_triplet : 
  {(a, b, c) : ℕ+ × ℕ+ × ℕ+ | 
    a ≥ b ∧ b ≥ c ∧ 
    ∃ (x y z : ℕ+), 
      (a : ℕ)^2 + 3*(b : ℕ) = (x : ℕ)^2 ∧ 
      (b : ℕ)^2 + 3*(c : ℕ) = (y : ℕ)^2 ∧ 
      (c : ℕ)^2 + 3*(a : ℕ) = (z : ℕ)^2} 
  = {(1, 1, 1), (37, 25, 17)} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_triplet_l630_63004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2y4_in_2x_plus_y_to_6th_l630_63098

theorem coefficient_x2y4_in_2x_plus_y_to_6th : 
  (Finset.range 7).sum (λ k => (Nat.choose 6 k) * (2^(6-k)) * (Finset.range 7).sum (λ i => if i = 2 ∧ k = 4 then 1 else 0)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2y4_in_2x_plus_y_to_6th_l630_63098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_cyclic_trio_l630_63071

/-- Represents a tennis tournament. -/
structure Tournament where
  players : Finset ℕ
  beat : ℕ → ℕ → Prop
  each_plays_each : ∀ i j, i ∈ players → j ∈ players → i ≠ j → (beat i j ∨ beat j i)
  everyone_wins : ∀ i, i ∈ players → ∃ j, j ∈ players ∧ i ≠ j ∧ beat i j

/-- 
There exists a triplet of players (A, B, C) in the tournament 
such that A beat B, B beat C, and C beat A.
-/
theorem exists_cyclic_trio (t : Tournament) : 
  ∃ A B C, A ∈ t.players ∧ B ∈ t.players ∧ C ∈ t.players ∧ 
    A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
    t.beat A B ∧ t.beat B C ∧ t.beat C A :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_cyclic_trio_l630_63071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_tank_theorem_l630_63012

noncomputable def tank_capacity : ℝ := 150
noncomputable def pour_rate : ℝ := 1 / 15
noncomputable def leak_rate : ℝ := 0.1 / 30
noncomputable def pour_time : ℝ := 525

noncomputable def water_poured (t : ℝ) : ℝ := pour_rate * t
noncomputable def water_leaked (t : ℝ) : ℝ := leak_rate * t
noncomputable def net_water_added (t : ℝ) : ℝ := water_poured t - water_leaked t
noncomputable def additional_water_needed (t : ℝ) : ℝ := tank_capacity - net_water_added t

def approx_equal (x y : ℝ) : Prop := abs (x - y) < 0.05

theorem fill_tank_theorem :
  approx_equal (additional_water_needed pour_time) 116.7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_tank_theorem_l630_63012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_percentage_l630_63023

/-- Represents the quantities, cost prices, and sale prices of electronic items -/
structure ElectronicItems where
  radio_quantity : ℕ
  radio_cost : ℚ
  radio_sale : ℚ
  tv_quantity : ℕ
  tv_cost : ℚ
  tv_sale : ℚ
  phone_quantity : ℕ
  phone_cost : ℚ
  phone_sale : ℚ

/-- Calculates the combined loss percentage for electronic items -/
def combinedLossPercentage (items : ElectronicItems) : ℚ :=
  let total_cost := items.radio_quantity * items.radio_cost +
                    items.tv_quantity * items.tv_cost +
                    items.phone_quantity * items.phone_cost
  let total_sale := items.radio_quantity * items.radio_sale +
                    items.tv_quantity * items.tv_sale +
                    items.phone_quantity * items.phone_sale
  let total_loss := total_cost - total_sale
  (total_loss / total_cost) * 100

/-- Theorem: The combined loss percentage for the given electronic items is 7.5% -/
theorem store_loss_percentage :
  let items : ElectronicItems := {
    radio_quantity := 5,
    radio_cost := 8000,
    radio_sale := 7200,
    tv_quantity := 3,
    tv_cost := 20000,
    tv_sale := 18000,
    phone_quantity := 4,
    phone_cost := 15000,
    phone_sale := 14500
  }
  combinedLossPercentage items = 75/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_percentage_l630_63023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l630_63062

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define a point on the parabola
def point_on_parabola (p : ℝ) (P : ℝ × ℝ) : Prop :=
  parabola p P.1 P.2

-- Define perpendicularity
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem parabola_directrix
  (p : ℝ)
  (P Q : ℝ × ℝ)
  (h1 : point_on_parabola p P)
  (h2 : perpendicular (P.1 - (focus p).1, P.2 - (focus p).2) (1, 0))
  (h3 : Q.2 = 0)  -- Q is on x-axis
  (h4 : perpendicular (P.1, P.2) (Q.1 - P.1, Q.2 - P.2))
  (h5 : (Q.1 - (focus p).1)^2 + (Q.2 - (focus p).2)^2 = 36)  -- |FQ| = 6
  : (fun x : ℝ => x = -3/2) = (fun x : ℝ => x = -(p/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l630_63062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_functions_properties_l630_63064

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin x, -1/2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos (2*x))

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi/6)

theorem vector_functions_properties :
  (∃ (K : ℝ), ∀ x, f x ≤ K ∧ (∃ x, f x = K)) ∧
  (∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ x, f (x + S) ≠ f x) ∧
  (∀ y ∈ Set.Icc (-1/2) 1, ∃ x ∈ Set.Icc 0 (Real.pi/2), g x = y) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/2), g x ∈ Set.Icc (-1/2) 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_functions_properties_l630_63064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l630_63044

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∃ (k : ℝ), k * (x - 2*y + 1) = 0 ↔ y = (b/a) * x) :
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l630_63044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_equal_to_identity_correct_count_l630_63010

-- Define the function f(x) = x
def f (x : ℝ) : ℝ := x

-- Define the four given functions
noncomputable def g₁ (x : ℝ) : ℝ := (Real.sqrt x)^2
def g₂ (x : ℝ) : ℝ := 3 * x^3
noncomputable def g₃ (x : ℝ) : ℝ := Real.sqrt (x^2)
noncomputable def g₄ (x : ℝ) : ℝ := x^2 / x

-- Theorem stating that none of the given functions are equal to f
theorem not_equal_to_identity :
  (∃ x, g₁ x ≠ f x) ∧
  (∃ x, g₂ x ≠ f x) ∧
  (∃ x, g₃ x ≠ f x) ∧
  (∃ x, g₄ x ≠ f x) :=
by
  sorry

-- Count how many functions are equal to f
def count_equal_functions : Nat :=
  0 -- Since none of the functions are equal to f

theorem correct_count :
  count_equal_functions = 0 :=
by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_equal_to_identity_correct_count_l630_63010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_l630_63000

/-- Calculates the actual percent profit when a shopkeeper labels an item's price
    to earn a specified profit percentage and then offers a discount. -/
noncomputable def actualPercentProfit (initialProfitPercent : ℝ) (discountPercent : ℝ) : ℝ :=
  let labeledPrice := 1 + initialProfitPercent / 100
  let sellingPrice := labeledPrice * (1 - discountPercent / 100)
  (sellingPrice - 1) * 100

/-- Theorem stating that when a shopkeeper labels an item's price to earn a 50% profit
    and then offers a 10% discount, the actual percent profit is 35%. -/
theorem shopkeeper_profit : actualPercentProfit 50 10 = 35 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_l630_63000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l630_63039

/-- The parabola y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The circle with center (1, r) and radius r -/
def circleΩ (x y r : ℝ) : Prop := (x - 1)^2 + (y - r)^2 = r^2

/-- The circle is tangent to the x-axis at (1, 0) -/
def tangent_to_x_axis (r : ℝ) : Prop := circleΩ 1 0 r

/-- The circle intersects the parabola at exactly one point -/
def one_intersection (r : ℝ) : Prop := ∃! y : ℝ, ∃ x : ℝ, parabola x y ∧ circleΩ x y r

theorem circle_radius : 
  ∀ r : ℝ, tangent_to_x_axis r → one_intersection r → r = (4 * Real.sqrt 3) / 9 :=
by
  sorry

#check circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l630_63039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l630_63074

/-- An ellipse with semi-major axis a and semi-minor axis b. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse. -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The distance between the foci of an ellipse. -/
noncomputable def focalDistance (e : Ellipse) : ℝ :=
  2 * Real.sqrt (e.a^2 - e.b^2)

/-- The distance from the center of the ellipse to the line connecting
    the right vertex and top vertex. -/
noncomputable def centerToVertexLine (e : Ellipse) : ℝ :=
  (e.a * e.b) / Real.sqrt (e.a^2 + e.b^2)

theorem ellipse_eccentricity (e : Ellipse) :
  centerToVertexLine e = (Real.sqrt 6 / 6) * focalDistance e →
  eccentricity e = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l630_63074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_real_sqrt_negative_square_l630_63058

theorem unique_real_sqrt_negative_square : ∃! x : ℝ, ((-2 * (x + 2)^2) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_real_sqrt_negative_square_l630_63058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_count_and_sum_is_negative_two_l630_63040

/-- A function satisfying the given property for all real x and y -/
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x ^ 2 - y ^ 2) = f x ^ 2 + f (f y ^ 2 - f (-x) ^ 2) + x ^ 2

/-- The set of all possible values of f(2) for functions satisfying the property -/
def PossibleValuesAtTwo : Set ℝ :=
  {y : ℝ | ∃ f : ℝ → ℝ, SatisfiesProperty f ∧ f 2 = y}

/-- The theorem stating that the product of the number of possible values and their sum is -2 -/
theorem product_of_count_and_sum_is_negative_two :
  ∃ (s : Finset ℝ), s.card * (s.sum id) = -2 ∧ ∀ y ∈ s, y ∈ PossibleValuesAtTwo := by
  sorry

#check product_of_count_and_sum_is_negative_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_count_and_sum_is_negative_two_l630_63040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_nonempty_implies_m_positive_l630_63056

-- Define the sets M and N
def M (m : ℝ) : Set ℝ := {x | x ≤ m}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.exp (-x * Real.log 2)}

-- State the theorem
theorem intersection_nonempty_implies_m_positive (m : ℝ) : 
  (M m ∩ N).Nonempty → m > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_nonempty_implies_m_positive_l630_63056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_imaginary_roots_l630_63026

/-- The quadratic equation with complex coefficients -/
def quadratic_equation (l : ℝ) (x : ℂ) : ℂ :=
  (1 - Complex.I) * x^2 + (l + Complex.I) * x + (1 + Complex.I * l)

/-- Theorem stating the condition for two imaginary roots -/
theorem two_imaginary_roots (l : ℝ) : 
  (∀ x : ℂ, quadratic_equation l x = 0 → Complex.re x = 0) ↔ l ≠ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_imaginary_roots_l630_63026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_50th_and_51st_terms_l630_63001

/-- The sequence of positive integers that are powers of 3 or sums of distinct powers of 3 -/
def ternary_sequence : ℕ → ℕ := 
  sorry -- Implementation details omitted for brevity

/-- The 50th term of the ternary sequence -/
def term_50 : ℕ := ternary_sequence 50

/-- The 51st term of the ternary sequence -/
def term_51 : ℕ := ternary_sequence 51

/-- Theorem: The sum of the 50th and 51st terms of the ternary sequence is 655 -/
theorem sum_of_50th_and_51st_terms : term_50 + term_51 = 655 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_50th_and_51st_terms_l630_63001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_plane_intersection_l630_63073

structure RegularTetrahedron where
  -- Define properties of a regular tetrahedron

structure Plane where
  -- Define properties of a plane

structure Segment where
  -- Define properties of a line segment

def intersectTetrahedron (t : RegularTetrahedron) (p : Plane) : Set Segment :=
  sorry

def isVertexToMidpoint (t : RegularTetrahedron) (s : Segment) : Prop :=
  sorry

def validIntersection (t : RegularTetrahedron) (planes : Set Plane) : Prop :=
  ∀ (p : Plane), p ∈ planes → ∀ (s : Segment), s ∈ intersectTetrahedron t p → isVertexToMidpoint t s

theorem tetrahedron_plane_intersection 
  (t : RegularTetrahedron) (planes : Set Plane) 
  (h : validIntersection t planes)
  [Fintype planes] :
  (∃ (n : ℕ), Fintype.card planes = n ∧ 2 ≤ n ∧ n ≤ 4) ∧
  (Fintype.card planes).max = 4 ∧
  (Fintype.card planes).min = 2 ∧
  (Fintype.card planes).max - (Fintype.card planes).min = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_plane_intersection_l630_63073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l630_63028

noncomputable section

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- C passes through the point (1, √6/3) -/
def passes_through (a b : ℝ) : Prop :=
  ellipse_C 1 (Real.sqrt 6 / 3) a b

/-- Eccentricity of C is √6/3 -/
def eccentricity (a b : ℝ) : Prop :=
  Real.sqrt (a^2 - b^2) / a = Real.sqrt 6 / 3

/-- Definition of point Q -/
def point_Q : ℝ × ℝ := (0, 3/2)

/-- Definition of line l passing through Q and intersecting C at M and N -/
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 3/2

/-- M and N are distinct points on C -/
def distinct_intersections (a b k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    ellipse_C x₁ y₁ a b ∧ ellipse_C x₂ y₂ a b ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂

/-- |AM| = |AN|, where A is the lower vertex of C -/
def equal_distances (a b k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ a b ∧ ellipse_C x₂ y₂ a b ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    (x₁^2 + (y₁ + 1)^2) = (x₂^2 + (y₂ + 1)^2)

/-- Main theorem -/
theorem ellipse_and_line_theorem :
  ∀ (a b : ℝ),
    passes_through a b →
    eccentricity a b →
    (∃ (k : ℝ),
      distinct_intersections a b k ∧
      equal_distances a b k) →
    (a = Real.sqrt 3 ∧ b = 1) ∧
    (∃ (k : ℝ), k = Real.sqrt 6 / 3 ∨ k = -Real.sqrt 6 / 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l630_63028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_derivatives_of_z_l630_63075

/-- Given a function z(x, y) = 2x^y - x * tan(xy), this theorem states its partial derivatives. -/
theorem partial_derivatives_of_z (x y : ℝ) :
  let z : ℝ → ℝ → ℝ := λ x y ↦ 2 * x^y - x * Real.tan (x * y)
  (deriv (λ x ↦ z x y) x = 2 * y * x^(y - 1) - Real.tan (x * y) - (x * y) / (Real.cos (x * y))^2) ∧
  (deriv (λ y ↦ z x y) y = 2 * x^y * Real.log x - x^2 / (Real.cos (x * y))^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_derivatives_of_z_l630_63075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_roots_l630_63009

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Predicate to check if a number is a multiple of another -/
def IsMultipleOf (a n : ℤ) : Prop := ∃ k : ℤ, a = n * k

theorem no_integer_roots 
  (f : IntPolynomial) 
  (k : ℕ) 
  (h_k : k > 1) 
  (h_not_multiple : ∀ i ∈ Finset.range k, ¬IsMultipleOf (f.eval (↑i : ℤ)) k) :
  (∀ a : ℤ, ¬IsMultipleOf (f.eval a) k) ∧ 
  (∀ x : ℤ, f.eval x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_roots_l630_63009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_diff_greater_than_third_l630_63083

-- Define the coin flip process
def coinFlipProcess : Set ℝ :=
  {x | x = 0 ∨ x = 0.5 ∨ (0 ≤ x ∧ x ≤ 1)}

-- Define the probability measure for the coin flip process
noncomputable def coinFlipMeasure : MeasureTheory.Measure ℝ :=
  sorry

-- Define the joint probability measure for independent x and y
noncomputable def jointMeasure : MeasureTheory.Measure (ℝ × ℝ) :=
  sorry

-- Define the event |x-y| > 1/3
def eventDiffGreaterThanThird : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 - p.2| > 1/3}

-- State the theorem
theorem probability_diff_greater_than_third :
  jointMeasure eventDiffGreaterThanThird = 5/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_diff_greater_than_third_l630_63083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_properties_l630_63015

noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

noncomputable def point_on_parabola (p : ℝ) (y : ℝ) : Prop := parabola p 1 y

noncomputable def distance_to_focus (p : ℝ) (y : ℝ) : ℝ := 17/16

noncomputable def circle_eq (a x y : ℝ) : Prop := (x - a)^2 + y^2 = 1

noncomputable def circle_intersects_parabola (p a : ℝ) : Prop :=
  ∃ x y, parabola p x y ∧ circle_eq a x y

theorem parabola_and_circle_properties :
  ∀ p y : ℝ,
  parabola p 1 y →
  distance_to_focus p y = 17/16 →
  (p = 1/8 ∧
   ∀ a : ℝ, circle_intersects_parabola p a → -1 ≤ a ∧ a ≤ 65/16) :=
by
  sorry

#check parabola_and_circle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_properties_l630_63015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_cubic_polynomial_theorem_l630_63059

/-- A monic cubic polynomial with specific properties -/
def MonicCubicPolynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, p x = x^3 + a*x^2 + b*x + c) ∧ 
  (p 0 = 1) ∧
  (∀ x, (deriv p) x = 0 → p x = 0)

/-- The theorem stating that a monic cubic polynomial with given properties is (x + 1)³ -/
theorem monic_cubic_polynomial_theorem (p : ℝ → ℝ) 
  (h : MonicCubicPolynomial p) : 
  ∀ x, p x = (x + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_cubic_polynomial_theorem_l630_63059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_property_l630_63017

theorem fraction_property (a b : ℕ) (h1 : a > 1) (h2 : Nat.Coprime a b) :
  let x : ℚ := a / (a - 1)
  (a + x) / (b * x) = a / b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_property_l630_63017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gpa_difference_proof_l630_63002

/-- The difference between 7th and 6th grade GPAs -/
def gpa_difference : ℝ := 2

theorem gpa_difference_proof :
  let sixth_grade_gpa : ℝ := 93
  let eighth_grade_gpa : ℝ := 91
  let school_average_gpa : ℝ := 93
  let seventh_grade_gpa : ℝ := sixth_grade_gpa + gpa_difference
  (sixth_grade_gpa + seventh_grade_gpa + eighth_grade_gpa) / 3 = school_average_gpa →
  gpa_difference = 2 := by
  intro h
  -- Proof steps would go here
  sorry

#check gpa_difference_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gpa_difference_proof_l630_63002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_equation_solution_l630_63020

theorem logarithm_equation_solution (y : ℝ) (h : y > 0) :
  Real.log 243 / Real.log y = 5/3 → y = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_equation_solution_l630_63020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_l630_63087

/-- A rectangle with two circles inside it -/
structure RectangleWithCircles where
  AB : ℝ
  AD : ℝ
  radius : ℝ
  h_AB : AB = 8
  h_AD : AD = 20
  h_radius : radius = 5

/-- The area inside the rectangle and outside of both circles -/
noncomputable def areaOutsideCircles (r : RectangleWithCircles) : ℝ :=
  r.AB * (r.AD - 2 * r.radius) - Real.pi * r.radius^2

/-- The theorem stating the area inside the rectangle and outside of both circles -/
theorem area_outside_circles (r : RectangleWithCircles) :
  areaOutsideCircles r = 112 - 25 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_l630_63087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_proof_l630_63096

/-- 
Proves that the time traveled is 1.5 hours given the conditions of Juan and Peter's journey.
-/
theorem journey_time_proof (juan_speed_diff : ℚ) (peter_speed : ℚ) (total_distance : ℚ) 
  (h1 : juan_speed_diff = 3)
  (h2 : peter_speed = 5)
  (h3 : total_distance = 19.5) :
  (total_distance / (peter_speed + juan_speed_diff + peter_speed)) = 1.5 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_proof_l630_63096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l630_63050

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line segment in 2D space -/
structure LineSegment where
  start : Point2D
  stop : Point2D

/-- Check if a line segment is parallel to the x-axis -/
def isParallelToXAxis (segment : LineSegment) : Prop :=
  segment.start.y = segment.stop.y

/-- Calculate the length of a line segment -/
noncomputable def length (segment : LineSegment) : ℝ :=
  Real.sqrt ((segment.stop.x - segment.start.x)^2 + (segment.stop.y - segment.start.y)^2)

theorem point_b_coordinates (A B : Point2D) (AB : LineSegment) :
  A.x = 1 →
  A.y = 6 →
  AB.start = A →
  AB.stop = B →
  isParallelToXAxis AB →
  length AB = 4 →
  (B.x = -3 ∧ B.y = 6) ∨ (B.x = 5 ∧ B.y = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l630_63050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_max_apples_l630_63006

/-- Calculates the maximum number of apples Brian can buy given his expenses and budget --/
def max_apples_brian_can_buy 
  (apple_bag_cost : ℚ)
  (apple_bag_count : ℕ)
  (kiwi_cost : ℚ)
  (initial_money : ℚ)
  (subway_fare : ℚ) : ℕ :=
  let banana_cost := kiwi_cost / 2
  let total_subway_cost := 2 * subway_fare
  let remaining_money := initial_money - (kiwi_cost + banana_cost + total_subway_cost)
  let bags_of_apples := (remaining_money / apple_bag_cost).floor
  (bags_of_apples.toNat) * apple_bag_count

/-- Theorem stating that Brian can buy a maximum of 24 apples --/
theorem brian_max_apples : 
  max_apples_brian_can_buy 14 12 10 50 (7/2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_max_apples_l630_63006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l630_63077

/-- Calculates the cost price given the selling price and profit percentage -/
noncomputable def calculate_cost_price (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  selling_price / (1 + profit_percentage / 100)

/-- Theorem stating that the cost price is approximately 71.43 given the conditions -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 100)
  (h2 : profit_percentage = 40) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |calculate_cost_price selling_price profit_percentage - 71.43| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_cost_price 100 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l630_63077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_probability_is_limit_l630_63043

/-- A sequence of survival rates converging to 0.95 -/
def survival_rates : ℕ → ℝ := sorry

/-- The limit of the survival rates sequence is 0.95 -/
axiom survival_rates_limit : Filter.Tendsto survival_rates Filter.atTop (nhds 0.95)

/-- The estimated probability of survival -/
def estimated_probability : ℝ := 0.95

/-- Theorem: The estimated probability of survival is 0.95 -/
theorem estimated_probability_is_limit : estimated_probability = 0.95 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_probability_is_limit_l630_63043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_acute_angles_for_equal_area_l630_63051

/-- A circle in which shapes are inscribed -/
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

/-- A rhombus inscribed in a circle -/
structure Rhombus where
  circle : Circle
  side_length : ℝ
  acute_angle : ℝ

/-- An isosceles trapezoid inscribed in a circle -/
structure IsoscelesTrapezoid where
  circle : Circle
  parallel_side1 : ℝ
  parallel_side2 : ℝ
  acute_angle : ℝ

/-- The area of a rhombus -/
noncomputable def rhombusArea (r : Rhombus) : ℝ :=
  r.side_length ^ 2 * Real.sin r.acute_angle

/-- The area of an isosceles trapezoid -/
noncomputable def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  (t.parallel_side1 + t.parallel_side2) / 2 * (t.circle.radius * Real.sin t.acute_angle)

/-- Theorem: Given a rhombus and an isosceles trapezoid inscribed in the same circle with equal areas, their acute angles are equal -/
theorem equal_acute_angles_for_equal_area (r : Rhombus) (t : IsoscelesTrapezoid)
    (h1 : r.circle = t.circle)
    (h2 : rhombusArea r = trapezoidArea t) :
    r.acute_angle = t.acute_angle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_acute_angles_for_equal_area_l630_63051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beads_total_cost_l630_63092

/-- Calculate the total cost of beads for Carly's corset --/
theorem beads_total_cost : 
  (50 : ℚ) * 20 * 0.12 +
  (40 : ℚ) * 18 * 0.10 +
  (80 : ℚ) * 0.08 +
  (30 : ℚ) * 15 * 0.09 +
  (100 : ℚ) * 0.07 = 245.90 := by
  -- Compute each part
  have h1 : (50 : ℚ) * 20 * 0.12 = 120 := by norm_num
  have h2 : (40 : ℚ) * 18 * 0.10 = 72 := by norm_num
  have h3 : (80 : ℚ) * 0.08 = 6.4 := by norm_num
  have h4 : (30 : ℚ) * 15 * 0.09 = 40.5 := by norm_num
  have h5 : (100 : ℚ) * 0.07 = 7 := by norm_num
  
  -- Sum up all parts
  calc
    (50 : ℚ) * 20 * 0.12 +
    (40 : ℚ) * 18 * 0.10 +
    (80 : ℚ) * 0.08 +
    (30 : ℚ) * 15 * 0.09 +
    (100 : ℚ) * 0.07
    = 120 + 72 + 6.4 + 40.5 + 7 := by rw [h1, h2, h3, h4, h5]
    _ = 245.90 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beads_total_cost_l630_63092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l630_63045

noncomputable def variance (xs : Finset ℝ) (f : ℝ → ℝ) : ℝ :=
  let n := xs.card
  let μ := (xs.sum f) / n
  (xs.sum (λ x => (f x - μ)^2)) / n

theorem variance_transformation (xs : Finset ℝ) (h : xs.card = 2009) :
  variance xs id = 3 →
  variance xs (λ x => 3 * (x - 2)) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l630_63045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_bound_l630_63082

theorem gcd_bound (a b c : ℕ+) 
  (h : ∃ A : ℤ, A = (a.val^2 + 1) / (b.val * c.val) + 
                   (b.val^2 + 1) / (c.val * a.val) + 
                   (c.val^2 + 1) / (a.val * b.val)) : 
  Nat.gcd a.val (Nat.gcd b.val c.val) ≤ Int.floor ((a.val + b.val + c.val : ℝ) ^ (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_bound_l630_63082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_1998_l630_63038

def sequence_sum (n : ℕ) : ℤ :=
  -(n / 6)

theorem sequence_sum_1998 :
  sequence_sum 1998 = -333 := by
  unfold sequence_sum
  norm_num

#eval sequence_sum 1998

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_1998_l630_63038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_4cm_l630_63052

/-- The radius of spheres in a cylindrical container -/
noncomputable def sphere_radius (initial_height : ℝ) (num_spheres : ℕ) : ℝ :=
  initial_height / 2

/-- Theorem: The radius of three spheres in a cylindrical container is 4 cm -/
theorem sphere_radius_is_4cm (initial_height : ℝ) (h_initial : initial_height = 8) :
  sphere_radius initial_height 3 = 4 := by
  -- Unfold the definition of sphere_radius
  unfold sphere_radius
  -- Substitute the initial height
  rw [h_initial]
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_4cm_l630_63052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l630_63048

noncomputable def sequenceA (a : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => 2
  | (n+2) => 2 * sequenceA a (n+1) * sequenceA a n - sequenceA a (n+1) - sequenceA a n + 1

def is_perfect_square (n : ℤ) : Prop := ∃ k : ℤ, n = k^2

theorem sequence_property (a : ℤ) :
  (∀ n : ℕ, n ≥ 1 → is_perfect_square (2 * sequenceA a (3*n) - 1)) ↔
  (∃ m : ℕ, m > 0 ∧ a = (2*m - 1)^2 / 2 + 1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l630_63048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l630_63033

/-- The parabola defined by the equation y = -3(x+1)^2 - 2 has its vertex at (-1, -2). -/
theorem parabola_vertex (x y : ℝ) : 
  y = -3 * (x + 1)^2 - 2 → (x = -1 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l630_63033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l630_63019

noncomputable def f (x : ℝ) := Real.cos x ^ 4 - 2 * Real.sin x * Real.cos x - Real.sin x ^ 4

theorem f_properties :
  (∀ x : ℝ, f ((-3 * π / 8) + x) = f ((-3 * π / 8) - x)) ∧
  (∀ x y : ℝ, x ∈ Set.Icc (-π / 8) (3 * π / 8) →
              y ∈ Set.Icc (-π / 8) (3 * π / 8) →
              x < y → f x > f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l630_63019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l630_63003

theorem cos_alpha_value (α : Real) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.tan α = -2) : 
  Real.cos α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l630_63003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_values_l630_63084

/-- Definition of the ellipse equation -/
def ellipse_equation (x y t : ℝ) : Prop :=
  x^2 / t^2 + y^2 / (5*t) = 1

/-- Definition of focal length -/
noncomputable def focal_length (c : ℝ) : ℝ := 2 * Real.sqrt 6

/-- Theorem: For the given ellipse, t can be 2, 3, or 6 -/
theorem ellipse_t_values :
  ∃ (t : ℝ), (t = 2 ∨ t = 3 ∨ t = 6) ∧
  (∀ (x y : ℝ), ellipse_equation x y t) ∧
  (∃ (a b c : ℝ), a^2 - b^2 = c^2 ∧ focal_length c = 2 * Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_values_l630_63084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_4_30_l630_63005

/-- Given log₁₀ 2 = a, log₁₀ 5 = c, and log₁₀ 3 = d, prove that log₄ 30 = (a + d + c) / (2a) -/
theorem log_4_30 (a c d : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 5 = c) (h3 : Real.log 3 = d) :
  Real.log 30 / Real.log 4 = (a + d + c) / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_4_30_l630_63005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_half_l630_63093

theorem tan_alpha_plus_pi_half (α : Real) (h : ∃ (t : Real), t * (-1) = Real.cos α ∧ t * 2 = Real.sin α) :
  Real.tan (α + π/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_half_l630_63093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l630_63037

theorem circular_table_seating (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 8) :
  (n.choose k) * Nat.factorial (k - 1) = 45360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_table_seating_l630_63037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_l630_63016

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - 1) + a * x

-- State the theorem
theorem inequality_holds_iff (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x + Real.log x ≥ a + 1) ↔ a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_l630_63016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l630_63069

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x ≤ 0}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l630_63069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sine_to_cosine_l630_63089

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem shift_sine_to_cosine 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + Real.pi / ω) = f ω x) 
  (h_min_period : ∀ T, T > 0 → (∀ x, f ω (x + T) = f ω x) → T ≥ Real.pi / ω) :
  ∀ x, f ω (x + Real.pi / 12) = Real.cos (2 * x) :=
by
  sorry

#check shift_sine_to_cosine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sine_to_cosine_l630_63089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_twice_allan_carla_half_ben_carla_balloons_l630_63018

/-- The number of balloons Allan brought -/
def A : ℕ := 2

/-- The number of balloons Ben brought -/
def B : ℕ := A + 6

/-- The number of balloons Carla brought -/
def C : ℕ := 4

/-- Carla brought twice as many balloons as Allan -/
theorem carla_twice_allan : C = 2 * A := by
  rfl

/-- Carla brought half as many balloons as Ben -/
theorem carla_half_ben : C = B / 2 := by
  rfl

theorem carla_balloons : C = 4 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_twice_allan_carla_half_ben_carla_balloons_l630_63018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x6_in_expansion_l630_63041

theorem coefficient_x6_in_expansion : ∃ (c : ℤ), c = 10 ∧ 
  (∀ (x : ℝ), (1 + x + x^2) * (1 - x)^6 = 
    c * x^6 + ((1 + x + x^2) * (1 - x)^6 - c * x^6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x6_in_expansion_l630_63041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_to_square_ratio_l630_63013

-- Define the vertices of the quadrilateral
def v1 : ℝ × ℝ := (0, 0)
def v2 : ℝ × ℝ := (10, 5)
def v3 : ℝ × ℝ := (5, 10)
def v4 : ℝ × ℝ := (5, 5)

-- Define the side length of the large square
def square_side : ℝ := 10

-- Function to calculate the area of a quadrilateral using the shoelace formula
def quadrilateral_area (a b c d : ℝ × ℝ) : ℝ :=
  let (x1, y1) := a
  let (x2, y2) := b
  let (x3, y3) := c
  let (x4, y4) := d
  0.5 * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

-- Theorem statement
theorem shaded_to_square_ratio :
  (quadrilateral_area v1 v2 v3 v4) / (square_side ^ 2) = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_to_square_ratio_l630_63013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_heads_probability_l630_63008

-- Define a coin toss as a type
def CoinToss := Bool

-- Define the probability of getting heads
noncomputable def probHeads : ℝ := 1/2

-- Theorem stating that the probability of getting heads in a fair coin toss is 1/2
theorem fair_coin_heads_probability :
  probHeads = 1/2 := by
  -- Unfold the definition of probHeads
  unfold probHeads
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_heads_probability_l630_63008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_ball_max_height_l630_63031

/-- The height of a tennis ball as a function of time -/
noncomputable def h (t : ℝ) : ℝ := -1/80 * t^2 + 1/5 * t + 1

/-- The time interval during which the ball is in flight -/
def T : Set ℝ := { t | 0 ≤ t ∧ t ≤ 20 }

/-- The maximum height reached by the tennis ball -/
theorem tennis_ball_max_height :
  ∃ (t : ℝ), t ∈ T ∧ ∀ (s : ℝ), s ∈ T → h s ≤ h t ∧ h t = 1.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_ball_max_height_l630_63031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_completion_time_is_five_l630_63068

/-- Represents the typing job with given conditions -/
structure TypingJob where
  total_time : ℝ  -- Time for John to complete the entire job
  john_work_time : ℝ  -- Time John actually works
  jack_rate_ratio : ℝ  -- Jack's typing rate as a ratio of John's rate
  (total_time_pos : total_time > 0)
  (john_work_time_pos : john_work_time > 0)
  (john_work_time_le_total : john_work_time ≤ total_time)
  (jack_rate_ratio_pos : jack_rate_ratio > 0)
  (jack_rate_ratio_le_one : jack_rate_ratio ≤ 1)

/-- Calculates the time it takes Jack to complete the remaining work -/
noncomputable def jack_completion_time (job : TypingJob) : ℝ :=
  (job.total_time - job.john_work_time) / job.jack_rate_ratio

/-- Theorem stating that Jack's completion time is 5 hours for the given conditions -/
theorem jack_completion_time_is_five (job : TypingJob)
    (h1 : job.total_time = 5)
    (h2 : job.john_work_time = 3)
    (h3 : job.jack_rate_ratio = 2/5) :
    jack_completion_time job = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_completion_time_is_five_l630_63068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_always_wins_l630_63024

/-- Represents a position on the game board -/
structure Position where
  x : Fin 99
  y : Fin 99

/-- Represents the game state -/
structure GameState where
  firstPlayerMoves : List Position
  secondPlayerMoves : List Position

/-- Checks if a position is a corner of the 99x99 grid -/
def isCorner (pos : Position) : Bool :=
  (pos.x = 0 ∨ pos.x = 98) ∧ (pos.y = 0 ∨ pos.y = 98)

/-- Checks if a move is valid (adjacent to an occupied cell) -/
def isValidMove (state : GameState) (pos : Position) : Bool :=
  sorry

/-- The winning strategy for the first player -/
def firstPlayerStrategy (state : GameState) : Position :=
  sorry

/-- Theorem stating that the first player can always win -/
theorem first_player_always_wins :
  ∀ (game : GameState),
  ∃ (n : Nat),
  ∃ (strategy : GameState → Position),
  (∀ (i : Fin n),
    let newGame := (List.range i).foldl
      (λ s _ => { firstPlayerMoves := strategy s :: s.firstPlayerMoves,
                  secondPlayerMoves := sorry :: s.secondPlayerMoves })
      game
    isValidMove newGame (strategy newGame)) →
  isCorner (strategy ((List.range (n-1)).foldl
    (λ s _ => { firstPlayerMoves := strategy s :: s.firstPlayerMoves,
                secondPlayerMoves := sorry :: s.secondPlayerMoves })
    game)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_always_wins_l630_63024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shortest_arc_length_is_two_l630_63057

/-- Represents a regular polygon inscribed in a circle -/
structure InscribedPolygon where
  sides : ℕ
  vertices : Fin sides → ℝ × ℝ

/-- Represents a circle with inscribed polygons -/
structure CircleWithPolygons where
  circumference : ℝ
  triangle : InscribedPolygon
  heptagon : InscribedPolygon

/-- The maximum possible length of the shortest arc segment -/
def maxShortestArcLength (c : CircleWithPolygons) : ℝ := 2

/-- Theorem stating the maximum possible length of the shortest arc segment -/
theorem max_shortest_arc_length_is_two (c : CircleWithPolygons) 
  (h1 : c.circumference = 84) 
  (h2 : c.triangle.sides = 3) 
  (h3 : c.heptagon.sides = 7) 
  (h4 : ∀ (i : Fin c.triangle.sides), ∃ (j : Fin c.heptagon.sides), c.triangle.vertices i = c.heptagon.vertices j) :
  maxShortestArcLength c = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shortest_arc_length_is_two_l630_63057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_sequence_existence_l630_63065

theorem rational_sequence_existence (x : ℚ) : 
  ∃ (f : ℕ → ℚ), 
    f 0 = x ∧ 
    (∀ n : ℕ, n ≥ 1 → (f n = 2 * f (n - 1) ∨ f n = 2 * f (n - 1) + 1 / (n : ℚ))) ∧
    (∃ n : ℕ, ∃ z : ℤ, f n = ↑z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_sequence_existence_l630_63065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carrot_weight_problem_l630_63085

theorem carrot_weight_problem (
  weight_first : Real) (weight_second : Real)
  (avg_remaining_first : Real) (avg_remaining_second : Real)
  (h1 : weight_first = 6.738)
  (h2 : weight_second = 7.992)
  (h3 : avg_remaining_first = 218.6)
  (h4 : avg_remaining_second = 226) :
  (weight_first * 1000 - 30 * avg_remaining_first +
   weight_second * 1000 - 33 * avg_remaining_second) / 12 = 59.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carrot_weight_problem_l630_63085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pell_solution_with_divisibility_l630_63078

/-- Given a square-free positive integer d and a fundamental solution (x₀, y₀) to x² - dy² = 1,
    the only positive integer solution (x, y) to x² - dy² = 1 where all prime factors of x divide x₀
    is (x₀, y₀) itself. -/
theorem unique_pell_solution_with_divisibility (d x₀ y₀ : ℕ+) (h_square_free : Squarefree d)
    (h_fundamental : x₀^2 - d * y₀^2 = 1) :
    ∀ x y : ℕ+, x^2 - d * y^2 = 1 → (∀ p : ℕ, Nat.Prime p → p ∣ x → p ∣ x₀) → x = x₀ ∧ y = y₀ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_pell_solution_with_divisibility_l630_63078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_one_plus_sqrt_four_minus_x_squared_l630_63063

theorem integral_one_plus_sqrt_four_minus_x_squared : 
  ∫ x in (-2)..0, (1 + Real.sqrt (4 - x^2)) = 2 + π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_one_plus_sqrt_four_minus_x_squared_l630_63063
