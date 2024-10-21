import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_equivalence_l1350_135040

structure Plane where

structure Line where

noncomputable def intersect (α β : Plane) : Line :=
  sorry

def parallel_line_plane (l : Line) (p : Plane) : Prop :=
  sorry

def parallel_lines (l1 l2 : Line) : Prop :=
  sorry

theorem parallel_equivalence 
  (α β : Plane) 
  (m n : Line) 
  (h_diff_planes : α ≠ β) 
  (h_diff_lines : m ≠ n) 
  (h_intersect : intersect α β = m) 
  (h_not_in_α : ¬ parallel_line_plane n α)
  (h_not_in_β : ¬ parallel_line_plane n β) :
  parallel_lines n m ↔ (parallel_line_plane n α ∧ parallel_line_plane n β) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_equivalence_l1350_135040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1350_135076

/-- The time (in seconds) for a train to pass a person moving in the opposite direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) : ℝ :=
  train_length / ((train_speed + person_speed) * (1000 / 3600))

/-- Theorem stating that the time for a 500 m long train traveling at 120 km/hr to pass
    a man moving at 10 km/hr in the opposite direction is approximately 138.5 seconds -/
theorem train_passing_time_approx :
  ∃ ε > 0, |train_passing_time 500 120 10 - 138.5| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1350_135076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_distance_l1350_135051

/-- Represents Jana's walking rate in miles per minute -/
def walking_rate : ℚ := 1 / 15

/-- Represents the time Jana walks in minutes -/
def walking_time : ℚ := 20

/-- Represents the distance Jana walks in miles -/
def distance : ℚ := walking_rate * walking_time

/-- Rounds a rational number to the nearest tenth -/
def round_to_tenth (q : ℚ) : ℚ := 
  (q * 10).floor / 10 + if (q * 10 - (q * 10).floor ≥ 1/2) then 1/10 else 0

theorem jana_walking_distance :
  round_to_tenth distance = 13/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_distance_l1350_135051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_l1350_135082

/-- The circle with equation x^2 + y^2 = 1 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The curve with equation x + 3y^2 = 4 -/
def curve_equation (x y : ℝ) : Prop := x + 3 * y^2 = 4

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin (center of the circle) -/
def O : Point := ⟨0, 0⟩

/-- Two points are perpendicular with respect to the origin if their dot product is zero -/
def perpendicular (p q : Point) : Prop :=
  p.x * q.x + p.y * q.y = 0

theorem intersection_perpendicular :
  ∀ (A B : Point),
    curve_equation A.x A.y →
    curve_equation B.x B.y →
    A ≠ B →
    perpendicular A B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_l1350_135082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1350_135077

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (oplus 1 x) * x - (oplus 2 x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 2 ∧
  f x = 6 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-2) 2 → f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1350_135077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1350_135011

noncomputable def ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

theorem equation_solutions :
  let f : ℂ → ℂ := λ x => (x^4 + 4*x^3*(Real.sqrt 3 : ℂ) + 12*x^2 + 8*(Real.sqrt 3 : ℂ)*x + 4) + (x - (Real.sqrt 3 : ℂ))
  {x : ℂ | f x = 0} = {(Real.sqrt 3 : ℂ), -1 + (Real.sqrt 3 : ℂ), ω - 1 + (Real.sqrt 3 : ℂ), ω^2 - 1 + (Real.sqrt 3 : ℂ)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1350_135011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_range_l1350_135090

-- Define the line
def line (m n x y : ℝ) : Prop := m * x + 2 * n * y - 4 = 0

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 4 = 0

-- Define the bisection property
def bisects (m n : ℝ) : Prop := ∀ x y, line m n x y → my_circle x y → 2 * m + 2 * n - 4 = 0

-- State the theorem
theorem mn_range (m n : ℝ) (h1 : m ≠ n) (h2 : bisects m n) : 
  m * n ∈ Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_range_l1350_135090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l1350_135047

theorem partition_theorem (m : ℕ+) :
  ∃ (k : ℕ+) (A : ℕ+ → Set ℕ+),
    (∀ i j, i ≠ j → A i ∩ A j = ∅) ∧
    (⋃ i, A i) = Set.univ ∧
    (∀ i, ∀ (a b c d : ℕ+), a ∈ A i → b ∈ A i → c ∈ A i → d ∈ A i → a * b - c * d ≠ m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l1350_135047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smoothie_mix_in_original_packet_l1350_135097

/-- The amount of smoothie mix in ounces required for one smoothie -/
noncomputable def smoothie_mix_per_smoothie (total_mix_packets : ℕ) (mix_per_packet : ℚ) (num_smoothies : ℕ) : ℚ :=
  (total_mix_packets : ℚ) * mix_per_packet / (num_smoothies : ℚ)

/-- The amount of smoothie mix in the original packet -/
theorem smoothie_mix_in_original_packet : 
  smoothie_mix_per_smoothie 180 2 150 = 12/5 := by sorry

#eval (12 : ℚ) / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smoothie_mix_in_original_packet_l1350_135097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_obtuse_triangle_probability_l1350_135060

-- Define a circle
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define a function to choose n points uniformly at random on the circle
noncomputable def chooseRandomPoints (n : ℕ) : Set (ℝ × ℝ) := sorry

-- Define a function to check if three points form an obtuse triangle with the center
def isObtuseTriangle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

-- Define the probability function
noncomputable def probability (event : Set (ℝ × ℝ) → Prop) (sample : Set (ℝ × ℝ)) : ℝ := sorry

-- Main theorem
theorem no_obtuse_triangle_probability :
  probability (λ points => ∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → ¬isObtuseTriangle p1 p2 p3) (chooseRandomPoints 4) = π^2 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_obtuse_triangle_probability_l1350_135060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_walking_time_l1350_135075

/-- 
Given a man who takes 24 minutes more to cover a distance when walking at 70% of his usual speed,
prove that his usual time to cover this distance is approximately 56 minutes.
-/
theorem usual_walking_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_time > 0)
  (h2 : usual_speed > 0)
  (h3 : usual_speed * usual_time = 0.7 * usual_speed * (usual_time + 24)) : 
  ∃ ε > 0, |usual_time - 56| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_walking_time_l1350_135075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paths_to_annas_apartment_l1350_135079

/-- Represents a point on a 2D grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points --/
def numPaths (start : Point) (finish : Point) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- Calculates the number of paths that pass through an intermediate point --/
def numPathsThrough (start : Point) (mid : Point) (finish : Point) : ℕ :=
  (numPaths start mid) * (numPaths mid finish)

/-- The main theorem to prove --/
theorem paths_to_annas_apartment :
  let start := Point.mk 0 0
  let finish := Point.mk 4 3
  let construction := Point.mk 2 1
  (numPaths start finish) - (numPathsThrough start construction finish) = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paths_to_annas_apartment_l1350_135079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_magnitude_l1350_135018

theorem complex_sum_magnitude : Complex.abs (3 - 5*Complex.I) + Complex.abs (3 + 5*Complex.I) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_magnitude_l1350_135018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_purchase_total_l1350_135093

theorem market_purchase_total (orange_price juice_price honey_price plant_price_pair : ℚ)
  (orange_quantity juice_quantity honey_quantity plant_quantity : ℕ) :
  orange_price = 4.5 →
  juice_price = 0.5 →
  honey_price = 5 →
  plant_price_pair = 18 →
  orange_quantity = 3 →
  juice_quantity = 7 →
  honey_quantity = 3 →
  plant_quantity = 4 →
  orange_price * orange_quantity +
  juice_price * juice_quantity +
  honey_price * honey_quantity +
  (plant_price_pair / 2) * plant_quantity = 68 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_purchase_total_l1350_135093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_average_l1350_135059

-- Define the starting integer and the average
variable (a : ℤ)
variable (b : ℚ)

-- Define the conditions
axiom a_positive : a > 0
axiom b_def : b = (7*a + 28) / 7

-- Define the theorem
theorem consecutive_integer_average : 
  let new_sequence := List.range 7
  (new_sequence.map (λ i => b + i)).sum / 7 = a + 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_average_l1350_135059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_to_left_evaluation_l1350_135073

/-- Evaluates an expression from right to left -/
noncomputable def evaluateRightToLeft (a b c d e : ℝ) : ℝ := a / (b - c * (d + e))

/-- The conventional representation of the expression -/
noncomputable def conventionalNotation (a b c d e : ℝ) : ℝ := a / (b - c * (d + e))

theorem right_to_left_evaluation (a b c d e : ℝ) :
  evaluateRightToLeft a b c d e = conventionalNotation a b c d e := by
  -- Unfold the definitions
  unfold evaluateRightToLeft conventionalNotation
  -- The expressions are identical, so we're done
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_to_left_evaluation_l1350_135073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_parametric_curve_l1350_135087

theorem max_value_of_parametric_curve : ∃ (M : ℝ),
  (∀ θ : ℝ, 
    Real.sqrt ((2 + Real.cos θ - 5)^2 + (Real.sin θ + 4)^2) ≤ M) ∧
  (∃ θ : ℝ, 
    Real.sqrt ((2 + Real.cos θ - 5)^2 + (Real.sin θ + 4)^2) = M) ∧
  M = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_parametric_curve_l1350_135087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_m_leq_neg_six_l1350_135069

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - 3*x + Real.log x

-- Define the theorem
theorem inequality_holds_iff_m_leq_neg_six :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 →
    x₁ * x₂ * (f x₁ - f x₂) - m * (x₁ - x₂) > 0) ↔ m ≤ -6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_m_leq_neg_six_l1350_135069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1350_135070

noncomputable def S (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

noncomputable def a (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  8 * (a a₁ q 2) - (a a₁ q 3) = 0 →
  (S a₁ q 4) / (S a₁ q 2) = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1350_135070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_midpoint_zero_l1350_135024

open Real
open BigOperators

noncomputable def f (x : ℝ) := sin x + tan x

theorem arithmetic_sequence_midpoint_zero
  (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_range : ∀ n, 1 ≤ n → n ≤ 31 → -π/2 < a n ∧ a n < π/2)
  (h_diff : a 2 - a 1 ≠ 0)
  (h_sum_zero : ∑ n in Finset.range 31, f (a (n + 1)) = 0) :
  f (a 16) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_midpoint_zero_l1350_135024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_c_value_l1350_135068

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes (m1 m2 : ℝ) : m1 = m2 ↔ (∀ x y : ℝ, m1 * x + y = m2 * x + y)

/-- The slope of the first line -/
def slope1 : ℝ := 12

/-- The slope of the second line -/
def slope2 (c : ℝ) : ℝ := 4 * c + 2

/-- The theorem stating that c = 5/2 when the lines are parallel -/
theorem parallel_lines_c_value :
  ∃ c : ℝ, (∀ x : ℝ, slope1 * x + 3 = slope2 c * x - 5) ↔ c = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_c_value_l1350_135068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_l1350_135043

noncomputable def numbers : List ℝ := [1200, 1300, 1510, 1520, 1530, 1200]

noncomputable def average (lst : List ℝ) : ℝ :=
  (lst.sum) / (lst.length : ℝ)

theorem correct_average :
  average numbers = 1460 ∧ average numbers ≠ 1380 := by
  -- Split the conjunction
  apply And.intro
  -- Prove the first part: average numbers = 1460
  · sorry
  -- Prove the second part: average numbers ≠ 1380
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_l1350_135043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_segment_l1350_135037

-- Define the fixed points F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points M satisfying the condition
def locus : Set (ℝ × ℝ) :=
  {M | distance M F₁ + distance M F₂ = 2}

-- Theorem statement
theorem locus_is_line_segment :
  ∃ (a b : ℝ × ℝ), locus = {M | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • a + t • b} := by
  sorry

#check locus_is_line_segment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_line_segment_l1350_135037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_sum_problem_l1350_135009

theorem vasya_sum_problem (n : ℕ) (numbers : Fin n → ℤ) : 
  (∀ i j : Fin n, i ≠ j → numbers i ≠ numbers j) →  -- numbers are distinct
  (∀ i : Fin n, (Finset.filter (fun j ↦ i ≠ j) Finset.univ).card = 13) →  -- each number appears in 13 sums
  ¬(14 * (Finset.univ.sum numbers) = 533) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_sum_problem_l1350_135009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1350_135003

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 4 * Real.sqrt 3 ∧ b = 4 ∧ Real.cos A = -1/2

theorem triangle_ABC_properties (a b c A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : 
  B = π/6 ∧ 
  ∀ (x : ℝ), 
    (∃ (k : ℤ), -π/3 + ↑k * π ≤ x ∧ x ≤ π/6 + ↑k * π) → 
    StrictMono (λ x => Real.cos (2*x) + c/2 * Real.sin (x^2 + B)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1350_135003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_1_inequality_solution_2_l1350_135000

-- Problem 1
theorem inequality_solution_1 (x : ℝ) :
  (2 * x + 1) / (x - 2) > 1 ↔ x ∈ Set.Ioi 2 ∪ Set.Iio (-3) :=
sorry

-- Problem 2
theorem inequality_solution_2 (x a : ℝ) :
  x^2 - 6*a*x + 5*a^2 ≤ 0 ↔
  (a > 0 ∧ x ∈ Set.Icc a (5*a)) ∨
  (a = 0 ∧ x = 0) ∨
  (a < 0 ∧ x ∈ Set.Icc (5*a) a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_1_inequality_solution_2_l1350_135000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l1350_135074

theorem roots_of_equation : 
  let f : ℝ → ℝ := fun x ↦ 4*x^4 - 21*x^3 + 34*x^2 - 21*x + 4
  ∀ x : ℝ, x ∈ ({4, 1/4, 1} : Set ℝ) → f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l1350_135074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_satisfies_conditions_l1350_135013

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

noncomputable def perimeter (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  distance x1 y1 x2 y2 + distance x2 y2 x3 y3 + distance x3 y3 x1 y1

noncomputable def area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  let a := distance x1 y1 x2 y2
  let b := distance x2 y2 x3 y3
  let c := distance x3 y3 x1 y1
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem no_triangle_satisfies_conditions :
  ∀ (x y : ℝ), 
    distance 0 0 12 0 = 12 →
    perimeter 0 0 12 0 x y ≠ 60 ∨ area 0 0 12 0 x y ≠ 150 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_satisfies_conditions_l1350_135013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_diagonal_ratio_l1350_135080

/-- The ratio of a diagonal to a side length in a regular pentagon -/
noncomputable def diagonal_to_side_ratio : ℝ := Real.sqrt ((6 + Real.sqrt 5) / 2)

/-- Predicate to check if a length is a diagonal of a regular pentagon with given side length -/
def is_diagonal_of_regular_pentagon (side diagonal : ℝ) : Prop := sorry

/-- Theorem stating the ratio of a diagonal to a side length in a regular pentagon -/
theorem regular_pentagon_diagonal_ratio :
  ∀ (side diagonal : ℝ), side > 0 → 
  is_diagonal_of_regular_pentagon side diagonal →
  diagonal / side = diagonal_to_side_ratio := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_diagonal_ratio_l1350_135080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_size_l1350_135091

theorem partnership_size : ∃ x : ℕ, 
  (5 * x + 45 = 7 * x + 3) ∧ 
  (x = 21) := by
  -- Define the number of people in the partnership
  let x : ℕ := 21

  -- Prove that x satisfies the equation
  have h1 : 5 * x + 45 = 7 * x + 3 := by
    -- Arithmetic calculation
    norm_num

  -- Prove that x is indeed 21
  have h2 : x = 21 := rfl

  -- Combine the proofs
  exact ⟨x, h1, h2⟩

  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_size_l1350_135091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l1350_135002

theorem system_solution (l x y : ℝ) :
  4 * x - 3 * y = l →
  5 * x + 6 * y = 2 * l + 3 →
  x = (4 * l + 3) / 13 ∧ y = (l + 4) / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l1350_135002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_proof_hyperbola_proof_l1350_135030

-- Part I: Ellipse
def ellipse_equation (x y : ℝ) := x^2 / 4 + y^2 / 2 = 1

theorem ellipse_proof :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (∀ (x y : ℝ), x^2 / a + y^2 / b = 1) ∧
    (Real.sqrt 2)^2 / a + 1^2 / b = 1 ∧
    (-1)^2 / a + (Real.sqrt 6 / 2)^2 / b = 1) →
  (∀ (x y : ℝ), ellipse_equation x y) :=
by sorry

-- Part II: Hyperbola
def original_hyperbola (x y : ℝ) := y^2 / 4 - x^2 / 3 = 1
def new_hyperbola (x y : ℝ) := x^2 / 6 - y^2 / 8 = 1

theorem hyperbola_proof :
  (∃ (k : ℝ), 
    (∀ (x y : ℝ), y^2 / 4 - x^2 / 3 = k) ∧
    3^2 / 3 - (-2)^2 / 4 = k) →
  (∀ (x y : ℝ), new_hyperbola x y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_proof_hyperbola_proof_l1350_135030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_plus_fifty_l1350_135007

theorem square_difference_plus_fifty : |(105 : ℝ)^2 - (103 : ℝ)^2| + 50 = 466 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_plus_fifty_l1350_135007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1350_135008

noncomputable section

def A : Set ℝ := {x | 2 * x^2 - 3*x - 2 ≤ 0}

def C (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a ≤ 0}

theorem problem_solution (a : ℝ) :
  (∃ x, x ∈ C a) ∧ 
  (∀ x, x ∈ C a → x ∈ A) ∧ 
  (∃ x, x ∈ A ∧ x ∉ C a) →
  (A = {x : ℝ | -1/2 ≤ x ∧ x ≤ 2}) ∧
  (a ∈ Set.Icc (-1/8 : ℝ) 0 ∪ Set.Icc 1 (4/3 : ℝ)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1350_135008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1350_135088

theorem hyperbola_eccentricity_range (a b c : ℝ) (h1 : a > 2) (h2 : b > 0) (h3 : c > 0) :
  let A : ℝ × ℝ := (a, 0)
  let B : ℝ × ℝ := (0, b)
  let d1 := abs (b * 2 + a * 0 - a * b) / Real.sqrt (b^2 + a^2)
  let d2 := abs (b * (-2) + a * 0 - a * b) / Real.sqrt (b^2 + a^2)
  let e := c / a
  c^2 = a^2 + b^2 →
  d1 + d2 ≥ 4/5 * c →
  Real.sqrt 5 / 2 ≤ e ∧ e ≤ Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1350_135088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blueberry_sales_analysis_l1350_135055

-- Define the piecewise function for the selling price
def selling_price (x : ℕ) (m n : ℚ) : ℚ :=
  if 1 ≤ x ∧ x < 20 then m * x - 76 * m else n

-- Define the daily sales volume
def sales_volume (x : ℕ) : ℕ := 20 + 4 * (x - 1)

-- Define the daily profit function
def daily_profit (x : ℕ) (m n : ℚ) : ℚ :=
  (selling_price x m n - 18) * sales_volume x

-- Main theorem
theorem blueberry_sales_analysis 
  (m n : ℚ) -- Parameters of the selling price function
  (h1 : selling_price 12 m n = 32) -- Price on day 12
  (h2 : selling_price 26 m n = 25) -- Price on day 26
  : 
  -- 1. Correct values of m and n
  (m = -1/2 ∧ n = 25) ∧ 
  -- 2. Maximum profit occurs on day 18 and equals 968
  (∀ x : ℕ, 1 ≤ x → x ≤ 30 → daily_profit x m n ≤ daily_profit 18 m n) ∧
  (daily_profit 18 m n = 968) ∧
  -- 3. Exactly 12 days have profit ≥ 870
  (Finset.card (Finset.filter (λ x => 1 ≤ x ∧ x ≤ 30 ∧ daily_profit x m n ≥ 870) (Finset.range 31)) = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blueberry_sales_analysis_l1350_135055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_form_l1350_135027

noncomputable def rationalize_denominator (x : ℝ) : ℝ := 4 / (3 + x^(1/3))

def is_not_cube_divisible (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^3 ∣ n)

theorem rationalized_form :
  ∃ (A B C D : ℤ),
    rationalize_denominator 7 = (A * 7^(1/3) + C) / D ∧
    D > 0 ∧
    is_not_cube_divisible B.natAbs ∧
    Int.gcd A (Int.gcd C D) = 1 ∧
    A = -2 ∧ B = 7 ∧ C = 6 ∧ D = 2 :=
by sorry

#check rationalized_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_form_l1350_135027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l1350_135046

noncomputable def complex_number : ℂ := (3 * Complex.I) / (1 - Complex.I)

theorem complex_number_in_second_quadrant :
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l1350_135046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1350_135017

/-- Represents the time (in hours) it takes for a person to complete the work alone -/
structure WorkTime where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Calculates the portion of work completed in one cycle (3 hours) -/
def work_per_cycle (wt : WorkTime) : ℚ :=
  1 / wt.a + 1 / wt.b + 1 / wt.c

/-- Theorem: Given the work times for a, b, and c, the work will be completed in 6 hours -/
theorem work_completion_time (wt : WorkTime) 
  (ha : wt.a = 4) 
  (hb : wt.b = 12) 
  (hc : wt.c = 6) : 
  work_per_cycle wt = 1/2 ∧ 2 * 3 = 6 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1350_135017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_digits_l1350_135056

theorem max_product_digits : ∀ a b : ℕ, 
  10000 ≤ a ∧ a < 100000 → 1000 ≤ b ∧ b < 10000 → 
  a * b < 10000000000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_digits_l1350_135056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_explorer_journey_solution_l1350_135045

/-- Represents the explorer's journey with speed reduction -/
structure ExplorerJourney where
  initial_speed : ℝ
  total_distance : ℝ
  speed_reduction_time : ℝ
  reduced_speed_factor : ℝ
  actual_delay : ℝ
  hypothetical_extra_distance : ℝ
  hypothetical_delay : ℝ

/-- Calculates the total time of the journey -/
noncomputable def total_journey_time (j : ExplorerJourney) : ℝ :=
  j.speed_reduction_time + (j.total_distance - j.speed_reduction_time * j.initial_speed) / (j.reduced_speed_factor * j.initial_speed)

/-- Theorem stating the conditions and solution of the explorer's journey -/
theorem explorer_journey_solution (j : ExplorerJourney) 
  (h1 : j.speed_reduction_time = 24)
  (h2 : j.reduced_speed_factor = 3/5)
  (h3 : j.actual_delay = 48)
  (h4 : j.hypothetical_extra_distance = 120)
  (h5 : j.hypothetical_delay = 24)
  (h6 : total_journey_time j = total_journey_time { j with 
    total_distance := j.total_distance + j.hypothetical_extra_distance } + j.hypothetical_delay) :
  j.total_distance = 320 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_explorer_journey_solution_l1350_135045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_cleaning_time_l1350_135050

noncomputable def alice_time : ℝ := 30

noncomputable def bob_time (alice : ℝ) : ℝ := (1 / 3) * alice

noncomputable def carol_time (bob : ℝ) : ℝ := (3 / 4) * bob

theorem carol_cleaning_time : carol_time (bob_time alice_time) = 7.5 := by
  -- Unfold definitions
  unfold carol_time bob_time alice_time
  -- Simplify the expression
  simp [mul_assoc]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_cleaning_time_l1350_135050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_in_nonagon_l1350_135026

/-- Calculates the number of diagonals in a regular polygon with n sides. -/
def number_of_diagonals_in_regular_polygon (n : ℕ) : ℕ :=
  (Nat.choose n 2) - n

/-- The number of diagonals in a regular nine-sided polygon is 27. -/
theorem diagonals_in_nonagon : ∃ (n : ℕ), n = 27 ∧ n = number_of_diagonals_in_regular_polygon 9 := by
  use 27
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_in_nonagon_l1350_135026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l1350_135028

/-- Represents the speed of a boat in still water and the speed of the current. -/
structure BoatSpeeds where
  stillWater : ℝ
  current : ℝ

/-- Calculates the total time for a boat journey given distances and speeds. -/
noncomputable def totalTime (d₁ d₂ : ℝ) (speeds : BoatSpeeds) : ℝ :=
  d₁ / (speeds.stillWater + speeds.current) + d₂ / (speeds.stillWater - speeds.current)

/-- The boat's speed in still water satisfies the given conditions. -/
theorem boat_speed_in_still_water :
  ∃ speeds : BoatSpeeds,
    totalTime 80 48 speeds = 9 ∧
    totalTime 64 96 speeds = 12 ∧
    speeds.stillWater = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l1350_135028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1350_135061

noncomputable def f (x : ℝ) := (abs (x - 2) - abs (x^2 - 4*x + 2)) / (2 * Real.sqrt (2*x^2 + 7*x + 3) - 3*x - 4)

def solution_set : Set ℝ := 
  Set.Icc (-1/2) 0 ∪ Set.Icc 1 2 ∪ Set.Ioo 2 3 ∪ Set.Ici 4

theorem inequality_solution :
  ∀ x : ℝ, 2*x^2 + 7*x + 3 ≥ 0 → 
  (2 * Real.sqrt (2*x^2 + 7*x + 3) - 3*x - 4 ≠ 0) →
  (f x ≥ 0 ↔ x ∈ solution_set) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1350_135061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_15_l1350_135006

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hours : ℕ)
  (minutes : ℕ)

/-- Calculates the angle of the hour hand from 12 o'clock position -/
noncomputable def hour_angle (c : Clock) : ℝ :=
  (c.hours % 12 + c.minutes / 60 : ℝ) * 30

/-- Calculates the angle of the minute hand from 12 o'clock position -/
noncomputable def minute_angle (c : Clock) : ℝ :=
  (c.minutes : ℝ) * 6

/-- Calculates the angle between hour and minute hands -/
noncomputable def angle_between_hands (c : Clock) : ℝ :=
  abs (hour_angle c - minute_angle c)

/-- Theorem: The angle between the hour and minute hands at 3:15 is 7.5 degrees -/
theorem angle_at_3_15 :
  angle_between_hands ⟨3, 15⟩ = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_15_l1350_135006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_of_seven_primes_l1350_135065

theorem smallest_of_seven_primes (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧
  Prime (a + b - c) ∧ Prime (a + c - b) ∧ Prime (b + c - a) ∧ Prime (a + b + c) ∧
  (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
   a ≠ (a + b - c) ∧ a ≠ (a + c - b) ∧ a ≠ (b + c - a) ∧ a ≠ (a + b + c) ∧
   b ≠ (a + b - c) ∧ b ≠ (a + c - b) ∧ b ≠ (b + c - a) ∧ b ≠ (a + b + c) ∧
   c ≠ (a + b - c) ∧ c ≠ (a + c - b) ∧ c ≠ (b + c - a) ∧ c ≠ (a + b + c) ∧
   (a + b - c) ≠ (a + c - b) ∧ (a + b - c) ≠ (b + c - a) ∧ (a + b - c) ≠ (a + b + c) ∧
   (a + c - b) ≠ (b + c - a) ∧ (a + c - b) ≠ (a + b + c) ∧
   (b + c - a) ≠ (a + b + c)) →
  Nat.min a (Nat.min b (Nat.min c (Nat.min (a + b - c) (Nat.min (a + c - b) (Nat.min (b + c - a) (a + b + c)))))) = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_of_seven_primes_l1350_135065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arctan_three_four_l1350_135066

theorem cos_arctan_three_four : Real.cos (Real.arctan (3 / 4)) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arctan_three_four_l1350_135066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1350_135048

-- Define the custom operation
noncomputable def circle_slash (a b : ℝ) : ℝ := (Real.sqrt (3 * a + b)) ^ 3

-- State the theorem
theorem solve_equation (y : ℝ) (h : circle_slash 5 y = 64) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1350_135048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_is_neg_sin_l1350_135029

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ x => Real.cos x
  | n + 1 => λ x => deriv (f n) x

-- State the theorem
theorem f_2010_is_neg_sin : f 2010 = λ x => -Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_is_neg_sin_l1350_135029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_unextended_twice_l1350_135035

/-- The function describing the vertical oscillations of a mass on a spring -/
noncomputable def x (a : ℝ) (t : ℝ) : ℝ := a * Real.exp (-2 * t) + (1 - 2 * a) * Real.exp (-t) + 1

/-- The theorem stating the condition for the spring to be in an unextended state twice -/
theorem spring_unextended_twice (a : ℝ) :
  (∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ t₁ ≠ t₂ ∧ x a t₁ = 0 ∧ x a t₂ = 0) ↔
  (1 + Real.sqrt 3 / 2 < a ∧ a < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_unextended_twice_l1350_135035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_diagonals_theorem_l1350_135049

/-- A simple n-gon -/
structure SimpleNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_simple : Prop

/-- An inner diagonal of a simple n-gon -/
def InnerDiagonal (n : ℕ) (P : SimpleNGon n) : Type := Fin n × Fin n

/-- The number of inner diagonals of a simple n-gon -/
def D (n : ℕ) (P : SimpleNGon n) : ℕ := sorry

/-- The minimum number of inner diagonals possible for any n-gon -/
def D_min (n : ℕ) : ℕ := sorry

/-- Two inner diagonals intersect -/
def intersect (n : ℕ) (P : SimpleNGon n) (d1 d2 : InnerDiagonal n P) : Prop := sorry

theorem inner_diagonals_theorem (n : ℕ) (h : n ≥ 3) (P : SimpleNGon n) :
  D n P = D_min n ↔ ∀ (d1 d2 : InnerDiagonal n P), d1 ≠ d2 → ¬intersect n P d1 d2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_diagonals_theorem_l1350_135049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_proof_l1350_135010

/-- The distance between the foci of an ellipse with semi-major axis 7 and semi-minor axis 3 -/
noncomputable def ellipse_foci_distance : ℝ := 4 * Real.sqrt 10

/-- Theorem: The distance between the foci of an ellipse with semi-major axis 7 and semi-minor axis 3 is 4√10 -/
theorem ellipse_foci_distance_proof (a b : ℝ) (h1 : a = 7) (h2 : b = 3) :
  2 * Real.sqrt (a^2 - b^2) = ellipse_foci_distance :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_proof_l1350_135010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_ellipse_with_foci_on_y_axis_l1350_135063

-- Define the angle q in the third quadrant
noncomputable def q : ℝ := Real.pi + Real.pi / 4  -- An example value in the third quadrant

-- Define the conditions for q being in the third quadrant
axiom q_in_third_quadrant : Real.pi < q ∧ q < 3 * Real.pi / 2

-- Define the equation of the curve
def curve_equation (x y : ℝ) : Prop :=
  x^2 + y^2 * Real.sin q = Real.cos q

-- Theorem stating that the curve is an ellipse with foci on the y-axis
theorem curve_is_ellipse_with_foci_on_y_axis :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ b > a ∧
  ∀ x y : ℝ, curve_equation x y ↔ (x^2 / a^2 + y^2 / b^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_ellipse_with_foci_on_y_axis_l1350_135063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1350_135071

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - x^2

theorem max_value_of_f :
  ∃ (a : ℝ), a = 34/4 ∧ ∀ (x : ℝ), x ≥ 0 → f x ≤ f a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1350_135071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_specific_arithmetic_sequence_l1350_135036

/-- Sum of an arithmetic sequence -/
noncomputable def sum_arithmetic_sequence (a₁ a₂ aₙ : ℝ) (n : ℕ) : ℝ :=
  (n / 2 : ℝ) * (a₁ + aₙ)

/-- Theorem: Sum of the specific arithmetic sequence -/
theorem sum_specific_arithmetic_sequence :
  let a₁ : ℝ := 5
  let a₂ : ℝ := 12
  let a₆ : ℝ := 47
  let n : ℕ := 6
  sum_arithmetic_sequence a₁ a₂ a₆ n = 156 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_specific_arithmetic_sequence_l1350_135036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_between_2000_and_3000_l1350_135092

theorem count_multiples_between_2000_and_3000 : 
  (Finset.filter (fun n => n % 10 = 0 ∧ n % 15 = 0 ∧ n % 30 = 0) (Finset.range 1001)).card = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_between_2000_and_3000_l1350_135092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_job_assignment_l1350_135089

-- Define the people and jobs as enumerated types
inductive Person : Type
  | XiaoWang : Person
  | XiaoLi : Person
  | XiaoZhao : Person

inductive Job : Type
  | Salesperson : Job
  | Worker : Job
  | Salesman : Job

-- Define a function that assigns a job to each person
variable (job_assignment : Person → Job)

-- Define age comparison
variable (is_older_than : Person → Person → Prop)

-- State the theorem
theorem unique_job_assignment :
  -- Condition 1: Xiao Zhao is older than the worker
  (∀ p, job_assignment p = Job.Worker → is_older_than Person.XiaoZhao p) →
  -- Condition 2: Xiao Wang and the salesperson are not of the same age
  (∀ p, job_assignment p = Job.Salesperson → ¬(is_older_than Person.XiaoWang p ∨ is_older_than p Person.XiaoWang)) →
  -- Condition 3: The salesperson is younger than Xiao Li
  (∀ p, job_assignment p = Job.Salesperson → is_older_than Person.XiaoLi p) →
  -- Each person has exactly one job
  (∀ j : Job, ∃! p : Person, job_assignment p = j) →
  -- Conclusion: The job assignment is as stated
  (job_assignment Person.XiaoZhao = Job.Salesperson ∧
   job_assignment Person.XiaoLi = Job.Salesman ∧
   job_assignment Person.XiaoWang = Job.Worker) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_job_assignment_l1350_135089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_defined_fraction_domain_l1350_135078

-- Define the fraction as a noncomputable function
noncomputable def f (x : ℝ) : ℝ := 3 / (x - 2)

-- Theorem stating that the fraction is defined when x ≠ 2
theorem fraction_defined (x : ℝ) : x ≠ 2 ↔ f x ≠ 0 := by
  sorry

-- Theorem stating the range of x for which the fraction is defined
theorem fraction_domain : Set ℝ := {x : ℝ | x ≠ 2}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_defined_fraction_domain_l1350_135078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_values_l1350_135034

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x + 1 else abs x

-- State the theorem
theorem x0_values (x₀ : ℝ) (h : f x₀ = 3) : x₀ = -3 ∨ x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_values_l1350_135034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_of_P_l1350_135015

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define a line not perpendicular to x-axis and not passing through the focus
structure Line where
  slope : ℝ
  intercept : ℝ
  not_perpendicular : slope ≠ 0
  not_through_focus : slope + intercept ≠ 1

-- Define intersection points
structure IntersectionPoints (l : Line) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  on_ellipse_M : ellipse M.1 M.2
  on_ellipse_N : ellipse N.1 N.2
  on_line_M : M.2 = l.slope * M.1 + l.intercept
  on_line_N : N.2 = l.slope * N.1 + l.intercept
  distinct : M ≠ N

-- Define the external angle bisector point
noncomputable def external_angle_bisector_point (M N F : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- Definition of P based on M, N, and F

-- Theorem statement
theorem x_coordinate_of_P (l : Line) (points : IntersectionPoints l) :
  (external_angle_bisector_point points.M points.N right_focus).1 = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_of_P_l1350_135015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_open_ray_l1350_135072

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- State the conditions
axiom f_symmetry (x : ℝ) : f x - f (-x) = 2 * x^3
axiom f'_lower_bound (x : ℝ) (h : x ≥ 0) : f' x > 3 * x^2

-- Define the solution set
def solution_set : Set ℝ := {x | f x - f (x - 1) > 3 * x^2 - 3 * x + 1}

-- State the theorem
theorem solution_set_eq_open_ray :
  solution_set = Set.Ioi (1/2 : ℝ) := by
  sorry

#check solution_set_eq_open_ray

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_open_ray_l1350_135072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1350_135094

-- Define the new operations
noncomputable def new_add (a b : ℝ) : ℝ := a * b
noncomputable def new_sub (a b : ℝ) : ℝ := a + b
noncomputable def new_mul (a b : ℝ) : ℝ := a / b
noncomputable def new_div (a b : ℝ) : ℝ := a - b

-- Define the equation
def equation (x : ℝ) : Prop :=
  new_sub 6 (new_add x (new_div (new_mul 8 3) 25)) = 5

-- Theorem statement
theorem solution_exists : ∃ x : ℝ, equation x ∧ x = 9 := by
  -- Proof goes here
  sorry

#check solution_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1350_135094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_foci_ellipses_same_foci_l1350_135032

-- Define the first ellipse
def ellipse1 (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the second ellipse with parameter k
def ellipse2 (k : ℝ) (x y : ℝ) : Prop := x^2 / (25 - k) + y^2 / (9 - k) = 1

-- Define the focal distance for an ellipse
noncomputable def focal_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

-- Theorem stating that the foci of both ellipses are the same
theorem same_foci (k : ℝ) (h : k < 9) :
  focal_distance 5 3 = focal_distance (Real.sqrt (25 - k)) (Real.sqrt (9 - k)) := by
  -- The proof is omitted for now
  sorry

-- Helper lemma to show that the focal distance of the first ellipse is 4
lemma focal_distance_first_ellipse : focal_distance 5 3 = 4 := by
  -- The proof is omitted for now
  sorry

-- Main theorem stating that both ellipses have the same foci
theorem ellipses_same_foci (k : ℝ) (h : k < 9) :
  ∃ c : ℝ, c = focal_distance 5 3 ∧ c = focal_distance (Real.sqrt (25 - k)) (Real.sqrt (9 - k)) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_foci_ellipses_same_foci_l1350_135032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ac_mn_is_90_l1350_135012

/-- Represents a rectangle with sides a and b -/
structure Rectangle (a b : ℝ) where
  sides_positive : 0 < a ∧ 0 < b
  b_greater_a : b > a

/-- Represents a fold in the rectangle -/
structure Fold (a b : ℝ) extends Rectangle a b where
  fold_line : ℝ × ℝ  -- Representing MN as a line
  a_coincides_c : ℝ × ℝ  -- Point where A coincides with C
  dihedral_angle : ℝ
  dihedral_angle_value : dihedral_angle = 57

/-- Function to calculate angle between two lines -/
noncomputable def angle_between_lines (line1 line2 : ℝ × ℝ) : ℝ :=
  sorry -- Placeholder for the actual implementation

/-- The main theorem stating that the angle between AC and MN is 90 degrees -/
theorem angle_ac_mn_is_90 (a b : ℝ) (fold : Fold a b) :
  angle_between_lines fold.a_coincides_c fold.fold_line = 90 :=
by
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ac_mn_is_90_l1350_135012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1350_135041

-- Define the sets A and B
def A : Set ℝ := {x | x > 4}
def B : Set ℝ := {x | -6 < x ∧ x < 6}

-- Define set difference operation
def setDiff (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

theorem set_operations :
  (A ∩ B = {x : ℝ | 4 < x ∧ x < 6}) ∧
  (Set.univ \ B = {x : ℝ | x ≥ 6 ∨ x ≤ -6}) ∧
  (setDiff A B = {x : ℝ | x ≥ 6}) ∧
  (setDiff A (setDiff A B) = {x : ℝ | 4 < x ∧ x < 6}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1350_135041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_1_to_4_range_of_m_for_inequality_l1350_135064

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 - 2) * (Real.log x / Real.log 4 - 1/2)

-- Part 1
theorem range_of_f_on_1_to_4 :
  ∀ y ∈ Set.range (fun x => f x) ∩ Set.Icc 1 4,
  -1/8 ≤ y ∧ y ≤ 1 :=
by sorry

-- Part 2
theorem range_of_m_for_inequality :
  ∀ m : ℝ, (∀ x ∈ Set.Icc 4 16, f x > m * (Real.log x / Real.log 4)) → m < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_1_to_4_range_of_m_for_inequality_l1350_135064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_is_one_fourth_l1350_135038

/-- Two internally tangent circles with equations (x-a)²+(y+2)²=4 and (x+b)²+(y+2)²=1 -/
structure TangentCircles where
  a : ℝ
  b : ℝ
  tangent : (a + b)^2 = 1

/-- The maximum value of ab for two internally tangent circles -/
noncomputable def max_ab (circles : TangentCircles) : ℝ := 1/4

/-- Theorem: The maximum value of ab for two internally tangent circles is 1/4 -/
theorem max_ab_is_one_fourth (circles : TangentCircles) : 
  max_ab circles = 1/4 ∧ circles.a * circles.b ≤ 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_is_one_fourth_l1350_135038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1350_135081

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt ((a^2 + b^2) / a^2)

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), hyperbola a b x y ∧ b / a = 2) →
  eccentricity a b = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1350_135081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_methods_count_l1350_135053

/-- The number of candidates --/
def total_candidates : ℕ := 8

/-- The number of candidates to be selected --/
def selected_candidates : ℕ := 3

/-- The number of activities --/
def activities : ℕ := 3

/-- The theorem stating the number of selection methods --/
theorem selection_methods_count : 
  (Nat.choose (total_candidates - 1) selected_candidates * 2) + 
  (Nat.descFactorial (total_candidates - 1) selected_candidates) = 294 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_methods_count_l1350_135053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_satisfying_equation_l1350_135083

theorem smallest_angle_satisfying_equation : 
  ∃ y : ℝ, y > 0 ∧ 
    (∀ z : ℝ, z > 0 → Real.sin (4 * z) * Real.sin (5 * z) = Real.cos (4 * z) * Real.cos (5 * z) → y ≤ z) ∧ 
    Real.sin (4 * y) * Real.sin (5 * y) = Real.cos (4 * y) * Real.cos (5 * y) ∧ 
    y * (180 / Real.pi) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_satisfying_equation_l1350_135083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1350_135099

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt ((1 - x^2) / (1 + x^2)) + a * Real.sqrt ((1 + x^2) / (1 - x^2))

theorem f_properties (a : ℝ) (ha : a > 0) :
  -- Domain of f is (-1, 1)
  (∀ x, -1 < x ∧ x < 1 → f a x ≠ 0) ∧
  -- f is an even function
  (∀ x, -1 < x ∧ x < 1 → f a x = f a (-x)) ∧
  -- When a = 1, the minimum value of f(x) is 2
  (a = 1 → ∀ x, -1 < x ∧ x < 1 → f 1 x ≥ 2) ∧
  (a = 1 → ∃ x, -1 < x ∧ x < 1 ∧ f 1 x = 2) ∧
  -- When a = 1, f(x) is increasing on [0, 1) and decreasing on (-1, 0]
  (a = 1 → ∀ x y, 0 ≤ x ∧ x < y ∧ y < 1 → f 1 x < f 1 y) ∧
  (a = 1 → ∀ x y, -1 < x ∧ x < y ∧ y ≤ 0 → f 1 x > f 1 y) ∧
  -- Triangle inequality condition
  (∀ r s t, -2*Real.sqrt 5/5 ≤ r ∧ r ≤ 2*Real.sqrt 5/5 ∧
            -2*Real.sqrt 5/5 ≤ s ∧ s ≤ 2*Real.sqrt 5/5 ∧
            -2*Real.sqrt 5/5 ≤ t ∧ t ≤ 2*Real.sqrt 5/5 →
            (f a r + f a s > f a t ∧
             f a s + f a t > f a r ∧
             f a t + f a r > f a s)) ↔
  (1/15 < a ∧ a < 5/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1350_135099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_coordinates_change_of_basis_l1350_135020

open Real

-- Define the vector space
variable (V : Type*) [AddCommGroup V] [Module ℝ V]

-- Define the basis vectors
variable (i j k : V)

-- Define the given basis vectors in terms of i, j, k
def a (V : Type*) [AddCommGroup V] [Module ℝ V] (i j : V) : V := i + j
def b (V : Type*) [AddCommGroup V] [Module ℝ V] (j k : V) : V := j + k
def c (V : Type*) [AddCommGroup V] [Module ℝ V] (k i : V) : V := k + i

-- Define the vector p in terms of a, b, c
def p (V : Type*) [AddCommGroup V] [Module ℝ V] (i j k : V) : V :=
  8 • (a V i j) + 6 • (b V j k) + 4 • (c V k i)

-- Theorem statement
theorem vector_coordinates_change_of_basis (V : Type*) [AddCommGroup V] [Module ℝ V] (i j k : V) :
  p V i j k = 12 • i + 14 • j + 10 • k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_coordinates_change_of_basis_l1350_135020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_condition_l1350_135052

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - a * (x^2 + 1)

-- State the theorem
theorem function_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ -2*a + Real.log x) ↔ a ≤ (Real.exp 1 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_condition_l1350_135052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_10_l1350_135023

/-- The number of degrees in a full circle -/
def full_circle : ℚ := 360

/-- The number of hours marked on a clock -/
def clock_hours : ℕ := 12

/-- The angle between each hour mark on a clock -/
noncomputable def hour_angle : ℚ := full_circle / clock_hours

/-- The position of the minute hand at 10:00 in degrees -/
def minute_hand_position : ℚ := 0

/-- The position of the hour hand at 10:00 in hours -/
def hour_hand_position : ℚ := 10

/-- The smaller angle formed by the hands of a clock at 10:00 -/
noncomputable def clock_angle : ℚ := hour_hand_position * hour_angle - minute_hand_position

theorem clock_angle_at_10 : clock_angle = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_10_l1350_135023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_not_divisible_by_five_l1350_135033

def base_expression (b : ℤ) : ℤ := 2*b^3 - 2*b^2 - b + 1

def is_valid_base (b : ℤ) : Prop := b = 4 ∨ b = 5 ∨ b = 7 ∨ b = 9 ∨ b = 10

theorem base_not_divisible_by_five :
  ∀ b : ℤ, is_valid_base b → (¬(5 ∣ base_expression b) ↔ (b = 4 ∨ b = 7 ∨ b = 9)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_not_divisible_by_five_l1350_135033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_equivalence_l1350_135016

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.pi / 2 - 2 * x)

theorem graph_shift_equivalence :
  ∀ x : ℝ, f x = g (x - Real.pi / 8) :=
by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_shift_equivalence_l1350_135016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perpendicular_sum_l1350_135004

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- represents ax + by + c = 0

/-- Function to calculate the perpendicular distance from a point to a line -/
noncomputable def perpDistance (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Function to calculate the tangent of an angle in a triangle -/
noncomputable def tanAngle (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Function to calculate the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

/-- The main theorem -/
theorem triangle_perpendicular_sum (t : Triangle) (l : Line) :
  let u := perpDistance t.A l
  let v := perpDistance t.B l
  let w := perpDistance t.C l
  u^2 * tanAngle t t.A + v^2 * tanAngle t t.B + w^2 * tanAngle t t.C ≥ 2 * triangleArea t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perpendicular_sum_l1350_135004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1350_135042

/-- Triangle ABC with given altitude equations and vertex A -/
structure Triangle where
  -- Vertex A coordinates
  ax : ℝ
  ay : ℝ
  -- Altitude equations
  altitude1 : ℝ → ℝ → ℝ
  altitude2 : ℝ → ℝ → ℝ

/-- The specific triangle from the problem -/
def triangle_ABC : Triangle where
  ax := 1
  ay := 2
  altitude1 := fun x y => 2 * x - 3 * y + 1
  altitude2 := fun x y => x + y

/-- The equation of line BC -/
noncomputable def line_BC (t : Triangle) : ℝ → ℝ → ℝ :=
  fun x y => 2 * x + 3 * y + 7

/-- The area of the triangle -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  45 / 2

/-- Main theorem stating the properties of triangle ABC -/
theorem triangle_ABC_properties :
  let t := triangle_ABC
  (∀ x y, line_BC t x y = 0 ↔ y = -2/3 * x - 7/3) ∧
  triangle_area t = 45 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1350_135042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_1_theorem_2_l1350_135086

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2 - x else x + 2

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := x * f x

-- Theorem 1: If F(a) = 3, then a = -3
theorem theorem_1 (a : ℝ) (h : F a = 3) : a = -3 := by
  sorry

-- Theorem 2: F(x) < 0 if and only if x ∈ (-2, 0) ∪ (2, +∞)
theorem theorem_2 (x : ℝ) : F x < 0 ↔ x ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi (2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_1_theorem_2_l1350_135086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_intersections_same_side_odd_intersections_opposite_side_l1350_135014

/-- A plane with an axis --/
structure AxisPlane where
  axis : Set (ℝ × ℝ)

/-- A point in the plane --/
structure PlanePoint where
  x : ℝ
  y : ℝ

/-- A curve in the plane --/
structure PlaneCurve where
  start : PlanePoint
  finish : PlanePoint

/-- Determines if two points are on the same side of an axis --/
def sameSide (p1 p2 : PlanePoint) (plane : AxisPlane) : Prop :=
  sorry

/-- Counts the number of intersections between a curve and an axis --/
def intersectionCount (curve : PlaneCurve) (plane : AxisPlane) : ℕ :=
  sorry

theorem even_intersections_same_side (A B : PlanePoint) (plane : AxisPlane) (curve : PlaneCurve) :
  curve.start = A →
  curve.finish = B →
  (A.x, A.y) ∉ plane.axis →
  (B.x, B.y) ∉ plane.axis →
  Even (intersectionCount curve plane) →
  sameSide A B plane :=
by
  sorry

theorem odd_intersections_opposite_side (A B : PlanePoint) (plane : AxisPlane) (curve : PlaneCurve) :
  curve.start = A →
  curve.finish = B →
  (A.x, A.y) ∉ plane.axis →
  (B.x, B.y) ∉ plane.axis →
  Odd (intersectionCount curve plane) →
  ¬(sameSide A B plane) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_intersections_same_side_odd_intersections_opposite_side_l1350_135014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_cosine_floor_l1350_135025

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex_quadrilateral (PQRS : Quadrilateral) : Prop := sorry

def angle_P_equals_angle_R (PQRS : Quadrilateral) : Prop := sorry

noncomputable def side_PQ_equals_200 (PQRS : Quadrilateral) : ℝ :=
  Real.sqrt ((PQRS.P.1 - PQRS.Q.1)^2 + (PQRS.P.2 - PQRS.Q.2)^2)

noncomputable def side_RS_equals_200 (PQRS : Quadrilateral) : ℝ :=
  Real.sqrt ((PQRS.R.1 - PQRS.S.1)^2 + (PQRS.R.2 - PQRS.S.2)^2)

def PR_not_equal_QS (PQRS : Quadrilateral) : Prop :=
  Real.sqrt ((PQRS.P.1 - PQRS.R.1)^2 + (PQRS.P.2 - PQRS.R.2)^2) ≠
  Real.sqrt ((PQRS.Q.1 - PQRS.S.1)^2 + (PQRS.Q.2 - PQRS.S.2)^2)

noncomputable def perimeter_equals_680 (PQRS : Quadrilateral) : ℝ :=
  side_PQ_equals_200 PQRS + side_RS_equals_200 PQRS +
  Real.sqrt ((PQRS.P.1 - PQRS.R.1)^2 + (PQRS.P.2 - PQRS.R.2)^2) +
  Real.sqrt ((PQRS.Q.1 - PQRS.S.1)^2 + (PQRS.Q.2 - PQRS.S.2)^2)

noncomputable def angle_P (PQRS : Quadrilateral) : ℝ := sorry

-- The main theorem
theorem quadrilateral_cosine_floor (PQRS : Quadrilateral) :
  is_convex_quadrilateral PQRS →
  angle_P_equals_angle_R PQRS →
  side_PQ_equals_200 PQRS = 200 →
  side_RS_equals_200 PQRS = 200 →
  PR_not_equal_QS PQRS →
  perimeter_equals_680 PQRS = 680 →
  ⌊1000 * Real.cos (angle_P PQRS)⌋ = 700 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_cosine_floor_l1350_135025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_equation_l1350_135022

/-- Given a triangle with vertices A(4,3), B(-4,-1), and C(9,-7),
    prove that the equation of the angle bisector of ∠A
    in the form 3x - by + c = 0 satisfies b + c = -6 -/
theorem angle_bisector_equation (b c : ℝ) : 
  let A : ℝ × ℝ := (4, 3)
  let B : ℝ × ℝ := (-4, -1)
  let C : ℝ × ℝ := (9, -7)
  let bisector_eq := fun (x y : ℝ) ↦ 3*x - b*y + c = 0
  bisector_eq A.1 A.2 ∧ 
  (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
    let D := (t * C.1 + (1-t) * B.1, t * C.2 + (1-t) * B.2)
    bisector_eq D.1 D.2) →
  b + c = -6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_equation_l1350_135022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_two_zeros_l1350_135084

/-- Definition of the function f --/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/2) * Real.sqrt (x^2 + 1) else -Real.log (1 - x)

/-- Definition of the function F --/
noncomputable def F (k : ℝ) (x : ℝ) : ℝ := f x - k * x

/-- Theorem stating the range of k --/
theorem k_range_for_two_zeros (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F k x₁ = 0 ∧ F k x₂ = 0) ∧
  (∀ x₃ : ℝ, (x₃ ≠ x₁ ∧ x₃ ≠ x₂) → F k x₃ ≠ 0) →
  k ∈ Set.Ioo (1/2) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_two_zeros_l1350_135084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l1350_135067

/-- The probability of the coin landing heads -/
def p : ℝ := sorry

/-- The expected value function after one flip -/
noncomputable def f (x : ℝ) : ℝ := p * (3 * x + 1) + (1 - p) * (x / 2)

/-- Theorem: If the expected value forms an arithmetic sequence, then p = 1/5 -/
theorem coin_flip_probability :
  (0 < p ∧ p < 1) →
  (∃ a b : ℝ, ∀ t : ℕ+, (f^[t]) 0 = a * t + b) →
  p = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l1350_135067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1350_135005

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 2^x

theorem f_properties : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x > f y) := by
  constructor
  · intro x
    simp [f]
    sorry
  · intro x y hxy
    simp [f]
    sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1350_135005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_selling_price_l1350_135031

theorem chocolate_selling_price 
  (num_bars : ℕ) 
  (cost_per_bar : ℚ) 
  (packaging_cost_per_bar : ℚ) 
  (total_profit : ℚ) 
  (h1 : num_bars = 5)
  (h2 : cost_per_bar = 5)
  (h3 : packaging_cost_per_bar = 2)
  (h4 : total_profit = 55)
  : 
  (num_bars * cost_per_bar + num_bars * packaging_cost_per_bar + total_profit : ℚ) = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_selling_price_l1350_135031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_largest_angle_l1350_135019

/-- The sum of angles in a hexagon -/
def hexagon_angle_sum : ℝ := 720

/-- The ratio of angles in the hexagon -/
def angle_ratio : List ℝ := [2, 3, 3, 4, 4, 6]

/-- The largest angle measure in the hexagon -/
def largest_angle : ℝ := 196.36

theorem hexagon_largest_angle :
  let sum_ratio := angle_ratio.sum
  let x := hexagon_angle_sum / sum_ratio
  (List.maximum? angle_ratio).map (· * x) = some largest_angle := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_largest_angle_l1350_135019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1350_135001

-- Define the curves
def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := -x + 3

-- Define the region
def region_bound (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

-- State the theorem
theorem area_of_region :
  (∫ x in Set.Icc 0 1, (f x - 0) + (min (g x) (f x) - f x)) = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1350_135001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_ring_area_implies_equal_side_length_l1350_135096

/-- Represents a regular polygon with its circumscribed and inscribed circles -/
structure RegularPolygon where
  n : ℕ
  side_length : ℝ
  inscribed_radius : ℝ
  circumscribed_radius : ℝ
  n_ge_3 : n ≥ 3
  pythagorean : side_length^2 + inscribed_radius^2 = circumscribed_radius^2

/-- The area of the ring between the circumscribed and inscribed circles -/
noncomputable def ring_area (p : RegularPolygon) : ℝ :=
  Real.pi * (p.circumscribed_radius^2 - p.inscribed_radius^2)

/-- Theorem: If the ring areas of a regular heptagon and a regular 17-gon are equal,
    then their side lengths are equal -/
theorem equal_ring_area_implies_equal_side_length
  (heptagon : RegularPolygon)
  (heptadecagon : RegularPolygon)
  (heptagon_sides : heptagon.n = 7)
  (heptadecagon_sides : heptadecagon.n = 17)
  (equal_ring_areas : ring_area heptagon = ring_area heptadecagon) :
  heptagon.side_length = heptadecagon.side_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_ring_area_implies_equal_side_length_l1350_135096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skier_route_time_comparison_l1350_135085

theorem skier_route_time_comparison (d v : ℝ) (d_pos : d > 0) (v_pos : v > 0) :
  (3*d)/(2*v) + d/(6*v) > d/v := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skier_route_time_comparison_l1350_135085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l1350_135039

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 4*x + 5) / Real.log (1/2)

theorem monotone_increasing_interval (m : ℝ) :
  (∀ x ∈ Set.Ioo (3*m - 2) (m + 2), StrictMono f) ↔ m ∈ Set.Icc (4/3) 2 ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l1350_135039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l1350_135098

theorem quadratic_root_form (a b c m n p : ℤ) : 
  a = 3 ∧ b = -7 ∧ c = 1 ∧ 
  (∃ x : ℚ, a * x^2 + b * x + c = 0 ∧ 
    ∃ s : ℤ, (s = 1 ∨ s = -1) ∧ x = (m + s * n.sqrt) / p) ∧
  m > 0 ∧ n > 0 ∧ p > 0 ∧
  Nat.gcd (Nat.gcd m.natAbs n.natAbs) p.natAbs = 1 →
  n = 37 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l1350_135098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_store_distribution_l1350_135044

/-- The number of ways to distribute n distinct items into m identical bags, 
    where some bags may be left empty -/
def number_of_distributions (n m : ℕ) : ℕ :=
  sorry

theorem grocery_store_distribution : 
  ∀ (n m : ℕ), n = 5 ∧ m = 4 → 
  (number_of_distributions n m) = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_store_distribution_l1350_135044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jade_diamond_weight_difference_l1350_135057

/-- The weight of a single diamond in grams -/
noncomputable def diamond_weight : ℝ := 100 / 5

/-- The weight of a single jade in grams -/
noncomputable def jade_weight : ℝ := (140 - 4 * diamond_weight) / 2

/-- Theorem stating the difference in weight between a jade and a diamond -/
theorem jade_diamond_weight_difference :
  jade_weight - diamond_weight = 10 := by
  -- Unfold the definitions
  unfold jade_weight diamond_weight
  -- Simplify the expression
  simp [sub_eq_add_neg, add_comm, add_left_comm]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jade_diamond_weight_difference_l1350_135057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_is_correct_l1350_135095

/-- The plane equation for the given points -/
def plane_equation (x y z : ℝ) : ℝ := x + y - 2*z - 6

/-- The three points that the plane should contain -/
def point1 : Fin 3 → ℝ := ![-3, 5, -2]
def point2 : Fin 3 → ℝ := ![1, 5, 0]
def point3 : Fin 3 → ℝ := ![3, 3, -1]

/-- The coefficients of the plane equation -/
def A : ℤ := 1
def B : ℤ := 1
def C : ℤ := -2
def D : ℤ := -6

theorem plane_equation_is_correct :
  (∀ (p : Fin 3 → ℝ), p = point1 ∨ p = point2 ∨ p = point3 → 
    plane_equation (p 0) (p 1) (p 2) = 0) ∧
  A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_is_correct_l1350_135095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_flow_increase_is_150_l1350_135058

/-- Represents the flow rate increase when swapping one pipe -/
def swap_increase : ℚ := 30

/-- Represents the initial number of pipes between A-B and B-C -/
def initial_pipes : ℕ := 10

/-- Calculates the maximum flow rate increase -/
def max_flow_increase (swap_increase : ℚ) (initial_pipes : ℕ) : ℚ :=
  swap_increase * (initial_pipes / 2)

/-- Theorem stating the maximum flow rate increase -/
theorem max_flow_increase_is_150 :
  max_flow_increase swap_increase initial_pipes = 150 := by
  -- Unfold the definition of max_flow_increase
  unfold max_flow_increase
  -- Simplify the expression
  simp [swap_increase, initial_pipes]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_flow_increase_is_150_l1350_135058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tablet_sales_theorem_l1350_135021

/-- Represents the financial details of tablet sales in a mall --/
structure TabletSales where
  type_a_cost : ℚ
  type_b_cost : ℚ
  type_b_quantity_ratio : ℚ
  type_b_price_difference : ℚ
  selling_price : ℚ
  discounted_quantity : ℕ
  discount_rate : ℚ

/-- Calculates the unit prices and profit for the given tablet sales --/
noncomputable def calculate_prices_and_profit (sales : TabletSales) : 
  (ℚ × ℚ × ℚ) := 
  let type_a_price := sales.type_a_cost / (sales.type_b_cost / (2 * (sales.type_a_cost / sales.type_b_quantity_ratio + sales.type_b_price_difference)))
  let type_b_price := type_a_price + sales.type_b_price_difference
  let total_quantity := (sales.type_a_cost / type_a_price) + (sales.type_b_cost / type_b_price)
  let full_price_sales := (total_quantity - sales.discounted_quantity) * sales.selling_price
  let discounted_sales := sales.discounted_quantity * sales.selling_price * (1 - sales.discount_rate)
  let profit := full_price_sales + discounted_sales - sales.type_a_cost - sales.type_b_cost
  (type_a_price, type_b_price, profit)

/-- Theorem stating the correct unit prices and profit for the given scenario --/
theorem tablet_sales_theorem (sales : TabletSales) 
  (h1 : sales.type_a_cost = 60000)
  (h2 : sales.type_b_cost = 128000)
  (h3 : sales.type_b_quantity_ratio = 2)
  (h4 : sales.type_b_price_difference = 40)
  (h5 : sales.selling_price = 700)
  (h6 : sales.discounted_quantity = 50)
  (h7 : sales.discount_rate = 1/5) : 
  calculate_prices_and_profit sales = (600, 640, 15000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tablet_sales_theorem_l1350_135021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_problem_l1350_135054

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem complex_power_problem (x y : ℝ) (h : (x - 1) * i - y = 2 + i) :
  (1 + i : ℂ) ^ (x - y : ℂ) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_problem_l1350_135054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_weakly_increasing_implies_a_eq_4_l1350_135062

-- Define the interval (0,2]
def interval : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + (4-a)*x + a

-- Define what it means for a function to be increasing on an interval
def increasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x < f y

-- Define what it means for a function to be decreasing on an interval
def decreasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x > f y

-- Define what it means for a function to be weakly increasing on an interval
def weakly_increasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  increasing_on f S ∧ decreasing_on (λ x ↦ f x / x) S

-- State the theorem
theorem g_weakly_increasing_implies_a_eq_4 :
  (∃ a : ℝ, weakly_increasing_on (g a) interval) →
  (∃! a : ℝ, weakly_increasing_on (g a) interval ∧ a = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_weakly_increasing_implies_a_eq_4_l1350_135062
