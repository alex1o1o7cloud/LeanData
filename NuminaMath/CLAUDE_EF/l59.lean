import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_value_theorem_l59_5993

def is_valid_list (l : List Int) : Prop :=
  ∃ mode, List.count mode l = List.maximum (List.map (List.count · l) l) ∧ mode = 35 ∧
  (List.sum l) / (List.length l) = 30 ∧
  List.minimum l = some 20 ∧
  ∃ m, m ∈ l ∧ (List.length (List.filter (· < m) l) ≤ List.length l / 2) ∧
         (List.length (List.filter (· > m) l) ≤ List.length l / 2)

def replace_median (l : List Int) (new_m : Int) : List Int :=
  sorry

noncomputable def median (l : List Int) : Option Int :=
  sorry

theorem median_value_theorem (l : List Int) (m : Int) : 
  is_valid_list l →
  median l = some m →
  (List.sum (replace_median l (m + 8))) / (List.length l) = 32 →
  median (replace_median l (m + 8)) = some (m + 8) →
  median (replace_median l (m - 10)) = some (m - 5) →
  m = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_value_theorem_l59_5993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_twenty_thirds_l59_5910

/-- A right triangle with sides 6, 8, and 10 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_eq : a = 6
  b_eq : b = 8
  c_eq : c = 10
  right_angle : a^2 + b^2 = c^2

/-- The length of the crease when folding point B onto point C -/
noncomputable def crease_length (t : RightTriangle) : ℝ := 20/3

/-- Theorem stating that the crease length is 20/3 inches -/
theorem crease_length_is_twenty_thirds (t : RightTriangle) :
  crease_length t = 20/3 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_twenty_thirds_l59_5910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l59_5996

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1) + 1

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1) + 1

-- Theorem statement
theorem inverse_function_proof (x : ℝ) (h : x > 1) :
  g (f x) = x ∧ f (g x) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l59_5996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_tan_cot_squared_l59_5948

theorem min_value_tan_cot_squared (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.tan x + 1 / Real.tan x)^2 + 4 ≥ 8 ∧
  ∃ y : ℝ, 0 < y ∧ y < Real.pi / 2 ∧ (Real.tan y + 1 / Real.tan y)^2 + 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_tan_cot_squared_l59_5948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l59_5972

/-- Parabola structure representing y = x^2 -/
structure Parabola where
  f : ℝ → ℝ
  eq : f = fun x ↦ x^2

/-- Focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ :=
  (0, 1/4)

/-- Theorem stating that the focus of y = x^2 is at (0, 1/4) -/
theorem parabola_focus (p : Parabola) : focus p = (0, 1/4) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l59_5972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l59_5968

-- Define the function f(x) = sin(-x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (-x)

-- Theorem stating that f is an odd function and decreasing on (0, 1)
theorem f_odd_and_decreasing :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f y < f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l59_5968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_age_multiplier_l59_5908

/-- The multiplier for Zion's age to get his dad's age, excluding the additional 3 years -/
def age_multiplier : ℕ → ℕ := sorry

/-- Zion's current age -/
def zion_age : ℕ := 8

/-- Years until the future condition is met -/
def years_passed : ℕ := 10

/-- Age difference between Zion's dad and Zion in the future -/
def future_age_difference : ℕ := 27

theorem find_age_multiplier :
  ∃ (x : ℕ), 
    (zion_age * x + 3) + years_passed = (zion_age + years_passed) + future_age_difference ∧
    age_multiplier zion_age = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_age_multiplier_l59_5908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l59_5967

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Acute triangle
  a^2 - 2*a + 2 = 0 →  -- a is a root of x^2 - 2x + 2 = 0
  b^2 - 2*b + 2 = 0 →  -- b is a root of x^2 - 2x + 2 = 0
  2 * Real.sin (A + B) - 1 = 0 →  -- Given condition
  -- Conclusions
  C = π/3 ∧  -- 60 degrees in radians
  c^2 = 6 ∧  -- c = √6
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l59_5967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_and_alternating_sum_l59_5939

def f (x : ℂ) (n : ℕ) : ℂ := (1 + x) ^ n.succ

theorem largest_coefficient_and_alternating_sum (n : ℕ) :
  (∃ (k : ℕ), Nat.choose 6 k = 20 ∧
    ∀ (j : ℕ), j ≠ k → Nat.choose 6 j ≤ 20) ∧
  (f Complex.I n.succ = 32 * Complex.I →
    Nat.choose n.succ 1 - Nat.choose n.succ 3 + Nat.choose n.succ 5 - Nat.choose n.succ 7 + Nat.choose n.succ 9 = 32) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_and_alternating_sum_l59_5939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_in_regular_hexagon_l59_5936

-- Define auxiliary functions outside the main theorem
def is_regular_hexagon (A B C D E F : ℝ × ℝ) : Prop :=
  sorry

def side_length (P Q : ℝ × ℝ) : ℝ :=
  sorry

def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  sorry

theorem area_triangle_in_regular_hexagon :
  ∀ (A B C D E F : ℝ × ℝ),
  is_regular_hexagon A B C D E F →
  side_length A B = 3 →
  area_triangle A C E = (27 * Real.sqrt 3) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_in_regular_hexagon_l59_5936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juliet_supporter_capulet_prob_l59_5918

/-- Represents the population distribution and voting preferences in Venezia --/
structure Venezia where
  total_population : ℝ
  montague_fraction : ℝ
  capulet_fraction : ℝ
  romeo_support_montague : ℝ
  juliet_support_capulet : ℝ
  montague_fraction_constraint : montague_fraction = 3/4
  capulet_fraction_constraint : capulet_fraction = 1/4
  population_sum : montague_fraction + capulet_fraction = 1
  romeo_support_constraint : romeo_support_montague = 4/5
  juliet_support_constraint : juliet_support_capulet = 7/10

/-- The probability that a randomly chosen Juliet supporter resides in Capulet --/
noncomputable def juliet_supporter_in_capulet (v : Venezia) : ℝ :=
  (v.capulet_fraction * v.juliet_support_capulet) /
  ((v.capulet_fraction * v.juliet_support_capulet) + (v.montague_fraction * (1 - v.romeo_support_montague)))

/-- Theorem stating that the probability of a randomly chosen Juliet supporter residing in Capulet is 7/13 --/
theorem juliet_supporter_capulet_prob (v : Venezia) :
  juliet_supporter_in_capulet v = 7/13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_juliet_supporter_capulet_prob_l59_5918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_proof_l59_5917

/-- The area of intersection of three strips of width 1, where one is horizontal
    and two intersect at an angle θ -/
noncomputable def intersection_area (θ : ℝ) : ℝ :=
  1 / Real.sin θ

/-- Theorem stating that the intersection area of three strips as described
    is equal to 1/sin(θ) -/
theorem intersection_area_proof (θ : ℝ) (h : 0 < θ ∧ θ < π) :
  intersection_area θ = 1 / Real.sin θ := by
  -- Unfold the definition of intersection_area
  unfold intersection_area
  -- The equality now holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_proof_l59_5917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_artist_pictures_theorem_l59_5978

/-- Given an artist who sold some pictures and has a certain ratio of remaining to sold pictures,
    calculate the total number of pictures painted. -/
def total_pictures (sold : ℕ) (remaining_ratio : ℚ) (sold_ratio : ℚ) : ℕ :=
  let remaining := (sold : ℚ) * remaining_ratio / sold_ratio
  (remaining + sold).floor.toNat

/-- Theorem stating that for an artist who sold 72 pictures and has a ratio of 9:8 for
    remaining to sold pictures, the total number of pictures painted is 153. -/
theorem artist_pictures_theorem :
  total_pictures 72 (9 : ℚ) (8 : ℚ) = 153 := by
  sorry

#eval total_pictures 72 (9 : ℚ) (8 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_artist_pictures_theorem_l59_5978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_solutions_l59_5923

open Real MeasureTheory Set

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := sin x ^ n + cos x ^ n

theorem eight_solutions :
  ∃! (S : Set ℝ), S.Finite ∧ S.ncard = 8 ∧
  (∀ x ∈ S, x ∈ Icc 0 π) ∧
  (∀ x ∈ Icc 0 π, (6 * f 4 x - 4 * f 6 x = 2 * f 2 x) ↔ x ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_solutions_l59_5923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_longest_edge_length_l59_5958

theorem final_longest_edge_length : 
  let initial_side_length : ℝ := 4
  let num_stages : ℕ := 4
  let final_piece_count : ℕ := 2^num_stages
  let longest_edge : ℝ := initial_side_length / (2^((num_stages - 1)/2 : ℝ))
  longest_edge = 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_longest_edge_length_l59_5958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flowchart_most_intuitive_flowchart_is_correct_answer_l59_5902

-- Define the possible representation methods
inductive RepresentationMethod
  | NaturalLanguage
  | Flowchart
  | MathematicalLanguage
  | LogicalLanguage

-- Define a function to measure intuitiveness
def intuitiveness (method : RepresentationMethod) : ℕ := 
  match method with
  | RepresentationMethod.NaturalLanguage => 1
  | RepresentationMethod.Flowchart => 4
  | RepresentationMethod.MathematicalLanguage => 2
  | RepresentationMethod.LogicalLanguage => 3

-- Theorem: Flowchart is the most intuitive method
theorem flowchart_most_intuitive :
  ∀ (method : RepresentationMethod),
    method ≠ RepresentationMethod.Flowchart →
    intuitiveness RepresentationMethod.Flowchart > intuitiveness method := by
  intro method h
  cases method
  all_goals (
    simp [intuitiveness]
    try rfl
    try contradiction
  )

-- The main theorem stating that Flowchart is the correct answer
theorem flowchart_is_correct_answer :
  ∃ (method : RepresentationMethod),
    (∀ (other : RepresentationMethod), intuitiveness method ≥ intuitiveness other) ∧
    method = RepresentationMethod.Flowchart := by
  use RepresentationMethod.Flowchart
  constructor
  · intro other
    simp [intuitiveness]
    cases other
    all_goals (simp [intuitiveness])
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flowchart_most_intuitive_flowchart_is_correct_answer_l59_5902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l59_5963

/-- A tuple of 6 non-negative integers satisfying the given system of equations -/
structure SolutionTuple where
  x₁ : ℕ
  x₂ : ℕ
  x₃ : ℕ
  x₄ : ℕ
  x₅ : ℕ
  x₆ : ℕ
  eq1 : x₁ * x₂ * (1 - x₃) = x₄ * x₅
  eq2 : x₂ * x₃ * (1 - x₄) = x₅ * x₆
  eq3 : x₃ * x₄ * (1 - x₅) = x₆ * x₁
  eq4 : x₄ * x₅ * (1 - x₆) = x₁ * x₂
  eq5 : x₅ * x₆ * (1 - x₁) = x₂ * x₃
  eq6 : x₆ * x₁ * (1 - x₂) = x₃ * x₄

/-- Characterization of solution tuples -/
inductive IsSolutionForm : SolutionTuple → Prop where
  | form1 (a b c d : ℕ+) (t : SolutionTuple) :
      t.x₁ = a * b ∧ t.x₂ = c * d ∧ t.x₃ = 0 ∧ t.x₄ = a * c ∧ t.x₅ = b * d ∧ t.x₆ = 0 →
      IsSolutionForm t
  | form2 (x y : ℕ) (t : SolutionTuple) :
      t.x₁ = 0 ∧ t.x₂ = x ∧ t.x₃ = 0 ∧ t.x₄ = 0 ∧ t.x₅ = y ∧ t.x₆ = 0 →
      IsSolutionForm t
  | form3 (x y : ℕ) (t : SolutionTuple) :
      t.x₁ = x ∧ t.x₂ = 0 ∧ t.x₃ = 0 ∧ t.x₄ = 0 ∧ t.x₅ = y ∧ t.x₆ = 0 →
      IsSolutionForm t
  | form4 (g h i : ℕ) (t : SolutionTuple) :
      t.x₁ = 0 ∧ t.x₂ = g ∧ t.x₃ = 0 ∧ t.x₄ = h ∧ t.x₅ = 0 ∧ t.x₆ = i →
      IsSolutionForm t
  | cyclic (t t' : SolutionTuple) :
      IsSolutionForm t' →
      t = ⟨t'.x₂, t'.x₃, t'.x₄, t'.x₅, t'.x₆, t'.x₁, t'.eq2, t'.eq3, t'.eq4, t'.eq5, t'.eq6, t'.eq1⟩ →
      IsSolutionForm t

/-- Main theorem: All solution tuples satisfy the characterization -/
theorem solution_characterization (t : SolutionTuple) : IsSolutionForm t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l59_5963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_approx_12_l59_5979

/-- Represents the work done by a group of men reaping acres of land -/
structure ReapingWork where
  men : ℕ
  days : ℕ
  acres : ℕ

/-- The first group's reaping work -/
def firstGroup : ReapingWork := { men := 0, days := 36, acres := 120 }

/-- The second group's reaping work -/
def secondGroup : ReapingWork := { men := 44, days := 54, acres := 660 }

/-- Work rate is proportional to the number of men and days worked -/
axiom work_rate_proportional (w : ReapingWork) : 
  (w.men : ℚ) * w.days / w.acres = (secondGroup.men : ℚ) * secondGroup.days / secondGroup.acres

/-- The theorem stating that the number of men in the first group is approximately 12 -/
theorem first_group_size_approx_12 : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |((firstGroup.men : ℚ) - 12)| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_approx_12_l59_5979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l59_5982

open Real

/-- The integrand function --/
noncomputable def f (x : ℝ) : ℝ := (((1 + x^(2/3))^2)^(1/3)) / (x^2 * x^(1/9))

/-- The antiderivative function --/
noncomputable def F (x : ℝ) : ℝ := -(9/10) * ((((1 + x^(2/3)) / x^(2/3))^(1/3))^5)

theorem integral_proof (x : ℝ) (hx : x > 0) : 
  deriv F x = f x := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l59_5982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_l59_5995

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_inequality {a x y : ℝ} (ha : 0 < a) (ha2 : a < 1) (h : a^x < a^y) : 
  floor x ≥ floor y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_l59_5995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l59_5955

/-- The length of each circular arc -/
noncomputable def arc_length : ℝ := Real.pi / 2

/-- The side length of the regular octagon -/
def octagon_side : ℝ := 3

/-- The number of circular arcs -/
def num_arcs : ℕ := 8

/-- The area enclosed by the curve composed of circular arcs -/
noncomputable def enclosed_area : ℝ := 54 + 54 * Real.sqrt 2 + 2 * Real.pi

/-- Theorem stating that the area enclosed by the curve is equal to 54 + 54√2 + 2π -/
theorem enclosed_area_theorem :
  enclosed_area = 54 + 54 * Real.sqrt 2 + 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l59_5955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l59_5905

noncomputable def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

noncomputable def slope1 (a : ℝ) : ℝ := -a

noncomputable def slope2 (a : ℝ) : ℝ := -4 / (a - 3)

theorem perpendicular_lines_a_value (a : ℝ) (h : a ≠ 3) :
  perpendicular (slope1 a) (slope2 a) → a = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l59_5905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_million_scientific_notation_l59_5933

-- Define scientific notation
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

-- State the theorem
theorem hundred_million_scientific_notation :
  (100000000 : ℝ) = scientific_notation 1 8 := by
  -- Convert 100000000 to real number
  have h1 : (100000000 : ℝ) = (100000000 : ℕ)
  · rfl
  
  -- Expand the definition of scientific_notation
  have h2 : scientific_notation 1 8 = 1 * (10 : ℝ) ^ 8
  · rfl
  
  -- Evaluate 10^8
  have h3 : (10 : ℝ) ^ 8 = 100000000
  · norm_num
  
  -- Rewrite using the above facts
  rw [h1, h2, h3]
  
  -- Simplify
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_million_scientific_notation_l59_5933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_value_l59_5950

theorem tan_2alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin (2 * α) = -Real.sin α) : 
  Real.tan (2 * α) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_value_l59_5950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_upper_bound_factorial_power_l59_5906

theorem smallest_upper_bound_factorial_power :
  ∃ (C : ℕ), (∀ (n : ℕ), n > 0 → (6 : ℝ)^(n : ℝ) / (Nat.factorial n : ℝ) ≤ (C : ℝ)) ∧
  (∀ (D : ℕ), D < C → ∃ (k : ℕ), k > 0 ∧ (6 : ℝ)^(k : ℝ) / (Nat.factorial k : ℝ) > (D : ℝ)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_upper_bound_factorial_power_l59_5906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recycle_definition_l59_5981

def recycle : String := "to reuse, to recycle"

theorem recycle_definition : recycle = "to reuse, to recycle" := by
  rfl

#check recycle_definition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recycle_definition_l59_5981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_knight_sum_l59_5990

/-- Represents a square on the chessboard -/
structure Square where
  row : Nat
  col : Nat
  value : Nat

/-- The chessboard -/
def Chessboard : Type := Array (Array Square)

/-- Creates a chessboard with numbers 1 to 64 -/
def createChessboard : Chessboard :=
  sorry

/-- Checks if two squares are on the same color -/
def sameColor (s1 s2 : Square) : Bool :=
  (s1.row + s1.col) % 2 = (s2.row + s2.col) % 2

/-- Sum of all squares of one color -/
def sumOneColor (board : Chessboard) (isBlack : Bool) : Nat :=
  sorry

/-- Theorem: The maximum sum of numbers on squares where knights can be placed without attacking each other is 1056 -/
theorem max_knight_sum (board : Chessboard) : 
  max (sumOneColor board true) (sumOneColor board false) = 1056 := by
  sorry

#eval max 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_knight_sum_l59_5990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_time_difference_l59_5966

-- Define the days Alice cycles
inductive CyclingDay
| monday
| wednesday
| friday
| saturday

-- Define a function for the distance cycled on each day
def distance : CyclingDay → ℚ
| CyclingDay.monday => 3
| CyclingDay.wednesday => 4
| CyclingDay.friday => 2
| CyclingDay.saturday => 2

-- Define a function for the speed on each day
def speed : CyclingDay → ℚ
| CyclingDay.monday => 6
| CyclingDay.wednesday => 4
| CyclingDay.friday => 4
| CyclingDay.saturday => 2

-- Calculate the total time spent cycling
noncomputable def actualTotalTime : ℚ :=
  (distance CyclingDay.monday / speed CyclingDay.monday +
   distance CyclingDay.wednesday / speed CyclingDay.wednesday +
   distance CyclingDay.friday / speed CyclingDay.friday +
   distance CyclingDay.saturday / speed CyclingDay.saturday) * 60

-- Calculate the total distance
def totalDistance : ℚ :=
  distance CyclingDay.monday + distance CyclingDay.wednesday +
  distance CyclingDay.friday + distance CyclingDay.saturday

-- Calculate the time if cycling at 3 mph
noncomputable def constantSpeedTime : ℚ :=
  (totalDistance / 3) * 60

-- Theorem to prove
theorem cycling_time_difference :
  constantSpeedTime - actualTotalTime = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_time_difference_l59_5966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_polygons_existence_l59_5962

theorem congruent_polygons_existence 
  (n p : ℕ) 
  (h1 : n ≥ 6) 
  (h2 : 4 ≤ p) 
  (h3 : p ≤ n / 2) :
  ∃ (k : ℕ) (red_polygon blue_polygon : Finset (Fin n)),
    k ≥ ⌊(p : ℚ) / 2⌋ + 1 ∧
    red_polygon.card = k ∧
    blue_polygon.card = k ∧
    (∀ v ∈ red_polygon, v.val < p) ∧
    (∀ v ∈ blue_polygon, v.val ≥ p) ∧
    ∃ σ : Equiv.Perm (Fin n), σ.symm '' red_polygon = blue_polygon :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_polygons_existence_l59_5962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_short_answer_test_type_l59_5954

/-- Represents the possible answers to the question -/
inductive Answer
  | Mixture
  | Collection
  | Compound
  | Compromise

/-- The correct answer is Compromise -/
def correctAnswer : Answer := Answer.Compromise

/-- Theorem stating that the correct answer is Compromise -/
theorem short_answer_test_type : correctAnswer = Answer.Compromise := by
  -- The proof is trivial since we defined correctAnswer as Compromise
  rfl

#check short_answer_test_type

end NUMINAMATH_CALUDE_ERRORFEEDBACK_short_answer_test_type_l59_5954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_distance_theorem_l59_5915

/-- The distance between two projectiles launched simultaneously and traveling towards each other -/
noncomputable def initial_distance (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 + speed2) * time / 60

theorem projectile_distance_theorem (speed1 speed2 time : ℝ) 
  (h1 : speed1 = 445)
  (h2 : speed2 = 545)
  (h3 : time = 84) :
  initial_distance speed1 speed2 time = 1385.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_distance_theorem_l59_5915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_expression_l59_5947

theorem negative_expression : 
  ∃ (a b c d : ℝ), 
    a = -(-2) ∧
    b = |(-2)| ∧
    c = (-2)^2 ∧
    d = (-2)^3 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_expression_l59_5947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_avg_mpg_l59_5965

/-- Calculates the average miles per gallon for a trip, rounded to the nearest tenth -/
def avgMilesPerGallon (initialOdometer finalOdometer : ℕ) (totalGasoline : ℚ) : ℚ :=
  let distance := finalOdometer - initialOdometer
  let mpg := (distance : ℚ) / totalGasoline
  (mpg * 10).floor / 10 + if (mpg * 10) % 1 ≥ 1/2 then 1/10 else 0

/-- Theorem stating that the average miles per gallon for the given trip is 14.6 -/
theorem trip_avg_mpg :
  avgMilesPerGallon 35400 36000 41 = 14.6 := by
  sorry

#eval avgMilesPerGallon 35400 36000 41

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_avg_mpg_l59_5965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_first_twelve_from_three_l59_5912

def first_twelve_from_three : List ℕ := List.range 12 |>.map (· + 3)

theorem median_of_first_twelve_from_three :
  let sorted_list := first_twelve_from_three
  let n := sorted_list.length
  let middle_index := n / 2
  (sorted_list.get! (middle_index - 1) + sorted_list.get! middle_index) / 2 = 17 / 2 := by
  sorry

#eval (17 : ℚ) / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_first_twelve_from_three_l59_5912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_1_5_unit_distance_l59_5904

/-- A point on the perimeter of a square -/
structure PerimeterPoint where
  x : ℚ
  y : ℚ

/-- The set of 12 points on the perimeter of a 3x3 square -/
def perimeterPoints : Finset PerimeterPoint := sorry

/-- The distance between two points -/
def distance (p1 p2 : PerimeterPoint) : ℚ := sorry

/-- The number of pairs of points that are 1.5 units apart -/
def favorablePairs : ℕ := sorry

/-- The set of all pairs of points -/
def allPairs : Finset (PerimeterPoint × PerimeterPoint) :=
  Finset.product perimeterPoints perimeterPoints

theorem probability_of_1_5_unit_distance : 
  (favorablePairs : ℚ) / ((Finset.card allPairs / 2) : ℚ) = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_1_5_unit_distance_l59_5904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l59_5924

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := Real.exp x + 1

-- Define the line
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y - 2| / Real.sqrt 2

-- State the theorem
theorem min_distance_curve_to_line :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), distance_to_line x (curve x) ≥ distance_to_line x₀ (curve x₀) ∧
  distance_to_line x₀ (curve x₀) = 2 * Real.sqrt 2 := by
  sorry

#check min_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l59_5924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisor_of_sequence_l59_5977

theorem max_divisor_of_sequence : 
  ∃ (m : ℕ), (∀ (k : ℕ), k > 0 → (5^k + 2 * 3^(k-1) + 1) % m = 0) ∧ 
  (∀ (d : ℕ), d > 0 → (∀ (k : ℕ), k > 0 → (5^k + 2 * 3^(k-1) + 1) % d = 0) → d ≤ m) ∧
  m = 8 :=
by
  -- We claim that m = 8 satisfies the conditions
  use 8
  constructor
  -- Prove that 8 divides the sequence for all k > 0
  · intro k hk
    sorry -- Proof omitted
  constructor
  -- Prove that 8 is the largest such divisor
  · intro d hd_pos hdiv
    sorry -- Proof omitted
  -- Prove that m = 8
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisor_of_sequence_l59_5977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_water_mixture_ratio_l59_5942

/-- Given two identical bottles with alcohol-to-water ratios of m:1 and n:1 respectively,
    when mixed, the resulting alcohol-to-water ratio is (m+n+2mn)/(m+n+2). -/
theorem alcohol_water_mixture_ratio (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (m + n + 2 * m * n) / (m + n + 2) = ((m / 1) + (n / 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_water_mixture_ratio_l59_5942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_value_l59_5983

noncomputable def z (θ : ℝ) : ℂ := 3 * Complex.cos θ + Complex.I * (2 * Complex.sin θ)

noncomputable def y (θ : ℝ) : ℝ := θ - Complex.arg (z θ)

theorem max_y_value (θ : ℝ) (h : 0 < θ ∧ θ < Real.pi / 2) :
  ∃ (θ_max : ℝ), θ_max = Real.arctan (Real.sqrt 6 / 2) ∧
  ∀ θ', 0 < θ' ∧ θ' < Real.pi / 2 → y θ' ≤ y θ_max :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_value_l59_5983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_number_after_transformations_l59_5997

def transformation (a : Vector ℤ 25) : Vector ℤ 25 :=
  Vector.ofFn (λ i => a[i] + a[(i + 1) % 25])

def initial_sequence : Vector ℤ 25 :=
  Vector.ofFn (λ i => if i < 13 then 1 else -1)

def iterate_transformation (n : ℕ) (a : Vector ℤ 25) : Vector ℤ 25 :=
  match n with
  | 0 => a
  | n + 1 => iterate_transformation n (transformation a)

theorem exists_large_number_after_transformations :
  ∃ x ∈ (iterate_transformation 100 initial_sequence).toList,
    x > 10^20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_number_after_transformations_l59_5997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l59_5998

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := ⌈x⌉ - 1/2

-- Theorem stating that g is neither even nor odd
theorem g_neither_even_nor_odd :
  (¬ ∀ x, g (-x) = g x) ∧ (¬ ∀ x, g (-x) = -g x) := by
  -- We'll use a sorry here as the proof is not implemented
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l59_5998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l59_5946

/-- Represents the principal amount invested -/
noncomputable def P : ℝ := sorry

/-- Represents the interest rate as a percentage -/
noncomputable def y : ℝ := sorry

/-- The simple interest earned over two years -/
noncomputable def simpleInterest : ℝ := P * y * 2 / 100

/-- The compound interest earned over two years -/
noncomputable def compoundInterest : ℝ := P * ((1 + y/100)^2 - 1)

theorem investment_problem :
  ∃ P y : ℝ,
    simpleInterest = 300 ∧
    compoundInterest = 307.50 ∧
    abs (P - 73.53) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l59_5946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_integral_polar_coords_l59_5989

open Set MeasureTheory Real

-- Define the region D
noncomputable def D : Set (ℝ × ℝ) := {p | 1 ≤ p.1^2 + p.2^2 ∧ p.1^2 + p.2^2 ≤ 4}

-- Define the integrand function
noncomputable def f (p : ℝ × ℝ) : ℝ := 1 / (p.1^2 + p.2^2)

-- State the theorem
theorem double_integral_polar_coords :
  ∫ p in D, f p = 2 * π * Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_integral_polar_coords_l59_5989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_circles_irrational_inscribed_triangles_l59_5974

/-- Given a circle with radius r, all triangles inscribed in this circle have at least one irrational side length -/
def all_inscribed_triangles_irrational_side (r : ℝ) : Prop :=
  ∀ a b c : ℝ, (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (r = (a * b * c) / (4 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)))) →
  (¬ ∃ q : ℚ, a = ↑q) ∨ (¬ ∃ q : ℚ, b = ↑q) ∨ (¬ ∃ q : ℚ, c = ↑q)

/-- There exists an infinite set of real numbers r such that all triangles inscribed in a circle with radius r have at least one irrational side length -/
theorem infinite_circles_irrational_inscribed_triangles :
  ∃ R : Set ℝ, Set.Infinite R ∧ ∀ r ∈ R, all_inscribed_triangles_irrational_side r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_circles_irrational_inscribed_triangles_l59_5974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_moment_of_inertia_oy_l59_5932

/-- Represents a hemisphere with radius R -/
structure Hemisphere (R : ℝ) where
  x : ℝ
  y : ℝ
  z : ℝ
  eq : x^2 + y^2 + z^2 = R^2
  nonneg : y ≥ 0

/-- Calculates the moment of inertia about the Oy axis for a hemisphere -/
noncomputable def momentOfInertiaOy (R : ℝ) : ℝ := (4/3) * Real.pi * R^3

/-- Theorem stating that the moment of inertia about the Oy axis of a hemisphere
    with radius R is equal to (4/3) * π * R^3 -/
theorem hemisphere_moment_of_inertia_oy (R : ℝ) (h : R > 0) :
  momentOfInertiaOy R = (4/3) * Real.pi * R^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_moment_of_inertia_oy_l59_5932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_roots_of_unity_l59_5945

theorem fourth_roots_of_unity (x : ℂ) : x^4 = 1 ↔ x ∈ ({1, -1, Complex.I, -Complex.I} : Set ℂ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_roots_of_unity_l59_5945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l59_5940

-- Define the function f as noncomputable due to its dependency on real numbers
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + 4*a else a/x

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ,
    (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) < 0) ↔
    (a ∈ Set.Icc (1/5 : ℝ) (1/2 : ℝ) ∧ a ≠ 1/2)) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l59_5940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_18th_innings_proof_l59_5913

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  averageRuns : Rat

/-- Represents the target and required run rate for a cricket match -/
structure MatchTarget where
  targetRuns : Nat
  totalOvers : Nat
  requiredRunRate : Rat

/-- Calculates the number of runs needed in the next innings to achieve a given average increase -/
def runsNeededForAverageIncrease (stats : BatsmanStats) (increase : Rat) : Nat :=
  ((stats.averageRuns + increase) * (stats.innings + 1 : Rat) - stats.totalRuns).ceil.toNat

/-- Checks if a given number of runs can be scored within a certain number of overs at a rate higher than the required run rate -/
def canMaintainHigherRunRate (runs : Nat) (overs : Nat) (requiredRunRate : Rat) : Prop :=
  (runs : Rat) / (overs : Rat) > requiredRunRate

theorem batsman_18th_innings_proof 
  (stats : BatsmanStats)
  (target : MatchTarget)
  (h1 : stats.innings = 17)
  (h2 : stats.totalRuns = (stats.averageRuns * 17).floor + 85)
  (h3 : stats.averageRuns = ((stats.totalRuns - 85 : Rat) / 16 + 3)) :
  let runsNeeded := runsNeededForAverageIncrease stats 2
  runsNeeded = 73 ∧ canMaintainHigherRunRate runsNeeded 9 target.requiredRunRate := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_18th_innings_proof_l59_5913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_balanced_lines_l59_5957

/-- A point in a plane with a color -/
structure ColoredPoint where
  x : ℝ
  y : ℝ
  color : Bool  -- True for blue, False for red

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line is balanced -/
def isBalanced (line : Line) (points : List ColoredPoint) : Bool :=
  sorry

theorem two_balanced_lines 
  (n : ℕ) 
  (k : ℕ) 
  (points : List ColoredPoint) 
  (h1 : points.length = n + 1)
  (h2 : (points.filter (fun p => p.color)).length = k)
  (h3 : (points.filter (fun p => ¬p.color)).length = n) :
  ∃ (l1 l2 : Line), l1 ≠ l2 ∧ isBalanced l1 points ∧ isBalanced l2 points := by
  sorry

#check two_balanced_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_balanced_lines_l59_5957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_set_l59_5927

/-- The set M of points (x, y) satisfying 27^x = (1/9) * 3^y -/
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (27 : ℝ)^p.1 = (1/9) * (3 : ℝ)^p.2}

/-- Theorem stating that the point (1, 5) belongs to set M -/
theorem point_in_set : (1, 5) ∈ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_set_l59_5927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l59_5900

/-- A function representing an inverse proportion with parameter m -/
noncomputable def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := (2*m - 3) / x

/-- Predicate to check if a function is in the first and third quadrants -/
def in_first_and_third_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x > 0 → f x > 0) ∧ (x < 0 → f x < 0)

/-- Theorem stating that if the inverse proportion function with parameter m
    is in the first and third quadrants, then m > 3/2 -/
theorem inverse_proportion_quadrants (m : ℝ) :
  in_first_and_third_quadrants (inverse_proportion m) → m > 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l59_5900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_vector_sum_l59_5959

theorem complex_vector_sum (z1 z2 z3 : ℂ) (lambda mu : ℝ) : 
  z1 = -1 + 2*I → 
  z2 = 1 - I → 
  z3 = 3 - 4*I → 
  z3 = lambda * z1 + mu * z2 → 
  lambda + mu = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_vector_sum_l59_5959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_hemisphere_equal_volume_lateral_area_l59_5951

variable (R : ℝ)

/-- The volume of a hemisphere with radius R -/
noncomputable def hemisphere_volume (R : ℝ) : ℝ := (2 / 3) * Real.pi * R^3

/-- The volume of a cone with base radius R and height h -/
noncomputable def cone_volume (R h : ℝ) : ℝ := (1 / 3) * Real.pi * R^2 * h

/-- The lateral surface area of a cone with base radius R and slant height l -/
noncomputable def cone_lateral_area (R l : ℝ) : ℝ := Real.pi * R * l

/-- 
For a cone and hemisphere sharing a common base of radius R, 
if their volumes are equal, then the lateral surface area of the cone is π R² √5.
-/
theorem cone_hemisphere_equal_volume_lateral_area (R : ℝ) (h_pos : R > 0) :
  ∃ h, cone_volume R h = hemisphere_volume R →
  ∃ l, l^2 = R^2 + h^2 ∧ cone_lateral_area R l = Real.pi * R^2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_hemisphere_equal_volume_lateral_area_l59_5951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l59_5987

/-- The equation satisfied by x and y -/
def equation (x y : ℕ+) : Prop :=
  (x : ℝ)^4 * (y : ℝ)^4 - 16 * (x : ℝ)^2 * (y : ℝ)^2 + 15 = 0

/-- The set of all ordered pairs (x,y) satisfying the equation -/
def solution_set : Set (ℕ+ × ℕ+) :=
  {p | equation p.1 p.2}

/-- There exists exactly one distinct ordered pair of positive integers satisfying the equation -/
theorem unique_solution : ∃! p : ℕ+ × ℕ+, p ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l59_5987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_line_not_through_center_l59_5930

/-- The line l: kx - y - k + 1 = 0 -/
def line (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y - k + 1 = 0

/-- The circle C: (x - 1)² + y² = 4 -/
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 4

/-- The minimum length of the chord formed by the intersection of l and C is 2√3 -/
theorem min_chord_length (k : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12 ∧
    ∀ (x y : ℝ), line k x y ∧ circle_C x y →
      (x - x₁)^2 + (y - y₁)^2 ≥ 12 ∧ (x - x₂)^2 + (y - y₂)^2 ≥ 12 :=
by sorry

/-- For any real k, l does not pass through the center of C -/
theorem line_not_through_center (k : ℝ) :
  ¬ (line k 1 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_line_not_through_center_l59_5930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_by_eight_sixteen_colors_impossible_l59_5922

theorem eight_by_eight_sixteen_colors_impossible :
  ¬ ∃ (coloring : Fin 8 → Fin 8 → Fin 16),
    ∀ (c1 c2 : Fin 16), c1 ≠ c2 → 
      ∃ (i j : Fin 8),
        (coloring i j = c1 ∧ coloring (i + 1) j = c2) ∨
        (coloring i j = c1 ∧ coloring i (j + 1) = c2) ∨
        (coloring i j = c2 ∧ coloring (i + 1) j = c1) ∨
        (coloring i j = c2 ∧ coloring i (j + 1) = c1) := by
  sorry

#check eight_by_eight_sixteen_colors_impossible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_by_eight_sixteen_colors_impossible_l59_5922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_region_C_l59_5925

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def in_region_C (x y : ℝ) : Prop :=
  (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0) ∨ (x < 0 ∧ y < 0 ∧ x + y ≥ -1)

theorem floor_inequality_region_C (x y : ℝ) :
  |x| < 1 → |y| < 1 → x * y ≠ 0 →
  (floor (x + y) ≤ floor x + floor y ↔ in_region_C x y) := by
  sorry

#check floor_inequality_region_C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_region_C_l59_5925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_negative_420_degrees_l59_5937

-- Define secant function
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Define the period of cosine
noncomputable def cosine_period : ℝ := 2 * Real.pi

-- Theorem statement
theorem sec_negative_420_degrees : sec ((-420 * Real.pi) / 180) = 2 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_negative_420_degrees_l59_5937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l59_5920

theorem diophantine_equation_solutions (x y m n : ℕ) (p : ℕ) :
  Nat.Prime p →
  x > 0 →
  y > 0 →
  m > 0 →
  n > 0 →
  x + y^2 = p^m →
  x^2 + y = p^n →
  ((x = 1 ∧ y = 1 ∧ p = 2 ∧ m = 1 ∧ n = 1) ∨
   (x = 5 ∧ y = 2 ∧ p = 3 ∧ m = 2 ∧ n = 3) ∨
   (x = 2 ∧ y = 5 ∧ p = 3 ∧ m = 3 ∧ n = 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l59_5920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_a_formula_T_formula_l59_5984

def sequence_a (n : ℕ+) : ℚ := 2 * n.val - 1

def sum_S (n : ℕ+) : ℚ := (Finset.range n.val).sum (λ i => sequence_a ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_property (n : ℕ+) :
  (sequence_a n)^2 + 2 * (sequence_a n) = 4 * (sum_S n) - 1 :=
sorry

theorem a_formula (n : ℕ+) : sequence_a n = 2 * n.val - 1 :=
by rfl

noncomputable def sequence_b (n : ℕ+) : ℚ := 3^(n.val - 1)

noncomputable def sequence_ratio (n : ℕ+) : ℚ := sequence_a n / sequence_b n

noncomputable def sum_T (n : ℕ+) : ℚ := 
  (Finset.range n.val).sum (λ i => sequence_ratio ⟨i + 1, Nat.succ_pos i⟩)

theorem T_formula (n : ℕ+) : sum_T n = 3 - (n.val + 1 : ℚ) / 3^(n.val - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_a_formula_T_formula_l59_5984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_satisfying_functions_l59_5934

/-- A function from integers to integers satisfying the given property -/
def SatisfyingFunction (f : ℤ → ℤ) : Prop :=
  ∀ h k : ℤ, f (h + k) + f (h * k) = f h * f k + 1

/-- The set of all functions satisfying the property -/
def SatisfyingFunctions : Set (ℤ → ℤ) :=
  {f | SatisfyingFunction f}

/-- The theorem stating that there are exactly 3 satisfying functions -/
theorem exactly_three_satisfying_functions :
  ∃ (s : Finset (ℤ → ℤ)), s.card = 3 ∧ ∀ f, f ∈ s ↔ SatisfyingFunction f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_satisfying_functions_l59_5934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_diagonal_opposite_larger_angle_l59_5931

/-- A parallelogram with vertices A, B, C, and D -/
structure Parallelogram (A B C D : ℝ × ℝ) : Prop where
  is_parallelogram : True -- Placeholder, replace with actual conditions if needed

/-- The angle ABC in a parallelogram -/
noncomputable def angle_ABC (p : Parallelogram A B C D) : ℝ := sorry

/-- The angle BAD in a parallelogram -/
noncomputable def angle_BAD (p : Parallelogram A B C D) : ℝ := sorry

/-- The length of diagonal AC in a parallelogram -/
noncomputable def diagonal_AC (p : Parallelogram A B C D) : ℝ := sorry

/-- The length of diagonal BD in a parallelogram -/
noncomputable def diagonal_BD (p : Parallelogram A B C D) : ℝ := sorry

/-- Theorem: In a parallelogram, the larger diagonal lies opposite the larger angle -/
theorem larger_diagonal_opposite_larger_angle 
  (A B C D : ℝ × ℝ) 
  (p : Parallelogram A B C D) 
  (h : angle_ABC p > angle_BAD p) : 
  diagonal_AC p > diagonal_BD p := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_diagonal_opposite_larger_angle_l59_5931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_is_4_sqrt_6_l59_5919

-- Define the given parameters
def circle_radius : ℝ := 8
def num_sectors : ℕ := 4
def slant_height : ℝ := 10

-- Define the cone's base radius
noncomputable def base_radius : ℝ := circle_radius * Real.pi / num_sectors

-- Define the height of the cone
noncomputable def cone_height : ℝ := Real.sqrt (slant_height^2 - base_radius^2)

-- Theorem statement
theorem cone_height_is_4_sqrt_6 : 
  cone_height = 4 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_is_4_sqrt_6_l59_5919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_curves_l59_5986

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The distance between two points in polar coordinates -/
def polarDistance (p1 p2 : PolarPoint) : ℝ :=
  |p1.r - p2.r|

/-- Curve C₁ with equation ρ = 4sinθ -/
noncomputable def C₁ (θ : ℝ) : PolarPoint :=
  { r := 4 * Real.sin θ, θ := θ }

/-- Curve C₂ with equation ρ = 8sinθ -/
noncomputable def C₂ (θ : ℝ) : PolarPoint :=
  { r := 8 * Real.sin θ, θ := θ }

/-- Theorem stating that the distance between points on C₁ and C₂ at any angle θ is 4sinθ -/
theorem distance_between_curves (θ : ℝ) :
  polarDistance (C₁ θ) (C₂ θ) = 4 * Real.sin θ := by
  sorry

#check distance_between_curves

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_curves_l59_5986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l59_5911

/-- The angle of inclination of a line given by the equation 2ax + (a² + 1)y - 1 = 0 -/
noncomputable def angle_of_inclination (a : ℝ) : ℝ :=
  Real.arctan (-2 * a / (a^2 + 1))

/-- The set representing the range of the angle of inclination -/
def angle_range : Set ℝ :=
  Set.Icc 0 (Real.pi / 4) ∪ Set.Ico (3 * Real.pi / 4) Real.pi

/-- Theorem stating that the range of the angle of inclination function
    is equal to the defined angle_range set -/
theorem angle_of_inclination_range :
  Set.range angle_of_inclination = angle_range := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l59_5911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_square_of_linear_l59_5994

noncomputable def G (x m : ℝ) : ℝ := (8 * x^2 + 24 * x + 5 * m) / 8

def LinearExpr (x a b : ℝ) : ℝ := a * x + b

theorem G_is_square_of_linear (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, G x m = (LinearExpr x a b)^2) → 
  (3 < m ∧ m < 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_square_of_linear_l59_5994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_cardinality_l59_5928

def A : Finset ℕ := {1, 2, 3, 4}
def B : Finset ℕ := {0, 1, 2, 4, 5}
def U : Finset ℕ := A ∪ B

theorem complement_intersection_cardinality :
  (U \ (A ∩ B)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_cardinality_l59_5928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_six_primes_mod_seventh_prime_l59_5943

-- Define a function to get the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Define a function to sum the first n prime numbers
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_first_six_primes_mod_seventh_prime :
  (sumFirstNPrimes 6) % (nthPrime 7) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_six_primes_mod_seventh_prime_l59_5943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_corner_sum_is_17_l59_5992

-- Define the type for a cube face
def Face := Fin 9

-- Define the type for a cube
structure Cube where
  faces : List Face
  opposite_sum : ∀ (f : Face), f ∈ faces → ∃ (g : Face), g ∈ faces ∧ f.val + g.val = 9
  distinct : ∀ (f g : Face), f ∈ faces → g ∈ faces → f ≠ g
  positive : ∀ (f : Face), f ∈ faces → f.val > 0
  six_faces : faces.length = 6

-- Define a function to get the maximum sum of three adjacent faces
def max_corner_sum (c : Cube) : ℕ := sorry

-- Theorem statement
theorem max_corner_sum_is_17 (c : Cube) : max_corner_sum c = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_corner_sum_is_17_l59_5992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_weight_l59_5935

/-- Calculates the total weight of a water tank and its contents -/
theorem water_tank_weight (tank_capacity : ℝ) (empty_tank_weight : ℝ) (fill_percentage : ℝ) (water_weight_per_gallon : ℝ) :
  tank_capacity = 200 →
  empty_tank_weight = 80 →
  fill_percentage = 0.8 →
  water_weight_per_gallon = 8 →
  empty_tank_weight + (tank_capacity * fill_percentage * water_weight_per_gallon) = 1360 := by
  intro h1 h2 h3 h4
  sorry

#check water_tank_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_weight_l59_5935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_DEF_l59_5973

/-- The diameter of the inscribed circle in a triangle with sides a, b, and c -/
noncomputable def inscribed_circle_diameter (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  2 * (2 * Real.sqrt (s * (s - a) * (s - b) * (s - c))) / (a + b + c)

/-- Theorem: The diameter of the inscribed circle in triangle DEF is 2√14 -/
theorem inscribed_circle_diameter_DEF :
  inscribed_circle_diameter 13 8 9 = 2 * Real.sqrt 14 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_DEF_l59_5973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l59_5961

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem first_term_of_geometric_sequence 
  (a : ℝ) (r : ℝ) 
  (h1 : geometric_sequence a r 5 = Nat.factorial 9) 
  (h2 : geometric_sequence a r 8 = Nat.factorial 11) : 
  a = 362880 / (110 ^ (1/3 : ℝ))^4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l59_5961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_functions_have_property_φ_l59_5926

-- Define the set A₀
def A₀ : Set ℝ := {x | 0 < x ∧ x < 1}

-- Define the recursive set Aₙ
def A (n : ℕ) (f : ℝ → ℝ) : Set ℝ := 
  match n with
  | 0 => A₀
  | n+1 => {y | ∃ x ∈ A n f, y = f x}

-- Define the property φ
def has_property_φ (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ+, (A n f) ∩ (A (n-1) f) = ∅

-- Define the four functions
noncomputable def f₁ : ℝ → ℝ := λ x ↦ 1/x
def f₂ : ℝ → ℝ := λ x ↦ x^2 + 1
def f₃ : ℝ → ℝ := λ x ↦ x
noncomputable def f₄ : ℝ → ℝ := λ x ↦ 2^x

-- Theorem statement
theorem three_functions_have_property_φ : 
  (has_property_φ f₁ ∧ has_property_φ f₂ ∧ ¬has_property_φ f₃ ∧ has_property_φ f₄) := by
  sorry

#check three_functions_have_property_φ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_functions_have_property_φ_l59_5926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_equals_half_compound_interest_l59_5975

/-- Calculates the compound interest for a given principal, rate, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Calculates the simple interest for a given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- The principal amount that satisfies the given conditions -/
def P : ℝ := 1272

theorem simple_interest_equals_half_compound_interest :
  simple_interest P 10 5 = (1/2) * compound_interest 5000 12 2 := by
  sorry

#eval P

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_equals_half_compound_interest_l59_5975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l59_5985

noncomputable def area_between_circles_and_square (π : ℝ) (r₁ r₂ : ℝ) : ℝ :=
  let square_side := 2 * r₁
  let square_area := square_side ^ 2
  let large_circle_area := π * r₁ ^ 2
  let small_circle_area := π * r₂ ^ 2
  let area_between_circles := large_circle_area - small_circle_area
  let area_outside_small_circle := square_area - small_circle_area
  area_between_circles + area_outside_small_circle

theorem area_calculation :
  area_between_circles_and_square Real.pi 12 7 = 46 * Real.pi + 576 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l59_5985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l59_5988

-- Define the angle α
variable (α : Real)

-- Define the point P
noncomputable def P : ℝ × ℝ := (-2 * Real.cos (30 * Real.pi / 180), 2 * Real.sin (30 * Real.pi / 180))

-- State the theorem
theorem sin_alpha_value (h : P = (Real.cos α, Real.sin α)) : 
  Real.sin α = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l59_5988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_roots_l59_5953

theorem max_product_roots (p : ℝ) : 
  (∀ x : ℝ, 5 * x^2 - 6 * x + p = 0 → x ∈ Set.univ) ∧ 
  (∀ q : ℝ, (∀ x : ℝ, 5 * x^2 - 6 * x + q = 0 → x ∈ Set.univ) → p ≥ q) →
  p = 1.8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_roots_l59_5953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l59_5964

theorem sufficient_not_necessary_condition (α : Real) (k : Int) :
  (∃ k, α = Real.pi / 6 + 2 * k * Real.pi) → Real.sin α = 1 / 2 ∧
  ¬(Real.sin α = 1 / 2 → ∃ k, α = Real.pi / 6 + 2 * k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l59_5964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_a_in_range_l59_5938

def has_unique_real_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, (0 < a * x + 1) ∧ (0 < x - a) ∧ (0 < 2 - x) ∧ 
    Real.log (a * x + 1) = Real.log (x - a) + Real.log (2 - x)

theorem unique_solution_iff_a_in_range :
  ∀ a : ℝ, has_unique_real_solution a ↔ -1/2 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_a_in_range_l59_5938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_simplification_and_addition_l59_5960

theorem complex_simplification_and_addition :
  (3 + 5*Complex.I) / (-2 + 3*Complex.I) + (1 - 2*Complex.I) = 
  Complex.ofReal (-8/13) + Complex.ofReal (-45/13) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_simplification_and_addition_l59_5960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_five_value_l59_5901

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

noncomputable def lg (x : ℝ) : ℝ := log 10 x

theorem lg_five_value (a b : ℝ) (h1 : log 8 3 = a) (h2 : log 3 5 = b) :
  lg 5 = (3 * a * b) / (1 + 3 * a * b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_five_value_l59_5901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l59_5956

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

-- State the theorem
theorem f_composition (x : ℝ) (h : -1 < x ∧ x < 1) : 
  f ((3*x + x^3) / (1 + 3*x^2)) = 3 * f x := by
  -- The proof steps would go here, but for now we'll use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l59_5956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l59_5991

/-- A partition of positive integers into two sets -/
def Partition := (Set ℕ) × (Set ℕ)

/-- Predicate to check if a partition is valid according to the problem conditions -/
def is_valid_partition (p : Partition) (a b : ℕ) : Prop :=
  let (H₁, H₂) := p
  (H₁ ∪ H₂ = Set.univ) ∧ 
  (H₁ ∩ H₂ = ∅) ∧
  (∀ x y, x ∈ H₁ → y ∈ H₁ → x - y ≠ a ∧ x - y ≠ b) ∧
  (∀ x y, x ∈ H₂ → y ∈ H₂ → x - y ≠ a ∧ x - y ≠ b)

/-- Main theorem statement -/
theorem partition_theorem (a b : ℕ) :
  (∃ p : Partition, is_valid_partition p a b) ↔ 
  (∃ n a' b' : ℕ, a = 2^n * a' ∧ b = 2^n * b' ∧ Odd a' ∧ Odd b') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l59_5991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_theorem_l59_5971

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The focus of an ellipse -/
noncomputable def focus (e : Ellipse) : Point :=
  { x := Real.sqrt (e.a^2 - e.b^2), y := 0 }

/-- Negation of a point -/
instance : Neg Point where
  neg p := { x := -p.x, y := -p.y }

theorem ellipse_line_theorem (e : Ellipse) (l : Line) :
  e.a = 2 ∧ e.b = 1 →
  (∃ (A B : Point),
    -- A and B are on the ellipse
    (A.x^2 / 4 + A.y^2 = 1) ∧ (B.x^2 / 4 + B.y^2 = 1) ∧
    -- A and B are on the line l
    (l.a * A.x + l.b * A.y + l.c = 0) ∧ (l.a * B.x + l.b * B.y + l.c = 0) ∧
    -- Line l passes through the left focus
    (l.a * (-focus e).x + l.b * (-focus e).y + l.c = 0) ∧
    -- |AF₂|, |AB|, and |BF₂| form an arithmetic sequence
    (distance A (focus e) + distance B (focus e) = 2 * distance A B)) →
  (l.a = 1 ∧ l.b = -Real.sqrt 5 ∧ l.c = Real.sqrt 3) ∨
  (l.a = 1 ∧ l.b = Real.sqrt 5 ∧ l.c = Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_theorem_l59_5971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_34_implies_m_equals_negative_3_l59_5903

theorem integral_equals_34_implies_m_equals_negative_3 :
  (∫ x in (2 : ℝ)..(3 : ℝ), (3 * x^2 - 2 * m * x)) = 34 → m = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_34_implies_m_equals_negative_3_l59_5903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_equivalence_l59_5969

noncomputable def original_function (x : ℝ) : ℝ := 3 * Real.sin x

noncomputable def transformed_function (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 5)

def halve_x_coordinate (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (2 * x)

def shift_left (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f (x + shift)

theorem transform_equivalence :
  ∀ x, transformed_function x = (shift_left (halve_x_coordinate original_function) (Real.pi / 10)) x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_equivalence_l59_5969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_five_equals_four_l59_5941

-- Define the function f as noncomputable
noncomputable def f : ℝ → ℝ := fun x => ((x - 1) / 2) ^ 2

-- State the theorem
theorem f_of_five_equals_four : f 5 = 4 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_five_equals_four_l59_5941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_congruent_in_pairs_l59_5921

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the centroid of a triangle -/
noncomputable def centroid (t : Triangle) : Point := sorry

/-- Represents a smaller triangle formed by the centroid and two points of the original triangle -/
structure SmallTriangle (t : Triangle) where
  X : Point
  Y : Point

/-- Returns the six smaller triangles formed by connecting the centroid to the vertices and midpoints -/
noncomputable def smallerTriangles (t : Triangle) : List (SmallTriangle t) := sorry

/-- Congruence relation between two SmallTriangles -/
def congruent (t1 t2 : SmallTriangle t) : Prop := sorry

/-- Main theorem: The six resulting triangles are congruent in opposite pairs -/
theorem triangles_congruent_in_pairs (t : Triangle) :
  ∃ (p1 p2 p3 : SmallTriangle t × SmallTriangle t),
    p1.1 ≠ p1.2 ∧ p2.1 ≠ p2.2 ∧ p3.1 ≠ p3.2 ∧
    congruent p1.1 p1.2 ∧ congruent p2.1 p2.2 ∧ congruent p3.1 p3.2 ∧
    (smallerTriangles t).length = 6 ∧
    (smallerTriangles t) = [p1.1, p1.2, p2.1, p2.2, p3.1, p3.2] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_congruent_in_pairs_l59_5921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_defective_five_two_l59_5976

/-- The probability of selecting exactly one defective item from a batch of products. -/
def probability_one_defective (total : ℕ) (defective : ℕ) (chosen : ℕ) : ℚ :=
  let quality := total - defective
  let favorable_outcomes := (Nat.choose defective 1) * (Nat.choose quality 1)
  let total_outcomes := Nat.choose total chosen
  (favorable_outcomes : ℚ) / total_outcomes

/-- Theorem stating the probability of selecting exactly one defective item
    when choosing 2 products from a batch of 5 products containing 2 defective items. -/
theorem probability_one_defective_five_two :
  probability_one_defective 5 2 2 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_defective_five_two_l59_5976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l59_5944

theorem distance_calculation (D : ℝ) 
  (h1 : D / 2 / 35 + D / 2 / 15 = 6) 
  (h2 : D / 10 = (D + 50) / (D / 2 / 35 + D / 2 / 15)) : 
  abs ((D + 50) - 54.34) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l59_5944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_5_l59_5914

/-- Definition of a geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a 1 else a 1 * (1 - q^n) / (1 - q)

theorem geometric_sum_5 :
  ∀ a : ℕ → ℝ,
  geometric_sequence a 4 →
  a 1 = 3 →
  geometric_sum a 4 5 = 1023 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_5_l59_5914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_properties_l59_5970

def data_set : Finset ℚ := {2, 3, 5, 7}

noncomputable def mode (s : Finset ℚ) : ℚ := 3

noncomputable def median (s : Finset ℚ) : ℚ := 3

noncomputable def mean (s : Finset ℚ) : ℚ := 4

theorem data_set_properties :
  ∃ x : ℚ, x ∈ data_set ∪ {3} ∧
  mode (data_set ∪ {3}) = 3 ∧
  median (data_set ∪ {3}) = 3 ∧
  mean (data_set ∪ {3}) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_properties_l59_5970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l59_5909

-- Define the custom operation
noncomputable def customOp (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := customOp 1 (2^x)

-- Theorem statement
theorem max_value_of_f :
  (∀ x : ℝ, f x ≤ 1) ∧ (∃ x : ℝ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l59_5909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l59_5980

/-- A complex number with modulus 5 in the third quadrant -/
def z : ℂ := -3 - 4*Complex.I

/-- The modulus of a complex number -/
noncomputable def modulus (c : ℂ) : ℝ := Real.sqrt (c.re^2 + c.im^2)

/-- Predicate for a complex number being in the third quadrant -/
def in_third_quadrant (c : ℂ) : Prop := c.re < 0 ∧ c.im < 0

/-- Theorem stating that z has modulus 5 and is in the third quadrant -/
theorem z_properties : modulus z = 5 ∧ in_third_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l59_5980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_approximation_l59_5999

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := Real.cos (100 * x^2)

-- Define the integral
noncomputable def integral : ℝ := ∫ x in (Set.Icc 0 0.1), f x

-- Define the approximation
def approximation : ℝ := 0.090

-- Define the accuracy
def α : ℝ := 0.001

-- Theorem statement
theorem integral_approximation : |integral - approximation| < α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_approximation_l59_5999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_beacons_necessary_and_sufficient_a1_d3_a5_achieve_unique_identification_l59_5952

/-- Represents a room in the maze --/
structure Room where
  x : Nat
  y : Nat
deriving Repr

/-- Represents the maze as a simple structure --/
structure Maze where
  rooms : List Room
deriving Repr

/-- Calculates the distance between two rooms in the maze --/
def distance (maze : Maze) (r1 r2 : Room) : Nat :=
  sorry

/-- Checks if a room's location can be uniquely determined by its distances to the beacons --/
def isUniquelyIdentifiable (maze : Maze) (beacons : List Room) (room : Room) : Bool :=
  sorry

/-- Checks if all rooms in the maze are uniquely identifiable given a set of beacons --/
def allRoomsUniquelyIdentifiable (maze : Maze) (beacons : List Room) : Bool :=
  sorry

/-- The main theorem stating that 3 beacons are necessary and sufficient --/
theorem three_beacons_necessary_and_sufficient (maze : Maze) :
  ∃ (beacons : List Room),
    beacons.length = 3 ∧
    allRoomsUniquelyIdentifiable maze beacons ∧
    ∀ (otherBeacons : List Room),
      otherBeacons.length < 3 →
      ¬ allRoomsUniquelyIdentifiable maze otherBeacons :=
by
  sorry

/-- Theorem stating that placing beacons in a1, d3, and a5 achieves unique identification --/
theorem a1_d3_a5_achieve_unique_identification (maze : Maze) :
  let beacons := [{ x := 1, y := 1 }, { x := 4, y := 3 }, { x := 1, y := 5 }]
  allRoomsUniquelyIdentifiable maze beacons :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_beacons_necessary_and_sufficient_a1_d3_a5_achieve_unique_identification_l59_5952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_angle_through_point_l59_5929

theorem sin_plus_cos_for_angle_through_point :
  ∀ α : ℝ,
  (∃ (x y : ℝ), x = 3 ∧ y = -4 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α + Real.cos α = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_angle_through_point_l59_5929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l59_5949

open Real

noncomputable def f (x : ℝ) : ℝ := 2^(sin x)

theorem problem_statement :
  (∃ x₁ x₂, x₁ ∈ Set.Ioo 0 π ∧ x₂ ∈ Set.Ioo 0 π ∧ f x₁ + f x₂ = 2) ∨
  (∀ x₁ x₂, x₁ ∈ Set.Ioo (-π/2) (π/2) ∧ x₂ ∈ Set.Ioo (-π/2) (π/2) ∧ x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l59_5949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l59_5916

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 15 - 2 * Real.cos (2 * x) - 4 * Real.sin x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((g x)^2 - 245)

-- State the theorem
theorem f_range : 
  (∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 14) ∧ 
  (∃ x : ℝ, f x = 0) ∧ 
  (∃ x : ℝ, f x = 14) := by
  sorry

#check f_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l59_5916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_prism_area_relation_l59_5907

/-- Represents an oblique prism ABC-A₁B₁C₁BC -/
structure ObliquePrism where
  ABB₁A₁ : Set (ℝ × ℝ × ℝ)
  BCC₁B₁ : Set (ℝ × ℝ × ℝ)

/-- The area of a triangle -/
noncomputable def triangleArea (t : Set (Fin 3 → ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The area of a quadrilateral -/
noncomputable def quadrilateralArea (q : Set (Fin 4 → ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The dihedral angle between two planes -/
noncomputable def dihedralAngle (p1 p2 : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

theorem oblique_prism_area_relation (prism : ObliquePrism) 
  (A₁C₁C BB₁A₁ : Set (Fin 3 → ℝ × ℝ × ℝ)) 
  (BCC₁B₁ : Set (Fin 4 → ℝ × ℝ × ℝ)) 
  (θ : ℝ) :
  θ = dihedralAngle prism.ABB₁A₁ prism.BCC₁B₁ →
  (triangleArea A₁C₁C)^2 = 
    (triangleArea BB₁A₁)^2 + 
    (quadrilateralArea BCC₁B₁)^2 - 
    2 * (triangleArea BB₁A₁) * (quadrilateralArea BCC₁B₁) * Real.cos θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_prism_area_relation_l59_5907
