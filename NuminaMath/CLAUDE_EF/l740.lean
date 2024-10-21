import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l740_74010

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log (sin (π / 4 - 2 * x))

-- State the theorem
theorem f_monotonic_increasing_interval :
  ∃ (a b : ℝ), a = 5 * π / 8 ∧ b = 7 * π / 8 ∧
  (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l740_74010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l740_74048

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line
def line (b x y : ℝ) : Prop := x + y + b = 0

-- Define the chord length
def chord_length (l : ℝ) : Prop := l = 2

-- Define the range of the ratio
def ratio_range (r : ℝ) : Prop := 2*Real.sqrt 6/3 ≤ r ∧ r < 2

-- Main theorem
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ellipse a b 1 (2*Real.sqrt 3/3) →
  (∃ (x y l : ℝ), x^2 + y^2 = 2 ∧ line b x y ∧ chord_length l) →
  (a^2 = 3 ∧ b^2 = 2) ∧
  (∀ (x y : ℝ), x ≠ 0 → ellipse a b x y → 
    ∃ (r : ℝ), ratio_range r) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l740_74048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_pair_a_value_l740_74062

/-- Two circles with a common chord of length 2√3 -/
structure CirclePair where
  a : ℝ
  h1 : a > 0
  circle1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
  circle2 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 2*a*p.2 - 6 = 0}

/-- The common chord length -/
noncomputable def common_chord_length : ℝ := 2 * Real.sqrt 3

/-- The theorem stating that for the given circle pair, a = 1 -/
theorem circle_pair_a_value (cp : CirclePair) : cp.a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_pair_a_value_l740_74062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_and_remainder_l740_74043

/-- Represents the sequence where the nth positive integer appears n times -/
def sequenceElement (n : ℕ) : ℕ := n

/-- The position of the last occurrence of n in the sequence -/
def lastPosition (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 1000th term of the sequence -/
def term1000 : ℕ := 45

theorem sequence_1000th_term_and_remainder :
  (∀ n : ℕ, sequenceElement n = n) →
  (∀ k : ℕ, k < lastPosition term1000 → sequenceElement k ≤ term1000) →
  (lastPosition (term1000 - 1) < 1000) →
  (1000 ≤ lastPosition term1000) →
  sequenceElement term1000 = term1000 ∧ term1000 % 3 = 0 := by
  sorry

#check sequence_1000th_term_and_remainder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_and_remainder_l740_74043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l740_74040

/-- Represents the time taken to complete a work -/
structure WorkTime where
  days : ℚ
  work : ℚ
  assumption : work > 0 ∧ days > 0

/-- Calculates the total time taken to complete the work -/
noncomputable def totalWorkTime (amitTime : WorkTime) (ananthuTime : WorkTime) (amitWorkDays : ℚ) : ℚ :=
  let amitWorkDone := amitWorkDays * (amitTime.work / amitTime.days)
  let remainingWork := 1 - amitWorkDone
  let ananthuWorkDays := remainingWork / (ananthuTime.work / ananthuTime.days)
  amitWorkDays + ananthuWorkDays

/-- Theorem stating that the total work time is 18 days -/
theorem work_completion_time 
  (amit : WorkTime) 
  (ananthu : WorkTime) 
  (h1 : amit.days = 10) 
  (h2 : ananthu.days = 20) 
  (h3 : amit.work = 1) 
  (h4 : ananthu.work = 1) : 
  totalWorkTime amit ananthu 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l740_74040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l740_74061

theorem triangle_abc_properties (a b c A B C : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Sine law holds
  a / Real.sin A = b / Real.sin B →
  -- Given equation
  2 * a * Real.cos C + c * Real.cos A = b →
  -- Conclusions
  (C = Real.pi / 2) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = Real.pi / 2 →
    Real.sin x * Real.cos y + Real.sin y ≤ 5 / 4) ∧
  (Real.sin (Real.pi / 3) * Real.cos (Real.pi / 6) + Real.sin (Real.pi / 6) = 5 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l740_74061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_reduction_theorem_l740_74041

/-- Calculates the price after applying a percentage change --/
noncomputable def apply_percentage_change (price : ℝ) (percentage : ℝ) : ℝ :=
  price * (1 + percentage / 100)

/-- Calculates the final price after a series of percentage changes --/
noncomputable def final_price (initial_price : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl apply_percentage_change initial_price

/-- Theorem stating the total reduction from the original price --/
theorem total_reduction_theorem (initial_price : ℝ) (changes : List ℝ) :
  initial_price = 500 →
  changes = [-6, 3, -4, 2, -5] →
  ∃ (reduction : ℝ), 
    (reduction ≥ 49.66 ∧ reduction ≤ 49.68) ∧
    (initial_price - final_price initial_price changes = reduction) := by
  sorry

-- Remove the #eval statement as it's causing issues with noncomputable definitions
-- #eval final_price 500 [-6, 3, -4, 2, -5]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_reduction_theorem_l740_74041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_with_terminal_point_l740_74012

theorem sin_angle_with_terminal_point (α : ℝ) :
  (∃ (P : ℝ × ℝ), P.1 = -1/2 ∧ P.2 = Real.sqrt 3/2 ∧ 
    Real.cos α * P.1 - Real.sin α * P.2 = 0) →
  Real.sin α = Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_with_terminal_point_l740_74012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_coordinates_l740_74007

def Point := ℝ × ℝ

def Triangle (A B C : Point) : Set Point := {A, B, C}

-- Midpoint is already defined in Mathlib, so we'll remove this definition

def isIsosceles (A B C : Point) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2

def isAltitude (A D B C : Point) : Prop :=
  (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0

theorem triangle_vertex_coordinates :
  ∀ (A B D : Point),
    A = (10, 10) →
    B = (1, -4) →
    D = (0, 1) →
    ∀ (C : Point),
      isIsosceles A B C →
      isAltitude A D B C →
      D = Prod.mk ((B.1 + C.1) / 2) ((B.2 + C.2) / 2) →
      C = (-1, 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_coordinates_l740_74007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_and_inequality_l740_74016

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

-- Define the set A
def A : Set ℝ := { x | -2 < f x ∧ f x < 0 }

-- Theorem statement
theorem solution_and_inequality :
  (A = Set.Ioo (-1/2) (1/2)) ∧
  (∀ m n, m ∈ A → n ∈ A → |1 - 4*m*n| > 2 * |m - n|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_and_inequality_l740_74016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l740_74038

/-- The distance between two parallel lines ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₁ - c₂) / Real.sqrt (a^2 + b^2)

/-- The first line: 4x - 3y + 5 = 0 -/
def line1 : ℝ → ℝ → ℝ := λ x y ↦ 4*x - 3*y + 5

/-- The second line: 8x - 6y + 5 = 0 -/
def line2 : ℝ → ℝ → ℝ := λ x y ↦ 8*x - 6*y + 5

theorem distance_between_given_lines :
  distance_between_parallel_lines 8 (-6) 10 5 = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l740_74038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sewage_treatment_problem_l740_74097

/-- Represents the sewage treatment scenario -/
structure SewageTreatment where
  /-- The initial amount of sewage in the pool -/
  initial_sewage : ℚ
  /-- The rate at which sewage flows into the pool per day -/
  inflow_rate : ℚ
  /-- The amount of sewage a single device can treat per day -/
  treatment_rate : ℚ

/-- Calculates the number of days required to treat all sewage given the number of devices -/
def days_to_treat (st : SewageTreatment) (devices : ℕ) : ℚ :=
  st.initial_sewage / (devices * st.treatment_rate - st.inflow_rate)

/-- The main theorem stating the solution to the sewage treatment problem -/
theorem sewage_treatment_problem (st : SewageTreatment) :
  days_to_treat st 4 = 36 →
  days_to_treat st 5 = 27 →
  days_to_treat st 7 = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sewage_treatment_problem_l740_74097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l740_74030

theorem triangle_side_length (A B C : ℝ) (area angle_A AB : ℝ) :
  area = 5 * Real.sqrt 3 →
  angle_A = π / 6 →
  AB = 5 →
  let BC := Real.sqrt ((4 * Real.sqrt 3)^2 + 5^2 - 2 * 4 * Real.sqrt 3 * 5 * Real.cos (π / 6))
  BC = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l740_74030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l740_74026

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, θ)

theorem rectangular_to_polar_conversion :
  let x := -2
  let y := 2 * Real.sqrt 3
  let (r, θ) := rectangular_to_polar x y
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 4 ∧ θ = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l740_74026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_sequence_formula_l740_74082

/-- A sequence where S_n - 1 is the geometric mean between a_n and S_n -/
def GeometricMeanSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (Finset.sum (Finset.range n) (a ∘ Nat.succ) - 1)^2 = a (n + 1) * Finset.sum (Finset.range n) (a ∘ Nat.succ)

theorem geometric_mean_sequence_formula (a : ℕ → ℝ) (h : GeometricMeanSequence a) :
  ∀ n : ℕ, a (n + 1) = 1 / ((n + 1) * (n + 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_sequence_formula_l740_74082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_class_average_weight_l740_74050

/-- Calculates the average weight of students in a dance class -/
theorem dance_class_average_weight 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (avg_weight_boys : ℝ) 
  (avg_weight_girls : ℝ) 
  (h1 : num_boys = 8) 
  (h2 : num_girls = 5) 
  (h3 : avg_weight_boys = 155) 
  (h4 : avg_weight_girls = 125) : 
  Int.floor ((num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) + 0.5) = 143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dance_class_average_weight_l740_74050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l740_74046

/-- Given a function f and a set A, proves that the range of a is (0, +∞) -/
theorem range_of_a (f : ℝ → ℝ) (A : Set ℝ) (a : ℝ) : 
  (∀ x, f x = x^2) →
  A = {x : ℝ | f (x + 1) = a * x} →
  A ∪ Set.Ici (0 : ℝ) ≠ Set.Ici (0 : ℝ) →
  a ∈ Set.Ioi 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l740_74046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_zero_l740_74014

-- Define the cube
def cube_edge_length : ℝ := 2

-- Define point P on edge AB
def P (t : ℝ) : ℝ × ℝ × ℝ := (t, 0, 0)

-- Define point Q on face CDFE
def Q (u v : ℝ) : ℝ × ℝ × ℝ := (cube_edge_length, u, v)

-- Define the distance function between P and Q
noncomputable def distance (t u v : ℝ) : ℝ :=
  Real.sqrt ((t - cube_edge_length)^2 + u^2 + v^2)

-- State the theorem
theorem min_distance_is_zero :
  ∃ t u v, 0 ≤ t ∧ t ≤ cube_edge_length ∧
           0 ≤ u ∧ u ≤ cube_edge_length ∧
           0 ≤ v ∧ v ≤ cube_edge_length ∧
           distance t u v = 0 := by
  sorry

#check min_distance_is_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_zero_l740_74014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_count_l740_74081

theorem marble_count (red blue green total : ℕ) : 
  red + blue + green = total →
  red = 1 * n ∧ blue = 5 * n ∧ green = 3 * n →
  green = 27 →
  total = 81 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_count_l740_74081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sweets_remaining_l740_74077

/-- Calculates the number of sweets remaining in a packet after certain actions are taken. -/
theorem sweets_remaining (cherry strawberry pineapple raspberry lemon : ℕ) 
  (h_cherry : cherry = 30)
  (h_strawberry : strawberry = 40)
  (h_pineapple : pineapple = 50)
  (h_raspberry : raspberry = 20)
  (h_lemon : lemon = 60) :
  (cherry - (cherry / 3 + 5)) +
  (strawberry - (strawberry / 4)) +
  (pineapple - (pineapple / 5)) +
  (raspberry * 2) +
  (lemon - (lemon * 3 / 10)) = 167 :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sweets_remaining_l740_74077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_inequality_l740_74034

/-- Given an arithmetic sequence {a_n} where a_3 + a_6 + a_9 = 9,
    the solution set of (x+3)[x^2 + (a_5 + a_7)x + 8] ≤ 0 is (-∞, -4] ∪ [-3, -2] -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h_arithmetic : ∀ n, a (n+1) - a n = a (n+2) - a (n+1))
    (h_sum : a 3 + a 6 + a 9 = 9) :
  {x : ℝ | (x+3)*(x^2 + (a 5 + a 7)*x + 8) ≤ 0} = Set.Iic (-4) ∪ Set.Icc (-3) (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_inequality_l740_74034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l740_74033

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_distance_theorem (A : ℝ × ℝ) :
  parabola A.1 A.2 →
  distance A focus = distance B focus →
  distance A B = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l740_74033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_a_l740_74008

-- Define the inequality function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.exp x - x * Real.cos x + Real.cos x * Real.log (Real.cos x) + a * x^2

-- Define the open interval (0, π/2)
def I : Set ℝ := Set.Ioo 0 (Real.pi / 2)

-- State the theorem
theorem smallest_integer_a : ∃ (a : ℤ), 
  (∀ x ∈ I, f (a : ℝ) x ≥ 1) ∧ 
  (∀ b : ℤ, b < a → ∃ x ∈ I, f (b : ℝ) x < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_a_l740_74008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medians_perpendicular_iff_sum_squares_l740_74075

/-- Triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The median from vertex A in a triangle -/
noncomputable def median_A (t : Triangle) : ℝ := Real.sqrt ((2 * t.b^2 + 2 * t.c^2 - t.a^2) / 4)

/-- The median from vertex B in a triangle -/
noncomputable def median_B (t : Triangle) : ℝ := Real.sqrt ((2 * t.a^2 + 2 * t.c^2 - t.b^2) / 4)

/-- Medians from A and B are perpendicular -/
def medians_perpendicular (t : Triangle) : Prop :=
  median_A t * median_B t = 0

/-- Main theorem: Medians from A and B are perpendicular iff a^2 + b^2 = 5c^2 -/
theorem medians_perpendicular_iff_sum_squares (t : Triangle) :
  medians_perpendicular t ↔ t.a^2 + t.b^2 = 5 * t.c^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_medians_perpendicular_iff_sum_squares_l740_74075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l740_74029

/-- Calculates the time for a faster train to cross a slower train -/
noncomputable def time_to_cross (length1 length2 speed1 speed2 distance : ℝ) : ℝ :=
  (length1 + length2 + distance) / (speed2 - speed1)

/-- Theorem: Given the specified train lengths, speeds, and distance, the time to cross is 60 seconds -/
theorem train_crossing_time :
  let length1 : ℝ := 100
  let length2 : ℝ := 150
  let speed1 : ℝ := 10
  let speed2 : ℝ := 15
  let distance : ℝ := 50
  time_to_cross length1 length2 speed1 speed2 distance = 60 := by
  -- Unfold the definition of time_to_cross
  unfold time_to_cross
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l740_74029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l740_74068

theorem sufficient_not_necessary : 
  (∀ x : ℝ, |x + 1| < 1 → |x| < 2) ∧ 
  (∃ x : ℝ, |x| < 2 ∧ ¬(|x + 1| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l740_74068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_Q_equation_no_perpendicular_bisector_l740_74083

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Define line l1 (implicitly through its properties)
def line_l1_intersects_C (M N : ℝ × ℝ) : Prop :=
  circle_C M.1 M.2 ∧ circle_C N.1 N.2 ∧ 
  (M.1 - point_P.1) * (N.2 - point_P.2) = (N.1 - point_P.1) * (M.2 - point_P.2)

-- Define the length of MN
noncomputable def MN_length (M N : ℝ × ℝ) : ℝ := 
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)

-- Define line ax - y + 1 = 0
def line_a (a x y : ℝ) : Prop := a*x - y + 1 = 0

-- Define the intersection of line_a with circle C
def line_a_intersects_C (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
  line_a a A.1 A.2 ∧ line_a a B.1 B.2

-- Theorem 1
theorem circle_Q_equation (M N : ℝ × ℝ) :
  line_l1_intersects_C M N →
  MN_length M N = 4 →
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 ↔ 
    (x - M.1)^2 + (y - M.2)^2 = (x - N.1)^2 + (y - N.2)^2 := by
  sorry

-- Theorem 2
theorem no_perpendicular_bisector :
  ¬ ∃ a : ℝ, ∀ A B : ℝ × ℝ, 
    line_a_intersects_C a A B →
    ∃ l2_slope : ℝ,
      l2_slope * (A.1 - point_P.1) = A.2 - point_P.2 ∧
      l2_slope * (B.1 - point_P.1) = B.2 - point_P.2 ∧
      (A.1 + B.1) / 2 = point_P.1 + l2_slope * ((A.2 + B.2) / 2 - point_P.2) ∧
      (A.2 + B.2) / 2 = point_P.2 + (1 / l2_slope) * ((A.1 + B.1) / 2 - point_P.1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_Q_equation_no_perpendicular_bisector_l740_74083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_purchase_equation_correct_l740_74086

/-- Represents the price and quantity of milk purchased on two different days -/
structure MilkPurchase where
  wednesday_quantity : ℕ
  wednesday_total : ℚ
  sunday_quantity : ℕ
  sunday_total : ℚ
  price_difference : ℚ

/-- The equation representing the milk purchase scenario -/
def milk_equation (x : ℚ) : Prop :=
  x^2 + 6*x - 40 = 0

/-- Theorem stating that the milk equation correctly represents the given scenario -/
theorem milk_purchase_equation_correct (purchase : MilkPurchase) 
  (h1 : purchase.wednesday_total = 10)
  (h2 : purchase.sunday_quantity = purchase.wednesday_quantity + 2)
  (h3 : purchase.sunday_total = purchase.wednesday_total + 2)
  (h4 : purchase.price_difference = 1/2)
  : milk_equation (purchase.wednesday_quantity : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_purchase_equation_correct_l740_74086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_abs_abs_minus_x_l740_74076

theorem abs_abs_abs_minus_x (x : ℤ) (h : x = -2023) : 
  (abs ((abs (abs x - x)) - abs x) - x) = 4046 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_abs_abs_minus_x_l740_74076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_in_interval_l740_74092

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * (Real.sin x)^3 - 7 * (Real.sin x)^2 + 3 * (Real.sin x) + 1

-- Define the interval
def I : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2 * Real.pi}

-- Theorem statement
theorem three_solutions_in_interval :
  ∃ (a b c : ℝ), a ∈ I ∧ b ∈ I ∧ c ∈ I ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x ∈ I, f x = 0 → x = a ∨ x = b ∨ x = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_in_interval_l740_74092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_benzene_molecular_weight_l740_74099

/-- The molecular weight of Benzene for a given number of moles -/
noncomputable def molecular_weight (moles : ℝ) : ℝ := 312 / 4 * moles

/-- Theorem: The molecular weight of one mole of Benzene is 78 g/mol -/
theorem benzene_molecular_weight : molecular_weight 1 = 78 := by
  unfold molecular_weight
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_benzene_molecular_weight_l740_74099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_is_infinite_l740_74002

-- Define a set of points in 2D space
def Point := ℝ × ℝ

-- Define what it means for a point to be a midpoint of two other points
def isMidpoint (p q r : Point) : Prop :=
  p.1 = (q.1 + r.1) / 2 ∧ p.2 = (q.2 + r.2) / 2

-- Define the property of the set S
def hasMidpointProperty (S : Set Point) : Prop :=
  ∀ p, p ∈ S → ∃ q r, q ∈ S ∧ r ∈ S ∧ isMidpoint p q r

-- Theorem statement
theorem midpoint_set_is_infinite (S : Set Point) (h : hasMidpointProperty S) : 
  Set.Infinite S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_is_infinite_l740_74002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_condition_l740_74055

/-- A function that represents x² + a ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log x

/-- The property of f being monotonically increasing on (1, +∞) -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 1 < x → x < y → f a x < f a y

/-- The theorem stating the necessary and sufficient condition for f to be monotonically increasing -/
theorem monotone_condition (a : ℝ) :
  is_monotone_increasing a ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_condition_l740_74055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_local_minimum_at_two_l740_74078

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 - (4*a + 1) * x + 4*a + 3) * Real.exp x

-- Part I: Tangent line parallel to x-axis at x=1
theorem tangent_parallel_to_x_axis (a : ℝ) :
  (deriv (f a) 1 = 0) → a = 1 := by sorry

-- Part II: Local minimum at x=2
theorem local_minimum_at_two (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 2| < δ → f a x ≥ f a 2) → a > 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_local_minimum_at_two_l740_74078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_b_l740_74032

theorem gcd_of_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 528 * k) :
  Int.gcd (4 * b^3 + 2 * b^2 + 5 * b + 66) b.natAbs = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_b_l740_74032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_trigonometric_equation_l740_74042

theorem smallest_angle_trigonometric_equation : 
  ∃ x : ℝ, x > 0 ∧
    (∀ y : ℝ, y > 0 → y < x → ¬(Real.sin (4 * y) * Real.sin (6 * y) = Real.cos (4 * y) * Real.cos (6 * y))) ∧
    (Real.sin (4 * x) * Real.sin (6 * x) = Real.cos (4 * x) * Real.cos (6 * x)) ∧
    x = 9 * π / 180 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_trigonometric_equation_l740_74042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_equation_l740_74028

theorem tan_value_from_sin_cos_equation (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin α ^ 2 + Real.cos (2 * α) = 1 / 4) : 
  Real.tan α = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_equation_l740_74028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_properties_l740_74053

/-- A right rectangular prism with given face areas -/
structure RectangularPrism where
  area_xy : ℝ
  area_xz : ℝ
  area_yz : ℝ

/-- Calculate the volume of the prism -/
noncomputable def volume (p : RectangularPrism) : ℝ :=
  Real.sqrt (p.area_xy * p.area_xz * p.area_yz)

/-- Calculate the sum of dimensions of the prism -/
noncomputable def sum_dimensions (p : RectangularPrism) : ℝ :=
  Real.sqrt (volume p / p.area_yz) +
  Real.sqrt (volume p / p.area_xz) +
  Real.sqrt (volume p / p.area_xy)

theorem prism_properties (p : RectangularPrism)
  (h_xy : p.area_xy = 40)
  (h_xz : p.area_xz = 48)
  (h_yz : p.area_yz = 60) :
  volume p = 240 ∧ sum_dimensions p = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_properties_l740_74053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l740_74087

/-- The length of a train given its speed, bridge length, and crossing time --/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 45 * 1000 / 3600 → 
  bridge_length = 215 → 
  crossing_time = 30 → 
  (train_speed * crossing_time) - bridge_length = 160 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l740_74087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_implies_p_range_l740_74093

/-- Given a quadratic function f(x) = x^2 + ax + b, if for any x, y ∈ ℝ and p, q ∈ ℝ satisfying
    p + q = 1, it holds that pf(x) + qf(y) ≥ f(px + qy), then 0 ≤ p ≤ 1. -/
theorem quadratic_inequality_implies_p_range (a b : ℝ) :
  (∀ (x y p q : ℝ), p + q = 1 →
    (fun t => t^2 + a*t + b) (p*x + q*y) ≤ p * ((fun t => t^2 + a*t + b) x) + q * ((fun t => t^2 + a*t + b) y)) →
  ∀ p : ℝ, (∃ q : ℝ, p + q = 1) → 0 ≤ p ∧ p ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_implies_p_range_l740_74093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l740_74079

-- Define the ellipse E
noncomputable def E (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the vertices A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define a point P on the ellipse
noncomputable def P (t : ℝ) : ℝ × ℝ := ((2*t^2 - 2)/(t^2 + 1), -2*t/(t^2 + 1))

-- Define point T
def T (t : ℝ) : ℝ × ℝ := (4, t)

-- Define point Q
noncomputable def Q (t : ℝ) : ℝ × ℝ := ((18 - 2*t^2)/(t^2 + 9), 6*t/(t^2 + 9))

-- Theorem statement
theorem ellipse_fixed_point (t : ℝ) :
  t ≠ 0 →
  E (P t).1 (P t).2 →
  (P t).1 ≠ A.1 ∧ (P t).1 ≠ B.1 →
  (∃ k : ℝ, (Q t).2 - (P t).2 = k * ((Q t).1 - (P t).1) ∧
             0 - (P t).2 = k * (1 - (P t).1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l740_74079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l740_74073

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    right focus F₂, left vertex A, and points P and Q on the hyperbola
    such that F₂PQ is perpendicular to the x-axis and AP ⊥ AQ,
    prove that the eccentricity of the hyperbola is 2. -/
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (F₂ A P Q : ℝ × ℝ) : 
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ ({P, Q} : Set (ℝ × ℝ))) →
  F₂.1 = A.1 + 2 * a →
  F₂.2 = 0 →
  A.1 = -a →
  A.2 = 0 →
  F₂.1 = P.1 →
  F₂.1 = Q.1 →
  (P.1 - A.1) * (Q.1 - A.1) + (P.2 - A.2) * (Q.2 - A.2) = 0 →
  (2 : ℝ) = (F₂.1 - A.1) / a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l740_74073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l740_74036

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 41 = 0
def circle_O2 (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

-- Define the centers and radii
def center_O1 : ℝ × ℝ := (2, -2)
def radius_O1 : ℝ := 7
def center_O2 : ℝ × ℝ := (-1, 2)
def radius_O2 : ℝ := 2

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ :=
  Real.sqrt ((center_O1.1 - center_O2.1)^2 + (center_O1.2 - center_O2.2)^2)

-- Theorem: The circles are internally tangent
theorem circles_internally_tangent :
  distance_between_centers = radius_O1 - radius_O2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l740_74036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_l740_74052

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

-- State the theorem
theorem f_greater_than_one (x : ℝ) : f x > 1 ↔ x < -1 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_l740_74052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_is_three_and_five_sixths_l740_74001

/-- The region defined by the inequality |x + yt + t^2| ≤ 1 for all t ∈ [0, 1] -/
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∀ t ∈ Set.Icc 0 1, |p.1 + p.2 * t + t^2| ≤ 1}

/-- The area of the region -/
noncomputable def area_of_region : ℝ := sorry

/-- Theorem stating that the area of the region is 3 5/6 -/
theorem area_of_region_is_three_and_five_sixths :
  area_of_region = 3 + 5/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_is_three_and_five_sixths_l740_74001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_specific_hyperbola_l740_74024

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- The eccentricity of the hyperbola x²/4 - y²/5 = 1 is 3/2 -/
theorem eccentricity_of_specific_hyperbola :
  hyperbola_eccentricity 2 (Real.sqrt 5) = 3/2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_specific_hyperbola_l740_74024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_odd_function_l740_74059

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (2^x + m) / (2^x - m)

theorem sufficient_condition_for_odd_function (m : ℝ) :
  (m = 1 ∨ m = -1) → (∀ x : ℝ, f (-x) m = -f x m) :=
by
  intro h
  intro x
  cases h with
  | inl h_left => 
    -- Case m = 1
    sorry
  | inr h_right => 
    -- Case m = -1
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_odd_function_l740_74059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_seven_digit_binary_l740_74045

theorem largest_seven_digit_binary : ∃ n : ℕ, 
  (∀ m : ℕ, 64 ≤ m ∧ m < 128 → m ≤ n) ∧
  64 ≤ n ∧ n < 128 ∧
  n = 127 := by
  use 127
  constructor
  · intros m hm
    linarith
  · constructor
    · linarith
    · constructor
      · linarith
      · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_seven_digit_binary_l740_74045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_fraction_tan_difference_identity_l740_74098

-- Part 1
theorem simplify_trig_fraction (θ : ℝ) (h : 1 + Real.cos θ + Real.cos (2*θ) ≠ 0) :
  (Real.sin θ + Real.sin (2*θ)) / (1 + Real.cos θ + Real.cos (2*θ)) = Real.tan θ := by
  sorry

-- Part 2
theorem tan_difference_identity (α : ℝ) 
  (h1 : Real.cos (π/4 + α) ≠ 0) (h2 : Real.cos (π/4 - α) ≠ 0) (h3 : Real.cos (2*α) ≠ 0) :
  Real.tan (π/4 + α) - Real.tan (π/4 - α) = 2 * Real.tan (2*α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_fraction_tan_difference_identity_l740_74098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farey_adjacent_to_half_l740_74088

/-- Farey sequence of order n -/
def farey_sequence (n : ℕ) : Set ℚ :=
  {q : ℚ | 0 ≤ q ∧ q ≤ 1 ∧ q.den ≤ n}

/-- Adjacent fractions to 1/2 in Farey sequence -/
def adjacent_to_half (n : ℕ) : Prop :=
  if n % 2 = 1 then
    (((n - 1) : ℚ) / (2 * n)) ∈ farey_sequence n ∧
    (((n + 1) : ℚ) / (2 * n)) ∈ farey_sequence n ∧
    ∀ q ∈ farey_sequence n, q < 1/2 → q ≤ ((n - 1) : ℚ) / (2 * n) ∧
    ∀ q ∈ farey_sequence n, q > 1/2 → q ≥ ((n + 1) : ℚ) / (2 * n)
  else
    (((n - 2) : ℚ) / (2 * (n - 1))) ∈ farey_sequence n ∧
    ((n : ℚ) / (2 * (n - 1))) ∈ farey_sequence n ∧
    ∀ q ∈ farey_sequence n, q < 1/2 → q ≤ ((n - 2) : ℚ) / (2 * (n - 1)) ∧
    ∀ q ∈ farey_sequence n, q > 1/2 → q ≥ (n : ℚ) / (2 * (n - 1))

theorem farey_adjacent_to_half (n : ℕ) :
  adjacent_to_half n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farey_adjacent_to_half_l740_74088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_surviving_babies_approx_l740_74056

/-- Represents the number of kettles -/
def num_kettles : ℕ := 15

/-- Represents the minimum average number of hawks per kettle -/
def min_hawks_per_kettle : ℚ := 18

/-- Represents the maximum average number of hawks per kettle -/
def max_hawks_per_kettle : ℚ := 25

/-- Represents the minimum pregnancy rate -/
def min_pregnancy_rate : ℚ := 2/5

/-- Represents the maximum pregnancy rate -/
def max_pregnancy_rate : ℚ := 3/5

/-- Represents the minimum number of babies per batch -/
def min_babies_per_batch : ℚ := 5

/-- Represents the maximum number of babies per batch -/
def max_babies_per_batch : ℚ := 8

/-- Represents the minimum survival rate -/
def min_survival_rate : ℚ := 3/5

/-- Represents the maximum survival rate -/
def max_survival_rate : ℚ := 4/5

/-- Calculates the expected number of surviving baby hawks -/
noncomputable def expected_surviving_babies : ℚ :=
  let avg_hawks_per_kettle := (min_hawks_per_kettle + max_hawks_per_kettle) / 2
  let avg_pregnancy_rate := (min_pregnancy_rate + max_pregnancy_rate) / 2
  let avg_babies_per_batch := (min_babies_per_batch + max_babies_per_batch) / 2
  let avg_survival_rate := (min_survival_rate + max_survival_rate) / 2
  let total_hawks := num_kettles * avg_hawks_per_kettle
  let pregnant_hawks := total_hawks * avg_pregnancy_rate
  let total_babies := pregnant_hawks * avg_babies_per_batch
  total_babies * avg_survival_rate

theorem expected_surviving_babies_approx :
  ⌊(expected_surviving_babies : ℝ)⌋ = 732 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_surviving_babies_approx_l740_74056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l740_74065

-- Define the function f
variable (f : ℝ → ℝ)

-- Hypotheses
axiom f_even : ∀ x, f (-x) = f x
axiom f_increasing : ∀ x y, 0 ≤ x → x < y → f x < f y
axiom f_2_eq_0 : f 2 = 0

-- Theorem to prove
theorem solution_set (x : ℝ) : f (x + 1) < 0 ↔ -3 < x ∧ x < 1 :=
sorry

-- The proof goes here
-- We'll use sorry to skip the actual proof for now

#check solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l740_74065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l740_74039

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin (π / 2 - x) * sin x - Real.sqrt 3 * (sin x)^2

-- State the theorem
theorem f_properties :
  -- The smallest positive period of f is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- The minimum value of f in [0, π/4] is -√3/2
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ π/4 → f x ≥ -Real.sqrt 3 / 2) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ π/4 ∧ f x = -Real.sqrt 3 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l740_74039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l740_74095

noncomputable def f (x : ℝ) : ℝ := x^3 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := (5 - x) / 5

def intersection_points : Set (ℝ × ℝ) :=
  {p | f p.1 = g p.1 ∧ p.2 = f p.1}

theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    x₁ + x₂ + x₃ = 0 ∧
    y₁ + y₂ + y₃ = 3 :=
by
  sorry

#check intersection_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l740_74095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_number_of_men_l740_74091

/-- Prove that the initial number of men is 9, given the conditions of the problem -/
theorem initial_number_of_men : Nat := 
  let age_man1 : Nat := 36
  let age_man2 : Nat := 32
  let avg_age_women : Nat := 52
  let avg_age_increase : Nat := 4
  let num_men : Nat := 9

  have h : (2 * avg_age_women - (age_man1 + age_man2)) = avg_age_increase * num_men := by sorry

  num_men

#check initial_number_of_men

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_number_of_men_l740_74091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_integers_between_5_and_20_l740_74011

theorem sum_of_even_integers_between_5_and_20 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ n > 5 ∧ n < 20) (Finset.range 20)).sum id = 84 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_integers_between_5_and_20_l740_74011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrogen_percentage_approx_l740_74027

/-- Atomic mass of Carbon in g/mol -/
noncomputable def carbon_mass : ℝ := 12.01

/-- Atomic mass of Hydrogen in g/mol -/
noncomputable def hydrogen_mass : ℝ := 1.008

/-- Number of Carbon atoms in Butane -/
def carbon_atoms : ℕ := 4

/-- Number of Hydrogen atoms in Butane -/
def hydrogen_atoms : ℕ := 10

/-- Calculate the mass percentage of Hydrogen in Butane -/
noncomputable def hydrogen_percentage_in_butane : ℝ :=
  let butane_mass := carbon_mass * carbon_atoms + hydrogen_mass * hydrogen_atoms
  let hydrogen_mass_in_butane := hydrogen_mass * hydrogen_atoms
  (hydrogen_mass_in_butane / butane_mass) * 100

/-- Theorem stating that the mass percentage of Hydrogen in Butane is approximately 17.33% -/
theorem hydrogen_percentage_approx :
  |hydrogen_percentage_in_butane - 17.33| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrogen_percentage_approx_l740_74027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l740_74047

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (x + 4) + 1 / (x^2 - 4) + 1 / (x^4 + 4)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≠ -4 ∧ x ≠ -2 ∧ x ≠ 2}

-- Theorem stating that the domain of f is correct
theorem domain_of_f :
  ∀ x : ℝ, x ∈ domain_f ↔ (∃ y : ℝ, f x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l740_74047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_and_sum_l740_74060

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (Finset.range n).sum (fun i => (2 * i + 1) * a (i + 1)) = 2 * n

theorem sequence_formula_and_sum (a : ℕ → ℝ) (h : sequence_property a) :
  (∀ n : ℕ, n > 0 → a n = 2 / (2 * n - 1)) ∧
  (∀ n : ℕ, n > 0 → (Finset.range n).sum (fun i => a (i + 1) / (2 * (i + 1) + 1)) = 2 * n / (2 * n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_and_sum_l740_74060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_average_score_l740_74089

/-- Represents the quiz scores of three students -/
structure QuizScores where
  dorothy : ℚ
  ivanna : ℚ
  tatuya : ℚ

/-- Calculates the average score of the three students -/
def average_score (scores : QuizScores) : ℚ :=
  (scores.dorothy + scores.ivanna + scores.tatuya) / 3

/-- Theorem stating that given the conditions, the average score is 84 -/
theorem quiz_average_score :
  ∀ (scores : QuizScores),
  scores.dorothy = 90 →
  scores.ivanna = 3/5 * scores.dorothy →
  scores.tatuya = 2 * scores.ivanna →
  average_score scores = 84 := by
  intro scores hdorothy hivanna htatuya
  simp [average_score, hdorothy, hivanna, htatuya]
  -- The proof steps would go here
  sorry

#eval average_score { dorothy := 90, ivanna := 54, tatuya := 108 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_average_score_l740_74089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_can_be_opened_l740_74017

/-- Represents the orientation of a key -/
inductive KeyOrientation
| Horizontal
| Vertical

/-- Represents a position in the 4x4 grid -/
structure Position where
  row : Fin 4
  col : Fin 4

/-- Represents the state of the lock -/
def LockState := Position → KeyOrientation

/-- Switches the orientation of a key -/
def switchOrientation (o : KeyOrientation) : KeyOrientation :=
  match o with
  | KeyOrientation.Horizontal => KeyOrientation.Vertical
  | KeyOrientation.Vertical => KeyOrientation.Horizontal

/-- Determines if a position is affected by switching a key -/
def isAffected (p q : Position) : Bool :=
  p.row = q.row ∨ p.col = q.col

/-- Applies a switch to the lock state -/
def applySwitch (state : LockState) (p : Position) : LockState :=
  fun q => if isAffected p q then switchOrientation (state q) else state q

/-- Checks if all keys are vertical -/
def allVertical (state : LockState) : Prop :=
  ∀ p, state p = KeyOrientation.Vertical

/-- The main theorem stating that any lock can be opened -/
theorem lock_can_be_opened (initial : LockState) :
  ∃ (switches : List Position), allVertical (switches.foldl applySwitch initial) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_can_be_opened_l740_74017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l740_74022

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := x + 4 / x

-- State the theorem
theorem f_decreasing_on_interval :
  -- f passes through (1,5) and (2,4)
  f 1 = 5 ∧ f 2 = 4 →
  -- f is decreasing on (0,2)
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 → f x > f y :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l740_74022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_calculation_l740_74072

/-- Calculates the speed of a car in km/h given the tire's rotation speed and circumference -/
noncomputable def car_speed (revolutions_per_minute : ℝ) (tire_circumference : ℝ) : ℝ :=
  (revolutions_per_minute * tire_circumference * 60) / 1000

/-- Theorem: A car with a tire rotating at 400 revolutions per minute and a circumference of 7 meters travels at 168 km/h -/
theorem car_speed_calculation :
  car_speed 400 7 = 168 := by
  -- Unfold the definition of car_speed
  unfold car_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_calculation_l740_74072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_from_tan_l740_74054

theorem cos_double_angle_from_tan (θ : Real) (h : Real.tan θ = 2) : Real.cos (2 * θ) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_from_tan_l740_74054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_from_linear_combination_l740_74005

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equality_from_linear_combination
  (a b : V)
  (h_not_collinear : ¬ ∃ (r : ℝ), b = r • a)
  (lambda1 lambda2 mu1 mu2 : ℝ)
  (h_eq : lambda1 • a + mu1 • b = lambda2 • a + mu2 • b) :
  lambda1 = lambda2 ∧ mu1 = mu2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_from_linear_combination_l740_74005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eta_expectation_and_variance_l740_74094

/-- The probability density function for a standard normal distribution -/
noncomputable def standardNormalPDF (x : ℝ) : ℝ :=
  1 / Real.sqrt (2 * Real.pi) * Real.exp (-x^2 / 2)

/-- A random variable with standard normal distribution -/
structure StandardNormal where
  pdf : ℝ → ℝ
  pdf_is_standard_normal : pdf = standardNormalPDF

/-- The random variable η defined as 10^ξ where ξ is standard normal -/
noncomputable def eta (ξ : StandardNormal) : ℝ → ℝ := fun x ↦ 10^x

/-- The expected value of η -/
noncomputable def expectedValueEta (ξ : StandardNormal) : ℝ :=
  Real.exp ((Real.log 10)^2 / 2)

/-- The variance of η -/
noncomputable def varianceEta (ξ : StandardNormal) : ℝ :=
  Real.exp (2 * (Real.log 10)^2) - Real.exp ((Real.log 10)^2)

/-- Theorem stating the expected value and variance of η -/
theorem eta_expectation_and_variance (ξ : StandardNormal) :
  (expectedValueEta ξ = Real.exp ((Real.log 10)^2 / 2)) ∧
  (varianceEta ξ = Real.exp (2 * (Real.log 10)^2) - Real.exp ((Real.log 10)^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eta_expectation_and_variance_l740_74094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_transformation_properties_l740_74044

/-- Definition of a valid table in set F -/
def IsValidTable (n : ℕ) (a : ℕ → ℕ → ℕ) : Prop :=
  ∀ i k, 1 ≤ i → i ≤ n → 1 ≤ k → k ≤ n → a k i ≤ a (k-1) i

/-- Definition of the transformation f -/
def f (n : ℕ) (a : ℕ → ℕ → ℕ) : ℕ → ℕ → ℕ :=
  λ k i ↦ a (n+1-i) k - k + i

/-- Main theorem -/
theorem table_transformation_properties (n : ℕ) (a : ℕ → ℕ → ℕ) 
  (h : IsValidTable n a) : 
  IsValidTable n (f n a) ∧ 
  (∀ k i, f n (f n (f n (f n (f n (f n a))))) k i = a k i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_transformation_properties_l740_74044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l740_74000

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio 
  (a₁ : ℝ) (q : ℝ) (h : q ≠ 1) :
  geometric_sum a₁ q 3 + 3 * geometric_sum a₁ q 2 = 0 → q = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l740_74000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_is_19_l740_74019

/-- A configuration of pieces on a 6x6 checkerboard -/
def Configuration := Fin 6 → Fin 6 → Bool

/-- Count the number of pieces in the same row or column as a given position, excluding the piece at that position -/
def countPiecesInRowCol (config : Configuration) (row col : Fin 6) : Nat :=
  (Finset.sum (Finset.univ.erase col) fun i => if config row i then 1 else 0) +
  (Finset.sum (Finset.univ.erase row) fun i => if config i col then 1 else 0)

/-- Check if a configuration satisfies the condition for each n from 2 to 10 -/
def isValidConfiguration (config : Configuration) : Prop :=
  ∀ n, 2 ≤ n ∧ n ≤ 10 →
    ∃ row col, config row col ∧ countPiecesInRowCol config row col = n

/-- Count the total number of pieces in a configuration -/
def totalPieces (config : Configuration) : Nat :=
  Finset.sum Finset.univ fun row => Finset.sum Finset.univ fun col => if config row col then 1 else 0

/-- The main theorem: the minimum number of pieces is 19 -/
theorem min_pieces_is_19 :
  (∃ config : Configuration, isValidConfiguration config ∧ totalPieces config = 19) ∧
  (∀ config : Configuration, isValidConfiguration config → totalPieces config ≥ 19) := by
  sorry

#check min_pieces_is_19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_is_19_l740_74019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_reciprocal_x_squared_l740_74064

/-- The function f(x) = |x-1| + |x+7| -/
def f (x : ℝ) : ℝ := |x - 1| + |x + 7|

/-- n is the minimum value of f(x) -/
noncomputable def n : ℕ := 8

theorem coefficient_of_reciprocal_x_squared :
  (Finset.range (n + 1)).sum (fun k => Nat.choose n k * (1 : ℝ)^(n - k) * (-1)^k) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_reciprocal_x_squared_l740_74064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_simplification_and_range_l740_74085

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x)

noncomputable def g (x : ℝ) : ℝ := Real.sin x * f (Real.sin (2 * x)) + (Real.sqrt 6 + Real.sqrt 2) / 4 * f (Real.cos (4 * x))

theorem g_simplification_and_range :
  ∀ x ∈ Set.Icc (-π/4) 0,
    g x = -Real.sin (2*x - π/6) - 1/2 ∧
    g x ∈ Set.Ioo 0 (1/2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_simplification_and_range_l740_74085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_circle_l740_74006

/-- Given non-zero complex numbers satisfying certain conditions, 
    prove that they all have the same absolute value of 2. -/
theorem points_on_circle (a₁ a₂ a₃ a₄ a₅ : ℂ) (S : ℝ) 
  (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0)
  (h_ratio : a₂ / a₁ = a₃ / a₂ ∧ a₃ / a₂ = a₄ / a₃ ∧ a₄ / a₃ = a₅ / a₄)
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅))
  (h_sum_eq_S : a₁ + a₂ + a₃ + a₄ + a₅ = S)
  (h_S_bound : Complex.abs S ≤ 2) :
  Complex.abs a₁ = 2 ∧ Complex.abs a₂ = 2 ∧ Complex.abs a₃ = 2 ∧ Complex.abs a₄ = 2 ∧ Complex.abs a₅ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_circle_l740_74006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_l740_74031

open Real

theorem sine_cosine_inequality (x : ℝ) (hx : x ∈ Set.Ioo 0 (π / 2)) :
  (∀ k : ℝ, k < 1 → k * sin x * cos x < x) ∧
  (∃ k : ℝ, k ≥ 1 ∧ k * sin x * cos x < x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_l740_74031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_properties_l740_74004

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the midpoint of chord AB
def midpoint_AB : ℝ × ℝ := (2, 1)

-- Define the equation of line AB
def line_AB (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define the length of chord AB
noncomputable def length_AB : ℝ := 2 * Real.sqrt 5

-- Theorem statement
theorem ellipse_chord_properties :
  ∀ (x_A y_A x_B y_B : ℝ),
  ellipse x_A y_A ∧ ellipse x_B y_B ∧
  ((x_A + x_B) / 2, (y_A + y_B) / 2) = midpoint_AB →
  (∀ (x y : ℝ), line_AB x y ↔ ∃ t : ℝ, x = x_A + t*(x_B - x_A) ∧ y = y_A + t*(y_B - y_A)) ∧
  Real.sqrt ((x_B - x_A)^2 + (y_B - y_A)^2) = length_AB :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_properties_l740_74004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_under_promotion_l740_74018

/-- Represents the relationship between selling price and sales volume -/
def sales_volume (x : ℝ) : ℝ := -2 * x + 200

/-- The cost price of each box -/
def cost_price : ℝ := 24

/-- The listed price of each box -/
def listed_price : ℝ := 45

/-- The minimum allowed selling price -/
def min_selling_price : ℝ := 0.8 * listed_price

/-- The maximum allowed selling price -/
def max_selling_price : ℝ := listed_price

/-- The cost of the free potatoes given with each box during promotion -/
def potato_cost : ℝ := 6

/-- The daily profit function during the promotion -/
def daily_profit (x : ℝ) : ℝ := (x - cost_price - potato_cost) * sales_volume x

/-- Theorem stating the maximum daily profit under the promotion -/
theorem max_profit_under_promotion :
  ∃ (x : ℝ), min_selling_price ≤ x ∧ x ≤ max_selling_price ∧
  daily_profit x = 1650 ∧
  ∀ (y : ℝ), min_selling_price ≤ y ∧ y ≤ max_selling_price →
  daily_profit y ≤ daily_profit x := by
  sorry

#eval daily_profit max_selling_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_under_promotion_l740_74018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitudes_sum_l740_74080

/-- Given a triangle ABC with side length b, opposite angle β, and the sum of altitudes from vertices A and C equal to s, 
    prove that cos((α - γ)/2) = s / (2b * cos(β/2)), where α and γ are the other two angles of the triangle. -/
theorem triangle_altitudes_sum (b s : ℝ) (β α γ : ℝ) (h_positive : b > 0) (h_angle : 0 < β ∧ β < π) 
  (h_sum_angles : α + β + γ = π) (h_altitudes_sum : b * Real.sin γ + b * Real.sin α = s) : 
  Real.cos ((α - γ)/2) = s / (2 * b * Real.cos (β/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitudes_sum_l740_74080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_l740_74013

/-- The equation of a potential circle -/
def potential_circle (a x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 2*a*x + a = 0

/-- Definition of a circle in the xy-plane -/
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ h k r, r > 0 ∧ ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that a = -1 is a necessary and sufficient condition for the equation to represent a circle -/
theorem circle_condition : 
  (is_circle (potential_circle (-1))) ∧ 
  (∀ a : ℝ, is_circle (potential_circle a) → a = -1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_l740_74013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotary_club_eggs_needed_l740_74049

/-- Calculates the number of eggs needed for the Rotary Club's Omelet Breakfast --/
theorem rotary_club_eggs_needed :
  let small_children : ℕ := 53
  let older_children : ℕ := 35
  let adults : ℕ := 75
  let seniors : ℕ := 37
  let small_children_omelets : ℚ := 1/2
  let older_children_omelets : ℚ := 1
  let adult_omelets : ℚ := 2
  let senior_omelets : ℚ := 3/2
  let extra_omelets : ℕ := 25
  let eggs_per_omelet : ℕ := 2

  let total_omelets : ℚ := 
    small_children * small_children_omelets +
    older_children * older_children_omelets +
    adults * adult_omelets +
    seniors * senior_omelets +
    extra_omelets

  let total_eggs : ℕ := (Int.ceil total_omelets).toNat * eggs_per_omelet

  total_eggs = 584 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotary_club_eggs_needed_l740_74049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_theorem_PQ_distance_theorem_l740_74057

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (a^2 - b^2) / a^2 = 1/4  -- eccentricity = 1/2

/-- Point on the ellipse -/
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : y^2 / e.a^2 + x^2 / e.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Right vertex of the ellipse -/
def rightVertex (e : Ellipse) : EllipsePoint e where
  x := e.b
  y := 0
  h := by sorry

/-- Upper focus of the ellipse -/
noncomputable def upperFocus (e : Ellipse) : ℝ × ℝ :=
  (Real.sqrt (e.a^2 - e.b^2), 0)

/-- Theorem 1: Sum of distances from A and B to D -/
theorem distance_sum_theorem (e : Ellipse) (A B : EllipsePoint e) 
  (h : distance (B.x, B.y) (e.b, 0) - distance (A.x, A.y) (upperFocus e) = 8 * Real.sqrt 3 / 39) :
  distance (A.x, A.y) (e.b, 0) + distance (B.x, B.y) (e.b, 0) = 112 * Real.sqrt 3 / 39 := by sorry

/-- Theorem 2: Distance between P and Q -/
theorem PQ_distance_theorem (e : Ellipse) (A B : EllipsePoint e) (E P Q : ℝ × ℝ) 
  (h1 : distance (B.x, B.y) (e.b, 0) - distance (A.x, A.y) (upperFocus e) = 8 * Real.sqrt 3 / 39)
  (h2 : True) -- Placeholder for additional conditions for E, P, and Q
  : distance P Q = 16/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_theorem_PQ_distance_theorem_l740_74057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l740_74035

theorem trigonometric_equation_solution (x : ℝ) : 
  Real.sin x + Real.sin (7 * x) - Real.cos (5 * x) + Real.cos (3 * x - 2 * Real.pi) = 0 →
  (∃ k : ℤ, x = (Real.pi * ↑k) / 4) ∨ (∃ n : ℤ, x = (Real.pi * (4 * ↑n + 3)) / 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l740_74035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l740_74070

/-- The length of the bridge in meters -/
noncomputable def bridge_length : ℝ := 199.9744020478362

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := 120

/-- The time taken by the train to cross the bridge in seconds -/
noncomputable def crossing_time : ℝ := 31.99744020478362

/-- The speed of the train in kilometers per hour -/
noncomputable def train_speed_kmph : ℝ := 36

/-- Conversion factor from km/h to m/s -/
noncomputable def kmph_to_ms : ℝ := 5 / 18

theorem bridge_length_calculation :
  bridge_length = train_speed_kmph * kmph_to_ms * crossing_time - train_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l740_74070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_schedule_l740_74063

/-- Represents a watch schedule for warriors. -/
def WatchSchedule := Fin 33 → Fin 33 → Bool

/-- A valid watch schedule satisfies the required conditions. -/
def is_valid_schedule (schedule : WatchSchedule) : Prop :=
  -- Each day has the correct number of warriors on watch
  (∀ d : Fin 33, (Finset.univ.filter (fun w ↦ schedule d w)).card = d.val + 1) ∧
  -- Each warrior goes on watch exactly 17 times
  (∀ w : Fin 33, (Finset.univ.filter (fun d ↦ schedule d w)).card = 17)

/-- There exists a valid watch schedule for 33 warriors over 33 days. -/
theorem exists_valid_schedule : ∃ (schedule : WatchSchedule), is_valid_schedule schedule := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_schedule_l740_74063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotone_increasing_l740_74058

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log x

-- State the theorem
theorem min_a_for_monotone_increasing :
  (∃ a₀ : ℝ, ∀ a : ℝ, a ≥ a₀ → 
    (∀ x y : ℝ, 1 < x ∧ x < y ∧ y < 2 → f a x ≤ f a y)) ∧
  (∀ ε > 0, ∃ a : ℝ, a < Real.exp (-1) + ε ∧ 
    ∃ x y : ℝ, 1 < x ∧ x < y ∧ y < 2 ∧ f a x > f a y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_monotone_increasing_l740_74058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_length_on_curve_E_l740_74023

/-- Curve E is defined by the equation y^2 = 4x -/
def curve_E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point A is defined as (4, 0) -/
def point_A : ℝ × ℝ := (4, 0)

/-- A function to calculate the distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Predicate to check if three points form an equilateral triangle -/
def is_equilateral (a b c : ℝ × ℝ) : Prop :=
  distance a b = distance b c ∧ distance b c = distance c a

theorem triangle_length_on_curve_E :
  ∃ (B C : ℝ × ℝ),
    B ∈ curve_E ∧
    C ∈ curve_E ∧
    is_equilateral point_A B C ∧
    distance point_A B = 8 * Real.sqrt 2 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_length_on_curve_E_l740_74023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_c_and_point_p_properties_l740_74090

/-- Circle C with given properties -/
structure CircleC where
  radius : ℝ
  center : ℝ × ℝ
  is_on_ray : center.1 < 0 ∧ center.2 = -2 * center.1
  is_tangent_to_line : |center.1 + center.2 + 1| / Real.sqrt 2 = radius

/-- Point P outside the circle with |PM| = |PO| -/
structure PointP (c : CircleC) where
  coords : ℝ × ℝ
  is_outside : (coords.1 + 1)^2 + (coords.2 - 2)^2 > 2
  tangent_equals_origin : (coords.1 + 1)^2 + (coords.2 - 2)^2 - 2 = coords.1^2 + coords.2^2

/-- Equation of circle C -/
def CircleC.equation (c : CircleC) : ℝ → ℝ → Prop :=
  λ x y ↦ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Area of triangle PMC -/
noncomputable def triangle_pmc_area (c : CircleC) (p : PointP c) : ℝ :=
  1/2 * Real.sqrt ((p.coords.1 + 1)^2 + (p.coords.2 - 2)^2 - 2) * c.radius

/-- Main theorem about CircleC and PointP -/
theorem circle_c_and_point_p_properties (c : CircleC) :
  c.radius = Real.sqrt 2 →
  c.center = (-1, 2) ∧
  ((λ x y ↦ (x + 1)^2 + (y - 2)^2 = 2) : ℝ → ℝ → Prop) = c.equation ∧
  ∃ (p : PointP c),
    triangle_pmc_area c p = 3 * Real.sqrt 10 / 20 ∧
    p.coords = (-3/10, 3/5) ∧
    ∀ (q : PointP c), triangle_pmc_area c q ≥ triangle_pmc_area c p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_c_and_point_p_properties_l740_74090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l740_74009

-- Part 1
theorem sin_bounds (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := by sorry

-- Part 2
noncomputable def f (a x : ℝ) : ℝ := Real.cos (a * x) - Real.log (1 - x^2)

theorem local_max_condition (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f a 0 ≥ f a x) ↔ 
  (a < -Real.sqrt 2 ∨ a > Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l740_74009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l740_74074

-- Define a piecewise linear function
noncomputable def PiecewiseLinearFunction (segments : List ((ℝ × ℝ) × (ℝ × ℝ))) : ℝ → ℝ := 
  fun x => sorry

-- Define our specific function f
noncomputable def f : ℝ → ℝ :=
  PiecewiseLinearFunction [
    ((-4, -5), (-2, -1)),
    ((-2, -1), (-1, -2)),
    ((-1, -2), (1, 2)),
    ((1, 2), (2, 1)),
    ((2, 1), (4, 5))
  ]

-- Define the line y = x + 2
def g (x : ℝ) : ℝ := x + 2

-- Theorem statement
theorem intersection_sum :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f x = g x) ∧ (S.sum id = 3.75) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l740_74074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car1_consumption_l740_74025

/-- Represents the consumption and mileage data for two cars in a family --/
structure TwoCarData where
  avg_mpg_sum : ℚ
  car2_gallons : ℚ
  total_miles : ℚ
  car1_avg_mpg : ℚ

/-- Calculates the number of gallons consumed by the first car --/
def car1_gallons (data : TwoCarData) : ℚ :=
  (data.total_miles - (data.avg_mpg_sum - data.car1_avg_mpg) * data.car2_gallons) / data.car1_avg_mpg

/-- Theorem stating that given the conditions, the first car consumed 26.25 gallons --/
theorem car1_consumption (data : TwoCarData)
  (h1 : data.avg_mpg_sum = 75)
  (h2 : data.car2_gallons = 35)
  (h3 : data.total_miles = 2275)
  (h4 : data.car1_avg_mpg = 40) :
  car1_gallons data = 105/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car1_consumption_l740_74025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_repetition_l740_74071

theorem difference_repetition (a : Fin 8 → ℕ) 
  (h_pos : ∀ i, a i > 0)
  (h_order : ∀ i j, i < j → a i < a j)
  (h_bound : a ⟨7, by norm_num⟩ ≤ 16) :
  ∃ k : ℤ, 3 ≤ (Finset.univ.filter 
    (λ p : Fin 8 × Fin 8 => p.1 ≠ p.2 ∧ (a p.1 : ℤ) - (a p.2 : ℤ) = k)).card :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_repetition_l740_74071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AP_is_four_l740_74084

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  bottomLeft : Point
  topRight : Point

/-- Calculate the area of a rectangle -/
noncomputable def rectangleArea (r : Rectangle) : ℝ :=
  (r.topRight.x - r.bottomLeft.x) * (r.topRight.y - r.bottomLeft.y)

/-- Calculate the area of the intersection of two rectangles -/
noncomputable def intersectionArea (r1 r2 : Rectangle) : ℝ :=
  let left := max r1.bottomLeft.x r2.bottomLeft.x
  let right := min r1.topRight.x r2.topRight.x
  let bottom := max r1.bottomLeft.y r2.bottomLeft.y
  let top := min r1.topRight.y r2.topRight.y
  max 0 (right - left) * max 0 (top - bottom)

theorem length_AP_is_four
  (squareABCD : Rectangle)
  (rectangleWXYZ : Rectangle)
  (h1 : squareABCD.topRight.x - squareABCD.bottomLeft.x = 8)
  (h2 : squareABCD.topRight.y - squareABCD.bottomLeft.y = 8)
  (h3 : rectangleWXYZ.topRight.x - rectangleWXYZ.bottomLeft.x = 12)
  (h4 : rectangleWXYZ.topRight.y - rectangleWXYZ.bottomLeft.y = 8)
  (h5 : squareABCD.bottomLeft.x = rectangleWXYZ.bottomLeft.x)
  (h6 : squareABCD.topRight.y = rectangleWXYZ.topRight.y)
  (h7 : intersectionArea squareABCD rectangleWXYZ = (1/3) * rectangleArea rectangleWXYZ) :
  squareABCD.topRight.y - rectangleWXYZ.bottomLeft.y = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AP_is_four_l740_74084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_conditions_l740_74096

def is_valid_digit (d : ℕ) : Prop := d = 5 ∨ d = 3 ∨ d = 1

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

def all_valid_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_valid_digit d

theorem largest_number_with_conditions (n : ℕ) :
  (all_valid_digits n ∧ digit_sum n = 15) → n ≤ 555 :=
by
  sorry

#eval digit_sum 555

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_conditions_l740_74096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l740_74069

theorem negation_of_proposition :
  (¬(∃ x : ℝ, x > 5 ∧ 2*x^2 - x + 1 > 0)) ↔ (∀ x : ℝ, x > 5 → 2*x^2 - x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l740_74069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_u_l740_74037

theorem min_value_u :
  ∃ (min_u : ℝ), min_u = 12/5 ∧
  ∀ x y : ℝ, -2 < x → x < 2 → -2 < y → y < 2 → x * y = -1 →
  4 / (4 - x^2) + 9 / (9 - y^2) ≥ min_u :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_u_l740_74037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_and_slope_right_angle_no_slope_l740_74067

-- Define a type for lines
structure Line where
  slope : Option ℝ
  angle : ℝ

-- Define a predicate for having an angle of inclination
def has_angle_of_inclination (l : Line) : Prop := True

-- Define a predicate for having a slope
def has_slope (l : Line) : Prop := l.slope.isSome

-- Define a right angle (90 degrees or π/2 radians)
def is_right_angle (a : ℝ) : Prop := a = Real.pi / 2

-- State the theorem
theorem angle_of_inclination_and_slope :
  (∀ l : Line, has_angle_of_inclination l) ∧
  (∃ l : Line, ¬ has_slope l) := by
  constructor
  · intro l
    trivial
  · use { slope := none, angle := Real.pi / 2 }
    simp [has_slope]

-- Additional theorem to capture the condition about right angles
theorem right_angle_no_slope :
  ∀ l : Line,
    has_angle_of_inclination l → is_right_angle l.angle →
    ¬ has_slope l := by
  intro l _ h_right_angle
  simp [has_slope, is_right_angle] at *
  sorry  -- The detailed proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_and_slope_right_angle_no_slope_l740_74067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_characterization_of_unique_m_l740_74021

/-- The number of elements in {k+1, ..., 2k} with exactly three 1s in binary representation -/
def f (k : ℕ+) : ℕ := sorry

/-- For every positive integer m, there exists a positive integer k such that f(k) = m -/
theorem existence_of_k (m : ℕ) : ∃ k : ℕ+, f k = m := sorry

/-- The set of m for which there exists a unique k such that f(k) = m -/
def unique_m_set : Set ℕ :=
  {m | ∃! k : ℕ+, f k = m}

/-- The set of m for which there exists a unique k such that f(k) = m
    is equal to {C(n+1, 2) + 1 | n ∈ ℕ} -/
theorem characterization_of_unique_m :
  unique_m_set = {m : ℕ | ∃ n : ℕ, m = Nat.choose (n + 1) 2 + 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_characterization_of_unique_m_l740_74021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_negative_three_slope_range_l740_74015

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x - 1)

-- Define the intersection points of the line and the parabola
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line_through_focus m p.1 p.2}

-- Part I
theorem dot_product_equals_negative_three (m : ℝ) :
  m = 1 →
  ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧
    A.1 * B.1 + A.2 * B.2 = -3 := by sorry

-- Part II
theorem slope_range (t : ℝ) :
  t ∈ Set.Icc 2 4 →
  ∃ m : ℝ, ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧
    B.1 - 1 = t * (A.1 - 1) ∧ B.2 = t * A.2 ∧
    (m ∈ Set.Icc (-2 * Real.sqrt 2) (-4/3) ∨
     m ∈ Set.Icc (4/3) (2 * Real.sqrt 2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_negative_three_slope_range_l740_74015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_cos_sin_prod_l740_74051

theorem max_value_cos_sin_prod (x : ℝ) : 
  Real.cos x + Real.sin x + Real.cos x * Real.sin x ≤ 1/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_cos_sin_prod_l740_74051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_factorials_trailing_zeroes_l740_74020

/-- Number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- The sum of factorials 53! + 54! has 12 trailing zeroes -/
theorem sum_factorials_trailing_zeroes :
  trailingZeroes 53 = 12 ∧ trailingZeroes 54 = 12 → 
  (∃ k : ℕ, Nat.factorial 53 + Nat.factorial 54 = k * 10^12 ∧ k % 10 ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_factorials_trailing_zeroes_l740_74020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_theorem_l740_74066

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def vector_problem (a b : V) : Prop :=
  ‖a‖ = 1 ∧ 
  inner a b = (1/4 : ℝ) ∧ 
  inner (a + b) (a - b) = (1/2 : ℝ)

theorem vector_theorem (a b : V) (h : vector_problem a b) :
  ‖b‖ = Real.sqrt 2 / 2 ∧
  inner (a + b) (a - b) / (‖a + b‖ * ‖a - b‖) = Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_theorem_l740_74066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heating_pad_cost_per_use_l740_74003

/-- Calculates the cost per use of a heating pad -/
theorem heating_pad_cost_per_use 
  (initial_cost operating_cost : ℚ)
  (uses_per_week weeks : ℕ)
  (h1 : initial_cost = 30)
  (h2 : operating_cost = 1/2)
  (h3 : uses_per_week = 3)
  (h4 : weeks = 2) :
  (initial_cost + operating_cost * (uses_per_week * weeks)) / (uses_per_week * weeks) = 11/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heating_pad_cost_per_use_l740_74003
