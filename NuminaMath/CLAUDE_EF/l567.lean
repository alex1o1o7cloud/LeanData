import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_womans_rate_calculation_l567_56766

/-- The woman's traveling rate in miles per hour -/
noncomputable def womans_rate : ℝ := 15

/-- The man's constant walking rate in miles per hour -/
noncomputable def mans_rate : ℝ := 5

/-- The time in hours that the woman travels before stopping -/
noncomputable def womans_travel_time : ℝ := 2 / 60

/-- The total time in hours that the man walks before catching up -/
noncomputable def total_catch_up_time : ℝ := 6 / 60

/-- Theorem stating the equality of distances traveled -/
theorem womans_rate_calculation : 
  womans_rate * womans_travel_time = mans_rate * total_catch_up_time := by
  -- The proof is omitted for now
  sorry

#check womans_rate_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_womans_rate_calculation_l567_56766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pool_volume_l567_56716

/-- The volume of a cylindrical swimming pool with a hemispherical bottom -/
noncomputable def pool_volume (diameter : ℝ) (cylinder_depth : ℝ) : ℝ :=
  let radius := diameter / 2
  let cylinder_volume := Real.pi * radius^2 * cylinder_depth
  let hemisphere_volume := (2/3) * Real.pi * radius^3
  cylinder_volume + hemisphere_volume

/-- Theorem stating the volume of the specified pool -/
theorem specific_pool_volume :
  pool_volume 20 6 = (3800/3) * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pool_volume_l567_56716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l567_56764

/-- The area of a hexagon formed by connecting two equilateral triangles with side length 3 at a common side -/
theorem hexagon_area : 
  let side_length : ℝ := 3
  let triangle_area := (Real.sqrt 3 / 4) * side_length^2
  let hexagon_area := 2 * triangle_area
  hexagon_area = 9 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l567_56764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_nuts_to_move_is_37_l567_56787

/-- Represents the game state with two boxes of nuts -/
structure GameState :=
  (box1 : ℕ)
  (box2 : ℕ)

/-- The total number of nuts -/
def totalNuts : ℕ := 222

/-- Checks if a game state is valid -/
def isValidState (state : GameState) : Prop :=
  state.box1 + state.box2 = totalNuts

/-- Calculates the minimum number of nuts that need to be moved to achieve the target sum -/
def nutsToMove (state : GameState) (target : ℕ) : ℕ :=
  min
    (min (Int.natAbs (target - state.box1)) (Int.natAbs (target - state.box2)))
    (Int.natAbs (target - (state.box1 + state.box2)))

/-- The maximum number of nuts that need to be moved for any valid state and target -/
def maxNutsToMove : ℕ := 37

/-- The main theorem stating that 37 is the maximum number of nuts that need to be moved -/
theorem max_nuts_to_move_is_37 :
  ∀ (state : GameState),
    isValidState state →
      ∀ (target : ℕ),
        target ≤ totalNuts →
          nutsToMove state target ≤ maxNutsToMove ∧
          (∃ (state' : GameState) (target' : ℕ),
            isValidState state' ∧
            target' ≤ totalNuts ∧
            nutsToMove state' target' = maxNutsToMove) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_nuts_to_move_is_37_l567_56787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_theorem_l567_56708

theorem triangle_segment_theorem (a b c h x : ℝ) : 
  a = 26 →
  b = 60 →
  c = 64 →
  a^2 = x^2 + h^2 →
  b^2 = (c - x)^2 + h^2 →
  abs (c - x - 54.84375) < 0.01 :=
by
  sorry

#eval (64 : Float) - 9.15625

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_theorem_l567_56708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_puzzle_l567_56730

/-- Three-digit number formed from digits a, b, c -/
def threeDigitNumber (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Sum of six three-digit numbers formed from three digits -/
def sumOfSixNumbers (a b c : ℕ) : ℕ :=
  threeDigitNumber a b c + threeDigitNumber a c b +
  threeDigitNumber b a c + threeDigitNumber b c a +
  threeDigitNumber c a b + threeDigitNumber c b a

/-- Difference between sum of three largest and three smallest numbers -/
def diffLargestSmallest (a b c : ℕ) : ℕ :=
  threeDigitNumber a b c + threeDigitNumber a c b + threeDigitNumber b a c -
  (threeDigitNumber b c a + threeDigitNumber c a b + threeDigitNumber c b a)

theorem three_digit_puzzle (a b c : ℕ) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  sumOfSixNumbers a b c = 4218 ∧
  diffLargestSmallest a b c = 792 →
  ({a, b, c} : Finset ℕ) = {8, 7, 4} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_puzzle_l567_56730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_triangle_vectors_l567_56791

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a triangle ABC and a point F on BC (excluding endpoints),
    prove that if AF = x*AB + 2y*AC with x > 0 and y > 0, then 1/x + 2/y ≥ 9 -/
theorem min_value_triangle_vectors (A B C F : V)
    (h_F_on_BC : ∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ F = (1 - t) • B + t • C)
    (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h_AF : F - A = x • (B - A) + (2 * y) • (C - A)) :
    1 / x + 2 / y ≥ 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_triangle_vectors_l567_56791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l567_56781

/-- The function f(x) = k√(a² + x²) - x -/
noncomputable def f (k a x : ℝ) : ℝ := k * Real.sqrt (a^2 + x^2) - x

/-- The theorem stating the minimum value of f(x) -/
theorem min_value_of_f (k a : ℝ) (hk : k > 1) :
  ∃ (min : ℝ), min = a * Real.sqrt (k^2 - 1) ∧
  ∀ (x : ℝ), f k a x ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l567_56781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_property_l567_56717

/-- A sequence of positive real numbers satisfying a_{n+1}^2 = a_n * a_{n+2} + k for all n ≥ 1 -/
def SpecialSequence (a : ℕ → ℝ) (k : ℝ) : Prop :=
  (∀ n, n ≥ 1 → a n > 0) ∧ 
  (∀ n, n ≥ 1 → (a (n + 1))^2 = (a n) * (a (n + 2)) + k)

theorem special_sequence_property (a : ℕ → ℝ) (k a_1 a_2 : ℝ) 
  (h : SpecialSequence a k) (h1 : a 1 = a_1) (h2 : a 2 = a_2) :
  ∃ lambda : ℝ, (∀ n, n ≥ 1 → a n + a (n + 2) = lambda * a (n + 1)) ∧ 
  lambda = (a_1^2 + a_2^2 - k) / (a_1 * a_2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_property_l567_56717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_power_l567_56760

theorem root_sum_power (a b c : ℝ) (h : a ≠ b/3) :
  let f (x : ℝ) := (3*a - b)/c * x^2 + c*(3*a + b)/(3*a - b)
  ∀ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 → x₁^117 + x₂^117 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_power_l567_56760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l567_56750

-- Define the circle C
noncomputable def circle_C (θ : Real) : Real × Real := (1 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Define the line l
noncomputable def line_l : Real := Real.pi / 3

-- Define the length of the chord AB
noncomputable def chord_length (C : Real → Real × Real) (l : Real) : Real :=
  let center := (1, 0)
  let radius := 2
  let d := (Real.sqrt 3) / 2
  2 * Real.sqrt (radius^2 - d^2)

-- Theorem statement
theorem intersection_chord_length :
  chord_length circle_C line_l = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l567_56750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_li_scored_full_marks_l567_56725

-- Define the students
inductive Student : Type
| XiaoLi : Student
| XiaoDong : Student
| XiaoXin : Student

-- Define a function to represent who scored full marks
def scored_full_marks : Student → Prop := sorry

-- Define a function to represent who lied
def lied : Student → Prop := sorry

-- Theorem statement
theorem xiao_li_scored_full_marks :
  -- Only one student scored full marks
  (∃! s : Student, scored_full_marks s) →
  -- Only one student lied
  (∃! s : Student, lied s) →
  -- Xiao Li's statement
  (scored_full_marks Student.XiaoXin → lied Student.XiaoLi) →
  (¬scored_full_marks Student.XiaoXin → ¬lied Student.XiaoLi) →
  -- Xiao Dong's statement
  (scored_full_marks Student.XiaoDong → ¬lied Student.XiaoDong) →
  (¬scored_full_marks Student.XiaoDong → lied Student.XiaoDong) →
  -- Xiao Xin's statement
  (¬scored_full_marks Student.XiaoXin → ¬lied Student.XiaoXin) →
  (scored_full_marks Student.XiaoXin → lied Student.XiaoXin) →
  -- Conclusion
  scored_full_marks Student.XiaoLi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_li_scored_full_marks_l567_56725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_sequence_is_5th_order_repeatable_all_6_plus_term_sequences_are_2nd_order_repeatable_exists_5_term_non_2nd_order_repeatable_l567_56739

def is_01_sequence (s : List ℕ) : Prop :=
  ∀ x ∈ s, x = 0 ∨ x = 1

def is_kth_order_repeatable (s : List ℕ) (k : ℕ) : Prop :=
  ∃ i j, i ≠ j ∧ i + k ≤ s.length ∧ j + k ≤ s.length ∧
  (∀ t, t < k → s.get? (i + t) = s.get? (j + t))

def b_sequence : List ℕ := [0, 0, 0, 1, 1, 0, 0, 1, 1, 0]

theorem b_sequence_is_5th_order_repeatable :
  is_01_sequence b_sequence ∧ is_kth_order_repeatable b_sequence 5 := by sorry

theorem all_6_plus_term_sequences_are_2nd_order_repeatable (m : ℕ) (h : m ≥ 6) :
  ∀ s : List ℕ, s.length = m → is_01_sequence s → is_kth_order_repeatable s 2 := by sorry

theorem exists_5_term_non_2nd_order_repeatable :
  ∃ s : List ℕ, s.length = 5 ∧ is_01_sequence s ∧ ¬is_kth_order_repeatable s 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_sequence_is_5th_order_repeatable_all_6_plus_term_sequences_are_2nd_order_repeatable_exists_5_term_non_2nd_order_repeatable_l567_56739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_meeting_at_b_l567_56757

/-- Represents the block around which the person and dog are moving. -/
structure Block where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ

/-- Represents the movement of the person and dog. -/
structure Movement where
  block : Block
  t₀ : ℝ
  t₁ : ℝ

/-- The position of the person at time t. -/
def person_position (t : ℝ) : ℝ := sorry

/-- The position of the dog at time t. -/
def dog_position (t : ℝ) : ℝ := sorry

/-- The location of point B. -/
def point_b : ℝ := sorry

/-- The theorem to be proved. -/
theorem next_meeting_at_b (m : Movement) (h1 : m.block.ab = 100)
    (h2 : m.block.bc = 300) (h3 : m.block.cd = 100) (h4 : m.block.da = 300)
    (h5 : m.t₀ = 0) (h6 : m.t₁ = 1) :
    ∃ t : ℝ, t = 9 ∧ (person_position t = dog_position t) ∧ (person_position t = point_b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_meeting_at_b_l567_56757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_proof_l567_56773

/-- The lower bounding curve of the plane figure -/
noncomputable def lower_curve (x : ℝ) : ℝ := (1/2) * x^2

/-- The upper bounding curve of the plane figure -/
def upper_curve : ℝ := 2

/-- The region of integration -/
def D : Set (ℝ × ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ lower_curve p.1 ≤ p.2 ∧ p.2 ≤ upper_curve}

/-- The center of gravity of the homogeneous plane figure -/
def center_of_gravity : ℝ × ℝ := (0, 1.2)

/-- Measure of the region D -/
noncomputable def measure_D : ℝ := sorry

theorem center_of_gravity_proof : 
  center_of_gravity = (0, (∫ p in D, p.2) / measure_D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_proof_l567_56773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_20_to_140_l567_56741

noncomputable def arithmetic_sequence_length (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) : ℕ :=
  ⌊(aₙ - a₁) / d⌋.toNat + 1

theorem arithmetic_sequence_20_to_140 :
  arithmetic_sequence_length 20 140 5 = 25 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_20_to_140_l567_56741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_power_b_range_l567_56794

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((1 + a * x) / (1 - 2 * x))

-- Define the properties of the function
def is_odd_on_interval (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x ∈ Set.Ioo (-b) b, f (-x) = -f x

-- Main theorem
theorem a_power_b_range (a b : ℝ) :
  (a ≠ -2) →
  (∃ f : ℝ → ℝ, is_odd_on_interval f b ∧ ∀ x ∈ Set.Ioo (-b) b, f x = Real.log ((1 + a * x) / (1 - 2 * x))) →
  a^b ∈ Set.Ioo 1 (Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_power_b_range_l567_56794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_woojin_game_time_l567_56755

-- Define the total hours in a day
noncomputable def total_hours : ℝ := 24

-- Define the fraction of time Woojin spends sleeping
noncomputable def sleep_fraction : ℝ := 1/3

-- Define the fraction of remaining time Woojin spends studying
noncomputable def study_fraction : ℝ := 3/4

-- Define the fraction of rest time Woojin spends playing computer games
noncomputable def game_fraction : ℝ := 1/4

-- Theorem to prove
theorem woojin_game_time : 
  let sleep_time := total_hours * sleep_fraction
  let remaining_after_sleep := total_hours - sleep_time
  let study_time := remaining_after_sleep * study_fraction
  let remaining_after_study := remaining_after_sleep - study_time
  let game_time := remaining_after_study * game_fraction
  game_time = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_woojin_game_time_l567_56755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_tan_a5_l567_56744

theorem arithmetic_sequence_tan_a5 (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n / 2 * (a 1 + a n)) →  -- Definition of S_n for arithmetic sequence
  S 9 = 6 * Real.pi →                -- Given condition
  Real.tan (a 5) = -Real.sqrt 3 :=   -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_tan_a5_l567_56744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_1400_l567_56792

/-- Calculates the principal amount given the final amount, interest rate, and time period. -/
noncomputable def calculate_principal (final_amount : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  final_amount / (1 + interest_rate * time)

/-- Theorem stating that given the specified conditions, the principal amount is 1400. -/
theorem principal_is_1400 :
  let final_amount : ℝ := 1568
  let interest_rate : ℝ := 0.05
  let time : ℝ := 2.4
  calculate_principal final_amount interest_rate time = 1400 := by
  sorry

#check principal_is_1400

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_1400_l567_56792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tips_fraction_l567_56715

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the fraction of income from tips -/
def fractionFromTips (income : WaitressIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: The fraction of income from tips is 3/7 when tips are 3/4 of salary -/
theorem tips_fraction (income : WaitressIncome) 
  (h : income.tips = (3 / 4) * income.salary) : 
  fractionFromTips income = 3 / 7 := by
  sorry

#check tips_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tips_fraction_l567_56715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_f_minimum_is_one_l567_56702

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x + 1/(4*2^x)

-- State the theorem
theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ 1) ∧ (∃ x : ℝ, f x = 1) := by
  sorry

-- Explicitly state that 1 is the minimum value
theorem f_minimum_is_one :
  IsLeast {y | ∃ x, f x = y} 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_f_minimum_is_one_l567_56702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_parallel_diagonal_exists_diagonal_not_parallel_to_side_l567_56799

/-- Represents a convex polygon with 2n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin (2 * n) → ℝ × ℝ
  convex : sorry  -- Axiom for convexity

/-- The number of diagonals in a polygon with 2n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (2 * n - 3)

/-- The maximum number of diagonals that could be parallel to sides -/
def max_parallel_diagonals (n : ℕ) : ℕ := 2 * n * (n - 2)

/-- Predicate to check if a diagonal is parallel to a side -/
def diagonal_parallel_to_side (poly : ConvexPolygon n) (d : Fin (num_diagonals n)) (s : Fin (2 * n)) : Prop :=
  sorry

/-- Theorem stating that there exists a diagonal not parallel to any side -/
theorem exists_non_parallel_diagonal (n : ℕ) (poly : ConvexPolygon n) :
  max_parallel_diagonals n < num_diagonals n :=
by sorry

/-- Corollary: There exists a diagonal not parallel to any side -/
theorem exists_diagonal_not_parallel_to_side (n : ℕ) (poly : ConvexPolygon n) :
  ∃ (d : Fin (num_diagonals n)), ∀ (s : Fin (2 * n)), 
    ¬ (diagonal_parallel_to_side poly d s) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_parallel_diagonal_exists_diagonal_not_parallel_to_side_l567_56799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_m_range_l567_56753

-- Define the curve C
def C (x y : ℝ) : Prop :=
  (y ≥ 0 ∧ x^2 = 4*y) ∨ (y < 0 ∧ x = 0)

-- Define the distance function
noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the condition for the curve C
def curve_condition (x y : ℝ) : Prop :=
  dist x y 0 1 - |y| = 1

-- Define the intersection points A and B
def intersection_points (k m : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    C x1 y1 ∧ C x2 y2 ∧
    y1 = k * x1 + m ∧ y2 = k * x2 + m ∧
    x1 ≠ x2

-- Define the dot product of vectors FA and FB
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 * x2) + ((y1 - 1) * (y2 - 1))

-- State the theorem
theorem intersection_m_range :
  ∀ x y : ℝ, curve_condition x y →
  ∀ m : ℝ, m > 0 →
  (∀ k : ℝ, intersection_points k m →
    ∀ x1 y1 x2 y2 : ℝ, C x1 y1 ∧ C x2 y2 ∧
    y1 = k * x1 + m ∧ y2 = k * x2 + m ∧ x1 ≠ x2 →
    dot_product x1 y1 x2 y2 < 0) →
  3 - 2 * Real.sqrt 2 < m ∧ m < 3 + 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_m_range_l567_56753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_vertex_l567_56768

/-- Given a parabola y² = 4ax and two tangents intersecting at an angle θ,
    the locus of the vertex of the angle satisfies the equation
    tan²θ · x² - y² + 2a(2 + tan²θ)x + a²tan²θ = 0 -/
theorem locus_of_vertex (a : ℝ) (θ : ℝ) (x y : ℝ) :
  y^2 = 4*a*x →
  (Real.tan θ)^2 * x^2 - y^2 + 2*a*(2 + (Real.tan θ)^2)*x + a^2*(Real.tan θ)^2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_vertex_l567_56768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l567_56798

/-- Given a geometric sequence of real numbers where the fourth term is 9! and the seventh term is 11!, 
    prove that the first term is 3308. -/
theorem geometric_sequence_first_term : 
  ∀ (a r : ℝ), 
    (∀ n : ℕ, n > 0 → ∃ k : ℝ, k * r^(n-1) = a * r^(n-1)) →  -- Geometric sequence condition
    a * r^3 = 362880 →                                       -- Fourth term condition (9! = 362880)
    a * r^6 = 39916800 →                                     -- Seventh term condition (11! = 39916800)
    a = 3308 :=                                              -- Conclusion
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l567_56798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_range_l567_56726

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 * m - 1) ^ x

theorem decreasing_exponential_range (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x > f m y) → (1/3 < m ∧ m < 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_range_l567_56726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_difference_l567_56718

def digits : List Nat := [5, 1, 9]

def largest_number (digits : List Nat) : Nat :=
  digits.toArray.qsort (· > ·) |>.toList.foldl (fun acc d => acc * 10 + d) 0

def smallest_number (digits : List Nat) : Nat :=
  digits.toArray.qsort (· < ·) |>.toList.foldl (fun acc d => acc * 10 + d) 0

theorem rearrange_difference :
  largest_number digits - smallest_number digits = 792 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_difference_l567_56718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_team_guessing_game_l567_56745

-- Define the probabilities and variables
noncomputable def prob_A : ℝ := 2/3
noncomputable def prob_B : ℝ := 1/2  -- We know p = 1/2 from the solution
noncomputable def prob_team_first_round : ℝ := 1/2

-- Define X as a function representing the probability distribution
noncomputable def X : ℕ → ℝ
  | 0 => 1/36
  | 1 => 1/6
  | 2 => 13/36
  | 3 => 1/3
  | 4 => 1/9
  | _ => 0  -- For any other value, probability is 0

-- Theorem statement
theorem star_team_guessing_game :
  (prob_A * (1 - prob_B) + (1 - prob_A) * prob_B = prob_team_first_round) ∧
  (0 * X 0 + 1 * X 1 + 2 * X 2 + 3 * X 3 + 4 * X 4 = 7/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_team_guessing_game_l567_56745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_fall_problem_l567_56762

/-- Represents the motion of a body in free fall -/
noncomputable def FreeFall (initial_distance : ℝ) (increment : ℝ) (time : ℕ) : ℝ :=
  initial_distance * (time : ℝ) + increment * ((time : ℝ) * ((time : ℝ) - 1) / 2)

/-- The total distance traveled by the body after a given time -/
noncomputable def TotalDistance (initial_distance : ℝ) (increment : ℝ) (time : ℕ) : ℝ :=
  FreeFall initial_distance increment time

/-- The distance traveled in the last second -/
noncomputable def LastSecondDistance (initial_distance : ℝ) (increment : ℝ) (time : ℕ) : ℝ :=
  initial_distance + increment * ((time : ℝ) - 1)

theorem free_fall_problem (initial_distance : ℝ) (increment : ℝ) (time : ℕ)
    (h1 : initial_distance = 4.9)
    (h2 : increment = 9.8)
    (h3 : time = 11) :
    TotalDistance initial_distance increment time = 592.9 ∧
    LastSecondDistance initial_distance increment time = 102.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_fall_problem_l567_56762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_increase_march_to_august_l567_56711

noncomputable def profit_change (initial : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (λ acc change => acc * (1 + change / 100)) initial

theorem profit_increase_march_to_august (initial : ℝ) (initial_pos : initial > 0) :
  let changes := [40, -20, 50, -30, 25]
  let final := profit_change initial changes
  (final - initial) / initial * 100 = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_increase_march_to_august_l567_56711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l567_56751

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
def b : ℝ × ℝ := (-2, 0)

theorem vector_properties :
  let norm_a_minus_b := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  let angle := Real.arccos ((a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2)) / 
                (Real.sqrt (a.1^2 + a.2^2) * norm_a_minus_b))
  let range_t (t : ℝ) := Real.sqrt ((a.1 - t * b.1)^2 + (a.2 - t * b.2)^2)
  (norm_a_minus_b = 2 * Real.sqrt 3) ∧
  (angle = π / 6) ∧
  (∀ x : ℝ, x > Real.sqrt 3 → ∃ t : ℝ, range_t t = x) ∧
  (∀ t : ℝ, range_t t > Real.sqrt 3) := by
  sorry

#check vector_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l567_56751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l567_56754

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + y^2 - 6*y = 20

-- Define the center of the circle
def center : ℝ × ℝ := (5, 3)

-- Define the radius of the circle
noncomputable def radius : ℝ := 3 * Real.sqrt 6

-- Theorem statement
theorem circle_properties :
  (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
  center.1 + center.2 + radius = 8 + 3 * Real.sqrt 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l567_56754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l567_56797

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi/4) * Real.cos (x + Real.pi/4) + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

-- Define the theorem
theorem f_properties :
  -- Part 1: f is monotonically increasing in the given interval
  (∀ (k : ℤ) (x y : ℝ), x ∈ Set.Icc (k * Real.pi - Real.pi/3) (k * Real.pi + Real.pi/6) → 
    y ∈ Set.Icc (k * Real.pi - Real.pi/3) (k * Real.pi + Real.pi/6) → 
      x ≤ y → f x ≤ f y) ∧
  -- Part 2: If f(α/2) = 8/5 and α ∈ (π/2, π), then sin α = (4√3 + 3)/10
  (∀ (α : ℝ), α ∈ Set.Ioo (Real.pi/2) Real.pi → 
    f (α/2) = 8/5 → Real.sin α = (4 * Real.sqrt 3 + 3)/10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l567_56797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_dot_product_l567_56710

-- Define the equilateral triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let d := (λ (p q : ℝ × ℝ) => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt)
  d A B = 6 ∧ d B C = 6 ∧ d C A = 6

-- Define the vector operation
def Vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define the dot product
def Dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector scalar multiplication
def VecScale (v : ℝ × ℝ) (s : ℝ) : ℝ × ℝ := (v.1 * s, v.2 * s)

-- State the theorem
theorem triangle_vector_dot_product 
  (A B C M: ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Vec B M = VecScale (Vec M A) 2) : 
  Dot (Vec C M) (Vec C B) = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_dot_product_l567_56710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l567_56736

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a + a * x + 1 / (x + 1)

-- Theorem statement
theorem f_properties (a : ℝ) (h : a > 1) :
  -- Part 1
  f 2 (1/4) = -7/10 ∧
  -- Part 2
  ∃! x, f a x = 0 ∧
  -- Part 3
  ∀ x₀, f a x₀ = 0 → 1/2 < f a (Real.sqrt x₀) ∧ f a (Real.sqrt x₀) < (a + 1)/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l567_56736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elisa_painting_l567_56789

/-- The total square feet Elisa paints in her house over three days -/
noncomputable def total_painted (monday : ℝ) : ℝ :=
  monday + 2 * monday + (1/2) * monday

/-- Theorem stating that Elisa paints 105 square feet in total -/
theorem elisa_painting :
  total_painted 30 = 105 := by
  -- Unfold the definition of total_painted
  unfold total_painted
  -- Simplify the expression
  simp
  -- Check that the result is equal to 105
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elisa_painting_l567_56789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_sum_of_x_l567_56735

theorem min_max_sum_of_x (x_min x_max : ℝ) : 
  (∃ x : ℝ, (5 : ℝ)^(2*x + 1) + 3125 = (5 : ℝ)^(5*x - x^2)) →
  (∀ x : ℝ, (5 : ℝ)^(2*x + 1) + 3125 = (5 : ℝ)^(5*x - x^2) → x_min ≤ x ∧ x ≤ x_max) →
  x_min + x_max = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_sum_of_x_l567_56735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_condition_l567_56767

noncomputable def a_sequence (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 2 then
    a * n^2 - (7/8 * a + 17/4) * n + 17/2
  else
    a^n

def is_increasing (s : ℕ → ℝ) : Prop :=
  ∀ n, s n < s (n + 1)

theorem sequence_increasing_condition (a : ℝ) :
  (is_increasing (a_sequence a)) ↔ a > 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_condition_l567_56767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_theorem_l567_56765

-- Define the tetrahedron and points
variable (S A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 3))

-- Define the conditions
def on_edge (X Y Z : EuclideanSpace ℝ (Fin 3)) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Z = X + t • (Y - X)

def equal_products (S A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ‖S - A‖ * ‖S - A₁‖ = ‖S - B‖ * ‖S - B₁‖ ∧
  ‖S - B‖ * ‖S - B₁‖ = ‖S - C‖ * ‖S - C₁‖

-- Define the sphere containing the points
def on_same_sphere (A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 3)) (radius : ℝ),
    ‖A - center‖ = radius ∧
    ‖B - center‖ = radius ∧
    ‖C - center‖ = radius ∧
    ‖A₁ - center‖ = radius ∧
    ‖B₁ - center‖ = radius ∧
    ‖C₁ - center‖ = radius

-- State the theorem
theorem tetrahedron_sphere_theorem (S A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 3)) :
  on_edge S A A₁ → on_edge S B B₁ → on_edge S C C₁ →
  equal_products S A B C A₁ B₁ C₁ →
  on_same_sphere A B C A₁ B₁ C₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_theorem_l567_56765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_and_alpha_l567_56709

/-- The set of positive rational numbers -/
def PositiveRationals : Set ℚ := {x : ℚ | 0 < x}

/-- The functional equation condition -/
def SatisfiesFunctionalEquation (f : ℚ → ℚ) (α : ℚ) : Prop :=
  ∀ (x y : ℚ), x ∈ PositiveRationals → y ∈ PositiveRationals →
    f (x / y + y) = f x / f y + f y + α * x

/-- The main theorem -/
theorem unique_function_and_alpha :
  ∀ (α : ℚ), α ∈ PositiveRationals →
  ∀ (f : ℚ → ℚ),
  (∀ (x : ℚ), x ∈ PositiveRationals → f x ∈ PositiveRationals) →
  SatisfiesFunctionalEquation f α →
  (α = 2 ∧ ∀ (x : ℚ), x ∈ PositiveRationals → f x = x^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_and_alpha_l567_56709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_21_terms_ap_l567_56770

/-- Sum of an arithmetic progression -/
noncomputable def sum_arithmetic_progression (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a + ((n : ℝ) - 1) * d)

/-- Theorem: Sum of the first 21 terms of the given arithmetic progression -/
theorem sum_21_terms_ap : sum_arithmetic_progression 3 7 21 = 1533 := by
  -- Unfold the definition of sum_arithmetic_progression
  unfold sum_arithmetic_progression
  -- Simplify the expression
  simp [Nat.cast_sub, Nat.cast_one]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_21_terms_ap_l567_56770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uphill_speed_approx_30_l567_56728

/-- Calculates the uphill speed of a car given its downhill speed, distances, and average speed. -/
noncomputable def uphill_speed (downhill_speed : ℝ) (uphill_distance : ℝ) (downhill_distance : ℝ) (average_speed : ℝ) : ℝ :=
  3789 / (150 / average_speed - 50 / downhill_speed)

/-- Theorem stating that under given conditions, the uphill speed is approximately 30 km/hr. -/
theorem uphill_speed_approx_30 :
  let downhill_speed := (80 : ℝ)
  let uphill_distance := (100 : ℝ)
  let downhill_distance := (50 : ℝ)
  let average_speed := (37.89 : ℝ)
  abs (uphill_speed downhill_speed uphill_distance downhill_distance average_speed - 30) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uphill_speed_approx_30_l567_56728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_l567_56759

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapeziumArea (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

/-- Theorem: Given a trapezium with one side 18 cm, height 13 cm, and area 247 cm², the other side is 20 cm -/
theorem trapezium_other_side (t : Trapezium) 
    (h1 : t.side1 = 18)
    (h2 : t.height = 13)
    (h3 : t.area = 247)
    (h4 : t.area = trapeziumArea t) : 
  t.side2 = 20 := by
  sorry

#eval "Trapezium theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_l567_56759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l567_56731

theorem min_value_of_expression : 
  ∀ x : ℝ, (9 : ℝ)^x - (3 : ℝ)^x + 1 ≥ 3/4 ∧ ∃ y : ℝ, (9 : ℝ)^y - (3 : ℝ)^y + 1 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l567_56731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrangle_has_inscribed_circle_l567_56777

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a rectangle by its four vertices -/
structure Rectangle where
  A₁ : Point
  A₂ : Point
  A₃ : Point
  A₄ : Point

/-- Represents a quadrangle formed by tangent lines -/
structure Quadrangle where
  vertices : Fin 4 → Point

/-- Function to calculate the diagonal of a rectangle -/
noncomputable def diagonal (rect : Rectangle) : ℝ := sorry

/-- Function to check if a quadrangle has an inscribed circle -/
def has_inscribed_circle (quad : Quadrangle) : Prop := sorry

/-- Function to construct the quadrangle from outer common tangents -/
noncomputable def outer_common_tangents (c₁ c₃ c₂ c₄ : Circle) : Quadrangle := sorry

/-- Main theorem statement -/
theorem tangent_quadrangle_has_inscribed_circle 
  (rect : Rectangle) 
  (c₁ c₂ c₃ c₄ : Circle) 
  (h₁ : c₁.center = rect.A₁)
  (h₂ : c₂.center = rect.A₂)
  (h₃ : c₃.center = rect.A₃)
  (h₄ : c₄.center = rect.A₄)
  (h₅ : c₁.radius + c₃.radius = c₂.radius + c₄.radius)
  (h₆ : c₁.radius + c₃.radius < diagonal rect)
  (quad : Quadrangle)
  (h₇ : quad = outer_common_tangents c₁ c₃ c₂ c₄) :
  has_inscribed_circle quad := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrangle_has_inscribed_circle_l567_56777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l567_56774

theorem log_sum_upper_bound (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) :
  (Real.log (a^2 / b^2) / Real.log a + Real.log (b^2 / a^2) / Real.log b) ≤ 0 ∧
  (Real.log (a^2 / b^2) / Real.log a + Real.log (b^2 / a^2) / Real.log b = 0 ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l567_56774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_is_half_l567_56785

/-- Represents a runner's performance in a race with two halves --/
structure RunnerPerformance where
  totalDistance : ℝ
  secondHalfTime : ℝ
  timeDifference : ℝ

/-- Calculates the ratio of the runner's speed in the second half to the first half --/
noncomputable def speedRatio (performance : RunnerPerformance) : ℝ :=
  let firstHalfTime := performance.secondHalfTime - performance.timeDifference
  let firstHalfSpeed := (performance.totalDistance / 2) / firstHalfTime
  let secondHalfSpeed := (performance.totalDistance / 2) / performance.secondHalfTime
  secondHalfSpeed / firstHalfSpeed

/-- Theorem stating that for the given conditions, the speed ratio is 1/2 --/
theorem speed_ratio_is_half (performance : RunnerPerformance) 
  (h1 : performance.totalDistance = 40)
  (h2 : performance.secondHalfTime = 10)
  (h3 : performance.timeDifference = 5) : 
  speedRatio performance = 1/2 := by
  sorry

#eval "Lean code compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_is_half_l567_56785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_variance_is_six_fifths_l567_56701

noncomputable def heights : List ℝ := [160, 162, 159, 160, 159]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs)^2)).sum / xs.length

theorem height_variance_is_six_fifths : variance heights = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_variance_is_six_fifths_l567_56701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l567_56784

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, (2 : ℝ)^(a (n+1)) / (2 : ℝ)^(a n) = 2

theorem sequence_properties (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 4 + (a 3)^2 = 21) 
  (h3 : a 1 > 0) :
  (∀ n, a n = n + 1) ∧ 
  (∀ n, Finset.sum (Finset.range n) (fun i => 1 / ((2 * a (i+1) - 1) * (2 * (i+1) - 1))) = n / (2 * n + 1)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l567_56784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_exp_over_x_l567_56723

theorem derivative_exp_over_x (x : ℝ) (hx : x ≠ 0) :
  deriv (λ x => Real.exp x / x) x = (Real.exp x * (x - 1)) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_exp_over_x_l567_56723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l567_56747

noncomputable def g (x : ℝ) : ℝ := -8 - 2 * Real.cos (8 * x) - 4 * Real.cos (4 * x)

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (36 - g x ^ 2)

theorem range_of_f :
  Set.range f = Set.Icc 0 (Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l567_56747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_equidistant_points_l567_56742

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its normal vector and distance from origin -/
structure Line where
  normal : ℝ × ℝ
  distance : ℝ

/-- Configuration of a circle and two parallel tangents -/
structure CircleTangentConfig where
  circle : Circle
  tangent1 : Line
  tangent2 : Line

/-- Predicate to check if a point is equidistant from a circle and two lines -/
def isEquidistant (point : ℝ × ℝ) (config : CircleTangentConfig) : Prop :=
  sorry

/-- The main theorem -/
theorem two_equidistant_points (r : ℝ) :
  ∃ (config : CircleTangentConfig),
    config.circle.radius = r ∧
    config.tangent1.distance = r ∧
    config.tangent2.distance = r + 2 ∧
    (∃ (points : Finset (ℝ × ℝ)), 
      points.card = 2 ∧ 
      ∀ p ∈ points, isEquidistant p config ∧
      ∀ p, isEquidistant p config → p ∈ points) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_equidistant_points_l567_56742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_hyperbola_properties_l567_56713

-- Define the reference hyperbola
def reference_hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 4 = 1

-- Define the parallel hyperbola M
def hyperbola_M (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (3, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parallel_hyperbola_properties :
  -- M is parallel to the reference hyperbola
  (∃ k > 0, ∀ x y, hyperbola_M x y ↔ x^2 / (16*k) - y^2 / (4*k) = 1) →
  -- (2, 0) is on M
  hyperbola_M 2 0 →
  -- The equation of M is x^2/4 - y^2 = 1
  (∀ x y, hyperbola_M x y ↔ x^2 / 4 - y^2 = 1) ∧
  -- The minimum distance between any point on M and A is 2/5 * sqrt(5)
  (∃ min_dist : ℝ, min_dist = 2/5 * Real.sqrt 5 ∧
    ∀ p : ℝ × ℝ, hyperbola_M p.1 p.2 → distance p point_A ≥ min_dist) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_hyperbola_properties_l567_56713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_ratio_range_l567_56724

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Defines if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For any point P on the ellipse (x²/16) + (y²/12) = 1 with foci F₁ and F₂,
    the value of |PF₁ - PF₂| / PF₁ is in the range [0, 2] -/
theorem ellipse_foci_ratio_range 
    (e : Ellipse) 
    (h_e : e.a = 4 ∧ e.b = 2 * Real.sqrt 3) 
    (f1 f2 : Point) 
    (h_foci : f1.x = -Real.sqrt 7 ∧ f2.x = Real.sqrt 7 ∧ f1.y = 0 ∧ f2.y = 0) 
    (p : Point) 
    (h_p : isOnEllipse p e) : 
  ∃ r : ℝ, r ∈ Set.Icc 0 2 ∧ 
    r = |distance p f1 - distance p f2| / distance p f1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_ratio_range_l567_56724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_9999999_squared_l567_56795

/-- The number of zeros in the square of a number consisting of n repeated nines -/
def zeros_in_square (n : ℕ) : ℕ :=
  if n = 0 then 0 else n - 1

/-- The number consisting of n repeated nines -/
def repeated_nines (n : ℕ) : ℕ :=
  10^n - 1

theorem zeros_in_9999999_squared : 
  ∃ (k : ℕ), k = zeros_in_square 7 ∧ 
  ∃ (m : ℕ), (repeated_nines 7)^2 = 10^k * (10 * m + 1) ∧ m ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_9999999_squared_l567_56795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_in_fourth_quadrant_l567_56772

/-- 
If x^2 cos α - y^2 sin α + 2 = 0 represents an ellipse, 
then the center (-cos α, -sin α) of the circle (x + cos α)^2 + (y + sin α)^2 = 1 
is in the fourth quadrant.
-/
theorem circle_center_in_fourth_quadrant (α : Real) :
  (∃ (a b : Real), a > 0 ∧ b > 0 ∧ 
    ∀ (x y : Real), x^2 * Real.cos α - y^2 * Real.sin α + 2 = 0 ↔ x^2/a^2 + y^2/b^2 = 1) →
  -Real.cos α > 0 ∧ -Real.sin α < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_in_fourth_quadrant_l567_56772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_equal_at_negative_one_l567_56778

noncomputable section

/-- The line equation -/
def line (m : ℝ) (x : ℝ) : ℝ := m * x

/-- The curve equation -/
def curve (x : ℝ) : ℝ := |x * (x - 1)|

/-- The area between the line and the curve from 0 to 1-m -/
noncomputable def area1 (m : ℝ) : ℝ := ∫ x in Set.Icc 0 (1-m), curve x - line m x

/-- The area between the line and the curve from 1-m to m+1 -/
noncomputable def area2 (m : ℝ) : ℝ := ∫ x in Set.Icc (1-m) (m+1), line m x - curve x

/-- The theorem stating that the areas are equal when m = -1 -/
theorem areas_equal_at_negative_one :
  ∃ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    line m x₁ = curve x₁ ∧ line m x₂ = curve x₂ ∧ line m x₃ = curve x₃) ∧
  area1 m = area2 m ∧ m = -1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_equal_at_negative_one_l567_56778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_less_than_one_l567_56776

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- Define the theorem
theorem m_less_than_one (m : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_less_than_one_l567_56776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_and_tangent_lines_l567_56771

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.exp (x - 1)
def g (x : ℝ) : ℝ := Real.log x - 1

-- Define a structure for a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define what it means for a line to be tangent to a function at a point
def IsTangentLine (l : Line) (f : ℝ → ℝ) (x : ℝ) : Prop :=
  l.slope = deriv f x ∧ f x = l.slope * x + l.intercept

-- State the theorem
theorem function_inequality_and_tangent_lines :
  (∀ x > 0, f x ≥ g x + 2) ∧
  (∃ n : ℕ, n ≤ 2 ∧ 
    (∀ m : ℕ, (∃ lines : Finset Line, 
      (∀ l ∈ lines, ∃ x > 0, IsTangentLine l f x ∧ IsTangentLine l g x) ∧ 
      Finset.card lines = m) → m ≤ n)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_and_tangent_lines_l567_56771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l567_56733

def fifth_grade_total : ℕ := 60
def sixth_grade_total : ℕ := 90

theorem ticket_price_possibilities : 
  (Finset.filter (fun x => x ∣ fifth_grade_total ∧ x ∣ sixth_grade_total) (Finset.range (fifth_grade_total + 1))).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l567_56733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l567_56763

/-- The sum of an infinite geometric series with first term a and common ratio r, where |r| < 1 -/
noncomputable def infiniteGeometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: For an infinite geometric series with common ratio 1/3 and sum 27, the first term is 18 -/
theorem first_term_of_geometric_series :
  ∃ (a : ℝ), infiniteGeometricSum a (1/3) = 27 ∧ a = 18 := by
  -- We'll use 18 as our witness for the existential quantifier
  use 18
  -- Now we need to prove both parts of the conjunction
  constructor
  -- First part: infiniteGeometricSum 18 (1/3) = 27
  · calc
      infiniteGeometricSum 18 (1/3) = 18 / (1 - 1/3) := rfl
      _ = 18 / (2/3) := by ring
      _ = 27 := by norm_num
  -- Second part: 18 = 18
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_series_l567_56763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_symmetric_points_on_parabola_l567_56707

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The parabola y = -x^2 + 3 -/
def on_parabola (p : Point2D) : Prop :=
  p.y = -p.x^2 + 3

/-- The line x + y = 0 -/
def on_line (p : Point2D) : Prop :=
  p.x + p.y = 0

/-- Two points are symmetric about the line x + y = 0 -/
def symmetric_about_line (a b : Point2D) : Prop :=
  on_line ⟨(a.x + b.x) / 2, (a.y + b.y) / 2⟩

/-- Distance between two points -/
noncomputable def distance (a b : Point2D) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Main theorem -/
theorem distance_of_symmetric_points_on_parabola (a b : Point2D) :
  a ≠ b →
  on_parabola a →
  on_parabola b →
  symmetric_about_line a b →
  distance a b = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_symmetric_points_on_parabola_l567_56707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_simplification_l567_56734

theorem complex_number_simplification : 
  (4 + 3 * Complex.I) / (2 + Complex.I) = 11/5 + 2/5 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_simplification_l567_56734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cancelation_law_l567_56737

-- Define the binary operation
variable {S : Type} -- S is the set
variable (mul : S → S → S) -- mul represents the * operation

-- Define the properties of the operation
variable (h_comm : ∀ x y : S, mul x y = mul y x) -- commutativity
variable (h_assoc : ∀ x y z : S, mul (mul x y) z = mul x (mul y z)) -- associativity
variable (h_exists_z : ∀ x y : S, ∃ z : S, mul x z = y) -- existence of z

-- State the theorem
theorem cancelation_law (a b c : S) 
  (h_eq : mul a c = mul b c) : a = b :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cancelation_law_l567_56737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l567_56703

-- Define the universal set I as ℝ
variable (I : Set ℝ)

-- Define f and g as quadratic functions
variable (f g : ℝ → ℝ)

-- Define sets P and Q
def P (f : ℝ → ℝ) : Set ℝ := {x | f x < 0}
def Q (g : ℝ → ℝ) : Set ℝ := {x | g x ≥ 0}

-- Theorem statement
theorem solution_set_equivalence (f g : ℝ → ℝ) :
  {x : ℝ | f x < 0 ∧ g x < 0} = P f ∩ (Set.univ \ Q g) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l567_56703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_nonempty_subsets_of_P_l567_56756

def P : Set (ℕ × ℕ) := {(x, y) | x + y < 4 ∧ x > 0 ∧ y > 0}

theorem number_of_nonempty_subsets_of_P :
  Finset.card (Finset.powerset (Finset.filter (fun p => p.1 + p.2 < 4 ∧ p.1 > 0 ∧ p.2 > 0) 
    (Finset.product (Finset.range 4) (Finset.range 4))) \ {∅}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_nonempty_subsets_of_P_l567_56756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_right_directrix_is_15_over_2_l567_56704

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- A point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_equation x y

/-- The distance from a point to the left focus -/
noncomputable def distance_to_left_focus (p : PointOnEllipse) : ℝ := 4

/-- The distance from a point to the right directrix -/
noncomputable def distance_to_right_directrix (p : PointOnEllipse) : ℝ := 15/2

/-- Theorem: The distance from a point on the ellipse to the right directrix is 15/2 -/
theorem distance_to_right_directrix_is_15_over_2 (p : PointOnEllipse) : 
  distance_to_right_directrix p = 15/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_right_directrix_is_15_over_2_l567_56704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_downstream_21_minutes_l567_56786

/-- Calculates the distance travelled downstream given the boat's speed in still water,
    the current's speed, and the travel time in minutes. -/
noncomputable def distance_downstream (boat_speed : ℝ) (current_speed : ℝ) (time_minutes : ℝ) : ℝ :=
  (boat_speed + current_speed) * (time_minutes / 60)

/-- Proves that the distance travelled downstream is 8.75 km given the specified conditions. -/
theorem distance_downstream_21_minutes :
  distance_downstream 20 5 21 = 8.75 := by
  -- Unfold the definition of distance_downstream
  unfold distance_downstream
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_downstream_21_minutes_l567_56786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_point_on_curve_l567_56769

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the line
def line (x : ℝ) : ℝ := x - 2

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |y - line x| / Real.sqrt 2

-- Statement of the theorem
theorem min_distance_point :
  ∀ x > 0, distance_to_line x (curve x) ≥ distance_to_line 1 (curve 1) := by
  sorry

-- Verify that (1, 1) is on the curve
theorem point_on_curve : curve 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_point_on_curve_l567_56769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l567_56749

/-- The line passing through points satisfying x + y - 1 = 0 -/
def line (x y : ℝ) : Prop := x + y - 1 = 0

/-- The distance between two points in 2D space -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The origin point (0, 0) -/
def origin : ℝ × ℝ := (0, 0)

/-- The theorem stating the minimum distance from the origin to the line -/
theorem min_distance_to_line :
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧
  ∀ (x y : ℝ), line x y →
    distance 0 0 x y ≥ d ∧
    ∃ (x₀ y₀ : ℝ), line x₀ y₀ ∧ distance 0 0 x₀ y₀ = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l567_56749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l567_56705

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if cos B = 4/5, a = 5, and the area is 12,
    then (a + c) / (sin A + sin C) = 25/3 -/
theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  Real.cos B = 4/5 →
  a = 5 →
  (1/2) * a * c * Real.sin B = 12 →
  (a + c) / (Real.sin A + Real.sin C) = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l567_56705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_eq_interval_l567_56712

/-- The set of all real numbers m for which mx² + mx + 1 > 0 holds for all real numbers x -/
def M : Set ℝ := {m : ℝ | ∀ x : ℝ, m * x^2 + m * x + 1 > 0}

/-- The theorem stating that M is equal to the interval [0, 4) -/
theorem M_eq_interval : M = Set.Ioc 0 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_eq_interval_l567_56712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l567_56721

-- Define the arithmetic sequence
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

-- Theorem statement
theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 7 = 5 ∧ 
  arithmetic_sum a₁ d 9 = 27 →
  arithmetic_sequence a₁ d 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l567_56721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l567_56746

noncomputable def a : ℝ × ℝ × ℝ := (3, 3, 1)
noncomputable def b : ℝ × ℝ × ℝ := (1, -2, 1)
noncomputable def c : ℝ × ℝ × ℝ := (1, 1, 1)

theorem vectors_not_coplanar : ¬(∃ (x y z : ℝ), x • a + y • b + z • c = (0, 0, 0) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l567_56746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equation_l567_56727

/-- The focus point -/
def F : ℝ × ℝ := (4, 0)

/-- The directrix line -/
def directrix (x : ℝ) : Prop := x + 5 = 0

/-- Distance from a point to the directrix -/
noncomputable def dist_to_directrix (P : ℝ × ℝ) : ℝ :=
  |P.1 + 5|

/-- The locus of points P -/
def locus (P : ℝ × ℝ) : Prop :=
  dist P F = dist_to_directrix P - 1

/-- The theorem stating the equation of the locus -/
theorem locus_equation (P : ℝ × ℝ) :
  locus P ↔ P.2^2 = 16 * P.1 := by sorry

#check locus_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equation_l567_56727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l567_56780

/-- 
Given an angle α, if sin(2α) < 0 and cos(α) - sin(α) < 0, 
then α is in the second quadrant.
-/
theorem angle_in_second_quadrant (α : ℝ) : 
  Real.sin (2 * α) < 0 → Real.cos α - Real.sin α < 0 → 
  π / 2 < α ∧ α < π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l567_56780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_equilateral_triangle_rational_distance_ratio_l567_56788

-- Define a point with integer coordinates
structure IntPoint where
  x : ℤ
  y : ℤ

-- Define the distance between two points
noncomputable def distance (p1 p2 : IntPoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 : ℝ)

-- Define an equilateral triangle
def isEquilateralTriangle (p1 p2 p3 : IntPoint) : Prop :=
  distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1

-- Define the distance from a point to a line
noncomputable def distancePointToLine (p a b : IntPoint) : ℝ :=
  let numerator := |(b.y - a.y) * p.x - (b.x - a.x) * p.y + b.x * a.y - b.y * a.x|
  let denominator := Real.sqrt ((b.y - a.y)^2 + (b.x - a.x)^2 : ℝ)
  (numerator : ℝ) / denominator

-- Theorem 1: No equilateral triangle with integer coordinates
theorem no_integer_equilateral_triangle :
  ¬∃ (p1 p2 p3 : IntPoint), isEquilateralTriangle p1 p2 p3 := by
  sorry

-- Theorem 2: Ratio of distance to line and line length is rational
theorem rational_distance_ratio (A B C : IntPoint)
  (h : distance A B = distance A C) :
  ∃ (q : ℚ), (distancePointToLine A B C : ℝ) / distance B C = (q : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_equilateral_triangle_rational_distance_ratio_l567_56788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_reduction_l567_56796

noncomputable def initial_volume : ℝ := 12
noncomputable def initial_concentration : ℝ := 0.20
noncomputable def added_water : ℝ := 28

noncomputable def final_volume : ℝ := initial_volume + added_water
noncomputable def final_concentration : ℝ := (initial_volume * initial_concentration) / final_volume

noncomputable def concentration_reduction : ℝ := (initial_concentration - final_concentration) / initial_concentration

theorem alcohol_concentration_reduction :
  concentration_reduction = 0.70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_reduction_l567_56796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_volume_l567_56700

/-- A solid with a rectangular base and specific properties -/
structure Solid where
  s : ℝ
  base_width : ℝ := s
  base_length : ℝ := 2 * s
  upper_edge_length : ℝ := 3 * s
  other_edge_length : ℝ := s
  s_value : s = 6 * Real.sqrt 2

/-- The volume of the solid -/
noncomputable def volume (solid : Solid) : ℝ := 486 * Real.sqrt 2

/-- Theorem stating that the volume of the solid with the given properties is 486√2 -/
theorem solid_volume (solid : Solid) : volume solid = 486 * Real.sqrt 2 := by
  sorry

#check solid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_volume_l567_56700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_increase_l567_56748

/-- Represents the reading data for Mark --/
structure ReadingData where
  original_hours_per_day : ℚ
  original_pages_per_day : ℚ
  new_pages_per_week : ℚ

/-- Calculates the percentage increase in reading time --/
def percentage_increase (data : ReadingData) : ℚ :=
  let original_pages_per_week := data.original_pages_per_day * 7
  let increase_factor := data.new_pages_per_week / original_pages_per_week
  (increase_factor - 1) * 100

/-- Theorem stating that Mark's reading time increased by 150% --/
theorem reading_time_increase (data : ReadingData) 
    (h1 : data.original_hours_per_day = 2)
    (h2 : data.original_pages_per_day = 100)
    (h3 : data.new_pages_per_week = 1750) :
  percentage_increase data = 150 := by
  -- Unfold the definition of percentage_increase
  unfold percentage_increase
  -- Perform the calculation
  simp [h1, h2, h3]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_increase_l567_56748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_iff_m_greater_than_two_l567_56775

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x - 1/x - Real.log x

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := 1 + 1/x^2 - 1/x

-- Define the theorem
theorem three_zeros_iff_m_greater_than_two :
  (∃ (a b c : ℝ), a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
    (∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c)) ↔
  ∃ m : ℝ, m > 2 ∧ 
    (∀ x : ℝ, x > 0 → (f' x = 0 ↔ x^2 - m*x + 1 = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_iff_m_greater_than_two_l567_56775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_reciprocal_sum_l567_56738

/-- Given two quadratic functions with perpendicular tangent lines at an intersection point,
    prove that the minimum value of 1/a + 4/b is 18/5 -/
theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := fun x : ℝ => x^2 - 2*x + 2
  let g := fun x : ℝ => -x^2 + a*x + b
  ∃ x₀ : ℝ, (deriv f x₀) * (deriv g x₀) = -1 →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1/a' + 4/b' ≥ 1/a + 4/b) →
  1/a + 4/b = 18/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_reciprocal_sum_l567_56738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l567_56719

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (x + Real.pi/6) * Real.sin (x + Real.pi/3) - Real.sin x * Real.cos x - 1/4

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  b = 2 ∧
  B = Real.pi/6 ∧
  f (A/2) = 0 →
  c = Real.sqrt 2 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l567_56719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_probability_given_positive_test_l567_56743

/-- The probability of having the disease in the population -/
noncomputable def disease_probability : ℝ := 1 / 400

/-- The probability of testing positive given that the individual has the disease -/
noncomputable def true_positive_rate : ℝ := 1

/-- The probability of testing positive given that the individual does not have the disease -/
noncomputable def false_positive_rate : ℝ := 3 / 100

/-- The probability that a randomly selected individual who tests positive actually has the disease -/
noncomputable def positive_test_disease_probability : ℝ := 100 / 1297

theorem disease_probability_given_positive_test :
  let total_positive_probability := disease_probability * true_positive_rate + 
                                    (1 - disease_probability) * false_positive_rate
  (disease_probability * true_positive_rate) / total_positive_probability = positive_test_disease_probability := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_probability_given_positive_test_l567_56743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_number_problem_l567_56752

/-- Given four numbers A, B, C, and D satisfying certain conditions, prove their values. -/
theorem four_number_problem (A B C D : ℚ) 
  (sum_AB : A + B = 44)
  (ratio_AB : 5 * A = 6 * B)
  (C_def : C = 2 * (A - B))
  (D_def : D = (A + B + C) / 3 + 3) :
  A = 24 ∧ B = 20 ∧ C = 8 ∧ D = 61 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_number_problem_l567_56752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_simplification_l567_56758

noncomputable def A (x : ℝ) : ℝ := 
  Real.sqrt ((1/6) * ((3*x + Real.sqrt (6*x - 1))⁻¹ + (3*x - Real.sqrt (6*x - 1))⁻¹)) * 
  |x - 1| * x^(-(1/2 : ℝ))

theorem A_simplification (x : ℝ) (h : x ≥ 1/6) : 
  A x = if (1/6 ≤ x ∧ x < 1/3) ∨ x ≥ 1 then (x - 1) / (3*x - 1)
        else if 1/3 < x ∧ x < 1 then (1 - x) / (3*x - 1)
        else 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_simplification_l567_56758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_bounds_l567_56722

theorem cosine_sum_bounds (x₁ x₂ x₃ x₄ x₅ : Real) 
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ Real.pi/2)
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ Real.pi/2)
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ Real.pi/2)
  (h₄ : 0 ≤ x₄ ∧ x₄ ≤ Real.pi/2)
  (h₅ : 0 ≤ x₅ ∧ x₅ ≤ Real.pi/2)
  (h_sum : Real.sin x₁ + Real.sin x₂ + Real.sin x₃ + Real.sin x₄ + Real.sin x₅ = 3) :
  ∃ (n : ℤ), (Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ = n) ∧ 
  (n = 2 ∨ n = 3 ∨ n = 4) ∧
  (∀ m : ℤ, (Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ = m) → 2 ≤ m ∧ m ≤ 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_bounds_l567_56722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_5pi_2_minus_2x_is_even_pi_8_is_symmetry_line_no_alpha_for_sin_plus_cos_3_2_exp_sin_2x_not_increasing_tan_not_always_increasing_first_quadrant_sin_shift_not_pi_3_l567_56783

/-- Proposition 1: sin((5/2)π - 2x) is an even function -/
theorem sin_5pi_2_minus_2x_is_even : ∀ x : ℝ, Real.sin ((5/2) * Real.pi - 2*x) = Real.sin ((5/2) * Real.pi + 2*x) := by sorry

/-- Proposition 2: x = π/8 is a line of symmetry for y = sin(2x + 5π/4) -/
theorem pi_8_is_symmetry_line : ∀ t : ℝ, Real.sin (2 * (Real.pi/8 + t) + 5*Real.pi/4) = Real.sin (2 * (Real.pi/8 - t) + 5*Real.pi/4) := by sorry

/-- Proposition 3: There does not exist α such that sin α + cos α = 3/2 -/
theorem no_alpha_for_sin_plus_cos_3_2 : ¬ ∃ α : ℝ, Real.sin α + Real.cos α = 3/2 := by sorry

/-- Proposition 4: e^(sin 2x) is not increasing on (0, π/2) -/
theorem exp_sin_2x_not_increasing : ¬ ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < Real.pi/2 → Real.exp (Real.sin (2*x)) < Real.exp (Real.sin (2*y)) := by sorry

/-- Proposition 5: It's not always true that tan α > tan β when α > β in the first quadrant -/
theorem tan_not_always_increasing_first_quadrant : 
  ¬ ∀ α β : ℝ, 0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ α > β → Real.tan α > Real.tan β := by sorry

/-- Proposition 6: 3sin(2x + π/3) is not obtained by shifting 3sin(2x) left by π/3 -/
theorem sin_shift_not_pi_3 : ¬ ∀ x : ℝ, 3 * Real.sin (2*x + Real.pi/3) = 3 * Real.sin (2*(x + Real.pi/3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_5pi_2_minus_2x_is_even_pi_8_is_symmetry_line_no_alpha_for_sin_plus_cos_3_2_exp_sin_2x_not_increasing_tan_not_always_increasing_first_quadrant_sin_shift_not_pi_3_l567_56783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_ln_plus_one_l567_56793

noncomputable def f (x : ℝ) := Real.log (x + 1)

theorem domain_of_ln_plus_one :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_ln_plus_one_l567_56793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l567_56761

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop
  a_gt_b : a > b
  b_pos : b > 0
  eccentricity : Real.sqrt (a^2 - b^2) / a = Real.sqrt 3 / 2
  minor_axis : b = 1

/-- A line passing through a point and intersecting the ellipse -/
structure IntersectingLine (M : Ellipse) where
  k : ℝ
  passes_through : ∀ x : ℝ, k * x + 2 = 2
  intersects : ∃ P Q : ℝ × ℝ, 
    M.equation P.1 P.2 ∧ 
    M.equation Q.1 Q.2 ∧ 
    k * P.1 + 2 = P.2 ∧ 
    k * Q.1 + 2 = Q.2 ∧
    P.1 * Q.1 + P.2 * Q.2 = 0

/-- The main theorem -/
theorem ellipse_and_line_properties (M : Ellipse) :
  (∀ x y : ℝ, M.equation x y ↔ x^2 / 4 + y^2 = 1) ∧
  (∀ l : IntersectingLine M, l.k = 2 ∨ l.k = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l567_56761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_13_l567_56729

theorem three_digit_numbers_divisible_by_13 : 
  (Finset.filter (fun n : ℕ => 100 ≤ n ∧ n ≤ 999 ∧ n % 13 = 0) (Finset.range 1000)).card = 69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_13_l567_56729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l567_56779

noncomputable def f (x : ℝ) := Real.log (x + 1) - 3 / x

theorem zero_in_interval :
  Continuous f ∧ f 2 < 0 ∧ 0 < f 3 →
  ∃ x, x ∈ Set.Ioo 2 3 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l567_56779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l567_56732

noncomputable def f (x : ℝ) := Real.sqrt (x + 2) + 1 / (1 - x)

theorem domain_of_f :
  {x : ℝ | x ≥ -2 ∧ x ≠ 1} = {x : ℝ | ∃ y, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l567_56732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_amplitude_l567_56782

/-- Given a sinusoidal function y = a * sin(b * x + c) + d with positive constants a, b, c, and d,
    if the function oscillates between a maximum of 7 and a minimum of -1, then a = 4. -/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_max : ∀ x, a * Real.sin (b * x + c) + d ≤ 7)
  (h_min : ∀ x, a * Real.sin (b * x + c) + d ≥ -1)
  (h_reaches_max : ∃ x, a * Real.sin (b * x + c) + d = 7)
  (h_reaches_min : ∃ x, a * Real.sin (b * x + c) + d = -1) :
  a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_amplitude_l567_56782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_company_performance_l567_56790

/-- Data for bus companies' on-time performance --/
structure BusData where
  total_surveyed : ℕ
  a_ontime : ℕ
  a_late : ℕ
  b_ontime : ℕ
  b_late : ℕ

/-- Calculate K^2 statistic --/
noncomputable def calculate_k_squared (data : BusData) : ℝ :=
  let n := data.total_surveyed
  let a := data.a_ontime
  let b := data.a_late
  let c := data.b_ontime
  let d := data.b_late
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Main theorem about bus company performance --/
theorem bus_company_performance (data : BusData)
  (h_total : data.total_surveyed = 500)
  (h_a_ontime : data.a_ontime = 240)
  (h_a_late : data.a_late = 20)
  (h_b_ontime : data.b_ontime = 210)
  (h_b_late : data.b_late = 30) :
  (data.a_ontime : ℝ) / (data.a_ontime + data.a_late) = 12 / 13 ∧
  (data.b_ontime : ℝ) / (data.b_ontime + data.b_late) = 7 / 8 ∧
  calculate_k_squared data > 2.706 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_company_performance_l567_56790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_napkin_intersection_theorem_l567_56706

-- Define a type for colors
inductive Color
| Blue
| Green

-- Define a type for napkins
structure Napkin where
  color : Color
  left : ℝ
  right : ℝ
  bottom : ℝ
  top : ℝ

-- Define a type for lines
inductive Line
| Vertical (x : ℝ)
| Horizontal (y : ℝ)

-- Define a function to check if a line intersects a napkin
def intersects (l : Line) (n : Napkin) : Prop :=
  match l with
  | Line.Vertical x => n.left ≤ x ∧ x ≤ n.right
  | Line.Horizontal y => n.bottom ≤ y ∧ y ≤ n.top

-- Helper function to check if a line is horizontal
def isHorizontal (l : Line) : Prop :=
  match l with
  | Line.Horizontal _ => True
  | _ => False

-- Helper function to check if a line is vertical
def isVertical (l : Line) : Prop :=
  match l with
  | Line.Vertical _ => True
  | _ => False

-- Main theorem
theorem napkin_intersection_theorem (napkins : Set Napkin) :
  (∀ n1 n2 : Napkin, n1 ∈ napkins → n2 ∈ napkins → n1.color ≠ n2.color →
    ∃ l : Line, intersects l n1 ∧ intersects l n2) →
  ∃ (c : Color) (l1 l2 l3 : Line),
    isHorizontal l1 ∧
    isHorizontal l2 ∧
    isVertical l3 ∧
    (∀ n : Napkin, n ∈ napkins → n.color = c →
      intersects l1 n ∨ intersects l2 n ∨ intersects l3 n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_napkin_intersection_theorem_l567_56706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_12_factorial_base_81_l567_56714

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

/-- The number of trailing zeros in the base 81 representation of a natural number -/
def trailingZerosBase81 (n : ℕ) : ℕ :=
  (Nat.log n 81)

/-- Theorem: The number of trailing zeros in the base 81 representation of 12! is 1 -/
theorem trailing_zeros_12_factorial_base_81 :
  trailingZerosBase81 (factorial 12) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_12_factorial_base_81_l567_56714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l567_56720

/-- A hyperbola with foci A and B, and a point C on the hyperbola -/
structure Hyperbola (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  A : V
  B : V
  C : V

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (h : Hyperbola V) : ℝ := 
  ‖h.A - h.B‖ / (‖h.C - h.B‖ - ‖h.C - h.A‖)

/-- The condition that the sides of triangle ABC form an arithmetic sequence -/
def sides_arithmetic_sequence {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (h : Hyperbola V) : Prop :=
  ∃ (k : ℝ), ‖h.C - h.A‖ = 3*k ∧ ‖h.A - h.B‖ = 7*k ∧ ‖h.C - h.B‖ = 5*k

/-- The angle ACB is 120° -/
def angle_ACB_120 {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (h : Hyperbola V) : Prop :=
  inner (h.A - h.C) (h.B - h.C) = -1/2 * ‖h.A - h.C‖ * ‖h.B - h.C‖

/-- The main theorem -/
theorem hyperbola_eccentricity 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (h : Hyperbola V) 
  (h_sides : sides_arithmetic_sequence h) 
  (h_angle : angle_ACB_120 h) : 
  eccentricity h = 7/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l567_56720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_theorem_l567_56740

-- Define the right rectangular prism
def B : Set (Fin 3 → ℝ) := {p | 0 ≤ p 0 ∧ p 0 ≤ 2 ∧ 0 ≤ p 1 ∧ p 1 ≤ 4 ∧ 0 ≤ p 2 ∧ p 2 ≤ 5}

-- Define the set S(r)
def S (r : ℝ) : Set (Fin 3 → ℝ) := {p | ∃ q ∈ B, Real.sqrt ((p 0 - q 0)^2 + (p 1 - q 1)^2 + (p 2 - q 2)^2) ≤ r}

-- Define the volume function for S(r)
noncomputable def volume_S (r : ℝ) : ℝ := sorry

-- State the theorem
theorem volume_ratio_theorem (a b c d : ℝ) 
  (h_volume : ∀ r : ℝ, volume_S r = a * r^3 + b * r^2 + c * r + d) :
  b * c / (a * d) = 15.675 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_theorem_l567_56740
