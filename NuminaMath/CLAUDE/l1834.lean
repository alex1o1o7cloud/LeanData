import Mathlib

namespace NUMINAMATH_CALUDE_exists_large_number_with_exchangeable_digits_l1834_183437

/-- A function that checks if two natural numbers have the same set of prime divisors -/
def samePrimeDivisors (a b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ a ↔ p ∣ b)

/-- A function that checks if a number can have two distinct non-zero digits exchanged -/
def canExchangeDigits (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ) (k m : ℕ),
    d₁ ≠ d₂ ∧ d₁ ≠ 0 ∧ d₂ ≠ 0 ∧
    (∃ n₁ n₂ : ℕ,
      n₁ = n + (d₁ - d₂) * 10^k ∧
      n₂ = n + (d₂ - d₁) * 10^m ∧
      samePrimeDivisors n₁ n₂)

/-- The main theorem -/
theorem exists_large_number_with_exchangeable_digits :
  ∃ n : ℕ, n > 10^1000 ∧ ¬(10 ∣ n) ∧ canExchangeDigits n :=
sorry

end NUMINAMATH_CALUDE_exists_large_number_with_exchangeable_digits_l1834_183437


namespace NUMINAMATH_CALUDE_snail_well_depth_l1834_183475

/-- The minimum depth of a well that allows a snail to reach the top during the day on the fifth day,
    given its daily climbing and nightly sliding distances. -/
def min_well_depth (day_climb : ℕ) (night_slide : ℕ) : ℕ :=
  (day_climb - night_slide) * 3 + day_climb + 1

/-- Theorem stating the minimum well depth for a snail with specific climbing characteristics. -/
theorem snail_well_depth :
  min_well_depth 110 40 = 321 := by
  sorry

#eval min_well_depth 110 40

end NUMINAMATH_CALUDE_snail_well_depth_l1834_183475


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1834_183445

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ + 3 = 0) → 
  (x₂^2 - 4*x₂ + 3 = 0) → 
  (x₁ + x₂ = 4) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1834_183445


namespace NUMINAMATH_CALUDE_expression_evaluation_l1834_183412

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℚ := -3
  (15 * x^3 * y - 10 * x^2 * y^2) / (5 * x * y) - (3*x + y) * (x - 3*y) = 18 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1834_183412


namespace NUMINAMATH_CALUDE_square_perimeter_l1834_183472

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ w * h = s^2 / 4 ∧ 2*(w + h) = 40) → 
  4 * s = 64 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l1834_183472


namespace NUMINAMATH_CALUDE_problem_solution_l1834_183473

theorem problem_solution (x : ℝ) (h_pos : x > 0) :
  x^(2 * x^6) = 3 → x = (3 : ℝ)^(1/6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1834_183473


namespace NUMINAMATH_CALUDE_equation_solution_range_l1834_183469

theorem equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 1 ∧ (2 * x + m) / (x - 1) = 1) → 
  (m ≤ -1 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l1834_183469


namespace NUMINAMATH_CALUDE_infinitely_many_very_good_pairs_l1834_183493

/-- A pair of natural numbers is "good" if they consist of the same prime divisors, possibly in different powers. -/
def isGoodPair (m n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)

/-- A pair of natural numbers is "very good" if both the pair and their successors form "good" pairs. -/
def isVeryGoodPair (m n : ℕ) : Prop :=
  isGoodPair m n ∧ isGoodPair (m + 1) (n + 1) ∧ m ≠ n

/-- There exist infinitely many "very good" pairs of natural numbers. -/
theorem infinitely_many_very_good_pairs :
  ∀ k : ℕ, ∃ m n : ℕ, m > k ∧ n > k ∧ isVeryGoodPair m n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_very_good_pairs_l1834_183493


namespace NUMINAMATH_CALUDE_car_overtake_distance_l1834_183496

theorem car_overtake_distance (speed_a speed_b time_to_overtake distance_ahead : ℝ) 
  (h1 : speed_a = 58)
  (h2 : speed_b = 50)
  (h3 : time_to_overtake = 4)
  (h4 : distance_ahead = 8) :
  speed_a * time_to_overtake - speed_b * time_to_overtake - distance_ahead = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_overtake_distance_l1834_183496


namespace NUMINAMATH_CALUDE_ratio_a_to_d_l1834_183449

theorem ratio_a_to_d (a b c d : ℚ) : 
  a / b = 8 / 3 →
  b / c = 1 / 5 →
  c / d = 3 / 2 →
  b = 27 →
  a / d = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_d_l1834_183449


namespace NUMINAMATH_CALUDE_boys_count_l1834_183407

theorem boys_count (total_students : ℕ) (girls_ratio boys_ratio : ℕ) (h1 : total_students = 30) 
  (h2 : girls_ratio = 1) (h3 : boys_ratio = 2) : 
  (total_students * boys_ratio) / (girls_ratio + boys_ratio) = 20 := by
  sorry

end NUMINAMATH_CALUDE_boys_count_l1834_183407


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1834_183428

def A : Set ℝ := {x | -2 < x ∧ x ≤ 2}
def B : Set ℝ := {-2, -1, 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1834_183428


namespace NUMINAMATH_CALUDE_arctangent_sum_equals_pi_over_four_l1834_183484

theorem arctangent_sum_equals_pi_over_four :
  ∃ (n : ℕ+), (Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/6) + Real.arctan (1/n) = π/4) ∧ n = 113 := by
  sorry

end NUMINAMATH_CALUDE_arctangent_sum_equals_pi_over_four_l1834_183484


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1834_183470

theorem arithmetic_sequence_sum (N : ℤ) : 
  (1001 : ℤ) + 1004 + 1007 + 1010 + 1013 = 5050 - N → N = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1834_183470


namespace NUMINAMATH_CALUDE_parallel_to_y_axis_second_quadrant_equal_distance_l1834_183487

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Define point Q
def Q : ℝ × ℝ := (4, 5)

-- Part 1
theorem parallel_to_y_axis (a : ℝ) :
  (P a).1 = Q.1 → P a = (4, 8) := by sorry

-- Part 2
theorem second_quadrant_equal_distance (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ -(P a).1 = (P a).2 → a^2023 + a^(1/3) = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_to_y_axis_second_quadrant_equal_distance_l1834_183487


namespace NUMINAMATH_CALUDE_distance_between_x_intercepts_specific_case_l1834_183410

/-- Two lines in a 2D plane -/
structure TwoLines where
  intersection : ℝ × ℝ
  slope1 : ℝ
  slope2 : ℝ

/-- Calculate the distance between x-intercepts of two lines -/
def distance_between_x_intercepts (lines : TwoLines) : ℝ :=
  sorry

/-- The main theorem -/
theorem distance_between_x_intercepts_specific_case :
  let lines : TwoLines := {
    intersection := (12, 20),
    slope1 := 7/2,
    slope2 := -3/2
  }
  distance_between_x_intercepts lines = 800/21 := by sorry

end NUMINAMATH_CALUDE_distance_between_x_intercepts_specific_case_l1834_183410


namespace NUMINAMATH_CALUDE_jacqueline_apples_l1834_183418

/-- The number of plums Jacqueline had initially -/
def plums : ℕ := 16

/-- The number of guavas Jacqueline had initially -/
def guavas : ℕ := 18

/-- The number of fruits Jacqueline gave to Jane -/
def given_fruits : ℕ := 40

/-- The number of fruits Jacqueline had left after giving some to Jane -/
def left_fruits : ℕ := 15

/-- The number of apples Jacqueline had initially -/
def apples : ℕ := 21

theorem jacqueline_apples :
  plums + guavas + apples = given_fruits + left_fruits :=
sorry

end NUMINAMATH_CALUDE_jacqueline_apples_l1834_183418


namespace NUMINAMATH_CALUDE_bells_toll_together_l1834_183401

theorem bells_toll_together (a b c d : ℕ) 
  (ha : a = 9) (hb : b = 10) (hc : c = 14) (hd : d = 18) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_bells_toll_together_l1834_183401


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_l1834_183423

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Theorem: In a geometric sequence where a_7 = 1/4 and a_3 * a_5 = 4(a_4 - 1), a_2 = 8 -/
theorem geometric_sequence_a2 (a : ℕ → ℚ) 
    (h_geom : GeometricSequence a) 
    (h_a7 : a 7 = 1/4) 
    (h_a3a5 : a 3 * a 5 = 4 * (a 4 - 1)) : 
  a 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_l1834_183423


namespace NUMINAMATH_CALUDE_escalator_step_count_l1834_183494

/-- Represents the number of steps a person counts while descending an escalator -/
def count_steps (escalator_length : ℕ) (walking_count : ℕ) (speed_multiplier : ℕ) : ℕ :=
  let escalator_speed := escalator_length - walking_count
  let speed_ratio := escalator_speed / walking_count
  let new_ratio := speed_ratio / speed_multiplier
  escalator_length / (new_ratio + 1)

/-- Theorem stating that given an escalator of 200 steps, where a person counts 50 steps while 
    walking down, the same person will count 80 steps when running twice as fast -/
theorem escalator_step_count :
  count_steps 200 50 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_escalator_step_count_l1834_183494


namespace NUMINAMATH_CALUDE_orange_ring_weight_l1834_183489

/-- Given the weights of three rings (purple, white, and orange) that sum to a total weight,
    prove that the weight of the orange ring is equal to the total weight minus the sum of
    the purple and white ring weights. -/
theorem orange_ring_weight
  (purple_weight white_weight total_weight : ℚ)
  (h1 : purple_weight = 0.3333333333333333)
  (h2 : white_weight = 0.4166666666666667)
  (h3 : total_weight = 0.8333333333333334)
  (h4 : ∃ orange_weight : ℚ, purple_weight + white_weight + orange_weight = total_weight) :
  ∃ orange_weight : ℚ, orange_weight = total_weight - (purple_weight + white_weight) :=
by
  sorry


end NUMINAMATH_CALUDE_orange_ring_weight_l1834_183489


namespace NUMINAMATH_CALUDE_triangle_problem_l1834_183454

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
    (h1 : Real.cos t.B * (Real.sqrt 3 * t.a - t.b * Real.sin t.C) - t.b * Real.sin t.B * Real.cos t.C = 0)
    (h2 : t.c = 2 * t.a)
    (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = Real.sqrt 3) : 
    t.B = π / 3 ∧ t.a + t.b + t.c = 3 * Real.sqrt 2 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1834_183454


namespace NUMINAMATH_CALUDE_sine_value_given_tangent_and_point_l1834_183448

theorem sine_value_given_tangent_and_point (α : Real) (m : Real) :
  (∃ (x y : Real), x = m ∧ y = 9 ∧ x^2 + y^2 ≠ 0 ∧ Real.tan α = y / x) →
  Real.tan α = 3 / 4 →
  Real.sin α = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_value_given_tangent_and_point_l1834_183448


namespace NUMINAMATH_CALUDE_no_real_solution_l1834_183403

theorem no_real_solution :
  ¬∃ (a b c d : ℝ), 
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l1834_183403


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l1834_183463

theorem complex_magnitude_equality (t : ℝ) :
  t > 0 → (Complex.abs (Complex.mk (-4) t) = 2 * Real.sqrt 13 ↔ t = 6) := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l1834_183463


namespace NUMINAMATH_CALUDE_triangle_xy_length_l1834_183460

-- Define the triangle
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  -- Right angle at X
  (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0 ∧
  -- 45° angle at Y
  (Z.1 - Y.1) * (X.1 - Y.1) + (Z.2 - Y.2) * (X.2 - Y.2) = 
    Real.sqrt ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2) * Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) / 2 ∧
  -- XZ = 12√2
  (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 288

-- Theorem statement
theorem triangle_xy_length (X Y Z : ℝ × ℝ) (h : Triangle X Y Z) :
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_triangle_xy_length_l1834_183460


namespace NUMINAMATH_CALUDE_tank_fill_time_proof_l1834_183488

/-- The time (in hours) it takes to fill the tank with the leak -/
def fill_time_with_leak : ℝ := 11

/-- The time (in hours) it takes for the tank to become empty due to the leak -/
def empty_time_due_to_leak : ℝ := 110

/-- The time (in hours) it takes to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 10

theorem tank_fill_time_proof :
  (1 / fill_time_without_leak) - (1 / empty_time_due_to_leak) = (1 / fill_time_with_leak) :=
sorry

end NUMINAMATH_CALUDE_tank_fill_time_proof_l1834_183488


namespace NUMINAMATH_CALUDE_maddie_had_15_books_l1834_183435

/-- The number of books Maddie had -/
def maddie_books : ℕ := sorry

/-- The number of books Luisa had -/
def luisa_books : ℕ := 18

/-- The number of books Amy had -/
def amy_books : ℕ := 6

/-- Theorem stating that Maddie had 15 books -/
theorem maddie_had_15_books : maddie_books = 15 := by
  have h1 : amy_books + luisa_books = maddie_books + 9 := sorry
  sorry

end NUMINAMATH_CALUDE_maddie_had_15_books_l1834_183435


namespace NUMINAMATH_CALUDE_total_frisbee_distance_l1834_183400

/-- The distance Bess can throw the Frisbee -/
def bess_throw_distance : ℕ := 20

/-- The number of times Bess throws the Frisbee -/
def bess_throw_count : ℕ := 4

/-- The distance Holly can throw the Frisbee -/
def holly_throw_distance : ℕ := 8

/-- The number of times Holly throws the Frisbee -/
def holly_throw_count : ℕ := 5

/-- Theorem stating the total distance traveled by both Frisbees -/
theorem total_frisbee_distance : 
  2 * bess_throw_distance * bess_throw_count + holly_throw_distance * holly_throw_count = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_frisbee_distance_l1834_183400


namespace NUMINAMATH_CALUDE_symmetry_point_l1834_183451

/-- Given two points A and B in a 2D plane, they are symmetric with respect to the origin
    if the sum of their coordinates is (0, 0) -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 + B.1 = 0 ∧ A.2 + B.2 = 0

theorem symmetry_point :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (2, -3)
  symmetric_wrt_origin A B → B = (2, -3) := by
sorry

end NUMINAMATH_CALUDE_symmetry_point_l1834_183451


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l1834_183495

/-- The equivalent single discount rate after applying successive discounts -/
def equivalent_discount (d1 d2 : ℝ) : ℝ :=
  1 - (1 - d1) * (1 - d2)

/-- Theorem stating that the equivalent single discount rate after applying
    successive discounts of 15% and 25% is 36.25% -/
theorem successive_discounts_equivalence :
  equivalent_discount 0.15 0.25 = 0.3625 := by
  sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l1834_183495


namespace NUMINAMATH_CALUDE_triangle_solution_l1834_183413

noncomputable def triangle_problem (a b c A B C : ℝ) : Prop :=
  (2 * b - c) * Real.cos A = a * Real.cos C ∧
  a = Real.sqrt 13 ∧
  (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3

theorem triangle_solution (a b c A B C : ℝ) 
  (h : triangle_problem a b c A B C) :
  A = π / 3 ∧ a + b + c = 7 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_solution_l1834_183413


namespace NUMINAMATH_CALUDE_crayons_remaining_l1834_183411

/-- Given a drawer with 7 crayons initially, prove that after removing 3 crayons, 4 crayons remain. -/
theorem crayons_remaining (initial : ℕ) (removed : ℕ) (remaining : ℕ) : 
  initial = 7 → removed = 3 → remaining = initial - removed → remaining = 4 := by sorry

end NUMINAMATH_CALUDE_crayons_remaining_l1834_183411


namespace NUMINAMATH_CALUDE_u_closed_under_multiplication_l1834_183461

def u : Set ℕ := {n : ℕ | ∃ m : ℕ, n = m * m ∧ m > 0}

theorem u_closed_under_multiplication :
  ∀ x y : ℕ, x ∈ u → y ∈ u → (x * y) ∈ u :=
by
  sorry

end NUMINAMATH_CALUDE_u_closed_under_multiplication_l1834_183461


namespace NUMINAMATH_CALUDE_container_capacity_l1834_183425

theorem container_capacity (C : ℝ) (h : 0.35 * C + 48 = 0.75 * C) : C = 120 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l1834_183425


namespace NUMINAMATH_CALUDE_max_harmonious_t_patterns_l1834_183422

/-- Represents a coloring of an 8x8 grid --/
def Coloring := Fin 8 → Fin 8 → Bool

/-- Represents a T-shaped pattern in the grid --/
structure TPattern where
  row : Fin 8
  col : Fin 8
  orientation : Fin 4

/-- The total number of T-shaped patterns in an 8x8 grid --/
def total_t_patterns : Nat := 168

/-- Checks if a T-pattern is harmonious under a given coloring --/
def is_harmonious (c : Coloring) (t : TPattern) : Bool :=
  sorry

/-- Counts the number of harmonious T-patterns for a given coloring --/
def count_harmonious (c : Coloring) : Nat :=
  sorry

/-- The maximum number of harmonious T-patterns possible --/
def max_harmonious : Nat := 132

theorem max_harmonious_t_patterns :
  ∃ (c : Coloring), count_harmonious c = max_harmonious ∧
  ∀ (c' : Coloring), count_harmonious c' ≤ max_harmonious :=
sorry

end NUMINAMATH_CALUDE_max_harmonious_t_patterns_l1834_183422


namespace NUMINAMATH_CALUDE_lowest_divisible_by_even_14_to_21_l1834_183477

theorem lowest_divisible_by_even_14_to_21 : ∃! n : ℕ+, 
  (∀ k : ℕ, 14 ≤ k ∧ k ≤ 21 ∧ Even k → (n : ℕ) % k = 0) ∧ 
  (∀ m : ℕ+, (∀ k : ℕ, 14 ≤ k ∧ k ≤ 21 ∧ Even k → (m : ℕ) % k = 0) → n ≤ m) ∧
  n = 5040 := by
sorry

end NUMINAMATH_CALUDE_lowest_divisible_by_even_14_to_21_l1834_183477


namespace NUMINAMATH_CALUDE_rhonda_marbles_l1834_183479

theorem rhonda_marbles (total : ℕ) (diff : ℕ) (rhonda : ℕ) : 
  total = 215 → diff = 55 → total = rhonda + (rhonda + diff) → rhonda = 80 := by
  sorry

end NUMINAMATH_CALUDE_rhonda_marbles_l1834_183479


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l1834_183491

-- Define the arithmetic sequence
def arithmeticSequence (a₁ aₙ d : ℕ) : List ℕ :=
  let n := (aₙ - a₁) / d + 1
  List.range n |>.map (λ i => a₁ + i * d)

-- Define the sum of a list of natural numbers
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

-- State the theorem
theorem arithmetic_sequence_sum_divisibility :
  let seq := arithmeticSequence 3 251 8
  let sum := sumList seq
  sum % 8 = 0 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l1834_183491


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l1834_183440

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop :=
  ∃ (t : ℝ), x = 2 + t ∧ y = t

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_through_focus x₁ y₁ ∧ line_through_focus x₂ y₂

-- Theorem statement
theorem parabola_intersection_length
  (x₁ y₁ x₂ y₂ : ℝ)
  (h_intersection : intersection_points x₁ y₁ x₂ y₂)
  (h_sum : x₁ + x₂ = 6) :
  ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2 : ℝ) = 10 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l1834_183440


namespace NUMINAMATH_CALUDE_red_jellybeans_count_l1834_183464

/-- Proves that the number of red jellybeans is 120 given the specified conditions -/
theorem red_jellybeans_count (total : ℕ) (blue : ℕ) (purple : ℕ) (orange : ℕ)
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_purple : purple = 26)
  (h_orange : orange = 40) :
  total - (blue + purple + orange) = 120 := by
  sorry

end NUMINAMATH_CALUDE_red_jellybeans_count_l1834_183464


namespace NUMINAMATH_CALUDE_grapes_purchased_l1834_183498

theorem grapes_purchased (grape_price : ℕ) (mango_quantity : ℕ) (mango_price : ℕ) (total_paid : ℕ) :
  grape_price = 70 →
  mango_quantity = 9 →
  mango_price = 55 →
  total_paid = 705 →
  ∃ grape_quantity : ℕ, grape_quantity * grape_price + mango_quantity * mango_price = total_paid ∧ grape_quantity = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_grapes_purchased_l1834_183498


namespace NUMINAMATH_CALUDE_inequality_proof_l1834_183415

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1834_183415


namespace NUMINAMATH_CALUDE_sequence_sum_implies_general_term_l1834_183408

/-- Given a sequence (aₙ) with sum Sₙ = (2/3)aₙ + 1/3, prove aₙ = (-2)^(n-1) -/
theorem sequence_sum_implies_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 1 → S n = (2/3) * a n + 1/3) :
  ∀ n : ℕ, n ≥ 1 → a n = (-2)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_implies_general_term_l1834_183408


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_11_l1834_183434

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isLucky (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

def isMultipleOf11 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 11 * k

theorem least_non_lucky_multiple_of_11 :
  (∀ m : ℕ, m < 11 → ¬(isMultipleOf11 m ∧ ¬isLucky m)) ∧
  (isMultipleOf11 11 ∧ ¬isLucky 11) :=
sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_11_l1834_183434


namespace NUMINAMATH_CALUDE_sams_age_l1834_183456

/-- Given that Sam and Drew have a combined age of 54 and Sam is half of Drew's age,
    prove that Sam is 18 years old. -/
theorem sams_age (total_age : ℕ) (drews_age : ℕ) (sams_age : ℕ) 
    (h1 : total_age = 54)
    (h2 : sams_age + drews_age = total_age)
    (h3 : sams_age = drews_age / 2) : 
  sams_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_sams_age_l1834_183456


namespace NUMINAMATH_CALUDE_consecutive_integers_with_prime_factors_l1834_183431

theorem consecutive_integers_with_prime_factors 
  (n s m : ℕ+) : 
  ∃ (x : ℕ), ∀ (j : ℕ), j ∈ Finset.range m → 
    (∃ (p : Finset ℕ), p.card = n ∧ 
      (∀ q ∈ p, Nat.Prime q ∧ 
        (∃ (k : ℕ), k ≥ s ∧ (q^k : ℕ) ∣ (x + j)))) :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_with_prime_factors_l1834_183431


namespace NUMINAMATH_CALUDE_units_digit_47_power_47_l1834_183462

theorem units_digit_47_power_47 : 47^47 ≡ 3 [ZMOD 10] := by sorry

end NUMINAMATH_CALUDE_units_digit_47_power_47_l1834_183462


namespace NUMINAMATH_CALUDE_train_combined_speed_l1834_183424

/-- The combined speed of two trains moving in opposite directions -/
theorem train_combined_speed 
  (train1_length : ℝ) 
  (train1_time : ℝ) 
  (train2_speed : ℝ) 
  (h1 : train1_length = 180) 
  (h2 : train1_time = 12) 
  (h3 : train2_speed = 30) : 
  train1_length / train1_time + train2_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_combined_speed_l1834_183424


namespace NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l1834_183483

/-- Given two solutions P and Q, where liquid X makes up 0.5% of P and 1.5% of Q,
    prove that mixing 200g of P with 800g of Q results in a solution containing 1.3% liquid X. -/
theorem liquid_x_percentage_in_mixed_solution :
  let p_weight : ℝ := 200
  let q_weight : ℝ := 800
  let p_percentage : ℝ := 0.5
  let q_percentage : ℝ := 1.5
  let x_in_p : ℝ := p_weight * (p_percentage / 100)
  let x_in_q : ℝ := q_weight * (q_percentage / 100)
  let total_x : ℝ := x_in_p + x_in_q
  let total_weight : ℝ := p_weight + q_weight
  let result_percentage : ℝ := (total_x / total_weight) * 100
  result_percentage = 1.3 := by sorry

end NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l1834_183483


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l1834_183442

theorem percentage_failed_hindi (failed_english : Real) (failed_both : Real) (passed_both : Real) :
  failed_english = 35 →
  failed_both = 40 →
  passed_both = 80 →
  ∃ (failed_hindi : Real), failed_hindi = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l1834_183442


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1834_183416

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 4*x + 1 = 0 ↔ (x + 2)^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1834_183416


namespace NUMINAMATH_CALUDE_treasure_count_conversion_l1834_183429

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The deep-sea creature's treasure count in base 7 -/
def treasureCountBase7 : Nat := 245

theorem treasure_count_conversion :
  base7ToBase10 treasureCountBase7 = 131 := by
  sorry

end NUMINAMATH_CALUDE_treasure_count_conversion_l1834_183429


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1834_183406

theorem quadratic_minimum (x : ℝ) : ∃ m : ℝ, m = 1337 ∧ ∀ x, 5*x^2 - 20*x + 1357 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1834_183406


namespace NUMINAMATH_CALUDE_sum_inequality_and_equality_condition_l1834_183465

theorem sum_inequality_and_equality_condition (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : 
  a + b + c ≥ 3 ∧ (a + b + c = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_and_equality_condition_l1834_183465


namespace NUMINAMATH_CALUDE_smallest_average_of_valid_pair_l1834_183443

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate for two numbers differing by 2 and having sum of digits divisible by 4 -/
def validPair (n m : ℕ) : Prop :=
  m = n + 2 ∧ (sumOfDigits n + sumOfDigits m) % 4 = 0

theorem smallest_average_of_valid_pair :
  ∃ (n m : ℕ), validPair n m ∧
  ∀ (k l : ℕ), validPair k l → (n + m : ℚ) / 2 ≤ (k + l : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_average_of_valid_pair_l1834_183443


namespace NUMINAMATH_CALUDE_zoo_feeding_theorem_l1834_183486

/-- Represents the number of bread and treats brought by each person -/
structure BreadAndTreats :=
  (bread : ℕ)
  (treats : ℕ)

/-- Calculates the total number of bread and treats -/
def totalItems (items : List BreadAndTreats) : ℕ :=
  (items.map (λ i => i.bread + i.treats)).sum

/-- Calculates the cost per pet -/
def costPerPet (totalBread totalTreats : ℕ) (x y : ℚ) (z : ℕ) : ℚ :=
  (totalBread * x + totalTreats * y) / z

theorem zoo_feeding_theorem 
  (jane_bread : ℕ) (jane_treats : ℕ)
  (wanda_bread : ℕ) (wanda_treats : ℕ)
  (carla_bread : ℕ) (carla_treats : ℕ)
  (peter_bread : ℕ) (peter_treats : ℕ)
  (x y : ℚ) (z : ℕ) :
  jane_bread = (75 * jane_treats) / 100 →
  wanda_treats = jane_treats / 2 →
  wanda_bread = 3 * wanda_treats →
  wanda_bread = 90 →
  carla_treats = (5 * carla_bread) / 2 →
  carla_bread = 40 →
  peter_bread = 2 * peter_treats →
  peter_bread + peter_treats = 140 →
  let items := [
    BreadAndTreats.mk jane_bread jane_treats,
    BreadAndTreats.mk wanda_bread wanda_treats,
    BreadAndTreats.mk carla_bread carla_treats,
    BreadAndTreats.mk peter_bread peter_treats
  ]
  totalItems items = 427 ∧
  costPerPet 235 192 x y z = (235 * x + 192 * y) / z :=
by sorry


end NUMINAMATH_CALUDE_zoo_feeding_theorem_l1834_183486


namespace NUMINAMATH_CALUDE_number_difference_l1834_183421

theorem number_difference (a b : ℕ) (h1 : a + b = 27630) (h2 : 5 * a + 5 = b) :
  b - a = 18421 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1834_183421


namespace NUMINAMATH_CALUDE_solution_difference_l1834_183452

theorem solution_difference (p q : ℝ) : 
  (p - 4) * (p + 4) = 28 * p - 84 →
  (q - 4) * (q + 4) = 28 * q - 84 →
  p ≠ q →
  p > q →
  p - q = 16 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1834_183452


namespace NUMINAMATH_CALUDE_sum_13_impossible_l1834_183455

-- Define the type for dice faces
def DieFace := Fin 6

-- Define the function to calculate the sum of two dice
def diceSum (d1 d2 : DieFace) : Nat := d1.val + d2.val + 2

-- Theorem statement
theorem sum_13_impossible :
  ¬ ∃ (d1 d2 : DieFace), diceSum d1 d2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_13_impossible_l1834_183455


namespace NUMINAMATH_CALUDE_impossible_last_digit_match_l1834_183430

theorem impossible_last_digit_match (n : ℕ) (h_n : n = 111) :
  ¬ ∃ (S : Finset ℕ),
    Finset.card S = n ∧
    (∀ x ∈ S, x ≤ 500) ∧
    (∀ x ∈ S, ∀ y ∈ S, x ≠ y → x ≠ y) ∧
    (∀ x ∈ S, x % 10 = (Finset.sum S id - x) % 10) :=
by sorry

end NUMINAMATH_CALUDE_impossible_last_digit_match_l1834_183430


namespace NUMINAMATH_CALUDE_simultaneous_ringing_l1834_183485

/-- The least common multiple of the bell ringing periods -/
def bell_lcm : ℕ := sorry

/-- The time difference in minutes between the first and next simultaneous ringing -/
def time_difference : ℕ := sorry

theorem simultaneous_ringing :
  bell_lcm = lcm 18 (lcm 24 (lcm 30 36)) ∧
  time_difference = bell_lcm ∧
  time_difference = 360 := by sorry

end NUMINAMATH_CALUDE_simultaneous_ringing_l1834_183485


namespace NUMINAMATH_CALUDE_password_count_l1834_183402

/-- The number of digits in the password -/
def password_length : ℕ := 4

/-- The number of available digits (0-9 excluding 7) -/
def available_digits : ℕ := 9

/-- The total number of possible passwords without restrictions -/
def total_passwords : ℕ := available_digits ^ password_length

/-- The number of ways to choose digits for a password with all different digits -/
def ways_to_choose_digits : ℕ := Nat.choose available_digits password_length

/-- The number of ways to arrange the chosen digits -/
def ways_to_arrange_digits : ℕ := Nat.factorial password_length

/-- The number of passwords with all different digits -/
def passwords_with_different_digits : ℕ := ways_to_choose_digits * ways_to_arrange_digits

/-- The number of passwords with at least two identical digits -/
def passwords_with_identical_digits : ℕ := total_passwords - passwords_with_different_digits

theorem password_count : passwords_with_identical_digits = 3537 := by
  sorry

end NUMINAMATH_CALUDE_password_count_l1834_183402


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l1834_183426

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_max_min_on_interval :
  let a := 0
  let b := Real.pi / 2
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 2 ∧ f x_min = -1 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l1834_183426


namespace NUMINAMATH_CALUDE_correct_calculation_l1834_183419

theorem correct_calculation : ∃ x : ℕ, (x + 30 = 86) ∧ (x * 30 = 1680) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1834_183419


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1834_183438

theorem constant_term_expansion (k : ℕ+) : k = 1 ↔ k^4 * (Nat.choose 6 4) < 120 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1834_183438


namespace NUMINAMATH_CALUDE_equation_solution_l1834_183459

theorem equation_solution (x y : ℝ) (h : x / (x - 1) = (y^2 + 2*y - 1) / (y^2 + 2*y - 2)) :
  x = y^2 + 2*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1834_183459


namespace NUMINAMATH_CALUDE_age_difference_l1834_183499

theorem age_difference (x y z : ℕ) (h : z = x - 18) : 
  (x + y) - (y + z) = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1834_183499


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1834_183458

/-- Given three circles with radii r₁, r₂, r₃, where r₁ is the largest,
    the radius of the circle inscribed in the quadrilateral formed by
    the tangents as described in the problem is
    (r₁ * r₂ * r₃) / (r₁ * r₃ + r₁ * r₂ - r₂ * r₃). -/
theorem inscribed_circle_radius
  (r₁ r₂ r₃ : ℝ)
  (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₃ > 0)
  (h₄ : r₁ > r₂) (h₅ : r₁ > r₃) :
  ∃ (r : ℝ), r = (r₁ * r₂ * r₃) / (r₁ * r₃ + r₁ * r₂ - r₂ * r₃) ∧
  r > 0 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1834_183458


namespace NUMINAMATH_CALUDE_inequality_proof_l1834_183446

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1834_183446


namespace NUMINAMATH_CALUDE_perp_planes_parallel_perp_plane_line_perp_l1834_183433

-- Define the types for lines and planes
variable (L : Type) [LinearOrderedField L]
variable (P : Type)

-- Define the relations
variable (parallel : L → L → Prop)
variable (perp : L → L → Prop)
variable (perp_plane : L → P → Prop)
variable (parallel_plane : P → P → Prop)
variable (contained : L → P → Prop)

-- Theorem 1
theorem perp_planes_parallel
  (m : L) (α β : P)
  (h1 : perp_plane m α)
  (h2 : perp_plane m β)
  : parallel_plane α β :=
sorry

-- Theorem 2
theorem perp_plane_line_perp
  (m n : L) (α : P)
  (h1 : perp_plane m α)
  (h2 : contained n α)
  : perp m n :=
sorry

end NUMINAMATH_CALUDE_perp_planes_parallel_perp_plane_line_perp_l1834_183433


namespace NUMINAMATH_CALUDE_max_knights_and_courtiers_l1834_183450

/-- Represents the number of people at the king's table -/
def kings_table : ℕ := 7

/-- Represents the minimum number of courtiers -/
def min_courtiers : ℕ := 12

/-- Represents the maximum number of courtiers -/
def max_courtiers : ℕ := 18

/-- Represents the minimum number of knights -/
def min_knights : ℕ := 10

/-- Represents the maximum number of knights -/
def max_knights : ℕ := 20

/-- Represents the rule that the lunch of a knight plus the lunch of a courtier equals the lunch of the king -/
def lunch_rule (courtiers knights : ℕ) : Prop :=
  (1 : ℚ) / courtiers + (1 : ℚ) / knights = (1 : ℚ) / kings_table

/-- The main theorem stating the maximum number of knights and courtiers -/
theorem max_knights_and_courtiers :
  ∃ (k c : ℕ), 
    min_courtiers ≤ c ∧ c ≤ max_courtiers ∧
    min_knights ≤ k ∧ k ≤ max_knights ∧
    lunch_rule c k ∧
    (∀ (k' c' : ℕ), 
      min_courtiers ≤ c' ∧ c' ≤ max_courtiers ∧
      min_knights ≤ k' ∧ k' ≤ max_knights ∧
      lunch_rule c' k' →
      k' ≤ k) ∧
    k = 14 ∧ c = 14 :=
  sorry

end NUMINAMATH_CALUDE_max_knights_and_courtiers_l1834_183450


namespace NUMINAMATH_CALUDE_water_poured_out_specific_cup_l1834_183466

/-- Represents a cylindrical cup -/
structure CylindricalCup where
  baseDiameter : ℝ
  height : ℝ

/-- Calculates the volume of water poured out when tilting a cylindrical cup -/
def waterPouredOut (cup : CylindricalCup) (initialAngle finalAngle : ℝ) : ℝ :=
  sorry

/-- Theorem stating the volume of water poured out for a specific cup and tilt angles -/
theorem water_poured_out_specific_cup :
  let cup : CylindricalCup := { baseDiameter := 8, height := 8 * Real.sqrt 3 }
  waterPouredOut cup (π/3) (π/6) = (128 * Real.sqrt 3 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_water_poured_out_specific_cup_l1834_183466


namespace NUMINAMATH_CALUDE_g_1001_value_l1834_183482

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + x = x * g y + g x

theorem g_1001_value
  (g : ℝ → ℝ)
  (h1 : FunctionalEquation g)
  (h2 : g 1 = -3) :
  g 1001 = -2001 := by
  sorry

end NUMINAMATH_CALUDE_g_1001_value_l1834_183482


namespace NUMINAMATH_CALUDE_carpet_area_calculation_l1834_183441

theorem carpet_area_calculation (rectangle_length rectangle_width triangle_base triangle_height : ℝ) 
  (h1 : rectangle_length = 12)
  (h2 : rectangle_width = 8)
  (h3 : triangle_base = 10)
  (h4 : triangle_height = 6) : 
  rectangle_length * rectangle_width + (triangle_base * triangle_height) / 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_calculation_l1834_183441


namespace NUMINAMATH_CALUDE_circle_center_range_l1834_183492

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Define the circle C
def circle_C (center_x center_y : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 3)

-- Define the origin O
def point_O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_center_range :
  ∀ center_x center_y : ℝ,
  line_l center_x center_y →
  (∃ x : ℝ, circle_C center_x center_y x 0 ∧ circle_C center_x center_y (-x) 0 ∧ x^2 = 3/4) →
  (∃ mx my : ℝ, circle_C center_x center_y mx my ∧
    (mx - point_A.1)^2 + (my - point_A.2)^2 = 4 * ((mx - center_x)^2 + (my - center_y)^2)) →
  0 ≤ center_x ∧ center_x ≤ 12/5 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_range_l1834_183492


namespace NUMINAMATH_CALUDE_work_days_calculation_l1834_183417

theorem work_days_calculation (total_days : ℕ) (work_pay : ℕ) (no_work_deduction : ℕ) (total_earnings : ℤ) :
  total_days = 30 ∧ 
  work_pay = 80 ∧ 
  no_work_deduction = 40 ∧ 
  total_earnings = 1600 →
  ∃ (days_not_worked : ℕ),
    days_not_worked = 20 ∧
    total_earnings = work_pay * (total_days - days_not_worked) - no_work_deduction * days_not_worked :=
by sorry


end NUMINAMATH_CALUDE_work_days_calculation_l1834_183417


namespace NUMINAMATH_CALUDE_no_valid_rectangle_l1834_183497

theorem no_valid_rectangle (a b x y : ℝ) : 
  a < b → 
  x < a → 
  y < a → 
  2 * (x + y) = (2/3) * (a + b) → 
  x * y = (1/3) * a * b → 
  False := by
sorry

end NUMINAMATH_CALUDE_no_valid_rectangle_l1834_183497


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l1834_183481

theorem simplify_and_ratio : 
  ∀ (k : ℝ), 
  (6 * k + 18) / 6 = k + 3 ∧ 
  ∃ (a b : ℤ), k + 3 = a * k + b ∧ (a : ℝ) / (b : ℝ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l1834_183481


namespace NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l1834_183405

/-- Given a rectangle and an ellipse with specific properties, prove that the rectangle's perimeter is 8√1003 -/
theorem rectangle_ellipse_perimeter :
  ∀ (x y a b : ℝ),
    -- Rectangle properties
    x > 0 ∧ y > 0 ∧
    x * y = 2006 ∧
    -- Ellipse properties
    a > 0 ∧ b > 0 ∧
    x + y = 2 * a ∧
    x^2 + y^2 = 4 * (a^2 - b^2) ∧
    π * a * b = 2006 * π →
    -- Conclusion: Perimeter of the rectangle
    2 * (x + y) = 8 * Real.sqrt 1003 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l1834_183405


namespace NUMINAMATH_CALUDE_ralphs_tv_time_l1834_183404

/-- The number of hours Ralph watches TV in one week -/
def total_tv_hours (weekday_hours weekday_days weekend_hours weekend_days : ℕ) : ℕ :=
  weekday_hours * weekday_days + weekend_hours * weekend_days

theorem ralphs_tv_time : total_tv_hours 4 5 6 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ralphs_tv_time_l1834_183404


namespace NUMINAMATH_CALUDE_value_of_R_l1834_183427

theorem value_of_R : ∀ P Q R : ℚ, 
  P = 4014 / 2 →
  Q = P / 4 →
  R = P - Q →
  R = 1505.25 := by
sorry

end NUMINAMATH_CALUDE_value_of_R_l1834_183427


namespace NUMINAMATH_CALUDE_squirrel_walnuts_l1834_183432

/-- The number of walnuts the boy squirrel effectively adds to the burrow -/
def boy_walnuts : ℕ := 5

/-- The number of walnuts the girl squirrel effectively adds to the burrow -/
def girl_walnuts : ℕ := 3

/-- The final number of walnuts in the burrow -/
def final_walnuts : ℕ := 20

/-- The initial number of walnuts in the burrow -/
def initial_walnuts : ℕ := 12

theorem squirrel_walnuts :
  initial_walnuts + boy_walnuts + girl_walnuts = final_walnuts :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_walnuts_l1834_183432


namespace NUMINAMATH_CALUDE_candy_bar_cost_l1834_183490

theorem candy_bar_cost (total_bars : ℕ) (dave_bars : ℕ) (john_paid : ℚ) : 
  total_bars = 20 → 
  dave_bars = 6 → 
  john_paid = 21 → 
  (john_paid / (total_bars - dave_bars : ℚ)) = 1.5 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l1834_183490


namespace NUMINAMATH_CALUDE_problem_statement_l1834_183471

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - a

-- Define the domain
def domain : Set ℝ := Set.Icc (-1) 1

theorem problem_statement (a : ℝ) :
  -- Part 1
  (f a 0 = f a 1) →
  (Set.Icc (-1) (1/2) = {x ∈ domain | |f a x - 1| < a * x + 3/4}) ∧
  -- Part 2
  (|a| ≤ 1) →
  (∀ x ∈ domain, |f a x| ≤ 5/4) :=
by sorry


end NUMINAMATH_CALUDE_problem_statement_l1834_183471


namespace NUMINAMATH_CALUDE_subtract_negative_l1834_183453

theorem subtract_negative : -3 - 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l1834_183453


namespace NUMINAMATH_CALUDE_fish_swimming_north_l1834_183474

theorem fish_swimming_north (west east north caught_east caught_west left : ℕ) :
  west = 1800 →
  east = 3200 →
  caught_east = (2 * east) / 5 →
  caught_west = (3 * west) / 4 →
  left = 2870 →
  west + east + north = caught_east + caught_west + left →
  north = 500 := by
sorry

end NUMINAMATH_CALUDE_fish_swimming_north_l1834_183474


namespace NUMINAMATH_CALUDE_subset_relation_l1834_183468

theorem subset_relation (M N : Set ℕ) : 
  M = {1, 2, 3, 4} → N = {2, 3, 4} → N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_subset_relation_l1834_183468


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_necessary_not_sufficient_condition_l1834_183447

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Theorem 1: Intersection of A and complement of B when m = 2
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x | -2 ≤ x ∧ x < -1 ∨ 3 < x ∧ x ≤ 4} := by sorry

-- Theorem 2: Necessary but not sufficient condition
theorem necessary_not_sufficient_condition :
  (∀ m > 0, (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m)) ↔ 0 < m ∧ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_necessary_not_sufficient_condition_l1834_183447


namespace NUMINAMATH_CALUDE_inequality_proof_l1834_183420

theorem inequality_proof (f : ℝ → ℝ) (a m n : ℝ) :
  (∀ x, f x = |x + a|) →
  (Set.Icc (-9 : ℝ) 1 = {x | f x ≤ 5}) →
  (m > 0) →
  (n > 0) →
  (1/m + 1/(2*n) = a) →
  m + 2*n ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1834_183420


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l1834_183457

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifteenth_term_of_sequence : 
  arithmetic_sequence 3 3 15 = 45 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l1834_183457


namespace NUMINAMATH_CALUDE_binomial_variance_theorem_l1834_183467

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ  -- The random variable

/-- Expected value of a random variable -/
def expectation (X : ℝ → ℝ) : ℝ := sorry

/-- Variance of a random variable -/
def variance (X : ℝ → ℝ) : ℝ := sorry

theorem binomial_variance_theorem (p q : ℝ) (X : BinomialDistribution 5 p) :
  expectation X.X = 2 → variance (fun ω => 2 * X.X ω + q) = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_theorem_l1834_183467


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1834_183478

/-- Given vectors a, b, and c in ℝ², where a = (1,2), b = (0,1), and c = (-2,k),
    if (a + 2b) is perpendicular to c, then k = 1/2. -/
theorem perpendicular_vectors_k_value :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![0, 1]
  let c : Fin 2 → ℝ := ![-2, k]
  (∀ i : Fin 2, (a i + 2 * b i) * c i = 0) →
  k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1834_183478


namespace NUMINAMATH_CALUDE_min_value_theorem_l1834_183436

theorem min_value_theorem (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ 
  ((n : ℝ) / 3 + 27 / (n : ℝ) = 6 ↔ n = 9) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1834_183436


namespace NUMINAMATH_CALUDE_slower_train_speed_l1834_183476

/-- Proves that the speed of the slower train is 37 km/hr given the conditions of the problem -/
theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 62.5)
  (h2 : faster_speed = 46)
  (h3 : passing_time = 45)
  : ∃ (slower_speed : ℝ), 
    slower_speed = 37 ∧ 
    2 * train_length = (faster_speed - slower_speed) * (5 / 18) * passing_time :=
by
  sorry

#check slower_train_speed

end NUMINAMATH_CALUDE_slower_train_speed_l1834_183476


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l1834_183480

theorem abs_inequality_equivalence (x : ℝ) : 
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by
sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l1834_183480


namespace NUMINAMATH_CALUDE_intersect_x_axis_and_derivative_negative_l1834_183444

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a

theorem intersect_x_axis_and_derivative_negative (a : ℝ) (x₁ x₂ : ℝ) :
  a > Real.exp 2 →
  x₁ < x₂ →
  f a x₁ = 0 →
  f a x₂ = 0 →
  let x₀ := Real.sqrt (x₁ * x₂)
  (deriv (f a)) x₀ < 0 :=
by sorry

end NUMINAMATH_CALUDE_intersect_x_axis_and_derivative_negative_l1834_183444


namespace NUMINAMATH_CALUDE_impossible_sum_110_l1834_183409

def coin_values : List ℕ := [1, 5, 10, 25, 50]

theorem impossible_sum_110 : 
  ¬ ∃ (coins : List ℕ), 
    coins.length = 6 ∧ 
    (∀ c ∈ coins, c ∈ coin_values) ∧ 
    coins.sum = 110 :=
sorry

end NUMINAMATH_CALUDE_impossible_sum_110_l1834_183409


namespace NUMINAMATH_CALUDE_min_races_for_top_3_l1834_183414

/-- Represents a horse in the race. -/
structure Horse :=
  (id : Nat)

/-- Represents a race with up to 6 horses. -/
structure Race :=
  (horses : Finset Horse)
  (condition : Nat)  -- Represents different race conditions

/-- A function to determine the ranking of horses in a race. -/
def raceResult (r : Race) : List Horse := sorry

/-- The total number of horses. -/
def totalHorses : Nat := 30

/-- The maximum number of horses that can race together. -/
def maxHorsesPerRace : Nat := 6

/-- A function to determine if we have found the top 3 horses. -/
def hasTop3 (races : List Race) : Bool := sorry

/-- Theorem stating the minimum number of races needed. -/
theorem min_races_for_top_3 :
  ∃ (races : List Race),
    races.length = 7 ∧
    hasTop3 races ∧
    ∀ (other_races : List Race),
      hasTop3 other_races → other_races.length ≥ 7 := by sorry

end NUMINAMATH_CALUDE_min_races_for_top_3_l1834_183414


namespace NUMINAMATH_CALUDE_range_of_a_l1834_183439

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (Real.exp x - a)^2 + x^2 - 2*a*x + a^2 ≤ 1/2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1834_183439
