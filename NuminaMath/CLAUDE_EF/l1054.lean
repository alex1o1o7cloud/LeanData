import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1054_105464

def sequence_a : ℕ → ℝ
  | 0 => 2  -- Add this case for n = 0
  | 1 => 2
  | (n + 2) => 2 * sequence_a (n + 1) + 3

theorem sequence_a_properties :
  (sequence_a 2 = 7 ∧ sequence_a 3 = 17 ∧ sequence_a 4 = 37) ∧
  (∀ n : ℕ, n ≥ 1 → (sequence_a (n + 1) + 3) / (sequence_a n + 3) = 2) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 5 * 2^(n - 1) - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1054_105464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1054_105463

theorem problem_solution :
  (∃ (S : Set (ℤ × ℤ)), S = {(2, 7), (2, -4), (-2, 7), (-2, -4)} ∧
    ∀ (m n : ℤ), (9 * m^2 + 3 * n = n^2 + 8) ↔ (m, n) ∈ S) ∧
  (∀ (a b : ℕ), a > 0 → b > 0 → 
    (a : ℝ)^(a : ℝ) + (a + b : ℝ)^((a + b : ℝ)) > (a : ℝ)^((a + b : ℝ)) + (a + b : ℝ)^(a : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1054_105463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meeting_time_l1054_105426

/-- Represents the time (in seconds) for two runners to meet or catch up on a circular track. -/
noncomputable def meetingTime (trackLength : ℝ) (speedA : ℝ) (speedB : ℝ) (sameDirection : Bool) : ℝ :=
  if sameDirection then
    trackLength / (speedB - speedA)
  else
    trackLength / (speedA + speedB)

theorem runners_meeting_time :
  let trackLength : ℝ := 440
  let speedA : ℝ := 5
  let speedB : ℝ := 6
  (meetingTime trackLength speedA speedB false = 40) ∧
  (meetingTime trackLength speedA speedB true = 440) := by
  sorry

-- Remove the #eval statements as they are not computable
-- #eval meetingTime 440 5 6 false
-- #eval meetingTime 440 5 6 true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meeting_time_l1054_105426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1054_105497

noncomputable def f (x : ℝ) : ℝ := |2/3 * x + 1|

theorem f_properties :
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≥ -|x| + a ∧ ∀ (b : ℝ), (∀ (x : ℝ), f x ≥ -|x| + b) → b ≤ a) ∧
  (∀ (x y : ℝ), |x + y + 1| ≤ 1/3 → |y - 1/3| ≤ 2/3 → f x ≤ 7/9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1054_105497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_t_wolf_cannot_catch_deer_l1054_105475

def wolf_deer_chase (t : ℚ) : Prop :=
  let deer_jump := (1 : ℚ)
  let wolf_jump := 0.78 * deer_jump
  let deer_jumps := (100 : ℚ)
  let wolf_jumps := deer_jumps * (1 + t / 100)
  deer_jump * deer_jumps ≥ wolf_jump * wolf_jumps

theorem largest_t_wolf_cannot_catch_deer :
  ∀ t : ℕ, t ≤ 28 → wolf_deer_chase (t : ℚ) ∧ ¬wolf_deer_chase ((t + 1 : ℕ) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_t_wolf_cannot_catch_deer_l1054_105475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_matches_conditions_l1054_105462

/-- The equation of the ellipse -/
def ellipse_eq (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The point through which the hyperbola passes -/
def point : ℝ × ℝ := (2, 1)

/-- The equation of the hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

/-- The foci of an ellipse with equation x^2/a^2 + y^2/b^2 = 1 are (±c, 0) where c^2 = a^2 - b^2 -/
noncomputable def ellipse_foci (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

/-- The foci of a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 are (±c, 0) where c^2 = a^2 + b^2 -/
noncomputable def hyperbola_foci (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)

theorem hyperbola_matches_conditions :
  /- The hyperbola shares the same foci with the ellipse -/
  (ellipse_foci 2 1 = hyperbola_foci (Real.sqrt 2) 1) ∧
  /- The hyperbola passes through the point (2,1) -/
  hyperbola_eq point.1 point.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_matches_conditions_l1054_105462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_card_cost_per_minute_l1054_105410

/-- The cost per minute for long distance calls on a prepaid phone card -/
noncomputable def cost_per_minute (initial_value : ℝ) (call_duration : ℝ) (remaining_credit : ℝ) : ℝ :=
  (initial_value - remaining_credit) / call_duration

/-- Theorem: The cost per minute for long distance calls is $0.16 -/
theorem phone_card_cost_per_minute :
  let initial_value : ℝ := 30
  let call_duration : ℝ := 22
  let remaining_credit : ℝ := 26.48
  cost_per_minute initial_value call_duration remaining_credit = 0.16 := by
  -- Unfold the definition of cost_per_minute
  unfold cost_per_minute
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_card_cost_per_minute_l1054_105410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_9ab_l1054_105401

/-- Represents a number with n repetitions of a digit in base 10 -/
def repeatedDigit (digit : ℕ) (n : ℕ) : ℕ :=
  (10^n - 1) / 9 * digit

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem sum_of_digits_9ab (a b : ℕ) : 
  a = repeatedDigit 9 1986 → 
  b = repeatedDigit 4 1986 → 
  (sumOfDigits (9 * a * b) = 15880) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_9ab_l1054_105401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1054_105436

/-- The parabola y^2 = -4x -/
def parabola (p : ℝ × ℝ) : Prop :=
  (p.2)^2 = -4 * p.1

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (-1, 0)

/-- Point A -/
def point_A : ℝ × ℝ := (-3, 2)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The sum of distances from a point to A and F -/
noncomputable def sum_distances (p : ℝ × ℝ) : ℝ :=
  distance p point_A + distance p focus

/-- The point P on the parabola -/
def point_P : ℝ × ℝ := (-1, 2)

theorem min_distance_point :
  parabola point_P ∧
  ∀ q : ℝ × ℝ, parabola q → sum_distances point_P ≤ sum_distances q := by
  sorry

#check min_distance_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1054_105436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_26_24_10_l1054_105450

/-- The area of a triangle given its side lengths using Heron's formula -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 26, 24, and 10 is 120 -/
theorem triangle_area_26_24_10 :
  triangle_area 26 24 10 = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_26_24_10_l1054_105450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l1054_105460

/-- Represents a player in the game -/
inductive Player : Type
| Abby : Player
| Bernardo : Player
| Carl : Player
| Debra : Player

/-- Represents a ball color -/
inductive BallColor : Type
| Green : BallColor
| Red : BallColor
| Blue : BallColor
| White : BallColor

/-- The game state, representing the number of coins each player has -/
structure GameState :=
  (abby : ℕ)
  (bernardo : ℕ)
  (carl : ℕ)
  (debra : ℕ)

/-- The initial game state -/
def initialState : GameState :=
  { abby := 5, bernardo := 4, carl := 3, debra := 4 }

/-- The number of rounds in the game -/
def numRounds : ℕ := 5

/-- The number of balls in the urn -/
def numBalls : ℕ := 5

/-- Function to update the game state based on a round's outcome -/
def updateState (state : GameState) (greenHolder red blue : Player) : GameState :=
  sorry

/-- Function to calculate the probability of a specific round outcome -/
def roundProbability : ℚ :=
  1 / 30

/-- The main theorem to prove -/
theorem coin_game_probability :
  ∃ (p : ℚ), p = 1 / 2430000 ∧
  p = (roundProbability ^ numRounds) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l1054_105460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_and_consecutive_sums_l1054_105499

def sum_consecutive_naturals (a k : ℕ) : ℕ :=
  (k * (2 * a + k - 1)) / 2

theorem power_of_two_and_consecutive_sums (M : ℕ) :
  (∃ n : ℕ, M = 2^n) ↔ ¬(∃ k a : ℕ, k ≥ 2 ∧ M = sum_consecutive_naturals a k) :=
by sorry

#check power_of_two_and_consecutive_sums

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_and_consecutive_sums_l1054_105499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutation_given_positive_test_l1054_105468

/-- The probability of having the mutation -/
noncomputable def prob_mutation : ℝ := 1 / 200

/-- The probability of not having the mutation -/
noncomputable def prob_no_mutation : ℝ := 1 - prob_mutation

/-- The sensitivity of the test (probability of testing positive given the mutation) -/
noncomputable def sensitivity : ℝ := 1

/-- The false positive rate of the test -/
noncomputable def false_positive_rate : ℝ := 0.05

/-- The probability of testing positive -/
noncomputable def prob_positive : ℝ := sensitivity * prob_mutation + false_positive_rate * prob_no_mutation

theorem mutation_given_positive_test : 
  (sensitivity * prob_mutation) / prob_positive = 20 / 219 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutation_given_positive_test_l1054_105468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_q3_l1054_105452

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Get the x-coordinates of a polygon's vertices -/
def xCoordinates (p : Polygon) : List ℝ :=
  p.vertices.map Prod.fst

/-- Sum of a list of real numbers -/
def sumList (l : List ℝ) : ℝ :=
  l.foldl (· + ·) 0

/-- Create a new polygon from the midpoints of sides of given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

theorem sum_x_coordinates_q3 (q1 : Polygon) 
  (h1 : q1.vertices.length = 50)
  (h2 : sumList (xCoordinates q1) = 1500) :
  let q2 := midpointPolygon q1
  let q3 := midpointPolygon q2
  sumList (xCoordinates q3) = 1500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_q3_l1054_105452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_specific_angles_l1054_105471

theorem tan_sum_specific_angles (a b : Real) 
  (ha : Real.tan a = 1/2) (hb : Real.tan b = 1/3) : 
  Real.tan (a + b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_specific_angles_l1054_105471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_for_integral_inequality_l1054_105419

open MeasureTheory Measure Real Set

theorem smallest_constant_for_integral_inequality 
  (n : ℕ) (hn : n ≥ 2) :
  ∃ (c : ℝ), c = n ∧ 
  (∀ (f : ℝ → ℝ), ContinuousOn f (Icc 0 1) → (∀ x ∈ Icc 0 1, f x ≥ 0) →
    ∫ x in (Icc 0 1), f (x^(1/n : ℝ)) ≤ c * ∫ x in (Icc 0 1), f x) ∧
  (∀ (c' : ℝ), c' < c →
    ∃ (f : ℝ → ℝ), ContinuousOn f (Icc 0 1) ∧ (∀ x ∈ Icc 0 1, f x ≥ 0) ∧
      ∫ x in (Icc 0 1), f (x^(1/n : ℝ)) > c' * ∫ x in (Icc 0 1), f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_for_integral_inequality_l1054_105419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1054_105404

/-- The cost of a single book -/
def book_cost : ℚ := 6

/-- The cost of a single magazine -/
def magazine_cost : ℚ := 7

/-- Condition: Cost of 2 books and 2 magazines is $26 -/
axiom first_group_cost : 2 * book_cost + 2 * magazine_cost = 26

/-- The number of books in the second group -/
def x : ℕ := 1

/-- The number of magazines in the second group -/
def y : ℕ := 3

/-- Condition: Cost of x books and y magazines is $27 -/
axiom second_group_cost : x * book_cost + y * magazine_cost = 27

/-- Theorem: There exists a unique solution where x = 1 and y = 3 -/
theorem unique_solution : x = 1 ∧ y = 3 ∧ ∀ (a b : ℕ), a * book_cost + b * magazine_cost = 27 → a = 1 ∧ b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1054_105404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_figure_l1054_105432

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := 2 * (t - Real.sin t)
noncomputable def y (t : ℝ) : ℝ := 2 * (1 - Real.cos t)

-- Define the line equation
def line_y : ℝ := 3

-- Define the bounds
noncomputable def lower_bound : ℝ := 2 * Real.pi / 3
noncomputable def upper_bound : ℝ := 5 * Real.pi / 3

-- Define the area calculation function
noncomputable def area : ℝ :=
  let S₀ := ∫ t in lower_bound..upper_bound, y t * (deriv x t)
  let S₁ := line_y * (x upper_bound - x lower_bound)
  S₀ - S₁

-- State the theorem
theorem area_of_enclosed_figure : area = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_figure_l1054_105432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_implies_a_geq_three_l1054_105488

/-- Given a function f(x) = e^x * (x^2 - a), where a is a real number,
    if f(x) is monotonically decreasing on (-3, 0),
    then a is greater than or equal to 3. -/
theorem monotonic_decreasing_implies_a_geq_three
  (a : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = Real.exp x * (x^2 - a))
  (h_monotonic : StrictMonoOn f (Set.Ioo (-3) 0)) :
  a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_implies_a_geq_three_l1054_105488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_winnings_is_1_75_l1054_105459

-- Define the probabilities for each outcome
noncomputable def prob_one : ℝ := 1/4
noncomputable def prob_two_or_three : ℝ := 1/2
noncomputable def prob_four_five_six : ℝ := 1/4

-- Define the winning/losing amounts for each outcome
def win_one : ℝ := 2
def win_two_or_three : ℝ := 4
def lose_four_five_six : ℝ := -3

-- Define the expected value calculation
noncomputable def expected_winnings : ℝ :=
  prob_one * win_one + 
  prob_two_or_three * win_two_or_three + 
  prob_four_five_six * lose_four_five_six

-- Theorem statement
theorem expected_winnings_is_1_75 : 
  expected_winnings = 1.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_winnings_is_1_75_l1054_105459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_sequence_first_four_terms_l1054_105400

def sequenceTerm (n : ℕ) : ℚ := (2 * n - 1) / (2 * n)

theorem sequence_general_term (n : ℕ) (h : n > 0) : 
  sequenceTerm n = (2 * n - 1) / (2 * n) :=
by
  -- The proof is trivial as it's the definition of sequenceTerm
  rfl

theorem sequence_first_four_terms :
  sequenceTerm 1 = 1/2 ∧ 
  sequenceTerm 2 = 3/4 ∧ 
  sequenceTerm 3 = 5/6 ∧ 
  sequenceTerm 4 = 7/8 :=
by
  -- Prove each part of the conjunction
  repeat (
    apply And.intro
    · -- Evaluate the sequenceTerm and simplify
      rw [sequenceTerm]
      norm_num
  )
  -- The last part doesn't need And.intro
  rw [sequenceTerm]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_sequence_first_four_terms_l1054_105400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l1054_105451

def billy_numbers : Finset ℕ := (Finset.range 300).filter (λ n => n > 0 ∧ n % 20 = 0)
def bobbi_numbers : Finset ℕ := (Finset.range 300).filter (λ n => n > 0 ∧ n % 30 = 0)
def common_numbers : Finset ℕ := billy_numbers ∩ bobbi_numbers

theorem same_number_probability :
  (Finset.card common_numbers : ℚ) /
  ((Finset.card billy_numbers * Finset.card bobbi_numbers) : ℚ) =
  1 / 30 := by
  sorry

#eval Finset.card billy_numbers
#eval Finset.card bobbi_numbers
#eval Finset.card common_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l1054_105451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_bead_bracelet_arrangements_l1054_105430

/-- The number of distinct arrangements of n beads on a bracelet, 
    considering rotations and reflections as the same arrangement -/
def bracelet_arrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- Theorem: The number of distinct arrangements of 8 beads on a bracelet, 
    considering rotations and reflections as the same arrangement, is 2520 -/
theorem eight_bead_bracelet_arrangements : 
  bracelet_arrangements 8 = 2520 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_bead_bracelet_arrangements_l1054_105430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_sum_l1054_105493

/-- Configuration of six equilateral triangles -/
structure TriangleConfiguration where
  /-- Side length of each equilateral triangle -/
  side_length : ℝ
  /-- Angle between two adjacent triangles -/
  angle : ℝ
  /-- There are six equilateral triangles -/
  triangle_count : Nat
  triangle_count_eq : triangle_count = 6
  /-- All triangles are equilateral -/
  equilateral : side_length > 0
  /-- The configuration forms a closed shape -/
  angle_sum : angle * (triangle_count : ℝ) = 2 * Real.pi

/-- Two specific triangles (shaded) -/
def shaded_triangles : Nat := 2

/-- Four specific triangles (painted) -/
def painted_triangles (config : TriangleConfiguration) : Nat :=
  config.triangle_count - shaded_triangles

/-- Area of an equilateral triangle -/
noncomputable def triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side ^ 2

/-- Theorem: The sum of areas of shaded triangles equals the sum of areas of painted triangles -/
theorem equal_area_sum (config : TriangleConfiguration) :
  (shaded_triangles : ℝ) * triangle_area config.side_length =
  (painted_triangles config : ℝ) * triangle_area config.side_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_sum_l1054_105493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_perimeter_a_l1054_105461

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Perimeter of quadrilateral PABN -/
noncomputable def perimeter (a : ℝ) : ℝ :=
  let A : Point := ⟨1, -3⟩
  let B : Point := ⟨4, -1⟩
  let P : Point := ⟨a, 0⟩
  let N : Point := ⟨a+1, 0⟩
  distance P A + distance A B + distance B N + distance N P

theorem minimal_perimeter_a (a : ℝ) :
  (∀ x : ℝ, perimeter a ≤ perimeter x) → a = 5/2 := by
  sorry

#check minimal_perimeter_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_perimeter_a_l1054_105461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_caravan_goats_solution_l1054_105424

/-- Represents the number of goats in a caravan -/
def num_goats (n : ℕ) : Prop := n = 35

/-- Given a caravan with 60 hens, 6 camels, 10 keepers, and some goats,
    if the total number of feet is 193 more than the number of heads,
    then there are 35 goats. -/
theorem caravan_goats :
  ∀ g : ℕ,
  (60 + g + 6 + 10) + 193 = (60 * 2 + g * 4 + 6 * 4 + 10 * 2) →
  num_goats g := by
  intro g h
  sorry

/-- There are 35 goats in the caravan -/
theorem solution : num_goats 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_caravan_goats_solution_l1054_105424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_count_is_34_l1054_105483

/-- The number of ordered quadruples (a, b, c, d) of positive odd integers satisfying a + b + c + 2d = 15 -/
def count_quadruples : ℕ := 
  (Finset.filter (fun (a, b, c, d) => 
    a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a + b + c + 2*d = 15)
    (Finset.product (Finset.range 16) (Finset.product (Finset.range 16) (Finset.product (Finset.range 16) (Finset.range 16))))).card

/-- The theorem statement -/
theorem quadruple_count_is_34 : count_quadruples = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_count_is_34_l1054_105483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_inequality_proof_l1054_105442

-- Define the function f
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x * (Real.log x + m)

-- State the theorems
theorem local_minimum_condition (m : ℝ) :
  (∃ x ∈ Set.Ioo 1 (Real.exp 1), IsLocalMin (f · m) x) ↔ m ∈ Set.Ioo (-2) (-1) := by
  sorry

theorem inequality_proof (x : ℝ) (m : ℝ) (h : x > 0) :
  f x m < x^3 * Real.exp x + (m - 1) * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_inequality_proof_l1054_105442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_content_paths_count_l1054_105457

/-- Represents the triangular arrangement of letters --/
structure LetterTriangle where
  letters : List (List Char)

/-- Defines a valid movement direction in the triangle --/
inductive Direction where
  | Horizontal
  | Vertical
  | Diagonal

/-- Represents a path in the letter triangle --/
structure ContentPath where
  moves : List Direction

/-- Checks if a path spells out "CONTENT" --/
def spellsContent (t : LetterTriangle) (p : ContentPath) : Bool :=
  sorry

/-- Counts the number of valid paths spelling "CONTENT" --/
def countContentPaths (t : LetterTriangle) : Nat :=
  sorry

/-- The specific triangle arrangement given in the problem --/
def problemTriangle : LetterTriangle :=
  { letters := 
    [['C'],
     ['C', 'O', 'C'],
     ['C', 'O', 'N', 'O', 'C'],
     ['C', 'O', 'N', 'T', 'N', 'O', 'C'],
     ['C', 'O', 'N', 'T', 'E', 'T', 'N', 'O', 'C'],
     ['C', 'O', 'N', 'T', 'E', 'N', 'E', 'T', 'N', 'O', 'C'],
     ['C', 'O', 'N', 'T', 'E', 'N', 'T', 'N', 'E', 'T', 'N', 'O', 'C']]
  }

theorem content_paths_count :
  countContentPaths problemTriangle = 729 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_content_paths_count_l1054_105457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_smallest_l1054_105482

/-- 
For any integer n ≥ 4, f(n) is the smallest integer such that 
any f(n)-element subset of {m, m+1, ..., m+n-1} (for any positive integer m) 
contains at least three pairwise coprime elements.
-/
def f (n : ℕ) : ℕ :=
  let k := n / 6
  let m := n % 6
  4 * k + m + 1 - (m / 4)

/-- 
Theorem: f(n) is the smallest integer satisfying the coprime property 
for any n ≥ 4
-/
theorem f_is_smallest (n : ℕ) (hn : n ≥ 4) :
  ∀ (m : ℕ+) (S : Finset ℕ),
    S ⊆ Finset.range n ∧ 
    S.card = f n →
    ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_smallest_l1054_105482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_4digit_difference_l1054_105414

/-- A function that checks if a list of four natural numbers forms a geometric sequence --/
def IsGeometricSequence (digits : List ℕ) : Prop :=
  digits.length = 4 ∧
  ∃ r : ℚ, r ≠ 0 ∧ 
    digits.get! 1 = digits.get! 0 * r ∧
    digits.get! 2 = digits.get! 1 * r ∧
    digits.get! 3 = digits.get! 2 * r

/-- A function that checks if a natural number is a 4-digit number --/
def IsFourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- A function that returns the digits of a natural number as a list --/
def Digits (n : ℕ) : List ℕ :=
  sorry

/-- The theorem stating the difference between the largest and smallest geometric 4-digit numbers --/
theorem geometric_4digit_difference : 
  ∃ (max min : ℕ),
    IsFourDigitNumber max ∧
    IsFourDigitNumber min ∧
    IsGeometricSequence (Digits max) ∧
    IsGeometricSequence (Digits min) ∧
    (∀ n : ℕ, IsFourDigitNumber n ∧ IsGeometricSequence (Digits n) → n ≤ max) ∧
    (∀ n : ℕ, IsFourDigitNumber n ∧ IsGeometricSequence (Digits n) → min ≤ n) ∧
    max - min = 7173 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_4digit_difference_l1054_105414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l1054_105449

/-- The distance between two parallel lines -/
noncomputable def distance_between_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the lines 2x + y + 1 = 0 and 4x + 2y - 3 = 0 is √5/2 -/
theorem distance_between_specific_lines :
  distance_between_lines 2 1 (-1) (3/2) = Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l1054_105449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_monotonic_increase_l1054_105411

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

theorem max_m_for_monotonic_increase :
  (∃ (m : ℝ), m > 0 ∧
    (∀ x y, x ∈ Set.Icc (-m) m → y ∈ Set.Icc (-m) m → x < y → f x < f y) ∧
    (∀ m' > m, ∃ x y, x ∈ Set.Icc (-m') m' ∧ y ∈ Set.Icc (-m') m' ∧ x < y ∧ f x ≥ f y)) →
  (∃ (m : ℝ), m = Real.pi / 12 ∧
    (∀ x y, x ∈ Set.Icc (-m) m → y ∈ Set.Icc (-m) m → x < y → f x < f y) ∧
    (∀ m' > m, ∃ x y, x ∈ Set.Icc (-m') m' ∧ y ∈ Set.Icc (-m') m' ∧ x < y ∧ f x ≥ f y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_monotonic_increase_l1054_105411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_g_2022_l1054_105455

noncomputable def g : ℝ → ℝ := sorry

axiom g_scale (x : ℝ) (h : x > 0) : g (4 * x) = 4 * g x

axiom g_def (x : ℝ) (h : 2 ≤ x ∧ x ≤ 6) : g x = 2 - |x - 4|

theorem smallest_x_equals_g_2022 :
  ∃ (x : ℝ), x > 0 ∧ g x = g 2022 ∧ ∀ (y : ℝ), y > 0 ∧ g y = g 2022 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_g_2022_l1054_105455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_negative_two_l1054_105406

-- Define the function representing the left side of the equation
def f (x : ℝ) : ℝ := x * abs x

-- Define the function representing the right side of the equation
def g (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem smallest_solution_is_negative_two :
  ∃ (x : ℝ), f x = g x ∧ (∀ (y : ℝ), f y = g y → x ≤ y) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_negative_two_l1054_105406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangents_impossible_l1054_105466

/-- Represents the number of common tangents between two circles -/
def CommonTangents : Type := Fin 5

/-- Two circles in the same plane with radii 3 and 5 -/
structure TwoCircles :=
  (radius1 : ℝ)
  (radius2 : ℝ)
  (h1 : radius1 = 3)
  (h2 : radius2 = 5)

/-- Predicate to check if a number of common tangents is possible for the given circles -/
def isPossibleNumberOfTangents (c : TwoCircles) (n : CommonTangents) : Prop :=
  ∃ (d : ℝ), 0 ≤ d ∧ 
    ((d > c.radius1 + c.radius2 ∧ n = ⟨4, by norm_num⟩) ∨
     (d < |c.radius2 - c.radius1| ∧ n = ⟨0, by norm_num⟩) ∨
     (|c.radius2 - c.radius1| < d ∧ d < c.radius1 + c.radius2 ∧ n = ⟨2, by norm_num⟩) ∨
     (d = |c.radius2 - c.radius1| ∧ n = ⟨1, by norm_num⟩))

/-- Theorem stating that 3 common tangents are impossible for the given circles -/
theorem three_tangents_impossible (c : TwoCircles) : 
  ¬ isPossibleNumberOfTangents c ⟨3, by norm_num⟩ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangents_impossible_l1054_105466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_l1054_105470

-- Define the circles and their properties
def larger_circle_radius : ℝ := 3

-- Define the arithmetic sequence property
def arithmetic_sequence (A₁ A₂ : ℝ) : Prop :=
  A₂ - A₁ = (A₁ + A₂) - A₂

-- Theorem statement
theorem smaller_circle_radius :
  ∃ (A₁ A₂ r : ℝ),
    arithmetic_sequence A₁ A₂ ∧
    (A₁ + A₂) = π * larger_circle_radius^2 ∧
    π * r^2 = A₁ ∧
    r^2 = 3 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_l1054_105470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_with_property_P_l1054_105435

-- Define the property (P)
def has_property_P (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + (deriv f) x) = f x

theorem function_with_property_P 
  (f : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hf_cont : Continuous (deriv f)) 
  (hf_prop : has_property_P f) :
  (∃ x, (deriv f) x = 0) ∧ 
  (∃ c, has_property_P (λ x ↦ -x^2 + c)) ∧
  (∃ a b, a ≠ b ∧ (deriv f) a = 0 ∧ (deriv f) b = 0 → ∀ x, (deriv f) x = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_with_property_P_l1054_105435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_cost_at_door_l1054_105420

/-- Proves that the cost of a ticket at the door is $22 -/
theorem ticket_cost_at_door (total_tickets : ℕ) (advanced_ticket_cost : ℚ) 
  (total_revenue : ℚ) (tickets_at_door : ℕ) 
  (h1 : total_tickets = 800)
  (h2 : advanced_ticket_cost = 29/2)  -- 14.50 as a rational number
  (h3 : total_revenue = 16640)
  (h4 : tickets_at_door = 672)
  : (total_revenue - (total_tickets - tickets_at_door : ℚ) * advanced_ticket_cost) / tickets_at_door = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_cost_at_door_l1054_105420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_three_l1054_105494

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_three : lg 8 + 3 * lg 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_three_l1054_105494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2021_div_7_l1054_105498

/-- The sequence where the nth positive integer appears n times -/
def seq (n : ℕ) : ℕ := Nat.sqrt (2 * n + 1)

/-- The 2021st term of the sequence -/
def term_2021 : ℕ := seq 2021

theorem sequence_2021_div_7 : term_2021 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2021_div_7_l1054_105498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_calculation_l1054_105438

def num_boys : ℕ := 8
def radius : ℝ := 50

noncomputable def chord_length (r : ℝ) : ℝ := 2 * r * Real.sin (Real.pi / 8)

noncomputable def total_distance : ℝ :=
  (num_boys : ℝ) * (num_boys - 3 : ℝ) * 2 * chord_length radius

theorem total_distance_calculation :
  total_distance = 8000 * Real.sqrt ((Real.sqrt 2 - 1) / (2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_calculation_l1054_105438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slices_per_pizza_l1054_105407

theorem pizza_slices_per_pizza (num_people : ℕ) (num_pizzas : ℕ) (slices_per_person : ℕ) :
  num_people > 0 →
  num_pizzas > 0 →
  slices_per_person > 0 →
  num_people * slices_per_person = num_pizzas * (num_people * slices_per_person / num_pizzas) :=
by
  sorry

-- The specific problem instance
example : 6 * 4 = 3 * (6 * 4 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slices_per_pizza_l1054_105407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1054_105441

noncomputable def f (x : ℝ) := 2 * Real.sin (x + Real.pi / 3)

theorem function_properties :
  ∃ (ω φ : ℝ),
    ω > 0 ∧
    0 < φ ∧ φ < Real.pi / 2 ∧
    (∀ x : ℝ, f x = 2 * Real.sin (ω * x + φ)) ∧
    (∀ x : ℝ, f (x + Real.pi) = f x) ∧
    (∀ x : ℝ, f x = 2 * Real.sin (x + Real.pi / 3)) ∧
    (∀ x : ℝ, (0 ≤ x ∧ x ≤ Real.pi / 6) → (∀ y : ℝ, 0 ≤ y ∧ y ≤ x → f y ≤ f x)) ∧
    (∀ x : ℝ, (7 * Real.pi / 6 ≤ x ∧ x ≤ 2 * Real.pi) → (∀ y : ℝ, 7 * Real.pi / 6 ≤ y ∧ y ≤ x → f y ≤ f x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1054_105441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_max_min_sum_l1054_105453

theorem product_of_max_min_sum (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_eq : (4 : ℝ)^(Real.sqrt (5*x + 9*y + 4*z)) - 68 * (2 : ℝ)^(Real.sqrt (5*x + 9*y + 4*z)) + 256 = 0) :
  ∃ (min_sum max_sum : ℝ), 
    (∀ a b c, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
      (4 : ℝ)^(Real.sqrt (5*a + 9*b + 4*c)) - 68 * (2 : ℝ)^(Real.sqrt (5*a + 9*b + 4*c)) + 256 = 0 →
      min_sum ≤ a + b + c ∧ a + b + c ≤ max_sum) ∧
    min_sum * max_sum = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_max_min_sum_l1054_105453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1054_105447

/-- A geometric sequence with b₉ = 1 -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), (∀ n, b (n + 1) = b n * r) ∧ b 9 = 1

/-- Product of terms from i to j in a sequence -/
def prod_seq (b : ℕ → ℝ) (i j : ℕ) : ℝ :=
  (Finset.range (j - i + 1)).prod (λ k ↦ b (i + k))

/-- Main theorem for the geometric sequence property -/
theorem geometric_sequence_property (b : ℕ → ℝ) (n : ℕ) :
  geometric_sequence b → 0 < n → n < 17 →
  prod_seq b 1 n = prod_seq b 1 (17 - n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1054_105447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_fixed_points_l1054_105408

/-- The function g(x) = (x+8)/x -/
noncomputable def g (x : ℝ) : ℝ := (x + 8) / x

/-- The sequence of functions g_n -/
noncomputable def g_n : ℕ → (ℝ → ℝ)
  | 0 => g
  | (n+1) => g ∘ g_n n

/-- The set of fixed points of g_n for some n -/
def fixed_points : Set ℝ := {x : ℝ | ∃ n : ℕ, g_n n x = x}

/-- The main theorem: there are exactly two fixed points -/
theorem two_fixed_points : ∃ (S : Finset ℝ), S.card = 2 ∧ ↑S = fixed_points := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_fixed_points_l1054_105408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentomino_l1054_105480

/-- A polyomino is a plane geometric figure formed by joining one or more equal squares edge to edge. -/
structure Polyomino where
  -- We'll leave the internal structure undefined for now
  mk :: -- Empty constructor

/-- The number of distinct polyominoes for a given number of unit squares. -/
def distinctPolyominoes : ℕ → ℕ := sorry

/-- Shapes that can be transformed into each other by translation, rotation, or reflection are considered the same. -/
def equivalent_shapes (p q : Polyomino) : Bool := sorry

/-- There is 1 distinct domino (n=2). -/
axiom domino : distinctPolyominoes 2 = 1

/-- There are 2 distinct triominoes (n=3). -/
axiom triomino : distinctPolyominoes 3 = 2

/-- There are 5 distinct tetrominoes (n=4). -/
axiom tetromino : distinctPolyominoes 4 = 5

/-- The number of distinct pentominoes (n=5) is 12. -/
theorem pentomino : distinctPolyominoes 5 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentomino_l1054_105480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_ten_daas_l1054_105417

-- Define the set S
variable (S : Type)

-- Define predicates for quib and daa
variable (quib daa : S → Prop)

-- Define the relation of a daa belonging to a quib
variable (belongs_to : S → S → Prop)

-- P'1: Every quib is a collection of daas
axiom P1 : ∀ q : S, quib q → ∃ d : S, daa d ∧ belongs_to d q

-- P'2: Any three distinct quibs have one and only one daa in common
axiom P2 : ∀ q1 q2 q3 : S, quib q1 ∧ quib q2 ∧ quib q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 →
  ∃! d : S, daa d ∧ belongs_to d q1 ∧ belongs_to d q2 ∧ belongs_to d q3

-- P'3: Every daa belongs to exactly three quibs
axiom P3 : ∀ d : S, daa d →
  ∃! q1 q2 q3 : S, quib q1 ∧ quib q2 ∧ quib q3 ∧
  q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧
  belongs_to d q1 ∧ belongs_to d q2 ∧ belongs_to d q3

-- P'4: There are exactly six quibs
axiom P4 : ∃! q1 q2 q3 q4 q5 q6 : S,
  quib q1 ∧ quib q2 ∧ quib q3 ∧ quib q4 ∧ quib q5 ∧ quib q6 ∧
  q1 ≠ q2 ∧ q1 ≠ q3 ∧ q1 ≠ q4 ∧ q1 ≠ q5 ∧ q1 ≠ q6 ∧
  q2 ≠ q3 ∧ q2 ≠ q4 ∧ q2 ≠ q5 ∧ q2 ≠ q6 ∧
  q3 ≠ q4 ∧ q3 ≠ q5 ∧ q3 ≠ q6 ∧
  q4 ≠ q5 ∧ q4 ≠ q6 ∧
  q5 ≠ q6

-- P'5: Every set of four quibs share exactly two daas in common
axiom P5 : ∀ q1 q2 q3 q4 : S, quib q1 ∧ quib q2 ∧ quib q3 ∧ quib q4 ∧
  q1 ≠ q2 ∧ q1 ≠ q3 ∧ q1 ≠ q4 ∧ q2 ≠ q3 ∧ q2 ≠ q4 ∧ q3 ≠ q4 →
  ∃! d1 d2 : S, d1 ≠ d2 ∧ daa d1 ∧ daa d2 ∧
  belongs_to d1 q1 ∧ belongs_to d1 q2 ∧ belongs_to d1 q3 ∧ belongs_to d1 q4 ∧
  belongs_to d2 q1 ∧ belongs_to d2 q2 ∧ belongs_to d2 q3 ∧ belongs_to d2 q4

-- Theorem: There are exactly ten daas
theorem exactly_ten_daas : ∃! d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 : S,
  daa d1 ∧ daa d2 ∧ daa d3 ∧ daa d4 ∧ daa d5 ∧
  daa d6 ∧ daa d7 ∧ daa d8 ∧ daa d9 ∧ daa d10 ∧
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧ d1 ≠ d9 ∧ d1 ≠ d10 ∧
  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧ d2 ≠ d9 ∧ d2 ≠ d10 ∧
  d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧ d3 ≠ d9 ∧ d3 ≠ d10 ∧
  d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧ d4 ≠ d9 ∧ d4 ≠ d10 ∧
  d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧ d5 ≠ d9 ∧ d5 ≠ d10 ∧
  d6 ≠ d7 ∧ d6 ≠ d8 ∧ d6 ≠ d9 ∧ d6 ≠ d10 ∧
  d7 ≠ d8 ∧ d7 ≠ d9 ∧ d7 ≠ d10 ∧
  d8 ≠ d9 ∧ d8 ≠ d10 ∧
  d9 ≠ d10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_ten_daas_l1054_105417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_implies_m_eq_neg_one_l1054_105425

/-- A function f : ℝ → ℝ is linear if there exist constants k and b such that f x = k * x + b for all x, with k ≠ 0 -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k b : ℝ), k ≠ 0 ∧ ∀ x, f x = k * x + b

/-- The given function y = (m-1)x^(m^2) + 1 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * (x ^ (m^2)) + 1

theorem linear_function_implies_m_eq_neg_one :
  ∀ m : ℝ, IsLinearFunction (f m) → m = -1 := by
  sorry

#check linear_function_implies_m_eq_neg_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_implies_m_eq_neg_one_l1054_105425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_zero_l1054_105428

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp x + k * x

theorem extremum_at_zero (k : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), |x| < ε → f k x ≥ f k 0) ∨
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), |x| < ε → f k x ≤ f k 0) → 
  k = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_zero_l1054_105428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l1054_105490

def unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  unit_vector a →
  unit_vector b →
  (a.1 * b.1 + a.2 * b.2 = Real.sqrt 2 / 2) →
  Real.sqrt ((a.1 - Real.sqrt 2 * b.1)^2 + (a.2 - Real.sqrt 2 * b.2)^2) = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l1054_105490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_euler_totient_l1054_105478

theorem divisibility_of_euler_totient (a n : ℕ) (h1 : a ≥ 2) : n ∣ Nat.totient (a^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_euler_totient_l1054_105478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1054_105415

/-- A polynomial satisfying the functional equation xP(x-1) ≡ (x-2)P(x) for all real x -/
def FunctionalEquationPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x * (P (x - 1)) = (x - 2) * (P x)

/-- The theorem stating that any polynomial satisfying the functional equation
    must be of the form a(x^2 - x) where a is a constant -/
theorem functional_equation_solution :
  ∀ P : ℝ → ℝ, FunctionalEquationPolynomial P →
  ∃ a : ℝ, ∀ x : ℝ, P x = a * (x^2 - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1054_105415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncovered_area_theorem_l1054_105422

/-- The area of the region inside a square with side length a that is not covered by two quarter circles of radius a constructed around two adjacent vertices of the square -/
noncomputable def uncoveredArea (a : ℝ) : ℝ := a^2 * (12 - 3 * Real.sqrt 3 - 2 * Real.pi) / 12

/-- Theorem stating that the uncovered area is correct -/
theorem uncovered_area_theorem (a : ℝ) (h : a > 0) :
  let square_area := a^2
  let quarter_circle_area := π * a^2 / 4
  let equilateral_triangle_area := a^2 * Real.sqrt 3 / 4
  let twelfth_circle_area := π * a^2 / 12
  square_area - 2 * quarter_circle_area + (2 * twelfth_circle_area) - equilateral_triangle_area = uncoveredArea a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncovered_area_theorem_l1054_105422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_categorize_numbers_l1054_105477

noncomputable def number_set : List ℝ := [5/6, Real.pi, -3, -|-3/4|, 4^2, 0, 0.6]

def is_negative (x : ℝ) : Prop := x < 0
def is_non_negative_integer (x : ℝ) : Prop := x ≥ 0 ∧ ∃ n : ℤ, x = n
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem categorize_numbers :
  (∀ x ∈ number_set, is_negative x ↔ x = -3 ∨ x = -|-3/4|) ∧
  (∀ x ∈ number_set, is_non_negative_integer x ↔ x = 4^2 ∨ x = 0) ∧
  (∀ x ∈ number_set, is_rational x ↔ x = 5/6 ∨ x = -3 ∨ x = -|-3/4| ∨ x = 4^2 ∨ x = 0 ∨ x = 0.6) :=
by sorry

#check categorize_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_categorize_numbers_l1054_105477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1054_105485

theorem triangle_side_length 
  (A B C : EuclideanSpace ℝ (Fin 2)) 
  (perimeter : ℝ) 
  (area : ℝ) 
  (angle_A : ℝ) :
  perimeter = 20 →
  area = 10 * Real.sqrt 3 →
  angle_A = π / 3 →
  ‖B - C‖ = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1054_105485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_theorem_l1054_105454

def work_completion_time (a b : ℕ) (t : ℚ) : Prop :=
  let work_done_by_a := (3 : ℚ) / a
  let remaining_work := 1 - work_done_by_a
  let combined_rate := (1 : ℚ) / a + (1 : ℚ) / b
  t * combined_rate = remaining_work

theorem work_completion_theorem (a b : ℕ) (h1 : a = 12) (h2 : b = 15) :
  ∃ t : ℚ, work_completion_time a b t ∧ t = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_theorem_l1054_105454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_male_red_ant_percentage_is_correct_l1054_105439

/-- Represents the percentage of red ants in the ant colony -/
noncomputable def red_ant_percentage : ℝ := 85

/-- Represents the percentage of female red ants among red ants -/
noncomputable def female_red_ant_percentage : ℝ := 45

/-- Calculates the percentage of male red ants in the total ant population -/
noncomputable def male_red_ant_percentage : ℝ := 
  red_ant_percentage - (red_ant_percentage * female_red_ant_percentage / 100)

theorem male_red_ant_percentage_is_correct :
  male_red_ant_percentage = 46.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_male_red_ant_percentage_is_correct_l1054_105439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l1054_105491

noncomputable def p : ℝ := (1/2) * Real.sqrt (2 - Real.sqrt 3)
noncomputable def q : ℝ := (1/2) * Real.sqrt (2 + Real.sqrt 3)

noncomputable def α : ℝ := 2 * Real.arccos p
noncomputable def β : ℝ := 2 * Real.arccos q

theorem area_between_curves : 
  p < q ∧ 
  0 < α ∧ α < Real.pi ∧ 
  0 < β ∧ β < Real.pi ∧ 
  Real.cos (α/2) = p ∧ 
  Real.cos (β/2) = q ∧ 
  p^2 + (1/(4*p))^2 = 1 ∧
  q^2 + (1/(4*q))^2 = 1 →
  2 * ∫ x in p..q, (Real.sqrt (1 - x^2) - 1/(4*x)) = 
    (1/2) * Real.log (2 - Real.sqrt 3) + Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l1054_105491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_pentagon_ABCDE_l1054_105444

/-- Given points in the coordinate plane -/
def A : ℚ × ℚ := (0, 2)
def B : ℚ × ℚ := (1, 7)
def C : ℚ × ℚ := (10, 7)
def D : ℚ × ℚ := (7, 1)

/-- E is the intersection point of lines AC and BD -/
def E : ℚ × ℚ := (4, 4)

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Theorem: The area of pentagon ABCDE is 36 -/
theorem area_of_pentagon_ABCDE : 
  triangleArea A B C + triangleArea B D C - triangleArea B E C = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_pentagon_ABCDE_l1054_105444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1054_105416

/-- The number of days it takes for person a to complete a work alone -/
noncomputable def days_a : ℝ := 14

/-- The number of days it takes for person b to complete the work alone -/
noncomputable def days_b : ℝ := 10

/-- The number of days it takes for both a and b to complete the work together -/
noncomputable def days_together : ℝ := 35 / 6

theorem work_completion_time :
  (1 / days_a) + (1 / days_b) = (1 / days_together) :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1054_105416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_2023_is_X_l1054_105479

def mySequence : ℕ → Char
  | n => let m := n % 6
         if m = 0 then 'X'
         else if m = 1 then 'Y'
         else if m = 2 then 'Z'
         else if m = 3 then 'Z'
         else if m = 4 then 'Y'
         else 'X'

theorem letter_2023_is_X : mySequence 2022 = 'X' := by
  rfl

#eval mySequence 2022

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_2023_is_X_l1054_105479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l1054_105492

def customSequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2010 ∧ 
  a 2 = 2011 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 3 * n + 1

theorem sequence_1000th_term (a : ℕ → ℤ) (h : customSequence a) : 
  a 1000 = 3009 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l1054_105492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_pi_fourth_eq_neg_three_implies_cos_squared_plus_two_sin_double_l1054_105467

theorem tan_alpha_pi_fourth_eq_neg_three_implies_cos_squared_plus_two_sin_double (α : ℝ) :
  Real.tan (α + π / 4) = -3 → Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_pi_fourth_eq_neg_three_implies_cos_squared_plus_two_sin_double_l1054_105467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_expenditure_is_correct_l1054_105405

/-- Calculates Joe's expenditure at the market in Euros -/
noncomputable def market_expenditure : ℝ :=
  let orange_count : ℕ := 3
  let juice_count : ℕ := 7
  let honey_count : ℕ := 3
  let plant_count : ℕ := 4
  let orange_price : ℝ := 4.50
  let juice_price : ℝ := 0.50
  let honey_price : ℝ := 5.00
  let plant_price : ℝ := 18.00 / 2
  let fruit_juice_discount : ℝ := 0.10
  let honey_discount : ℝ := 0.05
  let honey_special : ℕ := 2
  let sales_tax : ℝ := 0.08
  let exchange_rate : ℝ := 0.85

  let orange_cost := orange_count * orange_price
  let juice_cost := juice_count * juice_price
  let honey_cost := honey_special * honey_price
  let plant_cost := plant_count * plant_price

  let fruit_juice_total := orange_cost + juice_cost
  let fruit_juice_discounted := fruit_juice_total * (1 - fruit_juice_discount)
  let honey_discounted := honey_cost * (1 - honey_discount)

  let subtotal := fruit_juice_discounted + honey_discounted + plant_cost
  let total_with_tax := subtotal * (1 + sales_tax)
  let total_in_euros := total_with_tax * exchange_rate

  total_in_euros

/-- Theorem stating that Joe's expenditure at the market is approximately €55.81 -/
theorem market_expenditure_is_correct : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |market_expenditure - 55.81| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_expenditure_is_correct_l1054_105405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_theorem_l1054_105465

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def start : point := (-3, 6)
def middle : point := (0, 2)
def finish : point := (6, -3)

theorem total_distance_theorem :
  distance start middle + distance middle finish = 5 + Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_theorem_l1054_105465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_value_l1054_105429

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) / (x^2 + 1/2)

theorem f_zero_value (a : ℝ) :
  (∀ x : ℝ, f a x ≤ 1/2) →  -- Maximum value condition
  (∀ x : ℝ, f a x ≥ -1) →  -- Minimum value condition
  (∃ x : ℝ, f a x = 1/2) →  -- Maximum value is attained
  (∃ x : ℝ, f a x = -1) →  -- Minimum value is attained
  f a 0 = -1/2 := by
  intro h_max h_min h_max_attained h_min_attained
  sorry

#check f_zero_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_value_l1054_105429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_on_circle_l1054_105446

/-- The distance between two points in a 2D Cartesian coordinate system -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-3, 4)

/-- The radius of the circle -/
def circle_radius : ℝ := 5

/-- The origin point -/
def origin : ℝ × ℝ := (0, 0)

theorem origin_on_circle :
  distance circle_center.1 circle_center.2 origin.1 origin.2 = circle_radius := by
  -- Unfold definitions
  unfold distance circle_center circle_radius origin
  -- Simplify the left-hand side
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_on_circle_l1054_105446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_a_champion_probability_l1054_105431

/-- The probability of a team winning a single game -/
noncomputable def win_prob : ℝ := 1 / 2

/-- The number of games Team A needs to win to become champions -/
def team_a_games_needed : ℕ := 1

/-- The number of games Team B needs to win to become champions -/
def team_b_games_needed : ℕ := 2

/-- The probability of Team A becoming champions -/
noncomputable def team_a_champion_prob : ℝ := 3 / 4

/-- Theorem stating that the probability of Team A becoming champions
    is equal to the probability of winning the first game plus
    the probability of winning the second game after losing the first -/
theorem team_a_champion_probability :
  team_a_champion_prob = win_prob + win_prob * win_prob :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_a_champion_probability_l1054_105431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_product_with_odd_sum_l1054_105448

def digits : List Nat := [2, 3, 4, 6, 7, 8, 9]

def is_valid_pair (a b : Nat) : Prop :=
  a ≥ 1000 ∧ a < 10000 ∧ b ≥ 100 ∧ b < 1000 ∧
  (List.sum (a.digits 10) + List.sum (b.digits 10) = List.sum digits) ∧
  (∀ d, d ∈ a.digits 10 ∨ d ∈ b.digits 10 ↔ d ∈ digits)

def sum_of_digits (n : Nat) : Nat :=
  List.sum (n.digits 10)

theorem greatest_product_with_odd_sum :
  ∀ a b : Nat, is_valid_pair a b →
  sum_of_digits b % 2 = 1 →
  a * b ≤ 9423 * 863 ∧ is_valid_pair 9423 863 ∧ sum_of_digits 863 % 2 = 1 :=
by
  sorry

#eval 9423 * 863

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_product_with_odd_sum_l1054_105448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_theorem_l1054_105433

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x + 2) + (x - 1) ^ 0

noncomputable def g (x : ℝ) : ℝ := Real.log (2 - x)

def M : Set ℝ := {x : ℝ | x > -2 ∧ x ≠ 1}

def N : Set ℝ := Set.univ

theorem domain_intersection_theorem : M ∩ N = {x : ℝ | x > -2 ∧ x ≠ 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_theorem_l1054_105433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gretel_raise_percentage_l1054_105469

noncomputable def hansel_initial_salary : ℝ := 30000
noncomputable def hansel_raise_percentage : ℝ := 10
noncomputable def gretel_initial_salary : ℝ := 30000
noncomputable def salary_difference_after_raise : ℝ := 1500

noncomputable def hansel_new_salary : ℝ := hansel_initial_salary * (1 + hansel_raise_percentage / 100)
noncomputable def gretel_new_salary : ℝ := hansel_new_salary + salary_difference_after_raise

theorem gretel_raise_percentage :
  (gretel_new_salary - gretel_initial_salary) / gretel_initial_salary * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gretel_raise_percentage_l1054_105469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_conversion_l1054_105403

/-- Converts a point in spherical coordinates to standard spherical coordinate representation -/
noncomputable def to_standard_spherical (ρ : ℝ) (θ : ℝ) (φ : ℝ) : ℝ × ℝ × ℝ :=
  let ρ' := abs ρ
  let φ' := φ % (2 * Real.pi)
  let φ'' := if φ' > Real.pi then 2 * Real.pi - φ' else φ'
  let θ' := θ % (2 * Real.pi)
  (ρ', θ', φ'')

/-- Theorem stating that the given point in spherical coordinates is equivalent 
    to the standard spherical coordinate representation -/
theorem spherical_coordinate_conversion :
  to_standard_spherical 4 (11 * Real.pi / 6) (9 * Real.pi / 5) = (4, 5 * Real.pi / 6, Real.pi / 5) :=
by
  sorry

#check spherical_coordinate_conversion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coordinate_conversion_l1054_105403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1054_105487

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

def right_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x ↦ f (x - shift)

def down_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x ↦ f x - shift

noncomputable def transformed_function (x : ℝ) : ℝ := Real.sin (2 * x - 2 * Real.pi / 3) - 2

theorem function_transformation :
  ∀ x, down_shift (right_shift original_function (Real.pi / 2)) 2 x = transformed_function x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1054_105487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1054_105472

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.log (2 * x + 1)

-- Define the set representing the domain
def domain : Set ℝ := {x | x > -1/2 ∧ x ≠ 0}

-- Theorem stating that the domain of f is correct
theorem f_domain : {x : ℝ | ∃ y, f x = y} = domain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1054_105472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_is_symmetry_of_curve_l1054_105474

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x + y = 0

-- Define what it means for a point to be on the curve
def point_on_curve (x y : ℝ) : Prop := curve x y

-- Define the reflection of a point (x, y) across the line x + y = 0
def reflect (x y : ℝ) : ℝ × ℝ := (y, x)

-- Theorem: The line x + y = 0 is a line of symmetry for the curve x^2 + y^2 + 4x - 4y = 0
theorem line_is_symmetry_of_curve : 
  ∀ x y : ℝ, point_on_curve x y → 
  (let (x', y') := reflect x y
   point_on_curve x' y' ∧ line_of_symmetry ((x + x')/2) ((y + y')/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_is_symmetry_of_curve_l1054_105474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_difference_l1054_105476

theorem cos_angle_difference (α : Real) (h1 : α ∈ Set.Ioo 0 (π/2)) (h2 : Real.tan α = 2) :
  Real.cos (α - π/4) = (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_difference_l1054_105476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_through_center_l1054_105421

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

-- Define the line l1
def line_l1 (x y : ℝ) : Prop := 2*x - 3*y + 6 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -2)

-- Define the slope of l1
noncomputable def slope_l1 : ℝ := 2/3

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := 2*x - 3*y - 8 = 0

-- Theorem statement
theorem line_parallel_through_center :
  (∀ x y : ℝ, line_l1 x y → (y - (-2)) = slope_l1 * (x - 1)) ∧
  (∀ x y : ℝ, line_l x y → (x, y) = circle_center ∨ 
    (y - circle_center.2) = slope_l1 * (x - circle_center.1)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_through_center_l1054_105421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_alpha_l1054_105481

theorem sin_pi_minus_alpha (α : ℝ) 
  (h1 : Real.cos (2 * Real.pi - α) = Real.sqrt 5 / 3)
  (h2 : α > -Real.pi / 2 ∧ α < 0) : 
  Real.sin (Real.pi - α) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_alpha_l1054_105481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1054_105434

/-- The number of days it takes for person A to complete the work alone -/
noncomputable def days_A : ℝ := 30

/-- The fraction of work completed by A and B together in 3 days -/
noncomputable def work_fraction : ℝ := 1/6

/-- The number of days A and B work together -/
noncomputable def days_together : ℝ := 3

/-- The number of days it takes for person B to complete the work alone -/
noncomputable def days_B : ℝ := 45

theorem work_completion_time :
  (3 * (1/days_A + 1/days_B) = work_fraction) →
  days_B = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1054_105434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_call_cost_l1054_105402

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- Define the cost function
noncomputable def f (m : ℝ) : ℝ :=
  1.06 * (0.50 * (ceiling m : ℝ) + 1)

-- State the theorem
theorem phone_call_cost :
  ∀ m : ℝ, m > 0 → f m = 1.06 * (0.50 * (ceiling m : ℝ) + 1) →
  f 5.5 = 4.24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_call_cost_l1054_105402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_15_value_l1054_105456

def c : ℕ → ℕ
  | 0 => 3  -- We define c(0) as 3 to match c(1) in the original problem
  | 1 => 1  -- This matches c(2) in the original problem
  | n + 2 => 2 * (c (n + 1) + c n)

theorem c_15_value : c 14 = 1187008 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_15_value_l1054_105456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_b_n_b_n_plus_one_is_one_l1054_105437

def b (n : ℕ) : ℚ := (2 * 10^n - 1) / 9

theorem gcd_b_n_b_n_plus_one_is_one (n : ℕ) :
  Nat.gcd (Int.natAbs (Int.floor (b n))) (Int.natAbs (Int.floor (b (n + 1)))) = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_b_n_b_n_plus_one_is_one_l1054_105437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_theorem_l1054_105427

open Set

def A : Set ℝ := {x | 2 * x - x^2 > 0}
def B : Set ℝ := {x | x > 1}

theorem intersection_complement_theorem :
  ((compl B) ∩ A) = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_theorem_l1054_105427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l1054_105486

noncomputable def v1 : ℝ × ℝ := (6, 2)
noncomputable def v2 : ℝ × ℝ := (-2, 4)
noncomputable def p : ℝ × ℝ := (14/17, 56/17)

theorem projection_equality (v : ℝ × ℝ) : 
  (v1 • v / (v • v)) • v = p ∧ (v2 • v / (v • v)) • v = p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l1054_105486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_speed_increase_l1054_105489

-- Define the initial speed in km/h
noncomputable def initial_speed : ℝ := 60

-- Define the distance to cover in km
noncomputable def distance : ℝ := 1

-- Define the target time reduction in minutes
noncomputable def time_reduction : ℝ := 1

-- Define the current time to cover the distance
noncomputable def current_time : ℝ := distance / (initial_speed / 60)

-- Define the target time after reduction
noncomputable def target_time : ℝ := current_time - time_reduction

-- Theorem stating the impossibility of the speed increase
theorem impossible_speed_increase :
  ¬∃ (new_speed : ℝ), new_speed > initial_speed ∧ distance / (new_speed / 60) = target_time :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_speed_increase_l1054_105489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_proof_l1054_105409

/-- Calculates the final amount after compound interest --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- Checks if two real numbers are approximately equal within a small epsilon --/
def approximately_equal (x y : ℝ) (epsilon : ℝ) : Prop :=
  abs (x - y) < epsilon

theorem investment_rate_proof :
  let principal : ℝ := 10000
  let frequency : ℝ := 2
  let time : ℝ := 2
  let final_amount : ℝ := 10815.834432633617
  let rate : ℝ := 0.0398
  approximately_equal (compound_interest principal rate frequency time) final_amount 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_proof_l1054_105409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_integers_l1054_105445

theorem product_of_integers (x y z : ℝ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →
  x * y = 24 * (4^(1/4 : ℝ)) →
  x * z = 42 * (4^(1/4 : ℝ)) →
  y * z = 21 * (4^(1/4 : ℝ)) →
  x < y ∧ y < z →
  x * y * z = 291.2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_integers_l1054_105445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_a_2000_l1054_105413

def sequence_condition (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m ∣ n → m < n → a m ∣ a n ∧ a m < a n

theorem least_a_2000 (a : ℕ → ℕ) (h : sequence_condition a) : 
  ∀ k : ℕ → ℕ, sequence_condition k → a 2000 ≤ k 2000 := by
  sorry

#eval 2^7  -- Expected output: 128

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_a_2000_l1054_105413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_units_digit_same_units_and_tens_digits_l1054_105418

theorem same_units_digit (n : ℕ) :
  n % 10 = (n^2) % 10 ↔ n % 10 ∈ ({0, 1, 5, 6} : Set ℕ) :=
sorry

theorem same_units_and_tens_digits (n : ℕ) :
  n % 100 = (n^2) % 100 ↔ n ∈ ({0, 1, 25, 76} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_units_digit_same_units_and_tens_digits_l1054_105418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_fourth_quadrant_l1054_105484

theorem terminal_side_in_fourth_quadrant (α : Real) : 
  (∃ P : ℝ × ℝ, P.1 = Real.tan α ∧ P.2 = Real.cos α ∧ P.1 < 0 ∧ P.2 > 0) →
  (Real.tan α < 0 ∧ Real.cos α > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_fourth_quadrant_l1054_105484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_sequence_l1054_105458

open Real BigOperators

noncomputable def b (n : ℕ) : ℝ := 2^n * n * π

noncomputable def T (n : ℕ) : ℝ := ∑ k in Finset.range n, b (k + 1)

theorem sum_of_b_sequence (n : ℕ) : T n = ((n - 1) * 2^(n + 1) + 2) * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_sequence_l1054_105458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_proof_l1054_105440

open Real

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * sin (ω * x - π / 6)

-- State the theorem
theorem sine_value_proof (ω α : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x : ℝ, f ω (x + π / ω) = f ω x)
  (h_alpha_range : π / 6 < α ∧ α < 2 * π / 3)
  (h_f_value : f ω (α / 2) = Real.sqrt 3 / 4) :
  sin (α + π / 2) = (3 * Real.sqrt 5 - 1) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_proof_l1054_105440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_alternate_sides_product_equality_l1054_105473

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a plane -/
def Point := ℝ × ℝ

/-- Converts a Circle to a set of points -/
def Circle.toSet (c : Circle) : Set Point := 
  {p | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2}

/-- Predicate to check if six points form a non-self-intersecting hexagon -/
def IsNonSelfIntersectingHexagon (A B C D E F : Point) : Prop := sorry

/-- Given three circles and six points on their intersections, 
    proves that the product of alternate sides of the hexagon formed by these points are equal -/
theorem hexagon_alternate_sides_product_equality 
  (k₁ k₂ k₃ : Circle) 
  (A B C D E F : Point) 
  (h₁ : {A, D} = {p | p ∈ k₁.toSet ∧ p ∈ k₂.toSet})
  (h₂ : {B, E} = {p | p ∈ k₁.toSet ∧ p ∈ k₃.toSet})
  (h₃ : {C, F} = {p | p ∈ k₂.toSet ∧ p ∈ k₃.toSet})
  (h₄ : IsNonSelfIntersectingHexagon A B C D E F) :
  let d (p q : Point) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d A B) * (d C D) * (d E F) = (d B C) * (d D E) * (d F A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_alternate_sides_product_equality_l1054_105473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_silver_weight_problem_l1054_105443

/-- Represents the weight of a single piece of gold in taels. -/
def x : ℝ := sorry

/-- Represents the weight of a single piece of silver in taels. -/
def y : ℝ := sorry

/-- The system of equations correctly represents the problem of gold and silver weights. -/
theorem gold_silver_weight_problem :
  (9 * x = 11 * y) ∧ ((10 * y + x) - (8 * x + y) = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_silver_weight_problem_l1054_105443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_squared_l1054_105412

-- Define the function f(x) = sin²(x)
noncomputable def f (x : ℝ) := Real.sin x ^ 2

-- State the theorem
theorem derivative_of_sin_squared (x : ℝ) : 
  deriv f x = Real.sin (2 * x) := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_squared_l1054_105412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1054_105423

-- Define an ellipse with major axis 2a, minor axis 2b, and focus 2c
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : c > 0
  h4 : c < a
  h5 : b^2 = a^2 - c^2

-- Define the condition that the lengths form a geometric sequence
def geometric_sequence (e : Ellipse) : Prop :=
  4 * e.b^2 = 2 * e.a * 2 * e.c

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

-- State the theorem
theorem ellipse_eccentricity (e : Ellipse) (h : geometric_sequence e) :
  eccentricity e = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1054_105423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1054_105496

-- Define the greatest integer function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the equation
def equation (x : ℝ) : Prop :=
  (floor (3 * x - 4 * (5/6)) : ℝ) - 2 * x - 1 = 0

-- State the theorem
theorem equation_solution :
  ∃ (x : ℝ), equation x ∧ x = 13/2 := by
  -- The proof goes here
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1054_105496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pregnancy_fraction_l1054_105495

/-- Proves that the fraction of the population that got pregnant is 1/8, given the initial population, immigration, emigration, twin birth rate, and final population. -/
theorem pregnancy_fraction (initial_pop : ℕ) (immigration : ℕ) (emigration : ℕ) (final_pop : ℕ) 
  (h1 : initial_pop = 300000)
  (h2 : immigration = 50000)
  (h3 : emigration = 30000)
  (h4 : final_pop = 370000)
  (h5 : (1 : ℚ) / 4 = twin_birth_rate) :
  let pop_after_migration := initial_pop + immigration - emigration
  let fraction := (final_pop - pop_after_migration : ℚ) / pop_after_migration / (5 / 4)
  fraction = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pregnancy_fraction_l1054_105495
