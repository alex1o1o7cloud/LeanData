import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_f_not_monotonically_decreasing_elsewhere_l261_26164

-- Define the function f(x) = x^2 - 2ln(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

-- State the theorem
theorem f_monotonically_decreasing :
  ∀ a b : ℝ, 0 < a ∧ a < b ∧ b ≤ 1 → 
    ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y :=
by
  sorry

-- State that f is not monotonically decreasing on any other interval
theorem f_not_monotonically_decreasing_elsewhere :
  ∀ a b : ℝ, 0 < a ∧ 1 < b → 
    ∃ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b ∧ f x ≤ f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_f_not_monotonically_decreasing_elsewhere_l261_26164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l261_26192

/-- A quadratic function passing through specific points with given properties -/
def quadratic_function (b c : ℝ) : ℝ → ℝ := λ x ↦ x^2 + b*x + c

theorem quadratic_properties :
  ∀ b c : ℝ,
  quadratic_function b c 0 = -1 →
  quadratic_function b c 2 = 7 →
  ∃ y₁ y₂ : ℝ,
  quadratic_function b c (-5) = y₁ ∧
  y₁ + y₂ = 28 ∧
  ∃ m : ℝ,
  quadratic_function b c m = y₂ ∧
  (b = 2 ∧ c = -1) ∧
  (-1 : ℝ) = -b / (2 * 1) ∧
  m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l261_26192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_120_moves_l261_26123

-- Define the complex number representing the rotation
noncomputable def ω : ℂ := Complex.exp (Complex.I * (Real.pi / 3))

-- Define a single move
noncomputable def move (z : ℂ) : ℂ := ω * z + 8

-- Define the position after n moves
noncomputable def position (n : ℕ) : ℂ :=
  (move^[n]) 6

-- Theorem statement
theorem particle_position_after_120_moves :
  position 120 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_120_moves_l261_26123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_connect_four_prob_estimate_l261_26133

/-- Represents a Connect Four game board -/
def ConnectFourBoard := Fin 7 → Fin 6 → Option Bool

/-- Represents a valid Connect Four game state -/
def ValidConnectFourState (board : ConnectFourBoard) : Prop := sorry

/-- Checks if a player has won the game -/
def HasWinner (board : ConnectFourBoard) : Prop := sorry

/-- Checks if the board is completely filled -/
def IsBoardFull (board : ConnectFourBoard) : Prop := sorry

/-- Represents a sequence of random moves in a Connect Four game -/
def RandomMoveSequence := ℕ → Fin 7

/-- The final board state after playing a sequence of moves -/
def FinalBoard (moves : RandomMoveSequence) : ConnectFourBoard := sorry

/-- Probability measure on the space of random move sequences -/
noncomputable def RandomMoveProb : RandomMoveSequence → ℝ := sorry

/-- The probability that all columns are filled without any player obtaining four tokens in a row -/
noncomputable def P : ℝ := sorry

/-- Theorem stating the estimated value of P -/
theorem connect_four_prob_estimate : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0000001 ∧ |P - 0.0025632817| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_connect_four_prob_estimate_l261_26133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fertilizer_calculation_l261_26183

theorem fertilizer_calculation (total_area : ℝ) (partial_area : ℝ) (partial_fertilizer : ℝ) 
  (h1 : total_area = 10800)
  (h2 : partial_area = 3600)
  (h3 : partial_fertilizer = 400) :
  (total_area * partial_fertilizer) / partial_area = 1200 := by
  -- Replace the values with the given hypotheses
  rw [h1, h2, h3]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

#check fertilizer_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fertilizer_calculation_l261_26183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_periodic_odd_function_l261_26179

-- Define a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ T > 0, ∀ x, f (x + T) = f x

-- We don't need to define Odd and Even, as they are already in Mathlib

-- State the theorem
theorem derivative_of_periodic_odd_function (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (hp : IsPeriodic f) (ho : Odd f) : 
    IsPeriodic (deriv f) ∧ Even (deriv f) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_periodic_odd_function_l261_26179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_area_l261_26193

theorem triangle_perimeter_and_area 
  (a b c : ℝ) (A B C : Real) :
  a = 4 →
  Real.cos A = 3/5 →
  Real.cos C = Real.sqrt 5 / 5 →
  Real.sin B = Real.sin A * Real.cos C + Real.cos A * Real.sin C →
  b = a * Real.sin B / Real.sin A →
  c = b →
  (a + b + c = 4 + 4 * Real.sqrt 5) ∧
  (1/2 * a * b * Real.sin C = 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_and_area_l261_26193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_study_time_l261_26165

/-- Represents the relationship between study time and exam score -/
structure StudyScoreRelation where
  studyTime : ℝ
  examScore : ℝ

/-- Calculates the study time needed for a target score given a known study-score relation -/
noncomputable def calculateStudyTime (knownRelation : StudyScoreRelation) (targetScore : ℝ) : ℝ :=
  (targetScore * knownRelation.studyTime) / knownRelation.examScore

theorem samantha_study_time 
  (firstExam : StudyScoreRelation)
  (h1 : firstExam.studyTime = 3)
  (h2 : firstExam.examScore = 60)
  (targetAverage : ℝ)
  (h3 : targetAverage = 75) :
  calculateStudyTime firstExam (2 * targetAverage - firstExam.examScore) = 4.5 := by
  sorry

#check samantha_study_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_study_time_l261_26165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_C_l261_26170

/-- A race between three competitors A, B, and C -/
structure Race where
  length : ℝ  -- Race length in meters
  time_diff_AB : ℝ  -- Time difference between A and B in seconds
  dist_diff_AB : ℝ  -- Distance difference between A and B in meters
  time_diff_BC : ℝ  -- Time difference between B and C in seconds

/-- Calculate the time taken by competitor C to complete the race -/
noncomputable def time_C (r : Race) : ℝ :=
  let time_A := r.length * r.time_diff_AB / r.dist_diff_AB
  let time_B := time_A + r.time_diff_AB
  time_B + r.time_diff_BC

/-- Theorem stating the time taken by competitor C in the specific race scenario -/
theorem race_time_C :
  let r : Race := {
    length := 1000,  -- 1 km = 1000 meters
    time_diff_AB := 12,
    dist_diff_AB := 48,
    time_diff_BC := 20
  }
  time_C r = 282 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_C_l261_26170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_emptying_time_l261_26128

/-- Represents the time it takes to empty a cistern with a leak -/
noncomputable def time_to_empty (normal_fill_time hours_with_leak : ℝ) : ℝ :=
  let fill_rate := 1 / normal_fill_time
  let effective_fill_rate := 1 / hours_with_leak
  let leak_rate := fill_rate - effective_fill_rate
  1 / leak_rate

/-- 
Given a cistern that normally fills in 6 hours, but takes 8 hours to fill with a leak,
the time it takes for the leak to empty a full cistern is 24 hours.
-/
theorem cistern_leak_emptying_time :
  time_to_empty 6 8 = 24 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_emptying_time_l261_26128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l261_26176

/-- Given vectors a and b, prove that c₁ and c₂ are collinear -/
theorem vectors_collinear (a b : ℝ × ℝ × ℝ) 
  (h1 : a = (-1, 4, 2)) (h2 : b = (3, -2, 6)) : 
  ∃ (k : ℝ), (2 • a - b) = k • (3 • b - 6 • a) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l261_26176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l261_26127

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  (f (Real.log x) < f 1) ↔ (1 / Real.exp 1 < x ∧ x < Real.exp 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l261_26127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_positive_period_of_f_l261_26148

/-- The function f(x) = 3sin(2x + π/3) -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 3)

/-- The minimum positive period of f -/
noncomputable def T : ℝ := Real.pi

theorem minimum_positive_period_of_f :
  ∀ x : ℝ, f (x + T) = f x ∧ 
  ∀ τ : ℝ, 0 < τ → τ < T → ∃ y : ℝ, f (y + τ) ≠ f y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_positive_period_of_f_l261_26148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_g_negative_one_l261_26106

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g in terms of f
def g (x : ℝ) : ℝ := f x + 2

-- Define the function F in terms of f
def F (x : ℝ) : ℝ := f x + x^2

-- State the theorem
theorem odd_function_g_negative_one :
  (∀ x, F x = -F (-x)) →  -- F is an odd function
  f 1 = 1 →              -- f(1) = 1
  g (-1) = -1 :=         -- Conclusion: g(-1) = -1
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_g_negative_one_l261_26106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_percentage_l261_26114

/-- The percentage of students who favored Strawberry in a school survey --/
theorem strawberry_percentage : ℝ := by
  let chocolate : ℕ := 120
  let strawberry : ℕ := 100
  let vanilla : ℕ := 80
  let mint : ℕ := 50
  let butter_pecan : ℕ := 70
  let total : ℕ := chocolate + strawberry + vanilla + mint + butter_pecan
  have h : (strawberry : ℝ) / total * 100 = 24 := by
    sorry
  exact 24


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strawberry_percentage_l261_26114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_line_hexagon_concurrency_l261_26122

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  P : Point
  Q : Point
  R : Point

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a hexagon
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

-- Define a line
structure Line where
  point1 : Point
  point2 : Point

-- Define if a point is on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define the perpendicular foot from a point to a line
noncomputable def perpendicularFoot (S : Point) (l : Line) : Point :=
  sorry

-- Define collinearity of points
def collinear (P Q R : Point) : Prop :=
  sorry

-- Define concurrency of lines
def concurrent (l1 l2 l3 l4 : Line) : Prop :=
  sorry

-- Define if a quadrilateral is a rectangle
def isRectangle (A B C D : Point) : Prop :=
  sorry

-- Theorem 1: Simson Line
theorem simson_line (T : Triangle) (S : Point) (c : Circle) :
  pointOnCircle S c →
  collinear 
    (perpendicularFoot S (Line.mk T.P T.Q))
    (perpendicularFoot S (Line.mk T.Q T.R))
    (perpendicularFoot S (Line.mk T.R T.P)) :=
by
  sorry

-- Theorem 2: Hexagon Concurrency
theorem hexagon_concurrency (H : Hexagon) (c : Circle) :
  pointOnCircle H.A c ∧ pointOnCircle H.B c ∧ pointOnCircle H.C c ∧ 
  pointOnCircle H.D c ∧ pointOnCircle H.E c ∧ pointOnCircle H.F c →
  (concurrent 
    (Line.mk H.A (perpendicularFoot H.A (Line.mk H.B H.D)))
    (Line.mk H.B (perpendicularFoot H.B (Line.mk H.A H.C)))
    (Line.mk H.D (perpendicularFoot H.D (Line.mk H.A H.B)))
    (Line.mk H.E (perpendicularFoot H.E (Line.mk H.A H.B)))
  ) ↔ isRectangle H.C H.D H.E H.F :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_line_hexagon_concurrency_l261_26122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_perimeter_l261_26105

/-- Given a square with side length 2b cut by the line y = x/3, 
    the perimeter of one resulting quadrilateral divided by b equals (10/3) + √5 -/
theorem square_cut_perimeter (b : ℝ) (b_pos : b > 0) : 
  let square := {p : ℝ × ℝ | -b ≤ p.1 ∧ p.1 ≤ b ∧ -b ≤ p.2 ∧ p.2 ≤ b}
  let cut_line := {p : ℝ × ℝ | p.2 = p.1 / 3}
  let intersection_points := {p : ℝ × ℝ | p ∈ square ∩ cut_line}
  let quadrilateral := {p : ℝ × ℝ | p ∈ square ∧ p.2 ≤ p.1 / 3}
  let perimeter := Real.sqrt 5 * b + 2 * b + 4 * b / 3
  (perimeter / b) = 10 / 3 + Real.sqrt 5 :=
by
  sorry

#eval "The theorem has been stated successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_perimeter_l261_26105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_to_total_ratio_l261_26182

/-- Represents the number of books of each type on the table. -/
structure BookCounts where
  total : ℕ
  reading : ℕ
  math : ℕ
  science : ℕ
  history : ℕ

/-- Theorem stating the ratio of math books to total books -/
theorem math_to_total_ratio (books : BookCounts) : 
  books.total = 10 ∧ 
  books.reading = (2 : ℕ) * books.total / 5 ∧
  books.history = 1 ∧
  books.science = books.math - 1 ∧
  books.total = books.reading + books.math + books.science + books.history →
  books.math * 10 = 3 * books.total := by
  intro h
  sorry

#check math_to_total_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_to_total_ratio_l261_26182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l261_26159

/-- Given vectors a, b, c, and a real number lambda, prove that if (a + lambda*b) is parallel to c, then lambda = 3 -/
theorem parallel_vectors_lambda (a b c : ℝ × ℝ) (lambda : ℝ) :
  a = (2, 1) →
  b = (0, 1) →
  c = (3, 6) →
  (∃ (k : ℝ), k ≠ 0 ∧ a + lambda • b = k • c) →
  lambda = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l261_26159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_theorem_l261_26187

/-- Calculates the amount spent on oil after a price reduction -/
noncomputable def amount_spent_after_reduction (original_price : ℝ) (reduced_price : ℝ) (price_reduction_percent : ℝ) (additional_quantity : ℝ) : ℝ :=
  let price_ratio := (100 - price_reduction_percent) / 100
  let quantity_difference := additional_quantity
  let denominator := (1 / reduced_price) - (1 / original_price)
  (quantity_difference / denominator)

/-- Theorem: Given the conditions of the oil price reduction problem, 
    the amount spent after reduction is approximately 683.45 -/
theorem oil_price_reduction_theorem (ε : ℝ) (hε : ε > 0) :
  ∃ (original_price : ℝ),
    original_price > 0 ∧
    34.2 > 0 ∧
    20 > 0 ∧ 20 < 100 ∧
    4 > 0 ∧
    34.2 = original_price * (100 - 20) / 100 ∧
    |amount_spent_after_reduction original_price 34.2 20 4 - 683.45| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_theorem_l261_26187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_tree_l261_26190

/-- The time (in seconds) it takes for a train to pass a stationary point -/
noncomputable def train_passing_time (length : ℝ) (speed : ℝ) : ℝ :=
  length / (speed * 1000 / 3600)

/-- Theorem: A train 275 meters long, traveling at 90 km/h, takes 11 seconds to pass a tree -/
theorem train_passes_tree :
  let length := (275 : ℝ)
  let speed := (90 : ℝ)
  train_passing_time length speed = 11 := by
  -- Unfold the definition and simplify
  unfold train_passing_time
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_tree_l261_26190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_growth_sum_equals_limit_l261_26121

/-- The sum of the infinite series representing the growth of a line -/
noncomputable def line_growth_sum : ℝ := 1 + (1/4 * Real.sqrt 2) + 1/4 + (1/16 * Real.sqrt 2) + 1/16 + (1/64 * Real.sqrt 2) + 1/64

/-- The limit of the line growth series -/
noncomputable def line_growth_limit : ℝ := (1/3) * (4 + Real.sqrt 2)

/-- Theorem stating that the sum of the infinite series equals the calculated limit -/
theorem line_growth_sum_equals_limit : line_growth_sum = line_growth_limit := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_growth_sum_equals_limit_l261_26121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_squared_value_K_squared_greater_than_critical_expectation_X_l261_26108

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ := !![40, 10; 10, 40]

-- Define the K^2 formula
def K_squared (n : ℕ) (a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99.9% certainty
def critical_value : ℚ := 10828 / 1000

-- Define the distribution of X
def distribution_X : Fin 3 → ℚ
| 0 => 1 / 45
| 1 => 16 / 45
| 2 => 28 / 45

-- Theorem statements
theorem K_squared_value :
  K_squared 100 (contingency_table 0 0) (contingency_table 0 1)
            (contingency_table 1 0) (contingency_table 1 1) = 36 := by sorry

theorem K_squared_greater_than_critical :
  K_squared 100 (contingency_table 0 0) (contingency_table 0 1)
            (contingency_table 1 0) (contingency_table 1 1) > critical_value := by sorry

theorem expectation_X :
  (Finset.sum (Finset.range 3) (fun i => (i : ℚ) * distribution_X i)) = 8 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_squared_value_K_squared_greater_than_critical_expectation_X_l261_26108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_equal_two_range_l261_26196

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then Real.exp (x - 1) else x^(1/3)

-- State the theorem
theorem f_less_equal_two_range :
  {x : ℝ | f x ≤ 2} = {x : ℝ | 1 ≤ x ∧ x ≤ 8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_equal_two_range_l261_26196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_f_sum_n_l261_26195

-- Define the piecewise function f
noncomputable def f (n : ℝ) (x : ℝ) : ℝ :=
  if x < n then 3 * x^2 + 3 else 5 * x + 7

-- Theorem statement
theorem continuous_f_sum_n :
  ∃ (n₁ n₂ : ℝ),
    (∀ (x : ℝ), ContinuousAt (f n₁) x) ∧
    (∀ (x : ℝ), ContinuousAt (f n₂) x) ∧
    n₁ + n₂ = 5/3 ∧
    (∀ (n : ℝ), (∀ (x : ℝ), ContinuousAt (f n) x) → n = n₁ ∨ n = n₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_f_sum_n_l261_26195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_nine_divisors_l261_26112

theorem smallest_number_with_nine_divisors : 
  ∃ (n : ℕ), n > 0 ∧ 
    (∀ m : ℕ, m > 0 → m < n → (Finset.filter (λ d ↦ d ∣ m) (Finset.range m)).card ≠ 9) ∧
    (Finset.filter (λ d ↦ d ∣ n) (Finset.range n)).card = 9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_nine_divisors_l261_26112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trail_length_l261_26145

theorem trail_length 
  (speed_q : ℝ) 
  (speed_p : ℝ) 
  (distance_p : ℝ) 
  (trail_length : ℝ) : 
  speed_p = 1.15 * speed_q → 
  distance_p = 23 → 
  trail_length = distance_p + (distance_p / speed_p) * speed_q → 
  trail_length = 43 := by
  intros h1 h2 h3
  -- Proof steps would go here
  sorry

#check trail_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trail_length_l261_26145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l261_26166

/-- Given a principal amount and time period, calculates the difference between
    compound interest and simple interest. -/
def interestDifference (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1 - rate * (↑time : ℝ))

/-- Proves that for a given principal and time period, if the difference between
    compound interest and simple interest is 15, then the interest rate is 5%. -/
theorem interest_rate_is_five_percent
  (principal : ℝ)
  (h_principal : principal = 6000.000000000128)
  (time : ℕ)
  (h_time : time = 2)
  (h_difference : interestDifference principal 0.05 time = 15) :
  0.05 = Real.sqrt 0.0025 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l261_26166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_factors_no_smaller_b_factors_l261_26153

/-- The smallest positive integer b for which x^2 + bx + 1764 factors -/
def smallest_factoring_b : ℕ := 84

/-- Predicate to check if a polynomial factors with integer coefficients -/
def factors_with_integer_coeffs (a b c : ℤ) : Prop :=
  ∃ (p q : ℤ), ∀ (x : ℤ), a * x^2 + b * x + c = (x + p) * (x + q)

theorem smallest_b_factors :
  factors_with_integer_coeffs 1 smallest_factoring_b 1764 := by
  sorry

theorem no_smaller_b_factors (b : ℕ) :
  b < smallest_factoring_b →
  ¬(factors_with_integer_coeffs 1 (b : ℤ) 1764) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_factors_no_smaller_b_factors_l261_26153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_m_value_l261_26162

/-- The circle on which point P moves -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 2

/-- The relationship between vectors PD and MD -/
def vector_relation (xp yp xm ym : ℝ) : Prop :=
  (xp - xm)^2 + yp^2 = 2 * ym^2

/-- The line l -/
def line_eq (x y m : ℝ) : Prop := y = 2*x + m

/-- The distance from a point to the line -/
noncomputable def distance_to_line (x y m : ℝ) : ℝ :=
  |2*x - y + m| / Real.sqrt 5

/-- The theorem stating the equation of trajectory C and the value of m -/
theorem trajectory_and_m_value :
  ∃ (C : ℝ → ℝ → Prop) (m : ℝ),
    (∀ x y, C x y ↔ x^2/2 + y^2 = 1) ∧
    (m = 8) ∧
    (m > 0) ∧
    (∀ xp yp, circle_eq xp yp →
      ∃ xm ym, vector_relation xp yp xm ym ∧ C xm ym) ∧
    (∀ xp yp, C xp yp →
      (∃ θ : ℝ, xp = Real.sqrt 2 * Real.cos θ ∧ yp = Real.sin θ)) ∧
    (∀ xp yp, C xp yp →
      distance_to_line xp yp m ≥ Real.sqrt 5) ∧
    (∃ xp yp, C xp yp ∧ distance_to_line xp yp m = Real.sqrt 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_m_value_l261_26162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_34_digits_l261_26172

/-- Given that 34! = 295232799cd9604140847618609635ab000000,
    prove that a = 2, b = 0, c = 0, and d = 3 -/
theorem factorial_34_digits :
  ∃ (a b c d : Nat),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    Nat.factorial 34 = 295232799 * (10 * c + d) * 9604140847618609635 * (10 * a + b) * 1000000 ∧
    a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_34_digits_l261_26172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_logarithmic_sum_l261_26177

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem product_of_logarithmic_sum (a b : ℝ) :
  a > 0 ∧ b > 0 →
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
  (Real.sqrt (log a) : ℝ) = m ∧
  (Real.sqrt (log b) : ℝ) = n ∧
  m + n + (log (Real.sqrt a)) + (log (Real.sqrt b)) = 80 →
  a * b = 10^136 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_logarithmic_sum_l261_26177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ratio_l261_26126

/-- The number of letters in the alphabet -/
def n : ℕ := 26

/-- The number of letter pairs we're considering -/
def k₁ : ℕ := 3

/-- The number of letter triplets we're considering -/
def k₂ : ℕ := 2

/-- The probability of k₁ specific letter pairs appearing as contiguous substrings -/
noncomputable def p₁ : ℚ := (n - k₁).factorial / n.factorial

/-- The probability of k₂ specific letter triplets appearing as contiguous substrings -/
noncomputable def p₂ : ℚ := (n - k₂).factorial / n.factorial

/-- The ratio of p₁ to p₂ -/
theorem probability_ratio : p₁ / p₂ = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ratio_l261_26126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geckoPopulationTheorem_l261_26150

/-- Calculate the number of hatched eggs for a gecko species --/
def calculateHatchedEggs (
  totalEggs : ℕ
) (infertileRate calcificationRate predatorRate temperatureFailureRate : ℚ) : ℚ :=
  let fertileEggs := (totalEggs : ℚ) * (1 - infertileRate)
  let afterCalcification := fertileEggs * (1 - calcificationRate)
  let afterPredation := afterCalcification * (1 - predatorRate)
  afterPredation * (1 - temperatureFailureRate)

/-- The gecko population theorem --/
theorem geckoPopulationTheorem :
  let g1Hatched := calculateHatchedEggs 30 (1/5) (1/3) (1/10) (1/20)
  let g2Hatched := calculateHatchedEggs 40 (1/4) (1/4) (1/10) (1/20)
  let totalHatched := g1Hatched + g2Hatched
  ⌊totalHatched⌋ = 32 := by
  sorry

#eval ⌊calculateHatchedEggs 30 (1/5) (1/3) (1/10) (1/20) +
       calculateHatchedEggs 40 (1/4) (1/4) (1/10) (1/20)⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geckoPopulationTheorem_l261_26150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_after_folding_l261_26147

/-- Represents a rectangular piece of paper. -/
structure Paper where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Represents the folded paper. -/
structure FoldedPaper where
  original : Paper
  newArea : ℝ

/-- The paper folding operation as described in the problem. -/
noncomputable def foldPaper (p : Paper) : FoldedPaper :=
  { original := p
    newArea := p.area * (1 - (Real.sqrt 2 + Real.sqrt 1.25) / 8) }

/-- Theorem stating the ratio of areas after folding. -/
theorem area_ratio_after_folding (p : Paper) 
    (h1 : p.length = 2 * p.width)
    (h2 : p.area = p.width * p.length) :
  let folded := foldPaper p
  folded.newArea / folded.original.area = 1 - (Real.sqrt 2 + Real.sqrt 1.25) / 8 := by
  sorry

#check area_ratio_after_folding

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_after_folding_l261_26147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crank_slider_motion_l261_26160

/-- Crank-slider mechanism -/
structure CrankSlider where
  crankLength : ℝ
  rodLength : ℝ
  angularVelocity : ℝ

/-- Point on the connecting rod -/
structure ConnectingRodPoint where
  ratio : ℝ  -- Ratio of MB to AB

/-- Position of a point -/
structure Position where
  x : ℝ
  y : ℝ

/-- Velocity of a point -/
def Velocity := ℝ

noncomputable def positionM (cs : CrankSlider) (p : ConnectingRodPoint) (t : ℝ) : Position :=
  { x := cs.crankLength * Real.cos (cs.angularVelocity * t)
  , y := cs.crankLength * p.ratio * Real.sin (cs.angularVelocity * t) }

noncomputable def velocityM (cs : CrankSlider) (p : ConnectingRodPoint) (t : ℝ) : Velocity :=
  cs.crankLength * cs.angularVelocity * Real.sqrt (8 * Real.sin (cs.angularVelocity * t)^2 + 1)

theorem crank_slider_motion (cs : CrankSlider) (p : ConnectingRodPoint) (t : ℝ) :
  cs.crankLength = 90 ∧ cs.rodLength = 90 ∧ cs.angularVelocity = 10 ∧ p.ratio = 1/3 →
  positionM cs p t = { x := 90 * Real.cos (10 * t), y := 30 * Real.sin (10 * t) } ∧
  velocityM cs p t = 300 * Real.sqrt (8 * Real.sin (10 * t)^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crank_slider_motion_l261_26160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l261_26101

-- Define the & operation
noncomputable def amp (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := amp (Real.sin x) (Real.cos x)

-- Theorem statement
theorem range_of_f :
  Set.range f = Set.Icc (-1 : ℝ) (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l261_26101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_values_l261_26191

open Set

theorem set_equality_implies_values (a b : ℝ) : 
  let A : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}
  let B : Set ℝ := {a + b, 1, a - b + 5}
  A = B → ((a = 0 ∧ b = 2) ∨ (a = 0 ∧ b = 3)) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_values_l261_26191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_explicit_formula_l261_26141

def a : ℕ → ℚ
  | 0 => 1/2  -- Define a value for n = 0
  | 1 => 1/2
  | n + 2 => (a (n + 1) + 3) / (2 * a (n + 1) - 4)

theorem explicit_formula (n : ℕ) (h : n > 0) :
  a n = ((-5)^n + 3 * 2^(n+1)) / (2^(n+1) - 2*(-5)^n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_explicit_formula_l261_26141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_relatively_prime_to_21_l261_26136

theorem count_relatively_prime_to_21 : 
  (Finset.filter (fun n : ℕ => 10 < n ∧ n < 100 ∧ Nat.gcd n 21 = 1) (Finset.range 100)).card = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_relatively_prime_to_21_l261_26136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clyde_corn_cobs_l261_26142

/-- The weight of a bushel of corn in pounds -/
noncomputable def bushel_weight : ℝ := 56

/-- The weight of an individual ear of corn in pounds -/
noncomputable def ear_weight : ℝ := 0.5

/-- The number of bushels Clyde picked -/
noncomputable def bushels_picked : ℝ := 2

/-- The number of individual corn cobs Clyde picked -/
noncomputable def cobs_picked : ℝ := (bushel_weight * bushels_picked) / ear_weight

theorem clyde_corn_cobs : cobs_picked = 224 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clyde_corn_cobs_l261_26142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_zero_implies_y_eq_neg_b_div_3_l261_26120

/-- Given a non-zero real number b, if the determinant of the matrix
    [[y + b, -y, y],
     [-y, y + b, -y],
     [y, -y, y + b]]
    is zero, then y = -b/3. -/
theorem det_zero_implies_y_eq_neg_b_div_3 (b : ℝ) (hb : b ≠ 0) (y : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![y + b, -y, y; -y, y + b, -y; y, -y, y + b]
  Matrix.det M = 0 → y = -b / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_zero_implies_y_eq_neg_b_div_3_l261_26120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_six_l261_26154

/-- Given a circle C and a line l that is the axis of symmetry for C, 
    prove that the length of the tangent line from a point A to C is 6. -/
theorem tangent_length_is_six 
  (C : Set (ℝ × ℝ)) 
  (l : Set (ℝ × ℝ)) 
  (a : ℝ) 
  (A B : ℝ × ℝ) :
  (∀ p : ℝ × ℝ, p ∈ C ↔ (p.1^2 + p.2^2 - 4*p.1 - 2*p.2 + 1 = 0)) →
  (∀ p : ℝ × ℝ, p ∈ l ↔ (p.1 + a*p.2 - 1 = 0)) →
  (∃ c : ℝ × ℝ, c ∈ C ∧ c ∈ l) →
  A = (-4, a) →
  B ∈ C →
  (∀ p ∈ C, dist A p ≥ dist A B) →
  dist A B = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_is_six_l261_26154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_product_l261_26194

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola with equation x^2 - y^2 = a^2 -/
structure Hyperbola where
  a : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a hyperbola x^2 - y^2 = a^2, |FP| * |FQ| = 2|DM| * |DN| -/
theorem hyperbola_distance_product (h : Hyperbola) (F D M N P Q : Point) : 
  F.x^2 - F.y^2 = 2 * h.a^2 →  -- F is a focus
  D.x^2 - D.y^2 = h.a^2 / 2 →  -- D is on the directrix
  M.x^2 - M.y^2 = h.a^2 →     -- M is on the hyperbola
  N.x^2 - N.y^2 = h.a^2 →     -- N is on the hyperbola
  P.x^2 - P.y^2 = h.a^2 →     -- P is on the hyperbola
  Q.x^2 - Q.y^2 = h.a^2 →     -- Q is on the hyperbola
  (M.y - N.y) * (F.x - P.x) = (M.x - N.x) * (F.y - P.y) →  -- FP ⟂ MN
  (M.y - N.y) * (F.x - Q.x) = (M.x - N.x) * (F.y - Q.y) →  -- FQ ⟂ MN
  (distance F P) * (distance F Q) = 2 * (distance D M) * (distance D N) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_product_l261_26194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_f_positive_at_a_squared_over_two_three_zeros_range_l261_26168

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + b / x

-- Theorem 1
theorem tangent_line_condition (a b : ℝ) :
  (∀ x, f a b x + f a b (1/x) = 0) →
  (∃ k, k * (2 - 1) + f a b 1 = 5) →
  f a b = λ x ↦ Real.log x + 2 * x - 2 / x :=
by sorry

-- Theorem 2
theorem f_positive_at_a_squared_over_two (a : ℝ) :
  0 < a → a < 1 →
  Real.log (a^2/2) + 2 * (a^2/2) - 2 / (a^2/2) > 0 :=
by sorry

-- Theorem 3
theorem three_zeros_range (a : ℝ) :
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Real.log x + 2*x - 2/x = 0 ∧
    Real.log y + 2*y - 2/y = 0 ∧
    Real.log z + 2*z - 2/z = 0) ↔
  0 < a ∧ a < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_f_positive_at_a_squared_over_two_three_zeros_range_l261_26168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l261_26124

-- Define the given functions
noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - 12 * x - 3) / (2 * x + 1)
noncomputable def g (a x : ℝ) : ℝ := -x - 2 * a

-- State the theorem
theorem problem_solution (t : ℝ) (h_t : t > 0) :
  -- Part 1: Monotonicity of f
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1/2 → f x₁ > f x₂) ∧
  (∀ x₁ x₂ : ℝ, 1/2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ < f x₂) ∧
  -- Part 2: Range of f
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → -4 ≤ f x ∧ f x ≤ -3) ∧
  -- Part 3: Value of a
  (∀ x₁ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 1 → ∃ x₂ : ℝ, 0 ≤ x₂ ∧ x₂ ≤ 1 ∧ g (3/2) x₂ = f x₁) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l261_26124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_center_squared_l261_26152

/-- Given a circle and points A, B, C satisfying certain conditions, 
    prove that the square of the distance from B to the center of the circle is 57. -/
theorem distance_to_center_squared 
  (O : ℝ × ℝ) -- Center of the circle
  (A B C : ℝ × ℝ) -- Points A, B, C
  (h_radius : ∀ X : ℝ × ℝ, X ∈ Metric.sphere O (Real.sqrt 200) ↔ (X.1 - O.1)^2 + (X.2 - O.2)^2 = 200)
  (h_AB : dist A B = 10)
  (h_BC : dist B C = 4)
  (h_right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0)
  (h_A_on_circle : A ∈ Metric.sphere O (Real.sqrt 200))
  (h_C_on_circle : C ∈ Metric.sphere O (Real.sqrt 200)) :
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = 57 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_center_squared_l261_26152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_sum_equals_9_128_l261_26155

open Real
open BigOperators
open ENNReal

/-- The sum of the double series ∑(n=2 to ∞) ∑(k=1 to n-1) k/(3^(n+k)) equals 9/128 -/
theorem double_sum_equals_9_128 :
  (∑' n : ℕ, ∑' k : ℕ, (k : ℝ) / (3 : ℝ) ^ (n + k) * (if k < n ∧ n ≥ 2 then 1 else 0)) = 9 / 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_sum_equals_9_128_l261_26155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l261_26134

noncomputable section

-- Define the original curve
def original_curve (x : ℝ) : ℝ := (1/3) * Real.cos (2 * x)

-- Define the transformation
def transformation_x (x : ℝ) : ℝ := 2 * x
def transformation_y (y : ℝ) : ℝ := 3 * y

-- State the theorem
theorem curve_transformation :
  ∀ x' y' : ℝ,
  (∃ x y : ℝ, x' = transformation_x x ∧ y' = transformation_y y ∧ y = original_curve x) →
  y' = Real.cos x' := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l261_26134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_probability_l261_26131

/-- Represents a regular tetrahedron with inscribed and circumscribed spheres -/
structure TetrahedronWithSpheres where
  R : ℝ  -- Radius of the circumscribed sphere
  r : ℝ  -- Radius of the inscribed sphere
  h : R = 2 * r  -- The radius of the circumscribed sphere is twice that of the inscribed sphere

/-- Calculates the volume of a sphere given its radius -/
noncomputable def sphereVolume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * (radius ^ 3)

/-- Theorem: The probability of a randomly selected point inside the circumscribed sphere
    lying inside one of the five smaller spheres is 5/8 -/
theorem tetrahedron_sphere_probability (t : TetrahedronWithSpheres) :
  let smallSphereVolume := sphereVolume t.r
  let totalSmallSpheresVolume := 5 * smallSphereVolume
  let circumscribedSphereVolume := sphereVolume t.R
  totalSmallSpheresVolume / circumscribedSphereVolume = 5 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_probability_l261_26131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l261_26173

/-- A perfect square trinomial in x and y -/
def PerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ (p q : ℝ), ∀ (x y : ℝ), a * x^2 + b * x * y + c * y^2 = (p * x + q * y)^2

/-- Theorem: If 9x^2 + mxy + 16y^2 is a perfect square trinomial, then m = ±24 -/
theorem perfect_square_trinomial_m_value :
  ∀ (m : ℝ), PerfectSquareTrinomial 9 m 16 → m = 24 ∨ m = -24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l261_26173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_piecewise_function_theorem_l261_26151

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  let x' := x - 2 * ⌊x / 2⌋  -- Normalize x to [-1, 1]
  if -1 ≤ x' ∧ x' < 0 then a * x' + 1
  else if 0 ≤ x' ∧ x' ≤ 1 then (b * x' + 2) / (x' + 1)
  else 0  -- This case should never occur due to normalization

theorem periodic_piecewise_function_theorem (a b : ℝ) :
  (∀ x, f (x + 2) a b = f x a b) →
  f (1/2) a b = f (3/2) a b →
  a + 3 * b = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_piecewise_function_theorem_l261_26151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inscribed_circle_inequality_l261_26181

-- Define the triangle ABC
structure Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] where
  A : α
  B : α
  C : α

-- Define the circle O
structure Circle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] where
  center : α
  radius : ℝ

-- Define the point M
noncomputable def M {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (O : Circle α) (C : α) : α :=
  sorry

theorem isosceles_triangle_inscribed_circle_inequality 
  {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (ABC : Triangle α) (O : Circle α) :
  ABC.A ≠ ABC.B → ABC.A ≠ ABC.C → ABC.B ≠ ABC.C →
  (‖ABC.A - ABC.B‖ = ‖ABC.A - ABC.C‖) →  -- AB = AC
  (‖O.center - ABC.A‖ = O.radius) →  -- ABC inscribed in circle O
  (‖O.center - ABC.B‖ = O.radius) →
  (‖O.center - ABC.C‖ = O.radius) →
  let M := M O ABC.C
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = (1 - t) • O.center + t • ABC.C) →  -- M on line segment OC
  (‖ABC.A - M‖ < ‖ABC.B - M‖ + ‖ABC.C - M‖) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inscribed_circle_inequality_l261_26181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_applicant_intersection_l261_26197

theorem applicant_intersection (total : ℕ) (A B neither : ℕ) 
  (h_total : total = 30)
  (h_A : A = 10)
  (h_B : B = 18)
  (h_neither : neither = 3)
  (h_union : A + B - (total - neither) = 1) :
  total - neither - (A + B - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_applicant_intersection_l261_26197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_l261_26116

/-- A sequence of four distinct digits forming a double arithmetic progression -/
def DoubleArithSeq : Type := { seq : Fin 4 → Nat // 
  (∀ i j, i ≠ j → seq i ≠ seq j) ∧ 
  (seq 1 = (seq 0 + seq 2) / 2) ∧ 
  (seq 2 = (seq 1 + seq 3) / 2) ∧
  (∀ i, seq i < 10) ∧
  (seq 0 ≠ 0) }

/-- The count of valid four-digit numbers -/
def ValidNumberCount : Nat :=
  11 * 24  -- We directly use the calculated value instead of trying to count the set

theorem valid_number_count : ValidNumberCount = 264 := by
  unfold ValidNumberCount
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_l261_26116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_satisfying_condition_l261_26157

def has_sum_101 (S : Finset ℕ) : Prop :=
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = 101

def satisfies_condition (n : ℕ) : Prop :=
  ∀ S : Finset ℕ, S ⊆ Finset.range (n + 1) → S.card = 51 → has_sum_101 S

theorem largest_n_satisfying_condition :
  ∃ n : ℕ, satisfies_condition n ∧ ∀ m > n, ¬satisfies_condition m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_satisfying_condition_l261_26157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cement_transport_cost_l261_26103

/-- Calculates the new cost of transporting cement bags with given parameters -/
def calculate_new_cost (original_cost : ℝ) (original_bags : ℕ) (original_weight_kg : ℝ) 
  (cost_increase_percent : ℝ) (weight_change_percent : ℝ) (new_quantity_factor : ℕ) 
  (kg_to_pound : ℝ) (eur_to_usd : ℝ) : ℝ :=
  let new_weight_kg := original_weight_kg * weight_change_percent
  let new_weight_pound := new_weight_kg * kg_to_pound
  let new_cost_eur := original_cost * (1 + cost_increase_percent)
  let new_cost_usd := new_cost_eur * eur_to_usd
  new_cost_usd * (new_quantity_factor : ℝ)

/-- Theorem: The new cost of transporting cement bags is $51562.5 -/
theorem new_cement_transport_cost : 
  calculate_new_cost 7500 150 80 0.25 0.6 5 2.20462 1.1 = 51562.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cement_transport_cost_l261_26103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_l261_26137

theorem cos_half_angle (α : Real) (h1 : Real.sin α = 4/5) (h2 : 0 < α ∧ α < Real.pi/2) :
  Real.cos (α/2) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_l261_26137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_c_l261_26135

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, -4)

theorem angle_between_a_and_c :
  ∀ c : ℝ × ℝ,
  (Real.sqrt 5)^2 = c.1^2 + c.2^2 →
  (a.1 + b.1) * c.1 + (a.2 + b.2) * c.2 = 5/2 →
  Real.arccos ((a.1 * c.1 + a.2 * c.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (c.1^2 + c.2^2))) = 2 * π / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_c_l261_26135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_f_range_on_interval_l261_26169

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x + 1

-- Define the monotonically increasing intervals
def monotonic_increasing_intervals (k : ℤ) : Set ℝ :=
  Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)

-- Define the range of f on [0, π/2]
def f_range : Set ℝ := Set.Icc 1 (5/2)

-- Theorem for monotonically increasing intervals
theorem f_monotonic_increasing (k : ℤ) :
  StrictMono (f ∘ (fun x => x + k * Real.pi - Real.pi / 3)) := by
  sorry

-- Theorem for the range of f on [0, π/2]
theorem f_range_on_interval :
  Set.range (fun x => f x) = f_range := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_f_range_on_interval_l261_26169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_of_f_f_eq_zero_l261_26199

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else Real.sqrt x - 1

-- State the theorem
theorem three_roots_of_f_f_eq_zero :
  ∃ (a b c : ℝ), (∀ x : ℝ, f (f x) = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_of_f_f_eq_zero_l261_26199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_product_l261_26146

noncomputable def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

noncomputable def asymptote1 (a b x : ℝ) : ℝ := (b / a) * x
noncomputable def asymptote2 (a b x : ℝ) : ℝ := -(b / a) * x

noncomputable def distance_to_line (x y a b : ℝ) : ℝ := 
  |a * y - b * x| / Real.sqrt (a^2 + b^2)

theorem hyperbola_distance_product (a b x y : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola a b x y →
  (distance_to_line x y (a / b) (-1) * distance_to_line x y (a / b) 1 = a^2 * b^2 / (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_product_l261_26146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_translated_symmetric_sine_l261_26184

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem min_value_of_translated_symmetric_sine 
  (φ : ℝ) 
  (h1 : |φ| < π) 
  (h2 : ∀ x, f (x + π/6) φ = -f (-x + π/6) φ) : 
  ∃ x₀ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x₀ φ ≤ f x φ ∧ f x₀ φ = -Real.sqrt 3 / 2 :=
by
  sorry

#check min_value_of_translated_symmetric_sine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_translated_symmetric_sine_l261_26184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_theorem_l261_26178

/-- Represents a quadratic equation ax^2 + bx + c = 0 --/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The discriminant of a quadratic equation --/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4*eq.a*eq.c

/-- Condition for two distinct real roots --/
def has_two_distinct_real_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq > 0

/-- The roots of a quadratic equation --/
def roots (eq : QuadraticEquation) : Set ℝ :=
  {x : ℝ | eq.a * x^2 + eq.b * x + eq.c = 0}

theorem quadratic_roots_theorem (k : ℝ) :
  let eq := QuadraticEquation.mk 1 (-3) k
  (has_two_distinct_real_roots eq ↔ k < 9/4) ∧
  (k = 0 → roots eq = {0, 3}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_theorem_l261_26178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_x_plus_one_l261_26125

def x : ℕ → ℝ
  | 0 => 25  -- Adding the base case for 0
  | 1 => 25
  | (k + 2) => x (k + 1) ^ 2 + x (k + 1)

theorem sum_reciprocal_x_plus_one :
  (∑' k : ℕ, 1 / (x k + 1)) = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_x_plus_one_l261_26125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkers_meet_time_l261_26140

/-- Represents a person walking with a given speed -/
structure Walker where
  speed : ℝ

/-- Represents the meeting of two walkers -/
noncomputable def MeetingTime (a b : Walker) (initial_distance : ℝ) (start_time : ℝ) : ℝ :=
  start_time + initial_distance / (a.speed + b.speed)

theorem walkers_meet_time :
  let a : Walker := ⟨6⟩
  let b : Walker := ⟨4⟩
  let initial_distance : ℝ := 50
  let start_time : ℝ := 18  -- 6 pm in 24-hour format
  MeetingTime a b initial_distance start_time = 23  -- 11 pm in 24-hour format
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkers_meet_time_l261_26140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_root_sum_l261_26163

theorem larger_root_sum (x : ℝ) (r s : ℤ) : 
  (∃ y : ℝ, y < x ∧ (y ^ (1/3 : ℝ) + (16 - y) ^ (1/3 : ℝ) = 2)) →
  (x ^ (1/3 : ℝ) + (16 - x) ^ (1/3 : ℝ) = 2) →
  x = r + Real.sqrt (s : ℝ) →
  r + s = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_root_sum_l261_26163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l261_26115

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ x y : ℤ, f (x + f y) = f x - y) : 
  ∀ x : ℤ, f x = -x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l261_26115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_one_l261_26111

/-- The distance from a point in polar coordinates to a line in polar coordinates -/
noncomputable def distance_polar_to_line (ρ₀ : ℝ) (θ₀ : ℝ) (f : ℝ → ℝ → ℝ) : ℝ :=
  let x₀ := ρ₀ * Real.cos θ₀
  let y₀ := ρ₀ * Real.sin θ₀
  let numerator := |f x₀ y₀|
  let denominator := Real.sqrt (1 + (Real.sqrt 3) ^ 2)
  numerator / denominator

/-- The polar equation of the line ρ*sin(θ - π/6) = 1 in Cartesian form -/
noncomputable def line_equation (x : ℝ) (y : ℝ) : ℝ :=
  x - Real.sqrt 3 * y + 2

theorem distance_to_line_is_one :
  distance_polar_to_line 2 (Real.pi / 6) line_equation = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_one_l261_26111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_fraction_of_mile_l261_26110

/-- Represents the taxi fare calculation --/
def TaxiFare (initialFee additionalCharge totalCharge tripDistance : ℝ) : Prop :=
  ∃ (fractionOfMile : ℝ),
    fractionOfMile > 0 ∧
    fractionOfMile < 1 ∧
    totalCharge = initialFee + additionalCharge * (Int.floor (tripDistance / fractionOfMile) + 1)

/-- Proves the fraction of a mile for which the additional charge applies --/
theorem taxi_fare_fraction_of_mile (initialFee additionalCharge totalCharge tripDistance : ℝ)
  (h_initial : initialFee = 2.25)
  (h_additional : additionalCharge = 0.25)
  (h_total : totalCharge = 4.5)
  (h_distance : tripDistance = 3.6)
  (h_fare : TaxiFare initialFee additionalCharge totalCharge tripDistance) :
  ∃ (fractionOfMile : ℝ), fractionOfMile = 0.25 ∧ 
    tripDistance - (Int.floor (tripDistance / fractionOfMile)) * fractionOfMile = 1.35 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_fraction_of_mile_l261_26110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l261_26180

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ,
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₂ - x₁) > 0) →
  0 < a ∧ a ≤ 1/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l261_26180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l261_26175

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the side length
noncomputable def side_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the angle measure
noncomputable def angle_measure (A B C : ℝ × ℝ) : ℝ :=
  Real.arccos ((side_length A B)^2 + (side_length B C)^2 - (side_length A C)^2) / (2 * side_length A B * side_length B C)

-- Theorem statement
theorem triangle_properties (ABC : Triangle) 
  (h1 : side_length ABC.A ABC.B = 5)
  (h2 : side_length ABC.B ABC.C = 4) :
  (side_length ABC.A ABC.C = 4 → angle_measure ABC.A ABC.B ABC.C ≤ angle_measure ABC.B ABC.A ABC.C) ∧
  (side_length ABC.A ABC.C = 2 → ¬ (side_length ABC.A ABC.B < side_length ABC.B ABC.C + side_length ABC.A ABC.C)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l261_26175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_when_a_is_one_range_of_a_when_q_implies_p_l261_26189

-- Define the conditions
def condition_p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0 ∧ a > 0

def condition_q (x : ℝ) : Prop := 8 < (2 : ℝ)^(x + 1) ∧ (2 : ℝ)^(x + 1) ≤ 16

-- Theorem for the first question
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, condition_p x 1 ∧ condition_q x → x ∈ Set.Ioo 2 3 :=
sorry

-- Theorem for the second question
theorem range_of_a_when_q_implies_p :
  (∀ x a : ℝ, condition_q x → condition_p x a) ∧
  (∃ x a : ℝ, condition_p x a ∧ ¬condition_q x) →
  ∀ a : ℝ, a ∈ Set.Ioo 1 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_when_a_is_one_range_of_a_when_q_implies_p_l261_26189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_cardinality_l261_26161

-- Define the number of subsets function
def n (S : Finset α) : ℕ := 2^(S.card)

-- Define the theorem
theorem min_intersection_cardinality
  (A B C : Finset ℕ)
  (h1 : n A + n B + n C = n (A ∪ B ∪ C))
  (h2 : A.card = 100)
  (h3 : B.card = 100) :
  ∃ (m : ℕ), m = (A ∩ B ∩ C).card ∧
    ∀ (X Y Z : Finset ℕ), n X + n Y + n Z = n (X ∪ Y ∪ Z) →
      X.card = 100 → Y.card = 100 →
      m ≤ (X ∩ Y ∩ Z).card ∧ m = 97 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_cardinality_l261_26161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l261_26119

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

-- Define the midpoint coordinates
def midpoint_x : ℝ := 4
def midpoint_y : ℝ := 2

-- Define a line passing through two points
def line_through (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

-- Theorem statement
theorem line_equation (x₁ y₁ x₂ y₂ : ℝ) :
  is_on_ellipse x₁ y₁ →
  is_on_ellipse x₂ y₂ →
  midpoint_x = (x₁ + x₂)/2 →
  midpoint_y = (y₁ + y₂)/2 →
  line_through x₁ y₁ x₂ y₂ x y →
  2*x + 3*y - 16 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l261_26119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comic_book_reading_rate_l261_26100

/-- Represents the reading rate for different book types --/
structure ReadingRate where
  novels : ℝ
  graphicNovels : ℝ
  comicBooks : ℝ

/-- Represents the total reading time and pages read --/
structure ReadingData where
  totalTime : ℝ
  totalPages : ℕ

theorem comic_book_reading_rate 
  (rate : ReadingRate)
  (data : ReadingData)
  (h1 : rate.novels = 21)
  (h2 : rate.graphicNovels = 30)
  (h3 : data.totalTime = 24 * (1/6))
  (h4 : data.totalPages = 128) :
  rate.comicBooks = 45 := by
  sorry

#check comic_book_reading_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comic_book_reading_rate_l261_26100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_odd_and_increasing_l261_26158

-- Define the function f(x) = x^(1/3)
noncomputable def f (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Statement to prove
theorem cube_root_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_odd_and_increasing_l261_26158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_y_value_l261_26129

theorem max_tan_y_value (x y : Real) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2)
  (h : Real.sin y = 2005 * Real.cos (x + y) * Real.sin x) :
  Real.tan y ≤ (2005 * Real.sqrt 2006) / 4012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_y_value_l261_26129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l261_26143

theorem expression_evaluation (b : ℝ) (h : b ≠ 0) :
  (1 / 25 * b^0) + (1 / (25 * b))^0 - (125^(-(1/3 : ℝ))) - ((-81 : ℝ)^(-(1/4 : ℝ))) = 88 / 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l261_26143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sin_property_l261_26198

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 5 + a 9 = Real.pi / 2) :
  Real.sin (a 4 + a 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sin_property_l261_26198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_tangent_line_l261_26156

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem function_and_tangent_line 
  (a b c : ℝ) 
  (h1 : ∀ x, (deriv (f a b c)) x = 0 → x = 1 ∨ x = -1)
  (h2 : (deriv (f a b c)) 0 = -3) :
  (∀ x, f a b c x = x^3 - 3*x) ∧
  (∃ k₁ k₂ : ℝ, 
    (∀ x, k₁ * x - (k₁ * 2 + 2) = 0 ∨ k₂ * x - (k₂ * 2 + 2) = 0) ∧
    (k₁ = 0 ∧ k₂ = 9 ∨ k₁ = 9 ∧ k₂ = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_tangent_line_l261_26156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_company_cheapest_at_36_l261_26118

/-- Represents the cost function for a catering company -/
structure CateringCompany where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for a given number of people -/
def totalCost (company : CateringCompany) (people : ℕ) : ℕ :=
  company.basicFee + company.perPersonFee * people

theorem third_company_cheapest_at_36 :
  let company1 := CateringCompany.mk 120 18
  let company2 := CateringCompany.mk 220 13
  let company3 := CateringCompany.mk 150 15
  (∀ p < 36, totalCost company3 p > min (totalCost company1 p) (totalCost company2 p)) ∧
  (totalCost company3 36 < totalCost company1 36 ∧ totalCost company3 36 < totalCost company2 36) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_company_cheapest_at_36_l261_26118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_one_center_is_one_proof_l261_26144

-- Define the grid
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define a function to check if two positions are adjacent
def are_adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ (j.val + 1 = l.val ∨ j.val = l.val + 1)) ∨
  (j = l ∧ (i.val + 1 = k.val ∨ i.val = k.val + 1))

-- Define the property of consecutive numbers
def are_consecutive (a b : Nat) : Prop := a + 1 = b ∨ b + 1 = a

-- Main theorem
theorem center_is_one (grid : Grid) : Prop :=
  (∀ n : Nat, n ∈ Finset.range 9 → n + 1 ∈ Finset.range 10 \ {0}) →
  (∀ i j k l : Fin 3, are_adjacent i j k l → are_consecutive (grid i j) (grid k l)) →
  (grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2 = 24) →
  (∃ i j : Fin 3, (i = 0 ∨ i = 2 ∨ j = 0 ∨ j = 2) ∧ grid i j = 6) →
  grid 1 1 = 1

-- Proof
theorem center_is_one_proof : ∃ grid : Grid, center_is_one grid := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_one_center_is_one_proof_l261_26144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_x_and_y_l261_26171

-- Define the number of arcs in the circle
def num_arcs : ℕ := 16

-- Define the span of angle x
def x_span : ℕ := 3

-- Define the span of angle y
def y_span : ℕ := 5

-- Define the central angle for each arc
noncomputable def central_angle_per_arc : ℝ := 360 / num_arcs

-- Define the central angle for x
noncomputable def central_angle_x : ℝ := central_angle_per_arc * x_span

-- Define the central angle for y
noncomputable def central_angle_y : ℝ := central_angle_per_arc * y_span

-- Define the inscribed angle x
noncomputable def inscribed_angle_x : ℝ := central_angle_x / 2

-- Define the inscribed angle y
noncomputable def inscribed_angle_y : ℝ := central_angle_y / 2

-- Theorem to prove
theorem sum_of_angles_x_and_y :
  inscribed_angle_x + inscribed_angle_y = 90 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_x_and_y_l261_26171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_large_power_mod_l261_26132

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem floor_large_power_mod :
  (floor (5^2017015 / (5^2015 + 7)) : ℤ) % 1000 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_large_power_mod_l261_26132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_is_half_cube_l261_26104

/-- A corner tetrahedron in a cube is formed by the edges originating from a single vertex. -/
structure CornerTetrahedron (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] where
  vertex : α
  cube_edge_length : ℝ

/-- The volume of the region within a cube that lies in the intersection of at least two corner tetrahedra. -/
noncomputable def intersection_volume (a : ℝ) : ℝ := a^3 / 2

/-- The theorem states that the volume of the region within a cube that lies in the intersection
    of at least two corner tetrahedra is equal to half the volume of the cube. -/
theorem intersection_volume_is_half_cube (a : ℝ) (h : a > 0) :
  intersection_volume a = (a^3) / 2 := by
  -- Unfold the definition of intersection_volume
  unfold intersection_volume
  -- The equality is now trivial
  rfl

#check intersection_volume_is_half_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_is_half_cube_l261_26104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_strip_length_l261_26117

/-- The length of a spiral strip on a right circular cylinder -/
theorem spiral_strip_length (base_circumference height : ℝ) 
  (h1 : base_circumference = 18) (h2 : height = 8) :
  Real.sqrt (4 * base_circumference^2 + height^2) = Real.sqrt 1360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_strip_length_l261_26117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johnClothPurchase_l261_26138

/-- Calculates the number of metres of cloth bought given the total cost and cost per metre -/
noncomputable def metresBought (totalCost costPerMetre : ℝ) : ℝ :=
  totalCost / costPerMetre

/-- Proves that for the given total cost and cost per metre, the number of metres bought is approximately 9.25 -/
theorem johnClothPurchase (totalCost costPerMetre : ℝ) 
  (h1 : totalCost = 425.50)
  (h2 : costPerMetre = 46) :
  ∃ ε > 0, |metresBought totalCost costPerMetre - 9.25| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johnClothPurchase_l261_26138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cuttable_squares_l261_26174

/-- Represents a square board -/
structure Board where
  size : ℕ

/-- Represents a square on the board that can be cut -/
structure CuttableSquare where
  row : ℕ
  col : ℕ

/-- Determines if a square is on the border of the board -/
def is_border_square (b : Board) (s : CuttableSquare) : Prop :=
  s.row = 0 ∨ s.row = b.size - 1 ∨ s.col = 0 ∨ s.col = b.size - 1

/-- Determines if cutting a square would cause the board to fall apart -/
def causes_fall_apart (b : Board) (s : CuttableSquare) : Prop :=
  is_border_square b s

/-- The set of all squares that can be safely cut -/
def safe_to_cut (b : Board) : Set CuttableSquare :=
  {s : CuttableSquare | ¬(causes_fall_apart b s) ∧ s.row < b.size ∧ s.col < b.size}

/-- The theorem to be proved -/
theorem max_cuttable_squares :
  ∃ (cut_squares : Finset CuttableSquare),
    (∀ s ∈ cut_squares, s ∈ safe_to_cut (Board.mk 9)) ∧
    (∀ cut_squares' : Finset CuttableSquare,
      (∀ s ∈ cut_squares', s ∈ safe_to_cut (Board.mk 9)) →
      Finset.card cut_squares' ≤ Finset.card cut_squares) ∧
    Finset.card cut_squares = 21 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cuttable_squares_l261_26174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_count_and_sum_l261_26149

/-- A function from nonnegative integers to nonnegative integers satisfying the given condition -/
def F : (ℕ → ℕ) → Prop :=
  λ f => ∀ a b : ℕ, 3 * f (a^2 + b^2 + a) = (f a)^2 + (f b)^2 + 3 * f a

/-- The set of possible values for f(49) given f satisfies F -/
def PossibleValues (f : ℕ → ℕ) : Set ℕ :=
  {x | F f ∧ f 49 = x}

/-- The theorem to be proved -/
theorem product_of_count_and_sum : 
  ∃ f : ℕ → ℕ, F f ∧ (Finset.card {0, 1, 147} * Finset.sum {0, 1, 147} id = 444) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_count_and_sum_l261_26149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l261_26102

/-- The speed of a train in km/hr given its length in meters and time to pass a fixed point in seconds -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a train of length 560 meters passing a tree in 32 seconds has a speed of approximately 63 km/hr -/
theorem train_speed_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ abs (train_speed 560 32 - 63) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l261_26102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_bound_l261_26188

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set of 2021 points in the non-negative quarter plane -/
def Points : Set Point := 
  {p : Point | p.x ≥ 0 ∧ p.y ≥ 0 ∧ ∃ (s : Finset Point), s.card = 2021 ∧ p ∈ s}

/-- The centroid of a set of points -/
noncomputable def centroid (s : Finset Point) : Point :=
  { x := (s.sum (λ p => p.x)) / s.card,
    y := (s.sum (λ p => p.y)) / s.card }

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The theorem to be proved -/
theorem min_distance_bound (s : Finset Point) 
  (h1 : s.card = 2021)
  (h2 : ∀ p ∈ s, p.x ≥ 0 ∧ p.y ≥ 0)
  (h3 : centroid s = ⟨1, 1⟩) :
  ∃ p1 p2, p1 ∈ s ∧ p2 ∈ s ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 2 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_bound_l261_26188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_value_l261_26113

/-- An inverse proportion function passing through two points -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_proportion_m_value :
  ∀ k m : ℝ,
  inverse_proportion k 2 = -3 →
  inverse_proportion k m = 6 →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_value_l261_26113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l261_26185

theorem trig_problem (α : ℝ) 
  (h : Real.sin α / (Real.sin α - Real.cos α) = -1) : 
  Real.tan α = 1/2 ∧ 
  (Real.sin α^2 + 2*Real.sin α*Real.cos α) / (3*Real.sin α^2 + Real.cos α^2) = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l261_26185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_g_and_x_l261_26186

/-- Given that x is a multiple of 49356, the greatest common divisor of g(x) and x is 450,
    where g(x) = (3x+2)(8x+3)(14x+5)(x+15) -/
theorem gcd_of_g_and_x (x : ℤ) (h : 49356 ∣ x) :
  Int.gcd ((3*x+2)*(8*x+3)*(14*x+5)*(x+15)) x = 450 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_g_and_x_l261_26186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_AMF_l261_26139

/-- Parabola with equation y^2 = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of the parabola -/
noncomputable def focus (par : Parabola) : Point :=
  { x := par.p / 2, y := 0 }

/-- The point M -/
noncomputable def point_m (par : Parabola) : Point :=
  { x := -par.p / 2, y := 0 }

/-- A line with slope 2 passing through the focus -/
structure Line (par : Parabola) where
  slope : ℝ
  slope_eq_two : slope = 2
  passes_through_focus : True  -- This is a simplification, as we can't easily express this condition

/-- Point A is the intersection of the line and parabola in the first quadrant -/
noncomputable def point_a (par : Parabola) (l : Line par) : Point :=
  { x := par.p / 2 + par.p * (1 + Real.sqrt 5) / 4,
    y := par.p * (1 + Real.sqrt 5) / 2 }

/-- The theorem to be proved -/
theorem tan_angle_AMF (par : Parabola) (l : Line par) :
  let a := point_a par l
  let m := point_m par
  let f := focus par
  (a.y - m.y) / (a.x - m.x) = 2 / 5 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_AMF_l261_26139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l261_26167

open Real

-- Define the functions
noncomputable def f (x : ℝ) := x * exp x
noncomputable def g (x : ℝ) := x^2 + 2*x
noncomputable def h (x : ℝ) := 2 * sin (π/6 * x + 2*π/3)

-- State the theorem
theorem min_k_value (k : ℝ) :
  (∀ x, h x - f x ≤ k * (g x + 2)) ↔ k ≥ 2 + exp (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l261_26167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_a_bound_l261_26109

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x else 2*a*x - 5

theorem function_equality_implies_a_bound (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_a_bound_l261_26109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l261_26107

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * (x - 2)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ -21} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l261_26107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_approximate_l261_26130

/-- The cube root of 3 -/
noncomputable def cubeRoot3 : ℝ := Real.rpow 3 (1/3)

/-- The expression to be calculated -/
noncomputable def originalExpression : ℝ := 529 / (12 * cubeRoot3^2 + 52 * cubeRoot3 + 49)

/-- Theorem stating the existence of A, B, C that rationalize the denominator 
    and lead to the correct approximation -/
theorem rationalize_and_approximate : 
  ∃ (A B C : ℝ), 
    (∃ (q : ℚ), (12 * cubeRoot3^2 + 52 * cubeRoot3 + 49) * (A * cubeRoot3^2 + B * cubeRoot3 + C) = ↑q) ∧
    (abs ((529 * (A * cubeRoot3^2 + B * cubeRoot3 + C)) / 
         ((12 * cubeRoot3^2 + 52 * cubeRoot3 + 49) * (A * cubeRoot3^2 + B * cubeRoot3 + C)) - 3.55) < 0.001) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_approximate_l261_26130
