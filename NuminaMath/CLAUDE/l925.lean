import Mathlib

namespace NUMINAMATH_CALUDE_f_range_l925_92566

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.Icc (-2 : ℝ) 2, ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = y ∧
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_f_range_l925_92566


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l925_92507

theorem right_triangle_third_side_product (a b c d : ℝ) : 
  a = 6 → b = 8 → 
  (a^2 + b^2 = c^2 ∨ a^2 + d^2 = b^2) →
  c * d = 20 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l925_92507


namespace NUMINAMATH_CALUDE_coin_value_ratio_l925_92510

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of nickels -/
def num_nickels : ℕ := 4

/-- The number of dimes -/
def num_dimes : ℕ := 6

/-- The number of quarters -/
def num_quarters : ℕ := 2

theorem coin_value_ratio :
  ∃ (k : ℕ), k > 0 ∧
    num_nickels * nickel_value = 2 * k ∧
    num_dimes * dime_value = 6 * k ∧
    num_quarters * quarter_value = 5 * k :=
sorry

end NUMINAMATH_CALUDE_coin_value_ratio_l925_92510


namespace NUMINAMATH_CALUDE_sine_function_period_l925_92539

theorem sine_function_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x ∈ Set.Icc (-π) (5*π), ∃ y, y = a * Real.sin (b * x + c) + d) →
  (∃ n : ℕ, n = 5 ∧ (6*π) / n = (2*π) / b) →
  b = 5/3 := by
sorry

end NUMINAMATH_CALUDE_sine_function_period_l925_92539


namespace NUMINAMATH_CALUDE_bird_stork_difference_l925_92572

theorem bird_stork_difference : 
  ∀ (initial_storks initial_birds joining_birds : ℕ),
    initial_storks = 5 →
    initial_birds = 3 →
    joining_birds = 4 →
    (initial_birds + joining_birds) - initial_storks = 2 := by
  sorry

end NUMINAMATH_CALUDE_bird_stork_difference_l925_92572


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l925_92571

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (1 + 3*I) / (3 + I)
  0 < z.re ∧ 0 < z.im :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l925_92571


namespace NUMINAMATH_CALUDE_minimum_k_value_l925_92597

theorem minimum_k_value (k : ℝ) : 
  (∀ x y : ℝ, Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (x + y)) → 
  k ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_k_value_l925_92597


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_range_l925_92504

def integer_range : List ℤ := List.range 12 |>.map (λ i => i - 5)

theorem arithmetic_mean_of_range : (integer_range.sum : ℚ) / integer_range.length = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_range_l925_92504


namespace NUMINAMATH_CALUDE_skee_ball_tickets_count_l925_92592

/-- The number of tickets Tom won playing 'skee ball' -/
def skee_ball_tickets : ℕ := sorry

/-- The number of tickets Tom won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- The number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- The number of tickets Tom has left -/
def remaining_tickets : ℕ := 50

theorem skee_ball_tickets_count : skee_ball_tickets = 25 := by
  sorry

end NUMINAMATH_CALUDE_skee_ball_tickets_count_l925_92592


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l925_92503

/-- Given two polynomials in d with coefficients g and h, prove their sum equals 15.5 -/
theorem polynomial_product_sum (g h : ℚ) : 
  (∀ d : ℚ, (8*d^2 - 4*d + g) * (5*d^2 + h*d - 10) = 40*d^4 - 75*d^3 - 90*d^2 + 5*d + 20) →
  g + h = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l925_92503


namespace NUMINAMATH_CALUDE_angle_complement_measure_l925_92524

theorem angle_complement_measure : 
  ∀ x : ℝ, 
  (x + (3 * x + 10) = 90) →  -- Condition 1 and 2 combined
  (3 * x + 10 = 70) :=        -- The complement measure to prove
by
  sorry

end NUMINAMATH_CALUDE_angle_complement_measure_l925_92524


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l925_92517

/-- The number of ways to arrange the digits 3, 0, 5, 7, 0 into a 5-digit number -/
def digit_arrangements : ℕ :=
  let digits : Multiset ℕ := {3, 0, 5, 7, 0}
  let total_arrangements := Nat.factorial 5 / (Nat.factorial 2)  -- Total permutations with repetition
  let arrangements_starting_with_zero := Nat.factorial 4 / (Nat.factorial 2)  -- Arrangements starting with 0
  total_arrangements - arrangements_starting_with_zero

/-- The theorem stating that the number of valid arrangements is 48 -/
theorem valid_arrangements_count : digit_arrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l925_92517


namespace NUMINAMATH_CALUDE_triangle_problem_l925_92546

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  (2 * (Real.cos (A / 2))^2 + (Real.cos B - Real.sqrt 3 * Real.sin B) * Real.cos C = 1) →
  (c = 2) →
  (1/2 * a * b * Real.sin C = Real.sqrt 3) →
  -- Conclusions to prove
  (C = π/3) ∧ (a = 2) ∧ (b = 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l925_92546


namespace NUMINAMATH_CALUDE_building_height_percentage_l925_92583

theorem building_height_percentage (L M R : ℝ) : 
  M = 100 → 
  R = L + M - 20 → 
  L + M + R = 340 → 
  L / M * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_building_height_percentage_l925_92583


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_coord_l925_92568

/-- Given two vectors a and b in ℝ², prove that if a is perpendicular to b,
    then the x-coordinate of a is -2/3. -/
theorem perpendicular_vectors_x_coord
  (a b : ℝ × ℝ)
  (h1 : a.1 = x ∧ a.2 = x + 1)
  (h2 : b = (1, 2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_coord_l925_92568


namespace NUMINAMATH_CALUDE_square_root_squared_l925_92579

theorem square_root_squared (x : ℝ) (hx : x = 49) : (Real.sqrt x)^2 = x := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_l925_92579


namespace NUMINAMATH_CALUDE_right_triangles_with_increasing_sides_l925_92560

theorem right_triangles_with_increasing_sides (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pyth1 : a^2 + (b-100)^2 = (c-30)^2)
  (h_pyth2 : a^2 + b^2 = c^2)
  (h_pyth3 : a^2 + (b+100)^2 = (c+40)^2) :
  a = 819 ∧ b = 308 ∧ c = 875 := by
  sorry

end NUMINAMATH_CALUDE_right_triangles_with_increasing_sides_l925_92560


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l925_92536

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 4 + a 5 + a 6 = 168 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l925_92536


namespace NUMINAMATH_CALUDE_lily_remaining_milk_l925_92538

/-- Calculates the remaining milk after giving some away -/
def remaining_milk (initial : ℚ) (given_away : ℚ) : ℚ :=
  initial - given_away

/-- Proves that Lily has 17/7 gallons of milk left -/
theorem lily_remaining_milk :
  let initial_milk : ℚ := 5
  let given_to_james : ℚ := 18 / 7
  remaining_milk initial_milk given_to_james = 17 / 7 := by
  sorry

end NUMINAMATH_CALUDE_lily_remaining_milk_l925_92538


namespace NUMINAMATH_CALUDE_square_area_ratio_l925_92557

theorem square_area_ratio (p1 p2 : ℕ) (h1 : p1 = 32) (h2 : p2 = 20) : 
  (p1 / 4) ^ 2 / (p2 / 4) ^ 2 = 64 / 25 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l925_92557


namespace NUMINAMATH_CALUDE_light_off_after_odd_presses_l925_92500

def LightSwitch : Type := Bool

def press (state : LightSwitch) : LightSwitch :=
  !state

def press_n_times (state : LightSwitch) (n : ℕ) : LightSwitch :=
  match n with
  | 0 => state
  | m + 1 => press (press_n_times state m)

theorem light_off_after_odd_presses (n : ℕ) (h : Odd n) :
  press_n_times true n = false :=
sorry

end NUMINAMATH_CALUDE_light_off_after_odd_presses_l925_92500


namespace NUMINAMATH_CALUDE_sum_zero_implies_product_sum_nonpositive_l925_92526

theorem sum_zero_implies_product_sum_nonpositive
  (a b c : ℝ) (h : a + b + c = 0) :
  a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_implies_product_sum_nonpositive_l925_92526


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l925_92544

theorem complex_magnitude_problem (z : ℂ) (h : (1 - I) / I * z = 1) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l925_92544


namespace NUMINAMATH_CALUDE_total_yellow_balls_is_30_l925_92501

/-- The number of boxes containing balls -/
def num_boxes : ℕ := 6

/-- The number of yellow balls in each box -/
def yellow_balls_per_box : ℕ := 5

/-- The total number of yellow balls across all boxes -/
def total_yellow_balls : ℕ := num_boxes * yellow_balls_per_box

theorem total_yellow_balls_is_30 : total_yellow_balls = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_yellow_balls_is_30_l925_92501


namespace NUMINAMATH_CALUDE_sqrt_529_squared_l925_92530

theorem sqrt_529_squared : (Real.sqrt 529)^2 = 529 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_529_squared_l925_92530


namespace NUMINAMATH_CALUDE_max_books_borrowed_l925_92543

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℕ) (h1 : total_students = 20) (h2 : zero_books = 3) (h3 : one_book = 9) 
  (h4 : two_books = 4) (h5 : avg_books = 2) : 
  ∃ (max_books : ℕ), max_books = 14 ∧ 
  ∀ (student_books : ℕ), student_books ≤ max_books ∧
  (zero_books * 0 + one_book * 1 + two_books * 2 + 
   (total_students - zero_books - one_book - two_books) * 3 + 
   (max_books - 3) ≤ total_students * avg_books) :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l925_92543


namespace NUMINAMATH_CALUDE_winning_scores_count_l925_92552

-- Define the number of teams and runners per team
def num_teams : Nat := 3
def runners_per_team : Nat := 3

-- Define the total number of runners
def total_runners : Nat := num_teams * runners_per_team

-- Define the sum of all positions
def total_points : Nat := (total_runners * (total_runners + 1)) / 2

-- Define the maximum possible winning score
def max_winning_score : Nat := total_points / 2

-- Define the minimum possible winning score
def min_winning_score : Nat := 1 + 2 + 3

-- Theorem statement
theorem winning_scores_count :
  (∃ (winning_scores : Finset Nat),
    (∀ s ∈ winning_scores, min_winning_score ≤ s ∧ s ≤ max_winning_score) ∧
    (∀ s ∈ winning_scores, ∃ (a b c : Nat),
      a < b ∧ b < c ∧ c ≤ total_runners ∧ s = a + b + c) ∧
    winning_scores.card = 17) :=
by sorry

end NUMINAMATH_CALUDE_winning_scores_count_l925_92552


namespace NUMINAMATH_CALUDE_probability_no_three_consecutive_as_l925_92535

/-- A string of length 6 using symbols A, B, and C -/
def String6ABC := Fin 6 → Fin 3

/-- Check if a string contains three consecutive A's -/
def hasThreeConsecutiveAs (s : String6ABC) : Prop :=
  ∃ i : Fin 4, s i = 0 ∧ s (i + 1) = 0 ∧ s (i + 2) = 0

/-- The total number of possible strings -/
def totalStrings : ℕ := 3^6

/-- The number of strings without three consecutive A's -/
def stringsWithoutThreeAs : ℕ := 680

/-- The probability of a random string not having three consecutive A's -/
def probabilityNoThreeAs : ℚ := stringsWithoutThreeAs / totalStrings

theorem probability_no_three_consecutive_as :
  probabilityNoThreeAs = 680 / 729 :=
sorry

end NUMINAMATH_CALUDE_probability_no_three_consecutive_as_l925_92535


namespace NUMINAMATH_CALUDE_certain_number_problem_l925_92598

theorem certain_number_problem (n x : ℝ) (h1 : 4 / (n + 3 / x) = 1) (h2 : x = 1) : n = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l925_92598


namespace NUMINAMATH_CALUDE_line_L_equation_trajectory_Q_equation_l925_92515

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line L
def LineL (x y : ℝ) : Prop := x = 1 ∨ 3*x - 4*y + 5 = 0

-- Define the trajectory of Q
def TrajectoryQ (x y : ℝ) : Prop := x^2/4 + y^2/16 = 1

-- Theorem for Part I
theorem line_L_equation : 
  ∃ (A B : ℝ × ℝ), 
  Circle A.1 A.2 ∧ Circle B.1 B.2 ∧
  LineL A.1 A.2 ∧ LineL B.1 B.2 ∧
  LineL 1 2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
sorry

-- Theorem for Part II
theorem trajectory_Q_equation :
  ∀ (M : ℝ × ℝ), Circle M.1 M.2 →
  ∃ (Q : ℝ × ℝ), 
  Q.1 = M.1 ∧ Q.2 = 2 * M.2 ∧
  TrajectoryQ Q.1 Q.2 :=
sorry

end NUMINAMATH_CALUDE_line_L_equation_trajectory_Q_equation_l925_92515


namespace NUMINAMATH_CALUDE_new_individuals_weight_l925_92542

/-- The total weight of three new individuals joining a group, given specific conditions -/
theorem new_individuals_weight (W : ℝ) : 
  let initial_group_size : ℕ := 10
  let leaving_weights : List ℝ := [75, 80, 90]
  let average_weight_increase : ℝ := 6.5
  let new_individuals_count : ℕ := 3
  W - (initial_group_size : ℝ) * average_weight_increase = 
    (W - leaving_weights.sum) + (new_individuals_count : ℝ) * average_weight_increase →
  (∃ X : ℝ, X = (new_individuals_count : ℝ) * average_weight_increase ∧ X = 65) := by
sorry


end NUMINAMATH_CALUDE_new_individuals_weight_l925_92542


namespace NUMINAMATH_CALUDE_a_minus_2ab_plus_b_eq_zero_l925_92523

theorem a_minus_2ab_plus_b_eq_zero 
  (a b : ℝ) 
  (h1 : a + b = 2) 
  (h2 : a * b = 1) : 
  a - 2 * a * b + b = 0 := by
sorry

end NUMINAMATH_CALUDE_a_minus_2ab_plus_b_eq_zero_l925_92523


namespace NUMINAMATH_CALUDE_cannot_tile_modified_checkerboard_l925_92550

/-- Represents a checkerboard with two opposite corners removed -/
structure ModifiedCheckerboard :=
  (size : Nat)
  (cornersRemoved : Nat)

/-- Represents a domino used for tiling -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Defines the property of a checkerboard being tileable by dominoes -/
def is_tileable (board : ModifiedCheckerboard) (domino : Domino) : Prop :=
  ∃ (tiling : Nat), tiling > 0

/-- The main theorem stating that an 8x8 checkerboard with opposite corners removed cannot be tiled by 2x1 dominoes -/
theorem cannot_tile_modified_checkerboard :
  ¬ is_tileable (ModifiedCheckerboard.mk 8 2) (Domino.mk 2 1) := by
  sorry

end NUMINAMATH_CALUDE_cannot_tile_modified_checkerboard_l925_92550


namespace NUMINAMATH_CALUDE_same_value_point_m_two_distinct_same_value_points_l925_92531

/-- Quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + m

theorem same_value_point_m (m : ℝ) :
  f m 2 = 2 → m = -8 := by sorry

theorem two_distinct_same_value_points (m : ℝ) (a b : ℝ) :
  (∃ (a b : ℝ), a < 1 ∧ 1 < b ∧ f m a = a ∧ f m b = b) →
  m < -3 := by sorry

end NUMINAMATH_CALUDE_same_value_point_m_two_distinct_same_value_points_l925_92531


namespace NUMINAMATH_CALUDE_otimes_inequality_implies_a_unrestricted_l925_92591

/-- Custom operation ⊗ defined on real numbers -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating that if (x-a) ⊗ (x+a) < 1 holds for all real x, then a can be any real number -/
theorem otimes_inequality_implies_a_unrestricted :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → a ∈ Set.univ :=
by sorry

end NUMINAMATH_CALUDE_otimes_inequality_implies_a_unrestricted_l925_92591


namespace NUMINAMATH_CALUDE_perpendicular_slope_l925_92559

theorem perpendicular_slope (x y : ℝ) :
  let original_line := {(x, y) | 4 * x - 6 * y = 12}
  let original_slope := 2 / 3
  let perpendicular_slope := -1 / original_slope
  perpendicular_slope = -3 / 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l925_92559


namespace NUMINAMATH_CALUDE_triangle_longest_side_l925_92525

theorem triangle_longest_side (x : ℚ) : 
  (x + 3 : ℚ) + (2 * x - 1 : ℚ) + (3 * x + 5 : ℚ) = 45 → 
  max (x + 3) (max (2 * x - 1) (3 * x + 5)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l925_92525


namespace NUMINAMATH_CALUDE_polynomial_equality_main_result_l925_92511

def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let b₁ := 0
  let b₂ := -3
  let b₃ := 4
  let b₄ := -1
  (b₁, b₂, b₃, b₄)

theorem polynomial_equality (x : ℝ) (a₁ a₂ a₃ a₄ : ℝ) (b₁ b₂ b₃ b₄ : ℝ) :
  x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄ = (x+1)^4 + b₁*(x+1)^3 + b₂*(x+1)^2 + b₃*(x+1) + b₄ →
  f a₁ a₂ a₃ a₄ = (b₁, b₂, b₃, b₄) :=
by sorry

theorem main_result : f 4 3 2 1 = (0, -3, 4, -1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_main_result_l925_92511


namespace NUMINAMATH_CALUDE_investment_income_l925_92574

/-- Proves that an investment of $6800 in a 60% stock at a price of 136 yields an annual income of $3000 -/
theorem investment_income (investment : ℝ) (stock_percentage : ℝ) (stock_price : ℝ) (annual_income : ℝ) : 
  investment = 6800 ∧ 
  stock_percentage = 0.60 ∧ 
  stock_price = 136 ∧ 
  annual_income = 3000 → 
  investment * (stock_percentage / stock_price) = annual_income :=
by sorry

end NUMINAMATH_CALUDE_investment_income_l925_92574


namespace NUMINAMATH_CALUDE_equation_solution_l925_92541

theorem equation_solution (x : ℚ) :
  (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l925_92541


namespace NUMINAMATH_CALUDE_givenPointInSecondQuadrant_l925_92580

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point we want to prove is in the second quadrant -/
def givenPoint : Point :=
  { x := -1, y := 2 }

/-- Theorem stating that the given point is in the second quadrant -/
theorem givenPointInSecondQuadrant : isInSecondQuadrant givenPoint := by
  sorry

end NUMINAMATH_CALUDE_givenPointInSecondQuadrant_l925_92580


namespace NUMINAMATH_CALUDE_geometric_sequence_and_max_value_l925_92521

/-- Given real numbers a, b, c, and d forming a geometric sequence, and a function
    y = ln x - x attaining its maximum value c when x = b, prove that ad = -1 -/
theorem geometric_sequence_and_max_value (a b c d : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, x > 0 → Real.log x - x ≤ c) →        -- maximum value condition
  (Real.log b - b = c) →                         -- attains maximum at x = b
  a * d = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_max_value_l925_92521


namespace NUMINAMATH_CALUDE_existsNonSymmetricalEqualTriangles_l925_92577

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents an ellipse -/
structure Ellipse :=
  (center : Point)
  (semiMajorAxis : ℝ)
  (semiMinorAxis : ℝ)

/-- Checks if a point is inside or on the ellipse -/
def isPointInEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.semiMajorAxis^2 + (p.y - e.center.y)^2 / e.semiMinorAxis^2 ≤ 1

/-- Checks if a triangle is inscribed in an ellipse -/
def isTriangleInscribed (t : Triangle) (e : Ellipse) : Prop :=
  isPointInEllipse t.a e ∧ isPointInEllipse t.b e ∧ isPointInEllipse t.c e

/-- Checks if two triangles are equal -/
def areTrianglesEqual (t1 t2 : Triangle) : Prop :=
  -- Definition of triangle equality (e.g., same side lengths)
  sorry

/-- Checks if two triangles are symmetrical with respect to the x-axis -/
def areTrianglesSymmetricalXAxis (t1 t2 : Triangle) : Prop :=
  -- Definition of symmetry with respect to x-axis
  sorry

/-- Checks if two triangles are symmetrical with respect to the y-axis -/
def areTrianglesSymmetricalYAxis (t1 t2 : Triangle) : Prop :=
  -- Definition of symmetry with respect to y-axis
  sorry

/-- Checks if two triangles are symmetrical with respect to the center -/
def areTrianglesSymmetricalCenter (t1 t2 : Triangle) (e : Ellipse) : Prop :=
  -- Definition of symmetry with respect to center
  sorry

/-- Main theorem: There exist two equal triangles inscribed in an ellipse that are not symmetrical -/
theorem existsNonSymmetricalEqualTriangles :
  ∃ (e : Ellipse) (t1 t2 : Triangle),
    isTriangleInscribed t1 e ∧
    isTriangleInscribed t2 e ∧
    areTrianglesEqual t1 t2 ∧
    ¬(areTrianglesSymmetricalXAxis t1 t2 ∨
      areTrianglesSymmetricalYAxis t1 t2 ∨
      areTrianglesSymmetricalCenter t1 t2 e) :=
by
  sorry

end NUMINAMATH_CALUDE_existsNonSymmetricalEqualTriangles_l925_92577


namespace NUMINAMATH_CALUDE_dexter_sam_same_team_l925_92547

/-- The number of students in the dodgeball league -/
def total_students : ℕ := 12

/-- The number of players in each team -/
def team_size : ℕ := 6

/-- The number of students not including Dexter and Sam -/
def other_students : ℕ := total_students - 2

/-- The number of additional players needed to form a team with Dexter and Sam -/
def additional_players : ℕ := team_size - 2

theorem dexter_sam_same_team :
  (Nat.choose other_students additional_players) = 210 :=
sorry

end NUMINAMATH_CALUDE_dexter_sam_same_team_l925_92547


namespace NUMINAMATH_CALUDE_simplify_expression_l925_92569

theorem simplify_expression : (1 / ((-8^4)^2)) * (-8)^11 = -512 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l925_92569


namespace NUMINAMATH_CALUDE_male_cattle_percentage_l925_92551

/-- Represents the farmer's cattle statistics -/
structure CattleStats where
  total_milk : ℕ
  milk_per_cow : ℕ
  male_count : ℕ

/-- Calculates the percentage of male cattle -/
def male_percentage (stats : CattleStats) : ℚ :=
  let female_count := stats.total_milk / stats.milk_per_cow
  let total_cattle := stats.male_count + female_count
  (stats.male_count : ℚ) / (total_cattle : ℚ) * 100

/-- Theorem stating that the percentage of male cattle is 40% -/
theorem male_cattle_percentage (stats : CattleStats) 
  (h1 : stats.total_milk = 150)
  (h2 : stats.milk_per_cow = 2)
  (h3 : stats.male_count = 50) :
  male_percentage stats = 40 := by
  sorry

#eval male_percentage { total_milk := 150, milk_per_cow := 2, male_count := 50 }

end NUMINAMATH_CALUDE_male_cattle_percentage_l925_92551


namespace NUMINAMATH_CALUDE_number_problem_l925_92529

theorem number_problem (N : ℝ) : 
  (1/8 : ℝ) * (3/5 : ℝ) * (4/7 : ℝ) * (5/11 : ℝ) * N - (1/9 : ℝ) * (2/3 : ℝ) * (3/4 : ℝ) * (5/8 : ℝ) * N = 30 → 
  (75/100 : ℝ) * N = -1476 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l925_92529


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l925_92556

theorem arithmetic_calculation : 2 + 5 * 4 - 6 + 3 = 19 := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l925_92556


namespace NUMINAMATH_CALUDE_min_value_quadratic_l925_92540

theorem min_value_quadratic (x y : ℝ) : 
  3 ≤ 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l925_92540


namespace NUMINAMATH_CALUDE_transform_f1_to_f2_l925_92505

/-- Represents a quadratic function of the form y = a(x - h)^2 + k -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Applies a horizontal and vertical translation to a quadratic function -/
def translate (f : QuadraticFunction) (dx dy : ℝ) : QuadraticFunction :=
  { a := f.a
  , h := f.h - dx
  , k := f.k - dy }

/-- The original quadratic function y = -2(x - 1)^2 + 3 -/
def f1 : QuadraticFunction :=
  { a := -2
  , h := 1
  , k := 3 }

/-- The target quadratic function y = -2x^2 -/
def f2 : QuadraticFunction :=
  { a := -2
  , h := 0
  , k := 0 }

/-- Theorem stating that translating f1 by 1 unit left and 3 units down results in f2 -/
theorem transform_f1_to_f2 : translate f1 1 3 = f2 := by sorry

end NUMINAMATH_CALUDE_transform_f1_to_f2_l925_92505


namespace NUMINAMATH_CALUDE_multiples_of_10_average_l925_92582

theorem multiples_of_10_average : 
  let first := 10
  let last := 600
  let step := 10
  let count := (last - first) / step + 1
  let sum := count * (first + last) / 2
  sum / count = 305 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_10_average_l925_92582


namespace NUMINAMATH_CALUDE_y_greater_than_x_l925_92573

theorem y_greater_than_x (x y : ℝ) (h1 : x + y > 2*x) (h2 : x - y < 2*y) : y > x := by
  sorry

end NUMINAMATH_CALUDE_y_greater_than_x_l925_92573


namespace NUMINAMATH_CALUDE_subtract_negative_negative_two_minus_five_l925_92509

theorem subtract_negative (a b : ℤ) : a - b = a + (-b) := by sorry

theorem negative_two_minus_five : (-2 : ℤ) - 5 = -7 := by sorry

end NUMINAMATH_CALUDE_subtract_negative_negative_two_minus_five_l925_92509


namespace NUMINAMATH_CALUDE_bounded_sequence_with_lcm_condition_l925_92518

theorem bounded_sequence_with_lcm_condition (n : ℕ) (k : ℕ) (a : Fin k → ℕ) :
  (∀ i : Fin k, 1 ≤ a i) →
  (∀ i j : Fin k, i < j → a i < a j) →
  (∀ i : Fin k, a i ≤ n) →
  (∀ i j : Fin k, Nat.lcm (a i) (a j) ≤ n) →
  k ≤ 2 * Int.floor (Real.sqrt n) :=
by sorry

end NUMINAMATH_CALUDE_bounded_sequence_with_lcm_condition_l925_92518


namespace NUMINAMATH_CALUDE_simplify_expression_calculate_expression_l925_92520

-- Part 1
theorem simplify_expression (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (2 * a^(3/2) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4 * a^(11/6) := by
  sorry

-- Part 2
theorem calculate_expression :
  (2^(1/3) * 3^(1/2))^6 + (2^(1/2) * 2^(1/4))^(4/3) - 2^(1/4) * 8^(1/4) - (-2005)^0 = 100 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_calculate_expression_l925_92520


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l925_92567

theorem square_sum_given_conditions (a b c : ℝ) 
  (h1 : a * b + b * c + c * a = 4)
  (h2 : a + b + c = 17) : 
  a^2 + b^2 + c^2 = 281 := by sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l925_92567


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l925_92554

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = 0.0094 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l925_92554


namespace NUMINAMATH_CALUDE_max_rectangles_is_k_times_l_l925_92532

/-- A partition of a square into rectangles -/
structure SquarePartition where
  k : ℕ  -- number of rectangles intersected by a vertical line
  l : ℕ  -- number of rectangles intersected by a horizontal line
  no_interior_intersections : Bool  -- no two segments intersect at an interior point
  no_collinear_segments : Bool  -- no two segments lie on the same line

/-- The number of rectangles in a square partition -/
def num_rectangles (p : SquarePartition) : ℕ := sorry

/-- The maximum number of rectangles in any valid square partition -/
def max_rectangles (p : SquarePartition) : ℕ := p.k * p.l

/-- Theorem: The maximum number of rectangles in a valid square partition is k * l -/
theorem max_rectangles_is_k_times_l (p : SquarePartition) 
  (h1 : p.no_interior_intersections = true) 
  (h2 : p.no_collinear_segments = true) : 
  num_rectangles p ≤ max_rectangles p := by sorry

end NUMINAMATH_CALUDE_max_rectangles_is_k_times_l_l925_92532


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l925_92599

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n

theorem minimum_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  GeometricSequence a →
  a 3 = a 2 + 2 * a 1 →
  a m * a n = 64 * (a 1)^2 →
  (∀ k l : ℕ, a k * a l = 64 * (a 1)^2 → 1 / k + 9 / l ≥ 1 / m + 9 / n) →
  1 / m + 9 / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l925_92599


namespace NUMINAMATH_CALUDE_average_MTWT_is_48_l925_92562

/-- The average temperature for some days -/
def average_some_days : ℝ := 48

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday -/
def average_TWTF : ℝ := 46

/-- The temperature on Monday -/
def temp_Monday : ℝ := 42

/-- The temperature on Friday -/
def temp_Friday : ℝ := 34

/-- The number of days in the TWTF group -/
def num_days_TWTF : ℕ := 4

/-- The number of days in the MTWT group -/
def num_days_MTWT : ℕ := 4

/-- Theorem: The average temperature for Monday, Tuesday, Wednesday, and Thursday is 48 degrees -/
theorem average_MTWT_is_48 : 
  (temp_Monday + (average_TWTF * num_days_TWTF - temp_Friday)) / num_days_MTWT = 48 := by
  sorry

end NUMINAMATH_CALUDE_average_MTWT_is_48_l925_92562


namespace NUMINAMATH_CALUDE_license_plate_count_l925_92545

/-- The number of consonants in the English alphabet -/
def num_consonants : ℕ := 20

/-- The number of vowels in the English alphabet (including Y) -/
def num_vowels : ℕ := 6

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits

theorem license_plate_count : total_plates = 24000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l925_92545


namespace NUMINAMATH_CALUDE_football_shape_area_l925_92589

/-- The area of the football-shaped region formed by two circular sectors -/
theorem football_shape_area 
  (r1 : ℝ) 
  (r2 : ℝ) 
  (h1 : r1 = 2 * Real.sqrt 2) 
  (h2 : r2 = 2) 
  (θ : ℝ) 
  (h3 : θ = π / 2) : 
  (θ / (2 * π)) * π * r1^2 - (θ / (2 * π)) * π * r2^2 = π := by
  sorry

end NUMINAMATH_CALUDE_football_shape_area_l925_92589


namespace NUMINAMATH_CALUDE_ten_boys_handshakes_l925_92584

/-- The number of handshakes in a group of boys with special conditions -/
def specialHandshakes (n : ℕ) : ℕ :=
  n * (n - 1) / 2 - 2

/-- Theorem: In a group of 10 boys with the given handshake conditions, 
    the total number of handshakes is 43 -/
theorem ten_boys_handshakes : specialHandshakes 10 = 43 := by
  sorry

end NUMINAMATH_CALUDE_ten_boys_handshakes_l925_92584


namespace NUMINAMATH_CALUDE_quadratic_root_two_l925_92578

theorem quadratic_root_two (c : ℝ) : (2 : ℝ)^2 = c → c = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_two_l925_92578


namespace NUMINAMATH_CALUDE_clara_weight_l925_92587

/-- Given two weights satisfying certain conditions, prove that one of them is 88 pounds. -/
theorem clara_weight (alice_weight clara_weight : ℝ) 
  (h1 : alice_weight + clara_weight = 220)
  (h2 : clara_weight - alice_weight = clara_weight / 3) : 
  clara_weight = 88 := by
  sorry

end NUMINAMATH_CALUDE_clara_weight_l925_92587


namespace NUMINAMATH_CALUDE_compare_squares_l925_92596

theorem compare_squares (a : ℝ) : (a + 1)^2 > a^2 + 2*a := by
  sorry

end NUMINAMATH_CALUDE_compare_squares_l925_92596


namespace NUMINAMATH_CALUDE_trivia_game_points_per_question_l925_92502

/-- Given a trivia game where a player answers questions correctly and receives a total score,
    this theorem proves that if the player answers 10 questions correctly and scores 50 points,
    then each question is worth 5 points. -/
theorem trivia_game_points_per_question 
  (total_questions : ℕ) 
  (total_score : ℕ) 
  (points_per_question : ℕ) 
  (h1 : total_questions = 10) 
  (h2 : total_score = 50) : 
  points_per_question = 5 := by
  sorry

#check trivia_game_points_per_question

end NUMINAMATH_CALUDE_trivia_game_points_per_question_l925_92502


namespace NUMINAMATH_CALUDE_nine_candidates_l925_92514

/- Define the number of ways to select president and vice president -/
def selection_ways : ℕ := 72

/- Define the property that determines the number of candidates -/
def candidate_count (n : ℕ) : Prop :=
  n * (n - 1) = selection_ways

/- Theorem statement -/
theorem nine_candidates : 
  ∃ (n : ℕ), candidate_count n ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_nine_candidates_l925_92514


namespace NUMINAMATH_CALUDE_polynomial_equality_l925_92561

/-- A polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Evaluate a polynomial at a given point -/
def eval (p : IntPolynomial) (x : Int) : Int :=
  p.foldr (fun a acc => a + x * acc) 0

/-- Get the maximum absolute value of coefficients in a polynomial -/
def maxAbsCoeff (p : IntPolynomial) : Int :=
  p.foldl (fun acc a => max acc (Int.natAbs a)) 0

theorem polynomial_equality (f g : IntPolynomial) :
  (∃ t : Int, eval f t = eval g t ∧ t > 2 * max (maxAbsCoeff f) (maxAbsCoeff g)) ↔ f = g := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l925_92561


namespace NUMINAMATH_CALUDE_exists_alpha_for_sequence_l925_92576

/-- A sequence of non-zero real numbers satisfying the given condition -/
def SequenceA (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n ≠ 0 ∧ a n ^ 2 - a (n - 1) * a (n + 1) = 1

/-- The theorem to be proved -/
theorem exists_alpha_for_sequence (a : ℕ → ℝ) (h : SequenceA a) :
  ∃ α : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = α * a n - a (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_alpha_for_sequence_l925_92576


namespace NUMINAMATH_CALUDE_smallest_integer_with_gcd_lcm_constraint_l925_92553

theorem smallest_integer_with_gcd_lcm_constraint (x : ℕ) (m n : ℕ) 
  (h1 : x > 0)
  (h2 : m = 30)
  (h3 : Nat.gcd m n = x + 3)
  (h4 : Nat.lcm m n = x * (x + 3)) :
  n ≥ 162 ∧ ∃ (x : ℕ), x > 0 ∧ 
    Nat.gcd 30 162 = x + 3 ∧ 
    Nat.lcm 30 162 = x * (x + 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_gcd_lcm_constraint_l925_92553


namespace NUMINAMATH_CALUDE_max_tickets_buyable_l925_92537

def ticket_price : ℚ := 15
def available_money : ℚ := 120

theorem max_tickets_buyable : 
  ∀ n : ℕ, (n : ℚ) * ticket_price ≤ available_money ↔ n ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_tickets_buyable_l925_92537


namespace NUMINAMATH_CALUDE_range_of_m_l925_92516

theorem range_of_m (m : ℝ) : 
  Real.sqrt (2 * m + 1) > Real.sqrt (m^2 + m - 1) → 
  m ∈ Set.Ici ((Real.sqrt 5 - 1) / 2) ∩ Set.Iio 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l925_92516


namespace NUMINAMATH_CALUDE_amusement_park_total_cost_l925_92563

/-- The total cost of an amusement park trip for a group of children -/
def amusement_park_cost (num_children : ℕ) (ferris_wheel_riders : ℕ) (ferris_wheel_cost : ℕ) 
  (merry_go_round_cost : ℕ) (ice_cream_cones_per_child : ℕ) (ice_cream_cost : ℕ) : ℕ :=
  ferris_wheel_riders * ferris_wheel_cost + 
  num_children * merry_go_round_cost + 
  num_children * ice_cream_cones_per_child * ice_cream_cost

/-- Theorem stating the total cost for the given scenario -/
theorem amusement_park_total_cost : 
  amusement_park_cost 5 3 5 3 2 8 = 110 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_total_cost_l925_92563


namespace NUMINAMATH_CALUDE_drawings_on_last_page_l925_92528

-- Define the given conditions
def initial_notebooks : ℕ := 10
def pages_per_notebook : ℕ := 30
def initial_drawings_per_page : ℕ := 4
def new_drawings_per_page : ℕ := 8
def filled_notebooks : ℕ := 6
def filled_pages_in_seventh : ℕ := 25

-- Define the theorem
theorem drawings_on_last_page : 
  let total_drawings := initial_notebooks * pages_per_notebook * initial_drawings_per_page
  let full_pages := total_drawings / new_drawings_per_page
  let pages_in_complete_notebooks := filled_notebooks * pages_per_notebook
  let remaining_drawings := total_drawings - (full_pages * new_drawings_per_page)
  remaining_drawings = 0 := by
  sorry

end NUMINAMATH_CALUDE_drawings_on_last_page_l925_92528


namespace NUMINAMATH_CALUDE_tan_value_from_ratio_l925_92581

theorem tan_value_from_ratio (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan α = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_ratio_l925_92581


namespace NUMINAMATH_CALUDE_perimeter_ratio_of_similar_triangles_l925_92533

/-- Two triangles are similar with a given ratio -/
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) (r : ℝ) : Prop := sorry

/-- The perimeter of a triangle -/
def perimeter (t : Set (ℝ × ℝ)) : ℝ := sorry

theorem perimeter_ratio_of_similar_triangles 
  (abc a1b1c1 : Set (ℝ × ℝ)) : 
  similar_triangles abc a1b1c1 (1/2) → 
  perimeter abc / perimeter a1b1c1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_of_similar_triangles_l925_92533


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l925_92593

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, a (n + 1) - a n = 1) →  -- arithmetic sequence with common difference 1
  (∀ n, S n = n * a 1 + n * (n - 1) / 2) →  -- sum formula for arithmetic sequence
  (S 8 = 4 * S 4) →  -- given condition
  a 10 = 19 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l925_92593


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l925_92565

noncomputable def smallest_sum (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∃ x : ℝ, x^2 + (Real.sqrt 2 * a) * x + (Real.sqrt 2 * b) = 0) ∧
  (∃ x : ℝ, x^2 + (2 * b) * x + (Real.sqrt 2 * a) = 0) ∧
  a + b = (4 * Real.sqrt 2)^(2/3) / Real.sqrt 2 + (4 * Real.sqrt 2)^(1/3)

theorem smallest_sum_proof (a b : ℝ) :
  smallest_sum a b ↔ 
  (∀ c d : ℝ, c > 0 ∧ d > 0 ∧ 
   (∃ x : ℝ, x^2 + (Real.sqrt 2 * c) * x + (Real.sqrt 2 * d) = 0) ∧
   (∃ x : ℝ, x^2 + (2 * d) * x + (Real.sqrt 2 * c) = 0) →
   c + d ≥ (4 * Real.sqrt 2)^(2/3) / Real.sqrt 2 + (4 * Real.sqrt 2)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l925_92565


namespace NUMINAMATH_CALUDE_sum_of_seventh_powers_l925_92594

theorem sum_of_seventh_powers (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z = 0) (hprod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z * (x*y + x*z + y*z)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seventh_powers_l925_92594


namespace NUMINAMATH_CALUDE_negation_of_implication_l925_92564

theorem negation_of_implication (x : ℝ) :
  ¬(x^2 = 1 → x = 1 ∨ x = -1) ↔ (x^2 ≠ 1 ∧ x ≠ 1 ∧ x ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l925_92564


namespace NUMINAMATH_CALUDE_equation_solution_l925_92588

theorem equation_solution (x : ℝ) : 
  3 / (x + 2) = 2 / (x - 1) → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l925_92588


namespace NUMINAMATH_CALUDE_kendras_goal_is_sixty_l925_92519

/-- Kendra's goal for new words to learn before her eighth birthday -/
def kendras_goal (words_learned : ℕ) (words_needed : ℕ) : ℕ :=
  words_learned + words_needed

/-- Theorem: Kendra's goal is 60 words -/
theorem kendras_goal_is_sixty :
  kendras_goal 36 24 = 60 := by
  sorry

end NUMINAMATH_CALUDE_kendras_goal_is_sixty_l925_92519


namespace NUMINAMATH_CALUDE_max_a_for_same_range_l925_92522

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (1 / Real.exp 1) * Real.exp x + (a / 2) * x^2 - (a + 1) * x + a

theorem max_a_for_same_range : 
  ∃ (a_max : ℝ), a_max > 0 ∧ 
  (∀ (a : ℝ), a > 0 → 
    (Set.range (f a) = Set.range (fun x => f a (f a x))) → 
    a ≤ a_max) ∧
  (Set.range (f a_max) = Set.range (fun x => f a_max (f a_max x))) ∧
  a_max = 2 := by
sorry

end NUMINAMATH_CALUDE_max_a_for_same_range_l925_92522


namespace NUMINAMATH_CALUDE_banknote_replacement_theorem_l925_92590

/-- Represents the banknote replacement problem in the Magical Kingdom treasury --/
structure BanknoteReplacement where
  total_banknotes : ℕ
  machine_startup_cost : ℕ
  major_repair_cost : ℕ
  post_repair_capacity : ℕ
  budget : ℕ

/-- Calculates the number of banknotes replaced in a given number of days --/
def banknotes_replaced (br : BanknoteReplacement) (days : ℕ) : ℕ :=
  sorry

/-- Checks if all banknotes can be replaced within the budget --/
def can_replace_all (br : BanknoteReplacement) : Prop :=
  sorry

/-- The main theorem about banknote replacement --/
theorem banknote_replacement_theorem (br : BanknoteReplacement) 
  (h1 : br.total_banknotes = 3628800)
  (h2 : br.machine_startup_cost = 90000)
  (h3 : br.major_repair_cost = 700000)
  (h4 : br.post_repair_capacity = 1000000)
  (h5 : br.budget = 1000000) :
  (banknotes_replaced br 3 ≥ br.total_banknotes * 9 / 10) ∧
  (can_replace_all br) :=
sorry

end NUMINAMATH_CALUDE_banknote_replacement_theorem_l925_92590


namespace NUMINAMATH_CALUDE_expression_evaluation_l925_92575

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- The expression to be evaluated -/
def expression : ℂ := 2 * i^13 - 3 * i^18 + 4 * i^23 - 5 * i^28 + 6 * i^33

/-- The theorem stating the equality of the expression and its simplified form -/
theorem expression_evaluation : expression = 4 * i - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l925_92575


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l925_92586

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -4) (h2 : b = 1/2) :
  b * (a + b) + (-a + b) * (-a - b) - a^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l925_92586


namespace NUMINAMATH_CALUDE_distinct_paths_count_l925_92512

/-- Represents the number of purple arrows from point A -/
def purple_arrows : Nat := 2

/-- Represents the number of gray arrows each purple arrow leads to -/
def gray_arrows_per_purple : Nat := 2

/-- Represents the number of teal arrows each gray arrow leads to -/
def teal_arrows_per_gray : Nat := 3

/-- Represents the number of yellow arrows each teal arrow leads to -/
def yellow_arrows_per_teal : Nat := 2

/-- Represents the number of yellow arrows that lead to point B -/
def yellow_arrows_to_B : Nat := 4

/-- Theorem stating that the number of distinct paths from A to B is 96 -/
theorem distinct_paths_count : 
  purple_arrows * gray_arrows_per_purple * teal_arrows_per_gray * yellow_arrows_per_teal * yellow_arrows_to_B = 96 := by
  sorry

#eval purple_arrows * gray_arrows_per_purple * teal_arrows_per_gray * yellow_arrows_per_teal * yellow_arrows_to_B

end NUMINAMATH_CALUDE_distinct_paths_count_l925_92512


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l925_92527

theorem min_sum_with_reciprocal_constraint (x y : ℝ) : 
  x > 0 → y > 0 → (1 / (x + 2) + 1 / (y + 2) = 1 / 6) → 
  x + y ≥ 20 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / (x + 2) + 1 / (y + 2) = 1 / 6 ∧ x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l925_92527


namespace NUMINAMATH_CALUDE_textbook_savings_l925_92595

/-- Calculates the savings when buying a textbook from an external bookshop instead of the school bookshop -/
def calculate_savings (school_price : ℚ) (discount_percent : ℚ) : ℚ :=
  school_price * discount_percent / 100

/-- Represents the prices and discounts for the three textbooks -/
structure TextbookPrices where
  math_price : ℚ
  math_discount : ℚ
  science_price : ℚ
  science_discount : ℚ
  literature_price : ℚ
  literature_discount : ℚ

/-- Calculates the total savings for all three textbooks -/
def total_savings (prices : TextbookPrices) : ℚ :=
  calculate_savings prices.math_price prices.math_discount +
  calculate_savings prices.science_price prices.science_discount +
  calculate_savings prices.literature_price prices.literature_discount

/-- Theorem stating that the total savings is $29.25 -/
theorem textbook_savings :
  let prices : TextbookPrices := {
    math_price := 45,
    math_discount := 20,
    science_price := 60,
    science_discount := 25,
    literature_price := 35,
    literature_discount := 15
  }
  total_savings prices = 29.25 := by
  sorry

end NUMINAMATH_CALUDE_textbook_savings_l925_92595


namespace NUMINAMATH_CALUDE_total_pencils_l925_92548

theorem total_pencils (initial_pencils : ℕ) (additional_pencils : ℕ) :
  initial_pencils = 37 → additional_pencils = 17 →
  initial_pencils + additional_pencils = 54 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l925_92548


namespace NUMINAMATH_CALUDE_right_triangle_third_side_length_l925_92555

theorem right_triangle_third_side_length (a b c : ℝ) : 
  a = 8 → b = 15 → c ≥ 0 → a^2 + b^2 = c^2 → c ≥ 17 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_length_l925_92555


namespace NUMINAMATH_CALUDE_unique_solution_l925_92534

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := x + y = 3
def equation2 (x y : ℝ) : Prop := x - y = 1

-- Theorem statement
theorem unique_solution :
  ∃! (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l925_92534


namespace NUMINAMATH_CALUDE_inequality_transformation_l925_92506

theorem inequality_transformation (a b c : ℝ) (h1 : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by sorry

end NUMINAMATH_CALUDE_inequality_transformation_l925_92506


namespace NUMINAMATH_CALUDE_one_prime_in_sequence_l925_92549

/-- The number of digits in a natural number -/
def digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + digits (n / 10)

/-- The nth term of the sequence -/
def a : ℕ → ℕ
  | 0 => 37
  | n + 1 => 5 * 10^(digits (a n)) + a n

/-- The statement that there is exactly one prime in the sequence -/
theorem one_prime_in_sequence : ∃! k, k ∈ Set.range a ∧ Nat.Prime (a k) := by
  sorry

end NUMINAMATH_CALUDE_one_prime_in_sequence_l925_92549


namespace NUMINAMATH_CALUDE_minimum_race_distance_proof_l925_92585

/-- The minimum distance a runner must travel in the race setup -/
def minimum_race_distance : ℝ := 1011

/-- Point A's vertical distance from the wall -/
def distance_A_to_wall : ℝ := 400

/-- Point B's vertical distance above the wall -/
def distance_B_above_wall : ℝ := 600

/-- Point B's horizontal distance to the right of point A -/
def horizontal_distance_A_to_B : ℝ := 150

/-- Theorem stating the minimum distance a runner must travel -/
theorem minimum_race_distance_proof :
  let total_vertical_distance := distance_A_to_wall + distance_B_above_wall
  let squared_distance := horizontal_distance_A_to_B ^ 2 + total_vertical_distance ^ 2
  Real.sqrt squared_distance = minimum_race_distance := by sorry

end NUMINAMATH_CALUDE_minimum_race_distance_proof_l925_92585


namespace NUMINAMATH_CALUDE_number_of_divisors_3465_l925_92570

theorem number_of_divisors_3465 : Nat.card (Nat.divisors 3465) = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_3465_l925_92570


namespace NUMINAMATH_CALUDE_article_cost_l925_92558

/-- Proves that if selling an article for 350 gains 5% more than selling it for 340, then the cost is 140 -/
theorem article_cost (sell_price_high : ℝ) (sell_price_low : ℝ) (cost : ℝ) :
  sell_price_high = 350 ∧
  sell_price_low = 340 ∧
  (sell_price_high - cost) = (sell_price_low - cost) * 1.05 →
  cost = 140 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l925_92558


namespace NUMINAMATH_CALUDE_pig_count_l925_92513

theorem pig_count (initial_pigs additional_pigs : Float) 
  (h1 : initial_pigs = 64.0)
  (h2 : additional_pigs = 86.0) :
  initial_pigs + additional_pigs = 150.0 := by
sorry

end NUMINAMATH_CALUDE_pig_count_l925_92513


namespace NUMINAMATH_CALUDE_hall_ratio_l925_92508

/-- Given a rectangular hall with area 578 sq. m and difference between length and width 17 m,
    prove that the ratio of width to length is 1:2 -/
theorem hall_ratio (w l : ℝ) (hw : w > 0) (hl : l > 0) : 
  w * l = 578 → l - w = 17 → w / l = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hall_ratio_l925_92508
