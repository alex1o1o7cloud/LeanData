import Mathlib

namespace NUMINAMATH_CALUDE_power_function_properties_l2236_223619

-- Define the power function
def f (m : ℕ) (x : ℝ) : ℝ := x^(3*m - 5)

-- Define the theorem
theorem power_function_properties (m : ℕ) :
  (∀ x y, 0 < x ∧ x < y → f m y < f m x) ∧  -- f is decreasing on (0, +∞)
  (∀ x, f m (-x) = f m x) →                 -- f(-x) = f(x)
  m = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_properties_l2236_223619


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l2236_223695

/-- Represents the problem of a train crossing a bridge -/
def TrainCrossingBridge (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : Prop :=
  let total_distance : ℝ := train_length + bridge_length
  let train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
  let crossing_time : ℝ := total_distance / train_speed_mps
  crossing_time = 72.5

/-- Theorem stating that a train 250 meters long, running at 72 kmph, 
    takes 72.5 seconds to cross a bridge 1,200 meters in length -/
theorem train_crossing_bridge_time :
  TrainCrossingBridge 250 72 1200 := by
  sorry

#check train_crossing_bridge_time

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l2236_223695


namespace NUMINAMATH_CALUDE_f_properties_l2236_223649

noncomputable def f (x : ℝ) : ℝ := 2 * x / Real.log x

theorem f_properties :
  let e := Real.exp 1
  -- 1. f'(e^2) = 1/2
  (deriv f (e^2) = 1/2) ∧
  -- 2. f is monotonically decreasing on (0, 1) and (1, e)
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f y < f x) ∧
  (∀ x y, 1 < x ∧ x < y ∧ y < e → f y < f x) ∧
  -- 3. For all x > 0, x ≠ 1, f(x) > 2 / ln(x) + 2√x
  (∀ x, x > 0 ∧ x ≠ 1 → f x > 2 / Real.log x + 2 * Real.sqrt x) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l2236_223649


namespace NUMINAMATH_CALUDE_complement_A_intersection_A_complement_B_l2236_223600

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | x > 1}

-- Theorem for the first part
theorem complement_A : Set.compl A = {x : ℝ | x ≥ 2} := by sorry

-- Theorem for the second part
theorem intersection_A_complement_B : A ∩ Set.compl B = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersection_A_complement_B_l2236_223600


namespace NUMINAMATH_CALUDE_doughnuts_given_away_is_30_l2236_223686

/-- The number of doughnuts in each box -/
def doughnuts_per_box : ℕ := 10

/-- The total number of doughnuts made for the day -/
def total_doughnuts : ℕ := 300

/-- The number of boxes sold -/
def boxes_sold : ℕ := 27

/-- The number of doughnuts given away at the end of the day -/
def doughnuts_given_away : ℕ := total_doughnuts - (boxes_sold * doughnuts_per_box)

theorem doughnuts_given_away_is_30 : doughnuts_given_away = 30 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_given_away_is_30_l2236_223686


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2236_223637

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2236_223637


namespace NUMINAMATH_CALUDE_mean_visits_between_200_and_300_l2236_223682

def website_visits : List Nat := [300, 400, 300, 200, 200]

theorem mean_visits_between_200_and_300 :
  let mean := (website_visits.sum : ℚ) / website_visits.length
  200 < mean ∧ mean < 300 := by
  sorry

end NUMINAMATH_CALUDE_mean_visits_between_200_and_300_l2236_223682


namespace NUMINAMATH_CALUDE_pizza_slices_left_over_is_ten_l2236_223643

/-- Calculates the number of pizza slices left over given the conditions of the problem. -/
def pizza_slices_left_over : ℕ :=
  let small_pizza_slices : ℕ := 4
  let large_pizza_slices : ℕ := 8
  let small_pizzas_bought : ℕ := 3
  let large_pizzas_bought : ℕ := 2
  let george_slices : ℕ := 3
  let bob_slices : ℕ := george_slices + 1
  let susie_slices : ℕ := bob_slices / 2
  let bill_fred_mark_slices : ℕ := 3 * 3

  let total_slices : ℕ := small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought
  let total_eaten : ℕ := george_slices + bob_slices + susie_slices + bill_fred_mark_slices

  total_slices - total_eaten

theorem pizza_slices_left_over_is_ten : pizza_slices_left_over = 10 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_over_is_ten_l2236_223643


namespace NUMINAMATH_CALUDE_coin_flip_game_properties_l2236_223651

/-- Represents the coin-flipping game where a player wins if heads come up on an even-numbered throw
    or loses if tails come up on an odd-numbered throw. -/
def CoinFlipGame :=
  { win_prob : ℝ // win_prob = 1/3 } × { expected_flips : ℝ // expected_flips = 2 }

/-- The probability of winning the coin-flipping game is 1/3, and the expected number of flips is 2. -/
theorem coin_flip_game_properties : ∃ (game : CoinFlipGame), True :=
sorry

end NUMINAMATH_CALUDE_coin_flip_game_properties_l2236_223651


namespace NUMINAMATH_CALUDE_undefined_expression_l2236_223648

theorem undefined_expression (y : ℝ) : 
  (y^2 - 10*y + 25 = 0) ↔ (y = 5) := by
  sorry

#check undefined_expression

end NUMINAMATH_CALUDE_undefined_expression_l2236_223648


namespace NUMINAMATH_CALUDE_man_rowing_speed_l2236_223631

/-- The speed of the current downstream in kilometers per hour -/
def current_speed : ℝ := 3

/-- The time taken to cover the distance downstream in seconds -/
def time_downstream : ℝ := 9.390553103577801

/-- The distance covered downstream in meters -/
def distance_downstream : ℝ := 60

/-- The speed at which the man can row in still water in kilometers per hour -/
def rowing_speed : ℝ := 20

theorem man_rowing_speed :
  rowing_speed = 
    (distance_downstream / 1000) / (time_downstream / 3600) - current_speed :=
by sorry

end NUMINAMATH_CALUDE_man_rowing_speed_l2236_223631


namespace NUMINAMATH_CALUDE_at_least_three_prime_factors_l2236_223645

theorem at_least_three_prime_factors (n : ℕ) 
  (h1 : n > 0) 
  (h2 : n < 200) 
  (h3 : ∃ k : ℤ, (14 * n) / 60 = k) : 
  ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ∣ n ∧ q ∣ n ∧ r ∣ n :=
sorry

end NUMINAMATH_CALUDE_at_least_three_prime_factors_l2236_223645


namespace NUMINAMATH_CALUDE_smallest_n_for_jason_win_l2236_223638

/-- Represents the game board -/
structure GameBoard :=
  (width : Nat)
  (length : Nat)

/-- Represents a block that can be placed on the game board -/
structure Block :=
  (width : Nat)
  (length : Nat)

/-- Represents a player in the game -/
inductive Player
  | Jason
  | Jared

/-- Defines the game rules and conditions -/
def GameRules (board : GameBoard) (jasonBlock : Block) (jaredBlock : Block) :=
  board.width = 3 ∧
  board.length = 300 ∧
  jasonBlock.width = 2 ∧
  jasonBlock.length = 100 ∧
  jaredBlock.width = 2 ∧
  jaredBlock.length > 3

/-- Determines if a player can win given the game rules and block sizes -/
def CanWin (player : Player) (board : GameBoard) (jasonBlock : Block) (jaredBlock : Block) : Prop :=
  sorry

/-- The main theorem stating that 51 is the smallest n for Jason to guarantee a win -/
theorem smallest_n_for_jason_win (board : GameBoard) (jasonBlock : Block) (jaredBlock : Block) :
  GameRules board jasonBlock jaredBlock →
  (∀ n : Nat, n > 3 → n < 51 → ¬CanWin Player.Jason board jasonBlock {width := 2, length := n}) ∧
  CanWin Player.Jason board jasonBlock {width := 2, length := 51} :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_jason_win_l2236_223638


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2236_223656

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1
  left_focus_x : ℝ
  left_focus_on_directrix : left_focus_x = -5  -- directrix of y^2 = 20x is x = -5
  asymptote_slope : b / a = 4 / 3

/-- The standard equation of the hyperbola is x^2/9 - y^2/16 = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) : 
  h.a = 3 ∧ h.b = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2236_223656


namespace NUMINAMATH_CALUDE_range_of_m_for_positive_f_range_of_m_for_zero_in_interval_l2236_223610

/-- The function f(x) = x^2 - (m-1)x + 2m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-1)*x + 2*m

/-- Theorem 1: f(x) > 0 for all x in (0, +∞) iff -2√6 + 5 ≤ m ≤ 2√6 + 5 -/
theorem range_of_m_for_positive_f (m : ℝ) :
  (∀ x > 0, f m x > 0) ↔ -2*Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2*Real.sqrt 6 + 5 :=
sorry

/-- Theorem 2: f(x) has a zero point in (0, 1) iff m ∈ (-2, 0) -/
theorem range_of_m_for_zero_in_interval (m : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, f m x = 0) ↔ m > -2 ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_positive_f_range_of_m_for_zero_in_interval_l2236_223610


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l2236_223665

theorem least_positive_angle_theorem : ∃ θ : Real,
  θ > 0 ∧
  θ < 360 ∧
  Real.cos (15 * π / 180) = Real.sin (45 * π / 180) + Real.sin θ ∧
  θ = 195 * π / 180 ∧
  ∀ φ, 0 < φ ∧ φ < θ → Real.cos (15 * π / 180) ≠ Real.sin (45 * π / 180) + Real.sin φ :=
by sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l2236_223665


namespace NUMINAMATH_CALUDE_no_even_integers_satisfying_conditions_l2236_223641

theorem no_even_integers_satisfying_conditions : 
  ¬ ∃ (n : ℤ), 
    (n % 2 = 0) ∧ 
    (100 ≤ n) ∧ (n ≤ 1000) ∧ 
    (∃ (k : ℕ), n = 3 * k + 4) ∧ 
    (∃ (m : ℕ), n = 5 * m + 2) := by
  sorry

end NUMINAMATH_CALUDE_no_even_integers_satisfying_conditions_l2236_223641


namespace NUMINAMATH_CALUDE_at_op_difference_l2236_223627

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y + x - y

-- State the theorem
theorem at_op_difference : at_op 7 4 - at_op 4 7 = 6 := by sorry

end NUMINAMATH_CALUDE_at_op_difference_l2236_223627


namespace NUMINAMATH_CALUDE_cube_root_of_a_minus_m_l2236_223658

theorem cube_root_of_a_minus_m (a m : ℝ) (ha : 0 < a) 
  (h1 : (m + 7)^2 = a) (h2 : (2*m - 1)^2 = a) : 
  (a - m)^(1/3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_a_minus_m_l2236_223658


namespace NUMINAMATH_CALUDE_ricciana_long_jump_l2236_223687

/-- Ricciana's long jump problem -/
theorem ricciana_long_jump :
  ∀ (ricciana_run margarita_run ricciana_jump margarita_jump : ℕ),
  ricciana_run = 20 →
  margarita_run = 18 →
  margarita_jump = 2 * ricciana_jump - 1 →
  margarita_run + margarita_jump = ricciana_run + ricciana_jump + 1 →
  ricciana_jump = 22 := by
sorry

end NUMINAMATH_CALUDE_ricciana_long_jump_l2236_223687


namespace NUMINAMATH_CALUDE_cubic_root_sum_square_l2236_223626

theorem cubic_root_sum_square (a b c s : ℝ) : 
  (a^3 - 12*a^2 + 14*a - 1 = 0) →
  (b^3 - 12*b^2 + 14*b - 1 = 0) →
  (c^3 - 12*c^2 + 14*c - 1 = 0) →
  (s = Real.sqrt a + Real.sqrt b + Real.sqrt c) →
  (s^4 - 24*s^2 - 10*s = -144) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_square_l2236_223626


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l2236_223669

theorem smallest_x_absolute_value_equation : 
  (∃ x : ℝ, |4*x - 5| = 29) ∧ 
  (∀ x : ℝ, |4*x - 5| = 29 → x ≥ -6) ∧ 
  |4*(-6) - 5| = 29 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l2236_223669


namespace NUMINAMATH_CALUDE_impossible_card_arrangement_l2236_223624

/-- Represents the arrangement of cards --/
def CardArrangement := List ℕ

/-- Calculates the sum of spaces between pairs of identical digits --/
def sumOfSpaces (arr : CardArrangement) : ℕ := sorry

/-- Checks if an arrangement is valid according to the problem's conditions --/
def isValidArrangement (arr : CardArrangement) : Prop := sorry

/-- Theorem stating the impossibility of the desired arrangement --/
theorem impossible_card_arrangement : 
  ¬ ∃ (arr : CardArrangement), 
    (arr.length = 20) ∧ 
    (∀ d, (arr.count d = 2) ∨ (arr.count d = 0)) ∧
    (isValidArrangement arr) := by
  sorry

end NUMINAMATH_CALUDE_impossible_card_arrangement_l2236_223624


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_36_l2236_223620

/-- A set of consecutive positive integers -/
def ConsecutiveSet (start : ℕ) (length : ℕ) : Set ℕ :=
  {n : ℕ | start ≤ n ∧ n < start + length}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

/-- Theorem: There exists exactly one set of consecutive positive integers,
    containing at least two integers, whose sum is 36 -/
theorem unique_consecutive_sum_36 :
  ∃! (start length : ℕ), 
    length ≥ 2 ∧ 
    ConsecutiveSum start length = 36 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_36_l2236_223620


namespace NUMINAMATH_CALUDE_factorial_squared_gt_power_l2236_223602

theorem factorial_squared_gt_power (n : ℕ) (h : n > 2) : (n.factorial ^ 2 : ℕ) > n ^ n := by
  sorry

end NUMINAMATH_CALUDE_factorial_squared_gt_power_l2236_223602


namespace NUMINAMATH_CALUDE_area_of_divided_square_l2236_223628

/-- A square divided into rectangles of equal area with specific properties -/
structure DividedSquare where
  side : ℝ
  segment_AB : ℝ
  is_divided : Bool
  A_is_midpoint : Bool

/-- The area of a DividedSquare with given properties -/
def square_area (s : DividedSquare) : ℝ := s.side ^ 2

/-- Theorem stating the area of the square under given conditions -/
theorem area_of_divided_square (s : DividedSquare) 
  (h1 : s.is_divided = true)
  (h2 : s.segment_AB = 1)
  (h3 : s.A_is_midpoint = true) :
  square_area s = 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_divided_square_l2236_223628


namespace NUMINAMATH_CALUDE_table_tennis_sequences_l2236_223689

/-- Represents a sequence of matches in the table tennis competition -/
def MatchSequence := List ℕ

/-- The number of players in each team -/
def teamSize : ℕ := 5

/-- Calculates the number of possible sequences for a given player finishing the competition -/
def sequencesForPlayer (player : ℕ) : ℕ := sorry

/-- Calculates the total number of possible sequences for one team winning -/
def totalSequencesOneTeam : ℕ :=
  (List.range teamSize).map sequencesForPlayer |>.sum

/-- The total number of possible sequences in the competition -/
def totalSequences : ℕ := 2 * totalSequencesOneTeam

theorem table_tennis_sequences :
  totalSequences = 252 := by sorry

end NUMINAMATH_CALUDE_table_tennis_sequences_l2236_223689


namespace NUMINAMATH_CALUDE_symmetry_implies_k_and_b_values_l2236_223655

/-- A linear function f(x) = mx + c is symmetric with respect to the y-axis if f(x) = f(-x) for all x -/
def SymmetricToYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

/-- The first linear function f(x) = kx - 5 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x - 5

/-- The second linear function g(x) = 2x + b -/
def g (b : ℝ) (x : ℝ) : ℝ := 2 * x + b

theorem symmetry_implies_k_and_b_values :
  ∀ k b : ℝ, 
    SymmetricToYAxis (f k) ∧ 
    SymmetricToYAxis (g b) →
    k = -2 ∧ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_k_and_b_values_l2236_223655


namespace NUMINAMATH_CALUDE_impossibility_of_arrangement_l2236_223653

/-- Represents a 6x7 grid of natural numbers -/
def Grid := Fin 6 → Fin 7 → ℕ

/-- Checks if a given grid is a valid arrangement of numbers 1 to 42 -/
def is_valid_arrangement (g : Grid) : Prop :=
  (∀ i j, g i j ≥ 1 ∧ g i j ≤ 42) ∧
  (∀ i j k l, (i ≠ k ∨ j ≠ l) → g i j ≠ g k l)

/-- Checks if the sum of numbers in each 1x2 vertical rectangle is even -/
def has_even_vertical_sums (g : Grid) : Prop :=
  ∀ i j, Even (g i j + g (i.succ) j)

theorem impossibility_of_arrangement :
  ¬∃ (g : Grid), is_valid_arrangement g ∧ has_even_vertical_sums g :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_arrangement_l2236_223653


namespace NUMINAMATH_CALUDE_find_m_max_sum_squares_max_sum_squares_achievable_l2236_223615

-- Define the condition for the unique integer solution
def uniqueIntegerSolution (m : ℤ) : Prop :=
  ∃! (x : ℤ), |2 * x - m| ≤ 1

-- Define the condition for a, b, c
def abcCondition (a b c : ℝ) : Prop :=
  4 * a^4 + 4 * b^4 + 4 * c^4 = 6

-- Theorem 1: Prove m = 6
theorem find_m (m : ℤ) (h : uniqueIntegerSolution m) : m = 6 := by
  sorry

-- Theorem 2: Prove the maximum value of a^2 + b^2 + c^2
theorem max_sum_squares (a b c : ℝ) (h : abcCondition a b c) :
  a^2 + b^2 + c^2 ≤ 3 * Real.sqrt 2 / 2 := by
  sorry

-- Theorem 3: Prove the maximum value is achievable
theorem max_sum_squares_achievable :
  ∃ a b c : ℝ, abcCondition a b c ∧ a^2 + b^2 + c^2 = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_find_m_max_sum_squares_max_sum_squares_achievable_l2236_223615


namespace NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l2236_223661

structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagonal_faces : ℕ

def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_segments := (Q.vertices.choose 2)
  let face_diagonals := 2 * Q.quadrilateral_faces + 5 * Q.pentagonal_faces
  total_segments - Q.edges - face_diagonals

theorem space_diagonals_of_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 40,
    triangular_faces := 20,
    quadrilateral_faces := 15,
    pentagonal_faces := 5
  }
  space_diagonals Q = 310 := by sorry

end NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l2236_223661


namespace NUMINAMATH_CALUDE_even_function_sum_ab_eq_two_l2236_223691

/-- A function f is even on an interval if f(-x) = f(x) for all x in the interval -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x, x ∈ Set.Icc a b → f (-x) = f x

theorem even_function_sum_ab_eq_two (a b : ℝ) :
  let f := fun x => a * x^2 + (b - 1) * x + 3 * a
  let domain := Set.Icc (a - 3) (2 * a)
  IsEvenOn f (a - 3) (2 * a) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_ab_eq_two_l2236_223691


namespace NUMINAMATH_CALUDE_right_angled_triangle_l2236_223611

theorem right_angled_triangle (a b c : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 3) (h3 : c = 2) :
  a ^ 2 + b ^ 2 = c ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l2236_223611


namespace NUMINAMATH_CALUDE_college_student_count_l2236_223659

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The ratio of boys to girls is 5:7 -/
def ratio_condition (c : College) : Prop :=
  7 * c.boys = 5 * c.girls

/-- There are 140 girls -/
def girls_count (c : College) : Prop :=
  c.girls = 140

/-- The total number of students -/
def total_students (c : College) : ℕ :=
  c.boys + c.girls

/-- Theorem stating the total number of students in the college -/
theorem college_student_count (c : College) 
  (h1 : ratio_condition c) (h2 : girls_count c) : 
  total_students c = 240 := by
  sorry

end NUMINAMATH_CALUDE_college_student_count_l2236_223659


namespace NUMINAMATH_CALUDE_cereal_cost_l2236_223684

/-- Represents the cost of cereal boxes for a year -/
def cereal_problem (boxes_per_week : ℕ) (weeks_per_year : ℕ) (total_cost : ℕ) : Prop :=
  let total_boxes := boxes_per_week * weeks_per_year
  total_cost / total_boxes = 3

/-- Proves that each box of cereal costs $3 given the problem conditions -/
theorem cereal_cost : cereal_problem 2 52 312 := by
  sorry

end NUMINAMATH_CALUDE_cereal_cost_l2236_223684


namespace NUMINAMATH_CALUDE_china_gdp_growth_l2236_223644

/-- China's GDP growth model from 2011 to 2016 -/
theorem china_gdp_growth (a r : ℝ) (h : a > 0) (h2 : r > 0) :
  let initial_gdp := a
  let growth_rate := r / 100
  let years := 5
  let final_gdp := initial_gdp * (1 + growth_rate) ^ years
  final_gdp = a * (1 + r / 100) ^ 5 := by sorry

end NUMINAMATH_CALUDE_china_gdp_growth_l2236_223644


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l2236_223679

-- Define the operation ⊗
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3))))

-- Theorem statement
theorem bowtie_equation_solution (x : ℝ) :
  bowtie 3 x = 12 → x = 69 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l2236_223679


namespace NUMINAMATH_CALUDE_pet_ownership_l2236_223609

theorem pet_ownership (S : Finset Nat) (D C B : Finset Nat) : 
  S.card = 60 ∧
  (∀ s ∈ S, s ∈ D ∪ C ∪ B) ∧
  D.card = 35 ∧
  C.card = 45 ∧
  B.card = 10 ∧
  (∀ b ∈ B, b ∈ D ∪ C) →
  ((D ∩ C) \ B).card = 10 := by
sorry

end NUMINAMATH_CALUDE_pet_ownership_l2236_223609


namespace NUMINAMATH_CALUDE_xyz_product_l2236_223697

/-- Given complex numbers x, y, and z satisfying the specified equations,
    prove that their product equals 260/3. -/
theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -25) :
  x * y * z = 260 / 3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l2236_223697


namespace NUMINAMATH_CALUDE_functional_inequality_domain_l2236_223673

-- Define the function f
def f (n : ℕ) (x : ℝ) : ℝ := x^n

-- Define the theorem
theorem functional_inequality_domain (n : ℕ) (h_n : n > 1) :
  ∀ x : ℝ, (f n x + f n (1 - x) > 1) ↔ (x < 0 ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_functional_inequality_domain_l2236_223673


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2236_223662

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-1/2 : ℝ) (1/3 : ℝ) = {x : ℝ | a * x^2 + b * x + 2 > 0}) :
  {x : ℝ | 2 * x^2 + b * x + a < 0} = Set.Ioo (-2 : ℝ) (3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2236_223662


namespace NUMINAMATH_CALUDE_hyperbola_b_value_l2236_223650

/-- Given a hyperbola with equation x^2 - my^2 = 3m (where m > 0),
    prove that the value of b in its standard form is √3. -/
theorem hyperbola_b_value (m : ℝ) (h : m > 0) :
  let C : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 - m*y^2 = 3*m
  ∃ (a b : ℝ), (∀ (x y : ℝ), C (x, y) ↔ (x^2 / (a^2) - y^2 / (b^2) = 1)) ∧ b = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_b_value_l2236_223650


namespace NUMINAMATH_CALUDE_decimal_digit_13_14_l2236_223657

def decimal_cycle (n d : ℕ) (cycle : List ℕ) : Prop :=
  ∀ k : ℕ, (n * 10^k) % d = (cycle.take ((k - 1) % cycle.length + 1)).foldl (λ acc x => (10 * acc + x) % d) 0

theorem decimal_digit_13_14 :
  decimal_cycle 13 14 [9, 2, 8, 5, 7, 1] →
  (13 * 10^150) / 14 % 10 = 1 := by
sorry

end NUMINAMATH_CALUDE_decimal_digit_13_14_l2236_223657


namespace NUMINAMATH_CALUDE_non_negative_integer_solutions_of_inequality_l2236_223667

theorem non_negative_integer_solutions_of_inequality :
  {x : ℕ | x + 1 < 4} = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_non_negative_integer_solutions_of_inequality_l2236_223667


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2236_223642

theorem algebraic_expression_value (a b : ℝ) (h : 2 * a - b = 2) :
  4 * a^2 - b^2 - 4 * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2236_223642


namespace NUMINAMATH_CALUDE_flour_for_hundred_cookies_l2236_223675

-- Define the recipe's ratio
def recipe_cookies : ℕ := 40
def recipe_flour : ℚ := 3

-- Define the desired number of cookies
def desired_cookies : ℕ := 100

-- Define the function to calculate required flour
def required_flour (cookies : ℕ) : ℚ :=
  (recipe_flour / recipe_cookies) * cookies

-- Theorem statement
theorem flour_for_hundred_cookies :
  required_flour desired_cookies = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_flour_for_hundred_cookies_l2236_223675


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2236_223674

/-- Proves that a train with given length, crossing a bridge of given length in a given time, has a specific speed in km/hr -/
theorem train_speed_calculation (train_length bridge_length : Real) (crossing_time : Real) :
  train_length = 110 →
  bridge_length = 175 →
  crossing_time = 14.248860091192705 →
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2236_223674


namespace NUMINAMATH_CALUDE_tv_purchase_price_l2236_223634

/-- The purchase price of a TV -/
def purchase_price : ℝ := 2250

/-- The profit made on each TV -/
def profit : ℝ := 270

/-- The price increase percentage -/
def price_increase : ℝ := 0.4

/-- The discount percentage -/
def discount : ℝ := 0.2

theorem tv_purchase_price :
  (purchase_price + purchase_price * price_increase) * (1 - discount) - purchase_price = profit :=
by sorry

end NUMINAMATH_CALUDE_tv_purchase_price_l2236_223634


namespace NUMINAMATH_CALUDE_cube_with_specific_digits_l2236_223688

theorem cube_with_specific_digits : ∃! n : ℕ, 
  (n^3 ≥ 30000 ∧ n^3 < 40000) ∧ 
  (n^3 % 10 = 4) ∧
  (n = 34) := by
  sorry

end NUMINAMATH_CALUDE_cube_with_specific_digits_l2236_223688


namespace NUMINAMATH_CALUDE_car_dealership_count_l2236_223630

theorem car_dealership_count :
  ∀ (total_cars : ℕ),
    (total_cars : ℝ) * 0.6 * 0.6 = 216 →
    total_cars = 600 := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_count_l2236_223630


namespace NUMINAMATH_CALUDE_roof_shingle_width_l2236_223635

/-- The width of a rectangular roof shingle with length 10 inches and area 70 square inches is 7 inches. -/
theorem roof_shingle_width :
  ∀ (width : ℝ), 
    (10 : ℝ) * width = 70 → width = 7 := by
  sorry

end NUMINAMATH_CALUDE_roof_shingle_width_l2236_223635


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2236_223633

/-- Given a line and a circle, prove that the coefficient of x in the line equation is 2 when the chord length is 4 -/
theorem line_circle_intersection (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 4*x - 2*y + 1 = 0 ∧ a*x + y - 5 = 0) → -- Circle and line intersect
  (∃ x1 y1 x2 y2 : ℝ, 
    x1^2 + y1^2 - 4*x1 - 2*y1 + 1 = 0 ∧ 
    x2^2 + y2^2 - 4*x2 - 2*y2 + 1 = 0 ∧ 
    a*x1 + y1 - 5 = 0 ∧ 
    a*x2 + y2 - 5 = 0 ∧ 
    (x1 - x2)^2 + (y1 - y2)^2 = 16) → -- Chord length is 4
  a = 2 := by
  sorry

#check line_circle_intersection

end NUMINAMATH_CALUDE_line_circle_intersection_l2236_223633


namespace NUMINAMATH_CALUDE_thickness_after_13_folds_l2236_223601

/-- The thickness of a paper after n folds, given an initial thickness of a millimeters -/
def paper_thickness (a : ℝ) (n : ℕ) : ℝ :=
  a * 2^n

/-- Theorem: The thickness of a paper after 13 folds is 2^13 times its initial thickness -/
theorem thickness_after_13_folds (a : ℝ) :
  paper_thickness a 13 = a * 2^13 := by
  sorry

#check thickness_after_13_folds

end NUMINAMATH_CALUDE_thickness_after_13_folds_l2236_223601


namespace NUMINAMATH_CALUDE_bamboo_nine_sections_l2236_223694

/-- Given an arithmetic sequence of 9 terms, prove that if the sum of the first 4 terms is 3
    and the sum of the last 3 terms is 4, then the 5th term is 67/66 -/
theorem bamboo_nine_sections 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_sum_first_four : a 1 + a 2 + a 3 + a 4 = 3)
  (h_sum_last_three : a 7 + a 8 + a 9 = 4) :
  a 5 = 67 / 66 := by
sorry

end NUMINAMATH_CALUDE_bamboo_nine_sections_l2236_223694


namespace NUMINAMATH_CALUDE_original_plan_pages_l2236_223671

-- Define the total number of pages in the book
def total_pages : ℕ := 200

-- Define the number of days before changing the plan
def days_before_change : ℕ := 5

-- Define the additional pages read per day after changing the plan
def additional_pages : ℕ := 5

-- Define the number of days earlier the book was finished
def days_earlier : ℕ := 1

-- Define the function to calculate the total pages read
def total_pages_read (x : ℕ) : ℕ :=
  (days_before_change * x) + 
  ((x + additional_pages) * (total_pages / x - days_before_change - days_earlier))

-- Theorem stating that the original plan was to read 20 pages per day
theorem original_plan_pages : 
  ∃ (x : ℕ), x > 0 ∧ total_pages_read x = total_pages ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_plan_pages_l2236_223671


namespace NUMINAMATH_CALUDE_abc_modulo_seven_l2236_223693

theorem abc_modulo_seven (a b c : ℕ) 
  (ha : a < 7) (hb : b < 7) (hc : c < 7)
  (h1 : (a + 2*b + 3*c) % 7 = 0)
  (h2 : (2*a + 3*b + c) % 7 = 4)
  (h3 : (3*a + b + 2*c) % 7 = 4) :
  (a * b * c) % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_abc_modulo_seven_l2236_223693


namespace NUMINAMATH_CALUDE_percentage_failed_english_l2236_223636

theorem percentage_failed_english (total_percentage : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  total_percentage = 100 ∧
  failed_hindi = 30 ∧
  failed_both = 28 ∧
  passed_both = 56 →
  ∃ failed_english : ℝ,
    failed_english = 42 ∧
    total_percentage - passed_both = failed_hindi + failed_english - failed_both :=
by sorry

end NUMINAMATH_CALUDE_percentage_failed_english_l2236_223636


namespace NUMINAMATH_CALUDE_triangle_circumcircle_l2236_223678

/-- Given a triangle with sides defined by three linear equations, 
    prove that its circumscribed circle has the specified equation. -/
theorem triangle_circumcircle 
  (line1 : ℝ → ℝ → Prop) 
  (line2 : ℝ → ℝ → Prop)
  (line3 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, line1 x y ↔ x - 3*y = 2)
  (h2 : ∀ x y, line2 x y ↔ 7*x - y = 34)
  (h3 : ∀ x y, line3 x y ↔ x + 2*y = -8) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧ 
    radius = 5 ∧
    (∀ x y, (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ 
      (x - 1)^2 + (y + 2)^2 = 25) :=
sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_l2236_223678


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2236_223613

theorem complex_equation_solution (z : ℂ) : z * (2 - I) = 3 + I → z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2236_223613


namespace NUMINAMATH_CALUDE_kid_tickets_sold_l2236_223681

theorem kid_tickets_sold (adult_price kid_price total_tickets total_profit : ℕ) 
  (h1 : adult_price = 12)
  (h2 : kid_price = 5)
  (h3 : total_tickets = 275)
  (h4 : total_profit = 2150) :
  ∃ (adult_tickets kid_tickets : ℕ),
    adult_tickets + kid_tickets = total_tickets ∧
    adult_price * adult_tickets + kid_price * kid_tickets = total_profit ∧
    kid_tickets = 164 := by
  sorry

end NUMINAMATH_CALUDE_kid_tickets_sold_l2236_223681


namespace NUMINAMATH_CALUDE_inequality_proof_l2236_223677

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) :
  1/a + 1/b ≥ 2*(a^2 - a + 1)*(b^2 - b + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2236_223677


namespace NUMINAMATH_CALUDE_complete_square_d_value_l2236_223696

/-- Given a quadratic equation x^2 - 6x + 5 = 0, prove that when converted to the form (x + c)^2 = d, the value of d is 4 -/
theorem complete_square_d_value (x : ℝ) : 
  (x^2 - 6*x + 5 = 0) → 
  (∃ c d : ℝ, (x + c)^2 = d ∧ x^2 - 6*x + 5 = 0) →
  (∃ c : ℝ, (x + c)^2 = 4 ∧ x^2 - 6*x + 5 = 0) :=
by sorry


end NUMINAMATH_CALUDE_complete_square_d_value_l2236_223696


namespace NUMINAMATH_CALUDE_min_team_size_proof_l2236_223683

def P₁ : ℝ := 0.3

def individual_prob : ℝ := 0.1

def P₂ (n : ℕ) : ℝ := 1 - (1 - individual_prob) ^ n

def min_team_size : ℕ := 4

theorem min_team_size_proof :
  ∀ n : ℕ, (P₂ n ≥ P₁) → n ≥ min_team_size :=
sorry

end NUMINAMATH_CALUDE_min_team_size_proof_l2236_223683


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2236_223668

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 3 * x + 1 > 0}
def B : Set ℝ := {x : ℝ | |x - 1| < 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (-1/3 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2236_223668


namespace NUMINAMATH_CALUDE_bookshop_unsold_percentage_l2236_223639

def initial_stock : ℕ := 1200
def sales : List ℕ := [75, 50, 64, 78, 135]

def books_sold (sales : List ℕ) : ℕ := sales.sum

def books_not_sold (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

def percentage_not_sold (initial : ℕ) (not_sold : ℕ) : ℚ :=
  (not_sold : ℚ) / (initial : ℚ) * 100

theorem bookshop_unsold_percentage :
  let sold := books_sold sales
  let not_sold := books_not_sold initial_stock sold
  percentage_not_sold initial_stock not_sold = 66.5 := by
  sorry

end NUMINAMATH_CALUDE_bookshop_unsold_percentage_l2236_223639


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l2236_223654

/-- Given a line passing through points (4, -7) and (k, 25) that is parallel to the line 3x + 4y = 12, 
    the value of k is -116/3. -/
theorem parallel_line_k_value : 
  ∀ k : ℚ, 
  (∃ m b : ℚ, (∀ x y : ℚ, y = m * x + b → (x = 4 ∧ y = -7) ∨ (x = k ∧ y = 25)) ∧ 
               m = -(3 / 4)) → 
  k = -116 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l2236_223654


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2236_223605

/-- Proves that given the conditions of the age problem, the ratio of Michael's age to Monica's age is 3:5 -/
theorem age_ratio_problem (patrick_age michael_age monica_age : ℕ) : 
  patrick_age * 5 = michael_age * 3 →  -- Patrick and Michael's ages are in ratio 3:5
  patrick_age + michael_age + monica_age = 196 →  -- Sum of ages is 196
  monica_age - patrick_age = 64 →  -- Difference between Monica's and Patrick's ages is 64
  michael_age * 5 = monica_age * 3  -- Conclusion: Michael and Monica's ages are in ratio 3:5
:= by sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2236_223605


namespace NUMINAMATH_CALUDE_leading_coefficient_of_P_l2236_223646

/-- The polynomial in question -/
def P (x : ℝ) : ℝ := 5*(x^5 - 2*x^4 + 3*x^3) - 6*(x^5 + x^3 + x) + 3*(3*x^5 - x^4 + 4*x^2 + 2)

/-- The leading coefficient of a polynomial -/
def leading_coefficient (p : ℝ → ℝ) : ℝ := 
  sorry

theorem leading_coefficient_of_P : leading_coefficient P = 8 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_P_l2236_223646


namespace NUMINAMATH_CALUDE_ascending_order_abab_l2236_223685

theorem ascending_order_abab (a b : ℝ) (ha : a > 0) (hb : b < 0) (hab : a + b > 0) :
  -a < b ∧ b < -b ∧ -b < a := by sorry

end NUMINAMATH_CALUDE_ascending_order_abab_l2236_223685


namespace NUMINAMATH_CALUDE_shooter_conditional_probability_l2236_223690

/-- Given a shooter with probabilities of hitting a target, prove the conditional probability of hitting the target in a subsequent shot. -/
theorem shooter_conditional_probability
  (p_single : ℝ)
  (p_twice : ℝ)
  (h_single : p_single = 0.7)
  (h_twice : p_twice = 0.4) :
  p_twice / p_single = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_shooter_conditional_probability_l2236_223690


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2236_223660

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2236_223660


namespace NUMINAMATH_CALUDE_cos_eight_degrees_l2236_223632

theorem cos_eight_degrees (m : ℝ) (h : Real.sin (74 * π / 180) = m) :
  Real.cos (8 * π / 180) = Real.sqrt ((1 + m) / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_eight_degrees_l2236_223632


namespace NUMINAMATH_CALUDE_p_value_l2236_223663

/-- The maximum value of x satisfying the inequality |x^2-4x+p|+|x-3|≤5 is 3 -/
def max_x_condition (p : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 - 4*x + p| + |x - 3| ≤ 5 → x ≤ 3

/-- Theorem stating that p = 8 given the condition -/
theorem p_value : ∃ p : ℝ, max_x_condition p ∧ p = 8 :=
sorry

end NUMINAMATH_CALUDE_p_value_l2236_223663


namespace NUMINAMATH_CALUDE_student_average_score_l2236_223604

theorem student_average_score (math physics chem : ℕ) : 
  math + physics = 32 →
  chem = physics + 20 →
  (math + chem) / 2 = 26 := by
sorry

end NUMINAMATH_CALUDE_student_average_score_l2236_223604


namespace NUMINAMATH_CALUDE_sally_monday_seashells_l2236_223603

/-- The number of seashells Sally picked on Monday -/
def monday_seashells : ℕ := sorry

/-- The number of seashells Sally picked on Tuesday -/
def tuesday_seashells : ℕ := sorry

/-- The price of each seashell in dollars -/
def seashell_price : ℚ := 6/5

/-- The total amount Sally can make by selling all seashells in dollars -/
def total_amount : ℕ := 54

/-- Theorem stating the number of seashells Sally picked on Monday -/
theorem sally_monday_seashells : 
  monday_seashells = 30 ∧
  tuesday_seashells = monday_seashells / 2 ∧
  seashell_price * (monday_seashells + tuesday_seashells : ℚ) = total_amount := by
  sorry

end NUMINAMATH_CALUDE_sally_monday_seashells_l2236_223603


namespace NUMINAMATH_CALUDE_inequality_iff_in_solution_set_l2236_223608

/-- The solution set for the inequality 1/(x(x+2)) - 1/((x+2)(x+3)) < 1/4 -/
def solution_set : Set ℝ :=
  { x | x < -3 ∨ (-2 < x ∧ x < 0) ∨ 1 < x }

/-- The inequality function -/
def inequality (x : ℝ) : Prop :=
  1 / (x * (x + 2)) - 1 / ((x + 2) * (x + 3)) < 1 / 4

theorem inequality_iff_in_solution_set :
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_iff_in_solution_set_l2236_223608


namespace NUMINAMATH_CALUDE_sphere_volume_from_circumference_l2236_223676

/-- The volume of a sphere with circumference 30 cm is 4500/π² cm³ -/
theorem sphere_volume_from_circumference :
  ∀ (r : ℝ), 
    2 * π * r = 30 → 
    (4 / 3) * π * r ^ 3 = 4500 / π ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_circumference_l2236_223676


namespace NUMINAMATH_CALUDE_unique_solution_l2236_223622

theorem unique_solution : ∃! x : ℝ, 
  (|x - 3| + |x + 4| < 8) ∧ (x^2 - x - 12 = 0) :=
by
  -- The unique solution is x = -3
  use -3
  constructor
  · -- Prove that x = -3 satisfies both conditions
    constructor
    · -- Prove |(-3) - 3| + |(-3) + 4| < 8
      sorry
    · -- Prove (-3)^2 - (-3) - 12 = 0
      sorry
  · -- Prove that no other value satisfies both conditions
    sorry

#check unique_solution

end NUMINAMATH_CALUDE_unique_solution_l2236_223622


namespace NUMINAMATH_CALUDE_defective_probability_l2236_223692

/-- The probability of an item being produced by Machine 1 -/
def prob_machine1 : ℝ := 0.4

/-- The probability of an item being produced by Machine 2 -/
def prob_machine2 : ℝ := 0.6

/-- The probability of a defective item from Machine 1 -/
def defect_rate1 : ℝ := 0.03

/-- The probability of a defective item from Machine 2 -/
def defect_rate2 : ℝ := 0.02

/-- The probability of a randomly selected item being defective -/
def prob_defective : ℝ := prob_machine1 * defect_rate1 + prob_machine2 * defect_rate2

theorem defective_probability : prob_defective = 0.024 := by
  sorry

end NUMINAMATH_CALUDE_defective_probability_l2236_223692


namespace NUMINAMATH_CALUDE_rectangle_ratio_golden_ratio_l2236_223614

/-- Given a unit square AEFD and rectangles ABCD and BCFE, where the ratio of length to width
    of ABCD equals the ratio of length to width of BCFE, and AB has length W,
    prove that W = (1 + √5) / 2. -/
theorem rectangle_ratio_golden_ratio (W : ℝ) : 
  (W > 0) →  -- W is positive
  (W / 1 = 1 / (W - 1)) →  -- ratio equality condition
  W = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_golden_ratio_l2236_223614


namespace NUMINAMATH_CALUDE_arithmetic_progression_equality_l2236_223606

/-- An arithmetic progression with first term and difference as natural numbers -/
structure ArithmeticProgression :=
  (first : ℕ)
  (diff : ℕ)
  (coprime : Nat.Coprime first diff)

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℕ :=
  ap.first + (n - 1) * ap.diff

theorem arithmetic_progression_equality (ap1 ap2 : ArithmeticProgression) :
  (∀ n : ℕ, 
    (ArithmeticProgression.nthTerm ap1 n ^ 2 + ArithmeticProgression.nthTerm ap1 (n + 1) ^ 2) *
    (ArithmeticProgression.nthTerm ap2 n ^ 2 + ArithmeticProgression.nthTerm ap2 (n + 1) ^ 2) = m ^ 2 ∨
    (ArithmeticProgression.nthTerm ap1 n ^ 2 + ArithmeticProgression.nthTerm ap2 n ^ 2) *
    (ArithmeticProgression.nthTerm ap1 (n + 1) ^ 2 + ArithmeticProgression.nthTerm ap2 (n + 1) ^ 2) = k ^ 2) →
  ∀ n : ℕ, ArithmeticProgression.nthTerm ap1 n = ArithmeticProgression.nthTerm ap2 n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equality_l2236_223606


namespace NUMINAMATH_CALUDE_sum_always_negative_l2236_223623

/-- The function f(x) = -x - x^3 -/
def f (x : ℝ) : ℝ := -x - x^3

/-- Theorem stating that f(α) + f(β) + f(γ) is always negative under given conditions -/
theorem sum_always_negative (α β γ : ℝ) 
  (h1 : α + β > 0) (h2 : β + γ > 0) (h3 : γ + α > 0) : 
  f α + f β + f γ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_always_negative_l2236_223623


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l2236_223647

theorem weekend_rain_probability (prob_saturday prob_sunday : ℝ) 
  (h1 : prob_saturday = 0.3)
  (h2 : prob_sunday = 0.6)
  (h3 : 0 ≤ prob_saturday ∧ prob_saturday ≤ 1)
  (h4 : 0 ≤ prob_sunday ∧ prob_sunday ≤ 1) :
  1 - (1 - prob_saturday) * (1 - prob_sunday) = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l2236_223647


namespace NUMINAMATH_CALUDE_probability_larger_than_40_l2236_223698

def digits : Finset Nat := {1, 2, 3, 4, 5}

def is_valid_selection (a b : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ a ≠ b

def is_larger_than_40 (a b : Nat) : Prop :=
  is_valid_selection a b ∧ 10 * a + b > 40

def total_selections : Nat :=
  digits.card * (digits.card - 1)

def favorable_selections : Nat :=
  (digits.filter (λ x => x ≥ 4)).card * (digits.card - 1)

theorem probability_larger_than_40 :
  (favorable_selections : ℚ) / total_selections = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_larger_than_40_l2236_223698


namespace NUMINAMATH_CALUDE_hannahs_speed_l2236_223616

/-- 
Given two drivers, Glen and Hannah, driving towards each other and then away,
prove that Hannah's speed is 15 km/h under the following conditions:
- Glen drives at a constant speed of 37 km/h
- They are 130 km apart at 6 am and 11 am
- They pass each other at some point between 6 am and 11 am
-/
theorem hannahs_speed 
  (glen_speed : ℝ) 
  (initial_distance final_distance : ℝ)
  (time_interval : ℝ) :
  glen_speed = 37 →
  initial_distance = 130 →
  final_distance = 130 →
  time_interval = 5 →
  ∃ (hannah_speed : ℝ), hannah_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_speed_l2236_223616


namespace NUMINAMATH_CALUDE_avery_donation_l2236_223680

theorem avery_donation (shirts : ℕ) 
  (h1 : shirts + 2 * shirts + shirts = 16) : shirts = 4 := by
  sorry

end NUMINAMATH_CALUDE_avery_donation_l2236_223680


namespace NUMINAMATH_CALUDE_junior_score_l2236_223607

theorem junior_score (n : ℝ) (junior_score : ℝ) :
  n > 0 →
  0.15 * n * junior_score + 0.85 * n * 87 = n * 88 →
  junior_score = 94 := by
  sorry

end NUMINAMATH_CALUDE_junior_score_l2236_223607


namespace NUMINAMATH_CALUDE_quadratic_function_a_range_l2236_223625

/-- The function y = (a + 1)x^2 - 2x + 3 is quadratic with respect to x -/
def is_quadratic (a : ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, y = (a + 1) * x^2 - 2 * x + 3

/-- The range of values for a in the quadratic function y = (a + 1)x^2 - 2x + 3 -/
theorem quadratic_function_a_range :
  ∀ a : ℝ, is_quadratic a ↔ a ≠ -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_a_range_l2236_223625


namespace NUMINAMATH_CALUDE_triangle_angle_from_complex_trig_l2236_223666

theorem triangle_angle_from_complex_trig (A B C : Real) : 
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  A + B + C = π →
  (Complex.exp (I * A)) * (Complex.exp (I * B)) = Complex.exp (I * C) →
  C = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_from_complex_trig_l2236_223666


namespace NUMINAMATH_CALUDE_alloy_mixture_chromium_balance_l2236_223664

/-- Represents the composition of an alloy mixture -/
structure AlloyMixture where
  first_alloy_amount : ℝ
  first_alloy_chromium_percent : ℝ
  second_alloy_amount : ℝ
  second_alloy_chromium_percent : ℝ
  new_alloy_chromium_percent : ℝ

/-- The alloy mixture satisfies the chromium balance equation -/
def satisfies_chromium_balance (mixture : AlloyMixture) : Prop :=
  mixture.first_alloy_chromium_percent * mixture.first_alloy_amount +
  mixture.second_alloy_chromium_percent * mixture.second_alloy_amount =
  mixture.new_alloy_chromium_percent * (mixture.first_alloy_amount + mixture.second_alloy_amount)

/-- Theorem: The alloy mixture satisfies the chromium balance equation -/
theorem alloy_mixture_chromium_balance 
  (mixture : AlloyMixture)
  (h1 : mixture.second_alloy_amount = 35)
  (h2 : mixture.second_alloy_chromium_percent = 0.08)
  (h3 : mixture.new_alloy_chromium_percent = 0.101) :
  satisfies_chromium_balance mixture :=
sorry

end NUMINAMATH_CALUDE_alloy_mixture_chromium_balance_l2236_223664


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a10_l2236_223672

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a10
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a3 : a 3 = 12)
  (h_a6 : a 6 = 27) :
  a 10 = 47 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a10_l2236_223672


namespace NUMINAMATH_CALUDE_counterexample_exists_l2236_223629

theorem counterexample_exists : ∃ n : ℝ, n < 1 ∧ n^2 - 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2236_223629


namespace NUMINAMATH_CALUDE_double_reflection_F_l2236_223640

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point F -/
def F : ℝ × ℝ := (-2, -3)

theorem double_reflection_F :
  (reflect_x (reflect_y F)) = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_F_l2236_223640


namespace NUMINAMATH_CALUDE_purchase_problem_l2236_223618

theorem purchase_problem (a b c : ℕ) : 
  a + b + c = 50 →
  60 * a + 500 * b + 400 * c = 10000 →
  a = 30 :=
by sorry

end NUMINAMATH_CALUDE_purchase_problem_l2236_223618


namespace NUMINAMATH_CALUDE_sum_of_integers_l2236_223699

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 100)
  (h2 : x * y = 32) : 
  x + y = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2236_223699


namespace NUMINAMATH_CALUDE_equation_equivalence_l2236_223617

theorem equation_equivalence (a : ℝ) : (a - 1)^2 = a^3 - 2*a + 1 ↔ a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2236_223617


namespace NUMINAMATH_CALUDE_problem_statement_l2236_223670

theorem problem_statement (x : ℝ) (h : x + 1/x = 5) :
  (x - 2)^2 + 25/((x - 2)^2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2236_223670


namespace NUMINAMATH_CALUDE_no_natural_number_power_of_two_l2236_223652

theorem no_natural_number_power_of_two : 
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), n^2012 - 1 = 2^k := by
  sorry

end NUMINAMATH_CALUDE_no_natural_number_power_of_two_l2236_223652


namespace NUMINAMATH_CALUDE_chocolate_price_in_first_store_l2236_223621

def chocolates_per_week : ℕ := 2
def weeks : ℕ := 3
def promotion_price : ℚ := 2
def savings : ℚ := 6

theorem chocolate_price_in_first_store :
  let total_chocolates := chocolates_per_week * weeks
  let promotion_total := total_chocolates * promotion_price
  let first_store_total := promotion_total + savings
  first_store_total / total_chocolates = 3 := by
sorry

end NUMINAMATH_CALUDE_chocolate_price_in_first_store_l2236_223621


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2236_223612

/-- Represents a cube structure made of unit cubes -/
structure CubeStructure where
  side_length : ℕ
  removed_cubes : ℕ

/-- Calculates the volume of the cube structure -/
def volume (c : CubeStructure) : ℕ :=
  c.side_length^3 - c.removed_cubes

/-- Calculates the surface area of the cube structure -/
def surface_area (c : CubeStructure) : ℕ :=
  6 * c.side_length^2 - 4 * c.removed_cubes

/-- The specific cube structure described in the problem -/
def hollow_cube : CubeStructure :=
  { side_length := 3
  , removed_cubes := 1 }

/-- Theorem stating the ratio of volume to surface area for the hollow cube -/
theorem volume_to_surface_area_ratio :
  (volume hollow_cube : ℚ) / (surface_area hollow_cube : ℚ) = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2236_223612
