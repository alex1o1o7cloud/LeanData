import Mathlib

namespace range_of_m_l2855_285532

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

-- Define the interval [2, 4]
def I : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (∃ x ∈ I, m - f x > 0) → m > 5 := by sorry

end range_of_m_l2855_285532


namespace data_average_is_four_l2855_285556

def data : List ℝ := [6, 3, 3, 5, 1]

def isMode (x : ℝ) (l : List ℝ) : Prop :=
  ∀ y ∈ l, (l.count x ≥ l.count y)

theorem data_average_is_four (x : ℝ) (h1 : isMode 3 (x::data)) (h2 : isMode 6 (x::data)) :
  (x::data).sum / (x::data).length = 4 := by
  sorry

end data_average_is_four_l2855_285556


namespace negation_of_forall_exp_positive_l2855_285515

theorem negation_of_forall_exp_positive :
  (¬ ∀ x : ℝ, Real.exp x > 0) ↔ (∃ x : ℝ, Real.exp x ≤ 0) := by
  sorry

end negation_of_forall_exp_positive_l2855_285515


namespace point_B_in_third_quadrant_l2855_285597

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the third quadrant -/
def is_in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A(m, -n) is in the second quadrant, then B(-mn, m) is in the third quadrant -/
theorem point_B_in_third_quadrant 
  (m n : ℝ) 
  (h : is_in_second_quadrant ⟨m, -n⟩) : 
  is_in_third_quadrant ⟨-m*n, m⟩ := by
  sorry

end point_B_in_third_quadrant_l2855_285597


namespace binary_matrix_sum_theorem_l2855_285587

/-- A 5x5 matrix with entries 0 or 1 -/
def BinaryMatrix := Matrix (Fin 5) (Fin 5) Bool

/-- Get the 24 sequences from a BinaryMatrix as specified in the problem -/
def getSequences (X : BinaryMatrix) : Finset (List Bool) := sorry

/-- The sum of all entries in a BinaryMatrix -/
def matrixSum (X : BinaryMatrix) : ℕ := sorry

/-- Main theorem -/
theorem binary_matrix_sum_theorem (X : BinaryMatrix) :
  (getSequences X).card = 24 → matrixSum X = 12 ∨ matrixSum X = 13 := by sorry

end binary_matrix_sum_theorem_l2855_285587


namespace mashed_potatoes_vs_bacon_difference_l2855_285553

/-- The number of students who suggested adding bacon to the menu. -/
def bacon_students : ℕ := 269

/-- The number of students who suggested adding mashed potatoes to the menu. -/
def mashed_potatoes_students : ℕ := 330

/-- The number of students who suggested adding tomatoes to the menu. -/
def tomato_students : ℕ := 76

/-- The theorem states that the difference between the number of students who suggested
    mashed potatoes and the number of students who suggested bacon is 61. -/
theorem mashed_potatoes_vs_bacon_difference :
  mashed_potatoes_students - bacon_students = 61 := by
  sorry

end mashed_potatoes_vs_bacon_difference_l2855_285553


namespace factor_of_polynomial_l2855_285513

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^6 + 8 : ℝ) = (x^2 + 2) * q x := by
sorry

end factor_of_polynomial_l2855_285513


namespace seat_difference_is_two_l2855_285561

/-- Represents an airplane with first-class and coach class seats. -/
structure Airplane where
  total_seats : ℕ
  coach_seats : ℕ
  first_class_seats : ℕ
  h1 : total_seats = first_class_seats + coach_seats
  h2 : coach_seats > 4 * first_class_seats

/-- The difference between coach seats and 4 times first-class seats. -/
def seat_difference (a : Airplane) : ℕ :=
  a.coach_seats - 4 * a.first_class_seats

/-- Theorem stating the seat difference for a specific airplane configuration. -/
theorem seat_difference_is_two (a : Airplane)
  (h3 : a.total_seats = 387)
  (h4 : a.coach_seats = 310) :
  seat_difference a = 2 := by
  sorry


end seat_difference_is_two_l2855_285561


namespace bus_students_count_l2855_285546

/-- Calculates the number of students on the bus after all stops -/
def final_students (initial : ℕ) (second_on second_off third_on third_off : ℕ) : ℕ :=
  initial + second_on - second_off + third_on - third_off

/-- Theorem stating the final number of students on the bus -/
theorem bus_students_count :
  final_students 39 29 12 35 18 = 73 := by
  sorry

end bus_students_count_l2855_285546


namespace polygon_120_degree_angle_l2855_285525

/-- A triangular grid of equilateral triangles with unit sides -/
structure TriangularGrid where
  -- Add necessary fields here

/-- A non-self-intersecting polygon on a triangular grid -/
structure Polygon (grid : TriangularGrid) where
  vertices : List (ℕ × ℕ)
  is_non_self_intersecting : Bool
  perimeter : ℕ

/-- Checks if a polygon has a 120-degree angle (internal or external) -/
def has_120_degree_angle (grid : TriangularGrid) (p : Polygon grid) : Prop :=
  sorry

theorem polygon_120_degree_angle 
  (grid : TriangularGrid) 
  (p : Polygon grid) 
  (h1 : p.is_non_self_intersecting = true) 
  (h2 : p.perimeter = 1399) : 
  has_120_degree_angle grid p := by
  sorry

end polygon_120_degree_angle_l2855_285525


namespace expression_simplification_l2855_285572

theorem expression_simplification (x : ℝ) :
  (3*x^3 + 4*x^2 + 5)*(2*x - 1) - (2*x - 1)*(x^2 + 2*x - 8) + (x^2 - 2*x + 3)*(2*x - 1)*(x - 2) =
  8*x^4 - 2*x^3 - 5*x^2 + 32*x - 15 := by
  sorry

end expression_simplification_l2855_285572


namespace cubic_sum_of_quadratic_roots_l2855_285582

theorem cubic_sum_of_quadratic_roots : 
  ∀ a b : ℝ, 
  (3 * a^2 - 5 * a + 7 = 0) → 
  (3 * b^2 - 5 * b + 7 = 0) → 
  (a ≠ b) →
  (a^3 / b^3 + b^3 / a^3 = -190 / 343) := by
sorry

end cubic_sum_of_quadratic_roots_l2855_285582


namespace expand_and_compare_coefficients_l2855_285564

theorem expand_and_compare_coefficients (m n : ℤ) : 
  (∀ x : ℤ, (x + 4) * (x - 2) = x^2 + m*x + n) → m = 2 ∧ n = -8 := by
  sorry

end expand_and_compare_coefficients_l2855_285564


namespace two_books_from_different_genres_l2855_285528

/-- Represents the number of books in each genre -/
def booksPerGenre : Nat := 4

/-- Represents the number of genres -/
def numGenres : Nat := 3

/-- Theorem: The number of ways to select two books from different genres 
    given three genres with four books each is 48 -/
theorem two_books_from_different_genres :
  (booksPerGenre * booksPerGenre * (numGenres * (numGenres - 1) / 2)) = 48 := by
  sorry

end two_books_from_different_genres_l2855_285528


namespace fifteenth_term_geometric_sequence_l2855_285559

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

theorem fifteenth_term_geometric_sequence :
  geometric_sequence 12 (1/3) 15 = 4/1594323 := by
  sorry

end fifteenth_term_geometric_sequence_l2855_285559


namespace smallest_n_for_array_formation_l2855_285535

theorem smallest_n_for_array_formation : 
  ∃ n k : ℕ+, 
    (∀ m k' : ℕ+, 8 * m = 225 * k' + 3 → n ≤ m) ∧ 
    (8 * n = 225 * k + 3) ∧
    n = 141 := by
  sorry

end smallest_n_for_array_formation_l2855_285535


namespace three_digit_equation_solution_l2855_285569

/-- Given that 3 + 6AB = 691 and 6AB is a three-digit number, prove that A = 8 -/
theorem three_digit_equation_solution (A B : ℕ) : 
  (3 + 6 * A * 10 + B = 691) → 
  (100 ≤ 6 * A * 10 + B) →
  (6 * A * 10 + B < 1000) →
  A = 8 := by
  sorry

end three_digit_equation_solution_l2855_285569


namespace tea_brewing_time_proof_l2855_285586

/-- The time needed to wash the kettle and fill it with cold water -/
def wash_kettle_time : ℕ := 2

/-- The time needed to wash the teapot and cups -/
def wash_teapot_cups_time : ℕ := 2

/-- The time needed to get tea leaves -/
def get_tea_leaves_time : ℕ := 1

/-- The time needed to boil water -/
def boil_water_time : ℕ := 15

/-- The time needed to brew the tea -/
def brew_tea_time : ℕ := 1

/-- The shortest operation time for brewing a pot of tea -/
def shortest_operation_time : ℕ := 18

theorem tea_brewing_time_proof :
  shortest_operation_time = max wash_kettle_time (max boil_water_time brew_tea_time) :=
by sorry

end tea_brewing_time_proof_l2855_285586


namespace africa_fraction_proof_l2855_285523

def total_passengers : ℕ := 96

def north_america_fraction : ℚ := 1/4
def europe_fraction : ℚ := 1/8
def asia_fraction : ℚ := 1/6
def other_continents : ℕ := 36

theorem africa_fraction_proof :
  ∃ (africa_fraction : ℚ),
    africa_fraction * total_passengers +
    north_america_fraction * total_passengers +
    europe_fraction * total_passengers +
    asia_fraction * total_passengers +
    other_continents = total_passengers ∧
    africa_fraction = 1/12 :=
by sorry

end africa_fraction_proof_l2855_285523


namespace edward_board_game_cost_l2855_285552

def board_game_cost (total_cost : ℕ) (num_figures : ℕ) (figure_cost : ℕ) : ℕ :=
  total_cost - (num_figures * figure_cost)

theorem edward_board_game_cost :
  board_game_cost 30 4 7 = 2 := by
  sorry

end edward_board_game_cost_l2855_285552


namespace negation_of_existence_l2855_285543

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ P x) ↔ (∀ x : ℝ, x > 0 → ¬ P x) := by sorry

end negation_of_existence_l2855_285543


namespace elephant_park_problem_l2855_285590

theorem elephant_park_problem (initial_elephants : ℕ) (exodus_duration : ℕ) (exodus_rate : ℕ) 
  (entry_period : ℕ) (final_elephants : ℕ) : 
  initial_elephants = 30000 →
  exodus_duration = 4 →
  exodus_rate = 2880 →
  entry_period = 7 →
  final_elephants = 28980 →
  (final_elephants - (initial_elephants - exodus_duration * exodus_rate)) / entry_period = 1500 := by
sorry

end elephant_park_problem_l2855_285590


namespace andy_sock_ratio_l2855_285507

/-- The ratio of white socks to black socks -/
def sock_ratio (white : ℕ) (black : ℕ) : ℚ := white / black

theorem andy_sock_ratio :
  ∀ white : ℕ,
  let black := 6
  white / 2 = black + 6 →
  sock_ratio white black = 4 / 1 := by
sorry

end andy_sock_ratio_l2855_285507


namespace dividend_problem_l2855_285594

/-- Given a total amount of 585 to be divided among three people (a, b, c) such that
    4 times a's share equals 6 times b's share, which equals 3 times c's share,
    prove that c's share is equal to 135. -/
theorem dividend_problem (total : ℕ) (a b c : ℚ) 
    (h_total : total = 585)
    (h_sum : a + b + c = total)
    (h_prop : (4 * a = 6 * b) ∧ (6 * b = 3 * c)) :
  c = 135 := by
  sorry

end dividend_problem_l2855_285594


namespace math_exam_problem_l2855_285568

theorem math_exam_problem (total : ℕ) (correct : ℕ) (incorrect : ℕ) :
  total = 120 →
  incorrect = 3 * correct →
  total = correct + incorrect →
  correct = 30 := by
sorry

end math_exam_problem_l2855_285568


namespace b_win_probability_l2855_285557

/-- Represents the outcome of a single die roll -/
def DieRoll := Fin 6

/-- Represents the state of the game after each roll -/
structure GameState where
  rolls : List DieRoll
  turn : Bool  -- true for A's turn, false for B's turn

/-- Checks if a number is a multiple of 2 -/
def isMultipleOf2 (n : ℕ) : Bool := n % 2 = 0

/-- Checks if a number is a multiple of 3 -/
def isMultipleOf3 (n : ℕ) : Bool := n % 3 = 0

/-- Sums the last n rolls in the game state -/
def sumLastNRolls (state : GameState) (n : ℕ) : ℕ :=
  (state.rolls.take n).map (fun x => x.val + 1) |>.sum

/-- Determines if the game has ended and who the winner is -/
def gameResult (state : GameState) : Option Bool :=
  if state.rolls.length < 2 then
    none
  else if state.rolls.length < 3 then
    if isMultipleOf3 (sumLastNRolls state 2) then some false else none
  else
    let lastThreeSum := sumLastNRolls state 3
    let lastTwoSum := sumLastNRolls state 2
    if isMultipleOf2 lastThreeSum && !isMultipleOf3 lastTwoSum then
      some true  -- A wins
    else if isMultipleOf3 lastTwoSum && !isMultipleOf2 lastThreeSum then
      some false  -- B wins
    else
      none  -- Game continues

/-- The probability that player B wins the game -/
def probabilityBWins : ℚ := 5/9

theorem b_win_probability :
  probabilityBWins = 5/9 := by sorry

end b_win_probability_l2855_285557


namespace intersection_of_A_and_B_l2855_285595

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l2855_285595


namespace roses_in_vase_l2855_285599

theorem roses_in_vase (initial_roses : ℕ) (added_roses : ℕ) (total_roses : ℕ) : 
  added_roses = 11 → total_roses = 14 → initial_roses = 3 :=
by
  sorry

end roses_in_vase_l2855_285599


namespace largest_prime_factor_of_S_l2855_285558

/-- The product of non-zero digits of a positive integer -/
def p (n : ℕ+) : ℕ :=
  sorry

/-- The sum of p(n) from 1 to 999 -/
def S : ℕ :=
  (Finset.range 999).sum (λ i => p ⟨i + 1, Nat.succ_pos i⟩)

/-- The largest prime factor of S -/
theorem largest_prime_factor_of_S :
  ∃ (q : ℕ), Nat.Prime q ∧ q ∣ S ∧ ∀ (p : ℕ), Nat.Prime p → p ∣ S → p ≤ q :=
  sorry

end largest_prime_factor_of_S_l2855_285558


namespace colleen_pays_more_than_joy_l2855_285534

/-- Calculates the difference in cost between Colleen's and Joy's pencils -/
def pencil_cost_difference (joy_pencils colleen_pencils pencil_price : ℕ) : ℕ :=
  colleen_pencils * pencil_price - joy_pencils * pencil_price

theorem colleen_pays_more_than_joy :
  pencil_cost_difference 30 50 4 = 80 := by
  sorry

end colleen_pays_more_than_joy_l2855_285534


namespace arithmetic_sequence_implication_l2855_285575

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_implication
  (a b : ℕ → ℝ)
  (h : ∀ n : ℕ, b n = a n + a (n + 1)) :
  is_arithmetic_sequence a → is_arithmetic_sequence b ∧
  ¬(is_arithmetic_sequence b → is_arithmetic_sequence a) :=
sorry

end arithmetic_sequence_implication_l2855_285575


namespace real_part_of_complex_square_l2855_285540

theorem real_part_of_complex_square : Complex.re ((1 + 2 * Complex.I) ^ 2) = -3 := by
  sorry

end real_part_of_complex_square_l2855_285540


namespace shortest_chord_length_l2855_285539

/-- The circle with equation x^2 + y^2 - 2x - 4y + 1 = 0 -/
def circle1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 + 1 = 0}

/-- The center of circle1 -/
def center1 : ℝ × ℝ := (1, 2)

/-- The line of symmetry for circle1 -/
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t : ℝ, p = t • center1}

/-- The circle with center (0,0) and radius 3 -/
def circle2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 9}

/-- The shortest chord length theorem -/
theorem shortest_chord_length :
  ∃ (A B : ℝ × ℝ), A ∈ circle2 ∧ B ∈ circle2 ∧ A ∈ l ∧ B ∈ l ∧
    ∀ (C D : ℝ × ℝ), C ∈ circle2 → D ∈ circle2 → C ∈ l → D ∈ l →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 :=
sorry

end shortest_chord_length_l2855_285539


namespace tangent_identity_l2855_285519

theorem tangent_identity (α β : ℝ) 
  (h1 : Real.tan (α + β) ≠ 0) (h2 : Real.tan (α - β) ≠ 0) :
  (Real.tan α + Real.tan β) / Real.tan (α + β) + 
  (Real.tan α - Real.tan β) / Real.tan (α - β) + 
  2 * (Real.tan α)^2 = 2 / (Real.cos α)^2 := by
sorry

end tangent_identity_l2855_285519


namespace problem_solution_l2855_285588

def f (x : ℝ) : ℝ := |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x < |x| + 1 ↔ 0 < x ∧ x < 2) ∧
  (∀ x y : ℝ, |x - y - 1| ≤ 1/3 ∧ |2*y + 1| ≤ 1/6 → f x < 1) := by
  sorry

end problem_solution_l2855_285588


namespace perimeter_is_twentyone_l2855_285521

/-- A figure with 3 vertices where the distance between any 2 vertices is 7 -/
structure ThreeVertexFigure where
  vertices : Fin 3 → ℝ × ℝ
  distance_eq_seven : ∀ (i j : Fin 3), i ≠ j → Real.sqrt ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 7

/-- The perimeter of a ThreeVertexFigure is 21 -/
theorem perimeter_is_twentyone (f : ThreeVertexFigure) : 
  (Real.sqrt ((f.vertices 0).1 - (f.vertices 1).1)^2 + ((f.vertices 0).2 - (f.vertices 1).2)^2) +
  (Real.sqrt ((f.vertices 1).1 - (f.vertices 2).1)^2 + ((f.vertices 1).2 - (f.vertices 2).2)^2) +
  (Real.sqrt ((f.vertices 2).1 - (f.vertices 0).1)^2 + ((f.vertices 2).2 - (f.vertices 0).2)^2) = 21 :=
by sorry

end perimeter_is_twentyone_l2855_285521


namespace a_work_days_proof_a_work_days_unique_l2855_285518

-- Define the work rates and completion times
def total_work : ℝ := 1 -- Normalize total work to 1
def a_completion_time : ℝ := 15
def b_completion_time : ℝ := 26.999999999999996
def b_remaining_time : ℝ := 18

-- Define A's work days as a variable
def a_work_days : ℝ := 5 -- The value we want to prove

-- Theorem statement
theorem a_work_days_proof :
  (total_work / a_completion_time) * a_work_days +
  (total_work / b_completion_time) * b_remaining_time = total_work :=
by
  sorry -- Proof is omitted as per instructions

-- Additional theorem to show that this solution is unique
theorem a_work_days_unique (x : ℝ) :
  (total_work / a_completion_time) * x +
  (total_work / b_completion_time) * b_remaining_time = total_work →
  x = a_work_days :=
by
  sorry -- Proof is omitted as per instructions

end a_work_days_proof_a_work_days_unique_l2855_285518


namespace mango_profit_percentage_l2855_285508

/-- Represents the rate at which mangoes are bought (number of mangoes per rupee) -/
def buy_rate : ℚ := 6

/-- Represents the rate at which mangoes are sold (number of mangoes per rupee) -/
def sell_rate : ℚ := 3

/-- Calculates the profit percentage given buy and sell rates -/
def profit_percentage (buy : ℚ) (sell : ℚ) : ℚ :=
  ((sell⁻¹ - buy⁻¹) / buy⁻¹) * 100

theorem mango_profit_percentage :
  profit_percentage buy_rate sell_rate = 100 := by
  sorry

end mango_profit_percentage_l2855_285508


namespace shorter_more_frequent_steps_slower_l2855_285529

/-- Represents a tourist's walking characteristics -/
structure Tourist where
  step_length : ℝ
  step_count : ℕ

/-- Calculates the distance covered by a tourist -/
def distance_covered (t : Tourist) : ℝ := t.step_length * t.step_count

/-- Theorem stating that the tourist with shorter and more frequent steps is slower -/
theorem shorter_more_frequent_steps_slower (t1 t2 : Tourist) 
  (h1 : t1.step_length < t2.step_length) 
  (h2 : t1.step_count > t2.step_count) 
  (h3 : t1.step_length * t1.step_count < t2.step_length * t2.step_count) : 
  distance_covered t1 < distance_covered t2 := by
  sorry

#check shorter_more_frequent_steps_slower

end shorter_more_frequent_steps_slower_l2855_285529


namespace intersection_P_Q_l2855_285549

def P : Set ℝ := {x | -x^2 + 3*x + 4 < 0}
def Q : Set ℝ := {x | 2*x - 5 > 0}

theorem intersection_P_Q : P ∩ Q = {x | x > 4} := by sorry

end intersection_P_Q_l2855_285549


namespace product_of_x_values_l2855_285520

theorem product_of_x_values (x : ℝ) : 
  (|18 / x - 6| = 3) → (∃ y : ℝ, y ≠ x ∧ |18 / y - 6| = 3 ∧ x * y = 12) :=
by sorry

end product_of_x_values_l2855_285520


namespace hot_dogs_leftover_l2855_285584

theorem hot_dogs_leftover : 20146130 % 6 = 2 := by
  sorry

end hot_dogs_leftover_l2855_285584


namespace sum_of_w_and_y_is_three_l2855_285550

theorem sum_of_w_and_y_is_three :
  ∀ (W X Y Z : ℕ),
    W ∈ ({1, 2, 3, 4} : Set ℕ) →
    X ∈ ({1, 2, 3, 4} : Set ℕ) →
    Y ∈ ({1, 2, 3, 4} : Set ℕ) →
    Z ∈ ({1, 2, 3, 4} : Set ℕ) →
    W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
    (W : ℚ) / X + (Y : ℚ) / Z = 1 →
    W + Y = 3 := by
  sorry

end sum_of_w_and_y_is_three_l2855_285550


namespace fraction_calculation_l2855_285524

theorem fraction_calculation : (5 / 6 : ℚ) / (9 / 10) - 1 / 15 = 116 / 135 := by sorry

end fraction_calculation_l2855_285524


namespace scientific_notation_10870_l2855_285504

theorem scientific_notation_10870 :
  10870 = 1.087 * (10 ^ 4) := by
  sorry

end scientific_notation_10870_l2855_285504


namespace inscribed_quadrilateral_angle_measure_l2855_285517

-- Define the circle O
variable (O : ℝ × ℝ)

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define that ABCD is an inscribed quadrilateral of circle O
def is_inscribed_quadrilateral (O A B C D : ℝ × ℝ) : Prop :=
  sorry

-- Define the angle measure function
def angle_measure (P Q R : ℝ × ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem inscribed_quadrilateral_angle_measure 
  (h_inscribed : is_inscribed_quadrilateral O A B C D)
  (h_ratio : angle_measure B A D / angle_measure B C D = 4 / 5) :
  angle_measure B A D = 80 :=
sorry

end inscribed_quadrilateral_angle_measure_l2855_285517


namespace waiter_customers_theorem_l2855_285503

def final_customers (initial new left : ℕ) : ℕ :=
  initial - left + new

theorem waiter_customers_theorem (initial new left : ℕ) 
  (h1 : initial ≥ left) : 
  final_customers initial new left = initial - left + new :=
by
  sorry

end waiter_customers_theorem_l2855_285503


namespace vector_difference_magnitude_l2855_285576

/-- Given vectors a and b in ℝ², prove that the magnitude of their difference is 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  a = (2, 1) →
  b = (-2, 4) →
  ‖a - b‖ = 5 := by
  sorry

end vector_difference_magnitude_l2855_285576


namespace inscribed_right_triangle_diameter_l2855_285598

/-- Given a right triangle inscribed in a circle with legs of lengths 6 and 8,
    the diameter of the circle is 10. -/
theorem inscribed_right_triangle_diameter :
  ∀ (circle : Real → Real → Prop) (triangle : Real → Real → Real → Prop),
    (∃ (x y z : Real), triangle x y z ∧ x^2 + y^2 = z^2) →  -- Right triangle condition
    (∃ (a b : Real), triangle 6 8 a) →  -- Leg lengths condition
    (∀ (p q r : Real), triangle p q r → circle p q) →  -- Triangle inscribed in circle
    (∃ (d : Real), d = 10 ∧ ∀ (p q : Real), circle p q → (p - q)^2 ≤ d^2) :=
by sorry

end inscribed_right_triangle_diameter_l2855_285598


namespace school_travel_time_l2855_285505

/-- 
If a boy reaches school 4 minutes earlier when walking at 9/8 of his usual rate,
then his usual time to reach the school is 36 minutes.
-/
theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) 
  (h : usual_rate * usual_time = (9/8 * usual_rate) * (usual_time - 4)) :
  usual_time = 36 := by
  sorry

end school_travel_time_l2855_285505


namespace eighth_term_geometric_sequence_l2855_285512

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The 8th term of a geometric sequence given the 4th and 6th terms -/
theorem eighth_term_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_4 : a 4 = 7)
  (h_6 : a 6 = 21) : 
  a 8 = 63 := by
sorry

end eighth_term_geometric_sequence_l2855_285512


namespace cherry_tomato_jars_l2855_285592

theorem cherry_tomato_jars (total_tomatoes : ℕ) (tomatoes_per_jar : ℕ) (h1 : total_tomatoes = 56) (h2 : tomatoes_per_jar = 8) :
  (total_tomatoes / tomatoes_per_jar : ℕ) = 7 := by
  sorry

end cherry_tomato_jars_l2855_285592


namespace divisible_by_900_l2855_285547

theorem divisible_by_900 (n : ℕ) : ∃ k : ℤ, 6^(2*(n+1)) - 2^(n+3) * 3^(n+2) + 36 = 900 * k := by
  sorry

end divisible_by_900_l2855_285547


namespace ace_king_queen_probability_l2855_285530

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (aces : Nat)
  (kings : Nat)
  (queens : Nat)

/-- The probability of drawing a specific card from a deck -/
def drawProbability (n : Nat) (total : Nat) : ℚ :=
  n / total

/-- The standard 52-card deck -/
def standardDeck : Deck :=
  { cards := 52, aces := 4, kings := 4, queens := 4 }

theorem ace_king_queen_probability :
  let d := standardDeck
  let p1 := drawProbability d.aces d.cards
  let p2 := drawProbability d.kings (d.cards - 1)
  let p3 := drawProbability d.queens (d.cards - 2)
  p1 * p2 * p3 = 8 / 16575 := by
  sorry

end ace_king_queen_probability_l2855_285530


namespace max_value_less_than_two_l2855_285502

theorem max_value_less_than_two (m : ℝ) (hm1 : 1 < m) (hm2 : m < 1 + Real.sqrt 2) :
  ∀ x y : ℝ, y ≥ x → y ≤ m * x → x + y ≤ 1 → x + m * y < 2 := by
  sorry

#check max_value_less_than_two

end max_value_less_than_two_l2855_285502


namespace number_of_book_combinations_l2855_285577

-- Define the number of books and the number to choose
def total_books : ℕ := 15
def books_to_choose : ℕ := 3

-- Theorem statement
theorem number_of_book_combinations :
  Nat.choose total_books books_to_choose = 455 := by
  sorry

end number_of_book_combinations_l2855_285577


namespace greatest_common_multiple_9_15_under_120_l2855_285541

theorem greatest_common_multiple_9_15_under_120 :
  ∃ n : ℕ, n = 90 ∧ 
  (∀ m : ℕ, m < 120 → m % 9 = 0 → m % 15 = 0 → m ≤ n) ∧
  90 % 9 = 0 ∧ 90 % 15 = 0 ∧ 90 < 120 :=
by sorry

end greatest_common_multiple_9_15_under_120_l2855_285541


namespace rabbit_average_distance_l2855_285591

/-- A square with side length 8 meters -/
def square_side : ℝ := 8

/-- The x-coordinate of the rabbit's final position -/
def rabbit_x : ℝ := 6.4

/-- The y-coordinate of the rabbit's final position -/
def rabbit_y : ℝ := 2.4

/-- The average distance from the rabbit to the sides of the square -/
def average_distance : ℝ := 4

theorem rabbit_average_distance :
  let distances : List ℝ := [
    rabbit_x,  -- distance to left side
    rabbit_y,  -- distance to bottom side
    square_side - rabbit_x,  -- distance to right side
    square_side - rabbit_y   -- distance to top side
  ]
  (distances.sum / distances.length : ℝ) = average_distance := by
  sorry

end rabbit_average_distance_l2855_285591


namespace sqrt_D_irrational_l2855_285570

theorem sqrt_D_irrational (x : ℤ) : 
  let a : ℤ := x
  let b : ℤ := x + 2
  let c : ℤ := a * b
  let d : ℤ := b + c
  let D : ℤ := a^2 + b^2 + c^2 + d^2
  Irrational (Real.sqrt D) := by
  sorry

end sqrt_D_irrational_l2855_285570


namespace regular_polygon_exterior_angle_18_deg_has_20_sides_l2855_285555

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_deg_has_20_sides :
  ∀ (n : ℕ), 
  n > 0 → 
  (360 : ℝ) / n = 18 → 
  n = 20 := by
sorry

end regular_polygon_exterior_angle_18_deg_has_20_sides_l2855_285555


namespace always_two_real_roots_find_p_l2855_285565

-- Define the quadratic equation
def quadratic_equation (x p : ℝ) : Prop :=
  (x - 3) * (x - 2) = p * (p + 1)

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (p : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ p ∧ quadratic_equation x₂ p :=
sorry

-- Theorem 2: If the roots satisfy the given condition, then p = -2
theorem find_p (p : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation x₁ p)
  (h₂ : quadratic_equation x₂ p)
  (h₃ : x₁^2 + x₂^2 - x₁*x₂ = 3*p^2 + 1) :
  p = -2 :=
sorry

end always_two_real_roots_find_p_l2855_285565


namespace min_knights_in_village_l2855_285527

theorem min_knights_in_village (total_people : Nat) (total_statements : Nat) (liar_statements : Nat) :
  total_people = 7 →
  total_statements = total_people * (total_people - 1) →
  total_statements = 42 →
  liar_statements = 24 →
  ∃ (knights : Nat), knights ≥ 3 ∧ 
    knights + (total_people - knights) = total_people ∧
    2 * knights * (total_people - knights) = liar_statements :=
by sorry

end min_knights_in_village_l2855_285527


namespace monkey_climb_time_l2855_285544

/-- Represents the climbing process of a monkey on a tree. -/
structure MonkeyClimb where
  treeHeight : ℕ  -- Height of the tree in feet
  hopDistance : ℕ  -- Distance the monkey hops up each hour
  slipDistance : ℕ  -- Distance the monkey slips back each hour

/-- Calculates the time taken for the monkey to reach the top of the tree. -/
def timeToReachTop (climb : MonkeyClimb) : ℕ :=
  let netClimbPerHour := climb.hopDistance - climb.slipDistance
  let timeToReachNearTop := (climb.treeHeight - 1) / netClimbPerHour
  timeToReachNearTop + 1

/-- Theorem stating that for the given conditions, the monkey takes 19 hours to reach the top. -/
theorem monkey_climb_time :
  let climb : MonkeyClimb := { treeHeight := 19, hopDistance := 3, slipDistance := 2 }
  timeToReachTop climb = 19 := by
  sorry


end monkey_climb_time_l2855_285544


namespace extreme_values_and_tangent_lines_l2855_285551

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem extreme_values_and_tangent_lines :
  ∃ (a b : ℝ),
    (f' a b (-2/3) = 0 ∧ f' a b 1 = 0) ∧
    (a = -1/2 ∧ b = -2) ∧
    (∃ (t₁ t₂ : ℝ),
      (f a b t₁ - 1 = (f' a b t₁) * (-t₁) ∧ 2*t₁ + (f a b t₁) - 1 = 0) ∧
      (f a b t₂ - 1 = (f' a b t₂) * (-t₂) ∧ 33*t₂ + 16*(f a b t₂) - 16 = 0)) :=
by sorry

end extreme_values_and_tangent_lines_l2855_285551


namespace joan_makes_ten_ham_sandwiches_l2855_285581

/-- Represents the number of slices of cheese required for each type of sandwich -/
structure SandwichRecipe where
  ham_cheese_slices : ℕ
  grilled_cheese_slices : ℕ

/-- Represents the sandwich making scenario -/
structure SandwichScenario where
  recipe : SandwichRecipe
  total_cheese_slices : ℕ
  grilled_cheese_count : ℕ

/-- Calculates the number of ham sandwiches made -/
def ham_sandwiches_made (scenario : SandwichScenario) : ℕ :=
  (scenario.total_cheese_slices - scenario.grilled_cheese_count * scenario.recipe.grilled_cheese_slices) / scenario.recipe.ham_cheese_slices

/-- Theorem stating that Joan makes 10 ham sandwiches -/
theorem joan_makes_ten_ham_sandwiches (scenario : SandwichScenario) 
  (h1 : scenario.recipe.ham_cheese_slices = 2)
  (h2 : scenario.recipe.grilled_cheese_slices = 3)
  (h3 : scenario.total_cheese_slices = 50)
  (h4 : scenario.grilled_cheese_count = 10) :
  ham_sandwiches_made scenario = 10 := by
  sorry

#eval ham_sandwiches_made { 
  recipe := { ham_cheese_slices := 2, grilled_cheese_slices := 3 },
  total_cheese_slices := 50,
  grilled_cheese_count := 10
}

end joan_makes_ten_ham_sandwiches_l2855_285581


namespace least_n_for_phi_cube_l2855_285578

def phi (n : ℕ) : ℕ := sorry

theorem least_n_for_phi_cube (n : ℕ) : 
  (∀ m < n, phi (phi (phi m)) * phi (phi m) * phi m ≠ 64000) ∧ 
  (phi (phi (phi n)) * phi (phi n) * phi n = 64000) → 
  n = 41 := by sorry

end least_n_for_phi_cube_l2855_285578


namespace z_curve_not_simple_conic_l2855_285566

-- Define the complex number z
variable (z : ℂ)

-- Define the condition |z - 1/z| = 1
def condition (z : ℂ) : Prop := Complex.abs (z - z⁻¹) = 1

-- Define the curves we want to exclude
def is_ellipse (curve : ℂ → Prop) : Prop := sorry
def is_parabola (curve : ℂ → Prop) : Prop := sorry
def is_hyperbola (curve : ℂ → Prop) : Prop := sorry

-- Define the curve traced by z
def z_curve (z : ℂ) : Prop := condition z

-- State the theorem
theorem z_curve_not_simple_conic (z : ℂ) :
  condition z →
  ¬(is_ellipse z_curve ∨ is_parabola z_curve ∨ is_hyperbola z_curve) :=
sorry

end z_curve_not_simple_conic_l2855_285566


namespace shoe_price_is_50_l2855_285571

/-- Represents the original price of a pair of shoes -/
def original_shoe_price : ℝ := sorry

/-- Represents the discount rate for shoes -/
def shoe_discount : ℝ := 0.4

/-- Represents the discount rate for dresses -/
def dress_discount : ℝ := 0.2

/-- Represents the number of pairs of shoes bought -/
def num_shoes : ℕ := 2

/-- Represents the original price of the dress -/
def dress_price : ℝ := 100

/-- Represents the total amount spent -/
def total_spent : ℝ := 140

/-- Theorem stating that the original price of a pair of shoes is $50 -/
theorem shoe_price_is_50 : 
  (num_shoes : ℝ) * original_shoe_price * (1 - shoe_discount) + 
  dress_price * (1 - dress_discount) = total_spent → 
  original_shoe_price = 50 := by
  sorry

end shoe_price_is_50_l2855_285571


namespace solve_stamp_problem_l2855_285579

def stamp_problem (initial_stamps final_stamps mike_stamps : ℕ) : Prop :=
  let harry_stamps := final_stamps - initial_stamps - mike_stamps
  harry_stamps - 2 * mike_stamps = 10

theorem solve_stamp_problem :
  stamp_problem 3000 3061 17 := by
  sorry

end solve_stamp_problem_l2855_285579


namespace questions_to_complete_l2855_285593

/-- Calculates the number of questions Sasha still needs to complete -/
theorem questions_to_complete 
  (rate : ℕ)        -- Questions completed per hour
  (total : ℕ)       -- Total questions to complete
  (time_worked : ℕ) -- Hours worked
  (h1 : rate = 15)  -- Sasha's rate is 15 questions per hour
  (h2 : total = 60) -- Total questions is 60
  (h3 : time_worked = 2) -- Time worked is 2 hours
  : total - (rate * time_worked) = 30 :=
by sorry

end questions_to_complete_l2855_285593


namespace sandwich_problem_l2855_285533

theorem sandwich_problem (total : ℕ) (bologna : ℕ) (x : ℕ) :
  total = 80 →
  bologna = 35 →
  bologna = 7 * (total / (1 + 7 + x)) →
  x * (total / (1 + 7 + x)) = 40 :=
by sorry

end sandwich_problem_l2855_285533


namespace even_function_symmetry_l2855_285545

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_symmetry (f : ℝ → ℝ) (h : EvenFunction (fun x ↦ f (x + 1))) :
  ∀ x, f (1 + x) = f (1 - x) := by
  sorry

end even_function_symmetry_l2855_285545


namespace bug_path_tiles_l2855_285583

/-- Represents the number of tiles visited by a bug walking diagonally across a rectangular grid -/
def tilesVisited (width : ℕ) (length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- The playground dimensions in tile units -/
def playground_width : ℕ := 6
def playground_length : ℕ := 13

theorem bug_path_tiles :
  tilesVisited playground_width playground_length = 18 := by
  sorry

end bug_path_tiles_l2855_285583


namespace dumbbell_system_total_weight_l2855_285554

/-- The weight of a dumbbell system with three pairs of dumbbells -/
def dumbbell_system_weight (weight1 weight2 weight3 : ℕ) : ℕ :=
  2 * weight1 + 2 * weight2 + 2 * weight3

/-- Theorem: The weight of the specific dumbbell system is 32 lbs -/
theorem dumbbell_system_total_weight :
  dumbbell_system_weight 3 5 8 = 32 := by
  sorry

end dumbbell_system_total_weight_l2855_285554


namespace sequence_inequality_l2855_285560

theorem sequence_inequality (a : Fin 9 → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ a i) 
  (h_first : a 0 = 0) 
  (h_last : a 8 = 0) 
  (h_nonzero : ∃ i, a i ≠ 0) : 
  (∃ i : Fin 9, 1 < i.val ∧ i.val < 9 ∧ a (i - 1) + a (i + 1) < 2 * a i) ∧
  (∃ i : Fin 9, 1 < i.val ∧ i.val < 9 ∧ a (i - 1) + a (i + 1) < 1.9 * a i) :=
sorry

end sequence_inequality_l2855_285560


namespace abcd_equality_l2855_285516

theorem abcd_equality (a b c d : ℝ) 
  (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ 0)
  (h2 : a^2 + d^2 = 1)
  (h3 : b^2 + c^2 = 1)
  (h4 : a*c + b*d = 1/3) :
  a*b - c*d = 2*Real.sqrt 2/3 := by
sorry

end abcd_equality_l2855_285516


namespace rectangular_box_volume_l2855_285563

/-- The volume of a rectangular box with face areas 36, 18, and 8 square inches is 72 cubic inches. -/
theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 36)
  (area2 : w * h = 18)
  (area3 : l * h = 8) :
  l * w * h = 72 := by sorry

end rectangular_box_volume_l2855_285563


namespace unique_solution_for_n_l2855_285536

theorem unique_solution_for_n : ∃! (n : ℕ+), ∃ (x : ℕ+),
  n = 2^(2*x.val-1) - 5*x.val - 3 ∧
  n = (2^(x.val-1) - 1) * (2^x.val + 1) ∧
  n = 2015 := by
  sorry

end unique_solution_for_n_l2855_285536


namespace power_equality_l2855_285567

theorem power_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 := by sorry

end power_equality_l2855_285567


namespace solution_to_system_l2855_285562

theorem solution_to_system : 
  ∀ x y z : ℝ, 
  (4*x*y*z = (x+y)*(x*y+2) ∧ 
   4*x*y*z = (x+z)*(x*z+2) ∧ 
   4*x*y*z = (y+z)*(y*z+2)) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = Real.sqrt 2 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 2) ∨ 
   (x = -Real.sqrt 2 ∧ y = -Real.sqrt 2 ∧ z = -Real.sqrt 2)) :=
by sorry

end solution_to_system_l2855_285562


namespace correct_calculation_l2855_285500

theorem correct_calculation (x : ℝ) : 
  (x + 2.95 = 9.28) → (x - 2.95 = 3.38) :=
by sorry

end correct_calculation_l2855_285500


namespace polynomial_expansion_l2855_285537

theorem polynomial_expansion (t : ℝ) : 
  (3 * t^3 - 2 * t^2 + 4 * t - 1) * (-2 * t^2 + 3 * t + 6) = 
  -6 * t^5 + 5 * t^4 + 4 * t^3 + 22 * t^2 + 27 * t - 6 := by
  sorry

end polynomial_expansion_l2855_285537


namespace lending_problem_l2855_285522

/-- 
Proves that if A lends P amount to B at 10% per annum, B lends P to C at 14% per annum, 
and B's gain in 3 years is Rs. 420, then P = 3500.
-/
theorem lending_problem (P : ℝ) : 
  (P * (0.14 - 0.10) * 3 = 420) → P = 3500 := by
  sorry

end lending_problem_l2855_285522


namespace only_cone_cannot_have_rectangular_cross_section_l2855_285526

-- Define the types of geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | PentagonalPrism
  | Cube

-- Define a function that determines if a solid can have a rectangular cross-section
def canHaveRectangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => true
  | GeometricSolid.Cone => false
  | GeometricSolid.PentagonalPrism => true
  | GeometricSolid.Cube => true

-- Theorem statement
theorem only_cone_cannot_have_rectangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveRectangularCrossSection solid) ↔ solid = GeometricSolid.Cone :=
by sorry

end only_cone_cannot_have_rectangular_cross_section_l2855_285526


namespace cylinder_minus_cones_volume_l2855_285538

/-- The volume of a cylinder minus the volume of three congruent cones -/
theorem cylinder_minus_cones_volume (r h : ℝ) (hr : r = 8) (hh : h = 24) :
  π * r^2 * h - 3 * (1/3 * π * r^2 * (h/3)) = 1024 * π := by
  sorry

end cylinder_minus_cones_volume_l2855_285538


namespace heels_cost_equals_savings_plus_contribution_l2855_285574

/-- The cost of heels Miranda wants to buy -/
def heels_cost : ℕ := 260

/-- The number of months Miranda saved money -/
def months_saved : ℕ := 3

/-- The amount Miranda saved per month -/
def savings_per_month : ℕ := 70

/-- The amount Miranda's sister contributed -/
def sister_contribution : ℕ := 50

/-- Theorem stating that the cost of the heels is equal to Miranda's total savings plus her sister's contribution -/
theorem heels_cost_equals_savings_plus_contribution :
  heels_cost = months_saved * savings_per_month + sister_contribution :=
by sorry

end heels_cost_equals_savings_plus_contribution_l2855_285574


namespace monomial_exponent_difference_l2855_285596

theorem monomial_exponent_difference (a b : ℤ) : 
  ((-1 : ℚ) * X^3 * Y^1 = X^a * Y^(b-1)) → (a - b)^2022 = 1 := by
  sorry

end monomial_exponent_difference_l2855_285596


namespace complex_simplification_l2855_285514

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that the given complex expression simplifies to 5 -/
theorem complex_simplification : 2 * (3 - i) + i * (2 + i) = 5 := by
  sorry

end complex_simplification_l2855_285514


namespace imaginary_part_of_one_minus_i_squared_l2855_285510

theorem imaginary_part_of_one_minus_i_squared :
  Complex.im ((1 - Complex.I) ^ 2) = -2 := by
  sorry

end imaginary_part_of_one_minus_i_squared_l2855_285510


namespace complex_sum_zero_l2855_285511

theorem complex_sum_zero (z : ℂ) (h : z = Complex.exp (6 * Real.pi * I / 11)) :
  z^2 / (1 + z^3) + z^4 / (1 + z^6) + z^5 / (1 + z^9) = 0 := by
  sorry

end complex_sum_zero_l2855_285511


namespace taylor_family_reunion_l2855_285506

theorem taylor_family_reunion (kids : ℕ) (adults : ℕ) (tables : ℕ) 
  (h1 : kids = 45)
  (h2 : adults = 123)
  (h3 : tables = 14)
  : (kids + adults) / tables = 12 := by
  sorry

end taylor_family_reunion_l2855_285506


namespace triangle_side_length_l2855_285580

/-- Given a triangle ABC with angle A = 60°, side b = 8, and area = 12√3,
    prove that side a = 2√13 -/
theorem triangle_side_length (A B C : ℝ) (h_angle : A = 60 * π / 180)
    (h_side_b : B = 8) (h_area : (1/2) * B * C * Real.sin A = 12 * Real.sqrt 3) :
    A = 2 * Real.sqrt 13 := by
  sorry

end triangle_side_length_l2855_285580


namespace non_negative_integer_solutions_of_inequality_l2855_285531

theorem non_negative_integer_solutions_of_inequality : 
  {x : ℕ | -2 * (x : ℤ) > -4} = {0, 1} := by sorry

end non_negative_integer_solutions_of_inequality_l2855_285531


namespace or_true_implications_l2855_285548

theorem or_true_implications (p q : Prop) (h : p ∨ q) :
  ¬(
    ((p ∧ q) = True) ∨
    ((p ∧ q) = False) ∨
    ((¬p ∨ ¬q) = True) ∨
    ((¬p ∨ ¬q) = False)
  ) := by sorry

end or_true_implications_l2855_285548


namespace inequality_solution_l2855_285542

theorem inequality_solution (y : ℝ) : 
  (y^2 + y^3 - 3*y^4) / (y + y^2 - 3*y^3) ≥ -1 ↔ 
  y ∈ Set.Icc (-1) (-4/3) ∪ Set.Ioo (-4/3) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1 :=
by sorry

end inequality_solution_l2855_285542


namespace sin_cos_derivative_ratio_l2855_285589

theorem sin_cos_derivative_ratio (x : ℝ) (f : ℝ → ℝ) 
  (h1 : f = λ x => Real.sin x + Real.cos x)
  (h2 : deriv f = λ x => 3 * f x) :
  (Real.sin x)^2 - 3 / ((Real.cos x)^2 + 1) = -14/9 := by
  sorry

end sin_cos_derivative_ratio_l2855_285589


namespace function_convexity_concavity_l2855_285573

-- Function convexity/concavity theorem
theorem function_convexity_concavity :
  -- x² is convex everywhere
  (∀ (x₁ x₂ q₁ q₂ : ℝ), q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * x₁^2 + q₂ * x₂^2 - (q₁ * x₁ + q₂ * x₂)^2 ≥ 0) ∧
  -- √x is concave everywhere
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ > 0 → x₂ > 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * Real.sqrt x₁ + q₂ * Real.sqrt x₂ - Real.sqrt (q₁ * x₁ + q₂ * x₂) ≤ 0) ∧
  -- x³ is convex for x > 0 and concave for x < 0
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ > 0 → x₂ > 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * x₁^3 + q₂ * x₂^3 - (q₁ * x₁ + q₂ * x₂)^3 ≥ 0) ∧
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ < 0 → x₂ < 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * x₁^3 + q₂ * x₂^3 - (q₁ * x₁ + q₂ * x₂)^3 ≤ 0) ∧
  -- 1/x is convex for x > 0 and concave for x < 0
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ > 0 → x₂ > 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ / x₁ + q₂ / x₂ - 1 / (q₁ * x₁ + q₂ * x₂) ≥ 0) ∧
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ < 0 → x₂ < 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ / x₁ + q₂ / x₂ - 1 / (q₁ * x₁ + q₂ * x₂) ≤ 0) :=
by sorry

end function_convexity_concavity_l2855_285573


namespace furniture_shop_cost_price_l2855_285509

/-- Proves that if a shop owner charges 20% more than the cost price,
    and a customer paid 3600 for an item, then the cost price was 3000. -/
theorem furniture_shop_cost_price 
  (markup_percentage : ℝ) 
  (selling_price : ℝ) 
  (cost_price : ℝ) : 
  markup_percentage = 0.20 →
  selling_price = 3600 →
  selling_price = cost_price * (1 + markup_percentage) →
  cost_price = 3000 := by
sorry

end furniture_shop_cost_price_l2855_285509


namespace game_points_l2855_285501

/-- The number of points earned in a video game level --/
def points_earned (total_enemies : ℕ) (enemies_left : ℕ) (points_per_enemy : ℕ) : ℕ :=
  (total_enemies - enemies_left) * points_per_enemy

/-- Theorem: In a level with 8 enemies, destroying all but 6 of them, with 5 points per enemy, results in 10 points --/
theorem game_points : points_earned 8 6 5 = 10 := by
  sorry

end game_points_l2855_285501


namespace seed_fertilizer_ratio_is_three_to_one_l2855_285585

/-- Given a total amount of seed and fertilizer, and the amount of seed,
    calculate the ratio of seed to fertilizer. -/
def seedFertilizerRatio (total : ℚ) (seed : ℚ) : ℚ :=
  seed / (total - seed)

/-- Theorem stating that given 60 gallons total and 45 gallons of seed,
    the ratio of seed to fertilizer is 3:1. -/
theorem seed_fertilizer_ratio_is_three_to_one :
  seedFertilizerRatio 60 45 = 3 := by
  sorry

#eval seedFertilizerRatio 60 45

end seed_fertilizer_ratio_is_three_to_one_l2855_285585
