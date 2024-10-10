import Mathlib

namespace school_poll_intersection_l3084_308462

theorem school_poll_intersection (T C D : Finset ℕ) (h1 : T.card = 230) 
  (h2 : C.card = 171) (h3 : D.card = 137) 
  (h4 : (T \ C).card + (T \ D).card - T.card = 37) : 
  (C ∩ D).card = 115 := by
  sorry

end school_poll_intersection_l3084_308462


namespace equation_solutions_l3084_308468

theorem equation_solutions :
  (∃ x : ℝ, 2 * (x - 2) = 3 * x - 7 ∧ x = 3) ∧
  (∃ x : ℝ, (x - 1) / 2 - (2 * x + 3) / 6 = 1 ∧ x = 12) := by
  sorry

end equation_solutions_l3084_308468


namespace tournament_permutation_exists_l3084_308448

/-- Represents the result of a match between two players -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a tournament with n players -/
structure Tournament (n : Nat) where
  /-- The result of the match between player i and player j -/
  result : Fin n → Fin n → MatchResult

/-- A permutation of players -/
def PlayerPermutation (n : Nat) := Fin n → Fin n

/-- Checks if a player satisfies the condition with their neighbors -/
def satisfiesCondition (t : Tournament 1000) (p : PlayerPermutation 1000) (i : Fin 998) : Prop :=
  (t.result (p i) (p (i + 1)) = MatchResult.Win ∧ t.result (p i) (p (i + 2)) = MatchResult.Win) ∨
  (t.result (p i) (p (i + 1)) = MatchResult.Loss ∧ t.result (p i) (p (i + 2)) = MatchResult.Loss)

/-- The main theorem -/
theorem tournament_permutation_exists (t : Tournament 1000) :
  ∃ (p : PlayerPermutation 1000), ∀ (i : Fin 998), satisfiesCondition t p i := by
  sorry

end tournament_permutation_exists_l3084_308448


namespace min_value_reciprocal_sum_l3084_308476

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 := by
sorry

end min_value_reciprocal_sum_l3084_308476


namespace salt_solution_concentration_l3084_308423

/-- Proves that adding a specific amount of pure salt to a given salt solution results in the desired concentration -/
theorem salt_solution_concentration 
  (initial_weight : Real) 
  (initial_concentration : Real) 
  (added_salt : Real) 
  (final_concentration : Real) : 
  initial_weight = 100 ∧ 
  initial_concentration = 0.1 ∧ 
  added_salt = 28.571428571428573 ∧ 
  final_concentration = 0.3 →
  (initial_concentration * initial_weight + added_salt) / (initial_weight + added_salt) = final_concentration :=
by sorry

end salt_solution_concentration_l3084_308423


namespace gcd_problems_l3084_308422

theorem gcd_problems :
  (Nat.gcd 63 84 = 21) ∧ (Nat.gcd 351 513 = 27) := by
  sorry

end gcd_problems_l3084_308422


namespace lemonade_third_intermission_l3084_308436

theorem lemonade_third_intermission 
  (total : ℝ) 
  (first : ℝ) 
  (second : ℝ) 
  (h1 : total = 0.9166666666666666) 
  (h2 : first = 0.25) 
  (h3 : second = 0.4166666666666667) :
  total - (first + second) = 0.25 := by
sorry

end lemonade_third_intermission_l3084_308436


namespace decimal_0_04_is_4_percent_l3084_308456

/-- Converts a decimal fraction to a percentage -/
def decimal_to_percentage (d : ℝ) : ℝ := d * 100

/-- The decimal fraction we're working with -/
def given_decimal : ℝ := 0.04

/-- Theorem: The percentage representation of 0.04 is 4% -/
theorem decimal_0_04_is_4_percent : decimal_to_percentage given_decimal = 4 := by
  sorry

end decimal_0_04_is_4_percent_l3084_308456


namespace unique_two_digit_number_l3084_308433

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem unique_two_digit_number :
  ∃! n : ℕ, is_two_digit n ∧
    tens_digit n + 2 = ones_digit n ∧
    3 * (tens_digit n * ones_digit n) = n ∧
    n = 24 := by sorry

end unique_two_digit_number_l3084_308433


namespace solution_of_equation_l3084_308466

theorem solution_of_equation : ∃! x : ℝ, (3 / (x - 2) - 1 = 0) ∧ (x = 5) := by
  sorry

end solution_of_equation_l3084_308466


namespace albert_and_allison_marbles_albert_and_allison_marbles_proof_l3084_308435

/-- Proves that Albert and Allison have a total of 136 marbles given the conditions of the problem -/
theorem albert_and_allison_marbles : ℕ → ℕ → ℕ → Prop :=
  fun allison_marbles angela_marbles albert_marbles =>
    allison_marbles = 28 ∧
    angela_marbles = allison_marbles + 8 ∧
    albert_marbles = 3 * angela_marbles →
    albert_marbles + allison_marbles = 136

/-- Proof of the theorem -/
theorem albert_and_allison_marbles_proof :
  ∃ (allison_marbles angela_marbles albert_marbles : ℕ),
    albert_and_allison_marbles allison_marbles angela_marbles albert_marbles :=
by
  sorry

end albert_and_allison_marbles_albert_and_allison_marbles_proof_l3084_308435


namespace angle_relation_l3084_308431

theorem angle_relation (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.tan (α - β) = 1/3) (h4 : Real.tan β = 1/7) :
  2 * α - β = π/4 := by
  sorry

end angle_relation_l3084_308431


namespace complex_z_imaginary_part_l3084_308491

theorem complex_z_imaginary_part (z : ℂ) (h : (3 + 4 * Complex.I) * z = Complex.abs (3 - 4 * Complex.I)) : 
  z.im = -4/5 := by
  sorry

end complex_z_imaginary_part_l3084_308491


namespace sam_need_change_probability_l3084_308499

/-- The number of toys in the vending machine -/
def num_toys : ℕ := 9

/-- The price of Sam's favorite toy in half-dollars -/
def favorite_toy_price : ℕ := 5

/-- The number of half-dollar coins Sam has -/
def sam_coins : ℕ := 10

/-- The probability of Sam needing to break the twenty-dollar bill -/
def probability_need_change : ℚ := 55 / 63

/-- Theorem stating the probability of Sam needing to break the twenty-dollar bill -/
theorem sam_need_change_probability :
  let total_arrangements := (num_toys.factorial : ℚ)
  let favorable_outcomes := ((num_toys - 1).factorial : ℚ) + ((num_toys - 2).factorial : ℚ) + ((num_toys - 3).factorial : ℚ)
  (1 - favorable_outcomes / total_arrangements) = probability_need_change := by
  sorry


end sam_need_change_probability_l3084_308499


namespace arithmetic_geometric_ratio_l3084_308453

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) - a n = d

-- Define a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b / a = r ∧ c / b = r

-- Theorem statement
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : geometric_sequence (a 2) (a 3) (a 6)) : 
  ∃ r : ℝ, r = 5 / 3 ∧ (a 3) / (a 2) = r ∧ (a 6) / (a 3) = r :=
sorry

end arithmetic_geometric_ratio_l3084_308453


namespace graph_intersects_x_equals_one_at_most_once_f_equals_g_l3084_308426

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Statement 1: The graph of y = f(x) intersects with x = 1 at most at one point
theorem graph_intersects_x_equals_one_at_most_once (f : RealFunction) :
  ∃! y, f 1 = y :=
sorry

-- Statement 2: f(x) = x^2 - 2x + 1 and g(t) = t^2 - 2t + 1 are the same function
def f (x : ℝ) : ℝ := x^2 - 2*x + 1
def g (t : ℝ) : ℝ := t^2 - 2*t + 1

theorem f_equals_g : f = g :=
sorry

end graph_intersects_x_equals_one_at_most_once_f_equals_g_l3084_308426


namespace triangle_side_length_validity_l3084_308464

theorem triangle_side_length_validity 
  (a b c : ℝ) 
  (ha : a = 5) 
  (hb : b = 8) 
  (hc : c = 6) : 
  a + b > c ∧ a + c > b ∧ b + c > a :=
sorry

end triangle_side_length_validity_l3084_308464


namespace circle_tangent_theorem_l3084_308455

/-- Given two externally tangent circles and a tangent line satisfying certain conditions,
    prove the relationship between r, R, and p, and the length of BC. -/
theorem circle_tangent_theorem (r R p : ℝ) (h_pos_r : 0 < r) (h_pos_R : 0 < R) (h_pos_p : 0 < p) :
  -- Condition for the geometric configuration
  (p^2 / (4 * (p + 1)) < r / R ∧ r / R < p^2 / (2 * (p + 1))) →
  -- Length of BC
  ∃ (BC : ℝ), BC = p / (p + 1) * Real.sqrt (4 * (p + 1) * R * r - p^2 * R^2) := by
  sorry


end circle_tangent_theorem_l3084_308455


namespace solve_system_l3084_308424

theorem solve_system (x y b : ℚ) : 
  (4 * x + 2 * y = b) → 
  (3 * x + 7 * y = 3 * b) → 
  (y = 3) → 
  (b = 22 / 3) := by
sorry

end solve_system_l3084_308424


namespace function_fits_data_l3084_308451

/-- The set of data points representing the relationship between x and y -/
def data_points : List (ℚ × ℚ) := [(0, 200), (2, 160), (4, 80), (6, 0), (8, -120)]

/-- The proposed quadratic function -/
def f (x : ℚ) : ℚ := -10 * x^2 + 200

/-- Theorem stating that the proposed function fits all data points -/
theorem function_fits_data : ∀ (point : ℚ × ℚ), point ∈ data_points → f point.1 = point.2 := by
  sorry

end function_fits_data_l3084_308451


namespace expected_sticky_corn_l3084_308409

theorem expected_sticky_corn (total_corn : ℕ) (sticky_corn : ℕ) (sample_size : ℕ) :
  total_corn = 140 →
  sticky_corn = 56 →
  sample_size = 40 →
  (sample_size * sticky_corn) / total_corn = 16 := by
  sorry

end expected_sticky_corn_l3084_308409


namespace pinwheel_area_is_six_l3084_308408

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a kite in the pinwheel -/
structure Kite where
  center : GridPoint
  vertex1 : GridPoint
  vertex2 : GridPoint
  vertex3 : GridPoint

/-- Represents a pinwheel shape -/
structure Pinwheel where
  kites : Fin 4 → Kite
  grid_size : Nat
  h_grid_size : grid_size = 5

/-- Calculates the area of a pinwheel -/
noncomputable def pinwheel_area (p : Pinwheel) : ℝ :=
  sorry

/-- Theorem stating that the area of the described pinwheel is 6 square units -/
theorem pinwheel_area_is_six (p : Pinwheel) : pinwheel_area p = 6 := by
  sorry

end pinwheel_area_is_six_l3084_308408


namespace function_satisfying_equation_l3084_308480

theorem function_satisfying_equation (f : ℝ → ℝ) :
  (∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) → (∀ x : ℝ, f x = x + 1) :=
by sorry

end function_satisfying_equation_l3084_308480


namespace prob_three_non_defective_l3084_308457

/-- The probability of selecting 3 non-defective pencils from a box of 7 pencils, where 2 are defective. -/
theorem prob_three_non_defective (total : Nat) (defective : Nat) (selected : Nat) :
  total = 7 →
  defective = 2 →
  selected = 3 →
  (Nat.choose (total - defective) selected : ℚ) / (Nat.choose total selected : ℚ) = 2 / 7 := by
  sorry

end prob_three_non_defective_l3084_308457


namespace division_with_remainder_l3084_308477

theorem division_with_remainder : ∃ (q r : ℤ), 1234567 = 131 * q + r ∧ 0 ≤ r ∧ r < 131 ∧ r = 36 := by
  sorry

end division_with_remainder_l3084_308477


namespace special_sequence_tenth_term_l3084_308484

/-- A sequence satisfying the given condition -/
def SpecialSequence (a : ℕ+ → ℤ) : Prop :=
  ∀ m n : ℕ+, a m + a n = a (m + n) - 2 * (m.val * n.val)

/-- The theorem to be proved -/
theorem special_sequence_tenth_term (a : ℕ+ → ℤ) 
  (h : SpecialSequence a) (h1 : a 1 = 1) : a 10 = 100 := by
  sorry

end special_sequence_tenth_term_l3084_308484


namespace det_special_matrix_l3084_308467

/-- The determinant of the matrix [[y+2, y-1, y+1], [y+1, y+2, y-1], [y-1, y+1, y+2]] 
    is equal to 6y^2 + 23y + 14 for any real number y. -/
theorem det_special_matrix (y : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![y + 2, y - 1, y + 1],
    ![y + 1, y + 2, y - 1],
    ![y - 1, y + 1, y + 2]
  ]
  Matrix.det M = 6 * y^2 + 23 * y + 14 := by
  sorry

end det_special_matrix_l3084_308467


namespace gumball_distribution_l3084_308446

/-- Represents the number of gumballs each person has -/
structure Gumballs :=
  (joanna : ℕ)
  (jacques : ℕ)
  (julia : ℕ)

/-- Calculates the total number of gumballs -/
def total_gumballs (g : Gumballs) : ℕ :=
  g.joanna + g.jacques + g.julia

/-- Represents the purchase multipliers for each person -/
structure PurchaseMultipliers :=
  (joanna : ℕ)
  (jacques : ℕ)
  (julia : ℕ)

/-- Calculates the number of gumballs after purchases -/
def after_purchase (initial : Gumballs) (multipliers : PurchaseMultipliers) : Gumballs :=
  { joanna := initial.joanna + initial.joanna * multipliers.joanna,
    jacques := initial.jacques + initial.jacques * multipliers.jacques,
    julia := initial.julia + initial.julia * multipliers.julia }

/-- Theorem statement -/
theorem gumball_distribution 
  (initial : Gumballs) 
  (multipliers : PurchaseMultipliers) :
  initial.joanna = 40 ∧ 
  initial.jacques = 60 ∧ 
  initial.julia = 80 ∧
  multipliers.joanna = 5 ∧
  multipliers.jacques = 3 ∧
  multipliers.julia = 2 →
  let final := after_purchase initial multipliers
  (final.joanna = 240 ∧ 
   final.jacques = 240 ∧ 
   final.julia = 240) ∧
  (total_gumballs final / 3 = 240) :=
by sorry

end gumball_distribution_l3084_308446


namespace relationship_between_x_and_y_l3084_308479

theorem relationship_between_x_and_y 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (x : ℝ) 
  (hx : x = Real.sqrt (a + b) - Real.sqrt b) 
  (y : ℝ) 
  (hy : y = Real.sqrt b - Real.sqrt (b - a)) : 
  x < y := by
  sorry

end relationship_between_x_and_y_l3084_308479


namespace rectangle_area_preservation_l3084_308414

theorem rectangle_area_preservation (L W : ℝ) (h : L > 0 ∧ W > 0) :
  ∃ x : ℝ, x > 0 ∧ x < 100 ∧
  (L * (1 - x / 100)) * (W * 1.25) = L * W ∧
  x = 20 := by
sorry

end rectangle_area_preservation_l3084_308414


namespace same_last_six_digits_l3084_308415

/-- Given a positive integer N where N and N^2 both end in the same sequence
    of six digits abcdef in base 10 (with a ≠ 0), prove that the five-digit
    number abcde is equal to 48437. -/
theorem same_last_six_digits (N : ℕ) : 
  (N > 0) →
  (∃ (a b c d e f : ℕ), 
    a ≠ 0 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
    N % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f ∧
    (N^2) % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) →
  (∃ (a b c d e : ℕ),
    a ≠ 0 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    a * 10000 + b * 1000 + c * 100 + d * 10 + e = 48437) :=
by sorry

end same_last_six_digits_l3084_308415


namespace problem_1_problem_2_l3084_308434

-- Problem 1
theorem problem_1 : |(-12)| - (-6) + 5 - 10 = 13 := by sorry

-- Problem 2
theorem problem_2 : 64.83 - 5 * (18/19) + 35.17 - 44 * (1/19) = 50 := by sorry

end problem_1_problem_2_l3084_308434


namespace intersection_tangent_line_l3084_308438

theorem intersection_tangent_line (x₀ : ℝ) (hx₀ : x₀ ≠ 0) (h : Real.tan x₀ = -x₀) :
  (x₀^2 + 1) * (1 + Real.cos (2 * x₀)) = 2 := by
  sorry

end intersection_tangent_line_l3084_308438


namespace orange_juice_percentage_approx_48_l3084_308492

/-- Represents the juice yield from a specific fruit -/
structure JuiceYield where
  fruit : String
  count : Nat
  ounces : Rat

/-- Calculates the juice blend composition and returns the percentage of orange juice -/
def orangeJuicePercentage (appleYield pearYield orangeYield : JuiceYield) : Rat :=
  let appleJuicePerFruit := appleYield.ounces / appleYield.count
  let pearJuicePerFruit := pearYield.ounces / pearYield.count
  let orangeJuicePerFruit := orangeYield.ounces / orangeYield.count
  let totalJuice := appleJuicePerFruit + pearJuicePerFruit + orangeJuicePerFruit
  (orangeJuicePerFruit / totalJuice) * 100

/-- Theorem stating that the percentage of orange juice in the blend is approximately 48% -/
theorem orange_juice_percentage_approx_48 (appleYield pearYield orangeYield : JuiceYield) 
  (h1 : appleYield.fruit = "apple" ∧ appleYield.count = 5 ∧ appleYield.ounces = 9)
  (h2 : pearYield.fruit = "pear" ∧ pearYield.count = 4 ∧ pearYield.ounces = 10)
  (h3 : orangeYield.fruit = "orange" ∧ orangeYield.count = 3 ∧ orangeYield.ounces = 12) :
  ∃ (ε : Rat), abs (orangeJuicePercentage appleYield pearYield orangeYield - 48) < ε ∧ ε < 1 := by
  sorry

end orange_juice_percentage_approx_48_l3084_308492


namespace fourth_power_of_nested_sqrt_l3084_308437

theorem fourth_power_of_nested_sqrt (y : ℝ) :
  y = Real.sqrt (3 + Real.sqrt (3 + Real.sqrt 5)) →
  y^4 = 12 + 6 * Real.sqrt (3 + Real.sqrt 5) + Real.sqrt 5 := by
  sorry

end fourth_power_of_nested_sqrt_l3084_308437


namespace price_reduction_equation_l3084_308471

theorem price_reduction_equation (x : ℝ) : 
  (100 : ℝ) * (1 - x)^2 = 80 ↔ 
  (∃ (price1 price2 : ℝ), 
    price1 = 100 * (1 - x) ∧ 
    price2 = price1 * (1 - x) ∧ 
    price2 = 80) :=
by sorry

end price_reduction_equation_l3084_308471


namespace compound_interest_problem_l3084_308485

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Total amount calculation -/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

/-- Theorem statement -/
theorem compound_interest_problem (P : ℝ) :
  compound_interest P 0.04 2 = 326.40 →
  total_amount P 326.40 = 4326.40 := by
sorry

#eval compound_interest 4000 0.04 2
#eval total_amount 4000 326.40

end compound_interest_problem_l3084_308485


namespace chord_bisector_line_l3084_308460

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a parabola y² = 4x -/
def onParabola (p : Point) : Prop := p.y^2 = 4 * p.x

/-- Checks if a point lies on a line -/
def onLine (p : Point) (l : Line) : Prop := l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

/-- The main theorem -/
theorem chord_bisector_line (A B : Point) (P : Point) :
  onParabola A ∧ onParabola B ∧ 
  isMidpoint P A B ∧ 
  P.x = 1 ∧ P.y = 1 →
  ∃ l : Line, l.a = 2 ∧ l.b = -1 ∧ l.c = -1 ∧ onLine A l ∧ onLine B l :=
by sorry

end chord_bisector_line_l3084_308460


namespace biology_marks_proof_l3084_308412

def english_marks : ℕ := 72
def math_marks : ℕ := 45
def physics_marks : ℕ := 72
def chemistry_marks : ℕ := 77
def average_marks : ℚ := 68.2
def total_subjects : ℕ := 5

theorem biology_marks_proof :
  ∃ (biology_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / total_subjects = average_marks ∧
    biology_marks = 75 := by
  sorry

end biology_marks_proof_l3084_308412


namespace negative_double_greater_than_negative_abs_l3084_308440

theorem negative_double_greater_than_negative_abs :
  -(-(1/9 : ℚ)) > -|(-(1/9 : ℚ))| := by sorry

end negative_double_greater_than_negative_abs_l3084_308440


namespace two_different_color_chips_probability_l3084_308495

/-- The probability of drawing two chips of different colors from a bag containing
    6 green chips, 5 purple chips, and 4 orange chips, when drawing with replacement. -/
theorem two_different_color_chips_probability :
  let total_chips : ℕ := 6 + 5 + 4
  let green_chips : ℕ := 6
  let purple_chips : ℕ := 5
  let orange_chips : ℕ := 4
  let prob_green : ℚ := green_chips / total_chips
  let prob_purple : ℚ := purple_chips / total_chips
  let prob_orange : ℚ := orange_chips / total_chips
  let prob_not_green : ℚ := (purple_chips + orange_chips) / total_chips
  let prob_not_purple : ℚ := (green_chips + orange_chips) / total_chips
  let prob_not_orange : ℚ := (green_chips + purple_chips) / total_chips
  (prob_green * prob_not_green + prob_purple * prob_not_purple + prob_orange * prob_not_orange) = 148 / 225 := by
  sorry

end two_different_color_chips_probability_l3084_308495


namespace complex_equation_implies_product_l3084_308459

/-- Given that (1+mi)/i = 1+ni where m, n ∈ ℝ and i is the imaginary unit, prove that mn = -1 -/
theorem complex_equation_implies_product (m n : ℝ) : 
  (1 + m * Complex.I) / Complex.I = 1 + n * Complex.I → m * n = -1 := by
  sorry

end complex_equation_implies_product_l3084_308459


namespace triangle_tangent_ratio_l3084_308417

theorem triangle_tangent_ratio (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- Acute triangle condition
  A + B + C = π →  -- Triangle angle sum
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a / (2 * Real.sin A) = b / (2 * Real.sin B) →  -- Law of sines
  a / (2 * Real.sin A) = c / (2 * Real.sin C) →  -- Law of sines
  a / b + b / a = 6 * Real.cos C →  -- Given condition
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 4 := by
sorry

end triangle_tangent_ratio_l3084_308417


namespace expression_value_polynomial_simplification_l3084_308428

-- Part 1
theorem expression_value : (1/2)^(-2) - 0.01^(-1) + (-1 - 1/7)^0 = -95 := by sorry

-- Part 2
theorem polynomial_simplification (x : ℝ) : (x-2)*(x+1) - (x-1)^2 = x - 3 := by sorry

end expression_value_polynomial_simplification_l3084_308428


namespace geometric_sequence_floor_frac_l3084_308481

theorem geometric_sequence_floor_frac (x : ℝ) : 
  x ≠ 0 →
  let floor_x := ⌊x⌋
  let frac_x := x - floor_x
  (frac_x * floor_x = floor_x * x) →
  x = (5 + Real.sqrt 5) / 4 := by
sorry

end geometric_sequence_floor_frac_l3084_308481


namespace trig_fraction_equals_four_fifths_l3084_308450

theorem trig_fraction_equals_four_fifths (θ : ℝ) (h : Real.tan θ = 2) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4/5 := by
  sorry

end trig_fraction_equals_four_fifths_l3084_308450


namespace max_queens_101_88_l3084_308416

/-- Represents a chessboard with a red corner -/
structure RedCornerBoard :=
  (size : Nat)
  (red_size : Nat)
  (h_size : size > red_size)

/-- Represents the maximum number of non-attacking queens on a RedCornerBoard -/
def max_queens (board : RedCornerBoard) : Nat :=
  2 * (board.size - board.red_size)

/-- Theorem stating the maximum number of non-attacking queens on a 101x101 board with 88x88 red corner -/
theorem max_queens_101_88 :
  let board : RedCornerBoard := ⟨101, 88, by norm_num⟩
  max_queens board = 26 := by
  sorry

#eval max_queens ⟨101, 88, by norm_num⟩

end max_queens_101_88_l3084_308416


namespace sixteen_pow_six_mod_nine_l3084_308482

theorem sixteen_pow_six_mod_nine : 16^6 ≡ 1 [ZMOD 9] := by
  sorry

end sixteen_pow_six_mod_nine_l3084_308482


namespace circle_passes_through_points_l3084_308496

/-- A circle passing through three given points -/
def circle_through_points (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 + p.2 ^ 2 + 4 * p.1 - 2 * p.2) = 0}

/-- The three given points -/
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (0, 2)

/-- Theorem stating that the defined circle passes through the given points -/
theorem circle_passes_through_points :
  A ∈ circle_through_points A B C ∧
  B ∈ circle_through_points A B C ∧
  C ∈ circle_through_points A B C :=
sorry

end circle_passes_through_points_l3084_308496


namespace expand_expression_l3084_308442

theorem expand_expression (x y : ℝ) : (2*x - 3*y + 1) * (2*x + 3*y - 1) = 4*x^2 - 9*y^2 + 6*y - 1 := by
  sorry

end expand_expression_l3084_308442


namespace equation_solution_l3084_308443

theorem equation_solution (a : ℝ) : (a + 3) ^ (a + 1) = 1 ↔ a = -1 ∨ a = -2 := by
  sorry

end equation_solution_l3084_308443


namespace max_diff_even_digit_numbers_l3084_308458

/-- A function that checks if a natural number has all even digits -/
def all_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

/-- A function that checks if a natural number has at least one odd digit -/
def has_odd_digit (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d % 2 = 1

/-- The theorem stating the maximum difference between two 6-digit numbers with all even digits -/
theorem max_diff_even_digit_numbers :
  ∃ (a b : ℕ),
    100000 ≤ a ∧ a < b ∧ b < 1000000 ∧
    all_even_digits a ∧
    all_even_digits b ∧
    (∀ k, a < k ∧ k < b → has_odd_digit k) ∧
    b - a = 111112 ∧
    (∀ a' b', 100000 ≤ a' ∧ a' < b' ∧ b' < 1000000 ∧
              all_even_digits a' ∧
              all_even_digits b' ∧
              (∀ k, a' < k ∧ k < b' → has_odd_digit k) →
              b' - a' ≤ 111112) :=
by sorry

end max_diff_even_digit_numbers_l3084_308458


namespace boys_without_notebooks_l3084_308441

theorem boys_without_notebooks
  (total_boys : ℕ)
  (total_with_notebooks : ℕ)
  (girls_with_notebooks : ℕ)
  (h1 : total_boys = 24)
  (h2 : total_with_notebooks = 30)
  (h3 : girls_with_notebooks = 18) :
  total_boys - (total_with_notebooks - girls_with_notebooks) = 12 :=
by sorry

end boys_without_notebooks_l3084_308441


namespace conversion_1_conversion_2_conversion_3_conversion_4_l3084_308487

-- Define conversion rates
def sq_meter_to_sq_decimeter : ℝ := 100
def hectare_to_sq_meter : ℝ := 10000
def sq_decimeter_to_sq_centimeter : ℝ := 100
def sq_kilometer_to_hectare : ℝ := 100

-- Theorem statements
theorem conversion_1 : 3 * sq_meter_to_sq_decimeter = 300 := by sorry

theorem conversion_2 : 2 * hectare_to_sq_meter = 20000 := by sorry

theorem conversion_3 : 5000 / sq_decimeter_to_sq_centimeter = 50 := by sorry

theorem conversion_4 : 8 * sq_kilometer_to_hectare = 800 := by sorry

end conversion_1_conversion_2_conversion_3_conversion_4_l3084_308487


namespace square_triangle_apothem_ratio_l3084_308421

theorem square_triangle_apothem_ratio :
  ∀ (s t : ℝ),
  s > 0 → t > 0 →
  s * Real.sqrt 2 = 9 * t →  -- diagonal of square = 3 * perimeter of triangle
  s * s = 2 * s →           -- apothem of square = area of square
  (s / 2) / ((Real.sqrt 3 / 2 * t) / 3) = 9 * Real.sqrt 6 / 4 :=
by sorry

end square_triangle_apothem_ratio_l3084_308421


namespace ribbons_shipment_count_l3084_308439

/-- The number of ribbons that arrived in the shipment before lunch -/
def ribbons_in_shipment (initial : ℕ) (morning : ℕ) (afternoon : ℕ) (final : ℕ) : ℕ :=
  (afternoon + final) - (initial - morning)

/-- Theorem stating that the number of ribbons in the shipment is 4 -/
theorem ribbons_shipment_count :
  ribbons_in_shipment 38 14 16 12 = 4 := by
  sorry

end ribbons_shipment_count_l3084_308439


namespace min_value_theorem_solution_set_theorem_l3084_308432

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 1|

-- Theorem for part (1)
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = f (-1)) :
  (2/a + 1/b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = f (-1) ∧ 2/a₀ + 1/b₀ = 8 := by
  sorry

-- Theorem for part (2)
theorem solution_set_theorem (x : ℝ) :
  f x > 1/2 ↔ x < 5/4 := by
  sorry

end min_value_theorem_solution_set_theorem_l3084_308432


namespace sum_mod_seven_l3084_308444

theorem sum_mod_seven : (1001 + 1002 + 1003 + 1004 + 1005) % 7 = 3 := by
  sorry

end sum_mod_seven_l3084_308444


namespace paint_mixture_theorem_l3084_308488

theorem paint_mixture_theorem (total : ℝ) (blue_added : ℝ) (white_added : ℝ) :
  white_added = 20 →
  blue_added / total = 0.7 →
  white_added / total = 0.1 →
  blue_added = 140 := by
sorry

end paint_mixture_theorem_l3084_308488


namespace permutation_and_exponent_inequalities_l3084_308419

theorem permutation_and_exponent_inequalities 
  (i m n : ℕ) 
  (h1 : 1 < i) 
  (h2 : i ≤ m) 
  (h3 : m < n) : 
  n * (m.factorial / (m - i).factorial) < m * (n.factorial / (n - i).factorial) ∧ 
  (1 + m : ℝ) ^ n > (1 + n : ℝ) ^ m := by
  sorry

end permutation_and_exponent_inequalities_l3084_308419


namespace fraction_equivalence_l3084_308429

theorem fraction_equivalence : ∃ n : ℤ, (2 + n) / (7 + n) = 3 / 4 :=
by
  -- The proof goes here
  sorry

end fraction_equivalence_l3084_308429


namespace deposit_ratio_l3084_308418

def mark_deposit : ℚ := 88
def total_deposit : ℚ := 400

def bryan_deposit : ℚ := total_deposit - mark_deposit

theorem deposit_ratio :
  ∃ (n : ℚ), n > 1 ∧ bryan_deposit < n * mark_deposit →
  (bryan_deposit / mark_deposit) = 39 / 11 := by
sorry

end deposit_ratio_l3084_308418


namespace faye_finished_problems_l3084_308483

theorem faye_finished_problems (math_problems science_problems left_for_homework : ℕ)
  (h1 : math_problems = 46)
  (h2 : science_problems = 9)
  (h3 : left_for_homework = 15) :
  math_problems + science_problems - left_for_homework = 40 := by
  sorry

end faye_finished_problems_l3084_308483


namespace triangle_count_l3084_308461

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of collinear triplets in the given configuration -/
def collinearTriplets : ℕ := 16

/-- The total number of points in the configuration -/
def totalPoints : ℕ := 12

/-- The number of points needed to form a triangle -/
def pointsPerTriangle : ℕ := 3

theorem triangle_count :
  choose totalPoints pointsPerTriangle - collinearTriplets = 204 := by
  sorry

end triangle_count_l3084_308461


namespace probability_of_specific_arrangement_l3084_308427

theorem probability_of_specific_arrangement (n : ℕ) (r : ℕ) : 
  n = 4 → r = 2 → (1 : ℚ) / (n! / r!) = 1 / 12 := by
  sorry

end probability_of_specific_arrangement_l3084_308427


namespace largest_percentage_increase_l3084_308478

def students : Fin 6 → ℕ
  | 0 => 50  -- 2010
  | 1 => 55  -- 2011
  | 2 => 60  -- 2012
  | 3 => 72  -- 2013
  | 4 => 75  -- 2014
  | 5 => 90  -- 2015

def percentageIncrease (year : Fin 5) : ℚ :=
  (students (year.succ) - students year : ℚ) / students year * 100

theorem largest_percentage_increase :
  (∀ year : Fin 5, percentageIncrease year ≤ percentageIncrease 2 ∨ percentageIncrease year ≤ percentageIncrease 4) ∧
  percentageIncrease 2 = percentageIncrease 4 :=
sorry

end largest_percentage_increase_l3084_308478


namespace max_area_equilateral_triangle_in_rectangle_l3084_308402

/-- The maximum area of an equilateral triangle inscribed in a 12 by 5 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle :
  ∃ (A : ℝ),
    A = (15 : ℝ) * Real.sqrt 3 - 10 ∧
    (∀ (s : ℝ),
      s > 0 →
      s ≤ 5 →
      s * Real.sqrt 3 ≤ 12 →
      (Real.sqrt 3 / 4) * s^2 ≤ A) :=
by sorry

end max_area_equilateral_triangle_in_rectangle_l3084_308402


namespace closer_to_d_probability_l3084_308447

/-- Triangle DEF with side lengths -/
structure Triangle (DE EF FD : ℝ) where
  side_positive : 0 < DE ∧ 0 < EF ∧ 0 < FD
  triangle_inequality : DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

/-- The region closer to D than to E or F -/
def CloserToD (t : Triangle DE EF FD) : Set (ℝ × ℝ) := sorry

theorem closer_to_d_probability (t : Triangle 8 6 10) : 
  MeasureTheory.volume (CloserToD t) = (1/4) * MeasureTheory.volume (Set.univ : Set (ℝ × ℝ)) := by
  sorry

end closer_to_d_probability_l3084_308447


namespace determinant_value_l3084_308469

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_value (m : ℝ) (h : m^2 - 2*m - 3 = 0) : 
  determinant (m^2) (m-3) (1-2*m) (m-2) = 9 := by sorry

end determinant_value_l3084_308469


namespace trig_identity_proof_l3084_308449

theorem trig_identity_proof : 
  Real.sin (15 * π / 180) * Real.cos (45 * π / 180) + 
  Real.sin (75 * π / 180) * Real.sin (135 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end trig_identity_proof_l3084_308449


namespace percentage_of_male_students_l3084_308406

theorem percentage_of_male_students (male_percentage : ℝ) 
  (h1 : 0 ≤ male_percentage ∧ male_percentage ≤ 100)
  (h2 : 50 = 100 * (1 - (male_percentage / 100 * 0.5 + (100 - male_percentage) / 100 * 0.6)))
  : male_percentage = 40 := by
  sorry

end percentage_of_male_students_l3084_308406


namespace bridge_length_l3084_308465

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 250 →
  train_speed_kmh = 72 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 350 := by
  sorry

end bridge_length_l3084_308465


namespace line_hyperbola_intersection_l3084_308473

theorem line_hyperbola_intersection (k : ℝ) : 
  (∀ x y : ℝ, y = k * x - 1 ∧ x^2 - y^2 = 4 → ∃ x y : ℝ, y = k * x - 1 ∧ x^2 - y^2 = 4) ↔ 
  (k = 1 ∨ k = -1 ∨ (-Real.sqrt 5 / 2 ≤ k ∧ k ≤ Real.sqrt 5 / 2)) :=
sorry

end line_hyperbola_intersection_l3084_308473


namespace cube_squared_equals_sixth_power_l3084_308407

theorem cube_squared_equals_sixth_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end cube_squared_equals_sixth_power_l3084_308407


namespace fill_675_cans_in_36_minutes_l3084_308405

/-- A machine that fills cans of paint -/
structure PaintMachine where
  cans_per_batch : ℕ
  minutes_per_batch : ℕ

/-- Calculate the time needed to fill a given number of cans -/
def time_to_fill (machine : PaintMachine) (total_cans : ℕ) : ℕ :=
  (total_cans * machine.minutes_per_batch + machine.cans_per_batch - 1) / machine.cans_per_batch

/-- Theorem stating that it takes 36 minutes to fill 675 cans -/
theorem fill_675_cans_in_36_minutes :
  let machine : PaintMachine := { cans_per_batch := 150, minutes_per_batch := 8 }
  time_to_fill machine 675 = 36 := by
  sorry

end fill_675_cans_in_36_minutes_l3084_308405


namespace equal_grid_values_l3084_308474

/-- Represents a point in the infinite square grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents an admissible polygon on the grid --/
structure AdmissiblePolygon where
  vertices : List GridPoint
  area : ℕ
  area_gt_two : area > 2

/-- The grid of natural numbers --/
def Grid := GridPoint → ℕ

/-- The value of an admissible polygon --/
def value (grid : Grid) (polygon : AdmissiblePolygon) : ℕ := sorry

/-- Two polygons are congruent --/
def congruent (p1 p2 : AdmissiblePolygon) : Prop := sorry

/-- Main theorem --/
theorem equal_grid_values (grid : Grid) :
  (∀ p1 p2 : AdmissiblePolygon, congruent p1 p2 → value grid p1 = value grid p2) →
  (∀ p1 p2 : GridPoint, grid p1 = grid p2) := by sorry

end equal_grid_values_l3084_308474


namespace helen_thanksgiving_desserts_l3084_308498

/-- The number of chocolate chip cookies Helen baked -/
def chocolate_chip_cookies : ℕ := 435

/-- The number of sugar cookies Helen baked -/
def sugar_cookies : ℕ := 139

/-- The number of brownies Helen made -/
def brownies : ℕ := 215

/-- The total number of desserts Helen prepared for Thanksgiving -/
def total_desserts : ℕ := chocolate_chip_cookies + sugar_cookies + brownies

theorem helen_thanksgiving_desserts : total_desserts = 789 := by
  sorry

end helen_thanksgiving_desserts_l3084_308498


namespace quadratic_function_property_l3084_308452

theorem quadratic_function_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : f 0 = f 4)
  (h3 : f 0 > f 1) :
  a > 0 ∧ 4 * a + b = 0 := by
sorry

end quadratic_function_property_l3084_308452


namespace work_completion_time_l3084_308410

theorem work_completion_time (original_men : ℕ) (added_men : ℕ) (time_reduction : ℕ) : 
  original_men = 40 →
  added_men = 8 →
  time_reduction = 10 →
  ∃ (original_time : ℕ), 
    original_time * original_men = (original_time - time_reduction) * (original_men + added_men) ∧
    original_time = 60 :=
by sorry

end work_completion_time_l3084_308410


namespace train_crossing_time_l3084_308400

/-- Given a train and platform with specific dimensions and time to pass,
    prove the time it takes for the train to cross a point (tree) -/
theorem train_crossing_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (time_to_pass_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 700)
  (h3 : time_to_pass_platform = 190)
  : (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 120 :=
by sorry

end train_crossing_time_l3084_308400


namespace x_fourth_plus_y_fourth_l3084_308404

theorem x_fourth_plus_y_fourth (x y : ℕ+) (h : y * x^2 + x * y^2 = 70) : x^4 + y^4 = 641 := by
  sorry

end x_fourth_plus_y_fourth_l3084_308404


namespace circle_center_radius_sum_l3084_308425

/-- Given a circle D with equation x^2 + 10x + 2y^2 - 8y = 18,
    prove that the sum of its center coordinates and radius is -3 + √38 -/
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), 
    (∀ (x y : ℝ), x^2 + 10*x + 2*y^2 - 8*y = 18 ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
    a + b + r = -3 + Real.sqrt 38 := by
  sorry

end circle_center_radius_sum_l3084_308425


namespace arun_age_is_60_l3084_308486

/-- Proves that Arun's age is 60 years given the conditions from the problem -/
theorem arun_age_is_60 (arun_age madan_age gokul_age : ℕ) 
  (h1 : (arun_age - 6) / 18 = gokul_age)
  (h2 : gokul_age = madan_age - 2)
  (h3 : madan_age = 5) : 
  arun_age = 60 := by
  sorry

end arun_age_is_60_l3084_308486


namespace total_pepper_weight_l3084_308489

theorem total_pepper_weight :
  let green_peppers : ℝ := 3.25
  let red_peppers : ℝ := 2.5
  let yellow_peppers : ℝ := 1.75
  let orange_peppers : ℝ := 4.6
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 12.1 := by
  sorry

end total_pepper_weight_l3084_308489


namespace quarters_count_l3084_308472

/-- Represents the types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a collection of coins --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat
  total_coins : pennies + nickels + dimes + quarters = 11
  at_least_one : pennies ≥ 1 ∧ nickels ≥ 1 ∧ dimes ≥ 1 ∧ quarters ≥ 1
  total_value : pennies * coinValue Coin.Penny +
                nickels * coinValue Coin.Nickel +
                dimes * coinValue Coin.Dime +
                quarters * coinValue Coin.Quarter = 132

theorem quarters_count (cc : CoinCollection) : cc.quarters = 3 := by
  sorry

end quarters_count_l3084_308472


namespace solution_set_when_a_is_4_range_of_a_when_f_always_ge_4_l3084_308470

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solution_set_when_a_is_4 :
  {x : ℝ | f 4 x ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} := by sorry

-- Part 2
theorem range_of_a_when_f_always_ge_4 :
  (∀ x : ℝ, f a x ≥ 4) → (a ≤ -3 ∨ a ≥ 5) := by sorry

end solution_set_when_a_is_4_range_of_a_when_f_always_ge_4_l3084_308470


namespace new_students_average_age_l3084_308411

/-- Proves that the average age of new students is 32 years given the conditions of the problem -/
theorem new_students_average_age
  (original_average : ℝ)
  (new_students : ℕ)
  (new_average : ℝ)
  (original_strength : ℕ)
  (h1 : original_average = 40)
  (h2 : new_students = 12)
  (h3 : new_average = 36)
  (h4 : original_strength = 12) :
  (original_strength : ℝ) * original_average + (new_students : ℝ) * 32 =
    ((original_strength + new_students) : ℝ) * new_average :=
by sorry

end new_students_average_age_l3084_308411


namespace num_paths_is_126_l3084_308403

/-- The number of paths from A to C passing through B on a grid -/
def num_paths_through_B : ℕ :=
  let a_to_b_right := 5
  let a_to_b_down := 2
  let b_to_c_right := 2
  let b_to_c_down := 2
  let paths_a_to_b := Nat.choose (a_to_b_right + a_to_b_down) a_to_b_right
  let paths_b_to_c := Nat.choose (b_to_c_right + b_to_c_down) b_to_c_right
  paths_a_to_b * paths_b_to_c

/-- Theorem stating the number of paths from A to C passing through B is 126 -/
theorem num_paths_is_126 : num_paths_through_B = 126 := by
  sorry

end num_paths_is_126_l3084_308403


namespace houses_with_both_features_l3084_308497

theorem houses_with_both_features (total : ℕ) (garage : ℕ) (pool : ℕ) (neither : ℕ)
  (h_total : total = 85)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_neither : neither = 30) :
  ∃ (both : ℕ), both = garage + pool - (total - neither) :=
by
  sorry

end houses_with_both_features_l3084_308497


namespace stair_climbing_time_l3084_308445

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The time taken to climb stairs -/
theorem stair_climbing_time : arithmetic_sum 15 8 7 = 273 := by
  sorry

end stair_climbing_time_l3084_308445


namespace orthogonal_projection_area_l3084_308413

/-- A plane polygon -/
structure PlanePolygon where
  area : ℝ

/-- An orthogonal projection of a plane polygon onto another plane -/
structure OrthogonalProjection (P : PlanePolygon) where
  area : ℝ
  angle : ℝ  -- Angle between the original plane and the projection plane

/-- 
Theorem: The area of the orthogonal projection of a plane polygon 
onto a plane is equal to the area of the polygon being projected, 
multiplied by the cosine of the angle between the projection plane 
and the plane of the polygon.
-/
theorem orthogonal_projection_area 
  (P : PlanePolygon) (proj : OrthogonalProjection P) : 
  proj.area = P.area * Real.cos proj.angle := by
  sorry

end orthogonal_projection_area_l3084_308413


namespace star_equation_solution_l3084_308493

/-- Custom star operation -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

/-- Theorem: If 4 ⋆ x = 46, then x = 50/7 -/
theorem star_equation_solution (x : ℝ) (h : star 4 x = 46) : x = 50/7 := by
  sorry

end star_equation_solution_l3084_308493


namespace twenty_first_figure_squares_l3084_308430

/-- The number of squares in the nth figure of the sequence -/
def num_squares (n : ℕ) : ℕ := n^2 + (n-1)^2

/-- The theorem stating that the 21st figure has 841 squares -/
theorem twenty_first_figure_squares : num_squares 21 = 841 := by
  sorry

end twenty_first_figure_squares_l3084_308430


namespace angle_value_in_connected_triangles_l3084_308454

theorem angle_value_in_connected_triangles : ∀ x : ℝ,
  (∃ α β : ℝ,
    -- Left triangle
    3 * x + 4 * x + α = 180 ∧
    -- Middle triangle
    α + 5 * x + β = 180 ∧
    -- Right triangle
    β + 2 * x + 6 * x = 180) →
  x = 18 := by
  sorry

end angle_value_in_connected_triangles_l3084_308454


namespace sqrt_seven_identities_l3084_308475

theorem sqrt_seven_identities (a b : ℝ) (ha : a = Real.sqrt 7 + 2) (hb : b = Real.sqrt 7 - 2) :
  (a * b = 3) ∧ (a^2 + b^2 - a * b = 19) := by
  sorry

end sqrt_seven_identities_l3084_308475


namespace exponent_division_l3084_308463

theorem exponent_division (a : ℝ) : a^3 / a^2 = a := by
  sorry

end exponent_division_l3084_308463


namespace even_quadratic_implies_k_eq_one_l3084_308401

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The quadratic function f(x) = kx^2 + (k-1)x + 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

/-- If f(x) = kx^2 + (k-1)x + 2 is an even function, then k = 1 -/
theorem even_quadratic_implies_k_eq_one (k : ℝ) : IsEven (f k) → k = 1 := by
  sorry

end even_quadratic_implies_k_eq_one_l3084_308401


namespace snowfall_rate_hamilton_l3084_308494

/-- Snowfall rates and depths in Kingston and Hamilton --/
theorem snowfall_rate_hamilton (
  kingston_initial : ℝ) (hamilton_initial : ℝ) 
  (duration : ℝ) (kingston_rate : ℝ) (hamilton_rate : ℝ) :
  kingston_initial = 12.1 →
  hamilton_initial = 18.6 →
  duration = 13 →
  kingston_rate = 2.6 →
  kingston_initial + kingston_rate * duration = hamilton_initial + hamilton_rate * duration →
  hamilton_rate = 2.1 := by
  sorry

end snowfall_rate_hamilton_l3084_308494


namespace henrys_initial_book_count_l3084_308490

/-- Calculates the initial number of books in Henry's collection --/
def initialBookCount (boxCount : ℕ) (booksPerBox : ℕ) (roomBooks : ℕ) (tableBooks : ℕ) (kitchenBooks : ℕ) (pickedUpBooks : ℕ) (remainingBooks : ℕ) : ℕ :=
  boxCount * booksPerBox + roomBooks + tableBooks + kitchenBooks - pickedUpBooks + remainingBooks

/-- Theorem stating that Henry's initial book count is 99 --/
theorem henrys_initial_book_count :
  initialBookCount 3 15 21 4 18 12 23 = 99 := by
  sorry

end henrys_initial_book_count_l3084_308490


namespace two_points_determine_line_l3084_308420

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem: Two distinct points determine a unique line
theorem two_points_determine_line (p1 p2 : Point2D) (h : p1 ≠ p2) :
  ∃! l : Line2D, pointOnLine p1 l ∧ pointOnLine p2 l :=
sorry

end two_points_determine_line_l3084_308420
