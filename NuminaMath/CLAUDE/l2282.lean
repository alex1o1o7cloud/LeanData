import Mathlib

namespace geometric_sequence_fourth_term_l2282_228289

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h1 : a 1 + a 2 = -1)
  (h2 : a 1 - a 3 = -3) :
  a 4 = -8 :=
sorry

end geometric_sequence_fourth_term_l2282_228289


namespace find_y_value_l2282_228226

theorem find_y_value (x y : ℝ) (h1 : x^2 - x + 3 = y + 3) (h2 : x = -5) (h3 : y > 0) : y = 30 := by
  sorry

end find_y_value_l2282_228226


namespace arrangement_count_l2282_228278

/-- Represents the number of people wearing each color -/
structure ColorCount where
  red : Nat
  yellow : Nat
  blue : Nat

/-- Represents the total number of people -/
def totalPeople (cc : ColorCount) : Nat :=
  cc.red + cc.yellow + cc.blue

/-- Calculates the number of valid arrangements -/
noncomputable def validArrangements (cc : ColorCount) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem arrangement_count (cc : ColorCount) : 
  cc.red = 2 → cc.yellow = 2 → cc.blue = 1 → 
  totalPeople cc = 5 → validArrangements cc = 48 := by
  sorry

end arrangement_count_l2282_228278


namespace equal_area_rectangles_width_l2282_228228

/-- Represents the dimensions of a rectangle in inches -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles_width (carol_rect jordan_rect : Rectangle) 
  (h1 : carol_rect.length = 15)
  (h2 : carol_rect.width = 20)
  (h3 : jordan_rect.length = 6 * 12)
  (h4 : area carol_rect = area jordan_rect) :
  jordan_rect.width = 300 / (6 * 12) := by
  sorry

end equal_area_rectangles_width_l2282_228228


namespace unique_function_property_l2282_228273

theorem unique_function_property (f : ℕ → ℕ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n)
  (h3 : ∀ n : ℕ, f (n + 1) > f n) :
  ∀ n : ℕ, f n = n :=
by sorry

end unique_function_property_l2282_228273


namespace unique_integer_sum_pair_l2282_228211

theorem unique_integer_sum_pair (a : ℕ → ℝ) (h1 : 1 < a 1 ∧ a 1 < 2) 
  (h2 : ∀ k : ℕ, a (k + 1) = a k + k / a k) :
  ∃! (i j : ℕ), i ≠ j ∧ ∃ m : ℤ, (a i + a j : ℝ) = m := by
  sorry

end unique_integer_sum_pair_l2282_228211


namespace probability_at_least_one_red_l2282_228220

theorem probability_at_least_one_red (prob_red_A prob_red_B : ℝ) :
  prob_red_A = 1/3 →
  prob_red_B = 1/2 →
  1 - (1 - prob_red_A) * (1 - prob_red_B) = 2/3 :=
by sorry

end probability_at_least_one_red_l2282_228220


namespace max_value_product_sum_l2282_228284

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end max_value_product_sum_l2282_228284


namespace sqrt_square_minus_sqrt_nine_plus_cube_root_eight_equals_one_l2282_228233

theorem sqrt_square_minus_sqrt_nine_plus_cube_root_eight_equals_one :
  (Real.sqrt 2)^2 - Real.sqrt 9 + (8 : ℝ)^(1/3) = 1 := by
  sorry

end sqrt_square_minus_sqrt_nine_plus_cube_root_eight_equals_one_l2282_228233


namespace imaginary_part_of_product_l2282_228265

theorem imaginary_part_of_product : Complex.im ((2 - Complex.I) * (4 + Complex.I)) = -2 := by
  sorry

end imaginary_part_of_product_l2282_228265


namespace polynomial_division_remainder_l2282_228262

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^4 + 14 * X^3 - 55 * X^2 - 73 * X + 65 = 
  (X^2 + 8 * X - 6) * q + (-477 * X + 323) := by
  sorry

end polynomial_division_remainder_l2282_228262


namespace square_root_of_1024_l2282_228292

theorem square_root_of_1024 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 1024) : x = 32 := by
  sorry

end square_root_of_1024_l2282_228292


namespace fred_has_eighteen_balloons_l2282_228286

/-- The number of blue balloons Sally has -/
def sally_balloons : ℕ := 6

/-- The factor by which Fred has more balloons than Sally -/
def fred_factor : ℕ := 3

/-- The number of blue balloons Fred has -/
def fred_balloons : ℕ := sally_balloons * fred_factor

/-- Theorem stating that Fred has 18 blue balloons -/
theorem fred_has_eighteen_balloons : fred_balloons = 18 := by
  sorry

end fred_has_eighteen_balloons_l2282_228286


namespace work_completion_time_b_l2282_228274

/-- The number of days it takes for worker b to complete a work alone,
    given that workers a and b together can finish the work in 16 days,
    and worker a alone can do the same work in 32 days. -/
theorem work_completion_time_b (work_rate_a_and_b : ℚ) (work_rate_a : ℚ) :
  work_rate_a_and_b = 1 / 16 →
  work_rate_a = 1 / 32 →
  (1 : ℚ) / (work_rate_a_and_b - work_rate_a) = 32 := by
  sorry

end work_completion_time_b_l2282_228274


namespace apps_remaining_proof_l2282_228282

/-- Calculates the number of remaining apps after deletions -/
def remaining_apps (total : ℕ) (gaming : ℕ) (deleted_utility : ℕ) : ℕ :=
  total - gaming - deleted_utility

/-- Theorem: Given 12 total apps, 5 gaming apps, and deleting 3 utility apps,
    the number of remaining apps is 4 -/
theorem apps_remaining_proof :
  remaining_apps 12 5 3 = 4 := by
  sorry

end apps_remaining_proof_l2282_228282


namespace bella_soccer_goals_l2282_228210

def goals_first_6 : List Nat := [5, 3, 2, 4, 1, 6]

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

theorem bella_soccer_goals :
  ∀ (g7 g8 : Nat),
    g7 < 10 →
    g8 < 10 →
    is_integer ((goals_first_6.sum + g7) / 7) →
    is_integer ((goals_first_6.sum + g7 + g8) / 8) →
    g7 * g8 = 28 := by
  sorry

end bella_soccer_goals_l2282_228210


namespace point_coordinate_sum_l2282_228291

/-- Given points A, B, and C in a 2D plane, with specific conditions on their coordinates and the lines connecting them, prove that the sum of certain coordinate values is 1. -/
theorem point_coordinate_sum (a b : ℝ) : 
  let A : ℝ × ℝ := (a, 5)
  let B : ℝ × ℝ := (2, 2 - b)
  let C : ℝ × ℝ := (4, 2)
  (A.2 = B.2) →  -- AB is parallel to x-axis
  (A.1 = C.1) →  -- AC is parallel to y-axis
  a + b = 1 := by
sorry


end point_coordinate_sum_l2282_228291


namespace composition_equation_solution_l2282_228252

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 9 * x + 6
  ∃ x : ℝ, δ (φ x) = 10 ∧ x = -23/36 := by
  sorry

end composition_equation_solution_l2282_228252


namespace no_solution_factorial_equation_l2282_228287

theorem no_solution_factorial_equation :
  ∀ (m n : ℕ), m.factorial + 48 ≠ 48 * (m + 1) * n := by
  sorry

end no_solution_factorial_equation_l2282_228287


namespace isosceles_triangle_perimeter_l2282_228219

/-- An isosceles triangle with side lengths satisfying a specific equation has a perimeter of either 10 or 11. -/
theorem isosceles_triangle_perimeter (x y : ℝ) : 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    ((a = x ∧ b = y) ∨ (a = y ∧ b = x)) ∧
    (a = b ∨ a + a = b ∨ b + b = a)) →  -- isosceles condition
  |x^2 - 9| + (y - 4)^2 = 0 →
  (x + y + min x y = 10) ∨ (x + y + min x y = 11) := by
sorry

end isosceles_triangle_perimeter_l2282_228219


namespace choose_four_from_six_eq_fifteen_l2282_228204

/-- The number of ways to choose 4 items from a set of 6 items, where the order doesn't matter -/
def choose_four_from_six : ℕ := Nat.choose 6 4

/-- Theorem stating that choosing 4 items from a set of 6 items results in 15 combinations -/
theorem choose_four_from_six_eq_fifteen : choose_four_from_six = 15 := by
  sorry

end choose_four_from_six_eq_fifteen_l2282_228204


namespace smallest_dual_base_palindrome_l2282_228276

/-- Returns true if the given number is a palindrome in the specified base -/
def isPalindrome (n : ℕ) (base : ℕ) : Bool := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∃ (n : ℕ), n > 7 ∧
    isPalindrome n 3 = true ∧
    isPalindrome n 5 = true ∧
    (∀ (m : ℕ), m > 7 ∧ m < n →
      isPalindrome m 3 = false ∨ isPalindrome m 5 = false) ∧
    n = 26 := by sorry

end smallest_dual_base_palindrome_l2282_228276


namespace smallest_n_with_four_trailing_zeros_l2282_228205

def is_divisible_by_10000 (n : ℕ) : Prop :=
  (n.choose 4) % 10000 = 0

theorem smallest_n_with_four_trailing_zeros : 
  ∀ k : ℕ, k ≥ 4 ∧ k < 8128 → ¬(is_divisible_by_10000 k) ∧ is_divisible_by_10000 8128 :=
sorry

end smallest_n_with_four_trailing_zeros_l2282_228205


namespace circus_tickets_cost_l2282_228283

/-- Given the cost per ticket and the number of tickets bought, 
    calculate the total amount spent on tickets. -/
def total_spent (cost_per_ticket : ℕ) (num_tickets : ℕ) : ℕ :=
  cost_per_ticket * num_tickets

/-- Theorem: If each ticket costs 44 dollars and 7 tickets are bought,
    the total amount spent is 308 dollars. -/
theorem circus_tickets_cost :
  let cost_per_ticket : ℕ := 44
  let num_tickets : ℕ := 7
  total_spent cost_per_ticket num_tickets = 308 := by
  sorry

end circus_tickets_cost_l2282_228283


namespace least_days_same_date_l2282_228254

/-- A calendar date represented by a day and a month -/
structure CalendarDate where
  day : Nat
  month : Nat

/-- Function to move a given number of days forward or backward from a date -/
def moveDays (date : CalendarDate) (days : Int) : CalendarDate :=
  sorry

/-- Predicate to check if two dates have the same day of the month -/
def sameDayOfMonth (date1 date2 : CalendarDate) : Prop :=
  date1.day = date2.day

theorem least_days_same_date :
  ∃ k : Nat, k > 0 ∧
    (∀ date : CalendarDate, sameDayOfMonth (moveDays date k) (moveDays date (-k))) ∧
    (∀ j : Nat, 0 < j → j < k →
      ∃ date : CalendarDate, ¬sameDayOfMonth (moveDays date j) (moveDays date (-j))) ∧
    k = 14 :=
  sorry

end least_days_same_date_l2282_228254


namespace rectangle_circle_equality_l2282_228214

/-- Given a rectangle with sides a and b, where a = 2b, and a circle with radius 3,
    if the perimeter of the rectangle equals the circumference of the circle,
    then a = 2π and b = π. -/
theorem rectangle_circle_equality (a b : ℝ) :
  a = 2 * b →
  2 * (a + b) = 2 * π * 3 →
  a = 2 * π ∧ b = π := by
sorry

end rectangle_circle_equality_l2282_228214


namespace tangent_at_negative_one_range_of_a_l2282_228232

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Define the condition for the shared tangent line
def shared_tangent (a : ℝ) (x₁ : ℝ) : Prop :=
  ∃ (x₂ : ℝ), f' x₁ * (x₂ - x₁) + f x₁ = g a x₂ ∧ f' x₁ = 2 * x₂

-- Theorem for part 1
theorem tangent_at_negative_one (a : ℝ) :
  shared_tangent a (-1) → a = 3 := by sorry

-- Theorem for part 2
theorem range_of_a :
  ∀ a : ℝ, (∃ x₁ : ℝ, shared_tangent a x₁) ↔ a ≥ -1 := by sorry

end tangent_at_negative_one_range_of_a_l2282_228232


namespace sum_and_decimal_shift_l2282_228258

theorem sum_and_decimal_shift (A B : ℝ) (h1 : A + B = 13.2) (h2 : 10 * A = B) : A = 1.2 ∧ B = 12 := by
  sorry

end sum_and_decimal_shift_l2282_228258


namespace components_exceed_quarter_square_l2282_228297

/-- Represents a square grid of size n × n -/
structure Grid (n : ℕ) where
  size : n > 8

/-- Represents a diagonal in a cell of the grid -/
inductive Diagonal
  | TopLeft
  | TopRight

/-- Represents the configuration of diagonals in the grid -/
def DiagonalConfig (n : ℕ) := Fin n → Fin n → Diagonal

/-- Represents a connected component in the grid -/
structure Component (n : ℕ) where
  cells : Set (Fin n × Fin n)
  is_connected : True  -- Simplified connectivity condition

/-- The number of connected components in a given diagonal configuration -/
def num_components (n : ℕ) (config : DiagonalConfig n) : ℕ := sorry

/-- Theorem stating that the number of components can exceed n²/4 for n > 8 -/
theorem components_exceed_quarter_square {n : ℕ} (grid : Grid n) :
  ∃ (config : DiagonalConfig n), num_components n config > n^2 / 4 := by sorry

end components_exceed_quarter_square_l2282_228297


namespace solution_set_f_less_than_3_range_of_a_for_nonempty_solution_l2282_228263

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x - 2|

-- Theorem 1: Solution set of f(x) < 3
theorem solution_set_f_less_than_3 :
  {x : ℝ | f x < 3} = {x : ℝ | -1/2 < x ∧ x < 5/2} :=
sorry

-- Theorem 2: Range of a for non-empty solution set
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x < a) → a > 2 :=
sorry

end solution_set_f_less_than_3_range_of_a_for_nonempty_solution_l2282_228263


namespace roller_coaster_tickets_l2282_228213

theorem roller_coaster_tickets : ∃ x : ℕ, 
  (∀ y : ℕ, y = 3 → 7 * x + 4 * y = 47) → x = 5 := by
  sorry

end roller_coaster_tickets_l2282_228213


namespace batsman_average_after_17th_inning_l2282_228279

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  lastInningRuns : ℕ
  averageIncrease : ℚ

/-- Calculates the average score of a batsman -/
def calculateAverage (b : Batsman) : ℚ :=
  (b.totalRuns : ℚ) / b.innings

/-- Theorem stating the batsman's average after the 17th inning -/
theorem batsman_average_after_17th_inning (b : Batsman)
  (h1 : b.innings = 17)
  (h2 : b.lastInningRuns = 90)
  (h3 : b.averageIncrease = 3)
  (h4 : calculateAverage b = calculateAverage { b with
    innings := b.innings - 1,
    totalRuns := b.totalRuns - b.lastInningRuns
  } + b.averageIncrease) :
  calculateAverage b = 42 := by
  sorry


end batsman_average_after_17th_inning_l2282_228279


namespace product_closure_l2282_228201

def A : Set ℤ := {n | ∃ t s : ℤ, n = t^2 + s^2}

theorem product_closure (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := by
  sorry

end product_closure_l2282_228201


namespace multiply_polynomial_l2282_228225

theorem multiply_polynomial (x : ℝ) : 
  (x^4 + 24*x^2 + 576) * (x^2 - 24) = x^6 - 13824 := by
  sorry

end multiply_polynomial_l2282_228225


namespace haley_initial_marbles_l2282_228215

/-- The number of marbles Haley gives to each boy -/
def marbles_per_boy : ℕ := 2

/-- The number of boys in Haley's class who receive marbles -/
def number_of_boys : ℕ := 14

/-- The total number of marbles Haley had initially -/
def total_marbles : ℕ := marbles_per_boy * number_of_boys

theorem haley_initial_marbles : total_marbles = 28 := by
  sorry

end haley_initial_marbles_l2282_228215


namespace tens_digit_of_8_pow_2023_l2282_228288

-- Define a function to get the last two digits of 8^n
def lastTwoDigits (n : ℕ) : ℕ := 8^n % 100

-- Define the cycle of last two digits
def lastTwoDigitsCycle : List ℕ := [8, 64, 12, 96, 68, 44, 52, 16, 28, 24]

-- Theorem statement
theorem tens_digit_of_8_pow_2023 :
  (lastTwoDigits 2023 / 10) % 10 = 1 :=
sorry

end tens_digit_of_8_pow_2023_l2282_228288


namespace trigonometric_problem_l2282_228245

theorem trigonometric_problem (α : ℝ) 
  (h1 : Real.sin (α + π/3) + Real.sin α = 9 * Real.sqrt 7 / 14)
  (h2 : 0 < α)
  (h3 : α < π/3) :
  (Real.sin α = 2 * Real.sqrt 7 / 7) ∧ 
  (Real.cos (2*α - π/4) = (4 * Real.sqrt 6 - Real.sqrt 2) / 14) := by
  sorry

end trigonometric_problem_l2282_228245


namespace common_roots_product_l2282_228261

/-- Given two cubic equations with two common roots, prove that the product of these common roots is 10 * ∛2 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (u v w t : ℝ), 
    (u^3 + C*u^2 + 20 = 0) ∧ 
    (v^3 + C*v^2 + 20 = 0) ∧ 
    (w^3 + C*w^2 + 20 = 0) ∧
    (u^3 + D*u + 100 = 0) ∧ 
    (v^3 + D*v + 100 = 0) ∧ 
    (t^3 + D*t + 100 = 0) ∧
    (u ≠ v) ∧ 
    (u * v = 10 * (2 : ℝ)^(1/3)) := by
  sorry

end common_roots_product_l2282_228261


namespace power_three_times_three_l2282_228253

theorem power_three_times_three (x : ℝ) : x^3 * x^3 = x^6 := by
  sorry

end power_three_times_three_l2282_228253


namespace mary_income_90_percent_of_juan_l2282_228295

/-- Represents the income of an individual -/
structure Income where
  amount : ℝ
  amount_pos : amount > 0

/-- The relationship between incomes of Mary, Tim, Juan, Sophia, and Alex -/
structure IncomeRelationship where
  alex : Income
  sophia : Income
  juan : Income
  tim : Income
  mary : Income
  sophia_alex : sophia.amount = 1.25 * alex.amount
  juan_sophia : juan.amount = 0.7 * sophia.amount
  tim_juan : tim.amount = 0.6 * juan.amount
  mary_tim : mary.amount = 1.5 * tim.amount

/-- Theorem stating that Mary's income is 90% of Juan's income -/
theorem mary_income_90_percent_of_juan (r : IncomeRelationship) : 
  r.mary.amount = 0.9 * r.juan.amount := by sorry

end mary_income_90_percent_of_juan_l2282_228295


namespace horse_race_equation_l2282_228237

/-- The speed of the good horse in miles per day -/
def good_horse_speed : ℕ := 200

/-- The speed of the slow horse in miles per day -/
def slow_horse_speed : ℕ := 120

/-- The number of days the slow horse starts earlier -/
def head_start : ℕ := 10

/-- The number of days it takes for the good horse to catch up -/
def catch_up_days : ℕ := sorry

theorem horse_race_equation :
  good_horse_speed * catch_up_days = slow_horse_speed * catch_up_days + slow_horse_speed * head_start :=
by sorry

end horse_race_equation_l2282_228237


namespace cube_volume_problem_l2282_228267

theorem cube_volume_problem (a : ℝ) (h : a > 0) :
  (a^3 : ℝ) - ((a - 1) * a * (a + 1)) = 5 →
  a^3 = 125 := by
sorry

end cube_volume_problem_l2282_228267


namespace smallest_n_with_abc_property_l2282_228241

def has_abc_property (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = Finset.range n →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a * b = c)

theorem smallest_n_with_abc_property :
  (∀ k < 243, ¬ has_abc_property k) ∧ has_abc_property 243 :=
sorry

end smallest_n_with_abc_property_l2282_228241


namespace valid_placements_correct_l2282_228277

/-- Represents a chess piece type -/
inductive ChessPiece
| Rook
| King
| Bishop
| Knight
| Queen

/-- Represents the size of the chessboard -/
def boardSize : Nat := 8

/-- Calculates the number of ways to place two identical pieces of the given type on an 8x8 chessboard such that they do not capture each other -/
def validPlacements (piece : ChessPiece) : Nat :=
  match piece with
  | ChessPiece.Rook => 1568
  | ChessPiece.King => 1806
  | ChessPiece.Bishop => 1972
  | ChessPiece.Knight => 1848
  | ChessPiece.Queen => 1980

/-- Theorem stating the correct number of valid placements for each piece type -/
theorem valid_placements_correct :
  (validPlacements ChessPiece.Rook = 1568) ∧
  (validPlacements ChessPiece.King = 1806) ∧
  (validPlacements ChessPiece.Bishop = 1972) ∧
  (validPlacements ChessPiece.Knight = 1848) ∧
  (validPlacements ChessPiece.Queen = 1980) :=
by sorry

end valid_placements_correct_l2282_228277


namespace diagonal_sum_is_161_l2282_228221

/-- Represents a multiplication grid with missing factors --/
structure MultiplicationGrid where
  /-- Products in the grid --/
  wp : ℕ
  xp : ℕ
  wr : ℕ
  zr : ℕ
  xs : ℕ
  vs : ℕ
  vq : ℕ
  yq : ℕ
  yt : ℕ

/-- The sum of diagonal elements in the multiplication grid --/
def diagonalSum (grid : MultiplicationGrid) : ℕ :=
  let p := 3  -- Derived from wp and xp
  let w := grid.wp / p
  let x := grid.xp / p
  let r := grid.wr / w
  let z := grid.zr / r
  let s := grid.xs / x
  let v := grid.vs / s
  let q := grid.vq / v
  let y := grid.yq / q
  let t := grid.yt / y
  v * p + w * q + x * r + y * s + z * t

/-- Theorem stating that the diagonal sum is 161 for the given grid --/
theorem diagonal_sum_is_161 (grid : MultiplicationGrid) 
  (h1 : grid.wp = 15) (h2 : grid.xp = 18) (h3 : grid.wr = 40) 
  (h4 : grid.zr = 56) (h5 : grid.xs = 60) (h6 : grid.vs = 20) 
  (h7 : grid.vq = 10) (h8 : grid.yq = 20) (h9 : grid.yt = 24) : 
  diagonalSum grid = 161 := by
  sorry

end diagonal_sum_is_161_l2282_228221


namespace inequality_solution_set_l2282_228280

theorem inequality_solution_set (x : ℝ) : 9 > -3 * x ↔ x > -3 := by sorry

end inequality_solution_set_l2282_228280


namespace train_crossing_time_l2282_228238

/-- The time taken for two trains to cross each other -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 120 →
  train_speed_kmh = 18 →
  (2 * train_length) / (2 * train_speed_kmh * (1000 / 3600)) = 24 := by
  sorry

#check train_crossing_time

end train_crossing_time_l2282_228238


namespace city_population_ratio_l2282_228240

def population_ratio (pop_Z : ℝ) : Prop :=
  let pop_Y := 2.5 * pop_Z
  let pop_X := 6 * pop_Y
  let pop_A := 3 * pop_X
  let pop_B := 4 * pop_Y
  (pop_X / (pop_Z + pop_B)) = 15 / 11

theorem city_population_ratio :
  ∀ pop_Z : ℝ, pop_Z > 0 → population_ratio pop_Z :=
by
  sorry

end city_population_ratio_l2282_228240


namespace sufficient_but_not_necessary_l2282_228203

theorem sufficient_but_not_necessary (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end sufficient_but_not_necessary_l2282_228203


namespace coordinates_sum_of_A_l2282_228296

/-- Given points B and C, and the condition that AC/AB = BC/AB = 1/3, 
    prove that the sum of coordinates of point A is -22 -/
theorem coordinates_sum_of_A (B C : ℝ × ℝ) (h : B = (2, -3) ∧ C = (-2, 6)) :
  let A : ℝ × ℝ := (3 * C.1 - 2 * B.1, 3 * C.2 - 2 * B.2)
  (A.1 + A.2 : ℝ) = -22 := by
  sorry

end coordinates_sum_of_A_l2282_228296


namespace isabel_paper_left_l2282_228227

/-- The number of pieces of paper Isabel bought in her first purchase -/
def first_purchase : ℕ := 900

/-- The number of pieces of paper Isabel bought in her second purchase -/
def second_purchase : ℕ := 300

/-- The number of pieces of paper Isabel used for a school project -/
def school_project : ℕ := 156

/-- The number of pieces of paper Isabel used for her artwork -/
def artwork : ℕ := 97

/-- The number of pieces of paper Isabel used for writing letters -/
def letters : ℕ := 45

/-- The theorem stating that Isabel has 902 pieces of paper left -/
theorem isabel_paper_left : 
  first_purchase + second_purchase - (school_project + artwork + letters) = 902 := by
  sorry

end isabel_paper_left_l2282_228227


namespace sector_central_angle_invariant_l2282_228269

/-- Theorem: If both the radius and arc length of a circular sector are doubled, then the central angle of the sector remains unchanged. -/
theorem sector_central_angle_invariant 
  (r₁ r₂ l₁ l₂ θ₁ θ₂ : Real) 
  (h1 : r₂ = 2 * r₁) 
  (h2 : l₂ = 2 * l₁) 
  (h3 : θ₁ = l₁ / r₁) 
  (h4 : θ₂ = l₂ / r₂) : 
  θ₁ = θ₂ := by
sorry

end sector_central_angle_invariant_l2282_228269


namespace moon_temperature_difference_l2282_228244

theorem moon_temperature_difference : 
  let noon_temp : ℤ := 10
  let midnight_temp : ℤ := -150
  noon_temp - midnight_temp = 160 := by
sorry

end moon_temperature_difference_l2282_228244


namespace rectangular_solid_volume_l2282_228206

theorem rectangular_solid_volume (x y z : ℝ) 
  (h1 : x * y = 3) 
  (h2 : x * z = 5) 
  (h3 : y * z = 15) : 
  x * y * z = 15 := by
  sorry

end rectangular_solid_volume_l2282_228206


namespace geometric_sequence_ratio_l2282_228235

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  q ≠ 1 →  -- common ratio is not 1
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula for geometric sequence
  (2 * a 5 = (a 2 + 3 * a 8) / 2) →  -- arithmetic sequence condition
  (3 * S 3) / (S 6) = 9 / 4 := by
  sorry

end geometric_sequence_ratio_l2282_228235


namespace calculator_sequence_101_l2282_228234

def calculator_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 7
  | n + 1 => 1 / (1 - calculator_sequence n)

theorem calculator_sequence_101 : calculator_sequence 101 = 6 / 7 := by
  sorry

end calculator_sequence_101_l2282_228234


namespace problem_statement_l2282_228268

open Real

theorem problem_statement :
  (∀ x ∈ Set.Ioo (-π/2) 0, sin x > x) ∧
  ¬(Set.Ioo 0 1 = {x | log (1 - x) / log 10 < 1}) := by
  sorry

end problem_statement_l2282_228268


namespace members_playing_both_l2282_228229

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  badminton : ℕ
  tennis : ℕ
  neither : ℕ

/-- Calculate the number of members playing both badminton and tennis -/
def playBoth (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - (club.total - club.neither)

/-- Theorem stating the number of members playing both sports in the given scenario -/
theorem members_playing_both (club : SportsClub) 
  (h1 : club.total = 30)
  (h2 : club.badminton = 18)
  (h3 : club.tennis = 19)
  (h4 : club.neither = 2) :
  playBoth club = 9 := by
  sorry

#eval playBoth { total := 30, badminton := 18, tennis := 19, neither := 2 }

end members_playing_both_l2282_228229


namespace rower_upstream_speed_man_rowing_upstream_speed_l2282_228224

/-- Calculates the upstream speed of a rower given their still water speed and downstream speed -/
theorem rower_upstream_speed (v_still : ℝ) (v_downstream : ℝ) : 
  v_still > 0 → v_downstream > v_still → 
  2 * v_still - v_downstream = v_still - (v_downstream - v_still) := by
  sorry

/-- The specific problem instance -/
theorem man_rowing_upstream_speed : 
  let v_still : ℝ := 33
  let v_downstream : ℝ := 40
  2 * v_still - v_downstream = 26 := by
  sorry

end rower_upstream_speed_man_rowing_upstream_speed_l2282_228224


namespace cost_difference_white_brown_socks_l2282_228270

-- Define the cost of two white socks in cents
def cost_two_white_socks : ℕ := 45

-- Define the cost of 15 brown socks in cents
def cost_fifteen_brown_socks : ℕ := 300

-- Define the number of brown socks
def num_brown_socks : ℕ := 15

-- Theorem to prove
theorem cost_difference_white_brown_socks : 
  cost_two_white_socks - (cost_fifteen_brown_socks / num_brown_socks) = 25 := by
  sorry

end cost_difference_white_brown_socks_l2282_228270


namespace base_10_to_base_7_157_l2282_228251

def base_7_digit (n : Nat) : Char :=
  if n < 7 then Char.ofNat (n + 48) else Char.ofNat (n + 55)

def to_base_7 (n : Nat) : List Char :=
  if n < 7 then [base_7_digit n]
  else base_7_digit (n % 7) :: to_base_7 (n / 7)

theorem base_10_to_base_7_157 :
  to_base_7 157 = ['3', '1', '3'] := by sorry

end base_10_to_base_7_157_l2282_228251


namespace representations_of_2022_l2282_228223

/-- Represents a sequence of consecutive natural numbers. -/
structure ConsecutiveSequence where
  start : ℕ
  length : ℕ

/-- The sum of a consecutive sequence of natural numbers. -/
def sum_consecutive (seq : ConsecutiveSequence) : ℕ :=
  seq.length * (2 * seq.start + seq.length - 1) / 2

/-- Checks if a consecutive sequence sums to a given target. -/
def is_valid_representation (seq : ConsecutiveSequence) (target : ℕ) : Prop :=
  sum_consecutive seq = target

theorem representations_of_2022 :
  ∀ (seq : ConsecutiveSequence),
    is_valid_representation seq 2022 ↔
      (seq.start = 673 ∧ seq.length = 3) ∨
      (seq.start = 504 ∧ seq.length = 4) ∨
      (seq.start = 163 ∧ seq.length = 12) :=
by sorry

end representations_of_2022_l2282_228223


namespace min_distance_squared_l2282_228231

/-- The line on which point P(x,y) moves --/
def line (x y : ℝ) : Prop := x - y - 1 = 0

/-- The distance function from point (x,y) to (2,2) --/
def distance_squared (x y : ℝ) : ℝ := (x - 2)^2 + (y - 2)^2

/-- Theorem stating the minimum value of the distance function --/
theorem min_distance_squared :
  ∃ (min : ℝ), min = 1/2 ∧ 
  (∀ x y : ℝ, line x y → distance_squared x y ≥ min) ∧
  (∃ x y : ℝ, line x y ∧ distance_squared x y = min) :=
sorry

end min_distance_squared_l2282_228231


namespace get_ready_time_l2282_228209

/-- The time it takes Jack to put on his own shoes, in minutes. -/
def jack_shoes_time : ℕ := 4

/-- The additional time it takes Jack to help a toddler with their shoes, in minutes. -/
def additional_toddler_time : ℕ := 3

/-- The number of toddlers Jack needs to help. -/
def number_of_toddlers : ℕ := 2

/-- The total time it takes for Jack and his toddlers to get ready, in minutes. -/
def total_time : ℕ := jack_shoes_time + number_of_toddlers * (jack_shoes_time + additional_toddler_time)

theorem get_ready_time : total_time = 18 := by
  sorry

end get_ready_time_l2282_228209


namespace square_perimeter_l2282_228260

theorem square_perimeter (total_area common_area circle_area : ℝ) 
  (h1 : total_area = 329)
  (h2 : common_area = 101)
  (h3 : circle_area = 234) :
  let square_area := total_area + common_area - circle_area
  let side_length := Real.sqrt square_area
  4 * side_length = 56 := by sorry

end square_perimeter_l2282_228260


namespace quadrilateral_is_trapezoid_l2282_228217

/-- A quadrilateral with two parallel sides of different lengths is a trapezoid -/
def is_trapezoid (A B C D : ℝ × ℝ) : Prop :=
  ∃ (l₁ l₂ : ℝ), l₁ ≠ l₂ ∧ 
  (B.1 - A.1) / (B.2 - A.2) = (D.1 - C.1) / (D.2 - C.2) ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = l₁^2 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = l₂^2

/-- The quadratic equation whose roots are the lengths of AB and CD -/
def length_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - 3*m*x + 2*m^2 + m - 2 = 0

theorem quadrilateral_is_trapezoid (A B C D : ℝ × ℝ) (m : ℝ) :
  (∃ (l₁ l₂ : ℝ), 
    length_equation m l₁ ∧ 
    length_equation m l₂ ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = l₁^2 ∧
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = l₂^2 ∧
    (B.1 - A.1) / (B.2 - A.2) = (D.1 - C.1) / (D.2 - C.2)) →
  is_trapezoid A B C D :=
sorry

end quadrilateral_is_trapezoid_l2282_228217


namespace canoe_production_sum_l2282_228299

def geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * r^(n - 1)

def sum_geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  (a * (r^n - 1)) / (r - 1)

theorem canoe_production_sum :
  let a := 10
  let r := 3
  let n := 4
  sum_geometric_sequence a r n = 400 := by
sorry

end canoe_production_sum_l2282_228299


namespace range_when_p_true_range_when_p_and_q_true_l2282_228200

-- Define proposition p
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 3*x + m = 0

-- Define proposition q
def is_ellipse_with_x_foci (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (9 - m) + y^2 / (m - 2) = 1 ∧ 
  9 - m > 0 ∧ m - 2 > 0 ∧ 9 - m > m - 2

-- Theorem 1
theorem range_when_p_true (m : ℝ) :
  has_real_roots m → m ≤ 9/4 := by sorry

-- Theorem 2
theorem range_when_p_and_q_true (m : ℝ) :
  has_real_roots m ∧ is_ellipse_with_x_foci m → 2 < m ∧ m ≤ 9/4 := by sorry

end range_when_p_true_range_when_p_and_q_true_l2282_228200


namespace simplify_expression_l2282_228246

theorem simplify_expression (x : ℝ) : 120 * x - 52 * x = 68 * x := by
  sorry

end simplify_expression_l2282_228246


namespace student_ticket_price_is_correct_l2282_228222

/-- Represents the ticket sales data for a single day -/
structure DaySales where
  senior : ℕ
  student : ℕ
  adult : ℕ
  total : ℚ

/-- Represents the price changes for a day -/
structure PriceChange where
  senior : ℚ
  student : ℚ
  adult : ℚ

/-- Finds the initial price of a student ticket given the sales data and price changes -/
def find_student_ticket_price (sales : Vector DaySales 5) (day4_change : PriceChange) (day5_change : PriceChange) : ℚ :=
  sorry

/-- The main theorem stating that the initial price of a student ticket is approximately $8.83 -/
theorem student_ticket_price_is_correct (sales : Vector DaySales 5) (day4_change : PriceChange) (day5_change : PriceChange) :
  let price := find_student_ticket_price sales day4_change day5_change
  abs (price - 8.83) < 0.01 := by sorry

end student_ticket_price_is_correct_l2282_228222


namespace parallel_lines_slope_l2282_228256

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Two lines are parallel if and only if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem parallel_lines_slope (k : ℝ) :
  let line1 : Line := { slope := k, yIntercept := -7 }
  let line2 : Line := { slope := -3, yIntercept := 4 }
  parallel line1 line2 → k = -3 := by
  sorry

end parallel_lines_slope_l2282_228256


namespace sequence_sum_l2282_228264

/-- Given a geometric sequence a and an arithmetic sequence b,
    if 2a₃ - a₂a₄ = 0 and b₃ = a₃, then the sum of the first 5 terms of b is 10 -/
theorem sequence_sum (a b : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1)  -- a is geometric
  (h_arith : ∀ n : ℕ, b (n + 1) - b n = b 2 - b 1)  -- b is arithmetic
  (h_eq : 2 * a 3 - a 2 * a 4 = 0)
  (h_b3 : b 3 = a 3) :
  (b 1 + b 2 + b 3 + b 4 + b 5) = 10 := by
sorry

end sequence_sum_l2282_228264


namespace spongebob_earnings_l2282_228236

/-- Represents the sales data for a single item --/
structure ItemSales where
  quantity : ℕ
  price : ℚ

/-- Calculates the total earnings for a single item --/
def itemEarnings (item : ItemSales) : ℚ :=
  item.quantity * item.price

/-- Represents all sales data for the day --/
structure DailySales where
  burgers : ItemSales
  largeFries : ItemSales
  smallFries : ItemSales
  sodas : ItemSales
  milkshakes : ItemSales
  softServeCones : ItemSales

/-- Calculates the total earnings for the day --/
def totalEarnings (sales : DailySales) : ℚ :=
  itemEarnings sales.burgers +
  itemEarnings sales.largeFries +
  itemEarnings sales.smallFries +
  itemEarnings sales.sodas +
  itemEarnings sales.milkshakes +
  itemEarnings sales.softServeCones

/-- Spongebob's sales data for the day --/
def spongebobSales : DailySales :=
  { burgers := { quantity := 30, price := 2.5 }
  , largeFries := { quantity := 12, price := 1.75 }
  , smallFries := { quantity := 20, price := 1.25 }
  , sodas := { quantity := 50, price := 1 }
  , milkshakes := { quantity := 18, price := 2.85 }
  , softServeCones := { quantity := 40, price := 1.3 }
  }

theorem spongebob_earnings :
  totalEarnings spongebobSales = 274.3 := by
  sorry

end spongebob_earnings_l2282_228236


namespace candy_distribution_l2282_228257

theorem candy_distribution (total_candy : ℕ) (candy_per_bag : ℕ) (num_bags : ℕ) : 
  total_candy = 42 → 
  candy_per_bag = 21 → 
  total_candy = num_bags * candy_per_bag → 
  num_bags = 2 := by
sorry

end candy_distribution_l2282_228257


namespace sandy_shopping_money_l2282_228298

theorem sandy_shopping_money (remaining_money : ℝ) (spent_percentage : ℝ) (h1 : remaining_money = 224) (h2 : spent_percentage = 0.3) :
  let initial_money := remaining_money / (1 - spent_percentage)
  initial_money = 320 := by
sorry

end sandy_shopping_money_l2282_228298


namespace kellys_games_l2282_228285

/-- Kelly's Nintendo games problem -/
theorem kellys_games (initial_games given_away_games : ℕ) : 
  initial_games = 121 → given_away_games = 99 → 
  initial_games - given_away_games = 22 := by
  sorry

end kellys_games_l2282_228285


namespace solve_for_a_l2282_228218

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (2 * x + a) ^ 2

-- State the theorem
theorem solve_for_a (a : ℝ) : 
  (∀ x, deriv (f a) x = 8 * x + 4 * a) → 
  deriv (f a) 2 = 20 → 
  a = 1 := by sorry

end solve_for_a_l2282_228218


namespace point_C_coordinates_l2282_228216

/-- Given points A(-1,2) and B(2,8), if vector AB = 3 * vector AC, 
    then C has coordinates (0,4) -/
theorem point_C_coordinates 
  (A B C : ℝ × ℝ) 
  (h1 : A = (-1, 2)) 
  (h2 : B = (2, 8)) 
  (h3 : B - A = 3 • (C - A)) : 
  C = (0, 4) := by
sorry

end point_C_coordinates_l2282_228216


namespace quadratic_minimum_l2282_228272

theorem quadratic_minimum (x : ℝ) : 
  (∀ x, x^2 - 4*x + 3 ≥ -1) ∧ (∃ x, x^2 - 4*x + 3 = -1) := by
  sorry

end quadratic_minimum_l2282_228272


namespace gcf_64_144_l2282_228294

theorem gcf_64_144 : Nat.gcd 64 144 = 16 := by
  sorry

end gcf_64_144_l2282_228294


namespace stating_volume_division_ratio_l2282_228239

/-- Represents a truncated triangular pyramid -/
structure TruncatedTriangularPyramid where
  -- The ratio of corresponding sides of the upper and lower bases
  base_ratio : ℝ
  -- Assume base_ratio > 0
  base_ratio_pos : base_ratio > 0

/-- 
  Theorem stating that for a truncated triangular pyramid with base ratio 1:2,
  a plane drawn through a side of the upper base parallel to the opposite lateral edge
  divides the volume in the ratio 3:4
-/
theorem volume_division_ratio 
  (pyramid : TruncatedTriangularPyramid) 
  (h_ratio : pyramid.base_ratio = 1/2) :
  ∃ (v1 v2 : ℝ), v1 > 0 ∧ v2 > 0 ∧ v1 / v2 = 3/4 := by
  sorry

end stating_volume_division_ratio_l2282_228239


namespace probability_at_least_two_same_l2282_228255

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability of at least two dice showing the same number when rolling 8 fair 8-sided dice -/
theorem probability_at_least_two_same : 
  (1 - (Nat.factorial num_dice : ℚ) / (num_sides ^ num_dice : ℚ)) = 16736996 / 16777216 := by
  sorry

end probability_at_least_two_same_l2282_228255


namespace sphere_tangency_loci_l2282_228290

/-- Given a sphere of radius R touching a plane, and spheres of radius r
    touching both the given sphere and the plane, this theorem proves the radii
    of the circles formed by the centers and points of tangency of the r-radius spheres. -/
theorem sphere_tangency_loci (R r : ℝ) (h : R > 0) (h' : r > 0) :
  ∃ (center_locus tangent_plane_locus tangent_sphere_locus : ℝ),
    center_locus = 2 * Real.sqrt (R * r) ∧
    tangent_plane_locus = 2 * Real.sqrt (R * r) ∧
    tangent_sphere_locus = (2 * R * Real.sqrt (R * r)) / (R + r) :=
sorry

end sphere_tangency_loci_l2282_228290


namespace prime_square_plus_13_divisibility_l2282_228259

theorem prime_square_plus_13_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℕ, p^2 + 13 = 2*k + 2 := by
sorry

end prime_square_plus_13_divisibility_l2282_228259


namespace ellipse_eccentricity_l2282_228271

-- Define the polynomial
def p (z : ℂ) : ℂ := (z - 2) * (z^2 + 4*z + 10) * (z^2 + 6*z + 13)

-- Define the set of solutions
def solutions : Set ℂ := {z : ℂ | p z = 0}

-- Define the ellipse passing through the solutions
def E : Set ℂ := sorry

-- Define eccentricity
def eccentricity (E : Set ℂ) : ℝ := sorry

-- Theorem statement
theorem ellipse_eccentricity : eccentricity E = Real.sqrt (4/25) := by sorry

end ellipse_eccentricity_l2282_228271


namespace expression_simplification_l2282_228242

theorem expression_simplification 
  (x y z p q r : ℝ) 
  (hx : x ≠ p) 
  (hy : y ≠ q) 
  (hz : z ≠ r) 
  (hpq : p ≠ q) 
  (hqr : q ≠ r) 
  (hpr : p ≠ r) : 
  (2 * (x - p)) / (3 * (r - z)) * 
  (2 * (y - q)) / (3 * (p - x)) * 
  (2 * (z - r)) / (3 * (q - y)) = -8 / 27 := by
  sorry

end expression_simplification_l2282_228242


namespace compound_interest_problem_l2282_228207

/-- Proves that given the conditions, the principal amount is 20,000 --/
theorem compound_interest_problem (P : ℝ) : 
  P * ((1 + 0.1)^4 - (1 + 0.2)^2) = 482 → P = 20000 := by
  sorry

end compound_interest_problem_l2282_228207


namespace power_equality_l2282_228248

theorem power_equality (x : ℝ) : (1/8 : ℝ) * 2^36 = 4^x → x = 16.5 := by
  sorry

end power_equality_l2282_228248


namespace particle_position_at_5pm_l2282_228247

-- Define the particle's position as a function of time
def particle_position (t : ℝ) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem particle_position_at_5pm :
  -- Given conditions
  (particle_position 7 = (1, 2)) →
  (particle_position 9 = (3, -2)) →
  -- Constant speed along a straight line (slope remains constant)
  (∀ t₁ t₂ t₃ t₄ : ℝ, t₁ ≠ t₂ ∧ t₃ ≠ t₄ →
    (particle_position t₂).1 - (particle_position t₁).1 ≠ 0 →
    ((particle_position t₂).2 - (particle_position t₁).2) / ((particle_position t₂).1 - (particle_position t₁).1) =
    ((particle_position t₄).2 - (particle_position t₃).2) / ((particle_position t₄).1 - (particle_position t₃).1)) →
  -- Conclusion
  particle_position 17 = (11, -18) :=
by sorry

end particle_position_at_5pm_l2282_228247


namespace series_sum_l2282_228230

/-- The series defined by the problem -/
def series (n : ℕ) : ℚ :=
  if n % 3 = 1 then 1 / (2^n)
  else if n % 3 = 0 then -1 / (2^n)
  else -1 / (2^n)

/-- The sum of the series -/
noncomputable def S : ℚ := ∑' n, series n

/-- The theorem to be proved -/
theorem series_sum : S / (10 * 81) = 2 / 7 := by
  sorry

end series_sum_l2282_228230


namespace complex_modulus_problem_l2282_228243

theorem complex_modulus_problem : 
  Complex.abs ((3 + Complex.I) / (1 - Complex.I)) = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l2282_228243


namespace square_expression_l2282_228202

theorem square_expression (x y : ℝ) (square : ℝ) :
  4 * x^2 * square = 81 * x^3 * y → square = (81/4) * x * y := by
  sorry

end square_expression_l2282_228202


namespace cat_food_consumed_by_wednesday_l2282_228250

/-- Represents the days of the week -/
inductive Day : Type
| Monday : Day
| Tuesday : Day
| Wednesday : Day
| Thursday : Day
| Friday : Day
| Saturday : Day
| Sunday : Day

/-- Calculates the number of days until all cat food is consumed -/
def daysUntilFoodConsumed (morningPortion : ℚ) (eveningPortion : ℚ) (fullCans : ℕ) (leftoverCan : ℚ) (leftoverExpiry : Day) : Day :=
  sorry

/-- Theorem stating that all cat food will be consumed by Wednesday -/
theorem cat_food_consumed_by_wednesday :
  let morningPortion : ℚ := 1/4
  let eveningPortion : ℚ := 1/6
  let fullCans : ℕ := 10
  let leftoverCan : ℚ := 1/2
  let leftoverExpiry : Day := Day.Tuesday
  daysUntilFoodConsumed morningPortion eveningPortion fullCans leftoverCan leftoverExpiry = Day.Wednesday :=
by sorry

end cat_food_consumed_by_wednesday_l2282_228250


namespace arithmetic_sequence_tenth_term_l2282_228275

/-- Given an arithmetic sequence where a₁ = 1 and a₃ = 5, prove that a₁₀ = 19. -/
theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 1 = 1)  -- First term is 1
  (h2 : a 3 = 5)  -- Third term is 5
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 10 = 19 := by
  sorry

end arithmetic_sequence_tenth_term_l2282_228275


namespace sum_of_solutions_is_three_pi_halves_l2282_228266

theorem sum_of_solutions_is_three_pi_halves :
  ∃ (x₁ x₂ : Real),
    0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi ∧
    0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi ∧
    (1 / Real.sin x₁ + 1 / Real.cos x₁ = 4) ∧
    (1 / Real.sin x₂ + 1 / Real.cos x₂ = 4) ∧
    x₁ + x₂ = 3 * Real.pi / 2 :=
by sorry

end sum_of_solutions_is_three_pi_halves_l2282_228266


namespace cube_sum_eq_product_l2282_228249

theorem cube_sum_eq_product (m : ℕ) :
  (m = 1 ∨ m = 2 → ¬∃ (x y z : ℕ+), x^3 + y^3 + z^3 = m * x * y * z) ∧
  (m = 3 → ∀ (x y z : ℕ+), x^3 + y^3 + z^3 = 3 * x * y * z ↔ x = y ∧ y = z) :=
by sorry

end cube_sum_eq_product_l2282_228249


namespace wood_measurement_equations_l2282_228212

/-- Represents the wood measurement problem from "The Mathematical Classic of Sunzi" -/
def wood_measurement_problem (x y : ℝ) : Prop :=
  (y - x = 4.5) ∧ (y / 2 = x - 1)

/-- The correct system of equations for the wood measurement problem -/
theorem wood_measurement_equations :
  ∃ x y : ℝ,
    (x > 0) ∧  -- Length of wood is positive
    (y > 0) ∧  -- Length of rope is positive
    (y > x) ∧  -- Rope is longer than wood
    wood_measurement_problem x y :=
sorry

end wood_measurement_equations_l2282_228212


namespace absolute_value_inequality_l2282_228293

theorem absolute_value_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -13/2 < x ∧ x < 7/2 := by
  sorry

end absolute_value_inequality_l2282_228293


namespace function_value_at_negative_one_l2282_228208

/-- Given a function f(x) = a*sin(x) + b*x^3 + 5 where f(1) = 3, prove that f(-1) = 7 -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin x + b * x^3 + 5) 
  (h2 : f 1 = 3) : 
  f (-1) = 7 := by
sorry

end function_value_at_negative_one_l2282_228208


namespace lens_savings_l2282_228281

/-- The price of the more expensive lens before discount -/
def original_price : ℝ := 300

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.20

/-- The price of the cheaper lens -/
def cheaper_price : ℝ := 220

/-- The discounted price of the more expensive lens -/
def discounted_price : ℝ := original_price * (1 - discount_rate)

/-- The amount saved by buying the cheaper lens -/
def savings : ℝ := discounted_price - cheaper_price

theorem lens_savings : savings = 20 := by
  sorry

end lens_savings_l2282_228281
