import Mathlib

namespace sector_central_angle_l275_27565

theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) :
  l / r = 2 := by
  sorry

end sector_central_angle_l275_27565


namespace interest_rate_proof_l275_27538

/-- Given a principal amount, time, and the difference between compound and simple interest,
    prove that the interest rate is 25%. -/
theorem interest_rate_proof (P t : ℝ) (diff : ℝ) : 
  P = 3600 → t = 2 → diff = 225 →
  ∃ r : ℝ, r = 25 ∧ 
    P * ((1 + r / 100) ^ t - 1) - (P * r * t / 100) = diff :=
by sorry

end interest_rate_proof_l275_27538


namespace sqrt_fraction_equality_l275_27512

theorem sqrt_fraction_equality : 
  (2 * Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5 := by
  sorry

end sqrt_fraction_equality_l275_27512


namespace silver_coin_value_proof_l275_27528

/-- The value of a silver coin -/
def silver_coin_value : ℝ := 25

theorem silver_coin_value_proof :
  let gold_coin_value : ℝ := 50
  let num_gold_coins : ℕ := 3
  let num_silver_coins : ℕ := 5
  let cash : ℝ := 30
  let total_value : ℝ := 305
  silver_coin_value = (total_value - gold_coin_value * num_gold_coins - cash) / num_silver_coins :=
by
  sorry

end silver_coin_value_proof_l275_27528


namespace nested_square_root_value_l275_27544

/-- Given that x is a real number satisfying x = √(2 + x), prove that x = 2 -/
theorem nested_square_root_value (x : ℝ) (h : x = Real.sqrt (2 + x)) : x = 2 := by
  sorry

end nested_square_root_value_l275_27544


namespace complex_sum_equals_negative_ten_i_l275_27503

/-- Prove that the sum of complex numbers (5-5i)+(-2-i)-(3+4i) equals -10i -/
theorem complex_sum_equals_negative_ten_i : (5 - 5*I) + (-2 - I) - (3 + 4*I) = -10*I := by
  sorry

end complex_sum_equals_negative_ten_i_l275_27503


namespace steves_matching_socks_l275_27595

theorem steves_matching_socks (total_socks : ℕ) (mismatching_socks : ℕ) 
  (h1 : total_socks = 25) 
  (h2 : mismatching_socks = 17) : 
  (total_socks - mismatching_socks) / 2 = 4 := by
  sorry

end steves_matching_socks_l275_27595


namespace quadratic_solution_difference_squared_l275_27587

theorem quadratic_solution_difference_squared :
  ∀ d e : ℝ,
  (5 * d^2 + 20 * d - 55 = 0) →
  (5 * e^2 + 20 * e - 55 = 0) →
  (d - e)^2 = 600 := by
  sorry

end quadratic_solution_difference_squared_l275_27587


namespace knicks_win_probability_l275_27519

/-- The probability of the Bulls winning a single game -/
def p : ℚ := 3/4

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The maximum number of games in the series -/
def max_games : ℕ := 2 * games_to_win - 1

/-- The probability of the Knicks winning the series in exactly 7 games -/
def knicks_win_in_seven : ℚ := 135/4096

theorem knicks_win_probability :
  knicks_win_in_seven = (Nat.choose 6 3 : ℚ) * (1 - p)^3 * p^3 * (1 - p) :=
sorry

end knicks_win_probability_l275_27519


namespace quadratic_roots_expression_l275_27577

theorem quadratic_roots_expression (a b : ℝ) : 
  (a^2 - a - 1 = 0) → (b^2 - b - 1 = 0) → (3*a^2 + 2*b^2 - 3*a - 2*b = 5) := by
  sorry

end quadratic_roots_expression_l275_27577


namespace solution_difference_squared_l275_27552

theorem solution_difference_squared (α β : ℝ) : 
  α ≠ β ∧ α^2 = 2*α + 1 ∧ β^2 = 2*β + 1 → (α - β)^2 = 8 := by sorry

end solution_difference_squared_l275_27552


namespace marks_vote_ratio_l275_27506

theorem marks_vote_ratio (total_voters_first_area : ℕ) (win_percentage : ℚ) (total_votes : ℕ) : 
  total_voters_first_area = 100000 →
  win_percentage = 70 / 100 →
  total_votes = 210000 →
  (total_votes - (total_voters_first_area * win_percentage).floor) / 
  ((total_voters_first_area * win_percentage).floor) = 2 := by
  sorry

end marks_vote_ratio_l275_27506


namespace smallest_k_for_divisible_difference_l275_27551

theorem smallest_k_for_divisible_difference : ∃ (k : ℕ), k > 0 ∧
  (∀ (M : Finset ℕ), M ⊆ Finset.range 20 → M.card ≥ k →
    ∃ (a b c d : ℕ), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      20 ∣ (a - b + c - d)) ∧
  (∀ (k' : ℕ), k' < k →
    ∃ (M : Finset ℕ), M ⊆ Finset.range 20 ∧ M.card = k' ∧
      ∀ (a b c d : ℕ), a ∈ M → b ∈ M → c ∈ M → d ∈ M →
        a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
        ¬(20 ∣ (a - b + c - d))) ∧
  k = 7 :=
by sorry

end smallest_k_for_divisible_difference_l275_27551


namespace tic_tac_toe_rounds_l275_27586

theorem tic_tac_toe_rounds (total_rounds : ℕ) (difference : ℕ) (william_wins harry_wins : ℕ) : 
  total_rounds = 15 → 
  difference = 5 → 
  william_wins = harry_wins + difference → 
  william_wins + harry_wins = total_rounds → 
  william_wins = 10 := by
sorry

end tic_tac_toe_rounds_l275_27586


namespace team_total_score_l275_27566

/-- Given a team of 10 people in a shooting competition, prove that their total score is 905 points. -/
theorem team_total_score 
  (team_size : ℕ) 
  (best_score : ℕ) 
  (hypothetical_best : ℕ) 
  (hypothetical_average : ℕ) :
  team_size = 10 →
  best_score = 95 →
  hypothetical_best = 110 →
  hypothetical_average = 92 →
  (hypothetical_best - best_score + (team_size * hypothetical_average)) = (team_size * 905) :=
by sorry

end team_total_score_l275_27566


namespace pentagon_square_side_ratio_l275_27554

/-- The ratio of the side length of a regular pentagon to the side length of a square 
    with the same perimeter -/
theorem pentagon_square_side_ratio : 
  ∀ (pentagon_side square_side : ℝ),
  pentagon_side > 0 → square_side > 0 →
  5 * pentagon_side = 20 →
  4 * square_side = 20 →
  pentagon_side / square_side = 4 / 5 := by
sorry

end pentagon_square_side_ratio_l275_27554


namespace arithmetic_sum_proof_l275_27539

/-- 
Given an arithmetic sequence with:
- first term a₁ = k² + 1
- common difference d = 1
- number of terms n = 2k + 1

Prove that the sum of the first 2k + 1 terms is k³ + (k + 1)³
-/
theorem arithmetic_sum_proof (k : ℕ) : 
  let a₁ : ℕ := k^2 + 1
  let d : ℕ := 1
  let n : ℕ := 2 * k + 1
  let S := n * (2 * a₁ + (n - 1) * d) / 2
  S = k^3 + (k + 1)^3 := by
  sorry

end arithmetic_sum_proof_l275_27539


namespace stratified_sample_young_employees_l275_27509

/-- Calculates the number of employees to be drawn from a specific age group in a stratified sample. -/
def stratifiedSampleSize (totalEmployees : ℕ) (groupSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (groupSize * sampleSize) / totalEmployees

/-- Proves that the number of employees no older than 45 to be drawn in a stratified sample is 15. -/
theorem stratified_sample_young_employees :
  let totalEmployees : ℕ := 200
  let youngEmployees : ℕ := 120
  let sampleSize : ℕ := 25
  stratifiedSampleSize totalEmployees youngEmployees sampleSize = 15 := by
  sorry


end stratified_sample_young_employees_l275_27509


namespace half_month_days_l275_27522

/-- Prove that given a 30-day month with specific mean profits, each half of the month contains 15 days -/
theorem half_month_days (total_days : ℕ) (mean_profit : ℚ) (first_half_mean : ℚ) (second_half_mean : ℚ) :
  total_days = 30 ∧ 
  mean_profit = 350 ∧ 
  first_half_mean = 275 ∧ 
  second_half_mean = 425 →
  ∃ (half_days : ℕ), half_days = 15 ∧ total_days = 2 * half_days :=
by sorry

end half_month_days_l275_27522


namespace circle_passes_through_M_and_has_same_center_l275_27589

-- Define the center of the given circle
def center : ℝ × ℝ := (2, -3)

-- Define the point M
def point_M : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle we want to prove
def circle_equation (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = 25

-- Theorem statement
theorem circle_passes_through_M_and_has_same_center :
  -- The circle passes through point M
  circle_equation point_M.1 point_M.2 ∧
  -- The circle has the same center as (x-2)^2 + (y+3)^2 = 16
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = 25 ↔ circle_equation x y :=
by sorry

end circle_passes_through_M_and_has_same_center_l275_27589


namespace candies_remaining_l275_27578

-- Define the number of candies for each color
def red_candies : ℕ := 50
def yellow_candies : ℕ := 3 * red_candies - 35
def blue_candies : ℕ := (2 * yellow_candies) / 3
def green_candies : ℕ := 20
def purple_candies : ℕ := green_candies / 2
def silver_candies : ℕ := 10

-- Define the number of candies Carlos ate
def carlos_ate : ℕ := yellow_candies + green_candies / 2

-- Define the total number of candies
def total_candies : ℕ := red_candies + yellow_candies + blue_candies + green_candies + purple_candies + silver_candies

-- Theorem statement
theorem candies_remaining : total_candies - carlos_ate = 156 := by
  sorry

end candies_remaining_l275_27578


namespace intersection_and_coefficients_l275_27576

def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def B : Set ℝ := {x | -3 < x ∧ x < 1}

theorem intersection_and_coefficients :
  (A ∩ B = {x | -1 < x ∧ x < 1}) ∧
  (∃ a b : ℝ, (∀ x : ℝ, x ∈ B ↔ 2*x^2 + a*x + b < 0) ∧ a = 3 ∧ b = 4) := by
  sorry

end intersection_and_coefficients_l275_27576


namespace length_a_prime_b_prime_l275_27590

/-- Given points A, B, and C, where A' and B' are the intersections of lines AC and BC with the line y = x respectively, the length of A'B' is (3√2)/10. -/
theorem length_a_prime_b_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 7) →
  B = (0, 14) →
  C = (3, 5) →
  (A'.1 = A'.2) →  -- A' is on y = x
  (B'.1 = B'.2) →  -- B' is on y = x
  (C.2 - A.2) / (C.1 - A.1) = (A'.2 - A.2) / (A'.1 - A.1) →  -- A' is on line AC
  (C.2 - B.2) / (C.1 - B.1) = (B'.2 - B.2) / (B'.1 - B.1) →  -- B' is on line BC
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = (3 * Real.sqrt 2) / 10 := by
sorry

end length_a_prime_b_prime_l275_27590


namespace shortest_side_of_triangle_l275_27560

theorem shortest_side_of_triangle (a b c : ℕ) (area : ℕ) : 
  a = 21 →
  a + b + c = 48 →
  area * area = 24 * 3 * (24 - b) * (b - 3) →
  b ≤ c →
  b = 10 :=
sorry

end shortest_side_of_triangle_l275_27560


namespace quadratic_roots_sum_of_reciprocal_squares_l275_27573

theorem quadratic_roots_sum_of_reciprocal_squares :
  ∀ (r s : ℝ), 
    (2 * r^2 + 3 * r - 5 = 0) →
    (2 * s^2 + 3 * s - 5 = 0) →
    (r ≠ s) →
    (1 / r^2 + 1 / s^2 = 29 / 25) :=
by
  sorry

end quadratic_roots_sum_of_reciprocal_squares_l275_27573


namespace basketball_cricket_students_l275_27504

theorem basketball_cricket_students (B C : Finset Nat) : 
  (B.card = 7) → 
  (C.card = 8) → 
  ((B ∩ C).card = 5) → 
  ((B ∪ C).card = 10) :=
by
  sorry

end basketball_cricket_students_l275_27504


namespace six_students_five_lectures_l275_27597

/-- The number of ways to assign students to lectures -/
def assignment_count (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: The number of ways to assign 6 students to 5 lectures is 5^6 -/
theorem six_students_five_lectures :
  assignment_count 6 5 = 5^6 := by
  sorry

end six_students_five_lectures_l275_27597


namespace simplest_square_root_l275_27568

theorem simplest_square_root :
  let options : List ℝ := [Real.sqrt 5, Real.sqrt 4, Real.sqrt 12, Real.sqrt (1/2)]
  ∀ x ∈ options, x ≠ Real.sqrt 5 → ∃ y : ℝ, y * y = x ∧ y ≠ x :=
by sorry

end simplest_square_root_l275_27568


namespace jake_weight_proof_l275_27502

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 152

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 212 - jake_weight

theorem jake_weight_proof :
  (jake_weight - 32 = 2 * sister_weight) ∧
  (jake_weight + sister_weight = 212) →
  jake_weight = 152 :=
by sorry

end jake_weight_proof_l275_27502


namespace exterior_angles_sum_360_l275_27505

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  /-- The number of sides in the polygon. -/
  sides : ℕ
  /-- Assumption that the polygon has at least 3 sides. -/
  sides_ge_three : sides ≥ 3

/-- The sum of interior angles of a polygon. -/
def sum_of_interior_angles (p : Polygon) : ℝ := sorry

/-- The sum of exterior angles of a polygon. -/
def sum_of_exterior_angles (p : Polygon) : ℝ := sorry

/-- Theorem: For any polygon, if the sum of its interior angles is 1440°, 
    then the sum of its exterior angles is 360°. -/
theorem exterior_angles_sum_360 (p : Polygon) :
  sum_of_interior_angles p = 1440 → sum_of_exterior_angles p = 360 := by
  sorry

end exterior_angles_sum_360_l275_27505


namespace paper_width_covering_cube_l275_27558

/-- Given a rectangular piece of paper covering a cube, prove the width of the paper. -/
theorem paper_width_covering_cube 
  (paper_length : ℝ) 
  (cube_volume : ℝ) 
  (h1 : paper_length = 48)
  (h2 : cube_volume = 8) : 
  ∃ (paper_width : ℝ), paper_width = 72 ∧ 
    paper_length * paper_width = 6 * (12 * (cube_volume ^ (1/3)))^2 :=
by sorry

end paper_width_covering_cube_l275_27558


namespace f_lower_bound_l275_27520

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + 2 * Real.exp (-x) + (a - 2) * x

theorem f_lower_bound (a : ℝ) :
  (∀ x > 0, f a x ≥ (a + 2) * Real.cos x) → a ≥ 2 :=
by sorry

end f_lower_bound_l275_27520


namespace probability_not_special_number_l275_27521

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a^b = n

def is_power_of_three_halves (n : ℕ) : Prop :=
  ∃ (k : ℕ), (3/2)^k = n

def count_special_numbers : ℕ := 20

theorem probability_not_special_number :
  (200 - count_special_numbers) / 200 = 9 / 10 := by
  sorry

#check probability_not_special_number

end probability_not_special_number_l275_27521


namespace fraction_count_l275_27572

-- Define a function to check if an expression is a fraction
def is_fraction (expr : String) : Bool :=
  match expr with
  | "1/x" => true
  | "x^2+5x" => false
  | "1/2x" => false
  | "a/(3-2a)" => true
  | "3.14/π" => false
  | _ => false

-- Define the list of expressions
def expressions : List String := ["1/x", "x^2+5x", "1/2x", "a/(3-2a)", "3.14/π"]

-- Theorem statement
theorem fraction_count : (expressions.filter is_fraction).length = 2 := by
  sorry

end fraction_count_l275_27572


namespace stratified_sample_grade12_l275_27549

/-- Calculates the number of grade 12 students to be selected in a stratified sample -/
theorem stratified_sample_grade12 (total : ℕ) (grade10 : ℕ) (grade11 : ℕ) (sample_size : ℕ) :
  total = 1500 →
  grade10 = 550 →
  grade11 = 450 →
  sample_size = 300 →
  (sample_size * (total - grade10 - grade11)) / total = 100 :=
by sorry

end stratified_sample_grade12_l275_27549


namespace percentage_commutation_l275_27524

theorem percentage_commutation (n : ℝ) (h : 0.3 * (0.4 * n) = 36) :
  0.4 * (0.3 * n) = 0.3 * (0.4 * n) := by
  sorry

end percentage_commutation_l275_27524


namespace odd_nines_composite_l275_27599

theorem odd_nines_composite (k : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 10^(2*k) - 9 = a * b :=
sorry

end odd_nines_composite_l275_27599


namespace rectangle_diagonal_corners_l275_27536

/-- Represents a domino on a rectangular grid -/
structure Domino where
  x : ℕ
  y : ℕ
  horizontal : Bool

/-- Represents a diagonal in a domino -/
structure Diagonal where
  domino : Domino
  startCorner : Bool  -- true if the diagonal starts at the top-left or bottom-right corner

/-- Represents a rectangular grid filled with dominoes -/
structure RectangularGrid where
  width : ℕ
  height : ℕ
  dominoes : List Domino
  diagonals : List Diagonal

/-- Check if two diagonals have common endpoints -/
def diagonalsShareEndpoint (d1 d2 : Diagonal) : Bool := sorry

/-- Check if a point is a corner of the rectangle -/
def isRectangleCorner (x y : ℕ) (grid : RectangularGrid) : Bool := sorry

/-- Check if a point is an endpoint of a diagonal -/
def isDiagonalEndpoint (x y : ℕ) (diagonal : Diagonal) : Bool := sorry

/-- The main theorem -/
theorem rectangle_diagonal_corners (grid : RectangularGrid) :
  (∀ d1 d2 : Diagonal, d1 ∈ grid.diagonals → d2 ∈ grid.diagonals → d1 ≠ d2 → ¬(diagonalsShareEndpoint d1 d2)) →
  (∃! (n : ℕ), n = 2 ∧ 
    ∃ (corners : List (ℕ × ℕ)), corners.length = n ∧
      (∀ (x y : ℕ), (x, y) ∈ corners ↔ 
        (isRectangleCorner x y grid ∧ 
         ∃ d : Diagonal, d ∈ grid.diagonals ∧ isDiagonalEndpoint x y d))) :=
by sorry

end rectangle_diagonal_corners_l275_27536


namespace intersection_points_on_circle_l275_27557

theorem intersection_points_on_circle :
  ∀ (x y : ℝ), 
    ((x + 2*y = 19 ∨ y + 2*x = 98) ∧ y = 1/x) →
    (x - 34)^2 + (y - 215/4)^2 = 49785/16 := by
  sorry

end intersection_points_on_circle_l275_27557


namespace largest_initial_number_l275_27556

/-- Represents a sequence of five additions -/
structure FiveAdditions (n : ℕ) :=
  (a₁ a₂ a₃ a₄ a₅ : ℕ)
  (sum_eq : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100)
  (not_div₁ : ¬(n % a₁ = 0))
  (not_div₂ : ¬((n + a₁) % a₂ = 0))
  (not_div₃ : ¬((n + a₁ + a₂) % a₃ = 0))
  (not_div₄ : ¬((n + a₁ + a₂ + a₃) % a₄ = 0))
  (not_div₅ : ¬((n + a₁ + a₂ + a₃ + a₄) % a₅ = 0))

/-- The main theorem stating that 89 is the largest initial number -/
theorem largest_initial_number :
  (∃ (f : FiveAdditions 89), True) ∧
  (∀ n > 89, ¬∃ (f : FiveAdditions n), True) :=
sorry

end largest_initial_number_l275_27556


namespace floor_times_self_equals_108_l275_27567

theorem floor_times_self_equals_108 :
  ∃! (x : ℝ), (⌊x⌋ : ℝ) * x = 108 ∧ x = 10.8 := by sorry

end floor_times_self_equals_108_l275_27567


namespace fraction_above_line_is_five_sixths_l275_27548

/-- A square in the coordinate plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A line in the coordinate plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The fraction of the square's area above a given line -/
def fractionAboveLine (s : Square) (l : Line) : ℝ := sorry

/-- The specific square from the problem -/
def problemSquare : Square :=
  { bottomLeft := (2, 1),
    topRight := (5, 4) }

/-- The specific line from the problem -/
def problemLine : Line :=
  { point1 := (2, 3),
    point2 := (5, 1) }

theorem fraction_above_line_is_five_sixths :
  fractionAboveLine problemSquare problemLine = 5/6 := by sorry

end fraction_above_line_is_five_sixths_l275_27548


namespace charity_duck_race_money_raised_l275_27545

/-- The amount of money raised in a charity rubber duck race -/
theorem charity_duck_race_money_raised
  (regular_price : ℚ)
  (large_price : ℚ)
  (regular_sold : ℕ)
  (large_sold : ℕ)
  (h1 : regular_price = 3)
  (h2 : large_price = 5)
  (h3 : regular_sold = 221)
  (h4 : large_sold = 185) :
  regular_price * regular_sold + large_price * large_sold = 1588 :=
by sorry

end charity_duck_race_money_raised_l275_27545


namespace tens_digit_of_3_to_2023_l275_27535

-- Define the cycle length of the last two digits of powers of 3
def cycleLengthPowersOf3 : ℕ := 20

-- Define the function that gives the last two digits of 3^n
def lastTwoDigits (n : ℕ) : ℕ := 3^n % 100

-- Define the function that gives the tens digit of a number
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_3_to_2023 :
  tensDigit (lastTwoDigits 2023) = 2 :=
sorry

end tens_digit_of_3_to_2023_l275_27535


namespace solution_set_product_l275_27582

/-- An odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable (a b : ℝ)
variable (f g : ℝ → ℝ)

/-- The solution set for f(x) > 0 -/
def SolutionSetF : Set ℝ := Set.Ioo (a^2) b

/-- The solution set for g(x) > 0 -/
def SolutionSetG : Set ℝ := Set.Ioo (a^2/2) (b/2)

/-- The conditions given in the problem -/
structure ProblemConditions where
  f_odd : OddFunction f
  g_odd : OddFunction g
  f_solution : SolutionSetF a b = {x | f x > 0}
  g_solution : SolutionSetG a b = {x | g x > 0}
  b_gt_2a_squared : b > 2 * (a^2)

/-- The theorem to be proved -/
theorem solution_set_product (h : ProblemConditions a b f g) :
  {x | f x * g x > 0} = Set.Ioo (-b/2) (-a^2) ∪ Set.Ioo (a^2) (b/2) := by
  sorry

end solution_set_product_l275_27582


namespace geometric_sequence_fifth_term_l275_27550

/-- Given a geometric sequence of positive integers where the first term is 5
    and the fourth term is 405, the fifth term is 405. -/
theorem geometric_sequence_fifth_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 5 →                            -- First term is 5
  a 4 = 405 →                          -- Fourth term is 405
  a 5 = 405 :=                         -- Fifth term is 405
by sorry

end geometric_sequence_fifth_term_l275_27550


namespace no_14_consecutive_divisible_by_primes_lt_13_exist_21_consecutive_divisible_by_primes_lt_17_l275_27508

/-- The set of primes less than 13 -/
def primes_lt_13 : Set Nat := {p | Nat.Prime p ∧ p < 13}

/-- The set of primes less than 17 -/
def primes_lt_17 : Set Nat := {p | Nat.Prime p ∧ p < 17}

/-- A function that checks if a number is divisible by any prime in a given set -/
def divisible_by_any_prime (n : Nat) (primes : Set Nat) : Prop :=
  ∃ p ∈ primes, n % p = 0

/-- Theorem stating that there do not exist 14 consecutive positive integers
    each divisible by a prime less than 13 -/
theorem no_14_consecutive_divisible_by_primes_lt_13 :
  ¬ ∃ start : Nat, ∀ i ∈ Finset.range 14, divisible_by_any_prime (start + i) primes_lt_13 :=
sorry

/-- Theorem stating that there exist 21 consecutive positive integers
    each divisible by a prime less than 17 -/
theorem exist_21_consecutive_divisible_by_primes_lt_17 :
  ∃ start : Nat, ∀ i ∈ Finset.range 21, divisible_by_any_prime (start + i) primes_lt_17 :=
sorry

end no_14_consecutive_divisible_by_primes_lt_13_exist_21_consecutive_divisible_by_primes_lt_17_l275_27508


namespace no_integers_between_sqrt_bounds_l275_27531

theorem no_integers_between_sqrt_bounds (n : ℕ+) :
  ¬∃ (x y : ℕ+), (Real.sqrt n + Real.sqrt (n + 1) < Real.sqrt x + Real.sqrt y) ∧
                  (Real.sqrt x + Real.sqrt y < Real.sqrt (4 * n + 2)) :=
by sorry

end no_integers_between_sqrt_bounds_l275_27531


namespace one_third_of_number_l275_27526

theorem one_third_of_number (x : ℝ) : 
  (1 / 3 : ℝ) * x = 130.00000000000003 → x = 390.0000000000001 := by
  sorry

end one_third_of_number_l275_27526


namespace five_dice_not_same_probability_l275_27541

theorem five_dice_not_same_probability : 
  let n : ℕ := 5  -- number of dice
  let s : ℕ := 6  -- number of sides on each die
  let total_outcomes : ℕ := s^n
  let same_number_outcomes : ℕ := s
  let prob_not_same : ℚ := 1 - (same_number_outcomes : ℚ) / total_outcomes
  prob_not_same = 1295 / 1296 :=
by sorry

end five_dice_not_same_probability_l275_27541


namespace notebook_price_l275_27579

theorem notebook_price :
  ∀ (s n c : ℕ),
  s > 18 →
  s ≤ 36 →
  c > n →
  s * n * c = 990 →
  c = 15 :=
by
  sorry

end notebook_price_l275_27579


namespace tan_alpha_3_implies_fraction_eq_5_6_l275_27546

theorem tan_alpha_3_implies_fraction_eq_5_6 (α : ℝ) (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 := by
  sorry

end tan_alpha_3_implies_fraction_eq_5_6_l275_27546


namespace exists_integer_point_with_distance_l275_27500

theorem exists_integer_point_with_distance : ∃ (x y : ℤ),
  (x : ℝ)^2 + (y : ℝ)^2 = 2 * 2017^2 + 2 * 2018^2 := by
  sorry

end exists_integer_point_with_distance_l275_27500


namespace c_most_suitable_l275_27518

-- Define the structure for an athlete
structure Athlete where
  name : String
  average : ℝ
  variance : ℝ

-- Define the list of athletes
def athletes : List Athlete := [
  ⟨"A", 169, 6.0⟩,
  ⟨"B", 168, 17.3⟩,
  ⟨"C", 169, 5.0⟩,
  ⟨"D", 168, 19.5⟩
]

-- Function to determine if an athlete is suitable
def isSuitable (a : Athlete) : Prop :=
  ∀ b ∈ athletes, 
    a.average ≥ b.average ∧ 
    (a.average = b.average → a.variance ≤ b.variance)

-- Theorem stating that C is the most suitable candidate
theorem c_most_suitable : 
  ∃ c ∈ athletes, c.name = "C" ∧ isSuitable c :=
sorry

end c_most_suitable_l275_27518


namespace missing_digit_is_one_l275_27516

/-- Converts a number from base 3 to base 10 -/
def base3ToBase10 (digit1 digit2 : ℕ) : ℕ :=
  digit1 * 3 + digit2

/-- Converts a number from base 12 to base 10 -/
def base12ToBase10 (digit1 digit2 : ℕ) : ℕ :=
  digit1 * 12 + digit2

/-- The main theorem stating that the missing digit is 1 -/
theorem missing_digit_is_one :
  ∃ (triangle : ℕ), 
    triangle < 10 ∧ 
    base3ToBase10 5 triangle = base12ToBase10 triangle 4 ∧ 
    triangle = 1 := by
  sorry

#check missing_digit_is_one

end missing_digit_is_one_l275_27516


namespace square_area_from_two_points_square_area_specific_case_l275_27542

/-- The area of a square given two points on the same side -/
theorem square_area_from_two_points (x1 y1 x2 y2 : ℝ) (h : x1 = x2) :
  (y1 - y2) ^ 2 = 225 → 
  let side_length := |y1 - y2|
  (side_length ^ 2 : ℝ) = 225 := by
  sorry

/-- The specific case for the given coordinates -/
theorem square_area_specific_case : 
  let x1 : ℝ := 20
  let y1 : ℝ := 20
  let x2 : ℝ := 20
  let y2 : ℝ := 5
  (y1 - y2) ^ 2 = 225 ∧ 
  let side_length := |y1 - y2|
  (side_length ^ 2 : ℝ) = 225 := by
  sorry

end square_area_from_two_points_square_area_specific_case_l275_27542


namespace intersection_point_of_lines_l275_27570

theorem intersection_point_of_lines : ∃! p : ℚ × ℚ, 
  (3 * p.2 = -2 * p.1 + 6) ∧ (4 * p.2 = 3 * p.1 - 4) ∧ 
  p = (36/17, 10/17) := by
  sorry

end intersection_point_of_lines_l275_27570


namespace power_multiplication_l275_27537

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l275_27537


namespace fourth_root_difference_l275_27581

theorem fourth_root_difference : (81 : ℝ) ^ (1/4) - (1296 : ℝ) ^ (1/4) = -3 := by
  sorry

end fourth_root_difference_l275_27581


namespace second_group_men_count_l275_27593

/-- The work rate of one man -/
def man_rate : ℝ := sorry

/-- The work rate of one woman -/
def woman_rate : ℝ := sorry

/-- The number of men in the second group -/
def x : ℕ := sorry

theorem second_group_men_count : x = 6 :=
  sorry

end second_group_men_count_l275_27593


namespace assembly_line_average_output_l275_27517

/-- Represents the production data for an assembly line phase -/
structure ProductionPhase where
  cogs_produced : ℕ
  production_rate : ℕ

/-- Calculates the time taken for a production phase in hours -/
def time_taken (phase : ProductionPhase) : ℚ :=
  phase.cogs_produced / phase.production_rate

/-- Calculates the overall average output for two production phases -/
def overall_average_output (phase1 phase2 : ProductionPhase) : ℚ :=
  (phase1.cogs_produced + phase2.cogs_produced) / (time_taken phase1 + time_taken phase2)

/-- Theorem stating that the overall average output is 30 cogs per hour -/
theorem assembly_line_average_output :
  let phase1 : ProductionPhase := { cogs_produced := 60, production_rate := 20 }
  let phase2 : ProductionPhase := { cogs_produced := 60, production_rate := 60 }
  overall_average_output phase1 phase2 = 30 := by
  sorry

end assembly_line_average_output_l275_27517


namespace max_soap_boxes_in_carton_l275_27584

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dims : BoxDimensions) : ℕ :=
  dims.length * dims.width * dims.height

/-- Calculates the maximum number of smaller boxes that can fit in a larger box -/
def maxBoxesFit (largeBox : BoxDimensions) (smallBox : BoxDimensions) : ℕ :=
  (boxVolume largeBox) / (boxVolume smallBox)

/-- The dimensions of the carton -/
def cartonDims : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDims : BoxDimensions :=
  { length := 7, width := 6, height := 10 }

/-- Theorem stating that the maximum number of soap boxes that can fit in the carton is 150 -/
theorem max_soap_boxes_in_carton :
  maxBoxesFit cartonDims soapBoxDims = 150 := by
  sorry


end max_soap_boxes_in_carton_l275_27584


namespace divisor_problem_l275_27574

theorem divisor_problem (range_start : Nat) (range_end : Nat) (divisible_count : Nat) : 
  range_start = 10 → 
  range_end = 1000000 → 
  divisible_count = 111110 → 
  ∃ (d : Nat), d = 9 ∧ 
    (∀ n : Nat, range_start ≤ n ∧ n ≤ range_end → 
      (n % d = 0 ↔ ∃ k : Nat, k ≤ divisible_count ∧ n = range_start + (k - 1) * d)) :=
by
  sorry

end divisor_problem_l275_27574


namespace zeros_in_intervals_l275_27583

/-- A quadratic function -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem zeros_in_intervals (a b c m n p : ℝ) (h_a : a ≠ 0) (h_order : m < n ∧ n < p) :
  (∃ x y, m < x ∧ x < n ∧ n < y ∧ y < p ∧ 
    quadratic_function a b c x = 0 ∧ 
    quadratic_function a b c y = 0) ↔ 
  (quadratic_function a b c m) * (quadratic_function a b c n) < 0 ∧
  (quadratic_function a b c p) * (quadratic_function a b c n) < 0 :=
sorry

end zeros_in_intervals_l275_27583


namespace min_value_fraction_l275_27561

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a + b = 2) :
  (∃ (x : ℝ), ∀ (y : ℝ), (3*a - b) / (a^2 + 2*a*b - 3*b^2) ≥ x) ∧
  (∃ (z : ℝ), (3*z - (2-z)) / (z^2 + 2*z*(2-z) - 3*(2-z)^2) = (3 + Real.sqrt 5) / 4) :=
sorry

end min_value_fraction_l275_27561


namespace negation_of_existence_negation_of_rational_square_minus_two_l275_27510

theorem negation_of_existence (p : ℚ → Prop) : 
  (¬ ∃ x : ℚ, p x) ↔ (∀ x : ℚ, ¬ p x) := by sorry

theorem negation_of_rational_square_minus_two :
  (¬ ∃ x : ℚ, x^2 - 2 = 0) ↔ (∀ x : ℚ, x^2 - 2 ≠ 0) := by sorry

end negation_of_existence_negation_of_rational_square_minus_two_l275_27510


namespace chicken_farm_proof_l275_27592

/-- The number of chickens Michael has now -/
def initial_chickens : ℕ := 550

/-- The annual increase in the number of chickens -/
def annual_increase : ℕ := 150

/-- The number of years -/
def years : ℕ := 9

/-- The number of chickens after 9 years -/
def final_chickens : ℕ := 1900

/-- Theorem stating that the initial number of chickens plus the total increase over 9 years equals the final number of chickens -/
theorem chicken_farm_proof : 
  initial_chickens + (annual_increase * years) = final_chickens := by
  sorry


end chicken_farm_proof_l275_27592


namespace cistern_wet_surface_area_calculation_l275_27555

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width height water_depth : ℝ) : ℝ :=
  length * width + 2 * (length * water_depth) + 2 * (width * water_depth)

/-- Theorem stating that the wet surface area of the given cistern is 387.5 m² -/
theorem cistern_wet_surface_area_calculation :
  cistern_wet_surface_area 15 10 8 4.75 = 387.5 := by
  sorry

#eval cistern_wet_surface_area 15 10 8 4.75

end cistern_wet_surface_area_calculation_l275_27555


namespace linear_equation_solution_l275_27543

theorem linear_equation_solution (m : ℝ) : 
  (3 : ℝ) - m * (1 : ℝ) = 1 → m = 2 := by
  sorry

end linear_equation_solution_l275_27543


namespace cookie_scaling_l275_27594

/-- Given a recipe for cookies, calculate the required ingredients for a larger batch -/
theorem cookie_scaling (base_cookies : ℕ) (target_cookies : ℕ) 
  (base_flour : ℚ) (base_sugar : ℚ) 
  (target_flour : ℚ) (target_sugar : ℚ) : 
  base_cookies > 0 → 
  (target_flour = (target_cookies : ℚ) / base_cookies * base_flour) ∧ 
  (target_sugar = (target_cookies : ℚ) / base_cookies * base_sugar) →
  (base_cookies = 40 ∧ 
   base_flour = 3 ∧ 
   base_sugar = 1 ∧ 
   target_cookies = 200) →
  (target_flour = 15 ∧ target_sugar = 5) := by
  sorry

end cookie_scaling_l275_27594


namespace negation_of_forall_positive_x_squared_minus_x_geq_zero_l275_27540

theorem negation_of_forall_positive_x_squared_minus_x_geq_zero :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x < 0) :=
by sorry

end negation_of_forall_positive_x_squared_minus_x_geq_zero_l275_27540


namespace rect_to_cylindrical_conversion_l275_27575

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion 
  (x y z : ℝ) 
  (h_x : x = 3) 
  (h_y : y = -3 * Real.sqrt 3) 
  (h_z : z = 4) :
  ∃ (r θ : ℝ), 
    r > 0 ∧ 
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r = 6 ∧ 
    θ = 5 * Real.pi / 3 ∧
    r * (Real.cos θ) = x ∧
    r * (Real.sin θ) = y ∧
    z = 4 :=
by sorry

end rect_to_cylindrical_conversion_l275_27575


namespace cow_price_problem_l275_27511

/-- Given the total cost of cows and goats, the number of cows and goats, and the average price of a goat,
    calculate the average price of a cow. -/
def average_cow_price (total_cost : ℕ) (num_cows num_goats : ℕ) (avg_goat_price : ℕ) : ℕ :=
  (total_cost - num_goats * avg_goat_price) / num_cows

/-- Theorem: Given 2 cows and 10 goats with a total cost of 1500 rupees, 
    and an average price of 70 rupees per goat, the average price of a cow is 400 rupees. -/
theorem cow_price_problem : average_cow_price 1500 2 10 70 = 400 := by
  sorry

end cow_price_problem_l275_27511


namespace min_cubes_for_3x9x5_hollow_block_l275_27525

/-- The minimum number of cubes needed to create a hollow block -/
def min_cubes_for_hollow_block (length width depth : ℕ) : ℕ :=
  length * width * depth - (length - 2) * (width - 2) * (depth - 2)

/-- Theorem stating that the minimum number of cubes for a 3x9x5 hollow block is 114 -/
theorem min_cubes_for_3x9x5_hollow_block :
  min_cubes_for_hollow_block 3 9 5 = 114 := by
  sorry

end min_cubes_for_3x9x5_hollow_block_l275_27525


namespace beth_coin_sale_l275_27563

theorem beth_coin_sale (initial_coins : ℕ) (gift_coins : ℕ) : 
  initial_coins = 125 → gift_coins = 35 → 
  (initial_coins + gift_coins) / 2 = 80 := by
  sorry

end beth_coin_sale_l275_27563


namespace three_dozens_equals_42_l275_27598

/-- Calculates the total number of flowers a customer receives when buying dozens of flowers with a free flower promotion. -/
def totalFlowers (dozens : ℕ) : ℕ :=
  let boughtFlowers := dozens * 12
  let freeFlowers := dozens * 2
  boughtFlowers + freeFlowers

/-- Theorem stating that buying 3 dozens of flowers results in 42 total flowers. -/
theorem three_dozens_equals_42 :
  totalFlowers 3 = 42 := by
  sorry

end three_dozens_equals_42_l275_27598


namespace triangle_angle_proof_l275_27553

theorem triangle_angle_proof (A B C : Real) (a b c : Real) :
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  ((b * Real.cos C) / Real.cos B + c = (2 * Real.sqrt 3 / 3) * a) →
  B = π / 6 := by
sorry


end triangle_angle_proof_l275_27553


namespace kim_average_increase_l275_27534

/-- Given Kim's exam scores, prove that her average increases by 1 after the fourth exam. -/
theorem kim_average_increase (score1 score2 score3 score4 : ℕ) 
  (h1 : score1 = 87)
  (h2 : score2 = 83)
  (h3 : score3 = 88)
  (h4 : score4 = 90) :
  (score1 + score2 + score3 + score4) / 4 - (score1 + score2 + score3) / 3 = 1 := by
  sorry

#eval (87 + 83 + 88 + 90) / 4 - (87 + 83 + 88) / 3

end kim_average_increase_l275_27534


namespace sin_cos_difference_equals_neg_half_l275_27585

theorem sin_cos_difference_equals_neg_half : 
  Real.sin (119 * π / 180) * Real.cos (91 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 := by
  sorry

end sin_cos_difference_equals_neg_half_l275_27585


namespace football_club_penalty_kicks_l275_27507

/-- Calculates the total number of penalty kicks in a football club's shootout contest. -/
def total_penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  (total_players - goalies) * goalies

/-- Theorem stating that for a football club with 25 players including 4 goalies, 
    where each player takes a shot against each goalie, the total number of penalty kicks is 96. -/
theorem football_club_penalty_kicks :
  total_penalty_kicks 25 4 = 96 := by
  sorry

#eval total_penalty_kicks 25 4

end football_club_penalty_kicks_l275_27507


namespace sqrt_difference_equals_seven_sqrt_two_over_six_l275_27523

theorem sqrt_difference_equals_seven_sqrt_two_over_six :
  Real.sqrt (9 / 2) - Real.sqrt (2 / 9) = 7 * Real.sqrt 2 / 6 := by
  sorry

end sqrt_difference_equals_seven_sqrt_two_over_six_l275_27523


namespace intersection_line_l275_27527

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line : 
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → line x y :=
by sorry

end intersection_line_l275_27527


namespace rug_coverage_area_l275_27588

/-- Given three rugs with specified overlapping areas, calculate the total floor area covered -/
theorem rug_coverage_area (total_rug_area : ℝ) (two_layer_overlap : ℝ) (three_layer_overlap : ℝ) 
  (h1 : total_rug_area = 204)
  (h2 : two_layer_overlap = 24)
  (h3 : three_layer_overlap = 20) :
  total_rug_area - two_layer_overlap - 2 * three_layer_overlap = 140 := by
  sorry

end rug_coverage_area_l275_27588


namespace prism_faces_l275_27529

theorem prism_faces (E V : ℕ) (h : E + V = 30) : ∃ (F : ℕ), F = 8 ∧ F + V = E + 2 := by
  sorry

end prism_faces_l275_27529


namespace fourth_power_sum_l275_27596

theorem fourth_power_sum (a b c : ℝ) 
  (sum_1 : a + b + c = 1)
  (sum_2 : a^2 + b^2 + c^2 = 2)
  (sum_3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 25/6 := by
sorry

end fourth_power_sum_l275_27596


namespace not_product_of_consecutive_numbers_l275_27564

theorem not_product_of_consecutive_numbers (n : ℕ) :
  ¬ ∃ k : ℕ, 2 * (6^n + 1) = k * (k + 1) := by
sorry

end not_product_of_consecutive_numbers_l275_27564


namespace physics_class_size_l275_27547

theorem physics_class_size :
  ∀ (boys_biology girls_biology students_physics : ℕ),
    girls_biology = 3 * boys_biology →
    boys_biology = 25 →
    students_physics = 2 * (boys_biology + girls_biology) →
    students_physics = 200 := by
  sorry

end physics_class_size_l275_27547


namespace angle_inequality_l275_27513

theorem angle_inequality (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 2 → x^2 * Real.cos θ - 2*x*(1 - x) + (2 - x)^2 * Real.sin θ > 0) →
  Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12 := by
  sorry

end angle_inequality_l275_27513


namespace recipe_total_l275_27569

theorem recipe_total (eggs : ℕ) (flour : ℕ) : 
  eggs = 60 → flour = eggs / 2 → eggs + flour = 90 := by
  sorry

end recipe_total_l275_27569


namespace solution_exists_l275_27515

theorem solution_exists : ∃ a : ℝ, (-6) * (a^2) = 3 * (4*a + 2) ∧ a = -1 := by
  sorry

end solution_exists_l275_27515


namespace vector_properties_l275_27562

def a : ℝ × ℝ := (1, 2)
def b (t : ℝ) : ℝ × ℝ := (-4, t)

theorem vector_properties :
  (∀ t : ℝ, (∃ k : ℝ, a = k • b t) → t = -8) ∧
  (∃ t_min : ℝ, ∀ t : ℝ, ‖a - b t‖ ≥ ‖a - b t_min‖ ∧ ‖a - b t_min‖ = 5) ∧
  (∀ t : ℝ, ‖a + b t‖ = ‖a - b t‖ → t = 2) ∧
  (∀ t : ℝ, (a • b t < 0) → t < 2) :=
by sorry

end vector_properties_l275_27562


namespace last_score_is_70_l275_27501

def scores : List ℤ := [65, 70, 85, 90]

def is_divisible (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def valid_sequence (seq : List ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → n ≤ seq.length → is_divisible (seq.take n).sum n

theorem last_score_is_70 :
  ∃ (seq : List ℤ), seq.toFinset = scores.toFinset ∧
                    valid_sequence seq ∧
                    seq.getLast? = some 70 :=
sorry

end last_score_is_70_l275_27501


namespace inequality_proof_l275_27514

theorem inequality_proof (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  12 / (w + x + y + z) ≤ 1/(w + x) + 1/(w + y) + 1/(w + z) + 1/(x + y) + 1/(x + z) + 1/(y + z) ∧
  1/(w + x) + 1/(w + y) + 1/(w + z) + 1/(x + y) + 1/(x + z) + 1/(y + z) ≤ 3/4 * (1/w + 1/x + 1/y + 1/z) :=
by sorry


end inequality_proof_l275_27514


namespace negative_result_operation_only_A_is_negative_l275_27571

theorem negative_result_operation : ℤ → Prop :=
  fun x => x < 0

theorem only_A_is_negative :
  negative_result_operation ((-1) + (-3)) ∧
  ¬negative_result_operation (6 - (-3)) ∧
  ¬negative_result_operation ((-3) * (-2)) ∧
  ¬negative_result_operation (0 / (-7)) :=
by
  sorry

end negative_result_operation_only_A_is_negative_l275_27571


namespace two_digit_number_difference_l275_27532

/-- 
For a two-digit number where:
- The number is 26
- The product of the number and the sum of its digits is 208
Prove that the difference between the unit's digit and the 10's digit is 4.
-/
theorem two_digit_number_difference (n : ℕ) (h1 : n = 26) 
  (h2 : n * (n / 10 + n % 10) = 208) : n % 10 - n / 10 = 4 := by
  sorry

end two_digit_number_difference_l275_27532


namespace ticket_sales_total_l275_27591

/-- Calculates the total amount collected from ticket sales given the following conditions:
  * Adult ticket cost is $12
  * Child ticket cost is $4
  * Total number of tickets sold is 130
  * Number of adult tickets sold is 40
-/
theorem ticket_sales_total (adult_cost child_cost total_tickets adult_tickets : ℕ) : 
  adult_cost = 12 →
  child_cost = 4 →
  total_tickets = 130 →
  adult_tickets = 40 →
  adult_cost * adult_tickets + child_cost * (total_tickets - adult_tickets) = 840 :=
by
  sorry

#check ticket_sales_total

end ticket_sales_total_l275_27591


namespace five_T_three_equals_38_l275_27530

-- Define the operation T
def T (a b : ℝ) : ℝ := 4 * a + 6 * b

-- Theorem to prove
theorem five_T_three_equals_38 : T 5 3 = 38 := by
  sorry

end five_T_three_equals_38_l275_27530


namespace problem1_l275_27533

theorem problem1 (x y : ℝ) : (-3 * x * y)^2 * (4 * x^2) = 36 * x^4 * y^2 := by
  sorry

end problem1_l275_27533


namespace tomorrow_is_saturday_l275_27559

-- Define the days of the week
inductive Day : Type
  | Sunday : Day
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day

def next_day (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

def days_after (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => next_day (days_after d m)

theorem tomorrow_is_saturday 
  (day_before_yesterday : Day)
  (h : days_after day_before_yesterday 5 = Day.Monday) :
  days_after day_before_yesterday 3 = Day.Saturday :=
by sorry

end tomorrow_is_saturday_l275_27559


namespace shuffleboard_games_total_l275_27580

/-- Proves that the total number of games played is 32 given the conditions of the shuffleboard game. -/
theorem shuffleboard_games_total (jerry_wins dave_wins ken_wins : ℕ) 
  (h1 : ken_wins = dave_wins + 5)
  (h2 : dave_wins = jerry_wins + 3)
  (h3 : jerry_wins = 7) : 
  jerry_wins + dave_wins + ken_wins = 32 := by
  sorry

end shuffleboard_games_total_l275_27580
