import Mathlib

namespace x_fourth_plus_reciprocal_l3876_387688

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 + 1/x^4 = 727 := by
  sorry

end x_fourth_plus_reciprocal_l3876_387688


namespace same_grade_percentage_l3876_387684

-- Define the total number of students
def total_students : ℕ := 50

-- Define the number of students who got the same grade on both tests
def same_grade_A : ℕ := 3
def same_grade_B : ℕ := 6
def same_grade_C : ℕ := 7
def same_grade_D : ℕ := 2

-- Define the total number of students who got the same grade on both tests
def total_same_grade : ℕ := same_grade_A + same_grade_B + same_grade_C + same_grade_D

-- Define the percentage of students who got the same grade on both tests
def percentage_same_grade : ℚ := (total_same_grade : ℚ) / (total_students : ℚ) * 100

-- Theorem to prove
theorem same_grade_percentage :
  percentage_same_grade = 36 := by
  sorry

end same_grade_percentage_l3876_387684


namespace sum_reciprocals_bound_l3876_387680

theorem sum_reciprocals_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1/a + 1/b ≥ 4 ∧ ∀ M : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1/a + 1/b > M :=
by sorry

end sum_reciprocals_bound_l3876_387680


namespace equalizeTable_l3876_387636

-- Define the table as a matrix
def Table (n : ℕ) := Matrix (Fin n) (Fin n) ℕ

-- Initial configuration of the table
def initialTable (n : ℕ) : Table n :=
  Matrix.diagonal (λ _ => 1)

-- Define a rook path as a list of positions
def RookPath (n : ℕ) := List (Fin n × Fin n)

-- Predicate to check if a path is valid (closed and non-self-intersecting)
def isValidPath (n : ℕ) (path : RookPath n) : Prop := sorry

-- Function to apply a rook transformation
def applyRookTransformation (t : Table n) (path : RookPath n) : Table n := sorry

-- Predicate to check if all numbers in the table are equal
def allEqual (t : Table n) : Prop := sorry

-- The main theorem
theorem equalizeTable (n : ℕ) :
  (∃ (transformations : List (RookPath n)), 
    allEqual (transformations.foldl applyRookTransformation (initialTable n))) ↔ 
  Odd n := by sorry

end equalizeTable_l3876_387636


namespace number_puzzle_l3876_387657

theorem number_puzzle (x : ℝ) : 2 * x = 18 → x - 4 = 5 := by
  sorry

end number_puzzle_l3876_387657


namespace toucan_problem_l3876_387687

theorem toucan_problem (initial_toucans : ℕ) : 
  (initial_toucans + 1 = 3) → initial_toucans = 2 := by
sorry

end toucan_problem_l3876_387687


namespace range_of_a_l3876_387655

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing on [1,5]
def IsIncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 1 5 → y ∈ Set.Icc 1 5 → x < y → f x < f y

-- Define the theorem
theorem range_of_a (h1 : IsIncreasingOn f) 
  (h2 : ∀ a, f (a + 1) < f (2 * a - 1)) :
  ∃ a, a ∈ Set.Ioo 2 3 ∧ 
    (∀ x, x ∈ Set.Ioo 2 3 → 
      (f (x + 1) < f (2 * x - 1) ∧ 
       x + 1 ∈ Set.Icc 1 5 ∧ 
       2 * x - 1 ∈ Set.Icc 1 5)) :=
by sorry


end range_of_a_l3876_387655


namespace inequality_proofs_l3876_387666

theorem inequality_proofs (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end inequality_proofs_l3876_387666


namespace m_range_when_p_false_l3876_387628

theorem m_range_when_p_false :
  (¬∀ x : ℝ, ∃ m : ℝ, 4*x - 2*x + 1 + m = 0) →
  {m : ℝ | ∃ x : ℝ, 4*x - 2*x + 1 + m ≠ 0} = Set.Iio 1 := by
  sorry

end m_range_when_p_false_l3876_387628


namespace infinite_linear_combinations_l3876_387691

/-- An infinite sequence of positive integers with strictly increasing terms. -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, 0 < a k ∧ a k < a (k + 1)

/-- The property that infinitely many elements of the sequence can be written as a linear
    combination of two earlier terms with positive integer coefficients. -/
def InfinitelyManyLinearCombinations (a : ℕ → ℕ) : Prop :=
  ∀ N, ∃ m p q x y, N < m ∧ m > p ∧ p > q ∧ 0 < x ∧ 0 < y ∧ a m = x * a p + y * a q

/-- The main theorem stating that any strictly increasing sequence of positive integers
    has infinitely many elements that can be written as a linear combination of two earlier terms. -/
theorem infinite_linear_combinations
  (a : ℕ → ℕ) (h : StrictlyIncreasingSequence a) :
  InfinitelyManyLinearCombinations a := by
  sorry

end infinite_linear_combinations_l3876_387691


namespace equation_solutions_l3876_387694

-- Define the equation
def equation (x : ℝ) : Prop := (x ^ (1/4 : ℝ)) = 16 / (9 - (x ^ (1/4 : ℝ)))

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = 4096) :=
sorry

end equation_solutions_l3876_387694


namespace man_half_father_age_l3876_387683

theorem man_half_father_age (father_age : ℝ) (man_age : ℝ) (years_later : ℝ) : 
  father_age = 30.000000000000007 →
  man_age = (2/5) * father_age →
  man_age + years_later = (1/2) * (father_age + years_later) →
  years_later = 6 := by
sorry

end man_half_father_age_l3876_387683


namespace total_distance_driven_l3876_387695

def miles_per_gallon : ℝ := 25
def tank_capacity : ℝ := 18
def initial_gas : ℝ := 12
def first_leg_distance : ℝ := 250
def gas_purchased : ℝ := 10
def final_gas : ℝ := 3

theorem total_distance_driven : ℝ := by
  -- The total distance driven is 475 miles
  sorry

#check total_distance_driven

end total_distance_driven_l3876_387695


namespace polynomial_coefficient_b_l3876_387639

theorem polynomial_coefficient_b (a b c : ℚ) :
  (∀ x : ℚ, (3 * x^2 - 2 * x + 5/4) * (a * x^2 + b * x + c) = 
    9 * x^4 - 5 * x^3 + 31/4 * x^2 - 10/3 * x + 5/12) →
  b = 1/3 := by
sorry

end polynomial_coefficient_b_l3876_387639


namespace zeljko_distance_l3876_387692

/-- Calculates the total distance travelled given two segments of a journey -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Proves that Zeljko's total distance travelled is 20 km -/
theorem zeljko_distance : 
  let speed1 : ℝ := 30  -- km/h
  let time1  : ℝ := 20 / 60  -- 20 minutes in hours
  let speed2 : ℝ := 20  -- km/h
  let time2  : ℝ := 30 / 60  -- 30 minutes in hours
  total_distance speed1 time1 speed2 time2 = 20 := by
  sorry


end zeljko_distance_l3876_387692


namespace log_difference_cube_l3876_387617

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the theorem
theorem log_difference_cube (x y a : ℝ) (h : lg x - lg y = a) :
  lg ((x/2)^3) - lg ((y/2)^3) = 3*a := by
  sorry

end log_difference_cube_l3876_387617


namespace max_candies_eaten_l3876_387668

theorem max_candies_eaten (n : ℕ) (h : n = 32) : 
  (n * (n - 1)) / 2 = 496 := by
  sorry

end max_candies_eaten_l3876_387668


namespace blue_sky_project_expo_course_l3876_387634

theorem blue_sky_project_expo_course (n m : ℕ) (hn : n = 6) (hm : m = 6) :
  (Nat.choose n 2) * (m - 1) ^ (n - 2) = 
    (Nat.choose 6 2) * 5^4 :=
sorry

end blue_sky_project_expo_course_l3876_387634


namespace all_nonnegative_possible_l3876_387649

theorem all_nonnegative_possible (nums : List ℝ) (h1 : nums.length = 10) 
  (h2 : nums.sum / nums.length = 0) : 
  ∃ (nonneg_nums : List ℝ), nonneg_nums.length = 10 ∧ 
    nonneg_nums.sum / nonneg_nums.length = 0 ∧
    ∀ x ∈ nonneg_nums, x ≥ 0 :=
by
  sorry

end all_nonnegative_possible_l3876_387649


namespace alternating_draw_probability_l3876_387679

/-- The number of white balls in the box -/
def white_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 3

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of ways to choose positions for black balls -/
def total_arrangements : ℕ := Nat.choose total_balls black_balls

/-- The number of successful alternating color arrangements -/
def successful_arrangements : ℕ := Nat.choose (total_balls - 2) black_balls

/-- The probability of drawing an alternating color sequence -/
def alternating_probability : ℚ := successful_arrangements / total_arrangements

theorem alternating_draw_probability :
  alternating_probability = 5 / 14 := by sorry

end alternating_draw_probability_l3876_387679


namespace larger_number_l3876_387658

theorem larger_number (a b : ℝ) (sum : a + b = 40) (diff : a - b = 10) : max a b = 25 := by
  sorry

end larger_number_l3876_387658


namespace composite_sum_of_product_equal_l3876_387606

theorem composite_sum_of_product_equal (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ a^1984 + b^1984 + c^1984 + d^1984 = m * n :=
sorry

end composite_sum_of_product_equal_l3876_387606


namespace equation_solution_l3876_387627

theorem equation_solution :
  ∃! x : ℝ, x - 5 ≥ 0 ∧
  (7 / (Real.sqrt (x - 5) - 10) + 2 / (Real.sqrt (x - 5) - 3) +
   8 / (Real.sqrt (x - 5) + 3) + 13 / (Real.sqrt (x - 5) + 10) = 0) ∧
  x = 1486 / 225 := by
  sorry

end equation_solution_l3876_387627


namespace angle_B_60_iff_arithmetic_progression_l3876_387663

theorem angle_B_60_iff_arithmetic_progression (A B C : ℝ) : 
  (A + B + C = 180) →  -- Sum of angles in a triangle is 180°
  (B = 60 ↔ ∃ d : ℝ, A = B - d ∧ C = B + d) :=
sorry

end angle_B_60_iff_arithmetic_progression_l3876_387663


namespace sqrt_x_minus_2_defined_l3876_387612

theorem sqrt_x_minus_2_defined (x : ℝ) : 
  ∃ y : ℝ, y ^ 2 = x - 2 ↔ x ≥ 2 := by sorry

end sqrt_x_minus_2_defined_l3876_387612


namespace library_book_count_l3876_387653

/-- Given a library with shelves that each hold a fixed number of books,
    calculate the total number of books in the library. -/
def total_books (num_shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  num_shelves * books_per_shelf

/-- Theorem stating that a library with 1780 shelves, each holding 8 books,
    contains 14240 books in total. -/
theorem library_book_count : total_books 1780 8 = 14240 := by
  sorry

end library_book_count_l3876_387653


namespace trevor_coin_conversion_l3876_387642

/-- Represents the types of coins in the problem -/
inductive Coin
  | Quarter
  | Dime
  | Nickel
  | Penny

/-- Calculates the value of a coin in cents -/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- Represents the coin count in Trevor's bank -/
structure CoinCount where
  total : ℕ
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (cc : CoinCount) : ℕ :=
  cc.quarters * coinValue Coin.Quarter +
  cc.dimes * coinValue Coin.Dime +
  cc.nickels * coinValue Coin.Nickel +
  cc.pennies * coinValue Coin.Penny

/-- Converts total value to $5 bills and $1 coins -/
def convertToBillsAndCoins (value : ℕ) : (ℕ × ℕ) :=
  (value / 500, (value % 500) / 100)

theorem trevor_coin_conversion :
  let cc : CoinCount := {
    total := 153,
    quarters := 45,
    dimes := 34,
    nickels := 19,
    pennies := 153 - 45 - 34 - 19
  }
  let (fiveBills, oneDollars) := convertToBillsAndCoins (totalValue cc)
  fiveBills - oneDollars = 2 := by sorry

end trevor_coin_conversion_l3876_387642


namespace repeating_decimal_equals_fraction_l3876_387624

/-- Represents the repeating decimal 4.565656... -/
def repeating_decimal : ℚ := 4 + 56 / 99

/-- The fraction representation of 4.565656... -/
def fraction : ℚ := 452 / 99

theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end repeating_decimal_equals_fraction_l3876_387624


namespace school_basketballs_l3876_387696

/-- The number of classes that received basketballs -/
def num_classes : ℕ := 7

/-- The number of basketballs each class received -/
def basketballs_per_class : ℕ := 7

/-- The total number of basketballs bought by the school -/
def total_basketballs : ℕ := num_classes * basketballs_per_class

theorem school_basketballs : total_basketballs = 49 := by
  sorry

end school_basketballs_l3876_387696


namespace tax_rate_calculation_l3876_387605

/-- Given a purchase in country B with a tax-free threshold, calculate the tax rate -/
theorem tax_rate_calculation (total_value tax_free_threshold tax_paid : ℝ) : 
  total_value = 1720 →
  tax_free_threshold = 600 →
  tax_paid = 123.2 →
  (tax_paid / (total_value - tax_free_threshold)) * 100 = 11 := by
sorry

end tax_rate_calculation_l3876_387605


namespace f_3_range_l3876_387613

-- Define the function f(x) = a x^2 - c
def f (a c x : ℝ) : ℝ := a * x^2 - c

-- State the theorem
theorem f_3_range (a c : ℝ) :
  (∀ x : ℝ, f a c x = a * x^2 - c) →
  (-4 ≤ f a c 1 ∧ f a c 1 ≤ -1) →
  (-1 ≤ f a c 2 ∧ f a c 2 ≤ 5) →
  (-1 ≤ f a c 3 ∧ f a c 3 ≤ 20) :=
by sorry

end f_3_range_l3876_387613


namespace faye_earnings_l3876_387637

/-- Calculates the total amount earned from selling necklaces -/
def total_earned (bead_count gemstone_count pearl_count crystal_count : ℕ) 
                 (bead_price gemstone_price pearl_price crystal_price : ℕ) : ℕ :=
  bead_count * bead_price + 
  gemstone_count * gemstone_price + 
  pearl_count * pearl_price + 
  crystal_count * crystal_price

/-- Theorem: The total amount Faye earned is $190 -/
theorem faye_earnings : 
  total_earned 3 7 2 5 7 10 12 15 = 190 := by
  sorry

end faye_earnings_l3876_387637


namespace complex_power_2007_l3876_387699

theorem complex_power_2007 : (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I)) ^ 2007 = -Complex.I := by sorry

end complex_power_2007_l3876_387699


namespace reflection_matrix_correct_l3876_387652

/-- Reflection matrix over the line y = x -/
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1; 1, 0]

/-- A point in 2D space -/
def Point := Fin 2 → ℝ

/-- Reflect a point over the line y = x -/
def reflect (p : Point) : Point :=
  λ i => p (if i = 0 then 1 else 0)

theorem reflection_matrix_correct :
  ∀ (p : Point), reflection_matrix.mulVec p = reflect p :=
by sorry

end reflection_matrix_correct_l3876_387652


namespace fractional_equation_solution_l3876_387620

theorem fractional_equation_solution : 
  ∃ x : ℝ, (3 - x) / (x - 4) + 1 / (4 - x) = 1 ∧ x = 3 :=
by
  sorry

end fractional_equation_solution_l3876_387620


namespace joan_pencils_l3876_387635

theorem joan_pencils (initial_pencils final_pencils : ℕ) 
  (h1 : initial_pencils = 33)
  (h2 : final_pencils = 60) :
  final_pencils - initial_pencils = 27 := by
  sorry

end joan_pencils_l3876_387635


namespace correct_answers_for_given_score_l3876_387609

/-- Represents a test result -/
structure TestResult where
  totalQuestions : ℕ
  correctAnswers : ℕ
  score : ℤ

/-- Calculates the score based on correct and incorrect answers -/
def calculateScore (correct incorrect : ℕ) : ℤ :=
  (correct : ℤ) - 2 * (incorrect : ℤ)

theorem correct_answers_for_given_score 
  (result : TestResult) 
  (h1 : result.totalQuestions = 100)
  (h2 : result.score = calculateScore result.correctAnswers (result.totalQuestions - result.correctAnswers))
  (h3 : result.score = 76) :
  result.correctAnswers = 92 := by
  sorry


end correct_answers_for_given_score_l3876_387609


namespace janabel_widget_sales_l3876_387690

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem janabel_widget_sales : arithmetic_sequence_sum 2 3 15 = 345 := by
  sorry

end janabel_widget_sales_l3876_387690


namespace problem_solution_l3876_387615

theorem problem_solution (a b : ℝ) (h : Real.sqrt (a - 3) + abs (4 - b) = 0) :
  (a - b) ^ 2023 = -1 ∧ ∀ x n : ℝ, x > 0 → Real.sqrt x = a + n → Real.sqrt x = b - 2*n → x = 100 := by
  sorry

end problem_solution_l3876_387615


namespace necessary_but_not_sufficient_l3876_387647

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (((a > 2 ∧ b > 2) → (a + b > 4)) ∧ 
   (∃ x y : ℝ, x + y > 4 ∧ ¬(x > 2 ∧ y > 2))) :=
by sorry

end necessary_but_not_sufficient_l3876_387647


namespace range_of_f_l3876_387614

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Define the domain
def domain : Set ℝ := Set.Icc 1 5

-- Theorem statement
theorem range_of_f :
  Set.range (fun x => f x) ∩ (Set.image f domain) = Set.Ico 2 11 := by sorry

end range_of_f_l3876_387614


namespace ethanol_mixture_optimization_l3876_387689

theorem ethanol_mixture_optimization (initial_volume : ℝ) (initial_ethanol_percentage : ℝ)
  (added_ethanol : ℝ) (final_ethanol_percentage : ℝ) :
  initial_volume = 45 →
  initial_ethanol_percentage = 0.05 →
  added_ethanol = 2.5 →
  final_ethanol_percentage = 0.1 →
  (initial_volume * initial_ethanol_percentage + added_ethanol) /
    (initial_volume + added_ethanol) = final_ethanol_percentage :=
by sorry

end ethanol_mixture_optimization_l3876_387689


namespace intersection_point_distance_to_line_l3876_387648

-- Define the lines
def l1 (x y : ℝ) : Prop := x - y + 2 = 0
def l2 (x y : ℝ) : Prop := x - 2*y + 3 = 0
def l (x y : ℝ) : Prop := 3*x + 4*y - 10 = 0

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Theorem for the intersection point
theorem intersection_point : ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧ x = -1 ∧ y = 1 := by sorry

-- Theorem for the distance
theorem distance_to_line : 
  let d := |3 * P.1 + 4 * P.2 - 10| / Real.sqrt (3^2 + 4^2)
  d = 3 := by sorry

end intersection_point_distance_to_line_l3876_387648


namespace product_bounds_l3876_387698

theorem product_bounds (x₁ x₂ x₃ : ℝ) 
  (h_nonneg₁ : x₁ ≥ 0) (h_nonneg₂ : x₂ ≥ 0) (h_nonneg₃ : x₃ ≥ 0)
  (h_sum : x₁ + x₂ + x₃ = 1) :
  1 ≤ (x₁ + 3*x₂ + 5*x₃) * (x₁ + x₂/3 + x₃/5) ∧
  (x₁ + 3*x₂ + 5*x₃) * (x₁ + x₂/3 + x₃/5) ≤ 9/5 :=
by sorry

end product_bounds_l3876_387698


namespace intersection_when_a_is_4_union_condition_l3876_387686

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x > 5 ∨ x < -1}

-- Theorem 1: When a = 4, A ∩ B = {x | 6 < x ≤ 7}
theorem intersection_when_a_is_4 :
  A 4 ∩ B = {x | 6 < x ∧ x ≤ 7} := by sorry

-- Theorem 2: A ∪ B = B if and only if a < -4 or a > 5
theorem union_condition (a : ℝ) :
  A a ∪ B = B ↔ a < -4 ∨ a > 5 := by sorry

end intersection_when_a_is_4_union_condition_l3876_387686


namespace geometric_sequence_problem_l3876_387676

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, 280 * r = a ∧ a * r = 35 / 8) : a = 35 := by
  sorry

end geometric_sequence_problem_l3876_387676


namespace min_sum_squares_l3876_387665

theorem min_sum_squares (x y z : ℝ) (h : x + y + z = 1) : x^2 + y^2 + z^2 ≥ 1/3 := by
  sorry

end min_sum_squares_l3876_387665


namespace three_circles_equal_angle_points_l3876_387633

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Predicate to check if two circles do not intersect and neither is contained within the other -/
def are_separate (c1 c2 : Circle) : Prop := sorry

/-- The locus of points from which two circles are seen at the same angle -/
def equal_angle_locus (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

/-- The angle at which a circle is seen from a point -/
def viewing_angle (c : Circle) (p : ℝ × ℝ) : ℝ := sorry

theorem three_circles_equal_angle_points 
  (k1 k2 k3 : Circle)
  (h12 : are_separate k1 k2)
  (h23 : are_separate k2 k3)
  (h13 : are_separate k1 k3) :
  ∃ p : ℝ × ℝ, 
    viewing_angle k1 p = viewing_angle k2 p ∧ 
    viewing_angle k2 p = viewing_angle k3 p ∧
    p ∈ (equal_angle_locus k1 k2) ∩ (equal_angle_locus k2 k3) := by
  sorry

end three_circles_equal_angle_points_l3876_387633


namespace y_quadratic_iff_m_eq_2_y_linear_iff_m_special_l3876_387678

noncomputable def y (m : ℝ) (x : ℝ) : ℝ := (m + 3) * x^(m^2 + m - 4) + (m + 2) * x + 3

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

theorem y_quadratic_iff_m_eq_2 (m : ℝ) :
  is_quadratic (y m) ↔ m = 2 :=
sorry

theorem y_linear_iff_m_special (m : ℝ) :
  is_linear (y m) ↔ 
    m = -3 ∨ 
    m = (-1 + Real.sqrt 17) / 2 ∨ 
    m = (-1 - Real.sqrt 17) / 2 ∨
    m = (-1 + Real.sqrt 21) / 2 ∨
    m = (-1 - Real.sqrt 21) / 2 :=
sorry

end y_quadratic_iff_m_eq_2_y_linear_iff_m_special_l3876_387678


namespace count_nonincreasing_7digit_integers_l3876_387629

/-- The number of 7-digit positive integers with nonincreasing digits -/
def nonincreasing_7digit_integers : ℕ :=
  Nat.choose 16 7 - 1

/-- Proposition: The number of 7-digit positive integers with nonincreasing digits is 11439 -/
theorem count_nonincreasing_7digit_integers :
  nonincreasing_7digit_integers = 11439 := by
  sorry

end count_nonincreasing_7digit_integers_l3876_387629


namespace prob_jack_queen_king_ace_value_l3876_387602

-- Define the total number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of cards for each face value
def cards_per_face : ℕ := 4

-- Define the probability of drawing the specific sequence
def prob_jack_queen_king_ace : ℚ :=
  (cards_per_face : ℚ) / total_cards *
  (cards_per_face : ℚ) / (total_cards - 1) *
  (cards_per_face : ℚ) / (total_cards - 2) *
  (cards_per_face : ℚ) / (total_cards - 3)

-- Theorem statement
theorem prob_jack_queen_king_ace_value :
  prob_jack_queen_king_ace = 16 / 4048375 := by
  sorry

end prob_jack_queen_king_ace_value_l3876_387602


namespace magnitude_of_b_l3876_387671

/-- Given two planar vectors a and b satisfying the specified conditions, 
    the magnitude of b is 2. -/
theorem magnitude_of_b (a b : ℝ × ℝ) : 
  (‖a‖ = 1) →
  (‖a - 2 • b‖ = Real.sqrt 21) →
  (a.1 * b.1 + a.2 * b.2 = ‖a‖ * ‖b‖ * (-1/2)) →
  ‖b‖ = 2 := by sorry

end magnitude_of_b_l3876_387671


namespace penny_nickel_dime_heads_prob_l3876_387661

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (half_dollar : CoinOutcome)

/-- The probability of getting heads on the penny, nickel, and dime when flipping five coins -/
def prob_penny_nickel_dime_heads : ℚ :=
  1 / 8

/-- Theorem stating that the probability of getting heads on the penny, nickel, and dime
    when flipping five coins simultaneously is 1/8 -/
theorem penny_nickel_dime_heads_prob :
  prob_penny_nickel_dime_heads = 1 / 8 :=
by sorry

end penny_nickel_dime_heads_prob_l3876_387661


namespace parallel_lines_minimum_value_l3876_387625

theorem parallel_lines_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 2 * a + 3 * b ≥ 25 := by
  sorry

end parallel_lines_minimum_value_l3876_387625


namespace h_of_2_equals_2_l3876_387673

-- Define the function h
noncomputable def h : ℝ → ℝ := fun x => 
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) - 1) / (x^31 - 1)

-- Theorem statement
theorem h_of_2_equals_2 : h 2 = 2 := by
  sorry

end h_of_2_equals_2_l3876_387673


namespace temperature_difference_l3876_387616

/-- The temperature difference problem -/
theorem temperature_difference
  (morning_temp : ℝ)
  (noon_rise : ℝ)
  (night_drop : ℝ)
  (h_morning : morning_temp = 7)
  (h_noon_rise : noon_rise = 9)
  (h_night_drop : night_drop = 13)
  (h_highest : morning_temp + noon_rise = max morning_temp (morning_temp + noon_rise))
  (h_lowest : morning_temp + noon_rise - night_drop = min (morning_temp + noon_rise) (morning_temp + noon_rise - night_drop)) :
  (morning_temp + noon_rise) - (morning_temp + noon_rise - night_drop) = 13 := by
  sorry

end temperature_difference_l3876_387616


namespace certain_number_exists_l3876_387660

theorem certain_number_exists : ∃ N : ℝ, (7/13) * N = (5/16) * N + 500 := by
  sorry

end certain_number_exists_l3876_387660


namespace basketball_substitutions_l3876_387611

/-- The number of possible substitution methods in a basketball game with specific rules -/
def substitution_methods (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitute_players := total_players - starting_players
  1 + -- No substitutions
  (starting_players * substitute_players) + -- One substitution
  (starting_players * (starting_players - 1) * substitute_players * (substitute_players - 1) / 2) + -- Two substitutions
  (starting_players * (starting_players - 1) * (starting_players - 2) * substitute_players * (substitute_players - 1) * (substitute_players - 2) / 6) -- Three substitutions

/-- The main theorem stating the number of substitution methods and its remainder when divided by 1000 -/
theorem basketball_substitutions :
  let m := substitution_methods 18 9 3
  m = 45010 ∧ m % 1000 = 10 := by
  sorry


end basketball_substitutions_l3876_387611


namespace other_roots_form_new_equation_l3876_387623

theorem other_roots_form_new_equation (a₁ a₂ a₃ : ℝ) :
  let eq1 := fun x => x^2 + a₁*x + a₂*a₃
  let eq2 := fun x => x^2 + a₂*x + a₁*a₃
  let eq3 := fun x => x^2 + a₃*x + a₁*a₂
  (∃! α, eq1 α = 0 ∧ eq2 α = 0) →
  ∃ β γ, eq1 β = 0 ∧ eq2 γ = 0 ∧ β ≠ γ ∧ eq3 β = 0 ∧ eq3 γ = 0 :=
by sorry


end other_roots_form_new_equation_l3876_387623


namespace min_pizzas_cover_scooter_cost_l3876_387669

def scooter_cost : ℕ := 8000
def earning_per_pizza : ℕ := 12
def cost_per_delivery : ℕ := 4

def min_pizzas : ℕ := 1000

theorem min_pizzas_cover_scooter_cost :
  ∀ p : ℕ, p ≥ min_pizzas →
  p * (earning_per_pizza - cost_per_delivery) ≥ scooter_cost :=
by sorry

end min_pizzas_cover_scooter_cost_l3876_387669


namespace coat_original_price_l3876_387674

/-- Proves that if a coat is sold for 135 yuan after a 25% discount, its original price was 180 yuan -/
theorem coat_original_price (discounted_price : ℝ) (discount_percent : ℝ) 
  (h1 : discounted_price = 135)
  (h2 : discount_percent = 25) : 
  discounted_price / (1 - discount_percent / 100) = 180 := by
sorry

end coat_original_price_l3876_387674


namespace drama_club_ratio_l3876_387693

theorem drama_club_ratio (girls boys : ℝ) (h : boys = 0.8 * girls) :
  girls = 1.25 * boys := by
  sorry

end drama_club_ratio_l3876_387693


namespace max_distance_ellipse_to_line_l3876_387640

/-- An ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop := λ x y ↦ x^2 / (a^2) + y^2 / (b^2) = 1

/-- A line in the xy-plane represented by parametric equations -/
structure ParametricLine where
  fx : ℝ → ℝ
  fy : ℝ → ℝ

/-- The distance between a point and a line -/
def distance_point_to_line (x y : ℝ) (l : ParametricLine) : ℝ := sorry

/-- The maximum distance from a point on an ellipse to a line -/
def max_distance (e : Ellipse) (l : ParametricLine) : ℝ := sorry

theorem max_distance_ellipse_to_line :
  let e : Ellipse := { a := 4, b := 2, equation := λ x y ↦ x^2 / 16 + y^2 / 4 = 1 }
  let l : ParametricLine := { fx := λ t ↦ Real.sqrt 2 - t, fy := λ t ↦ t / 2 }
  max_distance e l = Real.sqrt 10 := by sorry

end max_distance_ellipse_to_line_l3876_387640


namespace jaco_gift_budget_l3876_387662

/-- Given a total budget, number of friends, and cost of parent gifts, 
    calculate the budget for each friend's gift -/
def friend_gift_budget (total_budget : ℕ) (num_friends : ℕ) (parent_gift_cost : ℕ) : ℕ :=
  (total_budget - 2 * parent_gift_cost) / num_friends

/-- Proof that Jaco's budget for each friend's gift is $9 -/
theorem jaco_gift_budget :
  friend_gift_budget 100 8 14 = 9 := by
  sorry

end jaco_gift_budget_l3876_387662


namespace toms_brick_cost_l3876_387675

/-- The total cost of bricks for Tom's shed -/
def total_cost (total_bricks : ℕ) (full_price : ℚ) (discount_percent : ℚ) : ℚ :=
  let half_bricks := total_bricks / 2
  let discounted_price := full_price * (1 - discount_percent)
  (half_bricks : ℚ) * discounted_price + (half_bricks : ℚ) * full_price

/-- Theorem stating the total cost for Tom's bricks -/
theorem toms_brick_cost :
  total_cost 1000 (1/2) (1/2) = 375 := by
  sorry

end toms_brick_cost_l3876_387675


namespace ellipse_hyperbola_product_l3876_387670

/-- Given an ellipse and a hyperbola with specific foci, prove the product of their semi-axes lengths -/
theorem ellipse_hyperbola_product (a b : ℝ) : 
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ (y = 5 ∨ y = -5))) → 
  (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (y = 0 ∧ (x = 7 ∨ x = -7))) → 
  |a * b| = 2 * Real.sqrt 111 := by
sorry

end ellipse_hyperbola_product_l3876_387670


namespace circle_area_from_circumference_l3876_387608

theorem circle_area_from_circumference (c : ℝ) (h : c = 24) :
  let r := c / (2 * Real.pi)
  (Real.pi * r * r) = 144 / Real.pi := by
  sorry

end circle_area_from_circumference_l3876_387608


namespace sqrt_inequality_l3876_387682

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt (a - 2) - Real.sqrt (a - 3) > Real.sqrt a - Real.sqrt (a - 1) := by
  sorry

end sqrt_inequality_l3876_387682


namespace orchestra_seat_price_l3876_387622

/-- Represents the theater ticket sales scenario --/
structure TheaterSales where
  orchestra_price : ℕ
  balcony_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ
  balcony_orchestra_diff : ℕ

/-- Theorem stating the orchestra seat price given the conditions --/
theorem orchestra_seat_price (ts : TheaterSales)
  (h1 : ts.balcony_price = 8)
  (h2 : ts.total_tickets = 340)
  (h3 : ts.total_revenue = 3320)
  (h4 : ts.balcony_orchestra_diff = 40) :
  ts.orchestra_price = 12 := by
  sorry


end orchestra_seat_price_l3876_387622


namespace power_function_through_sqrt2_l3876_387621

/-- A power function that passes through the point (2, √2) is equal to the square root function. -/
theorem power_function_through_sqrt2 (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x^α) →   -- f is a power function
  f 2 = Real.sqrt 2 →      -- f passes through (2, √2)
  ∀ x > 0, f x = Real.sqrt x := by
sorry

end power_function_through_sqrt2_l3876_387621


namespace average_age_combined_l3876_387697

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 35 →
  let total_age_students := num_students * avg_age_students
  let total_age_parents := num_parents * avg_age_parents
  let total_people := num_students + num_parents
  let total_age := total_age_students + total_age_parents
  (total_age / total_people : ℚ) = 25.8 := by
  sorry

end average_age_combined_l3876_387697


namespace statement_I_statement_II_statement_III_statement_IV_l3876_387618

-- Define the complex square root function
noncomputable def complexSqrt : ℂ → ℂ := sorry

-- Statement (I)
theorem statement_I (a b : ℂ) : complexSqrt (a^2 + b^2) = 0 ↔ a = 0 ∧ b = 0 := by sorry

-- Statement (II)
theorem statement_II : ¬∃ (a b : ℂ), (a ≠ 0 ∨ b ≠ 0) ∧ complexSqrt (a^2 + b^2) = a * b := by sorry

-- Statement (III)
theorem statement_III : ¬∃ (a b : ℂ), (a ≠ 0 ∨ b ≠ 0) ∧ complexSqrt (a^2 + b^2) = a + b := by sorry

-- Statement (IV)
theorem statement_IV : ¬∃ (a b : ℂ), (a ≠ 0 ∨ b ≠ 0) ∧ complexSqrt (a^2 + b^2) = a * b := by sorry

end statement_I_statement_II_statement_III_statement_IV_l3876_387618


namespace car_trade_profit_l3876_387646

theorem car_trade_profit (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let buying_price := original_price * (1 - 0.05)
  let selling_price := buying_price * (1 + 0.60)
  let profit := selling_price - original_price
  let profit_percentage := (profit / original_price) * 100
  profit_percentage = 52 := by sorry

end car_trade_profit_l3876_387646


namespace students_taking_courses_l3876_387601

theorem students_taking_courses (total : ℕ) (history : ℕ) (statistics : ℕ) (history_only : ℕ)
  (h_total : total = 89)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_only : history_only = 27) :
  ∃ (both : ℕ) (statistics_only : ℕ),
    history_only + statistics_only + both = 59 ∧
    both = history - history_only ∧
    statistics_only = statistics - both :=
by sorry

end students_taking_courses_l3876_387601


namespace commission_percentage_is_4_percent_l3876_387643

/-- Represents the commission rate as a real number between 0 and 1 -/
def CommissionRate : Type := { r : ℝ // 0 ≤ r ∧ r ≤ 1 }

/-- The first salary option -/
def salary1 : ℝ := 1800

/-- The base salary for the second option -/
def baseSalary : ℝ := 1600

/-- The sales amount at which both options are equal -/
def equalSalesAmount : ℝ := 5000

/-- The commission rate that makes both options equal at the given sales amount -/
def commissionRate : CommissionRate :=
  sorry

theorem commission_percentage_is_4_percent :
  (commissionRate.val * 100 : ℝ) = 4 :=
sorry

end commission_percentage_is_4_percent_l3876_387643


namespace equation_has_four_real_solutions_l3876_387600

theorem equation_has_four_real_solutions :
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, (5 * x) / (x^2 + 2*x + 4) + (7 * x) / (x^2 - 7*x + 4) = -2) ∧ 
    s.card = 4 := by
  sorry

end equation_has_four_real_solutions_l3876_387600


namespace shari_walking_distance_l3876_387645

-- Define Shari's walking speed
def walking_speed : ℝ := 4

-- Define the duration of the first walking segment
def first_segment_duration : ℝ := 2

-- Define the duration of the rest period (not used in calculation)
def rest_duration : ℝ := 0.5

-- Define the duration of the second walking segment
def second_segment_duration : ℝ := 1

-- Define the function to calculate distance
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem shari_walking_distance :
  distance walking_speed first_segment_duration +
  distance walking_speed second_segment_duration = 12 := by
  sorry

end shari_walking_distance_l3876_387645


namespace inequality_integer_solutions_l3876_387667

theorem inequality_integer_solutions :
  ∃! (s : Finset ℤ), (∀ x ∈ s, (x - 1 : ℚ) / 3 < 5 / 7 ∧ 5 / 7 < (x + 4 : ℚ) / 5) ∧
    s.card = 4 :=
by sorry

end inequality_integer_solutions_l3876_387667


namespace isosceles_triangle_base_range_l3876_387650

/-- An isosceles triangle with perimeter 16 -/
structure IsoscelesTriangle where
  x : ℝ  -- base length
  y : ℝ  -- leg length
  perimeter_eq : x + 2*y = 16
  leg_eq : y = -1/2 * x + 8

/-- The range of the base length x in an isosceles triangle -/
theorem isosceles_triangle_base_range (t : IsoscelesTriangle) : 0 < t.x ∧ t.x < 8 := by
  sorry

#check isosceles_triangle_base_range

end isosceles_triangle_base_range_l3876_387650


namespace factorization_problems_l3876_387619

theorem factorization_problems (x y : ℝ) : 
  (7 * x^2 - 63 = 7 * (x + 3) * (x - 3)) ∧ 
  (x^3 + 6 * x^2 * y + 9 * x * y^2 = x * (x + 3 * y)^2) := by
  sorry

end factorization_problems_l3876_387619


namespace smallest_coprime_to_210_l3876_387610

theorem smallest_coprime_to_210 :
  ∀ y : ℕ, y > 1 → y < 11 → Nat.gcd y 210 ≠ 1 ∧ Nat.gcd 11 210 = 1 := by
  sorry

end smallest_coprime_to_210_l3876_387610


namespace tamika_always_greater_l3876_387659

def tamika_set : Set ℕ := {6, 7, 8}
def carlos_set : Set ℕ := {2, 4, 5}

def tamika_product (a b : ℕ) : Prop := a ∈ tamika_set ∧ b ∈ tamika_set ∧ a ≠ b
def carlos_product (c d : ℕ) : Prop := c ∈ carlos_set ∧ d ∈ carlos_set ∧ c ≠ d

theorem tamika_always_greater :
  ∀ (a b c d : ℕ), tamika_product a b → carlos_product c d →
    a * b > c * d :=
sorry

end tamika_always_greater_l3876_387659


namespace cookie_sharing_l3876_387672

theorem cookie_sharing (total_cookies : ℕ) (cookies_per_person : ℕ) (h1 : total_cookies = 24) (h2 : cookies_per_person = 4) :
  total_cookies / cookies_per_person = 6 :=
by sorry

end cookie_sharing_l3876_387672


namespace convex_quadrilateral_symmetric_division_l3876_387677

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  -- We don't need to define the specifics of the quadrilateral,
  -- just that it exists and is convex
  isConvex : Bool

/-- A polygon with an axis of symmetry -/
structure SymmetricPolygon where
  -- We don't need to define the specifics of the polygon,
  -- just that it exists and has an axis of symmetry
  hasSymmetryAxis : Bool

/-- A division of a quadrilateral into polygons -/
structure QuadrilateralDivision (q : ConvexQuadrilateral) where
  polygons : List SymmetricPolygon
  divisionValid : Bool  -- This would ensure the division is valid

/-- The main theorem -/
theorem convex_quadrilateral_symmetric_division 
  (q : ConvexQuadrilateral) : 
  ∃ (d : QuadrilateralDivision q), 
    d.polygons.length = 5 ∧ 
    d.divisionValid ∧ 
    ∀ p ∈ d.polygons, p.hasSymmetryAxis := by
  sorry

end convex_quadrilateral_symmetric_division_l3876_387677


namespace rhombus_perimeter_l3876_387664

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

#check rhombus_perimeter

end rhombus_perimeter_l3876_387664


namespace incomplete_factor_multiple_statement_l3876_387630

theorem incomplete_factor_multiple_statement : ¬(56 / 7 = 8 → (∃n : ℕ, 56 = n * 7) ∧ (∃m : ℕ, 7 * m = 56)) := by
  sorry

end incomplete_factor_multiple_statement_l3876_387630


namespace ratio_of_x_intercepts_l3876_387631

/-- Given two lines with the same non-zero y-intercept, where the first line has
    a slope of 8 and an x-intercept of (u, 0), and the second line has a slope
    of 4 and an x-intercept of (v, 0), prove that the ratio of u to v is 1/2. -/
theorem ratio_of_x_intercepts (b : ℝ) (u v : ℝ) 
    (h1 : b ≠ 0)
    (h2 : 0 = 8 * u + b)
    (h3 : 0 = 4 * v + b) :
    u / v = 1 / 2 := by
  sorry

end ratio_of_x_intercepts_l3876_387631


namespace infinite_special_integers_l3876_387681

theorem infinite_special_integers : 
  ∃ f : ℕ → ℕ, Infinite {n : ℕ | ∃ m : ℕ, 
    n = m * (m + 1) + 2 ∧ 
    ∀ p : ℕ, Prime p → p ∣ (n^2 + 3) → 
      ∃ k : ℕ, k^2 < n ∧ p ∣ (k^2 + 3)} :=
sorry

end infinite_special_integers_l3876_387681


namespace multiplication_problem_l3876_387603

theorem multiplication_problem : 10 * (3/27) * 36 = 40 := by
  sorry

end multiplication_problem_l3876_387603


namespace racing_track_circumference_difference_l3876_387604

theorem racing_track_circumference_difference
  (r : ℝ)
  (inner_radius : ℝ)
  (outer_radius : ℝ)
  (track_width : ℝ)
  (h1 : inner_radius = 2 * r)
  (h2 : outer_radius = inner_radius + track_width)
  (h3 : track_width = 15)
  : 2 * Real.pi * outer_radius - 2 * Real.pi * inner_radius = 30 * Real.pi :=
by
  sorry

end racing_track_circumference_difference_l3876_387604


namespace couple_consistency_l3876_387656

-- Define the possible types of people
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a couple
structure Couple :=
  (husband : PersonType)
  (wife : PersonType)

-- Define the statement made by a person about their spouse
def makeStatement (speaker : PersonType) (spouse : PersonType) : Prop :=
  spouse ≠ PersonType.Normal

-- Define the consistency of statements with reality
def isConsistent (couple : Couple) : Prop :=
  match couple.husband, couple.wife with
  | PersonType.Knight, _ => makeStatement PersonType.Knight couple.wife
  | PersonType.Liar, _ => ¬(makeStatement PersonType.Liar couple.wife)
  | PersonType.Normal, PersonType.Knight => makeStatement PersonType.Normal couple.wife
  | PersonType.Normal, PersonType.Liar => ¬(makeStatement PersonType.Normal couple.wife)
  | PersonType.Normal, PersonType.Normal => True

-- Theorem stating that the only consistent solution is both being normal people
theorem couple_consistency :
  ∀ (couple : Couple),
    isConsistent couple ∧
    makeStatement couple.husband couple.wife ∧
    makeStatement couple.wife couple.husband →
    couple.husband = PersonType.Normal ∧
    couple.wife = PersonType.Normal :=
sorry

end couple_consistency_l3876_387656


namespace profit_maximized_at_100_l3876_387654

/-- The profit function L(x) for annual production x (in thousand units) -/
noncomputable def L (x : ℝ) : ℝ :=
  if x < 80 then
    -1/3 * x^2 + 40 * x - 250
  else
    1200 - (x + 10000 / x)

/-- Annual fixed cost in ten thousand yuan -/
def annual_fixed_cost : ℝ := 250

/-- Price per unit in ten thousand yuan -/
def price_per_unit : ℝ := 50

theorem profit_maximized_at_100 :
  ∀ x > 0, L x ≤ L 100 :=
sorry

end profit_maximized_at_100_l3876_387654


namespace h_at_two_l3876_387638

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the tangent line function g
def g (x : ℝ) : ℝ := (3*x^2 - 3)*x - 2*x^3

-- Define the function h
def h (x : ℝ) : ℝ := f x - g x

-- Theorem statement
theorem h_at_two : h 2 = 2^3 - 12*2 + 16 := by sorry

end h_at_two_l3876_387638


namespace coffee_brew_efficiency_l3876_387651

theorem coffee_brew_efficiency (total_lbs : ℕ) (cups_per_day : ℕ) (total_days : ℕ) 
  (h1 : total_lbs = 3)
  (h2 : cups_per_day = 3)
  (h3 : total_days = 40) :
  (cups_per_day * total_days) / total_lbs = 40 := by
  sorry

#check coffee_brew_efficiency

end coffee_brew_efficiency_l3876_387651


namespace simplify_fourth_root_l3876_387641

theorem simplify_fourth_root (a : ℝ) (h : a < 1/2) : 
  (2*a - 1)^2^(1/4) = Real.sqrt (1 - 2*a) := by
  sorry

end simplify_fourth_root_l3876_387641


namespace smallest_quadratic_nonresidue_bound_l3876_387644

theorem smallest_quadratic_nonresidue_bound (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ x : ℕ, x < Int.floor (Real.sqrt p + 1) ∧ x > 0 ∧ ¬ ∃ y : ℤ, (y * y) % p = x % p := by
  sorry

end smallest_quadratic_nonresidue_bound_l3876_387644


namespace xy_value_l3876_387626

theorem xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h1 : x^2 + y^2 = 3) (h2 : x^4 + y^4 = 15/8) : x * y = Real.sqrt 57 / 4 := by
  sorry

end xy_value_l3876_387626


namespace fraction_sum_equals_decimal_l3876_387685

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 10 + (5 : ℚ) / 100 + (7 : ℚ) / 1000 + (1 : ℚ) / 1000 = (358 : ℚ) / 1000 := by
  sorry

end fraction_sum_equals_decimal_l3876_387685


namespace angle_sum_around_point_l3876_387607

theorem angle_sum_around_point (x : ℝ) : 
  (3 * x + 7 * x + 4 * x + x = 360) → x = 24 := by
  sorry

end angle_sum_around_point_l3876_387607


namespace time_to_find_two_artifacts_l3876_387632

/-- The time it takes to find two artifacts given research and expedition times for the first, 
    and a multiplier for the second. -/
def time_to_find_artifacts (research_time : ℝ) (expedition_time : ℝ) (multiplier : ℝ) : ℝ :=
  let first_artifact_time := research_time + expedition_time
  let second_artifact_time := multiplier * first_artifact_time
  first_artifact_time + second_artifact_time

/-- Theorem stating that under the given conditions, it takes 10 years to find both artifacts. -/
theorem time_to_find_two_artifacts : 
  time_to_find_artifacts 0.5 2 3 = 10 := by
  sorry

end time_to_find_two_artifacts_l3876_387632
