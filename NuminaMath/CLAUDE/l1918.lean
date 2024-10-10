import Mathlib

namespace expression_simplification_l1918_191892

theorem expression_simplification :
  ((1 + 2 + 3 + 4 + 5 + 6) / 3) + ((3 * 5 + 12) / 4) = 13.75 := by
sorry

end expression_simplification_l1918_191892


namespace points_three_units_from_negative_two_l1918_191831

theorem points_three_units_from_negative_two (x : ℝ) : 
  (x = 1 ∨ x = -5) ↔ |x + 2| = 3 :=
by sorry

end points_three_units_from_negative_two_l1918_191831


namespace sample_capacity_l1918_191848

theorem sample_capacity (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ) 
  (h1 : frequency = 36)
  (h2 : frequency_rate = 1/4)
  (h3 : frequency_rate = frequency / n) : n = 144 := by
  sorry

end sample_capacity_l1918_191848


namespace sum_and_multiply_base8_l1918_191806

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 -/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Sums numbers from 1 to n in base 8 -/
def sumBase8 (n : ℕ) : ℕ := sorry

theorem sum_and_multiply_base8 :
  base10ToBase8 (3 * (sumBase8 (base8ToBase10 30))) = 1604 := by sorry

end sum_and_multiply_base8_l1918_191806


namespace count_strictly_ordered_three_digit_numbers_l1918_191871

/-- The number of three-digit numbers with digits from 1 to 9 in strictly increasing or decreasing order -/
def strictly_ordered_three_digit_numbers : ℕ :=
  2 * (Nat.choose 9 3)

/-- Theorem: The number of three-digit numbers with digits from 1 to 9 
    in strictly increasing or decreasing order is 168 -/
theorem count_strictly_ordered_three_digit_numbers :
  strictly_ordered_three_digit_numbers = 168 := by
  sorry

end count_strictly_ordered_three_digit_numbers_l1918_191871


namespace min_y_value_l1918_191867

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 8*x + 54*y) :
  ∃ (y_min : ℝ), y_min = 27 - Real.sqrt 745 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 8*x' + 54*y' → y' ≥ y_min :=
sorry

end min_y_value_l1918_191867


namespace remainder_conversion_l1918_191804

theorem remainder_conversion (N : ℕ) : 
  N % 72 = 68 → N % 24 = 20 := by
sorry

end remainder_conversion_l1918_191804


namespace exists_monochromatic_equilateral_triangle_l1918_191830

/-- A color type representing red or blue -/
inductive Color
  | Red
  | Blue

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A coloring function that assigns a color to each point in the plane -/
def Coloring := Point → Color

/-- Predicate to check if three points form an equilateral triangle -/
def IsEquilateralTriangle (p1 p2 p3 : Point) : Prop := sorry

/-- Theorem stating that in any coloring of the plane, there exist three points
    of the same color forming an equilateral triangle -/
theorem exists_monochromatic_equilateral_triangle (c : Coloring) :
  ∃ (p1 p2 p3 : Point) (col : Color),
    c p1 = col ∧ c p2 = col ∧ c p3 = col ∧
    IsEquilateralTriangle p1 p2 p3 := by
  sorry

end exists_monochromatic_equilateral_triangle_l1918_191830


namespace square_of_binomial_constant_l1918_191857

theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → a = 25 := by
  sorry

end square_of_binomial_constant_l1918_191857


namespace trapezoid_bases_count_l1918_191861

theorem trapezoid_bases_count : ∃! n : ℕ, 
  n = (Finset.filter (fun p : ℕ × ℕ => 
    10 ∣ p.1 ∧ 10 ∣ p.2 ∧ 
    (p.1 + p.2) * 30 = 1800 ∧ 
    0 < p.1 ∧ 0 < p.2) (Finset.product (Finset.range 181) (Finset.range 181))).card ∧
  n = 4 := by
sorry

end trapezoid_bases_count_l1918_191861


namespace cyclist_distance_difference_l1918_191887

/-- The difference in distance traveled by two cyclists after five hours -/
theorem cyclist_distance_difference
  (daniel_distance : ℝ)
  (evan_initial_distance : ℝ)
  (evan_initial_time : ℝ)
  (evan_break_time : ℝ)
  (total_time : ℝ)
  (h1 : daniel_distance = 65)
  (h2 : evan_initial_distance = 40)
  (h3 : evan_initial_time = 3)
  (h4 : evan_break_time = 0.5)
  (h5 : total_time = 5) :
  daniel_distance - (evan_initial_distance + (evan_initial_distance / evan_initial_time) * (total_time - evan_initial_time - evan_break_time)) = 5 := by
  sorry

end cyclist_distance_difference_l1918_191887


namespace smallest_prime_factor_of_2537_l1918_191865

theorem smallest_prime_factor_of_2537 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2537 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2537 → p ≤ q :=
by sorry

end smallest_prime_factor_of_2537_l1918_191865


namespace unpainted_area_calculation_l1918_191824

theorem unpainted_area_calculation (board_width1 board_width2 : ℝ) 
  (angle : ℝ) (h1 : board_width1 = 5) (h2 : board_width2 = 8) 
  (h3 : angle = 45 * π / 180) : 
  board_width1 * (board_width2 * Real.sin angle) = 20 * Real.sqrt 2 := by
  sorry

end unpainted_area_calculation_l1918_191824


namespace problem_solution_l1918_191872

theorem problem_solution (x y : ℝ) (h1 : y = 3) (h2 : x * y + 4 = 19) : x = 5 := by
  sorry

end problem_solution_l1918_191872


namespace product_abcd_l1918_191816

theorem product_abcd (a b c d : ℚ) : 
  (2 * a + 3 * b + 5 * c + 8 * d = 45) →
  (4 * (d + c) = b) →
  (4 * b + c = a) →
  (c + 1 = d) →
  (a * b * c * d = (1511 / 103) * (332 / 103) * (-7 / 103) * (96 / 103)) := by
  sorry

end product_abcd_l1918_191816


namespace part_one_part_two_l1918_191836

-- Define propositions p and q
def p (x a : ℝ) : Prop := (x + a) / (x - 3 * a) < 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - 5*x + 6 ≤ 0

-- Part I
theorem part_one (x : ℝ) (h : p x 1 ∧ q x) : 2 ≤ x ∧ x < 3 := by sorry

-- Part II
theorem part_two (a : ℝ) (h : ∀ x, q x → p x a) (h' : ∃ x, p x a ∧ ¬q x) : a > 1 := by sorry

end part_one_part_two_l1918_191836


namespace ana_overall_percentage_l1918_191877

-- Define the number of problems and percentage correct for each test
def test1_problems : ℕ := 20
def test1_percent : ℚ := 75 / 100

def test2_problems : ℕ := 50
def test2_percent : ℚ := 85 / 100

def test3_problems : ℕ := 30
def test3_percent : ℚ := 80 / 100

-- Define the total number of problems
def total_problems : ℕ := test1_problems + test2_problems + test3_problems

-- Define the total number of correct answers
def total_correct : ℚ := test1_problems * test1_percent + test2_problems * test2_percent + test3_problems * test3_percent

-- Theorem statement
theorem ana_overall_percentage :
  (total_correct / total_problems : ℚ) = 815 / 1000 := by
  sorry

end ana_overall_percentage_l1918_191877


namespace min_sum_of_reciprocal_line_l1918_191823

/-- Given a line (x/a) + (y/b) = 1 where a > 0 and b > 0, 
    and the line passes through the point (1, 1),
    the minimum value of a + b is 4. -/
theorem min_sum_of_reciprocal_line (a b : ℝ) : 
  a > 0 → b > 0 → (1 / a + 1 / b = 1) → 
  ∀ (a' b' : ℝ), a' > 0 → b' > 0 → (1 / a' + 1 / b' = 1) → 
  a + b ≤ a' + b' ∧ a + b = 4 := by
  sorry

end min_sum_of_reciprocal_line_l1918_191823


namespace ratio_equality_l1918_191840

theorem ratio_equality (x y : ℝ) (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 := by
  sorry

end ratio_equality_l1918_191840


namespace y_axis_reflection_l1918_191820

/-- Given a point P with coordinates (-5, 3), its reflection across the y-axis has coordinates (5, 3). -/
theorem y_axis_reflection :
  let P : ℝ × ℝ := (-5, 3)
  let P_reflected : ℝ × ℝ := (5, 3)
  P_reflected = (- P.1, P.2) :=
by sorry

end y_axis_reflection_l1918_191820


namespace right_triangle_area_l1918_191833

theorem right_triangle_area (base height : ℝ) (h1 : base = 3) (h2 : height = 4) :
  (1/2 : ℝ) * base * height = 6 := by
  sorry

end right_triangle_area_l1918_191833


namespace polynomial_simplification_l1918_191879

theorem polynomial_simplification (x y : ℝ) :
  (4 * x^9 + 3 * y^8 + 5 * x^7) + (2 * x^10 + 6 * x^9 + y^8 + 4 * x^7 + 2 * y^4 + 7 * x + 9) =
  2 * x^10 + 10 * x^9 + 4 * y^8 + 9 * x^7 + 2 * y^4 + 7 * x + 9 :=
by sorry

end polynomial_simplification_l1918_191879


namespace inequality_solution_set_range_l1918_191894

/-- The inequality we're working with -/
def inequality (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 - m * x - 1 < 2 * x^2 - 2 * x

/-- The solution set of the inequality with respect to x is R -/
def solution_set_is_R (m : ℝ) : Prop :=
  ∀ x : ℝ, inequality m x

/-- The range of m values for which the solution set is R -/
def m_range : Set ℝ :=
  {m : ℝ | m > -2 ∧ m ≤ 2}

/-- The main theorem to prove -/
theorem inequality_solution_set_range :
  ∀ m : ℝ, solution_set_is_R m ↔ m ∈ m_range :=
sorry

end inequality_solution_set_range_l1918_191894


namespace binary_arithmetic_equality_l1918_191897

-- Define binary numbers as natural numbers
def bin1010 : ℕ := 10
def bin111 : ℕ := 7
def bin1001 : ℕ := 9
def bin1011 : ℕ := 11
def bin10111 : ℕ := 23

-- Theorem statement
theorem binary_arithmetic_equality :
  (bin1010 + bin111) - bin1001 + bin1011 = bin10111 := by
  sorry

end binary_arithmetic_equality_l1918_191897


namespace closest_to_sqrt_two_l1918_191898

theorem closest_to_sqrt_two : 
  let a := Real.sqrt 3 * Real.cos (14 * π / 180) + Real.sin (14 * π / 180)
  let b := Real.sqrt 3 * Real.cos (24 * π / 180) + Real.sin (24 * π / 180)
  let c := Real.sqrt 3 * Real.cos (64 * π / 180) + Real.sin (64 * π / 180)
  let d := Real.sqrt 3 * Real.cos (74 * π / 180) + Real.sin (74 * π / 180)
  abs (d - Real.sqrt 2) < min (abs (a - Real.sqrt 2)) (min (abs (b - Real.sqrt 2)) (abs (c - Real.sqrt 2))) := by
  sorry

end closest_to_sqrt_two_l1918_191898


namespace greatest_common_multiple_10_15_under_100_l1918_191878

def is_common_multiple (n m k : ℕ) : Prop := k % n = 0 ∧ k % m = 0

theorem greatest_common_multiple_10_15_under_100 :
  ∃ (k : ℕ), k < 100 ∧ is_common_multiple 10 15 k ∧
  ∀ (j : ℕ), j < 100 → is_common_multiple 10 15 j → j ≤ k :=
by sorry

end greatest_common_multiple_10_15_under_100_l1918_191878


namespace solve_grocery_problem_l1918_191825

def grocery_problem (total_budget : ℝ) (chicken_cost bacon_cost vegetable_cost : ℝ)
  (apple_cost : ℝ) (apple_count : ℕ) (hummus_count : ℕ) : Prop :=
  let remaining_after_meat_and_veg := total_budget - (chicken_cost + bacon_cost + vegetable_cost)
  let remaining_after_apples := remaining_after_meat_and_veg - (apple_cost * apple_count)
  let hummus_total_cost := remaining_after_apples
  let hummus_unit_cost := hummus_total_cost / hummus_count
  hummus_unit_cost = 5

theorem solve_grocery_problem :
  grocery_problem 60 20 10 10 2 5 2 := by
  sorry

end solve_grocery_problem_l1918_191825


namespace problem_statement_l1918_191841

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) :
  (x - 1)^2 + 9/(x - 1)^2 = 3 + 8/x :=
by sorry

end problem_statement_l1918_191841


namespace vasyas_numbers_l1918_191895

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end vasyas_numbers_l1918_191895


namespace chessboard_pawn_placement_l1918_191838

/-- Represents a chess board configuration -/
structure ChessBoard :=
  (size : Nat)
  (pawns : Nat)

/-- Calculates the number of ways to place distinct pawns on a chess board -/
def placementWays (board : ChessBoard) : Nat :=
  (Nat.factorial board.size) * (Nat.factorial board.size)

/-- Theorem: The number of ways to place 5 distinct pawns on a 5x5 chess board,
    such that no row and no column contains more than one pawn, is 14400 -/
theorem chessboard_pawn_placement :
  let board : ChessBoard := ⟨5, 5⟩
  placementWays board = 14400 := by
  sorry

#eval placementWays ⟨5, 5⟩

end chessboard_pawn_placement_l1918_191838


namespace rbc_divisibility_l1918_191852

theorem rbc_divisibility (r b c : ℕ) : 
  r < 10 → b < 10 → c < 10 →
  (523 * 100 + r * 10 + b) * 10 + c ≡ 0 [MOD 7] →
  (523 * 100 + r * 10 + b) * 10 + c ≡ 0 [MOD 89] →
  r * b * c = 36 := by
sorry

end rbc_divisibility_l1918_191852


namespace expression_value_l1918_191864

theorem expression_value (m n a b x : ℝ) : 
  (m = -n) → 
  (a * b = 1) → 
  (abs x = 3) → 
  (x^3 - (1 + m + n - a*b) * x^2010 + (m + n) * x^2007 + (-a*b)^2009 = 26 ∨
   x^3 - (1 + m + n - a*b) * x^2010 + (m + n) * x^2007 + (-a*b)^2009 = -28) :=
by sorry

end expression_value_l1918_191864


namespace solve_for_y_l1918_191808

theorem solve_for_y (x y : ℝ) (h1 : 2 * (x - y) = 32) (h2 : x + y = -4) : y = -10 := by
  sorry

end solve_for_y_l1918_191808


namespace min_value_x2_plus_2y2_l1918_191856

theorem min_value_x2_plus_2y2 (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 = 2) :
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 - 2*a*b + 2*b^2 = 2 → a^2 + 2*b^2 ≥ m) ∧
             (∃ (c d : ℝ), c^2 - 2*c*d + 2*d^2 = 2 ∧ c^2 + 2*d^2 = m) ∧
             (m = 4 - 2 * Real.sqrt 2) :=
by sorry

end min_value_x2_plus_2y2_l1918_191856


namespace sin_2alpha_value_l1918_191842

theorem sin_2alpha_value (f : ℝ → ℝ) (a α : ℝ) :
  (∀ x, f x = 2 * Real.sin (2 * x + π / 6) + a * Real.cos (2 * x)) →
  (∀ x, f x = f (2 * π / 3 - x)) →
  0 < α →
  α < π / 3 →
  f α = 6 / 5 →
  Real.sin (2 * α) = (4 + 3 * Real.sqrt 3) / 10 := by
  sorry

end sin_2alpha_value_l1918_191842


namespace linda_total_coins_l1918_191814

/-- Represents the number of coins Linda has -/
structure Coins where
  dimes : Nat
  quarters : Nat
  nickels : Nat

/-- Calculates the total number of coins -/
def totalCoins (c : Coins) : Nat :=
  c.dimes + c.quarters + c.nickels

/-- Linda's initial coins -/
def initialCoins : Coins :=
  { dimes := 2, quarters := 6, nickels := 5 }

/-- Coins given by Linda's mother -/
def givenCoins (initial : Coins) : Coins :=
  { dimes := 2, quarters := 10, nickels := 2 * initial.nickels }

/-- Linda's final coins after receiving coins from her mother -/
def finalCoins (initial : Coins) : Coins :=
  { dimes := initial.dimes + (givenCoins initial).dimes,
    quarters := initial.quarters + (givenCoins initial).quarters,
    nickels := initial.nickels + (givenCoins initial).nickels }

theorem linda_total_coins :
  totalCoins (finalCoins initialCoins) = 35 := by
  sorry

end linda_total_coins_l1918_191814


namespace cubic_sum_inequality_l1918_191837

theorem cubic_sum_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  2 * (a^3 + b^3 + c^3) ≥ a^2*b + a*b^2 + a^2*c + a*c^2 + b^2*c + b*c^2 := by
  sorry

end cubic_sum_inequality_l1918_191837


namespace function_equation_implies_linear_l1918_191866

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f x - y * f y = (x - y) * f (x + y)

/-- The theorem stating that any function satisfying the equation must be linear -/
theorem function_equation_implies_linear (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := by
  sorry


end function_equation_implies_linear_l1918_191866


namespace equation_solutions_l1918_191873

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 2/3 ∧ ∀ x : ℝ, 3*x*(x-2) = 2*(x-2) ↔ x = x₁ ∨ x = x₂) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 7/2 ∧ y₂ = -2 ∧ ∀ x : ℝ, 2*x^2 - 3*x - 14 = 0 ↔ x = y₁ ∨ x = y₂) :=
by sorry

end equation_solutions_l1918_191873


namespace number_equation_l1918_191807

theorem number_equation : ∃ n : ℚ, n = (n - 5) * 4 := by
  use 20 / 3
  sorry

end number_equation_l1918_191807


namespace point_in_second_quadrant_l1918_191881

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def possible_b_values : Set ℝ := {-2, -1, 0, 2}

theorem point_in_second_quadrant (b : ℝ) :
  is_in_second_quadrant (-3) b ∧ b ∈ possible_b_values → b = 2 :=
by sorry

end point_in_second_quadrant_l1918_191881


namespace trapezoid_side_length_l1918_191819

/-- Represents a trapezoid ABCD with midline MN -/
structure Trapezoid where
  AD : ℝ  -- Length of side AD
  BC : ℝ  -- Length of side BC
  MN : ℝ  -- Length of midline MN
  is_trapezoid : AD ≠ BC  -- Ensures it's actually a trapezoid
  midline_property : MN = (AD + BC) / 2  -- Property of the midline

/-- Theorem: In a trapezoid with AD = 2 and MN = 6, BC must equal 10 -/
theorem trapezoid_side_length (T : Trapezoid) (h1 : T.AD = 2) (h2 : T.MN = 6) : T.BC = 10 := by
  sorry

end trapezoid_side_length_l1918_191819


namespace unique_solution_system_l1918_191869

/-- The system of equations has a unique solution when a = 1, 
    and the solution is x = -3/2, y = -1/2, z = 0 -/
theorem unique_solution_system (a x y z : ℝ) : 
  z = a * (x + 2 * y + 5/2) ∧ 
  x^2 + y^2 + 2*x - y + z = 0 ∧
  ((x + (a + 2)/2)^2 + (y + (2*a - 1)/2)^2 = ((a + 2)^2)/4 + ((2*a - 1)^2)/4 - 5*a/2) →
  (a = 1 ∧ x = -3/2 ∧ y = -1/2 ∧ z = 0) :=
by sorry

end unique_solution_system_l1918_191869


namespace orange_stack_theorem_l1918_191850

/-- Calculates the number of oranges in a trapezoidal layer -/
def trapezoidalLayer (a b h : ℕ) : ℕ := (a + b) * h / 2

/-- Calculates the total number of oranges in the stack -/
def orangeStack (baseA baseB height : ℕ) : ℕ :=
  let rec stackLayers (a b h : ℕ) : ℕ :=
    if h = 0 then 0
    else trapezoidalLayer a b h + stackLayers (a - 1) (b - 1) (h - 1)
  stackLayers baseA baseB height

theorem orange_stack_theorem :
  orangeStack 7 5 6 = 90 := by sorry

end orange_stack_theorem_l1918_191850


namespace union_of_sets_l1918_191809

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 4, 6}
  A ∪ B = {1, 2, 4, 6} := by
sorry

end union_of_sets_l1918_191809


namespace fish_count_l1918_191812

theorem fish_count (total_pets dogs cats : ℕ) (h1 : total_pets = 149) (h2 : dogs = 43) (h3 : cats = 34) :
  total_pets - (dogs + cats) = 72 := by
sorry

end fish_count_l1918_191812


namespace two_dice_probability_l1918_191835

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of favorable outcomes for the first die (rolling less than 4) -/
def favorableFirst : ℕ := 3

/-- The number of favorable outcomes for the second die (rolling greater than 4) -/
def favorableSecond : ℕ := 4

/-- The probability of the desired outcome when rolling two eight-sided dice -/
theorem two_dice_probability :
  (favorableFirst / numSides) * (favorableSecond / numSides) = 3 / 16 := by
  sorry

end two_dice_probability_l1918_191835


namespace complex_power_32_l1918_191858

open Complex

theorem complex_power_32 : (((1 : ℂ) - I) / (Real.sqrt 2 : ℂ)) ^ 32 = 1 := by
  sorry

end complex_power_32_l1918_191858


namespace g_of_two_equals_eighteen_l1918_191843

-- Define g as a function from ℝ to ℝ
variable (g : ℝ → ℝ)

-- State the theorem
theorem g_of_two_equals_eighteen
  (h : ∀ x : ℝ, g (3 * x - 7) = 4 * x + 6) :
  g 2 = 18 := by
  sorry

end g_of_two_equals_eighteen_l1918_191843


namespace find_a_value_l1918_191859

theorem find_a_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end find_a_value_l1918_191859


namespace right_triangle_side_length_l1918_191828

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2)  -- Pythagorean theorem
  (h2 : c = 10)           -- Hypotenuse is 10
  (h3 : a = 6)            -- One side is 6
  : b = 8 :=              -- Prove the other side is 8
by sorry

end right_triangle_side_length_l1918_191828


namespace point_labeling_theorem_l1918_191886

/-- A point in the space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- The set of n points in the space -/
def PointSet (n : ℕ) := Fin n → Point

theorem point_labeling_theorem (n : ℕ) (points : PointSet n) 
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    ∃ (p : Fin 3 → Fin n), angle (points (p 0)) (points (p 1)) (points (p 2)) > 120) :
  ∃ (σ : Equiv (Fin n) (Fin n)), 
    ∀ (i j k : Fin n), i < j → j < k → 
      angle (points (σ i)) (points (σ j)) (points (σ k)) > 120 :=
sorry

end point_labeling_theorem_l1918_191886


namespace garden_length_l1918_191885

/-- The length of a rectangular garden with perimeter 1800 m and breadth 400 m is 500 m. -/
theorem garden_length (perimeter breadth : ℝ) (h1 : perimeter = 1800) (h2 : breadth = 400) :
  (perimeter / 2 - breadth) = 500 := by
  sorry

end garden_length_l1918_191885


namespace not_perfect_square_l1918_191891

theorem not_perfect_square : ¬ ∃ (n : ℕ), n^2 = 425102348541 := by
  sorry

end not_perfect_square_l1918_191891


namespace subtraction_addition_equality_l1918_191880

theorem subtraction_addition_equality : ∃ x : ℤ, 100 - 70 = 70 + x ∧ x = -40 := by
  sorry

end subtraction_addition_equality_l1918_191880


namespace square_equation_solution_l1918_191849

theorem square_equation_solution (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 
  3*x^2*y^2 = 588 := by
  sorry

end square_equation_solution_l1918_191849


namespace complement_A_intersect_B_l1918_191845

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 2}

-- Define set B
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

end complement_A_intersect_B_l1918_191845


namespace probability_neither_mix_l1918_191832

/-- Represents the set of buyers -/
def Buyers : Type := Unit

/-- The total number of buyers -/
def total_buyers : ℕ := 100

/-- The number of buyers who purchase cake mix -/
def cake_mix_buyers : ℕ := 50

/-- The number of buyers who purchase muffin mix -/
def muffin_mix_buyers : ℕ := 40

/-- The number of buyers who purchase both cake mix and muffin mix -/
def both_mix_buyers : ℕ := 15

/-- The probability of selecting a buyer who purchases neither cake mix nor muffin mix -/
theorem probability_neither_mix (b : Buyers) : 
  (total_buyers - (cake_mix_buyers + muffin_mix_buyers - both_mix_buyers)) / total_buyers = 1/4 := by
  sorry

end probability_neither_mix_l1918_191832


namespace disc_probability_l1918_191889

theorem disc_probability (p_A p_B p_C p_D p_E : ℝ) : 
  p_A = 1/5 →
  p_B = 1/5 →
  p_C = p_D →
  p_C = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/5 := by
  sorry

end disc_probability_l1918_191889


namespace dog_walking_distance_l1918_191805

theorem dog_walking_distance (total_weekly_miles : ℝ) (dog2_daily_miles : ℝ) :
  total_weekly_miles = 70 →
  dog2_daily_miles = 8 →
  ∃ dog1_daily_miles : ℝ, 
    dog1_daily_miles * 7 + dog2_daily_miles * 7 = total_weekly_miles ∧
    dog1_daily_miles = 2 := by
  sorry

end dog_walking_distance_l1918_191805


namespace quadratic_inequality_range_l1918_191827

theorem quadratic_inequality_range (c : ℝ) : 
  (¬ ∀ x : ℝ, c ≤ -1/2 → x^2 + 4*c*x + 1 > 0) → c ≤ -1/2 := by
  sorry

end quadratic_inequality_range_l1918_191827


namespace gcd_bn_bn_plus_2_is_one_max_en_is_one_l1918_191826

theorem gcd_bn_bn_plus_2_is_one (n : ℕ) : 
  Nat.gcd ((10^n - 2) / 8) ((10^(n+2) - 2) / 8) = 1 := by
  sorry

theorem max_en_is_one : 
  ∀ n : ℕ, (∃ k : ℕ, Nat.gcd ((10^n - 2) / 8) ((10^(n+2) - 2) / 8) = k) → 
  Nat.gcd ((10^n - 2) / 8) ((10^(n+2) - 2) / 8) ≤ 1 := by
  sorry

end gcd_bn_bn_plus_2_is_one_max_en_is_one_l1918_191826


namespace square_of_1023_l1918_191803

theorem square_of_1023 : 1023^2 = 1046529 := by
  sorry

end square_of_1023_l1918_191803


namespace divisibility_of_all_ones_number_l1918_191876

/-- A positive integer whose decimal representation contains only ones -/
def all_ones_number (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem divisibility_of_all_ones_number (n : ℕ) (h : n > 0) :
  7 ∣ all_ones_number n → 13 ∣ all_ones_number n :=
by
  sorry

#check divisibility_of_all_ones_number

end divisibility_of_all_ones_number_l1918_191876


namespace range_of_p_l1918_191874

def h (x : ℝ) : ℝ := 4 * x - 3

def p (x : ℝ) : ℝ := h (h (h x))

theorem range_of_p :
  ∀ y ∈ Set.range (fun x => p x),
  (1 ≤ x ∧ x ≤ 3) → (1 ≤ y ∧ y ≤ 129) :=
by sorry

end range_of_p_l1918_191874


namespace largest_sum_is_five_sixths_l1918_191854

theorem largest_sum_is_five_sixths : 
  let sums := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/9, 1/3 + 1/6]
  ∀ x ∈ sums, x ≤ 5/6 ∧ (5/6 : ℚ) ∈ sums :=
by sorry

end largest_sum_is_five_sixths_l1918_191854


namespace min_original_tables_l1918_191813

/-- Given a restaurant scenario with customers and tables, prove that the minimum number of original tables is 3. -/
theorem min_original_tables (X Y Z A B C : ℕ) : 
  X = Z + A + B + C →  -- Total customers equals those who left plus those who remained
  Y ≥ 3 :=             -- The original number of tables is at least 3
by sorry

end min_original_tables_l1918_191813


namespace max_fleas_on_chessboard_l1918_191855

/-- Represents a 10x10 chessboard -/
def Chessboard := Fin 10 × Fin 10

/-- Represents the four possible directions a flea can move -/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents a flea's position and direction -/
structure Flea where
  position : Chessboard
  direction : Direction

/-- Represents the state of the board at a given time -/
def BoardState := List Flea

/-- Simulates the movement of fleas for one hour (60 minutes) -/
def simulateMovement (initial : BoardState) : List BoardState := sorry

/-- Checks if two fleas occupy the same square -/
def noCollision (state : BoardState) : Prop := sorry

/-- Checks if the simulation is valid (no collisions for 60 minutes) -/
def validSimulation (states : List BoardState) : Prop := sorry

/-- The main theorem: The maximum number of fleas on a 10x10 chessboard is 40 -/
theorem max_fleas_on_chessboard :
  ∀ (initial : BoardState),
    validSimulation (simulateMovement initial) →
    initial.length ≤ 40 := by
  sorry

end max_fleas_on_chessboard_l1918_191855


namespace angle_sum_equal_pi_over_two_l1918_191893

theorem angle_sum_equal_pi_over_two (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end angle_sum_equal_pi_over_two_l1918_191893


namespace product_of_exponents_l1918_191890

theorem product_of_exponents (p r s : ℕ) : 
  (2^p + 2^3 = 18) → 
  (3^r + 3 = 30) → 
  (4^s + 4^2 = 276) → 
  p * r * s = 48 := by
sorry

end product_of_exponents_l1918_191890


namespace tournament_rounds_theorem_l1918_191896

/-- Represents a person in the tournament -/
inductive Person : Type
  | A
  | B
  | C

/-- Represents the tournament data -/
structure TournamentData where
  rounds_played : Person → Nat
  referee_rounds : Person → Nat

/-- The total number of rounds in the tournament -/
def total_rounds (data : TournamentData) : Nat :=
  (data.rounds_played Person.A + data.rounds_played Person.B + data.rounds_played Person.C + 
   data.referee_rounds Person.A + data.referee_rounds Person.B + data.referee_rounds Person.C) / 2

theorem tournament_rounds_theorem (data : TournamentData) 
  (h1 : data.rounds_played Person.A = 5)
  (h2 : data.rounds_played Person.B = 6)
  (h3 : data.referee_rounds Person.C = 2) :
  total_rounds data = 9 := by
  sorry

end tournament_rounds_theorem_l1918_191896


namespace sets_properties_l1918_191800

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | ∃ y, y = x^2 + 1}
def B : Set ℝ := {y : ℝ | ∃ x, y = x^2 + 1}
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + 1}

-- Theorem stating the properties of A, B, and C
theorem sets_properties :
  (A = Set.univ) ∧
  (B = {y : ℝ | y ≥ 1}) ∧
  (C = {p : ℝ × ℝ | p.2 = p.1^2 + 1}) :=
by sorry

end sets_properties_l1918_191800


namespace perpendicular_lines_l1918_191884

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2-1) = 0 → 
    (a = -3/2 ∨ a = 0)) ∧
  (a = -3/2 ∨ a = 0 → 
    ∀ x y : ℝ, ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2-1) = 0) :=
by sorry

end perpendicular_lines_l1918_191884


namespace max_y_coordinate_sin_3theta_l1918_191883

theorem max_y_coordinate_sin_3theta (θ : Real) :
  let r := λ θ : Real => Real.sin (3 * θ)
  let y := λ θ : Real => r θ * Real.sin θ
  ∃ (max_y : Real), max_y = 9/64 ∧ ∀ θ', y θ' ≤ max_y := by sorry

end max_y_coordinate_sin_3theta_l1918_191883


namespace expression_simplification_l1918_191822

theorem expression_simplification (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  ((2*x + y) * (2*x - y) - (2*x - y)^2 - y*(x - 2*y)) / (2*x) = 3/2 := by
  sorry

end expression_simplification_l1918_191822


namespace folded_paper_area_l1918_191810

/-- The area of a folded rectangular paper -/
theorem folded_paper_area (length width : ℝ) (h_length : length = 17) (h_width : width = 8) :
  let original_area := length * width
  let folded_triangle_area := (1/2) * width * width
  original_area - folded_triangle_area = 104 :=
by
  sorry


end folded_paper_area_l1918_191810


namespace max_cylinder_volume_in_cube_l1918_191888

/-- The maximum volume of a cylinder inscribed in a cube with side length √3,
    where the cylinder's axis is along a diagonal of the cube. -/
theorem max_cylinder_volume_in_cube :
  let cube_side : ℝ := Real.sqrt 3
  let max_volume : ℝ := π / 2
  ∀ (cylinder_volume : ℝ),
    (∃ (cylinder_radius height : ℝ),
      cylinder_volume = π * cylinder_radius^2 * height ∧
      0 < cylinder_radius ∧
      0 < height ∧
      2 * Real.sqrt 2 * cylinder_radius + height = cube_side) →
    cylinder_volume ≤ max_volume :=
by sorry

end max_cylinder_volume_in_cube_l1918_191888


namespace annual_croissant_expenditure_is_858_l1918_191870

/-- The total annual expenditure on croissants -/
def annual_croissant_expenditure : ℚ :=
  let regular_cost : ℚ := 7/2
  let almond_cost : ℚ := 11/2
  let chocolate_cost : ℚ := 9/2
  let ham_cheese_cost : ℚ := 6
  let weeks_per_year : ℕ := 52
  regular_cost * weeks_per_year +
  almond_cost * weeks_per_year +
  chocolate_cost * weeks_per_year +
  ham_cheese_cost * (weeks_per_year / 2)

/-- Theorem stating that the annual croissant expenditure is $858.00 -/
theorem annual_croissant_expenditure_is_858 :
  annual_croissant_expenditure = 858 := by
  sorry

end annual_croissant_expenditure_is_858_l1918_191870


namespace k_range_l1918_191860

/-- The logarithm function to base 10 -/
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The equation lg kx = 2 lg (x+1) has only one real root -/
def has_unique_root (k : ℝ) : Prop :=
  ∃! x : ℝ, log10 (k * x) = 2 * log10 (x + 1)

/-- The range of k values for which the equation has only one real root -/
theorem k_range : ∀ k : ℝ, has_unique_root k ↔ k = 4 ∨ k < 0 := by sorry

end k_range_l1918_191860


namespace roses_per_set_l1918_191862

theorem roses_per_set (days_in_week : ℕ) (sets_per_day : ℕ) (total_roses : ℕ) :
  days_in_week = 7 →
  sets_per_day = 2 →
  total_roses = 168 →
  total_roses / (days_in_week * sets_per_day) = 12 :=
by
  sorry

end roses_per_set_l1918_191862


namespace fudge_piece_size_l1918_191875

/-- Given a rectangular pan of fudge with dimensions 18 inches by 29 inches,
    containing 522 square pieces, prove that each piece has a side length of 1 inch. -/
theorem fudge_piece_size (pan_length : ℝ) (pan_width : ℝ) (num_pieces : ℕ) 
    (h1 : pan_length = 18) 
    (h2 : pan_width = 29) 
    (h3 : num_pieces = 522) : 
  (pan_length * pan_width) / num_pieces = 1 := by
  sorry

#check fudge_piece_size

end fudge_piece_size_l1918_191875


namespace f_at_one_equals_neg_7007_l1918_191817

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 10
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 100*x + c

-- State the theorem
theorem f_at_one_equals_neg_7007 (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0 ∧
    f b c r₁ = 0 ∧ f b c r₂ = 0 ∧ f b c r₃ = 0) →
  f b c 1 = -7007 :=
by sorry


end f_at_one_equals_neg_7007_l1918_191817


namespace x_eq_one_sufficient_not_necessary_for_x_sq_eq_one_l1918_191853

theorem x_eq_one_sufficient_not_necessary_for_x_sq_eq_one :
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧
  (∃ x : ℝ, x ≠ 1 ∧ x^2 = 1) :=
by sorry

end x_eq_one_sufficient_not_necessary_for_x_sq_eq_one_l1918_191853


namespace parabola_trajectory_parabola_trajectory_is_parabola_l1918_191882

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

def parallelogram_point (A B F : Point) : Point :=
  Point.mk (A.x + B.x - F.x) (A.y + B.y - F.y)

def intersect_parabola_line (p : Parabola) (l : Line) : Set Point :=
  {P : Point | P.x^2 = 4 * P.y ∧ P.y = l.slope * P.x + l.intercept}

theorem parabola_trajectory (p : Parabola) (l : Line) (F : Point) :
  p.a = 1 ∧ p.h = 0 ∧ p.k = 0 ∧ 
  F.x = 0 ∧ F.y = 1 ∧
  l.intercept = -1 →
  ∃ (R : Point),
    (∃ (A B : Point), A ∈ intersect_parabola_line p l ∧ 
                      B ∈ intersect_parabola_line p l ∧ 
                      R = parallelogram_point A B F) ∧
    R.x^2 = 4 * (R.y + 3) ∧
    abs R.x > 4 :=
sorry

theorem parabola_trajectory_is_parabola (p : Parabola) (l : Line) (F : Point) :
  p.a = 1 ∧ p.h = 0 ∧ p.k = 0 ∧ 
  F.x = 0 ∧ F.y = 1 ∧
  l.intercept = -1 →
  ∃ (new_p : Parabola),
    new_p.a = 1 ∧ new_p.h = 0 ∧ new_p.k = -3 ∧
    (∀ (R : Point),
      (∃ (A B : Point), A ∈ intersect_parabola_line p l ∧ 
                        B ∈ intersect_parabola_line p l ∧ 
                        R = parallelogram_point A B F) →
      R.x^2 = 4 * (R.y + 3) ∧ abs R.x > 4) :=
sorry

end parabola_trajectory_parabola_trajectory_is_parabola_l1918_191882


namespace smallest_k_no_real_roots_l1918_191851

theorem smallest_k_no_real_roots : ∃ k : ℤ, 
  (∀ x : ℝ, 3 * x * (k * x - 5) - x^2 + 8 - x^3 ≠ 0) ∧ 
  (∀ m : ℤ, m < k → ∃ x : ℝ, 3 * x * (m * x - 5) - x^2 + 8 - x^3 = 0) ∧
  k = 1 := by
  sorry

end smallest_k_no_real_roots_l1918_191851


namespace h_at_two_equals_negative_three_l1918_191863

/-- The function h(x) = -5x + 7 -/
def h (x : ℝ) : ℝ := -5 * x + 7

/-- Theorem stating that h(2) = -3 -/
theorem h_at_two_equals_negative_three : h 2 = -3 := by
  sorry

end h_at_two_equals_negative_three_l1918_191863


namespace price_reduction_l1918_191839

theorem price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h1 : original_price = 120)
  (h2 : final_price = 85)
  (h3 : x > 0 ∧ x < 1) -- Assuming x is a valid percentage
  (h4 : final_price = original_price * (1 - x)^2) :
  120 * (1 - x)^2 = 85 := by sorry

end price_reduction_l1918_191839


namespace min_bananas_theorem_l1918_191801

/-- Represents the number of bananas a monkey takes from the pile -/
structure MonkeyTake where
  amount : ℕ

/-- Represents the final distribution of bananas among the monkeys -/
structure FinalDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of bananas in the pile -/
def totalBananas (t1 t2 t3 : MonkeyTake) : ℕ :=
  t1.amount + t2.amount + t3.amount

/-- Calculates the final distribution of bananas -/
def calculateDistribution (t1 t2 t3 : MonkeyTake) : FinalDistribution :=
  { first := 2 * t1.amount / 3 + t2.amount / 3 + 5 * t3.amount / 12
  , second := t1.amount / 6 + t2.amount / 3 + 5 * t3.amount / 12
  , third := t1.amount / 6 + t2.amount / 3 + t3.amount / 6 }

/-- Checks if the distribution satisfies the 4:3:2 ratio -/
def isValidRatio (d : FinalDistribution) : Prop :=
  3 * d.first = 4 * d.second ∧ 2 * d.second = 3 * d.third

/-- The main theorem stating the minimum number of bananas -/
theorem min_bananas_theorem (t1 t2 t3 : MonkeyTake) :
  (∀ d : FinalDistribution, d = calculateDistribution t1 t2 t3 → isValidRatio d) →
  totalBananas t1 t2 t3 ≥ 558 :=
sorry

end min_bananas_theorem_l1918_191801


namespace supplement_triple_angle_l1918_191811

theorem supplement_triple_angle : ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ x = 3 * (180 - x) ∧ x = 135 := by
  sorry

end supplement_triple_angle_l1918_191811


namespace fourth_term_is_2016_l1918_191815

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  second_term : a 2 = 606
  sum_first_four : a 1 + a 2 + a 3 + a 4 = 3834

/-- The fourth term of the arithmetic sequence is 2016 -/
theorem fourth_term_is_2016 (seq : ArithmeticSequence) : seq.a 4 = 2016 := by
  sorry

end fourth_term_is_2016_l1918_191815


namespace solve_fruit_problem_l1918_191834

def fruit_problem (apple_price orange_price : ℚ) (total_fruit : ℕ) (initial_avg_price : ℚ) (final_avg_price : ℚ) : Prop :=
  ∀ (apples oranges : ℕ),
    apple_price = 40 / 100 →
    orange_price = 60 / 100 →
    total_fruit = 10 →
    apples + oranges = total_fruit →
    (apple_price * apples + orange_price * oranges) / total_fruit = initial_avg_price →
    initial_avg_price = 56 / 100 →
    final_avg_price = 50 / 100 →
    ∃ (oranges_to_remove : ℕ),
      oranges_to_remove = 6 ∧
      (apple_price * apples + orange_price * (oranges - oranges_to_remove)) / (total_fruit - oranges_to_remove) = final_avg_price

theorem solve_fruit_problem :
  fruit_problem (40/100) (60/100) 10 (56/100) (50/100) :=
sorry

end solve_fruit_problem_l1918_191834


namespace class_size_from_mark_error_l1918_191802

theorem class_size_from_mark_error (mark_increase : ℕ) (average_increase : ℚ) : 
  mark_increase = 40 → average_increase = 1/2 → 
  (mark_increase : ℚ) / average_increase = 80 := by
  sorry

end class_size_from_mark_error_l1918_191802


namespace find_a_l1918_191818

-- Define the set A
def A (a : ℤ) : Set ℤ := {12, a^2 + 4*a, a - 2}

-- Theorem statement
theorem find_a : ∀ a : ℤ, -3 ∈ A a → a = -3 := by
  sorry

end find_a_l1918_191818


namespace cheat_sheet_distribution_l1918_191847

/-- Represents the number of pockets --/
def num_pockets : ℕ := 4

/-- Represents the number of cheat sheets --/
def num_cheat_sheets : ℕ := 6

/-- Represents the number of ways to place cheat sheets 1 and 2 --/
def ways_to_place_1_and_2 : ℕ := num_pockets

/-- Represents the number of ways to place cheat sheets 4 and 5 --/
def ways_to_place_4_and_5 : ℕ := num_pockets - 1

/-- Represents the number of ways to distribute the remaining cheat sheets --/
def ways_to_distribute_remaining : ℕ := 5

/-- Theorem stating the total number of ways to distribute the cheat sheets --/
theorem cheat_sheet_distribution :
  ways_to_place_1_and_2 * ways_to_place_4_and_5 * ways_to_distribute_remaining = 60 := by
  sorry

end cheat_sheet_distribution_l1918_191847


namespace bee_swarm_puzzle_l1918_191868

theorem bee_swarm_puzzle :
  ∃ (x : ℚ),
    x > 0 ∧
    (x / 5 + x / 3 + 3 * (x / 3 - x / 5) + 1 = x) ∧
    x = 15 :=
by sorry

end bee_swarm_puzzle_l1918_191868


namespace perpendicular_lines_l1918_191899

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line -/
def slope1 (a : ℝ) : ℝ := a

/-- The slope of the second line -/
def slope2 (a : ℝ) : ℝ := a + 2

/-- If the line y = ax - 2 is perpendicular to the line y = (a+2)x + 1, then a = -1 -/
theorem perpendicular_lines (a : ℝ) : 
  perpendicular (slope1 a) (slope2 a) → a = -1 := by sorry

end perpendicular_lines_l1918_191899


namespace circle_on_line_tangent_to_axes_l1918_191844

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - y = 3

-- Define the tangency condition to both axes
def tangent_to_axes (center_x center_y radius : ℝ) : Prop :=
  (abs center_x = radius ∧ abs center_y = radius)

-- Define the circle equation
def circle_equation (center_x center_y radius x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = radius^2

-- The main theorem
theorem circle_on_line_tangent_to_axes :
  ∀ (center_x center_y radius : ℝ),
    line_equation center_x center_y →
    tangent_to_axes center_x center_y radius →
    (∀ (x y : ℝ), circle_equation center_x center_y radius x y ↔
      ((x - 3)^2 + (y - 3)^2 = 9 ∨ (x - 1)^2 + (y + 1)^2 = 1)) :=
by sorry

end circle_on_line_tangent_to_axes_l1918_191844


namespace share_ratio_B_to_C_l1918_191829

def total_amount : ℕ := 510
def share_A : ℕ := 360
def share_B : ℕ := 90
def share_C : ℕ := 60

theorem share_ratio_B_to_C : 
  (share_B : ℚ) / (share_C : ℚ) = 3 / 2 :=
by sorry

end share_ratio_B_to_C_l1918_191829


namespace subletter_monthly_rent_subletter_rent_is_400_l1918_191821

/-- Calculates the monthly rent for each subletter given the number of subletters,
    John's monthly rent, and John's annual profit. -/
theorem subletter_monthly_rent 
  (num_subletters : ℕ) 
  (john_monthly_rent : ℕ) 
  (john_annual_profit : ℕ) : ℕ :=
  let total_annual_rent := john_monthly_rent * 12 + john_annual_profit
  total_annual_rent / (num_subletters * 12)

/-- Proves that each subletter pays $400 per month given the specific conditions. -/
theorem subletter_rent_is_400 :
  subletter_monthly_rent 3 900 3600 = 400 := by
  sorry

end subletter_monthly_rent_subletter_rent_is_400_l1918_191821


namespace function_has_positive_zero_l1918_191846

/-- The function f(x) = xe^x - ax - 1 has at least one positive zero for any real a. -/
theorem function_has_positive_zero (a : ℝ) : ∃ x : ℝ, x > 0 ∧ x * Real.exp x - a * x - 1 = 0 := by
  sorry

end function_has_positive_zero_l1918_191846
