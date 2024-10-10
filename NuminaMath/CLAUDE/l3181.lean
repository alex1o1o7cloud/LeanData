import Mathlib

namespace consecutive_integers_product_l3181_318179

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 15120 →
  e = 9 := by
  sorry

end consecutive_integers_product_l3181_318179


namespace remaining_cents_l3181_318143

-- Define the number of quarters Winston has
def initial_quarters : ℕ := 14

-- Define the value of a quarter in cents
def cents_per_quarter : ℕ := 25

-- Define the amount spent in cents (half a dollar)
def amount_spent : ℕ := 50

-- Theorem to prove
theorem remaining_cents :
  initial_quarters * cents_per_quarter - amount_spent = 300 := by
  sorry

end remaining_cents_l3181_318143


namespace smallest_square_division_smallest_square_division_is_two_total_squares_l3181_318133

theorem smallest_square_division (n : ℕ) : n > 0 ∧ 4*n - 4 = 2*n → n ≥ 2 :=
by sorry

theorem smallest_square_division_is_two : 
  ∃ (n : ℕ), n > 0 ∧ 4*n - 4 = 2*n ∧ ∀ (m : ℕ), (m > 0 ∧ 4*m - 4 = 2*m) → n ≤ m :=
by sorry

theorem total_squares (n : ℕ) : n > 0 ∧ 4*n - 4 = 2*n → n^2 = 4 :=
by sorry

end smallest_square_division_smallest_square_division_is_two_total_squares_l3181_318133


namespace problem_statement_l3181_318178

theorem problem_statement (a b : ℝ) (h : |a + 2| + (b - 3)^2 = 0) :
  (a + b)^2015 = 1 := by sorry

end problem_statement_l3181_318178


namespace smallest_gcd_multiple_l3181_318176

theorem smallest_gcd_multiple (p q : ℕ+) (h : Nat.gcd p q = 9) :
  (∀ p q : ℕ+, Nat.gcd p q = 9 → Nat.gcd (8 * p) (18 * q) ≥ 18) ∧
  (∃ p q : ℕ+, Nat.gcd p q = 9 ∧ Nat.gcd (8 * p) (18 * q) = 18) :=
by sorry

end smallest_gcd_multiple_l3181_318176


namespace rod_pieces_count_l3181_318185

/-- The length of the rod in meters -/
def rod_length : ℝ := 38.25

/-- The length of each piece in centimeters -/
def piece_length : ℝ := 85

/-- The number of pieces that can be cut from the rod -/
def num_pieces : ℕ := 45

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

theorem rod_pieces_count : 
  ⌊(rod_length * meters_to_cm) / piece_length⌋ = num_pieces := by
  sorry

end rod_pieces_count_l3181_318185


namespace solve_equation_l3181_318145

theorem solve_equation (C D : ℚ) 
  (eq1 : 5 * C + 3 * D - 4 = 47) 
  (eq2 : C = D + 2) : 
  C = 57 / 8 ∧ D = 41 / 8 := by
  sorry

end solve_equation_l3181_318145


namespace triangle_abc_equilateral_l3181_318165

theorem triangle_abc_equilateral 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 2 * a = b + c) 
  (h2 : Real.sin A ^ 2 = Real.sin B * Real.sin C) : 
  a = b ∧ b = c := by
  sorry

end triangle_abc_equilateral_l3181_318165


namespace rectangular_prism_sum_l3181_318166

/-- A rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The number of edges in a rectangular prism -/
def num_edges (p : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def num_corners (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (p : RectangularPrism) : ℕ := 6

/-- The theorem stating that the sum of edges, corners, and faces of any rectangular prism is 26 -/
theorem rectangular_prism_sum (p : RectangularPrism) :
  num_edges p + num_corners p + num_faces p = 26 := by
  sorry

#check rectangular_prism_sum

end rectangular_prism_sum_l3181_318166


namespace first_month_sale_is_3435_l3181_318135

/-- Calculates the sale in the first month given the sales for the next five months and the average sale --/
def calculate_first_month_sale (sales_2_to_5 : List ℕ) (sale_6 : ℕ) (average_sale : ℕ) : ℕ :=
  6 * average_sale - (sales_2_to_5.sum + sale_6)

/-- Theorem stating that the sale in the first month is 3435 given the specified conditions --/
theorem first_month_sale_is_3435 :
  let sales_2_to_5 := [3920, 3855, 4230, 3560]
  let sale_6 := 2000
  let average_sale := 3500
  calculate_first_month_sale sales_2_to_5 sale_6 average_sale = 3435 := by
  sorry

#eval calculate_first_month_sale [3920, 3855, 4230, 3560] 2000 3500

end first_month_sale_is_3435_l3181_318135


namespace second_number_in_sequence_l3181_318104

/-- The second number in the sequence of numbers that, when divided by 7, 9, and 11,
    always leaves a remainder of 5, given that 1398 - 22 = 1376 is the first such number. -/
theorem second_number_in_sequence (first_number : ℕ) (h1 : first_number = 1376) :
  ∃ (second_number : ℕ),
    second_number > first_number ∧
    second_number % 7 = 5 ∧
    second_number % 9 = 5 ∧
    second_number % 11 = 5 ∧
    ∀ (n : ℕ), first_number < n ∧ n < second_number →
      (n % 7 ≠ 5 ∨ n % 9 ≠ 5 ∨ n % 11 ≠ 5) :=
by sorry

end second_number_in_sequence_l3181_318104


namespace translated_circle_equation_l3181_318149

-- Define the points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (5, 8)

-- Define the translation vector u
def u : ℝ × ℝ := (2, -1)

-- Define the theorem
theorem translated_circle_equation :
  let diameter := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let radius := diameter / 2
  let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let new_center := (center.1 + u.1, center.2 + u.2)
  ∀ x y : ℝ, (x - new_center.1)^2 + (y - new_center.2)^2 = radius^2 :=
by sorry

end translated_circle_equation_l3181_318149


namespace bracket_calculation_l3181_318116

-- Define the single bracket operation
def single_bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Define the double bracket operation
def double_bracket (a b c d e f : ℚ) : ℚ := single_bracket (a + b) (d + e) (c + f)

-- State the theorem
theorem bracket_calculation :
  let result := single_bracket
    (double_bracket 10 20 30 40 30 70)
    (double_bracket 8 4 12 18 9 27)
    1
  result = 0.04 + 4/39 := by sorry

end bracket_calculation_l3181_318116


namespace tangent_line_equation_l3181_318186

/-- The equation of the tangent line to y = x^3 - 2x at (1, -1) is x - y - 2 = 0 -/
theorem tangent_line_equation (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f x = x^3 - 2*x →
  x₀ = 1 →
  y₀ = -1 →
  f x₀ = y₀ →
  (deriv f) x₀ = 1 →
  ∀ x y, (x - x₀) = (deriv f x₀) * (y - y₀) ↔ x - y - 2 = 0 :=
by sorry

end tangent_line_equation_l3181_318186


namespace highest_divisible_digit_l3181_318138

theorem highest_divisible_digit : 
  ∃ (a : ℕ), a ≤ 9 ∧ 
  (365 * 1000 + a * 100 + 16) % 8 = 0 ∧
  ∀ (b : ℕ), b ≤ 9 → b > a → (365 * 1000 + b * 100 + 16) % 8 ≠ 0 :=
by
  -- Proof goes here
  sorry

end highest_divisible_digit_l3181_318138


namespace exactly_one_head_and_two_heads_mutually_exclusive_but_not_complementary_l3181_318105

/-- Represents the outcome of a single coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing two coins -/
def TwoCoinsOutcome := CoinOutcome × CoinOutcome

/-- The event of exactly one head facing up -/
def exactlyOneHead (outcome : TwoCoinsOutcome) : Prop :=
  (outcome.1 = CoinOutcome.Heads ∧ outcome.2 = CoinOutcome.Tails) ∨
  (outcome.1 = CoinOutcome.Tails ∧ outcome.2 = CoinOutcome.Heads)

/-- The event of exactly two heads facing up -/
def exactlyTwoHeads (outcome : TwoCoinsOutcome) : Prop :=
  outcome.1 = CoinOutcome.Heads ∧ outcome.2 = CoinOutcome.Heads

/-- The sample space of all possible outcomes when tossing two coins -/
def sampleSpace : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Heads),
   (CoinOutcome.Heads, CoinOutcome.Tails),
   (CoinOutcome.Tails, CoinOutcome.Heads),
   (CoinOutcome.Tails, CoinOutcome.Tails)}

theorem exactly_one_head_and_two_heads_mutually_exclusive_but_not_complementary :
  (∀ (outcome : TwoCoinsOutcome), ¬(exactlyOneHead outcome ∧ exactlyTwoHeads outcome)) ∧
  (∃ (outcome : TwoCoinsOutcome), ¬exactlyOneHead outcome ∧ ¬exactlyTwoHeads outcome) :=
sorry

end exactly_one_head_and_two_heads_mutually_exclusive_but_not_complementary_l3181_318105


namespace zephyria_license_plates_l3181_318126

/-- The number of letters in the alphabet. -/
def num_letters : ℕ := 26

/-- The number of digits (0-9). -/
def num_digits : ℕ := 10

/-- The number of letters in a Zephyrian license plate. -/
def letters_in_plate : ℕ := 3

/-- The number of digits in a Zephyrian license plate. -/
def digits_in_plate : ℕ := 4

/-- The total number of valid license plates in Zephyria. -/
def total_license_plates : ℕ := num_letters ^ letters_in_plate * num_digits ^ digits_in_plate

theorem zephyria_license_plates :
  total_license_plates = 175760000 := by
  sorry

end zephyria_license_plates_l3181_318126


namespace binomial_coefficient_divisibility_l3181_318172

theorem binomial_coefficient_divisibility 
  (p : Nat) (α : Nat) (m : Nat) 
  (hp : Nat.Prime p) 
  (hp_odd : Odd p) 
  (hα : α ≥ 2) 
  (hm : m ≥ 2) : 
  ∃ k : Nat, Nat.choose (p^(α-2)) m = k * p^(α-m) := by
  sorry

end binomial_coefficient_divisibility_l3181_318172


namespace Z_in_first_quadrant_l3181_318192

def Z : ℂ := (5 + 4*Complex.I) + (-1 + 2*Complex.I)

theorem Z_in_first_quadrant : 
  Z.re > 0 ∧ Z.im > 0 := by sorry

end Z_in_first_quadrant_l3181_318192


namespace dice_probability_l3181_318152

def standard_die : Finset ℕ := Finset.range 6
def eight_sided_die : Finset ℕ := Finset.range 8

def prob_not_one (die : Finset ℕ) : ℚ :=
  (die.filter (· ≠ 1)).card / die.card

theorem dice_probability : 
  (prob_not_one standard_die)^2 * (prob_not_one eight_sided_die) = 175/288 := by
  sorry

end dice_probability_l3181_318152


namespace weighted_inequality_l3181_318111

theorem weighted_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  (a + 2*a*b + 2*a*c + b*c)^a * (b + 2*b*c + 2*b*a + c*a)^b * (c + 2*c*a + 2*c*b + a*b)^c ≤ 1 := by
  sorry

end weighted_inequality_l3181_318111


namespace triangle_with_altitudes_9_12_18_has_right_angle_l3181_318114

/-- A triangle with altitudes of lengths 9, 12, and 18 has a right angle as its largest angle. -/
theorem triangle_with_altitudes_9_12_18_has_right_angle :
  ∀ (a b c : ℝ) (α β γ : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c →
  α + β + γ = π →
  9 * a = 12 * b ∧ 12 * b = 18 * c →
  (∃ (h : ℝ), h * a = 9 ∧ h * b = 12 ∧ h * c = 18) →
  max α (max β γ) = π / 2 :=
by sorry

end triangle_with_altitudes_9_12_18_has_right_angle_l3181_318114


namespace exactly_fourteen_plus_signs_l3181_318170

/-- Represents a board with plus and minus signs -/
structure SignBoard where
  total_symbols : ℕ
  plus_signs : ℕ
  minus_signs : ℕ
  total_is_sum : total_symbols = plus_signs + minus_signs

/-- Predicate to check if any subset of size n contains at least one plus sign -/
def has_plus_in_subset (board : SignBoard) (n : ℕ) : Prop :=
  board.minus_signs < n

/-- Predicate to check if any subset of size n contains at least one minus sign -/
def has_minus_in_subset (board : SignBoard) (n : ℕ) : Prop :=
  board.plus_signs < n

/-- The main theorem to prove -/
theorem exactly_fourteen_plus_signs (board : SignBoard) 
  (h_total : board.total_symbols = 23)
  (h_plus_10 : has_plus_in_subset board 10)
  (h_minus_15 : has_minus_in_subset board 15) :
  board.plus_signs = 14 :=
sorry

end exactly_fourteen_plus_signs_l3181_318170


namespace relationship_a_x_l3181_318146

theorem relationship_a_x (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 + b^3 = 14*x^3) 
  (h3 : a + b = x) : 
  a = (Real.sqrt 165 - 3) / 6 * x ∨ a = -(Real.sqrt 165 + 3) / 6 * x := by
  sorry

end relationship_a_x_l3181_318146


namespace sum_a_b_equals_nine_l3181_318107

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (a b : ℝ) : Prop :=
  i * (a - i) = b - (2 * i) ^ 3

-- Theorem statement
theorem sum_a_b_equals_nine (a b : ℝ) (h : equation a b) : a + b = 9 := by
  sorry

end sum_a_b_equals_nine_l3181_318107


namespace product_of_roots_l3181_318127

theorem product_of_roots (x : ℝ) : (x - 1) * (x + 4) = 22 → ∃ y : ℝ, (x - 1) * (x + 4) = 22 ∧ (y - 1) * (y + 4) = 22 ∧ x * y = -26 := by
  sorry

end product_of_roots_l3181_318127


namespace large_square_area_l3181_318199

theorem large_square_area (s : ℝ) (S : ℝ) 
  (h1 : S = s + 20)
  (h2 : S^2 - s^2 = 880) :
  S^2 = 1024 := by
  sorry

end large_square_area_l3181_318199


namespace zero_point_existence_l3181_318131

def f (x : ℝ) : ℝ := -x^3 - 3*x + 5

theorem zero_point_existence :
  ∃ x₀ ∈ Set.Ioo 1 2, f x₀ = 0 :=
sorry

end zero_point_existence_l3181_318131


namespace susan_first_turn_l3181_318184

/-- The number of spaces on the board game --/
def total_spaces : ℕ := 48

/-- The number of spaces Susan moves on the first turn --/
def first_turn : ℕ := sorry

/-- The net movement on the second turn --/
def second_turn : ℤ := 2 - 5

/-- The movement on the third turn --/
def third_turn : ℕ := 6

/-- The remaining spaces to win after three turns --/
def remaining_spaces : ℕ := 37

/-- Theorem stating that Susan moved 8 spaces on the first turn --/
theorem susan_first_turn : first_turn = 8 := by sorry

end susan_first_turn_l3181_318184


namespace intersection_of_A_and_B_l3181_318123

def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x < 3}

theorem intersection_of_A_and_B :
  A ∩ B = {x | -1 < x ∧ x < 3} :=
by sorry

end intersection_of_A_and_B_l3181_318123


namespace keychain_manufacturing_cost_l3181_318130

theorem keychain_manufacturing_cost 
  (P : ℝ) -- Selling price
  (h1 : P > 0) -- Selling price is positive
  (h2 : P - 0.5 * P = 50) -- New manufacturing cost is $50
  : P - 0.4 * P = 60 := by
  sorry

end keychain_manufacturing_cost_l3181_318130


namespace abs_ratio_greater_than_one_l3181_318148

theorem abs_ratio_greater_than_one (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  |a| / |b| > 1 := by
  sorry

end abs_ratio_greater_than_one_l3181_318148


namespace smallest_divisible_by_8_13_14_l3181_318198

theorem smallest_divisible_by_8_13_14 : ∃ n : ℕ, n > 0 ∧ 
  8 ∣ n ∧ 13 ∣ n ∧ 14 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → 8 ∣ m → 13 ∣ m → 14 ∣ m → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_divisible_by_8_13_14_l3181_318198


namespace census_suitability_l3181_318156

-- Define the type for survey options
inductive SurveyOption
| A : SurveyOption  -- Favorite TV programs of middle school students
| B : SurveyOption  -- Printing errors on a certain exam paper
| C : SurveyOption  -- Survey on the service life of batteries
| D : SurveyOption  -- Internet usage of middle school students

-- Define what it means for a survey to be suitable for a census
def suitableForCensus (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.B => True
  | _ => False

-- Define the property of examining every item in a population
def examinesEveryItem (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.B => True
  | _ => False

-- Theorem statement
theorem census_suitability :
  ∀ s : SurveyOption, suitableForCensus s ↔ examinesEveryItem s :=
by sorry

end census_suitability_l3181_318156


namespace calculate_expression_solve_system_of_equations_l3181_318177

-- Part 1
theorem calculate_expression : (-2)^2 + (Real.sqrt 3 - Real.pi)^0 + abs (1 - Real.sqrt 3) = 4 + Real.sqrt 3 := by
  sorry

-- Part 2
theorem solve_system_of_equations :
  ∃ x y : ℝ, 2*x + y = 1 ∧ x - 2*y = 3 ∧ x = 1 ∧ y = -1 := by
  sorry

end calculate_expression_solve_system_of_equations_l3181_318177


namespace value_of_expression_l3181_318129

theorem value_of_expression (a b c d : ℝ) 
  (h1 : a - b = 3) 
  (h2 : c + d = 2) : 
  (a + c) - (b - d) = 5 := by
sorry

end value_of_expression_l3181_318129


namespace problem_solution_l3181_318159

theorem problem_solution (a b c d : ℝ) (h1 : a - b = -3) (h2 : c + d = 2) :
  (b + c) - (a - d) = 5 := by sorry

end problem_solution_l3181_318159


namespace parallel_lines_k_values_l3181_318169

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line -/
def l1 (k : ℝ) : Line :=
  { a := k - 3
    b := 4 - k
    c := 1 }

/-- The second line -/
def l2 (k : ℝ) : Line :=
  { a := 2 * (k - 3)
    b := -2
    c := 3 }

/-- Theorem: If l1 and l2 are parallel, then k is either 3 or 5 -/
theorem parallel_lines_k_values :
  ∀ k : ℝ, parallel (l1 k) (l2 k) → k = 3 ∨ k = 5 := by
  sorry

end parallel_lines_k_values_l3181_318169


namespace smallest_divisible_fraction_l3181_318119

def fraction1 : Rat := 6 / 7
def fraction2 : Rat := 5 / 14
def fraction3 : Rat := 10 / 21

def smallest_fraction : Rat := 1 / 42

theorem smallest_divisible_fraction :
  (∀ r : Rat, (fraction1 ∣ r ∧ fraction2 ∣ r ∧ fraction3 ∣ r) → smallest_fraction ≤ r) ∧
  (fraction1 ∣ smallest_fraction ∧ fraction2 ∣ smallest_fraction ∧ fraction3 ∣ smallest_fraction) :=
sorry

end smallest_divisible_fraction_l3181_318119


namespace condo_has_23_floors_l3181_318144

/-- Represents a condo development with regular and penthouse floors -/
structure CondoDevelopment where
  total_units : ℕ
  regular_units_per_floor : ℕ
  penthouse_units_per_floor : ℕ
  penthouse_floors : ℕ

/-- Calculates the total number of floors in a condo development -/
def total_floors (condo : CondoDevelopment) : ℕ :=
  let regular_floors := (condo.total_units - condo.penthouse_floors * condo.penthouse_units_per_floor) / condo.regular_units_per_floor
  regular_floors + condo.penthouse_floors

/-- Theorem stating that a condo development with the given specifications has 23 floors -/
theorem condo_has_23_floors :
  let condo := CondoDevelopment.mk 256 12 2 2
  total_floors condo = 23 := by
  sorry

end condo_has_23_floors_l3181_318144


namespace solve_for_a_l3181_318140

theorem solve_for_a (x : ℝ) (a : ℝ) (h1 : 2 * x - a - 5 = 0) (h2 : x = 3) : a = 1 := by
  sorry

end solve_for_a_l3181_318140


namespace normal_distribution_mean_l3181_318173

/-- 
Given a normal distribution with standard deviation σ,
if the value that is exactly k standard deviations less than the mean is x,
then the arithmetic mean μ of the distribution is x + k * σ.
-/
theorem normal_distribution_mean 
  (σ : ℝ) (k : ℝ) (x : ℝ) (μ : ℝ) 
  (hσ : σ = 1.5) 
  (hk : k = 2) 
  (hx : x = 11.5) 
  (h : x = μ - k * σ) : 
  μ = 14.5 := by
  sorry

#check normal_distribution_mean

end normal_distribution_mean_l3181_318173


namespace sequence_sum_l3181_318120

theorem sequence_sum (a b c d : ℕ+) : 
  (∃ r : ℚ, r > 1 ∧ b = a * r ∧ c = a * r^2) →  -- geometric progression
  (∃ k : ℤ, c - b = k ∧ d - c = k) →            -- arithmetic progression
  d = a + 40 →                                  -- difference between first and last term
  a + b + c + d = 104 := by
sorry

end sequence_sum_l3181_318120


namespace right_triangle_condition_l3181_318147

/-- If sin γ - cos α = cos β in a triangle, then the triangle is right-angled -/
theorem right_triangle_condition (α β γ : ℝ) (h_triangle : α + β + γ = Real.pi) 
  (h_condition : Real.sin γ - Real.cos α = Real.cos β) : 
  α = Real.pi / 2 ∨ β = Real.pi / 2 ∨ γ = Real.pi / 2 := by
  sorry

end right_triangle_condition_l3181_318147


namespace quadratic_function_properties_l3181_318103

/-- The quadratic function f(x) = ax² + mx + m - 1 -/
def f (a m x : ℝ) : ℝ := a * x^2 + m * x + m - 1

theorem quadratic_function_properties (a m : ℝ) (h_a : a ≠ 0) :
  /- Part 1: Number of zeros when f(-1) = 0 -/
  (f a m (-1) = 0 → (∃ x, f a m x = 0) ∧ (∃ x y, x ≠ y ∧ f a m x = 0 ∧ f a m y = 0)) ∧
  /- Part 2: Condition for always having two distinct zeros -/
  ((∀ m : ℝ, ∃ x y : ℝ, x ≠ y ∧ f a m x = 0 ∧ f a m y = 0) ↔ 0 < a ∧ a < 1) ∧
  /- Part 3: Existence of root between x₁ and x₂ -/
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a m x₁ ≠ f a m x₂ →
    ∃ x : ℝ, x₁ < x ∧ x < x₂ ∧ f a m x = (f a m x₁ + f a m x₂) / 2) :=
by sorry


end quadratic_function_properties_l3181_318103


namespace largest_prime_divisor_of_factorial_sum_l3181_318118

theorem largest_prime_divisor_of_factorial_sum : 
  ∃ p : ℕ, Prime p ∧ p ∣ (Nat.factorial 12 + Nat.factorial 13) ∧ 
  ∀ q : ℕ, Prime q → q ∣ (Nat.factorial 12 + Nat.factorial 13) → q ≤ p :=
by sorry

end largest_prime_divisor_of_factorial_sum_l3181_318118


namespace max_product_constrained_l3181_318150

theorem max_product_constrained (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_constraint : 3*x + 2*y = 12) :
  x * y ≤ 6 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3*x₀ + 2*y₀ = 12 ∧ x₀ * y₀ = 6 := by
  sorry

end max_product_constrained_l3181_318150


namespace count_valid_distributions_l3181_318182

/-- Represents an envelope containing two cards -/
def Envelope := Fin 6 × Fin 6

/-- Represents a valid distribution of cards into envelopes -/
def ValidDistribution := { d : Fin 3 → Envelope // 
  (∀ i j : Fin 3, i ≠ j → d i ≠ d j) ∧ 
  (∃ i : Fin 3, d i = ⟨1, 2⟩ ∨ d i = ⟨2, 1⟩) }

/-- The number of valid distributions -/
def numValidDistributions : ℕ := sorry

theorem count_valid_distributions : numValidDistributions = 18 := by sorry

end count_valid_distributions_l3181_318182


namespace x_minus_y_value_l3181_318151

theorem x_minus_y_value (x y : ℝ) 
  (h1 : |x| = 3) 
  (h2 : y^2 = 1/4) 
  (h3 : x + y < 0) : 
  x - y = -7/2 ∨ x - y = -5/2 := by
sorry

end x_minus_y_value_l3181_318151


namespace race_finish_orders_l3181_318102

theorem race_finish_orders (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end race_finish_orders_l3181_318102


namespace error_permutations_l3181_318197

/-- The number of incorrect permutations of the letters in "error" -/
def incorrect_permutations : ℕ :=
  Nat.factorial 5 / Nat.factorial 3 - 1

/-- The word "error" has 5 letters -/
def word_length : ℕ := 5

/-- The letter 'r' is repeated three times -/
def r_count : ℕ := 3

/-- The letters 'e' and 'o' appear once each -/
def unique_letters : ℕ := 2

theorem error_permutations :
  incorrect_permutations = (Nat.factorial word_length / Nat.factorial r_count) - 1 :=
by sorry

end error_permutations_l3181_318197


namespace list_property_l3181_318161

theorem list_property (S : ℝ) (n : ℝ) :
  let list_size : ℕ := 21
  let other_numbers_sum : ℝ := S - n
  let other_numbers_count : ℕ := list_size - 1
  let other_numbers_avg : ℝ := other_numbers_sum / other_numbers_count
  n = 4 * other_numbers_avg →
  n = S / 6 →
  other_numbers_count = 20 := by
  sorry

end list_property_l3181_318161


namespace total_soldiers_l3181_318191

theorem total_soldiers (n : ℕ) 
  (h1 : ∃ x y : ℕ, x + y = n ∧ y = x / 6)
  (h2 : ∃ x' y' : ℕ, x' + y' = n ∧ y' = x' / 7)
  (h3 : ∃ y y' : ℕ, y - y' = 2)
  (h4 : ∀ z : ℕ, z + n = n → z = 0) :
  n = 98 := by
  sorry

end total_soldiers_l3181_318191


namespace binary_110011_is_51_l3181_318115

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end binary_110011_is_51_l3181_318115


namespace total_time_circling_island_l3181_318106

-- Define the problem parameters
def time_per_round : ℕ := 30
def saturday_rounds : ℕ := 11
def sunday_rounds : ℕ := 15

-- State the theorem
theorem total_time_circling_island : 
  (saturday_rounds + sunday_rounds) * time_per_round = 780 := by
  sorry

end total_time_circling_island_l3181_318106


namespace prob_at_least_two_correct_value_l3181_318155

/-- The number of questions Jessica randomly guesses -/
def n : ℕ := 6

/-- The number of possible answers for each question -/
def m : ℕ := 3

/-- The probability of guessing a single question correctly -/
def p : ℚ := 1 / m

/-- The probability of guessing a single question incorrectly -/
def q : ℚ := 1 - p

/-- The probability of getting at least two correct answers out of n randomly guessed questions -/
def prob_at_least_two_correct : ℚ :=
  1 - (q ^ n + n * p * q ^ (n - 1))

theorem prob_at_least_two_correct_value : 
  prob_at_least_two_correct = 473 / 729 := by
  sorry

end prob_at_least_two_correct_value_l3181_318155


namespace opposite_of_negative_eleven_l3181_318132

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_negative_eleven : opposite (-11) = 11 := by
  sorry

end opposite_of_negative_eleven_l3181_318132


namespace arkansas_game_sales_l3181_318153

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts : ℕ := 172

/-- The number of t-shirts sold during the Texas Tech game -/
def texas_tech_shirts : ℕ := 186 - arkansas_shirts

/-- The revenue per t-shirt in dollars -/
def revenue_per_shirt : ℕ := 78

/-- The total number of t-shirts sold during both games -/
def total_shirts : ℕ := 186

/-- The revenue from the Texas Tech game in dollars -/
def texas_tech_revenue : ℕ := 1092

theorem arkansas_game_sales : 
  arkansas_shirts = 172 ∧ 
  texas_tech_shirts + arkansas_shirts = total_shirts ∧
  texas_tech_shirts * revenue_per_shirt = texas_tech_revenue :=
sorry

end arkansas_game_sales_l3181_318153


namespace johns_weekly_consumption_l3181_318175

/-- Represents John's daily beverage consumption --/
structure DailyConsumption where
  water : ℝ  -- in gallons
  milk : ℝ   -- in pints
  juice : ℝ  -- in fluid ounces

/-- Conversion factors --/
def gallon_to_quart : ℝ := 4
def pint_to_quart : ℝ := 0.5
def floz_to_quart : ℝ := 0.03125

/-- John's daily consumption --/
def johns_consumption : DailyConsumption := {
  water := 1.5,
  milk := 3,
  juice := 20
}

/-- Number of days in a week --/
def days_in_week : ℕ := 7

/-- Theorem stating John's weekly beverage consumption in quarts --/
theorem johns_weekly_consumption :
  (johns_consumption.water * gallon_to_quart +
   johns_consumption.milk * pint_to_quart +
   johns_consumption.juice * floz_to_quart) * days_in_week = 56.875 := by
  sorry

end johns_weekly_consumption_l3181_318175


namespace inequality_proof_l3181_318160

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  1/a + 1/b + 4/c + 16/d ≥ 64 / (a + b + c + d) := by
  sorry

end inequality_proof_l3181_318160


namespace sequence_product_l3181_318134

/-- Given that (-9, a, -1) is an arithmetic sequence and (-9, m, b, n, -1) is a geometric sequence,
    prove that ab = 5. -/
theorem sequence_product (a m b n : ℝ) : 
  ((-9 : ℝ) - a = a - (-1 : ℝ)) →  -- arithmetic sequence condition
  (m / (-9 : ℝ) = b / m) →         -- geometric sequence condition for first two terms
  (b / m = n / b) →                -- geometric sequence condition for middle terms
  (n / b = (-1 : ℝ) / n) →         -- geometric sequence condition for last two terms
  a * b = 5 := by
sorry

end sequence_product_l3181_318134


namespace money_split_l3181_318141

theorem money_split (total : ℝ) (share : ℝ) (n : ℕ) :
  n = 2 →
  share = 32.5 →
  n * share = total →
  total = 65 := by
sorry

end money_split_l3181_318141


namespace field_trip_girls_fraction_l3181_318195

theorem field_trip_girls_fraction (g : ℚ) (h1 : g > 0) : 
  let b := 2 * g
  let girls_on_trip := (4 / 5) * g
  let boys_on_trip := (3 / 4) * b
  let total_on_trip := girls_on_trip + boys_on_trip
  girls_on_trip / total_on_trip = 8 / 23 := by
sorry

end field_trip_girls_fraction_l3181_318195


namespace basic_computer_price_l3181_318121

/-- Proves that the price of a basic computer is $1500 given certain conditions. -/
theorem basic_computer_price (basic_price printer_price : ℕ) : 
  (basic_price + printer_price = 2500) →
  (printer_price = (basic_price + 500 + printer_price) / 3) →
  basic_price = 1500 := by
  sorry

#check basic_computer_price

end basic_computer_price_l3181_318121


namespace eight_divisors_l3181_318101

theorem eight_divisors (n : ℕ) : (Finset.card (Nat.divisors n) = 8) ↔ 
  (∃ p : ℕ, Nat.Prime p ∧ n = p^7) ∨ 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q^3) ∨ 
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ n = p * q * r) :=
by sorry

end eight_divisors_l3181_318101


namespace tangent_line_to_circle_l3181_318139

/-- The value of a for which a line with equation ρsin(θ+ π/3)=a is tangent to a circle with equation ρ = 2sinθ in the polar coordinate system -/
theorem tangent_line_to_circle (a : ℝ) : 
  (∃ θ ρ, ρ = 2 * Real.sin θ ∧ ρ * Real.sin (θ + π/3) = a ∧ 
   ∀ θ' ρ', ρ' = 2 * Real.sin θ' → ρ' * Real.sin (θ' + π/3) ≠ a ∨ (θ' = θ ∧ ρ' = ρ)) →
  a = 3/2 ∨ a = -1/2 :=
by sorry

end tangent_line_to_circle_l3181_318139


namespace simplify_nested_roots_l3181_318167

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b ^ 16) ^ (1 / 8)) ^ (1 / 4)) ^ 3 * (((b ^ 16) ^ (1 / 4)) ^ (1 / 8)) ^ 3 = b ^ 3 := by
  sorry

end simplify_nested_roots_l3181_318167


namespace direct_inverse_variation_l3181_318137

theorem direct_inverse_variation (k : ℝ) : 
  (∃ (R S T : ℝ), R = k * S / T ∧ R = 2 ∧ S = 6 ∧ T = 3) →
  (∀ (R S T : ℝ), R = k * S / T → R = 8 ∧ T = 2 → S = 16) :=
by sorry

end direct_inverse_variation_l3181_318137


namespace factorization_equality_l3181_318189

theorem factorization_equality (a b : ℝ) : 5 * a^2 * b - 20 * b^3 = 5 * b * (a + 2*b) * (a - 2*b) := by
  sorry

end factorization_equality_l3181_318189


namespace middle_number_proof_l3181_318110

theorem middle_number_proof (x y : ℝ) : 
  (3*x)^2 + (2*x)^2 + (5*x)^2 = 1862 →
  3*x + 2*x + 5*x + 4*y + 7*y = 155 →
  2*x = 14 :=
by
  sorry

end middle_number_proof_l3181_318110


namespace expression_simplification_l3181_318122

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 2) : 
  ((m^2 - 9) / (m^2 - 6*m + 9) - 3 / (m - 3)) / (m^2 / (m - 3)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l3181_318122


namespace valid_grid_exists_l3181_318128

/-- Represents a 6x6 grid of integers -/
def Grid := Matrix (Fin 6) (Fin 6) ℕ

/-- Checks if a given row in the grid contains distinct numbers from 1 to 6 -/
def validRow (g : Grid) (row : Fin 6) : Prop :=
  (Set.range fun col => g row col) = {1, 2, 3, 4, 5, 6}

/-- Checks if a given column in the grid contains distinct numbers from 1 to 6 -/
def validColumn (g : Grid) (col : Fin 6) : Prop :=
  (Set.range fun row => g row col) = {1, 2, 3, 4, 5, 6}

/-- Checks if the grid satisfies all row and column constraints -/
def validGrid (g : Grid) : Prop :=
  (∀ row, validRow g row) ∧ (∀ col, validColumn g col)

/-- The main theorem stating the existence of a valid grid with the given properties -/
theorem valid_grid_exists : ∃ (g : Grid), 
  validGrid g ∧ 
  g 1 1 = 5 ∧
  g 2 3 = 6 ∧
  g 5 0 = 4 ∧
  g 5 1 = 6 ∧
  g 5 2 = 1 ∧
  g 5 3 = 2 ∧
  g 5 4 = 3 :=
sorry


end valid_grid_exists_l3181_318128


namespace circle_m_equation_l3181_318100

/-- A circle M passing through two points with its center on a given line -/
structure CircleM where
  -- Circle M passes through these two points
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  -- The center of circle M lies on this line
  center_line : ℝ → ℝ → ℝ
  -- Conditions from the problem
  h1 : point1 = (0, 2)
  h2 : point2 = (0, 4)
  h3 : ∀ x y, center_line x y = 2*x - y - 1

/-- The equation of circle M -/
def circle_equation (c : CircleM) (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 5

/-- Theorem stating that the given conditions imply the circle equation -/
theorem circle_m_equation (c : CircleM) :
  ∀ x y, circle_equation c x y :=
sorry

end circle_m_equation_l3181_318100


namespace repeating_decimal_sum_l3181_318183

/-- Represents a repeating decimal with a single digit repeating -/
def repeating_decimal (n : ℕ) : ℚ := n / 9

theorem repeating_decimal_sum : 
  2 * (repeating_decimal 8 - repeating_decimal 2 + repeating_decimal 4) = 20 / 9 := by
  sorry

end repeating_decimal_sum_l3181_318183


namespace sum_vertices_is_nine_l3181_318190

/-- The number of vertices in a rectangle --/
def rectangle_vertices : ℕ := 4

/-- The number of vertices in a pentagon --/
def pentagon_vertices : ℕ := 5

/-- The sum of vertices of a rectangle and a pentagon --/
def sum_vertices : ℕ := rectangle_vertices + pentagon_vertices

theorem sum_vertices_is_nine : sum_vertices = 9 := by
  sorry

end sum_vertices_is_nine_l3181_318190


namespace floor_plus_x_eq_seventeen_fourths_l3181_318168

theorem floor_plus_x_eq_seventeen_fourths :
  ∃ (x : ℚ), (⌊x⌋ : ℚ) + x = 17 / 4 ∧ x = 9 / 4 := by
  sorry

end floor_plus_x_eq_seventeen_fourths_l3181_318168


namespace min_value_sum_product_l3181_318108

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 36 / 5 := by
  sorry

end min_value_sum_product_l3181_318108


namespace tara_bank_balance_l3181_318162

/-- Calculates the balance after one year given an initial amount and an annual interest rate. -/
def balance_after_one_year (initial_amount : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_amount * (1 + interest_rate)

/-- Theorem stating that with an initial amount of $90 and a 10% annual interest rate, 
    the balance after one year will be $99. -/
theorem tara_bank_balance : 
  balance_after_one_year 90 0.1 = 99 := by
  sorry

end tara_bank_balance_l3181_318162


namespace T_forms_three_lines_closed_region_l3181_318180

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    (4 ≤ x - 1 ∧ 4 ≤ y + 3 ∧ y + 3 < x - 1) ∨
    (4 ≤ x - 1 ∧ 4 ≤ y + 3 ∧ x - 1 < y + 3) ∨
    (4 ≤ x - 1 ∧ y + 3 < 4 ∧ y + 3 < x - 1) ∨
    (x - 1 < 4 ∧ 4 ≤ y + 3 ∧ x - 1 < y + 3) ∨
    (x - 1 ≤ y + 3 ∧ 4 < x - 1 ∧ 4 < y + 3)}

-- Define the three lines
def line1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 5 ∧ p.2 ≤ 1}
def line2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 1 ∧ p.1 ≤ 5}
def line3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 - 4 ∧ p.1 ≥ 5}

-- Theorem statement
theorem T_forms_three_lines_closed_region :
  ∃ (point : ℝ × ℝ), 
    point ∈ T ∧
    point ∈ line1 ∧ point ∈ line2 ∧ point ∈ line3 ∧
    T = line1 ∪ line2 ∪ line3 :=
sorry


end T_forms_three_lines_closed_region_l3181_318180


namespace nestedRadical_eq_six_l3181_318157

/-- The value of the infinite nested radical sqrt(18 + sqrt(18 + sqrt(18 + ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (18 + Real.sqrt (18 + Real.sqrt (18 + Real.sqrt (18 + Real.sqrt 18))))

/-- Theorem stating that the value of the nested radical is 6 -/
theorem nestedRadical_eq_six : nestedRadical = 6 := by
  sorry

end nestedRadical_eq_six_l3181_318157


namespace perfect_squares_divisibility_l3181_318124

theorem perfect_squares_divisibility (a b : ℕ+) :
  (∃ (S : Set (ℕ+ × ℕ+)), Set.Infinite S ∧
    ∀ (p : ℕ+ × ℕ+), p ∈ S →
      ∃ (k l : ℕ+), (p.1.val ^ 2 + a.val * p.2.val + b.val = k.val ^ 2) ∧
                    (p.2.val ^ 2 + a.val * p.1.val + b.val = l.val ^ 2)) →
  a.val ∣ (2 * b.val) :=
by sorry

end perfect_squares_divisibility_l3181_318124


namespace distance_is_95_over_17_l3181_318136

def point : ℝ × ℝ × ℝ := (2, 4, 5)
def line_point : ℝ × ℝ × ℝ := (5, 8, 9)
def line_direction : ℝ × ℝ × ℝ := (4, 3, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_95_over_17 : 
  distance_to_line point line_point line_direction = 95 / 17 := by
  sorry

end distance_is_95_over_17_l3181_318136


namespace sweater_a_markup_sweater_b_markup_l3181_318125

/-- Calculates the final price after applying a markup and two discounts -/
def final_price (wholesale : ℝ) (markup discount1 discount2 : ℝ) : ℝ :=
  wholesale * (1 + markup) * (1 - discount1) * (1 - discount2)

/-- Theorem for Sweater A -/
theorem sweater_a_markup (wholesale : ℝ) :
  final_price wholesale 3 0.2 0.5 = wholesale * 1.6 := by sorry

/-- Theorem for Sweater B -/
theorem sweater_b_markup (wholesale : ℝ) :
  ∃ ε > 0, ε < 0.0001 ∧ 
  |final_price wholesale 3.60606 0.25 0.45 - wholesale * 1.9| < ε := by sorry

end sweater_a_markup_sweater_b_markup_l3181_318125


namespace imaginary_power_sum_l3181_318187

theorem imaginary_power_sum (i : ℂ) (hi : i^2 = -1) :
  i^13 + i^18 + i^23 + i^28 + i^33 + i^38 = 0 := by sorry

end imaginary_power_sum_l3181_318187


namespace intersection_M_N_l3181_318113

def M : Set ℝ := {x | |x| ≤ 1}
def N : Set ℝ := {x | x^2 - x < 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_M_N_l3181_318113


namespace intersection_M_N_l3181_318181

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (1 - x) < 0}
def N : Set ℝ := {x | x^2 ≤ 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by
  sorry

end intersection_M_N_l3181_318181


namespace expression_evaluation_l3181_318164

theorem expression_evaluation : 3^(0^(2^5)) + ((3^0)^2)^5 = 2 := by
  sorry

end expression_evaluation_l3181_318164


namespace m_range_l3181_318158

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + (m - 1) * x

-- State the theorem
theorem m_range :
  (∀ (x : ℝ), x^2 + 4*x - m ≥ 0) ∧
  (∀ (x y : ℝ), x < y → x ≤ -3 → y ≤ -3 → f m x ≤ f m y) →
  m ∈ Set.Icc (-5 : ℝ) (-4 : ℝ) :=
sorry

end m_range_l3181_318158


namespace complex_power_sum_l3181_318109

theorem complex_power_sum (i : ℂ) (h1 : i^2 = -1) (h2 : i^4 = 1) :
  i^15 + i^22 + i^29 + i^36 + i^43 = -i := by
  sorry

end complex_power_sum_l3181_318109


namespace joan_balloons_l3181_318163

theorem joan_balloons (initial_balloons : ℕ) (lost_balloons : ℕ) (remaining_balloons : ℕ) : 
  initial_balloons = 9 → lost_balloons = 2 → remaining_balloons = initial_balloons - lost_balloons →
  remaining_balloons = 7 := by
  sorry

end joan_balloons_l3181_318163


namespace ponderosa_pine_price_l3181_318142

/-- The price of each ponderosa pine tree, given the total number of trees,
    number of Douglas fir trees, price of each Douglas fir, and total amount paid. -/
theorem ponderosa_pine_price
  (total_trees : ℕ)
  (douglas_fir_trees : ℕ)
  (douglas_fir_price : ℕ)
  (total_amount : ℕ)
  (h1 : total_trees = 850)
  (h2 : douglas_fir_trees = 350)
  (h3 : douglas_fir_price = 300)
  (h4 : total_amount = 217500) :
  (total_amount - douglas_fir_trees * douglas_fir_price) / (total_trees - douglas_fir_trees) = 225 := by
sorry


end ponderosa_pine_price_l3181_318142


namespace horner_method_v₂_l3181_318117

def f (x : ℝ) : ℝ := x^5 + x^4 + 2*x^3 + 3*x^2 + 4*x + 1

def horner_v₀ (x : ℝ) : ℝ := 1

def horner_v₁ (x : ℝ) : ℝ := horner_v₀ x * x + 4

def horner_v₂ (x : ℝ) : ℝ := horner_v₁ x * x + 3

theorem horner_method_v₂ : horner_v₂ 2 = 15 := by sorry

end horner_method_v₂_l3181_318117


namespace race_length_is_90_l3181_318196

/-- The race between Nicky and Cristina -/
structure Race where
  head_start : ℝ
  cristina_speed : ℝ
  nicky_speed : ℝ
  catch_up_time : ℝ

/-- Calculate the length of the race -/
def race_length (r : Race) : ℝ :=
  r.nicky_speed * r.catch_up_time

/-- Theorem stating that the race length is 90 meters -/
theorem race_length_is_90 (r : Race)
  (h1 : r.head_start = 12)
  (h2 : r.cristina_speed = 5)
  (h3 : r.nicky_speed = 3)
  (h4 : r.catch_up_time = 30) :
  race_length r = 90 := by
  sorry

#check race_length_is_90

end race_length_is_90_l3181_318196


namespace triangle_angle_measure_l3181_318174

theorem triangle_angle_measure (A B C : ℝ) (h1 : A = 3 * Real.pi / 4) (h2 : C > 0) (h3 : C < Real.pi / 4) (h4 : Real.sin C = 1 / 2) : C = Real.pi / 6 := by
  sorry

end triangle_angle_measure_l3181_318174


namespace hunting_season_quarter_year_l3181_318154

/-- Represents the hunting scenario -/
structure HuntingScenario where
  hunts_per_month : ℕ
  deers_per_hunt : ℕ
  deer_weight : ℕ
  kept_fraction : ℚ
  kept_weight : ℕ

/-- Calculates the fraction of the year the hunting season lasts -/
def hunting_season_fraction (scenario : HuntingScenario) : ℚ :=
  let total_catch := scenario.kept_weight / scenario.kept_fraction
  let catch_per_hunt := scenario.deers_per_hunt * scenario.deer_weight
  let hunts_per_year := total_catch / catch_per_hunt
  let months_of_hunting := hunts_per_year / scenario.hunts_per_month
  months_of_hunting / 12

/-- Theorem stating that for the given scenario, the hunting season lasts 1/4 of the year -/
theorem hunting_season_quarter_year (scenario : HuntingScenario) 
  (h1 : scenario.hunts_per_month = 6)
  (h2 : scenario.deers_per_hunt = 2)
  (h3 : scenario.deer_weight = 600)
  (h4 : scenario.kept_fraction = 1/2)
  (h5 : scenario.kept_weight = 10800) :
  hunting_season_fraction scenario = 1/4 := by
  sorry


end hunting_season_quarter_year_l3181_318154


namespace ball_max_height_l3181_318112

/-- The height of the ball as a function of time -/
def h (t : ℝ) : ℝ := -16 * t^2 + 80 * t + 35

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 135 := by
  sorry

end ball_max_height_l3181_318112


namespace sphere_surface_area_l3181_318193

/-- The surface area of a sphere with radius 14 meters is 4 * π * 14^2 square meters. -/
theorem sphere_surface_area :
  let r : ℝ := 14
  4 * Real.pi * r^2 = 4 * Real.pi * 14^2 := by sorry

end sphere_surface_area_l3181_318193


namespace number_percentage_equality_l3181_318188

theorem number_percentage_equality (x : ℚ) : 
  (35 / 100) * x = (15 / 100) * 40 → x = 17 + 1 / 7 := by
  sorry

end number_percentage_equality_l3181_318188


namespace cone_axial_angle_when_max_section_twice_axial_l3181_318194

/-- Represents a right circular cone -/
structure RightCircularCone where
  vertex : Point
  axialAngle : ℝ

/-- Represents a cross-section of a cone -/
structure ConeSection where
  cone : RightCircularCone
  angle : ℝ

/-- The area of a cone section -/
def sectionArea (s : ConeSection) : ℝ := sorry

/-- The maximum area cross-section of a cone -/
def maxSectionArea (c : RightCircularCone) : ℝ := sorry

/-- The axial cross-section of a cone -/
def axialSection (c : RightCircularCone) : ConeSection := sorry

theorem cone_axial_angle_when_max_section_twice_axial 
  (c : RightCircularCone) :
  maxSectionArea c = 2 * sectionArea (axialSection c) →
  c.axialAngle = 120 * π / 180 := by sorry

end cone_axial_angle_when_max_section_twice_axial_l3181_318194


namespace isosceles_triangle_larger_angle_l3181_318171

/-- The measure of a right angle in degrees -/
def right_angle : ℝ := 90

/-- An isosceles triangle with one angle 20% smaller than a right angle -/
structure IsoscelesTriangle where
  /-- The measure of the smallest angle in degrees -/
  small_angle : ℝ
  /-- The measure of one of the two equal larger angles in degrees -/
  large_angle : ℝ
  /-- The triangle is isosceles with two equal larger angles -/
  isosceles : large_angle = large_angle
  /-- The small angle is 20% smaller than a right angle -/
  small_angle_def : small_angle = right_angle * (1 - 0.2)
  /-- The sum of all angles in the triangle is 180° -/
  angle_sum : small_angle + 2 * large_angle = 180

/-- Theorem: In an isosceles triangle where one angle is 20% smaller than a right angle,
    each of the two equal larger angles measures 54° -/
theorem isosceles_triangle_larger_angle (t : IsoscelesTriangle) : t.large_angle = 54 := by
  sorry

end isosceles_triangle_larger_angle_l3181_318171
