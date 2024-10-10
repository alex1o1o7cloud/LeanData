import Mathlib

namespace probability_of_purple_marble_l3502_350284

theorem probability_of_purple_marble (blue_prob green_prob purple_prob : ℝ) :
  blue_prob = 0.25 →
  green_prob = 0.35 →
  blue_prob + green_prob + purple_prob = 1 →
  purple_prob = 0.4 := by
  sorry

end probability_of_purple_marble_l3502_350284


namespace third_term_of_sequence_l3502_350227

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem third_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 21 = 12 →
  arithmetic_sequence a₁ d 22 = 15 →
  arithmetic_sequence a₁ d 3 = -42 :=
by sorry

end third_term_of_sequence_l3502_350227


namespace exponent_division_l3502_350249

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end exponent_division_l3502_350249


namespace greatest_sum_consecutive_integers_l3502_350250

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 200) → (∀ m : ℕ, m > n → m * (m + 1) ≥ 200) → n + (n + 1) = 27 := by
  sorry

end greatest_sum_consecutive_integers_l3502_350250


namespace hockey_games_per_month_l3502_350244

/-- Proves that the number of hockey games played each month is 13,
    given that there are 182 hockey games in a 14-month season. -/
theorem hockey_games_per_month :
  let total_games : ℕ := 182
  let season_months : ℕ := 14
  let games_per_month : ℕ := total_games / season_months
  games_per_month = 13 :=
by
  sorry

end hockey_games_per_month_l3502_350244


namespace abs_z_squared_l3502_350255

-- Define the complex number z
variable (z : ℂ)

-- Define the condition z + |z| = 3 + 12i
def condition (z : ℂ) : Prop := z + Complex.abs z = 3 + 12 * Complex.I

-- Theorem statement
theorem abs_z_squared (h : condition z) : Complex.abs z ^ 2 = 650.25 := by
  sorry

end abs_z_squared_l3502_350255


namespace chess_tournament_participants_l3502_350277

theorem chess_tournament_participants (x : ℕ) (y : ℕ) : 
  (2 * y + 8 = (x + 2) * (x + 1) / 2) →
  (x * y + 8 = (x + 2) * (x + 1) / 2) →
  x = 7 := by
sorry

end chess_tournament_participants_l3502_350277


namespace no_half_rectangle_exists_l3502_350281

theorem no_half_rectangle_exists (a b : ℝ) (h : 0 < a ∧ a < b) :
  ¬ ∃ (x y : ℝ), 
    x < a / 2 ∧ 
    y < a / 2 ∧ 
    2 * (x + y) = a + b ∧ 
    x * y = a * b / 2 :=
by sorry

end no_half_rectangle_exists_l3502_350281


namespace polar_to_rectangular_conversion_l3502_350270

theorem polar_to_rectangular_conversion :
  let r : ℝ := 7
  let θ : ℝ := Real.pi / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2) := by
  sorry

end polar_to_rectangular_conversion_l3502_350270


namespace expression_equality_l3502_350231

theorem expression_equality : (2^1501 + 5^1500)^2 - (2^1501 - 5^1500)^2 = 8 * 10^1500 := by
  sorry

end expression_equality_l3502_350231


namespace quadratic_root_zero_l3502_350294

theorem quadratic_root_zero (a : ℝ) :
  (∃ x : ℝ, 2 * x^2 + x + a^2 - 1 = 0 ∧ x = 0) →
  (a = 1 ∨ a = -1) := by
  sorry

end quadratic_root_zero_l3502_350294


namespace max_difference_theorem_l3502_350239

/-- The maximum difference between the sum of ball numbers for two people --/
def maxDifference : ℕ := 9644

/-- The total number of balls --/
def totalBalls : ℕ := 200

/-- The starting number of the balls --/
def startNumber : ℕ := 101

/-- The ending number of the balls --/
def endNumber : ℕ := 300

/-- The number of balls each person takes --/
def ballsPerPerson : ℕ := 100

/-- The ball number that person A takes --/
def ballA : ℕ := 102

/-- The ball number that person B takes --/
def ballB : ℕ := 280

theorem max_difference_theorem :
  ∀ (sumA sumB : ℕ),
  sumA ≤ (startNumber + endNumber) * ballsPerPerson / 2 - (ballB - ballA) →
  sumB ≥ (startNumber + endNumber - totalBalls + 1) * ballsPerPerson / 2 + (ballB - ballA) →
  sumA - sumB ≤ maxDifference :=
sorry

end max_difference_theorem_l3502_350239


namespace no_solution_cubic_system_l3502_350285

theorem no_solution_cubic_system (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ¬∃ x : ℝ, (x^3 - a*x^2 + b^3 = 0) ∧ (x^3 - b*x^2 + c^3 = 0) ∧ (x^3 - c*x^2 + a^3 = 0) :=
by sorry

end no_solution_cubic_system_l3502_350285


namespace trapezoid_area_l3502_350245

/-- The area of a trapezoid given the areas of triangles formed by its diagonals -/
theorem trapezoid_area (S₁ S₂ : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) : 
  ∃ (A : ℝ), A = (Real.sqrt S₁ + Real.sqrt S₂)^2 ∧ A > 0 := by
  sorry

end trapezoid_area_l3502_350245


namespace hundred_power_ten_as_sum_of_tens_l3502_350280

theorem hundred_power_ten_as_sum_of_tens (n : ℕ) : (100 ^ 10 : ℕ) = n * 10 → n = 10 ^ 19 := by
  sorry

end hundred_power_ten_as_sum_of_tens_l3502_350280


namespace straight_line_angle_value_l3502_350274

/-- The sum of angles in a straight line is 180 degrees -/
def straight_line_angle_sum : ℝ := 180

/-- The angles along the straight line ABC -/
def angle1 (x : ℝ) : ℝ := x
def angle2 : ℝ := 21
def angle3 : ℝ := 21
def angle4 (x : ℝ) : ℝ := 2 * x
def angle5 : ℝ := 57

/-- Theorem: Given a straight line ABC with angles x°, 21°, 21°, 2x°, and 57°, the value of x is 27° -/
theorem straight_line_angle_value :
  ∀ x : ℝ, 
  angle1 x + angle2 + angle3 + angle4 x + angle5 = straight_line_angle_sum → 
  x = 27 := by
sorry


end straight_line_angle_value_l3502_350274


namespace train_length_l3502_350228

/-- The length of a train given its speed and time to pass a point -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 216) (h2 : time = 6) :
  speed * (5 / 18) * time = 360 :=
sorry

end train_length_l3502_350228


namespace stock_price_change_l3502_350242

/-- Calculates the net percentage change in stock price over three years -/
def netPercentageChange (year1Change : Real) (year2Change : Real) (year3Change : Real) : Real :=
  let price1 := 1 + year1Change
  let price2 := price1 * (1 + year2Change)
  let price3 := price2 * (1 + year3Change)
  (price3 - 1) * 100

/-- Theorem stating the net percentage change for the given scenario -/
theorem stock_price_change : 
  ∀ (ε : Real), ε > 0 → 
  |netPercentageChange (-0.08) 0.10 0.06 - 7.272| < ε :=
sorry

end stock_price_change_l3502_350242


namespace expected_sales_theorem_l3502_350221

/-- Represents the number of vehicles sold for each type -/
structure VehicleSales where
  sports_cars : ℕ
  sedans : ℕ
  trucks : ℕ

/-- The ratio of vehicle sales -/
def sales_ratio : VehicleSales :=
  { sports_cars := 3
    sedans := 5
    trucks := 4 }

/-- The expected number of sports cars to be sold -/
def expected_sports_cars : ℕ := 36

/-- Calculates the expected sales based on the ratio and expected sports car sales -/
def calculate_expected_sales (ratio : VehicleSales) (sports_cars : ℕ) : VehicleSales :=
  { sports_cars := sports_cars
    sedans := (sports_cars * ratio.sedans) / ratio.sports_cars
    trucks := (sports_cars * ratio.trucks) / ratio.sports_cars }

theorem expected_sales_theorem :
  let expected_sales := calculate_expected_sales sales_ratio expected_sports_cars
  expected_sales.sedans = 60 ∧ expected_sales.trucks = 48 := by
  sorry

end expected_sales_theorem_l3502_350221


namespace max_min_sum_absolute_value_l3502_350279

theorem max_min_sum_absolute_value (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0)
  (h2 : x + y - 1 ≥ 0)
  (h3 : 3 * x - y - 3 ≤ 0) :
  ∃ (z_max z_min : ℝ),
    (∀ (x' y' : ℝ), 
      x' - y' + 1 ≥ 0 → 
      x' + y' - 1 ≥ 0 → 
      3 * x' - y' - 3 ≤ 0 → 
      |x' - 4 * y' + 1| ≤ z_max ∧ 
      |x' - 4 * y' + 1| ≥ z_min) ∧
    z_max + z_min = 11 / Real.sqrt 17 :=
by sorry

end max_min_sum_absolute_value_l3502_350279


namespace probability_13_11_l3502_350282

/-- Represents a table tennis player -/
inductive Player : Type
| MaLong : Player
| FanZhendong : Player

/-- The probability of a player scoring when serving -/
def scoreProbability (server : Player) : ℚ :=
  match server with
  | Player.MaLong => 2/3
  | Player.FanZhendong => 1/2

/-- The probability of a player scoring when receiving -/
def receiveProbability (receiver : Player) : ℚ :=
  match receiver with
  | Player.MaLong => 1/2
  | Player.FanZhendong => 1/3

/-- Theorem stating the probability of reaching 13:11 score -/
theorem probability_13_11 :
  let initialServer := Player.MaLong
  let prob13_11 := (scoreProbability initialServer * receiveProbability Player.FanZhendong * scoreProbability initialServer * receiveProbability Player.FanZhendong) +
                   (receiveProbability Player.FanZhendong * scoreProbability Player.FanZhendong * scoreProbability initialServer * receiveProbability Player.FanZhendong) +
                   (receiveProbability Player.FanZhendong * scoreProbability Player.FanZhendong * receiveProbability initialServer * scoreProbability Player.FanZhendong) +
                   (scoreProbability initialServer * receiveProbability Player.FanZhendong * receiveProbability initialServer * scoreProbability Player.FanZhendong)
  prob13_11 = 1/4 := by
  sorry

end probability_13_11_l3502_350282


namespace triangle_angle_relation_l3502_350247

theorem triangle_angle_relation (A B C C₁ C₂ : ℝ) : 
  B = 2 * A →
  C + A + B = Real.pi →
  C₁ + A = Real.pi / 2 →
  C₂ + B = Real.pi / 2 →
  C = C₁ + C₂ →
  C₁ - C₂ = A :=
by sorry

end triangle_angle_relation_l3502_350247


namespace quadratic_root_square_condition_l3502_350201

theorem quadratic_root_square_condition (p q r : ℝ) (α β : ℝ) : 
  (p * α^2 + q * α + r = 0) →  -- α is a root of the quadratic equation
  (p * β^2 + q * β + r = 0) →  -- β is a root of the quadratic equation
  (β = α^2) →                  -- one root is the square of the other
  (p - 4*q ≥ 0) :=             -- the relationship between coefficients
by sorry

end quadratic_root_square_condition_l3502_350201


namespace trig_identity_l3502_350251

theorem trig_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.cos (x + y) =
  Real.sin x ^ 2 + Real.sin y ^ 2 := by
  sorry

end trig_identity_l3502_350251


namespace arithmetic_sequence_problem_l3502_350252

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Calculates the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + seq.diff * (n - 1)

theorem arithmetic_sequence_problem :
  ∃ (row col1 col2 : ArithmeticSequence),
    row.first = 15 ∧
    row.nthTerm 4 = 2 ∧
    col1.nthTerm 2 = 14 ∧
    col1.nthTerm 3 = 10 ∧
    col2.nthTerm 5 = -21 ∧
    col2.first = -13.5 := by
  sorry

end arithmetic_sequence_problem_l3502_350252


namespace max_min_sum_difference_l3502_350275

def three_digit_integer (a b c : ℕ) : Prop :=
  100 ≤ a * 100 + b * 10 + c ∧ a * 100 + b * 10 + c < 1000

def all_different (a b c d e f g h i : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i

theorem max_min_sum_difference :
  ∀ (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℕ),
  three_digit_integer a₁ b₁ c₁ →
  three_digit_integer a₂ b₂ c₂ →
  three_digit_integer a₃ b₃ c₃ →
  all_different a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ →
  (∀ (x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ : ℕ),
    three_digit_integer x₁ y₁ z₁ →
    three_digit_integer x₂ y₂ z₂ →
    three_digit_integer x₃ y₃ z₃ →
    all_different x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ →
    (a₁ * 100 + b₁ * 10 + c₁) + (a₂ * 100 + b₂ * 10 + c₂) + (a₃ * 100 + b₃ * 10 + c₃) ≥
    (x₁ * 100 + y₁ * 10 + z₁) + (x₂ * 100 + y₂ * 10 + z₂) + (x₃ * 100 + y₃ * 10 + z₃)) →
  (∀ (p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ : ℕ),
    three_digit_integer p₁ q₁ r₁ →
    three_digit_integer p₂ q₂ r₂ →
    three_digit_integer p₃ q₃ r₃ →
    all_different p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ →
    (p₁ * 100 + q₁ * 10 + r₁) + (p₂ * 100 + q₂ * 10 + r₂) + (p₃ * 100 + q₃ * 10 + r₃) ≥
    (a₁ * 100 + b₁ * 10 + c₁) + (a₂ * 100 + b₂ * 10 + c₂) + (a₃ * 100 + b₃ * 10 + c₃)) →
  ((a₁ * 100 + b₁ * 10 + c₁) + (a₂ * 100 + b₂ * 10 + c₂) + (a₃ * 100 + b₃ * 10 + c₃)) -
  ((p₁ * 100 + q₁ * 10 + r₁) + (p₂ * 100 + q₂ * 10 + r₂) + (p₃ * 100 + q₃ * 10 + r₃)) = 1845 :=
by sorry

end max_min_sum_difference_l3502_350275


namespace power_equality_implies_exponent_l3502_350291

theorem power_equality_implies_exponent (p : ℕ) : 16^6 = 4^p → p = 12 := by
  sorry

end power_equality_implies_exponent_l3502_350291


namespace percentage_of_older_female_students_l3502_350264

/-- Represents the percentage of female students who are 25 years old or older -/
def P : ℝ := 30

theorem percentage_of_older_female_students :
  let total_students : ℝ := 100
  let male_percentage : ℝ := 40
  let female_percentage : ℝ := 100 - male_percentage
  let older_male_percentage : ℝ := 40
  let younger_probability : ℝ := 0.66
  
  (male_percentage / 100 * (100 - older_male_percentage) / 100 +
   female_percentage / 100 * (100 - P) / 100) * total_students = younger_probability * total_students :=
by sorry

end percentage_of_older_female_students_l3502_350264


namespace simplify_radical_expression_l3502_350207

theorem simplify_radical_expression :
  Real.sqrt 18 - Real.sqrt 50 + 3 * Real.sqrt (1/2) = -Real.sqrt 2 / 2 := by
  sorry

end simplify_radical_expression_l3502_350207


namespace choose_four_from_seven_l3502_350208

theorem choose_four_from_seven : 
  Nat.choose 7 4 = 35 := by
  sorry

end choose_four_from_seven_l3502_350208


namespace gcd_lcm_product_30_75_l3502_350219

theorem gcd_lcm_product_30_75 : Nat.gcd 30 75 * Nat.lcm 30 75 = 2250 := by
  sorry

end gcd_lcm_product_30_75_l3502_350219


namespace course_selection_theorem_l3502_350258

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def total_courses : ℕ := physical_education_courses + art_courses

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def two_course_selections : ℕ := choose physical_education_courses 1 * choose art_courses 1

def three_course_selections : ℕ := 
  choose physical_education_courses 2 * choose art_courses 1 + 
  choose physical_education_courses 1 * choose art_courses 2

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_theorem : total_selections = 64 := by
  sorry

end course_selection_theorem_l3502_350258


namespace triangle_arithmetic_geometric_is_equilateral_l3502_350235

/-- A triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The property that the angles form an arithmetic sequence -/
def Triangle.angles_arithmetic_sequence (t : Triangle) : Prop :=
  ∃ d : ℝ, (t.B - t.A = d ∧ t.C - t.B = d) ∨ (t.A - t.B = d ∧ t.B - t.C = d) ∨ (t.C - t.A = d ∧ t.A - t.B = d)

/-- The property that the sides form a geometric sequence -/
def Triangle.sides_geometric_sequence (t : Triangle) : Prop :=
  (t.b^2 = t.a * t.c) ∨ (t.a^2 = t.b * t.c) ∨ (t.c^2 = t.a * t.b)

/-- A triangle is equilateral if all its sides are equal -/
def Triangle.is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- The main theorem -/
theorem triangle_arithmetic_geometric_is_equilateral (t : Triangle) :
  t.angles_arithmetic_sequence → t.sides_geometric_sequence → t.is_equilateral :=
sorry

end triangle_arithmetic_geometric_is_equilateral_l3502_350235


namespace count_integer_pairs_l3502_350297

theorem count_integer_pairs : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1^2 + p.2 = p.1 * p.2 + 1) ∧ 
    s.card = 5 := by
  sorry

end count_integer_pairs_l3502_350297


namespace solve_baseball_card_problem_l3502_350243

def baseball_card_problem (initial_cards : ℕ) (final_cards : ℕ) : Prop :=
  let cards_after_maria := initial_cards - (initial_cards + 1) / 2
  let cards_after_peter := cards_after_maria - 1
  let cards_paul_added := final_cards - cards_after_peter
  cards_paul_added = 12

theorem solve_baseball_card_problem :
  baseball_card_problem 15 18 := by
  sorry

end solve_baseball_card_problem_l3502_350243


namespace unique_solution_l3502_350217

/-- 
Given two positive integers x and y, prove that if they satisfy the equations
x^y + 4 = y^x and 3x^y = y^x + 10, then x = 7 and y = 1.
-/
theorem unique_solution (x y : ℕ+) 
  (h1 : x^(y:ℕ) + 4 = y^(x:ℕ)) 
  (h2 : 3 * x^(y:ℕ) = y^(x:ℕ) + 10) : 
  x = 7 ∧ y = 1 := by
  sorry

end unique_solution_l3502_350217


namespace candy_store_spending_l3502_350298

/-- Proves that given a weekly allowance of $2.25, after spending 3/5 of it at the arcade
    and 1/3 of the remainder at the toy store, the amount left for the candy store is $0.60. -/
theorem candy_store_spending (weekly_allowance : ℚ) (h1 : weekly_allowance = 2.25) :
  let arcade_spending := (3 / 5) * weekly_allowance
  let remaining_after_arcade := weekly_allowance - arcade_spending
  let toy_store_spending := (1 / 3) * remaining_after_arcade
  let candy_store_spending := remaining_after_arcade - toy_store_spending
  candy_store_spending = 0.60 := by
  sorry


end candy_store_spending_l3502_350298


namespace integral_x_over_sqrt_5_minus_x_l3502_350288

theorem integral_x_over_sqrt_5_minus_x (x : ℝ) :
  HasDerivAt (λ x => (2/3) * (5 - x)^(3/2) - 10 * (5 - x)^(1/2)) 
             (x / (5 - x)^(1/2)) 
             x :=
sorry

end integral_x_over_sqrt_5_minus_x_l3502_350288


namespace bird_multiple_l3502_350229

theorem bird_multiple : ∃ x : ℝ, x * 20 + 10 = 50 := by
  sorry

end bird_multiple_l3502_350229


namespace inverse_inequality_l3502_350296

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end inverse_inequality_l3502_350296


namespace unique_divisible_digit_l3502_350236

def is_single_digit (n : ℕ) : Prop := n < 10

def number_with_A (A : ℕ) : ℕ := 653802 * 10 + A

theorem unique_divisible_digit :
  ∃! A : ℕ, is_single_digit A ∧
    (∀ d : ℕ, d ∈ [2, 3, 4, 6, 8, 9, 25] → (number_with_A A) % d = 0) :=
by sorry

end unique_divisible_digit_l3502_350236


namespace estate_value_l3502_350292

/-- Represents the estate distribution problem --/
structure EstateDistribution where
  total : ℝ
  daughter1 : ℝ
  daughter2 : ℝ
  son : ℝ
  husband : ℝ
  gardener : ℝ

/-- The estate distribution satisfies the given conditions --/
def validDistribution (e : EstateDistribution) : Prop :=
  -- The two daughters and son receive 3/5 of the estate
  e.daughter1 + e.daughter2 + e.son = 3/5 * e.total
  -- The daughters and son share in the ratio of 5:3:2
  ∧ e.daughter1 = 5/10 * (e.daughter1 + e.daughter2 + e.son)
  ∧ e.daughter2 = 3/10 * (e.daughter1 + e.daughter2 + e.son)
  ∧ e.son = 2/10 * (e.daughter1 + e.daughter2 + e.son)
  -- The husband gets three times as much as the son
  ∧ e.husband = 3 * e.son
  -- The gardener receives $600
  ∧ e.gardener = 600
  -- The total estate is the sum of all shares
  ∧ e.total = e.daughter1 + e.daughter2 + e.son + e.husband + e.gardener

/-- The estate value is $15000 --/
theorem estate_value (e : EstateDistribution) (h : validDistribution e) : e.total = 15000 := by
  sorry

end estate_value_l3502_350292


namespace coin_flip_probability_l3502_350223

theorem coin_flip_probability : 
  let n : ℕ := 12  -- Total number of coins
  let k : ℕ := 3   -- Maximum number of heads we're interested in
  let favorable_outcomes : ℕ := (Finset.range (k + 1)).sum (λ i => Nat.choose n i)
  let total_outcomes : ℕ := 2^n
  (favorable_outcomes : ℚ) / total_outcomes = 299 / 4096 := by
sorry

end coin_flip_probability_l3502_350223


namespace range_of_a_l3502_350271

/-- The range of a given the conditions in the problem -/
theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - 4*a*x + 3*a^2 < 0 → |x - 3| > 1) ∧ 
  (∃ x, |x - 3| > 1 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0) ∧ 
  (a > 0) →
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3) :=
sorry


end range_of_a_l3502_350271


namespace triangle_theorem_l3502_350260

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.tan t.C / Real.tan t.B = -t.c / (2 * t.a + t.c))
  (h2 : t.b = 2 * Real.sqrt 3)
  (h3 : t.a + t.c = 4) :
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B : ℝ) = Real.sqrt 3 := by
  sorry

end triangle_theorem_l3502_350260


namespace unique_point_in_S_l3502_350202

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ Real.log (p.1^3 + (1/3)*p.2^3 + 1/9) = Real.log p.1 + Real.log p.2}

theorem unique_point_in_S : ∃! p : ℝ × ℝ, p ∈ S := by sorry

end unique_point_in_S_l3502_350202


namespace sqrt_equation_solution_l3502_350295

theorem sqrt_equation_solution :
  ∃! (x : ℝ), Real.sqrt (3 * x + 7) - Real.sqrt (2 * x - 1) + 2 = 0 ∧ x = 4 := by
  sorry

end sqrt_equation_solution_l3502_350295


namespace sqrt_16_equals_4_l3502_350269

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_16_equals_4_l3502_350269


namespace comparison_of_powers_l3502_350246

theorem comparison_of_powers : 
  let a : ℝ := Real.rpow 0.6 0.6
  let b : ℝ := Real.rpow 0.6 1.2
  let c : ℝ := Real.rpow 1.2 0.6
  b < a ∧ a < c := by sorry

end comparison_of_powers_l3502_350246


namespace probability_consecutive_points_l3502_350203

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of points in the quadrilateral -/
def q : ℕ := 4

/-- The number of points in the triangle -/
def t : ℕ := 3

/-- The number of ways to select 3 consecutive points from n points on a circle -/
def consecutive_selections (n : ℕ) : ℕ := n

/-- The total number of ways to select 3 points from n points -/
def total_selections (n : ℕ) : ℕ := n.choose 3

/-- The probability of selecting 3 consecutive points out of 7 points on a circle,
    given that 4 points have already been selected to form a quadrilateral -/
theorem probability_consecutive_points : 
  (consecutive_selections n : ℚ) / (total_selections n : ℚ) = 1 / 5 := by
  sorry

end probability_consecutive_points_l3502_350203


namespace function_value_plus_derivative_l3502_350278

open Real

/-- Given a differentiable function f : ℝ → ℝ satisfying f x = 2 * x * f.deriv 1 + log x for all x > 0,
    prove that f 1 + f.deriv 1 = -3 -/
theorem function_value_plus_derivative (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (h : ∀ x > 0, f x = 2 * x * (deriv f 1) + log x) :
  f 1 + deriv f 1 = -3 := by
  sorry

end function_value_plus_derivative_l3502_350278


namespace equation_solutions_l3502_350272

theorem equation_solutions :
  ∀ x : ℝ, x ≥ 4 →
    ((x / (2 * Real.sqrt 2) + 5 * Real.sqrt 2 / 2) * Real.sqrt (x^3 - 64*x + 200) = x^2 + 6*x - 40) ↔
    (x = 6 ∨ x = Real.sqrt 13 + 1) :=
by sorry

end equation_solutions_l3502_350272


namespace rectangle_area_l3502_350224

theorem rectangle_area (y : ℝ) (w : ℝ) (h : w > 0) : 
  w^2 + (3*w)^2 = y^2 → 3 * w^2 = (3 * y^2) / 10 :=
by
  sorry

end rectangle_area_l3502_350224


namespace chord_length_squared_l3502_350225

/-- Two circles with given properties and a line through their intersection point --/
structure TwoCirclesWithLine where
  /-- Radius of the first circle --/
  r₁ : ℝ
  /-- Radius of the second circle --/
  r₂ : ℝ
  /-- Distance between the centers of the circles --/
  d : ℝ
  /-- Length of the chord QP (equal to PR) --/
  x : ℝ

/-- Theorem stating the square of the chord length in the given configuration --/
theorem chord_length_squared (c : TwoCirclesWithLine)
  (h₁ : c.r₁ = 5)
  (h₂ : c.r₂ = 10)
  (h₃ : c.d = 16)
  (h₄ : c.x > 0) :
  c.x^2 = 65 := by
  sorry

end chord_length_squared_l3502_350225


namespace sophies_purchase_amount_l3502_350238

/-- Calculates the total amount Sophie spends on her purchase --/
def sophies_purchase (cupcake_price : ℚ) (doughnut_price : ℚ) (pie_price : ℚ)
  (cookie_price : ℚ) (chocolate_price : ℚ) (soda_price : ℚ) (gum_price : ℚ)
  (chips_price : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let subtotal := 5 * cupcake_price + 6 * doughnut_price + 4 * pie_price +
    15 * cookie_price + 8 * chocolate_price + 12 * soda_price +
    3 * gum_price + 10 * chips_price
  let discounted_total := subtotal * (1 - discount_rate)
  let tax_amount := discounted_total * tax_rate
  discounted_total + tax_amount

/-- Theorem stating that Sophie's total purchase amount is $69.45 --/
theorem sophies_purchase_amount :
  sophies_purchase 2 1 2 (6/10) (3/2) (6/5) (4/5) (11/10) (1/10) (6/100) = (6945/100) := by
  sorry

end sophies_purchase_amount_l3502_350238


namespace school_arrival_time_l3502_350283

/-- Represents the problem of calculating how late a boy arrived at school. -/
theorem school_arrival_time (distance : ℝ) (speed_day1 speed_day2 : ℝ) (early_time : ℝ) : 
  distance = 2.5 ∧ 
  speed_day1 = 5 ∧ 
  speed_day2 = 10 ∧ 
  early_time = 10/60 →
  (distance / speed_day1) * 60 - ((distance / speed_day2) * 60 + early_time * 60) = 5 := by
  sorry

#check school_arrival_time

end school_arrival_time_l3502_350283


namespace product_of_difference_and_sum_of_squares_l3502_350214

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a^2 + b^2 = 164) : 
  a * b = -50 := by
sorry

end product_of_difference_and_sum_of_squares_l3502_350214


namespace max_intersections_fifth_degree_polynomials_l3502_350293

/-- A polynomial of degree 5 with leading coefficient 1 -/
def Polynomial5 : Type := ℝ → ℝ

/-- The difference of two polynomials of degree 5 -/
def PolynomialDifference (p q : Polynomial5) : ℝ → ℝ := fun x => p x - q x

theorem max_intersections_fifth_degree_polynomials (p q : Polynomial5) 
  (h_diff : p ≠ q) : 
  (∃ (S : Finset ℝ), ∀ x : ℝ, p x = q x ↔ x ∈ S) ∧ 
  (∀ (S : Finset ℝ), (∀ x : ℝ, p x = q x ↔ x ∈ S) → S.card ≤ 4) :=
sorry

end max_intersections_fifth_degree_polynomials_l3502_350293


namespace triangle_cosine_inequality_l3502_350218

theorem triangle_cosine_inequality (A B C : Real) 
  (h_triangle : A + B + C = π) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  (Real.cos A / Real.cos B)^2 + (Real.cos B / Real.cos C)^2 + (Real.cos C / Real.cos A)^2 
  ≥ 4 * (Real.cos A^2 + Real.cos B^2 + Real.cos C^2) := by
  sorry

end triangle_cosine_inequality_l3502_350218


namespace cat_adoption_rate_is_25_percent_l3502_350222

def initial_dogs : ℕ := 30
def initial_cats : ℕ := 28
def initial_lizards : ℕ := 20
def dog_adoption_rate : ℚ := 1/2
def lizard_adoption_rate : ℚ := 1/5
def new_pets : ℕ := 13
def total_pets_after_month : ℕ := 65

theorem cat_adoption_rate_is_25_percent :
  let dogs_adopted := (initial_dogs : ℚ) * dog_adoption_rate
  let lizards_adopted := (initial_lizards : ℚ) * lizard_adoption_rate
  let remaining_dogs := initial_dogs - dogs_adopted.floor
  let remaining_lizards := initial_lizards - lizards_adopted.floor
  let remaining_pets := remaining_dogs + remaining_lizards + new_pets
  let remaining_cats := total_pets_after_month - remaining_pets
  let cats_adopted := initial_cats - remaining_cats
  (cats_adopted : ℚ) / initial_cats = 1/4 := by
    sorry

end cat_adoption_rate_is_25_percent_l3502_350222


namespace project_work_difference_l3502_350253

/-- Represents the work hours of three people on a project -/
structure ProjectWork where
  person1 : ℝ
  person2 : ℝ
  person3 : ℝ

/-- The conditions of the project work -/
def validProjectWork (work : ProjectWork) : Prop :=
  work.person1 > 0 ∧ work.person2 > 0 ∧ work.person3 > 0 ∧
  work.person2 = 2 * work.person1 ∧
  work.person3 = 3 * work.person1 ∧
  work.person1 + work.person2 + work.person3 = 120

theorem project_work_difference (work : ProjectWork) 
  (h : validProjectWork work) : 
  work.person3 - work.person1 = 40 := by
  sorry

#check project_work_difference

end project_work_difference_l3502_350253


namespace power_sum_equality_l3502_350290

theorem power_sum_equality : (-2)^2007 + (-2)^2008 = 2^2007 := by
  sorry

end power_sum_equality_l3502_350290


namespace successive_discounts_l3502_350200

theorem successive_discounts (original_price : ℝ) (first_discount second_discount : ℝ) 
  (h1 : first_discount = 0.25)
  (h2 : second_discount = 0.10) :
  (original_price * (1 - first_discount) * (1 - second_discount)) / original_price = 0.675 := by
  sorry

end successive_discounts_l3502_350200


namespace complex_fraction_power_l3502_350286

theorem complex_fraction_power (i : ℂ) (h : i^2 = -1) :
  ((1 + i) / (1 - i))^2006 = -1 := by sorry

end complex_fraction_power_l3502_350286


namespace lottery_winner_prize_l3502_350210

def lottery_prize (num_tickets : ℕ) (first_ticket_price : ℕ) (price_increase : ℕ) (profit : ℕ) : ℕ :=
  let total_revenue := (num_tickets * (2 * first_ticket_price + (num_tickets - 1) * price_increase)) / 2
  total_revenue - profit

theorem lottery_winner_prize :
  lottery_prize 5 1 1 4 = 11 := by
  sorry

end lottery_winner_prize_l3502_350210


namespace employee_pay_percentage_l3502_350215

/-- Given two employees X and Y with a total pay of 572, where Y is paid 260,
    prove that X's pay as a percentage of Y's pay is 120%. -/
theorem employee_pay_percentage (X Y : ℝ) : 
  Y = 260 → X + Y = 572 → (X / Y) * 100 = 120 := by sorry

end employee_pay_percentage_l3502_350215


namespace profit_sharing_ratio_l3502_350265

/-- Represents the business partnership between Praveen and Hari -/
structure Partnership where
  praveen_initial : ℚ
  hari_initial : ℚ
  total_months : ℕ
  hari_join_month : ℕ

/-- Calculates the effective contribution of a partner -/
def effective_contribution (initial : ℚ) (months : ℕ) : ℚ :=
  initial * months

/-- Theorem stating the profit-sharing ratio between Praveen and Hari -/
theorem profit_sharing_ratio (p : Partnership) 
  (h1 : p.praveen_initial = 3780)
  (h2 : p.hari_initial = 9720)
  (h3 : p.total_months = 12)
  (h4 : p.hari_join_month = 5) :
  (effective_contribution p.praveen_initial p.total_months) / 
  (effective_contribution p.hari_initial (p.total_months - p.hari_join_month)) = 2 / 3 := by
  sorry


end profit_sharing_ratio_l3502_350265


namespace red_balloon_probability_l3502_350256

/-- Calculates the probability of selecting a red balloon given the initial and additional counts of red and blue balloons. -/
theorem red_balloon_probability
  (initial_red : ℕ)
  (initial_blue : ℕ)
  (additional_red : ℕ)
  (additional_blue : ℕ)
  (h1 : initial_red = 2)
  (h2 : initial_blue = 4)
  (h3 : additional_red = 2)
  (h4 : additional_blue = 2) :
  (initial_red + additional_red : ℚ) / ((initial_red + additional_red + initial_blue + additional_blue) : ℚ) = 2/5 :=
by sorry

end red_balloon_probability_l3502_350256


namespace range_of_m_l3502_350262

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m^2 + 3*m - 3

/-- Proposition p: The minimum value of f(x) is less than 0 -/
def p (m : ℝ) : Prop := ∃ x, f m x < 0

/-- Proposition q: The equation represents an ellipse with foci on the x-axis -/
def q (m : ℝ) : Prop := 5*m - 1 > 0 ∧ m - 2 < 0 ∧ 5*m - 1 > -(m - 2)

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) (h1 : ¬(p m ∨ q m)) (h2 : ¬(p m ∧ q m)) : 
  m ≤ -4 ∨ m ≥ 2 := by sorry

end range_of_m_l3502_350262


namespace M_necessary_not_sufficient_for_N_l3502_350220

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem M_necessary_not_sufficient_for_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by
  sorry

end M_necessary_not_sufficient_for_N_l3502_350220


namespace quadratic_roots_angles_l3502_350226

theorem quadratic_roots_angles (Az m n φ ψ : ℝ) (hAz : Az ≠ 0) :
  (∀ x, Az * x^2 - m * x + n = 0 ↔ x = Real.tan φ ∨ x = Real.tan ψ) →
  Real.tan (φ + ψ) = m / (1 - n) ∧ Real.tan (φ - ψ) = Real.sqrt (m^2 - 4*n) / (1 + n) :=
by sorry

end quadratic_roots_angles_l3502_350226


namespace Z_in_second_quadrant_l3502_350211

-- Define the complex number Z
def Z : ℂ := Complex.I * (1 + Complex.I)

-- Theorem statement
theorem Z_in_second_quadrant : 
  Real.sign (Z.re) = -1 ∧ Real.sign (Z.im) = 1 :=
sorry

end Z_in_second_quadrant_l3502_350211


namespace jaces_debt_jaces_debt_value_l3502_350267

theorem jaces_debt (earned : ℝ) (gave_away_cents : ℕ) (current_balance : ℝ) : ℝ :=
  let gave_away : ℝ := (gave_away_cents : ℝ) / 100
  let debt : ℝ := earned - (current_balance + gave_away)
  debt

theorem jaces_debt_value : jaces_debt 1000 358 642 = 354.42 := by sorry

end jaces_debt_jaces_debt_value_l3502_350267


namespace will_had_28_bottles_l3502_350209

/-- The number of bottles Will had -/
def bottles : ℕ := sorry

/-- The number of days the bottles would last -/
def days : ℕ := 4

/-- The number of bottles Will would drink per day -/
def bottles_per_day : ℕ := 7

/-- Theorem stating that Will had 28 bottles -/
theorem will_had_28_bottles : bottles = 28 := by
  sorry

end will_had_28_bottles_l3502_350209


namespace parallel_lines_a_value_l3502_350216

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line) : Prop := l₁.slope = l₂.slope

theorem parallel_lines_a_value :
  ∀ a : ℝ,
  let l₁ : Line := ⟨1, a/2⟩
  let l₂ : Line := ⟨a^2 - 3, 1⟩
  parallel l₁ l₂ → a = -2 := by
sorry

end parallel_lines_a_value_l3502_350216


namespace cube_difference_l3502_350237

theorem cube_difference (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 26) : 
  a^3 - b^3 = 124 := by
sorry

end cube_difference_l3502_350237


namespace unique_base_nine_l3502_350204

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem unique_base_nine :
  ∃! b : Nat, b > 1 ∧ 
    to_decimal [1, 5, 2] b + to_decimal [1, 4, 3] b = to_decimal [3, 0, 5] b :=
by
  sorry

end unique_base_nine_l3502_350204


namespace roots_of_equation_l3502_350266

def equation (x : ℝ) : ℝ := (x^2 - 5*x + 6) * (x - 3) * (x + 2)

theorem roots_of_equation : 
  {x : ℝ | equation x = 0} = {-2, 2, 3} := by sorry

end roots_of_equation_l3502_350266


namespace mateo_deducted_salary_l3502_350230

/-- Calculates the deducted salary for a worker given their weekly salary and number of absent days. -/
def deducted_salary (weekly_salary : ℚ) (absent_days : ℕ) : ℚ :=
  weekly_salary - (weekly_salary / 5 * absent_days)

/-- Proves that Mateo's deducted salary is correct given his weekly salary and absent days. -/
theorem mateo_deducted_salary :
  deducted_salary 791 4 = 158.2 := by
  sorry

end mateo_deducted_salary_l3502_350230


namespace fourth_operation_result_l3502_350259

def pattern_result (a b : ℕ) : ℕ := a * b + a * (b - a)

theorem fourth_operation_result : pattern_result 5 8 = 55 := by
  sorry

end fourth_operation_result_l3502_350259


namespace fraction_value_l3502_350273

/-- Given a, b, c, d are real numbers satisfying certain relationships,
    prove that (a * c) / (b * d) = 15 -/
theorem fraction_value (a b c d : ℝ) 
    (h1 : a = 3 * b) 
    (h2 : b = 2 * c) 
    (h3 : c = 5 * d) 
    (h4 : b ≠ 0) 
    (h5 : d ≠ 0) : 
  (a * c) / (b * d) = 15 := by
  sorry

end fraction_value_l3502_350273


namespace complex_i_minus_one_in_third_quadrant_l3502_350234

theorem complex_i_minus_one_in_third_quadrant :
  let z : ℂ := Complex.I * (Complex.I - 1)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end complex_i_minus_one_in_third_quadrant_l3502_350234


namespace sandy_savings_l3502_350212

theorem sandy_savings (last_year_salary : ℝ) (last_year_savings_rate : ℝ) 
  (h1 : last_year_savings_rate > 0)
  (h2 : last_year_savings_rate < 1)
  (h3 : (1.1 * last_year_salary) * 0.09 = 1.65 * (last_year_salary * last_year_savings_rate)) :
  last_year_savings_rate = 0.6 := by
  sorry

end sandy_savings_l3502_350212


namespace vector_parallel_value_l3502_350289

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem vector_parallel_value (x : ℝ) :
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, x - 1)
  parallel a b → x = 3/2 := by
  sorry

end vector_parallel_value_l3502_350289


namespace banana_survey_l3502_350248

theorem banana_survey (total_students : ℕ) (banana_percentage : ℚ) : 
  total_students = 100 →
  banana_percentage = 1/5 →
  (banana_percentage * total_students : ℚ) = 20 := by
  sorry

end banana_survey_l3502_350248


namespace solve_for_a_l3502_350205

-- Define the operation *
def star_op (a b : ℚ) : ℚ := 2*a - b^2

-- Theorem statement
theorem solve_for_a :
  ∀ a : ℚ, star_op a 7 = -20 → a = 29/2 := by
  sorry

end solve_for_a_l3502_350205


namespace rationalize_denominator_l3502_350276

theorem rationalize_denominator : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end rationalize_denominator_l3502_350276


namespace ali_boxes_calculation_l3502_350254

/-- The number of boxes Ali used for each of his circles -/
def ali_boxes_per_circle : ℕ := 14

/-- The total number of boxes -/
def total_boxes : ℕ := 80

/-- The number of circles Ali made -/
def ali_circles : ℕ := 5

/-- The number of boxes Ernie used for his circle -/
def ernie_boxes : ℕ := 10

theorem ali_boxes_calculation :
  ali_boxes_per_circle * ali_circles + ernie_boxes = total_boxes :=
by sorry

end ali_boxes_calculation_l3502_350254


namespace committee_selection_l3502_350261

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committee_selection :
  choose 20 3 = 1140 := by
  sorry

end committee_selection_l3502_350261


namespace cohen_bird_count_l3502_350206

/-- The total number of fish-eater birds Cohen saw over three days -/
def total_birds (day1 : ℕ) (day2_factor : ℕ) (day3_reduction : ℕ) : ℕ :=
  day1 + day1 * day2_factor + (day1 * day2_factor - day3_reduction)

/-- Theorem stating the total number of fish-eater birds Cohen saw over three days -/
theorem cohen_bird_count :
  total_birds 300 2 200 = 1300 := by
  sorry

end cohen_bird_count_l3502_350206


namespace cashier_bills_l3502_350299

theorem cashier_bills (total_bills : ℕ) (total_value : ℕ) : 
  total_bills = 126 → total_value = 840 → ∃ (five_dollar_bills ten_dollar_bills : ℕ),
    five_dollar_bills + ten_dollar_bills = total_bills ∧
    5 * five_dollar_bills + 10 * ten_dollar_bills = total_value ∧
    five_dollar_bills = 84 := by
  sorry

end cashier_bills_l3502_350299


namespace fraction_contradiction_l3502_350257

theorem fraction_contradiction : ¬∃ (x : ℚ), (8 * x = 4) ∧ ((1/4) * 16 = 10 * x) := by
  sorry

end fraction_contradiction_l3502_350257


namespace systematic_sampling_first_group_l3502_350240

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalStudents : Nat
  numGroups : Nat
  sampleSize : Nat
  groupSize : Nat
  sixteenthGroupDraw : Nat

/-- Theorem for systematic sampling -/
theorem systematic_sampling_first_group
  (setup : SystematicSampling)
  (h1 : setup.totalStudents = 160)
  (h2 : setup.numGroups = 20)
  (h3 : setup.sampleSize = 20)
  (h4 : setup.groupSize = setup.totalStudents / setup.numGroups)
  (h5 : setup.sixteenthGroupDraw = 126) :
  ∃ (firstGroupDraw : Nat), firstGroupDraw = 6 ∧
    setup.sixteenthGroupDraw = (16 - 1) * setup.groupSize + firstGroupDraw :=
by sorry

end systematic_sampling_first_group_l3502_350240


namespace trigonometric_equality_l3502_350241

theorem trigonometric_equality (a b c : ℝ) (α β : ℝ) 
  (h1 : a * Real.cos α + b * Real.sin α = c)
  (h2 : a * Real.cos β + b * Real.sin β = 0)
  (h3 : ¬(a = 0 ∧ b = 0 ∧ c = 0)) :
  Real.sin (α - β) ^ 2 = c^2 / (a^2 + b^2) := by
sorry

end trigonometric_equality_l3502_350241


namespace certain_number_problem_l3502_350213

theorem certain_number_problem (x : ℚ) : 
  (((x + 5) * 2) / 5) - 5 = 44 / 2 → x = 62.5 := by
  sorry

end certain_number_problem_l3502_350213


namespace arithmetic_sequence_inequality_l3502_350263

-- Define the sequence a_n
def a (n : ℕ+) : ℕ := 2 * n - 1

-- Define the sum of the first n terms
def S (n : ℕ+) : ℕ := n * n

-- State the theorem
theorem arithmetic_sequence_inequality (m k p : ℕ+) (h : m + p = 2 * k) :
  1 / S m + 1 / S p ≥ 2 / S k := by
  sorry

end arithmetic_sequence_inequality_l3502_350263


namespace sin_equality_in_range_l3502_350232

theorem sin_equality_in_range (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 → Real.sin (n * π / 180) = Real.sin (750 * π / 180) → n = 30 := by
  sorry

end sin_equality_in_range_l3502_350232


namespace square_sum_of_integers_l3502_350233

theorem square_sum_of_integers (x y z : ℤ) 
  (eq1 : x^2*y + y^2*z + z^2*x = 2186)
  (eq2 : x*y^2 + y*z^2 + z*x^2 = 2188) :
  x^2 + y^2 + z^2 = 245 := by
  sorry

end square_sum_of_integers_l3502_350233


namespace ring_toss_earnings_l3502_350287

/-- The ring toss game made this amount in the first 44 days -/
def first_period_earnings : ℕ := 382

/-- The ring toss game made this amount in the remaining 10 days -/
def second_period_earnings : ℕ := 374

/-- The total earnings of the ring toss game -/
def total_earnings : ℕ := first_period_earnings + second_period_earnings

theorem ring_toss_earnings : total_earnings = 756 := by
  sorry

end ring_toss_earnings_l3502_350287


namespace multiply_b_is_eight_l3502_350268

theorem multiply_b_is_eight (a b x : ℝ) 
  (h1 : 7 * a = x * b) 
  (h2 : a * b ≠ 0) 
  (h3 : (a / 8) / (b / 7) = 1) : 
  x = 8 := by
  sorry

end multiply_b_is_eight_l3502_350268
