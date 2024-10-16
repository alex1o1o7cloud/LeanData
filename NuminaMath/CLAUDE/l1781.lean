import Mathlib

namespace NUMINAMATH_CALUDE_select_books_eq_42_l1781_178142

/-- The number of ways to select r items from n items -/
def combination (n r : ℕ) : ℕ := Nat.choose n r

/-- The total number of books -/
def total_books : ℕ := 9

/-- The number of identical Analects -/
def analects : ℕ := 3

/-- The number of different modern literary masterpieces -/
def masterpieces : ℕ := 6

/-- The number of books to be selected -/
def books_to_select : ℕ := 3

/-- The number of ways to select 3 books from the given collection -/
def select_books : ℕ :=
  1 + combination masterpieces 1 + combination masterpieces 2 + combination masterpieces 3

theorem select_books_eq_42 : select_books = 42 := by sorry

end NUMINAMATH_CALUDE_select_books_eq_42_l1781_178142


namespace NUMINAMATH_CALUDE_tan_negative_1125_degrees_l1781_178161

theorem tan_negative_1125_degrees : Real.tan ((-1125 : ℝ) * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_1125_degrees_l1781_178161


namespace NUMINAMATH_CALUDE_fourth_power_congruence_divisibility_l1781_178164

theorem fourth_power_congruence_divisibility (p a b c d : ℕ) (hp : Prime p) 
  (ha : 0 < a) (hab : a < b) (hbc : b < c) (hcd : c < d) (hdp : d < p)
  (hcong : ∃ k : ℕ, a^4 % p = k ∧ b^4 % p = k ∧ c^4 % p = k ∧ d^4 % p = k) :
  (a + b + c + d) ∣ (a^2013 + b^2013 + c^2013 + d^2013) := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_congruence_divisibility_l1781_178164


namespace NUMINAMATH_CALUDE_actual_weight_calculation_l1781_178181

/-- The dealer's percent -/
def dealer_percent : ℝ := 53.84615384615387

/-- The actual weight used per kg -/
def actual_weight : ℝ := 0.4615384615384613

/-- Theorem stating that the actual weight used per kg is correct given the dealer's percent -/
theorem actual_weight_calculation (ε : ℝ) (h : ε > 0) : 
  |actual_weight - (1 - dealer_percent / 100)| < ε :=
by sorry

end NUMINAMATH_CALUDE_actual_weight_calculation_l1781_178181


namespace NUMINAMATH_CALUDE_sum_of_radii_is_eight_l1781_178102

/-- A circle with center C that is tangent to the positive x and y-axes
    and externally tangent to a circle centered at (3,0) with radius 1 -/
def CircleC (r : ℝ) : Prop :=
  (∃ C : ℝ × ℝ, C.1 = r ∧ C.2 = r) ∧  -- Center of circle C is at (r,r)
  ((r - 3)^2 + r^2 = (r + 1)^2)  -- External tangency condition

/-- The theorem stating that the sum of all possible radii of CircleC is 8 -/
theorem sum_of_radii_is_eight :
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ CircleC r₁ ∧ CircleC r₂ ∧ r₁ + r₂ = 8 :=
sorry

end NUMINAMATH_CALUDE_sum_of_radii_is_eight_l1781_178102


namespace NUMINAMATH_CALUDE_class_composition_l1781_178177

theorem class_composition (total students : ℕ) (girls boys : ℕ) : 
  students = girls + boys →
  (girls : ℚ) / (students : ℚ) = 60 / 100 →
  ((girls - 1 : ℚ) / ((students - 3) : ℚ)) = 125 / 200 →
  girls = 21 ∧ boys = 14 :=
by sorry

end NUMINAMATH_CALUDE_class_composition_l1781_178177


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1781_178160

theorem constant_term_expansion (n : ℕ+) : 
  (∃ r : ℕ, r = 6 ∧ 3*n - 4*r = 0) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1781_178160


namespace NUMINAMATH_CALUDE_series_convergence_l1781_178122

theorem series_convergence (a : ℕ → ℝ) :
  (∃ S : ℝ, HasSum (λ n : ℕ => a n + 2 * a (n + 1)) S) →
  (∃ T : ℝ, HasSum a T) :=
by sorry

end NUMINAMATH_CALUDE_series_convergence_l1781_178122


namespace NUMINAMATH_CALUDE_tower_count_mod_1000_l1781_178123

/-- A function that calculates the number of towers for n cubes -/
def tower_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 32
  | m + 4 => 4 * tower_count (m + 3)

/-- The theorem stating that the number of towers for 10 cubes is congruent to 288 mod 1000 -/
theorem tower_count_mod_1000 :
  tower_count 10 ≡ 288 [MOD 1000] :=
sorry

end NUMINAMATH_CALUDE_tower_count_mod_1000_l1781_178123


namespace NUMINAMATH_CALUDE_nick_pennsylvania_quarters_l1781_178141

/-- Given a total number of quarters, calculate the number of Pennsylvania state quarters -/
def pennsylvania_quarters (total : ℕ) : ℕ :=
  let state_quarters := (2 * total) / 5
  (state_quarters / 2 : ℕ)

theorem nick_pennsylvania_quarters :
  pennsylvania_quarters 35 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nick_pennsylvania_quarters_l1781_178141


namespace NUMINAMATH_CALUDE_range_of_a_l1781_178184

theorem range_of_a (x a : ℝ) : 
  (∀ x, (x^2 + 2*x - 3 > 0 → x > a) ∧ 
   (x^2 + 2*x - 3 ≤ 0 → x ≤ a) ∧ 
   ∃ x, x^2 + 2*x - 3 > 0 ∧ x > a) →
  a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1781_178184


namespace NUMINAMATH_CALUDE_trader_profit_double_price_l1781_178106

theorem trader_profit_double_price (cost : ℝ) (initial_profit_percent : ℝ) 
  (h1 : initial_profit_percent = 40) : 
  let initial_price := cost * (1 + initial_profit_percent / 100)
  let new_price := 2 * initial_price
  let new_profit := new_price - cost
  new_profit / cost * 100 = 180 := by
sorry

end NUMINAMATH_CALUDE_trader_profit_double_price_l1781_178106


namespace NUMINAMATH_CALUDE_rajas_household_expenditure_percentage_l1781_178166

theorem rajas_household_expenditure_percentage 
  (monthly_income : ℝ) 
  (clothes_percentage : ℝ) 
  (medicines_percentage : ℝ) 
  (savings : ℝ) 
  (h1 : monthly_income = 37500) 
  (h2 : clothes_percentage = 20) 
  (h3 : medicines_percentage = 5) 
  (h4 : savings = 15000) : 
  (monthly_income - (monthly_income * clothes_percentage / 100 + 
   monthly_income * medicines_percentage / 100 + savings)) / monthly_income * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_rajas_household_expenditure_percentage_l1781_178166


namespace NUMINAMATH_CALUDE_cos_two_alpha_l1781_178117

theorem cos_two_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin (α - π / 4) = 1 / 3) : 
  Real.cos (2 * α) = -4 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_l1781_178117


namespace NUMINAMATH_CALUDE_george_eggs_boxes_l1781_178191

/-- Given a total number of eggs and eggs per box, calculates the number of boxes required. -/
def calculate_boxes (total_eggs : ℕ) (eggs_per_box : ℕ) : ℕ :=
  total_eggs / eggs_per_box

theorem george_eggs_boxes :
  let total_eggs : ℕ := 15
  let eggs_per_box : ℕ := 3
  calculate_boxes total_eggs eggs_per_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_george_eggs_boxes_l1781_178191


namespace NUMINAMATH_CALUDE_square_side_length_l1781_178176

theorem square_side_length (area : ℝ) (side_length : ℝ) :
  area = 4 ∧ area = side_length ^ 2 → side_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1781_178176


namespace NUMINAMATH_CALUDE_hyperbola_focus_m_value_l1781_178174

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / m - y^2 / (3 + m) = 1

-- Define the focus of the hyperbola
def focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem hyperbola_focus_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, hyperbola_equation x y m) → focus.1 = 2 → focus.2 = 0 → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_m_value_l1781_178174


namespace NUMINAMATH_CALUDE_sin_cos_relation_l1781_178114

theorem sin_cos_relation (x : Real) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x ^ 2 - Real.cos x ^ 2 = 15 / 17 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l1781_178114


namespace NUMINAMATH_CALUDE_power_equation_solution_l1781_178144

theorem power_equation_solution (m : ℕ) : (4 : ℝ)^m * 2^3 = 8^5 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1781_178144


namespace NUMINAMATH_CALUDE_pencil_cost_l1781_178193

theorem pencil_cost (x y : ℚ) 
  (eq1 : 4 * x + 3 * y = 224)
  (eq2 : 2 * x + 5 * y = 154) : 
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_pencil_cost_l1781_178193


namespace NUMINAMATH_CALUDE_equation_solutions_l1781_178130

theorem equation_solutions :
  (∃ x : ℝ, x - 2 * (5 + x) = -4 ∧ x = -6) ∧
  (∃ x : ℝ, (2 * x - 1) / 2 = 1 - (3 - x) / 4 ∧ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1781_178130


namespace NUMINAMATH_CALUDE_average_monthly_sales_l1781_178196

def monthly_sales : List ℝ := [80, 100, 75, 95, 110, 180, 90, 115, 130, 200, 160, 140]

theorem average_monthly_sales :
  (monthly_sales.sum / monthly_sales.length : ℝ) = 122.92 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_sales_l1781_178196


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1781_178197

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 1 < 0 ↔ 1/2 < x ∧ x < 2) → a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1781_178197


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1781_178113

/-- Given a polynomial equation, prove the sum of specific coefficients --/
theorem polynomial_coefficient_sum :
  ∀ (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ),
  (∀ x : ℝ, a + a₁ * (x + 2) + a₂ * (x + 2)^2 + a₃ * (x + 2)^3 + a₄ * (x + 2)^4 + 
             a₅ * (x + 2)^5 + a₆ * (x + 2)^6 + a₇ * (x + 2)^7 + a₈ * (x + 2)^8 + 
             a₉ * (x + 2)^9 + a₁₀ * (x + 2)^10 + a₁₁ * (x + 2)^11 + a₁₂ * (x + 2)^12 = 
             (x^2 - 2*x - 2)^6) →
  2*a₂ + 6*a₃ + 12*a₄ + 20*a₅ + 30*a₆ + 42*a₇ + 56*a₈ + 72*a₉ + 90*a₁₀ + 110*a₁₁ + 132*a₁₂ = 492 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1781_178113


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1781_178137

theorem complex_number_quadrant : ∃ (z : ℂ), z = (25 / (3 - 4*I)) * I ∧ (z.re < 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1781_178137


namespace NUMINAMATH_CALUDE_overtime_compensation_l1781_178185

def total_employees : ℕ := 350
def men_pay_rate : ℚ := 10
def women_pay_rate : ℚ := 815/100

theorem overtime_compensation 
  (total_men : ℕ) 
  (men_accepted : ℕ) 
  (h1 : total_men ≤ total_employees) 
  (h2 : men_accepted ≤ total_men) 
  (h3 : ∀ (m : ℕ), m ≤ total_men → 
    men_pay_rate * m + women_pay_rate * (total_employees - m) = 
    men_pay_rate * men_accepted + women_pay_rate * (total_employees - total_men)) :
  women_pay_rate * (total_employees - total_men) = 122250/100 := by
  sorry

end NUMINAMATH_CALUDE_overtime_compensation_l1781_178185


namespace NUMINAMATH_CALUDE_quadrilateral_circumcenter_l1781_178111

-- Define the types of quadrilaterals
inductive Quadrilateral
  | Square
  | NonSquareRectangle
  | NonSquareRhombus
  | Parallelogram
  | IsoscelesTrapezoid

-- Define a function to check if a quadrilateral has a circumcenter
def hasCircumcenter (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Square => True
  | Quadrilateral.NonSquareRectangle => True
  | Quadrilateral.NonSquareRhombus => False
  | Quadrilateral.Parallelogram => False
  | Quadrilateral.IsoscelesTrapezoid => True

-- Theorem stating which quadrilaterals have a circumcenter
theorem quadrilateral_circumcenter :
  ∀ q : Quadrilateral,
    hasCircumcenter q ↔
      (q = Quadrilateral.Square ∨
       q = Quadrilateral.NonSquareRectangle ∨
       q = Quadrilateral.IsoscelesTrapezoid) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_circumcenter_l1781_178111


namespace NUMINAMATH_CALUDE_bill_bouquet_profit_l1781_178168

/-- Represents the number of roses in a bouquet Bill buys -/
def roses_per_bought_bouquet : ℕ := 7

/-- Represents the number of roses in a bouquet Bill sells -/
def roses_per_sold_bouquet : ℕ := 5

/-- Represents the price of a bouquet (both buying and selling) in dollars -/
def price_per_bouquet : ℕ := 20

/-- Represents the target profit in dollars -/
def target_profit : ℕ := 1000

/-- Calculates the number of bouquets Bill needs to buy to earn the target profit -/
def bouquets_to_buy : ℕ :=
  let bought_bouquets_per_operation := roses_per_sold_bouquet
  let sold_bouquets_per_operation := roses_per_bought_bouquet
  let profit_per_operation := sold_bouquets_per_operation * price_per_bouquet - bought_bouquets_per_operation * price_per_bouquet
  let operations_needed := target_profit / profit_per_operation
  operations_needed * bought_bouquets_per_operation

theorem bill_bouquet_profit :
  bouquets_to_buy = 125 := by sorry

end NUMINAMATH_CALUDE_bill_bouquet_profit_l1781_178168


namespace NUMINAMATH_CALUDE_average_speed_two_segments_l1781_178135

/-- Calculate the average speed of a two-segment journey -/
theorem average_speed_two_segments 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 50) 
  (h2 : speed1 = 20) 
  (h3 : distance2 = 20) 
  (h4 : speed2 = 40) : 
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 70 / 3 := by
  sorry

#eval (70 : ℚ) / 3

end NUMINAMATH_CALUDE_average_speed_two_segments_l1781_178135


namespace NUMINAMATH_CALUDE_pattern_cannot_form_cube_l1781_178147

/-- Represents a square in the pattern -/
structure Square :=
  (id : ℕ)

/-- Represents the pattern of squares -/
structure Pattern :=
  (center : Square)
  (top : Square)
  (left : Square)
  (right : Square)
  (front : Square)

/-- Represents a cube -/
structure Cube :=
  (faces : Fin 6 → Square)

/-- Defines the given pattern -/
def given_pattern : Pattern :=
  { center := ⟨0⟩
  , top := ⟨1⟩
  , left := ⟨2⟩
  , right := ⟨3⟩
  , front := ⟨4⟩ }

/-- Theorem stating that the given pattern cannot form a cube -/
theorem pattern_cannot_form_cube :
  ¬ ∃ (c : Cube), c.faces 0 = given_pattern.center ∧
                  c.faces 1 = given_pattern.top ∧
                  c.faces 2 = given_pattern.left ∧
                  c.faces 3 = given_pattern.right ∧
                  c.faces 4 = given_pattern.front :=
by
  sorry


end NUMINAMATH_CALUDE_pattern_cannot_form_cube_l1781_178147


namespace NUMINAMATH_CALUDE_quadratic_root_value_l1781_178199

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℝ, 5 * x^2 + 20 * x + k = 0 ↔ x = (-20 + Real.sqrt 60) / 10 ∨ x = (-20 - Real.sqrt 60) / 10) 
  → k = 17 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l1781_178199


namespace NUMINAMATH_CALUDE_other_diagonal_length_l1781_178153

/-- Represents a rhombus with given properties -/
structure Rhombus where
  d1 : ℝ  -- Length of one diagonal
  d2 : ℝ  -- Length of the other diagonal
  area : ℝ -- Area of the rhombus

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with one diagonal of 16 cm and an area of 88 cm², 
    the length of the other diagonal is 11 cm -/
theorem other_diagonal_length (r : Rhombus) 
    (h1 : r.d2 = 16) 
    (h2 : r.area = 88) : 
    r.d1 = 11 := by
  sorry


end NUMINAMATH_CALUDE_other_diagonal_length_l1781_178153


namespace NUMINAMATH_CALUDE_even_digits_in_base8_523_l1781_178132

/-- Converts a natural number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of even digits in the base-8 representation of 523₁₀ is 1 -/
theorem even_digits_in_base8_523 :
  countEvenDigits (toBase8 523) = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_in_base8_523_l1781_178132


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l1781_178198

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let runsInFirstPart := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.targetRuns - runsInFirstPart
  remainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 6.2)
  (h4 : game.targetRuns = 282) :
  requiredRunRate game = 5.5 := by
  sorry

#eval requiredRunRate { totalOvers := 50, firstPartOvers := 10, firstPartRunRate := 6.2, targetRuns := 282 }

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l1781_178198


namespace NUMINAMATH_CALUDE_opposites_sum_l1781_178151

theorem opposites_sum (a b : ℝ) (h : a + b = 0) : 2006*a + 2 + 2006*b = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_l1781_178151


namespace NUMINAMATH_CALUDE_oranges_per_glass_l1781_178109

/-- Proves that the number of oranges per glass is 2, given 12 oranges used for 6 glasses of juice -/
theorem oranges_per_glass (total_oranges : ℕ) (total_glasses : ℕ) 
  (h1 : total_oranges = 12) (h2 : total_glasses = 6) :
  total_oranges / total_glasses = 2 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_glass_l1781_178109


namespace NUMINAMATH_CALUDE_simplify_expression_l1781_178190

theorem simplify_expression (a c d x y z : ℝ) (h : cx + dz ≠ 0) :
  (c*x*(a^3*x^3 + 3*a^3*y^3 + c^3*z^3) + d*z*(a^3*x^3 + 3*c^3*x^3 + c^3*z^3)) / (c*x + d*z) =
  a^3*x^3 + c^3*z^3 + (3*c*x*a^3*y^3)/(c*x + d*z) + (3*d*z*c^3*x^3)/(c*x + d*z) := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l1781_178190


namespace NUMINAMATH_CALUDE_expected_ones_is_half_l1781_178156

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expected_ones : ℚ :=
  0 * (prob_not_one ^ num_dice) +
  1 * (num_dice * prob_one * prob_not_one ^ 2) +
  2 * (num_dice * prob_one ^ 2 * prob_not_one) +
  3 * prob_one ^ num_dice

theorem expected_ones_is_half : expected_ones = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_ones_is_half_l1781_178156


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l1781_178145

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l1781_178145


namespace NUMINAMATH_CALUDE_pyramid_sum_l1781_178155

/-- Given a pyramid of numbers where each number is the sum of the two above it,
    prove that the top number is 381. -/
theorem pyramid_sum (y z : ℕ) (h1 : y + 600 = 1119) (h2 : z + 1119 = 2019) (h3 : 381 + y = z) :
  ∃ x : ℕ, x = 381 ∧ x + y = z :=
by sorry

end NUMINAMATH_CALUDE_pyramid_sum_l1781_178155


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1781_178188

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 + 5*x - 6 > 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -6 ∨ x > 1}

-- Theorem statement
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1781_178188


namespace NUMINAMATH_CALUDE_problem_solution_l1781_178195

theorem problem_solution (x y z : ℝ) 
  (h1 : (x + y)^2 + (y + z)^2 + (x + z)^2 = 94)
  (h2 : (x - y)^2 + (y - z)^2 + (x - z)^2 = 26) :
  (x * y + y * z + x * z = 17) ∧
  ((x + 2*y + 3*z)^2 + (y + 2*z + 3*x)^2 + (z + 2*x + 3*y)^2 = 794) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1781_178195


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1781_178165

theorem quadratic_equation_solutions (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1781_178165


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l1781_178162

theorem simplify_nested_roots (a : ℝ) : 
  (((a^16)^(1/3))^(1/4))^3 * (((a^16)^(1/4))^(1/3))^2 = a^(20/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l1781_178162


namespace NUMINAMATH_CALUDE_german_enrollment_l1781_178172

theorem german_enrollment (total_students : ℕ) (both_subjects : ℕ) (only_english : ℕ) 
  (h1 : total_students = 40)
  (h2 : both_subjects = 12)
  (h3 : only_english = 18)
  (h4 : ∃ (german : ℕ), german > 0)
  (h5 : total_students = both_subjects + only_english + (total_students - both_subjects - only_english)) :
  total_students - only_english = 22 := by
  sorry

end NUMINAMATH_CALUDE_german_enrollment_l1781_178172


namespace NUMINAMATH_CALUDE_desk_purchase_optimization_l1781_178121

/-- The total cost function for shipping and storage fees -/
def f (x : ℕ) : ℚ := 144 / x + 4 * x

/-- The number of desks to be purchased -/
def total_desks : ℕ := 36

/-- The value of each desk -/
def desk_value : ℕ := 20

/-- The shipping fee per batch -/
def shipping_fee : ℕ := 4

/-- Available funds for shipping and storage -/
def available_funds : ℕ := 48

theorem desk_purchase_optimization :
  /- 1. The total cost function is correct -/
  (∀ x : ℕ, x > 0 → f x = 144 / x + 4 * x) ∧
  /- 2. There exists an integer x between 4 and 9 inclusive that satisfies the budget -/
  (∃ x : ℕ, 4 ≤ x ∧ x ≤ 9 ∧ f x ≤ available_funds) ∧
  /- 3. The minimum value of f(x) occurs when x = 6 -/
  (∀ x : ℕ, x > 0 → f 6 ≤ f x) :=
by sorry

end NUMINAMATH_CALUDE_desk_purchase_optimization_l1781_178121


namespace NUMINAMATH_CALUDE_john_bus_meet_once_l1781_178129

/-- Represents the movement of John and the bus on a straight path --/
structure Movement where
  johnSpeed : ℝ
  busSpeed : ℝ
  benchDistance : ℝ
  busStopTime : ℝ

/-- Calculates the number of times John and the bus meet --/
def meetingCount (m : Movement) : ℕ :=
  sorry

/-- Theorem stating that John and the bus meet exactly once --/
theorem john_bus_meet_once (m : Movement) 
  (h1 : m.johnSpeed = 6)
  (h2 : m.busSpeed = 15)
  (h3 : m.benchDistance = 300)
  (h4 : m.busStopTime = 45) :
  meetingCount m = 1 := by
  sorry

end NUMINAMATH_CALUDE_john_bus_meet_once_l1781_178129


namespace NUMINAMATH_CALUDE_systems_solutions_l1781_178125

theorem systems_solutions :
  (∃ x y : ℝ, x = 2*y - 1 ∧ 3*x + 4*y = 17 ∧ x = 3 ∧ y = 2) ∧
  (∃ x y : ℝ, 2*x - y = 0 ∧ 3*x - 2*y = 5 ∧ x = -5 ∧ y = -10) :=
by sorry

end NUMINAMATH_CALUDE_systems_solutions_l1781_178125


namespace NUMINAMATH_CALUDE_lower_interest_rate_l1781_178154

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem lower_interest_rate 
  (principal : ℝ) 
  (high_rate low_rate : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) :
  principal = 12000 →
  high_rate = 0.15 →
  time = 2 →
  interest_difference = 720 →
  simple_interest principal high_rate time - simple_interest principal low_rate time = interest_difference →
  low_rate = 0.12 := by
sorry

end NUMINAMATH_CALUDE_lower_interest_rate_l1781_178154


namespace NUMINAMATH_CALUDE_opposite_numbers_solution_l1781_178167

theorem opposite_numbers_solution (x : ℝ) : 2 * (x - 3) = -(4 * (1 - x)) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_solution_l1781_178167


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1781_178107

/-- Given a complex number in the form (2-mi)/(1+2i) = A+Bi, where m, A, and B are real numbers,
    if A + B = 0, then m = 2 -/
theorem complex_equation_solution (m A B : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - m * Complex.I) / (1 + 2 * Complex.I) = A + B * Complex.I →
  A + B = 0 →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1781_178107


namespace NUMINAMATH_CALUDE_floss_leftover_and_cost_l1781_178179

/-- Represents the floss requirements for a class -/
structure ClassFlossRequirement where
  students : ℕ
  flossPerStudent : ℚ

/-- Represents the floss sale conditions -/
structure FlossSaleConditions where
  metersPerPacket : ℚ
  pricePerPacket : ℚ
  discountRate : ℚ
  discountThreshold : ℕ

def yardToMeter : ℚ := 0.9144

def classes : List ClassFlossRequirement := [
  ⟨20, 1.5⟩,
  ⟨25, 1.75⟩,
  ⟨30, 2⟩
]

def saleConditions : FlossSaleConditions := {
  metersPerPacket := 50,
  pricePerPacket := 5,
  discountRate := 0.1,
  discountThreshold := 2
}

theorem floss_leftover_and_cost 
  (classes : List ClassFlossRequirement) 
  (saleConditions : FlossSaleConditions) 
  (yardToMeter : ℚ) : 
  ∃ (cost leftover : ℚ), cost = 14.5 ∧ leftover = 27.737 := by
  sorry

end NUMINAMATH_CALUDE_floss_leftover_and_cost_l1781_178179


namespace NUMINAMATH_CALUDE_nineteen_power_calculation_l1781_178116

theorem nineteen_power_calculation : (19^11 / 19^8) * 19^3 = 47015881 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_power_calculation_l1781_178116


namespace NUMINAMATH_CALUDE_chocolate_count_l1781_178175

/-- The number of boxes of chocolates -/
def num_boxes : ℕ := 6

/-- The number of pieces of chocolate in each box -/
def pieces_per_box : ℕ := 500

/-- The total number of pieces of chocolate -/
def total_pieces : ℕ := num_boxes * pieces_per_box

theorem chocolate_count : total_pieces = 3000 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_count_l1781_178175


namespace NUMINAMATH_CALUDE_dog_arrangement_count_l1781_178118

theorem dog_arrangement_count : ∀ (n m k : ℕ),
  n = 15 →
  m = 3 →
  k = 12 →
  (Nat.choose k 3) * (Nat.choose (k - 3) 6) = 18480 :=
by
  sorry

end NUMINAMATH_CALUDE_dog_arrangement_count_l1781_178118


namespace NUMINAMATH_CALUDE_invariant_quotient_division_inequality_l1781_178163

-- Define division with remainder
def div_with_rem (a b : ℕ) : ℕ × ℕ :=
  (a / b, a % b)

-- Property of invariant quotient
theorem invariant_quotient (a b c : ℕ) (h : c ≠ 0) :
  div_with_rem (a * c) (b * c) = (a / b, c * (a % b)) :=
sorry

-- Main theorem
theorem division_inequality :
  div_with_rem 1700 500 ≠ div_with_rem 17 5 :=
sorry

end NUMINAMATH_CALUDE_invariant_quotient_division_inequality_l1781_178163


namespace NUMINAMATH_CALUDE_pencils_per_box_l1781_178170

theorem pencils_per_box (total_boxes : ℕ) (kept_pencils : ℕ) (num_friends : ℕ) (pencils_per_friend : ℕ) : 
  total_boxes = 10 →
  kept_pencils = 10 →
  num_friends = 5 →
  pencils_per_friend = 8 →
  (total_boxes * (kept_pencils + num_friends * pencils_per_friend)) / total_boxes = 5 :=
by sorry

end NUMINAMATH_CALUDE_pencils_per_box_l1781_178170


namespace NUMINAMATH_CALUDE_square_diagonal_quadrilateral_l1781_178146

/-- Given a square with side length a, this theorem proves the properties of a quadrilateral
    formed by the endpoints of a diagonal and the centers of inscribed circles of the two
    isosceles right triangles created by that diagonal. -/
theorem square_diagonal_quadrilateral (a : ℝ) (h : a > 0) :
  ∃ (perimeter area : ℝ),
    perimeter = 4 * a * Real.sqrt (2 - Real.sqrt 2) ∧
    area = a^2 * (Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_quadrilateral_l1781_178146


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1781_178178

theorem fraction_product_simplification :
  (240 : ℚ) / 18 * 7 / 210 * 9 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1781_178178


namespace NUMINAMATH_CALUDE_ratio_G_to_N_l1781_178157

-- Define the variables
variable (N : ℝ) -- Number of non-college graduates
variable (C : ℝ) -- Number of college graduates without a graduate degree
variable (G : ℝ) -- Number of college graduates with a graduate degree

-- Define the conditions
axiom ratio_C_to_N : C = (2/3) * N
axiom prob_G : G / (G + C) = 0.15789473684210525

-- Theorem to prove
theorem ratio_G_to_N : G = (1/8) * N := by sorry

end NUMINAMATH_CALUDE_ratio_G_to_N_l1781_178157


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1781_178169

/-- Represents a geometric sequence -/
structure GeometricSequence where
  firstTerm : ℝ
  ratio : ℝ

/-- Returns the nth term of a geometric sequence -/
def GeometricSequence.nthTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.firstTerm * seq.ratio ^ (n - 1)

theorem geometric_sequence_first_term
  (seq : GeometricSequence)
  (h3 : seq.nthTerm 3 = 720)
  (h7 : seq.nthTerm 7 = 362880) :
  seq.firstTerm = 20 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1781_178169


namespace NUMINAMATH_CALUDE_arrangement_counts_l1781_178104

/-- Represents the number of teachers -/
def num_teachers : Nat := 2

/-- Represents the number of students -/
def num_students : Nat := 4

/-- Represents the total number of people -/
def total_people : Nat := num_teachers + num_students

/-- Calculates the number of arrangements with teachers at the ends -/
def arrangements_teachers_at_ends : Nat :=
  Nat.factorial num_students * Nat.factorial num_teachers

/-- Calculates the number of arrangements with teachers next to each other -/
def arrangements_teachers_together : Nat :=
  Nat.factorial (total_people - 1) * Nat.factorial num_teachers

/-- Calculates the number of arrangements with teachers not next to each other -/
def arrangements_teachers_apart : Nat :=
  Nat.factorial num_students * (num_students + 1) * (num_students + 1)

/-- Calculates the number of arrangements with two students between teachers -/
def arrangements_two_students_between : Nat :=
  (Nat.factorial num_students / (Nat.factorial 2 * Nat.factorial (num_students - 2))) *
  Nat.factorial num_teachers * Nat.factorial 3

theorem arrangement_counts :
  arrangements_teachers_at_ends = 48 ∧
  arrangements_teachers_together = 240 ∧
  arrangements_teachers_apart = 480 ∧
  arrangements_two_students_between = 144 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_counts_l1781_178104


namespace NUMINAMATH_CALUDE_gcd_102_238_l1781_178173

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l1781_178173


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1781_178183

/-- Represents the sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The sum of interior numbers in the sixth row of Pascal's Triangle -/
def sixth_row_sum : ℕ := 30

/-- Theorem: If the sum of interior numbers in the sixth row of Pascal's Triangle is 30,
    then the sum of interior numbers in the eighth row is 126 -/
theorem pascal_triangle_interior_sum :
  interior_sum 6 = sixth_row_sum → interior_sum 8 = 126 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1781_178183


namespace NUMINAMATH_CALUDE_sqrt_difference_l1781_178143

theorem sqrt_difference : Real.sqrt 81 - Real.sqrt 144 = -7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_l1781_178143


namespace NUMINAMATH_CALUDE_triangle_and_circle_problem_l1781_178103

-- Define the curve C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := x^2 + (y + Real.sqrt 3)^2 = 1

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (on_C₁ : C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₁ C.1 C.2)
  (counterclockwise : sorry)  -- We would need to define this properly
  (A_coord : A = (2, 0))

-- State the theorem
theorem triangle_and_circle_problem (ABC : Triangle) :
  ABC.B = (-1, Real.sqrt 3) ∧
  ABC.C = (-1, -Real.sqrt 3) ∧
  ∀ (P : ℝ × ℝ), C₂ P.1 P.2 →
    8 ≤ ((P.1 - ABC.B.1)^2 + (P.2 - ABC.B.2)^2) +
        ((P.1 - ABC.C.1)^2 + (P.2 - ABC.C.2)^2) ∧
    ((P.1 - ABC.B.1)^2 + (P.2 - ABC.B.2)^2) +
    ((P.1 - ABC.C.1)^2 + (P.2 - ABC.C.2)^2) ≤ 24 :=
sorry

end NUMINAMATH_CALUDE_triangle_and_circle_problem_l1781_178103


namespace NUMINAMATH_CALUDE_vector_simplification_l1781_178152

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification 
  (A B C D : V) 
  (h1 : A - C = A - B - (C - B)) 
  (h2 : B - D = B - C - (D - C)) : 
  A - C - (B - D) + (C - D) - (A - B) = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_simplification_l1781_178152


namespace NUMINAMATH_CALUDE_pathway_width_l1781_178139

theorem pathway_width (r₁ r₂ : ℝ) (h₁ : r₁ > r₂) (h₂ : 2 * π * r₁ - 2 * π * r₂ = 20 * π) : 
  r₁ - r₂ + 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_pathway_width_l1781_178139


namespace NUMINAMATH_CALUDE_bottom_right_height_l1781_178150

/-- Represents a rectangle with area and height -/
structure Rectangle where
  area : ℝ
  height : Option ℝ

/-- Represents the layout of six rectangles -/
structure RectangleLayout where
  topLeft : Rectangle
  topMiddle : Rectangle
  topRight : Rectangle
  bottomLeft : Rectangle
  bottomMiddle : Rectangle
  bottomRight : Rectangle

/-- Given the layout of rectangles, prove the height of the bottom right rectangle is 5 -/
theorem bottom_right_height (layout : RectangleLayout) :
  layout.topLeft.area = 18 ∧
  layout.bottomLeft.area = 12 ∧
  layout.bottomMiddle.area = 16 ∧
  layout.topMiddle.area = 32 ∧
  layout.topRight.area = 48 ∧
  layout.bottomRight.area = 30 ∧
  layout.topLeft.height = some 6 →
  layout.bottomRight.height = some 5 := by
  sorry

#check bottom_right_height

end NUMINAMATH_CALUDE_bottom_right_height_l1781_178150


namespace NUMINAMATH_CALUDE_max_reflections_l1781_178128

/-- Represents the angle between lines AD and CD in degrees -/
def angle_CDA : ℝ := 12

/-- Represents the number of reflections -/
def n : ℕ := 7

/-- Theorem stating that n is the maximum number of reflections possible -/
theorem max_reflections (angle : ℝ) (num_reflections : ℕ) :
  angle = angle_CDA →
  num_reflections = n →
  (∀ m : ℕ, m > num_reflections → angle * m > 90) ∧
  angle * num_reflections ≤ 90 :=
sorry

#check max_reflections

end NUMINAMATH_CALUDE_max_reflections_l1781_178128


namespace NUMINAMATH_CALUDE_choose_marbles_eq_990_l1781_178124

/-- The number of ways to choose 5 marbles out of 15, where exactly 2 are chosen from a set of 4 special marbles -/
def choose_marbles : ℕ :=
  let total_marbles : ℕ := 15
  let special_marbles : ℕ := 4
  let choose_total : ℕ := 5
  let choose_special : ℕ := 2
  let normal_marbles : ℕ := total_marbles - special_marbles
  let choose_normal : ℕ := choose_total - choose_special
  (Nat.choose special_marbles choose_special) * (Nat.choose normal_marbles choose_normal)

theorem choose_marbles_eq_990 : choose_marbles = 990 := by
  sorry

end NUMINAMATH_CALUDE_choose_marbles_eq_990_l1781_178124


namespace NUMINAMATH_CALUDE_multiple_identification_l1781_178192

/-- Given two integers a and b that are multiples of n, and q is the set of consecutive integers
    between a and b (inclusive), prove that if q contains 11 multiples of n and 21 multiples of 7,
    then n = 14. -/
theorem multiple_identification (a b n : ℕ) (q : Finset ℕ) (h1 : a ∣ n) (h2 : b ∣ n)
    (h3 : q = Finset.Icc a b) (h4 : (q.filter (· ∣ n)).card = 11)
    (h5 : (q.filter (· ∣ 7)).card = 21) : n = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiple_identification_l1781_178192


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1781_178194

theorem sin_2alpha_value (α : Real) :
  2 * Real.cos (2 * α) = Real.sin (π / 4 - α) →
  Real.sin (2 * α) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1781_178194


namespace NUMINAMATH_CALUDE_special_line_equation_l1781_178189

/-- A line passing through (-2, 2) forming a triangle with area 1 with the coordinate axes -/
structure SpecialLine where
  /-- Slope of the line -/
  k : ℝ
  /-- The line passes through (-2, 2) -/
  passes_through : 2 = k * (-2) + 2
  /-- The area of the triangle formed with the axes is 1 -/
  triangle_area : |4 + 2/k + 2*k| = 1

/-- The equation of a SpecialLine is either x + 2y - 2 = 0 or 2x + y + 2 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.k = -1/2 ∧ ∀ x y, x + 2*y - 2 = 0 ↔ y = l.k * x + 2) ∨
  (l.k = -2 ∧ ∀ x y, 2*x + y + 2 = 0 ↔ y = l.k * x + 2) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l1781_178189


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l1781_178159

theorem recurring_decimal_sum : 
  let x := 1 / 3
  let y := 5 / 999
  let z := 7 / 9999
  x + y + z = 10170 / 29997 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l1781_178159


namespace NUMINAMATH_CALUDE_problem_solution_l1781_178138

theorem problem_solution (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : 3 ≤ y ∧ y ≤ 5) :
  (4 ≤ x + y ∧ x + y ≤ 8) ∧
  (∀ a b, (1 ≤ a ∧ a ≤ 3 ∧ 3 ≤ b ∧ b ≤ 5) → x + y + 1/x + 16/y ≤ a + b + 1/a + 16/b) ∧
  (∃ a b, (1 ≤ a ∧ a ≤ 3 ∧ 3 ≤ b ∧ b ≤ 5) ∧ x + y + 1/x + 16/y = a + b + 1/a + 16/b ∧ a + b + 1/a + 16/b = 10) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l1781_178138


namespace NUMINAMATH_CALUDE_coin_flips_count_l1781_178127

theorem coin_flips_count (heads : ℕ) (tails : ℕ) : heads = 65 → tails = heads + 81 → heads + tails = 211 := by
  sorry

end NUMINAMATH_CALUDE_coin_flips_count_l1781_178127


namespace NUMINAMATH_CALUDE_correct_reasoning_statements_l1781_178186

/-- Represents different types of reasoning -/
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
  | PartToWhole
  | GeneralToGeneral
  | GeneralToSpecific
  | SpecificToGeneral
  | SpecificToSpecific

/-- Defines the correct reasoning direction for each reasoning type -/
def correct_reasoning (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

/-- Theorem stating the correct reasoning directions for each type -/
theorem correct_reasoning_statements :
  (correct_reasoning ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (correct_reasoning ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (correct_reasoning ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end NUMINAMATH_CALUDE_correct_reasoning_statements_l1781_178186


namespace NUMINAMATH_CALUDE_intersection_line_canonical_equations_l1781_178136

/-- The canonical equations of the line formed by the intersection of two planes -/
theorem intersection_line_canonical_equations 
  (plane1 : ℝ → ℝ → ℝ → Prop) 
  (plane2 : ℝ → ℝ → ℝ → Prop) 
  (canonical_eq : ℝ → ℝ → ℝ → Prop) : 
  (∀ x y z, plane1 x y z ↔ 3*x + 3*y - 2*z - 1 = 0) →
  (∀ x y z, plane2 x y z ↔ 2*x - 3*y + z + 6 = 0) →
  (∀ x y z, canonical_eq x y z ↔ (x + 1)/(-3) = (y - 4/3)/(-7) ∧ (y - 4/3)/(-7) = z/(-15)) →
  ∀ x y z, (plane1 x y z ∧ plane2 x y z) ↔ canonical_eq x y z :=
sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_equations_l1781_178136


namespace NUMINAMATH_CALUDE_matrix_power_difference_l1781_178134

/-- Given a 2x2 matrix B, prove that B^30 - 3B^29 equals the specified result -/
theorem matrix_power_difference (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B = ![![2, 4], ![0, 1]]) : 
  B^30 - 3 * B^29 = ![![-2, 0], ![0, 2]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_difference_l1781_178134


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1781_178131

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1781_178131


namespace NUMINAMATH_CALUDE_triples_divisible_by_1000_l1781_178110

/-- The number of ordered triples (a,b,c) in {1, ..., 2016}³ such that a² + b² + c² ≡ 0 (mod 2017) is divisible by 1000. -/
theorem triples_divisible_by_1000 : ∃ N : ℕ,
  (N = (Finset.filter (fun (t : ℕ × ℕ × ℕ) =>
    let (a, b, c) := t
    1 ≤ a ∧ a ≤ 2016 ∧
    1 ≤ b ∧ b ≤ 2016 ∧
    1 ≤ c ∧ c ≤ 2016 ∧
    (a^2 + b^2 + c^2) % 2017 = 0)
    (Finset.product (Finset.range 2016) (Finset.product (Finset.range 2016) (Finset.range 2016)))).card) ∧
  N % 1000 = 0 :=
by sorry

end NUMINAMATH_CALUDE_triples_divisible_by_1000_l1781_178110


namespace NUMINAMATH_CALUDE_johnny_total_planks_l1781_178108

/-- Represents the number of planks needed for a table surface -/
def surface_planks (table_type : String) : ℕ :=
  match table_type with
  | "small" => 3
  | "medium" => 5
  | "large" => 7
  | _ => 0

/-- Represents the number of planks needed for table legs -/
def leg_planks : ℕ := 4

/-- Calculates the total planks needed for a given number of tables of a specific type -/
def planks_for_table_type (table_type : String) (num_tables : ℕ) : ℕ :=
  num_tables * (surface_planks table_type + leg_planks)

/-- Theorem: The total number of planks needed for Johnny's tables is 50 -/
theorem johnny_total_planks : 
  planks_for_table_type "small" 3 + 
  planks_for_table_type "medium" 2 + 
  planks_for_table_type "large" 1 = 50 := by
  sorry


end NUMINAMATH_CALUDE_johnny_total_planks_l1781_178108


namespace NUMINAMATH_CALUDE_halfway_point_l1781_178180

theorem halfway_point (a b : ℚ) (ha : a = 1/8) (hb : b = 3/10) :
  (a + b) / 2 = 17/80 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_l1781_178180


namespace NUMINAMATH_CALUDE_apples_per_adult_l1781_178171

def total_apples : ℕ := 450
def num_children : ℕ := 33
def apples_per_child : ℕ := 10
def num_adults : ℕ := 40

theorem apples_per_adult :
  (total_apples - num_children * apples_per_child) / num_adults = 3 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_adult_l1781_178171


namespace NUMINAMATH_CALUDE_half_coverage_days_l1781_178101

/-- Represents the number of days it takes for the lily pad patch to cover the entire lake -/
def full_coverage_days : ℕ := 48

/-- Represents the growth factor of the lily pad patch per day -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that the number of days required to cover half the lake
    is one day less than the number of days required to cover the full lake -/
theorem half_coverage_days : 
  full_coverage_days - 1 = full_coverage_days - (daily_growth_factor.log 2) := by
  sorry

end NUMINAMATH_CALUDE_half_coverage_days_l1781_178101


namespace NUMINAMATH_CALUDE_table_seats_l1781_178112

/-- The number of people sitting at the table -/
def n : ℕ := 10

/-- The sum of seeds taken in the first round -/
def first_round_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of seeds taken in the second round -/
def second_round_sum (n : ℕ) : ℕ := n * (n + 1) / 2 + n^2

/-- The theorem stating that n = 10 satisfies the conditions -/
theorem table_seats : 
  (second_round_sum n - first_round_sum n = 100) ∧ 
  (∀ m : ℕ, second_round_sum m - first_round_sum m = 100 → m = n) := by
  sorry

#check table_seats

end NUMINAMATH_CALUDE_table_seats_l1781_178112


namespace NUMINAMATH_CALUDE_semicircle_problem_l1781_178100

theorem semicircle_problem (M : ℕ) (r : ℝ) (h_positive : r > 0) : 
  (M * π * r^2 / 2) / (π * r^2 * (M^2 - M) / 2) = 1/4 → M = 5 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_problem_l1781_178100


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1781_178133

theorem rectangle_perimeter (a b : ℤ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  a * b = 4 * (2 * a + 2 * b) - 12 → 
  2 * (a + b) = 72 ∨ 2 * (a + b) = 100 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1781_178133


namespace NUMINAMATH_CALUDE_sin_shift_l1781_178126

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l1781_178126


namespace NUMINAMATH_CALUDE_digit1Sequence_1482_to_1484_l1781_178158

/-- A sequence of positive integers starting with digit 1 in increasing order -/
def digit1Sequence : ℕ → ℕ := sorry

/-- The nth digit in the concatenated sequence of digit1Sequence -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1482nd, 1483rd, and 1484th digits -/
def targetNumber : ℕ := 100 * (nthDigit 1482) + 10 * (nthDigit 1483) + (nthDigit 1484)

theorem digit1Sequence_1482_to_1484 : targetNumber = 129 := by sorry

end NUMINAMATH_CALUDE_digit1Sequence_1482_to_1484_l1781_178158


namespace NUMINAMATH_CALUDE_pasture_fence_posts_l1781_178149

/-- Calculates the number of posts needed for a given length of fence -/
def posts_for_length (length : ℕ) (post_spacing : ℕ) : ℕ :=
  (length / post_spacing) + 1

/-- The pasture dimensions -/
def pasture_width : ℕ := 36
def pasture_length : ℕ := 75

/-- The spacing between posts -/
def post_spacing : ℕ := 15

/-- The total number of posts required for the pasture -/
def total_posts : ℕ :=
  posts_for_length pasture_width post_spacing +
  2 * (posts_for_length pasture_length post_spacing - 1)

theorem pasture_fence_posts :
  total_posts = 14 := by sorry

end NUMINAMATH_CALUDE_pasture_fence_posts_l1781_178149


namespace NUMINAMATH_CALUDE_bottle_caps_difference_l1781_178148

-- Define the number of bottle caps found and thrown away each day
def monday_found : ℕ := 36
def monday_thrown : ℕ := 45
def tuesday_found : ℕ := 58
def tuesday_thrown : ℕ := 30
def wednesday_found : ℕ := 80
def wednesday_thrown : ℕ := 70

-- Define the final number of bottle caps left
def final_caps : ℕ := 65

-- Theorem to prove
theorem bottle_caps_difference :
  (monday_found + tuesday_found + wednesday_found) -
  (monday_thrown + tuesday_thrown + wednesday_thrown) = 29 :=
by sorry

end NUMINAMATH_CALUDE_bottle_caps_difference_l1781_178148


namespace NUMINAMATH_CALUDE_max_fall_time_bound_l1781_178140

/-- Represents the movement rules and conditions for ants on an m × m checkerboard. -/
structure AntCheckerboard (m : ℕ) :=
  (m_pos : m > 0)
  (board_size : Fin m → Fin m → Bool)
  (ant_positions : Set (Fin m × Fin m))
  (ant_directions : (Fin m × Fin m) → (Int × Int))
  (collision_rules : (Fin m × Fin m) → (Int × Int) → (Int × Int))

/-- The maximum time for the last ant to fall off the board. -/
def max_fall_time (m : ℕ) (board : AntCheckerboard m) : ℚ :=
  3 * m / 2 - 1

/-- Theorem stating that the maximum time for the last ant to fall off is 3m/2 - 1. -/
theorem max_fall_time_bound (m : ℕ) (board : AntCheckerboard m) :
  ∀ (t : ℚ), (∃ (ant : Fin m × Fin m), ant ∈ board.ant_positions) →
  t ≤ max_fall_time m board :=
sorry

end NUMINAMATH_CALUDE_max_fall_time_bound_l1781_178140


namespace NUMINAMATH_CALUDE_gold_coin_percentage_l1781_178119

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beadPercentage : ℝ
  silverCoinPercentage : ℝ
  goldCoinPercentage : ℝ

/-- The urn composition satisfies the given conditions --/
def validUrnComposition (u : UrnComposition) : Prop :=
  u.beadPercentage = 30 ∧
  u.silverCoinPercentage + u.goldCoinPercentage = 70 ∧
  u.silverCoinPercentage = 35

theorem gold_coin_percentage (u : UrnComposition) 
  (h : validUrnComposition u) : u.goldCoinPercentage = 35 := by
  sorry

#check gold_coin_percentage

end NUMINAMATH_CALUDE_gold_coin_percentage_l1781_178119


namespace NUMINAMATH_CALUDE_sector_area_l1781_178182

/-- Given a circular sector with arc length 3π and central angle 3/4π, its area is 6π. -/
theorem sector_area (r : ℝ) (h1 : (3/4) * π * r = 3 * π) : (1/2) * (3/4 * π) * r^2 = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1781_178182


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l1781_178120

theorem triangle_perimeter_bound : 
  ∀ (a b c : ℝ), 
    a = 7 → 
    b = 21 → 
    a + b > c → 
    a + c > b → 
    b + c > a → 
    a + b + c < 56 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l1781_178120


namespace NUMINAMATH_CALUDE_gross_revenue_increase_l1781_178115

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_reduction_rate : ℝ)
  (quantity_increase_rate : ℝ)
  (h1 : price_reduction_rate = 0.2)
  (h2 : quantity_increase_rate = 0.5)
  : (((1 - price_reduction_rate) * (1 + quantity_increase_rate) - 1) * 100 = 20) := by
  sorry

end NUMINAMATH_CALUDE_gross_revenue_increase_l1781_178115


namespace NUMINAMATH_CALUDE_mode_median_constant_l1781_178187

/-- Represents the age distribution of a club --/
structure AgeDistribution where
  age13 : ℕ
  age14 : ℕ
  age15 : ℕ
  age16 : ℕ
  age17 : ℕ
  total : ℕ
  sum_eq_total : age13 + age14 + age15 + age16 + age17 = total

/-- The age distribution of the club --/
def clubDistribution (x : ℕ) : AgeDistribution where
  age13 := 5
  age14 := 12
  age15 := x
  age16 := 11 - x
  age17 := 2
  total := 30
  sum_eq_total := by sorry

/-- The mode of the age distribution --/
def mode (d : AgeDistribution) : ℕ := 
  max d.age13 (max d.age14 (max d.age15 (max d.age16 d.age17)))

/-- The median of the age distribution --/
def median (d : AgeDistribution) : ℚ := 14

theorem mode_median_constant (x : ℕ) : 
  mode (clubDistribution x) = 14 ∧ median (clubDistribution x) = 14 := by sorry

end NUMINAMATH_CALUDE_mode_median_constant_l1781_178187


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1781_178105

theorem perpendicular_lines_b_value (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y + 1 = 0 ∧ 3*x + b*y + 5 = 0 → 
   ((-a/2) * (-3/b) = -1)) →
  b = -3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1781_178105
