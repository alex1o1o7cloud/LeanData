import Mathlib

namespace solve_system_l2781_278180

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x + 4 * y = 0) 
  (eq2 : y - 3 = x) : 
  5 * y = 45 / 7 := by
  sorry

end solve_system_l2781_278180


namespace position_3_1_is_B_l2781_278131

-- Define the grid type
def Grid := Fin 5 → Fin 5 → Char

-- Define the valid letters
def ValidLetter (c : Char) : Prop := c ∈ ['A', 'B', 'C', 'D', 'E']

-- Define the property of a valid grid
def ValidGrid (g : Grid) : Prop :=
  (∀ r c, ValidLetter (g r c)) ∧
  (∀ r, ∀ c₁ c₂, c₁ ≠ c₂ → g r c₁ ≠ g r c₂) ∧
  (∀ c, ∀ r₁ r₂, r₁ ≠ r₂ → g r₁ c ≠ g r₂ c) ∧
  (∀ i j, i ≠ j → g i i ≠ g j j) ∧
  (∀ i j, i ≠ j → g i (4 - i) ≠ g j (4 - j))

-- Define the theorem
theorem position_3_1_is_B (g : Grid) (h : ValidGrid g)
  (h1 : g 0 0 = 'A') (h2 : g 3 0 = 'D') (h3 : g 4 0 = 'E') :
  g 2 0 = 'B' := by
  sorry

end position_3_1_is_B_l2781_278131


namespace evaluate_expression_l2781_278144

theorem evaluate_expression : 7 - 5 * (6 - 2^3) * 3 = -23 := by
  sorry

end evaluate_expression_l2781_278144


namespace burger_share_inches_l2781_278152

/-- The length of a foot in inches -/
def foot_in_inches : ℝ := 12

/-- The length of the burger in feet -/
def burger_length_feet : ℝ := 1

/-- The number of people sharing the burger -/
def num_people : ℕ := 2

/-- Theorem: Each person's share of a foot-long burger is 6 inches when shared equally between two people -/
theorem burger_share_inches : 
  (burger_length_feet * foot_in_inches) / num_people = 6 := by sorry

end burger_share_inches_l2781_278152


namespace shopping_theorem_l2781_278194

def shopping_problem (shoe_price : ℝ) (shoe_discount : ℝ) (shirt_price : ℝ) (num_shirts : ℕ) (final_discount : ℝ) : Prop :=
  let discounted_shoe_price := shoe_price * (1 - shoe_discount)
  let total_shirt_price := shirt_price * num_shirts
  let subtotal := discounted_shoe_price + total_shirt_price
  let final_price := subtotal * (1 - final_discount)
  final_price = 285

theorem shopping_theorem :
  shopping_problem 200 0.30 80 2 0.05 := by
  sorry

end shopping_theorem_l2781_278194


namespace max_profit_is_1200_l2781_278195

/-- Represents the cost and profit calculation for a shopping mall's purchasing plan. -/
structure ShoppingMall where
  cost_A : ℝ  -- Cost price of good A
  cost_B : ℝ  -- Cost price of good B
  sell_A : ℝ  -- Selling price of good A
  sell_B : ℝ  -- Selling price of good B
  total_units : ℕ  -- Total units to purchase

/-- Calculates the profit for a given purchasing plan. -/
def profit (sm : ShoppingMall) (units_A : ℕ) : ℝ :=
  let units_B := sm.total_units - units_A
  (sm.sell_A * units_A + sm.sell_B * units_B) - (sm.cost_A * units_A + sm.cost_B * units_B)

/-- Theorem stating that the maximum profit is $1200 under the given conditions. -/
theorem max_profit_is_1200 (sm : ShoppingMall) 
  (h1 : sm.cost_A + 3 * sm.cost_B = 240)
  (h2 : 2 * sm.cost_A + sm.cost_B = 130)
  (h3 : sm.sell_A = 40)
  (h4 : sm.sell_B = 90)
  (h5 : sm.total_units = 100)
  : ∃ (units_A : ℕ), 
    units_A ≥ 4 * (sm.total_units - units_A) ∧ 
    ∀ (x : ℕ), x ≥ 4 * (sm.total_units - x) → profit sm units_A ≥ profit sm x :=
by sorry

end max_profit_is_1200_l2781_278195


namespace homework_ratio_l2781_278145

theorem homework_ratio (total : ℕ) (algebra_percent : ℚ) (linear_eq : ℕ) : 
  total = 140 →
  algebra_percent = 40/100 →
  linear_eq = 28 →
  (linear_eq : ℚ) / (algebra_percent * total) = 1/2 := by
sorry

end homework_ratio_l2781_278145


namespace gcd_lcm_product_24_36_l2781_278185

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end gcd_lcm_product_24_36_l2781_278185


namespace least_divisible_by_7_11_13_l2781_278110

theorem least_divisible_by_7_11_13 : ∃ n : ℕ, n > 0 ∧ 7 ∣ n ∧ 11 ∣ n ∧ 13 ∣ n ∧ ∀ m : ℕ, m > 0 → 7 ∣ m → 11 ∣ m → 13 ∣ m → n ≤ m :=
by sorry

end least_divisible_by_7_11_13_l2781_278110


namespace garden_area_increase_l2781_278157

/-- Represents a rectangular garden with given length and width -/
structure RectGarden where
  length : ℝ
  width : ℝ

/-- Represents a square garden with given side length -/
structure SquareGarden where
  side : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def RectGarden.perimeter (g : RectGarden) : ℝ :=
  2 * (g.length + g.width)

/-- Calculates the area of a rectangular garden -/
def RectGarden.area (g : RectGarden) : ℝ :=
  g.length * g.width

/-- Calculates the perimeter of a square garden -/
def SquareGarden.perimeter (g : SquareGarden) : ℝ :=
  4 * g.side

/-- Calculates the area of a square garden -/
def SquareGarden.area (g : SquareGarden) : ℝ :=
  g.side * g.side

/-- Theorem: Changing a 60 ft by 20 ft rectangular garden to a square garden 
    with the same perimeter increases the area by 400 square feet -/
theorem garden_area_increase :
  let rect := RectGarden.mk 60 20
  let square := SquareGarden.mk (rect.perimeter / 4)
  square.area - rect.area = 400 := by
  sorry


end garden_area_increase_l2781_278157


namespace system_solution_l2781_278190

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  2 * x - 3 * y = 5 ∧ 4 * x - y = 5

-- Theorem statement
theorem system_solution :
  ∃! (x y : ℝ), system x y ∧ x = 1 ∧ y = -1 :=
sorry

end system_solution_l2781_278190


namespace system_solution_triangle_side_range_l2781_278107

-- Problem 1
theorem system_solution (m : ℤ) : 
  (∃ x y : ℝ, 2*x + y = -3*m + 2 ∧ x + 2*y = 4 ∧ x + y > -3/2) ↔ 
  (m = 1 ∨ m = 2 ∨ m = 3) :=
sorry

-- Problem 2
theorem triangle_side_range (a b c : ℝ) :
  (a^2 + b^2 = 10*a + 8*b - 41) ∧
  (c ≥ a ∧ c ≥ b) →
  (5 ≤ c ∧ c < 9) :=
sorry

end system_solution_triangle_side_range_l2781_278107


namespace epidemic_competition_theorem_l2781_278166

/-- Represents a participant in the competition -/
structure Participant where
  first_round_prob : ℚ
  second_round_prob : ℚ

/-- Calculates the probability of a participant winning both rounds -/
def win_prob (p : Participant) : ℚ :=
  p.first_round_prob * p.second_round_prob

/-- Calculates the probability of at least one participant winning -/
def at_least_one_wins (p1 p2 : Participant) : ℚ :=
  1 - (1 - win_prob p1) * (1 - win_prob p2)

theorem epidemic_competition_theorem 
  (A B : Participant)
  (h_A_first : A.first_round_prob = 5/6)
  (h_A_second : A.second_round_prob = 2/3)
  (h_B_first : B.first_round_prob = 3/5)
  (h_B_second : B.second_round_prob = 3/4) :
  win_prob A > win_prob B ∧ at_least_one_wins A B = 34/45 := by
  sorry

end epidemic_competition_theorem_l2781_278166


namespace parabola_transformation_l2781_278122

/-- Represents a parabola of the form y = (x + a)^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Applies a horizontal shift to a parabola -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a - shift, b := p.b }

/-- Applies a vertical shift to a parabola -/
def vertical_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, b := p.b + shift }

/-- The initial parabola y = (x + 2)^2 + 3 -/
def initial_parabola : Parabola := { a := 2, b := 3 }

/-- The final parabola after transformations -/
def final_parabola : Parabola := { a := -1, b := 1 }

theorem parabola_transformation :
  (vertical_shift (horizontal_shift initial_parabola 3) (-2)) = final_parabola := by
  sorry

end parabola_transformation_l2781_278122


namespace hyperbola_focus_distance_l2781_278172

/-- The hyperbola equation -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

/-- The distance from a point to the left focus -/
def dist_to_left_focus (x y : ℝ) : ℝ := sorry

/-- The distance from a point to the right focus -/
def dist_to_right_focus (x y : ℝ) : ℝ := sorry

/-- Theorem: If P(x,y) is on the right branch of the hyperbola and
    its distance to the left focus is 12, then its distance to the right focus is 4 -/
theorem hyperbola_focus_distance (x y : ℝ) :
  is_on_hyperbola x y ∧ x > 0 ∧ dist_to_left_focus x y = 12 →
  dist_to_right_focus x y = 4 := by sorry

end hyperbola_focus_distance_l2781_278172


namespace comic_books_problem_l2781_278156

theorem comic_books_problem (sold : ℕ) (left : ℕ) (h1 : sold = 65) (h2 : left = 25) :
  sold + left = 90 := by
  sorry

end comic_books_problem_l2781_278156


namespace expression_evaluation_l2781_278118

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := 1
  let expr := -1/3 * (a^3*b - a*b) + a*b^3 - (a*b - b)/2 - 1/2*b + 1/3*a^3*b
  expr = 5/3 := by sorry

end expression_evaluation_l2781_278118


namespace diane_poker_loss_l2781_278139

/-- The total amount of money Diane lost in her poker game -/
def total_loss (initial_amount won_amount final_debt : ℝ) : ℝ :=
  initial_amount + won_amount + final_debt

/-- Theorem stating that Diane's total loss is $215 -/
theorem diane_poker_loss :
  let initial_amount : ℝ := 100
  let won_amount : ℝ := 65
  let final_debt : ℝ := 50
  total_loss initial_amount won_amount final_debt = 215 := by
  sorry

end diane_poker_loss_l2781_278139


namespace max_ratio_squared_l2781_278187

theorem max_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) : 
  (∃ ρ : ℝ, ρ > 0 ∧ ρ^2 = 2 ∧ 
    (∀ r : ℝ, r > ρ → 
      ¬∃ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧ 
        a^2 + y^2 = (a - x)^2 + (b - y)^2 ∧ 
        a^2 + y^2 = b^2 - x^2 + y^2 ∧ 
        r = a / b)) := by
  sorry

end max_ratio_squared_l2781_278187


namespace johns_allowance_l2781_278146

theorem johns_allowance (A : ℚ) : 
  (7/15 + 3/10 + 1/6 : ℚ) * A +  -- Spent on arcade, books, and clothes
  2/5 * (1 - (7/15 + 3/10 + 1/6 : ℚ)) * A +  -- Spent at toy store
  (6/5 : ℚ) = A  -- Last $1.20 spent at candy store (represented as 6/5)
  → A = 30 := by
sorry

end johns_allowance_l2781_278146


namespace min_value_reciprocal_sum_l2781_278132

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1 / x₀ + 4 / y₀ = 9 :=
sorry

end min_value_reciprocal_sum_l2781_278132


namespace village_population_growth_l2781_278149

theorem village_population_growth (
  adult_percentage : Real)
  (child_percentage : Real)
  (employed_adult_percentage : Real)
  (unemployed_adult_percentage : Real)
  (employed_adult_population : ℕ)
  (adult_growth_rate : Real)
  (h1 : adult_percentage = 0.6)
  (h2 : child_percentage = 0.4)
  (h3 : employed_adult_percentage = 0.7)
  (h4 : unemployed_adult_percentage = 0.3)
  (h5 : employed_adult_population = 18000)
  (h6 : adult_growth_rate = 0.05)
  (h7 : adult_percentage + child_percentage = 1)
  (h8 : employed_adult_percentage + unemployed_adult_percentage = 1) :
  ∃ (new_total_population : ℕ), new_total_population = 45000 := by
  sorry

#check village_population_growth

end village_population_growth_l2781_278149


namespace fold_angle_is_36_degrees_l2781_278103

/-- The angle of fold that creates a regular decagon when a piece of paper is folded and cut,
    given that all vertices except one lie on a circle centered at that vertex,
    and the angle between adjacent vertices at the center is 144°. -/
def fold_angle_for_decagon : ℝ := sorry

/-- The internal angle of a regular decagon. -/
def decagon_internal_angle : ℝ := sorry

/-- Theorem stating that the fold angle for creating a regular decagon
    under the given conditions is 36°. -/
theorem fold_angle_is_36_degrees :
  fold_angle_for_decagon = 36 * (π / 180) ∧
  0 < fold_angle_for_decagon ∧
  fold_angle_for_decagon < π / 2 ∧
  decagon_internal_angle = 144 * (π / 180) :=
sorry

end fold_angle_is_36_degrees_l2781_278103


namespace cone_generatrix_length_l2781_278167

/-- Given a cone with a 45° angle between the generatrix and base, and height 1,
    the length of the generatrix is √2. -/
theorem cone_generatrix_length 
  (angle : ℝ) 
  (height : ℝ) 
  (h_angle : angle = Real.pi / 4) 
  (h_height : height = 1) : 
  Real.sqrt 2 = 
    Real.sqrt (height ^ 2 + height ^ 2) := by
  sorry

end cone_generatrix_length_l2781_278167


namespace roots_product_l2781_278113

/-- Given that x₁ = ∛(17 - (27/4)√6) and x₂ = ∛(17 + (27/4)√6) are roots of x² - ax + b = 0, prove that ab = 10 -/
theorem roots_product (a b : ℝ) : 
  let x₁ : ℝ := (17 - (27/4) * Real.sqrt 6) ^ (1/3)
  let x₂ : ℝ := (17 + (27/4) * Real.sqrt 6) ^ (1/3)
  (x₁ ^ 2 - a * x₁ + b = 0) ∧ (x₂ ^ 2 - a * x₂ + b = 0) → a * b = 10 := by
  sorry


end roots_product_l2781_278113


namespace painted_cubes_l2781_278174

theorem painted_cubes (n : ℕ) (h : n = 10) : 
  n^3 - (n - 2)^3 = 488 := by
  sorry

end painted_cubes_l2781_278174


namespace percentage_problem_l2781_278186

/-- Given that P% of 820 is 20 less than 15% of 1500, prove that P = 25 -/
theorem percentage_problem (P : ℝ) (h : P / 100 * 820 = 15 / 100 * 1500 - 20) : P = 25 := by
  sorry

end percentage_problem_l2781_278186


namespace green_blue_difference_after_border_l2781_278104

/-- Represents the number of tiles in a hexagonal figure --/
structure HexFigure where
  blue : ℕ
  green : ℕ

/-- Adds a double-layer border of green tiles to a hexagonal figure --/
def addDoubleBorder (fig : HexFigure) : HexFigure :=
  { blue := fig.blue,
    green := fig.green + 12 + 18 }

/-- The initial hexagonal figure --/
def initialFigure : HexFigure :=
  { blue := 20, green := 10 }

/-- Theorem stating the difference between green and blue tiles after adding a double border --/
theorem green_blue_difference_after_border :
  let newFigure := addDoubleBorder initialFigure
  (newFigure.green - newFigure.blue) = 20 := by
  sorry

end green_blue_difference_after_border_l2781_278104


namespace parallel_line_equation_l2781_278126

/-- Given a line L passing through the point (1, 0) and parallel to the line x - 2y - 2 = 0,
    prove that the equation of L is x - 2y - 1 = 0 -/
theorem parallel_line_equation :
  ∀ (L : Set (ℝ × ℝ)),
  (∀ p ∈ L, ∃ x y : ℝ, p = (x, y) ∧ x - 2*y - 1 = 0) →
  (1, 0) ∈ L →
  (∀ p q : ℝ × ℝ, p ∈ L → q ∈ L → p.1 - q.1 = 2*(p.2 - q.2)) →
  ∀ p ∈ L, ∃ x y : ℝ, p = (x, y) ∧ x - 2*y - 1 = 0 :=
by sorry

end parallel_line_equation_l2781_278126


namespace equation_solution_l2781_278164

theorem equation_solution (x : ℝ) :
  x^2 + x + 1 = 1 / (x^2 - x + 1) ∧ x^2 - x + 1 ≠ 0 → x = 1 ∨ x = -1 := by
  sorry

end equation_solution_l2781_278164


namespace solve_equation_l2781_278154

theorem solve_equation (y : ℝ) : 
  5 * y^(1/4) - 3 * (y / y^(3/4)) = 9 + y^(1/4) ↔ y = 6561 :=
by sorry

end solve_equation_l2781_278154


namespace probability_two_black_two_white_l2781_278176

def total_balls : ℕ := 18
def black_balls : ℕ := 10
def white_balls : ℕ := 8
def drawn_balls : ℕ := 4
def drawn_black : ℕ := 2
def drawn_white : ℕ := 2

theorem probability_two_black_two_white :
  (Nat.choose black_balls drawn_black * Nat.choose white_balls drawn_white) /
  Nat.choose total_balls drawn_balls = 7 / 17 := by
  sorry

end probability_two_black_two_white_l2781_278176


namespace fraction_gt_one_not_equivalent_to_a_gt_b_l2781_278153

theorem fraction_gt_one_not_equivalent_to_a_gt_b :
  ¬(∀ (a b : ℝ), a / b > 1 ↔ a > b) :=
by
  sorry

end fraction_gt_one_not_equivalent_to_a_gt_b_l2781_278153


namespace xiaoming_savings_l2781_278127

/-- Represents the number of coins in each pile -/
structure CoinCount where
  pile1_2cent : ℕ
  pile1_5cent : ℕ
  pile2_2cent : ℕ
  pile2_5cent : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCount) : ℕ :=
  2 * (coins.pile1_2cent + coins.pile2_2cent) + 5 * (coins.pile1_5cent + coins.pile2_5cent)

theorem xiaoming_savings (coins : CoinCount) :
  coins.pile1_2cent = coins.pile1_5cent →
  2 * coins.pile2_2cent = 5 * coins.pile2_5cent →
  2 * coins.pile1_2cent + 5 * coins.pile1_5cent = 2 * coins.pile2_2cent + 5 * coins.pile2_5cent →
  500 ≤ totalValue coins →
  totalValue coins ≤ 600 →
  totalValue coins = 560 := by
  sorry

#check xiaoming_savings

end xiaoming_savings_l2781_278127


namespace polynomial_sum_equals_256_l2781_278112

theorem polynomial_sum_equals_256 
  (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (3 - x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) :
  a - a₁ + a₂ - a₃ + a₄ = 256 := by
  sorry

end polynomial_sum_equals_256_l2781_278112


namespace real_root_quadratic_l2781_278119

theorem real_root_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - (1 - Complex.I) * x + m + 2 * Complex.I = 0) → m = -6 := by
  sorry

end real_root_quadratic_l2781_278119


namespace mean_height_is_correct_l2781_278123

/-- Represents the heights of players on a basketball team -/
def heights : List Nat := [57, 62, 64, 64, 65, 67, 68, 70, 71, 72, 72, 73, 74, 75, 75]

/-- The number of players on the team -/
def num_players : Nat := heights.length

/-- The sum of all player heights -/
def total_height : Nat := heights.sum

/-- Calculates the mean height of the players -/
def mean_height : Rat := total_height / num_players

theorem mean_height_is_correct : mean_height = 1029 / 15 := by sorry

end mean_height_is_correct_l2781_278123


namespace product_of_three_numbers_l2781_278193

theorem product_of_three_numbers (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ab : a * b = 45 * Real.rpow 3 (1/3))
  (h_ac : a * c = 75 * Real.rpow 3 (1/3))
  (h_bc : b * c = 27 * Real.rpow 3 (1/3)) :
  a * b * c = 135 * Real.sqrt 15 := by
  sorry

end product_of_three_numbers_l2781_278193


namespace negation_of_existence_quadratic_equation_negation_l2781_278178

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem quadratic_equation_negation : 
  (¬∃ x : ℝ, x^2 + 2*x + 3 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 3 ≠ 0) :=
by sorry

end negation_of_existence_quadratic_equation_negation_l2781_278178


namespace square_minus_nine_l2781_278108

theorem square_minus_nine (x : ℤ) (h : x^2 = 1681) : (x + 3) * (x - 3) = 1672 := by
  sorry

end square_minus_nine_l2781_278108


namespace P_on_y_axis_after_move_l2781_278158

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of P -/
def P : Point := ⟨3, 4⟩

/-- Function to move a point left by a given number of units -/
def moveLeft (p : Point) (units : ℝ) : Point :=
  ⟨p.x - units, p.y⟩

/-- Predicate to check if a point is on the y-axis -/
def isOnYAxis (p : Point) : Prop :=
  p.x = 0

/-- Theorem stating that P lands on the y-axis after moving 3 units left -/
theorem P_on_y_axis_after_move : isOnYAxis (moveLeft P 3) := by
  sorry

end P_on_y_axis_after_move_l2781_278158


namespace pizza_coverage_l2781_278148

/-- Represents a circular pizza with cheese circles on it. -/
structure CheesePizza where
  pizza_diameter : ℝ
  cheese_circles_across : ℕ
  total_cheese_circles : ℕ

/-- Calculates the fraction of the pizza covered by cheese. -/
def fraction_covered (pizza : CheesePizza) : ℚ :=
  sorry

/-- Theorem stating the fraction of pizza covered by cheese -/
theorem pizza_coverage (pizza : CheesePizza) 
  (h1 : pizza.pizza_diameter = 15)
  (h2 : pizza.cheese_circles_across = 9)
  (h3 : pizza.total_cheese_circles = 36) : 
  fraction_covered pizza = 5 / 9 := by
  sorry

end pizza_coverage_l2781_278148


namespace stamp_collection_total_l2781_278121

/-- Represents a stamp collection with various categories of stamps. -/
structure StampCollection where
  foreign : ℕ
  old : ℕ
  both_foreign_and_old : ℕ
  neither_foreign_nor_old : ℕ

/-- Calculates the total number of stamps in the collection. -/
def total_stamps (collection : StampCollection) : ℕ :=
  collection.foreign + collection.old - collection.both_foreign_and_old + collection.neither_foreign_nor_old

/-- Theorem stating that the total number of stamps in the given collection is 220. -/
theorem stamp_collection_total :
  ∃ (collection : StampCollection),
    collection.foreign = 90 ∧
    collection.old = 70 ∧
    collection.both_foreign_and_old = 20 ∧
    collection.neither_foreign_nor_old = 60 ∧
    total_stamps collection = 220 := by
  sorry

end stamp_collection_total_l2781_278121


namespace divisibility_of_power_minus_one_l2781_278161

theorem divisibility_of_power_minus_one (n : ℕ) (h : n > 1) :
  ∃ k : ℤ, (n ^ (n - 1) : ℤ) - 1 = k * ((n - 1) ^ 2 : ℤ) := by
  sorry

end divisibility_of_power_minus_one_l2781_278161


namespace root_equation_implies_expression_value_l2781_278155

theorem root_equation_implies_expression_value (a : ℝ) : 
  a^2 + a - 1 = 0 → 2021 - 2*a^2 - 2*a = 2019 := by
  sorry

end root_equation_implies_expression_value_l2781_278155


namespace worker_travel_time_l2781_278130

theorem worker_travel_time (normal_speed : ℝ) (normal_time : ℝ) 
  (h1 : normal_speed > 0) (h2 : normal_time > 0) : 
  (3/4 * normal_speed) * (normal_time + 8) = normal_speed * normal_time → 
  normal_time = 24 := by
  sorry

end worker_travel_time_l2781_278130


namespace unique_solution_f_two_equals_four_l2781_278140

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * (f x) * y + y^2

/-- The theorem stating that x^2 is the only function satisfying the equation -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  ∀ x : ℝ, f x = x^2 :=
sorry

/-- The value of f(2) is 4 -/
theorem f_two_equals_four (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  f 2 = 4 :=
sorry

end unique_solution_f_two_equals_four_l2781_278140


namespace cola_price_is_three_l2781_278182

/-- Represents the cost and quantity of drinks sold in a store --/
structure DrinkSales where
  cola_price : ℝ
  cola_quantity : ℕ
  juice_price : ℝ
  juice_quantity : ℕ
  water_price : ℝ
  water_quantity : ℕ
  total_earnings : ℝ

/-- Theorem stating that the cola price is $3 given the specific sales conditions --/
theorem cola_price_is_three (sales : DrinkSales)
  (h_juice_price : sales.juice_price = 1.5)
  (h_water_price : sales.water_price = 1)
  (h_cola_quantity : sales.cola_quantity = 15)
  (h_juice_quantity : sales.juice_quantity = 12)
  (h_water_quantity : sales.water_quantity = 25)
  (h_total_earnings : sales.total_earnings = 88) :
  sales.cola_price = 3 := by
  sorry

#check cola_price_is_three

end cola_price_is_three_l2781_278182


namespace pages_difference_l2781_278160

/-- The number of pages Person A reads per day -/
def pages_per_day_A : ℕ := 8

/-- The number of pages Person B reads per day -/
def pages_per_day_B : ℕ := 13

/-- The number of days we're considering -/
def days : ℕ := 7

/-- The total number of pages Person A reads in the given number of days -/
def total_pages_A : ℕ := pages_per_day_A * days

/-- The total number of pages Person B reads in the given number of days -/
def total_pages_B : ℕ := pages_per_day_B * days

theorem pages_difference : total_pages_B - total_pages_A = 9 := by
  sorry

end pages_difference_l2781_278160


namespace company_manager_fraction_l2781_278137

/-- Fraction of employees who are managers -/
def manager_fraction (total_employees : ℕ) (total_managers : ℕ) : ℚ :=
  total_managers / total_employees

theorem company_manager_fraction :
  ∀ (total_employees : ℕ) (total_managers : ℕ) (male_employees : ℕ) (male_managers : ℕ),
    total_employees > 0 →
    male_employees > 0 →
    total_employees = 625 + male_employees →
    total_managers = 250 + male_managers →
    manager_fraction total_employees total_managers = manager_fraction 625 250 →
    manager_fraction total_employees total_managers = manager_fraction male_employees male_managers →
    manager_fraction total_employees total_managers = 2 / 5 := by
  sorry

end company_manager_fraction_l2781_278137


namespace intersection_of_A_and_B_l2781_278105

def A : Set ℝ := {x | |x| < 1}
def B : Set ℝ := {x | -2 < x ∧ x < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-1) 0 := by sorry

end intersection_of_A_and_B_l2781_278105


namespace cards_drawn_l2781_278198

theorem cards_drawn (total_cards : ℕ) (face_cards : ℕ) (prob : ℚ) (n : ℕ) : 
  total_cards = 52 →
  face_cards = 12 →
  prob = 12 / 52 →
  (face_cards : ℚ) / n = prob →
  n = total_cards :=
by sorry

end cards_drawn_l2781_278198


namespace ravish_exam_marks_l2781_278124

theorem ravish_exam_marks (pass_percentage : ℚ) (max_marks : ℕ) (fail_margin : ℕ) : 
  pass_percentage = 40 / 100 →
  max_marks = 200 →
  fail_margin = 40 →
  (pass_percentage * max_marks : ℚ) - fail_margin = 40 := by
  sorry

end ravish_exam_marks_l2781_278124


namespace f_minimum_value_g_range_condition_l2781_278199

noncomputable section

def f (x : ℝ) := 2 * x * Real.log x

def g (a x : ℝ) := -x^2 + a*x - 3

theorem f_minimum_value :
  ∃ (m : ℝ), m = 2 / Real.exp 1 ∧ ∀ x > 0, f x ≥ m :=
sorry

theorem g_range_condition (a : ℝ) :
  (∃ x > 0, f x ≤ g a x) → a ≥ 4 :=
sorry

end f_minimum_value_g_range_condition_l2781_278199


namespace square_fence_perimeter_16_posts_l2781_278165

/-- Calculates the outer perimeter of a square fence given the number of posts,
    post width, and gap between posts. -/
def squareFencePerimeter (numPosts : ℕ) (postWidth : ℚ) (gapWidth : ℚ) : ℚ :=
  let postsPerSide : ℕ := numPosts / 4
  let gapsPerSide : ℕ := postsPerSide - 1
  let sideLength : ℚ := (gapsPerSide : ℚ) * gapWidth + (postsPerSide : ℚ) * postWidth
  4 * sideLength

/-- The outer perimeter of a square fence with 16 posts, each 6 inches wide,
    and 4 feet between posts, is 56 feet. -/
theorem square_fence_perimeter_16_posts :
  squareFencePerimeter 16 (1/2) 4 = 56 := by
  sorry

end square_fence_perimeter_16_posts_l2781_278165


namespace product_of_roots_l2781_278111

theorem product_of_roots (x : ℝ) : 
  (3 * x^2 + 6 * x - 81 = 0) → 
  ∃ y : ℝ, (3 * y^2 + 6 * y - 81 = 0) ∧ (x * y = -27) := by
sorry

end product_of_roots_l2781_278111


namespace unique_b_for_two_integer_solutions_l2781_278134

theorem unique_b_for_two_integer_solutions :
  ∃! b : ℤ, ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x : ℤ, x ∈ s ↔ x^2 + b*x - 2 ≤ 0 :=
by sorry

end unique_b_for_two_integer_solutions_l2781_278134


namespace polynomial_expansion_l2781_278170

theorem polynomial_expansion :
  ∀ t : ℝ, (3 * t^3 - 2 * t^2 + t - 4) * (2 * t^2 - t + 3) = 
    6 * t^5 - 7 * t^4 + 5 * t^3 - 15 * t^2 + 7 * t - 12 := by
  sorry

end polynomial_expansion_l2781_278170


namespace perpendicular_vectors_k_value_l2781_278162

/-- Given two vectors a and b in ℝ², where a = (2, 3) and b = (k, -1),
    if a is perpendicular to b, then k = 3/2. -/
theorem perpendicular_vectors_k_value :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![k, -1]
  (∀ i, i < 2 → a i * b i = 0) →
  k = 3/2 := by
sorry

end perpendicular_vectors_k_value_l2781_278162


namespace sector_area_l2781_278173

/-- The area of a sector of a circle with radius 4 cm and arc length 3.5 cm is 7 cm² -/
theorem sector_area (r : ℝ) (arc_length : ℝ) (h1 : r = 4) (h2 : arc_length = 3.5) :
  (arc_length / (2 * π * r)) * (π * r^2) = 7 := by
  sorry

end sector_area_l2781_278173


namespace parabola_circle_intersection_l2781_278129

/-- Given a parabola y^2 = 2px (p > 0) and a point A(t, 0) (t > 0), 
    a line through A intersects the parabola at B and C. 
    Lines OB and OC intersect the line x = -t at M and N respectively. 
    This theorem states that the circle with diameter MN intersects 
    the x-axis at two fixed points. -/
theorem parabola_circle_intersection 
  (p t : ℝ) 
  (hp : p > 0) 
  (ht : t > 0) : 
  ∃ (x₁ x₂ : ℝ), 
    x₁ = -t + Real.sqrt (2 * p * t) ∧ 
    x₂ = -t - Real.sqrt (2 * p * t) ∧ 
    (∀ (x y : ℝ), 
      (x + t)^2 + y^2 = (x₁ + t)^2 → 
      y = 0 → x = x₁ ∨ x = x₂) :=
by sorry


end parabola_circle_intersection_l2781_278129


namespace smallest_dual_palindrome_l2781_278102

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Bool := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 6 → (isPalindrome n 3 ∧ isPalindrome n 5) → n ≥ 26 := by
  sorry

#check smallest_dual_palindrome

end smallest_dual_palindrome_l2781_278102


namespace homogeneous_de_solution_l2781_278163

/-- The homogeneous differential equation -/
def homogeneous_de (x y : ℝ) (dx dy : ℝ) : Prop :=
  (x^2 - y^2) * dy - 2 * y * x * dx = 0

/-- The general solution to the homogeneous differential equation -/
def general_solution (x y C : ℝ) : Prop :=
  x^2 + y^2 = C * y

/-- Theorem stating that the general solution satisfies the homogeneous differential equation -/
theorem homogeneous_de_solution (x y C : ℝ) :
  general_solution x y C →
  ∃ (dx dy : ℝ), homogeneous_de x y dx dy :=
sorry

end homogeneous_de_solution_l2781_278163


namespace total_seashells_l2781_278147

def sam_seashells : ℕ := 35
def joan_seashells : ℕ := 18

theorem total_seashells : sam_seashells + joan_seashells = 53 := by
  sorry

end total_seashells_l2781_278147


namespace five_percent_difference_l2781_278151

theorem five_percent_difference (x y : ℝ) 
  (hx : 5 = 0.25 * x) 
  (hy : 5 = 0.50 * y) : 
  x - y = 10 := by
sorry

end five_percent_difference_l2781_278151


namespace probability_not_red_marble_l2781_278136

theorem probability_not_red_marble (orange purple red yellow : ℕ) : 
  orange = 4 → purple = 7 → red = 8 → yellow = 5 → 
  (orange + purple + yellow) / (orange + purple + red + yellow) = 2 / 3 := by
  sorry

end probability_not_red_marble_l2781_278136


namespace birds_on_fence_l2781_278184

theorem birds_on_fence : ∃ x : ℕ, (2 * x + 10 = 50) ∧ (x = 20) :=
by sorry

end birds_on_fence_l2781_278184


namespace radius_of_circle_B_l2781_278138

/-- A configuration of four circles A, B, C, and D with specific properties -/
structure CircleConfiguration where
  /-- Radius of circle A -/
  radiusA : ℝ
  /-- Radius of circle B -/
  radiusB : ℝ
  /-- Radius of circle D -/
  radiusD : ℝ
  /-- Circles A, B, and C are externally tangent to each other -/
  externallyTangent : Bool
  /-- Circles A, B, and C are internally tangent to circle D -/
  internallyTangentD : Bool
  /-- Circles B and C are congruent -/
  bCongruentC : Bool
  /-- Circle A passes through the center of D -/
  aPassesThroughCenterD : Bool

/-- The theorem stating that given the specific configuration, the radius of circle B is 7/3 -/
theorem radius_of_circle_B (config : CircleConfiguration) 
  (h1 : config.radiusA = 2)
  (h2 : config.externallyTangent = true)
  (h3 : config.internallyTangentD = true)
  (h4 : config.bCongruentC = true)
  (h5 : config.aPassesThroughCenterD = true) :
  config.radiusB = 7/3 := by
  sorry

end radius_of_circle_B_l2781_278138


namespace polynomial_property_l2781_278116

-- Define the polynomial Q(x)
def Q (x a b c : ℝ) : ℝ := 3 * x^3 + a * x^2 + b * x + c

-- State the theorem
theorem polynomial_property (a b c : ℝ) :
  -- The y-intercept is 6
  Q 0 a b c = 6 →
  -- The mean of zeros, product of zeros, and sum of coefficients are equal
  (∃ m : ℝ, 
    -- Mean of zeros
    (-(a / 3) / 3 = m) ∧ 
    -- Product of zeros
    (-c / 3 = m) ∧ 
    -- Sum of coefficients
    (3 + a + b + c = m)) →
  -- Conclusion: b = -29
  b = -29 := by sorry

end polynomial_property_l2781_278116


namespace five_heads_before_two_tails_l2781_278141

/-- The probability of getting 5 heads before 2 consecutive tails when repeatedly flipping a fair coin -/
def probability_5H_before_2T : ℚ :=
  3 / 34

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℚ → Prop) : Prop :=
  p (1/2) ∧ p (1/2)

theorem five_heads_before_two_tails (p : ℚ → Prop) (h : fair_coin p) :
  probability_5H_before_2T = 3 / 34 :=
sorry

end five_heads_before_two_tails_l2781_278141


namespace kevin_collected_18_frisbees_l2781_278120

/-- The number of frisbees Kevin collected for prizes at the fair. -/
def num_frisbees (total_prizes stuffed_animals yo_yos : ℕ) : ℕ :=
  total_prizes - (stuffed_animals + yo_yos)

/-- Theorem stating that Kevin collected 18 frisbees. -/
theorem kevin_collected_18_frisbees : 
  num_frisbees 50 14 18 = 18 := by
  sorry

end kevin_collected_18_frisbees_l2781_278120


namespace tangent_circle_existence_and_radius_l2781_278179

/-- Given three circles with radii r₁, r₂, r₃, where r₁ > r₂ and r₁ > r₃,
    there exists a circle touching the four tangents drawn as described,
    with radius (r₁ * r₂ * r₃) / (r₁ * (r₂ + r₃) - r₂ * r₃) -/
theorem tangent_circle_existence_and_radius 
  (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ > r₂) 
  (h₂ : r₁ > r₃) 
  (h₃ : r₁ > 0) 
  (h₄ : r₂ > 0) 
  (h₅ : r₃ > 0) :
  ∃ (r : ℝ), r = (r₁ * r₂ * r₃) / (r₁ * (r₂ + r₃) - r₂ * r₃) ∧ 
  r > 0 :=
sorry

end tangent_circle_existence_and_radius_l2781_278179


namespace convex_polygon_division_theorem_l2781_278197

-- Define a type for polygons
def Polygon : Type := Set (ℝ × ℝ)

-- Define a type for motions (transformations)
def Motion : Type := (ℝ × ℝ) → (ℝ × ℝ)

-- Define a predicate for convex polygons
def IsConvex (p : Polygon) : Prop := sorry

-- Define a predicate for orientation-preserving motions
def IsOrientationPreserving (m : Motion) : Prop := sorry

-- Define a predicate for a polygon being dividable by a broken line into two polygons
def DividableByBrokenLine (p : Polygon) (p1 p2 : Polygon) : Prop := sorry

-- Define a predicate for a polygon being dividable by a segment into two polygons
def DividableBySegment (p : Polygon) (p1 p2 : Polygon) : Prop := sorry

-- Define a predicate for two polygons being transformable into each other by a motion
def Transformable (p1 p2 : Polygon) (m : Motion) : Prop := sorry

-- State the theorem
theorem convex_polygon_division_theorem (p : Polygon) :
  IsConvex p →
  (∃ (p1 p2 : Polygon) (m : Motion), 
    DividableByBrokenLine p p1 p2 ∧ 
    IsOrientationPreserving m ∧ 
    Transformable p1 p2 m) →
  (∃ (q1 q2 : Polygon) (n : Motion), 
    DividableBySegment p q1 q2 ∧ 
    IsOrientationPreserving n ∧ 
    Transformable q1 q2 n) :=
by sorry

end convex_polygon_division_theorem_l2781_278197


namespace fruit_mix_kiwis_l2781_278168

theorem fruit_mix_kiwis (total : ℕ) (s b o k : ℕ) : 
  total = 340 →
  s + b + o + k = total →
  s = 3 * b →
  o = 2 * k →
  k = 5 * s →
  k = 104 := by
  sorry

end fruit_mix_kiwis_l2781_278168


namespace rationalize_denominator_seven_sqrt_147_l2781_278115

theorem rationalize_denominator_seven_sqrt_147 :
  ∃ (a b : ℝ) (h : b ≠ 0), (7 / Real.sqrt 147) = (a * Real.sqrt b) / b :=
by
  -- The proof goes here
  sorry

end rationalize_denominator_seven_sqrt_147_l2781_278115


namespace window_treatment_cost_l2781_278183

def number_of_windows : ℕ := 3
def cost_of_sheers : ℚ := 40
def cost_of_drapes : ℚ := 60

def total_cost : ℚ := number_of_windows * (cost_of_sheers + cost_of_drapes)

theorem window_treatment_cost : total_cost = 300 := by
  sorry

end window_treatment_cost_l2781_278183


namespace unique_function_is_zero_l2781_278175

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f (x - y) + f (f (1 - x * y))

-- Theorem statement
theorem unique_function_is_zero :
  ∃! f : ℝ → ℝ, SatisfiesProperty f ∧ ∀ x : ℝ, f x = 0 :=
sorry

end unique_function_is_zero_l2781_278175


namespace smallest_difference_l2781_278114

/-- Vovochka's sum method for three-digit numbers -/
def vovochka_sum (a b c d e f : ℕ) : ℕ :=
  1000 * (a + d) + 100 * (b + e) + (c + f)

/-- Correct sum method for three-digit numbers -/
def correct_sum (a b c d e f : ℕ) : ℕ :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sum_difference (a b c d e f : ℕ) : ℕ :=
  vovochka_sum a b c d e f - correct_sum a b c d e f

theorem smallest_difference :
  ∀ a b c d e f : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 →
    sum_difference a b c d e f > 0 →
    sum_difference a b c d e f ≥ 1800 :=
sorry

end smallest_difference_l2781_278114


namespace first_three_digits_after_decimal_l2781_278181

theorem first_three_digits_after_decimal (n : ℕ) (x : ℝ) :
  n = 1200 →
  x = (10^n + 1)^(5/3) →
  ∃ (k : ℕ), x = k + 0.333 + r ∧ r < 0.001 :=
sorry

end first_three_digits_after_decimal_l2781_278181


namespace stratified_sample_medium_stores_l2781_278191

/-- Given a population of stores with a known number of medium-sized stores,
    calculate the number of medium-sized stores in a stratified sample. -/
theorem stratified_sample_medium_stores
  (total_stores : ℕ)
  (medium_stores : ℕ)
  (sample_size : ℕ)
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  (medium_stores : ℚ) / total_stores * sample_size = 5 := by
sorry

end stratified_sample_medium_stores_l2781_278191


namespace remainder_problem_l2781_278143

theorem remainder_problem (N : ℕ) : 
  N % 68 = 0 ∧ N / 68 = 269 → N % 67 = 1 := by
sorry

end remainder_problem_l2781_278143


namespace dagger_example_l2781_278133

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * ((n + q) / n)

-- Theorem statement
theorem dagger_example : dagger (9/5) (7/2) = 441/5 := by
  sorry

end dagger_example_l2781_278133


namespace john_annual_oil_change_cost_l2781_278189

/-- Calculates the annual cost of oil changes for a driver --/
def annual_oil_change_cost (miles_per_month : ℕ) (miles_per_oil_change : ℕ) (free_changes_per_year : ℕ) (cost_per_change : ℕ) : ℕ :=
  let total_miles := miles_per_month * 12
  let total_changes := total_miles / miles_per_oil_change
  let paid_changes := total_changes - free_changes_per_year
  paid_changes * cost_per_change

/-- Theorem stating that John pays $150 a year for oil changes --/
theorem john_annual_oil_change_cost :
  annual_oil_change_cost 1000 3000 1 50 = 150 := by
  sorry

end john_annual_oil_change_cost_l2781_278189


namespace ice_cream_melt_l2781_278128

theorem ice_cream_melt (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 3) (h_cylinder : r_cylinder = 10) :
  (4 / 3 * Real.pi * r_sphere ^ 3) / (Real.pi * r_cylinder ^ 2) = 9 / 25 := by
  sorry

end ice_cream_melt_l2781_278128


namespace not_in_second_quadrant_l2781_278150

-- Define the linear function
def f (x : ℝ) : ℝ := x - 1

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem not_in_second_quadrant :
  ∀ x : ℝ, ¬(second_quadrant x (f x)) := by
  sorry

end not_in_second_quadrant_l2781_278150


namespace marks_lost_per_wrong_answer_l2781_278192

theorem marks_lost_per_wrong_answer 
  (total_questions : ℕ)
  (marks_per_correct : ℕ)
  (total_marks : ℕ)
  (correct_answers : ℕ)
  (h1 : total_questions = 60)
  (h2 : marks_per_correct = 4)
  (h3 : total_marks = 160)
  (h4 : correct_answers = 44)
  : ℕ :=
by
  sorry

#check marks_lost_per_wrong_answer

end marks_lost_per_wrong_answer_l2781_278192


namespace all_statements_false_l2781_278117

theorem all_statements_false :
  (∀ x : ℝ, x^2 = 4 → x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 = 9 → x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, x^3 = -125 → x = -5) ∧
  (∀ x : ℝ, x^2 = 16 → x = 4 ∨ x = -4) :=
by sorry

end all_statements_false_l2781_278117


namespace function_properties_l2781_278169

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p ≠ 0 ∧ ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h_even : IsEven f)
    (h_shift : ∀ x, f (x + 1) = -f x)
    (h_incr : IsIncreasingOn f (-1) 0) :
    (IsPeriodic f 2) ∧ 
    (∀ x, f (2 - x) = f x) ∧
    (f 2 = f 0) := by
  sorry

end function_properties_l2781_278169


namespace roots_cube_theorem_l2781_278125

theorem roots_cube_theorem (a b c d x₁ x₂ : ℝ) :
  (x₁^2 - (a + d)*x₁ + ad - bc = 0) →
  (x₂^2 - (a + d)*x₂ + ad - bc = 0) →
  (x₁^3)^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*(x₁^3) + (a*d - b*c)^3 = 0 ∧
  (x₂^3)^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*(x₂^3) + (a*d - b*c)^3 = 0 :=
by sorry

end roots_cube_theorem_l2781_278125


namespace odd_product_plus_one_is_odd_l2781_278196

theorem odd_product_plus_one_is_odd (p q : ℕ) 
  (hp : Odd p) (hq : Odd q) (hp_pos : 0 < p) (hq_pos : 0 < q) : 
  Odd (4 * p * q + 1) := by
  sorry

end odd_product_plus_one_is_odd_l2781_278196


namespace cylinder_volume_increase_l2781_278177

theorem cylinder_volume_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let new_radius := 2.5 * r
  let new_height := 3 * h
  (π * new_radius^2 * new_height) / (π * r^2 * h) = 18.75 := by
sorry

end cylinder_volume_increase_l2781_278177


namespace binomial_square_constant_l2781_278109

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
  sorry

end binomial_square_constant_l2781_278109


namespace arithmetic_progression_squares_products_l2781_278106

/-- If a, b, and c form an arithmetic progression, then a^2 + ab + b^2, a^2 + ac + c^2, and b^2 + bc + c^2 form an arithmetic progression. -/
theorem arithmetic_progression_squares_products (a b c : ℝ) :
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →
  ∃ q : ℝ, (a^2 + a*c + c^2) - (a^2 + a*b + b^2) = q ∧
           (b^2 + b*c + c^2) - (a^2 + a*c + c^2) = q :=
by sorry

end arithmetic_progression_squares_products_l2781_278106


namespace triangle_inequality_l2781_278142

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b > c → b + c > a → c + a > b →
  a + b + c = 4 → 
  a^2 + b^2 + c^2 + a*b*c < 8 := by
sorry

end triangle_inequality_l2781_278142


namespace class_average_score_l2781_278135

theorem class_average_score (total_students : ℕ) 
  (assigned_day_percent : ℚ) (makeup_date_percent : ℚ) (later_date_percent : ℚ)
  (assigned_day_score : ℚ) (makeup_date_score : ℚ) (later_date_score : ℚ) :
  total_students = 100 →
  assigned_day_percent = 60 / 100 →
  makeup_date_percent = 30 / 100 →
  later_date_percent = 10 / 100 →
  assigned_day_score = 60 / 100 →
  makeup_date_score = 80 / 100 →
  later_date_score = 75 / 100 →
  (assigned_day_percent * assigned_day_score * total_students +
   makeup_date_percent * makeup_date_score * total_students +
   later_date_percent * later_date_score * total_students) / total_students = 675 / 1000 := by
  sorry

#eval (60 * 60 + 30 * 80 + 10 * 75) / 100  -- Expected output: 67.5

end class_average_score_l2781_278135


namespace largest_three_digit_with_digit_product_8_l2781_278159

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem largest_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → n ≤ 811 :=
sorry

end largest_three_digit_with_digit_product_8_l2781_278159


namespace power_greater_than_square_l2781_278171

theorem power_greater_than_square (n : ℕ) (h : n ≥ 8) : 2^(n-1) > (n+1)^2 := by
  sorry

end power_greater_than_square_l2781_278171


namespace books_left_to_read_l2781_278188

theorem books_left_to_read 
  (total_books : ℕ) 
  (books_read : ℕ) 
  (h1 : total_books = 19) 
  (h2 : books_read = 4) : 
  total_books - books_read = 15 := by
sorry

end books_left_to_read_l2781_278188


namespace bird_watching_percentage_difference_l2781_278100

-- Define the number of birds seen by each person
def gabrielle_robins : ℕ := 5
def gabrielle_cardinals : ℕ := 4
def gabrielle_blue_jays : ℕ := 3

def chase_robins : ℕ := 2
def chase_cardinals : ℕ := 5
def chase_blue_jays : ℕ := 3

-- Calculate total birds seen by each person
def gabrielle_total : ℕ := gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays
def chase_total : ℕ := chase_robins + chase_cardinals + chase_blue_jays

-- Define the percentage difference
def percentage_difference : ℚ := (gabrielle_total - chase_total : ℚ) / chase_total * 100

-- Theorem statement
theorem bird_watching_percentage_difference :
  percentage_difference = 20 :=
sorry

end bird_watching_percentage_difference_l2781_278100


namespace all_statements_imply_not_all_true_l2781_278101

theorem all_statements_imply_not_all_true (p q r : Prop) :
  -- Statement 1
  ((p ∧ q ∧ ¬r) → ¬(p ∧ q ∧ r)) ∧
  -- Statement 2
  ((p ∧ ¬q ∧ r) → ¬(p ∧ q ∧ r)) ∧
  -- Statement 3
  ((¬p ∧ q ∧ ¬r) → ¬(p ∧ q ∧ r)) ∧
  -- Statement 4
  ((¬p ∧ ¬q ∧ ¬r) → ¬(p ∧ q ∧ r)) :=
by sorry


end all_statements_imply_not_all_true_l2781_278101
