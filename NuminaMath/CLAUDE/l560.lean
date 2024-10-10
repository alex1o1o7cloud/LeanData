import Mathlib

namespace hong_travel_bound_l560_56025

/-- Represents a town in the country -/
structure Town where
  coins : ℕ

/-- Represents the country with its towns and roads -/
structure Country where
  towns : Set Town
  roads : Set (Town × Town)
  initial_coins : ℕ

/-- Represents Hong's travel -/
structure Travel where
  country : Country
  days : ℕ

/-- The maximum number of days Hong can travel -/
def max_travel_days (n : ℕ) : ℕ := n + 2 * n^(2/3)

theorem hong_travel_bound (c : Country) (t : Travel) (h_infinite : Infinite c.towns)
    (h_all_connected : ∀ a b : Town, a ≠ b → (a, b) ∈ c.roads)
    (h_initial_coins : ∀ town ∈ c.towns, town.coins = c.initial_coins)
    (h_coin_transfer : ∀ k : ℕ, ∀ a b : Town, 
      (a, b) ∈ c.roads → t.days = k → b.coins = b.coins - k ∧ a.coins = a.coins + k)
    (h_road_usage : ∀ a b : Town, (a, b) ∈ c.roads → (b, a) ∉ c.roads) :
  t.days ≤ max_travel_days c.initial_coins :=
sorry

end hong_travel_bound_l560_56025


namespace line_symmetry_l560_56038

-- Define a line by its coefficients a, b, and c in the equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define symmetry with respect to x = 1
def symmetricAboutX1 (l1 l2 : Line) : Prop :=
  ∀ x y : ℝ, l1.a * (2 - x) + l1.b * y + l1.c = 0 ↔ l2.a * x + l2.b * y + l2.c = 0

-- Theorem statement
theorem line_symmetry (l1 l2 : Line) :
  l1 = Line.mk 3 (-4) (-3) →
  symmetricAboutX1 l1 l2 →
  l2 = Line.mk 3 4 (-3) := by
  sorry

end line_symmetry_l560_56038


namespace trapezoid_area_theorem_l560_56097

/-- Represents a trapezoid with given diagonals and bases -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Calculates the area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  sorry

theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.diagonal1 = 7)
  (h2 : t.diagonal2 = 8)
  (h3 : t.base1 = 3)
  (h4 : t.base2 = 6) :
  trapezoidArea t = 12 * Real.sqrt 5 := by
  sorry

end trapezoid_area_theorem_l560_56097


namespace cost_of_goods_l560_56018

/-- The cost of goods A, B, and C given certain conditions -/
theorem cost_of_goods (x y z : ℚ) 
  (h1 : 2*x + 4*y + z = 90)
  (h2 : 4*x + 10*y + z = 110) : 
  x + y + z = 80 := by
sorry

end cost_of_goods_l560_56018


namespace geometric_series_sum_l560_56013

/-- The limiting sum of the geometric series 4 - 8/3 + 16/9 - ... equals 2.4 -/
theorem geometric_series_sum : 
  let a : ℝ := 4
  let r : ℝ := -2/3
  let s : ℝ := a / (1 - r)
  s = 2.4 := by sorry

end geometric_series_sum_l560_56013


namespace problem_statement_l560_56021

def a₁ (n : ℕ+) : ℤ := n.val^2 - 10*n.val + 23
def a₂ (n : ℕ+) : ℤ := n.val^2 - 9*n.val + 31
def a₃ (n : ℕ+) : ℤ := n.val^2 - 12*n.val + 46

theorem problem_statement :
  (∀ n : ℕ+, Even (a₁ n + a₂ n + a₃ n)) ∧
  (∀ n : ℕ+, (Prime (a₁ n) ∧ Prime (a₂ n) ∧ Prime (a₃ n)) ↔ n = 7) :=
by sorry

end problem_statement_l560_56021


namespace subset_iff_range_l560_56082

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - a + 1) * (x - a - 1) ≤ 0}
def B : Set ℝ := {x | |x - 1/2| ≤ 3/2}

-- State the theorem
theorem subset_iff_range (a : ℝ) : A a ⊆ B ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

end subset_iff_range_l560_56082


namespace simplify_expression_l560_56086

theorem simplify_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  3 * x ^ Real.sqrt 2 * (2 * x ^ (-Real.sqrt 2) * y * z) = 6 * y * z := by
  sorry

end simplify_expression_l560_56086


namespace dachshund_starting_weight_l560_56049

theorem dachshund_starting_weight :
  ∀ (labrador_start dachshund_start : ℝ),
    labrador_start = 40 →
    (labrador_start * 1.25 - dachshund_start * 1.25 = 35) →
    dachshund_start = 12 := by
  sorry

end dachshund_starting_weight_l560_56049


namespace sum_equals_negative_two_thirds_l560_56043

theorem sum_equals_negative_two_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 4) : 
  a + b + c + d = -2/3 := by
sorry

end sum_equals_negative_two_thirds_l560_56043


namespace christmas_ball_colors_l560_56071

/-- Given a total number of balls and the number of balls per color, 
    calculate the number of colors used. -/
def number_of_colors (total_balls : ℕ) (balls_per_color : ℕ) : ℕ :=
  total_balls / balls_per_color

/-- Prove that the number of colors used is 10 given the problem conditions. -/
theorem christmas_ball_colors :
  let total_balls : ℕ := 350
  let balls_per_color : ℕ := 35
  number_of_colors total_balls balls_per_color = 10 := by
  sorry

end christmas_ball_colors_l560_56071


namespace total_persimmons_in_boxes_l560_56016

/-- Given that each box contains 100 persimmons and there are 6 boxes,
    prove that the total number of persimmons is 600. -/
theorem total_persimmons_in_boxes : 
  let persimmons_per_box : ℕ := 100
  let number_of_boxes : ℕ := 6
  persimmons_per_box * number_of_boxes = 600 := by
  sorry

end total_persimmons_in_boxes_l560_56016


namespace josh_marbles_l560_56057

/-- The number of marbles Josh has after losing some -/
def remaining_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem: If Josh had 9 marbles initially and lost 5, he now has 4 marbles -/
theorem josh_marbles : remaining_marbles 9 5 = 4 := by
  sorry

end josh_marbles_l560_56057


namespace arithmetic_sequence_problem_l560_56001

/-- 
Given an arithmetic sequence where the third term is 3 and the eleventh term is 15,
prove that the first term is 0 and the common difference is 3/2.
-/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : a 3 = 3) 
  (h2 : a 11 = 15) 
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) : 
  a 1 = 0 ∧ a 2 - a 1 = 3/2 := by
  sorry

end arithmetic_sequence_problem_l560_56001


namespace math_textbooks_same_box_probability_l560_56042

/-- The probability of all mathematics textbooks ending up in the same box -/
theorem math_textbooks_same_box_probability :
  let total_books : ℕ := 15
  let math_books : ℕ := 4
  let box1_capacity : ℕ := 4
  let box2_capacity : ℕ := 5
  let box3_capacity : ℕ := 6
  
  -- Total number of ways to distribute books
  let total_distributions : ℕ := (Nat.choose total_books box1_capacity) * 
                                 (Nat.choose (total_books - box1_capacity) box2_capacity) *
                                 (Nat.choose (total_books - box1_capacity - box2_capacity) box3_capacity)
  
  -- Number of ways where all math books are in the same box
  let favorable_outcomes : ℕ := (Nat.choose (total_books - math_books) 0) +
                                (Nat.choose (total_books - math_books) 1) +
                                (Nat.choose (total_books - math_books) 2)
  
  (favorable_outcomes : ℚ) / total_distributions = 67 / 630630 :=
by sorry

end math_textbooks_same_box_probability_l560_56042


namespace allyson_age_l560_56035

theorem allyson_age (hiram_age : ℕ) (allyson_age : ℕ) 
  (h1 : hiram_age = 40)
  (h2 : hiram_age + 12 = 2 * allyson_age - 4) :
  allyson_age = 28 := by
  sorry

end allyson_age_l560_56035


namespace xy_equals_four_l560_56062

theorem xy_equals_four (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_x : x = w)
  (h_y : y = w)
  (h_w : w + w = w * w)
  (h_z : z = 3) : 
  x * y = 4 := by
sorry

end xy_equals_four_l560_56062


namespace final_score_is_94_l560_56091

/-- Represents the scoring system for a choir competition -/
structure ScoringSystem where
  songContentWeight : Real
  singingSkillsWeight : Real
  spiritWeight : Real
  weightSum : songContentWeight + singingSkillsWeight + spiritWeight = 1

/-- Represents the scores of a participating team -/
structure TeamScores where
  songContent : Real
  singingSkills : Real
  spirit : Real

/-- Calculates the final score given a scoring system and team scores -/
def calculateFinalScore (system : ScoringSystem) (scores : TeamScores) : Real :=
  system.songContentWeight * scores.songContent +
  system.singingSkillsWeight * scores.singingSkills +
  system.spiritWeight * scores.spirit

theorem final_score_is_94 (system : ScoringSystem) (scores : TeamScores)
    (h1 : system.songContentWeight = 0.3)
    (h2 : system.singingSkillsWeight = 0.4)
    (h3 : system.spiritWeight = 0.3)
    (h4 : scores.songContent = 90)
    (h5 : scores.singingSkills = 94)
    (h6 : scores.spirit = 98) :
    calculateFinalScore system scores = 94 := by
  sorry


end final_score_is_94_l560_56091


namespace circle_in_rectangle_l560_56008

theorem circle_in_rectangle (r x : ℝ) : 
  r > 0 →  -- radius is positive
  2 * r = x →  -- width of rectangle is diameter of circle
  r + (2 * x) / 3 + r = 10 →  -- length of rectangle
  x = 6 := by
  sorry

end circle_in_rectangle_l560_56008


namespace museum_ticket_fraction_l560_56024

def total_money : ℚ := 90
def sandwich_fraction : ℚ := 1/5
def book_fraction : ℚ := 1/2
def money_left : ℚ := 12

theorem museum_ticket_fraction :
  let spent := total_money - money_left
  let sandwich_cost := sandwich_fraction * total_money
  let book_cost := book_fraction * total_money
  let museum_cost := spent - (sandwich_cost + book_cost)
  museum_cost / total_money = 1/6 := by sorry

end museum_ticket_fraction_l560_56024


namespace green_chips_count_l560_56090

/-- Given a jar of chips where:
  * 3 blue chips represent 10% of the total
  * 50% of the chips are white
  * The remaining chips are green
  Prove that there are 12 green chips -/
theorem green_chips_count (total : ℕ) (blue white green : ℕ) : 
  blue = 3 ∧ 
  blue * 10 = total ∧ 
  2 * white = total ∧ 
  blue + white + green = total → 
  green = 12 := by
sorry

end green_chips_count_l560_56090


namespace pieces_with_high_product_bound_l560_56012

/-- Represents an infinite chessboard with pieces placed on it. -/
structure InfiniteChessboard where
  m : ℕ  -- Total number of pieces
  piece_positions : Finset (ℕ × ℕ)  -- Positions of pieces
  piece_count : piece_positions.card = m  -- Ensure the number of pieces matches m

/-- Calculates the number of pieces in a given row -/
def pieces_in_row (board : InfiniteChessboard) (row : ℕ) : ℕ :=
  (board.piece_positions.filter (fun p => p.1 = row)).card

/-- Calculates the number of pieces in a given column -/
def pieces_in_column (board : InfiniteChessboard) (col : ℕ) : ℕ :=
  (board.piece_positions.filter (fun p => p.2 = col)).card

/-- Calculates the product of pieces in the row and column for a given position -/
def product_for_position (board : InfiniteChessboard) (pos : ℕ × ℕ) : ℕ :=
  (pieces_in_row board pos.1) * (pieces_in_column board pos.2)

/-- The main theorem to be proved -/
theorem pieces_with_high_product_bound (board : InfiniteChessboard) :
  (board.piece_positions.filter (fun pos => product_for_position board pos ≥ 10 * board.m)).card ≤ board.m / 10 :=
sorry

end pieces_with_high_product_bound_l560_56012


namespace water_speed_calculation_l560_56029

/-- The speed of water in a river where a person who can swim at 4 km/h in still water
    takes 8 hours to swim 16 km against the current. -/
def water_speed : ℝ :=
  let still_water_speed : ℝ := 4
  let distance : ℝ := 16
  let time : ℝ := 8
  2

theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ)
    (h1 : still_water_speed = 4)
    (h2 : distance = 16)
    (h3 : time = 8)
    (h4 : distance = (still_water_speed - water_speed) * time) :
  water_speed = 2 := by
  sorry

end water_speed_calculation_l560_56029


namespace stock_value_change_l560_56093

theorem stock_value_change (initial_value : ℝ) (h : initial_value > 0) :
  let day1_value := initial_value * 0.85
  let day2_value := day1_value * 1.4
  (day2_value - initial_value) / initial_value = 0.19 := by
sorry

end stock_value_change_l560_56093


namespace andrews_stickers_l560_56064

theorem andrews_stickers (total : ℕ) (daniels : ℕ) (freds_extra : ℕ) 
  (h1 : total = 750)
  (h2 : daniels = 250)
  (h3 : freds_extra = 120) :
  total - (daniels + (daniels + freds_extra)) = 130 := by
  sorry

end andrews_stickers_l560_56064


namespace not_all_perfect_squares_l560_56098

theorem not_all_perfect_squares (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(∃ x y z : ℕ, (2 * a^2 + b^2 + 3 = x^2) ∧ (2 * b^2 + c^2 + 3 = y^2) ∧ (2 * c^2 + a^2 + 3 = z^2)) :=
by sorry

end not_all_perfect_squares_l560_56098


namespace shop_item_cost_prices_l560_56045

theorem shop_item_cost_prices :
  ∀ (c1 c2 : ℝ),
    (0.30 * c1 - 0.15 * c1 = 120) →
    (0.25 * c2 - 0.10 * c2 = 150) →
    c1 = 800 ∧ c2 = 1000 := by
  sorry

end shop_item_cost_prices_l560_56045


namespace family_ages_solution_l560_56084

/-- Represents the ages of a family at a given time --/
structure FamilyAges where
  man : ℕ
  father : ℕ
  sister : ℕ

/-- Checks if the given ages satisfy the problem conditions --/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.man = (2 * ages.father) / 5 ∧
  ages.man + 10 = (ages.father + 10) / 2 ∧
  ages.sister + 10 = (3 * (ages.father + 10)) / 4

/-- The theorem stating the solution to the problem --/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ 
    ages.man = 20 ∧ ages.father = 50 ∧ ages.sister = 35 := by
  sorry

end family_ages_solution_l560_56084


namespace people_disliking_tv_and_books_l560_56081

def total_surveyed : ℕ := 1500
def tv_dislike_percentage : ℚ := 25 / 100
def book_and_tv_dislike_percentage : ℚ := 15 / 100

theorem people_disliking_tv_and_books :
  ⌊(tv_dislike_percentage * total_surveyed : ℚ) * book_and_tv_dislike_percentage⌋ = 56 := by
  sorry

end people_disliking_tv_and_books_l560_56081


namespace system_of_equations_l560_56095

theorem system_of_equations (a b c d k : ℝ) 
  (h1 : a + b = 11)
  (h2 : b^2 + c^2 = k)
  (h3 : b + c = 9)
  (h4 : c + d = 3)
  (h5 : k > 0) :
  a + d = 5 := by
sorry

end system_of_equations_l560_56095


namespace system_solution_ratio_l560_56032

theorem system_solution_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hc : c ≠ 0)
  (eq1 : 8 * x - 5 * y = c) (eq2 : 10 * y - 16 * x = d) : d / c = -2 := by
  sorry

end system_solution_ratio_l560_56032


namespace binomial_expansion_terms_l560_56005

theorem binomial_expansion_terms (n : ℕ) : (Finset.range (2 * n + 1)).card = 2 * n + 1 := by
  sorry

end binomial_expansion_terms_l560_56005


namespace B_power_101_l560_56023

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_101 : B^101 = B^2 := by sorry

end B_power_101_l560_56023


namespace mark_marbles_count_l560_56053

def connie_marbles : ℕ := 323
def juan_marbles : ℕ := connie_marbles + 175
def mark_marbles : ℕ := juan_marbles * 3

theorem mark_marbles_count : mark_marbles = 1494 := by
  sorry

end mark_marbles_count_l560_56053


namespace tissue_cost_theorem_l560_56092

/-- Calculates the total cost of tissues given the number of boxes, packs per box,
    tissues per pack, and cost per tissue. -/
def totalCost (boxes : ℕ) (packsPerBox : ℕ) (tissuesPerPack : ℕ) (costPerTissue : ℚ) : ℚ :=
  (boxes * packsPerBox * tissuesPerPack : ℚ) * costPerTissue

/-- Proves that the total cost of 10 boxes of tissues, with 20 packs per box,
    100 tissues per pack, and 5 cents per tissue, is $1000. -/
theorem tissue_cost_theorem :
  totalCost 10 20 100 (5 / 100) = 1000 := by
  sorry

end tissue_cost_theorem_l560_56092


namespace medal_winners_combinations_l560_56067

theorem medal_winners_combinations (semifinalists : ℕ) (eliminated : ℕ) (medals : ℕ) : 
  semifinalists = 8 →
  eliminated = 2 →
  medals = 3 →
  Nat.choose (semifinalists - eliminated) medals = 20 :=
by sorry

end medal_winners_combinations_l560_56067


namespace curve_tangent_parallel_l560_56026

theorem curve_tangent_parallel (k : ℝ) : 
  let f := fun x : ℝ => k * x + Real.log x
  let f' := fun x : ℝ => k + 1 / x
  (f' 1 = 2) → k = 1 := by
sorry

end curve_tangent_parallel_l560_56026


namespace root_exists_in_interval_l560_56061

-- Define the function f(x) = x³ - 22 - x
def f (x : ℝ) := x^3 - 22 - x

-- Theorem statement
theorem root_exists_in_interval :
  ∃ x₀ ∈ Set.Ioo 1 2, f x₀ = 0 :=
by
  sorry


end root_exists_in_interval_l560_56061


namespace rectangle_width_l560_56075

/-- 
Given a rectangle where:
  - The length is 3 cm shorter than the width
  - The perimeter is 54 cm
Prove that the width of the rectangle is 15 cm
-/
theorem rectangle_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = width - 3 →
  perimeter = 54 →
  perimeter = 2 * width + 2 * length →
  width = 15 := by
  sorry

end rectangle_width_l560_56075


namespace mistake_position_l560_56017

theorem mistake_position (n : ℕ) (a₁ : ℤ) (d : ℤ) (sum : ℤ) (k : ℕ) : 
  n = 21 →
  a₁ = 51 →
  d = 5 →
  sum = 2021 →
  k ∈ Finset.range n →
  sum = (n * (2 * a₁ + (n - 1) * d)) / 2 - 10 * k →
  k = 10 :=
by sorry

end mistake_position_l560_56017


namespace impossibleConfiguration_l560_56054

/-- Represents a configuration of points on a circle -/
structure CircleConfiguration where
  numPoints : ℕ
  circumference : ℕ

/-- Checks if a configuration satisfies the arc length condition -/
def satisfiesArcLengthCondition (config : CircleConfiguration) : Prop :=
  ∃ (points : Fin config.numPoints → ℝ),
    (∀ i, 0 ≤ points i ∧ points i < config.circumference) ∧
    (∀ l : ℕ, 1 ≤ l ∧ l < config.circumference →
      ∃ i j, (points j - points i + config.circumference) % config.circumference = l)

/-- The main theorem stating the impossibility of the configuration -/
theorem impossibleConfiguration :
  ¬ satisfiesArcLengthCondition ⟨10, 90⟩ := by
  sorry


end impossibleConfiguration_l560_56054


namespace missing_sale_is_7562_l560_56077

/-- Calculates the missing sale amount given sales for 5 out of 6 months and the average sale -/
def calculate_missing_sale (sale1 sale2 sale3 sale4 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale3 + sale4 + sale6)

theorem missing_sale_is_7562 (sale1 sale2 sale3 sale4 sale6 average_sale : ℕ) 
  (h1 : sale1 = 7435)
  (h2 : sale2 = 7927)
  (h3 : sale3 = 7855)
  (h4 : sale4 = 8230)
  (h5 : sale6 = 5991)
  (h6 : average_sale = 7500) :
  calculate_missing_sale sale1 sale2 sale3 sale4 sale6 average_sale = 7562 := by
  sorry

#eval calculate_missing_sale 7435 7927 7855 8230 5991 7500

end missing_sale_is_7562_l560_56077


namespace equation_solution_l560_56063

theorem equation_solution (x y : ℝ) : (x + y)^2 = (x + 1) * (y - 1) → x = -1 ∧ y = 1 := by
  sorry

end equation_solution_l560_56063


namespace semiperimeter_equals_diagonal_l560_56073

/-- A rectangle inscribed in a square --/
structure InscribedRectangle where
  /-- Side length of the square --/
  a : ℝ
  /-- Width of the rectangle --/
  b : ℝ
  /-- Height of the rectangle --/
  c : ℝ
  /-- The rectangle is not a square --/
  not_square : b ≠ c
  /-- The rectangle is inscribed in the square --/
  inscribed : b + c = a * Real.sqrt 2

/-- The semiperimeter of the inscribed rectangle equals the diagonal of the square --/
theorem semiperimeter_equals_diagonal (rect : InscribedRectangle) :
  (rect.b + rect.c) / 2 = rect.a * Real.sqrt 2 / 2 := by
  sorry

#check semiperimeter_equals_diagonal

end semiperimeter_equals_diagonal_l560_56073


namespace mean_of_playground_counts_l560_56015

def playground_counts : List ℕ := [6, 12, 1, 12, 7, 3, 8]

theorem mean_of_playground_counts :
  (playground_counts.sum : ℚ) / playground_counts.length = 7 := by
  sorry

end mean_of_playground_counts_l560_56015


namespace naturalNumberDecimal_irrational_l560_56085

/-- Represents the infinite decimal 0.1234567891011121314... -/
def naturalNumberDecimal : ℝ :=
  sorry

/-- The digits of naturalNumberDecimal after the decimal point consist of all natural numbers in order -/
axiom naturalNumberDecimal_property : ∀ n : ℕ, ∃ k : ℕ, sorry

theorem naturalNumberDecimal_irrational : Irrational naturalNumberDecimal := by
  sorry

end naturalNumberDecimal_irrational_l560_56085


namespace overlapping_sectors_area_l560_56096

/-- The area of the overlapping region of two sectors in a circle -/
theorem overlapping_sectors_area (r : ℝ) (θ₁ θ₂ : ℝ) (h_r : r = 10) (h_θ₁ : θ₁ = 45) (h_θ₂ : θ₂ = 90) :
  let sector_area (θ : ℝ) := (θ / 360) * π * r^2
  min (sector_area θ₁) (sector_area θ₂) = 12.5 * π :=
by sorry

end overlapping_sectors_area_l560_56096


namespace vector_equality_l560_56056

/-- Given vectors a, b, and c in ℝ², prove that c = 1/2 * a - 3/2 * b -/
theorem vector_equality (a b c : Fin 2 → ℝ) 
  (ha : a = ![1, 1])
  (hb : b = ![1, -1])
  (hc : c = ![-1, 2]) :
  c = 1/2 • a - 3/2 • b := by
  sorry

end vector_equality_l560_56056


namespace oranges_used_proof_l560_56051

/-- Calculates the total number of oranges used to make juice -/
def total_oranges (oranges_per_glass : ℕ) (glasses : ℕ) : ℕ :=
  oranges_per_glass * glasses

/-- Proves that the total number of oranges used is 12 -/
theorem oranges_used_proof (oranges_per_glass : ℕ) (glasses : ℕ)
  (h1 : oranges_per_glass = 2)
  (h2 : glasses = 6) :
  total_oranges oranges_per_glass glasses = 12 := by
  sorry

end oranges_used_proof_l560_56051


namespace school_population_l560_56033

/-- Given a school population where:
  * b is the number of boys
  * g is the number of girls
  * t is the number of teachers
  * There are twice as many boys as girls
  * There are four times as many girls as teachers
Prove that the total population is 13t -/
theorem school_population (b g t : ℕ) (h1 : b = 2 * g) (h2 : g = 4 * t) :
  b + g + t = 13 * t := by
  sorry

end school_population_l560_56033


namespace arithmetic_sequence_nth_term_l560_56076

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 1 = 1
  h4 : (a 1) * (a 5) = (a 2) ^ 2

/-- The nth term of the arithmetic sequence is 2n - 1 -/
theorem arithmetic_sequence_nth_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n - 1 := by
  sorry

end arithmetic_sequence_nth_term_l560_56076


namespace exam_max_marks_l560_56044

theorem exam_max_marks (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) :
  percentage = 0.90 →
  scored_marks = 405 →
  percentage * max_marks = scored_marks →
  max_marks = 450 :=
by
  sorry

end exam_max_marks_l560_56044


namespace sqrt6_custom_op_approx_l560_56022

/-- Custom binary operation ¤ -/
def custom_op (x y : ℝ) : ℝ := x^2 + y^2 + 12

/-- Theorem stating that √6 ¤ √6 ≈ 23.999999999999996 -/
theorem sqrt6_custom_op_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1e-14 ∧ |custom_op (Real.sqrt 6) (Real.sqrt 6) - 23.999999999999996| < ε :=
sorry

end sqrt6_custom_op_approx_l560_56022


namespace digital_earth_purpose_theorem_l560_56000

/-- Represents the purpose of Digital Earth -/
structure DigitalEarthPurpose where
  dealWithIssues : Bool
  maximizeResources : Bool
  obtainInformation : Bool
  provideLocationData : Bool

/-- The developers of Digital Earth -/
inductive DigitalEarthDeveloper
  | ISDE
  | CAS

/-- The purpose of Digital Earth as developed by ISDE and CAS -/
def digitalEarthPurpose (developers : List DigitalEarthDeveloper) : DigitalEarthPurpose :=
  { dealWithIssues := true,
    maximizeResources := true,
    obtainInformation := true,
    provideLocationData := false }

theorem digital_earth_purpose_theorem (developers : List DigitalEarthDeveloper) 
  (h1 : DigitalEarthDeveloper.ISDE ∈ developers) 
  (h2 : DigitalEarthDeveloper.CAS ∈ developers) :
  let purpose := digitalEarthPurpose developers
  purpose.dealWithIssues ∧ purpose.maximizeResources ∧ purpose.obtainInformation ∧ ¬purpose.provideLocationData :=
by
  sorry

end digital_earth_purpose_theorem_l560_56000


namespace vec_b_is_correct_l560_56039

def vec_a : ℝ × ℝ := (6, -8)
def vec_b : ℝ × ℝ := (-4, -3)
def vec_c : ℝ × ℝ := (1, 0)

theorem vec_b_is_correct : 
  (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 0) ∧ 
  (vec_b.1^2 + vec_b.2^2 = 25) ∧
  (vec_b.1 * vec_c.1 + vec_b.2 * vec_c.2 < 0) ∧
  (∀ x y : ℝ, (vec_a.1 * x + vec_a.2 * y = 0) ∧ 
              (x^2 + y^2 = 25) ∧ 
              (x * vec_c.1 + y * vec_c.2 < 0) → 
              (x, y) = vec_b) :=
by sorry

end vec_b_is_correct_l560_56039


namespace isosceles_triangle_perimeter_l560_56034

-- Define an isosceles triangle with side lengths 6 and 14
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 14 ∧ b = 14 ∧ c = 6) ∨ (a = 6 ∧ b = 6 ∧ c = 14)

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → Perimeter a b c = 34 :=
by
  sorry

end isosceles_triangle_perimeter_l560_56034


namespace surrounding_circles_radius_l560_56027

theorem surrounding_circles_radius (r : ℝ) : 
  (∃ (A B C D : ℝ × ℝ),
    -- The centers of the surrounding circles form a square
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4*r)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (4*r)^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = (4*r)^2 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = (4*r)^2 ∧
    -- The diagonal of the square
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = (2 + 2*r)^2 ∧
    -- The surrounding circles touch the central circle
    ∃ (O : ℝ × ℝ), (A.1 - O.1)^2 + (A.2 - O.2)^2 = (r + 1)^2) →
  r = 1 + Real.sqrt 2 := by
sorry

end surrounding_circles_radius_l560_56027


namespace equation_solution_l560_56070

theorem equation_solution : {x : ℝ | x^2 = 2*x} = {0, 2} := by sorry

end equation_solution_l560_56070


namespace vector_dot_product_problem_l560_56040

noncomputable def m (a x : ℝ) : ℝ × ℝ := (a * Real.cos x, Real.cos x)
noncomputable def n (b x : ℝ) : ℝ × ℝ := (2 * Real.cos x, b * Real.sin x)

noncomputable def f (a b x : ℝ) : ℝ := (m a x).1 * (n b x).1 + (m a x).2 * (n b x).2

theorem vector_dot_product_problem (a b : ℝ) :
  (∃ x, f a b x = 2) ∧ 
  (f a b (π/3) = 1/2 + Real.sqrt 3/2) →
  (∃ x_min ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f a b x_min ≤ f a b x) ∧
  (∃ x_max ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f a b x ≤ f a b x_max) ∧
  (∀ θ, 0 < θ ∧ θ < π ∧ f a b (θ/2) = 3/2 → Real.tan θ = -(4 + Real.sqrt 7)/3) :=
by sorry

end vector_dot_product_problem_l560_56040


namespace ellipse_properties_l560_56014

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The sum of distances from a point to the foci of the ellipse -/
def Ellipse.foci_distance_sum (e : Ellipse) (x y : ℝ) : ℝ :=
  2 * e.a

theorem ellipse_properties (e : Ellipse) 
    (h_point : e.equation 0 (Real.sqrt 3))
    (h_sum : e.foci_distance_sum 0 (Real.sqrt 3) = 4) :
  e.a = 2 ∧ e.b = Real.sqrt 3 ∧ 
  (∀ x y, e.equation x y ↔ x^2/4 + y^2/3 = 1) ∧
  e.b * 2 = 2 * Real.sqrt 3 ∧
  2 * Real.sqrt (e.a^2 - e.b^2) = 2 := by
  sorry

end ellipse_properties_l560_56014


namespace collinear_points_x_value_l560_56030

/-- Given three points A, B, and C in 2D space, 
    returns true if they are collinear -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

/-- The main theorem: if A(-1,-2), B(4,8), and C(5,x) are collinear, 
    then x = 10 -/
theorem collinear_points_x_value :
  collinear (-1, -2) (4, 8) (5, x) → x = 10 := by
  sorry

end collinear_points_x_value_l560_56030


namespace max_rectangles_after_removal_l560_56078

/-- Represents a grid with some squares removed -/
structure Grid :=
  (size : Nat)
  (removedSquares : List (Nat × Nat × Nat))

/-- Represents a rectangle -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- The maximum number of rectangles that can be cut from a grid -/
def maxRectangles (g : Grid) (r : Rectangle) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem max_rectangles_after_removal :
  let initialGrid : Grid := { size := 8, removedSquares := [(2, 2, 3)] }
  let targetRectangle : Rectangle := { width := 1, height := 3 }
  maxRectangles initialGrid targetRectangle = 16 := by
  sorry

end max_rectangles_after_removal_l560_56078


namespace hiring_range_l560_56088

/-- The number of standard deviations that includes all accepted ages -/
def num_std_dev (avg : ℕ) (std_dev : ℕ) (num_ages : ℕ) : ℚ :=
  (num_ages - 1) / (2 * std_dev)

theorem hiring_range (avg : ℕ) (std_dev : ℕ) (num_ages : ℕ)
  (h_avg : avg = 20)
  (h_std_dev : std_dev = 8)
  (h_num_ages : num_ages = 17) :
  num_std_dev avg std_dev num_ages = 1 := by
sorry

end hiring_range_l560_56088


namespace emilys_spending_l560_56007

/-- Given Emily's spending pattern over four days and the total amount spent,
    prove that the amount she spent on Friday is equal to the total divided by 18. -/
theorem emilys_spending (X Y : ℝ) : 
  X > 0 →  -- Assuming X is positive
  Y > 0 →  -- Assuming Y is positive
  X + 2*X + 3*X + 4*(3*X) = Y →  -- Total spending equation
  X = Y / 18 := by
sorry

end emilys_spending_l560_56007


namespace largest_perfect_square_factor_of_1800_l560_56050

theorem largest_perfect_square_factor_of_1800 : 
  ∃ (n : ℕ), n * n = 3600 ∧ 
  3600 ∣ 1800 ∧
  ∀ (m : ℕ), m * m ∣ 1800 → m * m ≤ 3600 :=
sorry

end largest_perfect_square_factor_of_1800_l560_56050


namespace geometric_sequence_ratio_sum_l560_56003

theorem geometric_sequence_ratio_sum (m : ℝ) (a₂ a₃ b₂ b₃ : ℝ) :
  m ≠ 0 →
  (∃ x : ℝ, x ≠ 1 ∧ a₂ = m * x ∧ a₃ = m * x^2) →
  (∃ y : ℝ, y ≠ 1 ∧ b₂ = m * y ∧ b₃ = m * y^2) →
  (∀ x y : ℝ, a₂ = m * x ∧ a₃ = m * x^2 ∧ b₂ = m * y ∧ b₃ = m * y^2 → x ≠ y) →
  a₃ - b₃ = 3 * (a₂ - b₂) →
  ∃ x y : ℝ, (a₂ = m * x ∧ a₃ = m * x^2 ∧ b₂ = m * y ∧ b₃ = m * y^2) ∧ x + y = 3 :=
by sorry

end geometric_sequence_ratio_sum_l560_56003


namespace mary_paper_problem_l560_56079

/-- Represents the initial state of Mary's paper pieces -/
structure InitialState where
  squares : ℕ
  triangles : ℕ
  total_pieces : ℕ
  total_pieces_eq : squares + triangles = total_pieces

/-- Represents the final state after cutting some squares -/
structure FinalState where
  initial : InitialState
  squares_cut : ℕ
  total_vertices : ℕ
  squares_cut_constraint : squares_cut ≤ initial.squares

theorem mary_paper_problem (state : InitialState) (final : FinalState)
  (h_initial_pieces : state.total_pieces = 10)
  (h_squares_cut : final.squares_cut = 3)
  (h_final_pieces : state.total_pieces + final.squares_cut = 13)
  (h_final_vertices : final.total_vertices = 42)
  : state.triangles = 4 := by
  sorry

end mary_paper_problem_l560_56079


namespace mean_calculation_l560_56068

def set1 : List ℝ := [28, 42, 78, 104]
def set2 : List ℝ := [128, 255, 511, 1023]

theorem mean_calculation (x : ℝ) :
  (List.sum set1 + x) / 5 = 90 →
  (List.sum set2 + x) / 5 = 423 := by
  sorry

end mean_calculation_l560_56068


namespace reach_all_integers_l560_56002

/-- Represents the allowed operations on positive integers -/
inductive Operation
  | append4 : Operation
  | append0 : Operation
  | divideBy2 : Operation

/-- Applies an operation to a positive integer -/
def applyOperation (n : ℕ+) (op : Operation) : ℕ+ :=
  match op with
  | Operation.append4 => ⟨10 * n.val + 4, by sorry⟩
  | Operation.append0 => ⟨10 * n.val, by sorry⟩
  | Operation.divideBy2 => if n.val % 2 = 0 then ⟨n.val / 2, by sorry⟩ else n

/-- Applies a sequence of operations to a positive integer -/
def applyOperations (n : ℕ+) (ops : List Operation) : ℕ+ :=
  ops.foldl applyOperation n

/-- Theorem stating that any positive integer can be reached from 4 using the allowed operations -/
theorem reach_all_integers (n : ℕ+) : 
  ∃ (ops : List Operation), applyOperations ⟨4, by norm_num⟩ ops = n := by
  sorry

end reach_all_integers_l560_56002


namespace complex_magnitude_l560_56036

theorem complex_magnitude (z : ℂ) (h : (z - Complex.I) * (2 - Complex.I) = Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l560_56036


namespace car_speed_problem_l560_56052

theorem car_speed_problem (D : ℝ) (D_pos : D > 0) : 
  let total_time := D / 40
  let first_part_time := (0.75 * D) / 60
  let second_part_time := total_time - first_part_time
  let s := (0.25 * D) / second_part_time
  s = 20 := by sorry

end car_speed_problem_l560_56052


namespace student_arrangement_count_l560_56099

/-- The number of ways to arrange students among attractions -/
def arrange_students (n_students : ℕ) (n_attractions : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange students among attractions when two specific students are at the same attraction -/
def arrange_students_with_pair (n_students : ℕ) (n_attractions : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of arrangements under given conditions -/
theorem student_arrangement_count :
  let n_students : ℕ := 4
  let n_attractions : ℕ := 3
  arrange_students n_students n_attractions - arrange_students_with_pair n_students n_attractions = 30 :=
sorry

end student_arrangement_count_l560_56099


namespace rubber_band_calculation_l560_56010

/-- The number of rubber bands in a small ball -/
def small_ball_rubber_bands : ℕ := 50

/-- The number of rubber bands in a large ball -/
def large_ball_rubber_bands : ℕ := 300

/-- The total number of rubber bands -/
def total_rubber_bands : ℕ := 5000

/-- The number of small balls made -/
def small_balls_made : ℕ := 22

/-- The number of large balls that can be made with remaining rubber bands -/
def large_balls_possible : ℕ := 13

theorem rubber_band_calculation :
  small_ball_rubber_bands * small_balls_made +
  large_ball_rubber_bands * large_balls_possible = total_rubber_bands :=
by sorry

end rubber_band_calculation_l560_56010


namespace picnic_theorem_l560_56041

def picnic_problem (people : ℕ) (sandwich_price : ℚ) (fruit_salad_price : ℚ) (soda_price : ℚ) (sodas_per_person : ℕ) (snack_bags : ℕ) (total_spent : ℚ) : Prop :=
  let sandwich_cost := people * sandwich_price
  let fruit_salad_cost := people * fruit_salad_price
  let soda_cost := people * sodas_per_person * soda_price
  let food_cost := sandwich_cost + fruit_salad_cost + soda_cost
  let snack_cost := total_spent - food_cost
  snack_cost / snack_bags = 4

theorem picnic_theorem : 
  picnic_problem 4 5 3 2 2 3 60 := by
  sorry

end picnic_theorem_l560_56041


namespace sum_of_degrees_l560_56074

/-- Represents the degrees of four people in a specific ratio -/
structure DegreeRatio :=
  (a b c d : ℕ)
  (ratio : a = 5 ∧ b = 4 ∧ c = 6 ∧ d = 3)

/-- The theorem stating the sum of degrees given the ratio and highest degree -/
theorem sum_of_degrees (r : DegreeRatio) (highest_degree : ℕ) 
  (h : highest_degree = 150) : 
  (r.a + r.b + r.c + r.d) * (highest_degree / r.c) = 450 := by
  sorry

end sum_of_degrees_l560_56074


namespace pencils_bought_l560_56047

/-- 
Given:
- Amy initially had 3 pencils
- Amy now has a total of 10 pencils
Prove that Amy bought 7 pencils at the school store
-/
theorem pencils_bought (initial_pencils : ℕ) (total_pencils : ℕ) (bought_pencils : ℕ) : 
  initial_pencils = 3 → 
  total_pencils = 10 → 
  bought_pencils = total_pencils - initial_pencils → 
  bought_pencils = 7 := by
  sorry

end pencils_bought_l560_56047


namespace diamond_paths_count_l560_56059

/-- Represents a position in the diamond grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents a move in the diamond grid -/
inductive Move
  | Right
  | Down
  | DiagonalDown

/-- The diamond-shaped grid containing the word "DIAMOND" -/
def diamond_grid : List (List Char) := sorry

/-- Check if a move is valid in the diamond grid -/
def is_valid_move (grid : List (List Char)) (pos : Position) (move : Move) : Bool := sorry

/-- Get the next position after a move -/
def next_position (pos : Position) (move : Move) : Position := sorry

/-- Check if a path spells "DIAMOND" -/
def spells_diamond (grid : List (List Char)) (path : List Move) : Bool := sorry

/-- Count the number of valid paths spelling "DIAMOND" -/
def count_diamond_paths (grid : List (List Char)) : ℕ := sorry

theorem diamond_paths_count :
  count_diamond_paths diamond_grid = 64 := by sorry

end diamond_paths_count_l560_56059


namespace simplify_and_evaluate_l560_56046

theorem simplify_and_evaluate : 
  ∀ (a b : ℝ), 
    (a + 2*b)^2 - (a + b)*(a - b) = 4*a*b + 5*b^2 ∧
    (((-1/2) + 2*2)^2 - ((-1/2) + 2)*((-1/2) - 2) = 16) :=
by sorry

end simplify_and_evaluate_l560_56046


namespace valid_gift_wrapping_combinations_l560_56087

/-- The number of wrapping paper varieties -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 3

/-- The number of gift card types -/
def gift_card_types : ℕ := 5

/-- The number of invalid combinations (red ribbon with birthday card) -/
def invalid_combinations : ℕ := 1

/-- Theorem stating the number of valid gift wrapping combinations -/
theorem valid_gift_wrapping_combinations :
  wrapping_paper_varieties * ribbon_colors * gift_card_types - invalid_combinations = 149 := by
sorry

end valid_gift_wrapping_combinations_l560_56087


namespace intersection_of_M_and_N_l560_56069

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end intersection_of_M_and_N_l560_56069


namespace circles_are_externally_tangent_l560_56020

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii. -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The first circle: x^2 + y^2 = 4 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The second circle: x^2 + y^2 - 10x + 16 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 16 = 0

theorem circles_are_externally_tangent :
  externally_tangent (0, 0) (5, 0) 2 3 :=
sorry

end circles_are_externally_tangent_l560_56020


namespace find_a_value_l560_56066

noncomputable section

open Set Real

def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem find_a_value : ∃ (a : ℝ), 
  (∀ x ∈ (Ioo 0 2), StrictAntiOn (f a) (Ioo 0 2)) ∧
  (∀ x ∈ (Ioi 2), StrictMonoOn (f a) (Ioi 2)) ∧
  a = 4 := by
  sorry

end

end find_a_value_l560_56066


namespace sequence_properties_l560_56058

def sequence_a (n : ℕ) : ℝ := 2^n

def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - 2

def sequence_b (n : ℕ) : ℝ := (n + 1 : ℝ) * sequence_a n

def sum_T (n : ℕ) : ℝ := n * 2^(n + 1)

theorem sequence_properties (n : ℕ) :
  (∀ k, sum_S k = 2 * sequence_a k - 2) →
  (sequence_a n = 2^n ∧
   sum_T n = n * 2^(n + 1)) := by
  sorry

end sequence_properties_l560_56058


namespace quadratic_roots_sum_and_product_l560_56048

theorem quadratic_roots_sum_and_product :
  let f : ℝ → ℝ := λ x => x^2 - 18*x + 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ 
    (x₁ + x₂ = 18) ∧ (x₁ * x₂ = 16) := by
  sorry

end quadratic_roots_sum_and_product_l560_56048


namespace stream_speed_stream_speed_is_24_l560_56037

/-- Given a boat with speed in still water and the relationship between upstream and downstream times,
    calculate the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (upstream_time downstream_time : ℝ) : ℝ :=
  let stream_speed := (boat_speed : ℝ) / 3
  have h1 : upstream_time = 2 * downstream_time := by sorry
  have h2 : boat_speed = 72 := by sorry
  have h3 : upstream_time * (boat_speed - stream_speed) = downstream_time * (boat_speed + stream_speed) := by sorry
  stream_speed

/-- The speed of the stream is 24 kmph. -/
theorem stream_speed_is_24 : stream_speed 72 1 0.5 = 24 := by sorry

end stream_speed_stream_speed_is_24_l560_56037


namespace parallel_vectors_x_value_l560_56031

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = -4 := by
sorry

end parallel_vectors_x_value_l560_56031


namespace fraction_sum_product_l560_56006

theorem fraction_sum_product : 
  24 * (243 / 3 + 49 / 7 + 16 / 8 + 4 / 2 + 2) = 2256 := by
  sorry

end fraction_sum_product_l560_56006


namespace tromino_bounds_l560_56060

/-- A tromino is a 1 x 3 rectangle that covers exactly three squares on a board. -/
structure Tromino

/-- The board is an n x n grid where trominoes can be placed. -/
structure Board (n : ℕ) where
  size : n > 0

/-- f(n) is the smallest number of trominoes required to stop any more being placed on an n x n board. -/
noncomputable def f (n : ℕ) : ℕ :=
  sorry

/-- For all positive n, there exist real numbers h and k such that
    (n^2 / 7) + hn ≤ f(n) ≤ (n^2 / 5) + kn -/
theorem tromino_bounds (n : ℕ) (b : Board n) :
  ∃ (h k : ℝ), (n^2 / 7 : ℝ) + h * n ≤ f n ∧ (f n : ℝ) ≤ n^2 / 5 + k * n :=
sorry

end tromino_bounds_l560_56060


namespace angle_triple_complement_l560_56089

theorem angle_triple_complement : ∃ x : ℝ, x + (180 - x) = 180 ∧ x = 3 * (180 - x) ∧ x = 135 := by
  sorry

end angle_triple_complement_l560_56089


namespace pet_store_white_cats_l560_56004

theorem pet_store_white_cats 
  (total : ℕ) 
  (black : ℕ) 
  (gray : ℕ) 
  (h1 : total = 15) 
  (h2 : black = 10) 
  (h3 : gray = 3) 
  (h4 : ∃ white : ℕ, total = white + black + gray) : 
  ∃ white : ℕ, white = 2 ∧ total = white + black + gray :=
by
  sorry

end pet_store_white_cats_l560_56004


namespace student_meeting_distance_l560_56072

theorem student_meeting_distance (initial_distance : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  initial_distance = 350 →
  time = 100 →
  speed1 = 1.6 →
  speed2 = 1.9 →
  speed2 * time = 190 :=
by sorry

end student_meeting_distance_l560_56072


namespace x_eq_2_sufficient_not_necessary_l560_56083

theorem x_eq_2_sufficient_not_necessary :
  (∀ x : ℝ, x = 2 → (x - 2) * (x + 5) = 0) ∧
  (∃ x : ℝ, (x - 2) * (x + 5) = 0 ∧ x ≠ 2) :=
by sorry

end x_eq_2_sufficient_not_necessary_l560_56083


namespace soccer_ball_holes_percentage_l560_56065

theorem soccer_ball_holes_percentage 
  (total_balls : ℕ) 
  (successfully_inflated : ℕ) 
  (overinflation_rate : ℚ) :
  total_balls = 100 →
  successfully_inflated = 48 →
  overinflation_rate = 1/5 →
  ∃ (x : ℚ), 
    0 ≤ x ∧ 
    x ≤ 1 ∧ 
    (1 - x) * (1 - overinflation_rate) * total_balls = successfully_inflated ∧
    x = 2/5 := by
  sorry

end soccer_ball_holes_percentage_l560_56065


namespace vacation_cost_l560_56080

theorem vacation_cost (num_people : ℕ) (plane_ticket_cost : ℕ) (hotel_cost_per_day : ℕ) (num_days : ℕ) : 
  num_people = 2 → 
  plane_ticket_cost = 24 → 
  hotel_cost_per_day = 12 → 
  num_days = 3 → 
  num_people * plane_ticket_cost + num_people * hotel_cost_per_day * num_days = 120 := by
sorry

end vacation_cost_l560_56080


namespace trig_simplification_l560_56009

theorem trig_simplification :
  (Real.cos (20 * π / 180) * Real.sqrt (1 - Real.cos (40 * π / 180))) / Real.cos (50 * π / 180) = Real.sqrt 2 / 2 := by
sorry

end trig_simplification_l560_56009


namespace min_value_expression_l560_56011

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x * y = 4) :
  (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 16 := by
  sorry

end min_value_expression_l560_56011


namespace john_total_spent_l560_56094

def tshirt_price : ℕ := 20
def num_tshirts : ℕ := 3
def pants_price : ℕ := 50

def total_spent : ℕ := tshirt_price * num_tshirts + pants_price

theorem john_total_spent : total_spent = 110 := by
  sorry

end john_total_spent_l560_56094


namespace inequality_and_equality_condition_l560_56055

theorem inequality_and_equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c)) ∧
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) = 3 / (1 + a * b * c) ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end inequality_and_equality_condition_l560_56055


namespace letter_F_perimeter_is_19_l560_56028

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of the letter F given the specified conditions -/
def letter_F_perimeter (large : Rectangle) (small : Rectangle) (offset : ℝ) : ℝ :=
  2 * large.height + -- vertical sides of large rectangle
  (large.width - small.width) + -- uncovered top of large rectangle
  small.width -- bottom of small rectangle

/-- Theorem stating that the perimeter of the letter F is 19 inches -/
theorem letter_F_perimeter_is_19 :
  let large : Rectangle := { width := 2, height := 6 }
  let small : Rectangle := { width := 2, height := 2 }
  let offset : ℝ := 1
  letter_F_perimeter large small offset = 19 := by
  sorry

#eval letter_F_perimeter { width := 2, height := 6 } { width := 2, height := 2 } 1

end letter_F_perimeter_is_19_l560_56028


namespace pears_picked_total_l560_56019

/-- The number of pears Sara picked -/
def sara_pears : ℕ := 6

/-- The number of pears Tim picked -/
def tim_pears : ℕ := 5

/-- The total number of pears picked -/
def total_pears : ℕ := sara_pears + tim_pears

theorem pears_picked_total : total_pears = 11 := by
  sorry

end pears_picked_total_l560_56019
