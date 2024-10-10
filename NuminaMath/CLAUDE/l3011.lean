import Mathlib

namespace prob_ace_king_queen_value_l3011_301164

/-- The probability of drawing an Ace, then a King, then a Queen from a standard deck of 52 cards without replacement -/
def prob_ace_king_queen : ℚ :=
  let total_cards : ℕ := 52
  let num_aces : ℕ := 4
  let num_kings : ℕ := 4
  let num_queens : ℕ := 4
  (num_aces : ℚ) / total_cards *
  (num_kings : ℚ) / (total_cards - 1) *
  (num_queens : ℚ) / (total_cards - 2)

theorem prob_ace_king_queen_value : prob_ace_king_queen = 8 / 16575 := by
  sorry

end prob_ace_king_queen_value_l3011_301164


namespace friendship_fraction_l3011_301150

theorem friendship_fraction :
  ∀ (x y : ℕ), 
    x > 0 → y > 0 →
    (1 : ℚ) / 3 * y = (2 : ℚ) / 5 * x →
    ((1 : ℚ) / 3 * y + (2 : ℚ) / 5 * x) / (x + y : ℚ) = 4 / 11 :=
by sorry

end friendship_fraction_l3011_301150


namespace price_change_theorem_l3011_301105

theorem price_change_theorem (initial_price : ℝ) (x : ℝ) : 
  initial_price > 0 →
  let price1 := initial_price * (1 + 0.3)
  let price2 := price1 * (1 - 0.15)
  let price3 := price2 * (1 + 0.1)
  let price4 := price3 * (1 - x / 100)
  price4 = initial_price →
  x = 18 := by
sorry

end price_change_theorem_l3011_301105


namespace oranges_sold_l3011_301135

def total_bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def rotten_oranges : ℕ := 50
def oranges_for_juice : ℕ := 30

theorem oranges_sold : 
  (total_bags * oranges_per_bag - rotten_oranges - oranges_for_juice) = 220 := by
  sorry

end oranges_sold_l3011_301135


namespace inequality_and_range_proof_fraction_comparison_l3011_301124

theorem inequality_and_range_proof :
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → ∀ (x : ℝ), x ∈ Set.Icc (-2) 2 →
    |3*a + b| + |a - b| ≥ |a| * (|x + 1| + |x - 1|)) ∧
  (∀ (x : ℝ), (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
    |3*a + b| + |a - b| ≥ |a| * (|x + 1| + |x - 1|)) →
    x ∈ Set.Icc (-2) 2) :=
sorry

theorem fraction_comparison :
  ∀ (a b : ℝ), a ∈ Set.Ioo 0 1 → b ∈ Set.Ioo 0 1 →
    1 / (a * b) + 1 > 1 / a + 1 / b :=
sorry

end inequality_and_range_proof_fraction_comparison_l3011_301124


namespace sum_of_divisors_180_l3011_301197

def sumOfDivisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_180 : sumOfDivisors 180 = 546 := by sorry

end sum_of_divisors_180_l3011_301197


namespace cos_1275_degrees_l3011_301189

theorem cos_1275_degrees :
  Real.cos (1275 * π / 180) = -(Real.sqrt 2 + Real.sqrt 6) / 4 := by
  sorry

end cos_1275_degrees_l3011_301189


namespace point_on_curve_iff_satisfies_equation_l3011_301137

-- Define a curve C in 2D space
def Curve (F : ℝ → ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | F p.1 p.2 = 0}

-- Define a point P
def Point (a b : ℝ) : ℝ × ℝ := (a, b)

-- Theorem statement
theorem point_on_curve_iff_satisfies_equation (F : ℝ → ℝ → ℝ) (a b : ℝ) :
  Point a b ∈ Curve F ↔ F a b = 0 := by
  sorry

end point_on_curve_iff_satisfies_equation_l3011_301137


namespace min_value_expression_l3011_301112

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 16) / Real.sqrt (x - 4) ≥ 4 * Real.sqrt 5 ∧
  (∃ x₀ : ℝ, x₀ > 4 ∧ (x₀ + 16) / Real.sqrt (x₀ - 4) = 4 * Real.sqrt 5 ∧ x₀ = 24) :=
by sorry

end min_value_expression_l3011_301112


namespace weight_difference_l3011_301113

/-- Weights of different shapes in grams -/
def round_weight : ℕ := 200
def square_weight : ℕ := 300
def triangular_weight : ℕ := 150

/-- Number of weights on the left pan -/
def left_square : ℕ := 1
def left_triangular : ℕ := 2
def left_round : ℕ := 3

/-- Number of weights on the right pan -/
def right_triangular : ℕ := 1
def right_round : ℕ := 2
def right_square : ℕ := 3

/-- Total weight on the left pan -/
def left_total : ℕ := 
  left_square * square_weight + 
  left_triangular * triangular_weight + 
  left_round * round_weight

/-- Total weight on the right pan -/
def right_total : ℕ := 
  right_triangular * triangular_weight + 
  right_round * round_weight + 
  right_square * square_weight

/-- The difference in weight between the right and left pans -/
theorem weight_difference : right_total - left_total = 250 := by
  sorry

end weight_difference_l3011_301113


namespace isabel_homework_problems_l3011_301171

/-- Given the number of pages for math and reading homework, and the number of problems per page,
    calculate the total number of problems to complete. -/
def total_problems (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  (math_pages + reading_pages) * problems_per_page

/-- Prove that Isabel's total number of homework problems is 30. -/
theorem isabel_homework_problems :
  total_problems 2 4 5 = 30 := by
  sorry

end isabel_homework_problems_l3011_301171


namespace parabola_equation_l3011_301177

/-- Given a parabola with directrix x = -7, its standard equation is y^2 = 28x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ x = -7) →  -- directrix equation
  (∃ a b c : ℝ, ∀ x y, p (x, y) ↔ y^2 = 28*x + b*y + c) -- standard form of parabola
  :=
sorry

end parabola_equation_l3011_301177


namespace total_stickers_l3011_301136

/-- Given 25 stickers on each page and 35 pages of stickers, 
    the total number of stickers is 875. -/
theorem total_stickers (stickers_per_page pages : ℕ) : 
  stickers_per_page = 25 → pages = 35 → stickers_per_page * pages = 875 := by
  sorry

end total_stickers_l3011_301136


namespace bottle_display_sum_l3011_301166

/-- Represents a triangular bottle display -/
structure BottleDisplay where
  firstRow : ℕ
  commonDiff : ℕ
  lastRow : ℕ

/-- Calculates the total number of bottles in the display -/
def totalBottles (display : BottleDisplay) : ℕ :=
  let n := (display.lastRow - display.firstRow) / display.commonDiff + 1
  n * (display.firstRow + display.lastRow) / 2

/-- Theorem stating the total number of bottles in the specific display -/
theorem bottle_display_sum :
  let display : BottleDisplay := ⟨3, 3, 30⟩
  totalBottles display = 165 := by
  sorry

end bottle_display_sum_l3011_301166


namespace diet_soda_ratio_l3011_301104

theorem diet_soda_ratio (total bottles : ℕ) (regular_soda diet_soda fruit_juice sparkling_water : ℕ) :
  total = 60 →
  regular_soda = 18 →
  diet_soda = 14 →
  fruit_juice = 8 →
  sparkling_water = 10 →
  total = regular_soda + diet_soda + fruit_juice + sparkling_water + (total - regular_soda - diet_soda - fruit_juice - sparkling_water) →
  (diet_soda : ℚ) / total = 7 / 30 :=
by
  sorry

end diet_soda_ratio_l3011_301104


namespace class_average_score_l3011_301190

theorem class_average_score (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_avg : ℚ) (group2_avg : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 10 →
  group2_students = 10 →
  group1_avg = 80 →
  group2_avg = 60 →
  (group1_students * group1_avg + group2_students * group2_avg) / total_students = 70 := by
  sorry

end class_average_score_l3011_301190


namespace age_ratio_proof_l3011_301134

/-- Given the ages of Tony and Belinda, prove that their age ratio is 5/2 -/
theorem age_ratio_proof (tony_age belinda_age : ℕ) : 
  tony_age = 16 →
  belinda_age = 40 →
  tony_age + belinda_age = 56 →
  ∃ (k : ℕ), belinda_age = k * tony_age + 8 →
  (belinda_age : ℚ) / (tony_age : ℚ) = 5 / 2 := by
  sorry

end age_ratio_proof_l3011_301134


namespace partial_fraction_decomposition_l3011_301163

theorem partial_fraction_decomposition (x : ℝ) (A B : ℚ) : 
  (5 * x - 3) / ((x - 3) * (x + 6)) = A / (x - 3) + B / (x + 6) ↔ 
  A = 4/3 ∧ B = 11/3 := by
sorry

end partial_fraction_decomposition_l3011_301163


namespace a_upper_bound_l3011_301122

/-- Given a real number a, we define a function f and its derivative f' --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2

def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

/-- We define g as the sum of f and f' --/
def g (a : ℝ) (x : ℝ) : ℝ := f a x + f' a x

/-- Main theorem: If there exists x in [1, 3] such that g(x) ≤ 0, then a ≤ 9/4 --/
theorem a_upper_bound (a : ℝ) (h : ∃ x ∈ Set.Icc 1 3, g a x ≤ 0) : a ≤ 9/4 := by
  sorry

end a_upper_bound_l3011_301122


namespace line_equation_l3011_301142

/-- The ellipse E defined by the equation x^2/4 + y^2/2 = 1 -/
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 2 = 1}

/-- A line intersecting the ellipse E -/
def l : Set (ℝ × ℝ) := sorry

/-- Point A on the ellipse E and line l -/
def A : ℝ × ℝ := sorry

/-- Point B on the ellipse E and line l -/
def B : ℝ × ℝ := sorry

/-- The midpoint of AB is (1/2, -1) -/
axiom midpoint_AB : (A.1 + B.1) / 2 = 1/2 ∧ (A.2 + B.2) / 2 = -1

/-- Theorem stating that the equation of line l is x - 4y - 9/2 = 0 -/
theorem line_equation : l = {p : ℝ × ℝ | p.1 - 4 * p.2 - 9/2 = 0} := by
  sorry

end line_equation_l3011_301142


namespace a_greater_than_b_l3011_301154

theorem a_greater_than_b (x y : ℝ) (h1 : x < y) (h2 : y < 0) :
  (x^2 + y^2) * (x - y) > (x^2 - y^2) * (x + y) := by
  sorry

end a_greater_than_b_l3011_301154


namespace inscribed_circle_theorem_l3011_301109

def equilateral_triangle_with_inscribed_circle 
  (radius : ℝ) (height : ℝ) (x : ℝ) : Prop :=
  radius = 3/16 ∧ 
  height = 3 * radius ∧ 
  x = height - 1/2

theorem inscribed_circle_theorem :
  ∀ (radius height x : ℝ),
    equilateral_triangle_with_inscribed_circle radius height x →
    x = 1/16 := by
  sorry

end inscribed_circle_theorem_l3011_301109


namespace linear_function_not_in_third_quadrant_l3011_301118

/-- A point (x, y) is in the third quadrant if both x and y are negative. -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The linear function y = kx - k -/
def f (k x : ℝ) : ℝ := k * x - k

theorem linear_function_not_in_third_quadrant (k : ℝ) (h : k < 0) :
  ∀ x y : ℝ, f k x = y → ¬ in_third_quadrant x y :=
by sorry

end linear_function_not_in_third_quadrant_l3011_301118


namespace total_money_is_36000_l3011_301110

/-- The number of phones Vivienne has -/
def vivienne_phones : ℕ := 40

/-- The number of additional phones Aliyah has compared to Vivienne -/
def aliyah_extra_phones : ℕ := 10

/-- The price at which each phone is sold -/
def price_per_phone : ℕ := 400

/-- The total amount of money Aliyah and Vivienne have together after selling their phones -/
def total_money : ℕ := (vivienne_phones + (vivienne_phones + aliyah_extra_phones)) * price_per_phone

/-- Theorem stating that the total amount of money Aliyah and Vivienne have together is $36,000 -/
theorem total_money_is_36000 : total_money = 36000 := by
  sorry

end total_money_is_36000_l3011_301110


namespace ratio_when_a_is_20_percent_more_than_b_l3011_301138

theorem ratio_when_a_is_20_percent_more_than_b (A B : ℝ) (h : A = 1.2 * B) : A / B = 6 / 5 := by
  sorry

end ratio_when_a_is_20_percent_more_than_b_l3011_301138


namespace largest_B_divisible_by_4_l3011_301191

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def six_digit_number (B : ℕ) : ℕ := 400000 + 5000 + 784 + B * 10000

theorem largest_B_divisible_by_4 :
  ∀ B : ℕ, B ≤ 9 →
    (is_divisible_by_4 (six_digit_number B)) →
    (∀ C : ℕ, C ≤ 9 → is_divisible_by_4 (six_digit_number C) → C ≤ B) →
    B = 9 :=
by sorry

end largest_B_divisible_by_4_l3011_301191


namespace triangle_side_length_l3011_301131

theorem triangle_side_length (A B : Real) (a b : Real) :
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  a = 1 →
  b = a * Real.sin B / Real.sin A →
  b = Real.sqrt 2 := by
sorry

end triangle_side_length_l3011_301131


namespace original_average_problem_l3011_301147

theorem original_average_problem (n : ℕ) (original_avg new_avg added : ℝ) : 
  n = 15 → 
  new_avg = 51 → 
  added = 11 → 
  (n : ℝ) * new_avg = (n : ℝ) * (original_avg + added) → 
  original_avg = 40 := by
sorry

end original_average_problem_l3011_301147


namespace square_cut_rectangle_perimeter_l3011_301139

/-- Given a square with perimeter 20 cm cut into two rectangles, 
    where one rectangle has perimeter 16 cm, 
    prove that the other rectangle has perimeter 14 cm. -/
theorem square_cut_rectangle_perimeter :
  ∀ (square_perimeter : ℝ) (rectangle1_perimeter : ℝ),
    square_perimeter = 20 →
    rectangle1_perimeter = 16 →
    ∃ (rectangle2_perimeter : ℝ),
      rectangle2_perimeter = 14 ∧
      rectangle1_perimeter + rectangle2_perimeter = square_perimeter + 10 :=
by sorry

end square_cut_rectangle_perimeter_l3011_301139


namespace line_slope_point_sum_l3011_301192

/-- Represents a line in the form y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- Checks if a point (x, y) is on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.c

theorem line_slope_point_sum (l : Line) :
  l.m = -5 →
  l.contains 2 3 →
  l.m + l.c = 8 := by
  sorry

end line_slope_point_sum_l3011_301192


namespace problem_solution_l3011_301194

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^5 + 2*y^3) / 8 = 46.375 := by
  sorry

end problem_solution_l3011_301194


namespace xiao_ming_age_l3011_301102

theorem xiao_ming_age : ∃ (xiao_ming_age : ℕ), 
  (∃ (dad_age : ℕ), 
    dad_age - xiao_ming_age = 28 ∧ 
    dad_age = 3 * xiao_ming_age) → 
  xiao_ming_age = 14 := by
  sorry

end xiao_ming_age_l3011_301102


namespace smallest_positive_d_smallest_d_is_zero_l3011_301159

theorem smallest_positive_d (d : ℝ) (hd : d > 0) :
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + d * (x - y)^2 ≥ (x + y) / 2 := by
  sorry

/-- The smallest positive real number d that satisfies the inequality for all nonnegative x and y is 0 -/
theorem smallest_d_is_zero :
  ∀ ε > 0, ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) + ε * (x - y)^2 < (x + y) / 2 := by
  sorry

end smallest_positive_d_smallest_d_is_zero_l3011_301159


namespace arithmetic_sequence_8th_term_l3011_301165

/-- 
Given an arithmetic sequence with 30 terms, first term 4, and last term 88,
prove that the 8th term is equal to 676/29.
-/
theorem arithmetic_sequence_8th_term 
  (a₁ : ℚ) 
  (aₙ : ℚ) 
  (n : ℕ) 
  (h₁ : a₁ = 4) 
  (h₂ : aₙ = 88) 
  (h₃ : n = 30) : 
  a₁ + 7 * ((aₙ - a₁) / (n - 1)) = 676 / 29 := by
  sorry


end arithmetic_sequence_8th_term_l3011_301165


namespace chocolate_milk_remaining_l3011_301145

/-- The amount of chocolate milk remaining after drinking some on two consecutive days. -/
theorem chocolate_milk_remaining (initial : ℝ) (day1 : ℝ) (day2 : ℝ) (h1 : initial = 1.6) (h2 : day1 = 0.8) (h3 : day2 = 0.3) :
  initial - day1 - day2 = 0.5 := by
  sorry

end chocolate_milk_remaining_l3011_301145


namespace special_line_equation_l3011_301149

/-- A line passing through two points -/
structure Line where
  p : ℝ × ℝ
  q : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfiesEquation (point : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * point.1 + eq.b * point.2 + eq.c = 0

/-- The line l passing through P(x, y) and Q(4x + 2y, x + 3y) -/
def specialLine (x y : ℝ) : Line :=
  { p := (x, y)
    q := (4*x + 2*y, x + 3*y) }

/-- The possible equations for the special line -/
def possibleEquations : List LineEquation :=
  [{ a := 1, b := -1, c := 0 },  -- x - y = 0
   { a := 1, b := -2, c := 0 }]  -- x - 2y = 0

theorem special_line_equation (x y : ℝ) :
  ∃ (eq : LineEquation), eq ∈ possibleEquations ∧
    satisfiesEquation (specialLine x y).p eq ∧
    satisfiesEquation (specialLine x y).q eq :=
  sorry


end special_line_equation_l3011_301149


namespace solve_for_y_l3011_301128

theorem solve_for_y (x y : ℝ) (h1 : x + 2*y = 12) (h2 : x = 6) : y = 3 := by
  sorry

end solve_for_y_l3011_301128


namespace cross_figure_perimeter_l3011_301127

/-- A cross-shaped figure formed by five identical squares -/
structure CrossFigure where
  /-- The side length of each square in the figure -/
  side_length : ℝ
  /-- The total area of the figure is 125 cm² -/
  total_area_eq : 5 * side_length^2 = 125

/-- The perimeter of a cross-shaped figure -/
def perimeter (f : CrossFigure) : ℝ :=
  16 * f.side_length

/-- Theorem: The perimeter of the cross-shaped figure is 80 cm -/
theorem cross_figure_perimeter (f : CrossFigure) : perimeter f = 80 := by
  sorry

end cross_figure_perimeter_l3011_301127


namespace min_value_x_plus_y_l3011_301144

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (x - 1) * (y + 1) = 16) :
  x + y ≥ 8 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (x₀ - 1) * (y₀ + 1) = 16 ∧ x₀ + y₀ = 8 := by
  sorry

end min_value_x_plus_y_l3011_301144


namespace jerry_games_won_l3011_301141

theorem jerry_games_won (ken dave jerry : ℕ) 
  (h1 : ken = dave + 5)
  (h2 : dave = jerry + 3)
  (h3 : ken + dave + jerry = 32) : 
  jerry = 7 := by
  sorry

end jerry_games_won_l3011_301141


namespace function_properties_l3011_301174

-- Define the function f
def f (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

-- State the theorem
theorem function_properties (a b c : ℝ) :
  (∀ x < 0, ∀ y < x, f a b c x < f a b c y) →  -- f is decreasing on (-∞, 0)
  (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f a b c x < f a b c y) →  -- f is increasing on (0, 1)
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0) →  -- f has three real roots
  f a b c 1 = 0 →  -- 1 is a root of f
  b = 0 ∧ f a b c 2 > -5/2 ∧ 3/2 < a ∧ a < 2 * Real.sqrt 2 - 1 :=
by sorry

end function_properties_l3011_301174


namespace inequality_holds_iff_l3011_301198

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ -a * x * (x + y)) ↔ -2 ≤ a ∧ a ≤ 6 := by
  sorry

end inequality_holds_iff_l3011_301198


namespace smallest_dual_base_representation_l3011_301162

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 5 ∧ b > 5 ∧
    n = 1 * a + 5 ∧
    n = 5 * b + 1

theorem smallest_dual_base_representation :
  (∀ m : ℕ, m < 31 → ¬ is_valid_representation m) ∧
  is_valid_representation 31 :=
sorry

end smallest_dual_base_representation_l3011_301162


namespace additional_pecks_needed_to_fill_barrel_l3011_301114

-- Define the relationships
def peck_to_bushel : ℚ := 1/4
def bushel_to_barrel : ℚ := 1/9

-- Define the number of pecks already picked
def pecks_picked : ℕ := 1

-- Theorem statement
theorem additional_pecks_needed_to_fill_barrel : 
  ∀ (pecks_in_barrel : ℕ), 
    pecks_in_barrel = (1 / peck_to_bushel : ℚ) * (1 / bushel_to_barrel : ℚ) → 
    pecks_in_barrel - pecks_picked = 35 := by
  sorry

end additional_pecks_needed_to_fill_barrel_l3011_301114


namespace sequence_sum_theorem_l3011_301187

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_of_terms b n + b (n + 1)

theorem sequence_sum_theorem (a b c : ℕ → ℝ) (d : ℝ) :
  d > 0 ∧
  arithmetic_sequence a d ∧
  a 2 + a 5 = 12 ∧
  a 2 * a 5 = 27 ∧
  b 1 = 3 ∧
  (∀ n : ℕ, b (n + 1) = 2 * sum_of_terms b n + 3) ∧
  (∀ n : ℕ, c n = a n / b n) →
  ∀ n : ℕ, sum_of_terms c n = 1 - (n + 1 : ℝ) / 3^n := by
  sorry

end sequence_sum_theorem_l3011_301187


namespace reflection_sum_l3011_301172

-- Define the reflection line
structure ReflectionLine where
  m : ℝ
  b : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the reflection operation
def reflect (p : Point) (l : ReflectionLine) : Point :=
  sorry

-- Theorem statement
theorem reflection_sum (l : ReflectionLine) :
  reflect ⟨2, -2⟩ l = ⟨-4, 4⟩ → l.m + l.b = 3 := by
  sorry

end reflection_sum_l3011_301172


namespace factorization_of_M_l3011_301183

theorem factorization_of_M (a b c d : ℝ) :
  ((a - c)^2 + (b - d)^2) * (a^2 + b^2) - (a * d - b * c)^2 = (a * c + b * d - a^2 - b^2)^2 := by
  sorry

end factorization_of_M_l3011_301183


namespace equality_of_pairs_l3011_301103

theorem equality_of_pairs (a b x y : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_x : 0 < x) (pos_y : 0 < y)
  (sum_lt_two : a + b + x + y < 2)
  (eq_one : a + b^2 = x + y^2)
  (eq_two : a^2 + b = x^2 + y) :
  a = x ∧ b = y := by
sorry

end equality_of_pairs_l3011_301103


namespace smallest_cube_on_unit_cube_surface_l3011_301152

-- Define a cube type
structure Cube where
  edgeLength : ℝ

-- Define the unit cube K1
def K1 : Cube := ⟨1⟩

-- Define the property that a cube's vertices lie on the surface of K1
def verticesOnSurfaceOfK1 (c : Cube) : Prop := sorry

-- Theorem statement
theorem smallest_cube_on_unit_cube_surface :
  ∃ (minCube : Cube), 
    verticesOnSurfaceOfK1 minCube ∧ 
    minCube.edgeLength = 1 / Real.sqrt 2 ∧
    ∀ (c : Cube), verticesOnSurfaceOfK1 c → c.edgeLength ≥ minCube.edgeLength :=
sorry

end smallest_cube_on_unit_cube_surface_l3011_301152


namespace bus_stop_time_l3011_301161

/-- Calculates the stop time of a bus given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) :
  speed_without_stops = 54 →
  speed_with_stops = 45 →
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

#check bus_stop_time

end bus_stop_time_l3011_301161


namespace simplify_square_roots_l3011_301188

theorem simplify_square_roots : Real.sqrt 49 - Real.sqrt 256 = -9 := by
  sorry

end simplify_square_roots_l3011_301188


namespace profit_sharing_ratio_l3011_301140

/-- Represents the profit sharing ratio between two investors -/
structure ProfitRatio where
  praveen : ℕ
  hari : ℕ

/-- Calculates the profit sharing ratio based on investments and durations -/
def calculate_profit_ratio (praveen_investment : ℕ) (praveen_duration : ℕ) 
                           (hari_investment : ℕ) (hari_duration : ℕ) : ProfitRatio :=
  let praveen_contribution := praveen_investment * praveen_duration
  let hari_contribution := hari_investment * hari_duration
  let gcd := Nat.gcd praveen_contribution hari_contribution
  { praveen := praveen_contribution / gcd
  , hari := hari_contribution / gcd }

/-- Theorem stating the profit sharing ratio for the given problem -/
theorem profit_sharing_ratio : 
  calculate_profit_ratio 3220 12 8280 7 = ProfitRatio.mk 2 3 := by
  sorry

end profit_sharing_ratio_l3011_301140


namespace train_length_calculation_l3011_301130

/-- Calculates the length of a train given the speeds of a jogger and the train, 
    the initial distance between them, and the time it takes for the train to pass the jogger. -/
def train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  (train_speed - jogger_speed) * passing_time - initial_distance

/-- Theorem stating that given the specific conditions, the train length is 120 meters. -/
theorem train_length_calculation : 
  train_length (9 * (1000 / 3600)) (45 * (1000 / 3600)) 250 37 = 120 := by
  sorry

end train_length_calculation_l3011_301130


namespace sphere_in_parabolic_glass_l3011_301101

/-- The distance from the highest point of a sphere to the bottom of a parabolic wine glass --/
theorem sphere_in_parabolic_glass (x y : ℝ) (b : ℝ) : 
  (∀ y, 0 ≤ y → y < 15 → x^2 = 2*y) →  -- Parabola equation
  (x^2 + (y - b)^2 = 9) →               -- Sphere equation
  ((2 - 2*b)^2 - 4*(b^2 - 9) = 0) →     -- Tangency condition
  (b + 3 = 8) :=                        -- Distance from highest point to bottom
by sorry

end sphere_in_parabolic_glass_l3011_301101


namespace blanket_price_problem_l3011_301170

theorem blanket_price_problem (unknown_rate : ℕ) : 
  (3 * 100 + 1 * 150 + 2 * unknown_rate) / 6 = 150 → unknown_rate = 225 := by
  sorry

end blanket_price_problem_l3011_301170


namespace triangle_problem_l3011_301156

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (S₁ S₂ S₃ : ℝ) 
  (h₁ : S₁ - S₂ + S₃ = Real.sqrt 3 / 2)
  (h₂ : Real.sin B = 1 / 3)
  (h₃ : S₁ = Real.sqrt 3 / 4 * a^2)
  (h₄ : S₂ = Real.sqrt 3 / 4 * b^2)
  (h₅ : S₃ = Real.sqrt 3 / 4 * c^2)
  (h₆ : a > 0 ∧ b > 0 ∧ c > 0)
  (h₇ : 0 < A ∧ A < π)
  (h₈ : 0 < B ∧ B < π)
  (h₉ : 0 < C ∧ C < π)
  (h₁₀ : A + B + C = π) :
  (∃ (S : ℝ), S = Real.sqrt 2 / 8 ∧ S = 1/2 * a * c * Real.sin B) ∧
  (Real.sin A * Real.sin C = Real.sqrt 2 / 3 → b = 1 / 2) :=
by sorry

end triangle_problem_l3011_301156


namespace gmat_score_difference_l3011_301117

theorem gmat_score_difference (x y : ℝ) (h1 : x > y) (h2 : x / y = 4) :
  x - y = 3 * y := by
sorry

end gmat_score_difference_l3011_301117


namespace volunteer_schedule_lcm_l3011_301193

theorem volunteer_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end volunteer_schedule_lcm_l3011_301193


namespace investment_growth_l3011_301182

theorem investment_growth (initial_investment : ℝ) (first_year_loss_rate : ℝ) (second_year_gain_rate : ℝ) :
  initial_investment = 150 →
  first_year_loss_rate = 0.1 →
  second_year_gain_rate = 0.25 →
  let first_year_amount := initial_investment * (1 - first_year_loss_rate)
  let final_amount := first_year_amount * (1 + second_year_gain_rate)
  let overall_gain_rate := (final_amount - initial_investment) / initial_investment
  overall_gain_rate = 0.125 := by
  sorry

end investment_growth_l3011_301182


namespace line_symmetry_l3011_301179

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-axis -/
def y_axis : Line := { a := 1, b := 0, c := 0 }

/-- Check if two lines are symmetric with respect to a given line -/
def symmetric_wrt (l1 l2 axis : Line) : Prop :=
  -- Definition of symmetry with respect to a line
  sorry

/-- The original line x - y + 1 = 0 -/
def original_line : Line := { a := 1, b := -1, c := 1 }

/-- The proposed symmetric line x + y - 1 = 0 -/
def symmetric_line : Line := { a := 1, b := 1, c := -1 }

theorem line_symmetry : 
  symmetric_wrt original_line symmetric_line y_axis := by
  sorry

end line_symmetry_l3011_301179


namespace total_boxes_needed_l3011_301176

-- Define the amounts of wooden sticks and box capacities
def taehyung_total : ℚ := 21 / 11
def taehyung_per_box : ℚ := 7 / 11
def hoseok_total : ℚ := 8 / 17
def hoseok_per_box : ℚ := 2 / 17

-- Define the function to calculate the number of boxes needed
def boxes_needed (total : ℚ) (per_box : ℚ) : ℕ :=
  (total / per_box).ceil.toNat

-- Theorem statement
theorem total_boxes_needed :
  boxes_needed taehyung_total taehyung_per_box +
  boxes_needed hoseok_total hoseok_per_box = 7 := by
  sorry

end total_boxes_needed_l3011_301176


namespace statement_B_only_incorrect_l3011_301153

-- Define the structure for a statistical statement
structure StatisticalStatement where
  label : Char
  content : String
  isCorrect : Bool

-- Define the four statements
def statementA : StatisticalStatement := {
  label := 'A',
  content := "The absolute value of the correlation coefficient approaches 1 as the linear correlation between two random variables strengthens.",
  isCorrect := true
}

def statementB : StatisticalStatement := {
  label := 'B',
  content := "In a three-shot target shooting scenario, \"at least two hits\" and \"exactly one hit\" are complementary events.",
  isCorrect := false
}

def statementC : StatisticalStatement := {
  label := 'C',
  content := "The accuracy of a model fit increases as the band of residual points in a residual plot narrows.",
  isCorrect := true
}

def statementD : StatisticalStatement := {
  label := 'B',
  content := "The variance of a dataset remains unchanged when a constant is added to each data point.",
  isCorrect := true
}

-- Define the list of all statements
def allStatements : List StatisticalStatement := [statementA, statementB, statementC, statementD]

-- Theorem: Statement B is the only incorrect statement
theorem statement_B_only_incorrect :
  ∃! s : StatisticalStatement, s ∈ allStatements ∧ ¬s.isCorrect :=
sorry

end statement_B_only_incorrect_l3011_301153


namespace book_selling_price_l3011_301175

theorem book_selling_price (cost_price : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 250 →
  profit_percentage = 20 →
  selling_price = cost_price * (1 + profit_percentage / 100) →
  selling_price = 300 := by
sorry

end book_selling_price_l3011_301175


namespace cubic_extrema_l3011_301157

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x^2 + 4 * x - 7

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 4 * x + 4

/-- The discriminant of f' -/
def Δ (a : ℝ) : ℝ := (-4)^2 - 4 * 3 * a * 4

theorem cubic_extrema (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ f a max ∧ f a min ≤ f a x) ↔ 
  (a < 1/3 ∧ a ≠ 0) :=
sorry

end cubic_extrema_l3011_301157


namespace interior_perimeter_is_20_l3011_301155

/-- Represents a rectangular picture frame -/
structure Frame where
  outer_width : ℝ
  outer_height : ℝ
  border_width : ℝ
  frame_area : ℝ

/-- Calculates the sum of the lengths of the four interior edges of a frame -/
def interior_perimeter (f : Frame) : ℝ :=
  2 * ((f.outer_width - 2 * f.border_width) + (f.outer_height - 2 * f.border_width))

/-- Theorem stating that for a frame with given dimensions, the interior perimeter is 20 inches -/
theorem interior_perimeter_is_20 (f : Frame) 
  (h_outer_width : f.outer_width = 8)
  (h_outer_height : f.outer_height = 10)
  (h_border_width : f.border_width = 2)
  (h_frame_area : f.frame_area = 52) :
  interior_perimeter f = 20 := by
  sorry

#check interior_perimeter_is_20

end interior_perimeter_is_20_l3011_301155


namespace linear_system_elimination_l3011_301125

theorem linear_system_elimination (x y : ℝ) : 
  (6 * x - 5 * y = 3) → 
  (3 * x + y = -15) → 
  (5 * (3 * x + y) + (6 * x - 5 * y) = 21 * x) ∧ 
  (5 * (-15) + 3 = -72) := by
  sorry

end linear_system_elimination_l3011_301125


namespace x_difference_l3011_301186

theorem x_difference (x₁ x₂ : ℝ) : 
  ((x₁ + 3)^2 / (3*x₁ + 65) = 2) →
  ((x₂ + 3)^2 / (3*x₂ + 65) = 2) →
  x₁ ≠ x₂ →
  |x₁ - x₂| = 22 := by
sorry

end x_difference_l3011_301186


namespace ant_count_in_field_l3011_301158

/-- Calculates the number of ants in a rectangular field given its dimensions in feet and ant density per square inch -/
def number_of_ants (width_feet : ℝ) (length_feet : ℝ) (ants_per_sq_inch : ℝ) : ℝ :=
  width_feet * length_feet * 144 * ants_per_sq_inch

/-- Theorem stating that a 500 by 600 feet field with 4 ants per square inch contains 172,800,000 ants -/
theorem ant_count_in_field : number_of_ants 500 600 4 = 172800000 := by
  sorry

end ant_count_in_field_l3011_301158


namespace largest_number_l3011_301168

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 0.999) 
  (hb : b = 0.9099) 
  (hc : c = 0.9991) 
  (hd : d = 0.991) 
  (he : e = 0.9091) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e := by
  sorry

end largest_number_l3011_301168


namespace fraction_sum_equality_l3011_301169

theorem fraction_sum_equality (n : ℕ) (hn : n > 2009) :
  ∃ (a b c d : ℕ), a ≤ 2009 ∧ b ≤ 2009 ∧ c ≤ 2009 ∧ d ≤ 2009 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (1 : ℚ) / (n + 1 - a) + (1 : ℚ) / (n + 1 - b) =
  (1 : ℚ) / (n + 1 - c) + (1 : ℚ) / (n + 1 - d) :=
by sorry

end fraction_sum_equality_l3011_301169


namespace average_volume_of_three_cubes_l3011_301199

theorem average_volume_of_three_cubes (a b c : ℕ) (ha : a = 4) (hb : b = 5) (hc : c = 6) :
  (a^3 + b^3 + c^3) / 3 = 135 := by
  sorry

end average_volume_of_three_cubes_l3011_301199


namespace min_additional_marbles_l3011_301120

/-- The number of friends Tom has -/
def num_friends : ℕ := 12

/-- The initial number of marbles Tom has -/
def initial_marbles : ℕ := 40

/-- The sum of consecutive integers from 1 to n -/
def sum_consecutive (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the minimum number of additional marbles Tom needs -/
theorem min_additional_marbles : 
  sum_consecutive num_friends - initial_marbles = 38 := by sorry

end min_additional_marbles_l3011_301120


namespace wall_height_calculation_l3011_301123

/-- Calculates the height of a wall given brick dimensions and number of bricks --/
theorem wall_height_calculation (brick_length brick_width brick_height : ℝ)
                                (wall_length wall_width : ℝ)
                                (num_bricks : ℝ) :
  brick_length = 25 →
  brick_width = 11 →
  brick_height = 6 →
  wall_length = 200 →
  wall_width = 2 →
  num_bricks = 72.72727272727273 →
  ∃ (wall_height : ℝ), abs (wall_height - 436.3636363636364) < 0.0001 :=
by
  sorry

#check wall_height_calculation

end wall_height_calculation_l3011_301123


namespace point_on_curve_with_perpendicular_tangent_l3011_301180

/-- The function f(x) = x^4 - x -/
def f (x : ℝ) : ℝ := x^4 - x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4*x^3 - 1

theorem point_on_curve_with_perpendicular_tangent :
  ∀ x y : ℝ,
  f x = y →                           -- Point P(x, y) is on the curve f(x) = x^4 - x
  (f' x) * (-3) = 1 →                 -- Tangent line is perpendicular to x + 3y = 0
  x = 1 ∧ y = 0 := by                 -- Then P has coordinates (1, 0)
sorry

end point_on_curve_with_perpendicular_tangent_l3011_301180


namespace student_arrangement_l3011_301185

theorem student_arrangement (n : ℕ) (h : n = 5) :
  let total_arrangements := n.factorial
  let a_left_arrangements := 2 * (n - 1).factorial
  let a_left_b_right_arrangements := (n - 2).factorial
  let valid_arrangements := total_arrangements - a_left_arrangements + a_left_b_right_arrangements
  valid_arrangements = 78 :=
sorry

end student_arrangement_l3011_301185


namespace equation_solution_l3011_301132

theorem equation_solution : ∃ x : ℝ, 3 * x - 6 = |(-20 + 5)| ∧ x = 7 := by sorry

end equation_solution_l3011_301132


namespace incorrect_proposition_statement_l3011_301143

theorem incorrect_proposition_statement : ∃ (p q : Prop), 
  (¬(p ∧ q)) ∧ (p ∨ q) := by
  sorry

end incorrect_proposition_statement_l3011_301143


namespace periodic_function_value_l3011_301151

/-- Given a function f(x) = a*sin(π*x + θ) + b*cos(π*x + θ) + 3,
    where a, b, θ are non-zero real numbers, and f(2016) = -1,
    prove that f(2017) = 7. -/
theorem periodic_function_value (a b θ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hθ : θ ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + θ) + b * Real.cos (π * x + θ) + 3
  f 2016 = -1 → f 2017 = 7 := by
  sorry

end periodic_function_value_l3011_301151


namespace students_without_pens_l3011_301116

theorem students_without_pens (total students_with_blue students_with_red students_with_both : ℕ) 
  (h1 : total = 40)
  (h2 : students_with_blue = 18)
  (h3 : students_with_red = 26)
  (h4 : students_with_both = 10) :
  total - (students_with_blue + students_with_red - students_with_both) = 6 :=
by sorry

end students_without_pens_l3011_301116


namespace distance_propositions_l3011_301196

-- Define the distance measure
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₂ - x₁| + |y₂ - y₁|

-- Define propositions
def proposition1 (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (x ∈ Set.Icc x₁ x₂ ∧ y ∈ Set.Icc y₁ y₂) →
  distance x₁ y₁ x y + distance x y x₂ y₂ = distance x₁ y₁ x₂ y₂

def proposition2 (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (x - x₁) * (x₂ - x) + (y - y₁) * (y₂ - y) = 0 →
  (distance x₁ y₁ x y)^2 + (distance x y x₂ y₂)^2 = (distance x₁ y₁ x₂ y₂)^2

def proposition3 (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  distance x₁ y₁ x y + distance x y x₂ y₂ > distance x₁ y₁ x₂ y₂

-- Theorem statement
theorem distance_propositions :
  (∀ x₁ y₁ x₂ y₂ x y, proposition1 x₁ y₁ x₂ y₂ x y) ∧
  (∃ x₁ y₁ x₂ y₂ x y, ¬proposition2 x₁ y₁ x₂ y₂ x y) ∧
  (∃ x₁ y₁ x₂ y₂ x y, ¬proposition3 x₁ y₁ x₂ y₂ x y) :=
sorry

end distance_propositions_l3011_301196


namespace cat_puppy_weight_difference_l3011_301129

/-- The weight difference between cats and puppies -/
theorem cat_puppy_weight_difference :
  let puppy_weights : List ℝ := [6.5, 7.2, 8, 9.5]
  let cat_weight : ℝ := 2.8
  let num_cats : ℕ := 16
  (num_cats : ℝ) * cat_weight - puppy_weights.sum = 13.6 := by
  sorry

end cat_puppy_weight_difference_l3011_301129


namespace tractor_circuits_l3011_301115

theorem tractor_circuits (r₁ r₂ : ℝ) (n₁ : ℕ) (h₁ : r₁ = 30) (h₂ : r₂ = 10) (h₃ : n₁ = 20) :
  ∃ n₂ : ℕ, n₂ = 60 ∧ r₁ * n₁ = r₂ * n₂ := by
  sorry

end tractor_circuits_l3011_301115


namespace two_students_all_pets_l3011_301121

/-- Represents the number of students in each section of the Venn diagram --/
structure PetOwnership where
  total : ℕ
  dogs : ℕ
  cats : ℕ
  other : ℕ
  no_pets : ℕ
  dogs_only : ℕ
  cats_only : ℕ
  other_only : ℕ
  dogs_and_cats : ℕ
  cats_and_other : ℕ
  dogs_and_other : ℕ
  all_three : ℕ

/-- Theorem stating that 2 students have all three types of pets --/
theorem two_students_all_pets (po : PetOwnership) : po.all_three = 2 :=
  by
  have h1 : po.total = 40 := sorry
  have h2 : po.dogs = po.total / 2 := sorry
  have h3 : po.cats = po.total * 5 / 16 := sorry
  have h4 : po.other = 8 := sorry
  have h5 : po.no_pets = 7 := sorry
  have h6 : po.dogs_only = 12 := sorry
  have h7 : po.cats_only = 3 := sorry
  have h8 : po.other_only = 2 := sorry

  have total_pet_owners : po.dogs_only + po.cats_only + po.other_only + 
    po.dogs_and_cats + po.cats_and_other + po.dogs_and_other + po.all_three = 
    po.total - po.no_pets := sorry

  have dog_owners : po.dogs_only + po.dogs_and_cats + po.dogs_and_other + po.all_three = 
    po.dogs := sorry

  have cat_owners : po.cats_only + po.dogs_and_cats + po.cats_and_other + po.all_three = 
    po.cats := sorry

  have other_pet_owners : po.other_only + po.cats_and_other + po.dogs_and_other + 
    po.all_three = po.other := sorry

  sorry


end two_students_all_pets_l3011_301121


namespace special_sequence_properties_l3011_301133

/-- A sequence and its partial sums satisfying certain conditions -/
structure SpecialSequence where
  q : ℝ
  a : ℕ → ℝ
  S : ℕ → ℝ
  h1 : q * (q - 1) ≠ 0
  h2 : ∀ n : ℕ, (1 - q) * S n + q^n = 1
  h3 : S 3 - S 9 = S 9 - S 6

/-- The main theorem about the special sequence -/
theorem special_sequence_properties (seq : SpecialSequence) :
  (∀ n : ℕ, seq.a n = seq.q^(n - 1)) ∧
  (seq.a 2 - seq.a 8 = seq.a 8 - seq.a 5) := by
  sorry

end special_sequence_properties_l3011_301133


namespace negation_of_existential_proposition_l3011_301178

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 ≥ x) ↔ (∀ x : ℝ, x^2 < x) :=
by sorry

end negation_of_existential_proposition_l3011_301178


namespace larger_integer_proof_l3011_301148

theorem larger_integer_proof (a b : ℕ+) : 
  (a : ℕ) + 3 = (b : ℕ) → a * b = 88 → b = 11 := by
  sorry

end larger_integer_proof_l3011_301148


namespace frying_time_correct_l3011_301195

/-- Calculates the minimum time required to fry n pancakes -/
def min_frying_time (n : ℕ) : ℕ :=
  if n ≤ 2 then
    4
  else if n % 2 = 0 then
    2 * n
  else
    2 * (n - 1) + 2

theorem frying_time_correct :
  (min_frying_time 3 = 6) ∧ (min_frying_time 2016 = 4032) := by
  sorry

#eval min_frying_time 3
#eval min_frying_time 2016

end frying_time_correct_l3011_301195


namespace gcd_factorial_8_9_l3011_301119

theorem gcd_factorial_8_9 : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := by
  sorry

end gcd_factorial_8_9_l3011_301119


namespace right_triangle_area_l3011_301184

theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 12 →  -- hypotenuse is 12 inches
  θ = 30 * π / 180 →  -- one angle is 30 degrees (converted to radians)
  area = 18 * Real.sqrt 3 →  -- area is 18√3 square inches
  area = (h * h * Real.sin θ * Real.cos θ) / 2 :=
by sorry

#check right_triangle_area

end right_triangle_area_l3011_301184


namespace ball_color_distribution_l3011_301146

theorem ball_color_distribution (x y z : ℕ) : 
  x + y + z = 20 →
  x > 0 ∧ y > 0 ∧ z > 0 →
  (z : ℚ) / 20 - (2 * x : ℚ) / (2 * x + y + z) = 1 / 5 →
  x = 5 ∧ y = 3 ∧ z = 12 := by
sorry

end ball_color_distribution_l3011_301146


namespace complex_coordinates_l3011_301181

theorem complex_coordinates (z : ℂ) : z = (2 + Complex.I) / Complex.I → 
  Complex.re z = 1 ∧ Complex.im z = -2 := by
  sorry

end complex_coordinates_l3011_301181


namespace cuboid_surface_area_4_8_6_l3011_301106

/-- The surface area of a cuboid with given dimensions -/
def cuboid_surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem stating that the surface area of a cuboid with dimensions 4x8x6 is 208 -/
theorem cuboid_surface_area_4_8_6 :
  cuboid_surface_area 4 8 6 = 208 := by
  sorry

#eval cuboid_surface_area 4 8 6

end cuboid_surface_area_4_8_6_l3011_301106


namespace square_properties_l3011_301111

/-- Given a square and a rectangle with the same perimeter, where the rectangle
    has sides of 10 cm and 7 cm, this theorem proves the side length and area of the square. -/
theorem square_properties (square_perimeter rectangle_perimeter : ℝ)
                          (rectangle_side1 rectangle_side2 : ℝ)
                          (h1 : square_perimeter = rectangle_perimeter)
                          (h2 : rectangle_side1 = 10)
                          (h3 : rectangle_side2 = 7)
                          (h4 : rectangle_perimeter = 2 * (rectangle_side1 + rectangle_side2)) :
  ∃ (square_side : ℝ),
    square_side = 8.5 ∧
    square_perimeter = 4 * square_side ∧
    square_side ^ 2 = 72.25 := by
  sorry

end square_properties_l3011_301111


namespace circle_radii_relation_l3011_301173

/-- Given three circles with centers A, B, C, touching each other and a line l,
    with radii a, b, and c respectively, prove that 1/√c = 1/√a + 1/√b. -/
theorem circle_radii_relation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / Real.sqrt c = 1 / Real.sqrt a + 1 / Real.sqrt b := by
sorry

end circle_radii_relation_l3011_301173


namespace grid_solution_l3011_301126

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- The sum of adjacent numbers is less than 12 --/
def valid_sum (g : Grid) : Prop :=
  ∀ p1 p2 : Fin 3 × Fin 3, adjacent p1 p2 → g p1.1 p1.2 + g p2.1 p2.2 < 12

/-- The grid contains all numbers from 1 to 9 --/
def contains_all_numbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n.val + 1

/-- The given positions in the grid --/
def given_positions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 0 2 = 9 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7

/-- The theorem to prove --/
theorem grid_solution (g : Grid) 
  (h1 : valid_sum g) 
  (h2 : contains_all_numbers g) 
  (h3 : given_positions g) : 
  g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end grid_solution_l3011_301126


namespace morgan_paid_twenty_l3011_301160

/-- Represents the cost of Morgan's lunch items and the change received --/
structure LunchTransaction where
  hamburger_cost : ℕ
  onion_rings_cost : ℕ
  smoothie_cost : ℕ
  change_received : ℕ

/-- Calculates the total amount paid by Morgan --/
def amount_paid (t : LunchTransaction) : ℕ :=
  t.hamburger_cost + t.onion_rings_cost + t.smoothie_cost + t.change_received

/-- Theorem stating that Morgan paid $20 --/
theorem morgan_paid_twenty :
  ∀ (t : LunchTransaction),
    t.hamburger_cost = 4 →
    t.onion_rings_cost = 2 →
    t.smoothie_cost = 3 →
    t.change_received = 11 →
    amount_paid t = 20 :=
by sorry

end morgan_paid_twenty_l3011_301160


namespace difference_of_reciprocals_l3011_301108

theorem difference_of_reciprocals (p q : ℚ) 
  (hp : 3 / p = 6) 
  (hq : 3 / q = 15) : 
  p - q = 3 / 10 := by sorry

end difference_of_reciprocals_l3011_301108


namespace rectangular_plot_poles_l3011_301107

/-- The number of poles needed to enclose a rectangular plot -/
def poles_needed (length width pole_distance : ℕ) : ℕ :=
  ((2 * (length + width) + pole_distance - 1) / pole_distance : ℕ)

/-- Theorem: A 135m by 80m plot with poles 7m apart needs 62 poles -/
theorem rectangular_plot_poles :
  poles_needed 135 80 7 = 62 := by
  sorry

end rectangular_plot_poles_l3011_301107


namespace gunny_bag_capacity_is_13_tons_l3011_301100

/-- Represents the weight of a packet in pounds -/
def packet_weight : ℚ := 16 + 4 / 16

/-- Represents the number of packets -/
def num_packets : ℕ := 1840

/-- Represents the number of pounds in a ton -/
def pounds_per_ton : ℕ := 2300

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℚ := (packet_weight * num_packets) / pounds_per_ton

theorem gunny_bag_capacity_is_13_tons : gunny_bag_capacity = 13 := by
  sorry

end gunny_bag_capacity_is_13_tons_l3011_301100


namespace garden_length_l3011_301167

/-- Given a rectangular garden with area 120 m², if reducing its length by 2m results in a square,
    then the original length of the garden is 12 meters. -/
theorem garden_length (length width : ℝ) : 
  length * width = 120 →
  (length - 2) * (length - 2) = width * (length - 2) →
  length = 12 := by
sorry

end garden_length_l3011_301167
