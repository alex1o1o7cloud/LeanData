import Mathlib

namespace NUMINAMATH_CALUDE_probability_differ_by_three_l461_46149

/-- A type representing the possible outcomes of rolling a standard 6-sided die -/
inductive DieRoll : Type
  | one : DieRoll
  | two : DieRoll
  | three : DieRoll
  | four : DieRoll
  | five : DieRoll
  | six : DieRoll

/-- The total number of possible outcomes when rolling a die twice -/
def totalOutcomes : ℕ := 36

/-- A function that returns true if two die rolls differ by 3 -/
def differByThree (roll1 roll2 : DieRoll) : Prop :=
  match roll1, roll2 with
  | DieRoll.one, DieRoll.four => True
  | DieRoll.two, DieRoll.five => True
  | DieRoll.three, DieRoll.six => True
  | DieRoll.four, DieRoll.one => True
  | DieRoll.five, DieRoll.two => True
  | DieRoll.six, DieRoll.three => True
  | _, _ => False

/-- The number of favorable outcomes (pairs of rolls that differ by 3) -/
def favorableOutcomes : ℕ := 6

/-- The main theorem: the probability of rolling two numbers that differ by 3 is 1/6 -/
theorem probability_differ_by_three :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_probability_differ_by_three_l461_46149


namespace NUMINAMATH_CALUDE_birthday_candles_ratio_l461_46179

theorem birthday_candles_ratio (ambika_candles : ℕ) (total_candles : ℕ) : 
  ambika_candles = 4 → total_candles = 14 → 
  ∃ (aniyah_ratio : ℚ), aniyah_ratio = 2.5 ∧ 
  ambika_candles * (1 + aniyah_ratio) = total_candles :=
sorry

end NUMINAMATH_CALUDE_birthday_candles_ratio_l461_46179


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l461_46170

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 58) 
  (h2 : throwers = 37) 
  (h3 : throwers ≤ total_players) 
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures one-third of non-throwers can be left-handed
  : (throwers + ((total_players - throwers) - (total_players - throwers) / 3)) = 51 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l461_46170


namespace NUMINAMATH_CALUDE_x_over_3_is_directly_proportional_l461_46139

/-- A function f : ℝ → ℝ is directly proportional if there exists a non-zero constant k such that f(x) = k * x for all x -/
def IsDirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function f(x) = x/3 is directly proportional -/
theorem x_over_3_is_directly_proportional :
  IsDirectlyProportional (fun x => x / 3) := by
  sorry

end NUMINAMATH_CALUDE_x_over_3_is_directly_proportional_l461_46139


namespace NUMINAMATH_CALUDE_collinearity_condition_l461_46117

/-- Three points in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Collinearity condition for three points in a 2D plane -/
def collinear (A B C : Point2D) : Prop :=
  A.x * B.y + B.x * C.y + C.x * A.y = A.y * B.x + B.y * C.x + C.y * A.x

/-- Theorem: Three points are collinear iff they satisfy the collinearity condition -/
theorem collinearity_condition (A B C : Point2D) :
  collinear A B C ↔ A.x * B.y + B.x * C.y + C.x * A.y = A.y * B.x + B.y * C.x + C.y * A.x :=
sorry

end NUMINAMATH_CALUDE_collinearity_condition_l461_46117


namespace NUMINAMATH_CALUDE_angle_WYZ_measure_l461_46142

-- Define the angles
def angle_XYZ : ℝ := 40
def angle_XYW : ℝ := 15

-- Define the theorem
theorem angle_WYZ_measure :
  let angle_WYZ := angle_XYZ - angle_XYW
  angle_WYZ = 25 := by sorry

end NUMINAMATH_CALUDE_angle_WYZ_measure_l461_46142


namespace NUMINAMATH_CALUDE_power_of_product_exponent_l461_46173

theorem power_of_product_exponent (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_exponent_l461_46173


namespace NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l461_46183

theorem square_difference_divided (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (125^2 - 105^2) / 20 = 230 := by
  have h : 125 > 105 := by sorry
  have key := square_difference_divided 125 105 h
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l461_46183


namespace NUMINAMATH_CALUDE_unique_n_satisfying_equation_l461_46116

theorem unique_n_satisfying_equation : 
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 9⌋ - ⌊(n : ℚ) / 3⌋^2 = 5 ∧ n = 14 :=
by sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_equation_l461_46116


namespace NUMINAMATH_CALUDE_weighted_mean_car_sales_approx_l461_46190

/-- Represents the car sales data for a week -/
structure CarSalesWeek where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  tuesday_discount : ℚ
  wednesday_commission : ℚ
  friday_discount : ℚ
  saturday_commission : ℚ

/-- Calculates the weighted mean of car sales for a week -/
def weightedMeanCarSales (sales : CarSalesWeek) : ℚ :=
  let monday_weighted := sales.monday
  let tuesday_weighted := sales.tuesday * (1 - sales.tuesday_discount)
  let wednesday_weighted := sales.wednesday * (1 + sales.wednesday_commission)
  let thursday_weighted := sales.thursday
  let friday_weighted := sales.friday * (1 - sales.friday_discount)
  let saturday_weighted := sales.saturday * (1 + sales.saturday_commission)
  let total_weighted := monday_weighted + tuesday_weighted + wednesday_weighted + 
                        thursday_weighted + friday_weighted + saturday_weighted
  total_weighted / 6

/-- Theorem: The weighted mean of car sales for the given week is approximately 5.48 -/
theorem weighted_mean_car_sales_approx (sales : CarSalesWeek) 
  (h1 : sales.monday = 8)
  (h2 : sales.tuesday = 3)
  (h3 : sales.wednesday = 10)
  (h4 : sales.thursday = 4)
  (h5 : sales.friday = 4)
  (h6 : sales.saturday = 4)
  (h7 : sales.tuesday_discount = 1/10)
  (h8 : sales.wednesday_commission = 1/20)
  (h9 : sales.friday_discount = 3/20)
  (h10 : sales.saturday_commission = 7/100) :
  ∃ ε > 0, |weightedMeanCarSales sales - 548/100| < ε :=
sorry


end NUMINAMATH_CALUDE_weighted_mean_car_sales_approx_l461_46190


namespace NUMINAMATH_CALUDE_school_trip_theorem_l461_46114

/-- The number of school buses -/
def num_buses : ℕ := 95

/-- The number of seats in each school bus -/
def seats_per_bus : ℕ := 118

/-- The number of students in the school -/
def num_students : ℕ := num_buses * seats_per_bus

theorem school_trip_theorem : num_students = 11210 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_theorem_l461_46114


namespace NUMINAMATH_CALUDE_one_absent_out_of_three_l461_46150

def probability_absent : ℚ := 1 / 40

def probability_present : ℚ := 1 - probability_absent

def probability_one_absent_two_present : ℚ :=
  3 * probability_absent * probability_present * probability_present

theorem one_absent_out_of_three (ε : ℚ) (h : ε > 0) :
  |probability_one_absent_two_present - 4563 / 64000| < ε :=
sorry

end NUMINAMATH_CALUDE_one_absent_out_of_three_l461_46150


namespace NUMINAMATH_CALUDE_f_inequality_range_l461_46193

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

theorem f_inequality_range :
  ∀ x : ℝ, (f x + f (x - 1/2) > 1) ↔ (x > -1/4) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_range_l461_46193


namespace NUMINAMATH_CALUDE_sum_zero_ratio_theorem_l461_46160

theorem sum_zero_ratio_theorem (x y z w : ℝ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w) 
  (h_sum_zero : x + y + z + w = 0) : 
  (x*y + y*z + z*x + w*x + w*y + w*z) / (x^2 + y^2 + z^2 + w^2) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_zero_ratio_theorem_l461_46160


namespace NUMINAMATH_CALUDE_positive_reals_inequality_arithmetic_geometric_mean_inequality_l461_46158

theorem positive_reals_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequality_arithmetic_geometric_mean_inequality_l461_46158


namespace NUMINAMATH_CALUDE_license_plate_difference_l461_46178

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of possible license plates for Georgia (LLDLLL format). -/
def georgia_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible license plates for Nebraska (LLDDDDD format). -/
def nebraska_plates : ℕ := num_letters^2 * num_digits^5

/-- The difference between the number of possible license plates for Nebraska and Georgia. -/
theorem license_plate_difference : nebraska_plates - georgia_plates = 21902400 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l461_46178


namespace NUMINAMATH_CALUDE_triangle_area_l461_46194

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![1, 5]

theorem triangle_area : 
  (1/2 : ℝ) * |Matrix.det !![a 0, a 1; b 0, b 1]| = (13/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_triangle_area_l461_46194


namespace NUMINAMATH_CALUDE_total_flowers_l461_46157

def roses : ℕ := 5
def lilies : ℕ := 2

theorem total_flowers : roses + lilies = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l461_46157


namespace NUMINAMATH_CALUDE_clusters_per_spoonful_l461_46154

/-- Represents the number of clusters of oats in a box of cereal -/
def clusters_per_box : ℕ := 500

/-- Represents the number of bowlfuls in a box of cereal -/
def bowlfuls_per_box : ℕ := 5

/-- Represents the number of spoonfuls in a bowl of cereal -/
def spoonfuls_per_bowl : ℕ := 25

/-- Theorem stating that the number of clusters of oats in each spoonful is 4 -/
theorem clusters_per_spoonful :
  clusters_per_box / (bowlfuls_per_box * spoonfuls_per_bowl) = 4 := by
  sorry

end NUMINAMATH_CALUDE_clusters_per_spoonful_l461_46154


namespace NUMINAMATH_CALUDE_max_digits_after_subtraction_l461_46198

theorem max_digits_after_subtraction :
  ∀ (a b c : ℕ),
  10000 ≤ a ∧ a ≤ 99999 →
  1000 ≤ b ∧ b ≤ 9999 →
  0 ≤ c ∧ c ≤ 9 →
  (Nat.digits 10 (a * b - c)).length ≤ 9 ∧
  ∃ (x y z : ℕ),
    10000 ≤ x ∧ x ≤ 99999 ∧
    1000 ≤ y ∧ y ≤ 9999 ∧
    0 ≤ z ∧ z ≤ 9 ∧
    (Nat.digits 10 (x * y - z)).length = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_digits_after_subtraction_l461_46198


namespace NUMINAMATH_CALUDE_quadratic_even_iff_b_zero_l461_46127

/-- A quadratic function f(x) = ax² + bx + c is even if and only if b = 0 -/
theorem quadratic_even_iff_b_zero (a b c : ℝ) :
  (∀ x, (a * x^2 + b * x + c) = (a * (-x)^2 + b * (-x) + c)) ↔ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_even_iff_b_zero_l461_46127


namespace NUMINAMATH_CALUDE_gain_percent_example_l461_46189

/-- Calculates the gain percent given the cost price and selling price -/
def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that the gain percent is 50% when an article is bought for $10 and sold for $15 -/
theorem gain_percent_example : gain_percent 10 15 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_example_l461_46189


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l461_46145

theorem sum_of_a_and_b (a b : ℝ) (h1 : a + 4 * b = 33) (h2 : 6 * a + 3 * b = 51) : 
  a + b = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l461_46145


namespace NUMINAMATH_CALUDE_radio_cost_price_l461_46134

/-- Proves that the cost price of a radio is 2400 given the selling price and loss percentage --/
theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 2100 → loss_percentage = 12.5 → 
  ∃ (cost_price : ℝ), cost_price = 2400 ∧ selling_price = cost_price * (1 - loss_percentage / 100) :=
by
  sorry

#check radio_cost_price

end NUMINAMATH_CALUDE_radio_cost_price_l461_46134


namespace NUMINAMATH_CALUDE_corn_preference_percentage_l461_46175

theorem corn_preference_percentage (peas carrots corn : ℕ) : 
  peas = 6 → carrots = 9 → corn = 5 → 
  (corn : ℚ) / (peas + carrots + corn : ℚ) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_corn_preference_percentage_l461_46175


namespace NUMINAMATH_CALUDE_cube_difference_fifty_l461_46176

/-- The sum of cubes of the first n positive integers -/
def sumOfPositiveCubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

/-- The sum of cubes of the first n negative integers -/
def sumOfNegativeCubes (n : ℕ) : ℤ := -(sumOfPositiveCubes n)

/-- The difference between the sum of cubes of the first n positive integers
    and the sum of cubes of the first n negative integers -/
def cubeDifference (n : ℕ) : ℤ := (sumOfPositiveCubes n : ℤ) - sumOfNegativeCubes n

theorem cube_difference_fifty : cubeDifference 50 = 3251250 := by sorry

end NUMINAMATH_CALUDE_cube_difference_fifty_l461_46176


namespace NUMINAMATH_CALUDE_inequality_always_holds_l461_46186

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : a * |c| ≥ b * |c| := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l461_46186


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l461_46115

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l461_46115


namespace NUMINAMATH_CALUDE_first_question_percentage_l461_46155

/-- The percentage of students who answered the first question correctly -/
def first_question_percent : ℝ := 80

/-- The percentage of students who answered the second question correctly -/
def second_question_percent : ℝ := 55

/-- The percentage of students who answered neither question correctly -/
def neither_question_percent : ℝ := 20

/-- The percentage of students who answered both questions correctly -/
def both_questions_percent : ℝ := 55

theorem first_question_percentage :
  first_question_percent = 100 - neither_question_percent - second_question_percent + both_questions_percent :=
by sorry

end NUMINAMATH_CALUDE_first_question_percentage_l461_46155


namespace NUMINAMATH_CALUDE_triangle_side_length_l461_46152

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  (0 < a ∧ 0 < b ∧ 0 < c) →  -- Positive side lengths
  (0 < A ∧ A < π) →  -- Valid angle measures
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →  -- Sum of angles in a triangle
  (c * Real.cos B = 12) →  -- Given condition
  (b * Real.sin C = 5) →  -- Given condition
  (a / Real.sin A = b / Real.sin B) →  -- Sine rule
  (b / Real.sin B = c / Real.sin C) →  -- Sine rule
  c = 13 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l461_46152


namespace NUMINAMATH_CALUDE_average_cost_is_two_l461_46166

/-- Calculates the average cost per fruit given the costs and quantities of apples, bananas, and oranges. -/
def average_cost_per_fruit (apple_cost banana_cost orange_cost : ℚ) 
                           (apple_qty banana_qty orange_qty : ℕ) : ℚ :=
  let total_cost := apple_cost * apple_qty + banana_cost * banana_qty + orange_cost * orange_qty
  let total_qty := apple_qty + banana_qty + orange_qty
  total_cost / total_qty

/-- Proves that the average cost per fruit is $2 given the specific costs and quantities. -/
theorem average_cost_is_two :
  average_cost_per_fruit 2 1 3 12 4 4 = 2 := by
  sorry

#eval average_cost_per_fruit 2 1 3 12 4 4

end NUMINAMATH_CALUDE_average_cost_is_two_l461_46166


namespace NUMINAMATH_CALUDE_lily_book_count_l461_46119

/-- The number of books Lily read last month -/
def last_month_books : ℕ := 4

/-- The number of books Lily plans to read this month -/
def this_month_books : ℕ := 2 * last_month_books

/-- The total number of books Lily will read over two months -/
def total_books : ℕ := last_month_books + this_month_books

theorem lily_book_count : total_books = 12 := by
  sorry

end NUMINAMATH_CALUDE_lily_book_count_l461_46119


namespace NUMINAMATH_CALUDE_rationalize_denominator_l461_46111

theorem rationalize_denominator : 
  1 / (Real.sqrt 3 - 2) = -(Real.sqrt 3) - 2 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l461_46111


namespace NUMINAMATH_CALUDE_expression_evaluation_l461_46106

theorem expression_evaluation : (-7)^3 / 7^2 - 4^4 + 5^2 = -238 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l461_46106


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l461_46135

theorem min_value_fraction_sum (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : b + c ≥ a + d) : 
  b / (c + d) + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l461_46135


namespace NUMINAMATH_CALUDE_probability_both_colors_drawn_l461_46129

def total_balls : ℕ := 16
def black_balls : ℕ := 10
def white_balls : ℕ := 6
def drawn_balls : ℕ := 3

theorem probability_both_colors_drawn : 
  (1 : ℚ) - (Nat.choose black_balls drawn_balls + Nat.choose white_balls drawn_balls : ℚ) / 
  (Nat.choose total_balls drawn_balls : ℚ) = 3/4 :=
sorry

end NUMINAMATH_CALUDE_probability_both_colors_drawn_l461_46129


namespace NUMINAMATH_CALUDE_certain_number_proof_l461_46159

theorem certain_number_proof (a b x : ℝ) (h1 : x * a = 3 * b) (h2 : a * b ≠ 0) (h3 : a / 3 = b / 2) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l461_46159


namespace NUMINAMATH_CALUDE_sack_lunch_cost_l461_46133

/-- The cost of each sack lunch for a field trip -/
theorem sack_lunch_cost (num_children : ℕ) (num_chaperones : ℕ) (num_teachers : ℕ) (num_additional : ℕ) (total_cost : ℚ) : 
  num_children = 35 →
  num_chaperones = 5 →
  num_teachers = 1 →
  num_additional = 3 →
  total_cost = 308 →
  total_cost / (num_children + num_chaperones + num_teachers + num_additional) = 7 := by
sorry

end NUMINAMATH_CALUDE_sack_lunch_cost_l461_46133


namespace NUMINAMATH_CALUDE_probability_white_after_red_20_balls_l461_46177

/-- The probability of drawing a white ball after a red ball has been drawn -/
def probability_white_after_red (total : ℕ) (red : ℕ) (white : ℕ) : ℚ :=
  if total = red + white ∧ red > 0 then
    white / (total - 1 : ℚ)
  else
    0

theorem probability_white_after_red_20_balls :
  probability_white_after_red 20 10 10 = 10 / 19 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_after_red_20_balls_l461_46177


namespace NUMINAMATH_CALUDE_total_canoes_by_april_l461_46104

def canoes_built (month : Nat) : Nat :=
  match month with
  | 0 => 5  -- February (0-indexed)
  | n + 1 => 3 * canoes_built n

theorem total_canoes_by_april : 
  canoes_built 0 + canoes_built 1 + canoes_built 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_canoes_by_april_l461_46104


namespace NUMINAMATH_CALUDE_similar_triangles_problem_l461_46132

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- Given two similar triangles satisfying certain conditions, 
    prove that the corresponding side of the larger triangle is 12 feet -/
theorem similar_triangles_problem 
  (small large : Triangle)
  (area_diff : large.area - small.area = 72)
  (area_ratio : ∃ k : ℕ, large.area / small.area = k^2)
  (small_area_int : ∃ n : ℕ, small.area = n)
  (small_side : small.side = 6)
  : large.side = 12 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_problem_l461_46132


namespace NUMINAMATH_CALUDE_max_value_of_expression_l461_46161

theorem max_value_of_expression (x : ℝ) :
  ∃ (max : ℝ), max = (1 / 4 : ℝ) ∧ ∀ y : ℝ, 10^y - 100^y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l461_46161


namespace NUMINAMATH_CALUDE_double_price_profit_percentage_l461_46109

theorem double_price_profit_percentage (cost : ℝ) (initial_profit_percentage : ℝ) 
  (initial_selling_price : ℝ) (new_selling_price : ℝ) (new_profit_percentage : ℝ) :
  initial_profit_percentage = 20 →
  initial_selling_price = cost * (1 + initial_profit_percentage / 100) →
  new_selling_price = 2 * initial_selling_price →
  new_profit_percentage = ((new_selling_price - cost) / cost) * 100 →
  new_profit_percentage = 140 :=
by sorry

end NUMINAMATH_CALUDE_double_price_profit_percentage_l461_46109


namespace NUMINAMATH_CALUDE_train_length_train_length_approximation_l461_46108

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

/-- Prove that a train traveling at 50 km/h and crossing a pole in 18 seconds has a length of approximately 250 meters -/
theorem train_length_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ |train_length 50 18 - 250| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_approximation_l461_46108


namespace NUMINAMATH_CALUDE_sum_of_intercepts_l461_46138

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3 * x - 2 * y - 6 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -3

/-- Theorem: The sum of the x-intercept and y-intercept of the line 3x - 2y - 6 = 0 is -1 -/
theorem sum_of_intercepts :
  line_equation x_intercept 0 ∧ 
  line_equation 0 y_intercept ∧ 
  x_intercept + y_intercept = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_intercepts_l461_46138


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l461_46118

theorem pure_imaginary_modulus (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - 2 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (a + 2 * Complex.I) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l461_46118


namespace NUMINAMATH_CALUDE_five_thousand_five_hundred_scientific_notation_l461_46192

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem five_thousand_five_hundred_scientific_notation :
  toScientificNotation 5500 = ScientificNotation.mk 5.5 3 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_five_thousand_five_hundred_scientific_notation_l461_46192


namespace NUMINAMATH_CALUDE_ceiling_equality_iff_x_in_range_l461_46199

theorem ceiling_equality_iff_x_in_range (x : ℝ) : 
  ⌈⌈3*x⌉ + 1/2⌉ = ⌈x - 2⌉ ↔ x ∈ Set.Icc (-1) (-2/3) :=
sorry

end NUMINAMATH_CALUDE_ceiling_equality_iff_x_in_range_l461_46199


namespace NUMINAMATH_CALUDE_flag_height_l461_46156

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- The three fabric squares Bobby has -/
def fabric1 : Rectangle := ⟨8, 5⟩
def fabric2 : Rectangle := ⟨10, 7⟩
def fabric3 : Rectangle := ⟨5, 5⟩

/-- The desired length of the flag -/
def flagLength : ℝ := 15

/-- Theorem stating that the height of the flag will be 9 feet -/
theorem flag_height :
  (area fabric1 + area fabric2 + area fabric3) / flagLength = 9 := by
  sorry

end NUMINAMATH_CALUDE_flag_height_l461_46156


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l461_46136

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 1) :
  Complex.abs (a + b + c) = 1 ∨ Complex.abs (a + b + c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l461_46136


namespace NUMINAMATH_CALUDE_not_all_countries_have_complete_systems_l461_46113

/-- Represents a country's internet regulation system -/
structure InternetRegulation where
  country : String
  hasCompleteSystem : Bool

/-- Information about internet regulation systems in different countries -/
def countryRegulations : List InternetRegulation := [
  { country := "United States", hasCompleteSystem := false },
  { country := "United Kingdom", hasCompleteSystem := false },
  { country := "Russia", hasCompleteSystem := true }
]

/-- Theorem stating that not all countries (US, UK, and Russia) have complete internet regulation systems -/
theorem not_all_countries_have_complete_systems : 
  ¬ (∀ c ∈ countryRegulations, c.hasCompleteSystem = true) := by
  sorry

end NUMINAMATH_CALUDE_not_all_countries_have_complete_systems_l461_46113


namespace NUMINAMATH_CALUDE_unique_m_value_l461_46195

theorem unique_m_value (a b c m : ℤ) 
  (h1 : 0 ≤ m ∧ m ≤ 26)
  (h2 : (a + b + c) % 27 = m)
  (h3 : ((a - b) * (b - c) * (c - a)) % 27 = m) : 
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_m_value_l461_46195


namespace NUMINAMATH_CALUDE_unique_four_digit_number_with_geometric_property_l461_46103

def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def first_digit (n : ℕ) : ℕ :=
  n / 1000

def second_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

def third_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def fourth_digit (n : ℕ) : ℕ :=
  n % 10

def ab (n : ℕ) : ℕ :=
  10 * (first_digit n) + (second_digit n)

def bc (n : ℕ) : ℕ :=
  10 * (second_digit n) + (third_digit n)

def cd (n : ℕ) : ℕ :=
  10 * (third_digit n) + (fourth_digit n)

def is_increasing_geometric_sequence (x y z : ℕ) : Prop :=
  x < y ∧ y < z ∧ y * y = x * z

theorem unique_four_digit_number_with_geometric_property :
  ∃! n : ℕ, is_valid_four_digit_number n ∧
             first_digit n ≠ 0 ∧
             is_increasing_geometric_sequence (ab n) (bc n) (cd n) :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_with_geometric_property_l461_46103


namespace NUMINAMATH_CALUDE_smallest_flock_size_l461_46137

theorem smallest_flock_size (total_sparrows : ℕ) (parrot_flock_size : ℕ) : 
  total_sparrows = 182 →
  parrot_flock_size = 14 →
  ∃ (P : ℕ), total_sparrows = parrot_flock_size * P →
  (∀ (S : ℕ), S > 0 ∧ S ∣ total_sparrows ∧ (∃ (Q : ℕ), S ∣ (parrot_flock_size * Q)) → S ≥ 14) ∧
  14 ∣ total_sparrows ∧ (∃ (R : ℕ), 14 ∣ (parrot_flock_size * R)) :=
by sorry

#check smallest_flock_size

end NUMINAMATH_CALUDE_smallest_flock_size_l461_46137


namespace NUMINAMATH_CALUDE_number_divisible_by_5_power_1000_without_zero_digit_l461_46197

theorem number_divisible_by_5_power_1000_without_zero_digit :
  ∃ n : ℕ, (5^1000 ∣ n) ∧ (∀ d : ℕ, d < 10 → (n.digits 10).all (λ x => x ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_number_divisible_by_5_power_1000_without_zero_digit_l461_46197


namespace NUMINAMATH_CALUDE_angle_B_measure_l461_46125

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 1, b = 2cos(C), and sin(C)cos(A) - sin(π/4 - B)sin(π/4 + B) = 0,
    then B = π/6. -/
theorem angle_B_measure (A B C : Real) (a b c : Real) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a = 1 →
  b = 2 * Real.cos C →
  Real.sin C * Real.cos A - Real.sin (π/4 - B) * Real.sin (π/4 + B) = 0 →
  B = π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l461_46125


namespace NUMINAMATH_CALUDE_intersection_and_lines_l461_46105

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def l₃ (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define the parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

theorem intersection_and_lines :
  (∃ (x y : ℝ), l₁ x y ∧ l₂ x y) →
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) →
  (∀ (x y : ℝ), parallel_line x y ↔ (∃ (t : ℝ), x = P.1 + t ∧ y = P.2 + t * (-2))) ∧
  (∀ (x y : ℝ), perpendicular_line x y ↔ (∃ (t : ℝ), x = P.1 + t ∧ y = P.2 + t * (1/2))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_lines_l461_46105


namespace NUMINAMATH_CALUDE_parabola_sum_l461_46131

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_sum (p : Parabola) 
  (vertex_condition : p.y_at 1 = 3 ∧ (- p.b / (2 * p.a)) = 1)
  (point_condition : p.y_at 0 = 2) :
  p.a + p.b + p.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l461_46131


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l461_46181

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l461_46181


namespace NUMINAMATH_CALUDE_hawkeye_remaining_money_l461_46182

/-- Calculates the remaining money after battery charging -/
def remaining_money (charge_cost : ℚ) (num_charges : ℕ) (budget : ℚ) : ℚ :=
  budget - charge_cost * num_charges

/-- Theorem: Given the specified conditions, the remaining money is $6 -/
theorem hawkeye_remaining_money :
  remaining_money 3.5 4 20 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_remaining_money_l461_46182


namespace NUMINAMATH_CALUDE_number_problem_l461_46146

theorem number_problem (x : ℝ) : (0.2 * x = 0.2 * 650 + 190) → x = 1600 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l461_46146


namespace NUMINAMATH_CALUDE_sin_y_in_terms_of_c_and_d_l461_46144

theorem sin_y_in_terms_of_c_and_d (c d y : ℝ) 
  (h1 : c > d) (h2 : d > 0) (h3 : 0 < y) (h4 : y < π / 2)
  (h5 : Real.tan y = (3 * c * d) / (c^2 - d^2)) :
  Real.sin y = (3 * c * d) / Real.sqrt (c^4 + 7 * c^2 * d^2 + d^4) := by
  sorry

end NUMINAMATH_CALUDE_sin_y_in_terms_of_c_and_d_l461_46144


namespace NUMINAMATH_CALUDE_factorial_simplification_l461_46187

theorem factorial_simplification : (12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / 
  ((10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) + 3 * (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l461_46187


namespace NUMINAMATH_CALUDE_inequality_proof_l461_46102

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  24 * x * y * z ≤ 3 * (x + y) * (y + z) * (z + x) ∧ 
  3 * (x + y) * (y + z) * (z + x) ≤ 8 * (x^3 + y^3 + z^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l461_46102


namespace NUMINAMATH_CALUDE_cos_sin_sum_l461_46184

theorem cos_sin_sum (φ : Real) (h : Real.cos (π / 2 + φ) = Real.sqrt 3 / 2) :
  Real.cos (3 * π / 2 - φ) + Real.sin (φ - π) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_l461_46184


namespace NUMINAMATH_CALUDE_solution_range_l461_46143

-- Define the equation
def equation (m x : ℝ) : Prop :=
  m / (x - 2) + 1 = x / (2 - x)

-- Define the theorem
theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ equation m x) ↔ (m ≤ 2 ∧ m ≠ -2) := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l461_46143


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l461_46141

theorem sum_of_real_solutions (a : ℝ) (h : a > 1/2) :
  ∃ (x₁ x₂ : ℝ), 
    (Real.sqrt (3 * a - Real.sqrt (2 * a + x₁)) = x₁) ∧
    (Real.sqrt (3 * a - Real.sqrt (2 * a + x₂)) = x₂) ∧
    (x₁ + x₂ = Real.sqrt (3 * a + Real.sqrt (2 * a)) + Real.sqrt (3 * a - Real.sqrt (2 * a))) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l461_46141


namespace NUMINAMATH_CALUDE_range_of_m_l461_46167

/-- The proposition p: x^2 - 8x - 20 ≤ 0 -/
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

/-- The proposition q: [x-(1+m)][x-(1-m)] ≤ 0 -/
def q (x m : ℝ) : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0

/-- p is a sufficient condition for q -/
def p_sufficient_for_q (m : ℝ) : Prop :=
  ∀ x, p x → q x m

/-- p is not a necessary condition for q -/
def p_not_necessary_for_q (m : ℝ) : Prop :=
  ∃ x, q x m ∧ ¬(p x)

/-- m is positive -/
def m_positive (m : ℝ) : Prop := m > 0

theorem range_of_m :
  ∀ m : ℝ, (m_positive m ∧ p_sufficient_for_q m ∧ p_not_necessary_for_q m) ↔ m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l461_46167


namespace NUMINAMATH_CALUDE_division_remainder_l461_46107

theorem division_remainder (n : ℕ) : 
  (n / 8 = 8 ∧ n % 8 = 0) → n % 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l461_46107


namespace NUMINAMATH_CALUDE_ribbon_division_l461_46185

theorem ribbon_division (total_ribbon : ℚ) (num_boxes : ℕ) (ribbon_per_box : ℚ) : 
  total_ribbon = 5/12 → 
  num_boxes = 5 → 
  total_ribbon = num_boxes * ribbon_per_box → 
  ribbon_per_box = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_division_l461_46185


namespace NUMINAMATH_CALUDE_statement_is_valid_assignment_l461_46151

/-- Represents a variable in an assignment statement -/
structure Variable where
  name : String

/-- Represents an expression in an assignment statement -/
inductive Expression where
  | Var : Variable → Expression
  | Const : ℕ → Expression
  | Add : Expression → Expression → Expression

/-- Represents an assignment statement -/
structure AssignmentStatement where
  lhs : Variable
  rhs : Expression

/-- Checks if a given statement is a valid assignment statement -/
def isValidAssignmentStatement (stmt : AssignmentStatement) : Prop :=
  ∃ (v : Variable) (e : Expression), stmt.lhs = v ∧ stmt.rhs = e

/-- The statement "S = a + 1" -/
def statement : AssignmentStatement :=
  { lhs := ⟨"S"⟩,
    rhs := Expression.Add (Expression.Var ⟨"a"⟩) (Expression.Const 1) }

/-- Theorem: The statement "S = a + 1" is a valid assignment statement -/
theorem statement_is_valid_assignment : isValidAssignmentStatement statement := by
  sorry


end NUMINAMATH_CALUDE_statement_is_valid_assignment_l461_46151


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l461_46168

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

theorem geometric_sequence_sum (a : ℕ → ℝ) (n : ℕ) :
  (∀ k, a k > 0) →
  geometric_sequence a →
  a 2 = 3 →
  a 1 + a 3 = 10 →
  (∃ S_n : ℝ, S_n = (27/2) - (1/2) * 3^(n-3) ∨ S_n = (3^n - 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l461_46168


namespace NUMINAMATH_CALUDE_square_region_perimeter_l461_46171

theorem square_region_perimeter (total_area : ℝ) (num_squares : ℕ) (h1 : total_area = 144) (h2 : num_squares = 4) :
  let square_area : ℝ := total_area / num_squares
  let side_length : ℝ := Real.sqrt square_area
  let perimeter : ℝ := 2 * side_length * num_squares
  perimeter = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_region_perimeter_l461_46171


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l461_46140

/-- A quadratic function with the given properties -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The symmetry property of f -/
def symmetry_property (b c : ℝ) : Prop :=
  ∀ x, f b c (2 + x) = f b c (2 - x)

theorem quadratic_function_properties (b c : ℝ) 
  (h : symmetry_property b c) : 
  b = 4 ∧ 
  (∀ a : ℝ, f b c (5/4) ≥ f b c (-a^2 - a + 1)) ∧
  (∀ a : ℝ, f b c (5/4) = f b c (-a^2 - a + 1) ↔ a = -1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l461_46140


namespace NUMINAMATH_CALUDE_system_solution_l461_46174

theorem system_solution (x y : ℝ) : 
  (x^2 + y^2 ≤ 1 ∧ 
   16 * x^4 - 8 * x^2 * y^2 + y^4 - 40 * x^2 - 10 * y^2 + 25 = 0) ↔ 
  ((x = -2 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) ∨
   (x = -2 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
   (x = 2 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) ∨
   (x = 2 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l461_46174


namespace NUMINAMATH_CALUDE_gcf_18_30_l461_46164

theorem gcf_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_18_30_l461_46164


namespace NUMINAMATH_CALUDE_square_of_r_minus_three_l461_46180

theorem square_of_r_minus_three (r : ℝ) (h : r^2 - 6*r + 5 = 0) : (r - 3)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_r_minus_three_l461_46180


namespace NUMINAMATH_CALUDE_complex_equality_l461_46148

theorem complex_equality (u v : ℂ) 
  (h1 : 3 * Complex.abs (u + 1) * Complex.abs (v + 1) ≥ Complex.abs (u * v + 5 * u + 5 * v + 1))
  (h2 : Complex.abs (u + v) = Complex.abs (u * v + 1)) :
  u = 1 ∨ v = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_l461_46148


namespace NUMINAMATH_CALUDE_important_rectangle_difference_l461_46153

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (isBlack : Nat → Nat → Bool)

/-- Represents a rectangle on the chessboard -/
structure Rectangle :=
  (top : Nat)
  (left : Nat)
  (bottom : Nat)
  (right : Nat)

/-- Checks if a rectangle is important -/
def isImportantRectangle (board : Chessboard) (rect : Rectangle) : Bool :=
  board.isBlack rect.top rect.left &&
  board.isBlack rect.top rect.right &&
  board.isBlack rect.bottom rect.left &&
  board.isBlack rect.bottom rect.right

/-- Counts the number of important rectangles containing a square -/
def countImportantRectangles (board : Chessboard) (row : Nat) (col : Nat) : Nat :=
  sorry

/-- Sums the counts for all squares of a given color -/
def sumCounts (board : Chessboard) (isBlack : Bool) : Nat :=
  sorry

/-- The main theorem -/
theorem important_rectangle_difference (board : Chessboard) :
  board.size = 8 →
  (∀ i j, board.isBlack i j = ((i + j) % 2 = 0)) →
  (sumCounts board true) - (sumCounts board false) = 36 :=
sorry

end NUMINAMATH_CALUDE_important_rectangle_difference_l461_46153


namespace NUMINAMATH_CALUDE_simplify_expression_l461_46128

theorem simplify_expression : (7^5 + 2^8) * (2^3 - (-2)^3)^7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l461_46128


namespace NUMINAMATH_CALUDE_circle_through_line_intersections_l461_46147

/-- Given a line that intersects the coordinate axes, prove that the circle passing through
    the origin and the intersection points has a specific equation. -/
theorem circle_through_line_intersections (x y : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1 / 2 - A.2 / 4 = 1) ∧ 
    (B.1 / 2 - B.2 / 4 = 1) ∧ 
    (A.2 = 0) ∧ 
    (B.1 = 0) ∧
    ((x - 1)^2 + (y + 2)^2 = 5) ↔ 
    (x^2 + y^2 = A.1^2 + A.2^2 ∧ 
     x^2 + y^2 = B.1^2 + B.2^2)) :=
by sorry


end NUMINAMATH_CALUDE_circle_through_line_intersections_l461_46147


namespace NUMINAMATH_CALUDE_distribute_five_students_three_classes_l461_46169

/-- The number of ways to distribute n students among k classes with a maximum of m students per class -/
def distributeStudents (n k m : ℕ) : ℕ := sorry

/-- Theorem: Distributing 5 students among 3 classes with at most 2 students per class yields 90 possibilities -/
theorem distribute_five_students_three_classes : distributeStudents 5 3 2 = 90 := by sorry

end NUMINAMATH_CALUDE_distribute_five_students_three_classes_l461_46169


namespace NUMINAMATH_CALUDE_defective_units_shipped_l461_46122

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ) :
  defective_rate = 0.05 →
  shipped_rate = 0.04 →
  (defective_rate * shipped_rate * 100) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l461_46122


namespace NUMINAMATH_CALUDE_problem_solution_l461_46124

theorem problem_solution (x y : ℚ) : 
  x / y = 12 / 5 → y = 25 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l461_46124


namespace NUMINAMATH_CALUDE_banquet_solution_l461_46100

def banquet_problem (total_attendees : ℕ) (resident_price : ℚ) (non_resident_price : ℚ) (total_revenue : ℚ) : Prop :=
  ∃ (residents : ℕ),
    residents ≤ total_attendees ∧
    residents * resident_price + (total_attendees - residents) * non_resident_price = total_revenue

theorem banquet_solution :
  banquet_problem 586 (12.95 : ℚ) (17.95 : ℚ) (9423.70 : ℚ) →
  ∃ (residents : ℕ), residents = 220 ∧ banquet_problem 586 (12.95 : ℚ) (17.95 : ℚ) (9423.70 : ℚ) :=
by
  sorry

#check banquet_solution

end NUMINAMATH_CALUDE_banquet_solution_l461_46100


namespace NUMINAMATH_CALUDE_line_slope_l461_46165

/-- The slope of a line given by the equation 3y + 4x = 12 is -4/3 -/
theorem line_slope (x y : ℝ) : 3 * y + 4 * x = 12 → (y - 4) / (x - 0) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l461_46165


namespace NUMINAMATH_CALUDE_tuesday_rainfall_l461_46196

/-- Rainfall problem -/
theorem tuesday_rainfall (total_rainfall average_rainfall : ℝ) 
  (h1 : total_rainfall = 7 * average_rainfall)
  (h2 : average_rainfall = 3)
  (h3 : ∃ tuesday_rainfall : ℝ, 
    tuesday_rainfall = total_rainfall - tuesday_rainfall) :
  ∃ tuesday_rainfall : ℝ, tuesday_rainfall = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_rainfall_l461_46196


namespace NUMINAMATH_CALUDE_percent_relation_l461_46163

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) : 
  (4 * b) / a = 10/3 := by sorry

end NUMINAMATH_CALUDE_percent_relation_l461_46163


namespace NUMINAMATH_CALUDE_quadratic_factorization_l461_46126

theorem quadratic_factorization (a b c : ℤ) : 
  (∀ x, x^2 + 9*x + 14 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 30 = (x + b) * (x - c)) →
  a + b + c = 15 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l461_46126


namespace NUMINAMATH_CALUDE_weird_calculator_theorem_l461_46162

/-- Represents the calculator operations -/
inductive Operation
| DSharp : Operation  -- doubles and adds 1
| DFlat  : Operation  -- doubles and subtracts 1

/-- Applies a single operation to a number -/
def apply_operation (op : Operation) (x : ℕ) : ℕ :=
  match op with
  | Operation.DSharp => 2 * x + 1
  | Operation.DFlat  => 2 * x - 1

/-- Applies a sequence of operations to a number -/
def apply_sequence (ops : List Operation) (x : ℕ) : ℕ :=
  match ops with
  | [] => x
  | op :: rest => apply_sequence rest (apply_operation op x)

/-- The set of all possible results after 8 operations starting from 1 -/
def possible_results : Set ℕ :=
  {n | ∃ (ops : List Operation), ops.length = 8 ∧ apply_sequence ops 1 = n}

theorem weird_calculator_theorem :
  possible_results = {n : ℕ | n < 512 ∧ n % 2 = 1} :=
sorry

end NUMINAMATH_CALUDE_weird_calculator_theorem_l461_46162


namespace NUMINAMATH_CALUDE_no_hexagon_tiling_l461_46101

-- Define a grid hexagon
structure GridHexagon where
  -- Add necessary fields to define the hexagon
  -- This is a placeholder and should be adjusted based on the specific hexagon properties
  side_length : ℝ
  diagonal_length : ℝ

-- Define a grid rectangle
structure GridRectangle where
  width : ℕ
  height : ℕ

-- Define the tiling property
def can_tile (r : GridRectangle) (h : GridHexagon) : Prop :=
  -- This is a placeholder for the actual tiling condition
  -- It should represent that the rectangle can be tiled with the hexagons
  sorry

-- The main theorem
theorem no_hexagon_tiling (r : GridRectangle) (h : GridHexagon) : 
  ¬(can_tile r h) := by
  sorry

end NUMINAMATH_CALUDE_no_hexagon_tiling_l461_46101


namespace NUMINAMATH_CALUDE_like_terms_imply_exponents_l461_46172

/-- Two algebraic terms are considered like terms if they have the same variables with the same exponents. -/
def are_like_terms (term1 term2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (p q : ℕ), ∀ (a b : ℝ), term1 a b = c₁ * a^p * b^q ∧ term2 a b = c₂ * a^p * b^q

/-- The theorem states that if the given terms are like terms, then m = 4 and n = 2. -/
theorem like_terms_imply_exponents 
  (m n : ℕ) 
  (h : are_like_terms (λ a b => (1/3) * a^2 * b^m) (λ a b => (-1/2) * a^n * b^4)) : 
  m = 4 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_exponents_l461_46172


namespace NUMINAMATH_CALUDE_hayden_ironing_days_l461_46120

/-- Given that Hayden spends 8 minutes ironing clothes each day he does so,
    and over 4 weeks he spends 160 minutes ironing,
    prove that he irons his clothes 5 days per week. -/
theorem hayden_ironing_days (minutes_per_day : ℕ) (total_minutes : ℕ) (weeks : ℕ) :
  minutes_per_day = 8 →
  total_minutes = 160 →
  weeks = 4 →
  (total_minutes / weeks) / minutes_per_day = 5 :=
by sorry

end NUMINAMATH_CALUDE_hayden_ironing_days_l461_46120


namespace NUMINAMATH_CALUDE_first_number_proof_l461_46123

theorem first_number_proof (N : ℕ) : 
  (∃ k m : ℕ, N = 170 * k + 10 ∧ 875 = 170 * m + 25) →
  N = 860 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l461_46123


namespace NUMINAMATH_CALUDE_cone_rolling_theorem_l461_46130

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Predicate to check if a number is not divisible by the square of any prime -/
def notDivisibleBySquareOfPrime (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^2 ∣ n)

/-- The main theorem -/
theorem cone_rolling_theorem (cone : RightCircularCone) 
  (m n : ℕ) (h_sqrt : cone.h / cone.r = m * Real.sqrt n) 
  (h_prime : notDivisibleBySquareOfPrime n) 
  (h_rotations : (2 * Real.pi * Real.sqrt (cone.r^2 + cone.h^2)) = 50 * cone.r * Real.pi) :
  m + n = 50 := by sorry

end NUMINAMATH_CALUDE_cone_rolling_theorem_l461_46130


namespace NUMINAMATH_CALUDE_f_neg_one_eq_zero_iff_r_eq_neg_eight_l461_46110

/-- A polynomial function f(x) with a parameter r -/
def f (r : ℝ) (x : ℝ) : ℝ := 3 * x^4 + x^3 + 2 * x^2 - 4 * x + r

/-- Theorem stating that f(-1) = 0 if and only if r = -8 -/
theorem f_neg_one_eq_zero_iff_r_eq_neg_eight :
  ∀ r : ℝ, f r (-1) = 0 ↔ r = -8 := by sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_zero_iff_r_eq_neg_eight_l461_46110


namespace NUMINAMATH_CALUDE_cookie_difference_l461_46121

theorem cookie_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) :
  initial_sweet = 39 →
  initial_salty = 6 →
  eaten_sweet = 32 →
  eaten_salty = 23 →
  eaten_sweet - eaten_salty = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l461_46121


namespace NUMINAMATH_CALUDE_park_track_area_increase_l461_46112

def small_diameter : ℝ := 15
def large_diameter : ℝ := 20

theorem park_track_area_increase :
  let small_area := π * (small_diameter / 2)^2
  let large_area := π * (large_diameter / 2)^2
  (large_area - small_area) / small_area = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_park_track_area_increase_l461_46112


namespace NUMINAMATH_CALUDE_mike_current_salary_l461_46188

def mike_salary_five_months_ago : ℕ := 10000
def fred_salary_five_months_ago : ℕ := 1000
def salary_increase_percentage : ℕ := 40

theorem mike_current_salary :
  let total_salary_five_months_ago := mike_salary_five_months_ago + fred_salary_five_months_ago
  let salary_increase := (salary_increase_percentage * total_salary_five_months_ago) / 100
  mike_salary_five_months_ago + salary_increase = 15400 := by
  sorry

end NUMINAMATH_CALUDE_mike_current_salary_l461_46188


namespace NUMINAMATH_CALUDE_max_value_d_l461_46191

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 17) :
  d ≤ (5 + Real.sqrt 123) / 2 ∧ 
  ∃ (a' b' c' : ℝ), a' + b' + c' + (5 + Real.sqrt 123) / 2 = 10 ∧ 
    a'*b' + a'*c' + a'*((5 + Real.sqrt 123) / 2) + b'*c' + 
    b'*((5 + Real.sqrt 123) / 2) + c'*((5 + Real.sqrt 123) / 2) = 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_d_l461_46191
