import Mathlib

namespace NUMINAMATH_CALUDE_solve_linear_equation_l3032_303279

theorem solve_linear_equation (x : ℝ) (h : 3*x - 4*x + 7*x = 120) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3032_303279


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3032_303292

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3032_303292


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3032_303238

/-- Given a right-angled triangle with sides a, b, and hypotenuse c,
    and a point (m,n) on the line ax+by+2c=0,
    the minimum value of m^2 + n^2 is 4. -/
theorem min_distance_to_line (a b c : ℝ) (m n : ℝ → ℝ) :
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  (∀ t, a * (m t) + b * (n t) + 2*c = 0) →
  (∃ t₀, ∀ t, (m t)^2 + (n t)^2 ≥ (m t₀)^2 + (n t₀)^2) →
  ∃ t₀, (m t₀)^2 + (n t₀)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3032_303238


namespace NUMINAMATH_CALUDE_power_of_64_l3032_303235

theorem power_of_64 : 64^(5/3) = 1024 := by
  have h : 64 = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_64_l3032_303235


namespace NUMINAMATH_CALUDE_evaluate_expression_l3032_303215

theorem evaluate_expression :
  (2^1501 + 5^1502)^2 - (2^1501 - 5^1502)^2 = 20 * 10^1501 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3032_303215


namespace NUMINAMATH_CALUDE_square_eq_sixteen_l3032_303208

theorem square_eq_sixteen (x : ℝ) : (x - 3)^2 = 16 ↔ x = 7 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_eq_sixteen_l3032_303208


namespace NUMINAMATH_CALUDE_adam_tattoo_count_l3032_303240

/-- The number of tattoos Jason has on each arm -/
def jason_arm_tattoos : ℕ := 2

/-- The number of tattoos Jason has on each leg -/
def jason_leg_tattoos : ℕ := 3

/-- The number of arms Jason has -/
def jason_arms : ℕ := 2

/-- The number of legs Jason has -/
def jason_legs : ℕ := 2

/-- The total number of tattoos Jason has -/
def jason_total_tattoos : ℕ := jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs

/-- The number of tattoos Adam has -/
def adam_tattoos : ℕ := 2 * jason_total_tattoos + 3

theorem adam_tattoo_count : adam_tattoos = 23 := by
  sorry

end NUMINAMATH_CALUDE_adam_tattoo_count_l3032_303240


namespace NUMINAMATH_CALUDE_tutors_next_meeting_l3032_303211

theorem tutors_next_meeting (elena fiona george harry : ℕ) 
  (h_elena : elena = 5)
  (h_fiona : fiona = 6)
  (h_george : george = 8)
  (h_harry : harry = 9) :
  Nat.lcm (Nat.lcm (Nat.lcm elena fiona) george) harry = 360 := by
  sorry

end NUMINAMATH_CALUDE_tutors_next_meeting_l3032_303211


namespace NUMINAMATH_CALUDE_personal_income_tax_l3032_303230

/-- Personal income tax calculation -/
theorem personal_income_tax (salary : ℕ) (tax_free : ℕ) (rate1 : ℚ) (rate2 : ℚ) (threshold : ℕ) : 
  salary = 2900 ∧ 
  tax_free = 2000 ∧ 
  rate1 = 5/100 ∧ 
  rate2 = 10/100 ∧ 
  threshold = 500 → 
  (min threshold (salary - tax_free) : ℚ) * rate1 + 
  (max 0 ((salary - tax_free) - threshold) : ℚ) * rate2 = 65 := by
sorry

end NUMINAMATH_CALUDE_personal_income_tax_l3032_303230


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3032_303275

theorem arctan_equation_solution :
  ∃ y : ℝ, y > 0 ∧ Real.arctan (2 / y) + Real.arctan (1 / y^2) = π / 4 :=
by
  -- The proof would go here
  sorry

#check arctan_equation_solution

end NUMINAMATH_CALUDE_arctan_equation_solution_l3032_303275


namespace NUMINAMATH_CALUDE_self_centered_max_solutions_l3032_303261

/-- A polynomial is self-centered if it has integer coefficients and p(200) = 200 -/
def SelfCentered (p : ℤ → ℤ) : Prop :=
  (∀ x, ∃ n : ℕ, p x = (x : ℤ) ^ n) ∧ p 200 = 200

/-- The main theorem: any self-centered polynomial has at most 10 integer solutions to p(k) = k^4 -/
theorem self_centered_max_solutions (p : ℤ → ℤ) (h : SelfCentered p) :
  ∃ s : Finset ℤ, s.card ≤ 10 ∧ ∀ k : ℤ, p k = k^4 → k ∈ s := by
  sorry

end NUMINAMATH_CALUDE_self_centered_max_solutions_l3032_303261


namespace NUMINAMATH_CALUDE_steve_pages_written_l3032_303231

/-- Calculates the total number of pages Steve writes in a month -/
def total_pages_written (days_per_month : ℕ) (days_between_letters : ℕ) 
  (minutes_per_regular_letter : ℕ) (minutes_per_page : ℕ) 
  (minutes_for_long_letter : ℕ) : ℕ :=
  let regular_letters := days_per_month / days_between_letters
  let pages_per_regular_letter := minutes_per_regular_letter / minutes_per_page
  let regular_pages := regular_letters * pages_per_regular_letter
  let long_letter_pages := minutes_for_long_letter / (2 * minutes_per_page)
  regular_pages + long_letter_pages

theorem steve_pages_written :
  total_pages_written 30 3 20 10 80 = 24 := by sorry

end NUMINAMATH_CALUDE_steve_pages_written_l3032_303231


namespace NUMINAMATH_CALUDE_dice_sum_probability_l3032_303263

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The target sum we're aiming for -/
def target_sum : ℕ := 15

/-- 
The number of ways to achieve the target sum when rolling the specified number of dice.
This is equivalent to the coefficient of x^target_sum in the expansion of (x + x^2 + ... + x^num_faces)^num_dice.
-/
def num_ways_to_achieve_sum : ℕ := 2002

theorem dice_sum_probability : 
  num_ways_to_achieve_sum = 2002 := by sorry

end NUMINAMATH_CALUDE_dice_sum_probability_l3032_303263


namespace NUMINAMATH_CALUDE_z_values_l3032_303265

theorem z_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) :
  let z := ((x - 3)^2 * (x + 4)) / (2*x - 4)
  z = 64.8 ∨ z = -10.125 := by
sorry

end NUMINAMATH_CALUDE_z_values_l3032_303265


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3032_303246

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b / a = c / b

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  geometric_sequence (a 1) (a 2) (a 5) →
  d = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3032_303246


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3032_303298

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (n_balls : ℕ) (n_boxes : ℕ) : ℕ :=
  n_boxes ^ n_balls

/-- Theorem: The number of ways to distribute 6 distinguishable balls into 3 distinguishable boxes is 3^6 -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 3^6 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3032_303298


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3032_303270

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^5 - 2 * X^3 + 5 * X - 8 = (X^2 - 3 * X + 2) * q + (74 * X - 76) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3032_303270


namespace NUMINAMATH_CALUDE_fruit_cup_cost_calculation_l3032_303223

/-- The cost of a fruit cup in dollars -/
def fruit_cup_cost : ℝ := 1.80

/-- The cost of a muffin in dollars -/
def muffin_cost : ℝ := 2

/-- The number of muffins Francis had -/
def francis_muffins : ℕ := 2

/-- The number of fruit cups Francis had -/
def francis_fruit_cups : ℕ := 2

/-- The number of muffins Kiera had -/
def kiera_muffins : ℕ := 2

/-- The number of fruit cups Kiera had -/
def kiera_fruit_cups : ℕ := 1

/-- The total cost of their breakfast in dollars -/
def total_cost : ℝ := 17

theorem fruit_cup_cost_calculation :
  (francis_muffins * muffin_cost + francis_fruit_cups * fruit_cup_cost) +
  (kiera_muffins * muffin_cost + kiera_fruit_cups * fruit_cup_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_fruit_cup_cost_calculation_l3032_303223


namespace NUMINAMATH_CALUDE_plot_length_is_56_l3032_303228

/-- Proves that the length of a rectangular plot is 56 meters given the specified conditions -/
theorem plot_length_is_56 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 12 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.5 →
  total_cost = 5300 →
  total_cost = perimeter * cost_per_meter →
  length = 56 := by
sorry

end NUMINAMATH_CALUDE_plot_length_is_56_l3032_303228


namespace NUMINAMATH_CALUDE_black_hair_ratio_l3032_303210

/-- Represents the ratio of hair colors in the class -/
structure HairColorRatio :=
  (red : ℕ)
  (blonde : ℕ)
  (black : ℕ)

/-- Represents the class information -/
structure ClassInfo :=
  (ratio : HairColorRatio)
  (redHairedKids : ℕ)
  (totalKids : ℕ)

/-- The main theorem -/
theorem black_hair_ratio (c : ClassInfo) 
  (h1 : c.ratio = HairColorRatio.mk 3 6 7)
  (h2 : c.redHairedKids = 9)
  (h3 : c.totalKids = 48) : 
  (c.ratio.black * c.redHairedKids / c.ratio.red : ℚ) / c.totalKids = 7 / 16 := by
  sorry

#check black_hair_ratio

end NUMINAMATH_CALUDE_black_hair_ratio_l3032_303210


namespace NUMINAMATH_CALUDE_right_triangle_legs_from_altitude_areas_l3032_303224

/-- Given a right-angled triangle ABC with right angle at C, and altitude CD to hypotenuse AB
    dividing the triangle into two triangles with areas Q and q, 
    the legs of the triangle are √(2(q + Q)√(q/Q)) and √(2(q + Q)√(Q/q)). -/
theorem right_triangle_legs_from_altitude_areas (Q q : ℝ) (hQ : Q > 0) (hq : q > 0) :
  ∃ (AC BC : ℝ),
    AC = Real.sqrt (2 * (q + Q) * Real.sqrt (q / Q)) ∧
    BC = Real.sqrt (2 * (q + Q) * Real.sqrt (Q / q)) ∧
    AC^2 + BC^2 = (AC * BC)^2 / (Q + q) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_from_altitude_areas_l3032_303224


namespace NUMINAMATH_CALUDE_water_depth_approx_0_6_l3032_303259

/-- Represents a horizontal cylindrical tank partially filled with water -/
structure WaterTank where
  length : ℝ
  diameter : ℝ
  exposedArea : ℝ

/-- Calculates the depth of water in the tank -/
def waterDepth (tank : WaterTank) : ℝ :=
  sorry

/-- Theorem stating that the water depth is approximately 0.6 feet for the given tank -/
theorem water_depth_approx_0_6 (tank : WaterTank) 
  (h1 : tank.length = 12)
  (h2 : tank.diameter = 8)
  (h3 : tank.exposedArea = 50) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |waterDepth tank - 0.6| < ε :=
  sorry

end NUMINAMATH_CALUDE_water_depth_approx_0_6_l3032_303259


namespace NUMINAMATH_CALUDE_blue_eyes_count_l3032_303257

/-- The number of people in the theater -/
def total_people : ℕ := 100

/-- The number of people with brown eyes -/
def brown_eyes : ℕ := total_people / 2

/-- The number of people with black eyes -/
def black_eyes : ℕ := total_people / 4

/-- The number of people with green eyes -/
def green_eyes : ℕ := 6

/-- The number of people with blue eyes -/
def blue_eyes : ℕ := total_people - (brown_eyes + black_eyes + green_eyes)

theorem blue_eyes_count : blue_eyes = 19 := by
  sorry

end NUMINAMATH_CALUDE_blue_eyes_count_l3032_303257


namespace NUMINAMATH_CALUDE_jacket_cost_l3032_303287

/-- Represents the cost of clothing items and shipments -/
structure ClothingCost where
  sweater : ℝ
  jacket : ℝ

/-- Represents a shipment of clothing items -/
structure Shipment where
  sweaters : ℕ
  jackets : ℕ
  totalCost : ℝ

/-- The problem statement -/
theorem jacket_cost (cost : ClothingCost) (shipment1 shipment2 : Shipment) :
  shipment1.sweaters = 10 →
  shipment1.jackets = 20 →
  shipment1.totalCost = 800 →
  shipment2.sweaters = 5 →
  shipment2.jackets = 15 →
  shipment2.totalCost = 550 →
  shipment1.sweaters * cost.sweater + shipment1.jackets * cost.jacket = shipment1.totalCost →
  shipment2.sweaters * cost.sweater + shipment2.jackets * cost.jacket = shipment2.totalCost →
  cost.jacket = 30 := by
  sorry


end NUMINAMATH_CALUDE_jacket_cost_l3032_303287


namespace NUMINAMATH_CALUDE_twigs_to_find_l3032_303243

/-- The number of twigs already in the nest circle -/
def twigs_in_circle : ℕ := 12

/-- The number of additional twigs needed for each twig in the circle -/
def twigs_per_existing : ℕ := 6

/-- The fraction of needed twigs dropped by the tree -/
def tree_dropped_fraction : ℚ := 1/3

/-- Theorem stating how many twigs the bird still needs to find -/
theorem twigs_to_find : 
  (twigs_in_circle * twigs_per_existing : ℕ) - 
  (twigs_in_circle * twigs_per_existing : ℕ) * tree_dropped_fraction = 48 := by
  sorry

end NUMINAMATH_CALUDE_twigs_to_find_l3032_303243


namespace NUMINAMATH_CALUDE_bus_row_capacity_l3032_303207

/-- Represents a school bus with a given number of rows and total capacity. -/
structure SchoolBus where
  rows : ℕ
  totalCapacity : ℕ

/-- Calculates the capacity of each row in the school bus. -/
def rowCapacity (bus : SchoolBus) : ℕ :=
  bus.totalCapacity / bus.rows

/-- Theorem stating that for a bus with 20 rows and a total capacity of 80,
    the capacity of each row is 4. -/
theorem bus_row_capacity :
  let bus : SchoolBus := { rows := 20, totalCapacity := 80 }
  rowCapacity bus = 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_row_capacity_l3032_303207


namespace NUMINAMATH_CALUDE_angle_is_120_degrees_l3032_303289

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_120_degrees (a b : ℝ × ℝ) 
  (h1 : Real.sqrt (a.1^2 + a.2^2) = 4)
  (h2 : b = (-1, 0))
  (h3 : (a.1 + 2 * b.1) * b.1 + (a.2 + 2 * b.2) * b.2 = 0) :
  angle_between_vectors a b = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_angle_is_120_degrees_l3032_303289


namespace NUMINAMATH_CALUDE_marbles_cost_l3032_303255

/-- The amount spent on marbles, given the total spent on toys and the cost of a football -/
def amount_spent_on_marbles (total_spent : ℝ) (football_cost : ℝ) : ℝ :=
  total_spent - football_cost

/-- Theorem stating that the amount spent on marbles is $6.59 -/
theorem marbles_cost (total_spent : ℝ) (football_cost : ℝ)
  (h1 : total_spent = 12.30)
  (h2 : football_cost = 5.71) :
  amount_spent_on_marbles total_spent football_cost = 6.59 := by
  sorry

end NUMINAMATH_CALUDE_marbles_cost_l3032_303255


namespace NUMINAMATH_CALUDE_mathilda_debt_repayment_l3032_303212

/-- Mathilda's debt repayment problem -/
theorem mathilda_debt_repayment 
  (original_debt : ℝ) 
  (remaining_percentage : ℝ) 
  (initial_installment : ℝ) :
  original_debt = 500 ∧ 
  remaining_percentage = 75 ∧ 
  initial_installment = original_debt * (100 - remaining_percentage) / 100 →
  initial_installment = 125 := by
sorry

end NUMINAMATH_CALUDE_mathilda_debt_repayment_l3032_303212


namespace NUMINAMATH_CALUDE_smallest_enclosing_circle_l3032_303250

-- Define the lines
def line1 (x y : ℝ) : Prop := x + 2 * y - 5 = 0
def line2 (x y : ℝ) : Prop := y - 2 = 0
def line3 (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the triangle
def triangle (A B C : ℝ × ℝ) : Prop :=
  line1 A.1 A.2 ∧ line2 A.1 A.2 ∧
  line2 B.1 B.2 ∧ line3 B.1 B.2 ∧
  line1 C.1 C.2 ∧ line3 C.1 C.2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1.5)^2 = 1.25

-- Theorem statement
theorem smallest_enclosing_circle 
  (A B C : ℝ × ℝ) 
  (h : triangle A B C) :
  ∀ x y : ℝ, 
  (∀ px py : ℝ, (px = A.1 ∧ py = A.2) ∨ (px = B.1 ∧ py = B.2) ∨ (px = C.1 ∧ py = C.2) → 
    (x - px)^2 + (y - py)^2 ≤ 1.25) ↔ 
  circle_equation x y :=
sorry

end NUMINAMATH_CALUDE_smallest_enclosing_circle_l3032_303250


namespace NUMINAMATH_CALUDE_innovation_cup_award_eligibility_l3032_303278

/-- Represents the "Innovation Cup" basketball competition rules and Xiao Ming's team's goal --/
theorem innovation_cup_award_eligibility 
  (total_games : ℕ) 
  (min_points_for_award : ℕ) 
  (points_per_win : ℕ) 
  (points_per_loss : ℕ) 
  (h1 : total_games = 8)
  (h2 : min_points_for_award = 12)
  (h3 : points_per_win = 2)
  (h4 : points_per_loss = 1)
  : ∀ x : ℕ, x ≤ total_games → 
    (x * points_per_win + (total_games - x) * points_per_loss ≥ min_points_for_award ↔ 
     2 * x + (8 - x) ≥ 12) :=
by sorry

end NUMINAMATH_CALUDE_innovation_cup_award_eligibility_l3032_303278


namespace NUMINAMATH_CALUDE_dans_initial_money_l3032_303237

/-- Represents Dan's money transactions -/
def dans_money (initial : ℕ) (candy_cost : ℕ) (chocolate_cost : ℕ) (remaining : ℕ) : Prop :=
  initial = candy_cost + chocolate_cost + remaining

theorem dans_initial_money : 
  ∃ (initial : ℕ), dans_money initial 2 3 2 ∧ initial = 7 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_money_l3032_303237


namespace NUMINAMATH_CALUDE_probability_same_color_is_correct_l3032_303209

def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def green_marbles : ℕ := 4

def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

def probability_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem probability_same_color_is_correct : probability_same_color = 106 / 109725 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_correct_l3032_303209


namespace NUMINAMATH_CALUDE_max_value_theorem_l3032_303288

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if all given digits are distinct -/
def distinct (x y z w : Digit) : Prop :=
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w

/-- Converts a four-digit number to its integer representation -/
def toInt (a b c d : Digit) : Nat :=
  1000 * a.val + 100 * b.val + 10 * c.val + d.val

/-- Main theorem -/
theorem max_value_theorem (x y z w v_1 v_2 v_3 v_4 : Digit) :
  distinct x y z w →
  (x.val * y.val * z.val + w.val = toInt v_1 v_2 v_3 v_4) →
  ∀ (a b c d : Digit), distinct a b c d →
    (a.val * b.val * c.val + d.val ≤ toInt v_1 v_2 v_3 v_4) →
  toInt v_1 v_2 v_3 v_4 = 9898 ∧ w.val = 98 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3032_303288


namespace NUMINAMATH_CALUDE_power_division_simplification_l3032_303205

theorem power_division_simplification : 8^15 / 64^3 = 8^9 := by
  sorry

end NUMINAMATH_CALUDE_power_division_simplification_l3032_303205


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3032_303221

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3032_303221


namespace NUMINAMATH_CALUDE_summer_determination_l3032_303267

def has_entered_summer (temperatures : List ℤ) : Prop :=
  temperatures.length = 5 ∧ ∀ t ∈ temperatures, t ≥ 22

def median (l : List ℤ) : ℤ := sorry
def mode (l : List ℤ) : ℤ := sorry
def mean (l : List ℤ) : ℚ := sorry
def variance (l : List ℤ) : ℚ := sorry

theorem summer_determination :
  ∀ (temps_A temps_B temps_C temps_D : List ℤ),
    (median temps_A = 24 ∧ mode temps_A = 22) →
    (median temps_B = 25 ∧ mean temps_B = 24) →
    (mean temps_C = 22 ∧ mode temps_C = 22) →
    (28 ∈ temps_D ∧ mean temps_D = 24 ∧ variance temps_D = 4.8) →
    (has_entered_summer temps_A ∧
     has_entered_summer temps_D ∧
     ¬(has_entered_summer temps_B ∧ has_entered_summer temps_C)) :=
by sorry

end NUMINAMATH_CALUDE_summer_determination_l3032_303267


namespace NUMINAMATH_CALUDE_correct_num_schedules_l3032_303241

/-- Represents a subject in the school schedule -/
inductive Subject
| Chinese
| Mathematics
| English
| ScienceComprehensive

/-- Represents a class period -/
inductive ClassPeriod
| First
| Second
| Third

/-- A schedule is a function that assigns subjects to class periods -/
def Schedule := ClassPeriod → List Subject

/-- Checks if a schedule is valid according to the problem constraints -/
def isValidSchedule (s : Schedule) : Prop :=
  (∀ subject : Subject, ∃ period : ClassPeriod, subject ∈ s period) ∧
  (∀ period : ClassPeriod, s period ≠ []) ∧
  (∀ period : ClassPeriod, Subject.Mathematics ∈ s period → Subject.ScienceComprehensive ∉ s period) ∧
  (∀ period : ClassPeriod, Subject.ScienceComprehensive ∈ s period → Subject.Mathematics ∉ s period)

/-- The number of valid schedules -/
def numValidSchedules : ℕ := sorry

theorem correct_num_schedules : numValidSchedules = 30 := by sorry

end NUMINAMATH_CALUDE_correct_num_schedules_l3032_303241


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3032_303234

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two parallel vectors (1,2) and (m,1), prove that m = 1/2 -/
theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (m, 1)
  are_parallel a b → m = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3032_303234


namespace NUMINAMATH_CALUDE_chris_birthday_money_l3032_303269

/-- Calculates the total amount of money Chris has after receiving birthday gifts -/
def total_money (initial_amount grandmother_gift aunt_uncle_gift parents_gift : ℕ) : ℕ :=
  initial_amount + grandmother_gift + aunt_uncle_gift + parents_gift

/-- Proves that Chris's total money after receiving gifts is correct -/
theorem chris_birthday_money :
  total_money 159 25 20 75 = 279 := by
  sorry

end NUMINAMATH_CALUDE_chris_birthday_money_l3032_303269


namespace NUMINAMATH_CALUDE_eldest_age_difference_l3032_303218

/-- Represents the ages of three grandchildren -/
structure GrandchildrenAges where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : GrandchildrenAges) : Prop :=
  ages.middle = ages.youngest + 3 ∧
  ages.eldest = 3 * ages.youngest ∧
  ages.eldest = 15

theorem eldest_age_difference (ages : GrandchildrenAges) :
  satisfiesConditions ages →
  ages.eldest = ages.youngest + ages.middle + 2 := by
  sorry

end NUMINAMATH_CALUDE_eldest_age_difference_l3032_303218


namespace NUMINAMATH_CALUDE_exists_n_pow_half_n_eq_twelve_l3032_303266

theorem exists_n_pow_half_n_eq_twelve :
  ∃ n : ℝ, n > 0 ∧ n^(n/2) = 12 :=
by sorry

end NUMINAMATH_CALUDE_exists_n_pow_half_n_eq_twelve_l3032_303266


namespace NUMINAMATH_CALUDE_two_face_painted_count_l3032_303294

/-- Represents a 3x3x3 cube made up of smaller cubes --/
structure Cube3x3x3 where
  /-- The total number of smaller cubes --/
  total_cubes : Nat
  /-- All outer faces of the large cube are painted --/
  outer_faces_painted : Bool

/-- Counts the number of smaller cubes painted on exactly two faces --/
def count_two_face_painted (c : Cube3x3x3) : Nat :=
  12

/-- Theorem stating that in a 3x3x3 painted cube, 12 smaller cubes are painted on exactly two faces --/
theorem two_face_painted_count (c : Cube3x3x3) 
    (h1 : c.total_cubes = 27) 
    (h2 : c.outer_faces_painted = true) : 
  count_two_face_painted c = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_face_painted_count_l3032_303294


namespace NUMINAMATH_CALUDE_T1_T2_T3_l3032_303200

-- Define the types for pib and maa
variable (Pib Maa : Type)

-- Define the belongs_to relation
variable (belongs_to : Maa → Pib → Prop)

-- P1: Every pib is a collection of maas
axiom P1 : ∀ p : Pib, ∃ m : Maa, belongs_to m p

-- P2: Any two distinct pibs have one and only one maa in common
axiom P2 : ∀ p1 p2 : Pib, p1 ≠ p2 → ∃! m : Maa, belongs_to m p1 ∧ belongs_to m p2

-- P3: Every maa belongs to two and only two pibs
axiom P3 : ∀ m : Maa, ∃! p1 p2 : Pib, p1 ≠ p2 ∧ belongs_to m p1 ∧ belongs_to m p2

-- P4: There are exactly four pibs
axiom P4 : ∃! (a b c d : Pib), ∀ p : Pib, p = a ∨ p = b ∨ p = c ∨ p = d

-- T1: There are exactly six maas
theorem T1 : ∃! (a b c d e f : Maa), ∀ m : Maa, m = a ∨ m = b ∨ m = c ∨ m = d ∨ m = e ∨ m = f :=
sorry

-- T2: There are exactly three maas in each pib
theorem T2 : ∀ p : Pib, ∃! (a b c : Maa), (∀ m : Maa, belongs_to m p ↔ (m = a ∨ m = b ∨ m = c)) :=
sorry

-- T3: For each maa there is exactly one other maa not in the same pib with it
theorem T3 : ∀ m1 : Maa, ∃! m2 : Maa, m1 ≠ m2 ∧ ∀ p : Pib, ¬(belongs_to m1 p ∧ belongs_to m2 p) :=
sorry

end NUMINAMATH_CALUDE_T1_T2_T3_l3032_303200


namespace NUMINAMATH_CALUDE_unsatisfactory_tests_l3032_303216

theorem unsatisfactory_tests (n : ℕ) (k : ℕ) : 
  n < 50 →
  n % 7 = 0 →
  n % 3 = 0 →
  n % 2 = 0 →
  n / 7 + n / 3 + n / 2 + k = n →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_unsatisfactory_tests_l3032_303216


namespace NUMINAMATH_CALUDE_solve_for_P_l3032_303290

theorem solve_for_P : ∃ P : ℝ, (P ^ 3) ^ (1/2) = 9 * (81 ^ (1/6)) → P = 3 ^ (16/9) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_P_l3032_303290


namespace NUMINAMATH_CALUDE_kddk_divisible_by_7_l3032_303254

/-- Represents a base-6 digit -/
def Base6Digit : Type := { n : ℕ // n < 6 }

/-- Converts a base-6 number of the form kddk to base 10 -/
def toBase10 (k d : Base6Digit) : ℕ :=
  217 * k.val + 42 * d.val

theorem kddk_divisible_by_7 (k d : Base6Digit) :
  7 ∣ toBase10 k d ↔ k = d :=
sorry

end NUMINAMATH_CALUDE_kddk_divisible_by_7_l3032_303254


namespace NUMINAMATH_CALUDE_inequality_solution_l3032_303201

theorem inequality_solution (y : ℝ) : 
  (1 / (y * (y + 2)) - 1 / ((y + 2) * (y + 4)) < 1 / 4) ↔ 
  (y < -4 ∨ (-2 < y ∧ y < 0) ∨ 1 < y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3032_303201


namespace NUMINAMATH_CALUDE_total_games_attended_l3032_303277

def games_this_month : ℕ := 11
def games_last_month : ℕ := 17
def games_next_month : ℕ := 16

theorem total_games_attended : games_this_month + games_last_month + games_next_month = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_games_attended_l3032_303277


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3032_303252

theorem inequality_solution_set : 
  {x : ℝ | x^2 + x - 2 > 0} = {x : ℝ | x < -2 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3032_303252


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l3032_303227

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define our circle
def our_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- Define what it means for two circles to be tangent
def tangent (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ x y : ℝ, f x y ∧ g x y ∧ 
  ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
    ((x' - x)^2 + (y' - y)^2 < δ^2) → 
    (f x' y' ↔ g x' y') → (x' = x ∧ y' = y)

theorem circle_satisfies_conditions : 
  our_circle 3 1 ∧ 
  our_circle 1 1 ∧ 
  tangent our_circle C1 := by sorry

end NUMINAMATH_CALUDE_circle_satisfies_conditions_l3032_303227


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3032_303249

/-- Given a and b are real numbers satisfying a + bi = (1 + i)i^3, prove that a = 1 and b = -1 -/
theorem complex_equation_solution (a b : ℝ) (h : (↑a + ↑b * I) = (1 + I) * I^3) : a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3032_303249


namespace NUMINAMATH_CALUDE_cuboid_reduction_impossibility_l3032_303286

theorem cuboid_reduction_impossibility (a b c a' b' c' : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0)
  (haa' : a ≥ a') (hbb' : b ≥ b') (hcc' : c ≥ c') :
  ¬(a' * b' * c' = (1/2) * a * b * c ∧ 
    2 * (a' * b' + b' * c' + c' * a') = 2 * (a * b + b * c + c * a)) := by
  sorry

end NUMINAMATH_CALUDE_cuboid_reduction_impossibility_l3032_303286


namespace NUMINAMATH_CALUDE_least_number_with_remainder_five_l3032_303225

def is_valid_number (n : ℕ) : Prop :=
  ∃ (S : Set ℕ), 15 ∈ S ∧ ∀ m ∈ S, m > 0 ∧ n % m = 5

theorem least_number_with_remainder_five :
  is_valid_number 125 ∧ ∀ k < 125, ¬(is_valid_number k) :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_five_l3032_303225


namespace NUMINAMATH_CALUDE_sin_750_degrees_l3032_303280

theorem sin_750_degrees : Real.sin (750 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_750_degrees_l3032_303280


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l3032_303260

-- Define the function
def f (x : ℝ) : ℝ := -x^3 - x^2 + 2

-- Theorem statement
theorem f_has_max_and_min :
  (∃ x_max : ℝ, ∀ x : ℝ, f x ≤ f x_max) ∧
  (∃ x_min : ℝ, ∀ x : ℝ, f x_min ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l3032_303260


namespace NUMINAMATH_CALUDE_max_sum_abs_on_unit_sphere_l3032_303297

theorem max_sum_abs_on_unit_sphere :
  ∃ (M : ℝ), M = 2 ∧
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |x| + |y| + |z| ≤ M) ∧
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ |x| + |y| + |z| = M) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_abs_on_unit_sphere_l3032_303297


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3032_303220

/-- The function f(x) = x³ - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 0)

/-- Theorem: The equation of the tangent line to f(x) at (1, 0) is 2x - y - 2 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | 2 * x - y - 2 = 0} ↔
    y - f point.1 = f' point.1 * (x - point.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3032_303220


namespace NUMINAMATH_CALUDE_marble_count_l3032_303273

theorem marble_count (red_marbles : ℕ) (green_marbles : ℕ) (yellow_marbles : ℕ) (total_marbles : ℕ) :
  red_marbles = 20 →
  green_marbles = 3 * red_marbles →
  yellow_marbles = (20 * green_marbles) / 100 →
  total_marbles = green_marbles + 3 * green_marbles →
  total_marbles - red_marbles - green_marbles - yellow_marbles = 148 :=
by sorry

end NUMINAMATH_CALUDE_marble_count_l3032_303273


namespace NUMINAMATH_CALUDE_stock_selection_probabilities_l3032_303219

/-- The number of stocks available for purchase -/
def num_stocks : ℕ := 10

/-- The number of people buying stocks -/
def num_people : ℕ := 3

/-- The probability of all people selecting the same stock -/
def prob_all_same : ℚ := 1 / 100

/-- The probability of at least two people selecting the same stock -/
def prob_at_least_two_same : ℚ := 7 / 25

/-- Theorem stating the probabilities for the stock selection problem -/
theorem stock_selection_probabilities :
  (prob_all_same = 1 / num_stocks ^ (num_people - 1)) ∧
  (prob_at_least_two_same = 
    (1 / num_stocks ^ (num_people - 1)) + 
    (num_stocks * (num_people.choose 2) * (1 / num_stocks ^ 2) * ((num_stocks - 1) / num_stocks))) :=
by sorry

end NUMINAMATH_CALUDE_stock_selection_probabilities_l3032_303219


namespace NUMINAMATH_CALUDE_female_fraction_is_25_69_l3032_303276

/-- Represents the basketball club membership data --/
structure ClubData where
  maleLastYear : ℕ
  totalIncrease : ℚ
  maleIncrease : ℚ
  femaleIncrease : ℚ

/-- Calculates the fraction of female members this year --/
def femaleFraction (data : ClubData) : ℚ :=
  let maleThisYear := data.maleLastYear * (1 + data.maleIncrease)
  let femaleLastYear := (data.maleLastYear : ℚ) * (1 + data.totalIncrease - 1) / (data.femaleIncrease - 1)
  let femaleThisYear := femaleLastYear * (1 + data.femaleIncrease)
  let totalThisYear := maleThisYear + femaleThisYear
  femaleThisYear / totalThisYear

/-- Theorem stating that given the conditions, the fraction of female members this year is 25/69 --/
theorem female_fraction_is_25_69 (data : ClubData) 
  (h1 : data.maleLastYear = 30)
  (h2 : data.totalIncrease = 0.15)
  (h3 : data.maleIncrease = 0.10)
  (h4 : data.femaleIncrease = 0.25) :
  femaleFraction data = 25 / 69 := by
  sorry


end NUMINAMATH_CALUDE_female_fraction_is_25_69_l3032_303276


namespace NUMINAMATH_CALUDE_equality_of_cyclic_system_l3032_303281

theorem equality_of_cyclic_system (x y z : ℝ) 
  (eq1 : x^3 = 2*y - 1)
  (eq2 : y^3 = 2*z - 1)
  (eq3 : z^3 = 2*x - 1) :
  x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_equality_of_cyclic_system_l3032_303281


namespace NUMINAMATH_CALUDE_veggie_votes_l3032_303242

theorem veggie_votes (total_votes meat_votes : ℕ) 
  (h1 : total_votes = 672)
  (h2 : meat_votes = 335) : 
  total_votes - meat_votes = 337 := by
sorry

end NUMINAMATH_CALUDE_veggie_votes_l3032_303242


namespace NUMINAMATH_CALUDE_probability_white_and_black_l3032_303202

def total_balls : ℕ := 6
def red_balls : ℕ := 1
def white_balls : ℕ := 2
def black_balls : ℕ := 3
def drawn_balls : ℕ := 2

def favorable_outcomes : ℕ := white_balls * black_balls
def total_outcomes : ℕ := total_balls.choose drawn_balls

theorem probability_white_and_black :
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_white_and_black_l3032_303202


namespace NUMINAMATH_CALUDE_wall_length_proof_l3032_303283

def men_group1 : ℕ := 20
def men_group2 : ℕ := 86
def days : ℕ := 8
def wall_length_group2 : ℝ := 283.8

def wall_length_group1 : ℝ := 65.7

theorem wall_length_proof :
  (men_group1 * days * wall_length_group2) / (men_group2 * days) = wall_length_group1 := by
  sorry

end NUMINAMATH_CALUDE_wall_length_proof_l3032_303283


namespace NUMINAMATH_CALUDE_exists_different_reassembled_triangle_l3032_303296

/-- A triangle represented by its three vertices in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- A function that cuts a triangle into two parts -/
def cut (t : Triangle) : (Triangle × Triangle) :=
  sorry

/-- A function that reassembles two triangles into one -/
def reassemble (t1 t2 : Triangle) : Triangle :=
  sorry

/-- Theorem stating that there exists a triangle that can be cut and reassembled into a different triangle -/
theorem exists_different_reassembled_triangle :
  ∃ (t : Triangle), ∃ (t1 t2 : Triangle),
    (cut t = (t1, t2)) ∧ (reassemble t1 t2 ≠ t) := by
  sorry

end NUMINAMATH_CALUDE_exists_different_reassembled_triangle_l3032_303296


namespace NUMINAMATH_CALUDE_parabola_vertex_l3032_303217

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 2(x-1)^2 + 2 is at the point (1, 2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3032_303217


namespace NUMINAMATH_CALUDE_power_of_product_l3032_303233

theorem power_of_product (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l3032_303233


namespace NUMINAMATH_CALUDE_min_tablets_extracted_l3032_303214

/-- Represents the number of tablets of each medicine type in the box -/
structure MedicineBox where
  a : Nat
  b : Nat
  c : Nat

/-- Calculates the minimum number of tablets to extract to guarantee at least three of each type -/
def minTablets (box : MedicineBox) : Nat :=
  (box.a + box.b + box.c) - min (box.a - 3) 0 - min (box.b - 3) 0 - min (box.c - 3) 0

/-- Theorem: The minimum number of tablets to extract from the given box is 48 -/
theorem min_tablets_extracted (box : MedicineBox) 
  (ha : box.a = 20) (hb : box.b = 25) (hc : box.c = 15) : 
  minTablets box = 48 := by
  sorry

end NUMINAMATH_CALUDE_min_tablets_extracted_l3032_303214


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l3032_303293

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|5 * x₁| + 2 = 47 ∧ |5 * x₂| + 2 = 47) ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -81) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l3032_303293


namespace NUMINAMATH_CALUDE_rotate_A_180_l3032_303245

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Rotates a point 180 degrees clockwise about the origin -/
def rotate180 (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- The original point A -/
def A : Point := { x := -4, y := 1 }

/-- The expected result after rotation -/
def A_rotated : Point := { x := 4, y := -1 }

/-- Theorem stating that rotating A 180 degrees clockwise about the origin results in A_rotated -/
theorem rotate_A_180 : rotate180 A = A_rotated := by sorry

end NUMINAMATH_CALUDE_rotate_A_180_l3032_303245


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3032_303272

theorem angle_measure_proof (x : ℝ) : 
  (90 - x + 40 = (180 - x) / 2) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3032_303272


namespace NUMINAMATH_CALUDE_course_choice_related_to_gender_l3032_303248

-- Define the contingency table
def contingency_table := (40, 10, 30, 20)

-- Define the total number of students
def total_students : Nat := 100

-- Define the critical value for α = 0.05
def critical_value : Float := 3.841

-- Function to calculate χ²
def calculate_chi_square (a b c d : Nat) : Float :=
  let n := a + b + c + d
  let numerator := n * (a * d - b * c) ^ 2
  let denominator := (a + b) * (c + d) * (a + c) * (b + d)
  numerator.toFloat / denominator.toFloat

-- Theorem statement
theorem course_choice_related_to_gender (a b c d : Nat) 
  (h1 : (a, b, c, d) = contingency_table) 
  (h2 : a + b + c + d = total_students) : 
  calculate_chi_square a b c d > critical_value :=
by
  sorry


end NUMINAMATH_CALUDE_course_choice_related_to_gender_l3032_303248


namespace NUMINAMATH_CALUDE_two_solutions_set_equiv_l3032_303291

/-- The set of values for 'a' that satisfy the conditions for two distinct solutions -/
def TwoSolutionsSet : Set ℝ :=
  {a | 9 * (a - 2) > 0 ∧ 
       a > 0 ∧ 
       a^2 - 9*a + 18 > 0 ∧
       a ≠ 11 ∧
       ∃ (x y : ℝ), x ≠ y ∧ x = a + 3 * Real.sqrt (a - 2) ∧ y = a - 3 * Real.sqrt (a - 2)}

/-- The theorem stating the equivalence of the solution set -/
theorem two_solutions_set_equiv :
  TwoSolutionsSet = {a | (2 < a ∧ a < 3) ∨ (6 < a ∧ a < 11) ∨ (11 < a)} :=
by sorry

end NUMINAMATH_CALUDE_two_solutions_set_equiv_l3032_303291


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3032_303285

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3032_303285


namespace NUMINAMATH_CALUDE_orange_sales_theorem_l3032_303264

def planned_daily_sales : ℕ := 10
def deviations : List ℤ := [4, -3, -5, 7, -8, 21, -6]
def selling_price : ℕ := 80
def shipping_fee : ℕ := 7

theorem orange_sales_theorem :
  let first_five_days_sales := planned_daily_sales * 5 + (deviations.take 5).sum
  let total_deviation := deviations.sum
  let total_sales := planned_daily_sales * 7 + total_deviation
  let total_earnings := total_sales * selling_price - total_sales * shipping_fee
  (first_five_days_sales = 45) ∧
  (total_deviation > 0) ∧
  (total_earnings = 5840) := by
  sorry

end NUMINAMATH_CALUDE_orange_sales_theorem_l3032_303264


namespace NUMINAMATH_CALUDE_function_properties_l3032_303271

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 3|

-- State the theorem
theorem function_properties :
  (∀ x : ℝ, f x ≤ 1 ↔ 1 ≤ x ∧ x ≤ 2) ∧
  (∀ a b c x : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    f x - 2 * |x + 3| ≤ 1/a + 1/b + 1/c) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3032_303271


namespace NUMINAMATH_CALUDE_custom_operation_result_l3032_303226

def custom_operation (A B : Set ℕ) : Set ℕ :=
  {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y * (x + y)}

theorem custom_operation_result :
  let A : Set ℕ := {0, 1}
  let B : Set ℕ := {2, 3}
  custom_operation A B = {0, 6, 12} := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_result_l3032_303226


namespace NUMINAMATH_CALUDE_smallest_N_proof_l3032_303282

/-- The smallest number of pies per batch that satisfies the conditions --/
def smallest_N : ℕ := 80

/-- The number of batches of pies --/
def num_batches : ℕ := 21

/-- The number of pies per tray --/
def pies_per_tray : ℕ := 70

theorem smallest_N_proof :
  (∀ N : ℕ, N > 70 → (num_batches * N) % pies_per_tray = 0 → N ≥ smallest_N) ∧
  smallest_N > 70 ∧
  (num_batches * smallest_N) % pies_per_tray = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_N_proof_l3032_303282


namespace NUMINAMATH_CALUDE_bill_difference_l3032_303253

theorem bill_difference (christine_tip : ℝ) (christine_percent : ℝ)
  (alex_tip : ℝ) (alex_percent : ℝ) :
  christine_tip = 3 →
  christine_percent = 15 →
  alex_tip = 4 →
  alex_percent = 10 →
  christine_tip = (christine_percent / 100) * christine_bill →
  alex_tip = (alex_percent / 100) * alex_bill →
  alex_bill - christine_bill = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_bill_difference_l3032_303253


namespace NUMINAMATH_CALUDE_perimeter_of_new_arrangement_l3032_303284

/-- Represents a square arrangement -/
structure SquareArrangement where
  rows : ℕ
  columns : ℕ

/-- Calculates the perimeter of a square arrangement -/
def perimeter (arrangement : SquareArrangement) : ℕ :=
  2 * (arrangement.rows + arrangement.columns)

/-- The original square arrangement -/
def original : SquareArrangement :=
  { rows := 3, columns := 5 }

/-- The new square arrangement with an additional row -/
def new : SquareArrangement :=
  { rows := original.rows + 1, columns := original.columns }

theorem perimeter_of_new_arrangement :
  perimeter new = 37 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_new_arrangement_l3032_303284


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l3032_303206

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (avg_children_all : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : avg_children_all = 3)
  (h3 : childless_families = 3)
  : (total_families * avg_children_all) / (total_families - childless_families) = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l3032_303206


namespace NUMINAMATH_CALUDE_multiples_of_seven_l3032_303239

theorem multiples_of_seven (x : ℕ) : 
  (∃ n : ℕ, n = 47 ∧ 
   (∀ k : ℕ, x ≤ 7 * k ∧ 7 * k ≤ 343 → k ≤ n) ∧
   (∀ k : ℕ, k ≤ n → x ≤ 7 * k ∧ 7 * k ≤ 343)) →
  x = 14 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_seven_l3032_303239


namespace NUMINAMATH_CALUDE_franks_remaining_money_l3032_303236

/-- 
Given:
- Frank initially had $600.
- He spent 1/5 of his money on groceries.
- He then spent 1/4 of the remaining money on a magazine.

Prove that Frank has $360 left after buying groceries and the magazine.
-/
theorem franks_remaining_money (initial_amount : ℚ) 
  (h1 : initial_amount = 600)
  (grocery_fraction : ℚ) (h2 : grocery_fraction = 1/5)
  (magazine_fraction : ℚ) (h3 : magazine_fraction = 1/4) :
  let remaining_after_groceries := initial_amount - grocery_fraction * initial_amount
  let remaining_after_magazine := remaining_after_groceries - magazine_fraction * remaining_after_groceries
  remaining_after_magazine = 360 := by
sorry

end NUMINAMATH_CALUDE_franks_remaining_money_l3032_303236


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3032_303204

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ x^2 - x
  ∃ (x₁ x₂ : ℝ), (f x₁ = 0 ∧ f x₂ = 0) ∧ x₁ = 0 ∧ x₂ = 1 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3032_303204


namespace NUMINAMATH_CALUDE_nine_times_polygon_properties_l3032_303256

/-- A polygon with interior angles 9 times the exterior angles -/
structure NineTimesPolygon where
  n : ℕ -- number of sides
  interior_angles : Fin n → ℝ
  exterior_angles : Fin n → ℝ
  h_positive : ∀ i, interior_angles i > 0 ∧ exterior_angles i > 0
  h_relation : ∀ i, interior_angles i = 9 * exterior_angles i
  h_exterior_sum : (Finset.univ.sum exterior_angles) = 360

theorem nine_times_polygon_properties (Q : NineTimesPolygon) :
  (Finset.univ.sum Q.interior_angles = 3240) ∧
  (∃ (i j : Fin Q.n), Q.interior_angles i ≠ Q.interior_angles j ∨ Q.interior_angles i = Q.interior_angles j) :=
by sorry

end NUMINAMATH_CALUDE_nine_times_polygon_properties_l3032_303256


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3032_303222

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → perp α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3032_303222


namespace NUMINAMATH_CALUDE_f_properties_l3032_303251

noncomputable def f (x : ℝ) : ℝ := x^2 + x - Real.log x

theorem f_properties :
  (∀ x > 0, f x = x^2 + x - Real.log x) →
  (∃ m b : ℝ, ∀ x : ℝ, (x = 1 → f x = m * x + b) ∧ m = 2 ∧ b = 0) ∧
  (∃ x_min : ℝ, x_min > 0 ∧ ∀ x > 0, f x ≥ f x_min ∧ f x_min = 3/4 + Real.log 2) ∧
  (¬ ∃ x_max : ℝ, x_max > 0 ∧ ∀ x > 0, f x ≤ f x_max) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3032_303251


namespace NUMINAMATH_CALUDE_perimeter_approx_l3032_303262

/-- A right triangle with area 150 and one leg 15 units longer than the other -/
structure RightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  area_eq : (1/2) * shorter_leg * longer_leg = 150
  leg_diff : longer_leg = shorter_leg + 15
  pythagorean : shorter_leg^2 + longer_leg^2 = hypotenuse^2

/-- The perimeter of the triangle -/
def perimeter (t : RightTriangle) : ℝ :=
  t.shorter_leg + t.longer_leg + t.hypotenuse

/-- Theorem stating that the perimeter is approximately 66.47 -/
theorem perimeter_approx (t : RightTriangle) :
  abs (perimeter t - 66.47) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_approx_l3032_303262


namespace NUMINAMATH_CALUDE_function_equality_implies_k_range_l3032_303299

open Real

/-- Given a function f(x) = 1 + ln x + kx where k is a real number,
    if there exists a positive x such that e^x = f(x)/x, then k ≥ 1 -/
theorem function_equality_implies_k_range (k : ℝ) :
  (∃ x > 0, exp x = (1 + log x + k * x) / x) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_k_range_l3032_303299


namespace NUMINAMATH_CALUDE_largest_divisor_of_sequence_l3032_303229

theorem largest_divisor_of_sequence (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k = 105 ∧ 
  (∀ m : ℕ, m > k → ¬(m ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13))) ∧
  (k ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_sequence_l3032_303229


namespace NUMINAMATH_CALUDE_complex_fraction_sum_zero_l3032_303295

theorem complex_fraction_sum_zero : 
  let i : ℂ := Complex.I
  ((1 + i) / (1 - i)) ^ 2017 + ((1 - i) / (1 + i)) ^ 2017 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_zero_l3032_303295


namespace NUMINAMATH_CALUDE_cow_plus_cow_equals_milk_l3032_303213

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents an assignment of digits to letters -/
structure LetterAssignment where
  C : Digit
  O : Digit
  W : Digit
  M : Digit
  I : Digit
  L : Digit
  K : Digit
  all_different : C ≠ O ∧ C ≠ W ∧ C ≠ M ∧ C ≠ I ∧ C ≠ L ∧ C ≠ K ∧
                  O ≠ W ∧ O ≠ M ∧ O ≠ I ∧ O ≠ L ∧ O ≠ K ∧
                  W ≠ M ∧ W ≠ I ∧ W ≠ L ∧ W ≠ K ∧
                  M ≠ I ∧ M ≠ L ∧ M ≠ K ∧
                  I ≠ L ∧ I ≠ K ∧
                  L ≠ K

/-- Converts a LetterAssignment to the numeric value of COW -/
def cow_value (assignment : LetterAssignment) : ℕ :=
  100 * assignment.C.val + 10 * assignment.O.val + assignment.W.val

/-- Converts a LetterAssignment to the numeric value of MILK -/
def milk_value (assignment : LetterAssignment) : ℕ :=
  1000 * assignment.M.val + 100 * assignment.I.val + 10 * assignment.L.val + assignment.K.val

/-- The main theorem stating that there are exactly three solutions to the puzzle -/
theorem cow_plus_cow_equals_milk :
  ∃! (solutions : Finset LetterAssignment),
    solutions.card = 3 ∧
    (∀ assignment ∈ solutions, 2 * cow_value assignment = milk_value assignment) :=
sorry

end NUMINAMATH_CALUDE_cow_plus_cow_equals_milk_l3032_303213


namespace NUMINAMATH_CALUDE_rhombus_diagonals_not_always_equal_l3032_303244

/-- A rhombus is a quadrilateral with all four sides of equal length -/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The diagonals of a rhombus -/
def diagonals (r : Rhombus) : ℝ × ℝ :=
  sorry

/-- Theorem: The diagonals of a rhombus are not always equal -/
theorem rhombus_diagonals_not_always_equal :
  ¬ (∀ r : Rhombus, (diagonals r).1 = (diagonals r).2) :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_not_always_equal_l3032_303244


namespace NUMINAMATH_CALUDE_equation_solution_l3032_303268

theorem equation_solution : ∃ X : ℝ, 
  1.5 * ((3.6 * X * 2.50) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002 ∧ 
  X = 0.4800000000000001 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3032_303268


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3032_303274

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (θ : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (h3 : θ = Real.pi / 3) -- 60° in radians
  (h4 : c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ)) -- Law of Cosines
  : c = Real.sqrt 109 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3032_303274


namespace NUMINAMATH_CALUDE_gcf_is_correct_l3032_303258

def term1 (x y : ℕ) : ℕ := 9 * x^3 * y^2
def term2 (x y : ℕ) : ℕ := 12 * x^2 * y^3

def gcf (x y : ℕ) : ℕ := 3 * x^2 * y^2

theorem gcf_is_correct (x y : ℕ) :
  (gcf x y) ∣ (term1 x y) ∧ (gcf x y) ∣ (term2 x y) ∧
  ∀ (d : ℕ), d ∣ (term1 x y) ∧ d ∣ (term2 x y) → d ∣ (gcf x y) :=
sorry

end NUMINAMATH_CALUDE_gcf_is_correct_l3032_303258


namespace NUMINAMATH_CALUDE_area_of_triangle_AKF_l3032_303232

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Define points A, B, K, and F
variable (A B K : ℝ × ℝ)
def F : ℝ × ℝ := focus

-- State that A is on the parabola
axiom A_on_parabola : parabola A.1 A.2

-- State that B is on the directrix
axiom B_on_directrix : directrix B.1

-- State that K is on the directrix
axiom K_on_directrix : directrix K.1

-- State that A, F, and B are collinear
axiom A_F_B_collinear : ∃ (t : ℝ), A = F + t • (B - F) ∨ B = F + t • (A - F)

-- State that AK is perpendicular to the directrix
axiom AK_perp_directrix : (A.1 - K.1) * 0 + (A.2 - K.2) * 1 = 0

-- State that |AF| = |BF|
axiom AF_eq_BF : (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2

-- Theorem to prove
theorem area_of_triangle_AKF : 
  (1/2) * abs ((A.1 - F.1) * (K.2 - F.2) - (K.1 - F.1) * (A.2 - F.2)) = 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_AKF_l3032_303232


namespace NUMINAMATH_CALUDE_litter_patrol_collection_l3032_303203

theorem litter_patrol_collection (glass_bottles : ℕ) (aluminum_cans : ℕ) : 
  glass_bottles = 10 → aluminum_cans = 8 → glass_bottles + aluminum_cans = 18 := by
  sorry

end NUMINAMATH_CALUDE_litter_patrol_collection_l3032_303203


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_four_cube_l3032_303247

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  total_cubes : ℕ
  painted_corners : ℕ
  painted_edges : ℕ

/-- Properties of a 4x4x4 cube with painted corners -/
def four_cube : Cube 4 :=
  { side_length := 4
  , total_cubes := 64
  , painted_corners := 8
  , painted_edges := 12 }

/-- The number of unpainted cubes in a cube with painted corners -/
def unpainted_cubes (c : Cube n) : ℕ :=
  c.total_cubes - (c.painted_corners + c.painted_edges)

/-- Theorem: The number of unpainted cubes in a 4x4x4 cube with painted corners is 44 -/
theorem unpainted_cubes_in_four_cube :
  unpainted_cubes four_cube = 44 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_four_cube_l3032_303247
