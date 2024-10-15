import Mathlib

namespace NUMINAMATH_CALUDE_proportional_y_value_l2388_238847

/-- Given that y is directly proportional to x+1 and y=4 when x=1, 
    prove that y=6 when x=2 -/
theorem proportional_y_value (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = k * (x + 1)) →  -- y is directly proportional to x+1
  (4 = k * (1 + 1)) →                    -- when x=1, y=4
  (6 = k * (2 + 1)) :=                   -- prove y=6 when x=2
by
  sorry


end NUMINAMATH_CALUDE_proportional_y_value_l2388_238847


namespace NUMINAMATH_CALUDE_algebra_test_average_l2388_238856

theorem algebra_test_average : ∀ (male_count female_count : ℕ) 
  (male_avg female_avg overall_avg : ℚ),
  male_count = 8 →
  female_count = 28 →
  male_avg = 83 →
  female_avg = 92 →
  overall_avg = (male_count * male_avg + female_count * female_avg) / (male_count + female_count) →
  overall_avg = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_algebra_test_average_l2388_238856


namespace NUMINAMATH_CALUDE_min_distance_parabola_circle_l2388_238843

theorem min_distance_parabola_circle : 
  let parabola := {P : ℝ × ℝ | P.2^2 = P.1}
  let circle := {Q : ℝ × ℝ | (Q.1 - 3)^2 + Q.2^2 = 1}
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ parabola ∧ Q ∈ circle ∧
    ∀ (P' : ℝ × ℝ) (Q' : ℝ × ℝ), P' ∈ parabola → Q' ∈ circle →
      Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ∧
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = (Real.sqrt 11 - 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parabola_circle_l2388_238843


namespace NUMINAMATH_CALUDE_chessboard_coloring_theorem_l2388_238867

/-- Represents a color (Red or Blue) -/
inductive Color
| Red
| Blue

/-- Represents a 4 x 7 chessboard coloring -/
def Coloring := Fin 4 → Fin 7 → Color

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  top_left : Fin 4 × Fin 7
  bottom_right : Fin 4 × Fin 7

/-- Check if a rectangle has all corners of the same color -/
def has_same_color_corners (c : Coloring) (r : Rectangle) : Prop :=
  let (t, l) := r.top_left
  let (b, r) := r.bottom_right
  c t l = c t r ∧ c t l = c b l ∧ c t l = c b r

/-- Main theorem: For any coloring of a 4 x 7 chessboard, 
    there exists a rectangle with four corners of the same color -/
theorem chessboard_coloring_theorem :
  ∀ (c : Coloring), ∃ (r : Rectangle), has_same_color_corners c r :=
sorry

end NUMINAMATH_CALUDE_chessboard_coloring_theorem_l2388_238867


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2388_238862

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (p a b m n : ℝ) (hp : p > 0) (ha : a > 0) (hb : b > 0) :
  let parabola := fun x y => y^2 = 2*p*x
  let hyperbola := fun x y => x^2/a^2 - y^2/b^2 = 1
  let focus : ℝ × ℝ := (p/2, 0)
  let A : ℝ × ℝ := (p/2, p)
  let B : ℝ × ℝ := (p/2, -p)
  let M : ℝ × ℝ := (p/2, b^2/a)
  (∀ x y, parabola x y → hyperbola x y → (x = p/2 ∧ y = 0)) →
  (m + n = 1) →
  (m - n = b^2/(a*p)) →
  (m * n = 1/8) →
  let e := c/a
  let c := Real.sqrt (a^2 + b^2)
  e = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2388_238862


namespace NUMINAMATH_CALUDE_sum_is_composite_l2388_238838

theorem sum_is_composite (a b c d : ℕ) (h : a^2 + b^2 = c^2 + d^2) :
  ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ a + b + c + d = k * m :=
by sorry

end NUMINAMATH_CALUDE_sum_is_composite_l2388_238838


namespace NUMINAMATH_CALUDE_trader_theorem_l2388_238858

def trader_problem (profit goal donations : ℕ) : Prop :=
  let half_profit := profit / 2
  let total_available := half_profit + donations
  total_available - goal = 180

theorem trader_theorem : trader_problem 960 610 310 := by
  sorry

end NUMINAMATH_CALUDE_trader_theorem_l2388_238858


namespace NUMINAMATH_CALUDE_slope_range_l2388_238865

/-- A line passing through (0,2) that intersects the circle (x-2)^2 + (y-2)^2 = 1 -/
structure IntersectingLine where
  slope : ℝ
  passes_through_origin : (0 : ℝ) = slope * 0 + 2
  intersects_circle : ∃ (x y : ℝ), y = slope * x + 2 ∧ (x - 2)^2 + (y - 2)^2 = 1

/-- The theorem stating the range of possible slopes for the intersecting line -/
theorem slope_range (l : IntersectingLine) : 
  l.slope ∈ Set.Icc (-Real.sqrt 3 / 3) (Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_slope_range_l2388_238865


namespace NUMINAMATH_CALUDE_book_organization_time_l2388_238864

theorem book_organization_time (time_A time_B joint_time : ℝ) 
  (h1 : time_A = 6)
  (h2 : time_B = 8)
  (h3 : joint_time = 2)
  (h4 : joint_time * (1 / time_A + 1 / time_B) + 1 / time_A * remaining_time = 1) :
  remaining_time = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_book_organization_time_l2388_238864


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2388_238809

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = x + f y) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2388_238809


namespace NUMINAMATH_CALUDE_instantaneous_rate_of_change_at_e_l2388_238855

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem instantaneous_rate_of_change_at_e :
  deriv f e = 0 := by sorry

end NUMINAMATH_CALUDE_instantaneous_rate_of_change_at_e_l2388_238855


namespace NUMINAMATH_CALUDE_symmetric_point_example_l2388_238859

/-- Given a point (x, y) in a 2D coordinate system, this function returns the point that is symmetric to (x, y) with respect to the origin. -/
def symmetricPointOrigin (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- Theorem stating that the point symmetric to (-2, 5) with respect to the origin is (2, -5). -/
theorem symmetric_point_example : symmetricPointOrigin (-2) 5 = (2, -5) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_example_l2388_238859


namespace NUMINAMATH_CALUDE_f_two_zeros_iff_a_in_range_l2388_238850

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 - (a + 2) * x

def has_exactly_two_zeros (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
  ∀ z : ℝ, z > 0 → f a z = 0 → (z = x ∨ z = y)

theorem f_two_zeros_iff_a_in_range :
  ∀ a : ℝ, has_exactly_two_zeros a ↔ -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_f_two_zeros_iff_a_in_range_l2388_238850


namespace NUMINAMATH_CALUDE_find_n_l2388_238889

theorem find_n : ∃ n : ℤ, (7 : ℝ) ^ (2 * n) = (1 / 49) ^ (n - 12) ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_find_n_l2388_238889


namespace NUMINAMATH_CALUDE_exist_prime_sum_30_l2388_238881

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- State the theorem
theorem exist_prime_sum_30 : ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p + q = 30 := by
  sorry

end NUMINAMATH_CALUDE_exist_prime_sum_30_l2388_238881


namespace NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l2388_238834

-- Define the curve C
def CurveC (a b x y : ℝ) : Prop := x^2 / a + y^2 / b = 1

-- Define what it means for C to be an ellipse with foci on the x-axis
def IsEllipseOnXAxis (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ ∀ x y, CurveC a b x y → x^2 + y^2 < a^2

-- Theorem stating that a > b is a necessary but not sufficient condition
theorem a_gt_b_necessary_not_sufficient :
  (∀ a b : ℝ, IsEllipseOnXAxis a b → a > b) ∧
  ¬(∀ a b : ℝ, a > b → IsEllipseOnXAxis a b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l2388_238834


namespace NUMINAMATH_CALUDE_remainder_theorem_l2388_238878

theorem remainder_theorem (n : ℕ) 
  (h1 : n % 22 = 7) 
  (h2 : n % 33 = 18) : 
  n % 66 = 51 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2388_238878


namespace NUMINAMATH_CALUDE_joan_has_16_seashells_l2388_238829

/-- The number of seashells Joan has now, given that she found 79 and gave away 63. -/
def joans_remaining_seashells (found : ℕ) (gave_away : ℕ) : ℕ :=
  found - gave_away

/-- Theorem stating that Joan has 16 seashells now. -/
theorem joan_has_16_seashells : 
  joans_remaining_seashells 79 63 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_16_seashells_l2388_238829


namespace NUMINAMATH_CALUDE_ant_travel_distance_l2388_238811

/-- The number of nodes on the bamboo -/
def num_nodes : ℕ := 30

/-- The height of the first node in feet -/
def first_node_height : ℝ := 0.5

/-- The increase in height between consecutive nodes in feet -/
def node_height_diff : ℝ := 0.03

/-- The circumference of the first circle in feet -/
def first_circle_circumference : ℝ := 1.3

/-- The decrease in circumference between consecutive circles in feet -/
def circle_circumference_diff : ℝ := 0.013

/-- The total distance traveled by the ant in feet -/
def total_distance : ℝ := 61.395

/-- Theorem stating the total distance traveled by the ant -/
theorem ant_travel_distance :
  (num_nodes : ℝ) * first_node_height + 
  (num_nodes * (num_nodes - 1) / 2) * node_height_diff +
  (num_nodes : ℝ) * first_circle_circumference - 
  (num_nodes * (num_nodes - 1) / 2) * circle_circumference_diff = 
  total_distance :=
sorry

end NUMINAMATH_CALUDE_ant_travel_distance_l2388_238811


namespace NUMINAMATH_CALUDE_find_number_l2388_238807

theorem find_number (x : ℝ) : 6 + (1/2) * (1/3) * (1/5) * x = (1/15) * x → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2388_238807


namespace NUMINAMATH_CALUDE_min_value_expression_l2388_238860

theorem min_value_expression (a b : ℝ) (h : a * b > 0) :
  (a^4 + 4*b^4 + 1) / (a*b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2388_238860


namespace NUMINAMATH_CALUDE_extraneous_root_implies_m_value_l2388_238870

/-- Given a fractional equation (x - 3) / (x - 1) = m / (x - 1),
    if x = 1 is an extraneous root, then m = -2 -/
theorem extraneous_root_implies_m_value :
  ∀ (x m : ℝ), 
    (x - 3) / (x - 1) = m / (x - 1) →
    (1 : ℝ) ≠ 1 →  -- This represents that x = 1 is an extraneous root
    m = -2 := by
  sorry


end NUMINAMATH_CALUDE_extraneous_root_implies_m_value_l2388_238870


namespace NUMINAMATH_CALUDE_distance_between_foci_l2388_238825

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25

-- Define the foci
def focus1 : ℝ × ℝ := (4, 5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem statement
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 2 * Real.sqrt 29 := by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l2388_238825


namespace NUMINAMATH_CALUDE_order_total_price_l2388_238814

/-- Calculate the total price of an order given the number of ice-cream bars, number of sundaes,
    price per ice-cream bar, and price per sundae. -/
def total_price (ice_cream_bars : ℕ) (sundaes : ℕ) (price_ice_cream : ℚ) (price_sundae : ℚ) : ℚ :=
  ice_cream_bars * price_ice_cream + sundaes * price_sundae

/-- Theorem stating that the total price of the order is $200 given the specific quantities and prices. -/
theorem order_total_price :
  total_price 225 125 (60/100) (52/100) = 200 := by
  sorry

end NUMINAMATH_CALUDE_order_total_price_l2388_238814


namespace NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l2388_238800

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_increasing_on_positive_reals :
  ∀ (x₁ x₂ : ℝ), 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l2388_238800


namespace NUMINAMATH_CALUDE_one_fourths_in_two_thirds_l2388_238876

theorem one_fourths_in_two_thirds : (2 : ℚ) / 3 / ((1 : ℚ) / 4) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_one_fourths_in_two_thirds_l2388_238876


namespace NUMINAMATH_CALUDE_max_profit_at_70_optimal_selling_price_l2388_238842

def purchase_price : ℕ := 40
def initial_selling_price : ℕ := 50
def initial_sales_volume : ℕ := 500
def price_increment : ℕ := 1
def sales_volume_decrement : ℕ := 10

def profit (x : ℕ) : ℤ :=
  (initial_sales_volume - sales_volume_decrement * x) * (initial_selling_price + x) -
  (initial_sales_volume - sales_volume_decrement * x) * purchase_price

theorem max_profit_at_70 :
  ∀ x : ℕ, x ≤ 50 → profit x ≤ profit 20 := by sorry

theorem optimal_selling_price :
  ∃ x : ℕ, x ≤ 50 ∧ ∀ y : ℕ, y ≤ 50 → profit y ≤ profit x :=
by
  use 20
  sorry

#eval initial_selling_price + 20

end NUMINAMATH_CALUDE_max_profit_at_70_optimal_selling_price_l2388_238842


namespace NUMINAMATH_CALUDE_ellipse_equation_l2388_238885

/-- An ellipse passing through (3, 0) with eccentricity √6/3 has standard equations x²/9 + y²/3 = 1 or x²/9 + y²/27 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let e : ℝ := Real.sqrt 6 / 3
  let passes_through : Prop := x^2 + y^2 = 9 ∧ y = 0
  let equation1 : Prop := x^2 / 9 + y^2 / 3 = 1
  let equation2 : Prop := x^2 / 9 + y^2 / 27 = 1
  passes_through → (equation1 ∨ equation2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2388_238885


namespace NUMINAMATH_CALUDE_expand_expression_l2388_238891

theorem expand_expression (x : ℝ) : 25 * (3 * x - 4) = 75 * x - 100 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2388_238891


namespace NUMINAMATH_CALUDE_river_speed_is_three_l2388_238861

/-- Represents a ship with its upstream speed -/
structure Ship where
  speed : ℝ

/-- Represents the rescue scenario -/
structure RescueScenario where
  ships : List Ship
  timeToTurn : ℝ
  distanceToRescue : ℝ
  riverSpeed : ℝ

/-- Theorem: Given the conditions, the river speed is 3 km/h -/
theorem river_speed_is_three (scenario : RescueScenario) :
  scenario.ships = [Ship.mk 4, Ship.mk 6, Ship.mk 10] →
  scenario.timeToTurn = 1 →
  scenario.distanceToRescue = 6 →
  scenario.riverSpeed = 3 := by
  sorry


end NUMINAMATH_CALUDE_river_speed_is_three_l2388_238861


namespace NUMINAMATH_CALUDE_max_value_f_on_interval_l2388_238871

def f (x : ℝ) := x^3 - 3*x^2 + 2

theorem max_value_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1 : ℝ) 1 ∧ 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f x ≤ f c) ∧
  f c = 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_f_on_interval_l2388_238871


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l2388_238802

/-- Given a markup percentage and a selling price, calculates the cost price -/
def calculate_cost_price (markup_percentage : ℚ) (selling_price : ℚ) : ℚ :=
  selling_price / (1 + markup_percentage)

/-- Proves that for a 25% markup and selling price of 4800, the cost price is 3840 -/
theorem furniture_shop_cost_price :
  calculate_cost_price (25 / 100) 4800 = 3840 := by
  sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l2388_238802


namespace NUMINAMATH_CALUDE_not_lucky_1994_l2388_238874

/-- Represents a date in month/day/year format -/
structure Date where
  month : Nat
  day : Nat
  year : Nat

/-- Checks if a given date is valid -/
def isValidDate (d : Date) : Prop :=
  d.month ≥ 1 ∧ d.month ≤ 12 ∧ d.day ≥ 1 ∧ d.day ≤ 31

/-- Checks if a given year is lucky -/
def isLuckyYear (year : Nat) : Prop :=
  ∃ (d : Date), d.year = year ∧ isValidDate d ∧ (d.month * d.day = year % 100)

/-- Theorem stating that 1994 is not a lucky year -/
theorem not_lucky_1994 : ¬ isLuckyYear 1994 := by
  sorry


end NUMINAMATH_CALUDE_not_lucky_1994_l2388_238874


namespace NUMINAMATH_CALUDE_area_of_rotated_squares_l2388_238854

/-- Represents a square sheet of paper -/
structure Square :=
  (side_length : ℝ)

/-- Represents the configuration of three overlapping squares -/
structure OverlappingSquares :=
  (base : Square)
  (middle_rotation : ℝ)
  (top_rotation : ℝ)

/-- Calculates the area of the 24-sided polygon formed by the overlapping squares -/
def area_of_polygon (config : OverlappingSquares) : ℝ :=
  sorry

theorem area_of_rotated_squares :
  let config := OverlappingSquares.mk (Square.mk 8) (20 * π / 180) (45 * π / 180)
  area_of_polygon config = 192 := by
  sorry

end NUMINAMATH_CALUDE_area_of_rotated_squares_l2388_238854


namespace NUMINAMATH_CALUDE_wrench_sales_profit_l2388_238852

theorem wrench_sales_profit (selling_price : ℝ) : 
  selling_price > 0 →
  let profit_percent : ℝ := 0.25
  let loss_percent : ℝ := 0.15
  let cost_price1 : ℝ := selling_price / (1 + profit_percent)
  let cost_price2 : ℝ := selling_price / (1 - loss_percent)
  let total_cost : ℝ := cost_price1 + cost_price2
  let total_revenue : ℝ := 2 * selling_price
  let net_gain : ℝ := total_revenue - total_cost
  net_gain / selling_price = 0.028 :=
by sorry

end NUMINAMATH_CALUDE_wrench_sales_profit_l2388_238852


namespace NUMINAMATH_CALUDE_cube_root_243_equals_3_to_5_thirds_l2388_238836

theorem cube_root_243_equals_3_to_5_thirds : 
  (243 : ℝ) = 3^5 → (243 : ℝ)^(1/3) = 3^(5/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_243_equals_3_to_5_thirds_l2388_238836


namespace NUMINAMATH_CALUDE_min_participants_correct_l2388_238851

/-- Represents a participant in the race -/
inductive Participant
| Andrei
| Dima
| Lenya
| Other

/-- Represents the race results -/
def RaceResult := List Participant

/-- Checks if the race result satisfies the given conditions -/
def satisfiesConditions (result : RaceResult) : Prop :=
  let n := result.length
  ∃ (a d l : Nat),
    a + 1 + 2 * a = n ∧
    d + 1 + 3 * d = n ∧
    l + 1 + 4 * l = n ∧
    a ≠ d ∧ a ≠ l ∧ d ≠ l

/-- The minimum number of participants in the race -/
def minParticipants : Nat := 61

theorem min_participants_correct :
  ∃ (result : RaceResult),
    result.length = minParticipants ∧
    satisfiesConditions result ∧
    ∀ (result' : RaceResult),
      satisfiesConditions result' →
      result'.length ≥ minParticipants :=
sorry

end NUMINAMATH_CALUDE_min_participants_correct_l2388_238851


namespace NUMINAMATH_CALUDE_divisibility_problem_l2388_238890

theorem divisibility_problem : ∃ (a b : ℕ), 
  (7^3 ∣ a^2 + a*b + b^2) ∧ 
  ¬(7 ∣ a) ∧ 
  ¬(7 ∣ b) ∧
  a = 1 ∧ 
  b = 18 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2388_238890


namespace NUMINAMATH_CALUDE_triangle_side_length_l2388_238810

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  b = 7 →
  c = 5 →
  Real.cos (B - C) = 47 / 50 →
  a = Real.sqrt 54.4 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2388_238810


namespace NUMINAMATH_CALUDE_fish_market_customers_l2388_238839

theorem fish_market_customers (num_tuna : ℕ) (tuna_weight : ℕ) (customer_want : ℕ) (unserved : ℕ) : 
  num_tuna = 10 → 
  tuna_weight = 200 → 
  customer_want = 25 → 
  unserved = 20 → 
  (num_tuna * tuna_weight) / customer_want + unserved = 100 := by
sorry

end NUMINAMATH_CALUDE_fish_market_customers_l2388_238839


namespace NUMINAMATH_CALUDE_odd_square_plus_n_times_odd_plus_one_parity_l2388_238804

theorem odd_square_plus_n_times_odd_plus_one_parity (o n : ℤ) 
  (ho : ∃ k : ℤ, o = 2 * k + 1) :
  Odd (o^2 + n*o + 1) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_odd_square_plus_n_times_odd_plus_one_parity_l2388_238804


namespace NUMINAMATH_CALUDE_exists_left_absorbing_l2388_238806

variable {S : Type}
variable (star : S → S → S)

axiom commutative : ∀ a b : S, star a b = star b a
axiom associative : ∀ a b c : S, star (star a b) c = star a (star b c)
axiom exists_idempotent : ∃ a : S, star a a = a

theorem exists_left_absorbing : ∃ a : S, ∀ b : S, star a b = a := by
  sorry

end NUMINAMATH_CALUDE_exists_left_absorbing_l2388_238806


namespace NUMINAMATH_CALUDE_set_equality_l2388_238875

theorem set_equality (A B C : Set α) 
  (h1 : A ∪ B ⊆ C) 
  (h2 : A ∪ C ⊆ B) 
  (h3 : B ∪ C ⊆ A) : 
  A = B ∧ B = C := by
sorry

end NUMINAMATH_CALUDE_set_equality_l2388_238875


namespace NUMINAMATH_CALUDE_interior_angles_increase_l2388_238817

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: If a convex polygon with n sides has a sum of interior angles of 3240 degrees,
    then a convex polygon with n + 3 sides has a sum of interior angles of 3780 degrees. -/
theorem interior_angles_increase (n : ℕ) :
  sum_interior_angles n = 3240 → sum_interior_angles (n + 3) = 3780 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_increase_l2388_238817


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l2388_238815

/-- Given that 34 cows eat 34 bags of husk in 34 days, prove that one cow will eat one bag of husk in 34 days. -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 34 ∧ bags = 34 ∧ days = 34) :
  (1 : ℕ) * days = 34 := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l2388_238815


namespace NUMINAMATH_CALUDE_tina_money_left_is_40_l2388_238818

/-- Calculates the amount of money Tina has left after savings and expenses -/
def tina_money_left (june_savings july_savings august_savings book_expense shoe_expense : ℕ) : ℕ :=
  (june_savings + july_savings + august_savings) - (book_expense + shoe_expense)

/-- Theorem stating that Tina has $40 left given her savings and expenses -/
theorem tina_money_left_is_40 :
  tina_money_left 27 14 21 5 17 = 40 := by
  sorry

#eval tina_money_left 27 14 21 5 17

end NUMINAMATH_CALUDE_tina_money_left_is_40_l2388_238818


namespace NUMINAMATH_CALUDE_intersection_A_B_l2388_238848

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2388_238848


namespace NUMINAMATH_CALUDE_sequence_sum_equals_9972_l2388_238837

def otimes (m n : ℕ) : ℤ := m * m - n * n

def sequence_sum : ℤ :=
  otimes 2 4 - otimes 4 6 - otimes 6 8 - otimes 8 10 - otimes 10 12 - otimes 12 14 - 
  otimes 14 16 - otimes 16 18 - otimes 18 20 - otimes 20 22 - otimes 22 24 - 
  otimes 24 26 - otimes 26 28 - otimes 28 30 - otimes 30 32 - otimes 32 34 - 
  otimes 34 36 - otimes 36 38 - otimes 38 40 - otimes 40 42 - otimes 42 44 - 
  otimes 44 46 - otimes 46 48 - otimes 48 50 - otimes 50 52 - otimes 52 54 - 
  otimes 54 56 - otimes 56 58 - otimes 58 60 - otimes 60 62 - otimes 62 64 - 
  otimes 64 66 - otimes 66 68 - otimes 68 70 - otimes 70 72 - otimes 72 74 - 
  otimes 74 76 - otimes 76 78 - otimes 78 80 - otimes 80 82 - otimes 82 84 - 
  otimes 84 86 - otimes 86 88 - otimes 88 90 - otimes 90 92 - otimes 92 94 - 
  otimes 94 96 - otimes 96 98 - otimes 98 100

theorem sequence_sum_equals_9972 : sequence_sum = 9972 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_9972_l2388_238837


namespace NUMINAMATH_CALUDE_existence_of_m_n_l2388_238894

theorem existence_of_m_n (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l2388_238894


namespace NUMINAMATH_CALUDE_cost_effective_purchase_anton_offer_is_best_l2388_238827

/-- Represents a shareholder in the company -/
structure Shareholder where
  name : String
  shares : Nat
  sellPrice : Rat

/-- Represents the company and its shareholders -/
structure Company where
  totalShares : Nat
  sharePrice : Nat
  shareholders : List Shareholder

/-- Calculates the cost of buying shares from a shareholder -/
def buyCost (shareholder : Shareholder) : Rat :=
  shareholder.shares * shareholder.sellPrice

/-- Checks if a shareholder has enough shares to be the largest -/
def isLargestShareholder (company : Company) (shares : Nat) : Prop :=
  ∀ s : Shareholder, s ∈ company.shareholders → shares > s.shares

/-- The main theorem to prove -/
theorem cost_effective_purchase (company : Company) : Prop :=
  let arina : Shareholder := { name := "Arina", shares := 90001, sellPrice := 10 }
  let anton : Shareholder := { name := "Anton", shares := 15000, sellPrice := 14 }
  let arinaNewShares := arina.shares + anton.shares
  isLargestShareholder company arinaNewShares ∧
  ∀ s : Shareholder, s ∈ company.shareholders → s.name ≠ "Arina" →
    buyCost anton ≤ buyCost s ∨ ¬(isLargestShareholder company (arina.shares + s.shares))

/-- The company instance with given conditions -/
def jscCompany : Company := {
  totalShares := 300000,
  sharePrice := 10,
  shareholders := [
    { name := "Arina", shares := 90001, sellPrice := 10 },
    { name := "Maxim", shares := 104999, sellPrice := 11 },
    { name := "Inga", shares := 30000, sellPrice := 12.5 },
    { name := "Yuri", shares := 30000, sellPrice := 11.5 },
    { name := "Yulia", shares := 30000, sellPrice := 13 },
    { name := "Anton", shares := 15000, sellPrice := 14 }
  ]
}

/-- The main theorem applied to our specific company -/
theorem anton_offer_is_best : cost_effective_purchase jscCompany := by
  sorry


end NUMINAMATH_CALUDE_cost_effective_purchase_anton_offer_is_best_l2388_238827


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2388_238826

/-- The sum of the infinite series $\sum_{k = 1}^\infty \frac{k^3}{3^k}$ is equal to $\frac{39}{8}$. -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k^3 : ℝ) / 3^k) = 39 / 8 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2388_238826


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2388_238840

theorem arithmetic_sequence_formula (a : ℕ → ℤ) (n : ℕ) : 
  (a 1 = 1) → 
  (a 2 = 3) → 
  (a 3 = 5) → 
  (a 4 = 7) → 
  (a 5 = 9) → 
  (∀ k : ℕ, a (k + 1) - a k = 2) → 
  a n = 2 * n - 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2388_238840


namespace NUMINAMATH_CALUDE_pencil_sales_l2388_238841

/-- The number of pencils initially sold for a rupee -/
def N : ℕ := 20

/-- The cost price of one pencil -/
def C : ℚ := 1 / 13

/-- Theorem stating that N pencils sold for a rupee results in a 35% loss
    and 10 pencils sold for a rupee results in a 30% gain -/
theorem pencil_sales (N : ℕ) (C : ℚ) :
  (N : ℚ) * (0.65 * C) = 1 ∧ 10 * (1.3 * C) = 1 → N = 20 :=
by sorry

end NUMINAMATH_CALUDE_pencil_sales_l2388_238841


namespace NUMINAMATH_CALUDE_emily_spent_234_l2388_238853

/-- The cost of Charlie's purchase of 4 burgers and 3 sodas -/
def charlie_cost : ℝ := 4.40

/-- The cost of Dana's purchase of 3 burgers and 4 sodas -/
def dana_cost : ℝ := 3.80

/-- The number of burgers in Charlie's purchase -/
def charlie_burgers : ℕ := 4

/-- The number of sodas in Charlie's purchase -/
def charlie_sodas : ℕ := 3

/-- The number of burgers in Dana's purchase -/
def dana_burgers : ℕ := 3

/-- The number of sodas in Dana's purchase -/
def dana_sodas : ℕ := 4

/-- The number of burgers in Emily's purchase -/
def emily_burgers : ℕ := 2

/-- The number of sodas in Emily's purchase -/
def emily_sodas : ℕ := 1

/-- The cost of a single burger -/
noncomputable def burger_cost : ℝ := 
  (charlie_cost * dana_sodas - dana_cost * charlie_sodas) / 
  (charlie_burgers * dana_sodas - dana_burgers * charlie_sodas)

/-- The cost of a single soda -/
noncomputable def soda_cost : ℝ := 
  (charlie_cost * dana_burgers - dana_cost * charlie_burgers) / 
  (charlie_sodas * dana_burgers - dana_sodas * charlie_burgers)

/-- Emily's total cost -/
noncomputable def emily_cost : ℝ := emily_burgers * burger_cost + emily_sodas * soda_cost

theorem emily_spent_234 : ∃ ε > 0, |emily_cost - 2.34| < ε :=
sorry

end NUMINAMATH_CALUDE_emily_spent_234_l2388_238853


namespace NUMINAMATH_CALUDE_min_value_theorem_l2388_238896

-- Define the quadratic function
def f (a c x : ℝ) : ℝ := a * x^2 - 2 * x + c

-- State the theorem
theorem min_value_theorem (a c : ℝ) (h1 : a > 0) (h2 : c > 0) 
  (h3 : Set.range (f a c) = Set.Ici 0) :
  (∀ x : ℝ, 9 / a + 1 / c ≥ 6) ∧ (∃ x : ℝ, 9 / a + 1 / c = 6) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2388_238896


namespace NUMINAMATH_CALUDE_dog_grouping_ways_l2388_238899

def total_dogs : ℕ := 15
def group_1_size : ℕ := 4
def group_2_size : ℕ := 6
def group_3_size : ℕ := 5

def duke_in_group_1 : Prop := True
def bella_in_group_2 : Prop := True

theorem dog_grouping_ways : 
  total_dogs = group_1_size + group_2_size + group_3_size →
  duke_in_group_1 →
  bella_in_group_2 →
  (Nat.choose (total_dogs - 2) (group_1_size - 1)) * 
  (Nat.choose (total_dogs - group_1_size - 1) (group_2_size - 1)) = 72072 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_ways_l2388_238899


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2388_238830

theorem simplify_polynomial (b : ℝ) : (1 : ℝ) * (2 * b) * (3 * b^2) * (4 * b^3) * (5 * b^4) * (6 * b^5) = 720 * b^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2388_238830


namespace NUMINAMATH_CALUDE_triangle_area_from_altitudes_l2388_238819

/-- The area of a triangle given its three altitudes -/
theorem triangle_area_from_altitudes (h₁ h₂ h₃ : ℝ) (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (h₃_pos : h₃ > 0) :
  ∃ S : ℝ, S > 0 ∧ S = Real.sqrt ((1/h₁ + 1/h₂ + 1/h₃) * (-1/h₁ + 1/h₂ + 1/h₃) * (1/h₁ - 1/h₂ + 1/h₃) * (1/h₁ + 1/h₂ - 1/h₃)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_altitudes_l2388_238819


namespace NUMINAMATH_CALUDE_mixture_ratio_theorem_l2388_238812

/-- Represents the components of the mixture -/
inductive Component
  | Milk
  | Water
  | Juice

/-- Calculates the amount of a component in the initial mixture -/
def initial_amount (c : Component) : ℚ :=
  match c with
  | Component.Milk => 60 * (5 / 8)
  | Component.Water => 60 * (2 / 8)
  | Component.Juice => 60 * (1 / 8)

/-- Calculates the amount of a component after adding water and juice -/
def final_amount (c : Component) : ℚ :=
  match c with
  | Component.Milk => initial_amount Component.Milk
  | Component.Water => initial_amount Component.Water + 15
  | Component.Juice => initial_amount Component.Juice + 5

/-- Represents the final ratio of the mixture components -/
def final_ratio : Fin 3 → ℕ
  | 0 => 15  -- Milk
  | 1 => 12  -- Water
  | 2 => 5   -- Juice
  | _ => 0   -- This case is unreachable, but needed for completeness

theorem mixture_ratio_theorem :
  ∃ (k : ℚ), k > 0 ∧
    (final_amount Component.Milk = k * final_ratio 0) ∧
    (final_amount Component.Water = k * final_ratio 1) ∧
    (final_amount Component.Juice = k * final_ratio 2) :=
sorry

end NUMINAMATH_CALUDE_mixture_ratio_theorem_l2388_238812


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2388_238820

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156) 
  (h2 : a + b + c = 16) : 
  a * b + b * c + a * c = 50 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2388_238820


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2388_238849

/-- Given three numbers a, b, and c satisfying certain conditions, 
    prove that their product is equal to 369912000/4913 -/
theorem product_of_three_numbers (a b c m : ℚ) : 
  a + b + c = 180 ∧ 
  8 * a = m ∧ 
  b - 10 = m ∧ 
  c + 10 = m → 
  a * b * c = 369912000 / 4913 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2388_238849


namespace NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l2388_238828

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : Nat)
  (green : Nat)
  (blue : Nat)
  (black : Nat)

/-- The minimum number of socks needed to ensure at least n pairs -/
def minSocksForPairs (drawer : SockDrawer) (n : Nat) : Nat :=
  sorry

theorem min_socks_for_fifteen_pairs :
  let drawer := SockDrawer.mk 120 100 70 50
  minSocksForPairs drawer 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l2388_238828


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2388_238880

theorem no_integer_solutions : 
  ¬ ∃ (x y z : ℤ), x^1988 + y^1988 + z^1988 = 7^1990 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2388_238880


namespace NUMINAMATH_CALUDE_spaceship_total_distance_l2388_238888

/-- The distance traveled by a spaceship between three locations -/
def spaceship_distance (earth_to_x : ℝ) (x_to_y : ℝ) (y_to_earth : ℝ) : ℝ :=
  earth_to_x + x_to_y + y_to_earth

/-- Theorem: The total distance traveled by the spaceship is 0.7 light-years -/
theorem spaceship_total_distance :
  spaceship_distance 0.5 0.1 0.1 = 0.7 := by
  sorry

#eval spaceship_distance 0.5 0.1 0.1

end NUMINAMATH_CALUDE_spaceship_total_distance_l2388_238888


namespace NUMINAMATH_CALUDE_combined_net_earnings_proof_l2388_238832

def connor_hourly_rate : ℝ := 7.20
def connor_hours : ℝ := 8
def emily_hourly_rate : ℝ := 2 * connor_hourly_rate
def emily_hours : ℝ := 10
def sarah_hourly_rate : ℝ := 5 * connor_hourly_rate + connor_hourly_rate
def sarah_hours : ℝ := connor_hours

def connor_deduction_rate : ℝ := 0.05
def emily_deduction_rate : ℝ := 0.08
def sarah_deduction_rate : ℝ := 0.10

def connor_gross_earnings : ℝ := connor_hourly_rate * connor_hours
def emily_gross_earnings : ℝ := emily_hourly_rate * emily_hours
def sarah_gross_earnings : ℝ := sarah_hourly_rate * sarah_hours

def connor_net_earnings : ℝ := connor_gross_earnings * (1 - connor_deduction_rate)
def emily_net_earnings : ℝ := emily_gross_earnings * (1 - emily_deduction_rate)
def sarah_net_earnings : ℝ := sarah_gross_earnings * (1 - sarah_deduction_rate)

def combined_net_earnings : ℝ := connor_net_earnings + emily_net_earnings + sarah_net_earnings

theorem combined_net_earnings_proof : combined_net_earnings = 498.24 := by
  sorry

end NUMINAMATH_CALUDE_combined_net_earnings_proof_l2388_238832


namespace NUMINAMATH_CALUDE_min_blue_eyes_and_water_bottle_l2388_238877

theorem min_blue_eyes_and_water_bottle 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (water_bottle : ℕ) 
  (h1 : total_students = 35) 
  (h2 : blue_eyes = 18) 
  (h3 : water_bottle = 25) : 
  ∃ (both : ℕ), both ≥ 8 ∧ 
    both ≤ blue_eyes ∧ 
    both ≤ water_bottle ∧ 
    (∀ (x : ℕ), x < both → 
      x > blue_eyes - (total_students - water_bottle) ∨ 
      x > water_bottle - (total_students - blue_eyes)) :=
by sorry

end NUMINAMATH_CALUDE_min_blue_eyes_and_water_bottle_l2388_238877


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2388_238872

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 3) (h2 : x * y = 1) : x^2 + y^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2388_238872


namespace NUMINAMATH_CALUDE_cubes_with_three_painted_faces_l2388_238823

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_outside : Bool

/-- Represents a smaller cube cut from a larger cube -/
structure SmallCube where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Function to count the number of painted faces of a small cube -/
def count_painted_faces (c : Cube 4) (sc : SmallCube) : ℕ :=
  sorry

/-- Function to count the number of small cubes with at least three painted faces -/
def count_cubes_with_three_painted_faces (c : Cube 4) : ℕ :=
  sorry

/-- Theorem: In a 4x4x4 cube that is fully painted on the outside and then cut into 1x1x1 cubes,
    the number of 1x1x1 cubes with at least three faces painted is equal to 8 -/
theorem cubes_with_three_painted_faces (c : Cube 4) (h : c.painted_outside = true) :
  count_cubes_with_three_painted_faces c = 8 :=
by sorry

end NUMINAMATH_CALUDE_cubes_with_three_painted_faces_l2388_238823


namespace NUMINAMATH_CALUDE_exam_average_score_l2388_238821

theorem exam_average_score (max_score : ℕ) (amar_percent bhavan_percent chetan_percent deepak_percent : ℚ) :
  max_score = 1100 →
  amar_percent = 64 / 100 →
  bhavan_percent = 36 / 100 →
  chetan_percent = 44 / 100 →
  deepak_percent = 52 / 100 →
  let amar_score := (amar_percent * max_score : ℚ).floor
  let bhavan_score := (bhavan_percent * max_score : ℚ).floor
  let chetan_score := (chetan_percent * max_score : ℚ).floor
  let deepak_score := (deepak_percent * max_score : ℚ).floor
  let total_score := amar_score + bhavan_score + chetan_score + deepak_score
  (total_score / 4 : ℚ).floor = 539 := by
  sorry

#eval (64 / 100 : ℚ) * 1100  -- Expected output: 704
#eval (36 / 100 : ℚ) * 1100  -- Expected output: 396
#eval (44 / 100 : ℚ) * 1100  -- Expected output: 484
#eval (52 / 100 : ℚ) * 1100  -- Expected output: 572
#eval ((704 + 396 + 484 + 572) / 4 : ℚ)  -- Expected output: 539

end NUMINAMATH_CALUDE_exam_average_score_l2388_238821


namespace NUMINAMATH_CALUDE_fifth_grade_students_l2388_238844

theorem fifth_grade_students (total_boys : ℕ) (soccer_players : ℕ) (boys_soccer_percentage : ℚ) (girls_not_soccer : ℕ) :
  total_boys = 296 →
  soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  girls_not_soccer = 89 →
  ∃ (total_students : ℕ), total_students = 420 :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_grade_students_l2388_238844


namespace NUMINAMATH_CALUDE_greater_number_with_hcf_and_product_l2388_238886

theorem greater_number_with_hcf_and_product 
  (A B : ℕ+) 
  (hcf_condition : Nat.gcd A B = 11)
  (product_condition : A * B = 363) :
  max A B = 33 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_with_hcf_and_product_l2388_238886


namespace NUMINAMATH_CALUDE_cube_volume_relation_l2388_238813

theorem cube_volume_relation (V : ℝ) : 
  (∃ (s : ℝ), V = s^3 ∧ 512 = (2*s)^3) → V = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_relation_l2388_238813


namespace NUMINAMATH_CALUDE_height_increase_l2388_238845

/-- If a person's height increases by 5% to reach 147 cm, their original height was 140 cm. -/
theorem height_increase (original_height : ℝ) : 
  original_height * 1.05 = 147 → original_height = 140 :=
by sorry

end NUMINAMATH_CALUDE_height_increase_l2388_238845


namespace NUMINAMATH_CALUDE_both_knights_l2388_238846

-- Define the Person type
inductive Person : Type
| A : Person
| B : Person

-- Define the property of being a knight
def is_knight (p : Person) : Prop := sorry

-- Define A's statement
def A_statement : Prop :=
  ¬(is_knight Person.A) ∨ is_knight Person.B

-- Theorem: If A's statement is true, then both A and B are knights
theorem both_knights (h : A_statement) :
  is_knight Person.A ∧ is_knight Person.B := by
  sorry

end NUMINAMATH_CALUDE_both_knights_l2388_238846


namespace NUMINAMATH_CALUDE_floyd_jumps_exist_l2388_238863

def sum_of_decimal_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_decimal_digits (n / 10)

def floyd_sequence : ℕ → ℕ
  | 0 => 90
  | n + 1 => 2 * (10^(n + 2)) - 28

theorem floyd_jumps_exist :
  ∃ (a : ℕ → ℕ), (∀ n > 0, a n ≤ 2 * a (n - 1)) ∧
                 (∀ i j, i ≠ j → sum_of_decimal_digits (a i) ≠ sum_of_decimal_digits (a j)) :=
by
  sorry


end NUMINAMATH_CALUDE_floyd_jumps_exist_l2388_238863


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2388_238831

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 240) :
  a 9 - (1/3) * a 11 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2388_238831


namespace NUMINAMATH_CALUDE_largest_number_l2388_238824

theorem largest_number (a b c d e : ℝ) 
  (ha : a = 0.993) 
  (hb : b = 0.9899) 
  (hc : c = 0.990) 
  (hd : d = 0.989) 
  (he : e = 0.9909) : 
  a > b ∧ a > c ∧ a > d ∧ a > e :=
sorry

end NUMINAMATH_CALUDE_largest_number_l2388_238824


namespace NUMINAMATH_CALUDE_symmetry_axis_l2388_238898

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- State the property of f
axiom f_property : ∀ x : ℝ, f x = f (3 - x)

-- Define the concept of an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_axis :
  is_axis_of_symmetry (3/2) f :=
sorry

end NUMINAMATH_CALUDE_symmetry_axis_l2388_238898


namespace NUMINAMATH_CALUDE_power_of_64_l2388_238883

theorem power_of_64 : (64 : ℝ) ^ (5/3) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_l2388_238883


namespace NUMINAMATH_CALUDE_trains_meet_at_1108_l2388_238835

/-- Represents a train with its departure time and speed -/
structure Train where
  departureTime : Nat  -- minutes since midnight
  speed : Nat          -- km/h

/-- Represents a station with its distance from Station A -/
structure Station where
  distanceFromA : Nat  -- km

def stationA : Station := { distanceFromA := 0 }
def stationB : Station := { distanceFromA := 300 }
def stationC : Station := { distanceFromA := 150 }

def trainA : Train := { departureTime := 9 * 60 + 45, speed := 60 }
def trainB : Train := { departureTime := 10 * 60, speed := 80 }

def stopTime : Nat := 10  -- minutes

/-- Calculates the meeting time of two trains given the conditions -/
def calculateMeetingTime (trainA trainB : Train) (stationA stationB stationC : Station) (stopTime : Nat) : Nat :=
  sorry  -- Proof to be implemented

theorem trains_meet_at_1108 :
  calculateMeetingTime trainA trainB stationA stationB stationC stopTime = 11 * 60 + 8 := by
  sorry  -- Proof to be implemented

end NUMINAMATH_CALUDE_trains_meet_at_1108_l2388_238835


namespace NUMINAMATH_CALUDE_oliver_shelves_l2388_238805

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  ((total_books - books_taken) + (books_per_shelf - 1)) / books_per_shelf

theorem oliver_shelves :
  shelves_needed 46 10 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_oliver_shelves_l2388_238805


namespace NUMINAMATH_CALUDE_sum_of_squares_geometric_progression_l2388_238866

theorem sum_of_squares_geometric_progression
  (a r : ℝ) 
  (h1 : -1 < r ∧ r < 1) 
  (h2 : ∃ (S : ℝ), S = a / (1 - r)) : 
  ∃ (T : ℝ), T = a^2 / (1 - r^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_geometric_progression_l2388_238866


namespace NUMINAMATH_CALUDE_dinner_cost_is_36_l2388_238857

/-- Represents the farming scenario with kids planting corn --/
structure FarmingScenario where
  kids : ℕ
  ears_per_row : ℕ
  seeds_per_bag : ℕ
  seeds_per_ear : ℕ
  pay_per_row : ℚ
  bags_per_kid : ℕ

/-- Calculates the cost of dinner per kid based on the farming scenario --/
def dinner_cost_per_kid (scenario : FarmingScenario) : ℚ :=
  let ears_per_bag := scenario.seeds_per_bag / scenario.seeds_per_ear
  let total_ears := scenario.bags_per_kid * ears_per_bag
  let rows_planted := total_ears / scenario.ears_per_row
  let earnings := rows_planted * scenario.pay_per_row
  earnings / 2

/-- Theorem stating that the dinner cost per kid is $36 given the specific scenario --/
theorem dinner_cost_is_36 (scenario : FarmingScenario) 
  (h1 : scenario.kids = 4)
  (h2 : scenario.ears_per_row = 70)
  (h3 : scenario.seeds_per_bag = 48)
  (h4 : scenario.seeds_per_ear = 2)
  (h5 : scenario.pay_per_row = 3/2)
  (h6 : scenario.bags_per_kid = 140) :
  dinner_cost_per_kid scenario = 36 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_is_36_l2388_238857


namespace NUMINAMATH_CALUDE_greatest_x_value_l2388_238879

theorem greatest_x_value (x : ℝ) : 
  x ≠ 6 → x ≠ -4 → (x^2 - 3*x - 18) / (x - 6) = 2 / (x + 4) → 
  x ≤ -2 ∧ ∃ y : ℝ, y ≠ 6 ∧ y ≠ -4 ∧ (y^2 - 3*y - 18) / (y - 6) = 2 / (y + 4) ∧ y = -2 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2388_238879


namespace NUMINAMATH_CALUDE_chessboard_number_property_l2388_238808

theorem chessboard_number_property (n : ℕ) (X : Matrix (Fin n) (Fin n) ℝ) 
  (h : ∀ (i j k : Fin n), X i j + X j k + X k i = 0) :
  ∃ (t : Fin n → ℝ), ∀ (i j : Fin n), X i j = t i - t j := by
sorry

end NUMINAMATH_CALUDE_chessboard_number_property_l2388_238808


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l2388_238884

def markup_percentage : ℝ := 0.30
def discount_percentage : ℝ := 0.10

theorem merchant_profit_percentage :
  let marked_price := 1 + markup_percentage
  let discounted_price := marked_price * (1 - discount_percentage)
  (discounted_price - 1) * 100 = 17 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l2388_238884


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2388_238816

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2388_238816


namespace NUMINAMATH_CALUDE_size_relationship_l2388_238833

theorem size_relationship (a b : ℚ) 
  (ha : a > 0) 
  (hb : b < 0) 
  (hab : |a| > |b|) : 
  -a < -b ∧ -b < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_size_relationship_l2388_238833


namespace NUMINAMATH_CALUDE_jordan_rectangle_length_l2388_238801

/-- Given two rectangles with equal areas, where one rectangle measures 15 inches by 20 inches
    and the other has a width of 50 inches, prove that the length of the second rectangle is 6 inches. -/
theorem jordan_rectangle_length (carol_length carol_width jordan_width : ℝ)
    (carol_area jordan_area : ℝ) (h1 : carol_length = 15)
    (h2 : carol_width = 20) (h3 : jordan_width = 50)
    (h4 : carol_area = carol_length * carol_width)
    (h5 : jordan_area = jordan_width * 6)
    (h6 : carol_area = jordan_area) : 6 = jordan_area / jordan_width := by
  sorry

end NUMINAMATH_CALUDE_jordan_rectangle_length_l2388_238801


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l2388_238868

/-- The molecular weight of a compound given the total weight and number of moles -/
theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 525)
  (h2 : num_moles = 3)
  (h3 : total_weight > 0)
  (h4 : num_moles > 0) :
  total_weight / num_moles = 175 := by
sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l2388_238868


namespace NUMINAMATH_CALUDE_union_equals_set_implies_m_equals_one_l2388_238869

theorem union_equals_set_implies_m_equals_one :
  let A : Set ℝ := {-1, 2}
  let B : Set ℝ := {x | m * x + 1 = 0}
  ∀ m : ℝ, (A ∪ B = A) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_union_equals_set_implies_m_equals_one_l2388_238869


namespace NUMINAMATH_CALUDE_hours_to_minutes_l2388_238873

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Define the number of hours Ava watched television
def hours_watched : ℕ := 4

-- Theorem to prove
theorem hours_to_minutes :
  hours_watched * minutes_per_hour = 240 := by
  sorry

end NUMINAMATH_CALUDE_hours_to_minutes_l2388_238873


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2388_238887

theorem solve_exponential_equation :
  ∃ y : ℕ, (8 : ℝ)^4 = 2^y ∧ y = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2388_238887


namespace NUMINAMATH_CALUDE_h_properties_l2388_238897

-- Define the functions f, g, and h
def f : ℝ → ℝ := λ x => x

-- g is symmetric to f with respect to y = x
def g : ℝ → ℝ := λ x => x

def h : ℝ → ℝ := λ x => g (1 - |x|)

-- Theorem statement
theorem h_properties :
  (∀ x, h x = h (-x)) ∧  -- h is an even function
  (∃ m, ∀ x, h x ≥ m ∧ ∃ x₀, h x₀ = m ∧ m = 0) -- The minimum value of h is 0
  := by sorry

end NUMINAMATH_CALUDE_h_properties_l2388_238897


namespace NUMINAMATH_CALUDE_train_length_l2388_238822

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (cross_time_s : ℝ) (h1 : speed_kmh = 90) (h2 : cross_time_s = 10) :
  speed_kmh * (1000 / 3600) * cross_time_s = 250 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2388_238822


namespace NUMINAMATH_CALUDE_ear_muffs_total_l2388_238895

/-- The number of ear muffs bought before December -/
def before_december : ℕ := 1346

/-- The number of ear muffs bought during December -/
def during_december : ℕ := 6444

/-- The total number of ear muffs bought -/
def total_ear_muffs : ℕ := before_december + during_december

theorem ear_muffs_total : total_ear_muffs = 7790 := by
  sorry

end NUMINAMATH_CALUDE_ear_muffs_total_l2388_238895


namespace NUMINAMATH_CALUDE_trajectory_equation_constant_distance_fixed_point_l2388_238882

/-- The trajectory of point P given the conditions -/
def trajectory (x y : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -2 ∧ x^2 / 4 + y^2 = 1

/-- The line l intersecting the trajectory -/
def line_l (k m x y : ℝ) : Prop :=
  y = k * x + m

/-- Points M and N are on both the trajectory and line l -/
def intersection_points (x₁ y₁ x₂ y₂ k m : ℝ) : Prop :=
  trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
  line_l k m x₁ y₁ ∧ line_l k m x₂ y₂ ∧
  (x₁, y₁) ≠ (x₂, y₂)

/-- OM is perpendicular to ON -/
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

/-- The slopes of BM and BN satisfy the given condition -/
def slope_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ / (x₁ - 2)) * (y₂ / (x₂ - 2)) = -1/4

theorem trajectory_equation (x y : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (y / (x + 2)) * (y / (x - 2)) = -1/4 ↔ trajectory x y :=
sorry

theorem constant_distance (k m x₁ y₁ x₂ y₂ : ℝ)
  (h : intersection_points x₁ y₁ x₂ y₂ k m)
  (h_perp : perpendicular x₁ y₁ x₂ y₂) :
  |m| / Real.sqrt (1 + k^2) = 2 * Real.sqrt 5 / 5 :=
sorry

theorem fixed_point (k m x₁ y₁ x₂ y₂ : ℝ)
  (h : intersection_points x₁ y₁ x₂ y₂ k m)
  (h_slope : slope_condition x₁ y₁ x₂ y₂) :
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_constant_distance_fixed_point_l2388_238882


namespace NUMINAMATH_CALUDE_sum_of_digits_888_base8_l2388_238803

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the sum of digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits of the base 8 representation of 888₁₀ is 13 -/
theorem sum_of_digits_888_base8 : sumDigits (toBase8 888) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_888_base8_l2388_238803


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2388_238893

theorem gcd_lcm_product (a b : ℕ) (ha : a = 108) (hb : b = 250) :
  Nat.gcd a b * Nat.lcm a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2388_238893


namespace NUMINAMATH_CALUDE_max_teams_intramurals_l2388_238892

/-- Represents the number of participants in each category -/
structure Participants where
  girls : Nat
  boys : Nat
  teenagers : Nat

/-- Represents the sports preferences for girls -/
structure GirlsPreferences where
  basketball : Nat
  volleyball : Nat
  soccer : Nat

/-- Represents the sports preferences for boys -/
structure BoysPreferences where
  basketball : Nat
  soccer : Nat

/-- Represents the sports preferences for teenagers -/
structure TeenagersPreferences where
  volleyball : Nat
  mixed_sports : Nat

/-- The main theorem statement -/
theorem max_teams_intramurals
  (total : Participants)
  (girls_pref : GirlsPreferences)
  (boys_pref : BoysPreferences)
  (teens_pref : TeenagersPreferences)
  (h1 : total.girls = 120)
  (h2 : total.boys = 96)
  (h3 : total.teenagers = 72)
  (h4 : girls_pref.basketball = 40)
  (h5 : girls_pref.volleyball = 50)
  (h6 : girls_pref.soccer = 30)
  (h7 : boys_pref.basketball = 48)
  (h8 : boys_pref.soccer = 48)
  (h9 : teens_pref.volleyball = 24)
  (h10 : teens_pref.mixed_sports = 48)
  (h11 : girls_pref.basketball + girls_pref.volleyball + girls_pref.soccer = total.girls)
  (h12 : boys_pref.basketball + boys_pref.soccer = total.boys)
  (h13 : teens_pref.volleyball + teens_pref.mixed_sports = total.teenagers) :
  ∃ (n : Nat), n = 24 ∧ 
    n ∣ total.girls ∧ 
    n ∣ total.boys ∧ 
    n ∣ total.teenagers ∧
    n ∣ girls_pref.basketball ∧
    n ∣ girls_pref.volleyball ∧
    n ∣ girls_pref.soccer ∧
    n ∣ boys_pref.basketball ∧
    n ∣ boys_pref.soccer ∧
    n ∣ teens_pref.volleyball ∧
    n ∣ teens_pref.mixed_sports ∧
    ∀ (m : Nat), (m > n) → 
      ¬(m ∣ total.girls ∧ 
        m ∣ total.boys ∧ 
        m ∣ total.teenagers ∧
        m ∣ girls_pref.basketball ∧
        m ∣ girls_pref.volleyball ∧
        m ∣ girls_pref.soccer ∧
        m ∣ boys_pref.basketball ∧
        m ∣ boys_pref.soccer ∧
        m ∣ teens_pref.volleyball ∧
        m ∣ teens_pref.mixed_sports) :=
by
  sorry

end NUMINAMATH_CALUDE_max_teams_intramurals_l2388_238892
