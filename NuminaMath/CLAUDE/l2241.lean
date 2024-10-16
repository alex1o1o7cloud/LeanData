import Mathlib

namespace NUMINAMATH_CALUDE_probability_differ_by_three_l2241_224199

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


end NUMINAMATH_CALUDE_probability_differ_by_three_l2241_224199


namespace NUMINAMATH_CALUDE_goldfish_count_l2241_224134

theorem goldfish_count (daily_food_per_fish : ℝ) (special_food_percentage : ℝ) 
  (special_food_cost_per_ounce : ℝ) (total_special_food_cost : ℝ) 
  (h1 : daily_food_per_fish = 1.5)
  (h2 : special_food_percentage = 0.2)
  (h3 : special_food_cost_per_ounce = 3)
  (h4 : total_special_food_cost = 45) : 
  ∃ (total_fish : ℕ), total_fish = 50 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_count_l2241_224134


namespace NUMINAMATH_CALUDE_power_fraction_product_l2241_224116

theorem power_fraction_product : (-4/5)^2022 * (5/4)^2023 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_product_l2241_224116


namespace NUMINAMATH_CALUDE_abs_a_minus_three_l2241_224151

theorem abs_a_minus_three (a : ℝ) (h : a < 3) : |a - 3| = 3 - a := by
  sorry

end NUMINAMATH_CALUDE_abs_a_minus_three_l2241_224151


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l2241_224177

/-- Given a circle with radius 6 cm that is tangent to three sides of a rectangle,
    where the rectangle's area is four times the circle's area,
    prove that the length of the longer side of the rectangle is 12π cm. -/
theorem rectangle_longer_side (r : ℝ) (circle_area rectangle_area : ℝ) 
  (shorter_side longer_side : ℝ) :
  r = 6 →
  circle_area = Real.pi * r^2 →
  rectangle_area = 4 * circle_area →
  shorter_side = 2 * r →
  rectangle_area = shorter_side * longer_side →
  longer_side = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l2241_224177


namespace NUMINAMATH_CALUDE_age_difference_l2241_224135

theorem age_difference (A B C : ℤ) 
  (h1 : A + B = B + C + 15) 
  (h2 : C = A - 15) : 
  (A + B) - (B + C) = 15 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2241_224135


namespace NUMINAMATH_CALUDE_equation_solution_l2241_224110

theorem equation_solution : ∃! x : ℚ, (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2241_224110


namespace NUMINAMATH_CALUDE_library_visitors_average_l2241_224171

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays : ℕ := 5
  let totalOtherDays : ℕ := 25
  let totalVisitors : ℕ := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  totalVisitors / 30

/-- Theorem stating that the average number of visitors per day is 285 -/
theorem library_visitors_average (sundayVisitors : ℕ) (otherDayVisitors : ℕ) 
    (h1 : sundayVisitors = 510) (h2 : otherDayVisitors = 240) : 
    averageVisitorsPerDay sundayVisitors otherDayVisitors = 285 := by
  sorry

#eval averageVisitorsPerDay 510 240

end NUMINAMATH_CALUDE_library_visitors_average_l2241_224171


namespace NUMINAMATH_CALUDE_student_count_l2241_224132

theorem student_count (W : ℝ) (N : ℕ) (h1 : N > 0) :
  W / N - 12 = (W - 72 + 12) / N → N = 5 := by
sorry

end NUMINAMATH_CALUDE_student_count_l2241_224132


namespace NUMINAMATH_CALUDE_spending_difference_l2241_224163

def supermarket_spending (x : ℝ) : Prop :=
  x > 0 ∧ x < 350

def automobile_repair_cost : ℝ := 350

def total_spent : ℝ := 450

theorem spending_difference (x : ℝ) 
  (h1 : supermarket_spending x) 
  (h2 : x + automobile_repair_cost = total_spent) 
  (h3 : automobile_repair_cost > 3 * x) : 
  automobile_repair_cost - 3 * x = 50 := by
sorry

end NUMINAMATH_CALUDE_spending_difference_l2241_224163


namespace NUMINAMATH_CALUDE_xy_value_l2241_224168

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 40) : x * y = 85 / 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2241_224168


namespace NUMINAMATH_CALUDE_complex_system_solution_l2241_224179

theorem complex_system_solution (z₁ z₂ : ℂ) 
  (eq1 : z₁ - 2 * z₂ = 5 + Complex.I) 
  (eq2 : 2 * z₁ + z₂ = 3 * Complex.I) : 
  z₁ = 1 + (7 / 5) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_system_solution_l2241_224179


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_100_over_9_l2241_224192

theorem ceiling_neg_sqrt_100_over_9 : ⌈-Real.sqrt (100 / 9)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_100_over_9_l2241_224192


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2241_224194

def num_boys : ℕ := 1500
def num_girls : ℕ := 1200

theorem boys_to_girls_ratio :
  (num_boys : ℚ) / (num_girls : ℚ) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2241_224194


namespace NUMINAMATH_CALUDE_unique_m_value_l2241_224173

theorem unique_m_value (a b c m : ℤ) 
  (h1 : 0 ≤ m ∧ m ≤ 26)
  (h2 : (a + b + c) % 27 = m)
  (h3 : ((a - b) * (b - c) * (c - a)) % 27 = m) : 
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_m_value_l2241_224173


namespace NUMINAMATH_CALUDE_surface_area_of_sliced_solid_l2241_224157

/-- Right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ

/-- Point on an edge of the prism -/
structure EdgePoint where
  edge : Fin 3  -- 0: AC, 1: BC, 2: DC
  position : ℝ  -- Fraction of the way along the edge

/-- Solid formed by slicing the prism -/
def SlicedSolid (prism : RightPrism) (p q r : EdgePoint) : Type := sorry

/-- Surface area of a sliced solid -/
noncomputable def surfaceArea (prism : RightPrism) (solid : SlicedSolid prism p q r) : ℝ := sorry

/-- The main theorem -/
theorem surface_area_of_sliced_solid 
  (prism : RightPrism)
  (h_height : prism.height = 20)
  (h_base : prism.baseSideLength = 10)
  (p : EdgePoint)
  (h_p : p.edge = 0 ∧ p.position = 1/3)
  (q : EdgePoint)
  (h_q : q.edge = 1 ∧ q.position = 1/3)
  (r : EdgePoint)
  (h_r : r.edge = 2 ∧ r.position = 1/2)
  (solid : SlicedSolid prism p q r) :
  surfaceArea prism solid = (50 * Real.sqrt 3 + 25 * Real.sqrt 2 / 3 + 50 * Real.sqrt 10) / 3 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_sliced_solid_l2241_224157


namespace NUMINAMATH_CALUDE_bakery_flour_calculation_l2241_224150

theorem bakery_flour_calculation (total_flour : ℚ) :
  total_flour = 40 * (1/8 : ℚ) →
  total_flour = 25 * (1/5 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_bakery_flour_calculation_l2241_224150


namespace NUMINAMATH_CALUDE_rectangles_form_square_l2241_224186

/-- A rectangle represented by its width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculate the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Check if a list of rectangles can form a square of side length 16 --/
def canFormSquare (rectangles : List Rectangle) : Prop :=
  ∃ (arrangement : List Rectangle), 
    arrangement.length = rectangles.length ∧
    (∀ r ∈ arrangement, r ∈ rectangles) ∧
    (∀ r ∈ rectangles, r ∈ arrangement) ∧
    arrangement.foldr (λ r acc => acc + r.width * r.height) 0 = 16 * 16

/-- The main theorem to prove --/
theorem rectangles_form_square :
  ∃ (rectangles : List Rectangle),
    rectangles.foldr (λ r acc => acc + perimeter r) 0 = 100 ∧
    canFormSquare rectangles := by sorry

end NUMINAMATH_CALUDE_rectangles_form_square_l2241_224186


namespace NUMINAMATH_CALUDE_visitor_difference_l2241_224114

def visitors_previous_day : ℕ := 100
def visitors_that_day : ℕ := 666

theorem visitor_difference : visitors_that_day - visitors_previous_day = 566 := by
  sorry

end NUMINAMATH_CALUDE_visitor_difference_l2241_224114


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_seven_l2241_224129

theorem fraction_zero_implies_x_equals_seven :
  ∀ x : ℝ, (x^2 - 49) / (x + 7) = 0 → x = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_seven_l2241_224129


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2241_224124

theorem parallel_lines_m_value (x y : ℝ) :
  (∀ x y, 2*x + 3*y + 1 = 0 ↔ m*x + 6*y - 5 = 0) →
  m = 4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l2241_224124


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l2241_224196

/-- The distance between the foci of the hyperbola xy = 2 is 2√2 -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ),
    (∀ (x y : ℝ), x * y = 2 → (x - f₁.1) * (y - f₁.2) = (x - f₂.1) * (y - f₂.2)) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l2241_224196


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l2241_224143

theorem angle_sum_is_pi_over_two (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2) 
  (h_equation : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) : 
  α + β = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l2241_224143


namespace NUMINAMATH_CALUDE_max_cube_sum_on_sphere_l2241_224106

theorem max_cube_sum_on_sphere (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) :
  x^3 + y^3 + z^3 ≤ 27 ∧ ∃ a b c : ℝ, a^2 + b^2 + c^2 = 9 ∧ a^3 + b^3 + c^3 = 27 :=
by sorry

end NUMINAMATH_CALUDE_max_cube_sum_on_sphere_l2241_224106


namespace NUMINAMATH_CALUDE_sum_of_roots_l2241_224142

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2241_224142


namespace NUMINAMATH_CALUDE_Q_subset_P_l2241_224123

def P : Set ℝ := {x | x ≥ -1}
def Q : Set ℝ := {y | y ≥ 0}

theorem Q_subset_P : Q ⊆ P := by sorry

end NUMINAMATH_CALUDE_Q_subset_P_l2241_224123


namespace NUMINAMATH_CALUDE_samantha_pet_food_difference_l2241_224138

/-- Proves that Samantha bought 49 more cans of cat food than dog and bird food combined. -/
theorem samantha_pet_food_difference : 
  let cat_packages : ℕ := 8
  let dog_packages : ℕ := 5
  let bird_packages : ℕ := 3
  let cat_cans_per_package : ℕ := 12
  let dog_cans_per_package : ℕ := 7
  let bird_cans_per_package : ℕ := 4
  let total_cat_cans := cat_packages * cat_cans_per_package
  let total_dog_cans := dog_packages * dog_cans_per_package
  let total_bird_cans := bird_packages * bird_cans_per_package
  total_cat_cans - (total_dog_cans + total_bird_cans) = 49 := by
  sorry

#eval 8 * 12 - (5 * 7 + 3 * 4)  -- Should output 49

end NUMINAMATH_CALUDE_samantha_pet_food_difference_l2241_224138


namespace NUMINAMATH_CALUDE_caroline_sequence_l2241_224154

def alternating_operation (n : ℕ) (x : ℚ) : ℚ :=
  if n % 2 = 1 then x / 5 else x * 3

def final_result (initial : ℚ) (steps : ℕ) : ℚ :=
  (List.range steps).foldl (λ acc i => alternating_operation i acc) initial

theorem caroline_sequence :
  final_result (10^7 : ℚ) 14 = (2^7 * 3^7 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_caroline_sequence_l2241_224154


namespace NUMINAMATH_CALUDE_brians_trip_distance_l2241_224144

/-- Calculates the distance traveled given car efficiency and gas used -/
def distance_traveled (efficiency : ℝ) (gas_used : ℝ) : ℝ :=
  efficiency * gas_used

/-- Proves that given a car efficiency of 20 miles per gallon and 
    a gas usage of 3 gallons, the distance traveled is 60 miles -/
theorem brians_trip_distance : 
  distance_traveled 20 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_brians_trip_distance_l2241_224144


namespace NUMINAMATH_CALUDE_fruit_purchase_problem_l2241_224111

theorem fruit_purchase_problem (x y : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ x + y = 7 ∧ 5 * x + 8 * y = 41 → x = 5 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_problem_l2241_224111


namespace NUMINAMATH_CALUDE_not_perfect_cube_l2241_224113

theorem not_perfect_cube (n : ℕ) : ¬ ∃ k : ℤ, 2^(2^n) + 1 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_cube_l2241_224113


namespace NUMINAMATH_CALUDE_manager_selection_l2241_224188

theorem manager_selection (n m k : ℕ) (h1 : n = 8) (h2 : m = 4) (h3 : k = 2) : 
  (n.choose m) - ((n - k).choose (m - k)) = 55 := by
  sorry

end NUMINAMATH_CALUDE_manager_selection_l2241_224188


namespace NUMINAMATH_CALUDE_shopping_mall_problem_l2241_224180

/-- Represents the shopping mall's product purchasing scenario -/
structure ProductScenario where
  cost_a : ℚ  -- Cost price of product A
  cost_b : ℚ  -- Cost price of product B
  quantity_a : ℕ  -- Quantity of product A purchased
  quantity_b : ℕ  -- Quantity of product B purchased

/-- Theorem representing the shopping mall problem -/
theorem shopping_mall_problem 
  (scenario : ProductScenario) 
  (h1 : scenario.cost_a = scenario.cost_b - 2)
  (h2 : 80 / scenario.cost_a = 100 / scenario.cost_b)
  (h3 : scenario.quantity_a = 3 * scenario.quantity_b - 5)
  (h4 : scenario.quantity_a + scenario.quantity_b ≤ 95)
  (h5 : (12 - scenario.cost_a) * scenario.quantity_a + 
        (15 - scenario.cost_b) * scenario.quantity_b > 380) :
  (scenario.cost_a = 8 ∧ scenario.cost_b = 10) ∧
  (∀ s : ProductScenario, s.quantity_b ≤ 25) ∧
  (scenario.quantity_a = 67 ∧ scenario.quantity_b = 24) ∨
  (scenario.quantity_a = 70 ∧ scenario.quantity_b = 25) :=
sorry

end NUMINAMATH_CALUDE_shopping_mall_problem_l2241_224180


namespace NUMINAMATH_CALUDE_total_miles_jogged_l2241_224117

/-- The number of miles a person jogs per day on weekdays -/
def miles_per_day : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 3

/-- Theorem: A person who jogs 5 miles per day on weekdays will run 75 miles over three weeks -/
theorem total_miles_jogged : 
  miles_per_day * weekdays_per_week * num_weeks = 75 := by sorry

end NUMINAMATH_CALUDE_total_miles_jogged_l2241_224117


namespace NUMINAMATH_CALUDE_mikes_current_age_l2241_224153

theorem mikes_current_age :
  ∀ (M B : ℕ),
  B = M / 2 →
  M - B = 24 - 16 →
  M = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_mikes_current_age_l2241_224153


namespace NUMINAMATH_CALUDE_mike_toy_expenses_l2241_224165

theorem mike_toy_expenses : 
  let marbles_cost : ℚ := 9.05
  let football_cost : ℚ := 4.95
  let baseball_cost : ℚ := 6.52
  marbles_cost + football_cost + baseball_cost = 20.52 := by
sorry

end NUMINAMATH_CALUDE_mike_toy_expenses_l2241_224165


namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l2241_224167

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l2241_224167


namespace NUMINAMATH_CALUDE_a_minus_b_values_l2241_224187

theorem a_minus_b_values (a b : ℝ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a + b > 0) :
  (a - b = 4) ∨ (a - b = 8) :=
sorry

end NUMINAMATH_CALUDE_a_minus_b_values_l2241_224187


namespace NUMINAMATH_CALUDE_stock_b_highest_income_l2241_224181

/-- Represents a stock with its dividend rate and price per share -/
structure Stock where
  dividend_rate : Rat
  price_per_share : Nat

/-- Calculates the annual income from a stock given the total investment -/
def annual_income (stock : Stock) (total_investment : Nat) : Rat :=
  (total_investment : Rat) * stock.dividend_rate

/-- Theorem: Stock B yields the highest annual income among the three stocks -/
theorem stock_b_highest_income (total_investment : Nat) 
  (stock_a stock_b stock_c : Stock)
  (h_total : total_investment = 6800)
  (h_a : stock_a = { dividend_rate := 1/10, price_per_share := 136 })
  (h_b : stock_b = { dividend_rate := 12/100, price_per_share := 150 })
  (h_c : stock_c = { dividend_rate := 8/100, price_per_share := 100 }) :
  annual_income stock_b (150 * (total_investment / 150)) ≥ 
    max (annual_income stock_a (136 * (total_investment / 136)))
        (annual_income stock_c (100 * (total_investment / 100))) :=
by sorry


end NUMINAMATH_CALUDE_stock_b_highest_income_l2241_224181


namespace NUMINAMATH_CALUDE_javier_first_throw_distance_l2241_224191

/-- Represents the distances of Javier's three javelin throws -/
structure JavelinThrows where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Theorem stating the distance of Javier's first throw given the conditions -/
theorem javier_first_throw_distance (throws : JavelinThrows) :
  throws.first = 2 * throws.second ∧
  throws.first = throws.third / 2 ∧
  throws.first + throws.second + throws.third = 1050 →
  throws.first = 300 := by
  sorry

end NUMINAMATH_CALUDE_javier_first_throw_distance_l2241_224191


namespace NUMINAMATH_CALUDE_fraction_simplification_l2241_224148

theorem fraction_simplification (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : z - 1/x ≠ 0) :
  (x*z - 1/y) / (z - 1/x) = z :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2241_224148


namespace NUMINAMATH_CALUDE_N_mod_45_l2241_224141

/-- N is the number formed by concatenating integers from 1 to 52 -/
def N : ℕ := sorry

theorem N_mod_45 : N % 45 = 37 := by sorry

end NUMINAMATH_CALUDE_N_mod_45_l2241_224141


namespace NUMINAMATH_CALUDE_y_value_l2241_224176

theorem y_value (x y : ℝ) (h1 : x^3 - x - 2 = y + 2) (h2 : x = 3) : y = 20 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2241_224176


namespace NUMINAMATH_CALUDE_range_of_a_l2241_224183

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on [0, +∞) if f(x) ≥ f(y) for all 0 ≤ x ≤ y -/
def IsDecreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≥ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
    (h_even : IsEven f)
    (h_decreasing : IsDecreasingOnNonnegative f)
    (h_inequality : ∀ x, x ∈ Set.Ici 1 ∩ Set.Iio 3 → 
      f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) :
    a ∈ Set.Icc (Real.exp (-1)) ((2 + Real.log 3) / 3) := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l2241_224183


namespace NUMINAMATH_CALUDE_wrapping_paper_cost_l2241_224145

-- Define the problem parameters
def shirtBoxesPerRoll : ℕ := 5
def xlBoxesPerRoll : ℕ := 3
def totalShirtBoxes : ℕ := 20
def totalXlBoxes : ℕ := 12
def totalCost : ℚ := 32

-- Define the theorem
theorem wrapping_paper_cost :
  let rollsForShirtBoxes := totalShirtBoxes / shirtBoxesPerRoll
  let rollsForXlBoxes := totalXlBoxes / xlBoxesPerRoll
  let totalRolls := rollsForShirtBoxes + rollsForXlBoxes
  totalCost / totalRolls = 4 := by sorry

end NUMINAMATH_CALUDE_wrapping_paper_cost_l2241_224145


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_when_divided_by_8_l2241_224139

theorem largest_integer_less_than_100_remainder_5_when_divided_by_8 : 
  ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 → m % 8 = 5 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_when_divided_by_8_l2241_224139


namespace NUMINAMATH_CALUDE_triangle_area_l2241_224172

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![1, 5]

theorem triangle_area : 
  (1/2 : ℝ) * |Matrix.det !![a 0, a 1; b 0, b 1]| = (13/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2241_224172


namespace NUMINAMATH_CALUDE_min_fence_length_l2241_224175

theorem min_fence_length (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 100) :
  2 * (x + y) ≥ 40 ∧ (2 * (x + y) = 40 ↔ x = 10 ∧ y = 10) := by
  sorry

end NUMINAMATH_CALUDE_min_fence_length_l2241_224175


namespace NUMINAMATH_CALUDE_smallest_shift_l2241_224128

/-- A function with period 30 -/
def periodic_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 30) = g x

/-- The shift property for g(x/4) -/
def shift_property (g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 4) = g (x / 4)

/-- The theorem stating the smallest positive b is 120 -/
theorem smallest_shift (g : ℝ → ℝ) (h : periodic_function g) :
  ∃ b : ℝ, b > 0 ∧ shift_property g b ∧ ∀ b' : ℝ, b' > 0 → shift_property g b' → b ≤ b' :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_l2241_224128


namespace NUMINAMATH_CALUDE_ships_met_equals_journey_duration_atlantic_crossing_meets_seven_ships_l2241_224119

/-- Represents a steamship journey between two ports -/
structure Journey where
  departureDays : ℕ  -- Number of days since the first ship departed
  travelDays : ℕ     -- Number of days the journey takes

/-- The number of ships met during a journey -/
def shipsMetDuringJourney (j : Journey) : ℕ :=
  j.travelDays

/-- Theorem: The number of ships met during a journey is equal to the journey's duration -/
theorem ships_met_equals_journey_duration (j : Journey) :
  shipsMetDuringJourney j = j.travelDays :=
by sorry

/-- The specific journey described in the problem -/
def atlanticCrossing : Journey :=
  { departureDays := 1,  -- A ship departs every day
    travelDays := 7 }    -- The journey takes 7 days

/-- Theorem: A ship crossing the Atlantic meets 7 other ships -/
theorem atlantic_crossing_meets_seven_ships :
  shipsMetDuringJourney atlanticCrossing = 7 :=
by sorry

end NUMINAMATH_CALUDE_ships_met_equals_journey_duration_atlantic_crossing_meets_seven_ships_l2241_224119


namespace NUMINAMATH_CALUDE_square_equality_l2241_224156

theorem square_equality (a b : ℝ) : a = b → a^2 = b^2 := by sorry

end NUMINAMATH_CALUDE_square_equality_l2241_224156


namespace NUMINAMATH_CALUDE_f_of_one_eq_two_l2241_224102

def f (x : ℝ) := 3 * x - 1

theorem f_of_one_eq_two : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_eq_two_l2241_224102


namespace NUMINAMATH_CALUDE_problem_solution_l2241_224118

theorem problem_solution (a b c d : ℚ) :
  (2*a + 2 = 3*b + 3) ∧
  (3*b + 3 = 4*c + 4) ∧
  (4*c + 4 = 5*d + 5) ∧
  (5*d + 5 = 2*a + 3*b + 4*c + 5*d + 6) →
  2*a + 3*b + 4*c + 5*d = -10/3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2241_224118


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2241_224112

theorem quadratic_roots_sum (a b c : ℝ) : 
  (∀ x : ℝ, a * (x^4 + x^2)^2 + b * (x^4 + x^2) + c ≥ a * (x^3 + 2)^2 + b * (x^3 + 2) + c) →
  (∃ r₁ r₂ : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = a * (x - r₁) * (x - r₂)) →
  r₁ + r₂ = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2241_224112


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l2241_224166

theorem sum_of_numbers_with_lcm_and_ratio 
  (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 60)
  (h_ratio : a * 3 = b * 2) : 
  a + b = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l2241_224166


namespace NUMINAMATH_CALUDE_jade_cal_difference_l2241_224115

/-- The number of transactions handled by different people on Thursday -/
def thursday_transactions : ℕ → ℕ
| 0 => 90  -- Mabel's transactions
| 1 => (110 * thursday_transactions 0) / 100  -- Anthony's transactions
| 2 => (2 * thursday_transactions 1) / 3  -- Cal's transactions
| 3 => 84  -- Jade's transactions
| _ => 0

/-- The theorem stating the difference between Jade's and Cal's transactions -/
theorem jade_cal_difference : 
  thursday_transactions 3 - thursday_transactions 2 = 18 :=
sorry

end NUMINAMATH_CALUDE_jade_cal_difference_l2241_224115


namespace NUMINAMATH_CALUDE_dad_steps_l2241_224159

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between Dad's and Masha's steps -/
def dad_masha_ratio (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha

/-- Defines the relationship between Masha's and Yasha's steps -/
def masha_yasha_ratio (s : Steps) : Prop :=
  5 * s.masha = 3 * s.yasha

/-- States that Masha and Yasha together took 400 steps -/
def total_masha_yasha (s : Steps) : Prop :=
  s.masha + s.yasha = 400

theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s)
  (h2 : masha_yasha_ratio s)
  (h3 : total_masha_yasha s) :
  s.dad = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_l2241_224159


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l2241_224197

theorem trigonometric_expression_equality : 
  4 * (Real.sin (49 * π / 48) ^ 3 * Real.cos (49 * π / 16) + 
       Real.cos (49 * π / 48) ^ 3 * Real.sin (49 * π / 16)) * 
       Real.cos (49 * π / 12) = 0.75 := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l2241_224197


namespace NUMINAMATH_CALUDE_star_equation_solution_l2241_224122

/-- Custom binary operation ⋆ -/
def star (a b : ℝ) : ℝ := a * b + 2 * b - a

/-- Theorem stating that if 7 ⋆ y = 85, then y = 92/9 -/
theorem star_equation_solution (y : ℝ) (h : star 7 y = 85) : y = 92 / 9 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2241_224122


namespace NUMINAMATH_CALUDE_problem_statement_l2241_224107

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 1/2) : 
  m = 100 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2241_224107


namespace NUMINAMATH_CALUDE_max_floors_is_fourteen_fourteen_floors_is_feasible_l2241_224121

/-- Represents a building with elevators -/
structure Building where
  num_elevators : ℕ
  num_floors : ℕ
  stops_per_elevator : ℕ
  every_two_floors_connected : Bool

/-- The conditions of our specific building -/
def our_building : Building := {
  num_elevators := 7,
  num_floors := 14,  -- We'll prove this is the maximum
  stops_per_elevator := 6,
  every_two_floors_connected := true
}

/-- The theorem stating that 14 is the maximum number of floors -/
theorem max_floors_is_fourteen (b : Building) 
  (h1 : b.num_elevators = 7)
  (h2 : b.stops_per_elevator = 6)
  (h3 : b.every_two_floors_connected = true) :
  b.num_floors ≤ 14 := by
  sorry

/-- The theorem stating that 14 floors is feasible -/
theorem fourteen_floors_is_feasible (b : Building) 
  (h1 : b.num_elevators = 7)
  (h2 : b.stops_per_elevator = 6)
  (h3 : b.every_two_floors_connected = true) :
  ∃ (b' : Building), b'.num_floors = 14 ∧ 
    b'.num_elevators = b.num_elevators ∧ 
    b'.stops_per_elevator = b.stops_per_elevator ∧ 
    b'.every_two_floors_connected = b.every_two_floors_connected := by
  sorry

end NUMINAMATH_CALUDE_max_floors_is_fourteen_fourteen_floors_is_feasible_l2241_224121


namespace NUMINAMATH_CALUDE_train_length_l2241_224126

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 225 →
  train_speed * crossing_time - bridge_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2241_224126


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2241_224149

theorem trigonometric_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) /
  Real.cos (10 * π / 180) = (Real.sqrt 3 + 2) * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2241_224149


namespace NUMINAMATH_CALUDE_rectangle_area_l2241_224147

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 16 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2241_224147


namespace NUMINAMATH_CALUDE_revenue_change_l2241_224136

theorem revenue_change
  (initial_price initial_quantity : ℝ)
  (price_increase : ℝ)
  (quantity_decrease : ℝ)
  (h_price : price_increase = 0.4)
  (h_quantity : quantity_decrease = 0.2)
  : (1 + price_increase) * (1 - quantity_decrease) * initial_price * initial_quantity
    = 1.12 * initial_price * initial_quantity :=
by sorry

end NUMINAMATH_CALUDE_revenue_change_l2241_224136


namespace NUMINAMATH_CALUDE_line_equation_proof_l2241_224158

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def lies_on (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equation_proof (l : Line) :
  parallel l (Line.mk 2 (-1) 1) →
  lies_on (Point.mk 1 2) l →
  l = Line.mk 2 (-1) 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2241_224158


namespace NUMINAMATH_CALUDE_bookArrangements_eq_48_l2241_224104

/-- The number of ways to arrange 3 different math books and 2 different Chinese books in a row,
    with the Chinese books placed next to each other. -/
def bookArrangements : ℕ :=
  (Nat.factorial 4) * (Nat.factorial 2)

/-- The total number of arrangements is 48. -/
theorem bookArrangements_eq_48 : bookArrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_bookArrangements_eq_48_l2241_224104


namespace NUMINAMATH_CALUDE_addition_puzzle_l2241_224100

theorem addition_puzzle (P Q R : ℕ) : 
  P < 10 ∧ Q < 10 ∧ R < 10 →
  1000 * P + 100 * Q + 10 * P + R * 1000 + Q * 100 + Q * 10 + Q = 2009 →
  P + Q + R = 10 := by
  sorry

end NUMINAMATH_CALUDE_addition_puzzle_l2241_224100


namespace NUMINAMATH_CALUDE_election_votes_l2241_224108

theorem election_votes (total_votes : ℕ) (winner_votes loser_votes : ℕ) : 
  winner_votes = (56 : ℕ) * total_votes / 100 →
  loser_votes = total_votes - winner_votes →
  winner_votes - loser_votes = 288 →
  winner_votes = 1344 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l2241_224108


namespace NUMINAMATH_CALUDE_mango_juice_savings_l2241_224170

/-- Represents the volume and cost of a bottle of mango juice -/
structure Bottle where
  volume : ℕ  -- volume in ounces
  cost : ℕ    -- cost in pesetas

/-- Calculates the savings when buying a big bottle instead of equivalent small bottles -/
def calculateSavings (bigBottle smallBottle : Bottle) : ℕ :=
  let smallBottlesNeeded := bigBottle.volume / smallBottle.volume
  smallBottlesNeeded * smallBottle.cost - bigBottle.cost

/-- Theorem stating the savings when buying a big bottle instead of equivalent small bottles -/
theorem mango_juice_savings :
  let bigBottle : Bottle := { volume := 30, cost := 2700 }
  let smallBottle : Bottle := { volume := 6, cost := 600 }
  calculateSavings bigBottle smallBottle = 300 := by
  sorry


end NUMINAMATH_CALUDE_mango_juice_savings_l2241_224170


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l2241_224125

/-- Given a fractional equation (2x - m) / (x + 1) = 3 where x is positive,
    prove that m < -3 -/
theorem fractional_equation_solution_range (x m : ℝ) :
  (2 * x - m) / (x + 1) = 3 → x > 0 → m < -3 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l2241_224125


namespace NUMINAMATH_CALUDE_second_number_value_l2241_224155

theorem second_number_value (x y : ℝ) : 
  x - y = 88 → y = 0.2 * x → y = 22 := by sorry

end NUMINAMATH_CALUDE_second_number_value_l2241_224155


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2241_224152

/-- Represents a rectangular field with a given length, breadth, and perimeter. -/
structure RectangularField where
  length : ℝ
  breadth : ℝ
  perimeter : ℝ

/-- The area of a rectangular field. -/
def area (field : RectangularField) : ℝ :=
  field.length * field.breadth

/-- Theorem: For a rectangular field where the breadth is 60% of the length
    and the perimeter is 800 m, the area of the field is 37,500 square meters. -/
theorem rectangular_field_area :
  ∀ (field : RectangularField),
    field.breadth = 0.6 * field.length →
    field.perimeter = 800 →
    area field = 37500 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2241_224152


namespace NUMINAMATH_CALUDE_either_odd_or_even_l2241_224160

theorem either_odd_or_even (n : ℤ) : 
  (Odd (2*n - 1)) ∨ (Even (2*n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_either_odd_or_even_l2241_224160


namespace NUMINAMATH_CALUDE_inequality_implies_m_range_l2241_224131

theorem inequality_implies_m_range (m : ℝ) : 
  (∀ x > 0, (m * Real.exp x) / x ≥ 6 - 4 * x) → m ≥ 2 * Real.exp (-1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_m_range_l2241_224131


namespace NUMINAMATH_CALUDE_cubic_root_property_l2241_224198

/-- Given a cubic polynomial with roots α, β, and γ, 
    there exist constants A, B, and C such that Aα² + Bα + C = β or γ -/
theorem cubic_root_property (a b c : ℝ) (α β γ : ℝ) : 
  (∀ x, x^3 + a*x^2 + b*x + c = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  ∃ A B C : ℝ, A*α^2 + B*α + C = β ∨ A*α^2 + B*α + C = γ := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_property_l2241_224198


namespace NUMINAMATH_CALUDE_new_person_weight_l2241_224184

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 20 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 40 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2241_224184


namespace NUMINAMATH_CALUDE_problem_statement_l2241_224105

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The problem statement -/
theorem problem_statement :
  floor ((2015^2 : ℝ) / (2013 * 2014) - (2013^2 : ℝ) / (2014 * 2015)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2241_224105


namespace NUMINAMATH_CALUDE_john_trees_chopped_l2241_224133

/-- Represents the number of trees John chopped down -/
def num_trees : ℕ := 30

/-- Represents the number of planks that can be made from each tree -/
def planks_per_tree : ℕ := 25

/-- Represents the number of planks needed to make one table -/
def planks_per_table : ℕ := 15

/-- Represents the selling price of each table in dollars -/
def price_per_table : ℕ := 300

/-- Represents the total labor cost in dollars -/
def labor_cost : ℕ := 3000

/-- Represents the profit John made in dollars -/
def profit : ℕ := 12000

theorem john_trees_chopped :
  num_trees * planks_per_tree / planks_per_table * price_per_table - labor_cost = profit :=
sorry

end NUMINAMATH_CALUDE_john_trees_chopped_l2241_224133


namespace NUMINAMATH_CALUDE_gift_contribution_ratio_l2241_224189

theorem gift_contribution_ratio : 
  let lisa_savings : ℚ := 1200
  let mother_contribution : ℚ := 3/5 * lisa_savings
  let total_needed : ℚ := 3760
  let shortfall : ℚ := 400
  let total_contributions : ℚ := total_needed - shortfall
  let brother_contribution : ℚ := total_contributions - lisa_savings - mother_contribution
  brother_contribution / mother_contribution = 2 := by sorry

end NUMINAMATH_CALUDE_gift_contribution_ratio_l2241_224189


namespace NUMINAMATH_CALUDE_intersection_M_N_l2241_224130

def M : Set ℝ := {x | (x - 3) / (x + 1) ≤ 0}
def N : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_M_N : M ∩ N = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2241_224130


namespace NUMINAMATH_CALUDE_x_equals_three_l2241_224109

theorem x_equals_three (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 3 * x^2 + 18 * x * y = x^3 + 3 * x^2 * y + 6 * x) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_three_l2241_224109


namespace NUMINAMATH_CALUDE_phone_number_theorem_l2241_224120

def phone_number_count (n : ℕ) (k : ℕ) : ℕ := 2^n

theorem phone_number_theorem : phone_number_count 5 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_phone_number_theorem_l2241_224120


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2241_224195

theorem complex_fraction_simplification :
  (Complex.mk 3 (-5)) / (Complex.mk 2 (-7)) = Complex.mk (-41/45) (-11/45) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2241_224195


namespace NUMINAMATH_CALUDE_good_number_theorem_l2241_224169

def is_good (n : ℕ) : Prop :=
  ∃ (k₁ k₂ k₃ k₄ : ℕ), 
    k₁ > 0 ∧ k₂ > 0 ∧ k₃ > 0 ∧ k₄ > 0 ∧
    k₁ ≠ k₂ ∧ k₁ ≠ k₃ ∧ k₁ ≠ k₄ ∧ k₂ ≠ k₃ ∧ k₂ ≠ k₄ ∧ k₃ ≠ k₄ ∧
    (n + k₁ ∣ n + k₁^2) ∧ (n + k₂ ∣ n + k₂^2) ∧ (n + k₃ ∣ n + k₃^2) ∧ (n + k₄ ∣ n + k₄^2) ∧
    ∀ (k : ℕ), k > 0 ∧ k ≠ k₁ ∧ k ≠ k₂ ∧ k ≠ k₃ ∧ k ≠ k₄ → ¬(n + k ∣ n + k^2)

theorem good_number_theorem :
  is_good 58 ∧ 
  ∀ (p : ℕ), p > 2 → (is_good (2 * p) ↔ Nat.Prime p ∧ Nat.Prime (2 * p + 1)) :=
sorry

end NUMINAMATH_CALUDE_good_number_theorem_l2241_224169


namespace NUMINAMATH_CALUDE_students_per_group_l2241_224161

theorem students_per_group (total_students : ℕ) (num_teachers : ℕ) 
  (h1 : total_students = 850) (h2 : num_teachers = 23) :
  (total_students / num_teachers : ℕ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_students_per_group_l2241_224161


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l2241_224137

theorem subtraction_of_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l2241_224137


namespace NUMINAMATH_CALUDE_compute_expression_l2241_224185

theorem compute_expression : 
  20 * (150 / 3 + 50 / 6 + 16 / 25 + 2) = 90460 / 75 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2241_224185


namespace NUMINAMATH_CALUDE_negative_three_to_zero_power_l2241_224178

theorem negative_three_to_zero_power : (-3 : ℤ) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_to_zero_power_l2241_224178


namespace NUMINAMATH_CALUDE_platform_length_l2241_224162

/-- Given a train of length 600 meters that takes 54 seconds to cross a platform
    and 36 seconds to cross a signal pole, the length of the platform is 300 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 600 →
  time_platform = 54 →
  time_pole = 36 →
  ∃ platform_length : ℝ,
    platform_length = 300 ∧
    train_length / time_pole = (train_length + platform_length) / time_platform :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2241_224162


namespace NUMINAMATH_CALUDE_train_crossing_time_l2241_224164

/-- Proves that a train crossing a platform of equal length in 40 seconds will cross a signal pole in 20 seconds -/
theorem train_crossing_time (train_length platform_length : ℝ) 
  (platform_crossing_time : ℝ) (h1 : train_length = 250) 
  (h2 : platform_length = 250) (h3 : platform_crossing_time = 40) : 
  train_length / ((train_length + platform_length) / platform_crossing_time) = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2241_224164


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l2241_224127

/-- A parallelogram in 2D space --/
structure Parallelogram where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  d : ℝ × ℝ

/-- The property that defines a parallelogram --/
def isParallelogram (p : Parallelogram) : Prop :=
  (p.a.1 + p.c.1 = p.b.1 + p.d.1) ∧ 
  (p.a.2 + p.c.2 = p.b.2 + p.d.2)

theorem parallelogram_fourth_vertex 
  (p : Parallelogram)
  (h1 : p.a = (-1, 0))
  (h2 : p.b = (3, 0))
  (h3 : p.c = (1, -5))
  (h4 : isParallelogram p) :
  p.d = (1, 5) ∨ p.d = (-3, -5) := by
  sorry

#check parallelogram_fourth_vertex

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l2241_224127


namespace NUMINAMATH_CALUDE_second_player_wins_l2241_224174

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the state of the game -/
structure GameState :=
  (white_rook : Position)
  (black_rook : Position)
  (visited : Set Position)
  (current_player : Bool)  -- true for White, false for Black

/-- Checks if a move is valid according to the game rules -/
def is_valid_move (state : GameState) (new_pos : Position) : Bool :=
  -- Implementation details omitted
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Option Position

/-- Checks if a strategy is a winning strategy for the given player -/
def is_winning_strategy (strategy : Strategy) (player : Bool) : Prop :=
  -- Implementation details omitted
  sorry

/-- The main theorem stating that the second player (Black) has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy : Strategy), 
    is_winning_strategy strategy false ∧
    strategy { 
      white_rook := { row := 1, col := 1 },  -- b2
      black_rook := { row := 3, col := 2 },  -- c4
      visited := { { row := 1, col := 1 }, { row := 3, col := 2 } },
      current_player := true
    } ≠ none :=
  sorry

end NUMINAMATH_CALUDE_second_player_wins_l2241_224174


namespace NUMINAMATH_CALUDE_dodecagon_triangles_l2241_224146

/-- A regular dodecagon is a 12-sided polygon. -/
def regular_dodecagon : ℕ := 12

/-- The number of triangles that can be formed using the vertices of a regular dodecagon. -/
def num_triangles (n : ℕ) : ℕ := Nat.choose n 3

theorem dodecagon_triangles :
  num_triangles regular_dodecagon = 220 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_triangles_l2241_224146


namespace NUMINAMATH_CALUDE_watermelon_not_necessarily_split_l2241_224103

/-- Represents a spherical watermelon with given diameter and cut depth. -/
structure Watermelon where
  diameter : ℝ
  cut_depth : ℝ

/-- Determines if the watermelon is necessarily split into at least two pieces. -/
def is_necessarily_split (w : Watermelon) : Prop :=
  ∃ (configuration : ℝ → ℝ → ℝ → Prop),
    ∀ (x y z : ℝ),
      configuration x y z →
      (x^2 + y^2 + z^2 ≤ (w.diameter/2)^2) →
      (|x| ≤ w.cut_depth ∨ |y| ≤ w.cut_depth ∨ |z| ≤ w.cut_depth)

/-- Theorem stating that a watermelon with diameter 20 cm is not necessarily split
    for cut depths of 17 cm and 18 cm. -/
theorem watermelon_not_necessarily_split :
  let w₁ : Watermelon := ⟨20, 17⟩
  let w₂ : Watermelon := ⟨20, 18⟩
  ¬(is_necessarily_split w₁) ∧ ¬(is_necessarily_split w₂) := by
  sorry

end NUMINAMATH_CALUDE_watermelon_not_necessarily_split_l2241_224103


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2241_224182

theorem problem_1 (a b : ℚ) (h1 : a = -1/2) (h2 : b = -1) :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) + a * b^2 = -3/4 := by
  sorry

theorem problem_2 (x y : ℝ) (h : |2*x - 1| + (3*y + 2)^2 = 0) :
  5 * x^2 - (2*x*y - 3 * (1/3 * x*y + 2) + 5 * x^2) = 19/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2241_224182


namespace NUMINAMATH_CALUDE_negative_integer_squared_plus_self_equals_twenty_l2241_224190

theorem negative_integer_squared_plus_self_equals_twenty (N : ℤ) : 
  N < 0 → 2 * N^2 + N = 20 → N = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_squared_plus_self_equals_twenty_l2241_224190


namespace NUMINAMATH_CALUDE_leaves_first_hour_is_seven_l2241_224193

/-- The number of leaves that fell in the first hour -/
def leaves_first_hour : ℕ := 7

/-- The total number of hours -/
def total_hours : ℕ := 3

/-- The rate of leaves falling per hour in the second and third hour -/
def rate_later_hours : ℕ := 4

/-- The average number of leaves that fell per hour over the entire period -/
def average_leaves_per_hour : ℕ := 5

/-- Theorem stating that the number of leaves that fell in the first hour is 7 -/
theorem leaves_first_hour_is_seven :
  leaves_first_hour = 
    total_hours * average_leaves_per_hour - rate_later_hours * (total_hours - 1) :=
by sorry

end NUMINAMATH_CALUDE_leaves_first_hour_is_seven_l2241_224193


namespace NUMINAMATH_CALUDE_one_more_tile_possible_exists_blocking_configuration_l2241_224101

/-- Represents a 4 × 6 grid -/
def Grid := Fin 4 → Fin 6 → Bool

/-- Represents an L-shaped tile -/
structure LTile :=
  (pos : Fin 4 × Fin 6)

/-- Checks if a tile placement is valid -/
def is_valid_placement (g : Grid) (t : LTile) : Prop :=
  sorry

/-- Places a tile on the grid -/
def place_tile (g : Grid) (t : LTile) : Grid :=
  sorry

/-- Theorem: After placing two tiles, one more can always be placed -/
theorem one_more_tile_possible (g : Grid) (t1 t2 : LTile) 
  (h1 : is_valid_placement g t1)
  (h2 : is_valid_placement (place_tile g t1) t2) :
  ∃ t3 : LTile, is_valid_placement (place_tile (place_tile g t1) t2) t3 :=
sorry

/-- Theorem: There exists a configuration of three tiles that blocks further placement -/
theorem exists_blocking_configuration :
  ∃ g : Grid, ∃ t1 t2 t3 : LTile,
    is_valid_placement g t1 ∧
    is_valid_placement (place_tile g t1) t2 ∧
    is_valid_placement (place_tile (place_tile g t1) t2) t3 ∧
    ∀ t4 : LTile, ¬is_valid_placement (place_tile (place_tile (place_tile g t1) t2) t3) t4 :=
sorry

end NUMINAMATH_CALUDE_one_more_tile_possible_exists_blocking_configuration_l2241_224101


namespace NUMINAMATH_CALUDE_sum_exterior_angles_regular_pentagon_sum_exterior_angles_regular_pentagon_proof_l2241_224140

/-- The sum of the exterior angles of a regular pentagon is 360 degrees. -/
theorem sum_exterior_angles_regular_pentagon : ℝ :=
  360

/-- A polygon is a closed plane figure with straight sides. -/
def Polygon : Type := sorry

/-- A regular polygon is a polygon with all sides and angles equal. -/
def RegularPolygon (p : Polygon) : Prop := sorry

/-- A pentagon is a polygon with five sides. -/
def Pentagon (p : Polygon) : Prop := sorry

/-- The sum of the exterior angles of any polygon is constant. -/
axiom sum_exterior_angles_constant (p : Polygon) : ℝ

/-- The sum of the exterior angles of any polygon is 360 degrees. -/
axiom sum_exterior_angles_360 (p : Polygon) : sum_exterior_angles_constant p = 360

/-- Theorem: The sum of the exterior angles of a regular pentagon is 360 degrees. -/
theorem sum_exterior_angles_regular_pentagon_proof (p : Polygon) 
  (h1 : RegularPolygon p) (h2 : Pentagon p) : 
  sum_exterior_angles_constant p = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_regular_pentagon_sum_exterior_angles_regular_pentagon_proof_l2241_224140
