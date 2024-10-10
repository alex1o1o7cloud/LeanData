import Mathlib

namespace greatest_sum_consecutive_integers_product_less_500_l2925_292591

/-- The greatest possible sum of two consecutive integers whose product is less than 500 is 43 -/
theorem greatest_sum_consecutive_integers_product_less_500 : 
  (∃ n : ℤ, n * (n + 1) < 500 ∧ 
    ∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧
  (∀ n : ℤ, n * (n + 1) < 500 → n + (n + 1) ≤ 43) :=
by sorry

end greatest_sum_consecutive_integers_product_less_500_l2925_292591


namespace smallest_angle_in_quadrilateral_l2925_292516

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_angle_in_quadrilateral (p q r s : ℕ) : 
  is_prime p → is_prime q → is_prime r → is_prime s →
  p > q → q > r → r > s →
  p + q + r = 270 →
  p + q + r + s = 360 →
  s ≥ 97 :=
sorry

end smallest_angle_in_quadrilateral_l2925_292516


namespace max_value_expression_l2925_292528

theorem max_value_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 10 + 10 * y * z ≤ 5 * Real.sqrt 7 :=
sorry

end max_value_expression_l2925_292528


namespace match_problem_solution_l2925_292535

/-- Represents the number of matches in each pile -/
structure MatchPiles :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Performs the described operations on the piles -/
def performOperations (piles : MatchPiles) : MatchPiles :=
  let step1 := MatchPiles.mk
    (piles.first - piles.second)
    (piles.second + piles.second)
    piles.third
  let step2 := MatchPiles.mk
    step1.first
    (step1.second - step1.third)
    (step1.third + step1.third)
  MatchPiles.mk
    (step2.first + step2.third)
    step2.second
    (step2.third - step2.first)

/-- Theorem stating the solution to the match problem -/
theorem match_problem_solution (piles : MatchPiles) :
  piles.first + piles.second + piles.third = 96 →
  let final := performOperations piles
  final.first = final.second ∧ final.second = final.third →
  piles = MatchPiles.mk 44 28 24 := by
  sorry

end match_problem_solution_l2925_292535


namespace parcel_weight_proof_l2925_292538

theorem parcel_weight_proof (x y z : ℝ) 
  (h1 : x + y = 132)
  (h2 : y + z = 145)
  (h3 : z + x = 150) :
  x + y + z = 213.5 := by
sorry

end parcel_weight_proof_l2925_292538


namespace scientific_notation_of_20160_l2925_292526

theorem scientific_notation_of_20160 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 20160 = a * (10 : ℝ) ^ n ∧ a = 2.016 ∧ n = 4 :=
by sorry

end scientific_notation_of_20160_l2925_292526


namespace rectangle_max_area_l2925_292547

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) :
  2 * x + 2 * y = 40 → x * y ≤ 100 := by
  sorry

end rectangle_max_area_l2925_292547


namespace tiling_condition_l2925_292589

/-- A tile is represented by its dimensions -/
structure Tile :=
  (length : ℕ)
  (width : ℕ)

/-- A grid is represented by its side length -/
structure Grid :=
  (side : ℕ)

/-- Predicate to check if a grid can be tiled by a given tile -/
def can_be_tiled (g : Grid) (t : Tile) : Prop :=
  ∃ (k : ℕ), g.side = k * t.length ∧ g.side * g.side = k * k * (t.length * t.width)

/-- The main theorem stating the condition for tiling an n×n grid with 4×1 tiles -/
theorem tiling_condition (n : ℕ) :
  (∃ (g : Grid) (t : Tile), g.side = n ∧ t.length = 4 ∧ t.width = 1 ∧ can_be_tiled g t) ↔ 
  (∃ (k : ℕ), n = 4 * k) :=
sorry

end tiling_condition_l2925_292589


namespace x_intercept_ratio_l2925_292577

/-- Given two lines with the same non-zero y-intercept and different slopes,
    prove that the ratio of their x-intercepts is 1/2 -/
theorem x_intercept_ratio (b : ℝ) (u v : ℝ) : 
  b ≠ 0 →  -- The common y-intercept is non-zero
  0 = 8 * u + b →  -- First line equation at x-intercept
  0 = 4 * v + b →  -- Second line equation at x-intercept
  u / v = 1 / 2 :=
by sorry

end x_intercept_ratio_l2925_292577


namespace line_intersection_y_coordinate_l2925_292507

/-- Given a line with slope 3/4 passing through (400, 0), 
    prove that the y-coordinate at x = -12 is -309 -/
theorem line_intersection_y_coordinate 
  (slope : ℚ) 
  (x_intercept : ℝ) 
  (x_coord : ℝ) :
  slope = 3/4 →
  x_intercept = 400 →
  x_coord = -12 →
  let y_intercept := -(slope * x_intercept)
  let y_coord := slope * x_coord + y_intercept
  y_coord = -309 := by
sorry

end line_intersection_y_coordinate_l2925_292507


namespace point_relationship_l2925_292523

/-- Given two points A(-1/2, m) and B(2, n) on the line y = 3x + b, prove that m < n -/
theorem point_relationship (m n b : ℝ) : 
  ((-1/2 : ℝ), m) ∈ {(x, y) | y = 3*x + b} →
  ((2 : ℝ), n) ∈ {(x, y) | y = 3*x + b} →
  m < n :=
by sorry

end point_relationship_l2925_292523


namespace min_value_m2_plus_n2_l2925_292584

theorem min_value_m2_plus_n2 (m n : ℝ) (hm : m ≠ 0) :
  let f := λ x : ℝ => m * x^2 + (2*n + 1) * x - m - 2
  (∃ x ∈ Set.Icc 3 4, f x = 0) →
  (∀ a b : ℝ, (∃ x ∈ Set.Icc 3 4, a * x^2 + (2*b + 1) * x - a - 2 = 0) → a^2 + b^2 ≥ 1/100) ∧
  (∃ a b : ℝ, (∃ x ∈ Set.Icc 3 4, a * x^2 + (2*b + 1) * x - a - 2 = 0) ∧ a^2 + b^2 = 1/100) :=
by sorry

end min_value_m2_plus_n2_l2925_292584


namespace min_sides_rotatable_polygon_l2925_292593

theorem min_sides_rotatable_polygon (n : ℕ) (angle : ℚ) : 
  n > 0 ∧ 
  angle = 50 ∧ 
  (360 / n : ℚ) ∣ angle →
  n ≥ 36 :=
sorry

end min_sides_rotatable_polygon_l2925_292593


namespace max_value_expression_l2925_292530

theorem max_value_expression (a b c d : ℝ) 
  (ha : -10.5 ≤ a ∧ a ≤ 10.5)
  (hb : -10.5 ≤ b ∧ b ≤ 10.5)
  (hc : -10.5 ≤ c ∧ c ≤ 10.5)
  (hd : -10.5 ≤ d ∧ d ≤ 10.5) :
  (∀ x y z w, -10.5 ≤ x ∧ x ≤ 10.5 → -10.5 ≤ y ∧ y ≤ 10.5 → 
              -10.5 ≤ z ∧ z ≤ 10.5 → -10.5 ≤ w ∧ w ≤ 10.5 →
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 462) ∧
  (∃ x y z w, -10.5 ≤ x ∧ x ≤ 10.5 ∧ -10.5 ≤ y ∧ y ≤ 10.5 ∧
              -10.5 ≤ z ∧ z ≤ 10.5 ∧ -10.5 ≤ w ∧ w ≤ 10.5 ∧
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 462) :=
by sorry

end max_value_expression_l2925_292530


namespace total_students_in_class_l2925_292508

/-- 
Given a class where 45 students are present when 10% are absent,
prove that the total number of students in the class is 50.
-/
theorem total_students_in_class : 
  ∀ (total : ℕ), 
  (↑total * (1 - 0.1) : ℝ) = 45 → 
  total = 50 := by
sorry

end total_students_in_class_l2925_292508


namespace incorrect_equation_l2925_292501

/-- Represents a decimal number with a non-repeating segment followed by a repeating segment -/
structure DecimalNumber where
  X : ℕ  -- non-repeating segment
  Y : ℕ  -- repeating segment
  u : ℕ  -- number of digits in X
  v : ℕ  -- number of digits in Y

/-- Converts a DecimalNumber to its real value -/
def toReal (z : DecimalNumber) : ℚ :=
  (z.X : ℚ) / 10^z.u + (z.Y : ℚ) / (10^z.u * (10^z.v - 1))

/-- The main theorem stating that the given equation does not hold for all DecimalNumbers -/
theorem incorrect_equation (z : DecimalNumber) : 
  ¬(10^(2*z.u) * (10^z.v - 1) * toReal z = (z.Y : ℚ) * ((z.X : ℚ)^2 - 1)) := by
  sorry

end incorrect_equation_l2925_292501


namespace certain_number_multiplied_l2925_292548

theorem certain_number_multiplied (x : ℝ) : x - 7 = 9 → 3 * x = 48 := by
  sorry

end certain_number_multiplied_l2925_292548


namespace road_length_16_trees_l2925_292596

/-- Calculates the length of a road given the number of trees, space per tree, and space between trees. -/
def roadLength (numTrees : ℕ) (spacePerTree : ℕ) (spaceBetweenTrees : ℕ) : ℕ :=
  numTrees * spacePerTree + (numTrees - 1) * spaceBetweenTrees

/-- Proves that the length of the road with 16 trees, 1 foot per tree, and 9 feet between trees is 151 feet. -/
theorem road_length_16_trees : roadLength 16 1 9 = 151 := by
  sorry

end road_length_16_trees_l2925_292596


namespace harry_stamps_l2925_292529

theorem harry_stamps (total : ℕ) (harry_ratio : ℕ) (harry_stamps : ℕ) : 
  total = 240 →
  harry_ratio = 3 →
  harry_stamps = total * harry_ratio / (harry_ratio + 1) →
  harry_stamps = 180 := by
sorry

end harry_stamps_l2925_292529


namespace sales_tax_rate_is_twenty_percent_l2925_292592

/-- Calculates the sales tax rate given the cost of items and total amount spent --/
def calculate_sales_tax_rate (milk_cost banana_cost total_spent : ℚ) : ℚ :=
  let items_cost := milk_cost + banana_cost
  let tax_amount := total_spent - items_cost
  (tax_amount / items_cost) * 100

theorem sales_tax_rate_is_twenty_percent : 
  calculate_sales_tax_rate 3 2 6 = 20 := by sorry

end sales_tax_rate_is_twenty_percent_l2925_292592


namespace three_lines_intersection_l2925_292518

-- Define the lines
def line1 (x y : ℝ) := x - y + 1 = 0
def line2 (x y : ℝ) := 2*x + y - 4 = 0
def line3 (a x y : ℝ) := a*x - y + 2 = 0

-- Define the condition of exactly two intersection points
def has_two_intersections (a : ℝ) :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    (line1 x1 y1 ∧ line2 x1 y1 ∧ line3 a x1 y1) ∧
    (line1 x2 y2 ∧ line2 x2 y2 ∧ line3 a x2 y2) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (∀ x y, line1 x y ∧ line2 x y ∧ line3 a x y → (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

-- Theorem statement
theorem three_lines_intersection (a : ℝ) :
  has_two_intersections a → a = 1 ∨ a = -2 :=
by sorry

end three_lines_intersection_l2925_292518


namespace age_equality_time_l2925_292512

/-- Given two people a and b, where a is 5 years older than b and their present ages sum to 13,
    this theorem proves that it will take 11 years for thrice a's age to equal 4 times b's age. -/
theorem age_equality_time (a b : ℕ) : 
  a = b + 5 → 
  a + b = 13 → 
  ∃ x : ℕ, x = 11 ∧ 3 * (a + x) = 4 * (b + x) :=
by sorry

end age_equality_time_l2925_292512


namespace instantaneous_velocity_at_5_l2925_292506

noncomputable def s (t : ℝ) : ℝ := (1/4) * t^4 - 3

theorem instantaneous_velocity_at_5 :
  (deriv s) 5 = 125 := by sorry

end instantaneous_velocity_at_5_l2925_292506


namespace sum_of_digits_plus_two_l2925_292513

/-- T(n) represents the sum of the digits of a positive integer n -/
def T (n : ℕ+) : ℕ := sorry

/-- For a certain positive integer n, T(n) = 1598 implies T(n+2) = 1600 -/
theorem sum_of_digits_plus_two (n : ℕ+) (h : T n = 1598) : T (n + 2) = 1600 := by
  sorry

end sum_of_digits_plus_two_l2925_292513


namespace max_value_expression_l2925_292574

theorem max_value_expression (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (a - b^2) * (b - a^2) ≤ 1/16 := by
  sorry

end max_value_expression_l2925_292574


namespace population_change_l2925_292552

theorem population_change (P : ℝ) : 
  P > 0 →
  P * 0.9 * 1.1 * 0.9 = 4455 →
  P = 5000 := by
sorry

end population_change_l2925_292552


namespace parrot_count_l2925_292517

/-- Represents the number of animals in a zoo --/
structure ZooCount where
  parrots : ℕ
  snakes : ℕ
  monkeys : ℕ
  elephants : ℕ
  zebras : ℕ

/-- Checks if the zoo count satisfies the given conditions --/
def isValidZooCount (z : ZooCount) : Prop :=
  z.snakes = 3 * z.parrots ∧
  z.monkeys = 2 * z.snakes ∧
  z.elephants = (z.parrots + z.snakes) / 2 ∧
  z.zebras = z.elephants - 3 ∧
  z.monkeys - z.zebras = 35

/-- Theorem stating that there are 8 parrots in the zoo --/
theorem parrot_count : ∃ z : ZooCount, isValidZooCount z ∧ z.parrots = 8 := by
  sorry

end parrot_count_l2925_292517


namespace toy_shop_spending_l2925_292524

def total_spent (trevor_spending : ℕ) (reed_spending : ℕ) (quinn_spending : ℕ) (years : ℕ) : ℕ :=
  (trevor_spending + reed_spending + quinn_spending) * years

theorem toy_shop_spending (trevor_spending reed_spending quinn_spending : ℕ) :
  trevor_spending = reed_spending + 20 →
  reed_spending = 2 * quinn_spending →
  trevor_spending = 80 →
  total_spent trevor_spending reed_spending quinn_spending 4 = 680 :=
by
  sorry

end toy_shop_spending_l2925_292524


namespace vector_BC_l2925_292500

/-- Given points A and B, and vector AC, prove that vector BC is (-3, 2) -/
theorem vector_BC (A B C : ℝ × ℝ) : 
  A = (-1, 1) → B = (0, 2) → (C.1 - A.1, C.2 - A.2) = (-2, 3) → 
  (C.1 - B.1, C.2 - B.2) = (-3, 2) := by sorry

end vector_BC_l2925_292500


namespace prime_seven_mod_eight_not_sum_three_squares_l2925_292565

theorem prime_seven_mod_eight_not_sum_three_squares (p : ℕ) (hp : Nat.Prime p) (hm : p % 8 = 7) :
  ¬ ∃ (a b c : ℤ), (a * a + b * b + c * c : ℤ) = p := by
  sorry

end prime_seven_mod_eight_not_sum_three_squares_l2925_292565


namespace square_minus_two_x_plus_2023_l2925_292585

theorem square_minus_two_x_plus_2023 :
  let x : ℝ := 1 + Real.sqrt 3
  x^2 - 2*x + 2023 = 2025 := by sorry

end square_minus_two_x_plus_2023_l2925_292585


namespace lcm_problem_l2925_292554

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 84) (h2 : b = 4 * a) (h3 : b = 84) :
  Nat.lcm a b = 21 := by
  sorry

end lcm_problem_l2925_292554


namespace parabola_ellipse_shared_focus_l2925_292562

/-- Given a parabola and an ellipse with shared focus, prove p = 8 -/
theorem parabola_ellipse_shared_focus (p : ℝ) : 
  p > 0 → 
  (∃ x y, y^2 = 2*p*x) →  -- parabola equation
  (∃ x y, x^2/(3*p) + y^2/p = 1) →  -- ellipse equation
  (∃ x, x = p/2 ∧ x^2 = p^2/4) →  -- focus of parabola
  (∃ x, x^2 = 3*p^2/4) →  -- focus of ellipse
  p = 8 := by
sorry

end parabola_ellipse_shared_focus_l2925_292562


namespace fraction_equality_l2925_292521

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = 1 := by
  sorry

end fraction_equality_l2925_292521


namespace john_driving_time_l2925_292540

theorem john_driving_time (speed : ℝ) (time_before_lunch : ℝ) (total_distance : ℝ) :
  speed = 55 →
  time_before_lunch = 2 →
  total_distance = 275 →
  (total_distance - speed * time_before_lunch) / speed = 3 := by
  sorry

end john_driving_time_l2925_292540


namespace optimal_allocation_l2925_292599

/-- Represents the advertising problem for a company --/
structure AdvertisingProblem where
  totalTime : ℝ
  totalBudget : ℝ
  rateA : ℝ
  rateB : ℝ
  revenueA : ℝ
  revenueB : ℝ

/-- Represents an advertising allocation --/
structure Allocation where
  timeA : ℝ
  timeB : ℝ

/-- Calculates the total revenue for a given allocation --/
def totalRevenue (p : AdvertisingProblem) (a : Allocation) : ℝ :=
  p.revenueA * a.timeA + p.revenueB * a.timeB

/-- Checks if an allocation is valid given the problem constraints --/
def isValidAllocation (p : AdvertisingProblem) (a : Allocation) : Prop :=
  a.timeA ≥ 0 ∧ a.timeB ≥ 0 ∧
  a.timeA + a.timeB ≤ p.totalTime ∧
  p.rateA * a.timeA + p.rateB * a.timeB ≤ p.totalBudget

/-- The main theorem stating that the given allocation maximizes revenue --/
theorem optimal_allocation (p : AdvertisingProblem) 
  (h1 : p.totalTime = 300)
  (h2 : p.totalBudget = 90000)
  (h3 : p.rateA = 500)
  (h4 : p.rateB = 200)
  (h5 : p.revenueA = 0.3)
  (h6 : p.revenueB = 0.2) :
  ∃ (a : Allocation),
    isValidAllocation p a ∧
    totalRevenue p a = 70 ∧
    ∀ (b : Allocation), isValidAllocation p b → totalRevenue p b ≤ totalRevenue p a :=
by sorry

end optimal_allocation_l2925_292599


namespace negation_of_existence_quadratic_inequality_negation_l2925_292556

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem quadratic_inequality_negation : 
  (¬ ∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) := by sorry

end negation_of_existence_quadratic_inequality_negation_l2925_292556


namespace sleeves_weight_addition_l2925_292543

theorem sleeves_weight_addition (raw_squat : ℝ) (wrap_percentage : ℝ) (wrap_sleeve_difference : ℝ) 
  (h1 : raw_squat = 600)
  (h2 : wrap_percentage = 0.25)
  (h3 : wrap_sleeve_difference = 120) :
  let squat_with_wraps := raw_squat + wrap_percentage * raw_squat
  let squat_with_sleeves := squat_with_wraps - wrap_sleeve_difference
  squat_with_sleeves - raw_squat = 30 := by
sorry

end sleeves_weight_addition_l2925_292543


namespace stephanie_silverware_l2925_292544

/-- The number of types of silverware Stephanie needs to buy -/
def numTypes : ℕ := 4

/-- The initial number of pieces Stephanie plans to buy for each type -/
def initialPlan : ℕ := 5 + 10

/-- The reduction in the number of spoons and butter knives -/
def reductionSpoonsButter : ℕ := 4

/-- The reduction in the number of steak knives -/
def reductionSteak : ℕ := 5

/-- The reduction in the number of forks -/
def reductionForks : ℕ := 3

/-- The total number of silverware pieces Stephanie will buy -/
def totalSilverware : ℕ := 
  (initialPlan - reductionSpoonsButter) + 
  (initialPlan - reductionSpoonsButter) + 
  (initialPlan - reductionSteak) + 
  (initialPlan - reductionForks)

theorem stephanie_silverware : totalSilverware = 44 := by
  sorry

end stephanie_silverware_l2925_292544


namespace expand_expression_l2925_292534

theorem expand_expression (x : ℝ) : (x - 3) * (x + 6) = x^2 + 3*x - 18 := by
  sorry

end expand_expression_l2925_292534


namespace absolute_value_and_quadratic_equivalence_l2925_292570

theorem absolute_value_and_quadratic_equivalence :
  ∀ b c : ℝ,
  (∀ x : ℝ, |x - 3| = 4 ↔ x^2 + b*x + c = 0) →
  b = -6 ∧ c = -7 := by
sorry

end absolute_value_and_quadratic_equivalence_l2925_292570


namespace z_values_l2925_292537

theorem z_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  let z := (x - 3)^2 * (x + 4) / (3 * x - 4)
  z = 0 ∨ z = 192 := by sorry

end z_values_l2925_292537


namespace call_center_ratio_l2925_292590

theorem call_center_ratio (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  (6 / 5 : ℚ) * a * b = (3 / 4 : ℚ) * b * b → a / b = (5 / 8 : ℚ) := by
  sorry

end call_center_ratio_l2925_292590


namespace floor_of_expression_equals_eight_l2925_292557

theorem floor_of_expression_equals_eight :
  ⌊(1005^3 : ℝ) / (1003 * 1004) - (1003^3 : ℝ) / (1004 * 1005)⌋ = 8 := by
  sorry

end floor_of_expression_equals_eight_l2925_292557


namespace two_consistent_faces_l2925_292527

/-- A graph representing a convex polyhedron -/
structure ConvexPolyhedronGraph where
  V : Type* -- Vertices
  E : Type* -- Edges
  F : Type* -- Faces
  adj : V → List E -- Adjacent edges for each vertex
  face_edges : F → List E -- Edges for each face
  orientation : E → Bool -- Edge orientation (True for outgoing, False for incoming)

/-- Number of changes in edge orientation around a vertex -/
def vertex_orientation_changes (G : ConvexPolyhedronGraph) (v : G.V) : Nat :=
  sorry

/-- Number of changes in edge orientation around a face -/
def face_orientation_changes (G : ConvexPolyhedronGraph) (f : G.F) : Nat :=
  sorry

/-- Main theorem -/
theorem two_consistent_faces (G : ConvexPolyhedronGraph)
  (h1 : ∀ v : G.V, ∃ e1 e2 : G.E, e1 ∈ G.adj v ∧ e2 ∈ G.adj v ∧ G.orientation e1 ≠ G.orientation e2) :
  ∃ f1 f2 : G.F, f1 ≠ f2 ∧ face_orientation_changes G f1 = 0 ∧ face_orientation_changes G f2 = 0 :=
sorry

end two_consistent_faces_l2925_292527


namespace circle_theorem_l2925_292532

-- Define the set of complex numbers satisfying the condition
def S : Set ℂ := {z : ℂ | Complex.abs (z - 3 * Complex.I) = 10}

-- State the theorem
theorem circle_theorem : 
  S = {z : ℂ | Complex.abs (z - Complex.ofReal 0 - Complex.I * 3) = 10} := by
sorry

end circle_theorem_l2925_292532


namespace complement_A_U_equality_l2925_292505

-- Define the universal set U
def U : Set ℝ := {x | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

-- Define the complement of A with respect to U
def complement_A_U : Set ℝ := U \ A

-- Theorem statement
theorem complement_A_U_equality :
  complement_A_U = {x | (-3 < x ∧ x ≤ -2) ∨ (1 < x ∧ x < 3)} := by
  sorry

end complement_A_U_equality_l2925_292505


namespace job_completion_days_l2925_292510

/-- The number of days initially planned for a job to be completed, given:
  * 6 workers start the job
  * After 3 days, 4 more workers join
  * With 10 workers, the job is finished in 3 more days
  * Each worker has the same efficiency -/
def initial_days : ℕ := 6

/-- The total amount of work to be done -/
def total_work : ℝ := 1

/-- The number of workers that start the job -/
def initial_workers : ℕ := 6

/-- The number of days worked before additional workers join -/
def days_before_join : ℕ := 3

/-- The number of additional workers that join -/
def additional_workers : ℕ := 4

/-- The number of days needed to finish the job after additional workers join -/
def days_after_join : ℕ := 3

theorem job_completion_days :
  let work_rate := total_work / initial_days
  let work_done_before_join := days_before_join * work_rate
  let remaining_work := total_work - work_done_before_join
  let total_workers := initial_workers + additional_workers
  remaining_work / days_after_join = total_workers * work_rate
  → initial_days = 6 := by sorry

end job_completion_days_l2925_292510


namespace geometric_sequence_problem_l2925_292531

/-- Represents a geometric sequence with common ratio greater than 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : q > 1
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- Given conditions for the geometric sequence -/
def SequenceConditions (seq : GeometricSequence) : Prop :=
  seq.a 3 * seq.a 7 = 72 ∧ seq.a 2 + seq.a 8 = 27

theorem geometric_sequence_problem (seq : GeometricSequence) 
  (h : SequenceConditions seq) : seq.a 12 = 96 := by
  sorry

end geometric_sequence_problem_l2925_292531


namespace xyz_sum_and_inequality_l2925_292559

theorem xyz_sum_and_inequality (x y z : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_not_all_equal : ¬(x = y ∧ y = z))
  (h_equation : x^3 + y^3 + z^3 - 3*x*y*z - 3*(x^2 + y^2 + z^2 - x*y - y*z - z*x) = 0) :
  (x + y + z = 3) ∧ (x^2*(1 + y) + y^2*(1 + z) + z^2*(1 + x) > 6) := by
  sorry

end xyz_sum_and_inequality_l2925_292559


namespace regular_nonagon_angle_l2925_292572

/-- A regular nonagon inscribed in a circle -/
structure RegularNonagon :=
  (vertices : Fin 9 → ℝ × ℝ)
  (is_regular : ∀ i j : Fin 9, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))
  (is_inscribed : ∃ center : ℝ × ℝ, ∀ i : Fin 9, dist center (vertices i) = dist center (vertices 0))

/-- The angle measure between three consecutive vertices of a regular nonagon -/
def angle_measure (n : RegularNonagon) (i : Fin 9) : ℝ :=
  sorry

/-- Theorem: The angle measure between three consecutive vertices of a regular nonagon is 40 degrees -/
theorem regular_nonagon_angle (n : RegularNonagon) (i : Fin 9) :
  angle_measure n i = 40 := by sorry

end regular_nonagon_angle_l2925_292572


namespace equation_root_interval_l2925_292575

-- Define the function f(x) = lg(x+1) + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2 + x - 3

-- State the theorem
theorem equation_root_interval :
  ∃ (x : ℝ), x ∈ Set.Ioo 2 3 ∧ f x = 0 ∧
  ∀ (k : ℤ), (∃ (y : ℝ), y ∈ Set.Ioo (k : ℝ) (k + 1) ∧ f y = 0) → k = 2 := by
  sorry

end equation_root_interval_l2925_292575


namespace smallest_digit_not_in_odd_units_l2925_292588

def odd_units_digits : Set Nat := {1, 3, 5, 7, 9}

def is_digit (n : Nat) : Prop := n < 10

theorem smallest_digit_not_in_odd_units : 
  (∀ d, is_digit d → d ∉ odd_units_digits → d ≥ 0) ∧ 
  (0 ∉ odd_units_digits) ∧ 
  is_digit 0 :=
sorry

end smallest_digit_not_in_odd_units_l2925_292588


namespace coca_cola_purchase_l2925_292568

/-- The number of bottles of Coca-Cola to be purchased. -/
def num_bottles : ℕ := 40

/-- The price of each bottle of Coca-Cola in yuan. -/
def price_per_bottle : ℚ := 28/10

/-- The denomination of the banknotes in yuan. -/
def banknote_value : ℕ := 20

/-- The minimum number of banknotes needed to cover the total cost. -/
def min_banknotes : ℕ := 6

theorem coca_cola_purchase (n : ℕ) (p : ℚ) (b : ℕ) :
  n = num_bottles → p = price_per_bottle → b = banknote_value →
  min_banknotes = (n * p / b).ceil := by sorry

end coca_cola_purchase_l2925_292568


namespace sum_reciprocals_bound_l2925_292551

theorem sum_reciprocals_bound (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  9 ≤ (1/a + 1/b + 1/c) ∧ 
  ∀ M : ℝ, ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 1 ∧ 1/x + 1/y + 1/z > M :=
by sorry

end sum_reciprocals_bound_l2925_292551


namespace ninth_term_is_zero_l2925_292519

/-- An arithmetic sequence with a₄ = 5 and a₅ = 4 -/
def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a 4 = 5 ∧ a 5 = 4

theorem ninth_term_is_zero (a : ℕ → ℤ) (h : arithmeticSequence a) : a 9 = 0 := by
  sorry

end ninth_term_is_zero_l2925_292519


namespace mancino_garden_length_l2925_292573

theorem mancino_garden_length :
  ∀ (L : ℝ),
  (3 * L * 5 + 2 * 8 * 4 = 304) →
  L = 16 := by
sorry

end mancino_garden_length_l2925_292573


namespace probability_is_four_twentyfirsts_l2925_292555

/-- Represents a person with a unique age --/
structure Person :=
  (age : ℕ)

/-- The set of all possible orderings of people leaving the meeting --/
def Orderings : Type := List Person

/-- Checks if the youngest person leaves before the oldest in a given ordering --/
def youngest_before_oldest (ordering : Orderings) : Prop :=
  sorry

/-- Checks if the 3rd, 4th, and 5th people in the ordering are in ascending age order --/
def middle_three_ascending (ordering : Orderings) : Prop :=
  sorry

/-- The set of all valid orderings (where youngest leaves before oldest) --/
def valid_orderings (people : Finset Person) : Finset Orderings :=
  sorry

/-- The probability of the event occurring --/
def probability (people : Finset Person) : ℚ :=
  sorry

theorem probability_is_four_twentyfirsts 
  (people : Finset Person) 
  (h1 : people.card = 7) 
  (h2 : ∀ p q : Person, p ∈ people → q ∈ people → p ≠ q → p.age ≠ q.age) : 
  probability people = 4 / 21 :=
sorry

end probability_is_four_twentyfirsts_l2925_292555


namespace new_person_weight_l2925_292533

def group_size : ℕ := 8
def average_weight_increase : ℝ := 2.5
def replaced_person_weight : ℝ := 65

theorem new_person_weight (new_weight : ℝ) :
  (group_size : ℝ) * average_weight_increase = new_weight - replaced_person_weight →
  new_weight = 85 := by
sorry

end new_person_weight_l2925_292533


namespace polynomial_not_factorable_l2925_292542

theorem polynomial_not_factorable : ¬ ∃ (a b c d : ℤ),
  ∀ (x : ℝ), x^4 + 3*x^3 + 6*x^2 + 9*x + 12 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end polynomial_not_factorable_l2925_292542


namespace polynomial_division_theorem_l2925_292569

/-- The polynomial P(x) -/
def P (x : ℝ) : ℝ := x^6 - 6*x^4 - 4*x^3 + 9*x^2 + 12*x + 4

/-- The derivative of P(x) -/
def P' (x : ℝ) : ℝ := 6*x^5 - 24*x^3 - 12*x^2 + 18*x + 12

/-- The greatest common divisor of P(x) and P'(x) -/
noncomputable def Q (x : ℝ) : ℝ := x^4 + x^3 - 3*x^2 - 5*x - 2

/-- The resulting polynomial R(x) -/
def R (x : ℝ) : ℝ := x^2 - x - 2

theorem polynomial_division_theorem :
  ∀ x : ℝ, P x = Q x * R x :=
by sorry

end polynomial_division_theorem_l2925_292569


namespace problem_statement_l2925_292536

theorem problem_statement (x y a : ℝ) 
  (h1 : 2^x = a) 
  (h2 : 3^y = a) 
  (h3 : 1/x + 1/y = 2) : 
  a = Real.sqrt 6 := by
  sorry

end problem_statement_l2925_292536


namespace average_headcount_is_11600_l2925_292580

def fall_02_03_headcount : ℕ := 11700
def fall_03_04_headcount : ℕ := 11500
def fall_04_05_headcount : ℕ := 11600

def average_headcount : ℚ :=
  (fall_02_03_headcount + fall_03_04_headcount + fall_04_05_headcount) / 3

theorem average_headcount_is_11600 :
  average_headcount = 11600 := by sorry

end average_headcount_is_11600_l2925_292580


namespace sequence_general_term_l2925_292578

theorem sequence_general_term (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, (2 * n - 1 : ℝ) * a (n + 1) = (2 * n + 1 : ℝ) * a n) →
  ∀ n : ℕ, a n = 2 * n - 1 :=
by sorry

end sequence_general_term_l2925_292578


namespace perfect_square_condition_l2925_292560

theorem perfect_square_condition (n : ℕ) : 
  (∃ m : ℕ, n^4 + 4*n^3 + 5*n^2 + 6*n = m^2) ↔ n = 1 := by
  sorry

end perfect_square_condition_l2925_292560


namespace smallest_number_divisible_l2925_292511

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 7) % 24 = 0 ∧
  (n + 7) % 36 = 0 ∧
  (n + 7) % 50 = 0 ∧
  (n + 7) % 56 = 0 ∧
  (n + 7) % 81 = 0

theorem smallest_number_divisible : 
  is_divisible_by_all 113393 ∧ 
  ∀ m : ℕ, m < 113393 → ¬is_divisible_by_all m :=
sorry

end smallest_number_divisible_l2925_292511


namespace productivity_increase_l2925_292587

theorem productivity_increase (original_hours new_hours : ℝ) 
  (wage_increase : ℝ) (productivity_increase : ℝ) : 
  original_hours = 8 → 
  new_hours = 7 → 
  wage_increase = 0.05 →
  (new_hours / original_hours) * (1 + productivity_increase) = 1 + wage_increase →
  productivity_increase = 0.2 := by
  sorry

end productivity_increase_l2925_292587


namespace inverse_variation_problem_l2925_292583

/-- Given that y varies inversely as x, prove that if y = 6 when x = 3, then y = 3/2 when x = 12 -/
theorem inverse_variation_problem (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, x ≠ 0 → y x = k / x) →  -- y varies inversely as x
  y 3 = 6 →                     -- y = 6 when x = 3
  y 12 = 3 / 2 :=               -- y = 3/2 when x = 12
by
  sorry


end inverse_variation_problem_l2925_292583


namespace fewer_female_students_l2925_292509

theorem fewer_female_students (total_students : ℕ) (female_students : ℕ) 
  (h1 : total_students = 280) (h2 : female_students = 127) :
  total_students - female_students - female_students = 26 := by
  sorry

end fewer_female_students_l2925_292509


namespace shift_right_result_l2925_292563

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Shifts a linear function horizontally -/
def shift_right (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b - f.m * shift }

theorem shift_right_result :
  let original := LinearFunction.mk 2 4
  let shifted := shift_right original 2
  shifted = LinearFunction.mk 2 0 := by sorry

end shift_right_result_l2925_292563


namespace negation_of_no_slow_learners_attend_l2925_292576

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (slow_learner : U → Prop)
variable (attends_school : U → Prop)

-- State the theorem
theorem negation_of_no_slow_learners_attend (h : ¬∃ x, slow_learner x ∧ attends_school x) :
  ∃ x, slow_learner x ∧ attends_school x ↔ ¬(¬∃ x, slow_learner x ∧ attends_school x) :=
by sorry

end negation_of_no_slow_learners_attend_l2925_292576


namespace abs_sum_inequality_l2925_292597

theorem abs_sum_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -9/2 < x ∧ x < 7/2 :=
sorry

end abs_sum_inequality_l2925_292597


namespace odd_function_solution_set_l2925_292564

open Set

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def slope_condition (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 1

theorem odd_function_solution_set 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_slope : slope_condition f) 
  (h_f1 : f 1 = 1) :
  {x : ℝ | f x - x > 0} = Iio (-1) ∪ Ioo 0 1 := by
sorry

end odd_function_solution_set_l2925_292564


namespace sixth_group_52_implies_m_7_l2925_292566

/-- Represents a systematic sampling scheme with the given conditions -/
structure SystematicSampling where
  population : ℕ
  groups : ℕ
  sample_size : ℕ
  first_group_range : Set ℕ
  offset_rule : ℕ → ℕ → ℕ

/-- The specific systematic sampling scheme from the problem -/
def problem_sampling : SystematicSampling :=
  { population := 100
  , groups := 10
  , sample_size := 10
  , first_group_range := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  , offset_rule := λ m k => if m + k < 11 then (m + k - 1) % 10 else (m + k - 11) % 10
  }

/-- The theorem to be proved -/
theorem sixth_group_52_implies_m_7 (s : SystematicSampling) (h : s = problem_sampling) :
  ∃ (m : ℕ), m ∈ s.first_group_range ∧ s.offset_rule m 6 = 2 → m = 7 :=
sorry

end sixth_group_52_implies_m_7_l2925_292566


namespace trig_identity_l2925_292561

theorem trig_identity (α : Real) 
  (h : (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 2) :
  (1 + Real.sin (4 * α) - Real.cos (4 * α)) / 
  (1 + Real.sin (4 * α) + Real.cos (4 * α)) = 3 / 4 := by
  sorry

end trig_identity_l2925_292561


namespace jeans_sold_proof_l2925_292522

/-- The number of pairs of jeans sold by a clothing store -/
def num_jeans : ℕ := 10

theorem jeans_sold_proof (shirts : ℕ) (shirt_price : ℕ) (jeans_price : ℕ) (total_revenue : ℕ) :
  shirts = 20 →
  shirt_price = 10 →
  jeans_price = 2 * shirt_price →
  total_revenue = 400 →
  shirts * shirt_price + num_jeans * jeans_price = total_revenue :=
by sorry

end jeans_sold_proof_l2925_292522


namespace martha_blue_butterflies_l2925_292546

/-- Proves that Martha has 4 blue butterflies given the conditions of the problem -/
theorem martha_blue_butterflies :
  ∀ (total blue yellow black : ℕ),
    total = 11 →
    black = 5 →
    blue = 2 * yellow →
    total = blue + yellow + black →
    blue = 4 :=
by
  sorry

end martha_blue_butterflies_l2925_292546


namespace tennis_preference_theorem_l2925_292581

/-- Represents the percentage of students who prefer tennis -/
def tennis_preference (total : ℕ) (prefer : ℕ) : ℚ :=
  prefer / total

/-- Represents the total number of students who prefer tennis -/
def total_tennis_preference (north_total : ℕ) (north_prefer : ℕ) (south_total : ℕ) (south_prefer : ℕ) : ℕ :=
  north_prefer + south_prefer

/-- Represents the combined percentage of students who prefer tennis -/
def combined_tennis_preference (north_total : ℕ) (north_prefer : ℕ) (south_total : ℕ) (south_prefer : ℕ) : ℚ :=
  tennis_preference (north_total + south_total) (total_tennis_preference north_total north_prefer south_total south_prefer)

theorem tennis_preference_theorem (north_total south_total : ℕ) (north_prefer south_prefer : ℕ) :
  north_total = 1800 →
  south_total = 2700 →
  tennis_preference north_total north_prefer = 30 / 100 →
  tennis_preference south_total south_prefer = 25 / 100 →
  combined_tennis_preference north_total north_prefer south_total south_prefer = 27 / 100 :=
by sorry

end tennis_preference_theorem_l2925_292581


namespace gold_per_hour_l2925_292571

/-- Calculates the amount of gold coins found per hour during a scuba diving expedition. -/
theorem gold_per_hour (hours : ℕ) (chest_coins : ℕ) (num_bags : ℕ) : 
  hours > 0 → 
  chest_coins > 0 → 
  num_bags > 0 → 
  (chest_coins + num_bags * (chest_coins / 2)) / hours = 25 :=
by
  sorry

#check gold_per_hour 8 100 2

end gold_per_hour_l2925_292571


namespace broken_clock_theorem_l2925_292595

/-- Represents the time shown on a clock --/
structure ClockTime where
  hours : ℕ
  minutes : ℕ

/-- Calculates the time shown on the broken clock after a given number of real minutes --/
def brokenClockTime (startTime : ClockTime) (realMinutes : ℕ) : ClockTime :=
  let totalMinutes := startTime.hours * 60 + startTime.minutes + realMinutes * 5 / 4
  { hours := totalMinutes / 60
    minutes := totalMinutes % 60 }

theorem broken_clock_theorem :
  let startTime := ClockTime.mk 14 0
  let realMinutes := 40
  brokenClockTime startTime realMinutes = ClockTime.mk 14 50 :=
by sorry

end broken_clock_theorem_l2925_292595


namespace flower_shop_sales_ratio_l2925_292502

/-- Proves that the ratio of Tuesday's sales to Monday's sales is 3:1 given the conditions of the flower shop's three-day sale. -/
theorem flower_shop_sales_ratio : 
  ∀ (tuesday_sales : ℕ),
  12 + tuesday_sales + tuesday_sales / 3 = 60 →
  tuesday_sales / 12 = 3 := by
  sorry

end flower_shop_sales_ratio_l2925_292502


namespace water_usage_median_and_mode_l2925_292549

def water_usage : List ℝ := [7, 5, 6, 8, 9, 9, 10]

def median (l : List ℝ) : ℝ := sorry

def mode (l : List ℝ) : ℝ := sorry

theorem water_usage_median_and_mode :
  median water_usage = 8 ∧ mode water_usage = 9 := by sorry

end water_usage_median_and_mode_l2925_292549


namespace total_candies_l2925_292553

theorem total_candies (linda_candies chloe_candies olivia_candies : ℕ)
  (h1 : linda_candies = 34)
  (h2 : chloe_candies = 28)
  (h3 : olivia_candies = 43) :
  linda_candies + chloe_candies + olivia_candies = 105 := by
  sorry

end total_candies_l2925_292553


namespace square_sum_difference_l2925_292539

theorem square_sum_difference : 102 * 102 + 98 * 98 = 800 := by
  sorry

end square_sum_difference_l2925_292539


namespace circle_circumference_increase_l2925_292550

/-- Given two circles, where the diameter of the first increases by 2π,
    the proportional increase in the circumference of the second is 2π² -/
theorem circle_circumference_increase (d₁ d₂ : ℝ) : 
  let increase_diameter : ℝ := 2 * Real.pi
  let increase_circumference : ℝ → ℝ := λ x => Real.pi * x
  increase_circumference increase_diameter = 2 * Real.pi^2 :=
by sorry

end circle_circumference_increase_l2925_292550


namespace ski_price_after_discounts_l2925_292525

def original_price : ℝ := 200
def first_discount : ℝ := 0.4
def second_discount : ℝ := 0.2

theorem ski_price_after_discounts :
  let price_after_first := original_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  final_price = 96 := by sorry

end ski_price_after_discounts_l2925_292525


namespace smallest_x_composite_l2925_292567

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def absolute_value (n : ℤ) : ℕ := Int.natAbs n

theorem smallest_x_composite : 
  (∀ x : ℤ, x < 5 → ¬ is_composite (absolute_value (5 * x^2 - 38 * x + 7))) ∧ 
  is_composite (absolute_value (5 * 5^2 - 38 * 5 + 7)) := by
  sorry

end smallest_x_composite_l2925_292567


namespace shortest_distance_to_mount_fuji_l2925_292558

theorem shortest_distance_to_mount_fuji (a b c h : ℝ) : 
  a = 60 → b = 45 → c^2 = a^2 + b^2 → h * c = a * b → h = 36 := by
  sorry

end shortest_distance_to_mount_fuji_l2925_292558


namespace equation_has_real_roots_l2925_292579

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x + 2) := by
  sorry

end equation_has_real_roots_l2925_292579


namespace one_nonnegative_solution_l2925_292520

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 + 6*x = 0 :=
by sorry

end one_nonnegative_solution_l2925_292520


namespace subset_sum_exists_l2925_292541

theorem subset_sum_exists (nums : List ℕ) : 
  nums.length = 100 → 
  (∀ n ∈ nums, n ≤ 100) → 
  nums.sum = 200 → 
  ∃ subset : List ℕ, subset ⊆ nums ∧ subset.sum = 100 := by
sorry

end subset_sum_exists_l2925_292541


namespace octal_subtraction_l2925_292586

/-- Converts a base-8 number to base-10 --/
def octalToDecimal (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- Converts a base-10 number to base-8 --/
def decimalToOctal (n : ℕ) : ℕ :=
  let tens := n / 8
  let ones := n % 8
  tens * 10 + ones

/-- Proves that 346₈ - 255₈ = 71₈ --/
theorem octal_subtraction : decimalToOctal (octalToDecimal 346 - octalToDecimal 255) = 71 := by
  sorry


end octal_subtraction_l2925_292586


namespace geometric_solid_sum_of_edges_l2925_292515

/-- Represents a rectangular solid with sides in geometric progression -/
structure GeometricSolid where
  a : ℝ  -- shortest side
  r : ℝ  -- common ratio
  h : r > 0  -- ensure positive ratio

/-- Volume of a GeometricSolid -/
def volume (s : GeometricSolid) : ℝ := s.a * (s.a * s.r) * (s.a * s.r * s.r)

/-- Surface area of a GeometricSolid -/
def surfaceArea (s : GeometricSolid) : ℝ :=
  2 * (s.a * (s.a * s.r) + s.a * (s.a * s.r * s.r) + (s.a * s.r) * (s.a * s.r * s.r))

/-- Sum of lengths of all edges of a GeometricSolid -/
def sumOfEdges (s : GeometricSolid) : ℝ := 4 * (s.a + (s.a * s.r) + (s.a * s.r * s.r))

/-- Theorem statement -/
theorem geometric_solid_sum_of_edges :
  ∀ s : GeometricSolid,
    volume s = 125 →
    surfaceArea s = 150 →
    sumOfEdges s = 60 := by
  sorry

end geometric_solid_sum_of_edges_l2925_292515


namespace f_local_minimum_at_2_l2925_292545

def f (x : ℝ) := x^3 - 3*x^2 + 1

theorem f_local_minimum_at_2 :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 :=
sorry

end f_local_minimum_at_2_l2925_292545


namespace lucy_current_fish_l2925_292582

/-- The number of fish Lucy wants to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy would have after buying -/
def total_fish_after : ℕ := 280

/-- The current number of fish in Lucy's aquarium -/
def current_fish : ℕ := total_fish_after - fish_to_buy

theorem lucy_current_fish : current_fish = 212 := by
  sorry

end lucy_current_fish_l2925_292582


namespace coin_problem_l2925_292503

/-- Represents the number of different coin values that can be obtained -/
def different_values (five_cent : ℕ) (ten_cent : ℕ) : ℕ :=
  23 - five_cent

/-- Represents the total number of coins -/
def total_coins (five_cent : ℕ) (ten_cent : ℕ) : ℕ :=
  five_cent + ten_cent

theorem coin_problem (five_cent ten_cent : ℕ) :
  total_coins five_cent ten_cent = 12 →
  different_values five_cent ten_cent = 19 →
  ten_cent = 8 := by
  sorry

end coin_problem_l2925_292503


namespace consecutive_points_length_l2925_292594

/-- Given 5 consecutive points on a straight line, prove that ab = 5 -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 * cd
  (e - d = 7) →            -- de = 7
  (c - a = 11) →           -- ac = 11
  (e - a = 20) →           -- ae = 20
  (b - a = 5) :=           -- ab = 5
by sorry

end consecutive_points_length_l2925_292594


namespace simplify_fraction_product_l2925_292598

theorem simplify_fraction_product : (320 : ℚ) / 18 * 9 / 144 * 4 / 5 = 1 / 2 := by
  sorry

end simplify_fraction_product_l2925_292598


namespace dragon_cannot_be_killed_l2925_292514

/-- Represents the possible number of heads Arthur can cut off in a single swipe --/
inductive CutOff
  | fifteen
  | seventeen
  | twenty
  | five

/-- Represents the number of heads that grow back after a cut --/
def regrow (c : CutOff) : ℕ :=
  match c with
  | CutOff.fifteen => 24
  | CutOff.seventeen => 2
  | CutOff.twenty => 14
  | CutOff.five => 17

/-- Represents a single action of cutting off heads and regrowing --/
def action (c : CutOff) : ℤ :=
  match c with
  | CutOff.fifteen => 24 - 15
  | CutOff.seventeen => 2 - 17
  | CutOff.twenty => 14 - 20
  | CutOff.five => 17 - 5

/-- The main theorem stating that it's impossible to kill the dragon --/
theorem dragon_cannot_be_killed :
  ∀ (n : ℕ) (actions : List CutOff),
    (100 + (actions.map action).sum : ℤ) % 3 = 1 :=
by sorry

end dragon_cannot_be_killed_l2925_292514


namespace greatest_integer_with_gcd_6_l2925_292504

theorem greatest_integer_with_gcd_6 :
  ∃ n : ℕ, n < 150 ∧ n.gcd 12 = 6 ∧ ∀ m : ℕ, m < 150 → m.gcd 12 = 6 → m ≤ n :=
by
  -- The proof goes here
  sorry

end greatest_integer_with_gcd_6_l2925_292504
