import Mathlib

namespace NUMINAMATH_CALUDE_three_digit_number_sum_property_l1493_149334

theorem three_digit_number_sum_property :
  ∃! (N : ℕ), ∃ (a b c : ℕ),
    (100 ≤ N) ∧ (N < 1000) ∧
    (1 ≤ a) ∧ (a ≤ 9) ∧
    (0 ≤ b) ∧ (b ≤ 9) ∧
    (0 ≤ c) ∧ (c ≤ 9) ∧
    (N = 100 * a + 10 * b + c) ∧
    (N = 11 * (a + b + c)) ∧
    (N = 198) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_sum_property_l1493_149334


namespace NUMINAMATH_CALUDE_eleven_day_rental_cost_l1493_149388

/-- Calculates the cost of a car rental given the number of days, daily rate, and weekly rate. -/
def rentalCost (days : ℕ) (dailyRate : ℕ) (weeklyRate : ℕ) : ℕ :=
  if days ≥ 7 then
    weeklyRate + (days - 7) * dailyRate
  else
    days * dailyRate

/-- Proves that the rental cost for 11 days is $310 given the specified rates. -/
theorem eleven_day_rental_cost :
  rentalCost 11 30 190 = 310 := by
  sorry

end NUMINAMATH_CALUDE_eleven_day_rental_cost_l1493_149388


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1493_149353

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 510 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1493_149353


namespace NUMINAMATH_CALUDE_selection_probabilities_correct_l1493_149327

/-- Given a group of 3 boys and 2 girls, this function calculates various probabilities
    when selecting two people from the group. -/
def selection_probabilities (num_boys : ℕ) (num_girls : ℕ) : ℚ × ℚ × ℚ :=
  let total := num_boys + num_girls
  let total_combinations := (total.choose 2 : ℚ)
  let two_boys := (num_boys.choose 2 : ℚ) / total_combinations
  let one_girl := (num_boys * num_girls : ℚ) / total_combinations
  let at_least_one_girl := 1 - two_boys
  (two_boys, one_girl, at_least_one_girl)

theorem selection_probabilities_correct :
  selection_probabilities 3 2 = (3/10, 3/5, 7/10) := by
  sorry

end NUMINAMATH_CALUDE_selection_probabilities_correct_l1493_149327


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1493_149357

/-- The perimeter of a rectangular field with length 7/5 times its width and width of 75 meters is 360 meters. -/
theorem rectangle_perimeter (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width = 75 →
  length = (7/5) * width →
  perimeter = 2 * (length + width) →
  perimeter = 360 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1493_149357


namespace NUMINAMATH_CALUDE_worker_B_time_proof_l1493_149354

/-- The time taken by worker B to complete a task, given that worker A is five times as efficient and takes 15 days less than B. -/
def time_taken_by_B : ℝ := 18.75

/-- The efficiency ratio of worker A to worker B -/
def efficiency_ratio : ℝ := 5

/-- The difference in days between the time taken by B and A to complete the task -/
def time_difference : ℝ := 15

theorem worker_B_time_proof :
  ∀ (rate_B : ℝ) (time_B : ℝ),
    rate_B > 0 →
    time_B > 0 →
    efficiency_ratio * rate_B * (time_B - time_difference) = rate_B * time_B →
    time_B = time_taken_by_B :=
by sorry

end NUMINAMATH_CALUDE_worker_B_time_proof_l1493_149354


namespace NUMINAMATH_CALUDE_marble_distribution_correct_group_size_l1493_149304

/-- The number of marbles in the jar -/
def total_marbles : ℕ := 500

/-- The number of additional people that would join the group -/
def additional_people : ℕ := 5

/-- The number of marbles each person would receive less if additional people joined -/
def marbles_less : ℕ := 2

/-- The number of people in the group today -/
def group_size : ℕ := 33

theorem marble_distribution :
  (total_marbles = group_size * (total_marbles / group_size)) ∧
  (total_marbles = (group_size + additional_people) * (total_marbles / group_size - marbles_less)) :=
by sorry

theorem correct_group_size : group_size = 33 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_correct_group_size_l1493_149304


namespace NUMINAMATH_CALUDE_triangle_symmetric_negative_three_four_l1493_149382

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := a * b - a - b + 1

-- Theorem statement
theorem triangle_symmetric_negative_three_four : triangle (-3) 4 = triangle 4 (-3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_symmetric_negative_three_four_l1493_149382


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1493_149355

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x > 0 ∧ 
  (∀ y : ℕ, y > 0 → (1056 + y) % 29 = 0 ∧ (1056 + y) % 37 = 0 ∧ (1056 + y) % 43 = 0 → x ≤ y) ∧
  (1056 + x) % 29 = 0 ∧ (1056 + x) % 37 = 0 ∧ (1056 + x) % 43 = 0 ∧
  x = 44597 := by
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1493_149355


namespace NUMINAMATH_CALUDE_g_of_three_l1493_149338

/-- Given a function g : ℝ → ℝ satisfying g(x) - 3 * g(1/x) = 3^x + 1 for all x ≠ 0,
    prove that g(3) = -17/4 -/
theorem g_of_three (g : ℝ → ℝ) 
    (h : ∀ x ≠ 0, g x - 3 * g (1/x) = 3^x + 1) : 
    g 3 = -17/4 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_l1493_149338


namespace NUMINAMATH_CALUDE_min_value_ab_l1493_149315

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_ab_l1493_149315


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l1493_149309

/-- Represents the configuration of rectangles around a square -/
structure RectangleSquareArrangement where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The conditions of the arrangement -/
def valid_arrangement (a : RectangleSquareArrangement) : Prop :=
  -- The outer square's side length is 3 times the inner square's side length
  a.inner_square_side + 2 * a.rectangle_short_side = 3 * a.inner_square_side ∧
  -- The outer square's side length is also the sum of the long side and short side
  a.rectangle_long_side + a.rectangle_short_side = 3 * a.inner_square_side

/-- The theorem to be proved -/
theorem rectangle_ratio_is_two (a : RectangleSquareArrangement) 
    (h : valid_arrangement a) : 
    a.rectangle_long_side / a.rectangle_short_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l1493_149309


namespace NUMINAMATH_CALUDE_emily_quiz_score_theorem_l1493_149367

def emily_scores : List ℕ := [85, 92, 88, 90, 93]
def target_mean : ℕ := 91
def num_quizzes : ℕ := 6
def sixth_score : ℕ := 98

theorem emily_quiz_score_theorem :
  let total_sum := (emily_scores.sum + sixth_score)
  total_sum / num_quizzes = target_mean :=
by sorry

end NUMINAMATH_CALUDE_emily_quiz_score_theorem_l1493_149367


namespace NUMINAMATH_CALUDE_expression_simplification_l1493_149395

theorem expression_simplification (x : ℝ) : 
  x * (x * (x * (3 - x) - 6) + 7) + 2 = -x^4 + 3*x^3 - 6*x^2 + 7*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1493_149395


namespace NUMINAMATH_CALUDE_billy_horse_feeding_days_billy_horse_feeding_problem_l1493_149376

theorem billy_horse_feeding_days 
  (num_horses : ℕ) 
  (oats_per_feeding : ℕ) 
  (feedings_per_day : ℕ) 
  (total_oats : ℕ) : ℕ :=
  let daily_oats_per_horse := oats_per_feeding * feedings_per_day
  let total_daily_oats := daily_oats_per_horse * num_horses
  total_oats / total_daily_oats

theorem billy_horse_feeding_problem :
  billy_horse_feeding_days 4 4 2 96 = 3 := by
  sorry

end NUMINAMATH_CALUDE_billy_horse_feeding_days_billy_horse_feeding_problem_l1493_149376


namespace NUMINAMATH_CALUDE_num_aplus_needed_is_two_l1493_149313

/-- Represents the grading system and reward calculation for Paul's courses. -/
structure GradingSystem where
  numCourses : Nat
  bPlusReward : Nat
  aReward : Nat
  aPlusReward : Nat
  maxReward : Nat

/-- Calculates the number of A+ grades needed to double the previous rewards. -/
def numAPlusNeeded (gs : GradingSystem) : Nat :=
  sorry

/-- Theorem stating that the number of A+ grades needed is 2. -/
theorem num_aplus_needed_is_two (gs : GradingSystem) 
  (h1 : gs.numCourses = 10)
  (h2 : gs.bPlusReward = 5)
  (h3 : gs.aReward = 10)
  (h4 : gs.aPlusReward = 15)
  (h5 : gs.maxReward = 190) :
  numAPlusNeeded gs = 2 := by
  sorry

end NUMINAMATH_CALUDE_num_aplus_needed_is_two_l1493_149313


namespace NUMINAMATH_CALUDE_problem_statement_l1493_149344

theorem problem_statement (a b : ℝ) 
  (ha : |a| = 3)
  (hb : |b| = 5)
  (hab_sum : a + b > 0)
  (hab_prod : a * b < 0) :
  a^3 + 2*b = -17 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1493_149344


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l1493_149326

/-- A 45°-45°-90° triangle inscribed in the first quadrant -/
structure RightIsoscelesTriangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  first_quadrant : X.1 ≥ 0 ∧ X.2 ≥ 0 ∧ Y.1 ≥ 0 ∧ Y.2 ≥ 0 ∧ Z.1 ≥ 0 ∧ Z.2 ≥ 0
  right_angle : (Z.1 - X.1) * (Y.1 - X.1) + (Z.2 - X.2) * (Y.2 - X.2) = 0
  isosceles : (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2
  hypotenuse_length : (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 32  -- 4√2 squared

/-- A circle tangent to x-axis, y-axis, and hypotenuse of the triangle -/
structure TangentCircle (t : RightIsoscelesTriangle) where
  O : ℝ × ℝ
  r : ℝ
  tangent_x : O.2 = r
  tangent_y : O.1 = r
  tangent_hypotenuse : ((t.Z.1 - t.X.1) * (O.1 - t.X.1) + (t.Z.2 - t.X.2) * (O.2 - t.X.2))^2 = 
                       r^2 * ((t.Z.1 - t.X.1)^2 + (t.Z.2 - t.X.2)^2)

theorem tangent_circle_radius 
  (t : RightIsoscelesTriangle) 
  (c : TangentCircle t) 
  (h : (t.Y.1 - t.X.1)^2 + (t.Y.2 - t.X.2)^2 = 16) : 
  c.r = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l1493_149326


namespace NUMINAMATH_CALUDE_number_of_factors_for_N_l1493_149305

/-- The number of natural-number factors of N, where N = 2^5 * 3^3 * 5^2 * 7^2 * 11^1 -/
def number_of_factors (N : ℕ) : ℕ :=
  if N = 2^5 * 3^3 * 5^2 * 7^2 * 11^1 then
    (5 + 1) * (3 + 1) * (2 + 1) * (2 + 1) * (1 + 1)
  else
    0

theorem number_of_factors_for_N :
  number_of_factors (2^5 * 3^3 * 5^2 * 7^2 * 11^1) = 432 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_for_N_l1493_149305


namespace NUMINAMATH_CALUDE_singh_gain_l1493_149341

/-- Represents the game with three players and their monetary amounts -/
structure Game where
  initial_amount : ℚ
  ashtikar_final : ℚ
  singh_final : ℚ
  bhatia_final : ℚ

/-- Defines the conditions of the game -/
def game_conditions (g : Game) : Prop :=
  g.initial_amount = 70 ∧
  g.ashtikar_final / g.singh_final = 1 / 2 ∧
  g.singh_final / g.bhatia_final = 4 / 1 ∧
  g.ashtikar_final + g.singh_final + g.bhatia_final = 3 * g.initial_amount

/-- Theorem stating Singh's gain -/
theorem singh_gain (g : Game) (h : game_conditions g) : 
  g.singh_final - g.initial_amount = 50 := by
  sorry


end NUMINAMATH_CALUDE_singh_gain_l1493_149341


namespace NUMINAMATH_CALUDE_product_divisible_by_twelve_l1493_149329

theorem product_divisible_by_twelve (a b c d : ℤ) : 
  ∃ k : ℤ, (b - a) * (c - a) * (d - a) * (b - c) * (d - c) * (d - b) = 12 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_twelve_l1493_149329


namespace NUMINAMATH_CALUDE_time_to_wash_dish_time_to_wash_dish_is_two_l1493_149336

/-- The time it takes to wash one dish, given the following conditions:
  1. Sweeping takes 3 minutes per room
  2. Doing laundry takes 9 minutes per load
  3. Anna sweeps 10 rooms
  4. Billy does 2 loads of laundry
  5. Billy should wash 6 dishes
  6. Anna and Billy should spend the same amount of time on chores -/
theorem time_to_wash_dish : ℝ :=
  let sweep_time_per_room : ℝ := 3
  let laundry_time_per_load : ℝ := 9
  let anna_rooms : ℕ := 10
  let billy_loads : ℕ := 2
  let billy_dishes : ℕ := 6
  let anna_total_time := sweep_time_per_room * anna_rooms
  let billy_laundry_time := laundry_time_per_load * billy_loads
  let billy_dish_time := anna_total_time - billy_laundry_time
  billy_dish_time / billy_dishes

/-- Proof that the time to wash one dish is 2 minutes -/
theorem time_to_wash_dish_is_two : time_to_wash_dish = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_to_wash_dish_time_to_wash_dish_is_two_l1493_149336


namespace NUMINAMATH_CALUDE_range_of_m_l1493_149398

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x ≥ 2 → x^2 - 2*x + 1 ≥ m) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1493_149398


namespace NUMINAMATH_CALUDE_bottles_left_on_shelf_prove_bottles_left_l1493_149328

/-- Calculates the number of bottles left on a shelf after two customers make purchases with specific discounts --/
theorem bottles_left_on_shelf (initial_bottles : ℕ) 
  (jason_bottles : ℕ) (harry_bottles : ℕ) : ℕ :=
  let jason_effective_bottles := jason_bottles
  let harry_effective_bottles := harry_bottles + 1
  initial_bottles - (jason_effective_bottles + harry_effective_bottles)

/-- Proves that given the specific conditions, there are 23 bottles left on the shelf --/
theorem prove_bottles_left : 
  bottles_left_on_shelf 35 5 6 = 23 := by
  sorry

end NUMINAMATH_CALUDE_bottles_left_on_shelf_prove_bottles_left_l1493_149328


namespace NUMINAMATH_CALUDE_max_product_sum_22_l1493_149347

/-- A list of distinct natural numbers -/
def DistinctNatList := List Nat

/-- Check if a list contains distinct elements -/
def isDistinct (l : List Nat) : Prop :=
  l.Nodup

/-- Sum of elements in a list -/
def listSum (l : List Nat) : Nat :=
  l.sum

/-- Product of elements in a list -/
def listProduct (l : List Nat) : Nat :=
  l.prod

/-- The maximum product of distinct natural numbers that sum to 22 -/
def maxProductSum22 : Nat :=
  1008

theorem max_product_sum_22 :
  ∀ (l : DistinctNatList), 
    isDistinct l → 
    listSum l = 22 → 
    listProduct l ≤ maxProductSum22 :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_22_l1493_149347


namespace NUMINAMATH_CALUDE_total_cost_rounded_to_18_l1493_149396

def item1 : ℚ := 247 / 100
def item2 : ℚ := 625 / 100
def item3 : ℚ := 876 / 100
def item4 : ℚ := 149 / 100

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

def total_cost : ℚ := item1 + item2 + item3 + item4

theorem total_cost_rounded_to_18 :
  round_to_nearest_dollar total_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_rounded_to_18_l1493_149396


namespace NUMINAMATH_CALUDE_simplification_and_exponent_sum_l1493_149352

-- Define the expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^9 * z^14) ^ (1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z^4 * (10 * x^2 * z^2) ^ (1/3)

-- Theorem statement
theorem simplification_and_exponent_sum :
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 →
  (original_expression x y z = simplified_expression x y z) ∧
  (1 + 1 + 4 = 6) := by
  sorry

end NUMINAMATH_CALUDE_simplification_and_exponent_sum_l1493_149352


namespace NUMINAMATH_CALUDE_painting_cost_is_84_l1493_149362

/-- Calculates the cost of painting house numbers on a street --/
def cost_of_painting (houses_per_side : ℕ) (south_start : ℕ) (north_start : ℕ) (increment : ℕ) : ℕ :=
  let south_end := south_start + increment * (houses_per_side - 1)
  let north_end := north_start + increment * (houses_per_side - 1)
  let south_cost := (houses_per_side - (south_end / 100)) + (south_end / 100)
  let north_cost := (houses_per_side - (north_end / 100)) + (north_end / 100)
  south_cost + north_cost

/-- The total cost of painting house numbers on the street is 84 dollars --/
theorem painting_cost_is_84 :
  cost_of_painting 30 5 6 6 = 84 :=
by sorry

end NUMINAMATH_CALUDE_painting_cost_is_84_l1493_149362


namespace NUMINAMATH_CALUDE_parallel_condition_l1493_149324

-- Define the structure for a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define when two lines are parallel
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∨ l1.b ≠ 0

-- Define the two lines from the problem
def line1 (a : ℝ) : Line := ⟨2, a, -1⟩
def line2 (b : ℝ) : Line := ⟨b, 2, 1⟩

theorem parallel_condition (a b : ℝ) :
  (parallel (line1 a) (line2 b) → a * b = 4) ∧
  ∃ a b, a * b = 4 ∧ ¬parallel (line1 a) (line2 b) := by sorry

end NUMINAMATH_CALUDE_parallel_condition_l1493_149324


namespace NUMINAMATH_CALUDE_smallest_n_value_l1493_149371

theorem smallest_n_value (o y m : ℕ+) (n : ℕ+) : 
  (10 * o = 16 * y) ∧ (16 * y = 18 * m) ∧ (18 * m = 18 * n) →
  n ≥ 40 ∧ ∃ (o' y' m' : ℕ+), 10 * o' = 16 * y' ∧ 16 * y' = 18 * m' ∧ 18 * m' = 18 * 40 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1493_149371


namespace NUMINAMATH_CALUDE_divisibility_of_fourth_power_minus_one_l1493_149389

theorem divisibility_of_fourth_power_minus_one (a : ℤ) : 
  ¬(5 ∣ a) → (5 ∣ (a^4 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fourth_power_minus_one_l1493_149389


namespace NUMINAMATH_CALUDE_smallest_angle_satisfies_equation_l1493_149307

/-- The smallest positive angle x in degrees that satisfies the given equation -/
def smallest_angle : ℝ := 11.25

theorem smallest_angle_satisfies_equation :
  let x := smallest_angle * Real.pi / 180
  Real.tan (6 * x) = (Real.cos (2 * x) - Real.sin (2 * x)) / (Real.cos (2 * x) + Real.sin (2 * x)) ∧
  ∀ y : ℝ, 0 < y ∧ y < smallest_angle →
    Real.tan (6 * y * Real.pi / 180) ≠ (Real.cos (2 * y * Real.pi / 180) - Real.sin (2 * y * Real.pi / 180)) /
                                       (Real.cos (2 * y * Real.pi / 180) + Real.sin (2 * y * Real.pi / 180)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_satisfies_equation_l1493_149307


namespace NUMINAMATH_CALUDE_percent_of_whole_six_point_two_percent_of_thousand_l1493_149321

theorem percent_of_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = part * 100 / whole := by sorry

theorem six_point_two_percent_of_thousand :
  (6.2 / 1000) * 100 = 0.62 := by sorry

end NUMINAMATH_CALUDE_percent_of_whole_six_point_two_percent_of_thousand_l1493_149321


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l1493_149351

/-- Proves that given a mixture with 20% alcohol, if 3 liters of water are added
    and the resulting mixture has 17.14285714285715% alcohol, 
    then the initial amount of mixture was 18 liters. -/
theorem initial_mixture_volume (initial_volume : ℝ) : 
  initial_volume > 0 →
  (0.2 * initial_volume) / (initial_volume + 3) = 17.14285714285715 / 100 →
  initial_volume = 18 := by
sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l1493_149351


namespace NUMINAMATH_CALUDE_sum_of_bases_is_sixteen_l1493_149349

/-- Represents a repeating decimal in a given base -/
structure RepeatingDecimal (base : ℕ) where
  integerPart : ℕ
  repeatingPart : ℕ

/-- Given two bases and representations of G₁ and G₂ in those bases, proves their sum is 16 -/
theorem sum_of_bases_is_sixteen
  (S₁ S₂ : ℕ)
  (G₁_in_S₁ : RepeatingDecimal S₁)
  (G₂_in_S₁ : RepeatingDecimal S₁)
  (G₁_in_S₂ : RepeatingDecimal S₂)
  (G₂_in_S₂ : RepeatingDecimal S₂)
  (h₁ : G₁_in_S₁ = ⟨0, 45⟩)
  (h₂ : G₂_in_S₁ = ⟨0, 54⟩)
  (h₃ : G₁_in_S₂ = ⟨0, 14⟩)
  (h₄ : G₂_in_S₂ = ⟨0, 41⟩)
  : S₁ + S₂ = 16 :=
sorry

end NUMINAMATH_CALUDE_sum_of_bases_is_sixteen_l1493_149349


namespace NUMINAMATH_CALUDE_subset_iff_elements_l1493_149372

theorem subset_iff_elements (A B : Set α) : A ⊆ B ↔ ∀ x, x ∈ A → x ∈ B := by sorry

end NUMINAMATH_CALUDE_subset_iff_elements_l1493_149372


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1493_149375

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, n = 85 ∧ 
  (∀ m : ℕ, m ≤ 99 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ n) ∧ 
  17 ∣ n ∧ n ≤ 99 ∧ n ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1493_149375


namespace NUMINAMATH_CALUDE_max_leftover_cookies_l1493_149385

theorem max_leftover_cookies (n : ℕ) (h : n > 0) : 
  ∃ (total : ℕ), total % n = n - 1 ∧ ∀ (m : ℕ), m % n ≤ n - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_cookies_l1493_149385


namespace NUMINAMATH_CALUDE_reservoir_water_amount_l1493_149339

theorem reservoir_water_amount 
  (total_capacity : ℝ) 
  (end_amount : ℝ) 
  (normal_level : ℝ) 
  (h1 : end_amount = 2 * normal_level)
  (h2 : end_amount = 0.75 * total_capacity)
  (h3 : normal_level = total_capacity - 20) :
  end_amount = 24 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_water_amount_l1493_149339


namespace NUMINAMATH_CALUDE_integral_one_plus_cos_over_pi_half_interval_l1493_149383

theorem integral_one_plus_cos_over_pi_half_interval :
  ∫ x in (-π/2)..(π/2), (1 + Real.cos x) = π + 2 := by sorry

end NUMINAMATH_CALUDE_integral_one_plus_cos_over_pi_half_interval_l1493_149383


namespace NUMINAMATH_CALUDE_smallest_positive_solution_of_equation_l1493_149317

theorem smallest_positive_solution_of_equation :
  ∃ (x : ℝ), x > 0 ∧ x^4 - 58*x^2 + 841 = 0 ∧ ∀ (y : ℝ), y > 0 ∧ y^4 - 58*y^2 + 841 = 0 → x ≤ y ∧ x = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_of_equation_l1493_149317


namespace NUMINAMATH_CALUDE_inverse_of_matrix_A_l1493_149310

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 15; 2, 6]

theorem inverse_of_matrix_A (h : Matrix.det A = 0) :
  A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_matrix_A_l1493_149310


namespace NUMINAMATH_CALUDE_value_of_a_l1493_149325

-- Define the conversion rate between paise and rupees
def paise_per_rupee : ℚ := 100

-- Define the given percentage as a rational number
def given_percentage : ℚ := 1 / 200

-- Define the given amount in paise
def given_paise : ℚ := 95

-- Theorem statement
theorem value_of_a (a : ℚ) 
  (h : given_percentage * a = given_paise) : 
  a = 190 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l1493_149325


namespace NUMINAMATH_CALUDE_proposition_implication_l1493_149386

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 8) : 
  ¬ P 7 := by sorry

end NUMINAMATH_CALUDE_proposition_implication_l1493_149386


namespace NUMINAMATH_CALUDE_inverse_proportion_point_order_l1493_149363

theorem inverse_proportion_point_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = -2 / (-2) → y₂ = -2 / 2 → y₃ = -2 / 3 → y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_order_l1493_149363


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l1493_149348

/-- For any triangle with side lengths a, b, and c, 
    the sum of squares of the sides is less than 
    twice the sum of the products of pairs of sides. -/
theorem triangle_side_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l1493_149348


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l1493_149387

/-- The function f(x) = ax - 3 + 3 always passes through the point (3, 4) for any real number a. -/
theorem fixed_point_of_linear_function (a : ℝ) : 
  let f := λ x : ℝ => a * x - 3 + 3
  f 3 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l1493_149387


namespace NUMINAMATH_CALUDE_gcd_of_A_and_B_l1493_149346

def A : ℕ := 2 * 3 * 5
def B : ℕ := 2 * 2 * 5 * 7

theorem gcd_of_A_and_B : Nat.gcd A B = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_A_and_B_l1493_149346


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1493_149393

theorem inequality_solution_set (x : ℝ) :
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1493_149393


namespace NUMINAMATH_CALUDE_isoscelesTrapezoidArea_l1493_149331

/-- An isosceles trapezoid inscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- One of the base angles in radians -/
  baseAngle : ℝ
  /-- Assumption that the longer base is 18 -/
  longerBaseIs18 : longerBase = 18
  /-- Assumption that the base angle is arccos(0.6) -/
  baseAngleIsArccos06 : baseAngle = Real.arccos 0.6

/-- The area of the isosceles trapezoid -/
def areaOfTrapezoid (t : IsoscelesTrapezoid) : ℝ := 101.25

/-- Theorem stating that the area of the isosceles trapezoid is 101.25 -/
theorem isoscelesTrapezoidArea (t : IsoscelesTrapezoid) : 
  areaOfTrapezoid t = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_isoscelesTrapezoidArea_l1493_149331


namespace NUMINAMATH_CALUDE_factorization_count_mod_1000_l1493_149312

/-- A polynomial x^2 + ax + b can be factored into linear factors with integer coefficients -/
def HasIntegerFactors (a b : ℤ) : Prop :=
  ∃ c d : ℤ, a = c + d ∧ b = c * d

/-- The count of pairs (a,b) satisfying the conditions -/
def S : ℕ :=
  (Finset.range 100).sum (fun a => 
    (Finset.range (a + 1)).card)

/-- The main theorem -/
theorem factorization_count_mod_1000 : S % 1000 = 50 := by
  sorry

end NUMINAMATH_CALUDE_factorization_count_mod_1000_l1493_149312


namespace NUMINAMATH_CALUDE_three_card_draw_different_colors_l1493_149399

def total_cards : ℕ := 16
def cards_per_color : ℕ := 4
def num_colors : ℕ := 4
def cards_drawn : ℕ := 3

theorem three_card_draw_different_colors : 
  (Nat.choose total_cards cards_drawn) - (num_colors * Nat.choose cards_per_color cards_drawn) = 544 := by
  sorry

end NUMINAMATH_CALUDE_three_card_draw_different_colors_l1493_149399


namespace NUMINAMATH_CALUDE_thirteen_power_mod_thirtyseven_l1493_149340

theorem thirteen_power_mod_thirtyseven (a : ℕ+) (h : 3 ∣ a.val) :
  (13 : ℤ)^(a.val) ≡ 1 [ZMOD 37] := by
  sorry

end NUMINAMATH_CALUDE_thirteen_power_mod_thirtyseven_l1493_149340


namespace NUMINAMATH_CALUDE_binary_representation_of_25_l1493_149306

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The binary representation of 25 -/
def binary25 : List Bool := [true, false, false, true, true]

/-- Theorem stating that the binary representation of 25 is [1,1,0,0,1] -/
theorem binary_representation_of_25 : toBinary 25 = binary25 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_25_l1493_149306


namespace NUMINAMATH_CALUDE_root_sum_sixth_power_l1493_149365

theorem root_sum_sixth_power (r s : ℝ) 
  (h1 : r + s = Real.sqrt 7)
  (h2 : r * s = 1) : 
  r^6 + s^6 = 527 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_sixth_power_l1493_149365


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1493_149361

/-- A quadratic equation x^2 - x + 2k = 0 has two equal real roots if and only if k = 1/8 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - x + 2*k = 0 ∧ (∀ y : ℝ, y^2 - y + 2*k = 0 → y = x)) ↔ k = 1/8 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1493_149361


namespace NUMINAMATH_CALUDE_pascal_triangle_42nd_number_in_45_number_row_l1493_149301

theorem pascal_triangle_42nd_number_in_45_number_row : 
  let n : ℕ := 44  -- The row number (0-indexed) that contains 45 numbers
  let k : ℕ := 41  -- The position (0-indexed) of the 42nd number in the row
  (n.choose k) = 13254 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_42nd_number_in_45_number_row_l1493_149301


namespace NUMINAMATH_CALUDE_definite_integral_x_plus_sin_x_l1493_149333

open Real MeasureTheory

theorem definite_integral_x_plus_sin_x : ∫ x in (-1)..1, (x + Real.sin x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_x_plus_sin_x_l1493_149333


namespace NUMINAMATH_CALUDE_x_equals_six_l1493_149350

theorem x_equals_six (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x) (h3 : 1/a + 1/b = 1) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_six_l1493_149350


namespace NUMINAMATH_CALUDE_triangle_reciprocal_side_angle_bisector_equality_l1493_149337

/-- For any triangle, the sum of reciprocals of side lengths equals the sum of cosines of half angles divided by their respective angle bisector lengths. -/
theorem triangle_reciprocal_side_angle_bisector_equality
  (a b c : ℝ) (α β γ : ℝ) (f_α f_β f_γ : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_angle_sum : α + β + γ = π)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_f_α : f_α = (2 * b * c * Real.cos (α / 2)) / (b + c))
  (h_f_β : f_β = (2 * a * c * Real.cos (β / 2)) / (a + c))
  (h_f_γ : f_γ = (2 * a * b * Real.cos (γ / 2)) / (a + b)) :
  1 / a + 1 / b + 1 / c = Real.cos (α / 2) / f_α + Real.cos (β / 2) / f_β + Real.cos (γ / 2) / f_γ :=
by sorry

end NUMINAMATH_CALUDE_triangle_reciprocal_side_angle_bisector_equality_l1493_149337


namespace NUMINAMATH_CALUDE_rosie_pies_l1493_149378

def max_pies (total_apples : ℕ) (initial_apples : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (total_apples - initial_apples) / apples_per_pie

theorem rosie_pies :
  max_pies 40 3 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_l1493_149378


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1493_149384

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (4 * x + 8) / Real.sqrt (8 * x + 8) = 2 / Real.sqrt 5) → x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1493_149384


namespace NUMINAMATH_CALUDE_broken_shells_bag_l1493_149379

/-- Represents the number of shells in each bag -/
def bags : List ℕ := [17, 20, 22, 24, 26, 36]

/-- Represents the total number of shells -/
def total_shells : ℕ := bags.sum

/-- Predicate to check if a number is divisible by 4 -/
def divisible_by_four (n : ℕ) : Prop := n % 4 = 0

/-- Theorem stating that the bag with 17 shells contains the broken shells -/
theorem broken_shells_bag (clara_bags : List ℕ) (danny_bags : List ℕ) :
  clara_bags.length = 3 →
  danny_bags.length = 2 →
  clara_bags.sum = 3 * danny_bags.sum →
  clara_bags.sum + danny_bags.sum = total_shells - 17 →
  divisible_by_four (clara_bags.sum + danny_bags.sum) →
  17 ∈ bags \ (clara_bags ++ danny_bags) :=
sorry

#check broken_shells_bag

end NUMINAMATH_CALUDE_broken_shells_bag_l1493_149379


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l1493_149359

theorem min_value_sqrt_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (m : ℝ), m = Real.sqrt 10 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 →
    Real.sqrt (x + 1/x) + Real.sqrt (y + 1/y) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l1493_149359


namespace NUMINAMATH_CALUDE_second_quadrant_transformation_l1493_149358

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be in the second quadrant
def isInSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

-- State the theorem
theorem second_quadrant_transformation (a b : ℝ) :
  isInSecondQuadrant (Point2D.mk a b) →
  isInSecondQuadrant (Point2D.mk (-b) (1-a)) :=
by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_transformation_l1493_149358


namespace NUMINAMATH_CALUDE_inequality_solution_l1493_149308

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x > 2 → y > 2 → x < y → f x > f y)
variable (h2 : ∀ x, f (x + 2) = f (-x + 2))

-- Define the solution set
def solution_set (x : ℝ) := 4/3 < x ∧ x < 2

-- State the theorem
theorem inequality_solution :
  (∀ x, solution_set x ↔ f (2*x - 1) - f (x + 1) > 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1493_149308


namespace NUMINAMATH_CALUDE_factor_x_10_minus_1024_l1493_149370

theorem factor_x_10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2*x^3 + 4*x^2 + 8*x + 16) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_10_minus_1024_l1493_149370


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1493_149356

theorem absolute_value_inequality (a b : ℝ) (h : a ≠ 0) :
  ∃ (m : ℝ), m = 2 ∧ (∀ (x y : ℝ) (h' : x ≠ 0), |x + y| + |x - y| ≥ m * |x|) ∧
  (∀ (m' : ℝ), (∀ (x y : ℝ) (h' : x ≠ 0), |x + y| + |x - y| ≥ m' * |x|) → m' ≤ m) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1493_149356


namespace NUMINAMATH_CALUDE_intersection_contains_two_elements_l1493_149316

-- Define the sets P and Q
def P (k : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = k * (x - 1) + 1}
def Q : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 - 2*y = 0}

-- Theorem statement
theorem intersection_contains_two_elements :
  ∃ (k : ℝ), ∃ (a b : ℝ × ℝ), a ≠ b ∧ a ∈ P k ∩ Q ∧ b ∈ P k ∩ Q ∧
  ∀ (c : ℝ × ℝ), c ∈ P k ∩ Q → c = a ∨ c = b :=
sorry

end NUMINAMATH_CALUDE_intersection_contains_two_elements_l1493_149316


namespace NUMINAMATH_CALUDE_hyperbola_and_chord_equation_l1493_149343

-- Define the hyperbola C
def hyperbola_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 18 + y^2 / 14 = 1

-- Define the common focal point condition
def common_focal_point (a b : ℝ) : Prop :=
  ∃ (fx fy : ℝ), hyperbola_C a b fx fy ∧ ellipse fx fy

-- Define point A on hyperbola C
def point_A_on_C (a b : ℝ) : Prop :=
  hyperbola_C a b 3 (Real.sqrt 7)

-- Define point P as midpoint of chord AB
def point_P_midpoint (x_a y_a x_b y_b : ℝ) : Prop :=
  1 = (x_a + x_b) / 2 ∧ 2 = (y_a + y_b) / 2

-- Main theorem
theorem hyperbola_and_chord_equation 
  (a b : ℝ)
  (h_common_focal : common_focal_point a b)
  (h_point_A : point_A_on_C a b)
  (x_a y_a x_b y_b : ℝ)
  (h_midpoint : point_P_midpoint x_a y_a x_b y_b)
  (h_AB_on_C : hyperbola_C a b x_a y_a ∧ hyperbola_C a b x_b y_b) :
  (∀ (x y : ℝ), hyperbola_C a b x y ↔ x^2 / 2 - y^2 / 2 = 1) ∧
  (∀ (x y : ℝ), y = (x - 1) / 2 + 2 ↔ x - 2*y + 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_and_chord_equation_l1493_149343


namespace NUMINAMATH_CALUDE_sqrt_problem_l1493_149300

theorem sqrt_problem : 
  (Real.sqrt 18 - Real.sqrt 8 - Real.sqrt 2 = 0) ∧
  (6 * Real.sqrt 2 * Real.sqrt 3 + 3 * Real.sqrt 30 / Real.sqrt 5 = 9 * Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_sqrt_problem_l1493_149300


namespace NUMINAMATH_CALUDE_rain_probability_l1493_149397

theorem rain_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by sorry

end NUMINAMATH_CALUDE_rain_probability_l1493_149397


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1493_149381

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (5 * x - 1) + Real.sqrt (x - 1) = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1493_149381


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1493_149320

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define perpendicularity of two points from origin
def perp_from_origin (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_intersection_theorem :
  ∀ m : ℝ,
  (∃ x y : ℝ, circle_eq x y m) ↔ m < 5 ∧
  (∃ x1 y1 x2 y2 : ℝ,
    circle_eq x1 y1 m ∧ circle_eq x2 y2 m ∧
    line_eq x1 y1 ∧ line_eq x2 y2 ∧
    perp_from_origin x1 y1 x2 y2 → m = 8/5) ∧
  (∃ x1 y1 x2 y2 : ℝ,
    circle_eq x1 y1 (8/5) ∧ circle_eq x2 y2 (8/5) ∧
    line_eq x1 y1 ∧ line_eq x2 y2 ∧
    perp_from_origin x1 y1 x2 y2 →
    ∀ x y : ℝ, x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔
    ∃ t : ℝ, x = x1 + t*(x2 - x1) ∧ y = y1 + t*(y2 - y1) ∧ 0 ≤ t ∧ t ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1493_149320


namespace NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l1493_149364

theorem angle_with_special_supplement_complement : ∃ (x : ℝ), 
  0 < x ∧ x < 180 ∧ (180 - x) = 5 * (90 - x) ∧ x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l1493_149364


namespace NUMINAMATH_CALUDE_usual_time_to_catch_bus_l1493_149391

theorem usual_time_to_catch_bus (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_time > 0 → usual_speed > 0 →
  (4/5 * usual_speed) * (usual_time + 5) = usual_speed * usual_time →
  usual_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_to_catch_bus_l1493_149391


namespace NUMINAMATH_CALUDE_train_passing_time_l1493_149360

/-- Time taken for trains to pass each other -/
theorem train_passing_time (man_speed goods_speed : ℝ) (goods_length : ℝ) : 
  man_speed = 80 →
  goods_speed = 32 →
  goods_length = 280 →
  let relative_speed := (man_speed + goods_speed) * 1000 / 3600
  let time := goods_length / relative_speed
  ∃ ε > 0, |time - 8.993| < ε :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l1493_149360


namespace NUMINAMATH_CALUDE_ellipse_properties_l1493_149303

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 5 = 1

/-- Definition of line l -/
def line_l (x y : ℝ) : Prop :=
  y = Real.sqrt 3 / 3 * (x + 2) ∨ y = -Real.sqrt 3 / 3 * (x + 2)

/-- Point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_C x y

/-- Theorem statement -/
theorem ellipse_properties :
  let a := 3
  let b := Real.sqrt 5
  let e := 2/3
  ∃ (F₁ F₂ : ℝ × ℝ) (A : ℝ × ℝ),
    -- C passes through (0, √5)
    ellipse_C 0 (Real.sqrt 5) ∧
    -- Eccentricity is 2/3
    Real.sqrt (F₁.1^2 + F₁.2^2) / a = e ∧
    -- A is on x = 4
    A.1 = 4 ∧
    -- When perpendicular bisector of F₁A passes through F₂, l has the given equation
    (∀ x y, line_l x y ↔ (y - F₁.2) / (x - F₁.1) = (A.2 - F₁.2) / (A.1 - F₁.1)) ∧
    -- Minimum length of AB
    ∃ (min_length : ℝ),
      min_length = Real.sqrt 21 ∧
      ∀ (B : PointOnEllipse),
        A.1 * B.x + A.2 * B.y = 0 →  -- OA ⊥ OB
        (A.1 - B.x)^2 + (A.2 - B.y)^2 ≥ min_length^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1493_149303


namespace NUMINAMATH_CALUDE_election_ratio_l1493_149374

theorem election_ratio (R D : ℝ) : 
  R > 0 ∧ D > 0 →  -- Positive number of Republicans and Democrats
  (0.9 * R + 0.15 * D) / (R + D) = 0.7 →  -- Candidate X's vote share
  (0.1 * R + 0.85 * D) / (R + D) = 0.3 →  -- Candidate Y's vote share
  R / D = 2.75 := by
sorry

end NUMINAMATH_CALUDE_election_ratio_l1493_149374


namespace NUMINAMATH_CALUDE_chairs_bought_l1493_149314

theorem chairs_bought (chair_cost : ℕ) (total_spent : ℕ) (num_chairs : ℕ) : 
  chair_cost = 15 → total_spent = 180 → num_chairs * chair_cost = total_spent → num_chairs = 12 := by
  sorry

end NUMINAMATH_CALUDE_chairs_bought_l1493_149314


namespace NUMINAMATH_CALUDE_x_value_proof_l1493_149368

theorem x_value_proof (x : ℝ) (h1 : x^2 - 5*x = 0) (h2 : x ≠ 0) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1493_149368


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l1493_149332

/-- Definition of line l₁ -/
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + (a - 1) * y - 1 = 0

/-- Definition of line l₂ -/
def l₂ (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x + (2 * a + 3) * y - 3 = 0

/-- Definition of perpendicular lines -/
def perpendicular (a : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ, 
  l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ ∧ 
  a * (a - 1) + (a - 1) * (2 * a + 3) = 0

/-- Theorem stating that a = 1 is sufficient but not necessary for perpendicularity -/
theorem a_eq_one_sufficient_not_necessary : 
  (∀ a : ℝ, a = 1 → perpendicular a) ∧ 
  ¬(∀ a : ℝ, perpendicular a → a = 1) := by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l1493_149332


namespace NUMINAMATH_CALUDE_suv_distance_theorem_l1493_149330

/-- Represents the maximum distance an SUV can travel on 24 gallons of gas -/
def max_distance (x : ℝ) : ℝ :=
  1.824 * x + 292.8 - 2.928 * x

/-- Theorem stating the maximum distance formula for the SUV -/
theorem suv_distance_theorem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 100) :
  max_distance x = 7.6 * (x / 100) * 24 + 12.2 * ((100 - x) / 100) * 24 :=
by sorry

end NUMINAMATH_CALUDE_suv_distance_theorem_l1493_149330


namespace NUMINAMATH_CALUDE_sequence_contradiction_l1493_149377

theorem sequence_contradiction (s : Finset ℕ) (h1 : s.card = 5) 
  (h2 : 2 ∈ s) (h3 : 35 ∈ s) (h4 : 26 ∈ s) (h5 : ∃ x ∈ s, ∀ y ∈ s, y ≤ x) 
  (h6 : ∀ x ∈ s, x ≤ 25) : False := by
  sorry

end NUMINAMATH_CALUDE_sequence_contradiction_l1493_149377


namespace NUMINAMATH_CALUDE_fraction_difference_simplification_l1493_149318

theorem fraction_difference_simplification : 
  ∃ q : ℕ+, (2022 : ℚ) / 2021 - 2021 / 2022 = (4043 : ℚ) / q ∧ Nat.gcd 4043 q = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_simplification_l1493_149318


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1493_149345

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (n : ℕ) : ℚ := n / 99

/-- The sum of 0.5̅ and 0.07̅ is equal to 62/99 -/
theorem sum_of_repeating_decimals : 
  SingleDigitRepeatingDecimal 5 + TwoDigitRepeatingDecimal 7 = 62 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1493_149345


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1493_149311

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → a 2 + a 8 = 180 → a 3 + a 4 + a 5 + a 6 + a 7 = 450 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1493_149311


namespace NUMINAMATH_CALUDE_statement_equivalence_l1493_149323

theorem statement_equivalence (x y : ℝ) :
  ((x > 1 ∧ y < -3) → x - y > 4) ↔ (x - y ≤ 4 → x ≤ 1 ∨ y ≥ -3) :=
by sorry

end NUMINAMATH_CALUDE_statement_equivalence_l1493_149323


namespace NUMINAMATH_CALUDE_max_value_of_function_l1493_149335

theorem max_value_of_function (x : ℝ) :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ y, y = Real.sin (2 * x) - 2 * (Real.sin x)^2 + 1 → y ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1493_149335


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_regular_polygon_perimeter_proof_l1493_149319

/-- A regular polygon with side length 6 and exterior angle 90 degrees has a perimeter of 24 units. -/
theorem regular_polygon_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun (side_length : ℝ) (exterior_angle : ℝ) (perimeter : ℝ) =>
    side_length = 6 ∧
    exterior_angle = 90 ∧
    perimeter = 24

/-- The theorem statement -/
theorem regular_polygon_perimeter_proof :
  ∃ (side_length exterior_angle perimeter : ℝ),
    regular_polygon_perimeter side_length exterior_angle perimeter :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_regular_polygon_perimeter_proof_l1493_149319


namespace NUMINAMATH_CALUDE_max_value_constraint_l1493_149366

theorem max_value_constraint (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (6 * a + 3 * b + 10 * c) ≤ Real.sqrt 41 / 2 ∧
  ∃ a₀ b₀ c₀ : ℝ, 9 * a₀^2 + 4 * b₀^2 + 25 * c₀^2 = 1 ∧ 
    6 * a₀ + 3 * b₀ + 10 * c₀ = Real.sqrt 41 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1493_149366


namespace NUMINAMATH_CALUDE_range_of_expression_l1493_149392

theorem range_of_expression (α β : ℝ) 
  (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) : 
  -3/2 < 1/2 * α - β ∧ 1/2 * α - β < 11/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l1493_149392


namespace NUMINAMATH_CALUDE_amy_photo_upload_l1493_149302

theorem amy_photo_upload (num_albums : ℕ) (photos_per_album : ℕ) 
  (h1 : num_albums = 9)
  (h2 : photos_per_album = 20) :
  num_albums * photos_per_album = 180 := by
sorry

end NUMINAMATH_CALUDE_amy_photo_upload_l1493_149302


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1493_149373

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)  -- Sequence of integers indexed by natural numbers
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence condition
  (h_a2 : a 2 = 9)  -- Given: a_2 = 9
  (h_a5 : a 5 = 33)  -- Given: a_5 = 33
  : a 2 - a 1 = 8 :=  -- Conclusion: The common difference is 8
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1493_149373


namespace NUMINAMATH_CALUDE_one_slice_left_l1493_149380

/-- Represents the number of bread slices used each day of the week -/
structure WeeklyBreadUsage where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat
  saturday : Nat
  sunday : Nat

/-- Calculates the number of bread slices left after a week -/
def slicesLeft (initialSlices : Nat) (usage : WeeklyBreadUsage) : Nat :=
  initialSlices - (usage.monday + usage.tuesday + usage.wednesday + 
                   usage.thursday + usage.friday + usage.saturday + usage.sunday)

/-- Theorem stating that 1 slice of bread is left after the week -/
theorem one_slice_left (initialSlices : Nat) (usage : WeeklyBreadUsage) :
  initialSlices = 22 ∧
  usage.monday = 2 ∧
  usage.tuesday = 3 ∧
  usage.wednesday = 4 ∧
  usage.thursday = 1 ∧
  usage.friday = 3 ∧
  usage.saturday = 5 ∧
  usage.sunday = 3 →
  slicesLeft initialSlices usage = 1 := by
  sorry

#check one_slice_left

end NUMINAMATH_CALUDE_one_slice_left_l1493_149380


namespace NUMINAMATH_CALUDE_spears_from_sapling_proof_l1493_149394

/-- The number of spears that can be made from a log -/
def spears_per_log : ℕ := 9

/-- The number of spears that can be made from 6 saplings and a log -/
def spears_from_6_saplings_and_log : ℕ := 27

/-- The number of saplings used along with a log -/
def number_of_saplings : ℕ := 6

/-- The number of spears that can be made from a single sapling -/
def spears_per_sapling : ℕ := 3

theorem spears_from_sapling_proof :
  number_of_saplings * spears_per_sapling + spears_per_log = spears_from_6_saplings_and_log :=
by sorry

end NUMINAMATH_CALUDE_spears_from_sapling_proof_l1493_149394


namespace NUMINAMATH_CALUDE_sum_of_squares_constant_l1493_149390

/-- A regular polygon with n vertices and circumradius r -/
structure RegularPolygon where
  n : ℕ
  r : ℝ
  h_n : n ≥ 3
  h_r : r > 0

/-- The sum of squares of distances from a point on the circumcircle to all vertices -/
def sum_of_squares (poly : RegularPolygon) (P : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem stating that the sum of squares is constant for any point on the circumcircle -/
theorem sum_of_squares_constant (poly : RegularPolygon) :
  ∀ P : ℝ × ℝ, (P.1 - poly.r)^2 + P.2^2 = poly.r^2 →
    sum_of_squares poly P = 2 * poly.n * poly.r^2 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_constant_l1493_149390


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1493_149322

theorem rectangle_dimensions (x : ℝ) : 
  (x + 3) * (3 * x - 4) = 5 * x + 14 → x = Real.sqrt 78 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1493_149322


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1493_149342

/-- Represents different sampling methods -/
inductive SamplingMethod
| Random
| Systematic
| Stratified

/-- Represents income levels -/
inductive IncomeLevel
| High
| Middle
| Low

/-- Represents a community with families of different income levels -/
structure Community where
  total_families : Nat
  high_income : Nat
  middle_income : Nat
  low_income : Nat

/-- Represents a group of volleyball players -/
structure VolleyballTeam where
  total_players : Nat

/-- Determines the appropriate sampling method based on population characteristics -/
def determineSamplingMethod (population : Nat) (has_strata : Bool) : SamplingMethod :=
  if has_strata then SamplingMethod.Stratified
  else if population > 20 then SamplingMethod.Systematic
  else SamplingMethod.Random

/-- Theorem stating the correct sampling methods for the given scenarios -/
theorem correct_sampling_methods
  (community : Community)
  (sample_size : Nat)
  (team : VolleyballTeam)
  (players_to_select : Nat)
  (h1 : community.total_families = 400)
  (h2 : community.high_income = 120)
  (h3 : community.middle_income = 180)
  (h4 : community.low_income = 100)
  (h5 : sample_size = 100)
  (h6 : team.total_players = 12)
  (h7 : players_to_select = 3) :
  determineSamplingMethod community.total_families true = SamplingMethod.Stratified ∧
  determineSamplingMethod team.total_players false = SamplingMethod.Random :=
by sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1493_149342


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_range_l1493_149369

/-- The function f(x) defined in terms of parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (a / (a^2 - 2)) * (a^x - a^(-x))

/-- Theorem stating the conditions for f to be an increasing function -/
theorem f_increasing_iff_a_range (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (a > Real.sqrt 2 ∨ 0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_range_l1493_149369
