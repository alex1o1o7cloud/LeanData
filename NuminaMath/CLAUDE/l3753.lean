import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_solution_l3753_375345

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 + k * x + 2

-- Define the known root
def known_root : ℝ := -0.5

-- Theorem statement
theorem quadratic_solution :
  ∃ (k : ℝ),
    (quadratic_equation k known_root = 0) ∧
    (k = 6) ∧
    (∃ (other_root : ℝ), 
      (quadratic_equation k other_root = 0) ∧
      (other_root = -1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3753_375345


namespace NUMINAMATH_CALUDE_g_range_l3753_375357

noncomputable def g (x : ℝ) : ℝ := Real.arctan (x^3) + Real.arctan ((1 - x^3) / (1 + x^3))

theorem g_range :
  Set.range g = Set.Icc (-3 * Real.pi / 4) (Real.pi / 4) :=
sorry

end NUMINAMATH_CALUDE_g_range_l3753_375357


namespace NUMINAMATH_CALUDE_no_equation_fits_l3753_375360

def points : List (ℝ × ℝ) := [(0, 200), (1, 140), (2, 80), (3, 20), (4, 0)]

def equation1 (x : ℝ) : ℝ := 200 - 15 * x
def equation2 (x : ℝ) : ℝ := 200 - 20 * x + 5 * x^2
def equation3 (x : ℝ) : ℝ := 200 - 30 * x + 10 * x^2
def equation4 (x : ℝ) : ℝ := 150 - 50 * x

theorem no_equation_fits : 
  ∀ (x y : ℝ), (x, y) ∈ points → 
    (y ≠ equation1 x) ∨ 
    (y ≠ equation2 x) ∨ 
    (y ≠ equation3 x) ∨ 
    (y ≠ equation4 x) := by
  sorry

end NUMINAMATH_CALUDE_no_equation_fits_l3753_375360


namespace NUMINAMATH_CALUDE_corrected_mean_l3753_375366

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 100 ∧ original_mean = 45 ∧ incorrect_value = 32 ∧ correct_value = 87 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) = n * (45 + 55 / 100) :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l3753_375366


namespace NUMINAMATH_CALUDE_sixth_piggy_bank_coins_l3753_375380

def coin_sequence (n : ℕ) : ℕ := 72 + 9 * (n - 1)

theorem sixth_piggy_bank_coins :
  coin_sequence 6 = 117 := by
  sorry

end NUMINAMATH_CALUDE_sixth_piggy_bank_coins_l3753_375380


namespace NUMINAMATH_CALUDE_line_intersects_circle_twice_l3753_375349

/-- The line y = -x + a intersects the curve y = √(1 - x²) at two points
    if and only if a is in the range [1, √2). -/
theorem line_intersects_circle_twice (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   -x₁ + a = Real.sqrt (1 - x₁^2) ∧
   -x₂ + a = Real.sqrt (1 - x₂^2)) ↔ 
  1 ≤ a ∧ a < Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_twice_l3753_375349


namespace NUMINAMATH_CALUDE_water_left_in_ml_l3753_375305

-- Define the total amount of water in liters
def total_water : ℝ := 135.1

-- Define the size of each bucket in liters
def bucket_size : ℝ := 7

-- Define the conversion factor from liters to milliliters
def liters_to_ml : ℝ := 1000

-- Theorem statement
theorem water_left_in_ml :
  (total_water - bucket_size * ⌊total_water / bucket_size⌋) * liters_to_ml = 2100 := by
  sorry


end NUMINAMATH_CALUDE_water_left_in_ml_l3753_375305


namespace NUMINAMATH_CALUDE_line_equation_intersection_condition_max_value_condition_l3753_375368

-- Define the parabola and line
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 1
def line (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Theorem 1: Line equation
theorem line_equation :
  ∃ k b : ℝ, (line k b (-2) = -5/2) ∧ (line k b 3 = 0) →
  (k = 1/2 ∧ b = -3/2) :=
sorry

-- Theorem 2: Intersection condition
theorem intersection_condition (a : ℝ) :
  a ≠ 0 →
  (∃ x : ℝ, parabola a x = line (1/2) (-3/2) x) ↔
  (a ≤ 9/8) :=
sorry

-- Theorem 3: Maximum value condition
theorem max_value_condition :
  ∃ m : ℝ, (∀ x : ℝ, m ≤ x ∧ x ≤ m + 2 →
    parabola (-1) x ≤ -4) ∧
    (∃ x : ℝ, m ≤ x ∧ x ≤ m + 2 ∧ parabola (-1) x = -4) →
  (m = -3 ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_intersection_condition_max_value_condition_l3753_375368


namespace NUMINAMATH_CALUDE_equation_solution_l3753_375395

theorem equation_solution :
  ∃! x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3753_375395


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3753_375346

theorem triangle_angle_problem (left right top : ℝ) : 
  left + right + top = 250 →
  left = 2 * right →
  right = 60 →
  top = 70 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3753_375346


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l3753_375365

theorem cubic_roots_relation (m n p q : ℝ) : 
  (∃ α β : ℝ, α^2 + m*α + n = 0 ∧ β^2 + m*β + n = 0) →
  (∃ γ δ : ℝ, γ^2 + p*γ + q = 0 ∧ δ^2 + p*δ + q = 0) →
  (∀ α β γ δ : ℝ, 
    (α^2 + m*α + n = 0 ∧ β^2 + m*β + n = 0) →
    (γ^2 + p*γ + q = 0 ∧ δ^2 + p*δ + q = 0) →
    (γ = α^3 ∧ δ = β^3 ∨ γ = β^3 ∧ δ = α^3)) →
  p = m^3 - 3*m*n :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l3753_375365


namespace NUMINAMATH_CALUDE_pipe_filling_time_l3753_375387

theorem pipe_filling_time (t : ℝ) 
  (h1 : t > 0)  -- Pipe A filling time is positive
  (h2 : (1 / t + 6 / t) = 1 / 3)  -- Combined rate equals 1/3 tank per minute
  : t = 21 := by
sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l3753_375387


namespace NUMINAMATH_CALUDE_equation_solution_l3753_375389

theorem equation_solution : 
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ↔ (x = 4 + Real.sqrt 2 ∨ x = 4 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3753_375389


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l3753_375359

/-- The man's rowing speed in still water -/
def rowing_speed : ℝ := 3.9

/-- The speed of the current -/
def current_speed : ℝ := 1.3

/-- The ratio of time taken to row upstream compared to downstream -/
def time_ratio : ℝ := 2

theorem mans_rowing_speed :
  (rowing_speed + current_speed) * time_ratio = (rowing_speed - current_speed) * (time_ratio * 2) ∧
  rowing_speed = 3 * current_speed := by
  sorry

#check mans_rowing_speed

end NUMINAMATH_CALUDE_mans_rowing_speed_l3753_375359


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3753_375330

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Main theorem
theorem arithmetic_sequence_problem
  (a : ℕ → ℝ) (d m : ℝ) (h_d : d ≠ 0)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_seq : arithmetic_sequence a d)
  (h_m : ∃ m : ℕ, a m = 8) :
  ∃ m : ℕ, m = 8 ∧ a m = 8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3753_375330


namespace NUMINAMATH_CALUDE_race_last_part_length_l3753_375322

theorem race_last_part_length
  (total_length : ℝ)
  (first_part : ℝ)
  (second_part : ℝ)
  (third_part : ℝ)
  (h1 : total_length = 74.5)
  (h2 : first_part = 15.5)
  (h3 : second_part = 21.5)
  (h4 : third_part = 21.5) :
  total_length - (first_part + second_part + third_part) = 16 := by
  sorry

end NUMINAMATH_CALUDE_race_last_part_length_l3753_375322


namespace NUMINAMATH_CALUDE_cookies_ratio_l3753_375315

/-- Proves the ratio of cookies eaten by Monica's mother to her father -/
theorem cookies_ratio :
  ∀ (total mother_cookies father_cookies brother_cookies left : ℕ),
  total = 30 →
  father_cookies = 10 →
  brother_cookies = mother_cookies + 2 →
  left = 8 →
  total = mother_cookies + father_cookies + brother_cookies + left →
  (mother_cookies : ℚ) / father_cookies = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cookies_ratio_l3753_375315


namespace NUMINAMATH_CALUDE_function_behavior_l3753_375396

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
    (h_odd : is_odd f)
    (h_periodic : ∀ x, f x = f (x - 2))
    (h_decreasing : is_decreasing_on f 1 2) :
  is_decreasing_on f (-3) (-2) ∧ is_increasing_on f 0 1 := by
  sorry

end NUMINAMATH_CALUDE_function_behavior_l3753_375396


namespace NUMINAMATH_CALUDE_dinner_bill_calculation_l3753_375323

/-- Calculate the total cost of a food item including sales tax and service charge -/
def itemTotalCost (basePrice : ℚ) (salesTaxRate : ℚ) (serviceChargeRate : ℚ) : ℚ :=
  basePrice * (1 + salesTaxRate + serviceChargeRate)

/-- Calculate the total bill for the family's dinner -/
def totalBill : ℚ :=
  itemTotalCost 20 (7/100) (8/100) +
  itemTotalCost 15 (17/200) (1/10) +
  itemTotalCost 10 (3/50) (3/25)

/-- Theorem stating that the total bill is equal to $52.58 -/
theorem dinner_bill_calculation : 
  ∃ (n : ℕ), (n : ℚ) / 100 = totalBill ∧ n = 5258 :=
sorry

end NUMINAMATH_CALUDE_dinner_bill_calculation_l3753_375323


namespace NUMINAMATH_CALUDE_total_sand_weight_is_34_l3753_375355

/-- The number of buckets of sand carried by Eden -/
def eden_buckets : ℕ := 4

/-- The number of additional buckets Mary carried compared to Eden -/
def mary_extra_buckets : ℕ := 3

/-- The number of fewer buckets Iris carried compared to Mary -/
def iris_fewer_buckets : ℕ := 1

/-- The weight of sand in each bucket (in pounds) -/
def sand_per_bucket : ℕ := 2

/-- Calculates the total weight of sand collected by Eden, Mary, and Iris -/
def total_sand_weight : ℕ := 
  (eden_buckets + (eden_buckets + mary_extra_buckets) + 
   (eden_buckets + mary_extra_buckets - iris_fewer_buckets)) * sand_per_bucket

/-- Theorem stating that the total weight of sand collected is 34 pounds -/
theorem total_sand_weight_is_34 : total_sand_weight = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_sand_weight_is_34_l3753_375355


namespace NUMINAMATH_CALUDE_simplify_expression_l3753_375354

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (hxy : x^2 ≠ y^2) :
  (x^2 - y^2)⁻¹ * (x⁻¹ - z⁻¹) = (z - x) * x⁻¹ * z⁻¹ * (x^2 - y^2)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3753_375354


namespace NUMINAMATH_CALUDE_total_chickens_on_farm_l3753_375388

/-- Represents the number of chickens on a farm. -/
structure ChickenFarm where
  roosters : ℕ
  hens : ℕ

/-- The total number of chickens on the farm. -/
def ChickenFarm.total (farm : ChickenFarm) : ℕ :=
  farm.roosters + farm.hens

/-- A farm where roosters outnumber hens 2 to 1. -/
def RoostersOutnumberHens (farm : ChickenFarm) : Prop :=
  2 * farm.hens = farm.roosters

theorem total_chickens_on_farm (farm : ChickenFarm) 
  (h1 : RoostersOutnumberHens farm) 
  (h2 : farm.roosters = 6000) : 
  farm.total = 9000 := by
  sorry

end NUMINAMATH_CALUDE_total_chickens_on_farm_l3753_375388


namespace NUMINAMATH_CALUDE_x_intercept_implies_b_value_l3753_375321

/-- 
Given a line y = 2x - b with an x-intercept of 1, 
prove that b = 2
-/
theorem x_intercept_implies_b_value :
  ∀ b : ℝ, (∃ x : ℝ, x = 1 ∧ 2 * x - b = 0) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_implies_b_value_l3753_375321


namespace NUMINAMATH_CALUDE_all_same_number_probability_l3753_375328

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice thrown -/
def num_dice : ℕ := 5

/-- The total number of possible outcomes when throwing the dice -/
def total_outcomes : ℕ := num_faces ^ num_dice

/-- The number of favorable outcomes (all dice showing the same number) -/
def favorable_outcomes : ℕ := num_faces

/-- The probability of all dice showing the same number -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem all_same_number_probability :
  probability = 1 / 1296 := by sorry

end NUMINAMATH_CALUDE_all_same_number_probability_l3753_375328


namespace NUMINAMATH_CALUDE_polynomial_integer_root_theorem_l3753_375377

theorem polynomial_integer_root_theorem (n : ℕ+) :
  (∃ (k : Fin n → ℤ) (P : Polynomial ℤ),
    (∀ (i j : Fin n), i ≠ j → k i ≠ k j) ∧
    (Polynomial.degree P ≤ n) ∧
    (∀ (i : Fin n), P.eval (k i) = n) ∧
    (∃ (z : ℤ), P.eval z = 0)) ↔
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

#check polynomial_integer_root_theorem

end NUMINAMATH_CALUDE_polynomial_integer_root_theorem_l3753_375377


namespace NUMINAMATH_CALUDE_rectangle_area_l3753_375335

theorem rectangle_area (width : ℝ) (h1 : width > 0) : 
  let length := 2 * width
  let diagonal := 10
  width ^ 2 + length ^ 2 = diagonal ^ 2 →
  width * length = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3753_375335


namespace NUMINAMATH_CALUDE_anna_cupcakes_l3753_375393

def total_cupcakes : ℕ := 150

def classmates_fraction : ℚ := 2/5
def neighbors_fraction : ℚ := 1/3
def work_friends_fraction : ℚ := 1/10
def eating_fraction : ℚ := 7/15

def remaining_cupcakes : ℕ := 14

theorem anna_cupcakes :
  let given_away := (classmates_fraction + neighbors_fraction + work_friends_fraction) * total_cupcakes
  let after_giving := total_cupcakes - ⌊given_away⌋
  let eaten := ⌊eating_fraction * after_giving⌋
  total_cupcakes - ⌊given_away⌋ - eaten = remaining_cupcakes := by
  sorry

#check anna_cupcakes

end NUMINAMATH_CALUDE_anna_cupcakes_l3753_375393


namespace NUMINAMATH_CALUDE_probability_three_heads_in_seven_tosses_l3753_375338

def coin_tosses : ℕ := 7
def heads_count : ℕ := 3

theorem probability_three_heads_in_seven_tosses :
  (Nat.choose coin_tosses heads_count) / (2 ^ coin_tosses) = 35 / 128 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_seven_tosses_l3753_375338


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3753_375316

theorem isosceles_triangle_base_length 
  (eq_perimeter : ℝ) 
  (is_perimeter : ℝ) 
  (eq_side : ℝ) 
  (is_equal_side : ℝ) 
  (is_base : ℝ) 
  (vertex_angle : ℝ) 
  (h1 : eq_perimeter = 45) 
  (h2 : is_perimeter = 40) 
  (h3 : 3 * eq_side = eq_perimeter) 
  (h4 : 2 * is_equal_side + is_base = is_perimeter) 
  (h5 : is_equal_side = eq_side) 
  (h6 : 100 < vertex_angle ∧ vertex_angle < 120) : 
  is_base = 10 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3753_375316


namespace NUMINAMATH_CALUDE_smallest_period_scaled_l3753_375383

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x - p) = f x

theorem smallest_period_scaled (f : ℝ → ℝ) (h : is_periodic f 15) :
  ∃ a : ℝ, a > 0 ∧ (∀ x, f ((x - a) / 3) = f (x / 3)) ∧
    ∀ b, b > 0 → (∀ x, f ((x - b) / 3) = f (x / 3)) → a ≤ b :=
  sorry

end NUMINAMATH_CALUDE_smallest_period_scaled_l3753_375383


namespace NUMINAMATH_CALUDE_billy_weight_l3753_375334

theorem billy_weight (carl_weight brad_weight dave_weight billy_weight edgar_weight : ℝ) :
  carl_weight = 145 ∧
  brad_weight = carl_weight + 5 ∧
  dave_weight = carl_weight + 8 ∧
  dave_weight = 2 * brad_weight ∧
  edgar_weight = 3 * dave_weight - 20 ∧
  billy_weight = brad_weight + 9 →
  billy_weight = 85.5 := by
sorry

end NUMINAMATH_CALUDE_billy_weight_l3753_375334


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3753_375343

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence satisfying certain conditions, prove that a₇ + a₁₀ = 27/2 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h1 : a 3 + a 6 = 6) 
  (h2 : a 5 + a 8 = 9) : 
  a 7 + a 10 = 27/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3753_375343


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l3753_375394

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = 2 / (2 * x - 1)) ↔ x ≠ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l3753_375394


namespace NUMINAMATH_CALUDE_books_read_l3753_375364

theorem books_read (total_books : ℕ) (unread_books : ℕ) (h1 : total_books = 20) (h2 : unread_books = 5) :
  total_books - unread_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_read_l3753_375364


namespace NUMINAMATH_CALUDE_statue_weight_calculation_l3753_375325

/-- The weight of a marble statue after three successive cuts -/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  initial_weight * (1 - 0.3) * (1 - 0.3) * (1 - 0.15)

/-- Theorem stating the final weight of the statue -/
theorem statue_weight_calculation :
  final_statue_weight 300 = 124.95 := by
  sorry

#eval final_statue_weight 300

end NUMINAMATH_CALUDE_statue_weight_calculation_l3753_375325


namespace NUMINAMATH_CALUDE_expression_value_at_four_l3753_375374

theorem expression_value_at_four :
  let f (x : ℝ) := (x^8 - 32*x^4 + 256) / (x^4 - 16)
  f 4 = 240 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_four_l3753_375374


namespace NUMINAMATH_CALUDE_fly_speed_fly_speed_problem_l3753_375300

/-- The speed of a fly moving between two cyclists --/
theorem fly_speed (cyclist_speed : ℝ) (initial_distance : ℝ) (fly_distance : ℝ) : ℝ :=
  let relative_speed := 2 * cyclist_speed
  let meeting_time := initial_distance / relative_speed
  fly_distance / meeting_time

/-- Given the conditions of the problem, prove that the fly's speed is 15 miles/hour --/
theorem fly_speed_problem : fly_speed 10 50 37.5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fly_speed_fly_speed_problem_l3753_375300


namespace NUMINAMATH_CALUDE_three_intersections_iff_a_in_open_interval_l3753_375399

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the condition for three distinct intersection points
def has_three_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
  f x₁ = a ∧ f x₂ = a ∧ f x₃ = a

-- Theorem statement
theorem three_intersections_iff_a_in_open_interval :
  ∀ a : ℝ, has_three_intersections a ↔ -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_intersections_iff_a_in_open_interval_l3753_375399


namespace NUMINAMATH_CALUDE_prob_first_ace_is_one_eighth_l3753_375370

/-- Represents a deck of cards -/
structure Deck :=
  (size : ℕ)
  (num_aces : ℕ)

/-- Represents the card game setup -/
structure CardGame :=
  (deck : Deck)
  (num_players : ℕ)

/-- The probability of a player getting the first Ace -/
def prob_first_ace (game : CardGame) (player : ℕ) : ℚ :=
  1 / game.num_players

/-- Theorem stating that the probability of each player getting the first Ace is 1/8 -/
theorem prob_first_ace_is_one_eighth (game : CardGame) :
  game.deck.size = 32 ∧ game.deck.num_aces = 4 ∧ game.num_players = 4 →
  ∀ player, player > 0 ∧ player ≤ game.num_players →
    prob_first_ace game player = 1 / 8 :=
sorry

end NUMINAMATH_CALUDE_prob_first_ace_is_one_eighth_l3753_375370


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l3753_375342

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 3) :
  Complex.abs ((z - 2)^2 * (z + 2)) ≤ Real.sqrt 637 ∧
  ∃ w : ℂ, Complex.abs w = Real.sqrt 3 ∧ Complex.abs ((w - 2)^2 * (w + 2)) = Real.sqrt 637 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l3753_375342


namespace NUMINAMATH_CALUDE_seating_theorem_l3753_375386

/-- The number of seats in the row -/
def total_seats : ℕ := 8

/-- The number of people to be seated -/
def people_to_seat : ℕ := 3

/-- A function that calculates the number of seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) : ℕ :=
  -- The actual implementation is not provided in the problem
  sorry

/-- Theorem stating that the number of seating arrangements is 24 -/
theorem seating_theorem : seating_arrangements total_seats people_to_seat = 24 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l3753_375386


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3753_375306

/-- Represents a repeating decimal where the whole number part is 7 and the repeating part is 182. -/
def repeating_decimal : ℚ := 7 + 182 / 999

/-- The fraction representation of the repeating decimal. -/
def fraction : ℚ := 7175 / 999

/-- Theorem stating that the repeating decimal 7.182182... is equal to the fraction 7175/999. -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3753_375306


namespace NUMINAMATH_CALUDE_lcm_6_15_l3753_375312

theorem lcm_6_15 : Nat.lcm 6 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_6_15_l3753_375312


namespace NUMINAMATH_CALUDE_expression_simplification_l3753_375350

theorem expression_simplification (x y : ℚ) (hx : x = 1/9) (hy : y = 5) :
  -1/5 * x * y^2 - 3 * x^2 * y + x * y^2 + 2 * x^2 * y + 3 * x * y^2 + x^2 * y - 2 * x * y^2 = 20/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3753_375350


namespace NUMINAMATH_CALUDE_wire_length_ratio_l3753_375326

/-- The length of each wire piece used by Bonnie to construct her cube frame -/
def bonnie_wire_length : ℚ := 8

/-- The number of wire pieces used by Bonnie to construct her cube frame -/
def bonnie_wire_count : ℕ := 12

/-- The length of each wire piece used by Roark to construct unit cube frames -/
def roark_wire_length : ℚ := 2

/-- The volume of a unit cube constructed by Roark -/
def unit_cube_volume : ℚ := 1

/-- The number of edges in a cube -/
def cube_edge_count : ℕ := 12

theorem wire_length_ratio :
  let bonnie_total_length := bonnie_wire_length * bonnie_wire_count
  let bonnie_cube_volume := (bonnie_wire_length / 4) ^ 3
  let roark_unit_cube_wire_length := roark_wire_length * cube_edge_count
  let roark_cube_count := bonnie_cube_volume / unit_cube_volume
  let roark_total_length := roark_unit_cube_wire_length * roark_cube_count
  bonnie_total_length / roark_total_length = 1 / 128 := by
sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l3753_375326


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l3753_375369

-- Define the two lines
def line1 (x y : ℚ) : Prop := 2 * x - 3 * y = 3
def line2 (x y : ℚ) : Prop := 4 * x + 2 * y = 2

-- Define the intersection point
def intersection_point : ℚ × ℚ := (3/4, -1/2)

-- Theorem statement
theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → (x', y') = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l3753_375369


namespace NUMINAMATH_CALUDE_survey_results_l3753_375348

theorem survey_results (total : ℕ) (support_a : ℕ) (support_b : ℕ) (support_both : ℕ) (support_neither : ℕ) : 
  total = 50 ∧
  support_a = (3 * total) / 5 ∧
  support_b = support_a + 3 ∧
  support_neither = support_both / 3 + 1 ∧
  total = support_a + support_b - support_both + support_neither →
  support_both = 21 ∧ support_neither = 8 := by
  sorry

end NUMINAMATH_CALUDE_survey_results_l3753_375348


namespace NUMINAMATH_CALUDE_inequality_proof_l3753_375397

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2*b^2 + b^2*c^2 + c^2*a^2 ≥ 15/16 ∧
  (a^2 + b^2 + c^2 + a^2*b^2 + b^2*c^2 + c^2*a^2 = 15/16 ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3753_375397


namespace NUMINAMATH_CALUDE_jerry_added_ten_books_l3753_375361

/-- The number of books Jerry added to his shelf -/
def books_added (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem stating that Jerry added 10 books to his shelf -/
theorem jerry_added_ten_books (initial_books final_books : ℕ) 
  (h1 : initial_books = 9)
  (h2 : final_books = 19) : 
  books_added initial_books final_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_jerry_added_ten_books_l3753_375361


namespace NUMINAMATH_CALUDE_class_ratio_and_total_l3753_375363

theorem class_ratio_and_total (num_girls : ℕ) (num_boys : ℕ) : 
  (3 : ℚ) / 7 * num_boys = (6 : ℚ) / 11 * num_girls → 
  num_girls = 22 →
  (num_boys : ℚ) / num_girls = 14 / 11 ∧ num_boys + num_girls = 50 := by
  sorry

end NUMINAMATH_CALUDE_class_ratio_and_total_l3753_375363


namespace NUMINAMATH_CALUDE_store_profits_l3753_375332

theorem store_profits (profit_a profit_b : ℝ) 
  (h : profit_a * 1.2 = profit_b * 0.9) : 
  profit_a = 0.75 * profit_b := by
sorry

end NUMINAMATH_CALUDE_store_profits_l3753_375332


namespace NUMINAMATH_CALUDE_sum_in_base_8_l3753_375372

/-- Converts a decimal number to its octal (base 8) representation -/
def toOctal (n : ℕ) : List ℕ :=
  sorry

/-- Converts an octal (base 8) representation to its decimal value -/
def fromOctal (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in their octal representations -/
def octalAdd (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_in_base_8 :
  let a := 624
  let b := 112
  let expected_sum := [1, 3, 4, 0]
  octalAdd (toOctal a) (toOctal b) = expected_sum ∧
  fromOctal expected_sum = a + b :=
by sorry

end NUMINAMATH_CALUDE_sum_in_base_8_l3753_375372


namespace NUMINAMATH_CALUDE_problem_solution_l3753_375384

theorem problem_solution :
  (∃ (x y : ℝ),
    (x = (3 * Real.sqrt 18 + (1/5) * Real.sqrt 50 - 4 * Real.sqrt (1/2)) / Real.sqrt 32 ∧ x = 2) ∧
    (y = (1 + Real.sqrt 2 + Real.sqrt 3) * (1 + Real.sqrt 2 - Real.sqrt 3) ∧ y = 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3753_375384


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3753_375390

theorem repeating_decimal_sum : 
  ∃ (a b c : ℚ), 
    (∀ n : ℕ, a = (234 : ℚ) / 10^3 + (234 : ℚ) / (999 * 10^n)) ∧
    (∀ n : ℕ, b = (567 : ℚ) / 10^3 + (567 : ℚ) / (999 * 10^n)) ∧
    (∀ n : ℕ, c = (891 : ℚ) / 10^3 + (891 : ℚ) / (999 * 10^n)) ∧
    a - b + c = 186 / 333 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3753_375390


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l3753_375310

/-- The line equation is 5y + 3x = 15 -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line 5y + 3x = 15 with the y-axis is (0, 3) -/
theorem line_y_axis_intersection :
  ∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l3753_375310


namespace NUMINAMATH_CALUDE_bob_sandwich_options_l3753_375309

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of sandwiches with turkey and mozzarella cheese. -/
def turkey_mozzarella_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and beef. -/
def rye_beef_combos : ℕ := num_cheeses

/-- Calculates the total number of possible sandwich combinations. -/
def total_combos : ℕ := num_breads * num_meats * num_cheeses

/-- Theorem stating the number of different sandwiches Bob could order. -/
theorem bob_sandwich_options : 
  total_combos - turkey_mozzarella_combos - rye_beef_combos = 199 := by
  sorry

end NUMINAMATH_CALUDE_bob_sandwich_options_l3753_375309


namespace NUMINAMATH_CALUDE_paperclip_growth_l3753_375341

theorem paperclip_growth (n : ℕ) : (8 * 3^n > 1000) ↔ n ≥ 5 := by sorry

end NUMINAMATH_CALUDE_paperclip_growth_l3753_375341


namespace NUMINAMATH_CALUDE_regular_hexagon_radius_l3753_375339

/-- The radius of a regular hexagon with perimeter 12a is 2a -/
theorem regular_hexagon_radius (a : ℝ) (h : a > 0) :
  let perimeter := 12 * a
  ∃ (radius : ℝ), radius = 2 * a ∧ 
    (∃ (side : ℝ), side * 6 = perimeter ∧ radius = side) := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_radius_l3753_375339


namespace NUMINAMATH_CALUDE_bens_daily_start_amount_l3753_375303

/-- Proves that given the conditions of Ben's savings scenario, he must start with $50 each day -/
theorem bens_daily_start_amount :
  ∀ (X : ℚ),
  (∃ (D : ℕ),
    (2 * (D * (X - 15)) + 10 = 500) ∧
    (D = 7)) →
  X = 50 := by
  sorry

end NUMINAMATH_CALUDE_bens_daily_start_amount_l3753_375303


namespace NUMINAMATH_CALUDE_walter_zoo_time_l3753_375333

theorem walter_zoo_time (total_time seals penguins elephants : ℕ) : 
  total_time = 130 ∧ 
  penguins = 8 * seals ∧ 
  elephants = 13 ∧ 
  seals + penguins + elephants = total_time → 
  seals = 13 := by
sorry

end NUMINAMATH_CALUDE_walter_zoo_time_l3753_375333


namespace NUMINAMATH_CALUDE_sam_paul_study_difference_l3753_375302

def average_difference (differences : List Int) : Int :=
  (differences.sum / differences.length)

theorem sam_paul_study_difference : 
  let differences : List Int := [20, 5, -5, 0, 15, -10, 10]
  average_difference differences = 5 := by
  sorry

end NUMINAMATH_CALUDE_sam_paul_study_difference_l3753_375302


namespace NUMINAMATH_CALUDE_optimal_small_box_size_is_correct_l3753_375344

def total_balls : ℕ := 104
def big_box_capacity : ℕ := 25
def min_unboxed : ℕ := 5

def is_valid_small_box_size (size : ℕ) : Prop :=
  size > 0 ∧
  size < big_box_capacity ∧
  ∃ (big_boxes small_boxes : ℕ),
    big_boxes * big_box_capacity + small_boxes * size + min_unboxed = total_balls ∧
    small_boxes > 0

def optimal_small_box_size : ℕ := 12

theorem optimal_small_box_size_is_correct :
  is_valid_small_box_size optimal_small_box_size ∧
  ∀ (size : ℕ), is_valid_small_box_size size → size ≤ optimal_small_box_size :=
by sorry

end NUMINAMATH_CALUDE_optimal_small_box_size_is_correct_l3753_375344


namespace NUMINAMATH_CALUDE_remainder_845307_div_6_l3753_375362

theorem remainder_845307_div_6 : Nat.mod 845307 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_845307_div_6_l3753_375362


namespace NUMINAMATH_CALUDE_number_problem_l3753_375376

theorem number_problem (x : ℝ) (h : 6 * x = 72) : x + 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3753_375376


namespace NUMINAMATH_CALUDE_healthcare_worker_identity_l3753_375311

/-- Represents the number of healthcare workers of each type -/
structure HealthcareWorkers where
  male_doctors : ℕ
  female_doctors : ℕ
  female_nurses : ℕ
  male_nurses : ℕ

/-- Checks if the given numbers satisfy all conditions -/
def satisfies_conditions (hw : HealthcareWorkers) : Prop :=
  hw.male_doctors + hw.female_doctors + hw.female_nurses + hw.male_nurses = 17 ∧
  hw.male_doctors + hw.female_doctors ≥ hw.female_nurses + hw.male_nurses ∧
  hw.female_nurses > hw.male_doctors ∧
  hw.male_doctors > hw.female_doctors ∧
  hw.male_nurses ≥ 2

/-- The unique solution that satisfies all conditions -/
def solution : HealthcareWorkers :=
  { male_doctors := 5
    female_doctors := 4
    female_nurses := 6
    male_nurses := 2 }

/-- The statement to be proved -/
theorem healthcare_worker_identity :
  satisfies_conditions solution ∧
  satisfies_conditions { male_doctors := solution.male_doctors,
                         female_doctors := solution.female_doctors - 1,
                         female_nurses := solution.female_nurses,
                         male_nurses := solution.male_nurses } ∧
  ∀ (hw : HealthcareWorkers), satisfies_conditions hw → hw = solution :=
sorry

end NUMINAMATH_CALUDE_healthcare_worker_identity_l3753_375311


namespace NUMINAMATH_CALUDE_bobby_candy_l3753_375353

def candy_problem (initial : ℕ) (final : ℕ) (second_round : ℕ) : Prop :=
  ∃ (first_round : ℕ), 
    initial - (first_round + second_round) = final ∧
    first_round + second_round < initial

theorem bobby_candy : candy_problem 21 7 9 → ∃ (x : ℕ), x = 5 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_l3753_375353


namespace NUMINAMATH_CALUDE_triangle_area_l3753_375381

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 5, b = 4, and cos(A - B) = 31/32, then the area of the triangle is (15 * √7) / 4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  a = 5 →
  b = 4 →
  Real.cos (A - B) = 31/32 →
  (1/2) * a * b * Real.sin C = (15 * Real.sqrt 7) / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3753_375381


namespace NUMINAMATH_CALUDE_intersection_equals_T_l3753_375324

noncomputable def S : Set ℝ := {y | ∃ x : ℝ, y = x^3}
noncomputable def T : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

theorem intersection_equals_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_intersection_equals_T_l3753_375324


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l3753_375317

theorem quadratic_equation_value (a : ℝ) (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l3753_375317


namespace NUMINAMATH_CALUDE_range_equals_fixed_points_l3753_375314

theorem range_equals_fixed_points (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (m + f n) = f (f m) + f n) : 
  {n : ℕ | ∃ k : ℕ, f k = n} = {n : ℕ | f n = n} := by
sorry

end NUMINAMATH_CALUDE_range_equals_fixed_points_l3753_375314


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l3753_375347

theorem complex_absolute_value_product : Complex.abs (3 - 2*Complex.I) * Complex.abs (3 + 2*Complex.I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l3753_375347


namespace NUMINAMATH_CALUDE_revenue_calculation_impossible_l3753_375375

structure ShoeInventory where
  large_boots : ℕ
  medium_sandals : ℕ
  small_sneakers : ℕ
  large_sandals : ℕ
  medium_boots : ℕ
  small_boots : ℕ

def initial_stock : ShoeInventory :=
  { large_boots := 22
  , medium_sandals := 32
  , small_sneakers := 24
  , large_sandals := 45
  , medium_boots := 35
  , small_boots := 26 }

def prices : ShoeInventory :=
  { large_boots := 80
  , medium_sandals := 60
  , small_sneakers := 50
  , large_sandals := 65
  , medium_boots := 75
  , small_boots := 55 }

def total_pairs (stock : ShoeInventory) : ℕ :=
  stock.large_boots + stock.medium_sandals + stock.small_sneakers +
  stock.large_sandals + stock.medium_boots + stock.small_boots

def pairs_left : ℕ := 78

theorem revenue_calculation_impossible :
  ∀ (final_stock : ShoeInventory),
    total_pairs final_stock = pairs_left →
    ∃ (revenue₁ revenue₂ : ℕ),
      revenue₁ ≠ revenue₂ ∧
      (∃ (sold : ShoeInventory),
        total_pairs sold + total_pairs final_stock = total_pairs initial_stock ∧
        revenue₁ = sold.large_boots * prices.large_boots +
                   sold.medium_sandals * prices.medium_sandals +
                   sold.small_sneakers * prices.small_sneakers +
                   sold.large_sandals * prices.large_sandals +
                   sold.medium_boots * prices.medium_boots +
                   sold.small_boots * prices.small_boots) ∧
      (∃ (sold : ShoeInventory),
        total_pairs sold + total_pairs final_stock = total_pairs initial_stock ∧
        revenue₂ = sold.large_boots * prices.large_boots +
                   sold.medium_sandals * prices.medium_sandals +
                   sold.small_sneakers * prices.small_sneakers +
                   sold.large_sandals * prices.large_sandals +
                   sold.medium_boots * prices.medium_boots +
                   sold.small_boots * prices.small_boots) :=
by sorry

end NUMINAMATH_CALUDE_revenue_calculation_impossible_l3753_375375


namespace NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l3753_375385

theorem rectangle_triangle_equal_area (perimeter : ℝ) (height : ℝ) (x : ℝ) : 
  perimeter = 60 →
  height = 30 →
  ∃ a b : ℝ, 
    a + b = 30 ∧
    a * b = (1/2) * height * x →
  x = 15 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l3753_375385


namespace NUMINAMATH_CALUDE_equation_equivalence_l3753_375351

theorem equation_equivalence (y : ℝ) (Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) :
  10 * (6 * y + 14 * Real.pi) = 4 * Q :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3753_375351


namespace NUMINAMATH_CALUDE_intersection_sum_mod20_l3753_375352

/-- The sum of x-coordinates of intersection points of two modular functions -/
theorem intersection_sum_mod20 : ∃ (x₁ x₂ : ℕ),
  (x₁ < 20 ∧ x₂ < 20) ∧
  (∀ (y : ℕ), (7 * x₁ + 3) % 20 = y % 20 ↔ (13 * x₁ + 17) % 20 = y % 20) ∧
  (∀ (y : ℕ), (7 * x₂ + 3) % 20 = y % 20 ↔ (13 * x₂ + 17) % 20 = y % 20) ∧
  x₁ + x₂ = 12 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_mod20_l3753_375352


namespace NUMINAMATH_CALUDE_only_paintable_number_l3753_375382

/-- Represents a painting configuration for the railings. -/
structure PaintConfig where
  h : ℕ+  -- Harold's interval
  t : ℕ+  -- Tanya's interval
  u : ℕ+  -- Ulysses' interval

/-- Checks if a given railing number is painted by Harold. -/
def paintedByHarold (config : PaintConfig) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 1 + k * config.h

/-- Checks if a given railing number is painted by Tanya. -/
def paintedByTanya (config : PaintConfig) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 + k * config.t

/-- Checks if a given railing number is painted by Ulysses. -/
def paintedByUlysses (config : PaintConfig) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 + k * config.u

/-- Checks if every railing is painted exactly once. -/
def validPainting (config : PaintConfig) : Prop :=
  ∀ n : ℕ+, (paintedByHarold config n ∨ paintedByTanya config n ∨ paintedByUlysses config n) ∧
            ¬(paintedByHarold config n ∧ paintedByTanya config n) ∧
            ¬(paintedByHarold config n ∧ paintedByUlysses config n) ∧
            ¬(paintedByTanya config n ∧ paintedByUlysses config n)

/-- Computes the paintable number for a given configuration. -/
def paintableNumber (config : PaintConfig) : ℕ :=
  100 * config.h + 10 * config.t + config.u

/-- Theorem stating that 453 is the only paintable number. -/
theorem only_paintable_number :
  ∃! n : ℕ, n = 453 ∧ ∃ config : PaintConfig, validPainting config ∧ paintableNumber config = n :=
sorry

end NUMINAMATH_CALUDE_only_paintable_number_l3753_375382


namespace NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l3753_375378

/-- A positive geometric progression -/
def is_positive_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- An arithmetic progression -/
def is_arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n, b (n + 1) = b n + d

/-- The main theorem -/
theorem geometric_arithmetic_inequality
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_geo : is_positive_geometric_progression a)
  (h_arith : is_arithmetic_progression b)
  (h_eq : a 6 = b 7) :
  a 3 + a 9 ≥ b 4 + b 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l3753_375378


namespace NUMINAMATH_CALUDE_inverse_proportion_l3753_375304

theorem inverse_proportion (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : 1500 * 0.25 = k) :
  3000 * b = k → b = 0.125 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_l3753_375304


namespace NUMINAMATH_CALUDE_original_statement_converse_not_always_true_inverse_not_always_true_neither_converse_nor_inverse_always_true_l3753_375329

-- Define the properties
def is_rectangle (q : Quadrilateral) : Prop := sorry
def has_opposite_sides_equal (q : Quadrilateral) : Prop := sorry

-- Define the original statement
theorem original_statement : 
  ∀ q : Quadrilateral, is_rectangle q → has_opposite_sides_equal q := sorry

-- Prove that the converse is not always true
theorem converse_not_always_true : 
  ¬(∀ q : Quadrilateral, has_opposite_sides_equal q → is_rectangle q) := sorry

-- Prove that the inverse is not always true
theorem inverse_not_always_true : 
  ¬(∀ q : Quadrilateral, ¬is_rectangle q → ¬has_opposite_sides_equal q) := sorry

-- Combine the results
theorem neither_converse_nor_inverse_always_true : 
  (¬(∀ q : Quadrilateral, has_opposite_sides_equal q → is_rectangle q)) ∧
  (¬(∀ q : Quadrilateral, ¬is_rectangle q → ¬has_opposite_sides_equal q)) := sorry

end NUMINAMATH_CALUDE_original_statement_converse_not_always_true_inverse_not_always_true_neither_converse_nor_inverse_always_true_l3753_375329


namespace NUMINAMATH_CALUDE_infinite_points_in_region_l3753_375320

theorem infinite_points_in_region :
  {p : ℚ × ℚ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 7 ∧ p.1 ≥ 1}.Infinite :=
by sorry

end NUMINAMATH_CALUDE_infinite_points_in_region_l3753_375320


namespace NUMINAMATH_CALUDE_g_at_3_l3753_375391

def g (x : ℝ) : ℝ := 5 * x^3 - 3 * x^2 + 7 * x - 2

theorem g_at_3 : g 3 = 127 := by
  sorry

end NUMINAMATH_CALUDE_g_at_3_l3753_375391


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l3753_375371

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem: Given the conditions, prove that the man's speed against the current is 10 km/hr -/
theorem mans_speed_against_current :
  let speed_with_current : ℝ := 15
  let current_speed : ℝ := 2.5
  speed_against_current speed_with_current current_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l3753_375371


namespace NUMINAMATH_CALUDE_min_distinct_integers_for_progressions_l3753_375313

/-- A sequence of integers forms a geometric progression of length 5 -/
def is_geometric_progression (seq : Fin 5 → ℤ) : Prop :=
  ∃ (b q : ℤ), ∀ i : Fin 5, seq i = b * q ^ (i : ℕ)

/-- A sequence of integers forms an arithmetic progression of length 5 -/
def is_arithmetic_progression (seq : Fin 5 → ℤ) : Prop :=
  ∃ (a d : ℤ), ∀ i : Fin 5, seq i = a + (i : ℕ) * d

/-- The minimum number of distinct integers needed for both progressions -/
def min_distinct_integers : ℕ := 6

/-- Theorem stating the minimum number of distinct integers needed -/
theorem min_distinct_integers_for_progressions :
  ∀ (S : Finset ℤ),
  (∃ (seq_gp : Fin 5 → ℤ), (∀ i, seq_gp i ∈ S) ∧ is_geometric_progression seq_gp) ∧
  (∃ (seq_ap : Fin 5 → ℤ), (∀ i, seq_ap i ∈ S) ∧ is_arithmetic_progression seq_ap) →
  S.card ≥ min_distinct_integers :=
sorry

end NUMINAMATH_CALUDE_min_distinct_integers_for_progressions_l3753_375313


namespace NUMINAMATH_CALUDE_plotted_points_form_circle_l3753_375379

theorem plotted_points_form_circle :
  ∀ (x y : ℝ), (∃ t : ℝ, x = Real.cos t ∧ y = Real.sin t) →
  x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_plotted_points_form_circle_l3753_375379


namespace NUMINAMATH_CALUDE_bracket_equation_solution_l3753_375398

theorem bracket_equation_solution (x : ℝ) : 
  45 - (28 - (37 - (15 - x))) = 59 → x = 20 := by
sorry

end NUMINAMATH_CALUDE_bracket_equation_solution_l3753_375398


namespace NUMINAMATH_CALUDE_triangle_problem_l3753_375392

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Given conditions
  3 * Real.cos (B + C) = -1 ∧
  a = 3 ∧
  (1 / 2) * b * c * Real.sin A = 2 * Real.sqrt 2 →
  -- Conclusion
  Real.cos A = 1 / 3 ∧
  ((b = 2 ∧ c = 3) ∨ (b = 3 ∧ c = 2)) := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3753_375392


namespace NUMINAMATH_CALUDE_S_min_at_5_l3753_375337

/-- An arithmetic sequence with first term -9 and S_3 = S_7 -/
def ArithSeq : ℕ → ℤ := fun n => 2*n - 11

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n * (ArithSeq 1 + ArithSeq n) / 2

/-- The condition that S_3 = S_7 -/
axiom S3_eq_S7 : S 3 = S 7

/-- The theorem stating that S_n is minimized when n = 5 -/
theorem S_min_at_5 : ∀ n : ℕ, n ≠ 0 → S 5 ≤ S n :=
sorry

end NUMINAMATH_CALUDE_S_min_at_5_l3753_375337


namespace NUMINAMATH_CALUDE_jessica_letter_paper_weight_l3753_375319

/-- The weight of each piece of paper in Jessica's letter -/
def paper_weight (num_papers : ℕ) (envelope_weight total_weight : ℚ) : ℚ :=
  (total_weight - envelope_weight) / num_papers

/-- Theorem stating that each piece of paper in Jessica's letter weighs 1/5 of an ounce -/
theorem jessica_letter_paper_weight :
  let num_papers : ℕ := 8
  let envelope_weight : ℚ := 2/5
  let total_weight : ℚ := 2
  paper_weight num_papers envelope_weight total_weight = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_letter_paper_weight_l3753_375319


namespace NUMINAMATH_CALUDE_line_mb_equals_nine_l3753_375358

/-- Given a line with equation y = mx + b that intersects the y-axis at y = -3
    and rises 3 units for every 1 unit to the right, prove that mb = 9. -/
theorem line_mb_equals_nine (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  b = -3 →                      -- y-intercept
  (∀ x : ℝ, m * (x + 1) + b = m * x + b + 3) →  -- Slope condition
  m * b = 9 := by
sorry

end NUMINAMATH_CALUDE_line_mb_equals_nine_l3753_375358


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3753_375356

/-- The polynomial f(x) = x^4 - 4x^2 + 7 -/
def f (x : ℝ) : ℝ := x^4 - 4*x^2 + 7

/-- The remainder when f(x) is divided by (x - 1) -/
def remainder : ℝ := f 1

theorem polynomial_remainder : remainder = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3753_375356


namespace NUMINAMATH_CALUDE_inequality_solution_set_not_sufficient_l3753_375327

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → 0 ≤ a ∧ a < 1 :=
by sorry

theorem not_sufficient (a : ℝ) :
  ∃ a : ℝ, 0 ≤ a ∧ a < 1 ∧ ∃ x : ℝ, x^2 - 2*a*x + a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_not_sufficient_l3753_375327


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3753_375301

def U : Set ℤ := Set.univ

def A : Set ℤ := {-1, 1, 2}

def B : Set ℤ := {-1, 1}

theorem intersection_with_complement :
  A ∩ (Set.compl B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3753_375301


namespace NUMINAMATH_CALUDE_intersection_condition_l3753_375331

/-- The line y = k(x+1) intersects the circle (x-1)² + y² = 1 -/
def intersects (k : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * (x + 1) ∧ (x - 1)^2 + y^2 = 1

/-- The condition k > -√3/3 is neither sufficient nor necessary for intersection -/
theorem intersection_condition (k : ℝ) :
  ¬(k > -Real.sqrt 3 / 3 → intersects k) ∧
  ¬(intersects k → k > -Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l3753_375331


namespace NUMINAMATH_CALUDE_book_price_increase_l3753_375308

theorem book_price_increase (original_price new_price : ℝ) 
  (h1 : original_price = 300)
  (h2 : new_price = 390) :
  (new_price - original_price) / original_price * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l3753_375308


namespace NUMINAMATH_CALUDE_division_sum_theorem_l3753_375318

theorem division_sum_theorem (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 40)
  (h_divisor : divisor = 72)
  (h_remainder : remainder = 64) :
  divisor * quotient + remainder = 2944 :=
by sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l3753_375318


namespace NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l3753_375340

def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

theorem quadratic_function_passes_through_points :
  f (-1) = 0 ∧ f 3 = 0 ∧ f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l3753_375340


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3753_375336

theorem gcd_of_three_numbers :
  Nat.gcd 188094 (Nat.gcd 244122 395646) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3753_375336


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_five_l3753_375373

theorem complex_expression_equals_negative_five :
  Real.sqrt 27 + (-1/3)⁻¹ - |2 - Real.sqrt 3| - 8 * Real.cos (30 * π / 180) = -5 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_five_l3753_375373


namespace NUMINAMATH_CALUDE_average_equation_solution_l3753_375367

theorem average_equation_solution (x : ℝ) : 
  (1/4 : ℝ) * ((x + 8) + (7*x - 3) + (3*x + 10) + (-x + 6)) = 5*x - 4 → x = 3.7 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l3753_375367


namespace NUMINAMATH_CALUDE_kelly_baking_powder_l3753_375307

def yesterday_amount : ℝ := 0.4
def difference : ℝ := 0.1

theorem kelly_baking_powder :
  let current_amount := yesterday_amount - difference
  current_amount = 0.3 := by
sorry

end NUMINAMATH_CALUDE_kelly_baking_powder_l3753_375307
