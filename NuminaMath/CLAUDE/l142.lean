import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l142_14275

/-- Given a hyperbola with equation x²/32 - y²/4 = 1, the distance between its foci is 12 -/
theorem hyperbola_foci_distance (x y : ℝ) :
  x^2 / 32 - y^2 / 4 = 1 → ∃ c : ℝ, c = 6 ∧ 2 * c = 12 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l142_14275


namespace NUMINAMATH_CALUDE_john_game_period_duration_l142_14276

/-- Calculates the duration of each period in John's game --/
def period_duration (points_per_interval : ℕ) (total_points : ℕ) (num_periods : ℕ) : ℕ :=
  (total_points / points_per_interval * 4) / num_periods

/-- Proves that each period lasts 12 minutes given the game conditions --/
theorem john_game_period_duration :
  period_duration 7 42 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_john_game_period_duration_l142_14276


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l142_14268

theorem cow_chicken_problem (num_cows num_chickens : ℕ) : 
  (4 * num_cows + 2 * num_chickens = 2 * (num_cows + num_chickens) + 14) → 
  num_cows = 7 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l142_14268


namespace NUMINAMATH_CALUDE_trash_can_transfer_l142_14260

theorem trash_can_transfer (initial_veterans_park : ℕ) (final_veterans_park : ℕ) 
  (h1 : initial_veterans_park = 24)
  (h2 : final_veterans_park = 34) :
  ∃ (x : ℚ), 
    (x * initial_veterans_park : ℚ) = (final_veterans_park - initial_veterans_park : ℚ) ∧
    x = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_trash_can_transfer_l142_14260


namespace NUMINAMATH_CALUDE_divisibility_condition_l142_14239

theorem divisibility_condition (a : ℤ) : 
  0 ≤ a ∧ a < 13 ∧ (13 ∣ 51^2022 + a) → a = 12 := by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l142_14239


namespace NUMINAMATH_CALUDE_square_sum_ge_product_sum_l142_14265

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_product_sum_l142_14265


namespace NUMINAMATH_CALUDE_quadratic_negative_root_range_l142_14236

/-- Given a quadratic function f(x) = (m-2)x^2 - 4mx + 2m - 6,
    this theorem states the range of m for which f(x) has at least one negative root. -/
theorem quadratic_negative_root_range (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (m - 2) * x^2 - 4 * m * x + 2 * m - 6 = 0) ↔
  (1 ≤ m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_range_l142_14236


namespace NUMINAMATH_CALUDE_world_cup_merchandise_problem_l142_14218

def total_items : ℕ := 90
def ornament_cost : ℕ := 40
def pendant_cost : ℕ := 25
def total_cost : ℕ := 2850
def ornament_price : ℕ := 50
def pendant_price : ℕ := 30
def min_profit : ℕ := 725

theorem world_cup_merchandise_problem :
  ∃ (ornaments pendants : ℕ),
    ornaments + pendants = total_items ∧
    ornament_cost * ornaments + pendant_cost * pendants = total_cost ∧
    ornaments = 40 ∧
    pendants = 50 ∧
    (∀ m : ℕ,
      m ≤ total_items ∧
      (ornament_price - ornament_cost) * (total_items - m) + (pendant_price - pendant_cost) * m ≥ min_profit
      → m ≤ 35) :=
by sorry

end NUMINAMATH_CALUDE_world_cup_merchandise_problem_l142_14218


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l142_14295

theorem cube_volume_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_area_ratio : a^2 / b^2 = 9 / 25) : 
  b^3 / a^3 = 125 / 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l142_14295


namespace NUMINAMATH_CALUDE_binomial_variance_10_2_5_l142_14224

/-- The variance of a binomial distribution B(n, p) -/
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- Theorem: The variance of a binomial distribution B(10, 2/5) is 12/5 -/
theorem binomial_variance_10_2_5 :
  binomial_variance 10 (2/5) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_10_2_5_l142_14224


namespace NUMINAMATH_CALUDE_dans_cards_correct_l142_14288

/-- The number of Pokemon cards Sally initially had -/
def initial_cards : ℕ := 27

/-- The number of Pokemon cards Sally lost -/
def lost_cards : ℕ := 20

/-- The number of Pokemon cards Sally has now -/
def final_cards : ℕ := 48

/-- The number of Pokemon cards Dan gave Sally -/
def dans_cards : ℕ := 41

theorem dans_cards_correct : 
  initial_cards + dans_cards - lost_cards = final_cards :=
by sorry

end NUMINAMATH_CALUDE_dans_cards_correct_l142_14288


namespace NUMINAMATH_CALUDE_pyramid_section_ratio_l142_14250

/-- Represents a pyramid with a side edge and two points on it -/
structure Pyramid where
  -- Side edge length
  ab : ℝ
  -- Position of point K from A
  ak : ℝ
  -- Position of point M from A
  am : ℝ
  -- Conditions
  ab_pos : 0 < ab
  k_on_ab : 0 ≤ ak ∧ ak ≤ ab
  m_on_ab : 0 ≤ am ∧ am ≤ ab
  ak_eq_bm : ak = ab - am
  sections_area : (ak / ab)^2 + (am / ab)^2 = 2/3

/-- The main theorem -/
theorem pyramid_section_ratio (p : Pyramid) : (p.am - p.ak) / p.ab = 1 / Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_section_ratio_l142_14250


namespace NUMINAMATH_CALUDE_divisibility_of_2023_power_l142_14246

theorem divisibility_of_2023_power (n : ℕ) : 
  ∃ (k : ℕ), 2023^2023 - 2023^2021 = k * 2022 * 2023 * 2024 :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_2023_power_l142_14246


namespace NUMINAMATH_CALUDE_f_composite_value_l142_14251

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else a^x + b

theorem f_composite_value (a b : ℝ) :
  f 0 a b = 2 →
  f (-1) a b = 3 →
  f (f (-3) a b) a b = 2 := by
sorry

end NUMINAMATH_CALUDE_f_composite_value_l142_14251


namespace NUMINAMATH_CALUDE_equal_intercepts_condition_l142_14248

/-- The line equation ax + y - 2 - a = 0 has equal intercepts on x and y axes iff a = -2 or a = 1 -/
theorem equal_intercepts_condition (a : ℝ) : 
  (∃ (x y : ℝ), (a * x + y - 2 - a = 0 ∧ 
                ((x = 0 ∨ y = 0) ∧ 
                 (∀ x' y', a * x' + y' - 2 - a = 0 ∧ x' = 0 → y' = y) ∧
                 (∀ x' y', a * x' + y' - 2 - a = 0 ∧ y' = 0 → x' = x))))
  ↔ (a = -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_equal_intercepts_condition_l142_14248


namespace NUMINAMATH_CALUDE_chair_table_cost_fraction_l142_14201

theorem chair_table_cost_fraction :
  let table_cost : ℚ := 140
  let total_cost : ℚ := 220
  let chair_cost : ℚ := (total_cost - table_cost) / 4
  chair_cost / table_cost = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_chair_table_cost_fraction_l142_14201


namespace NUMINAMATH_CALUDE_double_roll_probability_l142_14293

def die_roll : Finset (Nat × Nat) := Finset.product (Finset.range 6) (Finset.range 6)

def favorable_outcomes : Finset (Nat × Nat) :=
  {(0, 1), (1, 3), (2, 5)}

theorem double_roll_probability :
  (favorable_outcomes.card : ℚ) / die_roll.card = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_double_roll_probability_l142_14293


namespace NUMINAMATH_CALUDE_sheep_to_horse_ratio_l142_14283

def daily_horse_food_per_horse : ℕ := 230
def total_daily_horse_food : ℕ := 12880
def number_of_sheep : ℕ := 56

theorem sheep_to_horse_ratio :
  (total_daily_horse_food / daily_horse_food_per_horse = number_of_sheep) →
  (number_of_sheep : ℚ) / (total_daily_horse_food / daily_horse_food_per_horse : ℚ) = 1 := by
sorry

end NUMINAMATH_CALUDE_sheep_to_horse_ratio_l142_14283


namespace NUMINAMATH_CALUDE_sandy_change_is_three_l142_14219

/-- Represents the cost and quantity of a drink order -/
structure DrinkOrder where
  name : String
  price : ℚ
  quantity : ℕ

/-- Calculates the total cost of a drink order -/
def orderCost (order : DrinkOrder) : ℚ :=
  order.price * order.quantity

/-- Calculates the total cost of multiple drink orders -/
def totalCost (orders : List DrinkOrder) : ℚ :=
  orders.map orderCost |>.sum

/-- Calculates the change from a given amount -/
def calculateChange (paid : ℚ) (cost : ℚ) : ℚ :=
  paid - cost

theorem sandy_change_is_three :
  let orders : List DrinkOrder := [
    { name := "Cappuccino", price := 2, quantity := 3 },
    { name := "Iced Tea", price := 3, quantity := 2 },
    { name := "Cafe Latte", price := 1.5, quantity := 2 },
    { name := "Espresso", price := 1, quantity := 2 }
  ]
  let total := totalCost orders
  let paid := 20
  calculateChange paid total = 3 := by sorry

end NUMINAMATH_CALUDE_sandy_change_is_three_l142_14219


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l142_14215

theorem digit_sum_puzzle (a p v m t r : ℕ) 
  (h1 : a + p = v)
  (h2 : v + m = t)
  (h3 : t + a = r)
  (h4 : p + m + r = 18)
  (h5 : a ≠ 0 ∧ p ≠ 0 ∧ v ≠ 0 ∧ m ≠ 0 ∧ t ≠ 0 ∧ r ≠ 0)
  (h6 : a ≠ p ∧ a ≠ v ∧ a ≠ m ∧ a ≠ t ∧ a ≠ r ∧
        p ≠ v ∧ p ≠ m ∧ p ≠ t ∧ p ≠ r ∧
        v ≠ m ∧ v ≠ t ∧ v ≠ r ∧
        m ≠ t ∧ m ≠ r ∧
        t ≠ r) :
  t = 9 := by
  sorry


end NUMINAMATH_CALUDE_digit_sum_puzzle_l142_14215


namespace NUMINAMATH_CALUDE_perpendicular_vectors_y_value_l142_14273

theorem perpendicular_vectors_y_value :
  let a : Fin 3 → ℝ := ![1, 2, 6]
  let b : Fin 3 → ℝ := ![2, y, -1]
  (∀ i : Fin 3, (a • b) = 0) → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_y_value_l142_14273


namespace NUMINAMATH_CALUDE_school_duration_in_minutes_l142_14230

-- Define the start and end times
def start_time : ℕ := 7
def end_time : ℕ := 11

-- Define the duration in hours
def duration_hours : ℕ := end_time - start_time

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem school_duration_in_minutes :
  duration_hours * minutes_per_hour = 240 :=
sorry

end NUMINAMATH_CALUDE_school_duration_in_minutes_l142_14230


namespace NUMINAMATH_CALUDE_surface_area_is_34_l142_14220

/-- A three-dimensional figure composed of unit cubes -/
structure CubeFigure where
  num_cubes : ℕ
  cube_side_length : ℝ
  top_area : ℝ
  bottom_area : ℝ
  front_area : ℝ
  back_area : ℝ
  left_area : ℝ
  right_area : ℝ

/-- The surface area of a CubeFigure -/
def surface_area (figure : CubeFigure) : ℝ :=
  figure.top_area + figure.bottom_area + figure.front_area + 
  figure.back_area + figure.left_area + figure.right_area

/-- Theorem stating that the surface area of the given figure is 34 -/
theorem surface_area_is_34 (figure : CubeFigure) 
  (h1 : figure.num_cubes = 10)
  (h2 : figure.cube_side_length = 1)
  (h3 : figure.top_area = 6)
  (h4 : figure.bottom_area = 6)
  (h5 : figure.front_area = 5)
  (h6 : figure.back_area = 5)
  (h7 : figure.left_area = 6)
  (h8 : figure.right_area = 6) :
  surface_area figure = 34 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_is_34_l142_14220


namespace NUMINAMATH_CALUDE_machine_time_calculation_l142_14253

/-- Given a machine that can make a certain number of shirts per minute
    and has made a total number of shirts, calculate the time it worked. -/
def machine_working_time (shirts_per_minute : ℕ) (total_shirts : ℕ) : ℚ :=
  total_shirts / shirts_per_minute

/-- Theorem stating that for a machine making 3 shirts per minute
    and having made 6 shirts in total, it worked for 2 minutes. -/
theorem machine_time_calculation :
  machine_working_time 3 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_machine_time_calculation_l142_14253


namespace NUMINAMATH_CALUDE_minimum_days_to_plant_trees_l142_14278

def trees_planted (n : ℕ) : ℕ := 2^n - 1

theorem minimum_days_to_plant_trees : 
  (∀ k : ℕ, k < 7 → trees_planted k < 100) ∧ 
  trees_planted 7 ≥ 100 := by
sorry

end NUMINAMATH_CALUDE_minimum_days_to_plant_trees_l142_14278


namespace NUMINAMATH_CALUDE_shoe_selection_theorem_l142_14282

/-- The number of pairs of shoes in the bag -/
def total_pairs : ℕ := 10

/-- The number of shoes randomly selected -/
def selected_shoes : ℕ := 4

/-- The number of ways to select 4 shoes such that none of them form a pair -/
def no_pairs : ℕ := 3360

/-- The number of ways to select 4 shoes such that exactly 2 pairs are formed -/
def two_pairs : ℕ := 45

/-- The number of ways to select 4 shoes such that 2 shoes form a pair and the other 2 do not -/
def one_pair : ℕ := 1440

theorem shoe_selection_theorem :
  (Nat.choose total_pairs selected_shoes * 2^selected_shoes = no_pairs) ∧
  (Nat.choose total_pairs 2 = two_pairs) ∧
  (total_pairs * Nat.choose (total_pairs - 1) 2 * 2^2 = one_pair) :=
by sorry

end NUMINAMATH_CALUDE_shoe_selection_theorem_l142_14282


namespace NUMINAMATH_CALUDE_fence_pole_count_l142_14247

/-- Represents a rectangular fence with an internal divider -/
structure RectangularFence where
  longer_side : ℕ
  shorter_side : ℕ
  has_internal_divider : Bool

/-- Calculates the total number of poles needed for a rectangular fence with an internal divider -/
def total_poles (fence : RectangularFence) : ℕ :=
  let perimeter_poles := 2 * (fence.longer_side + fence.shorter_side) - 4
  let internal_poles := if fence.has_internal_divider then fence.shorter_side - 1 else 0
  perimeter_poles + internal_poles

/-- Theorem stating that a rectangular fence with 35 poles on the longer side, 
    27 poles on the shorter side, and an internal divider needs 146 poles in total -/
theorem fence_pole_count : 
  let fence := RectangularFence.mk 35 27 true
  total_poles fence = 146 := by sorry

end NUMINAMATH_CALUDE_fence_pole_count_l142_14247


namespace NUMINAMATH_CALUDE_log_difference_equals_negative_one_l142_14205

theorem log_difference_equals_negative_one :
  ∀ (log : ℝ → ℝ → ℝ),
    (∀ (a N : ℝ), a > 0 → a ≠ 1 → ∃ b, N = a^b → log a N = b) →
    9 = 3^2 →
    125 = 5^3 →
    log 3 9 - log 5 125 = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_negative_one_l142_14205


namespace NUMINAMATH_CALUDE_cake_radius_increase_l142_14249

theorem cake_radius_increase (c₁ c₂ : ℝ) (h₁ : c₁ = 30) (h₂ : c₂ = 37.5) :
  (c₂ / (2 * Real.pi)) - (c₁ / (2 * Real.pi)) = 7.5 / (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cake_radius_increase_l142_14249


namespace NUMINAMATH_CALUDE_final_number_is_88_or_94_l142_14209

/-- Represents the two allowed operations on the number -/
inductive Operation
| replace_with_diff
| increase_decrease

/-- The initial number with 98 eights -/
def initial_number : Nat := 88888888  -- Simplified representation

/-- Applies a single operation to a number -/
def apply_operation (n : Nat) (op : Operation) : Nat :=
  match op with
  | Operation.replace_with_diff => sorry
  | Operation.increase_decrease => sorry

/-- Applies a sequence of operations to a number -/
def apply_operations (n : Nat) (ops : List Operation) : Nat :=
  match ops with
  | [] => n
  | op :: rest => apply_operations (apply_operation n op) rest

/-- The theorem stating that the final two-digit number must be 88 or 94 -/
theorem final_number_is_88_or_94 (ops : List Operation) :
  ∃ (result : Nat), apply_operations initial_number ops = result ∧ (result = 88 ∨ result = 94) :=
sorry

end NUMINAMATH_CALUDE_final_number_is_88_or_94_l142_14209


namespace NUMINAMATH_CALUDE_denise_crayons_l142_14234

/-- The number of friends Denise shares her crayons with -/
def num_friends : ℕ := 30

/-- The number of crayons each friend receives -/
def crayons_per_friend : ℕ := 7

/-- The total number of crayons Denise has -/
def total_crayons : ℕ := num_friends * crayons_per_friend

/-- Theorem stating that Denise has 210 crayons -/
theorem denise_crayons : total_crayons = 210 := by
  sorry

end NUMINAMATH_CALUDE_denise_crayons_l142_14234


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l142_14210

def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∃ (p : Nat), is_prime p ∧ digit_sum p = 23 ∧
  ∀ (q : Nat), is_prime q ∧ digit_sum q = 23 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l142_14210


namespace NUMINAMATH_CALUDE_equation_solution_l142_14287

theorem equation_solution (x : ℚ) : 9 / (5 + 3 / x) = 1 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l142_14287


namespace NUMINAMATH_CALUDE_hill_depth_ratio_l142_14254

/-- Given a hill with its base 300m above the seabed and a total height of 900m,
    prove that the ratio of the depth from the base to the seabed
    to the total height of the hill is 1/3. -/
theorem hill_depth_ratio (base_height : ℝ) (total_height : ℝ) :
  base_height = 300 →
  total_height = 900 →
  base_height / total_height = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hill_depth_ratio_l142_14254


namespace NUMINAMATH_CALUDE_unique_arrangement_l142_14284

-- Define the shapes and colors
inductive Shape : Type
| Triangle : Shape
| Circle : Shape
| Rectangle : Shape
| Rhombus : Shape

inductive Color : Type
| Red : Color
| Blue : Color
| Yellow : Color
| Green : Color

-- Define the position type
inductive Position : Type
| First : Position
| Second : Position
| Third : Position
| Fourth : Position

-- Define the figure type
structure Figure :=
(shape : Shape)
(color : Color)
(position : Position)

def Arrangement := List Figure

-- Define the conditions
def redBetweenBlueAndGreen (arr : Arrangement) : Prop := sorry
def rhombusRightOfYellow (arr : Arrangement) : Prop := sorry
def circleRightOfTriangleAndRhombus (arr : Arrangement) : Prop := sorry
def triangleNotAtEdge (arr : Arrangement) : Prop := sorry
def blueAndYellowNotAdjacent (arr : Arrangement) : Prop := sorry

-- Define the correct arrangement
def correctArrangement : Arrangement := [
  ⟨Shape.Rectangle, Color.Yellow, Position.First⟩,
  ⟨Shape.Rhombus, Color.Green, Position.Second⟩,
  ⟨Shape.Triangle, Color.Red, Position.Third⟩,
  ⟨Shape.Circle, Color.Blue, Position.Fourth⟩
]

-- Theorem statement
theorem unique_arrangement :
  ∀ (arr : Arrangement),
    (redBetweenBlueAndGreen arr) →
    (rhombusRightOfYellow arr) →
    (circleRightOfTriangleAndRhombus arr) →
    (triangleNotAtEdge arr) →
    (blueAndYellowNotAdjacent arr) →
    (arr = correctArrangement) :=
by sorry

end NUMINAMATH_CALUDE_unique_arrangement_l142_14284


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l142_14252

theorem cos_sixty_degrees : Real.cos (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l142_14252


namespace NUMINAMATH_CALUDE_inscribed_square_area_l142_14263

/-- The area of a square inscribed in a quadrant of a circle with radius 10 is equal to 40 -/
theorem inscribed_square_area (r : ℝ) (s : ℝ) (h1 : r = 10) (h2 : s > 0) 
  (h3 : s^2 + (3/2 * s)^2 = r^2) : s^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l142_14263


namespace NUMINAMATH_CALUDE_positive_number_problem_l142_14266

theorem positive_number_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x - 4 = 21 / x)
  (eq2 : x + y^2 = 45)
  (eq3 : y * z = x^3) :
  x = 7 ∧ y = Real.sqrt 38 ∧ z = 343 * Real.sqrt 38 / 38 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_problem_l142_14266


namespace NUMINAMATH_CALUDE_sum_of_two_and_repeating_third_l142_14258

-- Define the repeating decimal 0.3333...
def repeating_third : ℚ := 1 / 3

-- Theorem statement
theorem sum_of_two_and_repeating_third :
  2 + repeating_third = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_two_and_repeating_third_l142_14258


namespace NUMINAMATH_CALUDE_simplify_expression_l142_14256

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l142_14256


namespace NUMINAMATH_CALUDE_usual_time_to_catch_bus_l142_14221

/-- The usual time to catch the bus, given that walking with 4/5 of the usual speed
    results in missing the bus by 4 minutes, is 16 minutes. -/
theorem usual_time_to_catch_bus (T : ℝ) (S : ℝ) : T > 0 → S > 0 → S * T = (4/5 * S) * (T + 4) → T = 16 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_to_catch_bus_l142_14221


namespace NUMINAMATH_CALUDE_cricket_players_l142_14202

/-- The number of students who like to play basketball -/
def B : ℕ := 7

/-- The number of students who like to play both basketball and cricket -/
def B_and_C : ℕ := 5

/-- The number of students who like to play basketball or cricket or both -/
def B_or_C : ℕ := 10

/-- The number of students who like to play cricket -/
def C : ℕ := B_or_C - B + B_and_C

theorem cricket_players : C = 8 := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_l142_14202


namespace NUMINAMATH_CALUDE_min_value_expression_l142_14226

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 2) :
  (1/a + 1/(2*b) + 4*a*b) ≥ 4 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 2 ∧ (1/a + 1/(2*b) + 4*a*b) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l142_14226


namespace NUMINAMATH_CALUDE_cos_product_20_40_60_80_l142_14280

theorem cos_product_20_40_60_80 : 
  Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (60 * π / 180) * Real.cos (80 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_20_40_60_80_l142_14280


namespace NUMINAMATH_CALUDE_green_area_growth_rate_l142_14296

theorem green_area_growth_rate :
  ∀ x : ℝ, (1 + x)^2 = 1.44 → x = 0.2 :=
by
  sorry

end NUMINAMATH_CALUDE_green_area_growth_rate_l142_14296


namespace NUMINAMATH_CALUDE_nine_rings_five_classes_l142_14272

/-- Represents the number of classes in a school day based on bell rings --/
def number_of_classes (total_rings : ℕ) : ℕ :=
  let completed_classes := (total_rings - 1) / 2
  completed_classes + 1

/-- Theorem stating that 9 total bell rings corresponds to 5 classes --/
theorem nine_rings_five_classes : number_of_classes 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_nine_rings_five_classes_l142_14272


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_problem_l142_14291

-- Define a monic quartic polynomial
def is_monic_quartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

-- Define the polynomial p with given conditions
def p : ℝ → ℝ := sorry

-- State the theorem
theorem monic_quartic_polynomial_problem :
  is_monic_quartic p ∧ 
  p 1 = 2 ∧ 
  p 2 = 7 ∧ 
  p 3 = 10 ∧ 
  p 4 = 17 → 
  p 5 = 26 := by sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_problem_l142_14291


namespace NUMINAMATH_CALUDE_quadratic_to_linear_equations_l142_14290

theorem quadratic_to_linear_equations :
  ∀ x y : ℝ, x^2 - 4*x*y + 4*y^2 = 4 ↔ (x - 2*y + 2 = 0 ∨ x - 2*y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_to_linear_equations_l142_14290


namespace NUMINAMATH_CALUDE_percentage_problem_l142_14286

theorem percentage_problem (P : ℝ) : 
  (P ≥ 0 ∧ P ≤ 100) → 
  (P / 100) * 3200 = (20 / 100) * 650 + 190 → 
  P = 10 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l142_14286


namespace NUMINAMATH_CALUDE_third_student_number_l142_14255

theorem third_student_number (A B C D : ℤ) 
  (sum_eq : A + B + C + D = 531)
  (diff_eq : A + B = C + D + 31)
  (third_fourth_diff : C = D + 22) :
  C = 136 := by
  sorry

end NUMINAMATH_CALUDE_third_student_number_l142_14255


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l142_14204

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_common_difference 
  (a₁ d : ℚ) :
  arithmetic_sequence a₁ d 5 + arithmetic_sequence a₁ d 6 = -10 ∧
  sum_arithmetic_sequence a₁ d 14 = -14 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l142_14204


namespace NUMINAMATH_CALUDE_prob_three_same_suit_standard_deck_l142_14232

/-- A standard deck of cards --/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (black_suits : Nat)
  (red_suits : Nat)

/-- Standard 52-card deck --/
def standard_deck : Deck :=
  { cards := 52,
    ranks := 13,
    suits := 4,
    black_suits := 2,
    red_suits := 2 }

/-- Probability of drawing three cards of the same specific suit --/
def prob_three_same_suit (d : Deck) : Rat :=
  (d.ranks * (d.ranks - 1) * (d.ranks - 2)) / (d.cards * (d.cards - 1) * (d.cards - 2))

/-- Theorem stating the probability of drawing three cards of the same specific suit from a standard deck --/
theorem prob_three_same_suit_standard_deck :
  prob_three_same_suit standard_deck = 11 / 850 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_same_suit_standard_deck_l142_14232


namespace NUMINAMATH_CALUDE_log_square_equals_twenty_l142_14240

theorem log_square_equals_twenty (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (hx_neq_one : x ≠ 1) (hy_neq_one : y ≠ 1)
  (h_log : Real.log x / Real.log 2 = Real.log 8 / Real.log y)
  (h_product : x * y = 128) : 
  (Real.log (x / y) / Real.log 2)^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_log_square_equals_twenty_l142_14240


namespace NUMINAMATH_CALUDE_min_difference_for_equal_f_values_l142_14297

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then Real.log x else (1/2) * x + (1/2)

theorem min_difference_for_equal_f_values :
  ∃ (min_diff : ℝ),
    min_diff = 3 - 2 * Real.log 2 ∧
    ∀ (m n : ℝ), m < n → f m = f n → n - m ≥ min_diff :=
by sorry

end NUMINAMATH_CALUDE_min_difference_for_equal_f_values_l142_14297


namespace NUMINAMATH_CALUDE_interest_rate_difference_l142_14225

/-- Given a principal amount, time period, and difference in interest earned,
    calculate the difference between two interest rates. -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_diff : ℝ)
  (h1 : principal = 200)
  (h2 : time = 10)
  (h3 : interest_diff = 100) :
  (interest_diff / (principal * time)) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l142_14225


namespace NUMINAMATH_CALUDE_alcohol_concentration_after_addition_l142_14262

/-- Proves that adding 5.5 liters of alcohol and 4.5 liters of water to a 40-liter solution
    with 5% alcohol concentration results in a 15% alcohol solution. -/
theorem alcohol_concentration_after_addition :
  let initial_volume : ℝ := 40
  let initial_concentration : ℝ := 0.05
  let added_alcohol : ℝ := 5.5
  let added_water : ℝ := 4.5
  let final_concentration : ℝ := 0.15
  let final_volume : ℝ := initial_volume + added_alcohol + added_water
  initial_volume * initial_concentration + added_alcohol =
    final_volume * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_concentration_after_addition_l142_14262


namespace NUMINAMATH_CALUDE_equal_roots_cubic_l142_14212

theorem equal_roots_cubic (k : ℝ) :
  (∃ a b : ℝ, (3 * a^3 + 9 * a^2 - 150 * a + k = 0) ∧
              (3 * b^3 + 9 * b^2 - 150 * b + k = 0) ∧
              (a ≠ b)) ∧
  (∃ x : ℝ, (3 * x^3 + 9 * x^2 - 150 * x + k = 0) ∧
            (∃ y : ℝ, y ≠ x ∧ 3 * y^3 + 9 * y^2 - 150 * y + k = 0)) ∧
  (k > 0) →
  k = 950 / 27 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_cubic_l142_14212


namespace NUMINAMATH_CALUDE_mark_kate_difference_l142_14243

/-- Represents the hours charged by each person to the project -/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ
  sam : ℕ

/-- Conditions for the project hours -/
def valid_project_hours (h : ProjectHours) : Prop :=
  h.pat = 2 * h.kate ∧
  h.mark = 3 * h.pat ∧
  h.sam = (h.pat + h.mark) / 2 ∧
  h.kate + h.pat + h.mark + h.sam = 198

theorem mark_kate_difference (h : ProjectHours) 
  (hvalid : valid_project_hours h) : h.mark - h.kate = 75 := by
  sorry

end NUMINAMATH_CALUDE_mark_kate_difference_l142_14243


namespace NUMINAMATH_CALUDE_euro_calculation_l142_14281

/-- The € operation as defined in the problem -/
def euro (x y z : ℕ) : ℕ := (2 * x * y + y^2) * z^3

/-- The statement to be proven -/
theorem euro_calculation : euro 7 (euro 4 5 3) 2 = 24844760 := by
  sorry

end NUMINAMATH_CALUDE_euro_calculation_l142_14281


namespace NUMINAMATH_CALUDE_empty_chests_count_l142_14271

/-- Represents a nested chest system -/
structure ChestSystem where
  total_chests : ℕ
  non_empty_chests : ℕ
  hNonEmpty : non_empty_chests = 2006
  hTotal : total_chests = 10 * non_empty_chests + 1

/-- The number of empty chests in the system -/
def empty_chests (cs : ChestSystem) : ℕ :=
  cs.total_chests - (cs.non_empty_chests + 1)

/-- Theorem stating the number of empty chests in the given system -/
theorem empty_chests_count (cs : ChestSystem) : empty_chests cs = 18054 := by
  sorry

end NUMINAMATH_CALUDE_empty_chests_count_l142_14271


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l142_14206

/-- Given a cylinder with volume 72π cm³ and a cone with the same radius as the cylinder 
    and half its height, the volume of the cone is 12π cm³ -/
theorem cone_volume_from_cylinder (r h : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  π * r^2 * h = 72 * π → (1/3) * π * r^2 * (h/2) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l142_14206


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l142_14241

def vector_a (t : ℝ) : ℝ × ℝ := (1, t)
def vector_b (t : ℝ) : ℝ × ℝ := (t, 9)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_t_value :
  ∀ t : ℝ, parallel (vector_a t) (vector_b t) → t = 3 ∨ t = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l142_14241


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l142_14294

/-- Given a complex number z = a^2 - 1 + (a+1)i where a ∈ ℝ and z is a pure imaginary number,
    the imaginary part of 1/(z+a) is -2/5 -/
theorem imaginary_part_of_reciprocal (a : ℝ) (z : ℂ) : 
  z = a^2 - 1 + (a + 1) * I → 
  z.re = 0 → 
  z.im ≠ 0 → 
  Complex.im (1 / (z + a)) = -2/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l142_14294


namespace NUMINAMATH_CALUDE_expression_bounds_l142_14245

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) : 
  2 * Real.sqrt 2 + 2 ≤ 
    Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt ((b+1)^2 + (2 - c)^2) + 
    Real.sqrt ((c-1)^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2) ∧
  Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt ((b+1)^2 + (2 - c)^2) + 
  Real.sqrt ((c-1)^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l142_14245


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l142_14270

/-- The number of boys in a class given specific height conditions -/
theorem number_of_boys_in_class (n : ℕ) 
  (h1 : (n : ℝ) * 182 = (n : ℝ) * 182 + 166 - 106)
  (h2 : (n : ℝ) * 180 = (n : ℝ) * 182 + 106 - 166) : n = 30 := by
  sorry


end NUMINAMATH_CALUDE_number_of_boys_in_class_l142_14270


namespace NUMINAMATH_CALUDE_second_number_is_30_l142_14207

theorem second_number_is_30 (a b c : ℚ) : 
  a + b + c = 110 ∧ 
  a = 2 * b ∧ 
  c = (1 / 3) * a → 
  b = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_30_l142_14207


namespace NUMINAMATH_CALUDE_problem_solution_l142_14208

-- Define the propositions p and q
def p : Prop := ∀ x, 0 < x → x < Real.pi / 2 → Real.sin x > x
def q : Prop := ∀ x, 0 < x → x < Real.pi / 2 → Real.tan x > x

-- Theorem to prove
theorem problem_solution : (p ∨ q) ∧ (¬p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l142_14208


namespace NUMINAMATH_CALUDE_person_on_throne_l142_14244

-- Define the possible characteristics of the person
inductive PersonType
| Liar
| Monkey
| Knight

-- Define the statement made by the person
def statement (p : PersonType) : Prop :=
  p = PersonType.Liar ∨ p = PersonType.Monkey

-- Theorem to prove
theorem person_on_throne (p : PersonType) (h : statement p) : 
  p = PersonType.Monkey ∧ p ≠ PersonType.Liar :=
sorry

end NUMINAMATH_CALUDE_person_on_throne_l142_14244


namespace NUMINAMATH_CALUDE_sum_of_squares_power_of_three_l142_14257

theorem sum_of_squares_power_of_three (n : ℕ) :
  ∃ x y z : ℤ, (Nat.gcd (Nat.gcd x.natAbs y.natAbs) z.natAbs = 1) ∧
  (x^2 + y^2 + z^2 = 3^(2^n)) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_power_of_three_l142_14257


namespace NUMINAMATH_CALUDE_total_population_l142_14213

/-- Represents the population of a school -/
structure SchoolPopulation where
  boys : ℕ
  girls : ℕ
  teachers : ℕ
  staff : ℕ

/-- The ratios in the school population -/
def school_ratios (p : SchoolPopulation) : Prop :=
  p.boys = 4 * p.girls ∧ 
  p.girls = 8 * p.teachers ∧ 
  p.staff = 2 * p.teachers

theorem total_population (p : SchoolPopulation) 
  (h : school_ratios p) : 
  p.boys + p.girls + p.teachers + p.staff = (43 * p.boys) / 32 := by
  sorry

end NUMINAMATH_CALUDE_total_population_l142_14213


namespace NUMINAMATH_CALUDE_ceiling_sqrt_196_l142_14203

theorem ceiling_sqrt_196 : ⌈Real.sqrt 196⌉ = 14 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_196_l142_14203


namespace NUMINAMATH_CALUDE_cubic_equation_root_sum_l142_14222

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0 and roots 4 and -1, prove b+c = -13a -/
theorem cubic_equation_root_sum (a b c d : ℝ) (ha : a ≠ 0) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 4 ∨ x = -1 ∨ x = -(b + c + 13 * a) / a) →
  b + c = -13 * a :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_root_sum_l142_14222


namespace NUMINAMATH_CALUDE_shortest_player_height_l142_14237

theorem shortest_player_height (tallest_height : Float) (height_difference : Float) :
  tallest_height = 77.75 →
  height_difference = 9.5 →
  tallest_height - height_difference = 68.25 := by
sorry

end NUMINAMATH_CALUDE_shortest_player_height_l142_14237


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2014_l142_14267

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_2014 :
  ∃ n : ℕ, arithmetic_sequence 1 3 n = 2014 ∧ n = 672 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2014_l142_14267


namespace NUMINAMATH_CALUDE_system_solution_unique_l142_14299

theorem system_solution_unique :
  ∃! (x y : ℝ), x + 2*y = 9 ∧ 3*x - 2*y = -1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l142_14299


namespace NUMINAMATH_CALUDE_parabola_translation_l142_14264

-- Define the original and transformed parabolas
def original_parabola (x : ℝ) : ℝ := x^2
def transformed_parabola (x : ℝ) : ℝ := x^2 - 5

-- Define the translation
def translation (y : ℝ) : ℝ := y - 5

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, transformed_parabola x = translation (original_parabola x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l142_14264


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l142_14277

theorem divisible_by_twelve (n : Nat) : n ≤ 9 → 5148 = 514 * 10 + n ↔ (514 * 10 + n) % 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l142_14277


namespace NUMINAMATH_CALUDE_unique_n_divisibility_l142_14229

theorem unique_n_divisibility : ∃! (n : ℕ), n > 1 ∧
  ∀ (p : ℕ), Prime p → (p ∣ (n^6 - 1)) → (p ∣ ((n^3 - 1) * (n^2 - 1))) :=
by
  -- The unique n that satisfies the condition is 2
  use 2
  sorry

end NUMINAMATH_CALUDE_unique_n_divisibility_l142_14229


namespace NUMINAMATH_CALUDE_erased_number_l142_14298

theorem erased_number (n : ℕ) (x : ℕ) : 
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1) = 45 / 4 →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_erased_number_l142_14298


namespace NUMINAMATH_CALUDE_unclaimed_candy_fraction_verify_actual_taken_l142_14200

/-- Represents the fraction of candy taken by each person -/
structure CandyFraction where
  al : Rat
  bert : Rat
  carl : Rat

/-- The intended ratio for candy distribution -/
def intended_ratio : CandyFraction :=
  { al := 4/9, bert := 1/3, carl := 2/9 }

/-- The actual amount of candy taken by each person -/
def actual_taken : CandyFraction :=
  { al := 4/9, bert := 5/27, carl := 20/243 }

/-- The theorem stating the fraction of candy that goes unclaimed -/
theorem unclaimed_candy_fraction :
  1 - (actual_taken.al + actual_taken.bert + actual_taken.carl) = 230/243 := by
  sorry

/-- Verify that the actual taken amounts are correct based on the problem description -/
theorem verify_actual_taken :
  actual_taken.al = intended_ratio.al ∧
  actual_taken.bert = intended_ratio.bert * (1 - actual_taken.al) ∧
  actual_taken.carl = intended_ratio.carl * (1 - actual_taken.al - actual_taken.bert) := by
  sorry

end NUMINAMATH_CALUDE_unclaimed_candy_fraction_verify_actual_taken_l142_14200


namespace NUMINAMATH_CALUDE_exists_dihedral_equal_edge_not_equal_exists_edge_equal_dihedral_not_equal_dihedral_angles_neither_necessary_nor_sufficient_l142_14214

/-- A quadrilateral pyramid with vertex V and base ABCD. -/
structure QuadrilateralPyramid where
  V : Point
  A : Point
  B : Point
  C : Point
  D : Point

/-- The property that all dihedral angles between adjacent faces are equal. -/
def all_dihedral_angles_equal (pyramid : QuadrilateralPyramid) : Prop :=
  sorry

/-- The property that all angles between adjacent edges are equal. -/
def all_edge_angles_equal (pyramid : QuadrilateralPyramid) : Prop :=
  sorry

/-- There exists a pyramid where all dihedral angles are equal but not all edge angles are equal. -/
theorem exists_dihedral_equal_edge_not_equal :
  ∃ (pyramid : QuadrilateralPyramid),
    all_dihedral_angles_equal pyramid ∧ ¬all_edge_angles_equal pyramid :=
  sorry

/-- There exists a pyramid where all edge angles are equal but not all dihedral angles are equal. -/
theorem exists_edge_equal_dihedral_not_equal :
  ∃ (pyramid : QuadrilateralPyramid),
    all_edge_angles_equal pyramid ∧ ¬all_dihedral_angles_equal pyramid :=
  sorry

/-- The main theorem stating that the equality of dihedral angles is neither necessary nor sufficient for the equality of edge angles. -/
theorem dihedral_angles_neither_necessary_nor_sufficient :
  (∃ (pyramid : QuadrilateralPyramid), all_dihedral_angles_equal pyramid ∧ ¬all_edge_angles_equal pyramid) ∧
  (∃ (pyramid : QuadrilateralPyramid), all_edge_angles_equal pyramid ∧ ¬all_dihedral_angles_equal pyramid) :=
  sorry

end NUMINAMATH_CALUDE_exists_dihedral_equal_edge_not_equal_exists_edge_equal_dihedral_not_equal_dihedral_angles_neither_necessary_nor_sufficient_l142_14214


namespace NUMINAMATH_CALUDE_equal_distribution_contribution_l142_14235

def earnings : List ℕ := [10, 15, 20, 25, 30, 50]

theorem equal_distribution_contribution :
  let total := earnings.sum
  let equal_share := total / earnings.length
  let max_earner := earnings.maximum?
  max_earner.map (λ m => m - equal_share) = some 25 := by sorry

end NUMINAMATH_CALUDE_equal_distribution_contribution_l142_14235


namespace NUMINAMATH_CALUDE_combined_tax_rate_l142_14279

/-- Given two individuals with different tax rates and incomes, calculate their combined tax rate -/
theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.40) 
  (h2 : mindy_rate = 0.25) 
  (h3 : income_ratio = 4) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l142_14279


namespace NUMINAMATH_CALUDE_fifth_to_third_grade_ratio_l142_14211

/-- Proves that the ratio of fifth-graders to third-graders is 1:2 given the conditions -/
theorem fifth_to_third_grade_ratio : 
  ∀ (third_graders fourth_graders fifth_graders : ℕ),
  third_graders = 20 →
  fourth_graders = 2 * third_graders →
  third_graders + fourth_graders + fifth_graders = 70 →
  fifth_graders.gcd third_graders * 2 = fifth_graders ∧ 
  fifth_graders.gcd third_graders * 1 = fifth_graders.gcd third_graders :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_to_third_grade_ratio_l142_14211


namespace NUMINAMATH_CALUDE_student_correct_sums_l142_14233

theorem student_correct_sums (total : ℕ) (correct : ℕ) (wrong : ℕ) : 
  total = 24 → wrong = 2 * correct → total = correct + wrong → correct = 8 := by
  sorry

end NUMINAMATH_CALUDE_student_correct_sums_l142_14233


namespace NUMINAMATH_CALUDE_m_less_than_n_min_sum_a_b_l142_14269

-- Define the variables and conditions
variables (a b : ℝ) (m n : ℝ)

-- Define the relationships between variables
def m_def : m = a * b + 1 := by sorry
def n_def : n = a + b := by sorry

-- Part 1: Prove m < n when a > 1 and b < 1
theorem m_less_than_n (ha : a > 1) (hb : b < 1) : m < n := by sorry

-- Part 2: Prove minimum value of a + b is 16 when a > 1, b > 1, and m - n = 49
theorem min_sum_a_b (ha : a > 1) (hb : b > 1) (h_diff : m - n = 49) :
  ∃ (min_sum : ℝ), min_sum = 16 ∧ a + b ≥ min_sum := by sorry

end NUMINAMATH_CALUDE_m_less_than_n_min_sum_a_b_l142_14269


namespace NUMINAMATH_CALUDE_san_antonio_bound_bus_encounters_l142_14231

-- Define the time type (in minutes since midnight)
def Time := ℕ

-- Define the bus schedules
def austin_to_san_antonio_schedule (t : Time) : Prop :=
  ∃ n : ℕ, t = 360 + 120 * n

def san_antonio_to_austin_schedule (t : Time) : Prop :=
  ∃ n : ℕ, t = 390 + 60 * n

-- Define the travel time
def travel_time : ℕ := 360  -- 6 hours in minutes

-- Define the function to count encounters
def count_encounters (start_time : Time) : ℕ :=
  sorry  -- Implementation details omitted

-- Theorem statement
theorem san_antonio_bound_bus_encounters :
  ∀ (start_time : Time),
    san_antonio_to_austin_schedule start_time →
    count_encounters start_time = 2 :=
by sorry

end NUMINAMATH_CALUDE_san_antonio_bound_bus_encounters_l142_14231


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l142_14292

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l142_14292


namespace NUMINAMATH_CALUDE_birds_in_marsh_l142_14274

theorem birds_in_marsh (geese ducks : ℕ) (h1 : geese = 58) (h2 : ducks = 37) :
  geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_marsh_l142_14274


namespace NUMINAMATH_CALUDE_multiply_and_add_equality_l142_14259

theorem multiply_and_add_equality : 24 * 42 + 58 * 24 + 12 * 24 = 2688 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_equality_l142_14259


namespace NUMINAMATH_CALUDE_complex_function_property_l142_14223

open Complex

/-- Given a function f(z) = (a+bi)z where a and b are positive real numbers,
    if f(z) is equidistant from z and 2+2i for all complex z, and |a+bi| = 10,
    then b^2 = 287/17 -/
theorem complex_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ z : ℂ, ‖(a + b * I) * z - z‖ = ‖(a + b * I) * z - (2 + 2 * I)‖) →
  ‖(a : ℂ) + b * I‖ = 10 →
  b^2 = 287/17 := by
  sorry

end NUMINAMATH_CALUDE_complex_function_property_l142_14223


namespace NUMINAMATH_CALUDE_hexagon_side_length_l142_14289

/-- A regular hexagon with perimeter 60 inches has sides of length 10 inches. -/
theorem hexagon_side_length : ∀ (side_length : ℝ), 
  side_length > 0 →
  6 * side_length = 60 →
  side_length = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l142_14289


namespace NUMINAMATH_CALUDE_total_cement_is_15_1_l142_14285

/-- The amount of cement used for Lexi's street in tons -/
def lexiStreetCement : ℝ := 10

/-- The amount of cement used for Tess's street in tons -/
def tessStreetCement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company in tons -/
def totalCement : ℝ := lexiStreetCement + tessStreetCement

/-- Theorem stating that the total cement used is 15.1 tons -/
theorem total_cement_is_15_1 : totalCement = 15.1 := by sorry

end NUMINAMATH_CALUDE_total_cement_is_15_1_l142_14285


namespace NUMINAMATH_CALUDE_two_point_six_million_scientific_notation_l142_14227

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem two_point_six_million_scientific_notation :
  toScientificNotation 2600000 = ScientificNotation.mk 2.6 6 sorry := by
  sorry

end NUMINAMATH_CALUDE_two_point_six_million_scientific_notation_l142_14227


namespace NUMINAMATH_CALUDE_class_group_division_l142_14217

theorem class_group_division (total_students : ℕ) (students_per_group : ℕ) (h1 : total_students = 32) (h2 : students_per_group = 6) :
  total_students / students_per_group = 5 :=
by sorry

end NUMINAMATH_CALUDE_class_group_division_l142_14217


namespace NUMINAMATH_CALUDE_max_x_minus_y_l142_14261

theorem max_x_minus_y (x y z : ℝ) (sum_eq : x + y + z = 2) (prod_eq : x*y + y*z + z*x = 1) :
  ∃ (max : ℝ), max = 2 * Real.sqrt 3 / 3 ∧ ∀ (a b c : ℝ), a + b + c = 2 → a*b + b*c + c*a = 1 → |a - b| ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l142_14261


namespace NUMINAMATH_CALUDE_like_terms_exponents_l142_14216

theorem like_terms_exponents (a b x y : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ 5 * a^(|x|) * b^2 = k * (-0.2 * a^3 * b^(|y|))) → 
  |x| = 3 ∧ |y| = 2 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l142_14216


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l142_14238

/-- Given an arithmetic sequence where the first three terms are 3, 16, and 29,
    prove that the 15th term is 185. -/
theorem arithmetic_sequence_15th_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
    a 1 = 3 →                            -- first term
    a 2 = 16 →                           -- second term
    a 3 = 29 →                           -- third term
    a 15 = 185 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l142_14238


namespace NUMINAMATH_CALUDE_sam_initial_balloons_l142_14242

theorem sam_initial_balloons (S : ℝ) : 
  (S - 5 + 7 = 8) → S = 6 := by
  sorry

end NUMINAMATH_CALUDE_sam_initial_balloons_l142_14242


namespace NUMINAMATH_CALUDE_eight_entrepreneurs_not_attending_l142_14228

/-- The number of entrepreneurs who did not attend either session -/
def entrepreneurs_not_attending (total : ℕ) (digital : ℕ) (ecommerce : ℕ) (both : ℕ) : ℕ :=
  total - (digital + ecommerce - both)

/-- Theorem: Given the specified numbers of entrepreneurs, prove that 8 did not attend either session -/
theorem eight_entrepreneurs_not_attending :
  entrepreneurs_not_attending 40 22 18 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_entrepreneurs_not_attending_l142_14228
