import Mathlib

namespace NUMINAMATH_CALUDE_inequality_system_solution_l1518_151829

theorem inequality_system_solution :
  {x : ℝ | x + 3 ≥ 2 ∧ (3 * x - 1) / 2 < 4} = {x : ℝ | -1 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1518_151829


namespace NUMINAMATH_CALUDE_parabola_through_point_l1518_151843

/-- A parabola is defined by the equation y = ax^2 + bx + c where a ≠ 0 --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- A parabola opens upwards if a > 0 --/
def Parabola.opensUpwards (p : Parabola) : Prop := p.a > 0

/-- A point (x, y) lies on a parabola if it satisfies the parabola's equation --/
def Parabola.containsPoint (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The theorem states that there exists a parabola that opens upwards and passes through (0, -2) --/
theorem parabola_through_point : ∃ p : Parabola, 
  p.opensUpwards ∧ p.containsPoint 0 (-2) ∧ p.a = 1 ∧ p.b = 0 ∧ p.c = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_l1518_151843


namespace NUMINAMATH_CALUDE_angle_between_specific_vectors_l1518_151839

/-- The angle between two 2D vectors -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

/-- Converts degrees to radians -/
def deg_to_rad (deg : ℝ) : ℝ := sorry

theorem angle_between_specific_vectors :
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (-1/2, Real.sqrt 3/2)
  angle_between a b = deg_to_rad 120
  := by sorry

end NUMINAMATH_CALUDE_angle_between_specific_vectors_l1518_151839


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l1518_151827

theorem product_of_four_consecutive_integers (X : ℤ) :
  X * (X + 1) * (X + 2) * (X + 3) = (X^2 + 3*X + 1)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l1518_151827


namespace NUMINAMATH_CALUDE_caden_coin_ratio_l1518_151896

/-- Represents the number of coins of each type -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCounts) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes + 25 * coins.quarters

/-- Represents Caden's coin collection -/
def cadenCoins : CoinCounts where
  pennies := 120
  nickels := 40
  dimes := 8
  quarters := 16

theorem caden_coin_ratio :
  cadenCoins.pennies = 120 ∧
  cadenCoins.pennies = 3 * cadenCoins.nickels ∧
  cadenCoins.quarters = 2 * cadenCoins.dimes ∧
  totalValue cadenCoins = 800 →
  cadenCoins.nickels = 5 * cadenCoins.dimes :=
by sorry

end NUMINAMATH_CALUDE_caden_coin_ratio_l1518_151896


namespace NUMINAMATH_CALUDE_max_contacts_bugs_l1518_151833

/-- Represents the number of bugs on the stick -/
def total_bugs : ℕ := 2016

/-- Theorem stating that the maximum number of contacts between bugs is the square of half the total number of bugs -/
theorem max_contacts_bugs :
  ∃ (contacts : ℕ), contacts = (total_bugs / 2) ^ 2 ∧
  ∀ (a b : ℕ), a + b = total_bugs → a * b ≤ contacts :=
by sorry

end NUMINAMATH_CALUDE_max_contacts_bugs_l1518_151833


namespace NUMINAMATH_CALUDE_total_length_theorem_l1518_151871

/-- Calculates the total length of ladders climbed by two workers in centimeters -/
def total_length_climbed (keaton_ladder_height : ℕ) (keaton_climbs : ℕ) 
  (reece_ladder_diff : ℕ) (reece_climbs : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - reece_ladder_diff
  let keaton_total := keaton_ladder_height * keaton_climbs
  let reece_total := reece_ladder_height * reece_climbs
  (keaton_total + reece_total) * 100

/-- The total length of ladders climbed by both workers is 422000 centimeters -/
theorem total_length_theorem : 
  total_length_climbed 60 40 8 35 = 422000 := by
  sorry

end NUMINAMATH_CALUDE_total_length_theorem_l1518_151871


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1518_151886

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x^2 < 4}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1518_151886


namespace NUMINAMATH_CALUDE_base4_division_theorem_l1518_151823

/-- Convert a number from base 4 to base 10 -/
def base4To10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (4 ^ i)) 0

/-- Convert a number from base 10 to base 4 -/
def base10To4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Division in base 4 -/
def divBase4 (a b : List Nat) : List Nat :=
  base10To4 (base4To10 a / base4To10 b)

theorem base4_division_theorem :
  divBase4 [3, 1, 2, 2] [1, 2] = [2, 0, 1] := by sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l1518_151823


namespace NUMINAMATH_CALUDE_adam_final_spend_l1518_151851

/-- Represents a purchased item with its weight and price per kilogram -/
structure Item where
  weight : Float
  price_per_kg : Float

/-- Calculates the total cost of purchases before discounts -/
def total_cost (items : List Item) : Float :=
  items.foldl (λ acc item => acc + item.weight * item.price_per_kg) 0

/-- Applies the almonds and walnuts discount if eligible -/
def apply_nuts_discount (almonds_cost cashews_cost total : Float) : Float :=
  if almonds_cost + cashews_cost ≥ 2.5 * 10 then
    total - 0.1 * (almonds_cost + cashews_cost)
  else
    total

/-- Applies the overall purchase discount if eligible -/
def apply_overall_discount (total : Float) : Float :=
  if total > 100 then total * 0.95 else total

/-- Theorem stating that Adam's final spend is $69.1 -/
theorem adam_final_spend :
  let items : List Item := [
    { weight := 1.5, price_per_kg := 12 },  -- almonds
    { weight := 1,   price_per_kg := 10 },  -- walnuts
    { weight := 0.5, price_per_kg := 20 },  -- cashews
    { weight := 1,   price_per_kg := 8 },   -- raisins
    { weight := 1.5, price_per_kg := 6 },   -- apricots
    { weight := 0.8, price_per_kg := 15 },  -- pecans
    { weight := 0.7, price_per_kg := 7 }    -- dates
  ]
  let initial_total := total_cost items
  let almonds_cost := 1.5 * 12
  let walnuts_cost := 1 * 10
  let after_nuts_discount := apply_nuts_discount almonds_cost walnuts_cost initial_total
  let final_total := apply_overall_discount after_nuts_discount
  final_total = 69.1 := by
  sorry

end NUMINAMATH_CALUDE_adam_final_spend_l1518_151851


namespace NUMINAMATH_CALUDE_one_billion_scientific_notation_l1518_151850

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- One billion -/
def oneBillion : ℕ := 1000000000

/-- Theorem: The scientific notation of one billion is 1 × 10^9 -/
theorem one_billion_scientific_notation :
  ∃ (sn : ScientificNotation), sn.a = 1 ∧ sn.n = 9 ∧ (sn.a * (10 : ℝ) ^ sn.n = oneBillion) :=
sorry

end NUMINAMATH_CALUDE_one_billion_scientific_notation_l1518_151850


namespace NUMINAMATH_CALUDE_shaded_area_approx_l1518_151897

/-- The area of the shaded region formed by two circles with radii 3 and 6 -/
def shaded_area (π : ℝ) : ℝ :=
  let small_radius : ℝ := 3
  let large_radius : ℝ := 6
  let left_rectangle : ℝ := small_radius * (2 * small_radius)
  let right_rectangle : ℝ := large_radius * (2 * large_radius)
  let small_semicircle : ℝ := 0.5 * π * small_radius ^ 2
  let large_semicircle : ℝ := 0.5 * π * large_radius ^ 2
  (left_rectangle + right_rectangle) - (small_semicircle + large_semicircle)

theorem shaded_area_approx :
  ∃ (π : ℝ), abs (shaded_area π - 19.3) < 0.05 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_approx_l1518_151897


namespace NUMINAMATH_CALUDE_point_with_given_distances_l1518_151852

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ := |p.y|

/-- The distance of a point from the y-axis -/
def distanceFromYAxis (p : Point) : ℝ := |p.x|

/-- Theorem: A point with distance 3 from x-axis and 4 from y-axis has coordinates (4, 3) -/
theorem point_with_given_distances (p : Point) 
  (hx : distanceFromXAxis p = 3)
  (hy : distanceFromYAxis p = 4) :
  p = Point.mk 4 3 := by
  sorry


end NUMINAMATH_CALUDE_point_with_given_distances_l1518_151852


namespace NUMINAMATH_CALUDE_games_sale_value_l1518_151830

def initial_value : ℝ := 200
def value_increase_factor : ℝ := 3
def sold_percentage : ℝ := 0.40

theorem games_sale_value :
  initial_value * value_increase_factor * sold_percentage = 240 := by
  sorry

end NUMINAMATH_CALUDE_games_sale_value_l1518_151830


namespace NUMINAMATH_CALUDE_distance_before_collision_l1518_151848

/-- The distance between two boats one minute before collision -/
theorem distance_before_collision (v1 v2 d : ℝ) (hv1 : v1 = 5) (hv2 : v2 = 21) (hd : d = 20) :
  let total_speed := v1 + v2
  let time_to_collision := d / total_speed
  let distance_per_minute := total_speed / 60
  distance_per_minute = 0.4333 := by sorry

end NUMINAMATH_CALUDE_distance_before_collision_l1518_151848


namespace NUMINAMATH_CALUDE_modulus_of_z_l1518_151808

def i : ℂ := Complex.I

theorem modulus_of_z (z : ℂ) (h : z * (1 + i) = 2 - i) : Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1518_151808


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1518_151845

theorem line_inclination_angle (a : ℝ) (h : a < 0) :
  let line := {(x, y) : ℝ × ℝ | x - a * y + 2 = 0}
  let slope := (1 : ℝ) / a
  let inclination_angle := Real.pi + Real.arctan slope
  ∀ (x y : ℝ), (x, y) ∈ line → inclination_angle ∈ Set.Icc 0 Real.pi ∧
    Real.tan inclination_angle = slope :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1518_151845


namespace NUMINAMATH_CALUDE_three_digit_cube_divisible_by_16_l1518_151866

theorem three_digit_cube_divisible_by_16 :
  ∃! n : ℕ, 100 ≤ 64 * n^3 ∧ 64 * n^3 ≤ 999 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_cube_divisible_by_16_l1518_151866


namespace NUMINAMATH_CALUDE_least_four_digit_square_and_cube_l1518_151885

theorem least_four_digit_square_and_cube : ∃ n : ℕ,
  (1000 ≤ n ∧ n < 10000) ∧  -- four-digit number
  (∃ a : ℕ, n = a^2) ∧      -- perfect square
  (∃ b : ℕ, n = b^3) ∧      -- perfect cube
  (∀ m : ℕ, 
    (1000 ≤ m ∧ m < 10000) ∧ 
    (∃ x : ℕ, m = x^2) ∧ 
    (∃ y : ℕ, m = y^3) → 
    n ≤ m) ∧
  n = 4096 := by
sorry

end NUMINAMATH_CALUDE_least_four_digit_square_and_cube_l1518_151885


namespace NUMINAMATH_CALUDE_smoothie_servings_calculation_l1518_151893

/-- Calculates the number of smoothie servings that can be made given the volumes of ingredients and serving size. -/
def smoothie_servings (watermelon_puree : ℕ) (cream : ℕ) (serving_size : ℕ) : ℕ :=
  (watermelon_puree + cream) / serving_size

/-- Theorem: Given 500 ml of watermelon puree, 100 ml of cream, and a serving size of 150 ml, 4 servings of smoothie can be made. -/
theorem smoothie_servings_calculation :
  smoothie_servings 500 100 150 = 4 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_servings_calculation_l1518_151893


namespace NUMINAMATH_CALUDE_sqrt_sum_product_equals_twenty_l1518_151854

theorem sqrt_sum_product_equals_twenty : (Real.sqrt 8 + Real.sqrt (1/2)) * Real.sqrt 32 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_product_equals_twenty_l1518_151854


namespace NUMINAMATH_CALUDE_remainder_theorem_l1518_151816

theorem remainder_theorem (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1518_151816


namespace NUMINAMATH_CALUDE_robert_reading_theorem_l1518_151865

/-- Calculates the maximum number of complete books that can be read given the reading speed, book length, and available time. -/
def max_complete_books_read (reading_speed : ℕ) (book_length : ℕ) (available_time : ℕ) : ℕ :=
  (available_time * reading_speed) / book_length

/-- Theorem: Given Robert's reading speed of 120 pages per hour, the maximum number of complete 360-page books he can read in 8 hours is 2. -/
theorem robert_reading_theorem : 
  max_complete_books_read 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_theorem_l1518_151865


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1518_151876

/-- Represents a shooter with a given probability of hitting the target -/
structure Shooter where
  hit_prob : ℝ
  hit_prob_nonneg : 0 ≤ hit_prob
  hit_prob_le_one : hit_prob ≤ 1

/-- The probability of both shooters hitting the target -/
def both_hit (a b : Shooter) : ℝ := a.hit_prob * b.hit_prob

/-- The probability of at least one shooter hitting the target -/
def at_least_one_hit (a b : Shooter) : ℝ := 1 - (1 - a.hit_prob) * (1 - b.hit_prob)

theorem shooting_probabilities (a b : Shooter) 
  (ha : a.hit_prob = 0.9) (hb : b.hit_prob = 0.8) : 
  both_hit a b = 0.72 ∧ at_least_one_hit a b = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l1518_151876


namespace NUMINAMATH_CALUDE_total_silver_dollars_l1518_151873

/-- The number of silver dollars owned by Mr. Chiu -/
def chiu_dollars : ℕ := 56

/-- The number of silver dollars owned by Mr. Phung -/
def phung_dollars : ℕ := chiu_dollars + 16

/-- The number of silver dollars owned by Mr. Ha -/
def ha_dollars : ℕ := phung_dollars + 5

/-- The total number of silver dollars owned by all three -/
def total_dollars : ℕ := chiu_dollars + phung_dollars + ha_dollars

theorem total_silver_dollars :
  total_dollars = 205 := by sorry

end NUMINAMATH_CALUDE_total_silver_dollars_l1518_151873


namespace NUMINAMATH_CALUDE_percentage_problem_l1518_151847

theorem percentage_problem (x : ℝ) (hx : x > 0) : 
  x / 100 * 150 - 20 = 10 → x = 20 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l1518_151847


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1518_151894

/-- Given two adjacent points (1,2) and (4,6) on a square, the area of the square is 25 -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let distance_squared := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
  let area := distance_squared
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1518_151894


namespace NUMINAMATH_CALUDE_johnnys_walk_legs_l1518_151863

/-- The number of legs for a given organism type -/
def legs_count (organism : String) : ℕ :=
  match organism with
  | "human" => 2
  | "dog" => 4
  | _ => 0

/-- The total number of legs for a group of organisms -/
def total_legs (humans : ℕ) (dogs : ℕ) : ℕ :=
  humans * legs_count "human" + dogs * legs_count "dog"

/-- Theorem stating that the total number of legs in Johnny's walking group is 12 -/
theorem johnnys_walk_legs :
  let humans : ℕ := 2  -- Johnny and his son
  let dogs : ℕ := 2    -- Johnny's two dogs
  total_legs humans dogs = 12 := by
  sorry

end NUMINAMATH_CALUDE_johnnys_walk_legs_l1518_151863


namespace NUMINAMATH_CALUDE_stone_slab_length_l1518_151814

theorem stone_slab_length (num_slabs : ℕ) (total_area : ℝ) (slab_length : ℝ) :
  num_slabs = 30 →
  total_area = 67.5 →
  num_slabs * (slab_length ^ 2) = total_area →
  slab_length = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_stone_slab_length_l1518_151814


namespace NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l1518_151807

/-- Given a sphere and a right circular cylinder with equal surface areas,
    where the cylinder has height and diameter both equal to 14 cm,
    prove that the radius of the sphere is 7 cm. -/
theorem sphere_cylinder_equal_area (r : ℝ) : 
  r > 0 → -- Ensure the radius is positive
  (4 * Real.pi * r^2 = 2 * Real.pi * 7 * 14) → -- Surface areas are equal
  r = 7 := by
  sorry

#check sphere_cylinder_equal_area

end NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l1518_151807


namespace NUMINAMATH_CALUDE_function_composition_ratio_l1518_151875

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l1518_151875


namespace NUMINAMATH_CALUDE_sum_of_four_squares_of_five_l1518_151857

theorem sum_of_four_squares_of_five : 5^2 + 5^2 + 5^2 + 5^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_squares_of_five_l1518_151857


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1518_151858

theorem inequality_solution_set (c : ℝ) : 
  (c / 5 ≤ 4 + c ∧ 4 + c < -3 * (1 + 2 * c)) ↔ c ∈ Set.Icc (-5) (-1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1518_151858


namespace NUMINAMATH_CALUDE_hyperbola_parameter_sum_l1518_151862

/-- The hyperbola defined by two foci and the difference of distances to these foci. -/
structure Hyperbola where
  f₁ : ℝ × ℝ
  f₂ : ℝ × ℝ
  diff : ℝ

/-- The standard form of a hyperbola equation. -/
structure HyperbolaEquation where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Theorem stating the relationship between the hyperbola's parameters and its equation. -/
theorem hyperbola_parameter_sum (H : Hyperbola) (E : HyperbolaEquation) : 
  H.f₁ = (-3, 1 - Real.sqrt 5 / 4) →
  H.f₂ = (-3, 1 + Real.sqrt 5 / 4) →
  H.diff = 1 →
  E.a > 0 →
  E.b > 0 →
  (∀ (x y : ℝ), (y - E.k)^2 / E.a^2 - (x - E.h)^2 / E.b^2 = 1 ↔ 
    |((x - H.f₁.1)^2 + (y - H.f₁.2)^2).sqrt - ((x - H.f₂.1)^2 + (y - H.f₂.2)^2).sqrt| = H.diff) →
  E.h + E.k + E.a + E.b = -5/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parameter_sum_l1518_151862


namespace NUMINAMATH_CALUDE_factory_output_decrease_l1518_151825

theorem factory_output_decrease (initial_output : ℝ) : 
  let increased_output := initial_output * 1.1
  let holiday_output := increased_output * 1.4
  let required_decrease := (holiday_output - initial_output) / holiday_output * 100
  abs (required_decrease - 35.06) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_factory_output_decrease_l1518_151825


namespace NUMINAMATH_CALUDE_problem_solution_l1518_151867

-- Define the linear function
def linear_function (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

-- Define the quadratic function
def quadratic_function (m n : ℝ) : ℝ → ℝ := λ x ↦ x^2 + m * x + n

theorem problem_solution 
  (k b m n : ℝ) 
  (h1 : k ≠ 0)
  (h2 : linear_function k b (-3) = 0)
  (h3 : linear_function k b 0 = -3)
  (h4 : quadratic_function m n (-3) = 0)
  (h5 : quadratic_function m n 0 = 3)
  (h6 : n > 0)
  (h7 : m ≤ 5) :
  (∃ t : ℝ, 
    (k = -1 ∧ b = -3) ∧ 
    (∃ x y : ℝ, x^2 + m*x + n = -x - 3 ∧ ∀ z : ℝ, z^2 + m*z + n ≥ x^2 + m*x + n) ∧
    (-9/4 < t ∧ t ≤ -1/4 ∧ ∀ z : ℝ, z^2 + m*z + n ≥ t)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1518_151867


namespace NUMINAMATH_CALUDE_greening_investment_equation_l1518_151868

theorem greening_investment_equation (initial_investment : ℝ) (final_investment : ℝ) (x : ℝ) :
  initial_investment = 20000 →
  final_investment = 25000 →
  (initial_investment / 1000) * (1 + x)^2 = (final_investment / 1000) :=
by
  sorry

end NUMINAMATH_CALUDE_greening_investment_equation_l1518_151868


namespace NUMINAMATH_CALUDE_shape_cutting_theorem_l1518_151817

/-- Represents a cell in the shape --/
inductive Cell
| Black
| Gray

/-- Represents the shape as a list of cells --/
def Shape := List Cell

/-- A function to count the number of ways to cut the shape --/
def count_cuts (shape : Shape) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem shape_cutting_theorem (shape : Shape) :
  shape.length = 17 →
  count_cuts shape = 10 :=
sorry

end NUMINAMATH_CALUDE_shape_cutting_theorem_l1518_151817


namespace NUMINAMATH_CALUDE_simplify_expression_find_a_value_independence_condition_l1518_151831

-- Define A and B as functions of a and b
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b - 1

-- Theorem 1: Simplification of 4A - (3A - 2B)
theorem simplify_expression (a b : ℝ) :
  4 * A a b - (3 * A a b - 2 * B a b) = 5 * a * b - 2 * a - 3 := by sorry

-- Theorem 2: Value of a when b = 1 and 4A - (3A - 2B) = b - 2a
theorem find_a_value (a : ℝ) :
  (4 * A a 1 - (3 * A a 1 - 2 * B a 1) = 1 - 2 * a) → a = 4/5 := by sorry

-- Theorem 3: A + 2B is independent of a iff b = 2/5
theorem independence_condition (b : ℝ) :
  (∀ a₁ a₂ : ℝ, A a₁ b + 2 * B a₁ b = A a₂ b + 2 * B a₂ b) ↔ b = 2/5 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_find_a_value_independence_condition_l1518_151831


namespace NUMINAMATH_CALUDE_multiply_special_polynomials_l1518_151809

theorem multiply_special_polynomials (y : ℝ) :
  (y^4 + 30*y^2 + 900) * (y^2 - 30) = y^6 - 27000 := by
  sorry

end NUMINAMATH_CALUDE_multiply_special_polynomials_l1518_151809


namespace NUMINAMATH_CALUDE_both_are_dwarves_l1518_151846

-- Define the types of inhabitants
inductive Inhabitant : Type
| Elf : Inhabitant
| Dwarf : Inhabitant

-- Define the types of statements
inductive Statement : Type
| AboutGold : Statement
| AboutDwarf : Statement
| Other : Statement

-- Define the truth value of a statement given the speaker and the statement type
def isTruthful (speaker : Inhabitant) (statement : Statement) : Prop :=
  match speaker, statement with
  | Inhabitant.Dwarf, Statement.AboutGold => False
  | Inhabitant.Elf, Statement.AboutDwarf => False
  | _, _ => True

-- Define A's statement
def A_statement : Statement := Statement.AboutGold

-- Define B's statement about A
def B_statement (A_type : Inhabitant) : Statement :=
  match A_type with
  | Inhabitant.Dwarf => Statement.Other
  | Inhabitant.Elf => Statement.AboutDwarf

-- Theorem to prove
theorem both_are_dwarves :
  ∃ (A_type B_type : Inhabitant),
    A_type = Inhabitant.Dwarf ∧
    B_type = Inhabitant.Dwarf ∧
    isTruthful A_type A_statement = False ∧
    isTruthful B_type (B_statement A_type) = True :=
  sorry


end NUMINAMATH_CALUDE_both_are_dwarves_l1518_151846


namespace NUMINAMATH_CALUDE_prob_two_green_apples_l1518_151888

/-- The probability of selecting two green apples from a set of 8 apples,
    where 4 are green, when choosing 2 apples at random. -/
theorem prob_two_green_apples (total : ℕ) (green : ℕ) (choose : ℕ) 
    (h_total : total = 8) 
    (h_green : green = 4) 
    (h_choose : choose = 2) : 
    Nat.choose green choose / Nat.choose total choose = 3 / 14 := by
  sorry

#check prob_two_green_apples

end NUMINAMATH_CALUDE_prob_two_green_apples_l1518_151888


namespace NUMINAMATH_CALUDE_tourist_group_size_proof_l1518_151828

/-- Represents the number of people a large room can accommodate -/
def large_room_capacity : ℕ := 3

/-- Represents the number of large rooms rented -/
def large_rooms_rented : ℕ := 8

/-- Represents the total number of people in the tourist group -/
def tourist_group_size : ℕ := large_rooms_rented * large_room_capacity

theorem tourist_group_size_proof :
  (∀ n : ℕ, n ≠ tourist_group_size → 
    (∃ m k : ℕ, n = 3 * m + 2 * k ∧ m + k < large_rooms_rented) ∨
    (∃ m k : ℕ, n = 3 * m + 2 * k ∧ m > large_rooms_rented)) →
  tourist_group_size = 24 := by sorry

end NUMINAMATH_CALUDE_tourist_group_size_proof_l1518_151828


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1518_151861

theorem max_value_of_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_condition : x + y + z = 1) :
  x + y^2 + z^3 ≤ 1 ∧ ∃ (x' y' z' : ℝ), x' + y'^2 + z'^3 = 1 ∧ 
    x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1518_151861


namespace NUMINAMATH_CALUDE_days_without_calls_l1518_151891

-- Define the number of days in the year
def total_days : ℕ := 365

-- Define the periods of the calls
def period1 : ℕ := 3
def period2 : ℕ := 4
def period3 : ℕ := 5

-- Function to calculate the number of days with at least one call
def days_with_calls : ℕ :=
  (total_days / period1) +
  (total_days / period2) +
  (total_days / period3) -
  (total_days / (period1 * period2)) -
  (total_days / (period2 * period3)) -
  (total_days / (period1 * period3)) +
  (total_days / (period1 * period2 * period3))

-- Theorem to prove
theorem days_without_calls :
  total_days - days_with_calls = 146 :=
by sorry

end NUMINAMATH_CALUDE_days_without_calls_l1518_151891


namespace NUMINAMATH_CALUDE_lunch_total_is_fifteen_l1518_151810

/-- The total amount spent on lunch given the conditions -/
def total_lunch_amount (friend_spent : ℕ) (difference : ℕ) : ℕ :=
  friend_spent + (friend_spent - difference)

/-- Theorem: The total amount spent on lunch is $15 -/
theorem lunch_total_is_fifteen :
  total_lunch_amount 10 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_lunch_total_is_fifteen_l1518_151810


namespace NUMINAMATH_CALUDE_carpet_cost_calculation_l1518_151849

/-- Calculate the cost of a carpet given its dimensions and the price per square meter -/
def calculate_carpet_cost (length width price_per_sqm : ℝ) : ℝ :=
  length * width * price_per_sqm

/-- The problem statement -/
theorem carpet_cost_calculation :
  let first_carpet_breadth : ℝ := 6
  let first_carpet_length : ℝ := 1.44 * first_carpet_breadth
  let second_carpet_length : ℝ := first_carpet_length * 1.427
  let second_carpet_breadth : ℝ := first_carpet_breadth * 1.275
  let price_per_sqm : ℝ := 46.35
  
  abs (calculate_carpet_cost second_carpet_length second_carpet_breadth price_per_sqm - 4371.78) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_carpet_cost_calculation_l1518_151849


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l1518_151853

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y = 0

-- Define the symmetric circle C'
def circle_C' (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

-- Theorem statement
theorem symmetric_circle_equation :
  ∀ (x y : ℝ), circle_C' x y ↔ 
  (∃ (x₀ y₀ : ℝ), circle_C x₀ y₀ ∧ 
   (x + x₀ = y + y₀) ∧ -- Midpoint condition
   ((y - y₀) = (x - x₀))) -- Perpendicular condition
  := by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l1518_151853


namespace NUMINAMATH_CALUDE_fruit_selling_theorem_l1518_151832

/-- Represents the fruit selling scenario -/
structure FruitSelling where
  cost_price : ℝ
  base_price : ℝ
  base_volume : ℝ
  price_sensitivity : ℝ

/-- Calculates the sales volume for a given selling price -/
def sales_volume (fs : FruitSelling) (selling_price : ℝ) : ℝ :=
  fs.base_volume - fs.price_sensitivity * (selling_price - fs.base_price)

/-- Calculates the profit for a given selling price -/
def profit (fs : FruitSelling) (selling_price : ℝ) : ℝ :=
  (selling_price - fs.cost_price) * sales_volume fs selling_price

/-- Main theorem about the fruit selling scenario -/
theorem fruit_selling_theorem (fs : FruitSelling) 
  (h1 : fs.cost_price = 30)
  (h2 : fs.base_price = 40)
  (h3 : fs.base_volume = 400)
  (h4 : fs.price_sensitivity = 10) :
  (sales_volume fs 45 = 350) ∧ 
  ((profit fs 45 = 5250 ∨ profit fs 65 = 5250) ∧ 
   ∀ x, profit fs x ≤ 5250 ∨ x = 45 ∨ x = 65) ∧
  (∃ max_profit : ℝ, max_profit = 6250 ∧ 
   ∀ x, profit fs x ≤ max_profit ∧ 
   (profit fs x = max_profit → x = 55)) := by
  sorry

end NUMINAMATH_CALUDE_fruit_selling_theorem_l1518_151832


namespace NUMINAMATH_CALUDE_simplify_fraction_l1518_151881

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1518_151881


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1518_151856

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 5 = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1518_151856


namespace NUMINAMATH_CALUDE_salary_after_raise_l1518_151872

theorem salary_after_raise (original_salary : ℝ) (percentage_increase : ℝ) (new_salary : ℝ) :
  original_salary = 60 →
  percentage_increase = 83.33333333333334 →
  new_salary = original_salary * (1 + percentage_increase / 100) →
  new_salary = 110 := by
  sorry

end NUMINAMATH_CALUDE_salary_after_raise_l1518_151872


namespace NUMINAMATH_CALUDE_orange_ribbons_l1518_151826

theorem orange_ribbons (total : ℕ) (yellow purple orange silver : ℕ) : 
  yellow + purple + orange + silver = total →
  4 * yellow = total →
  3 * purple = total →
  6 * orange = total →
  silver = 40 →
  orange = 27 := by
sorry

end NUMINAMATH_CALUDE_orange_ribbons_l1518_151826


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1518_151895

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {2, 3, 4}

-- Define set B
def B : Set Nat := {1, 2}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1518_151895


namespace NUMINAMATH_CALUDE_sticker_difference_l1518_151805

/-- The number of stickers each person has -/
structure StickerCount where
  jerry : ℕ
  george : ℕ
  fred : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : StickerCount) : Prop :=
  s.jerry = 3 * s.george ∧
  s.george < s.fred ∧
  s.fred = 18 ∧
  s.jerry = 36

/-- The theorem to prove -/
theorem sticker_difference (s : StickerCount) 
  (h : problem_conditions s) : s.fred - s.george = 6 := by
  sorry

end NUMINAMATH_CALUDE_sticker_difference_l1518_151805


namespace NUMINAMATH_CALUDE_xyz_value_l1518_151802

theorem xyz_value (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xy : x * y = 40 * Real.rpow 4 (1/3))
  (h_xz : x * z = 56 * Real.rpow 4 (1/3))
  (h_yz : y * z = 32 * Real.rpow 4 (1/3))
  (h_sum : x + y = 18) :
  x * y * z = 16 * Real.sqrt 895 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1518_151802


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1518_151890

def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b : ℝ × ℝ := (-3, 4)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (vector_a x) vector_b → x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1518_151890


namespace NUMINAMATH_CALUDE_earth_land_area_scientific_notation_l1518_151804

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a given number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

theorem earth_land_area_scientific_notation :
  let earthLandArea : ℝ := 149000000
  let scientificNotation := toScientificNotation earthLandArea 3
  scientificNotation.coefficient = 1.49 ∧ scientificNotation.exponent = 8 := by
  sorry

end NUMINAMATH_CALUDE_earth_land_area_scientific_notation_l1518_151804


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_no_ten_consecutive_sum_2016_seven_consecutive_sum_2016_l1518_151806

theorem consecutive_numbers_sum (n : ℕ) (sum : ℕ) : Prop :=
  ∃ a : ℕ, (n * a + n * (n - 1) / 2 = sum)

theorem no_ten_consecutive_sum_2016 : ¬ consecutive_numbers_sum 10 2016 := by
  sorry

theorem seven_consecutive_sum_2016 : consecutive_numbers_sum 7 2016 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_no_ten_consecutive_sum_2016_seven_consecutive_sum_2016_l1518_151806


namespace NUMINAMATH_CALUDE_tangent_midpoint_parallel_l1518_151803

-- Define the ellipses C and T
def ellipse_C (x y : ℝ) : Prop := x^2/18 + y^2/2 = 1
def ellipse_T (x y : ℝ) : Prop := x^2/9 + y^2 = 1

-- Define a point on an ellipse
def point_on_ellipse (E : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  E P.1 P.2

-- Define a tangent line from a point to an ellipse
def is_tangent (P M : ℝ × ℝ) (E : ℝ → ℝ → Prop) : Prop :=
  point_on_ellipse E M ∧ 
  ∀ Q, point_on_ellipse E Q → (Q ≠ M → (Q.2 - P.2) * (M.1 - P.1) ≠ (Q.1 - P.1) * (M.2 - P.2))

-- Define parallel lines
def parallel (P₁ P₂ Q₁ Q₂ : ℝ × ℝ) : Prop :=
  (P₂.2 - P₁.2) * (Q₂.1 - Q₁.1) = (P₂.1 - P₁.1) * (Q₂.2 - Q₁.2)

theorem tangent_midpoint_parallel :
  ∀ P G H M N : ℝ × ℝ,
    point_on_ellipse ellipse_C P →
    point_on_ellipse ellipse_C G →
    point_on_ellipse ellipse_C H →
    is_tangent P M ellipse_T →
    is_tangent P N ellipse_T →
    G ≠ P →
    H ≠ P →
    (G.2 - P.2) * (M.1 - P.1) = (G.1 - P.1) * (M.2 - P.2) →
    (H.2 - P.2) * (N.1 - P.1) = (H.1 - P.1) * (N.2 - P.2) →
    parallel M N G H :=
by sorry

end NUMINAMATH_CALUDE_tangent_midpoint_parallel_l1518_151803


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l1518_151855

/-- An ellipse E with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The area of the quadrilateral formed by the vertices of an ellipse -/
def quadrilateral_area (E : Ellipse) : ℝ := 2 * E.a * E.b

theorem ellipse_equation_from_conditions (E : Ellipse) 
  (h_vertex : ellipse_equation E 0 (-2))
  (h_area : quadrilateral_area E = 4 * Real.sqrt 5) :
  ∀ x y, ellipse_equation E x y ↔ x^2 / 5 + y^2 / 4 = 1 := by
  sorry

#check ellipse_equation_from_conditions

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l1518_151855


namespace NUMINAMATH_CALUDE_upper_limit_of_set_D_l1518_151820

def is_prime (n : ℕ) : Prop := sorry

def set_D (upper_bound : ℕ) : Set ℕ :=
  {n : ℕ | 10 < n ∧ n ≤ upper_bound ∧ is_prime n}

theorem upper_limit_of_set_D (upper_bound : ℕ) :
  (∃ (a b : ℕ), a ∈ set_D upper_bound ∧ b ∈ set_D upper_bound ∧ b - a = 12) →
  (∃ (max : ℕ), max ∈ set_D upper_bound ∧ ∀ (x : ℕ), x ∈ set_D upper_bound → x ≤ max) →
  (∃ (max : ℕ), max ∈ set_D upper_bound ∧ ∀ (x : ℕ), x ∈ set_D upper_bound → x ≤ max ∧ max = 23) :=
by sorry

end NUMINAMATH_CALUDE_upper_limit_of_set_D_l1518_151820


namespace NUMINAMATH_CALUDE_smallest_n_for_sock_arrangement_l1518_151837

theorem smallest_n_for_sock_arrangement : 
  (∃ n : ℕ, n > 0 ∧ (n + 1) * (n + 2) / 2 > 1000000 ∧ 
   ∀ m : ℕ, m > 0 → (m + 1) * (m + 2) / 2 > 1000000 → m ≥ n) →
  (∃ n : ℕ, n > 0 ∧ (n + 1) * (n + 2) / 2 > 1000000 ∧ 
   ∀ m : ℕ, m > 0 → (m + 1) * (m + 2) / 2 > 1000000 → m ≥ n ∧ n = 1413) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_sock_arrangement_l1518_151837


namespace NUMINAMATH_CALUDE_fraction_inequality_l1518_151883

theorem fraction_inequality (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 3 →
  (5 * x + 3 > 9 - 3 * x) ↔ (x ∈ Set.Ioo (3/4 : ℝ) 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1518_151883


namespace NUMINAMATH_CALUDE_unbroken_seashells_l1518_151860

/-- Given that Tom found 7 seashells in total and 4 of them were broken,
    prove that the number of unbroken seashells is 3. -/
theorem unbroken_seashells (total : ℕ) (broken : ℕ) 
  (h1 : total = 7) 
  (h2 : broken = 4) : 
  total - broken = 3 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l1518_151860


namespace NUMINAMATH_CALUDE_zero_sum_and_product_implies_all_zero_l1518_151842

theorem zero_sum_and_product_implies_all_zero (a b c d : ℝ) 
  (sum_zero : a + b + c + d = 0)
  (product_zero : a*b + c*d + a*c + b*c + a*d + b*d = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_sum_and_product_implies_all_zero_l1518_151842


namespace NUMINAMATH_CALUDE_negation_existence_divisibility_l1518_151878

theorem negation_existence_divisibility :
  (¬ ∃ n : ℕ+, 10 ∣ (n^2 + 3*n)) ↔ (∀ n : ℕ+, ¬(10 ∣ (n^2 + 3*n))) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_divisibility_l1518_151878


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1518_151879

theorem fraction_equals_zero (x : ℝ) : 
  (x - 5) / (4 * x^2 - 1) = 0 ↔ x = 5 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1518_151879


namespace NUMINAMATH_CALUDE_part_one_disproof_part_two_proof_l1518_151859

-- Part 1
theorem part_one_disproof (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z ≥ 3) :
  ¬ (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z ≥ 3 → 1/x + 1/y + 1/z ≤ 3) :=
sorry

-- Part 2
theorem part_two_proof (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z ≤ 3) :
  1/x + 1/y + 1/z ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_part_one_disproof_part_two_proof_l1518_151859


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l1518_151818

-- Define the number of red and yellow balls
def num_red_balls : ℕ := 3
def num_yellow_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_yellow_balls

-- Define the probability of selecting a yellow ball
def prob_yellow : ℚ := num_yellow_balls / total_balls

-- Theorem statement
theorem yellow_ball_probability : prob_yellow = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l1518_151818


namespace NUMINAMATH_CALUDE_product_of_sums_equals_power_specific_product_equals_power_l1518_151836

theorem product_of_sums_equals_power (a b : ℕ) :
  (a + b) * (a^2 + b^2) * (a^4 + b^4) * (a^8 + b^8) * 
  (a^16 + b^16) * (a^32 + b^32) * (a^64 + b^64) = (a + b)^127 :=
by
  sorry

theorem specific_product_equals_power :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * 
  (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 9^127 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_power_specific_product_equals_power_l1518_151836


namespace NUMINAMATH_CALUDE_game_a_higher_prob_l1518_151874

def prob_heads : ℚ := 3/4
def prob_tails : ℚ := 1/4

def game_a_win_prob : ℚ := prob_heads^4 + prob_tails^4

def game_b_win_prob : ℚ := prob_heads^3 * prob_tails^2 + prob_tails^3 * prob_heads^2

theorem game_a_higher_prob : game_a_win_prob = game_b_win_prob + 1/4 := by
  sorry

end NUMINAMATH_CALUDE_game_a_higher_prob_l1518_151874


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_line_l1518_151801

/-- Given a line 3x - 4y + 12 = 0, the circle with diameter as the line segment
    enclosed between the two coordinate axes by this line has the equation
    x² + 4x + y² - 3y = 0 -/
theorem circle_equation_from_diameter_line :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), 3 * x₀ - 4 * y₀ + 12 = 0 ∧ 
                   x₀ * y₀ = 0 ∧
                   (x - (-2))^2 + (y - (3/2))^2 = (5/2)^2) →
  x^2 + 4*x + y^2 - 3*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_line_l1518_151801


namespace NUMINAMATH_CALUDE_clara_stickers_l1518_151815

theorem clara_stickers (initial : ℕ) : 
  initial ≥ 10 →
  (initial - 10) % 2 = 0 →
  (initial - 10) / 2 - 45 = 45 →
  initial = 100 := by
sorry

end NUMINAMATH_CALUDE_clara_stickers_l1518_151815


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1518_151800

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1518_151800


namespace NUMINAMATH_CALUDE_three_and_negative_three_are_opposite_l1518_151880

-- Define opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Theorem statement
theorem three_and_negative_three_are_opposite :
  are_opposite 3 (-3) :=
sorry

end NUMINAMATH_CALUDE_three_and_negative_three_are_opposite_l1518_151880


namespace NUMINAMATH_CALUDE_a_explicit_formula_l1518_151819

def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 + 4 * a n + (1 + 24 * a n).sqrt) / 16

theorem a_explicit_formula : ∀ n : ℕ, 
  a n = (1 / 3) * (1 + 1 / 2^n) * (1 + 1 / 2^(n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_a_explicit_formula_l1518_151819


namespace NUMINAMATH_CALUDE_number_from_percentages_l1518_151899

theorem number_from_percentages (x : ℝ) : 
  0.15 * 0.30 * 0.50 * x = 126 → x = 5600 := by
  sorry

end NUMINAMATH_CALUDE_number_from_percentages_l1518_151899


namespace NUMINAMATH_CALUDE_infinitely_many_not_sum_of_seven_sixth_powers_l1518_151824

theorem infinitely_many_not_sum_of_seven_sixth_powers :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ 
  (∀ a ∈ S, ∀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ, 
   a ≠ a₁^6 + a₂^6 + a₃^6 + a₄^6 + a₅^6 + a₆^6 + a₇^6) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_not_sum_of_seven_sixth_powers_l1518_151824


namespace NUMINAMATH_CALUDE_probability_three_non_defective_pencils_l1518_151892

theorem probability_three_non_defective_pencils :
  let total_pencils : ℕ := 10
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations : ℕ := Nat.choose total_pencils selected_pencils
  let non_defective_combinations : ℕ := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 7 / 15 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_non_defective_pencils_l1518_151892


namespace NUMINAMATH_CALUDE_kangaroo_fraction_sum_l1518_151811

theorem kangaroo_fraction_sum (total : ℕ) (grey pink : ℕ) : 
  total = grey + pink ∧ 
  grey > 0 ∧ 
  pink > 0 ∧
  total = 2016 →
  (grey : ℝ) * (pink / grey) + (pink : ℝ) * (grey / pink) = total := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_fraction_sum_l1518_151811


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l1518_151877

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), n = 998 ∧ 
  n < 1000 ∧ 
  n > 99 ∧
  70 * n % 350 = 210 % 350 ∧
  ∀ (m : ℕ), m < 1000 → m > 99 → 70 * m % 350 = 210 % 350 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l1518_151877


namespace NUMINAMATH_CALUDE_optimal_timing_problem_l1518_151835

/-- Represents the optimal timing problem for three people traveling between two points. -/
theorem optimal_timing_problem (distance : ℝ) (walking_speed : ℝ) (bicycle_speed : ℝ) 
  (h_distance : distance = 15)
  (h_walking_speed : walking_speed = 6)
  (h_bicycle_speed : bicycle_speed = 15) :
  ∃ (optimal_time : ℝ),
    optimal_time = 3 / 11 ∧
    (∀ (t : ℝ), 
      let time_A := distance / walking_speed + (distance - walking_speed * t) / bicycle_speed
      let time_B := t + (distance - bicycle_speed * t) / walking_speed
      let time_C := distance / walking_speed - t
      (time_A = time_B ∧ time_B = time_C) → t = optimal_time) :=
by sorry

end NUMINAMATH_CALUDE_optimal_timing_problem_l1518_151835


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l1518_151813

theorem largest_constant_inequality (D : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 4 ≥ D * (x + y)) ↔ D ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l1518_151813


namespace NUMINAMATH_CALUDE_problem_solution_l1518_151889

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (h1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 5)
  (h2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 20)
  (h3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 145) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 380 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1518_151889


namespace NUMINAMATH_CALUDE_farm_sections_l1518_151840

theorem farm_sections (section_area : ℝ) (total_area : ℝ) (h1 : section_area = 60) (h2 : total_area = 300) :
  total_area / section_area = 5 := by
  sorry

end NUMINAMATH_CALUDE_farm_sections_l1518_151840


namespace NUMINAMATH_CALUDE_complex_inequality_l1518_151841

theorem complex_inequality (a b c : ℂ) (h : a * Complex.abs (b * c) + b * Complex.abs (c * a) + c * Complex.abs (a * b) = 0) :
  Complex.abs ((a - b) * (b - c) * (c - a)) ≥ 3 * Real.sqrt 3 * Complex.abs (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l1518_151841


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1518_151898

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 - (k - 4)*x - k + 7 > 0) ↔ (k > 4 ∧ k < 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1518_151898


namespace NUMINAMATH_CALUDE_exam_logic_l1518_151870

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (got_all_right : Student → Prop)
variable (received_A : Student → Prop)

-- State the theorem
theorem exam_logic (s : Student) 
  (h : ∀ x, got_all_right x → received_A x) :
  ¬(received_A s) → ¬(got_all_right s) := by
sorry

end NUMINAMATH_CALUDE_exam_logic_l1518_151870


namespace NUMINAMATH_CALUDE_good_pair_exists_l1518_151884

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ 
  ∃ a b : ℕ, m * n = a ^ 2 ∧ (m + 1) * (n + 1) = b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_good_pair_exists_l1518_151884


namespace NUMINAMATH_CALUDE_y_increases_as_x_decreases_l1518_151887

theorem y_increases_as_x_decreases (α : Real) (h_acute : 0 < α ∧ α < π / 2) :
  let f : Real → Real := λ x ↦ (Real.sin α - 1) * x - 6
  ∀ x₁ x₂ : Real, x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_y_increases_as_x_decreases_l1518_151887


namespace NUMINAMATH_CALUDE_f_properties_l1518_151834

/-- The function f(m, n) represents the absolute difference between 
    the areas of black and white parts in a right triangle with legs m and n. -/
def f (m n : ℕ+) : ℝ :=
  sorry

theorem f_properties :
  (∀ m n : ℕ+, Even m.val → Even n.val → f m n = 0) ∧
  (∀ m n : ℕ+, Odd m.val → Odd n.val → f m n = 1/2) ∧
  (∀ m n : ℕ+, f m n ≤ (1/2 : ℝ) * max m.val n.val) ∧
  (∀ c : ℝ, ∃ m n : ℕ+, f m n ≥ c) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1518_151834


namespace NUMINAMATH_CALUDE_unique_valid_arrangement_l1518_151838

/-- Represents the positions in the hexagon --/
inductive Position
| A | B | C | D | E | F

/-- Represents a line in the hexagon --/
structure Line where
  p1 : Position
  p2 : Position
  p3 : Position

/-- The arrangement of digits in the hexagon --/
def Arrangement := Position → Fin 6

/-- The 7 lines in the hexagon --/
def lines : List Line := [
  ⟨Position.A, Position.B, Position.C⟩,
  ⟨Position.A, Position.D, Position.F⟩,
  ⟨Position.A, Position.E, Position.F⟩,
  ⟨Position.B, Position.C, Position.D⟩,
  ⟨Position.B, Position.E, Position.D⟩,
  ⟨Position.C, Position.E, Position.F⟩,
  ⟨Position.D, Position.E, Position.F⟩
]

/-- Check if an arrangement is valid --/
def isValidArrangement (arr : Arrangement) : Prop :=
  (∀ p : Position, arr p ∈ [1, 2, 3, 4, 5, 6]) ∧
  (∀ p q : Position, p ≠ q → arr p ≠ arr q) ∧
  (∀ l : Line, (arr l.p1).val + (arr l.p2).val + (arr l.p3).val = 15)

/-- The unique valid arrangement --/
def uniqueArrangement : Arrangement :=
  fun p => match p with
  | Position.A => 4
  | Position.B => 1
  | Position.C => 2
  | Position.D => 5
  | Position.E => 6
  | Position.F => 3

theorem unique_valid_arrangement :
  isValidArrangement uniqueArrangement ∧
  (∀ arr : Arrangement, isValidArrangement arr → arr = uniqueArrangement) := by
  sorry


end NUMINAMATH_CALUDE_unique_valid_arrangement_l1518_151838


namespace NUMINAMATH_CALUDE_monotone_increasing_ln_plus_ax_l1518_151844

open Real

theorem monotone_increasing_ln_plus_ax (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, Monotone (λ x => Real.log x + a * x)) →
  a ≥ -1/2 := by
sorry

end NUMINAMATH_CALUDE_monotone_increasing_ln_plus_ax_l1518_151844


namespace NUMINAMATH_CALUDE_lemonade_recipe_l1518_151821

/-- Lemonade recipe problem -/
theorem lemonade_recipe (lemon_juice sugar water : ℚ) : 
  water = 3 * sugar →  -- Water is 3 times sugar
  sugar = 3 * lemon_juice →  -- Sugar is 3 times lemon juice
  lemon_juice = 4 →  -- Luka uses 4 cups of lemon juice
  water = 36 := by  -- The amount of water needed is 36 cups
sorry


end NUMINAMATH_CALUDE_lemonade_recipe_l1518_151821


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l1518_151812

theorem real_part_of_complex_product : 
  let z : ℂ := (1 + 2*Complex.I) * (3 - Complex.I)
  Complex.re z = 5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l1518_151812


namespace NUMINAMATH_CALUDE_any_proof_to_contradiction_l1518_151822

theorem any_proof_to_contradiction (P : Prop) : P → ∃ (proof : ¬P → False), P :=
  sorry

end NUMINAMATH_CALUDE_any_proof_to_contradiction_l1518_151822


namespace NUMINAMATH_CALUDE_range_of_a_l1518_151882

-- Define the sets A and B
def A : Set ℝ := {0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1518_151882


namespace NUMINAMATH_CALUDE_april_cookie_spending_l1518_151869

/-- Calculates the total amount spent on cookies in April given the following conditions:
  * April has 30 days
  * On even days, 3 chocolate chip cookies and 2 sugar cookies are bought
  * On odd days, 4 oatmeal cookies and 1 snickerdoodle cookie are bought
  * Prices: chocolate chip $18, sugar $22, oatmeal $15, snickerdoodle $25
-/
theorem april_cookie_spending : 
  let days_in_april : ℕ := 30
  let even_days : ℕ := days_in_april / 2
  let odd_days : ℕ := days_in_april / 2
  let choc_chip_price : ℕ := 18
  let sugar_price : ℕ := 22
  let oatmeal_price : ℕ := 15
  let snickerdoodle_price : ℕ := 25
  let even_day_cost : ℕ := 3 * choc_chip_price + 2 * sugar_price
  let odd_day_cost : ℕ := 4 * oatmeal_price + 1 * snickerdoodle_price
  let total_cost : ℕ := even_days * even_day_cost + odd_days * odd_day_cost
  total_cost = 2745 := by
sorry


end NUMINAMATH_CALUDE_april_cookie_spending_l1518_151869


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1518_151864

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (length width : ℝ),
    r = 3 →
    length / width = 3 →
    2 * r = width →
    length * width = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1518_151864
