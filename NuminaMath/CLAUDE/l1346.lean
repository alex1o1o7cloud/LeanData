import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_three_numbers_with_same_product_last_digit_l1346_134660

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem existence_of_three_numbers_with_same_product_last_digit :
  ∃ (a b c : ℕ), 
    (lastDigit a ≠ lastDigit b) ∧ 
    (lastDigit b ≠ lastDigit c) ∧ 
    (lastDigit a ≠ lastDigit c) ∧
    (lastDigit (a * b) = lastDigit (b * c)) ∧
    (lastDigit (b * c) = lastDigit (a * c)) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_existence_of_three_numbers_with_same_product_last_digit_l1346_134660


namespace NUMINAMATH_CALUDE_total_faces_painted_l1346_134655

/-- The number of cuboids Amelia painted -/
def num_cuboids : ℕ := 6

/-- The number of faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- Theorem stating the total number of faces painted by Amelia -/
theorem total_faces_painted : num_cuboids * faces_per_cuboid = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_faces_painted_l1346_134655


namespace NUMINAMATH_CALUDE_box_height_is_15_inches_l1346_134687

/-- Proves that the height of a box is 15 inches given the specified conditions --/
theorem box_height_is_15_inches 
  (base_length : ℝ) 
  (base_width : ℝ) 
  (total_volume : ℝ) 
  (cost_per_box : ℝ) 
  (total_cost : ℝ) 
  (h : ℝ)
  (base_length_eq : base_length = 20)
  (base_width_eq : base_width = 20)
  (total_volume_eq : total_volume = 3060000)
  (cost_per_box_eq : cost_per_box = 1.3)
  (total_cost_eq : total_cost = 663)
  (volume_equation : total_volume = base_length * base_width * h * (total_cost / cost_per_box))
  : h = 15 := by
  sorry

end NUMINAMATH_CALUDE_box_height_is_15_inches_l1346_134687


namespace NUMINAMATH_CALUDE_fraction_inequality_l1346_134629

theorem fraction_inequality (x : ℝ) : (x - 1) / (x + 2) ≥ 0 ↔ x < -2 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1346_134629


namespace NUMINAMATH_CALUDE_sqrt_54_div_sqrt_9_eq_sqrt_6_l1346_134621

theorem sqrt_54_div_sqrt_9_eq_sqrt_6 : Real.sqrt 54 / Real.sqrt 9 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_54_div_sqrt_9_eq_sqrt_6_l1346_134621


namespace NUMINAMATH_CALUDE_wall_space_to_paint_is_560_l1346_134644

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents a rectangular feature on a wall (e.g., door or window) -/
structure WallFeature where
  width : ℝ
  height : ℝ

/-- Calculates the total area of wall space to paint in a room -/
def wallSpaceToPaint (room : RoomDimensions) (doorway1 : WallFeature) (window : WallFeature) (doorway2 : WallFeature) : ℝ :=
  let totalWallArea := 2 * (room.width * room.height + room.length * room.height)
  let featureArea := doorway1.width * doorway1.height + window.width * window.height + doorway2.width * doorway2.height
  totalWallArea - featureArea

/-- The main theorem stating that the wall space to paint is 560 square feet -/
theorem wall_space_to_paint_is_560 (room : RoomDimensions) (doorway1 : WallFeature) (window : WallFeature) (doorway2 : WallFeature) :
  room.width = 20 ∧ room.length = 20 ∧ room.height = 8 ∧
  doorway1.width = 3 ∧ doorway1.height = 7 ∧
  window.width = 6 ∧ window.height = 4 ∧
  doorway2.width = 5 ∧ doorway2.height = 7 →
  wallSpaceToPaint room doorway1 window doorway2 = 560 :=
by
  sorry

end NUMINAMATH_CALUDE_wall_space_to_paint_is_560_l1346_134644


namespace NUMINAMATH_CALUDE_max_profit_at_11_l1346_134646

/-- The cost price of each item in yuan -/
def cost_price : ℝ := 8

/-- The initial selling price in yuan -/
def initial_price : ℝ := 9

/-- The initial daily sales volume at the initial price -/
def initial_volume : ℝ := 20

/-- The rate at which sales volume decreases per yuan increase in price -/
def volume_decrease_rate : ℝ := 4

/-- The daily sales volume as a function of the selling price -/
def sales_volume (price : ℝ) : ℝ :=
  initial_volume - volume_decrease_rate * (price - initial_price)

/-- The daily profit as a function of the selling price -/
def daily_profit (price : ℝ) : ℝ :=
  sales_volume price * (price - cost_price)

/-- The theorem stating that the daily profit is maximized at 11 yuan -/
theorem max_profit_at_11 :
  ∃ (max_price : ℝ), max_price = 11 ∧
  ∀ (price : ℝ), price ≥ initial_price →
  daily_profit price ≤ daily_profit max_price :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_11_l1346_134646


namespace NUMINAMATH_CALUDE_root_product_sum_l1346_134679

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x, Real.sqrt 2023 * x^3 - 4047 * x^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 3 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l1346_134679


namespace NUMINAMATH_CALUDE_four_people_handshakes_l1346_134676

/-- The number of handshakes in a group where each person shakes hands with every other person exactly once -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 4 people, where each person shakes hands with every other person exactly once, the total number of handshakes is 6. -/
theorem four_people_handshakes : num_handshakes 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_four_people_handshakes_l1346_134676


namespace NUMINAMATH_CALUDE_hexagon_side_length_l1346_134632

theorem hexagon_side_length (perimeter : ℝ) (h : perimeter = 30) : 
  perimeter / 6 = 5 := by sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l1346_134632


namespace NUMINAMATH_CALUDE_guest_cars_count_l1346_134605

/-- Calculates the number of guest cars given the total number of wheels and parent cars -/
def guest_cars (total_wheels : ℕ) (parent_cars : ℕ) : ℕ :=
  (total_wheels - 4 * parent_cars) / 4

/-- Theorem: Given 48 total wheels and 2 parent cars, the number of guest cars is 10 -/
theorem guest_cars_count : guest_cars 48 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_guest_cars_count_l1346_134605


namespace NUMINAMATH_CALUDE_shirt_discount_calculation_l1346_134695

/-- Given an original price and a discounted price, calculate the percentage discount -/
def calculate_discount (original_price discounted_price : ℚ) : ℚ :=
  (original_price - discounted_price) / original_price * 100

/-- Theorem: The percentage discount for a shirt with original price 933.33 and discounted price 560 is 40% -/
theorem shirt_discount_calculation :
  let original_price : ℚ := 933.33
  let discounted_price : ℚ := 560
  calculate_discount original_price discounted_price = 40 :=
by
  sorry

#eval calculate_discount 933.33 560

end NUMINAMATH_CALUDE_shirt_discount_calculation_l1346_134695


namespace NUMINAMATH_CALUDE_lending_years_calculation_l1346_134691

/-- Proves that the number of years the first part is lent is 5 -/
theorem lending_years_calculation (total_sum : ℝ) (second_part : ℝ) 
  (first_rate : ℝ) (second_rate : ℝ) (second_years : ℕ) :
  total_sum = 2665 →
  second_part = 1332.5 →
  first_rate = 0.03 →
  second_rate = 0.05 →
  second_years = 3 →
  let first_part := total_sum - second_part
  let first_interest := first_part * first_rate
  let second_interest := second_part * second_rate * second_years
  first_interest * (5 : ℝ) = second_interest :=
by sorry

end NUMINAMATH_CALUDE_lending_years_calculation_l1346_134691


namespace NUMINAMATH_CALUDE_truck_transport_time_l1346_134680

theorem truck_transport_time (total_time : ℝ) (first_truck_portion : ℝ) (actual_time : ℝ)
  (h1 : total_time = 6)
  (h2 : first_truck_portion = 3/5)
  (h3 : actual_time = 12) :
  ∃ (t1 t2 : ℝ),
    ((t1 = 10 ∧ t2 = 15) ∨ (t1 = 12 ∧ t2 = 12)) ∧
    (1 / t1 + 1 / t2 = 1 / total_time) ∧
    (first_truck_portion / t1 + (1 - first_truck_portion) / t2 = 1 / actual_time) := by
  sorry

end NUMINAMATH_CALUDE_truck_transport_time_l1346_134680


namespace NUMINAMATH_CALUDE_commentator_mistake_l1346_134643

def round_robin_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem commentator_mistake (n : ℕ) (x y : ℚ) (h1 : n = 15) :
  ¬(∃ (x y : ℚ),
    (x > 0) ∧
    (y > x) ∧
    (y < 2 * x) ∧
    (3 * x + 13 * y = round_robin_games n)) :=
  sorry

end NUMINAMATH_CALUDE_commentator_mistake_l1346_134643


namespace NUMINAMATH_CALUDE_remaining_volume_of_prism_with_cylinder_removed_l1346_134671

/-- The remaining volume of a rectangular prism after removing a cylindrical section -/
theorem remaining_volume_of_prism_with_cylinder_removed :
  let prism_length : ℝ := 5
  let prism_width : ℝ := 5
  let prism_height : ℝ := 6
  let cylinder_radius : ℝ := 2.5
  let prism_volume := prism_length * prism_width * prism_height
  let cylinder_volume := π * cylinder_radius^2 * prism_width
  let remaining_volume := prism_volume - cylinder_volume
  remaining_volume = 150 - 31.25 * π := by
sorry


end NUMINAMATH_CALUDE_remaining_volume_of_prism_with_cylinder_removed_l1346_134671


namespace NUMINAMATH_CALUDE_harold_bought_three_doughnuts_l1346_134661

def harold_doughnuts (harold_coffee : ℕ) (harold_total : ℚ) 
  (melinda_doughnuts : ℕ) (melinda_coffee : ℕ) (melinda_total : ℚ) 
  (doughnut_price : ℚ) : Prop :=
  ∃ (coffee_price : ℚ),
    (doughnut_price * 3 + coffee_price * harold_coffee = harold_total) ∧
    (doughnut_price * melinda_doughnuts + coffee_price * melinda_coffee = melinda_total)

theorem harold_bought_three_doughnuts :
  harold_doughnuts 4 4.91 5 6 7.59 0.45 :=
sorry

end NUMINAMATH_CALUDE_harold_bought_three_doughnuts_l1346_134661


namespace NUMINAMATH_CALUDE_brick_surface_area_l1346_134667

/-- The surface area of a rectangular prism. -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm brick is 164 cm². -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l1346_134667


namespace NUMINAMATH_CALUDE_smallest_w_l1346_134665

theorem smallest_w (w : ℕ+) : 
  (∃ k : ℕ, 936 * w.val = k * 2^5) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 3^3) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 11^2) →
  w.val ≥ 4356 :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l1346_134665


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l1346_134648

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a + 1 = 0) → 
  (b^3 - 2*b + 1 = 0) → 
  (c^3 - 2*c + 1 = 0) → 
  (a ≠ b) → (b ≠ c) → (c ≠ a) →
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = 10 / 3) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l1346_134648


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1346_134678

theorem negation_of_existence (P : ℕ → Prop) : 
  (¬∃ n, P n) ↔ (∀ n, ¬P n) := by sorry

theorem negation_of_proposition :
  (¬∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1346_134678


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_cubes_l1346_134601

theorem roots_sum_reciprocal_cubes (r s : ℝ) : 
  (3 * r^2 + 5 * r + 2 = 0) → 
  (3 * s^2 + 5 * s + 2 = 0) → 
  (r ≠ s) →
  (1 / r^3 + 1 / s^3 = -27 / 35) :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_cubes_l1346_134601


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1346_134631

theorem cubic_equation_solutions :
  ∀ (z : ℂ), z^3 = -27 ↔ z = -3 ∨ z = (3 / 2 : ℂ) + (3 / 2 : ℂ) * Complex.I * Real.sqrt 3 ∨ z = (3 / 2 : ℂ) - (3 / 2 : ℂ) * Complex.I * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1346_134631


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1346_134613

/-- The total surface area of a cylinder with height 12 and radius 4 is 128π. -/
theorem cylinder_surface_area :
  let h : ℝ := 12
  let r : ℝ := 4
  let base_area : ℝ := π * r^2
  let lateral_area : ℝ := 2 * π * r * h
  let total_area : ℝ := 2 * base_area + lateral_area
  total_area = 128 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1346_134613


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1346_134625

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1346_134625


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1346_134664

theorem shopkeeper_profit (c s : ℝ) (p : ℝ) (h1 : c > 0) (h2 : s > c) :
  s = c * (1 + p / 100) ∧ 
  s = (0.9 * c) * (1 + (p + 12) / 100) →
  p = 8 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1346_134664


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_C_equals_C_l1346_134690

-- Define the sets A, B, and C
def A : Set ℝ := {x | -2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 * x - 5 ≥ x - 1}
def C (m : ℝ) : Set ℝ := {x | -x + m > 0}

-- Theorem for part 1
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < 5} := by sorry

-- Theorem for part 2
theorem union_A_C_equals_C (m : ℝ) : A ∪ C m = C m ↔ m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_C_equals_C_l1346_134690


namespace NUMINAMATH_CALUDE_volume_of_solid_T_l1346_134699

/-- The solid T is defined as the set of all points (x, y, z) in ℝ³ that satisfy
    the given inequalities. -/
def solid_T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | 
    let (x, y, z) := p
    (|x| + |y| ≤ 1.5) ∧ (|x| + |z| ≤ 1) ∧ (|y| + |z| ≤ 1)}

/-- The volume of a set in ℝ³. -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the volume of solid T is 2/3. -/
theorem volume_of_solid_T : volume solid_T = 2/3 := by sorry

end NUMINAMATH_CALUDE_volume_of_solid_T_l1346_134699


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l1346_134681

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (onion_weight carrot_weight : ℕ),
    onion_weight + carrot_weight = (bag_weights.sum - t) ∧
    carrot_weight = 2 * onion_weight ∧
    ∃ (onion_bags carrot_bags : List ℕ),
      onion_bags ++ carrot_bags = bag_weights.filter (· ≠ t) ∧
      onion_bags.sum = onion_weight ∧
      carrot_bags.sum = carrot_weight

theorem turnip_bag_weights :
  ∀ t, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 := by sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l1346_134681


namespace NUMINAMATH_CALUDE_probability_all_female_committee_l1346_134647

def total_group_size : ℕ := 8
def num_females : ℕ := 5
def num_males : ℕ := 3
def committee_size : ℕ := 3

theorem probability_all_female_committee :
  (Nat.choose num_females committee_size : ℚ) / (Nat.choose total_group_size committee_size) = 5 / 28 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_female_committee_l1346_134647


namespace NUMINAMATH_CALUDE_students_in_both_competitions_l1346_134677

theorem students_in_both_competitions 
  (total_students : ℕ) 
  (math_participants : ℕ) 
  (physics_participants : ℕ) 
  (no_competition_participants : ℕ) 
  (h1 : total_students = 40)
  (h2 : math_participants = 31)
  (h3 : physics_participants = 20)
  (h4 : no_competition_participants = 8) :
  total_students = math_participants + physics_participants + no_competition_participants - 19 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_competitions_l1346_134677


namespace NUMINAMATH_CALUDE_tea_hot_chocolate_difference_mo_drink_difference_l1346_134642

/-- Represents the drinking habits and week data for Mo --/
structure MoDrinkingHabits where
  n : ℕ  -- Number of hot chocolate cups on rainy days
  total_cups : ℕ  -- Total cups drunk in a week
  rainy_days : ℕ  -- Number of rainy days in a week

/-- Theorem stating the difference between tea and hot chocolate cups --/
theorem tea_hot_chocolate_difference (mo : MoDrinkingHabits) 
  (h1 : mo.total_cups = 26)
  (h2 : mo.rainy_days = 1) :
  3 * (7 - mo.rainy_days) - mo.n * mo.rainy_days = 10 := by
  sorry

/-- Main theorem proving the difference is 10 --/
theorem mo_drink_difference : ∃ mo : MoDrinkingHabits, 
  mo.total_cups = 26 ∧ 
  mo.rainy_days = 1 ∧ 
  3 * (7 - mo.rainy_days) - mo.n * mo.rainy_days = 10 := by
  sorry

end NUMINAMATH_CALUDE_tea_hot_chocolate_difference_mo_drink_difference_l1346_134642


namespace NUMINAMATH_CALUDE_divisor_equation_solution_l1346_134658

def is_sixth_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (d1 d2 d3 d4 d5 : ℕ), d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧ d4 ∣ n ∧ d5 ∣ n ∧
    1 < d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 < d)

def is_seventh_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (d1 d2 d3 d4 d5 d6 : ℕ), d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧ d4 ∣ n ∧ d5 ∣ n ∧ d6 ∣ n ∧
    1 < d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 < d6 ∧ d6 < d)

theorem divisor_equation_solution (n : ℕ) :
  (∃ (d6 d7 : ℕ), is_sixth_divisor n d6 ∧ is_seventh_divisor n d7 ∧ n = d6^2 + d7^2 - 1) →
  n = 144 ∨ n = 1984 :=
by sorry

end NUMINAMATH_CALUDE_divisor_equation_solution_l1346_134658


namespace NUMINAMATH_CALUDE_smallest_special_number_l1346_134652

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_below (n k : ℕ) : Prop := ∀ p : ℕ, p < k → is_prime p → n % p ≠ 0

theorem smallest_special_number : 
  ∀ n : ℕ, n > 0 → n < 4091 → 
  (¬ is_prime n ∧ ¬ is_perfect_square n ∧ has_no_prime_factor_below n 60) → False :=
sorry

#check smallest_special_number

end NUMINAMATH_CALUDE_smallest_special_number_l1346_134652


namespace NUMINAMATH_CALUDE_height_survey_groups_l1346_134600

theorem height_survey_groups (max_height min_height class_interval : ℝ) 
  (h1 : max_height = 173)
  (h2 : min_height = 140)
  (h3 : class_interval = 5) : 
  Int.ceil ((max_height - min_height) / class_interval) = 7 := by
  sorry

end NUMINAMATH_CALUDE_height_survey_groups_l1346_134600


namespace NUMINAMATH_CALUDE_parallel_line_divides_equally_l1346_134607

-- Define the shaded area
def shaded_area : ℝ := 10

-- Define the distance from MO to the parallel line
def distance_from_MO : ℝ := 2.6

-- Define the function that calculates the area above the parallel line
def area_above (d : ℝ) : ℝ := sorry

-- Theorem statement
theorem parallel_line_divides_equally :
  area_above distance_from_MO = shaded_area / 2 := by sorry

end NUMINAMATH_CALUDE_parallel_line_divides_equally_l1346_134607


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l1346_134623

/-- Represents the annual birth rate per 1000 people in a country. -/
def birth_rate : ℝ := sorry

/-- Represents the annual death rate per 1000 people in a country. -/
def death_rate : ℝ := 19.4

/-- Represents the number of years it takes for the population to double. -/
def doubling_time : ℝ := 35

/-- The Rule of 70 for population growth. -/
axiom rule_of_70 (growth_rate : ℝ) : 
  doubling_time = 70 / growth_rate

/-- The net growth rate is the difference between birth rate and death rate. -/
def net_growth_rate : ℝ := birth_rate - death_rate

theorem birth_rate_calculation : birth_rate = 21.4 := by sorry

end NUMINAMATH_CALUDE_birth_rate_calculation_l1346_134623


namespace NUMINAMATH_CALUDE_unit_circle_angle_properties_l1346_134640

theorem unit_circle_angle_properties (α : Real) :
  (∃ P : Real × Real, P.1^2 + P.2^2 = 1 ∧ P.1 = 3/5 ∧ P.2 = 4/5 ∧ 
   Real.cos α = P.1 ∧ Real.sin α = P.2) →
  Real.sin (π - α) = 4/5 ∧ Real.tan (π/4 + α) = -7 := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_angle_properties_l1346_134640


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1346_134666

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Check if four points form a valid parallelogram -/
def isValidParallelogram (p : Parallelogram) : Prop :=
  (p.v1.x + p.v3.x = p.v2.x + p.v4.x) ∧ 
  (p.v1.y + p.v3.y = p.v2.y + p.v4.y)

theorem parallelogram_fourth_vertex 
  (p : Parallelogram)
  (h1 : p.v1 = Point.mk (-1) 0)
  (h2 : p.v2 = Point.mk 3 0)
  (h3 : p.v3 = Point.mk 1 (-5)) :
  isValidParallelogram p →
  (p.v4 = Point.mk 5 (-5) ∨ p.v4 = Point.mk (-3) (-5) ∨ p.v4 = Point.mk 1 5) :=
by sorry


end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1346_134666


namespace NUMINAMATH_CALUDE_power_product_equality_l1346_134673

theorem power_product_equality : 3^5 * 6^5 = 1889568 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1346_134673


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l1346_134674

theorem quadratic_inequality_coefficient_sum (a b : ℝ) :
  (∀ x, ax^2 + b*x - 4 > 0 ↔ 1 < x ∧ x < 2) →
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l1346_134674


namespace NUMINAMATH_CALUDE_digit_D_is_nine_l1346_134604

/-- Represents a two-digit number --/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  tens_is_digit : tens < 10
  ones_is_digit : ones < 10

def value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem digit_D_is_nine
  (A B C D : Nat)
  (A_is_digit : A < 10)
  (B_is_digit : B < 10)
  (C_is_digit : C < 10)
  (D_is_digit : D < 10)
  (addition : value ⟨A, B, A_is_digit, B_is_digit⟩ + value ⟨C, B, C_is_digit, B_is_digit⟩ = value ⟨D, A, D_is_digit, A_is_digit⟩)
  (subtraction : value ⟨A, B, A_is_digit, B_is_digit⟩ - value ⟨C, B, C_is_digit, B_is_digit⟩ = A) :
  D = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_D_is_nine_l1346_134604


namespace NUMINAMATH_CALUDE_josh_marbles_problem_l1346_134651

/-- The number of marbles Josh lost -/
def marbles_lost : ℕ := 16

/-- The number of marbles Josh found -/
def marbles_found : ℕ := 8

/-- The initial number of marbles Josh had -/
def initial_marbles : ℕ := 4

theorem josh_marbles_problem :
  marbles_lost = marbles_found + 8 :=
by sorry

end NUMINAMATH_CALUDE_josh_marbles_problem_l1346_134651


namespace NUMINAMATH_CALUDE_problem_solution_l1346_134689

def problem (a b : ℝ × ℝ) : Prop :=
  let angle := 2 * Real.pi / 3
  let magnitude_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  (a = (2, 0)) ∧ 
  (magnitude_b = 1) ∧
  (a.1 * b.1 + a.2 * b.2 = Real.cos angle * magnitude_b * 2) →
  Real.sqrt (((a.1 + 2 * b.1) ^ 2) + ((a.2 + 2 * b.2) ^ 2)) = 2

theorem problem_solution : ∃ (a b : ℝ × ℝ), problem a b := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1346_134689


namespace NUMINAMATH_CALUDE_quadratic_sequence_proof_l1346_134620

/-- A quadratic function passing through the origin with given derivative -/
noncomputable def f (x : ℝ) : ℝ := sorry

/-- The sequence a_n -/
def a (n : ℕ+) : ℝ := sorry

/-- The sum of the first n terms of a_n -/
def S (n : ℕ+) : ℝ := sorry

/-- The sequence b_n -/
def b (n : ℕ+) : ℝ := sorry

/-- The sum of the first n terms of b_n -/
def T (n : ℕ+) : ℝ := sorry

theorem quadratic_sequence_proof 
  (h1 : f 0 = 0)
  (h2 : ∀ x, deriv f x = 6 * x - 2)
  (h3 : ∀ n : ℕ+, S n = f n) :
  (∀ n : ℕ+, a n = 6 * n - 5) ∧
  (∀ m : ℝ, (∀ n : ℕ+, T n ≥ m / 20) ↔ m ≤ 60 / 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sequence_proof_l1346_134620


namespace NUMINAMATH_CALUDE_negation_of_forall_exp_geq_x_plus_one_l1346_134628

theorem negation_of_forall_exp_geq_x_plus_one :
  (¬ ∀ x : ℝ, Real.exp x ≥ x + 1) ↔ (∃ x₀ : ℝ, Real.exp x₀ < x₀ + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_exp_geq_x_plus_one_l1346_134628


namespace NUMINAMATH_CALUDE_first_fun_friday_march31_l1346_134682

/-- Represents a date in a year -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

/-- Function to determine if a given date is a Friday -/
def isFriday (d : Date) (startDay : DayOfWeek) : Bool := sorry

/-- Function to count the number of Fridays in a month -/
def countFridays (month : Nat) (startDay : DayOfWeek) : Nat := sorry

/-- Function to determine if a date is a Fun Friday -/
def isFunFriday (d : Date) (startDay : DayOfWeek) : Bool := sorry

/-- Theorem stating that the first Fun Friday of the year is on March 31 -/
theorem first_fun_friday_march31 (startDay : DayOfWeek) :
  startDay = DayOfWeek.Wednesday →
  (∀ d : Date, d.month < 3 → ¬isFunFriday d startDay) →
  isFunFriday { month := 3, day := 31 } startDay :=
sorry

end NUMINAMATH_CALUDE_first_fun_friday_march31_l1346_134682


namespace NUMINAMATH_CALUDE_geometric_sequence_expression_zero_l1346_134669

/-- For a geometric sequence, the product of terms equidistant from the ends is constant -/
def geometric_sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a 1 * a n = a 2 * a (n - 1)

/-- The expression (a₁aₙ)² - a₂a₄aₙ₋₁aₙ₋₃ equals zero for any geometric sequence -/
theorem geometric_sequence_expression_zero (a : ℕ → ℝ) (n : ℕ) 
  (h : geometric_sequence_property a) : 
  (a 1 * a n)^2 - (a 2 * a 4 * a (n-1) * a (n-3)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_expression_zero_l1346_134669


namespace NUMINAMATH_CALUDE_function_with_two_symmetries_is_periodic_l1346_134693

/-- A function with two lines of symmetry is periodic -/
theorem function_with_two_symmetries_is_periodic
  (f : ℝ → ℝ) (m n : ℝ) (hm : m ≠ n)
  (sym_m : ∀ x, f x = f (2 * m - x))
  (sym_n : ∀ x, f x = f (2 * n - x)) :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

end NUMINAMATH_CALUDE_function_with_two_symmetries_is_periodic_l1346_134693


namespace NUMINAMATH_CALUDE_expression_evaluation_l1346_134654

theorem expression_evaluation : 
  (2020^3 - 3 * 2020^2 * 2021 + 5 * 2020 * 2021^2 - 2021^3 + 4) / (2020 * 2021) = 
  4042 + 3 / (4080420 : ℚ) := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1346_134654


namespace NUMINAMATH_CALUDE_circle_passes_through_intersection_point_l1346_134649

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + 2*y + 1 = 0
def line2 (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (4, 3)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 25

-- Theorem statement
theorem circle_passes_through_intersection_point :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ circle_equation x y :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_intersection_point_l1346_134649


namespace NUMINAMATH_CALUDE_extra_interest_proof_l1346_134694

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

def investment_amount : ℝ := 7000
def high_rate : ℝ := 0.18
def low_rate : ℝ := 0.12
def investment_time : ℝ := 2

theorem extra_interest_proof :
  simple_interest investment_amount high_rate investment_time -
  simple_interest investment_amount low_rate investment_time = 840 := by
  sorry

end NUMINAMATH_CALUDE_extra_interest_proof_l1346_134694


namespace NUMINAMATH_CALUDE_system_solution_l1346_134609

theorem system_solution : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 → 
  (Real.log x / Real.log 4 - Real.log y / Real.log 2 = 0) ∧ 
  (x^2 - 5*y^2 + 4 = 0) → 
  ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1346_134609


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_is_perpendicular_l1346_134672

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_parallel_is_perpendicular
  (α β : Plane) (m n : Line)
  (h1 : α ≠ β)
  (h2 : m ≠ n)
  (h3 : perpendicular_line_plane m β)
  (h4 : parallel_line_plane n β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_is_perpendicular_l1346_134672


namespace NUMINAMATH_CALUDE_tangent_line_coefficients_l1346_134618

/-- Given a curve y = x^2 + ax + b with a tangent line at (1, b) with equation x - y + 1 = 0,
    prove that a = -1 and b = 2 -/
theorem tangent_line_coefficients (a b : ℝ) : 
  (∀ x y : ℝ, y = x^2 + a*x + b) →
  (∃ y : ℝ, y = 1^2 + a*1 + b) →
  (∀ x y : ℝ, y = 1^2 + a*1 + b → x - y + 1 = 0 → x = 1) →
  a = -1 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_coefficients_l1346_134618


namespace NUMINAMATH_CALUDE_expression_simplification_inequality_system_equivalence_l1346_134610

-- Part 1
theorem expression_simplification (a : ℝ) :
  (a - 3)^2 + a*(4 - a) = -2*a + 9 := by sorry

-- Part 2
theorem inequality_system_equivalence (x : ℝ) :
  -2 ≤ x ∧ x < 3 ↔ 3*x - 5 < x + 1 ∧ 2*(2*x - 1) ≥ 3*x - 4 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_inequality_system_equivalence_l1346_134610


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1346_134683

theorem absolute_value_inequality (x : ℝ) : |x + 3| - |x - 2| ≥ 3 ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1346_134683


namespace NUMINAMATH_CALUDE_exercise_gender_relation_l1346_134653

-- Define the contingency table
def total_sample : ℕ := 100
def boys_dislike : ℕ := 10
def girls_like : ℕ := 20
def prob_dislike : ℚ := 4/10

-- Calculate other values
def boys_like : ℕ := total_sample / 2 - boys_dislike
def girls_dislike : ℕ := total_sample / 2 - girls_like
def total_like : ℕ := boys_like + girls_like
def total_dislike : ℕ := boys_dislike + girls_dislike

-- Define K^2 formula
def K_squared (a b c d : ℕ) : ℚ :=
  (total_sample * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Theorem to prove
theorem exercise_gender_relation :
  K_squared boys_like girls_like boys_dislike girls_dislike > 10828/1000 :=
sorry

end NUMINAMATH_CALUDE_exercise_gender_relation_l1346_134653


namespace NUMINAMATH_CALUDE_classrooms_needed_l1346_134614

/-- Given a school with 390 students and classrooms that hold 30 students each,
    prove that 13 classrooms are needed. -/
theorem classrooms_needed (total_students : Nat) (students_per_classroom : Nat) :
  total_students = 390 →
  students_per_classroom = 30 →
  (total_students + students_per_classroom - 1) / students_per_classroom = 13 := by
  sorry

end NUMINAMATH_CALUDE_classrooms_needed_l1346_134614


namespace NUMINAMATH_CALUDE_typing_competition_equation_l1346_134617

/-- Prove that in a typing competition where A types x characters per minute and 
    B types (x-10) characters per minute, if A types 900 characters and B types 840 characters 
    in the same amount of time, then the equation 900/x = 840/(x-10) holds. -/
theorem typing_competition_equation (x : ℝ) 
    (hx : x > 10) -- Ensure x - 10 is positive
    (hA : 900 / x = 840 / (x - 10)) : -- Time taken by A equals time taken by B
  900 / x = 840 / (x - 10) := by
  sorry

end NUMINAMATH_CALUDE_typing_competition_equation_l1346_134617


namespace NUMINAMATH_CALUDE_area_is_nine_l1346_134698

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D plane defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Triangular region formed by two lines and x-axis -/
structure TriangularRegion where
  line1 : Line
  line2 : Line

def line1 : Line := { p1 := ⟨0, 3⟩, p2 := ⟨6, 0⟩ }
def line2 : Line := { p1 := ⟨1, 6⟩, p2 := ⟨7, 1⟩ }

def region : TriangularRegion := { line1 := line1, line2 := line2 }

/-- Calculate the area of the triangular region -/
def calculateArea (r : TriangularRegion) : ℝ :=
  sorry

theorem area_is_nine : calculateArea region = 9 := by
  sorry

end NUMINAMATH_CALUDE_area_is_nine_l1346_134698


namespace NUMINAMATH_CALUDE_same_solution_implies_m_value_l1346_134688

theorem same_solution_implies_m_value : 
  ∀ (x m : ℝ), 
  (2 * x - m = 1 ∧ 3 * x = 2 * (x - 1)) → 
  m = -5 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_m_value_l1346_134688


namespace NUMINAMATH_CALUDE_rectangle_area_l1346_134675

/-- Given a rectangle with length 16 and diagonal 20, prove its area is 192. -/
theorem rectangle_area (length width diagonal : ℝ) : 
  length = 16 → 
  diagonal = 20 → 
  length^2 + width^2 = diagonal^2 → 
  length * width = 192 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1346_134675


namespace NUMINAMATH_CALUDE_triangle_max_tan_diff_l1346_134650

open Real

theorem triangle_max_tan_diff (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a * cos B - b * cos A = c / 2 →
  (∀ θ, 0 < θ ∧ θ < π → tan (A - B) ≤ tan (A - θ)) →
  B = π / 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_tan_diff_l1346_134650


namespace NUMINAMATH_CALUDE_bug_return_probability_l1346_134670

/-- Probability of returning to the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | 1 => 0
  | (n + 2) => (1 / 3) * (1 - Q (n + 1)) + (1 / 3) * Q n

/-- The probability of returning to the starting vertex on the tenth move is 34817/59049 -/
theorem bug_return_probability :
  Q 10 = 34817 / 59049 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l1346_134670


namespace NUMINAMATH_CALUDE_equation_system_solution_l1346_134616

theorem equation_system_solution (a b : ℝ) :
  (∃ (a' : ℝ), a' * 1 + 4 * (-1) = 23 ∧ 3 * 1 - b * (-1) = 5) →
  (∃ (b' : ℝ), a * 7 + 4 * (-3) = 23 ∧ 3 * 7 - b' * (-3) = 5) →
  (a^2 - 2*a*b + b^2 = 9) ∧
  (a * 3 + 4 * 2 = 23 ∧ 3 * 3 - b * 2 = 5) := by
sorry

end NUMINAMATH_CALUDE_equation_system_solution_l1346_134616


namespace NUMINAMATH_CALUDE_remainder_of_A_mod_9_l1346_134624

-- Define the arithmetic sequence
def arithmetic_sequence : List Nat :=
  List.range 502 |> List.map (fun k => 4 * k + 2)

-- Define the large number A as a string
def A : String :=
  arithmetic_sequence.foldl (fun acc x => acc ++ toString x) ""

-- Theorem statement
theorem remainder_of_A_mod_9 :
  (A.foldl (fun acc c => (10 * acc + c.toNat - '0'.toNat) % 9) 0) = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_A_mod_9_l1346_134624


namespace NUMINAMATH_CALUDE_function_difference_inequality_l1346_134684

theorem function_difference_inequality
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, deriv f x > deriv g x)
  {a b : ℝ}
  (hab : a > b) :
  f a - f b > g a - g b :=
sorry

end NUMINAMATH_CALUDE_function_difference_inequality_l1346_134684


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1346_134633

/-- A quadratic function f(x) = ax^2 + bx satisfying certain conditions -/
def QuadraticFunction (a b : ℝ) (ha : a ≠ 0) : ℝ → ℝ := fun x ↦ a * x^2 + b * x

theorem quadratic_function_properties 
  (a b : ℝ) (ha : a ≠ 0) 
  (h1 : ∀ x, QuadraticFunction a b ha (x - 1) = QuadraticFunction a b ha (3 - x))
  (h2 : ∃! x, QuadraticFunction a b ha x = 2 * x) :
  (∀ x, QuadraticFunction a b ha x = -x^2 + 2*x) ∧ 
  (∀ t, (t : ℝ) > 0 → 
    (∀ x, x ∈ Set.Icc 0 t → QuadraticFunction a b ha x ≤ 
      (if t > 1 then 1 else -t^2 + 2*t))) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l1346_134633


namespace NUMINAMATH_CALUDE_sqrt_relation_l1346_134686

theorem sqrt_relation (h1 : Real.sqrt 23.6 = 4.858) (h2 : Real.sqrt 2.36 = 1.536) :
  Real.sqrt 0.00236 = 0.04858 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_relation_l1346_134686


namespace NUMINAMATH_CALUDE_alpha_range_l1346_134639

theorem alpha_range (α : Real) (h1 : 0 ≤ α) (h2 : α ≤ π) 
  (h3 : ∀ x : Real, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) :
  α ∈ Set.Icc 0 (π / 6) ∪ Set.Icc (5 * π / 6) π := by
  sorry

end NUMINAMATH_CALUDE_alpha_range_l1346_134639


namespace NUMINAMATH_CALUDE_machine_x_production_rate_l1346_134635

/-- The number of sprockets produced by both machines -/
def total_sprockets : ℕ := 660

/-- The additional time taken by Machine X compared to Machine B -/
def time_difference : ℕ := 10

/-- The production rate of Machine B relative to Machine X -/
def rate_ratio : ℚ := 11/10

/-- The production rate of Machine X in sprockets per hour -/
def machine_x_rate : ℚ := 6

theorem machine_x_production_rate :
  ∃ (machine_b_rate : ℚ) (time_x time_b : ℚ),
    machine_b_rate = rate_ratio * machine_x_rate ∧
    time_x = time_b + time_difference ∧
    machine_x_rate * time_x = total_sprockets ∧
    machine_b_rate * time_b = total_sprockets :=
by sorry

end NUMINAMATH_CALUDE_machine_x_production_rate_l1346_134635


namespace NUMINAMATH_CALUDE_airport_gate_probability_l1346_134656

/-- The number of gates in the airport -/
def num_gates : ℕ := 16

/-- The distance between adjacent gates in feet -/
def distance_between_gates : ℕ := 75

/-- The maximum distance Dina is willing to walk in feet -/
def max_walking_distance : ℕ := 300

/-- The probability of walking 300 feet or less to the new gate -/
def probability_short_walk : ℚ := 8/15

theorem airport_gate_probability :
  let total_possibilities := num_gates * (num_gates - 1)
  let gates_within_distance := 2 * (max_walking_distance / distance_between_gates)
  let favorable_outcomes := num_gates * gates_within_distance
  (favorable_outcomes : ℚ) / total_possibilities = probability_short_walk :=
sorry

end NUMINAMATH_CALUDE_airport_gate_probability_l1346_134656


namespace NUMINAMATH_CALUDE_log_square_ratio_l1346_134603

theorem log_square_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (h2 : x * y = 243) : 
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_log_square_ratio_l1346_134603


namespace NUMINAMATH_CALUDE_carries_tshirt_purchase_l1346_134612

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℝ := 9.15

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 22

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := tshirt_cost * num_tshirts

/-- Theorem stating that the total cost of Carrie's t-shirt purchase is $201.30 -/
theorem carries_tshirt_purchase : total_cost = 201.30 := by
  sorry

end NUMINAMATH_CALUDE_carries_tshirt_purchase_l1346_134612


namespace NUMINAMATH_CALUDE_incorrect_rectangle_l1346_134636

/-- Represents a 3x3 grid of rectangle perimeters --/
structure PerimeterGrid :=
  (top_row : Fin 3 → ℕ)
  (middle_row : Fin 3 → ℕ)
  (bottom_row : Fin 3 → ℕ)

/-- The given grid of perimeters --/
def given_grid : PerimeterGrid :=
  { top_row := ![14, 16, 12],
    middle_row := ![18, 18, 2],
    bottom_row := ![16, 18, 14] }

/-- Predicate to check if a perimeter grid is valid --/
def is_valid_grid (grid : PerimeterGrid) : Prop :=
  ∀ i j, i < 3 → j < 3 → 
    (grid.top_row i > 0) ∧ 
    (grid.middle_row i > 0) ∧ 
    (grid.bottom_row i > 0)

/-- Theorem stating that the rectangle with perimeter 2 is incorrect --/
theorem incorrect_rectangle (grid : PerimeterGrid) 
  (h : is_valid_grid grid) : 
  ∃ i j, grid.middle_row j = 2 ∧ 
    (i = 1 ∨ j = 2) ∧ 
    ¬(∀ k l, k ≠ i ∨ l ≠ j → grid.middle_row l > 2) :=
sorry


end NUMINAMATH_CALUDE_incorrect_rectangle_l1346_134636


namespace NUMINAMATH_CALUDE_spherical_sector_central_angle_l1346_134615

theorem spherical_sector_central_angle (R : ℝ) (α : ℝ) :
  R > 0 →
  (∃ r m : ℝ, R * π * r = 2 * R * π * m ∧ 
              R^2 = r^2 + (R - m)^2 ∧ 
              0 < m ∧ m < R) →
  α = 2 * Real.arccos (3/5) :=
sorry

end NUMINAMATH_CALUDE_spherical_sector_central_angle_l1346_134615


namespace NUMINAMATH_CALUDE_ten_coin_flips_sequences_l1346_134692

/-- The number of distinct sequences when flipping a coin n times -/
def coin_flip_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences when flipping a coin 10 times is 1024 -/
theorem ten_coin_flips_sequences : coin_flip_sequences 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ten_coin_flips_sequences_l1346_134692


namespace NUMINAMATH_CALUDE_cube_probability_l1346_134634

/-- A cube with side length 3 -/
def Cube := Fin 3 → Fin 3 → Fin 3

/-- The number of unit cubes in the larger cube -/
def totalCubes : ℕ := 27

/-- The number of unit cubes with exactly two painted faces -/
def twoPaintedFaces : ℕ := 4

/-- The number of unit cubes with no painted faces -/
def noPaintedFaces : ℕ := 8

/-- The probability of selecting one cube with two painted faces and one with no painted faces -/
def probability : ℚ := 32 / 351

theorem cube_probability : 
  probability = (twoPaintedFaces * noPaintedFaces : ℚ) / (totalCubes.choose 2) := by sorry

end NUMINAMATH_CALUDE_cube_probability_l1346_134634


namespace NUMINAMATH_CALUDE_thirtieth_term_is_30_l1346_134611

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a₁ : ℝ
  -- Common difference of the sequence
  d : ℝ
  -- Sum of first 7 terms is 28
  sum_7 : (7 * a₁ + 21 * d) = 28
  -- Sum of 2nd, 5th, and 8th terms is 15
  sum_258 : (a₁ + d) + (a₁ + 4*d) + (a₁ + 7*d) = 15

/-- The 30th term of the arithmetic sequence is 30 -/
theorem thirtieth_term_is_30 (seq : ArithmeticSequence) : 
  seq.a₁ + 29 * seq.d = 30 := by sorry

end NUMINAMATH_CALUDE_thirtieth_term_is_30_l1346_134611


namespace NUMINAMATH_CALUDE_calvins_weight_after_one_year_l1346_134608

/-- Represents the weight loss from gym training for each month --/
def gym_training_loss : List Nat := [8, 5, 7, 6, 8, 7, 5, 7, 4, 6, 5, 7]

/-- Represents the weight loss from additional exercise routines for each month --/
def exercise_routines_loss : List Nat := [2, 3, 4, 3, 2, 4, 3, 2, 1, 3, 2, 4]

/-- Calculates Calvin's weight after one year --/
def calculate_final_weight (initial_weight : Nat) (gym_loss : List Nat) (diet_loss_per_month : Nat) (exercise_loss : List Nat) : Nat :=
  initial_weight - (gym_loss.sum + diet_loss_per_month * 12 + exercise_loss.sum)

/-- Theorem stating Calvin's weight after one year --/
theorem calvins_weight_after_one_year :
  calculate_final_weight 250 gym_training_loss 3 exercise_routines_loss = 106 := by
  sorry


end NUMINAMATH_CALUDE_calvins_weight_after_one_year_l1346_134608


namespace NUMINAMATH_CALUDE_modulo_congruence_solution_l1346_134685

theorem modulo_congruence_solution :
  ∃! k : ℤ, 0 ≤ k ∧ k < 17 ∧ -175 ≡ k [ZMOD 17] := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_solution_l1346_134685


namespace NUMINAMATH_CALUDE_cube_surface_area_l1346_134641

theorem cube_surface_area (volume : ℝ) (h : volume = 64) : 
  (6 : ℝ) * (volume ^ (1/3 : ℝ))^2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1346_134641


namespace NUMINAMATH_CALUDE_no_mode_in_set_l1346_134659

def number_set : Finset ℕ := {91, 85, 80, 83, 84}

def x : ℕ := 504 - (91 + 85 + 80 + 83 + 84)

def complete_set : Finset ℕ := number_set ∪ {x}

theorem no_mode_in_set :
  (Finset.card complete_set = 6) ∧
  (Finset.sum complete_set id / Finset.card complete_set = 84) →
  ∀ n : ℕ, (complete_set.filter (λ m => m = n)).card ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_no_mode_in_set_l1346_134659


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1346_134627

def a (n : ℕ) : ℤ := 2^n - (-1)^n

theorem arithmetic_sequence_properties :
  (∃ (n₁ n₂ n₃ : ℕ), n₁ < n₂ ∧ n₂ < n₃ ∧ 
    n₂ = n₁ + 1 ∧ n₃ = n₂ + 1 ∧
    2 * a n₂ = a n₁ + a n₃ ∧ n₁ = 2) ∧
  (∀ n₂ n₃ : ℕ, 1 < n₂ ∧ n₂ < n₃ ∧ 
    2 * a n₂ = a 1 + a n₃ → n₃ - n₂ = 1) ∧
  (∀ t : ℕ, t > 3 → 
    ¬∃ (s : ℕ → ℕ), Monotone s ∧ 
      (∀ i j : Fin t, i < j → s i < s j) ∧
      (∀ i : Fin (t - 1), 
        2 * a (s (i + 1)) = a (s i) + a (s (i + 2)))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1346_134627


namespace NUMINAMATH_CALUDE_unique_prime_square_sum_l1346_134637

theorem unique_prime_square_sum (p q : ℕ) : 
  Prime p → Prime q → ∃ (n : ℕ), p^(q+1) + q^(p+1) = n^2 → p = 2 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_square_sum_l1346_134637


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l1346_134645

theorem cube_surface_area_from_volume (volume : ℝ) (side_length : ℝ) (surface_area : ℝ) : 
  volume = 343 →
  volume = side_length ^ 3 →
  surface_area = 6 * side_length ^ 2 →
  surface_area = 294 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l1346_134645


namespace NUMINAMATH_CALUDE_sum_in_interval_l1346_134663

theorem sum_in_interval : 
  let a : ℚ := 4 + 5/9
  let b : ℚ := 5 + 3/4
  let c : ℚ := 7 + 8/17
  17.5 < a + b + c ∧ a + b + c < 18 :=
by sorry

end NUMINAMATH_CALUDE_sum_in_interval_l1346_134663


namespace NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l1346_134638

-- Problem 1
def problem1 (a b : ℕ) : Prop :=
  a ≠ b ∧
  ∃ p k : ℕ, Prime p ∧ b^2 + a = p^k ∧
  (b^2 + a) ∣ (a^2 + b)

theorem problem1_solution :
  ∀ a b : ℕ, problem1 a b ↔ (a = 5 ∧ b = 2) :=
sorry

-- Problem 2
def problem2 (a b : ℕ) : Prop :=
  a > 1 ∧ b > 1 ∧ a ≠ b ∧
  (b^2 + a - 1) ∣ (a^2 + b - 1)

theorem problem2_solution :
  ∀ a b : ℕ, problem2 a b →
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ (b^2 + a - 1) ∧ q ∣ (b^2 + a - 1) :=
sorry

end NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l1346_134638


namespace NUMINAMATH_CALUDE_calculation_proof_l1346_134668

theorem calculation_proof : 1525 + 140 / 70 - 225 = 1302 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1346_134668


namespace NUMINAMATH_CALUDE_special_property_implies_units_nine_l1346_134662

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : 1 ≤ tens ∧ tens ≤ 9 ∧ 0 ≤ units ∧ units ≤ 9

/-- The property described in the problem -/
def has_special_property (n : TwoDigitNumber) : Prop :=
  n.tens + n.units + n.tens * n.units = 10 * n.tens + n.units

theorem special_property_implies_units_nine :
  ∀ n : TwoDigitNumber, has_special_property n → n.units = 9 := by
  sorry

end NUMINAMATH_CALUDE_special_property_implies_units_nine_l1346_134662


namespace NUMINAMATH_CALUDE_interest_rate_middle_period_l1346_134696

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_middle_period 
  (principal : ℝ) 
  (rate1 : ℝ) 
  (rate3 : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (time3 : ℝ) 
  (total_interest : ℝ) :
  principal = 8000 →
  rate1 = 0.08 →
  rate3 = 0.12 →
  time1 = 4 →
  time2 = 6 →
  time3 = 5 →
  total_interest = 12160 →
  ∃ (rate2 : ℝ), 
    rate2 = 0.1 ∧
    total_interest = simple_interest principal rate1 time1 + 
                     simple_interest principal rate2 time2 + 
                     simple_interest principal rate3 time3 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_middle_period_l1346_134696


namespace NUMINAMATH_CALUDE_yadav_expenditure_l1346_134626

/-- Represents Mr. Yadav's monthly salary in some monetary unit -/
def monthly_salary : ℝ := sorry

/-- Represents the percentage of salary spent on consumable items -/
def consumable_percentage : ℝ := 0.6

/-- Represents the percentage of remaining salary spent on clothes and transport -/
def clothes_transport_percentage : ℝ := 0.5

/-- Represents the yearly savings -/
def yearly_savings : ℝ := 24624

theorem yadav_expenditure :
  let remaining_after_consumables := monthly_salary * (1 - consumable_percentage)
  let clothes_transport_expenditure := remaining_after_consumables * clothes_transport_percentage
  let monthly_savings := yearly_savings / 12
  clothes_transport_expenditure = 2052 := by sorry

end NUMINAMATH_CALUDE_yadav_expenditure_l1346_134626


namespace NUMINAMATH_CALUDE_equal_cost_mileage_l1346_134606

/-- Represents the cost function for a truck rental company -/
structure RentalCompany where
  baseCost : ℝ
  costPerMile : ℝ

/-- Calculates the total cost for a given mileage -/
def totalCost (company : RentalCompany) (miles : ℝ) : ℝ :=
  company.baseCost + company.costPerMile * miles

/-- Theorem: The mileage at which all three companies have the same cost is 150 miles, 
    and this common cost is $85.45 -/
theorem equal_cost_mileage 
  (safety : RentalCompany)
  (city : RentalCompany)
  (metro : RentalCompany)
  (h1 : safety.baseCost = 41.95 ∧ safety.costPerMile = 0.29)
  (h2 : city.baseCost = 38.95 ∧ city.costPerMile = 0.31)
  (h3 : metro.baseCost = 44.95 ∧ metro.costPerMile = 0.27) :
  ∃ (m : ℝ), 
    m = 150 ∧ 
    totalCost safety m = totalCost city m ∧
    totalCost city m = totalCost metro m ∧
    totalCost safety m = 85.45 :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_mileage_l1346_134606


namespace NUMINAMATH_CALUDE_negation_of_cube_odd_l1346_134602

theorem negation_of_cube_odd (P : ℕ → Prop) :
  (¬ ∀ x : ℕ, Odd x → Odd (x^3)) ↔ (∃ x : ℕ, Odd x ∧ Even (x^3)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_cube_odd_l1346_134602


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l1346_134697

-- Define the function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define A as the sum of digits of 4444^4144
def A : ℕ := sumOfDigits (4444^4144)

-- Define B as the sum of digits of A
def B : ℕ := sumOfDigits A

-- Theorem to prove
theorem sum_of_digits_of_B_is_seven : sumOfDigits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l1346_134697


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1346_134657

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1346_134657


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l1346_134619

/-- The number of bacteria after a given number of doubling periods -/
def bacteria_count (initial_count : ℕ) (periods : ℕ) : ℕ :=
  initial_count * 2^periods

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ),
    bacteria_count initial_count 8 = 262144 ∧
    initial_count = 1024 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l1346_134619


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l1346_134630

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * (1 - 0.4) = 420 → initial_apples = 700 :=
by sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l1346_134630


namespace NUMINAMATH_CALUDE_power_of_six_with_nine_tens_digit_l1346_134622

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem power_of_six_with_nine_tens_digit :
  ∃ (k : ℕ), k > 0 ∧ tens_digit (6^k) = 9 ∧ ∀ (m : ℕ), m > 0 ∧ m < k → tens_digit (6^m) ≠ 9 :=
sorry

end NUMINAMATH_CALUDE_power_of_six_with_nine_tens_digit_l1346_134622
