import Mathlib

namespace NUMINAMATH_CALUDE_last_stage_less_than_2014_l343_34337

theorem last_stage_less_than_2014 :
  ∀ k : ℕ, k > 0 → (2 * k^2 - 2 * k + 1 < 2014) ↔ k ≤ 32 :=
by sorry

end NUMINAMATH_CALUDE_last_stage_less_than_2014_l343_34337


namespace NUMINAMATH_CALUDE_distance_between_points_l343_34327

/-- The distance between equidistant points A, B, and C, given specific travel conditions. -/
theorem distance_between_points (v_car v_train t : ℝ) (h1 : v_car = 80) (h2 : v_train = 50) (h3 : t = 7) :
  let S := v_car * t * (25800 / 210)
  S = 861 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l343_34327


namespace NUMINAMATH_CALUDE_sum_greater_product_iff_one_l343_34336

theorem sum_greater_product_iff_one (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b > a * b ↔ a = 1 ∨ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_product_iff_one_l343_34336


namespace NUMINAMATH_CALUDE_product_evaluation_l343_34383

theorem product_evaluation (m : ℤ) (h : m = 3) : 
  (m - 2) * (m - 1) * m * (m + 1) * (m + 2) * (m + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l343_34383


namespace NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l343_34344

theorem xy_zero_necessary_not_sufficient (x y : ℝ) :
  (∀ x, x = 0 → x * y = 0) ∧ 
  ¬(∀ x y, x * y = 0 → x = 0) :=
sorry

end NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l343_34344


namespace NUMINAMATH_CALUDE_cubic_factorization_l343_34314

theorem cubic_factorization (k : ℕ) (hk : k ≥ 2) :
  let n : ℕ := 16 * k^3 + 12 * k^2 + 3 * k - 126
  let factor1 : ℕ := n + 4 * k + 1
  let factor2 : ℕ := (n - 4 * k - 1)^2 + (4 * k + 1) * n
  (n^3 + 4 * n + 505 = factor1 * factor2) ∧
  (factor1 > Real.sqrt n) ∧
  (factor2 > Real.sqrt n) :=
by sorry

end NUMINAMATH_CALUDE_cubic_factorization_l343_34314


namespace NUMINAMATH_CALUDE_cube_color_theorem_l343_34352

theorem cube_color_theorem (n : ℕ) (h : n = 82) :
  ∀ (coloring : Fin n → Type),
    (∃ (cubes : Fin 10 → Fin n), (∀ i j, i ≠ j → coloring (cubes i) ≠ coloring (cubes j))) ∨
    (∃ (color : Type) (cubes : Fin 10 → Fin n), (∀ i, coloring (cubes i) = color)) :=
by sorry

end NUMINAMATH_CALUDE_cube_color_theorem_l343_34352


namespace NUMINAMATH_CALUDE_bijection_between_sets_l343_34386

def N (n : ℕ) : ℕ := n^9 % 10000

def set_greater (m : ℕ) : Set ℕ :=
  {n : ℕ | n < m ∧ n % 2 = 1 ∧ N n > n}

def set_lesser (m : ℕ) : Set ℕ :=
  {n : ℕ | n < m ∧ n % 2 = 1 ∧ N n < n}

theorem bijection_between_sets :
  ∃ (f : set_greater 10000 → set_lesser 10000),
    Function.Bijective f :=
  sorry

end NUMINAMATH_CALUDE_bijection_between_sets_l343_34386


namespace NUMINAMATH_CALUDE_extreme_point_inequality_l343_34321

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1) + (1/2) * x^2 - x

theorem extreme_point_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  0 < a → a < 1 →
  x₁ < x₂ →
  x₁ = -Real.sqrt (1 - a) →
  x₂ = Real.sqrt (1 - a) →
  f a x₁ < f a x₂ →
  f a x₂ > f a x₁ →
  f a x₂ / x₁ < 1/2 := by
sorry

end NUMINAMATH_CALUDE_extreme_point_inequality_l343_34321


namespace NUMINAMATH_CALUDE_absolute_value_equality_l343_34374

theorem absolute_value_equality (a b : ℝ) : 
  (∀ x y : ℝ, |a * x + b * y| + |b * x + a * y| = |x| + |y|) ↔ 
  ((a = 1 ∧ b = 0) ∨ (a = 0 ∧ b = 1) ∨ (a = 0 ∧ b = -1) ∨ (a = -1 ∧ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l343_34374


namespace NUMINAMATH_CALUDE_sequence_problem_l343_34319

theorem sequence_problem : ∃ (x y : ℝ), 
  (x^2 = 2*y) ∧ (2*y = x + 20) ∧ ((x + y = 4) ∨ (x + y = 35/2)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l343_34319


namespace NUMINAMATH_CALUDE_average_of_numbers_l343_34396

def numbers : List ℝ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140]

theorem average_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 125397.5 ∧
  (numbers.sum / numbers.length : ℝ) ≠ 858.5454545454545 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l343_34396


namespace NUMINAMATH_CALUDE_shortest_path_in_room_l343_34372

theorem shortest_path_in_room (a b h : ℝ) 
  (ha : a = 7) (hb : b = 8) (hh : h = 4) : 
  let diagonal := Real.sqrt (a^2 + b^2 + h^2)
  let floor_path := Real.sqrt ((a^2 + b^2) + h^2)
  diagonal ≥ floor_path ∧ floor_path = Real.sqrt 265 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_in_room_l343_34372


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l343_34325

theorem cubic_equation_solutions (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l343_34325


namespace NUMINAMATH_CALUDE_problem_solution_l343_34301

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem problem_solution :
  (∀ x : ℝ, f 2 x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) ∧
  (∀ x : ℝ, f 2 x + f 2 (x + 5) ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l343_34301


namespace NUMINAMATH_CALUDE_restaurant_group_composition_l343_34395

theorem restaurant_group_composition (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) : 
  total_people = 11 → 
  adult_meal_cost = 8 → 
  total_cost = 72 → 
  ∃ (num_adults num_kids : ℕ), 
    num_adults + num_kids = total_people ∧ 
    num_adults * adult_meal_cost = total_cost ∧ 
    num_kids = 2 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_group_composition_l343_34395


namespace NUMINAMATH_CALUDE_least_crayons_l343_34304

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem least_crayons (n : ℕ) : 
  (is_divisible_by n 3 ∧ 
   is_divisible_by n 4 ∧ 
   is_divisible_by n 5 ∧ 
   is_divisible_by n 7 ∧ 
   is_divisible_by n 8) →
  (∀ m : ℕ, m < n → 
    ¬(is_divisible_by m 3 ∧ 
      is_divisible_by m 4 ∧ 
      is_divisible_by m 5 ∧ 
      is_divisible_by m 7 ∧ 
      is_divisible_by m 8)) →
  n = 840 := by
sorry

end NUMINAMATH_CALUDE_least_crayons_l343_34304


namespace NUMINAMATH_CALUDE_tangent_line_equation_l343_34355

/-- The equation of the line tangent to a circle at two points, which also passes through a given point -/
theorem tangent_line_equation (x y : ℝ → ℝ) :
  -- Given circle equation
  (∀ t, x t ^ 2 + (y t - 2) ^ 2 = 4) →
  -- Circle passes through (-2, 6)
  (∃ t, x t = -2 ∧ y t = 6) →
  -- Line equation
  (∃ a b c : ℝ, ∀ t, a * x t + b * y t + c = 0) →
  -- The line equation is x - 2y + 6 = 0
  (∃ t₁ t₂, t₁ ≠ t₂ ∧ x t₁ - 2 * y t₁ + 6 = 0 ∧ x t₂ - 2 * y t₂ + 6 = 0) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l343_34355


namespace NUMINAMATH_CALUDE_power_function_difference_l343_34387

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_difference (f : ℝ → ℝ) :
  isPowerFunction f → f 9 = 3 → f 2 - f 1 = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_difference_l343_34387


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l343_34366

/-- Theorem: Cylinder Volume Change
  Given a cylinder with an initial volume of 20 cubic feet,
  if its radius is tripled and its height is doubled,
  then its new volume will be 360 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 20 → π * (3*r)^2 * (2*h) = 360 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l343_34366


namespace NUMINAMATH_CALUDE_janes_current_age_l343_34306

theorem janes_current_age :
  let min_age : ℕ := 25
  let years_until_dara_eligible : ℕ := 14
  let years_until_half_age : ℕ := 6
  let dara_current_age : ℕ := min_age - years_until_dara_eligible
  ∀ jane_age : ℕ,
    (dara_current_age + years_until_half_age = (jane_age + years_until_half_age) / 2) →
    jane_age = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_janes_current_age_l343_34306


namespace NUMINAMATH_CALUDE_probability_divisible_by_three_l343_34385

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 15}

def is_divisible_by_three (x y z : ℕ) : Prop :=
  (x * y * z - x * y - y * z - z * x + x + y + z) % 3 = 0

def favorable_outcomes : ℕ := 60

def total_outcomes : ℕ := Nat.choose 15 3

theorem probability_divisible_by_three :
  (favorable_outcomes : ℚ) / total_outcomes = 12 / 91 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_three_l343_34385


namespace NUMINAMATH_CALUDE_tile_arrangements_l343_34398

/-- The number of distinguishable arrangements of tiles -/
def num_arrangements (orange purple blue red : ℕ) : ℕ :=
  Nat.factorial (orange + purple + blue + red) /
  (Nat.factorial orange * Nat.factorial purple * Nat.factorial blue * Nat.factorial red)

/-- Theorem stating that the number of distinguishable arrangements
    of 2 orange, 1 purple, 3 blue, and 2 red tiles is 1680 -/
theorem tile_arrangements :
  num_arrangements 2 1 3 2 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l343_34398


namespace NUMINAMATH_CALUDE_profit_percentage_10_12_l343_34381

/-- Calculates the profit percentage when selling n articles at the cost price of m articles -/
def profit_percentage (n m : ℕ) : ℚ :=
  ((m : ℚ) - (n : ℚ)) / (n : ℚ) * 100

/-- Theorem: The profit percentage when selling 10 articles at the cost price of 12 articles is 20% -/
theorem profit_percentage_10_12 : profit_percentage 10 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_10_12_l343_34381


namespace NUMINAMATH_CALUDE_least_integer_sum_l343_34318

theorem least_integer_sum (x y z : ℕ+) (h : (2 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val) :
  (∃ (n : ℤ), x.val + n + z.val = 26 ∧ ∀ (m : ℤ), x.val + m + z.val = 26 → n ≤ m) →
  (∃ (n : ℤ), x.val + n + z.val = 26 ∧ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_least_integer_sum_l343_34318


namespace NUMINAMATH_CALUDE_square_root_problem_l343_34342

theorem square_root_problem (m a b c n : ℝ) (hm : m > 0) :
  (Real.sqrt m = 2*n + 1 ∧ Real.sqrt m = 4 - 3*n) →
  (|a - 1| + Real.sqrt b + (c - n)^2 = 0) →
  (m = 121 ∨ m = 121/25) ∧ Real.sqrt (a + b + c) = Real.sqrt 6 ∨ Real.sqrt (a + b + c) = -Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l343_34342


namespace NUMINAMATH_CALUDE_article_price_l343_34375

theorem article_price (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  final_price = 126 ∧ discount1 = 0.1 ∧ discount2 = 0.2 →
  ∃ (original_price : ℝ), original_price = 175 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry

end NUMINAMATH_CALUDE_article_price_l343_34375


namespace NUMINAMATH_CALUDE_exists_non_increasing_log_l343_34335

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem exists_non_increasing_log :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ¬(∀ (x y : ℝ), x > y → log a x > log a y) :=
by sorry

end NUMINAMATH_CALUDE_exists_non_increasing_log_l343_34335


namespace NUMINAMATH_CALUDE_downstream_speed_l343_34302

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- Theorem stating the relationship between upstream, stillwater, and downstream speeds -/
theorem downstream_speed (s : RowingSpeed) 
  (h1 : s.upstream = 15) 
  (h2 : s.stillWater = 25) : 
  s.downstream = 35 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_l343_34302


namespace NUMINAMATH_CALUDE_simplify_expression_l343_34333

theorem simplify_expression (a b : ℝ) : (1 : ℝ) * (2 * a) * (3 * a^2 * b) * (4 * a^3 * b^2) * (5 * a^4 * b^3) = 120 * a^10 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l343_34333


namespace NUMINAMATH_CALUDE_jar_water_problem_l343_34388

theorem jar_water_problem (s l : ℝ) (hs : s > 0) (hl : l > 0) : 
  (1/8 : ℝ) * s = (1/6 : ℝ) * l → (1/6 : ℝ) * l + (1/8 : ℝ) * s = (1/3 : ℝ) * l :=
by sorry

end NUMINAMATH_CALUDE_jar_water_problem_l343_34388


namespace NUMINAMATH_CALUDE_product_of_ab_values_l343_34311

theorem product_of_ab_values (a b : ℝ) (h1 : a + 1/b = 4) (h2 : 1/a + b = 16/15) : 
  (5/3 * 3/5 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_product_of_ab_values_l343_34311


namespace NUMINAMATH_CALUDE_arrangement_count_is_factorial_squared_l343_34347

/-- The number of ways to arrange 5 different objects in a 5x5 grid,
    such that each row and each column contains exactly one object. -/
def arrangement_count : ℕ := (5 : ℕ).factorial ^ 2

/-- Theorem stating that the number of arrangements is equal to (5!)^2 -/
theorem arrangement_count_is_factorial_squared :
  arrangement_count = 14400 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_factorial_squared_l343_34347


namespace NUMINAMATH_CALUDE_timothy_total_cost_l343_34338

/-- The total cost of Timothy's purchases -/
def total_cost (land_acres : ℕ) (land_price_per_acre : ℕ) 
               (house_price : ℕ) 
               (cow_count : ℕ) (cow_price : ℕ) 
               (chicken_count : ℕ) (chicken_price : ℕ) 
               (solar_install_hours : ℕ) (solar_install_price_per_hour : ℕ) 
               (solar_equipment_price : ℕ) : ℕ :=
  land_acres * land_price_per_acre + 
  house_price + 
  cow_count * cow_price + 
  chicken_count * chicken_price + 
  solar_install_hours * solar_install_price_per_hour + 
  solar_equipment_price

/-- Theorem stating that Timothy's total cost is $147,700 -/
theorem timothy_total_cost : 
  total_cost 30 20 120000 20 1000 100 5 6 100 6000 = 147700 := by
  sorry

end NUMINAMATH_CALUDE_timothy_total_cost_l343_34338


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l343_34378

/-- Given a right triangle with a point on its hypotenuse and lines drawn parallel to the legs,
    dividing it into a rectangle and two smaller right triangles, this theorem states the
    relationship between the areas of the smaller triangles and the rectangle. -/
theorem triangle_area_ratio (a b m : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_m : m > 0) :
  let rectangle_area := a * b
  let small_triangle1_area := m * rectangle_area
  let small_triangle2_area := (b^2) / (4 * m)
  (small_triangle2_area / rectangle_area) = b / (4 * m * a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l343_34378


namespace NUMINAMATH_CALUDE_correct_match_probability_l343_34368

theorem correct_match_probability (n : Nat) (h : n = 4) :
  (1 : ℚ) / n.factorial = (1 : ℚ) / 24 := by
  sorry

#check correct_match_probability

end NUMINAMATH_CALUDE_correct_match_probability_l343_34368


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l343_34360

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : has_period f 5)
  (h1 : f 1 = 1)
  (h2 : f 2 = 2) :
  f 3 - f 4 = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l343_34360


namespace NUMINAMATH_CALUDE_system_solution_l343_34373

theorem system_solution :
  let f (x y z : ℝ) := x^2 = 2 * Real.sqrt (y^2 + 1) ∧
                       y^2 = 2 * Real.sqrt (z^2 - 1) - 2 ∧
                       z^2 = 4 * Real.sqrt (x^2 + 2) - 6
  (∀ x y z : ℝ, f x y z ↔ 
    ((x = Real.sqrt 2 ∧ y = 0 ∧ z = Real.sqrt 2) ∨
     (x = Real.sqrt 2 ∧ y = 0 ∧ z = -Real.sqrt 2) ∨
     (x = -Real.sqrt 2 ∧ y = 0 ∧ z = Real.sqrt 2) ∨
     (x = -Real.sqrt 2 ∧ y = 0 ∧ z = -Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l343_34373


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l343_34393

theorem problem_1 (m n : ℝ) : 
  (∀ x, (x - 3) * (x - 4) = x^2 + m*x + n) → m = -7 ∧ n = 12 := by sorry

theorem problem_2 (a b : ℝ) :
  (∀ x, (x + a) * (x + b) = x^2 - 3*x + 1/3) → 
  ((a - 1) * (b - 1) = 13/3) ∧ (1/a^2 + 1/b^2 = 75) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l343_34393


namespace NUMINAMATH_CALUDE_concert_attendance_l343_34389

/-- Represents the number of adults attending the concert. -/
def num_adults : ℕ := sorry

/-- Represents the number of children attending the concert. -/
def num_children : ℕ := sorry

/-- The cost of an adult ticket in dollars. -/
def adult_ticket_cost : ℕ := 7

/-- The cost of a child ticket in dollars. -/
def child_ticket_cost : ℕ := 3

/-- The total revenue from ticket sales in dollars. -/
def total_revenue : ℕ := 6000

theorem concert_attendance :
  (num_children = 3 * num_adults) ∧
  (num_adults * adult_ticket_cost + num_children * child_ticket_cost = total_revenue) →
  (num_adults + num_children = 1500) := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l343_34389


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisibility_l343_34357

theorem product_of_five_consecutive_integers_divisibility 
  (m : ℤ) 
  (k : ℤ) 
  (h1 : m = k * (k + 1) * (k + 2) * (k + 3) * (k + 4)) 
  (h2 : 11 ∣ m) : 
  (10 ∣ m) ∧ (22 ∣ m) ∧ (33 ∣ m) ∧ (55 ∣ m) ∧ ¬(∀ m, 66 ∣ m) :=
by sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisibility_l343_34357


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l343_34310

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l343_34310


namespace NUMINAMATH_CALUDE_polynomial_degree_three_l343_34346

def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 5*x^3 + 7*x^4
def g (x : ℝ) : ℝ := 4 - 3*x - 8*x^3 + 12*x^4

def c : ℚ := -7/12

theorem polynomial_degree_three :
  ∃ (a b d : ℝ), ∀ (x : ℝ),
    f x + c * g x = a*x^3 + b*x^2 + d*x + (2 + 4*c) ∧ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_three_l343_34346


namespace NUMINAMATH_CALUDE_roses_cut_l343_34350

/-- The number of roses Jessica cut is equal to the difference between the final number of roses in the vase and the initial number of roses in the vase. -/
theorem roses_cut (initial_roses final_roses : ℕ) (h : initial_roses = 2 ∧ final_roses = 23) :
  final_roses - initial_roses = 21 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l343_34350


namespace NUMINAMATH_CALUDE_absolute_value_equality_l343_34376

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x - 5| → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l343_34376


namespace NUMINAMATH_CALUDE_x_percent_of_x_equals_nine_l343_34345

theorem x_percent_of_x_equals_nine (x : ℝ) : 
  x > 0 → (x / 100) * x = 9 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_x_percent_of_x_equals_nine_l343_34345


namespace NUMINAMATH_CALUDE_seashells_total_l343_34394

theorem seashells_total (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_seashells_total_l343_34394


namespace NUMINAMATH_CALUDE_sqrt_difference_complex_expression_system_of_equations_l343_34367

-- Problem 1
theorem sqrt_difference : Real.sqrt 8 - Real.sqrt 50 = -3 * Real.sqrt 2 := by sorry

-- Problem 2
theorem complex_expression : 
  Real.sqrt 27 * Real.sqrt (1/3) - (Real.sqrt 3 - Real.sqrt 2)^2 = 2 * Real.sqrt 6 - 2 := by sorry

-- Problem 3
theorem system_of_equations :
  ∃ (x y : ℝ), x + y = 2 ∧ x + 2*y = 6 ∧ x = -2 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_difference_complex_expression_system_of_equations_l343_34367


namespace NUMINAMATH_CALUDE_incorrect_statement_identification_l343_34377

theorem incorrect_statement_identification :
  ((-64 : ℚ)^(1/3) = -4) ∧ 
  ((49 : ℚ)^(1/2) = 7) ∧ 
  ((1/27 : ℚ)^(1/3) = 1/3) →
  ¬((1/16 : ℚ)^(1/2) = 1/4) :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_identification_l343_34377


namespace NUMINAMATH_CALUDE_slope_of_line_from_equation_l343_34349

theorem slope_of_line_from_equation (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : x₁ ≠ x₂) 
  (h₂ : 4 / x₁ + 5 / y₁ = 0) 
  (h₃ : 4 / x₂ + 5 / y₂ = 0) : 
  (y₂ - y₁) / (x₂ - x₁) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_from_equation_l343_34349


namespace NUMINAMATH_CALUDE_derivative_of_even_function_is_odd_l343_34370

/-- A function f: ℝ → ℝ that is even, i.e., f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The derivative of a function f: ℝ → ℝ -/
def DerivativeOf (g f : ℝ → ℝ) : Prop :=
  ∀ x, HasDerivAt f (g x) x

theorem derivative_of_even_function_is_odd
  (f g : ℝ → ℝ) (hf : EvenFunction f) (hg : DerivativeOf g f) :
  ∀ x, g (-x) = -g x :=
sorry

end NUMINAMATH_CALUDE_derivative_of_even_function_is_odd_l343_34370


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_proof_l343_34300

/-- The smallest four-digit number divisible by 2, 3, 8, and 9 -/
def smallest_four_digit_divisible : ℕ := 1008

/-- Predicate to check if a number is four digits -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_proof :
  is_four_digit smallest_four_digit_divisible ∧
  smallest_four_digit_divisible % 2 = 0 ∧
  smallest_four_digit_divisible % 3 = 0 ∧
  smallest_four_digit_divisible % 8 = 0 ∧
  smallest_four_digit_divisible % 9 = 0 ∧
  ∀ n : ℕ, is_four_digit n →
    n % 2 = 0 → n % 3 = 0 → n % 8 = 0 → n % 9 = 0 →
    n ≥ smallest_four_digit_divisible :=
by sorry

#eval smallest_four_digit_divisible

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_proof_l343_34300


namespace NUMINAMATH_CALUDE_fox_can_equalize_cheese_l343_34384

/-- Represents the state of the cheese pieces -/
structure CheeseState where
  piece1 : ℕ
  piece2 : ℕ
  piece3 : ℕ

/-- Represents a single cut operation by the fox -/
inductive CutOperation
  | cut12 : CutOperation  -- Cut 1g from piece1 and piece2
  | cut13 : CutOperation  -- Cut 1g from piece1 and piece3
  | cut23 : CutOperation  -- Cut 1g from piece2 and piece3

/-- Applies a single cut operation to a cheese state -/
def applyCut (state : CheeseState) (cut : CutOperation) : CheeseState :=
  match cut with
  | CutOperation.cut12 => ⟨state.piece1 - 1, state.piece2 - 1, state.piece3⟩
  | CutOperation.cut13 => ⟨state.piece1 - 1, state.piece2, state.piece3 - 1⟩
  | CutOperation.cut23 => ⟨state.piece1, state.piece2 - 1, state.piece3 - 1⟩

/-- Applies a sequence of cut operations to a cheese state -/
def applyCuts (state : CheeseState) (cuts : List CutOperation) : CheeseState :=
  cuts.foldl applyCut state

/-- The theorem to be proved -/
theorem fox_can_equalize_cheese :
  ∃ (cuts : List CutOperation),
    let finalState := applyCuts ⟨5, 8, 11⟩ cuts
    finalState.piece1 = finalState.piece2 ∧
    finalState.piece2 = finalState.piece3 ∧
    finalState.piece1 > 0 :=
  sorry


end NUMINAMATH_CALUDE_fox_can_equalize_cheese_l343_34384


namespace NUMINAMATH_CALUDE_danny_collection_difference_l343_34322

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 11

/-- The number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 28

/-- The difference between wrappers and bottle caps found at the park -/
def difference : ℕ := wrappers_found - bottle_caps_found

theorem danny_collection_difference : difference = 17 := by
  sorry

end NUMINAMATH_CALUDE_danny_collection_difference_l343_34322


namespace NUMINAMATH_CALUDE_reciprocal_of_complex_l343_34380

/-- Given a complex number z = -1 + √3i, prove that its reciprocal is -1/4 - (√3/4)i -/
theorem reciprocal_of_complex (z : ℂ) : 
  z = -1 + Complex.I * Real.sqrt 3 → 
  z⁻¹ = -(1/4 : ℂ) - Complex.I * ((Real.sqrt 3)/4) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_complex_l343_34380


namespace NUMINAMATH_CALUDE_simplify_fraction_l343_34354

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  (12 * x^2 * y^3) / (9 * x * y^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l343_34354


namespace NUMINAMATH_CALUDE_tank_fill_problem_l343_34313

theorem tank_fill_problem (tank_capacity : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  tank_capacity = 48 →
  added_amount = 8 →
  final_fraction = 9/10 →
  (tank_capacity * final_fraction - added_amount) / tank_capacity = 4/10 :=
by sorry

end NUMINAMATH_CALUDE_tank_fill_problem_l343_34313


namespace NUMINAMATH_CALUDE_books_from_second_shop_l343_34361

/-- Proves the number of books bought from the second shop given the conditions -/
theorem books_from_second_shop 
  (first_shop_books : ℕ) 
  (first_shop_cost : ℕ) 
  (second_shop_cost : ℕ) 
  (average_price : ℚ) 
  (h1 : first_shop_books = 65)
  (h2 : first_shop_cost = 1150)
  (h3 : second_shop_cost = 920)
  (h4 : average_price = 18) : 
  ℕ := by
  sorry

#check books_from_second_shop

end NUMINAMATH_CALUDE_books_from_second_shop_l343_34361


namespace NUMINAMATH_CALUDE_prime_divisibility_l343_34392

theorem prime_divisibility (a b p q : ℕ) : 
  a > 0 → b > 0 → Prime p → Prime q → 
  ¬(p ∣ q - 1) → (q ∣ a^p - b^p) → (q ∣ a - b) :=
by sorry

end NUMINAMATH_CALUDE_prime_divisibility_l343_34392


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l343_34307

theorem salt_solution_mixture (x : ℝ) : 
  (0.6 * x = 0.1 * (x + 1)) → x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l343_34307


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l343_34358

theorem sqrt_expression_equality : 3 * Real.sqrt 12 / (3 * Real.sqrt (1/3)) - 2 * Real.sqrt 3 = 6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l343_34358


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l343_34331

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧
  a * b * c * d * e = 15120 →
  e = 9 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l343_34331


namespace NUMINAMATH_CALUDE_perfect_square_equation_l343_34363

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_equation_l343_34363


namespace NUMINAMATH_CALUDE_company_workforce_l343_34365

/-- Proves the number of employees after hiring, given initial conditions and hiring information -/
theorem company_workforce (initial_female_percentage : ℚ) 
                          (final_female_percentage : ℚ)
                          (additional_male_workers : ℕ) : ℕ :=
  let initial_female_percentage : ℚ := 60 / 100
  let final_female_percentage : ℚ := 55 / 100
  let additional_male_workers : ℕ := 30
  360

#check company_workforce

end NUMINAMATH_CALUDE_company_workforce_l343_34365


namespace NUMINAMATH_CALUDE_inscribable_iff_equal_sums_l343_34340

/-- A convex polyhedral angle -/
structure ConvexPolyhedralAngle where
  -- Add necessary fields

/-- The property of being inscribable in a cone -/
def isInscribableInCone (angle : ConvexPolyhedralAngle) : Prop :=
  sorry

/-- The property of having equal sums of opposite dihedral angles -/
def hasEqualSumsOfOppositeDihedralAngles (angle : ConvexPolyhedralAngle) : Prop :=
  sorry

/-- Theorem: A convex polyhedral angle can be inscribed in a cone if and only if 
    the sums of its opposite dihedral angles are equal -/
theorem inscribable_iff_equal_sums 
  (angle : ConvexPolyhedralAngle) : 
  isInscribableInCone angle ↔ hasEqualSumsOfOppositeDihedralAngles angle :=
sorry

end NUMINAMATH_CALUDE_inscribable_iff_equal_sums_l343_34340


namespace NUMINAMATH_CALUDE_bread_pieces_in_pond_l343_34334

theorem bread_pieces_in_pond :
  ∀ (total : ℕ),
    (∃ (duck1 duck2 duck3 : ℕ),
      duck1 = total / 2 ∧
      duck2 = 13 ∧
      duck3 = 7 ∧
      duck1 + duck2 + duck3 + 30 = total) →
    total = 100 := by
sorry

end NUMINAMATH_CALUDE_bread_pieces_in_pond_l343_34334


namespace NUMINAMATH_CALUDE_second_side_bisected_l343_34353

/-- A nonagon circumscribed around a circle -/
structure CircumscribedNonagon where
  /-- The lengths of the sides of the nonagon -/
  sides : Fin 9 → ℕ
  /-- All sides have positive integer lengths -/
  all_positive : ∀ i, sides i > 0
  /-- The first and third sides have length 1 -/
  first_third_one : sides 0 = 1 ∧ sides 2 = 1

/-- The point of tangency divides the second side into two equal segments -/
theorem second_side_bisected (n : CircumscribedNonagon) :
  ∃ (x : ℚ), x = 1/2 ∧ x * n.sides 1 = (1 - x) * n.sides 1 :=
sorry

end NUMINAMATH_CALUDE_second_side_bisected_l343_34353


namespace NUMINAMATH_CALUDE_point_Q_in_second_quadrant_l343_34391

/-- Given point P(0,a) on the negative half-axis of the y-axis, 
    point Q(-a^2-1, -a+1) lies in the second quadrant. -/
theorem point_Q_in_second_quadrant (a : ℝ) 
  (h_a_neg : a < 0) : 
  let P : ℝ × ℝ := (0, a)
  let Q : ℝ × ℝ := (-a^2 - 1, -a + 1)
  (-a^2 - 1 < 0) ∧ (-a + 1 > 0) := by
sorry

end NUMINAMATH_CALUDE_point_Q_in_second_quadrant_l343_34391


namespace NUMINAMATH_CALUDE_marble_collection_weight_l343_34330

/-- The weight of Courtney's marble collection -/
def total_weight (jar1_count : ℕ) (jar1_weight : ℚ) (jar2_weight : ℚ) (jar3_weight : ℚ) (jar4_weight : ℚ) : ℚ :=
  let jar2_count := 2 * jar1_count
  let jar3_count := (1 : ℚ) / 4 * jar1_count
  let jar4_count := (3 : ℚ) / 5 * jar2_count
  jar1_count * jar1_weight + jar2_count * jar2_weight + jar3_count * jar3_weight + jar4_count * jar4_weight

/-- Theorem stating the total weight of Courtney's marble collection -/
theorem marble_collection_weight :
  total_weight 80 (35 / 100) (45 / 100) (25 / 100) (55 / 100) = 1578 / 10 := by
  sorry

end NUMINAMATH_CALUDE_marble_collection_weight_l343_34330


namespace NUMINAMATH_CALUDE_unique_valid_number_l343_34356

def is_valid_number (n : Fin 10 → Nat) : Prop :=
  (∀ i : Fin 8, n i * n (i + 1) * n (i + 2) = 24) ∧
  n 4 = 2 ∧
  n 8 = 3

theorem unique_valid_number :
  ∃! n : Fin 10 → Nat, is_valid_number n ∧ 
    (∀ i : Fin 10, n i = ([4, 2, 3, 4, 2, 3, 4, 2, 3, 4] : List Nat)[i]) :=
by sorry

end NUMINAMATH_CALUDE_unique_valid_number_l343_34356


namespace NUMINAMATH_CALUDE_min_sum_squares_l343_34326

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l343_34326


namespace NUMINAMATH_CALUDE_fiftieth_term_is_448_l343_34320

/-- Checks if a natural number contains the digit 4 --/
def containsDigitFour (n : ℕ) : Bool :=
  n.repr.any (· = '4')

/-- The sequence of positive multiples of 4 that contain at least one digit 4 --/
def specialSequence : ℕ → ℕ
  | 0 => 4  -- The first term is always 4
  | n + 1 => 
      let next := specialSequence n + 4
      if containsDigitFour next then next
      else specialSequence (n + 1)

/-- The 50th term of the special sequence is 448 --/
theorem fiftieth_term_is_448 : specialSequence 49 = 448 := by
  sorry

#eval specialSequence 49  -- This should output 448

end NUMINAMATH_CALUDE_fiftieth_term_is_448_l343_34320


namespace NUMINAMATH_CALUDE_cross_ratio_equality_l343_34351

theorem cross_ratio_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cross_ratio_equality_l343_34351


namespace NUMINAMATH_CALUDE_work_completion_time_l343_34328

theorem work_completion_time (D_A : ℝ) 
  (h1 : D_A > 0)
  (h2 : 1 / D_A + 2 / D_A = 1 / 4) : 
  D_A = 12 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l343_34328


namespace NUMINAMATH_CALUDE_length_to_width_ratio_l343_34379

def field_perimeter : ℝ := 384
def field_width : ℝ := 80

theorem length_to_width_ratio :
  let field_length := (field_perimeter - 2 * field_width) / 2
  field_length / field_width = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_length_to_width_ratio_l343_34379


namespace NUMINAMATH_CALUDE_reciprocal_problem_l343_34312

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 3) : 200 / x = 1600 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l343_34312


namespace NUMINAMATH_CALUDE_parametric_equations_represent_line_l343_34329

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y + 1 = 0

/-- The parametric equations -/
def parametric_x (t : ℝ) : ℝ := 1 - 4 * t
def parametric_y (t : ℝ) : ℝ := -1 + 3 * t

/-- Theorem stating that the parametric equations represent the line -/
theorem parametric_equations_represent_line :
  ∀ t : ℝ, line_equation (parametric_x t) (parametric_y t) :=
by
  sorry

end NUMINAMATH_CALUDE_parametric_equations_represent_line_l343_34329


namespace NUMINAMATH_CALUDE_stuffed_animals_gcd_l343_34369

theorem stuffed_animals_gcd : Nat.gcd 26 (Nat.gcd 14 (Nat.gcd 18 22)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_gcd_l343_34369


namespace NUMINAMATH_CALUDE_arithmetic_equation_l343_34309

theorem arithmetic_equation : 3 * 13 + 3 * 14 + 3 * 17 + 11 = 143 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l343_34309


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_l343_34371

theorem polynomial_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∃ (x : ℤ), x^3 - 2023*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 102 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_l343_34371


namespace NUMINAMATH_CALUDE_system_solution_l343_34390

theorem system_solution (x y z : ℚ) : 
  (x * y = 5 * (x + y) ∧ 
   x * z = 4 * (x + z) ∧ 
   y * z = 2 * (y + z)) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = -40 ∧ y = 40/9 ∧ z = 40/11)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l343_34390


namespace NUMINAMATH_CALUDE_juniors_in_sports_l343_34305

def total_students : ℕ := 500
def junior_percentage : ℚ := 40 / 100
def sports_percentage : ℚ := 70 / 100

theorem juniors_in_sports :
  (total_students : ℚ) * junior_percentage * sports_percentage = 140 := by
  sorry

end NUMINAMATH_CALUDE_juniors_in_sports_l343_34305


namespace NUMINAMATH_CALUDE_inverse_of_A_l343_34397

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, 3; -1, 7]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![7/17, -3/17; 1/17, 2/17]
  A * A_inv = 1 ∧ A_inv * A = 1 :=
sorry

end NUMINAMATH_CALUDE_inverse_of_A_l343_34397


namespace NUMINAMATH_CALUDE_problem_solution_l343_34324

theorem problem_solution (m n c d a : ℝ) 
  (h1 : m + n = 0)  -- m and n are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : a = ⌊Real.sqrt 5⌋) -- a is the integer part of √5
  : Real.sqrt (c * d) + 2 * (m + n) - a = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l343_34324


namespace NUMINAMATH_CALUDE_fraction_equality_l343_34359

theorem fraction_equality (a b : ℝ) : |a^2 - b^2| / |(a - b)^2| = |a + b| / |a - b| := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l343_34359


namespace NUMINAMATH_CALUDE_wendi_chicken_count_l343_34362

/-- The number of chickens Wendi has after a series of events -/
def final_chicken_count (initial : ℕ) (doubled : ℕ) (lost : ℕ) (found : ℕ) : ℕ :=
  initial + doubled - lost + found

/-- Theorem stating the final number of chickens Wendi has -/
theorem wendi_chicken_count : 
  final_chicken_count 4 4 1 6 = 13 := by sorry

end NUMINAMATH_CALUDE_wendi_chicken_count_l343_34362


namespace NUMINAMATH_CALUDE_union_cardinality_l343_34339

def A : Finset ℕ := {1, 3, 5}
def B : Finset ℕ := {2, 3}

theorem union_cardinality : Finset.card (A ∪ B) = 4 := by
  sorry

end NUMINAMATH_CALUDE_union_cardinality_l343_34339


namespace NUMINAMATH_CALUDE_winnie_balloons_l343_34399

/-- The number of balloons Winnie keeps for herself when distributing balloons among friends -/
theorem winnie_balloons (total_balloons : ℕ) (num_friends : ℕ) (h1 : total_balloons = 226) (h2 : num_friends = 11) :
  total_balloons % num_friends = 6 := by
  sorry

end NUMINAMATH_CALUDE_winnie_balloons_l343_34399


namespace NUMINAMATH_CALUDE_circle_trajectory_l343_34323

-- Define the circle equation as a function of m, x, and y
def circle_equation (m x y : ℝ) : Prop :=
  x^2 + y^2 - (4*m + 2)*x - 2*m*y + 4*m^2 + 4*m + 1 = 0

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  x - 2*y - 1 = 0

-- State the theorem
theorem circle_trajectory :
  ∀ m x y : ℝ, x ≠ 1 →
  (∃ m, circle_equation m x y) ↔ trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_l343_34323


namespace NUMINAMATH_CALUDE_brendas_age_l343_34343

/-- Given the ages of Addison, Brenda, and Janet, prove that Brenda's age is 8/3 years. -/
theorem brendas_age (A B J : ℚ) 
  (h1 : A = 4 * B)  -- Addison's age is four times Brenda's age
  (h2 : J = B + 8)  -- Janet is eight years older than Brenda
  (h3 : A = J)      -- Addison and Janet are twins (same age)
  : B = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brendas_age_l343_34343


namespace NUMINAMATH_CALUDE_opinion_change_range_l343_34315

def initial_yes : ℝ := 40
def initial_no : ℝ := 60
def final_yes : ℝ := 60
def final_no : ℝ := 40

theorem opinion_change_range :
  let min_change := |final_yes - initial_yes|
  let max_change := min initial_yes initial_no + min final_yes final_no
  max_change - min_change = 40 := by sorry

end NUMINAMATH_CALUDE_opinion_change_range_l343_34315


namespace NUMINAMATH_CALUDE_largest_decimal_l343_34308

theorem largest_decimal : ∀ (a b c d e : ℝ), 
  a = 0.989 → b = 0.998 → c = 0.981 → d = 0.899 → e = 0.9801 →
  (b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e) := by
  sorry

end NUMINAMATH_CALUDE_largest_decimal_l343_34308


namespace NUMINAMATH_CALUDE_max_value_theorem_l343_34332

theorem max_value_theorem (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a^2 + b^2/2 = 1) :
  ∃ (M : ℝ), M = (3 * Real.sqrt 2) / 4 ∧ a * Real.sqrt (1 + b^2) ≤ M ∧
  ∃ (a₀ b₀ : ℝ), a₀ * Real.sqrt (1 + b₀^2) = M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l343_34332


namespace NUMINAMATH_CALUDE_specific_arrangement_probability_l343_34364

def total_lamps : ℕ := 8
def red_lamps : ℕ := 4
def blue_lamps : ℕ := 4
def lamps_on : ℕ := 4

def ways_to_arrange_colors : ℕ := Nat.choose total_lamps red_lamps
def ways_to_turn_on : ℕ := Nat.choose total_lamps lamps_on

def remaining_positions : ℕ := 5
def remaining_red : ℕ := 3
def remaining_blue : ℕ := 2
def remaining_on : ℕ := 2

def ways_to_arrange_remaining : ℕ := Nat.choose remaining_positions remaining_red
def ways_to_turn_on_remaining : ℕ := Nat.choose remaining_positions remaining_on

theorem specific_arrangement_probability :
  (ways_to_arrange_remaining * ways_to_turn_on_remaining : ℚ) / 
  (ways_to_arrange_colors * ways_to_turn_on) = 1 / 49 := by
  sorry

end NUMINAMATH_CALUDE_specific_arrangement_probability_l343_34364


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_correct_l343_34303

/-- The area of a quadrilateral with vertices at (2, 2), (2, -1), (3, -1), and (2007, 2008) -/
def quadrilateralArea : ℝ := 2008006.5

/-- The vertices of the quadrilateral -/
def vertices : List (ℝ × ℝ) := [(2, 2), (2, -1), (3, -1), (2007, 2008)]

/-- Theorem stating that the area of the quadrilateral with the given vertices is 2008006.5 -/
theorem quadrilateral_area_is_correct : 
  let area := quadrilateralArea
  ∃ (f : List (ℝ × ℝ) → ℝ), f vertices = area :=
by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_is_correct_l343_34303


namespace NUMINAMATH_CALUDE_coin_value_equality_l343_34382

theorem coin_value_equality (m : ℕ) : 
  (25 : ℕ) * 25 + 15 * 10 = m * 25 + 40 * 10 → m = 15 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_equality_l343_34382


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l343_34316

/-- 
Given an arithmetic sequence {a_n} with common difference -2,
if a_1, a_4, and a_5 form a geometric sequence, then a_3 = 5.
-/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n - 2) →  -- arithmetic sequence with common difference -2
  (a 4)^2 = a 1 * a 5 →         -- a_1, a_4, a_5 form a geometric sequence
  a 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l343_34316


namespace NUMINAMATH_CALUDE_degree_of_minus_5xy_squared_l343_34317

/-- The type of monomials with integer coefficients in two variables -/
structure Monomial :=
  (coeff : ℤ)
  (x_exp : ℕ)
  (y_exp : ℕ)

/-- The degree of a monomial is the sum of its exponents -/
def degree (m : Monomial) : ℕ := m.x_exp + m.y_exp

/-- The monomial -5xy^2 -/
def m : Monomial := ⟨-5, 1, 2⟩

theorem degree_of_minus_5xy_squared :
  degree m = 3 := by sorry

end NUMINAMATH_CALUDE_degree_of_minus_5xy_squared_l343_34317


namespace NUMINAMATH_CALUDE_prime_sum_of_squares_l343_34341

theorem prime_sum_of_squares (p : ℕ) (hp : Nat.Prime p) :
  (∃ k : ℕ, p = 4 * k + 1) → (∃ a b : ℤ, p = a^2 + b^2) ∧
  (∃ k : ℕ, p = 8 * k + 3) → (∃ a b c : ℤ, p = a^2 + b^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_of_squares_l343_34341


namespace NUMINAMATH_CALUDE_orchestra_members_count_l343_34348

theorem orchestra_members_count :
  ∃! n : ℕ, 150 < n ∧ n < 250 ∧
    n % 4 = 2 ∧
    n % 5 = 3 ∧
    n % 7 = 4 ∧
    n = 158 := by
  sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l343_34348
