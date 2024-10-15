import Mathlib

namespace NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l1282_128220

/-- The ratio of car speed to pedestrian speed on a bridge -/
theorem car_pedestrian_speed_ratio :
  ∀ (L : ℝ) (vp vc : ℝ),
  L > 0 →  -- The bridge has positive length
  vp > 0 →  -- The pedestrian's speed is positive
  vc > 0 →  -- The car's speed is positive
  (2 / 5 * L) / vp = L / vc →  -- Time for pedestrian to return equals time for car to reach start
  (3 / 5 * L) / vp = L / vc →  -- Time for pedestrian to finish equals time for car to finish
  vc / vp = 5 := by
sorry

end NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l1282_128220


namespace NUMINAMATH_CALUDE_base_equation_solution_l1282_128275

/-- Converts a base-10 number to base-a representation --/
def toBaseA (n : ℕ) (a : ℕ) : List ℕ := sorry

/-- Converts a base-a number to base-10 representation --/
def fromBaseA (digits : List ℕ) (a : ℕ) : ℕ := sorry

/-- Adds two numbers in base-a --/
def addBaseA (n1 : List ℕ) (n2 : List ℕ) (a : ℕ) : List ℕ := sorry

theorem base_equation_solution :
  ∃! a : ℕ, 
    a > 11 ∧ 
    addBaseA (toBaseA 396 a) (toBaseA 574 a) a = toBaseA (96 * 11) a := by
  sorry

end NUMINAMATH_CALUDE_base_equation_solution_l1282_128275


namespace NUMINAMATH_CALUDE_annual_fixed_costs_satisfy_profit_equation_l1282_128292

/-- Represents the annual fixed costs for Model X -/
def annual_fixed_costs : ℝ := 50200000

/-- Represents the desired annual profit -/
def desired_profit : ℝ := 30500000

/-- Represents the selling price per unit -/
def selling_price : ℝ := 9035

/-- Represents the variable cost per unit -/
def variable_cost : ℝ := 5000

/-- Represents the number of units sold -/
def units_sold : ℝ := 20000

/-- The profit equation -/
def profit_equation (fixed_costs : ℝ) : ℝ :=
  selling_price * units_sold - variable_cost * units_sold - fixed_costs

/-- Theorem stating that the annual fixed costs satisfy the profit equation -/
theorem annual_fixed_costs_satisfy_profit_equation :
  profit_equation annual_fixed_costs = desired_profit := by
  sorry

end NUMINAMATH_CALUDE_annual_fixed_costs_satisfy_profit_equation_l1282_128292


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l1282_128266

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence
  (a : ℕ → ℕ)
  (h_seq : fibonacci_like_sequence a)
  (h_7 : a 7 = 42)
  (h_9 : a 9 = 110) :
  a 4 = 10 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l1282_128266


namespace NUMINAMATH_CALUDE_inverse_variation_result_l1282_128230

/-- A function representing the inverse variation of 7y with the cube of x -/
def inverse_variation (x y : ℝ) : Prop :=
  ∃ k : ℝ, 7 * y = k / (x ^ 3)

/-- The theorem stating that given the inverse variation and initial condition,
    when x = 4, y = 1 -/
theorem inverse_variation_result :
  (∃ y₀ : ℝ, inverse_variation 2 y₀ ∧ y₀ = 8) →
  (∃ y : ℝ, inverse_variation 4 y ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_result_l1282_128230


namespace NUMINAMATH_CALUDE_gcd_459_357_l1282_128213

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1282_128213


namespace NUMINAMATH_CALUDE_estate_distribution_theorem_l1282_128227

/-- Represents the estate distribution problem -/
structure EstateProblem where
  num_beneficiaries : Nat
  min_ratio : Real
  known_amount : Real

/-- Calculates the smallest possible range between the highest and lowest amounts -/
def smallest_range (problem : EstateProblem) : Real :=
  sorry

/-- The theorem stating the smallest possible range for the given problem -/
theorem estate_distribution_theorem (problem : EstateProblem) 
  (h1 : problem.num_beneficiaries = 8)
  (h2 : problem.min_ratio = 1.4)
  (h3 : problem.known_amount = 80000) :
  smallest_range problem = 72412 := by
  sorry

end NUMINAMATH_CALUDE_estate_distribution_theorem_l1282_128227


namespace NUMINAMATH_CALUDE_tara_wrong_questions_l1282_128284

theorem tara_wrong_questions
  (total_questions : ℕ)
  (t u v w : ℕ)
  (h1 : t + u = v + w)
  (h2 : t + w = u + v + 6)
  (h3 : v = 3)
  (h4 : total_questions = 40) :
  t = 9 := by
sorry

end NUMINAMATH_CALUDE_tara_wrong_questions_l1282_128284


namespace NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l1282_128222

/-- Two lines are parallel if their slopes are equal -/
def parallel (a1 b1 a2 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

/-- The first line: x + (1+m)y + (m-2) = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := x + (1+m)*y + (m-2) = 0

/-- The second line: mx + 2y + 8 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop := m*x + 2*y + 8 = 0

theorem parallel_lines_m_equals_one :
  ∀ m : ℝ, parallel 1 (1+m) m 2 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l1282_128222


namespace NUMINAMATH_CALUDE_comparison_theorem_l1282_128258

theorem comparison_theorem (n : ℕ) (h : n ≥ 2) :
  (2^(2^2) * n < 3^(3^(3^3)) * n - 1) ∧
  (3^(3^(3^3)) * n > 4^(4^(4^4)) * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l1282_128258


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l1282_128242

/-- Proves that the percentage of Sikh boys in a school is 10% -/
theorem percentage_of_sikh_boys (total_boys : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  muslim_percent = 34 / 100 →
  hindu_percent = 28 / 100 →
  other_boys = 238 →
  (total_boys - (muslim_percent * total_boys + hindu_percent * total_boys + other_boys : ℚ)) / total_boys * 100 = 10 := by
sorry


end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l1282_128242


namespace NUMINAMATH_CALUDE_gcd_square_le_sum_l1282_128288

theorem gcd_square_le_sum (a b : ℕ) (h1 : (a + 1) % b = 0) (h2 : (b + 1) % a = 0) : 
  (Nat.gcd a b)^2 ≤ a + b := by
  sorry

end NUMINAMATH_CALUDE_gcd_square_le_sum_l1282_128288


namespace NUMINAMATH_CALUDE_fly_path_distance_l1282_128276

theorem fly_path_distance (r : ℝ) (last_leg : ℝ) (h1 : r = 60) (h2 : last_leg = 85) :
  let diameter := 2 * r
  let second_leg := Real.sqrt (diameter^2 - last_leg^2)
  diameter + last_leg + second_leg = 205 + Real.sqrt 7175 := by
  sorry

end NUMINAMATH_CALUDE_fly_path_distance_l1282_128276


namespace NUMINAMATH_CALUDE_pen_average_price_l1282_128224

/-- Given the purchase of pens and pencils with specific quantities and prices,
    prove that the average price of a pen is $12. -/
theorem pen_average_price
  (num_pens : ℕ)
  (num_pencils : ℕ)
  (total_cost : ℚ)
  (pencil_avg_price : ℚ)
  (h1 : num_pens = 30)
  (h2 : num_pencils = 75)
  (h3 : total_cost = 510)
  (h4 : pencil_avg_price = 2) :
  (total_cost - num_pencils * pencil_avg_price) / num_pens = 12 :=
by sorry

end NUMINAMATH_CALUDE_pen_average_price_l1282_128224


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l1282_128280

theorem easter_egg_hunt (total_eggs : ℕ) 
  (hannah_ratio : ℕ) (harry_extra : ℕ) : 
  total_eggs = 63 ∧ hannah_ratio = 2 ∧ harry_extra = 3 →
  ∃ (helen_eggs hannah_eggs harry_eggs : ℕ),
    helen_eggs = 12 ∧
    hannah_eggs = 24 ∧
    harry_eggs = 27 ∧
    hannah_eggs = hannah_ratio * helen_eggs ∧
    harry_eggs = hannah_eggs + harry_extra ∧
    helen_eggs + hannah_eggs + harry_eggs = total_eggs :=
by sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l1282_128280


namespace NUMINAMATH_CALUDE_sashas_work_portion_l1282_128217

theorem sashas_work_portion (car1 car2 car3 : ℚ) 
  (h1 : car1 = 1 / 3)
  (h2 : car2 = 1 / 5)
  (h3 : car3 = 1 / 15) :
  (car1 + car2 + car3) / 3 = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_sashas_work_portion_l1282_128217


namespace NUMINAMATH_CALUDE_sophia_stamp_collection_value_l1282_128293

/-- Given a collection of stamps with equal value, calculate the total value. -/
def stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℕ) : ℕ :=
  total_stamps * (sample_value / sample_stamps)

/-- Theorem: Sophia's stamp collection is worth 120 dollars. -/
theorem sophia_stamp_collection_value :
  stamp_collection_value 24 8 40 = 120 := by
  sorry

#eval stamp_collection_value 24 8 40

end NUMINAMATH_CALUDE_sophia_stamp_collection_value_l1282_128293


namespace NUMINAMATH_CALUDE_g_3_equals_109_l1282_128286

def g (x : ℝ) : ℝ := 7 * x^3 - 8 * x^2 - 5 * x + 7

theorem g_3_equals_109 : g 3 = 109 := by
  sorry

end NUMINAMATH_CALUDE_g_3_equals_109_l1282_128286


namespace NUMINAMATH_CALUDE_abc_inequality_l1282_128231

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a / (a^3 - a^2 + 3)) + (b / (b^3 - b^2 + 3)) + (c / (c^3 - c^2 + 3)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1282_128231


namespace NUMINAMATH_CALUDE_prism_volume_l1282_128273

/-- The volume of a right rectangular prism with given face areas and sum of dimensions -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 18) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) 
  (h4 : a + b + c = 11) : 
  a * b * c = 12 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1282_128273


namespace NUMINAMATH_CALUDE_problem_solution_l1282_128243

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1282_128243


namespace NUMINAMATH_CALUDE_discounted_price_is_correct_l1282_128246

/-- Calculate the final price after applying two successive discounts -/
def final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initial_price * (1 - discount1) * (1 - discount2)

/-- Theorem stating that the final price after discounts is approximately 59.85 -/
theorem discounted_price_is_correct :
  let initial_price : ℝ := 70
  let discount1 : ℝ := 0.1  -- 10%
  let discount2 : ℝ := 0.04999999999999997  -- 4.999999999999997%
  abs (final_price initial_price discount1 discount2 - 59.85) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_is_correct_l1282_128246


namespace NUMINAMATH_CALUDE_star_polygon_is_pyramid_net_l1282_128228

/-- Represents a star-shaped polygon constructed from two concentric circles and an inscribed regular polygon -/
structure StarPolygon where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  n : ℕ  -- Number of sides of the inscribed regular polygon
  h : R > r  -- Condition that the larger circle's radius is greater than the smaller circle's radius

/-- Determines whether a star-shaped polygon is the net of a pyramid -/
def is_pyramid_net (s : StarPolygon) : Prop :=
  s.R > 2 * s.r

/-- Theorem stating the condition for a star-shaped polygon to be the net of a pyramid -/
theorem star_polygon_is_pyramid_net (s : StarPolygon) :
  is_pyramid_net s ↔ s.R > 2 * s.r :=
sorry

end NUMINAMATH_CALUDE_star_polygon_is_pyramid_net_l1282_128228


namespace NUMINAMATH_CALUDE_inequality_range_l1282_128209

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * a * x + a - 2 < 0) ↔ a ∈ Set.Ioc (-8/5) 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l1282_128209


namespace NUMINAMATH_CALUDE_average_weight_of_class_l1282_128296

theorem average_weight_of_class (group1_count : Nat) (group1_avg : Real) 
  (group2_count : Nat) (group2_avg : Real) :
  group1_count = 22 →
  group2_count = 8 →
  group1_avg = 50.25 →
  group2_avg = 45.15 →
  let total_count := group1_count + group2_count
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  (total_weight / total_count) = 48.89 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_class_l1282_128296


namespace NUMINAMATH_CALUDE_negation_of_implication_l1282_128297

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1282_128297


namespace NUMINAMATH_CALUDE_rounding_down_2A3_l1282_128262

def round_down_to_nearest_ten (n : ℕ) : ℕ :=
  (n / 10) * 10

theorem rounding_down_2A3 (A : ℕ) (h1 : A < 10) :
  (round_down_to_nearest_ten (200 + 10 * A + 3) = 280) → A = 8 := by
  sorry

end NUMINAMATH_CALUDE_rounding_down_2A3_l1282_128262


namespace NUMINAMATH_CALUDE_perimeter_plus_area_of_specific_parallelogram_l1282_128235

/-- A parallelogram in a 2D coordinate plane -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := sorry

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- The sum of perimeter and area of a specific parallelogram -/
theorem perimeter_plus_area_of_specific_parallelogram :
  let p := Parallelogram.mk (2, 1) (7, 1) (5, 6) (10, 6)
  perimeter p + area p = 35 + 2 * Real.sqrt 34 := by sorry

end NUMINAMATH_CALUDE_perimeter_plus_area_of_specific_parallelogram_l1282_128235


namespace NUMINAMATH_CALUDE_pool_water_volume_l1282_128237

/-- Calculates the remaining water volume in a pool after evaporation --/
def remaining_water_volume (initial_volume : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_volume - evaporation_rate * days

/-- Theorem: The remaining water volume after 45 days is 355 gallons --/
theorem pool_water_volume : 
  remaining_water_volume 400 1 45 = 355 := by
  sorry

end NUMINAMATH_CALUDE_pool_water_volume_l1282_128237


namespace NUMINAMATH_CALUDE_lcm_of_9_16_21_l1282_128289

theorem lcm_of_9_16_21 : Nat.lcm 9 (Nat.lcm 16 21) = 1008 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_16_21_l1282_128289


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l1282_128236

-- Part 1
theorem inequality_one (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

-- Part 2
theorem inequality_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  a*b + b*c + c*a ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l1282_128236


namespace NUMINAMATH_CALUDE_floor_sqrt_equality_l1282_128255

theorem floor_sqrt_equality (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 1)⌋ ∧
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_equality_l1282_128255


namespace NUMINAMATH_CALUDE_volunteer_selection_l1282_128240

theorem volunteer_selection (n : ℕ) (h : n = 5) : 
  (n.choose 1) * ((n - 1).choose 1 * (n - 2).choose 1) = 60 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_l1282_128240


namespace NUMINAMATH_CALUDE_total_bills_is_126_l1282_128298

/-- Represents the number of bills and their total value -/
structure CashierMoney where
  five_dollar_bills : ℕ
  ten_dollar_bills : ℕ
  total_value : ℕ

/-- Theorem stating that given the conditions, the total number of bills is 126 -/
theorem total_bills_is_126 (money : CashierMoney) 
  (h1 : money.five_dollar_bills = 84)
  (h2 : money.total_value = 840)
  (h3 : money.total_value = 5 * money.five_dollar_bills + 10 * money.ten_dollar_bills) :
  money.five_dollar_bills + money.ten_dollar_bills = 126 := by
  sorry


end NUMINAMATH_CALUDE_total_bills_is_126_l1282_128298


namespace NUMINAMATH_CALUDE_x_power_twenty_equals_one_l1282_128278

theorem x_power_twenty_equals_one (x : ℝ) (h : x + 1/x = 2) : x^20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_twenty_equals_one_l1282_128278


namespace NUMINAMATH_CALUDE_octal_perfect_square_last_digit_l1282_128294

/-- A perfect square in octal form (abc)₈ where a ≠ 0 always has c = 1 -/
theorem octal_perfect_square_last_digit (a b c : Nat) (h1 : a ≠ 0) 
  (h2 : ∃ (n : Nat), n^2 = a * 8^2 + b * 8 + c) : c = 1 := by
  sorry

end NUMINAMATH_CALUDE_octal_perfect_square_last_digit_l1282_128294


namespace NUMINAMATH_CALUDE_kenya_peanuts_count_l1282_128250

/-- The number of peanuts Jose has -/
def jose_peanuts : ℕ := 85

/-- The additional number of peanuts Kenya has compared to Jose -/
def kenya_extra_peanuts : ℕ := 48

/-- The number of peanuts Kenya has -/
def kenya_peanuts : ℕ := jose_peanuts + kenya_extra_peanuts

theorem kenya_peanuts_count : kenya_peanuts = 133 := by
  sorry

end NUMINAMATH_CALUDE_kenya_peanuts_count_l1282_128250


namespace NUMINAMATH_CALUDE_k_squared_upper_bound_l1282_128238

theorem k_squared_upper_bound (k n : ℕ) (h1 : 121 < k^2) (h2 : k^2 < n) 
  (h3 : ∀ m : ℕ, 121 < m^2 → m^2 < n → m ≤ k + 5) : n ≤ 324 :=
sorry

end NUMINAMATH_CALUDE_k_squared_upper_bound_l1282_128238


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1282_128295

theorem circle_tangent_to_line (m : ℝ) (h : m > 0) :
  ∃ (x y : ℝ), x^2 + y^2 = 4*m ∧ x + y = 2*Real.sqrt m ∧
  ∀ (x' y' : ℝ), x'^2 + y'^2 = 4*m → x' + y' = 2*Real.sqrt m →
  (x' - x)^2 + (y' - y)^2 = 0 ∨ (x' - x)^2 + (y' - y)^2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1282_128295


namespace NUMINAMATH_CALUDE_half_of_four_power_2022_l1282_128205

theorem half_of_four_power_2022 : (4 ^ 2022) / 2 = 2 ^ 4043 := by
  sorry

end NUMINAMATH_CALUDE_half_of_four_power_2022_l1282_128205


namespace NUMINAMATH_CALUDE_room_length_proof_l1282_128269

theorem room_length_proof (x : ℝ) 
  (room_width : ℝ) (room_height : ℝ)
  (door_width : ℝ) (door_height : ℝ)
  (large_window_width : ℝ) (large_window_height : ℝ)
  (small_window_width : ℝ) (small_window_height : ℝ)
  (paint_cost_per_sqm : ℝ) (total_paint_cost : ℝ)
  (h1 : room_width = 7)
  (h2 : room_height = 5)
  (h3 : door_width = 1)
  (h4 : door_height = 3)
  (h5 : large_window_width = 2)
  (h6 : large_window_height = 1.5)
  (h7 : small_window_width = 1)
  (h8 : small_window_height = 1.5)
  (h9 : paint_cost_per_sqm = 3)
  (h10 : total_paint_cost = 474)
  (h11 : total_paint_cost = paint_cost_per_sqm * 
    (2 * (x * room_height + room_width * room_height) - 
    2 * (door_width * door_height) - 
    (large_window_width * large_window_height) - 
    2 * (small_window_width * small_window_height))) :
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l1282_128269


namespace NUMINAMATH_CALUDE_right_triangle_3_4_5_l1282_128206

theorem right_triangle_3_4_5 (a b c : ℝ) : 
  a = 3 → b = 4 → c = 5 → a^2 + b^2 = c^2 :=
by
  sorry

#check right_triangle_3_4_5

end NUMINAMATH_CALUDE_right_triangle_3_4_5_l1282_128206


namespace NUMINAMATH_CALUDE_libor_number_theorem_l1282_128277

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10) → d % 2 = 1

def no_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10) → d % 2 = 0

theorem libor_number_theorem :
  ∀ n : ℕ, is_three_digit n ∧ all_digits_odd n ∧ is_three_digit (n + 421) ∧ no_odd_digits (n + 421) →
    n = 179 ∨ n = 199 ∨ n = 379 ∨ n = 399 :=
sorry

end NUMINAMATH_CALUDE_libor_number_theorem_l1282_128277


namespace NUMINAMATH_CALUDE_car_travel_time_l1282_128216

/-- Given a car and a train traveling between two stations, this theorem proves
    the time taken by the car to reach the destination. -/
theorem car_travel_time (car_time train_time : ℝ) : 
  train_time = car_time + 2 →  -- The train takes 2 hours longer than the car
  car_time + train_time = 11 → -- The combined time is 11 hours
  car_time = 4.5 := by
  sorry

#check car_travel_time

end NUMINAMATH_CALUDE_car_travel_time_l1282_128216


namespace NUMINAMATH_CALUDE_no_geometric_sequence_trig_l1282_128247

open Real

theorem no_geometric_sequence_trig (θ : ℝ) : 
  0 < θ ∧ θ < 2 * π ∧ ¬ ∃ k : ℤ, θ = k * (π / 2) →
  ¬ (cos θ * tan θ = sin θ ^ 3 ∨ sin θ * cos θ = cos θ ^ 2 * tan θ) :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_sequence_trig_l1282_128247


namespace NUMINAMATH_CALUDE_quadratic_root_fraction_l1282_128219

theorem quadratic_root_fraction (a b : ℝ) (h1 : a ≠ b) (h2 : a + b - 20 = 0) :
  (a^2 - b^2) / (2*a - 2*b) = 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_fraction_l1282_128219


namespace NUMINAMATH_CALUDE_min_balls_guarantee_l1282_128261

def red_balls : ℕ := 35
def blue_balls : ℕ := 25
def green_balls : ℕ := 22
def yellow_balls : ℕ := 18
def white_balls : ℕ := 14
def black_balls : ℕ := 12

def total_balls : ℕ := red_balls + blue_balls + green_balls + yellow_balls + white_balls + black_balls

def min_balls_for_guarantee : ℕ := 95

theorem min_balls_guarantee :
  ∀ (drawn : ℕ), drawn ≥ min_balls_for_guarantee →
    ∃ (color : ℕ), color ≥ 18 ∧
      (color ≤ red_balls ∨ color ≤ blue_balls ∨ color ≤ green_balls ∨
       color ≤ yellow_balls ∨ color ≤ white_balls ∨ color ≤ black_balls) :=
by sorry

end NUMINAMATH_CALUDE_min_balls_guarantee_l1282_128261


namespace NUMINAMATH_CALUDE_chemistry_class_gender_difference_l1282_128272

theorem chemistry_class_gender_difference :
  ∀ (boys girls : ℕ),
  (3 : ℕ) * boys = (4 : ℕ) * girls →
  boys + girls = 42 →
  girls - boys = 6 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_class_gender_difference_l1282_128272


namespace NUMINAMATH_CALUDE_milk_for_flour_batch_l1282_128200

/-- Given that 60 mL of milk is used for every 300 mL of flour,
    prove that 300 mL of milk is needed for 1500 mL of flour. -/
theorem milk_for_flour_batch (milk_per_portion : ℝ) (flour_per_portion : ℝ) 
    (total_flour : ℝ) (h1 : milk_per_portion = 60) 
    (h2 : flour_per_portion = 300) (h3 : total_flour = 1500) : 
    (total_flour / flour_per_portion) * milk_per_portion = 300 :=
by sorry

end NUMINAMATH_CALUDE_milk_for_flour_batch_l1282_128200


namespace NUMINAMATH_CALUDE_square_root_of_x_plus_y_l1282_128290

theorem square_root_of_x_plus_y (x y : ℝ) : 
  (Real.sqrt (3 - x) + Real.sqrt (x - 3) + 1 = y) → 
  Real.sqrt (x + y) = 2 := by
sorry

end NUMINAMATH_CALUDE_square_root_of_x_plus_y_l1282_128290


namespace NUMINAMATH_CALUDE_tan_135_degrees_l1282_128259

theorem tan_135_degrees : 
  let angle : Real := 135 * Real.pi / 180
  let point : Fin 2 → Real := ![-(Real.sqrt 2) / 2, (Real.sqrt 2) / 2]
  Real.tan angle = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_135_degrees_l1282_128259


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l1282_128263

def a : ℝ × ℝ × ℝ := (1, 2, 3)
def b : ℝ × ℝ × ℝ := (0, 1, -4)

theorem vector_magnitude_proof : ‖a - 2 • b‖ = Real.sqrt 122 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l1282_128263


namespace NUMINAMATH_CALUDE_parabola_intersection_l1282_128245

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Define line l1
def line_l1 (x m : ℝ) : ℝ := -x + m

-- Define the axis of symmetry of the parabola
def axis_of_symmetry : ℝ := -1

-- Define the property that l2 is symmetric with respect to the axis of symmetry
def l2_symmetric (B D : ℝ × ℝ) : Prop :=
  B.1 + D.1 = 2 * axis_of_symmetry

-- Define the condition that A and D are above x-axis, B and C are below
def points_position (A B C D : ℝ × ℝ) : Prop :=
  A.2 > 0 ∧ D.2 > 0 ∧ B.2 < 0 ∧ C.2 < 0

-- Define the condition AC · BD = 26
def product_condition (A B C D : ℝ × ℝ) : Prop :=
  ((A.1 - C.1)^2 + (A.2 - C.2)^2) * ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 26

-- Theorem statement
theorem parabola_intersection (m : ℝ) 
  (A B C D : ℝ × ℝ) 
  (h1 : ∀ x, parabola x = line_l1 x m → (x = A.1 ∨ x = C.1))
  (h2 : l2_symmetric B D)
  (h3 : points_position A B C D)
  (h4 : product_condition A B C D) :
  m = -2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1282_128245


namespace NUMINAMATH_CALUDE_intersection_condition_l1282_128215

/-- The line equation -/
def line_equation (x y m : ℝ) : Prop := x - y + m = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 1 = 0

/-- Two distinct intersection points exist -/
def has_two_distinct_intersections (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    line_equation x₁ y₁ m ∧ circle_equation x₁ y₁ ∧
    line_equation x₂ y₂ m ∧ circle_equation x₂ y₂

/-- The theorem statement -/
theorem intersection_condition (m : ℝ) :
  0 < m → m < 1 → has_two_distinct_intersections m :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_l1282_128215


namespace NUMINAMATH_CALUDE_average_tv_watching_l1282_128239

def tv_hours : List ℝ := [10, 8, 12]

theorem average_tv_watching :
  (tv_hours.sum / tv_hours.length : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_tv_watching_l1282_128239


namespace NUMINAMATH_CALUDE_six_digit_number_concatenation_divisibility_l1282_128260

theorem six_digit_number_concatenation_divisibility : 
  let a : ℕ := 166667
  let b : ℕ := 333334
  -- a and b are six-digit numbers
  (100000 ≤ a ∧ a < 1000000) ∧
  (100000 ≤ b ∧ b < 1000000) ∧
  -- The concatenated number is divisible by the product
  (1000000 * a + b) % (a * b) = 0 := by
sorry

end NUMINAMATH_CALUDE_six_digit_number_concatenation_divisibility_l1282_128260


namespace NUMINAMATH_CALUDE_cow_value_increase_is_600_l1282_128232

/-- Calculates the increase in a cow's value after weight gain -/
def cow_value_increase (initial_weight : ℝ) (weight_increase_factor : ℝ) (price_per_pound : ℝ) : ℝ :=
  (initial_weight * weight_increase_factor * price_per_pound) - (initial_weight * price_per_pound)

/-- Theorem stating that the increase in the cow's value is $600 -/
theorem cow_value_increase_is_600 :
  cow_value_increase 400 1.5 3 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cow_value_increase_is_600_l1282_128232


namespace NUMINAMATH_CALUDE_cos_95_cos_25_minus_sin_95_sin_25_l1282_128283

theorem cos_95_cos_25_minus_sin_95_sin_25 :
  Real.cos (95 * π / 180) * Real.cos (25 * π / 180) - 
  Real.sin (95 * π / 180) * Real.sin (25 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_95_cos_25_minus_sin_95_sin_25_l1282_128283


namespace NUMINAMATH_CALUDE_michael_digging_time_l1282_128253

/-- The time it takes Michael to dig his hole given the conditions -/
theorem michael_digging_time 
  (father_rate : ℝ) 
  (father_time : ℝ) 
  (michael_rate : ℝ) 
  (michael_depth_diff : ℝ) :
  father_rate = 4 →
  father_time = 400 →
  michael_rate = father_rate →
  michael_depth_diff = 400 →
  (2 * (father_rate * father_time) - michael_depth_diff) / michael_rate = 700 :=
by sorry

end NUMINAMATH_CALUDE_michael_digging_time_l1282_128253


namespace NUMINAMATH_CALUDE_power_equation_solution_l1282_128202

theorem power_equation_solution (y : ℕ) : 8^5 + 8^5 + 2 * 8^5 = 2^y → y = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1282_128202


namespace NUMINAMATH_CALUDE_sphere_surface_area_containing_unit_cube_l1282_128274

/-- The surface area of a sphere that contains all eight vertices of a unit cube -/
theorem sphere_surface_area_containing_unit_cube : ℝ := by
  -- Define a cube with edge length 1
  let cube_edge_length : ℝ := 1

  -- Define the sphere that contains all vertices of the cube
  let sphere_radius : ℝ := (Real.sqrt 3) / 2

  -- Define the surface area of the sphere
  let sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius^2

  -- Prove that the surface area equals 3π
  have : sphere_surface_area = 3 * Real.pi := by sorry

  -- Return the result
  exact 3 * Real.pi


end NUMINAMATH_CALUDE_sphere_surface_area_containing_unit_cube_l1282_128274


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1282_128203

def set_A : Set ℝ := {x | x < -1 ∨ x > 3}
def set_B : Set ℝ := {x | x - 2 ≥ 0}

theorem union_of_A_and_B : set_A ∪ set_B = {x | x < -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1282_128203


namespace NUMINAMATH_CALUDE_parabola_inequality_l1282_128214

def f (x : ℝ) : ℝ := -(x - 2)^2

theorem parabola_inequality : f (-1) < f 4 ∧ f 4 < f 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_inequality_l1282_128214


namespace NUMINAMATH_CALUDE_min_tile_A_1011_l1282_128267

/-- Represents a tile type -/
inductive Tile
| A  -- Covers 3 squares: 2 in one row and 1 in the adjacent row
| B  -- Covers 4 squares: 2 in one row and 2 in the adjacent row

/-- Represents a tiling of a square grid -/
def Tiling (n : ℕ) := List (Tile × ℕ × ℕ)  -- List of (tile type, row, column)

/-- Checks if a tiling is valid for an n×n square -/
def isValidTiling (n : ℕ) (t : Tiling n) : Prop := sorry

/-- Counts the number of tiles of type A in a tiling -/
def countTileA (t : Tiling n) : ℕ := sorry

/-- Theorem: The minimum number of tiles A required to tile a 1011×1011 square is 2023 -/
theorem min_tile_A_1011 :
  ∀ t : Tiling 1011, isValidTiling 1011 t → countTileA t ≥ 2023 ∧
  ∃ t' : Tiling 1011, isValidTiling 1011 t' ∧ countTileA t' = 2023 := by
  sorry

#check min_tile_A_1011

end NUMINAMATH_CALUDE_min_tile_A_1011_l1282_128267


namespace NUMINAMATH_CALUDE_arithmetic_progression_equiv_square_product_l1282_128285

theorem arithmetic_progression_equiv_square_product 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (∃ d : ℝ, Real.log y - Real.log x = d ∧ Real.log z - Real.log y = d) ↔ 
  y^2 = x*z := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equiv_square_product_l1282_128285


namespace NUMINAMATH_CALUDE_cuboid_volume_example_l1282_128248

/-- The volume of a cuboid with given base area and height -/
def cuboid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  base_area * height

/-- Theorem: The volume of a cuboid with base area 14 cm² and height 13 cm is 182 cm³ -/
theorem cuboid_volume_example : cuboid_volume 14 13 = 182 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_example_l1282_128248


namespace NUMINAMATH_CALUDE_deposit_calculation_l1282_128223

theorem deposit_calculation (total_price : ℝ) (deposit_percentage : ℝ) (remaining_amount : ℝ) : 
  deposit_percentage = 0.1 →
  remaining_amount = 945 →
  total_price * (1 - deposit_percentage) = remaining_amount →
  total_price * deposit_percentage = 105 := by
sorry

end NUMINAMATH_CALUDE_deposit_calculation_l1282_128223


namespace NUMINAMATH_CALUDE_balloon_difference_l1282_128291

def your_balloons : ℕ := 7
def friend_balloons : ℕ := 5

theorem balloon_difference : your_balloons - friend_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l1282_128291


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1282_128249

theorem cos_alpha_value (α : Real) 
  (h1 : π/4 < α) 
  (h2 : α < 3*π/4) 
  (h3 : Real.sin (α - π/4) = 4/5) : 
  Real.cos α = -Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1282_128249


namespace NUMINAMATH_CALUDE_total_savings_is_40_l1282_128229

-- Define the number of coins each child has
def teagan_pennies : ℕ := 200
def rex_nickels : ℕ := 100
def toni_dimes : ℕ := 330

-- Define the conversion rates
def pennies_per_dollar : ℕ := 100
def nickels_per_dollar : ℕ := 20
def dimes_per_dollar : ℕ := 10

-- Define the total savings
def total_savings : ℚ :=
  (teagan_pennies : ℚ) / pennies_per_dollar +
  (rex_nickels : ℚ) / nickels_per_dollar +
  (toni_dimes : ℚ) / dimes_per_dollar

-- Theorem statement
theorem total_savings_is_40 : total_savings = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_is_40_l1282_128229


namespace NUMINAMATH_CALUDE_derivatives_verification_l1282_128208

theorem derivatives_verification :
  (∀ x : ℝ, deriv (λ x => x^2) x = 2 * x) ∧
  (∀ x : ℝ, deriv Real.sin x = Real.cos x) ∧
  (∀ x : ℝ, deriv (λ x => Real.exp (-x)) x = -Real.exp (-x)) ∧
  (∀ x : ℝ, x ≠ -1 → deriv (λ x => Real.log (x + 1)) x = 1 / (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_derivatives_verification_l1282_128208


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1282_128201

/-- The quadratic function f(x) = -x^2 + bx - 7 is negative only for x < 2 or x > 6 -/
def quadratic_inequality (b : ℝ) : Prop :=
  ∀ x : ℝ, (-x^2 + b*x - 7 < 0) ↔ (x < 2 ∨ x > 6)

/-- Given the quadratic inequality condition, prove that b = 8 -/
theorem quadratic_inequality_solution :
  ∃ b : ℝ, quadratic_inequality b ∧ b = 8 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1282_128201


namespace NUMINAMATH_CALUDE_diagonal_length_of_regular_hexagon_l1282_128282

/-- A regular hexagon with side length 12 units -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 12)

/-- The length of a diagonal in a regular hexagon -/
def diagonal_length (h : RegularHexagon) : ℝ := 2 * h.side_length

/-- Theorem: The diagonal length of a regular hexagon with side length 12 is 24 -/
theorem diagonal_length_of_regular_hexagon (h : RegularHexagon) :
  diagonal_length h = 24 := by
  sorry

#check diagonal_length_of_regular_hexagon

end NUMINAMATH_CALUDE_diagonal_length_of_regular_hexagon_l1282_128282


namespace NUMINAMATH_CALUDE_right_triangle_k_value_l1282_128251

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the vectors
def vector_AB (k : ℝ) : ℝ × ℝ := (k, 1)
def vector_AC : ℝ × ℝ := (2, 3)

-- Define the right angle condition
def is_right_angle (t : Triangle) : Prop :=
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  (t.C.1 - t.A.1) * BC.1 + (t.C.2 - t.A.2) * BC.2 = 0

-- Theorem statement
theorem right_triangle_k_value (t : Triangle) (k : ℝ) :
  is_right_angle t →
  t.B - t.A = vector_AB k →
  t.C - t.A = vector_AC →
  k = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_k_value_l1282_128251


namespace NUMINAMATH_CALUDE_binary_rep_of_31_l1282_128210

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Theorem: The binary representation of 31 is [true, true, true, true, true] -/
theorem binary_rep_of_31 : toBinary 31 = [true, true, true, true, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_rep_of_31_l1282_128210


namespace NUMINAMATH_CALUDE_blue_pill_cost_is_21_l1282_128221

/-- The cost of a blue pill given the conditions of Ben's medication regimen -/
def blue_pill_cost (total_cost : ℚ) (duration_days : ℕ) (blue_red_diff : ℚ) : ℚ :=
  let daily_cost : ℚ := total_cost / duration_days
  let x : ℚ := (daily_cost + blue_red_diff) / 2
  x

theorem blue_pill_cost_is_21 :
  blue_pill_cost 819 21 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_blue_pill_cost_is_21_l1282_128221


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l1282_128271

/-- The sum of the first n terms of the sequence -/
def S (a n : ℕ) : ℕ := a * n^2 + n

/-- The n-th term of the sequence -/
def a_n (a n : ℕ) : ℤ := S a n - S a (n-1)

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic_sequence (f : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, f (n+1) - f n = d

theorem sequence_is_arithmetic (a : ℕ) (h : a > 0) :
  is_arithmetic_sequence (a_n a) := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l1282_128271


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l1282_128287

theorem product_of_cubic_fractions :
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 2) * (f 3) * (f 4) * (f 5) * (f 6) = 43 / 63 := by
sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l1282_128287


namespace NUMINAMATH_CALUDE_readers_of_both_genres_l1282_128234

theorem readers_of_both_genres (total : ℕ) (sci_fi : ℕ) (literary : ℕ) 
  (h_total : total = 150)
  (h_sci_fi : sci_fi = 120)
  (h_literary : literary = 90) :
  sci_fi + literary - total = 60 := by
  sorry

end NUMINAMATH_CALUDE_readers_of_both_genres_l1282_128234


namespace NUMINAMATH_CALUDE_remainder_21_pow_2051_mod_29_l1282_128270

theorem remainder_21_pow_2051_mod_29 : 21^2051 % 29 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_21_pow_2051_mod_29_l1282_128270


namespace NUMINAMATH_CALUDE_walters_coins_theorem_l1282_128279

/-- The value of a penny in cents -/
def penny : ℕ := 1

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The value of a quarter in cents -/
def quarter : ℕ := 25

/-- The number of cents in a dollar -/
def cents_in_dollar : ℕ := 100

/-- The percentage of a dollar represented by Walter's coins -/
def walters_coins_percentage : ℚ := (penny + nickel + dime + quarter : ℚ) / cents_in_dollar * 100

theorem walters_coins_theorem : walters_coins_percentage = 41 := by
  sorry

end NUMINAMATH_CALUDE_walters_coins_theorem_l1282_128279


namespace NUMINAMATH_CALUDE_p_and_not_q_is_true_l1282_128211

-- Define proposition p
def p : Prop := ∃ x : ℝ, x - 2 > Real.log x

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp x > 1

-- Theorem statement
theorem p_and_not_q_is_true : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_p_and_not_q_is_true_l1282_128211


namespace NUMINAMATH_CALUDE_probability_same_color_l1282_128254

/-- The number of marbles of each color in the box -/
def marbles_per_color : ℕ := 3

/-- The total number of colors -/
def num_colors : ℕ := 3

/-- The total number of marbles in the box -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- The number of marbles drawn -/
def drawn_marbles : ℕ := 3

/-- The probability of drawing 3 marbles of the same color -/
theorem probability_same_color :
  (num_colors * (Nat.choose marbles_per_color drawn_marbles)) /
  (Nat.choose total_marbles drawn_marbles) = 1 / 28 :=
sorry

end NUMINAMATH_CALUDE_probability_same_color_l1282_128254


namespace NUMINAMATH_CALUDE_elements_beginning_with_3_l1282_128252

/-- The set of powers of 7 from 0 to 2011 -/
def T : Set ℕ := {n : ℕ | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 2011 ∧ n = 7^k}

/-- The number of digits in 7^2011 -/
def digits_of_7_2011 : ℕ := 1602

/-- Function to check if a natural number begins with the digit 3 -/
def begins_with_3 (n : ℕ) : Prop := sorry

/-- The count of elements in T that begin with 3 -/
def count_begins_with_3 (S : Set ℕ) : ℕ := sorry

theorem elements_beginning_with_3 :
  count_begins_with_3 T = 45 :=
sorry

end NUMINAMATH_CALUDE_elements_beginning_with_3_l1282_128252


namespace NUMINAMATH_CALUDE_investment_period_l1282_128225

theorem investment_period (emma_investment briana_investment : ℝ)
  (emma_yield briana_yield : ℝ) (difference : ℝ) :
  emma_investment = 300 →
  briana_investment = 500 →
  emma_yield = 0.15 →
  briana_yield = 0.10 →
  difference = 10 →
  ∃ t : ℝ, t = 2 ∧ 
    t * (briana_investment * briana_yield - emma_investment * emma_yield) = difference :=
by sorry

end NUMINAMATH_CALUDE_investment_period_l1282_128225


namespace NUMINAMATH_CALUDE_polynomial_positivity_l1282_128264

theorem polynomial_positivity (P : ℕ → ℝ) 
  (h0 : P 0 > 0)
  (h1 : P 1 > P 0)
  (h2 : P 2 > 2 * P 1 - P 0)
  (h3 : P 3 > 3 * P 2 - 3 * P 1 + P 0)
  (h4 : ∀ n : ℕ, P (n + 4) > 4 * P (n + 3) - 6 * P (n + 2) + 4 * P (n + 1) - P n) :
  ∀ n : ℕ, n > 0 → P n > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_positivity_l1282_128264


namespace NUMINAMATH_CALUDE_locus_of_Q_l1282_128299

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -1)
def C : ℝ × ℝ := (1, 3)

-- Define a point P on line BC
def P : ℝ → ℝ × ℝ := λ t => ((1 - t) * B.1 + t * C.1, (1 - t) * B.2 + t * C.2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Define the locus equation
def locus_eq (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- State the theorem
theorem locus_of_Q (t : ℝ) :
  let p := P t
  let q := vec_add p (vec_add (vec_sub A p) (vec_add (vec_sub B p) (vec_sub C p)))
  locus_eq q.1 q.2 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_Q_l1282_128299


namespace NUMINAMATH_CALUDE_new_person_weight_l1282_128212

theorem new_person_weight (initial_count : ℕ) (leaving_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  leaving_weight = 70 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + leaving_weight = 90 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1282_128212


namespace NUMINAMATH_CALUDE_original_number_l1282_128244

theorem original_number (N : ℕ) : (∀ k : ℕ, N - 7 ≠ 12 * k) ∧ (∃ k : ℕ, N - 7 = 12 * k) → N = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1282_128244


namespace NUMINAMATH_CALUDE_first_2500_even_integers_digits_l1282_128268

/-- The total number of digits used to write the first n positive even integers -/
def totalDigits (n : ℕ) : ℕ :=
  sorry

/-- The 2500th positive even integer -/
def evenInteger2500 : ℕ := 5000

theorem first_2500_even_integers_digits :
  totalDigits 2500 = 9449 :=
sorry

end NUMINAMATH_CALUDE_first_2500_even_integers_digits_l1282_128268


namespace NUMINAMATH_CALUDE_mike_seashells_l1282_128257

/-- The total number of seashells Mike found -/
def total_seashells (initial : ℝ) (later : ℝ) : ℝ := initial + later

/-- Theorem stating that Mike found 10.75 seashells in total -/
theorem mike_seashells :
  let initial_seashells : ℝ := 6.5
  let later_seashells : ℝ := 4.25
  total_seashells initial_seashells later_seashells = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_mike_seashells_l1282_128257


namespace NUMINAMATH_CALUDE_james_fish_purchase_l1282_128207

theorem james_fish_purchase (fish_per_roll : ℕ) (bad_fish_percent : ℚ) (rolls_made : ℕ) :
  fish_per_roll = 40 →
  bad_fish_percent = 1/5 →
  rolls_made = 8 →
  ∃ (total_fish : ℕ), total_fish = 400 ∧ 
    (total_fish : ℚ) * (1 - bad_fish_percent) = (fish_per_roll * rolls_made : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_james_fish_purchase_l1282_128207


namespace NUMINAMATH_CALUDE_b_sixth_congruence_l1282_128241

theorem b_sixth_congruence (n : ℕ+) (b : ℤ) (h : b^3 ≡ 1 [ZMOD n]) :
  b^6 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_b_sixth_congruence_l1282_128241


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l1282_128233

theorem quadratic_inequalities (x : ℝ) :
  (((1/2 : ℝ) * x^2 - 4*x + 6 < 0) ↔ (2 < x ∧ x < 6)) ∧
  ((4*x^2 - 4*x + 1 ≥ 0) ↔ True) ∧
  ((2*x^2 - x - 1 ≤ 0) ↔ (-1/2 ≤ x ∧ x ≤ 1)) ∧
  ((3*(x-2)*(x+2) - 4*(x+1)^2 + 1 < 0) ↔ (x < -5 ∨ x > -3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l1282_128233


namespace NUMINAMATH_CALUDE_perpendicular_distance_is_six_l1282_128204

/-- A rectangular parallelepiped with dimensions 6 × 5 × 4 -/
structure Parallelepiped where
  length : ℝ := 6
  width : ℝ := 5
  height : ℝ := 4

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The perpendicular distance from a point to a plane -/
def perpendicularDistance (S : Point3D) (P Q R : Point3D) : ℝ := sorry

theorem perpendicular_distance_is_six :
  let p : Parallelepiped := { }
  let S : Point3D := ⟨6, 0, 0⟩
  let P : Point3D := ⟨0, 0, 0⟩
  let Q : Point3D := ⟨0, 5, 0⟩
  let R : Point3D := ⟨0, 0, 4⟩
  perpendicularDistance S P Q R = 6 := by sorry

end NUMINAMATH_CALUDE_perpendicular_distance_is_six_l1282_128204


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_300_l1282_128218

def sum_of_divisors (n : ℕ) : ℕ := sorry

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_300 :
  largest_prime_factor (sum_of_divisors 300) = 31 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_300_l1282_128218


namespace NUMINAMATH_CALUDE_red_peaches_count_l1282_128256

theorem red_peaches_count (total : ℕ) (yellow : ℕ) (green : ℕ) (red : ℕ) 
  (h1 : total = 30)
  (h2 : yellow = 15)
  (h3 : green = 8)
  (h4 : total = red + yellow + green) : 
  red = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l1282_128256


namespace NUMINAMATH_CALUDE_average_after_removal_l1282_128226

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) :
  Finset.card numbers = 10 →
  sum = Finset.sum numbers id →
  sum / 10 = 85 →
  72 ∈ numbers →
  78 ∈ numbers →
  ((sum - 72 - 78) / 8) = 87.5 :=
sorry

end NUMINAMATH_CALUDE_average_after_removal_l1282_128226


namespace NUMINAMATH_CALUDE_magic_shop_cost_correct_l1282_128265

/-- Calculates the total cost for Tom and his friend at the magic shop --/
def magic_shop_cost (trick_deck_price : ℚ) (gimmick_coin_price : ℚ) 
  (trick_deck_count : ℕ) (gimmick_coin_count : ℕ) 
  (trick_deck_discount : ℚ) (gimmick_coin_discount : ℚ) 
  (sales_tax : ℚ) : ℚ :=
  let total_trick_decks := 2 * trick_deck_count * trick_deck_price
  let total_gimmick_coins := 2 * gimmick_coin_count * gimmick_coin_price
  let discounted_trick_decks := 
    if trick_deck_count > 2 then total_trick_decks * (1 - trick_deck_discount) 
    else total_trick_decks
  let discounted_gimmick_coins := 
    if gimmick_coin_count > 3 then total_gimmick_coins * (1 - gimmick_coin_discount) 
    else total_gimmick_coins
  let total_after_discounts := discounted_trick_decks + discounted_gimmick_coins
  let total_with_tax := total_after_discounts * (1 + sales_tax)
  total_with_tax

theorem magic_shop_cost_correct : 
  magic_shop_cost 8 12 3 4 (1/10) (1/20) (7/100) = 14381/100 := by
  sorry

end NUMINAMATH_CALUDE_magic_shop_cost_correct_l1282_128265


namespace NUMINAMATH_CALUDE_quadratic_roots_max_reciprocal_sum_l1282_128281

theorem quadratic_roots_max_reciprocal_sum (t q r₁ r₂ : ℝ) : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 1003 → r₁^n + r₂^n = r₁ + r₂) →
  r₁ * r₂ = q →
  r₁ + r₂ = t →
  r₁ ≠ 0 →
  r₂ ≠ 0 →
  (1 / r₁^1004 + 1 / r₂^1004) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_max_reciprocal_sum_l1282_128281
