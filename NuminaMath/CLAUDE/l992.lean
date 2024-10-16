import Mathlib

namespace NUMINAMATH_CALUDE_set_equality_implies_m_zero_l992_99277

theorem set_equality_implies_m_zero (m : ℝ) : 
  ({3, m} : Set ℝ) = ({3 * m, 3} : Set ℝ) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_zero_l992_99277


namespace NUMINAMATH_CALUDE_ratio_arithmetic_sequence_property_l992_99291

/-- Definition of a ratio arithmetic sequence -/
def is_ratio_arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 2) / a (n + 1) - a (n + 1) / a n = d

/-- Theorem about the specific ratio arithmetic sequence -/
theorem ratio_arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) :
  is_ratio_arithmetic a d →
  a 1 = 1 →
  a 2 = 1 →
  a 3 = 2 →
  a 2009 / a 2006 = 2006 := by
sorry

end NUMINAMATH_CALUDE_ratio_arithmetic_sequence_property_l992_99291


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l992_99237

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

/-- The main theorem -/
theorem collinear_points_x_value :
  let A : Point := ⟨-1, 1⟩
  let B : Point := ⟨2, -4⟩
  let C : Point := ⟨x, -9⟩
  collinear A B C → x = 5 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_x_value_l992_99237


namespace NUMINAMATH_CALUDE_number_of_grades_l992_99201

theorem number_of_grades (students_per_grade : ℕ) (total_students : ℕ) : 
  students_per_grade = 75 → total_students = 22800 → total_students / students_per_grade = 304 := by
  sorry

end NUMINAMATH_CALUDE_number_of_grades_l992_99201


namespace NUMINAMATH_CALUDE_cookie_pattern_proof_l992_99236

def cookie_sequence (n : ℕ) : ℕ := 
  match n with
  | 1 => 5
  | 2 => 5  -- This is what we want to prove
  | 3 => 10
  | 4 => 14
  | 5 => 19
  | 6 => 25
  | _ => 0  -- For other values, we don't care in this problem

theorem cookie_pattern_proof : 
  (cookie_sequence 1 = 5) ∧ 
  (cookie_sequence 3 = 10) ∧ 
  (cookie_sequence 4 = 14) ∧ 
  (cookie_sequence 5 = 19) ∧ 
  (cookie_sequence 6 = 25) ∧ 
  (∀ n : ℕ, n > 2 → cookie_sequence n - cookie_sequence (n-1) = 
    if n % 2 = 0 then 4 else 5) →
  cookie_sequence 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_cookie_pattern_proof_l992_99236


namespace NUMINAMATH_CALUDE_daisy_milk_leftover_l992_99211

/-- Calculates the amount of milk left over given the total production, percentage consumed by kids, and percentage of remainder used for cooking. -/
def milk_left_over (total_milk : ℝ) (kids_consumption_percent : ℝ) (cooking_percent : ℝ) : ℝ :=
  let remaining_after_kids := total_milk * (1 - kids_consumption_percent)
  remaining_after_kids * (1 - cooking_percent)

/-- Theorem stating that given 16 cups of milk per day, with 75% consumed by kids and 50% of the remainder used for cooking, 2 cups of milk are left over. -/
theorem daisy_milk_leftover :
  milk_left_over 16 0.75 0.5 = 2 := by
  sorry

#eval milk_left_over 16 0.75 0.5

end NUMINAMATH_CALUDE_daisy_milk_leftover_l992_99211


namespace NUMINAMATH_CALUDE_at_least_one_zero_l992_99258

theorem at_least_one_zero (a b : ℝ) : (a ≠ 0 ∧ b ≠ 0) → False := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_zero_l992_99258


namespace NUMINAMATH_CALUDE_vector_operation_result_l992_99239

/-- Proves that the given vector operation results in (4, -7) -/
theorem vector_operation_result : 
  4 • !![3, -9] - 3 • !![2, -7] + 2 • !![-1, 4] = !![4, -7] := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_result_l992_99239


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l992_99229

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l992_99229


namespace NUMINAMATH_CALUDE_sequence_general_term_1_l992_99266

theorem sequence_general_term_1 (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h : ∀ n, S n = 2 * n^2 - 3 * n + 2) :
  (a 1 = 1 ∧ ∀ n ≥ 2, a n = 4 * n - 5) ↔ 
  (∀ n, n ≥ 1 → a n = S n - S (n-1)) :=
sorry


end NUMINAMATH_CALUDE_sequence_general_term_1_l992_99266


namespace NUMINAMATH_CALUDE_restaurant_ratio_change_l992_99260

theorem restaurant_ratio_change (initial_cooks : ℕ) (initial_waiters : ℕ) 
  (additional_waiters : ℕ) :
  initial_cooks = 9 →
  initial_cooks * 10 = initial_waiters * 3 →
  additional_waiters = 12 →
  (initial_cooks : ℚ) / (initial_waiters + additional_waiters : ℚ) = 3 / 14 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_ratio_change_l992_99260


namespace NUMINAMATH_CALUDE_niles_collection_l992_99289

/-- The total amount collected by Niles from the book club members -/
def total_collected (num_members : ℕ) (snack_fee : ℕ) (num_hardcover : ℕ) (hardcover_price : ℕ) (num_paperback : ℕ) (paperback_price : ℕ) : ℕ :=
  num_members * (snack_fee + num_hardcover * hardcover_price + num_paperback * paperback_price)

/-- Theorem stating the total amount collected by Niles -/
theorem niles_collection : 
  total_collected 6 150 6 30 6 12 = 2412 := by
  sorry

end NUMINAMATH_CALUDE_niles_collection_l992_99289


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l992_99222

/-- Proves that the initial ratio of milk to water is 3:2 given the conditions -/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (new_ratio_milk : ℝ) 
  (new_ratio_water : ℝ) 
  (h1 : total_volume = 155) 
  (h2 : added_water = 62) 
  (h3 : new_ratio_milk = 3) 
  (h4 : new_ratio_water = 4) : 
  ∃ (initial_milk initial_water : ℝ), 
    initial_milk + initial_water = total_volume ∧ 
    initial_milk / (initial_water + added_water) = new_ratio_milk / new_ratio_water ∧
    initial_milk / initial_water = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l992_99222


namespace NUMINAMATH_CALUDE_toothpicks_count_l992_99292

/-- The number of small triangles in a row, starting from the base --/
def num_triangles_in_row (n : ℕ) : ℕ := 2500 - n + 1

/-- The total number of small triangles in the large triangle --/
def total_small_triangles : ℕ := (2500 * 2501) / 2

/-- The number of toothpicks needed for the interior and remaining exterior of the large triangle --/
def toothpicks_needed : ℕ := ((3 * total_small_triangles) / 2) + 2 * 2500

theorem toothpicks_count : toothpicks_needed = 4694375 := by sorry

end NUMINAMATH_CALUDE_toothpicks_count_l992_99292


namespace NUMINAMATH_CALUDE_inequality_proof_l992_99243

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  2 * (a ^ (1/2)) + 3 * (b ^ (1/3)) ≥ 5 * ((a * b) ^ (1/5)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l992_99243


namespace NUMINAMATH_CALUDE_workers_in_first_group_l992_99228

/-- Given two groups of workers building walls, this theorem proves the number of workers in the first group. -/
theorem workers_in_first_group 
  (wall_length_1 : ℝ) 
  (days_1 : ℝ) 
  (wall_length_2 : ℝ) 
  (days_2 : ℝ) 
  (workers_2 : ℕ) 
  (h1 : wall_length_1 = 66) 
  (h2 : days_1 = 12) 
  (h3 : wall_length_2 = 189.2) 
  (h4 : days_2 = 8) 
  (h5 : workers_2 = 86) :
  ∃ (workers_1 : ℕ), workers_1 = 57 ∧ 
    (workers_1 : ℝ) * days_1 * wall_length_2 = (workers_2 : ℝ) * days_2 * wall_length_1 :=
by sorry

end NUMINAMATH_CALUDE_workers_in_first_group_l992_99228


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l992_99203

theorem lcm_from_product_and_hcf (a b : ℕ+) (h1 : a * b = 987153000) (h2 : Nat.gcd a b = 440) :
  Nat.lcm a b = 2243525 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l992_99203


namespace NUMINAMATH_CALUDE_ratio_A_B_between_zero_and_one_l992_99288

def A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
def B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

theorem ratio_A_B_between_zero_and_one : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_A_B_between_zero_and_one_l992_99288


namespace NUMINAMATH_CALUDE_geometric_progression_condition_l992_99253

/-- Given real numbers a, b, c with b < 0, prove that b^2 = ac is necessary and 
    sufficient for a, b, c to form a geometric progression -/
theorem geometric_progression_condition (a b c : ℝ) (h : b < 0) :
  (b^2 = a*c) ↔ ∃ r : ℝ, (r ≠ 0 ∧ b = a*r ∧ c = b*r) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_condition_l992_99253


namespace NUMINAMATH_CALUDE_orange_distribution_l992_99210

theorem orange_distribution (total_oranges : ℕ) (bad_oranges : ℕ) (num_students : ℕ) 
    (h1 : total_oranges = 108)
    (h2 : bad_oranges = 36)
    (h3 : num_students = 12)
    (h4 : bad_oranges < total_oranges) :
  (total_oranges / num_students) - ((total_oranges - bad_oranges) / num_students) = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l992_99210


namespace NUMINAMATH_CALUDE_trig_simplification_l992_99227

theorem trig_simplification :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_trig_simplification_l992_99227


namespace NUMINAMATH_CALUDE_line_product_l992_99233

/-- Given a line y = mx + b passing through points (0, -3) and (3, 6), prove that mb = -9 -/
theorem line_product (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b ∧ 
  (-3 : ℝ) = m * 0 + b ∧ 
  (6 : ℝ) = m * 3 + b → 
  m * b = -9 := by
  sorry

end NUMINAMATH_CALUDE_line_product_l992_99233


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l992_99257

/-- Given a geometric sequence with first three terms 32, -48, and 72,
    prove that the common ratio is -3/2 and the fourth term is -108 -/
theorem geometric_sequence_proof (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 32) (h₂ : a₂ = -48) (h₃ : a₃ = 72) :
  ∃ (r : ℚ), r = -3/2 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₃ * r = -108 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l992_99257


namespace NUMINAMATH_CALUDE_volume_alteration_percentage_l992_99285

def original_volume : ℝ := 20 * 15 * 12

def removed_volume : ℝ := 4 * (4 * 4 * 4)

def added_volume : ℝ := 4 * (2 * 2 * 2)

def net_volume_change : ℝ := removed_volume - added_volume

theorem volume_alteration_percentage :
  (net_volume_change / original_volume) * 100 = 6.22 := by
  sorry

end NUMINAMATH_CALUDE_volume_alteration_percentage_l992_99285


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l992_99255

theorem quadratic_inequality_range (x : ℝ) (h : x^2 - 3*x + 2 < 0) :
  ∃ y ∈ Set.Ioo (-0.25 : ℝ) 0, y = x^2 - 3*x + 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l992_99255


namespace NUMINAMATH_CALUDE_cottage_build_time_l992_99295

/-- Represents the time (in days) it takes to build a cottage -/
def build_time (num_builders : ℕ) (days : ℕ) : Prop :=
  num_builders * days = 24

theorem cottage_build_time :
  build_time 3 8 → build_time 6 4 := by sorry

end NUMINAMATH_CALUDE_cottage_build_time_l992_99295


namespace NUMINAMATH_CALUDE_compare_abc_l992_99271

def tower_exp (base : ℕ) : ℕ → ℕ
| 0 => 1
| (n + 1) => base ^ (tower_exp base n)

def a : ℕ := tower_exp 3 25
def b : ℕ := tower_exp 4 20
def c : ℕ := 5^5

theorem compare_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_compare_abc_l992_99271


namespace NUMINAMATH_CALUDE_yogurt_combinations_l992_99231

theorem yogurt_combinations (n_flavors : ℕ) (n_toppings : ℕ) : 
  n_flavors = 4 → n_toppings = 8 → 
  n_flavors * (n_toppings.choose 3) = 224 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l992_99231


namespace NUMINAMATH_CALUDE_cats_remaining_after_sale_l992_99241

theorem cats_remaining_after_sale
  (initial_siamese : ℕ)
  (initial_persian : ℕ)
  (initial_house : ℕ)
  (sold_siamese : ℕ)
  (sold_persian : ℕ)
  (sold_house : ℕ)
  (h1 : initial_siamese = 20)
  (h2 : initial_persian = 12)
  (h3 : initial_house = 8)
  (h4 : sold_siamese = 8)
  (h5 : sold_persian = 5)
  (h6 : sold_house = 3) :
  initial_siamese + initial_persian + initial_house -
  (sold_siamese + sold_persian + sold_house) = 24 :=
by sorry

end NUMINAMATH_CALUDE_cats_remaining_after_sale_l992_99241


namespace NUMINAMATH_CALUDE_circle_radius_l992_99278

theorem circle_radius (x y : ℝ) : 
  x^2 + y^2 - 2*x + 4*y = 0 → ∃ (h k r : ℝ), r = Real.sqrt 5 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l992_99278


namespace NUMINAMATH_CALUDE_angle_measure_l992_99286

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l992_99286


namespace NUMINAMATH_CALUDE_gratuity_calculation_correct_l992_99251

/-- Calculates the gratuity for a restaurant bill given the individual dish prices,
    discount rate, sales tax rate, and tip rate. -/
def calculate_gratuity (prices : List ℝ) (discount_rate sales_tax_rate tip_rate : ℝ) : ℝ :=
  let total_before_discount := prices.sum
  let discounted_total := total_before_discount * (1 - discount_rate)
  let total_with_tax := discounted_total * (1 + sales_tax_rate)
  total_with_tax * tip_rate

/-- The gratuity calculated for the given restaurant bill is correct. -/
theorem gratuity_calculation_correct :
  let prices := [21, 15, 26, 13, 20]
  let discount_rate := 0.15
  let sales_tax_rate := 0.08
  let tip_rate := 0.18
  calculate_gratuity prices discount_rate sales_tax_rate tip_rate = 15.70 := by
  sorry

#eval calculate_gratuity [21, 15, 26, 13, 20] 0.15 0.08 0.18

end NUMINAMATH_CALUDE_gratuity_calculation_correct_l992_99251


namespace NUMINAMATH_CALUDE_lambda_positive_infinite_lambda_negative_infinite_l992_99204

/-- Definition of Ω(n) -/
def Omega (n : ℕ) : ℕ := sorry

/-- Definition of λ(n) -/
def lambda (n : ℕ) : Int := (-1) ^ (Omega n)

/-- The set of positive integers n such that λ(n) = λ(n+1) = 1 is infinite -/
theorem lambda_positive_infinite : Set.Infinite {n : ℕ | lambda n = 1 ∧ lambda (n + 1) = 1} := by sorry

/-- The set of positive integers n such that λ(n) = λ(n+1) = -1 is infinite -/
theorem lambda_negative_infinite : Set.Infinite {n : ℕ | lambda n = -1 ∧ lambda (n + 1) = -1} := by sorry

end NUMINAMATH_CALUDE_lambda_positive_infinite_lambda_negative_infinite_l992_99204


namespace NUMINAMATH_CALUDE_valerie_light_bulb_shortage_l992_99264

structure LightBulb where
  price : Float
  quantity : Nat

def small_bulb : LightBulb := { price := 8.75, quantity := 3 }
def medium_bulb : LightBulb := { price := 11.25, quantity := 4 }
def large_bulb : LightBulb := { price := 15.50, quantity := 3 }
def extra_small_bulb : LightBulb := { price := 6.10, quantity := 4 }

def budget : Float := 120.00

def total_cost : Float :=
  small_bulb.price * small_bulb.quantity.toFloat +
  medium_bulb.price * medium_bulb.quantity.toFloat +
  large_bulb.price * large_bulb.quantity.toFloat +
  extra_small_bulb.price * extra_small_bulb.quantity.toFloat

theorem valerie_light_bulb_shortage :
  total_cost - budget = 22.15 := by
  sorry


end NUMINAMATH_CALUDE_valerie_light_bulb_shortage_l992_99264


namespace NUMINAMATH_CALUDE_sequence_a_increasing_l992_99265

def a (n : ℕ) : ℚ := (n - 1 : ℚ) / (n + 1 : ℚ)

theorem sequence_a_increasing : ∀ n ≥ 2, a n < a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_increasing_l992_99265


namespace NUMINAMATH_CALUDE_initial_cells_theorem_l992_99273

/-- Calculates the number of cells after one hour given the initial number of cells -/
def cellsAfterOneHour (initialCells : ℕ) : ℕ :=
  2 * (initialCells - 2)

/-- Calculates the number of cells after n hours given the initial number of cells -/
def cellsAfterNHours (initialCells : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initialCells
  | n + 1 => cellsAfterOneHour (cellsAfterNHours initialCells n)

/-- Theorem stating that 9 initial cells result in 164 cells after 5 hours -/
theorem initial_cells_theorem :
  cellsAfterNHours 9 5 = 164 :=
by sorry

end NUMINAMATH_CALUDE_initial_cells_theorem_l992_99273


namespace NUMINAMATH_CALUDE_larger_integer_value_l992_99212

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * (b : ℕ) = 189) :
  max a b = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l992_99212


namespace NUMINAMATH_CALUDE_vitamin_a_intake_in_grams_l992_99216

/-- Conversion factor from grams to milligrams -/
def gram_to_mg : ℝ := 1000

/-- Conversion factor from milligrams to micrograms -/
def mg_to_μg : ℝ := 1000

/-- Daily intake of vitamin A for adult women in micrograms -/
def vitamin_a_intake : ℝ := 750

/-- Theorem stating that 750 micrograms is equal to 7.5 × 10^-4 grams -/
theorem vitamin_a_intake_in_grams :
  (vitamin_a_intake / (gram_to_mg * mg_to_μg)) = 7.5e-4 := by
  sorry

end NUMINAMATH_CALUDE_vitamin_a_intake_in_grams_l992_99216


namespace NUMINAMATH_CALUDE_extremal_point_and_range_l992_99284

noncomputable def f (a x : ℝ) : ℝ := (x - a)^2 * Real.log x

theorem extremal_point_and_range (e : ℝ) (h_e : Real.exp 1 = e) :
  (∃ a : ℝ, (deriv (f a)) e = 0 ↔ (a = e ∨ a = 3*e)) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Ioc 0 (3*e) → f a x ≤ 4*e^2) ↔ 
    a ∈ Set.Icc (3*e - 2*e / Real.sqrt (Real.log (3*e))) (3*e)) :=
by sorry

end NUMINAMATH_CALUDE_extremal_point_and_range_l992_99284


namespace NUMINAMATH_CALUDE_absolute_value_theorem_l992_99205

theorem absolute_value_theorem (x : ℝ) (h : x < -1) :
  |x - Real.sqrt ((x + 2)^2)| = -2*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_theorem_l992_99205


namespace NUMINAMATH_CALUDE_solution_pairs_l992_99234

theorem solution_pairs (x y a n m : ℕ) (h1 : x + y = a^n) (h2 : x^2 + y^2 = a^m) :
  ∃ k : ℕ, x = 2^k ∧ y = 2^k := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l992_99234


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l992_99256

theorem quadratic_inequality_solution (b : ℝ) : 
  (∀ x, x^2 - b*x + 6 < 0 ↔ 2 < x ∧ x < 3) → b = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l992_99256


namespace NUMINAMATH_CALUDE_molly_gift_cost_per_package_l992_99209

/-- The cost per package for Molly's Christmas gifts --/
def cost_per_package (total_relatives : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / total_relatives

/-- Theorem: The cost per package for Molly's Christmas gifts is $5 --/
theorem molly_gift_cost_per_package :
  let total_relatives : ℕ := 14
  let total_cost : ℚ := 70
  cost_per_package total_relatives total_cost = 5 := by
  sorry


end NUMINAMATH_CALUDE_molly_gift_cost_per_package_l992_99209


namespace NUMINAMATH_CALUDE_women_per_table_l992_99230

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 6 →
  men_per_table = 5 →
  total_customers = 48 →
  ∃ (women_per_table : ℕ),
    women_per_table * num_tables + men_per_table * num_tables = total_customers ∧
    women_per_table = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_women_per_table_l992_99230


namespace NUMINAMATH_CALUDE_range_of_3a_minus_b_l992_99296

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : -1 < a + b ∧ a + b < 3) 
  (h2 : 2 < a - b ∧ a - b < 4) : 
  (∀ x, 3*a - b ≥ x → x ≥ 3) ∧ 
  (∀ y, 3*a - b ≤ y → y ≤ 11) :=
sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_b_l992_99296


namespace NUMINAMATH_CALUDE_quadratic_diophantine_bound_l992_99293

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The number of integer solutions to a quadratic Diophantine equation -/
def num_solutions (A B C D E : ℤ) : ℕ := sorry

theorem quadratic_diophantine_bound
  (A B C D E : ℤ)
  (hB : B ≠ 0)
  (hF : A * D^2 - B * C * D + B^2 * E ≠ 0) :
  num_solutions A B C D E ≤ 2 * num_divisors (Int.natAbs (A * D^2 - B * C * D + B^2 * E)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_diophantine_bound_l992_99293


namespace NUMINAMATH_CALUDE_equivalent_operation_l992_99252

theorem equivalent_operation (x : ℝ) : (x / (5/6)) * (4/7) = x * (24/35) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l992_99252


namespace NUMINAMATH_CALUDE_book_selection_theorem_l992_99215

theorem book_selection_theorem (math_books : Nat) (physics_books : Nat) : 
  math_books = 3 → physics_books = 2 → math_books * physics_books = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l992_99215


namespace NUMINAMATH_CALUDE_dollar_symmetric_sum_l992_99235

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem dollar_symmetric_sum (x y : ℝ) : dollar (x + y) (y + x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dollar_symmetric_sum_l992_99235


namespace NUMINAMATH_CALUDE_coin_toss_probability_l992_99283

theorem coin_toss_probability : 
  let n : ℕ := 5  -- Total number of coins
  let k : ℕ := 3  -- Number of tails (or heads, whichever is smaller)
  let p : ℚ := 1/2  -- Probability of getting tails (or heads) on a single toss
  (n.choose k) * p^n = 5/16 :=
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l992_99283


namespace NUMINAMATH_CALUDE_smallest_y_value_l992_99218

theorem smallest_y_value : ∃ y : ℝ, 
  (∀ z : ℝ, 3 * z^2 + 27 * z - 90 = z * (z + 15) → y ≤ z) ∧ 
  (3 * y^2 + 27 * y - 90 = y * (y + 15)) ∧ 
  y = -9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_value_l992_99218


namespace NUMINAMATH_CALUDE_work_completion_time_l992_99276

/-- Given that:
    1. Ravi can do a piece of work in 15 days
    2. Ravi and another person together can do the work in 10 days
    Prove that the other person can do the work alone in 30 days -/
theorem work_completion_time (ravi_time : ℝ) (joint_time : ℝ) (other_time : ℝ) :
  ravi_time = 15 →
  joint_time = 10 →
  (1 / ravi_time + 1 / other_time = 1 / joint_time) →
  other_time = 30 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l992_99276


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l992_99248

theorem difference_of_squares_special_case : (727 : ℤ) * 727 - 726 * 728 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l992_99248


namespace NUMINAMATH_CALUDE_function_intersects_axes_l992_99219

-- Define the function
def f (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem function_intersects_axes : 
  (∃ x : ℝ, x < 0 ∧ f x = 0) ∧ 
  (∃ y : ℝ, y > 0 ∧ f 0 = y) := by
  sorry

end NUMINAMATH_CALUDE_function_intersects_axes_l992_99219


namespace NUMINAMATH_CALUDE_good_function_k_range_l992_99220

/-- A function f is "good" if it's monotonic on its domain D and there exists [m,n] ⊆ D
    such that the range of f on [m,n] is [½m, ½n] -/
def is_good_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  Monotone f ∧ ∃ m n, m ≤ n ∧ Set.Icc m n ⊆ D ∧
    Set.image f (Set.Icc m n) = Set.Icc (m/2) (n/2)

/-- The logarithm function with base a -/
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The function f(x) = logₐ(aˣ + k) -/
noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := log_base a (a^x + k)

theorem good_function_k_range (a : ℝ) (h_a : a > 0 ∧ a ≠ 1) :
  ∃ D : Set ℝ, ∀ k : ℝ, is_good_function (f a k) D ↔ k ∈ Set.Ioo 0 (1/4) :=
sorry

end NUMINAMATH_CALUDE_good_function_k_range_l992_99220


namespace NUMINAMATH_CALUDE_tom_remaining_seashells_l992_99232

def initial_seashells : ℕ := 5
def seashells_given_away : ℕ := 2

theorem tom_remaining_seashells : 
  initial_seashells - seashells_given_away = 3 := by sorry

end NUMINAMATH_CALUDE_tom_remaining_seashells_l992_99232


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l992_99282

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem complex_fraction_simplification :
  (1 + i) / (1 - i) = i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l992_99282


namespace NUMINAMATH_CALUDE_classroom_chairs_l992_99269

theorem classroom_chairs (blue_chairs : ℕ) (green_chairs : ℕ) (white_chairs : ℕ) :
  blue_chairs = 10 →
  green_chairs = 3 * blue_chairs →
  white_chairs = blue_chairs + green_chairs - 13 →
  blue_chairs + green_chairs + white_chairs = 67 := by
sorry

end NUMINAMATH_CALUDE_classroom_chairs_l992_99269


namespace NUMINAMATH_CALUDE_stating_reach_target_probability_approx_l992_99244

/-- Represents the probability of winning in a single bet -/
def win_probability : ℝ := 0.1

/-- Represents the cost of a single bet -/
def bet_cost : ℝ := 10

/-- Represents the amount won in a single successful bet -/
def win_amount : ℝ := 30

/-- Represents the initial amount of money -/
def initial_amount : ℝ := 20

/-- Represents the target amount to reach -/
def target_amount : ℝ := 45

/-- 
Represents the probability of reaching the target amount 
starting from the initial amount through a series of bets
-/
noncomputable def reach_target_probability : ℝ := sorry

/-- 
Theorem stating that the probability of reaching the target amount 
is approximately 0.033
-/
theorem reach_target_probability_approx : 
  |reach_target_probability - 0.033| < 0.001 := by sorry

end NUMINAMATH_CALUDE_stating_reach_target_probability_approx_l992_99244


namespace NUMINAMATH_CALUDE_marbles_problem_l992_99238

theorem marbles_problem (initial_marbles given_marbles remaining_marbles : ℕ) : 
  given_marbles = 8 → remaining_marbles = 24 → initial_marbles = given_marbles + remaining_marbles →
  initial_marbles = 32 := by
sorry

end NUMINAMATH_CALUDE_marbles_problem_l992_99238


namespace NUMINAMATH_CALUDE_converse_negation_equivalence_triangle_angles_arithmetic_sequence_inequality_system_not_equivalent_squared_inequality_implication_l992_99299

-- 1. Converse and negation of a proposition
theorem converse_negation_equivalence (P Q : Prop) : 
  (P → Q) ↔ ¬Q → ¬P := by sorry

-- 2. Triangle angles forming arithmetic sequence
theorem triangle_angles_arithmetic_sequence (A B C : ℝ) :
  (A + B + C = 180) → (B = 60 ↔ 2 * B = A + C) := by sorry

-- 3. Inequality system counterexample
theorem inequality_system_not_equivalent :
  ∃ x y : ℝ, (x + y > 3 ∧ x * y > 2) ∧ ¬(x > 1 ∧ y > 2) := by sorry

-- 4. Squared inequality implication
theorem squared_inequality_implication (a b : ℝ) :
  (∀ m : ℝ, a * m^2 < b * m^2 → a < b) ∧
  ¬(∀ a b : ℝ, a < b → ∀ m : ℝ, a * m^2 < b * m^2) := by sorry

end NUMINAMATH_CALUDE_converse_negation_equivalence_triangle_angles_arithmetic_sequence_inequality_system_not_equivalent_squared_inequality_implication_l992_99299


namespace NUMINAMATH_CALUDE_department_store_sales_multiple_l992_99207

theorem department_store_sales_multiple (M : ℝ) :
  (∀ (A : ℝ), A > 0 →
    M * A = 0.15384615384615385 * (11 * A + M * A)) →
  M = 2 := by
sorry

end NUMINAMATH_CALUDE_department_store_sales_multiple_l992_99207


namespace NUMINAMATH_CALUDE_inequality_solution_set_l992_99225

theorem inequality_solution_set :
  {x : ℝ | (1 : ℝ) / (x - 1) < -1} = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l992_99225


namespace NUMINAMATH_CALUDE_stratified_sample_category_a_l992_99268

/-- Represents the number of students in each school category -/
structure SchoolCategories where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the number of students to be sampled from Category A schools
    using stratified sampling -/
def sampleSizeA (categories : SchoolCategories) (totalSample : ℕ) : ℕ :=
  (categories.a * totalSample) / (categories.a + categories.b + categories.c)

/-- Theorem stating that for the given school categories and sample size,
    the number of students to be selected from Category A is 200 -/
theorem stratified_sample_category_a 
  (categories : SchoolCategories)
  (h1 : categories.a = 2000)
  (h2 : categories.b = 3000)
  (h3 : categories.c = 4000)
  (totalSample : ℕ)
  (h4 : totalSample = 900) :
  sampleSizeA categories totalSample = 200 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_category_a_l992_99268


namespace NUMINAMATH_CALUDE_monica_classes_count_l992_99272

/-- Represents the number of students in each of Monica's classes -/
def class_sizes : List Nat := [20, 25, 25, 10, 28, 28]

/-- The total number of students Monica sees each day -/
def total_students : Nat := 136

/-- Theorem stating that Monica has 6 classes per day -/
theorem monica_classes_count : List.length class_sizes = 6 ∧ List.sum class_sizes = total_students := by
  sorry

end NUMINAMATH_CALUDE_monica_classes_count_l992_99272


namespace NUMINAMATH_CALUDE_power_multiplication_calculate_3000_power_l992_99259

theorem power_multiplication (a : ℕ) (m n : ℕ) :
  a * (a ^ n) = a ^ (n + 1) :=
by sorry

theorem calculate_3000_power :
  3000 * (3000 ^ 1999) = 3000 ^ 2000 :=
by sorry

end NUMINAMATH_CALUDE_power_multiplication_calculate_3000_power_l992_99259


namespace NUMINAMATH_CALUDE_det_A_zero_for_n_gt_five_l992_99254

def A (n : ℕ) : Matrix (Fin n) (Fin n) ℤ :=
  λ i j => (i.val^j.val + j.val^i.val) % 3

theorem det_A_zero_for_n_gt_five (n : ℕ) (h : n > 5) :
  Matrix.det (A n) = 0 :=
sorry

end NUMINAMATH_CALUDE_det_A_zero_for_n_gt_five_l992_99254


namespace NUMINAMATH_CALUDE_unique_solution_condition_l992_99270

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 6) = -52 + k * x) ↔ 
  (k = 4 * Real.sqrt 21 - 14 ∨ k = -4 * Real.sqrt 21 - 14) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l992_99270


namespace NUMINAMATH_CALUDE_ellipse_and_circle_l992_99279

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  focal_length : ℝ
  short_axis_length : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_focal : focal_length = 2 * Real.sqrt 6
  h_short : short_axis_length = 2 * Real.sqrt 2

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The line that intersects the ellipse -/
def intersecting_line (x y : ℝ) : Prop :=
  y = x + 2

/-- Main theorem about the ellipse and its intersecting circle -/
theorem ellipse_and_circle (e : Ellipse) :
  (∀ x y, ellipse_equation e x y ↔ x^2 / 8 + y^2 / 2 = 1) ∧
  (∃ A B : ℝ × ℝ,
    ellipse_equation e A.1 A.2 ∧
    ellipse_equation e B.1 B.2 ∧
    intersecting_line A.1 A.2 ∧
    intersecting_line B.1 B.2 ∧
    ∀ x y, (x + 8/5)^2 + (y - 2/5)^2 = 48/25 ↔
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
        x = (1 - t) * A.1 + t * B.1 ∧
        y = (1 - t) * A.2 + t * B.2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_l992_99279


namespace NUMINAMATH_CALUDE_prime_power_sum_implies_power_of_three_l992_99262

theorem prime_power_sum_implies_power_of_three (n : ℕ) :
  Nat.Prime (1 + 2^n + 4^n) → ∃ k : ℕ+, n = 3^(k : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_implies_power_of_three_l992_99262


namespace NUMINAMATH_CALUDE_rahul_savings_l992_99298

/-- Rahul's savings problem -/
theorem rahul_savings (nsc ppf : ℕ) : 
  (3 * nsc = 2 * ppf) →  -- One-third of NSC equals one-half of PPF
  (nsc + ppf = 180000) → -- Total savings
  (ppf = 72000) :=       -- PPF savings to prove
by sorry

end NUMINAMATH_CALUDE_rahul_savings_l992_99298


namespace NUMINAMATH_CALUDE_biased_coin_probability_l992_99213

theorem biased_coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1/2) :
  (Nat.choose 6 2 : ℝ) * p^2 * (1 - p)^4 = 1/8 → p = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l992_99213


namespace NUMINAMATH_CALUDE_find_x_and_y_l992_99297

theorem find_x_and_y :
  ∃ (x y : ℚ), 3 * (2 * x + 9 * y) = 75 ∧ x + y = 10 ∧ x = 65/7 ∧ y = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_find_x_and_y_l992_99297


namespace NUMINAMATH_CALUDE_roots_transformation_l992_99249

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + r₁ + 6 = 0) → 
  (r₂^3 - 4*r₂^2 + r₂ + 6 = 0) → 
  (r₃^3 - 4*r₃^2 + r₃ + 6 = 0) → 
  ∀ x, (x - 3*r₁) * (x - 3*r₂) * (x - 3*r₃) = x^3 - 12*x^2 + 9*x + 162 :=
by sorry

end NUMINAMATH_CALUDE_roots_transformation_l992_99249


namespace NUMINAMATH_CALUDE_three_draw_probability_l992_99247

def blue_chips : ℕ := 6
def yellow_chips : ℕ := 4
def total_chips : ℕ := blue_chips + yellow_chips

def prob_different_colors : ℚ := 72 / 625

theorem three_draw_probability :
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_yellow : ℚ := yellow_chips / total_chips
  let prob_diff_first_second : ℚ := prob_blue * prob_yellow + prob_yellow * prob_blue
  prob_diff_first_second * (prob_blue * prob_yellow + prob_yellow * prob_blue) = prob_different_colors :=
by sorry

end NUMINAMATH_CALUDE_three_draw_probability_l992_99247


namespace NUMINAMATH_CALUDE_inequality_proof_l992_99223

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 4) : 
  |a*c + b*d| ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l992_99223


namespace NUMINAMATH_CALUDE_bus_ride_cost_proof_l992_99245

def bus_ride_cost : ℚ := 1.40
def train_ride_cost : ℚ := bus_ride_cost + 6.85
def combined_cost : ℚ := 9.65
def price_multiple : ℚ := 0.35

theorem bus_ride_cost_proof :
  (train_ride_cost = bus_ride_cost + 6.85) ∧
  (train_ride_cost + bus_ride_cost = combined_cost) ∧
  (∃ n : ℕ, bus_ride_cost = n * price_multiple) ∧
  (∃ m : ℕ, train_ride_cost = m * price_multiple) →
  bus_ride_cost = 1.40 :=
by sorry

end NUMINAMATH_CALUDE_bus_ride_cost_proof_l992_99245


namespace NUMINAMATH_CALUDE_c_younger_than_a_l992_99263

-- Define variables for the ages of A, B, and C
variable (a b c : ℕ)

-- Define the condition given in the problem
def age_difference : Prop := a + b = b + c + 11

-- Theorem to prove
theorem c_younger_than_a (h : age_difference a b c) : a - c = 11 := by
  sorry

end NUMINAMATH_CALUDE_c_younger_than_a_l992_99263


namespace NUMINAMATH_CALUDE_opposite_of_negative_seven_l992_99274

-- Definition of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_seven :
  opposite (-7) = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_seven_l992_99274


namespace NUMINAMATH_CALUDE_square_of_binomial_l992_99294

theorem square_of_binomial (a : ℚ) :
  (∃ b : ℚ, ∀ x : ℚ, 9 * x^2 + 15 * x + a = (3 * x + b)^2) → a = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l992_99294


namespace NUMINAMATH_CALUDE_polar_to_rectangular_on_circle_l992_99287

/-- Proves that the point (5, 3π/4) in polar coordinates lies on the circle x^2 + y^2 = 25 when converted to rectangular coordinates. -/
theorem polar_to_rectangular_on_circle :
  let r : ℝ := 5
  let θ : ℝ := 3 * Real.pi / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x^2 + y^2 = 25 := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_on_circle_l992_99287


namespace NUMINAMATH_CALUDE_fraction_sum_and_lcd_l992_99217

theorem fraction_sum_and_lcd : 
  let fractions : List ℚ := [1/2, 1/3, 1/4, 1/5, 1/6, 1/8, 1/9]
  let lcd := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)))))
  lcd = 360 ∧ fractions.sum = 607 / 360 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_lcd_l992_99217


namespace NUMINAMATH_CALUDE_peanuts_added_l992_99206

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) 
  (h1 : initial_peanuts = 4)
  (h2 : final_peanuts = 10) :
  final_peanuts - initial_peanuts = 6 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_added_l992_99206


namespace NUMINAMATH_CALUDE_fisherman_catch_l992_99280

/-- The number of bass caught by the fisherman -/
def bass : ℕ := 32

/-- The number of trout caught by the fisherman -/
def trout : ℕ := bass / 4

/-- The number of bluegill caught by the fisherman -/
def bluegill : ℕ := 2 * bass

/-- The total number of fish caught by the fisherman -/
def total_fish : ℕ := 104

theorem fisherman_catch :
  bass + trout + bluegill = total_fish ∧
  trout = bass / 4 ∧
  bluegill = 2 * bass :=
sorry

end NUMINAMATH_CALUDE_fisherman_catch_l992_99280


namespace NUMINAMATH_CALUDE_ladder_problem_l992_99214

theorem ladder_problem (ladder_length height_on_wall base_distance : ℝ) : 
  ladder_length = 13 ∧ height_on_wall = 12 ∧ 
  ladder_length^2 = height_on_wall^2 + base_distance^2 → 
  base_distance = 5 := by
sorry

end NUMINAMATH_CALUDE_ladder_problem_l992_99214


namespace NUMINAMATH_CALUDE_earth_surface_available_for_living_l992_99261

theorem earth_surface_available_for_living : 
  let earth_surface : ℝ := 1
  let land_fraction : ℝ := 1 / 3
  let inhabitable_fraction : ℝ := 1 / 4
  let residential_fraction : ℝ := 0.6
  earth_surface * land_fraction * inhabitable_fraction * residential_fraction = 1 / 20 :=
by sorry

end NUMINAMATH_CALUDE_earth_surface_available_for_living_l992_99261


namespace NUMINAMATH_CALUDE_can_obtain_next_number_l992_99240

/-- Represents the allowed operations on a number -/
inductive Operation
  | AddNine : Operation
  | DeleteOne : Operation

/-- Applies a sequence of operations to a number -/
def applyOperations (a : ℕ) (ops : List Operation) : ℕ := sorry

/-- Theorem stating that A+1 can always be obtained from A using the allowed operations -/
theorem can_obtain_next_number (A : ℕ) : 
  A > 0 → ∃ (ops : List Operation), applyOperations A ops = A + 1 := by sorry

end NUMINAMATH_CALUDE_can_obtain_next_number_l992_99240


namespace NUMINAMATH_CALUDE_kelvin_winning_strategy_l992_99208

/-- Represents a player in the game -/
inductive Player
| Kelvin
| Alex

/-- Represents a single move in the game -/
structure Move where
  digit : Nat
  position : Nat

/-- Represents the state of the game -/
structure GameState where
  number : Nat
  currentPlayer : Player

/-- A strategy for Kelvin -/
def KelvinStrategy := GameState → Move

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Bool :=
  sorry

/-- Plays the game given Kelvin's strategy and Alex's moves -/
def playGame (strategy : KelvinStrategy) (alexMoves : List Move) : Bool :=
  sorry

/-- Theorem stating that Kelvin has a winning strategy -/
theorem kelvin_winning_strategy :
  ∃ (strategy : KelvinStrategy),
    ∀ (alexMoves : List Move),
      ¬(playGame strategy alexMoves) :=
sorry

end NUMINAMATH_CALUDE_kelvin_winning_strategy_l992_99208


namespace NUMINAMATH_CALUDE_non_zero_digits_count_l992_99275

def expression : ℚ := 180 / (2^4 * 5^6 * 3^2)

def count_non_zero_decimal_digits (q : ℚ) : ℕ :=
  sorry

theorem non_zero_digits_count : count_non_zero_decimal_digits expression = 1 := by
  sorry

end NUMINAMATH_CALUDE_non_zero_digits_count_l992_99275


namespace NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l992_99250

theorem min_sum_of_parallel_vectors (x y : ℝ) : 
  x > 0 → y > 0 → 
  (∃ (k : ℝ), k ≠ 0 ∧ (1 - x, x) = k • (1, -y)) →
  4 ≤ x + y ∧ (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    (∃ (k : ℝ), k ≠ 0 ∧ (1 - x₀, x₀) = k • (1, -y₀)) ∧ 
    x₀ + y₀ = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l992_99250


namespace NUMINAMATH_CALUDE_other_stamp_price_l992_99224

-- Define the total number of stamps
def total_stamps : ℕ := 75

-- Define the total amount received in cents
def total_amount : ℕ := 480

-- Define the price of the known stamp type
def known_stamp_price : ℕ := 8

-- Define the number of stamps sold of one kind
def stamps_of_one_kind : ℕ := 40

-- Define the function to calculate the price of the unknown stamp type
def unknown_stamp_price (x : ℕ) : Prop :=
  (stamps_of_one_kind * known_stamp_price + (total_stamps - stamps_of_one_kind) * x = total_amount) ∧
  (x > 0) ∧ (x < known_stamp_price)

-- Theorem stating that the price of the unknown stamp type is 5 cents
theorem other_stamp_price : unknown_stamp_price 5 := by
  sorry

end NUMINAMATH_CALUDE_other_stamp_price_l992_99224


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l992_99226

theorem cubic_equation_solution (x : ℝ) : 
  x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l992_99226


namespace NUMINAMATH_CALUDE_sum_of_squares_l992_99267

theorem sum_of_squares (a b c d e f : ℤ) :
  (∀ x : ℝ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l992_99267


namespace NUMINAMATH_CALUDE_square_root_problem_l992_99221

theorem square_root_problem (m : ℝ) (x : ℝ) 
  (h1 : m > 0) 
  (h2 : Real.sqrt m = x + 1) 
  (h3 : Real.sqrt m = x - 3) : 
  m = 4 := by sorry

end NUMINAMATH_CALUDE_square_root_problem_l992_99221


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l992_99290

-- Define the intersection point
def intersection_point : ℝ × ℝ := (3, 2)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 3 * x - 2 * y + 4 = 0

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 4 * x - 3 * y - 7 = 0

-- Theorem for the parallel case
theorem parallel_line_equation :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (x, y) = intersection_point →
    (3 * x - 2 * y + m = 0 ↔ parallel_line x y) :=
sorry

-- Theorem for the perpendicular case
theorem perpendicular_line_equation :
  ∃ (n : ℝ), ∀ (x y : ℝ),
    (x, y) = intersection_point →
    (3 * x + 4 * y + n = 0 ↔ perpendicular_line x y) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l992_99290


namespace NUMINAMATH_CALUDE_wen_family_movie_cost_l992_99242

def ticket_cost (regular_price : ℚ) (discount : ℚ) : ℚ :=
  regular_price * (1 - discount)

theorem wen_family_movie_cost :
  let senior_price : ℚ := 6
  let senior_discount : ℚ := 1/4
  let children_discount : ℚ := 1/2
  let regular_price : ℚ := senior_price / (1 - senior_discount)
  let num_people_per_generation : ℕ := 2
  
  num_people_per_generation * senior_price +
  num_people_per_generation * regular_price +
  num_people_per_generation * (ticket_cost regular_price children_discount) = 36
  := by sorry

end NUMINAMATH_CALUDE_wen_family_movie_cost_l992_99242


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l992_99202

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let focal_distance := 2 * c
  let asymptote_slope := b / a
  let focus_to_asymptote_distance := b * c / Real.sqrt (a^2 + b^2)
  focus_to_asymptote_distance = (1/4) * focal_distance →
  c / a = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l992_99202


namespace NUMINAMATH_CALUDE_stating_max_squares_specific_cases_l992_99200

/-- 
Given a rectangular grid of dimensions m × n, this function calculates 
the maximum number of squares that can be cut along the grid lines.
-/
def max_squares (m n : ℕ) : ℕ := sorry

/--
Theorem stating that for specific grid dimensions (8, 11) and (8, 12),
the maximum number of squares that can be cut is 5.
-/
theorem max_squares_specific_cases : 
  (max_squares 8 11 = 5) ∧ (max_squares 8 12 = 5) := by sorry

end NUMINAMATH_CALUDE_stating_max_squares_specific_cases_l992_99200


namespace NUMINAMATH_CALUDE_midpoint_octahedron_volume_ratio_l992_99281

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  -- We don't need to specify the vertices or edge length here

/-- An octahedron formed by the midpoints of a tetrahedron's edges -/
def midpoint_octahedron (t : RegularTetrahedron) : Set (Fin 4 → ℝ) :=
  sorry

/-- The volume of a regular tetrahedron -/
def volume_tetrahedron (t : RegularTetrahedron) : ℝ :=
  sorry

/-- The volume of the octahedron formed by midpoints -/
def volume_midpoint_octahedron (t : RegularTetrahedron) : ℝ :=
  sorry

/-- Theorem: The ratio of the volume of the midpoint octahedron to the volume of the regular tetrahedron is 3/16 -/
theorem midpoint_octahedron_volume_ratio (t : RegularTetrahedron) :
  volume_midpoint_octahedron t / volume_tetrahedron t = 3 / 16 :=
sorry

end NUMINAMATH_CALUDE_midpoint_octahedron_volume_ratio_l992_99281


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l992_99246

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 + 5*x - 24 = 0) ∧
  (∃ x : ℝ, 3*x^2 = 2*(2-x)) ∧
  (∀ x : ℝ, x^2 + 5*x - 24 = 0 ↔ (x = -8 ∨ x = 3)) ∧
  (∀ x : ℝ, 3*x^2 = 2*(2-x) ↔ (x = (-1 + Real.sqrt 13) / 3 ∨ x = (-1 - Real.sqrt 13) / 3)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l992_99246
