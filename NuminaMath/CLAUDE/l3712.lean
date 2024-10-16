import Mathlib

namespace NUMINAMATH_CALUDE_determinant_inequality_range_l3712_371234

theorem determinant_inequality_range (x : ℝ) : 
  (Matrix.det !![x + 3, x^2; 1, 4] < 0) ↔ (x ∈ Set.Iio (-2) ∪ Set.Ioi 6) := by
  sorry

end NUMINAMATH_CALUDE_determinant_inequality_range_l3712_371234


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3712_371202

/-- Given two vectors a and b in R², where a is parallel to (a - b), 
    prove that the magnitude of their sum is 3√5/2. -/
theorem vector_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), a = k • (a - b)) → 
  ‖a + b‖ = (3 * Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3712_371202


namespace NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l3712_371215

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l3712_371215


namespace NUMINAMATH_CALUDE_kay_age_is_32_l3712_371248

-- Define Kay's age and the number of siblings
def kay_age : ℕ := sorry
def num_siblings : ℕ := 14

-- Define the ages of the youngest and oldest siblings
def youngest_sibling_age : ℕ := kay_age / 2 - 5
def oldest_sibling_age : ℕ := 4 * youngest_sibling_age

-- State the theorem
theorem kay_age_is_32 :
  num_siblings = 14 ∧
  youngest_sibling_age = kay_age / 2 - 5 ∧
  oldest_sibling_age = 4 * youngest_sibling_age ∧
  oldest_sibling_age = 44 →
  kay_age = 32 :=
by sorry

end NUMINAMATH_CALUDE_kay_age_is_32_l3712_371248


namespace NUMINAMATH_CALUDE_cos_graph_transformation_l3712_371232

theorem cos_graph_transformation (x : ℝ) : 
  4 * Real.cos (2 * x) = 4 * Real.cos (2 * (x - π/8) + π/4) :=
by sorry

end NUMINAMATH_CALUDE_cos_graph_transformation_l3712_371232


namespace NUMINAMATH_CALUDE_min_sum_positive_reals_l3712_371213

theorem min_sum_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 3 * (1 / Real.rpow 162 (1/3)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_positive_reals_l3712_371213


namespace NUMINAMATH_CALUDE_expression_evaluation_l3712_371257

theorem expression_evaluation : 
  0.064^(-1/3) - (-7/9)^0 + ((-2)^3)^(1/3) - 16^(-0.75) = -5/8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3712_371257


namespace NUMINAMATH_CALUDE_probability_divisible_by_2_3_5_or_7_l3712_371260

theorem probability_divisible_by_2_3_5_or_7 : 
  let S : Finset ℕ := Finset.range 120
  let A : Finset ℕ := S.filter (fun n => n % 2 = 0)
  let B : Finset ℕ := S.filter (fun n => n % 3 = 0)
  let C : Finset ℕ := S.filter (fun n => n % 5 = 0)
  let D : Finset ℕ := S.filter (fun n => n % 7 = 0)
  (A ∪ B ∪ C ∪ D).card / S.card = 13 / 15 := by
sorry


end NUMINAMATH_CALUDE_probability_divisible_by_2_3_5_or_7_l3712_371260


namespace NUMINAMATH_CALUDE_employee_reduction_l3712_371286

theorem employee_reduction (original_employees : ℝ) (reduction_percentage : ℝ) : 
  original_employees = 243.75 → 
  reduction_percentage = 0.20 → 
  original_employees * (1 - reduction_percentage) = 195 := by
  sorry


end NUMINAMATH_CALUDE_employee_reduction_l3712_371286


namespace NUMINAMATH_CALUDE_existence_of_monochromatic_right_angled_pentagon_l3712_371203

-- Define a color type
inductive Color
| Red
| Yellow

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define a convex pentagon
def ConvexPentagon (p₁ p₂ p₃ p₄ p₅ : Point) : Prop := sorry

-- Define a right angle
def RightAngle (p₁ p₂ p₃ : Point) : Prop := sorry

-- Define the theorem
theorem existence_of_monochromatic_right_angled_pentagon :
  ∃ (p₁ p₂ p₃ p₄ p₅ : Point),
    ConvexPentagon p₁ p₂ p₃ p₄ p₅ ∧
    RightAngle p₁ p₂ p₃ ∧
    RightAngle p₂ p₃ p₄ ∧
    RightAngle p₃ p₄ p₅ ∧
    ((coloring p₁ = Color.Red ∧ coloring p₂ = Color.Red ∧ coloring p₃ = Color.Red ∧ 
      coloring p₄ = Color.Red ∧ coloring p₅ = Color.Red) ∨
     (coloring p₁ = Color.Yellow ∧ coloring p₂ = Color.Yellow ∧ coloring p₃ = Color.Yellow ∧ 
      coloring p₄ = Color.Yellow ∧ coloring p₅ = Color.Yellow)) :=
by sorry


end NUMINAMATH_CALUDE_existence_of_monochromatic_right_angled_pentagon_l3712_371203


namespace NUMINAMATH_CALUDE_age_of_35th_student_l3712_371252

/-- The age of the 35th student in a class, given the following conditions:
  - There are 35 students in total
  - The average age of all 35 students is 16.5 years
  - 10 students have an average age of 15.3 years
  - 17 students have an average age of 16.7 years
  - 6 students have an average age of 18.4 years
  - 1 student has an age of 14.7 years
-/
theorem age_of_35th_student 
  (total_students : Nat) 
  (avg_age_all : ℝ)
  (num_group1 : Nat) (avg_age_group1 : ℝ)
  (num_group2 : Nat) (avg_age_group2 : ℝ)
  (num_group3 : Nat) (avg_age_group3 : ℝ)
  (num_group4 : Nat) (age_group4 : ℝ)
  (h1 : total_students = 35)
  (h2 : avg_age_all = 16.5)
  (h3 : num_group1 = 10)
  (h4 : avg_age_group1 = 15.3)
  (h5 : num_group2 = 17)
  (h6 : avg_age_group2 = 16.7)
  (h7 : num_group3 = 6)
  (h8 : avg_age_group3 = 18.4)
  (h9 : num_group4 = 1)
  (h10 : age_group4 = 14.7)
  (h11 : num_group1 + num_group2 + num_group3 + num_group4 + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((num_group1 : ℝ) * avg_age_group1 + 
   (num_group2 : ℝ) * avg_age_group2 + 
   (num_group3 : ℝ) * avg_age_group3 + 
   (num_group4 : ℝ) * age_group4) = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_age_of_35th_student_l3712_371252


namespace NUMINAMATH_CALUDE_unique_solution_modular_system_l3712_371265

theorem unique_solution_modular_system :
  ∃! x : ℕ, x < 12 ∧ (5 * x + 3) % 15 = 7 % 15 ∧ x % 4 = 2 % 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_modular_system_l3712_371265


namespace NUMINAMATH_CALUDE_summer_determination_l3712_371217

def has_entered_summer (temperatures : List ℤ) : Prop :=
  temperatures.length = 5 ∧ ∀ t ∈ temperatures, t ≥ 22

def median (l : List ℤ) : ℤ := sorry
def mode (l : List ℤ) : ℤ := sorry
def mean (l : List ℤ) : ℚ := sorry
def variance (l : List ℤ) : ℚ := sorry

theorem summer_determination :
  ∀ (temps_A temps_B temps_C temps_D : List ℤ),
    (median temps_A = 24 ∧ mode temps_A = 22) →
    (median temps_B = 25 ∧ mean temps_B = 24) →
    (mean temps_C = 22 ∧ mode temps_C = 22) →
    (28 ∈ temps_D ∧ mean temps_D = 24 ∧ variance temps_D = 4.8) →
    (has_entered_summer temps_A ∧
     has_entered_summer temps_D ∧
     ¬(has_entered_summer temps_B ∧ has_entered_summer temps_C)) :=
by sorry

end NUMINAMATH_CALUDE_summer_determination_l3712_371217


namespace NUMINAMATH_CALUDE_i_to_2023_l3712_371242

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem: i^2023 = -i -/
theorem i_to_2023 : i^2023 = -i := by sorry

end NUMINAMATH_CALUDE_i_to_2023_l3712_371242


namespace NUMINAMATH_CALUDE_max_xyz_value_l3712_371267

theorem max_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + 2 * z = (x + z) * (y + z))
  (h2 : x + y + 2 * z = 2) :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
  a * b + 2 * c = (a + c) * (b + c) →
  a + b + 2 * c = 2 →
  x * y * z ≥ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_max_xyz_value_l3712_371267


namespace NUMINAMATH_CALUDE_product_mod_75_l3712_371287

theorem product_mod_75 : ∃ m : ℕ, 198 * 864 ≡ m [ZMOD 75] ∧ 0 ≤ m ∧ m < 75 :=
by
  use 72
  sorry

end NUMINAMATH_CALUDE_product_mod_75_l3712_371287


namespace NUMINAMATH_CALUDE_carnival_rides_l3712_371268

theorem carnival_rides (total_time hours roller_coaster_time tilt_a_whirl_time giant_slide_time : ℕ) 
  (roller_coaster_rides tilt_a_whirl_rides : ℕ) : 
  total_time = hours * 60 →
  roller_coaster_time = 30 →
  tilt_a_whirl_time = 60 →
  giant_slide_time = 15 →
  hours = 4 →
  roller_coaster_rides = 4 →
  tilt_a_whirl_rides = 1 →
  (total_time - (roller_coaster_rides * roller_coaster_time + tilt_a_whirl_rides * tilt_a_whirl_time)) / giant_slide_time = 4 :=
by sorry

end NUMINAMATH_CALUDE_carnival_rides_l3712_371268


namespace NUMINAMATH_CALUDE_range_of_a_l3712_371272

def has_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - a*x + 4 = 0

def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x ≥ 3, (4*x + a) ≥ 0

theorem range_of_a (a : ℝ) :
  has_real_roots a ∧ is_increasing_on_interval a →
  a ∈ Set.Icc (-12) (-4) ∪ Set.Ioi 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3712_371272


namespace NUMINAMATH_CALUDE_composite_division_l3712_371247

def first_four_composites : List Nat := [4, 6, 8, 9]
def next_four_composites : List Nat := [10, 12, 14, 15]

theorem composite_division :
  (first_four_composites.prod : ℚ) / (next_four_composites.prod : ℚ) = 12 / 175 := by
  sorry

end NUMINAMATH_CALUDE_composite_division_l3712_371247


namespace NUMINAMATH_CALUDE_water_depth_approx_0_6_l3712_371200

/-- Represents a horizontal cylindrical tank partially filled with water -/
structure WaterTank where
  length : ℝ
  diameter : ℝ
  exposedArea : ℝ

/-- Calculates the depth of water in the tank -/
def waterDepth (tank : WaterTank) : ℝ :=
  sorry

/-- Theorem stating that the water depth is approximately 0.6 feet for the given tank -/
theorem water_depth_approx_0_6 (tank : WaterTank) 
  (h1 : tank.length = 12)
  (h2 : tank.diameter = 8)
  (h3 : tank.exposedArea = 50) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |waterDepth tank - 0.6| < ε :=
  sorry

end NUMINAMATH_CALUDE_water_depth_approx_0_6_l3712_371200


namespace NUMINAMATH_CALUDE_hotel_revenue_calculation_l3712_371219

/-- The total revenue of a hotel for one night, given the number of single and double rooms booked and their respective prices. -/
def hotel_revenue (total_rooms single_price double_price double_rooms : ℕ) : ℕ :=
  let single_rooms := total_rooms - double_rooms
  single_rooms * single_price + double_rooms * double_price

/-- Theorem stating that under the given conditions, the hotel's revenue for one night is $14,000. -/
theorem hotel_revenue_calculation :
  hotel_revenue 260 35 60 196 = 14000 := by
  sorry

end NUMINAMATH_CALUDE_hotel_revenue_calculation_l3712_371219


namespace NUMINAMATH_CALUDE_grid_number_is_333_l3712_371291

/-- Represents a shape type -/
inductive Shape : Type
| A
| B
| C

/-- Represents a row in the grid -/
structure Row :=
  (shape : Shape)
  (count : Nat)

/-- The problem setup -/
def grid_setup : List Row :=
  [⟨Shape.A, 3⟩, ⟨Shape.B, 3⟩, ⟨Shape.C, 3⟩]

/-- Converts a list of rows to a natural number -/
def rows_to_number (rows : List Row) : Nat :=
  rows.foldl (fun acc row => acc * 10 + row.count) 0

/-- The main theorem -/
theorem grid_number_is_333 :
  rows_to_number grid_setup = 333 := by
  sorry

end NUMINAMATH_CALUDE_grid_number_is_333_l3712_371291


namespace NUMINAMATH_CALUDE_billy_homework_questions_l3712_371258

/-- Represents the number of questions solved in each hour -/
structure HourlyQuestions where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Theorem: Given the conditions, Billy solved 132 questions in the third hour -/
theorem billy_homework_questions (q : HourlyQuestions) : 
  q.third = 2 * q.second ∧ 
  q.third = 3 * q.first ∧ 
  q.first + q.second + q.third = 242 → 
  q.third = 132 := by
  sorry

end NUMINAMATH_CALUDE_billy_homework_questions_l3712_371258


namespace NUMINAMATH_CALUDE_probability_not_adjacent_l3712_371207

def total_chairs : ℕ := 10
def broken_chair : ℕ := 5
def available_chairs : ℕ := total_chairs - 1

theorem probability_not_adjacent : 
  let total_ways := available_chairs.choose 2
  let adjacent_pairs := 6
  (1 - (adjacent_pairs : ℚ) / total_ways) = 5/6 := by sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_l3712_371207


namespace NUMINAMATH_CALUDE_unique_m_solution_set_minimum_a_inequality_minimum_a_achieved_l3712_371243

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part 1
theorem unique_m_solution_set (m : ℝ) :
  m > 0 ∧
  (∀ x, |2 * (x + 1/2) - 1| ≤ 2 * m + 1 ↔ x ≤ -2 ∨ x ≥ 2) →
  m = 3/2 :=
sorry

-- Part 2
theorem minimum_a_inequality :
  ∀ a : ℝ,
  (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) →
  a ≥ 4 :=
sorry

theorem minimum_a_achieved :
  ∀ x y : ℝ, f x ≤ 2^y + 4/(2^y) + |2*x + 3| :=
sorry

end NUMINAMATH_CALUDE_unique_m_solution_set_minimum_a_inequality_minimum_a_achieved_l3712_371243


namespace NUMINAMATH_CALUDE_largest_reciprocal_l3712_371241

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 3/4 → b = 5/3 → c = -1/6 → d = 7 → e = 3 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l3712_371241


namespace NUMINAMATH_CALUDE_coke_to_sprite_ratio_l3712_371283

/-- Represents the ratio of ingredients in a drink -/
structure DrinkRatio where
  coke : ℚ
  sprite : ℚ
  mountainDew : ℚ

/-- Represents the composition of a drink -/
structure Drink where
  ratio : DrinkRatio
  cokeAmount : ℚ
  totalAmount : ℚ

/-- Theorem: Given a drink with the specified ratio and amounts, prove that the ratio of Coke to Sprite is 2:1 -/
theorem coke_to_sprite_ratio 
  (drink : Drink) 
  (h1 : drink.ratio.sprite = 1)
  (h2 : drink.ratio.mountainDew = 3)
  (h3 : drink.cokeAmount = 6)
  (h4 : drink.totalAmount = 18) :
  drink.ratio.coke / drink.ratio.sprite = 2 := by
  sorry


end NUMINAMATH_CALUDE_coke_to_sprite_ratio_l3712_371283


namespace NUMINAMATH_CALUDE_pencil_carton_cost_pencil_carton_cost_proof_l3712_371221

/-- The cost of a carton of pencils given the following conditions:
  1. Erasers cost 3 dollars per carton
  2. Total order is 100 cartons
  3. Total order cost is 360 dollars
  4. The order includes 20 cartons of pencils -/
theorem pencil_carton_cost : ℝ :=
  let eraser_cost : ℝ := 3
  let total_cartons : ℕ := 100
  let total_cost : ℝ := 360
  let pencil_cartons : ℕ := 20
  6

/-- Proof that the cost of a carton of pencils is 6 dollars -/
theorem pencil_carton_cost_proof :
  pencil_carton_cost = 6 := by sorry

end NUMINAMATH_CALUDE_pencil_carton_cost_pencil_carton_cost_proof_l3712_371221


namespace NUMINAMATH_CALUDE_ferry_problem_l3712_371279

/-- The ferry problem -/
theorem ferry_problem (speed_p speed_q : ℝ) (time_p distance_q : ℝ) :
  speed_p = 8 →
  time_p = 2 →
  distance_q = 3 * speed_p * time_p →
  speed_q = speed_p + 4 →
  distance_q / speed_q - time_p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ferry_problem_l3712_371279


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3712_371299

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (m : ℕ) 
  (h1 : m > 0)
  (h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q))
  (h3 : ∀ n, a (n + 1) = q * a n)
  (h4 : S (2 * m) / S m = 9)
  (h5 : a (2 * m) / a m = (5 * m + 1) / (m - 1)) :
  q = 2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3712_371299


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l3712_371204

theorem average_speed_round_trip (speed_xy speed_yx : ℝ) (h1 : speed_xy = 43) (h2 : speed_yx = 34) :
  (2 * speed_xy * speed_yx) / (speed_xy + speed_yx) = 38 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_l3712_371204


namespace NUMINAMATH_CALUDE_work_left_fraction_l3712_371269

theorem work_left_fraction (a_days b_days work_days : ℚ) 
  (ha : a_days = 20)
  (hb : b_days = 30)
  (hw : work_days = 4) : 
  1 - work_days * (1 / a_days + 1 / b_days) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_work_left_fraction_l3712_371269


namespace NUMINAMATH_CALUDE_unfilled_holes_l3712_371256

theorem unfilled_holes (total : ℕ) (filled_percentage : ℚ) : 
  total = 8 → filled_percentage = 75 / 100 → total - (filled_percentage * total).floor = 2 := by
sorry

end NUMINAMATH_CALUDE_unfilled_holes_l3712_371256


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l3712_371201

-- Define the function
def f (x : ℝ) : ℝ := -x^3 - x^2 + 2

-- Theorem statement
theorem f_has_max_and_min :
  (∃ x_max : ℝ, ∀ x : ℝ, f x ≤ f x_max) ∧
  (∃ x_min : ℝ, ∀ x : ℝ, f x_min ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l3712_371201


namespace NUMINAMATH_CALUDE_adam_tattoo_count_l3712_371212

/-- The number of tattoos Jason has on each arm -/
def jason_arm_tattoos : ℕ := 2

/-- The number of tattoos Jason has on each leg -/
def jason_leg_tattoos : ℕ := 3

/-- The number of arms Jason has -/
def jason_arms : ℕ := 2

/-- The number of legs Jason has -/
def jason_legs : ℕ := 2

/-- The total number of tattoos Jason has -/
def jason_total_tattoos : ℕ := jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs

/-- The number of tattoos Adam has -/
def adam_tattoos : ℕ := 2 * jason_total_tattoos + 3

theorem adam_tattoo_count : adam_tattoos = 23 := by
  sorry

end NUMINAMATH_CALUDE_adam_tattoo_count_l3712_371212


namespace NUMINAMATH_CALUDE_uncovered_volume_is_229_l3712_371251

def shoebox_volume : ℝ := 4 * 6 * 12

def object1_volume : ℝ := 5 * 3 * 1
def object2_volume : ℝ := 2 * 2 * 3
def object3_volume : ℝ := 4 * 2 * 4

def total_object_volume : ℝ := object1_volume + object2_volume + object3_volume

theorem uncovered_volume_is_229 : 
  shoebox_volume - total_object_volume = 229 := by sorry

end NUMINAMATH_CALUDE_uncovered_volume_is_229_l3712_371251


namespace NUMINAMATH_CALUDE_wrong_to_right_ratio_l3712_371275

theorem wrong_to_right_ratio (total : ℕ) (correct : ℕ) 
  (h1 : total = 54) (h2 : correct = 18) :
  (total - correct) / correct = 2 := by
  sorry

end NUMINAMATH_CALUDE_wrong_to_right_ratio_l3712_371275


namespace NUMINAMATH_CALUDE_sqrt_13_parts_sum_l3712_371244

theorem sqrt_13_parts_sum (x y : ℝ) : 
  (x = ⌊Real.sqrt 13⌋) → 
  (y = Real.sqrt 13 - ⌊Real.sqrt 13⌋) → 
  (2 * x - y + Real.sqrt 13 = 9) := by
sorry

end NUMINAMATH_CALUDE_sqrt_13_parts_sum_l3712_371244


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_is_five_l3712_371205

/-- Represents the rental information for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_kayak_ratio : Rat
  total_revenue : ℕ

/-- Calculates the difference between canoes and kayaks rented --/
def canoe_kayak_difference (info : RentalInfo) : ℕ :=
  let kayaks := (info.total_revenue : ℚ) * 3 / (11 * 4 + 16 * 3)
  let canoes := kayaks * info.canoe_kayak_ratio
  (canoes - kayaks).ceil.toNat

/-- Theorem stating the difference between canoes and kayaks rented --/
theorem canoe_kayak_difference_is_five (info : RentalInfo) 
  (h1 : info.canoe_cost = 11)
  (h2 : info.kayak_cost = 16)
  (h3 : info.canoe_kayak_ratio = 4 / 3)
  (h4 : info.total_revenue = 460) :
  canoe_kayak_difference info = 5 := by
  sorry

#eval canoe_kayak_difference { 
  canoe_cost := 11, 
  kayak_cost := 16, 
  canoe_kayak_ratio := 4 / 3, 
  total_revenue := 460 
}

end NUMINAMATH_CALUDE_canoe_kayak_difference_is_five_l3712_371205


namespace NUMINAMATH_CALUDE_james_pizza_toppings_cost_l3712_371294

/-- Calculates the cost of pizza toppings eaten by James -/
theorem james_pizza_toppings_cost :
  let num_pizzas : ℕ := 2
  let slices_per_pizza : ℕ := 6
  let topping_costs : List ℚ := [3/2, 2, 5/4]
  let james_portion : ℚ := 2/3

  let total_slices : ℕ := num_pizzas * slices_per_pizza
  let total_topping_cost : ℚ := (num_pizzas : ℚ) * (topping_costs.sum)
  let james_topping_cost : ℚ := james_portion * total_topping_cost

  james_topping_cost = 633/100 :=
by
  sorry

end NUMINAMATH_CALUDE_james_pizza_toppings_cost_l3712_371294


namespace NUMINAMATH_CALUDE_fruit_rate_proof_l3712_371278

/-- The rate per kg for both apples and mangoes -/
def R : ℝ := 70

/-- The weight of apples purchased in kg -/
def apple_weight : ℝ := 8

/-- The weight of mangoes purchased in kg -/
def mango_weight : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 1190

theorem fruit_rate_proof :
  apple_weight * R + mango_weight * R = total_paid :=
by sorry

end NUMINAMATH_CALUDE_fruit_rate_proof_l3712_371278


namespace NUMINAMATH_CALUDE_sum_natural_numbers_not_end_72_73_74_l3712_371209

theorem sum_natural_numbers_not_end_72_73_74 (N : ℕ) : 
  ¬ (∃ k : ℕ, (N * (N + 1)) / 2 = 100 * k + 72 ∨ 
               (N * (N + 1)) / 2 = 100 * k + 73 ∨ 
               (N * (N + 1)) / 2 = 100 * k + 74) := by
  sorry


end NUMINAMATH_CALUDE_sum_natural_numbers_not_end_72_73_74_l3712_371209


namespace NUMINAMATH_CALUDE_num_outfits_is_480_l3712_371228

/-- Number of shirts available --/
def num_shirts : ℕ := 8

/-- Number of ties available --/
def num_ties : ℕ := 5

/-- Number of pants available --/
def num_pants : ℕ := 3

/-- Number of belts available --/
def num_belts : ℕ := 4

/-- Number of belts that can be worn with a tie --/
def num_belts_with_tie : ℕ := 2

/-- Calculate the number of different outfits --/
def num_outfits : ℕ :=
  let outfits_without_tie := num_shirts * num_pants * (num_belts + 1)
  let outfits_with_tie := num_shirts * num_pants * num_ties * (num_belts_with_tie + 1)
  outfits_without_tie + outfits_with_tie

/-- Theorem stating that the number of different outfits is 480 --/
theorem num_outfits_is_480 : num_outfits = 480 := by
  sorry

end NUMINAMATH_CALUDE_num_outfits_is_480_l3712_371228


namespace NUMINAMATH_CALUDE_range_of_4a_minus_2b_l3712_371263

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : 0 ≤ a - b) (h2 : a - b ≤ 1) 
  (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) : 
  2 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_4a_minus_2b_l3712_371263


namespace NUMINAMATH_CALUDE_complex_number_problem_l3712_371282

theorem complex_number_problem (a : ℝ) (z : ℂ) (h1 : z = a + I) 
  (h2 : (Complex.I * 2 + 1) * z ∈ {w : ℂ | w.re = 0 ∧ w.im ≠ 0}) :
  z = 2 + I ∧ Complex.abs (z / (2 - I)) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3712_371282


namespace NUMINAMATH_CALUDE_point_not_in_region_l3712_371284

def planar_region (x y : ℝ) : Prop := 2 * x + 3 * y < 6

theorem point_not_in_region :
  ¬ (planar_region 0 2) ∧
  (planar_region 0 0) ∧
  (planar_region 1 1) ∧
  (planar_region 2 0) :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_region_l3712_371284


namespace NUMINAMATH_CALUDE_max_slope_no_lattice_points_l3712_371289

theorem max_slope_no_lattice_points :
  let max_a : ℚ := 25 / 49
  ∀ a : ℚ, (∀ m x y : ℚ,
    (1 / 2 < m) → (m < a) →
    (0 < x) → (x ≤ 50) →
    (y = m * x + 3) →
    (∃ n : ℤ, x = ↑n) →
    (∃ n : ℤ, y = ↑n) →
    False) →
  a ≤ max_a :=
by sorry

end NUMINAMATH_CALUDE_max_slope_no_lattice_points_l3712_371289


namespace NUMINAMATH_CALUDE_min_cuts_for_eleven_days_max_rings_for_n_cuts_l3712_371271

/-- Represents a chain of rings -/
structure Chain where
  rings : ℕ

/-- Represents a stay at the inn -/
structure Stay where
  days : ℕ

/-- Calculates the minimum number of cuts required for a given chain and stay -/
def minCuts (chain : Chain) (stay : Stay) : ℕ :=
  sorry

/-- Calculates the maximum number of rings in a chain for a given number of cuts -/
def maxRings (cuts : ℕ) : ℕ :=
  sorry

theorem min_cuts_for_eleven_days (chain : Chain) (stay : Stay) :
  chain.rings = 11 → stay.days = 11 → minCuts chain stay = 2 :=
  sorry

theorem max_rings_for_n_cuts (n : ℕ) :
  maxRings n = (n + 1) * 2^n - 1 :=
  sorry

end NUMINAMATH_CALUDE_min_cuts_for_eleven_days_max_rings_for_n_cuts_l3712_371271


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_product_12_l3712_371288

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_product_12_l3712_371288


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3712_371239

theorem quadratic_equation_roots (θ : Real) (m : Real) :
  θ ∈ Set.Ioo 0 (2 * Real.pi) →
  (∃ x, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0) →
  (∃ x, x = Real.sin θ ∨ x = Real.cos θ) →
  (Real.sin θ / (1 - Real.cos θ) + Real.cos θ / (1 - Real.tan θ) = (3 + 5 * Real.sqrt 3) / 4) ∧
  (m = Real.sqrt 3 / 4) ∧
  ((Real.sin θ = Real.sqrt 3 / 2 ∧ Real.cos θ = 1 / 2 ∧ θ = Real.pi / 3) ∨
   (Real.sin θ = 1 / 2 ∧ Real.cos θ = Real.sqrt 3 / 2 ∧ θ = Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3712_371239


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3712_371231

theorem inequality_equivalence (y : ℝ) : 
  3/20 + |y - 1/5| < 1/4 ↔ y ∈ Set.Ioo (1/10 : ℝ) (3/10 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3712_371231


namespace NUMINAMATH_CALUDE_sin_squared_minus_three_sin_plus_two_range_l3712_371227

theorem sin_squared_minus_three_sin_plus_two_range :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi →
  ∃ y : ℝ, y = Real.sin x ^ 2 - 3 * Real.sin x + 2 ∧ 0 ≤ y ∧ y ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_sin_squared_minus_three_sin_plus_two_range_l3712_371227


namespace NUMINAMATH_CALUDE_expression_value_l3712_371230

theorem expression_value (x y : ℝ) (h : x - 2*y = 3) : 1 - 2*x + 4*y = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3712_371230


namespace NUMINAMATH_CALUDE_root_implies_a_in_interval_l3712_371266

/-- Given that for all real m, the function f(x) = m(x^2 - 1) + x - a always has a root,
    prove that a is in the interval [-1, 1] -/
theorem root_implies_a_in_interval :
  (∀ m : ℝ, ∃ x : ℝ, m * (x^2 - 1) + x - a = 0) →
  a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_in_interval_l3712_371266


namespace NUMINAMATH_CALUDE_z_squared_and_modulus_l3712_371237

-- Define the complex number z
def z : ℂ := 5 + 3 * Complex.I

-- Theorem statement
theorem z_squared_and_modulus :
  z ^ 2 = 16 + 30 * Complex.I ∧ Complex.abs (z ^ 2) = 34 := by
  sorry

end NUMINAMATH_CALUDE_z_squared_and_modulus_l3712_371237


namespace NUMINAMATH_CALUDE_point_difference_l3712_371295

def zachScore : ℕ := 42
def benScore : ℕ := 21

theorem point_difference : zachScore - benScore = 21 := by
  sorry

end NUMINAMATH_CALUDE_point_difference_l3712_371295


namespace NUMINAMATH_CALUDE_max_value_expression_l3712_371222

theorem max_value_expression :
  (∃ x : ℝ, |x - 1| - |x + 4| - 5 = 0) ∧
  (∀ x : ℝ, |x - 1| - |x + 4| - 5 ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3712_371222


namespace NUMINAMATH_CALUDE_rod_cutting_l3712_371245

/-- Given a rod of length 34 meters that can be cut into 40 equal pieces,
    prove that each piece is 0.85 meters long. -/
theorem rod_cutting (rod_length : ℝ) (num_pieces : ℕ) (piece_length : ℝ) 
  (h1 : rod_length = 34)
  (h2 : num_pieces = 40)
  (h3 : piece_length * num_pieces = rod_length) :
  piece_length = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l3712_371245


namespace NUMINAMATH_CALUDE_prob_two_slate_is_11_105_l3712_371226

-- Define the number of rocks for each type
def slate_rocks : ℕ := 12
def pumice_rocks : ℕ := 16
def granite_rocks : ℕ := 8

-- Define the total number of rocks
def total_rocks : ℕ := slate_rocks + pumice_rocks + granite_rocks

-- Define the probability of selecting two slate rocks
def prob_two_slate : ℚ := (slate_rocks : ℚ) / total_rocks * (slate_rocks - 1) / (total_rocks - 1)

theorem prob_two_slate_is_11_105 : prob_two_slate = 11 / 105 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_slate_is_11_105_l3712_371226


namespace NUMINAMATH_CALUDE_contemporary_probability_correct_l3712_371235

/-- The duration in years of the period considered -/
def period : ℕ := 800

/-- The lifespan of each mathematician in years -/
def lifespan : ℕ := 150

/-- The probability that two mathematicians born within a given period
    are contemporaries, given their lifespans and assuming uniform distribution
    of birth years -/
def contemporaryProbability (p : ℕ) (l : ℕ) : ℚ :=
  let totalArea := p * p
  let nonOverlapArea := 2 * (p - l) * l / 2
  let overlapArea := totalArea - nonOverlapArea
  overlapArea / totalArea

theorem contemporary_probability_correct :
  contemporaryProbability period lifespan = 27125 / 32000 := by
  sorry

end NUMINAMATH_CALUDE_contemporary_probability_correct_l3712_371235


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l3712_371218

theorem thirty_percent_less_than_eighty (x : ℝ) : x + x/2 = 80 * (1 - 0.3) → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l3712_371218


namespace NUMINAMATH_CALUDE_candy_solution_l3712_371223

/-- Represents the candy distribution problem -/
def candy_problem (billy_initial caleb_initial andy_initial father_bought billy_received caleb_received : ℕ) : Prop :=
  let andy_received := father_bought - billy_received - caleb_received
  let billy_final := billy_initial + billy_received
  let caleb_final := caleb_initial + caleb_received
  let andy_final := andy_initial + andy_received
  andy_final - caleb_final = 4

/-- Theorem stating the solution to the candy distribution problem -/
theorem candy_solution :
  candy_problem 6 11 9 36 8 11 := by
  sorry

#check candy_solution

end NUMINAMATH_CALUDE_candy_solution_l3712_371223


namespace NUMINAMATH_CALUDE_average_string_length_l3712_371233

theorem average_string_length : 
  let string_lengths : List ℝ := [1.5, 4.5, 6, 3]
  let n : ℕ := string_lengths.length
  let sum : ℝ := string_lengths.sum
  sum / n = 3.75 := by
sorry

end NUMINAMATH_CALUDE_average_string_length_l3712_371233


namespace NUMINAMATH_CALUDE_ticket_problem_l3712_371225

/-- Represents the ticket distribution and pricing for a football match --/
structure TicketInfo where
  total : ℕ  -- Total number of tickets
  typeA : ℕ  -- Number of Type A tickets
  m : ℕ      -- Price parameter

/-- Conditions for the ticket distribution and pricing --/
def validTicketInfo (info : TicketInfo) : Prop :=
  info.total = 500 ∧
  info.typeA ≥ 3 * (info.total - info.typeA) ∧
  500 * (1 + (info.m + 10) / 100) * (info.m + 20) = 56000 ∧
  info.m > 0

theorem ticket_problem (info : TicketInfo) (h : validTicketInfo info) :
  info.typeA ≥ 375 ∧ info.m = 50 := by
  sorry


end NUMINAMATH_CALUDE_ticket_problem_l3712_371225


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3712_371259

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l3712_371259


namespace NUMINAMATH_CALUDE_meeting_distance_calculation_l3712_371274

/-- Represents the problem of calculating the distance to a meeting location --/
theorem meeting_distance_calculation (initial_speed : ℝ) (speed_increase : ℝ) 
  (late_time : ℝ) (early_time : ℝ) :
  initial_speed = 40 →
  speed_increase = 20 →
  late_time = 1.5 →
  early_time = 1 →
  ∃ (distance : ℝ) (total_time : ℝ),
    distance = initial_speed * (total_time + late_time) ∧
    distance = initial_speed + (initial_speed + speed_increase) * (total_time - early_time - 1) ∧
    distance = 420 := by
  sorry


end NUMINAMATH_CALUDE_meeting_distance_calculation_l3712_371274


namespace NUMINAMATH_CALUDE_manager_count_is_two_l3712_371254

/-- Represents the daily salary structure and employee count in a grocery store -/
structure GroceryStore where
  managerSalary : ℕ
  clerkSalary : ℕ
  clerkCount : ℕ
  totalSalary : ℕ

/-- Calculates the number of managers in the grocery store -/
def managerCount (store : GroceryStore) : ℕ :=
  (store.totalSalary - store.clerkSalary * store.clerkCount) / store.managerSalary

/-- Theorem stating that the number of managers in the given scenario is 2 -/
theorem manager_count_is_two :
  let store : GroceryStore := {
    managerSalary := 5,
    clerkSalary := 2,
    clerkCount := 3,
    totalSalary := 16
  }
  managerCount store = 2 := by sorry

end NUMINAMATH_CALUDE_manager_count_is_two_l3712_371254


namespace NUMINAMATH_CALUDE_quarter_circle_area_l3712_371214

/-- The area of a quarter circle with radius 2 is equal to π -/
theorem quarter_circle_area : 
  let r : Real := 2
  let circle_area : Real := π * r^2
  let quarter_circle_area : Real := circle_area / 4
  quarter_circle_area = π := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_area_l3712_371214


namespace NUMINAMATH_CALUDE_sphere_expansion_l3712_371273

/-- Given a sphere with initial radius 1 and final radius m, 
    if the volume expansion rate is 28π/3, then m = 2 -/
theorem sphere_expansion (m : ℝ) : 
  m > 0 →  -- Ensure m is positive (as it's a radius)
  (4 * π / 3 * (m^3 - 1)) / (m - 1) = 28 * π / 3 →
  m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_sphere_expansion_l3712_371273


namespace NUMINAMATH_CALUDE_exists_n_pow_half_n_eq_twelve_l3712_371216

theorem exists_n_pow_half_n_eq_twelve :
  ∃ n : ℝ, n > 0 ∧ n^(n/2) = 12 :=
by sorry

end NUMINAMATH_CALUDE_exists_n_pow_half_n_eq_twelve_l3712_371216


namespace NUMINAMATH_CALUDE_simple_interest_years_l3712_371281

theorem simple_interest_years (principal interest_amount rate : ℝ) (h1 : principal = 1600)
  (h2 : interest_amount = 200) (h3 : rate = 3.125) :
  (interest_amount * 100) / (principal * rate) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_years_l3712_371281


namespace NUMINAMATH_CALUDE_multiples_of_seven_l3712_371211

theorem multiples_of_seven (x : ℕ) : 
  (∃ n : ℕ, n = 47 ∧ 
   (∀ k : ℕ, x ≤ 7 * k ∧ 7 * k ≤ 343 → k ≤ n) ∧
   (∀ k : ℕ, k ≤ n → x ≤ 7 * k ∧ 7 * k ≤ 343)) →
  x = 14 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_seven_l3712_371211


namespace NUMINAMATH_CALUDE_pt_length_in_special_quadrilateral_l3712_371292

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (quad : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def intersection_point (p1 p2 p3 p4 : Point) : Point := sorry

theorem pt_length_in_special_quadrilateral 
  (PQRS : Quadrilateral) 
  (T : Point)
  (h_convex : is_convex PQRS)
  (h_PQ : distance PQRS.P PQRS.Q = 10)
  (h_RS : distance PQRS.R PQRS.S = 15)
  (h_PR : distance PQRS.P PQRS.R = 18)
  (h_T : T = intersection_point PQRS.P PQRS.R PQRS.Q PQRS.S)
  (h_equal_areas : triangle_area PQRS.P T PQRS.S = triangle_area PQRS.Q T PQRS.R) :
  distance PQRS.P T = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_pt_length_in_special_quadrilateral_l3712_371292


namespace NUMINAMATH_CALUDE_model_comparison_theorem_l3712_371255

/-- A model for fitting data -/
structure Model where
  /-- The sum of squared residuals for this model -/
  sumSquaredResiduals : ℝ
  /-- Whether the residual points are uniformly distributed in a horizontal band -/
  uniformResiduals : Prop

/-- Compares the fitting effects of two models -/
def betterFit (m1 m2 : Model) : Prop :=
  m1.sumSquaredResiduals < m2.sumSquaredResiduals

/-- Indicates whether a model is appropriate based on its residual plot -/
def appropriateModel (m : Model) : Prop :=
  m.uniformResiduals

theorem model_comparison_theorem :
  ∀ (m1 m2 : Model),
    (betterFit m1 m2 → m1.sumSquaredResiduals < m2.sumSquaredResiduals) ∧
    (appropriateModel m1 ↔ m1.uniformResiduals) :=
by sorry

end NUMINAMATH_CALUDE_model_comparison_theorem_l3712_371255


namespace NUMINAMATH_CALUDE_percentage_B_of_D_l3712_371285

theorem percentage_B_of_D (A B C D : ℝ) 
  (h1 : B = 1.71 * A)
  (h2 : C = 1.80 * A)
  (h3 : D = 1.90 * B)
  (h4 : B = 1.62 * C)
  (h5 : A = 0.65 * D)
  (h6 : C = 0.55 * D) :
  B = 1.1115 * D := by
sorry

end NUMINAMATH_CALUDE_percentage_B_of_D_l3712_371285


namespace NUMINAMATH_CALUDE_white_area_is_69_l3712_371206

/-- Represents the dimensions of a rectangular sign -/
structure SignDimensions where
  width : ℕ
  height : ℕ

/-- Represents the area covered by a letter -/
structure LetterArea where
  area : ℕ

/-- Calculates the total area of the sign -/
def totalSignArea (dim : SignDimensions) : ℕ :=
  dim.width * dim.height

/-- Calculates the area covered by the letter M -/
def mArea : LetterArea :=
  { area := 2 * (6 * 1) + 2 * 2 }

/-- Calculates the area covered by the letter A -/
def aArea : LetterArea :=
  { area := 2 * 4 + 1 * 2 }

/-- Calculates the area covered by the letter T -/
def tArea : LetterArea :=
  { area := 1 * 4 + 6 * 1 }

/-- Calculates the area covered by the letter H -/
def hArea : LetterArea :=
  { area := 2 * (6 * 1) + 1 * 3 }

/-- Calculates the total area covered by all letters -/
def totalLettersArea : ℕ :=
  mArea.area + aArea.area + tArea.area + hArea.area

/-- The main theorem: proves that the white area of the sign is 69 square units -/
theorem white_area_is_69 (sign : SignDimensions) 
    (h1 : sign.width = 20) 
    (h2 : sign.height = 6) : 
    totalSignArea sign - totalLettersArea = 69 := by
  sorry


end NUMINAMATH_CALUDE_white_area_is_69_l3712_371206


namespace NUMINAMATH_CALUDE_problem_sculpture_area_l3712_371298

/-- Represents a pyramid-like sculpture made of unit cubes -/
structure PyramidSculpture where
  total_cubes : ℕ
  num_layers : ℕ
  layer_sizes : List ℕ
  (total_cubes_sum : total_cubes = layer_sizes.sum)
  (layer_count : num_layers = layer_sizes.length)

/-- Calculates the exposed surface area of a pyramid sculpture -/
def exposed_surface_area (p : PyramidSculpture) : ℕ :=
  sorry

/-- The specific pyramid sculpture described in the problem -/
def problem_sculpture : PyramidSculpture :=
  { total_cubes := 19
  , num_layers := 4
  , layer_sizes := [1, 3, 5, 10]
  , total_cubes_sum := by sorry
  , layer_count := by sorry
  }

/-- Theorem stating that the exposed surface area of the problem sculpture is 43 square meters -/
theorem problem_sculpture_area : exposed_surface_area problem_sculpture = 43 := by
  sorry

end NUMINAMATH_CALUDE_problem_sculpture_area_l3712_371298


namespace NUMINAMATH_CALUDE_blocks_and_colors_l3712_371253

theorem blocks_and_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (colors_used : ℕ) : 
  total_blocks = 49 → blocks_per_color = 7 → colors_used = total_blocks / blocks_per_color →
  colors_used = 7 := by
  sorry

end NUMINAMATH_CALUDE_blocks_and_colors_l3712_371253


namespace NUMINAMATH_CALUDE_books_loaned_out_l3712_371208

theorem books_loaned_out (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) :
  initial_books = 75 →
  return_rate = 65 / 100 →
  final_books = 61 →
  (initial_books - final_books : ℚ) / (1 - return_rate) = 40 := by
sorry

end NUMINAMATH_CALUDE_books_loaned_out_l3712_371208


namespace NUMINAMATH_CALUDE_complex_number_existence_l3712_371297

theorem complex_number_existence : ∃! (z₁ z₂ : ℂ),
  (z₁ + 10 / z₁).im = 0 ∧
  (z₂ + 10 / z₂).im = 0 ∧
  (z₁ + 4).re = -(z₁ + 4).im ∧
  (z₂ + 4).re = -(z₂ + 4).im ∧
  z₁ = -1 - 3*I ∧
  z₂ = -3 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_existence_l3712_371297


namespace NUMINAMATH_CALUDE_probability_of_selecting_specific_car_type_l3712_371293

theorem probability_of_selecting_specific_car_type 
  (total_car_types : ℕ) 
  (cars_selected : ℕ) 
  (h1 : total_car_types = 5) 
  (h2 : cars_selected = 2) :
  (cars_selected : ℚ) / (total_car_types.choose cars_selected) = 2/5 := by
sorry

end NUMINAMATH_CALUDE_probability_of_selecting_specific_car_type_l3712_371293


namespace NUMINAMATH_CALUDE_complexity_not_greater_for_power_of_two_exists_number_with_greater_or_equal_complexity_l3712_371238

/-- The complexity of an integer is the number of factors in its prime factorization -/
def complexity (n : ℕ) : ℕ := sorry

/-- For n = 2^k, all numbers between n and 2n have complexity not greater than that of n -/
theorem complexity_not_greater_for_power_of_two (k : ℕ) :
  ∀ m : ℕ, 2^k ≤ m → m ≤ 2^(k+1) → complexity m ≤ complexity (2^k) := by sorry

/-- For any n > 1, there exists at least one number between n and 2n with complexity greater than or equal to that of n -/
theorem exists_number_with_greater_or_equal_complexity (n : ℕ) (h : n > 1) :
  ∃ m : ℕ, n < m ∧ m < 2*n ∧ complexity m ≥ complexity n := by sorry

end NUMINAMATH_CALUDE_complexity_not_greater_for_power_of_two_exists_number_with_greater_or_equal_complexity_l3712_371238


namespace NUMINAMATH_CALUDE_solution_x_equals_three_l3712_371240

theorem solution_x_equals_three : ∃ (f : ℝ → ℝ), f 3 = 0 ∧ (∀ x, f x = 0 → x = 3) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_solution_x_equals_three_l3712_371240


namespace NUMINAMATH_CALUDE_harry_book_count_l3712_371264

/-- The number of books Harry has -/
def harry_books : ℕ := 50

/-- The number of books Flora has -/
def flora_books : ℕ := 2 * harry_books

/-- The number of books Gary has -/
def gary_books : ℕ := harry_books / 2

/-- The total number of books -/
def total_books : ℕ := 175

theorem harry_book_count : 
  harry_books + flora_books + gary_books = total_books ∧ harry_books = 50 := by
  sorry

end NUMINAMATH_CALUDE_harry_book_count_l3712_371264


namespace NUMINAMATH_CALUDE_only_one_satisfies_property_l3712_371220

theorem only_one_satisfies_property : ∃! (n : ℕ), 
  n > 0 ∧ 
  (∀ (a : ℤ), Odd a → (a^2 : ℤ) ≤ n → a ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_only_one_satisfies_property_l3712_371220


namespace NUMINAMATH_CALUDE_cars_left_parking_lot_l3712_371250

/-- Calculates the number of cars that have left a parking lot -/
def cars_left (initial : ℕ) (current : ℕ) : ℕ :=
  initial - current

theorem cars_left_parking_lot (initial : ℕ) (current : ℕ) 
  (h1 : initial = 12) 
  (h2 : current = 9) 
  (h3 : current ≤ initial) : 
  cars_left initial current = 3 := by
  sorry

end NUMINAMATH_CALUDE_cars_left_parking_lot_l3712_371250


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_odds_divisible_by_five_l3712_371276

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def consecutive_odds (a b c d : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ is_odd c ∧ is_odd d ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2

def not_divisible_by_three (n : ℕ) : Prop := n % 3 ≠ 0

theorem smallest_sum_of_four_odds_divisible_by_five :
  ∃ a b c d : ℕ,
    consecutive_odds a b c d ∧
    not_divisible_by_three a ∧
    not_divisible_by_three b ∧
    not_divisible_by_three c ∧
    not_divisible_by_three d ∧
    (a + b + c + d) % 5 = 0 ∧
    a + b + c + d = 40 ∧
    (∀ w x y z : ℕ,
      consecutive_odds w x y z →
      not_divisible_by_three w →
      not_divisible_by_three x →
      not_divisible_by_three y →
      not_divisible_by_three z →
      (w + x + y + z) % 5 = 0 →
      w + x + y + z ≥ 40) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_odds_divisible_by_five_l3712_371276


namespace NUMINAMATH_CALUDE_eight_digit_permutations_eq_1680_l3712_371280

/-- The number of different positive, eight-digit integers that can be formed
    using the digits 2, 2, 2, 5, 5, 7, 9, and 9 -/
def eight_digit_permutations : ℕ :=
  Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive, eight-digit integers
    that can be formed using the digits 2, 2, 2, 5, 5, 7, 9, and 9 is 1680 -/
theorem eight_digit_permutations_eq_1680 :
  eight_digit_permutations = 1680 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_permutations_eq_1680_l3712_371280


namespace NUMINAMATH_CALUDE_yuna_weekly_problems_l3712_371236

/-- The number of English problems Yuna solves in one week -/
def problems_in_week (problems_per_day : ℕ) : ℕ :=
  problems_per_day * 7

/-- Theorem: Yuna solves 56 English problems in one week -/
theorem yuna_weekly_problems : problems_in_week 8 = 56 := by
  sorry

end NUMINAMATH_CALUDE_yuna_weekly_problems_l3712_371236


namespace NUMINAMATH_CALUDE_forces_form_hyperboloid_rulings_l3712_371229

-- Define a 3D vector
def Vector3D := ℝ × ℝ × ℝ

-- Define a line in 3D space
structure Line3D where
  point : Vector3D
  direction : Vector3D

-- Define a force as a line with a magnitude
structure Force where
  line : Line3D
  magnitude : ℝ

-- Define the concept of equilibrium
def is_equilibrium (forces : List Force) : Prop := sorry

-- Define the concept of non-coplanarity
def are_non_coplanar (lines : List Line3D) : Prop := sorry

-- Define the concept of a hyperboloid
def is_hyperboloid_ruling (lines : List Line3D) : Prop := sorry

-- The main theorem
theorem forces_form_hyperboloid_rulings 
  (forces : List Force) 
  (h_count : forces.length = 4)
  (h_equilibrium : is_equilibrium forces)
  (h_non_coplanar : are_non_coplanar (forces.map Force.line)) :
  is_hyperboloid_ruling (forces.map Force.line) := by sorry

end NUMINAMATH_CALUDE_forces_form_hyperboloid_rulings_l3712_371229


namespace NUMINAMATH_CALUDE_unique_determination_of_polynomial_minimality_of_points_l3712_371246

/-- A polynomial of degree 2017 with integer coefficients and leading coefficient 1 -/
def IntPolynomial2017 : Type := 
  {p : Polynomial ℤ // p.degree = 2017 ∧ p.leadingCoeff = 1}

/-- The minimum number of points needed to uniquely determine the polynomial -/
def minPointsForUniqueness : ℕ := 2017

theorem unique_determination_of_polynomial (p q : IntPolynomial2017) 
  (points : Fin minPointsForUniqueness → ℤ) :
  (∀ i : Fin minPointsForUniqueness, p.val.eval (points i) = q.val.eval (points i)) →
  p = q :=
sorry

theorem minimality_of_points :
  ∀ k : ℕ, k < minPointsForUniqueness →
  ∃ (p q : IntPolynomial2017) (points : Fin k → ℤ),
    (∀ i : Fin k, p.val.eval (points i) = q.val.eval (points i)) ∧
    p ≠ q :=
sorry

end NUMINAMATH_CALUDE_unique_determination_of_polynomial_minimality_of_points_l3712_371246


namespace NUMINAMATH_CALUDE_cost_price_is_4_l3712_371224

/-- The cost price of a pen in yuan. -/
def cost_price : ℝ := 4

/-- The retail price of a pen in the first scenario. -/
def retail_price1 : ℝ := 7

/-- The retail price of a pen in the second scenario. -/
def retail_price2 : ℝ := 8

/-- The number of pens sold in the first scenario. -/
def num_pens1 : ℕ := 20

/-- The number of pens sold in the second scenario. -/
def num_pens2 : ℕ := 15

theorem cost_price_is_4 : 
  num_pens1 * (retail_price1 - cost_price) = num_pens2 * (retail_price2 - cost_price) → 
  cost_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_4_l3712_371224


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3712_371290

/-- The vertex of the parabola y = -x^2 + 4x - 5 has coordinates (2, -1) -/
theorem parabola_vertex_coordinates :
  let f (x : ℝ) := -x^2 + 4*x - 5
  ∃! (a b : ℝ), (∀ x, f x ≤ f a) ∧ f a = b ∧ a = 2 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l3712_371290


namespace NUMINAMATH_CALUDE_march_first_is_thursday_l3712_371261

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that March 15th is a Thursday, prove that March 1st is also a Thursday -/
theorem march_first_is_thursday (march15 : MarchDate) 
    (h : march15.day = 15 ∧ march15.dayOfWeek = DayOfWeek.Thursday) :
    ∃ (march1 : MarchDate), march1.day = 1 ∧ march1.dayOfWeek = DayOfWeek.Thursday :=
  sorry

end NUMINAMATH_CALUDE_march_first_is_thursday_l3712_371261


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_d_equals_five_l3712_371270

theorem infinite_solutions_imply_d_equals_five (d : ℝ) :
  (∀ (S : Set ℝ), S.Infinite → (∀ x ∈ S, 3 * (5 + d * x) = 15 * x + 15)) →
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_d_equals_five_l3712_371270


namespace NUMINAMATH_CALUDE_instrument_players_fraction_l3712_371296

theorem instrument_players_fraction 
  (total_people : ℕ) 
  (two_or_more : ℕ) 
  (prob_exactly_one : ℚ) 
  (h1 : total_people = 800) 
  (h2 : two_or_more = 128) 
  (h3 : prob_exactly_one = 1/25) : 
  (↑two_or_more + ↑total_people * prob_exactly_one) / ↑total_people = 1/5 := by
sorry

end NUMINAMATH_CALUDE_instrument_players_fraction_l3712_371296


namespace NUMINAMATH_CALUDE_parallel_tangents_and_range_l3712_371249

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (2*a + 1) * x + 2 * Real.log x

theorem parallel_tangents_and_range (a : ℝ) :
  (∀ x, x > 0 → ∃ y, f a x = y) →
  ((deriv (f a)) 1 = (deriv (f a)) 3 → a = 2/3) ∧
  ((∀ x₁ ∈ Set.Ioc 0 2, ∃ x₂ ∈ Set.Icc 0 2, f a x₁ < x₂^2 - 2*x₂) → a > Real.log 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_parallel_tangents_and_range_l3712_371249


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3712_371210

/-- Given a right-angled triangle with sides a, b, and hypotenuse c,
    and a point (m,n) on the line ax+by+2c=0,
    the minimum value of m^2 + n^2 is 4. -/
theorem min_distance_to_line (a b c : ℝ) (m n : ℝ → ℝ) :
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  (∀ t, a * (m t) + b * (n t) + 2*c = 0) →
  (∃ t₀, ∀ t, (m t)^2 + (n t)^2 ≥ (m t₀)^2 + (n t₀)^2) →
  ∃ t₀, (m t₀)^2 + (n t₀)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3712_371210


namespace NUMINAMATH_CALUDE_upsilon_value_l3712_371262

theorem upsilon_value (Υ : ℤ) : 5 * (-3) = Υ - 3 → Υ = -12 := by
  sorry

end NUMINAMATH_CALUDE_upsilon_value_l3712_371262


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3712_371277

/-- 
An arithmetic sequence starting at 5, with a common difference of 3, 
and ending at 140, contains 46 terms.
-/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ → ℕ), 
    a 0 = 5 → 
    (∀ n, a (n + 1) = a n + 3) → 
    (∃ m, a m = 140) → 
    (∃ n, a n = 140 ∧ n = 45) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3712_371277
