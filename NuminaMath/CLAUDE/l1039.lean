import Mathlib

namespace NUMINAMATH_CALUDE_cube_of_product_l1039_103907

theorem cube_of_product (x y : ℝ) : (-3 * x^2 * y)^3 = -27 * x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_l1039_103907


namespace NUMINAMATH_CALUDE_fraction_inequality_l1039_103930

theorem fraction_inequality (x : ℝ) (h : x ≠ -5) :
  x / (x + 5) ≥ 0 ↔ x ∈ Set.Ici 0 ∪ Set.Iic (-5) :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1039_103930


namespace NUMINAMATH_CALUDE_watch_cost_price_l1039_103964

theorem watch_cost_price (loss_percentage : ℚ) (gain_percentage : ℚ) (price_difference : ℚ) :
  loss_percentage = 21/100 →
  gain_percentage = 4/100 →
  price_difference = 140 →
  ∃ (cost_price : ℚ),
    cost_price * (1 - loss_percentage) + price_difference = cost_price * (1 + gain_percentage) ∧
    cost_price = 560 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1039_103964


namespace NUMINAMATH_CALUDE_tangent_lines_and_intersection_l1039_103932

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (-3, 2)

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = -3
def tangent_line_2 (x y : ℝ) : Prop := 3*x + 4*y + 1 = 0

-- Define the circle with diameter PC
def circle_PC (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x - 2*y - 1 = 0

theorem tangent_lines_and_intersection (x y : ℝ) :
  (∀ x y, circle_C x y → (tangent_line_1 x ∨ tangent_line_2 x y)) ∧
  (∃ A B : ℝ × ℝ, 
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    circle_PC A.1 A.2 ∧ circle_PC B.1 B.2 ∧
    line_AB A.1 A.2 ∧ line_AB B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (8*Real.sqrt 5 / 5)^2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_and_intersection_l1039_103932


namespace NUMINAMATH_CALUDE_side_significant_digits_l1039_103997

-- Define the area of the square
def square_area : Real := 3.0625

-- Define the precision of the area measurement
def area_precision : Real := 0.001

-- Define a function to calculate the number of significant digits
def count_significant_digits (x : Real) : Nat :=
  sorry

-- Theorem statement
theorem side_significant_digits :
  let side := Real.sqrt square_area
  count_significant_digits side = 3 :=
sorry

end NUMINAMATH_CALUDE_side_significant_digits_l1039_103997


namespace NUMINAMATH_CALUDE_transform_range_transform_uniform_l1039_103978

/-- A uniform random variable in the interval [0,1] -/
def uniform_01 : Type := {x : ℝ // 0 ≤ x ∧ x ≤ 1}

/-- The transformation function -/
def transform (a₁ : uniform_01) : ℝ := a₁.val * 5 - 2

/-- Theorem stating that the transformation maps [0,1] to [-2,3] -/
theorem transform_range :
  ∀ (a₁ : uniform_01), -2 ≤ transform a₁ ∧ transform a₁ ≤ 3 := by
  sorry

/-- Theorem stating that the transformation preserves uniformity -/
theorem transform_uniform :
  uniform_01 → {x : ℝ // -2 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_transform_range_transform_uniform_l1039_103978


namespace NUMINAMATH_CALUDE_train_speed_conversion_l1039_103974

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The speed of the train in meters per second -/
def train_speed_mps : ℝ := 30.002399999999998

/-- Theorem stating that the train's speed in km/h is equal to 108.00863999999999 -/
theorem train_speed_conversion :
  train_speed_mps * mps_to_kmph = 108.00863999999999 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l1039_103974


namespace NUMINAMATH_CALUDE_calculate_expression_l1039_103917

theorem calculate_expression : 3.75 - 1.267 + 0.48 = 2.963 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1039_103917


namespace NUMINAMATH_CALUDE_park_trees_theorem_l1039_103923

/-- The number of dogwood trees remaining in the park after a day's work -/
def remaining_trees (first_part : ℝ) (second_part : ℝ) (third_part : ℝ) 
  (trees_cut : ℝ) (trees_planted : ℝ) : ℝ :=
  first_part + second_part + third_part - trees_cut + trees_planted

/-- Theorem stating the number of remaining trees after the day's work -/
theorem park_trees_theorem (first_part : ℝ) (second_part : ℝ) (third_part : ℝ) 
  (trees_cut : ℝ) (trees_planted : ℝ) :
  first_part = 5.0 →
  second_part = 4.0 →
  third_part = 6.0 →
  trees_cut = 7.0 →
  trees_planted = 3.0 →
  remaining_trees first_part second_part third_part trees_cut trees_planted = 11.0 :=
by
  sorry

#eval remaining_trees 5.0 4.0 6.0 7.0 3.0

end NUMINAMATH_CALUDE_park_trees_theorem_l1039_103923


namespace NUMINAMATH_CALUDE_total_juice_boxes_for_school_year_l1039_103953

-- Define the structure for a child
structure Child where
  name : String
  juiceBoxesPerWeek : ℕ
  schoolWeeks : ℕ

-- Define Peyton's children
def john : Child := { name := "John", juiceBoxesPerWeek := 10, schoolWeeks := 16 }
def samantha : Child := { name := "Samantha", juiceBoxesPerWeek := 5, schoolWeeks := 14 }
def heather : Child := { name := "Heather", juiceBoxesPerWeek := 11, schoolWeeks := 15 }

def children : List Child := [john, samantha, heather]

-- Function to calculate total juice boxes for a child
def totalJuiceBoxes (child : Child) : ℕ :=
  child.juiceBoxesPerWeek * child.schoolWeeks

-- Theorem to prove
theorem total_juice_boxes_for_school_year :
  (children.map totalJuiceBoxes).sum = 395 := by
  sorry

end NUMINAMATH_CALUDE_total_juice_boxes_for_school_year_l1039_103953


namespace NUMINAMATH_CALUDE_parking_space_difference_l1039_103916

/-- Represents a parking garage with four levels -/
structure ParkingGarage where
  level1 : Nat
  level2 : Nat
  level3 : Nat
  level4 : Nat

/-- Theorem stating the difference in parking spaces between the third and fourth levels -/
theorem parking_space_difference (garage : ParkingGarage) : 
  garage.level1 = 90 →
  garage.level2 = garage.level1 + 8 →
  garage.level3 = garage.level2 + 12 →
  garage.level1 + garage.level2 + garage.level3 + garage.level4 = 299 →
  garage.level3 - garage.level4 = 109 := by
  sorry

end NUMINAMATH_CALUDE_parking_space_difference_l1039_103916


namespace NUMINAMATH_CALUDE_inequality_proof_l1039_103992

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c = 1) : 
  Real.sqrt (a * b / (c + a * b)) + 
  Real.sqrt (b * c / (a + b * c)) + 
  Real.sqrt (c * a / (b + c * a)) ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1039_103992


namespace NUMINAMATH_CALUDE_chantal_profit_l1039_103935

def sweater_profit (balls_per_sweater : ℕ) (yarn_cost : ℕ) (sell_price : ℕ) (num_sweaters : ℕ) : ℕ :=
  let total_balls := balls_per_sweater * num_sweaters
  let total_cost := total_balls * yarn_cost
  let total_revenue := sell_price * num_sweaters
  total_revenue - total_cost

theorem chantal_profit :
  sweater_profit 4 6 35 28 = 308 := by
  sorry

end NUMINAMATH_CALUDE_chantal_profit_l1039_103935


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_not_square_l1039_103941

theorem product_of_five_consecutive_not_square :
  ∀ a : ℕ, a > 0 →
    ¬∃ n : ℕ, a * (a + 1) * (a + 2) * (a + 3) * (a + 4) = n^2 :=
by sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_not_square_l1039_103941


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1039_103955

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 6) * x^2 - 8 * x + 9 = 0) ↔ (a ≤ 70 / 9 ∧ a ≠ 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1039_103955


namespace NUMINAMATH_CALUDE_base_eight_53_equals_43_l1039_103911

/-- Converts a two-digit base-eight number to base-ten. -/
def baseEightToBaseTen (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  tens * 8 + ones

/-- The base-eight number 53 is equal to 43 in base-ten. -/
theorem base_eight_53_equals_43 : baseEightToBaseTen 53 = 43 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_53_equals_43_l1039_103911


namespace NUMINAMATH_CALUDE_dog_bones_total_l1039_103999

theorem dog_bones_total (initial_bones dug_up_bones : ℕ) 
  (h1 : initial_bones = 493) 
  (h2 : dug_up_bones = 367) : 
  initial_bones + dug_up_bones = 860 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_total_l1039_103999


namespace NUMINAMATH_CALUDE_unique_seventh_digit_l1039_103994

def is_seven_digit (n : ℕ) : Prop := 1000000 ≤ n ∧ n ≤ 9999999

def digit_sum (n : ℕ) : ℕ := sorry

theorem unique_seventh_digit (a b : ℕ) (h1 : is_seven_digit a) (h2 : b = digit_sum a) 
  (h3 : is_seven_digit (a - b)) (h4 : ∃ (d : Fin 7 → ℕ), 
    (∀ i, d i ∈ ({1, 2, 3, 4, 6, 7} : Set ℕ)) ∧ 
    (∃ j, d j ∉ ({1, 2, 3, 4, 6, 7} : Set ℕ)) ∧
    (a - b = d 0 * 1000000 + d 1 * 100000 + d 2 * 10000 + d 3 * 1000 + d 4 * 100 + d 5 * 10 + d 6)) :
  ∃! x, x ∉ ({1, 2, 3, 4, 6, 7} : Set ℕ) ∧ x < 10 ∧ 
    (a - b = x * 1000000 + 1 * 100000 + 2 * 10000 + 3 * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + x * 100000 + 2 * 10000 + 3 * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + x * 10000 + 3 * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + x * 1000 + 4 * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + 4 * 1000 + x * 100 + 6 * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + 4 * 1000 + 6 * 100 + x * 10 + 7 ∨
     a - b = 1 * 1000000 + 2 * 100000 + 3 * 10000 + 4 * 1000 + 6 * 100 + 7 * 10 + x) :=
by sorry

end NUMINAMATH_CALUDE_unique_seventh_digit_l1039_103994


namespace NUMINAMATH_CALUDE_factor_expression_l1039_103996

theorem factor_expression (x : ℝ) : 63 * x + 28 = 7 * (9 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1039_103996


namespace NUMINAMATH_CALUDE_product_division_theorem_l1039_103939

theorem product_division_theorem :
  ∃ x : ℕ, (400 * 7000 : ℕ) = x * (100^1) ∧ x = 28000 := by sorry

end NUMINAMATH_CALUDE_product_division_theorem_l1039_103939


namespace NUMINAMATH_CALUDE_work_completion_theorem_l1039_103906

theorem work_completion_theorem (total_work : ℝ) :
  (34 : ℝ) * 18 * total_work = 17 * 36 * total_work := by
  sorry

#check work_completion_theorem

end NUMINAMATH_CALUDE_work_completion_theorem_l1039_103906


namespace NUMINAMATH_CALUDE_range_of_m_l1039_103975

-- Define the conditions
def p (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  ((∀ x, q x m → p x) ∧ (∃ x, p x ∧ ¬q x m) → m ≥ 9) ∧
  ((∀ x, ¬q x m → ¬p x) ∧ (∃ x, ¬p x ∧ q x m) → 0 < m ∧ m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1039_103975


namespace NUMINAMATH_CALUDE_train_length_l1039_103968

/-- Given a train traveling at constant speed through a tunnel, this theorem
    proves the length of the train based on the given conditions. -/
theorem train_length
  (tunnel_length : ℝ)
  (total_time : ℝ)
  (light_time : ℝ)
  (h1 : tunnel_length = 310)
  (h2 : total_time = 18)
  (h3 : light_time = 8)
  (h4 : total_time > 0)
  (h5 : light_time > 0)
  (h6 : light_time < total_time) :
  ∃ (train_length : ℝ),
    train_length = 248 ∧
    (tunnel_length + train_length) / total_time = train_length / light_time :=
by sorry


end NUMINAMATH_CALUDE_train_length_l1039_103968


namespace NUMINAMATH_CALUDE_isabellas_house_paintable_area_l1039_103958

/-- Represents the dimensions of a bedroom -/
structure BedroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total paintable wall area in all bedrooms -/
def totalPaintableArea (dimensions : BedroomDimensions) (numBedrooms : ℕ) (nonPaintableArea : ℝ) : ℝ :=
  let wallArea := 2 * (dimensions.length * dimensions.height + dimensions.width * dimensions.height)
  let paintableAreaPerRoom := wallArea - nonPaintableArea
  numBedrooms * paintableAreaPerRoom

/-- Theorem stating that the total paintable wall area in Isabella's house is 1194 square feet -/
theorem isabellas_house_paintable_area :
  let dimensions : BedroomDimensions := { length := 15, width := 11, height := 9 }
  let numBedrooms : ℕ := 3
  let nonPaintableArea : ℝ := 70
  totalPaintableArea dimensions numBedrooms nonPaintableArea = 1194 := by
  sorry


end NUMINAMATH_CALUDE_isabellas_house_paintable_area_l1039_103958


namespace NUMINAMATH_CALUDE_train_length_l1039_103973

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 7 → ∃ length : ℝ, 
  (length ≥ 116.68 ∧ length ≤ 116.70) ∧ 
  length = speed * 1000 / 3600 * time :=
sorry

end NUMINAMATH_CALUDE_train_length_l1039_103973


namespace NUMINAMATH_CALUDE_existence_of_non_divisible_k_l1039_103924

theorem existence_of_non_divisible_k (a b c n : ℤ) (h : n ≥ 3) :
  ∃ k : ℤ, ¬(n ∣ (k + a)) ∧ ¬(n ∣ (k + b)) ∧ ¬(n ∣ (k + c)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_divisible_k_l1039_103924


namespace NUMINAMATH_CALUDE_solve_for_q_l1039_103988

theorem solve_for_q (n m q : ℚ) 
  (h1 : 5/6 = n/60)
  (h2 : 5/6 = (m - n)/66)
  (h3 : 5/6 = (q - m)/150) : 
  q = 230 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l1039_103988


namespace NUMINAMATH_CALUDE_ant_return_probability_l1039_103989

/-- Represents the probability of an ant returning to its starting vertex
    after n steps on a regular tetrahedron. -/
def P (n : ℕ) : ℚ :=
  1/4 - 1/4 * (-1/3)^(n-1)

/-- The probability of an ant returning to its starting vertex
    after 6 steps on a regular tetrahedron with edge length 1. -/
theorem ant_return_probability :
  P 6 = 61/243 :=
sorry

end NUMINAMATH_CALUDE_ant_return_probability_l1039_103989


namespace NUMINAMATH_CALUDE_pam_total_fruits_l1039_103943

-- Define the given conditions
def pam_apple_bags : ℕ := 6
def pam_orange_bags : ℕ := 4
def gerald_apple_bags : ℕ := 5
def gerald_orange_bags : ℕ := 4
def gerald_apples_per_bag : ℕ := 30
def gerald_oranges_per_bag : ℕ := 25
def pam_apple_ratio : ℕ := 3
def pam_orange_ratio : ℕ := 2

-- Theorem to prove
theorem pam_total_fruits :
  pam_apple_bags * (pam_apple_ratio * gerald_apples_per_bag) +
  pam_orange_bags * (pam_orange_ratio * gerald_oranges_per_bag) = 740 := by
  sorry


end NUMINAMATH_CALUDE_pam_total_fruits_l1039_103943


namespace NUMINAMATH_CALUDE_combined_savings_difference_l1039_103938

/-- Regular price of a window -/
def regular_price : ℕ := 120

/-- Number of windows needed to get one free -/
def windows_for_free : ℕ := 5

/-- Discount per window for large purchases -/
def bulk_discount : ℕ := 10

/-- Minimum number of windows for bulk discount -/
def bulk_min : ℕ := 10

/-- Number of windows Alice needs -/
def alice_windows : ℕ := 9

/-- Number of windows Bob needs -/
def bob_windows : ℕ := 12

/-- Calculate the cost of windows with discounts applied -/
def calculate_cost (num_windows : ℕ) : ℕ :=
  let free_windows := num_windows / windows_for_free
  let paid_windows := num_windows - free_windows
  let price_per_window := if num_windows > bulk_min then regular_price - bulk_discount else regular_price
  paid_windows * price_per_window

/-- Calculate savings compared to regular price -/
def calculate_savings (num_windows : ℕ) : ℕ :=
  num_windows * regular_price - calculate_cost num_windows

/-- Theorem: Combined savings minus individual savings equals 300 -/
theorem combined_savings_difference : 
  calculate_savings (alice_windows + bob_windows) - 
  (calculate_savings alice_windows + calculate_savings bob_windows) = 300 := by
  sorry

end NUMINAMATH_CALUDE_combined_savings_difference_l1039_103938


namespace NUMINAMATH_CALUDE_problem_solution_l1039_103900

theorem problem_solution (m n : ℕ) 
  (h_pos_m : m > 0) 
  (h_pos_n : n > 0) 
  (h_inequality : m + 8 < n - 1) 
  (h_mean : (m + (m + 3) + (m + 8) + (n - 1) + (n + 3) + (2 * n - 2)) / 6 = n) 
  (h_median : (m + 8 + n - 1) / 2 = n) : 
  m + n = 47 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1039_103900


namespace NUMINAMATH_CALUDE_negation_of_existence_l1039_103940

theorem negation_of_existence (p : Prop) :
  (¬∃ (n : ℕ), n > 1 ∧ n^2 > 2^n) ↔ (∀ (n : ℕ), n > 1 → n^2 ≤ 2^n) :=
sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1039_103940


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1039_103934

theorem fraction_to_decimal : (7 : ℚ) / 125 = (56 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1039_103934


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1039_103954

theorem min_value_quadratic (x y : ℝ) : 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1039_103954


namespace NUMINAMATH_CALUDE_shoes_outside_library_l1039_103990

/-- The number of shoes for a group of people, given the number of people and shoes per person. -/
def total_shoes (num_people : ℕ) (shoes_per_person : ℕ) : ℕ :=
  num_people * shoes_per_person

/-- Theorem: For a group of 10 people, where each person wears 2 shoes,
    the total number of shoes when everyone takes them off is 20. -/
theorem shoes_outside_library :
  total_shoes 10 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shoes_outside_library_l1039_103990


namespace NUMINAMATH_CALUDE_constant_function_shifted_l1039_103925

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 5

-- State the theorem
theorem constant_function_shifted (x : ℝ) : f (x + 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_shifted_l1039_103925


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1039_103971

theorem geometric_sequence_common_ratio 
  (a : ℝ) : 
  let seq := λ (n : ℕ) => a + Real.log 3 / Real.log (2^(2^n))
  ∃ (q : ℝ), q = 1/3 ∧ ∀ (n : ℕ), seq (n+1) / seq n = q :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1039_103971


namespace NUMINAMATH_CALUDE_correct_operation_l1039_103901

theorem correct_operation (x : ℤ) : (x - 7) * 20 = -380 → (x * 7) - 20 = -104 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1039_103901


namespace NUMINAMATH_CALUDE_jaylen_vegetables_l1039_103956

theorem jaylen_vegetables (x y z g k h : ℕ) : 
  x = 5 → 
  y = 2 → 
  z = 2 * k → 
  g = (h / 2) - 3 → 
  k = 2 → 
  h = 20 → 
  x + y + z + g = 18 := by
sorry

end NUMINAMATH_CALUDE_jaylen_vegetables_l1039_103956


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l1039_103929

theorem power_equality_implies_exponent (n : ℕ) : 4^8 = 16^n → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l1039_103929


namespace NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l1039_103959

/-- Given an arithmetic sequence with first term 7 and fifteenth term 35,
    prove that the sixtieth term is 125. -/
theorem arithmetic_sequence_60th_term
  (a : ℕ → ℤ)  -- The arithmetic sequence
  (h1 : a 1 = 7)  -- First term is 7
  (h2 : a 15 = 35)  -- Fifteenth term is 35
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Arithmetic sequence property
  : a 60 = 125 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l1039_103959


namespace NUMINAMATH_CALUDE_wall_clock_ring_interval_l1039_103993

/-- Represents a wall clock that rings multiple times a day at equal intervals -/
structure WallClock where
  rings_per_day : ℕ
  minutes_per_day : ℕ

/-- Calculates the time between two consecutive rings in minutes -/
def time_between_rings (clock : WallClock) : ℕ :=
  clock.minutes_per_day / (clock.rings_per_day - 1)

theorem wall_clock_ring_interval :
  let clock : WallClock := { rings_per_day := 6, minutes_per_day := 24 * 60 }
  time_between_rings clock = 288 := by
  sorry

end NUMINAMATH_CALUDE_wall_clock_ring_interval_l1039_103993


namespace NUMINAMATH_CALUDE_negative_a_squared_cubed_div_negative_a_squared_l1039_103972

theorem negative_a_squared_cubed_div_negative_a_squared (a : ℝ) :
  (-a^2)^3 / (-a)^2 = -a^4 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_squared_cubed_div_negative_a_squared_l1039_103972


namespace NUMINAMATH_CALUDE_inverse_proportion_points_order_l1039_103913

theorem inverse_proportion_points_order (x₁ x₂ x₃ : ℝ) : 
  10 / x₁ = -5 → 10 / x₂ = 2 → 10 / x₃ = 5 → x₁ < x₃ ∧ x₃ < x₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_order_l1039_103913


namespace NUMINAMATH_CALUDE_two_out_of_three_accurate_l1039_103998

/-- The probability of an accurate forecast -/
def p_accurate : ℝ := 0.9

/-- The probability of an inaccurate forecast -/
def p_inaccurate : ℝ := 1 - p_accurate

/-- The probability of exactly 2 out of 3 forecasts being accurate -/
def p_two_accurate : ℝ := 3 * (p_accurate ^ 2 * p_inaccurate)

theorem two_out_of_three_accurate :
  p_two_accurate = 0.243 := by sorry

end NUMINAMATH_CALUDE_two_out_of_three_accurate_l1039_103998


namespace NUMINAMATH_CALUDE_darwin_money_left_l1039_103967

theorem darwin_money_left (initial_amount : ℝ) (gas_fraction : ℝ) (food_fraction : ℝ) : 
  initial_amount = 600 →
  gas_fraction = 1/3 →
  food_fraction = 1/4 →
  initial_amount - (gas_fraction * initial_amount) - (food_fraction * (initial_amount - gas_fraction * initial_amount)) = 300 := by
sorry

end NUMINAMATH_CALUDE_darwin_money_left_l1039_103967


namespace NUMINAMATH_CALUDE_negation_of_zero_product_implication_l1039_103942

theorem negation_of_zero_product_implication :
  (∀ x y : ℝ, xy = 0 → x = 0 ∨ y = 0) ↔
  (∀ x y : ℝ, xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_zero_product_implication_l1039_103942


namespace NUMINAMATH_CALUDE_carl_accident_cost_l1039_103961

/-- Carl's car accident cost calculation -/
theorem carl_accident_cost (property_damage medical_bills : ℕ) 
  (h1 : property_damage = 40000)
  (h2 : medical_bills = 70000)
  (carl_percentage : ℚ)
  (h3 : carl_percentage = 1/5) :
  carl_percentage * (property_damage + medical_bills : ℚ) = 22000 := by
sorry

end NUMINAMATH_CALUDE_carl_accident_cost_l1039_103961


namespace NUMINAMATH_CALUDE_tank_full_after_45_minutes_l1039_103904

/-- Represents the state of a water tank system with three pipes. -/
structure TankSystem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ

/-- Calculates the net water gain in one cycle. -/
def net_gain_per_cycle (system : TankSystem) : ℕ :=
  system.pipeA_rate + system.pipeB_rate - system.pipeC_rate

/-- Calculates the number of cycles needed to fill the tank. -/
def cycles_to_fill (system : TankSystem) : ℕ :=
  system.capacity / net_gain_per_cycle system

/-- Calculates the time in minutes to fill the tank. -/
def time_to_fill (system : TankSystem) : ℕ :=
  cycles_to_fill system * 3

/-- Theorem stating that the given tank system will be full after 45 minutes. -/
theorem tank_full_after_45_minutes (system : TankSystem)
  (h_capacity : system.capacity = 750)
  (h_pipeA : system.pipeA_rate = 40)
  (h_pipeB : system.pipeB_rate = 30)
  (h_pipeC : system.pipeC_rate = 20) :
  time_to_fill system = 45 := by
  sorry

end NUMINAMATH_CALUDE_tank_full_after_45_minutes_l1039_103904


namespace NUMINAMATH_CALUDE_inequality_proof_l1039_103902

-- Define the logarithm function with base 1/8
noncomputable def log_base_1_8 (x : ℝ) : ℝ := Real.log x / Real.log (1/8)

-- State the theorem
theorem inequality_proof (x : ℝ) (h1 : x ≥ 1/2) (h2 : x < 1) :
  9.244 * Real.sqrt (1 - 9 * (log_base_1_8 x)^2) > 1 - 4 * log_base_1_8 x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1039_103902


namespace NUMINAMATH_CALUDE_min_value_and_y_l1039_103982

theorem min_value_and_y (x y z : ℝ) (h : 2*x - 3*y + z = 3) :
  ∃ (min_val : ℝ), 
    (∀ x' y' z' : ℝ, 2*x' - 3*y' + z' = 3 → x'^2 + (y' - 1)^2 + z'^2 ≥ min_val) ∧
    (x^2 + (y - 1)^2 + z^2 = min_val ↔ y = -2/7) ∧
    min_val = 18/7 :=
sorry

end NUMINAMATH_CALUDE_min_value_and_y_l1039_103982


namespace NUMINAMATH_CALUDE_no_two_obtuse_angles_l1039_103937

-- Define a triangle as a structure with three angles
structure Triangle where
  a : Real
  b : Real
  c : Real
  angle_sum : a + b + c = 180
  positive_angles : 0 < a ∧ 0 < b ∧ 0 < c

-- Theorem: A triangle cannot have two obtuse angles
theorem no_two_obtuse_angles (t : Triangle) : ¬(t.a > 90 ∧ t.b > 90) ∧ ¬(t.a > 90 ∧ t.c > 90) ∧ ¬(t.b > 90 ∧ t.c > 90) := by
  sorry

end NUMINAMATH_CALUDE_no_two_obtuse_angles_l1039_103937


namespace NUMINAMATH_CALUDE_monkeys_eating_birds_l1039_103991

theorem monkeys_eating_birds (initial_monkeys initial_birds : ℕ) 
  (h1 : initial_monkeys = 6)
  (h2 : initial_birds = 6)
  (h3 : ∃ (monkeys_ate : ℕ), 
    (initial_monkeys : ℚ) / (initial_monkeys + initial_birds - monkeys_ate) = 3/5) :
  ∃ (monkeys_ate : ℕ), monkeys_ate = 2 := by
sorry

end NUMINAMATH_CALUDE_monkeys_eating_birds_l1039_103991


namespace NUMINAMATH_CALUDE_expression_value_l1039_103966

theorem expression_value : 
  let x : ℤ := 25
  let y : ℤ := 30
  let z : ℤ := 10
  (x - (y - z)) - ((x - y) - z) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1039_103966


namespace NUMINAMATH_CALUDE_f_constant_iff_max_value_expression_exists_max_value_expression_l1039_103918

-- Part 1
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem f_constant_iff (x : ℝ) : (∀ y ∈ Set.Icc (-3) 1, f y = f x) ↔ x ∈ Set.Icc (-3) 1 := by sorry

-- Part 2
theorem max_value_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  Real.sqrt 2 * x + Real.sqrt 2 * y + Real.sqrt 5 * z ≤ 3 := by sorry

theorem exists_max_value_expression :
  ∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ Real.sqrt 2 * x + Real.sqrt 2 * y + Real.sqrt 5 * z = 3 := by sorry

end NUMINAMATH_CALUDE_f_constant_iff_max_value_expression_exists_max_value_expression_l1039_103918


namespace NUMINAMATH_CALUDE_compare_M_and_N_range_of_m_l1039_103986

-- Problem 1
theorem compare_M_and_N : ∀ x : ℝ, 2 * x^2 + 1 > x^2 + 2*x - 1 := by sorry

-- Problem 2
theorem range_of_m : 
  (∀ m : ℝ, (∀ x : ℝ, 2*m ≤ x ∧ x ≤ m+1 → -1 ≤ x ∧ x ≤ 1) → -1/2 ≤ m ∧ m ≤ 0) := by sorry

end NUMINAMATH_CALUDE_compare_M_and_N_range_of_m_l1039_103986


namespace NUMINAMATH_CALUDE_chantel_final_bracelets_l1039_103977

def bracelets_made_first_period : ℕ := 5 * 2
def bracelets_given_school : ℕ := 3
def bracelets_made_second_period : ℕ := 4 * 3
def bracelets_given_soccer : ℕ := 6

theorem chantel_final_bracelets :
  bracelets_made_first_period - bracelets_given_school + bracelets_made_second_period - bracelets_given_soccer = 13 := by
  sorry

end NUMINAMATH_CALUDE_chantel_final_bracelets_l1039_103977


namespace NUMINAMATH_CALUDE_smallest_value_complex_sum_l1039_103922

theorem smallest_value_complex_sum (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_omega_cube : ω^3 = 1)
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
  (∀ (x y z : ℤ) (h_xyz_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z),
    Complex.abs (↑x + ↑y * ω + ↑z * ω^2) ≥ m) ∧
  (∃ (p q r : ℤ) (h_pqr_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r),
    Complex.abs (↑p + ↑q * ω + ↑r * ω^2) = m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_complex_sum_l1039_103922


namespace NUMINAMATH_CALUDE_sin_cos_problem_l1039_103909

theorem sin_cos_problem (x : ℝ) (h : Real.sin x = 3 * Real.cos x) :
  Real.sin x * Real.cos x = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_problem_l1039_103909


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l1039_103962

theorem divisibility_by_seven (a b : ℕ) : 
  (7 ∣ (a * b)) → (7 ∣ a) ∨ (7 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l1039_103962


namespace NUMINAMATH_CALUDE_grid_arrangement_impossibility_l1039_103927

theorem grid_arrangement_impossibility :
  ¬ ∃ (grid : Fin 25 → Fin 41 → ℤ),
    (∀ i j i' j', grid i j = grid i' j' → (i = i' ∧ j = j')) ∧
    (∀ i j,
      (i.val + 1 < 25 → |grid i j - grid ⟨i.val + 1, sorry⟩ j| ≤ 16) ∧
      (j.val + 1 < 41 → |grid i j - grid i ⟨j.val + 1, sorry⟩| ≤ 16)) :=
sorry

end NUMINAMATH_CALUDE_grid_arrangement_impossibility_l1039_103927


namespace NUMINAMATH_CALUDE_product_of_one_plus_roots_l1039_103963

theorem product_of_one_plus_roots (u v w : ℝ) : 
  u^3 - 15*u^2 + 25*u - 12 = 0 ∧ 
  v^3 - 15*v^2 + 25*v - 12 = 0 ∧ 
  w^3 - 15*w^2 + 25*w - 12 = 0 → 
  (1 + u) * (1 + v) * (1 + w) = 29 := by
sorry

end NUMINAMATH_CALUDE_product_of_one_plus_roots_l1039_103963


namespace NUMINAMATH_CALUDE_f_of_3_equals_neg_9_l1039_103979

/-- Given a function f(x) = 2x^7 - 3x^3 + 4x - 6 where f(-3) = -3, prove that f(3) = -9 -/
theorem f_of_3_equals_neg_9 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2*x^7 - 3*x^3 + 4*x - 6)
  (h2 : f (-3) = -3) : 
  f 3 = -9 := by
sorry


end NUMINAMATH_CALUDE_f_of_3_equals_neg_9_l1039_103979


namespace NUMINAMATH_CALUDE_light_bulb_packs_theorem_l1039_103949

/-- Calculates the number of light bulb packs needed given the number of bulbs required in each room -/
def light_bulb_packs_needed (bedroom bathroom kitchen basement : ℕ) : ℕ :=
  let total_without_garage := bedroom + bathroom + kitchen + basement
  let garage := total_without_garage / 2
  let total := total_without_garage + garage
  (total + 1) / 2

/-- Theorem stating that given the specific number of light bulbs needed in each room,
    the number of packs needed is 6 -/
theorem light_bulb_packs_theorem :
  light_bulb_packs_needed 2 1 1 4 = 6 := by
  sorry

#eval light_bulb_packs_needed 2 1 1 4

end NUMINAMATH_CALUDE_light_bulb_packs_theorem_l1039_103949


namespace NUMINAMATH_CALUDE_sum_in_base8_l1039_103936

/-- Converts a base-8 number represented as a list of digits to a natural number. -/
def base8ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits. -/
def natToBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: natToBase8 (n / 8)

theorem sum_in_base8 :
  let a := base8ToNat [3, 5, 6]  -- 653₈
  let b := base8ToNat [4, 7, 2]  -- 274₈
  let c := base8ToNat [7, 6, 1]  -- 167₈
  natToBase8 (a + b + c) = [6, 5, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base8_l1039_103936


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1039_103931

theorem gcd_of_three_numbers : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1039_103931


namespace NUMINAMATH_CALUDE_largest_among_a_ab_aplusb_l1039_103910

theorem largest_among_a_ab_aplusb (a b : ℚ) (h : b < 0) :
  (a - b) = max a (max (a - b) (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_largest_among_a_ab_aplusb_l1039_103910


namespace NUMINAMATH_CALUDE_stone_68_is_10_l1039_103952

/-- The number of stones in the circle -/
def n : ℕ := 15

/-- The length of a full cycle (clockwise + counterclockwise) -/
def cycle_length : ℕ := n + (n - 1)

/-- The stone number corresponding to a given count -/
def stone_number (count : ℕ) : ℕ :=
  let effective_count := count % cycle_length
  if effective_count ≤ n then effective_count else n - (effective_count - n)

theorem stone_68_is_10 : stone_number 68 = 10 := by sorry

end NUMINAMATH_CALUDE_stone_68_is_10_l1039_103952


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1039_103976

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a^2 + c^2 = b^2 + Real.sqrt 2 * a * c →
  c = Real.sqrt 3 + 1 →
  Real.sin A = 1/2 →
  (B = π/4 ∧ 1/2 * a * b * Real.sin C = (Real.sqrt 3 + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1039_103976


namespace NUMINAMATH_CALUDE_calculation_proof_l1039_103908

theorem calculation_proof :
  ((-1/3 : ℚ) - 15 + (-2/3) + 1 = -15) ∧
  (16 / (-2)^3 - (-1/8) * (-4 : ℚ) = -5/2) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1039_103908


namespace NUMINAMATH_CALUDE_total_gifts_needed_l1039_103957

/-- The number of teams participating in the world cup -/
def num_teams : ℕ := 12

/-- The number of invited members per team who receive a gift -/
def members_per_team : ℕ := 4

/-- Theorem stating the total number of gifts needed for the event -/
theorem total_gifts_needed : num_teams * members_per_team = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_gifts_needed_l1039_103957


namespace NUMINAMATH_CALUDE_intersection_fixed_point_l1039_103933

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line
def line (k n x y : ℝ) : Prop := y = k * x + n

-- Define the intersection points
def intersection (k n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line k n x₁ y₁ ∧ line k n x₂ y₂

-- Main theorem
theorem intersection_fixed_point (k n x₁ y₁ x₂ y₂ : ℝ) 
  (hk : k ≠ 0)
  (h_int : intersection k n x₁ y₁ x₂ y₂)
  (h_slope : 3 * (y₁ / x₁ + y₂ / x₂) = 8 * k) :
  n = 1/2 ∨ n = -1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_fixed_point_l1039_103933


namespace NUMINAMATH_CALUDE_unknown_number_l1039_103981

theorem unknown_number (x n : ℝ) : 
  (5 * x + n = 10 * x - 17) → (x = 4) → (n = 3) := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_l1039_103981


namespace NUMINAMATH_CALUDE_range_of_m_l1039_103950

theorem range_of_m (x m : ℝ) : 
  (∀ x, (1 < x ∧ x < m - 2) → (1 < x ∧ x < 4)) ∧ 
  (∃ x, (1 < x ∧ x < 4) ∧ ¬(1 < x ∧ x < m - 2)) → 
  m ∈ Set.Ioi 6 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1039_103950


namespace NUMINAMATH_CALUDE_distance_to_work_is_18_l1039_103946

/-- The distance Esther drives to work -/
def distance_to_work : ℝ := 18

/-- The average speed to work in miles per hour -/
def speed_to_work : ℝ := 45

/-- The average speed from work in miles per hour -/
def speed_from_work : ℝ := 30

/-- The total commute time in hours -/
def total_commute_time : ℝ := 1

/-- Theorem stating that the distance to work is 18 miles given the conditions -/
theorem distance_to_work_is_18 :
  (distance_to_work / speed_to_work) + (distance_to_work / speed_from_work) = total_commute_time :=
by sorry

end NUMINAMATH_CALUDE_distance_to_work_is_18_l1039_103946


namespace NUMINAMATH_CALUDE_least_sum_m_n_l1039_103965

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (m.val > 0 ∧ n.val > 0) ∧
  (Nat.gcd (m.val + n.val) 210 = 1) ∧
  (∃ (k : ℕ), m.val ^ m.val = k * (n.val ^ n.val)) ∧
  (¬ ∃ (l : ℕ), m.val = l * n.val) ∧
  (m.val + n.val = 407) ∧
  (∀ (p q : ℕ+), 
    (p.val > 0 ∧ q.val > 0) →
    (Nat.gcd (p.val + q.val) 210 = 1) →
    (∃ (k : ℕ), p.val ^ p.val = k * (q.val ^ q.val)) →
    (¬ ∃ (l : ℕ), p.val = l * q.val) →
    (p.val + q.val ≥ 407)) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l1039_103965


namespace NUMINAMATH_CALUDE_visitor_growth_l1039_103984

/-- Represents the growth of visitors at a tourist attraction from January to March. -/
theorem visitor_growth (initial_visitors final_visitors : ℕ) (x : ℝ) :
  initial_visitors = 60000 →
  final_visitors = 150000 →
  (initial_visitors : ℝ) / 10000 * (1 + x)^2 = (final_visitors : ℝ) / 10000 →
  6 * (1 + x)^2 = 15 := by
  sorry

#check visitor_growth

end NUMINAMATH_CALUDE_visitor_growth_l1039_103984


namespace NUMINAMATH_CALUDE_kevin_cards_problem_l1039_103969

/-- Given that Kevin finds 47 cards and ends up with 54 cards, prove that he started with 7 cards. -/
theorem kevin_cards_problem (found_cards : ℕ) (total_cards : ℕ) (h1 : found_cards = 47) (h2 : total_cards = 54) :
  total_cards - found_cards = 7 := by
sorry

end NUMINAMATH_CALUDE_kevin_cards_problem_l1039_103969


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_symmetry_l1039_103985

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of elements in row n of Pascal's triangle -/
def rowLength (n : ℕ) : ℕ := n + 1

theorem pascal_triangle_row20_symmetry :
  let n := 20
  let k := 5
  let row_length := rowLength n
  binomial n (k - 1) = binomial n (row_length - k) ∧
  binomial n (k - 1) = 4845 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_symmetry_l1039_103985


namespace NUMINAMATH_CALUDE_eves_diner_purchase_l1039_103919

/-- The cost of a sandwich at Eve's Diner -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda at Eve's Diner -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 7

/-- The number of sodas purchased -/
def num_sodas : ℕ := 12

/-- The total cost of the purchase at Eve's Diner -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem eves_diner_purchase :
  total_cost = 64 := by sorry

end NUMINAMATH_CALUDE_eves_diner_purchase_l1039_103919


namespace NUMINAMATH_CALUDE_min_minutes_for_plan_c_l1039_103960

/-- Represents the cost of a cell phone plan in cents -/
def PlanCost (flatFee minutes perMinute : ℕ) : ℕ := flatFee * 100 + minutes * perMinute

/-- Checks if Plan C is cheaper than both Plan A and Plan B for a given number of minutes -/
def IsPlanCCheaper (minutes : ℕ) : Prop :=
  PlanCost 15 minutes 10 < PlanCost 0 minutes 15 ∧ 
  PlanCost 15 minutes 10 < PlanCost 25 minutes 8

theorem min_minutes_for_plan_c : ∀ m : ℕ, m ≥ 301 → IsPlanCCheaper m ∧ ∀ n : ℕ, n < 301 → ¬IsPlanCCheaper n := by
  sorry

end NUMINAMATH_CALUDE_min_minutes_for_plan_c_l1039_103960


namespace NUMINAMATH_CALUDE_geometry_propositions_l1039_103944

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (intersect : Line → Line → Prop)

-- Theorem statement
theorem geometry_propositions 
  (a b c : Line) (α β : Plane) :
  (∀ (α β : Plane) (c : Line), 
    parallel_plane α β → perpendicular_plane c α → perpendicular_plane c β) ∧
  (∀ (a b c : Line),
    perpendicular a c → perpendicular b c → 
    (parallel a b ∨ skew a b ∨ intersect a b)) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1039_103944


namespace NUMINAMATH_CALUDE_min_value_of_f_l1039_103951

def f (x : ℝ) := |3 - x| + |x - 2|

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1039_103951


namespace NUMINAMATH_CALUDE_chess_tournament_players_l1039_103948

/-- Chess tournament with specific conditions -/
structure ChessTournament where
  n : ℕ
  total_score : ℕ
  two_player_score : ℕ
  avg_score_others : ℕ
  odd_players : Odd n
  two_player_score_eq : two_player_score = 16
  even_avg_score : Even avg_score_others
  total_score_eq : total_score = n * (n - 1)

/-- Theorem stating that under given conditions, the number of players is 9 -/
theorem chess_tournament_players (t : ChessTournament) : t.n = 9 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l1039_103948


namespace NUMINAMATH_CALUDE_chess_tournament_girls_l1039_103983

theorem chess_tournament_girls (n : ℕ) (x : ℕ) : 
  (n > 0) →  -- number of girls is positive
  (2 * n * x + 16 = (n + 2) * (n + 1)) →  -- total points equation
  (x > 0) →  -- each girl's score is positive
  (n = 7 ∨ n = 14) := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_girls_l1039_103983


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1039_103980

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | 2*x - 3 < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 3/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1039_103980


namespace NUMINAMATH_CALUDE_v_closed_under_multiplication_l1039_103915

/-- The set of cubes of positive integers -/
def v : Set ℕ := {n | ∃ m : ℕ+, n = m^3}

/-- Proof that v is closed under multiplication -/
theorem v_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v := by
  sorry

end NUMINAMATH_CALUDE_v_closed_under_multiplication_l1039_103915


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l1039_103912

theorem unique_solution_quadratic_equation :
  ∃! x : ℝ, (2016 + 3*x)^2 = (3*x)^2 ∧ x = -336 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l1039_103912


namespace NUMINAMATH_CALUDE_drama_club_ticket_sales_l1039_103921

theorem drama_club_ticket_sales 
  (total_tickets : ℕ) 
  (adult_price student_price : ℚ) 
  (total_amount : ℚ) 
  (h1 : total_tickets = 1500)
  (h2 : adult_price = 12)
  (h3 : student_price = 6)
  (h4 : total_amount = 16200) :
  ∃ (adult_tickets student_tickets : ℕ),
    adult_tickets + student_tickets = total_tickets ∧
    adult_price * adult_tickets + student_price * student_tickets = total_amount ∧
    student_tickets = 300 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_ticket_sales_l1039_103921


namespace NUMINAMATH_CALUDE_window_pane_length_l1039_103928

theorem window_pane_length 
  (num_panes : ℕ) 
  (pane_width : ℝ) 
  (total_area : ℝ) : ℝ :=
  let pane_area := total_area / num_panes
  let pane_length := pane_area / pane_width
  have h1 : num_panes = 8 := by sorry
  have h2 : pane_width = 8 := by sorry
  have h3 : total_area = 768 := by sorry
  have h4 : pane_length = 12 := by sorry
  pane_length

#check window_pane_length

end NUMINAMATH_CALUDE_window_pane_length_l1039_103928


namespace NUMINAMATH_CALUDE_stating_sticks_at_100th_stage_l1039_103970

/-- 
Given a sequence where:
- The first term is 4
- Each subsequent term increases by 4
This function calculates the nth term of the sequence
-/
def sticksAtStage (n : ℕ) : ℕ := 4 + 4 * (n - 1)

/-- 
Theorem stating that the 100th stage of the stick pattern contains 400 sticks
-/
theorem sticks_at_100th_stage : sticksAtStage 100 = 400 := by sorry

end NUMINAMATH_CALUDE_stating_sticks_at_100th_stage_l1039_103970


namespace NUMINAMATH_CALUDE_parallelogram_altitude_l1039_103945

/-- Represents a parallelogram ABCD with altitudes DE and DF -/
structure Parallelogram where
  -- Lengths of sides and segments
  DC : ℝ
  EB : ℝ
  DE : ℝ
  -- Condition that ABCD is a parallelogram
  is_parallelogram : True

/-- Theorem: In a parallelogram ABCD with given conditions, DF = 7 -/
theorem parallelogram_altitude (p : Parallelogram)
  (h1 : p.DC = 15)
  (h2 : p.EB = 5)
  (h3 : p.DE = 7) :
  ∃ DF : ℝ, DF = 7 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_l1039_103945


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l1039_103903

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 15 [ZMOD 25])
  (h2 : 2 * x ≡ 10 [ZMOD 25]) : 
  x^2 ≡ 0 [ZMOD 25] := by
sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l1039_103903


namespace NUMINAMATH_CALUDE_unique_rearrangement_difference_l1039_103987

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 100 * a + 10 * b + c ∧
    max a (max b c) * 100 + (a + b + c - max a (max b c) - min a (min b c)) * 10 + min a (min b c) -
    (min a (min b c) * 100 + (a + b + c - max a (max b c) - min a (min b c)) * 10 + max a (max b c)) = n

theorem unique_rearrangement_difference :
  ∃! n : ℕ, is_valid_number n :=
by sorry

end NUMINAMATH_CALUDE_unique_rearrangement_difference_l1039_103987


namespace NUMINAMATH_CALUDE_woman_birth_year_l1039_103995

theorem woman_birth_year (x : ℕ) (h1 : x > 0) (h2 : x^2 - x ≥ 1950) (h3 : x^2 - x < 2000) (h4 : x^2 ≥ 2000) : x^2 - x = 1980 := by
  sorry

end NUMINAMATH_CALUDE_woman_birth_year_l1039_103995


namespace NUMINAMATH_CALUDE_product_of_base8_digits_9876_l1039_103926

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem product_of_base8_digits_9876 :
  productOfList (toBase8 9876) = 96 :=
by sorry

end NUMINAMATH_CALUDE_product_of_base8_digits_9876_l1039_103926


namespace NUMINAMATH_CALUDE_even_function_properties_l1039_103914

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem even_function_properties (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_sum_zero : ∀ x, f x + f (2 - x) = 0) :
  is_periodic f 4 ∧ is_odd (fun x ↦ f (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_even_function_properties_l1039_103914


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l1039_103920

theorem root_difference_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0 ∧
   (r+3)^2 - k*(r+3) + 10 = 0 ∧ (s+3)^2 - k*(s+3) + 10 = 0) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l1039_103920


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l1039_103905

theorem complex_modulus_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 6 - 9*I) : 
  Complex.abs z^2 = 39/4 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l1039_103905


namespace NUMINAMATH_CALUDE_pentagon_area_l1039_103947

/-- Given integers p and q where 0 < q < p, and points P, Q, R, S, T defined by reflections,
    if the area of pentagon PQRST is 700, then 5pq - q² = 700 -/
theorem pentagon_area (p q : ℤ) (h1 : 0 < q) (h2 : q < p) 
  (h3 : (5 * p * q - q^2 : ℤ) = 700) : True := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l1039_103947
