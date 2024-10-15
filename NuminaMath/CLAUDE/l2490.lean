import Mathlib

namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2490_249046

theorem complex_expression_evaluation : 
  let z₁ : ℂ := (1 + 3 * Complex.I) / (1 - 3 * Complex.I)
  let z₂ : ℂ := (1 - 3 * Complex.I) / (1 + 3 * Complex.I)
  let z₃ : ℂ := 1 / (8 * Complex.I^3)
  z₁ + z₂ + z₃ = -1.6 + 0.125 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2490_249046


namespace NUMINAMATH_CALUDE_output_theorem_l2490_249027

/-- Represents the output of the program at each step -/
structure ProgramOutput :=
  (x : ℕ)
  (y : ℤ)

/-- The sequence of outputs from the program -/
def output_sequence : ℕ → ProgramOutput := sorry

/-- The theorem stating that when y = -10, x = 32 in the output sequence -/
theorem output_theorem :
  ∃ n : ℕ, (output_sequence n).y = -10 ∧ (output_sequence n).x = 32 := by
  sorry

end NUMINAMATH_CALUDE_output_theorem_l2490_249027


namespace NUMINAMATH_CALUDE_min_transportation_cost_l2490_249053

-- Define the problem parameters
def total_items : ℕ := 320
def water_excess : ℕ := 80
def type_a_water_capacity : ℕ := 40
def type_a_veg_capacity : ℕ := 10
def type_b_capacity : ℕ := 20
def total_trucks : ℕ := 8
def type_a_cost : ℕ := 400
def type_b_cost : ℕ := 360

-- Define the transportation cost function
def transportation_cost (num_type_a : ℕ) : ℕ :=
  type_a_cost * num_type_a + type_b_cost * (total_trucks - num_type_a)

-- Theorem statement
theorem min_transportation_cost :
  ∃ (num_water num_veg : ℕ),
    num_water + num_veg = total_items ∧
    num_water - num_veg = water_excess ∧
    (∀ (num_type_a : ℕ),
      2 ≤ num_type_a ∧ num_type_a ≤ 4 →
      type_a_water_capacity * num_type_a + type_b_capacity * (total_trucks - num_type_a) ≥ num_water ∧
      type_a_veg_capacity * num_type_a + type_b_capacity * (total_trucks - num_type_a) ≥ num_veg) ∧
    (∀ (num_type_a : ℕ),
      2 ≤ num_type_a ∧ num_type_a ≤ 4 →
      transportation_cost 2 ≤ transportation_cost num_type_a) ∧
    transportation_cost 2 = 2960 := by
  sorry

end NUMINAMATH_CALUDE_min_transportation_cost_l2490_249053


namespace NUMINAMATH_CALUDE_gcd_of_squares_l2490_249090

theorem gcd_of_squares : Nat.gcd (121^2 + 233^2 + 345^2) (120^2 + 232^2 + 346^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l2490_249090


namespace NUMINAMATH_CALUDE_oliver_ate_seventeen_fruits_l2490_249021

/-- The number of fruits Oliver ate -/
def fruits_eaten (initial_cherries initial_strawberries initial_blueberries
                  final_cherries final_strawberries final_blueberries : ℕ) : ℕ :=
  (initial_cherries - final_cherries) +
  (initial_strawberries - final_strawberries) +
  (initial_blueberries - final_blueberries)

/-- Theorem stating that Oliver ate 17 fruits in total -/
theorem oliver_ate_seventeen_fruits :
  fruits_eaten 16 10 20 6 8 15 = 17 := by
  sorry

end NUMINAMATH_CALUDE_oliver_ate_seventeen_fruits_l2490_249021


namespace NUMINAMATH_CALUDE_complement_of_M_l2490_249049

def M : Set ℝ := {x | x + 3 > 0}

theorem complement_of_M : 
  (Set.univ : Set ℝ) \ M = {x : ℝ | x ≤ -3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2490_249049


namespace NUMINAMATH_CALUDE_bob_same_color_probability_l2490_249018

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 3

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the number of marbles each person takes -/
def marbles_taken : ℕ := 3

/-- Calculates the probability of Bob getting 3 marbles of the same color -/
def prob_bob_same_color : ℚ :=
  let total_marbles := marbles_per_color * num_colors
  let total_outcomes := (total_marbles.choose marbles_taken) * 
                        ((total_marbles - marbles_taken).choose marbles_taken) * 
                        ((total_marbles - 2*marbles_taken).choose marbles_taken)
  let favorable_outcomes := num_colors * ((total_marbles - marbles_taken).choose marbles_taken)
  favorable_outcomes / total_outcomes

theorem bob_same_color_probability :
  prob_bob_same_color = 1 / 28 := by sorry

end NUMINAMATH_CALUDE_bob_same_color_probability_l2490_249018


namespace NUMINAMATH_CALUDE_vector_OA_coordinates_l2490_249058

/-- Given that O is the origin, point A is in the second quadrant,
    |OA| = 2, and ∠xOA = 150°, prove that the coordinates of vector OA are (-√3, 1). -/
theorem vector_OA_coordinates (A : ℝ × ℝ) :
  A.1 < 0 ∧ A.2 > 0 →  -- A is in the second quadrant
  A.1^2 + A.2^2 = 4 →  -- |OA| = 2
  Real.cos (150 * π / 180) = A.1 / 2 ∧ Real.sin (150 * π / 180) = A.2 / 2 →  -- ∠xOA = 150°
  A = (-Real.sqrt 3, 1) :=
by sorry

end NUMINAMATH_CALUDE_vector_OA_coordinates_l2490_249058


namespace NUMINAMATH_CALUDE_units_digit_of_l_squared_plus_two_to_l_l2490_249067

def l : ℕ := 15^2 + 2^15

theorem units_digit_of_l_squared_plus_two_to_l (l : ℕ) (h : l = 15^2 + 2^15) : 
  (l^2 + 2^l) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_l_squared_plus_two_to_l_l2490_249067


namespace NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l2490_249054

theorem right_triangle_with_hypotenuse_65 :
  ∃! (a b : ℕ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 25 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l2490_249054


namespace NUMINAMATH_CALUDE_subset_condition_implies_a_geq_three_l2490_249026

/-- Given a > 0, if the set A is a subset of set B, then a ≥ 3 -/
theorem subset_condition_implies_a_geq_three (a : ℝ) (h : a > 0) :
  ({x : ℝ | (x - 2) * (x - 3 * a - 2) < 0} ⊆ {x : ℝ | (x - 1) * (x - a^2 - 2) < 0}) →
  a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_implies_a_geq_three_l2490_249026


namespace NUMINAMATH_CALUDE_correct_first_year_caps_l2490_249057

/-- The number of caps Lilith collects per month in the first year -/
def first_year_monthly_caps : ℕ := 3

/-- The number of years Lilith has been collecting caps -/
def total_years : ℕ := 5

/-- The number of caps Lilith collects per month after the first year -/
def later_years_monthly_caps : ℕ := 5

/-- The number of caps Lilith receives each Christmas -/
def christmas_caps : ℕ := 40

/-- The number of caps Lilith loses each year -/
def yearly_lost_caps : ℕ := 15

/-- The total number of caps Lilith has collected after 5 years -/
def total_caps : ℕ := 401

theorem correct_first_year_caps : 
  first_year_monthly_caps * 12 + 
  (total_years - 1) * 12 * later_years_monthly_caps + 
  total_years * christmas_caps - 
  total_years * yearly_lost_caps = total_caps := by
  sorry

end NUMINAMATH_CALUDE_correct_first_year_caps_l2490_249057


namespace NUMINAMATH_CALUDE_max_of_4_2_neg5_l2490_249079

def find_max (a b c : Int) : Int :=
  let max1 := max a b
  max max1 c

theorem max_of_4_2_neg5 :
  find_max 4 2 (-5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_of_4_2_neg5_l2490_249079


namespace NUMINAMATH_CALUDE_geometric_mean_combined_sets_l2490_249080

theorem geometric_mean_combined_sets :
  ∀ (y₁ y₂ y₃ y₄ z₁ z₂ z₃ z₄ : ℝ),
    y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ y₄ > 0 ∧
    z₁ > 0 ∧ z₂ > 0 ∧ z₃ > 0 ∧ z₄ > 0 →
    (y₁ * y₂ * y₃ * y₄) ^ (1/4 : ℝ) = 2048 →
    (z₁ * z₂ * z₃ * z₄) ^ (1/4 : ℝ) = 8 →
    (y₁ * y₂ * y₃ * y₄ * z₁ * z₂ * z₃ * z₄) ^ (1/8 : ℝ) = 128 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_combined_sets_l2490_249080


namespace NUMINAMATH_CALUDE_parallel_vectors_l2490_249009

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def IsParallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 = t * w.1 ∧ v.2 = t * w.2

theorem parallel_vectors (k : ℝ) :
  let a : ℝ × ℝ := (-1, 1)
  let b : ℝ × ℝ := (2, 3)
  let c : ℝ × ℝ := (-2, k)
  IsParallel (a.1 + b.1, a.2 + b.2) c → k = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2490_249009


namespace NUMINAMATH_CALUDE_melanie_total_dimes_l2490_249056

def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4

theorem melanie_total_dimes : 
  initial_dimes + dimes_from_dad + dimes_from_mom = 19 := by
  sorry

end NUMINAMATH_CALUDE_melanie_total_dimes_l2490_249056


namespace NUMINAMATH_CALUDE_subset_condition_empty_intersection_l2490_249014

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem for part 1
theorem subset_condition (m : ℝ) : A ⊆ B m ↔ m ≤ -2 := by sorry

-- Theorem for part 2
theorem empty_intersection (m : ℝ) : A ∩ B m = ∅ ↔ 0 ≤ m := by sorry

end NUMINAMATH_CALUDE_subset_condition_empty_intersection_l2490_249014


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l2490_249015

theorem polynomial_expansion_problem (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  (45 * p^8 * q^2 = 120 * p^7 * q^3) → 
  (p + q = 3/4) → 
  p = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l2490_249015


namespace NUMINAMATH_CALUDE_fish_tank_problem_l2490_249055

theorem fish_tank_problem (tank1_goldfish tank2 tank3 : ℕ) : 
  tank1_goldfish = 7 →
  tank3 = 10 →
  tank2 = 3 * tank3 →
  tank2 = 2 * (tank1_goldfish + (tank1_beta : ℕ)) →
  tank1_beta = 8 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l2490_249055


namespace NUMINAMATH_CALUDE_min_speed_against_current_l2490_249032

/-- The minimum speed against the current given the following conditions:
    - Man's speed with the current is 35 km/hr
    - Speed of the current varies between 5.6 km/hr and 8.4 km/hr
    - Wind resistance provides a decelerating force between 0.1 to 0.3 times his speed -/
theorem min_speed_against_current (speed_with_current : ℝ) 
  (current_speed_min current_speed_max : ℝ) 
  (wind_resistance_min wind_resistance_max : ℝ) :
  speed_with_current = 35 →
  current_speed_min = 5.6 →
  current_speed_max = 8.4 →
  wind_resistance_min = 0.1 →
  wind_resistance_max = 0.3 →
  ∃ (speed_against_current : ℝ), 
    speed_against_current ≥ 14.7 ∧ 
    (∀ (actual_current_speed actual_wind_resistance : ℝ),
      actual_current_speed ≥ current_speed_min →
      actual_current_speed ≤ current_speed_max →
      actual_wind_resistance ≥ wind_resistance_min →
      actual_wind_resistance ≤ wind_resistance_max →
      speed_against_current ≤ speed_with_current - actual_current_speed - 
        actual_wind_resistance * (speed_with_current - actual_current_speed)) :=
by sorry

end NUMINAMATH_CALUDE_min_speed_against_current_l2490_249032


namespace NUMINAMATH_CALUDE_unique_function_property_l2490_249052

def FunctionProperty (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, n^2 + 4 * f n = f (f (f n))

theorem unique_function_property :
  ∃! f : ℤ → ℤ, FunctionProperty f ∧ ∀ n : ℤ, f n = n + 1 :=
sorry

end NUMINAMATH_CALUDE_unique_function_property_l2490_249052


namespace NUMINAMATH_CALUDE_ice_cream_volume_l2490_249060

/-- The volume of ice cream in a right circular cone topped with a hemisphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (2/3) * π * r^3
  cone_volume + hemisphere_volume = 48 * π :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l2490_249060


namespace NUMINAMATH_CALUDE_probability_seven_odd_in_ten_rolls_l2490_249083

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of rolls -/
def num_rolls : ℕ := 10

/-- The number of desired odd rolls -/
def desired_odd_rolls : ℕ := 7

/-- The probability of getting exactly 7 odd numbers in 10 rolls of a fair 6-sided die -/
theorem probability_seven_odd_in_ten_rolls :
  Nat.choose num_rolls desired_odd_rolls * prob_odd ^ desired_odd_rolls * (1 - prob_odd) ^ (num_rolls - desired_odd_rolls) = 15/128 := by
  sorry

end NUMINAMATH_CALUDE_probability_seven_odd_in_ten_rolls_l2490_249083


namespace NUMINAMATH_CALUDE_inequality_and_not_all_greater_l2490_249012

theorem inequality_and_not_all_greater (m a b x y z : ℝ) : 
  m > 0 → 
  0 < x → x < 2 → 
  0 < y → y < 2 → 
  0 < z → z < 2 → 
  ((a + m * b) / (1 + m))^2 ≤ (a^2 + m * b^2) / (1 + m) ∧ 
  ¬(x * (2 - y) > 1 ∧ y * (2 - z) > 1 ∧ z * (2 - x) > 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_not_all_greater_l2490_249012


namespace NUMINAMATH_CALUDE_triangle_shape_l2490_249037

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- positive side lengths
  (A > 0 ∧ B > 0 ∧ C > 0) →  -- positive angles
  (A + B + C = 180) →        -- sum of angles in a triangle
  (A = 30 ∨ B = 30 ∨ C = 30) →  -- one angle is 30°
  (a = 2*b ∨ b = 2*c ∨ c = 2*a) →  -- one side is twice another
  (¬(A < 90 ∧ B < 90 ∧ C < 90) ∧ (C = 90 ∨ C > 90 ∨ B > 90)) := by
sorry

end NUMINAMATH_CALUDE_triangle_shape_l2490_249037


namespace NUMINAMATH_CALUDE_sqrt_five_is_quadratic_radical_l2490_249028

/-- A number is non-negative if it's greater than or equal to zero. -/
def NonNegative (x : ℝ) : Prop := x ≥ 0

/-- A quadratic radical is an expression √x where x is non-negative. -/
def QuadraticRadical (x : ℝ) : Prop := NonNegative x

/-- Theorem: √5 is a quadratic radical. -/
theorem sqrt_five_is_quadratic_radical : QuadraticRadical 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_is_quadratic_radical_l2490_249028


namespace NUMINAMATH_CALUDE_nina_running_distance_l2490_249085

theorem nina_running_distance : 
  0.08333333333333333 + 0.08333333333333333 + 0.6666666666666666 = 0.8333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_nina_running_distance_l2490_249085


namespace NUMINAMATH_CALUDE_average_exercise_days_l2490_249036

def exercise_data : List (Nat × Nat) := [
  (1, 1), (2, 3), (3, 2), (4, 6), (5, 8), (6, 3), (7, 2)
]

def total_exercise_days : Nat :=
  (exercise_data.map (fun (days, freq) => days * freq)).sum

def total_students : Nat :=
  (exercise_data.map (fun (_, freq) => freq)).sum

theorem average_exercise_days :
  (total_exercise_days : ℚ) / (total_students : ℚ) = 436 / 100 := by sorry

end NUMINAMATH_CALUDE_average_exercise_days_l2490_249036


namespace NUMINAMATH_CALUDE_cloth_sale_cost_price_l2490_249040

/-- Given the conditions of a cloth sale, prove the cost price per metre -/
theorem cloth_sale_cost_price 
  (total_metres : ℕ) 
  (total_selling_price : ℕ) 
  (loss_per_metre : ℕ) 
  (discount_rate : ℚ) 
  (tax_rate : ℚ) 
  (h1 : total_metres = 300)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_metre = 5)
  (h4 : discount_rate = 1/10)
  (h5 : tax_rate = 1/20)
  : ℕ := by
  sorry

#check cloth_sale_cost_price

end NUMINAMATH_CALUDE_cloth_sale_cost_price_l2490_249040


namespace NUMINAMATH_CALUDE_bowl_water_problem_l2490_249048

theorem bowl_water_problem (C : ℝ) (h1 : C > 0) :
  C / 2 + 4 = 0.7 * C → 0.7 * C = 14 := by
  sorry

end NUMINAMATH_CALUDE_bowl_water_problem_l2490_249048


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l2490_249086

theorem rectangle_width_decrease (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_L := 1.5 * L
  let new_W := W * (L / new_L)
  (W - new_W) / W = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l2490_249086


namespace NUMINAMATH_CALUDE_product_of_zeros_range_l2490_249031

noncomputable section

def f (x : ℝ) : ℝ := 
  if x ≥ 1 then Real.log x else 1 - x / 2

def F (m : ℝ) (x : ℝ) : ℝ := f (f x + 1) + m

theorem product_of_zeros_range (m : ℝ) 
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F m x₁ = 0 ∧ F m x₂ = 0) :
  ∃ p : ℝ, p < Real.sqrt (Real.exp 1) ∧ 
    ∀ q : ℝ, q < p → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F m x₁ = 0 ∧ F m x₂ = 0 ∧ x₁ * x₂ = q :=
sorry

end

end NUMINAMATH_CALUDE_product_of_zeros_range_l2490_249031


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2490_249081

def a : ℝ × ℝ := (2, 1)

def b (k : ℝ) : ℝ × ℝ := (1 - 2, k - 1)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_vectors_k_value :
  ∀ k : ℝ, perpendicular a (b k) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2490_249081


namespace NUMINAMATH_CALUDE_number_divided_by_004_l2490_249004

theorem number_divided_by_004 :
  ∃ x : ℝ, x / 0.04 = 500.90000000000003 ∧ x = 20.036 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_004_l2490_249004


namespace NUMINAMATH_CALUDE_cos_alpha_plus_20_eq_neg_alpha_l2490_249005

theorem cos_alpha_plus_20_eq_neg_alpha (α : ℝ) (h : Real.sin (α - 70 * Real.pi / 180) = α) :
  Real.cos (α + 20 * Real.pi / 180) = -α := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_20_eq_neg_alpha_l2490_249005


namespace NUMINAMATH_CALUDE_equal_celsius_fahrenheit_temp_l2490_249035

/-- Converts Celsius temperature to Fahrenheit -/
def celsius_to_fahrenheit (c : ℝ) : ℝ := 1.8 * c + 32

/-- Theorem stating that there exists a unique temperature where Celsius and Fahrenheit are equal -/
theorem equal_celsius_fahrenheit_temp :
  ∃! t : ℝ, t = celsius_to_fahrenheit t :=
by
  sorry

end NUMINAMATH_CALUDE_equal_celsius_fahrenheit_temp_l2490_249035


namespace NUMINAMATH_CALUDE_max_y_value_l2490_249002

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l2490_249002


namespace NUMINAMATH_CALUDE_michael_work_time_l2490_249088

/-- Given that:
    - Michael and Adam can complete a work together in 20 days
    - They work together for 18 days, then Michael stops
    - Adam completes the remaining work in 10 days
    Prove that Michael can complete the work separately in 25 days -/
theorem michael_work_time (total_work : ℝ) (michael_rate : ℝ) (adam_rate : ℝ)
  (h1 : michael_rate + adam_rate = total_work / 20)
  (h2 : 18 * (michael_rate + adam_rate) = 9 / 10 * total_work)
  (h3 : adam_rate = total_work / 100) :
  michael_rate = total_work / 25 := by
  sorry

end NUMINAMATH_CALUDE_michael_work_time_l2490_249088


namespace NUMINAMATH_CALUDE_hotel_room_charge_comparison_l2490_249006

theorem hotel_room_charge_comparison 
  (P R G : ℝ) 
  (h1 : P = R - 0.4 * R) 
  (h2 : P = G - 0.1 * G) : 
  (R - G) / G = 0.5 := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charge_comparison_l2490_249006


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2490_249073

theorem quadratic_coefficient (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 20 = (x + m)^2 + 8) → 
  b = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2490_249073


namespace NUMINAMATH_CALUDE_lot_width_calculation_l2490_249084

/-- Given a rectangular lot with length 40 m, height 2 m, and volume 1600 m³, 
    the width of the lot is 20 m. -/
theorem lot_width_calculation (length height volume width : ℝ) 
  (h_length : length = 40)
  (h_height : height = 2)
  (h_volume : volume = 1600)
  (h_relation : volume = length * width * height) : 
  width = 20 := by
  sorry

end NUMINAMATH_CALUDE_lot_width_calculation_l2490_249084


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a1_value_l2490_249077

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (a 1 + a 5 + a 7 + a 9 + a 13 = 100) ∧
  (a 6 - a 2 = 12)

/-- The theorem stating that a_1 = 2 for the given arithmetic sequence -/
theorem arithmetic_sequence_a1_value (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a1_value_l2490_249077


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l2490_249050

theorem isabellas_hair_growth (initial_length growth : ℕ) (h1 : initial_length = 18) (h2 : growth = 6) :
  initial_length + growth = 24 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l2490_249050


namespace NUMINAMATH_CALUDE_polynomial_equality_l2490_249025

theorem polynomial_equality (x y : ℝ) (h : x + y = -1) :
  x^4 + 5*x^3*y + x^2*y + 8*x^2*y^2 + x*y^2 + 5*x*y^3 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2490_249025


namespace NUMINAMATH_CALUDE_complex_below_real_axis_l2490_249061

theorem complex_below_real_axis (t : ℝ) : 
  let z : ℂ := (2 * t^2 + 5 * t - 3) + (t^2 + 2 * t + 2) * I
  Complex.im z < 0 := by
sorry

end NUMINAMATH_CALUDE_complex_below_real_axis_l2490_249061


namespace NUMINAMATH_CALUDE_house_price_calculation_l2490_249070

theorem house_price_calculation (P : ℝ) 
  (h1 : P > 0)
  (h2 : 0.56 * P = 56000) : P = 100000 :=
by
  sorry

end NUMINAMATH_CALUDE_house_price_calculation_l2490_249070


namespace NUMINAMATH_CALUDE_optimal_price_theorem_l2490_249096

-- Define the problem parameters
def initial_price : ℝ := 60
def initial_sales : ℝ := 300
def cost_price : ℝ := 40
def target_profit : ℝ := 6080
def price_sales_ratio : ℝ := 20

-- Define the profit function
def profit (price : ℝ) : ℝ :=
  (price - cost_price) * (initial_sales + price_sales_ratio * (initial_price - price))

-- State the theorem
theorem optimal_price_theorem :
  ∃ (optimal_price : ℝ),
    profit optimal_price = target_profit ∧
    optimal_price < initial_price ∧
    ∀ (p : ℝ), p < optimal_price → profit p < target_profit :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_optimal_price_theorem_l2490_249096


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l2490_249029

theorem imaginary_unit_power (i : ℂ) : i ^ 2 = -1 → i ^ 2023 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l2490_249029


namespace NUMINAMATH_CALUDE_negation_of_conjunction_l2490_249063

theorem negation_of_conjunction (p q : Prop) : ¬(p ∧ q) ↔ (¬p ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_negation_of_conjunction_l2490_249063


namespace NUMINAMATH_CALUDE_work_completion_time_l2490_249097

/-- Given workers a, b, and c, and their work rates, prove that b alone completes the work in 48 days -/
theorem work_completion_time (a b c : ℝ) 
  (h1 : a + b = 1 / 16)  -- a and b together finish in 16 days
  (h2 : a = 1 / 24)      -- a alone finishes in 24 days
  (h3 : c = 1 / 48)      -- c alone finishes in 48 days
  : b = 1 / 48 :=        -- b alone finishes in 48 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2490_249097


namespace NUMINAMATH_CALUDE_total_tires_is_101_l2490_249099

/-- The number of tires on a car -/
def car_tires : ℕ := 4

/-- The number of tires on a bicycle -/
def bicycle_tires : ℕ := 2

/-- The number of tires on a pickup truck -/
def pickup_truck_tires : ℕ := 4

/-- The number of tires on a tricycle -/
def tricycle_tires : ℕ := 3

/-- The number of cars Juan saw -/
def cars_seen : ℕ := 15

/-- The number of bicycles Juan saw -/
def bicycles_seen : ℕ := 3

/-- The number of pickup trucks Juan saw -/
def pickup_trucks_seen : ℕ := 8

/-- The number of tricycles Juan saw -/
def tricycles_seen : ℕ := 1

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 
  cars_seen * car_tires + 
  bicycles_seen * bicycle_tires + 
  pickup_trucks_seen * pickup_truck_tires + 
  tricycles_seen * tricycle_tires

theorem total_tires_is_101 : total_tires = 101 := by
  sorry

end NUMINAMATH_CALUDE_total_tires_is_101_l2490_249099


namespace NUMINAMATH_CALUDE_custom_mul_solution_l2490_249059

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 2 * a - b^2

/-- Theorem stating that if a * 4 = 9 under the custom multiplication, then a = 12.5 -/
theorem custom_mul_solution :
  ∃ a : ℝ, custom_mul a 4 = 9 ∧ a = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_solution_l2490_249059


namespace NUMINAMATH_CALUDE_representatives_count_l2490_249062

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of boys -/
def numBoys : ℕ := 4

/-- The number of girls -/
def numGirls : ℕ := 4

/-- The total number of representatives to be selected -/
def totalReps : ℕ := 3

/-- The minimum number of girls to be selected -/
def minGirls : ℕ := 2

theorem representatives_count :
  (choose numGirls 2 * choose numBoys 1) + (choose numGirls 3) = 28 := by sorry

end NUMINAMATH_CALUDE_representatives_count_l2490_249062


namespace NUMINAMATH_CALUDE_multiples_equality_l2490_249042

/-- The average of the first 7 positive multiples of 5 -/
def a : ℚ := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7

/-- The median of the first 3 positive multiples of n -/
def b (n : ℕ+) : ℚ := 2 * n

/-- Theorem stating that if a^2 - b^2 = 0, then n = 10 -/
theorem multiples_equality (n : ℕ+) : a^2 - (b n)^2 = 0 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_multiples_equality_l2490_249042


namespace NUMINAMATH_CALUDE_teacher_score_calculation_l2490_249001

def teacher_total_score (written_score interview_score : ℝ) (written_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

theorem teacher_score_calculation :
  let written_score : ℝ := 80
  let interview_score : ℝ := 60
  let written_weight : ℝ := 0.6
  let interview_weight : ℝ := 0.4
  teacher_total_score written_score interview_score written_weight interview_weight = 72 := by
  sorry

end NUMINAMATH_CALUDE_teacher_score_calculation_l2490_249001


namespace NUMINAMATH_CALUDE_tangent_equality_mod_180_l2490_249069

theorem tangent_equality_mod_180 (m : ℤ) : 
  -180 < m ∧ m < 180 ∧ Real.tan (m * π / 180) = Real.tan (2530 * π / 180) → m = 10 := by
  sorry

end NUMINAMATH_CALUDE_tangent_equality_mod_180_l2490_249069


namespace NUMINAMATH_CALUDE_trailing_zeroes_of_six_factorial_l2490_249038

-- Define the function z(n) that counts trailing zeroes in n!
def z (n : ℕ) : ℕ := 
  (n / 5) + (n / 25) + (n / 125)

-- State the theorem
theorem trailing_zeroes_of_six_factorial : z (z 6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeroes_of_six_factorial_l2490_249038


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2490_249008

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 + 4 * x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 + 8 * x

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (14 * x - y - 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2490_249008


namespace NUMINAMATH_CALUDE_expense_increase_percentage_is_ten_percent_l2490_249074

/-- Calculates the percentage increase in monthly expenses given the initial salary,
    savings rate, and new savings amount. -/
def calculate_expense_increase_percentage (salary : ℚ) (savings_rate : ℚ) (new_savings : ℚ) : ℚ :=
  let original_savings := salary * savings_rate
  let original_expenses := salary - original_savings
  let additional_expense := original_savings - new_savings
  (additional_expense / original_expenses) * 100

/-- Theorem stating that for the given conditions, the expense increase percentage is 10% -/
theorem expense_increase_percentage_is_ten_percent :
  calculate_expense_increase_percentage 20000 (1/10) 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_is_ten_percent_l2490_249074


namespace NUMINAMATH_CALUDE_rectangle_area_from_perimeter_and_diagonal_l2490_249091

/-- The area of a rectangle given its perimeter and diagonal -/
theorem rectangle_area_from_perimeter_and_diagonal (p d : ℝ) (hp : p > 0) (hd : d > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 * (x + y) = p ∧ x^2 + y^2 = d^2 ∧
  x * y = (p^2 - 4 * d^2) / 8 := by
  sorry

#check rectangle_area_from_perimeter_and_diagonal

end NUMINAMATH_CALUDE_rectangle_area_from_perimeter_and_diagonal_l2490_249091


namespace NUMINAMATH_CALUDE_simplify_sqrt_180_l2490_249019

theorem simplify_sqrt_180 : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_180_l2490_249019


namespace NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l2490_249075

/-- A board is a 22x22 grid of natural numbers -/
def Board : Type := Fin 22 → Fin 22 → ℕ

/-- A cell is a position on the board -/
def Cell : Type := Fin 22 × Fin 22

/-- Two cells are adjacent if they share a side or vertex -/
def adjacent (c1 c2 : Cell) : Prop :=
  let (x1, y1) := c1
  let (x2, y2) := c2
  (x1 = x2 ∧ y1.val + 1 = y2.val) ∨
  (x1 = x2 ∧ y2.val + 1 = y1.val) ∨
  (y1 = y2 ∧ x1.val + 1 = x2.val) ∨
  (y1 = y2 ∧ x2.val + 1 = x1.val) ∨
  (x1.val + 1 = x2.val ∧ y1.val + 1 = y2.val) ∨
  (x2.val + 1 = x1.val ∧ y1.val + 1 = y2.val) ∨
  (x1.val + 1 = x2.val ∧ y2.val + 1 = y1.val) ∨
  (x2.val + 1 = x1.val ∧ y2.val + 1 = y1.val)

/-- A valid board contains numbers from 1 to 22² -/
def valid_board (b : Board) : Prop :=
  ∀ x y, 1 ≤ b x y ∧ b x y ≤ 22^2

theorem adjacent_sum_divisible_by_four (b : Board) (h : valid_board b) :
  ∃ c1 c2 : Cell, adjacent c1 c2 ∧ (b c1.1 c1.2 + b c2.1 c2.2) % 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l2490_249075


namespace NUMINAMATH_CALUDE_three_valid_floor_dimensions_l2490_249092

/-- 
Represents the number of valid floor dimensions (m, n) satisfying:
1. n > m
2. (m-6)(n-6) = 12
3. m ≥ 7 and n ≥ 7
where m and n are positive integers, and the unpainted border is 2 feet wide on each side.
-/
def validFloorDimensions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let m := p.1
    let n := p.2
    n > m ∧ (m - 6) * (n - 6) = 12 ∧ m ≥ 7 ∧ n ≥ 7
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The main theorem stating that there are exactly 3 valid floor dimensions. -/
theorem three_valid_floor_dimensions : validFloorDimensions = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_valid_floor_dimensions_l2490_249092


namespace NUMINAMATH_CALUDE_hex_to_binary_max_bits_l2490_249064

theorem hex_to_binary_max_bits :
  ∀ (A B C D : Nat),
  A < 16 → B < 16 → C < 16 → D < 16 →
  ∃ (n : Nat),
  n ≤ 16 ∧
  A * 16^3 + B * 16^2 + C * 16^1 + D < 2^n :=
by sorry

end NUMINAMATH_CALUDE_hex_to_binary_max_bits_l2490_249064


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l2490_249044

theorem units_digit_of_expression : 2^2023 * 5^2024 * 11^2025 % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l2490_249044


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l2490_249076

theorem least_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * |x| + 4 < 19) → x ≥ -4 ∧
  ∃ y : ℤ, y = -4 ∧ (3 * |y| + 4 < 19) :=
by sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l2490_249076


namespace NUMINAMATH_CALUDE_factorial_prime_factors_l2490_249045

theorem factorial_prime_factors (x i k m p : ℕ) : 
  x = (Finset.range 8).prod (λ n => n + 1) →
  x = 2^i * 3^k * 5^m * 7^p →
  i > 0 ∧ k > 0 ∧ m > 0 ∧ p > 0 →
  i + k + m + p = 11 := by
sorry

end NUMINAMATH_CALUDE_factorial_prime_factors_l2490_249045


namespace NUMINAMATH_CALUDE_triangles_from_parallel_lines_l2490_249000

/-- The number of points on line a -/
def points_on_a : ℕ := 5

/-- The number of points on line b -/
def points_on_b : ℕ := 6

/-- The total number of triangles that can be formed -/
def total_triangles : ℕ := 135

/-- Theorem stating that the total number of triangles formed by points on two parallel lines is correct -/
theorem triangles_from_parallel_lines : 
  (points_on_a.choose 1 * points_on_b.choose 2) + (points_on_a.choose 2 * points_on_b.choose 1) = total_triangles :=
by sorry

end NUMINAMATH_CALUDE_triangles_from_parallel_lines_l2490_249000


namespace NUMINAMATH_CALUDE_unique_integer_l2490_249071

theorem unique_integer (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) : 
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_l2490_249071


namespace NUMINAMATH_CALUDE_unique_solution_l2490_249007

theorem unique_solution (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 48) :
  a = 13 ∧ b = 11 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2490_249007


namespace NUMINAMATH_CALUDE_sum_of_integers_l2490_249047

theorem sum_of_integers (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x = y + 3) (h4 : x^3 - y^3 = 63) :
  x + y = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2490_249047


namespace NUMINAMATH_CALUDE_hypotenuse_plus_diameter_eq_sum_of_legs_l2490_249043

/-- Represents a right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  a : ℝ     -- Length of one leg
  b : ℝ     -- Length of the other leg
  c : ℝ     -- Length of the hypotenuse
  ρ : ℝ     -- Radius of the inscribed circle
  h_right : a^2 + b^2 = c^2  -- Pythagorean theorem
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ ρ > 0  -- Positive lengths

/-- 
The sum of the hypotenuse and the diameter of the inscribed circle 
is equal to the sum of the two legs in a right-angled triangle
-/
theorem hypotenuse_plus_diameter_eq_sum_of_legs 
  (t : RightTriangleWithInscribedCircle) : t.c + 2 * t.ρ = t.a + t.b := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_plus_diameter_eq_sum_of_legs_l2490_249043


namespace NUMINAMATH_CALUDE_marie_messages_per_day_l2490_249034

/-- Represents the problem of calculating the number of messages read per day -/
def messages_read_per_day (initial_unread : ℕ) (new_messages_per_day : ℕ) (days_to_clear : ℕ) : ℕ :=
  (initial_unread + new_messages_per_day * days_to_clear) / days_to_clear

/-- Theorem stating that Marie reads 20 messages per day -/
theorem marie_messages_per_day :
  messages_read_per_day 98 6 7 = 20 := by
  sorry

#eval messages_read_per_day 98 6 7

end NUMINAMATH_CALUDE_marie_messages_per_day_l2490_249034


namespace NUMINAMATH_CALUDE_jills_study_hours_l2490_249011

/-- Represents Jill's study schedule over three days -/
structure StudySchedule where
  day1 : ℝ  -- Hours studied on day 1
  day2 : ℝ  -- Hours studied on day 2
  day3 : ℝ  -- Hours studied on day 3

/-- The theorem representing Jill's study problem -/
theorem jills_study_hours (schedule : StudySchedule) :
  schedule.day2 = 2 * schedule.day1 ∧
  schedule.day3 = 2 * schedule.day1 - 1 ∧
  schedule.day1 + schedule.day2 + schedule.day3 = 9 →
  schedule.day1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_jills_study_hours_l2490_249011


namespace NUMINAMATH_CALUDE_no_valid_A_exists_l2490_249023

theorem no_valid_A_exists : ¬∃ (A : ℕ), 1 ≤ A ∧ A ≤ 9 ∧
  ∃ (x : ℕ), x^2 - (2*A)*x + (A+1)*0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_A_exists_l2490_249023


namespace NUMINAMATH_CALUDE_cookie_distribution_ways_l2490_249016

/-- The number of ways to distribute cookies among students -/
def distribute_cookies (total_cookies : ℕ) (num_students : ℕ) (min_cookies : ℕ) : ℕ :=
  Nat.choose (total_cookies - num_students * min_cookies + num_students - 1) (num_students - 1)

/-- Theorem: The number of ways to distribute 30 cookies among 5 students, 
    with each student receiving at least 3 cookies, is 3876 -/
theorem cookie_distribution_ways : distribute_cookies 30 5 3 = 3876 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_ways_l2490_249016


namespace NUMINAMATH_CALUDE_perpendicular_AC_AD_l2490_249017

/-- The curve E in the xy-plane -/
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + 3 * p.2^2 / 4 = 1 ∧ p.1 ≠ 2 ∧ p.1 ≠ -2}

/-- Point A -/
def A : ℝ × ℝ := (-2, 0)

/-- Point Q -/
def Q : ℝ × ℝ := (-1, 0)

/-- A line with non-zero slope passing through Q -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * (p.1 + 1) ∧ m ≠ 0}

/-- The intersection points of the line and curve E -/
def intersection (m : ℝ) : Set (ℝ × ℝ) :=
  E ∩ line_through_Q m

theorem perpendicular_AC_AD (m : ℝ) 
  (hm : m ≠ 0) 
  (h_intersect : ∃ C D, C ∈ intersection m ∧ D ∈ intersection m ∧ C ≠ D) :
  ∀ C D, C ∈ intersection m → D ∈ intersection m → C ≠ D →
  (C.1 + 2) * (D.1 + 2) + C.2 * D.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_AC_AD_l2490_249017


namespace NUMINAMATH_CALUDE_percentage_calculation_l2490_249051

theorem percentage_calculation : 
  (0.47 * 1442 - 0.36 * 1412) + 63 = 232.42 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2490_249051


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l2490_249068

theorem smallest_n_for_candy_purchase : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → 24 * m = Nat.lcm (Nat.lcm 18 16) 20 → n ≤ m) ∧
  24 * n = Nat.lcm (Nat.lcm 18 16) 20 ∧ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l2490_249068


namespace NUMINAMATH_CALUDE_no_common_terms_except_first_l2490_249024

def X : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => X (n + 1) + 2 * X n

def Y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * Y (n + 1) + 3 * Y n

theorem no_common_terms_except_first : ∀ n m : ℕ, X n = Y m → n = 0 ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_terms_except_first_l2490_249024


namespace NUMINAMATH_CALUDE_direction_vector_of_line_l_l2490_249089

/-- The line l is defined by the equation (x-1)/3 = (y+1)/4 -/
def line_l (x y : ℝ) : Prop := (x - 1) / 3 = (y + 1) / 4

/-- A direction vector of a line is a vector parallel to the line -/
def is_direction_vector (v : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (t : ℝ) (x y : ℝ), l x y → l (x + t * v.1) (y + t * v.2)

/-- Prove that (3,4) is a direction vector of the line l -/
theorem direction_vector_of_line_l : is_direction_vector (3, 4) line_l := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_of_line_l_l2490_249089


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l2490_249094

theorem min_value_abs_sum (x : ℝ) : 
  |x + 1| + |x - 2| + |x - 3| ≥ 4 ∧ ∃ y : ℝ, |y + 1| + |y - 2| + |y - 3| = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l2490_249094


namespace NUMINAMATH_CALUDE_seafood_price_seafood_price_proof_l2490_249072

/-- The regular price for two pounds of seafood given a 75% discount and $4 discounted price for one pound -/
theorem seafood_price : ℝ → ℝ → ℝ → Prop :=
  fun discount_percent discounted_price_per_pound regular_price_two_pounds =>
    discount_percent = 75 ∧
    discounted_price_per_pound = 4 →
    regular_price_two_pounds = 32

/-- Proof of the seafood price theorem -/
theorem seafood_price_proof :
  seafood_price 75 4 32 := by
  sorry

end NUMINAMATH_CALUDE_seafood_price_seafood_price_proof_l2490_249072


namespace NUMINAMATH_CALUDE_school_distance_l2490_249093

/-- The distance between a girl's house and school, given her travel speeds and total round trip time. -/
theorem school_distance (speed_to_school speed_from_school : ℝ) (total_time : ℝ) : 
  speed_to_school = 6 →
  speed_from_school = 4 →
  total_time = 10 →
  (1 / speed_to_school + 1 / speed_from_school) * (speed_to_school * speed_from_school / (speed_to_school + speed_from_school)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_school_distance_l2490_249093


namespace NUMINAMATH_CALUDE_always_positive_product_l2490_249082

theorem always_positive_product (a b c : ℝ) (h : a > b ∧ b > c) : (a - b) * |c - b| > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_product_l2490_249082


namespace NUMINAMATH_CALUDE_range_of_a_l2490_249020

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- Theorem statement
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (1 - a) > f (2 * a - 1)) :
  2 / 3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2490_249020


namespace NUMINAMATH_CALUDE_circles_intersection_m_range_l2490_249010

/-- Circle C₁ with equation x² + y² - 2mx + m² - 4 = 0 -/
def C₁ (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*m*p.1 + m^2 - 4 = 0}

/-- Circle C₂ with equation x² + y² + 2x - 4my + 4m² - 8 = 0 -/
def C₂ (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 4*m*p.2 + 4*m^2 - 8 = 0}

/-- The theorem stating that if C₁ and C₂ intersect, then m is in the specified range -/
theorem circles_intersection_m_range (m : ℝ) :
  (C₁ m ∩ C₂ m).Nonempty →
  m ∈ Set.Ioo (-12/5) (-2/5) ∪ Set.Ioo (3/5) 2 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersection_m_range_l2490_249010


namespace NUMINAMATH_CALUDE_max_cut_trees_100x100_l2490_249066

/-- Represents a square grid of trees -/
structure TreeGrid where
  size : ℕ
  trees : Fin size → Fin size → Bool

/-- Checks if a tree can be cut without making any other cut tree visible -/
def canCutTree (grid : TreeGrid) (x y : Fin grid.size) : Bool := sorry

/-- Counts the maximum number of trees that can be cut in a grid -/
def maxCutTrees (grid : TreeGrid) : ℕ := sorry

/-- Theorem: In a 100x100 grid, the maximum number of trees that can be cut
    while ensuring no stump is visible from any other stump is 2500 -/
theorem max_cut_trees_100x100 :
  ∀ (grid : TreeGrid), grid.size = 100 → maxCutTrees grid = 2500 := by sorry

end NUMINAMATH_CALUDE_max_cut_trees_100x100_l2490_249066


namespace NUMINAMATH_CALUDE_problem_solution_l2490_249013

theorem problem_solution (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 6 = 104 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2490_249013


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2490_249078

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) * Real.sqrt ((y + z) * (z + x)) +
  (y + z) * Real.sqrt ((z + x) * (x + y)) +
  (z + x) * Real.sqrt ((x + y) * (y + z)) ≥
  4 * (x * y + y * z + z * x) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2490_249078


namespace NUMINAMATH_CALUDE_sphere_radius_increase_l2490_249022

theorem sphere_radius_increase (r : ℝ) (h : r > 0) : 
  let A := 4 * Real.pi * r^2
  let r' := Real.sqrt (2.25 * r^2)
  let A' := 4 * Real.pi * r'^2
  A' = 2.25 * A → r' = 1.5 * r :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_increase_l2490_249022


namespace NUMINAMATH_CALUDE_camp_kids_count_l2490_249041

theorem camp_kids_count (total : ℕ) 
  (h1 : total / 2 = total / 2) -- Half of the kids are going to soccer camp
  (h2 : (total / 2) / 4 = (total / 2) / 4) -- 1/4 of soccer camp kids go in the morning
  (h3 : ((total / 2) * 3) / 4 = 750) -- 750 kids go to soccer camp in the afternoon
  : total = 2000 := by
  sorry

end NUMINAMATH_CALUDE_camp_kids_count_l2490_249041


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l2490_249030

theorem coefficient_x3y5_in_expansion_of_x_plus_y_to_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * 1^k * 1^(8-k)) = 256 ∧
  Nat.choose 8 3 = 56 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l2490_249030


namespace NUMINAMATH_CALUDE_equation_solution_l2490_249087

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (8 * x)^16 = (32 * x)^8 → x = 1/2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2490_249087


namespace NUMINAMATH_CALUDE_cubic_derivative_value_l2490_249098

theorem cubic_derivative_value (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = x^3) →
  (deriv f) x₀ = 3 →
  x₀ = 1 ∨ x₀ = -1 := by
sorry

end NUMINAMATH_CALUDE_cubic_derivative_value_l2490_249098


namespace NUMINAMATH_CALUDE_largest_power_of_two_divisor_l2490_249065

theorem largest_power_of_two_divisor (n : ℕ) :
  (∃ (k : ℕ), 2^k ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋ ∧
    ∀ (m : ℕ), m > k → ¬(2^m ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋)) →
  (∃! (k : ℕ), k = n ∧ 2^k ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋ ∧
    ∀ (m : ℕ), m > k → ¬(2^m ∣ ⌊(3 + Real.sqrt 11)^(2*n - 1)⌋)) :=
by sorry

#check largest_power_of_two_divisor

end NUMINAMATH_CALUDE_largest_power_of_two_divisor_l2490_249065


namespace NUMINAMATH_CALUDE_simplify_w_squared_series_l2490_249039

theorem simplify_w_squared_series (w : ℝ) : 
  3 * w^2 + 6 * w^2 + 9 * w^2 + 12 * w^2 + 15 * w^2 + 24 = 45 * w^2 + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_w_squared_series_l2490_249039


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_equal_l2490_249095

theorem quadratic_roots_sum_squares_equal (a : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    x₁^2 + 2*x₁ + a = 0 ∧
    x₂^2 + 2*x₂ + a = 0 ∧
    y₁^2 + a*y₁ + 2 = 0 ∧
    y₂^2 + a*y₂ + 2 = 0 ∧
    x₁^2 + x₂^2 = y₁^2 + y₂^2) →
  a = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_equal_l2490_249095


namespace NUMINAMATH_CALUDE_sin_75_cos_75_double_l2490_249033

theorem sin_75_cos_75_double : 2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_75_double_l2490_249033


namespace NUMINAMATH_CALUDE_ellipse_properties_l2490_249003

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/9 + y^2 = 1

-- Define the line that intersects the ellipse
def intersecting_line (x y : ℝ) : Prop := y = x + 2

-- Define the condition for a point to be on the circle with AB as diameter
def on_circle_AB (x y : ℝ) : Prop := 
  ∃ (x1 y1 x2 y2 : ℝ), 
    ellipse_C x1 y1 ∧ 
    ellipse_C x2 y2 ∧ 
    intersecting_line x1 y1 ∧ 
    intersecting_line x2 y2 ∧ 
    x * (x1 + x2) + y * (y1 + y2) = (x1^2 + y1^2 + x2^2 + y2^2) / 2

theorem ellipse_properties :
  (∀ x y, ellipse_C x y ↔ x^2/9 + y^2 = 1) ∧ 
  ¬(on_circle_AB 0 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2490_249003
