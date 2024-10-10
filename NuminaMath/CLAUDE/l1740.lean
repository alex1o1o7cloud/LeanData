import Mathlib

namespace exists_iceberg_with_properties_l1740_174007

/-- Represents a convex polyhedron floating in water --/
structure FloatingPolyhedron where
  totalVolume : ℝ
  submergedVolume : ℝ
  totalSurfaceArea : ℝ
  submergedSurfaceArea : ℝ
  volume_nonneg : 0 < totalVolume
  submerged_volume_le_total : submergedVolume ≤ totalVolume
  surface_area_nonneg : 0 < totalSurfaceArea
  submerged_surface_le_total : submergedSurfaceArea ≤ totalSurfaceArea

/-- Theorem stating the existence of a floating polyhedron with the required properties --/
theorem exists_iceberg_with_properties :
  ∃ (iceberg : FloatingPolyhedron),
    iceberg.submergedVolume ≥ 0.9 * iceberg.totalVolume ∧
    iceberg.submergedSurfaceArea ≤ 0.5 * iceberg.totalSurfaceArea :=
sorry

end exists_iceberg_with_properties_l1740_174007


namespace termite_ridden_collapsing_homes_l1740_174024

theorem termite_ridden_collapsing_homes 
  (total_homes : ℕ) 
  (termite_ridden : ℚ) 
  (termite_not_collapsing : ℚ) 
  (h1 : termite_ridden = 1/3) 
  (h2 : termite_not_collapsing = 1/7) : 
  (termite_ridden - termite_not_collapsing) / termite_ridden = 4/21 := by
sorry

end termite_ridden_collapsing_homes_l1740_174024


namespace inequality_proof_l1740_174062

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 4) 
  (h2 : c^2 + d^2 = 16) : 
  a*c + b*d ≤ 8 := by
sorry

end inequality_proof_l1740_174062


namespace alpha_is_two_thirds_l1740_174084

theorem alpha_is_two_thirds (α : ℚ) 
  (h1 : 0 < α) 
  (h2 : α < 1) 
  (h3 : Real.cos (3 * Real.pi * α) + 2 * Real.cos (2 * Real.pi * α) = 0) : 
  α = 2/3 := by
sorry

end alpha_is_two_thirds_l1740_174084


namespace deepak_age_l1740_174048

/-- Given the ratio of Arun's age to Deepak's age and Arun's future age, 
    prove Deepak's current age -/
theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end deepak_age_l1740_174048


namespace simultaneous_pipe_filling_time_l1740_174085

/-- Given two pipes that can fill a tank in 10 and 20 hours respectively,
    prove that when both are opened simultaneously, the tank fills in 20/3 hours. -/
theorem simultaneous_pipe_filling_time :
  ∀ (tank_capacity : ℝ) (pipe_a_rate pipe_b_rate : ℝ),
    pipe_a_rate = tank_capacity / 10 →
    pipe_b_rate = tank_capacity / 20 →
    tank_capacity / (pipe_a_rate + pipe_b_rate) = 20 / 3 := by
  sorry

end simultaneous_pipe_filling_time_l1740_174085


namespace cost_of_one_plank_l1740_174050

/-- The cost of one plank given the conditions for building birdhouses -/
theorem cost_of_one_plank : 
  ∀ (plank_cost : ℝ),
  (4 * (7 * plank_cost + 20 * 0.05) = 88) →
  plank_cost = 3 :=
by
  sorry

end cost_of_one_plank_l1740_174050


namespace cube_of_negative_four_equals_negative_cube_of_four_l1740_174088

theorem cube_of_negative_four_equals_negative_cube_of_four : (-4)^3 = -4^3 := by
  sorry

end cube_of_negative_four_equals_negative_cube_of_four_l1740_174088


namespace intersection_length_theorem_l1740_174086

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1
def F₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

-- Define the locus C
def C (x y : ℝ) : Prop := x^2 - y^2/3 = 1 ∧ x < -1

-- Define a line through F₁
def line_through_F₁ (m : ℝ) (x y : ℝ) : Prop := x = m * y - 2

-- Theorem statement
theorem intersection_length_theorem 
  (A B P Q : ℝ × ℝ) 
  (m : ℝ) 
  (h₁ : C A.1 A.2) 
  (h₂ : C B.1 B.2) 
  (h₃ : F₂ P.1 P.2) 
  (h₄ : F₂ Q.1 Q.2) 
  (h₅ : line_through_F₁ m A.1 A.2) 
  (h₆ : line_through_F₁ m B.1 B.2) 
  (h₇ : line_through_F₁ m P.1 P.2) 
  (h₈ : line_through_F₁ m Q.1 Q.2) 
  (h₉ : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 :=
sorry

end intersection_length_theorem_l1740_174086


namespace quadratic_inequality_solution_set_l1740_174053

theorem quadratic_inequality_solution_set (x : ℝ) :
  x^2 - 2*x - 3 > 0 ↔ x < -1 ∨ x > 3 := by
  sorry

end quadratic_inequality_solution_set_l1740_174053


namespace time_addition_theorem_l1740_174038

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time -/
def addTime (initial : Time) (dHours dMinutes dSeconds : Nat) : Time :=
  sorry

/-- Converts 24-hour time to 12-hour time -/
def to12Hour (time : Time) : Time :=
  sorry

/-- Calculates the sum of hours, minutes, and seconds -/
def sumTimeComponents (time : Time) : Nat :=
  sorry

theorem time_addition_theorem :
  let initial_time := Time.mk 15 15 30  -- 3:15:30 PM
  let duration_hours := 174
  let duration_minutes := 58
  let duration_seconds := 16
  let final_time := to12Hour (addTime initial_time duration_hours duration_minutes duration_seconds)
  final_time = Time.mk 10 13 46 ∧ sumTimeComponents final_time = 69 := by
  sorry

end time_addition_theorem_l1740_174038


namespace new_average_after_drop_l1740_174030

/-- Theorem: New average after student drops class -/
theorem new_average_after_drop (n : ℕ) (old_avg : ℚ) (drop_score : ℚ) :
  n = 16 →
  old_avg = 62.5 →
  drop_score = 70 →
  (n : ℚ) * old_avg - drop_score = ((n - 1) : ℚ) * 62 :=
by sorry

end new_average_after_drop_l1740_174030


namespace cheryl_material_usage_l1740_174054

theorem cheryl_material_usage (bought_type1 bought_type2 leftover : ℚ) :
  bought_type1 = 5/9 →
  bought_type2 = 1/3 →
  leftover = 8/24 →
  bought_type1 + bought_type2 - leftover = 5/9 := by
sorry

end cheryl_material_usage_l1740_174054


namespace masha_numbers_proof_l1740_174051

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a % 2 = 0 ∨ b % 2 = 0)

def is_unique_pair (a b : ℕ) : Prop :=
  ∀ x y : ℕ, x + y = a + b → is_valid_pair x y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)

theorem masha_numbers_proof :
  ∃! (a b : ℕ), is_valid_pair a b ∧ is_unique_pair a b ∧ a + b = 28 :=
sorry

end masha_numbers_proof_l1740_174051


namespace probability_of_specific_arrangement_l1740_174082

def total_tiles : ℕ := 6
def x_tiles : ℕ := 4
def o_tiles : ℕ := 2

theorem probability_of_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 1 / 15 := by
  sorry

end probability_of_specific_arrangement_l1740_174082


namespace parabola_equation_l1740_174045

/-- A parabola with vertex at the origin, coordinate axes as axes of symmetry, 
    and passing through point (-4, -2) has a standard equation of either 
    x^2 = -8y or y^2 = -x -/
theorem parabola_equation (f : ℝ → ℝ) : 
  (∀ x y, f x = y ↔ (x^2 = -8*y ∨ y^2 = -x)) ↔ 
  (f 0 = 0 ∧ 
   (∀ x, f x = f (-x)) ∧ 
   (∀ y, f (f y) = y) ∧
   f (-4) = -2) :=
sorry

end parabola_equation_l1740_174045


namespace largest_number_in_ratio_l1740_174037

theorem largest_number_in_ratio (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (b = (5 * a) / 3) → 
  (c = (7 * a) / 3) → 
  (c - a = 40) → 
  c = 70 := by
sorry

end largest_number_in_ratio_l1740_174037


namespace payment_proof_l1740_174063

/-- Given a total payment of $80 using $20 and $10 bills, where the number of $20 bills
    is one more than the number of $10 bills, prove that the number of $10 bills used is 2. -/
theorem payment_proof (total : ℕ) (ten_bills : ℕ) (twenty_bills : ℕ) : 
  total = 80 →
  twenty_bills = ten_bills + 1 →
  10 * ten_bills + 20 * twenty_bills = total →
  ten_bills = 2 := by
  sorry

end payment_proof_l1740_174063


namespace sum_of_repeating_decimals_l1740_174092

-- Define the repeating decimal 0.3333...
def repeating_3 : ℚ := 1 / 3

-- Define the repeating decimal 0.2121...
def repeating_21 : ℚ := 7 / 33

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_3 + repeating_21 = 6 / 11 := by
  sorry

end sum_of_repeating_decimals_l1740_174092


namespace salesman_pears_morning_sales_salesman_pears_morning_sales_proof_l1740_174056

/-- Proof that a salesman sold 120 kilograms of pears in the morning -/
theorem salesman_pears_morning_sales : ℝ → Prop :=
  fun morning_sales : ℝ =>
    let afternoon_sales := 240
    let total_sales := 360
    (afternoon_sales = 2 * morning_sales) ∧
    (total_sales = morning_sales + afternoon_sales) →
    morning_sales = 120

-- The proof is omitted
theorem salesman_pears_morning_sales_proof : salesman_pears_morning_sales 120 := by
  sorry

end salesman_pears_morning_sales_salesman_pears_morning_sales_proof_l1740_174056


namespace total_distance_proof_l1740_174075

/-- The total distance across the country in kilometers -/
def total_distance : ℕ := 8205

/-- The distance Amelia drove on Monday in kilometers -/
def monday_distance : ℕ := 907

/-- The distance Amelia drove on Tuesday in kilometers -/
def tuesday_distance : ℕ := 582

/-- The remaining distance Amelia has to drive in kilometers -/
def remaining_distance : ℕ := 6716

/-- Theorem stating that the total distance is the sum of the distances driven on Monday, Tuesday, and the remaining distance -/
theorem total_distance_proof : 
  total_distance = monday_distance + tuesday_distance + remaining_distance := by
  sorry

end total_distance_proof_l1740_174075


namespace problem_statement_l1740_174094

theorem problem_statement (a b c : ℝ) (h1 : a - b = 3) (h2 : a - c = 1) :
  (c - b)^2 - 2*(c - b) + 2 = 2 := by
  sorry

end problem_statement_l1740_174094


namespace min_integral_abs_exp_minus_a_l1740_174033

theorem min_integral_abs_exp_minus_a :
  let f (a : ℝ) := ∫ x in (0 : ℝ)..1, |Real.exp (-x) - a|
  ∃ m : ℝ, (∀ a : ℝ, f a ≥ m) ∧ (∃ a : ℝ, f a = m) ∧ m = 1 - 2 * Real.exp (-1) := by
  sorry

end min_integral_abs_exp_minus_a_l1740_174033


namespace first_purchase_quantities_second_purchase_max_profit_new_selling_price_B_l1740_174064

-- Definitions based on the problem conditions
def purchase_price_A : ℝ := 30
def purchase_price_B : ℝ := 25
def selling_price_A : ℝ := 45
def selling_price_B : ℝ := 37
def total_keychains : ℕ := 30
def total_cost : ℝ := 850
def second_purchase_total : ℕ := 80
def second_purchase_max_cost : ℝ := 2200
def original_daily_sales_B : ℕ := 4
def price_reduction_effect : ℝ := 2

-- Part 1
theorem first_purchase_quantities (x y : ℕ) :
  purchase_price_A * x + purchase_price_B * y = total_cost ∧
  x + y = total_keychains →
  x = 20 ∧ y = 10 := by sorry

-- Part 2
theorem second_purchase_max_profit (m : ℕ) :
  m ≤ 40 →
  ∃ (w : ℝ), w = 3 * m + 960 ∧
  w ≤ 1080 ∧
  (m = 40 → w = 1080) := by sorry

-- Part 3
theorem new_selling_price_B (a : ℝ) :
  (a - purchase_price_B) * (78 - 2 * a) = 90 →
  a = 30 ∨ a = 34 := by sorry

end first_purchase_quantities_second_purchase_max_profit_new_selling_price_B_l1740_174064


namespace ring_diameter_theorem_l1740_174068

/-- The diameter of ring X -/
def diameter_X : ℝ := 16

/-- The fraction of ring X's surface not covered by ring Y -/
def uncovered_fraction : ℝ := 0.2098765432098765

/-- The diameter of ring Y -/
noncomputable def diameter_Y : ℝ := 14.222

/-- Theorem stating that given the diameter of ring X and the uncovered fraction,
    the diameter of ring Y is approximately 14.222 inches -/
theorem ring_diameter_theorem (ε : ℝ) (h : ε > 0) :
  ∃ (d : ℝ), abs (d - diameter_Y) < ε ∧ 
  d^2 / 4 = diameter_X^2 / 4 * (1 - uncovered_fraction) :=
sorry

end ring_diameter_theorem_l1740_174068


namespace log_equality_implies_y_value_l1740_174040

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the theorem
theorem log_equality_implies_y_value 
  (a b c x : ℝ) 
  (p q r y : ℝ) 
  (h1 : log a / p = log b / q)
  (h2 : log b / q = log c / r)
  (h3 : log c / r = log x)
  (h4 : x ≠ 1)
  (h5 : b^3 / (a^2 * c) = x^y) :
  y = 3*q - 2*p - r := by
  sorry

#check log_equality_implies_y_value

end log_equality_implies_y_value_l1740_174040


namespace volume_equality_l1740_174067

/-- The volume of the solid obtained by rotating the region bounded by x² = 4y, x² = -4y, x = 4, and x = -4 about the y-axis -/
def V₁ : ℝ := sorry

/-- The volume of the solid obtained by rotating the region defined by x² + y² ≤ 16, x² + (y-2)² ≥ 4, and x² + (y+2)² ≥ 4 about the y-axis -/
def V₂ : ℝ := sorry

/-- Theorem stating that V₁ equals V₂ -/
theorem volume_equality : V₁ = V₂ := by sorry

end volume_equality_l1740_174067


namespace marys_income_percentage_l1740_174090

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * (1 - 0.5))
  (h2 : mary = tim * (1 + 0.6)) :
  mary = juan * 0.8 := by sorry

end marys_income_percentage_l1740_174090


namespace ellipse_major_axis_length_l1740_174076

/-- The length of the major axis of the ellipse 16x^2 + 9y^2 = 144 is 8 -/
theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | 16 * x^2 + 9 * y^2 = 144}
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ellipse = {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1} ∧
    2 * a = 8 :=
by sorry

end ellipse_major_axis_length_l1740_174076


namespace missing_sale_is_8562_l1740_174026

/-- Calculates the missing sale amount given sales for 5 months and the average -/
def calculate_missing_sale (sale1 sale2 sale3 sale4 sale6 average : ℚ) : ℚ :=
  6 * average - (sale1 + sale2 + sale3 + sale4 + sale6)

theorem missing_sale_is_8562 :
  let sale1 : ℚ := 8435
  let sale2 : ℚ := 8927
  let sale3 : ℚ := 8855
  let sale4 : ℚ := 9230
  let sale6 : ℚ := 6991
  let average : ℚ := 8500
  calculate_missing_sale sale1 sale2 sale3 sale4 sale6 average = 8562 := by
  sorry

#eval calculate_missing_sale 8435 8927 8855 9230 6991 8500

end missing_sale_is_8562_l1740_174026


namespace min_value_and_range_l1740_174011

-- Define the function f(x, y, a) = 2xy - x - y - a(x^2 + y^2)
def f (x y a : ℝ) : ℝ := 2 * x * y - x - y - a * (x^2 + y^2)

theorem min_value_and_range {x y a : ℝ} (hx : x > 0) (hy : y > 0) (hf : f x y a = 0) :
  -- Part 1: When a = 0, minimum value of 2x + 4y and corresponding x, y
  (a = 0 → 2 * x + 4 * y ≥ 3 + 2 * Real.sqrt 2 ∧
    (2 * x + 4 * y = 3 + 2 * Real.sqrt 2 ↔ x = (1 + Real.sqrt 2) / 2 ∧ y = (2 + Real.sqrt 2) / 4)) ∧
  -- Part 2: When a = 1/2, range of x + y
  (a = 1/2 → x + y ≥ 4) := by
  sorry

end min_value_and_range_l1740_174011


namespace cone_base_radius_l1740_174071

/-- Given a cone with slant height 12 cm and central angle of unfolded lateral surface 150°, 
    the radius of its base is 5 cm. -/
theorem cone_base_radius (slant_height : ℝ) (central_angle : ℝ) : 
  slant_height = 12 → central_angle = 150 → ∃ (base_radius : ℝ), base_radius = 5 := by
  sorry

end cone_base_radius_l1740_174071


namespace sum_of_arcs_equals_180_degrees_l1740_174060

-- Define a circle in a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define an arc on a circle
structure Arc :=
  (circle : Circle)
  (start_angle : ℝ)
  (end_angle : ℝ)

-- Define the arrangement of three circles
def triangle_arrangement (c1 c2 c3 : Circle) : Prop :=
  -- This is a placeholder for the specific arrangement condition
  True

-- Define the theorem
theorem sum_of_arcs_equals_180_degrees 
  (c1 c2 c3 : Circle) 
  (ab : Arc) 
  (cd : Arc) 
  (ef : Arc) 
  (h1 : c1.radius = c2.radius ∧ c2.radius = c3.radius)
  (h2 : triangle_arrangement c1 c2 c3)
  (h3 : ab.circle = c1 ∧ cd.circle = c2 ∧ ef.circle = c3) :
  ab.end_angle - ab.start_angle + 
  cd.end_angle - cd.start_angle + 
  ef.end_angle - ef.start_angle = π :=
sorry

end sum_of_arcs_equals_180_degrees_l1740_174060


namespace metal_bar_weight_l1740_174017

/-- The weight of Harry's custom creation at the gym -/
def total_weight : ℕ := 25

/-- The weight of each blue weight -/
def blue_weight : ℕ := 2

/-- The weight of each green weight -/
def green_weight : ℕ := 3

/-- The number of blue weights Harry put on the bar -/
def num_blue_weights : ℕ := 4

/-- The number of green weights Harry put on the bar -/
def num_green_weights : ℕ := 5

/-- The weight of the metal bar -/
def bar_weight : ℕ := total_weight - (num_blue_weights * blue_weight + num_green_weights * green_weight)

theorem metal_bar_weight : bar_weight = 2 := by
  sorry

end metal_bar_weight_l1740_174017


namespace power_product_squared_l1740_174043

theorem power_product_squared (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end power_product_squared_l1740_174043


namespace article_cost_price_l1740_174057

theorem article_cost_price (C : ℝ) (S : ℝ) : 
  S = 1.05 * C ∧ 
  S - 1 = 1.045 * C → 
  C = 200 := by
sorry

end article_cost_price_l1740_174057


namespace farm_dogs_count_l1740_174012

theorem farm_dogs_count (num_houses : ℕ) (dogs_per_house : ℕ) (h1 : num_houses = 5) (h2 : dogs_per_house = 4) :
  num_houses * dogs_per_house = 20 := by
  sorry

end farm_dogs_count_l1740_174012


namespace product_of_one_plus_tangents_l1740_174089

theorem product_of_one_plus_tangents (A B C : Real) : 
  A = π / 12 →  -- 15°
  B = π / 6 →   -- 30°
  A + B + C = π / 2 →  -- 90°
  (1 + Real.tan A) * (1 + Real.tan B) * (1 + Real.tan C) = (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end product_of_one_plus_tangents_l1740_174089


namespace quadratic_function_range_difference_l1740_174058

-- Define the quadratic function
def f (x c : ℝ) : ℝ := -2 * x^2 + c

-- Define the theorem
theorem quadratic_function_range_difference (c m : ℝ) :
  (m + 2 ≤ 0) →
  (∃ (min : ℝ), ∀ (m' : ℝ), m' + 2 ≤ 0 → 
    (f (m' + 2) c - f m' c) ≥ min) ∧
  (¬∃ (max : ℝ), ∀ (m' : ℝ), m' + 2 ≤ 0 → 
    (f (m' + 2) c - f m' c) ≤ max) :=
by sorry

end quadratic_function_range_difference_l1740_174058


namespace remainder_theorem_l1740_174009

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 50 * k - 49) :
  (n^2 + 4*n + 5) % 50 = 10 := by
sorry

end remainder_theorem_l1740_174009


namespace oliver_water_usage_l1740_174019

/-- Calculates the weekly water usage for Oliver's baths given the specified conditions. -/
def weekly_water_usage (bucket_capacity : ℕ) (fill_count : ℕ) (remove_count : ℕ) (days_per_week : ℕ) : ℕ :=
  (fill_count * bucket_capacity - remove_count * bucket_capacity) * days_per_week

/-- Theorem stating that Oliver's weekly water usage is 9240 ounces under the given conditions. -/
theorem oliver_water_usage :
  weekly_water_usage 120 14 3 7 = 9240 := by
  sorry

#eval weekly_water_usage 120 14 3 7

end oliver_water_usage_l1740_174019


namespace purely_imaginary_complex_number_l1740_174018

theorem purely_imaginary_complex_number (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 + x - 2) (x + 2)
  (z.re = 0 ∧ z.im ≠ 0) → x = 1 := by
  sorry

end purely_imaginary_complex_number_l1740_174018


namespace postage_cost_correct_l1740_174031

-- Define the postage pricing structure
def base_rate : ℚ := 50 / 100
def additional_rate : ℚ := 15 / 100
def weight_increment : ℚ := 1 / 2
def package_weight : ℚ := 28 / 10
def cost_cap : ℚ := 130 / 100

-- Calculate the postage cost
def postage_cost : ℚ :=
  base_rate + additional_rate * (Int.ceil ((package_weight - 1) / weight_increment))

-- Theorem to prove
theorem postage_cost_correct : 
  postage_cost = 110 / 100 ∧ postage_cost ≤ cost_cap := by
  sorry

end postage_cost_correct_l1740_174031


namespace tissues_left_proof_l1740_174042

/-- The number of tissues left after buying boxes and using some tissues. -/
def tissues_left (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_used : ℕ) : ℕ :=
  tissues_per_box * boxes_bought - tissues_used

/-- Theorem: Given the conditions, prove that the number of tissues left is 270. -/
theorem tissues_left_proof :
  let tissues_per_box : ℕ := 160
  let boxes_bought : ℕ := 3
  let tissues_used : ℕ := 210
  tissues_left tissues_per_box boxes_bought tissues_used = 270 := by
  sorry

end tissues_left_proof_l1740_174042


namespace even_decreasing_inequality_l1740_174006

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem even_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) (h_dec : decreasing_on_nonneg f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 := by sorry

end even_decreasing_inequality_l1740_174006


namespace max_red_beads_l1740_174047

/-- Represents a string of beads with red, blue, and green colors. -/
structure BeadString where
  total_beads : ℕ
  red_beads : ℕ
  blue_beads : ℕ
  green_beads : ℕ
  sum_constraint : total_beads = red_beads + blue_beads + green_beads
  green_constraint : ∀ n : ℕ, n + 6 ≤ total_beads → ∃ i, n ≤ i ∧ i < n + 6 ∧ green_beads > 0
  blue_constraint : ∀ n : ℕ, n + 11 ≤ total_beads → ∃ i, n ≤ i ∧ i < n + 11 ∧ blue_beads > 0

/-- The maximum number of red beads in a string of 150 beads with given constraints is 112. -/
theorem max_red_beads :
  ∀ bs : BeadString, bs.total_beads = 150 → bs.red_beads ≤ 112 :=
by sorry

end max_red_beads_l1740_174047


namespace value_of_expression_l1740_174028

theorem value_of_expression (m n : ℤ) (h1 : |m| = 5) (h2 : |n| = 4) (h3 : m * n < 0) :
  m^2 - m*n + n = 41 ∨ m^2 - m*n + n = 49 :=
sorry

end value_of_expression_l1740_174028


namespace quadratic_root_difference_l1740_174035

theorem quadratic_root_difference : 
  let a : ℝ := 5 + 3 * Real.sqrt 5
  let b : ℝ := 5 + Real.sqrt 5
  let c : ℝ := -3
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  abs (root1 - root2) = 1/2 + 2 * Real.sqrt 5 := by
sorry

end quadratic_root_difference_l1740_174035


namespace car_trip_speed_l1740_174015

/-- Given a 6-hour trip with an average speed of 38 miles per hour,
    where the speed for the last 2 hours is 44 miles per hour,
    prove that the average speed for the first 4 hours is 35 miles per hour. -/
theorem car_trip_speed :
  ∀ (first_4_hours_speed : ℝ),
    (first_4_hours_speed * 4 + 44 * 2) / 6 = 38 →
    first_4_hours_speed = 35 :=
by
  sorry

end car_trip_speed_l1740_174015


namespace unique_prime_with_no_cubic_sum_l1740_174073

-- Define the property for a prime p
def has_no_cubic_sum (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℤ, ∀ x y : ℤ, (x^3 + y^3) % p ≠ n % p

-- State the theorem
theorem unique_prime_with_no_cubic_sum :
  ∀ p : ℕ, has_no_cubic_sum p ↔ p = 7 :=
sorry

end unique_prime_with_no_cubic_sum_l1740_174073


namespace intersection_M_N_l1740_174083

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x ^ 2 ≤ 4}

def N : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end intersection_M_N_l1740_174083


namespace probability_at_least_four_successes_l1740_174080

theorem probability_at_least_four_successes (n : ℕ) (p : ℝ) (h1 : n = 5) (h2 : p = 3/5) :
  let binomial := fun (k : ℕ) => n.choose k * p^k * (1 - p)^(n - k)
  binomial 4 + binomial 5 = 1053/3125 := by sorry

end probability_at_least_four_successes_l1740_174080


namespace meaningful_expression_l1740_174013

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (Real.sqrt (a + 1)) / (a - 2)) ↔ (a ≥ -1 ∧ a ≠ 2) :=
by sorry

end meaningful_expression_l1740_174013


namespace whiteboard_ink_cost_l1740_174036

/-- Calculates the cost of whiteboard ink usage for one day -/
theorem whiteboard_ink_cost (num_classes : ℕ) (boards_per_class : ℕ) (ink_per_board : ℝ) (cost_per_ml : ℝ) : 
  num_classes = 5 → 
  boards_per_class = 2 → 
  ink_per_board = 20 → 
  cost_per_ml = 0.5 → 
  (num_classes * boards_per_class * ink_per_board * cost_per_ml : ℝ) = 100 := by
sorry

end whiteboard_ink_cost_l1740_174036


namespace value_of_expression_l1740_174001

theorem value_of_expression (s t : ℝ) 
  (hs : 19 * s^2 + 99 * s + 1 = 0)
  (ht : t^2 + 99 * t + 19 = 0)
  (hst : s * t ≠ 1) :
  (s * t + 4 * s + 1) / t = -5 := by
  sorry

end value_of_expression_l1740_174001


namespace satellite_sensor_ratio_l1740_174052

theorem satellite_sensor_ratio (total_units : Nat) (upgrade_fraction : Rat) : 
  total_units = 24 → 
  upgrade_fraction = 1 / 7 → 
  (∃ (non_upgraded_per_unit total_upgraded : Nat), 
    (non_upgraded_per_unit : Rat) / (total_upgraded : Rat) = 1 / 4) :=
by sorry

end satellite_sensor_ratio_l1740_174052


namespace oil_redistribution_l1740_174025

theorem oil_redistribution (trucks_type1 trucks_type2 boxes_per_truck1 boxes_per_truck2 containers_per_box final_trucks : ℕ) 
  (h1 : trucks_type1 = 7)
  (h2 : trucks_type2 = 5)
  (h3 : boxes_per_truck1 = 20)
  (h4 : boxes_per_truck2 = 12)
  (h5 : containers_per_box = 8)
  (h6 : final_trucks = 10) :
  (trucks_type1 * boxes_per_truck1 + trucks_type2 * boxes_per_truck2) * containers_per_box / final_trucks = 160 := by
  sorry

#check oil_redistribution

end oil_redistribution_l1740_174025


namespace range_of_a_given_negative_root_l1740_174032

/-- Given that the equation 5^x = (a+3)/(5-a) has a negative root, 
    prove that the range of values for a is -3 < a < 1 -/
theorem range_of_a_given_negative_root (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ 5^x = (a+3)/(5-a)) → -3 < a ∧ a < 1 := by
  sorry

end range_of_a_given_negative_root_l1740_174032


namespace regression_line_estimate_l1740_174091

/-- Represents a linear regression line y = ax + b -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y-value for a given x-value on the regression line -/
def RegressionLine.evaluate (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

theorem regression_line_estimate :
  ∀ (line : RegressionLine),
    line.slope = 1.23 →
    line.evaluate 4 = 5 →
    line.evaluate 2 = 2.54 := by
  sorry

end regression_line_estimate_l1740_174091


namespace perpendicular_necessary_not_sufficient_l1740_174029

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Two lines are different -/
def Line.different (l1 l2 : Line) : Prop := sorry

/-- A line is in a plane -/
def Line.inPlane (l : Line) (p : Plane) : Prop := sorry

/-- A line is outside a plane -/
def Line.outsidePlane (l : Line) (p : Plane) : Prop := sorry

/-- A line is perpendicular to another line -/
def Line.perpendicular (l1 l2 : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def Line.perpendicularToPlane (l : Line) (p : Plane) : Prop := sorry

theorem perpendicular_necessary_not_sufficient
  (α : Plane) (a b l : Line)
  (h1 : a.inPlane α)
  (h2 : b.inPlane α)
  (h3 : l.outsidePlane α)
  (h4 : a.different b) :
  (l.perpendicularToPlane α → (l.perpendicular a ∧ l.perpendicular b)) ∧
  ¬((l.perpendicular a ∧ l.perpendicular b) → l.perpendicularToPlane α) :=
by sorry

end perpendicular_necessary_not_sufficient_l1740_174029


namespace trig_identity_l1740_174065

theorem trig_identity (α : Real) (h : Real.sin α + Real.sin α ^ 2 = 1) :
  Real.cos α ^ 2 + Real.cos α ^ 4 = 1 := by
  sorry

end trig_identity_l1740_174065


namespace complex_product_magnitude_l1740_174041

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = t - 3 * Complex.I →
  t = 6 * Real.sqrt 6 := by
sorry

end complex_product_magnitude_l1740_174041


namespace number_at_21_21_l1740_174046

/-- Represents the number at a given position in the matrix -/
def matrixNumber (row : ℕ) (col : ℕ) : ℕ :=
  row^2 - (col - 1)

/-- The theorem stating that the number in the 21st row and 21st column is 421 -/
theorem number_at_21_21 : matrixNumber 21 21 = 421 := by
  sorry

end number_at_21_21_l1740_174046


namespace cubic_real_root_l1740_174010

/-- Given a cubic polynomial ax³ + 3x² + bx - 125 = 0 where a and b are real numbers,
    if -3 - 4i is a root of this polynomial, then 5 is the real root of the polynomial. -/
theorem cubic_real_root (a b : ℝ) :
  (∃ (z : ℂ), z = -3 - 4*I ∧ a * z^3 + 3 * z^2 + b * z - 125 = 0) →
  (∃ (x : ℝ), x = 5 ∧ a * x^3 + 3 * x^2 + b * x - 125 = 0) :=
by sorry

end cubic_real_root_l1740_174010


namespace log_equality_implies_ratio_l1740_174074

theorem log_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.log a / Real.log 9 = Real.log b / Real.log 12 ∧ 
       Real.log a / Real.log 9 = Real.log (a + b) / Real.log 16) : 
  b / a = (1 + Real.sqrt 5) / 2 := by
sorry

end log_equality_implies_ratio_l1740_174074


namespace product_of_r_values_l1740_174059

theorem product_of_r_values : ∃ (r₁ r₂ : ℝ), 
  (∀ x : ℝ, x ≠ 0 → (1 / (3 * x) = (r₁ - x) / 8 ↔ 1 / (3 * x) = (r₂ - x) / 8)) ∧ 
  (∀ r : ℝ, (∃! x : ℝ, x ≠ 0 ∧ 1 / (3 * x) = (r - x) / 8) → (r = r₁ ∨ r = r₂)) ∧
  r₁ * r₂ = -32/3 :=
sorry

end product_of_r_values_l1740_174059


namespace line_slope_l1740_174022

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 6 * x + 7 * y - 3 = 0

-- State the theorem
theorem line_slope :
  ∃ m b : ℝ, (∀ x y : ℝ, line_equation x y ↔ y = m * x + b) ∧ m = -6/7 :=
sorry

end line_slope_l1740_174022


namespace max_bug_contacts_l1740_174061

/-- The number of bugs on the stick -/
def total_bugs : ℕ := 2016

/-- The maximum number of contacts between bugs -/
def max_contacts : ℕ := 1016064

/-- Theorem stating that the maximum number of contacts is achieved when half the bugs move in each direction -/
theorem max_bug_contacts :
  ∀ (a b : ℕ), a + b = total_bugs → a * b ≤ max_contacts :=
by sorry

end max_bug_contacts_l1740_174061


namespace double_mean_value_function_range_l1740_174021

/-- A function f is a double mean value function on [a,b] if there exist
    two distinct points x₁ and x₂ in (a,b) such that
    f''(x₁) = f''(x₂) = (f(b) - f(a)) / (b - a) -/
def is_double_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
  (deriv^[2] f) x₁ = (f b - f a) / (b - a) ∧
  (deriv^[2] f) x₂ = (f b - f a) / (b - a)

/-- The main theorem -/
theorem double_mean_value_function_range :
  ∀ m : ℝ, is_double_mean_value_function (fun x ↦ x^3 - 6/5 * x^2) 0 m →
  3/5 < m ∧ m ≤ 6/5 := by sorry

end double_mean_value_function_range_l1740_174021


namespace max_visible_cubes_l1740_174099

/-- The size of the cube's edge -/
def n : ℕ := 12

/-- The number of unit cubes on one face of the large cube -/
def face_cubes : ℕ := n^2

/-- The number of unit cubes along one edge of the large cube -/
def edge_cubes : ℕ := n

/-- The number of visible faces from a corner -/
def visible_faces : ℕ := 3

/-- The number of visible edges from a corner -/
def visible_edges : ℕ := 3

/-- The number of visible corners from a corner -/
def visible_corners : ℕ := 1

theorem max_visible_cubes :
  visible_faces * face_cubes - (visible_edges * edge_cubes - visible_corners) = 398 := by
  sorry

end max_visible_cubes_l1740_174099


namespace correct_calculation_l1740_174095

theorem correct_calculation (x : ℤ) (h : x - 954 = 468) : x + 954 = 2376 :=
by sorry

end correct_calculation_l1740_174095


namespace smallest_four_digit_multiple_of_112_l1740_174003

theorem smallest_four_digit_multiple_of_112 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 112 ∣ n → 1008 ≤ n :=
by sorry

end smallest_four_digit_multiple_of_112_l1740_174003


namespace first_season_episodes_l1740_174049

/-- The number of seasons in the TV show -/
def num_seasons : ℕ := 5

/-- The cost per episode for the first season in dollars -/
def first_season_cost : ℕ := 100000

/-- The cost per episode for seasons after the first in dollars -/
def other_season_cost : ℕ := 2 * first_season_cost

/-- The increase factor for the number of episodes in each season after the first -/
def episode_increase_factor : ℚ := 3/2

/-- The number of episodes in the last season -/
def last_season_episodes : ℕ := 24

/-- The total cost to produce all episodes in dollars -/
def total_cost : ℕ := 16800000

/-- Calculate the total cost of all seasons given the number of episodes in the first season -/
def calculate_total_cost (first_season_episodes : ℕ) : ℚ :=
  let first_season := first_season_cost * first_season_episodes
  let second_season := other_season_cost * (episode_increase_factor * first_season_episodes)
  let third_season := other_season_cost * (episode_increase_factor^2 * first_season_episodes)
  let fourth_season := other_season_cost * (episode_increase_factor^3 * first_season_episodes)
  let fifth_season := other_season_cost * last_season_episodes
  first_season + second_season + third_season + fourth_season + fifth_season

/-- Theorem stating that the number of episodes in the first season is 8 -/
theorem first_season_episodes : ∃ (x : ℕ), x = 8 ∧ calculate_total_cost x = total_cost := by
  sorry

end first_season_episodes_l1740_174049


namespace definite_integral_ln_squared_over_sqrt_l1740_174044

theorem definite_integral_ln_squared_over_sqrt (e : Real) :
  let f : Real → Real := fun x => (Real.log x)^2 / Real.sqrt x
  let a : Real := 1
  let b : Real := Real.exp 2
  e > 0 →
  ∫ x in a..b, f x = 24 * e - 32 := by
sorry

end definite_integral_ln_squared_over_sqrt_l1740_174044


namespace not_all_rectangles_similar_l1740_174077

/-- A rectangle is a parallelogram with all interior angles equal to 90 degrees. -/
structure Rectangle where
  sides : Fin 4 → ℝ
  angle_measure : ℝ
  is_parallelogram : True
  right_angles : angle_measure = 90

/-- Similarity in shapes means corresponding angles are equal and ratios of corresponding sides are constant. -/
def are_similar (r1 r2 : Rectangle) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 4, r1.sides i = k * r2.sides i

/-- Theorem: Not all rectangles are similar to each other. -/
theorem not_all_rectangles_similar : ¬ ∀ r1 r2 : Rectangle, are_similar r1 r2 := by
  sorry

end not_all_rectangles_similar_l1740_174077


namespace cubic_function_nonnegative_implies_parameter_bound_l1740_174070

theorem cubic_function_nonnegative_implies_parameter_bound 
  (f : ℝ → ℝ) (a : ℝ) 
  (h_def : ∀ x, f x = a * x^3 - 3 * x + 1)
  (h_nonneg : ∀ x ∈ Set.Icc 0 1, f x ≥ 0) :
  a ≥ 4 := by
  sorry

end cubic_function_nonnegative_implies_parameter_bound_l1740_174070


namespace hamburger_sales_l1740_174002

theorem hamburger_sales (total_target : ℕ) (price_per_hamburger : ℕ) (remaining_hamburgers : ℕ) : 
  total_target = 50 →
  price_per_hamburger = 5 →
  remaining_hamburgers = 4 →
  (total_target - remaining_hamburgers * price_per_hamburger) / price_per_hamburger = 6 :=
by sorry

end hamburger_sales_l1740_174002


namespace arun_speed_ratio_l1740_174069

/-- Represents the problem of finding the ratio of Arun's new speed to his original speed. -/
theorem arun_speed_ratio :
  let distance : ℝ := 30
  let arun_original_speed : ℝ := 5
  let anil_time := distance / anil_speed
  let arun_original_time := distance / arun_original_speed
  let arun_new_time := distance / arun_new_speed
  arun_original_time = anil_time + 2 →
  arun_new_time = anil_time - 1 →
  arun_new_speed / arun_original_speed = 2 :=
by
  sorry


end arun_speed_ratio_l1740_174069


namespace income_distribution_l1740_174014

theorem income_distribution (total_income : ℝ) (wife_percentage : ℝ) (orphan_percentage : ℝ) 
  (final_amount : ℝ) (num_children : ℕ) :
  total_income = 1000 →
  wife_percentage = 0.2 →
  orphan_percentage = 0.1 →
  final_amount = 500 →
  num_children = 2 →
  let remaining_after_wife := total_income * (1 - wife_percentage)
  let remaining_after_orphan := remaining_after_wife * (1 - orphan_percentage)
  let amount_to_children := remaining_after_orphan - final_amount
  let amount_per_child := amount_to_children / num_children
  amount_per_child / total_income = 0.11 := by
sorry

end income_distribution_l1740_174014


namespace smallest_cube_root_integer_l1740_174072

theorem smallest_cube_root_integer (m n : ℕ) (s : ℝ) : 
  (0 < n) →
  (0 < s) →
  (s < 1 / 2000) →
  (m = (n + s)^3) →
  (∀ k < n, ∀ t > 0, t < 1 / 2000 → ¬ (∃ l : ℕ, l = (k + t)^3)) →
  (n = 26) := by
sorry

end smallest_cube_root_integer_l1740_174072


namespace simplify_expression_l1740_174087

theorem simplify_expression : 2023^2 - 2022 * 2024 = 1 := by
  sorry

end simplify_expression_l1740_174087


namespace student_rabbit_difference_l1740_174093

-- Define the number of students per classroom
def students_per_classroom : ℕ := 24

-- Define the number of rabbits per classroom
def rabbits_per_classroom : ℕ := 3

-- Define the total number of classrooms
def total_classrooms : ℕ := 5

-- Define the number of absent rabbits
def absent_rabbits : ℕ := 1

-- Theorem statement
theorem student_rabbit_difference :
  students_per_classroom * total_classrooms - 
  (rabbits_per_classroom * total_classrooms) = 105 := by
  sorry


end student_rabbit_difference_l1740_174093


namespace additional_charge_correct_l1740_174034

/-- The charge for each additional 1/5 of a mile in a taxi ride -/
def additional_charge : ℝ := 0.40

/-- The initial charge for the first 1/5 of a mile -/
def initial_charge : ℝ := 2.50

/-- The total distance of the ride in miles -/
def total_distance : ℝ := 8

/-- The total charge for the ride -/
def total_charge : ℝ := 18.10

/-- Theorem stating that the additional charge is correct given the conditions -/
theorem additional_charge_correct :
  initial_charge + (total_distance - 1/5) / (1/5) * additional_charge = total_charge := by
  sorry

end additional_charge_correct_l1740_174034


namespace inequality_solution_sets_l1740_174078

theorem inequality_solution_sets : ∃ (x : ℝ), 
  (x > 15 ∧ ¬(x - 7 < 2*x + 8)) ∨ (x - 7 < 2*x + 8 ∧ ¬(x > 15)) ∧
  (∀ y : ℝ, (5*y > 10 ↔ 3*y > 6)) ∧
  (∀ z : ℝ, (6*z - 9 < 3*z + 6 ↔ z < 5)) ∧
  (∀ w : ℝ, (w < -2 ↔ -14*w > 28)) :=
sorry

end inequality_solution_sets_l1740_174078


namespace function_value_at_pi_third_l1740_174005

/-- Given a function f(x) = 2tan(ωx + φ) with the following properties:
    - ω > 0
    - |φ| < π/2
    - f(0) = 2√3/3
    - The period T ∈ (π/4, 3π/4)
    - (π/6, 0) is the center of symmetry of f(x)
    Prove that f(π/3) = -2√3/3 -/
theorem function_value_at_pi_third 
  (f : ℝ → ℝ) 
  (ω φ : ℝ) 
  (h1 : ∀ x, f x = 2 * Real.tan (ω * x + φ))
  (h2 : ω > 0)
  (h3 : abs φ < Real.pi / 2)
  (h4 : f 0 = 2 * Real.sqrt 3 / 3)
  (h5 : ∃ T, T ∈ Set.Ioo (Real.pi / 4) (3 * Real.pi / 4) ∧ ∀ x, f (x + T) = f x)
  (h6 : ∀ x, f (Real.pi / 3 - x) = f (Real.pi / 3 + x)) :
  f (Real.pi / 3) = -2 * Real.sqrt 3 / 3 := by
  sorry

end function_value_at_pi_third_l1740_174005


namespace fixed_distance_theorem_l1740_174039

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_fixed_distance (p a b : E) : Prop :=
  ∃ (c : ℝ), ∀ (q : E), ‖p - b‖ = 3 * ‖p - a‖ → ‖q - b‖ = 3 * ‖q - a‖ → 
    ‖p - ((9/8 : ℝ) • a - (1/8 : ℝ) • b)‖ = ‖q - ((9/8 : ℝ) • a - (1/8 : ℝ) • b)‖

theorem fixed_distance_theorem (a b p : E) :
  ‖p - b‖ = 3 * ‖p - a‖ → is_fixed_distance p a b :=
by sorry

end fixed_distance_theorem_l1740_174039


namespace system_solution_l1740_174098

theorem system_solution : 
  ∃! (x y : ℚ), 3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33 :=
by
  -- The proof goes here
  sorry

end system_solution_l1740_174098


namespace carousel_seating_arrangement_l1740_174016

-- Define the friends
inductive Friend
| Alan
| Bella
| Chloe
| David
| Emma

-- Define the seats
inductive Seat
| One
| Two
| Three
| Four
| Five

-- Define the seating arrangement
def SeatingArrangement := Friend → Seat

-- Define the condition of being opposite
def isOpposite (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ s2 = Seat.Three) ∨ (s1 = Seat.Two ∧ s2 = Seat.Four) ∨
  (s1 = Seat.Three ∧ s2 = Seat.Five) ∨ (s1 = Seat.Four ∧ s2 = Seat.One) ∨
  (s1 = Seat.Five ∧ s2 = Seat.Two)

-- Define the condition of being two seats away
def isTwoSeatsAway (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ s2 = Seat.Three) ∨ (s1 = Seat.Two ∧ s2 = Seat.Four) ∨
  (s1 = Seat.Three ∧ s2 = Seat.Five) ∨ (s1 = Seat.Four ∧ s2 = Seat.One) ∨
  (s1 = Seat.Five ∧ s2 = Seat.Two)

-- Define the condition of being next to each other
def isNextTo (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ (s2 = Seat.Two ∨ s2 = Seat.Five)) ∨
  (s1 = Seat.Two ∧ (s2 = Seat.One ∨ s2 = Seat.Three)) ∨
  (s1 = Seat.Three ∧ (s2 = Seat.Two ∨ s2 = Seat.Four)) ∨
  (s1 = Seat.Four ∧ (s2 = Seat.Three ∨ s2 = Seat.Five)) ∨
  (s1 = Seat.Five ∧ (s2 = Seat.Four ∨ s2 = Seat.One))

-- Define the condition of being to the immediate left
def isImmediateLeft (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ s2 = Seat.Two) ∨
  (s1 = Seat.Two ∧ s2 = Seat.Three) ∨
  (s1 = Seat.Three ∧ s2 = Seat.Four) ∨
  (s1 = Seat.Four ∧ s2 = Seat.Five) ∨
  (s1 = Seat.Five ∧ s2 = Seat.One)

theorem carousel_seating_arrangement 
  (seating : SeatingArrangement)
  (h1 : isOpposite (seating Friend.Chloe) (seating Friend.Emma))
  (h2 : isTwoSeatsAway (seating Friend.David) (seating Friend.Alan))
  (h3 : ¬isNextTo (seating Friend.Alan) (seating Friend.Emma))
  (h4 : isNextTo (seating Friend.Bella) (seating Friend.Emma))
  : isImmediateLeft (seating Friend.Chloe) (seating Friend.Alan) :=
sorry

end carousel_seating_arrangement_l1740_174016


namespace negative_two_cubed_equality_l1740_174081

theorem negative_two_cubed_equality : -2^3 = (-2)^3 := by sorry

end negative_two_cubed_equality_l1740_174081


namespace painting_selections_l1740_174066

/-- The number of traditional Chinese paintings -/
def traditional_paintings : Nat := 5

/-- The number of oil paintings -/
def oil_paintings : Nat := 2

/-- The number of watercolor paintings -/
def watercolor_paintings : Nat := 7

/-- The number of ways to choose one painting from each category -/
def one_from_each : Nat := traditional_paintings * oil_paintings * watercolor_paintings

/-- The number of ways to choose two paintings of different types -/
def two_different_types : Nat := 
  traditional_paintings * oil_paintings + 
  traditional_paintings * watercolor_paintings + 
  oil_paintings * watercolor_paintings

theorem painting_selections :
  one_from_each = 70 ∧ two_different_types = 59 := by sorry

end painting_selections_l1740_174066


namespace intersection_of_A_and_B_l1740_174008

def A : Set ℕ := {x | ∃ n : ℕ, x = 3 * n + 2}
def B : Set ℕ := {6, 8, 10, 12, 14}

theorem intersection_of_A_and_B : A ∩ B = {8, 14} := by sorry

end intersection_of_A_and_B_l1740_174008


namespace inequality_proof_l1740_174079

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / ((a * b * c) ^ (1/3) * (1 + (a * b * c) ^ (1/3))) :=
by sorry

end inequality_proof_l1740_174079


namespace range_of_a_l1740_174055

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 - (a + 1)*x + 1 > 0

def Q (a : ℝ) : Prop := ∀ x : ℝ, |x - 1| ≥ a + 2

-- State the theorem
theorem range_of_a (a : ℝ) : (¬(P a ∨ Q a)) → a ≥ 1 := by
  sorry

end range_of_a_l1740_174055


namespace highest_score_is_143_l1740_174027

/-- Represents a batsman's performance in a cricket tournament --/
structure BatsmanPerformance where
  totalInnings : ℕ
  averageRuns : ℚ
  highestScore : ℕ
  lowestScore : ℕ
  centuryCount : ℕ

/-- Theorem stating the highest score of the batsman given the conditions --/
theorem highest_score_is_143 (b : BatsmanPerformance) : 
  b.totalInnings = 46 ∧
  b.averageRuns = 58 ∧
  b.highestScore - b.lowestScore = 150 ∧
  (b.totalInnings * b.averageRuns - b.highestScore - b.lowestScore) / (b.totalInnings - 2) = b.averageRuns ∧
  b.centuryCount = 5 ∧
  ∀ score, score ≠ b.highestScore → score < 100 →
  b.highestScore = 143 := by
  sorry


end highest_score_is_143_l1740_174027


namespace solution_set_of_inequality_l1740_174023

theorem solution_set_of_inequality (x : ℝ) :
  x * |x - 1| > 0 ↔ x ∈ Set.Ioo 0 1 ∪ Set.Ioi 1 := by sorry

end solution_set_of_inequality_l1740_174023


namespace euclidean_algorithm_bound_l1740_174097

/-- The number of divisions performed by the Euclidean algorithm -/
def euclidean_divisions (a b : ℕ) : ℕ := sorry

/-- The number of digits of a natural number in decimal -/
def num_digits (n : ℕ) : ℕ := sorry

theorem euclidean_algorithm_bound (a b : ℕ) (h1 : a > b) (h2 : b > 0) :
  euclidean_divisions a b ≤ 5 * (num_digits b) := by sorry

end euclidean_algorithm_bound_l1740_174097


namespace reflection_theorem_l1740_174004

noncomputable def C₁ (x : ℝ) : ℝ := Real.arccos (-x)

theorem reflection_theorem (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) :
  ∃ y, C₁ y = x ∧ y = -Real.cos x :=
by sorry

end reflection_theorem_l1740_174004


namespace y_finishing_time_l1740_174000

/-- The number of days it takes y to finish the remaining work after x has worked for 8 days -/
def days_for_y_to_finish (x_total_days y_total_days x_worked_days : ℕ) : ℕ :=
  (y_total_days * (x_total_days - x_worked_days)) / x_total_days

theorem y_finishing_time 
  (x_total_days : ℕ) 
  (y_total_days : ℕ) 
  (x_worked_days : ℕ) 
  (h1 : x_total_days = 40)
  (h2 : y_total_days = 40)
  (h3 : x_worked_days = 8) :
  days_for_y_to_finish x_total_days y_total_days x_worked_days = 32 := by
sorry

#eval days_for_y_to_finish 40 40 8

end y_finishing_time_l1740_174000


namespace geometric_sequence_problem_l1740_174096

theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n : ℕ, a n > 0) →  -- Positive sequence
  (∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n) →  -- Geometric sequence
  a 3 = 3 →  -- Given condition
  a 5 = 8 * a 7 →  -- Given condition
  a 10 = 3 * Real.sqrt 2 / 128 := by
sorry

end geometric_sequence_problem_l1740_174096


namespace unique_x_divisible_by_15_l1740_174020

def is_valid_x (x : ℕ) : Prop :=
  x < 10 ∧ (∃ n : ℕ, x * 1000 + 200 + x * 10 + 3 = 15 * n)

theorem unique_x_divisible_by_15 : ∃! x : ℕ, is_valid_x x :=
  sorry

end unique_x_divisible_by_15_l1740_174020
