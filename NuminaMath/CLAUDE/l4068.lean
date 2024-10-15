import Mathlib

namespace NUMINAMATH_CALUDE_garden_perimeter_l4068_406851

/-- The perimeter of a rectangular garden with given length and breadth -/
theorem garden_perimeter (length breadth : ℝ) (h1 : length = 360) (h2 : breadth = 240) :
  2 * (length + breadth) = 1200 := by
  sorry

#check garden_perimeter

end NUMINAMATH_CALUDE_garden_perimeter_l4068_406851


namespace NUMINAMATH_CALUDE_b_power_sum_l4068_406876

theorem b_power_sum (b : ℝ) (h : 5 = b + b⁻¹) : b^6 + b⁻¹^6 = 12239 := by sorry

end NUMINAMATH_CALUDE_b_power_sum_l4068_406876


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l4068_406811

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l4068_406811


namespace NUMINAMATH_CALUDE_minyoung_fruit_sale_l4068_406861

theorem minyoung_fruit_sale :
  ∀ (tangerines apples : ℕ),
    tangerines = 2 →
    apples = 7 →
    tangerines + apples = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_minyoung_fruit_sale_l4068_406861


namespace NUMINAMATH_CALUDE_min_value_of_expression_l4068_406845

theorem min_value_of_expression (m n : ℝ) : 
  m > 0 → n > 0 → 2 * m - n * (-2) - 2 = 0 → 
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 2 * m' - n' * (-2) - 2 = 0 → 
    1 / m + 2 / n ≤ 1 / m' + 2 / n') → 
  1 / m + 2 / n = 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l4068_406845


namespace NUMINAMATH_CALUDE_payment_calculation_l4068_406831

theorem payment_calculation (payment_per_room : ℚ) (rooms_cleaned : ℚ) : 
  payment_per_room = 15 / 4 →
  rooms_cleaned = 9 / 5 →
  payment_per_room * rooms_cleaned = 27 / 4 := by
sorry

end NUMINAMATH_CALUDE_payment_calculation_l4068_406831


namespace NUMINAMATH_CALUDE_first_month_bill_is_50_l4068_406890

/-- Represents Elvin's monthly telephone bill --/
structure PhoneBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- The total bill is the sum of call charge and internet charge --/
def PhoneBill.total (bill : PhoneBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem first_month_bill_is_50 
  (firstMonth secondMonth : PhoneBill)
  (h1 : firstMonth.total = 50)
  (h2 : secondMonth.total = 76)
  (h3 : secondMonth.callCharge = 2 * firstMonth.callCharge)
  (h4 : firstMonth.internetCharge = secondMonth.internetCharge) :
  firstMonth.total = 50 := by
  sorry

#check first_month_bill_is_50

end NUMINAMATH_CALUDE_first_month_bill_is_50_l4068_406890


namespace NUMINAMATH_CALUDE_range_of_a_l4068_406819

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def B : Set ℝ := {x : ℝ | x ≥ 2}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (Set.univ \ B) ∪ A a = A a → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4068_406819


namespace NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l4068_406873

/-- Represents the fishing schedule in the coastal village -/
structure FishingSchedule where
  daily : ℕ
  everyOtherDay : ℕ
  everyThreeDay : ℕ
  yesterdayCount : ℕ
  todayCount : ℕ

/-- Calculates the number of people fishing tomorrow given the fishing schedule -/
def tomorrowFishers (schedule : FishingSchedule) : ℕ :=
  schedule.daily + schedule.everyThreeDay + (schedule.everyOtherDay - (schedule.yesterdayCount - schedule.daily))

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_fishers_tomorrow (schedule : FishingSchedule) 
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterdayCount = 12)
  (h5 : schedule.todayCount = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

#eval tomorrowFishers { daily := 7, everyOtherDay := 8, everyThreeDay := 3, yesterdayCount := 12, todayCount := 10 }

end NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l4068_406873


namespace NUMINAMATH_CALUDE_functions_characterization_l4068_406854

variable (f g : ℚ → ℚ)

-- Define the conditions
axiom condition1 : ∀ x y : ℚ, f (g x + g y) = f (g x) + y
axiom condition2 : ∀ x y : ℚ, g (f x + f y) = g (f x) + y

-- Define the theorem
theorem functions_characterization :
  ∃ a b : ℚ, (a * b = 1) ∧ (∀ x : ℚ, f x = a * x ∧ g x = b * x) :=
sorry

end NUMINAMATH_CALUDE_functions_characterization_l4068_406854


namespace NUMINAMATH_CALUDE_prob_three_same_color_l4068_406867

def total_marbles : ℕ := 23
def red_marbles : ℕ := 6
def white_marbles : ℕ := 8
def blue_marbles : ℕ := 9

def prob_same_color : ℚ := 160 / 1771

theorem prob_three_same_color :
  let prob_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
  let prob_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
  let prob_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2))
  prob_red + prob_white + prob_blue = prob_same_color := by
sorry

end NUMINAMATH_CALUDE_prob_three_same_color_l4068_406867


namespace NUMINAMATH_CALUDE_no_solution_to_system_l4068_406884

theorem no_solution_to_system :
  ¬ ∃ (x y : ℝ), (2 * x - 3 * y = 8) ∧ (6 * y - 4 * x = 9) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l4068_406884


namespace NUMINAMATH_CALUDE_tommy_house_price_l4068_406898

/-- The original price of Tommy's first house -/
def original_price : ℝ := 100000

/-- The increased value of Tommy's first house -/
def increased_value : ℝ := original_price * 1.25

/-- The cost of Tommy's new house -/
def new_house_cost : ℝ := 500000

/-- The percentage Tommy paid for the new house from his own funds -/
def own_funds_percentage : ℝ := 0.25

theorem tommy_house_price :
  original_price = 100000 ∧
  increased_value = original_price * 1.25 ∧
  new_house_cost = 500000 ∧
  own_funds_percentage = 0.25 ∧
  new_house_cost * own_funds_percentage = increased_value - original_price :=
by sorry

end NUMINAMATH_CALUDE_tommy_house_price_l4068_406898


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l4068_406872

/-- Given two positive integers in ratio 4:5 with LCM 180, prove the smaller number is 144 -/
theorem smaller_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 →
  Nat.lcm a b = 180 →
  a = 144 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l4068_406872


namespace NUMINAMATH_CALUDE_angle_measure_problem_l4068_406814

theorem angle_measure_problem (C D : ℝ) 
  (h1 : C + D = 360)
  (h2 : C = 5 * D) : 
  C = 300 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l4068_406814


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l4068_406821

theorem quadratic_rewrite_sum (x : ℝ) : 
  ∃ (u v : ℝ), (9 * x^2 - 36 * x - 81 = 0 ↔ (x + u)^2 = v) ∧ u + v = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l4068_406821


namespace NUMINAMATH_CALUDE_circumcircle_equation_l4068_406812

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-1, 5)
def B : ℝ × ℝ := (5, 5)
def C : ℝ × ℝ := (6, -2)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Theorem statement
theorem circumcircle_equation :
  (circle_equation A.1 A.2) ∧
  (circle_equation B.1 B.2) ∧
  (circle_equation C.1 C.2) ∧
  (∀ (x y : ℝ), circle_equation x y → 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l4068_406812


namespace NUMINAMATH_CALUDE_flower_bouquet_carnations_percentage_l4068_406805

theorem flower_bouquet_carnations_percentage 
  (total_flowers : ℕ) 
  (pink_flowers red_flowers pink_roses red_roses pink_carnations red_carnations : ℕ) :
  (pink_flowers = total_flowers / 2) →
  (red_flowers = total_flowers / 2) →
  (pink_roses = pink_flowers * 2 / 5) →
  (red_carnations = red_flowers * 2 / 3) →
  (pink_carnations = pink_flowers - pink_roses) →
  (red_roses = red_flowers - red_carnations) →
  (((pink_carnations + red_carnations : ℚ) / total_flowers) * 100 = 63) := by
  sorry

end NUMINAMATH_CALUDE_flower_bouquet_carnations_percentage_l4068_406805


namespace NUMINAMATH_CALUDE_pythagoras_academy_olympiad_students_l4068_406810

/-- The number of distinct students taking the Math Olympiad at Pythagoras Academy -/
def distinctStudents (eulerStudents gaussStudents fibonacciStudents doubleCountedStudents : ℕ) : ℕ :=
  eulerStudents + gaussStudents + fibonacciStudents - doubleCountedStudents

/-- Theorem stating the number of distinct students taking the Math Olympiad -/
theorem pythagoras_academy_olympiad_students :
  distinctStudents 15 10 12 3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_pythagoras_academy_olympiad_students_l4068_406810


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l4068_406804

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l4068_406804


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4068_406841

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  (1/2 * a * b * Real.sin C = Real.sqrt 3 → a = 2 ∧ b = 2) ∧
  (Real.sin B = 2 * Real.sin A → 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l4068_406841


namespace NUMINAMATH_CALUDE_function_increasing_range_l4068_406815

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then Real.log x / Real.log a else a * x - 2

theorem function_increasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 
  1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_function_increasing_range_l4068_406815


namespace NUMINAMATH_CALUDE_H2O_formation_l4068_406807

-- Define the chemical reaction
def reaction_ratio : ℚ := 1

-- Define the given amounts of reactants
def KOH_moles : ℚ := 3
def NH4I_moles : ℚ := 3

-- Define the theorem
theorem H2O_formation (h : KOH_moles = NH4I_moles) :
  min KOH_moles NH4I_moles = 3 ∧ 
  reaction_ratio * min KOH_moles NH4I_moles = 3 :=
by sorry

end NUMINAMATH_CALUDE_H2O_formation_l4068_406807


namespace NUMINAMATH_CALUDE_carlos_baseball_cards_l4068_406880

theorem carlos_baseball_cards :
  ∀ (jorge matias carlos : ℕ),
    jorge = matias →
    matias = carlos - 6 →
    jorge + matias + carlos = 48 →
    carlos = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_carlos_baseball_cards_l4068_406880


namespace NUMINAMATH_CALUDE_max_value_expression_l4068_406835

theorem max_value_expression (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  (x^2 - 2*x*y + 2*y^2) * (x^2 - 2*x*z + 2*z^2) * (y^2 - 2*y*z + 2*z^2) ≤ 12 ∧
  ∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 3 ∧
    (x^2 - 2*x*y + 2*y^2) * (x^2 - 2*x*z + 2*z^2) * (y^2 - 2*y*z + 2*z^2) = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l4068_406835


namespace NUMINAMATH_CALUDE_reciprocal_roots_l4068_406849

theorem reciprocal_roots (a b c : ℝ) (x y : ℝ) : 
  (a * x^2 + b * x + c = 0 ↔ c * (1/x)^2 + b * (1/x) + a = 0) ∧ 
  (c * y^2 + b * y + a = 0 ↔ a * (1/y)^2 + b * (1/y) + c = 0) := by
sorry

end NUMINAMATH_CALUDE_reciprocal_roots_l4068_406849


namespace NUMINAMATH_CALUDE_truck_rental_miles_driven_l4068_406838

theorem truck_rental_miles_driven 
  (rental_fee : ℚ) 
  (charge_per_mile : ℚ) 
  (total_paid : ℚ) 
  (h1 : rental_fee = 2099 / 100)
  (h2 : charge_per_mile = 25 / 100)
  (h3 : total_paid = 9574 / 100) : 
  (total_paid - rental_fee) / charge_per_mile = 299 := by
sorry

#eval (9574 / 100 - 2099 / 100) / (25 / 100)

end NUMINAMATH_CALUDE_truck_rental_miles_driven_l4068_406838


namespace NUMINAMATH_CALUDE_unique_pair_satisfying_conditions_l4068_406860

theorem unique_pair_satisfying_conditions :
  ∀ a b : ℕ+,
  a + b + (Nat.gcd a b)^2 = Nat.lcm a b ∧
  Nat.lcm a b = 2 * Nat.lcm (a - 1) b →
  a = 6 ∧ b = 15 := by
sorry

end NUMINAMATH_CALUDE_unique_pair_satisfying_conditions_l4068_406860


namespace NUMINAMATH_CALUDE_not_always_intersects_x_axis_l4068_406868

/-- Represents a circle in a 2D plane -/
structure Circle where
  a : ℝ  -- x-coordinate of the center
  b : ℝ  -- y-coordinate of the center
  r : ℝ  -- radius
  r_pos : r > 0

/-- Predicate to check if a circle intersects the x-axis -/
def intersects_x_axis (c : Circle) : Prop :=
  ∃ x : ℝ, (x - c.a)^2 + c.b^2 = c.r^2

/-- Theorem stating that b < r does not always imply intersection with x-axis -/
theorem not_always_intersects_x_axis :
  ¬ (∀ c : Circle, c.b < c.r → intersects_x_axis c) :=
sorry

end NUMINAMATH_CALUDE_not_always_intersects_x_axis_l4068_406868


namespace NUMINAMATH_CALUDE_digit_sum_divisibility_l4068_406846

theorem digit_sum_divisibility (n k : ℕ) (hn : n > 0) (hk : k ≥ n) (h3 : ¬3 ∣ n) :
  ∃ m : ℕ, m > 0 ∧ n ∣ m ∧ (∃ digits : List ℕ, m.digits 10 = digits ∧ digits.sum = k) :=
sorry

end NUMINAMATH_CALUDE_digit_sum_divisibility_l4068_406846


namespace NUMINAMATH_CALUDE_well_volume_l4068_406855

/-- The volume of a cylindrical well with diameter 2 meters and depth 14 meters is π * 14 cubic meters -/
theorem well_volume (π : ℝ) (h : π = Real.pi) :
  let diameter : ℝ := 2
  let depth : ℝ := 14
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * depth
  volume = π * 14 := by sorry

end NUMINAMATH_CALUDE_well_volume_l4068_406855


namespace NUMINAMATH_CALUDE_trapezoid_median_l4068_406848

/-- Given a triangle with base 24 inches and area 192 square inches, and a trapezoid with the same 
    height and area as the triangle, the median of the trapezoid is 12 inches. -/
theorem trapezoid_median (triangle_base : ℝ) (triangle_area : ℝ) (trapezoid_height : ℝ) 
  (trapezoid_median : ℝ) : 
  triangle_base = 24 → 
  triangle_area = 192 → 
  triangle_area = (1/2) * triangle_base * trapezoid_height → 
  triangle_area = trapezoid_median * trapezoid_height → 
  trapezoid_median = 12 := by
  sorry

#check trapezoid_median

end NUMINAMATH_CALUDE_trapezoid_median_l4068_406848


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l4068_406891

theorem product_from_lcm_gcd : 
  ∀ (a b : ℕ+), 
    Nat.lcm a b = 72 → 
    Nat.gcd a b = 8 → 
    (a : ℕ) * b = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l4068_406891


namespace NUMINAMATH_CALUDE_square_roots_values_l4068_406827

theorem square_roots_values (m : ℝ) (a : ℝ) (h1 : a > 0) 
  (h2 : (3 * m - 1)^2 = a) (h3 : (-2 * m - 2)^2 = a) :
  a = 64 ∨ a = 64/25 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_values_l4068_406827


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4068_406862

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4068_406862


namespace NUMINAMATH_CALUDE_missing_digit_is_three_l4068_406818

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def six_digit_number (a b c d e f : ℕ) : ℕ := a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f

theorem missing_digit_is_three :
  ∃ (x : ℕ), x < 10 ∧ is_divisible_by_9 (six_digit_number 3 4 6 x 9 2) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_digit_is_three_l4068_406818


namespace NUMINAMATH_CALUDE_point_on_line_with_distance_l4068_406887

theorem point_on_line_with_distance (x₀ y₀ : ℝ) :
  (3 * x₀ + y₀ - 5 = 0) →
  (|x₀ - y₀ - 1| / Real.sqrt 2 = Real.sqrt 2) →
  ((x₀ = 1 ∧ y₀ = 2) ∨ (x₀ = 2 ∧ y₀ = -1)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_with_distance_l4068_406887


namespace NUMINAMATH_CALUDE_max_value_of_f_l4068_406839

/-- The function f(x) = x(1-2x) -/
def f (x : ℝ) := x * (1 - 2 * x)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/8 ∧ ∀ x, 0 < x → x < 1/2 → f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4068_406839


namespace NUMINAMATH_CALUDE_problem_solution_l4068_406834

def f (t : ℝ) : ℝ := t^2003 + 2002*t

theorem problem_solution (x y : ℝ) 
  (h1 : f (x - 1) = -1)
  (h2 : f (y - 2) = 1) : 
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4068_406834


namespace NUMINAMATH_CALUDE_sample_size_is_80_l4068_406865

/-- Represents the ratio of quantities for products A, B, and C -/
structure ProductRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a stratified sample -/
structure StratifiedSample where
  ratio : ProductRatio
  units_of_a : ℕ

/-- Theorem stating that given the specific conditions, the sample size is 80 -/
theorem sample_size_is_80 (sample : StratifiedSample) 
  (h_ratio : sample.ratio = ProductRatio.mk 2 3 5)
  (h_units_a : sample.units_of_a = 16) : 
  (sample.units_of_a / sample.ratio.a) * (sample.ratio.a + sample.ratio.b + sample.ratio.c) = 80 := by
  sorry

#check sample_size_is_80

end NUMINAMATH_CALUDE_sample_size_is_80_l4068_406865


namespace NUMINAMATH_CALUDE_intersection_equals_N_l4068_406859

def U := ℝ

def M : Set ℝ := {x : ℝ | x < 1}

def N : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_equals_N : M ∩ N = N := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_N_l4068_406859


namespace NUMINAMATH_CALUDE_rachel_reading_homework_l4068_406886

/-- The number of pages of math homework Rachel had to complete -/
def math_homework_pages : ℕ := 8

/-- The additional pages of reading homework compared to math homework -/
def additional_reading_pages : ℕ := 6

/-- The total number of pages of reading homework Rachel had to complete -/
def reading_homework_pages : ℕ := math_homework_pages + additional_reading_pages

theorem rachel_reading_homework : reading_homework_pages = 14 := by
  sorry

end NUMINAMATH_CALUDE_rachel_reading_homework_l4068_406886


namespace NUMINAMATH_CALUDE_function_value_at_three_pi_four_l4068_406866

noncomputable def f (A φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (x + φ)

theorem function_value_at_three_pi_four
  (A φ : ℝ)
  (h1 : A > 0)
  (h2 : 0 < φ)
  (h3 : φ < Real.pi)
  (h4 : ∀ x, f A φ x ≤ 1)
  (h5 : ∃ x, f A φ x = 1)
  (h6 : f A φ (Real.pi / 3) = 1 / 2) :
  f A φ (3 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_function_value_at_three_pi_four_l4068_406866


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l4068_406823

theorem consecutive_even_numbers_sum (n : ℤ) : 
  (∃ (a b c d : ℤ), 
    a = n ∧ 
    b = n + 2 ∧ 
    c = n + 4 ∧ 
    d = n + 6 ∧ 
    a + b + c + d = 52) → 
  n + 4 = 14 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l4068_406823


namespace NUMINAMATH_CALUDE_yellow_last_probability_l4068_406853

/-- Represents a bag of marbles -/
structure Bag where
  yellow : ℕ
  blue : ℕ
  white : ℕ
  black : ℕ
  green : ℕ
  red : ℕ

/-- The probability of drawing a yellow marble as the last marble -/
def last_yellow_probability (bagA bagB bagC bagD : Bag) : ℚ :=
  sorry

/-- The theorem stating the probability of drawing a yellow marble last -/
theorem yellow_last_probability :
  let bagA : Bag := { yellow := 0, blue := 0, white := 5, black := 5, green := 0, red := 0 }
  let bagB : Bag := { yellow := 8, blue := 6, white := 0, black := 0, green := 0, red := 0 }
  let bagC : Bag := { yellow := 3, blue := 7, white := 0, black := 0, green := 0, red := 0 }
  let bagD : Bag := { yellow := 0, blue := 0, white := 0, black := 0, green := 4, red := 6 }
  last_yellow_probability bagA bagB bagC bagD = 73 / 140 := by
  sorry

end NUMINAMATH_CALUDE_yellow_last_probability_l4068_406853


namespace NUMINAMATH_CALUDE_knights_arrangement_exists_l4068_406826

/-- Represents a knight in King Arthur's court -/
structure Knight where
  id : ℕ

/-- Represents the relationship between knights -/
inductive Relationship
  | Friend
  | Enemy

/-- Represents the seating arrangement of knights around a round table -/
def Arrangement := List Knight

/-- Function to determine if two knights are enemies -/
def areEnemies (k1 k2 : Knight) : Prop := sorry

/-- Function to count the number of enemies a knight has -/
def enemyCount (k : Knight) (knights : List Knight) : ℕ := sorry

/-- Function to check if an arrangement is valid (no adjacent enemies) -/
def isValidArrangement (arr : Arrangement) : Prop := sorry

/-- Main theorem: There exists a valid arrangement of knights -/
theorem knights_arrangement_exists (n : ℕ) (knights : List Knight) :
  knights.length = 2 * n →
  (∀ k ∈ knights, enemyCount k knights ≤ n - 1) →
  ∃ arr : Arrangement, arr.length = 2 * n ∧ isValidArrangement arr :=
sorry

end NUMINAMATH_CALUDE_knights_arrangement_exists_l4068_406826


namespace NUMINAMATH_CALUDE_pen_bag_discount_l4068_406842

theorem pen_bag_discount (price : ℝ) (discount : ℝ) (savings : ℝ) :
  price = 18 →
  discount = 0.1 →
  savings = 36 →
  ∃ (x : ℝ),
    price * (x + 1) * (1 - discount) = price * x - savings ∧
    x = 30 ∧
    price * (x + 1) * (1 - discount) = 486 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_bag_discount_l4068_406842


namespace NUMINAMATH_CALUDE_distinct_digits_base_eight_l4068_406801

/-- The number of three-digit numbers with distinct digits in base b -/
def distinctDigitNumbers (b : ℕ) : ℕ := (b - 1) * (b - 1) * (b - 2)

/-- Theorem stating that there are 250 three-digit numbers with distinct digits in base 8 -/
theorem distinct_digits_base_eight :
  distinctDigitNumbers 8 = 250 := by
  sorry

end NUMINAMATH_CALUDE_distinct_digits_base_eight_l4068_406801


namespace NUMINAMATH_CALUDE_equation_solution_l4068_406897

theorem equation_solution : 
  ∃ y : ℚ, (5 * y - 2) / (6 * y - 6) = 3 / 4 ∧ y = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4068_406897


namespace NUMINAMATH_CALUDE_mustard_total_l4068_406857

theorem mustard_total (table1 table2 table3 : ℚ) 
  (h1 : table1 = 0.25)
  (h2 : table2 = 0.25)
  (h3 : table3 = 0.38) :
  table1 + table2 + table3 = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_mustard_total_l4068_406857


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l4068_406888

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) : 
  a 5 = -8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l4068_406888


namespace NUMINAMATH_CALUDE_modulus_of_z_l4068_406832

theorem modulus_of_z (z : ℂ) (h : z / (Real.sqrt 3 - Complex.I) = 1 + Real.sqrt 3 * Complex.I) : 
  Complex.abs z = 4 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_z_l4068_406832


namespace NUMINAMATH_CALUDE_cyclist_problem_l4068_406877

/-- Proves that given the conditions of the cyclist problem, the speed of cyclist A is 10 mph --/
theorem cyclist_problem (distance : ℝ) (speed_difference : ℝ) (meeting_distance : ℝ)
  (h1 : distance = 100)
  (h2 : speed_difference = 5)
  (h3 : meeting_distance = 20) :
  ∃ (speed_a : ℝ), speed_a = 10 ∧ 
    (distance - meeting_distance) / speed_a = 
    (distance + meeting_distance) / (speed_a + speed_difference) :=
by
  sorry


end NUMINAMATH_CALUDE_cyclist_problem_l4068_406877


namespace NUMINAMATH_CALUDE_rebus_solution_l4068_406875

theorem rebus_solution :
  ∃! (A B C : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) = (100 * A + 10 * C + C) ∧
    100 * A + 10 * C + C = 1416 :=
by sorry

end NUMINAMATH_CALUDE_rebus_solution_l4068_406875


namespace NUMINAMATH_CALUDE_equation_solution_l4068_406802

theorem equation_solution :
  ∃! r : ℚ, (r + 4) / (r - 3) = (r - 2) / (r + 2) ∧ r = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4068_406802


namespace NUMINAMATH_CALUDE_trajectory_line_passes_fixed_point_l4068_406830

/-- The trajectory C is defined by the equation y^2 = 4x -/
def trajectory (x y : ℝ) : Prop := y^2 = 4*x

/-- A point P is on the trajectory if it satisfies the equation -/
def on_trajectory (P : ℝ × ℝ) : Prop :=
  trajectory P.1 P.2

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- A line passing through two points -/
def line_through (A B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

theorem trajectory_line_passes_fixed_point :
  ∀ A B : ℝ × ℝ,
  A ≠ (0, 0) → B ≠ (0, 0) → A ≠ B →
  on_trajectory A → on_trajectory B →
  dot_product A B = 0 →
  line_through A B (4, 0) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_line_passes_fixed_point_l4068_406830


namespace NUMINAMATH_CALUDE_largest_number_with_sum_16_l4068_406828

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 4

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def all_digits_valid (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, is_valid_digit d

theorem largest_number_with_sum_16 :
  ∀ n : ℕ,
    all_digits_valid n →
    digit_sum n = 16 →
    n ≤ 4432 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_sum_16_l4068_406828


namespace NUMINAMATH_CALUDE_root_between_consecutive_integers_l4068_406895

theorem root_between_consecutive_integers :
  ∃ (A B : ℤ), B = A + 1 ∧
  ∃ (x : ℝ), A < x ∧ x < B ∧ x^3 + 5*x^2 - 3*x + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_between_consecutive_integers_l4068_406895


namespace NUMINAMATH_CALUDE_tin_silver_ratio_l4068_406874

/-- Represents the composition of a metal bar made of tin and silver -/
structure MetalBar where
  tin : ℝ
  silver : ℝ

/-- Properties of the metal bar -/
def bar_properties (bar : MetalBar) : Prop :=
  bar.tin + bar.silver = 40 ∧
  0.1375 * bar.tin + 0.075 * bar.silver = 4

/-- The ratio of tin to silver in the bar is 2:3 -/
theorem tin_silver_ratio (bar : MetalBar) :
  bar_properties bar → bar.tin / bar.silver = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tin_silver_ratio_l4068_406874


namespace NUMINAMATH_CALUDE_gcd_problem_l4068_406825

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = (2 * k + 1) * 8723) :
  Int.gcd (8 * b^2 + 55 * b + 144) (4 * b + 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l4068_406825


namespace NUMINAMATH_CALUDE_valid_pairs_l4068_406808

def is_valid_pair (square : Nat) (B : Nat) : Prop :=
  let num := 532900 + square * 10 + B
  (num % 6 = 0) ∧ 
  (square % 2 = 0) ∧ 
  (square ≤ 9) ∧ 
  (B ≤ 9)

theorem valid_pairs : 
  ∀ square B, is_valid_pair square B ↔ 
    ((square = 0 ∧ B = 3) ∨ 
     (square = 2 ∧ B = 1) ∨ 
     (square = 4 ∧ B = 2) ∨ 
     (square = 6 ∧ B = 0) ∨ 
     (square = 8 ∧ B = 1)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l4068_406808


namespace NUMINAMATH_CALUDE_inscribed_circle_ratio_l4068_406896

/-- A circle inscribed in a semicircle -/
structure InscribedCircle where
  R : ℝ  -- Radius of the semicircle
  r : ℝ  -- Radius of the inscribed circle
  O : ℝ × ℝ  -- Center of the semicircle
  A : ℝ × ℝ  -- One end of the semicircle's diameter
  P : ℝ × ℝ  -- Center of the inscribed circle
  h₁ : R > 0  -- Radius of semicircle is positive
  h₂ : r > 0  -- Radius of inscribed circle is positive
  h₃ : A = (O.1 - R, O.2)  -- A is R units to the left of O
  h₄ : dist P O = dist P A  -- P is equidistant from O and A

/-- The ratio of radii in an inscribed circle is 3:8 -/
theorem inscribed_circle_ratio (c : InscribedCircle) : c.r / c.R = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_ratio_l4068_406896


namespace NUMINAMATH_CALUDE_old_barbell_cost_l4068_406844

theorem old_barbell_cost (new_barbell_cost : ℝ) (percentage_increase : ℝ) : 
  new_barbell_cost = 325 →
  percentage_increase = 0.30 →
  new_barbell_cost = (1 + percentage_increase) * (new_barbell_cost / (1 + percentage_increase)) →
  new_barbell_cost / (1 + percentage_increase) = 250 := by
sorry

end NUMINAMATH_CALUDE_old_barbell_cost_l4068_406844


namespace NUMINAMATH_CALUDE_logan_corn_purchase_l4068_406843

/-- Proves that Logan bought 15.0 pounds of corn given the problem conditions -/
theorem logan_corn_purchase 
  (corn_price : ℝ) 
  (bean_price : ℝ) 
  (total_weight : ℝ) 
  (total_cost : ℝ) 
  (h1 : corn_price = 1.20)
  (h2 : bean_price = 0.60)
  (h3 : total_weight = 30)
  (h4 : total_cost = 27.00) : 
  ∃ (corn_weight : ℝ) (bean_weight : ℝ),
    corn_weight + bean_weight = total_weight ∧ 
    corn_price * corn_weight + bean_price * bean_weight = total_cost ∧ 
    corn_weight = 15.0 := by
  sorry

end NUMINAMATH_CALUDE_logan_corn_purchase_l4068_406843


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l4068_406800

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l4068_406800


namespace NUMINAMATH_CALUDE_decimal_representation_of_fraction_l4068_406892

theorem decimal_representation_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 16 / 50 → (n : ℚ) / d = 0.32 := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_of_fraction_l4068_406892


namespace NUMINAMATH_CALUDE_cookie_revenue_l4068_406856

/-- Calculates the total revenue from selling chocolate and vanilla cookies -/
theorem cookie_revenue (chocolate_count : ℕ) (vanilla_count : ℕ) 
  (chocolate_price : ℚ) (vanilla_price : ℚ) : 
  chocolate_count * chocolate_price + vanilla_count * vanilla_price = 360 :=
by
  -- Assuming chocolate_count = 220, vanilla_count = 70, 
  -- chocolate_price = 1, and vanilla_price = 2
  have h1 : chocolate_count = 220 := by sorry
  have h2 : vanilla_count = 70 := by sorry
  have h3 : chocolate_price = 1 := by sorry
  have h4 : vanilla_price = 2 := by sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_cookie_revenue_l4068_406856


namespace NUMINAMATH_CALUDE_sock_order_ratio_l4068_406806

theorem sock_order_ratio (black_pairs blue_pairs : ℕ) (price_blue : ℝ) :
  black_pairs = 4 →
  (4 * 2 * price_blue + blue_pairs * price_blue) * 1.5 = blue_pairs * 2 * price_blue + 4 * price_blue →
  blue_pairs = 16 :=
by sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l4068_406806


namespace NUMINAMATH_CALUDE_problem_statement_l4068_406820

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x
noncomputable def g (x : ℝ) : ℝ := Real.log x + x + 1

theorem problem_statement :
  (∀ x : ℝ, f x > 0) ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ g x₀ = 0) ∧
  (¬(∀ x : ℝ, f x > 0) ↔ (∃ x₀ : ℝ, f x₀ ≤ 0)) ∧
  (¬(∃ x₀ : ℝ, x₀ > 0 ∧ g x₀ = 0) ↔ (∀ x : ℝ, x > 0 → g x ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l4068_406820


namespace NUMINAMATH_CALUDE_exactly_one_absent_probability_l4068_406869

/-- The probability of an employee being absent on a given day -/
def p_absent : ℚ := 1 / 30

/-- The probability of an employee being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of employees selected -/
def n : ℕ := 3

/-- The number of employees that should be absent -/
def k : ℕ := 1

theorem exactly_one_absent_probability :
  (n.choose k : ℚ) * p_absent^k * p_present^(n - k) = 841 / 9000 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_absent_probability_l4068_406869


namespace NUMINAMATH_CALUDE_chosen_numbers_divisibility_l4068_406881

theorem chosen_numbers_divisibility 
  (S : Finset ℕ) 
  (h_card : S.card = 250) 
  (h_bound : ∀ n ∈ S, n ≤ 501) :
  ∀ t : ℤ, ∃ a₁ a₂ a₃ a₄ : ℕ, 
    a₁ ∈ S ∧ a₂ ∈ S ∧ a₃ ∈ S ∧ a₄ ∈ S ∧ 
    23 ∣ (a₁ + a₂ + a₃ + a₄ - t) :=
by sorry

end NUMINAMATH_CALUDE_chosen_numbers_divisibility_l4068_406881


namespace NUMINAMATH_CALUDE_simons_raft_sticks_l4068_406882

theorem simons_raft_sticks (S : ℕ) : 
  S + (2 * S / 3) + (S + (2 * S / 3) + 9) = 129 → S = 51 := by
  sorry

end NUMINAMATH_CALUDE_simons_raft_sticks_l4068_406882


namespace NUMINAMATH_CALUDE_inequality_proof_l4068_406852

theorem inequality_proof (x y z p q : ℝ) (n : Nat) (h1 : y = x^n + p*x + q) (h2 : z = y^n + p*y + q) (h3 : x = z^n + p*z + q) (h4 : n = 2 ∨ n = 2010) :
  x^2*y + y^2*z + z^2*x ≥ x^2*z + y^2*x + z^2*y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4068_406852


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l4068_406858

theorem profit_percentage_previous_year 
  (revenue_prev : ℝ) 
  (profit_prev : ℝ) 
  (revenue_1999 : ℝ) 
  (profit_1999 : ℝ) 
  (h1 : revenue_1999 = 0.7 * revenue_prev) 
  (h2 : profit_1999 = 0.15 * revenue_1999) 
  (h3 : profit_1999 = 1.0499999999999999 * profit_prev) : 
  profit_prev / revenue_prev = 0.1 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l4068_406858


namespace NUMINAMATH_CALUDE_min_c_value_l4068_406822

/-- Given five consecutive positive integers a, b, c, d, e,
    if b + c + d is a perfect square and a + b + c + d + e is a perfect cube,
    then the minimum value of c is 675. -/
theorem min_c_value (a b c d e : ℕ) : 
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e → 
  ∃ m : ℕ, b + c + d = m^2 →
  ∃ n : ℕ, a + b + c + d + e = n^3 →
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' + 1 = b' ∧ b' + 1 = c' ∧ c' + 1 = d' ∧ d' + 1 = e' ∧
    ∃ m' : ℕ, b' + c' + d' = m'^2 ∧
    ∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3) →
  c' ≥ 675 :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l4068_406822


namespace NUMINAMATH_CALUDE_saree_discount_problem_l4068_406893

/-- Proves that the first discount percentage is 10% given the conditions of the saree pricing problem -/
theorem saree_discount_problem (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 600 →
  second_discount = 5 →
  final_price = 513 →
  ∃ (first_discount : ℝ),
    first_discount = 10 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_saree_discount_problem_l4068_406893


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_eq_two_l4068_406883

theorem sqrt_eight_div_sqrt_two_eq_two : Real.sqrt 8 / Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_eq_two_l4068_406883


namespace NUMINAMATH_CALUDE_number_problem_l4068_406809

theorem number_problem (x : ℝ) : (258/100 * x) / 6 = 543.95 → x = 1265 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4068_406809


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l4068_406840

theorem inverse_proportion_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = -12 / (-3) → 
  y₂ = -12 / (-2) → 
  y₃ = -12 / 2 → 
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l4068_406840


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_equals_2sqrt3_quadratic_equation_solutions_l4068_406878

-- Problem 1
theorem sqrt_sum_difference_equals_2sqrt3 :
  Real.sqrt 12 + Real.sqrt 27 / 9 - Real.sqrt (1/3) = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem quadratic_equation_solutions (x : ℝ) :
  x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_equals_2sqrt3_quadratic_equation_solutions_l4068_406878


namespace NUMINAMATH_CALUDE_conner_start_rocks_l4068_406864

/-- Represents the number of rocks collected by each person on each day -/
structure RockCollection where
  sydney_start : ℕ
  conner_start : ℕ
  sydney_day1 : ℕ
  conner_day1 : ℕ
  sydney_day2 : ℕ
  conner_day2 : ℕ
  sydney_day3 : ℕ
  conner_day3 : ℕ

/-- The rock collecting contest scenario -/
def contest_scenario : RockCollection where
  sydney_start := 837
  conner_start := 723  -- This is what we want to prove
  sydney_day1 := 4
  conner_day1 := 8 * 4
  sydney_day2 := 0
  conner_day2 := 123
  sydney_day3 := 2 * (8 * 4)
  conner_day3 := 27

/-- Calculates the total rocks for each person at the end of the contest -/
def total_rocks (rc : RockCollection) : ℕ × ℕ :=
  (rc.sydney_start + rc.sydney_day1 + rc.sydney_day2 + rc.sydney_day3,
   rc.conner_start + rc.conner_day1 + rc.conner_day2 + rc.conner_day3)

/-- Theorem stating that Conner must have started with 723 rocks to at least tie Sydney -/
theorem conner_start_rocks : 
  let (sydney_total, conner_total) := total_rocks contest_scenario
  conner_total ≥ sydney_total ∧ contest_scenario.conner_start = 723 := by
  sorry


end NUMINAMATH_CALUDE_conner_start_rocks_l4068_406864


namespace NUMINAMATH_CALUDE_quadratic_form_b_l4068_406829

/-- Given a quadratic of the form x^2 + bx + 54 where b is positive,
    if it can be rewritten as (x+m)^2 + 18, then b = 12 -/
theorem quadratic_form_b (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 54 = (x+m)^2 + 18) → 
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_b_l4068_406829


namespace NUMINAMATH_CALUDE_emily_roses_purchase_l4068_406850

theorem emily_roses_purchase (flower_cost : ℕ) (total_spent : ℕ) : 
  flower_cost = 3 →
  total_spent = 12 →
  ∃ (roses : ℕ), roses * 2 * flower_cost = total_spent ∧ roses = 2 :=
by sorry

end NUMINAMATH_CALUDE_emily_roses_purchase_l4068_406850


namespace NUMINAMATH_CALUDE_fabric_per_shirt_is_two_l4068_406879

/-- Represents the daily production and fabric usage in a tailoring business -/
structure TailoringBusiness where
  shirts_per_day : ℕ
  pants_per_day : ℕ
  fabric_per_pants : ℕ
  total_fabric_3days : ℕ

/-- Calculates the amount of fabric used for each shirt -/
def fabric_per_shirt (tb : TailoringBusiness) : ℚ :=
  let total_pants_3days := tb.pants_per_day * 3
  let fabric_for_pants := total_pants_3days * tb.fabric_per_pants
  let fabric_for_shirts := tb.total_fabric_3days - fabric_for_pants
  let total_shirts_3days := tb.shirts_per_day * 3
  fabric_for_shirts / total_shirts_3days

/-- Theorem stating that the amount of fabric per shirt is 2 yards -/
theorem fabric_per_shirt_is_two (tb : TailoringBusiness) 
    (h1 : tb.shirts_per_day = 3)
    (h2 : tb.pants_per_day = 5)
    (h3 : tb.fabric_per_pants = 5)
    (h4 : tb.total_fabric_3days = 93) :
    fabric_per_shirt tb = 2 := by
  sorry

#eval fabric_per_shirt { shirts_per_day := 3, pants_per_day := 5, fabric_per_pants := 5, total_fabric_3days := 93 }

end NUMINAMATH_CALUDE_fabric_per_shirt_is_two_l4068_406879


namespace NUMINAMATH_CALUDE_darry_total_steps_l4068_406894

/-- The number of steps Darry climbed in total -/
def total_steps (full_ladder_steps : ℕ) (full_ladder_climbs : ℕ) 
                (small_ladder_steps : ℕ) (small_ladder_climbs : ℕ) : ℕ :=
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs

/-- Proof that Darry climbed 152 steps in total -/
theorem darry_total_steps : 
  total_steps 11 10 6 7 = 152 := by
  sorry

end NUMINAMATH_CALUDE_darry_total_steps_l4068_406894


namespace NUMINAMATH_CALUDE_pencils_calculation_l4068_406824

/-- Given a setup of pencils and crayons in rows, calculates the number of pencils per row. -/
def pencils_per_row (total_items : ℕ) (rows : ℕ) (crayons_per_row : ℕ) : ℕ :=
  (total_items - rows * crayons_per_row) / rows

theorem pencils_calculation :
  pencils_per_row 638 11 27 = 31 := by
  sorry

end NUMINAMATH_CALUDE_pencils_calculation_l4068_406824


namespace NUMINAMATH_CALUDE_frog_climb_time_l4068_406837

/-- Represents the frog's climbing problem in the well -/
structure FrogClimb where
  well_depth : ℕ := 12
  climb_distance : ℕ := 3
  slide_distance : ℕ := 1
  time_to_climb : ℕ := 3
  time_to_slide : ℕ := 1
  time_at_3m_from_top : ℕ := 17

/-- Calculates the total time for the frog to reach the top of the well -/
def total_climb_time (f : FrogClimb) : ℕ :=
  sorry

/-- Theorem stating that the total climb time is 22 minutes -/
theorem frog_climb_time (f : FrogClimb) : total_climb_time f = 22 :=
  sorry

end NUMINAMATH_CALUDE_frog_climb_time_l4068_406837


namespace NUMINAMATH_CALUDE_balloons_left_after_distribution_l4068_406833

def red_balloons : ℕ := 23
def blue_balloons : ℕ := 39
def green_balloons : ℕ := 71
def yellow_balloons : ℕ := 89
def num_friends : ℕ := 10

theorem balloons_left_after_distribution :
  (red_balloons + blue_balloons + green_balloons + yellow_balloons) % num_friends = 2 :=
by sorry

end NUMINAMATH_CALUDE_balloons_left_after_distribution_l4068_406833


namespace NUMINAMATH_CALUDE_davis_remaining_sticks_l4068_406836

/-- The number of popsicle sticks Miss Davis had initially -/
def initial_sticks : ℕ := 170

/-- The number of groups in Miss Davis's class -/
def num_groups : ℕ := 10

/-- The number of popsicle sticks given to each group -/
def sticks_per_group : ℕ := 15

/-- The number of popsicle sticks Miss Davis has left -/
def remaining_sticks : ℕ := initial_sticks - (num_groups * sticks_per_group)

theorem davis_remaining_sticks :
  remaining_sticks = 20 := by sorry

end NUMINAMATH_CALUDE_davis_remaining_sticks_l4068_406836


namespace NUMINAMATH_CALUDE_toms_journey_to_virgo_l4068_406847

theorem toms_journey_to_virgo (
  train_ride : ℝ)
  (first_layover : ℝ)
  (bus_ride : ℝ)
  (second_layover : ℝ)
  (first_flight : ℝ)
  (third_layover : ℝ)
  (fourth_layover : ℝ)
  (car_drive : ℝ)
  (first_boat_ride : ℝ)
  (fifth_layover : ℝ)
  (final_walk : ℝ)
  (h1 : train_ride = 5)
  (h2 : first_layover = 1.5)
  (h3 : bus_ride = 4)
  (h4 : second_layover = 0.5)
  (h5 : first_flight = 6)
  (h6 : third_layover = 2)
  (h7 : fourth_layover = 3)
  (h8 : car_drive = 3.5)
  (h9 : first_boat_ride = 1.5)
  (h10 : fifth_layover = 0.75)
  (h11 : final_walk = 1.25) :
  train_ride + first_layover + bus_ride + second_layover + first_flight + 
  third_layover + (3 * bus_ride) + fourth_layover + car_drive + 
  first_boat_ride + fifth_layover + (2 * first_boat_ride - 0.5) + final_walk = 44 := by
  sorry


end NUMINAMATH_CALUDE_toms_journey_to_virgo_l4068_406847


namespace NUMINAMATH_CALUDE_five_digit_divisibility_count_l4068_406816

/-- The count of 5-digit numbers with a specific divisibility property -/
theorem five_digit_divisibility_count : 
  (Finset.filter 
    (fun n : ℕ => 
      10000 ≤ n ∧ n ≤ 99999 ∧ 
      (n / 50 + n % 50) % 7 = 0)
    (Finset.range 100000)).card = 14400 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_count_l4068_406816


namespace NUMINAMATH_CALUDE_isabellas_hair_length_l4068_406803

theorem isabellas_hair_length (current_length cut_length : ℕ) 
  (h1 : current_length = 9)
  (h2 : cut_length = 9) :
  current_length + cut_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_length_l4068_406803


namespace NUMINAMATH_CALUDE_unique_n_value_l4068_406870

theorem unique_n_value (n : ℤ) 
  (h1 : 50 ≤ n ∧ n ≤ 120) 
  (h2 : ∃ k : ℤ, n = 5 * k) 
  (h3 : n % 6 = 3) 
  (h4 : n % 7 = 4) : 
  n = 165 := by sorry

end NUMINAMATH_CALUDE_unique_n_value_l4068_406870


namespace NUMINAMATH_CALUDE_prob_two_empty_given_at_least_one_empty_l4068_406885

/-- The number of balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 4

/-- The number of ways to place balls into boxes with exactly one empty box -/
def ways_one_empty : ℕ := 144

/-- The number of ways to place balls into boxes with exactly two empty boxes -/
def ways_two_empty : ℕ := 84

/-- The number of ways to place balls into boxes with exactly three empty boxes -/
def ways_three_empty : ℕ := 4

/-- The probability of exactly two boxes being empty given at least one box is empty -/
theorem prob_two_empty_given_at_least_one_empty :
  (ways_two_empty : ℚ) / (ways_one_empty + ways_two_empty + ways_three_empty) = 21 / 58 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_empty_given_at_least_one_empty_l4068_406885


namespace NUMINAMATH_CALUDE_hex_pattern_theorem_l4068_406863

/-- Represents a hexagonal tile pattern -/
structure HexPattern where
  blue_tiles : ℕ
  green_tiles : ℕ
  red_tiles : ℕ

/-- Creates a new pattern by adding green and red tiles -/
def add_border (initial : HexPattern) (green_layers : ℕ) : HexPattern :=
  let new_green := initial.green_tiles + green_layers * 24
  let new_red := 12
  { blue_tiles := initial.blue_tiles,
    green_tiles := new_green,
    red_tiles := new_red }

theorem hex_pattern_theorem (initial : HexPattern) :
  initial.blue_tiles = 20 →
  initial.green_tiles = 9 →
  let new_pattern := add_border initial 2
  new_pattern.red_tiles = 12 ∧
  new_pattern.green_tiles + new_pattern.red_tiles - new_pattern.blue_tiles = 25 := by
  sorry

end NUMINAMATH_CALUDE_hex_pattern_theorem_l4068_406863


namespace NUMINAMATH_CALUDE_orange_cost_theorem_l4068_406889

/-- The rate at which oranges are sold in dollars per kilogram -/
def orange_rate : ℚ := 5 / 3

/-- The amount of oranges in kilograms to be purchased -/
def amount_to_buy : ℚ := 12

/-- The cost of buying a given amount of oranges in dollars -/
def cost (kg : ℚ) : ℚ := kg * orange_rate

theorem orange_cost_theorem : cost amount_to_buy = 20 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_theorem_l4068_406889


namespace NUMINAMATH_CALUDE_saree_price_calculation_l4068_406899

theorem saree_price_calculation (final_price : ℝ) 
  (h : final_price = 378.675) : ∃ (original_price : ℝ), 
  original_price * 0.85 * 0.90 = final_price ∧ 
  original_price = 495 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l4068_406899


namespace NUMINAMATH_CALUDE_tangent_slope_at_x_4_l4068_406817

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 8

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 5

-- Theorem statement
theorem tangent_slope_at_x_4 :
  f' 4 = 29 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_x_4_l4068_406817


namespace NUMINAMATH_CALUDE_f_2023_equals_2_l4068_406813

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2023_equals_2 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_symmetry : ∀ x, f (1 + x) = f (1 - x))
  (h_interval : ∀ x ∈ Set.Icc 0 1, f x = 2^x) :
  f 2023 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2023_equals_2_l4068_406813


namespace NUMINAMATH_CALUDE_vector_problem_l4068_406871

def a : ℝ × ℝ := (3, -1)

theorem vector_problem (b : ℝ × ℝ) (x : ℝ) :
  let c := λ x => x • a + (1 - x) • b
  let dot_product := λ u v : ℝ × ℝ => u.1 * v.1 + u.2 * v.2
  dot_product a b = -5 ∧ ‖b‖ = Real.sqrt 5 →
  (dot_product a (c x) = 0 → x = 1/3) ∧
  (∃ x₀, ∀ x, ‖c x₀‖ ≤ ‖c x‖ ∧ ‖c x₀‖ = 1) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l4068_406871
