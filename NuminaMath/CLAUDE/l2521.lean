import Mathlib

namespace NUMINAMATH_CALUDE_mulch_price_per_pound_l2521_252182

/-- Given the cost of mulch in tons, calculate the price per pound -/
theorem mulch_price_per_pound (cost : ℝ) (tons : ℝ) (pounds_per_ton : ℝ) : 
  cost = 15000 → tons = 3 → pounds_per_ton = 2000 →
  cost / (tons * pounds_per_ton) = 2.5 := by sorry

end NUMINAMATH_CALUDE_mulch_price_per_pound_l2521_252182


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l2521_252179

theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 30 * π → π * r^2 = 225 * π :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l2521_252179


namespace NUMINAMATH_CALUDE_seashells_sum_l2521_252131

/-- The number of seashells Joan found on the beach -/
def total_seashells : ℕ := 70

/-- The number of seashells Joan gave to Sam -/
def seashells_given : ℕ := 43

/-- The number of seashells Joan has left -/
def seashells_left : ℕ := 27

/-- Theorem stating that the total number of seashells is the sum of those given away and those left -/
theorem seashells_sum : total_seashells = seashells_given + seashells_left := by
  sorry

end NUMINAMATH_CALUDE_seashells_sum_l2521_252131


namespace NUMINAMATH_CALUDE_three_distinct_roots_l2521_252153

open Real

theorem three_distinct_roots (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    (abs (x₁^3 - a^3) = x₁ - a) ∧
    (abs (x₂^3 - a^3) = x₂ - a) ∧
    (abs (x₃^3 - a^3) = x₃ - a)) ↔ 
  (-2 / sqrt 3 < a ∧ a < -1 / sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_three_distinct_roots_l2521_252153


namespace NUMINAMATH_CALUDE_probability_second_class_correct_l2521_252104

/-- The probability of selecting at least one second-class product when
    randomly choosing 4 products from a batch of 100 products containing
    90 first-class and 10 second-class products. -/
def probability_second_class (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ) : ℚ :=
  1 - (first_class / total) * ((first_class - 1) / (total - 1)) *
      ((first_class - 2) / (total - 2)) * ((first_class - 3) / (total - 3))

/-- The theorem stating that the probability of selecting at least one
    second-class product is correct for the given conditions. -/
theorem probability_second_class_correct :
  probability_second_class 100 90 10 4 = 1 - (90/100 * 89/99 * 88/98 * 87/97) :=
by sorry

end NUMINAMATH_CALUDE_probability_second_class_correct_l2521_252104


namespace NUMINAMATH_CALUDE_students_with_both_pets_l2521_252136

theorem students_with_both_pets (total : ℕ) (dog : ℕ) (cat : ℕ) (no_pet : ℕ) 
  (h_total : total = 50)
  (h_dog : dog = 30)
  (h_cat : cat = 35)
  (h_no_pet : no_pet = 3)
  (h_at_least_one : ∀ s, s ∈ Finset.range total → 
    (s ∈ Finset.range dog ∨ s ∈ Finset.range cat ∨ s ∈ Finset.range no_pet)) :
  Finset.card (Finset.range dog ∩ Finset.range cat) = 18 := by
  sorry

end NUMINAMATH_CALUDE_students_with_both_pets_l2521_252136


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2521_252100

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 125)
  (h2 : divisor = 15)
  (h3 : quotient = 8)
  (h4 : dividend = divisor * quotient + remainder) :
  remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2521_252100


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2521_252171

theorem trigonometric_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + 2 * y) ^ 2 - 2 * Real.sin x * Real.sin (2 * y) * Real.cos (x + 2 * y) =
  2 * Real.sin x ^ 2 - Real.sin x ^ 2 * Real.cos y ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2521_252171


namespace NUMINAMATH_CALUDE_sum_of_xy_l2521_252158

theorem sum_of_xy (x y : ℕ+) (h : (2 * x - 5) * (2 * y - 5) = 25) :
  x + y = 18 ∨ x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l2521_252158


namespace NUMINAMATH_CALUDE_max_cookies_bound_l2521_252127

def num_jars : Nat := 2023

/-- Represents the state of cookie jars -/
def JarState := Fin num_jars → Nat

/-- Elmo's action of adding cookies to two distinct jars -/
def elmo_action (state : JarState) : JarState := sorry

/-- Cookie Monster's action of eating cookies from the jar with the most cookies -/
def monster_action (state : JarState) : JarState := sorry

/-- One complete cycle of Elmo's and Cookie Monster's actions -/
def cycle (state : JarState) : JarState := monster_action (elmo_action state)

/-- The maximum number of cookies in any jar -/
def max_cookies (state : JarState) : Nat :=
  Finset.sup (Finset.univ : Finset (Fin num_jars)) (fun i => state i)

theorem max_cookies_bound (initial_state : JarState) :
  ∀ n : Nat, max_cookies ((cycle^[n]) initial_state) ≤ 12 := by sorry

end NUMINAMATH_CALUDE_max_cookies_bound_l2521_252127


namespace NUMINAMATH_CALUDE_tan_function_property_l2521_252192

/-- Given a function f(x) = a tan(bx) where a and b are positive constants,
    if f has roots at ±π/4 and passes through (π/8, 1), then a · b = 2 -/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.tan (b * x) = 0 ↔ x = π/4 ∨ x = -π/4) →
  a * Real.tan (b * π/8) = 1 →
  a * b = 2 := by sorry

end NUMINAMATH_CALUDE_tan_function_property_l2521_252192


namespace NUMINAMATH_CALUDE_brick_length_proof_l2521_252196

theorem brick_length_proof (w h A : ℝ) (hw : w = 4) (hh : h = 3) (hA : A = 164) :
  let l := (A - 2 * w * h) / (2 * (w + h))
  l = 10 := by sorry

end NUMINAMATH_CALUDE_brick_length_proof_l2521_252196


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l2521_252110

theorem alpha_beta_sum (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α = 1) 
  (hβ : β^3 - 3*β^2 + 5*β = 5) : 
  α + β = 2 := by sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l2521_252110


namespace NUMINAMATH_CALUDE_pencil_final_price_l2521_252173

/-- Given a pencil with an original cost and a discount, calculate the final price. -/
theorem pencil_final_price (original_cost discount : ℚ) 
  (h1 : original_cost = 4)
  (h2 : discount = 63 / 100) :
  original_cost - discount = 337 / 100 := by
  sorry

end NUMINAMATH_CALUDE_pencil_final_price_l2521_252173


namespace NUMINAMATH_CALUDE_rectangle_side_length_l2521_252135

/-- Given three rectangles with equal areas and integer sides, where one side is 29, prove that another side is 870 -/
theorem rectangle_side_length (a b k l : ℕ) : 
  let S := 29 * (a + b)
  a * k = S ∧ 
  b * l = S ∧ 
  k * l = 29 * (k + l) →
  k = 870 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l2521_252135


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_l2521_252150

/-- The volume of a cube with space diagonal 5√3 is 125 -/
theorem cube_volume_from_diagonal : 
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 5 * Real.sqrt 3 → s^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_l2521_252150


namespace NUMINAMATH_CALUDE_figure_area_theorem_l2521_252103

theorem figure_area_theorem (x : ℝ) : 
  let square1_area := (2 * x)^2
  let square2_area := (5 * x)^2
  let triangle_area := (1/2) * (2 * x) * (5 * x)
  square1_area + square2_area + triangle_area = 850 → x = 5 := by
sorry

end NUMINAMATH_CALUDE_figure_area_theorem_l2521_252103


namespace NUMINAMATH_CALUDE_chris_box_percentage_l2521_252124

theorem chris_box_percentage (k c : ℕ) (h : k = 2 * c / 3) : 
  (c : ℚ) / ((k : ℚ) + c) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_chris_box_percentage_l2521_252124


namespace NUMINAMATH_CALUDE_triangle_perpendicular_theorem_l2521_252181

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the foot of a perpendicular
def perpFoot (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Main theorem
theorem triangle_perpendicular_theorem (ABC : Triangle) :
  let A := ABC.A
  let B := ABC.B
  let C := ABC.C
  let D := perpFoot A B C
  length A B = 12 →
  length A C = 20 →
  (length B D) / (length C D) = 3 / 4 →
  length A D = 36 * Real.sqrt 14 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perpendicular_theorem_l2521_252181


namespace NUMINAMATH_CALUDE_price_large_bottle_correct_l2521_252189

/-- The price of a large bottle, given the following conditions:
  * 1365 large bottles were purchased at this price
  * 720 small bottles were purchased at $1.42 each
  * The average price of all bottles was approximately $1.73
-/
def price_large_bottle : ℝ := 1.89

theorem price_large_bottle_correct : 
  let num_large : ℕ := 1365
  let num_small : ℕ := 720
  let price_small : ℝ := 1.42
  let avg_price : ℝ := 1.73
  let total_bottles : ℕ := num_large + num_small
  let total_cost : ℝ := (num_large : ℝ) * price_large_bottle + (num_small : ℝ) * price_small
  abs (total_cost / (total_bottles : ℝ) - avg_price) < 0.01 ∧ 
  abs (price_large_bottle - 1.89) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_price_large_bottle_correct_l2521_252189


namespace NUMINAMATH_CALUDE_function_equality_implies_sum_l2521_252121

theorem function_equality_implies_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x - 2) = 4 * x^2 + 9 * x + 5) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 68 := by
sorry

end NUMINAMATH_CALUDE_function_equality_implies_sum_l2521_252121


namespace NUMINAMATH_CALUDE_mrs_blue_orchard_yield_l2521_252159

/-- Calculates the expected apple yield from a rectangular orchard -/
def expected_apple_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let length_ft := length_steps * step_length
  let width_ft := width_steps * step_length
  let area_sqft := length_ft * width_ft
  area_sqft * yield_per_sqft

/-- Theorem stating the expected apple yield for Mrs. Blue's orchard -/
theorem mrs_blue_orchard_yield :
  expected_apple_yield 25 20 2.5 0.75 = 2343.75 := by
  sorry

end NUMINAMATH_CALUDE_mrs_blue_orchard_yield_l2521_252159


namespace NUMINAMATH_CALUDE_jaime_savings_time_l2521_252178

/-- Calculates the number of weeks needed to save a target amount given weekly savings and bi-weekly expenses -/
def weeksToSave (weeklySavings : ℚ) (biWeeklyExpense : ℚ) (targetAmount : ℚ) : ℚ :=
  let netBiWeeklySavings := 2 * weeklySavings - biWeeklyExpense
  let netWeeklySavings := netBiWeeklySavings / 2
  targetAmount / netWeeklySavings

/-- Proves that it takes 5 weeks to save $135 with $50 weekly savings and $46 bi-weekly expense -/
theorem jaime_savings_time : weeksToSave 50 46 135 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jaime_savings_time_l2521_252178


namespace NUMINAMATH_CALUDE_days_without_calls_is_244_l2521_252157

/-- The number of days in the year --/
def year_days : ℕ := 365

/-- The intervals at which the nephews call --/
def call_intervals : List ℕ := [4, 6, 8]

/-- Calculate the number of days without calls --/
def days_without_calls (total_days : ℕ) (intervals : List ℕ) : ℕ :=
  total_days - (total_days / intervals.head! + total_days / intervals.tail.head! + total_days / intervals.tail.tail.head! -
    total_days / (intervals.head!.lcm intervals.tail.head!) - 
    total_days / (intervals.head!.lcm intervals.tail.tail.head!) - 
    total_days / (intervals.tail.head!.lcm intervals.tail.tail.head!) +
    total_days / (intervals.head!.lcm intervals.tail.head!).lcm intervals.tail.tail.head!)

theorem days_without_calls_is_244 :
  days_without_calls year_days call_intervals = 244 := by
  sorry

end NUMINAMATH_CALUDE_days_without_calls_is_244_l2521_252157


namespace NUMINAMATH_CALUDE_school_gender_ratio_l2521_252199

/-- Given a school with a 5:4 ratio of boys to girls and 1500 boys, prove there are 1200 girls -/
theorem school_gender_ratio (num_boys : ℕ) (num_girls : ℕ) : 
  num_boys = 1500 →
  (5 : ℚ) / 4 = num_boys / num_girls →
  num_girls = 1200 := by
sorry

end NUMINAMATH_CALUDE_school_gender_ratio_l2521_252199


namespace NUMINAMATH_CALUDE_january_salary_l2521_252141

/-- Given the average salaries for two sets of four months and the salary for May,
    calculate the salary for January. -/
theorem january_salary
  (avg_jan_to_apr : ℝ)
  (avg_feb_to_may : ℝ)
  (may_salary : ℝ)
  (h1 : avg_jan_to_apr = 8000)
  (h2 : avg_feb_to_may = 8500)
  (h3 : may_salary = 6500) :
  ∃ (jan feb mar apr : ℝ),
    (jan + feb + mar + apr) / 4 = avg_jan_to_apr ∧
    (feb + mar + apr + may_salary) / 4 = avg_feb_to_may ∧
    jan = 4500 := by
  sorry

#check january_salary

end NUMINAMATH_CALUDE_january_salary_l2521_252141


namespace NUMINAMATH_CALUDE_modulus_of_z_l2521_252198

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2521_252198


namespace NUMINAMATH_CALUDE_digit_sum_problem_l2521_252183

theorem digit_sum_problem :
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
    a + c = 10 →
    b + c + 1 = 10 →
    a + d + 1 = 11 →
    1000 * a + 100 * b + 10 * c + d + 100 * c + 10 * a = 1100 →
    a + b + c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l2521_252183


namespace NUMINAMATH_CALUDE_fourth_grade_students_l2521_252101

theorem fourth_grade_students (initial_students : ℝ) (left_students : ℝ) (transferred_students : ℝ) :
  initial_students = 42.0 →
  left_students = 4.0 →
  transferred_students = 10.0 →
  initial_students - left_students - transferred_students = 28.0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l2521_252101


namespace NUMINAMATH_CALUDE_rudys_running_time_l2521_252155

/-- Calculates the total running time for Rudy given two separate runs -/
def totalRunningTime (distance1 : ℝ) (rate1 : ℝ) (distance2 : ℝ) (rate2 : ℝ) : ℝ :=
  distance1 * rate1 + distance2 * rate2

/-- Proves that Rudy's total running time is 88 minutes -/
theorem rudys_running_time :
  totalRunningTime 5 10 4 9.5 = 88 := by
  sorry

end NUMINAMATH_CALUDE_rudys_running_time_l2521_252155


namespace NUMINAMATH_CALUDE_parabola_line_intersection_property_l2521_252115

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop :=
  ∃ (k : ℝ), y = k * (x - 1) ∨ x = 1

-- Theorem statement
theorem parabola_line_intersection_property 
  (x₁ y₁ x₂ y₂ : ℝ) (h_distinct : (x₁, y₁) ≠ (x₂, y₂)) :
  (∀ x y, line_through_points x₁ y₁ x₂ y₂ x y → line_through_focus x y → 
    parabola x₁ y₁ → parabola x₂ y₂ → x₁ * x₂ = 1) ∧
  (∃ x₁' y₁' x₂' y₂', (x₁', y₁') ≠ (x₂', y₂') ∧
    parabola x₁' y₁' ∧ parabola x₂' y₂' ∧ x₁' * x₂' = 1 ∧
    ¬(∀ x y, line_through_points x₁' y₁' x₂' y₂' x y → line_through_focus x y)) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_property_l2521_252115


namespace NUMINAMATH_CALUDE_fir_trees_count_l2521_252125

theorem fir_trees_count :
  ∀ n : ℕ,
  n < 25 →
  n % 11 = 0 →
  n = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_fir_trees_count_l2521_252125


namespace NUMINAMATH_CALUDE_equation_solution_l2521_252163

theorem equation_solution :
  let x : ℚ := -21/20
  (Real.sqrt (2 * x + 7) / Real.sqrt (4 * x + 7) = Real.sqrt 7 / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2521_252163


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l2521_252109

/-- Represents the problem from "The Mathematical Classic of Sunzi" --/
theorem sunzi_wood_measurement (x y : ℝ) :
  y = x + 4.5 ∧ 0.5 * y = x - 1 →
  (y - x = 4.5 ∧ y / 2 - x = 1) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l2521_252109


namespace NUMINAMATH_CALUDE_martha_jackets_bought_l2521_252190

theorem martha_jackets_bought (J : ℕ) : 
  (J + J / 2 : ℕ) + (9 + 9 / 3 : ℕ) = 18 → J = 4 :=
by sorry

end NUMINAMATH_CALUDE_martha_jackets_bought_l2521_252190


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l2521_252162

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0

-- Define proposition q
def q : Prop := ∀ m : ℝ, m > -2 → ∃ x : ℝ, x^2 + 2*x - m = 0

-- Theorem statement
theorem p_or_q_is_true : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l2521_252162


namespace NUMINAMATH_CALUDE_car_speed_problem_l2521_252133

/-- Given a car traveling for two hours, prove that if its speed in the second hour
    is 45 km/h and its average speed over the two hours is 55 km/h, then its speed
    in the first hour must be 65 km/h. -/
theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ)
    (h1 : speed_second_hour = 45)
    (h2 : average_speed = 55) :
    let speed_first_hour := 2 * average_speed - speed_second_hour
    speed_first_hour = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2521_252133


namespace NUMINAMATH_CALUDE_parallel_vectors_dot_product_l2521_252146

def a (m n : ℝ) : Fin 3 → ℝ := ![1, 3*m - 1, n - 2]
def b (m n : ℝ) : Fin 3 → ℝ := ![2, 3*m + 1, 3*n - 4]

def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (i : Fin 3), u i = k * v i

def dot_product (u v : Fin 3 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1) + (u 2) * (v 2)

theorem parallel_vectors_dot_product (m n : ℝ) :
  parallel (a m n) (b m n) → dot_product (a m n) (b m n) = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_dot_product_l2521_252146


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l2521_252191

/-- The y-coordinate of the point on the y-axis that is equidistant from A(-2, 0) and B(-1, 4) -/
theorem equidistant_point_y_coordinate :
  ∃ y : ℝ, ((-2 : ℝ) - 0)^2 + (0 - y)^2 = ((-1 : ℝ) - 0)^2 + (4 - y)^2 ∧ y = 13/8 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l2521_252191


namespace NUMINAMATH_CALUDE_largest_divisible_n_l2521_252113

theorem largest_divisible_n : 
  ∀ n : ℕ, n > 882 → ¬(n + 9 ∣ n^3 + 99) ∧ (882 + 9 ∣ 882^3 + 99) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l2521_252113


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2521_252138

theorem sqrt_equation_solution (x : ℝ) : 
  (Real.sqrt (4 * x + 6) / Real.sqrt (8 * x + 12) = Real.sqrt 2 / 2) → x ≥ -3/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2521_252138


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2521_252145

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  4 * Real.pi * radius^2 = 50 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2521_252145


namespace NUMINAMATH_CALUDE_sum_of_four_digit_odd_and_multiples_of_five_l2521_252119

/-- The number of four-digit odd numbers -/
def A : ℕ := 4500

/-- The number of four-digit multiples of 5 -/
def B : ℕ := 1800

/-- The sum of four-digit odd numbers and four-digit multiples of 5 is 6300 -/
theorem sum_of_four_digit_odd_and_multiples_of_five : A + B = 6300 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_odd_and_multiples_of_five_l2521_252119


namespace NUMINAMATH_CALUDE_linearly_dependent_implies_k_equals_six_l2521_252144

/-- Two vectors in ℝ² are linearly dependent if there exist non-zero scalars such that their linear combination is zero. -/
def linearlyDependent (v w : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ a • v + b • w = (0, 0)

/-- The theorem states that if the vectors (2, 3) and (4, k) are linearly dependent, then k must equal 6. -/
theorem linearly_dependent_implies_k_equals_six :
  linearlyDependent (2, 3) (4, k) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_linearly_dependent_implies_k_equals_six_l2521_252144


namespace NUMINAMATH_CALUDE_right_triangle_probability_l2521_252137

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Finset (ℕ × ℕ)
  regular : vertices.card = 8
  -- Additional properties of a regular octagon could be added here

/-- A triangle formed by three vertices of a regular octagon -/
structure OctagonTriangle (octagon : RegularOctagon) where
  vertices : Finset (ℕ × ℕ)
  subset : vertices ⊆ octagon.vertices
  three_points : vertices.card = 3

/-- Predicate to determine if a triangle is right-angled -/
def is_right_triangle (triangle : OctagonTriangle octagon) : Prop :=
  sorry -- Definition of a right triangle in terms of the octagon's geometry

/-- The set of all possible triangles from an octagon -/
def all_triangles (octagon : RegularOctagon) : Finset (OctagonTriangle octagon) :=
  sorry

/-- The set of right triangles from an octagon -/
def right_triangles (octagon : RegularOctagon) : Finset (OctagonTriangle octagon) :=
  sorry

/-- The main theorem -/
theorem right_triangle_probability (octagon : RegularOctagon) :
  (right_triangles octagon).card / (all_triangles octagon).card = 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_probability_l2521_252137


namespace NUMINAMATH_CALUDE_qin_jiushao_area_formula_l2521_252174

theorem qin_jiushao_area_formula (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  let S := Real.sqrt ((c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2) / 4)
  a = 25 → b = 24 → c = 14 → S = (105 * Real.sqrt 39) / 4 :=
by sorry

end NUMINAMATH_CALUDE_qin_jiushao_area_formula_l2521_252174


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l2521_252170

theorem triangle_angle_inequality (a b c α β γ : Real) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < α ∧ 0 < β ∧ 0 < γ ∧
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  α + β + γ = π →
  π / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l2521_252170


namespace NUMINAMATH_CALUDE_cab_driver_income_l2521_252118

/-- Cab driver's income problem -/
theorem cab_driver_income 
  (income : Fin 5 → ℕ) 
  (h1 : income 0 = 600)
  (h2 : income 1 = 250)
  (h3 : income 2 = 450)
  (h4 : income 3 = 400)
  (h_avg : (income 0 + income 1 + income 2 + income 3 + income 4) / 5 = 500) :
  income 4 = 800 := by
sorry


end NUMINAMATH_CALUDE_cab_driver_income_l2521_252118


namespace NUMINAMATH_CALUDE_tim_sweets_multiple_of_four_l2521_252120

/-- The number of grape-flavored sweets Peter has -/
def peter_sweets : ℕ := 44

/-- The largest possible number of sweets in each tray without remainder -/
def tray_size : ℕ := 4

/-- The number of orange-flavored sweets Tim has -/
def tim_sweets : ℕ := sorry

theorem tim_sweets_multiple_of_four :
  ∃ k : ℕ, tim_sweets = k * tray_size ∧ peter_sweets % tray_size = 0 :=
by sorry

end NUMINAMATH_CALUDE_tim_sweets_multiple_of_four_l2521_252120


namespace NUMINAMATH_CALUDE_chord_line_equation_l2521_252176

/-- Given an ellipse and a point as the midpoint of a chord, find the equation of the line containing the chord -/
theorem chord_line_equation (x y : ℝ) :
  (x^2 / 36 + y^2 / 9 = 1) →  -- Ellipse equation
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 / 36 + y₁^2 / 9 = 1) ∧  -- Point (x₁, y₁) is on the ellipse
    (x₂^2 / 36 + y₂^2 / 9 = 1) ∧  -- Point (x₂, y₂) is on the ellipse
    ((x₁ + x₂) / 2 = 1) ∧  -- Midpoint x-coordinate is 1
    ((y₁ + y₂) / 2 = 1) →  -- Midpoint y-coordinate is 1
  ∃ (m : ℝ), m = -1/4 ∧ y - 1 = m * (x - 1) :=  -- Line equation
by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l2521_252176


namespace NUMINAMATH_CALUDE_match_rectangle_properties_l2521_252167

/-- Represents a rectangle made of matches -/
structure MatchRectangle where
  m : ℕ
  n : ℕ
  h : m > n

/-- Total number of matches used to construct the rectangle -/
def totalMatches (r : MatchRectangle) : ℕ :=
  2 * r.m * r.n + r.m + r.n

/-- Total number of possible rectangles in the figure -/
def totalRectangles (r : MatchRectangle) : ℚ :=
  (r.m * r.n * (r.m + 1) * (r.n + 1)) / 4

/-- Total number of possible squares in the figure -/
def totalSquares (r : MatchRectangle) : ℚ :=
  (r.n * (r.n + 1) * (3 * r.m - r.n + 1)) / 6

theorem match_rectangle_properties (r : MatchRectangle) :
  (totalMatches r = 2 * r.m * r.n + r.m + r.n) ∧
  (totalRectangles r = (r.m * r.n * (r.m + 1) * (r.n + 1)) / 4) ∧
  (totalSquares r = (r.n * (r.n + 1) * (3 * r.m - r.n + 1)) / 6) := by
  sorry

end NUMINAMATH_CALUDE_match_rectangle_properties_l2521_252167


namespace NUMINAMATH_CALUDE_sqrt_six_irrational_l2521_252140

theorem sqrt_six_irrational : Irrational (Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_irrational_l2521_252140


namespace NUMINAMATH_CALUDE_min_distance_point_to_line_l2521_252177

/-- The minimum distance between a point in the feasible region and a line -/
theorem min_distance_point_to_line :
  ∀ (x y : ℝ),
  (2 * x + y - 4 ≥ 0) →
  (x - y - 2 ≤ 0) →
  (y - 3 ≤ 0) →
  ∃ (x' y' : ℝ),
  (y' = -2 * x' + 2) →
  ∀ (x'' y'' : ℝ),
  (y'' = -2 * x'' + 2) →
  Real.sqrt ((x - x')^2 + (y - y')^2) ≥ (2 * Real.sqrt 5) / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_line_l2521_252177


namespace NUMINAMATH_CALUDE_express_1997_using_fours_l2521_252195

theorem express_1997_using_fours : 
  4 * 444 + 44 * 4 + 44 + 4 / 4 = 1997 :=
by sorry

end NUMINAMATH_CALUDE_express_1997_using_fours_l2521_252195


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2521_252197

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 - x - 4 ≥ 0) ↔ (x ≤ -1 ∨ x ≥ 4/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2521_252197


namespace NUMINAMATH_CALUDE_coin_division_problem_l2521_252186

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (n % 8 = 6) →
  (n % 7 = 5) →
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 9 = 0) :=
sorry

end NUMINAMATH_CALUDE_coin_division_problem_l2521_252186


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l2521_252194

theorem muffin_banana_price_ratio :
  ∀ (muffin_price banana_price : ℚ),
  (5 * muffin_price + 4 * banana_price = 20) →
  (3 * muffin_price + 18 * banana_price = 60) →
  muffin_price / banana_price = 13 / 4 := by
sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l2521_252194


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l2521_252168

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * l + 2 * w = 2 * Real.pi * r) : 
  (l * w) / (Real.pi * r ^ 2) = 18 / Real.pi ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l2521_252168


namespace NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l2521_252105

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x + 5 else 3 * a / x

theorem function_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) < 0) ↔ 0 < a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_function_decreasing_iff_a_in_range_l2521_252105


namespace NUMINAMATH_CALUDE_birthday_money_mean_l2521_252187

def aunt_gift : ℝ := 9
def uncle_gift : ℝ := 9
def friend_gifts : List ℝ := [22, 23, 22, 22]
def sister_gift : ℝ := 7

def total_amount : ℝ := aunt_gift + uncle_gift + sister_gift + friend_gifts.sum
def total_gifts : ℕ := 1 + 1 + friend_gifts.length + 1

theorem birthday_money_mean :
  total_amount / total_gifts = 16.29 := by sorry

end NUMINAMATH_CALUDE_birthday_money_mean_l2521_252187


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2521_252114

theorem line_intersects_circle (a : ℝ) (h : a ≥ 0) :
  ∃ (x y : ℝ), (a * x - y + Real.sqrt 2 * a = 0) ∧ (x^2 + y^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2521_252114


namespace NUMINAMATH_CALUDE_point_on_line_with_vector_condition_l2521_252164

/-- Given two points in a 2D plane and a third point satisfying certain conditions,
    prove that the third point has specific coordinates. -/
theorem point_on_line_with_vector_condition (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (1, 3) →
  P₂ = (4, -6) →
  (∃ t : ℝ, P = (1 - t) • P₁ + t • P₂) →  -- P is on the line P₁P₂
  (P.1 - P₁.1, P.2 - P₁.2) = 2 • (P₂.1 - P.1, P₂.2 - P.2) →  -- Vector condition
  P = (3, -3) := by
sorry

end NUMINAMATH_CALUDE_point_on_line_with_vector_condition_l2521_252164


namespace NUMINAMATH_CALUDE_problem_statement_l2521_252107

theorem problem_statement (a b x y : ℕ+) (P : ℕ) 
  (h1 : ∃ k : ℕ, (a * x + b * y : ℕ) = k * (a^2 + b^2))
  (h2 : P = x^2 + y^2)
  (h3 : Nat.Prime P) :
  (P ∣ (a^2 + b^2 : ℕ)) ∧ (a = x ∧ b = y) := by
sorry


end NUMINAMATH_CALUDE_problem_statement_l2521_252107


namespace NUMINAMATH_CALUDE_x_sixth_minus_six_x_when_three_l2521_252112

theorem x_sixth_minus_six_x_when_three :
  let x : ℝ := 3
  x^6 - 6*x = 711 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_minus_six_x_when_three_l2521_252112


namespace NUMINAMATH_CALUDE_regular_polygon_with_36_degree_exterior_angle_is_decagon_l2521_252151

/-- A regular polygon with exterior angles measuring 36° has 10 sides -/
theorem regular_polygon_with_36_degree_exterior_angle_is_decagon :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 0 →
    exterior_angle = 36 →
    (n : ℝ) * exterior_angle = 360 →
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_36_degree_exterior_angle_is_decagon_l2521_252151


namespace NUMINAMATH_CALUDE_f_properties_l2521_252116

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 - 2

-- State the theorem
theorem f_properties :
  -- Function f is decreasing on (-∞, 0) and (2, +∞), and increasing on (0, 2)
  (∀ x y, x < y ∧ ((x < 0 ∧ y < 0) ∨ (x > 2 ∧ y > 2)) → f x > f y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y) ∧
  -- Maximum value on [-2, 2] is 18
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≤ 18) ∧
  (∃ x, x ∈ Set.Icc (-2) 2 ∧ f x = 18) ∧
  -- Minimum value on [-2, 2] is -2
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≥ -2) ∧
  (∃ x, x ∈ Set.Icc (-2) 2 ∧ f x = -2) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l2521_252116


namespace NUMINAMATH_CALUDE_max_triangle_area_l2521_252175

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define a chord passing through the right focus
def chord_through_right_focus (m : ℝ) (y : ℝ) : ℝ := m * y + 1

-- Define the area of triangle PF₁Q
def triangle_area (y₁ y₂ : ℝ) : ℝ := |y₁ - y₂|

-- Theorem statement
theorem max_triangle_area :
  ∃ (max_area : ℝ), max_area = 3 ∧
  ∀ (m : ℝ) (y₁ y₂ : ℝ),
    ellipse (chord_through_right_focus m y₁) y₁ →
    ellipse (chord_through_right_focus m y₂) y₂ →
    triangle_area y₁ y₂ ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l2521_252175


namespace NUMINAMATH_CALUDE_correct_robes_count_l2521_252130

/-- The number of robes a school already has for their choir. -/
def robes_already_have (total_singers : ℕ) (robe_cost : ℕ) (total_spend : ℕ) : ℕ :=
  total_singers - (total_spend / robe_cost)

/-- Theorem stating that the number of robes the school already has is correct. -/
theorem correct_robes_count :
  robes_already_have 30 2 36 = 12 := by sorry

end NUMINAMATH_CALUDE_correct_robes_count_l2521_252130


namespace NUMINAMATH_CALUDE_time_difference_to_halfway_point_l2521_252122

/-- Given that Danny can reach Steve's house in 29 minutes and Steve takes twice as long to reach Danny's house,
    prove that Steve takes 14.5 minutes longer than Danny to reach the halfway point between their houses. -/
theorem time_difference_to_halfway_point (danny_time : ℝ) (steve_time : ℝ) : 
  danny_time = 29 → steve_time = 2 * danny_time → steve_time / 2 - danny_time / 2 = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_to_halfway_point_l2521_252122


namespace NUMINAMATH_CALUDE_prob_same_color_is_zero_l2521_252147

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  green : Nat
  white : Nat
  blue : Nat
  red : Nat

/-- Calculates the total number of balls in the bag -/
def totalBalls (counts : BallCounts) : Nat :=
  counts.green + counts.white + counts.blue + counts.red

/-- Represents the number of balls to be drawn -/
def ballsToDraw : Nat := 5

/-- Calculates the probability of drawing all balls of the same color -/
def probSameColor (counts : BallCounts) : ℚ :=
  if counts.green ≥ ballsToDraw ∨ counts.white ≥ ballsToDraw ∨ 
     counts.blue ≥ ballsToDraw ∨ counts.red ≥ ballsToDraw
  then 1 / (totalBalls counts).choose ballsToDraw
  else 0

/-- Theorem: The probability of drawing 5 balls of the same color is 0 -/
theorem prob_same_color_is_zero : 
  probSameColor { green := 10, white := 9, blue := 7, red := 4 } = 0 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_zero_l2521_252147


namespace NUMINAMATH_CALUDE_simplify_expression_l2521_252117

theorem simplify_expression (x : ℝ) : 8*x + 15 - 3*x + 27 = 5*x + 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2521_252117


namespace NUMINAMATH_CALUDE_M_properties_l2521_252129

-- Define the set M
def M : Set (ℝ × ℝ) := {p | Real.sqrt 2 * p.1 - 1 < p.2 ∧ p.2 < Real.sqrt 2 * p.1}

-- Define what it means for a point to have integer coordinates
def hasIntegerCoordinates (p : ℝ × ℝ) : Prop := ∃ (i j : ℤ), p = (↑i, ↑j)

-- Statement of the theorem
theorem M_properties :
  Convex ℝ M ∧
  (∃ (S : Set (ℝ × ℝ)), S ⊆ M ∧ Set.Infinite S ∧ ∀ p ∈ S, hasIntegerCoordinates p) ∧
  ∀ (a b : ℝ), let L := {p : ℝ × ℝ | p.2 = a * p.1 + b}
    (∃ (S : Set (ℝ × ℝ)), S ⊆ (M ∩ L) ∧ Set.Finite S ∧
      ∀ p ∈ (M ∩ L), hasIntegerCoordinates p → p ∈ S) :=
by
  sorry

end NUMINAMATH_CALUDE_M_properties_l2521_252129


namespace NUMINAMATH_CALUDE_bobs_walking_rate_l2521_252165

/-- Proves that Bob's walking rate is 7 miles per hour given the conditions of the problem -/
theorem bobs_walking_rate 
  (total_distance : ℝ) 
  (yolanda_rate : ℝ) 
  (bob_distance : ℝ) 
  (head_start : ℝ) 
  (h1 : total_distance = 65) 
  (h2 : yolanda_rate = 5) 
  (h3 : bob_distance = 35) 
  (h4 : head_start = 1) : 
  (bob_distance / (total_distance - yolanda_rate * head_start - bob_distance) * yolanda_rate) = 7 :=
sorry

end NUMINAMATH_CALUDE_bobs_walking_rate_l2521_252165


namespace NUMINAMATH_CALUDE_special_point_properties_l2521_252102

/-- A point in the second quadrant with coordinate product -10 -/
def special_point : ℝ × ℝ := (-2, 5)

theorem special_point_properties :
  let (x, y) := special_point
  x < 0 ∧ y > 0 ∧ x * y = -10 := by
  sorry

end NUMINAMATH_CALUDE_special_point_properties_l2521_252102


namespace NUMINAMATH_CALUDE_four_level_pyramid_books_l2521_252166

def pyramid_books (levels : ℕ) (ratio : ℝ) (top_level_books : ℕ) : ℝ :=
  let rec sum_levels (n : ℕ) : ℝ :=
    if n = 0 then 0
    else (top_level_books : ℝ) * (ratio ^ (n - 1)) + sum_levels (n - 1)
  sum_levels levels

theorem four_level_pyramid_books :
  pyramid_books 4 (1 / 0.8) 64 = 369 := by sorry

end NUMINAMATH_CALUDE_four_level_pyramid_books_l2521_252166


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2521_252143

theorem ellipse_m_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2/m + y^2/(2*m-1) = 1 ∧ 
   ∃ (a b : ℝ), a > b ∧ a^2 = m ∧ b^2 = 2*m-1) ↔ 
  (1/2 < m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2521_252143


namespace NUMINAMATH_CALUDE_exists_x_satisfying_conditions_l2521_252149

theorem exists_x_satisfying_conditions : ∃ x : ℝ,
  ({1, 3, x^2 - 2*x} : Set ℝ) = {1, 3, 0} ∧
  ({1, |2*x - 1|} : Set ℝ) = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_exists_x_satisfying_conditions_l2521_252149


namespace NUMINAMATH_CALUDE_tan_alpha_minus_2beta_l2521_252172

theorem tan_alpha_minus_2beta (α β : Real) 
  (h1 : Real.tan (α - β) = 2/5)
  (h2 : Real.tan β = 1/2) :
  Real.tan (α - 2*β) = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_2beta_l2521_252172


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_l2521_252193

/-- Represents an unfair eight-sided die -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  prob_eight : ℚ
  /-- The probability of rolling any number from 1 to 7 -/
  prob_others : ℚ
  /-- The sum of all probabilities is 1 -/
  prob_sum : prob_eight + 7 * prob_others = 1
  /-- The probability of rolling an 8 is 3/8 -/
  eight_is_three_eighths : prob_eight = 3 / 8

/-- Calculates the expected value of a roll of the unfair die -/
def expected_value (d : UnfairDie) : ℚ :=
  (d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7)) + (d.prob_eight * 8)

/-- Theorem stating that the expected value of the unfair die is 77/14 -/
theorem unfair_die_expected_value (d : UnfairDie) : expected_value d = 77 / 14 := by
  sorry

end NUMINAMATH_CALUDE_unfair_die_expected_value_l2521_252193


namespace NUMINAMATH_CALUDE_point_A_coordinates_l2521_252132

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally and vertically -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem point_A_coordinates :
  ∀ (x y : ℝ),
  let A : Point := ⟨2*x + y, x - 2*y⟩
  let B : Point := translate A 1 (-4)
  B = ⟨x - y, y⟩ →
  A = ⟨1, 3⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_point_A_coordinates_l2521_252132


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_l2521_252139

theorem smallest_sum_of_factors (a b c d : ℕ+) 
  (h : a * b * c * d = Nat.factorial 10) : 
  a + b + c + d ≥ 175 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_l2521_252139


namespace NUMINAMATH_CALUDE_log_inequality_range_l2521_252185

theorem log_inequality_range (a : ℝ) : 
  (a > 0 ∧ ∀ x : ℝ, 0 < x ∧ x ≤ 1 → 4 * x < Real.log x / Real.log a) ↔ 
  (0 < a ∧ a < 1) := by
sorry

end NUMINAMATH_CALUDE_log_inequality_range_l2521_252185


namespace NUMINAMATH_CALUDE_total_cakes_eaten_l2521_252128

def monday_cakes : ℕ := 6
def friday_cakes : ℕ := 9
def saturday_cakes : ℕ := 3 * monday_cakes

theorem total_cakes_eaten : monday_cakes + friday_cakes + saturday_cakes = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_cakes_eaten_l2521_252128


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_l2521_252111

/-- A function f(x) = -x³ + ax² - x - 1 is monotonic on ℝ iff a ∈ [-√3, √3] -/
theorem monotonic_cubic_function (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ 
  a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_l2521_252111


namespace NUMINAMATH_CALUDE_angle_minus_510_in_third_quadrant_l2521_252126

-- Define the function to convert an angle to its equivalent within 0° to 360°
def convertAngle (angle : Int) : Int :=
  angle % 360

-- Define the function to determine the quadrant of an angle
def getQuadrant (angle : Int) : Nat :=
  let convertedAngle := convertAngle angle
  if 0 ≤ convertedAngle ∧ convertedAngle < 90 then 1
  else if 90 ≤ convertedAngle ∧ convertedAngle < 180 then 2
  else if 180 ≤ convertedAngle ∧ convertedAngle < 270 then 3
  else 4

-- Theorem statement
theorem angle_minus_510_in_third_quadrant :
  getQuadrant (-510) = 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_minus_510_in_third_quadrant_l2521_252126


namespace NUMINAMATH_CALUDE_max_sum_with_constraint_l2521_252108

theorem max_sum_with_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 2) :
  a + b ≤ 3/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_constraint_l2521_252108


namespace NUMINAMATH_CALUDE_inequality_solution_case1_inequality_solution_case2_l2521_252156

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := a * x^2 + (1 - a) * x - 1 > 0

-- Define the solution set for case 1
def solution_set_case1 : Set ℝ := {x | x < -1/2 ∨ x > 1}

-- Define the solution set for case 2
def solution_set_case2 (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 1}
  else if a > 0 then {x | x < -1/a ∨ x > 1}
  else {x | 1 < x ∧ x < -1/a}

-- Theorem for case 1
theorem inequality_solution_case1 :
  ∀ x, inequality 2 x ↔ x ∈ solution_set_case1 :=
sorry

-- Theorem for case 2
theorem inequality_solution_case2 :
  ∀ a, a > -1 → ∀ x, inequality a x ↔ x ∈ solution_set_case2 a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_case1_inequality_solution_case2_l2521_252156


namespace NUMINAMATH_CALUDE_course_size_is_400_l2521_252106

/-- Proves that the total number of students in a course is 400, given the distribution of grades --/
theorem course_size_is_400 (T : ℕ) 
  (grade_A : ℕ := T / 5)
  (grade_B : ℕ := T / 4)
  (grade_C : ℕ := T / 2)
  (grade_D : ℕ := 20)
  (total_sum : T = grade_A + grade_B + grade_C + grade_D) : T = 400 := by
  sorry

end NUMINAMATH_CALUDE_course_size_is_400_l2521_252106


namespace NUMINAMATH_CALUDE_books_ratio_proof_l2521_252142

def books_problem (initial_books : ℕ) (rebecca_books : ℕ) (remaining_books : ℕ) : Prop :=
  let mara_books := initial_books - remaining_books - rebecca_books
  mara_books / rebecca_books = 3

theorem books_ratio_proof (initial_books : ℕ) (rebecca_books : ℕ) (remaining_books : ℕ)
  (h1 : initial_books = 220)
  (h2 : rebecca_books = 40)
  (h3 : remaining_books = 60) :
  books_problem initial_books rebecca_books remaining_books :=
by
  sorry

end NUMINAMATH_CALUDE_books_ratio_proof_l2521_252142


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2521_252152

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / (2 - x) ≥ 0 ↔ Real.log (x - 2) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2521_252152


namespace NUMINAMATH_CALUDE_candy_bar_cost_after_tax_l2521_252161

-- Define the initial amount Peter has
def initial_amount : ℝ := 10

-- Define the cost per ounce of soda
def soda_cost_per_ounce : ℝ := 0.25

-- Define the number of ounces of soda bought
def soda_ounces : ℝ := 16

-- Define the original price of chips
def chips_original_price : ℝ := 2.50

-- Define the discount rate for chips
def chips_discount_rate : ℝ := 0.1

-- Define the price of the candy bar
def candy_bar_price : ℝ := 1.25

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.08

-- Define the function to calculate the discounted price of chips
def discounted_chips_price : ℝ := chips_original_price * (1 - chips_discount_rate)

-- Define the function to calculate the total cost before tax
def total_cost_before_tax : ℝ := soda_cost_per_ounce * soda_ounces + discounted_chips_price + candy_bar_price

-- Define the function to calculate the total cost after tax
def total_cost_after_tax : ℝ := total_cost_before_tax * (1 + sales_tax_rate)

-- Theorem: The cost of the candy bar after tax is $1.35
theorem candy_bar_cost_after_tax :
  candy_bar_price * (1 + sales_tax_rate) = 1.35 ∧ total_cost_after_tax = initial_amount :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_after_tax_l2521_252161


namespace NUMINAMATH_CALUDE_topsoil_cost_l2521_252184

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards : ℝ := 8

/-- The total cost of topsoil in dollars -/
def total_cost : ℝ := cost_per_cubic_foot * cubic_feet_per_cubic_yard * cubic_yards

theorem topsoil_cost : total_cost = 1728 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l2521_252184


namespace NUMINAMATH_CALUDE_percentage_problem_l2521_252154

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 100) : 1.2 * x = 600 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2521_252154


namespace NUMINAMATH_CALUDE_sequence_square_l2521_252188

theorem sequence_square (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 2 → a n = a (n - 1) + 2 * n - 1) →
  ∀ n : ℕ, n > 0 → a n = n^2 := by
sorry

end NUMINAMATH_CALUDE_sequence_square_l2521_252188


namespace NUMINAMATH_CALUDE_rosencrans_wins_iff_odd_l2521_252169

/-- Represents a chord-drawing game on a circle with n points. -/
structure ChordGame where
  n : ℕ
  h : n ≥ 5

/-- Represents the outcome of the game. -/
inductive Outcome
  | RosencransWins
  | GildensternWins

/-- Determines the winner of the chord game based on the number of points. -/
def ChordGame.winner (game : ChordGame) : Outcome :=
  if game.n % 2 = 1 then Outcome.RosencransWins else Outcome.GildensternWins

/-- Theorem stating that Rosencrans wins if and only if n is odd. -/
theorem rosencrans_wins_iff_odd (game : ChordGame) :
  game.winner = Outcome.RosencransWins ↔ Odd game.n :=
sorry

end NUMINAMATH_CALUDE_rosencrans_wins_iff_odd_l2521_252169


namespace NUMINAMATH_CALUDE_johnson_finley_class_difference_l2521_252180

theorem johnson_finley_class_difference (finley_class : ℕ) (johnson_class : ℕ) : 
  finley_class = 24 →
  johnson_class = 22 →
  johnson_class > finley_class / 2 →
  johnson_class - finley_class / 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_johnson_finley_class_difference_l2521_252180


namespace NUMINAMATH_CALUDE_sum_a_d_l2521_252123

theorem sum_a_d (a b c d : ℤ) 
  (eq1 : a + b = 14) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_l2521_252123


namespace NUMINAMATH_CALUDE_line_passes_through_other_lattice_points_l2521_252148

theorem line_passes_through_other_lattice_points :
  ∃ (x y : ℤ), x ≠ 0 ∧ x ≠ 5 ∧ y ≠ 0 ∧ y ≠ 3 ∧ 5 * y = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_other_lattice_points_l2521_252148


namespace NUMINAMATH_CALUDE_final_position_theorem_l2521_252160

/-- Represents the position of the letter L --/
inductive LPosition
  | PosXPosY  -- Base along positive x-axis, stem along positive y-axis
  | NegXPosY  -- Base along negative x-axis, stem along positive y-axis
  | PosXNegY  -- Base along positive x-axis, stem along negative y-axis
  | NegXNegY  -- Base along negative x-axis, stem along negative y-axis

/-- Represents the transformations --/
inductive Transformation
  | RotateClockwise180
  | ReflectXAxis
  | RotateHalfTurn
  | ReflectYAxis

/-- Applies a single transformation to a given position --/
def applyTransformation (pos : LPosition) (t : Transformation) : LPosition :=
  sorry

/-- Applies a sequence of transformations to a given position --/
def applyTransformations (pos : LPosition) (ts : List Transformation) : LPosition :=
  sorry

theorem final_position_theorem :
  let initialPos := LPosition.PosXPosY
  let transformations := [
    Transformation.RotateClockwise180,
    Transformation.ReflectXAxis,
    Transformation.RotateHalfTurn,
    Transformation.ReflectYAxis
  ]
  applyTransformations initialPos transformations = LPosition.NegXNegY :=
sorry

end NUMINAMATH_CALUDE_final_position_theorem_l2521_252160


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2521_252134

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 3) ↔ x ≥ -3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2521_252134
