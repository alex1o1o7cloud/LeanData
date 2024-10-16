import Mathlib

namespace NUMINAMATH_CALUDE_same_distinct_prime_factors_l197_19749

-- Define the set of distinct prime factors
def distinct_prime_factors (n : ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ p ∣ n}

-- State the theorem
theorem same_distinct_prime_factors (k : ℕ) (h : k > 1) :
  let A := 2^k - 2
  let B := 2^k * A
  (distinct_prime_factors A = distinct_prime_factors B) ∧
  (distinct_prime_factors (A + 1) = distinct_prime_factors (B + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_same_distinct_prime_factors_l197_19749


namespace NUMINAMATH_CALUDE_johns_work_days_l197_19763

/-- Proves that John drives to work 5 days a week given his car's efficiency,
    distance to work, leisure travel, and weekly gas usage. -/
theorem johns_work_days (efficiency : ℝ) (distance_to_work : ℝ) (leisure_miles : ℝ) (gas_usage : ℝ)
    (h1 : efficiency = 30)
    (h2 : distance_to_work = 20)
    (h3 : leisure_miles = 40)
    (h4 : gas_usage = 8) :
    (gas_usage * efficiency - leisure_miles) / (2 * distance_to_work) = 5 := by
  sorry

end NUMINAMATH_CALUDE_johns_work_days_l197_19763


namespace NUMINAMATH_CALUDE_absolute_value_problem_l197_19720

theorem absolute_value_problem (x y : ℝ) 
  (hx : |x| = 3) 
  (hy : |y| = 2) :
  (x < y → x - y = -5 ∨ x - y = -1) ∧
  (x * y > 0 → x + y = 5) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l197_19720


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l197_19746

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (1 - 3*x + x^2)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                               a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l197_19746


namespace NUMINAMATH_CALUDE_candy_container_problem_l197_19776

theorem candy_container_problem (V₁ V₂ n₁ : ℝ) (h₁ : V₁ = 72) (h₂ : V₂ = 216) (h₃ : n₁ = 30) :
  let n₂ := (n₁ / V₁) * V₂
  n₂ = 90 := by
sorry

end NUMINAMATH_CALUDE_candy_container_problem_l197_19776


namespace NUMINAMATH_CALUDE_area_ratio_rectangle_square_l197_19774

/-- Given a square S and a rectangle R where:
    - The longer side of R is 20% more than a side of S
    - The shorter side of R is 15% less than a side of S
    Prove that the ratio of the area of R to the area of S is 51/50 -/
theorem area_ratio_rectangle_square (S : Real) (R : Real × Real) : 
  R.1 = 1.2 * S ∧ R.2 = 0.85 * S → 
  (R.1 * R.2) / (S * S) = 51 / 50 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_rectangle_square_l197_19774


namespace NUMINAMATH_CALUDE_julias_change_julias_change_is_eight_l197_19773

/-- Calculates Julia's change after purchasing Snickers and M&M's -/
theorem julias_change (snickers_price : ℝ) (snickers_quantity : ℕ) (mms_quantity : ℕ) 
  (payment : ℝ) : ℝ :=
  let mms_price := 2 * snickers_price
  let total_cost := snickers_price * snickers_quantity + mms_price * mms_quantity
  payment - total_cost

/-- Proves that Julia's change is $8 given the specific conditions -/
theorem julias_change_is_eight :
  julias_change 1.5 2 3 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_julias_change_julias_change_is_eight_l197_19773


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l197_19722

theorem profit_percent_calculation (P : ℝ) (C : ℝ) (h : P > 0) (h2 : C > 0) :
  (2/3 * P = 0.86 * C) → ((P - C) / C * 100 = 29) :=
by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l197_19722


namespace NUMINAMATH_CALUDE_marble_count_l197_19735

theorem marble_count (total : ℕ) (yellow : ℕ) (blue_ratio : ℕ) (red_ratio : ℕ) 
  (h1 : total = 19)
  (h2 : yellow = 5)
  (h3 : blue_ratio = 3)
  (h4 : red_ratio = 4) :
  let remaining := total - yellow
  let share := remaining / (blue_ratio + red_ratio)
  let red := red_ratio * share
  red - yellow = 3 := by sorry

end NUMINAMATH_CALUDE_marble_count_l197_19735


namespace NUMINAMATH_CALUDE_hotel_profit_maximized_l197_19747

/-- Represents a hotel with pricing and occupancy information -/
structure Hotel where
  totalRooms : ℕ
  basePrice : ℕ
  priceIncrement : ℕ
  occupancyDecrease : ℕ
  expensePerRoom : ℕ

/-- Calculates the profit for a given price increase -/
def profit (h : Hotel) (priceIncrease : ℕ) : ℤ :=
  let price := h.basePrice + priceIncrease * h.priceIncrement
  let occupiedRooms := h.totalRooms - priceIncrease * h.occupancyDecrease
  (price - h.expensePerRoom) * occupiedRooms

/-- Theorem stating that the profit is maximized at a specific price -/
theorem hotel_profit_maximized (h : Hotel) :
  h.totalRooms = 50 ∧
  h.basePrice = 180 ∧
  h.priceIncrement = 10 ∧
  h.occupancyDecrease = 1 ∧
  h.expensePerRoom = 20 →
  ∃ (maxPriceIncrease : ℕ),
    (∀ (x : ℕ), profit h x ≤ profit h maxPriceIncrease) ∧
    h.basePrice + maxPriceIncrease * h.priceIncrement = 350 :=
sorry

end NUMINAMATH_CALUDE_hotel_profit_maximized_l197_19747


namespace NUMINAMATH_CALUDE_like_terms_example_l197_19706

/-- Two monomials are like terms if they have the same variables raised to the same powers. -/
def are_like_terms (expr1 expr2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), expr1 x y ≠ 0 ∧ expr2 x y ≠ 0 → 
    (∃ (c1 c2 : ℚ), expr1 x y = c1 * x^5 * y^4 ∧ expr2 x y = c2 * x^5 * y^4)

theorem like_terms_example (a b : ℕ) (h1 : a = 2) (h2 : b = 3) :
  are_like_terms (λ x y => b * x^(2*a+1) * y^4) (λ x y => a * x^5 * y^(b+1)) :=
by
  sorry

end NUMINAMATH_CALUDE_like_terms_example_l197_19706


namespace NUMINAMATH_CALUDE_solution_and_minimum_value_l197_19710

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m|

-- State the theorem
theorem solution_and_minimum_value :
  (∀ x : ℝ, f x 2 ≤ 3 ↔ x ∈ Set.Icc (-1) 5) ∧
  (∀ a b c : ℝ, a - 2*b + c = 2 → a^2 + b^2 + c^2 ≥ 2/3) ∧
  (∃ a b c : ℝ, a - 2*b + c = 2 ∧ a^2 + b^2 + c^2 = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_solution_and_minimum_value_l197_19710


namespace NUMINAMATH_CALUDE_fraction_equals_244_375_l197_19715

/-- The fraction in the original problem -/
def original_fraction : ℚ :=
  (12^4+400)*(24^4+400)*(36^4+400)*(48^4+400)*(60^4+400) /
  ((6^4+400)*(18^4+400)*(30^4+400)*(42^4+400)*(54^4+400))

/-- The theorem stating that the original fraction equals 244.375 -/
theorem fraction_equals_244_375 : original_fraction = 244.375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_244_375_l197_19715


namespace NUMINAMATH_CALUDE_tangent_line_equation_l197_19780

/-- The function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 2 * x

/-- The point of tangency -/
def tangent_point : ℝ := 1

/-- The slope of the tangent line at x = 1 -/
def tangent_slope : ℝ := f_deriv tangent_point

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := -(tangent_slope * tangent_point - f tangent_point)

theorem tangent_line_equation :
  ∀ x y : ℝ, y = tangent_slope * x + y_intercept ↔ y = 2 * x - 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l197_19780


namespace NUMINAMATH_CALUDE_flax_acreage_is_80_l197_19753

/-- Represents the acreage of a farm with sunflowers and flax -/
structure FarmAcreage where
  total : ℕ
  flax : ℕ
  sunflowers : ℕ
  total_eq : total = flax + sunflowers
  sunflower_excess : sunflowers = flax + 80

/-- The theorem stating that for a 240-acre farm with the given conditions, 
    the flax acreage is 80 acres -/
theorem flax_acreage_is_80 (farm : FarmAcreage) 
    (h : farm.total = 240) : farm.flax = 80 := by
  sorry

end NUMINAMATH_CALUDE_flax_acreage_is_80_l197_19753


namespace NUMINAMATH_CALUDE_y_squared_times_three_l197_19782

theorem y_squared_times_three (x y : ℤ) 
  (eq1 : 3 * x + y = 40) 
  (eq2 : 2 * x - y = 20) : 
  3 * y^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_times_three_l197_19782


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l197_19761

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l197_19761


namespace NUMINAMATH_CALUDE_product_of_base8_digits_7432_l197_19758

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- The product of the digits in the base 8 representation of 7432₁₀ is 192 --/
theorem product_of_base8_digits_7432 :
  productOfList (toBase8 7432) = 192 := by
  sorry

end NUMINAMATH_CALUDE_product_of_base8_digits_7432_l197_19758


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l197_19767

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 3.4)
  (h2 : (c + d) / 2 = 3.85)
  (h3 : (e + f) / 2 = 4.45) :
  (a + b + c + d + e + f) / 6 = 3.9 := by
  sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l197_19767


namespace NUMINAMATH_CALUDE_triangle_side_length_l197_19770

/-- Given a triangle ABC with specific properties, prove that the length of side b is √7 -/
theorem triangle_side_length (A B C : ℝ) (α l a b c : ℝ) : 
  0 < α → 0 < l → 0 < a → 0 < b → 0 < c →
  B = π / 3 →
  (a * c : ℝ) * Real.cos B = 3 / 2 →
  a + c = 4 →
  b ^ 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l197_19770


namespace NUMINAMATH_CALUDE_correct_num_technicians_l197_19748

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := 7

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 21

/-- Represents the average salary of all workers in Rupees -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians in Rupees -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technician workers in Rupees -/
def avg_salary_rest : ℕ := 6000

/-- Theorem stating that the number of technicians is correct given the conditions -/
theorem correct_num_technicians :
  num_technicians = 7 ∧
  num_technicians ≤ total_workers ∧
  num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_rest =
    total_workers * avg_salary_all :=
by sorry

end NUMINAMATH_CALUDE_correct_num_technicians_l197_19748


namespace NUMINAMATH_CALUDE_max_lilacs_purchase_lilac_purchase_proof_l197_19788

theorem max_lilacs_purchase (cost_per_lilac : ℕ) (max_total_cost : ℕ) : ℕ :=
  let max_lilacs := max_total_cost / cost_per_lilac
  if max_lilacs * cost_per_lilac > max_total_cost then
    max_lilacs - 1
  else
    max_lilacs

theorem lilac_purchase_proof :
  max_lilacs_purchase 6 5000 = 833 :=
by sorry

end NUMINAMATH_CALUDE_max_lilacs_purchase_lilac_purchase_proof_l197_19788


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l197_19701

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 3*x - 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l197_19701


namespace NUMINAMATH_CALUDE_proposition_q_false_l197_19745

theorem proposition_q_false (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬((¬p) ∧ q)) : 
  ¬q := by
  sorry

end NUMINAMATH_CALUDE_proposition_q_false_l197_19745


namespace NUMINAMATH_CALUDE_number_equation_solution_l197_19734

theorem number_equation_solution : 
  ∃ x : ℚ, x^2 + 145 = (x - 19)^2 ∧ x = 108/19 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l197_19734


namespace NUMINAMATH_CALUDE_line_intersects_circle_through_center_l197_19752

open Real

/-- Proves that a line intersects a circle through its center -/
theorem line_intersects_circle_through_center (α : ℝ) :
  let line := fun (x y : ℝ) => x * cos α - y * sin α = 1
  let circle := fun (x y : ℝ) => (x - cos α)^2 + (y + sin α)^2 = 4
  let center := (cos α, -sin α)
  line center.1 center.2 ∧ circle center.1 center.2 := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_through_center_l197_19752


namespace NUMINAMATH_CALUDE_students_per_class_l197_19711

theorem students_per_class 
  (total_students : ℕ) 
  (num_classrooms : ℕ) 
  (h1 : total_students = 120) 
  (h2 : num_classrooms = 24) 
  (h3 : total_students % num_classrooms = 0) -- Ensures equal distribution
  : total_students / num_classrooms = 5 := by
sorry

end NUMINAMATH_CALUDE_students_per_class_l197_19711


namespace NUMINAMATH_CALUDE_water_evaporation_per_day_l197_19717

/-- Proves that given a bowl with 10 ounces of water, where 0.04% of the original amount
    evaporates over 50 days, the amount of water evaporated each day is 0.0008 ounces. -/
theorem water_evaporation_per_day
  (initial_water : Real)
  (evaporation_period : Nat)
  (evaporation_percentage : Real)
  (h1 : initial_water = 10)
  (h2 : evaporation_period = 50)
  (h3 : evaporation_percentage = 0.04)
  : (initial_water * evaporation_percentage / 100) / evaporation_period = 0.0008 := by
  sorry

#check water_evaporation_per_day

end NUMINAMATH_CALUDE_water_evaporation_per_day_l197_19717


namespace NUMINAMATH_CALUDE_quadratic_shift_l197_19700

/-- Represents a quadratic function of the form y = (x + a)² + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_horizontal (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a - d, b := f.b }

/-- Shifts a quadratic function vertically -/
def shift_vertical (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b - d }

/-- The main theorem stating that shifting y = (x + 1)² + 3 by 2 units right and 1 unit down
    results in y = (x - 1)² + 2 -/
theorem quadratic_shift :
  let f := QuadraticFunction.mk 1 3
  let g := shift_vertical (shift_horizontal f 2) 1
  g = QuadraticFunction.mk (-1) 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_shift_l197_19700


namespace NUMINAMATH_CALUDE_line_inclination_angle_ratio_l197_19781

theorem line_inclination_angle_ratio (θ : Real) : 
  (2 : Real) * Real.tan θ + 1 = 0 →
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_ratio_l197_19781


namespace NUMINAMATH_CALUDE_new_average_salary_l197_19750

theorem new_average_salary
  (initial_average : ℚ)
  (old_supervisor_salary : ℚ)
  (new_supervisor_salary : ℚ)
  (num_people : ℕ)
  (h1 : initial_average = 430)
  (h2 : old_supervisor_salary = 870)
  (h3 : new_supervisor_salary = 690)
  (h4 : num_people = 9)
  : (num_people - 1 : ℚ) * initial_average + new_supervisor_salary - old_supervisor_salary = 410 * num_people :=
by
  sorry

#eval (9 - 1 : ℚ) * 430 + 690 - 870
#eval 410 * 9

end NUMINAMATH_CALUDE_new_average_salary_l197_19750


namespace NUMINAMATH_CALUDE_angle_not_in_second_quadrant_l197_19768

def is_in_second_quadrant (angle : ℝ) : Prop :=
  let normalized_angle := angle % 360
  90 < normalized_angle ∧ normalized_angle ≤ 180

theorem angle_not_in_second_quadrant :
  is_in_second_quadrant 160 ∧
  is_in_second_quadrant 480 ∧
  is_in_second_quadrant (-960) ∧
  ¬ is_in_second_quadrant 1530 :=
by sorry

end NUMINAMATH_CALUDE_angle_not_in_second_quadrant_l197_19768


namespace NUMINAMATH_CALUDE_b_contribution_l197_19760

def a_investment : ℕ := 3500
def a_months : ℕ := 12
def b_months : ℕ := 7
def a_share : ℕ := 2
def b_share : ℕ := 3

theorem b_contribution (x : ℕ) : 
  (a_investment * a_months) / (x * b_months) = a_share / b_share → 
  x = 9000 := by
  sorry

end NUMINAMATH_CALUDE_b_contribution_l197_19760


namespace NUMINAMATH_CALUDE_nonnegative_inequality_l197_19742

theorem nonnegative_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a * (a - b) * (a - 2 * b) + b * (b - c) * (b - 2 * c) + c * (c - a) * (c - 2 * a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_inequality_l197_19742


namespace NUMINAMATH_CALUDE_circle_C_equation_l197_19703

/-- A circle C in the xy-plane -/
structure CircleC where
  /-- x-coordinate of a point on the circle -/
  x : ℝ → ℝ
  /-- y-coordinate of a point on the circle -/
  y : ℝ → ℝ
  /-- The parameter θ ranges over all real numbers -/
  θ : ℝ
  /-- x-coordinate is defined as 2 + 2cos(θ) -/
  x_eq : x θ = 2 + 2 * Real.cos θ
  /-- y-coordinate is defined as 2sin(θ) -/
  y_eq : y θ = 2 * Real.sin θ

/-- The standard equation of circle C -/
def standard_equation (c : CircleC) (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

/-- Theorem stating that the parametric equations of CircleC satisfy its standard equation -/
theorem circle_C_equation (c : CircleC) :
  ∀ θ, standard_equation c (c.x θ) (c.y θ) := by
  sorry

end NUMINAMATH_CALUDE_circle_C_equation_l197_19703


namespace NUMINAMATH_CALUDE_tangent_slope_and_sum_inequality_l197_19769

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem tangent_slope_and_sum_inequality
  (a : ℝ)
  (h1 : (deriv (f a)) 0 = -1)
  (x₁ x₂ : ℝ)
  (h2 : x₁ < Real.log 2)
  (h3 : x₂ > Real.log 2)
  (h4 : f a x₁ = f a x₂) :
  x₁ + x₂ < 2 * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_and_sum_inequality_l197_19769


namespace NUMINAMATH_CALUDE_classroom_paint_area_l197_19789

/-- Calculates the area to be painted in a classroom given its dimensions and the area of doors, windows, and blackboard. -/
def areaToPaint (length width height doorWindowBlackboardArea : Real) : Real :=
  let ceilingArea := length * width
  let wallArea := 2 * (length * height + width * height)
  let totalArea := ceilingArea + wallArea
  totalArea - doorWindowBlackboardArea

/-- Theorem stating that the area to be painted in the given classroom is 121.5 square meters. -/
theorem classroom_paint_area :
  areaToPaint 8 6 3.5 24.5 = 121.5 := by
  sorry

end NUMINAMATH_CALUDE_classroom_paint_area_l197_19789


namespace NUMINAMATH_CALUDE_bianca_carrots_l197_19775

/-- The number of carrots Bianca picked the next day -/
def carrots_picked_next_day (initial_carrots thrown_out_carrots final_total : ℕ) : ℕ :=
  final_total - (initial_carrots - thrown_out_carrots)

/-- Theorem stating that Bianca picked 47 carrots the next day -/
theorem bianca_carrots : carrots_picked_next_day 23 10 60 = 47 := by
  sorry

end NUMINAMATH_CALUDE_bianca_carrots_l197_19775


namespace NUMINAMATH_CALUDE_base_conversion_256_to_base_5_l197_19729

theorem base_conversion_256_to_base_5 :
  (2 * 5^3 + 0 * 5^2 + 1 * 5^1 + 1 * 5^0) = 256 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_256_to_base_5_l197_19729


namespace NUMINAMATH_CALUDE_average_sleep_is_eight_l197_19718

def monday_sleep : ℕ := 8
def tuesday_sleep : ℕ := 7
def wednesday_sleep : ℕ := 8
def thursday_sleep : ℕ := 10
def friday_sleep : ℕ := 7

def total_days : ℕ := 5

def total_sleep : ℕ := monday_sleep + tuesday_sleep + wednesday_sleep + thursday_sleep + friday_sleep

theorem average_sleep_is_eight :
  (total_sleep : ℚ) / total_days = 8 := by sorry

end NUMINAMATH_CALUDE_average_sleep_is_eight_l197_19718


namespace NUMINAMATH_CALUDE_lcm_problem_l197_19714

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 6) (h2 : a * b = 432) :
  Nat.lcm a b = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l197_19714


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l197_19708

theorem quadratic_inequality_and_constraint (a b : ℝ) : 
  (∀ x, x < 1 ∨ x > b ↔ a * x^2 - 3 * x + 2 > 0) →
  b > 1 →
  (a = 1 ∧ b = 2) ∧
  (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ 8) ∧
  (∀ k, (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ k^2 + k + 2) ↔ -3 ≤ k ∧ k ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l197_19708


namespace NUMINAMATH_CALUDE_cube_surface_area_l197_19766

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 11) :
  6 * edge_length^2 = 726 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l197_19766


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l197_19772

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculates the total cost of fencing for a rectangular plot. -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot. -/
theorem fencing_cost_calculation (plot : RectangularPlot)
  (h1 : plot.length = 61)
  (h2 : plot.breadth = plot.length - 22)
  (h3 : plot.fencing_cost_per_meter = 26.50) :
  total_fencing_cost plot = 5300 := by
  sorry

#eval total_fencing_cost { length := 61, breadth := 39, fencing_cost_per_meter := 26.50 }

end NUMINAMATH_CALUDE_fencing_cost_calculation_l197_19772


namespace NUMINAMATH_CALUDE_power_inequality_l197_19712

theorem power_inequality : 0.2^0.3 < 0.3^0.3 ∧ 0.3^0.3 < 0.3^0.2 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l197_19712


namespace NUMINAMATH_CALUDE_gcf_lcm_product_8_12_l197_19731

theorem gcf_lcm_product_8_12 : Nat.gcd 8 12 * Nat.lcm 8 12 = 96 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_product_8_12_l197_19731


namespace NUMINAMATH_CALUDE_count_less_than_one_l197_19741

theorem count_less_than_one : 
  let numbers : List ℝ := [0.03, 1.5, -0.2, 0.76]
  (numbers.filter (λ x => x < 1)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_less_than_one_l197_19741


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l197_19778

theorem rectangle_dimensions (perimeter : ℝ) (area : ℝ) 
  (h_perimeter : perimeter = 26) (h_area : area = 42) :
  ∃ (length width : ℝ),
    length + width = perimeter / 2 ∧
    length * width = area ∧
    length = 7 ∧
    width = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l197_19778


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l197_19793

-- Define the line ax + by + c = 0
def line (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Define the quadrants
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- State the theorem
theorem line_passes_through_quadrants (a b c : ℝ) 
  (h1 : a * c < 0) (h2 : b * c < 0) :
  ∃ (x1 y1 x2 y2 x4 y4 : ℝ),
    line a b c x1 y1 ∧ first_quadrant x1 y1 ∧
    line a b c x2 y2 ∧ second_quadrant x2 y2 ∧
    line a b c x4 y4 ∧ fourth_quadrant x4 y4 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l197_19793


namespace NUMINAMATH_CALUDE_transform_equivalence_l197_19784

-- Define the original function
def f : ℝ → ℝ := sorry

-- Define the transformation
def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2*x - 2) + 1

-- Define the horizontal shift
def shift_right (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2*(x - 1))

-- Define the vertical shift
def shift_up (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1

-- Theorem statement
theorem transform_equivalence (x : ℝ) : 
  transform f x = shift_up (shift_right (f ∘ (fun x => 2*x))) x := by sorry

end NUMINAMATH_CALUDE_transform_equivalence_l197_19784


namespace NUMINAMATH_CALUDE_ellipse_equation_l197_19790

/-- The standard equation of an ellipse passing through (-3, 2) with the same foci as x²/9 + y²/4 = 1 -/
theorem ellipse_equation : ∃ (a b : ℝ), 
  (a > 0 ∧ b > 0) ∧ 
  ((-3)^2 / a^2 + 2^2 / b^2 = 1) ∧
  (a^2 - b^2 = 9 - 4) ∧
  (a^2 = 15 ∧ b^2 = 10) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l197_19790


namespace NUMINAMATH_CALUDE_stock_price_increase_l197_19716

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (h1 : opening_price = 8) 
  (h2 : closing_price = 9) : 
  (closing_price - opening_price) / opening_price * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l197_19716


namespace NUMINAMATH_CALUDE_roots_magnitude_l197_19792

theorem roots_magnitude (q : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + q*r₁ - 10 = 0 → 
  r₂^2 + q*r₂ - 10 = 0 → 
  (|r₁| > 4 ∨ |r₂| > 4) :=
by
  sorry

end NUMINAMATH_CALUDE_roots_magnitude_l197_19792


namespace NUMINAMATH_CALUDE_percentage_of_students_with_birds_l197_19799

/-- Given a school with 500 students where 75 students own birds,
    prove that 15% of the students own birds. -/
theorem percentage_of_students_with_birds :
  ∀ (total_students : ℕ) (students_with_birds : ℕ),
    total_students = 500 →
    students_with_birds = 75 →
    (students_with_birds : ℚ) / total_students * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_with_birds_l197_19799


namespace NUMINAMATH_CALUDE_water_intake_increase_l197_19795

theorem water_intake_increase (current : ℕ) (recommended : ℕ) : 
  current = 15 → recommended = 21 → 
  (((recommended - current) : ℚ) / current) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_water_intake_increase_l197_19795


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l197_19719

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ a, a > 2 → a * (a - 2) > 0) ∧ 
  (∃ a, a * (a - 2) > 0 ∧ ¬(a > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l197_19719


namespace NUMINAMATH_CALUDE_jimmy_passing_points_l197_19762

/-- The minimum number of points required to pass the class -/
def passingScore : ℕ := 50

/-- The number of exams Jimmy took -/
def numExams : ℕ := 3

/-- The number of points Jimmy earned per exam -/
def pointsPerExam : ℕ := 20

/-- The number of points Jimmy lost for bad behavior -/
def pointsLost : ℕ := 5

/-- The maximum number of additional points Jimmy can lose while still passing -/
def maxAdditionalPointsLost : ℕ := 5

theorem jimmy_passing_points : 
  numExams * pointsPerExam - pointsLost - maxAdditionalPointsLost ≥ passingScore := by
  sorry

end NUMINAMATH_CALUDE_jimmy_passing_points_l197_19762


namespace NUMINAMATH_CALUDE_time_to_write_michaels_name_l197_19726

/-- The number of letters in Michael's name -/
def name_length : ℕ := 7

/-- The number of rearrangements Michael can write per minute -/
def rearrangements_per_minute : ℕ := 10

/-- Calculate the total number of rearrangements for a name with distinct letters -/
def total_rearrangements (n : ℕ) : ℕ := Nat.factorial n

/-- Calculate the time in hours to write all rearrangements -/
def time_to_write_all (name_len : ℕ) (rearr_per_min : ℕ) : ℚ :=
  (total_rearrangements name_len : ℚ) / (rearr_per_min : ℚ) / 60

/-- Theorem: It takes 8.4 hours to write all rearrangements of Michael's name -/
theorem time_to_write_michaels_name :
  time_to_write_all name_length rearrangements_per_minute = 84 / 10 := by
  sorry

end NUMINAMATH_CALUDE_time_to_write_michaels_name_l197_19726


namespace NUMINAMATH_CALUDE_weight_of_new_person_l197_19736

theorem weight_of_new_person
  (initial_count : ℕ)
  (weight_increase : ℝ)
  (replaced_weight : ℝ)
  (h1 : initial_count = 8)
  (h2 : weight_increase = 6)
  (h3 : replaced_weight = 40)
  : ℝ :=
by
  -- The weight of the new person
  let new_weight := replaced_weight + initial_count * weight_increase
  -- Prove that new_weight = 88
  sorry

#check weight_of_new_person

end NUMINAMATH_CALUDE_weight_of_new_person_l197_19736


namespace NUMINAMATH_CALUDE_tree_distance_l197_19777

/-- Given two buildings 220 meters apart with 10 trees planted at equal intervals,
    the distance between the 1st tree and the 6th tree is 100 meters. -/
theorem tree_distance (building_distance : ℝ) (num_trees : ℕ) 
  (h1 : building_distance = 220)
  (h2 : num_trees = 10) : 
  let interval := building_distance / (num_trees + 1)
  (6 - 1) * interval = 100 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l197_19777


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l197_19730

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) : 
  leg = 15 → angle = 45 → ∃ (hypotenuse : ℝ), hypotenuse = 15 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l197_19730


namespace NUMINAMATH_CALUDE_non_juniors_playing_sport_l197_19740

theorem non_juniors_playing_sport (total_students : ℕ) 
  (junior_sport_percent : ℚ) (non_junior_no_sport_percent : ℚ) 
  (total_no_sport_percent : ℚ) :
  total_students = 600 →
  junior_sport_percent = 1/2 →
  non_junior_no_sport_percent = 2/5 →
  total_no_sport_percent = 13/25 →
  ∃ (non_juniors : ℕ), 
    non_juniors ≤ total_students ∧ 
    (non_juniors : ℚ) * (1 - non_junior_no_sport_percent) = 72 := by
  sorry

end NUMINAMATH_CALUDE_non_juniors_playing_sport_l197_19740


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_solution_for_27461_answer_is_seven_l197_19705

theorem smallest_addition_for_divisibility (n : ℕ) : 
  (∃ (k : ℕ), k < 9 ∧ (n + k) % 9 = 0) → 
  (∃ (m : ℕ), m < 9 ∧ (n + m) % 9 = 0 ∧ ∀ (l : ℕ), l < m → (n + l) % 9 ≠ 0) :=
by sorry

theorem solution_for_27461 : 
  ∃ (k : ℕ), k < 9 ∧ (27461 + k) % 9 = 0 ∧ ∀ (l : ℕ), l < k → (27461 + l) % 9 ≠ 0 :=
by sorry

theorem answer_is_seven : 
  ∃ (k : ℕ), k = 7 ∧ (27461 + k) % 9 = 0 ∧ ∀ (l : ℕ), l < k → (27461 + l) % 9 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_solution_for_27461_answer_is_seven_l197_19705


namespace NUMINAMATH_CALUDE_divisibility_of_f_l197_19759

def f (x : ℕ) : ℕ := x^3 + 17

theorem divisibility_of_f (n : ℕ) (hn : n ≥ 2) :
  ∃ x : ℕ, (3^n ∣ f x) ∧ ¬(3^(n+1) ∣ f x) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_f_l197_19759


namespace NUMINAMATH_CALUDE_arithmetic_progression_five_digit_term_l197_19733

theorem arithmetic_progression_five_digit_term (n : ℕ) (k : ℕ) : 
  let a : ℕ → ℤ := λ i => -1 + (i - 1) * 19
  let is_all_fives : ℤ → Prop := λ x => ∃ m : ℕ, x = 5 * ((10^m - 1) / 9)
  (∃ n, is_all_fives (a n)) ↔ k = 3 ∧ 19 * n - 20 = 5 * ((10^k - 1) / 9) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_five_digit_term_l197_19733


namespace NUMINAMATH_CALUDE_unique_coin_combination_l197_19721

/-- Represents the number of coins of each denomination -/
structure CoinCombination where
  bronze : Nat
  silver : Nat
  gold : Nat

/-- Calculates the total value of a coin combination -/
def totalValue (c : CoinCombination) : Nat :=
  c.bronze + 9 * c.silver + 81 * c.gold

/-- Calculates the total number of coins in a combination -/
def totalCoins (c : CoinCombination) : Nat :=
  c.bronze + c.silver + c.gold

/-- Checks if a coin combination is valid for the problem -/
def isValidCombination (c : CoinCombination) : Prop :=
  totalCoins c = 23 ∧ totalValue c < 700

/-- Checks if a coin combination has the minimum number of coins for its value -/
def isMinimalCombination (c : CoinCombination) : Prop :=
  ∀ c', isValidCombination c' → totalValue c' = totalValue c → totalCoins c' ≥ totalCoins c

/-- The main theorem to prove -/
theorem unique_coin_combination : 
  ∃! c : CoinCombination, isValidCombination c ∧ isMinimalCombination c ∧ totalValue c = 647 :=
sorry

end NUMINAMATH_CALUDE_unique_coin_combination_l197_19721


namespace NUMINAMATH_CALUDE_simplify_nested_sqrt_l197_19787

theorem simplify_nested_sqrt (a : ℝ) (ha : a ≥ 0) :
  Real.sqrt (Real.sqrt (a^(1/2)) * Real.sqrt (Real.sqrt (a^(1/2)) * Real.sqrt a)) = a^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_sqrt_l197_19787


namespace NUMINAMATH_CALUDE_probability_alternating_colors_is_correct_l197_19713

def total_balls : ℕ := 12
def white_balls : ℕ := 6
def black_balls : ℕ := 6

def alternating_sequence : List Bool := [true, false, true, false, true, false, true, false, true, false, true, false]

def probability_alternating_colors : ℚ :=
  1 / (total_balls.choose white_balls)

theorem probability_alternating_colors_is_correct :
  probability_alternating_colors = 1 / 924 :=
by sorry

end NUMINAMATH_CALUDE_probability_alternating_colors_is_correct_l197_19713


namespace NUMINAMATH_CALUDE_unrepresentable_iff_perfect_square_l197_19707

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The sequence a_n defined as floor(n + sqrt(n) + 1/2) -/
noncomputable def a (n : ℕ) : ℤ :=
  floor (n + Real.sqrt n + 1/2)

/-- A natural number is a perfect square -/
def is_perfect_square (k : ℕ) : Prop :=
  ∃ m : ℕ, k = m^2

/-- Main theorem: k cannot be represented by a(n) iff k is a perfect square -/
theorem unrepresentable_iff_perfect_square (k : ℕ) :
  (∀ n : ℕ, a n ≠ k) ↔ is_perfect_square k :=
sorry

end NUMINAMATH_CALUDE_unrepresentable_iff_perfect_square_l197_19707


namespace NUMINAMATH_CALUDE_rowing_coach_votes_l197_19764

theorem rowing_coach_votes (num_coaches : ℕ) (votes_per_coach : ℕ) (coaches_per_voter : ℕ) : 
  num_coaches = 36 → 
  votes_per_coach = 5 → 
  coaches_per_voter = 3 → 
  (num_coaches * votes_per_coach) / coaches_per_voter = 60 := by
  sorry

end NUMINAMATH_CALUDE_rowing_coach_votes_l197_19764


namespace NUMINAMATH_CALUDE_tory_sold_seven_guns_l197_19709

/-- The number of toy guns Tory sold -/
def tory_guns : ℕ := sorry

/-- The price of each toy phone Bert sold -/
def bert_phone_price : ℕ := 18

/-- The number of toy phones Bert sold -/
def bert_phones : ℕ := 8

/-- The price of each toy gun Tory sold -/
def tory_gun_price : ℕ := 20

/-- The difference in earnings between Bert and Tory -/
def earning_difference : ℕ := 4

theorem tory_sold_seven_guns :
  tory_guns = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_tory_sold_seven_guns_l197_19709


namespace NUMINAMATH_CALUDE_water_volume_for_spheres_in_cylinder_l197_19756

/-- The volume of water required to cover two spheres in a cylinder -/
theorem water_volume_for_spheres_in_cylinder (cylinder_diameter cylinder_height : ℝ)
  (small_sphere_radius large_sphere_radius : ℝ) :
  cylinder_diameter = 27 →
  cylinder_height = 30 →
  small_sphere_radius = 6 →
  large_sphere_radius = 9 →
  (π * (cylinder_diameter / 2)^2 * (large_sphere_radius + small_sphere_radius + large_sphere_radius)) -
  (4/3 * π * small_sphere_radius^3 + 4/3 * π * large_sphere_radius^3) = 3114 * π :=
by sorry

end NUMINAMATH_CALUDE_water_volume_for_spheres_in_cylinder_l197_19756


namespace NUMINAMATH_CALUDE_job_completion_time_l197_19738

/-- The time taken for two workers to complete a job together, given their relative efficiencies and the time taken by one worker. -/
theorem job_completion_time 
  (p_efficiency : ℝ) 
  (q_efficiency : ℝ) 
  (p_time : ℝ) 
  (h1 : p_efficiency = q_efficiency + 0.6 * q_efficiency) 
  (h2 : p_time = 26) :
  (p_efficiency * q_efficiency * p_time) / (p_efficiency * q_efficiency + p_efficiency * p_efficiency) = 1690 / 91 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l197_19738


namespace NUMINAMATH_CALUDE_chorus_selection_probability_equal_l197_19702

/-- Represents a two-stage sampling process in a high school chorus selection -/
structure ChorusSelection where
  total_students : ℕ
  eliminated_students : ℕ
  selected_students : ℕ

/-- The probability of a student being selected for the chorus -/
def selection_probability (cs : ChorusSelection) : ℚ :=
  cs.selected_students / cs.total_students

/-- Theorem stating that the selection probability is equal for all students -/
theorem chorus_selection_probability_equal
  (cs : ChorusSelection)
  (h1 : cs.total_students = 1815)
  (h2 : cs.eliminated_students = 15)
  (h3 : cs.selected_students = 30)
  (h4 : cs.total_students = cs.eliminated_students + (cs.total_students - cs.eliminated_students))
  (h5 : cs.selected_students ≤ cs.total_students - cs.eliminated_students) :
  selection_probability cs = 30 / 1815 := by
  sorry

#check chorus_selection_probability_equal

end NUMINAMATH_CALUDE_chorus_selection_probability_equal_l197_19702


namespace NUMINAMATH_CALUDE_davids_english_marks_l197_19732

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  english : ℕ
  average : ℚ

/-- Theorem stating that given David's marks in other subjects and his average,
    his English marks must be 90 -/
theorem davids_english_marks (david : StudentMarks) 
  (math_marks : david.mathematics = 92)
  (physics_marks : david.physics = 85)
  (chemistry_marks : david.chemistry = 87)
  (biology_marks : david.biology = 85)
  (avg_marks : david.average = 87.8)
  : david.english = 90 := by
  sorry

#check davids_english_marks

end NUMINAMATH_CALUDE_davids_english_marks_l197_19732


namespace NUMINAMATH_CALUDE_solution_set_inequality_l197_19755

theorem solution_set_inequality (x : ℝ) :
  x * (2 - x) ≤ 0 ↔ x ≤ 0 ∨ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l197_19755


namespace NUMINAMATH_CALUDE_no_valid_chessboard_config_l197_19724

/-- A chessboard configuration is a function from (Fin 8 × Fin 8) to Fin 64 -/
def ChessboardConfig := Fin 8 × Fin 8 → Fin 64

/-- A 2x2 square on the chessboard -/
structure Square (config : ChessboardConfig) where
  row : Fin 7
  col : Fin 7

/-- The sum of numbers in a 2x2 square -/
def squareSum (config : ChessboardConfig) (square : Square config) : ℕ :=
  (config (square.row, square.col)).val + 1 +
  (config (square.row, square.col.succ)).val + 1 +
  (config (square.row.succ, square.col)).val + 1 +
  (config (square.row.succ, square.col.succ)).val + 1

/-- A valid configuration satisfies the divisibility condition for all 2x2 squares -/
def isValidConfig (config : ChessboardConfig) : Prop :=
  (∀ square : Square config, (squareSum config square) % 5 = 0) ∧
  Function.Injective config

theorem no_valid_chessboard_config : ¬ ∃ config : ChessboardConfig, isValidConfig config := by
  sorry

end NUMINAMATH_CALUDE_no_valid_chessboard_config_l197_19724


namespace NUMINAMATH_CALUDE_age_difference_l197_19765

theorem age_difference (A B C : ℕ) : A + B = B + C + 14 → A = C + 14 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l197_19765


namespace NUMINAMATH_CALUDE_garden_dimensions_l197_19725

/-- Represents a rectangular garden with walkways -/
structure Garden where
  L : ℝ  -- Length of the garden
  W : ℝ  -- Width of the garden
  w : ℝ  -- Width of the walkways
  h_L_gt_W : L > W  -- Length is greater than width

/-- The theorem representing the garden problem -/
theorem garden_dimensions (g : Garden) 
  (h1 : g.w * g.L = 228)  -- First walkway area
  (h2 : g.w * g.W = 117)  -- Second walkway area
  (h3 : g.w * g.L - g.w^2 = 219)  -- Third walkway area
  (h4 : g.w * g.L - (g.w * g.L - g.w^2) = g.w^2)  -- Difference between first and third walkway areas
  : g.L = 76 ∧ g.W = 42 ∧ g.w = 3 := by
  sorry

end NUMINAMATH_CALUDE_garden_dimensions_l197_19725


namespace NUMINAMATH_CALUDE_square_difference_from_sum_and_difference_l197_19791

theorem square_difference_from_sum_and_difference (x y : ℝ) 
  (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_from_sum_and_difference_l197_19791


namespace NUMINAMATH_CALUDE_apple_pear_equivalence_l197_19754

theorem apple_pear_equivalence (apple_value pear_value : ℚ) :
  (3 / 4 : ℚ) * 12 * apple_value = 10 * pear_value →
  (3 / 5 : ℚ) * 15 * apple_value = 10 * pear_value :=
by
  sorry

end NUMINAMATH_CALUDE_apple_pear_equivalence_l197_19754


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l197_19704

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) : 
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 41 / 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l197_19704


namespace NUMINAMATH_CALUDE_soda_distribution_l197_19739

theorem soda_distribution (total_sodas : ℕ) (sisters : ℕ) : 
  total_sodas = 12 →
  sisters = 2 →
  let brothers := 2 * sisters
  let total_siblings := sisters + brothers
  total_sodas / total_siblings = 2 := by
  sorry

end NUMINAMATH_CALUDE_soda_distribution_l197_19739


namespace NUMINAMATH_CALUDE_thirtieth_digit_of_sum_l197_19757

-- Define the fractions
def f1 : ℚ := 1 / 13
def f2 : ℚ := 1 / 11

-- Define the sum of the fractions
def sum : ℚ := f1 + f2

-- Define a function to get the nth digit after the decimal point
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem thirtieth_digit_of_sum : nthDigitAfterDecimal sum 30 = 9 := by sorry

end NUMINAMATH_CALUDE_thirtieth_digit_of_sum_l197_19757


namespace NUMINAMATH_CALUDE_corn_acreage_l197_19727

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l197_19727


namespace NUMINAMATH_CALUDE_point_on_graph_l197_19785

/-- A point (x, y) lies on the graph of y = 2x - 1 -/
def lies_on_graph (x y : ℝ) : Prop := y = 2 * x - 1

/-- The point (2, 3) lies on the graph of y = 2x - 1 -/
theorem point_on_graph : lies_on_graph 2 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l197_19785


namespace NUMINAMATH_CALUDE_sin_alpha_value_l197_19779

theorem sin_alpha_value (α : Real) 
  (h : (Real.sqrt 2 / 2) * (Real.sin (α / 2) - Real.cos (α / 2)) = Real.sqrt 6 / 3) : 
  Real.sin α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l197_19779


namespace NUMINAMATH_CALUDE_inscribed_sphere_pyramid_volume_l197_19751

/-- A regular quadrilateral pyramid with an inscribed sphere -/
structure InscribedSpherePyramid where
  /-- Side length of the base of the pyramid -/
  a : ℝ
  /-- The sphere touches the base and all lateral faces -/
  sphere_touches_all_faces : True
  /-- The sphere divides the height in a 4:5 ratio from the apex -/
  height_ratio : True

/-- Volume of the pyramid -/
noncomputable def pyramid_volume (p : InscribedSpherePyramid) : ℝ :=
  2 * p.a^3 / 5

/-- Theorem stating the volume of the pyramid -/
theorem inscribed_sphere_pyramid_volume (p : InscribedSpherePyramid) :
  pyramid_volume p = 2 * p.a^3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_pyramid_volume_l197_19751


namespace NUMINAMATH_CALUDE_total_vegetables_collected_schoolchildren_vegetable_collection_l197_19743

/-- Represents the amount of vegetables collected by each grade -/
structure VegetableCollection where
  fourth_cabbage : ℕ
  fourth_carrots : ℕ
  fifth_cucumbers : ℕ
  sixth_cucumbers : ℕ
  sixth_onions : ℕ

/-- Theorem stating the total amount of vegetables collected -/
theorem total_vegetables_collected (vc : VegetableCollection) : ℕ :=
  vc.fourth_cabbage + vc.fourth_carrots + vc.fifth_cucumbers + vc.sixth_cucumbers + vc.sixth_onions

/-- Main theorem proving the total amount of vegetables collected is 49 centners -/
theorem schoolchildren_vegetable_collection : 
  ∃ (vc : VegetableCollection), 
    vc.fourth_cabbage = 18 ∧ 
    vc.fourth_carrots = vc.sixth_onions ∧
    vc.fifth_cucumbers < vc.sixth_cucumbers ∧
    vc.fifth_cucumbers > vc.fourth_carrots ∧
    vc.sixth_onions = 7 ∧
    vc.sixth_cucumbers = vc.fourth_cabbage / 2 ∧
    total_vegetables_collected vc = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetables_collected_schoolchildren_vegetable_collection_l197_19743


namespace NUMINAMATH_CALUDE_triangle_area_l197_19728

/-- A triangle with integral sides and perimeter 12 has area 6 -/
theorem triangle_area (a b c : ℕ) : 
  a + b + c = 12 → 
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  (a * b : ℝ) / 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l197_19728


namespace NUMINAMATH_CALUDE_permutation_combination_equality_l197_19771

theorem permutation_combination_equality (n : ℕ) : 
  (n * (n - 1) * (n - 2) = n * (n - 1) * (n - 2) * (n - 3) / 24) → n = 27 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_equality_l197_19771


namespace NUMINAMATH_CALUDE_swimming_time_difference_l197_19798

theorem swimming_time_difference 
  (distance : ℝ) 
  (jack_speed : ℝ) 
  (jill_speed : ℝ) 
  (h1 : distance = 1) 
  (h2 : jack_speed = 10) 
  (h3 : jill_speed = 4) : 
  (distance / jill_speed - distance / jack_speed) * 60 = 9 := by
  sorry

end NUMINAMATH_CALUDE_swimming_time_difference_l197_19798


namespace NUMINAMATH_CALUDE_andy_solves_two_problems_l197_19737

/-- Returns true if n is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- Returns the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Returns true if n has an odd digit sum, false otherwise -/
def hasOddDigitSum (n : ℕ) : Prop := sorry

/-- The set of numbers we're considering -/
def problemSet : Set ℕ := {n : ℕ | 78 ≤ n ∧ n ≤ 125}

/-- The count of prime numbers with odd digit sums in our problem set -/
def countPrimesWithOddDigitSum : ℕ := sorry

theorem andy_solves_two_problems : countPrimesWithOddDigitSum = 2 := by sorry

end NUMINAMATH_CALUDE_andy_solves_two_problems_l197_19737


namespace NUMINAMATH_CALUDE_divisors_of_cube_l197_19723

theorem divisors_of_cube (n : ℕ) : 
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n} ∧ d.card = 5) →
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n^3} ∧ (d.card = 13 ∨ d.card = 16)) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_cube_l197_19723


namespace NUMINAMATH_CALUDE_bakery_items_l197_19794

theorem bakery_items (total : ℕ) (bread_rolls : ℕ) (bagels : ℕ) (croissants : ℕ)
  (h1 : total = 90)
  (h2 : bread_rolls = 49)
  (h3 : bagels = 22)
  (h4 : total = bread_rolls + croissants + bagels) :
  croissants = 19 := by
sorry

end NUMINAMATH_CALUDE_bakery_items_l197_19794


namespace NUMINAMATH_CALUDE_mary_weight_loss_l197_19783

/-- Given Mary's weight changes, prove her initial weight loss --/
theorem mary_weight_loss (initial_weight final_weight : ℝ) 
  (h1 : initial_weight = 99)
  (h2 : final_weight = 81) : 
  ∃ x : ℝ, x = 10.5 ∧ initial_weight - x + 2*x - 3*x + 3 = final_weight :=
by sorry

end NUMINAMATH_CALUDE_mary_weight_loss_l197_19783


namespace NUMINAMATH_CALUDE_same_remainder_mod_ten_l197_19744

theorem same_remainder_mod_ten (a b c : ℕ) 
  (h : ∃ r : ℕ, (2*a + b) % 10 = r ∧ (2*b + c) % 10 = r ∧ (2*c + a) % 10 = r) :
  ∃ s : ℕ, a % 10 = s ∧ b % 10 = s ∧ c % 10 = s := by
  sorry

end NUMINAMATH_CALUDE_same_remainder_mod_ten_l197_19744


namespace NUMINAMATH_CALUDE_parallelogram_angles_l197_19796

theorem parallelogram_angles (A B C D : Real) : 
  -- ABCD is a parallelogram
  (A + C = 180) →
  (B + D = 180) →
  -- ∠B - ∠A = 30°
  (B - A = 30) →
  -- Prove that ∠A = 75°, ∠B = 105°, ∠C = 75°, and ∠D = 105°
  (A = 75 ∧ B = 105 ∧ C = 75 ∧ D = 105) := by
sorry

end NUMINAMATH_CALUDE_parallelogram_angles_l197_19796


namespace NUMINAMATH_CALUDE_problem_statement_l197_19786

theorem problem_statement (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a ^ b = b ^ a) (h2 : b = 9 * a) : a = (3 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l197_19786


namespace NUMINAMATH_CALUDE_incorrect_step_is_count_bacteria_l197_19797

/-- Represents a step in the bacterial counting experiment -/
inductive ExperimentStep
  | PrepMedium
  | SpreadSamples
  | Incubate
  | CountBacteria

/-- Represents a range of bacterial counts -/
structure CountRange where
  lower : ℕ
  upper : ℕ

/-- Defines the correct count range for bacterial counting -/
def correct_count_range : CountRange := { lower := 30, upper := 300 }

/-- Defines whether a step is correct in the experiment -/
def is_correct_step (step : ExperimentStep) : Prop :=
  match step with
  | ExperimentStep.PrepMedium => True
  | ExperimentStep.SpreadSamples => True
  | ExperimentStep.Incubate => True
  | ExperimentStep.CountBacteria => False

/-- Theorem stating that the CountBacteria step is the incorrect one -/
theorem incorrect_step_is_count_bacteria :
  ∃ (step : ExperimentStep), ¬(is_correct_step step) ↔ step = ExperimentStep.CountBacteria :=
sorry

end NUMINAMATH_CALUDE_incorrect_step_is_count_bacteria_l197_19797
