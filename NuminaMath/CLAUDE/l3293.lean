import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3293_329333

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b) ∨ (b = c) ∨ (a = c)
  sumOfAngles : a + b + c = 180

-- Define the condition of angle ratio
def angleRatio (t : IsoscelesTriangle) : Prop :=
  (t.a = 2 * t.b) ∨ (t.b = 2 * t.c) ∨ (t.c = 2 * t.a)

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (t : IsoscelesTriangle) 
  (h : angleRatio t) : 
  (t.a = 90 ∨ t.b = 90 ∨ t.c = 90) ∨ 
  (t.a = 36 ∨ t.b = 36 ∨ t.c = 36) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3293_329333


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l3293_329334

theorem sin_cos_sum_equals_half :
  Real.sin (13 * π / 180) * Real.cos (343 * π / 180) +
  Real.cos (13 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l3293_329334


namespace NUMINAMATH_CALUDE_airplane_hovering_time_l3293_329302

/-- Calculates the total hovering time for an airplane over two days -/
theorem airplane_hovering_time 
  (mountain_day1 : ℕ) 
  (central_day1 : ℕ) 
  (eastern_day1 : ℕ) 
  (additional_time : ℕ) 
  (h1 : mountain_day1 = 3)
  (h2 : central_day1 = 4)
  (h3 : eastern_day1 = 2)
  (h4 : additional_time = 2) :
  mountain_day1 + central_day1 + eastern_day1 + 
  (mountain_day1 + additional_time) + 
  (central_day1 + additional_time) + 
  (eastern_day1 + additional_time) = 24 := by
sorry


end NUMINAMATH_CALUDE_airplane_hovering_time_l3293_329302


namespace NUMINAMATH_CALUDE_replaced_man_age_l3293_329311

theorem replaced_man_age (A B C D : ℝ) (new_avg : ℝ) :
  A = 23 →
  (A + B + C + D) / 4 < (52 + C + D) / 4 →
  B < 29 := by
sorry

end NUMINAMATH_CALUDE_replaced_man_age_l3293_329311


namespace NUMINAMATH_CALUDE_range_of_m_l3293_329301

theorem range_of_m (x y m : ℝ) : 
  (x + 2*y = 4*m) → 
  (2*x + y = 2*m + 1) → 
  (-1 < x - y) → 
  (x - y < 0) → 
  (1/2 < m ∧ m < 1) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3293_329301


namespace NUMINAMATH_CALUDE_floor_equality_iff_interval_l3293_329323

theorem floor_equality_iff_interval (x : ℝ) : 
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 2 ≤ x ∧ x < 7/3 := by sorry

end NUMINAMATH_CALUDE_floor_equality_iff_interval_l3293_329323


namespace NUMINAMATH_CALUDE_arrangements_with_restrictions_total_arrangements_prove_total_arrangements_l3293_329375

-- Define the number of people
def n : ℕ := 5

-- Define the function to calculate permutations
def permutations (k : ℕ) : ℕ := Nat.factorial k

-- Define the function to calculate arrangements
def arrangements (n k : ℕ) : ℕ := permutations n / permutations (n - k)

-- Theorem statement
theorem arrangements_with_restrictions :
  arrangements n n - 2 * arrangements (n - 1) (n - 1) + arrangements (n - 2) (n - 2) = 78 := by
  sorry

-- The result we want to prove
theorem total_arrangements : ℕ := 78

-- The main theorem
theorem prove_total_arrangements :
  arrangements n n - 2 * arrangements (n - 1) (n - 1) + arrangements (n - 2) (n - 2) = total_arrangements := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_restrictions_total_arrangements_prove_total_arrangements_l3293_329375


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3293_329385

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 - 3 * x^4 + x^3 + 5 * x^2 - 8 * x + 15) +
  (-5 * x^4 - 2 * x^3 + 3 * x^2 + 8 * x + 9) =
  2 * x^5 - 8 * x^4 - x^3 + 8 * x^2 + 24 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3293_329385


namespace NUMINAMATH_CALUDE_part_one_part_two_l3293_329306

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Part 1 of the problem -/
theorem part_one (t : Triangle) (h1 : t.A = π / 3) (h2 : t.a = 4 * Real.sqrt 3) (h3 : t.b = 4 * Real.sqrt 2) :
  t.B = π / 4 := by
  sorry

/-- Part 2 of the problem -/
theorem part_two (t : Triangle) (h1 : t.a = 3 * Real.sqrt 3) (h2 : t.c = 2) (h3 : t.B = 5 * π / 6) :
  t.b = 7 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3293_329306


namespace NUMINAMATH_CALUDE_total_black_dots_l3293_329317

theorem total_black_dots (num_butterflies : ℕ) (black_dots_per_butterfly : ℕ) 
  (h1 : num_butterflies = 397) 
  (h2 : black_dots_per_butterfly = 12) : 
  num_butterflies * black_dots_per_butterfly = 4764 := by
  sorry

end NUMINAMATH_CALUDE_total_black_dots_l3293_329317


namespace NUMINAMATH_CALUDE_modified_triathlon_speed_l3293_329313

theorem modified_triathlon_speed (total_time : ℝ) (swim_distance swim_speed : ℝ)
  (run_distance run_speed : ℝ) (kayak_distance kayak_speed : ℝ)
  (bike_distance : ℝ) :
  total_time = 3 ∧
  swim_distance = 1/2 ∧ swim_speed = 2 ∧
  run_distance = 5 ∧ run_speed = 10 ∧
  kayak_distance = 1 ∧ kayak_speed = 3 ∧
  bike_distance = 20 →
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed + kayak_distance / kayak_speed))) = 240/23 :=
by sorry

end NUMINAMATH_CALUDE_modified_triathlon_speed_l3293_329313


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3293_329305

def A : Set ℤ := {1, 2, 3, 5, 7}
def B : Set ℤ := {x : ℤ | 1 < x ∧ x ≤ 6}
def U : Set ℤ := A ∪ B

theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3293_329305


namespace NUMINAMATH_CALUDE_sin_monotone_increasing_l3293_329364

open Real

theorem sin_monotone_increasing (t : ℝ) (h : 0 < t ∧ t < π / 6) :
  StrictMonoOn (fun x => sin (2 * x + π / 6)) (Set.Ioo (-t) t) := by
  sorry

end NUMINAMATH_CALUDE_sin_monotone_increasing_l3293_329364


namespace NUMINAMATH_CALUDE_area_enclosed_theorem_l3293_329332

/-- Represents a configuration of three intersecting circles -/
structure CircleConfiguration where
  radius : ℝ
  centralAngle : ℝ
  numCircles : ℕ

/-- Calculates the area enclosed by the arcs of the circle configuration -/
def areaEnclosedByArcs (config : CircleConfiguration) : ℝ :=
  sorry

/-- Represents the coefficients of the area formula a√b + cπ -/
structure AreaCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the area can be expressed as a√b + cπ with a + b + c = 40.5 -/
theorem area_enclosed_theorem (config : CircleConfiguration) 
  (h1 : config.radius = 5)
  (h2 : config.centralAngle = π / 2)
  (h3 : config.numCircles = 3) :
  ∃ (coef : AreaCoefficients), 
    areaEnclosedByArcs config = coef.a * Real.sqrt coef.b + coef.c * π ∧
    coef.a + coef.b + coef.c = 40.5 :=
  sorry

end NUMINAMATH_CALUDE_area_enclosed_theorem_l3293_329332


namespace NUMINAMATH_CALUDE_slope_constraint_implies_a_bound_l3293_329352

/-- Given a function f(x) = x ln(x) + ax^2, if there exists a point where the slope is 3,
    then a is greater than or equal to -1 / (2e^3). -/
theorem slope_constraint_implies_a_bound (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (Real.log x + 1 + 2 * a * x = 3)) →
  a ≥ -1 / (2 * Real.exp 3) :=
by sorry

end NUMINAMATH_CALUDE_slope_constraint_implies_a_bound_l3293_329352


namespace NUMINAMATH_CALUDE_greatest_power_under_500_l3293_329337

theorem greatest_power_under_500 :
  ∃ (a b : ℕ), 
    a > 0 ∧ 
    b > 1 ∧ 
    a^b < 500 ∧ 
    (∀ (c d : ℕ), c > 0 → d > 1 → c^d < 500 → c^d ≤ a^b) ∧ 
    a + b = 24 :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_under_500_l3293_329337


namespace NUMINAMATH_CALUDE_samson_utility_l3293_329362

/-- Utility function -/
def utility (math_hours : ℝ) (frisbee_hours : ℝ) : ℝ :=
  (math_hours + 2) * frisbee_hours

/-- The problem statement -/
theorem samson_utility (s : ℝ) : 
  utility (10 - 2*s) s = utility (2*s + 4) (3 - s) → s = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_samson_utility_l3293_329362


namespace NUMINAMATH_CALUDE_geometry_problem_l3293_329321

-- Define the points
def M : ℝ × ℝ := (2, -2)
def N : ℝ × ℝ := (4, 4)
def P : ℝ × ℝ := (2, -3)

-- Define the equations
def perpendicular_bisector (x y : ℝ) : Prop := x + 3*y - 6 = 0
def parallel_line (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem geometry_problem :
  (∀ x y : ℝ, perpendicular_bisector x y ↔ 
    (x - M.1)^2 + (y - M.2)^2 = (x - N.1)^2 + (y - N.2)^2) ∧
  (∀ x y : ℝ, parallel_line x y ↔ 
    (y - P.2) = ((N.2 - M.2) / (N.1 - M.1)) * (x - P.1)) :=
by sorry

end NUMINAMATH_CALUDE_geometry_problem_l3293_329321


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3293_329314

/-- Calculates the average speed of a trip given the conditions specified in the problem -/
theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3293_329314


namespace NUMINAMATH_CALUDE_sqrt_102_between_consecutive_integers_l3293_329341

theorem sqrt_102_between_consecutive_integers : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 102 ∧ 
  Real.sqrt 102 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 110 := by
sorry

end NUMINAMATH_CALUDE_sqrt_102_between_consecutive_integers_l3293_329341


namespace NUMINAMATH_CALUDE_verbal_equals_algebraic_l3293_329316

/-- The verbal description of the algebraic expression "5-4a" -/
def verbal_description : String := "the difference of 5 and 4 times a"

/-- The algebraic expression -/
def algebraic_expression (a : ℝ) : ℝ := 5 - 4 * a

theorem verbal_equals_algebraic :
  ∀ a : ℝ, verbal_description = "the difference of 5 and 4 times a" ↔ 
  algebraic_expression a = 5 - 4 * a :=
by sorry

end NUMINAMATH_CALUDE_verbal_equals_algebraic_l3293_329316


namespace NUMINAMATH_CALUDE_student_photo_count_l3293_329308

theorem student_photo_count :
  ∀ (m n : ℕ),
    m > 0 →
    n > 0 →
    m + 4 = n - 1 →  -- First rearrangement condition
    m + 3 = n - 2 →  -- Second rearrangement condition
    m * n = 24 :=    -- Total number of students
by
  sorry

end NUMINAMATH_CALUDE_student_photo_count_l3293_329308


namespace NUMINAMATH_CALUDE_cube_root_of_64_l3293_329370

theorem cube_root_of_64 : ∃ (a : ℝ), a^3 = 64 ∧ a = 4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l3293_329370


namespace NUMINAMATH_CALUDE_commission_percentage_problem_l3293_329376

/-- Calculates the commission percentage given the commission amount and total sales. -/
def commission_percentage (commission : ℚ) (total_sales : ℚ) : ℚ :=
  (commission / total_sales) * 100

/-- Theorem stating that for the given commission and sales values, the commission percentage is 4%. -/
theorem commission_percentage_problem :
  let commission : ℚ := 25/2  -- Rs. 12.50
  let total_sales : ℚ := 625/2  -- Rs. 312.5
  commission_percentage commission total_sales = 4 := by
  sorry

end NUMINAMATH_CALUDE_commission_percentage_problem_l3293_329376


namespace NUMINAMATH_CALUDE_tumbler_payment_denomination_l3293_329338

/-- Proves that the denomination of bills used to pay for tumblers is $100 given the specified conditions -/
theorem tumbler_payment_denomination :
  ∀ (num_tumblers : ℕ) (cost_per_tumbler : ℕ) (num_bills : ℕ) (change : ℕ),
    num_tumblers = 10 →
    cost_per_tumbler = 45 →
    num_bills = 5 →
    change = 50 →
    (num_tumblers * cost_per_tumbler + change) / num_bills = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_tumbler_payment_denomination_l3293_329338


namespace NUMINAMATH_CALUDE_sally_found_two_balloons_l3293_329356

/-- The number of additional orange balloons Sally found -/
def additional_balloons (initial final : ℝ) : ℝ := final - initial

/-- Theorem stating that Sally found 2.0 more orange balloons -/
theorem sally_found_two_balloons (initial final : ℝ) 
  (h1 : initial = 9.0) 
  (h2 : final = 11) : 
  additional_balloons initial final = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_sally_found_two_balloons_l3293_329356


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3293_329318

/-- The y-intercept of the line described by x - 2y + x^2 = 8 is -4 -/
theorem y_intercept_of_line (x y : ℝ) : x - 2*y + x^2 = 8 → (0 - 2*y + 0^2 = 8 → y = -4) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3293_329318


namespace NUMINAMATH_CALUDE_pony_discount_rate_l3293_329358

/-- Represents the discount rate for Fox jeans -/
def F : ℝ := sorry

/-- Represents the discount rate for Pony jeans -/
def P : ℝ := sorry

/-- Regular price of Fox jeans -/
def fox_price : ℝ := 15

/-- Regular price of Pony jeans -/
def pony_price : ℝ := 18

/-- Total savings from purchasing 3 pairs of Fox jeans and 2 pairs of Pony jeans -/
def total_savings : ℝ := 8.64

/-- The sum of discount rates for Fox and Pony jeans -/
def total_discount : ℝ := 22

theorem pony_discount_rate : 
  F + P = total_discount ∧ 
  3 * (fox_price * F / 100) + 2 * (pony_price * P / 100) = total_savings →
  P = 14 := by sorry

end NUMINAMATH_CALUDE_pony_discount_rate_l3293_329358


namespace NUMINAMATH_CALUDE_new_person_weight_l3293_329343

/-- Given a group of 15 people, proves that if replacing a person weighing 45 kg 
    with a new person increases the average weight by 8 kg, 
    then the new person weighs 165 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) 
  (replaced_weight : ℝ) (new_weight : ℝ) : 
  initial_count = 15 → 
  weight_increase = 8 → 
  replaced_weight = 45 → 
  new_weight = initial_count * weight_increase + replaced_weight → 
  new_weight = 165 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3293_329343


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l3293_329351

theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < 0 → 0 < x₂ → y₁ = 6 / x₁ → y₂ = 6 / x₂ → y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l3293_329351


namespace NUMINAMATH_CALUDE_tan_period_l3293_329383

/-- The period of y = tan(3x/4) is 4π/3 -/
theorem tan_period (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.tan (3 * x / 4)
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = 4 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_period_l3293_329383


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_G_powers_of_two_l3293_329327

def G : ℕ → ℚ
  | 0 => 0
  | 1 => 5/2
  | (n + 2) => 7/2 * G (n + 1) - G n

theorem sum_of_reciprocal_G_powers_of_two : ∑' n, 1 / G (2^n) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_G_powers_of_two_l3293_329327


namespace NUMINAMATH_CALUDE_most_likely_outcome_l3293_329379

-- Define the number of children
def n : ℕ := 5

-- Define the probability of a child being a boy or a girl
def p : ℚ := 1/2

-- Define the probability of all children being the same gender
def prob_all_same : ℚ := 2 * p^n

-- Define the probability of having 2 of one gender and 3 of the other
def prob_2_3 : ℚ := (n.choose 2) * p^n

-- Define the probability of having 4 of one gender and 1 of the other
def prob_4_1 : ℚ := 2 * (n.choose 1) * p^n

-- Theorem statement
theorem most_likely_outcome :
  prob_2_3 + prob_4_1 > prob_all_same :=
by sorry

end NUMINAMATH_CALUDE_most_likely_outcome_l3293_329379


namespace NUMINAMATH_CALUDE_apples_given_theorem_l3293_329380

/-- The number of apples Joan gave to Melanie -/
def apples_given_to_melanie (initial_apples current_apples : ℕ) : ℕ :=
  initial_apples - current_apples

/-- Proof that the number of apples given to Melanie is correct -/
theorem apples_given_theorem (initial_apples current_apples : ℕ) 
  (h : initial_apples ≥ current_apples) :
  apples_given_to_melanie initial_apples current_apples = initial_apples - current_apples :=
by
  sorry

/-- Verifying the specific case in the problem -/
example : apples_given_to_melanie 43 16 = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_apples_given_theorem_l3293_329380


namespace NUMINAMATH_CALUDE_factorization_equality_l3293_329319

theorem factorization_equality (x y : ℝ) : x^2*y - 2*x*y^2 + y^3 = y*(x-y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3293_329319


namespace NUMINAMATH_CALUDE_shopping_cost_other_goods_l3293_329391

def tuna_packs : ℕ := 5
def tuna_price : ℚ := 2
def water_bottles : ℕ := 4
def water_price : ℚ := 3/2
def discount_rate : ℚ := 1/10
def paid_after_discount : ℚ := 56
def conversion_rate : ℚ := 3/2

theorem shopping_cost_other_goods :
  let total_cost := paid_after_discount / (1 - discount_rate)
  let tuna_water_cost := tuna_packs * tuna_price + water_bottles * water_price
  let other_goods_local := total_cost - tuna_water_cost
  let other_goods_home := other_goods_local / conversion_rate
  other_goods_home = 30.81 := by sorry

end NUMINAMATH_CALUDE_shopping_cost_other_goods_l3293_329391


namespace NUMINAMATH_CALUDE_simplify_fraction_l3293_329303

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) :
  (x + 2) / (x^2 - 2*x) / ((8*x / (x - 2)) + x - 2) = 1 / (x * (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3293_329303


namespace NUMINAMATH_CALUDE_ac_plus_bd_value_l3293_329365

theorem ac_plus_bd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 10)
  (eq2 : a + b + d = -6)
  (eq3 : a + c + d = 0)
  (eq4 : b + c + d = 15) :
  a * c + b * d = -1171 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ac_plus_bd_value_l3293_329365


namespace NUMINAMATH_CALUDE_winter_sports_camp_l3293_329353

theorem winter_sports_camp (total_students : ℕ) (boys girls : ℕ) (pine_students oak_students : ℕ)
  (seventh_grade eighth_grade : ℕ) (pine_girls : ℕ) :
  total_students = 120 →
  boys = 70 →
  girls = 50 →
  pine_students = 70 →
  oak_students = 50 →
  seventh_grade = 60 →
  eighth_grade = 60 →
  pine_girls = 30 →
  pine_students / 2 = seventh_grade →
  ∃ (oak_eighth_boys : ℕ), oak_eighth_boys = 15 :=
by sorry

end NUMINAMATH_CALUDE_winter_sports_camp_l3293_329353


namespace NUMINAMATH_CALUDE_unique_zero_implies_t_bound_l3293_329371

/-- A cubic function parameterized by t -/
def f (t : ℝ) (x : ℝ) : ℝ := -2 * x^3 + 2 * t * x^2 + 1

/-- The derivative of f with respect to x -/
def f_deriv (t : ℝ) (x : ℝ) : ℝ := -6 * x^2 + 4 * t * x

/-- Theorem stating that if f has a unique zero, then t > -3/2 -/
theorem unique_zero_implies_t_bound (t : ℝ) :
  (∃! x, f t x = 0) → t > -3/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_implies_t_bound_l3293_329371


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_equality_l3293_329369

theorem rectangle_triangle_area_equality (l w h : ℝ) (l_pos : l > 0) (w_pos : w > 0) (h_pos : h > 0) :
  l * w = (1 / 2) * l * h → h = 2 * w := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_equality_l3293_329369


namespace NUMINAMATH_CALUDE_friday_texts_l3293_329390

/-- Represents the number of texts sent to each friend on a given day -/
structure DailyTexts where
  allison : ℕ
  brittney : ℕ
  carol : ℕ
  dylan : ℕ

/-- Calculates the total number of texts sent in a day -/
def totalTexts (d : DailyTexts) : ℕ := d.allison + d.brittney + d.carol + d.dylan

/-- Sydney's texting schedule from Monday to Thursday -/
def textSchedule : List DailyTexts := [
  ⟨5, 5, 5, 5⟩,        -- Monday
  ⟨15, 10, 12, 8⟩,     -- Tuesday
  ⟨20, 18, 7, 14⟩,     -- Wednesday
  ⟨0, 25, 10, 5⟩       -- Thursday
]

/-- Cost of a single text in cents -/
def textCost : ℕ := 10

/-- Weekly budget in cents -/
def weeklyBudget : ℕ := 2000

/-- Theorem: Sydney can send 36 texts on Friday given her schedule and budget -/
theorem friday_texts : 
  (weeklyBudget - (textSchedule.map totalTexts).sum * textCost) / textCost = 36 := by
  sorry

end NUMINAMATH_CALUDE_friday_texts_l3293_329390


namespace NUMINAMATH_CALUDE_ants_after_five_hours_l3293_329350

/-- The number of ants in the jar after a given number of hours -/
def antsInJar (initialAnts : ℕ) (hours : ℕ) : ℕ :=
  initialAnts * (2 ^ hours)

/-- Theorem stating that 50 ants doubling every hour for 5 hours results in 1600 ants -/
theorem ants_after_five_hours :
  antsInJar 50 5 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_ants_after_five_hours_l3293_329350


namespace NUMINAMATH_CALUDE_high_school_ten_season_games_l3293_329355

/-- Represents a basketball conference -/
structure BasketballConference where
  teamCount : ℕ
  intraConferenceGamesPerPair : ℕ
  nonConferenceGamesPerTeam : ℕ

/-- Calculates the total number of games in a season for a given basketball conference -/
def totalSeasonGames (conf : BasketballConference) : ℕ :=
  let intraConferenceGames := conf.teamCount.choose 2 * conf.intraConferenceGamesPerPair
  let nonConferenceGames := conf.teamCount * conf.nonConferenceGamesPerTeam
  intraConferenceGames + nonConferenceGames

/-- The High School Ten basketball conference -/
def highSchoolTen : BasketballConference :=
  { teamCount := 10
  , intraConferenceGamesPerPair := 2
  , nonConferenceGamesPerTeam := 6 }

theorem high_school_ten_season_games :
  totalSeasonGames highSchoolTen = 150 := by
  sorry

end NUMINAMATH_CALUDE_high_school_ten_season_games_l3293_329355


namespace NUMINAMATH_CALUDE_ages_sum_l3293_329340

/-- Given the ages of Al, Bob, and Carl satisfying certain conditions, prove their sum is 80 -/
theorem ages_sum (a b c : ℕ) : 
  a = b + c + 20 → 
  a^2 = (b + c)^2 + 2000 → 
  a + b + c = 80 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l3293_329340


namespace NUMINAMATH_CALUDE_specific_rhombus_area_l3293_329394

/-- Represents a rhombus with given properties -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  diagonals_perpendicular_bisectors : Bool

/-- Calculates the area of a rhombus given its properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of a specific rhombus -/
theorem specific_rhombus_area :
  let r : Rhombus := {
    side_length := Real.sqrt 113,
    diagonal_difference := 8,
    diagonals_perpendicular_bisectors := true
  }
  rhombus_area r = 97 := by sorry

end NUMINAMATH_CALUDE_specific_rhombus_area_l3293_329394


namespace NUMINAMATH_CALUDE_highest_power_of_three_in_M_l3293_329388

def M : ℕ := sorry

def is_highest_power_of_three (n : ℕ) (j : ℕ) : Prop :=
  3^j ∣ n ∧ ∀ k > j, ¬(3^k ∣ n)

theorem highest_power_of_three_in_M :
  is_highest_power_of_three M 0 := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_in_M_l3293_329388


namespace NUMINAMATH_CALUDE_median_and_area_of_triangle_l3293_329398

/-- Triangle DEF with given side lengths -/
structure Triangle where
  DE : ℝ
  DF : ℝ
  EF : ℝ

/-- The isosceles triangle DEF with given side lengths -/
def isoscelesTriangle : Triangle where
  DE := 13
  DF := 13
  EF := 14

/-- The length of the median DM in triangle DEF -/
def medianLength (t : Triangle) : ℝ := sorry

/-- The area of triangle DEF -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Theorem stating the length of the median and the area of the triangle -/
theorem median_and_area_of_triangle :
  medianLength isoscelesTriangle = 2 * Real.sqrt 30 ∧
  triangleArea isoscelesTriangle = 84 := by sorry

end NUMINAMATH_CALUDE_median_and_area_of_triangle_l3293_329398


namespace NUMINAMATH_CALUDE_f_neg_five_l3293_329366

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 + 1

-- State the theorem
theorem f_neg_five (a b : ℝ) (h : f a b 5 = 7) : f a b (-5) = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_five_l3293_329366


namespace NUMINAMATH_CALUDE_car_trip_duration_l3293_329361

/-- Represents the duration of a car trip with varying speeds -/
def car_trip (first_speed second_speed average_speed : ℝ) (first_duration : ℝ) : ℝ → Prop :=
  λ total_duration : ℝ =>
    let second_duration := total_duration - first_duration
    let total_distance := first_speed * first_duration + second_speed * second_duration
    (total_distance / total_duration = average_speed) ∧
    (total_duration > first_duration) ∧
    (first_duration > 0) ∧
    (second_duration > 0)

/-- Theorem stating that the car trip with given parameters lasts 7.5 hours -/
theorem car_trip_duration :
  car_trip 30 42 34 5 7.5 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_duration_l3293_329361


namespace NUMINAMATH_CALUDE_stream_speed_l3293_329307

/-- The speed of a stream given downstream and upstream speeds -/
theorem stream_speed (downstream upstream : ℝ) (h1 : downstream = 13) (h2 : upstream = 8) :
  (downstream - upstream) / 2 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3293_329307


namespace NUMINAMATH_CALUDE_cube_coloring_probability_l3293_329325

/-- The probability of a single color being chosen for a face -/
def color_probability : ℚ := 1/3

/-- The number of pairs of opposite faces in a cube -/
def opposite_face_pairs : ℕ := 3

/-- The probability that a pair of opposite faces has different colors -/
def diff_color_prob : ℚ := 2/3

/-- The probability that all pairs of opposite faces have different colors -/
def all_diff_prob : ℚ := diff_color_prob ^ opposite_face_pairs

theorem cube_coloring_probability :
  1 - all_diff_prob = 19/27 := by sorry

end NUMINAMATH_CALUDE_cube_coloring_probability_l3293_329325


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3293_329367

open Set

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {1, 2, 3, 4, 5}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3293_329367


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3293_329344

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 150 → volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3293_329344


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3293_329373

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (Real.sqrt 13 / Real.sqrt 15) = 
  (3 * Real.sqrt 3003) / 231 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3293_329373


namespace NUMINAMATH_CALUDE_beans_remaining_fraction_l3293_329377

/-- The fraction of beans remaining in a jar after some have been removed -/
theorem beans_remaining_fraction (jar_weight : ℝ) (full_beans_weight : ℝ) 
  (h1 : jar_weight = 0.1 * (jar_weight + full_beans_weight))
  (h2 : ∃ remaining_beans : ℝ, jar_weight + remaining_beans = 0.6 * (jar_weight + full_beans_weight)) :
  ∃ remaining_beans : ℝ, remaining_beans / full_beans_weight = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_beans_remaining_fraction_l3293_329377


namespace NUMINAMATH_CALUDE_expand_expression_l3293_329372

theorem expand_expression (x : ℝ) : (x - 2) * (x + 2) * (x^2 + x + 6) = x^4 + x^3 + 2*x^2 - 4*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3293_329372


namespace NUMINAMATH_CALUDE_function_inequality_l3293_329300

theorem function_inequality (f : ℝ → ℝ) :
  (∀ x ≥ 1, f x ≤ x) →
  (∀ x ≥ 1, f (2 * x) / Real.sqrt 2 ≤ f x) →
  (∀ x ≥ 1, f x < Real.sqrt (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3293_329300


namespace NUMINAMATH_CALUDE_barbaras_to_mikes_age_ratio_l3293_329346

/-- Given that Mike is currently 16 years old and Barbara will be 16 years old
    when Mike is 24 years old, prove that the ratio of Barbara's current age
    to Mike's current age is 1:2. -/
theorem barbaras_to_mikes_age_ratio :
  let mike_current_age : ℕ := 16
  let mike_future_age : ℕ := 24
  let barbara_future_age : ℕ := 16
  let age_difference : ℕ := mike_future_age - mike_current_age
  let barbara_current_age : ℕ := barbara_future_age - age_difference
  (barbara_current_age : ℚ) / mike_current_age = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_barbaras_to_mikes_age_ratio_l3293_329346


namespace NUMINAMATH_CALUDE_missing_fraction_proof_l3293_329335

theorem missing_fraction_proof :
  let given_fractions : List ℚ := [1/3, 1/2, -5/6, 1/4, -9/20, -5/6]
  let missing_fraction : ℚ := 56/30
  let total_sum : ℚ := 5/6
  (given_fractions.sum + missing_fraction = total_sum) := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_proof_l3293_329335


namespace NUMINAMATH_CALUDE_martin_crayon_boxes_l3293_329360

theorem martin_crayon_boxes
  (crayons_per_box : ℕ)
  (total_crayons : ℕ)
  (h1 : crayons_per_box = 7)
  (h2 : total_crayons = 56) :
  total_crayons / crayons_per_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_martin_crayon_boxes_l3293_329360


namespace NUMINAMATH_CALUDE_expression_evaluation_l3293_329322

theorem expression_evaluation (a b : ℤ) (h1 : a = 3) (h2 : b = 2) : 
  (a^2 + a*b + b^2)^2 - (a^2 - a*b + b^2)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3293_329322


namespace NUMINAMATH_CALUDE_other_replaced_man_age_l3293_329382

/-- The age of the other replaced man in a group replacement scenario --/
def age_of_other_replaced_man (initial_count : ℕ) (age_increase : ℕ) 
  (age_first_replaced : ℕ) (avg_age_new_men : ℕ) : ℕ :=
  let total_age_increase := initial_count * age_increase
  let total_age_new_men := 2 * avg_age_new_men
  total_age_new_men - total_age_increase - age_first_replaced

/-- Theorem stating the age of the other replaced man is 23 --/
theorem other_replaced_man_age :
  age_of_other_replaced_man 10 2 21 32 = 23 := by
  sorry

#eval age_of_other_replaced_man 10 2 21 32

end NUMINAMATH_CALUDE_other_replaced_man_age_l3293_329382


namespace NUMINAMATH_CALUDE_position_of_2013_l3293_329399

/-- Represents the position of a number in the arrangement -/
structure Position where
  row : Nat
  column : Nat
  deriving Repr

/-- Calculates the position of a given odd number in the arrangement -/
def position_of_odd_number (n : Nat) : Position :=
  sorry

theorem position_of_2013 : position_of_odd_number 2013 = ⟨45, 17⟩ := by
  sorry

end NUMINAMATH_CALUDE_position_of_2013_l3293_329399


namespace NUMINAMATH_CALUDE_circle_and_lines_theorem_l3293_329348

/-- Represents a circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- Represents a line y = k(x - 2) -/
structure Line where
  k : ℝ

/-- Checks if a circle satisfies the given conditions -/
def satisfiesConditions (c : Circle) : Prop :=
  c.r > 0 ∧
  2 * c.a + c.b = 0 ∧
  (2 - c.a)^2 + (-1 - c.b)^2 = c.r^2 ∧
  |c.a + c.b - 1| / Real.sqrt 2 = c.r

/-- Checks if a line divides the circle into arcs with length ratio 1:2 -/
def dividesCircle (c : Circle) (l : Line) : Prop :=
  ∃ (θ : ℝ), θ = Real.arccos ((1 - l.k * (c.a - 2) - c.b) / (c.r * Real.sqrt (1 + l.k^2))) ∧
              θ / (2 * Real.pi - θ) = 1 / 2

/-- The main theorem stating the properties of the circle and lines -/
theorem circle_and_lines_theorem (c : Circle) (l : Line) :
  satisfiesConditions c →
  dividesCircle c l →
  (c.a = 1 ∧ c.b = -2 ∧ c.r = Real.sqrt 2) ∧
  (l.k = 1 ∨ l.k = 7) := by
  sorry

end NUMINAMATH_CALUDE_circle_and_lines_theorem_l3293_329348


namespace NUMINAMATH_CALUDE_herd_size_l3293_329386

theorem herd_size (first_son_fraction : ℚ) (second_son_fraction : ℚ) (third_son_fraction : ℚ) 
  (village_cows : ℕ) (fourth_son_cows : ℕ) :
  first_son_fraction = 1/3 →
  second_son_fraction = 1/6 →
  third_son_fraction = 3/10 →
  village_cows = 10 →
  fourth_son_cows = 9 →
  ∃ (total_cows : ℕ), 
    total_cows = 95 ∧
    (first_son_fraction + second_son_fraction + third_son_fraction) * total_cows + 
    village_cows + fourth_son_cows = total_cows :=
by sorry

end NUMINAMATH_CALUDE_herd_size_l3293_329386


namespace NUMINAMATH_CALUDE_solve_cupcake_problem_l3293_329345

def cupcake_problem (initial_cupcakes : ℕ) (sold_cupcakes : ℕ) (final_cupcakes : ℕ) : Prop :=
  initial_cupcakes - sold_cupcakes + (final_cupcakes - (initial_cupcakes - sold_cupcakes)) = 20

theorem solve_cupcake_problem :
  cupcake_problem 26 20 26 := by
  sorry

end NUMINAMATH_CALUDE_solve_cupcake_problem_l3293_329345


namespace NUMINAMATH_CALUDE_salary_comparison_l3293_329359

def hansel_initial : ℕ := 30000
def hansel_raise : ℚ := 10 / 100

def gretel_initial : ℕ := 30000
def gretel_raise : ℚ := 15 / 100

def rapunzel_initial : ℕ := 40000
def rapunzel_raise : ℚ := 8 / 100

def rumpelstiltskin_initial : ℕ := 35000
def rumpelstiltskin_raise : ℚ := 12 / 100

def new_salary (initial : ℕ) (raise : ℚ) : ℚ :=
  initial * (1 + raise)

theorem salary_comparison :
  (new_salary gretel_initial gretel_raise - new_salary hansel_initial hansel_raise = 1500) ∧
  (new_salary gretel_initial gretel_raise < new_salary rapunzel_initial rapunzel_raise) ∧
  (new_salary gretel_initial gretel_raise < new_salary rumpelstiltskin_initial rumpelstiltskin_raise) :=
by sorry

end NUMINAMATH_CALUDE_salary_comparison_l3293_329359


namespace NUMINAMATH_CALUDE_set_of_positive_rationals_l3293_329387

theorem set_of_positive_rationals (S : Set ℚ) :
  (∀ a b : ℚ, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a * b) ∈ S) →
  (∀ r : ℚ, (r ∈ S ∧ -r ∉ S ∧ r ≠ 0) ∨ (-r ∈ S ∧ r ∉ S ∧ r ≠ 0) ∨ (r = 0 ∧ r ∉ S ∧ -r ∉ S)) →
  S = {r : ℚ | r > 0} :=
by sorry

end NUMINAMATH_CALUDE_set_of_positive_rationals_l3293_329387


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l3293_329381

theorem distinct_prime_factors_count : 
  (Finset.card (Nat.factors (85 * 87 * 91 * 94)).toFinset) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l3293_329381


namespace NUMINAMATH_CALUDE_age_ratio_change_l3293_329349

/-- Proves the number of years it takes for a parent to become 2.5 times as old as their son -/
theorem age_ratio_change (parent_age son_age : ℕ) (x : ℕ) 
  (h1 : parent_age = 45)
  (h2 : son_age = 15)
  (h3 : parent_age = 3 * son_age) :
  (parent_age + x) = (5/2 : ℚ) * (son_age + x) ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_change_l3293_329349


namespace NUMINAMATH_CALUDE_rahul_salary_l3293_329328

def salary_calculation (salary : ℝ) : ℝ :=
  let after_rent := salary * 0.8
  let after_education := after_rent * 0.9
  let after_clothes := after_education * 0.9
  after_clothes

theorem rahul_salary : ∃ (salary : ℝ), salary_calculation salary = 1377 ∧ salary = 2125 := by
  sorry

end NUMINAMATH_CALUDE_rahul_salary_l3293_329328


namespace NUMINAMATH_CALUDE_sequence_term_40_l3293_329393

theorem sequence_term_40 (n : ℕ+) (a : ℕ+ → ℕ) : 
  (∀ k : ℕ+, a k = 3 * k + 1) → a 13 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_40_l3293_329393


namespace NUMINAMATH_CALUDE_parabola_vertex_l3293_329396

/-- A parabola is defined by the equation y^2 + 10y + 4x + 9 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 10*y + 4*x + 9 = 0

/-- The vertex of a parabola is the point where it turns -/
def is_vertex (x y : ℝ) : Prop :=
  ∀ t : ℝ, parabola_equation (x + t) (y + t) → t = 0

/-- The vertex of the parabola y^2 + 10y + 4x + 9 = 0 is the point (4, -5) -/
theorem parabola_vertex : is_vertex 4 (-5) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3293_329396


namespace NUMINAMATH_CALUDE_cos_angle_relation_l3293_329397

theorem cos_angle_relation (α : ℝ) (h : Real.cos (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 - α) = -(Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_relation_l3293_329397


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_circle_area_ratio_l3293_329363

theorem right_triangle_inscribed_circle_area_ratio 
  (h a r : ℝ) (h_pos : h > 0) (a_pos : a > 0) (r_pos : r > 0) (h_gt_a : h > a) :
  let A := (1/2) * a * Real.sqrt (h^2 - a^2)
  (π * r^2) / A = 4 * π * A / (a + Real.sqrt (h^2 - a^2) + h)^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_circle_area_ratio_l3293_329363


namespace NUMINAMATH_CALUDE_prime_odd_sum_l3293_329326

theorem prime_odd_sum (x y : ℕ) 
  (hx : Nat.Prime x) 
  (hy : Odd y) 
  (heq : x^2 + y = 2005) : 
  x + y = 2003 := by
sorry

end NUMINAMATH_CALUDE_prime_odd_sum_l3293_329326


namespace NUMINAMATH_CALUDE_only_rainbow_statement_correct_l3293_329310

/-- Represents the conditions for seeing a rainbow --/
structure RainbowConditions :=
  (sunlight : Bool)
  (rain : Bool)
  (observer_position : ℝ × ℝ × ℝ)

/-- Represents the outcome of a coin flip --/
inductive CoinFlip
  | Heads
  | Tails

/-- Represents the precipitation data for a city --/
structure PrecipitationData :=
  (average : ℝ)
  (variance : ℝ)

/-- The set of statements about random events and statistical concepts --/
inductive Statement
  | RainbowRandomEvent
  | AircraftRandomSampling
  | CoinFlipDeterministic
  | PrecipitationStability

/-- Function to determine if seeing a rainbow is random given the conditions --/
def is_rainbow_random (conditions : RainbowConditions) : Prop :=
  ∃ (c1 c2 : RainbowConditions), c1 ≠ c2 ∧ 
    (c1.sunlight ∧ c1.rain) ∧ (c2.sunlight ∧ c2.rain) ∧ 
    c1.observer_position ≠ c2.observer_position

/-- Function to determine if a statement is correct --/
def is_correct_statement (s : Statement) : Prop :=
  match s with
  | Statement.RainbowRandomEvent => ∀ c, is_rainbow_random c
  | _ => False

/-- Theorem stating that only the rainbow statement is correct --/
theorem only_rainbow_statement_correct :
  ∀ s, is_correct_statement s ↔ s = Statement.RainbowRandomEvent :=
sorry

end NUMINAMATH_CALUDE_only_rainbow_statement_correct_l3293_329310


namespace NUMINAMATH_CALUDE_platform_length_l3293_329320

/-- The length of a platform crossed by two trains moving in opposite directions -/
theorem platform_length 
  (x y : ℝ) -- lengths of trains A and B in meters
  (p q : ℝ) -- speeds of trains A and B in km/h
  (t : ℝ) -- time taken to cross the platform in seconds
  (h_positive : x > 0 ∧ y > 0 ∧ p > 0 ∧ q > 0 ∧ t > 0) -- All values are positive
  : ∃ (L : ℝ), L = (p + q) * (5 * t / 18) - (x + y) :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3293_329320


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3293_329395

theorem inequality_solution_set (x : ℝ) : (x - 2) * (3 - x) > 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3293_329395


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l3293_329339

theorem sqrt_product_plus_one : 
  Real.sqrt ((26 : ℝ) * 25 * 24 * 23 + 1) = 599 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l3293_329339


namespace NUMINAMATH_CALUDE_building_residents_contradiction_l3293_329374

theorem building_residents_contradiction (chess : ℕ) (arkhangelsk : ℕ) (airplane : ℕ)
  (chess_airplane : ℕ) (arkhangelsk_airplane : ℕ) (chess_arkhangelsk : ℕ)
  (chess_arkhangelsk_airplane : ℕ) :
  chess = 25 →
  arkhangelsk = 30 →
  airplane = 28 →
  chess_airplane = 18 →
  arkhangelsk_airplane = 17 →
  chess_arkhangelsk = 16 →
  chess_arkhangelsk_airplane = 15 →
  chess + arkhangelsk + airplane - chess_arkhangelsk - chess_airplane - arkhangelsk_airplane + chess_arkhangelsk_airplane > 45 :=
by sorry

end NUMINAMATH_CALUDE_building_residents_contradiction_l3293_329374


namespace NUMINAMATH_CALUDE_team_probability_l3293_329312

/-- Given 27 players randomly split into 3 teams of 9 each, with Zack, Mihir, and Andrew on different teams,
    the probability that Zack and Andrew are on the same team is 8/17. -/
theorem team_probability (total_players : Nat) (teams : Nat) (players_per_team : Nat)
  (h1 : total_players = 27)
  (h2 : teams = 3)
  (h3 : players_per_team = 9)
  (h4 : total_players = teams * players_per_team)
  (zack mihir andrew : Nat)
  (h5 : zack ≠ mihir)
  (h6 : mihir ≠ andrew)
  (h7 : zack ≠ andrew) :
  (8 : ℚ) / 17 = (players_per_team - 1 : ℚ) / (total_players - 2 * players_per_team) :=
sorry

end NUMINAMATH_CALUDE_team_probability_l3293_329312


namespace NUMINAMATH_CALUDE_inequality_proof_l3293_329336

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * a) / (a^2 + b * c) + (2 * b) / (b^2 + c * a) + (2 * c) / (c^2 + a * b) ≤
  a / (b * c) + b / (c * a) + c / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3293_329336


namespace NUMINAMATH_CALUDE_milk_sharing_problem_l3293_329354

/-- Given a total amount of milk and a difference between two people's consumption,
    calculate the amount consumed by the person drinking more. -/
def calculate_larger_share (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

/-- Proof that given 2100 ml of milk shared between two people,
    where one drinks 200 ml more than the other,
    the person drinking more consumes 1150 ml. -/
theorem milk_sharing_problem :
  calculate_larger_share 2100 200 = 1150 := by
  sorry

end NUMINAMATH_CALUDE_milk_sharing_problem_l3293_329354


namespace NUMINAMATH_CALUDE_sci_fi_readers_l3293_329368

theorem sci_fi_readers (total : ℕ) (literary : ℕ) (both : ℕ) : 
  total = 650 → literary = 550 → both = 150 → 
  total = literary + (total - literary + both) - both :=
by
  sorry

#check sci_fi_readers

end NUMINAMATH_CALUDE_sci_fi_readers_l3293_329368


namespace NUMINAMATH_CALUDE_cousin_arrangement_count_l3293_329331

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- There are 4 rooms -/
def num_rooms : ℕ := 4

/-- The theorem stating the number of ways to arrange the cousins -/
theorem cousin_arrangement_count :
  distribute num_cousins num_rooms = 51 := by sorry

end NUMINAMATH_CALUDE_cousin_arrangement_count_l3293_329331


namespace NUMINAMATH_CALUDE_jennys_grade_is_95_l3293_329347

-- Define the grades as natural numbers
def jennys_grade : ℕ := sorry
def jasons_grade : ℕ := sorry
def bobs_grade : ℕ := sorry

-- State the conditions
axiom condition1 : jasons_grade = jennys_grade - 25
axiom condition2 : bobs_grade = jasons_grade / 2
axiom condition3 : bobs_grade = 35

-- Theorem to prove
theorem jennys_grade_is_95 : jennys_grade = 95 := by sorry

end NUMINAMATH_CALUDE_jennys_grade_is_95_l3293_329347


namespace NUMINAMATH_CALUDE_min_pouches_is_sixty_l3293_329389

/-- Represents the number of gold coins Flint has. -/
def total_coins : ℕ := 60

/-- Represents the possible number of sailors among whom the coins might be distributed. -/
def possible_sailors : List ℕ := [2, 3, 4, 5]

/-- Defines a valid distribution as one where each sailor receives an equal number of coins. -/
def is_valid_distribution (num_pouches : ℕ) : Prop :=
  ∀ n ∈ possible_sailors, (total_coins / num_pouches) * n = total_coins

/-- States that the number of pouches is minimal if no smaller number satisfies the distribution criteria. -/
def is_minimal (num_pouches : ℕ) : Prop :=
  is_valid_distribution num_pouches ∧
  ∀ k < num_pouches, ¬is_valid_distribution k

/-- The main theorem stating that 60 is the minimum number of pouches required for valid distribution. -/
theorem min_pouches_is_sixty :
  is_minimal total_coins :=
sorry

end NUMINAMATH_CALUDE_min_pouches_is_sixty_l3293_329389


namespace NUMINAMATH_CALUDE_distance_to_left_focus_l3293_329309

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the hyperbola C₂
def C₂ (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the point P as the intersection of C₁ and C₂ in the first quadrant
def P : ℝ × ℝ := sorry

-- State that P satisfies both C₁ and C₂
axiom P_on_C₁ : C₁ P.1 P.2
axiom P_on_C₂ : C₂ P.1 P.2

-- State that P is in the first quadrant
axiom P_first_quadrant : P.1 > 0 ∧ P.2 > 0

-- Define the left focus of the ellipse
def left_focus : ℝ × ℝ := sorry

-- Theorem stating the distance from P to the left focus is 4
theorem distance_to_left_focus :
  Real.sqrt ((P.1 - left_focus.1)^2 + (P.2 - left_focus.2)^2) = 4 := by sorry

end NUMINAMATH_CALUDE_distance_to_left_focus_l3293_329309


namespace NUMINAMATH_CALUDE_four_digit_divisible_count_l3293_329324

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_all (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0

theorem four_digit_divisible_count :
  ∃! (s : Finset ℕ), s.card = 4 ∧
  (∀ n : ℕ, n ∈ s ↔ (is_four_digit n ∧ divisible_by_all n)) :=
sorry

end NUMINAMATH_CALUDE_four_digit_divisible_count_l3293_329324


namespace NUMINAMATH_CALUDE_root_product_expression_l3293_329357

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) 
  (h1 : α^2 + p*α + 2 = 0) 
  (h2 : β^2 + p*β + 2 = 0)
  (h3 : γ^2 + q*γ + 2 = 0)
  (h4 : δ^2 + q*δ + 2 = 0) :
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = -2*(p^2 - q^2) + 4 := by
  sorry

end NUMINAMATH_CALUDE_root_product_expression_l3293_329357


namespace NUMINAMATH_CALUDE_marble_difference_l3293_329304

theorem marble_difference (e d : ℕ) (h1 : e > d) (h2 : e = (d - 8) + 30) : e - d = 22 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l3293_329304


namespace NUMINAMATH_CALUDE_x4_plus_y4_l3293_329329

theorem x4_plus_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x^4 + y^4 = 175 := by
  sorry

end NUMINAMATH_CALUDE_x4_plus_y4_l3293_329329


namespace NUMINAMATH_CALUDE_num_true_propositions_even_l3293_329330

/-- A proposition type representing a logical statement. -/
structure Proposition : Type :=
  (is_true : Bool)

/-- A set of four related propositions (original, converse, inverse, and contrapositive). -/
structure RelatedPropositions : Type :=
  (original : Proposition)
  (converse : Proposition)
  (inverse : Proposition)
  (contrapositive : Proposition)

/-- The number of true propositions in a set of related propositions. -/
def num_true_propositions (rp : RelatedPropositions) : Nat :=
  (if rp.original.is_true then 1 else 0) +
  (if rp.converse.is_true then 1 else 0) +
  (if rp.inverse.is_true then 1 else 0) +
  (if rp.contrapositive.is_true then 1 else 0)

/-- Theorem stating that the number of true propositions in a set of related propositions
    can only be 0, 2, or 4. -/
theorem num_true_propositions_even (rp : RelatedPropositions) :
  num_true_propositions rp = 0 ∨ num_true_propositions rp = 2 ∨ num_true_propositions rp = 4 :=
by sorry

end NUMINAMATH_CALUDE_num_true_propositions_even_l3293_329330


namespace NUMINAMATH_CALUDE_total_cat_food_cases_l3293_329384

/-- Represents the number of cases of cat food sold during a sale --/
def catFoodSale (first_group : Nat) (second_group : Nat) (third_group : Nat)
  (first_group_cases : Nat) (second_group_cases : Nat) (third_group_cases : Nat) : Nat :=
  first_group * first_group_cases + second_group * second_group_cases + third_group * third_group_cases

/-- Theorem stating that the total number of cases of cat food sold is 40 --/
theorem total_cat_food_cases : 
  catFoodSale 8 4 8 3 2 1 = 40 := by
  sorry

#check total_cat_food_cases

end NUMINAMATH_CALUDE_total_cat_food_cases_l3293_329384


namespace NUMINAMATH_CALUDE_function_value_inequality_l3293_329392

theorem function_value_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) (h2 : ∀ x, deriv f x > 1) :
  f 3 > f 1 + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_inequality_l3293_329392


namespace NUMINAMATH_CALUDE_mashed_potatoes_tomatoes_difference_l3293_329378

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 144

/-- The number of students who suggested bacon -/
def bacon : ℕ := 467

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 79

/-- The theorem stating the difference between the number of students who suggested
    mashed potatoes and those who suggested tomatoes -/
theorem mashed_potatoes_tomatoes_difference :
  mashed_potatoes - tomatoes = 65 := by sorry

end NUMINAMATH_CALUDE_mashed_potatoes_tomatoes_difference_l3293_329378


namespace NUMINAMATH_CALUDE_ninety_percent_of_600_equals_fifty_percent_of_x_l3293_329342

theorem ninety_percent_of_600_equals_fifty_percent_of_x (x : ℝ) :
  (90 / 100) * 600 = (50 / 100) * x → x = 1080 := by
sorry

end NUMINAMATH_CALUDE_ninety_percent_of_600_equals_fifty_percent_of_x_l3293_329342


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3293_329315

theorem sqrt_sum_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  Real.sqrt (x / 2) + Real.sqrt (y / 2) ≤ Real.sqrt (x + y) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3293_329315
