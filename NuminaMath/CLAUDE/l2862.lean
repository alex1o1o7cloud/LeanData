import Mathlib

namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2862_286218

/-- 
Given a point P with coordinates (-5, 3) in the Cartesian coordinate system,
prove that its coordinates with respect to the origin are (-5, 3).
-/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-5, 3)
  P = (-5, 3) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2862_286218


namespace NUMINAMATH_CALUDE_largest_quantity_l2862_286240

def A : ℚ := 3003 / 3002 + 3003 / 3004
def B : ℚ := 3003 / 3004 + 3005 / 3004
def C : ℚ := 3004 / 3003 + 3004 / 3005

theorem largest_quantity : A > B ∧ A > C := by sorry

end NUMINAMATH_CALUDE_largest_quantity_l2862_286240


namespace NUMINAMATH_CALUDE_range_of_a_l2862_286204

/-- Given that for any x ≥ 1, ln x - a(1 - 1/x) ≥ 0, prove that a ≤ 1 -/
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → Real.log x - a * (1 - 1/x) ≥ 0) → 
  a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2862_286204


namespace NUMINAMATH_CALUDE_work_completion_time_l2862_286280

theorem work_completion_time (a_days b_days : ℝ) (work_left : ℝ) : 
  a_days > 0 → b_days > 0 → 
  work_left = 0.7666666666666666 →
  (1 / a_days + 1 / b_days) * (1 - work_left) = 1 / 2 := by
  sorry

#check work_completion_time 15 20 0.7666666666666666

end NUMINAMATH_CALUDE_work_completion_time_l2862_286280


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l2862_286224

-- Define quadrilaterals
structure Quadrilateral :=
  (is_rhombus : Bool)
  (is_parallelogram : Bool)

-- The given statement (not used in the proof, but included for completeness)
axiom rhombus_implies_parallelogram :
  ∀ q : Quadrilateral, q.is_rhombus → q.is_parallelogram

-- Theorem to prove
theorem converse_and_inverse_false :
  (∃ q : Quadrilateral, q.is_parallelogram ∧ ¬q.is_rhombus) ∧
  (∃ q : Quadrilateral, ¬q.is_rhombus ∧ q.is_parallelogram) := by
  sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l2862_286224


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2862_286298

theorem sum_of_coefficients : 
  let p (x : ℝ) := -3*(x^8 - x^5 + 2*x^3 - 6) + 5*(x^4 + 3*x^2) - 4*(x^6 - 5)
  p 1 = 48 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2862_286298


namespace NUMINAMATH_CALUDE_purely_imaginary_and_circle_l2862_286200

-- Define the complex number z
def z (a : ℝ) : ℂ := a * (1 + Complex.I) - 2 * Complex.I

-- State the theorem
theorem purely_imaginary_and_circle (a : ℝ) :
  (∃ b : ℝ, z a = Complex.I * b) →
  (a = 2 ∧ ∀ w : ℂ, Complex.abs w = 3 ↔ w.re ^ 2 + w.im ^ 2 = 3 ^ 2) :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_and_circle_l2862_286200


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2862_286212

theorem arithmetic_calculation : 6 * (5 - 2) + 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2862_286212


namespace NUMINAMATH_CALUDE_least_number_remainder_l2862_286291

theorem least_number_remainder : ∃ (r : ℕ), r > 0 ∧ 386 % 35 = r ∧ 386 % 11 = r := by
  sorry

end NUMINAMATH_CALUDE_least_number_remainder_l2862_286291


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l2862_286266

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 8

-- State the theorem
theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  -- Assuming f(1) < 0 and f(2) > 0
  have h1 : f 1 < 0 := by sorry
  have h2 : f 2 > 0 := by sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l2862_286266


namespace NUMINAMATH_CALUDE_extra_time_with_decreased_speed_l2862_286208

theorem extra_time_with_decreased_speed
  (original_time : ℝ)
  (speed_decrease_percentage : ℝ)
  (h1 : original_time = 40)
  (h2 : speed_decrease_percentage = 20) :
  let new_time := original_time / (1 - speed_decrease_percentage / 100)
  new_time - original_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_extra_time_with_decreased_speed_l2862_286208


namespace NUMINAMATH_CALUDE_todds_profit_l2862_286237

/-- Calculates Todd's remaining money after his snow cone business venture -/
def todds_remaining_money (borrowed : ℕ) (repay : ℕ) (ingredients_cost : ℕ) 
  (num_sold : ℕ) (price_per_cone : ℚ) : ℚ :=
  let total_sales := num_sold * price_per_cone
  let remaining := total_sales - repay
  remaining

/-- Proves that Todd's remaining money is $40 after his snow cone business venture -/
theorem todds_profit : 
  todds_remaining_money 100 110 75 200 (75/100) = 40 := by
  sorry

end NUMINAMATH_CALUDE_todds_profit_l2862_286237


namespace NUMINAMATH_CALUDE_students_not_in_biology_l2862_286220

theorem students_not_in_biology (total_students : ℕ) (enrolled_percentage : ℚ) 
  (h1 : total_students = 880)
  (h2 : enrolled_percentage = 40 / 100) :
  (total_students : ℚ) * (1 - enrolled_percentage) = 528 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l2862_286220


namespace NUMINAMATH_CALUDE_class_preferences_l2862_286279

theorem class_preferences (total_students : ℕ) (men_math : ℕ) (men_lit : ℕ) (neither : ℕ) 
  (total_men : ℕ) (both : ℕ) (only_math : ℕ) 
  (h1 : total_students = 35)
  (h2 : men_math = 7)
  (h3 : men_lit = 6)
  (h4 : neither = 13)
  (h5 : total_men = 16)
  (h6 : both = 5)
  (h7 : only_math = 11) : 
  (∃ (men_both women_only_lit : ℕ),
    men_both = 2 ∧ 
    women_only_lit = 6 ∧
    men_both ≤ men_math ∧ 
    men_both ≤ men_lit ∧
    men_math - men_both + men_both + men_lit - men_both + (neither - (total_students - total_men)) = total_men ∧
    only_math + both + women_only_lit + neither = total_students) :=
by
  sorry

end NUMINAMATH_CALUDE_class_preferences_l2862_286279


namespace NUMINAMATH_CALUDE_amy_pencils_before_l2862_286286

/-- The number of pencils Amy bought at the school store -/
def pencils_bought : ℕ := 7

/-- The total number of pencils Amy has now -/
def total_pencils : ℕ := 10

/-- The number of pencils Amy had before buying more -/
def pencils_before : ℕ := total_pencils - pencils_bought

theorem amy_pencils_before : pencils_before = 3 := by
  sorry

end NUMINAMATH_CALUDE_amy_pencils_before_l2862_286286


namespace NUMINAMATH_CALUDE_dragon_rope_problem_l2862_286263

theorem dragon_rope_problem (a b c : ℕ) (h_prime : Nat.Prime c) :
  let tower_radius : ℝ := 10
  let rope_length : ℝ := 25
  let height_difference : ℝ := 3
  let rope_touching_tower : ℝ := (a - Real.sqrt b) / c
  (tower_radius > 0 ∧ rope_length > tower_radius ∧ height_difference > 0 ∧
   rope_touching_tower > 0 ∧ rope_touching_tower < rope_length) →
  a + b + c = 352 :=
by sorry

end NUMINAMATH_CALUDE_dragon_rope_problem_l2862_286263


namespace NUMINAMATH_CALUDE_taxi_ride_cost_l2862_286246

theorem taxi_ride_cost (uber_cost lyft_cost taxi_cost tip_percentage : ℝ) : 
  uber_cost = lyft_cost + 3 →
  lyft_cost = taxi_cost + 4 →
  uber_cost = 22 →
  tip_percentage = 0.2 →
  taxi_cost + (tip_percentage * taxi_cost) = 18 := by
sorry

end NUMINAMATH_CALUDE_taxi_ride_cost_l2862_286246


namespace NUMINAMATH_CALUDE_largest_x_and_ratio_l2862_286292

theorem largest_x_and_ratio (a b c d : ℤ) (x : ℝ) : 
  (7 * x / 8 + 1 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (x ≤ (-8 + 4 * Real.sqrt 15) / 7) →
  (x = (-8 + 4 * Real.sqrt 15) / 7 → a = -8 ∧ b = 4 ∧ c = 15 ∧ d = 7) →
  (a * c * d / b = -210) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_and_ratio_l2862_286292


namespace NUMINAMATH_CALUDE_smallest_percentage_both_l2862_286290

/-- The smallest possible percentage of people eating both ice cream and chocolate in a town -/
theorem smallest_percentage_both (ice_cream_eaters chocolate_eaters : ℝ) 
  (h_ice_cream : ice_cream_eaters = 0.9)
  (h_chocolate : chocolate_eaters = 0.8) :
  ∃ (both : ℝ), both ≥ 0.7 ∧ 
    ∀ (x : ℝ), x ≥ 0 ∧ x ≤ 1 ∧ ice_cream_eaters + chocolate_eaters - x ≤ 1 → x ≥ both := by
  sorry


end NUMINAMATH_CALUDE_smallest_percentage_both_l2862_286290


namespace NUMINAMATH_CALUDE_oil_leak_during_work_l2862_286245

/-- The amount of oil leaked while engineers were working is equal to the difference between the total oil leak and the initial oil leak. -/
theorem oil_leak_during_work (initial_leak total_leak : ℕ) (h : initial_leak = 6522 ∧ total_leak = 11687) :
  total_leak - initial_leak = 5165 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_during_work_l2862_286245


namespace NUMINAMATH_CALUDE_monthly_salary_calculation_l2862_286267

def monthly_salary (rent : ℚ) (savings : ℚ) : ℚ :=
  let food := (5 : ℚ) / 9 * rent
  let mortgage := 5 * food
  let utilities := (1 : ℚ) / 5 * mortgage
  let transportation := (1 : ℚ) / 3 * food
  let insurance := (2 : ℚ) / 3 * utilities
  let healthcare := (3 : ℚ) / 8 * food
  let car_maintenance := (1 : ℚ) / 4 * transportation
  let taxes := (4 : ℚ) / 9 * savings
  rent + food + mortgage + utilities + transportation + insurance + healthcare + car_maintenance + savings + taxes

theorem monthly_salary_calculation (rent savings : ℚ) :
  monthly_salary rent savings = rent + (5 : ℚ) / 9 * rent + 5 * ((5 : ℚ) / 9 * rent) +
    (1 : ℚ) / 5 * (5 * ((5 : ℚ) / 9 * rent)) + (1 : ℚ) / 3 * ((5 : ℚ) / 9 * rent) +
    (2 : ℚ) / 3 * ((1 : ℚ) / 5 * (5 * ((5 : ℚ) / 9 * rent))) +
    (3 : ℚ) / 8 * ((5 : ℚ) / 9 * rent) +
    (1 : ℚ) / 4 * ((1 : ℚ) / 3 * ((5 : ℚ) / 9 * rent)) +
    savings + (4 : ℚ) / 9 * savings :=
by sorry

example : monthly_salary 850 2200 = 8022 + (98 : ℚ) / 100 :=
by sorry

end NUMINAMATH_CALUDE_monthly_salary_calculation_l2862_286267


namespace NUMINAMATH_CALUDE_trig_expression_equals_three_halves_l2862_286215

theorem trig_expression_equals_three_halves (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_three_halves_l2862_286215


namespace NUMINAMATH_CALUDE_students_not_enrolled_l2862_286268

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : french = 65)
  (h3 : german = 50)
  (h4 : both = 25) :
  total - (french + german - both) = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l2862_286268


namespace NUMINAMATH_CALUDE_pen_price_calculation_l2862_286251

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 450 ∧ num_pens = 30 ∧ num_pencils = 75 ∧ pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 10 := by
sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l2862_286251


namespace NUMINAMATH_CALUDE_arcsin_cos_two_pi_thirds_l2862_286206

theorem arcsin_cos_two_pi_thirds : 
  Real.arcsin (Real.cos (2 * π / 3)) = -π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_cos_two_pi_thirds_l2862_286206


namespace NUMINAMATH_CALUDE_tim_added_fourteen_rulers_l2862_286211

/-- Given an initial number of rulers and a final number of rulers,
    calculate the number of rulers added. -/
def rulers_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given 11 initial rulers and 25 final rulers,
    the number of rulers added is 14. -/
theorem tim_added_fourteen_rulers :
  rulers_added 11 25 = 14 := by
  sorry

end NUMINAMATH_CALUDE_tim_added_fourteen_rulers_l2862_286211


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_factorial_l2862_286225

theorem smallest_multiple_of_seven_factorial : ∃ (m : ℕ), m > 0 ∧ 
  (∀ (k : ℕ), k < 7 → ¬(m ∣ Nat.factorial k)) ∧ 
  (m ∣ Nat.factorial 7) ∧
  (∀ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), k < 7 → ¬(n ∣ Nat.factorial k)) ∧ (n ∣ Nat.factorial 7) → m ≤ n) :=
by
  use 5040
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_factorial_l2862_286225


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2862_286230

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | |4*x - 3| < a ∧ a > 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 < 0}

-- Theorem statement
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x, x ∈ B → x ∈ A a) ∧ (∃ x, x ∈ A a ∧ x ∉ B) → 0 < a ∧ a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2862_286230


namespace NUMINAMATH_CALUDE_inequality_proof_l2862_286210

theorem inequality_proof (n : ℕ) : 
  2 * n * (n.factorial / (3 * n).factorial) ^ (1 / (2 * n)) < Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2862_286210


namespace NUMINAMATH_CALUDE_g_sum_zero_l2862_286287

theorem g_sum_zero (f : ℝ → ℝ) : 
  let g := λ x => f x - f (2010 - x)
  ∀ x, g x + g (2010 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_zero_l2862_286287


namespace NUMINAMATH_CALUDE_n2o3_molecular_weight_is_76_02_l2862_286233

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of N2O3 in g/mol -/
def n2o3_molecular_weight : ℝ := nitrogen_weight * nitrogen_count + oxygen_weight * oxygen_count

/-- Theorem stating that the molecular weight of N2O3 is 76.02 g/mol -/
theorem n2o3_molecular_weight_is_76_02 : 
  n2o3_molecular_weight = 76.02 := by sorry

end NUMINAMATH_CALUDE_n2o3_molecular_weight_is_76_02_l2862_286233


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l2862_286229

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l2862_286229


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_with_foci_on_y_axis_l2862_286294

theorem curve_is_hyperbola_with_foci_on_y_axis (θ : Real) 
  (h1 : π < θ ∧ θ < 3*π/2) -- θ is in the third quadrant
  (h2 : ∀ x y : Real, x^2 + y^2 * Real.sin θ = Real.cos θ) -- curve equation
  : ∃ (a b : Real), 
    a > 0 ∧ b > 0 ∧ 
    (∀ x y : Real, y^2 / b^2 - x^2 / a^2 = 1) ∧ -- standard form of hyperbola with foci on y-axis
    (∃ c : Real, c > 0 ∧ c^2 = a^2 + b^2) -- condition for foci on y-axis
  := by sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_with_foci_on_y_axis_l2862_286294


namespace NUMINAMATH_CALUDE_xyz_value_l2862_286271

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  4 * x * y * z = 48 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l2862_286271


namespace NUMINAMATH_CALUDE_star_sqrt3_minus_one_minus_sqrt7_l2862_286234

/-- Custom operation ※ -/
def star (a b : ℝ) : ℝ := (a + 1)^2 - b^2

/-- Theorem stating that (√3-1)※(-√7) = -4 -/
theorem star_sqrt3_minus_one_minus_sqrt7 :
  star (Real.sqrt 3 - 1) (-Real.sqrt 7) = -4 := by
  sorry

end NUMINAMATH_CALUDE_star_sqrt3_minus_one_minus_sqrt7_l2862_286234


namespace NUMINAMATH_CALUDE_sandy_molly_age_ratio_l2862_286202

/-- The ratio of Sandy's age to Molly's age -/
def age_ratio (sandy_age molly_age : ℕ) : ℚ :=
  sandy_age / molly_age

theorem sandy_molly_age_ratio :
  let sandy_age : ℕ := 56
  let molly_age : ℕ := sandy_age + 16
  age_ratio sandy_age molly_age = 7 / 9 := by
  sorry


end NUMINAMATH_CALUDE_sandy_molly_age_ratio_l2862_286202


namespace NUMINAMATH_CALUDE_index_card_area_l2862_286226

theorem index_card_area (length width : ℝ) (h1 : length = 5) (h2 : width = 7) : 
  (∃ side, (side - 2) * width = 21 ∨ length * (side - 2) = 21) →
  (length * (width - 2) = 25 ∨ (length - 2) * width = 25) := by
sorry

end NUMINAMATH_CALUDE_index_card_area_l2862_286226


namespace NUMINAMATH_CALUDE_probability_two_odd_numbers_l2862_286296

/-- A fair eight-sided die -/
def EightSidedDie : Finset ℕ := Finset.range 8 

/-- The set of odd numbers on an eight-sided die -/
def OddNumbers : Finset ℕ := Finset.filter (fun x => x % 2 = 1) EightSidedDie

/-- The probability of an event occurring when rolling two fair eight-sided dice -/
def probability (event : Finset (ℕ × ℕ)) : ℚ :=
  event.card / (EightSidedDie.card * EightSidedDie.card)

/-- The event of rolling two odd numbers -/
def TwoOddNumbers : Finset (ℕ × ℕ) :=
  Finset.product OddNumbers OddNumbers

theorem probability_two_odd_numbers : probability TwoOddNumbers = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_odd_numbers_l2862_286296


namespace NUMINAMATH_CALUDE_three_inverse_mod_191_l2862_286288

theorem three_inverse_mod_191 : ∃ x : ℕ, x < 191 ∧ (3 * x) % 191 = 1 ∧ x = 64 := by sorry

end NUMINAMATH_CALUDE_three_inverse_mod_191_l2862_286288


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l2862_286227

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l2862_286227


namespace NUMINAMATH_CALUDE_root_in_interval_l2862_286274

-- Define the function f(x) = x³ - x - 3
def f (x : ℝ) : ℝ := x^3 - x - 3

-- State the theorem
theorem root_in_interval :
  Continuous f ∧ f 1 < 0 ∧ 0 < f 2 →
  ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l2862_286274


namespace NUMINAMATH_CALUDE_g_solutions_l2862_286260

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the property that g must satisfy
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

-- State the theorem
theorem g_solutions :
  ∀ g : ℝ → ℝ, g_property g →
    (∀ x, g x = 3 * x - 1) ∨ (∀ x, g x = -3 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_g_solutions_l2862_286260


namespace NUMINAMATH_CALUDE_cookies_per_bag_example_l2862_286276

/-- Given the number of chocolate chip cookies, oatmeal cookies, and baggies,
    calculate the number of cookies in each bag. -/
def cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (baggies : ℕ) : ℕ :=
  (chocolate_chip + oatmeal) / baggies

/-- Theorem stating that with 5 chocolate chip cookies, 19 oatmeal cookies,
    and 3 baggies, there are 8 cookies in each bag. -/
theorem cookies_per_bag_example : cookies_per_bag 5 19 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_example_l2862_286276


namespace NUMINAMATH_CALUDE_factorization_equality_l2862_286223

theorem factorization_equality (x y : ℝ) : 
  x^2 - y^2 + 3*x - y + 2 = (x + y + 2)*(x - y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2862_286223


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l2862_286213

/-- Proves that the cost of each adult ticket is $4.50 -/
theorem adult_ticket_cost :
  let student_ticket_price : ℚ := 2
  let total_tickets : ℕ := 20
  let total_income : ℚ := 60
  let student_tickets_sold : ℕ := 12
  let adult_tickets_sold : ℕ := total_tickets - student_tickets_sold
  let adult_ticket_price : ℚ := (total_income - (student_ticket_price * student_tickets_sold)) / adult_tickets_sold
  adult_ticket_price = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l2862_286213


namespace NUMINAMATH_CALUDE_smallest_among_three_l2862_286248

theorem smallest_among_three : 
  min ((-2)^3) (min (-3^2) (-(-1))) = -3^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_among_three_l2862_286248


namespace NUMINAMATH_CALUDE_equal_division_of_trout_l2862_286283

theorem equal_division_of_trout (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) :
  total_trout = 18 →
  num_people = 2 →
  trout_per_person = total_trout / num_people →
  trout_per_person = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_trout_l2862_286283


namespace NUMINAMATH_CALUDE_paint_wall_theorem_l2862_286201

/-- The length of wall that can be painted by a group of boys in a given time -/
def wall_length (num_boys : ℕ) (days : ℝ) (rate : ℝ) : ℝ :=
  num_boys * days * rate

theorem paint_wall_theorem (rate : ℝ) :
  wall_length 8 3.125 rate = 50 →
  wall_length 6 5 rate = 106.67 := by
  sorry

end NUMINAMATH_CALUDE_paint_wall_theorem_l2862_286201


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2862_286278

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 6 + a 11 = 100) : 
  2 * a 7 - a 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2862_286278


namespace NUMINAMATH_CALUDE_wills_earnings_after_deductions_l2862_286289

/-- Calculates Will's earnings after tax deductions for a five-day work week --/
def willsEarnings (monday_wage monday_hours tuesday_wage tuesday_hours
                   wednesday_wage wednesday_hours thursday_friday_wage
                   thursday_friday_hours tax_rate : ℚ) : ℚ :=
  let total_earnings := monday_wage * monday_hours +
                        tuesday_wage * tuesday_hours +
                        wednesday_wage * wednesday_hours +
                        2 * thursday_friday_wage * thursday_friday_hours
  let tax_deduction := tax_rate * total_earnings
  total_earnings - tax_deduction

/-- Theorem stating that Will's earnings after deductions equal $170.72 --/
theorem wills_earnings_after_deductions :
  willsEarnings 8 8 10 2 9 6 7 4 (12/100) = 17072/100 :=
by sorry

end NUMINAMATH_CALUDE_wills_earnings_after_deductions_l2862_286289


namespace NUMINAMATH_CALUDE_inequality_proofs_l2862_286255

theorem inequality_proofs :
  (∀ (a b : ℝ), a > 0 → b > 0 → a^3 + b^3 ≥ a*b^2 + a^2*b) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y > 2 → (1 + y) / x < 2 ∨ (1 + x) / y < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l2862_286255


namespace NUMINAMATH_CALUDE_f_min_at_neg_four_l2862_286207

/-- The quadratic function f(x) = x^2 + 8x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 7

/-- The theorem stating that f(x) has a minimum value of -9 at x = -4 -/
theorem f_min_at_neg_four :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∀ (x : ℝ), f x ≥ f (-4)) ∧
  f (-4) = -9 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_neg_four_l2862_286207


namespace NUMINAMATH_CALUDE_square_side_length_l2862_286236

theorem square_side_length (d : ℝ) (s : ℝ) : d = 2 * Real.sqrt 2 → s * Real.sqrt 2 = d → s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2862_286236


namespace NUMINAMATH_CALUDE_laundry_theorem_l2862_286217

/-- Represents the laundry problem --/
structure LaundryProblem where
  machine_capacity : ℕ  -- in pounds
  shirts_per_pound : ℕ
  pants_pairs_per_pound : ℕ
  shirts_to_wash : ℕ
  loads : ℕ

/-- Calculates the number of pants pairs that can be washed --/
def pants_to_wash (p : LaundryProblem) : ℕ :=
  let total_capacity := p.machine_capacity * p.loads
  let shirt_weight := p.shirts_to_wash / p.shirts_per_pound
  let remaining_capacity := total_capacity - shirt_weight
  remaining_capacity * p.pants_pairs_per_pound

/-- States the theorem for the laundry problem --/
theorem laundry_theorem (p : LaundryProblem) 
  (h1 : p.machine_capacity = 5)
  (h2 : p.shirts_per_pound = 4)
  (h3 : p.pants_pairs_per_pound = 2)
  (h4 : p.shirts_to_wash = 20)
  (h5 : p.loads = 3) :
  pants_to_wash p = 20 := by
  sorry

#eval pants_to_wash { 
  machine_capacity := 5,
  shirts_per_pound := 4,
  pants_pairs_per_pound := 2,
  shirts_to_wash := 20,
  loads := 3
}

end NUMINAMATH_CALUDE_laundry_theorem_l2862_286217


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l2862_286277

theorem smaller_number_in_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  b / a = 11 / 7 ∧
  b - a = 16 →
  a = 28 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l2862_286277


namespace NUMINAMATH_CALUDE_crushing_load_calculation_l2862_286256

theorem crushing_load_calculation (T H L : ℝ) : 
  T = 3 → H = 9 → L = (36 * T^3) / H^3 → L = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_crushing_load_calculation_l2862_286256


namespace NUMINAMATH_CALUDE_equation_solution_l2862_286247

theorem equation_solution : 
  ∀ (x y : ℝ), (16 * x^2 + 1) * (y^2 + 1) = 16 * x * y ↔ 
  ((x = 1/4 ∧ y = 1) ∨ (x = -1/4 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2862_286247


namespace NUMINAMATH_CALUDE_total_paint_used_l2862_286228

-- Define the amount of white paint used
def white_paint : ℕ := 660

-- Define the amount of blue paint used
def blue_paint : ℕ := 6029

-- Theorem stating the total amount of paint used
theorem total_paint_used : white_paint + blue_paint = 6689 := by
  sorry

end NUMINAMATH_CALUDE_total_paint_used_l2862_286228


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2862_286273

/-- The eccentricity of a hyperbola with equation x²/2 - y² = -1 is √3 -/
theorem hyperbola_eccentricity : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/2 - y^2 = -1
  ∃ e : ℝ, e = Real.sqrt 3 ∧ 
    ∀ x y : ℝ, h x y → 
      e = Real.sqrt (1 + (x^2/2)/(y^2)) := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2862_286273


namespace NUMINAMATH_CALUDE_final_number_lower_bound_l2862_286281

/-- Represents a sequence of operations on the blackboard -/
def BlackboardOperation := List (Nat × Nat)

/-- The result of applying a sequence of operations to the initial numbers -/
def applyOperations (n : Nat) (ops : BlackboardOperation) : Nat :=
  sorry

/-- Theorem: The final number after any sequence of operations is at least 4/9 * n^3 -/
theorem final_number_lower_bound (n : Nat) (ops : BlackboardOperation) :
  applyOperations n ops ≥ (4 * n^3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_final_number_lower_bound_l2862_286281


namespace NUMINAMATH_CALUDE_train_length_l2862_286275

/-- The length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) :
  speed = 36 * (1000 / 3600) →
  time = 25.997920166386688 →
  bridge_length = 150 →
  speed * time - bridge_length = 109.97920166386688 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2862_286275


namespace NUMINAMATH_CALUDE_cloud_9_diving_bookings_l2862_286203

/-- Cloud 9 Diving Company bookings problem -/
theorem cloud_9_diving_bookings 
  (total_after_cancellations : ℕ) 
  (group_bookings : ℕ) 
  (cancellation_returns : ℕ) 
  (h1 : total_after_cancellations = 26400)
  (h2 : group_bookings = 16000)
  (h3 : cancellation_returns = 1600) :
  total_after_cancellations + cancellation_returns - group_bookings = 12000 :=
by sorry

end NUMINAMATH_CALUDE_cloud_9_diving_bookings_l2862_286203


namespace NUMINAMATH_CALUDE_sampling_survey_appropriate_l2862_286282

/-- Represents a survey method -/
inductive SurveyMethod
| Census
| SamplingSurvey

/-- Represents the characteristics of a population survey -/
structure PopulationSurvey where
  populationSize : Nat
  needsEfficiency : Bool

/-- Determines the appropriate survey method based on population characteristics -/
def appropriateSurveyMethod (survey : PopulationSurvey) : SurveyMethod :=
  if survey.populationSize > 1000000 && survey.needsEfficiency
  then SurveyMethod.SamplingSurvey
  else SurveyMethod.Census

/-- Theorem: For a large population requiring efficient data collection,
    sampling survey is the appropriate method -/
theorem sampling_survey_appropriate
  (survey : PopulationSurvey)
  (h1 : survey.populationSize > 1000000)
  (h2 : survey.needsEfficiency) :
  appropriateSurveyMethod survey = SurveyMethod.SamplingSurvey :=
by
  sorry


end NUMINAMATH_CALUDE_sampling_survey_appropriate_l2862_286282


namespace NUMINAMATH_CALUDE_field_division_l2862_286293

theorem field_division (total_area smaller_area larger_area : ℝ) : 
  total_area = 700 ∧ 
  smaller_area + larger_area = total_area ∧ 
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 315 :=
by sorry

end NUMINAMATH_CALUDE_field_division_l2862_286293


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2862_286264

def S : Finset ℕ := {1, 2, 3, 4}

def f (k x y z : ℕ) : ℕ := k * x^y - z

theorem max_value_of_expression :
  ∃ (k x y z : ℕ), k ∈ S ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S ∧
    f k x y z = 127 ∧
    ∀ (k' x' y' z' : ℕ), k' ∈ S → x' ∈ S → y' ∈ S → z' ∈ S →
      f k' x' y' z' ≤ 127 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2862_286264


namespace NUMINAMATH_CALUDE_minimum_guests_l2862_286269

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 323) (h2 : max_per_guest = 2) :
  ∃ (min_guests : ℕ), min_guests = 162 ∧ min_guests * max_per_guest ≥ total_food ∧
  ∀ (n : ℕ), n * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end NUMINAMATH_CALUDE_minimum_guests_l2862_286269


namespace NUMINAMATH_CALUDE_asymptote_sum_l2862_286249

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x^3 + A*x^2 + B*x + C = 0 ↔ x = -3 ∨ x = 0 ∨ x = 2) → 
  A + B + C = -5 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l2862_286249


namespace NUMINAMATH_CALUDE_partition_has_all_distances_l2862_286221

-- Define a partition of a metric space into three sets
def Partition (X : Type*) [MetricSpace X] (M₁ M₂ M₃ : Set X) : Prop :=
  (M₁ ∪ M₂ ∪ M₃ = Set.univ) ∧ (M₁ ∩ M₂ = ∅) ∧ (M₁ ∩ M₃ = ∅) ∧ (M₂ ∩ M₃ = ∅)

-- Define the property that a set contains two points with any positive distance
def HasAllDistances (X : Type*) [MetricSpace X] (M : Set X) : Prop :=
  ∀ a : ℝ, a > 0 → ∃ x y : X, x ∈ M ∧ y ∈ M ∧ dist x y = a

-- State the theorem
theorem partition_has_all_distances (X : Type*) [MetricSpace X] (M₁ M₂ M₃ : Set X) 
  (h : Partition X M₁ M₂ M₃) : 
  HasAllDistances X M₁ ∨ HasAllDistances X M₂ ∨ HasAllDistances X M₃ := by
  sorry


end NUMINAMATH_CALUDE_partition_has_all_distances_l2862_286221


namespace NUMINAMATH_CALUDE_even_function_extension_l2862_286261

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- State the theorem
theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_nonneg : ∀ x : ℝ, x ≥ 0 → f x = 2^x + 1) :
  ∀ x : ℝ, x < 0 → f x = 2^(-x) + 1 :=
sorry

end NUMINAMATH_CALUDE_even_function_extension_l2862_286261


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2862_286254

theorem system_solution_ratio (x y a b : ℝ) (h1 : 4 * x - 2 * y = a) 
  (h2 : 6 * y - 12 * x = b) (h3 : b ≠ 0) : a / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2862_286254


namespace NUMINAMATH_CALUDE_diagonals_in_nonagon_l2862_286216

/-- The number of diagonals in a regular nine-sided polygon -/
theorem diagonals_in_nonagon : 
  (let n : ℕ := 9
   let total_connections := n.choose 2
   let num_sides := n
   total_connections - num_sides) = 27 := by
sorry

end NUMINAMATH_CALUDE_diagonals_in_nonagon_l2862_286216


namespace NUMINAMATH_CALUDE_danny_soda_consumption_l2862_286253

theorem danny_soda_consumption (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 →
  (1 - x / 100) + (0.3 + 0.3) = 0.7 →
  x = 90 := by
sorry

end NUMINAMATH_CALUDE_danny_soda_consumption_l2862_286253


namespace NUMINAMATH_CALUDE_min_value_T_l2862_286238

/-- Given a quadratic inequality that holds for all real x, prove the minimum value of T -/
theorem min_value_T (a b c : ℝ) (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) (h2 : a < b) :
  (∀ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' ≥ 0) → a' < b' → 
    (a' + b' + c') / (b' - a') ≥ (a + b + c) / (b - a)) → 
  (a + b + c) / (b - a) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_T_l2862_286238


namespace NUMINAMATH_CALUDE_number_ratio_l2862_286284

theorem number_ratio (A B C : ℚ) (k : ℤ) (h1 : A = 2 * B) (h2 : A = k * C)
  (h3 : (A + B + C) / 3 = 88) (h4 : A - C = 96) : A / C = 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l2862_286284


namespace NUMINAMATH_CALUDE_point_on_curve_l2862_286259

/-- The curve C defined by y = x^3 - 10x + 3 -/
def C : ℝ → ℝ := λ x ↦ x^3 - 10*x + 3

/-- The derivative of curve C -/
def C' : ℝ → ℝ := λ x ↦ 3*x^2 - 10

theorem point_on_curve (x y : ℝ) :
  x < 0 →  -- P is in the second quadrant (x < 0)
  y > 0 →  -- P is in the second quadrant (y > 0)
  y = C x →  -- P lies on the curve C
  C' x = 2 →  -- The slope of the tangent line at P is 2
  x = -2 ∧ y = 15 := by  -- P has coordinates (-2, 15)
  sorry

end NUMINAMATH_CALUDE_point_on_curve_l2862_286259


namespace NUMINAMATH_CALUDE_truncated_pyramid_overlap_l2862_286205

/-- Regular triangular pyramid with planar angle α at the vertex -/
structure RegularTriangularPyramid where
  α : ℝ  -- Planar angle at the vertex

/-- Regular truncated pyramid cut from a regular triangular pyramid -/
structure RegularTruncatedPyramid (p : RegularTriangularPyramid) where

/-- Unfolded development of a regular truncated pyramid -/
def UnfoldedDevelopment (t : RegularTruncatedPyramid p) : Type := sorry

/-- Predicate to check if an unfolded development overlaps itself -/
def is_self_overlapping (d : UnfoldedDevelopment t) : Prop := sorry

theorem truncated_pyramid_overlap (p : RegularTriangularPyramid) 
  (t : RegularTruncatedPyramid p) (d : UnfoldedDevelopment t) :
  is_self_overlapping d ↔ 100 * π / 180 < p.α ∧ p.α < 120 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_truncated_pyramid_overlap_l2862_286205


namespace NUMINAMATH_CALUDE_dishwasher_manager_ratio_l2862_286265

/-- The hourly wages of employees at Joe's Steakhouse -/
structure Wages where
  manager : ℝ
  chef : ℝ
  dishwasher : ℝ

/-- The conditions of the wages at Joe's Steakhouse -/
def wage_conditions (w : Wages) : Prop :=
  w.chef = w.dishwasher * 1.22 ∧
  w.manager = 8.50 ∧
  w.manager = w.chef + 3.315

/-- The theorem stating the ratio of dishwasher to manager wages -/
theorem dishwasher_manager_ratio (w : Wages) :
  wage_conditions w → w.dishwasher / w.manager = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_dishwasher_manager_ratio_l2862_286265


namespace NUMINAMATH_CALUDE_smallest_range_of_four_integers_with_mean_2017_l2862_286297

/-- Given four different positive integers with a mean of 2017, 
    the smallest possible range between the largest and smallest of these integers is 4. -/
theorem smallest_range_of_four_integers_with_mean_2017 :
  ∀ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  (a + b + c + d) / 4 = 2017 →
  (∀ (w x y z : ℕ), 
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 →
    (w + x + y + z) / 4 = 2017 →
    max w (max x (max y z)) - min w (min x (min y z)) ≥ 4) ∧
  (∃ (p q r s : ℕ),
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 →
    (p + q + r + s) / 4 = 2017 →
    max p (max q (max r s)) - min p (min q (min r s)) = 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_range_of_four_integers_with_mean_2017_l2862_286297


namespace NUMINAMATH_CALUDE_checkerboard_interior_probability_l2862_286209

/-- Represents a square checkerboard -/
structure Checkerboard where
  size : ℕ

/-- Calculates the number of squares on the perimeter of the checkerboard -/
def perimeterSquares (board : Checkerboard) : ℕ :=
  4 * board.size - 4

/-- Calculates the total number of squares on the checkerboard -/
def totalSquares (board : Checkerboard) : ℕ :=
  board.size * board.size

/-- Calculates the number of squares not on the perimeter of the checkerboard -/
def interiorSquares (board : Checkerboard) : ℕ :=
  totalSquares board - perimeterSquares board

/-- The probability of choosing a square not on the perimeter -/
def interiorProbability (board : Checkerboard) : ℚ :=
  interiorSquares board / totalSquares board

theorem checkerboard_interior_probability :
  interiorProbability (Checkerboard.mk 10) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_interior_probability_l2862_286209


namespace NUMINAMATH_CALUDE_special_triangle_area_l2862_286222

/-- A triangle with specific properties -/
structure SpecialTriangle where
  /-- The height of the triangle -/
  height : ℝ
  /-- The smaller part of the base -/
  smaller_base : ℝ
  /-- The ratio of the divided angle -/
  angle_ratio : ℝ
  /-- The height is 2 -/
  height_is_two : height = 2
  /-- The smaller part of the base is 1 -/
  smaller_base_is_one : smaller_base = 1
  /-- The height divides the angle in the ratio 2:1 -/
  angle_ratio_is_two_to_one : angle_ratio = 2/1

/-- The area of the SpecialTriangle is 11/3 -/
theorem special_triangle_area (t : SpecialTriangle) : 
  (1/2) * t.height * (t.smaller_base + (8/3)) = 11/3 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_area_l2862_286222


namespace NUMINAMATH_CALUDE_total_cost_theorem_l2862_286239

/-- The total cost of buying thermometers and masks -/
def total_cost (a b : ℝ) : ℝ := 3 * a + b

/-- Theorem: The total cost of buying 3 thermometers at 'a' yuan each
    and 'b' masks at 1 yuan each is equal to (3a + b) yuan -/
theorem total_cost_theorem (a b : ℝ) :
  total_cost a b = 3 * a + b := by sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l2862_286239


namespace NUMINAMATH_CALUDE_towel_shrinkage_l2862_286272

/-- If a rectangle's breadth decreases by 10% and its area decreases by 28%, then its length decreases by 20%. -/
theorem towel_shrinkage (L B : ℝ) (L' B' : ℝ) (h1 : B' = 0.9 * B) (h2 : L' * B' = 0.72 * L * B) :
  L' = 0.8 * L := by
  sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l2862_286272


namespace NUMINAMATH_CALUDE_candy_necklace_problem_l2862_286270

/-- Candy necklace problem -/
theorem candy_necklace_problem (blocks : ℕ) (pieces_per_block : ℕ) (people : ℕ) 
  (h1 : blocks = 3) 
  (h2 : pieces_per_block = 30) 
  (h3 : people = 9) :
  (blocks * pieces_per_block) / people = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_necklace_problem_l2862_286270


namespace NUMINAMATH_CALUDE_geometric_sequence_logarithm_l2862_286235

/-- Given a geometric sequence {a_n} with common ratio -√2, 
    prove that ln(a_{2017})^2 - ln(a_{2016})^2 = ln(2) -/
theorem geometric_sequence_logarithm (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = a n * (-Real.sqrt 2)) :
  (Real.log (a 2017))^2 - (Real.log (a 2016))^2 = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_logarithm_l2862_286235


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l2862_286252

theorem smallest_number_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < 1572 → ¬((y + 3) % 9 = 0 ∧ (y + 3) % 35 = 0 ∧ (y + 3) % 25 = 0 ∧ (y + 3) % 21 = 0)) ∧
  ((1572 + 3) % 9 = 0 ∧ (1572 + 3) % 35 = 0 ∧ (1572 + 3) % 25 = 0 ∧ (1572 + 3) % 21 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l2862_286252


namespace NUMINAMATH_CALUDE_fraction_problem_l2862_286243

theorem fraction_problem (x : ℚ) : (x * 48 + 15 = 27) → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2862_286243


namespace NUMINAMATH_CALUDE_sum_of_repeated_digit_numbers_theorem_l2862_286250

theorem sum_of_repeated_digit_numbers_theorem :
  ∃ (a b c : ℕ),
    (∃ (d : ℕ), a = d * 11111 ∧ d < 10) ∧
    (∃ (e : ℕ), b = e * 1111 ∧ e < 10) ∧
    (∃ (f : ℕ), c = f * 111 ∧ f < 10) ∧
    (10000 ≤ a + b + c ∧ a + b + c < 100000) ∧
    (∃ (v w x y z : ℕ),
      v < 10 ∧ w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
      v ≠ w ∧ v ≠ x ∧ v ≠ y ∧ v ≠ z ∧
      w ≠ x ∧ w ≠ y ∧ w ≠ z ∧
      x ≠ y ∧ x ≠ z ∧
      y ≠ z ∧
      a + b + c = v * 10000 + w * 1000 + x * 100 + y * 10 + z) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeated_digit_numbers_theorem_l2862_286250


namespace NUMINAMATH_CALUDE_tomatoes_picked_yesterday_l2862_286244

theorem tomatoes_picked_yesterday (initial : Nat) (picked_today : Nat) (left : Nat) :
  initial = 171 →
  picked_today = 30 →
  left = 7 →
  initial - (initial - picked_today - left) - picked_today = 134 := by
sorry

end NUMINAMATH_CALUDE_tomatoes_picked_yesterday_l2862_286244


namespace NUMINAMATH_CALUDE_complex_power_40_l2862_286219

theorem complex_power_40 :
  (Complex.exp (Complex.I * (150 * π / 180)))^40 = -1/2 - Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_40_l2862_286219


namespace NUMINAMATH_CALUDE_trees_cut_l2862_286258

theorem trees_cut (original : ℕ) (died : ℕ) (left : ℕ) (cut : ℕ) : 
  original = 86 → died = 15 → left = 48 → cut = original - died - left → cut = 23 := by
  sorry

end NUMINAMATH_CALUDE_trees_cut_l2862_286258


namespace NUMINAMATH_CALUDE_point_on_line_l2862_286285

/-- A point (x, 3) lies on the straight line joining (1, 5) and (5, -3) if and only if x = 2 -/
theorem point_on_line (x : ℝ) : 
  (3 - 5) / (x - 1) = (-3 - 5) / (5 - 1) ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2862_286285


namespace NUMINAMATH_CALUDE_five_students_two_teachers_arrangement_l2862_286242

/-- The number of ways two teachers can join a fixed line of students -/
def teacher_line_arrangements (num_students : ℕ) (num_teachers : ℕ) : ℕ :=
  (num_students + 1) * (num_students + 2)

/-- Theorem: With 5 students in fixed order and 2 teachers, there are 42 ways to arrange the line -/
theorem five_students_two_teachers_arrangement :
  teacher_line_arrangements 5 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_five_students_two_teachers_arrangement_l2862_286242


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l2862_286232

-- Define the vertices of the tetrahedron
def A₁ : ℝ × ℝ × ℝ := (1, 5, -7)
def A₂ : ℝ × ℝ × ℝ := (-3, 6, 3)
def A₃ : ℝ × ℝ × ℝ := (-2, 7, 3)
def A₄ : ℝ × ℝ × ℝ := (-4, 8, -12)

-- Define a function to calculate the volume of a tetrahedron
def tetrahedronVolume (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the height of a tetrahedron
def tetrahedronHeight (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem stating the volume and height of the tetrahedron
theorem tetrahedron_properties :
  tetrahedronVolume A₁ A₂ A₃ A₄ = 17.5 ∧
  tetrahedronHeight A₁ A₂ A₃ A₄ = 7 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l2862_286232


namespace NUMINAMATH_CALUDE_sum_of_pairs_l2862_286214

theorem sum_of_pairs : (5 + 2) + (8 + 6) + (4 + 7) + (3 + 2) = 37 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_pairs_l2862_286214


namespace NUMINAMATH_CALUDE_rent_calculation_l2862_286231

theorem rent_calculation (monthly_earnings : ℝ) 
  (h1 : monthly_earnings * 0.07 + monthly_earnings * 0.5 + 817 = monthly_earnings) : 
  monthly_earnings * 0.07 = 133 := by
  sorry

end NUMINAMATH_CALUDE_rent_calculation_l2862_286231


namespace NUMINAMATH_CALUDE_jessica_age_when_justin_born_jessica_age_proof_l2862_286295

theorem jessica_age_when_justin_born (justin_current_age : ℕ) (james_jessica_age_diff : ℕ) (james_future_age : ℕ) (years_to_future : ℕ) : ℕ :=
  let james_current_age := james_future_age - years_to_future
  let jessica_current_age := james_current_age - james_jessica_age_diff
  jessica_current_age - justin_current_age

/- Proof that Jessica was 6 years old when Justin was born -/
theorem jessica_age_proof :
  jessica_age_when_justin_born 26 7 44 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_jessica_age_when_justin_born_jessica_age_proof_l2862_286295


namespace NUMINAMATH_CALUDE_derivative_of_y_l2862_286257

-- Define the function y
def y (x a b c : ℝ) : ℝ := (x - a) * (x - b) * (x - c)

-- State the theorem
theorem derivative_of_y (x a b c : ℝ) :
  deriv (fun x => y x a b c) x = 3 * x^2 - 2 * (a + b + c) * x + (a * b + a * c + b * c) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l2862_286257


namespace NUMINAMATH_CALUDE_octal_subtraction_theorem_l2862_286299

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Addition in base 8 --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtraction in base 8 --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Conversion from octal to decimal --/
def from_octal (n : OctalNumber) : ℕ :=
  sorry

theorem octal_subtraction_theorem :
  octal_sub (to_octal 52) (to_octal 27) = to_octal 25 :=
sorry

end NUMINAMATH_CALUDE_octal_subtraction_theorem_l2862_286299


namespace NUMINAMATH_CALUDE_quadratic_roots_prime_sum_of_digits_l2862_286241

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The main theorem -/
theorem quadratic_roots_prime_sum_of_digits (c : ℕ) :
  (∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ 
    p ≠ q ∧
    p * q = c ∧
    p + q = 85 ∧
    ∀ x : ℝ, x^2 - 85*x + c = 0 ↔ x = p ∨ x = q) →
  sum_of_digits c = 13 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_prime_sum_of_digits_l2862_286241


namespace NUMINAMATH_CALUDE_parallel_vectors_problem_l2862_286262

/-- Given two vectors a and b in ℝ², where a = (1, -2), |b| = 2√5, and a is parallel to b,
    prove that b = (2, -4) or b = (-2, 4) -/
theorem parallel_vectors_problem (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, -2)
  (‖b‖ = 2 * Real.sqrt 5) →
  (∃ (k : ℝ), b = k • a) →
  (b = (2, -4) ∨ b = (-2, 4)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_problem_l2862_286262
