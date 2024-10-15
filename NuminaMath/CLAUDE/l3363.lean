import Mathlib

namespace NUMINAMATH_CALUDE_calculator_reciprocal_l3363_336348

theorem calculator_reciprocal (x : ℝ) :
  (1 / (1/x - 1)) - 1 = -0.75 → x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_calculator_reciprocal_l3363_336348


namespace NUMINAMATH_CALUDE_andrews_blue_balloons_l3363_336330

/-- Given information about Andrew's balloons, prove the number of blue balloons. -/
theorem andrews_blue_balloons :
  ∀ (total_balloons remaining_balloons purple_balloons : ℕ),
    total_balloons = 2 * remaining_balloons →
    remaining_balloons = 378 →
    purple_balloons = 453 →
    total_balloons - purple_balloons = 303 := by
  sorry

end NUMINAMATH_CALUDE_andrews_blue_balloons_l3363_336330


namespace NUMINAMATH_CALUDE_integral_sum_reciprocal_and_semicircle_l3363_336380

open Real MeasureTheory

theorem integral_sum_reciprocal_and_semicircle :
  ∫ x in (1 : ℝ)..3, (1 / x + Real.sqrt (1 - (x - 2)^2)) = Real.log 3 + π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sum_reciprocal_and_semicircle_l3363_336380


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3363_336362

theorem polynomial_division_remainder (x : ℤ) : 
  x^1010 % ((x^2 - 1) * (x + 1)) = 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3363_336362


namespace NUMINAMATH_CALUDE_square_eq_neg_two_i_implies_a_eq_one_coordinates_of_z_over_one_plus_i_l3363_336375

-- Define the complex number z
def z (a : ℝ) : ℂ := a - Complex.I

-- Theorem 1
theorem square_eq_neg_two_i_implies_a_eq_one :
  ∀ a : ℝ, (z a)^2 = -2 * Complex.I → a = 1 := by sorry

-- Theorem 2
theorem coordinates_of_z_over_one_plus_i :
  let z : ℂ := z 2
  (z / (1 + Complex.I)).re = 1/2 ∧ (z / (1 + Complex.I)).im = -3/2 := by sorry

end NUMINAMATH_CALUDE_square_eq_neg_two_i_implies_a_eq_one_coordinates_of_z_over_one_plus_i_l3363_336375


namespace NUMINAMATH_CALUDE_max_notebook_price_l3363_336352

def entrance_fee : ℕ := 3
def total_budget : ℕ := 160
def num_notebooks : ℕ := 15
def tax_rate : ℚ := 8 / 100

theorem max_notebook_price :
  ∃ (price : ℕ),
    price ≤ 9 ∧
    (price : ℚ) * (1 + tax_rate) * num_notebooks + entrance_fee ≤ total_budget ∧
    ∀ (p : ℕ), p > price →
      (p : ℚ) * (1 + tax_rate) * num_notebooks + entrance_fee > total_budget :=
by sorry

end NUMINAMATH_CALUDE_max_notebook_price_l3363_336352


namespace NUMINAMATH_CALUDE_product_of_divisors_60_has_three_prime_factors_l3363_336341

def divisors (n : ℕ) : Finset ℕ :=
  sorry

def product_of_divisors (n : ℕ) : ℕ :=
  (divisors n).prod id

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  sorry

theorem product_of_divisors_60_has_three_prime_factors :
  num_distinct_prime_factors (product_of_divisors 60) = 3 :=
sorry

end NUMINAMATH_CALUDE_product_of_divisors_60_has_three_prime_factors_l3363_336341


namespace NUMINAMATH_CALUDE_geometric_sum_problem_l3363_336314

-- Define the sum of a geometric sequence
def GeometricSum (n : ℕ) := ℝ

-- State the theorem
theorem geometric_sum_problem :
  ∀ (S : ℕ → ℝ),
  (S 2 = 4) →
  (S 4 = 6) →
  (S 6 = 7) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_problem_l3363_336314


namespace NUMINAMATH_CALUDE_sum_of_roots_l3363_336390

theorem sum_of_roots (k c : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  (4 * x₁^2 - k * x₁ = c) → 
  (4 * x₂^2 - k * x₂ = c) → 
  x₁ + x₂ = k / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3363_336390


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l3363_336385

theorem smallest_prime_dividing_sum : ∃ (p : Nat), 
  Prime p ∧ 
  p ∣ (2^14 + 7^9) ∧ 
  ∀ (q : Nat), Prime q → q ∣ (2^14 + 7^9) → p ≤ q :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l3363_336385


namespace NUMINAMATH_CALUDE_tan_sum_minus_product_62_73_l3363_336394

theorem tan_sum_minus_product_62_73 :
  Real.tan (62 * π / 180) + Real.tan (73 * π / 180) - 
  Real.tan (62 * π / 180) * Real.tan (73 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_minus_product_62_73_l3363_336394


namespace NUMINAMATH_CALUDE_unique_a_for_cubic_property_l3363_336356

theorem unique_a_for_cubic_property (a : ℕ+) :
  (∀ n : ℕ+, ∃ k : ℤ, 4 * (a.val ^ n.val + 1) = k ^ 3) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_a_for_cubic_property_l3363_336356


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3363_336332

theorem system_solution_ratio (a b x y : ℝ) : 
  8 * x - 6 * y = a →
  9 * x - 12 * y = b →
  x ≠ 0 →
  y ≠ 0 →
  b ≠ 0 →
  a / b = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3363_336332


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l3363_336343

/-- The focus of the parabola y = 8x^2 has coordinates (0, 1/32) -/
theorem parabola_focus_coordinates :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ y - 8 * x^2
  ∃! p : ℝ × ℝ, p = (0, 1/32) ∧ ∀ x y, f (x, y) = 0 → (x - p.1)^2 = 4 * p.2 * (y - p.2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l3363_336343


namespace NUMINAMATH_CALUDE_minimum_loss_for_1997_pills_l3363_336350

/-- Represents a bottle of medicine --/
structure Bottle where
  capacity : ℕ
  pills : ℕ

/-- Represents the state of all bottles --/
structure State where
  a : Bottle
  b : Bottle
  c : Bottle
  loss : ℕ

/-- Calculates the minimum total loss of active ingredient --/
def minimumTotalLoss (initialPills : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum total loss for the given problem --/
theorem minimum_loss_for_1997_pills :
  minimumTotalLoss 1997 = 32401 := by
  sorry

#check minimum_loss_for_1997_pills

end NUMINAMATH_CALUDE_minimum_loss_for_1997_pills_l3363_336350


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_condition_l3363_336364

/-- The eccentricity of an ellipse with equation x^2 + y^2/m = 1 (m > 0) is greater than 1/2
    if and only if 0 < m < 4/3 or m > 3/4 -/
theorem ellipse_eccentricity_condition (m : ℝ) :
  (m > 0) →
  (∃ (x y : ℝ), x^2 + y^2/m = 1) →
  (∃ (e : ℝ), e > 1/2 ∧ e^2 = 1 - (min 1 m) / (max 1 m)) ↔
  (0 < m ∧ m < 4/3) ∨ m > 3/4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_condition_l3363_336364


namespace NUMINAMATH_CALUDE_unique_solution_floor_ceiling_l3363_336338

theorem unique_solution_floor_ceiling (a : ℝ) :
  (⌊a⌋ = 3 * a + 6) ∧ (⌈a⌉ = 4 * a + 9) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_floor_ceiling_l3363_336338


namespace NUMINAMATH_CALUDE_solve_steak_problem_l3363_336316

def steak_problem (cost_per_pound change_received : ℕ) : Prop :=
  let amount_paid : ℕ := 20
  let amount_spent : ℕ := amount_paid - change_received
  let pounds_bought : ℕ := amount_spent / cost_per_pound
  (cost_per_pound = 7 ∧ change_received = 6) → pounds_bought = 2

theorem solve_steak_problem :
  ∀ (cost_per_pound change_received : ℕ),
    steak_problem cost_per_pound change_received :=
by
  sorry

end NUMINAMATH_CALUDE_solve_steak_problem_l3363_336316


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l3363_336325

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a_value :
  ∀ a : ℝ, f' a (-1) = 8 → a = -6 :=
by
  sorry

#check tangent_slope_implies_a_value

end NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l3363_336325


namespace NUMINAMATH_CALUDE_num_divisors_2310_l3363_336353

/-- The number of positive divisors of 2310 is 32 -/
theorem num_divisors_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_2310_l3363_336353


namespace NUMINAMATH_CALUDE_expression_evaluation_l3363_336309

theorem expression_evaluation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  let f := (((x + 2)^2 * (x^2 - 2*x + 4)^2) / (x^3 + 8)^2)^2 *
            (((x - 2)^2 * (x^2 + 2*x + 4)^2) / (x^3 - 8)^2)^2
  f = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3363_336309


namespace NUMINAMATH_CALUDE_five_consecutive_not_square_l3363_336365

theorem five_consecutive_not_square (n : ℤ) : 
  ∃ (m : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ≠ m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_five_consecutive_not_square_l3363_336365


namespace NUMINAMATH_CALUDE_intersection_value_l3363_336333

theorem intersection_value (A B : Set ℝ) (a : ℝ) :
  A = {x : ℝ | x ≤ 1} →
  B = {x : ℝ | x ≥ a} →
  A ∩ B = {1} →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_value_l3363_336333


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3363_336328

theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 160) :
  let square_side := square_perimeter / 4
  let rect_length := square_side
  let rect_width := square_side / 4
  let rect_perimeter := 2 * (rect_length + rect_width)
  rect_perimeter = 100 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3363_336328


namespace NUMINAMATH_CALUDE_good_games_count_l3363_336339

def games_from_friend : ℕ := 11
def games_from_garage_sale : ℕ := 22
def non_working_games : ℕ := 19

theorem good_games_count :
  games_from_friend + games_from_garage_sale - non_working_games = 14 := by
  sorry

end NUMINAMATH_CALUDE_good_games_count_l3363_336339


namespace NUMINAMATH_CALUDE_plan_y_cheaper_at_601_l3363_336324

/-- Represents an internet service plan with a flat fee and per-gigabyte charge -/
structure InternetPlan where
  flatFee : ℕ
  perGBCharge : ℕ

/-- Calculates the total cost in cents for a given plan and number of gigabytes -/
def totalCost (plan : InternetPlan) (gigabytes : ℕ) : ℕ :=
  plan.flatFee * 100 + plan.perGBCharge * gigabytes

theorem plan_y_cheaper_at_601 :
  let planX : InternetPlan := ⟨50, 15⟩
  let planY : InternetPlan := ⟨80, 10⟩
  ∀ g : ℕ, g ≥ 601 → totalCost planY g < totalCost planX g ∧
  ∀ g : ℕ, g < 601 → totalCost planX g ≤ totalCost planY g :=
by sorry

end NUMINAMATH_CALUDE_plan_y_cheaper_at_601_l3363_336324


namespace NUMINAMATH_CALUDE_simplify_power_expression_l3363_336313

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^5 = 243 * x^20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l3363_336313


namespace NUMINAMATH_CALUDE_area_BEDC_is_30_l3363_336367

/-- Represents a parallelogram ABCD with a line DE parallel to AB -/
structure Parallelogram :=
  (AB : ℝ)
  (height : ℝ)
  (DE : ℝ)
  (is_parallelogram : Bool)
  (DE_parallel_AB : Bool)
  (E_midpoint_DC : Bool)

/-- Calculate the area of region BEDC in the given parallelogram -/
def area_BEDC (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem stating that the area of region BEDC is 30 under given conditions -/
theorem area_BEDC_is_30 (p : Parallelogram) 
  (h1 : p.AB = 12)
  (h2 : p.height = 10)
  (h3 : p.DE = 6)
  (h4 : p.is_parallelogram = true)
  (h5 : p.DE_parallel_AB = true)
  (h6 : p.E_midpoint_DC = true) :
  area_BEDC p = 30 :=
sorry

end NUMINAMATH_CALUDE_area_BEDC_is_30_l3363_336367


namespace NUMINAMATH_CALUDE_triangle_area_l3363_336347

theorem triangle_area (R : ℝ) (A : ℝ) (b c : ℝ) (h1 : R = 4) (h2 : A = π / 3) (h3 : b - c = 4) :
  let S := (1 / 2) * b * c * Real.sin A
  S = 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3363_336347


namespace NUMINAMATH_CALUDE_jar_price_calculation_l3363_336396

noncomputable def jar_price (d h p : ℝ) (d' h' : ℝ) : ℝ :=
  p * (d' / d)^2 * (h' / h)

theorem jar_price_calculation (d₁ h₁ p₁ d₂ h₂ : ℝ) 
  (hd₁ : d₁ = 2) (hh₁ : h₁ = 5) (hp₁ : p₁ = 0.75)
  (hd₂ : d₂ = 4) (hh₂ : h₂ = 8) :
  jar_price d₁ h₁ p₁ d₂ h₂ = 2.40 := by
  sorry

end NUMINAMATH_CALUDE_jar_price_calculation_l3363_336396


namespace NUMINAMATH_CALUDE_tangent_difference_l3363_336371

/-- Given two circles in a plane, this theorem proves that the difference between
    the squares of their external and internal tangent lengths is 30. -/
theorem tangent_difference (r₁ r₂ x y A₁₀ : ℝ) : 
  r₁ > 0 → r₂ > 0 → A₁₀ > 0 →
  r₁ * r₂ = 15 / 2 →
  x^2 + (r₁ + r₂)^2 = A₁₀^2 →
  y^2 + (r₁ - r₂)^2 = A₁₀^2 →
  y^2 - x^2 = 30 := by
sorry

end NUMINAMATH_CALUDE_tangent_difference_l3363_336371


namespace NUMINAMATH_CALUDE_quadratic_sum_l3363_336342

/-- Given a quadratic function f(x) = 8x^2 - 48x - 320, prove that when written in the form a(x+b)^2+c, the sum a + b + c equals -387. -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 8*x^2 - 48*x - 320) →
  (∀ x, f x = a*(x+b)^2 + c) →
  a + b + c = -387 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3363_336342


namespace NUMINAMATH_CALUDE_dan_marbles_count_l3363_336389

/-- The total number of marbles Dan has after receiving red marbles from Mary -/
def total_marbles (violet_marbles red_marbles : ℕ) : ℕ :=
  violet_marbles + red_marbles

/-- Theorem stating that Dan has 78 marbles in total -/
theorem dan_marbles_count :
  total_marbles 64 14 = 78 := by
  sorry

end NUMINAMATH_CALUDE_dan_marbles_count_l3363_336389


namespace NUMINAMATH_CALUDE_students_under_three_l3363_336302

/-- Represents the number of students in different age groups in a nursery school -/
structure NurserySchool where
  total : ℕ
  fourAndOlder : ℕ
  underThree : ℕ
  notBetweenThreeAndFour : ℕ

/-- Theorem stating the number of students under three years old in the nursery school -/
theorem students_under_three (school : NurserySchool) 
  (h1 : school.total = 300)
  (h2 : school.fourAndOlder = school.total / 10)
  (h3 : school.notBetweenThreeAndFour = 50)
  (h4 : school.notBetweenThreeAndFour = school.fourAndOlder + school.underThree) :
  school.underThree = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_under_three_l3363_336302


namespace NUMINAMATH_CALUDE_election_winner_votes_l3363_336304

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (58 : ℚ) / 100 * total_votes - (42 : ℚ) / 100 * total_votes = 288) :
  ⌊(58 : ℚ) / 100 * total_votes⌋ = 1044 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3363_336304


namespace NUMINAMATH_CALUDE_zero_most_frequent_units_digit_l3363_336311

-- Define the set of numbers
def numbers : Finset ℕ := Finset.range 9

-- Function to calculate the units digit of a sum
def unitsDigitOfSum (a b : ℕ) : ℕ := (a + b) % 10

-- Function to count occurrences of a specific units digit
def countOccurrences (digit : ℕ) : ℕ :=
  numbers.card * numbers.card

-- Theorem stating that 0 is the most frequent units digit
theorem zero_most_frequent_units_digit :
  ∀ d : ℕ, d ∈ Finset.range 10 → d ≠ 0 →
    countOccurrences 0 > countOccurrences d :=
sorry

end NUMINAMATH_CALUDE_zero_most_frequent_units_digit_l3363_336311


namespace NUMINAMATH_CALUDE_tangent_line_intersection_extreme_values_l3363_336399

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x - 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

-- Theorem for the tangent line intersection points
theorem tangent_line_intersection (x₀ : ℝ) (b : ℝ) :
  f' x₀ = -9 ∧ f x₀ = -9 * x₀ + b → b = -3 ∨ b = -7 :=
sorry

-- Theorem for the extreme values of f(x)
theorem extreme_values :
  (∃ x : ℝ, f x = 2 ∧ ∀ y : ℝ, f y ≤ f x) ∧
  (∃ x : ℝ, f x = -30 ∧ ∀ y : ℝ, f y ≥ f x) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_extreme_values_l3363_336399


namespace NUMINAMATH_CALUDE_triple_minus_double_equals_eight_point_five_l3363_336393

theorem triple_minus_double_equals_eight_point_five (x : ℝ) : 
  3 * x = 2 * x + 8.5 → 3 * x - 2 * x = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_triple_minus_double_equals_eight_point_five_l3363_336393


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l3363_336360

/-- The number of y-intercepts of the parabola x = 3y^2 - 6y + 3 -/
theorem parabola_y_intercepts : 
  let f : ℝ → ℝ := fun y => 3 * y^2 - 6 * y + 3
  ∃! y : ℝ, f y = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l3363_336360


namespace NUMINAMATH_CALUDE_rhombus_area_l3363_336344

/-- The area of a rhombus with vertices at (0, 4.5), (8, 0), (0, -4.5), and (-8, 0) is 72 square units. -/
theorem rhombus_area : ℝ := by
  -- Define the vertices of the rhombus
  let v1 : ℝ × ℝ := (0, 4.5)
  let v2 : ℝ × ℝ := (8, 0)
  let v3 : ℝ × ℝ := (0, -4.5)
  let v4 : ℝ × ℝ := (-8, 0)

  -- Define the diagonals of the rhombus
  let d1 : ℝ := ‖v1.2 - v3.2‖ -- Distance between y-coordinates of v1 and v3
  let d2 : ℝ := ‖v2.1 - v4.1‖ -- Distance between x-coordinates of v2 and v4

  -- Calculate the area of the rhombus
  let area : ℝ := (d1 * d2) / 2

  -- Prove that the area is 72 square units
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3363_336344


namespace NUMINAMATH_CALUDE_product_sum_theorem_l3363_336327

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a + b + c = 20) : 
  a*b + b*c + a*c = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l3363_336327


namespace NUMINAMATH_CALUDE_henry_age_is_27_l3363_336315

/-- Henry's present age -/
def henry_age : ℕ := sorry

/-- Jill's present age -/
def jill_age : ℕ := sorry

/-- The sum of Henry and Jill's present ages is 43 -/
axiom sum_of_ages : henry_age + jill_age = 43

/-- 5 years ago, Henry was twice the age of Jill -/
axiom age_relation : henry_age - 5 = 2 * (jill_age - 5)

/-- Theorem: Henry's present age is 27 years -/
theorem henry_age_is_27 : henry_age = 27 := by sorry

end NUMINAMATH_CALUDE_henry_age_is_27_l3363_336315


namespace NUMINAMATH_CALUDE_coffee_shop_weekly_production_l3363_336322

/-- Represents the brewing rate and operating hours of a coffee shop for a specific day type -/
structure DayType where
  brewingRate : ℕ
  operatingHours : ℕ

/-- Calculates the total number of coffee cups brewed in a week -/
def totalCupsPerWeek (weekday : DayType) (weekend : DayType) : ℕ :=
  weekday.brewingRate * weekday.operatingHours * 5 +
  weekend.brewingRate * weekend.operatingHours * 2

/-- Theorem: The coffee shop brews 400 cups in a week -/
theorem coffee_shop_weekly_production :
  let weekday : DayType := { brewingRate := 10, operatingHours := 5 }
  let weekend : DayType := { brewingRate := 15, operatingHours := 5 }
  totalCupsPerWeek weekday { weekend with operatingHours := 6 } = 400 :=
by sorry

end NUMINAMATH_CALUDE_coffee_shop_weekly_production_l3363_336322


namespace NUMINAMATH_CALUDE_divisor_problem_l3363_336370

theorem divisor_problem : ∃ (x : ℕ), x > 0 ∧ 181 = 9 * x + 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3363_336370


namespace NUMINAMATH_CALUDE_opposite_roots_quadratic_l3363_336303

theorem opposite_roots_quadratic (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (k^2 - 4)*x₁ + k + 1 = 0 ∧ 
    x₂^2 + (k^2 - 4)*x₂ + k + 1 = 0 ∧
    x₁ = -x₂) → 
  k = -2 := by
sorry

end NUMINAMATH_CALUDE_opposite_roots_quadratic_l3363_336303


namespace NUMINAMATH_CALUDE_brian_chris_fishing_l3363_336340

theorem brian_chris_fishing (brian_trips chris_trips : ℕ) 
  (brian_fish_per_trip : ℕ) (total_fish : ℕ) :
  brian_trips = 2 * chris_trips →
  brian_fish_per_trip = 400 →
  chris_trips = 10 →
  total_fish = 13600 →
  (1 - (brian_fish_per_trip : ℚ) / ((total_fish - brian_trips * brian_fish_per_trip) / chris_trips : ℚ)) = 2/7 := by
sorry

end NUMINAMATH_CALUDE_brian_chris_fishing_l3363_336340


namespace NUMINAMATH_CALUDE_complex_expression_value_l3363_336357

theorem complex_expression_value : 
  ∃ (i : ℂ), i^2 = -1 ∧ i^3 * (1 + i)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l3363_336357


namespace NUMINAMATH_CALUDE_basketball_league_games_l3363_336378

/-- The number of games played in a basketball league season -/
def total_games (n : ℕ) (games_per_pairing : ℕ) : ℕ :=
  n * (n - 1) * games_per_pairing / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team,
    the total number of games played is 180. -/
theorem basketball_league_games :
  total_games 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_basketball_league_games_l3363_336378


namespace NUMINAMATH_CALUDE_partition_exists_l3363_336397

-- Define the type for our partition function
def PartitionFunction := ℕ+ → Fin 100

-- Define the property that the partition satisfies the required condition
def SatisfiesCondition (f : PartitionFunction) : Prop :=
  ∀ a b c : ℕ+, a + 99 * b = c → f a = f b ∨ f a = f c ∨ f b = f c

-- State the theorem
theorem partition_exists : ∃ f : PartitionFunction, SatisfiesCondition f := by
  sorry

end NUMINAMATH_CALUDE_partition_exists_l3363_336397


namespace NUMINAMATH_CALUDE_average_first_12_even_numbers_l3363_336373

theorem average_first_12_even_numbers : 
  let first_12_even : List ℕ := List.range 12 |>.map (fun n => 2 * (n + 1))
  (first_12_even.sum / first_12_even.length : ℚ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_average_first_12_even_numbers_l3363_336373


namespace NUMINAMATH_CALUDE_cube_division_theorem_l3363_336358

/-- A point in 3D space represented by rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Represents that a point is inside the unit cube -/
def insideUnitCube (p : RationalPoint) : Prop :=
  0 < p.x ∧ p.x < 1 ∧ 0 < p.y ∧ p.y < 1 ∧ 0 < p.z ∧ p.z < 1

theorem cube_division_theorem (points : Finset RationalPoint) 
    (h : points.card = 2003) 
    (h_inside : ∀ p ∈ points, insideUnitCube p) :
    ∃ (n : ℕ), n > 2003 ∧ 
    ∀ p ∈ points, ∃ (i j k : ℕ), 
      i < n ∧ j < n ∧ k < n ∧
      (i : ℚ) / n < p.x ∧ p.x < ((i + 1) : ℚ) / n ∧
      (j : ℚ) / n < p.y ∧ p.y < ((j + 1) : ℚ) / n ∧
      (k : ℚ) / n < p.z ∧ p.z < ((k + 1) : ℚ) / n :=
by sorry


end NUMINAMATH_CALUDE_cube_division_theorem_l3363_336358


namespace NUMINAMATH_CALUDE_train_crossing_time_l3363_336331

/-- Proves the time taken by a train to cross a platform -/
theorem train_crossing_time (train_length platform_length : ℝ) 
  (time_to_cross_pole : ℝ) (h1 : train_length = 300) 
  (h2 : platform_length = 675) (h3 : time_to_cross_pole = 12) : 
  (train_length + platform_length) / (train_length / time_to_cross_pole) = 39 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3363_336331


namespace NUMINAMATH_CALUDE_class_size_l3363_336387

/-- Proves that in a class where the number of girls is 0.4 of the number of boys
    and there are 10 girls, the total number of students is 35. -/
theorem class_size (boys girls : ℕ) : 
  girls = 10 → 
  girls = (2 / 5 : ℚ) * boys → 
  boys + girls = 35 := by
sorry

end NUMINAMATH_CALUDE_class_size_l3363_336387


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l3363_336346

/-- Systematic sampling function that returns the nth element of the sample -/
def systematicSample (populationSize sampleSize start n : ℕ) : ℕ :=
  start + (populationSize / sampleSize) * n

/-- Theorem: In a systematic sample of size 5 from a population of 55,
    if students 3, 25, and 47 are in the sample,
    then the other two students in the sample have numbers 14 and 36 -/
theorem systematic_sample_theorem :
  let populationSize : ℕ := 55
  let sampleSize : ℕ := 5
  let start : ℕ := 3
  (systematicSample populationSize sampleSize start 0 = 3) →
  (systematicSample populationSize sampleSize start 2 = 25) →
  (systematicSample populationSize sampleSize start 4 = 47) →
  (systematicSample populationSize sampleSize start 1 = 14) ∧
  (systematicSample populationSize sampleSize start 3 = 36) :=
by
  sorry


end NUMINAMATH_CALUDE_systematic_sample_theorem_l3363_336346


namespace NUMINAMATH_CALUDE_class_gender_ratio_l3363_336363

theorem class_gender_ratio (total_students : ℕ) (female_students : ℕ) : 
  total_students = 52 → female_students = 13 → 
  (total_students - female_students) / female_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_class_gender_ratio_l3363_336363


namespace NUMINAMATH_CALUDE_sum_of_first_four_terms_l3363_336312

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_four_terms
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_8th : a 8 = 21)
  (h_9th : a 9 = 17)
  (h_10th : a 10 = 13) :
  (a 1) + (a 2) + (a 3) + (a 4) = 172 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_four_terms_l3363_336312


namespace NUMINAMATH_CALUDE_fraction_simplification_l3363_336369

theorem fraction_simplification
  (a b c k : ℝ)
  (h1 : a * b = c * k)
  (h2 : c * k ≠ 0) :
  (a - b - c + k) / (a + b + c + k) = (a - c) / (a + c) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3363_336369


namespace NUMINAMATH_CALUDE_magical_stack_size_l3363_336310

/-- A magical stack is a stack of cards where at least one card from each pile retains its original position after restacking. -/
def MagicalStack (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≤ n ∧ b ≤ n ∧ a ≠ b

/-- The position of a card in the original stack. -/
def OriginalPosition (card : ℕ) (n : ℕ) : ℕ :=
  if card ≤ n then card else card - n

/-- The position of a card in the restacked stack. -/
def RestackedPosition (card : ℕ) (n : ℕ) : ℕ :=
  2 * card - 1 - (if card ≤ n then 0 else 1)

/-- A card retains its position if its original position equals its restacked position. -/
def RetainsPosition (card : ℕ) (n : ℕ) : Prop :=
  OriginalPosition card n = RestackedPosition card n

theorem magical_stack_size :
  ∀ n : ℕ,
    MagicalStack n →
    RetainsPosition 111 n →
    RetainsPosition 90 n →
    2 * n ≥ 332 →
    2 * n = 332 :=
sorry

end NUMINAMATH_CALUDE_magical_stack_size_l3363_336310


namespace NUMINAMATH_CALUDE_complex_magnitude_in_complement_range_l3363_336372

theorem complex_magnitude_in_complement_range (a : ℝ) : 
  let z : ℂ := 1 + a * Complex.I
  let M : Set ℝ := {x | x > 2}
  Complex.abs z ∈ (Set.Iic 2) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_in_complement_range_l3363_336372


namespace NUMINAMATH_CALUDE_percentage_relation_l3363_336395

theorem percentage_relation (A B T : ℝ) 
  (h1 : A = 0.2 * B) 
  (h2 : B = 0.3 * T) : 
  A = 0.06 * T := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3363_336395


namespace NUMINAMATH_CALUDE_line_equation_l3363_336305

/-- Circle C with equation x^2 + (y-1)^2 = 5 -/
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

/-- Line l with equation mx - y + 1 - m = 0 -/
def line_l (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

/-- Point P(1,1) -/
def point_P : ℝ × ℝ := (1, 1)

/-- Chord AB of circle C -/
def chord_AB (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  circle_C x₁ y₁ ∧ circle_C x₂ y₂

/-- Point P divides chord AB with ratio AP:PB = 1:2 -/
def divides_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  chord_AB x₁ y₁ x₂ y₂ ∧ 2*(1 - x₁) = x₂ - 1 ∧ 2*(1 - y₁) = y₂ - 1

theorem line_equation (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, line_l m 1 1 ∧ divides_chord x₁ y₁ x₂ y₂) →
  (line_l 1 x y ∨ line_l (-1) x y) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3363_336305


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l3363_336326

/-- Given the conditions of a class of boys with height measurements:
    - initial_average: The initially calculated average height
    - wrong_height: The wrongly recorded height of one boy
    - correct_height: The correct height of the boy with the wrong measurement
    - actual_average: The actual average height after correction
    
    Prove that the number of boys in the class is equal to the given value.
-/
theorem number_of_boys_in_class 
  (initial_average : ℝ) 
  (wrong_height : ℝ) 
  (correct_height : ℝ) 
  (actual_average : ℝ) 
  (h1 : initial_average = 180) 
  (h2 : wrong_height = 156) 
  (h3 : correct_height = 106) 
  (h4 : actual_average = 178) : 
  ∃ n : ℕ, n * actual_average = n * initial_average - (wrong_height - correct_height) ∧ n = 25 :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l3363_336326


namespace NUMINAMATH_CALUDE_expression_equality_l3363_336319

theorem expression_equality : 
  Real.sqrt 32 + (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) - Real.sqrt 4 - 6 * Real.sqrt (1/2) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3363_336319


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3363_336354

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), 
    (∀ x : ℚ, x ≠ 9 ∧ x ≠ -4 →
      (6 * x + 5) / (x^2 - 5*x - 36) = C / (x - 9) + D / (x + 4)) ∧
    C = 59 / 13 ∧
    D = 19 / 13 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3363_336354


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3363_336376

def total_amount : ℕ := 5000
def x_amount : ℕ := 1000

theorem ratio_x_to_y :
  (x_amount : ℚ) / (total_amount - x_amount : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3363_336376


namespace NUMINAMATH_CALUDE_tens_digit_of_3_pow_405_l3363_336317

theorem tens_digit_of_3_pow_405 : 3^405 % 100 = 43 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_pow_405_l3363_336317


namespace NUMINAMATH_CALUDE_quadratic_single_intersection_l3363_336381

theorem quadratic_single_intersection (m : ℝ) : 
  (∃! x, (m + 1) * x^2 - 2*(m + 1) * x - 1 = 0) ↔ m = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_single_intersection_l3363_336381


namespace NUMINAMATH_CALUDE_f_of_3_equals_155_l3363_336323

-- Define the function f
def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x^2 - 4 * x + 5

-- Theorem statement
theorem f_of_3_equals_155 : f 3 = 155 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_155_l3363_336323


namespace NUMINAMATH_CALUDE_median_same_variance_decreases_l3363_336306

def original_data : List ℝ := [2, 2, 4, 4]
def new_data : List ℝ := [2, 2, 3, 4, 4]

def median (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem median_same_variance_decreases :
  median original_data = median new_data ∧
  variance new_data < variance original_data := by sorry

end NUMINAMATH_CALUDE_median_same_variance_decreases_l3363_336306


namespace NUMINAMATH_CALUDE_quadratic_vertex_l3363_336382

/-- A quadratic function f(x) = x^2 + bx + c -/
def QuadraticFunction (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem quadratic_vertex (b c : ℝ) :
  (QuadraticFunction b c 1 = 0) →
  (∀ x, QuadraticFunction b c (2 + x) = QuadraticFunction b c (2 - x)) →
  (∃ y, QuadraticFunction b c 2 = y ∧ ∀ x, QuadraticFunction b c x ≥ y) →
  (2, -1) = (2, QuadraticFunction b c 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l3363_336382


namespace NUMINAMATH_CALUDE_science_fair_competition_l3363_336334

theorem science_fair_competition (k h n : ℕ) : 
  h = (3 * k) / 5 →
  n = 2 * (k + h) →
  k + h + n = 240 →
  k = 50 ∧ h = 30 ∧ n = 160 := by
sorry

end NUMINAMATH_CALUDE_science_fair_competition_l3363_336334


namespace NUMINAMATH_CALUDE_ap_to_gp_ratio_is_positive_integer_l3363_336384

/-- An arithmetic progression starting with 1 -/
def AP (x : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => AP x n + (x - 1)

/-- A geometric progression starting with 1 -/
def GP (a : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => GP a n * a

/-- The property that a GP is formed by deleting some terms from an AP -/
def isSubsequence (x a : ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, GP a n = AP x m

theorem ap_to_gp_ratio_is_positive_integer (x : ℝ) (hx : x ≥ 1) (a : ℝ) (ha : a > 0)
    (h : isSubsequence x a) : ∃ k : ℕ+, a = k :=
  sorry

end NUMINAMATH_CALUDE_ap_to_gp_ratio_is_positive_integer_l3363_336384


namespace NUMINAMATH_CALUDE_sqrt_three_inequality_l3363_336392

theorem sqrt_three_inequality (n : ℕ+) :
  (n : ℝ) + 3 < n * Real.sqrt 3 ∧ n * Real.sqrt 3 < (n : ℝ) + 4 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_inequality_l3363_336392


namespace NUMINAMATH_CALUDE_max_edges_bipartite_graph_l3363_336359

/-- 
Given a complete bipartite graph K_{m,n} where m and n are positive integers and m + n = 21,
prove that the maximum number of edges is 110.
-/
theorem max_edges_bipartite_graph : 
  ∀ m n : ℕ+, 
  m + n = 21 → 
  ∃ (max_edges : ℕ), 
    max_edges = m * n ∧ 
    ∀ k l : ℕ+, k + l = 21 → k * l ≤ max_edges :=
by
  sorry

end NUMINAMATH_CALUDE_max_edges_bipartite_graph_l3363_336359


namespace NUMINAMATH_CALUDE_banana_fraction_proof_l3363_336337

theorem banana_fraction_proof (jefferson_bananas : ℕ) (walter_bananas : ℚ) (f : ℚ) :
  jefferson_bananas = 56 →
  walter_bananas = 56 - 56 * f →
  (56 + (56 - 56 * f)) / 2 = 49 →
  f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_banana_fraction_proof_l3363_336337


namespace NUMINAMATH_CALUDE_practice_problems_count_l3363_336336

theorem practice_problems_count (N : ℕ) 
  (h1 : N > 0)
  (h2 : (4 / 5 : ℚ) * (3 / 4 : ℚ) * (2 / 3 : ℚ) * N = 24) : N = 60 := by
  sorry

#check practice_problems_count

end NUMINAMATH_CALUDE_practice_problems_count_l3363_336336


namespace NUMINAMATH_CALUDE_expected_value_coin_flip_l3363_336377

/-- The expected value of a coin flip game -/
theorem expected_value_coin_flip (p_heads : ℚ) (p_tails : ℚ) 
  (win_heads : ℚ) (lose_tails : ℚ) : 
  p_heads = 1/3 → p_tails = 2/3 → win_heads = 3 → lose_tails = 2 →
  p_heads * win_heads - p_tails * lose_tails = -1/3 := by
  sorry

#check expected_value_coin_flip

end NUMINAMATH_CALUDE_expected_value_coin_flip_l3363_336377


namespace NUMINAMATH_CALUDE_pentagon_point_reconstruction_l3363_336388

/-- Given a pentagon ABCDE with extended sides, prove the relation between A and A', B', C', D' -/
theorem pentagon_point_reconstruction (A B C D E A' B' C' D' : ℝ × ℝ) : 
  A'B = 2 * AB → 
  B'C = BC → 
  C'D = CD → 
  D'E = 2 * DE → 
  A = (1/9 : ℝ) • A' + (2/9 : ℝ) • B' + (4/9 : ℝ) • C' + (8/9 : ℝ) • D' := by
  sorry


end NUMINAMATH_CALUDE_pentagon_point_reconstruction_l3363_336388


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l3363_336383

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 6 * x^2 + 6 * x - 12

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-2 : ℝ) 1, f' x < 0 :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l3363_336383


namespace NUMINAMATH_CALUDE_cubic_gp_roots_iff_a_60_l3363_336361

/-- A cubic polynomial with parameter a -/
def cubic (a : ℝ) (x : ℝ) : ℝ := x^3 - 15*x^2 + a*x - 64

/-- Predicate for three distinct real roots in geometric progression -/
def has_three_distinct_gp_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, 
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    cubic a x₁ = 0 ∧ cubic a x₂ = 0 ∧ cubic a x₃ = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ x₂ = x₁ * q ∧ x₃ = x₂ * q

/-- The main theorem -/
theorem cubic_gp_roots_iff_a_60 :
  ∀ a : ℝ, has_three_distinct_gp_roots a ↔ a = 60 := by sorry

end NUMINAMATH_CALUDE_cubic_gp_roots_iff_a_60_l3363_336361


namespace NUMINAMATH_CALUDE_final_bill_is_520_20_l3363_336398

/-- The final bill amount after applying two consecutive 2% late charges to an original bill of $500 -/
def final_bill_amount (original_bill : ℝ) (late_charge_rate : ℝ) : ℝ :=
  original_bill * (1 + late_charge_rate) * (1 + late_charge_rate)

/-- Theorem stating that the final bill amount is $520.20 -/
theorem final_bill_is_520_20 :
  final_bill_amount 500 0.02 = 520.20 := by sorry

end NUMINAMATH_CALUDE_final_bill_is_520_20_l3363_336398


namespace NUMINAMATH_CALUDE_rap_song_requests_l3363_336366

/-- Represents the number of song requests for different genres in a night --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rap song requests given the conditions --/
theorem rap_song_requests (s : SongRequests) : s.rap = 2 :=
  by
  have h1 : s.total = 30 := by sorry
  have h2 : s.electropop = s.total / 2 := by sorry
  have h3 : s.dance = s.electropop / 3 := by sorry
  have h4 : s.rock = 5 := by sorry
  have h5 : s.oldies = s.rock - 3 := by sorry
  have h6 : s.dj_choice = s.oldies / 2 := by sorry
  have h7 : s.total = s.electropop + s.dance + s.rock + s.oldies + s.dj_choice + s.rap := by sorry
  sorry

end NUMINAMATH_CALUDE_rap_song_requests_l3363_336366


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_45_l3363_336320

-- Definition of a prime number
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Theorem statement
theorem no_primes_divisible_by_45 : ¬∃ p : ℕ, isPrime p ∧ 45 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_45_l3363_336320


namespace NUMINAMATH_CALUDE_equilateral_triangle_segment_length_l3363_336355

-- Define the triangle and points
def Triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C A)

def EquilateralTriangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  Triangle A B C ∧ dist A B = dist B C

def OnSegment (X P Q : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) • P + t • Q

-- State the theorem
theorem equilateral_triangle_segment_length 
  (A B C K L M : EuclideanSpace ℝ (Fin 2)) :
  EquilateralTriangle A B C →
  OnSegment K A B →
  OnSegment L B C →
  OnSegment M B C →
  OnSegment L B M →
  dist K L = dist K M →
  dist B L = 2 →
  dist A K = 3 →
  dist C M = 5 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_segment_length_l3363_336355


namespace NUMINAMATH_CALUDE_balloon_arrangement_count_l3363_336386

def balloon_arrangements : ℕ :=
  let total_letters := 7
  let num_L := 2
  let num_O := 3
  let remaining_letters := 3  -- B, A, N
  let block_positions := 4    -- LLLOOO can be in 4 positions
  
  block_positions * remaining_letters.factorial * 
  (total_letters.factorial / (num_L.factorial * num_O.factorial))

theorem balloon_arrangement_count : balloon_arrangements = 10080 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangement_count_l3363_336386


namespace NUMINAMATH_CALUDE_smallest_d_value_l3363_336349

theorem smallest_d_value (d : ℝ) : 
  (∃ d₀ : ℝ, d₀ > 0 ∧ Real.sqrt (40 + (4 * d₀ - 2)^2) = 10 * d₀ ∧
   ∀ d' : ℝ, d' > 0 → Real.sqrt (40 + (4 * d' - 2)^2) = 10 * d' → d₀ ≤ d') →
  d = (4 + Real.sqrt 940) / 42 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_value_l3363_336349


namespace NUMINAMATH_CALUDE_alyssa_puppies_left_l3363_336368

/-- The number of puppies Alyssa has left after giving some away -/
def puppies_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Alyssa has 5 puppies left -/
theorem alyssa_puppies_left : puppies_left 12 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_puppies_left_l3363_336368


namespace NUMINAMATH_CALUDE_percentage_of_375_l3363_336335

theorem percentage_of_375 (x : ℝ) :
  (x / 100) * 375 = 5.4375 → x = 1.45 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_375_l3363_336335


namespace NUMINAMATH_CALUDE_fraction_problem_l3363_336374

theorem fraction_problem (x y : ℕ) (h1 : x + y = 122) (h2 : (x - 19) / (y - 19) = 1 / 5) :
  x / y = 33 / 89 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3363_336374


namespace NUMINAMATH_CALUDE_volunteer_selection_theorem_l3363_336379

/-- The number of volunteers --/
def n : ℕ := 20

/-- The number of volunteers to be selected --/
def k : ℕ := 4

/-- The number of the first specific volunteer that must be selected --/
def a : ℕ := 5

/-- The number of the second specific volunteer that must be selected --/
def b : ℕ := 14

/-- The number of volunteers with numbers less than the first specific volunteer --/
def m : ℕ := a - 1

/-- The number of volunteers with numbers greater than the second specific volunteer --/
def p : ℕ := n - b

/-- The total number of ways to select the volunteers under the given conditions --/
def total_ways : ℕ := Nat.choose m 2 + Nat.choose p 2

theorem volunteer_selection_theorem :
  total_ways = 21 := by sorry

end NUMINAMATH_CALUDE_volunteer_selection_theorem_l3363_336379


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3363_336391

theorem polynomial_simplification (x : ℝ) : 
  (3*x + 2) * (3*x - 2) - (3*x - 1)^2 = 6*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3363_336391


namespace NUMINAMATH_CALUDE_special_array_determination_l3363_336351

/-- Represents an m×n array of positive integers -/
def SpecialArray (m n : ℕ) := Fin m → Fin n → ℕ+

/-- The condition that must hold for any four numbers in the array -/
def SpecialCondition (A : SpecialArray m n) : Prop :=
  ∀ (i₁ i₂ : Fin m) (j₁ j₂ : Fin n),
    A i₁ j₁ + A i₂ j₂ = A i₁ j₂ + A i₂ j₁

/-- The theorem stating that m+n-1 elements are sufficient to determine the entire array -/
theorem special_array_determination (m n : ℕ) (A : SpecialArray m n) 
  (hA : SpecialCondition A) :
  ∃ (S : Finset ((Fin m) × (Fin n))),
    S.card = m + n - 1 ∧ 
    (∀ (B : SpecialArray m n), 
      SpecialCondition B → 
      (∀ (p : (Fin m) × (Fin n)), p ∈ S → A p.1 p.2 = B p.1 p.2) → 
      A = B) :=
sorry

end NUMINAMATH_CALUDE_special_array_determination_l3363_336351


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l3363_336308

theorem quadratic_form_ratio (j : ℝ) :
  ∃ (c p q : ℝ), 
    (∀ j, 8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ 
    q / p = -151 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l3363_336308


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3363_336301

theorem rationalize_denominator :
  Real.sqrt (5 / (2 + Real.sqrt 2)) = Real.sqrt 5 - Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3363_336301


namespace NUMINAMATH_CALUDE_little_john_money_little_john_initial_money_l3363_336345

/-- Little John's money problem -/
theorem little_john_money : ℝ → Prop :=
  fun initial_money =>
    let spent_on_sweets : ℝ := 3.25
    let given_to_each_friend : ℝ := 2.20
    let number_of_friends : ℕ := 2
    let money_left : ℝ := 2.45
    initial_money = spent_on_sweets + (given_to_each_friend * number_of_friends) + money_left ∧
    initial_money = 10.10

/-- Proof of Little John's initial money amount -/
theorem little_john_initial_money : ∃ (m : ℝ), little_john_money m :=
  sorry

end NUMINAMATH_CALUDE_little_john_money_little_john_initial_money_l3363_336345


namespace NUMINAMATH_CALUDE_chocolates_remaining_day5_l3363_336307

/-- Calculates the number of chocolates remaining after 4 days of consumption -/
def chocolates_remaining (initial : ℕ) (day1 : ℕ) : ℕ :=
  let day2 := 2 * day1 - 3
  let day3 := day1 - 2
  let day4 := day3 - 1
  initial - (day1 + day2 + day3 + day4)

/-- Theorem stating that given the initial conditions, 12 chocolates remain on Day 5 -/
theorem chocolates_remaining_day5 :
  chocolates_remaining 24 4 = 12 := by
  sorry

#eval chocolates_remaining 24 4

end NUMINAMATH_CALUDE_chocolates_remaining_day5_l3363_336307


namespace NUMINAMATH_CALUDE_live_bargaining_theorem_l3363_336321

/-- Represents the price reduction scenario in a live streaming bargaining event. -/
def live_bargaining_price_reduction (initial_price final_price : ℝ) (num_rounds : ℕ) (reduction_rate : ℝ) : Prop :=
  initial_price * (1 - reduction_rate) ^ num_rounds = final_price

/-- The live bargaining price reduction theorem. -/
theorem live_bargaining_theorem :
  ∃ (x : ℝ), live_bargaining_price_reduction 120 43.2 2 x :=
sorry

end NUMINAMATH_CALUDE_live_bargaining_theorem_l3363_336321


namespace NUMINAMATH_CALUDE_equation_solutions_l3363_336318

theorem equation_solutions :
  (∃ x : ℝ, x - 0.4 * x = 120 ∧ x = 200) ∧
  (∃ x : ℝ, 5 * x - 5 / 6 = 5 / 4 ∧ x = 5 / 12) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3363_336318


namespace NUMINAMATH_CALUDE_complement_intersection_empty_l3363_336300

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {2, 4, 5}

theorem complement_intersection_empty :
  (Aᶜ ∩ Bᶜ : Set Nat) = ∅ :=
by
  sorry

#check complement_intersection_empty

end NUMINAMATH_CALUDE_complement_intersection_empty_l3363_336300


namespace NUMINAMATH_CALUDE_profit_and_marginal_profit_maxima_l3363_336329

def R (x : ℕ) : ℝ := 3000 * x - 20 * x^2
def C (x : ℕ) : ℝ := 600 * x + 2000
def p (x : ℕ) : ℝ := R x - C x
def Mp (x : ℕ) : ℝ := p (x + 1) - p x

theorem profit_and_marginal_profit_maxima 
  (h : ∀ x : ℕ, 0 < x ∧ x ≤ 100) :
  (∃ x : ℕ, p x = 74000 ∧ ∀ y : ℕ, p y ≤ 74000) ∧
  (∃ x : ℕ, Mp x = 2340 ∧ ∀ y : ℕ, Mp y ≤ 2340) :=
sorry

end NUMINAMATH_CALUDE_profit_and_marginal_profit_maxima_l3363_336329
