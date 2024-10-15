import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_proof_l838_83852

theorem quadratic_root_proof : ∃ (a b c : ℤ), 
  a ≠ 0 ∧ 
  (a * (2 - Real.sqrt 7)^2 + b * (2 - Real.sqrt 7) + c = 0) ∧
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x^2 - 4*x - 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l838_83852


namespace NUMINAMATH_CALUDE_inequality_transformation_l838_83820

theorem inequality_transformation (a b c : ℝ) (hc : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by sorry

end NUMINAMATH_CALUDE_inequality_transformation_l838_83820


namespace NUMINAMATH_CALUDE_power_product_equality_l838_83808

theorem power_product_equality : (-2/3)^2023 * (3/2)^2024 = -3/2 := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l838_83808


namespace NUMINAMATH_CALUDE_quadratic_transformation_l838_83838

theorem quadratic_transformation (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (h k s : ℝ) (hs : s ≠ 0), ∀ x : ℝ,
    a * x^2 + b * x + c = s^2 * ((x - h)^2 + k) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l838_83838


namespace NUMINAMATH_CALUDE_lee_family_concert_cost_is_86_l838_83831

/-- Represents the cost calculation for the Lee family concert tickets --/
def lee_family_concert_cost : ℝ :=
  let regular_ticket_cost : ℝ := 10
  let booking_fee : ℝ := 1.5
  let youngest_discount : ℝ := 0.4
  let oldest_discount : ℝ := 0.3
  let middle_discount : ℝ := 0.2
  let youngest_count : ℕ := 3
  let oldest_count : ℕ := 3
  let middle_count : ℕ := 4
  let total_tickets : ℕ := youngest_count + oldest_count + middle_count

  let youngest_cost : ℝ := youngest_count * (regular_ticket_cost * (1 - youngest_discount))
  let oldest_cost : ℝ := oldest_count * (regular_ticket_cost * (1 - oldest_discount))
  let middle_cost : ℝ := middle_count * (regular_ticket_cost * (1 - middle_discount))
  
  let total_ticket_cost : ℝ := youngest_cost + oldest_cost + middle_cost
  let total_booking_fees : ℝ := total_tickets * booking_fee

  total_ticket_cost + total_booking_fees

/-- Theorem stating that the total cost for the Lee family concert tickets is $86.00 --/
theorem lee_family_concert_cost_is_86 : lee_family_concert_cost = 86 := by
  sorry

end NUMINAMATH_CALUDE_lee_family_concert_cost_is_86_l838_83831


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l838_83827

/-- Given that i² = -1, prove that (3 + 2i) / (4 - 5i) = 2/41 + (23/41)i -/
theorem complex_fraction_simplification :
  (Complex.I : ℂ)^2 = -1 →
  (3 + 2 * Complex.I) / (4 - 5 * Complex.I) = (2 : ℂ) / 41 + (23 : ℂ) / 41 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l838_83827


namespace NUMINAMATH_CALUDE_max_value_of_f_l838_83810

open Real

-- Define the function
def f (x : ℝ) : ℝ := x * (3 - 2 * x)

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Ioo 0 (3/2) ∧
  (∀ x, x ∈ Set.Ioo 0 (3/2) → f x ≤ f c) ∧
  f c = 9/8 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l838_83810


namespace NUMINAMATH_CALUDE_hamburger_combinations_l838_83814

-- Define the number of patty options
def patty_options : Nat := 4

-- Define the number of condiments
def num_condiments : Nat := 9

-- Theorem statement
theorem hamburger_combinations :
  (patty_options * 2^num_condiments) = 2048 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l838_83814


namespace NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l838_83803

theorem power_of_product_equals_product_of_powers (a b : ℝ) : 
  (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l838_83803


namespace NUMINAMATH_CALUDE_classroom_paint_area_l838_83825

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

end NUMINAMATH_CALUDE_classroom_paint_area_l838_83825


namespace NUMINAMATH_CALUDE_alyssa_fruit_expenditure_l838_83861

/-- The amount Alyssa paid for grapes in dollars -/
def grapes_cost : ℚ := 12.08

/-- The amount Alyssa paid for cherries in dollars -/
def cherries_cost : ℚ := 9.85

/-- The total amount Alyssa spent on fruits -/
def total_cost : ℚ := grapes_cost + cherries_cost

theorem alyssa_fruit_expenditure : total_cost = 21.93 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_fruit_expenditure_l838_83861


namespace NUMINAMATH_CALUDE_sum_equals_two_thirds_l838_83829

theorem sum_equals_two_thirds : 
  let original_sum := (1/3 : ℚ) + 1/6 + 1/9 + 1/12 + 1/15 + 1/18
  let remaining_sum := (1/3 : ℚ) + 1/6 + 1/9 + 1/18
  remaining_sum = 2/3 := by sorry

end NUMINAMATH_CALUDE_sum_equals_two_thirds_l838_83829


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l838_83822

theorem sum_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l838_83822


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l838_83894

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (1/3)^2 + (1/4)^2 = (13*x / 53*y) * ((1/5)^2 + (1/6)^2) →
  Real.sqrt x / Real.sqrt y = 1092 / 338 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l838_83894


namespace NUMINAMATH_CALUDE_rationalize_denominator_l838_83877

theorem rationalize_denominator :
  5 / (2 + Real.sqrt 5) = -10 + 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l838_83877


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l838_83897

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l838_83897


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l838_83850

theorem complex_magnitude_problem (z : ℂ) (h : (3 - 4*I) * z = 4 + 3*I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l838_83850


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l838_83863

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube. -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  ∃ (inner_cube_volume : ℝ),
    inner_cube_volume = 192 * Real.sqrt 3 ∧
    inner_cube_volume = (outer_cube_edge / Real.sqrt 3) ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l838_83863


namespace NUMINAMATH_CALUDE_investment_difference_l838_83806

def emma_investment : ℝ := 300
def briana_investment : ℝ := 500
def emma_yield_rate : ℝ := 0.15
def briana_yield_rate : ℝ := 0.10
def time_period : ℝ := 2

theorem investment_difference :
  briana_investment * briana_yield_rate * time_period - 
  emma_investment * emma_yield_rate * time_period = 10 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_l838_83806


namespace NUMINAMATH_CALUDE_william_farm_tax_l838_83867

/-- Calculates an individual's farm tax payment given the total tax collected and their land percentage -/
def individual_farm_tax (total_tax : ℝ) (land_percentage : ℝ) : ℝ :=
  land_percentage * total_tax

/-- Proves that given the conditions, Mr. William's farm tax payment is $960 -/
theorem william_farm_tax :
  let total_tax : ℝ := 3840
  let william_land_percentage : ℝ := 0.25
  individual_farm_tax total_tax william_land_percentage = 960 := by
sorry

end NUMINAMATH_CALUDE_william_farm_tax_l838_83867


namespace NUMINAMATH_CALUDE_binary_multiplication_example_l838_83830

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec toBinary (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
    toBinary n

theorem binary_multiplication_example :
  let a := [false, true, true, false, true, true]  -- 110110₂
  let b := [true, true, true]  -- 111₂
  let result := [false, true, false, false, true, false, false, true]  -- 10010010₂
  binaryToNat a * binaryToNat b = binaryToNat result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_example_l838_83830


namespace NUMINAMATH_CALUDE_laptop_down_payment_percentage_l838_83886

theorem laptop_down_payment_percentage
  (laptop_cost : ℝ)
  (monthly_installment : ℝ)
  (additional_down_payment : ℝ)
  (balance_after_four_months : ℝ)
  (h1 : laptop_cost = 1000)
  (h2 : monthly_installment = 65)
  (h3 : additional_down_payment = 20)
  (h4 : balance_after_four_months = 520) :
  let down_payment_percentage := 100 * (laptop_cost - balance_after_four_months - 4 * monthly_installment - additional_down_payment) / laptop_cost
  down_payment_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_laptop_down_payment_percentage_l838_83886


namespace NUMINAMATH_CALUDE_araceli_luana_numbers_l838_83892

theorem araceli_luana_numbers : ∃ (a b c : ℕ), 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  100 * a + 10 * b + c = (10 * a + b) + (10 * b + c) + (10 * c + a) ∧
  a = 1 ∧ b = 9 ∧ c = 8 := by
sorry

end NUMINAMATH_CALUDE_araceli_luana_numbers_l838_83892


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l838_83828

-- Problem 1
theorem problem_1 (t : ℝ) : 
  (∀ x : ℝ, |2*x + t| - t ≤ 8 ↔ -5 ≤ x ∧ x ≤ 4) → t = 1 :=
sorry

-- Problem 2
theorem problem_2 (x y z : ℝ) :
  x^2 + (1/4)*y^2 + (1/9)*z^2 = 2 →
  x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l838_83828


namespace NUMINAMATH_CALUDE_local_minimum_condition_l838_83893

/-- The function f(x) = x(x - m)^2 attains a local minimum at x = 1 -/
theorem local_minimum_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x => x * (x - m)^2
  (∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1) →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_condition_l838_83893


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l838_83876

theorem polynomial_division_theorem (x : ℝ) :
  4 * x^4 - 3 * x^3 + 6 * x^2 - 9 * x + 3 = 
  (x + 2) * (4 * x^3 - 11 * x^2 + 28 * x - 65) + 133 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l838_83876


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l838_83859

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 6 = 30 → a 3 + a 9 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l838_83859


namespace NUMINAMATH_CALUDE_subtraction_with_division_l838_83849

theorem subtraction_with_division : 5100 - (102 / 20.4) = 5095 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_with_division_l838_83849


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l838_83884

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4 + a 7 = 19) →
  (a 3 + 5 * a 6 = 57) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l838_83884


namespace NUMINAMATH_CALUDE_money_lending_problem_l838_83870

/-- Given a sum of money divided into two parts where:
    1. The interest on the first part for 8 years at 3% per annum is equal to
       the interest on the second part for 3 years at 5% per annum.
    2. The second part is Rs. 1656.
    Prove that the total sum lent is Rs. 2691. -/
theorem money_lending_problem (first_part second_part total_sum : ℚ) : 
  second_part = 1656 →
  (first_part * 3 / 100 * 8 = second_part * 5 / 100 * 3) →
  total_sum = first_part + second_part →
  total_sum = 2691 := by
  sorry

#check money_lending_problem

end NUMINAMATH_CALUDE_money_lending_problem_l838_83870


namespace NUMINAMATH_CALUDE_inequality_proof_l838_83833

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x / y + y / z + z / x) / 3 ≥ 1) ∧
  (x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ (x / y + y / z + z / x)^2 / 3) ∧
  (x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ x / y + y / z + z / x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l838_83833


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l838_83851

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 2*(m-3)*x + 16 = (a*x + b)^2) →
  (m = 7 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l838_83851


namespace NUMINAMATH_CALUDE_number_order_l838_83818

theorem number_order (a b : ℝ) (ha : a = 7) (hb : b = 0.3) :
  a^b > b^a ∧ b^a > Real.log b := by sorry

end NUMINAMATH_CALUDE_number_order_l838_83818


namespace NUMINAMATH_CALUDE_numbers_statistics_l838_83873

def numbers : List ℝ := [158, 149, 155, 157, 156, 162, 155, 168]

def median (xs : List ℝ) : ℝ := sorry

def mean (xs : List ℝ) : ℝ := sorry

def mode (xs : List ℝ) : ℝ := sorry

theorem numbers_statistics :
  median numbers = 155.5 ∧
  mean numbers = 157.5 ∧
  mode numbers = 155 := by sorry

end NUMINAMATH_CALUDE_numbers_statistics_l838_83873


namespace NUMINAMATH_CALUDE_parametric_equation_of_lineL_l838_83865

/-- A line in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (l : Line2D) (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p.1 = l.point.1 + t * l.direction.1 ∧ p.2 = l.point.2 + t * l.direction.2

/-- The line passing through (3, 5) and parallel to (4, 2) -/
def lineL : Line2D :=
  { point := (3, 5)
    direction := (4, 2) }

/-- Theorem: The parametric equation (x - 3)/4 = (y - 5)/2 represents lineL -/
theorem parametric_equation_of_lineL :
  ∀ x y : ℝ, pointOnLine lineL (x, y) ↔ (x - 3) / 4 = (y - 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_equation_of_lineL_l838_83865


namespace NUMINAMATH_CALUDE_red_balls_count_l838_83848

theorem red_balls_count (total : ℕ) (prob : ℚ) : 
  total = 15 → 
  prob = 1 / 21 →
  ∃ (red : ℕ), red ≤ total ∧ 
    (red : ℚ) / total * (red - 1) / (total - 1) = prob ∧
    red = 5 :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l838_83848


namespace NUMINAMATH_CALUDE_problem_sampling_is_systematic_l838_83826

/-- Represents a sampling method -/
inductive SamplingMethod
| DrawingLots
| RandomNumberTable
| SystematicSampling
| Other

/-- Represents a high school with classes and student numbering -/
structure HighSchool where
  num_classes : Nat
  students_per_class : Nat
  selected_number : Nat

/-- Defines the conditions of the problem -/
def problem_conditions : HighSchool :=
  { num_classes := 12
  , students_per_class := 50
  , selected_number := 20 }

/-- Defines systematic sampling -/
def is_systematic_sampling (school : HighSchool) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.SystematicSampling ∧
  school.num_classes > 0 ∧
  school.students_per_class > 0 ∧
  school.selected_number > 0 ∧
  school.selected_number ≤ school.students_per_class

/-- Theorem stating that the sampling method in the problem is systematic sampling -/
theorem problem_sampling_is_systematic :
  is_systematic_sampling problem_conditions SamplingMethod.SystematicSampling :=
by
  sorry

end NUMINAMATH_CALUDE_problem_sampling_is_systematic_l838_83826


namespace NUMINAMATH_CALUDE_decimal_333_to_octal_l838_83872

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The octal representation of decimal 333 is 515 -/
theorem decimal_333_to_octal :
  decimal_to_octal 333 = 515 := by
  sorry

end NUMINAMATH_CALUDE_decimal_333_to_octal_l838_83872


namespace NUMINAMATH_CALUDE_square_difference_equality_l838_83896

theorem square_difference_equality : 1005^2 - 995^2 - 1003^2 + 997^2 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l838_83896


namespace NUMINAMATH_CALUDE_number_of_teams_l838_83882

theorem number_of_teams (n : ℕ) (k : ℕ) : n = 10 → k = 5 → Nat.choose n k = 252 := by
  sorry

end NUMINAMATH_CALUDE_number_of_teams_l838_83882


namespace NUMINAMATH_CALUDE_largest_s_value_l838_83862

theorem largest_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) : 
  (r - 2) * s * 61 = (s - 2) * r * 60 → s ≤ 121 ∧ ∃ (r' : ℕ), r' ≥ 121 ∧ (r' - 2) * 121 * 61 = 119 * r' * 60 :=
sorry

end NUMINAMATH_CALUDE_largest_s_value_l838_83862


namespace NUMINAMATH_CALUDE_units_digit_of_seven_pow_five_cubed_l838_83802

theorem units_digit_of_seven_pow_five_cubed : 7^(5^3) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_pow_five_cubed_l838_83802


namespace NUMINAMATH_CALUDE_seventh_rack_dvd_count_l838_83889

/-- Calculates the number of DVDs on a given rack based on the previous two racks -/
def dvd_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 3  -- First rack
  | 1 => 4  -- Second rack
  | n + 2 => ((dvd_count (n + 1) - dvd_count n) * 2) + dvd_count (n + 1)

/-- The number of DVDs on the seventh rack is 66 -/
theorem seventh_rack_dvd_count :
  dvd_count 6 = 66 := by sorry

end NUMINAMATH_CALUDE_seventh_rack_dvd_count_l838_83889


namespace NUMINAMATH_CALUDE_square_sum_eq_two_l838_83890

theorem square_sum_eq_two (a b : ℝ) : (a^2 + b^2)^4 - 8*(a^2 + b^2)^2 + 16 = 0 → a^2 + b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_eq_two_l838_83890


namespace NUMINAMATH_CALUDE_house_cleaning_time_l838_83816

/-- The time it takes for three people to clean a house together, given their individual cleaning rates. -/
theorem house_cleaning_time 
  (john_time : ℝ) 
  (nick_time : ℝ) 
  (mary_time : ℝ) 
  (h1 : john_time = 6) 
  (h2 : nick_time / 3 = john_time / 2) 
  (h3 : mary_time = nick_time + 2) : 
  1 / (1 / john_time + 1 / nick_time + 1 / mary_time) = 198 / 73 :=
sorry

end NUMINAMATH_CALUDE_house_cleaning_time_l838_83816


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l838_83811

theorem abs_inequality_solution_set (x : ℝ) : 
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l838_83811


namespace NUMINAMATH_CALUDE_alex_earnings_l838_83805

/-- Alex's work hours and earnings problem -/
theorem alex_earnings (hours_week3 : ℕ) (hours_difference : ℕ) (earnings_difference : ℕ) :
  hours_week3 = 28 →
  hours_difference = 10 →
  earnings_difference = 80 →
  (hours_week3 - hours_difference) * (earnings_difference / hours_difference) +
  hours_week3 * (earnings_difference / hours_difference) = 368 := by
  sorry

end NUMINAMATH_CALUDE_alex_earnings_l838_83805


namespace NUMINAMATH_CALUDE_difference_of_squares_l838_83813

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l838_83813


namespace NUMINAMATH_CALUDE_quadratic_sum_l838_83800

/-- A quadratic function y = ax^2 + bx + c with a minimum value of 61
    that passes through the points (1,0) and (3,0) -/
def QuadraticFunction (a b c : ℝ) : Prop :=
  (∀ x, a*x^2 + b*x + c ≥ 61) ∧
  (∃ x₀, a*x₀^2 + b*x₀ + c = 61) ∧
  (a*1^2 + b*1 + c = 0) ∧
  (a*3^2 + b*3 + c = 0)

theorem quadratic_sum (a b c : ℝ) :
  QuadraticFunction a b c → a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l838_83800


namespace NUMINAMATH_CALUDE_total_length_S_l838_83821

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
    ((|x| - 2)^2 + (|y| - 2)^2)^(1/2) = 2 - |1 - ((|x| - 2)^2 + (|y| - 2)^2)^(1/2)|}

-- Define the length function for S
noncomputable def length_S : ℝ := sorry

-- Theorem statement
theorem total_length_S : length_S = 20 * Real.pi := by sorry

end NUMINAMATH_CALUDE_total_length_S_l838_83821


namespace NUMINAMATH_CALUDE_satellite_units_l838_83853

/-- Represents a satellite with modular units and sensors. -/
structure Satellite where
  units : ℕ  -- Number of modular units
  non_upgraded_per_unit : ℕ  -- Number of non-upgraded sensors per unit
  total_upgraded : ℕ  -- Total number of upgraded sensors

/-- The conditions given in the problem. -/
def satellite_conditions (s : Satellite) : Prop :=
  -- Condition 2: Non-upgraded sensors per unit is 1/8 of total upgraded
  s.non_upgraded_per_unit = s.total_upgraded / 8 ∧
  -- Condition 3: 25% of all sensors are upgraded
  s.total_upgraded = (s.units * s.non_upgraded_per_unit + s.total_upgraded) / 4

/-- The theorem stating that a satellite satisfying the given conditions has 24 units. -/
theorem satellite_units (s : Satellite) (h : satellite_conditions s) : s.units = 24 := by
  sorry


end NUMINAMATH_CALUDE_satellite_units_l838_83853


namespace NUMINAMATH_CALUDE_constant_relationship_l838_83857

theorem constant_relationship (a b c d : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < π / 2 →
    (a * Real.sin θ + b * Real.cos θ - c = 0) ∧
    (a * Real.cos θ - b * Real.sin θ + d = 0)) →
  a^2 + b^2 = c^2 + d^2 := by sorry

end NUMINAMATH_CALUDE_constant_relationship_l838_83857


namespace NUMINAMATH_CALUDE_real_part_of_i_times_one_plus_i_l838_83855

theorem real_part_of_i_times_one_plus_i : Complex.re (Complex.I * (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_i_times_one_plus_i_l838_83855


namespace NUMINAMATH_CALUDE_unique_four_digit_square_divisible_by_11_ending_in_1_l838_83899

theorem unique_four_digit_square_divisible_by_11_ending_in_1 :
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ ∃ k : ℕ, n = k^2 ∧ n % 11 = 0 ∧ n % 10 = 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_divisible_by_11_ending_in_1_l838_83899


namespace NUMINAMATH_CALUDE_expansion_properties_l838_83879

theorem expansion_properties (x : ℝ) : 
  let expansion := (x + 1) * (x + 2)^4
  ∃ (a b c d e f : ℝ), 
    expansion = a*x^5 + b*x^4 + c*x^3 + 56*x^2 + d*x + e ∧
    a + b + c + 56 + d + e = 162 :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l838_83879


namespace NUMINAMATH_CALUDE_sum_of_roots_l838_83817

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a - 1 = 0)
  (hb : b^3 - 3*b^2 + 5*b - 5 = 0) : 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l838_83817


namespace NUMINAMATH_CALUDE_unanswered_questions_l838_83845

/-- Calculates the number of unanswered questions on a test -/
theorem unanswered_questions
  (total_questions : ℕ)
  (answering_time_hours : ℕ)
  (time_per_question_minutes : ℕ)
  (h1 : total_questions = 100)
  (h2 : answering_time_hours = 2)
  (h3 : time_per_question_minutes = 2) :
  total_questions - (answering_time_hours * 60) / time_per_question_minutes = 40 :=
by sorry

end NUMINAMATH_CALUDE_unanswered_questions_l838_83845


namespace NUMINAMATH_CALUDE_johns_work_days_l838_83866

/-- Proves that John drives to work 5 days a week given his car's efficiency,
    distance to work, leisure travel, and weekly gas usage. -/
theorem johns_work_days (efficiency : ℝ) (distance_to_work : ℝ) (leisure_miles : ℝ) (gas_usage : ℝ)
    (h1 : efficiency = 30)
    (h2 : distance_to_work = 20)
    (h3 : leisure_miles = 40)
    (h4 : gas_usage = 8) :
    (gas_usage * efficiency - leisure_miles) / (2 * distance_to_work) = 5 := by
  sorry

end NUMINAMATH_CALUDE_johns_work_days_l838_83866


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l838_83847

theorem line_equation_through_point_with_slope (x y : ℝ) :
  let point : ℝ × ℝ := (-2, 1)
  let slope : ℝ := Real.tan (135 * π / 180)
  (x - point.1) * slope = y - point.2 →
  x + y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l838_83847


namespace NUMINAMATH_CALUDE_balloon_radius_ratio_l838_83843

theorem balloon_radius_ratio (V_L V_S r_L r_S : ℝ) : 
  V_L = 450 * Real.pi →
  V_S = 0.08 * V_L →
  V_L = (4/3) * Real.pi * r_L^3 →
  V_S = (4/3) * Real.pi * r_S^3 →
  r_S / r_L = Real.rpow 2 (1/3) / 5 := by
  sorry

end NUMINAMATH_CALUDE_balloon_radius_ratio_l838_83843


namespace NUMINAMATH_CALUDE_inclined_prism_volume_l838_83883

/-- The volume of an inclined prism with a parallelogram base and inclined lateral edge. -/
theorem inclined_prism_volume 
  (base_side1 base_side2 lateral_edge : ℝ) 
  (base_angle lateral_angle : ℝ) : 
  base_side1 = 3 →
  base_side2 = 6 →
  lateral_edge = 4 →
  base_angle = Real.pi / 4 →
  lateral_angle = Real.pi / 6 →
  (base_side1 * base_side2 * Real.sin base_angle) * (lateral_edge * Real.sin lateral_angle) = 18 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inclined_prism_volume_l838_83883


namespace NUMINAMATH_CALUDE_complement_of_M_l838_83888

def U : Set Nat := {1, 2, 3, 4}

def M : Set Nat := {x ∈ U | x^2 - 5*x + 6 = 0}

theorem complement_of_M :
  (U \ M) = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l838_83888


namespace NUMINAMATH_CALUDE_lemonade_sales_difference_l838_83856

/-- 
Given Stanley's and Carl's hourly lemonade sales rates and a fixed time period,
prove the difference in their total sales.
-/
theorem lemonade_sales_difference 
  (stanley_rate : ℕ) 
  (carl_rate : ℕ) 
  (time_period : ℕ) 
  (h1 : stanley_rate = 4)
  (h2 : carl_rate = 7)
  (h3 : time_period = 3) :
  carl_rate * time_period - stanley_rate * time_period = 9 := by
  sorry

#check lemonade_sales_difference

end NUMINAMATH_CALUDE_lemonade_sales_difference_l838_83856


namespace NUMINAMATH_CALUDE_stratified_sampling_car_models_l838_83804

/-- Represents the number of units to sample from a stratum in stratified sampling -/
def stratified_sample_size (stratum_size : ℕ) (total_population : ℕ) (total_sample : ℕ) : ℕ :=
  (stratum_size * total_sample) / total_population

/-- Theorem stating the correct sample sizes for the given problem -/
theorem stratified_sampling_car_models :
  let model1_size : ℕ := 1200
  let model2_size : ℕ := 6000
  let model3_size : ℕ := 2000
  let total_population : ℕ := model1_size + model2_size + model3_size
  let total_sample : ℕ := 46
  stratified_sample_size model1_size total_population total_sample = 6 ∧
  stratified_sample_size model2_size total_population total_sample = 30 ∧
  stratified_sample_size model3_size total_population total_sample = 10 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_car_models_l838_83804


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l838_83844

def total_shoes : ℕ := 8
def red_shoes : ℕ := 4
def green_shoes : ℕ := 4

theorem probability_two_red_shoes :
  (red_shoes : ℚ) / total_shoes * (red_shoes - 1) / (total_shoes - 1) = 3 / 14 :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l838_83844


namespace NUMINAMATH_CALUDE_ticket_sales_income_l838_83854

/-- Calculates the total income from ticket sales given the number of student and adult tickets sold and their respective prices. -/
def total_income (student_tickets : ℕ) (adult_tickets : ℕ) (student_price : ℚ) (adult_price : ℚ) : ℚ :=
  student_tickets * student_price + adult_tickets * adult_price

/-- Proves that the total income from selling 20 tickets, where 12 are student tickets at $2.00 each and 8 are adult tickets at $4.50 each, is equal to $60.00. -/
theorem ticket_sales_income :
  let student_tickets : ℕ := 12
  let adult_tickets : ℕ := 8
  let student_price : ℚ := 2
  let adult_price : ℚ := 9/2
  total_income student_tickets adult_tickets student_price adult_price = 60 := by
  sorry


end NUMINAMATH_CALUDE_ticket_sales_income_l838_83854


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_ratio_l838_83874

theorem isosceles_triangle_side_ratio (a b : ℝ) (h_isosceles : b > 0) (h_vertex_angle : Real.cos (20 * π / 180) = a / (2 * b)) : 2 < b / a ∧ b / a < 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_ratio_l838_83874


namespace NUMINAMATH_CALUDE_friend_reading_time_l838_83837

/-- Proves that given the conditions on reading speeds and time, 
    the friend's reading time for one volume is 0.3 hours -/
theorem friend_reading_time 
  (my_speed : ℝ) 
  (friend_speed : ℝ) 
  (my_time_two_volumes : ℝ) 
  (h1 : my_speed = (1 / 5) * friend_speed) 
  (h2 : my_time_two_volumes = 3) : 
  (my_time_two_volumes / 2) / 5 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_friend_reading_time_l838_83837


namespace NUMINAMATH_CALUDE_probability_all_heads_or_some_tails_l838_83840

def num_coins : ℕ := 5

def coin_outcomes : ℕ := 2

def all_outcomes : ℕ := coin_outcomes ^ num_coins

theorem probability_all_heads_or_some_tails :
  (1 : ℚ) / all_outcomes + ((all_outcomes - 1 : ℕ) : ℚ) / all_outcomes = 1 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_heads_or_some_tails_l838_83840


namespace NUMINAMATH_CALUDE_space_for_another_circle_l838_83895

/-- The side length of the large square N -/
def N : ℝ := 6

/-- The side length of the small squares -/
def small_square_side : ℝ := 1

/-- The diameter of the circles -/
def circle_diameter : ℝ := 1

/-- The number of small squares -/
def num_squares : ℕ := 4

/-- The number of circles -/
def num_circles : ℕ := 3

/-- The theorem stating that there is space for another circle -/
theorem space_for_another_circle :
  (N - 1)^2 - (num_squares * (small_square_side^2 + small_square_side * circle_diameter + Real.pi * (circle_diameter / 2)^2) +
   num_circles * Real.pi * (circle_diameter / 2)^2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_space_for_another_circle_l838_83895


namespace NUMINAMATH_CALUDE_joans_savings_l838_83815

/-- The number of quarters Joan has saved --/
def num_quarters : ℕ := 6

/-- The value of one quarter in cents --/
def cents_per_quarter : ℕ := 25

/-- Theorem: The total value of Joan's quarters in cents --/
theorem joans_savings : num_quarters * cents_per_quarter = 150 := by
  sorry

end NUMINAMATH_CALUDE_joans_savings_l838_83815


namespace NUMINAMATH_CALUDE_simple_interest_problem_l838_83875

/-- 
Given a principal amount P and an interest rate R, 
if increasing the rate by 3% for 4 years results in Rs. 120 more interest, 
then P = 1000.
-/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 3) * 4) / 100 - (P * R * 4) / 100 = 120 → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l838_83875


namespace NUMINAMATH_CALUDE_kittens_count_l838_83846

-- Define the number of puppies
def num_puppies : ℕ := 32

-- Define the number of kittens in terms of puppies
def num_kittens : ℕ := 2 * num_puppies + 14

-- Theorem to prove
theorem kittens_count : num_kittens = 78 := by
  sorry

end NUMINAMATH_CALUDE_kittens_count_l838_83846


namespace NUMINAMATH_CALUDE_largest_coeff_x3_sum_64_l838_83880

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The condition that the coefficient of x^3 is the largest in (1+x)^n -/
def coeff_x3_largest (n : ℕ) : Prop :=
  ∀ k, k ≠ 3 → binomial n 3 ≥ binomial n k

/-- The sum of all coefficients in the expansion of (1+x)^n -/
def sum_coefficients (n : ℕ) : ℕ := 2^n

theorem largest_coeff_x3_sum_64 :
  ∀ n : ℕ, coeff_x3_largest n → sum_coefficients n = 64 := by sorry

end NUMINAMATH_CALUDE_largest_coeff_x3_sum_64_l838_83880


namespace NUMINAMATH_CALUDE_not_convex_pentagon_with_diagonals_l838_83841

/-- A list of segment lengths -/
def segment_lengths : List ℝ := [2, 3, 5, 7, 8, 9, 10, 11, 13, 15]

/-- A predicate that checks if three real numbers can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- A predicate that checks if a list of real numbers can form a convex pentagon with sides and diagonals -/
def is_convex_pentagon_with_diagonals (lengths : List ℝ) : Prop :=
  lengths.length = 10 ∧
  ∀ (a b c : ℝ), a ∈ lengths → b ∈ lengths → c ∈ lengths →
    a ≠ b ∧ b ≠ c ∧ a ≠ c → is_triangle a b c

/-- Theorem stating that the given segment lengths cannot form a convex pentagon with diagonals -/
theorem not_convex_pentagon_with_diagonals :
  ¬ is_convex_pentagon_with_diagonals segment_lengths := by
  sorry

end NUMINAMATH_CALUDE_not_convex_pentagon_with_diagonals_l838_83841


namespace NUMINAMATH_CALUDE_boat_upstream_speed_l838_83835

/-- Proves that given a boat with a speed of 7 km/hr in still water and a
    downstream speed of 10 km/hr, the upstream speed of the boat is 4 km/hr. -/
theorem boat_upstream_speed
  (still_water_speed : ℝ)
  (downstream_speed : ℝ)
  (h1 : still_water_speed = 7)
  (h2 : downstream_speed = 10) :
  still_water_speed - (downstream_speed - still_water_speed) = 4 := by
  sorry


end NUMINAMATH_CALUDE_boat_upstream_speed_l838_83835


namespace NUMINAMATH_CALUDE_dogs_in_garden_l838_83809

/-- The number of dogs in a garden with ducks and a specific number of feet. -/
def num_dogs (total_feet : ℕ) (num_ducks : ℕ) (feet_per_dog : ℕ) (feet_per_duck : ℕ) : ℕ :=
  (total_feet - num_ducks * feet_per_duck) / feet_per_dog

/-- Theorem stating that under the given conditions, there are 6 dogs in the garden. -/
theorem dogs_in_garden : num_dogs 28 2 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_dogs_in_garden_l838_83809


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l838_83871

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℚ), x ≠ 2 → x ≠ 4 →
      (3 * x + 7) / ((x - 4) * (x - 2)^2) =
      A / (x - 4) + B / (x - 2) + C / (x - 2)^2 ∧
      A = 19 / 4 ∧ B = -19 / 4 ∧ C = -13 / 2 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l838_83871


namespace NUMINAMATH_CALUDE_horner_rule_v₂_l838_83823

def f (x : ℝ) : ℝ := 4 * x^4 + 3 * x^3 - 6 * x^2 + x - 1

def v₀ : ℝ := 4

def v₁ (x : ℝ) : ℝ := v₀ * x + 3

def v₂ (x : ℝ) : ℝ := v₁ x * x - 6

theorem horner_rule_v₂ : v₂ (-1) = -5 := by sorry

end NUMINAMATH_CALUDE_horner_rule_v₂_l838_83823


namespace NUMINAMATH_CALUDE_average_increase_food_expenditure_l838_83819

/-- Represents the regression line equation for annual income and food expenditure -/
def regression_line (x : ℝ) : ℝ := 0.245 * x + 0.321

/-- Theorem stating that the average increase in food expenditure for a unit increase in income is 0.245 -/
theorem average_increase_food_expenditure :
  ∀ x : ℝ, regression_line (x + 1) - regression_line x = 0.245 := by
sorry


end NUMINAMATH_CALUDE_average_increase_food_expenditure_l838_83819


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l838_83834

/-- Given a triangle ABC with the specified conditions, prove that A = π/6 and a = 2 -/
theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π →
  A + B + C = π →
  Real.sqrt 3 * a * Real.sin C = c * Real.cos A →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  b + c = 2 + 2 * Real.sqrt 3 →
  A = π / 6 ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l838_83834


namespace NUMINAMATH_CALUDE_area_ratio_correct_l838_83807

/-- Represents a rectangle inscribed in a circle with a smaller rectangle inside it. -/
structure InscribedRectangles where
  /-- The ratio of the smaller rectangle's width to the larger rectangle's width -/
  x : ℝ
  /-- The ratio of the smaller rectangle's height to the larger rectangle's height -/
  y : ℝ
  /-- Constraint ensuring the smaller rectangle's vertices lie on the circle -/
  h_circle : 4 * y^2 + 4 * y + x^2 = 1

/-- The area ratio of the smaller rectangle to the larger rectangle -/
def areaRatio (r : InscribedRectangles) : ℝ := r.x * r.y

theorem area_ratio_correct (r : InscribedRectangles) : 
  areaRatio r = r.x * r.y := by sorry

end NUMINAMATH_CALUDE_area_ratio_correct_l838_83807


namespace NUMINAMATH_CALUDE_no_prime_roots_sum_64_l838_83839

theorem no_prime_roots_sum_64 : ¬∃ (p q k : ℕ), 
  Prime p ∧ Prime q ∧ 
  p * q = k ∧
  p + q = 64 ∧
  p^2 - 64*p + k = 0 ∧
  q^2 - 64*q + k = 0 :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_sum_64_l838_83839


namespace NUMINAMATH_CALUDE_common_difference_from_sum_condition_l838_83898

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n * (a 1 + a n)) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_from_sum_condition (seq : ArithmeticSequence) 
    (h : seq.S 4 / 12 - seq.S 3 / 9 = 1) : 
    common_difference seq = 6 := by
  sorry


end NUMINAMATH_CALUDE_common_difference_from_sum_condition_l838_83898


namespace NUMINAMATH_CALUDE_set_intersection_problem_l838_83868

theorem set_intersection_problem (M N P : Set Nat) 
  (hM : M = {1})
  (hN : N = {1, 2})
  (hP : P = {1, 2, 3}) :
  (M ∪ N) ∩ P = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l838_83868


namespace NUMINAMATH_CALUDE_percentage_equality_l838_83864

theorem percentage_equality :
  ∃! k : ℚ, (k / 100) * 25 = (20 / 100) * 30 := by sorry

end NUMINAMATH_CALUDE_percentage_equality_l838_83864


namespace NUMINAMATH_CALUDE_min_value_fraction_l838_83885

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^2 + b^2) / (a*b - b^2) ≥ 2 + 2*Real.sqrt 2 ∧
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (a^2 + b^2) / (a*b - b^2) = 2 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l838_83885


namespace NUMINAMATH_CALUDE_polygon_sides_l838_83801

theorem polygon_sides (sum_interior_angles : ℕ) : sum_interior_angles = 1440 → ∃ n : ℕ, n = 10 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l838_83801


namespace NUMINAMATH_CALUDE_triangle_angle_adjustment_l838_83860

/-- 
Given a triangle with interior angles in a 3:4:9 ratio, prove that if the largest angle is 
decreased by x degrees such that the smallest angle doubles its initial value while 
maintaining the sum of angles as 180 degrees, then x = 33.75 degrees.
-/
theorem triangle_angle_adjustment (k : ℝ) (x : ℝ) 
  (h1 : 3*k + 4*k + 9*k = 180)  -- Sum of initial angles is 180 degrees
  (h2 : 3*k + 4*k + (9*k - x) = 180)  -- Sum of angles after adjustment is 180 degrees
  (h3 : 2*(3*k) = 3*k + 4*k)  -- Smallest angle doubles its initial value
  : x = 33.75 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_adjustment_l838_83860


namespace NUMINAMATH_CALUDE_jenna_concert_spending_percentage_l838_83824

/-- Proves that Jenna spends 10% of her monthly salary on a concert outing -/
theorem jenna_concert_spending_percentage :
  let concert_ticket_cost : ℚ := 181
  let drink_ticket_cost : ℚ := 7
  let num_drink_tickets : ℕ := 5
  let hourly_wage : ℚ := 18
  let weekly_hours : ℕ := 30
  let weeks_per_month : ℕ := 4

  let total_outing_cost : ℚ := concert_ticket_cost + drink_ticket_cost * num_drink_tickets
  let weekly_salary : ℚ := hourly_wage * weekly_hours
  let monthly_salary : ℚ := weekly_salary * weeks_per_month
  let spending_percentage : ℚ := total_outing_cost / monthly_salary * 100

  spending_percentage = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_jenna_concert_spending_percentage_l838_83824


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l838_83878

theorem inequality_implies_upper_bound (m : ℝ) : 
  (∀ x : ℝ, |x + 4| + |x + 8| ≥ m) → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l838_83878


namespace NUMINAMATH_CALUDE_sum_areas_circles_6_8_10_triangle_l838_83832

/-- Given a 6-8-10 right triangle with vertices as centers of three mutually externally tangent circles,
    the sum of the areas of these circles is 56π. -/
theorem sum_areas_circles_6_8_10_triangle : 
  ∃ (α β γ : ℝ),
    α + β = 6 ∧
    α + γ = 8 ∧
    β + γ = 10 ∧
    α > 0 ∧ β > 0 ∧ γ > 0 →
    π * (α^2 + β^2 + γ^2) = 56 * π := by
  sorry


end NUMINAMATH_CALUDE_sum_areas_circles_6_8_10_triangle_l838_83832


namespace NUMINAMATH_CALUDE_cost_of_2500_pencils_l838_83858

/-- The cost of a given number of pencils, given the cost of 100 pencils -/
def cost_of_pencils (cost_per_100 : ℚ) (num_pencils : ℕ) : ℚ :=
  (cost_per_100 * num_pencils) / 100

/-- Theorem stating that 2500 pencils cost $750 when 100 pencils cost $30 -/
theorem cost_of_2500_pencils :
  cost_of_pencils 30 2500 = 750 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_2500_pencils_l838_83858


namespace NUMINAMATH_CALUDE_break_even_point_l838_83881

def parts_cost : ℕ := 3600
def patent_cost : ℕ := 4500
def variable_cost : ℕ := 25
def marketing_cost : ℕ := 2000
def selling_price : ℕ := 180

def total_fixed_cost : ℕ := parts_cost + patent_cost + marketing_cost
def contribution_margin : ℕ := selling_price - variable_cost

def break_even (n : ℕ) : Prop :=
  n * selling_price ≥ total_fixed_cost + n * variable_cost

theorem break_even_point : 
  ∀ m : ℕ, break_even m → m ≥ 66 :=
by sorry

end NUMINAMATH_CALUDE_break_even_point_l838_83881


namespace NUMINAMATH_CALUDE_tom_teaching_years_l838_83887

theorem tom_teaching_years :
  ∀ (tom_years devin_years : ℕ),
    tom_years + devin_years = 70 →
    devin_years = tom_years / 2 - 5 →
    tom_years = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_tom_teaching_years_l838_83887


namespace NUMINAMATH_CALUDE_max_product_sum_300_l838_83836

theorem max_product_sum_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 ∧ ∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l838_83836


namespace NUMINAMATH_CALUDE_ellipse_symmetric_points_exist_l838_83812

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x + 1/3

-- Define point Q
def Q : ℝ × ℝ := (0, 3)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition for symmetry with respect to the line
def symmetric_points (A B : ℝ × ℝ) : Prop :=
  ∃ (P : ℝ × ℝ), line P.1 P.2 ∧ 
    A.1 + B.1 = 2 * P.1 ∧ A.2 + B.2 = 2 * P.2

-- State the theorem
theorem ellipse_symmetric_points_exist : 
  ∃ (A B : ℝ × ℝ), 
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    symmetric_points A B ∧ 
    3 * dot_product (A.1 - Q.1, A.2 - Q.2) (B.1 - Q.1, B.2 - Q.2) = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_points_exist_l838_83812


namespace NUMINAMATH_CALUDE_rectangular_container_volume_l838_83891

theorem rectangular_container_volume 
  (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : 
  a * b * c = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangular_container_volume_l838_83891


namespace NUMINAMATH_CALUDE_parenthesization_pigeonhole_l838_83869

theorem parenthesization_pigeonhole : ∃ (n : ℕ) (k : ℕ), 
  n > 0 ∧ 
  k > 0 ∧ 
  (2 ^ n > (k * (k + 1))) ∧ 
  (∀ (f : Fin (2^n) → ℤ), ∃ (i j : Fin (2^n)), i ≠ j ∧ f i = f j) := by
  sorry

end NUMINAMATH_CALUDE_parenthesization_pigeonhole_l838_83869


namespace NUMINAMATH_CALUDE_min_blocking_tiles_18x8_l838_83842

/-- Represents an L-shaped tile that covers exactly 3 squares --/
structure LTile :=
  (covers : Nat)

/-- Represents a chessboard --/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Calculates the total number of squares on the chessboard --/
def totalSquares (board : Chessboard) : Nat :=
  board.rows * board.cols

/-- Defines the minimum number of L-tiles needed to block further placement --/
def minBlockingTiles (board : Chessboard) (tile : LTile) : Nat :=
  11

/-- Main theorem: The minimum number of L-tiles to block further placement on an 18x8 board is 11 --/
theorem min_blocking_tiles_18x8 :
  let board : Chessboard := ⟨18, 8⟩
  let tile : LTile := ⟨3⟩
  minBlockingTiles board tile = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_blocking_tiles_18x8_l838_83842
