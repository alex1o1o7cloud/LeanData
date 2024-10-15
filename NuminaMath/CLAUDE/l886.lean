import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l886_88688

theorem min_value_theorem (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m > 0) (h3 : n > 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 2 → (1 / m + 2 / n) ≤ (1 / x + 2 / y) ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b = 2 ∧ 1 / a + 2 / b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l886_88688


namespace NUMINAMATH_CALUDE_expression_sum_l886_88691

theorem expression_sum (d e : ℤ) (h : d ≠ 0) : 
  let original := (16 * d + 17 + 18 * d^2) + (4 * d + 3) + 2 * e
  ∃ (a b c : ℤ), 
    original = a * d + b + c * d^2 + d * e ∧ 
    a + b + c + e = 60 := by
  sorry

end NUMINAMATH_CALUDE_expression_sum_l886_88691


namespace NUMINAMATH_CALUDE_field_trip_attendance_is_76_l886_88611

/-- The number of people on a field trip given the number of vans and buses,
    and the number of people in each vehicle type. -/
def field_trip_attendance (num_vans num_buses : ℕ) (people_per_van people_per_bus : ℕ) : ℕ :=
  num_vans * people_per_van + num_buses * people_per_bus

/-- Theorem stating that the total number of people on the field trip is 76. -/
theorem field_trip_attendance_is_76 :
  field_trip_attendance 2 3 8 20 = 76 := by
  sorry

#eval field_trip_attendance 2 3 8 20

end NUMINAMATH_CALUDE_field_trip_attendance_is_76_l886_88611


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l886_88654

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 24| + |x - 20| = |2*x - 44| :=
by
  -- The unique solution is x = 22
  use 22
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l886_88654


namespace NUMINAMATH_CALUDE_binomial_congruence_l886_88673

theorem binomial_congruence (p a b : ℕ) (hp : Nat.Prime p) (hab : a ≥ b) (hb : b ≥ 0) :
  (Nat.choose (p * (a - b)) p) ≡ (Nat.choose a b) [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_congruence_l886_88673


namespace NUMINAMATH_CALUDE_mean_salary_proof_l886_88685

def salaries : List ℝ := [1000, 2500, 3100, 3650, 1500, 2000]

theorem mean_salary_proof :
  (salaries.sum / salaries.length : ℝ) = 2458.33 := by
  sorry

end NUMINAMATH_CALUDE_mean_salary_proof_l886_88685


namespace NUMINAMATH_CALUDE_no_all_ones_reverse_product_l886_88681

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number consists only of the digit 1 -/
def allOnes (n : ℕ) : Prop := sorry

/-- 
There does not exist a natural number n > 1 such that n multiplied by 
the number formed by reversing its digits results in a number comprised 
entirely of the digit one.
-/
theorem no_all_ones_reverse_product : 
  ¬ ∃ (n : ℕ), n > 1 ∧ allOnes (n * reverseDigits n) := by
  sorry

end NUMINAMATH_CALUDE_no_all_ones_reverse_product_l886_88681


namespace NUMINAMATH_CALUDE_min_value_of_ratio_l886_88652

theorem min_value_of_ratio (x y : ℝ) (h1 : x + y - 3 ≤ 0) (h2 : x - y + 1 ≥ 0) (h3 : y ≥ 1) :
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ - 3 ≤ 0 ∧ x₀ - y₀ + 1 ≥ 0 ∧ y₀ ≥ 1 ∧
    ∀ (x' y' : ℝ), x' + y' - 3 ≤ 0 → x' - y' + 1 ≥ 0 → y' ≥ 1 → y₀ / x₀ ≤ y' / x' :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_ratio_l886_88652


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l886_88675

/-- Proves that the initial alcohol percentage is 25% given the problem conditions -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (final_percentage : ℝ)
  (h_initial_volume : initial_volume = 6)
  (h_added_alcohol : added_alcohol = 3)
  (h_final_percentage : final_percentage = 50)
  (h_alcohol_balance : initial_volume * (initial_percentage / 100) + added_alcohol = 
                       (initial_volume + added_alcohol) * (final_percentage / 100)) :
  initial_percentage = 25 :=
by
  sorry

#check initial_alcohol_percentage

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l886_88675


namespace NUMINAMATH_CALUDE_print_360_pages_in_15_minutes_l886_88674

/-- Calculates the time needed to print a given number of pages at a specific rate. -/
def print_time (pages : ℕ) (rate : ℕ) : ℚ :=
  pages / rate

/-- Theorem stating that printing 360 pages at a rate of 24 pages per minute takes 15 minutes. -/
theorem print_360_pages_in_15_minutes :
  print_time 360 24 = 15 := by
  sorry

end NUMINAMATH_CALUDE_print_360_pages_in_15_minutes_l886_88674


namespace NUMINAMATH_CALUDE_committee_formation_count_l886_88605

/-- The number of ways to form a committee with specified conditions -/
def committee_formations (total_members : ℕ) (committee_size : ℕ) (required_members : ℕ) : ℕ :=
  Nat.choose (total_members - required_members) (committee_size - required_members)

/-- Theorem: The number of ways to form a 5-person committee from a 12-member club,
    where two specific members must always be included, is equal to 120. -/
theorem committee_formation_count :
  committee_formations 12 5 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l886_88605


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l886_88636

/-- Given parallel vectors a and b, prove the minimum value of 3/x + 2/y is 8 -/
theorem min_value_parallel_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a : Fin 2 → ℝ := ![3, -2]
  let b : Fin 2 → ℝ := ![x, y - 1]
  (∃ (k : ℝ), b = k • a) →
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → 3 / x' + 2 / y' ≥ 8) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l886_88636


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l886_88666

/-- Isosceles triangle with given side length and area -/
structure IsoscelesTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Isosceles condition
  isIsosceles : ab = ac
  -- Given side length
  bcLength : bc = 16
  -- Area
  area : ℝ
  areaValue : area = 120

/-- The length of AB in the isosceles triangle -/
def sideLength (t : IsoscelesTriangle) : ℝ := t.ab

/-- Theorem: The length of AB in the given isosceles triangle is 17 -/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) :
  sideLength t = 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l886_88666


namespace NUMINAMATH_CALUDE_ashleys_notebooks_l886_88641

theorem ashleys_notebooks :
  ∀ (notebook_price pencil_price : ℕ) (notebooks_in_93 : ℕ),
    notebook_price + pencil_price = 5 →
    21 * pencil_price + notebooks_in_93 * notebook_price = 93 →
    notebooks_in_93 = 15 →
    ∃ (notebooks_in_5 : ℕ),
      notebooks_in_5 * notebook_price + 1 * pencil_price = 5 ∧
      notebooks_in_5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ashleys_notebooks_l886_88641


namespace NUMINAMATH_CALUDE_isosceles_triangle_part1_isosceles_triangle_part2_l886_88622

/-- Represents an isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  isIsosceles : leg ≥ base
  perimeter : ℝ
  sumOfSides : base + 2 * leg = perimeter

/-- Theorem for part 1 of the problem -/
theorem isosceles_triangle_part1 :
  ∃ (t : IsoscelesTriangle),
    t.perimeter = 20 ∧ t.leg = 2 * t.base ∧ t.base = 4 ∧ t.leg = 8 := by
  sorry

/-- Theorem for part 2 of the problem -/
theorem isosceles_triangle_part2 :
  ∃ (t : IsoscelesTriangle),
    t.perimeter = 20 ∧ t.base = 5 ∧ t.leg = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_part1_isosceles_triangle_part2_l886_88622


namespace NUMINAMATH_CALUDE_vip_seat_cost_l886_88623

theorem vip_seat_cost (total_tickets : ℕ) (total_revenue : ℕ) 
  (general_price : ℕ) (vip_difference : ℕ) :
  total_tickets = 320 →
  total_revenue = 7500 →
  general_price = 15 →
  vip_difference = 212 →
  ∃ (vip_price : ℕ), 
    vip_price = 65 ∧
    (total_tickets - vip_difference) * general_price + 
    vip_difference * vip_price = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_vip_seat_cost_l886_88623


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l886_88631

/-- The probability of selecting two plates of the same color -/
theorem same_color_plate_probability 
  (total_plates : ℕ) 
  (red_plates : ℕ) 
  (blue_plates : ℕ) 
  (h1 : total_plates = red_plates + blue_plates) 
  (h2 : total_plates = 13) 
  (h3 : red_plates = 7) 
  (h4 : blue_plates = 6) : 
  (red_plates.choose 2 + blue_plates.choose 2 : ℚ) / total_plates.choose 2 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_same_color_plate_probability_l886_88631


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tangent_sum_l886_88638

theorem arithmetic_sequence_tangent_sum (x y z : Real) 
  (h1 : y - x = π/3) 
  (h2 : z - y = π/3) : 
  Real.tan x * Real.tan y + Real.tan y * Real.tan z + Real.tan z * Real.tan x = -3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tangent_sum_l886_88638


namespace NUMINAMATH_CALUDE_johns_pill_cost_l886_88662

/-- Calculates the out-of-pocket cost for pills in a 30-day month given the following conditions:
  * Daily pill requirement
  * Cost per pill
  * Insurance coverage percentage
  * Number of days in a month
-/
def outOfPocketCost (dailyPills : ℕ) (costPerPill : ℚ) (insuranceCoverage : ℚ) (daysInMonth : ℕ) : ℚ :=
  let totalPills := dailyPills * daysInMonth
  let totalCost := totalPills * costPerPill
  let insuranceAmount := totalCost * insuranceCoverage
  totalCost - insuranceAmount

/-- Proves that given the specified conditions, John's out-of-pocket cost for pills in a 30-day month is $54 -/
theorem johns_pill_cost :
  outOfPocketCost 2 (3/2) (2/5) 30 = 54 := by
  sorry

end NUMINAMATH_CALUDE_johns_pill_cost_l886_88662


namespace NUMINAMATH_CALUDE_quadratic_function_property_l886_88672

-- Define a quadratic function
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

-- Define the inverse function property
def HasInverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, f (g x) = x) ∧ (∀ x, g (f x) = x)

theorem quadratic_function_property (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f)
  (h2 : HasInverse f)
  (h3 : ∀ x, f x = 3 * (Classical.choose h2) x + 5)
  (h4 : f 1 = 5) :
  f 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l886_88672


namespace NUMINAMATH_CALUDE_fraction_cube_multiply_l886_88699

theorem fraction_cube_multiply (a b : ℚ) : (1 / 3 : ℚ)^3 * (1 / 5 : ℚ) = 1 / 135 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cube_multiply_l886_88699


namespace NUMINAMATH_CALUDE_line_points_k_value_l886_88649

theorem line_points_k_value (m n k : ℝ) : 
  (m = 2 * n + 5) → 
  (m + 2 = 2 * (n + k) + 5) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l886_88649


namespace NUMINAMATH_CALUDE_uncle_bob_parking_probability_l886_88633

/-- The number of parking spaces -/
def total_spaces : ℕ := 20

/-- The number of cars that have already parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent spaces needed for Uncle Bob's truck -/
def needed_spaces : ℕ := 2

/-- The probability of having at least two adjacent empty spaces -/
def probability_adjacent_spaces : ℚ := 232 / 323

theorem uncle_bob_parking_probability :
  let total_combinations := Nat.choose total_spaces parked_cars
  let unfavorable_combinations := Nat.choose (parked_cars + needed_spaces + 1) (needed_spaces + 1)
  (1 : ℚ) - (unfavorable_combinations : ℚ) / (total_combinations : ℚ) = probability_adjacent_spaces :=
sorry

end NUMINAMATH_CALUDE_uncle_bob_parking_probability_l886_88633


namespace NUMINAMATH_CALUDE_smallest_palindrome_div_by_7_l886_88625

/-- A function that checks if a number is a four-digit palindrome -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- A function that checks if a number has an odd first digit -/
def has_odd_first_digit (n : ℕ) : Prop :=
  (n / 1000) % 2 = 1

/-- The theorem stating that 1661 is the smallest four-digit palindrome divisible by 7 with an odd first digit -/
theorem smallest_palindrome_div_by_7 :
  (∀ n : ℕ, is_four_digit_palindrome n ∧ has_odd_first_digit n ∧ n % 7 = 0 → n ≥ 1661) ∧
  is_four_digit_palindrome 1661 ∧ has_odd_first_digit 1661 ∧ 1661 % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_palindrome_div_by_7_l886_88625


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l886_88669

/-- Represents a rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  area_eq : area = 21 * breadth
  length_eq : length = breadth + 10

/-- Theorem stating that a rectangular plot with the given properties has a breadth of 11 meters -/
theorem rectangular_plot_breadth (plot : RectangularPlot) : plot.breadth = 11 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l886_88669


namespace NUMINAMATH_CALUDE_complex_number_equality_l886_88683

theorem complex_number_equality (b : ℝ) : 
  (Complex.re ((1 + b * Complex.I) / (2 + Complex.I)) = 
   Complex.im ((1 + b * Complex.I) / (2 + Complex.I))) → 
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l886_88683


namespace NUMINAMATH_CALUDE_mary_james_seating_probability_l886_88610

-- Define the number of chairs
def total_chairs : ℕ := 10

-- Define the number of available chairs (excluding first and last)
def available_chairs : ℕ := total_chairs - 2

-- Define the probability of not sitting next to each other
def prob_not_adjacent : ℚ := 3/4

theorem mary_james_seating_probability :
  (1 : ℚ) - (available_chairs - 1 : ℚ) / (available_chairs.choose 2 : ℚ) = prob_not_adjacent :=
by sorry

end NUMINAMATH_CALUDE_mary_james_seating_probability_l886_88610


namespace NUMINAMATH_CALUDE_angle_properties_l886_88670

theorem angle_properties (α : Real) (y : Real) :
  -- Angle α is in the second quadrant
  π / 2 < α ∧ α < π →
  -- Point P on its terminal side has coordinates (-√2, y)
  ∃ P : Real × Real, P = (-Real.sqrt 2, y) →
  -- sin α = (√2/4)y
  Real.sin α = (Real.sqrt 2 / 4) * y →
  -- Prove: tan α = -√3
  Real.tan α = -Real.sqrt 3 ∧
  -- Prove: (3sin α · cos α) / (4sin²α + 2cos²α) = -3√3/14
  (3 * Real.sin α * Real.cos α) / (4 * Real.sin α ^ 2 + 2 * Real.cos α ^ 2) = -3 * Real.sqrt 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l886_88670


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l886_88617

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : Nat
  full_shaded : Nat
  half_shaded : Nat

/-- Calculates the fraction of shaded area in a quilt block -/
def shaded_fraction (q : QuiltBlock) : Rat :=
  (q.full_shaded + q.half_shaded / 2 : Rat) / (q.size * q.size)

/-- Theorem stating that a 4x4 quilt block with 2 fully shaded squares and 4 half-shaded squares has 1/4 of its area shaded -/
theorem quilt_shaded_fraction :
  let q : QuiltBlock := { size := 4, full_shaded := 2, half_shaded := 4 }
  shaded_fraction q = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l886_88617


namespace NUMINAMATH_CALUDE_transfer_ratio_l886_88626

def initial_balance : ℕ := 190
def mom_transfer : ℕ := 60
def final_balance : ℕ := 100

def sister_transfer : ℕ := initial_balance - mom_transfer - final_balance

theorem transfer_ratio : 
  sister_transfer * 2 = mom_transfer := by sorry

end NUMINAMATH_CALUDE_transfer_ratio_l886_88626


namespace NUMINAMATH_CALUDE_brother_lower_limit_l886_88624

-- Define Arun's weight
def W : ℝ := sorry

-- Define brother's lower limit
def B : ℝ := sorry

-- Arun's opinion
axiom arun_opinion : 64 < W ∧ W < 72

-- Brother's opinion
axiom brother_opinion : B < W ∧ W < 70

-- Mother's opinion
axiom mother_opinion : W ≤ 67

-- Average weight
axiom average_weight : (W + 67) / 2 = 66

-- Theorem to prove
theorem brother_lower_limit : B > 64 := by sorry

end NUMINAMATH_CALUDE_brother_lower_limit_l886_88624


namespace NUMINAMATH_CALUDE_prob_A_and_B_l886_88629

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.55

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.60

/-- The theorem stating that the probability of both A and B occurring simultaneously
    is equal to the product of their individual probabilities -/
theorem prob_A_and_B : prob_A * prob_B = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_and_B_l886_88629


namespace NUMINAMATH_CALUDE_apples_handed_out_to_students_l886_88696

theorem apples_handed_out_to_students 
  (initial_apples : ℕ) 
  (pies : ℕ) 
  (apples_per_pie : ℕ) 
  (h1 : initial_apples = 62) 
  (h2 : pies = 6) 
  (h3 : apples_per_pie = 9) :
  initial_apples - pies * apples_per_pie = 8 := by
sorry

end NUMINAMATH_CALUDE_apples_handed_out_to_students_l886_88696


namespace NUMINAMATH_CALUDE_binary_multiplication_subtraction_l886_88657

-- Define binary numbers as natural numbers
def binary_11011 : ℕ := 27
def binary_1101 : ℕ := 13
def binary_1010 : ℕ := 10

-- Define the expected result
def expected_result : ℕ := 409

-- Theorem statement
theorem binary_multiplication_subtraction :
  (binary_11011 * binary_1101) - binary_1010 = expected_result :=
by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_subtraction_l886_88657


namespace NUMINAMATH_CALUDE_palic_characterization_l886_88684

/-- Palic function definition -/
def isPalic (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  Continuous f ∧
  ∀ x y z : ℝ, f x + f y + f z = f (a*x + b*y + c*z) + f (b*x + c*y + a*z) + f (c*x + a*y + b*z)

/-- Theorem: Characterization of Palic functions -/
theorem palic_characterization (a b c : ℝ) 
    (h1 : a + b + c = 1)
    (h2 : a^2 + b^2 + c^2 = 1)
    (h3 : a^3 + b^3 + c^3 ≠ 1)
    (f : ℝ → ℝ)
    (hf : isPalic f a b c) :
  ∃ p q r : ℝ, ∀ x : ℝ, f x = p * x^2 + q * x + r :=
sorry

end NUMINAMATH_CALUDE_palic_characterization_l886_88684


namespace NUMINAMATH_CALUDE_f_log_one_third_36_l886_88677

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x % 3 ∧ x % 3 < 1 then 3^(x % 3) - 1
  else if 1 ≤ x % 3 ∧ x % 3 < 2 then -(3^(2 - (x % 3)) - 1)
  else -(3^((x % 3) - 2) - 1)

-- State the theorem
theorem f_log_one_third_36 (h1 : ∀ x, f (-x) = -f x) 
                            (h2 : ∀ x, f (x + 3) = f x) 
                            (h3 : ∀ x, 0 ≤ x → x < 1 → f x = 3^x - 1) :
  f (Real.log 36 / Real.log (1/3)) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_f_log_one_third_36_l886_88677


namespace NUMINAMATH_CALUDE_cosine_value_l886_88648

theorem cosine_value (α : ℝ) (h : 2 * Real.cos (2 * α) + 9 * Real.sin α = 4) :
  Real.cos α = Real.sqrt 15 / 4 ∨ Real.cos α = -Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_l886_88648


namespace NUMINAMATH_CALUDE_sum_of_numbers_l886_88680

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l886_88680


namespace NUMINAMATH_CALUDE_complex_equation_solution_l886_88619

theorem complex_equation_solution (b : ℝ) : 
  (2 - Complex.I) * (4 * Complex.I) = 4 + b * Complex.I → b = 8 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l886_88619


namespace NUMINAMATH_CALUDE_cricket_average_problem_l886_88695

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : Nat
  totalRuns : Nat
  deriving Repr

/-- Calculates the average runs per innings -/
def averageRuns (player : CricketPlayer) : Rat :=
  player.totalRuns / player.innings

theorem cricket_average_problem (player : CricketPlayer) 
  (h1 : player.innings = 20)
  (h2 : averageRuns { innings := player.innings + 1, totalRuns := player.totalRuns + 158 } = 
        averageRuns player + 6) :
  averageRuns player = 32 := by
  sorry


end NUMINAMATH_CALUDE_cricket_average_problem_l886_88695


namespace NUMINAMATH_CALUDE_circle_area_in_square_l886_88659

theorem circle_area_in_square (square_area : Real) (circle_area : Real) : 
  square_area = 400 →
  circle_area = Real.pi * (Real.sqrt square_area / 2)^2 →
  circle_area = 100 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_area_in_square_l886_88659


namespace NUMINAMATH_CALUDE_sum_distinct_prime_divisors_of_1260_l886_88658

/-- The sum of the distinct prime integer divisors of 1260 is 17. -/
theorem sum_distinct_prime_divisors_of_1260 : 
  (Finset.sum (Finset.filter Nat.Prime (Nat.divisors 1260)) id) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_divisors_of_1260_l886_88658


namespace NUMINAMATH_CALUDE_vehicle_value_fraction_l886_88660

def vehicle_value_this_year : ℚ := 16000
def vehicle_value_last_year : ℚ := 20000

theorem vehicle_value_fraction :
  vehicle_value_this_year / vehicle_value_last_year = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_fraction_l886_88660


namespace NUMINAMATH_CALUDE_vectors_same_direction_l886_88671

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define points A, B, C
variable (A B C : V)

-- Define the vectors
def AB : V := B - A
def AC : V := C - A
def BC : V := C - B

-- Define the theorem
theorem vectors_same_direction (h : ‖AB A B‖ = ‖AC A C‖ + ‖BC B C‖) :
  ∃ (k : ℝ), k > 0 ∧ AC A C = k • (BC B C) := by
  sorry

end NUMINAMATH_CALUDE_vectors_same_direction_l886_88671


namespace NUMINAMATH_CALUDE_equal_slope_implies_parallel_l886_88606

/-- Two lines in a plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Theorem: If two non-intersecting lines have equal slopes, then they are parallel -/
theorem equal_slope_implies_parallel (l1 l2 : Line) 
  (h1 : l1.slope = l2.slope) 
  (h2 : l1.yIntercept ≠ l2.yIntercept) : 
  parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_equal_slope_implies_parallel_l886_88606


namespace NUMINAMATH_CALUDE_aladdin_journey_theorem_l886_88698

-- Define the circle (equator)
def Equator : Real := 40000

-- Define Aladdin's path
def AladdinPath : Set ℝ → Prop :=
  λ path => ∀ x, x ∈ path → 0 ≤ x ∧ x < Equator

-- Define the property of covering every point on the equator
def CoversEquator (path : Set ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x < Equator → ∃ y ∈ path, y % Equator = x

-- Define the westward travel limit
def WestwardLimit : Real := 19000

-- Define the theorem
theorem aladdin_journey_theorem (path : Set ℝ) 
  (h_path : AladdinPath path)
  (h_covers : CoversEquator path)
  (h_westward : ∀ x ∈ path, x ≤ WestwardLimit) :
  ∃ x ∈ path, abs (x % Equator - x) ≥ Equator / 2 := by
sorry

end NUMINAMATH_CALUDE_aladdin_journey_theorem_l886_88698


namespace NUMINAMATH_CALUDE_coin_toss_probability_l886_88613

theorem coin_toss_probability : 
  let p_head : ℝ := 1/2  -- Probability of getting heads on a single toss
  let n : ℕ := 3  -- Number of tosses
  let p_all_tails : ℝ := (1 - p_head)^n  -- Probability of getting all tails
  1 - p_all_tails = 7/8 := by sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l886_88613


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_range_l886_88667

theorem sqrt_x_minus_3_real_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_range_l886_88667


namespace NUMINAMATH_CALUDE_cos_585_degrees_l886_88643

theorem cos_585_degrees : Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_585_degrees_l886_88643


namespace NUMINAMATH_CALUDE_q_value_l886_88628

theorem q_value (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 12) : q = 6 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_q_value_l886_88628


namespace NUMINAMATH_CALUDE_function_odd_iff_sum_squares_zero_l886_88634

/-- The function f(x) = x|x-a| + b is odd if and only if a^2 + b^2 = 0 -/
theorem function_odd_iff_sum_squares_zero (a b : ℝ) :
  (∀ x : ℝ, x * |x - a| + b = -((-x) * |(-x) - a| + b)) ↔ a^2 + b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_odd_iff_sum_squares_zero_l886_88634


namespace NUMINAMATH_CALUDE_area_le_sqrt_product_area_eq_sqrt_product_iff_rectangle_l886_88668

/-- A quadrilateral circumscribed about a circle -/
structure CircumscribedQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  area_pos : 0 < area

/-- The theorem stating that the area of a circumscribed quadrilateral is at most the square root of the product of its side lengths -/
theorem area_le_sqrt_product (q : CircumscribedQuadrilateral) : 
  q.area ≤ Real.sqrt (q.a * q.b * q.c * q.d) := by
  sorry

/-- The condition for equality in the above inequality -/
def is_rectangle (q : CircumscribedQuadrilateral) : Prop :=
  (q.a = q.c ∧ q.b = q.d) ∨ (q.a = q.b ∧ q.c = q.d)

/-- The theorem stating that equality holds if and only if the quadrilateral is a rectangle -/
theorem area_eq_sqrt_product_iff_rectangle (q : CircumscribedQuadrilateral) :
  q.area = Real.sqrt (q.a * q.b * q.c * q.d) ↔ is_rectangle q := by
  sorry

end NUMINAMATH_CALUDE_area_le_sqrt_product_area_eq_sqrt_product_iff_rectangle_l886_88668


namespace NUMINAMATH_CALUDE_bowler_previous_wickets_l886_88600

/-- Bowling average calculation -/
def bowling_average (runs : ℚ) (wickets : ℚ) : ℚ := runs / wickets

theorem bowler_previous_wickets 
  (initial_average : ℚ)
  (last_match_wickets : ℚ)
  (last_match_runs : ℚ)
  (average_decrease : ℚ)
  (h1 : initial_average = 12.4)
  (h2 : last_match_wickets = 7)
  (h3 : last_match_runs = 26)
  (h4 : average_decrease = 0.4) :
  ∃ (previous_wickets : ℚ),
    previous_wickets = 145 ∧
    bowling_average (initial_average * previous_wickets + last_match_runs) (previous_wickets + last_match_wickets) = initial_average - average_decrease :=
sorry

end NUMINAMATH_CALUDE_bowler_previous_wickets_l886_88600


namespace NUMINAMATH_CALUDE_polynomial_factorization_l886_88621

theorem polynomial_factorization (x : ℝ) :
  x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l886_88621


namespace NUMINAMATH_CALUDE_savings_is_six_dollars_l886_88682

-- Define the number of notebooks
def num_notebooks : ℕ := 8

-- Define the original price per notebook
def original_price : ℚ := 3

-- Define the discount rate
def discount_rate : ℚ := 1/4

-- Define the function to calculate savings
def calculate_savings (n : ℕ) (p : ℚ) (d : ℚ) : ℚ :=
  n * p * d

-- Theorem stating that the savings is $6.00
theorem savings_is_six_dollars :
  calculate_savings num_notebooks original_price discount_rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_savings_is_six_dollars_l886_88682


namespace NUMINAMATH_CALUDE_undefined_fraction_l886_88639

theorem undefined_fraction (a b : ℝ) (h1 : a = 4) (h2 : b = -4) :
  ¬∃x : ℝ, x = 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_undefined_fraction_l886_88639


namespace NUMINAMATH_CALUDE_book_sales_proof_l886_88618

/-- Calculates the number of copies sold given the revenue per book, agent's commission percentage, and total amount kept by the author. -/
def calculate_copies_sold (revenue_per_book : ℚ) (agent_commission_percent : ℚ) (total_kept : ℚ) : ℚ :=
  total_kept / (revenue_per_book * (1 - agent_commission_percent / 100))

/-- Proves that given the specific conditions, the number of copies sold is 900,000. -/
theorem book_sales_proof (revenue_per_book : ℚ) (agent_commission_percent : ℚ) (total_kept : ℚ) 
    (h1 : revenue_per_book = 2)
    (h2 : agent_commission_percent = 10)
    (h3 : total_kept = 1620000) :
  calculate_copies_sold revenue_per_book agent_commission_percent total_kept = 900000 := by
  sorry

end NUMINAMATH_CALUDE_book_sales_proof_l886_88618


namespace NUMINAMATH_CALUDE_sock_drawer_problem_l886_88607

/-- The number of ways to choose k items from n distinguishable items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose a pair of socks of the same color -/
def sameColorPairs (white brown blue red : ℕ) : ℕ :=
  choose white 2 + choose brown 2 + choose blue 2 + choose red 2

theorem sock_drawer_problem :
  sameColorPairs 5 5 3 2 = 24 := by sorry

end NUMINAMATH_CALUDE_sock_drawer_problem_l886_88607


namespace NUMINAMATH_CALUDE_max_values_on_sphere_l886_88637

theorem max_values_on_sphere (x y z : ℝ) :
  x^2 + y^2 + z^2 = 4 →
  (∃ (max_xz_yz : ℝ), ∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 4 → x' * z' + y' * z' ≤ max_xz_yz ∧ max_xz_yz = 2 * Real.sqrt 2) ∧
  (x + y + z = 0 →
    ∃ (max_z : ℝ), ∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 4 ∧ x' + y' + z' = 0 → z' ≤ max_z ∧ max_z = (2 * Real.sqrt 6) / 3) := by
  sorry

end NUMINAMATH_CALUDE_max_values_on_sphere_l886_88637


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l886_88632

theorem arithmetic_evaluation : 3 * 4^2 - (8 / 2) = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l886_88632


namespace NUMINAMATH_CALUDE_roots_sum_product_l886_88603

theorem roots_sum_product (a b : ℝ) : 
  (a^4 - 6*a - 1 = 0) →
  (b^4 - 6*b - 1 = 0) →
  (a ≠ b) →
  (ab + 2*a + 2*b = 1.5 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_product_l886_88603


namespace NUMINAMATH_CALUDE_forty_men_handshakes_l886_88686

/-- The maximum number of handshakes without cyclic handshakes for n people -/
def max_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 40 men, the maximum number of handshakes without cyclic handshakes is 780 -/
theorem forty_men_handshakes : max_handshakes 40 = 780 := by
  sorry

end NUMINAMATH_CALUDE_forty_men_handshakes_l886_88686


namespace NUMINAMATH_CALUDE_solution_comparison_l886_88609

theorem solution_comparison (a a' b b' : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0) :
  (-b / a < -b' / a') ↔ (b' / a' < b / a) := by
  sorry

end NUMINAMATH_CALUDE_solution_comparison_l886_88609


namespace NUMINAMATH_CALUDE_multiple_of_second_number_l886_88690

theorem multiple_of_second_number (x y m : ℤ) : 
  y = m * x + 3 → 
  x + y = 27 → 
  y = 19 → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_second_number_l886_88690


namespace NUMINAMATH_CALUDE_common_difference_is_three_l886_88655

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_is_three
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 11 = 24)
  (h_a4 : a 4 = 3) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_three_l886_88655


namespace NUMINAMATH_CALUDE_range_of_2x_plus_y_range_of_a_l886_88653

-- Define the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2*y

-- Theorem for the range of 2x + y
theorem range_of_2x_plus_y :
  ∀ x y : ℝ, on_circle x y → -Real.sqrt 5 + 1 ≤ 2*x + y ∧ 2*x + y ≤ Real.sqrt 5 + 1 := by
  sorry

-- Theorem for the range of a
theorem range_of_a :
  (∀ x y : ℝ, on_circle x y → ∀ a : ℝ, x + y + a ≥ 0) →
  ∀ a : ℝ, a ≥ Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2x_plus_y_range_of_a_l886_88653


namespace NUMINAMATH_CALUDE_cos_2x_satisfies_conditions_l886_88689

theorem cos_2x_satisfies_conditions (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos (2 * x)
  (f x = f (-x)) ∧ (f (x - π) = f x) := by sorry

end NUMINAMATH_CALUDE_cos_2x_satisfies_conditions_l886_88689


namespace NUMINAMATH_CALUDE_bike_tractor_speed_ratio_l886_88635

/-- Given the speeds and distances of vehicles, prove the ratio of bike speed to tractor speed --/
theorem bike_tractor_speed_ratio :
  ∀ (car_speed bike_speed tractor_speed : ℝ),
  car_speed = (9/5) * bike_speed →
  tractor_speed = 575 / 25 →
  car_speed = 331.2 / 4 →
  bike_speed / tractor_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_bike_tractor_speed_ratio_l886_88635


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_equal_coefficients_l886_88663

theorem infinite_solutions_imply_equal_coefficients (a b : ℝ) :
  (∀ x : ℝ, a * (a - x) - b * (b - x) = 0) →
  a - b = 0 := by
sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_equal_coefficients_l886_88663


namespace NUMINAMATH_CALUDE_distance_between_homes_l886_88612

/-- The distance between Xiaohong's and Xiaoli's homes given their walking speeds and arrival times -/
theorem distance_between_homes 
  (x_speed : ℝ) 
  (l_speed_to_cinema : ℝ) 
  (l_speed_from_cinema : ℝ) 
  (delay : ℝ) 
  (h_x_speed : x_speed = 52) 
  (h_l_speed_to_cinema : l_speed_to_cinema = 70) 
  (h_l_speed_from_cinema : l_speed_from_cinema = 90) 
  (h_delay : delay = 4) : 
  ∃ (t : ℝ), x_speed * t + l_speed_to_cinema * t = 2196 ∧ 
  x_speed * (t + delay + (x_speed * t / x_speed)) = l_speed_from_cinema * ((x_speed * t / x_speed) - delay) :=
sorry

end NUMINAMATH_CALUDE_distance_between_homes_l886_88612


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l886_88646

theorem gcd_of_three_numbers : Nat.gcd 9118 (Nat.gcd 12173 33182) = 47 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l886_88646


namespace NUMINAMATH_CALUDE_product_even_implies_factor_even_l886_88678

theorem product_even_implies_factor_even (a b : ℕ) : 
  Even (a * b) → Even a ∨ Even b := by sorry

end NUMINAMATH_CALUDE_product_even_implies_factor_even_l886_88678


namespace NUMINAMATH_CALUDE_max_squares_covered_l886_88656

/-- Represents a square card with side length 2 inches -/
structure Card where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- Represents a checkerboard with 1-inch squares -/
structure Checkerboard where
  square_size : ℝ
  square_size_eq : square_size = 1

/-- The maximum number of squares that can be covered by the card -/
def max_covered_squares : ℕ := 16

/-- Theorem stating the maximum number of squares that can be covered -/
theorem max_squares_covered (card : Card) (board : Checkerboard) :
  ∃ (n : ℕ), n = max_covered_squares ∧ 
  n = (max_covered_squares : ℝ) ∧
  ∀ (m : ℕ), (m : ℝ) ≤ (card.side_length / board.square_size) ^ 2 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_squares_covered_l886_88656


namespace NUMINAMATH_CALUDE_power_function_through_point_l886_88614

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the theorem
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l886_88614


namespace NUMINAMATH_CALUDE_annas_money_l886_88616

theorem annas_money (original : ℝ) : 
  (original - original * (1/4) = 24) → original = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_annas_money_l886_88616


namespace NUMINAMATH_CALUDE_sum_between_equals_1999_l886_88642

theorem sum_between_equals_1999 :
  ∀ x y : ℕ, x < y →
  (((x + 1 + (y - 1)) / 2) * (y - x - 1) = 1999) →
  ((x = 1998 ∧ y = 2000) ∨ (x = 998 ∧ y = 1001)) :=
by sorry

end NUMINAMATH_CALUDE_sum_between_equals_1999_l886_88642


namespace NUMINAMATH_CALUDE_polar_equation_circle_l886_88661

theorem polar_equation_circle (ρ : ℝ → ℝ → ℝ) (x y : ℝ) :
  (ρ = λ _ _ => 5) → (x^2 + y^2 = 25) :=
sorry

end NUMINAMATH_CALUDE_polar_equation_circle_l886_88661


namespace NUMINAMATH_CALUDE_computers_waiting_for_parts_l886_88630

theorem computers_waiting_for_parts (total : ℕ) (unfixable_percent : ℚ) (fixed_immediately : ℕ) : 
  total = 20 →
  unfixable_percent = 1/5 →
  fixed_immediately = 8 →
  (total - (unfixable_percent * total).num - fixed_immediately : ℚ) / total = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_computers_waiting_for_parts_l886_88630


namespace NUMINAMATH_CALUDE_point_position_l886_88644

theorem point_position (x : ℝ) (h1 : x < -2) (h2 : |x - (-2)| = 5) : x = -7 := by
  sorry

end NUMINAMATH_CALUDE_point_position_l886_88644


namespace NUMINAMATH_CALUDE_integral_split_l886_88608

-- Define f as a real-valued function on the real line
variable (f : ℝ → ℝ)

-- State the theorem
theorem integral_split (h : ∫ x in (1:ℝ)..(3:ℝ), f x = 56) :
  ∫ x in (1:ℝ)..(2:ℝ), f x + ∫ x in (2:ℝ)..(3:ℝ), f x = 56 := by
  sorry

end NUMINAMATH_CALUDE_integral_split_l886_88608


namespace NUMINAMATH_CALUDE_triangle_area_proof_l886_88601

theorem triangle_area_proof (z₁ z₂ : ℂ) (h1 : Complex.abs z₂ = 4) 
  (h2 : 4 * z₁^2 - 2 * z₁ * z₂ + z₂^2 = 0) : 
  let O : ℂ := 0
  let P : ℂ := z₁
  let Q : ℂ := z₂
  Real.sqrt 3 * (Complex.abs (z₁ - O) * Complex.abs (z₂ - O) * Real.sin (Real.pi / 3)) = 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l886_88601


namespace NUMINAMATH_CALUDE_areas_product_eq_volume_squared_l886_88676

/-- A rectangular prism with dimensions x, y, and z. -/
structure RectangularPrism where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The volume of a rectangular prism. -/
def volume (p : RectangularPrism) : ℝ :=
  p.x * p.y * p.z

/-- The areas of the top, back, and lateral face of a rectangular prism. -/
def areas (p : RectangularPrism) : ℝ × ℝ × ℝ :=
  (p.x * p.y, p.y * p.z, p.z * p.x)

/-- The theorem stating that the product of the areas equals the square of the volume. -/
theorem areas_product_eq_volume_squared (p : RectangularPrism) :
  let (top, back, lateral) := areas p
  top * back * lateral = (volume p) ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_areas_product_eq_volume_squared_l886_88676


namespace NUMINAMATH_CALUDE_roots_shift_l886_88647

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 6*x + 5

-- Define the resulting polynomial
def resulting_poly (x : ℝ) : ℝ := x^3 + 9*x^2 + 21*x + 14

theorem roots_shift :
  ∀ (a b c : ℝ),
  (original_poly a = 0 ∧ original_poly b = 0 ∧ original_poly c = 0) →
  (∀ x : ℝ, resulting_poly x = 0 ↔ (x = a - 3 ∨ x = b - 3 ∨ x = c - 3)) :=
by sorry

end NUMINAMATH_CALUDE_roots_shift_l886_88647


namespace NUMINAMATH_CALUDE_stratified_sampling_sum_l886_88692

def total_population : ℕ := 40 + 10 + 30 + 20

def strata : List ℕ := [40, 10, 30, 20]

def sample_size : ℕ := 20

def stratified_sample (stratum : ℕ) : ℕ :=
  (stratum * sample_size) / total_population

theorem stratified_sampling_sum :
  stratified_sample (strata[1]) + stratified_sample (strata[3]) = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sum_l886_88692


namespace NUMINAMATH_CALUDE_indeterminate_m_l886_88651

/-- An odd function from ℝ to ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem indeterminate_m (f : ℝ → ℝ) (m : ℝ) 
  (hodd : OddFunction f) (hm : f m = 2) (hm2 : f (m^2 - 2) = -2) :
  ¬ (∀ n : ℝ, f n = 2 → n = m) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_m_l886_88651


namespace NUMINAMATH_CALUDE_square_fraction_count_l886_88664

theorem square_fraction_count : 
  ∃! (count : ℕ), count = 2 ∧ 
    (∀ n : ℤ, (∃ k : ℤ, n / (30 - 2*n) = k^2) ↔ (n = 0 ∨ n = 10)) := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_count_l886_88664


namespace NUMINAMATH_CALUDE_equation_solution_l886_88627

theorem equation_solution (x y : ℝ) :
  (Real.sqrt (8 * x) / Real.sqrt (4 * (y - 2)) = 3) →
  (x = (9 * y - 18) / 2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l886_88627


namespace NUMINAMATH_CALUDE_spider_permutations_l886_88679

/-- Represents the number of legs a spider has -/
def num_legs : ℕ := 8

/-- Represents the number of items per leg -/
def items_per_leg : ℕ := 3

/-- Represents the total number of items -/
def total_items : ℕ := num_legs * items_per_leg

/-- Represents the number of valid orderings per leg -/
def valid_orderings_per_leg : ℕ := 3

/-- Represents the total number of orderings per leg -/
def total_orderings_per_leg : ℕ := 6

/-- Represents the probability of a valid ordering for one leg -/
def prob_valid_ordering : ℚ := 1 / 2

/-- Theorem: The number of valid permutations for a spider to put on its items
    with the given constraints is equal to 24! / 2^8 -/
theorem spider_permutations :
  (Nat.factorial total_items) / (2 ^ num_legs) =
  (Nat.factorial total_items) * (prob_valid_ordering ^ num_legs) :=
sorry

end NUMINAMATH_CALUDE_spider_permutations_l886_88679


namespace NUMINAMATH_CALUDE_sqrt_mixed_fraction_equality_l886_88650

theorem sqrt_mixed_fraction_equality (k n : ℝ) (h1 : k > 0) (h2 : n > 0) (h3 : n + 1 = k^2) :
  Real.sqrt (k * (k / n)) = k * Real.sqrt (k / n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_mixed_fraction_equality_l886_88650


namespace NUMINAMATH_CALUDE_total_triangles_is_53_l886_88645

/-- Represents a rectangular figure with internal divisions -/
structure RectangularFigure where
  /-- The number of smallest right triangles -/
  small_right_triangles : ℕ
  /-- The number of isosceles triangles with base equal to the width -/
  width_isosceles_triangles : ℕ
  /-- The number of isosceles triangles with base equal to half the length -/
  half_length_isosceles_triangles : ℕ
  /-- The number of large right triangles -/
  large_right_triangles : ℕ
  /-- The number of large isosceles triangles with base equal to the full width -/
  large_isosceles_triangles : ℕ

/-- Calculates the total number of triangles in the figure -/
def total_triangles (figure : RectangularFigure) : ℕ :=
  figure.small_right_triangles +
  figure.width_isosceles_triangles +
  figure.half_length_isosceles_triangles +
  figure.large_right_triangles +
  figure.large_isosceles_triangles

/-- The specific rectangular figure described in the problem -/
def problem_figure : RectangularFigure :=
  { small_right_triangles := 24
  , width_isosceles_triangles := 6
  , half_length_isosceles_triangles := 8
  , large_right_triangles := 12
  , large_isosceles_triangles := 3
  }

/-- Theorem stating that the total number of triangles in the problem figure is 53 -/
theorem total_triangles_is_53 : total_triangles problem_figure = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_53_l886_88645


namespace NUMINAMATH_CALUDE_air_quality_probability_l886_88604

theorem air_quality_probability (p_good : ℝ) (p_consecutive : ℝ) 
  (h1 : p_good = 0.75) (h2 : p_consecutive = 0.6) : 
  p_consecutive / p_good = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probability_l886_88604


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_16_l886_88640

theorem sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 ∨ Real.sqrt (Real.sqrt 16) = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_16_l886_88640


namespace NUMINAMATH_CALUDE_compute_expression_simplify_expression_l886_88620

-- Part 1
theorem compute_expression : (1/2)⁻¹ - Real.sqrt 3 * Real.cos (30 * π / 180) + (2014 - Real.pi)^0 = 3/2 := by
  sorry

-- Part 2
theorem simplify_expression (a : ℝ) : a * (a + 1) - (a + 1) * (a - 1) = a + 1 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_simplify_expression_l886_88620


namespace NUMINAMATH_CALUDE_possible_b_values_l886_88693

/-- The cubic polynomial p(x) = x^3 + ax + b -/
def p (a b x : ℝ) : ℝ := x^3 + a*x + b

/-- The cubic polynomial q(x) = x^3 + ax + b + 150 -/
def q (a b x : ℝ) : ℝ := x^3 + a*x + b + 150

/-- Theorem stating the possible values of b given the conditions -/
theorem possible_b_values (a b r s : ℝ) : 
  (p a b r = 0 ∧ p a b s = 0) →  -- r and s are roots of p(x)
  (q a b (r+3) = 0 ∧ q a b (s-5) = 0) →  -- r+3 and s-5 are roots of q(x)
  b = 0 ∨ b = 12082 := by
sorry

end NUMINAMATH_CALUDE_possible_b_values_l886_88693


namespace NUMINAMATH_CALUDE_shortest_chord_parallel_and_separate_l886_88687

/-- Circle with center at origin and radius r -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

/-- Point P inside the circle -/
structure PointInCircle (r : ℝ) where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0
  h3 : a^2 + b^2 < r^2

/-- Line l1 containing the shortest chord through P -/
def ShortestChordLine (r : ℝ) (p : PointInCircle r) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | p.a * q.1 + p.b * q.2 = p.a^2 + p.b^2}

/-- Line l2 -/
def Line_l2 (r : ℝ) (p : PointInCircle r) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | p.b * q.1 - p.a * q.2 + r^2 = 0}

/-- Two lines are parallel -/
def Parallel (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l1 ↔ (x, y) ∈ l2 ∨ (x + k, y) ∈ l2

/-- A line is separate from a circle -/
def Separate (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∀ (p : ℝ × ℝ), p ∈ l → p ∉ c

theorem shortest_chord_parallel_and_separate (r : ℝ) (p : PointInCircle r) :
  Parallel (ShortestChordLine r p) (Line_l2 r p) ∧
  Separate (Line_l2 r p) (Circle r) := by
  sorry

end NUMINAMATH_CALUDE_shortest_chord_parallel_and_separate_l886_88687


namespace NUMINAMATH_CALUDE_inequality_property_l886_88602

theorem inequality_property (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a * c < b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l886_88602


namespace NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l886_88665

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l886_88665


namespace NUMINAMATH_CALUDE_no_arithmetic_mean_among_fractions_l886_88615

theorem no_arithmetic_mean_among_fractions : 
  let a := 8 / 13
  let b := 11 / 17
  let c := 5 / 8
  ¬(a = (b + c) / 2 ∨ b = (a + c) / 2 ∨ c = (a + b) / 2) := by
sorry

end NUMINAMATH_CALUDE_no_arithmetic_mean_among_fractions_l886_88615


namespace NUMINAMATH_CALUDE_equation_has_three_solutions_l886_88697

/-- The number of distinct complex solutions to the equation (z^4 - 1) / (z^3 - 3z + 2) = 0 -/
def num_solutions : ℕ := 3

/-- The equation (z^4 - 1) / (z^3 - 3z + 2) = 0 -/
def equation (z : ℂ) : Prop :=
  (z^4 - 1) / (z^3 - 3*z + 2) = 0

theorem equation_has_three_solutions :
  ∃ (S : Finset ℂ), S.card = num_solutions ∧
    (∀ z ∈ S, equation z) ∧
    (∀ z : ℂ, equation z → z ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_equation_has_three_solutions_l886_88697


namespace NUMINAMATH_CALUDE_perfect_squares_of_cube_sums_l886_88694

theorem perfect_squares_of_cube_sums : 
  ∃ (a b c d : ℕ),
    (1^3 + 2^3 = a^2) ∧ 
    (1^3 + 2^3 + 3^3 = b^2) ∧ 
    (1^3 + 2^3 + 3^3 + 4^3 = c^2) ∧ 
    (1^3 + 2^3 + 3^3 + 4^3 + 5^3 = d^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_of_cube_sums_l886_88694
