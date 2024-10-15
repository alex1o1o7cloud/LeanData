import Mathlib

namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l1333_133383

theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∃ x : ℕ+, (3 * x^2 + a * x + 26) / (x + 1) ≤ 2) →
  a ≤ -15 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l1333_133383


namespace NUMINAMATH_CALUDE_sin_to_cos_l1333_133332

theorem sin_to_cos (α : Real) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_to_cos_l1333_133332


namespace NUMINAMATH_CALUDE_derivative_sin_3x_at_pi_9_l1333_133325

theorem derivative_sin_3x_at_pi_9 :
  let f : ℝ → ℝ := λ x ↦ Real.sin (3 * x)
  let x₀ : ℝ := π / 9
  (deriv f) x₀ = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_3x_at_pi_9_l1333_133325


namespace NUMINAMATH_CALUDE_tan_160_gt_tan_neg_23_l1333_133337

theorem tan_160_gt_tan_neg_23 : Real.tan (160 * π / 180) > Real.tan (-23 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tan_160_gt_tan_neg_23_l1333_133337


namespace NUMINAMATH_CALUDE_tony_preparation_time_l1333_133357

/-- The total time Tony spent preparing to be an astronaut -/
def total_preparation_time (
  science_degree_time : ℝ
  ) (num_other_degrees : ℕ
  ) (physics_grad_time : ℝ
  ) (scientist_work_time : ℝ
  ) (num_internships : ℕ
  ) (internship_duration : ℝ
  ) : ℝ :=
  science_degree_time +
  num_other_degrees * science_degree_time +
  physics_grad_time +
  scientist_work_time +
  num_internships * internship_duration

/-- Theorem stating that Tony's total preparation time is 18.5 years -/
theorem tony_preparation_time :
  total_preparation_time 4 2 2 3 3 0.5 = 18.5 := by
  sorry


end NUMINAMATH_CALUDE_tony_preparation_time_l1333_133357


namespace NUMINAMATH_CALUDE_pi_is_irrational_l1333_133358

-- Define the property of being an infinite non-repeating decimal
def is_infinite_non_repeating_decimal (x : ℝ) : Prop := sorry

-- Define the property of being an irrational number
def is_irrational (x : ℝ) : Prop := sorry

-- State the theorem
theorem pi_is_irrational :
  is_infinite_non_repeating_decimal π →
  (∀ x : ℝ, is_infinite_non_repeating_decimal x → is_irrational x) →
  is_irrational π :=
by sorry

end NUMINAMATH_CALUDE_pi_is_irrational_l1333_133358


namespace NUMINAMATH_CALUDE_shaded_area_square_circles_l1333_133306

/-- The shaded area between a square and four circles --/
theorem shaded_area_square_circles (s : ℝ) (r : ℝ) (h1 : s = 10) (h2 : r = 3 * Real.sqrt 3) :
  s^2 - 4 * (π * r^2 / 4) - 8 * (s / 2 * Real.sqrt ((3 * Real.sqrt 3)^2 - (s / 2)^2) / 2) = 
    100 - 27 * π - 20 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_circles_l1333_133306


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1333_133316

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 225) : x + y = 650 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1333_133316


namespace NUMINAMATH_CALUDE_units_digit_of_p_plus_5_l1333_133333

/-- A function that returns the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A predicate that checks if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem units_digit_of_p_plus_5 (p : ℕ) 
  (h1 : p > 0)
  (h2 : isEven p)
  (h3 : unitsDigit p > 0)
  (h4 : unitsDigit (p^3) - unitsDigit (p^2) = 0) :
  unitsDigit (p + 5) = 1 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_p_plus_5_l1333_133333


namespace NUMINAMATH_CALUDE_extended_triangle_area_l1333_133392

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem extended_triangle_area (t : Triangle) :
  ∃ (new_area : ℝ), new_area = 7 * t.area :=
sorry

end NUMINAMATH_CALUDE_extended_triangle_area_l1333_133392


namespace NUMINAMATH_CALUDE_rectangle_area_l1333_133374

/-- Given a rectangle with perimeter 40 and one side length 5, prove its area is 75 -/
theorem rectangle_area (perimeter : ℝ) (side : ℝ) (h1 : perimeter = 40) (h2 : side = 5) :
  let other_side := perimeter / 2 - side
  side * other_side = 75 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1333_133374


namespace NUMINAMATH_CALUDE_product_of_two_digit_numbers_l1333_133319

theorem product_of_two_digit_numbers (a b c d : ℕ) : 
  a < 10 → b < 10 → c < 10 → d < 10 →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (10 * a + b) * (10 * c + b) = 111 * d →
  a + b + c + d = 21 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_digit_numbers_l1333_133319


namespace NUMINAMATH_CALUDE_f_properties_l1333_133356

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x + 8

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 3

-- Theorem for extreme values and max/min in the interval
theorem f_properties :
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x) ∧
  (∀ (x : ℝ), x ∈ interval → f x ≤ 14) ∧
  (∀ (x : ℝ), x ∈ interval → f x ≥ -15) ∧
  (f 1 = 13 ∧ f 2 = 12) ∧
  (∀ (x : ℝ), f x ≥ 12 → x = 1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1333_133356


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1333_133314

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a / (a + 3 * b)) + Real.sqrt (b / (b + 3 * a)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1333_133314


namespace NUMINAMATH_CALUDE_average_running_time_l1333_133365

theorem average_running_time (total_students : ℕ) 
  (sixth_grade_time seventh_grade_time eighth_grade_time : ℕ)
  (sixth_to_seventh_ratio seventh_to_eighth_ratio : ℕ) :
  total_students = 210 →
  sixth_grade_time = 10 →
  seventh_grade_time = 12 →
  eighth_grade_time = 14 →
  sixth_to_seventh_ratio = 3 →
  seventh_to_eighth_ratio = 4 →
  (let eighth_grade_count := total_students / (1 + seventh_to_eighth_ratio + sixth_to_seventh_ratio * seventh_to_eighth_ratio);
   let seventh_grade_count := seventh_to_eighth_ratio * eighth_grade_count;
   let sixth_grade_count := sixth_to_seventh_ratio * seventh_grade_count;
   let total_minutes := sixth_grade_count * sixth_grade_time + 
                        seventh_grade_count * seventh_grade_time + 
                        eighth_grade_count * eighth_grade_time;
   (total_minutes : ℚ) / total_students = 420 / 39) :=
by
  sorry

end NUMINAMATH_CALUDE_average_running_time_l1333_133365


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1333_133322

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1333_133322


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l1333_133399

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, p < k → is_prime p → ¬(n % p = 0)

theorem smallest_non_prime_non_square_no_small_factors : 
  ∃! n : ℕ, n > 0 ∧ 
    ¬(is_prime n) ∧ 
    ¬(is_perfect_square n) ∧ 
    has_no_prime_factor_less_than n 60 ∧
    ∀ m : ℕ, m > 0 → 
      ¬(is_prime m) → 
      ¬(is_perfect_square m) → 
      has_no_prime_factor_less_than m 60 → 
      n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l1333_133399


namespace NUMINAMATH_CALUDE_joans_marbles_l1333_133345

/-- Given that Mary has 9 yellow marbles and the total number of yellow marbles
    between Mary and Joan is 12, prove that Joan has 3 yellow marbles. -/
theorem joans_marbles (mary_marbles : ℕ) (total_marbles : ℕ) (joan_marbles : ℕ) 
    (h1 : mary_marbles = 9)
    (h2 : total_marbles = 12)
    (h3 : mary_marbles + joan_marbles = total_marbles) :
  joan_marbles = 3 := by
  sorry

end NUMINAMATH_CALUDE_joans_marbles_l1333_133345


namespace NUMINAMATH_CALUDE_calculation_proof_l1333_133369

theorem calculation_proof : 
  Real.rpow 27 (1/3) + (Real.sqrt 2 - 1)^2 - (1/2)⁻¹ + 2 / (Real.sqrt 2 - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1333_133369


namespace NUMINAMATH_CALUDE_sin_pi_third_value_l1333_133372

theorem sin_pi_third_value (f : ℝ → ℝ) :
  (∀ α : ℝ, f (Real.sin α + Real.cos α) = (1/2) * Real.sin (2 * α)) →
  f (Real.sin (π/3)) = -1/8 := by
sorry

end NUMINAMATH_CALUDE_sin_pi_third_value_l1333_133372


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1333_133304

theorem polynomial_factorization (m n : ℝ) :
  (m + n)^2 - 10*(m + n) + 25 = (m + n - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1333_133304


namespace NUMINAMATH_CALUDE_fahrenheit_from_kelvin_l1333_133393

theorem fahrenheit_from_kelvin (K F C : ℝ) : 
  K = 300 → 
  C = (5/9) * (F - 32) → 
  C = K - 273 → 
  F = 80.6 := by sorry

end NUMINAMATH_CALUDE_fahrenheit_from_kelvin_l1333_133393


namespace NUMINAMATH_CALUDE_correct_matching_probability_l1333_133330

/-- The number of Earthly Branches and zodiac signs -/
def n : ℕ := 12

/-- The number of cards selected from each color -/
def k : ℕ := 3

/-- The probability of correctly matching the selected cards -/
def matching_probability : ℚ := 1 / (n.choose k)

/-- Theorem stating the probability of correctly matching the selected cards -/
theorem correct_matching_probability :
  matching_probability = 1 / 220 :=
by sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l1333_133330


namespace NUMINAMATH_CALUDE_smaller_root_of_quadratic_l1333_133360

theorem smaller_root_of_quadratic (x : ℚ) : 
  (x - 4/5) * (x - 4/5) + (x - 4/5) * (x - 2/3) + 1/15 = 0 →
  (x = 11/15 ∨ x = 4/5) ∧ 11/15 < 4/5 :=
sorry

end NUMINAMATH_CALUDE_smaller_root_of_quadratic_l1333_133360


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l1333_133382

def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic a)
  (h_sum : a 4 + a 9 = 24)
  (h_a6 : a 6 = 11) :
  a 7 = 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l1333_133382


namespace NUMINAMATH_CALUDE_matching_socks_probability_l1333_133336

def blue_socks : ℕ := 12
def green_socks : ℕ := 10
def red_socks : ℕ := 9

def total_socks : ℕ := blue_socks + green_socks + red_socks

def matching_pairs : ℕ := (blue_socks.choose 2) + (green_socks.choose 2) + (red_socks.choose 2)

def total_pairs : ℕ := total_socks.choose 2

theorem matching_socks_probability :
  (matching_pairs : ℚ) / total_pairs = 147 / 465 :=
sorry

end NUMINAMATH_CALUDE_matching_socks_probability_l1333_133336


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1333_133349

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1333_133349


namespace NUMINAMATH_CALUDE_specific_circle_diameter_l1333_133334

/-- A circle tangent to the y-axis and a line, passing through a specific point -/
structure TangentCircle where
  /-- The circle is tangent to the y-axis -/
  tangent_y_axis : Bool
  /-- The slope of the line the circle is tangent to -/
  line_slope : ℝ
  /-- The x-coordinate of the point the circle passes through -/
  point_x : ℝ
  /-- The y-coordinate of the point the circle passes through -/
  point_y : ℝ

/-- The diameter of a TangentCircle -/
def circle_diameter (c : TangentCircle) : Set ℝ :=
  {d : ℝ | d = 2 ∨ d = 14/3}

/-- Theorem stating the diameter of the specific TangentCircle -/
theorem specific_circle_diameter :
  let c : TangentCircle := {
    tangent_y_axis := true,
    line_slope := Real.sqrt 3 / 3,
    point_x := 2,
    point_y := Real.sqrt 3
  }
  ∀ d ∈ circle_diameter c, d = 2 ∨ d = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_specific_circle_diameter_l1333_133334


namespace NUMINAMATH_CALUDE_english_chinese_difference_l1333_133320

/-- The number of hours Ryan spends learning English daily -/
def hours_english : ℕ := 6

/-- The number of hours Ryan spends learning Chinese daily -/
def hours_chinese : ℕ := 2

/-- The difference in hours between English and Chinese learning -/
def hour_difference : ℕ := hours_english - hours_chinese

theorem english_chinese_difference : hour_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_english_chinese_difference_l1333_133320


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1333_133381

theorem smaller_number_problem (x y : ℤ) (h1 : x + y = 62) (h2 : y = x + 12) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1333_133381


namespace NUMINAMATH_CALUDE_price_reduction_l1333_133387

theorem price_reduction (x : ℝ) : 
  (100 - x) * 0.9 = 85.5 → x = 5 := by sorry

end NUMINAMATH_CALUDE_price_reduction_l1333_133387


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_values_l1333_133368

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 ≠ 1}

-- State the theorem
theorem set_inclusion_implies_a_values :
  ∀ a : ℝ, (B a ⊆ A) ↔ (a ≤ -1 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_values_l1333_133368


namespace NUMINAMATH_CALUDE_midpoint_property_l1333_133353

/-- Given two points D and E in the plane, if F is their midpoint, 
    then 3 times the x-coordinate of F minus 5 times the y-coordinate of F equals 4. -/
theorem midpoint_property (D E F : ℝ × ℝ) : 
  D = (15, 3) → 
  E = (6, 8) → 
  F.1 = (D.1 + E.1) / 2 →
  F.2 = (D.2 + E.2) / 2 →
  3 * F.1 - 5 * F.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_property_l1333_133353


namespace NUMINAMATH_CALUDE_sqrt_12_similar_to_sqrt_3_l1333_133377

/-- Two quadratic radicals are similar if they have the same radicand when simplified. -/
def similar_radicals (a b : ℝ) : Prop :=
  ∃ (k₁ k₂ : ℝ), k₁ > 0 ∧ k₂ > 0 ∧ a = k₁^2 * b

/-- √12 is of the same type as √3 -/
theorem sqrt_12_similar_to_sqrt_3 : similar_radicals 12 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_similar_to_sqrt_3_l1333_133377


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1333_133327

theorem sum_of_fractions : 
  let a := 1 + 3 + 5
  let b := 2 + 4 + 6
  (a / b) + (b / a) = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1333_133327


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1333_133329

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_difference 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 5 = 8) 
  (h2 : seq.S 3 = 6) : 
  common_difference seq = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1333_133329


namespace NUMINAMATH_CALUDE_distribution_counterexample_l1333_133378

-- Define a type for random variables
def RandomVariable := Real → Real

-- Define a type for distribution functions
def DistributionFunction := Real → Real

-- Function to get the distribution function of a random variable
def getDistribution (X : RandomVariable) : DistributionFunction := sorry

-- Function to check if two distribution functions are identical
def distributionsIdentical (F G : DistributionFunction) : Prop := sorry

-- Function to multiply two random variables
def multiply (X Y : RandomVariable) : RandomVariable := sorry

theorem distribution_counterexample :
  ∃ (ξ η ζ : RandomVariable),
    distributionsIdentical (getDistribution ξ) (getDistribution η) ∧
    ¬distributionsIdentical (getDistribution (multiply ξ ζ)) (getDistribution (multiply η ζ)) := by
  sorry

end NUMINAMATH_CALUDE_distribution_counterexample_l1333_133378


namespace NUMINAMATH_CALUDE_dandan_age_problem_l1333_133323

theorem dandan_age_problem (dandan_age : ℕ) (father_age : ℕ) (a : ℕ) :
  dandan_age = 4 →
  father_age = 28 →
  father_age + a = 3 * (dandan_age + a) →
  a = 8 :=
by sorry

end NUMINAMATH_CALUDE_dandan_age_problem_l1333_133323


namespace NUMINAMATH_CALUDE_fathers_age_three_times_xiaojuns_l1333_133331

theorem fathers_age_three_times_xiaojuns (xiaojun_age : ℕ) (father_age : ℕ) (years_passed : ℕ) :
  xiaojun_age = 5 →
  father_age = 31 →
  years_passed = 8 →
  father_age + years_passed = 3 * (xiaojun_age + years_passed) :=
by
  sorry

#check fathers_age_three_times_xiaojuns

end NUMINAMATH_CALUDE_fathers_age_three_times_xiaojuns_l1333_133331


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1333_133354

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (80 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 16 / (80 - c) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1333_133354


namespace NUMINAMATH_CALUDE_bbq_attendance_l1333_133347

def ice_per_person : ℕ := 2
def bags_per_pack : ℕ := 10
def price_per_pack : ℚ := 3
def total_spent : ℚ := 9

theorem bbq_attendance : ℕ := by
  sorry

end NUMINAMATH_CALUDE_bbq_attendance_l1333_133347


namespace NUMINAMATH_CALUDE_circle_proof_l1333_133351

/-- The equation of the given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 6*y + 5 = 0

/-- The equation of the circle we want to prove -/
def our_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

/-- Point A -/
def point_A : ℝ × ℝ := (4, -1)

/-- Point B -/
def point_B : ℝ × ℝ := (1, 2)

/-- Two circles are tangent if they intersect at exactly one point -/
def tangent (c1 c2 : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop :=
  c1 p.1 p.2 ∧ c2 p.1 p.2 ∧ ∀ x y, c1 x y ∧ c2 x y → (x, y) = p

theorem circle_proof :
  our_circle point_A.1 point_A.2 ∧
  tangent given_circle our_circle point_B :=
sorry

end NUMINAMATH_CALUDE_circle_proof_l1333_133351


namespace NUMINAMATH_CALUDE_stella_stamps_count_l1333_133376

/-- The number of stamps in Stella's album -/
def total_stamps : ℕ :=
  let total_pages : ℕ := 50
  let first_pages : ℕ := 10
  let stamps_per_row : ℕ := 30
  let rows_per_first_page : ℕ := 5
  let stamps_per_remaining_page : ℕ := 50
  
  let stamps_in_first_pages : ℕ := first_pages * rows_per_first_page * stamps_per_row
  let remaining_pages : ℕ := total_pages - first_pages
  let stamps_in_remaining_pages : ℕ := remaining_pages * stamps_per_remaining_page
  
  stamps_in_first_pages + stamps_in_remaining_pages

/-- Theorem stating that the total number of stamps in Stella's album is 3500 -/
theorem stella_stamps_count : total_stamps = 3500 := by
  sorry

end NUMINAMATH_CALUDE_stella_stamps_count_l1333_133376


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_successors_l1333_133313

theorem unique_prime_with_prime_successors :
  ∀ p : ℕ, Prime p ∧ Prime (p + 4) ∧ Prime (p + 8) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_successors_l1333_133313


namespace NUMINAMATH_CALUDE_smallest_valid_number_divisible_by_51_l1333_133388

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 = (n / 100) % 10) ∧
  ((n / 10) % 10 = n % 10)

theorem smallest_valid_number_divisible_by_51 :
  ∃ (A : ℕ), is_valid_number A ∧ A % 51 = 0 ∧
  ∀ (B : ℕ), is_valid_number B ∧ B % 51 = 0 → A ≤ B ∧ A = 1122 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_divisible_by_51_l1333_133388


namespace NUMINAMATH_CALUDE_linear_function_properties_l1333_133342

/-- A linear function passing through two given points -/
structure LinearFunction where
  b : ℝ
  k : ℝ
  point1 : b * (-2) + k = -3
  point2 : b * 1 + k = 3

/-- Theorem stating the properties of the linear function -/
theorem linear_function_properties (f : LinearFunction) :
  f.k = 1 ∧ f.b = 2 ∧ f.b * (-2) + f.k ≠ 3 := by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1333_133342


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l1333_133367

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (36 * π / 180) * Real.cos (24 * π / 180) +
  Real.cos (36 * π / 180) * Real.sin (156 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l1333_133367


namespace NUMINAMATH_CALUDE_divisor_log_sum_l1333_133311

theorem divisor_log_sum (n : ℕ) : (n * (n + 1)^2) / 2 = 1080 ↔ n = 12 := by sorry

end NUMINAMATH_CALUDE_divisor_log_sum_l1333_133311


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1333_133324

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the problem statement
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 8) / 2 = 10 →
  a 1 + a 10 = 20 := by
    sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1333_133324


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l1333_133389

theorem line_hyperbola_intersection (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ 
    x₁^2 - (k*x₁ + 1)^2 = 1 ∧ 
    x₂^2 - (k*x₂ + 1)^2 = 1) → 
  k > 1 ∧ k < Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l1333_133389


namespace NUMINAMATH_CALUDE_optimal_garden_is_best_l1333_133328

/-- Represents a rectangular garden with one side against a wall --/
structure Garden where
  width : ℝ  -- Width of the garden (perpendicular to the wall)
  length : ℝ  -- Length of the garden (parallel to the wall)

/-- The wall length --/
def wall_length : ℝ := 600

/-- The fence cost per foot --/
def fence_cost_per_foot : ℝ := 6

/-- The total budget for fencing --/
def fence_budget : ℝ := 1800

/-- The minimum required area --/
def min_area : ℝ := 6000

/-- Calculate the area of the garden --/
def area (g : Garden) : ℝ := g.width * g.length

/-- Calculate the perimeter of the garden --/
def perimeter (g : Garden) : ℝ := 2 * g.width + g.length + wall_length

/-- Check if the garden satisfies the budget constraint --/
def satisfies_budget (g : Garden) : Prop :=
  (2 * g.width + g.length) * fence_cost_per_foot ≤ fence_budget

/-- Check if the garden satisfies the area constraint --/
def satisfies_area (g : Garden) : Prop :=
  area g ≥ min_area

/-- The optimal garden dimensions --/
def optimal_garden : Garden :=
  { width := 75, length := 150 }

/-- Theorem stating that the optimal garden maximizes perimeter while satisfying constraints --/
theorem optimal_garden_is_best :
  satisfies_budget optimal_garden ∧
  satisfies_area optimal_garden ∧
  ∀ g : Garden, satisfies_budget g → satisfies_area g →
    perimeter g ≤ perimeter optimal_garden :=
by sorry

end NUMINAMATH_CALUDE_optimal_garden_is_best_l1333_133328


namespace NUMINAMATH_CALUDE_max_discount_l1333_133309

/-- Given a product with a marked price and markup percentage, calculate the maximum discount --/
theorem max_discount (marked_price : ℝ) (markup_percent : ℝ) (min_markup_percent : ℝ) : 
  marked_price = 360 ∧ 
  markup_percent = 0.8 ∧ 
  min_markup_percent = 0.2 →
  marked_price - (marked_price / (1 + markup_percent) * (1 + min_markup_percent)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_max_discount_l1333_133309


namespace NUMINAMATH_CALUDE_problem_grid_paths_l1333_133361

/-- Represents a grid with forbidden segments -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (forbidden_segments : List (ℕ × ℕ × ℕ × ℕ))

/-- Calculates the number of paths in a grid with forbidden segments -/
def count_paths (g : Grid) : ℕ :=
  sorry

/-- The specific grid described in the problem -/
def problem_grid : Grid :=
  { rows := 4
  , cols := 7
  , forbidden_segments := [(1, 2, 3, 4), (2, 3, 5, 6)] }

/-- Theorem stating that the number of paths in the problem grid is 64 -/
theorem problem_grid_paths :
  count_paths problem_grid = 64 :=
sorry

end NUMINAMATH_CALUDE_problem_grid_paths_l1333_133361


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l1333_133379

theorem smallest_five_digit_multiple_of_18 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 18 ∣ n → 10008 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l1333_133379


namespace NUMINAMATH_CALUDE_fourth_side_length_l1333_133391

/-- A quadrilateral inscribed in a circle with radius 300, where three sides have lengths 300, 300, and 150√2 -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the first side -/
  side1 : ℝ
  /-- The length of the second side -/
  side2 : ℝ
  /-- The length of the third side -/
  side3 : ℝ
  /-- The length of the fourth side -/
  side4 : ℝ
  /-- Condition that the quadrilateral is inscribed in a circle with radius 300 -/
  radius_eq : radius = 300
  /-- Condition that two sides have length 300 -/
  side1_eq : side1 = 300
  side2_eq : side2 = 300
  /-- Condition that one side has length 150√2 -/
  side3_eq : side3 = 150 * Real.sqrt 2

/-- Theorem stating that the fourth side of the inscribed quadrilateral has length 450 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.side4 = 450 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l1333_133391


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1333_133380

/-- Given a circle and a line with specific properties, prove the center and radius of the circle -/
theorem circle_center_and_radius 
  (x y : ℝ) 
  (m : ℝ) 
  (circle_eq : x^2 + y^2 + x - 6*y + m = 0) 
  (line_eq : x + 2*y - 3 = 0) 
  (P Q : ℝ × ℝ) 
  (intersect : (P.1^2 + P.2^2 + P.1 - 6*P.2 + m = 0 ∧ P.1 + 2*P.2 - 3 = 0) ∧ 
               (Q.1^2 + Q.2^2 + Q.1 - 6*Q.2 + m = 0 ∧ Q.1 + 2*Q.2 - 3 = 0)) 
  (perpendicular : P.1 * Q.1 + P.2 * Q.2 = 0) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (-1/2, 3) ∧ radius = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1333_133380


namespace NUMINAMATH_CALUDE_billys_book_pages_l1333_133312

/-- Proves that given Billy's reading habits and time allocation, each book he reads contains 80 pages. -/
theorem billys_book_pages : 
  -- Billy's free time per day
  (free_time_per_day : ℕ) →
  -- Number of weekend days
  (weekend_days : ℕ) →
  -- Percentage of time spent on video games
  (video_game_percentage : ℚ) →
  -- Pages Billy can read per hour
  (pages_per_hour : ℕ) →
  -- Number of books Billy reads
  (number_of_books : ℕ) →
  -- Conditions
  (free_time_per_day = 8) →
  (weekend_days = 2) →
  (video_game_percentage = 3/4) →
  (pages_per_hour = 60) →
  (number_of_books = 3) →
  -- Conclusion: each book contains 80 pages
  (∃ (pages_per_book : ℕ), pages_per_book = 80 ∧ 
    pages_per_book * number_of_books = 
      (1 - video_game_percentage) * (free_time_per_day * weekend_days : ℚ) * pages_per_hour) :=
by
  sorry


end NUMINAMATH_CALUDE_billys_book_pages_l1333_133312


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l1333_133305

-- Define the set of numbers
def S : Finset ℕ := Finset.range 7

-- Define a type for a selection of 5 numbers
def Selection := {s : Finset ℕ // s.card = 5 ∧ s ⊆ S}

-- Define the median of a selection
def median (sel : Selection) : ℚ :=
  sorry

-- Define the average of a selection
def average (sel : Selection) : ℚ :=
  sorry

-- Define event A: median is 4
def eventA (sel : Selection) : Prop :=
  median sel = 4

-- Define event B: average is 4
def eventB (sel : Selection) : Prop :=
  average sel = 4

-- Define the probability measure
noncomputable def P : Set Selection → ℝ :=
  sorry

-- State the theorem
theorem conditional_probability_B_given_A :
  P {sel : Selection | eventB sel ∧ eventA sel} / P {sel : Selection | eventA sel} = 1/3 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l1333_133305


namespace NUMINAMATH_CALUDE_sequence_general_term_l1333_133395

theorem sequence_general_term (a : ℕ+ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ+, a (n + 1) = a n + 2 * n) →
  (∀ n : ℕ+, a n = n^2 - n + 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1333_133395


namespace NUMINAMATH_CALUDE_nes_sale_price_l1333_133370

/-- The sale price of an NES given trade-in and cash transactions -/
theorem nes_sale_price 
  (snes_value : ℝ) 
  (trade_in_percentage : ℝ) 
  (cash_given : ℝ) 
  (change_received : ℝ) 
  (game_value : ℝ) 
  (h1 : snes_value = 150)
  (h2 : trade_in_percentage = 0.8)
  (h3 : cash_given = 80)
  (h4 : change_received = 10)
  (h5 : game_value = 30) :
  snes_value * trade_in_percentage + cash_given - change_received - game_value = 160 :=
by sorry

end NUMINAMATH_CALUDE_nes_sale_price_l1333_133370


namespace NUMINAMATH_CALUDE_solve_pq_system_l1333_133340

theorem solve_pq_system (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_pq_system_l1333_133340


namespace NUMINAMATH_CALUDE_line_division_theorem_l1333_133335

/-- A line on a plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines divide the plane into six parts -/
def divides_into_six_parts (l1 l2 l3 : Line) : Prop :=
  sorry

/-- The set of k values that satisfy the condition -/
def k_values : Set ℝ := {0, -1, -2}

/-- Theorem stating the relationship between the lines and k values -/
theorem line_division_theorem (k : ℝ) :
  let l1 : Line := ⟨1, -2, 1⟩
  let l2 : Line := ⟨1, 0, -1⟩
  let l3 : Line := ⟨1, k, 0⟩
  divides_into_six_parts l1 l2 l3 → k ∈ k_values := by
  sorry

end NUMINAMATH_CALUDE_line_division_theorem_l1333_133335


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1333_133321

theorem min_value_sum_of_reciprocals (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 3) :
  (1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x)) ≥ 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1333_133321


namespace NUMINAMATH_CALUDE_all_propositions_incorrect_l1333_133339

/-- Represents a proposition with potential flaws in statistical reasoning --/
structure Proposition where
  hasTemporalityIgnorance : Bool
  hasSpeciesCharacteristicsIgnorance : Bool
  hasCausalityMisinterpretation : Bool
  hasIncorrectUsageRange : Bool

/-- Determines if a proposition is incorrect based on its flaws --/
def isIncorrect (p : Proposition) : Bool :=
  p.hasTemporalityIgnorance ∨ 
  p.hasSpeciesCharacteristicsIgnorance ∨ 
  p.hasCausalityMisinterpretation ∨ 
  p.hasIncorrectUsageRange

/-- Counts the number of incorrect propositions in a list --/
def countIncorrectPropositions (props : List Proposition) : Nat :=
  props.filter isIncorrect |>.length

/-- The main theorem stating that all given propositions are incorrect --/
theorem all_propositions_incorrect (props : List Proposition) 
  (h1 : props.length = 4)
  (h2 : ∀ p ∈ props, isIncorrect p = true) : 
  countIncorrectPropositions props = 4 := by
  sorry

#check all_propositions_incorrect

end NUMINAMATH_CALUDE_all_propositions_incorrect_l1333_133339


namespace NUMINAMATH_CALUDE_chessboard_rectangle_same_color_l1333_133350

-- Define the chessboard as a 4x7 matrix of booleans (true for black, false for white)
def Chessboard := Matrix (Fin 4) (Fin 7) Bool

-- Define a rectangle on the chessboard
def Rectangle (board : Chessboard) (r1 r2 : Fin 4) (c1 c2 : Fin 7) : Prop :=
  r1 < r2 ∧ c1 < c2

-- Define the property of a rectangle having all corners of the same color
def SameColorCorners (board : Chessboard) (r1 r2 : Fin 4) (c1 c2 : Fin 7) : Prop :=
  Rectangle board r1 r2 c1 c2 ∧
  board r1 c1 = board r1 c2 ∧
  board r1 c1 = board r2 c1 ∧
  board r1 c1 = board r2 c2

-- The main theorem
theorem chessboard_rectangle_same_color (board : Chessboard) :
  ∃ (r1 r2 : Fin 4) (c1 c2 : Fin 7), SameColorCorners board r1 r2 c1 c2 := by
  sorry


end NUMINAMATH_CALUDE_chessboard_rectangle_same_color_l1333_133350


namespace NUMINAMATH_CALUDE_two_propositions_are_true_l1333_133373

/-- Represents the truth value of a proposition -/
inductive PropositionTruth
  | True
  | False

/-- The four propositions in the problem -/
def proposition1 : PropositionTruth := PropositionTruth.False
def proposition2 : PropositionTruth := PropositionTruth.True
def proposition3 : PropositionTruth := PropositionTruth.False
def proposition4 : PropositionTruth := PropositionTruth.True

/-- Counts the number of true propositions -/
def countTruePropositions (p1 p2 p3 p4 : PropositionTruth) : Nat :=
  match p1, p2, p3, p4 with
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.True, PropositionTruth.True => 4
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.True, PropositionTruth.False => 3
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.False, PropositionTruth.True => 3
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.False, PropositionTruth.False => 2
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.True, PropositionTruth.True => 3
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.True, PropositionTruth.False => 2
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.False, PropositionTruth.True => 2
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.False, PropositionTruth.False => 1
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.True, PropositionTruth.True => 3
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.True, PropositionTruth.False => 2
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.False, PropositionTruth.True => 2
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.False, PropositionTruth.False => 1
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.True, PropositionTruth.True => 2
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.True, PropositionTruth.False => 1
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.False, PropositionTruth.True => 1
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.False, PropositionTruth.False => 0

/-- Theorem stating that exactly 2 out of 4 given propositions are true -/
theorem two_propositions_are_true :
  countTruePropositions proposition1 proposition2 proposition3 proposition4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_propositions_are_true_l1333_133373


namespace NUMINAMATH_CALUDE_smallest_share_for_200_people_l1333_133341

/-- Represents a family with land inheritance rules -/
structure Family :=
  (size : ℕ)
  (has_founder : size > 0)

/-- The smallest possible share of the original plot for any family member -/
def smallest_share (f : Family) : ℚ :=
  1 / (4 * 3^65)

/-- Theorem stating the smallest possible share for a family of 200 people -/
theorem smallest_share_for_200_people (f : Family) (h : f.size = 200) :
  smallest_share f = 1 / (4 * 3^65) := by
  sorry

end NUMINAMATH_CALUDE_smallest_share_for_200_people_l1333_133341


namespace NUMINAMATH_CALUDE_science_price_relation_spending_condition_min_literature_books_proof_l1333_133375

-- Define the prices of books
def literature_price : ℚ := 5
def science_price : ℚ := 15 / 2

-- Define the condition that science book price is half higher than literature book price
theorem science_price_relation : science_price = literature_price * (3/2) := by sorry

-- Define the spending condition
theorem spending_condition (lit_count science_count : ℕ) : 
  lit_count * literature_price + science_count * science_price = 15 ∧ lit_count = science_count + 1 := by sorry

-- Define the budget condition
def total_books : ℕ := 10
def total_budget : ℚ := 60

-- Define the function to calculate the minimum number of literature books
def min_literature_books : ℕ := 6

-- Theorem to prove the minimum number of literature books
theorem min_literature_books_proof :
  ∀ m : ℕ, m * literature_price + (total_books - m) * science_price ≤ total_budget → m ≥ min_literature_books := by sorry

end NUMINAMATH_CALUDE_science_price_relation_spending_condition_min_literature_books_proof_l1333_133375


namespace NUMINAMATH_CALUDE_car_speed_proof_l1333_133348

/-- Proves that a car covering 400 meters in 12 seconds has a speed of 120 kilometers per hour -/
theorem car_speed_proof (distance : ℝ) (time : ℝ) (speed_mps : ℝ) (speed_kmph : ℝ) : 
  distance = 400 ∧ time = 12 ∧ speed_mps = distance / time ∧ speed_kmph = speed_mps * 3.6 →
  speed_kmph = 120 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_proof_l1333_133348


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1333_133303

open Set

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1333_133303


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1333_133366

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r ≠ 0 ∧ a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 1 * a 9 = 16) :
  a 2 * a 5 * a 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1333_133366


namespace NUMINAMATH_CALUDE_babysitting_hours_l1333_133364

/-- Represents the babysitting scenario -/
structure BabysittingScenario where
  hourly_rate : ℚ
  makeup_fraction : ℚ
  skincare_fraction : ℚ
  remaining_amount : ℚ

/-- Calculates the number of hours babysitted per day -/
def hours_per_day (scenario : BabysittingScenario) : ℚ :=
  ((1 - scenario.makeup_fraction - scenario.skincare_fraction) * scenario.remaining_amount) /
  (7 * scenario.hourly_rate)

/-- Theorem stating that given the specific scenario, the person babysits for 3 hours each day -/
theorem babysitting_hours (scenario : BabysittingScenario) 
  (h1 : scenario.hourly_rate = 10)
  (h2 : scenario.makeup_fraction = 3/10)
  (h3 : scenario.skincare_fraction = 2/5)
  (h4 : scenario.remaining_amount = 63) :
  hours_per_day scenario = 3 := by
  sorry

#eval hours_per_day { hourly_rate := 10, makeup_fraction := 3/10, skincare_fraction := 2/5, remaining_amount := 63 }

end NUMINAMATH_CALUDE_babysitting_hours_l1333_133364


namespace NUMINAMATH_CALUDE_max_value_expression_l1333_133355

theorem max_value_expression (a b c d : ℝ) 
  (ha : -8.5 ≤ a ∧ a ≤ 8.5)
  (hb : -8.5 ≤ b ∧ b ≤ 8.5)
  (hc : -8.5 ≤ c ∧ c ≤ 8.5)
  (hd : -8.5 ≤ d ∧ d ≤ 8.5) :
  ∃ (m : ℝ), m = 306 ∧ 
  ∀ (a' b' c' d' : ℝ), 
    -8.5 ≤ a' ∧ a' ≤ 8.5 → 
    -8.5 ≤ b' ∧ b' ≤ 8.5 → 
    -8.5 ≤ c' ∧ c' ≤ 8.5 → 
    -8.5 ≤ d' ∧ d' ≤ 8.5 → 
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1333_133355


namespace NUMINAMATH_CALUDE_triangle_properties_l1333_133385

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, angle C and the sum of sines of A and B
    have specific values. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  c^2 = a^2 + b^2 + a*b →
  c = 4 * Real.sqrt 7 →
  a + b + c = 12 + 4 * Real.sqrt 7 →
  C = 2 * Real.pi / 3 ∧
  Real.sin A + Real.sin B = 3 * Real.sqrt 21 / 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1333_133385


namespace NUMINAMATH_CALUDE_m_positive_l1333_133384

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 2 / (2^x + 1) + 1

theorem m_positive (m : ℝ) (h : f (m - 1) + f (1 - 2*m) > 4) : m > 0 := by
  sorry

end NUMINAMATH_CALUDE_m_positive_l1333_133384


namespace NUMINAMATH_CALUDE_derivative_of_f_l1333_133396

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = (1 - log x) / (x^2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l1333_133396


namespace NUMINAMATH_CALUDE_bertrand_odd_conjecture_counterexample_l1333_133394

-- Define what we mean by a "large" number
def isLarge (n : ℕ) : Prop := n ≥ 100

-- Define an odd number
def isOdd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

-- Define a prime number
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

-- Bertrand's Odd Conjecture
def bertrandOddConjecture : Prop := 
  ∀ n, isLarge n → isOdd n → 
    ∃ p q r, isPrime p ∧ isPrime q ∧ isPrime r ∧ 
             isOdd p ∧ isOdd q ∧ isOdd r ∧
             n = p + q + r

-- Theorem: There exists a counterexample to Bertrand's Odd Conjecture
theorem bertrand_odd_conjecture_counterexample :
  ∃ n, isLarge n ∧ isOdd n ∧ 
    ¬(∃ p q r, isPrime p ∧ isPrime q ∧ isPrime r ∧ 
               isOdd p ∧ isOdd q ∧ isOdd r ∧
               n = p + q + r) :=
by sorry

end NUMINAMATH_CALUDE_bertrand_odd_conjecture_counterexample_l1333_133394


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1333_133300

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1333_133300


namespace NUMINAMATH_CALUDE_bulb_toggling_theorem_l1333_133315

/-- Represents the state of a light bulb (on or off) -/
inductive BulbState
| Off
| On

/-- Toggles the state of a light bulb -/
def toggleBulb : BulbState → BulbState
| BulbState.Off => BulbState.On
| BulbState.On => BulbState.Off

/-- Returns the number of positive divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- Returns true if a natural number is a perfect square, false otherwise -/
def isPerfectSquare (n : ℕ) : Bool := sorry

/-- Simulates the process of students toggling light bulbs -/
def toggleBulbs (n : ℕ) : List BulbState := sorry

/-- Counts the number of bulbs that are on after the toggling process -/
def countOnBulbs (bulbs : List BulbState) : ℕ := sorry

/-- Counts the number of perfect squares less than or equal to a given number -/
def countPerfectSquares (n : ℕ) : ℕ := sorry

theorem bulb_toggling_theorem :
  countOnBulbs (toggleBulbs 100) = countPerfectSquares 100 := by sorry

end NUMINAMATH_CALUDE_bulb_toggling_theorem_l1333_133315


namespace NUMINAMATH_CALUDE_divisibility_of_power_minus_odd_l1333_133346

theorem divisibility_of_power_minus_odd (k m : ℕ) (hk : k > 0) (hm : Odd m) :
  ∃ n : ℕ, n > 0 ∧ (2^k : ℕ) ∣ (n^n - m) :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_power_minus_odd_l1333_133346


namespace NUMINAMATH_CALUDE_sum_of_integers_l1333_133343

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 145)
  (h2 : x * y = 40) : 
  x + y = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1333_133343


namespace NUMINAMATH_CALUDE_max_single_painted_face_theorem_l1333_133318

/-- Represents a large cube composed of smaller cubes -/
structure LargeCube where
  size : Nat
  painted_faces : Nat

/-- Calculates the maximum number of smaller cubes with exactly one face painted -/
def max_single_painted_face (cube : LargeCube) : Nat :=
  if cube.size = 4 ∧ cube.painted_faces = 3 then 32 else 0

/-- Theorem stating the maximum number of smaller cubes with exactly one face painted -/
theorem max_single_painted_face_theorem (cube : LargeCube) :
  cube.size = 4 ∧ cube.painted_faces = 3 →
  max_single_painted_face cube = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_single_painted_face_theorem_l1333_133318


namespace NUMINAMATH_CALUDE_total_paths_is_4312_l1333_133301

/-- Represents the number of paths between different points in the lattice --/
structure LatticePathCounts where
  a_to_red1 : Nat
  a_to_red2 : Nat
  red1_to_blue : Nat
  red2_to_blue : Nat
  blue12_to_green : Nat
  blue34_to_green : Nat
  green_to_b : Nat
  green_to_c : Nat

/-- Calculates the total number of distinct paths to reach points B and C --/
def totalPaths (counts : LatticePathCounts) : Nat :=
  let paths_to_blue := counts.a_to_red1 * counts.red1_to_blue * 2 + 
                       counts.a_to_red2 * counts.red2_to_blue * 2
  let paths_to_green := paths_to_blue * (counts.blue12_to_green * 2 + counts.blue34_to_green * 2)
  paths_to_green * (counts.green_to_b + counts.green_to_c)

/-- The theorem stating that the total number of distinct paths is 4312 --/
theorem total_paths_is_4312 (counts : LatticePathCounts)
  (h1 : counts.a_to_red1 = 1)
  (h2 : counts.a_to_red2 = 2)
  (h3 : counts.red1_to_blue = 3)
  (h4 : counts.red2_to_blue = 4)
  (h5 : counts.blue12_to_green = 5)
  (h6 : counts.blue34_to_green = 6)
  (h7 : counts.green_to_b = 3)
  (h8 : counts.green_to_c = 4) :
  totalPaths counts = 4312 := by
  sorry


end NUMINAMATH_CALUDE_total_paths_is_4312_l1333_133301


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l1333_133310

theorem rectangle_area_problem (square_area : Real) (rectangle_breadth : Real) :
  square_area = 784 →
  rectangle_breadth = 5 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := circle_radius / 4
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area = 35 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l1333_133310


namespace NUMINAMATH_CALUDE_sticks_left_in_yard_l1333_133359

def sticks_picked_up : ℕ := 14
def difference : ℕ := 10

theorem sticks_left_in_yard : sticks_picked_up - difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_sticks_left_in_yard_l1333_133359


namespace NUMINAMATH_CALUDE_tangent_parallel_to_BC_l1333_133338

/-- Two circles in a plane -/
structure TwoCircles where
  circle1 : Set (ℝ × ℝ)
  circle2 : Set (ℝ × ℝ)

/-- Points of intersection and other significant points -/
structure CirclePoints (tc : TwoCircles) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  on_circle1_P : P ∈ tc.circle1
  on_circle2_P : P ∈ tc.circle2
  on_circle1_Q : Q ∈ tc.circle1
  on_circle2_Q : Q ∈ tc.circle2
  on_circle1_A : A ∈ tc.circle1
  on_circle2_B : B ∈ tc.circle2
  on_circle2_C : C ∈ tc.circle2

/-- Line represented by two points -/
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Tangent line to a circle at a point -/
def TangentLine (circle : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Two lines are parallel -/
def Parallel (l1 l2 : Set (ℝ × ℝ)) : Prop := sorry

/-- Main theorem -/
theorem tangent_parallel_to_BC (tc : TwoCircles) (cp : CirclePoints tc) : 
  Parallel (TangentLine tc.circle1 cp.A) (Line cp.B cp.C) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_BC_l1333_133338


namespace NUMINAMATH_CALUDE_right_triangle_seven_units_contains_28_triangles_l1333_133371

/-- Represents a right-angled triangle on a grid -/
structure GridTriangle where
  leg_length : ℕ
  is_right_angled : Bool

/-- Calculates the maximum number of triangles that can be formed within a GridTriangle -/
def max_triangles (t : GridTriangle) : ℕ :=
  if t.is_right_angled && t.leg_length > 0 then
    (t.leg_length + 1).choose 2
  else
    0

/-- Theorem stating that a right-angled triangle with legs of 7 units on a grid contains 28 triangles -/
theorem right_triangle_seven_units_contains_28_triangles :
  let t : GridTriangle := { leg_length := 7, is_right_angled := true }
  max_triangles t = 28 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_seven_units_contains_28_triangles_l1333_133371


namespace NUMINAMATH_CALUDE_exact_blue_marbles_probability_l1333_133352

def total_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 4
def num_draws : ℕ := 7
def target_blue : ℕ := 4

theorem exact_blue_marbles_probability :
  (Nat.choose num_draws target_blue : ℚ) * (blue_marbles ^ target_blue * red_marbles ^ (num_draws - target_blue)) / (total_marbles ^ num_draws) = 35 * (16 : ℚ) / 2187 := by
  sorry

end NUMINAMATH_CALUDE_exact_blue_marbles_probability_l1333_133352


namespace NUMINAMATH_CALUDE_treasure_value_proof_l1333_133362

def base7ToBase10 (n : Nat) : Nat :=
  let digits := n.digits 7
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (7 ^ i)) 0

theorem treasure_value_proof :
  let diamonds := 5643
  let silver := 1652
  let spices := 236
  (base7ToBase10 diamonds) + (base7ToBase10 silver) + (base7ToBase10 spices) = 2839 := by
  sorry

end NUMINAMATH_CALUDE_treasure_value_proof_l1333_133362


namespace NUMINAMATH_CALUDE_freezer_temp_calculation_l1333_133326

-- Define the temperature of the refrigerator compartment
def refrigerator_temp : ℝ := 4

-- Define the temperature difference between compartments
def temp_difference : ℝ := 22

-- Theorem to prove
theorem freezer_temp_calculation :
  refrigerator_temp - temp_difference = -18 := by
  sorry

end NUMINAMATH_CALUDE_freezer_temp_calculation_l1333_133326


namespace NUMINAMATH_CALUDE_sphere_packing_radius_l1333_133398

/-- A structure representing a sphere packing in a cube -/
structure SpherePacking where
  cube_side_length : ℝ
  num_spheres : ℕ
  sphere_radius : ℝ
  is_valid : Prop

/-- The theorem stating the radius of spheres in the given packing configuration -/
theorem sphere_packing_radius (packing : SpherePacking) : 
  packing.cube_side_length = 2 ∧ 
  packing.num_spheres = 10 ∧ 
  packing.is_valid →
  packing.sphere_radius = 0.5 :=
sorry

end NUMINAMATH_CALUDE_sphere_packing_radius_l1333_133398


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_l1333_133308

/-- The function f(x) = -x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 1

/-- The monotonic decreasing interval of f(x) = -x^2 + 2x + 1 is [1, +∞) -/
theorem monotonic_decreasing_interval_f :
  {x : ℝ | ∀ y, x ≤ y → f x ≥ f y} = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_f_l1333_133308


namespace NUMINAMATH_CALUDE_equilibrium_concentration_Ca_OH_2_l1333_133317

-- Define the reaction components
inductive Species
| CaO
| H2O
| Ca_OH_2

-- Define the reaction
def reaction : List Species := [Species.CaO, Species.H2O, Species.Ca_OH_2]

-- Define the equilibrium constant
def Kp : ℝ := 0.02

-- Define the equilibrium concentration function
noncomputable def equilibrium_concentration (s : Species) : ℝ :=
  match s with
  | Species.CaO => 0     -- Not applicable (solid)
  | Species.H2O => 0     -- Not applicable (liquid)
  | Species.Ca_OH_2 => Kp -- Equilibrium concentration equals Kp

-- Theorem statement
theorem equilibrium_concentration_Ca_OH_2 :
  equilibrium_concentration Species.Ca_OH_2 = Kp := by sorry

end NUMINAMATH_CALUDE_equilibrium_concentration_Ca_OH_2_l1333_133317


namespace NUMINAMATH_CALUDE_number_classification_l1333_133390

-- Define a number type that can represent both decimal and natural numbers
inductive Number
  | Decimal (integerPart : Int) (fractionalPart : Nat)
  | Natural (value : Nat)

-- Define a function to check if a number is decimal
def isDecimal (n : Number) : Prop :=
  match n with
  | Number.Decimal _ _ => True
  | Number.Natural _ => False

-- Define a function to check if a number is natural
def isNatural (n : Number) : Prop :=
  match n with
  | Number.Decimal _ _ => False
  | Number.Natural _ => True

-- Theorem statement
theorem number_classification (n : Number) :
  (isDecimal n ∧ ¬isNatural n) ∨ (¬isDecimal n ∧ isNatural n) :=
by sorry

end NUMINAMATH_CALUDE_number_classification_l1333_133390


namespace NUMINAMATH_CALUDE_speed_conversion_correct_l1333_133307

/-- Converts meters per second to kilometers per hour -/
def mps_to_kmh (speed_mps : ℚ) : ℚ :=
  speed_mps * 3.6

theorem speed_conversion_correct : 
  mps_to_kmh (13/48) = 39/40 :=
by sorry

end NUMINAMATH_CALUDE_speed_conversion_correct_l1333_133307


namespace NUMINAMATH_CALUDE_max_intersections_convex_ngon_l1333_133386

/-- The maximum number of intersection points of diagonals in a convex n-gon -/
def max_intersections (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) / 24

/-- Theorem: In a convex n-gon with all diagonals drawn, the maximum number of 
    intersection points of the diagonals is equal to C(n,4) = n(n-1)(n-2)(n-3)/24 -/
theorem max_intersections_convex_ngon (n : ℕ) (h : n ≥ 4) :
  max_intersections n = Nat.choose n 4 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_convex_ngon_l1333_133386


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1333_133344

/-- An isosceles triangle with sides of length 2 and 5 has a perimeter of 12 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 5 → b = 5 → c = 2 → a + b + c = 12 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1333_133344


namespace NUMINAMATH_CALUDE_common_number_in_list_l1333_133302

theorem common_number_in_list (list : List ℝ) : 
  list.length = 7 →
  (list.take 4).sum / 4 = 7 →
  (list.drop 3).sum / 4 = 10 →
  list.sum / 7 = 8 →
  ∃ x ∈ list.take 4 ∩ list.drop 3, x = 12 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_list_l1333_133302


namespace NUMINAMATH_CALUDE_problem_solution_l1333_133363

theorem problem_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1333_133363


namespace NUMINAMATH_CALUDE_configurations_formula_l1333_133397

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def num_configurations (n : ℕ) : ℕ :=
  factorial (n * (n + 1) / 2) /
  (List.range n).foldl (λ acc i => acc * factorial (n - i)) 1

theorem configurations_formula (n : ℕ) :
  num_configurations n = factorial (n * (n + 1) / 2) /
    (List.range n).foldl (λ acc i => acc * factorial (n - i)) 1 :=
by sorry

end NUMINAMATH_CALUDE_configurations_formula_l1333_133397
