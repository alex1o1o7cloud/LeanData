import Mathlib

namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l3938_393809

theorem logarithmic_equation_solution (x : ℝ) (h1 : x > 1) :
  (Real.log x - 1) / Real.log 5 + 
  (Real.log (x^2 - 1)) / (Real.log 5 / 2) + 
  (Real.log (x - 1)) / (Real.log (1/5)) = 3 →
  x = Real.sqrt (5 * Real.sqrt 5 + 1) := by
sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l3938_393809


namespace NUMINAMATH_CALUDE_sphere_radius_is_sqrt_six_over_four_l3938_393845

/-- A sphere circumscribing a right circular cone -/
structure CircumscribedCone where
  /-- The radius of the circumscribing sphere -/
  sphere_radius : ℝ
  /-- The diameter of the base of the cone -/
  base_diameter : ℝ
  /-- Assertion that the base diameter is 1 -/
  base_diameter_is_one : base_diameter = 1
  /-- Assertion that the apex of the cone is on the sphere -/
  apex_on_sphere : True
  /-- Assertion about the perpendicularity condition -/
  perpendicular_condition : True

/-- Theorem stating that the radius of the circumscribing sphere is √6/4 -/
theorem sphere_radius_is_sqrt_six_over_four (cone : CircumscribedCone) :
  cone.sphere_radius = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_sqrt_six_over_four_l3938_393845


namespace NUMINAMATH_CALUDE_correlation_coefficient_is_one_l3938_393801

/-- A structure representing a set of sample points -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  n_ge_2 : n ≥ 2
  not_all_x_equal : ∃ i j, i ≠ j ∧ x i ≠ x j

/-- The sample correlation coefficient -/
def sampleCorrelationCoefficient (data : SampleData) : ℝ := sorry

/-- All points lie on the line y = 2x + 1 -/
def allPointsOnLine (data : SampleData) : Prop :=
  ∀ i, data.y i = 2 * data.x i + 1

/-- Theorem stating that if all points lie on y = 2x + 1, then the correlation coefficient is 1 -/
theorem correlation_coefficient_is_one (data : SampleData) 
  (h : allPointsOnLine data) : sampleCorrelationCoefficient data = 1 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_is_one_l3938_393801


namespace NUMINAMATH_CALUDE_smallest_number_l3938_393827

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

theorem smallest_number : 
  let base_9 := to_decimal [5, 8] 9
  let base_4 := to_decimal [0, 0, 0, 1] 4
  let base_2 := to_decimal [1, 1, 1, 1, 1, 1] 2
  base_2 < base_4 ∧ base_2 < base_9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3938_393827


namespace NUMINAMATH_CALUDE_area_triangle_AOB_l3938_393814

/-- Given two points A and B in polar coordinates, prove that the area of triangle AOB is 6 -/
theorem area_triangle_AOB (A B : ℝ × ℝ) : 
  A.1 = 3 ∧ A.2 = π/3 ∧ B.1 = 4 ∧ B.2 = 5*π/6 → 
  (1/2) * A.1 * B.1 * Real.sin (B.2 - A.2) = 6 := by
sorry

end NUMINAMATH_CALUDE_area_triangle_AOB_l3938_393814


namespace NUMINAMATH_CALUDE_piano_practice_minutes_l3938_393807

theorem piano_practice_minutes (practice_time_6days : ℕ) (practice_time_2days : ℕ) 
  (total_days : ℕ) (average_minutes : ℕ) :
  practice_time_6days = 100 →
  practice_time_2days = 80 →
  total_days = 9 →
  average_minutes = 100 →
  (6 * practice_time_6days + 2 * practice_time_2days + 
   (average_minutes * total_days - (6 * practice_time_6days + 2 * practice_time_2days))) / total_days = average_minutes :=
by
  sorry

end NUMINAMATH_CALUDE_piano_practice_minutes_l3938_393807


namespace NUMINAMATH_CALUDE_susie_pizza_price_l3938_393860

/-- The price of a whole pizza given the conditions of Susie's pizza sales -/
theorem susie_pizza_price (price_per_slice : ℚ) (slices_sold : ℕ) (whole_pizzas_sold : ℕ) (total_revenue : ℚ) :
  price_per_slice = 3 →
  slices_sold = 24 →
  whole_pizzas_sold = 3 →
  total_revenue = 117 →
  ∃ (whole_pizza_price : ℚ), whole_pizza_price = 15 ∧
    price_per_slice * slices_sold + whole_pizza_price * whole_pizzas_sold = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_susie_pizza_price_l3938_393860


namespace NUMINAMATH_CALUDE_liza_reading_speed_l3938_393813

/-- Given that Suzie reads 15 pages in an hour and Liza reads 15 more pages than Suzie in 3 hours,
    prove that Liza reads 20 pages in an hour. -/
theorem liza_reading_speed (suzie_pages_per_hour : ℕ) (liza_extra_pages : ℕ) :
  suzie_pages_per_hour = 15 →
  liza_extra_pages = 15 →
  ∃ (liza_pages_per_hour : ℕ),
    liza_pages_per_hour * 3 = suzie_pages_per_hour * 3 + liza_extra_pages ∧
    liza_pages_per_hour = 20 :=
by sorry

end NUMINAMATH_CALUDE_liza_reading_speed_l3938_393813


namespace NUMINAMATH_CALUDE_average_weight_B_and_C_l3938_393805

theorem average_weight_B_and_C (A B C : ℝ) : 
  (A + B + C) / 3 = 45 →
  (A + B) / 2 = 40 →
  B = 31 →
  (B + C) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_B_and_C_l3938_393805


namespace NUMINAMATH_CALUDE_eiffel_tower_lower_than_burj_khalifa_l3938_393873

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height : ℝ := 324

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height : ℝ := 830

/-- The difference in height between the Burj Khalifa and the Eiffel Tower -/
def height_difference : ℝ := burj_khalifa_height - eiffel_tower_height

/-- Theorem stating that the Eiffel Tower is 506 meters lower than the Burj Khalifa -/
theorem eiffel_tower_lower_than_burj_khalifa : 
  height_difference = 506 := by sorry

end NUMINAMATH_CALUDE_eiffel_tower_lower_than_burj_khalifa_l3938_393873


namespace NUMINAMATH_CALUDE_factorization_equality_l3938_393822

theorem factorization_equality (a b : ℝ) : a^2 * b - b^3 = b * (a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l3938_393822


namespace NUMINAMATH_CALUDE_point_coordinates_l3938_393812

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the second quadrant
def second_quadrant (p : Point) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Define distance to x-axis
def distance_to_x_axis (p : Point) : ℝ :=
  |p.2|

-- Define distance to y-axis
def distance_to_y_axis (p : Point) : ℝ :=
  |p.1|

theorem point_coordinates :
  ∀ p : Point,
    second_quadrant p →
    distance_to_x_axis p = 4 →
    distance_to_y_axis p = 3 →
    p = (-3, 4) :=
by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3938_393812


namespace NUMINAMATH_CALUDE_max_F_value_l3938_393835

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  units : Nat
  is_four_digit : thousands ≥ 1 ∧ thousands ≤ 9

/-- Checks if a number is an "eternal number" -/
def is_eternal (n : FourDigitNumber) : Prop :=
  n.hundreds + n.tens + n.units = 12

/-- Swaps digits as described in the problem -/
def swap_digits (n : FourDigitNumber) : FourDigitNumber :=
  { thousands := n.hundreds
  , hundreds := n.thousands
  , tens := n.units
  , units := n.tens
  , is_four_digit := by sorry }

/-- Calculates F(M) as defined in the problem -/
def F (m : FourDigitNumber) : Int :=
  let n := swap_digits m
  let m_val := 1000 * m.thousands + 100 * m.hundreds + 10 * m.tens + m.units
  let n_val := 1000 * n.thousands + 100 * n.hundreds + 10 * n.tens + n.units
  (m_val - n_val) / 9

/-- Main theorem -/
theorem max_F_value (m : FourDigitNumber) 
  (h1 : is_eternal m)
  (h2 : m.thousands = m.hundreds - m.units)
  (h3 : (F m) % 9 = 0) :
  F m ≤ 9 ∧ ∃ (m' : FourDigitNumber), F m' = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_F_value_l3938_393835


namespace NUMINAMATH_CALUDE_probability_three_girls_out_of_six_l3938_393811

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem probability_three_girls_out_of_six :
  binomial_probability 6 3 (1/2) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_girls_out_of_six_l3938_393811


namespace NUMINAMATH_CALUDE_unique_triangle_l3938_393820

/-- 
A triple of positive integers (a, a, b) represents an acute-angled isosceles triangle 
with perimeter 31 if and only if it satisfies the following conditions:
1. 2a + b = 31 (perimeter condition)
2. a < b < 2a (acute-angled isosceles condition)
-/
def is_valid_triangle (a b : ℕ) : Prop :=
  2 * a + b = 31 ∧ a < b ∧ b < 2 * a

/-- There exists exactly one triple of positive integers (a, a, b) that represents 
an acute-angled isosceles triangle with perimeter 31. -/
theorem unique_triangle : ∃! p : ℕ × ℕ, is_valid_triangle p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triangle_l3938_393820


namespace NUMINAMATH_CALUDE_factorization_equality_l3938_393874

theorem factorization_equality (x : ℝ) : 
  2*x*(x-3) + 3*(x-3) + 5*x^2*(x-3) = (x-3)*(5*x^2 + 2*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3938_393874


namespace NUMINAMATH_CALUDE_remainder_sum_l3938_393803

theorem remainder_sum (c d : ℤ) (hc : c % 60 = 58) (hd : d % 90 = 85) :
  (c + d) % 30 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3938_393803


namespace NUMINAMATH_CALUDE_curve_points_difference_l3938_393847

theorem curve_points_difference : 
  ∀ (a b : ℝ), a ≠ b → 
  (4 + a^2 = 8*a - 5) → 
  (4 + b^2 = 8*b - 5) → 
  |a - b| = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_curve_points_difference_l3938_393847


namespace NUMINAMATH_CALUDE_equation_solution_l3938_393826

theorem equation_solution : ∃ x : ℚ, (2 * x + 1 = 0) ∧ (x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3938_393826


namespace NUMINAMATH_CALUDE_maryville_population_increase_l3938_393885

/-- Calculates the average annual population increase given initial and final populations and the time period. -/
def averageAnnualIncrease (initialPopulation finalPopulation : ℕ) (years : ℕ) : ℚ :=
  (finalPopulation - initialPopulation : ℚ) / years

/-- Theorem stating that the average annual population increase in Maryville between 2000 and 2005 is 3400. -/
theorem maryville_population_increase : averageAnnualIncrease 450000 467000 5 = 3400 := by
  sorry

end NUMINAMATH_CALUDE_maryville_population_increase_l3938_393885


namespace NUMINAMATH_CALUDE_max_area_isosceles_trapezoidal_canal_l3938_393838

/-- 
Given an isosceles trapezoidal canal where the legs are equal to the smaller base,
this theorem states that the cross-sectional area is maximized when the angle of 
inclination of the legs is π/3 radians.
-/
theorem max_area_isosceles_trapezoidal_canal :
  ∀ (a : ℝ) (α : ℝ), 
  0 < a → 
  0 < α → 
  α < π / 2 →
  let S := a^2 * (1 + Real.cos α) * Real.sin α
  ∀ (β : ℝ), 0 < β → β < π / 2 → 
  a^2 * (1 + Real.cos β) * Real.sin β ≤ S →
  α = π / 3 :=
by sorry


end NUMINAMATH_CALUDE_max_area_isosceles_trapezoidal_canal_l3938_393838


namespace NUMINAMATH_CALUDE_probability_of_six_consecutive_heads_l3938_393898

-- Define a coin flip sequence as a list of booleans (true for heads, false for tails)
def CoinFlipSequence := List Bool

-- Function to check if a sequence has at least n consecutive heads
def hasConsecutiveHeads (n : Nat) (seq : CoinFlipSequence) : Bool :=
  sorry

-- Function to generate all possible coin flip sequences of length n
def allSequences (n : Nat) : List CoinFlipSequence :=
  sorry

-- Count the number of sequences with at least n consecutive heads
def countSequencesWithConsecutiveHeads (n : Nat) (seqs : List CoinFlipSequence) : Nat :=
  sorry

-- Theorem to prove
theorem probability_of_six_consecutive_heads :
  let allSeqs := allSequences 9
  let favorableSeqs := countSequencesWithConsecutiveHeads 6 allSeqs
  (favorableSeqs : ℚ) / (allSeqs.length : ℚ) = 49 / 512 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_six_consecutive_heads_l3938_393898


namespace NUMINAMATH_CALUDE_cosine_product_sqrt_eight_l3938_393844

theorem cosine_product_sqrt_eight : 
  Real.sqrt ((3 - Real.cos (π / 8) ^ 2) * (3 - Real.cos (π / 4) ^ 2) * (3 - Real.cos (3 * π / 8) ^ 2)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_sqrt_eight_l3938_393844


namespace NUMINAMATH_CALUDE_ruths_sandwiches_l3938_393804

theorem ruths_sandwiches (total : ℕ) (brother : ℕ) (first_cousin : ℕ) (other_cousins : ℕ) (left : ℕ) :
  total = 10 →
  brother = 2 →
  first_cousin = 2 →
  other_cousins = 2 →
  left = 3 →
  total - (brother + first_cousin + other_cousins + left) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ruths_sandwiches_l3938_393804


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_l3938_393858

theorem arithmetic_progression_squares (a d : ℝ) : 
  (a - d)^2 + a^2 = 100 ∧ a^2 + (a + d)^2 = 164 →
  ((a - d, a, a + d) = (6, 8, 10) ∨
   (a - d, a, a + d) = (-10, -8, -6) ∨
   (a - d, a, a + d) = (-7 * Real.sqrt 2, Real.sqrt 2, 9 * Real.sqrt 2) ∨
   (a - d, a, a + d) = (10 * Real.sqrt 2, 8 * Real.sqrt 2, Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_l3938_393858


namespace NUMINAMATH_CALUDE_quiz_total_points_l3938_393819

/-- Represents a quiz with a specified number of questions, where each question after
    the first is worth a fixed number of points more than the preceding question. -/
structure Quiz where
  num_questions : ℕ
  point_increment : ℕ
  third_question_points : ℕ

/-- Calculates the total points for a given quiz. -/
def total_points (q : Quiz) : ℕ :=
  let first_question_points := q.third_question_points - 2 * q.point_increment
  let last_question_points := first_question_points + (q.num_questions - 1) * q.point_increment
  (first_question_points + last_question_points) * q.num_questions / 2

/-- Theorem stating that a quiz with 8 questions, where each question after the first
    is worth 4 points more than the preceding question, and the third question is
    worth 39 points, has a total of 360 points. -/
theorem quiz_total_points :
  ∀ (q : Quiz), q.num_questions = 8 ∧ q.point_increment = 4 ∧ q.third_question_points = 39 →
  total_points q = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_total_points_l3938_393819


namespace NUMINAMATH_CALUDE_motel_rent_theorem_l3938_393849

/-- Represents the total rent charged by a motel --/
def TotalRent (x y : ℕ) : ℕ := 40 * x + 60 * y

/-- The problem statement --/
theorem motel_rent_theorem (x y : ℕ) :
  (TotalRent (x + 10) (y - 10) = (TotalRent x y) / 2) →
  TotalRent x y = 800 :=
by sorry

end NUMINAMATH_CALUDE_motel_rent_theorem_l3938_393849


namespace NUMINAMATH_CALUDE_unique_circle_circumference_equals_area_l3938_393886

theorem unique_circle_circumference_equals_area :
  ∃! r : ℝ, r > 0 ∧ 2 * Real.pi * r = Real.pi * r^2 := by sorry

end NUMINAMATH_CALUDE_unique_circle_circumference_equals_area_l3938_393886


namespace NUMINAMATH_CALUDE_semicircle_problem_l3938_393883

theorem semicircle_problem (N : ℕ) (r : ℝ) (h_positive : r > 0) : 
  let A := (N * π * r^2) / 2
  let B := (π * r^2 / 2) * (N^2 - N)
  (N ≥ 1) → (A / B = 1 / 24) → (N = 25) := by
sorry

end NUMINAMATH_CALUDE_semicircle_problem_l3938_393883


namespace NUMINAMATH_CALUDE_largest_value_l3938_393817

theorem largest_value (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  (a - b > a) ∧ (a - b > a + b) ∧ (a - b > a * b) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l3938_393817


namespace NUMINAMATH_CALUDE_program_output_for_351_l3938_393868

def program_output (x : ℕ) : ℕ :=
  if 100 < x ∧ x < 1000 then
    let a := x / 100
    let b := (x - a * 100) / 10
    let c := x % 10
    100 * c + 10 * b + a
  else
    x

theorem program_output_for_351 :
  program_output 351 = 153 :=
by
  sorry

end NUMINAMATH_CALUDE_program_output_for_351_l3938_393868


namespace NUMINAMATH_CALUDE_investment_principal_l3938_393892

/-- Proves that an investment with a monthly interest payment of $228 and a simple annual interest rate of 9% has a principal amount of $30,400. -/
theorem investment_principal (monthly_interest : ℝ) (annual_rate : ℝ) (principal : ℝ) : 
  monthly_interest = 228 →
  annual_rate = 0.09 →
  principal = (monthly_interest * 12) / annual_rate →
  principal = 30400 := by
  sorry


end NUMINAMATH_CALUDE_investment_principal_l3938_393892


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3938_393878

theorem quadratic_root_problem (b : ℝ) :
  (∃ x₀ : ℝ, x₀^2 - 4*x₀ + b = 0 ∧ (-x₀)^2 + 4*(-x₀) - b = 0) →
  (∃ x : ℝ, x > 0 ∧ x^2 + b*x - 4 = 0 ∧ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3938_393878


namespace NUMINAMATH_CALUDE_largest_in_set_l3938_393857

def S (a : ℝ) : Set ℝ := {-2*a, 3*a, 18/a, a^2, 2}

theorem largest_in_set :
  ∀ a : ℝ, a = 3 → 
  ∃ m : ℝ, m ∈ S a ∧ ∀ x ∈ S a, x ≤ m ∧ 
  m = 3*a ∧ m = a^2 := by sorry

end NUMINAMATH_CALUDE_largest_in_set_l3938_393857


namespace NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l3938_393846

theorem tens_digit_of_2023_pow_2024_minus_2025 :
  (2023^2024 - 2025) % 100 / 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l3938_393846


namespace NUMINAMATH_CALUDE_infinite_solutions_implies_c_equals_three_l3938_393899

theorem infinite_solutions_implies_c_equals_three :
  (∀ (c : ℝ), (∃ (S : Set ℝ), Set.Infinite S ∧ 
    ∀ (y : ℝ), y ∈ S → (3 * (5 + 2 * c * y) = 18 * y + 15))) →
  c = 3 :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_implies_c_equals_three_l3938_393899


namespace NUMINAMATH_CALUDE_product_zero_iff_one_zero_l3938_393881

theorem product_zero_iff_one_zero (a b c : ℝ) : a * b * c = 0 ↔ a = 0 ∨ b = 0 ∨ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_iff_one_zero_l3938_393881


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3938_393823

theorem polynomial_divisibility (F : ℤ → ℤ) 
  (h1 : ∀ x y : ℤ, F (x + y) - F x - F y = (x * y) * (F 1 - 1))
  (h2 : (F 2) % 5 = 0)
  (h3 : (F 5) % 2 = 0) :
  (F 7) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3938_393823


namespace NUMINAMATH_CALUDE_defective_units_count_prove_defective_units_l3938_393859

theorem defective_units_count : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_units customer_a customer_b customer_c defective_units =>
    total_units = 20 ∧
    customer_a = 3 ∧
    customer_b = 5 ∧
    customer_c = 7 ∧
    defective_units = total_units - (customer_a + customer_b + customer_c) ∧
    defective_units = 5

theorem prove_defective_units : ∃ (d : ℕ), defective_units_count 20 3 5 7 d :=
  sorry

end NUMINAMATH_CALUDE_defective_units_count_prove_defective_units_l3938_393859


namespace NUMINAMATH_CALUDE_min_ones_in_sum_l3938_393841

/-- Count the number of '1's in the binary representation of an integer -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem min_ones_in_sum (a b : ℕ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (ca : countOnes a = 20041) 
  (cb : countOnes b = 20051) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ countOnes x = 20041 ∧ countOnes y = 20051 ∧ countOnes (x + y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_ones_in_sum_l3938_393841


namespace NUMINAMATH_CALUDE_rectangular_plot_perimeter_l3938_393856

/-- Given a rectangular plot with length 10 meters more than width,
    and fencing cost of 1430 Rs at 6.5 Rs/meter,
    prove the perimeter is 220 meters. -/
theorem rectangular_plot_perimeter :
  ∀ (width length : ℝ),
  length = width + 10 →
  6.5 * (2 * (length + width)) = 1430 →
  2 * (length + width) = 220 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_perimeter_l3938_393856


namespace NUMINAMATH_CALUDE_chord_length_l3938_393834

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by sorry

end NUMINAMATH_CALUDE_chord_length_l3938_393834


namespace NUMINAMATH_CALUDE_susan_reading_time_l3938_393871

/-- Represents the ratio of time spent on different activities -/
structure TimeRatio where
  swimming : ℕ
  reading : ℕ
  hangingOut : ℕ

/-- Calculates the time spent on an activity given the total time of another activity -/
def calculateTime (ratio : TimeRatio) (knownActivity : ℕ) (knownTime : ℕ) (targetActivity : ℕ) : ℕ :=
  (targetActivity * knownTime) / knownActivity

theorem susan_reading_time (ratio : TimeRatio) 
    (h1 : ratio.swimming = 1)
    (h2 : ratio.reading = 4)
    (h3 : ratio.hangingOut = 10)
    (h4 : calculateTime ratio ratio.hangingOut 20 ratio.reading = 8) : 
  ∃ (readingTime : ℕ), readingTime = 8 ∧ 
    readingTime = calculateTime ratio ratio.hangingOut 20 ratio.reading :=
by sorry

end NUMINAMATH_CALUDE_susan_reading_time_l3938_393871


namespace NUMINAMATH_CALUDE_fish_theorem_l3938_393870

def fish_problem (leo agrey sierra returned : ℕ) : Prop :=
  let total := leo + agrey + sierra
  agrey = leo + 20 ∧ 
  sierra = agrey + 15 ∧ 
  leo = 40 ∧ 
  returned = 30 ∧ 
  total - returned = 145

theorem fish_theorem : 
  ∃ (leo agrey sierra returned : ℕ), fish_problem leo agrey sierra returned :=
by
  sorry

end NUMINAMATH_CALUDE_fish_theorem_l3938_393870


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_geq_one_l3938_393880

theorem negation_of_absolute_value_geq_one :
  (¬ ∀ x : ℝ, |x| ≥ 1) ↔ (∃ x : ℝ, |x| < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_geq_one_l3938_393880


namespace NUMINAMATH_CALUDE_age_and_marriage_relations_l3938_393810

-- Define the people
inductive Person : Type
| Roman : Person
| Oleg : Person
| Ekaterina : Person
| Zhanna : Person

-- Define the age relation
def olderThan : Person → Person → Prop := sorry

-- Define the marriage relation
def marriedTo : Person → Person → Prop := sorry

-- Theorem statement
theorem age_and_marriage_relations :
  -- Each person has a different age
  (∀ p q : Person, p ≠ q → (olderThan p q ∨ olderThan q p)) →
  -- Each husband is older than his wife
  (∀ p q : Person, marriedTo p q → olderThan p q) →
  -- Zhanna is older than Oleg
  olderThan Person.Zhanna Person.Oleg →
  -- There are exactly two married couples
  (∃! (p1 p2 q1 q2 : Person),
    p1 ≠ p2 ∧ q1 ≠ q2 ∧ p1 ≠ q1 ∧ p1 ≠ q2 ∧ p2 ≠ q1 ∧ p2 ≠ q2 ∧
    marriedTo p1 p2 ∧ marriedTo q1 q2) →
  -- Conclusion: Oleg is older than Ekaterina and Roman is the oldest and married to Zhanna
  olderThan Person.Oleg Person.Ekaterina ∧
  marriedTo Person.Roman Person.Zhanna ∧
  (∀ p : Person, p ≠ Person.Roman → olderThan Person.Roman p) :=
by sorry

end NUMINAMATH_CALUDE_age_and_marriage_relations_l3938_393810


namespace NUMINAMATH_CALUDE_range_of_m_l3938_393850

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + 1

theorem range_of_m (m : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioc 0 1 ∧
    ∀ a : ℝ, a ∈ Set.Icc (-2) 0 →
      2*m*(Real.exp a) + f a x₀ > a^2 + 2*a + 4) →
  m ∈ Set.Ioo 1 (Real.exp 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3938_393850


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3938_393828

theorem rectangular_field_area (L W : ℝ) (h1 : L = 30) (h2 : 2 * W + L = 84) : L * W = 810 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3938_393828


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3938_393830

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3938_393830


namespace NUMINAMATH_CALUDE_star_calculation_l3938_393852

/-- The ⋆ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x^2 + y) * (x - y)

/-- Theorem stating that 2 ⋆ (3 ⋆ 4) = -135 -/
theorem star_calculation : star 2 (star 3 4) = -135 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l3938_393852


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3938_393855

theorem polynomial_remainder (x : ℝ) : 
  let Q := fun (x : ℝ) => 8*x^4 - 18*x^3 - 6*x^2 + 4*x - 30
  let divisor := fun (x : ℝ) => 2*x - 8
  Q 4 = 786 ∧ (∃ P : ℝ → ℝ, ∀ x, Q x = P x * divisor x + 786) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3938_393855


namespace NUMINAMATH_CALUDE_smallest_divisible_integer_l3938_393816

theorem smallest_divisible_integer : ∃ (M : ℕ), 
  (M = 362) ∧ 
  (∀ (k : ℕ), k < M → ¬(
    (∃ (i : Fin 3), 2^2 ∣ (k + i)) ∧
    (∃ (i : Fin 3), 3^2 ∣ (k + i)) ∧
    (∃ (i : Fin 3), 7^2 ∣ (k + i)) ∧
    (∃ (i : Fin 3), 11^2 ∣ (k + i))
  )) ∧
  (∃ (i : Fin 3), 2^2 ∣ (M + i)) ∧
  (∃ (i : Fin 3), 3^2 ∣ (M + i)) ∧
  (∃ (i : Fin 3), 7^2 ∣ (M + i)) ∧
  (∃ (i : Fin 3), 11^2 ∣ (M + i)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_integer_l3938_393816


namespace NUMINAMATH_CALUDE_parabola_equation_l3938_393867

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  p > 0 ∧ y^2 = 2 * p * x

-- Define the area of triangle AOB
def triangle_area (area : ℝ) : Prop := area = Real.sqrt 3

-- Theorem statement
theorem parabola_equation (a b p : ℝ) :
  (∃ x y : ℝ, hyperbola a b x y) →
  eccentricity 2 →
  (∃ x y : ℝ, parabola p x y) →
  triangle_area (Real.sqrt 3) →
  p = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3938_393867


namespace NUMINAMATH_CALUDE_father_walking_time_l3938_393802

/-- The time (in minutes) it takes Xiaoming to cycle from the meeting point to B -/
def meeting_to_B : ℝ := 18

/-- Xiaoming's cycling speed is 4 times his father's walking speed -/
def speed_ratio : ℝ := 4

/-- The time (in minutes) it takes Xiaoming's father to walk from the meeting point to A -/
def father_time : ℝ := 288

theorem father_walking_time :
  ∀ (xiaoming_speed father_speed : ℝ),
  xiaoming_speed > 0 ∧ father_speed > 0 →
  xiaoming_speed = speed_ratio * father_speed →
  father_time = 4 * (speed_ratio * meeting_to_B) := by
  sorry

end NUMINAMATH_CALUDE_father_walking_time_l3938_393802


namespace NUMINAMATH_CALUDE_quadratic_equation_always_has_real_root_l3938_393829

theorem quadratic_equation_always_has_real_root (a : ℝ) : ∃ x : ℝ, a * x^2 - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_always_has_real_root_l3938_393829


namespace NUMINAMATH_CALUDE_paint_needed_to_buy_l3938_393843

theorem paint_needed_to_buy (total_paint : ℕ) (available_paint : ℕ) : 
  total_paint = 333 → available_paint = 157 → total_paint - available_paint = 176 := by
  sorry

end NUMINAMATH_CALUDE_paint_needed_to_buy_l3938_393843


namespace NUMINAMATH_CALUDE_absolute_value_theorem_l3938_393853

theorem absolute_value_theorem (x q : ℝ) (h1 : |x - 3| = q) (h2 : x < 3) :
  x - q = 3 - 2*q := by sorry

end NUMINAMATH_CALUDE_absolute_value_theorem_l3938_393853


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l3938_393864

/-- Represents a seating arrangement in an examination room -/
structure ExamRoom where
  rows : Nat
  columns : Nat
  total_seats : Nat

/-- Calculates the number of possible seating arrangements for two students
    who cannot sit adjacent to each other in the given exam room -/
def count_seating_arrangements (room : ExamRoom) : Nat :=
  sorry

/-- Theorem stating that the number of seating arrangements for two students
    in a 5x6 exam room with 30 seats, where they cannot sit adjacent to each other,
    is 772 -/
theorem seating_arrangements_count :
  let exam_room : ExamRoom := ⟨5, 6, 30⟩
  count_seating_arrangements exam_room = 772 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l3938_393864


namespace NUMINAMATH_CALUDE_polynomial_equality_l3938_393831

theorem polynomial_equality (x : ℝ) : 
  (3*x^3 + 2*x^2 + 5*x + 9)*(x - 2) - (x - 2)*(2*x^3 + 5*x^2 - 74) + (4*x - 17)*(x - 2)*(x + 4) 
  = x^4 + 2*x^3 - 5*x^2 + 9*x - 30 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3938_393831


namespace NUMINAMATH_CALUDE_hexagon_area_from_square_l3938_393879

theorem hexagon_area_from_square (s : ℝ) (h_square_area : s^2 = Real.sqrt 3) :
  let hexagon_area := 6 * (Real.sqrt 3 / 4 * s^2)
  hexagon_area = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_from_square_l3938_393879


namespace NUMINAMATH_CALUDE_opposite_signs_and_greater_absolute_value_l3938_393837

theorem opposite_signs_and_greater_absolute_value (a b : ℝ) 
  (h1 : a * b < 0) (h2 : a + b > 0) : 
  (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0) ∧ 
  (a > 0 → |a| > |b|) ∧ 
  (b > 0 → |b| > |a|) := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_and_greater_absolute_value_l3938_393837


namespace NUMINAMATH_CALUDE_cake_difference_l3938_393889

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes bought and sold is 63. -/
theorem cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ)
    (h1 : initial = 13)
    (h2 : sold = 91)
    (h3 : bought = 154) :
    bought - sold = 63 := by
  sorry

end NUMINAMATH_CALUDE_cake_difference_l3938_393889


namespace NUMINAMATH_CALUDE_raja_income_proof_l3938_393897

/-- Raja's monthly income in rupees -/
def monthly_income : ℝ := 25000

/-- The amount Raja saves in rupees -/
def savings : ℝ := 5000

/-- Percentage of income spent on household items -/
def household_percentage : ℝ := 0.60

/-- Percentage of income spent on clothes -/
def clothes_percentage : ℝ := 0.10

/-- Percentage of income spent on medicines -/
def medicine_percentage : ℝ := 0.10

theorem raja_income_proof :
  monthly_income * household_percentage +
  monthly_income * clothes_percentage +
  monthly_income * medicine_percentage +
  savings = monthly_income :=
by sorry

end NUMINAMATH_CALUDE_raja_income_proof_l3938_393897


namespace NUMINAMATH_CALUDE_triangle_type_l3938_393833

theorem triangle_type (A : Real) (h1 : 0 < A ∧ A < π) 
  (h2 : Real.sin A + Real.cos A = 7/12) : 
  ∀ (B C : Real), 0 < B ∧ 0 < C → A + B + C = π → 
  (A < π/2 ∧ B < π/2 ∧ C < π/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_type_l3938_393833


namespace NUMINAMATH_CALUDE_special_function_properties_l3938_393893

open Real

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y - 1) ∧
  (∀ x, x > 0 → f x > 1) ∧
  (f 3 = 4)

/-- The main theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (∀ x y, x < y → f x < f y) ∧ (f 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l3938_393893


namespace NUMINAMATH_CALUDE_subset_implies_a_zero_l3938_393821

theorem subset_implies_a_zero (a : ℝ) :
  let P : Set ℝ := {x | x^2 ≠ 1}
  let Q : Set ℝ := {x | a * x = 1}
  Q ⊆ P → a = 0 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_zero_l3938_393821


namespace NUMINAMATH_CALUDE_inheritance_calculation_l3938_393865

theorem inheritance_calculation (x : ℝ) : 
  0.25 * x + 0.15 * (0.75 * x - 5000) + 5000 = 16500 → x = 33794 :=
by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l3938_393865


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l3938_393840

theorem cubic_root_sum_squares (a b c : ℝ) : 
  (a^3 - 4*a^2 + 7*a - 2 = 0) → 
  (b^3 - 4*b^2 + 7*b - 2 = 0) → 
  (c^3 - 4*c^2 + 7*c - 2 = 0) → 
  a^2 + b^2 + c^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l3938_393840


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3938_393869

theorem root_sum_theorem (a b : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I : ℂ) ^ 3 + a * (2 + Complex.I : ℂ) + b = 0 →
  a + b = 9 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3938_393869


namespace NUMINAMATH_CALUDE_fraction_simplification_l3938_393863

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hab : a ≠ b) :
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3938_393863


namespace NUMINAMATH_CALUDE_divisibility_of_quotient_l3938_393854

theorem divisibility_of_quotient (a b n : ℕ) (h1 : a ≠ b) (h2 : n ∣ (a^n - b^n)) :
  n ∣ ((a^n - b^n) / (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_quotient_l3938_393854


namespace NUMINAMATH_CALUDE_incorrect_classification_l3938_393891

/-- Represents a proof method -/
inductive ProofMethod
| Synthetic
| Analytic

/-- Represents the nature of a proof -/
inductive ProofNature
| Direct
| Indirect

/-- Defines the correct classification of proof methods -/
def correct_classification (method : ProofMethod) : ProofNature :=
  match method with
  | ProofMethod.Synthetic => ProofNature.Direct
  | ProofMethod.Analytic => ProofNature.Direct

/-- Theorem stating that the given classification is incorrect -/
theorem incorrect_classification :
  ¬(correct_classification ProofMethod.Synthetic = ProofNature.Direct ∧
    correct_classification ProofMethod.Analytic = ProofNature.Indirect) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_classification_l3938_393891


namespace NUMINAMATH_CALUDE_simplify_expression_l3938_393895

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) : x^3 * (y^3 / x)^2 = x * y^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3938_393895


namespace NUMINAMATH_CALUDE_add_3333_minutes_to_leap_day_noon_l3938_393887

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Represents the starting date and time -/
def startDateTime : DateTime :=
  { year := 2020, month := 2, day := 29, hour := 12, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 3333

/-- The expected result date and time -/
def expectedDateTime : DateTime :=
  { year := 2020, month := 3, day := 2, hour := 19, minute := 33 }

/-- Function to add minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

theorem add_3333_minutes_to_leap_day_noon :
  addMinutes startDateTime minutesToAdd = expectedDateTime := by sorry

end NUMINAMATH_CALUDE_add_3333_minutes_to_leap_day_noon_l3938_393887


namespace NUMINAMATH_CALUDE_evaluate_F_with_f_l3938_393882

-- Define function f
def f (a : ℝ) : ℝ := a^2 - 1

-- Define function F
def F (a b : ℝ) : ℝ := 3*b^2 + 2*a

-- Theorem statement
theorem evaluate_F_with_f : F 2 (f 3) = 196 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_F_with_f_l3938_393882


namespace NUMINAMATH_CALUDE_color_stamps_count_l3938_393876

/-- The number of color stamps sold by the postal service -/
def color_stamps : ℕ := 1102609 - 523776

/-- The total number of stamps sold by the postal service -/
def total_stamps : ℕ := 1102609

/-- The number of black-and-white stamps sold by the postal service -/
def bw_stamps : ℕ := 523776

theorem color_stamps_count : color_stamps = 578833 := by
  sorry

end NUMINAMATH_CALUDE_color_stamps_count_l3938_393876


namespace NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l3938_393825

theorem intersection_empty_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
  let B : Set ℝ := {x | x*(x-m) > 0}
  (A ∩ B = ∅) → m ≥ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_m_range_l3938_393825


namespace NUMINAMATH_CALUDE_derivative_through_point_l3938_393884

theorem derivative_through_point (a : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + a*x + 1
  let f' : ℝ → ℝ := λ x => 2*x + a
  f' 2 = 4 → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_through_point_l3938_393884


namespace NUMINAMATH_CALUDE_six_less_than_twice_square_of_four_l3938_393818

theorem six_less_than_twice_square_of_four : (2 * 4^2) - 6 = 26 := by
  sorry

end NUMINAMATH_CALUDE_six_less_than_twice_square_of_four_l3938_393818


namespace NUMINAMATH_CALUDE_triangle_shape_l3938_393890

theorem triangle_shape (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (eq : Real.sin C + Real.sin (B - A) = Real.sin (2 * A)) : 
  (A = B ∨ A = π / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l3938_393890


namespace NUMINAMATH_CALUDE_james_missed_two_questions_l3938_393806

/-- Represents the quiz bowl scoring system and James' performance -/
structure QuizBowl where
  points_per_correct : ℕ := 2
  bonus_points : ℕ := 4
  num_rounds : ℕ := 5
  questions_per_round : ℕ := 5
  james_points : ℕ := 66

/-- Calculates the number of questions James missed based on his score -/
def questions_missed (qb : QuizBowl) : ℕ :=
  let max_points := qb.num_rounds * (qb.questions_per_round * qb.points_per_correct + qb.bonus_points)
  (max_points - qb.james_points) / qb.points_per_correct

/-- Theorem stating that James missed exactly 2 questions -/
theorem james_missed_two_questions (qb : QuizBowl) : questions_missed qb = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_missed_two_questions_l3938_393806


namespace NUMINAMATH_CALUDE_triangle_area_l3938_393836

-- Define the triangle
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  cos_angle : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.side1 = 5 ∧ 
  t.side2 = 3 ∧ 
  5 * t.cos_angle^2 - 7 * t.cos_angle - 6 = 0

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h : triangle_conditions t) : 
  (1/2) * t.side1 * t.side2 * Real.sqrt (1 - t.cos_angle^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3938_393836


namespace NUMINAMATH_CALUDE_avery_build_time_l3938_393842

theorem avery_build_time (tom_time : ℝ) (total_time : ℝ) : 
  tom_time = 4 →
  (1 / 2 + 1 / tom_time) + 1 / tom_time = 1 →
  2 = total_time :=
by sorry

end NUMINAMATH_CALUDE_avery_build_time_l3938_393842


namespace NUMINAMATH_CALUDE_selena_remaining_money_is_33_74_l3938_393862

/-- Calculates the amount Selena will be left with after paying for her meal including taxes. -/
def selena_remaining_money (tip : ℚ) (steak_price : ℚ) (burger_price : ℚ) (ice_cream_price : ℚ)
  (steak_tax : ℚ) (burger_tax : ℚ) (ice_cream_tax : ℚ) : ℚ :=
  let steak_total := 2 * steak_price * (1 + steak_tax)
  let burger_total := 2 * burger_price * (1 + burger_tax)
  let ice_cream_total := 3 * ice_cream_price * (1 + ice_cream_tax)
  tip - (steak_total + burger_total + ice_cream_total)

/-- Theorem stating that Selena will be left with $33.74 after paying for her meal including taxes. -/
theorem selena_remaining_money_is_33_74 :
  selena_remaining_money 99 24 3.5 2 0.07 0.06 0.08 = 33.74 := by
  sorry

end NUMINAMATH_CALUDE_selena_remaining_money_is_33_74_l3938_393862


namespace NUMINAMATH_CALUDE_middle_brother_height_l3938_393872

theorem middle_brother_height (h₁ h₂ h₃ : ℝ) :
  h₁ ≤ h₂ ∧ h₂ ≤ h₃ →
  (h₁ + h₂ + h₃) / 3 = 1.74 →
  (h₁ + h₃) / 2 = 1.75 →
  h₂ = 1.72 := by
sorry

end NUMINAMATH_CALUDE_middle_brother_height_l3938_393872


namespace NUMINAMATH_CALUDE_waiter_tables_l3938_393851

/-- Calculates the number of tables given the initial number of customers,
    the number of customers who left, and the number of people at each remaining table. -/
def calculate_tables (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) : ℕ :=
  (initial_customers - customers_left) / people_per_table

/-- Theorem stating that for the given problem, the number of tables is 5. -/
theorem waiter_tables : calculate_tables 62 17 9 = 5 := by
  sorry


end NUMINAMATH_CALUDE_waiter_tables_l3938_393851


namespace NUMINAMATH_CALUDE_golf_distance_l3938_393808

/-- 
Given a golf scenario where:
1. The distance from the starting tee to the hole is 250 yards.
2. On the second turn, the ball traveled half as far as it did on the first turn.
3. After the second turn, the ball landed 20 yards beyond the hole.
This theorem proves that the distance the ball traveled on the first turn is 180 yards.
-/
theorem golf_distance (first_turn : ℝ) (second_turn : ℝ) : 
  (first_turn + second_turn = 250 + 20) →  -- Total distance is to the hole plus 20 yards beyond
  (second_turn = first_turn / 2) →         -- Second turn is half of the first turn
  (first_turn = 180) :=                    -- The distance of the first turn is 180 yards
by sorry

end NUMINAMATH_CALUDE_golf_distance_l3938_393808


namespace NUMINAMATH_CALUDE_carol_to_cathy_ratio_ratio_is_one_to_one_l3938_393815

-- Define the number of cars each person owns
def cathy_cars : ℕ := 5
def carol_cars : ℕ := cathy_cars
def lindsey_cars : ℕ := cathy_cars + 4
def susan_cars : ℕ := carol_cars - 2

-- Theorem to prove
theorem carol_to_cathy_ratio : 
  carol_cars = cathy_cars := by sorry

-- The ratio is 1:1 if the numbers are equal
theorem ratio_is_one_to_one : 
  carol_cars = cathy_cars → (carol_cars : ℚ) / cathy_cars = 1 := by sorry

end NUMINAMATH_CALUDE_carol_to_cathy_ratio_ratio_is_one_to_one_l3938_393815


namespace NUMINAMATH_CALUDE_power_of_two_triplets_l3938_393894

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def satisfies_conditions (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_power_of_two (a * b - c) ∧
  is_power_of_two (b * c - a) ∧
  is_power_of_two (c * a - b)

theorem power_of_two_triplets :
  ∀ a b c : ℕ,
    satisfies_conditions a b c ↔
      (a = 2 ∧ b = 2 ∧ c = 2) ∨
      (a = 2 ∧ b = 2 ∧ c = 3) ∨
      (a = 2 ∧ b = 3 ∧ c = 3) ∨
      (a = 2 ∧ b = 3 ∧ c = 6) ∨
      (a = 3 ∧ b = 5 ∧ c = 7) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_triplets_l3938_393894


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l3938_393861

/-- Represents the chemical mixture problem --/
theorem chemical_mixture_problem (a w : ℝ) 
  (h1 : a / (a + w + 2) = 1 / 4)
  (h2 : (a + 2) / (a + w + 2) = 3 / 8) :
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l3938_393861


namespace NUMINAMATH_CALUDE_vector_magnitude_l3938_393875

/-- Given plane vectors a and b, prove that the magnitude of a + 2b is 5√2 -/
theorem vector_magnitude (a b : ℝ × ℝ) (ha : a = (3, 5)) (hb : b = (-2, 1)) :
  ‖a + 2 • b‖ = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3938_393875


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l3938_393824

def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]

theorem matrix_equation_proof :
  N^3 - 3 • N^2 + 3 • N = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l3938_393824


namespace NUMINAMATH_CALUDE_pyramid_base_area_l3938_393839

theorem pyramid_base_area (slant_height height : ℝ) :
  slant_height = 5 →
  height = 7 →
  ∃ (side_length : ℝ), 
    side_length ^ 2 + slant_height ^ 2 = height ^ 2 ∧
    (side_length ^ 2) * 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_area_l3938_393839


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l3938_393848

theorem cos_two_theta_value (θ : Real) 
  (h : Real.sin (θ / 2) - Real.cos (θ / 2) = Real.sqrt 6 / 3) : 
  Real.cos (2 * θ) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l3938_393848


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_l3938_393896

/-- The number of kids staying home during summer break in Lawrence county -/
def kids_staying_home : ℕ := 907611

/-- The number of kids going to camp from Lawrence county -/
def kids_going_to_camp : ℕ := 455682

/-- The total number of kids in Lawrence county -/
def total_kids : ℕ := kids_staying_home + kids_going_to_camp

/-- Theorem stating that the total number of kids in Lawrence county
    is equal to the sum of kids staying home and kids going to camp -/
theorem lawrence_county_kids_count :
  total_kids = kids_staying_home + kids_going_to_camp := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_count_l3938_393896


namespace NUMINAMATH_CALUDE_find_a20_l3938_393832

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem find_a20 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) 
  (h2 : a 1 + a 3 + a 5 = 105) (h3 : a 2 + a 4 + a 6 = 99) : 
  a 20 = 1 := by sorry

end NUMINAMATH_CALUDE_find_a20_l3938_393832


namespace NUMINAMATH_CALUDE_sqrt_factorial_fraction_l3938_393800

theorem sqrt_factorial_fraction : 
  let factorial_10 : ℕ := 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  let denominator : ℕ := 2 * 3 * 7 * 7
  Real.sqrt (factorial_10 / denominator) = 120 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_sqrt_factorial_fraction_l3938_393800


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3938_393888

theorem inequality_solution_range (k : ℝ) : 
  (∃ (x y : ℕ), x ≠ y ∧ 
    (∀ (z : ℕ), z > 0 → (k * (z : ℝ)^2 ≤ Real.log z + 1) ↔ (z = x ∨ z = y))) →
  ((Real.log 3 + 1) / 9 < k ∧ k ≤ (Real.log 2 + 1) / 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3938_393888


namespace NUMINAMATH_CALUDE_sum_of_digits_653xy_divisible_by_80_l3938_393866

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem sum_of_digits_653xy_divisible_by_80 (x y : ℕ) :
  x < 10 →
  y < 10 →
  is_divisible_by (653 * 100 + x * 10 + y) 80 →
  x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_653xy_divisible_by_80_l3938_393866


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3938_393877

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ + a₁₀ = 16, the sum a₄ + a₆ + a₈ = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 2 + a 10 = 16) : 
  a 4 + a 6 + a 8 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3938_393877
