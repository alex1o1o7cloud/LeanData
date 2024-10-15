import Mathlib

namespace NUMINAMATH_CALUDE_negative_three_plus_nine_equals_six_l3136_313600

theorem negative_three_plus_nine_equals_six : (-3) + 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_plus_nine_equals_six_l3136_313600


namespace NUMINAMATH_CALUDE_remainder_theorem_l3136_313653

theorem remainder_theorem (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3136_313653


namespace NUMINAMATH_CALUDE_tangent_parallel_line_a_value_l3136_313685

-- Define the curve
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 6 * x + 2

-- Define the point where the tangent touches the curve
def point : ℝ × ℝ := (1, 5)

-- Define the general form of the line parallel to the tangent
def parallel_line (a x y : ℝ) : Prop := 2 * a * x - y - 6 = 0

-- Theorem statement
theorem tangent_parallel_line_a_value :
  ∃ (a : ℝ), 
    (f point.1 = point.2) ∧ 
    (f' point.1 = 2 * a) ∧ 
    (parallel_line a point.1 point.2) ∧
    (a = 4) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_line_a_value_l3136_313685


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l3136_313684

theorem square_sum_given_sum_and_product (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 3) :
  2 * a^2 + 2 * b^2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l3136_313684


namespace NUMINAMATH_CALUDE_visits_needed_is_eleven_l3136_313679

/-- The cost of headphones in rubles -/
def headphones_cost : ℕ := 275

/-- The cost of a combined pool and sauna visit in rubles -/
def combined_cost : ℕ := 250

/-- The difference in cost between pool-only and sauna-only visits in rubles -/
def pool_sauna_diff : ℕ := 200

/-- Calculates the number of pool-only visits needed to save for headphones -/
def visits_needed : ℕ :=
  let sauna_cost := (combined_cost - pool_sauna_diff) / 2
  let pool_only_cost := sauna_cost + pool_sauna_diff
  let savings_per_visit := combined_cost - pool_only_cost
  (headphones_cost + savings_per_visit - 1) / savings_per_visit

theorem visits_needed_is_eleven : visits_needed = 11 := by
  sorry

end NUMINAMATH_CALUDE_visits_needed_is_eleven_l3136_313679


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3136_313647

/-- A quadratic function satisfying specific conditions -/
def f (x : ℝ) : ℝ := 2 * (x - 2)^2 - 8

/-- The theorem stating the properties of the quadratic function f -/
theorem quadratic_function_properties :
  (∀ x, f (x + 2) = f (2 - x)) ∧
  (∀ x, f x ≥ -8) ∧
  (f 1 = -6) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) 4, -8 ≤ f x ∧ f x < f (-1)) := by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_quadratic_function_properties_l3136_313647


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3136_313619

theorem complex_modulus_problem (a : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (2 - i) / (a + i) = b * i) (b : ℝ) : 
  Complex.abs (4 * a + Complex.I * Real.sqrt 2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3136_313619


namespace NUMINAMATH_CALUDE_saheed_kayla_earnings_ratio_l3136_313661

/-- Proves that the ratio of Saheed's earnings to Kayla's earnings is 4:1 -/
theorem saheed_kayla_earnings_ratio :
  let vika_earnings : ℕ := 84
  let kayla_earnings : ℕ := vika_earnings - 30
  let saheed_earnings : ℕ := 216
  (saheed_earnings : ℚ) / kayla_earnings = 4 := by
  sorry

end NUMINAMATH_CALUDE_saheed_kayla_earnings_ratio_l3136_313661


namespace NUMINAMATH_CALUDE_negative_a_power_five_l3136_313687

theorem negative_a_power_five (a : ℝ) : (-a)^3 * (-a)^2 = -a^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_power_five_l3136_313687


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3136_313621

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  ¬((a - 1) * (a - 2) = 0 → a = 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3136_313621


namespace NUMINAMATH_CALUDE_function_monotonicity_l3136_313612

theorem function_monotonicity (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π) 
  (f : Real → Real) 
  (hf : ∀ x, f x = Real.sqrt 3 * Real.sin (2 * x + θ) + Real.cos (2 * x + θ))
  (h3 : f (π / 2) = 0) :
  StrictMonoOn f (Set.Ioo (π / 4) (3 * π / 4)) := by
sorry

end NUMINAMATH_CALUDE_function_monotonicity_l3136_313612


namespace NUMINAMATH_CALUDE_exists_class_with_at_least_35_students_l3136_313611

/-- Proves that in a school with 33 classes and 1150 students, there exists at least one class with 35 or more students. -/
theorem exists_class_with_at_least_35_students 
  (num_classes : ℕ) 
  (total_students : ℕ) 
  (h1 : num_classes = 33) 
  (h2 : total_students = 1150) : 
  ∃ (class_size : ℕ), class_size ≥ 35 ∧ class_size ≤ total_students := by
  sorry

#check exists_class_with_at_least_35_students

end NUMINAMATH_CALUDE_exists_class_with_at_least_35_students_l3136_313611


namespace NUMINAMATH_CALUDE_rectangle_minimum_width_l3136_313609

/-- A rectangle with length 1.5 times its width and area at least 450 square feet has a minimum width of 10√3 feet. -/
theorem rectangle_minimum_width (w : ℝ) (h_positive : w > 0) : 
  1.5 * w * w ≥ 450 → w ≥ 10 * Real.sqrt 3 :=
by
  sorry

#check rectangle_minimum_width

end NUMINAMATH_CALUDE_rectangle_minimum_width_l3136_313609


namespace NUMINAMATH_CALUDE_function_inequality_implies_bound_l3136_313640

theorem function_inequality_implies_bound (a : ℝ) : 
  (∃ x : ℝ, 4 - x^2 ≥ |x - a| + a) → a ≤ 17/8 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_bound_l3136_313640


namespace NUMINAMATH_CALUDE_prime_representation_l3136_313697

theorem prime_representation (k : ℕ) (h : k ∈ Finset.range 7 \ {0}) :
  (∀ p : ℕ, Prime p → 
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ p^2 = a^2 + k*b^2) → 
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ p = x^2 + k*y^2)) ↔ 
  k ∈ ({1, 2, 4} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_prime_representation_l3136_313697


namespace NUMINAMATH_CALUDE_yogurt_combinations_l3136_313635

theorem yogurt_combinations (flavors : ℕ) (toppings : ℕ) (sizes : ℕ) :
  flavors = 5 → toppings = 8 → sizes = 3 →
  flavors * (toppings.choose 2) * sizes = 420 :=
by sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l3136_313635


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l3136_313624

theorem average_of_eleven_numbers
  (first_six_avg : Real)
  (last_six_avg : Real)
  (sixth_number : Real)
  (h1 : first_six_avg = 58)
  (h2 : last_six_avg = 65)
  (h3 : sixth_number = 78) :
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l3136_313624


namespace NUMINAMATH_CALUDE_shaded_area_square_minus_circles_l3136_313696

/-- The shaded area of a square with side length 10 and four circles of radius 3√2 at its vertices -/
theorem shaded_area_square_minus_circles :
  let square_side : ℝ := 10
  let circle_radius : ℝ := 3 * Real.sqrt 2
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let total_circles_area : ℝ := 4 * circle_area
  let shaded_area : ℝ := square_area - total_circles_area
  shaded_area = 100 - 72 * π := by
  sorry

#check shaded_area_square_minus_circles

end NUMINAMATH_CALUDE_shaded_area_square_minus_circles_l3136_313696


namespace NUMINAMATH_CALUDE_only_set_B_forms_triangle_l3136_313618

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem only_set_B_forms_triangle :
  ¬(can_form_triangle 1 2 3) ∧
  can_form_triangle 2 3 4 ∧
  ¬(can_form_triangle 3 4 9) ∧
  ¬(can_form_triangle 2 2 4) :=
sorry

end NUMINAMATH_CALUDE_only_set_B_forms_triangle_l3136_313618


namespace NUMINAMATH_CALUDE_function_property_l3136_313686

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - x + 2

theorem function_property (a : ℝ) (h : a ≠ 0) :
  (∀ x₁ ∈ Set.Icc 1 (exp 1), ∃ x₂ ∈ Set.Icc 1 (exp 1), f a x₁ + f a x₂ = 4) →
  a = exp 1 + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3136_313686


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3136_313620

/-- Given a triangle ABC with side lengths a, b, and c, if a^2 + b^2 - c^2 = ab, 
    then the measure of angle C is 60°. -/
theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
    (h_eq : a^2 + b^2 - c^2 = a * b) : 
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3136_313620


namespace NUMINAMATH_CALUDE_car_distance_theorem_l3136_313603

/-- Given a car traveling at 160 km/h for 5 hours, the distance covered is 800 km. -/
theorem car_distance_theorem (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 160 ∧ time = 5 → distance = speed * time → distance = 800 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l3136_313603


namespace NUMINAMATH_CALUDE_sum_of_roots_l3136_313665

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

-- State the theorem
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3136_313665


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3136_313668

theorem stratified_sampling_theorem (total_population : ℕ) (sample_size : ℕ) (stratum_size : ℕ) 
  (h1 : total_population = 500) 
  (h2 : sample_size = 100) 
  (h3 : stratum_size = 95) :
  (stratum_size : ℚ) / total_population * sample_size = 19 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3136_313668


namespace NUMINAMATH_CALUDE_matching_socks_probability_l3136_313673

/-- The number of gray-bottomed socks -/
def gray_socks : ℕ := 12

/-- The number of white-bottomed socks -/
def white_socks : ℕ := 10

/-- The total number of socks -/
def total_socks : ℕ := gray_socks + white_socks

/-- The number of ways to choose 2 socks from n socks -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of selecting a matching pair of socks -/
theorem matching_socks_probability :
  (choose_two gray_socks + choose_two white_socks : ℚ) / choose_two total_socks = 111 / 231 := by
  sorry

end NUMINAMATH_CALUDE_matching_socks_probability_l3136_313673


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3136_313613

theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (2, 1) →
  b = (1, m) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3136_313613


namespace NUMINAMATH_CALUDE_x_minus_p_in_terms_of_p_l3136_313677

theorem x_minus_p_in_terms_of_p (x p : ℝ) : 
  (|x - 3| = p + 1) → (x < 3) → (x - p = 2 - 2*p) := by
  sorry

end NUMINAMATH_CALUDE_x_minus_p_in_terms_of_p_l3136_313677


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3136_313663

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum : ℕ → ℝ
  sum_def : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  term_def : ∀ n, a n = a 1 + (n - 1) * d

/-- Theorem: For an arithmetic sequence where S_5 = 3(a_2 + a_8), a_5 / a_3 = 5/6 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.sum 5 = 3 * (seq.a 2 + seq.a 8)) :
  seq.a 5 / seq.a 3 = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3136_313663


namespace NUMINAMATH_CALUDE_large_pizza_slices_correct_l3136_313691

/-- The number of slices a small pizza gives -/
def small_pizza_slices : ℕ := 4

/-- The number of small pizzas purchased -/
def small_pizzas_bought : ℕ := 3

/-- The number of large pizzas purchased -/
def large_pizzas_bought : ℕ := 2

/-- The number of slices George eats -/
def george_slices : ℕ := 3

/-- The number of slices Bob eats -/
def bob_slices : ℕ := george_slices + 1

/-- The number of slices Susie eats -/
def susie_slices : ℕ := bob_slices / 2

/-- The number of slices Bill, Fred, and Mark each eat -/
def others_slices : ℕ := 3

/-- The number of slices left over -/
def leftover_slices : ℕ := 10

/-- The number of slices a large pizza gives -/
def large_pizza_slices : ℕ := 8

theorem large_pizza_slices_correct : 
  small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought = 
  george_slices + bob_slices + susie_slices + 3 * others_slices + leftover_slices :=
by sorry

end NUMINAMATH_CALUDE_large_pizza_slices_correct_l3136_313691


namespace NUMINAMATH_CALUDE_ternary_121_equals_16_l3136_313632

/-- Converts a ternary number represented as a list of digits to its decimal equivalent -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The ternary representation of the number -/
def ternary_121 : List Nat := [1, 2, 1]

theorem ternary_121_equals_16 :
  ternary_to_decimal ternary_121 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_equals_16_l3136_313632


namespace NUMINAMATH_CALUDE_wall_height_proof_l3136_313657

/-- Given a wall and a painting, proves that the wall height is 5 feet -/
theorem wall_height_proof (wall_width painting_width painting_height painting_area_percentage : ℝ) :
  wall_width = 10 ∧ 
  painting_width = 2 ∧ 
  painting_height = 4 ∧ 
  painting_area_percentage = 0.16 ∧
  painting_width * painting_height = painting_area_percentage * (wall_width * (wall_width * painting_height / (painting_width * painting_height))) →
  wall_width * painting_height / (painting_width * painting_height) = 5 := by
sorry

end NUMINAMATH_CALUDE_wall_height_proof_l3136_313657


namespace NUMINAMATH_CALUDE_vasya_incorrect_l3136_313695

theorem vasya_incorrect : ¬∃ (x y : ℤ), (x + y = 2021) ∧ ((10 * x + y = 2221) ∨ (x + 10 * y = 2221)) := by
  sorry

end NUMINAMATH_CALUDE_vasya_incorrect_l3136_313695


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_arithmetic_sequence_l3136_313651

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

/-- The 10th term of the arithmetic sequence with first term 10 and common difference -2 is -8 -/
theorem tenth_term_of_specific_arithmetic_sequence :
  arithmeticSequence 10 (-2) 10 = -8 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_arithmetic_sequence_l3136_313651


namespace NUMINAMATH_CALUDE_cube_root_bound_l3136_313615

theorem cube_root_bound (n : ℕ) (hn : n ≥ 2) :
  (n : ℝ) + 0.6 < (((n : ℝ)^3 + 2*(n : ℝ)^2 + (n : ℝ))^(1/3 : ℝ)) ∧
  (((n : ℝ)^3 + 2*(n : ℝ)^2 + (n : ℝ))^(1/3 : ℝ)) < (n : ℝ) + 0.7 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_bound_l3136_313615


namespace NUMINAMATH_CALUDE_original_triangle_area_l3136_313674

theorem original_triangle_area
  (original : Real)  -- Area of the original triangle
  (new : Real)       -- Area of the new triangle
  (h1 : new = 256)   -- The area of the new triangle is 256 square feet
  (h2 : new = 16 * original)  -- The new triangle's area is 16 times the original
  : original = 16 :=
by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3136_313674


namespace NUMINAMATH_CALUDE_initial_men_count_initial_men_count_correct_l3136_313608

/-- The number of days the initial food supply lasts for the initial group -/
def initial_days : ℕ := 22

/-- The number of days that pass before new men join -/
def days_before_joining : ℕ := 2

/-- The number of new men that join -/
def new_men : ℕ := 1140

/-- The number of additional days the food lasts after new men join -/
def additional_days : ℕ := 8

/-- Proves that the initial number of men is 760 -/
theorem initial_men_count : ℕ :=
  760

/-- Theorem stating that the initial_men_count satisfies the given conditions -/
theorem initial_men_count_correct :
  initial_men_count * initial_days =
  (initial_men_count + new_men) * additional_days +
  initial_men_count * days_before_joining :=
by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_initial_men_count_correct_l3136_313608


namespace NUMINAMATH_CALUDE_daily_wage_calculation_l3136_313637

/-- Proves the daily wage for a worker given total days, idle days, total pay, and idle day deduction --/
theorem daily_wage_calculation (total_days idle_days : ℕ) (total_pay idle_day_deduction : ℚ) :
  total_days = 60 →
  idle_days = 40 →
  total_pay = 280 →
  idle_day_deduction = 3 →
  ∃ (daily_wage : ℚ), 
    daily_wage * (total_days - idle_days : ℚ) - idle_day_deduction * idle_days = total_pay ∧
    daily_wage = 20 := by
  sorry

end NUMINAMATH_CALUDE_daily_wage_calculation_l3136_313637


namespace NUMINAMATH_CALUDE_opposite_seven_eighteen_implies_twentytwo_l3136_313617

/-- Represents a circular arrangement of people -/
structure CircularArrangement where
  total : ℕ
  is_valid : total > 0

/-- Defines the property of two positions being opposite in a circular arrangement -/
def are_opposite (c : CircularArrangement) (p1 p2 : ℕ) : Prop :=
  p1 ≤ c.total ∧ p2 ≤ c.total ∧ (2 * p1 - 1) % c.total = (2 * p2 - 1) % c.total

/-- Theorem: In a circular arrangement where the 7th person is opposite the 18th, there are 22 people -/
theorem opposite_seven_eighteen_implies_twentytwo :
  ∀ c : CircularArrangement, are_opposite c 7 18 → c.total = 22 :=
sorry

end NUMINAMATH_CALUDE_opposite_seven_eighteen_implies_twentytwo_l3136_313617


namespace NUMINAMATH_CALUDE_count_distinct_lines_l3136_313602

def S : Set ℕ := {0, 1, 2, 3}

def is_valid_line (a b : ℕ) : Prop := a ∈ S ∧ b ∈ S

def distinct_lines : ℕ := sorry

theorem count_distinct_lines :
  distinct_lines = 9 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_lines_l3136_313602


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l3136_313682

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) 
  (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0) (h_inv : ∃ k, k ≠ 0 ∧ ∀ x y, x * y = k) 
  (h_ratio : x₁ / x₂ = 3 / 5) : 
  y₁ / y₂ = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l3136_313682


namespace NUMINAMATH_CALUDE_circle_equation_to_standard_form_1_circle_equation_to_standard_form_2_l3136_313694

/-- Proves that the given circle equation is equivalent to its standard form -/
theorem circle_equation_to_standard_form_1 (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y - 3 = 0 ↔ (x - 2)^2 + (y + 3)^2 = 16 := by
  sorry

/-- Proves that the given circle equation is equivalent to its standard form -/
theorem circle_equation_to_standard_form_2 (x y : ℝ) :
  4*x^2 + 4*y^2 - 8*x + 4*y - 11 = 0 ↔ (x - 1)^2 + (y + 1/2)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_to_standard_form_1_circle_equation_to_standard_form_2_l3136_313694


namespace NUMINAMATH_CALUDE_final_selling_price_l3136_313610

/-- The final selling price of a commodity after markup and reduction -/
theorem final_selling_price (a : ℝ) : 
  let initial_markup : ℝ := 1.25
  let price_reduction : ℝ := 0.9
  a * initial_markup * price_reduction = 1.125 * a := by sorry

end NUMINAMATH_CALUDE_final_selling_price_l3136_313610


namespace NUMINAMATH_CALUDE_final_position_l3136_313693

/-- A point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- Reflect a point about the origin -/
def reflectOrigin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- The theorem stating the final position of the point after translation and reflection -/
theorem final_position :
  let initial := Point.mk 3 2
  let translated := translateRight initial 2
  let final := reflectOrigin translated
  final = Point.mk (-5) (-2) := by sorry

end NUMINAMATH_CALUDE_final_position_l3136_313693


namespace NUMINAMATH_CALUDE_equal_principal_repayment_formula_l3136_313675

/-- Repayment amount for the nth month -/
def repayment_amount (n : ℕ) : ℚ :=
  3928 - 8 * n

/-- Properties of the loan -/
def loan_amount : ℚ := 480000
def repayment_years : ℕ := 20
def monthly_interest_rate : ℚ := 4 / 1000

theorem equal_principal_repayment_formula :
  ∀ n : ℕ, n > 0 → n ≤ repayment_years * 12 →
  repayment_amount n =
    loan_amount / (repayment_years * 12) +
    (loan_amount - (n - 1) * (loan_amount / (repayment_years * 12))) * monthly_interest_rate :=
by sorry

end NUMINAMATH_CALUDE_equal_principal_repayment_formula_l3136_313675


namespace NUMINAMATH_CALUDE_x_complements_c_l3136_313680

/-- Represents a date in a month --/
structure Date :=
  (value : ℕ)
  (h : value > 0 ∧ value ≤ 31)

/-- Represents letters on the calendar --/
inductive Letter
| A | B | C | X

/-- A calendar is a function that assigns a date to each letter --/
def Calendar := Letter → Date

/-- The condition that B is two weeks after A --/
def twoWeeksAfter (cal : Calendar) : Prop :=
  (cal Letter.B).value = (cal Letter.A).value + 14

/-- The condition that the sum of dates behind C and X equals the sum of dates behind A and B --/
def sumEqual (cal : Calendar) : Prop :=
  (cal Letter.C).value + (cal Letter.X).value = (cal Letter.A).value + (cal Letter.B).value

/-- The main theorem --/
theorem x_complements_c (cal : Calendar) 
  (h1 : twoWeeksAfter cal) 
  (h2 : sumEqual cal) : 
  (cal Letter.X).value = (cal Letter.C).value + 18 :=
sorry

end NUMINAMATH_CALUDE_x_complements_c_l3136_313680


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3136_313631

theorem point_on_x_axis (m : ℝ) :
  (m + 5, 2 * m + 8) = (1, 0) ↔ (m + 5, 2 * m + 8).2 = 0 := by
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3136_313631


namespace NUMINAMATH_CALUDE_negation_of_implication_l3136_313690

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → 2*a > 2*b) ↔ (a ≤ b → 2*a ≤ 2*b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3136_313690


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3136_313639

theorem cubic_equation_solution :
  ∃ x : ℝ, x^3 + 3*x^2 + 3*x + 7 = 0 ∧ x = -1 - Real.rpow 6 (1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3136_313639


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l3136_313626

def johnsons_share : ℕ := 2500
def mikes_shirt_cost : ℕ := 200
def mikes_remaining : ℕ := 800

def mikes_share : ℕ := mikes_remaining + mikes_shirt_cost

def ratio_numerator : ℕ := 2
def ratio_denominator : ℕ := 5

theorem profit_sharing_ratio :
  (mikes_share : ℚ) / johnsons_share = ratio_numerator / ratio_denominator :=
by sorry

end NUMINAMATH_CALUDE_profit_sharing_ratio_l3136_313626


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3136_313652

/-- Given a geometric sequence {a_n} where 3a_1, (1/2)a_5, and 2a_3 form an arithmetic sequence,
    prove that (a_9 + a_10) / (a_7 + a_8) = 3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = q * a n) 
    (h2 : (1/2) * a 5 = (3 * a 1 + 2 * a 3) / 2) :
  (a 9 + a 10) / (a 7 + a 8) = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3136_313652


namespace NUMINAMATH_CALUDE_roots_equation_value_l3136_313646

theorem roots_equation_value (α β : ℝ) :
  α^2 - 3*α - 2 = 0 →
  β^2 - 3*β - 2 = 0 →
  5 * α^4 + 12 * β^3 = 672.5 + 31.5 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_value_l3136_313646


namespace NUMINAMATH_CALUDE_random_event_identification_l3136_313662

-- Define the three events
def event1 : Prop := ∃ x y : ℝ, x * y < 0 ∧ x + y < 0
def event2 : Prop := ∀ x y : ℝ, x * y < 0 → x * y > 0
def event3 : Prop := ∀ x y : ℝ, x * y < 0 → x / y < 0

-- Define what it means for an event to be certain
def is_certain (e : Prop) : Prop := e ∨ ¬e

-- Theorem stating that event1 is not certain, while event2 and event3 are certain
theorem random_event_identification :
  ¬(is_certain event1) ∧ (is_certain event2) ∧ (is_certain event3) :=
sorry

end NUMINAMATH_CALUDE_random_event_identification_l3136_313662


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_bound_l3136_313669

-- Define the function f(x) = -x^2 + 2ax + 3
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 3

-- Define what it means for a function to be decreasing on an interval
def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem decreasing_quadratic_implies_a_bound :
  ∀ a : ℝ, decreasing_on (f a) 2 6 → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_bound_l3136_313669


namespace NUMINAMATH_CALUDE_two_integers_sum_l3136_313688

theorem two_integers_sum (x y : ℕ+) : 
  (x : ℤ) - (y : ℤ) = 5 ∧ 
  (x : ℕ) * y = 180 → 
  (x : ℕ) + y = 25 := by
sorry

end NUMINAMATH_CALUDE_two_integers_sum_l3136_313688


namespace NUMINAMATH_CALUDE_translation_result_l3136_313692

def translate_point (x y dx dy : ℝ) : ℝ × ℝ :=
  (x + dx, y + dy)

theorem translation_result :
  let P : ℝ × ℝ := (-2, 1)
  let dx : ℝ := 3
  let dy : ℝ := 4
  let P' : ℝ × ℝ := translate_point P.1 P.2 dx dy
  P' = (1, 5) := by sorry

end NUMINAMATH_CALUDE_translation_result_l3136_313692


namespace NUMINAMATH_CALUDE_extreme_value_of_f_l3136_313648

-- Define the function
def f (x : ℝ) : ℝ := (x^2 - 1)^3 + 1

-- State the theorem
theorem extreme_value_of_f :
  ∃ (e : ℝ), e = 0 ∧ ∀ (x : ℝ), f x ≥ e :=
sorry

end NUMINAMATH_CALUDE_extreme_value_of_f_l3136_313648


namespace NUMINAMATH_CALUDE_unique_injective_function_l3136_313604

/-- Iterate a function n times -/
def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (iterate f n x)

/-- The property that f must satisfy -/
def satisfies_equation (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, iterate f (f a) b * iterate f (f b) a = (f (a + b))^2

/-- The main theorem statement -/
theorem unique_injective_function :
  ∀ f : ℕ → ℕ, Function.Injective f → satisfies_equation f → ∀ x : ℕ, f x = x + 1 := by
  sorry


end NUMINAMATH_CALUDE_unique_injective_function_l3136_313604


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3136_313676

-- Define atomic weights
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cr : ℝ := 51.996
def atomic_weight_O : ℝ := 15.999
def atomic_weight_N : ℝ := 14.007
def atomic_weight_C : ℝ := 12.011

-- Define number of atoms for each element
def num_H : ℕ := 4
def num_Cr : ℕ := 2
def num_O : ℕ := 4
def num_N : ℕ := 3
def num_C : ℕ := 5

-- Define the molecular weight calculation function
def molecular_weight : ℝ :=
  (num_H : ℝ) * atomic_weight_H +
  (num_Cr : ℝ) * atomic_weight_Cr +
  (num_O : ℝ) * atomic_weight_O +
  (num_N : ℝ) * atomic_weight_N +
  (num_C : ℝ) * atomic_weight_C

-- Theorem statement
theorem compound_molecular_weight :
  molecular_weight = 274.096 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3136_313676


namespace NUMINAMATH_CALUDE_given_square_is_magic_l3136_313601

/-- Represents a 3x3 magic square -/
def MagicSquare : Type := Fin 3 → Fin 3 → ℕ

/-- Converts a number from base 5 to base 10 -/
def toBase10 (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 5 => 5
  | 6 => 6
  | 10 => 5
  | 20 => 10
  | 21 => 11
  | 22 => 12
  | 23 => 13
  | _ => n  -- Default case

/-- The given magic square -/
def givenSquare : MagicSquare :=
  fun i j => match i, j with
    | 0, 0 => 22
    | 0, 1 => 2
    | 0, 2 => 20
    | 1, 0 => 5
    | 1, 1 => 10
    | 1, 2 => 21
    | 2, 0 => 6
    | 2, 1 => 23
    | 2, 2 => 3

/-- Sum of a row in the magic square -/
def rowSum (s : MagicSquare) (i : Fin 3) : ℕ :=
  (toBase10 (s i 0)) + (toBase10 (s i 1)) + (toBase10 (s i 2))

/-- Sum of a column in the magic square -/
def colSum (s : MagicSquare) (j : Fin 3) : ℕ :=
  (toBase10 (s 0 j)) + (toBase10 (s 1 j)) + (toBase10 (s 2 j))

/-- Sum of the main diagonal of the magic square -/
def mainDiagSum (s : MagicSquare) : ℕ :=
  (toBase10 (s 0 0)) + (toBase10 (s 1 1)) + (toBase10 (s 2 2))

/-- Sum of the other diagonal of the magic square -/
def otherDiagSum (s : MagicSquare) : ℕ :=
  (toBase10 (s 0 2)) + (toBase10 (s 1 1)) + (toBase10 (s 2 0))

/-- Theorem: The given square is a magic square when interpreted in base 5 -/
theorem given_square_is_magic : 
  (∀ i : Fin 3, rowSum givenSquare i = 21) ∧ 
  (∀ j : Fin 3, colSum givenSquare j = 21) ∧ 
  mainDiagSum givenSquare = 21 ∧ 
  otherDiagSum givenSquare = 21 := by
  sorry

end NUMINAMATH_CALUDE_given_square_is_magic_l3136_313601


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3136_313649

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    (4 * x + 8) / (x^2 - 3*x - 18) = 32 / (9 * (x - 6)) + 4 / (9 * (x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3136_313649


namespace NUMINAMATH_CALUDE_consecutive_sum_inequality_l3136_313664

theorem consecutive_sum_inequality (nums : Fin 100 → ℝ) 
  (h_distinct : ∀ i j : Fin 100, i ≠ j → nums i ≠ nums j) :
  ∃ i : Fin 100, nums i + nums ((i + 3) % 100) > nums ((i + 1) % 100) + nums ((i + 2) % 100) :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_inequality_l3136_313664


namespace NUMINAMATH_CALUDE_cube_split_theorem_l3136_313672

def is_split_number (m : ℕ) (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2 * k + 1 ∧ 
  ∃ (start : ℕ), (Finset.range m).sum (λ i => 2 * (start + i) + 1) = m^3 ∧
  ∃ (i : Fin m), n = 2 * (start + i) + 1

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) (h2 : is_split_number m 333) : m = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_theorem_l3136_313672


namespace NUMINAMATH_CALUDE_balls_color_probability_l3136_313634

def num_balls : ℕ := 6
def probability_black : ℚ := 1/2
def probability_white : ℚ := 1/2

theorem balls_color_probability :
  let favorable_outcomes := (num_balls.choose (num_balls / 2))
  let total_outcomes := 2^num_balls
  (favorable_outcomes : ℚ) / total_outcomes = 5/16 := by
sorry

end NUMINAMATH_CALUDE_balls_color_probability_l3136_313634


namespace NUMINAMATH_CALUDE_f_min_value_l3136_313667

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 4 * y^2 - 8 * x + y

/-- The minimum value of the function f -/
def min_value : ℝ := 3.7391

theorem f_min_value :
  ∀ x y : ℝ, f x y ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l3136_313667


namespace NUMINAMATH_CALUDE_hundredth_term_of_sequence_l3136_313698

def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem hundredth_term_of_sequence (a₁ d : ℕ) (h₁ : a₁ = 5) (h₂ : d = 4) :
  arithmeticSequence a₁ d 100 = 401 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_of_sequence_l3136_313698


namespace NUMINAMATH_CALUDE_power_equality_l3136_313605

theorem power_equality : 32^4 * 4^5 = 2^30 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3136_313605


namespace NUMINAMATH_CALUDE_binomial_sum_first_six_l3136_313666

theorem binomial_sum_first_six (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x + 1)^11 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1024 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_first_six_l3136_313666


namespace NUMINAMATH_CALUDE_B_subset_M_M_closed_under_mult_l3136_313654

-- Define the set M
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

-- Define the set B
def B : Set ℤ := {b | ∃ n : ℕ, b = 2*n + 1}

-- Theorem 1: B is a subset of M
theorem B_subset_M : B ⊆ M := by sorry

-- Theorem 2: M is closed under multiplication
theorem M_closed_under_mult : ∀ a₁ a₂ : ℤ, a₁ ∈ M → a₂ ∈ M → (a₁ * a₂) ∈ M := by sorry

end NUMINAMATH_CALUDE_B_subset_M_M_closed_under_mult_l3136_313654


namespace NUMINAMATH_CALUDE_smallest_y_value_l3136_313638

theorem smallest_y_value (y : ℝ) (h : y > 0) :
  (y / 7 + 2 / (7 * y) = 1 / 3) → y ≥ 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_y_value_l3136_313638


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l3136_313659

/-- Given a rectangular room with length and width, and the total cost of paving,
    calculate the rate of paving per square meter. -/
theorem paving_rate_calculation 
  (length width total_cost : ℝ) 
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) : 
  total_cost / (length * width) = 800 := by
  sorry

end NUMINAMATH_CALUDE_paving_rate_calculation_l3136_313659


namespace NUMINAMATH_CALUDE_range_of_a_l3136_313622

theorem range_of_a (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 5 ∧ ∀ x ∈ s, (1 + a ≤ x ∧ x < 2)) → 
  (-5 < a ∧ a ≤ -4) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3136_313622


namespace NUMINAMATH_CALUDE_smallest_value_S_l3136_313625

theorem smallest_value_S (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ d₁ d₂ d₃ : ℕ) : 
  ({a₁, a₂, a₃, b₁, b₂, b₃, c₁, c₂, c₃, d₁, d₂, d₃} = Finset.range 12) →
  (a₁ * a₂ * a₃ + b₁ * b₂ * b₃ + c₁ * c₂ * c₃ + d₁ * d₂ * d₃ ≥ 613) ∧
  (∃ (a₁' a₂' a₃' b₁' b₂' b₃' c₁' c₂' c₃' d₁' d₂' d₃' : ℕ),
    {a₁', a₂', a₃', b₁', b₂', b₃', c₁', c₂', c₃', d₁', d₂', d₃'} = Finset.range 12 ∧
    a₁' * a₂' * a₃' + b₁' * b₂' * b₃' + c₁' * c₂' * c₃' + d₁' * d₂' * d₃' = 613) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_S_l3136_313625


namespace NUMINAMATH_CALUDE_perpendicular_line_to_parallel_plane_l3136_313629

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the non-coincidence property
variable (non_coincident_planes : Plane → Plane → Prop)
variable (non_coincident_lines : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_line_to_parallel_plane
  (α β : Plane) (m n : Line)
  (h_non_coincident_planes : non_coincident_planes α β)
  (h_non_coincident_lines : non_coincident_lines m n)
  (h_parallel_lines : parallel_lines m n)
  (h_perp_n_α : perpendicular_line_plane n α)
  (h_parallel_planes : parallel_planes α β) :
  perpendicular_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_parallel_plane_l3136_313629


namespace NUMINAMATH_CALUDE_race_result_kilometer_race_result_l3136_313607

/-- Represents a runner in the race -/
structure Runner where
  time : ℝ  -- Time taken to complete the race in seconds
  distance : ℝ  -- Distance covered in meters

/-- The race scenario -/
def race_scenario (race_distance : ℝ) (a b : Runner) : Prop :=
  a.distance = race_distance ∧
  b.distance = race_distance ∧
  a.time + 10 = b.time ∧
  a.time = 390

/-- The theorem to be proved -/
theorem race_result (race_distance : ℝ) (a b : Runner) 
  (h : race_scenario race_distance a b) : 
  a.distance - b.distance * (a.time / b.time) = 25 := by
  sorry

/-- Main theorem stating the race result -/
theorem kilometer_race_result :
  ∃ (a b : Runner), race_scenario 1000 a b ∧ 
  a.distance - b.distance * (a.time / b.time) = 25 := by
  sorry

end NUMINAMATH_CALUDE_race_result_kilometer_race_result_l3136_313607


namespace NUMINAMATH_CALUDE_max_sum_of_products_l3136_313655

/-- The maximum sum of products for four distinct values from {3, 4, 5, 6} -/
theorem max_sum_of_products : 
  ∀ (f g h j : ℕ), 
    f ∈ ({3, 4, 5, 6} : Set ℕ) → 
    g ∈ ({3, 4, 5, 6} : Set ℕ) → 
    h ∈ ({3, 4, 5, 6} : Set ℕ) → 
    j ∈ ({3, 4, 5, 6} : Set ℕ) → 
    f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j → 
    f * g + g * h + h * j + j * f ≤ 80 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_products_l3136_313655


namespace NUMINAMATH_CALUDE_three_intersections_implies_a_value_l3136_313614

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then Real.sin x else x^3 - 9*x^2 + 25*x + a

theorem three_intersections_implies_a_value (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = x₁ ∧ f a x₂ = x₂ ∧ f a x₃ = x₃) →
  a = -20 ∨ a = -16 := by
  sorry

end NUMINAMATH_CALUDE_three_intersections_implies_a_value_l3136_313614


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3136_313642

-- Problem 1
theorem problem_1 (x : ℚ) : 
  16 * (6*x - 1) * (2*x - 1) * (3*x + 1) * (x - 1) + 25 = (24*x^2 - 16*x - 3)^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℚ) : 
  (6*x - 1) * (2*x - 1) * (3*x - 1) * (x - 1) + x^2 = (6*x^2 - 6*x + 1)^2 := by sorry

-- Problem 3
theorem problem_3 (x : ℚ) : 
  (6*x - 1) * (4*x - 1) * (3*x - 1) * (x - 1) + 9*x^4 = (9*x^2 - 7*x + 1)^2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3136_313642


namespace NUMINAMATH_CALUDE_volume_of_extended_box_l3136_313623

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def extendedVolume (b : Box) : ℝ :=
  sorry

/-- The specific box in the problem -/
def problemBox : Box :=
  { length := 4, width := 5, height := 6 }

/-- The theorem to be proved -/
theorem volume_of_extended_box :
  extendedVolume problemBox = (804 + 139 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_extended_box_l3136_313623


namespace NUMINAMATH_CALUDE_w_to_twelve_power_l3136_313660

theorem w_to_twelve_power (w : ℂ) (h : w = (-Real.sqrt 3 + Complex.I) / 3) :
  w^12 = 400 / 531441 := by
  sorry

end NUMINAMATH_CALUDE_w_to_twelve_power_l3136_313660


namespace NUMINAMATH_CALUDE_nursing_home_flowers_l3136_313606

/-- The number of flower sets bought by Mayor Harvey -/
def num_sets : ℕ := 3

/-- The number of flowers in each set -/
def flowers_per_set : ℕ := 90

/-- The total number of flowers bought for the nursing home -/
def total_flowers : ℕ := num_sets * flowers_per_set

theorem nursing_home_flowers : total_flowers = 270 := by
  sorry

end NUMINAMATH_CALUDE_nursing_home_flowers_l3136_313606


namespace NUMINAMATH_CALUDE_triangular_prism_skew_lines_l3136_313650

/-- A triangular prism -/
structure TriangularPrism where
  vertices : Finset (ℝ × ℝ × ℝ)
  edges : Finset (Finset (ℝ × ℝ × ℝ))
  is_valid : vertices.card = 6 ∧ edges.card = 9

/-- A line in 3D space -/
def Line3D := Finset (ℝ × ℝ × ℝ)

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop := sorry

/-- The set of all lines passing through any two vertices of the prism -/
def all_lines (p : TriangularPrism) : Finset Line3D := sorry

/-- The set of all pairs of skew lines in the prism -/
def skew_line_pairs (p : TriangularPrism) : Finset (Line3D × Line3D) := sorry

theorem triangular_prism_skew_lines (p : TriangularPrism) :
  (all_lines p).card = 15 → (skew_line_pairs p).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_skew_lines_l3136_313650


namespace NUMINAMATH_CALUDE_number_of_signups_l3136_313689

/-- The number of students --/
def num_students : ℕ := 5

/-- The number of sports competitions --/
def num_competitions : ℕ := 3

/-- Theorem: The number of ways for students to sign up for competitions --/
theorem number_of_signups : (num_competitions ^ num_students) = 243 := by
  sorry

end NUMINAMATH_CALUDE_number_of_signups_l3136_313689


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_63_with_digit_sum_63_l3136_313699

def digit_sum (n : ℕ) : ℕ := sorry

def is_divisible_by (n m : ℕ) : Prop := sorry

theorem smallest_number_divisible_by_63_with_digit_sum_63 :
  ∃ (n : ℕ),
    is_divisible_by n 63 ∧
    digit_sum n = 63 ∧
    (∀ m : ℕ, m < n → ¬(is_divisible_by m 63 ∧ digit_sum m = 63)) ∧
    n = 63999999 :=
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_63_with_digit_sum_63_l3136_313699


namespace NUMINAMATH_CALUDE_only_cylinder_has_quadrilateral_cross_section_l3136_313670

-- Define the types of solids
inductive Solid
| Cone
| Cylinder
| Sphere

-- Define a function that determines if a solid can have a quadrilateral cross-section
def has_quadrilateral_cross_section (s : Solid) : Prop :=
  match s with
  | Solid.Cone => False
  | Solid.Cylinder => True
  | Solid.Sphere => False

-- Theorem statement
theorem only_cylinder_has_quadrilateral_cross_section :
  ∀ s : Solid, has_quadrilateral_cross_section s ↔ s = Solid.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_only_cylinder_has_quadrilateral_cross_section_l3136_313670


namespace NUMINAMATH_CALUDE_convex_quad_probability_l3136_313644

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The probability of forming a convex quadrilateral -/
def prob_convex_quad : ℚ := (n.choose k : ℚ) / (total_chords.choose k : ℚ)

/-- Theorem stating the probability of forming a convex quadrilateral -/
theorem convex_quad_probability : prob_convex_quad = 2 / 585 := by
  sorry

end NUMINAMATH_CALUDE_convex_quad_probability_l3136_313644


namespace NUMINAMATH_CALUDE_arrangement_count_is_288_l3136_313683

/-- The number of ways to arrange 4 mathematics books and 4 history books with constraints -/
def arrangement_count : ℕ :=
  let math_books : ℕ := 4
  let history_books : ℕ := 4
  let block_arrangements : ℕ := 2  -- Math block and history block
  let math_internal_arrangements : ℕ := Nat.factorial (math_books - 1)  -- Excluding M1
  let history_internal_arrangements : ℕ := Nat.factorial history_books
  block_arrangements * math_internal_arrangements * history_internal_arrangements

/-- Theorem stating that the number of valid arrangements is 288 -/
theorem arrangement_count_is_288 : arrangement_count = 288 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_288_l3136_313683


namespace NUMINAMATH_CALUDE_total_sales_theorem_l3136_313671

/-- Calculate total sales from lettuce and tomatoes -/
def total_sales (customers : ℕ) (lettuce_per_customer : ℕ) (lettuce_price : ℚ) 
  (tomatoes_per_customer : ℕ) (tomato_price : ℚ) : ℚ :=
  (customers * lettuce_per_customer * lettuce_price) + 
  (customers * tomatoes_per_customer * tomato_price)

/-- Theorem: Total sales from lettuce and tomatoes is $2000 per month -/
theorem total_sales_theorem : 
  total_sales 500 2 1 4 (1/2) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_theorem_l3136_313671


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3136_313678

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- If 2S₃ = 3S₂ + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3136_313678


namespace NUMINAMATH_CALUDE_sqrt_n_squared_plus_n_bounds_l3136_313636

theorem sqrt_n_squared_plus_n_bounds (n : ℕ) :
  (n : ℝ) + 0.4 < Real.sqrt ((n : ℝ)^2 + n) ∧ Real.sqrt ((n : ℝ)^2 + n) < (n : ℝ) + 0.5 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_n_squared_plus_n_bounds_l3136_313636


namespace NUMINAMATH_CALUDE_not_in_first_quadrant_l3136_313627

/-- Proves that the complex number z = (m-2i)/(1+2i) cannot be in the first quadrant for any real m -/
theorem not_in_first_quadrant (m : ℝ) : 
  let z : ℂ := (m - 2*Complex.I) / (1 + 2*Complex.I)
  ¬ (z.re > 0 ∧ z.im > 0) := by
  sorry


end NUMINAMATH_CALUDE_not_in_first_quadrant_l3136_313627


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l3136_313643

-- Define the normal distribution parameters
def mean : ℝ := 17.5
def std_dev : ℝ := 2.5

-- Define the value we want to prove
def value : ℝ := 12.5

-- Theorem statement
theorem two_std_dev_below_mean : 
  mean - 2 * std_dev = value := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l3136_313643


namespace NUMINAMATH_CALUDE_prob_A_value_l3136_313656

/-- The probability that person A speaks the truth -/
def prob_A : ℝ := sorry

/-- The probability that person B speaks the truth -/
def prob_B : ℝ := 0.6

/-- The probability that both A and B speak the truth simultaneously -/
def prob_A_and_B : ℝ := 0.48

/-- The events of A and B speaking the truth are independent -/
axiom independence : prob_A_and_B = prob_A * prob_B

theorem prob_A_value : prob_A = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_value_l3136_313656


namespace NUMINAMATH_CALUDE_min_b_for_real_roots_F_monotonic_iff_l3136_313641

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - Real.log x

-- Define the function F
def F (a : ℝ) (x : ℝ) : ℝ := f a x * Real.exp (-x)

-- Theorem for part 1
theorem min_b_for_real_roots (x : ℝ) :
  ∃ (b : ℝ), b ≥ 0 ∧ ∃ (x : ℝ), x > 0 ∧ f (-1) x = b / x ∧
  ∀ (b' : ℝ), b' < b → ¬∃ (x : ℝ), x > 0 ∧ f (-1) x = b' / x :=
sorry

-- Theorem for part 2
theorem F_monotonic_iff (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → F a x₁ < F a x₂) ∨
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → F a x₁ > F a x₂) ↔
  a ≤ 2 :=
sorry

end

end NUMINAMATH_CALUDE_min_b_for_real_roots_F_monotonic_iff_l3136_313641


namespace NUMINAMATH_CALUDE_not_divisible_by_121_l3136_313630

theorem not_divisible_by_121 : ∀ n : ℤ, ¬(121 ∣ (n^2 + 2*n + 2014)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_121_l3136_313630


namespace NUMINAMATH_CALUDE_sphere_surface_area_increase_l3136_313633

theorem sphere_surface_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 1.1 * r
  let original_area := 4 * Real.pi * r^2
  let new_area := 4 * Real.pi * new_radius^2
  (new_area - original_area) / original_area = 0.21 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_increase_l3136_313633


namespace NUMINAMATH_CALUDE_verna_haley_weight_difference_l3136_313616

/-- Given the weights of Verna, Haley, and Sherry, prove that Verna weighs 17 pounds more than Haley -/
theorem verna_haley_weight_difference :
  ∀ (verna_weight haley_weight sherry_weight : ℕ),
    verna_weight > haley_weight →
    verna_weight = sherry_weight / 2 →
    haley_weight = 103 →
    verna_weight + sherry_weight = 360 →
    verna_weight - haley_weight = 17 := by
  sorry

end NUMINAMATH_CALUDE_verna_haley_weight_difference_l3136_313616


namespace NUMINAMATH_CALUDE_inverse_proportion_example_l3136_313681

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (p q : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, p x * q x = k

theorem inverse_proportion_example :
  ∀ p q : ℝ → ℝ,
  InverselyProportional p q →
  p 6 = 25 →
  p 15 = 10 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_example_l3136_313681


namespace NUMINAMATH_CALUDE_greatest_power_of_two_l3136_313658

theorem greatest_power_of_two (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (10^1006 - 6^503) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1006 - 6^503) → m ≤ k) → 
  (∃ k : ℕ, k = 503 ∧ 2^k ∣ (10^1006 - 6^503) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1006 - 6^503) → m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_l3136_313658


namespace NUMINAMATH_CALUDE_same_terminal_side_M_subset_N_l3136_313645

/-- Represents an angle in degrees -/
structure Angle :=
  (value : ℝ)

/-- Defines the terminal side of an angle -/
def terminalSide (a : Angle) : ℝ × ℝ := sorry

/-- Defines set M -/
def M : Set ℝ := {x | ∃ k : ℤ, x = 45 + k * 90}

/-- Defines set N -/
def N : Set ℝ := {y | ∃ k : ℤ, y = 90 + k * 45}

/-- Theorem stating that angles α and β have the same terminal side -/
theorem same_terminal_side (k : ℤ) :
  terminalSide (Angle.mk ((2 * k + 1) * 180)) = terminalSide (Angle.mk ((4 * k + 1) * 180)) ∧
  terminalSide (Angle.mk ((2 * k + 1) * 180)) = terminalSide (Angle.mk ((4 * k - 1) * 180)) :=
sorry

/-- Theorem stating that M is a subset of N -/
theorem M_subset_N : M ⊆ N :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_M_subset_N_l3136_313645


namespace NUMINAMATH_CALUDE_seventh_observation_l3136_313628

theorem seventh_observation (n : ℕ) (x : ℝ) (y : ℝ) :
  n = 6 →
  x = 16 →
  y = x - 1 →
  (n * x + 9) / (n + 1) = y →
  9 = 9 :=
by sorry

end NUMINAMATH_CALUDE_seventh_observation_l3136_313628
