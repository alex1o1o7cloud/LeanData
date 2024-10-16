import Mathlib

namespace NUMINAMATH_CALUDE_apple_purchase_difference_l1106_110688

theorem apple_purchase_difference : 
  ∀ (bonnie_apples samuel_apples : ℕ),
    bonnie_apples = 8 →
    samuel_apples > bonnie_apples →
    samuel_apples - (samuel_apples / 2) - (samuel_apples / 7) = 10 →
    samuel_apples - bonnie_apples = 20 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_difference_l1106_110688


namespace NUMINAMATH_CALUDE_fraction_equality_l1106_110646

theorem fraction_equality (a b : ℝ) (h : (1/a + 1/b)/(1/a - 1/b) = 1009) :
  (a + b)/(a - b) = -1009 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1106_110646


namespace NUMINAMATH_CALUDE_divisors_of_36_l1106_110618

theorem divisors_of_36 : 
  ∃ (divs : List Nat), 
    (∀ d, d ∈ divs ↔ d ∣ 36) ∧ 
    divs.length = 9 ∧
    divs = [1, 2, 3, 4, 6, 9, 12, 18, 36] :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_36_l1106_110618


namespace NUMINAMATH_CALUDE_fencing_requirement_l1106_110651

theorem fencing_requirement (length width : ℝ) (h1 : length = 30) (h2 : length * width = 810) :
  2 * width + length = 84 := by
  sorry

end NUMINAMATH_CALUDE_fencing_requirement_l1106_110651


namespace NUMINAMATH_CALUDE_chocolate_distribution_l1106_110695

/-- 
Given:
- Ingrid starts with n chocolates
- Jin receives 1/3 of Ingrid's chocolates
- Jin gives 8 chocolates to Brian
- Jin eats half of her remaining chocolates
- Jin ends up with 5 chocolates

Prove: n = 54
-/
theorem chocolate_distribution (n : ℕ) : 
  (n / 3 - 8) / 2 = 5 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l1106_110695


namespace NUMINAMATH_CALUDE_rope_around_cylinders_l1106_110611

theorem rope_around_cylinders (rope_length : ℝ) (r1 r2 : ℝ) (rounds1 : ℕ) :
  r1 = 14 →
  r2 = 20 →
  rounds1 = 70 →
  rope_length = 2 * π * r1 * (rounds1 : ℝ) →
  ∃ (rounds2 : ℕ), rounds2 = 49 ∧ rope_length = 2 * π * r2 * (rounds2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_rope_around_cylinders_l1106_110611


namespace NUMINAMATH_CALUDE_unique_solution_l1106_110691

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Define the equation
def equation (x : ℕ) : Prop :=
  combination x 3 + combination x 2 = 12 * (x - 1)

-- State the theorem
theorem unique_solution :
  ∃! x : ℕ, x ≥ 3 ∧ equation x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1106_110691


namespace NUMINAMATH_CALUDE_smallest_a_is_correct_l1106_110625

/-- A polynomial of the form x^3 - ax^2 + bx - 2310 with three positive integer roots -/
structure PolynomialWithThreeRoots where
  a : ℕ
  b : ℕ
  root1 : ℕ+
  root2 : ℕ+
  root3 : ℕ+
  is_root1 : (root1 : ℝ)^3 - a*(root1 : ℝ)^2 + b*(root1 : ℝ) - 2310 = 0
  is_root2 : (root2 : ℝ)^3 - a*(root2 : ℝ)^2 + b*(root2 : ℝ) - 2310 = 0
  is_root3 : (root3 : ℝ)^3 - a*(root3 : ℝ)^2 + b*(root3 : ℝ) - 2310 = 0

/-- The smallest possible value of a for a polynomial with three positive integer roots -/
def smallest_a (p : PolynomialWithThreeRoots) : ℕ := 78

theorem smallest_a_is_correct (p : PolynomialWithThreeRoots) :
  p.a ≥ smallest_a p :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_correct_l1106_110625


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_x_equals_one_l1106_110621

/-- Represents a parabola of the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The axis of symmetry of a parabola --/
def axisOfSymmetry (p : Parabola) : ℝ := p.h

/-- The given parabola y = -2(x-1)^2 + 3 --/
def givenParabola : Parabola := ⟨-2, 1, 3⟩

theorem axis_of_symmetry_is_x_equals_one :
  axisOfSymmetry givenParabola = 1 := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_is_x_equals_one_l1106_110621


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_power_l1106_110630

def last_two_digits (n : ℕ) : ℕ := n % 100

def periodic_sequence (a : ℕ → ℕ) (period : ℕ) :=
  ∀ n : ℕ, a (n + period) = a n

theorem last_two_digits_of_7_power (n : ℕ) (h : n ≥ 2) :
  periodic_sequence (λ k => last_two_digits (7^k)) 4 →
  last_two_digits (7^2017) = last_two_digits (7^5) :=
by
  sorry

#eval last_two_digits (7^5)  -- Should output 7

end NUMINAMATH_CALUDE_last_two_digits_of_7_power_l1106_110630


namespace NUMINAMATH_CALUDE_race_passengers_l1106_110676

/-- The number of cars in the race -/
def num_cars : ℕ := 20

/-- The total number of people in all cars at the end of the race -/
def total_people_end : ℕ := 80

/-- The number of passengers in each car at the beginning of the race -/
def initial_passengers : ℕ := 2

theorem race_passengers :
  (num_cars * (initial_passengers + 1)) + num_cars = total_people_end :=
by sorry

end NUMINAMATH_CALUDE_race_passengers_l1106_110676


namespace NUMINAMATH_CALUDE_monthly_savings_proof_l1106_110632

-- Define income tax rate
def income_tax_rate : ℚ := 13/100

-- Define salaries and pensions (in rubles)
def ivan_salary : ℕ := 55000
def vasilisa_salary : ℕ := 45000
def mother_salary : ℕ := 18000
def mother_pension : ℕ := 10000
def father_salary : ℕ := 20000
def son_state_scholarship : ℕ := 3000
def son_nonstate_scholarship : ℕ := 15000

-- Define monthly expenses (in rubles)
def monthly_expenses : ℕ := 74000

-- Function to calculate net income after tax
def net_income (gross_income : ℕ) : ℚ :=
  (gross_income : ℚ) * (1 - income_tax_rate)

-- Total monthly net income before 01.05.2018
def total_net_income_before_may : ℚ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  net_income mother_salary + net_income father_salary + son_state_scholarship

-- Total monthly net income from 01.05.2018 to 31.08.2018
def total_net_income_may_to_aug : ℚ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + son_state_scholarship

-- Total monthly net income from 01.09.2018 for 1 year
def total_net_income_from_sep : ℚ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + 
  son_state_scholarship + net_income son_nonstate_scholarship

-- Theorem to prove monthly savings for different periods
theorem monthly_savings_proof :
  (total_net_income_before_may - monthly_expenses = 49060) ∧
  (total_net_income_may_to_aug - monthly_expenses = 43400) ∧
  (total_net_income_from_sep - monthly_expenses = 56450) := by
  sorry


end NUMINAMATH_CALUDE_monthly_savings_proof_l1106_110632


namespace NUMINAMATH_CALUDE_crude_oil_temperature_l1106_110657

-- Define the function f(x) = x^2 - 7x + 15 on the interval [0, 8]
def f (x : ℝ) : ℝ := x^2 - 7*x + 15

-- Define the domain of f
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 8 }

theorem crude_oil_temperature (x : ℝ) (h : x ∈ domain) : 
  -- The derivative of f at x = 4 is 1
  (deriv f) 4 = 1 ∧ 
  -- The function is increasing at x = 4
  (deriv f) 4 > 0 := by
  sorry

end NUMINAMATH_CALUDE_crude_oil_temperature_l1106_110657


namespace NUMINAMATH_CALUDE_sqrt_16_l1106_110667

theorem sqrt_16 : {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_sqrt_16_l1106_110667


namespace NUMINAMATH_CALUDE_bella_steps_to_meet_l1106_110645

/-- The number of steps Bella takes before meeting Ella -/
def steps_to_meet (total_distance : ℕ) (bella_step_length : ℕ) (ella_speed_multiplier : ℕ) : ℕ :=
  let distance_to_meet := total_distance / 2
  let bella_speed := 1
  let ella_speed := ella_speed_multiplier * bella_speed
  let combined_speed := bella_speed + ella_speed
  let distance_bella_walks := (distance_to_meet * bella_speed) / combined_speed
  distance_bella_walks / bella_step_length

/-- Theorem stating that Bella takes 528 steps before meeting Ella -/
theorem bella_steps_to_meet :
  steps_to_meet 15840 3 4 = 528 := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_to_meet_l1106_110645


namespace NUMINAMATH_CALUDE_probability_all_heads_or_some_tails_l1106_110670

def num_coins : ℕ := 5

def coin_outcomes : ℕ := 2

def all_outcomes : ℕ := coin_outcomes ^ num_coins

theorem probability_all_heads_or_some_tails :
  (1 : ℚ) / all_outcomes + ((all_outcomes - 1 : ℕ) : ℚ) / all_outcomes = 1 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_heads_or_some_tails_l1106_110670


namespace NUMINAMATH_CALUDE_sin_15_cos_15_l1106_110693

theorem sin_15_cos_15 : 
  (∀ θ : ℝ, Real.sin (2 * θ) = 2 * Real.sin θ * Real.cos θ) →
  Real.sin (30 * π / 180) = 1 / 2 →
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_l1106_110693


namespace NUMINAMATH_CALUDE_suma_work_time_l1106_110675

/-- Given that Renu can do a piece of work in 8 days and that Renu and Suma together can do it in 3 days,
    this theorem proves that Suma can do the work alone in 4.8 days. -/
theorem suma_work_time (total_work : ℝ) (renu_rate : ℝ) (suma_rate : ℝ) :
  renu_rate = total_work / 8 →
  renu_rate + suma_rate = total_work / 3 →
  total_work / suma_rate = 4.8 :=
by sorry

end NUMINAMATH_CALUDE_suma_work_time_l1106_110675


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1106_110620

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + x^5 + 3 * x^4 + 7 * x^2 + 2 * x + 25) - 
  (x^6 + 2 * x^5 + x^4 + x^3 + 8 * x^2 + 15) = 
  x^6 - x^5 + 2 * x^4 - x^3 - x^2 + 2 * x + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1106_110620


namespace NUMINAMATH_CALUDE_intersection_slope_inequality_l1106_110672

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := 3/2 * x^2 - (6+a)*x + 2*a * f x

noncomputable def g (x : ℝ) : ℝ := f x / (deriv f x)

theorem intersection_slope_inequality (a k x₁ x₂ : ℝ) (h₁ : a > 0) (h₂ : x₁ < x₂) 
  (h₃ : ∃ y₁ y₂, (k * x₁ + y₁ = deriv g x₁) ∧ (k * x₂ + y₂ = deriv g x₂)) :
  x₁ < 1/k ∧ 1/k < x₂ := by sorry

end NUMINAMATH_CALUDE_intersection_slope_inequality_l1106_110672


namespace NUMINAMATH_CALUDE_circle_equation_l1106_110640

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points A and B
def A : ℝ × ℝ := (0, -6)
def B : ℝ × ℝ := (1, -5)

-- Define the line l
def line_l (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0

-- Theorem statement
theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    A ∈ Circle center radius ∧
    B ∈ Circle center radius ∧
    line_l center ∧
    center = (-3, -2) ∧
    radius = 5 :=
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1106_110640


namespace NUMINAMATH_CALUDE_algebra_test_male_students_l1106_110692

/-- Proves that given the conditions of the algebra test problem, the number of male students is 8 -/
theorem algebra_test_male_students
  (total_average : ℝ)
  (male_average : ℝ)
  (female_average : ℝ)
  (female_count : ℕ)
  (h_total_average : total_average = 90)
  (h_male_average : male_average = 83)
  (h_female_average : female_average = 92)
  (h_female_count : female_count = 28) :
  ∃ (male_count : ℕ),
    (male_count : ℝ) * male_average + (female_count : ℝ) * female_average =
      (male_count + female_count : ℝ) * total_average ∧
    male_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_male_students_l1106_110692


namespace NUMINAMATH_CALUDE_sector_angle_l1106_110624

theorem sector_angle (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 2) :
  ∃ (α r : ℝ), α * r = arc_length ∧ (1/2) * α * r^2 = area ∧ α = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1106_110624


namespace NUMINAMATH_CALUDE_pie_rows_l1106_110622

/-- Given the number of pecan and apple pies, and the number of pies per row,
    calculate the number of complete rows. -/
def calculate_rows (pecan_pies apple_pies pies_per_row : ℕ) : ℕ :=
  (pecan_pies + apple_pies) / pies_per_row

/-- Theorem stating that with 16 pecan pies and 14 apple pies,
    arranged in rows of 5 pies each, there will be 6 complete rows. -/
theorem pie_rows : calculate_rows 16 14 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_rows_l1106_110622


namespace NUMINAMATH_CALUDE_extremum_derivative_zero_not_sufficient_l1106_110615

-- Define a differentiable function f on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for a point to be an extremum
def IsExtremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

-- Theorem statement
theorem extremum_derivative_zero
  (x₀ : ℝ)
  (h_extremum : IsExtremum f x₀) :
  deriv f x₀ = 0 :=
sorry

-- Counter-example to show the converse is not always true
def counter_example : ℝ → ℝ := fun x ↦ x^3

theorem not_sufficient
  (h_deriv_zero : deriv counter_example 0 = 0)
  (h_not_extremum : ¬ IsExtremum counter_example 0) :
  ∃ f : ℝ → ℝ, ∃ x₀ : ℝ, deriv f x₀ = 0 ∧ ¬ IsExtremum f x₀ :=
sorry

end NUMINAMATH_CALUDE_extremum_derivative_zero_not_sufficient_l1106_110615


namespace NUMINAMATH_CALUDE_book_has_fifty_pages_l1106_110653

/-- Calculates the number of pages in a book based on reading speed and book structure -/
def book_pages (sentences_per_hour : ℕ) (paragraphs_per_page : ℕ) (sentences_per_paragraph : ℕ) (total_reading_hours : ℕ) : ℕ :=
  (sentences_per_hour * total_reading_hours) / (sentences_per_paragraph * paragraphs_per_page)

/-- Theorem stating that given the specific conditions, the book has 50 pages -/
theorem book_has_fifty_pages :
  book_pages 200 20 10 50 = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_has_fifty_pages_l1106_110653


namespace NUMINAMATH_CALUDE_distance_to_market_is_40_l1106_110627

/-- The distance between Andy's house and the market -/
def distance_to_market (distance_to_school : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance - 2 * distance_to_school

/-- Theorem: The distance between Andy's house and the market is 40 meters -/
theorem distance_to_market_is_40 :
  distance_to_market 50 140 = 40 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_market_is_40_l1106_110627


namespace NUMINAMATH_CALUDE_gcf_20_pair_l1106_110647

theorem gcf_20_pair : ∃! (a b : ℕ), 
  ((a = 200 ∧ b = 2000) ∨ 
   (a = 40 ∧ b = 50) ∨ 
   (a = 20 ∧ b = 40) ∨ 
   (a = 20 ∧ b = 25)) ∧ 
  Nat.gcd a b = 20 := by
  sorry

end NUMINAMATH_CALUDE_gcf_20_pair_l1106_110647


namespace NUMINAMATH_CALUDE_green_peaches_count_l1106_110665

/-- The number of peaches in a basket --/
structure Basket :=
  (red : ℕ)
  (yellow : ℕ)
  (green : ℕ)

/-- The basket with the given conditions --/
def my_basket : Basket :=
  { red := 2
  , yellow := 6
  , green := 6 + 8 }

/-- Theorem stating that the number of green peaches is 14 --/
theorem green_peaches_count (b : Basket) 
  (h1 : b.red = 2) 
  (h2 : b.yellow = 6) 
  (h3 : b.green = b.yellow + 8) : 
  b.green = 14 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l1106_110665


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l1106_110684

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l1106_110684


namespace NUMINAMATH_CALUDE_dog_paws_on_ground_l1106_110634

theorem dog_paws_on_ground : ∀ (total_dogs : ℕ) (dogs_on_back_legs : ℕ) (dogs_on_all_fours : ℕ),
  total_dogs = 12 →
  dogs_on_back_legs = total_dogs / 2 →
  dogs_on_all_fours = total_dogs / 2 →
  dogs_on_back_legs + dogs_on_all_fours = total_dogs →
  dogs_on_all_fours * 4 + dogs_on_back_legs * 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_dog_paws_on_ground_l1106_110634


namespace NUMINAMATH_CALUDE_product_inequality_l1106_110652

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  (a * b) ^ (1/4 : ℝ) + (b * c) ^ (1/4 : ℝ) + (c * a) ^ (1/4 : ℝ) < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1106_110652


namespace NUMINAMATH_CALUDE_jefferson_high_school_groups_l1106_110616

/-- Represents the number of students in exactly two groups -/
def students_in_two_groups (total_students : ℕ) (orchestra : ℕ) (band : ℕ) (chorus : ℕ) (in_any_group : ℕ) : ℕ :=
  orchestra + band + chorus - in_any_group

/-- Theorem: Given the conditions from Jefferson High School, 
    the number of students in exactly two groups is 130 -/
theorem jefferson_high_school_groups : 
  students_in_two_groups 500 120 190 220 400 = 130 := by
  sorry

end NUMINAMATH_CALUDE_jefferson_high_school_groups_l1106_110616


namespace NUMINAMATH_CALUDE_factorization_equality_l1106_110600

theorem factorization_equality (x y : ℝ) :
  -12 * x * y^2 * (x + y) + 18 * x^2 * y * (x + y) = 6 * x * y * (x + y) * (3 * x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1106_110600


namespace NUMINAMATH_CALUDE_perimeter_of_picture_area_l1106_110698

/-- Given a sheet of paper and a margin, calculate the perimeter of the remaining area --/
def perimeter_of_remaining_area (paper_width paper_length margin : ℕ) : ℕ :=
  2 * ((paper_width - 2 * margin) + (paper_length - 2 * margin))

/-- Theorem: The perimeter of the remaining area for a 12x16 inch paper with 2-inch margins is 40 inches --/
theorem perimeter_of_picture_area : perimeter_of_remaining_area 12 16 2 = 40 := by
  sorry

#eval perimeter_of_remaining_area 12 16 2

end NUMINAMATH_CALUDE_perimeter_of_picture_area_l1106_110698


namespace NUMINAMATH_CALUDE_max_insurmountable_questions_max_insurmountable_questions_is_10_l1106_110677

theorem max_insurmountable_questions :
  ∀ (x₀ x₁ x₂ x₃ : ℕ),
    x₀ + x₁ + x₂ + x₃ = 40 →
    3 * x₃ + 2 * x₂ + x₁ = 64 →
    x₂ = 2 * x₀ →
    x₀ ≤ 10 :=
by
  sorry

theorem max_insurmountable_questions_is_10 :
  ∃ (x₀ x₁ x₂ x₃ : ℕ),
    x₀ + x₁ + x₂ + x₃ = 40 ∧
    3 * x₃ + 2 * x₂ + x₁ = 64 ∧
    x₂ = 2 * x₀ ∧
    x₀ = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_max_insurmountable_questions_max_insurmountable_questions_is_10_l1106_110677


namespace NUMINAMATH_CALUDE_school_survey_result_l1106_110633

/-- Calculates the number of girls in a school based on stratified sampling -/
def girlsInSchool (totalStudents sampleSize girlsInSample : ℕ) : ℕ :=
  (girlsInSample * totalStudents) / sampleSize

/-- Theorem stating that given the problem conditions, the number of girls in the school is 760 -/
theorem school_survey_result :
  let totalStudents : ℕ := 1600
  let sampleSize : ℕ := 200
  let girlsInSample : ℕ := 95
  girlsInSchool totalStudents sampleSize girlsInSample = 760 := by
  sorry

end NUMINAMATH_CALUDE_school_survey_result_l1106_110633


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1106_110614

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| = 3 - x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1106_110614


namespace NUMINAMATH_CALUDE_range_of_a_l1106_110603

-- Define the propositions p and q
def p (x a : ℝ) : Prop := |x - a| < 3
def q (x : ℝ) : Prop := x^2 - 2*x - 3 < 0

-- Define the theorem
theorem range_of_a :
  (∀ x, q x → p x a) ∧ 
  (∃ x, p x a ∧ ¬q x) →
  0 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1106_110603


namespace NUMINAMATH_CALUDE_average_of_z_multiples_l1106_110669

/-- The average of z, 4z, 10z, 22z, and 46z is 16.6z -/
theorem average_of_z_multiples (z : ℝ) : 
  (z + 4*z + 10*z + 22*z + 46*z) / 5 = 16.6 * z := by
  sorry

end NUMINAMATH_CALUDE_average_of_z_multiples_l1106_110669


namespace NUMINAMATH_CALUDE_village_blocks_l1106_110654

/-- The number of blocks in a village, given the number of children per block and the total number of children. -/
def number_of_blocks (children_per_block : ℕ) (total_children : ℕ) : ℕ :=
  total_children / children_per_block

/-- Theorem: Given 6 children per block and 54 total children, there are 9 blocks in the village. -/
theorem village_blocks :
  number_of_blocks 6 54 = 9 := by
  sorry

end NUMINAMATH_CALUDE_village_blocks_l1106_110654


namespace NUMINAMATH_CALUDE_f_half_equals_half_l1106_110608

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 1 - 2 * x^2

-- State the theorem
theorem f_half_equals_half :
  (∀ x : ℝ, f (Real.sin x) = Real.cos (2 * x)) →
  f (1/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_half_equals_half_l1106_110608


namespace NUMINAMATH_CALUDE_farm_animals_l1106_110687

theorem farm_animals (goats chickens ducks pigs : ℕ) : 
  chickens = 2 * goats →
  ducks = (goats + chickens) / 2 →
  pigs = ducks / 3 →
  goats = pigs + 33 →
  goats = 66 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l1106_110687


namespace NUMINAMATH_CALUDE_jerry_reading_proof_l1106_110666

def remaining_pages (total_pages pages_read_day1 pages_read_day2 : ℕ) : ℕ :=
  total_pages - (pages_read_day1 + pages_read_day2)

theorem jerry_reading_proof :
  remaining_pages 93 30 20 = 43 := by
  sorry

end NUMINAMATH_CALUDE_jerry_reading_proof_l1106_110666


namespace NUMINAMATH_CALUDE_P_n_has_n_distinct_real_roots_P_2018_has_2018_distinct_real_roots_l1106_110655

-- Define the sequence of polynomials
def P : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 1
  | 1 => λ x => x
  | (n + 2) => λ x => x * P (n + 1) x - P n x

-- Define a function to count distinct real roots
noncomputable def count_distinct_real_roots (f : ℝ → ℝ) : ℕ := sorry

-- State the theorem
theorem P_n_has_n_distinct_real_roots (n : ℕ) :
  count_distinct_real_roots (P n) = n := by sorry

-- The specific case for P₂₀₁₈
theorem P_2018_has_2018_distinct_real_roots :
  count_distinct_real_roots (P 2018) = 2018 := by sorry

end NUMINAMATH_CALUDE_P_n_has_n_distinct_real_roots_P_2018_has_2018_distinct_real_roots_l1106_110655


namespace NUMINAMATH_CALUDE_increase_average_by_transfer_l1106_110601

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def removeElement (l : List ℕ) (x : ℕ) : List ℕ :=
  l.filter (· ≠ x)

theorem increase_average_by_transfer :
  ∃ g ∈ group1,
    average (removeElement group1 g) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end NUMINAMATH_CALUDE_increase_average_by_transfer_l1106_110601


namespace NUMINAMATH_CALUDE_closest_fraction_l1106_110668

def medals_won : ℚ := 24 / 150

def fractions : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (f : ℚ), f ∈ fractions ∧
  ∀ (g : ℚ), g ∈ fractions → |medals_won - f| ≤ |medals_won - g| ∧
  f = 1/6 :=
sorry

end NUMINAMATH_CALUDE_closest_fraction_l1106_110668


namespace NUMINAMATH_CALUDE_rectangle_perimeters_l1106_110661

/-- The perimeter of a rectangle given its width and height. -/
def rectanglePerimeter (width height : ℝ) : ℝ := 2 * (width + height)

/-- The theorem stating the perimeters of the three rectangles formed from four photographs. -/
theorem rectangle_perimeters (photo_perimeter : ℝ) 
  (h1 : photo_perimeter = 20)
  (h2 : ∃ (w h : ℝ), rectanglePerimeter w h = photo_perimeter ∧ 
                      rectanglePerimeter (2*w) (2*h) = 40 ∧
                      rectanglePerimeter (4*w) h = 44 ∧
                      rectanglePerimeter w (4*h) = 56) :
  ∃ (p1 p2 p3 : ℝ), p1 = 40 ∧ p2 = 44 ∧ p3 = 56 ∧
    (p1 = 40 ∨ p1 = 44 ∨ p1 = 56) ∧
    (p2 = 40 ∨ p2 = 44 ∨ p2 = 56) ∧
    (p3 = 40 ∨ p3 = 44 ∨ p3 = 56) ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeters_l1106_110661


namespace NUMINAMATH_CALUDE_gcd_90_450_l1106_110635

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by sorry

end NUMINAMATH_CALUDE_gcd_90_450_l1106_110635


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l1106_110626

theorem reciprocal_of_sum : (1 / (1/3 + 1/4) : ℚ) = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l1106_110626


namespace NUMINAMATH_CALUDE_cases_in_1975_l1106_110602

/-- Calculates the number of disease cases in a given year, assuming a linear decrease --/
def casesInYear (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let yearlyDecrease := totalDecrease / totalYears
  let targetYearDiff := targetYear - initialYear
  initialCases - (yearlyDecrease * targetYearDiff)

/-- Theorem stating that given the conditions, the number of cases in 1975 would be 300,150 --/
theorem cases_in_1975 :
  casesInYear 1950 600000 2000 300 1975 = 300150 := by
  sorry

end NUMINAMATH_CALUDE_cases_in_1975_l1106_110602


namespace NUMINAMATH_CALUDE_givenVectorIsDirectionVector_l1106_110686

/-- A line in 2D space --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A vector in 2D space --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The line x-3y+1=0 --/
def givenLine : Line2D :=
  { a := 1, b := -3, c := 1 }

/-- The vector (3,1) --/
def givenVector : Vector2D :=
  { x := 3, y := 1 }

/-- Definition: A vector is a direction vector of a line if it's parallel to the line --/
def isDirectionVector (v : Vector2D) (l : Line2D) : Prop :=
  v.x * l.b = -v.y * l.a

/-- Theorem: The vector (3,1) is a direction vector of the line x-3y+1=0 --/
theorem givenVectorIsDirectionVector : isDirectionVector givenVector givenLine := by
  sorry

end NUMINAMATH_CALUDE_givenVectorIsDirectionVector_l1106_110686


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l1106_110609

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  x * (x - 4) = 2 * x - 8 ↔ x = 4 ∨ x = 2 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (2 * x) / (2 * x - 3) - 4 / (2 * x + 3) = 1 ↔ x = 10.5 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l1106_110609


namespace NUMINAMATH_CALUDE_no_real_m_for_equal_roots_l1106_110660

theorem no_real_m_for_equal_roots : 
  ¬∃ (m : ℝ), ∃ (x : ℝ), 
    (x * (x + 2) - (m + 3)) / ((x + 2) * (m + 2)) = x / m ∧
    ∀ (y : ℝ), (y * (y + 2) - (m + 3)) / ((y + 2) * (m + 2)) = y / m → y = x :=
by sorry

end NUMINAMATH_CALUDE_no_real_m_for_equal_roots_l1106_110660


namespace NUMINAMATH_CALUDE_ruby_reading_homework_l1106_110658

theorem ruby_reading_homework (nina_math : ℕ) (nina_reading : ℕ) (ruby_math : ℕ) (ruby_reading : ℕ) :
  nina_math = 4 * ruby_math →
  nina_reading = 8 * ruby_reading →
  ruby_math = 6 →
  nina_math + nina_reading = 48 →
  ruby_reading = 3 := by
sorry

end NUMINAMATH_CALUDE_ruby_reading_homework_l1106_110658


namespace NUMINAMATH_CALUDE_cost_of_2500_pencils_l1106_110694

/-- The cost of a given number of pencils, given the cost of 100 pencils -/
def cost_of_pencils (cost_per_100 : ℚ) (num_pencils : ℕ) : ℚ :=
  (cost_per_100 * num_pencils) / 100

/-- Theorem stating that 2500 pencils cost $750 when 100 pencils cost $30 -/
theorem cost_of_2500_pencils :
  cost_of_pencils 30 2500 = 750 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_2500_pencils_l1106_110694


namespace NUMINAMATH_CALUDE_medium_boxes_count_l1106_110613

def tape_large : ℕ := 4
def tape_medium : ℕ := 2
def tape_small : ℕ := 1
def tape_label : ℕ := 1
def large_boxes : ℕ := 2
def small_boxes : ℕ := 5
def total_tape : ℕ := 44

theorem medium_boxes_count : 
  ∃ (medium_boxes : ℕ), 
    large_boxes * (tape_large + tape_label) + 
    medium_boxes * (tape_medium + tape_label) + 
    small_boxes * (tape_small + tape_label) = total_tape ∧ 
    medium_boxes = 8 := by
sorry

end NUMINAMATH_CALUDE_medium_boxes_count_l1106_110613


namespace NUMINAMATH_CALUDE_candy_final_temperature_l1106_110664

/-- Calculates the final temperature of a candy mixture given the initial conditions and rates. -/
theorem candy_final_temperature 
  (initial_temp : ℝ) 
  (max_temp : ℝ) 
  (heating_rate : ℝ) 
  (cooling_rate : ℝ) 
  (total_time : ℝ) 
  (h1 : initial_temp = 60)
  (h2 : max_temp = 240)
  (h3 : heating_rate = 5)
  (h4 : cooling_rate = 7)
  (h5 : total_time = 46) :
  let heating_time := (max_temp - initial_temp) / heating_rate
  let cooling_time := total_time - heating_time
  let temp_drop := cooling_rate * cooling_time
  max_temp - temp_drop = 170 := by
  sorry

end NUMINAMATH_CALUDE_candy_final_temperature_l1106_110664


namespace NUMINAMATH_CALUDE_chosen_number_proof_l1106_110637

theorem chosen_number_proof (x : ℝ) : (x / 9) - 100 = 10 → x = 990 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l1106_110637


namespace NUMINAMATH_CALUDE_concert_tickets_l1106_110689

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem concert_tickets : choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_concert_tickets_l1106_110689


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l1106_110636

theorem tripled_base_and_exponent (c d : ℤ) (y : ℚ) (h1 : d ≠ 0) :
  (3 * c : ℚ) ^ (3 * d) = c ^ d * y ^ d → y = 27 * c ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l1106_110636


namespace NUMINAMATH_CALUDE_mabel_handled_90_transactions_l1106_110639

-- Define the number of transactions for each person
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := (110 * mabel_transactions) / 100
def cal_transactions : ℕ := (2 * anthony_transactions) / 3
def jade_transactions : ℕ := 80

-- State the theorem
theorem mabel_handled_90_transactions :
  mabel_transactions = 90 ∧
  anthony_transactions = (110 * mabel_transactions) / 100 ∧
  cal_transactions = (2 * anthony_transactions) / 3 ∧
  jade_transactions = cal_transactions + 14 ∧
  jade_transactions = 80 := by
  sorry


end NUMINAMATH_CALUDE_mabel_handled_90_transactions_l1106_110639


namespace NUMINAMATH_CALUDE_problem_statement_l1106_110610

theorem problem_statement (a b : ℝ) (h : Real.exp a + Real.exp b = 4) :
  a + b ≤ 2 * Real.log 2 ∧ Real.exp a + b ≤ 3 ∧ Real.exp (2 * a) + Real.exp (2 * b) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1106_110610


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1106_110648

theorem complex_fraction_equality : (1 - 2*I) / (1 + I) = (-1 - 3*I) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1106_110648


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1106_110681

theorem regular_polygon_sides : ∃ n : ℕ, n > 2 ∧ n - (n * (n - 3) / 4) = 0 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1106_110681


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1106_110638

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - r - 1 = 0 → s^3 - s - 1 = 0 → t^3 - t - 1 = 0 → 
  (1 + r) / (1 - r) + (1 + s) / (1 - s) + (1 + t) / (1 - t) = -7 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1106_110638


namespace NUMINAMATH_CALUDE_cd_player_cost_l1106_110644

/-- The amount spent on the CD player, given the total amount spent and the amounts spent on speakers and tires. -/
theorem cd_player_cost (total spent_on_speakers spent_on_tires : ℚ) 
  (h_total : total = 387.85)
  (h_speakers : spent_on_speakers = 136.01)
  (h_tires : spent_on_tires = 112.46) :
  total - (spent_on_speakers + spent_on_tires) = 139.38 := by
  sorry

end NUMINAMATH_CALUDE_cd_player_cost_l1106_110644


namespace NUMINAMATH_CALUDE_research_project_hours_difference_l1106_110682

/-- The research project problem -/
theorem research_project_hours_difference
  (total_payment : ℝ)
  (wage_difference : ℝ)
  (wage_ratio : ℝ)
  (h1 : total_payment = 480)
  (h2 : wage_difference = 8)
  (h3 : wage_ratio = 1.5) :
  ∃ (hours_p hours_q : ℝ),
    hours_q - hours_p = 10 ∧
    hours_p * (wage_ratio * (total_payment / hours_q)) = total_payment ∧
    hours_q * (total_payment / hours_q) = total_payment ∧
    wage_ratio * (total_payment / hours_q) = (total_payment / hours_q) + wage_difference :=
by sorry


end NUMINAMATH_CALUDE_research_project_hours_difference_l1106_110682


namespace NUMINAMATH_CALUDE_calculation_result_quadratic_solution_l1106_110649

-- Problem 1
theorem calculation_result : Real.sqrt 9 + |1 - Real.sqrt 2| + ((-8 : ℝ) ^ (1/3)) - Real.sqrt 2 = 0 := by
  sorry

-- Problem 2
theorem quadratic_solution (x : ℝ) (h : 4 * x^2 - 16 = 0) : x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_quadratic_solution_l1106_110649


namespace NUMINAMATH_CALUDE_cube_volume_problem_l1106_110663

theorem cube_volume_problem (a : ℕ) : 
  (a + 1) * (a + 1) * (a - 2) = a^3 - 27 → a^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l1106_110663


namespace NUMINAMATH_CALUDE_triangle_angles_l1106_110604

open Real

theorem triangle_angles (a b c : ℝ) (A B C : ℝ) : 
  let m : ℝ × ℝ := (sqrt 3, -1)
  let n : ℝ × ℝ := (cos A, sin A)
  (m.1 * n.1 + m.2 * n.2 = 0) →  -- m ⊥ n
  (a * cos B + b * cos A = c * sin C) →  -- given condition
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- positive side lengths
  (A > 0 ∧ B > 0 ∧ C > 0) →  -- positive angles
  (A + B + C = π) →  -- sum of angles in a triangle
  (A = π/3 ∧ B = π/6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_l1106_110604


namespace NUMINAMATH_CALUDE_custom_operation_value_l1106_110696

/-- Custom operation * for non-zero integers -/
def star (a b : ℤ) : ℚ :=
  1 / a + 1 / b

theorem custom_operation_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 10) (prod_eq : a * b = 24) : 
  star a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_value_l1106_110696


namespace NUMINAMATH_CALUDE_certain_number_problem_l1106_110680

theorem certain_number_problem (x : ℝ) : 
  3.6 * x * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001 → x = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1106_110680


namespace NUMINAMATH_CALUDE_twenty_fifth_in_base5_l1106_110617

/-- Converts a natural number to its representation in base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid number in base 5 --/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

/-- Converts a list of base 5 digits to a natural number --/
def fromBase5 (l : List ℕ) : ℕ :=
  l.foldl (fun acc d => 5 * acc + d) 0

theorem twenty_fifth_in_base5 :
  ∃ (l : List ℕ), isValidBase5 l ∧ fromBase5 l = 25 ∧ l = [1, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_twenty_fifth_in_base5_l1106_110617


namespace NUMINAMATH_CALUDE_polynomial_coefficient_problem_l1106_110643

theorem polynomial_coefficient_problem (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 + x) * (a - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) →
  a₃ = -5 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_problem_l1106_110643


namespace NUMINAMATH_CALUDE_dividend_division_theorem_l1106_110685

theorem dividend_division_theorem : ∃ (q r : ℕ), 
  220030 = (555 + 445) * q + r ∧ 
  r < (555 + 445) ∧ 
  r = 30 ∧ 
  q = 2 * (555 - 445) :=
by sorry

end NUMINAMATH_CALUDE_dividend_division_theorem_l1106_110685


namespace NUMINAMATH_CALUDE_lucy_groceries_l1106_110656

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 12

/-- The number of packs of noodles Lucy bought -/
def noodles : ℕ := 16

/-- The total number of packs of groceries Lucy bought -/
def total_groceries : ℕ := cookies + noodles

theorem lucy_groceries : total_groceries = 28 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l1106_110656


namespace NUMINAMATH_CALUDE_strawberry_sales_formula_l1106_110697

/-- The relationship between strawberry sales volume and total sales price -/
theorem strawberry_sales_formula (n : ℕ+) :
  let price_increase : ℝ := 40.5
  let total_price : ℕ+ → ℝ := λ k => k.val * price_increase
  total_price n = n.val * price_increase :=
by sorry

end NUMINAMATH_CALUDE_strawberry_sales_formula_l1106_110697


namespace NUMINAMATH_CALUDE_blue_pick_fraction_l1106_110650

def guitar_pick_collection (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : Prop :=
  red + blue + yellow = total ∧ red = total / 2 ∧ blue = 12 ∧ yellow = 6

theorem blue_pick_fraction 
  (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) 
  (h : guitar_pick_collection total red blue yellow) : 
  blue = total / 3 := by
sorry

end NUMINAMATH_CALUDE_blue_pick_fraction_l1106_110650


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1106_110678

theorem solve_linear_equation :
  ∀ x : ℚ, -4 * x - 15 = 12 * x + 5 → x = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1106_110678


namespace NUMINAMATH_CALUDE_white_then_red_probability_l1106_110674

/-- Represents the colors of the balls in the bag -/
inductive Color
  | Red | Blue | Green | Yellow | Purple | White

/-- The total number of balls in the bag -/
def total_balls : ℕ := 6

/-- The number of colored balls (excluding white) -/
def colored_balls : ℕ := 5

/-- The probability of extracting a specific ball from the bag -/
def prob_extract (n : ℕ) : ℚ := 1 / n

/-- The probability of extracting a white ball first and a red ball second -/
def prob_white_then_red : ℚ :=
  prob_extract total_balls * prob_extract colored_balls

theorem white_then_red_probability :
  prob_white_then_red = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_white_then_red_probability_l1106_110674


namespace NUMINAMATH_CALUDE_f_not_in_second_quadrant_l1106_110659

/-- A linear function f(x) = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x - 3

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem: The graph of f(x) = 2x - 3 does not pass through the second quadrant -/
theorem f_not_in_second_quadrant :
  ∀ x y : ℝ, f x = y → ¬(second_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_f_not_in_second_quadrant_l1106_110659


namespace NUMINAMATH_CALUDE_ratio_proof_l1106_110631

theorem ratio_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (hx : x = 1.25 * a) (hm : m = 0.2 * b) (hm_x : m / x = 0.2) : 
  a / b = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_proof_l1106_110631


namespace NUMINAMATH_CALUDE_not_convex_pentagon_with_diagonals_l1106_110671

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

end NUMINAMATH_CALUDE_not_convex_pentagon_with_diagonals_l1106_110671


namespace NUMINAMATH_CALUDE_best_fit_model_l1106_110690

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared_nonneg : 0 ≤ r_squared
  h_r_squared_le_one : r_squared ≤ 1

/-- Determines if one model has a better fit than another based on R² -/
def better_fit (m1 m2 : RegressionModel) : Prop :=
  m1.r_squared > m2.r_squared

theorem best_fit_model (m1 m2 m3 m4 : RegressionModel)
  (h1 : m1.r_squared = 0.87)
  (h2 : m2.r_squared = 0.97)
  (h3 : m3.r_squared = 0.50)
  (h4 : m4.r_squared = 0.25) :
  better_fit m2 m1 ∧ better_fit m2 m3 ∧ better_fit m2 m4 := by
  sorry

end NUMINAMATH_CALUDE_best_fit_model_l1106_110690


namespace NUMINAMATH_CALUDE_chestnut_picking_l1106_110612

/-- The amount of chestnuts picked by Mary, Peter, and Lucy -/
theorem chestnut_picking (mary peter lucy : ℝ) 
  (h1 : mary = 2 * peter)  -- Mary picked twice as much as Peter
  (h2 : lucy = peter + 2)  -- Lucy picked 2 kg more than Peter
  (h3 : mary = 12)         -- Mary picked 12 kg
  : mary + peter + lucy = 26 := by
  sorry

#check chestnut_picking

end NUMINAMATH_CALUDE_chestnut_picking_l1106_110612


namespace NUMINAMATH_CALUDE_eraser_ratio_is_two_to_one_l1106_110662

-- Define the number of erasers for each person
def tanya_total : ℕ := 20
def hanna_total : ℕ := 4

-- Define the number of red erasers Tanya has
def tanya_red : ℕ := tanya_total / 2

-- Define Rachel's erasers in terms of Tanya's red erasers
def rachel_total : ℕ := tanya_red / 2 - 3

-- Define the ratio of Hanna's erasers to Rachel's erasers
def eraser_ratio : ℚ := hanna_total / rachel_total

-- Theorem to prove
theorem eraser_ratio_is_two_to_one :
  eraser_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_eraser_ratio_is_two_to_one_l1106_110662


namespace NUMINAMATH_CALUDE_chessboard_rearrangement_l1106_110606

def knight_distance (a b : ℕ × ℕ) : ℕ :=
  sorry

def king_distance (a b : ℕ × ℕ) : ℕ :=
  sorry

def is_valid_rearrangement (N : ℕ) (f : (ℕ × ℕ) → (ℕ × ℕ)) : Prop :=
  ∀ a b : ℕ × ℕ, a.1 < N ∧ a.2 < N ∧ b.1 < N ∧ b.2 < N →
    knight_distance a b = 1 → king_distance (f a) (f b) = 1

theorem chessboard_rearrangement :
  (∃ f : (ℕ × ℕ) → (ℕ × ℕ), is_valid_rearrangement 3 f) ∧
  (¬ ∃ f : (ℕ × ℕ) → (ℕ × ℕ), is_valid_rearrangement 8 f) :=
sorry

end NUMINAMATH_CALUDE_chessboard_rearrangement_l1106_110606


namespace NUMINAMATH_CALUDE_total_height_is_24cm_l1106_110629

/-- The number of washers in the stack -/
def num_washers : ℕ := 11

/-- The thickness of each washer in cm -/
def washer_thickness : ℝ := 2

/-- The outer diameter of the top washer in cm -/
def top_diameter : ℝ := 24

/-- The outer diameter of the bottom washer in cm -/
def bottom_diameter : ℝ := 4

/-- The decrease in diameter between consecutive washers in cm -/
def diameter_decrease : ℝ := 2

/-- The extra height for hooks at top and bottom in cm -/
def hook_height : ℝ := 2

theorem total_height_is_24cm : 
  (num_washers : ℝ) * washer_thickness + hook_height = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_height_is_24cm_l1106_110629


namespace NUMINAMATH_CALUDE_smallest_multiple_37_3_mod_97_l1106_110605

theorem smallest_multiple_37_3_mod_97 : ∃ n : ℕ, 
  n > 0 ∧ 
  37 ∣ n ∧ 
  n % 97 = 3 ∧ 
  ∀ m : ℕ, m > 0 → 37 ∣ m → m % 97 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_37_3_mod_97_l1106_110605


namespace NUMINAMATH_CALUDE_runner_time_difference_l1106_110628

theorem runner_time_difference (danny_time steve_time : ℝ) (h1 : danny_time = 27) 
  (h2 : danny_time = steve_time / 2) : steve_time / 4 - danny_time / 2 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_runner_time_difference_l1106_110628


namespace NUMINAMATH_CALUDE_circle_plus_minus_balance_l1106_110699

theorem circle_plus_minus_balance (a b p q : ℕ) : a - b = p - q :=
  sorry

end NUMINAMATH_CALUDE_circle_plus_minus_balance_l1106_110699


namespace NUMINAMATH_CALUDE_base8_digit_product_l1106_110642

/-- Convert a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculate the product of a list of natural numbers -/
def productList (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 8 representation of 6543₁₀ is 168 -/
theorem base8_digit_product :
  productList (toBase8 6543) = 168 :=
sorry

end NUMINAMATH_CALUDE_base8_digit_product_l1106_110642


namespace NUMINAMATH_CALUDE_optimal_procurement_plan_l1106_110673

/-- Represents a snowflake model type -/
inductive ModelType
| A
| B

/-- Represents the number of pipes needed for a model -/
structure PipeCount where
  long : ℕ
  short : ℕ

/-- Represents the store's inventory -/
structure Inventory where
  long : ℕ
  short : ℕ

/-- Represents a procurement plan -/
structure ProcurementPlan where
  modelA : ℕ
  modelB : ℕ

def pipe_price : ℚ := 1/2

def long_pipe_price : ℚ := 2 * pipe_price

def inventory : Inventory := ⟨267, 2130⟩

def budget : ℚ := 1280

def pipes_per_model (t : ModelType) : PipeCount :=
  match t with
  | ModelType.A => ⟨3, 21⟩
  | ModelType.B => ⟨3, 27⟩

def cost_of_plan (plan : ProcurementPlan) : ℚ :=
  let total_long := plan.modelA * (pipes_per_model ModelType.A).long + 
                    plan.modelB * (pipes_per_model ModelType.B).long
  let total_short := plan.modelA * (pipes_per_model ModelType.A).short + 
                     plan.modelB * (pipes_per_model ModelType.B).short - 
                     (total_long / 3)
  total_long * long_pipe_price + total_short * pipe_price

def is_valid_plan (plan : ProcurementPlan) : Prop :=
  let total_long := plan.modelA * (pipes_per_model ModelType.A).long + 
                    plan.modelB * (pipes_per_model ModelType.B).long
  let total_short := plan.modelA * (pipes_per_model ModelType.A).short + 
                     plan.modelB * (pipes_per_model ModelType.B).short
  total_long ≤ inventory.long ∧ 
  total_short ≤ inventory.short ∧ 
  cost_of_plan plan = budget

theorem optimal_procurement_plan :
  ∀ plan : ProcurementPlan,
    is_valid_plan plan →
    plan.modelA + plan.modelB ≤ 49 ∧
    (plan.modelA + plan.modelB = 49 → plan.modelA = 48 ∧ plan.modelB = 1) :=
sorry

end NUMINAMATH_CALUDE_optimal_procurement_plan_l1106_110673


namespace NUMINAMATH_CALUDE_three_distinct_real_roots_l1106_110619

/-- The cubic function f(x) = x^3 - 3x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x - a

/-- Theorem stating the condition for three distinct real roots -/
theorem three_distinct_real_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ 
  -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_distinct_real_roots_l1106_110619


namespace NUMINAMATH_CALUDE_correct_answers_count_l1106_110679

/-- Represents a test with a specific scoring system. -/
structure Test where
  total_questions : ℕ
  score : ℕ → ℕ → ℤ
  all_answered : ℕ → ℕ → Prop

/-- Theorem stating the number of correct answers given the test conditions. -/
theorem correct_answers_count (test : Test)
    (h_total : test.total_questions = 100)
    (h_score : ∀ c i, test.score c i = c - 2 * i)
    (h_all_answered : ∀ c i, test.all_answered c i ↔ c + i = test.total_questions)
    (h_student_score : ∃ c i, test.all_answered c i ∧ test.score c i = 73) :
    ∃ c i, test.all_answered c i ∧ test.score c i = 73 ∧ c = 91 := by
  sorry

#check correct_answers_count

end NUMINAMATH_CALUDE_correct_answers_count_l1106_110679


namespace NUMINAMATH_CALUDE_digit_sum_equals_sixteen_l1106_110683

/-- Given distinct digits a, b, and c satisfying aba + aba = cbc,
    prove that a + b + c = 16 -/
theorem digit_sum_equals_sixteen
  (a b c : ℕ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_digits : a < 10 ∧ b < 10 ∧ c < 10)
  (h_equation : 100 * a + 10 * b + a + 100 * a + 10 * b + a = 100 * c + 10 * b + c) :
  a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_equals_sixteen_l1106_110683


namespace NUMINAMATH_CALUDE_square_area_change_l1106_110623

theorem square_area_change (original_area : ℝ) (decrease_percent : ℝ) (increase_percent : ℝ) : 
  original_area = 625 →
  decrease_percent = 0.2 →
  increase_percent = 0.2 →
  let original_side : ℝ := Real.sqrt original_area
  let new_side1 : ℝ := original_side * (1 - decrease_percent)
  let new_side2 : ℝ := original_side * (1 + increase_percent)
  new_side1 * new_side2 = 600 := by
sorry

end NUMINAMATH_CALUDE_square_area_change_l1106_110623


namespace NUMINAMATH_CALUDE_merchant_profit_problem_l1106_110641

theorem merchant_profit_problem (X : ℕ) (C S : ℝ) : 
  X * C = 25 * S → -- Cost price of X articles equals selling price of 25 articles
  S = 1.6 * C →    -- 60% profit, selling price is 160% of cost price
  X = 40           -- Number of articles bought at cost price is 40
  := by sorry

end NUMINAMATH_CALUDE_merchant_profit_problem_l1106_110641


namespace NUMINAMATH_CALUDE_degrees_to_radians_l1106_110607

theorem degrees_to_radians (π : Real) (h : π = 180) : 
  (240 : Real) * π / 180 = 4 * π / 3 := by sorry

end NUMINAMATH_CALUDE_degrees_to_radians_l1106_110607
