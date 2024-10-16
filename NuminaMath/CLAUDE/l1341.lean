import Mathlib

namespace NUMINAMATH_CALUDE_john_profit_is_13100_l1341_134110

/-- Calculates the profit made by John from chopping trees and selling tables -/
def john_profit : ℕ := by
  -- Define the number of trees in each group
  let trees_group1 : ℕ := 10
  let trees_group2 : ℕ := 10
  let trees_group3 : ℕ := 10

  -- Define the number of planks per tree in each group
  let planks_per_tree_group1 : ℕ := 20
  let planks_per_tree_group2 : ℕ := 25
  let planks_per_tree_group3 : ℕ := 30

  -- Define the labor cost per tree in each group
  let labor_cost_group1 : ℕ := 120
  let labor_cost_group2 : ℕ := 80
  let labor_cost_group3 : ℕ := 60

  -- Define the number of planks required to make a table
  let planks_per_table : ℕ := 15

  -- Define the selling price for each group of tables
  let price_tables_1_10 : ℕ := 350
  let price_tables_11_30 : ℕ := 325
  let price_decrease_per_5_tables : ℕ := 10

  -- Calculate the total number of planks
  let total_planks : ℕ := 
    trees_group1 * planks_per_tree_group1 +
    trees_group2 * planks_per_tree_group2 +
    trees_group3 * planks_per_tree_group3

  -- Calculate the total number of tables
  let total_tables : ℕ := total_planks / planks_per_table

  -- Calculate the total labor cost
  let total_labor_cost : ℕ := 
    trees_group1 * labor_cost_group1 +
    trees_group2 * labor_cost_group2 +
    trees_group3 * labor_cost_group3

  -- Calculate the total revenue
  let total_revenue : ℕ := 
    10 * price_tables_1_10 +
    20 * price_tables_11_30 +
    5 * (price_tables_11_30 - price_decrease_per_5_tables) +
    5 * (price_tables_11_30 - 2 * price_decrease_per_5_tables) +
    5 * (price_tables_11_30 - 3 * price_decrease_per_5_tables) +
    5 * (price_tables_11_30 - 4 * price_decrease_per_5_tables)

  -- Calculate the profit
  let profit : ℕ := total_revenue - total_labor_cost

  -- Prove that the profit is equal to 13100
  sorry

theorem john_profit_is_13100 : john_profit = 13100 := by sorry

end NUMINAMATH_CALUDE_john_profit_is_13100_l1341_134110


namespace NUMINAMATH_CALUDE_polynomial_division_l1341_134173

/-- The dividend polynomial -/
def dividend (x : ℚ) : ℚ := 9*x^4 + 27*x^3 - 8*x^2 + 8*x + 5

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 3*x + 4

/-- The quotient polynomial -/
def quotient (x : ℚ) : ℚ := 3*x^3 + 5*x^2 - (28/3)*x + 136/9

/-- The remainder -/
def remainder : ℚ := 5 - 544/9

theorem polynomial_division :
  ∀ x, dividend x = divisor x * quotient x + remainder := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l1341_134173


namespace NUMINAMATH_CALUDE_smallest_base_for_100_l1341_134199

theorem smallest_base_for_100 : 
  ∃ b : ℕ, (b ≥ 5 ∧ b^2 ≤ 100 ∧ 100 < b^3) ∧ 
  (∀ c : ℕ, c < b → (c^2 > 100 ∨ 100 ≥ c^3)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_100_l1341_134199


namespace NUMINAMATH_CALUDE_polynomial_equality_solutions_l1341_134186

theorem polynomial_equality_solutions : 
  ∀ (a b c : ℤ), 
  (∀ x : ℤ, (x - a) * (x - 8) + 4 = (x + b) * (x + c)) → 
  ((a = 20 ∧ b = -6 ∧ c = -6) ∨ (a = 29 ∧ b = -9 ∧ c = -12)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_solutions_l1341_134186


namespace NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l1341_134182

/-- Proves that for a rectangular hall with width half the length and area 128 sq. m,
    the difference between length and width is 8 meters. -/
theorem rectangular_hall_dimension_difference :
  ∀ (length width : ℝ),
    width = length / 2 →
    length * width = 128 →
    length - width = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l1341_134182


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_equation_l1341_134143

theorem linear_coefficient_of_quadratic_equation :
  let equation := fun x : ℝ => x^2 - 2022*x - 2023
  ∃ a b c : ℝ, (∀ x, equation x = a*x^2 + b*x + c) ∧ b = -2022 :=
sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_equation_l1341_134143


namespace NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l1341_134160

/-- Represents the four quadrants in a coordinate system -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines the quadrant of an angle given in degrees -/
def angle_quadrant (α : ℝ) : Quadrant :=
  sorry

/-- Theorem: For any integer k, the angle α = k·180° + 45° lies in either the first or third quadrant -/
theorem angle_in_first_or_third_quadrant (k : ℤ) :
  let α := k * 180 + 45
  (angle_quadrant α = Quadrant.first) ∨ (angle_quadrant α = Quadrant.third) :=
sorry

end NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l1341_134160


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1341_134167

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point on a hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Left focus of a hyperbola -/
def left_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Right focus of a hyperbola -/
def right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_min : ∀ p : HyperbolaPoint h, 
    (distance (p.x, p.y) (right_focus h))^2 / distance (p.x, p.y) (left_focus h) ≥ 9 * h.a) :
  eccentricity h = 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1341_134167


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_l1341_134172

theorem smallest_k_for_64_power (k : ℕ) (some_exponent : ℕ) : k = 6 → some_exponent < 18 → 64^k > 4^some_exponent := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_l1341_134172


namespace NUMINAMATH_CALUDE_proportion_solution_l1341_134192

theorem proportion_solution (x : ℝ) : (0.60 / x = 6 / 4) → x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1341_134192


namespace NUMINAMATH_CALUDE_power_fraction_minus_one_l1341_134175

theorem power_fraction_minus_one : (5 / 3 : ℚ) ^ 4 - 1 = 544 / 81 := by sorry

end NUMINAMATH_CALUDE_power_fraction_minus_one_l1341_134175


namespace NUMINAMATH_CALUDE_unique_prime_product_power_l1341_134120

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The product of the first n prime numbers -/
def primeProduct (n : ℕ) : ℕ := sorry

theorem unique_prime_product_power :
  ∀ k : ℕ, k > 0 →
    (∃ a n : ℕ, n > 1 ∧ primeProduct k - 1 = a^n) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_product_power_l1341_134120


namespace NUMINAMATH_CALUDE_circles_tangency_l1341_134132

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O2 (x y a : ℝ) : Prop := (x + 4)^2 + (y - a)^2 = 25

-- Define the condition of having exactly one common point
def have_one_common_point (a : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, circle_O1 p.1 p.2 ∧ circle_O2 p.1 p.2 a

-- State the theorem
theorem circles_tangency (a : ℝ) :
  have_one_common_point a → a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 ∨ a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_circles_tangency_l1341_134132


namespace NUMINAMATH_CALUDE_fourth_sphere_radius_l1341_134191

/-- Given four spheres on a table, where each sphere touches the table and the other three spheres,
    and three of the spheres have radius R, the radius of the fourth sphere is 4R/3. -/
theorem fourth_sphere_radius (R : ℝ) (R_pos : R > 0) : ∃ r : ℝ,
  (∀ (i j : Fin 4), i ≠ j → ∃ (x y z : ℝ),
    (i.val < 3 → norm (⟨x, y, z⟩ : ℝ × ℝ × ℝ) = R) ∧
    (i.val = 3 → norm (⟨x, y, z⟩ : ℝ × ℝ × ℝ) = r) ∧
    (j.val < 3 → ∃ (x' y' z' : ℝ), norm (⟨x - x', y - y', z - z'⟩ : ℝ × ℝ × ℝ) = R + R) ∧
    (j.val = 3 → ∃ (x' y' z' : ℝ), norm (⟨x - x', y - y', z - z'⟩ : ℝ × ℝ × ℝ) = R + r) ∧
    z ≥ R ∧ z' ≥ R) ∧
  r = 4 * R / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_sphere_radius_l1341_134191


namespace NUMINAMATH_CALUDE_inverse_f_sum_l1341_134117

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 2*x - x^2 + 1

theorem inverse_f_sum :
  let f_inv := Function.invFun f
  f_inv (-1) + f_inv 1 + f_inv 5 = 4 + Real.sqrt 3 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_sum_l1341_134117


namespace NUMINAMATH_CALUDE_billy_sandwiches_l1341_134112

theorem billy_sandwiches (billy katelyn chloe : ℕ) : 
  katelyn = billy + 47 →
  chloe = (katelyn : ℚ) / 4 →
  billy + katelyn + chloe = 169 →
  billy = 49 := by
sorry

end NUMINAMATH_CALUDE_billy_sandwiches_l1341_134112


namespace NUMINAMATH_CALUDE_greatest_common_factor_40_120_100_l1341_134133

theorem greatest_common_factor_40_120_100 : Nat.gcd 40 (Nat.gcd 120 100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_40_120_100_l1341_134133


namespace NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l1341_134184

/-- Calculates Bhanu's expenditure on house rent given his spending pattern and petrol expenditure -/
theorem bhanu_house_rent_expenditure (income : ℝ) (petrol_expenditure : ℝ) :
  petrol_expenditure = 0.3 * income →
  0.3 * (0.7 * income) = 210 := by
  sorry

end NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l1341_134184


namespace NUMINAMATH_CALUDE_log_equation_solution_l1341_134144

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1)
  (h_eq : Real.log x / (3 * Real.log b) + Real.log b / (3 * Real.log x) = 1) :
  x = b ∨ x = b ^ ((3 - Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1341_134144


namespace NUMINAMATH_CALUDE_village_population_l1341_134108

theorem village_population (P : ℝ) : 
  (P * 1.25 * 0.75 = 18750) → P = 20000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1341_134108


namespace NUMINAMATH_CALUDE_binary_addition_correct_l1341_134198

/-- Represents a binary number as a list of bits (least significant bit first) -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- The four binary numbers given in the problem -/
def num1 : BinaryNumber := [true, false, true, true]
def num2 : BinaryNumber := [false, true, true]
def num3 : BinaryNumber := [true, true, false, true]
def num4 : BinaryNumber := [false, false, true, true, true]

/-- The expected sum in binary -/
def expected_sum : BinaryNumber := [true, false, true, false, false, true]

theorem binary_addition_correct :
  binary_to_decimal num1 + binary_to_decimal num2 + 
  binary_to_decimal num3 + binary_to_decimal num4 = 
  binary_to_decimal expected_sum := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_correct_l1341_134198


namespace NUMINAMATH_CALUDE_average_of_numbers_l1341_134180

def numbers : List ℕ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 114391 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1341_134180


namespace NUMINAMATH_CALUDE_tangent_product_inequality_l1341_134185

theorem tangent_product_inequality (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.tan (α/2) * Real.tan (β/2) * Real.tan (γ/2) ≤ Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_inequality_l1341_134185


namespace NUMINAMATH_CALUDE_difference_of_squares_l1341_134159

theorem difference_of_squares (a b : ℝ) : (a + b) * (b - a) = b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1341_134159


namespace NUMINAMATH_CALUDE_field_trip_attendance_l1341_134164

/-- The number of people who went on the field trip -/
def total_people (num_vans : ℕ) (num_buses : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ) : ℕ :=
  num_vans * people_per_van + num_buses * people_per_bus

/-- Theorem stating that the total number of people on the field trip is 342 -/
theorem field_trip_attendance : total_people 9 10 8 27 = 342 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_attendance_l1341_134164


namespace NUMINAMATH_CALUDE_original_savings_amount_l1341_134104

/-- Proves that the original savings amount was $11,000 given the spending pattern and remaining balance. -/
theorem original_savings_amount (initial_savings : ℝ) : 
  initial_savings * (1 - 0.2 - 0.4) - 1500 = 2900 → 
  initial_savings = 11000 := by
sorry

end NUMINAMATH_CALUDE_original_savings_amount_l1341_134104


namespace NUMINAMATH_CALUDE_hostel_expenditure_hostel_expenditure_result_l1341_134129

/-- Calculates the new total expenditure of a hostel after accommodating more students -/
theorem hostel_expenditure (initial_students : ℕ) (additional_students : ℕ) 
  (average_decrease : ℕ) (total_increase : ℕ) : ℕ :=
  let new_students := initial_students + additional_students
  let original_average := (total_increase + new_students * average_decrease) / (new_students - initial_students)
  new_students * (original_average - average_decrease)

/-- The total expenditure of the hostel after accommodating more students is 7500 rupees -/
theorem hostel_expenditure_result : 
  hostel_expenditure 100 25 10 500 = 7500 := by
  sorry

end NUMINAMATH_CALUDE_hostel_expenditure_hostel_expenditure_result_l1341_134129


namespace NUMINAMATH_CALUDE_onion_weight_problem_l1341_134113

theorem onion_weight_problem (total_weight : Real) (avg_weight_35 : Real) :
  total_weight = 7.68 ∧ avg_weight_35 = 0.190 →
  (total_weight * 1000 - 35 * avg_weight_35 * 1000) / 5 = 206 := by
  sorry

end NUMINAMATH_CALUDE_onion_weight_problem_l1341_134113


namespace NUMINAMATH_CALUDE_shelter_cat_count_l1341_134187

/-- Represents the state of the animal shelter --/
structure AnimalShelter where
  initialCats : ℕ
  newCats : ℕ
  adoptedCats : ℕ
  bornKittens : ℕ
  claimedPets : ℕ

/-- Calculates the final number of cats in the shelter --/
def finalCatCount (shelter : AnimalShelter) : ℕ :=
  shelter.initialCats + shelter.newCats - shelter.adoptedCats + shelter.bornKittens - shelter.claimedPets

/-- Theorem stating the final number of cats in the shelter --/
theorem shelter_cat_count : ∃ (shelter : AnimalShelter),
  shelter.initialCats = 60 ∧
  shelter.newCats = 30 ∧
  shelter.adoptedCats = 20 ∧
  shelter.bornKittens = 15 ∧
  shelter.claimedPets = 2 ∧
  finalCatCount shelter = 83 := by
  sorry

#check shelter_cat_count

end NUMINAMATH_CALUDE_shelter_cat_count_l1341_134187


namespace NUMINAMATH_CALUDE_prob_three_dice_l1341_134193

/-- The number of faces on a die -/
def num_faces : ℕ := 6

/-- The number of favorable outcomes on a single die (numbers greater than 2) -/
def favorable_outcomes : ℕ := 4

/-- The number of dice thrown simultaneously -/
def num_dice : ℕ := 3

/-- The probability of getting a number greater than 2 on a single die -/
def prob_single_die : ℚ := favorable_outcomes / num_faces

/-- The probability of getting a number greater than 2 on each of three dice -/
theorem prob_three_dice : (prob_single_die ^ num_dice : ℚ) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_dice_l1341_134193


namespace NUMINAMATH_CALUDE_max_salary_soccer_team_l1341_134190

/-- Represents the maximum salary problem for a soccer team -/
theorem max_salary_soccer_team 
  (num_players : ℕ) 
  (min_salary : ℕ) 
  (max_total_salary : ℕ) 
  (h1 : num_players = 25)
  (h2 : min_salary = 20000)
  (h3 : max_total_salary = 900000) :
  ∃ (max_single_salary : ℕ),
    max_single_salary = 420000 ∧
    max_single_salary + (num_players - 1) * min_salary ≤ max_total_salary ∧
    ∀ (salary : ℕ), 
      salary > max_single_salary → 
      salary + (num_players - 1) * min_salary > max_total_salary :=
by sorry

end NUMINAMATH_CALUDE_max_salary_soccer_team_l1341_134190


namespace NUMINAMATH_CALUDE_dividend_divisor_product_l1341_134194

theorem dividend_divisor_product (d : ℤ) (D : ℤ) : 
  D = d + 78 → D = 6 * d + 3 → D * d = 1395 := by
sorry

end NUMINAMATH_CALUDE_dividend_divisor_product_l1341_134194


namespace NUMINAMATH_CALUDE_train_speed_problem_l1341_134105

/-- Prove that given two trains on a 200 km track, where one starts at 7 am and the other at 8 am
    traveling towards each other, meeting at 12 pm, and the second train travels at 25 km/h,
    the speed of the first train is 20 km/h. -/
theorem train_speed_problem (total_distance : ℝ) (second_train_speed : ℝ) 
  (first_train_start_time : ℝ) (second_train_start_time : ℝ) (meeting_time : ℝ) :
  total_distance = 200 →
  second_train_speed = 25 →
  first_train_start_time = 7 →
  second_train_start_time = 8 →
  meeting_time = 12 →
  ∃ (first_train_speed : ℝ), first_train_speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1341_134105


namespace NUMINAMATH_CALUDE_science_club_neither_subject_l1341_134124

theorem science_club_neither_subject (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ)
  (h_total : total = 75)
  (h_biology : biology = 40)
  (h_chemistry : chemistry = 30)
  (h_both : both = 18) :
  total - (biology + chemistry - both) = 23 := by
  sorry

end NUMINAMATH_CALUDE_science_club_neither_subject_l1341_134124


namespace NUMINAMATH_CALUDE_initial_goldfish_count_l1341_134126

/-- The number of goldfish Paige initially raised in the pond -/
def initial_goldfish : ℕ := 15

/-- The number of goldfish remaining in the pond -/
def remaining_goldfish : ℕ := 4

/-- The number of goldfish that disappeared -/
def disappeared_goldfish : ℕ := 11

/-- Theorem: The initial number of goldfish is equal to the sum of the remaining and disappeared goldfish -/
theorem initial_goldfish_count : initial_goldfish = remaining_goldfish + disappeared_goldfish := by
  sorry

end NUMINAMATH_CALUDE_initial_goldfish_count_l1341_134126


namespace NUMINAMATH_CALUDE_function_second_derivative_at_e_l1341_134169

open Real

theorem function_second_derivative_at_e (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_def : ∀ x, f x = 2 * x * (deriv^[2] f e) + log x) : 
  deriv^[2] f e = -1 / e := by
  sorry

end NUMINAMATH_CALUDE_function_second_derivative_at_e_l1341_134169


namespace NUMINAMATH_CALUDE_no_solution_trigonometric_equation_l1341_134168

open Real

theorem no_solution_trigonometric_equation (m : ℝ) :
  ¬ ∃ x : ℝ, (sin (3 * x) * cos (π / 3 - x) + 1) / (sin (π / 3 - 7 * x) - cos (π / 6 + x) + m) = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_trigonometric_equation_l1341_134168


namespace NUMINAMATH_CALUDE_negative_one_to_zero_power_equals_one_l1341_134121

theorem negative_one_to_zero_power_equals_one : (-1 : ℤ) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_to_zero_power_equals_one_l1341_134121


namespace NUMINAMATH_CALUDE_triangle_inequality_l1341_134151

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p := (a + b + c) / 2
  a * Real.sqrt ((p - b) * (p - c) / (b * c)) +
  b * Real.sqrt ((p - c) * (p - a) / (a * c)) +
  c * Real.sqrt ((p - a) * (p - b) / (a * b)) ≥ p := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1341_134151


namespace NUMINAMATH_CALUDE_favorite_number_is_27_l1341_134139

theorem favorite_number_is_27 : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n^2 = (n / 10 + n % 10)^3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_favorite_number_is_27_l1341_134139


namespace NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l1341_134147

-- Define the function f
noncomputable def f (c : ℝ) : ℝ → ℝ := fun x =>
  if x ≥ 0 then 3^x - 2*x + c else -(3^(-x) - 2*(-x) + c)

-- State the theorem
theorem odd_function_value_at_negative_one (c : ℝ) :
  (∀ x, f c x = -(f c (-x))) → f c (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l1341_134147


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1341_134116

/-- Calculates the man's speed against the current with wind and waves -/
def speed_against_current_with_wind_and_waves 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (wind_effect : ℝ) 
  (wave_effect : ℝ) : ℝ :=
  speed_with_current - current_speed - wind_effect - current_speed - wave_effect

/-- Theorem stating the man's speed against the current with wind and waves -/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (wind_effect : ℝ) 
  (wave_effect : ℝ) 
  (h1 : speed_with_current = 20) 
  (h2 : current_speed = 5) 
  (h3 : wind_effect = 2) 
  (h4 : wave_effect = 1) : 
  speed_against_current_with_wind_and_waves speed_with_current current_speed wind_effect wave_effect = 7 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l1341_134116


namespace NUMINAMATH_CALUDE_maddy_chocolate_eggs_l1341_134140

/-- Calculates the number of chocolate eggs eaten per day -/
def chocolate_eggs_per_day (total_eggs : ℕ) (weeks : ℕ) : ℕ :=
  let days := weeks * 7
  (total_eggs + days - 1) / days

/-- Proves that given 40 chocolate eggs lasting 4 weeks, 
    eating the same number of eggs each day after school results in 1 egg per day -/
theorem maddy_chocolate_eggs : chocolate_eggs_per_day 40 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_maddy_chocolate_eggs_l1341_134140


namespace NUMINAMATH_CALUDE_max_value_is_320_l1341_134177

def operation := ℝ → ℝ → ℝ

def add : operation := λ x y => x + y
def sub : operation := λ x y => x - y
def mul : operation := λ x y => x * y

def evaluate (op1 op2 op3 op4 : operation) : ℝ :=
  op4 (op3 (op2 (op1 25 1.2) 15) 18.8) 2.3

def is_valid_operation (op : operation) : Prop :=
  op = add ∨ op = sub ∨ op = mul

theorem max_value_is_320 :
  ∀ op1 op2 op3 op4 : operation,
    is_valid_operation op1 →
    is_valid_operation op2 →
    is_valid_operation op3 →
    is_valid_operation op4 →
    evaluate op1 op2 op3 op4 ≤ 320 :=
sorry

end NUMINAMATH_CALUDE_max_value_is_320_l1341_134177


namespace NUMINAMATH_CALUDE_f_max_value_l1341_134136

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l1341_134136


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_height_ratio_l1341_134157

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The height corresponding to the hypotenuse -/
  m : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- m is positive -/
  m_pos : 0 < m
  /-- r is positive -/
  r_pos : 0 < r

/-- The ratio of the inscribed circle radius to the height is between 0.4 and 0.5 -/
theorem inscribed_circle_radius_height_ratio 
  (t : RightTriangleWithInscribedCircle) : 0.4 < t.r / t.m ∧ t.r / t.m < 0.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_height_ratio_l1341_134157


namespace NUMINAMATH_CALUDE_salary_calculation_l1341_134138

/-- The monthly salary of a man who saves 20% of his salary and can save Rs. 230 when expenses increase by 20% -/
def monthlySalary : ℝ := 1437.5

theorem salary_calculation (savings_rate : ℝ) (expense_increase : ℝ) (reduced_savings : ℝ)
    (h1 : savings_rate = 0.20)
    (h2 : expense_increase = 0.20)
    (h3 : reduced_savings = 230)
    (h4 : savings_rate * monthlySalary - expense_increase * (savings_rate * monthlySalary) = reduced_savings) :
  monthlySalary = 1437.5 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l1341_134138


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l1341_134148

/-- A circle tangent to the y-axis with center on x - 3y = 0 and passing through (6, 1) -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_y_axis : center.1 = radius
  center_on_line : center.1 - 3 * center.2 = 0
  passes_through_point : (center.1 - 6)^2 + (center.2 - 1)^2 = radius^2

/-- The equation of the circle is either (x - 3)² + (y - 1)² = 9 or (x - 111)² + (y - 37)² = 9 × 37² -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) →
  ((∀ x y, (x - 3)^2 + (y - 1)^2 = 9) ∨
   (∀ x y, (x - 111)^2 + (y - 37)^2 = 9 * 37^2)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l1341_134148


namespace NUMINAMATH_CALUDE_remainder_yards_value_l1341_134102

/-- The number of half-marathons Jacob has run -/
def num_half_marathons : ℕ := 15

/-- The length of a half-marathon in miles -/
def half_marathon_miles : ℕ := 13

/-- The additional length of a half-marathon in yards -/
def half_marathon_extra_yards : ℕ := 193

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The total distance Jacob has run in yards -/
def total_distance_yards : ℕ :=
  num_half_marathons * (half_marathon_miles * yards_per_mile + half_marathon_extra_yards)

/-- The remainder y in yards when the total distance is expressed as m miles and y yards -/
def remainder_yards : ℕ := total_distance_yards % yards_per_mile

theorem remainder_yards_value : remainder_yards = 1135 := by
  sorry

end NUMINAMATH_CALUDE_remainder_yards_value_l1341_134102


namespace NUMINAMATH_CALUDE_no_intersection_implies_k_plus_minus_one_l1341_134197

theorem no_intersection_implies_k_plus_minus_one (k : ℤ) :
  (∀ x y : ℝ, x^2 + y^2 = k^2 → x * y ≠ k) →
  k = 1 ∨ k = -1 := by
sorry

end NUMINAMATH_CALUDE_no_intersection_implies_k_plus_minus_one_l1341_134197


namespace NUMINAMATH_CALUDE_ivan_payment_l1341_134165

/-- The total amount paid for discounted Uno Giant Family Cards -/
def total_paid (original_price discount quantity : ℕ) : ℕ :=
  (original_price - discount) * quantity

/-- Theorem: Ivan paid $100 for 10 Uno Giant Family Cards with a $2 discount each -/
theorem ivan_payment :
  let original_price : ℕ := 12
  let discount : ℕ := 2
  let quantity : ℕ := 10
  total_paid original_price discount quantity = 100 := by
sorry

end NUMINAMATH_CALUDE_ivan_payment_l1341_134165


namespace NUMINAMATH_CALUDE_expected_value_S_l1341_134114

def num_boys : ℕ := 7
def num_girls : ℕ := 13
def total_people : ℕ := num_boys + num_girls

def prob_boy_girl : ℚ := (num_boys : ℚ) / total_people * (num_girls : ℚ) / (total_people - 1)
def prob_girl_boy : ℚ := (num_girls : ℚ) / total_people * (num_boys : ℚ) / (total_people - 1)

def prob_adjacent_pair : ℚ := prob_boy_girl + prob_girl_boy
def num_adjacent_pairs : ℕ := total_people - 1

theorem expected_value_S : (num_adjacent_pairs : ℚ) * prob_adjacent_pair = 91 / 10 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_S_l1341_134114


namespace NUMINAMATH_CALUDE_zuminglish_word_count_mod_500_l1341_134196

/-- Represents the alphabet of Zuminglish --/
inductive ZuminglishLetter
| M
| O
| P

/-- Represents whether a letter is a vowel or consonant --/
def isVowel (l : ZuminglishLetter) : Bool :=
  match l with
  | ZuminglishLetter.O => true
  | _ => false

/-- A Zuminglish word is a list of ZuminglishLetters --/
def ZuminglishWord := List ZuminglishLetter

/-- Check if a Zuminglish word is valid --/
def isValidWord (w : ZuminglishWord) : Bool :=
  sorry

/-- Count the number of valid 10-letter Zuminglish words --/
def countValidWords : Nat :=
  sorry

/-- The main theorem to prove --/
theorem zuminglish_word_count_mod_500 :
  countValidWords % 500 = 160 :=
sorry

end NUMINAMATH_CALUDE_zuminglish_word_count_mod_500_l1341_134196


namespace NUMINAMATH_CALUDE_complex_power_difference_l1341_134183

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference :
  (1 + 2*i)^8 - (1 - 2*i)^8 = 672*i :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1341_134183


namespace NUMINAMATH_CALUDE_marble_drawing_probability_l1341_134163

/-- Represents the total number of marbles in the bag. -/
def total_marbles : ℕ := 800

/-- Represents the number of different colors of marbles. -/
def num_colors : ℕ := 100

/-- Represents the number of marbles of each color. -/
def marbles_per_color : ℕ := 8

/-- Represents the number of marbles drawn so far. -/
def marbles_drawn : ℕ := 699

/-- Represents the target number of marbles of the same color to stop drawing. -/
def target_same_color : ℕ := 8

/-- Represents the probability of stopping after drawing the 700th marble. -/
def stop_probability : ℚ := 99 / 101

theorem marble_drawing_probability :
  total_marbles = num_colors * marbles_per_color ∧
  marbles_drawn < total_marbles ∧
  marbles_drawn ≥ (num_colors - 1) * (target_same_color - 1) + (target_same_color - 2) →
  stop_probability = 99 / 101 :=
by sorry

end NUMINAMATH_CALUDE_marble_drawing_probability_l1341_134163


namespace NUMINAMATH_CALUDE_max_sum_consecutive_integers_l1341_134118

/-- Given consecutive integers x, y, and z satisfying 1/x + 1/y + 1/z > 1/45,
    the maximum value of x + y + z is 402. -/
theorem max_sum_consecutive_integers (x y z : ℤ) :
  (y = x + 1) →
  (z = y + 1) →
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z > (1 : ℚ) / 45 →
  ∀ a b c : ℤ, (b = a + 1) → (c = b + 1) →
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c > (1 : ℚ) / 45 →
    x + y + z ≥ a + b + c →
  x + y + z = 402 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_consecutive_integers_l1341_134118


namespace NUMINAMATH_CALUDE_simplify_calculations_l1341_134146

theorem simplify_calculations :
  ((999 : ℕ) * 999 + 1999 = 1000000) ∧
  ((9 : ℕ) * 72 * 125 = 81000) ∧
  ((416 : ℤ) - 327 + 184 - 273 = 0) := by
  sorry

end NUMINAMATH_CALUDE_simplify_calculations_l1341_134146


namespace NUMINAMATH_CALUDE_faster_train_length_l1341_134101

/-- The length of a train given its speed relative to another train and the time it takes to pass --/
def train_length (relative_speed : ℝ) (passing_time : ℝ) : ℝ :=
  relative_speed * passing_time

theorem faster_train_length :
  let faster_speed : ℝ := 108 * (1000 / 3600)  -- Convert km/h to m/s
  let slower_speed : ℝ := 36 * (1000 / 3600)   -- Convert km/h to m/s
  let relative_speed : ℝ := faster_speed - slower_speed
  let passing_time : ℝ := 17
  train_length relative_speed passing_time = 340 := by
  sorry

#check faster_train_length

end NUMINAMATH_CALUDE_faster_train_length_l1341_134101


namespace NUMINAMATH_CALUDE_lcm_is_120_l1341_134149

def problem (a b : ℕ) : Prop :=
  a + b = 55 ∧
  Nat.gcd a b = 5 ∧
  (1 : ℚ) / a + (1 : ℚ) / b = 11 / 120

theorem lcm_is_120 (a b : ℕ) (h : problem a b) :
  Nat.lcm a b = 120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_is_120_l1341_134149


namespace NUMINAMATH_CALUDE_quadratic_roots_characterization_l1341_134107

/-- The quadratic equation parameterized by r -/
def quadratic (r : ℝ) (x : ℝ) : ℝ := (r - 4) * x^2 - 2 * (r - 3) * x + r

/-- The discriminant of the quadratic equation -/
def discriminant (r : ℝ) : ℝ := (-2 * (r - 3))^2 - 4 * (r - 4) * r

/-- Predicate for the quadratic having two distinct roots -/
def has_two_distinct_roots (r : ℝ) : Prop := discriminant r > 0

/-- Predicate for both roots being greater than -1 -/
def roots_greater_than_neg_one (r : ℝ) : Prop :=
  ∀ x, quadratic r x = 0 → x > -1

theorem quadratic_roots_characterization (r : ℝ) :
  (3.5 < r ∧ r < 4.5) ↔ (has_two_distinct_roots r ∧ roots_greater_than_neg_one r) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_characterization_l1341_134107


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l1341_134137

theorem arccos_equation_solution :
  ∃! x : ℝ, Real.arccos (2 * x) - Real.arccos x = π / 3 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l1341_134137


namespace NUMINAMATH_CALUDE_modified_full_house_probability_l1341_134125

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of ranks in a standard deck -/
def NumberOfRanks : ℕ := 13

/-- Represents the number of cards per rank -/
def CardsPerRank : ℕ := 4

/-- Represents the number of cards drawn -/
def CardsDrawn : ℕ := 6

/-- Represents a modified full house -/
structure ModifiedFullHouse :=
  (rank1 : Fin NumberOfRanks)
  (rank2 : Fin NumberOfRanks)
  (rank3 : Fin NumberOfRanks)
  (h1 : rank1 ≠ rank2)
  (h2 : rank1 ≠ rank3)
  (h3 : rank2 ≠ rank3)

/-- The probability of drawing a modified full house -/
def probabilityModifiedFullHouse : ℚ :=
  24 / 2977

theorem modified_full_house_probability :
  probabilityModifiedFullHouse = (NumberOfRanks * (CardsPerRank.choose 3) * (NumberOfRanks - 1) * (CardsPerRank.choose 2) * ((NumberOfRanks - 2) * CardsPerRank)) / (StandardDeck.choose CardsDrawn) :=
by sorry

end NUMINAMATH_CALUDE_modified_full_house_probability_l1341_134125


namespace NUMINAMATH_CALUDE_seashells_left_l1341_134152

def initial_seashells : ℕ := 62
def seashells_given : ℕ := 49

theorem seashells_left : initial_seashells - seashells_given = 13 := by
  sorry

end NUMINAMATH_CALUDE_seashells_left_l1341_134152


namespace NUMINAMATH_CALUDE_gcd_1458_1479_l1341_134178

theorem gcd_1458_1479 : Nat.gcd 1458 1479 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1458_1479_l1341_134178


namespace NUMINAMATH_CALUDE_bryan_bookshelves_l1341_134131

/-- Given a total number of books and books per bookshelf, calculates the number of bookshelves -/
def calculate_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books / books_per_shelf

/-- Theorem stating that with 504 total books and 56 books per shelf, there are 9 bookshelves -/
theorem bryan_bookshelves :
  calculate_bookshelves 504 56 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bryan_bookshelves_l1341_134131


namespace NUMINAMATH_CALUDE_lipstick_ratio_l1341_134170

/-- Proves that the ratio of students wearing blue lipstick to those wearing red lipstick is 1:5 -/
theorem lipstick_ratio (total_students : ℕ) (blue_lipstick : ℕ) 
  (h1 : total_students = 200)
  (h2 : blue_lipstick = 5)
  (h3 : 2 * (total_students / 2) = total_students)  -- Half of students wore lipstick
  (h4 : 4 * (total_students / 2 / 4) = total_students / 2)  -- Quarter of lipstick wearers wore red
  (h5 : blue_lipstick = total_students / 2 / 4)  -- Same number wore blue as red
  : (blue_lipstick : ℚ) / (total_students / 2 / 4 : ℚ) = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_lipstick_ratio_l1341_134170


namespace NUMINAMATH_CALUDE_sum_of_distances_is_ten_l1341_134158

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the focus F
def focus : ℝ × ℝ := (3, 0)

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define that P is on the line AB
def P_on_line_AB (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = (1 - t) • A + t • B

-- Define that P is the midpoint of AB
def P_is_midpoint (A B : ℝ × ℝ) : Prop :=
  P = (A + B) / 2

-- Define that A and B are on the parabola
def points_on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2

-- State the theorem
theorem sum_of_distances_is_ten (A B : ℝ × ℝ) 
  (h1 : P_on_line_AB A B)
  (h2 : P_is_midpoint A B)
  (h3 : points_on_parabola A B) :
  dist A focus + dist B focus = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_distances_is_ten_l1341_134158


namespace NUMINAMATH_CALUDE_smallest_a_for_divisibility_l1341_134155

theorem smallest_a_for_divisibility : 
  (∃ (a : ℕ), a > 0 ∧ 
    (∃ (n : ℕ), n > 0 ∧ Odd n ∧ 
      (2001 ∣ 55^n + a * 32^n))) ∧ 
  (∀ (a : ℕ), a > 0 → 
    (∃ (n : ℕ), n > 0 ∧ Odd n ∧ 
      (2001 ∣ 55^n + a * 32^n)) → 
    a ≥ 436) := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_for_divisibility_l1341_134155


namespace NUMINAMATH_CALUDE_men_in_second_scenario_l1341_134135

/-- Calculates the number of men working in the second scenario given the conditions --/
theorem men_in_second_scenario 
  (hours_per_day_first : ℕ) 
  (hours_per_day_second : ℕ)
  (men_first : ℕ)
  (earnings_first : ℚ)
  (earnings_second : ℚ)
  (days_per_week : ℕ) :
  hours_per_day_first = 10 →
  hours_per_day_second = 6 →
  men_first = 4 →
  earnings_first = 1400 →
  earnings_second = 1890.0000000000002 →
  days_per_week = 7 →
  ∃ (men_second : ℕ), men_second = 9 ∧ 
    (men_second * hours_per_day_second * days_per_week : ℚ) * 
    (earnings_first / (men_first * hours_per_day_first * days_per_week : ℚ)) = 
    earnings_second :=
by sorry

end NUMINAMATH_CALUDE_men_in_second_scenario_l1341_134135


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1341_134188

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1341_134188


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l1341_134111

def N : ℕ := 18 * 52 * 75 * 98

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors N) * 30 = sum_even_divisors N :=
sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l1341_134111


namespace NUMINAMATH_CALUDE_only_points_in_circle_form_set_l1341_134145

-- Define a type for the objects in question
inductive Object
| MaleStudents
| DifficultProblems
| OutgoingGirls
| PointsInCircle

-- Define a predicate for whether an object can form a set
def CanFormSet (obj : Object) : Prop :=
  match obj with
  | Object.PointsInCircle => True
  | _ => False

-- State the theorem
theorem only_points_in_circle_form_set :
  ∀ (obj : Object), CanFormSet obj ↔ obj = Object.PointsInCircle :=
by sorry

end NUMINAMATH_CALUDE_only_points_in_circle_form_set_l1341_134145


namespace NUMINAMATH_CALUDE_acute_triangle_existence_l1341_134103

theorem acute_triangle_existence (d : Fin 12 → ℝ) 
  (h_range : ∀ i, 1 < d i ∧ d i < 12) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (d i < d j + d k) ∧ 
    (d j < d i + d k) ∧ 
    (d k < d i + d j) ∧
    (d i)^2 + (d j)^2 > (d k)^2 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_existence_l1341_134103


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_5_l1341_134171

theorem smallest_perfect_square_divisible_by_2_and_5 : ∃ n : ℕ, 
  n > 0 ∧ 
  (∃ m : ℕ, n = m ^ 2) ∧ 
  n % 2 = 0 ∧ 
  n % 5 = 0 ∧
  (∀ k : ℕ, k > 0 → (∃ m : ℕ, k = m ^ 2) → k % 2 = 0 → k % 5 = 0 → k ≥ n) ∧
  n = 100 := by
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_5_l1341_134171


namespace NUMINAMATH_CALUDE_bacteria_population_after_15_days_l1341_134166

/-- Calculates the population of bacteria cells after a given number of days -/
def bacteriaPopulation (initialCells : ℕ) (daysPerDivision : ℕ) (totalDays : ℕ) : ℕ :=
  initialCells * (3 ^ (totalDays / daysPerDivision))

/-- Theorem stating that the bacteria population after 15 days is 1215 cells -/
theorem bacteria_population_after_15_days :
  bacteriaPopulation 5 3 15 = 1215 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_population_after_15_days_l1341_134166


namespace NUMINAMATH_CALUDE_part_one_part_two_l1341_134181

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |x - a^2|

-- Define the solution set for part (1)
def solution_set : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

-- Theorem for part (1)
theorem part_one : 
  ∀ x, f x 1 ≥ 4 ↔ x ∈ solution_set :=
sorry

-- Theorem for part (2)
theorem part_two :
  (∀ x, ∃ a ∈ Set.Ioo (-1) 3, m < f x a) → m < 12 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1341_134181


namespace NUMINAMATH_CALUDE_initial_money_calculation_l1341_134123

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 300 →
  initial_money = 750 := by
sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l1341_134123


namespace NUMINAMATH_CALUDE_baseball_card_pages_l1341_134134

theorem baseball_card_pages (cards_per_page new_cards old_cards : ℕ) 
  (h1 : cards_per_page = 3)
  (h2 : new_cards = 8)
  (h3 : old_cards = 10) :
  (new_cards + old_cards) / cards_per_page = 6 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_pages_l1341_134134


namespace NUMINAMATH_CALUDE_triangle_balls_proof_l1341_134150

/-- The number of balls in an equilateral triangle arrangement -/
def triangle_balls : ℕ := 820

/-- The number of balls added to form a square -/
def added_balls : ℕ := 424

/-- The difference in side length between the triangle and the square -/
def side_difference : ℕ := 8

/-- Formula for the sum of the first n natural numbers -/
def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The side length of the triangle -/
def triangle_side : ℕ := 40

/-- The side length of the square -/
def square_side : ℕ := triangle_side - side_difference

theorem triangle_balls_proof :
  triangle_balls = triangle_sum triangle_side ∧
  triangle_balls + added_balls = square_side^2 ∧
  triangle_side = square_side + side_difference :=
sorry

end NUMINAMATH_CALUDE_triangle_balls_proof_l1341_134150


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l1341_134176

-- Define the function f and its derivative
noncomputable def f : ℝ → ℝ := sorry

-- State the derivative condition
axiom f_derivative (x : ℝ) : deriv f x = x * (1 - x)

-- Theorem statement
theorem f_monotone_increasing : MonotoneOn f (Set.Icc 0 1) := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l1341_134176


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1341_134141

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan (x^2 * Real.sin (1 / (5 * x)))
  else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1341_134141


namespace NUMINAMATH_CALUDE_consecutive_binomial_ratio_l1341_134154

theorem consecutive_binomial_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k+1) : ℚ) = 1/3 ∧
  (n.choose (k+1) : ℚ) / (n.choose (k+2) : ℚ) = 3/5 →
  n + k = 8 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_binomial_ratio_l1341_134154


namespace NUMINAMATH_CALUDE_number_equation_solution_l1341_134162

theorem number_equation_solution : ∃ x : ℝ, 5 * x + 4 = 19 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1341_134162


namespace NUMINAMATH_CALUDE_ali_total_money_l1341_134153

def five_dollar_bills : ℕ := 7
def ten_dollar_bills : ℕ := 1
def five_dollar_value : ℕ := 5
def ten_dollar_value : ℕ := 10

theorem ali_total_money :
  five_dollar_bills * five_dollar_value + ten_dollar_bills * ten_dollar_value = 45 := by
sorry

end NUMINAMATH_CALUDE_ali_total_money_l1341_134153


namespace NUMINAMATH_CALUDE_square_sum_value_l1341_134122

theorem square_sum_value (a b : ℝ) : 
  (a^2 + b^2 + 2) * (a^2 + b^2 - 2) = 45 → a^2 + b^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l1341_134122


namespace NUMINAMATH_CALUDE_tan_identities_l1341_134161

theorem tan_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_identities_l1341_134161


namespace NUMINAMATH_CALUDE_range_of_a_l1341_134128

/-- The range of a given the conditions in the problem -/
theorem range_of_a (x a : ℝ) : 
  (∀ x, (1/2 ≤ x ∧ x ≤ 1) ↔ ¬((x < 1/2) ∨ (1 < x))) →
  (∀ x, ((x - a) * (x - a - 1) ≤ 0) ↔ (a ≤ x ∧ x ≤ a + 1)) →
  (∀ x, ¬((1/2 ≤ x ∧ x ≤ 1)) → ¬((x - a) * (x - a - 1) ≤ 0)) →
  (∃ x, ¬((1/2 ≤ x ∧ x ≤ 1)) ∧ ((x - a) * (x - a - 1) ≤ 0)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1341_134128


namespace NUMINAMATH_CALUDE_correct_observation_value_l1341_134195

theorem correct_observation_value
  (n : ℕ)
  (initial_mean : ℝ)
  (wrong_value : ℝ)
  (corrected_mean : ℝ)
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : wrong_value = 21)
  (h4 : corrected_mean = 36.54)
  : ∃ (correct_value : ℝ),
    n * corrected_mean = n * initial_mean - wrong_value + correct_value ∧
    correct_value = 48 :=
by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l1341_134195


namespace NUMINAMATH_CALUDE_tree_growth_rate_l1341_134142

def initial_height : ℝ := 600
def final_height : ℝ := 720
def growth_period : ℝ := 240

theorem tree_growth_rate :
  (final_height - initial_height) / growth_period = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_rate_l1341_134142


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_l1341_134100

/-- A quadratic function with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - (k + 1) * x + k + 4

/-- The discriminant of the quadratic function -/
def discriminant (k : ℝ) : ℝ := (k + 1)^2 - 4 * (k + 4)

/-- The function has two distinct zeros iff the discriminant is positive -/
def has_two_distinct_zeros (k : ℝ) : Prop := discriminant k > 0

/-- The range of k for which the function has two distinct zeros -/
def k_range (k : ℝ) : Prop := k < -3 ∨ k > 5

theorem quadratic_two_zeros (k : ℝ) : 
  has_two_distinct_zeros k ↔ k_range k := by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_l1341_134100


namespace NUMINAMATH_CALUDE_x_plus_y_values_l1341_134109

theorem x_plus_y_values (x y : ℝ) 
  (hx : |x| = 2) 
  (hy : |y| = 5) 
  (hxy : x * y < 0) : 
  (x + y = -3) ∨ (x + y = 3) := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l1341_134109


namespace NUMINAMATH_CALUDE_add_5_16_base8_l1341_134127

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Adds two base-8 numbers and returns the result in base-8 --/
def addBase8 (a b : ℕ) : ℕ :=
  base10ToBase8 (base8ToBase10 a + base8ToBase10 b)

theorem add_5_16_base8 :
  addBase8 5 16 = 23 := by sorry

end NUMINAMATH_CALUDE_add_5_16_base8_l1341_134127


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1341_134156

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 := by
  sorry

theorem problem_solution : 
  let n := 568219
  let d := 89
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 ∧ k = 45 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1341_134156


namespace NUMINAMATH_CALUDE_min_expression_less_equal_one_l1341_134106

theorem min_expression_less_equal_one (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 3) : 
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_expression_less_equal_one_l1341_134106


namespace NUMINAMATH_CALUDE_right_triangles_count_l1341_134119

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Checks if a triangle is right-angled with vertex c as the right angle -/
def isRightTriangle (t : Triangle) : Prop := sorry

/-- Checks if a point is a lattice point (has integer coordinates) -/
def isLatticePoint (p : Point) : Prop := sorry

/-- Calculates the incenter of a triangle -/
def incenter (t : Triangle) : Point := sorry

/-- Counts the number of right triangles satisfying the given conditions -/
def countRightTriangles (p : ℕ) (isPrime : Nat.Prime p) : ℕ := sorry

/-- The main theorem -/
theorem right_triangles_count (p : ℕ) (isPrime : Nat.Prime p) :
  let m := Point.mk (p * 1994) (7 * p * 1994)
  countRightTriangles p isPrime =
    if p = 2 then 18
    else if p = 997 then 20
    else 36 := by
  sorry

end NUMINAMATH_CALUDE_right_triangles_count_l1341_134119


namespace NUMINAMATH_CALUDE_probability_JQKA_standard_deck_l1341_134179

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards of each rank (Jack, Queen, King, Ace) in a standard deck -/
def CardsPerRank : ℕ := 4

/-- Calculates the probability of drawing a specific sequence of four cards (Jack, Queen, King, Ace) from a standard deck without replacement -/
def probability_JQKA (deck : ℕ) (cards_per_rank : ℕ) : ℚ :=
  (cards_per_rank : ℚ) / deck *
  (cards_per_rank : ℚ) / (deck - 1) *
  (cards_per_rank : ℚ) / (deck - 2) *
  (cards_per_rank : ℚ) / (deck - 3)

/-- Theorem stating that the probability of drawing Jack, Queen, King, Ace in order from a standard deck without replacement is 64/1624350 -/
theorem probability_JQKA_standard_deck :
  probability_JQKA StandardDeck CardsPerRank = 64 / 1624350 := by
  sorry

end NUMINAMATH_CALUDE_probability_JQKA_standard_deck_l1341_134179


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1341_134130

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x - 3| :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1341_134130


namespace NUMINAMATH_CALUDE_tan_equation_solution_set_l1341_134174

theorem tan_equation_solution_set :
  {x : Real | Real.tan x = 2} = {x : Real | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2} := by sorry

end NUMINAMATH_CALUDE_tan_equation_solution_set_l1341_134174


namespace NUMINAMATH_CALUDE_height_ratio_equals_similarity_ratio_l1341_134115

/-- Two similar triangles with heights h₁ and h₂ and similarity ratio 1:4 -/
structure SimilarTriangles where
  h₁ : ℝ
  h₂ : ℝ
  h₁_pos : h₁ > 0
  h₂_pos : h₂ > 0
  similarity_ratio : h₁ / h₂ = 1 / 4

/-- The ratio of heights of similar triangles with similarity ratio 1:4 is 1:4 -/
theorem height_ratio_equals_similarity_ratio (t : SimilarTriangles) :
  t.h₁ / t.h₂ = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_height_ratio_equals_similarity_ratio_l1341_134115


namespace NUMINAMATH_CALUDE_savings_difference_is_75_cents_l1341_134189

/-- The price of the book in dollars -/
def book_price : ℚ := 30

/-- The fixed discount amount in dollars -/
def fixed_discount : ℚ := 5

/-- The percentage discount as a decimal -/
def percent_discount : ℚ := 0.15

/-- The cost after applying the fixed discount first, then the percentage discount -/
def cost_fixed_first : ℚ := (book_price - fixed_discount) * (1 - percent_discount)

/-- The cost after applying the percentage discount first, then the fixed discount -/
def cost_percent_first : ℚ := book_price * (1 - percent_discount) - fixed_discount

/-- The difference in savings between the two discount sequences, in cents -/
def savings_difference : ℚ := (cost_fixed_first - cost_percent_first) * 100

theorem savings_difference_is_75_cents : savings_difference = 75 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_is_75_cents_l1341_134189
