import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2844_284449

/-- Given a line L1 with equation x - y - 2 = 0 and a point A (2, 6),
    prove that the line L2 with equation x + y - 8 = 0 passes through A
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point 
  (L1 : Set (ℝ × ℝ)) 
  (A : ℝ × ℝ) :
  let L2 := {(x, y) : ℝ × ℝ | x + y - 8 = 0}
  (∀ (x y : ℝ), (x, y) ∈ L1 ↔ x - y - 2 = 0) →
  A = (2, 6) →
  A ∈ L2 ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁, y₁) ∈ L1 → (x₂, y₂) ∈ L1 → x₁ ≠ x₂ →
    (x₁ - x₂) * (2 - 2) + (y₁ - y₂) * (6 - 6) = 0) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2844_284449


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l2844_284461

theorem power_fraction_simplification :
  ((2^5) * (9^2)) / ((8^2) * (3^5)) = 1/6 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l2844_284461


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2844_284454

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2844_284454


namespace NUMINAMATH_CALUDE_sqrt_6_equality_l2844_284420

theorem sqrt_6_equality : (3 : ℝ) / Real.sqrt 6 = Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_6_equality_l2844_284420


namespace NUMINAMATH_CALUDE_number_of_petri_dishes_l2844_284481

/-- The number of petri dishes in a lab, given the total number of germs and germs per dish -/
theorem number_of_petri_dishes 
  (total_germs : ℝ) 
  (germs_per_dish : ℝ) 
  (h1 : total_germs = 0.036 * 10^5)
  (h2 : germs_per_dish = 47.99999999999999)
  : ℤ :=
75

#check number_of_petri_dishes

end NUMINAMATH_CALUDE_number_of_petri_dishes_l2844_284481


namespace NUMINAMATH_CALUDE_books_left_l2844_284422

/-- Given that Paul had 242 books initially and sold 137 books, prove that he has 105 books left. -/
theorem books_left (initial_books : ℕ) (sold_books : ℕ) (h1 : initial_books = 242) (h2 : sold_books = 137) :
  initial_books - sold_books = 105 := by
  sorry

end NUMINAMATH_CALUDE_books_left_l2844_284422


namespace NUMINAMATH_CALUDE_digits_after_decimal_point_of_fraction_l2844_284450

/-- The number of digits to the right of the decimal point when 5^8 / (10^6 * 16) is expressed as a decimal is 3. -/
theorem digits_after_decimal_point_of_fraction : ∃ (n : ℕ) (d : ℕ+), 
  5^8 / (10^6 * 16) = n / d ∧ 
  (∃ (k : ℕ), 10^3 * (n / d) = k ∧ 10^2 * (n / d) < 1) :=
by sorry

end NUMINAMATH_CALUDE_digits_after_decimal_point_of_fraction_l2844_284450


namespace NUMINAMATH_CALUDE_sqrt_inequality_equivalence_l2844_284435

theorem sqrt_inequality_equivalence :
  (Real.sqrt 2 - Real.sqrt 3 < Real.sqrt 6 - Real.sqrt 7) ↔
  ((Real.sqrt 2 + Real.sqrt 7)^2 < (Real.sqrt 6 + Real.sqrt 3)^2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_equivalence_l2844_284435


namespace NUMINAMATH_CALUDE_square_plus_integer_l2844_284423

theorem square_plus_integer (y : ℝ) : y^2 + 14*y + 48 = (y+7)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_integer_l2844_284423


namespace NUMINAMATH_CALUDE_park_oaks_l2844_284406

/-- The number of huge ancient oaks in a park -/
def huge_ancient_oaks (total_trees medium_firs saplings : ℕ) : ℕ :=
  total_trees - medium_firs - saplings

/-- Theorem: There are 15 huge ancient oaks in the park -/
theorem park_oaks : huge_ancient_oaks 96 23 58 = 15 := by
  sorry

end NUMINAMATH_CALUDE_park_oaks_l2844_284406


namespace NUMINAMATH_CALUDE_first_day_is_saturday_l2844_284436

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  saturdays : Nat
  sundays : Nat

/-- Theorem: In a 30-day month with 5 Saturdays and 5 Sundays, the first day is Saturday -/
theorem first_day_is_saturday (m : Month) (h1 : m.days = 30) (h2 : m.saturdays = 5) (h3 : m.sundays = 5) :
  ∃ (first_day : DayOfWeek), first_day = DayOfWeek.Saturday := by
  sorry


end NUMINAMATH_CALUDE_first_day_is_saturday_l2844_284436


namespace NUMINAMATH_CALUDE_factorization_equality_l2844_284403

theorem factorization_equality (p : ℝ) : (p - 4) * (p + 1) + 3 * p = (p + 2) * (p - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2844_284403


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2844_284466

theorem isosceles_right_triangle_area (side_length : ℝ) : 
  side_length = 12 →
  ∃ (r s : ℝ), 
    r > 0 ∧ s > 0 ∧
    2 * (r ^ 2 + s ^ 2) = side_length ^ 2 ∧
    4 * (r ^ 2 / 2) = 72 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2844_284466


namespace NUMINAMATH_CALUDE_max_abs_z_on_circle_l2844_284433

open Complex

theorem max_abs_z_on_circle (z : ℂ) (h : abs (z - (3 + 4*I)) = 1) : 
  (∀ w : ℂ, abs (w - (3 + 4*I)) = 1 → abs w ≤ abs z) → abs z = 6 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_on_circle_l2844_284433


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l2844_284434

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l2844_284434


namespace NUMINAMATH_CALUDE_percentage_on_rent_is_14_l2844_284457

/-- Calculates the percentage of remaining income spent on house rent --/
def percentage_on_rent (total_income : ℚ) (petrol_expense : ℚ) (rent_expense : ℚ) : ℚ :=
  let remaining_income := total_income - petrol_expense
  (rent_expense / remaining_income) * 100

/-- Theorem: Given the conditions, the percentage spent on house rent is 14% --/
theorem percentage_on_rent_is_14 :
  ∀ (total_income : ℚ),
  total_income > 0 →
  total_income * (30 / 100) = 300 →
  percentage_on_rent total_income 300 98 = 14 := by
  sorry

#eval percentage_on_rent 1000 300 98

end NUMINAMATH_CALUDE_percentage_on_rent_is_14_l2844_284457


namespace NUMINAMATH_CALUDE_expression_evaluation_l2844_284460

theorem expression_evaluation (b x : ℝ) (h : x = b + 9) :
  2*x - b + 5 = b + 23 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2844_284460


namespace NUMINAMATH_CALUDE_daughter_weight_l2844_284470

/-- Proves that the weight of the daughter is 48 kg given the conditions of the problem -/
theorem daughter_weight (M D C : ℝ) 
  (total_weight : M + D + C = 120)
  (daughter_child_weight : D + C = 60)
  (child_grandmother_ratio : C = (1/5) * M) :
  D = 48 := by sorry

end NUMINAMATH_CALUDE_daughter_weight_l2844_284470


namespace NUMINAMATH_CALUDE_harry_sister_stamps_l2844_284456

theorem harry_sister_stamps (total : ℕ) (harry_ratio : ℕ) (sister_stamps : ℕ) : 
  total = 240 → 
  harry_ratio = 3 → 
  sister_stamps + harry_ratio * sister_stamps = total → 
  sister_stamps = 60 := by
sorry

end NUMINAMATH_CALUDE_harry_sister_stamps_l2844_284456


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2844_284485

theorem algebraic_expression_value : ∀ a : ℝ, a^2 + a = 3 → 2*a^2 + 2*a - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2844_284485


namespace NUMINAMATH_CALUDE_f_one_less_than_f_two_necessary_not_sufficient_l2844_284467

def increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem f_one_less_than_f_two_necessary_not_sufficient :
  ∃ f : ℝ → ℝ, (increasing f → f 1 < f 2) ∧
  ¬(f 1 < f 2 → increasing f) :=
by sorry

end NUMINAMATH_CALUDE_f_one_less_than_f_two_necessary_not_sufficient_l2844_284467


namespace NUMINAMATH_CALUDE_fidos_yard_exploration_l2844_284429

theorem fidos_yard_exploration (s : ℝ) (s_pos : s > 0) :
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let circle_area := π * s^2
  let ratio := circle_area / hexagon_area
  ∃ (a b : ℕ), (a > 0 ∧ b > 0) ∧ 
    ratio = Real.sqrt a / b * π ∧
    a * b = 27 := by
  sorry

end NUMINAMATH_CALUDE_fidos_yard_exploration_l2844_284429


namespace NUMINAMATH_CALUDE_digit_equation_solution_l2844_284477

theorem digit_equation_solution :
  ∃! (A B C D : ℕ),
    (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) ∧
    ((A + B * Real.sqrt 3) ^ 2 = (10 * C + D) + (10 * B + C) * Real.sqrt 3) ∧
    ((A + C * Real.sqrt 3) ^ 2 = (10 * D + C) + (10 * C + D) * Real.sqrt 3) ∧
    A = 6 ∧ B = 2 ∧ C = 4 ∧ D = 8 :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l2844_284477


namespace NUMINAMATH_CALUDE_line_slope_and_angle_l2844_284428

/-- Theorem: For a line passing through points (-2,3) and (-1,2), its slope is -1
    and the angle it makes with the positive x-axis is 3π/4 -/
theorem line_slope_and_angle :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (-1, 2)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let angle : ℝ := Real.arctan slope
  slope = -1 ∧ angle = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_and_angle_l2844_284428


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l2844_284463

theorem largest_prime_factors_difference (n : Nat) (h : n = 261943) :
  ∃ (p q : Nat), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p > q ∧ 
    (∀ r : Nat, Nat.Prime r ∧ r ∣ n → r ≤ p) ∧
    (∀ r : Nat, Nat.Prime r ∧ r ∣ n ∧ r ≠ p → r ≤ q) ∧
    p - q = 110 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l2844_284463


namespace NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l2844_284421

-- Define the expression under the cube root
def radicand (x y z : ℝ) : ℝ := 40 * x^5 * y^9 * z^14

-- Define the function to calculate the sum of exponents outside the radical
def sum_of_exponents_outside_radical (x y z : ℝ) : ℕ :=
  let simplified := (radicand x y z)^(1/3)
  -- The actual calculation of exponents would be implemented here
  -- For now, we'll use a placeholder
  8

-- Theorem statement
theorem sum_of_exponents_is_eight (x y z : ℝ) :
  sum_of_exponents_outside_radical x y z = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l2844_284421


namespace NUMINAMATH_CALUDE_choir_average_age_l2844_284462

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (total_people : ℕ) 
  (h1 : num_females = 10) 
  (h2 : num_males = 15) 
  (h3 : avg_age_females = 30) 
  (h4 : avg_age_males = 35) 
  (h5 : total_people = num_females + num_males) :
  (num_females * avg_age_females + num_males * avg_age_males) / total_people = 33 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l2844_284462


namespace NUMINAMATH_CALUDE_original_price_calculation_shirt_price_proof_l2844_284405

/-- 
Given two successive discounts and a final sale price, 
calculate the original price of an item.
-/
theorem original_price_calculation 
  (discount1 : ℝ) 
  (discount2 : ℝ) 
  (final_price : ℝ) : ℝ :=
  let remaining_factor1 := 1 - discount1
  let remaining_factor2 := 1 - discount2
  let original_price := final_price / (remaining_factor1 * remaining_factor2)
  original_price

/-- 
Prove that given discounts of 15% and 2%, 
if the final sale price is 830, 
then the original price is approximately 996.40.
-/
theorem shirt_price_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (original_price_calculation 0.15 0.02 830 - 996.40) < ε :=
sorry

end NUMINAMATH_CALUDE_original_price_calculation_shirt_price_proof_l2844_284405


namespace NUMINAMATH_CALUDE_expression_simplification_l2844_284437

theorem expression_simplification (a x : ℝ) (h : a ≠ 3*x) :
  1.4 * (3*a^2 + 2*a*x - x^2) / ((3*x + a)*(a + x)) - 2 + 10 * (a*x - 3*x^2) / (a^2 + 9*x^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2844_284437


namespace NUMINAMATH_CALUDE_john_number_is_13_l2844_284441

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem john_number_is_13 :
  ∃! x : ℕ, is_two_digit x ∧
    92 ≤ switch_digits (4 * x + 17) ∧
    switch_digits (4 * x + 17) ≤ 96 ∧
    x = 13 :=
by sorry

end NUMINAMATH_CALUDE_john_number_is_13_l2844_284441


namespace NUMINAMATH_CALUDE_percentage_calculation_l2844_284490

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 6000 → 
  P / 100 * (30 / 100) * (50 / 100) * N = 90 → 
  P = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2844_284490


namespace NUMINAMATH_CALUDE_bus_arrival_time_difference_l2844_284474

/-- Proves that a person walking to a bus stand will arrive 10 minutes early when doubling their speed -/
theorem bus_arrival_time_difference (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (miss_time : ℝ) : 
  distance = 2.2 →
  speed1 = 3 →
  speed2 = 6 →
  miss_time = 12 →
  (distance / speed2 * 60) = ((distance / speed1 * 60) - miss_time) - 10 :=
by sorry

end NUMINAMATH_CALUDE_bus_arrival_time_difference_l2844_284474


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_six_l2844_284447

theorem sum_of_solutions_is_six : 
  ∃ (x₁ x₂ : ℂ), 
    (2 : ℂ) ^ (x₁^2 - 3*x₁ - 2) = (8 : ℂ) ^ (x₁ - 5) ∧
    (2 : ℂ) ^ (x₂^2 - 3*x₂ - 2) = (8 : ℂ) ^ (x₂ - 5) ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 6 ∧
    ∀ (y : ℂ), (2 : ℂ) ^ (y^2 - 3*y - 2) = (8 : ℂ) ^ (y - 5) → y = x₁ ∨ y = x₂ :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_six_l2844_284447


namespace NUMINAMATH_CALUDE_intersection_M_N_l2844_284404

-- Define set M
def M : Set ℝ := {x | x^2 - x ≤ 0}

-- Define set N (domain of log|x|)
def N : Set ℝ := {x | x ≠ 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2844_284404


namespace NUMINAMATH_CALUDE_cupcake_price_is_two_l2844_284455

/-- Calculates the price per cupcake given the number of trays, cupcakes per tray,
    fraction of cupcakes sold, and total earnings. -/
def price_per_cupcake (num_trays : ℕ) (cupcakes_per_tray : ℕ) 
                      (fraction_sold : ℚ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (fraction_sold * (num_trays * cupcakes_per_tray))

/-- Proves that the price per cupcake is $2 given the specific conditions. -/
theorem cupcake_price_is_two :
  price_per_cupcake 4 20 (3/5) 96 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_price_is_two_l2844_284455


namespace NUMINAMATH_CALUDE_simplify_absolute_expression_l2844_284432

theorem simplify_absolute_expression : |(-4^2 + 6^2 - 2)| = 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_expression_l2844_284432


namespace NUMINAMATH_CALUDE_spencer_walking_distance_l2844_284497

/-- The total distance walked by Spencer -/
def total_distance (initial_distance : ℝ) (first_segment : ℝ) (second_segment : ℝ) : ℝ :=
  first_segment + second_segment + initial_distance

/-- Theorem: Spencer's total walking distance is 1400 meters -/
theorem spencer_walking_distance :
  let initial_distance : ℝ := 1000
  let first_segment : ℝ := 200
  let second_segment : ℝ := 200
  total_distance initial_distance first_segment second_segment = 1400 := by
  sorry

#eval total_distance 1000 200 200

end NUMINAMATH_CALUDE_spencer_walking_distance_l2844_284497


namespace NUMINAMATH_CALUDE_cornelia_european_countries_l2844_284478

/-- Represents the number of countries visited in different regions --/
structure CountriesVisited where
  total : Nat
  southAmerica : Nat
  asia : Nat

/-- Calculates the number of European countries visited --/
def europeanCountries (c : CountriesVisited) : Nat :=
  c.total - c.southAmerica - 2 * c.asia

/-- Theorem stating that Cornelia visited 20 European countries --/
theorem cornelia_european_countries :
  ∃ c : CountriesVisited, c.total = 42 ∧ c.southAmerica = 10 ∧ c.asia = 6 ∧ europeanCountries c = 20 := by
  sorry

end NUMINAMATH_CALUDE_cornelia_european_countries_l2844_284478


namespace NUMINAMATH_CALUDE_binary_10111_equals_23_l2844_284483

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_10111_equals_23 :
  binary_to_decimal [true, true, true, false, true] = 23 := by
  sorry

end NUMINAMATH_CALUDE_binary_10111_equals_23_l2844_284483


namespace NUMINAMATH_CALUDE_father_daughter_ages_l2844_284431

theorem father_daughter_ages (father_age daughter_age : ℕ) : 
  father_age = 4 * daughter_age ∧ father_age = daughter_age + 30 →
  father_age = 40 ∧ daughter_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_father_daughter_ages_l2844_284431


namespace NUMINAMATH_CALUDE_product_simplification_l2844_284482

theorem product_simplification (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l2844_284482


namespace NUMINAMATH_CALUDE_thermodynamic_expansion_l2844_284472

/-- First law of thermodynamics --/
def first_law (Q Δu A : ℝ) : Prop := Q = Δu + A

/-- Ideal gas law --/
def ideal_gas_law (P V R T : ℝ) : Prop := P * V = R * T

theorem thermodynamic_expansion 
  (Q Δu A cᵥ T T₀ k x P S n R P₀ V₀ : ℝ) 
  (h_Q : Q = 0)
  (h_Δu : Δu = cᵥ * (T - T₀))
  (h_A : A = (k * x^2) / 2)
  (h_kx : k * x = P * S)
  (h_V : S * x = V₀ * (n - 1) / n)
  (h_first_law : first_law Q Δu A)
  (h_ideal_gas_initial : ideal_gas_law P₀ V₀ R T₀)
  (h_ideal_gas_final : ideal_gas_law P (n * V₀) R T)
  (h_positive : cᵥ > 0 ∧ n > 1 ∧ R > 0 ∧ T₀ > 0 ∧ P₀ > 0) :
  P = P₀ / (n * (1 + ((n - 1) * R) / (2 * n * cᵥ))) :=
sorry

end NUMINAMATH_CALUDE_thermodynamic_expansion_l2844_284472


namespace NUMINAMATH_CALUDE_alphabet_dot_no_line_l2844_284499

theorem alphabet_dot_no_line (total : ℕ) (both : ℕ) (line_no_dot : ℕ) 
  (h1 : total = 50)
  (h2 : both = 16)
  (h3 : line_no_dot = 30)
  (h4 : total = both + line_no_dot + (total - (both + line_no_dot))) :
  total - (both + line_no_dot) = 4 := by
sorry

end NUMINAMATH_CALUDE_alphabet_dot_no_line_l2844_284499


namespace NUMINAMATH_CALUDE_bus_passengers_l2844_284453

theorem bus_passengers (total : ℕ) 
  (h1 : 3 * total = 5 * (total / 5 * 3))  -- 3/5 of total are Dutch
  (h2 : (total / 5 * 3) / 2 * 2 = total / 5 * 3)  -- 1/2 of Dutch are American
  (h3 : ((total / 5 * 3) / 2) / 3 * 3 = (total / 5 * 3) / 2)  -- 1/3 of Dutch Americans got window seats
  (h4 : ((total / 5 * 3) / 2) / 3 = 9)  -- Number of Dutch Americans at windows is 9
  : total = 90 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l2844_284453


namespace NUMINAMATH_CALUDE_system_solution_quadratic_expression_l2844_284426

theorem system_solution_quadratic_expression :
  ∀ x y z : ℚ,
  (2 * x + 3 * y + z = 20) →
  (x + 2 * y + 3 * z = 26) →
  (3 * x + y + 2 * z = 29) →
  ∃ k : ℚ, 12 * x^2 + 22 * x * y + 12 * y^2 + 12 * x * z + 12 * y * z + 12 * z^2 = k :=
by
  sorry


end NUMINAMATH_CALUDE_system_solution_quadratic_expression_l2844_284426


namespace NUMINAMATH_CALUDE_revenue_ratio_theorem_l2844_284444

/-- Represents the revenue data for a product line -/
structure ProductLine where
  lastYearRevenue : ℝ
  projectedIncrease : ℝ
  actualDecrease : ℝ

/-- Calculates the projected revenue for a product line -/
def projectedRevenue (p : ProductLine) : ℝ :=
  p.lastYearRevenue * (1 + p.projectedIncrease)

/-- Calculates the actual revenue for a product line -/
def actualRevenue (p : ProductLine) : ℝ :=
  p.lastYearRevenue * (1 - p.actualDecrease)

/-- Theorem stating that the ratio of total actual revenue to total projected revenue
    is approximately 0.5276 for the given product lines -/
theorem revenue_ratio_theorem (standardGum sugarFreeGum bubbleGum : ProductLine)
    (h1 : standardGum.lastYearRevenue = 100000)
    (h2 : standardGum.projectedIncrease = 0.3)
    (h3 : standardGum.actualDecrease = 0.2)
    (h4 : sugarFreeGum.lastYearRevenue = 150000)
    (h5 : sugarFreeGum.projectedIncrease = 0.5)
    (h6 : sugarFreeGum.actualDecrease = 0.3)
    (h7 : bubbleGum.lastYearRevenue = 200000)
    (h8 : bubbleGum.projectedIncrease = 0.4)
    (h9 : bubbleGum.actualDecrease = 0.25) :
    let totalActualRevenue := actualRevenue standardGum + actualRevenue sugarFreeGum + actualRevenue bubbleGum
    let totalProjectedRevenue := projectedRevenue standardGum + projectedRevenue sugarFreeGum + projectedRevenue bubbleGum
    abs (totalActualRevenue / totalProjectedRevenue - 0.5276) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_revenue_ratio_theorem_l2844_284444


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2844_284494

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ z w : ℝ, z > 0 → w > 0 → 2*z + 8*w - z*w = 0 → x + y ≤ z + w ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 8*y₀ - x₀*y₀ = 0 ∧ x₀ + y₀ = 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2844_284494


namespace NUMINAMATH_CALUDE_bush_current_age_l2844_284414

def matt_future_age : ℕ := 25
def years_to_future : ℕ := 10
def age_difference : ℕ := 3

theorem bush_current_age : 
  matt_future_age - years_to_future - age_difference = 12 := by
  sorry

end NUMINAMATH_CALUDE_bush_current_age_l2844_284414


namespace NUMINAMATH_CALUDE_combined_average_age_l2844_284469

theorem combined_average_age (room_a_count room_b_count room_c_count : ℕ)
                             (room_a_avg room_b_avg room_c_avg : ℝ) :
  room_a_count = 8 →
  room_b_count = 5 →
  room_c_count = 7 →
  room_a_avg = 30 →
  room_b_avg = 35 →
  room_c_avg = 40 →
  let total_count := room_a_count + room_b_count + room_c_count
  let total_age := room_a_count * room_a_avg + room_b_count * room_b_avg + room_c_count * room_c_avg
  (total_age / total_count : ℝ) = 34.75 := by
sorry

end NUMINAMATH_CALUDE_combined_average_age_l2844_284469


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2844_284468

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1 ∧ a > 0 ∧ b > 0

-- Define the focus
def focus (x y : ℝ) : Prop := x = 2 ∧ y = 0

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) :
  (∃ x y : ℝ, hyperbola a b x y ∧ focus x y) →
  eccentricity 2 →
  (∀ x y : ℝ, hyperbola a b x y ↔ x^2 - y^2/3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2844_284468


namespace NUMINAMATH_CALUDE_sum_not_five_implies_not_two_or_not_three_l2844_284445

theorem sum_not_five_implies_not_two_or_not_three (a b : ℝ) : 
  a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3 := by
sorry

end NUMINAMATH_CALUDE_sum_not_five_implies_not_two_or_not_three_l2844_284445


namespace NUMINAMATH_CALUDE_total_crayons_l2844_284427

theorem total_crayons (billy_crayons jane_crayons : ℝ) 
  (h1 : billy_crayons = 62.0) 
  (h2 : jane_crayons = 52.0) : 
  billy_crayons + jane_crayons = 114.0 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l2844_284427


namespace NUMINAMATH_CALUDE_simplify_expression_l2844_284411

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 = 9*b^3 + 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2844_284411


namespace NUMINAMATH_CALUDE_unique_data_set_l2844_284484

def mean (xs : Fin 4 → ℕ+) : ℚ :=
  (xs 0 + xs 1 + xs 2 + xs 3 : ℚ) / 4

def median (xs : Fin 4 → ℕ+) : ℚ :=
  (xs 1 + xs 2 : ℚ) / 2

def variance (xs : Fin 4 → ℕ+) (μ : ℚ) : ℚ :=
  ((xs 0 - μ)^2 + (xs 1 - μ)^2 + (xs 2 - μ)^2 + (xs 3 - μ)^2) / 4

def stdDev (xs : Fin 4 → ℕ+) (μ : ℚ) : ℚ :=
  (variance xs μ).sqrt

theorem unique_data_set (xs : Fin 4 → ℕ+) 
    (h_ordered : ∀ i j : Fin 4, i ≤ j → xs i ≤ xs j)
    (h_mean : mean xs = 2)
    (h_median : median xs = 2)
    (h_stddev : stdDev xs 2 = 1) :
    xs 0 = 1 ∧ xs 1 = 1 ∧ xs 2 = 3 ∧ xs 3 = 3 := by
  sorry

#check unique_data_set

end NUMINAMATH_CALUDE_unique_data_set_l2844_284484


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l2844_284492

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (garden.length : ℝ) * step_length * (garden.width : ℝ) * step_length * yield_per_sqft

/-- Theorem: The expected potato yield from Mr. Green's garden is 2109.375 pounds -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 18 25
  let step_length := 2.5
  let yield_per_sqft := 0.75
  expected_potato_yield garden step_length yield_per_sqft = 2109.375 := by
  sorry


end NUMINAMATH_CALUDE_mr_green_potato_yield_l2844_284492


namespace NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l2844_284480

theorem square_sum_from_product_and_sum (p q : ℝ) 
  (h1 : p * q = 9) 
  (h2 : p + q = 6) : 
  p^2 + q^2 = 18 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l2844_284480


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2844_284430

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (8 + 3 * z) = 12 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2844_284430


namespace NUMINAMATH_CALUDE_camp_wonka_marshmallows_l2844_284491

theorem camp_wonka_marshmallows : 
  ∀ (total_campers : ℕ) 
    (boys_fraction girls_fraction : ℚ) 
    (boys_toast_percent girls_toast_percent : ℚ),
  total_campers = 96 →
  boys_fraction = 2/3 →
  girls_fraction = 1/3 →
  boys_toast_percent = 1/2 →
  girls_toast_percent = 3/4 →
  (boys_fraction * ↑total_campers * boys_toast_percent + 
   girls_fraction * ↑total_campers * girls_toast_percent : ℚ) = 56 := by
sorry

end NUMINAMATH_CALUDE_camp_wonka_marshmallows_l2844_284491


namespace NUMINAMATH_CALUDE_expression_evaluation_l2844_284407

theorem expression_evaluation (z p q : ℝ) (hz : z ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  ((z^(2/p) + z^(2/q))^2 - 4*z^(2/p + 2/q)) / ((z^(1/p) - z^(1/q))^2 + 4*z^(1/p + 1/q)) = (|z^(1/p) - z^(1/q)|)^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2844_284407


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2844_284458

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to seat 8 people in a row with restrictions. -/
def seatingArrangements : ℕ :=
  let totalArrangements := factorial 8
  let wilmaAndPaulTogether := factorial 7 * factorial 2
  let adamAndEveTogether := factorial 7 * factorial 2
  let bothPairsTogether := factorial 6 * factorial 2 * factorial 2
  totalArrangements - (wilmaAndPaulTogether + adamAndEveTogether - bothPairsTogether)

/-- Theorem stating that the number of seating arrangements is 23040. -/
theorem seating_arrangements_count :
  seatingArrangements = 23040 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2844_284458


namespace NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l2844_284416

open Complex

theorem complex_exp_13pi_div_2 : exp (13 * π / 2 * I) = I := by sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l2844_284416


namespace NUMINAMATH_CALUDE_miltons_zoology_books_l2844_284487

theorem miltons_zoology_books :
  ∀ (z b : ℕ), b = 4 * z → z + b = 80 → z = 16 := by sorry

end NUMINAMATH_CALUDE_miltons_zoology_books_l2844_284487


namespace NUMINAMATH_CALUDE_f_always_negative_iff_m_in_range_l2844_284464

/-- The function f(x) defined as mx^2 - mx - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

/-- Theorem stating that f(x) < 0 for all real x if and only if m is in the interval (-4, 0] -/
theorem f_always_negative_iff_m_in_range :
  (∀ x : ℝ, f m x < 0) ↔ m ∈ Set.Ioc (-4) 0 := by sorry

end NUMINAMATH_CALUDE_f_always_negative_iff_m_in_range_l2844_284464


namespace NUMINAMATH_CALUDE_book_purchase_problem_l2844_284489

/-- Represents the number of books purchased -/
def num_books : ℕ := 8

/-- Represents the number of albums purchased -/
def num_albums : ℕ := num_books - 6

/-- Represents the price of a book in kopecks -/
def price_book : ℕ := 1056 / num_books

/-- Represents the price of an album in kopecks -/
def price_album : ℕ := 56 / num_albums

/-- Theorem stating that the given conditions are satisfied by the defined values -/
theorem book_purchase_problem :
  (num_books : ℤ) = (num_albums : ℤ) + 6 ∧
  num_books * price_book = 1056 ∧
  num_albums * price_album = 56 ∧
  price_book > price_album + 100 :=
by sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l2844_284489


namespace NUMINAMATH_CALUDE_two_complex_roots_iff_k_values_l2844_284442

/-- The equation has exactly two complex roots if and only if k is 0, 2i, or -2i -/
theorem two_complex_roots_iff_k_values (k : ℂ) : 
  (∃! (r₁ r₂ : ℂ), ∀ (x : ℂ), x ≠ -3 ∧ x ≠ -4 → 
    (x / (x + 3) + x / (x + 4) = k * x ↔ x = 0 ∨ x = r₁ ∨ x = r₂)) ↔ 
  (k = 0 ∨ k = 2*I ∨ k = -2*I) :=
sorry

end NUMINAMATH_CALUDE_two_complex_roots_iff_k_values_l2844_284442


namespace NUMINAMATH_CALUDE_gnome_ratio_l2844_284459

/-- Represents the properties of garden gnomes -/
structure GnomeProperties where
  total : Nat
  bigNoses : Nat
  blueHatsBigNoses : Nat
  redHatsSmallNoses : Nat

/-- Theorem: The ratio of gnomes with red hats to total gnomes is 3:4 -/
theorem gnome_ratio (g : GnomeProperties) 
  (h1 : g.total = 28)
  (h2 : g.bigNoses = g.total / 2)
  (h3 : g.blueHatsBigNoses = 6)
  (h4 : g.redHatsSmallNoses = 13) :
  (g.redHatsSmallNoses + (g.bigNoses - g.blueHatsBigNoses)) * 4 = g.total * 3 := by
  sorry

#check gnome_ratio

end NUMINAMATH_CALUDE_gnome_ratio_l2844_284459


namespace NUMINAMATH_CALUDE_weight_of_A_l2844_284496

theorem weight_of_A (A B C D E : ℝ) : 
  (A + B + C) / 3 = 60 →
  (A + B + C + D) / 4 = 65 →
  E = D + 3 →
  (B + C + D + E) / 4 = 64 →
  A = 87 := by
sorry

end NUMINAMATH_CALUDE_weight_of_A_l2844_284496


namespace NUMINAMATH_CALUDE_percentage_calculation_l2844_284475

theorem percentage_calculation (P : ℝ) : 
  P * 5600 = 126 → 
  (0.3 * 0.5 * 5600 : ℝ) = 840 → 
  P = 0.0225 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2844_284475


namespace NUMINAMATH_CALUDE_triangle_side_length_l2844_284408

/-- Given a triangle ABC with side lengths a and b, and angle C (in radians),
    proves that the length of side c is equal to √2 -/
theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a = 2 → b = Real.sqrt 3 - 1 → C = π / 6 → c = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2844_284408


namespace NUMINAMATH_CALUDE_sum_of_powers_positive_l2844_284479

theorem sum_of_powers_positive (a b c : ℝ) (h1 : a * b * c > 0) (h2 : a + b + c > 0) :
  ∀ n : ℕ, a ^ n + b ^ n + c ^ n > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_positive_l2844_284479


namespace NUMINAMATH_CALUDE_replacement_paint_intensity_l2844_284419

/-- Proves that the intensity of the replacement paint is 25% given the conditions of the problem -/
theorem replacement_paint_intensity
  (original_intensity : ℝ)
  (new_mixture_intensity : ℝ)
  (replaced_fraction : ℝ)
  (h1 : original_intensity = 50)
  (h2 : new_mixture_intensity = 35)
  (h3 : replaced_fraction = 0.6)
  : (1 - replaced_fraction) * original_intensity + replaced_fraction * 25 = new_mixture_intensity :=
by sorry

end NUMINAMATH_CALUDE_replacement_paint_intensity_l2844_284419


namespace NUMINAMATH_CALUDE_distinct_paths_eq_120_l2844_284409

/-- The number of distinct paths in a grid from point C to point D,
    where every step must either move up or to the right,
    and one has to move 7 steps to the right and 3 steps up. -/
def distinct_paths : ℕ := Nat.choose 10 3

/-- Theorem stating that the number of distinct paths is equal to 120. -/
theorem distinct_paths_eq_120 : distinct_paths = 120 := by sorry

end NUMINAMATH_CALUDE_distinct_paths_eq_120_l2844_284409


namespace NUMINAMATH_CALUDE_school_enrollment_problem_l2844_284439

theorem school_enrollment_problem (x y : ℝ) : 
  x + y = 4000 →
  0.07 * x - 0.03 * y = 40 →
  y = 2400 :=
by
  sorry

end NUMINAMATH_CALUDE_school_enrollment_problem_l2844_284439


namespace NUMINAMATH_CALUDE_Q_subset_P_l2844_284476

def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {y | y < 1}

theorem Q_subset_P : Q ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_Q_subset_P_l2844_284476


namespace NUMINAMATH_CALUDE_intersection_M_N_l2844_284400

def U := ℝ

def M : Set ℝ := {-1, 1, 2}

def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_M_N : N ∩ M = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2844_284400


namespace NUMINAMATH_CALUDE_farmers_additional_cost_l2844_284425

/-- The additional cost for Farmer Brown's new hay requirements -/
def additional_cost (original_bales : ℕ) (original_price : ℕ) (new_bales : ℕ) (new_price : ℕ) : ℕ :=
  new_bales * new_price - original_bales * original_price

/-- Theorem: The additional cost for Farmer Brown's new requirements is $210 -/
theorem farmers_additional_cost :
  additional_cost 10 15 20 18 = 210 := by
  sorry

end NUMINAMATH_CALUDE_farmers_additional_cost_l2844_284425


namespace NUMINAMATH_CALUDE_ball_count_problem_l2844_284424

/-- Proves that given the initial ratio of green to yellow balls is 3:7, 
    and after removing 9 balls of each color the new ratio becomes 1:3, 
    the original number of balls in the bag was 90. -/
theorem ball_count_problem (g y : ℕ) : 
  g * 7 = y * 3 →  -- initial ratio is 3:7
  (g - 9) * 3 = (y - 9) * 1 →  -- new ratio is 1:3 after removing 9 of each
  g + y = 90 := by  -- total number of balls is 90
sorry

end NUMINAMATH_CALUDE_ball_count_problem_l2844_284424


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l2844_284446

theorem arithmetic_progression_of_primes (p q r d : ℕ) : 
  Prime p → Prime q → Prime r → 
  p > 3 → q > 3 → r > 3 →
  q = p + d → r = p + 2*d → 
  6 ∣ d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l2844_284446


namespace NUMINAMATH_CALUDE_limit_special_function_l2844_284465

/-- The limit of (4^(5x) - 9^(-2x)) / (sin(x) - tan(x^3)) as x approaches 0 is ln(1024 * 81) -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ →
    |(4^(5*x) - 9^(-2*x)) / (Real.sin x - Real.tan (x^3)) - Real.log (1024 * 81)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_special_function_l2844_284465


namespace NUMINAMATH_CALUDE_elastic_collision_inelastic_collision_l2844_284410

/-- Represents a dumbbell with a weightless rod and identical small spheres at the ends -/
structure Dumbbell where
  length : ℝ  -- Half-length of the rod
  mass : ℝ    -- Mass of each sphere
  velocity : ℝ -- Initial velocity

/-- Represents the collision system of two dumbbells -/
structure CollisionSystem where
  dumbbell1 : Dumbbell
  dumbbell2 : Dumbbell

/-- Theorem for perfectly elastic collision -/
theorem elastic_collision 
  (system : CollisionSystem) 
  (h_identical : system.dumbbell1 = system.dumbbell2) 
  (h_opposite : system.dumbbell1.velocity = -system.dumbbell2.velocity) :
  let v_final1 := system.dumbbell1.velocity
  let v_final2 := -system.dumbbell2.velocity
  v_final1 = system.dumbbell1.velocity ∧ v_final2 = system.dumbbell2.velocity :=
sorry

/-- Theorem for perfectly inelastic collision -/
theorem inelastic_collision
  (system : CollisionSystem)
  (h_identical : system.dumbbell1 = system.dumbbell2)
  (h_opposite : system.dumbbell1.velocity = -system.dumbbell2.velocity) :
  let ω := system.dumbbell1.velocity / (2 * system.dumbbell1.length)
  ω = system.dumbbell1.velocity / (2 * system.dumbbell1.length) :=
sorry

end NUMINAMATH_CALUDE_elastic_collision_inelastic_collision_l2844_284410


namespace NUMINAMATH_CALUDE_profit_percentage_change_l2844_284440

def company_profits (revenue2008 : ℝ) : Prop :=
  let profit2008 := 0.1 * revenue2008
  let revenue2009 := 0.8 * revenue2008
  let profit2009 := 0.18 * revenue2009
  let revenue2010 := 1.25 * revenue2009
  let profit2010 := 0.15 * revenue2010
  let profit_change := (profit2010 - profit2008) / profit2008
  profit_change = 0.5

theorem profit_percentage_change (revenue2008 : ℝ) (h : revenue2008 > 0) :
  company_profits revenue2008 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_change_l2844_284440


namespace NUMINAMATH_CALUDE_cone_volume_arithmetic_progression_l2844_284418

/-- The volume of a right circular cone with radius, slant height, and height in arithmetic progression. -/
theorem cone_volume_arithmetic_progression (r s h d : ℝ) : 
  r > 0 → s > 0 → h > 0 → d > 0 →
  s = r + d → h = r + 2 * d →
  (1 / 3 : ℝ) * Real.pi * r^2 * h = (1 / 3 : ℝ) * Real.pi * (r^3 + 2 * d * r^2) := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_arithmetic_progression_l2844_284418


namespace NUMINAMATH_CALUDE_shooter_score_problem_l2844_284488

/-- A shooter's competition score problem -/
theorem shooter_score_problem 
  (first_six_shots : ℕ) 
  (record : ℕ) 
  (h1 : first_six_shots = 52) 
  (h2 : record = 89) 
  (h3 : ∀ shot, shot ∈ Set.Icc 1 10) :
  /- (1) Minimum score on 7th shot to break record -/
  (∃ x : ℕ, x ≥ 8 ∧ first_six_shots + x + 30 > record) ∧
  /- (2) Number of 10s needed in last 3 shots if 7th shot is 8 -/
  (first_six_shots + 8 + 30 > record) ∧
  /- (3) Necessity of at least one 10 in last 3 shots if 7th shot is 10 -/
  (∃ x y z : ℕ, x ∈ Set.Icc 1 10 ∧ y ∈ Set.Icc 1 10 ∧ z ∈ Set.Icc 1 10 ∧
    first_six_shots + 10 + x + y + z > record ∧ (x = 10 ∨ y = 10 ∨ z = 10)) := by
  sorry


end NUMINAMATH_CALUDE_shooter_score_problem_l2844_284488


namespace NUMINAMATH_CALUDE_max_abs_u_for_unit_circle_l2844_284471

theorem max_abs_u_for_unit_circle (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^4 - z^3 - 3*z^2*Complex.I - z + 1) ≤ 5 ∧
  Complex.abs ((-1 : ℂ)^4 - (-1 : ℂ)^3 - 3*(-1 : ℂ)^2*Complex.I - (-1 : ℂ) + 1) = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_u_for_unit_circle_l2844_284471


namespace NUMINAMATH_CALUDE_average_weight_abc_l2844_284417

theorem average_weight_abc (a b c : ℝ) : 
  (a + b) / 2 = 42 →
  (b + c) / 2 = 43 →
  b = 35 →
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_abc_l2844_284417


namespace NUMINAMATH_CALUDE_other_number_proof_l2844_284452

theorem other_number_proof (A B : ℕ) : 
  A > 0 → B > 0 →
  Nat.lcm A B = 9699690 →
  Nat.gcd A B = 385 →
  A = 44530 →
  B = 83891 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l2844_284452


namespace NUMINAMATH_CALUDE_perfect_squares_condition_l2844_284495

theorem perfect_squares_condition (n : ℕ+) : 
  (∃ a b : ℕ, (8 * n.val - 7 = a ^ 2) ∧ (18 * n.val - 35 = b ^ 2)) ↔ (n.val = 2 ∨ n.val = 22) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_condition_l2844_284495


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2844_284413

/-- The focal length of a hyperbola with equation y²/4 - x² = 1 is 2√5 -/
theorem hyperbola_focal_length :
  let hyperbola := {(x, y) : ℝ × ℝ | y^2 / 4 - x^2 = 1}
  ∃ (f : ℝ), f = 2 * Real.sqrt 5 ∧ 
    ∀ (p q : ℝ × ℝ), p ∈ hyperbola → q ∈ hyperbola → 
      abs (dist p (0, f) - dist p (0, -f)) = 2 * abs (p.1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2844_284413


namespace NUMINAMATH_CALUDE_original_average_age_proof_l2844_284451

theorem original_average_age_proof (N : ℕ) (A : ℝ) : 
  A = 50 →
  (N * A + 12 * 32) / (N + 12) = 46 →
  A = 50 := by
sorry

end NUMINAMATH_CALUDE_original_average_age_proof_l2844_284451


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2844_284438

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) : 
  Real.sin (2 * α) = -17 / 18 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2844_284438


namespace NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l2844_284498

def consecutive_integers (start : Int) (n : Nat) : List Int :=
  List.range n |>.map (λ i => start + i)

def positive_integers (list : List Int) : List Int :=
  list.filter (λ x => x > 0)

def range_of_list (list : List Int) : Int :=
  list.maximum?.getD 0 - list.minimum?.getD 0

theorem range_of_positive_integers_in_list (k : List Int) :
  k = consecutive_integers (-4) 14 →
  range_of_list (positive_integers k) = 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l2844_284498


namespace NUMINAMATH_CALUDE_balloon_count_l2844_284415

/-- Calculates the total number of balloons given the number of gold, silver, and black balloons -/
def total_balloons (gold : ℕ) (silver : ℕ) (black : ℕ) : ℕ :=
  gold + silver + black

/-- Proves that the total number of balloons is 573 given the specified conditions -/
theorem balloon_count : 
  let gold : ℕ := 141
  let silver : ℕ := 2 * gold
  let black : ℕ := 150
  total_balloons gold silver black = 573 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l2844_284415


namespace NUMINAMATH_CALUDE_min_boxes_for_cube_l2844_284493

/-- The width of the box in centimeters -/
def box_width : ℕ := 8

/-- The length of the box in centimeters -/
def box_length : ℕ := 12

/-- The height of the box in centimeters -/
def box_height : ℕ := 30

/-- The volume of a single box in cubic centimeters -/
def box_volume : ℕ := box_width * box_length * box_height

/-- The side length of the smallest cube that can be formed -/
def cube_side : ℕ := Nat.lcm (Nat.lcm box_width box_length) box_height

/-- The volume of the smallest cube that can be formed -/
def cube_volume : ℕ := cube_side ^ 3

/-- The theorem stating the minimum number of boxes needed to form a cube -/
theorem min_boxes_for_cube : cube_volume / box_volume = 600 := by
  sorry

end NUMINAMATH_CALUDE_min_boxes_for_cube_l2844_284493


namespace NUMINAMATH_CALUDE_ellipse_area_l2844_284412

/-- The area of an ellipse with semi-major axis a and semi-minor axis b is k*π where k = a*b -/
theorem ellipse_area (a b : ℝ) (h1 : a = 12) (h2 : b = 6) : ∃ k : ℝ, k = 72 ∧ a * b * π = k * π := by
  sorry

end NUMINAMATH_CALUDE_ellipse_area_l2844_284412


namespace NUMINAMATH_CALUDE_x_to_y_value_l2844_284448

theorem x_to_y_value (x y : ℝ) (h : (x + 2)^2 + |y - 3| = 0) : x^y = -8 := by
  sorry

end NUMINAMATH_CALUDE_x_to_y_value_l2844_284448


namespace NUMINAMATH_CALUDE_estimate_total_balls_l2844_284402

/-- Represents a box containing red and green balls -/
structure BallBox where
  redBalls : ℕ
  totalBalls : ℕ
  hRedBalls : redBalls > 0
  hTotalBalls : totalBalls ≥ redBalls

/-- The probability of drawing a red ball -/
def drawRedProbability (box : BallBox) : ℚ :=
  box.redBalls / box.totalBalls

theorem estimate_total_balls
  (box : BallBox)
  (hRedBalls : box.redBalls = 5)
  (hProbability : drawRedProbability box = 1/4) :
  box.totalBalls = 20 := by
sorry

end NUMINAMATH_CALUDE_estimate_total_balls_l2844_284402


namespace NUMINAMATH_CALUDE_reservoir_capacity_l2844_284486

theorem reservoir_capacity : ∀ (capacity : ℚ),
  (1/8 : ℚ) * capacity + 200 = (1/2 : ℚ) * capacity →
  capacity = 1600/3 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_l2844_284486


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l2844_284401

def is_smallest_satisfying_number (n : ℕ) : Prop :=
  (∀ m < n, ∃ p, Nat.Prime p ∧ m % (p - 1) = 0 ∧ m % p ≠ 0) ∧
  (∀ p, Nat.Prime p → n % (p - 1) = 0 → n % p = 0)

theorem smallest_satisfying_number :
  is_smallest_satisfying_number 1806 :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l2844_284401


namespace NUMINAMATH_CALUDE_inequality_always_holds_l2844_284473

theorem inequality_always_holds (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 1| > a) ↔ a < 2 := by sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l2844_284473


namespace NUMINAMATH_CALUDE_binomial_2024_1_l2844_284443

theorem binomial_2024_1 : Nat.choose 2024 1 = 2024 := by sorry

end NUMINAMATH_CALUDE_binomial_2024_1_l2844_284443
