import Mathlib

namespace NUMINAMATH_CALUDE_visitor_difference_l1018_101869

def visitors_current_day : ℕ := 317
def visitors_previous_day : ℕ := 295

theorem visitor_difference : visitors_current_day - visitors_previous_day = 22 := by
  sorry

end NUMINAMATH_CALUDE_visitor_difference_l1018_101869


namespace NUMINAMATH_CALUDE_mike_shortfall_l1018_101878

def max_marks : ℕ := 780
def mike_score : ℕ := 212
def passing_percentage : ℚ := 30 / 100

theorem mike_shortfall :
  (↑max_marks * passing_percentage).floor - mike_score = 22 := by
  sorry

end NUMINAMATH_CALUDE_mike_shortfall_l1018_101878


namespace NUMINAMATH_CALUDE_num_solutions_eq_1176_l1018_101885

/-- The number of distinct ordered triples (a, b, c) of positive integers satisfying a + b + c = 50 -/
def num_solutions : ℕ :=
  (Finset.range 49).sum (λ k ↦ 49 - k)

/-- Theorem stating that the number of solutions is 1176 -/
theorem num_solutions_eq_1176 : num_solutions = 1176 := by
  sorry

end NUMINAMATH_CALUDE_num_solutions_eq_1176_l1018_101885


namespace NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequences_l1018_101837

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : ℚ) : Prop :=
  b^2 = a * c

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression (a b c : ℚ) : Prop :=
  2 * b = a + c

/-- Represents a triple of rational numbers -/
structure Triple where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a triple satisfies all the conditions -/
def satisfies_conditions (t : Triple) : Prop :=
  is_geometric_progression t.a t.b t.c ∧
  is_arithmetic_progression t.a t.b (t.c - 4) ∧
  is_geometric_progression t.a (t.b - 1) (t.c - 5)

theorem geometric_arithmetic_geometric_sequences :
  ∃ t₁ t₂ : Triple,
    satisfies_conditions t₁ ∧
    satisfies_conditions t₂ ∧
    t₁ = ⟨1/9, 7/9, 49/9⟩ ∧
    t₂ = ⟨1, 3, 9⟩ :=
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequences_l1018_101837


namespace NUMINAMATH_CALUDE_income_left_percentage_l1018_101887

/-- Given a man's spending habits, calculate the percentage of income left --/
theorem income_left_percentage (total_income : ℝ) (food_percent : ℝ) (education_percent : ℝ) (rent_percent : ℝ)
  (h1 : food_percent = 50)
  (h2 : education_percent = 15)
  (h3 : rent_percent = 50)
  (h4 : total_income > 0) :
  let remaining_after_food := total_income * (1 - food_percent / 100)
  let remaining_after_education := remaining_after_food - (total_income * education_percent / 100)
  let remaining_after_rent := remaining_after_education * (1 - rent_percent / 100)
  remaining_after_rent / total_income * 100 = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_income_left_percentage_l1018_101887


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l1018_101828

theorem max_ratio_two_digit_integers (x y : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ y ≥ 10 ∧ y ≤ 99 → -- x and y are two-digit positive integers
  (x + y) / 2 = 65 → -- mean is 65
  x * y = 1950 → -- product is 1950
  ∀ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (a + b) / 2 = 65 ∧ a * b = 1950 →
    (a : ℚ) / b ≤ 99 / 31 :=
by sorry

#check max_ratio_two_digit_integers

end NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l1018_101828


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l1018_101873

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 18 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 18 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l1018_101873


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l1018_101881

/-- The coefficient of x^2 in the expansion of (3x^2 + 5x + 2)(4x^2 + 2x + 1) -/
def coefficient_x_squared : ℤ := 21

/-- The first polynomial in the product -/
def p (x : ℚ) : ℚ := 3 * x^2 + 5 * x + 2

/-- The second polynomial in the product -/
def q (x : ℚ) : ℚ := 4 * x^2 + 2 * x + 1

/-- The theorem stating that the coefficient of x^2 in the expansion of (3x^2 + 5x + 2)(4x^2 + 2x + 1) is 21 -/
theorem coefficient_x_squared_expansion :
  ∃ (a b c d e : ℚ), (p * q) = (λ x => a * x^4 + b * x^3 + coefficient_x_squared * x^2 + d * x + e) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l1018_101881


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1018_101889

theorem regular_hexagon_perimeter (s : ℝ) (h : s > 0) : 
  (3 * Real.sqrt 3 / 2) * s^2 = s → 6 * s = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1018_101889


namespace NUMINAMATH_CALUDE_stock_discount_calculation_l1018_101822

/-- Calculates the discount on a stock given its original price, brokerage fee, and final cost price. -/
theorem stock_discount_calculation (original_price brokerage_rate final_cost_price : ℝ) : 
  original_price = 100 →
  brokerage_rate = 1 / 500 →
  final_cost_price = 95.2 →
  ∃ (discount : ℝ), 
    (original_price - discount) * (1 + brokerage_rate) = final_cost_price ∧
    abs (discount - 4.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_stock_discount_calculation_l1018_101822


namespace NUMINAMATH_CALUDE_fiftieth_digit_of_seventh_l1018_101891

/-- The decimal representation of 1/7 as a list of digits -/
def seventhDecimal : List Nat := [1, 4, 2, 8, 5, 7]

/-- The length of the repeating part in the decimal representation of 1/7 -/
def repeatLength : Nat := 6

/-- The 50th digit after the decimal point in the decimal representation of 1/7 -/
def fiftiethDigit : Nat := seventhDecimal[(50 - 1) % repeatLength]

theorem fiftieth_digit_of_seventh :
  fiftiethDigit = 4 := by sorry

end NUMINAMATH_CALUDE_fiftieth_digit_of_seventh_l1018_101891


namespace NUMINAMATH_CALUDE_magic_square_x_value_l1018_101832

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ

/-- The sum of each row, column, and diagonal in a magic square -/
def magicSum (s : MagicSquare) : ℕ := s.a + s.b + s.c

/-- Predicate for a valid magic square -/
def isMagicSquare (s : MagicSquare) : Prop :=
  -- All rows have the same sum
  s.a + s.b + s.c = magicSum s ∧
  s.d + s.e + s.f = magicSum s ∧
  s.g + s.h + s.i = magicSum s ∧
  -- All columns have the same sum
  s.a + s.d + s.g = magicSum s ∧
  s.b + s.e + s.h = magicSum s ∧
  s.c + s.f + s.i = magicSum s ∧
  -- Both diagonals have the same sum
  s.a + s.e + s.i = magicSum s ∧
  s.c + s.e + s.g = magicSum s

theorem magic_square_x_value (s : MagicSquare) 
  (h1 : isMagicSquare s)
  (h2 : s.b = 19 ∧ s.e = 15 ∧ s.h = 11)  -- Second column condition
  (h3 : s.b = 19 ∧ s.c = 14)  -- First row condition
  (h4 : s.e = 15 ∧ s.i = 12)  -- Diagonal condition
  : s.g = 18 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_x_value_l1018_101832


namespace NUMINAMATH_CALUDE_olivia_remaining_money_l1018_101867

def initial_amount : ℝ := 200
def grocery_cost : ℝ := 65
def shoe_original_price : ℝ := 75
def shoe_discount_rate : ℝ := 0.15
def belt_cost : ℝ := 25

def remaining_money : ℝ :=
  initial_amount - (grocery_cost + (shoe_original_price * (1 - shoe_discount_rate)) + belt_cost)

theorem olivia_remaining_money :
  remaining_money = 46.25 := by sorry

end NUMINAMATH_CALUDE_olivia_remaining_money_l1018_101867


namespace NUMINAMATH_CALUDE_quadratic_properties_l1018_101811

def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_properties (b c : ℝ) 
  (h1 : f b c 1 = 0) 
  (h2 : f b c 3 = 0) : 
  (f b c (-1) = 8) ∧ 
  (∀ x ∈ Set.Icc 2 4, f b c x ≤ 3) ∧
  (∃ x ∈ Set.Icc 2 4, f b c x = 3) ∧
  (∀ x ∈ Set.Icc 2 4, -1 ≤ f b c x) ∧
  (∃ x ∈ Set.Icc 2 4, f b c x = -1) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f b c x ≤ f b c y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1018_101811


namespace NUMINAMATH_CALUDE_fabian_accessories_cost_l1018_101858

def mouse_cost : ℕ := 16

def keyboard_cost (m : ℕ) : ℕ := 3 * m

def total_cost (m k : ℕ) : ℕ := m + k

theorem fabian_accessories_cost :
  total_cost mouse_cost (keyboard_cost mouse_cost) = 64 := by
  sorry

end NUMINAMATH_CALUDE_fabian_accessories_cost_l1018_101858


namespace NUMINAMATH_CALUDE_marys_number_l1018_101874

theorem marys_number (n : ℕ) : 
  150 ∣ n → 
  45 ∣ n → 
  1000 ≤ n → 
  n ≤ 3000 → 
  n = 1350 ∨ n = 1800 ∨ n = 2250 ∨ n = 2700 := by
sorry

end NUMINAMATH_CALUDE_marys_number_l1018_101874


namespace NUMINAMATH_CALUDE_ella_seventh_test_score_l1018_101834

def is_valid_score_set (scores : List ℤ) : Prop :=
  scores.length = 8 ∧
  scores.all (λ s => 88 ≤ s ∧ s ≤ 97) ∧
  scores.Nodup ∧
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 8 → (scores.take k).sum % k = 0) ∧
  scores.get! 7 = 90

theorem ella_seventh_test_score (scores : List ℤ) :
  is_valid_score_set scores → scores.get! 6 = 95 := by
  sorry

#check ella_seventh_test_score

end NUMINAMATH_CALUDE_ella_seventh_test_score_l1018_101834


namespace NUMINAMATH_CALUDE_pedal_triangle_perimeter_and_area_l1018_101892

/-- Given a triangle with circumradius R and angles α, β, and γ,
    this theorem states the formulas for the perimeter and twice the area of its pedal triangle. -/
theorem pedal_triangle_perimeter_and_area 
  (R : ℝ) (α β γ : ℝ) : 
  ∃ (k t : ℝ),
    k = 4 * R * Real.sin α * Real.sin β * Real.sin γ ∧ 
    2 * t = R^2 * Real.sin (2*α) * Real.sin (2*β) * Real.sin (2*γ) := by
  sorry

end NUMINAMATH_CALUDE_pedal_triangle_perimeter_and_area_l1018_101892


namespace NUMINAMATH_CALUDE_A_not_lose_probability_l1018_101888

/-- The probability of player A winning -/
def prob_A_win : ℝ := 0.30

/-- The probability of a draw between players A and B -/
def prob_draw : ℝ := 0.25

/-- The probability that player A does not lose -/
def prob_A_not_lose : ℝ := prob_A_win + prob_draw

theorem A_not_lose_probability : prob_A_not_lose = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_A_not_lose_probability_l1018_101888


namespace NUMINAMATH_CALUDE_three_digit_equation_solution_l1018_101829

/-- Represents a three-digit number ABC --/
def threeDigitNumber (A B C : ℕ) : ℕ := 100 * A + 10 * B + C

/-- Represents a two-digit number AB --/
def twoDigitNumber (A B : ℕ) : ℕ := 10 * A + B

/-- Checks if three numbers are distinct digits --/
def areDistinctDigits (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ A ≥ 1 ∧ B ≥ 1 ∧ C ≥ 1

theorem three_digit_equation_solution :
  ∀ A B C : ℕ,
    areDistinctDigits A B C →
    threeDigitNumber A B C = twoDigitNumber A B * C + twoDigitNumber B C * A + twoDigitNumber C A * B →
    ((A = 7 ∧ B = 8 ∧ C = 1) ∨ (A = 5 ∧ B = 1 ∧ C = 7)) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_equation_solution_l1018_101829


namespace NUMINAMATH_CALUDE_response_rate_percentage_l1018_101866

theorem response_rate_percentage : 
  ∀ (responses_needed : ℕ) (questionnaires_mailed : ℕ),
  responses_needed = 210 →
  questionnaires_mailed = 350 →
  (responses_needed : ℝ) / (questionnaires_mailed : ℝ) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_response_rate_percentage_l1018_101866


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l1018_101862

theorem polynomial_division_quotient :
  ∀ (x : ℝ),
  ∃ (r : ℝ),
  8 * x^3 + 5 * x^2 - 4 * x - 7 = (x + 3) * (8 * x^2 - 19 * x + 53) + r :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l1018_101862


namespace NUMINAMATH_CALUDE_min_value_of_f_l1018_101872

/-- The function f(x) = x^2 + 8x + 12 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 12

/-- The minimum value of f(x) is -4 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -4 ∧ ∀ (x : ℝ), f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1018_101872


namespace NUMINAMATH_CALUDE_function_bound_l1018_101880

theorem function_bound (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = Real.sqrt 3 * Real.sin (3 * x) + Real.cos (3 * x)) →
  (∀ x : ℝ, |f x| ≤ a) →
  a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l1018_101880


namespace NUMINAMATH_CALUDE_line_equation_through_two_points_l1018_101883

/-- The line passing through points A(-2, 4) and B(-1, 3) has the equation y = -x + 2 -/
theorem line_equation_through_two_points :
  let A : ℝ × ℝ := (-2, 4)
  let B : ℝ × ℝ := (-1, 3)
  let line_eq : ℝ → ℝ := λ x => -x + 2
  (line_eq A.1 = A.2) ∧ (line_eq B.1 = B.2) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_two_points_l1018_101883


namespace NUMINAMATH_CALUDE_not_both_rational_l1018_101868

theorem not_both_rational (x : ℝ) : ¬(∃ (a b : ℚ), (x + Real.sqrt 3 : ℝ) = a ∧ (x^3 + 5 * Real.sqrt 3 : ℝ) = b) :=
sorry

end NUMINAMATH_CALUDE_not_both_rational_l1018_101868


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1018_101830

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  B = π / 3 → a = Real.sqrt 2 →
  (b = Real.sqrt 3 → A = π / 4) ∧
  (1 / 2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2 → b = Real.sqrt 14) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1018_101830


namespace NUMINAMATH_CALUDE_second_agency_per_mile_charge_l1018_101863

theorem second_agency_per_mile_charge : 
  let first_agency_daily_charge : ℝ := 20.25
  let first_agency_per_mile_charge : ℝ := 0.14
  let second_agency_daily_charge : ℝ := 18.25
  let miles_at_equal_cost : ℝ := 25
  let second_agency_per_mile_charge : ℝ := 
    (first_agency_daily_charge + first_agency_per_mile_charge * miles_at_equal_cost - second_agency_daily_charge) / miles_at_equal_cost
  second_agency_per_mile_charge = 0.22 := by
sorry

end NUMINAMATH_CALUDE_second_agency_per_mile_charge_l1018_101863


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l1018_101884

theorem rectangle_area_proof : 
  let card1 : ℝ := 15
  let card2 : ℝ := card1 * 0.9
  let area : ℝ := card1 * card2
  area = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l1018_101884


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1018_101800

/-- Given a quadratic function f(x) = ax^2 + (1-a)x + a - 2,
    if f(x) ≥ -2 for all real x, then a ≥ 1/3 -/
theorem quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (1 - a) * x + a - 2 ≥ -2) → a ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1018_101800


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1018_101847

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 4*x - 12 = 0 ↔ x = 6 ∨ x = -2) ∧
  (∀ x : ℝ, (2*x - 1)^2 = 3*(2*x - 1) ↔ x = 1/2 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1018_101847


namespace NUMINAMATH_CALUDE_even_square_operation_l1018_101819

theorem even_square_operation (x : ℕ) (h : x > 0) : ∃ k : ℕ, (2 * x)^2 = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_even_square_operation_l1018_101819


namespace NUMINAMATH_CALUDE_negative_two_cubed_minus_squared_l1018_101875

theorem negative_two_cubed_minus_squared : (-2)^3 - (-2)^2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_minus_squared_l1018_101875


namespace NUMINAMATH_CALUDE_remainder_problem_l1018_101810

theorem remainder_problem (G : ℕ) (a b : ℕ) (h1 : G = 127) (h2 : a = 1661) (h3 : b = 2045) 
  (h4 : b % G = 13) (h5 : ∀ d : ℕ, d > G → (a % d ≠ 0 ∨ b % d ≠ 0)) :
  a % G = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1018_101810


namespace NUMINAMATH_CALUDE_product_of_powers_equals_power_of_sum_l1018_101871

theorem product_of_powers_equals_power_of_sum :
  (10 ^ 0.4) * (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.05) * (10 ^ 1.1) * (10 ^ (-0.1)) = 10 ^ 1.85 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_power_of_sum_l1018_101871


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1018_101843

theorem diophantine_equation_solution (n m : ℤ) : 
  n^4 + 2*n^3 + 2*n^2 + 2*n + 1 = m^2 ↔ (n = 0 ∧ (m = 1 ∨ m = -1)) ∨ (n = -1 ∧ m = 0) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1018_101843


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l1018_101841

theorem sqrt_sum_equals_2sqrt14 : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l1018_101841


namespace NUMINAMATH_CALUDE_employee_savings_l1018_101861

/-- Calculate the combined savings of three employees after four weeks -/
theorem employee_savings (hourly_wage : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ)
  (robby_save_ratio : ℚ) (jaylen_save_ratio : ℚ) (miranda_save_ratio : ℚ)
  (h1 : hourly_wage = 10)
  (h2 : hours_per_day = 10)
  (h3 : days_per_week = 5)
  (h4 : num_weeks = 4)
  (h5 : robby_save_ratio = 2/5)
  (h6 : jaylen_save_ratio = 3/5)
  (h7 : miranda_save_ratio = 1/2) :
  (hourly_wage * hours_per_day * days_per_week * num_weeks) *
  (robby_save_ratio + jaylen_save_ratio + miranda_save_ratio) = 3000 := by
  sorry


end NUMINAMATH_CALUDE_employee_savings_l1018_101861


namespace NUMINAMATH_CALUDE_marble_count_l1018_101890

theorem marble_count (total : ℕ) (red blue white : ℕ) : 
  total = 108 →
  blue = red / 3 →
  white = blue / 2 →
  red + blue + white = total →
  white < red ∧ white < blue :=
by sorry

end NUMINAMATH_CALUDE_marble_count_l1018_101890


namespace NUMINAMATH_CALUDE_lcm_of_210_and_913_l1018_101817

theorem lcm_of_210_and_913 :
  let a : ℕ := 210
  let b : ℕ := 913
  let hcf : ℕ := 83
  Nat.lcm a b = 2310 :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_210_and_913_l1018_101817


namespace NUMINAMATH_CALUDE_johns_new_height_l1018_101844

/-- Calculates the new height in feet after a growth spurt -/
def new_height_in_feet (initial_height_inches : ℕ) (growth_rate_inches_per_month : ℕ) (growth_duration_months : ℕ) : ℚ :=
  (initial_height_inches + growth_rate_inches_per_month * growth_duration_months) / 12

/-- Proves that John's new height is 6 feet -/
theorem johns_new_height :
  new_height_in_feet 66 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_johns_new_height_l1018_101844


namespace NUMINAMATH_CALUDE_division_theorem_problem_1999_division_l1018_101848

theorem division_theorem (n d q r : ℕ) (h : n = d * q + r) (h_r : r < d) :
  (n / d = q ∧ n % d = r) :=
sorry

theorem problem_1999_division :
  1999 / 40 = 49 ∧ 1999 % 40 = 39 :=
sorry

end NUMINAMATH_CALUDE_division_theorem_problem_1999_division_l1018_101848


namespace NUMINAMATH_CALUDE_sequence_2019th_term_l1018_101893

theorem sequence_2019th_term (a : ℕ → ℤ) 
  (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n - 2) : 
  a 2019 = -4034 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2019th_term_l1018_101893


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1018_101870

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 4 * x + 1 = 0) ↔ (k ≤ 4 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l1018_101870


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1018_101879

theorem cubic_root_sum_cubes (p q r : ℝ) : 
  (p^3 - 2*p^2 + 3*p - 4 = 0) ∧ 
  (q^3 - 2*q^2 + 3*q - 4 = 0) ∧ 
  (r^3 - 2*r^2 + 3*r - 4 = 0) →
  p^3 + q^3 + r^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1018_101879


namespace NUMINAMATH_CALUDE_f_definition_f_2019_l1018_101877

def a_n (n : ℕ) : ℕ := Nat.sqrt n

def b_n (n : ℕ) : ℕ := n - (a_n n)^2

def f (n : ℕ) : ℕ :=
  if b_n n ≤ a_n n then
    (a_n n)^2 + 1
  else if a_n n < b_n n ∧ b_n n ≤ 2 * (a_n n) + 1 then
    (a_n n)^2 + a_n n + 1
  else
    0  -- This case should never occur based on the problem definition

theorem f_definition (n : ℕ) :
  f n = if b_n n ≤ a_n n then
          (a_n n)^2 + 1
        else
          (a_n n)^2 + a_n n + 1 :=
by sorry

theorem f_2019 : f 2019 = 1981 :=
by sorry

end NUMINAMATH_CALUDE_f_definition_f_2019_l1018_101877


namespace NUMINAMATH_CALUDE_jills_salary_l1018_101859

/-- Represents a person's monthly finances -/
structure MonthlyFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  giftAmount : ℝ

/-- Conditions for Jill's monthly finances -/
def jillsFinances (f : MonthlyFinances) : Prop :=
  f.discretionaryIncome = f.netSalary / 5 ∧
  f.giftAmount = f.discretionaryIncome * 0.2 ∧
  f.giftAmount = 111

/-- Theorem: If Jill's finances meet the given conditions, her net monthly salary is $2775 -/
theorem jills_salary (f : MonthlyFinances) (h : jillsFinances f) : f.netSalary = 2775 := by
  sorry

end NUMINAMATH_CALUDE_jills_salary_l1018_101859


namespace NUMINAMATH_CALUDE_point_not_in_second_quadrant_l1018_101896

theorem point_not_in_second_quadrant : ¬ ((-Real.sqrt 2 < 0) ∧ (-Real.sqrt 3 > 0)) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_second_quadrant_l1018_101896


namespace NUMINAMATH_CALUDE_f_max_value_l1018_101803

/-- The function f(x) = 5x - x^2 -/
def f (x : ℝ) : ℝ := 5 * x - x^2

/-- The maximum value of f(x) is 6.25 -/
theorem f_max_value : ∃ (c : ℝ), ∀ (x : ℝ), f x ≤ c ∧ ∃ (x₀ : ℝ), f x₀ = c :=
  sorry

end NUMINAMATH_CALUDE_f_max_value_l1018_101803


namespace NUMINAMATH_CALUDE_simplify_expression_l1018_101851

theorem simplify_expression (x : ℝ) : (2*x)^5 + (3*x)*(x^4) + 2*x^3 = 35*x^5 + 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1018_101851


namespace NUMINAMATH_CALUDE_sonya_falls_l1018_101840

/-- The number of times each person fell while ice skating --/
structure FallCounts where
  steven : ℕ
  stephanie : ℕ
  sonya : ℕ
  sam : ℕ
  sophie : ℕ

/-- The conditions given in the problem --/
def carnival_conditions (fc : FallCounts) : Prop :=
  fc.steven = 3 ∧
  fc.stephanie = fc.steven + 13 ∧
  fc.sonya = fc.stephanie / 2 - 2 ∧
  fc.sam = 1 ∧
  fc.sophie = fc.sam + 4

/-- Theorem stating that Sonya fell 6 times --/
theorem sonya_falls (fc : FallCounts) (h : carnival_conditions fc) : fc.sonya = 6 := by
  sorry

end NUMINAMATH_CALUDE_sonya_falls_l1018_101840


namespace NUMINAMATH_CALUDE_wire_radius_from_sphere_l1018_101826

/-- The radius of a wire's cross section when a sphere is melted and drawn into a wire -/
theorem wire_radius_from_sphere (r_sphere : ℝ) (l_wire : ℝ) (r_wire : ℝ) : 
  r_sphere = 12 →
  l_wire = 144 →
  (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_wire^2 * l_wire →
  r_wire = 4 := by
  sorry

#check wire_radius_from_sphere

end NUMINAMATH_CALUDE_wire_radius_from_sphere_l1018_101826


namespace NUMINAMATH_CALUDE_right_triangle_legs_product_divisible_by_12_l1018_101816

theorem right_triangle_legs_product_divisible_by_12 
  (a b c : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  12 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_product_divisible_by_12_l1018_101816


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1018_101860

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem statement -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x := by
sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l1018_101860


namespace NUMINAMATH_CALUDE_mike_seashells_l1018_101876

/-- The number of seashells Mike found -/
def total_seashells (unbroken_seashells broken_seashells : ℕ) : ℕ :=
  unbroken_seashells + broken_seashells

/-- Theorem stating that Mike found 6 seashells in total -/
theorem mike_seashells : total_seashells 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mike_seashells_l1018_101876


namespace NUMINAMATH_CALUDE_remaining_space_for_regular_toenails_l1018_101850

/-- Represents the capacity of the jar in terms of regular toenails -/
def jarCapacity : ℕ := 100

/-- Represents the space occupied by a big toenail in terms of regular toenails -/
def bigToenailSpace : ℕ := 2

/-- Represents the number of big toenails already in the jar -/
def bigToenailsInJar : ℕ := 20

/-- Represents the number of regular toenails already in the jar -/
def regularToenailsInJar : ℕ := 40

/-- Theorem stating that the remaining space in the jar can fit exactly 20 regular toenails -/
theorem remaining_space_for_regular_toenails : 
  jarCapacity - (bigToenailsInJar * bigToenailSpace + regularToenailsInJar) = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_space_for_regular_toenails_l1018_101850


namespace NUMINAMATH_CALUDE_food_consumption_reduction_l1018_101853

/-- Proves that given a 15% decrease in students and 20% increase in food price,
    the consumption reduction factor to maintain the same total cost is approximately 0.98039 -/
theorem food_consumption_reduction (N : ℝ) (P : ℝ) (h1 : N > 0) (h2 : P > 0) :
  let new_students := 0.85 * N
  let new_price := 1.2 * P
  let consumption_factor := (N * P) / (new_students * new_price)
  ∃ ε > 0, abs (consumption_factor - 0.98039) < ε :=
by sorry

end NUMINAMATH_CALUDE_food_consumption_reduction_l1018_101853


namespace NUMINAMATH_CALUDE_sequence_inequality_l1018_101801

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ k m : ℕ, |a (k + m) - a k - a m| ≤ 1

theorem sequence_inequality (a : ℕ → ℝ) (h : sequence_condition a) :
  ∀ p q : ℕ, p ≠ 0 → q ≠ 0 → |a p / p - a q / q| < 1 / p + 1 / q :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1018_101801


namespace NUMINAMATH_CALUDE_track_circumference_l1018_101886

/-- The circumference of a circular track given specific running conditions -/
theorem track_circumference (brenda_first_meeting : ℝ) (sally_second_meeting : ℝ) 
  (h1 : brenda_first_meeting = 120)
  (h2 : sally_second_meeting = 180) :
  let circumference := brenda_first_meeting * 3/2
  circumference = 180 := by sorry

end NUMINAMATH_CALUDE_track_circumference_l1018_101886


namespace NUMINAMATH_CALUDE_system_solution_l1018_101812

theorem system_solution : ∃ (x y : ℝ), x + y = 0 ∧ 2*x + 3*y = 3 ∧ x = -3 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1018_101812


namespace NUMINAMATH_CALUDE_zoo_visit_cost_l1018_101804

/-- Calculates the total cost of a zoo visit for a group with a discount applied -/
theorem zoo_visit_cost 
  (num_children num_adults num_seniors : ℕ)
  (child_price adult_price senior_price : ℚ)
  (discount_rate : ℚ)
  (h_children : num_children = 6)
  (h_adults : num_adults = 10)
  (h_seniors : num_seniors = 4)
  (h_child_price : child_price = 12)
  (h_adult_price : adult_price = 20)
  (h_senior_price : senior_price = 15)
  (h_discount : discount_rate = 0.15) :
  (num_children : ℚ) * child_price + 
  (num_adults : ℚ) * adult_price + 
  (num_seniors : ℚ) * senior_price - 
  ((num_children : ℚ) * child_price + 
   (num_adults : ℚ) * adult_price + 
   (num_seniors : ℚ) * senior_price) * discount_rate = 282.20 := by
sorry

end NUMINAMATH_CALUDE_zoo_visit_cost_l1018_101804


namespace NUMINAMATH_CALUDE_no_integer_solution_l1018_101820

theorem no_integer_solution (P : Polynomial ℤ) (a b c d : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_values : P.eval a = 5 ∧ P.eval b = 5 ∧ P.eval c = 5 ∧ P.eval d = 5) :
  ¬∃ k : ℤ, P.eval k = 8 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1018_101820


namespace NUMINAMATH_CALUDE_inequality_solution_l1018_101808

def inequality_solution_set : Set ℝ :=
  {x | x < -3 ∨ (-3 < x ∧ x < 3)}

theorem inequality_solution :
  {x : ℝ | (x^2 - 9) / (x + 3) < 0} = inequality_solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1018_101808


namespace NUMINAMATH_CALUDE_arcsin_one_over_sqrt_two_l1018_101894

theorem arcsin_one_over_sqrt_two (π : ℝ) : Real.arcsin (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_over_sqrt_two_l1018_101894


namespace NUMINAMATH_CALUDE_max_vertex_sum_l1018_101833

-- Define a cube with numbered faces
def Cube := Fin 6 → ℕ

-- Define a function to get the sum of three faces at a vertex
def vertexSum (c : Cube) (v : Fin 8) : ℕ :=
  sorry -- Implementation details omitted as per instructions

-- Theorem statement
theorem max_vertex_sum (c : Cube) : 
  (∀ v : Fin 8, vertexSum c v ≤ 14) ∧ (∃ v : Fin 8, vertexSum c v = 14) :=
sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l1018_101833


namespace NUMINAMATH_CALUDE_smallest_two_digit_factor_of_5082_l1018_101898

theorem smallest_two_digit_factor_of_5082 (a b : ℕ) 
  (h1 : 10 ≤ a) (h2 : a < b) (h3 : b ≤ 99) (h4 : a * b = 5082) : a = 34 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_factor_of_5082_l1018_101898


namespace NUMINAMATH_CALUDE_project_completion_l1018_101818

theorem project_completion (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (1 : ℚ) / a + (1 : ℚ) / b * 4 = 1) : 
  a + b = 9 ∨ a + b = 10 := by
sorry

end NUMINAMATH_CALUDE_project_completion_l1018_101818


namespace NUMINAMATH_CALUDE_price_increase_for_constant_revenue_l1018_101839

/-- Proves that a 25% price increase is necessary to maintain constant revenue when demand decreases by 20% --/
theorem price_increase_for_constant_revenue 
  (original_price original_demand : ℝ) 
  (new_demand : ℝ) 
  (h_demand_decrease : new_demand = 0.8 * original_demand) 
  (h_revenue_constant : original_price * original_demand = (original_price * (1 + 0.25)) * new_demand) :
  (original_price * (1 + 0.25) - original_price) / original_price = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_price_increase_for_constant_revenue_l1018_101839


namespace NUMINAMATH_CALUDE_second_month_sale_proof_l1018_101855

/-- Calculates the sale in the second month given the sales of other months and the required total sales. -/
def second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (total_sales : ℕ) : ℕ :=
  total_sales - (first_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Proves that the sale in the second month is 11690 given the specific sales figures. -/
theorem second_month_sale_proof :
  second_month_sale 5266 5678 6029 5922 4937 33600 = 11690 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_proof_l1018_101855


namespace NUMINAMATH_CALUDE_carrie_cake_days_l1018_101856

/-- Proves that Carrie worked 4 days on the cake given the specified conditions. -/
theorem carrie_cake_days : 
  ∀ (hours_per_day : ℕ) (hourly_rate : ℕ) (supply_cost : ℕ) (profit : ℕ),
    hours_per_day = 2 →
    hourly_rate = 22 →
    supply_cost = 54 →
    profit = 122 →
    ∃ (days : ℕ), 
      days = 4 ∧ 
      profit = hours_per_day * hourly_rate * days - supply_cost :=
by
  sorry


end NUMINAMATH_CALUDE_carrie_cake_days_l1018_101856


namespace NUMINAMATH_CALUDE_roots_polynomial_sum_l1018_101835

theorem roots_polynomial_sum (α β : ℝ) : 
  (α^2 - 3*α + 1 = 0) → 
  (β^2 - 3*β + 1 = 0) → 
  3*α^3 + 7*β^4 = 448 := by
sorry

end NUMINAMATH_CALUDE_roots_polynomial_sum_l1018_101835


namespace NUMINAMATH_CALUDE_pure_imaginary_square_root_l1018_101899

theorem pure_imaginary_square_root (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ b : ℝ, (1 + a * Complex.I)^2 = b * Complex.I) →
  (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_root_l1018_101899


namespace NUMINAMATH_CALUDE_sophie_uses_one_sheet_per_load_l1018_101802

-- Define the given conditions
def loads_per_week : ℕ := 4
def box_cost : ℚ := 5.5
def sheets_per_box : ℕ := 104
def yearly_savings : ℚ := 11

-- Define the function to calculate the number of dryer sheets per load
def dryer_sheets_per_load : ℚ :=
  (yearly_savings / box_cost) * sheets_per_box / (loads_per_week * 52)

-- Theorem statement
theorem sophie_uses_one_sheet_per_load : 
  dryer_sheets_per_load = 1 := by sorry

end NUMINAMATH_CALUDE_sophie_uses_one_sheet_per_load_l1018_101802


namespace NUMINAMATH_CALUDE_school_bus_problem_l1018_101897

theorem school_bus_problem (total_distance bus_speed walking_speed rest_time : ℝ) 
  (h_total : total_distance = 21)
  (h_bus : bus_speed = 60)
  (h_walk : walking_speed = 4)
  (h_rest : rest_time = 1/6) :
  ∃ (x : ℝ), 
    (x = 19 ∧ total_distance - x = 2) ∧ 
    ((total_distance - x) / walking_speed + rest_time = (total_distance + x) / bus_speed) :=
by sorry

end NUMINAMATH_CALUDE_school_bus_problem_l1018_101897


namespace NUMINAMATH_CALUDE_b_hire_charges_l1018_101815

/-- Calculates the hire charges for a specific person given the total cost,
    and the hours used by each person. -/
def hireCharges (totalCost : ℚ) (hoursA hoursB hoursC : ℚ) : ℚ :=
  let totalHours := hoursA + hoursB + hoursC
  let costPerHour := totalCost / totalHours
  costPerHour * hoursB

theorem b_hire_charges :
  hireCharges 720 9 10 13 = 225 := by
  sorry

end NUMINAMATH_CALUDE_b_hire_charges_l1018_101815


namespace NUMINAMATH_CALUDE_inequality_proof_l1018_101827

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a + b > |c - d|) 
  (h2 : c + d > |a - b|) : 
  a + c > |b - d| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1018_101827


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1018_101854

/-- The quadratic function g(x) = x^2 + 2bx + 2b -/
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x + 2*b

/-- The theorem stating the condition for exactly one solution -/
theorem unique_solution_condition (b : ℝ) :
  (∃! x : ℝ, |g b x| ≤ 3) ↔ (b = 3 ∨ b = -1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1018_101854


namespace NUMINAMATH_CALUDE_gold_coin_problem_l1018_101813

theorem gold_coin_problem (c : ℕ+) (h1 : 8 * (c - 1) = 5 * c + 4) : 
  ∃ n : ℕ, n = 24 ∧ 8 * (c - 1) = n ∧ 5 * c + 4 = n :=
sorry

end NUMINAMATH_CALUDE_gold_coin_problem_l1018_101813


namespace NUMINAMATH_CALUDE_log2_order_relation_l1018_101809

-- Define the logarithm function with base 2
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log2_order_relation :
  (∀ a b : ℝ, f a > f b → a > b) ∧
  ¬(∀ a b : ℝ, a > b → f a > f b) :=
sorry

end NUMINAMATH_CALUDE_log2_order_relation_l1018_101809


namespace NUMINAMATH_CALUDE_equation_proof_l1018_101821

theorem equation_proof : 144 + 2 * 12 * 7 + 49 = 361 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1018_101821


namespace NUMINAMATH_CALUDE_father_son_walk_l1018_101852

/-- The distance traveled when two people with different step lengths walk together -/
def distanceTraveled (fatherStepLength sonStepLength : ℕ) (coincidences : ℕ) : ℕ :=
  let lcm := Nat.lcm fatherStepLength sonStepLength
  (coincidences - 1) * lcm

theorem father_son_walk (fatherStepLength sonStepLength coincidences : ℕ)
  (h1 : fatherStepLength = 80)
  (h2 : sonStepLength = 60)
  (h3 : coincidences = 601) :
  distanceTraveled fatherStepLength sonStepLength coincidences = 144000 := by
  sorry

#eval distanceTraveled 80 60 601

end NUMINAMATH_CALUDE_father_son_walk_l1018_101852


namespace NUMINAMATH_CALUDE_expression_simplification_l1018_101823

variables (a b : ℝ)

theorem expression_simplification :
  (3*a + 2*b - 5*a - b = -2*a + b) ∧
  (5*(3*a^2*b - a*b^2) - (a*b^2 + 3*a^2*b) = 12*a^2*b - 6*a*b^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1018_101823


namespace NUMINAMATH_CALUDE_max_sum_smallest_angles_l1018_101842

/-- A line in a plane --/
structure Line where
  -- We don't need to define the structure of a line for this statement

/-- Represents the configuration of lines in a plane --/
structure LineConfiguration where
  lines : Finset Line
  general_position : Bool

/-- Calculates the sum of smallest angles at intersections --/
def sum_smallest_angles (config : LineConfiguration) : ℝ :=
  sorry

/-- The theorem statement --/
theorem max_sum_smallest_angles :
  ∀ (config : LineConfiguration),
    config.lines.card = 10 ∧ config.general_position →
    ∃ (max_sum : ℝ), 
      (∀ (c : LineConfiguration), c.lines.card = 10 ∧ c.general_position → 
        sum_smallest_angles c ≤ max_sum) ∧
      max_sum = 2250 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_smallest_angles_l1018_101842


namespace NUMINAMATH_CALUDE_pizza_burger_cost_ratio_l1018_101849

/-- The cost ratio of pizza to burger given certain conditions -/
theorem pizza_burger_cost_ratio :
  let burger_cost : ℚ := 9
  let pizza_cost : ℚ → ℚ := λ k => k * burger_cost
  ∀ k : ℚ, pizza_cost k + 3 * burger_cost = 45 →
  pizza_cost k / burger_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_pizza_burger_cost_ratio_l1018_101849


namespace NUMINAMATH_CALUDE_class_average_problem_l1018_101864

theorem class_average_problem (x : ℝ) :
  (0.45 * x + 0.50 * 78 + 0.05 * 60 = 84.75) →
  x = 95 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l1018_101864


namespace NUMINAMATH_CALUDE_two_p_plus_q_value_l1018_101857

theorem two_p_plus_q_value (p q : ℚ) (h : p / q = 2 / 7) : 2 * p + q = (11 / 2) * p := by
  sorry

end NUMINAMATH_CALUDE_two_p_plus_q_value_l1018_101857


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l1018_101882

/-- The ratio of the volume of a cone to the volume of a cylinder with shared base radius -/
theorem cone_cylinder_volume_ratio 
  (r : ℝ) 
  (h_cyl h_cone : ℝ) 
  (h_r : r = 5)
  (h_h_cyl : h_cyl = 18)
  (h_h_cone : h_cone = 9) :
  (1 / 3 * π * r^2 * h_cone) / (π * r^2 * h_cyl) = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l1018_101882


namespace NUMINAMATH_CALUDE_gcd_cube_plus_three_cubed_l1018_101831

theorem gcd_cube_plus_three_cubed (n : ℕ) (h : n > 3) :
  Nat.gcd (n^3 + 3^3) (n + 4) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_three_cubed_l1018_101831


namespace NUMINAMATH_CALUDE_space_diagonals_of_Q_l1018_101865

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  hexagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_line_segments := (Q.vertices.choose 2)
  let non_edge_segments := total_line_segments - Q.edges
  let face_diagonals := Q.hexagonal_faces * 9
  non_edge_segments - face_diagonals

/-- Theorem stating the number of space diagonals in the specific polyhedron Q -/
theorem space_diagonals_of_Q : 
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 32,
    hexagonal_faces := 12
  }
  space_diagonals Q = 255 := by sorry

end NUMINAMATH_CALUDE_space_diagonals_of_Q_l1018_101865


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l1018_101814

theorem monotonic_f_implies_a_range (a : ℝ) :
  (∀ x : ℝ, StrictMono (fun x => x - (1/3) * Real.sin (2*x) + a * Real.sin x)) →
  -1/3 ≤ a ∧ a ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l1018_101814


namespace NUMINAMATH_CALUDE_boxes_in_case_l1018_101805

/-- Given that George has 12 blocks in total, each box holds 6 blocks,
    and George has 2 boxes of blocks, prove that there are 2 boxes in a case. -/
theorem boxes_in_case (total_blocks : ℕ) (blocks_per_box : ℕ) (boxes_of_blocks : ℕ) : 
  total_blocks = 12 → blocks_per_box = 6 → boxes_of_blocks = 2 → 
  (total_blocks / blocks_per_box : ℕ) = boxes_of_blocks := by
  sorry

#check boxes_in_case

end NUMINAMATH_CALUDE_boxes_in_case_l1018_101805


namespace NUMINAMATH_CALUDE_line_equation_proof_l1018_101895

/-- Given a line described by the vector equation (3, -4) · ((x, y) - (-2, 8)) = 0,
    prove that its slope-intercept form y = mx + b has m = 3/4 and b = 19/2. -/
theorem line_equation_proof :
  let vector_eq := fun (x y : ℝ) => 3 * (x + 2) + (-4) * (y - 8) = 0
  ∃ m b : ℝ, (∀ x y : ℝ, vector_eq x y ↔ y = m * x + b) ∧ m = 3/4 ∧ b = 19/2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1018_101895


namespace NUMINAMATH_CALUDE_units_digit_17_pow_2024_l1018_101836

theorem units_digit_17_pow_2024 : (17^2024) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_2024_l1018_101836


namespace NUMINAMATH_CALUDE_flagpole_height_l1018_101845

theorem flagpole_height (wire_ground_distance : Real) (person_distance : Real) (person_height : Real) :
  wire_ground_distance = 5 →
  person_distance = 3 →
  person_height = 1.8 →
  ∃ (flagpole_height : Real),
    flagpole_height = 4.5 ∧
    (flagpole_height / wire_ground_distance = person_height / (wire_ground_distance - person_distance)) :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_l1018_101845


namespace NUMINAMATH_CALUDE_lg_ratio_theorem_l1018_101825

theorem lg_ratio_theorem (m n : ℝ) (hm : Real.log 2 = m) (hn : Real.log 3 = n) :
  (Real.log 12) / (Real.log 15) = (2*m + n) / (1 - m + n) := by
  sorry

end NUMINAMATH_CALUDE_lg_ratio_theorem_l1018_101825


namespace NUMINAMATH_CALUDE_smallest_c_value_l1018_101806

-- Define the polynomial
def polynomial (c d x : ℤ) : ℤ := x^3 - c*x^2 + d*x - 2310

-- Define the property that the polynomial has three positive integer roots
def has_three_positive_integer_roots (c d : ℤ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℤ), r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
    ∀ x, polynomial c d x = (x - r₁) * (x - r₂) * (x - r₃)

-- State the theorem
theorem smallest_c_value (c d : ℤ) :
  has_three_positive_integer_roots c d →
  (∀ c' d', has_three_positive_integer_roots c' d' → c ≤ c') →
  c = 52 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l1018_101806


namespace NUMINAMATH_CALUDE_dice_hidden_sum_l1018_101807

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The number of dice -/
def num_dice : ℕ := 4

/-- The sum of visible numbers -/
def visible_sum : ℕ := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6

/-- The number of visible faces -/
def num_visible : ℕ := 8

theorem dice_hidden_sum :
  (num_dice * die_sum) - visible_sum = 54 :=
by sorry

end NUMINAMATH_CALUDE_dice_hidden_sum_l1018_101807


namespace NUMINAMATH_CALUDE_gumball_problem_l1018_101846

theorem gumball_problem :
  ∀ x : ℕ,
  (19 ≤ (17 + 12 + x) / 3 ∧ (17 + 12 + x) / 3 ≤ 25) →
  (∃ max min : ℕ,
    (∀ y : ℕ, (19 ≤ (17 + 12 + y) / 3 ∧ (17 + 12 + y) / 3 ≤ 25) → y ≤ max) ∧
    (∀ y : ℕ, (19 ≤ (17 + 12 + y) / 3 ∧ (17 + 12 + y) / 3 ≤ 25) → min ≤ y) ∧
    max - min = 18) :=
by sorry

end NUMINAMATH_CALUDE_gumball_problem_l1018_101846


namespace NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l1018_101838

theorem tangent_line_to_logarithmic_curve (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ k * x - 3 = 2 * Real.log x ∧
    (∀ y : ℝ, y > 0 → k * y - 3 ≥ 2 * Real.log y)) →
  k = 2 * Real.sqrt (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l1018_101838


namespace NUMINAMATH_CALUDE_proportion_fourth_term_l1018_101824

theorem proportion_fourth_term (x y : ℝ) : 
  (0.75 : ℝ) / x = 5 / y ∧ x = 1.2 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportion_fourth_term_l1018_101824
