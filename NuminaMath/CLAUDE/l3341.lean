import Mathlib

namespace patio_rearrangement_l3341_334140

theorem patio_rearrangement (r c : ℕ) : 
  r * c = 48 ∧ 
  (r + 4) * (c - 2) = 48 ∧ 
  c > 2 →
  r = 6 :=
by sorry

end patio_rearrangement_l3341_334140


namespace arithmetic_sequence_property_l3341_334172

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_4 + a_8 = 16, a_6 = 8 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 4 + a 8 = 16) : 
  a 6 = 8 := by
sorry

end arithmetic_sequence_property_l3341_334172


namespace binomial_max_prob_l3341_334103

/-- The probability mass function of a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The value of k that maximizes the probability mass function for B(200, 1/2) -/
theorem binomial_max_prob (ζ : ℕ → ℝ) (h : ∀ k, ζ k = binomial_pmf 200 (1/2) k) :
  ∃ k : ℕ, k = 100 ∧ ∀ j : ℕ, ζ k ≥ ζ j :=
sorry

end binomial_max_prob_l3341_334103


namespace same_solutions_imply_coefficients_l3341_334131

-- Define the absolute value equation
def abs_equation (x : ℝ) : Prop := |x - 3| = 4

-- Define the quadratic equation
def quadratic_equation (x b c : ℝ) : Prop := x^2 + b*x + c = 0

-- Theorem statement
theorem same_solutions_imply_coefficients :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ abs_equation x₁ ∧ abs_equation x₂) →
  (∀ x : ℝ, abs_equation x ↔ ∃ b c : ℝ, quadratic_equation x b c) →
  ∃! (b c : ℝ), ∀ x : ℝ, abs_equation x ↔ quadratic_equation x b c :=
by sorry

end same_solutions_imply_coefficients_l3341_334131


namespace factorial_ratio_problem_l3341_334156

theorem factorial_ratio_problem (m n : ℕ) : 
  m > 1 → n > 1 → (Nat.factorial (n + m)) / (Nat.factorial n) = 17297280 → 
  n / m = 1 ∨ n / m = 31 / 2 := by
sorry

end factorial_ratio_problem_l3341_334156


namespace quadratic_inequality_range_l3341_334124

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by
  sorry

end quadratic_inequality_range_l3341_334124


namespace h_domain_l3341_334173

def f_domain : Set ℝ := Set.Icc (-3) 6

def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-3 * x)

theorem h_domain (f : ℝ → ℝ) : 
  {x : ℝ | ∃ y ∈ f_domain, y = -3 * x} = Set.Icc (-2) 1 := by sorry

end h_domain_l3341_334173


namespace combined_storage_temperature_l3341_334174

-- Define the temperature ranges for each type of vegetable
def type_A_range : Set ℝ := {x | 3 ≤ x ∧ x ≤ 8}
def type_B_range : Set ℝ := {x | 5 ≤ x ∧ x ≤ 10}

-- Define the combined suitable temperature range
def combined_range : Set ℝ := type_A_range ∩ type_B_range

-- Theorem to prove
theorem combined_storage_temperature :
  combined_range = {x | 5 ≤ x ∧ x ≤ 8} := by
  sorry

end combined_storage_temperature_l3341_334174


namespace quadratic_equation_solution_l3341_334106

/-- A quadratic equation with parameter m -/
def quadratic_equation (m : ℤ) (x : ℤ) : Prop :=
  m * x^2 - (m + 1) * x + 1 = 0

/-- The property that the equation has two distinct integer roots -/
def has_two_distinct_integer_roots (m : ℤ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

theorem quadratic_equation_solution :
  ∀ m : ℤ, has_two_distinct_integer_roots m → m = -1 :=
by sorry

end quadratic_equation_solution_l3341_334106


namespace fifth_power_sum_l3341_334190

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a*x + b*y = 2)
  (h2 : a*x^2 + b*y^2 = 5)
  (h3 : a*x^3 + b*y^3 = 15)
  (h4 : a*x^4 + b*y^4 = 35) :
  a*x^5 + b*y^5 = 10 := by
sorry

end fifth_power_sum_l3341_334190


namespace complex_magnitude_l3341_334171

/-- Given a real number a, if (a^2 * i) / (1 + i) is imaginary, then |a + i| = √5 -/
theorem complex_magnitude (a : ℝ) : 
  (((a^2 * Complex.I) / (1 + Complex.I)).im ≠ 0 ∧ ((a^2 * Complex.I) / (1 + Complex.I)).re = 0) → 
  Complex.abs (a + Complex.I) = Real.sqrt 5 := by
sorry

end complex_magnitude_l3341_334171


namespace book_price_percentage_l3341_334151

/-- The percentage of the suggested retail price that Bob paid for a book -/
theorem book_price_percentage (suggested_retail_price : ℝ) : 
  suggested_retail_price > 0 →
  let marked_price := 0.6 * suggested_retail_price
  let bob_paid := 0.6 * marked_price
  bob_paid / suggested_retail_price = 0.36 :=
by sorry

end book_price_percentage_l3341_334151


namespace intercepts_sum_l3341_334110

theorem intercepts_sum (x₀ y₀ : ℕ) : 
  x₀ < 42 → y₀ < 42 → 
  (5 * x₀) % 42 = 40 → 
  (3 * y₀) % 42 = 2 → 
  x₀ + y₀ = 36 := by
sorry

end intercepts_sum_l3341_334110


namespace farm_egg_yolks_l3341_334191

/-- Represents the number of yolks in an egg carton -/
def yolks_in_carton (total_eggs : ℕ) (double_yolk_eggs : ℕ) : ℕ :=
  2 * double_yolk_eggs + (total_eggs - double_yolk_eggs)

/-- Theorem: A carton of 12 eggs with 5 double-yolk eggs has 17 yolks in total -/
theorem farm_egg_yolks : yolks_in_carton 12 5 = 17 := by
  sorry

end farm_egg_yolks_l3341_334191


namespace fourth_guard_theorem_l3341_334117

/-- Represents a rectangular perimeter with guards at each corner -/
structure GuardedRectangle where
  perimeter : ℝ
  three_guard_distance : ℝ

/-- Calculates the distance run by the fourth guard -/
def fourth_guard_distance (rect : GuardedRectangle) : ℝ :=
  rect.perimeter - rect.three_guard_distance

/-- Theorem stating that for a rectangle with perimeter 1000 meters,
    if three guards run 850 meters, the fourth guard runs 150 meters -/
theorem fourth_guard_theorem (rect : GuardedRectangle)
  (h1 : rect.perimeter = 1000)
  (h2 : rect.three_guard_distance = 850) :
  fourth_guard_distance rect = 150 := by
  sorry

end fourth_guard_theorem_l3341_334117


namespace game_points_total_l3341_334196

theorem game_points_total (eric_points mark_points samanta_points : ℕ) : 
  eric_points = 6 →
  mark_points = eric_points + eric_points / 2 →
  samanta_points = mark_points + 8 →
  samanta_points + mark_points + eric_points = 32 := by
sorry

end game_points_total_l3341_334196


namespace equal_one_two_digit_prob_l3341_334177

-- Define a 12-sided die
def twelveSidedDie : Finset ℕ := Finset.range 12

-- Define one-digit numbers on the die
def oneDigitNumbers : Finset ℕ := Finset.filter (λ n => n < 10) twelveSidedDie

-- Define two-digit numbers on the die
def twoDigitNumbers : Finset ℕ := Finset.filter (λ n => n ≥ 10) twelveSidedDie

-- Define the probability of rolling a one-digit number
def probOneDigit : ℚ := (oneDigitNumbers.card : ℚ) / (twelveSidedDie.card : ℚ)

-- Define the probability of rolling a two-digit number
def probTwoDigit : ℚ := (twoDigitNumbers.card : ℚ) / (twelveSidedDie.card : ℚ)

-- Theorem stating the probability of rolling 4 dice and getting an equal number of one-digit and two-digit numbers
theorem equal_one_two_digit_prob : 
  (Finset.card oneDigitNumbers * Finset.card twoDigitNumbers * 6 : ℚ) / (twelveSidedDie.card ^ 4 : ℚ) = 27 / 128 :=
by sorry

end equal_one_two_digit_prob_l3341_334177


namespace fourth_power_sum_l3341_334159

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 3)
  (sum_cubes : a^3 + b^3 + c^3 = 6) :
  a^4 + b^4 + c^4 = 34/3 := by
  sorry

end fourth_power_sum_l3341_334159


namespace smallest_addition_for_divisibility_l3341_334121

theorem smallest_addition_for_divisibility : 
  ∃! x : ℕ, x < 169 ∧ (2714 + x) % 169 = 0 ∧ ∀ y : ℕ, y < x → (2714 + y) % 169 ≠ 0 :=
by
  use 119
  sorry

end smallest_addition_for_divisibility_l3341_334121


namespace exam_marks_calculation_l3341_334130

theorem exam_marks_calculation (T : ℕ) : 
  (T * 20 / 100 + 40 = 160) → 
  (T * 30 / 100 - 160 = 20) := by
  sorry

end exam_marks_calculation_l3341_334130


namespace tangent_lines_theorem_intersection_theorem_l3341_334148

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the point M
def point_M : ℝ × ℝ := (3, 1)

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 3
def tangent_line_2 (x y : ℝ) : Prop := 3*x - 4*y - 5 = 0

-- Define the family of lines
def line_family (a x y : ℝ) : Prop := a*x - y + 3 = 0

-- Theorem for tangent lines
theorem tangent_lines_theorem :
  (∀ x y : ℝ, tangent_line_1 x → circle_C x y → x = 3 ∧ y = 1 ∨ x = 3 ∧ y = 3) ∧
  (∀ x y : ℝ, tangent_line_2 x y → circle_C x y → x = 3 ∧ y = 1 ∨ x = 0 ∧ y = 5/4) :=
sorry

-- Theorem for intersection
theorem intersection_theorem :
  ∀ a : ℝ, ∃ x y : ℝ, line_family a x y ∧ circle_C x y :=
sorry

end tangent_lines_theorem_intersection_theorem_l3341_334148


namespace happy_formations_correct_l3341_334123

def happy_formations (n : ℕ) : ℕ :=
  if n % 3 = 1 then 0
  else if n % 3 = 0 then
    (Nat.choose (n-1) (2*n/3) - Nat.choose (n-1) ((2*n+6)/3))^3 +
    (Nat.choose (n-1) ((2*n-3)/3) - Nat.choose (n-1) ((2*n+3)/3))^3
  else
    (Nat.choose (n-1) ((2*n-1)/3) - Nat.choose (n-1) ((2*n+1)/3))^3 +
    (Nat.choose (n-1) ((2*n-4)/3) - Nat.choose (n-1) ((2*n-2)/3))^3

theorem happy_formations_correct (n : ℕ) :
  happy_formations n =
    if n % 3 = 1 then 0
    else if n % 3 = 0 then
      (Nat.choose (n-1) (2*n/3) - Nat.choose (n-1) ((2*n+6)/3))^3 +
      (Nat.choose (n-1) ((2*n-3)/3) - Nat.choose (n-1) ((2*n+3)/3))^3
    else
      (Nat.choose (n-1) ((2*n-1)/3) - Nat.choose (n-1) ((2*n+1)/3))^3 +
      (Nat.choose (n-1) ((2*n-4)/3) - Nat.choose (n-1) ((2*n-2)/3))^3 :=
by sorry

end happy_formations_correct_l3341_334123


namespace no_sum_equal_powers_l3341_334184

theorem no_sum_equal_powers : ¬∃ (n m : ℕ), n * (n + 1) / 2 = 2^m + 3^m := by
  sorry

end no_sum_equal_powers_l3341_334184


namespace cubic_sum_l3341_334192

theorem cubic_sum (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a * b + a * c + b * c = 7) 
  (h3 : a * b * c = -18) : 
  a^3 + b^3 + c^3 = 29 := by
sorry


end cubic_sum_l3341_334192


namespace largest_four_digit_divisible_by_98_l3341_334194

theorem largest_four_digit_divisible_by_98 : 
  ∀ n : ℕ, n ≤ 9998 ∧ n ≥ 1000 ∧ n % 98 = 0 → n ≤ 9998 :=
by
  sorry

end largest_four_digit_divisible_by_98_l3341_334194


namespace interval_intersection_l3341_334185

theorem interval_intersection (x : ℝ) : 
  (2 < 3*x ∧ 3*x < 3) ∧ (2 < 4*x ∧ 4*x < 3) ↔ (2/3 < x ∧ x < 3/4) := by
  sorry

end interval_intersection_l3341_334185


namespace second_term_of_geometric_series_l3341_334199

theorem second_term_of_geometric_series (a : ℝ) (r : ℝ) (S : ℝ) : 
  r = (1 : ℝ) / 4 →
  S = 16 →
  S = a / (1 - r) →
  a * r = 3 :=
by sorry

end second_term_of_geometric_series_l3341_334199


namespace janet_freelance_income_difference_l3341_334101

/-- Calculates how much more Janet would make per month as a freelancer compared to her current job -/
theorem janet_freelance_income_difference :
  let hours_per_week : ℕ := 40
  let weeks_per_month : ℕ := 4
  let current_hourly_rate : ℚ := 30
  let freelance_hourly_rate : ℚ := 40
  let extra_fica_per_week : ℚ := 25
  let healthcare_premium_per_month : ℚ := 400

  let current_monthly_income := (hours_per_week * weeks_per_month : ℚ) * current_hourly_rate
  let freelance_monthly_income := (hours_per_week * weeks_per_month : ℚ) * freelance_hourly_rate
  let extra_costs_per_month := extra_fica_per_week * weeks_per_month + healthcare_premium_per_month

  freelance_monthly_income - extra_costs_per_month - current_monthly_income = 1100 :=
by sorry

end janet_freelance_income_difference_l3341_334101


namespace equation_positive_root_l3341_334182

theorem equation_positive_root (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (3 * x - 1) / (x + 1) - m / (x + 1) = 1) → m = -4 := by
  sorry

end equation_positive_root_l3341_334182


namespace complex_cube_root_unity_l3341_334104

theorem complex_cube_root_unity (i : ℂ) (y : ℂ) :
  i^2 = -1 →
  y = (1 + i * Real.sqrt 3) / 2 →
  1 / (y^3 - y) = -1/2 + (i * Real.sqrt 3) / 6 := by sorry

end complex_cube_root_unity_l3341_334104


namespace gcd_102_238_l3341_334127

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l3341_334127


namespace same_day_ticket_cost_l3341_334168

/-- Proves that the cost of same-day tickets is $30 given the specified conditions -/
theorem same_day_ticket_cost
  (total_tickets : ℕ)
  (total_receipts : ℕ)
  (advance_ticket_cost : ℕ)
  (advance_tickets_sold : ℕ)
  (h1 : total_tickets = 60)
  (h2 : total_receipts = 1600)
  (h3 : advance_ticket_cost = 20)
  (h4 : advance_tickets_sold = 20) :
  (total_receipts - advance_ticket_cost * advance_tickets_sold) / (total_tickets - advance_tickets_sold) = 30 :=
by sorry

end same_day_ticket_cost_l3341_334168


namespace min_value_quadratic_l3341_334157

theorem min_value_quadratic (x : ℝ) : 
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 4 * x^2 + 8 * x + 16 → y ≥ y_min ∧ y_min = 12 :=
by sorry

end min_value_quadratic_l3341_334157


namespace intersection_A_complement_B_l3341_334111

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x ≥ 1}

-- Define the complement of B in the universal set (real numbers)
def C_U_B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_A_complement_B : A ∩ C_U_B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_A_complement_B_l3341_334111


namespace f_minus_g_at_one_l3341_334132

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)
def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h x = -h (-x)

-- State the theorem
theorem f_minus_g_at_one
  (h1 : is_even f)
  (h2 : is_odd g)
  (h3 : ∀ x, f x + g x = x^3 + x^2 + 1) :
  f 1 - g 1 = 1 := by sorry

end f_minus_g_at_one_l3341_334132


namespace max_sum_of_squares_l3341_334163

theorem max_sum_of_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - (k-2)*x₁ + (k^2 + 3*k + 5) = 0 →
  x₂^2 - (k-2)*x₂ + (k^2 + 3*k + 5) = 0 →
  x₁ ≠ x₂ →
  ∃ (M : ℝ), M = 18 ∧ x₁^2 + x₂^2 ≤ M :=
by sorry

end max_sum_of_squares_l3341_334163


namespace exists_surjective_function_with_property_l3341_334141

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x else x - 1

-- State the theorem
theorem exists_surjective_function_with_property :
  ∃ (f : ℝ → ℝ), Function.Surjective f ∧
  (∀ x y : ℝ, (f (x + y) - f x - f y) ∈ ({0, 1} : Set ℝ)) :=
by
  -- The proof would go here
  sorry

end exists_surjective_function_with_property_l3341_334141


namespace marble_fraction_l3341_334100

theorem marble_fraction (total : ℝ) (h : total > 0) : 
  let initial_blue := (2/3) * total
  let initial_red := total - initial_blue
  let new_blue := 2 * initial_blue
  let new_red := 3 * initial_red
  let new_total := new_blue + new_red
  new_red / new_total = 3/7 := by sorry

end marble_fraction_l3341_334100


namespace line_with_45_degree_slope_l3341_334129

/-- Given a line passing through points (1, -2) and (a, 3) with a slope angle of 45°, 
    the value of a is 6. -/
theorem line_with_45_degree_slope (a : ℝ) : 
  (((3 - (-2)) / (a - 1) = Real.tan (π / 4)) → a = 6) :=
by sorry

end line_with_45_degree_slope_l3341_334129


namespace reading_time_is_fifty_l3341_334167

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of paragraphs per page -/
def paragraphs_per_page : ℕ := 20

/-- Represents the number of sentences per paragraph -/
def sentences_per_paragraph : ℕ := 10

/-- Represents the total number of pages in the book -/
def total_pages : ℕ := 50

/-- Calculates the total time in hours needed to read the book -/
def reading_time : ℚ :=
  (total_pages * paragraphs_per_page * sentences_per_paragraph) / reading_speed

theorem reading_time_is_fifty : reading_time = 50 := by
  sorry

end reading_time_is_fifty_l3341_334167


namespace extreme_values_and_tangent_line_l3341_334137

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8

/-- The derivative of f(x) with respect to x -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_tangent_line (a b : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (a = -3 ∧ b = 4) ∧
  (∃ k m : ℝ, k * 0 + m = f (-3) 4 0 ∧ k = f' (-3) 4 0 ∧ k = 12 ∧ m = 8) :=
by sorry

end extreme_values_and_tangent_line_l3341_334137


namespace edwin_alvin_age_fraction_l3341_334175

/-- The fraction of Alvin's age that Edwin will be 20 more than in two years -/
def fraction_of_alvins_age : ℚ := 1 / 29

theorem edwin_alvin_age_fraction :
  let alvin_current_age : ℚ := (30.99999999 - 6) / 2
  let edwin_current_age : ℚ := alvin_current_age + 6
  let alvin_future_age : ℚ := alvin_current_age + 2
  let edwin_future_age : ℚ := edwin_current_age + 2
  edwin_future_age = fraction_of_alvins_age * alvin_future_age + 20 :=
by sorry

end edwin_alvin_age_fraction_l3341_334175


namespace married_men_fraction_l3341_334166

theorem married_men_fraction (total_women : ℕ) (h_total_women_pos : total_women > 0) :
  let single_women : ℕ := (3 * total_women) / 7
  let married_women : ℕ := total_women - single_women
  let married_men : ℕ := married_women
  let total_people : ℕ := total_women + married_men
  (↑single_women : ℚ) / total_women = 3 / 7 →
  (↑married_men : ℚ) / total_people = 4 / 11 :=
by sorry

end married_men_fraction_l3341_334166


namespace california_texas_plates_equal_l3341_334145

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^3 * num_digits^3

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_digits^3 * num_letters^3

/-- Theorem stating that California and Texas can issue the same number of license plates -/
theorem california_texas_plates_equal : california_plates = texas_plates := by
  sorry

end california_texas_plates_equal_l3341_334145


namespace isosceles_triangle_properties_l3341_334122

/-- An isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  /-- Length of the equal sides -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : x ≠ y
  /-- The perimeter is 20 -/
  perimeter : x + x + y = 20

/-- Properties of the isosceles triangle -/
theorem isosceles_triangle_properties (t : IsoscelesTriangle) :
  (t.y = -2 * t.x + 20) ∧ (5 < t.x ∧ t.x < 10) := by
  sorry

end isosceles_triangle_properties_l3341_334122


namespace sally_has_more_cards_l3341_334160

theorem sally_has_more_cards (sally_initial : ℕ) (sally_bought : ℕ) (dan_cards : ℕ)
  (h1 : sally_initial = 27)
  (h2 : sally_bought = 20)
  (h3 : dan_cards = 41) :
  sally_initial + sally_bought - dan_cards = 6 := by
  sorry

end sally_has_more_cards_l3341_334160


namespace perpendicular_lines_k_values_l3341_334150

/-- Given two lines l₁ and l₂ defined by their equations, 
    this theorem states that if they are perpendicular, 
    then k must be either 0 or 3. -/
theorem perpendicular_lines_k_values 
  (k : ℝ) 
  (l₁ : ℝ → ℝ → Prop) 
  (l₂ : ℝ → ℝ → Prop) 
  (h₁ : ∀ x y, l₁ x y ↔ x + k * y - 2 * k = 0) 
  (h₂ : ∀ x y, l₂ x y ↔ k * x - (k - 2) * y + 1 = 0) 
  (h_perp : (∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₂ x₂ y₂ → (x₂ - x₁) * (y₂ - y₁) = 0)) : 
  k = 0 ∨ k = 3 := by
sorry

end perpendicular_lines_k_values_l3341_334150


namespace pyramid_height_equals_cube_volume_l3341_334193

/-- The height of a square-based pyramid with the same volume as a cube -/
theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (h : ℝ) : 
  cube_edge = 5 →
  pyramid_base = 10 →
  (1 / 3) * pyramid_base^2 * h = cube_edge^3 →
  h = 3.75 := by
sorry

end pyramid_height_equals_cube_volume_l3341_334193


namespace blue_pill_cost_is_correct_l3341_334165

/-- The cost of the blue pill in dollars -/
def blue_pill_cost : ℝ := 17

/-- The cost of the red pill in dollars -/
def red_pill_cost : ℝ := blue_pill_cost - 2

/-- The number of days for the treatment -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 672

theorem blue_pill_cost_is_correct :
  blue_pill_cost = 17 ∧
  red_pill_cost = blue_pill_cost - 2 ∧
  treatment_days * (blue_pill_cost + red_pill_cost) = total_cost := by
  sorry

#eval blue_pill_cost

end blue_pill_cost_is_correct_l3341_334165


namespace jake_balloons_l3341_334158

def total_balloons : ℕ := 3
def allan_balloons : ℕ := 2

theorem jake_balloons : total_balloons - allan_balloons = 1 := by
  sorry

end jake_balloons_l3341_334158


namespace remainder_3_pow_19_mod_10_l3341_334164

theorem remainder_3_pow_19_mod_10 : (3^19) % 10 = 7 := by
  sorry

end remainder_3_pow_19_mod_10_l3341_334164


namespace min_games_for_equal_play_l3341_334107

/-- Represents a bridge game between 2 guys and 2 girls -/
structure BridgeGame where
  guys : Fin 2 → Fin 5
  girls : Fin 2 → Fin 5

/-- The minimum number of games required for the given conditions -/
def minGames : Nat := 25

/-- Checks if a set of games satisfies the equal play condition -/
def satisfiesEqualPlay (games : List BridgeGame) : Prop :=
  ∀ (guy : Fin 5) (girl : Fin 5),
    (games.filter (λ g => g.guys 0 = guy ∨ g.guys 1 = guy)).length =
    (games.filter (λ g => g.girls 0 = girl ∨ g.girls 1 = girl)).length

theorem min_games_for_equal_play :
  ∀ (games : List BridgeGame),
    satisfiesEqualPlay games →
    games.length ≥ minGames :=
  sorry

end min_games_for_equal_play_l3341_334107


namespace equation_not_linear_l3341_334153

/-- A linear equation in two variables contains exactly two variables and the highest degree of terms involving these variables is 1. -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, f x y = a * x + b * y + c

/-- The equation xy = 3 -/
def equation (x y : ℝ) : ℝ := x * y - 3

theorem equation_not_linear : ¬ is_linear_equation_in_two_variables equation := by
  sorry

end equation_not_linear_l3341_334153


namespace power_of_four_exponent_l3341_334105

theorem power_of_four_exponent (n : ℕ) (x : ℕ) 
  (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^x) 
  (hn : n = 25) : x = 26 := by
  sorry

end power_of_four_exponent_l3341_334105


namespace alf3_weight_calculation_l3341_334188

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_F : ℝ := 19.00

-- Define the number of moles
def num_moles : ℝ := 7

-- Define the molecular weight calculation function
def molecular_weight (al_weight f_weight : ℝ) : ℝ :=
  al_weight + 3 * f_weight

-- Define the total weight calculation function
def total_weight (mol_weight num_mol : ℝ) : ℝ :=
  mol_weight * num_mol

-- Theorem statement
theorem alf3_weight_calculation :
  total_weight (molecular_weight atomic_weight_Al atomic_weight_F) num_moles = 587.86 := by
  sorry


end alf3_weight_calculation_l3341_334188


namespace triangle_max_area_l3341_334186

noncomputable def triangle_area (x : Real) : Real :=
  4 * Real.sqrt 3 * Real.sin x * Real.sin ((2 * Real.pi / 3) - x)

theorem triangle_max_area :
  ∀ x : Real, 0 < x → x < 2 * Real.pi / 3 →
    triangle_area x ≤ triangle_area (Real.pi / 3) ∧
    triangle_area (Real.pi / 3) = 3 * Real.sqrt 3 :=
by sorry

end triangle_max_area_l3341_334186


namespace three_circles_sum_l3341_334195

theorem three_circles_sum (triangle circle : ℚ) 
  (eq1 : 5 * triangle + 2 * circle = 27)
  (eq2 : 2 * triangle + 5 * circle = 29) :
  3 * circle = 13 := by
sorry

end three_circles_sum_l3341_334195


namespace relay_race_proof_l3341_334133

-- Define the total race distance
def total_distance : ℕ := 2004

-- Define the maximum time allowed (one week in hours)
def max_time : ℕ := 168

-- Define the properties of the race
theorem relay_race_proof :
  ∃ (stage_length : ℕ) (num_stages : ℕ),
    stage_length > 0 ∧
    num_stages > 0 ∧
    num_stages ≤ max_time ∧
    stage_length * num_stages = total_distance ∧
    num_stages = 167 :=
by sorry

end relay_race_proof_l3341_334133


namespace joe_weight_lifting_l3341_334142

theorem joe_weight_lifting (first_lift second_lift : ℕ) 
  (h1 : first_lift + second_lift = 1500)
  (h2 : 2 * first_lift = second_lift + 300) :
  first_lift = 600 := by
sorry

end joe_weight_lifting_l3341_334142


namespace system_solution_l3341_334180

/-- Given a system of equations, prove the solutions. -/
theorem system_solution (a b c : ℝ) :
  let eq1 := (y : ℝ) ^ 2 - z * x = a * (x + y + z) ^ 2
  let eq2 := x ^ 2 - y * z = b * (x + y + z) ^ 2
  let eq3 := z ^ 2 - x * y = c * (x + y + z) ^ 2
  (∃ s : ℝ,
    x = (2 * c - a - b + 1) * s ∧
    y = (2 * a - b - c + 1) * s ∧
    z = (2 * b - c - a + 1) * s ∧
    a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a = a + b + c) ∨
  (x = 0 ∧ y = 0 ∧ z = 0 ∧
    a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a ≠ a + b + c) :=
by
  sorry

end system_solution_l3341_334180


namespace largest_prime_to_test_for_500_to_550_l3341_334183

theorem largest_prime_to_test_for_500_to_550 (n : ℕ) :
  500 ≤ n ∧ n ≤ 550 →
  (∀ p : ℕ, Prime p ∧ p ≤ Real.sqrt n → p ≤ 23) ∧
  Prime 23 ∧ 23 ≤ Real.sqrt n :=
sorry

end largest_prime_to_test_for_500_to_550_l3341_334183


namespace stratified_sampling_management_l3341_334152

theorem stratified_sampling_management (total_employees : ℕ) (management_personnel : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 160)
  (h2 : management_personnel = 32)
  (h3 : sample_size = 20) :
  (management_personnel : ℚ) * (sample_size : ℚ) / (total_employees : ℚ) = 4 := by
  sorry

end stratified_sampling_management_l3341_334152


namespace inequality_minimum_l3341_334119

theorem inequality_minimum (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → x * y ≤ a * x^2 + 2 * y^2) → 
  a ≥ -1 := by
sorry

end inequality_minimum_l3341_334119


namespace simplify_and_evaluate_l3341_334135

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2023 + 3) :
  (3 / (x + 3) - 1) / (x / (x^2 - 9)) = -Real.sqrt 2023 := by
  sorry

end simplify_and_evaluate_l3341_334135


namespace parameterization_validity_l3341_334170

def line (x : ℝ) : ℝ := -3 * x + 4

def is_valid_parameterization (p : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line (p.1 + t * v.1) = p.2 + t * v.2

theorem parameterization_validity :
  (is_valid_parameterization (0, 4) (1, -3)) ∧
  (is_valid_parameterization (-4, 16) (1/3, -1)) ∧
  ¬(is_valid_parameterization (1/3, 0) (3, -1)) ∧
  ¬(is_valid_parameterization (2, -2) (4, -12)) ∧
  ¬(is_valid_parameterization (1, 1) (0.5, -1.5)) :=
sorry

end parameterization_validity_l3341_334170


namespace fish_problem_l3341_334169

theorem fish_problem (ken_fish : ℕ) (kendra_fish : ℕ) :
  ken_fish = 2 * kendra_fish - 3 →
  ken_fish + kendra_fish = 87 →
  kendra_fish = 30 := by
sorry

end fish_problem_l3341_334169


namespace yellow_balls_count_l3341_334115

theorem yellow_balls_count (purple_count blue_count : ℕ) 
  (min_tries : ℕ) (yellow_count : ℕ) : 
  purple_count = 7 → 
  blue_count = 5 → 
  min_tries = 19 →
  yellow_count = min_tries - (purple_count + blue_count + 1) →
  yellow_count = 6 := by
  sorry

end yellow_balls_count_l3341_334115


namespace lesser_fraction_l3341_334144

theorem lesser_fraction (x y : ℝ) (sum_eq : x + y = 7/8) (prod_eq : x * y = 1/12) :
  min x y = (7 - Real.sqrt 17) / 16 := by
  sorry

end lesser_fraction_l3341_334144


namespace no_negative_roots_l3341_334118

/-- Given f(x) = a^x + (x-2)/(x+1) where a > 1, prove that f(x) ≠ 0 for all x < 0 -/
theorem no_negative_roots (a : ℝ) (h : a > 1) :
  ∀ x : ℝ, x < 0 → a^x + (x - 2) / (x + 1) ≠ 0 := by
  sorry

end no_negative_roots_l3341_334118


namespace line_y_intercept_l3341_334154

/-- A straight line in the xy-plane with slope 2 passing through (269, 540) has y-intercept 2 -/
theorem line_y_intercept (m slope : ℝ) (x₀ y₀ : ℝ) :
  slope = 2 →
  x₀ = 269 →
  y₀ = 540 →
  y₀ = slope * x₀ + m →
  m = 2 := by sorry

end line_y_intercept_l3341_334154


namespace ratio_abc_l3341_334116

theorem ratio_abc (a b c : ℝ) (h : 14 * (a^2 + b^2 + c^2) = (a + 2*b + 3*c)^2) :
  ∃ k : ℝ, a = k ∧ b = 2*k ∧ c = 3*k :=
sorry

end ratio_abc_l3341_334116


namespace total_stones_in_five_piles_l3341_334128

/-- Given five piles of stones with the following properties:
    1. The number of stones in the fifth pile is six times the number of stones in the third pile
    2. The number of stones in the second pile is twice the total number of stones in the third and fifth piles combined
    3. The number of stones in the first pile is three times less than the number in the fifth pile and 10 less than the number in the fourth pile
    4. The number of stones in the fourth pile is half the number in the second pile
    Prove that the total number of stones in all five piles is 60. -/
theorem total_stones_in_five_piles (p1 p2 p3 p4 p5 : ℕ) 
  (h1 : p5 = 6 * p3)
  (h2 : p2 = 2 * (p3 + p5))
  (h3 : p1 = p5 / 3 ∧ p1 = p4 - 10)
  (h4 : p4 = p2 / 2) :
  p1 + p2 + p3 + p4 + p5 = 60 := by
  sorry

end total_stones_in_five_piles_l3341_334128


namespace no_distinct_complex_numbers_satisfying_equation_l3341_334126

theorem no_distinct_complex_numbers_satisfying_equation :
  ∀ (a b c d : ℂ), 
    (a^3 - b*c*d = b^3 - a*c*d) ∧ 
    (b^3 - a*c*d = c^3 - a*b*d) ∧ 
    (c^3 - a*b*d = d^3 - a*b*c) →
    (a = b) ∨ (a = c) ∨ (a = d) ∨ (b = c) ∨ (b = d) ∨ (c = d) :=
by sorry

end no_distinct_complex_numbers_satisfying_equation_l3341_334126


namespace number_ordering_l3341_334112

theorem number_ordering : (2 : ℕ)^30 < (6 : ℕ)^10 ∧ (6 : ℕ)^10 < (3 : ℕ)^20 := by sorry

end number_ordering_l3341_334112


namespace tan_alpha_plus_pi_over_four_l3341_334147

theorem tan_alpha_plus_pi_over_four (α : Real) (h : Real.tan α = 2) :
  Real.tan (α + π / 4) = -3 := by
  sorry

end tan_alpha_plus_pi_over_four_l3341_334147


namespace correct_sqrt_calculation_l3341_334125

theorem correct_sqrt_calculation :
  (∃ (x y z : ℝ), x = Real.sqrt 2 ∧ y = Real.sqrt 3 ∧ z = Real.sqrt 6 ∧ x * y = z) ∧
  (∀ (x y z : ℝ), x = Real.sqrt 2 ∧ y = Real.sqrt 3 ∧ z = Real.sqrt 5 → x + y ≠ z) ∧
  (∀ (x y : ℝ), x = Real.sqrt 3 ∧ y = Real.sqrt 2 → x - y ≠ 1) ∧
  (∀ (x y : ℝ), x = Real.sqrt 4 ∧ y = Real.sqrt 2 → x / y ≠ 2) :=
by sorry


end correct_sqrt_calculation_l3341_334125


namespace total_markers_l3341_334146

theorem total_markers (red_markers blue_markers : ℕ) : 
  red_markers = 2315 → blue_markers = 1028 → red_markers + blue_markers = 3343 := by
  sorry

end total_markers_l3341_334146


namespace sum_of_exponents_product_divisors_360000_l3341_334109

/-- The product of all positive integer divisors of a natural number n -/
def product_of_divisors (n : ℕ) : ℕ := sorry

/-- The sum of exponents in the prime factorization of a natural number n -/
def sum_of_exponents (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of exponents in the prime factorization of the product of all positive integer divisors of 360000 is 630 -/
theorem sum_of_exponents_product_divisors_360000 :
  sum_of_exponents (product_of_divisors 360000) = 630 := by sorry

end sum_of_exponents_product_divisors_360000_l3341_334109


namespace max_truthful_students_2015_l3341_334136

/-- The maximum number of truthful students in the described arrangement --/
def max_truthful_students (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that for n = 2015, the maximum number of truthful students is 2031120 --/
theorem max_truthful_students_2015 :
  max_truthful_students 2015 = 2031120 := by
  sorry

#eval max_truthful_students 2015

end max_truthful_students_2015_l3341_334136


namespace ratatouille_cost_per_quart_l3341_334178

/-- Calculate the cost per quart of ratatouille --/
theorem ratatouille_cost_per_quart :
  let eggplant_weight : ℝ := 5.5
  let eggplant_price : ℝ := 2.20
  let zucchini_weight : ℝ := 3.8
  let zucchini_price : ℝ := 1.85
  let tomato_weight : ℝ := 4.6
  let tomato_price : ℝ := 3.75
  let onion_weight : ℝ := 2.7
  let onion_price : ℝ := 1.10
  let basil_weight : ℝ := 1.0
  let basil_price : ℝ := 2.70 * 4  -- Price per pound (4 quarters)
  let pepper_weight : ℝ := 0.75
  let pepper_price : ℝ := 3.15
  let total_yield : ℝ := 4.5

  let total_cost : ℝ := 
    eggplant_weight * eggplant_price +
    zucchini_weight * zucchini_price +
    tomato_weight * tomato_price +
    onion_weight * onion_price +
    basil_weight * basil_price +
    pepper_weight * pepper_price

  let cost_per_quart : ℝ := total_cost / total_yield

  cost_per_quart = 11.67 := by sorry

end ratatouille_cost_per_quart_l3341_334178


namespace factorization_xy_squared_minus_x_l3341_334187

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end factorization_xy_squared_minus_x_l3341_334187


namespace marias_piggy_bank_l3341_334189

/-- Represents the number of coins of each type -/
structure CoinCount where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- Calculates the total value of coins in dollars -/
def totalValue (coins : CoinCount) : ℚ :=
  0.10 * coins.dimes + 0.25 * coins.quarters + 0.05 * coins.nickels

/-- The problem statement -/
theorem marias_piggy_bank (initialCoins : CoinCount) :
  initialCoins.dimes = 4 →
  initialCoins.quarters = 4 →
  totalValue { dimes := initialCoins.dimes,
               quarters := initialCoins.quarters + 5,
               nickels := initialCoins.nickels } = 3 →
  initialCoins.nickels = 7 := by
  sorry

end marias_piggy_bank_l3341_334189


namespace system_solution_equation_solution_l3341_334138

-- Part 1: System of equations
theorem system_solution :
  ∃! (x y : ℝ), (2 * x - y = 3) ∧ (x + y = -12) ∧ (x = -3) ∧ (y = -9) := by
  sorry

-- Part 2: Single equation
theorem equation_solution :
  ∃! x : ℝ, (2 / (1 - x) + 1 = x / (1 + x)) ∧ (x = -3) := by
  sorry

end system_solution_equation_solution_l3341_334138


namespace number_solution_l3341_334197

theorem number_solution : ∃ x : ℝ, (35 - 3 * x = 14) ∧ (x = 7) := by sorry

end number_solution_l3341_334197


namespace statement_independent_of_parallel_postulate_l3341_334162

-- Define a geometry
class Geometry where
  -- Define the concept of a line
  Line : Type
  -- Define the concept of a point
  Point : Type
  -- Define the concept of parallelism
  parallel : Line → Line → Prop
  -- Define the concept of intersection
  intersects : Line → Line → Prop

-- Define the statement to be proven
def statement (G : Geometry) : Prop :=
  ∀ (l₁ l₂ l₃ : G.Line),
    G.parallel l₁ l₂ → G.intersects l₃ l₁ → G.intersects l₃ l₂

-- Define the parallel postulate
def parallel_postulate (G : Geometry) : Prop :=
  ∀ (p : G.Point) (l : G.Line),
    ∃! (m : G.Line), G.parallel l m

-- Theorem: The statement is independent of the parallel postulate
theorem statement_independent_of_parallel_postulate :
  ∀ (G : Geometry),
    (statement G ↔ statement G) ∧ 
    (¬(parallel_postulate G → statement G)) ∧
    (¬(statement G → parallel_postulate G)) :=
sorry

end statement_independent_of_parallel_postulate_l3341_334162


namespace min_squares_15_step_staircase_l3341_334120

/-- Represents a staircase with a given number of steps -/
structure Staircase :=
  (steps : ℕ)

/-- The minimum number of squares required to cover a staircase -/
def min_squares_to_cover (s : Staircase) : ℕ := s.steps

/-- Theorem: The minimum number of squares required to cover a 15-step staircase is 15 -/
theorem min_squares_15_step_staircase :
  ∀ (s : Staircase), s.steps = 15 → min_squares_to_cover s = 15 := by
  sorry

/-- Lemma: Cutting can only be done along the boundaries of the cells -/
lemma cut_along_boundaries (s : Staircase) : True := by
  sorry

/-- Lemma: Each step in the staircase forms a unit square -/
lemma step_is_unit_square (s : Staircase) : True := by
  sorry

end min_squares_15_step_staircase_l3341_334120


namespace angle_negative_2015_in_second_quadrant_l3341_334155

/-- The quadrant of an angle in degrees -/
inductive Quadrant
| first
| second
| third
| fourth

/-- Determine the quadrant of an angle in degrees -/
def angleQuadrant (angle : ℤ) : Quadrant :=
  let normalizedAngle := angle % 360
  if 0 ≤ normalizedAngle && normalizedAngle < 90 then Quadrant.first
  else if 90 ≤ normalizedAngle && normalizedAngle < 180 then Quadrant.second
  else if 180 ≤ normalizedAngle && normalizedAngle < 270 then Quadrant.third
  else Quadrant.fourth

theorem angle_negative_2015_in_second_quadrant :
  angleQuadrant (-2015) = Quadrant.second := by
  sorry

end angle_negative_2015_in_second_quadrant_l3341_334155


namespace camp_skills_l3341_334143

theorem camp_skills (total : ℕ) (cant_sing cant_dance cant_perform : ℕ) :
  total = 100 ∧
  cant_sing = 42 ∧
  cant_dance = 65 ∧
  cant_perform = 29 →
  ∃ (only_sing only_dance only_perform sing_dance sing_perform dance_perform : ℕ),
    only_sing + only_dance + only_perform + sing_dance + sing_perform + dance_perform = total ∧
    only_dance + only_perform + dance_perform = cant_sing ∧
    only_sing + only_perform + sing_perform = cant_dance ∧
    only_sing + only_dance + sing_dance = cant_perform ∧
    sing_dance + sing_perform + dance_perform = 64 :=
by sorry

end camp_skills_l3341_334143


namespace inequality_solution_set_l3341_334198

theorem inequality_solution_set (x : ℝ) :
  (x^2 + x - 6 ≤ 0) ↔ (-3 ≤ x ∧ x ≤ 2) := by sorry

end inequality_solution_set_l3341_334198


namespace percentage_of_women_in_study_group_l3341_334176

theorem percentage_of_women_in_study_group :
  let percentage_women_lawyers : ℝ := 0.4
  let prob_woman_lawyer : ℝ := 0.32
  let percentage_women : ℝ := prob_woman_lawyer / percentage_women_lawyers
  percentage_women = 0.8 := by sorry

end percentage_of_women_in_study_group_l3341_334176


namespace park_walking_area_l3341_334161

/-- The area available for walking in a rectangular park with a centered circular fountain -/
theorem park_walking_area (park_length park_width fountain_radius : ℝ) 
  (h1 : park_length = 50)
  (h2 : park_width = 30)
  (h3 : fountain_radius = 5) : 
  park_length * park_width - Real.pi * fountain_radius^2 = 1500 - 25 * Real.pi := by
  sorry

end park_walking_area_l3341_334161


namespace prob_higher_roll_and_sum_l3341_334113

/-- The number of sides on a standard die -/
def die_sides : ℕ := 6

/-- The probability of rolling a higher number on one die compared to another -/
def prob_higher_roll : ℚ :=
  (die_sides * (die_sides - 1) / 2) / (die_sides^2 : ℚ)

/-- The sum of the numerator and denominator of the probability fraction in lowest terms -/
def sum_num_denom : ℕ := 17

theorem prob_higher_roll_and_sum :
  prob_higher_roll = 5/12 ∧ sum_num_denom = 17 := by sorry

end prob_higher_roll_and_sum_l3341_334113


namespace broken_flagpole_theorem_l3341_334181

/-- Represents a broken flagpole -/
structure BrokenFlagpole where
  initial_height : ℝ
  tip_height : ℝ
  break_point : ℝ

/-- The condition for a valid broken flagpole configuration -/
def is_valid_broken_flagpole (f : BrokenFlagpole) : Prop :=
  f.initial_height > 0 ∧
  f.tip_height > 0 ∧
  f.tip_height < f.initial_height ∧
  f.break_point > 0 ∧
  f.break_point < f.initial_height ∧
  (f.initial_height - f.break_point) * 2 = f.initial_height - f.tip_height

theorem broken_flagpole_theorem (f : BrokenFlagpole)
  (h_valid : is_valid_broken_flagpole f)
  (h_initial : f.initial_height = 12)
  (h_tip : f.tip_height = 2) :
  f.break_point = 7 := by
sorry

end broken_flagpole_theorem_l3341_334181


namespace job_completion_time_l3341_334134

theorem job_completion_time (job : ℝ) (days_A : ℝ) (efficiency_C : ℝ) : 
  job > 0 → days_A > 0 → efficiency_C > 0 →
  (job / days_A) * efficiency_C * 16 = job :=
by
  sorry

#check job_completion_time

end job_completion_time_l3341_334134


namespace max_bottles_from_C_and_D_l3341_334149

/-- Represents the shops selling recyclable bottles -/
inductive Shop
| A
| B
| C
| D

/-- The price of a bottle at each shop -/
def price (s : Shop) : ℕ :=
  match s with
  | Shop.A => 1
  | Shop.B => 2
  | Shop.C => 3
  | Shop.D => 5

/-- Don's initial budget -/
def initial_budget : ℕ := 600

/-- Number of bottles Don buys from Shop A -/
def bottles_from_A : ℕ := 150

/-- Number of bottles Don buys from Shop B -/
def bottles_from_B : ℕ := 180

/-- The remaining budget after buying from shops A and B -/
def remaining_budget : ℕ := 
  initial_budget - (bottles_from_A * price Shop.A + bottles_from_B * price Shop.B)

/-- The theorem stating the maximum number of bottles Don can buy from shops C and D combined -/
theorem max_bottles_from_C_and_D : 
  (remaining_budget / price Shop.C) = 30 := by sorry

end max_bottles_from_C_and_D_l3341_334149


namespace probability_selection_l3341_334114

def research_team (total : ℝ) : Prop :=
  let women := 0.75 * total
  let men := 0.25 * total
  let women_lawyers := 0.60 * women
  let women_engineers := 0.25 * women
  let women_doctors := 0.15 * women
  let men_lawyers := 0.40 * men
  let men_engineers := 0.35 * men
  let men_doctors := 0.25 * men
  (women + men = total) ∧
  (women_lawyers + women_engineers + women_doctors = women) ∧
  (men_lawyers + men_engineers + men_doctors = men)

theorem probability_selection (total : ℝ) (h : research_team total) :
  (0.75 * 0.60 * total + 0.75 * 0.25 * total + 0.25 * 0.25 * total) / total = 0.70 :=
by sorry

end probability_selection_l3341_334114


namespace group_size_problem_l3341_334139

theorem group_size_problem (total_paise : ℕ) (h1 : total_paise = 4624) : ∃ n : ℕ, n * n = total_paise ∧ n = 68 := by
  sorry

end group_size_problem_l3341_334139


namespace partner_q_investment_time_l3341_334179

/-- The investment time of partner q given the investment and profit ratios -/
theorem partner_q_investment_time
  (investment_ratio : ℚ) -- Ratio of p's investment to q's investment
  (profit_ratio : ℚ) -- Ratio of p's profit to q's profit
  (p_time : ℕ) -- Time p invested in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 10)
  (h3 : p_time = 8) :
  ∃ q_time : ℕ, q_time = 16 ∧ 
  profit_ratio * investment_ratio * q_time = p_time :=
by sorry

end partner_q_investment_time_l3341_334179


namespace same_color_inevitable_l3341_334108

/-- A type representing the colors of the balls -/
inductive Color
| Red
| Yellow

/-- The total number of balls in the bag -/
def total_balls : Nat := 6

/-- The number of red balls in the bag -/
def red_balls : Nat := 3

/-- The number of yellow balls in the bag -/
def yellow_balls : Nat := 3

/-- The number of balls drawn from the bag -/
def drawn_balls : Nat := 3

/-- A function that determines if a drawing of balls inevitably results in at least two balls of the same color -/
def inevitable_same_color (total : Nat) (red : Nat) (yellow : Nat) (drawn : Nat) : Prop :=
  ∀ (selection : Finset Color), selection.card = drawn → selection.card ≥ 2

/-- Theorem stating that drawing 3 balls from the bag inevitably results in at least two balls of the same color -/
theorem same_color_inevitable :
  inevitable_same_color total_balls red_balls yellow_balls drawn_balls :=
by sorry

end same_color_inevitable_l3341_334108


namespace weightlifter_total_lift_l3341_334102

/-- The weight a weightlifter can lift in one hand -/
def weight_per_hand : ℕ := 7

/-- The number of hands a weightlifter has -/
def number_of_hands : ℕ := 2

/-- The total weight a weightlifter can lift at once -/
def total_weight : ℕ := weight_per_hand * number_of_hands

/-- Theorem: The total weight a weightlifter can lift at once is 14 pounds -/
theorem weightlifter_total_lift : total_weight = 14 := by
  sorry

end weightlifter_total_lift_l3341_334102
