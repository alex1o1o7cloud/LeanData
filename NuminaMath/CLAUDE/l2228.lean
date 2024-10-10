import Mathlib

namespace product_equation_sum_l2228_222870

theorem product_equation_sum (p q r s : ℤ) : 
  (∀ x, (x^2 + p*x + q) * (x^2 + r*x + s) = x^4 - x^3 + 3*x^2 - 4*x + 4) →
  p + q + r + s = -1 := by
sorry

end product_equation_sum_l2228_222870


namespace rational_cube_sum_ratio_l2228_222854

theorem rational_cube_sum_ratio (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end rational_cube_sum_ratio_l2228_222854


namespace inequality_proof_l2228_222810

def f (x : ℝ) := |2*x - 1| + |2*x + 1|

theorem inequality_proof (a b : ℝ) :
  (∀ x, -1 < x ∧ x < 1 → f x < 4) →
  -1 < a ∧ a < 1 →
  -1 < b ∧ b < 1 →
  |a + b| / |a*b + 1| < 1 :=
by sorry

end inequality_proof_l2228_222810


namespace towel_set_price_l2228_222883

/-- The price of towel sets for guest and master bathrooms -/
theorem towel_set_price (guest_sets master_sets : ℕ) (master_price : ℝ) 
  (discount : ℝ) (total_spent : ℝ) (h1 : guest_sets = 2) 
  (h2 : master_sets = 4) (h3 : master_price = 50) 
  (h4 : discount = 0.2) (h5 : total_spent = 224) : 
  ∃ (guest_price : ℝ), guest_price = 40 ∧ 
  (1 - discount) * (guest_sets * guest_price + master_sets * master_price) = total_spent :=
by
  sorry

#check towel_set_price

end towel_set_price_l2228_222883


namespace complex_number_quadrant_l2228_222882

theorem complex_number_quadrant (z : ℂ) (h : (1 - Complex.I) / z = 4 + 2 * Complex.I) : 
  Complex.re z > 0 ∧ Complex.im z < 0 := by sorry

end complex_number_quadrant_l2228_222882


namespace pieces_left_l2228_222833

/-- The number of medieval art pieces Alicia originally had -/
def original_pieces : ℕ := 70

/-- The number of medieval art pieces Alicia donated -/
def donated_pieces : ℕ := 46

/-- Theorem: The number of medieval art pieces Alicia has left is 24 -/
theorem pieces_left : original_pieces - donated_pieces = 24 := by
  sorry

end pieces_left_l2228_222833


namespace multiplication_mistake_l2228_222800

theorem multiplication_mistake (x : ℤ) : 
  (43 * x - 34 * x = 1206) → x = 134 := by
  sorry

end multiplication_mistake_l2228_222800


namespace equation_solution_l2228_222885

theorem equation_solution (a b : ℝ) (h : a ≠ -1) :
  let x := (a^2 - b^2 + 2*a - 2*b) / (2*(a+1))
  x^2 + (b+1)^2 = (a+1 - x)^2 := by
  sorry

end equation_solution_l2228_222885


namespace chip_notebook_packs_l2228_222889

/-- The number of packs of notebook paper Chip will use after 6 weeks -/
def notebook_packs_used (pages_per_class_per_day : ℕ) (num_classes : ℕ) 
  (days_per_week : ℕ) (sheets_per_pack : ℕ) (num_weeks : ℕ) : ℕ :=
  (pages_per_class_per_day * num_classes * days_per_week * num_weeks) / sheets_per_pack

/-- Theorem stating the number of packs Chip will use -/
theorem chip_notebook_packs : 
  notebook_packs_used 2 5 5 100 6 = 3 := by sorry

end chip_notebook_packs_l2228_222889


namespace sucrose_solution_volume_l2228_222897

/-- Given a sucrose solution where 60 cubic centimeters contain 6 grams of sucrose,
    prove that 100 cubic centimeters contain 10 grams of sucrose. -/
theorem sucrose_solution_volume (solution_volume : ℝ) (sucrose_mass : ℝ) :
  (60 : ℝ) / solution_volume = 6 / sucrose_mass →
  (100 : ℝ) / solution_volume = 10 / sucrose_mass :=
by
  sorry

end sucrose_solution_volume_l2228_222897


namespace total_books_count_l2228_222832

/-- The total number of Iesha's books -/
def total_books : ℕ := sorry

/-- The number of Iesha's school books -/
def school_books : ℕ := 19

/-- The number of Iesha's sports books -/
def sports_books : ℕ := 39

/-- Theorem: The total number of Iesha's books is 58 -/
theorem total_books_count : total_books = school_books + sports_books ∧ total_books = 58 := by
  sorry

end total_books_count_l2228_222832


namespace tens_digit_of_2013_squared_minus_2013_l2228_222840

theorem tens_digit_of_2013_squared_minus_2013 : (2013^2 - 2013) % 100 = 56 := by
  sorry

end tens_digit_of_2013_squared_minus_2013_l2228_222840


namespace student_arrangement_count_l2228_222893

/-- The number of ways to arrange 6 students with specific constraints -/
def arrangement_count : ℕ := 144

/-- Two specific students (A and B) must be adjacent -/
def adjacent_pair : ℕ := 2

/-- Number of students excluding A, B, and C -/
def other_students : ℕ := 3

/-- Number of valid positions for student C -/
def valid_positions_for_c : ℕ := 3

theorem student_arrangement_count :
  arrangement_count = 
    (Nat.factorial other_students) * 
    (Nat.factorial (other_students + 1) / Nat.factorial (other_students - 1)) * 
    adjacent_pair := by
  sorry

end student_arrangement_count_l2228_222893


namespace sqrt_sum_equals_seven_l2228_222820

theorem sqrt_sum_equals_seven (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) : 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end sqrt_sum_equals_seven_l2228_222820


namespace average_weight_problem_l2228_222895

theorem average_weight_problem (d e f : ℝ) 
  (h1 : (d + e) / 2 = 35)
  (h2 : (e + f) / 2 = 41)
  (h3 : e = 26) :
  (d + e + f) / 3 = 42 := by
  sorry

end average_weight_problem_l2228_222895


namespace cupcake_business_loan_payment_l2228_222880

/-- Calculates the monthly payment for a loan given the total loan amount, down payment, and loan term in years. -/
def calculate_monthly_payment (total_loan : ℕ) (down_payment : ℕ) (years : ℕ) : ℕ :=
  let amount_to_finance := total_loan - down_payment
  let months := years * 12
  amount_to_finance / months

/-- Proves that for a loan of $46,000 with a $10,000 down payment to be paid over 5 years, the monthly payment is $600. -/
theorem cupcake_business_loan_payment :
  calculate_monthly_payment 46000 10000 5 = 600 := by
  sorry

end cupcake_business_loan_payment_l2228_222880


namespace pizza_problem_l2228_222852

theorem pizza_problem : ∃! (m d : ℕ), 
  m > 0 ∧ d > 0 ∧ 
  7 * m + 2 * d > 36 ∧
  8 * m + 4 * d < 48 := by
  sorry

end pizza_problem_l2228_222852


namespace square_divides_power_plus_one_l2228_222877

theorem square_divides_power_plus_one (n : ℕ) : n^2 ∣ 2^n + 1 ↔ n = 1 := by
  sorry

end square_divides_power_plus_one_l2228_222877


namespace range_of_power_function_l2228_222830

/-- The range of g(x) = x^m for m > 0 on the interval (0, 1) is (0, 1) -/
theorem range_of_power_function (m : ℝ) (hm : m > 0) :
  Set.range (fun x : ℝ => x ^ m) ∩ Set.Ioo 0 1 = Set.Ioo 0 1 := by
  sorry

end range_of_power_function_l2228_222830


namespace optimal_price_for_max_revenue_l2228_222898

/-- Revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The theorem stating the optimal price for maximum revenue --/
theorem optimal_price_for_max_revenue :
  ∃ (p : ℝ), 0 < p ∧ p ≤ 37.5 ∧
  ∀ (q : ℝ), 0 < q → q ≤ 37.5 → R p ≥ R q ∧
  p = 18.75 := by
  sorry

end optimal_price_for_max_revenue_l2228_222898


namespace f_monotonicity_and_tangent_intersection_l2228_222887

/-- The function f(x) = x³ - x² + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem f_monotonicity_and_tangent_intersection (a : ℝ) :
  (∀ x : ℝ, f' a x ≥ 0 → a ≥ 1/3) ∧
  (∃ t : ℝ, t * f' a 1 = f a 1 ∧ f a (-1) = -t * f' a (-1)) :=
sorry


end f_monotonicity_and_tangent_intersection_l2228_222887


namespace min_value_product_quotient_l2228_222886

theorem min_value_product_quotient (x y z k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k ≥ 2) :
  (x^2 + k*x + 1) * (y^2 + k*y + 1) * (z^2 + k*z + 1) / (x*y*z) ≥ (2+k)^3 := by
  sorry

end min_value_product_quotient_l2228_222886


namespace traffic_light_is_random_l2228_222805

-- Define the concept of a random event
def is_random_event (event : String) : Prop := sorry

-- Define the phenomena
def water_boiling : String := "Under standard atmospheric pressure, water will boil when heated to 100°C"
def traffic_light : String := "Encountering a red light when walking to a crossroads"
def rectangle_area : String := "The area of a rectangle with length and width a and b respectively is a × b"
def linear_equation : String := "A linear equation with real coefficients must have a real root"

-- Theorem to prove
theorem traffic_light_is_random : is_random_event traffic_light :=
by sorry

end traffic_light_is_random_l2228_222805


namespace milk_savings_l2228_222888

-- Define the problem parameters
def gallons : ℕ := 8
def original_price : ℚ := 3.20
def discount_rate : ℚ := 0.25

-- Define the function to calculate savings
def calculate_savings (g : ℕ) (p : ℚ) (d : ℚ) : ℚ :=
  g * p * d

-- Theorem statement
theorem milk_savings :
  calculate_savings gallons original_price discount_rate = 6.40 := by
  sorry


end milk_savings_l2228_222888


namespace arccos_negative_half_l2228_222823

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by
  sorry

end arccos_negative_half_l2228_222823


namespace seating_probability_l2228_222826

/-- The number of people seated at the round table -/
def total_people : ℕ := 12

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 4

/-- The number of biology majors -/
def biology_majors : ℕ := 3

/-- The probability of the desired seating arrangement -/
def desired_probability : ℚ := 18/175

theorem seating_probability :
  let total_arrangements := (total_people - 1).factorial
  let math_block_arrangements := total_people * (math_majors - 1).factorial
  let physics_arrangements := physics_majors.factorial
  let biology_arrangements := (physics_majors + 1).choose biology_majors * biology_majors.factorial
  let favorable_arrangements := math_block_arrangements * physics_arrangements * biology_arrangements
  (favorable_arrangements : ℚ) / total_arrangements = desired_probability := by
  sorry

end seating_probability_l2228_222826


namespace degree_to_radian_conversion_l2228_222815

theorem degree_to_radian_conversion (π : ℝ) :
  (1 : ℝ) * π / 180 = π / 180 →
  (-150 : ℝ) * π / 180 = -5 * π / 6 :=
by sorry

end degree_to_radian_conversion_l2228_222815


namespace equation_solutions_l2228_222878

theorem equation_solutions :
  (∃ x : ℝ, 4.8 - 3 * x = 1.8 ∧ x = 1) ∧
  (∃ x : ℝ, (1/8) / (1/5) = x / 24 ∧ x = 15) ∧
  (∃ x : ℝ, 7.5 * x + 6.5 * x = 2.8 ∧ x = 0.2) := by
  sorry

end equation_solutions_l2228_222878


namespace spade_nested_operation_l2228_222842

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_nested_operation : spade 5 (spade 3 9) = 1 := by
  sorry

end spade_nested_operation_l2228_222842


namespace polynomial_property_l2228_222814

/-- Given a polynomial P(x) = ax^2 + bx + c where a, b, c are real numbers,
    if P(a) = bc, P(b) = ac, and P(c) = ab, then (a - b)(b - c)(c - a)(a + b + c) = 0 -/
theorem polynomial_property (a b c : ℝ) (P : ℝ → ℝ)
  (h_poly : ∀ x, P x = a * x^2 + b * x + c)
  (h_Pa : P a = b * c)
  (h_Pb : P b = a * c)
  (h_Pc : P c = a * b) :
  (a - b) * (b - c) * (c - a) * (a + b + c) = 0 := by
  sorry

end polynomial_property_l2228_222814


namespace rationalize_denominator_l2228_222804

theorem rationalize_denominator : 
  (7 : ℝ) / (Real.sqrt 175 - Real.sqrt 75) = 7 * (Real.sqrt 7 + Real.sqrt 3) / 20 := by
  sorry

end rationalize_denominator_l2228_222804


namespace gabriel_jaxon_toy_ratio_l2228_222876

theorem gabriel_jaxon_toy_ratio :
  ∀ (g j x : ℕ),
  j = g + 8 →
  x = 15 →
  g + j + x = 83 →
  g = 2 * x :=
by
  sorry

end gabriel_jaxon_toy_ratio_l2228_222876


namespace cubic_equation_roots_l2228_222809

theorem cubic_equation_roots : ∃! (p n₁ n₂ : ℝ), 
  p > 0 ∧ n₁ < 0 ∧ n₂ < 0 ∧
  p^3 + 3*p^2 - 4*p + 12 = 0 ∧
  n₁^3 + 3*n₁^2 - 4*n₁ + 12 = 0 ∧
  n₂^3 + 3*n₂^2 - 4*n₂ + 12 = 0 ∧
  p ≠ n₁ ∧ p ≠ n₂ ∧ n₁ ≠ n₂ :=
by sorry

end cubic_equation_roots_l2228_222809


namespace box_long_side_length_l2228_222892

/-- The length of the long sides of a box, given its dimensions and total velvet needed. -/
theorem box_long_side_length (total_velvet : ℝ) (short_side_length short_side_width : ℝ) 
  (long_side_width : ℝ) (top_bottom_area : ℝ) :
  total_velvet = 236 ∧
  short_side_length = 5 ∧
  short_side_width = 6 ∧
  long_side_width = 6 ∧
  top_bottom_area = 40 →
  ∃ long_side_length : ℝ,
    long_side_length = 8 ∧
    total_velvet = 2 * (short_side_length * short_side_width) + 
                   2 * top_bottom_area + 
                   2 * (long_side_length * long_side_width) :=
by sorry

end box_long_side_length_l2228_222892


namespace value_of_expression_l2228_222822

theorem value_of_expression : (-0.125)^2009 * (-8)^2010 = -8 := by
  sorry

end value_of_expression_l2228_222822


namespace lunch_calories_calculation_l2228_222839

def daily_calorie_allowance : ℕ := 2200
def breakfast_calories : ℕ := 353
def snack_calories : ℕ := 130
def dinner_calories_left : ℕ := 832

theorem lunch_calories_calculation : 
  daily_calorie_allowance - breakfast_calories - snack_calories - dinner_calories_left = 885 := by
  sorry

end lunch_calories_calculation_l2228_222839


namespace half_coverage_days_l2228_222874

/-- Represents the number of days it takes for the lily pad patch to cover the entire lake -/
def full_coverage_days : ℕ := 48

/-- Represents the growth factor of the lily pad patch per day -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that the number of days required to cover half the lake
    is one day less than the number of days required to cover the full lake -/
theorem half_coverage_days : 
  full_coverage_days - 1 = full_coverage_days - (daily_growth_factor.log 2) := by
  sorry

end half_coverage_days_l2228_222874


namespace factorization_equality_l2228_222855

theorem factorization_equality (a b : ℝ) :
  (a - b)^4 + (a + b)^4 + (a + b)^2 * (a - b)^2 = (3*a^2 + b^2) * (a^2 + 3*b^2) := by
  sorry

end factorization_equality_l2228_222855


namespace three_digit_multiple_of_2_3_5_l2228_222865

theorem three_digit_multiple_of_2_3_5 (n : ℕ) :
  100 ≤ n ∧ n ≤ 999 ∧ 
  2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n →
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m → 120 ≤ m) ∧
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m → m ≤ 990) :=
by sorry

end three_digit_multiple_of_2_3_5_l2228_222865


namespace petyas_journey_contradiction_l2228_222831

theorem petyas_journey_contradiction (S T : ℝ) (hS : S > 0) (hT : T > 0) : 
  ¬(∃ (S T : ℝ), 
    S / 2 = 4 * (T / 2) ∧ 
    S / 2 = 5 * (T / 2)) :=
by
  sorry

end petyas_journey_contradiction_l2228_222831


namespace total_dinners_sold_l2228_222884

def monday_sales : ℕ := 40

def tuesday_sales : ℕ := monday_sales + 40

def wednesday_sales : ℕ := tuesday_sales / 2

def thursday_sales : ℕ := wednesday_sales + 3

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales

theorem total_dinners_sold : total_sales = 203 := by
  sorry

end total_dinners_sold_l2228_222884


namespace van_helsing_werewolf_removal_percentage_l2228_222806

def vampire_price : ℕ := 5
def werewolf_price : ℕ := 10
def total_earnings : ℕ := 105
def werewolves_removed : ℕ := 8
def werewolf_vampire_ratio : ℕ := 4

theorem van_helsing_werewolf_removal_percentage 
  (vampires : ℕ) (werewolves : ℕ) : 
  vampire_price * (vampires / 2) + werewolf_price * werewolves_removed = total_earnings →
  werewolves = werewolf_vampire_ratio * vampires →
  (werewolves_removed : ℚ) / werewolves * 100 = 20 := by
  sorry

end van_helsing_werewolf_removal_percentage_l2228_222806


namespace functional_equation_problem_l2228_222849

/-- The functional equation problem -/
theorem functional_equation_problem (α : ℝ) (hα : α ≠ 0) :
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, f (f (x + y)) = f (x + y) + f x * f y + α * x * y) ↔
  (α = -1 ∧ ∃! f : ℝ → ℝ, ∀ x : ℝ, f x = x) :=
by sorry

end functional_equation_problem_l2228_222849


namespace total_savings_after_tax_l2228_222803

def total_income : ℝ := 18000

def income_ratio_a : ℝ := 3
def income_ratio_b : ℝ := 2
def income_ratio_c : ℝ := 1

def tax_rate_a : ℝ := 0.1
def tax_rate_b : ℝ := 0.15
def tax_rate_c : ℝ := 0

def expenditure_ratio : ℝ := 5
def income_ratio : ℝ := 9

theorem total_savings_after_tax :
  let income_a := (income_ratio_a / (income_ratio_a + income_ratio_b + income_ratio_c)) * total_income
  let income_b := (income_ratio_b / (income_ratio_a + income_ratio_b + income_ratio_c)) * total_income
  let income_c := (income_ratio_c / (income_ratio_a + income_ratio_b + income_ratio_c)) * total_income
  let tax_a := tax_rate_a * income_a
  let tax_b := tax_rate_b * income_b
  let tax_c := tax_rate_c * income_c
  let total_tax := tax_a + tax_b + tax_c
  let income_after_tax := total_income - total_tax
  let expenditure := (expenditure_ratio / income_ratio) * total_income
  let savings := income_after_tax - expenditure
  savings = 6200 := by sorry

end total_savings_after_tax_l2228_222803


namespace min_participants_is_eleven_l2228_222879

/-- Represents the number of participants in each grade --/
structure Participants where
  fifth : Nat
  sixth : Nat
  seventh : Nat

/-- Checks if the given number of participants satisfies all conditions --/
def satisfiesConditions (n : Nat) (p : Participants) : Prop :=
  p.fifth + p.sixth + p.seventh = n ∧
  (25 * n < 100 * p.fifth) ∧ (100 * p.fifth < 35 * n) ∧
  (30 * n < 100 * p.sixth) ∧ (100 * p.sixth < 40 * n) ∧
  (35 * n < 100 * p.seventh) ∧ (100 * p.seventh < 45 * n)

/-- States that 11 is the minimum number of participants satisfying all conditions --/
theorem min_participants_is_eleven :
  ∃ (p : Participants), satisfiesConditions 11 p ∧
  ∀ (m : Nat) (q : Participants), m < 11 → ¬satisfiesConditions m q :=
by sorry

end min_participants_is_eleven_l2228_222879


namespace simple_interest_problem_l2228_222807

/-- Given a sum of money P put at simple interest for 7 years at rate R%,
    if increasing the rate by 2% results in 140 more interest, then P = 1000. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 2) * 7) / 100 = (P * R * 7) / 100 + 140 → P = 1000 :=
by sorry

end simple_interest_problem_l2228_222807


namespace range_of_a_l2228_222828

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - a| < 1}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 2}

-- Define the complement of B
def complementB : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  A a ⊆ complementB → 3 ≤ a ∧ a ≤ 4 :=
by sorry

end range_of_a_l2228_222828


namespace not_all_products_of_two_primes_l2228_222837

theorem not_all_products_of_two_primes (q : ℕ) (hq : Nat.Prime q) (hodd : Odd q) :
  ∃ k : ℕ, k ∈ Finset.range (q - 1) ∧ ¬∃ p₁ p₂ : ℕ, Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ k^2 + k + q = p₁ * p₂ := by
  sorry

end not_all_products_of_two_primes_l2228_222837


namespace perpendicular_vectors_x_value_l2228_222875

/-- Two vectors a and b in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, perpendicular (x, 2) (1, -1) → x = 2 := by
  sorry

end perpendicular_vectors_x_value_l2228_222875


namespace unique_solution_for_xy_equation_l2228_222843

theorem unique_solution_for_xy_equation :
  ∀ x y : ℤ, x > y ∧ y > 0 ∧ x + y + x * y = 99 → x = 49 := by
  sorry

end unique_solution_for_xy_equation_l2228_222843


namespace ellipse_properties_l2228_222824

def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * a = 4 * Real.sqrt 3) 
  (h4 : (2^2 / a^2) + ((Real.sqrt 2)^2 / b^2) = 1) 
  (h5 : ∃ (A B : ℝ × ℝ), A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧ 
    (A.1 + B.1) / 2 = -8/5 ∧ (A.2 + B.2) / 2 = 2/5) :
  (a^2 = 12 ∧ b^2 = 3) ∧ 
  (∀ (A B : ℝ × ℝ), A ∈ Ellipse a b → B ∈ Ellipse a b → 
    (A.1 + B.1) / 2 = -8/5 → (A.2 + B.2) / 2 = 2/5 → 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 22 / 5) := by
  sorry

end ellipse_properties_l2228_222824


namespace intersection_of_A_and_B_l2228_222851

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end intersection_of_A_and_B_l2228_222851


namespace correct_average_mark_l2228_222834

theorem correct_average_mark (n : ℕ) (initial_avg : ℚ) (wrong_mark correct_mark : ℚ) :
  n = 30 →
  initial_avg = 100 →
  wrong_mark = 70 →
  correct_mark = 10 →
  (n : ℚ) * initial_avg - wrong_mark + correct_mark = 98 * n :=
by sorry

end correct_average_mark_l2228_222834


namespace age_difference_ratio_l2228_222827

/-- Represents the ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.roy = ages.julia + 8 ∧
  ages.roy + 4 = 2 * (ages.julia + 4) ∧
  (ages.roy + 4) * (ages.kelly + 4) = 192

/-- The theorem to prove -/
theorem age_difference_ratio (ages : Ages) :
  satisfiesConditions ages →
  (ages.roy - ages.julia) / (ages.roy - ages.kelly) = 2 := by
  sorry

end age_difference_ratio_l2228_222827


namespace ball_attendees_l2228_222836

theorem ball_attendees :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end ball_attendees_l2228_222836


namespace largest_five_digit_with_product_factorial_l2228_222850

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_product (n : Nat) : Nat :=
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

def is_five_digit (n : Nat) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_with_product_factorial :
  ∃ (n : Nat), is_five_digit n ∧
               digit_product n = factorial 8 ∧
               ∀ (m : Nat), is_five_digit m ∧ digit_product m = factorial 8 → m ≤ n :=
by
  use 98752
  sorry

end largest_five_digit_with_product_factorial_l2228_222850


namespace passing_grade_fraction_l2228_222825

theorem passing_grade_fraction (a b c d f : ℚ) : 
  a = 1/4 → b = 1/2 → c = 1/8 → d = 1/12 → f = 1/24 → a + b + c = 7/8 := by
  sorry

end passing_grade_fraction_l2228_222825


namespace project_hours_difference_l2228_222811

theorem project_hours_difference (total_hours : ℝ) 
  (h_total : total_hours = 350) 
  (h_pat_kate : ∃ k : ℝ, pat = 2 * k ∧ kate = k)
  (h_pat_mark : ∃ m : ℝ, pat = (1/3) * m ∧ mark = m)
  (h_alex_kate : ∃ k : ℝ, alex = 1.5 * k ∧ kate = k)
  (h_sum : pat + kate + mark + alex = total_hours) :
  mark - (kate + alex) = 350/3 :=
sorry

end project_hours_difference_l2228_222811


namespace inverse_variation_solution_l2228_222802

/-- The constant k in the inverse variation relationship -/
def k (a b : ℝ) : ℝ := a^3 * b^2

/-- The inverse variation relationship between a and b -/
def inverse_variation (a b : ℝ) : Prop := k a b = k 5 2

theorem inverse_variation_solution :
  ∀ a b : ℝ,
  inverse_variation a b →
  (a = 5 ∧ b = 2) ∨ (a = 2.5 ∧ b = 8) :=
sorry

end inverse_variation_solution_l2228_222802


namespace semicircle_problem_l2228_222873

theorem semicircle_problem (M : ℕ) (r : ℝ) (h_positive : r > 0) : 
  (M * π * r^2 / 2) / (π * r^2 * (M^2 - M) / 2) = 1/4 → M = 5 := by
  sorry

end semicircle_problem_l2228_222873


namespace books_before_adding_l2228_222801

theorem books_before_adding (total_after : ℕ) (added : ℕ) (h1 : total_after = 19) (h2 : added = 10) :
  total_after - added = 9 := by
  sorry

end books_before_adding_l2228_222801


namespace system_solution_l2228_222859

theorem system_solution :
  ∀ x y z : ℝ,
  (x + y - 2 + 4*x*y = 0 ∧
   y + z - 2 + 4*y*z = 0 ∧
   z + x - 2 + 4*z*x = 0) ↔
  ((x = -1 ∧ y = -1 ∧ z = -1) ∨
   (x = 1/2 ∧ y = 1/2 ∧ z = 1/2)) :=
by sorry

end system_solution_l2228_222859


namespace inequality_theorem_range_theorem_l2228_222857

-- Theorem 1
theorem inequality_theorem (x y : ℝ) : x^2 + 2*y^2 ≥ 2*x*y + 2*y - 1 := by
  sorry

-- Theorem 2
theorem range_theorem (a b : ℝ) (h1 : -2 < a ∧ a ≤ 3) (h2 : 1 ≤ b ∧ b < 2) :
  -1 < a + b ∧ a + b < 5 ∧ -10 < 2*a - 3*b ∧ 2*a - 3*b ≤ 3 := by
  sorry

end inequality_theorem_range_theorem_l2228_222857


namespace data_median_and_variance_l2228_222845

def data : List ℝ := [2, 3, 3, 3, 6, 6, 4, 5]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_median_and_variance :
  median data = 3.5 ∧ variance data = 2 := by sorry

end data_median_and_variance_l2228_222845


namespace equal_division_of_cakes_l2228_222853

theorem equal_division_of_cakes (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) :
  total_cakes = 18 →
  num_children = 3 →
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
  sorry

end equal_division_of_cakes_l2228_222853


namespace sloth_shoe_pairs_needed_l2228_222899

/-- Represents the number of feet a sloth has -/
def sloth_feet : ℕ := 3

/-- Represents the number of shoes in a complete set for the sloth -/
def shoes_per_set : ℕ := 3

/-- Represents the number of shoes in a pair -/
def shoes_per_pair : ℕ := 2

/-- Represents the number of complete sets the sloth already owns -/
def owned_sets : ℕ := 1

/-- Represents the total number of complete sets the sloth needs -/
def total_sets_needed : ℕ := 5

/-- Theorem stating the number of pairs of shoes the sloth needs to buy -/
theorem sloth_shoe_pairs_needed :
  (total_sets_needed - owned_sets) * shoes_per_set / shoes_per_pair = 6 := by
  sorry

end sloth_shoe_pairs_needed_l2228_222899


namespace geometric_series_common_ratio_l2228_222864

theorem geometric_series_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_sum : a 2 + a 4 = 3) 
  (h_product : a 3 * a 5 = 2) : 
  q = Real.sqrt ((3 * Real.sqrt 2 + 2) / 7) :=
sorry

end geometric_series_common_ratio_l2228_222864


namespace equation_has_29_solutions_l2228_222861

/-- The number of real solutions to the equation x/50 = sin x -/
def num_solutions : ℕ := 29

/-- The equation we're considering -/
def equation (x : ℝ) : Prop := x / 50 = Real.sin x

theorem equation_has_29_solutions :
  ∃! (s : Set ℝ), (∀ x ∈ s, equation x) ∧ Finite s ∧ Nat.card s = num_solutions :=
sorry

end equation_has_29_solutions_l2228_222861


namespace inverse_sum_equals_six_l2228_222818

-- Define the function f
def f (x : ℝ) : ℝ := x * |x|^2

-- State the theorem
theorem inverse_sum_equals_six :
  ∃ (a b : ℝ), f a = 8 ∧ f b = -64 ∧ a + b = 6 := by sorry

end inverse_sum_equals_six_l2228_222818


namespace intersection_sum_l2228_222872

theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, x = (1/3) * y + c ↔ y = (1/3) * x + d) → 
  (3 = (1/3) * 6 + c ∧ 6 = (1/3) * 3 + d) → 
  c + d = 6 := by
sorry

end intersection_sum_l2228_222872


namespace equation_solution_l2228_222819

theorem equation_solution : ∃! x : ℚ, x + 5/8 = 1/4 - 2/5 + 7/10 ∧ x = -3/40 := by
  sorry

end equation_solution_l2228_222819


namespace associates_hired_l2228_222813

theorem associates_hired (initial_ratio_partners : ℕ) (initial_ratio_associates : ℕ)
  (current_partners : ℕ) (new_ratio_partners : ℕ) (new_ratio_associates : ℕ) :
  initial_ratio_partners = 2 →
  initial_ratio_associates = 63 →
  current_partners = 14 →
  new_ratio_partners = 1 →
  new_ratio_associates = 34 →
  ∃ (current_associates : ℕ) (hired_associates : ℕ),
    current_associates * initial_ratio_partners = current_partners * initial_ratio_associates ∧
    (current_associates + hired_associates) * new_ratio_partners = current_partners * new_ratio_associates ∧
    hired_associates = 35 :=
by
  sorry

end associates_hired_l2228_222813


namespace arithmetic_sequence_problem_l2228_222808

theorem arithmetic_sequence_problem (a : ℕ → ℕ) (n : ℕ) :
  (∀ k, a (k + 1) = a k + 5) →  -- arithmetic sequence with common difference 5
  a 1 = 1 →                    -- first term is 1
  a n = 2016 →                 -- n-th term is 2016
  n = 404 :=                   -- prove n is 404
by sorry

end arithmetic_sequence_problem_l2228_222808


namespace sine_sum_gt_cosine_sum_in_acute_triangle_l2228_222846

/-- In any acute-angled triangle ABC, the sum of the sines of its angles is greater than the sum of the cosines of its angles. -/
theorem sine_sum_gt_cosine_sum_in_acute_triangle (A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_triangle : A + B + C = π) : 
  Real.sin A + Real.sin B + Real.sin C > Real.cos A + Real.cos B + Real.cos C := by
  sorry

end sine_sum_gt_cosine_sum_in_acute_triangle_l2228_222846


namespace zero_product_implies_zero_factor_l2228_222844

theorem zero_product_implies_zero_factor (x y : ℝ) : 
  x * y = 0 → x = 0 ∨ y = 0 := by
  sorry

end zero_product_implies_zero_factor_l2228_222844


namespace total_shells_count_l2228_222847

def purple_shells : ℕ := 13
def pink_shells : ℕ := 8
def yellow_shells : ℕ := 18
def blue_shells : ℕ := 12
def orange_shells : ℕ := 14

theorem total_shells_count :
  purple_shells + pink_shells + yellow_shells + blue_shells + orange_shells = 65 := by
  sorry

end total_shells_count_l2228_222847


namespace peaches_picked_l2228_222863

def initial_peaches : ℕ := 34
def current_peaches : ℕ := 86

theorem peaches_picked (initial : ℕ) (current : ℕ) :
  current ≥ initial → current - initial = current - initial :=
by sorry

end peaches_picked_l2228_222863


namespace largest_b_value_l2228_222860

/-- The polynomial function representing the equation -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^4 - a*x^3 - b*x^2 - c*x - 2007

/-- Predicate to check if a number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Predicate to check if the equation has exactly three distinct integer solutions -/
def hasThreeDistinctIntegerSolutions (a b c : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    isInteger x ∧ isInteger y ∧ isInteger z ∧
    f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0 ∧
    ∀ w : ℝ, f a b c w = 0 → w = x ∨ w = y ∨ w = z

/-- The main theorem -/
theorem largest_b_value :
  ∃ b_max : ℝ, (∀ a c b : ℝ, hasThreeDistinctIntegerSolutions a b c → b ≤ b_max) ∧
    (∃ a c : ℝ, hasThreeDistinctIntegerSolutions a b_max c) ∧
    b_max = 3343 := by sorry

end largest_b_value_l2228_222860


namespace fish_eaten_l2228_222868

theorem fish_eaten (initial_fish : ℕ) (temp_added : ℕ) (exchanged : ℕ) (final_fish : ℕ)
  (h1 : initial_fish = 14)
  (h2 : temp_added = 2)
  (h3 : exchanged = 3)
  (h4 : final_fish = 11) :
  initial_fish - (final_fish - exchanged) = 6 :=
by sorry

end fish_eaten_l2228_222868


namespace he_has_21_apples_l2228_222821

/-- The number of apples Adam and Jackie have together -/
def total_adam_jackie : ℕ := 12

/-- The number of additional apples He has compared to Adam and Jackie together -/
def additional_apples : ℕ := 9

/-- The number of additional apples Adam has compared to Jackie -/
def adam_more_than_jackie : ℕ := 8

/-- The number of apples He has -/
def he_apples : ℕ := total_adam_jackie + additional_apples

theorem he_has_21_apples : he_apples = 21 := by
  sorry

end he_has_21_apples_l2228_222821


namespace line_in_three_quadrants_coeff_products_l2228_222856

/-- A line passing through the first, second, and third quadrants -/
structure LineInThreeQuadrants where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_first_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + b * y + c = 0
  passes_second_quadrant : ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ a * x + b * y + c = 0
  passes_third_quadrant : ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0

/-- Theorem: If a line passes through the first, second, and third quadrants, 
    then the product of its coefficients satisfies ab < 0 and bc < 0 -/
theorem line_in_three_quadrants_coeff_products (line : LineInThreeQuadrants) :
  line.a * line.b < 0 ∧ line.b * line.c < 0 := by
  sorry

end line_in_three_quadrants_coeff_products_l2228_222856


namespace beacon_school_earnings_l2228_222848

/-- Represents a school's participation in the community project -/
structure School where
  name : String
  students : ℕ
  weekdays : ℕ
  weekendDays : ℕ

/-- Calculates the total earnings for a school given the daily rates -/
def schoolEarnings (s : School) (weekdayRate weekendRate : ℚ) : ℚ :=
  s.students * (s.weekdays * weekdayRate + s.weekendDays * weekendRate)

/-- The main theorem stating that Beacon school's earnings are $336.00 -/
theorem beacon_school_earnings :
  let apex : School := ⟨"Apex", 9, 4, 2⟩
  let beacon : School := ⟨"Beacon", 6, 6, 1⟩
  let citadel : School := ⟨"Citadel", 7, 8, 3⟩
  let schools : List School := [apex, beacon, citadel]
  let totalPaid : ℚ := 1470
  ∃ (weekdayRate : ℚ),
    weekdayRate > 0 ∧
    (schools.map (fun s => schoolEarnings s weekdayRate (2 * weekdayRate))).sum = totalPaid ∧
    schoolEarnings beacon weekdayRate (2 * weekdayRate) = 336 := by
  sorry

end beacon_school_earnings_l2228_222848


namespace milk_water_ratio_problem_l2228_222835

/-- Given two vessels with volumes in ratio 3:5, where the first vessel has a milk to water ratio
    of 1:2, and when mixed the overall milk to water ratio is 1:1, prove that the milk to water
    ratio in the second vessel must be 3:2. -/
theorem milk_water_ratio_problem (v : ℝ) (x y : ℝ) (h_x_pos : x > 0) (h_y_pos : y > 0) : 
  (1 : ℝ) + (5 * x) / (x + y) = (2 : ℝ) + (5 * y) / (x + y) → x / y = 3 / 2 :=
by sorry

end milk_water_ratio_problem_l2228_222835


namespace sum_of_roots_l2228_222867

theorem sum_of_roots (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 17 = 0)
  (hβ : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := by sorry

end sum_of_roots_l2228_222867


namespace books_not_sold_percentage_l2228_222890

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem books_not_sold_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentage_not_sold - 80.57| < ε :=
sorry

end books_not_sold_percentage_l2228_222890


namespace min_value_cubic_expression_l2228_222838

theorem min_value_cubic_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^3 + y^3 - 5*x*y ≥ -125/27 := by sorry

end min_value_cubic_expression_l2228_222838


namespace total_gumballs_l2228_222862

/-- Represents the number of gumballs of each color in a gumball machine. -/
structure GumballMachine where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Defines the properties of the gumball machine as described in the problem. -/
def validGumballMachine (m : GumballMachine) : Prop :=
  m.blue = m.red / 2 ∧ m.green = 4 * m.blue ∧ m.red = 16

/-- Theorem stating that a valid gumball machine contains 56 gumballs in total. -/
theorem total_gumballs (m : GumballMachine) (h : validGumballMachine m) :
  m.red + m.blue + m.green = 56 := by
  sorry

#check total_gumballs

end total_gumballs_l2228_222862


namespace white_chips_percentage_l2228_222817

theorem white_chips_percentage
  (total : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (h1 : blue = 3)
  (h2 : blue = total / 10)
  (h3 : green = 12) :
  (total - blue - green) * 100 / total = 50 :=
sorry

end white_chips_percentage_l2228_222817


namespace polygon_arrangement_exists_l2228_222869

/-- A polygon constructed from squares and equilateral triangles -/
structure PolygonArrangement where
  squares : ℕ
  triangles : ℕ
  side_length : ℝ
  perimeter : ℝ

/-- The existence of a polygon arrangement with the given properties -/
theorem polygon_arrangement_exists : ∃ (p : PolygonArrangement), 
  p.squares = 9 ∧ 
  p.triangles = 19 ∧ 
  p.side_length = 1 ∧ 
  p.perimeter = 15 := by
  sorry

end polygon_arrangement_exists_l2228_222869


namespace triangle_properties_l2228_222866

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (abc : Triangle) (R : Real) :
  2 * Real.sqrt 3 * (Real.sin (abc.A / 2))^2 + Real.sin abc.A - Real.sqrt 3 = 0 →
  (1/2) * abc.b * abc.c * Real.sin abc.A = Real.sqrt 3 →
  R = Real.sqrt 3 →
  abc.A = π/3 ∧ abc.a + abc.b + abc.c = 3 + Real.sqrt 17 := by
  sorry


end triangle_properties_l2228_222866


namespace concentric_circles_radii_difference_l2228_222894

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : (R^2) / (r^2) = 4) :
  R - r = r :=
sorry

end concentric_circles_radii_difference_l2228_222894


namespace broomstick_charge_theorem_l2228_222829

/-- Represents the state of the broomstick at a given time -/
structure BroomState where
  minutes : Nat  -- Minutes since midnight
  charge : Nat   -- Current charge (0-100)

/-- Calculates the charge of the broomstick given the number of minutes since midnight -/
def calculateCharge (minutes : Nat) : Nat :=
  100 - minutes / 6

/-- Checks if the given time (in minutes since midnight) is a solution -/
def isSolution (minutes : Nat) : Bool :=
  let charge := calculateCharge minutes
  let minutesPastHour := minutes % 60
  charge == minutesPastHour

/-- The set of solution times -/
def solutionTimes : List BroomState :=
  [
    { minutes := 292, charge := 52 },  -- 04:52
    { minutes := 343, charge := 43 },  -- 05:43
    { minutes := 395, charge := 35 },  -- 06:35
    { minutes := 446, charge := 26 },  -- 07:26
    { minutes := 549, charge := 9 }    -- 09:09
  ]

/-- Main theorem: The given solution times are correct and complete -/
theorem broomstick_charge_theorem :
  (∀ t ∈ solutionTimes, isSolution t.minutes) ∧
  (∀ m, 0 ≤ m ∧ m < 600 → isSolution m → (∃ t ∈ solutionTimes, t.minutes = m)) :=
sorry


end broomstick_charge_theorem_l2228_222829


namespace machine_net_worth_l2228_222896

/-- Calculate the total net worth of a machine after 2 years given depreciation and maintenance costs -/
theorem machine_net_worth 
  (initial_value : ℝ)
  (depreciation_rate : ℝ)
  (initial_maintenance_cost : ℝ)
  (maintenance_increase_rate : ℝ)
  (h1 : initial_value = 40000)
  (h2 : depreciation_rate = 0.05)
  (h3 : initial_maintenance_cost = 2000)
  (h4 : maintenance_increase_rate = 0.03) :
  let value_after_year_1 := initial_value * (1 - depreciation_rate)
  let value_after_year_2 := value_after_year_1 * (1 - depreciation_rate)
  let maintenance_cost_year_1 := initial_maintenance_cost
  let maintenance_cost_year_2 := initial_maintenance_cost * (1 + maintenance_increase_rate)
  let total_maintenance_cost := maintenance_cost_year_1 + maintenance_cost_year_2
  let net_worth := value_after_year_2 - total_maintenance_cost
  net_worth = 32040 := by
  sorry


end machine_net_worth_l2228_222896


namespace modulus_of_complex_reciprocal_l2228_222881

theorem modulus_of_complex_reciprocal (i : ℂ) (h : i^2 = -1) :
  Complex.abs (1 / (i - 1)) = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_complex_reciprocal_l2228_222881


namespace second_divisor_is_24_l2228_222816

theorem second_divisor_is_24 (m n : ℕ) (h1 : m % 288 = 47) (h2 : m % n = 23) (h3 : n > 23) : n = 24 := by
  sorry

end second_divisor_is_24_l2228_222816


namespace cos_double_angle_special_case_l2228_222891

theorem cos_double_angle_special_case (α : Real) 
  (h : Real.sin (Real.pi + α) = 2/3) : 
  Real.cos (2 * α) = 1/9 := by
  sorry

end cos_double_angle_special_case_l2228_222891


namespace seed_germination_percentage_l2228_222858

theorem seed_germination_percentage
  (seeds_plot1 : ℕ)
  (seeds_plot2 : ℕ)
  (germination_rate_plot1 : ℚ)
  (germination_rate_plot2 : ℚ)
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 15 / 100)
  (h4 : germination_rate_plot2 = 35 / 100)
  : (((seeds_plot1 * germination_rate_plot1 + seeds_plot2 * germination_rate_plot2) / (seeds_plot1 + seeds_plot2)) : ℚ) = 23 / 100 := by
  sorry

end seed_germination_percentage_l2228_222858


namespace infinite_series_sum_l2228_222871

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 7/6 -/
theorem infinite_series_sum : 
  ∑' n : ℕ+, (3 * n - 2 : ℚ) / (n * (n + 1) * (n + 3)) = 7/6 := by
  sorry

end infinite_series_sum_l2228_222871


namespace dog_food_cup_weight_l2228_222812

/-- The weight of a cup of dog food in pounds -/
def cup_weight : ℝ := 0.25

/-- The number of dogs -/
def num_dogs : ℕ := 2

/-- The number of cups of dog food consumed by each dog per day -/
def cups_per_dog_per_day : ℕ := 12

/-- The number of days in a month -/
def days_per_month : ℕ := 30

/-- The number of bags of dog food bought per month -/
def bags_per_month : ℕ := 9

/-- The weight of each bag of dog food in pounds -/
def bag_weight : ℝ := 20

/-- Theorem stating that the weight of a cup of dog food is 0.25 pounds -/
theorem dog_food_cup_weight :
  cup_weight = (bags_per_month * bag_weight) / (num_dogs * cups_per_dog_per_day * days_per_month) :=
by sorry

end dog_food_cup_weight_l2228_222812


namespace equal_values_at_fixed_distance_l2228_222841

theorem equal_values_at_fixed_distance (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 1, ContinuousAt f x) →
  f 0 = 0 →
  f 1 = 0 →
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ |x₁ - x₂| = 0.1 ∧ f x₁ = f x₂ := by
  sorry


end equal_values_at_fixed_distance_l2228_222841
