import Mathlib

namespace NUMINAMATH_CALUDE_no_real_solutions_l2200_220001

theorem no_real_solutions : ¬∃ x : ℝ, (2*x - 10*x + 24)^2 + 4 = -2*|x| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2200_220001


namespace NUMINAMATH_CALUDE_green_hats_count_l2200_220058

/-- Proves that the number of green hats is 20 given the conditions of the problem -/
theorem green_hats_count (total_hats : ℕ) (blue_price green_price total_price : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_price = 6)
  (h3 : green_price = 7)
  (h4 : total_price = 530) :
  ∃ (blue_hats green_hats : ℕ), 
    blue_hats + green_hats = total_hats ∧
    blue_price * blue_hats + green_price * green_hats = total_price ∧
    green_hats = 20 := by
  sorry

#check green_hats_count

end NUMINAMATH_CALUDE_green_hats_count_l2200_220058


namespace NUMINAMATH_CALUDE_fast_food_order_l2200_220014

/-- A problem about friends ordering fast food --/
theorem fast_food_order (num_friends : ℕ) (hamburger_cost : ℚ) 
  (fries_sets : ℕ) (fries_cost : ℚ) (soda_cups : ℕ) (soda_cost : ℚ)
  (spaghetti_platters : ℕ) (spaghetti_cost : ℚ) (individual_payment : ℚ) :
  num_friends = 5 →
  hamburger_cost = 3 →
  fries_sets = 4 →
  fries_cost = 6/5 →
  soda_cups = 5 →
  soda_cost = 1/2 →
  spaghetti_platters = 1 →
  spaghetti_cost = 27/10 →
  individual_payment = 5 →
  ∃ (num_hamburgers : ℕ), 
    num_hamburgers * hamburger_cost + 
    fries_sets * fries_cost + 
    soda_cups * soda_cost + 
    spaghetti_platters * spaghetti_cost = 
    num_friends * individual_payment ∧
    num_hamburgers = 5 := by
  sorry


end NUMINAMATH_CALUDE_fast_food_order_l2200_220014


namespace NUMINAMATH_CALUDE_spa_nail_polish_l2200_220016

/-- The number of girls who went to the spa -/
def num_girls : ℕ := 8

/-- The number of fingers on each girl's hands -/
def fingers_per_girl : ℕ := 10

/-- The number of toes on each girl's feet -/
def toes_per_girl : ℕ := 10

/-- The total number of digits polished at the spa -/
def total_digits_polished : ℕ := num_girls * (fingers_per_girl + toes_per_girl)

theorem spa_nail_polish :
  total_digits_polished = 160 := by sorry

end NUMINAMATH_CALUDE_spa_nail_polish_l2200_220016


namespace NUMINAMATH_CALUDE_string_average_length_l2200_220047

theorem string_average_length : 
  let strings : List ℚ := [2, 5, 7]
  (strings.sum / strings.length : ℚ) = 14/3 := by
sorry

end NUMINAMATH_CALUDE_string_average_length_l2200_220047


namespace NUMINAMATH_CALUDE_items_left_in_store_l2200_220094

theorem items_left_in_store (ordered : ℕ) (sold : ℕ) (in_storeroom : ℕ) 
  (h_ordered : ordered = 4458)
  (h_sold : sold = 1561)
  (h_storeroom : in_storeroom = 575)
  (h_damaged : ⌊(5 : ℝ) / 100 * ordered⌋ = 222) : 
  ordered - sold - ⌊(5 : ℝ) / 100 * ordered⌋ + in_storeroom = 3250 := by
  sorry

end NUMINAMATH_CALUDE_items_left_in_store_l2200_220094


namespace NUMINAMATH_CALUDE_sum_coefficients_when_binomial_sum_is_8_l2200_220093

/-- Given a natural number n, this function represents the sum of the binomial coefficients
    of the expansion of (x^2 - 2/x)^n when x = 1 -/
def sumBinomialCoefficients (n : ℕ) : ℤ := (-1 : ℤ) ^ n

/-- Given a natural number n, this function represents the sum of the coefficients
    of the expansion of (x^2 - 2/x)^n when x = 1 -/
def sumCoefficients (n : ℕ) : ℤ := ((-1 : ℤ) - 2) ^ n

theorem sum_coefficients_when_binomial_sum_is_8 :
  ∃ n : ℕ, sumBinomialCoefficients n = 8 ∧ sumCoefficients n = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_coefficients_when_binomial_sum_is_8_l2200_220093


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2200_220027

theorem quadratic_roots_relation (p q : ℝ) : 
  (∃ a b : ℝ, 
    (2 * a^2 - 6 * a + 1 = 0) ∧ 
    (2 * b^2 - 6 * b + 1 = 0) ∧
    ((3 * a - 1)^2 + p * (3 * a - 1) + q = 0) ∧
    ((3 * b - 1)^2 + p * (3 * b - 1) + q = 0)) →
  q = -0.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2200_220027


namespace NUMINAMATH_CALUDE_least_multiple_33_greater_500_l2200_220040

theorem least_multiple_33_greater_500 : ∃ (n : ℕ), n * 33 = 528 ∧ 
  528 > 500 ∧ 
  (∀ (m : ℕ), m * 33 > 500 → m * 33 ≥ 528) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_33_greater_500_l2200_220040


namespace NUMINAMATH_CALUDE_number_difference_l2200_220020

theorem number_difference (x y : ℝ) 
  (sum_eq : x + y = 15) 
  (diff_eq : x - y = 10) 
  (square_diff_eq : x^2 - y^2 = 150) : 
  x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2200_220020


namespace NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l2200_220052

theorem absolute_value_square_sum_zero (x y : ℝ) :
  |x + 2| + (y - 1)^2 = 0 → x = -2 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l2200_220052


namespace NUMINAMATH_CALUDE_order_of_f_values_l2200_220097

noncomputable def f (x : ℝ) : ℝ := 2 / (4^x) - x

noncomputable def a : ℝ := 0
noncomputable def b : ℝ := Real.log 2 / Real.log 0.4
noncomputable def c : ℝ := Real.log 3 / Real.log 4

theorem order_of_f_values :
  f a < f c ∧ f c < f b := by sorry

end NUMINAMATH_CALUDE_order_of_f_values_l2200_220097


namespace NUMINAMATH_CALUDE_sum_congruence_l2200_220033

theorem sum_congruence (a b c : ℕ) : 
  a < 11 → b < 11 → c < 11 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 11 = 1 → 
  (7 * c) % 11 = 4 → 
  (8 * b) % 11 = (5 + b) % 11 → 
  (a + b + c) % 11 = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_congruence_l2200_220033


namespace NUMINAMATH_CALUDE_exponent_rules_l2200_220070

theorem exponent_rules (a : ℝ) : 
  (a^2 * a^4 ≠ a^8) ∧ 
  ((-2*a^2)^3 ≠ -6*a^6) ∧ 
  (a^4 / a = a^3) ∧ 
  (2*a + 3*a ≠ 5*a^2) := by
sorry

end NUMINAMATH_CALUDE_exponent_rules_l2200_220070


namespace NUMINAMATH_CALUDE_even_sum_probability_l2200_220073

-- Define the properties of the wheels
def first_wheel_sections : ℕ := 5
def first_wheel_even_sections : ℕ := 2
def first_wheel_odd_sections : ℕ := 3

def second_wheel_sections : ℕ := 4
def second_wheel_even_sections : ℕ := 1
def second_wheel_odd_sections : ℕ := 2
def second_wheel_special_sections : ℕ := 1

-- Define the probability of getting an even sum
def prob_even_sum : ℚ := 1/2

-- Theorem statement
theorem even_sum_probability :
  let p_even_first : ℚ := first_wheel_even_sections / first_wheel_sections
  let p_odd_first : ℚ := first_wheel_odd_sections / first_wheel_sections
  let p_even_second : ℚ := second_wheel_even_sections / second_wheel_sections
  let p_odd_second : ℚ := second_wheel_odd_sections / second_wheel_sections
  let p_special_second : ℚ := second_wheel_special_sections / second_wheel_sections
  
  -- Probability of both numbers being even (including special section effect)
  let p_both_even : ℚ := p_even_first * p_even_second + p_even_first * p_special_second
  
  -- Probability of both numbers being odd
  let p_both_odd : ℚ := p_odd_first * p_odd_second
  
  -- Total probability of an even sum
  p_both_even + p_both_odd = prob_even_sum :=
by sorry

end NUMINAMATH_CALUDE_even_sum_probability_l2200_220073


namespace NUMINAMATH_CALUDE_prob_at_least_one_event_l2200_220071

theorem prob_at_least_one_event (P₁ P₂ : ℝ) 
  (h₁ : 0 ≤ P₁ ∧ P₁ ≤ 1) 
  (h₂ : 0 ≤ P₂ ∧ P₂ ≤ 1) :
  P₁ + P₂ - P₁ * P₂ = 1 - (1 - P₁) * (1 - P₂) :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_event_l2200_220071


namespace NUMINAMATH_CALUDE_impossibleDivision_l2200_220045

/-- Represents an employee with their salary -/
structure Employee :=
  (salary : ℝ)

/-- Represents a region with its employees -/
structure Region :=
  (employees : List Employee)

/-- The total salary of a region -/
def totalSalary (r : Region) : ℝ :=
  (r.employees.map Employee.salary).sum

/-- The condition that 10% of employees get 90% of total salary -/
def salaryDistributionCondition (employees : List Employee) : Prop :=
  ∃ (highPaidEmployees : List Employee),
    highPaidEmployees.length = (employees.length / 10) ∧
    (highPaidEmployees.map Employee.salary).sum ≥ 0.9 * ((employees.map Employee.salary).sum)

/-- The condition for a valid region division -/
def validRegionDivision (regions : List Region) : Prop :=
  ∀ r ∈ regions, ∀ subset : List Employee,
    subset.length = (r.employees.length / 10) →
    (subset.map Employee.salary).sum ≤ 0.11 * totalSalary r

/-- The main theorem -/
theorem impossibleDivision :
  ∃ (employees : List Employee),
    salaryDistributionCondition employees ∧
    ¬∃ (regions : List Region),
      (regions.map Region.employees).join = employees ∧
      validRegionDivision regions :=
sorry

end NUMINAMATH_CALUDE_impossibleDivision_l2200_220045


namespace NUMINAMATH_CALUDE_publishing_break_even_point_l2200_220024

/-- Represents the break-even point calculation for a publishing company --/
theorem publishing_break_even_point 
  (fixed_cost : ℝ) 
  (variable_cost_per_book : ℝ) 
  (selling_price_per_book : ℝ) 
  (h1 : fixed_cost = 56430)
  (h2 : variable_cost_per_book = 8.25)
  (h3 : selling_price_per_book = 21.75) :
  ∃ (x : ℝ), 
    x = 4180 ∧ 
    fixed_cost + x * variable_cost_per_book = x * selling_price_per_book :=
by sorry

end NUMINAMATH_CALUDE_publishing_break_even_point_l2200_220024


namespace NUMINAMATH_CALUDE_exterior_angle_square_octagon_l2200_220017

-- Define the necessary structures
structure Polygon :=
  (sides : ℕ)

-- Define the square and octagon
def square : Polygon := ⟨4⟩
def octagon : Polygon := ⟨8⟩

-- Define the function to calculate interior angle of a regular polygon
def interior_angle (p : Polygon) : ℚ :=
  180 * (p.sides - 2) / p.sides

-- Define the theorem
theorem exterior_angle_square_octagon :
  let octagon_interior_angle := interior_angle octagon
  let square_interior_angle := 90
  360 - octagon_interior_angle - square_interior_angle = 135 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_square_octagon_l2200_220017


namespace NUMINAMATH_CALUDE_max_product_2015_l2200_220028

/-- Given the digits 2, 0, 1, and 5, the maximum product obtained by rearranging
    these digits into two numbers and multiplying them is 1050. -/
theorem max_product_2015 : ∃ (a b : ℕ),
  (a ≤ 99 ∧ b ≤ 99) ∧
  (∀ (d : ℕ), d ∈ [a.div 10, a % 10, b.div 10, b % 10] → d ∈ [2, 0, 1, 5]) ∧
  (a * b = 1050) ∧
  (∀ (c d : ℕ), c ≤ 99 → d ≤ 99 →
    (∀ (e : ℕ), e ∈ [c.div 10, c % 10, d.div 10, d % 10] → e ∈ [2, 0, 1, 5]) →
    c * d ≤ 1050) :=
by sorry

end NUMINAMATH_CALUDE_max_product_2015_l2200_220028


namespace NUMINAMATH_CALUDE_meeting_time_on_circular_track_l2200_220035

/-- The time taken for two people to meet on a circular track -/
theorem meeting_time_on_circular_track 
  (track_circumference : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : track_circumference = 528)
  (h2 : speed1 = 4.5)
  (h3 : speed2 = 3.75) :
  (track_circumference / ((speed1 + speed2) * 1000 / 60)) = 3.84 := by
  sorry

end NUMINAMATH_CALUDE_meeting_time_on_circular_track_l2200_220035


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l2200_220084

theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = !![3, 7; -2, -5]) : 
  (A^3)⁻¹ = !![13, -15; -14, -29] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l2200_220084


namespace NUMINAMATH_CALUDE_final_stamp_count_l2200_220003

/-- Represents the number of stamps in Tom's collection -/
def stamps_collection (initial : ℕ) (mike_gift : ℕ) : ℕ → ℕ
  | harry_gift => initial + mike_gift + harry_gift

/-- Theorem: Tom's final stamp collection contains 3,061 stamps -/
theorem final_stamp_count :
  let initial := 3000
  let mike_gift := 17
  let harry_gift := 2 * mike_gift + 10
  stamps_collection initial mike_gift harry_gift = 3061 := by
  sorry

#check final_stamp_count

end NUMINAMATH_CALUDE_final_stamp_count_l2200_220003


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l2200_220077

theorem min_value_2x_plus_y :
  ∀ x y : ℝ, (|y| ≤ 2 - x ∧ x ≥ -1) → (∀ x' y' : ℝ, |y'| ≤ 2 - x' ∧ x' ≥ -1 → 2*x + y ≤ 2*x' + y') ∧ (∃ x₀ y₀ : ℝ, |y₀| ≤ 2 - x₀ ∧ x₀ ≥ -1 ∧ 2*x₀ + y₀ = -5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l2200_220077


namespace NUMINAMATH_CALUDE_naH_required_for_h2O_l2200_220000

-- Define the molecules and their molar ratios in the reactions
structure Reaction :=
  (naH : ℚ) (h2O : ℚ) (naOH : ℚ) (h2 : ℚ)

-- Define the first step reaction
def firstStepReaction : Reaction :=
  { naH := 1, h2O := 1, naOH := 1, h2 := 1 }

-- Theorem stating that 1 mole of NaH is required to react with 1 mole of H2O
theorem naH_required_for_h2O :
  firstStepReaction.naH = firstStepReaction.h2O := by sorry

end NUMINAMATH_CALUDE_naH_required_for_h2O_l2200_220000


namespace NUMINAMATH_CALUDE_female_democrat_ratio_l2200_220041

theorem female_democrat_ratio (total_participants male_participants female_participants : ℕ)
  (female_democrats : ℕ) (h1 : total_participants = 870)
  (h2 : male_participants + female_participants = total_participants)
  (h3 : female_democrats = 145)
  (h4 : 4 * (male_participants / 4 + female_democrats) = total_participants) :
  2 * female_democrats = female_participants :=
by sorry

end NUMINAMATH_CALUDE_female_democrat_ratio_l2200_220041


namespace NUMINAMATH_CALUDE_complex_product_real_l2200_220043

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 - 2*I
  let z₂ : ℂ := 1 + a*I
  (z₁ * z₂).im = 0 → a = 2/3 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_l2200_220043


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2200_220057

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two vectors a and b, where a = (3,1) and b = (x,-1), 
    if a is parallel to b, then x = -3 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -1)
  are_parallel a b → x = -3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2200_220057


namespace NUMINAMATH_CALUDE_checkerboard_fraction_l2200_220068

/-- The number of rectangles formed on a 7x7 checkerboard with 8 horizontal and 8 vertical lines -/
def r : ℕ := 784

/-- The number of squares formed on a 7x7 checkerboard -/
def s : ℕ := 140

/-- m and n are relatively prime positive integers such that s/r = m/n -/
theorem checkerboard_fraction (m n : ℕ) (h1 : m.gcd n = 1) (h2 : m > 0) (h3 : n > 0) 
  (h4 : s * n = r * m) : m + n = 33 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_fraction_l2200_220068


namespace NUMINAMATH_CALUDE_bens_savings_proof_l2200_220004

/-- Ben's daily savings before his parents' contributions --/
def daily_savings : ℕ := 50 - 15

/-- The number of days Ben saved money --/
def num_days : ℕ := 7

/-- Ben's total savings after his mom doubled it --/
def doubled_savings : ℕ := 2 * (daily_savings * num_days)

/-- Ben's final amount after 7 days --/
def final_amount : ℕ := 500

/-- The additional amount Ben's dad gave him --/
def dads_contribution : ℕ := final_amount - doubled_savings

theorem bens_savings_proof :
  dads_contribution = 10 :=
sorry

end NUMINAMATH_CALUDE_bens_savings_proof_l2200_220004


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_l2200_220099

theorem count_four_digit_numbers :
  let first_four_digit : Nat := 1000
  let last_four_digit : Nat := 9999
  (last_four_digit - first_four_digit + 1 : Nat) = 9000 := by
sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_l2200_220099


namespace NUMINAMATH_CALUDE_red_bacon_bits_count_l2200_220010

def salad_problem (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ) : Prop :=
  mushrooms = 3 ∧
  cherry_tomatoes = 2 * mushrooms ∧
  pickles = 4 * cherry_tomatoes ∧
  bacon_bits = 4 * pickles ∧
  red_bacon_bits = bacon_bits / 3

theorem red_bacon_bits_count : ∃ (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ),
  salad_problem mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits ∧ red_bacon_bits = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_red_bacon_bits_count_l2200_220010


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l2200_220015

theorem twenty_five_percent_less_than_80 (x : ℚ) : x + (1/4) * x = 60 → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l2200_220015


namespace NUMINAMATH_CALUDE_continued_fraction_result_l2200_220007

/-- Given x satisfying the infinite continued fraction equation,
    prove that 1/((x+2)(x-3)) equals (-√3 - 2) / 2 -/
theorem continued_fraction_result (x : ℝ) 
  (hx : x = 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / x))) :
  1 / ((x + 2) * (x - 3)) = (-Real.sqrt 3 - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_result_l2200_220007


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l2200_220088

-- Define a square with a given area
def Square (area : ℝ) := {side : ℝ // side^2 = area}

-- Define the perimeter of a square
def perimeter (s : Square area) : ℝ := 4 * s.val

-- Theorem statement
theorem square_perimeter_from_area (s : Square 225) : 
  perimeter s = 60 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l2200_220088


namespace NUMINAMATH_CALUDE_arrangement_of_six_objects_l2200_220046

theorem arrangement_of_six_objects (n : ℕ) (h : n = 6) : 
  Nat.factorial n = 720 :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_of_six_objects_l2200_220046


namespace NUMINAMATH_CALUDE_oranges_discarded_per_day_l2200_220005

theorem oranges_discarded_per_day 
  (harvest_per_day : ℕ) 
  (days : ℕ) 
  (remaining_sacks : ℕ) 
  (h1 : harvest_per_day = 74)
  (h2 : days = 51)
  (h3 : remaining_sacks = 153) :
  (harvest_per_day * days - remaining_sacks) / days = 71 := by
  sorry

end NUMINAMATH_CALUDE_oranges_discarded_per_day_l2200_220005


namespace NUMINAMATH_CALUDE_function_minimum_implies_a_less_than_one_l2200_220008

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- State the theorem
theorem function_minimum_implies_a_less_than_one :
  ∀ a : ℝ, (∃ m : ℝ, ∀ x < 1, f a x ≥ f a m) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_implies_a_less_than_one_l2200_220008


namespace NUMINAMATH_CALUDE_marble_remainder_l2200_220080

theorem marble_remainder (n m k : ℤ) : ∃ q : ℤ, (8*n + 5) + (8*m + 3) + (8*k + 7) = 8*q + 7 := by
  sorry

end NUMINAMATH_CALUDE_marble_remainder_l2200_220080


namespace NUMINAMATH_CALUDE_variables_related_probability_l2200_220055

/-- The k-value obtained from a 2×2 contingency table -/
def k : ℝ := 4.073

/-- The probability that k^2 is greater than or equal to 3.841 -/
def p_3841 : ℝ := 0.05

/-- The probability that k^2 is greater than or equal to 5.024 -/
def p_5024 : ℝ := 0.025

/-- The theorem stating the probability of two variables being related -/
theorem variables_related_probability : ℝ := by
  sorry

end NUMINAMATH_CALUDE_variables_related_probability_l2200_220055


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l2200_220081

theorem pure_imaginary_modulus (a : ℝ) : 
  let z : ℂ := a^2 - 1 + (a + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l2200_220081


namespace NUMINAMATH_CALUDE_root_sum_squares_l2200_220029

theorem root_sum_squares (p q r : ℝ) : 
  (p + q + r = 15) → 
  (p * q + q * r + r * p = 22) →
  (p * q * r = 8) →
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 406 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l2200_220029


namespace NUMINAMATH_CALUDE_turtle_problem_l2200_220076

theorem turtle_problem (initial_turtles : ℕ) : initial_turtles = 9 →
  let additional_turtles := 3 * initial_turtles - 2
  let total_turtles := initial_turtles + additional_turtles
  let remaining_turtles := total_turtles / 2
  remaining_turtles = 17 := by
sorry

end NUMINAMATH_CALUDE_turtle_problem_l2200_220076


namespace NUMINAMATH_CALUDE_grandfather_grandson_ages_l2200_220092

theorem grandfather_grandson_ages :
  ∀ (x y a b : ℕ),
    x > 70 →
    x - a = 10 * (y - a) →
    x + b = 8 * (y + b) →
    x = 71 ∧ y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_grandfather_grandson_ages_l2200_220092


namespace NUMINAMATH_CALUDE_blue_to_black_pen_ratio_l2200_220013

/-- Given the conditions of John's pen collection, prove the ratio of blue to black pens --/
theorem blue_to_black_pen_ratio :
  ∀ (blue black red : ℕ),
  blue + black + red = 31 →
  black = red + 5 →
  blue = 18 →
  blue / black = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_to_black_pen_ratio_l2200_220013


namespace NUMINAMATH_CALUDE_certain_number_problem_l2200_220082

theorem certain_number_problem (x : ℝ) : 
  x - (1/4)*2 - (1/3)*3 - (1/7)*x = 27 → 
  (10/100) * x = 3.325 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2200_220082


namespace NUMINAMATH_CALUDE_z_squared_in_first_quadrant_l2200_220026

theorem z_squared_in_first_quadrant (z : ℂ) (h : (z - I) / (1 + I) = 2 - 2*I) :
  (z^2).re > 0 ∧ (z^2).im > 0 :=
by sorry

end NUMINAMATH_CALUDE_z_squared_in_first_quadrant_l2200_220026


namespace NUMINAMATH_CALUDE_target_hit_probability_l2200_220083

theorem target_hit_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h_prob_A : prob_A = 0.8) 
  (h_prob_B : prob_B = 0.7) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - prob_A) * (1 - prob_B) = 0.94 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l2200_220083


namespace NUMINAMATH_CALUDE_work_completion_time_l2200_220095

theorem work_completion_time (q p : ℝ) (h1 : q = 20) 
  (h2 : 4 * (1/p + 1/q) = 1 - 0.5333333333333333) : p = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2200_220095


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2200_220025

def repeating_decimal_123 : ℚ := 123 / 999
def repeating_decimal_0123 : ℚ := 123 / 9999
def repeating_decimal_000123 : ℚ := 123 / 999999

theorem sum_of_repeating_decimals :
  repeating_decimal_123 + repeating_decimal_0123 + repeating_decimal_000123 =
  (123 * 1000900) / (999 * 9999 * 100001) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2200_220025


namespace NUMINAMATH_CALUDE_solution_opposite_implies_a_l2200_220091

theorem solution_opposite_implies_a (a : ℝ) : 
  (∃ x : ℝ, 5 * x - 1 = 2 * x + a) ∧ 
  (∃ y : ℝ, 4 * y + 3 = 7) ∧
  (∀ x y : ℝ, (5 * x - 1 = 2 * x + a ∧ 4 * y + 3 = 7) → x = -y) →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_solution_opposite_implies_a_l2200_220091


namespace NUMINAMATH_CALUDE_abs_and_reciprocal_l2200_220079

theorem abs_and_reciprocal :
  (abs (-9 : ℝ) = 9) ∧ (((-3 : ℝ)⁻¹) = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_abs_and_reciprocal_l2200_220079


namespace NUMINAMATH_CALUDE_polynomial_sum_l2200_220051

theorem polynomial_sum (d a b c e : ℤ) (h : d ≠ 0) :
  (10 * d + 15 + 12 * d^2 + 2 * d^3) + (4 * d - 3 + 2 * d^2) = a * d^3 + b * d^2 + c * d + e →
  a + b + c + e = 42 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2200_220051


namespace NUMINAMATH_CALUDE_cookie_eating_contest_l2200_220065

theorem cookie_eating_contest (first_friend second_friend : ℚ) 
  (h1 : first_friend = 5/6)
  (h2 : second_friend = 2/3) :
  first_friend - second_friend = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_cookie_eating_contest_l2200_220065


namespace NUMINAMATH_CALUDE_normal_hours_calculation_l2200_220039

/-- Represents a worker's pay structure and a specific workday --/
structure WorkDay where
  regularRate : ℝ  -- Regular hourly rate
  overtimeMultiplier : ℝ  -- Overtime rate multiplier
  totalHours : ℝ  -- Total hours worked on a specific day
  totalEarnings : ℝ  -- Total earnings for the specific day
  normalHours : ℝ  -- Normal working hours per day

/-- Theorem stating that given the specific conditions, the normal working hours are 7.5 --/
theorem normal_hours_calculation (w : WorkDay) 
  (h1 : w.regularRate = 3.5)
  (h2 : w.overtimeMultiplier = 1.5)
  (h3 : w.totalHours = 10.5)
  (h4 : w.totalEarnings = 42)
  : w.normalHours = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_normal_hours_calculation_l2200_220039


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l2200_220012

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 187 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 187 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l2200_220012


namespace NUMINAMATH_CALUDE_solve_equation_l2200_220063

theorem solve_equation : 
  ∃ y : ℚ, (y^2 - 9*y + 8)/(y - 1) + (3*y^2 + 16*y - 12)/(3*y - 2) = -3 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2200_220063


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l2200_220011

theorem product_of_sum_of_squares (a b c d : ℤ) :
  let m := a^2 + b^2
  let n := c^2 + d^2
  m * n = (a*c - b*d)^2 + (a*d + b*c)^2 := by sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l2200_220011


namespace NUMINAMATH_CALUDE_minimal_points_double_star_l2200_220038

/-- Represents a regular n-pointed double star polygon -/
structure DoubleStarPolygon where
  n : ℕ
  angleA : ℝ
  angleB : ℝ

/-- Conditions for a valid double star polygon -/
def isValidDoubleStarPolygon (d : DoubleStarPolygon) : Prop :=
  d.n > 0 ∧
  d.angleA > 0 ∧
  d.angleB > 0 ∧
  d.angleA = d.angleB + 15 ∧
  d.n * 15 = 360

theorem minimal_points_double_star :
  ∀ d : DoubleStarPolygon, isValidDoubleStarPolygon d → d.n ≥ 24 :=
by sorry

end NUMINAMATH_CALUDE_minimal_points_double_star_l2200_220038


namespace NUMINAMATH_CALUDE_number_difference_l2200_220059

theorem number_difference (A B : ℕ) (h1 : A + B = 1812) (h2 : A = 7 * B + 4) : A - B = 1360 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2200_220059


namespace NUMINAMATH_CALUDE_rectangle_area_and_diagonal_l2200_220060

/-- Represents a rectangle with length, width, perimeter, area, and diagonal --/
structure Rectangle where
  length : ℝ
  width : ℝ
  perimeter : ℝ
  area : ℝ
  diagonal : ℝ

/-- Theorem about the area and diagonal of a specific rectangle --/
theorem rectangle_area_and_diagonal (r : Rectangle) 
  (h1 : r.length = 4 * r.width)
  (h2 : r.perimeter = 200) :
  r.area = 1600 ∧ r.diagonal = Real.sqrt 6800 := by
  sorry

#check rectangle_area_and_diagonal

end NUMINAMATH_CALUDE_rectangle_area_and_diagonal_l2200_220060


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2200_220075

theorem largest_multiple_of_15_under_500 : ∃ (n : ℕ), n = 495 ∧ 
  (∀ m : ℕ, m % 15 = 0 → m < 500 → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2200_220075


namespace NUMINAMATH_CALUDE_goat_roaming_area_specific_case_l2200_220053

/-- Represents the dimensions of a rectangular shed -/
structure ShedDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area a goat can roam when tied to the corner of a rectangular shed -/
def goatRoamingArea (shed : ShedDimensions) (leashLength : ℝ) : ℝ :=
  sorry

/-- Theorem stating the area a goat can roam under specific conditions -/
theorem goat_roaming_area_specific_case :
  let shed : ShedDimensions := { length := 5, width := 4 }
  let leashLength : ℝ := 4
  goatRoamingArea shed leashLength = 12.25 * Real.pi := by sorry

end NUMINAMATH_CALUDE_goat_roaming_area_specific_case_l2200_220053


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2200_220002

theorem quadratic_inequality_condition (a : ℝ) :
  ((∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 ≤ a ∧ a ≤ 1)) ∧
  ¬((0 ≤ a ∧ a ≤ 1) → (∀ x : ℝ, x^2 - 2*a*x + a > 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2200_220002


namespace NUMINAMATH_CALUDE_intersection_range_l2200_220030

/-- The range of k for which the line y = kx + 2 intersects the right branch of 
    the hyperbola x^2 - y^2 = 6 at two distinct points -/
theorem intersection_range :
  ∀ k : ℝ, 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁ > 0 ∧ x₂ > 0 ∧
   x₁^2 - (k * x₁ + 2)^2 = 6 ∧
   x₂^2 - (k * x₂ + 2)^2 = 6) →
  -Real.sqrt 15 / 3 < k ∧ k < -1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l2200_220030


namespace NUMINAMATH_CALUDE_f_properties_l2200_220023

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x * x^2

theorem f_properties :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 →
  (∀ x ∈ Set.Ioo 0 (Real.exp (-1/2)), f x ≥ f (Real.exp (-1/2))) ∧
  (∀ x ∈ Set.Ioi (Real.exp (-1/2)), f x ≥ f (Real.exp (-1/2))) ∧
  f (Real.exp (-1/2)) = -1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2200_220023


namespace NUMINAMATH_CALUDE_science_club_enrollment_l2200_220049

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_math : math = 95)
  (h_physics : physics = 70)
  (h_both : both = 25) :
  total - (math + physics - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l2200_220049


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l2200_220034

theorem triangle_side_and_area 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h1 : c = 2)
  (h2 : Real.sin A = 2 * Real.sin C)
  (h3 : Real.cos B = 1/4)
  (h4 : a = c * Real.sin A / Real.sin C) -- Sine law
  (h5 : 0 < Real.sin C) -- Assumption to avoid division by zero
  (h6 : 0 ≤ A ∧ A < π) -- Assumption for valid angle A
  (h7 : 0 ≤ B ∧ B < π) -- Assumption for valid angle B
  (h8 : 0 ≤ C ∧ C < π) -- Assumption for valid angle C
  : a = 4 ∧ (1/2 * a * c * Real.sin B = Real.sqrt 15) := by
  sorry

#check triangle_side_and_area

end NUMINAMATH_CALUDE_triangle_side_and_area_l2200_220034


namespace NUMINAMATH_CALUDE_point_four_units_from_negative_three_l2200_220069

theorem point_four_units_from_negative_three :
  {x : ℝ | |x - (-3)| = 4} = {1, -7} := by sorry

end NUMINAMATH_CALUDE_point_four_units_from_negative_three_l2200_220069


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2200_220042

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y = f (x - y)) →
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2200_220042


namespace NUMINAMATH_CALUDE_conference_handshakes_l2200_220096

/-- The number of handshakes in a conference where each person shakes hands with every other person exactly once. -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a conference of 10 people where each person shakes hands with every other person exactly once, there are 45 handshakes. -/
theorem conference_handshakes : num_handshakes 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l2200_220096


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2200_220044

theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  let angle := Real.pi / 3
  (a.1^2 + a.2^2 = 1) →
  (b.1^2 + b.2^2 = 4) →
  (a.1 * b.1 + a.2 * b.2 = 2 * Real.cos angle) →
  ((2*a.1 - b.1)^2 + (2*a.2 - b.2)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2200_220044


namespace NUMINAMATH_CALUDE_intersection_on_unit_circle_l2200_220037

theorem intersection_on_unit_circle (k₁ k₂ : ℝ) (h : k₁ * k₂ + 1 = 0) :
  ∃ (x y : ℝ), (y = k₁ * x + 1) ∧ (y = k₂ * x - 1) ∧ (x^2 + y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_on_unit_circle_l2200_220037


namespace NUMINAMATH_CALUDE_inverse_f_sum_l2200_220072

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem inverse_f_sum : ∃ y z : ℝ, f y = 9 ∧ f z = -81 ∧ y + z = -6 := by sorry

end NUMINAMATH_CALUDE_inverse_f_sum_l2200_220072


namespace NUMINAMATH_CALUDE_line_equation_proof_l2200_220009

/-- A line in the xy-plane passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The line passing through (2,3) and (4,4) -/
def line_through_points : Line :=
  { a := 1, b := 8, c := -26 }

theorem line_equation_proof :
  (line_through_points.contains 2 3) ∧
  (line_through_points.contains 4 4) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2200_220009


namespace NUMINAMATH_CALUDE_max_stable_angle_l2200_220056

/-- A sign consisting of two uniform legs attached by a frictionless hinge -/
structure Sign where
  μ : ℝ  -- coefficient of friction between the ground and the legs
  θ : ℝ  -- angle between the legs

/-- The condition for the sign to be in equilibrium -/
def is_stable (s : Sign) : Prop :=
  Real.tan (s.θ / 2) = 2 * s.μ

/-- Theorem stating the maximum angle for stability -/
theorem max_stable_angle (s : Sign) :
  is_stable s ↔ s.θ = Real.arctan (2 * s.μ) * 2 :=
sorry

end NUMINAMATH_CALUDE_max_stable_angle_l2200_220056


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l2200_220064

/-- Systematic sampling problem -/
theorem systematic_sampling_problem 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (group_size : ℕ) 
  (sample_size : ℕ) 
  (sixteenth_group_num : ℕ) :
  total_students = 160 →
  num_groups = 20 →
  group_size = 8 →
  sample_size = 20 →
  sixteenth_group_num = 126 →
  ∃ (first_group_num : ℕ), 
    first_group_num + (15 * group_size) = sixteenth_group_num ∧
    first_group_num = 6 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_problem_l2200_220064


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l2200_220098

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |x - 3|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x + |2*x - 5| ≥ 6} = {x : ℝ | x ≥ 4 ∨ x ≤ 0} := by sorry

-- Part II
theorem range_of_a_part_ii (h : Set.Icc (-1 : ℝ) 2 ⊆ Set.range (g a)) :
  a ≤ 1 ∨ a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l2200_220098


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l2200_220006

theorem lcm_gcf_problem (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 4) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l2200_220006


namespace NUMINAMATH_CALUDE_lecture_scheduling_l2200_220032

theorem lecture_scheduling (n : ℕ) (h : n = 6) :
  (n! / 2 : ℕ) = 360 := by
  sorry

#check lecture_scheduling

end NUMINAMATH_CALUDE_lecture_scheduling_l2200_220032


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l2200_220089

theorem smallest_positive_integer_satisfying_congruences :
  ∃ x : ℕ+, 
    (45 * x.val + 15) % 25 = 5 ∧ 
    x.val % 4 = 3 ∧
    ∀ y : ℕ+, 
      ((45 * y.val + 15) % 25 = 5 ∧ y.val % 4 = 3) → 
      x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l2200_220089


namespace NUMINAMATH_CALUDE_convention_center_tables_l2200_220086

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- Calculates the number of tables needed given the total number of people and people per table -/
def calculateTables (totalPeople : Nat) (peoplePerTable : Nat) : Nat :=
  totalPeople / peoplePerTable

theorem convention_center_tables :
  let seatingCapacityBase7 : Nat := 315
  let peoplePerTable : Nat := 3
  let totalPeopleBase10 : Nat := base7ToBase10 seatingCapacityBase7
  calculateTables totalPeopleBase10 peoplePerTable = 53 := by
  sorry

end NUMINAMATH_CALUDE_convention_center_tables_l2200_220086


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2200_220031

/-- 
Given a boat that covers the same distance downstream and upstream,
with known travel times and stream speed, this theorem proves
the speed of the boat in still water.
-/
theorem boat_speed_in_still_water 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (stream_speed : ℝ) 
  (h1 : downstream_time = 1)
  (h2 : upstream_time = 1.5)
  (h3 : stream_speed = 3) : 
  ∃ (boat_speed : ℝ), boat_speed = 15 ∧ 
    downstream_time * (boat_speed + stream_speed) = 
    upstream_time * (boat_speed - stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2200_220031


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2200_220018

theorem sqrt_equation_solution :
  ∃ y : ℝ, (Real.sqrt 27 + Real.sqrt y) / Real.sqrt 75 = 2.4 → y = 243 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2200_220018


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l2200_220078

/-- Given two positive integers with specific HCF and LCM properties, prove the other factor of their LCM -/
theorem lcm_factor_proof (A B : ℕ) (hA : A = 460) (hHCF : Nat.gcd A B = 20) 
  (hLCM_factor : ∃ (k : ℕ), Nat.lcm A B = 20 * 21 * k) : 
  ∃ (k : ℕ), Nat.lcm A B = 20 * 21 * 23 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l2200_220078


namespace NUMINAMATH_CALUDE_triangle_medians_inequalities_l2200_220048

-- Define a structure for a triangle with medians and circumradius
structure Triangle where
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  R : ℝ
  h_positive : m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧ R > 0

-- Theorem statement
theorem triangle_medians_inequalities (t : Triangle) : 
  t.m_a^2 + t.m_b^2 + t.m_c^2 ≤ (27 * t.R^2) / 4 ∧ 
  t.m_a + t.m_b + t.m_c ≤ (9 * t.R) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_medians_inequalities_l2200_220048


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2200_220021

/-- Sum of first n terms of an arithmetic sequence -/
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := sorry

/-- The sequence a is arithmetic -/
def is_arithmetic (a : ℕ → ℝ) : Prop := sorry

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (h : is_arithmetic a) 
  (h1 : S 3 a / S 6 a = 1 / 3) : S 9 a / S 6 a = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2200_220021


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_twenty_two_thirds_l2200_220036

theorem greatest_integer_less_than_negative_twenty_two_thirds :
  Int.floor (-22 / 3) = -8 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_twenty_two_thirds_l2200_220036


namespace NUMINAMATH_CALUDE_distance_to_external_point_specific_distance_to_external_point_l2200_220019

/-- Given a circle with radius r and two tangents drawn from a common external point P
    with a sum length of s, the distance from the center O to P is sqrt(r^2 + (s/2)^2). -/
theorem distance_to_external_point (r s : ℝ) (hr : r > 0) (hs : s > 0) :
  let d := Real.sqrt (r^2 + (s/2)^2)
  d = Real.sqrt (r^2 + (s/2)^2) := by
  sorry

/-- For a circle with radius 11 and two tangents with sum length 120,
    the distance from the center to the external point is 61. -/
theorem specific_distance_to_external_point :
  let r : ℝ := 11
  let s : ℝ := 120
  let d := Real.sqrt (r^2 + (s/2)^2)
  d = 61 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_external_point_specific_distance_to_external_point_l2200_220019


namespace NUMINAMATH_CALUDE_cafeteria_seats_unseated_fraction_l2200_220074

theorem cafeteria_seats_unseated_fraction :
  let total_tables : ℕ := 15
  let seats_per_table : ℕ := 10
  let seats_taken : ℕ := 135
  let total_seats : ℕ := total_tables * seats_per_table
  let seats_unseated : ℕ := total_seats - seats_taken
  (seats_unseated : ℚ) / total_seats = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_seats_unseated_fraction_l2200_220074


namespace NUMINAMATH_CALUDE_range_of_y_over_x_l2200_220090

theorem range_of_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) :
  ∃ (k : ℝ), y / x = k ∧ -Real.sqrt 3 ≤ k ∧ k ≤ Real.sqrt 3 ∧
  (∀ (m : ℝ), y / x = m → -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_l2200_220090


namespace NUMINAMATH_CALUDE_potato_cost_theorem_l2200_220066

-- Define the given conditions
def people_count : ℕ := 40
def potatoes_per_person : ℚ := 3/2
def bag_weight : ℕ := 20
def bag_cost : ℕ := 5

-- Define the theorem
theorem potato_cost_theorem : 
  (people_count : ℚ) * potatoes_per_person / bag_weight * bag_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_potato_cost_theorem_l2200_220066


namespace NUMINAMATH_CALUDE_approximating_functions_theorem1_approximating_functions_theorem2_l2200_220067

-- Define the concept of "approximating functions"
def approximating_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → -1 ≤ f x - g x ∧ f x - g x ≤ 1

-- Define the functions
def f1 (x : ℝ) : ℝ := x - 5
def f2 (x : ℝ) : ℝ := x^2 - 4*x
def g1 (x : ℝ) : ℝ := x^2 - 1
def g2 (x : ℝ) : ℝ := 2*x^2 - x

-- State the theorems to be proved
theorem approximating_functions_theorem1 :
  approximating_functions f1 f2 3 4 := by sorry

theorem approximating_functions_theorem2 :
  approximating_functions g1 g2 0 1 := by sorry

end NUMINAMATH_CALUDE_approximating_functions_theorem1_approximating_functions_theorem2_l2200_220067


namespace NUMINAMATH_CALUDE_smallest_number_property_l2200_220050

/-- The smallest natural number divisible by 5 with a digit sum of 100 -/
def smallest_number : ℕ := 599999999995

/-- Function to calculate the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- Theorem stating that 599999999995 is the smallest natural number divisible by 5 with a digit sum of 100 -/
theorem smallest_number_property :
  (∀ m : ℕ, m < smallest_number → (m % 5 = 0 → digit_sum m ≠ 100)) ∧
  smallest_number % 5 = 0 ∧
  digit_sum smallest_number = 100 :=
by sorry

#eval smallest_number
#eval digit_sum smallest_number
#eval smallest_number % 5

end NUMINAMATH_CALUDE_smallest_number_property_l2200_220050


namespace NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l2200_220085

/-- Represents the number of points for each type of goal -/
def threePointValue : ℕ := 3
def twoPointValue : ℕ := 2

/-- Represents the number of goals Marcus scored -/
def marcusThreePointers : ℕ := 5
def marcusTwoPointers : ℕ := 10

/-- Represents the total points scored by the team -/
def teamTotalPoints : ℕ := 70

/-- Calculates the total points scored by Marcus -/
def marcusTotalPoints : ℕ :=
  marcusThreePointers * threePointValue + marcusTwoPointers * twoPointValue

/-- Theorem: Marcus scored 50% of the team's total points -/
theorem marcus_percentage_of_team_points :
  (marcusTotalPoints : ℚ) / teamTotalPoints = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l2200_220085


namespace NUMINAMATH_CALUDE_first_term_is_two_l2200_220054

/-- An increasing arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  arithmetic : ∃ d : ℝ, ∀ n, a (n + 1) - a n = d
  sum_first_three : a 1 + a 2 + a 3 = 12
  product_first_three : a 1 * a 2 * a 3 = 48

/-- The first term of the arithmetic sequence is 2 -/
theorem first_term_is_two (seq : ArithmeticSequence) : seq.a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_two_l2200_220054


namespace NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l2200_220061

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by sorry

end NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l2200_220061


namespace NUMINAMATH_CALUDE_prob_b_in_middle_l2200_220022

def number_of_people : ℕ := 3

def total_arrangements (n : ℕ) : ℕ := n.factorial

def middle_arrangements (n : ℕ) : ℕ := (n - 1).factorial

def probability_in_middle (n : ℕ) : ℚ :=
  (middle_arrangements n : ℚ) / (total_arrangements n : ℚ)

theorem prob_b_in_middle :
  probability_in_middle number_of_people = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_b_in_middle_l2200_220022


namespace NUMINAMATH_CALUDE_min_triangle_count_l2200_220062

structure Graph (n : ℕ) :=
  (m : ℕ)
  (edges : Finset (Fin n × Fin n))
  (edge_count : edges.card = m)
  (edge_distinct : ∀ (e : Fin n × Fin n), e ∈ edges → e.1 ≠ e.2)

def triangle_count (n : ℕ) (G : Graph n) : ℕ := sorry

theorem min_triangle_count (n : ℕ) (G : Graph n) :
  triangle_count n G ≥ (4 * G.m : ℚ) / (3 * n) * (G.m - n^2 / 4) :=
sorry

end NUMINAMATH_CALUDE_min_triangle_count_l2200_220062


namespace NUMINAMATH_CALUDE_triangle_area_with_consecutive_integer_sides_and_height_l2200_220087

theorem triangle_area_with_consecutive_integer_sides_and_height :
  ∀ (a b c h : ℕ),
    a + 1 = b →
    b + 1 = c →
    c + 1 = h →
    (1 / 2 : ℚ) * b * h = 84 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_consecutive_integer_sides_and_height_l2200_220087
