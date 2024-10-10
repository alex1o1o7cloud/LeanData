import Mathlib

namespace witch_clock_theorem_l891_89111

def clock_cycle (t : ℕ) : ℕ :=
  (5 * (t / 8 + 1) - 3 * (t / 8)) % 60

theorem witch_clock_theorem (t : ℕ) (h : t = 2022) :
  clock_cycle t = 28 := by
  sorry

end witch_clock_theorem_l891_89111


namespace cereal_eating_time_l891_89127

/-- The time required for two people to eat a certain amount of cereal together -/
def eating_time (quick_rate : ℚ) (slow_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (quick_rate + slow_rate)

/-- Theorem: Mr. Quick and Mr. Slow eat 5 pounds of cereal in 600/11 minutes -/
theorem cereal_eating_time :
  let quick_rate : ℚ := 1 / 15
  let slow_rate : ℚ := 1 / 40
  let total_amount : ℚ := 5
  eating_time quick_rate slow_rate total_amount = 600 / 11 := by
  sorry

#eval eating_time (1/15 : ℚ) (1/40 : ℚ) 5

end cereal_eating_time_l891_89127


namespace bobby_paycheck_l891_89185

/-- Calculates the final amount in Bobby's paycheck after deductions --/
def final_paycheck (gross_salary : ℚ) : ℚ :=
  let federal_tax := gross_salary * (1/3)
  let state_tax := gross_salary * (8/100)
  let local_tax := gross_salary * (5/100)
  let health_insurance := 50
  let life_insurance := 20
  let parking_fee := 10
  let retirement_contribution := gross_salary * (3/100)
  let total_deductions := federal_tax + state_tax + local_tax + health_insurance + life_insurance + parking_fee + retirement_contribution
  gross_salary - total_deductions

/-- Proves that Bobby's final paycheck amount is $148 --/
theorem bobby_paycheck : final_paycheck 450 = 148 := by
  sorry

#eval final_paycheck 450

end bobby_paycheck_l891_89185


namespace isosceles_triangle_base_length_l891_89165

/-- An isosceles triangle with congruent sides of 6 cm and perimeter of 20 cm has a base of 8 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base : ℝ), 
    base > 0 → 
    6 + 6 + base = 20 → 
    base = 8 :=
by sorry

end isosceles_triangle_base_length_l891_89165


namespace car_travel_time_l891_89182

/-- Given a car and a train traveling between two stations, this theorem proves
    the time taken by the car to reach the destination. -/
theorem car_travel_time (car_time train_time : ℝ) : 
  train_time = car_time + 2 →  -- The train takes 2 hours longer than the car
  car_time + train_time = 11 → -- The combined time is 11 hours
  car_time = 4.5 := by
  sorry

#check car_travel_time

end car_travel_time_l891_89182


namespace beach_probability_l891_89178

def beach_scenario (total_sunglasses : ℕ) (total_caps : ℕ) (prob_cap_given_sunglasses : ℚ) : Prop :=
  ∃ (both : ℕ),
    total_sunglasses = 60 ∧
    total_caps = 40 ∧
    prob_cap_given_sunglasses = 1/3 ∧
    both ≤ total_sunglasses ∧
    both ≤ total_caps ∧
    (both : ℚ) / total_sunglasses = prob_cap_given_sunglasses

theorem beach_probability (total_sunglasses total_caps : ℕ) (prob_cap_given_sunglasses : ℚ) :
  beach_scenario total_sunglasses total_caps prob_cap_given_sunglasses →
  (∃ (both : ℕ), (both : ℚ) / total_caps = 1/2) :=
by sorry

end beach_probability_l891_89178


namespace min_distance_point_coordinates_l891_89186

/-- Given two fixed points C(0,4) and K(6,0) in a Cartesian coordinate system,
    with A being a moving point on the line segment OK,
    D being the midpoint of AC,
    and B obtained by rotating AD clockwise 90° around A,
    prove that when BK reaches its minimum value,
    the coordinates of point B are (26/5, 8/5). -/
theorem min_distance_point_coordinates :
  ∀ (A : ℝ × ℝ) (B : ℝ × ℝ),
  let C : ℝ × ℝ := (0, 4)
  let K : ℝ × ℝ := (6, 0)
  let O : ℝ × ℝ := (0, 0)
  -- A is on line segment OK
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A = (t * K.1, 0)) →
  -- D is midpoint of AC
  let D : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  -- B is obtained by rotating AD 90° clockwise around A
  (B.1 = A.1 + (D.2 - A.2) ∧ B.2 = A.2 - (D.1 - A.1)) →
  -- When BK reaches its minimum value
  (∀ (A' : ℝ × ℝ) (B' : ℝ × ℝ),
    (∃ t' : ℝ, 0 ≤ t' ∧ t' ≤ 1 ∧ A' = (t' * K.1, 0)) →
    let D' : ℝ × ℝ := ((A'.1 + C.1) / 2, (A'.2 + C.2) / 2)
    (B'.1 = A'.1 + (D'.2 - A'.2) ∧ B'.2 = A'.2 - (D'.1 - A'.1)) →
    (B.1 - K.1)^2 + (B.2 - K.2)^2 ≤ (B'.1 - K.1)^2 + (B'.2 - K.2)^2) →
  -- Then the coordinates of B are (26/5, 8/5)
  B = (26/5, 8/5) := by sorry

end min_distance_point_coordinates_l891_89186


namespace license_plate_count_l891_89126

/-- The number of digits in a license plate -/
def num_digits : ℕ := 5

/-- The number of letters in a license plate -/
def num_letters : ℕ := 3

/-- The number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- The number of possible letters -/
def letter_choices : ℕ := 26

/-- The number of non-vowel letters -/
def non_vowel_choices : ℕ := 21

/-- The number of positions where the letter block can be placed -/
def block_positions : ℕ := num_digits + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  block_positions * digit_choices^num_digits * (letter_choices^num_letters - non_vowel_choices^num_letters)

theorem license_plate_count : total_license_plates = 4989000000 := by
  sorry

end license_plate_count_l891_89126


namespace quadratic_function_properties_l891_89198

/-- A quadratic function satisfying specific conditions -/
def f (x : ℝ) : ℝ := 6 * x^2 - 4

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties :
  (f (-1) = 2) ∧
  (deriv f 0 = 0) ∧
  (∫ x in (0)..(1), f x = -2) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≥ -4) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≤ 2) ∧
  (f 0 = -4) ∧
  (f 1 = 2) ∧
  (f (-1) = 2) := by
  sorry

end quadratic_function_properties_l891_89198


namespace simplify_negative_fraction_power_l891_89139

theorem simplify_negative_fraction_power :
  (-1 / 343 : ℝ) ^ (-3/5 : ℝ) = -343 := by sorry

end simplify_negative_fraction_power_l891_89139


namespace three_digit_divisible_by_nine_l891_89110

theorem three_digit_divisible_by_nine :
  ∀ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- Three-digit number
    n % 10 = 2 ∧          -- Units digit is 2
    n / 100 = 4 ∧         -- Hundreds digit is 4
    n % 9 = 0             -- Divisible by 9
    → n = 432 :=
by sorry

end three_digit_divisible_by_nine_l891_89110


namespace jim_sara_savings_equality_l891_89175

/-- Proves that Jim and Sara will have saved the same amount after 820 weeks -/
theorem jim_sara_savings_equality :
  let sara_initial : ℕ := 4100
  let sara_weekly : ℕ := 10
  let jim_weekly : ℕ := 15
  let weeks : ℕ := 820
  sara_initial + sara_weekly * weeks = jim_weekly * weeks :=
by sorry

end jim_sara_savings_equality_l891_89175


namespace yoongi_has_bigger_number_l891_89141

theorem yoongi_has_bigger_number : ∀ (yoongi_number jungkook_number : ℕ),
  yoongi_number = 4 →
  jungkook_number = 6 / 3 →
  yoongi_number > jungkook_number :=
by
  sorry

end yoongi_has_bigger_number_l891_89141


namespace smaller_circle_radius_l891_89162

theorem smaller_circle_radius (r_large : ℝ) (A₁ A₂ : ℝ) : 
  r_large = 4 →
  A₁ + A₂ = π * (2 * r_large)^2 →
  2 * A₂ = A₁ + (A₁ + A₂) →
  A₁ = π * r_small^2 →
  r_small = 4 := by
  sorry

end smaller_circle_radius_l891_89162


namespace problems_finished_at_school_l891_89196

def math_problems : ℕ := 18
def science_problems : ℕ := 11
def problems_left : ℕ := 5

theorem problems_finished_at_school :
  math_problems + science_problems - problems_left = 24 := by
  sorry

end problems_finished_at_school_l891_89196


namespace circle_area_ratio_l891_89197

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * π * r₁)) = (48 / 360 * (2 * π * r₂)) →
  (π * r₁^2) / (π * r₂^2) = 16 / 25 := by
  sorry

end circle_area_ratio_l891_89197


namespace column_sorting_preserves_row_order_l891_89130

/-- Represents a 10x10 table of natural numbers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Check if a row is sorted in ascending order -/
def is_row_sorted (t : Table) (row : Fin 10) : Prop :=
  ∀ i j : Fin 10, i < j → t row i ≤ t row j

/-- Check if a column is sorted in ascending order -/
def is_column_sorted (t : Table) (col : Fin 10) : Prop :=
  ∀ i j : Fin 10, i < j → t i col ≤ t j col

/-- Check if all rows are sorted in ascending order -/
def are_all_rows_sorted (t : Table) : Prop :=
  ∀ row : Fin 10, is_row_sorted t row

/-- Check if all columns are sorted in ascending order -/
def are_all_columns_sorted (t : Table) : Prop :=
  ∀ col : Fin 10, is_column_sorted t col

/-- The table contains the first 100 natural numbers -/
def contains_first_100_numbers (t : Table) : Prop :=
  ∀ n : ℕ, n ≤ 100 → ∃ i j : Fin 10, t i j = n

theorem column_sorting_preserves_row_order :
  ∀ t : Table,
  contains_first_100_numbers t →
  are_all_rows_sorted t →
  ∃ t' : Table,
    (∀ i j : Fin 10, t i j ≤ t' i j) ∧
    are_all_columns_sorted t' ∧
    are_all_rows_sorted t' :=
sorry

end column_sorting_preserves_row_order_l891_89130


namespace quadratic_equation_equivalence_l891_89134

theorem quadratic_equation_equivalence : ∃ (x : ℝ), 16 * x^2 - 32 * x - 512 = 0 ↔ ∃ (x : ℝ), (x - 1)^2 = 33 := by
  sorry

end quadratic_equation_equivalence_l891_89134


namespace perfect_square_between_prime_sums_l891_89129

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumOfFirstNPrimes (n : ℕ) : ℕ := (List.range n).map (nthPrime ∘ (· + 1)) |>.sum

/-- There exists a perfect square between the sum of the first n primes and the sum of the first n+1 primes -/
theorem perfect_square_between_prime_sums (n : ℕ) :
  ∃ k : ℕ, sumOfFirstNPrimes n < k^2 ∧ k^2 < sumOfFirstNPrimes (n + 1) := by sorry

end perfect_square_between_prime_sums_l891_89129


namespace a_investment_value_l891_89118

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- The theorem states that given the specific conditions of the partnership,
    a's investment must be 24000 --/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 32000)
  (hc : p.c_investment = 36000)
  (hp : p.total_profit = 92000)
  (hcs : p.c_profit_share = 36000)
  (h_profit_distribution : p.c_profit_share = p.c_investment * p.total_profit / (p.a_investment + p.b_investment + p.c_investment)) :
  p.a_investment = 24000 := by
  sorry


end a_investment_value_l891_89118


namespace line_segment_endpoint_l891_89194

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  ((1 - (-8))^2 + (y - 3)^2)^(1/2 : ℝ) = 15 → 
  y = 15 := by
sorry

end line_segment_endpoint_l891_89194


namespace intersection_condition_l891_89176

/-- The line equation -/
def line_equation (x y m : ℝ) : Prop := x - y + m = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 1 = 0

/-- Two distinct intersection points exist -/
def has_two_distinct_intersections (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    line_equation x₁ y₁ m ∧ circle_equation x₁ y₁ ∧
    line_equation x₂ y₂ m ∧ circle_equation x₂ y₂

/-- The theorem statement -/
theorem intersection_condition (m : ℝ) :
  0 < m → m < 1 → has_two_distinct_intersections m :=
by sorry

end intersection_condition_l891_89176


namespace valid_choices_count_l891_89135

/-- Represents a point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle in the plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a straight line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The set of 9 points created by the intersection of two lines and two circles -/
def intersection_points : Finset Point := sorry

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if three points lie on the same circle -/
def on_same_circle (p q r : Point) (c1 c2 : Circle) : Prop := sorry

/-- The number of ways to choose 4 points from the intersection points
    such that no 3 of them are collinear or on the same circle -/
def valid_choices : ℕ := sorry

theorem valid_choices_count :
  valid_choices = 114 :=
sorry

end valid_choices_count_l891_89135


namespace intersection_complement_theorem_l891_89159

-- Define the universal set I as ℝ
def I : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^(Real.sqrt (3 + 2*x - x^2))}

-- Define set N
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 2)}

-- Theorem statement
theorem intersection_complement_theorem : M ∩ (I \ N) = Set.Icc 1 2 := by
  sorry

end intersection_complement_theorem_l891_89159


namespace arithmetic_sequence_first_term_l891_89167

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 3) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term (a : ℚ) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → S a (3 * n) / S a n = c) →
  a = 3 / 2 := by
  sorry

end arithmetic_sequence_first_term_l891_89167


namespace prob_not_pass_overall_prob_pass_technical_given_overall_l891_89199

-- Define the probabilities of not passing each review aspect
def p_not_pass_norms : ℚ := 4/25
def p_not_pass_account : ℚ := 13/48
def p_not_pass_content : ℚ := 1/5

-- Define the probability of passing both overall review and technical skills test
def p_pass_both : ℚ := 35/100

-- Theorem for the probability of not passing overall review
theorem prob_not_pass_overall : 
  1 - (1 - p_not_pass_norms) * (1 - p_not_pass_account) * (1 - p_not_pass_content) = 51/100 := by sorry

-- Theorem for the probability of passing technical skills test given passing overall review
theorem prob_pass_technical_given_overall : 
  let p_pass_overall := 1 - (1 - (1 - p_not_pass_norms) * (1 - p_not_pass_account) * (1 - p_not_pass_content))
  p_pass_both / p_pass_overall = 5/7 := by sorry

end prob_not_pass_overall_prob_pass_technical_given_overall_l891_89199


namespace conference_attendees_l891_89142

theorem conference_attendees (men : ℕ) : 
  (men : ℝ) * 0.1 + 300 * 0.6 + 500 * 0.7 = (men + 300 + 500 : ℝ) * (1 - 0.5538461538461539) →
  men = 500 := by
sorry

end conference_attendees_l891_89142


namespace sandbox_area_calculation_l891_89106

/-- The area of a rectangular sandbox in square centimeters -/
def sandbox_area (length_meters : ℝ) (width_cm : ℝ) : ℝ :=
  (length_meters * 100) * width_cm

/-- Theorem: The area of a rectangular sandbox with length 3.12 meters and width 146 centimeters is 45552 square centimeters -/
theorem sandbox_area_calculation :
  sandbox_area 3.12 146 = 45552 := by
  sorry

end sandbox_area_calculation_l891_89106


namespace min_distance_curve_line_l891_89158

/-- Given real numbers a, b, c, d satisfying the conditions,
    prove that the minimum value of (a-c)^2 + (b-d)^2 is (9/5) * (ln(e/3))^2 -/
theorem min_distance_curve_line (a b c d : ℝ) 
    (h1 : (a + 3 * Real.log a) / b = 1)
    (h2 : (d - 3) / (2 * c) = 1) :
    ∃ (min : ℝ), min = (9/5) * (Real.log (Real.exp 1 / 3))^2 ∧
    ∀ (x y z w : ℝ), 
    (x + 3 * Real.log x) / y = 1 → 
    (w - 3) / (2 * z) = 1 → 
    (x - z)^2 + (y - w)^2 ≥ min :=
by sorry

end min_distance_curve_line_l891_89158


namespace sqrt_product_simplification_l891_89147

theorem sqrt_product_simplification : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end sqrt_product_simplification_l891_89147


namespace partial_fraction_decomposition_l891_89137

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    (5 * x - 3) / (x^2 - 3*x - 18) = C / (x - 6) + D / (x + 3)) →
  C = 3 ∧ D = 2 := by
sorry

end partial_fraction_decomposition_l891_89137


namespace complex_equation_solution_l891_89117

/-- Given a complex number Z satisfying (1+i)Z = 2, prove that Z = 1 - i -/
theorem complex_equation_solution (Z : ℂ) (h : (1 + Complex.I) * Z = 2) : Z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l891_89117


namespace part_one_part_two_l891_89125

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 2*a| - |x - a|

-- Part I
theorem part_one (a : ℝ) : f a 1 > 1 ↔ a ∈ Set.Iic (-1) ∪ Set.Ioi 1 := by sorry

-- Part II
theorem part_two (a : ℝ) (h : a < 0) :
  (∀ x y : ℝ, x ≤ a → y ≤ a → f a x ≤ |y + 2020| + |y - a|) ↔
  a ∈ Set.Icc (-1010) 0 := by sorry

end part_one_part_two_l891_89125


namespace other_communities_count_l891_89131

/-- The number of boys belonging to other communities in a school with given total and percentages of specific communities -/
theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 14 / 100 →
  sikh_percent = 10 / 100 →
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 272 := by
  sorry

#check other_communities_count

end other_communities_count_l891_89131


namespace blue_to_red_light_ratio_l891_89169

/-- Proves that the ratio of blue lights to red lights is 3:1 given the problem conditions -/
theorem blue_to_red_light_ratio :
  let initial_white_lights : ℕ := 59
  let red_lights : ℕ := 12
  let green_lights : ℕ := 6
  let remaining_to_buy : ℕ := 5
  let total_colored_lights : ℕ := initial_white_lights - remaining_to_buy
  let blue_lights : ℕ := total_colored_lights - (red_lights + green_lights)
  (blue_lights : ℚ) / red_lights = 3 / 1 := by
sorry

end blue_to_red_light_ratio_l891_89169


namespace inequality_proof_l891_89122

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem inequality_proof (a s t : ℝ) (h1 : s > 0) (h2 : t > 0) 
  (h3 : 2 * s + t = a) 
  (h4 : Set.Icc (-1) 7 = {x | f a x ≤ 4}) : 
  1 / s + 8 / t ≥ 6 := by
  sorry


end inequality_proof_l891_89122


namespace pyramid_volume_l891_89163

theorem pyramid_volume (base_side : ℝ) (height : ℝ) (volume : ℝ) : 
  base_side = 1/3 → height = 1 → volume = (1/3) * (base_side^2) * height → volume = 1/27 := by
  sorry

#check pyramid_volume

end pyramid_volume_l891_89163


namespace grandparents_count_l891_89108

/-- Represents the amount of money each grandparent gave to John -/
def money_per_grandparent : ℕ := 50

/-- Represents the total amount of money John received -/
def total_money : ℕ := 100

/-- The number of grandparents who gave John money -/
def num_grandparents : ℕ := 2

/-- Theorem stating that the number of grandparents who gave John money is 2 -/
theorem grandparents_count :
  num_grandparents = 2 ∧ total_money = num_grandparents * money_per_grandparent :=
sorry

end grandparents_count_l891_89108


namespace product_325_67_base_7_units_digit_l891_89152

theorem product_325_67_base_7_units_digit : 
  (325 * 67) % 7 = 5 := by
sorry

end product_325_67_base_7_units_digit_l891_89152


namespace koi_fish_problem_l891_89195

theorem koi_fish_problem (num_koi : ℕ) (subtracted_num : ℕ) : 
  num_koi = 39 → 
  2 * num_koi - subtracted_num = 64 → 
  subtracted_num = 14 := by
  sorry

end koi_fish_problem_l891_89195


namespace sashas_work_portion_l891_89183

theorem sashas_work_portion (car1 car2 car3 : ℚ) 
  (h1 : car1 = 1 / 3)
  (h2 : car2 = 1 / 5)
  (h3 : car3 = 1 / 15) :
  (car1 + car2 + car3) / 3 = 1 / 5 := by
sorry

end sashas_work_portion_l891_89183


namespace game_results_l891_89164

/-- A game between two players A and B with specific winning conditions -/
structure Game where
  pA : ℝ  -- Probability of A winning a single game
  pB : ℝ  -- Probability of B winning a single game
  hpA : pA = 2/3
  hpB : pB = 1/3
  hprob : pA + pB = 1

/-- The number of games played when the match is decided -/
def num_games (g : Game) : ℕ → ℝ
  | 2 => g.pA^2 + g.pB^2
  | 3 => g.pB * g.pA^2 + g.pA * g.pB^2
  | 4 => g.pA * g.pB * g.pA^2 + g.pB * g.pA * g.pB^2
  | 5 => g.pB * g.pA * g.pB * g.pA + g.pA * g.pB * g.pA * g.pB
  | _ => 0

/-- The probability that B wins exactly one game and A wins the match -/
def prob_B_wins_one (g : Game) : ℝ :=
  g.pB * g.pA^2 + g.pA * g.pB * g.pA^2

/-- The expected number of games played -/
def expected_games (g : Game) : ℝ :=
  2 * (num_games g 2) + 3 * (num_games g 3) + 4 * (num_games g 4) + 5 * (num_games g 5)

theorem game_results (g : Game) :
  prob_B_wins_one g = 20/81 ∧
  num_games g 2 = 5/9 ∧
  num_games g 3 = 2/9 ∧
  num_games g 4 = 10/81 ∧
  num_games g 5 = 8/81 ∧
  expected_games g = 224/81 := by
  sorry

end game_results_l891_89164


namespace sqrt_product_division_problem_statement_l891_89132

theorem sqrt_product_division (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt a * Real.sqrt b / (1 / Real.sqrt c) = c → a * b = c :=
by sorry

theorem problem_statement : 
  Real.sqrt 2 * Real.sqrt 3 / (1 / Real.sqrt 6) = 6 :=
by sorry

end sqrt_product_division_problem_statement_l891_89132


namespace right_triangle_hypotenuse_difference_l891_89189

theorem right_triangle_hypotenuse_difference (longer_side shorter_side hypotenuse : ℝ) : 
  hypotenuse = 17 →
  shorter_side = longer_side - 7 →
  longer_side^2 + shorter_side^2 = hypotenuse^2 →
  hypotenuse - longer_side = 2 := by
  sorry

end right_triangle_hypotenuse_difference_l891_89189


namespace count_sevens_in_range_l891_89144

/-- Count of digit 7 appearances in integers from 1 to 1000 -/
def count_sevens : ℕ := sorry

/-- The range of integers we're considering -/
def range_start : ℕ := 1
def range_end : ℕ := 1000

theorem count_sevens_in_range : count_sevens = 300 := by sorry

end count_sevens_in_range_l891_89144


namespace ellipse_standard_equation_l891_89161

/-- The standard equation of an ellipse given its parametric form -/
theorem ellipse_standard_equation (x y α : ℝ) :
  (x = 5 * Real.cos α) ∧ (y = 3 * Real.sin α) →
  (x^2 / 25 + y^2 / 9 = 1) := by
  sorry

end ellipse_standard_equation_l891_89161


namespace correct_system_l891_89171

/-- Represents the money owned by person A -/
def money_A : ℝ := sorry

/-- Represents the money owned by person B -/
def money_B : ℝ := sorry

/-- Condition 1: If B gives half of his money to A, then A will have 50 units of money -/
axiom condition1 : money_A + (1/2 : ℝ) * money_B = 50

/-- Condition 2: If A gives two-thirds of his money to B, then B will have 50 units of money -/
axiom condition2 : (2/3 : ℝ) * money_A + money_B = 50

/-- The system of equations correctly represents the given conditions -/
theorem correct_system : 
  (money_A + (1/2 : ℝ) * money_B = 50) ∧ 
  ((2/3 : ℝ) * money_A + money_B = 50) := by sorry

end correct_system_l891_89171


namespace sin_neg_ten_thirds_pi_l891_89101

theorem sin_neg_ten_thirds_pi : Real.sin (-10/3 * Real.pi) = Real.sqrt 3 / 2 := by
  sorry

end sin_neg_ten_thirds_pi_l891_89101


namespace min_value_on_line_equality_condition_l891_89155

theorem min_value_on_line (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b - 1 = 0 → 4/(a + b) + 1/b ≥ 9 :=
by sorry

theorem equality_condition (a b : ℝ) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b - 1 = 0 ∧ 4/(a + b) + 1/b = 9 :=
by sorry

end min_value_on_line_equality_condition_l891_89155


namespace angle_value_l891_89177

theorem angle_value (a : ℝ) : 3 * a + 150 = 360 → a = 70 := by
  sorry

end angle_value_l891_89177


namespace solve_system_l891_89103

theorem solve_system (a b x y : ℝ) 
  (eq1 : a * x + b * y = 16)
  (eq2 : b * x - a * y = -12)
  (sol_x : x = 2)
  (sol_y : y = 4) : 
  a = 4 ∧ b = 2 := by
sorry

end solve_system_l891_89103


namespace polynomial_equality_implies_two_one_l891_89153

/-- 
Given two positive integers r and s with r > s, and two distinct non-constant polynomials P and Q 
with real coefficients such that P(x)^r - P(x)^s = Q(x)^r - Q(x)^s for all real x, 
prove that r = 2 and s = 1.
-/
theorem polynomial_equality_implies_two_one (r s : ℕ) (P Q : ℝ → ℝ) : 
  r > s → 
  s > 0 →
  (∀ x : ℝ, P x ≠ Q x) → 
  (∃ a b c d : ℝ, a ≠ 0 ∧ c ≠ 0 ∧ ∀ x : ℝ, P x = a * x + b ∧ Q x = c * x + d) →
  (∀ x : ℝ, (P x)^r - (P x)^s = (Q x)^r - (Q x)^s) →
  r = 2 ∧ s = 1 := by
  sorry

end polynomial_equality_implies_two_one_l891_89153


namespace four_noncoplanar_points_determine_four_planes_l891_89188

/-- A set of four points in three-dimensional space. -/
structure FourPoints where
  p1 : ℝ × ℝ × ℝ
  p2 : ℝ × ℝ × ℝ
  p3 : ℝ × ℝ × ℝ
  p4 : ℝ × ℝ × ℝ

/-- Predicate to check if four points are non-coplanar. -/
def NonCoplanar (points : FourPoints) : Prop := sorry

/-- The number of planes determined by a set of four points. -/
def NumPlanesDetermined (points : FourPoints) : ℕ := sorry

/-- Theorem stating that four non-coplanar points determine exactly four planes. -/
theorem four_noncoplanar_points_determine_four_planes (points : FourPoints) :
  NonCoplanar points → NumPlanesDetermined points = 4 := by sorry

end four_noncoplanar_points_determine_four_planes_l891_89188


namespace young_inequality_l891_89124

theorem young_inequality (x y α β : ℝ) 
  (hx : x > 0) (hy : y > 0) (hα : α > 0) (hβ : β > 0) (hsum : α + β = 1) :
  x^α * y^β ≤ α*x + β*y :=
sorry

end young_inequality_l891_89124


namespace cookie_tin_weight_is_9_l891_89173

/-- The weight of a tin of cookies in ounces -/
def cookie_tin_weight (chip_bag_weight : ℕ) (num_chip_bags : ℕ) (cookie_tin_multiplier : ℕ) (total_weight_pounds : ℕ) : ℕ :=
  let total_weight_ounces : ℕ := total_weight_pounds * 16
  let total_chip_weight : ℕ := chip_bag_weight * num_chip_bags
  let num_cookie_tins : ℕ := num_chip_bags * cookie_tin_multiplier
  let total_cookie_weight : ℕ := total_weight_ounces - total_chip_weight
  total_cookie_weight / num_cookie_tins

/-- Theorem stating that a tin of cookies weighs 9 ounces under the given conditions -/
theorem cookie_tin_weight_is_9 :
  cookie_tin_weight 20 6 4 21 = 9 := by
  sorry

end cookie_tin_weight_is_9_l891_89173


namespace wood_weight_calculation_l891_89187

/-- Given a square piece of wood with side length 4 inches weighing 20 ounces,
    calculate the weight of a second square piece with side length 7 inches. -/
theorem wood_weight_calculation (thickness : ℝ) (density : ℝ) :
  let side1 : ℝ := 4
  let weight1 : ℝ := 20
  let side2 : ℝ := 7
  let area1 : ℝ := side1 * side1
  let area2 : ℝ := side2 * side2
  let weight2 : ℝ := weight1 * (area2 / area1)
  weight2 = 61.25 := by sorry

end wood_weight_calculation_l891_89187


namespace only_negative_one_squared_is_negative_l891_89114

theorem only_negative_one_squared_is_negative :
  ((-1 : ℝ)^0 < 0 ∨ |-1| < 0 ∨ Real.sqrt 1 < 0 ∨ -(1^2) < 0) ∧
  ((-1 : ℝ)^0 ≥ 0 ∧ |-1| ≥ 0 ∧ Real.sqrt 1 ≥ 0) ∧
  (-(1^2) < 0) :=
by sorry

end only_negative_one_squared_is_negative_l891_89114


namespace quadratic_equation_roots_l891_89184

theorem quadratic_equation_roots (α β : ℝ) : 
  ((1 + β) / (2 + β) = -1 / α) ∧ 
  ((α * β^2 + 121) / (1 - α^2 * β) = 1) →
  (∃ a b c : ℝ, (a * α^2 + b * α + c = 0) ∧ 
               (a * β^2 + b * β + c = 0) ∧ 
               ((a = 1 ∧ b = 12 ∧ c = 10) ∨ 
                (a = 1 ∧ b = -10 ∧ c = -12))) := by
  sorry

end quadratic_equation_roots_l891_89184


namespace author_writing_speed_l891_89133

/-- Given an author who writes 25,000 words in 50 hours, prove that their average writing speed is 500 words per hour. -/
theorem author_writing_speed :
  let total_words : ℕ := 25000
  let total_hours : ℕ := 50
  let average_speed : ℕ := total_words / total_hours
  average_speed = 500 :=
by sorry

end author_writing_speed_l891_89133


namespace exchange_of_segments_is_structure_variation_l891_89170

-- Define the basic concepts
def ChromosomalVariation : Type := sorry
def NonHomologousChromosome : Type := sorry
def ChromosomeStructure : Type := sorry
def Translocation : Type := sorry

-- Define the exchange of partial segments between non-homologous chromosomes
def PartialSegmentExchange (c1 c2 : NonHomologousChromosome) : Translocation := sorry

-- Define what constitutes a variation in chromosome structure
def IsChromosomeStructureVariation (t : Translocation) : Prop := sorry

-- Theorem to prove
theorem exchange_of_segments_is_structure_variation 
  (c1 c2 : NonHomologousChromosome) : 
  IsChromosomeStructureVariation (PartialSegmentExchange c1 c2) := by
  sorry

end exchange_of_segments_is_structure_variation_l891_89170


namespace absolute_value_difference_l891_89181

theorem absolute_value_difference (a b : ℝ) : 
  (a < b → |a - b| = b - a) ∧ (a ≥ b → |a - b| = a - b) := by sorry

end absolute_value_difference_l891_89181


namespace fraction_problem_l891_89104

theorem fraction_problem (a b : ℚ) (h1 : a + b = 100) (h2 : b = 60) : 
  (3 / 10) * a = (1 / 5) * b := by
sorry

end fraction_problem_l891_89104


namespace fourth_root_of_81_l891_89191

theorem fourth_root_of_81 : Real.sqrt (Real.sqrt 81) = 3 := by
  sorry

end fourth_root_of_81_l891_89191


namespace point_inside_circle_l891_89143

/-- Given a line ax + by + 1 = 0 and a circle x² + y² = 1 that are separate,
    prove that the point P(a, b) is inside the circle. -/
theorem point_inside_circle (a b : ℝ) 
  (h_separate : (1 : ℝ) / Real.sqrt (a^2 + b^2) > 1) : 
  a^2 + b^2 < 1 := by
  sorry

end point_inside_circle_l891_89143


namespace quadratic_inequality_solution_l891_89180

/-- The quadratic function f(x) = -x^2 + bx - 7 is negative only for x < 2 or x > 6 -/
def quadratic_inequality (b : ℝ) : Prop :=
  ∀ x : ℝ, (-x^2 + b*x - 7 < 0) ↔ (x < 2 ∨ x > 6)

/-- Given the quadratic inequality condition, prove that b = 8 -/
theorem quadratic_inequality_solution :
  ∃ b : ℝ, quadratic_inequality b ∧ b = 8 :=
sorry

end quadratic_inequality_solution_l891_89180


namespace price_decrease_after_increase_l891_89128

theorem price_decrease_after_increase (original_price : ℝ) (original_price_pos : original_price > 0) :
  let increased_price := original_price * 1.3
  let decrease_factor := 1 - (1 / 1.3)
  increased_price * (1 - decrease_factor) = original_price :=
by sorry

end price_decrease_after_increase_l891_89128


namespace example_linear_equation_l891_89120

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and at least one of a or b is non-zero. --/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y = a * x + b * y - c

/-- The equation x + 4y = 6 is a linear equation in two variables. --/
theorem example_linear_equation :
  IsLinearEquationInTwoVariables (fun x y ↦ x + 4 * y - 6) := by
  sorry

end example_linear_equation_l891_89120


namespace diophantine_equation_only_trivial_solution_l891_89121

theorem diophantine_equation_only_trivial_solution (x y z : ℤ) :
  3 * x^2 + 7 * y^2 = z^4 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end diophantine_equation_only_trivial_solution_l891_89121


namespace square_sum_theorem_l891_89119

theorem square_sum_theorem (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 11) : 
  x^2 + y^2 = 2893/36 := by
sorry

end square_sum_theorem_l891_89119


namespace not_P_sufficient_for_not_q_l891_89148

-- Define the propositions P and q
def P (x : ℝ) : Prop := |5*x - 2| > 3
def q (x : ℝ) : Prop := 1 / (x^2 + 4*x - 5) > 0

-- State the theorem
theorem not_P_sufficient_for_not_q :
  (∀ x : ℝ, ¬(P x) → ¬(q x)) ∧
  ¬(∀ x : ℝ, ¬(q x) → ¬(P x)) :=
sorry

end not_P_sufficient_for_not_q_l891_89148


namespace product_of_integers_l891_89146

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 26)
  (diff_squares_eq : x^2 - y^2 = 52) :
  x * y = 168 := by
  sorry

end product_of_integers_l891_89146


namespace perpendicular_vectors_l891_89168

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

theorem perpendicular_vectors (t : ℝ) : 
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) → t = -2 := by
  sorry

end perpendicular_vectors_l891_89168


namespace angle4_is_70_l891_89156

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 : ℝ)

-- Define the conditions
axiom angle1_plus_angle2 : angle1 + angle2 = 180
axiom angle4_eq_angle5 : angle4 = angle5
axiom triangle_sum : angle1 + angle3 + angle5 = 180
axiom angle1_value : angle1 = 50
axiom angle3_value : angle3 = 60

-- Theorem to prove
theorem angle4_is_70 : angle4 = 70 := by
  sorry

end angle4_is_70_l891_89156


namespace solve_equation_l891_89100

theorem solve_equation (x y : ℝ) : y = 1 / (2 * x + 2) → y = 2 → x = -3/4 := by
  sorry

end solve_equation_l891_89100


namespace stating_club_officer_selection_count_l891_89166

/-- Represents the number of members in the club -/
def total_members : ℕ := 24

/-- Represents the number of boys in the club -/
def num_boys : ℕ := 12

/-- Represents the number of girls in the club -/
def num_girls : ℕ := 12

/-- Represents the number of offices to be filled -/
def num_offices : ℕ := 3

/-- 
Theorem stating that the number of ways to choose a president, vice-president, and secretary 
from a club of 24 members (12 boys and 12 girls) is 5808, given that the president and 
vice-president must be of the same gender, the secretary can be of any gender, and no one 
can hold more than one office.
-/
theorem club_officer_selection_count : 
  (num_boys * (num_boys - 1) + num_girls * (num_girls - 1)) * (total_members - 2) = 5808 := by
  sorry

end stating_club_officer_selection_count_l891_89166


namespace bookstore_sales_l891_89116

/-- Calculates the number of bookmarks sold given the number of books sold and the ratio of books to bookmarks. -/
def bookmarks_sold (books : ℕ) (book_ratio : ℕ) (bookmark_ratio : ℕ) : ℕ :=
  (books * bookmark_ratio) / book_ratio

/-- Theorem stating that given 72 books sold and a 9:2 ratio of books to bookmarks, 16 bookmarks were sold. -/
theorem bookstore_sales : bookmarks_sold 72 9 2 = 16 := by
  sorry

end bookstore_sales_l891_89116


namespace integral_of_polynomial_l891_89145

theorem integral_of_polynomial : ∫ (x : ℝ) in (0)..(2), (3*x^2 + 4*x^3) = 24 := by sorry

end integral_of_polynomial_l891_89145


namespace geometric_mean_problem_l891_89140

theorem geometric_mean_problem (k : ℝ) : (2 * k)^2 = k * (k + 3) → k = 1 := by
  sorry

end geometric_mean_problem_l891_89140


namespace june_found_17_eggs_l891_89174

/-- The total number of bird eggs June found -/
def total_eggs (tree1_nests tree1_eggs_per_nest tree2_eggs frontyard_eggs : ℕ) : ℕ :=
  tree1_nests * tree1_eggs_per_nest + tree2_eggs + frontyard_eggs

/-- Theorem stating that June found 17 eggs in total -/
theorem june_found_17_eggs : 
  total_eggs 2 5 3 4 = 17 := by
  sorry

end june_found_17_eggs_l891_89174


namespace divisibility_problem_l891_89102

theorem divisibility_problem (a b c : ℕ) 
  (ha : a > 1) 
  (hb : b > c) 
  (hc : c > 1) 
  (hdiv : (a * b * c + 1) % (a * b - b + 1) = 0) : 
  b % a = 0 := by
sorry

end divisibility_problem_l891_89102


namespace find_point_B_l891_89107

/-- Given vector a, point A, and a line y = 2x, find point B on the line such that AB is parallel to a -/
theorem find_point_B (a : ℝ × ℝ) (A : ℝ × ℝ) :
  a = (1, 1) →
  A = (-3, -1) →
  ∃ B : ℝ × ℝ,
    B.2 = 2 * B.1 ∧
    ∃ k : ℝ, k • a = (B.1 - A.1, B.2 - A.2) ∧
    B = (2, 4) := by
  sorry


end find_point_B_l891_89107


namespace correct_counting_error_l891_89157

/-- The error in cents to be subtracted when quarters are mistakenly counted as half dollars
    and nickels are mistakenly counted as dimes. -/
def counting_error (x y : ℕ) : ℕ := 25 * x + 5 * y

/-- The value of a quarter in cents. -/
def quarter_value : ℕ := 25

/-- The value of a half dollar in cents. -/
def half_dollar_value : ℕ := 50

/-- The value of a nickel in cents. -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents. -/
def dime_value : ℕ := 10

theorem correct_counting_error (x y : ℕ) :
  counting_error x y = (half_dollar_value - quarter_value) * x + (dime_value - nickel_value) * y :=
by sorry

end correct_counting_error_l891_89157


namespace smallest_n_correct_l891_89160

/-- The smallest positive integer n for which (x^3 - 1/x^2)^n contains a non-zero constant term -/
def smallest_n : ℕ := 5

/-- Predicate to check if (x^3 - 1/x^2)^n has a non-zero constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ (k : ℕ), k ≠ 0 ∧ (3 * n = 5 * k)

theorem smallest_n_correct :
  (has_constant_term smallest_n) ∧
  (∀ m : ℕ, m < smallest_n → ¬(has_constant_term m)) :=
by sorry

#check smallest_n_correct

end smallest_n_correct_l891_89160


namespace at_least_one_side_not_exceeding_double_l891_89138

-- Define a structure for a parallelogram
structure Parallelogram :=
  (side1 : ℝ)
  (side2 : ℝ)
  (area : ℝ)

-- Define the problem setup
def parallelogram_inscriptions (P1 P2 P3 : Parallelogram) : Prop :=
  -- P2 is inscribed in P1
  P2.area < P1.area ∧
  -- P3 is inscribed in P2
  P3.area < P2.area ∧
  -- The sides of P3 are parallel to the sides of P1
  (P3.side1 < P1.side1 ∧ P3.side2 < P1.side2)

-- Theorem statement
theorem at_least_one_side_not_exceeding_double :
  ∀ (P1 P2 P3 : Parallelogram),
  parallelogram_inscriptions P1 P2 P3 →
  (P1.side1 ≤ 2 * P3.side1 ∨ P1.side2 ≤ 2 * P3.side2) :=
sorry

end at_least_one_side_not_exceeding_double_l891_89138


namespace student_arrangements_l891_89115

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def adjacent_arrangement (n : ℕ) : ℕ := 2 * factorial (n - 1)

def non_adjacent_arrangement (n : ℕ) : ℕ := factorial (n - 2) * (n * (n - 1))

def special_arrangement (n : ℕ) : ℕ := 
  factorial n - 3 * factorial (n - 1) + 2 * factorial (n - 2)

theorem student_arrangements :
  adjacent_arrangement 5 = 48 ∧
  non_adjacent_arrangement 5 = 72 ∧
  special_arrangement 5 = 60 := by
  sorry

#eval adjacent_arrangement 5
#eval non_adjacent_arrangement 5
#eval special_arrangement 5

end student_arrangements_l891_89115


namespace estimate_households_with_three_plus_houses_l891_89172

/-- Estimate the number of households owning 3 or more houses -/
theorem estimate_households_with_three_plus_houses
  (total_households : ℕ)
  (ordinary_households : ℕ)
  (high_income_households : ℕ)
  (sample_ordinary : ℕ)
  (sample_high_income : ℕ)
  (sample_ordinary_with_three_plus : ℕ)
  (sample_high_income_with_three_plus : ℕ)
  (h1 : total_households = 100000)
  (h2 : ordinary_households = 99000)
  (h3 : high_income_households = 1000)
  (h4 : sample_ordinary = 990)
  (h5 : sample_high_income = 100)
  (h6 : sample_ordinary_with_three_plus = 50)
  (h7 : sample_high_income_with_three_plus = 70)
  (h8 : total_households = ordinary_households + high_income_households) :
  ⌊(sample_ordinary_with_three_plus : ℚ) / sample_ordinary * ordinary_households +
   (sample_high_income_with_three_plus : ℚ) / sample_high_income * high_income_households⌋ = 5700 :=
by sorry


end estimate_households_with_three_plus_houses_l891_89172


namespace estate_distribution_theorem_l891_89192

/-- Represents the estate distribution problem -/
structure EstateProblem where
  num_beneficiaries : Nat
  min_ratio : Real
  known_amount : Real

/-- Calculates the smallest possible range between the highest and lowest amounts -/
def smallest_range (problem : EstateProblem) : Real :=
  sorry

/-- The theorem stating the smallest possible range for the given problem -/
theorem estate_distribution_theorem (problem : EstateProblem) 
  (h1 : problem.num_beneficiaries = 8)
  (h2 : problem.min_ratio = 1.4)
  (h3 : problem.known_amount = 80000) :
  smallest_range problem = 72412 := by
  sorry

end estate_distribution_theorem_l891_89192


namespace fixed_deposit_equation_l891_89136

theorem fixed_deposit_equation (x : ℝ) : 
  (∀ (interest_rate deposit_tax_rate final_amount : ℝ),
    interest_rate = 0.0198 →
    deposit_tax_rate = 0.20 →
    final_amount = 1300 →
    x + interest_rate * x * (1 - deposit_tax_rate) = final_amount) :=
by sorry

end fixed_deposit_equation_l891_89136


namespace second_player_wins_alice_wins_l891_89154

/-- Represents the frequency of each letter in the string -/
def LetterFrequency := Char → Nat

/-- The game state -/
structure GameState where
  frequencies : LetterFrequency
  playerTurn : Bool -- true for first player, false for second player

/-- Checks if all frequencies are even -/
def allEven (freq : LetterFrequency) : Prop :=
  ∀ c, Even (freq c)

/-- Represents a valid move in the game -/
inductive Move where
  | erase (c : Char) (n : Nat)

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.erase c n =>
      { frequencies := λ x => if x = c then state.frequencies x - n else state.frequencies x,
        playerTurn := ¬state.playerTurn }

/-- Checks if the game is over (all frequencies are zero) -/
def isGameOver (state : GameState) : Prop :=
  ∀ c, state.frequencies c = 0

/-- The winning strategy for the second player -/
def secondPlayerStrategy (state : GameState) : Move :=
  sorry -- Implementation not required for the statement

/-- The main theorem stating that the second player can always win -/
theorem second_player_wins (initialState : GameState) :
  ¬initialState.playerTurn →
  ∃ (strategy : GameState → Move),
    ∀ (moves : List Move),
      let finalState := (moves.foldl applyMove initialState)
      (isGameOver finalState ∧ ¬finalState.playerTurn) ∨
      (¬isGameOver finalState ∧ allEven finalState.frequencies) :=
  sorry

/-- The specific game instance from the problem -/
def initialGameState : GameState :=
  { frequencies := λ c =>
      if c = 'А' then 3
      else if c = 'О' then 3
      else if c = 'Д' then 2
      else if c = 'Я' then 2
      else if c ∈ ['Г', 'Р', 'С', 'К', 'У', 'Т', 'Н', 'Л', 'И', 'М', 'П'] then 1
      else 0,
    playerTurn := false }

/-- Theorem specific to the given problem instance -/
theorem alice_wins : 
  ∃ (strategy : GameState → Move),
    ∀ (moves : List Move),
      let finalState := (moves.foldl applyMove initialGameState)
      (isGameOver finalState ∧ ¬finalState.playerTurn) ∨
      (¬isGameOver finalState ∧ allEven finalState.frequencies) :=
  sorry

end second_player_wins_alice_wins_l891_89154


namespace third_term_value_l891_89112

def S (n : ℕ) : ℤ := 2 * n^2 - 1

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem third_term_value : a 3 = 10 := by
  sorry

end third_term_value_l891_89112


namespace f_3_equals_18_l891_89179

def f : ℕ → ℕ
  | 0     => 3
  | (n+1) => (n+1) * f n

theorem f_3_equals_18 : f 3 = 18 := by
  sorry

end f_3_equals_18_l891_89179


namespace customer_payment_percentage_l891_89150

theorem customer_payment_percentage (savings_percentage : ℝ) (payment_percentage : ℝ) :
  savings_percentage = 14.5 →
  payment_percentage = 100 - savings_percentage →
  payment_percentage = 85.5 :=
by sorry

end customer_payment_percentage_l891_89150


namespace necessary_condition_equality_l891_89151

theorem necessary_condition_equality (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end necessary_condition_equality_l891_89151


namespace pressure_change_l891_89123

/-- Given a relationship between pressure (P), area (A), and velocity (V),
    prove that doubling the area and increasing velocity from 20 to 30
    results in a specific pressure change. -/
theorem pressure_change (k : ℝ) :
  (∃ (P₀ A₀ V₀ : ℝ), P₀ = k * A₀ * V₀^2 ∧ P₀ = 0.5 ∧ A₀ = 1 ∧ V₀ = 20) →
  (∃ (P₁ A₁ V₁ : ℝ), P₁ = k * A₁ * V₁^2 ∧ A₁ = 2 ∧ V₁ = 30 ∧ P₁ = 2.25) :=
by sorry

end pressure_change_l891_89123


namespace total_filets_meeting_limit_l891_89109

/- Define fish species -/
inductive Species
| Bluefish
| Yellowtail
| RedSnapper

/- Define a structure for a fish -/
structure Fish where
  species : Species
  length : Nat

/- Define the minimum size limits -/
def minSizeLimit (s : Species) : Nat :=
  match s with
  | Species.Bluefish => 7
  | Species.Yellowtail => 6
  | Species.RedSnapper => 8

/- Define a function to check if a fish meets the size limit -/
def meetsLimit (f : Fish) : Bool :=
  f.length ≥ minSizeLimit f.species

/- Define the list of all fish caught -/
def allFish : List Fish := [
  {species := Species.Bluefish, length := 5},
  {species := Species.Bluefish, length := 9},
  {species := Species.Yellowtail, length := 9},
  {species := Species.Yellowtail, length := 9},
  {species := Species.RedSnapper, length := 11},
  {species := Species.Bluefish, length := 6},
  {species := Species.Yellowtail, length := 6},
  {species := Species.Yellowtail, length := 10},
  {species := Species.RedSnapper, length := 4},
  {species := Species.Bluefish, length := 8},
  {species := Species.RedSnapper, length := 3},
  {species := Species.Yellowtail, length := 7},
  {species := Species.Yellowtail, length := 12},
  {species := Species.Bluefish, length := 12},
  {species := Species.Bluefish, length := 12}
]

/- Define the number of filets per fish -/
def filetsPerFish : Nat := 2

/- Theorem: The total number of filets from fish meeting size limits is 22 -/
theorem total_filets_meeting_limit : 
  (allFish.filter meetsLimit).length * filetsPerFish = 22 := by
  sorry

end total_filets_meeting_limit_l891_89109


namespace double_thrice_one_is_eight_l891_89105

def double (n : ℕ) : ℕ := 2 * n

def iterate_double (n : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => n
  | k + 1 => iterate_double (double n) k

theorem double_thrice_one_is_eight :
  iterate_double 1 3 = 8 := by sorry

end double_thrice_one_is_eight_l891_89105


namespace bottle_height_l891_89149

/-- Represents a bottle composed of two cylinders -/
structure Bottle where
  r1 : ℝ  -- radius of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h_right : ℝ  -- water height when right side up
  h_upside : ℝ  -- water height when upside down

/-- The total height of the bottle -/
def total_height (b : Bottle) : ℝ :=
  29

/-- Theorem stating that the total height of the bottle is 29 cm -/
theorem bottle_height (b : Bottle) 
  (h_r1 : b.r1 = 1) 
  (h_r2 : b.r2 = 3) 
  (h_right : b.h_right = 20) 
  (h_upside : b.h_upside = 28) : 
  total_height b = 29 := by
  sorry

end bottle_height_l891_89149


namespace power_of_three_expression_l891_89113

theorem power_of_three_expression : ∀ (a b c d e f g h : ℕ), 
  a = 0 ∧ b = 1 ∧ c = 2 ∧ d = 4 ∧ e = 8 ∧ f = 16 ∧ g = 32 ∧ h = 64 →
  3^a * 3^b / 3^c / 3^d / 3^e * 3^f * 3^g * 3^h = 3^99 := by
  sorry

end power_of_three_expression_l891_89113


namespace complement_of_union_l891_89190

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem complement_of_union : U \ (A ∪ B) = {4} := by sorry

end complement_of_union_l891_89190


namespace star_polygon_is_pyramid_net_l891_89193

/-- Represents a star-shaped polygon constructed from two concentric circles and an inscribed regular polygon -/
structure StarPolygon where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  n : ℕ  -- Number of sides of the inscribed regular polygon
  h : R > r  -- Condition that the larger circle's radius is greater than the smaller circle's radius

/-- Determines whether a star-shaped polygon is the net of a pyramid -/
def is_pyramid_net (s : StarPolygon) : Prop :=
  s.R > 2 * s.r

/-- Theorem stating the condition for a star-shaped polygon to be the net of a pyramid -/
theorem star_polygon_is_pyramid_net (s : StarPolygon) :
  is_pyramid_net s ↔ s.R > 2 * s.r :=
sorry

end star_polygon_is_pyramid_net_l891_89193
