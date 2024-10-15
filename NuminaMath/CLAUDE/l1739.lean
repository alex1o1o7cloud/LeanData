import Mathlib

namespace NUMINAMATH_CALUDE_score_order_l1739_173929

/-- Represents the scores of contestants in a math competition. -/
structure Scores where
  alice : ℕ
  brian : ℕ
  cindy : ℕ
  donna : ℕ

/-- Conditions for the math competition scores. -/
def valid_scores (s : Scores) : Prop :=
  -- Brian + Donna = Alice + Cindy
  s.brian + s.donna = s.alice + s.cindy ∧
  -- If Brian and Cindy were swapped, Alice + Cindy > Brian + Donna + 10
  s.alice + s.brian > s.cindy + s.donna + 10 ∧
  -- Donna > Brian + Cindy + 20
  s.donna > s.brian + s.cindy + 20 ∧
  -- Total score is 200
  s.alice + s.brian + s.cindy + s.donna = 200

/-- The theorem to prove -/
theorem score_order (s : Scores) (h : valid_scores s) :
  s.donna > s.alice ∧ s.alice > s.brian ∧ s.brian > s.cindy := by
  sorry

end NUMINAMATH_CALUDE_score_order_l1739_173929


namespace NUMINAMATH_CALUDE_derivative_of_f_l1739_173986

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp (2*x)

theorem derivative_of_f :
  deriv f = λ x => Real.exp (2*x) * (2*x + 2*x^2) := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l1739_173986


namespace NUMINAMATH_CALUDE_simplify_expression_l1739_173928

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x * y - 4 ≠ 0) :
  (x^2 - 4 / y) / (y^2 - 4 / x) = x / y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1739_173928


namespace NUMINAMATH_CALUDE_giorgios_class_size_l1739_173971

theorem giorgios_class_size (cookies_per_student : ℕ) 
  (oatmeal_raisin_percentage : ℚ) (oatmeal_raisin_cookies : ℕ) :
  cookies_per_student = 2 →
  oatmeal_raisin_percentage = 1/10 →
  oatmeal_raisin_cookies = 8 →
  ∃ (total_students : ℕ), 
    total_students = 40 ∧
    (oatmeal_raisin_cookies : ℚ) / cookies_per_student = oatmeal_raisin_percentage * total_students :=
by
  sorry

#check giorgios_class_size

end NUMINAMATH_CALUDE_giorgios_class_size_l1739_173971


namespace NUMINAMATH_CALUDE_cubic_polynomial_relation_l1739_173953

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 7

theorem cubic_polynomial_relation (h : ℝ → ℝ) 
  (h_cubic : ∃ a b c d : ℝ, ∀ x, h x = a*x^3 + b*x^2 + c*x + d)
  (h_zero : h 0 = 7)
  (h_roots : ∀ r : ℝ, f r = 0 → ∃ s : ℝ, h s = 0 ∧ s = r^3) :
  h (-8) = -1813 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_relation_l1739_173953


namespace NUMINAMATH_CALUDE_tuesday_kids_count_l1739_173988

def monday_kids : ℕ := 12
def total_kids : ℕ := 19

theorem tuesday_kids_count : total_kids - monday_kids = 7 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_kids_count_l1739_173988


namespace NUMINAMATH_CALUDE_total_fruit_mass_is_7425_l1739_173926

/-- The number of apple trees in the orchard -/
def num_apple_trees : ℕ := 30

/-- The mass of apples produced by each apple tree (in kg) -/
def apple_yield_per_tree : ℕ := 150

/-- The number of peach trees in the orchard -/
def num_peach_trees : ℕ := 45

/-- The average mass of peaches produced by each peach tree (in kg) -/
def peach_yield_per_tree : ℕ := 65

/-- The total mass of fruit harvested in the orchard (in kg) -/
def total_fruit_mass : ℕ :=
  num_apple_trees * apple_yield_per_tree + num_peach_trees * peach_yield_per_tree

theorem total_fruit_mass_is_7425 : total_fruit_mass = 7425 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_mass_is_7425_l1739_173926


namespace NUMINAMATH_CALUDE_brothers_total_goals_l1739_173909

/-- The total number of goals scored by Louie and his brother -/
def total_goals (louie_last_match : ℕ) (louie_previous : ℕ) (brother_seasons : ℕ) (games_per_season : ℕ) : ℕ :=
  let louie_total := louie_last_match + louie_previous
  let brother_per_game := 2 * louie_last_match
  let brother_total := brother_seasons * games_per_season * brother_per_game
  louie_total + brother_total

/-- Theorem stating the total number of goals scored by the brothers -/
theorem brothers_total_goals :
  total_goals 4 40 3 50 = 1244 := by
  sorry

end NUMINAMATH_CALUDE_brothers_total_goals_l1739_173909


namespace NUMINAMATH_CALUDE_product_of_numbers_l1739_173975

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 48) : x * y = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1739_173975


namespace NUMINAMATH_CALUDE_three_digit_not_multiple_of_6_or_8_count_l1739_173935

/-- The count of three-digit numbers -/
def three_digit_count : ℕ := 900

/-- The count of three-digit numbers that are multiples of 6 -/
def multiples_of_6_count : ℕ := 150

/-- The count of three-digit numbers that are multiples of 8 -/
def multiples_of_8_count : ℕ := 112

/-- The count of three-digit numbers that are multiples of both 6 and 8 (i.e., multiples of 24) -/
def multiples_of_24_count : ℕ := 37

/-- Theorem: The count of three-digit numbers that are not multiples of 6 or 8 is 675 -/
theorem three_digit_not_multiple_of_6_or_8_count : 
  three_digit_count - (multiples_of_6_count + multiples_of_8_count - multiples_of_24_count) = 675 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_not_multiple_of_6_or_8_count_l1739_173935


namespace NUMINAMATH_CALUDE_outfits_count_l1739_173996

/-- The number of different outfits that can be made with a given number of shirts, pants, and ties. -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) : ℕ :=
  shirts * pants * (ties + 1)

/-- Theorem stating that the number of outfits with 7 shirts, 5 pants, and 4 ties (plus the option of no tie) is 175. -/
theorem outfits_count : number_of_outfits 7 5 4 = 175 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l1739_173996


namespace NUMINAMATH_CALUDE_inequality_range_l1739_173972

theorem inequality_range (a : ℝ) : 
  (∀ x > 0, 2 * x + 1 / x - a > 0) → a < 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1739_173972


namespace NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l1739_173998

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem f_monotonicity_and_intersection (a : ℝ) :
  (∀ x : ℝ, a ≥ 1/3 → Monotone (f a)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = a + 1 ∧ f a x = y ∧ f' a x * x = y) ∧
  (∃ x y : ℝ, x = -1 ∧ y = -a - 1 ∧ f a x = y ∧ f' a x * x = y) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l1739_173998


namespace NUMINAMATH_CALUDE_garden_remaining_area_l1739_173941

/-- A rectangular garden plot with a shed in one corner -/
structure GardenPlot where
  length : ℝ
  width : ℝ
  shedSide : ℝ

/-- Calculate the remaining area of a garden plot available for planting -/
def remainingArea (garden : GardenPlot) : ℝ :=
  garden.length * garden.width - garden.shedSide * garden.shedSide

/-- Theorem: The remaining area of a 20ft by 18ft garden plot with a 4ft by 4ft shed is 344 sq ft -/
theorem garden_remaining_area :
  let garden : GardenPlot := { length := 20, width := 18, shedSide := 4 }
  remainingArea garden = 344 := by sorry

end NUMINAMATH_CALUDE_garden_remaining_area_l1739_173941


namespace NUMINAMATH_CALUDE_incorrect_proposition_statement_l1739_173948

theorem incorrect_proposition_statement : 
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proposition_statement_l1739_173948


namespace NUMINAMATH_CALUDE_samantha_birth_year_proof_l1739_173924

/-- The year of the first AMC 8 -/
def first_amc8_year : ℕ := 1983

/-- The number of AMC 8 contests Samantha has taken -/
def samantha_amc8_count : ℕ := 9

/-- Samantha's age when she took her last AMC 8 -/
def samantha_age : ℕ := 13

/-- The year Samantha was born -/
def samantha_birth_year : ℕ := 1978

theorem samantha_birth_year_proof :
  samantha_birth_year = first_amc8_year + samantha_amc8_count - 1 - samantha_age :=
by sorry

end NUMINAMATH_CALUDE_samantha_birth_year_proof_l1739_173924


namespace NUMINAMATH_CALUDE_complex_sum_equals_minus_ten_i_l1739_173987

theorem complex_sum_equals_minus_ten_i :
  (5 - 5 * Complex.I) + (-2 - Complex.I) - (3 + 4 * Complex.I) = -10 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equals_minus_ten_i_l1739_173987


namespace NUMINAMATH_CALUDE_solve_equation_for_x_l1739_173955

theorem solve_equation_for_x (x y : ℤ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : x + y + x * y = 80) : 
  x = 26 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_for_x_l1739_173955


namespace NUMINAMATH_CALUDE_jerry_shower_limit_l1739_173946

/-- Calculates the number of full showers Jerry can take in July --/
def showers_in_july (total_water : ℕ) (drinking_cooking : ℕ) (shower_water : ℕ)
  (pool_length : ℕ) (pool_width : ℕ) (pool_depth : ℕ)
  (odd_day_leakage : ℕ) (even_day_leakage : ℕ) (evaporation_rate : ℕ)
  (odd_days : ℕ) (even_days : ℕ) : ℕ :=
  let pool_volume := pool_length * pool_width * pool_depth
  let total_leakage := odd_day_leakage * odd_days + even_day_leakage * even_days
  let total_evaporation := evaporation_rate * (odd_days + even_days)
  let pool_water_usage := pool_volume + total_leakage + total_evaporation
  let remaining_water := total_water - drinking_cooking - pool_water_usage
  remaining_water / shower_water

/-- Theorem stating that Jerry can take at most 1 full shower in July --/
theorem jerry_shower_limit :
  showers_in_july 1000 100 20 10 10 6 5 8 2 16 15 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_jerry_shower_limit_l1739_173946


namespace NUMINAMATH_CALUDE_two_times_zero_times_one_plus_one_l1739_173964

theorem two_times_zero_times_one_plus_one : 2 * 0 * 1 + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_two_times_zero_times_one_plus_one_l1739_173964


namespace NUMINAMATH_CALUDE_ann_age_l1739_173970

/-- The complex age relationship between Ann and Barbara --/
def age_relationship (a b : ℕ) : Prop :=
  ∃ y : ℕ, b = a / 2 + 2 * y ∧ y = a - b

/-- The theorem stating Ann's age given the conditions --/
theorem ann_age :
  ∀ a b : ℕ,
  age_relationship a b →
  a + b = 54 →
  a = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_ann_age_l1739_173970


namespace NUMINAMATH_CALUDE_avocados_bought_by_sister_georgie_guacamole_problem_l1739_173963

theorem avocados_bought_by_sister (avocados_per_serving : ℕ) (initial_avocados : ℕ) (servings_made : ℕ) : ℕ :=
  let total_avocados_needed := avocados_per_serving * servings_made
  let additional_avocados := total_avocados_needed - initial_avocados
  additional_avocados

theorem georgie_guacamole_problem :
  avocados_bought_by_sister 3 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_avocados_bought_by_sister_georgie_guacamole_problem_l1739_173963


namespace NUMINAMATH_CALUDE_sum_of_roots_l1739_173944

theorem sum_of_roots (p q : ℝ) (hp_neq_q : p ≠ q) : 
  (∃ x : ℝ, x^2 + p*x + q = 0 ∧ x^2 + q*x + p = 0) → p + q = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1739_173944


namespace NUMINAMATH_CALUDE_sum_first_6_primes_l1739_173973

-- Define a function that returns the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Define a function that sums the first n prime numbers
def sumFirstNPrimes (n : ℕ) : ℕ :=
  (List.range n).map (fun i => nthPrime (i + 1)) |>.sum

-- Theorem stating that the sum of the first 6 prime numbers is 41
theorem sum_first_6_primes : sumFirstNPrimes 6 = 41 := by sorry

end NUMINAMATH_CALUDE_sum_first_6_primes_l1739_173973


namespace NUMINAMATH_CALUDE_simplify_expression_l1739_173989

theorem simplify_expression (s : ℝ) : 180 * s - 88 * s = 92 * s := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1739_173989


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1739_173907

theorem sqrt_equation_solution (a b : ℕ) : 
  (Real.sqrt (8 + b / a) = 2 * Real.sqrt (b / a)) → (a = 63 ∧ b = 8) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1739_173907


namespace NUMINAMATH_CALUDE_fraction_irreducible_l1739_173904

theorem fraction_irreducible (n : ℕ+) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l1739_173904


namespace NUMINAMATH_CALUDE_milk_selling_price_l1739_173901

/-- Proves that the selling price of milk per litre is twice the cost price,
    given the mixing ratio of water to milk and the profit percentage. -/
theorem milk_selling_price 
  (x : ℝ) -- cost price of pure milk per litre
  (water_ratio : ℝ) -- ratio of water added to pure milk
  (milk_ratio : ℝ) -- ratio of pure milk
  (profit_percentage : ℝ) -- profit percentage
  (h1 : water_ratio = 2) -- 2 litres of water are added
  (h2 : milk_ratio = 6) -- to every 6 litres of pure milk
  (h3 : profit_percentage = 166.67) -- profit percentage is 166.67%
  : ∃ (selling_price : ℝ), selling_price = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_milk_selling_price_l1739_173901


namespace NUMINAMATH_CALUDE_paige_picture_upload_l1739_173921

/-- The number of pictures Paige uploaded to Facebook -/
def total_pictures : ℕ := 35

/-- The number of pictures in the first album -/
def first_album : ℕ := 14

/-- The number of additional albums -/
def additional_albums : ℕ := 3

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 7

/-- Theorem stating that the total number of pictures uploaded is correct -/
theorem paige_picture_upload :
  total_pictures = first_album + additional_albums * pictures_per_additional_album :=
by sorry

end NUMINAMATH_CALUDE_paige_picture_upload_l1739_173921


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1739_173902

/-- Calculates the simple interest rate given loan details and total interest --/
def calculate_interest_rate (loan1_principal loan1_time loan2_principal loan2_time total_interest : ℚ) : ℚ :=
  let total_interest_fraction := (loan1_principal * loan1_time + loan2_principal * loan2_time) / 100
  total_interest / total_interest_fraction

theorem interest_rate_calculation (loan1_principal loan1_time loan2_principal loan2_time total_interest : ℚ) 
  (h1 : loan1_principal = 5000)
  (h2 : loan1_time = 2)
  (h3 : loan2_principal = 3000)
  (h4 : loan2_time = 4)
  (h5 : total_interest = 2200) :
  calculate_interest_rate loan1_principal loan1_time loan2_principal loan2_time total_interest = 10 := by
  sorry

#eval calculate_interest_rate 5000 2 3000 4 2200

end NUMINAMATH_CALUDE_interest_rate_calculation_l1739_173902


namespace NUMINAMATH_CALUDE_no_special_two_digit_primes_l1739_173908

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_special_two_digit_primes :
  ∀ n : ℕ, 10 ≤ n → n < 100 →
    is_prime n ∧ is_prime (reverse_digits n) ∧ is_prime (digit_sum n) → False :=
sorry

end NUMINAMATH_CALUDE_no_special_two_digit_primes_l1739_173908


namespace NUMINAMATH_CALUDE_banquet_plates_l1739_173910

/-- The total number of plates served at a banquet -/
theorem banquet_plates (lobster_rolls : ℕ) (spicy_hot_noodles : ℕ) (seafood_noodles : ℕ)
  (h1 : lobster_rolls = 25)
  (h2 : spicy_hot_noodles = 14)
  (h3 : seafood_noodles = 16) :
  lobster_rolls + spicy_hot_noodles + seafood_noodles = 55 := by
  sorry

end NUMINAMATH_CALUDE_banquet_plates_l1739_173910


namespace NUMINAMATH_CALUDE_runner_daily_distance_l1739_173940

theorem runner_daily_distance (total_distance : ℝ) (total_weeks : ℝ) (daily_distance : ℝ) : 
  total_distance = 42 ∧ total_weeks = 3 ∧ daily_distance * (total_weeks * 7) = total_distance →
  daily_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_runner_daily_distance_l1739_173940


namespace NUMINAMATH_CALUDE_union_A_B_when_m_2_range_m_for_A_subset_B_l1739_173927

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) / (x - 3/2) < 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - (m + 1)*x + m ≤ 0}

-- Theorem for part (1)
theorem union_A_B_when_m_2 : A ∪ B 2 = {x | -2 < x ∧ x ≤ 2} := by sorry

-- Theorem for part (2)
theorem range_m_for_A_subset_B : 
  {m | A ⊆ B m} = {m | -2 < m ∧ m < 3/2} := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_2_range_m_for_A_subset_B_l1739_173927


namespace NUMINAMATH_CALUDE_thursday_tuesday_difference_l1739_173937

/-- The amount of money Max's mom gave him on Tuesday -/
def tuesday_amount : ℕ := 8

/-- The amount of money Max's mom gave him on Wednesday -/
def wednesday_amount : ℕ := 5 * tuesday_amount

/-- The amount of money Max's mom gave him on Thursday -/
def thursday_amount : ℕ := wednesday_amount + 9

/-- The theorem stating the difference between Thursday's and Tuesday's amounts -/
theorem thursday_tuesday_difference :
  thursday_amount - tuesday_amount = 41 := by sorry

end NUMINAMATH_CALUDE_thursday_tuesday_difference_l1739_173937


namespace NUMINAMATH_CALUDE_certain_number_proof_l1739_173959

theorem certain_number_proof (x : ℤ) (N : ℝ) 
  (h1 : N * (10 : ℝ)^(x : ℝ) < 220000)
  (h2 : ∀ y : ℤ, y > 5 → N * (10 : ℝ)^(y : ℝ) ≥ 220000) :
  N = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1739_173959


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1739_173915

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1739_173915


namespace NUMINAMATH_CALUDE_systematic_sampling_validity_l1739_173984

/-- Represents a set of student numbers -/
def StudentSet : Type := List Nat

/-- Checks if a list of natural numbers is arithmetic progression with common difference d -/
def isArithmeticProgression (l : List Nat) (d : Nat) : Prop :=
  l.zipWith (· - ·) (l.tail) = List.replicate (l.length - 1) d

/-- Checks if a set of student numbers is a valid systematic sample -/
def isValidSystematicSample (s : StudentSet) (totalStudents numSelected : Nat) : Prop :=
  s.length = numSelected ∧
  s.all (· ≤ totalStudents) ∧
  isArithmeticProgression s (totalStudents / numSelected)

theorem systematic_sampling_validity :
  let totalStudents : Nat := 50
  let numSelected : Nat := 5
  let sampleSet : StudentSet := [6, 16, 26, 36, 46]
  isValidSystematicSample sampleSet totalStudents numSelected := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_validity_l1739_173984


namespace NUMINAMATH_CALUDE_fraction_value_l1739_173918

theorem fraction_value (t k : ℚ) (f : ℚ) : 
  t = f * (k - 32) → t = 35 → k = 95 → f = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1739_173918


namespace NUMINAMATH_CALUDE_square_perimeter_count_l1739_173981

/-- The number of students on each side of the square arrangement -/
def side_length : ℕ := 10

/-- The number of students on the perimeter of a square arrangement -/
def perimeter_count (n : ℕ) : ℕ := 4 * n - 4

theorem square_perimeter_count :
  perimeter_count side_length = 36 := by
  sorry


end NUMINAMATH_CALUDE_square_perimeter_count_l1739_173981


namespace NUMINAMATH_CALUDE_band_members_count_l1739_173922

/-- Calculates the number of band members given the earnings per member, total earnings, and number of gigs. -/
def band_members (earnings_per_member : ℕ) (total_earnings : ℕ) (num_gigs : ℕ) : ℕ :=
  (total_earnings / num_gigs) / earnings_per_member

/-- Proves that the number of band members is 4 given the specified conditions. -/
theorem band_members_count : band_members 20 400 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_band_members_count_l1739_173922


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_parabola_l1739_173991

def f (x : ℝ) : ℝ := (x - 3)^2 - 8

theorem minimum_point_of_translated_parabola :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ x₀ = 3 ∧ f x₀ = -8 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_parabola_l1739_173991


namespace NUMINAMATH_CALUDE_square_measurement_error_l1739_173942

theorem square_measurement_error (actual_side : ℝ) (measured_side : ℝ) :
  measured_side^2 = actual_side^2 * (1 + 0.050625) →
  (measured_side - actual_side) / actual_side = 0.025 := by
sorry

end NUMINAMATH_CALUDE_square_measurement_error_l1739_173942


namespace NUMINAMATH_CALUDE_new_shoes_lifespan_l1739_173947

/-- Proves that the lifespan of new shoes is 2 years given the costs and conditions -/
theorem new_shoes_lifespan (repair_cost : ℝ) (repair_lifespan : ℝ) (new_cost : ℝ) (cost_increase_percentage : ℝ) :
  repair_cost = 14.50 →
  repair_lifespan = 1 →
  new_cost = 32.00 →
  cost_increase_percentage = 10.344827586206897 →
  let new_lifespan := new_cost / (repair_cost * (1 + cost_increase_percentage / 100))
  new_lifespan = 2 := by
  sorry

end NUMINAMATH_CALUDE_new_shoes_lifespan_l1739_173947


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_3_l1739_173979

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - a

/-- f is monotonically decreasing on (-1, 1) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x ∈ Set.Ioo (-1) 1, f_deriv a x ≤ 0

theorem monotone_decreasing_implies_a_geq_3 :
  ∀ a : ℝ, is_monotone_decreasing a → a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_3_l1739_173979


namespace NUMINAMATH_CALUDE_complex_real_condition_l1739_173951

theorem complex_real_condition (m : ℝ) : 
  (Complex.I : ℂ) * (1 + m * Complex.I) + (m^2 : ℂ) * (1 + m * Complex.I) ∈ Set.range (Complex.ofReal) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1739_173951


namespace NUMINAMATH_CALUDE_roses_planted_over_three_days_l1739_173949

/-- Represents the number of roses planted by each person on a given day -/
structure PlantingData where
  susan : ℕ
  maria : ℕ
  john : ℕ

/-- Calculates the total roses planted on a given day -/
def total_roses (data : PlantingData) : ℕ :=
  data.susan + data.maria + data.john

/-- Represents the planting data for all three days -/
structure ThreeDayPlanting where
  day1 : PlantingData
  day2 : PlantingData
  day3 : PlantingData

theorem roses_planted_over_three_days :
  ∀ (planting : ThreeDayPlanting),
    (planting.day1.susan + planting.day1.maria + planting.day1.john = 50) →
    (planting.day1.maria = 2 * planting.day1.susan) →
    (planting.day1.john = planting.day1.susan + 10) →
    (total_roses planting.day2 = total_roses planting.day1 + 20) →
    (planting.day2.susan * 5 = planting.day1.susan * 7) →
    (planting.day2.maria * 5 = planting.day1.maria * 7) →
    (planting.day2.john * 5 = planting.day1.john * 7) →
    (total_roses planting.day3 = 2 * total_roses planting.day1) →
    (planting.day3.susan = planting.day1.susan) →
    (planting.day3.maria = planting.day1.maria + (planting.day1.maria / 4)) →
    (planting.day3.john = planting.day1.john - (planting.day1.john / 10)) →
    (total_roses planting.day1 + total_roses planting.day2 + total_roses planting.day3 = 173) :=
by sorry

end NUMINAMATH_CALUDE_roses_planted_over_three_days_l1739_173949


namespace NUMINAMATH_CALUDE_age_difference_l1739_173919

/-- Given three people A, B, and C, where C is 12 years younger than A,
    prove that the total age of A and B is 12 years more than the total age of B and C. -/
theorem age_difference (A B C : ℕ) (h : C = A - 12) :
  (A + B) - (B + C) = 12 :=
sorry

end NUMINAMATH_CALUDE_age_difference_l1739_173919


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1739_173920

theorem concentric_circles_radii_difference (s L : ℝ) (h : s > 0) :
  (L^2 / s^2 = 9 / 4) → (L - s = 0.5 * s) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1739_173920


namespace NUMINAMATH_CALUDE_unique_solution_l1739_173985

/-- Calculates the cost per person based on the number of participants -/
def costPerPerson (n : ℕ) : ℕ :=
  if n ≤ 30 then 80
  else max 50 (80 - (n - 30))

/-- Calculates the total cost for a given number of participants -/
def totalCost (n : ℕ) : ℕ :=
  n * costPerPerson n

/-- States that there exists a unique number of employees that satisfies the problem conditions -/
theorem unique_solution : ∃! n : ℕ, n > 30 ∧ totalCost n = 2800 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1739_173985


namespace NUMINAMATH_CALUDE_leopards_count_l1739_173958

def zoo_problem (leopards : ℕ) : Prop :=
  let snakes := 100
  let arctic_foxes := 80
  let bee_eaters := 10 * leopards
  let cheetahs := snakes / 2
  let alligators := 2 * (arctic_foxes + leopards)
  let total_animals := 670
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = total_animals

theorem leopards_count : ∃ (l : ℕ), zoo_problem l ∧ l = 20 := by
  sorry

end NUMINAMATH_CALUDE_leopards_count_l1739_173958


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1739_173905

/-- Theorem: The quadratic equation x^2 - 6x + 9 = 0 has two equal real roots. -/
theorem quadratic_equal_roots :
  ∃ (x : ℝ), (x^2 - 6*x + 9 = 0) ∧ (∀ y : ℝ, y^2 - 6*y + 9 = 0 → y = x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1739_173905


namespace NUMINAMATH_CALUDE_existence_of_sequence_l1739_173906

/-- The number of positive divisors of n -/
def d (n : ℕ) : ℕ := sorry

/-- The smallest prime divisor of n -/
def s (n : ℕ) : ℕ := sorry

/-- The theorem stating the existence of a sequence satisfying the given conditions -/
theorem existence_of_sequence : ∃ (a : ℕ → ℕ), 
  (∀ k ∈ Finset.range 2022, a (k + 1) > a k + 1) ∧ 
  (∀ k ∈ Finset.range 2022, d (a (k + 1) - a k - 1) > 2023^k) ∧
  (∀ k ∈ Finset.range 2022, s (a (k + 1) - a k) > 2023^k) :=
sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l1739_173906


namespace NUMINAMATH_CALUDE_initial_walnut_trees_l1739_173990

/-- The number of walnut trees initially in the park -/
def initial_trees : ℕ := sorry

/-- The number of walnut trees planted today -/
def planted_trees : ℕ := 33

/-- The total number of walnut trees after planting -/
def final_trees : ℕ := 55

/-- Theorem stating that the initial number of walnut trees is 22 -/
theorem initial_walnut_trees : initial_trees = 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_walnut_trees_l1739_173990


namespace NUMINAMATH_CALUDE_fifth_term_is_five_l1739_173950

/-- An arithmetic sequence with specific conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Sum function
  h1 : S 6 = 3
  h2 : a 4 = 2

/-- The fifth term of the arithmetic sequence is 5 -/
theorem fifth_term_is_five (seq : ArithmeticSequence) : seq.a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_five_l1739_173950


namespace NUMINAMATH_CALUDE_number_difference_l1739_173967

theorem number_difference (a b : ℕ) (h1 : a + b = 56) (h2 : a < b) (h3 : a = 22) (h4 : b = 34) :
  b - a = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1739_173967


namespace NUMINAMATH_CALUDE_convex_pentagon_inner_lattice_point_l1739_173992

/-- A point in the 2D Cartesian plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A pentagon defined by five points -/
structure Pentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- Checks if a pentagon is convex -/
def isConvex (p : Pentagon) : Prop :=
  sorry

/-- Checks if a point is a lattice point -/
def isLatticePoint (p : Point) : Prop :=
  sorry

/-- Constructs the inner pentagon formed by the intersection of diagonals -/
def innerPentagon (p : Pentagon) : Pentagon :=
  sorry

/-- Checks if a point is inside or on the boundary of a pentagon -/
def isInOrOnPentagon (point : Point) (p : Pentagon) : Prop :=
  sorry

/-- The main theorem -/
theorem convex_pentagon_inner_lattice_point (p : Pentagon) :
  isConvex p →
  isLatticePoint p.A ∧ isLatticePoint p.B ∧ isLatticePoint p.C ∧ isLatticePoint p.D ∧ isLatticePoint p.E →
  ∃ (point : Point), isLatticePoint point ∧ isInOrOnPentagon point (innerPentagon p) :=
sorry

end NUMINAMATH_CALUDE_convex_pentagon_inner_lattice_point_l1739_173992


namespace NUMINAMATH_CALUDE_tom_gave_sixteen_balloons_l1739_173913

/-- The number of balloons Tom gave to Fred -/
def balloons_given (initial_balloons remaining_balloons : ℕ) : ℕ :=
  initial_balloons - remaining_balloons

/-- Theorem: Tom gave 16 balloons to Fred -/
theorem tom_gave_sixteen_balloons :
  let initial_balloons : ℕ := 30
  let remaining_balloons : ℕ := 14
  balloons_given initial_balloons remaining_balloons = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_gave_sixteen_balloons_l1739_173913


namespace NUMINAMATH_CALUDE_expression_evaluation_l1739_173960

theorem expression_evaluation (x y z : ℝ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5)
  (hd1 : x + 2 ≠ 0)
  (hd2 : y - 3 ≠ 0)
  (hd3 : z + 7 ≠ 0) :
  ((x + 3) / (x + 2)) * ((y - 2) / (y - 3)) * ((z + 9) / (z + 7)) = 72 / 35 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1739_173960


namespace NUMINAMATH_CALUDE_double_value_points_range_l1739_173994

/-- A point (k, 2k) is a double value point -/
def DoubleValuePoint (k : ℝ) : ℝ × ℝ := (k, 2 * k)

/-- The quadratic function -/
def QuadraticFunction (t s : ℝ) (x : ℝ) : ℝ := (t + 1) * x^2 + (t + 2) * x + s

theorem double_value_points_range (t s : ℝ) (h : t ≠ -1) :
  (∀ k₁ k₂ : ℝ, k₁ ≠ k₂ → 
    ∃ (p₁ p₂ : ℝ × ℝ), p₁ = DoubleValuePoint k₁ ∧ p₂ = DoubleValuePoint k₂ ∧
    QuadraticFunction t s (p₁.1) = p₁.2 ∧ QuadraticFunction t s (p₂.1) = p₂.2) →
  -1 < s ∧ s < 0 := by
  sorry

end NUMINAMATH_CALUDE_double_value_points_range_l1739_173994


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l1739_173931

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3  -- 60 degrees in radians
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  angle = Real.arccos (dot_product / (magnitude a * magnitude b)) →
  magnitude a = 2 →
  magnitude b = 5 →
  magnitude (2 • a - b) = Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l1739_173931


namespace NUMINAMATH_CALUDE_fraction_equality_l1739_173977

theorem fraction_equality (x y : ℝ) (h : (1/x + 1/y)/(1/x - 1/y) = 101) : 
  (x - y)/(x + y) = -1/5101 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1739_173977


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l1739_173969

theorem smallest_fraction_between (p q : ℕ+) : 
  (6 : ℚ) / 11 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 9 ∧ 
  (∀ (p' q' : ℕ+), (6 : ℚ) / 11 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 9 → q ≤ q') →
  q.val - p.val = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l1739_173969


namespace NUMINAMATH_CALUDE_initial_loss_percentage_l1739_173930

/-- Proves that for an article with a cost price of $400, if increasing the selling price
    by $100 results in a 5% gain, then the initial loss percentage is 20%. -/
theorem initial_loss_percentage 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (h1 : cost_price = 400)
  (h2 : selling_price + 100 = 1.05 * cost_price) :
  (cost_price - selling_price) / cost_price * 100 = 20 := by
  sorry

#check initial_loss_percentage

end NUMINAMATH_CALUDE_initial_loss_percentage_l1739_173930


namespace NUMINAMATH_CALUDE_function_not_in_third_quadrant_l1739_173914

theorem function_not_in_third_quadrant 
  (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  ∀ x : ℝ, x < 0 → a^x + b - 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_not_in_third_quadrant_l1739_173914


namespace NUMINAMATH_CALUDE_pet_store_puzzle_l1739_173917

theorem pet_store_puzzle (initial_birds initial_puppies initial_cats initial_spiders : ℕ)
  (final_total : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_cats = 5 →
  initial_spiders = 15 →
  final_total = 25 →
  ∃ (adopted_puppies : ℕ),
    adopted_puppies = initial_puppies - (
      initial_birds + initial_puppies + initial_cats + initial_spiders -
      (initial_birds / 2 + 7 + final_total)
    ) :=
by sorry

end NUMINAMATH_CALUDE_pet_store_puzzle_l1739_173917


namespace NUMINAMATH_CALUDE_parabola_symmetric_points_l1739_173943

/-- Prove that for a parabola y = ax^2 (a > 0) where the distance from focus to directrix is 1/4,
    and two points A(x₁, y₁) and B(x₂, y₂) on the parabola are symmetric about y = x + m,
    and x₁x₂ = -1/2, then m = 3/2 -/
theorem parabola_symmetric_points (a : ℝ) (x₁ y₁ x₂ y₂ m : ℝ) : 
  a > 0 →
  (1 / (4 * a) = 1 / 4) →
  y₁ = a * x₁^2 →
  y₂ = a * x₂^2 →
  y₁ + y₂ = x₁ + x₂ + 2 * m →
  x₁ * x₂ = -1 / 2 →
  m = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetric_points_l1739_173943


namespace NUMINAMATH_CALUDE_square_of_binomial_exclusion_l1739_173956

theorem square_of_binomial_exclusion (a b x m : ℝ) : 
  (∃ p q : ℝ, (x + a) * (x - a) = p^2 - q^2) ∧ 
  (∃ p q : ℝ, (-x - b) * (x - b) = -(p^2 - q^2)) ∧ 
  (∃ p q : ℝ, (b + m) * (m - b) = p^2 - q^2) ∧ 
  ¬(∃ p : ℝ, (a + b) * (-a - b) = p^2) :=
by sorry

end NUMINAMATH_CALUDE_square_of_binomial_exclusion_l1739_173956


namespace NUMINAMATH_CALUDE_odd_numbers_mean_median_impossibility_l1739_173912

theorem odd_numbers_mean_median_impossibility :
  ∀ (a b c d e f g : ℤ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g →
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧ Odd g →
    (a + b + c + d + e + f + g) / 7 ≠ d + 3 / 7 :=
by sorry

end NUMINAMATH_CALUDE_odd_numbers_mean_median_impossibility_l1739_173912


namespace NUMINAMATH_CALUDE_unique_triplet_l1739_173952

theorem unique_triplet :
  ∃! (a b c : ℕ), 1 < a ∧ a < b ∧ b < c ∧
  (c ∣ a * b + 1) ∧
  (b ∣ a * c + 1) ∧
  (a ∣ b * c + 1) ∧
  a = 2 ∧ b = 3 ∧ c = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_triplet_l1739_173952


namespace NUMINAMATH_CALUDE_julians_initial_debt_l1739_173957

/-- Given that Julian will owe Jenny 28 dollars if he borrows 8 dollars more,
    prove that Julian's initial debt to Jenny is 20 dollars. -/
theorem julians_initial_debt (current_debt additional_borrow total_after_borrow : ℕ) :
  additional_borrow = 8 →
  total_after_borrow = 28 →
  total_after_borrow = current_debt + additional_borrow →
  current_debt = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_julians_initial_debt_l1739_173957


namespace NUMINAMATH_CALUDE_smallest_first_term_of_arithmetic_sequence_l1739_173916

def arithmetic_sequence (c₁ : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => c₁
  | n+1 => arithmetic_sequence c₁ d n + d

def sum_of_terms (c₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * c₁ + (n - 1 : ℚ) * d) / 2

theorem smallest_first_term_of_arithmetic_sequence :
  ∃ (c₁ d : ℚ),
    (c₁ ≥ 1/3) ∧
    (∃ (S₃ S₇ : ℕ),
      sum_of_terms c₁ d 3 = S₃ ∧
      sum_of_terms c₁ d 7 = S₇) ∧
    (∀ (c₁' d' : ℚ),
      (c₁' ≥ 1/3) →
      (∃ (S₃' S₇' : ℕ),
        sum_of_terms c₁' d' 3 = S₃' ∧
        sum_of_terms c₁' d' 7 = S₇') →
      c₁' ≥ c₁) ∧
    c₁ = 5/14 :=
sorry

end NUMINAMATH_CALUDE_smallest_first_term_of_arithmetic_sequence_l1739_173916


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1739_173978

theorem smaller_number_proof (x y : ℝ) : 
  y = 2 * x - 3 → 
  x + y = 51 → 
  min x y = 18 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1739_173978


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1739_173995

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^6 + X^5 + 2*X^3 - X^2 + 3 = (X + 2) * (X - 1) * q + (-X + 5) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1739_173995


namespace NUMINAMATH_CALUDE_distance_AC_l1739_173936

theorem distance_AC (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt 27
  let BC := 2
  let angle_ABC := 150 * Real.pi / 180
  let AC := Real.sqrt ((AB^2 + BC^2) - 2 * AB * BC * Real.cos angle_ABC)
  AC = 7 := by sorry

end NUMINAMATH_CALUDE_distance_AC_l1739_173936


namespace NUMINAMATH_CALUDE_at_least_one_truth_teller_not_knight_l1739_173945

-- Define the possible types of people
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a person with a type and a statement about the other person
structure Person where
  type : PersonType
  statement : PersonType → Prop

-- Define what it means for a person to be telling the truth
def isTellingTruth (p : Person) (otherType : PersonType) : Prop :=
  match p.type with
  | PersonType.Knight => p.statement otherType
  | PersonType.Liar => ¬(p.statement otherType)
  | PersonType.Normal => True

-- Define the specific statements made by A and B
def statementA (typeB : PersonType) : Prop := typeB = PersonType.Knight
def statementB (typeA : PersonType) : Prop := typeA ≠ PersonType.Knight

-- Define A and B
def A : Person := { type := PersonType.Knight, statement := statementA }
def B : Person := { type := PersonType.Knight, statement := statementB }

-- The main theorem
theorem at_least_one_truth_teller_not_knight :
  ∃ p : Person, p ∈ [A, B] ∧ 
    (∃ otherType : PersonType, isTellingTruth p otherType) ∧ 
    p.type ≠ PersonType.Knight :=
sorry

end NUMINAMATH_CALUDE_at_least_one_truth_teller_not_knight_l1739_173945


namespace NUMINAMATH_CALUDE_train_passing_time_l1739_173997

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 150 ∧ 
  train_speed = 62 * (1000 / 3600) ∧ 
  man_speed = 8 * (1000 / 3600) →
  train_length / (train_speed - man_speed) = 10 := by
  sorry


end NUMINAMATH_CALUDE_train_passing_time_l1739_173997


namespace NUMINAMATH_CALUDE_alex_remaining_money_l1739_173900

theorem alex_remaining_money (weekly_income : ℝ) (tax_rate : ℝ) (water_bill : ℝ) 
  (tithe_rate : ℝ) (groceries : ℝ) (transportation : ℝ) :
  weekly_income = 900 →
  tax_rate = 0.15 →
  water_bill = 75 →
  tithe_rate = 0.20 →
  groceries = 150 →
  transportation = 50 →
  weekly_income - (tax_rate * weekly_income) - water_bill - (tithe_rate * weekly_income) - 
    groceries - transportation = 310 := by
  sorry

end NUMINAMATH_CALUDE_alex_remaining_money_l1739_173900


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l1739_173911

theorem factorial_ratio_equals_sixty_sevenths : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l1739_173911


namespace NUMINAMATH_CALUDE_two_rats_boring_theorem_l1739_173932

/-- The distance burrowed by the larger rat on day n -/
def larger_rat_distance (n : ℕ) : ℚ := 2^(n-1)

/-- The distance burrowed by the smaller rat on day n -/
def smaller_rat_distance (n : ℕ) : ℚ := (1/2)^(n-1)

/-- The total distance burrowed by both rats after n days -/
def total_distance (n : ℕ) : ℚ := 
  (Finset.range n).sum (λ i => larger_rat_distance (i+1) + smaller_rat_distance (i+1))

/-- The theorem stating that the total distance burrowed after 5 days is 32 15/16 -/
theorem two_rats_boring_theorem : total_distance 5 = 32 + 15/16 := by sorry

end NUMINAMATH_CALUDE_two_rats_boring_theorem_l1739_173932


namespace NUMINAMATH_CALUDE_z_is_real_z_is_complex_z_is_pure_imaginary_z_in_fourth_quadrant_l1739_173903

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 1) (m^2 - m - 2)

-- 1. z is a real number iff m = -1 or m = 2
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = -1 ∨ m = 2 := by sorry

-- 2. z is a complex number iff m ≠ -1 and m ≠ 2
theorem z_is_complex (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ -1 ∧ m ≠ 2 := by sorry

-- 3. z is a pure imaginary number iff m = 1
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 1 := by sorry

-- 4. z is in the fourth quadrant iff 1 < m < 2
theorem z_in_fourth_quadrant (m : ℝ) : 
  (z m).re > 0 ∧ (z m).im < 0 ↔ 1 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_z_is_real_z_is_complex_z_is_pure_imaginary_z_in_fourth_quadrant_l1739_173903


namespace NUMINAMATH_CALUDE_no_prime_roots_l1739_173934

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The quadratic equation x^2 - 108x + k = 0 -/
def quadraticEquation (x k : ℝ) : Prop := x^2 - 108*x + k = 0

/-- Both roots of the quadratic equation are prime numbers -/
def bothRootsPrime (k : ℝ) : Prop :=
  ∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ 
    (∀ x : ℝ, quadraticEquation x k ↔ x = p ∨ x = q)

/-- There are no values of k for which both roots of the quadratic equation are prime -/
theorem no_prime_roots : ¬∃ k : ℝ, bothRootsPrime k := by sorry

end NUMINAMATH_CALUDE_no_prime_roots_l1739_173934


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1739_173980

theorem quadratic_root_relation (p q : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x - y = 2 ∧ x = 2*y) → p = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1739_173980


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1739_173933

theorem fraction_equals_zero (x : ℝ) (h : x ≠ 0) :
  x = 3 → (2 * x - 6) / (5 * x) = 0 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1739_173933


namespace NUMINAMATH_CALUDE_choir_members_count_l1739_173923

theorem choir_members_count : ∃! n : ℕ, 
  200 < n ∧ n < 300 ∧ 
  (n + 4) % 10 = 0 ∧ 
  (n + 5) % 11 = 0 ∧ 
  n = 226 := by
sorry

end NUMINAMATH_CALUDE_choir_members_count_l1739_173923


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1739_173993

theorem triangle_angle_sum (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = 180) (h5 : A ≤ B) (h6 : A ≤ C) (h7 : C ≤ B) (h8 : 2 * B = 5 * A) : 
  ∃ (m n : ℝ), m = max B C ∧ n = min B C ∧ m + n = 175 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1739_173993


namespace NUMINAMATH_CALUDE_total_legs_of_bokyungs_animals_l1739_173999

-- Define the number of legs for puppies and chicks
def puppy_legs : ℕ := 4
def chick_legs : ℕ := 2

-- Define the number of puppies and chicks Bokyung has
def num_puppies : ℕ := 3
def num_chicks : ℕ := 7

-- Theorem to prove
theorem total_legs_of_bokyungs_animals : 
  num_puppies * puppy_legs + num_chicks * chick_legs = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_of_bokyungs_animals_l1739_173999


namespace NUMINAMATH_CALUDE_yuanxiao_sales_problem_l1739_173938

/-- Yuanxiao sales problem -/
theorem yuanxiao_sales_problem 
  (cost : ℝ) 
  (min_price : ℝ) 
  (base_sales : ℝ) 
  (base_price : ℝ) 
  (price_sensitivity : ℝ) 
  (max_price : ℝ) 
  (min_profit : ℝ)
  (h1 : cost = 20)
  (h2 : min_price = 25)
  (h3 : base_sales = 250)
  (h4 : base_price = 25)
  (h5 : price_sensitivity = 10)
  (h6 : max_price = 38)
  (h7 : min_profit = 2000) :
  let sales_volume (x : ℝ) := -price_sensitivity * x + (base_sales + price_sensitivity * base_price)
  let profit (x : ℝ) := (x - cost) * (sales_volume x)
  ∃ (optimal_price : ℝ) (max_profit : ℝ) (min_sales : ℝ),
    (∀ x, sales_volume x = -10 * x + 500) ∧
    (optimal_price = 35 ∧ max_profit = 2250 ∧ 
     ∀ x, x ≥ min_price → profit x ≤ max_profit) ∧
    (min_sales = 120 ∧
     ∀ x, min_price ≤ x ∧ x ≤ max_price → 
     profit x ≥ min_profit → sales_volume x ≥ min_sales) := by
  sorry

end NUMINAMATH_CALUDE_yuanxiao_sales_problem_l1739_173938


namespace NUMINAMATH_CALUDE_bread_distribution_l1739_173939

theorem bread_distribution (total_loaves : ℕ) (num_people : ℕ) : 
  total_loaves = 100 →
  num_people = 5 →
  ∃ (a d : ℚ), 
    (∀ i : ℕ, i ≤ 5 → a + (i - 1) * d ≥ 0) ∧
    (a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = total_loaves) ∧
    ((a + 2*d) + (a + 3*d) + (a + 4*d) = 3 * (a + (a + d))) →
  (a + 4*d ≤ 30) :=
by sorry

end NUMINAMATH_CALUDE_bread_distribution_l1739_173939


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1739_173954

theorem inequality_equivalence (x : ℝ) : 3 - 2 / (3 * x + 2) < 5 ↔ x > -2/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1739_173954


namespace NUMINAMATH_CALUDE_solve_beef_problem_l1739_173974

def beef_problem (pounds_per_pack : ℝ) (price_per_pound : ℝ) (total_paid : ℝ) : Prop :=
  let price_per_pack := pounds_per_pack * price_per_pound
  let num_packs := total_paid / price_per_pack
  num_packs = 5

theorem solve_beef_problem :
  beef_problem 4 5.50 110 := by
  sorry

end NUMINAMATH_CALUDE_solve_beef_problem_l1739_173974


namespace NUMINAMATH_CALUDE_discount_calculation_l1739_173961

theorem discount_calculation (original_price discount_percentage : ℝ) 
  (h1 : original_price = 80)
  (h2 : discount_percentage = 15) : 
  original_price * (1 - discount_percentage / 100) = 68 := by
sorry

end NUMINAMATH_CALUDE_discount_calculation_l1739_173961


namespace NUMINAMATH_CALUDE_player_matches_l1739_173925

/-- The number of matches played by a player -/
def num_matches : ℕ := sorry

/-- The current average runs per match -/
def current_average : ℚ := 32

/-- The runs to be scored in the next match -/
def next_match_runs : ℕ := 98

/-- The increase in average after the next match -/
def average_increase : ℚ := 6

theorem player_matches :
  (current_average * num_matches + next_match_runs) / (num_matches + 1) = current_average + average_increase →
  num_matches = 10 := by sorry

end NUMINAMATH_CALUDE_player_matches_l1739_173925


namespace NUMINAMATH_CALUDE_root_two_implies_a_and_other_root_always_real_roots_l1739_173982

-- Define the equation
def equation (x a : ℝ) : Prop := x^2 + a*x + a - 1 = 0

-- Theorem 1: If 2 is a root, then a = -1 and the other root is -1
theorem root_two_implies_a_and_other_root (a : ℝ) :
  equation 2 a → a = -1 ∧ equation (-1) a := by sorry

-- Theorem 2: The equation always has real roots
theorem always_real_roots (a : ℝ) :
  ∃ x : ℝ, equation x a := by sorry

end NUMINAMATH_CALUDE_root_two_implies_a_and_other_root_always_real_roots_l1739_173982


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1739_173966

theorem trigonometric_inequality (a b A B : ℝ) :
  (∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ) ≥ 0) →
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1739_173966


namespace NUMINAMATH_CALUDE_alex_savings_l1739_173976

def car_cost : ℝ := 14600
def trip_charge : ℝ := 1.5
def grocery_percentage : ℝ := 0.05
def num_trips : ℕ := 40
def grocery_value : ℝ := 800

theorem alex_savings (initial_savings : ℝ) : 
  initial_savings + 
  (num_trips : ℝ) * trip_charge + 
  grocery_percentage * grocery_value = 
  car_cost :=
by sorry

end NUMINAMATH_CALUDE_alex_savings_l1739_173976


namespace NUMINAMATH_CALUDE_distance_to_AB_l1739_173968

/-- Triangle ABC with point M inside -/
structure TriangleWithPoint where
  -- Define the triangle
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Define the distances from M to sides
  distMAC : ℝ
  distMBC : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  AC_positive : AC > 0
  distMAC_positive : distMAC > 0
  distMBC_positive : distMBC > 0
  -- Triangle inequality
  triangle_inequality : AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB
  -- M is inside the triangle
  M_inside : distMAC < AC ∧ distMBC < BC

/-- The theorem to be proved -/
theorem distance_to_AB (t : TriangleWithPoint) 
  (h1 : t.AB = 10) 
  (h2 : t.BC = 17) 
  (h3 : t.AC = 21) 
  (h4 : t.distMAC = 2) 
  (h5 : t.distMBC = 4) : 
  ∃ (distMAB : ℝ), distMAB = 29 / 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_AB_l1739_173968


namespace NUMINAMATH_CALUDE_team_average_score_l1739_173962

theorem team_average_score (lefty_score : ℕ) (righty_score : ℕ) (other_score : ℕ) :
  lefty_score = 20 →
  righty_score = lefty_score / 2 →
  other_score = righty_score * 6 →
  (lefty_score + righty_score + other_score) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_team_average_score_l1739_173962


namespace NUMINAMATH_CALUDE_tv_cost_is_250_l1739_173983

/-- The cost of the TV given Linda's savings and furniture expenditure -/
def tv_cost (savings : ℚ) (furniture_fraction : ℚ) : ℚ :=
  savings * (1 - furniture_fraction)

/-- Theorem stating that the TV cost is $250 given the problem conditions -/
theorem tv_cost_is_250 :
  tv_cost 1000 (3/4) = 250 := by
  sorry

end NUMINAMATH_CALUDE_tv_cost_is_250_l1739_173983


namespace NUMINAMATH_CALUDE_union_of_sets_l1739_173965

theorem union_of_sets (p q : ℝ) :
  let A := {x : ℝ | x^2 + p*x + q = 0}
  let B := {x : ℝ | x^2 - p*x - 2*q = 0}
  (A ∩ B = {-1}) →
  (A ∪ B = {-1, -2, 4}) := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1739_173965
