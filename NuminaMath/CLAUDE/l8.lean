import Mathlib

namespace loan_duration_for_b_l8_865

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem loan_duration_for_b (principal_b principal_c duration_c total_interest : ℝ) 
  (rate : ℝ) (h1 : principal_b = 5000)
  (h2 : principal_c = 3000)
  (h3 : duration_c = 4)
  (h4 : rate = 0.1)
  (h5 : simple_interest principal_b rate (duration_b) + 
        simple_interest principal_c rate duration_c = total_interest)
  (h6 : total_interest = 2200) :
  duration_b = 2 := by
  sorry

#check loan_duration_for_b

end loan_duration_for_b_l8_865


namespace next_occurrence_theorem_l8_842

/-- Represents a date and time -/
structure DateTime where
  day : Nat
  month : Nat
  year : Nat
  hour : Nat
  minute : Nat

/-- Check if a DateTime is valid -/
def isValidDateTime (dt : DateTime) : Prop :=
  dt.day ≥ 1 ∧ dt.day ≤ 31 ∧
  dt.month ≥ 1 ∧ dt.month ≤ 12 ∧
  dt.year ≥ 1900 ∧ dt.year ≤ 2100 ∧
  dt.hour ≥ 0 ∧ dt.hour ≤ 23 ∧
  dt.minute ≥ 0 ∧ dt.minute ≤ 59

/-- Check if two DateTimes use the same set of digits -/
def hasSameDigits (dt1 dt2 : DateTime) : Prop :=
  sorry

/-- The initial DateTime -/
def initialDateTime : DateTime :=
  { day := 25, month := 5, year := 1994, hour := 2, minute := 45 }

/-- The next DateTime with the same digits -/
def nextDateTime : DateTime :=
  { day := 1, month := 8, year := 1994, hour := 2, minute := 45 }

theorem next_occurrence_theorem :
  isValidDateTime initialDateTime ∧
  isValidDateTime nextDateTime ∧
  hasSameDigits initialDateTime nextDateTime ∧
  ∀ dt, isValidDateTime dt →
        hasSameDigits initialDateTime dt →
        (dt.year < nextDateTime.year ∨
         (dt.year = nextDateTime.year ∧ dt.month < nextDateTime.month) ∨
         (dt.year = nextDateTime.year ∧ dt.month = nextDateTime.month ∧ dt.day < nextDateTime.day) ∨
         (dt.year = nextDateTime.year ∧ dt.month = nextDateTime.month ∧ dt.day = nextDateTime.day ∧ dt.hour < nextDateTime.hour) ∨
         (dt.year = nextDateTime.year ∧ dt.month = nextDateTime.month ∧ dt.day = nextDateTime.day ∧ dt.hour = nextDateTime.hour ∧ dt.minute ≤ nextDateTime.minute)) :=
  by sorry

end next_occurrence_theorem_l8_842


namespace mud_efficacy_ratio_l8_808

/-- Represents the number of sprigs of mint in the original mud mixture -/
def original_mint_sprigs : ℕ := 3

/-- Represents the number of green tea leaves per sprig of mint -/
def tea_leaves_per_sprig : ℕ := 2

/-- Represents the number of green tea leaves needed in the new mud for the same efficacy -/
def new_mud_tea_leaves : ℕ := 12

/-- Calculates the ratio of efficacy of new mud to original mud -/
def efficacy_ratio : ℚ := 1 / 2

theorem mud_efficacy_ratio :
  efficacy_ratio = 1 / 2 := by sorry

end mud_efficacy_ratio_l8_808


namespace contestant_selection_probabilities_l8_849

/-- Represents the probability of selecting two females from a group of contestants. -/
def prob_two_females (total : ℕ) (females : ℕ) : ℚ :=
  (females.choose 2 : ℚ) / (total.choose 2 : ℚ)

/-- Represents the probability of selecting at least one male from a group of contestants. -/
def prob_at_least_one_male (total : ℕ) (females : ℕ) : ℚ :=
  1 - prob_two_females total females

/-- Theorem stating the probabilities for selecting contestants from a group of 8 with 5 females and 3 males. -/
theorem contestant_selection_probabilities :
  let total := 8
  let females := 5
  prob_two_females total females = 5 / 14 ∧
  prob_at_least_one_male total females = 9 / 14 := by
  sorry

end contestant_selection_probabilities_l8_849


namespace peach_difference_l8_820

/-- Proves that the difference between green and red peaches is 150 --/
theorem peach_difference : 
  let total_baskets : ℕ := 20
  let odd_red : ℕ := 12
  let odd_green : ℕ := 22
  let even_red : ℕ := 15
  let even_green : ℕ := 20
  let total_odd : ℕ := total_baskets / 2
  let total_even : ℕ := total_baskets / 2
  let total_red : ℕ := odd_red * total_odd + even_red * total_even
  let total_green : ℕ := odd_green * total_odd + even_green * total_even
  total_green - total_red = 150 := by
  sorry

end peach_difference_l8_820


namespace division_properties_l8_894

theorem division_properties (a b : ℤ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (¬ (a ∣ b^2 ↔ a ∣ b)) ∧ (a^2 ∣ b^2 ↔ a ∣ b) := by
  sorry

end division_properties_l8_894


namespace fraction_value_l8_882

theorem fraction_value (x y : ℝ) (hx : x = 4) (hy : y = -3) :
  (x - 2*y) / (x + y) = 10 := by
  sorry

end fraction_value_l8_882


namespace max_fraction_bound_l8_846

theorem max_fraction_bound (A B : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (hAB : A ≠ B) 
  (hA1000 : A < 1000) (hB1000 : B < 1000) : 
  (A : ℚ) - B ≤ 499 * ((A : ℚ) + B) / 500 :=
sorry

end max_fraction_bound_l8_846


namespace equation_positive_roots_l8_869

theorem equation_positive_roots (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ a * |x| + |x + a| = 0) ↔ -1 < a ∧ a < 0 := by
  sorry

end equation_positive_roots_l8_869


namespace roots_of_f_eq_x_none_or_infinite_l8_828

theorem roots_of_f_eq_x_none_or_infinite (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) = f x + 1) :
  (∀ x : ℝ, f x ≠ x) ∨ (∃ S : Set ℝ, Set.Infinite S ∧ ∀ x ∈ S, f x = x) :=
sorry

end roots_of_f_eq_x_none_or_infinite_l8_828


namespace polynomial_factorization_l8_850

theorem polynomial_factorization (a : ℝ) : 
  -3*a + 12*a^2 - 12*a^3 = -3*a*(1 - 2*a)^2 := by
  sorry

end polynomial_factorization_l8_850


namespace pat_final_sticker_count_l8_819

/-- Calculates the final number of stickers Pat has at the end of the week -/
def final_sticker_count (initial : ℕ) (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday_out : ℕ) (thursday_in : ℕ) (friday : ℕ) : ℕ :=
  initial + monday - tuesday + wednesday - thursday_out + thursday_in + friday

/-- Theorem stating that Pat ends up with 43 stickers at the end of the week -/
theorem pat_final_sticker_count :
  final_sticker_count 39 15 22 10 12 8 5 = 43 := by
  sorry

end pat_final_sticker_count_l8_819


namespace dolphin_population_estimate_l8_880

/-- Estimate the number of dolphins in a coastal area on January 1st -/
theorem dolphin_population_estimate (tagged_initial : ℕ) (captured_june : ℕ) (tagged_june : ℕ)
  (migration_rate : ℚ) (new_arrival_rate : ℚ) :
  tagged_initial = 100 →
  captured_june = 90 →
  tagged_june = 4 →
  migration_rate = 1/5 →
  new_arrival_rate = 1/2 →
  ∃ (initial_population : ℕ), initial_population = 1125 :=
by sorry

end dolphin_population_estimate_l8_880


namespace constant_grid_values_l8_815

theorem constant_grid_values (f : ℤ × ℤ → ℕ) 
  (h : ∀ (x y : ℤ), f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4) : 
  ∃ (c : ℕ), ∀ (x y : ℤ), f (x, y) = c :=
sorry

end constant_grid_values_l8_815


namespace arithmetic_progression_tenth_term_zero_l8_852

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.a + (n - 1 : ℝ) * ap.d

theorem arithmetic_progression_tenth_term_zero
  (ap : ArithmeticProgression)
  (h : ap.nthTerm 5 + ap.nthTerm 21 = ap.nthTerm 8 + ap.nthTerm 15 + ap.nthTerm 13) :
  ap.nthTerm 10 = 0 := by
  sorry

end arithmetic_progression_tenth_term_zero_l8_852


namespace geometric_sequence_solution_l8_871

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_solution (a : ℕ → ℝ) (q : ℝ) 
  (h1 : geometric_sequence a q)
  (h2 : a 5 - a 1 = 15)
  (h3 : a 4 - a 2 = 6) :
  (q = 2 ∧ a 3 = 4) ∨ (q = 1/2 ∧ a 3 = -4) :=
sorry

end geometric_sequence_solution_l8_871


namespace percentage_of_muslim_boys_l8_890

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) : 
  total_boys = 850 →
  hindu_percentage = 32/100 →
  sikh_percentage = 10/100 →
  other_boys = 119 →
  (total_boys - (hindu_percentage * total_boys).num - (sikh_percentage * total_boys).num - other_boys) / total_boys = 44/100 := by
sorry

end percentage_of_muslim_boys_l8_890


namespace cubic_polynomial_coefficient_expression_l8_854

/-- Represents a cubic polynomial of the form ax^3 + bx^2 + cx + d -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluates the cubic polynomial at a given x -/
def CubicPolynomial.evaluate (p : CubicPolynomial) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The specific cubic polynomial f(x) = 2x^3 - 3x^2 + 5x - 7 -/
def f : CubicPolynomial :=
  { a := 2, b := -3, c := 5, d := -7 }

theorem cubic_polynomial_coefficient_expression :
  16 * f.a - 9 * f.b + 3 * f.c - 2 * f.d = 88 := by
  sorry

end cubic_polynomial_coefficient_expression_l8_854


namespace percentage_of_muslim_boys_l8_838

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 400 →
  hindu_percentage = 28 / 100 →
  sikh_percentage = 10 / 100 →
  other_boys = 72 →
  (total_boys - (hindu_percentage * total_boys + sikh_percentage * total_boys + other_boys : ℚ)) / total_boys = 44 / 100 := by
  sorry

end percentage_of_muslim_boys_l8_838


namespace trapezoid_area_l8_887

/-- Given an outer equilateral triangle with area 36, an inner equilateral triangle
    with area 4, and three congruent trapezoids between them, the area of one
    trapezoid is 32/3. -/
theorem trapezoid_area (outer_triangle : Real) (inner_triangle : Real) (num_trapezoids : Nat) :
  outer_triangle = 36 →
  inner_triangle = 4 →
  num_trapezoids = 3 →
  (outer_triangle - inner_triangle) / num_trapezoids = 32 / 3 := by
  sorry

end trapezoid_area_l8_887


namespace scientific_notation_of_58_billion_l8_841

theorem scientific_notation_of_58_billion :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 58000000000 = a * (10 : ℝ) ^ n ∧ a = 5.8 ∧ n = 10 := by
  sorry

end scientific_notation_of_58_billion_l8_841


namespace trust_fund_remaining_zero_l8_829

/-- Represents the ratio of distribution for each beneficiary -/
structure DistributionRatio :=
  (dina : Rat)
  (eva : Rat)
  (frank : Rat)

/-- Theorem stating that the remaining fraction of the fund is 0 -/
theorem trust_fund_remaining_zero (ratio : DistributionRatio) 
  (h1 : ratio.dina = 4/8)
  (h2 : ratio.eva = 3/8)
  (h3 : ratio.frank = 1/8)
  (h4 : ratio.dina + ratio.eva + ratio.frank = 1) :
  let remaining : Rat := 1 - (ratio.dina + (1 - ratio.dina) * ratio.eva + (1 - ratio.dina - (1 - ratio.dina) * ratio.eva) * ratio.frank)
  remaining = 0 := by sorry

end trust_fund_remaining_zero_l8_829


namespace max_xy_in_all_H_l8_840

-- Define the set H_n recursively
def H : ℕ → Set (ℝ × ℝ)
| 0 => {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}
| (n+1) => {p : ℝ × ℝ | ∃ q ∈ H n, 
    (n % 2 = 0 ∧ ((p.1 = q.1 - 1/(2^(n+2)) ∧ p.2 = q.2 + 1/(2^(n+2))) ∨ 
                  (p.1 = q.1 + 1/(2^(n+2)) ∧ p.2 = q.2 - 1/(2^(n+2))))) ∨
    (n % 2 = 1 ∧ ((p.1 = q.1 + 1/(2^(n+2)) ∧ p.2 = q.2 + 1/(2^(n+2))) ∨ 
                  (p.1 = q.1 - 1/(2^(n+2)) ∧ p.2 = q.2 - 1/(2^(n+2)))))}

-- Define the set of points that lie in all H_n
def InAllH := {p : ℝ × ℝ | ∀ n : ℕ, p ∈ H n}

-- State the theorem
theorem max_xy_in_all_H : 
  ∀ p ∈ InAllH, p.1 * p.2 ≤ 11/16 :=
sorry

end max_xy_in_all_H_l8_840


namespace darrel_coin_counting_machine_result_l8_891

/-- Calculates the amount received after fees for a given coin type -/
def amountAfterFee (coinValue : ℚ) (count : ℕ) (feePercentage : ℚ) : ℚ :=
  let totalValue := coinValue * count
  totalValue - (totalValue * feePercentage / 100)

/-- Theorem stating the total amount Darrel receives after fees -/
theorem darrel_coin_counting_machine_result : 
  let quarterCount : ℕ := 127
  let dimeCount : ℕ := 183
  let nickelCount : ℕ := 47
  let pennyCount : ℕ := 237
  let halfDollarCount : ℕ := 64
  
  let quarterValue : ℚ := 25 / 100
  let dimeValue : ℚ := 10 / 100
  let nickelValue : ℚ := 5 / 100
  let pennyValue : ℚ := 1 / 100
  let halfDollarValue : ℚ := 50 / 100
  
  let quarterFee : ℚ := 12
  let dimeFee : ℚ := 7
  let nickelFee : ℚ := 15
  let pennyFee : ℚ := 10
  let halfDollarFee : ℚ := 5
  
  let totalAfterFees := 
    amountAfterFee quarterValue quarterCount quarterFee +
    amountAfterFee dimeValue dimeCount dimeFee +
    amountAfterFee nickelValue nickelCount nickelFee +
    amountAfterFee pennyValue pennyCount pennyFee +
    amountAfterFee halfDollarValue halfDollarCount halfDollarFee
  
  totalAfterFees = 7949 / 100 := by
  sorry


end darrel_coin_counting_machine_result_l8_891


namespace flight_time_around_earth_l8_889

def earth_radius : ℝ := 6000
def jet_speed : ℝ := 600

theorem flight_time_around_earth :
  let circumference := 2 * Real.pi * earth_radius
  let flight_time := circumference / jet_speed
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ abs (flight_time - 63) < ε := by
sorry

end flight_time_around_earth_l8_889


namespace pet_store_cages_l8_811

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : puppies_per_cage = 5) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end pet_store_cages_l8_811


namespace unpainted_cubes_in_6x6x3_prism_l8_877

/-- Represents a rectangular prism -/
structure RectangularPrism :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Counts the number of unpainted cubes in a painted rectangular prism -/
def count_unpainted_cubes (prism : RectangularPrism) : ℕ :=
  if prism.height ≤ 2 then 0
  else (prism.length - 2) * (prism.width - 2)

/-- Theorem stating that a 6 × 6 × 3 painted prism has 16 unpainted cubes -/
theorem unpainted_cubes_in_6x6x3_prism :
  let prism : RectangularPrism := ⟨6, 6, 3⟩
  count_unpainted_cubes prism = 16 := by
  sorry

#eval count_unpainted_cubes ⟨6, 6, 3⟩

end unpainted_cubes_in_6x6x3_prism_l8_877


namespace quadratic_factorization_l8_805

theorem quadratic_factorization (x : ℝ) :
  x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end quadratic_factorization_l8_805


namespace initial_friends_correct_l8_888

/-- The number of friends James had initially -/
def initial_friends : ℕ := 20

/-- The number of friends James lost due to an argument -/
def friends_lost : ℕ := 2

/-- The number of new friends James made -/
def new_friends : ℕ := 1

/-- The number of friends James has now -/
def current_friends : ℕ := 19

/-- Theorem stating that the initial number of friends is correct given the conditions -/
theorem initial_friends_correct :
  initial_friends = current_friends + friends_lost - new_friends :=
by sorry

end initial_friends_correct_l8_888


namespace point_c_coordinates_l8_853

/-- Point with x and y coordinates -/
structure Point where
  x : ℚ
  y : ℚ

/-- Distance between two points -/
def distance (p q : Point) : ℚ :=
  ((p.x - q.x)^2 + (p.y - q.y)^2).sqrt

/-- Check if a point is on a line segment -/
def isOnSegment (p q r : Point) : Prop :=
  distance p r + distance r q = distance p q

theorem point_c_coordinates :
  let a : Point := ⟨-3, 2⟩
  let b : Point := ⟨5, 10⟩
  ∀ c : Point,
    isOnSegment a c b →
    distance a c = 2 * distance c b →
    c = ⟨7/3, 22/3⟩ := by
  sorry

end point_c_coordinates_l8_853


namespace max_sum_of_integers_l8_833

theorem max_sum_of_integers (A B C D : ℕ) : 
  (10 ≤ A) ∧ (A < 100) ∧
  (10 ≤ B) ∧ (B < 100) ∧
  (10 ≤ C) ∧ (C < 100) ∧
  (10 ≤ D) ∧ (D < 100) ∧
  (B = 3 * C) ∧
  (D = 2 * B - C) ∧
  (A = B + D) →
  A + B + C + D ≤ 204 :=
by
  sorry

end max_sum_of_integers_l8_833


namespace office_employees_l8_858

theorem office_employees (total_employees : ℕ) 
  (h1 : (45 : ℚ) / 100 * total_employees = total_males)
  (h2 : (50 : ℚ) / 100 * total_males = males_50_and_above)
  (h3 : 1170 = total_males - males_50_and_above) :
  total_employees = 5200 :=
by sorry

end office_employees_l8_858


namespace ferry_problem_l8_845

/-- Represents the ferry problem and proves the speed of the current and distance between docks. -/
theorem ferry_problem (still_water_speed time_against time_with : ℝ) 
  (h1 : still_water_speed = 12)
  (h2 : time_against = 10)
  (h3 : time_with = 6) :
  ∃ (current_speed distance : ℝ),
    current_speed = 3 ∧
    distance = 90 ∧
    time_with * (still_water_speed + current_speed) = time_against * (still_water_speed - current_speed) ∧
    distance = (still_water_speed + current_speed) * time_with :=
by
  sorry


end ferry_problem_l8_845


namespace line_equation_l8_812

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

/-- The point through which the line passes -/
def point : ℝ × ℝ := (2, 1)

/-- Predicate to check if a point is on the line -/
def on_line (m b x y : ℝ) : Prop := y = m * x + b

/-- Predicate to check if a point bisects a chord -/
def bisects_chord (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    on_line 8 (-15) x₁ y₁ ∧
    on_line 8 (-15) x₂ y₂ ∧
    bisects_chord point.1 point.2 x₁ y₁ x₂ y₂ :=
sorry

end line_equation_l8_812


namespace f_inequality_solution_l8_832

noncomputable def f (a x : ℝ) : ℝ := |x - a| - |x + 3|

theorem f_inequality_solution (a : ℝ) :
  (a = -1 → {x : ℝ | f a x ≤ 1} = {x : ℝ | x ≥ -5/2}) ∧
  ({a : ℝ | ∀ x ∈ Set.Icc 0 3, f a x ≤ 4} = Set.Icc (-7) 7) :=
sorry

end f_inequality_solution_l8_832


namespace rightmost_book_price_l8_898

/-- Represents the price of a book at a given position. -/
def book_price (first_price : ℕ) (position : ℕ) : ℕ :=
  first_price + 3 * (position - 1)

/-- The theorem states that for a sequence of 41 books with the given conditions,
    the price of the rightmost book is $150. -/
theorem rightmost_book_price (first_price : ℕ) :
  (book_price first_price 41 = 
   book_price first_price 20 + 
   book_price first_price 21 + 
   book_price first_price 22) →
  book_price first_price 41 = 150 := by
sorry

end rightmost_book_price_l8_898


namespace a_range_proof_l8_825

-- Define the propositions p and q
def p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4

def q (x : ℝ) : Prop := (x - 1) * (x - 3) < 0

-- Define the range of a
def a_range (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 5

-- State the theorem
theorem a_range_proof :
  (∀ x a : ℝ, q x → p x a) ∧  -- q is sufficient for p
  (∃ x a : ℝ, p x a ∧ ¬(q x)) ∧  -- q is not necessary for p
  (∀ a : ℝ, a_range a ↔ ∀ x : ℝ, q x → p x a) :=
by sorry

end a_range_proof_l8_825


namespace max_value_of_f_l8_835

-- Define the function
def f (x : ℝ) : ℝ := (x + 1)^2 - 4

-- State the theorem
theorem max_value_of_f : 
  ∃ (M : ℝ), M = 5 ∧ ∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≤ M :=
sorry

end max_value_of_f_l8_835


namespace grasshopper_jump_distance_l8_803

/-- The jumping contest between a grasshopper and a frog -/
theorem grasshopper_jump_distance 
  (frog_jump : ℕ) 
  (frog_grasshopper_difference : ℕ) 
  (h1 : frog_jump = 12)
  (h2 : frog_jump = frog_grasshopper_difference + grasshopper_jump) :
  grasshopper_jump = 9 :=
by
  sorry

end grasshopper_jump_distance_l8_803


namespace share_distribution_l8_866

theorem share_distribution (x y z : ℚ) (a : ℚ) : 
  (x + y + z = 156) →  -- total amount
  (y = 36) →           -- y's share
  (z = x * (1/2)) →    -- z gets 50 paisa for each rupee x gets
  (y = x * a) →        -- y gets 'a' for each rupee x gets
  (a = 9/20) := by
    sorry

end share_distribution_l8_866


namespace road_vehicles_l8_885

/-- Given a road with the specified conditions, prove the total number of vehicles -/
theorem road_vehicles (lanes : Nat) (trucks_per_lane : Nat) (cars_multiplier : Nat) : 
  lanes = 4 → 
  trucks_per_lane = 60 → 
  cars_multiplier = 2 →
  (lanes * trucks_per_lane + lanes * cars_multiplier * lanes * trucks_per_lane) = 2160 := by
  sorry

#check road_vehicles

end road_vehicles_l8_885


namespace impossibleRectangle_l8_895

/-- Represents the counts of sticks of each length -/
structure StickCounts where
  one_cm : Nat
  two_cm : Nat
  three_cm : Nat
  four_cm : Nat

/-- Calculates the total length of all sticks -/
def totalLength (counts : StickCounts) : Nat :=
  counts.one_cm * 1 + counts.two_cm * 2 + counts.three_cm * 3 + counts.four_cm * 4

/-- Theorem stating that it's impossible to form a rectangle with the given sticks -/
theorem impossibleRectangle (counts : StickCounts) 
  (h1 : counts.one_cm = 4)
  (h2 : counts.two_cm = 4)
  (h3 : counts.three_cm = 7)
  (h4 : counts.four_cm = 5)
  (h5 : totalLength counts = 53) :
  ¬∃ (a b : Nat), a + b = (totalLength counts) / 2 := by
  sorry

#eval totalLength { one_cm := 4, two_cm := 4, three_cm := 7, four_cm := 5 }

end impossibleRectangle_l8_895


namespace bike_race_distance_difference_l8_872

/-- Represents a cyclist with their distance traveled and time taken -/
structure Cyclist where
  distance : ℝ
  time : ℝ

/-- The difference in distance traveled between two cyclists -/
def distanceDifference (c1 c2 : Cyclist) : ℝ :=
  c1.distance - c2.distance

theorem bike_race_distance_difference :
  let carlos : Cyclist := { distance := 70, time := 5 }
  let dana : Cyclist := { distance := 50, time := 5 }
  distanceDifference carlos dana = 20 := by
  sorry

end bike_race_distance_difference_l8_872


namespace quadratic_equation_solution_l8_844

theorem quadratic_equation_solution (k : ℝ) (x : ℝ) :
  k * x^2 - (3 * k + 3) * x + 2 * k + 6 = 0 →
  (k = 0 → x = 2) ∧
  (k ≠ 0 → (x = 2 ∨ x = 1 + 3 / k)) := by
  sorry

end quadratic_equation_solution_l8_844


namespace average_price_23_story_building_l8_826

/-- The average price per square meter of a 23-story building with specific pricing conditions. -/
theorem average_price_23_story_building (a₁ a₂ a : ℝ) : 
  let floor_prices : List ℝ := 
    [a₁] ++ (List.range 21).map (λ i => a + i * (a / 100)) ++ [a₂]
  (floor_prices.sum / 23 : ℝ) = (a₁ + a₂ + 23.1 * a) / 23 := by
  sorry

end average_price_23_story_building_l8_826


namespace sphere_radius_in_tetrahedron_l8_824

/-- A regular tetrahedron with side length 1 containing four spheres --/
structure TetrahedronWithSpheres where
  /-- The side length of the regular tetrahedron --/
  sideLength : ℝ
  /-- The radius of each sphere --/
  sphereRadius : ℝ
  /-- The number of spheres --/
  numSpheres : ℕ
  /-- Each sphere is tangent to three faces of the tetrahedron --/
  tangentToFaces : Prop
  /-- Each sphere is tangent to the other three spheres --/
  tangentToOtherSpheres : Prop
  /-- The side length of the tetrahedron is 1 --/
  sideLength_eq_one : sideLength = 1
  /-- There are exactly four spheres --/
  numSpheres_eq_four : numSpheres = 4

/-- The theorem stating the radius of the spheres in the tetrahedron --/
theorem sphere_radius_in_tetrahedron (t : TetrahedronWithSpheres) :
  t.sphereRadius = (Real.sqrt 6 - 1) / 10 := by
  sorry

end sphere_radius_in_tetrahedron_l8_824


namespace decreasing_interval_of_f_l8_822

-- Define the function
def f (x : ℝ) : ℝ := x^2 * (x - 3)

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem decreasing_interval_of_f :
  ∀ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioo 0 2, x < y → f y < f x :=
by sorry

end decreasing_interval_of_f_l8_822


namespace one_slice_left_l8_821

/-- Represents the number of slices in a whole pizza after cutting -/
def total_slices : ℕ := 8

/-- Represents the number of friends who receive 1 slice each -/
def friends_one_slice : ℕ := 3

/-- Represents the number of friends who receive 2 slices each -/
def friends_two_slices : ℕ := 2

/-- Represents the number of slices given to friends -/
def slices_given : ℕ := friends_one_slice * 1 + friends_two_slices * 2

/-- Theorem stating that there is 1 slice left after distribution -/
theorem one_slice_left : total_slices - slices_given = 1 := by
  sorry

end one_slice_left_l8_821


namespace hyperbola_asymptote_slope_l8_897

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote
def asymptote (a b x y : ℝ) : Prop := y = (b / a) * x ∨ y = -(b / a) * x

-- Define the perpendicularity condition
def perpendicular (F₁ F₂ A : ℝ × ℝ) : Prop :=
  (A.2 - F₂.2) * (F₂.1 - F₁.1) = (F₂.2 - F₁.2) * (A.1 - F₂.1)

-- Define the distance condition
def distance_condition (O F₁ A : ℝ × ℝ) : Prop :=
  let d := abs ((A.2 - F₁.2) * O.1 - (A.1 - F₁.1) * O.2 + A.1 * F₁.2 - A.2 * F₁.1) /
            Real.sqrt ((A.2 - F₁.2)^2 + (A.1 - F₁.1)^2)
  d = (1/3) * Real.sqrt (F₁.1^2 + F₁.2^2)

-- Main theorem
theorem hyperbola_asymptote_slope (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (A F₁ F₂ O : ℝ × ℝ) :
  hyperbola a b A.1 A.2 →
  asymptote a b A.1 A.2 →
  perpendicular F₁ F₂ A →
  distance_condition O F₁ A →
  (b / a = Real.sqrt 2 / 2) ∨ (b / a = -Real.sqrt 2 / 2) :=
by sorry

end hyperbola_asymptote_slope_l8_897


namespace total_grapes_is_83_l8_834

/-- The number of grapes in Rob's bowl -/
def rob_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allie_grapes : ℕ := rob_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyn_grapes : ℕ := allie_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := rob_grapes + allie_grapes + allyn_grapes

theorem total_grapes_is_83 : total_grapes = 83 := by
  sorry

end total_grapes_is_83_l8_834


namespace crayon_ratio_l8_837

def karen_crayons : ℕ := 128
def judah_crayons : ℕ := 8

def gilbert_crayons : ℕ := 4 * judah_crayons
def beatrice_crayons : ℕ := 2 * gilbert_crayons

theorem crayon_ratio :
  karen_crayons / beatrice_crayons = 2 := by
  sorry

end crayon_ratio_l8_837


namespace multiply_63_37_l8_892

theorem multiply_63_37 : 63 * 37 = 2331 := by
  sorry

end multiply_63_37_l8_892


namespace base_8_4513_equals_2379_l8_836

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4513_equals_2379 :
  base_8_to_10 [3, 1, 5, 4] = 2379 := by
  sorry

end base_8_4513_equals_2379_l8_836


namespace even_function_sum_l8_818

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is symmetric about the origin if its endpoints are additive inverses -/
def SymmetricDomain (a : ℝ) : Prop :=
  a - 1 = -3 * a

theorem even_function_sum (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : f = fun x ↦ a * x^2 + b * x)
    (h2 : IsEven f)
    (h3 : SymmetricDomain a) : 
  a + b = 1/4 := by
  sorry

end even_function_sum_l8_818


namespace customer_coin_count_l8_879

/-- Represents the quantity of each type of coin turned in by the customer --/
structure CoinQuantities where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat
  halfDollars : Nat
  oneDollarCoins : Nat
  twoDollarCoins : Nat
  australianFiftyCentCoins : Nat
  mexicanOnePesoCoins : Nat

/-- Calculates the total number of coins turned in --/
def totalCoins (coins : CoinQuantities) : Nat :=
  coins.pennies +
  coins.nickels +
  coins.dimes +
  coins.quarters +
  coins.halfDollars +
  coins.oneDollarCoins +
  coins.twoDollarCoins +
  coins.australianFiftyCentCoins +
  coins.mexicanOnePesoCoins

/-- Theorem: The total number of coins turned in by the customer is 159 --/
theorem customer_coin_count :
  ∃ (coins : CoinQuantities),
    coins.pennies = 38 ∧
    coins.nickels = 27 ∧
    coins.dimes = 19 ∧
    coins.quarters = 24 ∧
    coins.halfDollars = 13 ∧
    coins.oneDollarCoins = 17 ∧
    coins.twoDollarCoins = 5 ∧
    coins.australianFiftyCentCoins = 4 ∧
    coins.mexicanOnePesoCoins = 12 ∧
    totalCoins coins = 159 := by
  sorry

end customer_coin_count_l8_879


namespace choose_from_four_and_three_l8_881

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (set1_size set2_size : ℕ) : ℕ :=
  set1_size * set2_size

/-- Theorem: Choosing one item from a set of 4 and one from a set of 3 results in 12 possibilities -/
theorem choose_from_four_and_three :
  choose_one_from_each 4 3 = 12 := by
  sorry

end choose_from_four_and_three_l8_881


namespace parity_of_solutions_l8_859

theorem parity_of_solutions (n m p q : ℤ) : 
  (∃ k : ℤ, n = 2 * k) →  -- n is even
  (∃ k : ℤ, m = 2 * k + 1) →  -- m is odd
  p - 1988 * q = n →  -- first equation
  11 * p + 27 * q = m →  -- second equation
  (∃ k : ℤ, p = 2 * k) ∧ (∃ k : ℤ, q = 2 * k + 1) :=  -- p is even and q is odd
by sorry

end parity_of_solutions_l8_859


namespace sum_of_two_primes_52_l8_883

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem sum_of_two_primes_52 :
  ∃! (count : ℕ), ∃ (pairs : List (ℕ × ℕ)),
    (∀ (p q : ℕ), (p, q) ∈ pairs → is_prime p ∧ is_prime q ∧ p + q = 52) ∧
    (∀ (p q : ℕ), is_prime p → is_prime q → p + q = 52 → (p, q) ∈ pairs ∨ (q, p) ∈ pairs) ∧
    count = pairs.length ∧
    count = 3 :=
sorry

end sum_of_two_primes_52_l8_883


namespace largest_prime_factor_l8_830

/-- The binary to decimal conversion for 10010000 -/
def binary_to_decimal_1 : Nat := 144

/-- The binary to decimal conversion for 100100000 -/
def binary_to_decimal_2 : Nat := 288

/-- The function to check if a number is prime -/
def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem stating that 3 is the largest prime factor of both numbers -/
theorem largest_prime_factor :
  ∃ (p : Nat), is_prime p ∧ 
    p ∣ binary_to_decimal_1 ∧ 
    p ∣ binary_to_decimal_2 ∧
    ∀ (q : Nat), is_prime q → 
      q ∣ binary_to_decimal_1 → 
      q ∣ binary_to_decimal_2 → 
      q ≤ p :=
by sorry

end largest_prime_factor_l8_830


namespace jerry_removed_figures_l8_839

/-- The number of old action figures removed from Jerry's shelf. -/
def old_figures_removed (initial : ℕ) (added : ℕ) (current : ℕ) : ℕ :=
  initial + added - current

/-- Theorem stating the number of old action figures Jerry removed. -/
theorem jerry_removed_figures : old_figures_removed 7 11 8 = 10 := by
  sorry

end jerry_removed_figures_l8_839


namespace B_is_smallest_l8_870

def A : ℕ := 32 + 7
def B : ℕ := 3 * 10 + 3
def C : ℕ := 50 - 9

theorem B_is_smallest : B ≤ A ∧ B ≤ C := by
  sorry

end B_is_smallest_l8_870


namespace octagon_area_is_225_l8_873

-- Define the triangle and circle
structure Triangle :=
  (P Q R : ℝ × ℝ)

def circumradius : ℝ := 10

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ := 45

-- Define the points P', Q', R' as intersections of perpendicular bisectors with circumcircle
def P' (t : Triangle) : ℝ × ℝ := sorry
def Q' (t : Triangle) : ℝ × ℝ := sorry
def R' (t : Triangle) : ℝ × ℝ := sorry

-- Define S as reflection of circumcenter over PQ
def S (t : Triangle) : ℝ × ℝ := sorry

-- Define the area of the octagon
def octagon_area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem octagon_area_is_225 (t : Triangle) :
  octagon_area t = 225 := by sorry

end octagon_area_is_225_l8_873


namespace min_value_theorem_l8_857

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 1/x + 1/y + 1/z = 9 →
  a^4 * b^3 * c^2 ≤ x^4 * y^3 * z^2 ∧
  a^4 * b^3 * c^2 = 1/1152 := by
  sorry

end min_value_theorem_l8_857


namespace power_zero_eq_one_l8_874

theorem power_zero_eq_one (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end power_zero_eq_one_l8_874


namespace expression_value_approx_l8_848

def x : Real := 2.2
def a : Real := 3.6
def b : Real := 0.48
def c : Real := 2.50
def d : Real := 0.12
def e : Real := 0.09
def f : Real := 0.5

theorem expression_value_approx :
  ∃ (ε : Real), ε > 0 ∧ ε < 0.01 ∧ 
  |3 * x * ((a^2 * b * Real.log c) / (Real.sqrt d * Real.sin e * Real.log f)) + 720.72| < ε :=
by sorry

end expression_value_approx_l8_848


namespace orangeade_price_day_two_l8_899

/-- Represents the price of orangeade per glass on a given day. -/
structure OrangeadePrice where
  day : Nat
  price : ℚ

/-- Represents the amount of ingredients used to make orangeade on a given day. -/
structure OrangeadeIngredients where
  day : Nat
  orange_juice : ℚ
  water : ℚ

/-- Calculates the total volume of orangeade made on a given day. -/
def totalVolume (ingredients : OrangeadeIngredients) : ℚ :=
  ingredients.orange_juice + ingredients.water

/-- Calculates the revenue from selling orangeade on a given day. -/
def revenue (price : OrangeadePrice) (ingredients : OrangeadeIngredients) : ℚ :=
  price.price * totalVolume ingredients

/-- Theorem stating that the price of orangeade on the second day is $0.40 given the conditions. -/
theorem orangeade_price_day_two
  (day1_price : OrangeadePrice)
  (day1_ingredients : OrangeadeIngredients)
  (day2_ingredients : OrangeadeIngredients)
  (h1 : day1_price.day = 1)
  (h2 : day1_price.price = 6/10)
  (h3 : day1_ingredients.day = 1)
  (h4 : day1_ingredients.orange_juice = day1_ingredients.water)
  (h5 : day2_ingredients.day = 2)
  (h6 : day2_ingredients.orange_juice = day1_ingredients.orange_juice)
  (h7 : day2_ingredients.water = 2 * day1_ingredients.water)
  (h8 : revenue day1_price day1_ingredients = revenue { day := 2, price := 4/10 } day2_ingredients) :
  ∃ (day2_price : OrangeadePrice), day2_price.day = 2 ∧ day2_price.price = 4/10 :=
by sorry


end orangeade_price_day_two_l8_899


namespace complement_union_problem_l8_807

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 2}
def B : Set Int := {-1, 0, 1}

theorem complement_union_problem : (U \ B) ∪ A = {-2, -1, 2} := by
  sorry

end complement_union_problem_l8_807


namespace quadratic_function_y_order_l8_861

/-- Given a quadratic function f(x) = -x² - 4x + m, where m is a constant,
    and three points A, B, C on its graph, prove that the y-coordinate of B
    is greater than that of A, which is greater than that of C. -/
theorem quadratic_function_y_order (m : ℝ) (y₁ y₂ y₃ : ℝ) : 
  ((-3)^2 + 4*(-3) + m = y₁) →
  ((-2)^2 + 4*(-2) + m = y₂) →
  (1^2 + 4*1 + m = y₃) →
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry


end quadratic_function_y_order_l8_861


namespace quadratic_inequality_condition_l8_864

theorem quadratic_inequality_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + m > 0) ↔ m > 1 := by
  sorry

end quadratic_inequality_condition_l8_864


namespace max_nonmanagers_for_nine_managers_l8_862

/-- Represents a department in the corporation -/
structure Department where
  managers : ℕ
  nonManagers : ℕ

/-- The conditions for a valid department -/
def isValidDepartment (d : Department) : Prop :=
  d.managers > 0 ∧
  d.managers * 37 > 7 * d.nonManagers ∧
  d.managers ≥ 5 ∧
  d.managers + d.nonManagers ≤ 300 ∧
  d.managers = (d.managers + d.nonManagers) * 12 / 100

theorem max_nonmanagers_for_nine_managers :
  ∀ d : Department,
    isValidDepartment d →
    d.managers = 9 →
    d.nonManagers ≤ 66 :=
by sorry

end max_nonmanagers_for_nine_managers_l8_862


namespace coefficient_sum_equality_l8_868

theorem coefficient_sum_equality (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end coefficient_sum_equality_l8_868


namespace triangle_side_length_l8_847

-- Define the triangle PQS
structure Triangle :=
  (P Q S : ℝ × ℝ)

-- Define the length function
def length (A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_side_length (PQS : Triangle) :
  length PQS.P PQS.Q = 2 → length PQS.P PQS.S = 1.5 := by
  sorry

end triangle_side_length_l8_847


namespace card_addition_l8_856

theorem card_addition (initial_cards : ℕ) (added_cards : ℕ) : 
  initial_cards = 9 → added_cards = 4 → initial_cards + added_cards = 13 := by
sorry

end card_addition_l8_856


namespace eight_digit_divisible_by_eleven_l8_867

theorem eight_digit_divisible_by_eleven (n : ℕ) : 
  n < 10 →
  (965 * 10^7 + n * 10^6 + 8 * 10^5 + 4 * 10^4 + 3 * 10^3 + 2 * 10^2) % 11 = 0 →
  n = 1 := by
sorry

end eight_digit_divisible_by_eleven_l8_867


namespace nathan_harvest_earnings_is_186_l8_802

/-- Calculates the total earnings from Nathan's harvest --/
def nathan_harvest_earnings : ℕ :=
  let strawberry_plants : ℕ := 5
  let tomato_plants : ℕ := 7
  let strawberries_per_plant : ℕ := 14
  let tomatoes_per_plant : ℕ := 16
  let fruits_per_basket : ℕ := 7
  let price_strawberry_basket : ℕ := 9
  let price_tomato_basket : ℕ := 6

  let total_strawberries : ℕ := strawberry_plants * strawberries_per_plant
  let total_tomatoes : ℕ := tomato_plants * tomatoes_per_plant

  let strawberry_baskets : ℕ := total_strawberries / fruits_per_basket
  let tomato_baskets : ℕ := total_tomatoes / fruits_per_basket

  let earnings_strawberries : ℕ := strawberry_baskets * price_strawberry_basket
  let earnings_tomatoes : ℕ := tomato_baskets * price_tomato_basket

  earnings_strawberries + earnings_tomatoes

theorem nathan_harvest_earnings_is_186 : nathan_harvest_earnings = 186 := by
  sorry

end nathan_harvest_earnings_is_186_l8_802


namespace students_before_yoongi_l8_896

theorem students_before_yoongi (total_students : ℕ) (finished_after_yoongi : ℕ) : 
  total_students = 20 → finished_after_yoongi = 11 → 
  total_students - finished_after_yoongi - 1 = 8 := by
sorry

end students_before_yoongi_l8_896


namespace remaining_money_theorem_l8_863

def calculate_remaining_money (initial_amount : ℚ) : ℚ :=
  let day1_remaining := initial_amount * (1 - 3/5)
  let day2_remaining := day1_remaining * (1 - 7/12)
  let day3_remaining := day2_remaining * (1 - 2/3)
  let day4_remaining := day3_remaining * (1 - 1/6)
  let day5_remaining := day4_remaining * (1 - 5/8)
  let day6_remaining := day5_remaining * (1 - 3/5)
  day6_remaining

theorem remaining_money_theorem :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  abs (calculate_remaining_money 500 - 347/100) < ε := by
  sorry

end remaining_money_theorem_l8_863


namespace banana_permutations_eq_60_l8_851

/-- The number of letters in the word "BANANA" -/
def total_letters : Nat := 6

/-- The number of occurrences of 'A' in "BANANA" -/
def count_A : Nat := 3

/-- The number of occurrences of 'N' in "BANANA" -/
def count_N : Nat := 2

/-- The number of occurrences of 'B' in "BANANA" -/
def count_B : Nat := 1

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : Nat := Nat.factorial total_letters / (Nat.factorial count_A * Nat.factorial count_N)

theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end banana_permutations_eq_60_l8_851


namespace tan_half_product_squared_l8_886

theorem tan_half_product_squared (a b : ℝ) :
  7 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 3 := by
sorry

end tan_half_product_squared_l8_886


namespace geometric_sequence_sum_l8_823

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) * a m = a n * a (m + 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36 →
  a 5 + a 7 = 6 := by
sorry

end geometric_sequence_sum_l8_823


namespace total_contribution_l8_817

def contribution_problem (niraj brittany angela : ℕ) : Prop :=
  brittany = 3 * niraj ∧
  angela = 3 * brittany ∧
  niraj = 80

theorem total_contribution :
  ∀ niraj brittany angela : ℕ,
  contribution_problem niraj brittany angela →
  niraj + brittany + angela = 1040 :=
by
  sorry

end total_contribution_l8_817


namespace largest_solution_of_equation_l8_814

theorem largest_solution_of_equation (y : ℝ) :
  (3 * y^2 + 18 * y - 90 = y * (y + 17)) →
  y ≤ 3 :=
by sorry

end largest_solution_of_equation_l8_814


namespace arrangements_with_space_theorem_l8_806

/-- The number of arrangements of 6 people in a row where person A and person B
    have at least one person between them. -/
def arrangements_with_space_between (total_arrangements : ℕ) 
                                    (adjacent_arrangements : ℕ) : ℕ :=
  total_arrangements - adjacent_arrangements

/-- Theorem stating that the number of arrangements of 6 people in a row
    where person A and person B have at least one person between them is 480. -/
theorem arrangements_with_space_theorem :
  arrangements_with_space_between 720 240 = 480 := by
  sorry

end arrangements_with_space_theorem_l8_806


namespace a_minus_b_squared_l8_827

theorem a_minus_b_squared (a b : ℝ) (h1 : (a + b)^2 = 49) (h2 : a * b = 6) : 
  (a - b)^2 = 25 := by
  sorry

end a_minus_b_squared_l8_827


namespace parabola_contradiction_l8_813

theorem parabola_contradiction (a b c : ℝ) : 
  ¬(((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c > 0)) ∧
    ((a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c < 0))) :=
by
  sorry

end parabola_contradiction_l8_813


namespace cricket_target_runs_l8_843

/-- Calculates the target number of runs in a cricket game given specific conditions -/
theorem cricket_target_runs (total_overs run_rate_first_12 run_rate_remaining_38 : ℝ) 
  (h1 : total_overs = 50)
  (h2 : run_rate_first_12 = 4.5)
  (h3 : run_rate_remaining_38 = 8.052631578947368) : 
  ∃ (target : ℕ), target = 360 ∧ 
  target = ⌊run_rate_first_12 * 12 + run_rate_remaining_38 * (total_overs - 12)⌋ := by
  sorry

#check cricket_target_runs

end cricket_target_runs_l8_843


namespace intersection_complement_proof_l8_810

def U : Set Nat := {1, 2, 3, 4}

theorem intersection_complement_proof
  (A B : Set Nat)
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : (U \ (A ∪ B)) = {4})
  (h4 : B = {1, 2}) :
  A ∩ (U \ B) = {3} :=
by
  sorry

end intersection_complement_proof_l8_810


namespace christmas_monday_implies_jan25_thursday_l8_809

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in a year -/
structure Date where
  month : Nat
  day : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (d : Date) : DayOfWeek :=
  sorry

/-- Function to get the date of the next year -/
def nextYearDate (d : Date) : Date :=
  sorry

theorem christmas_monday_implies_jan25_thursday
  (h : dayOfWeek ⟨12, 25⟩ = DayOfWeek.Monday) :
  dayOfWeek (nextYearDate ⟨1, 25⟩) = DayOfWeek.Thursday :=
sorry

end christmas_monday_implies_jan25_thursday_l8_809


namespace complement_of_at_least_two_defective_l8_801

/-- The number of products sampled -/
def n : ℕ := 10

/-- The event of having at least two defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- The complementary event of event_A -/
def complement_A (defective : ℕ) : Prop := defective ≤ 1

/-- Theorem stating that the complement of event_A is having at most one defective product -/
theorem complement_of_at_least_two_defective :
  ∀ defective : ℕ, defective ≤ n → (¬ event_A defective ↔ complement_A defective) :=
by sorry

end complement_of_at_least_two_defective_l8_801


namespace faster_speed_calculation_l8_884

/-- Proves that given a 1200-mile trip, if driving at a certain speed saves 4 hours
    compared to driving at 50 miles per hour, then that certain speed is 60 miles per hour. -/
theorem faster_speed_calculation (trip_distance : ℝ) (original_speed : ℝ) (time_saved : ℝ) 
    (faster_speed : ℝ) : 
    trip_distance = 1200 → 
    original_speed = 50 → 
    time_saved = 4 → 
    trip_distance / original_speed - trip_distance / faster_speed = time_saved → 
    faster_speed = 60 := by
  sorry

#check faster_speed_calculation

end faster_speed_calculation_l8_884


namespace expression_evaluation_l8_800

theorem expression_evaluation (a b c d : ℝ) :
  d = c + 5 →
  c = b - 8 →
  b = a + 3 →
  a = 3 →
  a - 1 ≠ 0 →
  d - 6 ≠ 0 →
  c + 4 ≠ 0 →
  ((a + 3) / (a - 1)) * ((d - 3) / (d - 6)) * ((c + 9) / (c + 4)) = 0 :=
by sorry

end expression_evaluation_l8_800


namespace product_A_sample_size_l8_831

/-- Represents the ratio of quantities for products A, B, and C -/
def productRatio : Fin 3 → ℕ
| 0 => 2  -- Product A
| 1 => 3  -- Product B
| 2 => 5  -- Product C
| _ => 0  -- Unreachable case

/-- The total sample size -/
def sampleSize : ℕ := 80

/-- Calculates the number of items for a given product in the sample -/
def itemsInSample (product : Fin 3) : ℕ :=
  (sampleSize * productRatio product) / (productRatio 0 + productRatio 1 + productRatio 2)

theorem product_A_sample_size :
  itemsInSample 0 = 16 := by sorry

end product_A_sample_size_l8_831


namespace more_students_than_rabbits_l8_875

theorem more_students_than_rabbits : 
  let num_classrooms : ℕ := 6
  let students_per_classroom : ℕ := 22
  let rabbits_per_classroom : ℕ := 4
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_rabbits : ℕ := num_classrooms * rabbits_per_classroom
  total_students - total_rabbits = 108 := by
sorry

end more_students_than_rabbits_l8_875


namespace candy_bar_cost_is_25_l8_855

/-- The cost of a candy bar in cents -/
def candy_bar_cost : ℕ := sorry

/-- The cost of a piece of chocolate in cents -/
def chocolate_cost : ℕ := 75

/-- The cost of a pack of juice in cents -/
def juice_cost : ℕ := 50

/-- The number of quarters needed to buy the items -/
def quarters_needed : ℕ := 11

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of candy bars purchased -/
def candy_bars_bought : ℕ := 3

/-- The number of chocolate pieces purchased -/
def chocolates_bought : ℕ := 2

/-- The number of juice packs purchased -/
def juices_bought : ℕ := 1

theorem candy_bar_cost_is_25 : 
  candy_bar_cost = 25 :=
by
  have h1 : quarters_needed * quarter_value = 
    candy_bars_bought * candy_bar_cost + 
    chocolates_bought * chocolate_cost + 
    juices_bought * juice_cost := by sorry
  sorry

end candy_bar_cost_is_25_l8_855


namespace consecutive_integers_median_l8_893

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (h1 : n = 81) (h2 : sum = 9^5) :
  let median := sum / n
  median = 729 := by
sorry

end consecutive_integers_median_l8_893


namespace circle_diameter_l8_816

theorem circle_diameter (r : ℝ) (h : r > 0) : 
  3 * (2 * π * r) = 2 * (π * r^2) → 2 * r = 6 := by
sorry

end circle_diameter_l8_816


namespace expand_and_simplify_l8_804

theorem expand_and_simplify (x : ℝ) : (1 + x^3) * (1 - x^4) = 1 + x^3 - x^4 - x^7 := by
  sorry

end expand_and_simplify_l8_804


namespace quadratic_function_ratio_bound_l8_860

/-- A quadratic function f(x) = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The function value at x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The derivative of the quadratic function -/
def QuadraticFunction.derivative (f : QuadraticFunction) (x : ℝ) : ℝ :=
  2 * f.a * x + f.b

theorem quadratic_function_ratio_bound (f : QuadraticFunction) 
    (h1 : f.derivative 0 > 0)
    (h2 : ∀ x : ℝ, f.eval x ≥ 0) :
    f.eval 1 / f.derivative 0 ≥ 2 := by
  sorry

end quadratic_function_ratio_bound_l8_860


namespace geometric_arithmetic_sequence_problem_l8_878

theorem geometric_arithmetic_sequence_problem (x y z : ℝ) 
  (h1 : (12 * y)^2 = 9 * x * 15 * z)  -- 9x, 12y, 15z form a geometric sequence
  (h2 : 2 / y = 1 / x + 1 / z)        -- 1/x, 1/y, 1/z form an arithmetic sequence
  : x / z + z / x = 34 / 15 := by
  sorry

end geometric_arithmetic_sequence_problem_l8_878


namespace newspaper_reading_time_l8_876

/-- Represents Hank's daily reading habits and total weekly reading time -/
structure ReadingHabits where
  newspaper_time : ℕ  -- Time spent reading newspaper each weekday morning
  novel_time : ℕ      -- Time spent reading novel each weekday evening (60 minutes)
  weekday_count : ℕ   -- Number of weekdays (5)
  weekend_count : ℕ   -- Number of weekend days (2)
  total_time : ℕ      -- Total reading time in a week (810 minutes)

/-- Theorem stating that given Hank's reading habits, he spends 30 minutes reading the newspaper each morning -/
theorem newspaper_reading_time (h : ReadingHabits) 
  (h_novel : h.novel_time = 60)
  (h_weekday : h.weekday_count = 5)
  (h_weekend : h.weekend_count = 2)
  (h_total : h.total_time = 810) :
  h.newspaper_time = 30 := by
  sorry


end newspaper_reading_time_l8_876
