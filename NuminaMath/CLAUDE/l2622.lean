import Mathlib

namespace min_electricity_price_l2622_262214

theorem min_electricity_price (a : ℝ) (h_a : a > 0) :
  let f (x : ℝ) := (a + 0.2 * a / (x - 0.4)) * (x - 0.3)
  ∃ x_min : ℝ, x_min = 0.6 ∧
    (∀ x : ℝ, 0.55 ≤ x ∧ x ≤ 0.75 ∧ f x ≥ 0.6 * a → x ≥ x_min) :=
by sorry

end min_electricity_price_l2622_262214


namespace sum_four_digit_ending_zero_value_l2622_262264

/-- The sum of all four-digit positive integers ending in 0 -/
def sum_four_digit_ending_zero : ℕ :=
  let first_term := 1000
  let last_term := 9990
  let common_difference := 10
  let num_terms := (last_term - first_term) / common_difference + 1
  num_terms * (first_term + last_term) / 2

theorem sum_four_digit_ending_zero_value : 
  sum_four_digit_ending_zero = 4945500 := by
  sorry

end sum_four_digit_ending_zero_value_l2622_262264


namespace prob_no_female_ends_correct_l2622_262259

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The probability that neither end is a female student when arranging the students in a row -/
def prob_no_female_ends : ℚ := 1 / 5

theorem prob_no_female_ends_correct :
  (num_male.choose 2 * (total_students - 2).factorial) / total_students.factorial = prob_no_female_ends :=
sorry

end prob_no_female_ends_correct_l2622_262259


namespace extra_bananas_distribution_l2622_262299

theorem extra_bananas_distribution (total_children absent_children : ℕ) 
  (original_distribution : ℕ) (h1 : total_children = 610) 
  (h2 : absent_children = 305) (h3 : original_distribution = 2) : 
  (total_children * original_distribution) / (total_children - absent_children) - 
   original_distribution = 2 := by
  sorry

end extra_bananas_distribution_l2622_262299


namespace M_divisible_by_52_l2622_262257

/-- The number formed by concatenating integers from 1 to 51 -/
def M : ℕ :=
  -- We don't actually compute M, just define it conceptually
  sorry

/-- M is divisible by 52 -/
theorem M_divisible_by_52 : 52 ∣ M := by
  sorry

end M_divisible_by_52_l2622_262257


namespace max_sum_of_squares_l2622_262248

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17)
  (h2 : a * b + c + d = 85)
  (h3 : a * d + b * c = 180)
  (h4 : c * d = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 934 :=
sorry

end max_sum_of_squares_l2622_262248


namespace unique_consecutive_sum_21_l2622_262286

/-- A set of consecutive positive integers -/
def ConsecutiveSet (start : ℕ) (length : ℕ) : Set ℕ :=
  {n | start ≤ n ∧ n < start + length}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

/-- Predicate for a valid set of consecutive integers summing to 21 -/
def ValidSet (start : ℕ) (length : ℕ) : Prop :=
  length ≥ 2 ∧ ConsecutiveSum start length = 21

theorem unique_consecutive_sum_21 :
  ∃! p : ℕ × ℕ, ValidSet p.1 p.2 := by sorry

end unique_consecutive_sum_21_l2622_262286


namespace inequality_equivalence_l2622_262278

/-- The function f(x) = x^2 - 2x + 6 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 6

/-- Theorem stating that f(m+3) > f(2m) is equivalent to -1/3 < m < 3 -/
theorem inequality_equivalence (m : ℝ) : 
  f (m + 3) > f (2 * m) ↔ -1/3 < m ∧ m < 3 := by sorry

end inequality_equivalence_l2622_262278


namespace cruise_liner_travelers_l2622_262273

theorem cruise_liner_travelers :
  ∃ a : ℕ,
    250 ≤ a ∧ a ≤ 400 ∧
    a % 15 = 8 ∧
    a % 25 = 17 ∧
    (a = 292 ∨ a = 367) :=
by sorry

end cruise_liner_travelers_l2622_262273


namespace magician_marbles_problem_l2622_262268

theorem magician_marbles_problem (initial_red : ℕ) (initial_blue : ℕ) 
  (red_taken : ℕ) (blue_taken_multiplier : ℕ) :
  initial_red = 20 →
  initial_blue = 30 →
  red_taken = 3 →
  blue_taken_multiplier = 4 →
  (initial_red - red_taken) + (initial_blue - (blue_taken_multiplier * red_taken)) = 35 :=
by sorry

end magician_marbles_problem_l2622_262268


namespace function_value_at_2010_l2622_262265

def positive_reals : Set ℝ := {x : ℝ | x > 0}

def function_property (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 3)

theorem function_value_at_2010 (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ positive_reals, f x > 0)
  (h2 : function_property f) :
  f 2010 = 3 := by
  sorry

end function_value_at_2010_l2622_262265


namespace mixed_fraction_product_l2622_262282

theorem mixed_fraction_product (X Y : ℤ) : 
  (5 + 1 / X : ℚ) * (Y + 1 / 2 : ℚ) = 43 →
  5 < (5 + 1 / X : ℚ) →
  (5 + 1 / X : ℚ) ≤ 5.5 →
  X = 17 ∧ Y = 8 := by
sorry

end mixed_fraction_product_l2622_262282


namespace problem_solution_l2622_262206

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi/6) + Real.cos (x - Real.pi/3)

noncomputable def g (x : ℝ) : ℝ := 2 * (Real.sin (x/2))^2

theorem problem_solution (θ : ℝ) (k : ℤ) :
  (0 < θ ∧ θ < Real.pi/2) →  -- θ is in the first quadrant
  f θ = 3 * Real.sqrt 3 / 5 →
  g θ = 1/5 ∧
  (∀ x, f x ≥ g x ↔ ∃ k, 2 * k * Real.pi ≤ x ∧ x ≤ 2 * k * Real.pi + 2 * Real.pi / 3) :=
by sorry

end problem_solution_l2622_262206


namespace max_value_xy_difference_l2622_262210

theorem max_value_xy_difference (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  x^2 * y - y^2 * x ≤ 1/4 := by
  sorry

end max_value_xy_difference_l2622_262210


namespace fraction_equivalence_l2622_262262

theorem fraction_equivalence : 
  let x : ℚ := 13/2
  (4 + x) / (7 + x) = 7 / 9 := by sorry

end fraction_equivalence_l2622_262262


namespace mean_car_sales_l2622_262241

def car_sales : List Nat := [8, 3, 10, 4, 4, 4]

theorem mean_car_sales :
  (car_sales.sum : ℚ) / car_sales.length = 5.5 := by sorry

end mean_car_sales_l2622_262241


namespace largest_integer_negative_quadratic_l2622_262239

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 28 < 0 → n ≤ 6 :=
by sorry

end largest_integer_negative_quadratic_l2622_262239


namespace driver_net_hourly_rate_l2622_262205

/-- Calculates the driver's net hourly rate after deducting gas expenses -/
theorem driver_net_hourly_rate
  (travel_time : ℝ)
  (speed : ℝ)
  (gasoline_efficiency : ℝ)
  (gasoline_cost : ℝ)
  (driver_compensation : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : gasoline_efficiency = 25)
  (h4 : gasoline_cost = 2.5)
  (h5 : driver_compensation = 0.6)
  : (driver_compensation * speed * travel_time - 
     (speed * travel_time / gasoline_efficiency) * gasoline_cost) / travel_time = 25 :=
by sorry

end driver_net_hourly_rate_l2622_262205


namespace fly_revolutions_at_midnight_l2622_262277

/-- Represents a clock hand --/
inductive ClockHand
| Second
| Minute
| Hour

/-- Represents the state of the fly on the clock --/
structure FlyState where
  currentHand : ClockHand
  revolutions : ℕ

/-- The number of revolutions each hand makes in 12 hours --/
def handRevolutions (hand : ClockHand) : ℕ :=
  match hand with
  | ClockHand.Second => 720
  | ClockHand.Minute => 12
  | ClockHand.Hour => 1

/-- The total number of revolutions made by all hands in 12 hours --/
def totalRevolutions : ℕ :=
  (handRevolutions ClockHand.Second) +
  (handRevolutions ClockHand.Minute) +
  (handRevolutions ClockHand.Hour)

/-- Theorem stating that the fly makes 245 revolutions by midnight --/
theorem fly_revolutions_at_midnight :
  ∃ (finalState : FlyState),
    finalState.currentHand = ClockHand.Second →
    (∀ t, t ∈ Set.Icc (0 : ℝ) 12 →
      ¬ (∃ (h1 h2 h3 : ClockHand), h1 ≠ h2 ∧ h2 ≠ h3 ∧ h1 ≠ h3 ∧
        handRevolutions h1 * t = handRevolutions h2 * t ∧
        handRevolutions h2 * t = handRevolutions h3 * t)) →
    finalState.revolutions = 245 :=
sorry

end fly_revolutions_at_midnight_l2622_262277


namespace calculation_proof_l2622_262274

theorem calculation_proof : (-0.75) / 3 * (-2/5) = 1/10 := by
  sorry

end calculation_proof_l2622_262274


namespace cost_of_750_apples_l2622_262287

/-- The cost of buying a given number of apples, given the price and quantity of a bag of apples -/
def cost_of_apples (apples_per_bag : ℕ) (price_per_bag : ℕ) (total_apples : ℕ) : ℕ :=
  (total_apples / apples_per_bag) * price_per_bag

/-- Theorem: The cost of 750 apples is $120, given that a bag of 50 apples costs $8 -/
theorem cost_of_750_apples :
  cost_of_apples 50 8 750 = 120 := by
  sorry

#eval cost_of_apples 50 8 750

end cost_of_750_apples_l2622_262287


namespace value_of_a_l2622_262246

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 9) 
  (eq3 : c = 4) : 
  a = -1 := by sorry

end value_of_a_l2622_262246


namespace lance_reading_plan_l2622_262204

/-- Given a book with a certain number of pages, calculate the number of pages 
    to read on the third day to finish the book, given the pages read on the 
    first two days and constraints for the third day. -/
def pagesOnThirdDay (totalPages : ℕ) (day1Pages : ℕ) (day2Reduction : ℕ) : ℕ :=
  let day2Pages := day1Pages - day2Reduction
  let remainingPages := totalPages - (day1Pages + day2Pages)
  ((remainingPages + 9) / 10) * 10

theorem lance_reading_plan :
  pagesOnThirdDay 100 35 5 = 40 := by sorry

end lance_reading_plan_l2622_262204


namespace clock_hands_straight_in_day_l2622_262281

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents when the clock hands are straight -/
inductive ClockHandsStraight
  | coinciding
  | opposite

/-- Represents the position of the minute hand when the clock hands are straight -/
inductive MinuteHandPosition
  | zero_minutes
  | thirty_minutes

/-- The number of times the clock hands are straight in a day -/
def straight_hands_count : ℕ := 44

/-- Theorem stating that the clock hands are straight 44 times in a day -/
theorem clock_hands_straight_in_day :
  straight_hands_count = 44 :=
by sorry

end clock_hands_straight_in_day_l2622_262281


namespace arithmetic_sequence_probability_l2622_262240

/-- The set of numbers from which we select -/
def NumberSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 20}

/-- A function to check if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop := a + c = 2 * b

/-- The total number of ways to choose 3 numbers from the set -/
def totalSelections : ℕ := Nat.choose 20 3

/-- The number of ways to choose 3 numbers that form an arithmetic sequence -/
def arithmeticSequenceSelections : ℕ := 90

/-- The probability of selecting 3 numbers that form an arithmetic sequence -/
def probability : ℚ := arithmeticSequenceSelections / totalSelections

theorem arithmetic_sequence_probability :
  probability = 3 / 38 := by sorry

end arithmetic_sequence_probability_l2622_262240


namespace degree_of_polynomial_l2622_262216

/-- The degree of the polynomial (5x^3 + 7)^10 is 30 -/
theorem degree_of_polynomial (x : ℝ) : Polynomial.degree ((5 * X ^ 3 + 7 : Polynomial ℝ) ^ 10) = 30 := by
  sorry

end degree_of_polynomial_l2622_262216


namespace total_people_in_tribes_l2622_262202

theorem total_people_in_tribes (cannoneers : ℕ) (women : ℕ) (men : ℕ) : 
  cannoneers = 63 → 
  women = 2 * cannoneers → 
  men = 2 * women → 
  cannoneers + women + men = 378 := by
  sorry

end total_people_in_tribes_l2622_262202


namespace complex_modulus_cos_sin_three_l2622_262231

theorem complex_modulus_cos_sin_three : 
  let z : ℂ := Complex.mk (Real.cos 3) (Real.sin 3)
  |(Complex.abs z - 1)| < 1e-10 := by
sorry

end complex_modulus_cos_sin_three_l2622_262231


namespace ski_class_ratio_l2622_262235

theorem ski_class_ratio (b g : ℕ) : 
  b + g ≥ 66 →
  (b + 11 : ℤ) = (g - 13 : ℤ) →
  b ≠ 5 ∨ g ≠ 11 :=
by sorry

end ski_class_ratio_l2622_262235


namespace six_matchsticks_remain_l2622_262215

/-- The number of matchsticks remaining in a box after Elvis and Ralph make squares -/
def remaining_matchsticks (total : ℕ) (elvis_squares : ℕ) (ralph_squares : ℕ) : ℕ :=
  total - (4 * elvis_squares + 8 * ralph_squares)

/-- Theorem stating that 6 matchsticks remain when Elvis makes 5 squares and Ralph makes 3 squares from a box of 50 matchsticks -/
theorem six_matchsticks_remain : remaining_matchsticks 50 5 3 = 6 := by
  sorry

end six_matchsticks_remain_l2622_262215


namespace quadratic_roots_opposite_signs_l2622_262220

theorem quadratic_roots_opposite_signs (p : ℝ) (hp : p > 2) :
  let f (x : ℝ) := 5 * x^2 - 4 * (p + 3) * x + 4 - p^2
  let x₁ := p + 2
  let x₂ := (-p + 2) / 5
  (f x₁ = 0) ∧ (f x₂ = 0) ∧ (x₁ * x₂ < 0) := by
  sorry

#check quadratic_roots_opposite_signs

end quadratic_roots_opposite_signs_l2622_262220


namespace difference_of_squares_l2622_262234

theorem difference_of_squares (x : ℝ) : x^2 - 25 = (x + 5) * (x - 5) := by
  sorry

end difference_of_squares_l2622_262234


namespace sequence_matches_l2622_262211

/-- The sequence defined by a_n = 2^n - 1 -/
def a (n : ℕ) : ℕ := 2^n - 1

/-- The first four terms of the sequence match 1, 3, 7, 15 -/
theorem sequence_matches : 
  (a 1 = 1) ∧ (a 2 = 3) ∧ (a 3 = 7) ∧ (a 4 = 15) := by
  sorry

#eval a 1  -- Expected: 1
#eval a 2  -- Expected: 3
#eval a 3  -- Expected: 7
#eval a 4  -- Expected: 15

end sequence_matches_l2622_262211


namespace better_scores_seventh_grade_l2622_262295

/-- Represents the grade level of students -/
inductive Grade
  | Seventh
  | Eighth

/-- Represents statistical measures for a set of scores -/
structure ScoreStatistics where
  mean : ℝ
  median : ℝ
  mode : ℝ
  variance : ℝ

/-- The test scores for a grade -/
def scores (g : Grade) : List ℝ :=
  match g with
  | Grade.Seventh => [96, 85, 90, 86, 93, 92, 95, 81, 75, 81]
  | Grade.Eighth => [68, 95, 83, 93, 94, 75, 85, 95, 95, 77]

/-- The statistical measures for a grade -/
def statistics (g : Grade) : ScoreStatistics :=
  match g with
  | Grade.Seventh => ⟨87.4, 88, 81, 43.44⟩
  | Grade.Eighth => ⟨86, 89, 95, 89.2⟩

/-- Maximum possible score -/
def maxScore : ℝ := 100

theorem better_scores_seventh_grade :
  (statistics Grade.Seventh).median = 88 ∧
  (statistics Grade.Eighth).mode = 95 ∧
  (statistics Grade.Seventh).mean > (statistics Grade.Eighth).mean ∧
  (statistics Grade.Seventh).variance < (statistics Grade.Eighth).variance :=
by sorry

end better_scores_seventh_grade_l2622_262295


namespace inequality_solution_range_l2622_262225

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℤ, (x < 0 ∧ -4 * x - k ≤ 0) ↔ (x = -1 ∨ x = -2)) →
  (8 ≤ k ∧ k < 12) :=
sorry

end inequality_solution_range_l2622_262225


namespace intersection_equals_open_interval_l2622_262275

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- Define the open interval (2, 3)
def openInterval : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = openInterval := by
  sorry

end intersection_equals_open_interval_l2622_262275


namespace total_lemons_l2622_262256

/-- The number of lemons each person has -/
structure LemonCounts where
  levi : ℕ
  jayden : ℕ
  eli : ℕ
  ian : ℕ

/-- The conditions of the lemon problem -/
def lemon_problem (c : LemonCounts) : Prop :=
  c.levi = 5 ∧
  c.jayden = c.levi + 6 ∧
  c.jayden * 3 = c.eli ∧
  c.eli * 2 = c.ian

/-- The theorem stating the total number of lemons -/
theorem total_lemons (c : LemonCounts) :
  lemon_problem c → c.levi + c.jayden + c.eli + c.ian = 115 := by
  sorry

end total_lemons_l2622_262256


namespace inequality_proof_l2622_262249

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 2 * (x * y + y * z + z * x) = x * y * z) :
  (1 / ((x - 2) * (y - 2) * (z - 2))) + (8 / ((x + 2) * (y + 2) * (z + 2))) ≤ 1 / 32 :=
by sorry

end inequality_proof_l2622_262249


namespace units_digit_of_p_plus_two_l2622_262254

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Predicate for a number being even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem units_digit_of_p_plus_two (p : ℕ) 
  (h1 : p > 0)
  (h2 : isEven p)
  (h3 : unitsDigit p > 0)
  (h4 : unitsDigit (p^3) - unitsDigit (p^2) = 0) :
  unitsDigit (p + 2) = 8 := by
  sorry

end units_digit_of_p_plus_two_l2622_262254


namespace largest_m_is_138_l2622_262245

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_valid_pair (x y : ℕ) : Prop :=
  x < 15 ∧ y < 15 ∧ x ≠ y ∧ is_prime x ∧ is_prime y ∧ is_prime (x + y) ∧ is_prime (10 * x + y)

def m (x y : ℕ) : ℕ := x * y * (10 * x + y)

theorem largest_m_is_138 :
  ∀ x y : ℕ, is_valid_pair x y → m x y ≤ 138 :=
sorry

end largest_m_is_138_l2622_262245


namespace polynomial_identity_l2622_262242

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (2 - Real.sqrt 3 * x)^8 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  (a₀ + a₂ + a₄ + a₆ + a₈)^2 - (a₁ + a₃ + a₅ + a₇)^2 = 1 := by
sorry

end polynomial_identity_l2622_262242


namespace inequalities_proof_l2622_262293

theorem inequalities_proof (k : ℕ) (x : Fin k → ℝ) 
  (h_pos : ∀ i, x i > 0) (h_diff : ∀ i j, i ≠ j → x i ≠ x j) : 
  (Real.sqrt ((Finset.univ.sum (λ i => (x i)^2)) / k) > 
   (Finset.univ.sum (λ i => x i)) / k) ∧
  ((Finset.univ.sum (λ i => x i)) / k > 
   k / (Finset.univ.sum (λ i => 1 / (x i)))) := by
  sorry


end inequalities_proof_l2622_262293


namespace karlson_expenditure_can_exceed_2000_l2622_262261

theorem karlson_expenditure_can_exceed_2000 :
  ∃ (n m : ℕ), 25 * n + 340 * m > 2000 :=
by sorry

end karlson_expenditure_can_exceed_2000_l2622_262261


namespace smallest_stairs_solution_l2622_262217

theorem smallest_stairs_solution (n : ℕ) : 
  (n > 20 ∧ n % 6 = 5 ∧ n % 7 = 4) → n ≥ 53 :=
by sorry

end smallest_stairs_solution_l2622_262217


namespace inequality_proof_l2622_262253

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end inequality_proof_l2622_262253


namespace height_difference_l2622_262290

/-- Proves that the difference between Ron's height and Dean's height is 8 feet -/
theorem height_difference (water_depth : ℝ) (ron_height : ℝ) (dean_height : ℝ)
  (h1 : water_depth = 2 * dean_height)
  (h2 : ron_height = 14)
  (h3 : water_depth = 12) :
  ron_height - dean_height = 8 := by
  sorry

end height_difference_l2622_262290


namespace employee_pay_theorem_l2622_262252

def employee_pay (total : ℚ) (x_ratio : ℚ) (z_ratio : ℚ) :
  (ℚ × ℚ × ℚ) :=
  let y := total / (1 + x_ratio + z_ratio)
  let x := x_ratio * y
  let z := z_ratio * y
  (x, y, z)

theorem employee_pay_theorem (total : ℚ) (x_ratio : ℚ) (z_ratio : ℚ) :
  let (x, y, z) := employee_pay total x_ratio z_ratio
  (x + y + z = total) ∧ (x = x_ratio * y) ∧ (z = z_ratio * y) :=
by sorry

#eval employee_pay 934 1.2 0.8

end employee_pay_theorem_l2622_262252


namespace second_term_is_seven_general_formula_l2622_262219

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  monotone : Monotone a
  is_arithmetic : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d
  sum_first_three : a 1 + a 2 + a 3 = 21
  product_first_three : a 1 * a 2 * a 3 = 231

/-- The second term of the sequence is 7 -/
theorem second_term_is_seven (seq : ArithmeticSequence) : seq.a 2 = 7 := by
  sorry

/-- The general formula for the n-th term -/
theorem general_formula (seq : ArithmeticSequence) : ∀ n : ℕ, seq.a n = 4 * n - 1 := by
  sorry

end second_term_is_seven_general_formula_l2622_262219


namespace quadratic_inequality_solution_l2622_262213

theorem quadratic_inequality_solution (a b : ℚ) : 
  (∀ x, ax^2 - (a+1)*x + b < 0 ↔ 1 < x ∧ x < 5) → a + b = 6/5 := by
  sorry

end quadratic_inequality_solution_l2622_262213


namespace anne_age_when_paul_is_38_l2622_262244

/-- Given the initial ages of Paul and Anne in 2015, this theorem proves
    Anne's age when Paul is 38 years old. -/
theorem anne_age_when_paul_is_38 (paul_age_2015 anne_age_2015 : ℕ) 
    (h1 : paul_age_2015 = 11) 
    (h2 : anne_age_2015 = 14) : 
    anne_age_2015 + (38 - paul_age_2015) = 41 := by
  sorry

#check anne_age_when_paul_is_38

end anne_age_when_paul_is_38_l2622_262244


namespace sabrina_basil_leaves_l2622_262288

/-- The number of basil leaves Sabrina needs -/
def basil : ℕ := 12

/-- The number of sage leaves Sabrina needs -/
def sage : ℕ := 6

/-- The number of verbena leaves Sabrina needs -/
def verbena : ℕ := 11

/-- Theorem stating the correct number of basil leaves Sabrina needs -/
theorem sabrina_basil_leaves :
  (basil = 2 * sage) ∧
  (sage = verbena - 5) ∧
  (basil + sage + verbena = 29) ∧
  (basil = 12) := by
  sorry

#check sabrina_basil_leaves

end sabrina_basil_leaves_l2622_262288


namespace sum_inequality_l2622_262230

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) :
  a / (a^3 + b*c) + b / (b^3 + a*c) + c / (c^3 + a*b) > 3 := by
  sorry

end sum_inequality_l2622_262230


namespace intersection_point_satisfies_equations_intersection_point_unique_l2622_262209

/-- The system of linear equations representing two lines -/
def line1 (x y : ℚ) : Prop := 12 * x - 5 * y = 40
def line2 (x y : ℚ) : Prop := 8 * x + 2 * y = 20

/-- The intersection point of the two lines -/
def intersection_point : ℚ × ℚ := (45/16, -5/4)

/-- Theorem stating that the intersection point satisfies both equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
sorry

/-- Theorem stating that the intersection point is unique -/
theorem intersection_point_unique (x y : ℚ) :
  line1 x y → line2 x y → (x, y) = intersection_point :=
sorry

end intersection_point_satisfies_equations_intersection_point_unique_l2622_262209


namespace relay_game_error_l2622_262260

def initial_equation (x : ℝ) : Prop :=
  3 / (x - 1) = 1 - x / (x + 1)

def step1 (x : ℝ) : Prop :=
  3 * (x + 1) = (x + 1) * (x - 1) - x * (x - 1)

def step2 (x : ℝ) : Prop :=
  3 * x + 3 = x^2 + 1 - x^2 + x

def step3 (x : ℝ) : Prop :=
  3 * x - x = 1 - 3

def step4 (x : ℝ) : Prop :=
  x = -1

theorem relay_game_error :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 →
    (initial_equation x ↔ step1 x) ∧
    ¬(initial_equation x ↔ step2 x) ∧
    (initial_equation x ↔ step3 x) ∧
    (initial_equation x ↔ step4 x) :=
by sorry

end relay_game_error_l2622_262260


namespace exists_x0_f_less_than_g_l2622_262224

noncomputable def f (x : ℝ) : ℝ := 2017 * x + Real.sin x ^ 2017

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2017 + 2017 ^ x

theorem exists_x0_f_less_than_g :
  ∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > x₀, f x < g x := by sorry

end exists_x0_f_less_than_g_l2622_262224


namespace eighteen_team_tournament_games_l2622_262272

/-- Calculates the number of games in a knockout tournament with byes -/
def knockout_tournament_games (total_teams : ℕ) (bye_teams : ℕ) : ℕ :=
  total_teams - 1

/-- Theorem: A knockout tournament with 18 teams and 2 byes has 17 games -/
theorem eighteen_team_tournament_games :
  knockout_tournament_games 18 2 = 17 := by
  sorry

#eval knockout_tournament_games 18 2

end eighteen_team_tournament_games_l2622_262272


namespace prime_power_sum_l2622_262226

theorem prime_power_sum (p q : Nat) (m n : Nat) : 
  Nat.Prime p → Nat.Prime q → p < q →
  (∃ c : Nat, (p^(m+1) - 1) / (p - 1) = q^c) →
  (∃ d : Nat, (q^(n+1) - 1) / (q - 1) = p^d) →
  (p = 2 ∧ ∃ t : Nat, Nat.Prime t ∧ q = 2^t - 1) :=
by sorry

end prime_power_sum_l2622_262226


namespace hyperbola_eccentricity_l2622_262271

/-- Given an arithmetic sequence -1, a, b, m, 7, prove the eccentricity of x²/a² - y²/b² = 1 is √10 -/
theorem hyperbola_eccentricity (a b m : ℝ) : 
  (∃ d : ℝ, a = -1 + d ∧ b = a + d ∧ m = b + d ∧ 7 = m + d) →
  Real.sqrt ((b / a)^2 + 1) = Real.sqrt 10 := by
  sorry

end hyperbola_eccentricity_l2622_262271


namespace restaurant_group_size_restaurant_group_size_proof_l2622_262236

theorem restaurant_group_size (adult_meal_cost : ℕ) (kids_in_group : ℕ) (total_cost : ℕ) : ℕ :=
  let adults_in_group := total_cost / adult_meal_cost
  let total_people := adults_in_group + kids_in_group
  total_people

#check restaurant_group_size 8 2 72 = 11

theorem restaurant_group_size_proof 
  (adult_meal_cost : ℕ) 
  (kids_in_group : ℕ) 
  (total_cost : ℕ) 
  (h1 : adult_meal_cost = 8)
  (h2 : kids_in_group = 2)
  (h3 : total_cost = 72) :
  restaurant_group_size adult_meal_cost kids_in_group total_cost = 11 := by
  sorry

end restaurant_group_size_restaurant_group_size_proof_l2622_262236


namespace ice_cream_cost_calculation_l2622_262203

/-- Calculates the cost of each ice-cream cup given the order details and total amount paid --/
theorem ice_cream_cost_calculation
  (chapati_count : ℕ)
  (rice_count : ℕ)
  (vegetable_count : ℕ)
  (ice_cream_count : ℕ)
  (chapati_cost : ℕ)
  (rice_cost : ℕ)
  (vegetable_cost : ℕ)
  (total_paid : ℕ)
  (h1 : chapati_count = 16)
  (h2 : rice_count = 5)
  (h3 : vegetable_count = 7)
  (h4 : ice_cream_count = 6)
  (h5 : chapati_cost = 6)
  (h6 : rice_cost = 45)
  (h7 : vegetable_cost = 70)
  (h8 : total_paid = 961) :
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 25 := by
  sorry


end ice_cream_cost_calculation_l2622_262203


namespace exam_maximum_marks_l2622_262229

theorem exam_maximum_marks (ashley_marks : ℕ) (ashley_percentage : ℚ) :
  ashley_marks = 332 →
  ashley_percentage = 83 / 100 →
  (ashley_marks : ℚ) / ashley_percentage = 400 :=
by
  sorry

end exam_maximum_marks_l2622_262229


namespace base_eight_1563_to_ten_l2622_262238

def base_eight_to_ten (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem base_eight_1563_to_ten :
  base_eight_to_ten 1563 = 883 := by
  sorry

end base_eight_1563_to_ten_l2622_262238


namespace distance_covered_l2622_262294

/-- Proves that the total distance covered is 6 km given the specified conditions -/
theorem distance_covered (walking_speed running_speed : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 4)
  (h2 : running_speed = 8)
  (h3 : total_time = 2.25)
  (h4 : (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time)
  : total_distance = 6 :=
by
  sorry

#check distance_covered

end distance_covered_l2622_262294


namespace john_total_cost_l2622_262212

def nike_cost : ℝ := 150
def boot_cost : ℝ := 120
def tax_rate : ℝ := 0.1

def total_cost (nike : ℝ) (boot : ℝ) (tax : ℝ) : ℝ :=
  let subtotal := nike + boot
  let tax_amount := subtotal * tax
  subtotal + tax_amount

theorem john_total_cost :
  total_cost nike_cost boot_cost tax_rate = 297 :=
sorry

end john_total_cost_l2622_262212


namespace brand_d_highest_sales_l2622_262221

/-- Represents the sales volume of a brand -/
structure BrandSales where
  name : String
  sales : ℕ

/-- Theorem: Brand D has the highest sales volume -/
theorem brand_d_highest_sales (total : ℕ) (a b c d : BrandSales) :
  total = 100 ∧
  a.name = "A" ∧ a.sales = 15 ∧
  b.name = "B" ∧ b.sales = 30 ∧
  c.name = "C" ∧ c.sales = 12 ∧
  d.name = "D" ∧ d.sales = 43 →
  d.sales ≥ a.sales ∧ d.sales ≥ b.sales ∧ d.sales ≥ c.sales :=
by sorry

end brand_d_highest_sales_l2622_262221


namespace hyperbola_standard_equation_l2622_262279

/-- The standard equation of a hyperbola sharing a focus with the parabola x² = 8y and having eccentricity 2 -/
theorem hyperbola_standard_equation :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ x₀ y₀ : ℝ, x₀^2 = 8*y₀ ∧ (x₀, y₀) = (0, 2)) →
  (a = 1 ∧ b^2 = 3) →
  ∀ x y : ℝ, y^2 - x^2 / 3 = 1 :=
by sorry

end hyperbola_standard_equation_l2622_262279


namespace fish_population_estimate_l2622_262227

theorem fish_population_estimate 
  (initially_marked : ℕ) 
  (second_catch : ℕ) 
  (marked_in_second : ℕ) 
  (h1 : initially_marked = 30)
  (h2 : second_catch = 50)
  (h3 : marked_in_second = 2) :
  (initially_marked * second_catch) / marked_in_second = 750 :=
by
  sorry

#check fish_population_estimate

end fish_population_estimate_l2622_262227


namespace smallest_yellow_candy_quantity_l2622_262266

def red_candy_cost : ℕ := 8
def green_candy_cost : ℕ := 12
def blue_candy_cost : ℕ := 15
def yellow_candy_cost : ℕ := 24

def red_candy_quantity : ℕ := 10
def green_candy_quantity : ℕ := 18
def blue_candy_quantity : ℕ := 20

def red_total_cost : ℕ := red_candy_cost * red_candy_quantity
def green_total_cost : ℕ := green_candy_cost * green_candy_quantity
def blue_total_cost : ℕ := blue_candy_cost * blue_candy_quantity

theorem smallest_yellow_candy_quantity :
  ∃ (n : ℕ), n > 0 ∧
  (yellow_candy_cost * n) % red_total_cost = 0 ∧
  (yellow_candy_cost * n) % green_total_cost = 0 ∧
  (yellow_candy_cost * n) % blue_total_cost = 0 ∧
  ∀ (m : ℕ), m > 0 →
    (yellow_candy_cost * m) % red_total_cost = 0 →
    (yellow_candy_cost * m) % green_total_cost = 0 →
    (yellow_candy_cost * m) % blue_total_cost = 0 →
    m ≥ n :=
by sorry

end smallest_yellow_candy_quantity_l2622_262266


namespace sufficient_not_necessary_condition_l2622_262284

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧ (∃ x : ℝ, x ≤ 1 ∧ x^2 > 1) := by
  sorry

end sufficient_not_necessary_condition_l2622_262284


namespace sugar_amount_proof_l2622_262200

/-- Recipe proportions and conversion factors -/
def butter_to_flour : ℚ := 5 / 7
def salt_to_flour : ℚ := 3 / 1.5
def sugar_to_flour : ℚ := 2 / 2.5
def butter_multiplier : ℚ := 4
def salt_multiplier : ℚ := 3.5
def sugar_multiplier : ℚ := 3
def butter_used : ℚ := 12
def ounce_to_gram : ℚ := 28.35
def cup_flour_to_gram : ℚ := 125
def tsp_salt_to_gram : ℚ := 5
def tbsp_sugar_to_gram : ℚ := 15

/-- Theorem stating that the amount of sugar needed is 604.8 grams -/
theorem sugar_amount_proof :
  let flour_cups := butter_used / butter_to_flour
  let flour_grams := flour_cups * cup_flour_to_gram
  let sugar_tbsp := (sugar_to_flour * flour_cups * sugar_multiplier)
  sugar_tbsp * tbsp_sugar_to_gram = 604.8 := by
  sorry

end sugar_amount_proof_l2622_262200


namespace outfit_combinations_l2622_262255

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 6

/-- The number of different types of clothing items -/
def num_items : ℕ := 4

/-- The number of valid outfit combinations -/
def valid_combinations : ℕ := num_colors * (num_colors - 1) * (num_colors - 2) * (num_colors - 3)

theorem outfit_combinations :
  valid_combinations = 360 :=
sorry

end outfit_combinations_l2622_262255


namespace fraction_equality_l2622_262223

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 4 * y) / (4 * x - y) = 39 / 37 := by
  sorry

end fraction_equality_l2622_262223


namespace parabola_vertex_l2622_262250

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 3 * (x - 2)^2 - 5

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, -5)

/-- Theorem: The vertex of the parabola y = 3(x-2)^2 - 5 is (2, -5) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end parabola_vertex_l2622_262250


namespace chess_game_theorem_l2622_262251

/-- Represents a three-player turn-based game system -/
structure GameSystem where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The game system satisfies the conditions of the problem -/
def valid_game_system (g : GameSystem) : Prop :=
  g.total_games = 27 ∧
  g.player1_games = 27 ∧
  g.player2_games = 13 ∧
  g.player3_games = g.total_games - g.player2_games

theorem chess_game_theorem (g : GameSystem) (h : valid_game_system g) :
  g.player3_games = 14 := by
  sorry


end chess_game_theorem_l2622_262251


namespace projection_relations_l2622_262233

-- Define a plane
structure Plane where
  -- Add necessary fields

-- Define a line
structure Line where
  -- Add necessary fields

-- Define the projection of a line onto a plane
def project (l : Line) (p : Plane) : Line :=
  sorry

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  sorry

-- Define perpendicular lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

-- Define coincident lines
def coincident (l1 l2 : Line) : Prop :=
  sorry

theorem projection_relations (α : Plane) (m n : Line) :
  let m1 := project m α
  let n1 := project n α
  -- All four propositions are false
  (¬ (parallel m1 n1 → parallel m n)) ∧
  (¬ (parallel m n → (parallel m1 n1 ∨ coincident m1 n1))) ∧
  (¬ (perpendicular m1 n1 → perpendicular m n)) ∧
  (¬ (perpendicular m n → perpendicular m1 n1)) :=
by
  sorry

end projection_relations_l2622_262233


namespace triangle_properties_l2622_262247

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- Define the altitude line
def altitude_line (x y : ℝ) : Prop := 2 * x - 3 * y + 14 = 0

-- Define the equidistant lines
def equidistant_line1 (x y : ℝ) : Prop := 7 * x - 6 * y + 4 = 0
def equidistant_line2 (x y : ℝ) : Prop := 3 * x + 2 * y - 44 = 0

-- Theorem statement
theorem triangle_properties :
  -- 1. The altitude from A to BC
  (∀ x y : ℝ, altitude_line x y ↔ 
    (x - A.1) * (B.1 - C.1) + (y - A.2) * (B.2 - C.2) = 0 ∧ 
    ∃ t : ℝ, x = A.1 + t * (B.2 - C.2) ∧ y = A.2 - t * (B.1 - C.1)) ∧
  -- 2. The lines through B equidistant from A and C
  (∀ x y : ℝ, (equidistant_line1 x y ∨ equidistant_line2 x y) ↔
    abs ((y - A.2) * (B.1 - A.1) - (x - A.1) * (B.2 - A.2)) = 
    abs ((y - C.2) * (B.1 - C.1) - (x - C.1) * (B.2 - C.2))) :=
sorry


end triangle_properties_l2622_262247


namespace equation_solution_l2622_262237

theorem equation_solution :
  ∃ x : ℝ, (64 : ℝ) ^ (3 * x + 1) = (16 : ℝ) ^ (4 * x - 5) ∧ x = -13 := by
  sorry

end equation_solution_l2622_262237


namespace girls_attending_sports_event_l2622_262280

theorem girls_attending_sports_event (total_students : ℕ) (attending_students : ℕ) 
  (h1 : total_students = 1500)
  (h2 : attending_students = 900)
  (h3 : ∃ (girls boys : ℕ), girls + boys = total_students ∧ 
                             (girls / 2 : ℚ) + (3 * boys / 5 : ℚ) = attending_students) :
  ∃ (girls : ℕ), girls / 2 = 500 := by
sorry

end girls_attending_sports_event_l2622_262280


namespace line_segment_ratio_l2622_262296

/-- Given points P, Q, R, and S on a straight line in that order,
    with PQ = 3, QR = 7, and PS = 20, prove that PR:QS = 1 -/
theorem line_segment_ratio (P Q R S : ℝ) 
  (h_order : P < Q ∧ Q < R ∧ R < S)
  (h_PQ : Q - P = 3)
  (h_QR : R - Q = 7)
  (h_PS : S - P = 20) :
  (R - P) / (S - Q) = 1 := by
  sorry

end line_segment_ratio_l2622_262296


namespace bacteria_growth_l2622_262269

-- Define the division rate of bacteria
def division_rate : ℕ := 10

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 1

-- Define the time passed in minutes
def time_passed : ℕ := 120

-- Define the function to calculate the number of bacteria
def num_bacteria (t : ℕ) : ℕ := 2 ^ (t / division_rate)

-- Theorem to prove
theorem bacteria_growth :
  num_bacteria time_passed = 2^12 :=
by sorry

end bacteria_growth_l2622_262269


namespace candy_division_problem_l2622_262291

theorem candy_division_problem :
  ∃! x : ℕ, 120 ≤ x ∧ x ≤ 150 ∧ x % 5 = 2 ∧ x % 6 = 5 ∧ x = 137 := by
  sorry

end candy_division_problem_l2622_262291


namespace max_volume_right_prism_l2622_262207

theorem max_volume_right_prism (b c : ℝ) (h1 : b + c = 8) (h2 : b > 0) (h3 : c > 0) :
  let volume := fun x => (1/2) * b * x^2
  let x := Real.sqrt (64 - 16*b)
  (∀ y, volume y ≤ volume x) ∧ volume x = 32 := by
  sorry

end max_volume_right_prism_l2622_262207


namespace smallest_five_digit_divisible_by_smallest_primes_l2622_262243

def smallest_five_digit_number_divisible_by_smallest_primes : ℕ := 11550

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def is_divisible_by_smallest_primes (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0

theorem smallest_five_digit_divisible_by_smallest_primes :
  (is_five_digit smallest_five_digit_number_divisible_by_smallest_primes) ∧
  (is_divisible_by_smallest_primes smallest_five_digit_number_divisible_by_smallest_primes) ∧
  (∀ m : ℕ, m < smallest_five_digit_number_divisible_by_smallest_primes →
    ¬(is_five_digit m ∧ is_divisible_by_smallest_primes m)) :=
by sorry

end smallest_five_digit_divisible_by_smallest_primes_l2622_262243


namespace inscribed_circle_radius_squared_l2622_262228

/-- A circle inscribed in quadrilateral EFGH -/
structure InscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The circle is tangent to EF at R -/
  tangent_EF : True
  /-- The circle is tangent to GH at S -/
  tangent_GH : True
  /-- The circle is tangent to EH at T -/
  tangent_EH : True
  /-- ER = 25 -/
  ER : r = 25
  /-- RF = 35 -/
  RF : r = 35
  /-- GS = 40 -/
  GS : r = 40
  /-- SH = 20 -/
  SH : r = 20
  /-- ET = 45 -/
  ET : r = 45

/-- The square of the radius of the inscribed circle is 3600 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle) : c.r^2 = 3600 := by
  sorry

end inscribed_circle_radius_squared_l2622_262228


namespace long_tennis_players_l2622_262222

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 38 →
  football = 26 →
  both = 17 →
  neither = 9 →
  ∃ long_tennis : ℕ, long_tennis = 20 ∧ total = football + long_tennis - both + neither :=
by sorry

end long_tennis_players_l2622_262222


namespace crayon_selection_theorem_l2622_262270

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of metallic crayons in the box -/
def metallic_crayons : ℕ := 2

/-- The number of crayons to be selected -/
def selection_size : ℕ := 5

/-- The number of ways to select crayons with the given conditions -/
def selection_ways : ℕ := metallic_crayons * choose (total_crayons - metallic_crayons) (selection_size - 1)

theorem crayon_selection_theorem : selection_ways = 1430 := by sorry

end crayon_selection_theorem_l2622_262270


namespace ratio_equality_implies_fraction_value_l2622_262208

theorem ratio_equality_implies_fraction_value
  (a b c : ℝ)
  (h : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  (a + b) / (b - c) = -7 :=
by sorry

end ratio_equality_implies_fraction_value_l2622_262208


namespace first_number_value_l2622_262283

-- Define the custom operation
def custom_op (m n : ℤ) : ℤ := n^2 - m

-- Theorem statement
theorem first_number_value :
  ∃ x : ℤ, custom_op x 3 = 6 ∧ x = 3 :=
by
  sorry

end first_number_value_l2622_262283


namespace cube_sum_reciprocal_l2622_262298

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end cube_sum_reciprocal_l2622_262298


namespace largest_valid_number_nine_zero_nine_nine_is_valid_nine_zero_nine_nine_is_largest_l2622_262267

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n / 10) % 10 = (n / 1000) % 10 + (n / 100) % 10 ∧
  n % 10 = (n / 100) % 10 + (n / 10) % 10

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 9099 :=
by sorry

theorem nine_zero_nine_nine_is_valid :
  is_valid_number 9099 :=
by sorry

theorem nine_zero_nine_nine_is_largest :
  ∀ n : ℕ, is_valid_number n → n = 9099 ∨ n < 9099 :=
by sorry

end largest_valid_number_nine_zero_nine_nine_is_valid_nine_zero_nine_nine_is_largest_l2622_262267


namespace relationship_abc_l2622_262297

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.rpow 0.8 0.8)
  (hb : b = Real.rpow 0.8 1.2)
  (hc : c = Real.rpow 1.2 0.8) : 
  c > a ∧ a > b :=
sorry

end relationship_abc_l2622_262297


namespace composition_of_even_is_even_l2622_262276

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by
  sorry

end composition_of_even_is_even_l2622_262276


namespace total_bills_count_l2622_262263

/-- Represents the number of bills and their total value -/
structure WalletContents where
  num_five_dollar_bills : ℕ
  num_ten_dollar_bills : ℕ
  total_value : ℕ

/-- Theorem stating that given the conditions, the total number of bills is 12 -/
theorem total_bills_count (w : WalletContents) 
  (h1 : w.num_five_dollar_bills = 4)
  (h2 : w.total_value = 100)
  (h3 : w.total_value = 5 * w.num_five_dollar_bills + 10 * w.num_ten_dollar_bills) :
  w.num_five_dollar_bills + w.num_ten_dollar_bills = 12 := by
  sorry

end total_bills_count_l2622_262263


namespace father_daughter_age_sum_l2622_262292

theorem father_daughter_age_sum :
  ∀ (father_age daughter_age : ℕ),
    father_age - daughter_age = 22 →
    daughter_age = 16 →
    father_age + daughter_age = 54 :=
by
  sorry

end father_daughter_age_sum_l2622_262292


namespace cos_sixty_degrees_l2622_262232

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end cos_sixty_degrees_l2622_262232


namespace circle_sum_center_radius_l2622_262218

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 2*x - 8*y - 7 = -y^2 - 6*x

-- Define the center and radius
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_sum_center_radius :
  ∃ (a b r : ℝ), is_center_radius a b r ∧ a + b + r = Real.sqrt 39 := by
  sorry

end circle_sum_center_radius_l2622_262218


namespace inequality_system_solution_condition_l2622_262258

theorem inequality_system_solution_condition (m : ℝ) :
  (∃ x : ℝ, x - 2 * m < 0 ∧ x + m > 2) → m > 2 / 3 :=
by sorry

end inequality_system_solution_condition_l2622_262258


namespace grid_filling_ways_l2622_262201

/-- Represents a 6x6 grid with special cells -/
structure Grid :=
  (size : Nat)
  (specialCells : Nat)
  (valuesPerSpecialCell : Nat)

/-- Calculates the number of ways to fill the grid -/
def numberOfWays (g : Grid) : Nat :=
  (g.valuesPerSpecialCell ^ g.specialCells) ^ 4

/-- Theorem: The number of ways to fill the grid is 16 -/
theorem grid_filling_ways (g : Grid) 
  (h1 : g.size = 6)
  (h2 : g.specialCells = 4)
  (h3 : g.valuesPerSpecialCell = 2) :
  numberOfWays g = 16 := by
  sorry

#eval numberOfWays { size := 6, specialCells := 4, valuesPerSpecialCell := 2 }

end grid_filling_ways_l2622_262201


namespace quadratic_root_discriminant_square_relation_l2622_262285

theorem quadratic_root_discriminant_square_relation 
  (a b c t : ℝ) (h1 : a ≠ 0) (h2 : a * t^2 + b * t + c = 0) :
  b^2 - 4*a*c = (2*a*t + b)^2 := by sorry

end quadratic_root_discriminant_square_relation_l2622_262285


namespace cut_cube_total_count_l2622_262289

/-- Represents a cube that has been cut into smaller cubes -/
structure CutCube where
  /-- The number of smaller cubes along each edge of the original cube -/
  edge_count : ℕ
  /-- The number of smaller cubes painted on exactly two faces -/
  two_face_painted : ℕ

/-- Theorem stating that if a cube is cut such that 12 smaller cubes are painted on 2 faces,
    then the total number of smaller cubes is 27 -/
theorem cut_cube_total_count (c : CutCube) (h : c.two_face_painted = 12) : 
  c.edge_count ^ 3 = 27 := by
  sorry

#check cut_cube_total_count

end cut_cube_total_count_l2622_262289
