import Mathlib

namespace NUMINAMATH_CALUDE_mod_seven_difference_powers_l156_15696

theorem mod_seven_difference_powers : (81^1814 - 25^1814) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_seven_difference_powers_l156_15696


namespace NUMINAMATH_CALUDE_f_properties_l156_15669

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (10 - 2 * x) / Real.log (1/2)

-- Theorem statement
theorem f_properties :
  -- 1. Domain of f(x) is (-∞, 5)
  (∀ x, f x ≠ 0 → x < 5) ∧
  -- 2. f(x) is increasing on its domain
  (∀ x y, x < y → x < 5 → y < 5 → f x < f y) ∧
  -- 3. Maximum value of m for which f(x) ≥ (1/2)ˣ + m holds for all x ∈ [3, 4] is -17/8
  (∀ m, (∀ x, x ∈ Set.Icc 3 4 → f x ≥ (1/2)^x + m) → m ≤ -17/8) ∧
  (∃ x, x ∈ Set.Icc 3 4 ∧ f x = (1/2)^x + (-17/8)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l156_15669


namespace NUMINAMATH_CALUDE_barons_claim_impossible_l156_15679

/-- Represents the number of games played by each participant -/
def GameDistribution := List ℕ

/-- A chess tournament with the given rules -/
structure ChessTournament where
  participants : ℕ
  initialGamesPerParticipant : ℕ
  claimedDistribution : GameDistribution

/-- Checks if a game distribution is valid for the given tournament rules -/
def isValidDistribution (t : ChessTournament) (d : GameDistribution) : Prop :=
  d.length = t.participants ∧
  d.sum = t.participants * t.initialGamesPerParticipant + 2 * (d.sum / 2 - t.participants * t.initialGamesPerParticipant / 2)

/-- The specific tournament described in the problem -/
def baronsTournament : ChessTournament where
  participants := 8
  initialGamesPerParticipant := 7
  claimedDistribution := [11, 11, 10, 8, 8, 8, 7, 7]

/-- Theorem stating that the Baron's claim is impossible -/
theorem barons_claim_impossible :
  ¬ isValidDistribution baronsTournament baronsTournament.claimedDistribution :=
sorry

end NUMINAMATH_CALUDE_barons_claim_impossible_l156_15679


namespace NUMINAMATH_CALUDE_stuffed_animal_sales_difference_l156_15610

theorem stuffed_animal_sales_difference (thor jake quincy : ℕ) 
  (h1 : jake = thor + 10)
  (h2 : quincy = 10 * thor)
  (h3 : quincy = 200) :
  quincy - jake = 170 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animal_sales_difference_l156_15610


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l156_15688

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 + a 3 = 4) →
  (a 2 + a 3 + a 4 = -2) →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 7/8) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l156_15688


namespace NUMINAMATH_CALUDE_largest_sum_simplification_l156_15630

theorem largest_sum_simplification : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/7, 1/3 + 1/8]
  (∀ s ∈ sums, s ≤ 1/3 + 1/2) ∧ 
  (1/3 + 1/2 = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_simplification_l156_15630


namespace NUMINAMATH_CALUDE_sandwich_filler_percentage_l156_15659

/-- Given a sandwich with a total weight of 180 grams and filler weight of 45 grams,
    prove that the percentage of the sandwich that is not filler is 75%. -/
theorem sandwich_filler_percentage (total_weight filler_weight : ℝ) 
    (h1 : total_weight = 180)
    (h2 : filler_weight = 45) :
    (total_weight - filler_weight) / total_weight = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_filler_percentage_l156_15659


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l156_15657

theorem smallest_solution_floor_equation :
  ∀ x : ℝ, (⌊x⌋ : ℝ) = 7 + 50 * (x - ⌊x⌋) → x ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l156_15657


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l156_15651

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem fourth_term_of_geometric_progression 
  (a : ℝ) (r : ℝ) (h1 : a > 0) (h2 : r > 0) :
  geometric_progression a r 1 = 4 ∧ 
  geometric_progression a r 2 = Real.sqrt 4 ∧ 
  geometric_progression a r 3 = 4^(1/4) →
  geometric_progression a r 4 = 4^(1/8) := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l156_15651


namespace NUMINAMATH_CALUDE_purchase_cost_l156_15658

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda in dollars -/
def soda_cost : ℕ := 1

/-- The number of sandwiches to purchase -/
def num_sandwiches : ℕ := 6

/-- The number of sodas to purchase -/
def num_sodas : ℕ := 10

/-- The total cost of the purchase in dollars -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem purchase_cost : total_cost = 34 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l156_15658


namespace NUMINAMATH_CALUDE_complement_N_Nstar_is_finite_l156_15627

def complement_N_Nstar : Set ℕ := {0}

theorem complement_N_Nstar_is_finite :
  Set.Finite complement_N_Nstar :=
sorry

end NUMINAMATH_CALUDE_complement_N_Nstar_is_finite_l156_15627


namespace NUMINAMATH_CALUDE_area_covered_by_strips_l156_15637

/-- Represents a rectangular strip -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℝ := s.length * s.width

/-- Calculates the total area of strips without considering overlaps -/
def totalAreaNoOverlap (strips : List Strip) : ℝ :=
  (strips.map stripArea).sum

/-- Represents an overlap between two strips -/
structure Overlap where
  length : ℝ
  width : ℝ

/-- Calculates the area of an overlap -/
def overlapArea (o : Overlap) : ℝ := o.length * o.width

/-- Calculates the total area of overlaps -/
def totalOverlapArea (overlaps : List Overlap) : ℝ :=
  (overlaps.map overlapArea).sum

/-- Theorem: The area covered by five strips with given dimensions and overlaps is 58 -/
theorem area_covered_by_strips :
  let strips : List Strip := List.replicate 5 ⟨12, 1⟩
  let overlaps : List Overlap := List.replicate 4 ⟨0.5, 1⟩
  totalAreaNoOverlap strips - totalOverlapArea overlaps = 58 := by
  sorry

end NUMINAMATH_CALUDE_area_covered_by_strips_l156_15637


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l156_15609

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + 3 * y = 4) :
  y = (4 - 2 * x) / 3 := by
sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l156_15609


namespace NUMINAMATH_CALUDE_baseball_cleats_price_l156_15602

/-- Proves that the price of each pair of baseball cleats is $10 -/
theorem baseball_cleats_price :
  let cards_price : ℝ := 25
  let bat_price : ℝ := 10
  let glove_original_price : ℝ := 30
  let glove_discount_percentage : ℝ := 0.2
  let total_sales : ℝ := 79
  let num_cleats_pairs : ℕ := 2

  let glove_sale_price : ℝ := glove_original_price * (1 - glove_discount_percentage)
  let non_cleats_sales : ℝ := cards_price + bat_price + glove_sale_price
  let cleats_total_price : ℝ := total_sales - non_cleats_sales
  let cleats_pair_price : ℝ := cleats_total_price / num_cleats_pairs

  cleats_pair_price = 10 := by
    sorry

end NUMINAMATH_CALUDE_baseball_cleats_price_l156_15602


namespace NUMINAMATH_CALUDE_log_relationship_l156_15682

theorem log_relationship (c d x : ℝ) (hc : c > 0) (hd : d > 0) (hx : x > 0 ∧ x ≠ 1) :
  6 * (Real.log x / Real.log c)^2 + 5 * (Real.log x / Real.log d)^2 = 12 * (Real.log x)^2 / (Real.log c * Real.log d) →
  d = c^(5 / (6 + Real.sqrt 6)) ∨ d = c^(5 / (6 - Real.sqrt 6)) :=
by sorry

end NUMINAMATH_CALUDE_log_relationship_l156_15682


namespace NUMINAMATH_CALUDE_esteban_exercise_days_l156_15634

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of minutes in an hour -/
def minutesInHour : ℕ := 60

/-- Represents Natasha's daily exercise time in minutes -/
def natashasDailyExercise : ℕ := 30

/-- Represents Esteban's daily exercise time in minutes -/
def estebansDailyExercise : ℕ := 10

/-- Represents the total exercise time of Natasha and Esteban in hours -/
def totalExerciseTime : ℕ := 5

/-- Theorem stating that Esteban exercised for 9 days -/
theorem esteban_exercise_days : 
  ∃ (estebanDays : ℕ), 
    estebanDays * estebansDailyExercise + 
    daysInWeek * natashasDailyExercise = 
    totalExerciseTime * minutesInHour ∧ 
    estebanDays = 9 := by
  sorry

end NUMINAMATH_CALUDE_esteban_exercise_days_l156_15634


namespace NUMINAMATH_CALUDE_quadrant_passing_implies_negative_m_l156_15625

/-- A linear function passing through the second, third, and fourth quadrants -/
structure QuadrantPassingFunction where
  m : ℝ
  passes_second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = -3 * x + m
  passes_third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = -3 * x + m
  passes_fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = -3 * x + m

/-- Theorem: If a linear function y = -3x + m passes through the second, third, and fourth quadrants, then m is negative -/
theorem quadrant_passing_implies_negative_m (f : QuadrantPassingFunction) : f.m < 0 :=
  sorry

end NUMINAMATH_CALUDE_quadrant_passing_implies_negative_m_l156_15625


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_11_l156_15691

def is_smallest_positive_integer_ending_in_6_divisible_by_11 (n : ℕ) : Prop :=
  n > 0 ∧ n % 10 = 6 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 6 → m % 11 = 0 → m ≥ n

theorem smallest_positive_integer_ending_in_6_divisible_by_11 :
  is_smallest_positive_integer_ending_in_6_divisible_by_11 116 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_11_l156_15691


namespace NUMINAMATH_CALUDE_sum_negative_implies_at_most_one_positive_l156_15621

theorem sum_negative_implies_at_most_one_positive (a b : ℚ) :
  a + b < 0 → (0 < a ∧ 0 < b) → False := by sorry

end NUMINAMATH_CALUDE_sum_negative_implies_at_most_one_positive_l156_15621


namespace NUMINAMATH_CALUDE_impossible_all_defective_l156_15641

/-- Given 10 products with 2 defective ones, the probability of selecting 3 defective products
    when randomly choosing 3 is zero. -/
theorem impossible_all_defective (total : Nat) (defective : Nat) (selected : Nat)
    (h1 : total = 10)
    (h2 : defective = 2)
    (h3 : selected = 3) :
  Nat.choose defective selected / Nat.choose total selected = 0 := by
  sorry

end NUMINAMATH_CALUDE_impossible_all_defective_l156_15641


namespace NUMINAMATH_CALUDE_expand_cube_105_plus_1_l156_15600

theorem expand_cube_105_plus_1 : 105^3 + 3*(105^2) + 3*105 + 1 = 11856 := by
  sorry

end NUMINAMATH_CALUDE_expand_cube_105_plus_1_l156_15600


namespace NUMINAMATH_CALUDE_axis_of_symmetry_for_quadratic_with_roots_1_and_5_l156_15693

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem axis_of_symmetry_for_quadratic_with_roots_1_and_5 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x, quadratic a b c x = 0 ↔ x = 1 ∨ x = 5) →
  (∃ k, ∀ x, quadratic a b c (k + x) = quadratic a b c (k - x)) ∧
  (∀ k, (∀ x, quadratic a b c (k + x) = quadratic a b c (k - x)) → k = 3) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_for_quadratic_with_roots_1_and_5_l156_15693


namespace NUMINAMATH_CALUDE_investment_problem_l156_15631

/-- Calculates the final amount for a simple interest investment -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Given conditions and proof goal -/
theorem investment_problem (rate : ℝ) :
  simpleInterest 150 rate 6 = 210 →
  simpleInterest 200 rate 3 = 240 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l156_15631


namespace NUMINAMATH_CALUDE_six_people_charity_arrangements_l156_15620

/-- The number of ways to distribute n people into 2 charity activities,
    with each activity accommodating no more than 4 people -/
def charity_arrangements (n : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating that there are 50 ways to distribute 6 people
    into 2 charity activities with the given constraints -/
theorem six_people_charity_arrangements :
  charity_arrangements 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_six_people_charity_arrangements_l156_15620


namespace NUMINAMATH_CALUDE_retail_price_increase_l156_15652

theorem retail_price_increase (W R : ℝ) 
  (h : 0.80 * R = 1.44000000000000014 * W) : 
  (R - W) / W * 100 = 80.000000000000017 :=
by sorry

end NUMINAMATH_CALUDE_retail_price_increase_l156_15652


namespace NUMINAMATH_CALUDE_dividing_line_slope_absolute_value_is_one_l156_15683

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line that equally divides the total area of the circles --/
structure DividingLine where
  slope : ℝ
  passes_through : ℝ × ℝ

/-- The problem setup --/
def problem_setup : (Circle × Circle × Circle) × DividingLine := 
  let c1 : Circle := ⟨(10, 90), 4⟩
  let c2 : Circle := ⟨(15, 70), 4⟩
  let c3 : Circle := ⟨(20, 80), 4⟩
  let line : DividingLine := ⟨0, (15, 70)⟩  -- slope initialized to 0
  ((c1, c2, c3), line)

/-- The theorem to be proved --/
theorem dividing_line_slope_absolute_value_is_one 
  (setup : (Circle × Circle × Circle) × DividingLine) : 
  abs setup.2.slope = 1 := by
  sorry

end NUMINAMATH_CALUDE_dividing_line_slope_absolute_value_is_one_l156_15683


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l156_15633

/-- Given a rectangular pen with a perimeter of 60 feet, the maximum possible area is 225 square feet. -/
theorem max_area_rectangular_pen :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * (x + y) = 60 →
  x * y ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l156_15633


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l156_15642

/-- The magnitude of the sum of two vectors given specific conditions -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) :
  a = (1, 0) →
  ‖b‖ = Real.sqrt 2 →
  a • b = 1 →
  ‖2 • a + b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l156_15642


namespace NUMINAMATH_CALUDE_atlantic_charge_calculation_l156_15611

/-- Represents the additional charge per minute for Atlantic Call -/
def atlantic_charge_per_minute : ℚ := 1/5

/-- United Telephone's base rate -/
def united_base_rate : ℚ := 11

/-- United Telephone's charge per minute -/
def united_charge_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate -/
def atlantic_base_rate : ℚ := 12

/-- Number of minutes for which the bills are equal -/
def equal_bill_minutes : ℕ := 20

theorem atlantic_charge_calculation :
  united_base_rate + united_charge_per_minute * equal_bill_minutes =
  atlantic_base_rate + atlantic_charge_per_minute * equal_bill_minutes :=
sorry

end NUMINAMATH_CALUDE_atlantic_charge_calculation_l156_15611


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l156_15671

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 4 = 0 ∧ y^2 + m*y + 4 = 0) ↔ 
  (m < -4 ∨ m > 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l156_15671


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_is_two_l156_15664

/-- A geometric sequence with specified properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  a_5_eq_2 : a 5 = 2
  a_6_a_8_eq_8 : a 6 * a 8 = 8

/-- The ratio of differences in a geometric sequence with specific properties is 2 -/
theorem geometric_sequence_ratio_is_two (seq : GeometricSequence) :
  (seq.a 2018 - seq.a 2016) / (seq.a 2014 - seq.a 2012) = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_is_two_l156_15664


namespace NUMINAMATH_CALUDE_total_jeans_purchased_l156_15632

/-- Represents the number of pairs of Fox jeans purchased -/
def fox_jeans : ℕ := 3

/-- Represents the number of pairs of Pony jeans purchased -/
def pony_jeans : ℕ := 2

/-- Regular price of Fox jeans in dollars -/
def fox_price : ℚ := 15

/-- Regular price of Pony jeans in dollars -/
def pony_price : ℚ := 20

/-- Total discount in dollars -/
def total_discount : ℚ := 9

/-- Sum of discount rates as a percentage -/
def sum_discount_rates : ℚ := 22

/-- Discount rate on Pony jeans as a percentage -/
def pony_discount_rate : ℚ := 18.000000000000014

/-- Theorem stating the total number of jeans purchased -/
theorem total_jeans_purchased : fox_jeans + pony_jeans = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_jeans_purchased_l156_15632


namespace NUMINAMATH_CALUDE_correct_divisor_l156_15613

theorem correct_divisor : ∃ (X : ℕ) (incorrect_divisor correct_divisor : ℕ),
  incorrect_divisor = 87 ∧
  X / incorrect_divisor = 24 ∧
  X / correct_divisor = 58 ∧
  correct_divisor = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l156_15613


namespace NUMINAMATH_CALUDE_segment_length_problem_l156_15614

/-- Given a line segment AD of length 56 units, divided into three segments AB, BC, and CD,
    where AB : BC = 1 : 2 and BC : CD = 6 : 5, the length of AB is 12 units. -/
theorem segment_length_problem (AB BC CD : ℝ) : 
  AB + BC + CD = 56 → 
  AB / BC = 1 / 2 → 
  BC / CD = 6 / 5 → 
  AB = 12 := by
sorry

end NUMINAMATH_CALUDE_segment_length_problem_l156_15614


namespace NUMINAMATH_CALUDE_quadratic_factorization_l156_15616

theorem quadratic_factorization (y : ℝ) : 3 * y^2 - 6 * y + 3 = 3 * (y - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l156_15616


namespace NUMINAMATH_CALUDE_mechanic_average_earning_l156_15678

/-- The average earning of a mechanic for a week, given specific conditions -/
theorem mechanic_average_earning
  (first_four_avg : ℝ)
  (last_four_avg : ℝ)
  (fourth_day_earning : ℝ)
  (h1 : first_four_avg = 25)
  (h2 : last_four_avg = 22)
  (h3 : fourth_day_earning = 20) :
  (4 * first_four_avg + 4 * last_four_avg - fourth_day_earning) / 7 = 24 := by
  sorry

#check mechanic_average_earning

end NUMINAMATH_CALUDE_mechanic_average_earning_l156_15678


namespace NUMINAMATH_CALUDE_probability_3_or_more_babies_speak_l156_15672

def probability_at_least_3_out_of_7 (p : ℝ) : ℝ :=
  1 - (Nat.choose 7 0 * p^0 * (1-p)^7 +
       Nat.choose 7 1 * p^1 * (1-p)^6 +
       Nat.choose 7 2 * p^2 * (1-p)^5)

theorem probability_3_or_more_babies_speak :
  probability_at_least_3_out_of_7 (1/3) = 939/2187 := by
  sorry

end NUMINAMATH_CALUDE_probability_3_or_more_babies_speak_l156_15672


namespace NUMINAMATH_CALUDE_orchid_bushes_total_l156_15604

theorem orchid_bushes_total (current : ℕ) (today : ℕ) (tomorrow : ℕ) :
  current = 47 → today = 37 → tomorrow = 25 →
  current + today + tomorrow = 109 := by
  sorry

end NUMINAMATH_CALUDE_orchid_bushes_total_l156_15604


namespace NUMINAMATH_CALUDE_solve_for_x_l156_15612

theorem solve_for_x (x y : ℝ) (h1 : x + 3 * y = 33) (h2 : y = 10) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l156_15612


namespace NUMINAMATH_CALUDE_vanilla_syrup_cost_vanilla_syrup_cost_is_correct_l156_15690

/-- The cost of vanilla syrup in a coffee order -/
theorem vanilla_syrup_cost : ℝ :=
  let drip_coffee_cost : ℝ := 2.25
  let drip_coffee_quantity : ℕ := 2
  let espresso_cost : ℝ := 3.50
  let espresso_quantity : ℕ := 1
  let latte_cost : ℝ := 4.00
  let latte_quantity : ℕ := 2
  let cold_brew_cost : ℝ := 2.50
  let cold_brew_quantity : ℕ := 2
  let cappuccino_cost : ℝ := 3.50
  let cappuccino_quantity : ℕ := 1
  let total_order_cost : ℝ := 25.00

  have h1 : ℝ := drip_coffee_cost * drip_coffee_quantity +
                 espresso_cost * espresso_quantity +
                 latte_cost * latte_quantity +
                 cold_brew_cost * cold_brew_quantity +
                 cappuccino_cost * cappuccino_quantity

  have h2 : ℝ := total_order_cost - h1

  h2

theorem vanilla_syrup_cost_is_correct : vanilla_syrup_cost = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_vanilla_syrup_cost_vanilla_syrup_cost_is_correct_l156_15690


namespace NUMINAMATH_CALUDE_peter_chip_cost_l156_15653

/-- Calculates the cost to consume a given number of calories from chips, given the calorie content per chip, chips per bag, and cost per bag. -/
def cost_for_calories (calories_per_chip : ℕ) (chips_per_bag : ℕ) (cost_per_bag : ℚ) (target_calories : ℕ) : ℚ :=
  let calories_per_bag := calories_per_chip * chips_per_bag
  let bags_needed := (target_calories + calories_per_bag - 1) / calories_per_bag
  bags_needed * cost_per_bag

/-- Theorem stating that Peter needs to spend $4 to consume 480 calories of chips. -/
theorem peter_chip_cost : cost_for_calories 10 24 2 480 = 4 := by
  sorry

end NUMINAMATH_CALUDE_peter_chip_cost_l156_15653


namespace NUMINAMATH_CALUDE_terminal_side_of_half_angle_l156_15662

theorem terminal_side_of_half_angle (θ : Real) 
  (h1 : |Real.cos θ| = Real.cos θ) 
  (h2 : |Real.tan θ| = -Real.tan θ) : 
  (∃ (k : Int), 
    (k * Real.pi + Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ k * Real.pi + Real.pi) ∨
    (k * Real.pi + 3 * Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ k * Real.pi + 2 * Real.pi) ∨
    (∃ (n : Int), θ / 2 = n * Real.pi)) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_of_half_angle_l156_15662


namespace NUMINAMATH_CALUDE_class_size_problem_l156_15622

theorem class_size_problem :
  ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 29 :=
by sorry

end NUMINAMATH_CALUDE_class_size_problem_l156_15622


namespace NUMINAMATH_CALUDE_same_number_of_friends_l156_15646

theorem same_number_of_friends (n : ℕ) (h : n > 0) :
  ∃ (f : Fin n → Fin n),
    ∃ (i j : Fin n), i ≠ j ∧ f i = f j :=
by
  sorry

end NUMINAMATH_CALUDE_same_number_of_friends_l156_15646


namespace NUMINAMATH_CALUDE_problem_solving_probability_l156_15615

theorem problem_solving_probability (p_a p_either : ℝ) (h1 : p_a = 0.7) (h2 : p_either = 0.94) :
  ∃ p_b : ℝ, p_b = 0.8 ∧ p_either = 1 - (1 - p_a) * (1 - p_b) := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l156_15615


namespace NUMINAMATH_CALUDE_parabola_transformation_sum_l156_15636

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a transformation of a quadratic function -/
inductive Transformation
  | Reflect
  | Translate (d : ℝ)

/-- Applies a transformation to a quadratic function -/
def applyTransformation (q : QuadraticFunction) (t : Transformation) : QuadraticFunction :=
  match t with
  | Transformation.Reflect => { a := q.a, b := -q.b, c := q.c }
  | Transformation.Translate d => { a := q.a, b := q.b - 2 * q.a * d, c := q.a * d^2 - q.b * d + q.c }

/-- Sums two quadratic functions -/
def sumQuadraticFunctions (q1 q2 : QuadraticFunction) : QuadraticFunction :=
  { a := q1.a + q2.a, b := q1.b + q2.b, c := q1.c + q2.c }

theorem parabola_transformation_sum (q : QuadraticFunction) :
  let f := applyTransformation (applyTransformation q Transformation.Reflect) (Transformation.Translate (-7))
  let g := applyTransformation q (Transformation.Translate 3)
  let sum := sumQuadraticFunctions f g
  sum.a = 2 * q.a ∧ sum.b = 8 * q.a - 2 * q.b ∧ sum.c = 58 * q.a - 4 * q.b + 2 * q.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_sum_l156_15636


namespace NUMINAMATH_CALUDE_abc_product_range_l156_15663

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 9 then |Real.log x / Real.log 3 - 1|
  else if x > 9 then 4 - Real.sqrt x
  else 0

theorem abc_product_range (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  f a = f b ∧ f b = f c →
  81 < a * b * c ∧ a * b * c < 144 := by
sorry

end NUMINAMATH_CALUDE_abc_product_range_l156_15663


namespace NUMINAMATH_CALUDE_boxes_per_case_l156_15629

theorem boxes_per_case (total_boxes : ℕ) (num_cases : ℕ) (h1 : total_boxes = 24) (h2 : num_cases = 3) :
  total_boxes / num_cases = 8 := by
sorry

end NUMINAMATH_CALUDE_boxes_per_case_l156_15629


namespace NUMINAMATH_CALUDE_mango_jelly_dishes_l156_15656

theorem mango_jelly_dishes (total_dishes : ℕ) 
  (mango_salsa_dishes : ℕ) (fresh_mango_dishes : ℕ) 
  (oliver_pickout_dishes : ℕ) (oliver_left_dishes : ℕ) :
  total_dishes = 36 →
  mango_salsa_dishes = 3 →
  fresh_mango_dishes = total_dishes / 6 →
  oliver_pickout_dishes = 2 →
  oliver_left_dishes = 28 →
  total_dishes - oliver_left_dishes - (mango_salsa_dishes + (fresh_mango_dishes - oliver_pickout_dishes)) = 1 :=
by
  sorry

#check mango_jelly_dishes

end NUMINAMATH_CALUDE_mango_jelly_dishes_l156_15656


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l156_15686

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 0; 0, 1]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0, 5]

theorem matrix_equation_solution :
  ∀ X : Matrix (Fin 2) (Fin 1) ℝ,
  B⁻¹ * A⁻¹ * X = !![5; 1] →
  X = !![28; 5] := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l156_15686


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l156_15666

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3 * x - 15 → x ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l156_15666


namespace NUMINAMATH_CALUDE_union_equals_first_set_l156_15676

theorem union_equals_first_set (I M N : Set α) : 
  M ⊂ I → N ⊂ I → M ≠ N → M.Nonempty → N.Nonempty → N ∩ (I \ M) = ∅ → M ∪ N = M := by
  sorry

end NUMINAMATH_CALUDE_union_equals_first_set_l156_15676


namespace NUMINAMATH_CALUDE_complex_equation_solution_l156_15673

theorem complex_equation_solution :
  ∀ z : ℂ, (3 - z) * Complex.I = 2 * Complex.I → z = 3 + 2 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l156_15673


namespace NUMINAMATH_CALUDE_ellipse_properties_l156_15617

-- Define the ellipse C
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the condition for equilateral triangle formed by foci and minor axis endpoint
def equilateralCondition (a b c : ℝ) : Prop :=
  a = 2*c ∧ b = Real.sqrt 3 * c

-- Define the tangency condition for the circle
def tangencyCondition (a b c : ℝ) : Prop :=
  |c + 2| / Real.sqrt 2 = (Real.sqrt 6 / 2) * b

-- Define the vector addition condition
def vectorAdditionCondition (A B M : ℝ × ℝ) (t : ℝ) : Prop :=
  A.1 + B.1 = t * M.1 ∧ A.2 + B.2 = t * M.2

-- Main theorem
theorem ellipse_properties (a b : ℝ) (h : a > b ∧ b > 0) :
  ∀ (c : ℝ), equilateralCondition a b c →
  tangencyCondition a b c →
  (∀ (A B M : ℝ × ℝ) (t : ℝ),
    A ∈ ellipse a b h →
    B ∈ ellipse a b h →
    M ∈ ellipse a b h →
    (∃ (k : ℝ), A.2 = k*(A.1 - 3) ∧ B.2 = k*(B.1 - 3)) →
    vectorAdditionCondition A B M t →
    (a = 2 ∧ b = Real.sqrt 3 ∧ c = 1) ∧
    ((a^2 - b^2) / a^2 = 1/4) ∧
    (ellipse a b h = {p : ℝ × ℝ | p.1^2/4 + p.2^2/3 = 1}) ∧
    (-2 < t ∧ t < 2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l156_15617


namespace NUMINAMATH_CALUDE_correct_algebraic_equality_l156_15687

theorem correct_algebraic_equality (x y : ℝ) : -4 * x * y + 3 * x * y = -x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_algebraic_equality_l156_15687


namespace NUMINAMATH_CALUDE_worker_travel_time_l156_15689

/-- Proves that the usual travel time is 40 minutes given the conditions -/
theorem worker_travel_time (normal_speed : ℝ) (normal_time : ℝ) 
  (h1 : normal_speed > 0) 
  (h2 : normal_time > 0) 
  (h3 : normal_speed * normal_time = (4/5 * normal_speed) * (normal_time + 10)) : 
  normal_time = 40 := by
sorry

end NUMINAMATH_CALUDE_worker_travel_time_l156_15689


namespace NUMINAMATH_CALUDE_change_in_cubic_expression_l156_15649

theorem change_in_cubic_expression (x a : ℝ) (ha : a > 0) :
  abs ((x + a)^3 - 3*(x + a) - (x^3 - 3*x)) = 3*a*x^2 + 3*a^2*x + a^3 - 3*a ∧
  abs ((x - a)^3 - 3*(x - a) - (x^3 - 3*x)) = 3*a*x^2 + 3*a^2*x + a^3 - 3*a :=
by sorry

end NUMINAMATH_CALUDE_change_in_cubic_expression_l156_15649


namespace NUMINAMATH_CALUDE_isosceles_at_second_iteration_l156_15697

/-- Represents a triangle with angles α, β, and γ -/
structure Triangle where
  α : Real
  β : Real
  γ : Real

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { α := t.β, β := t.α, γ := 90 }

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop :=
  t.α = t.β ∨ t.β = t.γ ∨ t.γ = t.α

/-- The initial triangle A₀B₀C₀ -/
def A₀B₀C₀ : Triangle :=
  { α := 30, β := 60, γ := 90 }

/-- Generates the nth triangle in the sequence -/
def nthTriangle (n : Nat) : Triangle :=
  match n with
  | 0 => A₀B₀C₀
  | n + 1 => nextTriangle (nthTriangle n)

theorem isosceles_at_second_iteration :
  ∃ n : Nat, n > 0 ∧ isIsosceles (nthTriangle n) ∧ ∀ m : Nat, 0 < m ∧ m < n → ¬isIsosceles (nthTriangle m) :=
  sorry

end NUMINAMATH_CALUDE_isosceles_at_second_iteration_l156_15697


namespace NUMINAMATH_CALUDE_original_light_wattage_l156_15640

/-- Given a new light with 25% higher wattage than the original light,
    proves that if the new light has 100 watts, then the original light had 80 watts. -/
theorem original_light_wattage (new_wattage : ℝ) (h1 : new_wattage = 100) :
  let original_wattage := new_wattage / 1.25
  original_wattage = 80 := by
sorry

end NUMINAMATH_CALUDE_original_light_wattage_l156_15640


namespace NUMINAMATH_CALUDE_min_x_plus_y_l156_15655

theorem min_x_plus_y (x y : ℝ) (h1 : x > 1) (h2 : x * y = 2 * x + y + 2) :
  x + y ≥ 7 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 1 ∧ x₀ * y₀ = 2 * x₀ + y₀ + 2 ∧ x₀ + y₀ = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_x_plus_y_l156_15655


namespace NUMINAMATH_CALUDE_no_double_application_increment_l156_15635

theorem no_double_application_increment :
  ¬∃ f : ℤ → ℤ, ∀ x : ℤ, f (f x) = x + 1 := by sorry

end NUMINAMATH_CALUDE_no_double_application_increment_l156_15635


namespace NUMINAMATH_CALUDE_rice_yield_80kg_l156_15605

/-- Linear regression equation for rice yield prediction -/
def rice_yield_prediction (x : ℝ) : ℝ := 5 * x + 250

/-- Theorem: The predicted rice yield for 80 kg of fertilizer is 650 kg -/
theorem rice_yield_80kg : rice_yield_prediction 80 = 650 := by
  sorry

end NUMINAMATH_CALUDE_rice_yield_80kg_l156_15605


namespace NUMINAMATH_CALUDE_bus_total_capacity_l156_15645

/-- Represents the seating capacity of a bus with specific seating arrangements -/
def bus_capacity (left_seats : ℕ) (right_seat_diff : ℕ) (people_per_seat : ℕ) (back_seat_capacity : ℕ) : ℕ :=
  let right_seats := left_seats - right_seat_diff
  let total_regular_seats := left_seats + right_seats
  let regular_capacity := total_regular_seats * people_per_seat
  regular_capacity + back_seat_capacity

/-- Theorem stating the total seating capacity of the bus -/
theorem bus_total_capacity :
  bus_capacity 15 3 3 8 = 89 := by
  sorry

#eval bus_capacity 15 3 3 8

end NUMINAMATH_CALUDE_bus_total_capacity_l156_15645


namespace NUMINAMATH_CALUDE_max_cone_volume_in_sphere_l156_15695

/-- The maximum volume of a cone formed by a circular section of a sphere --/
theorem max_cone_volume_in_sphere (R : ℝ) (h : R = 9) : 
  ∃ (V : ℝ), V = 54 * Real.sqrt 3 * Real.pi ∧ 
  ∀ (r h : ℝ), r^2 + h^2 = R^2 → 
  (1/3 : ℝ) * Real.pi * r^2 * h ≤ V := by
  sorry

end NUMINAMATH_CALUDE_max_cone_volume_in_sphere_l156_15695


namespace NUMINAMATH_CALUDE_two_true_propositions_l156_15644

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define a predicate for a right angle
def is_right_angle (angle : Real) : Prop := angle = 90

-- Define a predicate for a right triangle
def is_right_triangle (t : Triangle) : Prop := ∃ angle, is_right_angle angle

-- Define the original proposition
def original_prop (t : Triangle) : Prop :=
  is_right_angle t.C → is_right_triangle t

-- Define the converse proposition
def converse_prop (t : Triangle) : Prop :=
  is_right_triangle t → is_right_angle t.C

-- Define the inverse proposition
def inverse_prop (t : Triangle) : Prop :=
  ¬(is_right_angle t.C) → ¬(is_right_triangle t)

-- Define the contrapositive proposition
def contrapositive_prop (t : Triangle) : Prop :=
  ¬(is_right_triangle t) → ¬(is_right_angle t.C)

-- Theorem stating that exactly two of these propositions are true
theorem two_true_propositions :
  ∃ (t : Triangle),
    (original_prop t ∧ contrapositive_prop t) ∧
    ¬(converse_prop t ∨ inverse_prop t) :=
  sorry

end NUMINAMATH_CALUDE_two_true_propositions_l156_15644


namespace NUMINAMATH_CALUDE_max_inspector_sum_l156_15623

/-- Represents the configuration of towers in the city of Flat -/
structure TowerConfiguration where
  one_floor : ℕ  -- Number of 1-floor towers
  two_floor : ℕ  -- Number of 2-floor towers

/-- Calculates the total height of all towers -/
def total_height (config : TowerConfiguration) : ℕ :=
  config.one_floor + 2 * config.two_floor

/-- Calculates the inspector's sum for a given configuration -/
def inspector_sum (config : TowerConfiguration) : ℕ :=
  config.one_floor * config.two_floor

/-- Theorem stating that the maximum inspector's sum is 112 -/
theorem max_inspector_sum :
  ∃ (config : TowerConfiguration),
    total_height config = 30 ∧
    inspector_sum config = 112 ∧
    ∀ (other : TowerConfiguration),
      total_height other = 30 →
      inspector_sum other ≤ 112 := by
  sorry

end NUMINAMATH_CALUDE_max_inspector_sum_l156_15623


namespace NUMINAMATH_CALUDE_opposite_of_sqrt3_minus_2_l156_15619

theorem opposite_of_sqrt3_minus_2 :
  -(Real.sqrt 3 - 2) = 2 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt3_minus_2_l156_15619


namespace NUMINAMATH_CALUDE_parallelogram_base_l156_15681

/-- Given a parallelogram with area 308 square centimeters and height 14 cm, its base is 22 cm. -/
theorem parallelogram_base (area height base : ℝ) : 
  area = 308 ∧ height = 14 ∧ area = base * height → base = 22 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l156_15681


namespace NUMINAMATH_CALUDE_power_sum_equals_zero_l156_15677

theorem power_sum_equals_zero : (-1)^2021 + 1^2022 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_zero_l156_15677


namespace NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_for_skew_lines_l156_15638

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the intersection relation between two planes
variable (intersect : Plane → Plane → Prop)

-- Define the skew relation between two lines
variable (skew : Line → Line → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

theorem planes_intersect_necessary_not_sufficient_for_skew_lines
  (α β : Plane) (m n : Line)
  (h_distinct : α ≠ β)
  (h_perp_m : perp m α)
  (h_perp_n : perp n β) :
  (∀ α β m n, skew m n → intersect α β) ∧
  (∃ α β m n, intersect α β ∧ perp m α ∧ perp n β ∧ ¬skew m n) :=
sorry

end NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_for_skew_lines_l156_15638


namespace NUMINAMATH_CALUDE_fraction_value_l156_15675

theorem fraction_value : 
  let a : ℕ := 2003
  let b : ℕ := 2002
  let four : ℕ := 2^2
  let six : ℕ := 2 * 3
  (four^a * 3^b) / (six^b * 2^a) = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l156_15675


namespace NUMINAMATH_CALUDE_f_monotonically_decreasing_l156_15648

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 2*x - 2

-- Theorem statement
theorem f_monotonically_decreasing :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 1 → f x₁ > f x₂ := by sorry

end NUMINAMATH_CALUDE_f_monotonically_decreasing_l156_15648


namespace NUMINAMATH_CALUDE_last_two_digits_same_l156_15694

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, (a (n + 1) = a n + 54) ∨ (a (n + 1) = a n + 77)

theorem last_two_digits_same (a : ℕ → ℕ) (h : sequence_property a) :
  ∃ k : ℕ, a k % 100 = a (k + 1) % 100 :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_same_l156_15694


namespace NUMINAMATH_CALUDE_sum_of_coefficients_eq_value_at_one_l156_15661

/-- The polynomial for which we want to find the sum of coefficients -/
def p (x : ℝ) : ℝ := 3*(x^8 - x^5 + 2*x^3 - 6) - 5*(x^4 + 3*x^2) + 2*(x^6 - 5)

/-- The sum of coefficients of a polynomial is equal to its value at x = 1 -/
theorem sum_of_coefficients_eq_value_at_one :
  p 1 = -40 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_eq_value_at_one_l156_15661


namespace NUMINAMATH_CALUDE_exists_good_permutation_iff_power_of_two_l156_15654

/-- A permutation is "good" if for any i < j < k, n doesn't divide (aᵢ + aₖ - 2aⱼ) -/
def is_good_permutation (n : ℕ) (a : Fin n → ℕ) : Prop :=
  ∀ i j k : Fin n, i < j → j < k → ¬(n ∣ a i + a k - 2 * a j)

/-- A natural number n ≥ 3 has a good permutation if and only if it's a power of 2 -/
theorem exists_good_permutation_iff_power_of_two (n : ℕ) (h : n ≥ 3) :
  (∃ (a : Fin n → ℕ), Function.Bijective a ∧ is_good_permutation n a) ↔ ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_exists_good_permutation_iff_power_of_two_l156_15654


namespace NUMINAMATH_CALUDE_connie_marbles_problem_l156_15607

/-- Proves that Connie started with 143 marbles given the conditions of the problem -/
theorem connie_marbles_problem :
  ∀ (initial : ℕ),
  initial - 73 = 70 →
  initial = 143 :=
by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_problem_l156_15607


namespace NUMINAMATH_CALUDE_quadratic_ratio_l156_15608

/-- Given a quadratic function f(x) = x^2 + 1500x + 1500, 
    prove that when expressed as (x + b)^2 + c, 
    the ratio c/b equals -748 -/
theorem quadratic_ratio (f : ℝ → ℝ) (b c : ℝ) : 
  (∀ x, f x = x^2 + 1500*x + 1500) → 
  (∀ x, f x = (x + b)^2 + c) → 
  c / b = -748 := by
sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l156_15608


namespace NUMINAMATH_CALUDE_constant_jump_returns_to_start_increasing_jump_returns_to_start_l156_15680

-- Define the number of stones
def num_stones : ℕ := 10

-- Define the number of jumps
def num_jumps : ℕ := 100

-- Function to calculate the position after constant jumps
def constant_jump_position (jump_size : ℕ) : ℕ :=
  (1 + jump_size * num_jumps) % num_stones

-- Function to calculate the position after increasing jumps
def increasing_jump_position : ℕ :=
  (1 + (num_jumps * (num_jumps + 1) / 2)) % num_stones

-- Theorem for constant jump scenario
theorem constant_jump_returns_to_start :
  constant_jump_position 2 = 1 := by sorry

-- Theorem for increasing jump scenario
theorem increasing_jump_returns_to_start :
  increasing_jump_position = 1 := by sorry

end NUMINAMATH_CALUDE_constant_jump_returns_to_start_increasing_jump_returns_to_start_l156_15680


namespace NUMINAMATH_CALUDE_age_problem_l156_15628

/-- Given three people a, b, and c, where:
    - a is two years older than b
    - b is twice as old as c
    - The sum of their ages is 72
    Prove that b is 28 years old. -/
theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : a + b + c = 72) : 
  b = 28 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l156_15628


namespace NUMINAMATH_CALUDE_sandys_comic_books_l156_15674

theorem sandys_comic_books (initial : ℕ) : 
  (initial / 2 + 6 = 13) → initial = 14 := by
  sorry

end NUMINAMATH_CALUDE_sandys_comic_books_l156_15674


namespace NUMINAMATH_CALUDE_existence_of_n_l156_15699

theorem existence_of_n : ∃ n : ℕ+, 
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ 10 → 
    ∃ p : ℕ+, 
      (↑p + 2015/10000 : ℝ)^k < n ∧ n < (↑p + 2016/10000 : ℝ)^k := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l156_15699


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l156_15667

/-- The percentage of water in fresh grapes -/
def water_percentage_fresh : ℝ := 90

/-- The percentage of water in dried grapes -/
def water_percentage_dried : ℝ := 20

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 5

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 0.625

theorem water_percentage_in_fresh_grapes :
  (100 - water_percentage_fresh) / 100 * fresh_weight = 
  (100 - water_percentage_dried) / 100 * dried_weight := by sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l156_15667


namespace NUMINAMATH_CALUDE_negation_of_sum_even_both_even_l156_15660

theorem negation_of_sum_even_both_even :
  (¬ ∀ (a b : ℤ), Even (a + b) → (Even a ∧ Even b)) ↔
  (∃ (a b : ℤ), Even (a + b) ∧ (¬ Even a ∨ ¬ Even b)) :=
sorry

end NUMINAMATH_CALUDE_negation_of_sum_even_both_even_l156_15660


namespace NUMINAMATH_CALUDE_factory_growth_rate_l156_15618

theorem factory_growth_rate (x : ℝ) : 
  (1 + x)^2 = 1.2 → x < 0.1 := by sorry

end NUMINAMATH_CALUDE_factory_growth_rate_l156_15618


namespace NUMINAMATH_CALUDE_arithmetic_progression_terms_l156_15692

/-- 
Given an arithmetic progression with:
- First term: 2
- Last term: 62
- Common difference: 2

Prove that the number of terms in this arithmetic progression is 31.
-/
theorem arithmetic_progression_terms : 
  let a := 2  -- First term
  let L := 62 -- Last term
  let d := 2  -- Common difference
  let n := (L - a) / d + 1 -- Number of terms formula
  n = 31 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_terms_l156_15692


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_holds_l156_15626

/-- An isosceles triangle with two sides of length 12 and a third side of length 17 has a perimeter of 41. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 12 ∧ b = 12 ∧ c = 17) →  -- Two sides are 12, third side is 17
    (a = b)  →                    -- Isosceles triangle condition
    (a + b + c = 41)              -- Perimeter is 41

/-- The theorem holds for the given triangle. -/
theorem isosceles_triangle_perimeter_holds : isosceles_triangle_perimeter 12 12 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_holds_l156_15626


namespace NUMINAMATH_CALUDE_printer_z_time_l156_15643

/-- Given printers X, Y, and Z with the following properties:
  - The ratio of time for X alone to Y and Z together is 2.25
  - X can do the job in 15 hours
  - Y can do the job in 10 hours
Prove that Z takes 20 hours to do the job alone. -/
theorem printer_z_time (tx ty tz : ℝ) : 
  tx = 15 → 
  ty = 10 → 
  tx = 2.25 * (1 / (1 / ty + 1 / tz)) → 
  tz = 20 := by
sorry

end NUMINAMATH_CALUDE_printer_z_time_l156_15643


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l156_15650

/-- Represents a hyperbola with equation x²/m - y²/6 = 1 -/
structure Hyperbola where
  m : ℝ
  eq : ∀ x y : ℝ, x^2 / m - y^2 / 6 = 1

/-- The focal distance of a hyperbola -/
def focal_distance (h : Hyperbola) : ℝ := 6

theorem hyperbola_m_value (h : Hyperbola) (hf : focal_distance h = 6) : h.m = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l156_15650


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l156_15639

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 2) :
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l156_15639


namespace NUMINAMATH_CALUDE_inequalities_given_sum_positive_l156_15698

/-- Given two real numbers a and b such that a + b > 0, 
    the following statements are true:
    1. a^5 * b^2 + a^4 * b^3 ≥ 0
    2. a^21 + b^21 > 0
    3. (a+2)*(b+2) > a*b
-/
theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_sum_positive_l156_15698


namespace NUMINAMATH_CALUDE_triangle_tangent_difference_l156_15603

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² = b² + 4bc sin A and tan A · tan B = 2, then tan B - tan A = -8 -/
theorem triangle_tangent_difference (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 = b^2 + 4*b*c*(Real.sin A) →
  Real.tan A * Real.tan B = 2 →
  Real.tan B - Real.tan A = -8 := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_difference_l156_15603


namespace NUMINAMATH_CALUDE_min_value_product_l156_15606

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (3 * x + y) * (x + 3 * z) * (y + z + 1) ≥ 48 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (3 * x₀ + y₀) * (x₀ + 3 * z₀) * (y₀ + z₀ + 1) = 48 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l156_15606


namespace NUMINAMATH_CALUDE_solve_baseball_card_problem_l156_15665

def baseball_card_problem (initial_cards : ℕ) (final_cards : ℕ) : Prop :=
  ∃ (cards_to_peter : ℕ),
    let cards_after_maria := initial_cards - (initial_cards + 1) / 2
    let cards_before_paul := cards_after_maria - cards_to_peter
    3 * cards_before_paul = final_cards ∧
    cards_to_peter = 1

theorem solve_baseball_card_problem :
  baseball_card_problem 15 18 :=
sorry

end NUMINAMATH_CALUDE_solve_baseball_card_problem_l156_15665


namespace NUMINAMATH_CALUDE_quadratic_value_l156_15668

/-- A quadratic function with specific properties -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 2)^2 + 7

theorem quadratic_value (a : ℝ) :
  (∀ x, f a x ≤ 7) →  -- Maximum value condition
  (f a 2 = 7) →       -- Maximum occurs at x = 2
  (f a 0 = -7) →      -- Passes through (0, -7)
  (a < 0) →           -- Implied by maximum condition
  (f a 5 = -24.5) :=  -- The value at x = 5
by sorry

end NUMINAMATH_CALUDE_quadratic_value_l156_15668


namespace NUMINAMATH_CALUDE_points_form_parabola_l156_15647

-- Define the set of points (x, y) parametrically
def S : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p.1 = Real.cos t ^ 2 ∧ p.2 = Real.sin (2 * t)}

-- Define a parabola in general form
def IsParabola (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧
    ∀ p ∈ S, a * p.1^2 + b * p.1 * p.2 + c * p.2^2 + d * p.1 + e * p.2 = 0

-- Theorem statement
theorem points_form_parabola : IsParabola S := by
  sorry

end NUMINAMATH_CALUDE_points_form_parabola_l156_15647


namespace NUMINAMATH_CALUDE_trading_cards_theorem_l156_15685

/-- The number of cards in a partially filled box -/
def partially_filled_box (total_cards : ℕ) (cards_per_box : ℕ) : ℕ :=
  total_cards % cards_per_box

theorem trading_cards_theorem :
  let pokemon_cards := 65
  let magic_cards := 55
  let yugioh_cards := 40
  let pokemon_per_box := 8
  let magic_per_box := 10
  let yugioh_per_box := 12
  (partially_filled_box pokemon_cards pokemon_per_box = 1) ∧
  (partially_filled_box magic_cards magic_per_box = 5) ∧
  (partially_filled_box yugioh_cards yugioh_per_box = 4) :=
by sorry

end NUMINAMATH_CALUDE_trading_cards_theorem_l156_15685


namespace NUMINAMATH_CALUDE_sqrt_D_always_irrational_l156_15684

-- Define the relationship between a and b as consecutive integers
def consecutive (a b : ℤ) : Prop := b = a + 1

-- Define D in terms of a and b
def D (a b : ℤ) : ℤ := a^2 + b^2 + (a + b)^2

-- Theorem statement
theorem sqrt_D_always_irrational (a b : ℤ) (h : consecutive a b) :
  Irrational (Real.sqrt (D a b)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_D_always_irrational_l156_15684


namespace NUMINAMATH_CALUDE_w_range_l156_15670

theorem w_range (x y w : ℝ) : 
  -x + y = 2 → x < 3 → y ≥ 0 → w = x + y - 2 → -4 ≤ w ∧ w < 6 :=
by sorry

end NUMINAMATH_CALUDE_w_range_l156_15670


namespace NUMINAMATH_CALUDE_adams_final_score_l156_15624

/-- Calculates the final score in a trivia game -/
def final_score (correct_first_half correct_second_half points_per_question : ℕ) : ℕ :=
  (correct_first_half + correct_second_half) * points_per_question

/-- Theorem: Adam's final score in the trivia game is 50 points -/
theorem adams_final_score : 
  final_score 5 5 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_adams_final_score_l156_15624


namespace NUMINAMATH_CALUDE_score_difference_is_negative_1_75_l156_15601

def score_distribution : List (Float × Float) := [
  (0.15, 80),
  (0.40, 90),
  (0.25, 95),
  (0.20, 100)
]

def median (dist : List (Float × Float)) : Float :=
  90  -- The median is 90 as per the problem description

def mean (dist : List (Float × Float)) : Float :=
  dist.foldr (λ (p, s) acc => acc + p * s) 0

theorem score_difference_is_negative_1_75 :
  median score_distribution - mean score_distribution = -1.75 := by
  sorry

#eval median score_distribution - mean score_distribution

end NUMINAMATH_CALUDE_score_difference_is_negative_1_75_l156_15601
