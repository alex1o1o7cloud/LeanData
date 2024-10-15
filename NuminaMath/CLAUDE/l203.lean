import Mathlib

namespace NUMINAMATH_CALUDE_math_problem_proof_l203_20344

theorem math_problem_proof (b m n : ℕ) (B C : ℝ) (D : ℝ) :
  b = 4 →
  m = 1 →
  n = 1 →
  (b^m)^n + b^(m+n) = 20 →
  2^20 = B^10 →
  B > 0 →
  Real.sqrt ((20 * B + 45) / C) = C →
  D = C * Real.sin (30 * π / 180) →
  A = 20 ∧ B = 4 ∧ C = 5 ∧ D = 2.5 :=
by sorry

end NUMINAMATH_CALUDE_math_problem_proof_l203_20344


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l203_20394

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 + I) / (4 + 3 * I) → z.im = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l203_20394


namespace NUMINAMATH_CALUDE_opening_night_customers_count_l203_20389

/-- Represents the revenue and customer data for a movie theater on a specific day. -/
structure TheaterData where
  matineePrice : ℕ
  eveningPrice : ℕ
  openingNightPrice : ℕ
  popcornPrice : ℕ
  matineeCustomers : ℕ
  eveningCustomers : ℕ
  totalRevenue : ℕ

/-- Calculates the number of opening night customers given theater data. -/
def openingNightCustomers (data : TheaterData) : ℕ :=
  let totalCustomers := data.matineeCustomers + data.eveningCustomers + (data.totalRevenue - 
    (data.matineePrice * data.matineeCustomers + 
     data.eveningPrice * data.eveningCustomers + 
     (data.popcornPrice * (data.matineeCustomers + data.eveningCustomers)) / 2)) / data.openingNightPrice
  (data.totalRevenue - 
   (data.matineePrice * data.matineeCustomers + 
    data.eveningPrice * data.eveningCustomers + 
    data.popcornPrice * totalCustomers / 2)) / data.openingNightPrice

theorem opening_night_customers_count (data : TheaterData) 
  (h1 : data.matineePrice = 5)
  (h2 : data.eveningPrice = 7)
  (h3 : data.openingNightPrice = 10)
  (h4 : data.popcornPrice = 10)
  (h5 : data.matineeCustomers = 32)
  (h6 : data.eveningCustomers = 40)
  (h7 : data.totalRevenue = 1670) :
  openingNightCustomers data = 58 := by
  sorry

#eval openingNightCustomers {
  matineePrice := 5,
  eveningPrice := 7,
  openingNightPrice := 10,
  popcornPrice := 10,
  matineeCustomers := 32,
  eveningCustomers := 40,
  totalRevenue := 1670
}

end NUMINAMATH_CALUDE_opening_night_customers_count_l203_20389


namespace NUMINAMATH_CALUDE_malou_average_score_l203_20353

def malou_quiz_scores : List ℝ := [91, 90, 92]

theorem malou_average_score : 
  (malou_quiz_scores.sum / malou_quiz_scores.length : ℝ) = 91 := by
  sorry

end NUMINAMATH_CALUDE_malou_average_score_l203_20353


namespace NUMINAMATH_CALUDE_square_sum_equals_three_l203_20307

theorem square_sum_equals_three (a b : ℝ) (h : a^4 + b^4 = a^2 - 2*a^2*b^2 + b^2 + 6) : 
  a^2 + b^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_three_l203_20307


namespace NUMINAMATH_CALUDE_customers_stayed_behind_l203_20343

theorem customers_stayed_behind (initial_customers : ℕ) 
  (h1 : initial_customers = 11) 
  (stayed : ℕ) 
  (left : ℕ) 
  (h2 : left = stayed + 5) 
  (h3 : stayed + left = initial_customers) : 
  stayed = 3 := by
  sorry

end NUMINAMATH_CALUDE_customers_stayed_behind_l203_20343


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l203_20339

/-- Given the teaching years of Virginia, Adrienne, and Dennis, prove that Dennis has taught for 46 years. -/
theorem dennis_teaching_years 
  (total : ℕ) 
  (h_total : total = 102)
  (h_virginia_adrienne : ∃ (a : ℕ), virginia = a + 9)
  (h_virginia_dennis : ∃ (d : ℕ), virginia = d - 9)
  (h_sum : virginia + adrienne + dennis = total)
  : dennis = 46 := by
  sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l203_20339


namespace NUMINAMATH_CALUDE_room_width_calculation_l203_20386

theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 9 →
  cost_per_sqm = 900 →
  total_cost = 38475 →
  (total_cost / cost_per_sqm) / length = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l203_20386


namespace NUMINAMATH_CALUDE_ourDie_expected_value_l203_20360

/-- Represents the four-sided die with its probabilities and winnings --/
structure UnusualDie where
  side1_prob : ℚ
  side1_win : ℚ
  side2_prob : ℚ
  side2_win : ℚ
  side3_prob : ℚ
  side3_win : ℚ
  side4_prob : ℚ
  side4_win : ℚ

/-- The specific unusual die described in the problem --/
def ourDie : UnusualDie :=
  { side1_prob := 1/4
  , side1_win := 2
  , side2_prob := 1/4
  , side2_win := 4
  , side3_prob := 1/3
  , side3_win := -6
  , side4_prob := 1/6
  , side4_win := 0 }

/-- Calculates the expected value of rolling the die --/
def expectedValue (d : UnusualDie) : ℚ :=
  d.side1_prob * d.side1_win +
  d.side2_prob * d.side2_win +
  d.side3_prob * d.side3_win +
  d.side4_prob * d.side4_win

/-- Theorem stating that the expected value of rolling ourDie is -1/2 --/
theorem ourDie_expected_value :
  expectedValue ourDie = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ourDie_expected_value_l203_20360


namespace NUMINAMATH_CALUDE_purchase_cost_l203_20312

/-- The cost of a single pencil in dollars -/
def pencil_cost : ℚ := 2.5

/-- The cost of a single pen in dollars -/
def pen_cost : ℚ := 3.5

/-- The number of pencils bought -/
def num_pencils : ℕ := 38

/-- The number of pens bought -/
def num_pens : ℕ := 56

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := pencil_cost * num_pencils + pen_cost * num_pens

theorem purchase_cost : total_cost = 291 := by sorry

end NUMINAMATH_CALUDE_purchase_cost_l203_20312


namespace NUMINAMATH_CALUDE_max_value_of_expression_l203_20316

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l203_20316


namespace NUMINAMATH_CALUDE_cookie_average_l203_20332

theorem cookie_average (packages : List Nat) 
  (h1 : packages = [9, 11, 14, 12, 0, 18, 15, 16, 19, 21]) : 
  (packages.sum : Rat) / packages.length = 27/2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_average_l203_20332


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l203_20308

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingDigits : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (x : RepeatingDecimal) : ℚ :=
  x.nonRepeating + x.repeating / (1 - (1 / 10 ^ x.repeatingDigits))

theorem repeating_decimal_to_fraction :
  let x : RepeatingDecimal := { nonRepeating := 7/10, repeating := 36/100, repeatingDigits := 2 }
  x.toRational = 27 / 37 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l203_20308


namespace NUMINAMATH_CALUDE_price_comparison_l203_20321

theorem price_comparison (a : ℝ) (h : a > 0) : a * (1.1^5) * (0.9^5) < a := by
  sorry

end NUMINAMATH_CALUDE_price_comparison_l203_20321


namespace NUMINAMATH_CALUDE_investment_comparison_l203_20302

def initial_investment : ℝ := 200

def delta_year1_change : ℝ := 1.10
def delta_year2_change : ℝ := 0.90

def echo_year1_change : ℝ := 0.70
def echo_year2_change : ℝ := 1.50

def foxtrot_year1_change : ℝ := 1.00
def foxtrot_year2_change : ℝ := 0.95

def final_delta : ℝ := initial_investment * delta_year1_change * delta_year2_change
def final_echo : ℝ := initial_investment * echo_year1_change * echo_year2_change
def final_foxtrot : ℝ := initial_investment * foxtrot_year1_change * foxtrot_year2_change

theorem investment_comparison : final_foxtrot < final_delta ∧ final_delta < final_echo := by
  sorry

end NUMINAMATH_CALUDE_investment_comparison_l203_20302


namespace NUMINAMATH_CALUDE_distance_from_origin_l203_20351

theorem distance_from_origin (x : ℝ) : 
  |x| = Real.sqrt 5 ↔ x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l203_20351


namespace NUMINAMATH_CALUDE_otimes_self_otimes_self_l203_20303

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^2 - y^2

/-- Theorem: For any real number a, (a ⊗ a) ⊗ (a ⊗ a) = 0 -/
theorem otimes_self_otimes_self (a : ℝ) : otimes (otimes a a) (otimes a a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_otimes_self_otimes_self_l203_20303


namespace NUMINAMATH_CALUDE_puppies_adopted_l203_20314

/-- The cost to get a cat ready for adoption -/
def cat_cost : ℕ := 50

/-- The cost to get an adult dog ready for adoption -/
def adult_dog_cost : ℕ := 100

/-- The cost to get a puppy ready for adoption -/
def puppy_cost : ℕ := 150

/-- The number of cats adopted -/
def cats_adopted : ℕ := 2

/-- The number of adult dogs adopted -/
def adult_dogs_adopted : ℕ := 3

/-- The total cost for all adopted animals -/
def total_cost : ℕ := 700

/-- Theorem stating that the number of puppies adopted is 2 -/
theorem puppies_adopted : 
  ∃ (p : ℕ), p = 2 ∧ 
  cat_cost * cats_adopted + adult_dog_cost * adult_dogs_adopted + puppy_cost * p = total_cost :=
sorry

end NUMINAMATH_CALUDE_puppies_adopted_l203_20314


namespace NUMINAMATH_CALUDE_ribbon_division_l203_20363

theorem ribbon_division (total_ribbon : ℚ) (num_boxes : ℕ) (ribbon_per_box : ℚ) :
  total_ribbon = 5 / 8 →
  num_boxes = 5 →
  ribbon_per_box = total_ribbon / num_boxes →
  ribbon_per_box = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_division_l203_20363


namespace NUMINAMATH_CALUDE_part1_part2_l203_20325

-- Define the operation
def star_op (a b : ℚ) : ℚ := (a * b) / (a + b)

-- Part 1: Prove the specific calculation
theorem part1 : star_op (-3) (-1/3) = -3/10 := by sorry

-- Part 2: Prove when the operation is undefined
theorem part2 (a b : ℚ) : 
  a + b = 0 → ¬ ∃ (q : ℚ), star_op a b = q := by sorry

end NUMINAMATH_CALUDE_part1_part2_l203_20325


namespace NUMINAMATH_CALUDE_frank_candy_bags_l203_20327

/-- The number of bags Frank used to store his candy -/
def num_bags (total_candy : ℕ) (candy_per_bag : ℕ) : ℕ :=
  total_candy / candy_per_bag

/-- Theorem: Frank used 26 bags to store his candy -/
theorem frank_candy_bags : num_bags 858 33 = 26 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_bags_l203_20327


namespace NUMINAMATH_CALUDE_triangle_properties_l203_20328

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Define triangle ABC
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Sine law
  (a / Real.sin A = b / Real.sin B) →
  -- Cosine law
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  -- Statement A
  (a / Real.cos A = b / Real.sin B → A = π/4) ∧
  -- Statement D
  (A < π/2 ∧ B < π/2 ∧ C < π/2 → Real.sin A + Real.sin B > Real.cos A + Real.cos B) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l203_20328


namespace NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l203_20366

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem fourth_term_of_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 0 = 12 →
  a 5 = 47 →
  a 3 = 29.5 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l203_20366


namespace NUMINAMATH_CALUDE_convex_pentagon_side_comparison_l203_20365

/-- A circle in which pentagons are inscribed -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A convex pentagon inscribed in a circle -/
structure ConvexPentagon (c : Circle) where
  vertices : Fin 5 → ℝ × ℝ
  inscribed : ∀ i, (vertices i).1^2 + (vertices i).2^2 = c.radius^2
  convex : sorry  -- Additional condition to ensure convexity

/-- The side length of a regular pentagon inscribed in a circle -/
def regularPentagonSideLength (c : Circle) : ℝ := sorry

/-- The side lengths of a convex pentagon -/
def pentagonSideLengths (c : Circle) (p : ConvexPentagon c) : Fin 5 → ℝ := sorry

theorem convex_pentagon_side_comparison (c : Circle) (p : ConvexPentagon c) :
  ∃ i : Fin 5, pentagonSideLengths c p i ≤ regularPentagonSideLength c := by sorry

end NUMINAMATH_CALUDE_convex_pentagon_side_comparison_l203_20365


namespace NUMINAMATH_CALUDE_calculate_death_rate_l203_20320

/-- Calculates the death rate given birth rate and population growth rate -/
theorem calculate_death_rate (birth_rate : ℝ) (growth_rate : ℝ) : 
  birth_rate = 32 → growth_rate = 0.021 → 
  ∃ (death_rate : ℝ), death_rate = 11 ∧ birth_rate - death_rate = 1000 * growth_rate :=
by
  sorry

#check calculate_death_rate

end NUMINAMATH_CALUDE_calculate_death_rate_l203_20320


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_36_l203_20322

theorem five_digit_divisible_by_36 (n : ℕ) : 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∃ a b : ℕ, n = a * 10000 + 1000 + 200 + 30 + b) ∧  -- form ⬜123⬜
  (n % 36 = 0) →  -- divisible by 36
  (n = 11232 ∨ n = 61236) := by
sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_36_l203_20322


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l203_20362

open Complex

theorem smallest_distance_between_complex_points (z w : ℂ) 
  (hz : Complex.abs (z + 3 + 4*I) = 2)
  (hw : Complex.abs (w - 6 - 10*I) = 4) :
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), 
      Complex.abs (z' + 3 + 4*I) = 2 → 
      Complex.abs (w' - 6 - 10*I) = 4 → 
      Complex.abs (z' - w') ≥ min_dist) ∧ 
    min_dist = Real.sqrt 277 - 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l203_20362


namespace NUMINAMATH_CALUDE_speed_conversion_correct_l203_20379

/-- Conversion factor from km/h to m/s -/
def kmh_to_ms : ℝ := 0.277778

/-- Given speed in km/h -/
def speed_kmh : ℝ := 84

/-- Equivalent speed in m/s -/
def speed_ms : ℝ := speed_kmh * kmh_to_ms

theorem speed_conversion_correct : 
  ∃ ε > 0, |speed_ms - 23.33| < ε :=
sorry

end NUMINAMATH_CALUDE_speed_conversion_correct_l203_20379


namespace NUMINAMATH_CALUDE_largest_x_for_prime_f_l203_20384

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def f (x : ℤ) : ℤ := |4*x^2 - 41*x + 21|

theorem largest_x_for_prime_f :
  ∀ x : ℤ, x > 2 → ¬(is_prime (f x).toNat) ∧ is_prime (f 2).toNat :=
sorry

end NUMINAMATH_CALUDE_largest_x_for_prime_f_l203_20384


namespace NUMINAMATH_CALUDE_printer_ink_problem_l203_20381

/-- The problem of calculating the additional money needed for printer inks --/
theorem printer_ink_problem (initial_amount : ℕ) (black_cost red_cost yellow_cost : ℕ)
  (black_quantity red_quantity yellow_quantity : ℕ) : 
  initial_amount = 50 →
  black_cost = 11 →
  red_cost = 15 →
  yellow_cost = 13 →
  black_quantity = 2 →
  red_quantity = 3 →
  yellow_quantity = 2 →
  (black_cost * black_quantity + red_cost * red_quantity + yellow_cost * yellow_quantity) - initial_amount = 43 := by
  sorry

#check printer_ink_problem

end NUMINAMATH_CALUDE_printer_ink_problem_l203_20381


namespace NUMINAMATH_CALUDE_remainder_sum_l203_20345

theorem remainder_sum (x y : ℤ) (hx : x % 60 = 53) (hy : y % 45 = 17) :
  (x + y) % 15 = 10 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l203_20345


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l203_20398

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water :
  let current_speed : ℝ := 4
  let downstream_distance : ℝ := 5.133333333333334
  let downstream_time : ℝ := 14 / 60
  ∃ v : ℝ, v > 0 ∧ (v + current_speed) * downstream_time = downstream_distance ∧ v = 18 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l203_20398


namespace NUMINAMATH_CALUDE_fourth_grade_students_l203_20309

/-- Calculates the final number of students in fourth grade -/
def final_student_count (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem stating that the final number of students is 43 -/
theorem fourth_grade_students : final_student_count 4 3 42 = 43 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l203_20309


namespace NUMINAMATH_CALUDE_replaced_girl_weight_l203_20367

theorem replaced_girl_weight
  (n : ℕ)
  (original_average : ℝ)
  (new_average : ℝ)
  (new_girl_weight : ℝ)
  (h1 : n = 25)
  (h2 : new_average = original_average + 1)
  (h3 : new_girl_weight = 80) :
  ∃ (replaced_weight : ℝ),
    replaced_weight = new_girl_weight - n * (new_average - original_average) ∧
    replaced_weight = 55 := by
  sorry

end NUMINAMATH_CALUDE_replaced_girl_weight_l203_20367


namespace NUMINAMATH_CALUDE_penelope_candy_count_l203_20305

/-- Given a ratio of M&M candies to Starbursts candies and a number of Starbursts,
    calculate the number of M&M candies. -/
def calculate_mm_candies (mm_ratio : ℕ) (starburst_ratio : ℕ) (starburst_count : ℕ) : ℕ :=
  (starburst_count / starburst_ratio) * mm_ratio

/-- Theorem stating that given the specific ratio and Starburst count,
    the number of M&M candies is 25. -/
theorem penelope_candy_count :
  calculate_mm_candies 5 3 15 = 25 := by
  sorry

end NUMINAMATH_CALUDE_penelope_candy_count_l203_20305


namespace NUMINAMATH_CALUDE_village_leadership_choices_l203_20358

/-- The number of members in the village -/
def villageSize : ℕ := 16

/-- The number of deputy mayors -/
def numDeputyMayors : ℕ := 3

/-- The number of council members per deputy mayor -/
def councilMembersPerDeputy : ℕ := 3

/-- The total number of council members -/
def totalCouncilMembers : ℕ := numDeputyMayors * councilMembersPerDeputy

/-- The number of ways to choose the village leadership -/
def leadershipChoices : ℕ := 
  villageSize * 
  (villageSize - 1) * 
  (villageSize - 2) * 
  (villageSize - 3) * 
  Nat.choose (villageSize - 4) councilMembersPerDeputy * 
  Nat.choose (villageSize - 4 - councilMembersPerDeputy) councilMembersPerDeputy * 
  Nat.choose (villageSize - 4 - 2 * councilMembersPerDeputy) councilMembersPerDeputy

theorem village_leadership_choices : 
  leadershipChoices = 154828800 := by sorry

end NUMINAMATH_CALUDE_village_leadership_choices_l203_20358


namespace NUMINAMATH_CALUDE_expression_simplification_l203_20352

theorem expression_simplification (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  (x + 2) * (y - 2) - 2 * (x * y - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l203_20352


namespace NUMINAMATH_CALUDE_min_value_theorem_l203_20338

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 1) :
  (1 / a + 1 / b ≥ 2 * Real.sqrt 2) ∧ (b / a^3 + a / b^3 ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l203_20338


namespace NUMINAMATH_CALUDE_overtake_time_problem_l203_20370

/-- Proves that under given conditions, k started 10 hours after a. -/
theorem overtake_time_problem (speed_a speed_b speed_k : ℝ) 
  (start_delay_b : ℝ) (overtake_time : ℝ) :
  speed_a = 30 →
  speed_b = 40 →
  speed_k = 60 →
  start_delay_b = 5 →
  speed_a * overtake_time = speed_b * (overtake_time - start_delay_b) →
  speed_a * overtake_time = speed_k * (overtake_time - (overtake_time - 10)) →
  overtake_time - (overtake_time - 10) = 10 :=
by sorry

end NUMINAMATH_CALUDE_overtake_time_problem_l203_20370


namespace NUMINAMATH_CALUDE_smallest_difference_l203_20340

def Digits : Finset Nat := {0, 2, 4, 5, 7}

def is_valid_arrangement (a b c d x y z : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧
  x ∈ Digits ∧ y ∈ Digits ∧ z ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ x ∧ a ≠ y ∧ a ≠ z ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ x ∧ b ≠ y ∧ b ≠ z ∧
  c ≠ d ∧ c ≠ x ∧ c ≠ y ∧ c ≠ z ∧
  d ≠ x ∧ d ≠ y ∧ d ≠ z ∧
  x ≠ y ∧ x ≠ z ∧
  y ≠ z ∧
  a ≠ 0 ∧ x ≠ 0

def difference (a b c d x y z : Nat) : Nat :=
  1000 * a + 100 * b + 10 * c + d - (100 * x + 10 * y + z)

theorem smallest_difference :
  ∀ a b c d x y z,
    is_valid_arrangement a b c d x y z →
    difference a b c d x y z ≥ 1325 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_l203_20340


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l203_20395

/-- Calculates the final price of a shirt given its original cost, profit margin, and discount percentage. -/
def final_price (original_cost : ℝ) (profit_margin : ℝ) (discount : ℝ) : ℝ :=
  let selling_price := original_cost * (1 + profit_margin)
  selling_price * (1 - discount)

/-- Theorem stating that a shirt with an original cost of $20, a 30% profit margin, and a 50% discount has a final price of $13. -/
theorem shirt_price_calculation :
  final_price 20 0.3 0.5 = 13 := by
  sorry

#eval final_price 20 0.3 0.5

end NUMINAMATH_CALUDE_shirt_price_calculation_l203_20395


namespace NUMINAMATH_CALUDE_eighth_square_fully_shaded_l203_20323

/-- Represents the number of shaded squares and total squares in the nth diagram -/
def squarePattern (n : ℕ) : ℕ := n^2

/-- The fraction of shaded squares in the nth diagram -/
def shadedFraction (n : ℕ) : ℚ := squarePattern n / squarePattern n

theorem eighth_square_fully_shaded :
  shadedFraction 8 = 1 := by sorry

end NUMINAMATH_CALUDE_eighth_square_fully_shaded_l203_20323


namespace NUMINAMATH_CALUDE_store_annual_profits_l203_20399

/-- Calculates the annual profits given the profits for each quarter -/
def annual_profits (q1 q2 q3 q4 : ℕ) : ℕ :=
  q1 + q2 + q3 + q4

/-- Theorem stating that the annual profits are $8,000 given the quarterly profits -/
theorem store_annual_profits :
  let q1 : ℕ := 1500
  let q2 : ℕ := 1500
  let q3 : ℕ := 3000
  let q4 : ℕ := 2000
  annual_profits q1 q2 q3 q4 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_store_annual_profits_l203_20399


namespace NUMINAMATH_CALUDE_permutations_mod_1000_l203_20324

/-- The number of characters in the string -/
def n : ℕ := 16

/-- The number of A's in the string -/
def num_a : ℕ := 4

/-- The number of B's in the string -/
def num_b : ℕ := 5

/-- The number of C's in the string -/
def num_c : ℕ := 4

/-- The number of D's in the string -/
def num_d : ℕ := 3

/-- The number of positions where A's cannot be placed -/
def no_a_positions : ℕ := 5

/-- The number of positions where B's cannot be placed -/
def no_b_positions : ℕ := 5

/-- The number of positions where C's and D's cannot be placed -/
def no_cd_positions : ℕ := 6

/-- The function that calculates the number of permutations satisfying the conditions -/
def permutations : ℕ :=
  (Nat.choose no_cd_positions num_d) *
  (Nat.choose (no_cd_positions - num_d) (num_c - (no_cd_positions - num_d))) *
  (Nat.choose no_a_positions num_b) *
  (Nat.choose no_b_positions num_a)

theorem permutations_mod_1000 :
  permutations ≡ 75 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_permutations_mod_1000_l203_20324


namespace NUMINAMATH_CALUDE_base_problem_l203_20372

theorem base_problem (b : ℕ) : (3 * b + 1)^2 = b^3 + 2 * b + 1 → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_problem_l203_20372


namespace NUMINAMATH_CALUDE_sample_customers_l203_20388

theorem sample_customers (samples_per_box : ℕ) (boxes_opened : ℕ) (samples_left : ℕ) : 
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_left = 5 →
  (samples_per_box * boxes_opened - samples_left) = 235 :=
by
  sorry

end NUMINAMATH_CALUDE_sample_customers_l203_20388


namespace NUMINAMATH_CALUDE_congruence_problem_l203_20348

theorem congruence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n < 25 ∧ -175 ≡ n [ZMOD 25] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l203_20348


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l203_20350

theorem isosceles_triangle_perimeter (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) →
  (∃ (base leg : ℝ), 
    (base^2 - 6*base + 8 = 0) ∧ 
    (leg^2 - 6*leg + 8 = 0) ∧
    (base ≠ leg) ∧
    (base + 2*leg = 10)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l203_20350


namespace NUMINAMATH_CALUDE_operation_result_l203_20306

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation
def op : Element → Element → Element
  | Element.one, Element.one => Element.two
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.four
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.two
  | Element.two, Element.four => Element.four
  | Element.three, Element.one => Element.four
  | Element.three, Element.two => Element.two
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.four
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.two

theorem operation_result : 
  op (op Element.three Element.one) (op Element.four Element.two) = Element.two := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l203_20306


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l203_20349

/-- Given two lines p and q that intersect at a point, prove the value of k -/
theorem intersecting_lines_k_value (k : ℝ) : 
  let p : ℝ → ℝ := λ x => -2 * x + 3
  let q : ℝ → ℝ := λ x => k * x + 9
  (p 6 = -9) ∧ (q 6 = -9) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l203_20349


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l203_20341

theorem quadratic_roots_condition (n : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + n*x + 9 = 0 ∧ y^2 + n*y + 9 = 0) ↔ 
  n < -6 ∨ n > 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l203_20341


namespace NUMINAMATH_CALUDE_min_value_quadratic_l203_20377

theorem min_value_quadratic (x : ℝ) : 
  (∀ x, x^2 + 6*x ≥ -9) ∧ (∃ x, x^2 + 6*x = -9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l203_20377


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l203_20329

theorem complex_subtraction_simplification :
  (7 : ℂ) - 5*I - ((3 : ℂ) - 7*I) = (4 : ℂ) + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l203_20329


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percentage_l203_20335

theorem shopkeeper_loss_percentage
  (profit_rate : ℝ)
  (theft_rate : ℝ)
  (h_profit : profit_rate = 0.1)
  (h_theft : theft_rate = 0.2) :
  let selling_price := 1 + profit_rate
  let remaining_goods := 1 - theft_rate
  let cost_price_remaining := remaining_goods
  let selling_price_remaining := selling_price * remaining_goods
  let loss := theft_rate
  loss / cost_price_remaining = 0.25 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_percentage_l203_20335


namespace NUMINAMATH_CALUDE_bouquet_count_l203_20373

/-- The number of narcissus flowers available -/
def narcissus : ℕ := 75

/-- The number of chrysanthemums available -/
def chrysanthemums : ℕ := 90

/-- The number of flowers in each bouquet -/
def flowers_per_bouquet : ℕ := 5

/-- The total number of bouquets that can be made -/
def total_bouquets : ℕ := (narcissus / flowers_per_bouquet) + (chrysanthemums / flowers_per_bouquet)

theorem bouquet_count : total_bouquets = 33 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_count_l203_20373


namespace NUMINAMATH_CALUDE_prove_january_salary_l203_20318

def january_salary (feb mar apr may : ℕ) : Prop :=
  let jan := 32000 - (feb + mar + apr)
  (feb + mar + apr + may) / 4 = 8100 ∧
  (jan + feb + mar + apr) / 4 = 8000 ∧
  may = 6500 →
  jan = 6100

theorem prove_january_salary :
  ∀ (feb mar apr may : ℕ),
  january_salary feb mar apr may :=
by
  sorry

end NUMINAMATH_CALUDE_prove_january_salary_l203_20318


namespace NUMINAMATH_CALUDE_all_roots_of_polynomial_l203_20397

/-- The polynomial function we're considering -/
def f (x : ℝ) : ℝ := x^3 - x^2 - 4*x + 4

/-- The set of roots we claim are correct -/
def roots : Set ℝ := {-2, 1, 2}

/-- Theorem stating that the given set contains all roots of the polynomial -/
theorem all_roots_of_polynomial :
  ∀ x : ℝ, f x = 0 ↔ x ∈ roots := by sorry

end NUMINAMATH_CALUDE_all_roots_of_polynomial_l203_20397


namespace NUMINAMATH_CALUDE_original_element_l203_20390

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

/-- Theorem: If f(x, y) = (3, 1), then (x, y) = (1, 1) -/
theorem original_element (x y : ℝ) (h : f (x, y) = (3, 1)) : (x, y) = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_original_element_l203_20390


namespace NUMINAMATH_CALUDE_opposite_numbers_l203_20304

theorem opposite_numbers : ((-5)^2 : ℤ) = -(-5^2) :=
sorry

end NUMINAMATH_CALUDE_opposite_numbers_l203_20304


namespace NUMINAMATH_CALUDE_power_expansion_l203_20383

theorem power_expansion (x : ℝ) : (3*x)^2 * x^2 = 9*x^4 := by
  sorry

end NUMINAMATH_CALUDE_power_expansion_l203_20383


namespace NUMINAMATH_CALUDE_diameter_segments_length_l203_20313

theorem diameter_segments_length (r : ℝ) (chord_length : ℝ) :
  r = 6 ∧ chord_length = 10 →
  ∃ (a b : ℝ), a + b = 2 * r ∧ a * b = (chord_length / 2) ^ 2 ∧
  a = 6 - Real.sqrt 11 ∧ b = 6 + Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_diameter_segments_length_l203_20313


namespace NUMINAMATH_CALUDE_fifty_bees_honey_production_l203_20364

/-- The amount of honey (in grams) produced by a given number of bees in 50 days -/
def honey_production (num_bees : ℕ) : ℕ :=
  num_bees * 1

theorem fifty_bees_honey_production :
  honey_production 50 = 50 := by sorry

end NUMINAMATH_CALUDE_fifty_bees_honey_production_l203_20364


namespace NUMINAMATH_CALUDE_overlap_rectangle_area_l203_20374

theorem overlap_rectangle_area : 
  let rect1_width : ℝ := 8
  let rect1_height : ℝ := 10
  let rect2_width : ℝ := 9
  let rect2_height : ℝ := 12
  let overlap_area : ℝ := 37
  let rect1_area : ℝ := rect1_width * rect1_height
  let rect2_area : ℝ := rect2_width * rect2_height
  let grey_area : ℝ := rect2_area - (rect1_area - overlap_area)
  grey_area = 65 := by
sorry

end NUMINAMATH_CALUDE_overlap_rectangle_area_l203_20374


namespace NUMINAMATH_CALUDE_min_value_sum_l203_20310

theorem min_value_sum (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h : 1/p + 1/q + 1/r = 1) : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 1/x + 1/y + 1/z = 1 → p + q + r ≤ x + y + z ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a + 1/b + 1/c = 1 ∧ a + b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l203_20310


namespace NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l203_20315

-- Define the matrix A
def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 3; 5, d]

-- State the theorem
theorem matrix_inverse_scalar_multiple
  (d k : ℝ) :
  (A d)⁻¹ = k • (A d) →
  d = -2 ∧ k = 1/19 :=
sorry

end NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l203_20315


namespace NUMINAMATH_CALUDE_crazy_silly_school_movies_l203_20334

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 11

/-- The difference between the number of movies and books -/
def movie_book_difference : ℕ := 6

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := num_books + movie_book_difference

theorem crazy_silly_school_movies :
  num_movies = 17 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_movies_l203_20334


namespace NUMINAMATH_CALUDE_siblings_total_age_l203_20333

/-- Given the age ratio of Halima, Beckham, and Michelle as 4:3:7, and the age difference
    between Halima and Beckham as 9 years, prove that the total age of the three siblings
    is 126 years. -/
theorem siblings_total_age
  (halima_ratio : ℕ) (beckham_ratio : ℕ) (michelle_ratio : ℕ)
  (age_ratio : halima_ratio = 4 ∧ beckham_ratio = 3 ∧ michelle_ratio = 7)
  (age_difference : ℕ) (halima_beckham_diff : age_difference = 9)
  : ∃ (x : ℕ), 
    halima_ratio * x - beckham_ratio * x = age_difference ∧
    halima_ratio * x + beckham_ratio * x + michelle_ratio * x = 126 := by
  sorry

end NUMINAMATH_CALUDE_siblings_total_age_l203_20333


namespace NUMINAMATH_CALUDE_alicia_tax_deduction_l203_20331

/-- Calculates the tax deduction in cents given an hourly wage in dollars and a tax rate percentage. -/
def tax_deduction_cents (hourly_wage : ℚ) (tax_rate_percent : ℚ) : ℚ :=
  hourly_wage * 100 * (tax_rate_percent / 100)

/-- Proves that Alicia's tax deduction is 50 cents per hour. -/
theorem alicia_tax_deduction :
  tax_deduction_cents 25 2 = 50 := by
  sorry

#eval tax_deduction_cents 25 2

end NUMINAMATH_CALUDE_alicia_tax_deduction_l203_20331


namespace NUMINAMATH_CALUDE_ravenswood_gnomes_remaining_l203_20361

/-- The number of gnomes in Westerville woods -/
def westerville_gnomes : ℕ := 20

/-- The ratio of gnomes in Ravenswood forest compared to Westerville woods -/
def ravenswood_ratio : ℕ := 4

/-- The percentage of gnomes taken by the forest owner -/
def taken_percentage : ℚ := 40 / 100

/-- The number of gnomes remaining in Ravenswood forest after some are taken -/
def remaining_ravenswood_gnomes : ℕ := 48

theorem ravenswood_gnomes_remaining :
  remaining_ravenswood_gnomes = 
    (ravenswood_ratio * westerville_gnomes) - 
    (ravenswood_ratio * westerville_gnomes * taken_percentage).floor := by
  sorry

end NUMINAMATH_CALUDE_ravenswood_gnomes_remaining_l203_20361


namespace NUMINAMATH_CALUDE_cubic_yard_to_cubic_inches_l203_20375

-- Define the conversion factor
def inches_per_yard : ℕ := 36

-- Theorem statement
theorem cubic_yard_to_cubic_inches :
  (inches_per_yard ^ 3 : ℕ) = 46656 :=
sorry

end NUMINAMATH_CALUDE_cubic_yard_to_cubic_inches_l203_20375


namespace NUMINAMATH_CALUDE_equation_solution_exists_l203_20330

theorem equation_solution_exists : ∃ (x y z : ℕ+), x + y + z + 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l203_20330


namespace NUMINAMATH_CALUDE_right_angled_triangle_set_l203_20382

theorem right_angled_triangle_set : ∃ (a b c : ℝ), 
  (a = Real.sqrt 2 ∧ b = Real.sqrt 3 ∧ c = Real.sqrt 5) ∧ 
  a^2 + b^2 = c^2 ∧ 
  (∀ (x y z : ℝ), 
    ((x = Real.sqrt 3 ∧ y = 2 ∧ z = Real.sqrt 5) ∨ 
     (x = 3 ∧ y = 4 ∧ z = 5) ∨ 
     (x = 1 ∧ y = 2 ∧ z = 3)) → 
    x^2 + y^2 ≠ z^2) :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_set_l203_20382


namespace NUMINAMATH_CALUDE_pencil_cost_l203_20347

/-- Given that 150 pencils cost $45, prove that 3200 pencils cost $960 -/
theorem pencil_cost (box_size : ℕ) (box_cost : ℚ) (target_quantity : ℕ) :
  box_size = 150 →
  box_cost = 45 →
  target_quantity = 3200 →
  (target_quantity : ℚ) * (box_cost / box_size) = 960 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l203_20347


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l203_20369

/-- Given a line L1 with equation x - y + 2 = 0 and a point P (1, 0),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation x + y - 1 = 0 -/
theorem perpendicular_line_through_point (L1 L2 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  (L1 = {(x, y) | x - y + 2 = 0}) →
  (P = (1, 0)) →
  (L2 = {(x, y) | (x, y) ∈ L2 ∧ (∀ (a b : ℝ × ℝ), a ∈ L1 → b ∈ L1 → (a.1 - b.1) * (P.1 - x) + (a.2 - b.2) * (P.2 - y) = 0)}) →
  (L2 = {(x, y) | x + y - 1 = 0}) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l203_20369


namespace NUMINAMATH_CALUDE_product_sixty_sum_diff_equality_l203_20371

theorem product_sixty_sum_diff_equality (A B C D : ℕ+) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * B = 60 →
  C * D = 60 →
  A - B = C + D →
  A = 20 := by
sorry

end NUMINAMATH_CALUDE_product_sixty_sum_diff_equality_l203_20371


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l203_20319

theorem cube_surface_area_increase (s : ℝ) (h : s > 0) :
  let original_area := 6 * s^2
  let new_edge := 1.6 * s
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 1.56 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l203_20319


namespace NUMINAMATH_CALUDE_problem_statement_l203_20376

theorem problem_statement (a b x y : ℝ) 
  (sum_ab : a + b = 2)
  (sum_xy : x + y = 2)
  (product_sum : a * x + b * y = 5) :
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = -5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l203_20376


namespace NUMINAMATH_CALUDE_square_root_product_l203_20392

theorem square_root_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_square_root_product_l203_20392


namespace NUMINAMATH_CALUDE_cycling_route_length_l203_20342

/-- The total length of a rectangular cycling route -/
def total_length (upper_horizontal : ℝ) (left_vertical : ℝ) : ℝ :=
  2 * (upper_horizontal + left_vertical)

/-- Theorem: The total length of the cycling route is 52 km -/
theorem cycling_route_length :
  let upper_horizontal := 4 + 7 + 2
  let left_vertical := 6 + 7
  total_length upper_horizontal left_vertical = 52 := by
  sorry

end NUMINAMATH_CALUDE_cycling_route_length_l203_20342


namespace NUMINAMATH_CALUDE_terrell_hike_distance_l203_20368

theorem terrell_hike_distance (saturday_distance sunday_distance : ℝ) 
  (h1 : saturday_distance = 8.2)
  (h2 : sunday_distance = 1.6) :
  saturday_distance + sunday_distance = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_terrell_hike_distance_l203_20368


namespace NUMINAMATH_CALUDE_max_squares_after_triangles_l203_20387

/-- Represents the number of matchsticks used to form triangles efficiently -/
def triangleMatchsticks : ℕ := 13

/-- Represents the total number of matchsticks available -/
def totalMatchsticks : ℕ := 24

/-- Represents the number of matchsticks required to form a square -/
def matchsticksPerSquare : ℕ := 4

/-- Represents the number of triangles to be formed -/
def numTriangles : ℕ := 6

/-- Theorem stating the maximum number of squares that can be formed -/
theorem max_squares_after_triangles :
  (totalMatchsticks - triangleMatchsticks) / matchsticksPerSquare = 4 :=
sorry

end NUMINAMATH_CALUDE_max_squares_after_triangles_l203_20387


namespace NUMINAMATH_CALUDE_inequality_solution_set_l203_20380

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, (a * x) / (x - 1) < 1 ↔ (x < b ∨ x > 3)) →
  (a * 3) / (3 - 1) = 1 →
  a - b = -1/3 := by
    sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l203_20380


namespace NUMINAMATH_CALUDE_light_year_scientific_notation_l203_20354

def light_year : ℝ := 9500000000000

theorem light_year_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), light_year = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = 12 ∧ a = 9.5 :=
by sorry

end NUMINAMATH_CALUDE_light_year_scientific_notation_l203_20354


namespace NUMINAMATH_CALUDE_equation_solution_l203_20311

theorem equation_solution : ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ x = 105 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l203_20311


namespace NUMINAMATH_CALUDE_dice_roll_probability_l203_20391

def is_valid_roll (a b : Nat) : Prop :=
  a ≤ 6 ∧ b ≤ 6 ∧ a + b ≤ 10 ∧ (a > 3 ∨ b > 3)

def total_outcomes : Nat := 36

def valid_outcomes : Nat := 24

theorem dice_roll_probability : 
  (valid_outcomes : ℚ) / total_outcomes = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l203_20391


namespace NUMINAMATH_CALUDE_system_solutions_l203_20337

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  4 * x^2 / (1 + 4 * x^2) = y ∧
  4 * y^2 / (1 + 4 * y^2) = z ∧
  4 * z^2 / (1 + 4 * z^2) = x

-- Theorem statement
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l203_20337


namespace NUMINAMATH_CALUDE_log_range_l203_20359

def log_defined (a : ℝ) : Prop :=
  a - 2 > 0 ∧ a - 2 ≠ 1 ∧ 5 - a > 0

theorem log_range : 
  {a : ℝ | log_defined a} = {a : ℝ | (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5)} :=
by sorry

end NUMINAMATH_CALUDE_log_range_l203_20359


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l203_20326

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter (small large : Triangle) :
  small.isIsosceles ∧
  small.a = 15 ∧ small.b = 15 ∧ small.c = 6 ∧
  small.isSimilar large ∧
  large.c = 18 →
  large.perimeter = 108 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l203_20326


namespace NUMINAMATH_CALUDE_min_quotient_value_l203_20378

def is_valid_number (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧
  1 ≤ b ∧ b ≤ 9 ∧
  1 ≤ c ∧ c ≤ 9 ∧
  1 ≤ d ∧ d ≤ 9 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a = c + 1 ∧
  b = d + 1

def number_value (a b c d : ℕ) : ℕ :=
  1000 * a + 100 * b + 10 * c + d

def digit_sum (a b c d : ℕ) : ℕ :=
  a + b + c + d

def quotient (a b c d : ℕ) : ℚ :=
  (number_value a b c d : ℚ) / (digit_sum a b c d : ℚ)

theorem min_quotient_value :
  ∀ a b c d : ℕ, is_valid_number a b c d →
  quotient a b c d ≥ 192.67 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_value_l203_20378


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l203_20396

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℤ)
  (h_arithmetic : isArithmeticSequence a)
  (h_a4 : a 4 = -4)
  (h_a8 : a 8 = 4) :
  a 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l203_20396


namespace NUMINAMATH_CALUDE_product_of_base8_digits_7354_l203_20385

/-- The base 8 representation of a natural number -/
def base8Representation (n : ℕ) : List ℕ :=
  sorry

/-- The product of a list of natural numbers -/
def productList (l : List ℕ) : ℕ :=
  sorry

theorem product_of_base8_digits_7354 :
  productList (base8Representation 7354) = 0 :=
sorry

end NUMINAMATH_CALUDE_product_of_base8_digits_7354_l203_20385


namespace NUMINAMATH_CALUDE_maurice_job_search_l203_20356

/-- The probability of a single application being accepted -/
def p_accept : ℚ := 1 / 5

/-- The probability threshold for stopping -/
def p_threshold : ℚ := 3 / 4

/-- The number of letters Maurice needs to write -/
def num_letters : ℕ := 7

theorem maurice_job_search :
  (1 - (1 - p_accept) ^ num_letters) ≥ p_threshold ∧
  ∀ n : ℕ, n < num_letters → (1 - (1 - p_accept) ^ n) < p_threshold :=
by sorry

end NUMINAMATH_CALUDE_maurice_job_search_l203_20356


namespace NUMINAMATH_CALUDE_inequality_equivalence_l203_20317

theorem inequality_equivalence (n : ℕ) (hn : n > 0) :
  (2 * n - 1 : ℝ) * Real.log (1 + Real.log 2023 / Real.log 2) > 
  Real.log 2023 / Real.log 2 * (Real.log 2 + Real.log n) ↔ 
  n ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l203_20317


namespace NUMINAMATH_CALUDE_ab_and_a_reciprocal_b_relationship_l203_20346

theorem ab_and_a_reciprocal_b_relationship (a b : ℝ) (h : a * b ≠ 0) :
  ¬(∀ a b, a * b > 1 → a > 1 / b) ∧ 
  ¬(∀ a b, a > 1 / b → a * b > 1) ∧
  ¬(∀ a b, a * b > 1 ↔ a > 1 / b) :=
by sorry

end NUMINAMATH_CALUDE_ab_and_a_reciprocal_b_relationship_l203_20346


namespace NUMINAMATH_CALUDE_equation_solution_l203_20300

theorem equation_solution (x : ℝ) : 
  x^6 - 22*x^2 - Real.sqrt 21 = 0 ↔ x = Real.sqrt ((Real.sqrt 21 + 5)/2) ∨ x = -Real.sqrt ((Real.sqrt 21 + 5)/2) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l203_20300


namespace NUMINAMATH_CALUDE_james_muffins_count_l203_20336

def arthur_muffins : ℝ := 115.0
def baking_ratio : ℝ := 12.0

theorem james_muffins_count : 
  arthur_muffins / baking_ratio = 9.5833 := by sorry

end NUMINAMATH_CALUDE_james_muffins_count_l203_20336


namespace NUMINAMATH_CALUDE_pi_estimation_l203_20393

theorem pi_estimation (n : ℕ) (m : ℕ) (h1 : n = 120) (h2 : m = 34) :
  let π_estimate := 4 * (m / n + 1 / 2)
  π_estimate = 47 / 15 := by
  sorry

end NUMINAMATH_CALUDE_pi_estimation_l203_20393


namespace NUMINAMATH_CALUDE_mrs_snyder_pink_cookies_l203_20357

/-- The total number of cookies Mrs. Snyder made -/
def total_cookies : ℕ := 86

/-- The number of red cookies Mrs. Snyder made -/
def red_cookies : ℕ := 36

/-- The number of pink cookies Mrs. Snyder made -/
def pink_cookies : ℕ := total_cookies - red_cookies

theorem mrs_snyder_pink_cookies : pink_cookies = 50 := by
  sorry

end NUMINAMATH_CALUDE_mrs_snyder_pink_cookies_l203_20357


namespace NUMINAMATH_CALUDE_negation_is_true_l203_20355

theorem negation_is_true : 
  (∀ x : ℝ, x^2 ≥ 1 → (x ≤ -1 ∨ x ≥ 1)) := by sorry

end NUMINAMATH_CALUDE_negation_is_true_l203_20355


namespace NUMINAMATH_CALUDE_max_value_of_A_l203_20301

theorem max_value_of_A (A B : ℕ) (h1 : A = 5 * 2 + B) (h2 : B < 5) : A ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_A_l203_20301
