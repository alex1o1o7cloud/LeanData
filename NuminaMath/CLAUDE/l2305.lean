import Mathlib

namespace NUMINAMATH_CALUDE_piggy_bank_sequence_l2305_230583

theorem piggy_bank_sequence (sequence : Fin 6 → ℕ) 
  (h1 : sequence 0 = 72)
  (h2 : sequence 1 = 81)
  (h4 : sequence 3 = 99)
  (h5 : sequence 4 = 108)
  (h6 : sequence 5 = 117)
  (h_arithmetic : ∀ i : Fin 5, sequence (i + 1) - sequence i = sequence 1 - sequence 0) :
  sequence 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_sequence_l2305_230583


namespace NUMINAMATH_CALUDE_unique_solution_l2305_230550

/-- Definition of the diamond operation -/
def diamond (a b c d : ℝ) : ℝ × ℝ :=
  (a * c - b * d, a * d + b * c)

/-- Theorem stating the unique solution to the equation -/
theorem unique_solution :
  ∀ x y : ℝ, diamond x 3 x y = (6, 0) ↔ x = 0 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2305_230550


namespace NUMINAMATH_CALUDE_washer_dryer_cost_difference_l2305_230544

theorem washer_dryer_cost_difference :
  ∀ (washer_cost dryer_cost : ℝ),
    dryer_cost = 490 →
    washer_cost > dryer_cost →
    washer_cost + dryer_cost = 1200 →
    washer_cost - dryer_cost = 220 :=
by
  sorry

end NUMINAMATH_CALUDE_washer_dryer_cost_difference_l2305_230544


namespace NUMINAMATH_CALUDE_largest_number_proof_l2305_230509

def is_hcf (a b h : ℕ) : Prop := h ∣ a ∧ h ∣ b ∧ ∀ k : ℕ, k ∣ a → k ∣ b → k ≤ h

def is_lcm (a b l : ℕ) : Prop := a ∣ l ∧ b ∣ l ∧ ∀ k : ℕ, a ∣ k → b ∣ k → l ∣ k

theorem largest_number_proof (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  is_hcf a b 23 → (∃ l : ℕ, is_lcm a b l ∧ 13 ∣ l ∧ 14 ∣ l) → max a b = 322 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_proof_l2305_230509


namespace NUMINAMATH_CALUDE_joint_purchase_effectiveness_l2305_230537

/-- Represents the benefits of joint purchases -/
structure JointPurchaseBenefits where
  cost_savings : ℝ
  quality_assessment : ℝ
  community_trust : ℝ

/-- Represents the drawbacks of joint purchases -/
structure JointPurchaseDrawbacks where
  transaction_costs : ℝ
  organizational_efforts : ℝ
  convenience_issues : ℝ
  potential_disputes : ℝ

/-- Represents the characteristics of a group making joint purchases -/
structure PurchaseGroup where
  size : ℕ
  is_localized : Bool

/-- Calculates the total benefit of joint purchases for a group -/
def calculate_total_benefit (benefits : JointPurchaseBenefits) (group : PurchaseGroup) : ℝ :=
  benefits.cost_savings + benefits.quality_assessment + benefits.community_trust

/-- Calculates the total drawback of joint purchases for a group -/
def calculate_total_drawback (drawbacks : JointPurchaseDrawbacks) (group : PurchaseGroup) : ℝ :=
  drawbacks.transaction_costs + drawbacks.organizational_efforts + drawbacks.convenience_issues + drawbacks.potential_disputes

/-- Theorem stating that joint purchases are beneficial for large groups but not for small, localized groups -/
theorem joint_purchase_effectiveness (benefits : JointPurchaseBenefits) (drawbacks : JointPurchaseDrawbacks) :
  ∀ (group : PurchaseGroup),
    (group.size > 100 → calculate_total_benefit benefits group > calculate_total_drawback drawbacks group) ∧
    (group.size ≤ 100 ∧ group.is_localized → calculate_total_benefit benefits group ≤ calculate_total_drawback drawbacks group) :=
by sorry

end NUMINAMATH_CALUDE_joint_purchase_effectiveness_l2305_230537


namespace NUMINAMATH_CALUDE_expression_simplification_l2305_230515

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 - 1/a) / ((a^2 - 2*a + 1)/a) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2305_230515


namespace NUMINAMATH_CALUDE_quadratic_property_l2305_230538

/-- A quadratic function with a real coefficient b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- The range of f is [0, +∞) -/
def has_nonnegative_range (b : ℝ) : Prop :=
  ∀ y, (∃ x, f b x = y) → y ≥ 0

/-- The solution set of f(x) < c is an open interval of length 8 -/
def has_solution_interval_of_length_eight (b c : ℝ) : Prop :=
  ∃ m, ∀ x, f b x < c ↔ m - 8 < x ∧ x < m

theorem quadratic_property (b : ℝ) (h1 : has_nonnegative_range b) 
  (h2 : has_solution_interval_of_length_eight b c) : c = 16 :=
sorry

end NUMINAMATH_CALUDE_quadratic_property_l2305_230538


namespace NUMINAMATH_CALUDE_total_age_proof_l2305_230569

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 28 years old
  Prove that the total of their ages is 72 years. -/
theorem total_age_proof (a b c : ℕ) : 
  b = 28 → a = b + 2 → b = 2 * c → a + b + c = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_age_proof_l2305_230569


namespace NUMINAMATH_CALUDE_fourth_term_of_specific_gp_l2305_230512

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem fourth_term_of_specific_gp :
  let a₁ := 2
  let a₂ := 2 * Real.sqrt 2
  let a₃ := 4
  let r := a₂ / a₁
  let a₄ := geometric_progression a₁ r 4
  a₄ = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_fourth_term_of_specific_gp_l2305_230512


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2305_230560

theorem complex_fraction_equality : 
  2013 * (5.7 * 4.2 + (21/5) * 4.3) / ((14/73) * 15 + (5/73) * 177 + 656) = 126 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2305_230560


namespace NUMINAMATH_CALUDE_correct_average_l2305_230575

theorem correct_average (n : ℕ) (initial_avg : ℚ) (correction1 : ℚ) (wrong2 : ℚ) (correct2 : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  correction1 = 19 →
  wrong2 = 13 →
  correct2 = 31 →
  let initial_sum := n * initial_avg
  let corrected_sum := initial_sum - correction1 - wrong2 + correct2
  let corrected_avg := corrected_sum / n
  corrected_avg = 40.1 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l2305_230575


namespace NUMINAMATH_CALUDE_average_marks_l2305_230536

theorem average_marks (total_subjects : ℕ) (subjects_avg : ℕ) (last_subject_mark : ℕ) :
  total_subjects = 6 →
  subjects_avg = 74 →
  last_subject_mark = 110 →
  (subjects_avg * (total_subjects - 1) + last_subject_mark) / total_subjects = 80 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_l2305_230536


namespace NUMINAMATH_CALUDE_point_on_number_line_l2305_230571

theorem point_on_number_line (x : ℝ) : 
  abs x = 5.5 → x = 5.5 ∨ x = -5.5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_number_line_l2305_230571


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l2305_230519

theorem geometric_sequence_minimum (b₁ b₂ b₃ : ℝ) : 
  b₁ = 1 → (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) → 
  (∀ b₂' b₃' : ℝ, (∃ r' : ℝ, b₂' = b₁ * r' ∧ b₃' = b₂' * r') → 
    3 * b₂ + 4 * b₃ ≤ 3 * b₂' + 4 * b₃') →
  3 * b₂ + 4 * b₃ = -9/16 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l2305_230519


namespace NUMINAMATH_CALUDE_choose_four_from_six_l2305_230510

theorem choose_four_from_six : Nat.choose 6 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_six_l2305_230510


namespace NUMINAMATH_CALUDE_art_collection_area_is_282_l2305_230547

/-- Calculates the total area of Davonte's art collection -/
def art_collection_area : ℕ :=
  let square_painting_area := 3 * (6 * 6)
  let small_painting_area := 4 * (2 * 3)
  let large_painting_area := 10 * 15
  square_painting_area + small_painting_area + large_painting_area

/-- Proves that the total area of Davonte's art collection is 282 square feet -/
theorem art_collection_area_is_282 : art_collection_area = 282 := by
  sorry

end NUMINAMATH_CALUDE_art_collection_area_is_282_l2305_230547


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2305_230565

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^(n - 1)

/-- The property that 2a_2 + a_3 = a_4 for a geometric sequence -/
def property1 (a q : ℝ) : Prop :=
  2 * (geometric_sequence a q 2) + (geometric_sequence a q 3) = geometric_sequence a q 4

/-- The property that (a_2 + 1)(a_3 + 1) = a_5 - 1 for a geometric sequence -/
def property2 (a q : ℝ) : Prop :=
  (geometric_sequence a q 2 + 1) * (geometric_sequence a q 3 + 1) = geometric_sequence a q 5 - 1

/-- Theorem stating that for a geometric sequence satisfying both properties, a_1 ≠ 2 -/
theorem geometric_sequence_property (a q : ℝ) (h1 : property1 a q) (h2 : property2 a q) : a ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2305_230565


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l2305_230562

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = a * b) :
  a + 2 * b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l2305_230562


namespace NUMINAMATH_CALUDE_grunters_win_probability_l2305_230525

/-- The probability of winning a single game for the Grunters -/
def p : ℚ := 3/5

/-- The number of games played -/
def n : ℕ := 5

/-- The probability of winning all games -/
def win_all : ℚ := p^n

theorem grunters_win_probability : win_all = 243/3125 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l2305_230525


namespace NUMINAMATH_CALUDE_budget_remainder_l2305_230553

-- Define the given conditions
def weekly_budget : ℝ := 80
def fried_chicken_cost : ℝ := 12
def beef_pounds : ℝ := 4.5
def beef_price_per_pound : ℝ := 3
def soup_cans : ℕ := 3
def soup_cost_per_can : ℝ := 2
def milk_original_price : ℝ := 4
def milk_discount_percentage : ℝ := 0.1

-- Define the theorem
theorem budget_remainder : 
  let beef_cost := beef_pounds * beef_price_per_pound
  let soup_cost := (soup_cans - 1) * soup_cost_per_can
  let milk_cost := milk_original_price * (1 - milk_discount_percentage)
  let total_cost := fried_chicken_cost + beef_cost + soup_cost + milk_cost
  weekly_budget - total_cost = 46.90 := by
  sorry

end NUMINAMATH_CALUDE_budget_remainder_l2305_230553


namespace NUMINAMATH_CALUDE_last_digit_difference_l2305_230521

theorem last_digit_difference (p q : ℕ) : 
  p > q → 
  p % 10 ≠ 0 → 
  q % 10 ≠ 0 → 
  ∃ k : ℕ, p * q = 10^k → 
  (p - q) % 10 ≠ 5 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_difference_l2305_230521


namespace NUMINAMATH_CALUDE_one_fourth_of_eight_point_eight_l2305_230587

theorem one_fourth_of_eight_point_eight (x : ℚ) : x = 8.8 → (1 / 4 : ℚ) * x = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_eight_point_eight_l2305_230587


namespace NUMINAMATH_CALUDE_remainder_problem_l2305_230529

theorem remainder_problem : (8 * 7^19 + 1^19) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2305_230529


namespace NUMINAMATH_CALUDE_smallest_integer_negative_quadratic_l2305_230593

theorem smallest_integer_negative_quadratic :
  ∃ (x : ℤ), (∀ (y : ℤ), y^2 - 11*y + 24 < 0 → x ≤ y) ∧ x^2 - 11*x + 24 < 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_negative_quadratic_l2305_230593


namespace NUMINAMATH_CALUDE_traveler_money_problem_l2305_230552

/-- Represents the amount of money a traveler has at the start of each day -/
def money_at_day (initial_money : ℚ) : ℕ → ℚ
  | 0 => initial_money
  | n + 1 => (money_at_day initial_money n / 2) - 1

theorem traveler_money_problem (initial_money : ℚ) :
  (money_at_day initial_money 0 > 0) ∧
  (money_at_day initial_money 1 > 0) ∧
  (money_at_day initial_money 2 > 0) ∧
  (money_at_day initial_money 3 = 0) →
  initial_money = 14 := by
sorry

end NUMINAMATH_CALUDE_traveler_money_problem_l2305_230552


namespace NUMINAMATH_CALUDE_tan_addition_formula_l2305_230526

theorem tan_addition_formula (x : Real) (h : Real.tan x = 3) :
  Real.tan (x + π / 6) = 5 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_addition_formula_l2305_230526


namespace NUMINAMATH_CALUDE_train_speed_conversion_l2305_230516

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The train's speed in meters per second -/
def train_speed_mps : ℝ := 60.0048

/-- Theorem: Given a train's speed of 60.0048 meters per second, 
    its speed in kilometers per hour is equal to 216.01728 -/
theorem train_speed_conversion :
  train_speed_mps * mps_to_kmph = 216.01728 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l2305_230516


namespace NUMINAMATH_CALUDE_complex_power_abs_l2305_230564

theorem complex_power_abs : Complex.abs ((2 : ℂ) + 2 * Complex.I * Real.sqrt 3) ^ 6 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_abs_l2305_230564


namespace NUMINAMATH_CALUDE_six_selected_in_interval_l2305_230578

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ
  (population_positive : population > 0)
  (sample_size_positive : sample_size > 0)
  (sample_size_le_population : sample_size ≤ population)
  (interval_valid : interval_start ≤ interval_end)
  (interval_in_range : interval_end ≤ population)

/-- Calculates the number of selected individuals within a given interval -/
def selected_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) + (s.population / s.sample_size - 1)) / (s.population / s.sample_size)

/-- Theorem stating that for the given parameters, 6 individuals are selected within the interval -/
theorem six_selected_in_interval (s : SystematicSample) 
  (h_pop : s.population = 420)
  (h_sample : s.sample_size = 21)
  (h_start : s.interval_start = 241)
  (h_end : s.interval_end = 360) :
  selected_in_interval s = 6 := by
  sorry

#eval selected_in_interval {
  population := 420,
  sample_size := 21,
  interval_start := 241,
  interval_end := 360,
  population_positive := by norm_num,
  sample_size_positive := by norm_num,
  sample_size_le_population := by norm_num,
  interval_valid := by norm_num,
  interval_in_range := by norm_num
}

end NUMINAMATH_CALUDE_six_selected_in_interval_l2305_230578


namespace NUMINAMATH_CALUDE_morning_sales_is_eight_l2305_230543

/-- Represents the sale of souvenirs at the London Olympics --/
structure SouvenirSale where
  total_souvenirs : Nat
  morning_price : Nat
  afternoon_price : Nat
  morning_sales : Nat
  afternoon_sales : Nat

/-- Checks if the given SouvenirSale satisfies all conditions --/
def is_valid_sale (sale : SouvenirSale) : Prop :=
  sale.total_souvenirs = 24 ∧
  sale.morning_price = 7 ∧
  sale.morning_sales < sale.total_souvenirs / 2 ∧
  sale.morning_sales + sale.afternoon_sales = sale.total_souvenirs ∧
  sale.morning_sales * sale.morning_price + sale.afternoon_sales * sale.afternoon_price = 120

/-- Theorem: The number of souvenirs sold in the morning is 8 --/
theorem morning_sales_is_eight :
  ∃ (sale : SouvenirSale), is_valid_sale sale ∧ sale.morning_sales = 8 :=
sorry

end NUMINAMATH_CALUDE_morning_sales_is_eight_l2305_230543


namespace NUMINAMATH_CALUDE_negation_of_implication_l2305_230503

theorem negation_of_implication (a b : ℝ) :
  ¬(a = 0 → a * b = 0) ↔ (a ≠ 0 → a * b ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2305_230503


namespace NUMINAMATH_CALUDE_loan_amount_l2305_230567

/-- Proves that given the conditions of the loan, the sum lent must be 500 Rs. -/
theorem loan_amount (interest_rate : ℚ) (time : ℕ) (interest_difference : ℚ) : 
  interest_rate = 4/100 →
  time = 8 →
  interest_difference = 340 →
  ∃ (principal : ℚ), 
    principal * interest_rate * time = principal - interest_difference ∧
    principal = 500 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_l2305_230567


namespace NUMINAMATH_CALUDE_forty_bees_honey_l2305_230524

/-- The amount of honey (in grams) produced by one honey bee in 40 days -/
def honey_per_bee : ℕ := 1

/-- The number of honey bees -/
def num_bees : ℕ := 40

/-- The amount of honey (in grams) produced by a group of honey bees in 40 days -/
def total_honey (bees : ℕ) : ℕ := bees * honey_per_bee

/-- Theorem stating that 40 honey bees produce 40 grams of honey in 40 days -/
theorem forty_bees_honey : total_honey num_bees = 40 := by
  sorry

end NUMINAMATH_CALUDE_forty_bees_honey_l2305_230524


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l2305_230595

theorem candidate_vote_difference :
  let total_votes : ℝ := 25000.000000000007
  let candidate_percentage : ℝ := 0.4
  let rival_percentage : ℝ := 1 - candidate_percentage
  let candidate_votes : ℝ := total_votes * candidate_percentage
  let rival_votes : ℝ := total_votes * rival_percentage
  let vote_difference : ℝ := rival_votes - candidate_votes
  ∃ (ε : ℝ), ε > 0 ∧ |vote_difference - 5000| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l2305_230595


namespace NUMINAMATH_CALUDE_expression_simplification_l2305_230508

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2) :
  (a^2 - 1) / (a^2 - a) / (2 + (a^2 + 1) / a) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2305_230508


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_meaningful_l2305_230582

theorem sqrt_x_plus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_meaningful_l2305_230582


namespace NUMINAMATH_CALUDE_larger_integer_proof_l2305_230542

theorem larger_integer_proof (x y : ℕ) : 
  x > 0 → y > 0 → x - y = 8 → x * y = 272 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l2305_230542


namespace NUMINAMATH_CALUDE_correct_calculation_l2305_230594

theorem correct_calculation (x y : ℝ) : -4 * x * y + 3 * x * y = -x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2305_230594


namespace NUMINAMATH_CALUDE_living_room_curtain_length_l2305_230535

/-- Given the dimensions of a bolt of fabric, bedroom curtain, and living room curtain width,
    as well as the remaining fabric area, prove the length of the living room curtain. -/
theorem living_room_curtain_length
  (bolt_width : ℝ)
  (bolt_length : ℝ)
  (bedroom_width : ℝ)
  (bedroom_length : ℝ)
  (living_room_width : ℝ)
  (remaining_area : ℝ)
  (h1 : bolt_width = 16)
  (h2 : bolt_length = 12)
  (h3 : bedroom_width = 2)
  (h4 : bedroom_length = 4)
  (h5 : living_room_width = 4)
  (h6 : remaining_area = 160)
  (h7 : bolt_width * bolt_length - (bedroom_width * bedroom_length + living_room_width * living_room_length) = remaining_area) :
  living_room_length = 6 :=
by sorry

#check living_room_curtain_length

end NUMINAMATH_CALUDE_living_room_curtain_length_l2305_230535


namespace NUMINAMATH_CALUDE_range_of_x_l2305_230505

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) 
  (h3 : Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) : 
  x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4) := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l2305_230505


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2305_230506

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2305_230506


namespace NUMINAMATH_CALUDE_third_day_sale_l2305_230546

/-- Proves that given an average sale of 625 for 5 days, and sales of 435, 927, 230, and 562
    for 4 of those days, the sale on the remaining day must be 971. -/
theorem third_day_sale (average : ℕ) (day1 day2 day4 day5 : ℕ) :
  average = 625 →
  day1 = 435 →
  day2 = 927 →
  day4 = 230 →
  day5 = 562 →
  ∃ day3 : ℕ, day3 = 971 ∧ (day1 + day2 + day3 + day4 + day5) / 5 = average :=
by sorry

end NUMINAMATH_CALUDE_third_day_sale_l2305_230546


namespace NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l2305_230599

theorem shaded_area_of_concentric_circles :
  ∀ (r R : ℝ),
  R > 0 →
  r = R / 2 →
  π * R^2 = 100 * π →
  (π * R^2) / 2 + (π * r^2) / 2 = 62.5 * π :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l2305_230599


namespace NUMINAMATH_CALUDE_expression_value_l2305_230557

theorem expression_value : (45 - 13)^2 - (45^2 + 13^2) = -1170 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2305_230557


namespace NUMINAMATH_CALUDE_fraction_between_main_theorem_l2305_230585

theorem fraction_between (a b c d m n : ℕ) (h1 : 0 < b) (h2 : 0 < d) (h3 : 0 < n) :
  a * d < c * b → c * n < m * d → a * n < m * b →
  (a : ℚ) / b < (m : ℚ) / n ∧ (m : ℚ) / n < (c : ℚ) / d :=
by sorry

theorem main_theorem :
  (5 : ℚ) / 14 < (8 : ℚ) / 21 ∧ (8 : ℚ) / 21 < (5 : ℚ) / 12 :=
by sorry

end NUMINAMATH_CALUDE_fraction_between_main_theorem_l2305_230585


namespace NUMINAMATH_CALUDE_k_set_characterization_l2305_230527

/-- Given h = 2^r for some non-negative integer r, k(h) is the set of all natural numbers k
    such that there exist an odd natural number m > 1 and a natural number n where
    k divides m^k - 1 and m divides n^((n^k - 1)/k) + 1 -/
def k_set (h : ℕ) : Set ℕ :=
  {k : ℕ | ∃ (m n : ℕ), m > 1 ∧ m % 2 = 1 ∧ 
    (m^k - 1) % k = 0 ∧ (n^((n^k - 1)/k) + 1) % m = 0}

/-- For h = 2^r, the set k(h) is equal to {2^(r+s) * t | s, t ∈ ℕ, t is odd} -/
theorem k_set_characterization (r : ℕ) :
  k_set (2^r) = {k : ℕ | ∃ (s t : ℕ), k = 2^(r+s) * t ∧ t % 2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_k_set_characterization_l2305_230527


namespace NUMINAMATH_CALUDE_max_take_home_pay_l2305_230545

/-- The take-home pay function for a given income x (in thousands of dollars) -/
def takehomePay (x : ℝ) : ℝ := 1000 * x - 20 * x^2

/-- The income that maximizes take-home pay -/
def maxTakeHomeIncome : ℝ := 25

theorem max_take_home_pay :
  ∀ x : ℝ, takehomePay x ≤ takehomePay maxTakeHomeIncome :=
sorry

end NUMINAMATH_CALUDE_max_take_home_pay_l2305_230545


namespace NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l2305_230597

-- Define the conversion factor from yards to feet
def yards_to_feet : ℝ := 3

-- Define the volume in cubic yards
def volume_cubic_yards : ℝ := 7

-- Theorem: 7 cubic yards are equal to 189 cubic feet
theorem cubic_yards_to_cubic_feet :
  (volume_cubic_yards * yards_to_feet ^ 3 : ℝ) = 189 :=
by sorry

end NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l2305_230597


namespace NUMINAMATH_CALUDE_quadratic_roots_l2305_230541

theorem quadratic_roots (x : ℝ) : x^2 - 3*x + 2 = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2305_230541


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2305_230570

theorem trigonometric_identity (α β : ℝ) :
  1 - Real.sin α ^ 2 - Real.sin β ^ 2 + 2 * Real.sin α * Real.sin β * Real.cos (α - β) = 
  Real.cos (α - β) ^ 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2305_230570


namespace NUMINAMATH_CALUDE_equation_not_equivalent_l2305_230522

theorem equation_not_equivalent (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) ↔ 
  ¬((3*x + 2*y = x*y) ∨ 
    (y = 3*x/(5 - y)) ∨ 
    (x/3 + y/2 = 3) ∨ 
    (3*y/(y - 5) = x)) := by
  sorry

end NUMINAMATH_CALUDE_equation_not_equivalent_l2305_230522


namespace NUMINAMATH_CALUDE_movie_ticket_theorem_l2305_230511

def movie_ticket_problem (child_ticket_price adult_ticket_price : ℚ) : Prop :=
  let total_spent : ℚ := 30
  let num_child_tickets : ℕ := 4
  let num_adult_tickets : ℕ := 2
  let discount : ℚ := 2
  child_ticket_price = 4.25 ∧
  adult_ticket_price > child_ticket_price ∧
  num_child_tickets + num_adult_tickets > 3 ∧
  num_child_tickets * child_ticket_price + num_adult_tickets * adult_ticket_price - discount = total_spent ∧
  adult_ticket_price - child_ticket_price = 3.25

theorem movie_ticket_theorem :
  ∃ (adult_ticket_price : ℚ), movie_ticket_problem 4.25 adult_ticket_price :=
sorry

end NUMINAMATH_CALUDE_movie_ticket_theorem_l2305_230511


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2305_230533

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  fun n => a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (seq : ℕ → ℝ) :
  (∃ a₁ d : ℝ, seq = ArithmeticSequence a₁ d ∧ 
    seq 3 = 14 ∧ seq 6 = 32) →
  seq 10 = 56 ∧ (∃ d : ℝ, ∀ n : ℕ, seq (n + 1) - seq n = d ∧ d = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2305_230533


namespace NUMINAMATH_CALUDE_toy_production_lot_l2305_230518

theorem toy_production_lot (total : ℕ) 
  (h_red : total * 2 / 5 = total * 40 / 100)
  (h_small : total / 2 = total * 50 / 100)
  (h_red_small : total / 10 = total * 10 / 100)
  (h_red_large : total * 3 / 10 = 60) :
  total * 2 / 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_toy_production_lot_l2305_230518


namespace NUMINAMATH_CALUDE_peanut_price_in_mixed_nuts_l2305_230551

/-- Calculates the price per pound of peanuts in a mixed nut blend --/
theorem peanut_price_in_mixed_nuts
  (total_weight : ℝ)
  (mixed_price : ℝ)
  (cashew_weight : ℝ)
  (cashew_price : ℝ)
  (h1 : total_weight = 100)
  (h2 : mixed_price = 2.5)
  (h3 : cashew_weight = 60)
  (h4 : cashew_price = 4) :
  (total_weight * mixed_price - cashew_weight * cashew_price) / (total_weight - cashew_weight) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_peanut_price_in_mixed_nuts_l2305_230551


namespace NUMINAMATH_CALUDE_triangle_problem_l2305_230531

open Real

theorem triangle_problem (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  sin C * sin (A - B) = sin B * sin (C - A) ∧
  A = 2 * B →
  C = 5 * π / 8 ∧ 2 * a^2 = b^2 + c^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2305_230531


namespace NUMINAMATH_CALUDE_sammy_math_problems_l2305_230580

theorem sammy_math_problems (total : ℕ) (left : ℕ) (finished : ℕ) : 
  total = 9 → left = 7 → finished = total - left → finished = 2 := by
sorry

end NUMINAMATH_CALUDE_sammy_math_problems_l2305_230580


namespace NUMINAMATH_CALUDE_xiao_ying_score_l2305_230596

/-- Given an average score and a student's score relative to the average,
    calculate the student's actual score. -/
def calculate_score (average : ℕ) (relative_score : ℤ) : ℕ :=
  (average : ℤ) + relative_score |>.toNat

/-- The problem statement -/
theorem xiao_ying_score :
  let average_score : ℕ := 83
  let xiao_ying_relative_score : ℤ := -3
  calculate_score average_score xiao_ying_relative_score = 80 := by
  sorry

#eval calculate_score 83 (-3)

end NUMINAMATH_CALUDE_xiao_ying_score_l2305_230596


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l2305_230507

theorem final_sum_after_operations (a b S : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry


end NUMINAMATH_CALUDE_final_sum_after_operations_l2305_230507


namespace NUMINAMATH_CALUDE_integral_of_f_equals_seven_sixths_l2305_230532

-- Define the function f
def f (x : ℝ) (f'₁ : ℝ) : ℝ := f'₁ * x^2 + x + 1

-- State the theorem
theorem integral_of_f_equals_seven_sixths :
  ∃ (f'₁ : ℝ), (∫ x in (0:ℝ)..(1:ℝ), f x f'₁) = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_f_equals_seven_sixths_l2305_230532


namespace NUMINAMATH_CALUDE_min_pizzas_break_even_l2305_230577

/-- The minimum number of whole pizzas John must deliver to break even -/
def min_pizzas : ℕ := 1000

/-- The cost of the car -/
def car_cost : ℕ := 8000

/-- The earning per pizza -/
def earning_per_pizza : ℕ := 12

/-- The gas cost per pizza -/
def gas_cost_per_pizza : ℕ := 4

/-- Theorem stating that min_pizzas is the minimum number of whole pizzas
    John must deliver to at least break even on his car purchase -/
theorem min_pizzas_break_even :
  min_pizzas = (car_cost + gas_cost_per_pizza - 1) / (earning_per_pizza - gas_cost_per_pizza) :=
by sorry

end NUMINAMATH_CALUDE_min_pizzas_break_even_l2305_230577


namespace NUMINAMATH_CALUDE_at_least_one_true_l2305_230573

theorem at_least_one_true (p q : Prop) : ¬(¬(p ∨ q)) → (p ∨ q) := by sorry

end NUMINAMATH_CALUDE_at_least_one_true_l2305_230573


namespace NUMINAMATH_CALUDE_election_result_l2305_230548

/-- Represents the total number of votes in an election -/
def total_votes : ℕ := sorry

/-- Represents the percentage of votes received by Candidate A -/
def candidate_a_percentage : ℚ := 45/100

/-- Represents the percentage of votes received by Candidate B -/
def candidate_b_percentage : ℚ := 35/100

/-- Represents the percentage of votes received by Candidate C -/
def candidate_c_percentage : ℚ := 20/100

/-- Represents the difference in votes between Candidate A and Candidate B -/
def vote_difference : ℕ := 1800

theorem election_result :
  (candidate_a_percentage * total_votes - candidate_b_percentage * total_votes = vote_difference) ∧
  (candidate_a_percentage + candidate_b_percentage + candidate_c_percentage = 1) ∧
  (total_votes = 18000) :=
sorry

end NUMINAMATH_CALUDE_election_result_l2305_230548


namespace NUMINAMATH_CALUDE_tangent_half_angle_identity_l2305_230534

theorem tangent_half_angle_identity (α : Real) (h : Real.tan (α / 2) = 2) :
  (1 + Real.cos α) / Real.sin α = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_half_angle_identity_l2305_230534


namespace NUMINAMATH_CALUDE_square_difference_equality_l2305_230579

theorem square_difference_equality : 1004^2 - 996^2 - 1002^2 + 998^2 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2305_230579


namespace NUMINAMATH_CALUDE_kite_area_from_shifted_triangles_l2305_230563

/-- The area of a kite-shaped figure formed by the intersection of two equilateral triangles -/
theorem kite_area_from_shifted_triangles (square_side : ℝ) (shift : ℝ) : 
  square_side = 4 →
  shift = 1 →
  let triangle_side := square_side
  let triangle_height := (Real.sqrt 3 / 2) * triangle_side
  let kite_base := square_side - shift
  let kite_area := kite_base * triangle_height
  kite_area = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_kite_area_from_shifted_triangles_l2305_230563


namespace NUMINAMATH_CALUDE_similar_right_triangles_leg_l2305_230523

/-- Given two similar right triangles, where one has legs 12 and 9, and the other has legs y and 6, prove that y = 8 -/
theorem similar_right_triangles_leg (y : ℝ) : 
  (12 : ℝ) / y = 9 / 6 → y = 8 := by sorry

end NUMINAMATH_CALUDE_similar_right_triangles_leg_l2305_230523


namespace NUMINAMATH_CALUDE_stating_spinner_points_east_l2305_230528

/-- Represents the four cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a clockwise rotation in revolutions --/
def clockwise_rotation : ℚ := 7/2

/-- Represents a counterclockwise rotation in revolutions --/
def counterclockwise_rotation : ℚ := 11/4

/-- Represents the initial direction of the spinner --/
def initial_direction : Direction := Direction.South

/-- 
  Theorem stating that after the given rotations, 
  the spinner will point east
--/
theorem spinner_points_east : 
  ∃ (final_direction : Direction),
    final_direction = Direction.East :=
by sorry

end NUMINAMATH_CALUDE_stating_spinner_points_east_l2305_230528


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l2305_230502

theorem distance_from_origin_to_point : 
  let x : ℝ := 3
  let y : ℝ := -4
  Real.sqrt (x^2 + y^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l2305_230502


namespace NUMINAMATH_CALUDE_dinner_bill_problem_l2305_230566

theorem dinner_bill_problem (P : ℝ) : 
  (P * 0.9 + P * 0.08 + P * 0.15) - (P * 0.85 + P * 0.06 + P * 0.85 * 0.15) = 1 → 
  P = 400 / 37 := by
sorry

end NUMINAMATH_CALUDE_dinner_bill_problem_l2305_230566


namespace NUMINAMATH_CALUDE_sum_of_two_primes_10003_l2305_230500

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem sum_of_two_primes_10003 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10003 :=
sorry

end NUMINAMATH_CALUDE_sum_of_two_primes_10003_l2305_230500


namespace NUMINAMATH_CALUDE_inequality_proof_l2305_230581

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2305_230581


namespace NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l2305_230598

theorem power_of_product_equals_product_of_powers (a : ℝ) : 
  (-2 * a^3)^4 = 16 * a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l2305_230598


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2305_230556

theorem quadratic_root_property (m : ℝ) : 
  m^2 + m - 1 = 0 → 2*m^2 + 2*m + 2025 = 2027 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2305_230556


namespace NUMINAMATH_CALUDE_hemisphere_volume_l2305_230513

theorem hemisphere_volume (r : ℝ) (h : (4/3) * π * r^3 = 150 * π) : 
  (2/3) * π * r^3 = 75 * π := by
sorry

end NUMINAMATH_CALUDE_hemisphere_volume_l2305_230513


namespace NUMINAMATH_CALUDE_circle_intersection_and_reflection_l2305_230574

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + (y - 1)^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 1

-- Define point A
def A : ℝ × ℝ := (4, 0)

-- Define the reflecting line
def reflecting_line (x y : ℝ) : Prop := x - y - 3 = 0

theorem circle_intersection_and_reflection :
  -- Part I: Equation of line l
  (∃ (k : ℝ), (∀ x y : ℝ, y = k * (x - 4) → 
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ C₁ x₁ (k * (x₁ - 4)) ∧ C₁ x₂ (k * (x₂ - 4)) ∧ 
    (x₁ - x₂)^2 + (k * (x₁ - 4) - k * (x₂ - 4))^2 = 12)) ↔
    (k = 0 ∨ 7 * x + 24 * y - 28 = 0)) ∧
  -- Part II: Range of slope of reflected line
  (∀ k : ℝ, (∃ x y : ℝ, C₂ x y ∧ k * x - y - 4 * k - 6 = 0) ↔ 
    (k ≤ -2 * Real.sqrt 30 ∨ k ≥ 2 * Real.sqrt 30)) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_reflection_l2305_230574


namespace NUMINAMATH_CALUDE_complete_square_sum_l2305_230590

theorem complete_square_sum (x : ℝ) : ∃ (a b c : ℤ), 
  (64 * x^2 + 96 * x - 81 = 0) ∧ 
  (a > 0) ∧
  ((a : ℝ) * x + b)^2 = c ∧
  a + b + c = 131 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l2305_230590


namespace NUMINAMATH_CALUDE_expand_product_l2305_230591

theorem expand_product (x : ℝ) : (x + 3) * (x^2 + 2*x + 4) = x^3 + 5*x^2 + 10*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2305_230591


namespace NUMINAMATH_CALUDE_pizza_slices_left_l2305_230561

theorem pizza_slices_left (total_slices : Nat) (john_slices : Nat) (sam_multiplier : Nat) : 
  total_slices = 12 → 
  john_slices = 3 → 
  sam_multiplier = 2 → 
  total_slices - (john_slices + sam_multiplier * john_slices) = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l2305_230561


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2305_230559

theorem simplify_trig_expression (α : Real) 
  (h : -3 * Real.pi < α ∧ α < -(5/2) * Real.pi) : 
  Real.sqrt ((1 + Real.cos (α - 2018 * Real.pi)) / 2) = -Real.cos (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2305_230559


namespace NUMINAMATH_CALUDE_root_in_interval_l2305_230501

noncomputable def f (x : ℝ) : ℝ := 4 - 4*x - Real.exp x

theorem root_in_interval :
  ∃ x ∈ Set.Ioo 0 1, f x = 0 :=
by
  have h1 : Continuous f := sorry
  have h2 : StrictMono (fun x => -f x) := sorry
  have h3 : f 0 > 0 := sorry
  have h4 : f 1 < 0 := sorry
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2305_230501


namespace NUMINAMATH_CALUDE_equality_condition_abs_inequality_l2305_230514

theorem equality_condition_abs_inequality (a b : ℝ) :
  (|a - b| = |a - 1| + |b - 1|) ↔ ((a - 1) * (b - 1) ≤ 0) := by sorry

end NUMINAMATH_CALUDE_equality_condition_abs_inequality_l2305_230514


namespace NUMINAMATH_CALUDE_easiest_to_pick_black_l2305_230584

structure Box where
  label : Char
  black_balls : ℕ
  white_balls : ℕ

def probability_black (b : Box) : ℚ :=
  b.black_balls / (b.black_balls + b.white_balls)

def boxes : List Box := [
  ⟨'A', 12, 4⟩,
  ⟨'B', 10, 10⟩,
  ⟨'C', 4, 2⟩,
  ⟨'D', 10, 5⟩
]

theorem easiest_to_pick_black (boxes : List Box) :
  ∃ b ∈ boxes, ∀ b' ∈ boxes, probability_black b ≥ probability_black b' :=
sorry

end NUMINAMATH_CALUDE_easiest_to_pick_black_l2305_230584


namespace NUMINAMATH_CALUDE_integer_solutions_equation_l2305_230555

theorem integer_solutions_equation :
  ∀ x y : ℤ, 2*x^2 + 8*y^2 = 17*x*y - 423 ↔ (x = 11 ∧ y = 19) ∨ (x = -11 ∧ y = -19) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_equation_l2305_230555


namespace NUMINAMATH_CALUDE_time_to_school_gate_l2305_230586

/-- Proves that the time to arrive at the school gate is 15 minutes -/
theorem time_to_school_gate 
  (total_time : ℕ) 
  (gate_to_building : ℕ) 
  (building_to_room : ℕ) 
  (h1 : total_time = 30) 
  (h2 : gate_to_building = 6) 
  (h3 : building_to_room = 9) : 
  total_time - gate_to_building - building_to_room = 15 := by
sorry

end NUMINAMATH_CALUDE_time_to_school_gate_l2305_230586


namespace NUMINAMATH_CALUDE_systematic_sampling_l2305_230592

theorem systematic_sampling (total_bags : Nat) (num_groups : Nat) (fourth_group_sample : Nat) (first_group_sample : Nat) :
  total_bags = 50 →
  num_groups = 5 →
  fourth_group_sample = 36 →
  first_group_sample = 6 →
  (total_bags / num_groups) * 3 + first_group_sample = fourth_group_sample :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l2305_230592


namespace NUMINAMATH_CALUDE_expression_evaluation_l2305_230539

theorem expression_evaluation (a b c : ℝ) (ha : a = 14) (hb : b = 19) (hc : c = 13) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2305_230539


namespace NUMINAMATH_CALUDE_rex_cards_left_is_150_l2305_230588

/-- The number of Pokemon cards Rex has left after dividing his cards among himself and his siblings -/
def rexCardsLeft (nicolesCards : ℕ) : ℕ :=
  let cindysCards := 2 * nicolesCards
  let combinedTotal := nicolesCards + cindysCards
  let rexCards := combinedTotal / 2
  rexCards / 4

/-- Theorem stating that Rex has 150 cards left given the initial conditions -/
theorem rex_cards_left_is_150 : rexCardsLeft 400 = 150 := by
  sorry

end NUMINAMATH_CALUDE_rex_cards_left_is_150_l2305_230588


namespace NUMINAMATH_CALUDE_area_equality_l2305_230554

/-- Represents a regular hexagon with side length 1 -/
def RegularHexagon : Set (ℝ × ℝ) := sorry

/-- Represents an equilateral triangle with side length 1 -/
def EquilateralTriangle : Set (ℝ × ℝ) := sorry

/-- Represents the region R, which is the union of the hexagon and 18 triangles -/
def R : Set (ℝ × ℝ) := sorry

/-- Represents the smallest convex polygon S that contains R -/
def S : Set (ℝ × ℝ) := sorry

/-- The area of a set in the plane -/
def area : Set (ℝ × ℝ) → ℝ := sorry

theorem area_equality : area S = area R := by sorry

end NUMINAMATH_CALUDE_area_equality_l2305_230554


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2305_230558

theorem unique_prime_solution :
  ∀ p q r : ℕ,
    Prime p ∧ Prime q ∧ Prime r →
    p + q^2 = r^4 →
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2305_230558


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2305_230568

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, 2 * Real.sqrt 2 ≤ (x^2 + 2) / x ∧ (x^2 + 2) / x ≤ 3 → Real.sqrt 2 / 2 ≤ x ∧ x ≤ 2 * Real.sqrt 2) ∧
  (∃ x : ℝ, Real.sqrt 2 / 2 ≤ x ∧ x ≤ 2 * Real.sqrt 2 ∧ (2 * Real.sqrt 2 > (x^2 + 2) / x ∨ (x^2 + 2) / x > 3)) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2305_230568


namespace NUMINAMATH_CALUDE_employment_percentage_l2305_230576

theorem employment_percentage (total_population : ℝ) (employed_population : ℝ) 
  (h1 : employed_population > 0) 
  (h2 : employed_population ≤ total_population)
  (h3 : employed_population * 0.7 + employed_population * 0.3 = employed_population)
  (h4 : employed_population * 0.3 = total_population * 0.21) : 
  employed_population / total_population = 0.7 := by
sorry

end NUMINAMATH_CALUDE_employment_percentage_l2305_230576


namespace NUMINAMATH_CALUDE_zero_shaded_area_l2305_230589

/-- Represents a square tile with a pattern of triangles -/
structure Tile where
  sideLength : ℝ
  triangleArea : ℝ

/-- Represents a rectangular floor tiled with square tiles -/
structure Floor where
  length : ℝ
  width : ℝ
  tile : Tile

/-- Calculates the total shaded area of the floor -/
def totalShadedArea (floor : Floor) : ℝ :=
  let totalTiles := floor.length * floor.width
  let tileArea := floor.tile.sideLength ^ 2
  let shadedAreaPerTile := tileArea - 4 * floor.tile.triangleArea
  totalTiles * shadedAreaPerTile

/-- Theorem stating that the total shaded area of the specific floor is 0 -/
theorem zero_shaded_area :
  let tile : Tile := {
    sideLength := 1,
    triangleArea := 1/4
  }
  let floor : Floor := {
    length := 12,
    width := 9,
    tile := tile
  }
  totalShadedArea floor = 0 := by sorry

end NUMINAMATH_CALUDE_zero_shaded_area_l2305_230589


namespace NUMINAMATH_CALUDE_quadratic_roots_inequality_l2305_230540

theorem quadratic_roots_inequality (t : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - t*x₁ + t = 0) → 
  (x₂^2 - t*x₂ + t = 0) → 
  (x₁^2 + x₂^2 ≥ 2*(x₁ + x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_inequality_l2305_230540


namespace NUMINAMATH_CALUDE_cost_39_roses_l2305_230504

/-- Represents the cost of a bouquet of roses -/
def bouquet_cost (roses : ℕ) : ℚ :=
  sorry

/-- The price of a bouquet is directly proportional to the number of roses -/
axiom price_proportional (r₁ r₂ : ℕ) : 
  bouquet_cost r₁ / bouquet_cost r₂ = r₁ / r₂

/-- A bouquet of 12 roses costs $20 -/
axiom dozen_cost : bouquet_cost 12 = 20

theorem cost_39_roses : bouquet_cost 39 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cost_39_roses_l2305_230504


namespace NUMINAMATH_CALUDE_paths_in_7x8_grid_l2305_230520

/-- The number of paths in a grid moving only up or right -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- Theorem: The number of paths in a 7x8 grid moving only up or right is 6435 -/
theorem paths_in_7x8_grid : grid_paths 7 8 = 6435 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_7x8_grid_l2305_230520


namespace NUMINAMATH_CALUDE_jenny_hotel_cost_l2305_230572

/-- The total cost of a hotel stay for a group of people -/
def total_cost (cost_per_night_per_person : ℕ) (num_people : ℕ) (num_nights : ℕ) : ℕ :=
  cost_per_night_per_person * num_people * num_nights

/-- Theorem stating the total cost for Jenny and her friends' hotel stay -/
theorem jenny_hotel_cost :
  total_cost 40 3 3 = 360 := by
  sorry

end NUMINAMATH_CALUDE_jenny_hotel_cost_l2305_230572


namespace NUMINAMATH_CALUDE_function_is_constant_one_l2305_230549

/-- A function satisfying the given conditions is constant and equal to 1 -/
theorem function_is_constant_one (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = 2 - f x) 
  (h2 : ∀ x, f (x + 3) ≥ f x) : 
  ∀ x, f x = 1 := by sorry

end NUMINAMATH_CALUDE_function_is_constant_one_l2305_230549


namespace NUMINAMATH_CALUDE_revenue_comparison_l2305_230517

/-- Given a projected revenue increase of 40% and an actual revenue decrease of 30% from the previous year,
    the actual revenue is 50% of the projected revenue. -/
theorem revenue_comparison (previous_revenue : ℝ) (projected_increase : ℝ) (actual_decrease : ℝ)
    (h1 : projected_increase = 0.4)
    (h2 : actual_decrease = 0.3) :
    (previous_revenue * (1 - actual_decrease)) / (previous_revenue * (1 + projected_increase)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_revenue_comparison_l2305_230517


namespace NUMINAMATH_CALUDE_kombucha_half_fill_time_l2305_230530

/-- Represents the area of kombucha in the jar as a fraction of the full jar -/
def kombucha_area (days : ℕ) : ℚ :=
  1 / 2^(19 - days)

theorem kombucha_half_fill_time : 
  (∀ d : ℕ, d < 19 → kombucha_area (d + 1) = 2 * kombucha_area d) →
  kombucha_area 19 = 1 →
  kombucha_area 18 = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_kombucha_half_fill_time_l2305_230530
