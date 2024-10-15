import Mathlib

namespace NUMINAMATH_CALUDE_robins_hair_length_l2992_299240

theorem robins_hair_length (initial_length cut_length growth_length final_length : ℕ) :
  cut_length = 11 →
  growth_length = 12 →
  final_length = 17 →
  final_length = initial_length - cut_length + growth_length →
  initial_length = 16 :=
by sorry

end NUMINAMATH_CALUDE_robins_hair_length_l2992_299240


namespace NUMINAMATH_CALUDE_travelers_checks_denomination_l2992_299295

theorem travelers_checks_denomination (total_checks : ℕ) (total_worth : ℝ) (spendable_checks : ℕ) (remaining_average : ℝ) :
  total_checks = 30 →
  total_worth = 1800 →
  spendable_checks = 18 →
  remaining_average = 75 →
  (total_worth - (total_checks - spendable_checks : ℝ) * remaining_average) / spendable_checks = 50 :=
by sorry

end NUMINAMATH_CALUDE_travelers_checks_denomination_l2992_299295


namespace NUMINAMATH_CALUDE_calculate_ambulance_ride_cost_l2992_299215

/-- Given a hospital bill with various components, calculate the cost of the ambulance ride. -/
theorem calculate_ambulance_ride_cost (total_bill : ℝ) (medication_percent : ℝ) 
  (imaging_percent : ℝ) (surgical_percent : ℝ) (overnight_percent : ℝ) (doctor_percent : ℝ) 
  (food_fee : ℝ) (consultation_fee : ℝ) (therapy_fee : ℝ) 
  (h1 : total_bill = 18000)
  (h2 : medication_percent = 35)
  (h3 : imaging_percent = 15)
  (h4 : surgical_percent = 25)
  (h5 : overnight_percent = 10)
  (h6 : doctor_percent = 5)
  (h7 : food_fee = 300)
  (h8 : consultation_fee = 450)
  (h9 : therapy_fee = 600) :
  total_bill - (medication_percent / 100 * total_bill + 
                imaging_percent / 100 * total_bill + 
                surgical_percent / 100 * total_bill + 
                overnight_percent / 100 * total_bill + 
                doctor_percent / 100 * total_bill + 
                food_fee + consultation_fee + therapy_fee) = 450 := by
  sorry


end NUMINAMATH_CALUDE_calculate_ambulance_ride_cost_l2992_299215


namespace NUMINAMATH_CALUDE_local_language_letters_l2992_299298

theorem local_language_letters (n : ℕ) : 
  (n + n^2) - ((n - 1) + (n - 1)^2) = 139 → n = 69 := by
  sorry

end NUMINAMATH_CALUDE_local_language_letters_l2992_299298


namespace NUMINAMATH_CALUDE_younger_person_age_l2992_299206

/-- Given two persons whose ages differ by 20 years, and 5 years ago the elder one was 5 times as old as the younger one, the present age of the younger person is 10 years. -/
theorem younger_person_age (y e : ℕ) : 
  e = y + 20 →                  -- The ages differ by 20 years
  e - 5 = 5 * (y - 5) →         -- 5 years ago, elder was 5 times younger
  y = 10                        -- The younger person's age is 10
  := by sorry

end NUMINAMATH_CALUDE_younger_person_age_l2992_299206


namespace NUMINAMATH_CALUDE_marble_difference_l2992_299200

/-- The number of marbles each person has -/
structure Marbles where
  laurie : ℕ
  kurt : ℕ
  dennis : ℕ

/-- Given conditions about the marbles -/
def marble_conditions (m : Marbles) : Prop :=
  m.laurie = 37 ∧ m.laurie = m.kurt + 12 ∧ m.dennis = 70

/-- Theorem stating the difference between Dennis's and Kurt's marbles -/
theorem marble_difference (m : Marbles) (h : marble_conditions m) :
  m.dennis - m.kurt = 45 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l2992_299200


namespace NUMINAMATH_CALUDE_class_outing_minimum_fee_l2992_299237

/-- Calculates the minimum rental fee for a class outing --/
def minimum_rental_fee (total_students : ℕ) (small_boat_capacity : ℕ) (small_boat_cost : ℕ) 
  (large_boat_capacity : ℕ) (large_boat_cost : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum rental fee for the given conditions --/
theorem class_outing_minimum_fee : 
  minimum_rental_fee 48 3 16 5 24 = 232 := by
  sorry

end NUMINAMATH_CALUDE_class_outing_minimum_fee_l2992_299237


namespace NUMINAMATH_CALUDE_remainder_71_cubed_73_fifth_mod_8_l2992_299229

theorem remainder_71_cubed_73_fifth_mod_8 : (71^3 * 73^5) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_71_cubed_73_fifth_mod_8_l2992_299229


namespace NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_for_maximum_l2992_299202

theorem monotonic_sufficient_not_necessary_for_maximum 
  (f : ℝ → ℝ) (hf : ContinuousOn f (Set.Icc 0 1)) :
  (MonotoneOn f (Set.Icc 0 1) → ∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f y ≤ f x) ∧
  ¬(∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f y ≤ f x → MonotoneOn f (Set.Icc 0 1)) :=
sorry

end NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_for_maximum_l2992_299202


namespace NUMINAMATH_CALUDE_investment_return_is_25_percent_l2992_299255

/-- Calculates the percentage return on investment for a given dividend rate, face value, and purchase price of shares. -/
def percentageReturn (dividendRate : ℚ) (faceValue : ℚ) (purchasePrice : ℚ) : ℚ :=
  (dividendRate * faceValue / purchasePrice) * 100

/-- Theorem stating that for the given conditions, the percentage return on investment is 25%. -/
theorem investment_return_is_25_percent :
  let dividendRate : ℚ := 125 / 1000
  let faceValue : ℚ := 40
  let purchasePrice : ℚ := 20
  percentageReturn dividendRate faceValue purchasePrice = 25 := by
  sorry

end NUMINAMATH_CALUDE_investment_return_is_25_percent_l2992_299255


namespace NUMINAMATH_CALUDE_chord_length_in_circle_l2992_299270

theorem chord_length_in_circle (r d : ℝ) (hr : r = 5) (hd : d = 3) :
  let half_chord := Real.sqrt (r^2 - d^2)
  2 * half_chord = 8 := by sorry

end NUMINAMATH_CALUDE_chord_length_in_circle_l2992_299270


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l2992_299286

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-3, 4)

theorem reflection_across_y_axis :
  reflect_y P = (3, 4) := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l2992_299286


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l2992_299271

/-- The number of ways to partition n indistinguishable balls into k indistinguishable boxes,
    with at least one ball in each box. -/
def partition_count (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are exactly 2 ways to partition 6 indistinguishable balls into 4 indistinguishable boxes,
    with at least one ball in each box. -/
theorem six_balls_four_boxes : partition_count 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l2992_299271


namespace NUMINAMATH_CALUDE_quadratic_and_optimization_l2992_299241

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ (x < 1 ∨ x > b)

-- Define the constraint equation
def constraint (x y : ℝ) : Prop :=
  (1 / (x + 1)) + (2 / (y + 1)) = 1

-- Define the objective function
def objective (x y : ℝ) : ℝ := 2 * x + y + 3

-- State the theorem
theorem quadratic_and_optimization :
  ∃ a b : ℝ,
    solution_set a b ∧
    (a = 1 ∧ b = 2) ∧
    (∀ x y : ℝ, x > 0 → y > 0 → constraint x y →
      objective x y ≥ 8 ∧
      ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ constraint x₀ y₀ ∧ objective x₀ y₀ = 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_and_optimization_l2992_299241


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l2992_299227

theorem five_digit_multiple_of_nine : ∃ (n : ℕ), n = 45675 ∧ n % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l2992_299227


namespace NUMINAMATH_CALUDE_profit_equals_700_at_5_profit_equals_original_at_3_total_profit_after_two_years_l2992_299222

-- Define the profit function
def profit_function (x : ℝ) : ℝ := 10 * x^2 + 90 * x

-- Define the original monthly profit
def original_monthly_profit : ℝ := 1.2

-- Theorem 1
theorem profit_equals_700_at_5 :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 12 ∧ profit_function x = 700 ∧ x = 5 :=
sorry

-- Theorem 2
theorem profit_equals_original_at_3 :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 12 ∧ profit_function x = 120 * x ∧ x = 3 :=
sorry

-- Theorem 3
theorem total_profit_after_two_years :
  12 * (10 * 12 + 90) + 12 * 320 = 6360 :=
sorry

end NUMINAMATH_CALUDE_profit_equals_700_at_5_profit_equals_original_at_3_total_profit_after_two_years_l2992_299222


namespace NUMINAMATH_CALUDE_algebraic_identity_l2992_299272

theorem algebraic_identity (a b c d : ℝ) :
  (a^2 + b^2) * (a*b + c*d) - a*b * (a^2 + b^2 - c^2 - d^2) = (a*c + b*d) * (a*d + b*c) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identity_l2992_299272


namespace NUMINAMATH_CALUDE_excursion_dates_correct_l2992_299245

/-- Represents the four excursion locations --/
inductive Location
| Carpathians
| Kyiv
| Forest
| Museum

/-- Represents a calendar month --/
structure Month where
  number : Nat
  days : Nat
  first_day_sunday : Bool

/-- Represents an excursion --/
structure Excursion where
  location : Location
  month : Month
  day : Nat

/-- Checks if a given day is the first Sunday after the first Saturday --/
def is_first_sunday_after_saturday (m : Month) (d : Nat) : Prop :=
  d = 8 ∧ m.first_day_sunday

/-- The theorem to prove --/
theorem excursion_dates_correct (feb mar : Month) 
  (e1 e2 e3 e4 : Excursion) : 
  feb.number = 2 → 
  mar.number = 3 → 
  feb.days = 28 → 
  mar.days = 31 → 
  feb.first_day_sunday = true → 
  mar.first_day_sunday = true → 
  e1.location = Location.Carpathians → 
  e2.location = Location.Kyiv → 
  e3.location = Location.Forest → 
  e4.location = Location.Museum → 
  e1.month = feb ∧ e1.day = 1 ∧
  e2.month = feb ∧ is_first_sunday_after_saturday feb e2.day ∧
  e3.month = mar ∧ e3.day = 1 ∧
  e4.month = mar ∧ is_first_sunday_after_saturday mar e4.day :=
sorry

end NUMINAMATH_CALUDE_excursion_dates_correct_l2992_299245


namespace NUMINAMATH_CALUDE_deposit_percentage_l2992_299205

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) :
  deposit = 150 →
  remaining = 1350 →
  (deposit / (deposit + remaining)) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_deposit_percentage_l2992_299205


namespace NUMINAMATH_CALUDE_leadership_selection_count_l2992_299254

def tribe_size : ℕ := 12
def num_supporting_chiefs : ℕ := 2
def num_inferior_officers_per_chief : ℕ := 2

def leadership_selection_ways : ℕ :=
  tribe_size *
  (Nat.choose (tribe_size - 1) num_supporting_chiefs) *
  (Nat.choose (tribe_size - 1 - num_supporting_chiefs) (num_supporting_chiefs * num_inferior_officers_per_chief) /
   Nat.factorial num_supporting_chiefs)

theorem leadership_selection_count :
  leadership_selection_ways = 248040 :=
by sorry

end NUMINAMATH_CALUDE_leadership_selection_count_l2992_299254


namespace NUMINAMATH_CALUDE_business_investment_problem_l2992_299294

theorem business_investment_problem (y_investment : ℕ) (total_profit : ℕ) (x_profit_share : ℕ) (x_investment : ℕ) :
  y_investment = 15000 →
  total_profit = 1600 →
  x_profit_share = 400 →
  x_profit_share * y_investment = (total_profit - x_profit_share) * x_investment →
  x_investment = 5000 := by
sorry

end NUMINAMATH_CALUDE_business_investment_problem_l2992_299294


namespace NUMINAMATH_CALUDE_expand_product_l2992_299231

theorem expand_product (x : ℝ) : (3*x + 4) * (2*x + 7) = 6*x^2 + 29*x + 28 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2992_299231


namespace NUMINAMATH_CALUDE_min_value_abc_l2992_299228

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 9*a + 4*b = a*b*c) : 
  a + b + c ≥ 10 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    9*a₀ + 4*b₀ = a₀*b₀*c₀ ∧ a₀ + b₀ + c₀ = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abc_l2992_299228


namespace NUMINAMATH_CALUDE_negative_integer_solutions_of_inequality_l2992_299256

theorem negative_integer_solutions_of_inequality :
  {x : ℤ | x < 0 ∧ 3 * x + 1 ≥ -5} = {-2, -1} := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_solutions_of_inequality_l2992_299256


namespace NUMINAMATH_CALUDE_cubic_sum_equals_zero_l2992_299282

theorem cubic_sum_equals_zero (a b c : ℝ) :
  a^2 + b^2 + c^2 - 2*(a + b + c) + 3 = 0 →
  a^3 + b^3 + c^3 - 3*a*b*c = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_zero_l2992_299282


namespace NUMINAMATH_CALUDE_average_weight_problem_l2992_299224

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions and the average weight of b and c is 43. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 ∧ 
  (a + b) / 2 = 40 ∧ 
  b = 31 → 
  (b + c) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l2992_299224


namespace NUMINAMATH_CALUDE_sum_of_19th_powers_zero_l2992_299276

theorem sum_of_19th_powers_zero (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_cubes_zero : a^3 + b^3 + c^3 = 0) : 
  a^19 + b^19 + c^19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_19th_powers_zero_l2992_299276


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2992_299217

theorem complex_equation_solution (z : ℂ) (h : z * (1 + Complex.I) = 2) : z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2992_299217


namespace NUMINAMATH_CALUDE_solution_set_theorem_range_of_a_theorem_l2992_299216

def f (x : ℝ) : ℝ := |x + 2| - 2 * |x - 1|

theorem solution_set_theorem (x : ℝ) :
  f x ≥ -2 ↔ -2/3 ≤ x ∧ x ≤ 6 :=
sorry

theorem range_of_a_theorem :
  (∀ x : ℝ, f x ≤ x - a) ↔ a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_range_of_a_theorem_l2992_299216


namespace NUMINAMATH_CALUDE_second_oil_price_l2992_299269

/-- Given two types of oil mixed together, calculate the price of the second oil -/
theorem second_oil_price (volume1 volume2 : ℝ) (price1 mixed_price : ℝ) :
  volume1 = 10 →
  volume2 = 5 →
  price1 = 50 →
  mixed_price = 55.33 →
  (volume1 * price1 + volume2 * (volume1 * price1 + volume2 * mixed_price - volume1 * price1) / volume2) / (volume1 + volume2) = mixed_price →
  (volume1 * price1 + volume2 * mixed_price - volume1 * price1) / volume2 = 65.99 := by
  sorry

#eval (10 * 50 + 5 * 55.33 * 3 - 10 * 50) / 5

end NUMINAMATH_CALUDE_second_oil_price_l2992_299269


namespace NUMINAMATH_CALUDE_tim_bought_three_dozens_l2992_299226

/-- The number of dozens of eggs Tim bought -/
def dozens_bought (egg_price : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid / (12 * egg_price)

/-- Theorem stating that Tim bought 3 dozens of eggs -/
theorem tim_bought_three_dozens :
  dozens_bought (1/2) 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tim_bought_three_dozens_l2992_299226


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l2992_299278

theorem a_equals_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → (a - 1) * (a - 2) = 0) ∧
  (∃ a : ℝ, a ≠ 1 ∧ (a - 1) * (a - 2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l2992_299278


namespace NUMINAMATH_CALUDE_last_four_digits_5_power_2017_l2992_299261

/-- The last four digits of 5^n, represented as an integer between 0 and 9999 -/
def lastFourDigits (n : ℕ) : ℕ := 5^n % 10000

theorem last_four_digits_5_power_2017 :
  lastFourDigits 5 = 3125 ∧
  lastFourDigits 6 = 5625 ∧
  lastFourDigits 7 = 8125 →
  lastFourDigits 2017 = 3125 := by
sorry

end NUMINAMATH_CALUDE_last_four_digits_5_power_2017_l2992_299261


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l2992_299244

theorem mayoral_election_votes 
  (votes_Z : ℕ) 
  (h1 : votes_Z = 25000)
  (votes_Y : ℕ) 
  (h2 : votes_Y = votes_Z - (2 / 5 : ℚ) * votes_Z)
  (votes_X : ℕ) 
  (h3 : votes_X = votes_Y + (1 / 2 : ℚ) * votes_Y) :
  votes_X = 22500 := by
sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l2992_299244


namespace NUMINAMATH_CALUDE_bridget_skittles_l2992_299263

/-- If Bridget has 4 Skittles, Henry has 4 Skittles, and Henry gives all of his Skittles to Bridget,
    then Bridget will have 8 Skittles in total. -/
theorem bridget_skittles (bridget_initial : ℕ) (henry_initial : ℕ)
    (h1 : bridget_initial = 4)
    (h2 : henry_initial = 4) :
    bridget_initial + henry_initial = 8 := by
  sorry

end NUMINAMATH_CALUDE_bridget_skittles_l2992_299263


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2992_299288

theorem expand_and_simplify (n b : ℝ) : (n + 2*b)^2 - 4*b^2 = n^2 + 4*n*b := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2992_299288


namespace NUMINAMATH_CALUDE_range_of_a_l2992_299280

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Iic 6, StrictMonoOn (f a) (Set.Iic x)) →
  a ∈ Set.Iic (-5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2992_299280


namespace NUMINAMATH_CALUDE_sqrt_neg_two_squared_l2992_299211

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_two_squared_l2992_299211


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_103_l2992_299219

theorem largest_n_divisible_by_103 : 
  ∀ n : ℕ, n < 103 ∧ 103 ∣ (n^3 - 1) → n ≤ 52 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_103_l2992_299219


namespace NUMINAMATH_CALUDE_power_division_rule_l2992_299287

theorem power_division_rule (a : ℝ) (h : a ≠ 0) : a^5 / a^3 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l2992_299287


namespace NUMINAMATH_CALUDE_sqrt_7x_equals_14_l2992_299207

theorem sqrt_7x_equals_14 (x : ℝ) (h : x / 2 - 5 = 9) : Real.sqrt (7 * x) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_7x_equals_14_l2992_299207


namespace NUMINAMATH_CALUDE_total_chestnuts_weight_l2992_299277

/-- The weight of chestnuts Eun-soo picked in kilograms -/
def eun_soo_kg : ℝ := 2

/-- The weight of chestnuts Eun-soo picked in grams (in addition to the kilograms) -/
def eun_soo_g : ℝ := 600

/-- The weight of chestnuts Min-gi picked in grams -/
def min_gi_g : ℝ := 3700

/-- The conversion factor from kilograms to grams -/
def kg_to_g : ℝ := 1000

theorem total_chestnuts_weight : 
  eun_soo_kg * kg_to_g + eun_soo_g + min_gi_g = 6300 := by
  sorry

end NUMINAMATH_CALUDE_total_chestnuts_weight_l2992_299277


namespace NUMINAMATH_CALUDE_susan_tuesday_candies_l2992_299239

/-- Represents the number of candies Susan bought on each day -/
structure CandyPurchases where
  tuesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Represents Susan's candy consumption and remaining candies -/
structure CandyConsumption where
  eaten : ℕ
  remaining : ℕ

/-- Calculates the total number of candies Susan had -/
def totalCandies (purchases : CandyPurchases) (consumption : CandyConsumption) : ℕ :=
  purchases.tuesday + purchases.thursday + purchases.friday

/-- Theorem: Susan bought 3 candies on Tuesday -/
theorem susan_tuesday_candies (purchases : CandyPurchases) (consumption : CandyConsumption) :
  purchases.thursday = 5 →
  purchases.friday = 2 →
  consumption.eaten = 6 →
  consumption.remaining = 4 →
  totalCandies purchases consumption = consumption.eaten + consumption.remaining →
  purchases.tuesday = 3 := by
  sorry

end NUMINAMATH_CALUDE_susan_tuesday_candies_l2992_299239


namespace NUMINAMATH_CALUDE_no_curious_numbers_l2992_299292

def CuriousNumber (f : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, f x = f (a - x)

theorem no_curious_numbers (f : ℤ → ℤ) 
  (h1 : ∀ x : ℤ, f x ≠ x) :
  ¬ (∃ a ∈ ({60, 62, 823} : Set ℤ), CuriousNumber f a) :=
sorry

end NUMINAMATH_CALUDE_no_curious_numbers_l2992_299292


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2992_299230

theorem inequality_solution_set :
  ∀ x : ℝ, (1 - x) * (2 + x) < 0 ↔ x < -2 ∨ x > 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2992_299230


namespace NUMINAMATH_CALUDE_value_added_to_number_l2992_299220

theorem value_added_to_number : ∃ v : ℝ, 3 * (9 + v) = 9 + 24 ∧ v = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_number_l2992_299220


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_7_pow_6_l2992_299257

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 3^(7^6) is 3 -/
theorem units_digit_of_3_pow_7_pow_6 : unitsDigit (3^(7^6)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_7_pow_6_l2992_299257


namespace NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l2992_299221

/-- Calculates the alcohol percentage in a mixture after adding water -/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 9)
  (h2 : initial_alcohol_percentage = 57)
  (h3 : water_added = 3) :
  let alcohol_volume := initial_volume * (initial_alcohol_percentage / 100)
  let total_volume := initial_volume + water_added
  let new_alcohol_percentage := (alcohol_volume / total_volume) * 100
  new_alcohol_percentage = 42.75 := by
sorry

end NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l2992_299221


namespace NUMINAMATH_CALUDE_angle_sum_inequality_l2992_299232

theorem angle_sum_inequality (α β γ x y z : ℝ) 
  (h_angles : α + β + γ = Real.pi)
  (h_sum : x + y + z = 0) :
  y * z * Real.sin α ^ 2 + z * x * Real.sin β ^ 2 + x * y * Real.sin γ ^ 2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_inequality_l2992_299232


namespace NUMINAMATH_CALUDE_grid_segment_sums_equal_area_l2992_299260

/-- Represents a convex polygon with vertices at integer grid points --/
structure ConvexGridPolygon where
  vertices : List (Int × Int)
  is_convex : Bool
  no_sides_on_gridlines : Bool

/-- Calculates the sum of lengths of horizontal grid segments within the polygon --/
def sum_horizontal_segments (polygon : ConvexGridPolygon) : ℝ :=
  sorry

/-- Calculates the sum of lengths of vertical grid segments within the polygon --/
def sum_vertical_segments (polygon : ConvexGridPolygon) : ℝ :=
  sorry

/-- Calculates the area of the polygon --/
def polygon_area (polygon : ConvexGridPolygon) : ℝ :=
  sorry

/-- Theorem stating that for a convex polygon with vertices at integer grid points
    and no sides along grid lines, the sum of horizontal grid segment lengths equals
    the sum of vertical grid segment lengths, and both equal the polygon's area --/
theorem grid_segment_sums_equal_area (polygon : ConvexGridPolygon) :
  sum_horizontal_segments polygon = sum_vertical_segments polygon ∧
  sum_horizontal_segments polygon = polygon_area polygon :=
  sorry

end NUMINAMATH_CALUDE_grid_segment_sums_equal_area_l2992_299260


namespace NUMINAMATH_CALUDE_distance_AB_l2992_299284

-- Define the points and distances
structure Points where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

-- Define the speeds
structure Speeds where
  vA : ℝ
  vB : ℝ

-- Define the problem conditions
structure Conditions where
  points : Points
  speeds : Speeds
  CD_distance : ℝ
  B_remaining_distance : ℝ
  speed_ratio : ℝ
  speed_reduction : ℝ

-- Theorem statement
theorem distance_AB (c : Conditions) : 
  c.CD_distance = 900 ∧ 
  c.B_remaining_distance = 720 ∧ 
  c.speed_ratio = 5/4 ∧ 
  c.speed_reduction = 4/5 →
  c.points.B - c.points.A = 5265 := by
  sorry


end NUMINAMATH_CALUDE_distance_AB_l2992_299284


namespace NUMINAMATH_CALUDE_yah_to_bah_conversion_l2992_299201

-- Define the exchange rates
def bah_to_rah_rate : ℚ := 30 / 18
def rah_to_yah_rate : ℚ := 25 / 10

-- Define the conversion function
def convert_yah_to_bah (yahs : ℚ) : ℚ :=
  yahs / (rah_to_yah_rate * bah_to_rah_rate)

-- Theorem statement
theorem yah_to_bah_conversion :
  convert_yah_to_bah 1250 = 300 := by
  sorry

end NUMINAMATH_CALUDE_yah_to_bah_conversion_l2992_299201


namespace NUMINAMATH_CALUDE_collinear_points_m_equals_four_l2992_299208

-- Define the points
def A : ℝ × ℝ := (-2, 12)
def B : ℝ × ℝ := (1, 3)
def C : ℝ → ℝ × ℝ := λ m ↦ (m, -6)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - p.1) = (r.2 - p.2) * (q.1 - p.1)

-- Theorem statement
theorem collinear_points_m_equals_four :
  collinear A B (C 4) := by sorry

end NUMINAMATH_CALUDE_collinear_points_m_equals_four_l2992_299208


namespace NUMINAMATH_CALUDE_planes_parallel_if_infinitely_many_parallel_lines_l2992_299293

-- Define the concept of a plane in 3D space
variable (α β : Set (ℝ × ℝ × ℝ))

-- Define what it means for a line to be parallel to a plane
def LineParallelToPlane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define what it means for two planes to be parallel
def PlanesParallel (p1 p2 : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the concept of infinitely many lines in a plane
def InfinitelyManyParallelLines (p1 p2 : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ (S : Set (Set (ℝ × ℝ × ℝ))), Infinite S ∧ (∀ l ∈ S, l ⊆ p1 ∧ LineParallelToPlane l p2)

-- State the theorem
theorem planes_parallel_if_infinitely_many_parallel_lines (α β : Set (ℝ × ℝ × ℝ)) :
  InfinitelyManyParallelLines α β → PlanesParallel α β := by sorry

end NUMINAMATH_CALUDE_planes_parallel_if_infinitely_many_parallel_lines_l2992_299293


namespace NUMINAMATH_CALUDE_a_equals_plus_minus_two_l2992_299258

-- Define the sets A and B
def A : Set ℝ := {0, 2}
def B (a : ℝ) : Set ℝ := {1, a^2}

-- Define the theorem
theorem a_equals_plus_minus_two (a : ℝ) : 
  A ∪ B a = {0, 1, 2, 4} → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_plus_minus_two_l2992_299258


namespace NUMINAMATH_CALUDE_triangle_dot_product_l2992_299243

/-- Given a triangle ABC with side lengths a, b, c, prove that if a = 2, b - c = 1,
    and the area of the triangle is √3, then the dot product of vectors AB and AC is 13/4 -/
theorem triangle_dot_product (a b c : ℝ) (A : ℝ) :
  a = 2 →
  b - c = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  b * c * Real.cos A = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_dot_product_l2992_299243


namespace NUMINAMATH_CALUDE_factor_sum_l2992_299251

theorem factor_sum (R S : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + R*X^2 + S) → 
  R + S = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l2992_299251


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2992_299297

theorem min_value_of_expression (x y : ℝ) : 
  x^2 + 4*x*y + 5*y^2 - 8*x - 4*y + x^3 ≥ -11.9 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 + 4*x₀*y₀ + 5*y₀^2 - 8*x₀ - 4*y₀ + x₀^3 = -11.9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2992_299297


namespace NUMINAMATH_CALUDE_little_john_theorem_l2992_299264

def little_john_problem (initial_amount : ℚ) (given_to_each_friend : ℚ) (num_friends : ℕ) (amount_left : ℚ) : ℚ :=
  initial_amount - (given_to_each_friend * num_friends) - amount_left

theorem little_john_theorem (initial_amount : ℚ) (given_to_each_friend : ℚ) (num_friends : ℕ) (amount_left : ℚ) :
  little_john_problem initial_amount given_to_each_friend num_friends amount_left =
  initial_amount - (given_to_each_friend * num_friends) - amount_left :=
by
  sorry

#eval little_john_problem 10.50 2.20 2 3.85

end NUMINAMATH_CALUDE_little_john_theorem_l2992_299264


namespace NUMINAMATH_CALUDE_boys_usual_time_to_school_l2992_299283

/-- Proves that if a boy walks at 7/6 of his usual rate and arrives at school 4 minutes early, 
    his usual time to reach school is 28 minutes. -/
theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) 
  (h1 : usual_rate > 0) 
  (h2 : usual_time > 0) 
  (h3 : usual_rate * usual_time = (7/6 * usual_rate) * (usual_time - 4)) : 
  usual_time = 28 := by
sorry

end NUMINAMATH_CALUDE_boys_usual_time_to_school_l2992_299283


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l2992_299291

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -3 ∨ x > 1}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

-- Theorem for (¬ᵤA) ∩ (¬ᵤB)
theorem intersection_complement_A_B : (Aᶜ) ∩ (Bᶜ) = {x : ℝ | -3 ≤ x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l2992_299291


namespace NUMINAMATH_CALUDE_ten_thousandths_digit_of_7_32_l2992_299265

theorem ten_thousandths_digit_of_7_32 : ∃ (d : ℕ), d < 10 ∧ 
  (∃ (n : ℕ), (7 : ℚ) / 32 = (n * 10 + d : ℚ) / 100000 ∧ d = 5) := by
  sorry

end NUMINAMATH_CALUDE_ten_thousandths_digit_of_7_32_l2992_299265


namespace NUMINAMATH_CALUDE_cube_side_length_l2992_299246

theorem cube_side_length (C R T : ℝ) (h1 : C = 36.50) (h2 : R = 16) (h3 : T = 876) :
  ∃ L : ℝ, L = 8 ∧ T = (6 * L^2) * (C / R) :=
sorry

end NUMINAMATH_CALUDE_cube_side_length_l2992_299246


namespace NUMINAMATH_CALUDE_congruence_problem_l2992_299225

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % (3^3) = 2^2 % (3^3))
  (h2 : (6 + x) % (5^3) = 3^2 % (5^3))
  (h3 : (8 + x) % (7^3) = 5^2 % (7^3)) :
  x % 105 = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2992_299225


namespace NUMINAMATH_CALUDE_circle_intersection_axes_l2992_299275

theorem circle_intersection_axes (m : ℝ) :
  (∃ x y : ℝ, (x - m + 1)^2 + (y - m)^2 = 1) ∧
  (∃ x : ℝ, (x - m + 1)^2 + m^2 = 1) ∧
  (∃ y : ℝ, (1 - m)^2 + (y - m)^2 = 1) →
  0 ≤ m ∧ m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_axes_l2992_299275


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l2992_299279

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 3 ∨ x = 4 ∨ x = 5) :
  c / d = 47 / 60 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l2992_299279


namespace NUMINAMATH_CALUDE_committee_formation_count_l2992_299253

def club_size : ℕ := 30
def committee_size : ℕ := 5

def ways_to_form_committee : ℕ :=
  club_size * (Nat.choose (club_size - 1) (committee_size - 1))

theorem committee_formation_count :
  ways_to_form_committee = 712530 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l2992_299253


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l2992_299212

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored) / (b.innings + 1)

theorem batsman_average_after_12th_innings
  (b : Batsman)
  (h1 : b.innings = 11)
  (h2 : newAverage b 65 = b.average + 3)
  : newAverage b 65 = 32 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l2992_299212


namespace NUMINAMATH_CALUDE_race_distance_race_distance_proof_l2992_299203

/-- The total distance of a race where:
    - The ratio of speeds of contestants A and B is 2:4
    - A has a start of 300 m
    - A wins by 100 m
-/
theorem race_distance : ℝ :=
  let speed_ratio : ℚ := 2 / 4
  let head_start : ℝ := 300
  let winning_margin : ℝ := 100
  500

theorem race_distance_proof (speed_ratio : ℚ) (head_start winning_margin : ℝ) :
  speed_ratio = 2 / 4 →
  head_start = 300 →
  winning_margin = 100 →
  race_distance = 500 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_race_distance_proof_l2992_299203


namespace NUMINAMATH_CALUDE_circle_equation_l2992_299289

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 2}

-- Define the line L: 2x - 3y - 1 = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - 3 * p.2 - 1 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (3, 0)

theorem circle_equation :
  (∃ c : ℝ × ℝ, c ∈ L ∧ c ∈ C) ∧  -- The center of C lies on L
  A ∈ C ∧ B ∈ C →                  -- C passes through A and B
  C = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 2} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2992_299289


namespace NUMINAMATH_CALUDE_derivative_at_one_is_negative_one_l2992_299268

open Real

theorem derivative_at_one_is_negative_one
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x > 0, f x = 2 * x * (deriv f 1) + log x) →
  deriv f 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_is_negative_one_l2992_299268


namespace NUMINAMATH_CALUDE_x_value_l2992_299242

theorem x_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2992_299242


namespace NUMINAMATH_CALUDE_base5_multiplication_addition_l2992_299210

/-- Converts a base 5 number to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 5 --/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The main theorem --/
theorem base5_multiplication_addition :
  decimalToBase5 (base5ToDecimal [1, 3, 2] * base5ToDecimal [1, 3] + base5ToDecimal [4, 1]) =
  [0, 3, 1, 0, 1] := by sorry

end NUMINAMATH_CALUDE_base5_multiplication_addition_l2992_299210


namespace NUMINAMATH_CALUDE_q_sum_l2992_299262

/-- Given a function q: ℝ → ℝ where q(1) = 3, prove that q(1) + q(2) = 8 -/
theorem q_sum (q : ℝ → ℝ) (h : q 1 = 3) : q 1 + q 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_q_sum_l2992_299262


namespace NUMINAMATH_CALUDE_max_sequence_length_l2992_299204

theorem max_sequence_length (x : ℕ) : 
  (68000 - 55 * x > 0) ∧ (34 * x - 42000 > 0) ↔ x = 1236 :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_l2992_299204


namespace NUMINAMATH_CALUDE_min_functional_digits_l2992_299296

def is_representable (digits : Finset ℕ) (n : ℕ) : Prop :=
  n ∈ digits ∨ ∃ a b, a ∈ digits ∧ b ∈ digits ∧ a + b = n

def is_valid_digit_set (digits : Finset ℕ) : Prop :=
  ∀ n, n ≥ 1 ∧ n ≤ 99999999 → is_representable digits n

theorem min_functional_digits :
  ∃ digits : Finset ℕ, digits.card = 5 ∧ is_valid_digit_set digits ∧
  ∀ smaller_digits : Finset ℕ, smaller_digits.card < 5 → ¬is_valid_digit_set smaller_digits :=
sorry

end NUMINAMATH_CALUDE_min_functional_digits_l2992_299296


namespace NUMINAMATH_CALUDE_team_a_win_probability_l2992_299235

/-- The probability of Team A winning a non-fifth set -/
def p_regular : ℚ := 2/3

/-- The probability of Team A winning the fifth set -/
def p_fifth : ℚ := 1/2

/-- The probability of Team A winning the match -/
def p_win : ℚ := 20/27

/-- Theorem stating that the probability of Team A winning the match is 20/27 -/
theorem team_a_win_probability : 
  p_win = p_regular^3 + 
          3 * p_regular^2 * (1 - p_regular) * p_regular + 
          6 * p_regular^2 * (1 - p_regular)^2 * p_fifth := by
  sorry

#check team_a_win_probability

end NUMINAMATH_CALUDE_team_a_win_probability_l2992_299235


namespace NUMINAMATH_CALUDE_benny_candy_bars_l2992_299259

/-- The number of candy bars Benny bought -/
def num_candy_bars : ℕ := sorry

/-- The cost of the soft drink in dollars -/
def soft_drink_cost : ℕ := 2

/-- The cost of each candy bar in dollars -/
def candy_bar_cost : ℕ := 5

/-- The total amount Benny spent in dollars -/
def total_spent : ℕ := 27

theorem benny_candy_bars :
  soft_drink_cost + num_candy_bars * candy_bar_cost = total_spent ∧
  num_candy_bars = 5 := by sorry

end NUMINAMATH_CALUDE_benny_candy_bars_l2992_299259


namespace NUMINAMATH_CALUDE_katy_brownies_theorem_l2992_299218

/-- The number of brownies Katy made -/
def total_brownies : ℕ := 15

/-- The number of brownies Katy ate on Monday -/
def monday_brownies : ℕ := 5

/-- The number of brownies Katy ate on Tuesday -/
def tuesday_brownies : ℕ := 2 * monday_brownies

theorem katy_brownies_theorem :
  total_brownies = monday_brownies + tuesday_brownies :=
by sorry

end NUMINAMATH_CALUDE_katy_brownies_theorem_l2992_299218


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2992_299249

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 4)
  (h_sum : a 1 + a 2 + a 3 = 14) :
  (∀ n : ℕ, a n = 2^n) ∧
  (∀ m n p : ℕ, m < n → n < p → a m + a p ≠ 2 * a n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2992_299249


namespace NUMINAMATH_CALUDE_max_oranges_donated_l2992_299248

theorem max_oranges_donated (n : ℕ) : ∃ (q r : ℕ), n = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_oranges_donated_l2992_299248


namespace NUMINAMATH_CALUDE_f_of_g_of_3_l2992_299266

/-- Given two functions f and g, prove that f(g(3)) = 97 -/
theorem f_of_g_of_3 (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 3 * x^2 - 2 * x + 1) 
  (hg : ∀ x, g x = x + 3) : 
  f (g 3) = 97 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_of_3_l2992_299266


namespace NUMINAMATH_CALUDE_spherical_coordinate_conversion_l2992_299281

theorem spherical_coordinate_conversion (ρ θ φ : Real) :
  ρ = 5 ∧ θ = 5 * Real.pi / 7 ∧ φ = 11 * Real.pi / 6 →
  ∃ (ρ' θ' φ' : Real),
    ρ' > 0 ∧
    0 ≤ θ' ∧ θ' < 2 * Real.pi ∧
    0 ≤ φ' ∧ φ' ≤ Real.pi ∧
    ρ' = 5 ∧
    θ' = 12 * Real.pi / 7 ∧
    φ' = Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_conversion_l2992_299281


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_cube_l2992_299267

theorem smallest_sum_of_squares_cube (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 →
  x ≠ y → y ≠ z → x ≠ z →
  x^2 + y^2 = z^3 →
  ∀ a b c : ℕ, a > 0 → b > 0 → c > 0 → 
    a ≠ b → b ≠ c → a ≠ c →
    a^2 + b^2 = c^3 →
    x + y + z ≤ a + b + c →
  x + y + z = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_cube_l2992_299267


namespace NUMINAMATH_CALUDE_solve_for_c_l2992_299233

/-- Given two functions p and q, where p(x) = 3x - 9 and q(x) = 4x - c,
    prove that c = 4 when p(q(3)) = 15 -/
theorem solve_for_c (p q : ℝ → ℝ) (c : ℝ) 
    (hp : ∀ x, p x = 3 * x - 9)
    (hq : ∀ x, q x = 4 * x - c)
    (h_eq : p (q 3) = 15) : 
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l2992_299233


namespace NUMINAMATH_CALUDE_set_inclusion_condition_l2992_299252

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 8 < 0}

def B : Set ℝ := {x : ℝ | x^2 + 2*x - 3 > 0}

def C (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 3*a*x + 2*a^2 < 0}

theorem set_inclusion_condition (a : ℝ) : 
  C a ⊆ A ∩ B ↔ (1 ≤ a ∧ a ≤ 2) ∨ a = 0 := by sorry

end NUMINAMATH_CALUDE_set_inclusion_condition_l2992_299252


namespace NUMINAMATH_CALUDE_field_length_is_16_l2992_299273

/-- Proves that the length of a rectangular field is 16 meters given specific conditions --/
theorem field_length_is_16 (w : ℝ) (l : ℝ) : 
  l = 2 * w →  -- length is double the width
  16 = (1/8) * (l * w) →  -- pond area (4^2) is 1/8 of field area
  l = 16 := by
sorry


end NUMINAMATH_CALUDE_field_length_is_16_l2992_299273


namespace NUMINAMATH_CALUDE_probability_of_selecting_female_student_l2992_299247

theorem probability_of_selecting_female_student :
  let total_students : ℕ := 4
  let female_students : ℕ := 3
  let male_students : ℕ := 1
  female_students + male_students = total_students →
  (female_students : ℚ) / total_students = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_female_student_l2992_299247


namespace NUMINAMATH_CALUDE_triangle_area_l2992_299213

theorem triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 12) (h3 : c = 5) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2992_299213


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l2992_299234

theorem min_value_of_sum_of_roots (x y : ℝ) :
  let z := Real.sqrt (x^2 + y^2 - 2*x - 2*y + 2) + Real.sqrt (x^2 + y^2 - 4*y + 4)
  z ≥ Real.sqrt 2 ∧
  (z = Real.sqrt 2 ↔ y = 2 - x ∧ 1 ≤ x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l2992_299234


namespace NUMINAMATH_CALUDE_cyclist_distance_difference_l2992_299223

/-- Represents a cyclist with a constant speed --/
structure Cyclist where
  speed : ℝ

/-- Calculates the distance traveled by a cyclist in a given time --/
def distance_traveled (cyclist : Cyclist) (time : ℝ) : ℝ :=
  cyclist.speed * time

/-- Theorem: The difference in distance traveled between two cyclists after 5 hours --/
theorem cyclist_distance_difference 
  (carlos dana : Cyclist)
  (h1 : carlos.speed = 0.9)
  (h2 : dana.speed = 0.72)
  : distance_traveled carlos 5 - distance_traveled dana 5 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_distance_difference_l2992_299223


namespace NUMINAMATH_CALUDE_fifth_power_sum_l2992_299214

theorem fifth_power_sum (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : 1 / (x + y^2) + 1 / (x^2 + y) = 1 / 2) : 
  x^5 + y^5 = 123 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l2992_299214


namespace NUMINAMATH_CALUDE_probability_A_and_B_selected_l2992_299274

theorem probability_A_and_B_selected (total_students : ℕ) (selected_students : ℕ) 
  (h1 : total_students = 5) (h2 : selected_students = 3) :
  (Nat.choose (total_students - 2) (selected_students - 2)) / (Nat.choose total_students selected_students) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_and_B_selected_l2992_299274


namespace NUMINAMATH_CALUDE_lisa_book_purchase_l2992_299250

theorem lisa_book_purchase (total_volumes : ℕ) (standard_cost deluxe_cost total_cost : ℕ) 
  (h1 : total_volumes = 15)
  (h2 : standard_cost = 20)
  (h3 : deluxe_cost = 30)
  (h4 : total_cost = 390) :
  ∃ (deluxe_count : ℕ), 
    deluxe_count * deluxe_cost + (total_volumes - deluxe_count) * standard_cost = total_cost ∧
    deluxe_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_lisa_book_purchase_l2992_299250


namespace NUMINAMATH_CALUDE_quadratic_roots_identities_l2992_299299

theorem quadratic_roots_identities (x₁ x₂ S P : ℝ) 
  (hS : S = x₁ + x₂) 
  (hP : P = x₁ * x₂) : 
  (x₁^2 + x₂^2 = S^2 - 2*P) ∧ 
  (x₁^3 + x₂^3 = S^3 - 3*S*P) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_identities_l2992_299299


namespace NUMINAMATH_CALUDE_superhero_payment_l2992_299290

/-- Superhero payment calculation -/
theorem superhero_payment (W : ℝ) : 
  let superman_productivity := 0.1 * W
  let flash_productivity := 2 * superman_productivity
  let combined_productivity := superman_productivity + flash_productivity
  let remaining_work := 0.9 * W
  let combined_time := remaining_work / combined_productivity
  let superman_total_time := 1 + combined_time
  let flash_total_time := combined_time
  let payment (t : ℝ) := 90 / t
  (payment superman_total_time = 22.5) ∧ (payment flash_total_time = 30) :=
by sorry

end NUMINAMATH_CALUDE_superhero_payment_l2992_299290


namespace NUMINAMATH_CALUDE_max_intersections_pentagon_circle_l2992_299209

/-- A regular pentagon -/
structure RegularPentagon where
  -- We don't need to define the structure, just declare it exists
  
/-- A circle -/
structure Circle where
  -- We don't need to define the structure, just declare it exists

/-- The maximum number of intersections between a line segment and a circle is 2 -/
axiom max_intersections_line_circle : ℕ

/-- The number of sides in a regular pentagon -/
def pentagon_sides : ℕ := 5

/-- Theorem: The maximum number of intersections between a regular pentagon and a circle is 10 -/
theorem max_intersections_pentagon_circle (p : RegularPentagon) (c : Circle) :
  (max_intersections_line_circle * pentagon_sides : ℕ) = 10 := by
  sorry

#check max_intersections_pentagon_circle

end NUMINAMATH_CALUDE_max_intersections_pentagon_circle_l2992_299209


namespace NUMINAMATH_CALUDE_inequality_proof_l2992_299236

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2992_299236


namespace NUMINAMATH_CALUDE_nested_cube_root_simplification_l2992_299238

theorem nested_cube_root_simplification (N : ℝ) (h : N > 1) :
  (4 * N * (8 * N * (12 * N)^(1/3))^(1/3))^(1/3) = 2 * 3^(1/3) * N^(13/27) := by
  sorry

end NUMINAMATH_CALUDE_nested_cube_root_simplification_l2992_299238


namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l2992_299285

/-- Sine of 30 degrees is 1/2 -/
theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l2992_299285
