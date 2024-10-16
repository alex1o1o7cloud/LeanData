import Mathlib

namespace NUMINAMATH_CALUDE_tablets_consumed_l3314_331443

/-- Proves that given a person who takes one tablet every 15 minutes and consumes all tablets in 60 minutes, the total number of tablets taken is 4. -/
theorem tablets_consumed (interval : ℕ) (total_time : ℕ) (h1 : interval = 15) (h2 : total_time = 60) :
  total_time / interval = 4 := by
  sorry

end NUMINAMATH_CALUDE_tablets_consumed_l3314_331443


namespace NUMINAMATH_CALUDE_crust_vs_bread_expenditure_l3314_331450

/-- Represents the percentage increase in expenditure when buying crust instead of bread -/
def expenditure_increase : ℝ := 36

/-- The ratio of crust weight to bread weight -/
def crust_weight_ratio : ℝ := 0.75

/-- The ratio of crust price to bread price -/
def crust_price_ratio : ℝ := 1.2

/-- The ratio of bread that is actually consumed -/
def bread_consumption_ratio : ℝ := 0.85

/-- The ratio of crust that is actually consumed -/
def crust_consumption_ratio : ℝ := 1

theorem crust_vs_bread_expenditure :
  expenditure_increase = 
    ((crust_consumption_ratio / crust_weight_ratio) / 
     bread_consumption_ratio * crust_price_ratio - 1) * 100 := by
  sorry

#eval expenditure_increase

end NUMINAMATH_CALUDE_crust_vs_bread_expenditure_l3314_331450


namespace NUMINAMATH_CALUDE_base_difference_l3314_331498

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_difference : 
  let base_9_number := [1, 2, 3]  -- 321 in base 9, least significant digit first
  let base_6_number := [5, 6, 1]  -- 165 in base 6, least significant digit first
  (to_base_10 base_9_number 9) - (to_base_10 base_6_number 6) = 221 := by
  sorry


end NUMINAMATH_CALUDE_base_difference_l3314_331498


namespace NUMINAMATH_CALUDE_phone_initial_price_l3314_331494

/-- Given a phone's negotiated price of $480, which is 20% of its initial price,
    prove that the initial price of the phone is $2400. -/
theorem phone_initial_price (negotiated_price : ℝ) (percentage : ℝ) 
    (h1 : negotiated_price = 480)
    (h2 : percentage = 0.20)
    (h3 : negotiated_price = percentage * initial_price) : 
    initial_price = 2400 :=
by sorry

end NUMINAMATH_CALUDE_phone_initial_price_l3314_331494


namespace NUMINAMATH_CALUDE_log_relationship_l3314_331431

theorem log_relationship (a b : ℝ) : 
  a = Real.log 243 / Real.log 5 → b = Real.log 27 / Real.log 3 → a = 5 * b / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_relationship_l3314_331431


namespace NUMINAMATH_CALUDE_dodecagon_arrangement_impossible_l3314_331451

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def arrangement := Fin 12 → Fin 12

def valid_arrangement (a : arrangement) : Prop :=
  ∀ i : Fin 12, ∃ j : Fin 12, a j = i

def adjacent_sum_prime (a : arrangement) : Prop :=
  ∀ i : Fin 12, is_prime ((a i).val + 1 + (a ((i + 1) % 12)).val + 1)

def skip_two_sum_prime (a : arrangement) : Prop :=
  ∀ i : Fin 12, is_prime ((a i).val + 1 + (a ((i + 3) % 12)).val + 1)

theorem dodecagon_arrangement_impossible :
  ¬∃ a : arrangement, valid_arrangement a ∧ adjacent_sum_prime a ∧ skip_two_sum_prime a :=
sorry

end NUMINAMATH_CALUDE_dodecagon_arrangement_impossible_l3314_331451


namespace NUMINAMATH_CALUDE_sumata_family_driving_l3314_331481

/-- The Sumata family's driving problem -/
theorem sumata_family_driving (days : ℝ) (miles_per_day : ℝ) 
  (h1 : days = 5.0)
  (h2 : miles_per_day = 50) :
  days * miles_per_day = 250 := by
  sorry

end NUMINAMATH_CALUDE_sumata_family_driving_l3314_331481


namespace NUMINAMATH_CALUDE_jake_earnings_l3314_331440

/-- Jake's earnings calculation -/
theorem jake_earnings (jacob_hourly_rate : ℝ) (jake_daily_hours : ℝ) (days : ℝ) :
  jacob_hourly_rate = 6 →
  jake_daily_hours = 8 →
  days = 5 →
  (3 * jacob_hourly_rate * jake_daily_hours * days : ℝ) = 720 := by
  sorry

end NUMINAMATH_CALUDE_jake_earnings_l3314_331440


namespace NUMINAMATH_CALUDE_first_day_hike_distance_l3314_331419

/-- A hike with two participants -/
structure Hike where
  total_distance : ℕ
  distance_left : ℕ
  tripp_backpack_weight : ℕ
  charlotte_backpack_weight : ℕ
  (charlotte_lighter : charlotte_backpack_weight = tripp_backpack_weight - 7)

/-- The distance hiked on the first day -/
def distance_hiked_first_day (h : Hike) : ℕ :=
  h.total_distance - h.distance_left

theorem first_day_hike_distance (h : Hike) 
  (h_total : h.total_distance = 36) 
  (h_left : h.distance_left = 27) : 
  distance_hiked_first_day h = 9 := by
  sorry

end NUMINAMATH_CALUDE_first_day_hike_distance_l3314_331419


namespace NUMINAMATH_CALUDE_free_younger_son_time_l3314_331469

/-- Given a total number of tape strands and cutting rates for Hannah and her son,
    calculate the time needed to cut all strands. -/
def time_to_cut (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℚ :=
  (total_strands : ℚ) / ((hannah_rate + son_rate) : ℚ)

/-- Theorem stating that it takes 5 minutes to cut 45 strands of tape
    when Hannah cuts 7 strands per minute and her son cuts 2 strands per minute. -/
theorem free_younger_son_time :
  time_to_cut 45 7 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_free_younger_son_time_l3314_331469


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3314_331453

theorem geometric_sequence_seventh_term 
  (a : ℝ) (a₃ : ℝ) (n : ℕ) (h₁ : a = 3) (h₂ : a₃ = 3/64) (h₃ : n = 7) :
  a * (a₃ / a) ^ ((n - 1) / 2) = 3/262144 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3314_331453


namespace NUMINAMATH_CALUDE_impossible_all_black_l3314_331401

/-- Represents the color of a square on the board -/
inductive Color
| White
| Black

/-- Represents a 4x4 board -/
def Board := Fin 4 → Fin 4 → Color

/-- Represents a 1x3 rectangle on the board -/
structure Rectangle :=
  (row : Fin 4)
  (col : Fin 4)
  (horizontal : Bool)

/-- Initial state of the board where all squares are white -/
def initialBoard : Board :=
  λ _ _ => Color.White

/-- Applies a move to the board by flipping colors in a 1x3 rectangle -/
def applyMove (b : Board) (r : Rectangle) : Board :=
  sorry

/-- Checks if all squares on the board are black -/
def allBlack (b : Board) : Prop :=
  ∀ i j, b i j = Color.Black

/-- Theorem stating that it's impossible to make all squares black -/
theorem impossible_all_black :
  ¬ ∃ (moves : List Rectangle), allBlack (moves.foldl applyMove initialBoard) :=
sorry

end NUMINAMATH_CALUDE_impossible_all_black_l3314_331401


namespace NUMINAMATH_CALUDE_max_type_c_test_tubes_l3314_331457

/-- Represents the number of test tubes of each type -/
structure TestTubes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the solution percentages are valid -/
def validSolution (t : TestTubes) : Prop :=
  10 * t.a + 20 * t.b + 90 * t.c = 2017 * (t.a + t.b + t.c)

/-- Checks if the total number of test tubes is 1000 -/
def totalIs1000 (t : TestTubes) : Prop :=
  t.a + t.b + t.c = 1000

/-- Checks if test tubes of the same type are not used consecutively -/
def noConsecutiveSameType (t : TestTubes) : Prop :=
  7 * t.c ≤ 517 ∧ 8 * t.c ≥ 518 ∧ t.c ≤ 500

/-- Theorem: The maximum number of type C test tubes is 73 -/
theorem max_type_c_test_tubes :
  ∃ (t : TestTubes),
    validSolution t ∧
    totalIs1000 t ∧
    noConsecutiveSameType t ∧
    (∀ (t' : TestTubes),
      validSolution t' ∧ totalIs1000 t' ∧ noConsecutiveSameType t' →
      t'.c ≤ t.c) ∧
    t.c = 73 :=
  sorry

end NUMINAMATH_CALUDE_max_type_c_test_tubes_l3314_331457


namespace NUMINAMATH_CALUDE_min_value_theorem_l3314_331424

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  x + 4 / (x - 1) ≥ 5 ∧ (x + 4 / (x - 1) = 5 ↔ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3314_331424


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3314_331438

theorem no_integer_solutions : ¬ ∃ x : ℤ, ∃ k : ℤ, x^2 + x + 13 = 121 * k := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3314_331438


namespace NUMINAMATH_CALUDE_mod_product_equiv_l3314_331400

theorem mod_product_equiv (m : ℕ) : 
  (264 * 391 ≡ m [ZMOD 100]) → 
  (0 ≤ m ∧ m < 100) → 
  m = 24 := by
  sorry

end NUMINAMATH_CALUDE_mod_product_equiv_l3314_331400


namespace NUMINAMATH_CALUDE_sin_transformation_l3314_331437

theorem sin_transformation (x : ℝ) : 
  2 * Real.sin (x / 3 - π / 6) = 2 * Real.sin ((x - π / 2) / 3) := by sorry

end NUMINAMATH_CALUDE_sin_transformation_l3314_331437


namespace NUMINAMATH_CALUDE_proportion_not_greater_than_30_proportion_as_percentage_l3314_331480

-- Define the sample size
def sample_size : ℕ := 50

-- Define the number of data points greater than 30
def data_greater_than_30 : ℕ := 3

-- Define the proportion calculation function
def calculate_proportion (total : ℕ) (part : ℕ) : ℚ :=
  (total - part : ℚ) / total

-- Theorem statement
theorem proportion_not_greater_than_30 :
  calculate_proportion sample_size data_greater_than_30 = 47/50 :=
by
  sorry

-- Additional theorem to show the decimal representation
theorem proportion_as_percentage :
  (calculate_proportion sample_size data_greater_than_30 * 100 : ℚ) = 94 :=
by
  sorry

end NUMINAMATH_CALUDE_proportion_not_greater_than_30_proportion_as_percentage_l3314_331480


namespace NUMINAMATH_CALUDE_num_chords_from_nine_points_l3314_331478

/-- The number of points on the circumference of the circle -/
def num_points : ℕ := 9

/-- A function to calculate the number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the number of distinct chords from 9 points is 36 -/
theorem num_chords_from_nine_points : 
  choose_two num_points = 36 := by sorry

end NUMINAMATH_CALUDE_num_chords_from_nine_points_l3314_331478


namespace NUMINAMATH_CALUDE_laura_charge_account_theorem_l3314_331446

/-- Calculates the total amount owed after one year with simple interest -/
def total_amount_owed (principal : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * interest_rate * time

/-- Proves that the total amount owed after one year is $37.45 -/
theorem laura_charge_account_theorem :
  let principal : ℝ := 35
  let interest_rate : ℝ := 0.07
  let time : ℝ := 1
  total_amount_owed principal interest_rate time = 37.45 := by
sorry

end NUMINAMATH_CALUDE_laura_charge_account_theorem_l3314_331446


namespace NUMINAMATH_CALUDE_union_of_sets_l3314_331472

theorem union_of_sets : 
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3314_331472


namespace NUMINAMATH_CALUDE_math_test_questions_math_test_questions_proof_l3314_331486

theorem math_test_questions : ℕ → Prop :=
  fun total_questions =>
    let word_problems : ℕ := 17
    let addition_subtraction_problems : ℕ := 28
    let steve_answered : ℕ := 38
    let difference : ℕ := 7
    
    (total_questions - steve_answered = difference) ∧
    (word_problems + addition_subtraction_problems ≤ total_questions) ∧
    (steve_answered < total_questions) →
    total_questions = 45

-- The proof is omitted
theorem math_test_questions_proof : math_test_questions 45 := by sorry

end NUMINAMATH_CALUDE_math_test_questions_math_test_questions_proof_l3314_331486


namespace NUMINAMATH_CALUDE_polynomial_derivative_sum_l3314_331410

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_derivative_sum_l3314_331410


namespace NUMINAMATH_CALUDE_purchase_in_fourth_month_l3314_331433

/-- Represents the financial state of the family --/
structure FamilyFinance where
  monthlyIncome : ℕ
  monthlyExpenses : ℕ
  initialSavings : ℕ
  furnitureCost : ℕ

/-- Calculates the month when the family can make the purchase --/
def purchaseMonth (finance : FamilyFinance) : ℕ :=
  let monthlySavings := finance.monthlyIncome - finance.monthlyExpenses
  let additionalRequired := finance.furnitureCost - finance.initialSavings
  (additionalRequired + monthlySavings - 1) / monthlySavings + 1

/-- The main theorem stating that the family can make the purchase in the 4th month --/
theorem purchase_in_fourth_month (finance : FamilyFinance) 
  (h1 : finance.monthlyIncome = 150000)
  (h2 : finance.monthlyExpenses = 115000)
  (h3 : finance.initialSavings = 45000)
  (h4 : finance.furnitureCost = 127000) :
  purchaseMonth finance = 4 := by
  sorry

#eval purchaseMonth { 
  monthlyIncome := 150000, 
  monthlyExpenses := 115000, 
  initialSavings := 45000, 
  furnitureCost := 127000 
}

end NUMINAMATH_CALUDE_purchase_in_fourth_month_l3314_331433


namespace NUMINAMATH_CALUDE_circle_outside_triangle_percentage_l3314_331484

theorem circle_outside_triangle_percentage
  (A : ℝ) -- Total area
  (A_intersection : ℝ) -- Area of intersection
  (A_triangle_outside : ℝ) -- Area of triangle outside circle
  (h1 : A > 0) -- Total area is positive
  (h2 : A_intersection = 0.45 * A) -- Intersection is 45% of total area
  (h3 : A_triangle_outside = 0.4 * A) -- Triangle outside is 40% of total area
  : (A - A_intersection - A_triangle_outside) / (A_intersection + (A - A_intersection - A_triangle_outside)) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_outside_triangle_percentage_l3314_331484


namespace NUMINAMATH_CALUDE_line_equation_60_degrees_l3314_331434

/-- The equation of a line with a slope of 60° and a y-intercept of -1 -/
theorem line_equation_60_degrees (x y : ℝ) :
  let slope : ℝ := Real.tan (60 * π / 180)
  let y_intercept : ℝ := -1
  slope * x - y - y_intercept = 0 ↔ Real.sqrt 3 * x - y - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_60_degrees_l3314_331434


namespace NUMINAMATH_CALUDE_equation_solutions_l3314_331497

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ > 0 ∧ x₂ > 0) ∧
    (∀ x : ℝ, x > 0 → 
      ((1/2) * (4*x^2 - 1) = (x^2 - 60*x - 20) * (x^2 + 30*x + 10)) ↔ 
      (x = x₁ ∨ x = x₂)) ∧
    x₁ = 30 + Real.sqrt 919 ∧
    x₂ = -15 + Real.sqrt 216 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3314_331497


namespace NUMINAMATH_CALUDE_not_always_greater_quotient_l3314_331413

theorem not_always_greater_quotient : ¬ ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → (∃ n : ℤ, b = n / 10) → a / b > a := by
  sorry

end NUMINAMATH_CALUDE_not_always_greater_quotient_l3314_331413


namespace NUMINAMATH_CALUDE_power_two_minus_one_div_by_seven_l3314_331403

theorem power_two_minus_one_div_by_seven (n : ℕ) : 
  7 ∣ (2^n - 1) ↔ 3 ∣ n :=
sorry

end NUMINAMATH_CALUDE_power_two_minus_one_div_by_seven_l3314_331403


namespace NUMINAMATH_CALUDE_opposite_of_three_l3314_331414

theorem opposite_of_three : -(3 : ℤ) = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3314_331414


namespace NUMINAMATH_CALUDE_speed_conversion_l3314_331425

/-- Conversion of speed from m/s to km/h -/
theorem speed_conversion (speed_ms : ℚ) (conversion_factor : ℚ) :
  speed_ms = 13/36 →
  conversion_factor = 36/10 →
  speed_ms * conversion_factor = 13/10 := by
  sorry

#eval (13/36 : ℚ) * (36/10 : ℚ) -- To verify the result

end NUMINAMATH_CALUDE_speed_conversion_l3314_331425


namespace NUMINAMATH_CALUDE_mike_practice_hours_l3314_331483

/-- Calculates the number of hours Mike practices every weekday -/
def weekday_practice_hours (days_in_week : ℕ) (practice_days_per_week : ℕ) 
  (saturday_hours : ℕ) (total_weeks : ℕ) (total_practice_hours : ℕ) : ℕ :=
  let total_practice_days := practice_days_per_week * total_weeks
  let total_saturdays := total_weeks
  let saturday_practice_hours := saturday_hours * total_saturdays
  let weekday_practice_hours := total_practice_hours - saturday_practice_hours
  let total_weekdays := (practice_days_per_week - 1) * total_weeks
  weekday_practice_hours / total_weekdays

/-- Theorem stating that Mike practices 3 hours every weekday -/
theorem mike_practice_hours : 
  weekday_practice_hours 7 6 5 3 60 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mike_practice_hours_l3314_331483


namespace NUMINAMATH_CALUDE_min_difference_l3314_331464

noncomputable def f (x : ℝ) : ℝ := Real.exp (4 * x - 1)

noncomputable def g (x : ℝ) : ℝ := 1/2 + Real.log (2 * x)

theorem min_difference (m n : ℝ) (h : f m = g n) :
  ∃ (m₀ n₀ : ℝ), f m₀ = g n₀ ∧ ∀ m' n', f m' = g n' → n₀ - m₀ ≤ n' - m' ∧ n₀ - m₀ = (1 + Real.log 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_min_difference_l3314_331464


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3314_331406

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 2*x₁ - 3 = 0) ∧ 
  (x₂^2 - 2*x₂ - 3 = 0) ∧ 
  x₁ = 3 ∧ 
  x₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3314_331406


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3314_331441

/-- Proves the equality of the expanded polynomial expression -/
theorem polynomial_expansion (y : ℝ) : 
  (3 * y + 2) * (5 * y^12 - y^11 + 3 * y^10 + 2) = 
  15 * y^13 + 7 * y^12 + 7 * y^11 + 6 * y^10 + 6 * y + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3314_331441


namespace NUMINAMATH_CALUDE_paving_stone_width_l3314_331416

/-- Given a rectangular courtyard and paving stones with specific dimensions,
    prove that the width of each paving stone is 1 meter. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (stone_count : ℕ)
  (h1 : courtyard_length = 30)
  (h2 : courtyard_width = 16)
  (h3 : stone_length = 2)
  (h4 : stone_count = 240)
  : ∃ (stone_width : ℝ), 
    stone_width = 1 ∧ 
    courtyard_length * courtyard_width = ↑stone_count * stone_length * stone_width :=
by sorry

end NUMINAMATH_CALUDE_paving_stone_width_l3314_331416


namespace NUMINAMATH_CALUDE_vladimir_investment_opportunity_l3314_331479

/-- Represents the value of 1 kg of buckwheat in rubles -/
def buckwheat_value : ℝ := 85

/-- Represents the initial price of 1 kg of buckwheat in rubles -/
def initial_price : ℝ := 70

/-- Calculates the value after a one-year deposit at the given rate -/
def one_year_deposit (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

/-- Calculates the value after a two-year deposit at the given rate -/
def two_year_deposit (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate) * (1 + rate)

/-- Represents the annual deposit rate for 2015 -/
def rate_2015 : ℝ := 0.16

/-- Represents the annual deposit rate for 2016 -/
def rate_2016 : ℝ := 0.10

/-- Represents the two-year deposit rate starting from 2015 -/
def rate_2015_2016 : ℝ := 0.15

theorem vladimir_investment_opportunity : 
  let option1 := one_year_deposit (one_year_deposit initial_price rate_2015) rate_2016
  let option2 := two_year_deposit initial_price rate_2015_2016
  max option1 option2 > buckwheat_value := by sorry

end NUMINAMATH_CALUDE_vladimir_investment_opportunity_l3314_331479


namespace NUMINAMATH_CALUDE_vectors_problem_l3314_331417

def a : ℝ × ℝ := (3, -1)
def b (k : ℝ) : ℝ × ℝ := (1, k)

theorem vectors_problem (k : ℝ) 
  (h : a.1 * (b k).1 + a.2 * (b k).2 = 0) : 
  k = 3 ∧ 
  (a.1 + (b k).1) * (a.1 - (b k).1) + (a.2 + (b k).2) * (a.2 - (b k).2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_problem_l3314_331417


namespace NUMINAMATH_CALUDE_least_common_period_l3314_331426

-- Define the property that f should satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) + f (x - 3) = f x

-- Define what it means for a function to have a period
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_common_period :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
    (∃ p : ℝ, p > 0 ∧ HasPeriod f p) →
    (∀ q : ℝ, q > 0 ∧ HasPeriod f q → q ≥ 18) ∧
    HasPeriod f 18 :=
sorry

end NUMINAMATH_CALUDE_least_common_period_l3314_331426


namespace NUMINAMATH_CALUDE_heptagon_interior_angle_sum_heptagon_interior_angle_sum_proof_l3314_331435

/-- The sum of the interior angles of a heptagon is 900 degrees. -/
theorem heptagon_interior_angle_sum : ℝ :=
  900

/-- A heptagon is a polygon with 7 sides. -/
def heptagon_sides : ℕ := 7

/-- The formula for the sum of interior angles of a polygon with n sides. -/
def polygon_interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

theorem heptagon_interior_angle_sum_proof :
  polygon_interior_angle_sum heptagon_sides = heptagon_interior_angle_sum :=
by
  sorry

end NUMINAMATH_CALUDE_heptagon_interior_angle_sum_heptagon_interior_angle_sum_proof_l3314_331435


namespace NUMINAMATH_CALUDE_unique_valid_number_l3314_331490

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  n / 1000 = 764 ∧
  n % 8 = 0 ∧ n % 9 = 0 ∧ n % 11 = 0

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 764280 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3314_331490


namespace NUMINAMATH_CALUDE_graduates_distribution_l3314_331466

def distribute_graduates (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem graduates_distribution :
  distribute_graduates 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_graduates_distribution_l3314_331466


namespace NUMINAMATH_CALUDE_divisibility_by_x_squared_minus_one_cubed_l3314_331428

theorem divisibility_by_x_squared_minus_one_cubed (n : ℕ) :
  ∃ P : Polynomial ℚ, 
    X^(4*n+2) - (2*n+1) * X^(2*n+2) + (2*n+1) * X^(2*n) - 1 = 
    (X^2 - 1)^3 * P :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_x_squared_minus_one_cubed_l3314_331428


namespace NUMINAMATH_CALUDE_max_pages_copied_l3314_331470

/-- The number of pages that can be copied given a budget and copying costs -/
def pages_copied (cost_per_4_pages : ℕ) (flat_fee : ℕ) (budget : ℕ) : ℕ :=
  ((budget - flat_fee) * 4) / cost_per_4_pages

/-- Theorem stating the maximum number of pages that can be copied under given conditions -/
theorem max_pages_copied : 
  pages_copied 7 100 3000 = 1657 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_copied_l3314_331470


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3314_331489

theorem quadratic_one_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x, 3 * x^2 + b * x + 12 = 0) →
  ((b = 12 ∧ ∃ x, 3 * x^2 + b * x + 12 = 0 ∧ x = -2) ∨
   (b = -12 ∧ ∃ x, 3 * x^2 + b * x + 12 = 0 ∧ x = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3314_331489


namespace NUMINAMATH_CALUDE_cistern_filling_time_l3314_331432

/-- Given a cistern that can be emptied by a tap in 10 hours, and when both this tap and another tap
    are opened simultaneously the cistern gets filled in 30/7 hours, prove that the time it takes
    for the other tap alone to fill the cistern is 3 hours. -/
theorem cistern_filling_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) :
  empty_rate = 10 →
  combined_fill_time = 30 / 7 →
  1 / fill_time - 1 / empty_rate = 1 / combined_fill_time →
  fill_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l3314_331432


namespace NUMINAMATH_CALUDE_solve_for_a_l3314_331444

theorem solve_for_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l3314_331444


namespace NUMINAMATH_CALUDE_fraction_increase_l3314_331456

theorem fraction_increase (n : ℚ) (x : ℚ) : 
  n / (n + 5) = 6 / 11 → 
  (n + x) / (n + 5 + x) = 7 / 12 → 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_increase_l3314_331456


namespace NUMINAMATH_CALUDE_next_joint_work_day_is_360_l3314_331485

/-- Represents the work schedule of a tutor -/
structure TutorSchedule where
  cycle : ℕ

/-- Represents the lab schedule -/
structure LabSchedule where
  openDays : Fin 7 → Bool

/-- Calculates the next day all tutors work together -/
def nextJointWorkDay (emma noah olivia liam : TutorSchedule) (lab : LabSchedule) : ℕ :=
  sorry

theorem next_joint_work_day_is_360 :
  let emma : TutorSchedule := { cycle := 5 }
  let noah : TutorSchedule := { cycle := 8 }
  let olivia : TutorSchedule := { cycle := 9 }
  let liam : TutorSchedule := { cycle := 10 }
  let lab : LabSchedule := { openDays := fun d => d < 5 }
  nextJointWorkDay emma noah olivia liam lab = 360 := by
  sorry

end NUMINAMATH_CALUDE_next_joint_work_day_is_360_l3314_331485


namespace NUMINAMATH_CALUDE_paper_towel_savings_l3314_331499

theorem paper_towel_savings : 
  let case_price : ℚ := 9
  let individual_price : ℚ := 1
  let rolls_per_case : ℕ := 12
  let case_price_per_roll : ℚ := case_price / rolls_per_case
  let savings_per_roll : ℚ := individual_price - case_price_per_roll
  let percent_savings : ℚ := (savings_per_roll / individual_price) * 100
  percent_savings = 25 := by sorry

end NUMINAMATH_CALUDE_paper_towel_savings_l3314_331499


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3314_331407

/-- Given roots r, s, and t of the equation 10x³ + 500x + 1500 = 0,
    prove that (r+s)³ + (t+s)³ + (r+t)³ = -450 -/
theorem cubic_root_sum_cubes (r s t : ℝ) :
  (10 * r^3 + 500 * r + 1500 = 0) →
  (10 * s^3 + 500 * s + 1500 = 0) →
  (10 * t^3 + 500 * t + 1500 = 0) →
  (r + s)^3 + (t + s)^3 + (r + t)^3 = -450 := by
  sorry


end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3314_331407


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_l3314_331467

theorem units_digit_of_7_power : 7^(100^6) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_l3314_331467


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3314_331420

/-- A line passing through a point -/
def line_passes_through (k : ℝ) (x y : ℝ) : Prop :=
  2 - k * x = -5 * y

/-- The theorem stating that the line passes through the given point when k = -0.5 -/
theorem line_passes_through_point :
  line_passes_through (-0.5) 6 (-1) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3314_331420


namespace NUMINAMATH_CALUDE_division_ways_correct_l3314_331421

/-- The number of ways to divide 6 distinct objects into three groups,
    where one group has 4 objects and the other two groups have 1 object each. -/
def divisionWays : ℕ := 15

/-- The total number of objects to be divided. -/
def totalObjects : ℕ := 6

/-- The number of objects in the largest group. -/
def largestGroupSize : ℕ := 4

/-- The number of groups. -/
def numberOfGroups : ℕ := 3

/-- Theorem stating that the number of ways to divide the objects is correct. -/
theorem division_ways_correct :
  divisionWays = Nat.choose totalObjects largestGroupSize :=
sorry

end NUMINAMATH_CALUDE_division_ways_correct_l3314_331421


namespace NUMINAMATH_CALUDE_max_value_expression_l3314_331491

theorem max_value_expression (a b c : ℝ) 
  (ha : -1 ≤ a ∧ a ≤ 1) 
  (hb : -1 ≤ b ∧ b ≤ 1) 
  (hc : -1 ≤ c ∧ c ≤ 1) : 
  ∀ x y z : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ y ∧ y ≤ 1 → -1 ≤ z ∧ z ≤ 1 → 
  2 * Real.sqrt (a * b * c) + Real.sqrt ((1 - a^2) * (1 - b^2) * (1 - c^2)) ≤ 
  2 * Real.sqrt (x * y * z) + Real.sqrt ((1 - x^2) * (1 - y^2) * (1 - z^2)) → 
  2 * Real.sqrt (x * y * z) + Real.sqrt ((1 - x^2) * (1 - y^2) * (1 - z^2)) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3314_331491


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l3314_331429

theorem inverse_proposition_false : 
  ¬(∀ (a b c : ℝ), a > b → a / (c^2) > b / (c^2)) := by
sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l3314_331429


namespace NUMINAMATH_CALUDE_modulus_of_Z_l3314_331477

-- Define the operation
def matrix_op (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem modulus_of_Z : ∃ (Z : ℂ), 
  (matrix_op Z i 1 i = 1 + i) ∧ (Complex.abs Z = Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_Z_l3314_331477


namespace NUMINAMATH_CALUDE_weight_difference_l3314_331430

/-- Given weights of five individuals A, B, C, D, and E, prove that E weighs 6 kg more than D
    under specific average weight conditions. -/
theorem weight_difference (W_A W_B W_C W_D W_E : ℝ) : 
  (W_A + W_B + W_C) / 3 = 84 →
  (W_A + W_B + W_C + W_D) / 4 = 80 →
  (W_B + W_C + W_D + W_E) / 4 = 79 →
  W_A = 78 →
  W_E - W_D = 6 := by
sorry

end NUMINAMATH_CALUDE_weight_difference_l3314_331430


namespace NUMINAMATH_CALUDE_pauls_money_duration_l3314_331487

/-- Given Paul's earnings and weekly spending, prove how long the money will last. -/
theorem pauls_money_duration (lawn_earnings weed_earnings weekly_spending : ℕ) 
  (h1 : lawn_earnings = 68)
  (h2 : weed_earnings = 13)
  (h3 : weekly_spending = 9) :
  (lawn_earnings + weed_earnings) / weekly_spending = 9 := by
  sorry

end NUMINAMATH_CALUDE_pauls_money_duration_l3314_331487


namespace NUMINAMATH_CALUDE_toothpick_grid_theorem_l3314_331460

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ
  has_diagonals : Bool

/-- Calculates the total number of toothpicks in the grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontal := (grid.height + 1) * grid.width
  let vertical := (grid.width + 1) * grid.height
  let diagonal := if grid.has_diagonals then grid.height * grid.width else 0
  horizontal + vertical + diagonal

/-- The theorem to be proved -/
theorem toothpick_grid_theorem (grid : ToothpickGrid) :
  grid.height = 15 → grid.width = 12 → grid.has_diagonals = true →
  total_toothpicks grid = 567 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grid_theorem_l3314_331460


namespace NUMINAMATH_CALUDE_profit_growth_rate_l3314_331439

/-- The average monthly growth rate of profit from March to May -/
def average_growth_rate : ℝ := 0.2

/-- The profit in March -/
def march_profit : ℝ := 5000

/-- The profit in May -/
def may_profit : ℝ := 7200

/-- The number of months between March and May -/
def months_between : ℕ := 2

theorem profit_growth_rate :
  march_profit * (1 + average_growth_rate) ^ months_between = may_profit :=
sorry

end NUMINAMATH_CALUDE_profit_growth_rate_l3314_331439


namespace NUMINAMATH_CALUDE_p_or_q_and_not_p_implies_q_l3314_331418

theorem p_or_q_and_not_p_implies_q (p q : Prop) :
  (p ∨ q) → ¬p → q := by sorry

end NUMINAMATH_CALUDE_p_or_q_and_not_p_implies_q_l3314_331418


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l3314_331468

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (4 - a/2) * x + 2 else a^x

/-- The theorem stating the range of a for which f is increasing on ℝ -/
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (4 ≤ a ∧ a < 8) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l3314_331468


namespace NUMINAMATH_CALUDE_line_no_dot_count_l3314_331445

/-- Represents the properties of an alphabet with dots and lines -/
structure Alphabet where
  total_letters : ℕ
  dot_and_line : ℕ
  dot_no_line : ℕ
  has_dot_or_line : Prop

/-- The number of letters with a straight line but no dot -/
def line_no_dot (α : Alphabet) : ℕ :=
  α.total_letters - (α.dot_and_line + α.dot_no_line)

/-- Theorem stating the number of letters with a line but no dot in the given alphabet -/
theorem line_no_dot_count (α : Alphabet) 
  (h1 : α.total_letters = 40)
  (h2 : α.dot_and_line = 11)
  (h3 : α.dot_no_line = 5)
  (h4 : α.has_dot_or_line) :
  line_no_dot α = 24 := by
  sorry

end NUMINAMATH_CALUDE_line_no_dot_count_l3314_331445


namespace NUMINAMATH_CALUDE_max_d_value_l3314_331488

def a (n : ℕ) : ℕ := n^3 + 4

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ (k : ℕ), d k = 433 ∧ ∀ (n : ℕ), d n ≤ 433 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3314_331488


namespace NUMINAMATH_CALUDE_divisibility_property_l3314_331495

theorem divisibility_property (p n q : ℕ) : 
  Prime p → 
  n > 0 → 
  q > 0 → 
  q ∣ ((n + 1)^p - n^p) → 
  p ∣ (q - 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l3314_331495


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_negative_29_l3314_331458

theorem sum_of_coefficients_equals_negative_29 :
  let p (x : ℝ) := 5 * (2 * x^8 - 9 * x^3 + 6) - 4 * (x^6 + 8 * x^3 - 3)
  (p 1) = -29 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_negative_29_l3314_331458


namespace NUMINAMATH_CALUDE_square_minus_one_divisible_by_three_l3314_331454

theorem square_minus_one_divisible_by_three (x : ℤ) (h : ¬ 3 ∣ x) : 3 ∣ (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_square_minus_one_divisible_by_three_l3314_331454


namespace NUMINAMATH_CALUDE_cosine_value_on_unit_circle_l3314_331452

theorem cosine_value_on_unit_circle (α : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 1/2 ∧ y = Real.sqrt 3 / 2) →
  Real.cos α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_on_unit_circle_l3314_331452


namespace NUMINAMATH_CALUDE_smallest_of_three_l3314_331459

theorem smallest_of_three : ∀ (a b c : ℕ), a = 10 ∧ b = 11 ∧ c = 12 → a < b ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_l3314_331459


namespace NUMINAMATH_CALUDE_asymptote_sum_l3314_331475

/-- Given a rational function y = x / (x^3 + Ax^2 + Bx + C) with integer coefficients A, B, C,
    if the graph has vertical asymptotes at x = -3, 0, 3, then A + B + C = -9 -/
theorem asymptote_sum (A B C : ℤ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 →
    ∃ y : ℝ, y = x / (x^3 + A * x^2 + B * x + C)) →
  (A + B + C = -9) := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l3314_331475


namespace NUMINAMATH_CALUDE_evening_ticket_price_l3314_331436

/-- The cost of an evening movie ticket --/
def evening_ticket_cost : ℝ := 10

/-- The cost of a large popcorn & drink combo --/
def combo_cost : ℝ := 10

/-- The discount rate for tickets during the special offer --/
def ticket_discount_rate : ℝ := 0.2

/-- The discount rate for food combos during the special offer --/
def combo_discount_rate : ℝ := 0.5

/-- The amount saved by going to the earlier movie --/
def savings : ℝ := 7

theorem evening_ticket_price :
  evening_ticket_cost = 10 ∧
  combo_cost = 10 ∧
  ticket_discount_rate = 0.2 ∧
  combo_discount_rate = 0.5 ∧
  savings = 7 →
  evening_ticket_cost + combo_cost - 
  (evening_ticket_cost * (1 - ticket_discount_rate) + combo_cost * (1 - combo_discount_rate)) = savings :=
by sorry

end NUMINAMATH_CALUDE_evening_ticket_price_l3314_331436


namespace NUMINAMATH_CALUDE_max_value_of_n_l3314_331448

theorem max_value_of_n (a b c d n : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (h4 : 1 / (a - b) + 1 / (b - c) + 1 / (c - d) ≥ n / (a - d)) :
  n ≤ 9 ∧ ∃ (a b c d : ℝ), a > b ∧ b > c ∧ c > d ∧ 
    1 / (a - b) + 1 / (b - c) + 1 / (c - d) = 9 / (a - d) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_n_l3314_331448


namespace NUMINAMATH_CALUDE_train_speeds_l3314_331442

-- Define the problem parameters
def distance : ℝ := 450
def time : ℝ := 5
def speed_difference : ℝ := 6

-- Define the theorem
theorem train_speeds (slower_speed faster_speed : ℝ) : 
  slower_speed > 0 ∧ 
  faster_speed = slower_speed + speed_difference ∧
  distance = (slower_speed + faster_speed) * time →
  slower_speed = 42 ∧ faster_speed = 48 := by
sorry

end NUMINAMATH_CALUDE_train_speeds_l3314_331442


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_f_positive_when_a_eq_two_max_value_on_interval_l3314_331461

/-- The function f(x) = e^x - ax --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

/-- Theorem for the tangent line equation when a = 2 --/
theorem tangent_line_at_zero (x y : ℝ) :
  (f 2) 0 = 1 →
  (∀ h, deriv (f 2) h = Real.exp h - 2) →
  x + y - 1 = 0 ↔ y - 1 = -(x - 0) :=
sorry

/-- Theorem for f(x) > 0 when a = 2 --/
theorem f_positive_when_a_eq_two :
  ∀ x, f 2 x > 0 :=
sorry

/-- Theorem for the maximum value of f(x) when a > 1 --/
theorem max_value_on_interval (a : ℝ) :
  a > 1 →
  ∃ x ∈ Set.Icc 0 a, ∀ y ∈ Set.Icc 0 a, f a x ≥ f a y ∧ f a x = Real.exp a - a^2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_f_positive_when_a_eq_two_max_value_on_interval_l3314_331461


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3314_331423

theorem complex_equation_solution (z : ℂ) : z = -Complex.I / 7 ↔ 3 + 2 * Complex.I * z = 4 - 5 * Complex.I * z := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3314_331423


namespace NUMINAMATH_CALUDE_prime_between_30_and_50_l3314_331463

theorem prime_between_30_and_50 (n : ℕ) :
  Prime n →
  30 < n →
  n < 50 →
  n % 6 = 1 →
  n % 5 ≠ 0 →
  n = 31 ∨ n = 37 ∨ n = 43 := by
sorry

end NUMINAMATH_CALUDE_prime_between_30_and_50_l3314_331463


namespace NUMINAMATH_CALUDE_field_trip_total_cost_l3314_331473

/-- Calculates the total cost of a field trip for multiple classes --/
def field_trip_cost (num_classes : ℕ) (students_per_class : ℕ) (adults_per_class : ℕ) 
                    (student_fee : ℚ) (adult_fee : ℚ) : ℚ :=
  let total_students := num_classes * students_per_class
  let total_adults := num_classes * adults_per_class
  (total_students : ℚ) * student_fee + (total_adults : ℚ) * adult_fee

/-- Theorem stating the total cost of the field trip --/
theorem field_trip_total_cost : 
  field_trip_cost 4 40 5 (11/2) (13/2) = 1010 := by
  sorry

#eval field_trip_cost 4 40 5 (11/2) (13/2)

end NUMINAMATH_CALUDE_field_trip_total_cost_l3314_331473


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3314_331462

theorem complex_modulus_problem (z : ℂ) (h : (z - 2*Complex.I) * (1 - Complex.I) = -2) : 
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3314_331462


namespace NUMINAMATH_CALUDE_chocolate_division_l3314_331471

theorem chocolate_division (total : ℚ) (piles : ℕ) (h1 : total = 60 / 7) (h2 : piles = 5) :
  let pile_weight := total / piles
  let received := pile_weight
  let given_back := received / 2
  received - given_back = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l3314_331471


namespace NUMINAMATH_CALUDE_right_triangle_ratio_squared_l3314_331447

/-- Given a right triangle with legs a and b, and hypotenuse c, 
    where b > a, a/b = (1/2) * (b/c), and a + b + c = 12, 
    prove that (a/b)² = 1/2 -/
theorem right_triangle_ratio_squared (a b c : ℝ) 
  (h1 : b > a)
  (h2 : a / b = (1 / 2) * (b / c))
  (h3 : a + b + c = 12)
  (h4 : c^2 = a^2 + b^2) : 
  (a / b)^2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_squared_l3314_331447


namespace NUMINAMATH_CALUDE_total_money_l3314_331474

theorem total_money (a b c : ℕ) : 
  a + c = 200 → b + c = 310 → c = 10 → a + b + c = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3314_331474


namespace NUMINAMATH_CALUDE_median_of_special_arithmetic_sequence_l3314_331412

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h1 : d ≠ 0 -- Non-zero common difference
  h2 : ∀ n, a (n + 1) = a n + d -- Arithmetic sequence property
  h3 : a 3 = 8 -- Third term is 8
  h4 : ∃ r, r ≠ 0 ∧ a 1 * r = a 3 ∧ a 3 * r = a 7 -- Geometric sequence property for a₁, a₃, a₇

/-- The median of a 9-term arithmetic sequence with specific properties is 24 -/
theorem median_of_special_arithmetic_sequence (seq : ArithmeticSequence) : 
  seq.a 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_median_of_special_arithmetic_sequence_l3314_331412


namespace NUMINAMATH_CALUDE_point_B_complex_number_l3314_331402

theorem point_B_complex_number 
  (A C : ℂ) 
  (AC BC : ℂ) 
  (h1 : A = 3 + I) 
  (h2 : AC = -2 - 4*I) 
  (h3 : BC = -4 - I) 
  (h4 : C = A + AC) :
  A + AC + BC = 5 - 2*I := by
sorry

end NUMINAMATH_CALUDE_point_B_complex_number_l3314_331402


namespace NUMINAMATH_CALUDE_ring_arrangements_l3314_331493

theorem ring_arrangements (n k f : ℕ) (hn : n = 10) (hk : k = 7) (hf : f = 5) :
  (n.choose k) * k.factorial * ((k + f - 1).choose (f - 1)) = 200160000 :=
sorry

end NUMINAMATH_CALUDE_ring_arrangements_l3314_331493


namespace NUMINAMATH_CALUDE_nine_times_eleven_and_two_fifths_l3314_331411

theorem nine_times_eleven_and_two_fifths (x : ℝ) : 
  9 * (11 + 2/5) = 102 + 3/5 := by
  sorry

end NUMINAMATH_CALUDE_nine_times_eleven_and_two_fifths_l3314_331411


namespace NUMINAMATH_CALUDE_parabola_and_circle_problem_l3314_331476

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point K
def K : ℝ × ℝ := (-1, 0)

-- Define the line l passing through K
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y - 1

-- Define the condition for points A and B on the parabola and line l
def point_on_parabola_and_line (x y m : ℝ) : Prop :=
  parabola x y ∧ line_l m x y

-- Define the symmetry condition for points A and D
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = x₂ ∧ y₁ = -y₂

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 1) * (x₂ - 1) + y₁ * y₂ = 8/9

-- Main theorem
theorem parabola_and_circle_problem
  (x₁ y₁ x₂ y₂ xd yd m : ℝ)
  (h₁ : point_on_parabola_and_line x₁ y₁ m)
  (h₂ : point_on_parabola_and_line x₂ y₂ m)
  (h₃ : symmetric_points x₁ y₁ xd yd)
  (h₄ : dot_product_condition x₁ y₁ x₂ y₂) :
  (∃ (k : ℝ), focus.1 = k * (x₂ - xd) + xd ∧ focus.2 = k * (y₂ + yd)) ∧
  (∃ (c : ℝ × ℝ) (r : ℝ), c = (1/9, 0) ∧ r = 2/3 ∧
    ∀ (x y : ℝ), (x - c.1)^2 + (y - c.2)^2 = r^2 ↔
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
        x = t * x₂ + (1-t) * K.1 ∧
        y = t * y₂ + (1-t) * K.2) ∨
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
        x = t * xd + (1-t) * K.1 ∧
        y = t * yd + (1-t) * K.2) ∨
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
        x = t * x₂ + (1-t) * xd ∧
        y = t * y₂ + (1-t) * yd)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_circle_problem_l3314_331476


namespace NUMINAMATH_CALUDE_firefly_group_size_l3314_331449

theorem firefly_group_size (butterfly_group_size : ℕ) (min_butterflies : ℕ) 
  (h1 : butterfly_group_size = 44)
  (h2 : min_butterflies = 748) :
  ∃ (firefly_group_size : ℕ),
    firefly_group_size = 
      (((min_butterflies + butterfly_group_size - 1) / butterfly_group_size) * butterfly_group_size) :=
by
  sorry

end NUMINAMATH_CALUDE_firefly_group_size_l3314_331449


namespace NUMINAMATH_CALUDE_equation_represents_empty_set_l3314_331465

theorem equation_represents_empty_set : 
  ∀ (x y : ℝ), 3 * x^2 + 5 * y^2 - 9 * x - 20 * y + 30 + 4 * x * y ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_represents_empty_set_l3314_331465


namespace NUMINAMATH_CALUDE_intersection_sum_l3314_331408

-- Define the constants and variables
variable (n c : ℝ)
variable (x y : ℝ)

-- Define the two lines
def line1 (x : ℝ) : ℝ := n * x + 5
def line2 (x : ℝ) : ℝ := 4 * x + c

-- State the theorem
theorem intersection_sum (h1 : line1 5 = 15) (h2 : line2 5 = 15) : c + n = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l3314_331408


namespace NUMINAMATH_CALUDE_f_2_value_l3314_331404

/-- An odd function -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- An even function -/
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

/-- The main theorem -/
theorem f_2_value (f g : ℝ → ℝ) (a : ℝ) :
  odd_function f →
  even_function g →
  (∀ x, f x + g x = a^x - a^(-x) + 2) →
  a > 0 →
  a ≠ 1 →
  g 2 = a →
  f 2 = 15/4 := by
  sorry


end NUMINAMATH_CALUDE_f_2_value_l3314_331404


namespace NUMINAMATH_CALUDE_election_winner_votes_l3314_331409

/-- The number of candidates in the election -/
def num_candidates : ℕ := 4

/-- The percentage of votes received by the winning candidate -/
def winner_percentage : ℚ := 468 / 1000

/-- The percentage of votes received by the second-place candidate -/
def second_percentage : ℚ := 326 / 1000

/-- The margin of victory in number of votes -/
def margin : ℕ := 752

/-- The total number of votes cast in the election -/
def total_votes : ℕ := 5296

/-- The number of votes received by the winning candidate -/
def winner_votes : ℕ := 2479

theorem election_winner_votes :
  num_candidates = 4 ∧
  winner_percentage = 468 / 1000 ∧
  second_percentage = 326 / 1000 ∧
  margin = 752 ∧
  total_votes = 5296 →
  winner_votes = 2479 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3314_331409


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l3314_331415

theorem largest_gcd_of_sum_1008 :
  ∃ (a b : ℕ+), a + b = 1008 ∧ 
  ∀ (c d : ℕ+), c + d = 1008 → Nat.gcd a b ≥ Nat.gcd c d ∧
  Nat.gcd a b = 504 :=
by sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l3314_331415


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3314_331482

/-- The line mx-y+3+m=0 passes through the point (-1, 3) for any real number m -/
theorem line_passes_through_fixed_point (m : ℝ) : m * (-1) - 3 + 3 + m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3314_331482


namespace NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l3314_331496

theorem triangle_perimeter_impossibility (a b x : ℝ) : 
  a = 20 → b = 15 → x > 0 → a + b + x ≠ 72 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l3314_331496


namespace NUMINAMATH_CALUDE_cut_square_corners_l3314_331422

/-- Given a square with side length 24 units, if each corner is cut to form an isoscelos right
    triangle resulting in a smaller rectangle, then the total area of the four removed triangles
    is 288 square units. -/
theorem cut_square_corners (r s : ℝ) : 
  (r + s)^2 + (r - s)^2 = 24^2 → r^2 + s^2 = 288 := by sorry

end NUMINAMATH_CALUDE_cut_square_corners_l3314_331422


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3314_331455

/-- Given a quadratic function f(x) = ax^2 - 2ax + c where f(2017) < f(-2016),
    prove that the set of real numbers m satisfying f(m) ≤ f(0) is [0, 2] -/
theorem quadratic_function_range (a c : ℝ) : 
  let f := λ x : ℝ => a * x^2 - 2 * a * x + c
  (f 2017 < f (-2016)) → 
  {m : ℝ | f m ≤ f 0} = Set.Icc 0 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3314_331455


namespace NUMINAMATH_CALUDE_f_geq_one_solution_set_g_max_value_l3314_331492

def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

def g (x : ℝ) : ℝ := f x - x^2 + x

theorem f_geq_one_solution_set :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

theorem g_max_value :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

end NUMINAMATH_CALUDE_f_geq_one_solution_set_g_max_value_l3314_331492


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3314_331427

theorem simplify_sqrt_expression : 
  Real.sqrt (28 - 12 * Real.sqrt 2) = 6 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3314_331427


namespace NUMINAMATH_CALUDE_lines_are_parallel_l3314_331405

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem lines_are_parallel : 
  let line1 : Line := ⟨3, 1, 1⟩
  let line2 : Line := ⟨6, 2, 1⟩
  parallel line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l3314_331405
