import Mathlib

namespace NUMINAMATH_CALUDE_congruence_problem_l3596_359658

theorem congruence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n < 23 ∧ -135 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3596_359658


namespace NUMINAMATH_CALUDE_pitchers_needed_l3596_359611

def glasses_per_pitcher : ℝ := 4.5
def total_glasses_served : ℕ := 30

theorem pitchers_needed : 
  ∃ (n : ℕ), n * glasses_per_pitcher ≥ total_glasses_served ∧ 
  ∀ (m : ℕ), m * glasses_per_pitcher ≥ total_glasses_served → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_pitchers_needed_l3596_359611


namespace NUMINAMATH_CALUDE_trip_speed_calculation_l3596_359693

theorem trip_speed_calculation (v : ℝ) : 
  v > 0 → -- Ensuring speed is positive
  (35 / v + 35 / 24 = 70 / 32) → -- Average speed equation
  v = 48 := by
sorry

end NUMINAMATH_CALUDE_trip_speed_calculation_l3596_359693


namespace NUMINAMATH_CALUDE_problem_statement_l3596_359673

theorem problem_statement (x y a : ℝ) 
  (h1 : 2^x = a) 
  (h2 : 3^y = a) 
  (h3 : 1/x + 1/y = 2) : 
  a = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3596_359673


namespace NUMINAMATH_CALUDE_courtyard_paving_l3596_359638

/-- Given a rectangular courtyard and rectangular bricks, calculate the number of bricks needed to pave the courtyard -/
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width : ℕ) 
  (h1 : courtyard_length = 25)
  (h2 : courtyard_width = 16)
  (h3 : brick_length = 20)
  (h4 : brick_width = 10) :
  (courtyard_length * 100) * (courtyard_width * 100) / (brick_length * brick_width) = 20000 := by
  sorry

#check courtyard_paving

end NUMINAMATH_CALUDE_courtyard_paving_l3596_359638


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l3596_359678

/-- Given two lines with the same non-zero y-intercept and different slopes,
    prove that the ratio of their x-intercepts is 1/2 -/
theorem x_intercept_ratio (b : ℝ) (u v : ℝ) : 
  b ≠ 0 →  -- The common y-intercept is non-zero
  0 = 8 * u + b →  -- First line equation at x-intercept
  0 = 4 * v + b →  -- Second line equation at x-intercept
  u / v = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l3596_359678


namespace NUMINAMATH_CALUDE_inequality_range_l3596_359699

theorem inequality_range : 
  (∃ (a : ℝ), ∀ (x : ℝ), |x - 1| - |x + 1| ≤ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), |x - 1| - |x + 1| ≤ b) → b ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3596_359699


namespace NUMINAMATH_CALUDE_five_point_questions_count_l3596_359661

/-- Represents a test with two types of questions -/
structure Test where
  total_points : ℕ
  total_questions : ℕ
  five_point_questions : ℕ
  ten_point_questions : ℕ

/-- Checks if a test configuration is valid -/
def is_valid_test (t : Test) : Prop :=
  t.total_questions = t.five_point_questions + t.ten_point_questions ∧
  t.total_points = 5 * t.five_point_questions + 10 * t.ten_point_questions

theorem five_point_questions_count (t : Test) 
  (h1 : t.total_points = 200)
  (h2 : t.total_questions = 30)
  (h3 : is_valid_test t) :
  t.five_point_questions = 20 := by
  sorry

end NUMINAMATH_CALUDE_five_point_questions_count_l3596_359661


namespace NUMINAMATH_CALUDE_unique_m_existence_l3596_359663

theorem unique_m_existence : ∃! m : ℤ,
  50 ≤ m ∧ m ≤ 120 ∧
  m % 7 = 0 ∧
  m % 8 = 5 ∧
  m % 5 = 4 ∧
  m = 189 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_existence_l3596_359663


namespace NUMINAMATH_CALUDE_expand_product_l3596_359665

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3596_359665


namespace NUMINAMATH_CALUDE_smallest_x_composite_l3596_359688

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def absolute_value (n : ℤ) : ℕ := Int.natAbs n

theorem smallest_x_composite : 
  (∀ x : ℤ, x < 5 → ¬ is_composite (absolute_value (5 * x^2 - 38 * x + 7))) ∧ 
  is_composite (absolute_value (5 * 5^2 - 38 * 5 + 7)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_composite_l3596_359688


namespace NUMINAMATH_CALUDE_pigeonhole_apples_l3596_359606

theorem pigeonhole_apples (n : ℕ) (m : ℕ) (h1 : n = 25) (h2 : m = 3) :
  ∃ (c : Fin m), (n / m : ℚ) ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_apples_l3596_359606


namespace NUMINAMATH_CALUDE_coefficient_of_a_l3596_359624

theorem coefficient_of_a (a b : ℝ) (h1 : a = 2) (h2 : b = 15) : 
  42 * b = 630 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_a_l3596_359624


namespace NUMINAMATH_CALUDE_calculation_proof_l3596_359691

theorem calculation_proof :
  (Real.sqrt 48 * Real.sqrt (1/2) + Real.sqrt 12 + Real.sqrt 24 = 4 * Real.sqrt 6 + 2 * Real.sqrt 3) ∧
  ((Real.sqrt 5 + 1)^2 + (Real.sqrt 13 + 3) * (Real.sqrt 13 - 3) = 10 + 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l3596_359691


namespace NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l3596_359684

theorem absolute_value_and_quadratic_equivalence :
  ∀ b c : ℝ,
  (∀ x : ℝ, |x - 3| = 4 ↔ x^2 + b*x + c = 0) →
  b = -6 ∧ c = -7 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l3596_359684


namespace NUMINAMATH_CALUDE_first_dog_bones_l3596_359615

theorem first_dog_bones (total_bones : ℕ) (total_dogs : ℕ) 
  (h_total_bones : total_bones = 12)
  (h_total_dogs : total_dogs = 5)
  (first_dog : ℕ)
  (second_dog : ℕ)
  (third_dog : ℕ)
  (fourth_dog : ℕ)
  (fifth_dog : ℕ)
  (h_second_dog : second_dog = first_dog - 1)
  (h_third_dog : third_dog = 2 * second_dog)
  (h_fourth_dog : fourth_dog = 1)
  (h_fifth_dog : fifth_dog = 2 * fourth_dog)
  (h_all_bones : first_dog + second_dog + third_dog + fourth_dog + fifth_dog = total_bones) :
  first_dog = 3 := by
sorry

end NUMINAMATH_CALUDE_first_dog_bones_l3596_359615


namespace NUMINAMATH_CALUDE_sqrt_14_bounds_l3596_359649

theorem sqrt_14_bounds : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14_bounds_l3596_359649


namespace NUMINAMATH_CALUDE_cubic_root_property_l3596_359674

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0,
    if 4 and -3 are roots of the equation, then (b+c)/a = -13 -/
theorem cubic_root_property (a b c d : ℝ) (ha : a ≠ 0) :
  (a * (4 : ℝ)^3 + b * (4 : ℝ)^2 + c * (4 : ℝ) + d = 0) →
  (a * (-3 : ℝ)^3 + b * (-3 : ℝ)^2 + c * (-3 : ℝ) + d = 0) →
  (b + c) / a = -13 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_property_l3596_359674


namespace NUMINAMATH_CALUDE_parallelogram_area_specific_l3596_359634

/-- The area of a parallelogram with base b and height h. -/
def parallelogram_area (b h : ℝ) : ℝ := b * h

/-- Theorem: The area of a parallelogram with a base of 15 meters and an altitude
    that is twice the base is 450 square meters. -/
theorem parallelogram_area_specific : 
  let base : ℝ := 15
  let height : ℝ := 2 * base
  parallelogram_area base height = 450 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_specific_l3596_359634


namespace NUMINAMATH_CALUDE_potato_bundle_size_l3596_359682

theorem potato_bundle_size (total_potatoes : ℕ) (potato_bundle_price : ℚ)
  (total_carrots : ℕ) (carrots_per_bundle : ℕ) (carrot_bundle_price : ℚ)
  (total_revenue : ℚ) :
  total_potatoes = 250 →
  potato_bundle_price = 19/10 →
  total_carrots = 320 →
  carrots_per_bundle = 20 →
  carrot_bundle_price = 2 →
  total_revenue = 51 →
  ∃ (potatoes_per_bundle : ℕ),
    potatoes_per_bundle = 25 ∧
    (potato_bundle_price * (total_potatoes / potatoes_per_bundle : ℚ) +
     carrot_bundle_price * (total_carrots / carrots_per_bundle : ℚ) = total_revenue) :=
by sorry

end NUMINAMATH_CALUDE_potato_bundle_size_l3596_359682


namespace NUMINAMATH_CALUDE_chosen_number_calculation_l3596_359627

theorem chosen_number_calculation : 
  let chosen_number : ℕ := 208
  let divided_result : ℚ := chosen_number / 2
  let final_result : ℚ := divided_result - 100
  final_result = 4 := by
sorry

end NUMINAMATH_CALUDE_chosen_number_calculation_l3596_359627


namespace NUMINAMATH_CALUDE_bulk_bag_contains_40_oz_l3596_359613

/-- Calculates the number of ounces in a bulk bag of mixed nuts -/
def bulkBagOunces (originalCost : ℚ) (couponValue : ℚ) (costPerServing : ℚ) : ℚ :=
  (originalCost - couponValue) / costPerServing

/-- Theorem stating that the bulk bag contains 40 ounces of mixed nuts -/
theorem bulk_bag_contains_40_oz :
  bulkBagOunces 25 5 (1/2) = 40 := by
  sorry

end NUMINAMATH_CALUDE_bulk_bag_contains_40_oz_l3596_359613


namespace NUMINAMATH_CALUDE_newspaper_collection_target_l3596_359614

structure Section where
  name : String
  first_week_collection : ℝ

def second_week_increase : ℝ := 0.10
def third_week_increase : ℝ := 0.30

def sections : List Section := [
  ⟨"A", 260⟩,
  ⟨"B", 290⟩,
  ⟨"C", 250⟩,
  ⟨"D", 270⟩,
  ⟨"E", 300⟩,
  ⟨"F", 310⟩,
  ⟨"G", 280⟩,
  ⟨"H", 265⟩
]

def first_week_total : ℝ := (sections.map (·.first_week_collection)).sum

def second_week_total : ℝ :=
  (sections.map (fun s => s.first_week_collection * (1 + second_week_increase))).sum

def third_week_total : ℝ :=
  (sections.map (fun s => s.first_week_collection * (1 + second_week_increase) * (1 + third_week_increase))).sum

def target : ℝ := first_week_total + second_week_total + third_week_total

theorem newspaper_collection_target :
  target = 7854.25 := by sorry

end NUMINAMATH_CALUDE_newspaper_collection_target_l3596_359614


namespace NUMINAMATH_CALUDE_binomial_8_5_l3596_359628

theorem binomial_8_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_5_l3596_359628


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l3596_359636

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (2 * x - 7) + Real.sqrt (5 - x)) ↔ 3.5 ≤ x ∧ x ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l3596_359636


namespace NUMINAMATH_CALUDE_cube_root_64_minus_sqrt_8_squared_l3596_359632

theorem cube_root_64_minus_sqrt_8_squared : 
  (64 ^ (1/3) - Real.sqrt 8) ^ 2 = 24 - 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_64_minus_sqrt_8_squared_l3596_359632


namespace NUMINAMATH_CALUDE_range_of_m_l3596_359660

-- Define the equation
def equation (m x : ℝ) : Prop := (m - 1) / (x + 1) = 1

-- Define the theorem
theorem range_of_m (m x : ℝ) : 
  equation m x ∧ x < 0 → m < 2 ∧ m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3596_359660


namespace NUMINAMATH_CALUDE_exists_equivalent_expression_l3596_359642

/-- Define a type for the unknown operations -/
inductive UnknownOp
| add
| sub

/-- Define a function that applies the unknown operation -/
def applyOp (op : UnknownOp) (x y : ℝ) : ℝ :=
  match op with
  | UnknownOp.add => x + y
  | UnknownOp.sub => x - y

/-- Define a function that represents the reversed subtraction -/
def revSub (x y : ℝ) : ℝ := y - x

theorem exists_equivalent_expression :
  ∃ (op1 op2 : UnknownOp) (f1 f2 : ℝ → ℝ → ℝ),
    (f1 = applyOp op1 ∧ f2 = applyOp op2) ∨
    (f1 = applyOp op1 ∧ f2 = revSub) ∨
    (f1 = revSub ∧ f2 = applyOp op2) →
    ∀ (a b : ℝ), ∃ (expr : ℝ), expr = 20 * a - 18 * b :=
by sorry

end NUMINAMATH_CALUDE_exists_equivalent_expression_l3596_359642


namespace NUMINAMATH_CALUDE_water_transfer_l3596_359662

theorem water_transfer (a b x : ℝ) : 
  a = 13.2 ∧ 
  (13.2 - x = (1/3) * (b + x)) ∧ 
  (b - x = (1/2) * (13.2 + x)) → 
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_water_transfer_l3596_359662


namespace NUMINAMATH_CALUDE_jersey_t_shirt_price_difference_l3596_359653

/-- The price difference between a jersey and a t-shirt -/
def price_difference (jersey_profit t_shirt_profit : ℕ) : ℕ :=
  jersey_profit - t_shirt_profit

/-- Theorem stating that the price difference between a jersey and a t-shirt is $90 -/
theorem jersey_t_shirt_price_difference :
  price_difference 115 25 = 90 := by
  sorry

end NUMINAMATH_CALUDE_jersey_t_shirt_price_difference_l3596_359653


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3596_359630

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement --/
theorem interest_rate_calculation (principal time : ℕ) (final_amount : ℝ) 
  (h1 : principal = 6000)
  (h2 : time = 2)
  (h3 : final_amount = 7260) :
  ∃ (rate : ℝ), compound_interest principal rate time = final_amount ∧ rate = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3596_359630


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l3596_359650

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n = 990 ∧ 
  (∀ m : ℕ, m < 1000 → m % 5 = 0 → m % 6 = 0 → m ≤ n) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l3596_359650


namespace NUMINAMATH_CALUDE_magnitude_of_z_l3596_359696

/-- Given a complex number z satisfying (z-i)i = 2+i, prove that |z| = √5 -/
theorem magnitude_of_z (z : ℂ) (h : (z - Complex.I) * Complex.I = 2 + Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l3596_359696


namespace NUMINAMATH_CALUDE_intersection_and_complement_union_l3596_359618

-- Define the universe U as the real numbers
def U := ℝ

-- Define set M
def M : Set ℝ := {x | x ≥ 1}

-- Define set N
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem intersection_and_complement_union :
  (M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 5}) ∧
  ((Mᶜ ∪ Nᶜ) = {x : ℝ | x < 1 ∨ x ≥ 5}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_complement_union_l3596_359618


namespace NUMINAMATH_CALUDE_experiment_sequences_l3596_359623

/-- Represents the number of procedures in the experiment -/
def num_procedures : ℕ := 5

/-- Represents the condition that procedure A can only be first or last -/
def a_first_or_last : ℕ := 2

/-- Represents the number of ways to arrange C and D adjacently -/
def cd_adjacent : ℕ := 2

/-- Represents the number of ways to arrange the remaining procedures -/
def remaining_arrangements : ℕ := 3

/-- The total number of possible sequences for the experiment -/
def total_sequences : ℕ := a_first_or_last * cd_adjacent * remaining_arrangements.factorial

theorem experiment_sequences :
  total_sequences = 24 := by sorry

end NUMINAMATH_CALUDE_experiment_sequences_l3596_359623


namespace NUMINAMATH_CALUDE_amy_age_2005_l3596_359610

/-- Amy's age at the end of 2000 -/
def amy_age_2000 : ℕ := sorry

/-- Amy's grandfather's age at the end of 2000 -/
def grandfather_age_2000 : ℕ := sorry

/-- The year 2000 -/
def year_2000 : ℕ := 2000

/-- The sum of Amy's and her grandfather's birth years -/
def birth_years_sum : ℕ := 3900

theorem amy_age_2005 : 
  grandfather_age_2000 = 3 * amy_age_2000 →
  year_2000 - amy_age_2000 + (year_2000 - grandfather_age_2000) = birth_years_sum →
  amy_age_2000 + 5 = 30 := by sorry

end NUMINAMATH_CALUDE_amy_age_2005_l3596_359610


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3596_359617

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 + I) / (1 + 2*I) ∧ z.re > 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3596_359617


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3596_359668

theorem perfect_square_condition (n : ℕ) : 
  (∃ m : ℕ, n^4 + 4*n^3 + 5*n^2 + 6*n = m^2) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3596_359668


namespace NUMINAMATH_CALUDE_hot_dog_eating_contest_l3596_359683

theorem hot_dog_eating_contest (first_competitor second_competitor third_competitor : ℕ) :
  first_competitor = 12 →
  third_competitor = 18 →
  third_competitor = (3 * second_competitor) / 4 →
  second_competitor / first_competitor = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_eating_contest_l3596_359683


namespace NUMINAMATH_CALUDE_square_minus_two_x_plus_2023_l3596_359681

theorem square_minus_two_x_plus_2023 :
  let x : ℝ := 1 + Real.sqrt 3
  x^2 - 2*x + 2023 = 2025 := by sorry

end NUMINAMATH_CALUDE_square_minus_two_x_plus_2023_l3596_359681


namespace NUMINAMATH_CALUDE_investment_interest_rate_l3596_359655

theorem investment_interest_rate 
  (principal : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 900) 
  (h2 : rate1 = 0.04) 
  (h3 : time = 7) 
  (h4 : principal * rate2 * time - principal * rate1 * time = interest_difference) 
  (h5 : interest_difference = 31.50) : 
  rate2 = 0.045 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l3596_359655


namespace NUMINAMATH_CALUDE_temperature_problem_l3596_359600

/-- Given the average temperatures for two sets of three consecutive days and the temperature of the last day, prove the temperature of the first day. -/
theorem temperature_problem (T W Th F : ℝ) 
  (h1 : (T + W + Th) / 3 = 42)
  (h2 : (W + Th + F) / 3 = 44)
  (h3 : F = 43) :
  T = 37 := by
  sorry

end NUMINAMATH_CALUDE_temperature_problem_l3596_359600


namespace NUMINAMATH_CALUDE_fifth_seat_is_37_l3596_359622

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  selectedSeats : Finset ℕ

/-- The seat number of the fifth selected student in the systematic sampling. -/
def fifthSelectedSeat (sampling : SystematicSampling) : ℕ :=
  37

/-- Theorem stating that given the conditions, the fifth selected seat is 37. -/
theorem fifth_seat_is_37 (sampling : SystematicSampling) 
  (h1 : sampling.totalStudents = 55)
  (h2 : sampling.sampleSize = 5)
  (h3 : sampling.selectedSeats = {4, 15, 26, 48}) :
  fifthSelectedSeat sampling = 37 := by
  sorry

#check fifth_seat_is_37

end NUMINAMATH_CALUDE_fifth_seat_is_37_l3596_359622


namespace NUMINAMATH_CALUDE_senior_tickets_first_day_l3596_359646

/- Define the variables -/
def student_ticket_price : ℕ := 9
def first_day_student_tickets : ℕ := 3
def first_day_total : ℕ := 79
def second_day_senior_tickets : ℕ := 12
def second_day_student_tickets : ℕ := 10
def second_day_total : ℕ := 246

/- Theorem to prove -/
theorem senior_tickets_first_day :
  ∃ (senior_ticket_price : ℕ) (first_day_senior_tickets : ℕ),
    senior_ticket_price * second_day_senior_tickets + 
    student_ticket_price * second_day_student_tickets = second_day_total ∧
    senior_ticket_price * first_day_senior_tickets + 
    student_ticket_price * first_day_student_tickets = first_day_total ∧
    first_day_senior_tickets = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_senior_tickets_first_day_l3596_359646


namespace NUMINAMATH_CALUDE_tennis_preference_theorem_l3596_359694

/-- Represents the percentage of students who prefer tennis -/
def tennis_preference (total : ℕ) (prefer : ℕ) : ℚ :=
  prefer / total

/-- Represents the total number of students who prefer tennis -/
def total_tennis_preference (north_total : ℕ) (north_prefer : ℕ) (south_total : ℕ) (south_prefer : ℕ) : ℕ :=
  north_prefer + south_prefer

/-- Represents the combined percentage of students who prefer tennis -/
def combined_tennis_preference (north_total : ℕ) (north_prefer : ℕ) (south_total : ℕ) (south_prefer : ℕ) : ℚ :=
  tennis_preference (north_total + south_total) (total_tennis_preference north_total north_prefer south_total south_prefer)

theorem tennis_preference_theorem (north_total south_total : ℕ) (north_prefer south_prefer : ℕ) :
  north_total = 1800 →
  south_total = 2700 →
  tennis_preference north_total north_prefer = 30 / 100 →
  tennis_preference south_total south_prefer = 25 / 100 →
  combined_tennis_preference north_total north_prefer south_total south_prefer = 27 / 100 :=
by sorry

end NUMINAMATH_CALUDE_tennis_preference_theorem_l3596_359694


namespace NUMINAMATH_CALUDE_pizza_coverage_l3596_359616

theorem pizza_coverage (pizza_diameter : ℝ) (pepperoni_diameter : ℝ) (num_pepperoni : ℕ) : 
  pizza_diameter = 2 * pepperoni_diameter →
  num_pepperoni = 32 →
  (num_pepperoni * (pepperoni_diameter / 2)^2 * π) / ((pizza_diameter / 2)^2 * π) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_coverage_l3596_359616


namespace NUMINAMATH_CALUDE_polynomial_inequality_l3596_359652

theorem polynomial_inequality (x : ℝ) : 
  x^4 - 4*x^3 + 8*x^2 - 8*x ≤ 96 → -2 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l3596_359652


namespace NUMINAMATH_CALUDE_strawberry_milk_probability_l3596_359675

theorem strawberry_milk_probability :
  let n : ℕ := 7  -- number of days
  let k : ℕ := 5  -- number of successful days
  let p : ℚ := 3/5  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 20412/78125 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_milk_probability_l3596_359675


namespace NUMINAMATH_CALUDE_gcd_of_225_and_135_l3596_359605

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_225_and_135_l3596_359605


namespace NUMINAMATH_CALUDE_parallelogram_in_grid_l3596_359643

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a vector between two points in the grid -/
structure GridVector where
  dx : ℤ
  dy : ℤ

/-- The theorem to be proved -/
theorem parallelogram_in_grid (n : ℕ) (h : n ≥ 2) :
  ∀ (chosen : Finset GridPoint),
    chosen.card = 2 * n →
    ∃ (a b c d : GridPoint),
      a ∈ chosen ∧ b ∈ chosen ∧ c ∈ chosen ∧ d ∈ chosen ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      (GridVector.mk (b.x - a.x) (b.y - a.y) =
       GridVector.mk (d.x - c.x) (d.y - c.y)) ∧
      (GridVector.mk (c.x - a.x) (c.y - a.y) =
       GridVector.mk (d.x - b.x) (d.y - b.y)) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_in_grid_l3596_359643


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3596_359656

/-- 
Given a hyperbola with equation x^2 + my^2 = 1, where m is a real number,
if its conjugate axis is twice the length of its transverse axis,
then its eccentricity e is equal to √5.
-/
theorem hyperbola_eccentricity (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b = 2*a) →
  (∃ e : ℝ, e = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3596_359656


namespace NUMINAMATH_CALUDE_equation_solution_l3596_359647

theorem equation_solution (x y : ℝ) : 
  (x^4 + 1) * (y^4 + 1) = 4 * x^2 * y^2 ↔ 
  ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3596_359647


namespace NUMINAMATH_CALUDE_xyz_sum_and_inequality_l3596_359680

theorem xyz_sum_and_inequality (x y z : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_not_all_equal : ¬(x = y ∧ y = z))
  (h_equation : x^3 + y^3 + z^3 - 3*x*y*z - 3*(x^2 + y^2 + z^2 - x*y - y*z - z*x) = 0) :
  (x + y + z = 3) ∧ (x^2*(1 + y) + y^2*(1 + z) + z^2*(1 + x) > 6) := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_and_inequality_l3596_359680


namespace NUMINAMATH_CALUDE_units_digit_base_8_l3596_359609

theorem units_digit_base_8 (n₁ n₂ : ℕ) (h₁ : n₁ = 198) (h₂ : n₂ = 53) :
  (((n₁ - 3) * (n₂ + 7)) % 8) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_base_8_l3596_359609


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_not_square_l3596_359664

theorem product_of_five_consecutive_integers_not_square (n : ℤ) : 
  ∃ (m : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ≠ m ^ 2 := by
sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_not_square_l3596_359664


namespace NUMINAMATH_CALUDE_ball_drawing_game_l3596_359601

/-- The probability that the last ball is white in a ball-drawing game -/
def last_ball_white_probability (p q : ℕ) : ℚ :=
  if p % 2 = 0 then 0 else 1

/-- The ball-drawing game process -/
theorem ball_drawing_game (p q : ℕ) :
  let initial_total := p + q
  let final_total := 1
  let draw_count := initial_total - final_total
  ∀ (draw_process : ℕ → ℕ × ℕ),
    (∀ i < draw_count, 
      let (w, b) := draw_process i
      let (w', b') := draw_process (i + 1)
      ((w = w' ∧ b = b' + 1) ∨ (w = w' - 1 ∧ b = b' + 1) ∨ (w = w' - 2 ∧ b = b' + 1))) →
    (draw_process 0 = (p, q)) →
    (draw_process draw_count).fst + (draw_process draw_count).snd = final_total →
    (last_ball_white_probability p q = if (draw_process draw_count).fst = 1 then 1 else 0) :=
sorry

end NUMINAMATH_CALUDE_ball_drawing_game_l3596_359601


namespace NUMINAMATH_CALUDE_sales_tax_rate_is_twenty_percent_l3596_359687

/-- Calculates the sales tax rate given the cost of items and total amount spent --/
def calculate_sales_tax_rate (milk_cost banana_cost total_spent : ℚ) : ℚ :=
  let items_cost := milk_cost + banana_cost
  let tax_amount := total_spent - items_cost
  (tax_amount / items_cost) * 100

theorem sales_tax_rate_is_twenty_percent : 
  calculate_sales_tax_rate 3 2 6 = 20 := by sorry

end NUMINAMATH_CALUDE_sales_tax_rate_is_twenty_percent_l3596_359687


namespace NUMINAMATH_CALUDE_ellipse_properties_l3596_359608

/-- The ellipse C: x^2/4 + y^2/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line l: y = kx, where k ≠ 0 -/
def line_l (k x y : ℝ) : Prop := y = k * x ∧ k ≠ 0

/-- M and N are intersection points of line l and ellipse C -/
def intersection_points (M N : ℝ × ℝ) (k : ℝ) : Prop :=
  ellipse_C M.1 M.2 ∧ ellipse_C N.1 N.2 ∧
  line_l k M.1 M.2 ∧ line_l k N.1 N.2

/-- F₁ and F₂ are the foci of the ellipse C -/
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-1, 0) ∧ F₂ = (1, 0)

/-- B is the top vertex of the ellipse C -/
def top_vertex (B : ℝ × ℝ) : Prop :=
  B = (0, Real.sqrt 3)

/-- The perimeter of quadrilateral MF₁NF₂ is 8 -/
def perimeter_is_8 (M N F₁ F₂ : ℝ × ℝ) : Prop :=
  dist M F₁ + dist F₁ N + dist N F₂ + dist F₂ M = 8

/-- The product of the slopes of lines BM and BN is -3/4 -/
def slope_product (M N B : ℝ × ℝ) : Prop :=
  ((M.2 - B.2) / (M.1 - B.1)) * ((N.2 - B.2) / (N.1 - B.1)) = -3/4

theorem ellipse_properties (k : ℝ) (M N F₁ F₂ B : ℝ × ℝ) :
  intersection_points M N k →
  foci F₁ F₂ →
  top_vertex B →
  perimeter_is_8 M N F₁ F₂ ∧ slope_product M N B :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3596_359608


namespace NUMINAMATH_CALUDE_perfect_square_4p_minus_3_l3596_359629

theorem perfect_square_4p_minus_3 (n p : ℕ) (hn : n > 1) (hp : p > 1) (p_prime : Nat.Prime p)
  (n_divides_p_minus_1 : n ∣ (p - 1)) (p_divides_n_cube_minus_1 : p ∣ (n^3 - 1)) :
  ∃ k : ℤ, (4 : ℤ) * p - 3 = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_4p_minus_3_l3596_359629


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l3596_359640

theorem isosceles_triangle_quadratic_roots (k : ℝ) : 
  (∃ (a b : ℝ), 
    (a^2 - 6*a + k = 0) ∧ 
    (b^2 - 6*b + k = 0) ∧ 
    (a = b ∨ a = 2 ∨ b = 2) ∧
    (a + b > 2) ∧ (a + 2 > b) ∧ (b + 2 > a)) → k = 9 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l3596_359640


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3596_359654

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x ∈ Set.Ioo 0 3 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ x ∉ Set.Ioo 0 3) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3596_359654


namespace NUMINAMATH_CALUDE_total_candies_l3596_359669

theorem total_candies (linda_candies chloe_candies olivia_candies : ℕ)
  (h1 : linda_candies = 34)
  (h2 : chloe_candies = 28)
  (h3 : olivia_candies = 43) :
  linda_candies + chloe_candies + olivia_candies = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l3596_359669


namespace NUMINAMATH_CALUDE_certain_number_multiplied_l3596_359666

theorem certain_number_multiplied (x : ℝ) : x - 7 = 9 → 3 * x = 48 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_multiplied_l3596_359666


namespace NUMINAMATH_CALUDE_highway_length_l3596_359637

theorem highway_length (speed1 speed2 time : ℝ) 
  (h1 : speed1 = 14)
  (h2 : speed2 = 16)
  (h3 : time = 1.5)
  : speed1 * time + speed2 * time = 45 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_l3596_359637


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_time_to_water_surface_l3596_359635

-- Define the height function
def h (t : ℝ) : ℝ := -4.8 * t^2 + 8 * t + 10

-- Theorem for instantaneous velocity at t = 2
theorem instantaneous_velocity_at_2 : 
  (deriv h) 2 = -11.2 := by sorry

-- Theorem for time when athlete reaches water surface
theorem time_to_water_surface : 
  ∃ t : ℝ, t = 2.5 ∧ h t = 0 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_time_to_water_surface_l3596_359635


namespace NUMINAMATH_CALUDE_sqrt_9025_squared_l3596_359695

theorem sqrt_9025_squared : (Real.sqrt 9025)^2 = 9025 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_9025_squared_l3596_359695


namespace NUMINAMATH_CALUDE_gold_per_hour_l3596_359685

/-- Calculates the amount of gold coins found per hour during a scuba diving expedition. -/
theorem gold_per_hour (hours : ℕ) (chest_coins : ℕ) (num_bags : ℕ) : 
  hours > 0 → 
  chest_coins > 0 → 
  num_bags > 0 → 
  (chest_coins + num_bags * (chest_coins / 2)) / hours = 25 :=
by
  sorry

#check gold_per_hour 8 100 2

end NUMINAMATH_CALUDE_gold_per_hour_l3596_359685


namespace NUMINAMATH_CALUDE_geometry_biology_overlap_l3596_359633

theorem geometry_biology_overlap (total : ℕ) (geometry : ℕ) (biology : ℕ) 
  (h1 : total = 350) (h2 : geometry = 210) (h3 : biology = 175) :
  let max_overlap := min geometry biology
  let min_overlap := max 0 (geometry + biology - total)
  max_overlap - min_overlap = 140 := by
sorry

end NUMINAMATH_CALUDE_geometry_biology_overlap_l3596_359633


namespace NUMINAMATH_CALUDE_inequality_implies_lower_bound_l3596_359690

theorem inequality_implies_lower_bound (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, 4^x - 2^(x+1) - a ≤ 0) → a ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_lower_bound_l3596_359690


namespace NUMINAMATH_CALUDE_john_driving_time_l3596_359672

theorem john_driving_time (speed : ℝ) (time_before_lunch : ℝ) (total_distance : ℝ) :
  speed = 55 →
  time_before_lunch = 2 →
  total_distance = 275 →
  (total_distance - speed * time_before_lunch) / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_driving_time_l3596_359672


namespace NUMINAMATH_CALUDE_complex_equidistant_point_l3596_359603

theorem complex_equidistant_point : ∃! (z : ℂ), Complex.abs (z - 2) = Complex.abs (z + 4) ∧ Complex.abs (z - 2) = Complex.abs (z - 3*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equidistant_point_l3596_359603


namespace NUMINAMATH_CALUDE_exactly_three_prime_values_l3596_359602

def polynomial (n : ℕ+) : ℤ := (n.val : ℤ)^3 - 6*(n.val : ℤ)^2 + 17*(n.val : ℤ) - 19

def is_prime_for_n (n : ℕ+) : Prop := Nat.Prime (Int.natAbs (polynomial n))

theorem exactly_three_prime_values :
  ∃! (s : Finset ℕ+), s.card = 3 ∧ ∀ n, n ∈ s ↔ is_prime_for_n n :=
sorry

end NUMINAMATH_CALUDE_exactly_three_prime_values_l3596_359602


namespace NUMINAMATH_CALUDE_volunteer_assignment_count_l3596_359619

/-- The number of ways to assign volunteers to areas --/
def assignmentCount (volunteers : ℕ) (areas : ℕ) : ℕ :=
  areas^volunteers - areas * (areas - 1)^volunteers + areas * (areas - 2)^volunteers

/-- Theorem stating that the number of ways to assign 5 volunteers to 3 areas,
    with at least one volunteer in each area, is equal to 150 --/
theorem volunteer_assignment_count :
  assignmentCount 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_assignment_count_l3596_359619


namespace NUMINAMATH_CALUDE_max_perfect_squares_among_products_l3596_359631

theorem max_perfect_squares_among_products (a b : ℕ) (h : a ≠ b) : 
  let products := {a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)}
  (∃ (s : Finset ℕ), s ⊆ products ∧ (∀ x ∈ s, ∃ y : ℕ, x = y * y) ∧ s.card = 2) ∧
  (∀ (s : Finset ℕ), s ⊆ products → (∀ x ∈ s, ∃ y : ℕ, x = y * y) → s.card ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_perfect_squares_among_products_l3596_359631


namespace NUMINAMATH_CALUDE_lcm_problem_l3596_359670

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 84) (h2 : b = 4 * a) (h3 : b = 84) :
  Nat.lcm a b = 21 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3596_359670


namespace NUMINAMATH_CALUDE_circulation_within_period_l3596_359639

/-- Represents the average yearly circulation for magazine P from 1962 to 1970 -/
def average_circulation : ℝ := sorry

/-- Represents the circulation of magazine P in 1961 -/
def circulation_1961 : ℝ := sorry

/-- Represents the year when the circulation was 4 times the average -/
def special_year : ℕ := sorry

/-- The circulation in the special year -/
def special_circulation : ℝ := 4 * average_circulation

/-- The total circulation from 1961 to 1970 -/
def total_circulation : ℝ := circulation_1961 + 9 * average_circulation

/-- The ratio of special circulation to total circulation -/
def circulation_ratio : ℝ := 0.2857142857142857

theorem circulation_within_period : 
  (special_circulation / total_circulation = circulation_ratio) →
  (circulation_1961 = 5 * average_circulation) →
  (special_year ≥ 1961 ∧ special_year ≤ 1970) :=
by sorry

end NUMINAMATH_CALUDE_circulation_within_period_l3596_359639


namespace NUMINAMATH_CALUDE_product_of_square_roots_l3596_359692

theorem product_of_square_roots (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (7 * q^3) * Real.sqrt (8 * q^5) = 210 * q^4 * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l3596_359692


namespace NUMINAMATH_CALUDE_negation_of_no_slow_learners_attend_l3596_359677

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (slow_learner : U → Prop)
variable (attends_school : U → Prop)

-- State the theorem
theorem negation_of_no_slow_learners_attend (h : ¬∃ x, slow_learner x ∧ attends_school x) :
  ∃ x, slow_learner x ∧ attends_school x ↔ ¬(¬∃ x, slow_learner x ∧ attends_school x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_no_slow_learners_attend_l3596_359677


namespace NUMINAMATH_CALUDE_nine_integer_lengths_l3596_359604

/-- Represents a right triangle with integer leg lengths -/
structure RightTriangle where
  leg1 : ℕ
  leg2 : ℕ

/-- Counts the number of distinct integer lengths of line segments
    that can be drawn from a vertex to the opposite side in a right triangle -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem stating that for a right triangle with legs 24 and 25,
    there are exactly 9 distinct integer lengths of line segments
    that can be drawn from a vertex to the hypotenuse -/
theorem nine_integer_lengths :
  let t : RightTriangle := { leg1 := 24, leg2 := 25 }
  countIntegerLengths t = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_nine_integer_lengths_l3596_359604


namespace NUMINAMATH_CALUDE_quadratic_properties_l3596_359626

-- Define a quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties :
  ∀ (a b c : ℝ),
  (∃ (x_min : ℝ), ∀ (x : ℝ), quadratic a b c x ≥ quadratic a b c x_min ∧ quadratic a b c x_min = 1) →
  quadratic a b c 0 = 3 →
  quadratic a b c 2 = 3 →
  (a = 2 ∧ b = -4 ∧ c = 3) ∧
  (∀ (a_range : ℝ), (∃ (x y : ℝ), 2 * a_range ≤ x ∧ x < y ∧ y ≤ a_range + 1 ∧
    (quadratic 2 (-4) 3 x < quadratic 2 (-4) 3 y ∧ quadratic 2 (-4) 3 y > quadratic 2 (-4) 3 (a_range + 1))) ↔
    (0 < a_range ∧ a_range < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3596_359626


namespace NUMINAMATH_CALUDE_square_sum_difference_l3596_359671

theorem square_sum_difference : 102 * 102 + 98 * 98 = 800 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_l3596_359671


namespace NUMINAMATH_CALUDE_hyperbola_unique_solution_l3596_359657

/-- The hyperbola equation -/
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (2 * m^2) - y^2 / (3 * m) = 1

/-- The focal length of the hyperbola -/
def focal_length : ℝ := 6

/-- Theorem stating that 3/2 is the only positive real solution for m -/
theorem hyperbola_unique_solution :
  ∃! m : ℝ, m > 0 ∧ 
  (∀ x y : ℝ, hyperbola_equation x y m) ∧
  (∃ c : ℝ, c^2 = 2 * m^2 + 3 * m ∧ c = focal_length / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_unique_solution_l3596_359657


namespace NUMINAMATH_CALUDE_sequence_general_term_l3596_359689

theorem sequence_general_term (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, (2 * n - 1 : ℝ) * a (n + 1) = (2 * n + 1 : ℝ) * a n) →
  ∀ n : ℕ, a n = 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3596_359689


namespace NUMINAMATH_CALUDE_product_of_reals_l3596_359644

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 8) (cube_sum_eq : a^3 + b^3 = 152) : a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reals_l3596_359644


namespace NUMINAMATH_CALUDE_value_of_x_l3596_359676

theorem value_of_x : ∃ x : ℝ, (0.25 * x = 0.15 * 1500 - 20) ∧ x = 820 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3596_359676


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_divisibility_l3596_359645

theorem consecutive_numbers_product_divisibility (n : ℕ) (hn : n > 1) :
  ∃ k : ℕ, ∀ p : ℕ,
    Prime p →
    (∀ i : ℕ, i < n → (p ∣ (k + i + 1))) ↔ p ≤ 2 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_divisibility_l3596_359645


namespace NUMINAMATH_CALUDE_shortest_distance_to_mount_fuji_l3596_359679

theorem shortest_distance_to_mount_fuji (a b c h : ℝ) : 
  a = 60 → b = 45 → c^2 = a^2 + b^2 → h * c = a * b → h = 36 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_to_mount_fuji_l3596_359679


namespace NUMINAMATH_CALUDE_voltage_meter_max_value_l3596_359651

/-- Represents a voltage meter with a maximum recordable value -/
structure VoltageMeter where
  max_value : ℝ
  records_nonnegative : 0 ≤ max_value

/-- Theorem: Given the conditions, the maximum recordable value is 14 volts -/
theorem voltage_meter_max_value (meter : VoltageMeter) 
  (avg_recording : ℝ) 
  (min_recording : ℝ) 
  (h1 : avg_recording = 6)
  (h2 : min_recording = 2)
  (h3 : ∃ (a b c : ℝ), 
    0 ≤ a ∧ a ≤ meter.max_value ∧
    0 ≤ b ∧ b ≤ meter.max_value ∧
    0 ≤ c ∧ c ≤ meter.max_value ∧
    (a + b + c) / 3 = avg_recording ∧
    min_recording ≤ a ∧ min_recording ≤ b ∧ min_recording ≤ c) :
  meter.max_value = 14 := by
sorry

end NUMINAMATH_CALUDE_voltage_meter_max_value_l3596_359651


namespace NUMINAMATH_CALUDE_sqrt_x_plus_2y_is_plus_minus_one_l3596_359697

theorem sqrt_x_plus_2y_is_plus_minus_one (x y : ℝ) 
  (h : Real.sqrt (x - 2) + abs (2 * y + 1) = 0) : 
  Real.sqrt (x + 2 * y) = 1 ∨ Real.sqrt (x + 2 * y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_2y_is_plus_minus_one_l3596_359697


namespace NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_three_l3596_359641

/-- The number of four-digit odd numbers -/
def A : ℕ := 4500

/-- The number of four-digit multiples of 3 -/
def B : ℕ := 3000

/-- The sum of four-digit odd numbers and four-digit multiples of 3 is 7500 -/
theorem sum_of_odd_and_multiples_of_three : A + B = 7500 := by sorry

end NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_three_l3596_359641


namespace NUMINAMATH_CALUDE_S_eq_EvenPositive_l3596_359621

/-- The set of all positive integers that can be written in the form ([x, y] + [y, z]) / [x, z] -/
def S : Set ℕ+ :=
  {n | ∃ (x y z : ℕ+), n = (Nat.lcm x y + Nat.lcm y z) / Nat.lcm x z}

/-- The set of all even positive integers -/
def EvenPositive : Set ℕ+ :=
  {n | ∃ (k : ℕ+), n = 2 * k}

/-- Theorem stating that S is equal to the set of all even positive integers -/
theorem S_eq_EvenPositive : S = EvenPositive := by
  sorry

end NUMINAMATH_CALUDE_S_eq_EvenPositive_l3596_359621


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_product_less_500_l3596_359686

/-- The greatest possible sum of two consecutive integers whose product is less than 500 is 43 -/
theorem greatest_sum_consecutive_integers_product_less_500 : 
  (∃ n : ℤ, n * (n + 1) < 500 ∧ 
    ∀ m : ℤ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) ∧
  (∀ n : ℤ, n * (n + 1) < 500 → n + (n + 1) ≤ 43) :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_product_less_500_l3596_359686


namespace NUMINAMATH_CALUDE_marco_painting_fraction_l3596_359698

theorem marco_painting_fraction (marco_rate carla_rate : ℚ) : 
  marco_rate = 1 / 60 →
  marco_rate + carla_rate = 1 / 40 →
  marco_rate * 32 = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_marco_painting_fraction_l3596_359698


namespace NUMINAMATH_CALUDE_limit_one_minus_cos_x_over_x_squared_l3596_359648

theorem limit_one_minus_cos_x_over_x_squared :
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((1 - Real.cos x) / x^2) - (1/2)| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_one_minus_cos_x_over_x_squared_l3596_359648


namespace NUMINAMATH_CALUDE_min_value_m2_plus_n2_l3596_359667

theorem min_value_m2_plus_n2 (m n : ℝ) (hm : m ≠ 0) :
  let f := λ x : ℝ => m * x^2 + (2*n + 1) * x - m - 2
  (∃ x ∈ Set.Icc 3 4, f x = 0) →
  (∀ a b : ℝ, (∃ x ∈ Set.Icc 3 4, a * x^2 + (2*b + 1) * x - a - 2 = 0) → a^2 + b^2 ≥ 1/100) ∧
  (∃ a b : ℝ, (∃ x ∈ Set.Icc 3 4, a * x^2 + (2*b + 1) * x - a - 2 = 0) ∧ a^2 + b^2 = 1/100) :=
by sorry

end NUMINAMATH_CALUDE_min_value_m2_plus_n2_l3596_359667


namespace NUMINAMATH_CALUDE_dvd_shipping_cost_percentage_l3596_359620

/-- Given Mike's DVD cost, Steve's DVD cost as twice Mike's, and Steve's total cost,
    prove that the shipping cost percentage of Steve's DVD price is 80% -/
theorem dvd_shipping_cost_percentage
  (mike_cost : ℝ)
  (steve_dvd_cost : ℝ)
  (steve_total_cost : ℝ)
  (h1 : mike_cost = 5)
  (h2 : steve_dvd_cost = 2 * mike_cost)
  (h3 : steve_total_cost = 18) :
  (steve_total_cost - steve_dvd_cost) / steve_dvd_cost * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_dvd_shipping_cost_percentage_l3596_359620


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3596_359659

-- Define sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2*x > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x < 3/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3596_359659


namespace NUMINAMATH_CALUDE_least_multiple_75_with_digit_product_75_l3596_359612

def is_multiple_of_75 (n : ℕ) : Prop := ∃ k : ℕ, n = 75 * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of_75 n ∧ is_multiple_of_75 (digit_product n)

theorem least_multiple_75_with_digit_product_75 :
  satisfies_conditions 75375 ∧ ∀ m : ℕ, m < 75375 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_least_multiple_75_with_digit_product_75_l3596_359612


namespace NUMINAMATH_CALUDE_strawberry_price_proof_l3596_359607

/-- The cost of strawberries in dollars per pound -/
def strawberry_cost : ℝ := sorry

/-- The cost of cherries in dollars per pound -/
def cherry_cost : ℝ := sorry

/-- The total cost of 5 pounds of strawberries and 5 pounds of cherries -/
def total_cost : ℝ := sorry

theorem strawberry_price_proof :
  (cherry_cost = 6 * strawberry_cost) →
  (total_cost = 5 * strawberry_cost + 5 * cherry_cost) →
  (total_cost = 77) →
  (strawberry_cost = 2.2) := by
  sorry

end NUMINAMATH_CALUDE_strawberry_price_proof_l3596_359607


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3596_359625

theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ := 2 * x - 4
  let a₂ := 3 * x + 2
  let a₃ := 5 * x - 1
  (a₂ - a₁ = a₃ - a₂) → x = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3596_359625
