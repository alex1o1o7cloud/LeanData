import Mathlib

namespace NUMINAMATH_CALUDE_josephs_speed_josephs_speed_proof_l2363_236387

/-- Joseph's driving problem -/
theorem josephs_speed : ℝ → Prop :=
  fun speed : ℝ =>
    let kyle_distance : ℝ := 62 * 2
    let joseph_distance : ℝ := kyle_distance + 1
    let joseph_time : ℝ := 2.5
    speed * joseph_time = joseph_distance → speed = 50

/-- Proof of Joseph's speed -/
theorem josephs_speed_proof : ∃ (speed : ℝ), josephs_speed speed := by
  sorry

end NUMINAMATH_CALUDE_josephs_speed_josephs_speed_proof_l2363_236387


namespace NUMINAMATH_CALUDE_cost_per_pill_is_five_l2363_236366

/-- Represents the annual costs and medication details for Tom --/
structure AnnualMedication where
  pillsPerDay : ℕ
  doctorVisitsPerYear : ℕ
  doctorVisitCost : ℕ
  insuranceCoveragePercent : ℚ
  totalAnnualCost : ℕ

/-- Calculates the cost per pill before insurance coverage --/
def costPerPillBeforeInsurance (am : AnnualMedication) : ℚ :=
  let totalPillsPerYear := am.pillsPerDay * 365
  let annualDoctorVisitsCost := am.doctorVisitsPerYear * am.doctorVisitCost
  let annualMedicationCost := am.totalAnnualCost - annualDoctorVisitsCost
  let totalMedicationCostBeforeInsurance := annualMedicationCost / (1 - am.insuranceCoveragePercent)
  totalMedicationCostBeforeInsurance / totalPillsPerYear

/-- Theorem stating that the cost per pill before insurance is $5 --/
theorem cost_per_pill_is_five (am : AnnualMedication) 
    (h1 : am.pillsPerDay = 2)
    (h2 : am.doctorVisitsPerYear = 2)
    (h3 : am.doctorVisitCost = 400)
    (h4 : am.insuranceCoveragePercent = 4/5)
    (h5 : am.totalAnnualCost = 1530) : 
  costPerPillBeforeInsurance am = 5 := by
  sorry

#eval costPerPillBeforeInsurance {
  pillsPerDay := 2,
  doctorVisitsPerYear := 2,
  doctorVisitCost := 400,
  insuranceCoveragePercent := 4/5,
  totalAnnualCost := 1530
}

end NUMINAMATH_CALUDE_cost_per_pill_is_five_l2363_236366


namespace NUMINAMATH_CALUDE_divides_cubic_minus_one_l2363_236397

theorem divides_cubic_minus_one (a : ℤ) : 
  35 ∣ (a^3 - 1) ↔ a % 35 = 1 ∨ a % 35 = 11 ∨ a % 35 = 16 := by
  sorry

end NUMINAMATH_CALUDE_divides_cubic_minus_one_l2363_236397


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3328_l2363_236354

theorem largest_prime_factor_of_3328 : 
  (Nat.factors 3328).maximum? = some 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3328_l2363_236354


namespace NUMINAMATH_CALUDE_couch_cost_is_750_l2363_236329

/-- The cost of the couch Daria bought -/
def couch_cost : ℕ := sorry

/-- The amount Daria has saved -/
def savings : ℕ := 500

/-- The cost of the table Daria bought -/
def table_cost : ℕ := 100

/-- The cost of the lamp Daria bought -/
def lamp_cost : ℕ := 50

/-- The amount Daria still owes after paying her savings -/
def remaining_debt : ℕ := 400

/-- Theorem stating that the couch cost is $750 -/
theorem couch_cost_is_750 :
  couch_cost = 750 ∧
  couch_cost + table_cost + lamp_cost = savings + remaining_debt :=
sorry

end NUMINAMATH_CALUDE_couch_cost_is_750_l2363_236329


namespace NUMINAMATH_CALUDE_twenty_bees_honey_production_l2363_236389

/-- The amount of honey (in grams) produced by a given number of bees in 20 days. -/
def honey_production (num_bees : ℕ) : ℝ :=
  num_bees * 1

/-- Theorem stating that 20 honey bees produce 20 grams of honey in 20 days. -/
theorem twenty_bees_honey_production :
  honey_production 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_bees_honey_production_l2363_236389


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2363_236328

theorem complex_number_quadrant : ∀ z : ℂ, 
  z = (3 + 4*I) / I → (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2363_236328


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_seven_undefined_inverse_smallest_is_seven_l2363_236342

theorem smallest_undefined_inverse (b : ℕ) : b > 0 ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 77]) → 
  b ≥ 7 :=
by sorry

theorem seven_undefined_inverse : 
  ¬ (∃ x : ℕ, x * 7 ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * 7 ≡ 1 [MOD 77]) :=
by sorry

theorem smallest_is_seven : 
  ∃ (b : ℕ), b > 0 ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 77]) ∧
  ∀ (c : ℕ), c > 0 ∧ 
  ¬ (∃ x : ℕ, x * c ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * c ≡ 1 [MOD 77]) →
  c ≥ b :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_seven_undefined_inverse_smallest_is_seven_l2363_236342


namespace NUMINAMATH_CALUDE_smallest_base_10_integer_l2363_236362

def is_valid_base_6_digit (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ 5

def is_valid_base_8_digit (y : ℕ) : Prop := y ≥ 0 ∧ y ≤ 7

def base_6_to_decimal (x : ℕ) : ℕ := 6 * x + x

def base_8_to_decimal (y : ℕ) : ℕ := 8 * y + y

theorem smallest_base_10_integer : 
  ∃ (x y : ℕ), 
    is_valid_base_6_digit x ∧ 
    is_valid_base_8_digit y ∧ 
    base_6_to_decimal x = 63 ∧ 
    base_8_to_decimal y = 63 ∧ 
    (∀ (x' y' : ℕ), 
      is_valid_base_6_digit x' ∧ 
      is_valid_base_8_digit y' ∧ 
      base_6_to_decimal x' = base_8_to_decimal y' → 
      base_6_to_decimal x' ≥ 63) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_10_integer_l2363_236362


namespace NUMINAMATH_CALUDE_division_problem_l2363_236356

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 12401 → 
  divisor = 163 → 
  remainder = 13 → 
  dividend = divisor * quotient + remainder → 
  quotient = 76 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2363_236356


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l2363_236353

/-- Definition of "equation number pair" -/
def is_equation_number_pair (a b : ℝ) : Prop :=
  ∃ x : ℝ, (a / x) + 1 = b ∧ x = 1 / (a + b)

/-- Part 1: Prove [3,-5] is an "equation number pair" and [-2,4] is not -/
theorem part_one :
  is_equation_number_pair 3 (-5) ∧ ¬is_equation_number_pair (-2) 4 := by sorry

/-- Part 2: If [n,3-n] is an "equation number pair", then n = 1/2 -/
theorem part_two (n : ℝ) :
  is_equation_number_pair n (3 - n) → n = 1/2 := by sorry

/-- Part 3: If [m-k,k] is an "equation number pair" (m ≠ -1, m ≠ 0, k ≠ 1), then k = (m^2 + 1) / (m + 1) -/
theorem part_three (m k : ℝ) (hm1 : m ≠ -1) (hm2 : m ≠ 0) (hk : k ≠ 1) :
  is_equation_number_pair (m - k) k → k = (m^2 + 1) / (m + 1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l2363_236353


namespace NUMINAMATH_CALUDE_simplify_expression_l2363_236330

theorem simplify_expression (a x y : ℝ) : a^2 * x^2 - a^2 * y^2 = a^2 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2363_236330


namespace NUMINAMATH_CALUDE_carpenter_theorem_l2363_236382

def carpenter_problem (total_woodblocks : ℕ) (current_logs : ℕ) (woodblocks_per_log : ℕ) : ℕ :=
  let current_woodblocks := current_logs * woodblocks_per_log
  let remaining_woodblocks := total_woodblocks - current_woodblocks
  remaining_woodblocks / woodblocks_per_log

theorem carpenter_theorem :
  carpenter_problem 80 8 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_carpenter_theorem_l2363_236382


namespace NUMINAMATH_CALUDE_triangle_angle_B_l2363_236369

theorem triangle_angle_B (A B C : Real) (a b c : Real) : 
  A = 2 * Real.pi / 3 →  -- 120° in radians
  a = 2 →
  b = 2 * Real.sqrt 3 / 3 →
  B = Real.pi / 6  -- 30° in radians
:= by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l2363_236369


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2363_236308

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) :
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a - 2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2363_236308


namespace NUMINAMATH_CALUDE_principal_calculation_l2363_236345

/-- Simple interest calculation --/
def simple_interest (principal rate time : ℝ) : ℝ := principal * (1 + rate * time)

/-- Theorem: Principal calculation given two-year and three-year amounts --/
theorem principal_calculation (amount_2_years amount_3_years : ℝ) 
  (h1 : amount_2_years = 3450)
  (h2 : amount_3_years = 3655)
  (h3 : ∃ (p r : ℝ), simple_interest p r 2 = amount_2_years ∧ simple_interest p r 3 = amount_3_years) :
  ∃ (p r : ℝ), p = 3245 ∧ simple_interest p r 2 = amount_2_years ∧ simple_interest p r 3 = amount_3_years := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l2363_236345


namespace NUMINAMATH_CALUDE_kenny_jumping_jacks_wednesday_l2363_236374

/-- Represents the number of jumping jacks Kenny did on each day of the week -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for a week -/
def totalJumpingJacks (week : WeeklyJumpingJacks) : ℕ :=
  week.sunday + week.monday + week.tuesday + week.wednesday + week.thursday + week.friday + week.saturday

theorem kenny_jumping_jacks_wednesday (lastWeek : ℕ) (thisWeek : WeeklyJumpingJacks) 
    (h1 : lastWeek = 324)
    (h2 : thisWeek.sunday = 34)
    (h3 : thisWeek.monday = 20)
    (h4 : thisWeek.tuesday = 0)
    (h5 : thisWeek.thursday = 64 ∨ thisWeek.wednesday = 64)
    (h6 : thisWeek.friday = 23)
    (h7 : thisWeek.saturday = 61)
    (h8 : totalJumpingJacks thisWeek > lastWeek) :
  thisWeek.wednesday = 59 := by
  sorry

#check kenny_jumping_jacks_wednesday

end NUMINAMATH_CALUDE_kenny_jumping_jacks_wednesday_l2363_236374


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l2363_236335

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes with no empty boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 3 ways to distribute 7 indistinguishable balls into 4 indistinguishable boxes with no empty boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l2363_236335


namespace NUMINAMATH_CALUDE_book_pages_count_l2363_236358

/-- Given a book with pages numbered consecutively starting from 1,
    this function calculates the total number of digits used to number the pages. -/
def totalDigits (n : ℕ) : ℕ :=
  (n.min 9) + 
  (n - 9).max 0 * 2 + 
  (n - 99).max 0 * 3

/-- Theorem stating that a book has 369 pages if the total number of digits
    used in numbering is 999. -/
theorem book_pages_count : totalDigits 369 = 999 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l2363_236358


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_23_l2363_236376

def digit_sum (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def all_digits_different (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

theorem largest_number_with_digit_sum_23 :
  ∀ n : Nat, n ≤ 999 →
    (digit_sum n = 23 ∧ all_digits_different n) →
    n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_23_l2363_236376


namespace NUMINAMATH_CALUDE_trig_identity_l2363_236338

theorem trig_identity (α : Real) (h : Real.tan α = 3) : 
  (Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2363_236338


namespace NUMINAMATH_CALUDE_total_dry_grapes_weight_l2363_236327

/-- Calculates the total weight of Dry Grapes after dehydrating Fresh Grapes Type A and B -/
theorem total_dry_grapes_weight 
  (water_content_A : Real) 
  (water_content_B : Real)
  (weight_A : Real) 
  (weight_B : Real) :
  water_content_A = 0.92 →
  water_content_B = 0.88 →
  weight_A = 30 →
  weight_B = 50 →
  (1 - water_content_A) * weight_A + (1 - water_content_B) * weight_B = 8.4 :=
by sorry

end NUMINAMATH_CALUDE_total_dry_grapes_weight_l2363_236327


namespace NUMINAMATH_CALUDE_mark_sold_eight_less_l2363_236365

theorem mark_sold_eight_less (total : ℕ) (mark_sold : ℕ) (ann_sold : ℕ) 
  (h_total : total = 9)
  (h_mark : mark_sold < total)
  (h_ann : ann_sold = total - 2)
  (h_mark_positive : mark_sold ≥ 1)
  (h_ann_positive : ann_sold ≥ 1)
  (h_total_greater : mark_sold + ann_sold < total) :
  total - mark_sold = 8 := by
sorry

end NUMINAMATH_CALUDE_mark_sold_eight_less_l2363_236365


namespace NUMINAMATH_CALUDE_x_wins_probability_l2363_236307

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  games_per_team : Nat
  win_probability : ℚ
  
/-- Represents the outcome of the tournament for two specific teams -/
structure TournamentOutcome where
  team_x_points : Nat
  team_y_points : Nat

/-- Calculates the probability of team X finishing with more points than team Y -/
def probability_x_wins (t : SoccerTournament) : ℚ :=
  sorry

/-- The main theorem stating the probability for the given conditions -/
theorem x_wins_probability (t : SoccerTournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.games_per_team = 7)
  (h3 : t.win_probability = 1/2) :
  probability_x_wins t = 561/1024 := by
  sorry

end NUMINAMATH_CALUDE_x_wins_probability_l2363_236307


namespace NUMINAMATH_CALUDE_second_fraction_in_compound_ratio_l2363_236361

theorem second_fraction_in_compound_ratio
  (compound_ratio : ℝ)
  (h_ratio : compound_ratio = 0.07142857142857142)
  (f1 : ℝ) (h_f1 : f1 = 2/3)
  (f3 : ℝ) (h_f3 : f3 = 1/3)
  (f4 : ℝ) (h_f4 : f4 = 3/8) :
  ∃ x : ℝ, x * f1 * f3 * f4 = compound_ratio ∧ x = 0.8571428571428571 := by
  sorry

end NUMINAMATH_CALUDE_second_fraction_in_compound_ratio_l2363_236361


namespace NUMINAMATH_CALUDE_repeating_decimal_proof_l2363_236380

/-- The repeating decimal 0.4̅67̅ as a rational number -/
def repeating_decimal : ℚ := 463 / 990

/-- Proof that 0.4̅67̅ is equal to 463/990 and is in lowest terms -/
theorem repeating_decimal_proof :
  repeating_decimal = 463 / 990 ∧
  (∀ n d : ℤ, n / d = 463 / 990 → d ≠ 0 → d.natAbs ≤ 990 → d = 990) :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_proof_l2363_236380


namespace NUMINAMATH_CALUDE_digit_multiplication_l2363_236304

theorem digit_multiplication (A B : ℕ) : 
  A < 10 ∧ B < 10 ∧ A ≠ B ∧ A * (10 * A + B) = 100 * B + 11 * A → A = 8 ∧ B = 6 := by
  sorry

end NUMINAMATH_CALUDE_digit_multiplication_l2363_236304


namespace NUMINAMATH_CALUDE_y_congruence_l2363_236318

theorem y_congruence (y : ℤ) 
  (h1 : (2 + y) % (2^3) = (2 * 2) % (2^3))
  (h2 : (4 + y) % (4^3) = (4 * 2) % (4^3))
  (h3 : (6 + y) % (6^3) = (6 * 2) % (6^3)) :
  y % 24 = 2 := by
  sorry

end NUMINAMATH_CALUDE_y_congruence_l2363_236318


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l2363_236391

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l2363_236391


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l2363_236367

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed of the car in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 80) 
  (h2 : average_speed = 70) : 
  (2 * average_speed - speed_first_hour) = 60 := by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l2363_236367


namespace NUMINAMATH_CALUDE_ocean_area_scientific_notation_l2363_236350

-- Define the original number
def original_number : ℝ := 2997000

-- Define the scientific notation components
def scientific_base : ℝ := 2.997
def scientific_exponent : ℤ := 6

-- Theorem statement
theorem ocean_area_scientific_notation :
  original_number = scientific_base * (10 : ℝ) ^ scientific_exponent :=
by sorry

end NUMINAMATH_CALUDE_ocean_area_scientific_notation_l2363_236350


namespace NUMINAMATH_CALUDE_power_equality_implies_y_equals_four_l2363_236339

theorem power_equality_implies_y_equals_four :
  ∀ y : ℝ, (4 : ℝ)^12 = 64^y → y = 4 := by
sorry

end NUMINAMATH_CALUDE_power_equality_implies_y_equals_four_l2363_236339


namespace NUMINAMATH_CALUDE_pet_store_cages_l2363_236355

/-- Represents the number of birds in each cage -/
def birds_per_cage : ℕ := 10

/-- Represents the total number of birds in the store -/
def total_birds : ℕ := 40

/-- Represents the number of bird cages in the store -/
def num_cages : ℕ := total_birds / birds_per_cage

theorem pet_store_cages : num_cages = 4 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2363_236355


namespace NUMINAMATH_CALUDE_marble_problem_l2363_236385

/-- The number of marbles Doug lost at the playground -/
def marbles_lost (ed_initial : ℕ) (doug_initial : ℕ) (ed_final : ℕ) (doug_final : ℕ) : ℕ :=
  doug_initial - doug_final

theorem marble_problem (ed_initial : ℕ) (doug_initial : ℕ) (ed_final : ℕ) (doug_final : ℕ) :
  ed_initial = doug_initial + 10 →
  ed_initial = 45 →
  ed_final = doug_final + 21 →
  ed_initial = ed_final →
  marbles_lost ed_initial doug_initial ed_final doug_final = 11 := by
  sorry

#check marble_problem

end NUMINAMATH_CALUDE_marble_problem_l2363_236385


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l2363_236370

theorem not_p_and_q_implies_at_most_one_true (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l2363_236370


namespace NUMINAMATH_CALUDE_max_dot_product_on_circle_l2363_236312

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define points M and N
def M : ℝ × ℝ := (2, 0)
def N : ℝ × ℝ := (0, -2)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem max_dot_product_on_circle :
  ∀ (P : ℝ × ℝ), Circle P.1 P.2 →
    dot_product (P.1 - M.1, P.2 - M.2) (P.1 - N.1, P.2 - N.2) ≤ 4 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_on_circle_l2363_236312


namespace NUMINAMATH_CALUDE_fraction_sum_l2363_236352

theorem fraction_sum : (3 : ℚ) / 9 + (7 : ℚ) / 14 = (5 : ℚ) / 6 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_l2363_236352


namespace NUMINAMATH_CALUDE_at_least_one_negative_l2363_236395

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  (a < 0) ∨ (b < 0) ∨ (c < 0) ∨ (d < 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l2363_236395


namespace NUMINAMATH_CALUDE_problem_statement_l2363_236384

theorem problem_statement (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z*(x*y + x*z + y*z)) = -7 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2363_236384


namespace NUMINAMATH_CALUDE_triangle_problem_geometric_sequence_problem_l2363_236351

-- Triangle problem
theorem triangle_problem (a b : ℝ) (B : ℝ) 
  (ha : a = Real.sqrt 3) 
  (hb : b = Real.sqrt 2) 
  (hB : B = 45 * π / 180) :
  (∃ (A C c : ℝ),
    (A = 60 * π / 180 ∧ C = 75 * π / 180 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨
    (A = 120 * π / 180 ∧ C = 15 * π / 180 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) :=
by sorry

-- Geometric sequence problem
theorem geometric_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hS2 : S 2 = 7)
  (hS6 : S 6 = 91)
  (h_geom : ∀ n, S (n+1) - S n = (S 2 - S 1) * (S 2 / S 1) ^ (n-1)) :
  S 4 = 28 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_geometric_sequence_problem_l2363_236351


namespace NUMINAMATH_CALUDE_expression_evaluation_l2363_236371

theorem expression_evaluation :
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  x^2 + y^2 - z^2 + 2*x*y + 2*y*z = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2363_236371


namespace NUMINAMATH_CALUDE_problem_solution_l2363_236381

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -6)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2363_236381


namespace NUMINAMATH_CALUDE_simplify_expression_l2363_236372

theorem simplify_expression (x : ℝ) : 114 * x - 69 * x + 15 = 45 * x + 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2363_236372


namespace NUMINAMATH_CALUDE_range_of_f_l2363_236364

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the domain
def domain : Set ℝ := { x | 1 ≤ x ∧ x < 5 }

-- State the theorem
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -4 ≤ y ∧ y < 5 } := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2363_236364


namespace NUMINAMATH_CALUDE_final_result_l2363_236334

def alternateOperations (start : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => start
  | n + 1 => if n % 2 = 0 
             then alternateOperations start n * 3 
             else alternateOperations start n / 2

theorem final_result : alternateOperations 1458 5 = 3^9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l2363_236334


namespace NUMINAMATH_CALUDE_store_a_highest_capacity_l2363_236379

/-- Represents a store with its CD storage capacity -/
structure Store where
  shelves : ℕ
  racks_per_shelf : ℕ
  cds_per_rack : ℕ

/-- Calculates the total CD capacity of a store -/
def total_capacity (s : Store) : ℕ :=
  s.shelves * s.racks_per_shelf * s.cds_per_rack

/-- The three stores with their respective capacities -/
def store_a : Store := ⟨5, 6, 9⟩
def store_b : Store := ⟨8, 4, 7⟩
def store_c : Store := ⟨10, 3, 8⟩

/-- Theorem stating that Store A has the highest total CD capacity -/
theorem store_a_highest_capacity :
  total_capacity store_a > total_capacity store_b ∧
  total_capacity store_a > total_capacity store_c :=
by
  sorry


end NUMINAMATH_CALUDE_store_a_highest_capacity_l2363_236379


namespace NUMINAMATH_CALUDE_set_intersection_condition_l2363_236396

theorem set_intersection_condition (m : ℝ) : 
  let A := {x : ℝ | x^2 - 3*x + 2 = 0}
  let C := {x : ℝ | x^2 - m*x + 2 = 0}
  (A ∩ C = C) ↔ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_set_intersection_condition_l2363_236396


namespace NUMINAMATH_CALUDE_average_marks_first_five_subjects_l2363_236394

theorem average_marks_first_five_subjects 
  (total_subjects : Nat) 
  (average_six_subjects : ℝ) 
  (marks_sixth_subject : ℝ) 
  (h1 : total_subjects = 6) 
  (h2 : average_six_subjects = 76) 
  (h3 : marks_sixth_subject = 86) : 
  (total_subjects - 1 : ℝ)⁻¹ * (total_subjects * average_six_subjects - marks_sixth_subject) = 74 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_first_five_subjects_l2363_236394


namespace NUMINAMATH_CALUDE_min_keystrokes_for_2018_l2363_236305

/-- Represents the state of the screen and copy buffer -/
structure ScreenState where
  screen : ℕ  -- number of 'a's on screen
  buffer : ℕ  -- number of 'a's in copy buffer

/-- Represents the possible operations -/
inductive Operation
  | Copy
  | Paste

/-- Applies an operation to the screen state -/
def applyOperation (state : ScreenState) (op : Operation) : ScreenState :=
  match op with
  | Operation.Copy => { state with buffer := state.screen }
  | Operation.Paste => { state with screen := state.screen + state.buffer }

/-- Applies a sequence of operations to the initial screen state -/
def applyOperations (ops : List Operation) : ScreenState :=
  ops.foldl applyOperation { screen := 1, buffer := 0 }

/-- Checks if a sequence of operations achieves the goal -/
def achievesGoal (ops : List Operation) : Prop :=
  (applyOperations ops).screen ≥ 2018

theorem min_keystrokes_for_2018 :
  ∃ (ops : List Operation), achievesGoal ops ∧ ops.length = 21 ∧
  (∀ (other_ops : List Operation), achievesGoal other_ops → other_ops.length ≥ 21) :=
sorry

end NUMINAMATH_CALUDE_min_keystrokes_for_2018_l2363_236305


namespace NUMINAMATH_CALUDE_rotation_volume_of_specific_trapezoid_l2363_236309

/-- A trapezoid with given properties -/
structure Trapezoid where
  larger_base : ℝ
  smaller_base : ℝ
  adjacent_angle : ℝ

/-- The volume of the solid formed by rotating the trapezoid about its larger base -/
def rotation_volume (t : Trapezoid) : ℝ := sorry

/-- The theorem stating the volume of the rotated trapezoid -/
theorem rotation_volume_of_specific_trapezoid :
  let t : Trapezoid := {
    larger_base := 8,
    smaller_base := 2,
    adjacent_angle := Real.pi / 4  -- 45° in radians
  }
  rotation_volume t = 36 * Real.pi := by sorry

end NUMINAMATH_CALUDE_rotation_volume_of_specific_trapezoid_l2363_236309


namespace NUMINAMATH_CALUDE_skittles_theorem_l2363_236399

def skittles_problem (brandon_initial bonnie_initial brandon_loss : ℕ) : Prop :=
  let brandon_after_loss := brandon_initial - brandon_loss
  let combined := brandon_after_loss + bonnie_initial
  let each_share := combined / 4
  let chloe_initial := each_share
  let dylan_initial := each_share
  let chloe_to_dylan := chloe_initial / 2
  let dylan_after_chloe := dylan_initial + chloe_to_dylan
  let dylan_to_bonnie := dylan_after_chloe / 3
  let dylan_final := dylan_after_chloe - dylan_to_bonnie
  dylan_final = 22

theorem skittles_theorem : skittles_problem 96 4 9 := by sorry

end NUMINAMATH_CALUDE_skittles_theorem_l2363_236399


namespace NUMINAMATH_CALUDE_union_equals_reals_subset_of_complement_l2363_236325

-- Define the sets A and B
def A : Set ℝ := {x | x < 0 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2*a}

-- Theorem for part (1)
theorem union_equals_reals (a : ℝ) : 
  A ∪ B a = Set.univ ↔ a ∈ Set.Iic 0 :=
sorry

-- Theorem for part (2)
theorem subset_of_complement (a : ℝ) :
  B a ⊆ (Set.univ \ A) ↔ a ∈ Set.Ici (1/2) :=
sorry

end NUMINAMATH_CALUDE_union_equals_reals_subset_of_complement_l2363_236325


namespace NUMINAMATH_CALUDE_cos_forty_five_degrees_l2363_236359

theorem cos_forty_five_degrees : Real.cos (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_forty_five_degrees_l2363_236359


namespace NUMINAMATH_CALUDE_smallest_x_y_sum_l2363_236306

/-- The smallest positive integer x such that 720x is a square number -/
def x : ℕ+ := sorry

/-- The smallest positive integer y such that 720y is a fourth power -/
def y : ℕ+ := sorry

theorem smallest_x_y_sum : 
  (∀ x' : ℕ+, x' < x → ¬∃ n : ℕ+, 720 * x' = n^2) ∧
  (∀ y' : ℕ+, y' < y → ¬∃ n : ℕ+, 720 * y' = n^4) ∧
  (∃ n : ℕ+, 720 * x = n^2) ∧
  (∃ n : ℕ+, 720 * y = n^4) ∧
  (x : ℕ) + (y : ℕ) = 1130 := by sorry

end NUMINAMATH_CALUDE_smallest_x_y_sum_l2363_236306


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l2363_236383

theorem quadratic_complete_square (x : ℝ) :
  25 * x^2 + 20 * x - 1000 = 0 →
  ∃ (p t : ℝ), (x + p)^2 = t ∧ t = 104/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l2363_236383


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2363_236311

def polynomial (x : ℝ) : ℝ := 3 * (3 * x^7 + 8 * x^4 - 7) + 7 * (x^5 - 7 * x^2 + 5)

theorem sum_of_coefficients : polynomial 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2363_236311


namespace NUMINAMATH_CALUDE_train_length_l2363_236332

/-- The length of a train given its speed and time to cross a pole --/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 18 → ∃ (length_m : ℝ), abs (length_m - 300) < 1 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2363_236332


namespace NUMINAMATH_CALUDE_janelle_has_72_marbles_l2363_236336

/-- The number of marbles Janelle has after buying blue marbles and giving some away as a gift. -/
def janelles_marbles : ℕ :=
  let initial_green : ℕ := 26
  let blue_bags : ℕ := 6
  let marbles_per_bag : ℕ := 10
  let gift_green : ℕ := 6
  let gift_blue : ℕ := 8
  let total_blue : ℕ := blue_bags * marbles_per_bag
  let total_before_gift : ℕ := initial_green + total_blue
  let total_gift : ℕ := gift_green + gift_blue
  total_before_gift - total_gift

/-- Theorem stating that Janelle has 72 marbles after the transactions. -/
theorem janelle_has_72_marbles : janelles_marbles = 72 := by
  sorry

end NUMINAMATH_CALUDE_janelle_has_72_marbles_l2363_236336


namespace NUMINAMATH_CALUDE_sally_sunday_sandwiches_l2363_236388

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of pieces of bread used per sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := 6

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

theorem sally_sunday_sandwiches :
  sunday_sandwiches = total_bread / bread_per_sandwich - saturday_sandwiches :=
sorry

end NUMINAMATH_CALUDE_sally_sunday_sandwiches_l2363_236388


namespace NUMINAMATH_CALUDE_unique_right_triangle_from_medians_l2363_236324

/-- Given two positive real numbers representing the lengths of medians from the endpoints of a hypotenuse,
    there exists at most one right triangle with these medians. -/
theorem unique_right_triangle_from_medians (s_a s_b : ℝ) (h_sa : s_a > 0) (h_sb : s_b > 0) :
  ∃! (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
    s_a^2 = (b/2)^2 + (c/2)^2 ∧ s_b^2 = (a/2)^2 + (c/2)^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_from_medians_l2363_236324


namespace NUMINAMATH_CALUDE_b_contribution_is_16200_l2363_236363

/-- Calculates the partner's contribution given the investment details and profit ratio -/
def calculate_partner_contribution (a_investment : ℕ) (total_months : ℕ) (b_join_month : ℕ) (a_profit_share : ℕ) (b_profit_share : ℕ) : ℕ :=
  let a_months := total_months
  let b_months := total_months - b_join_month
  (a_investment * a_months * b_profit_share) / (a_profit_share * b_months)

/-- Proves that B's contribution to the capital is 16200 rs given the problem conditions -/
theorem b_contribution_is_16200 :
  let a_investment := 4500
  let total_months := 12
  let b_join_month := 7
  let a_profit_share := 2
  let b_profit_share := 3
  calculate_partner_contribution a_investment total_months b_join_month a_profit_share b_profit_share = 16200 := by
  sorry

end NUMINAMATH_CALUDE_b_contribution_is_16200_l2363_236363


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_k_bound_l2363_236360

def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 + 2 * k * x - 8

theorem monotone_decreasing_implies_k_bound :
  (∀ x₁ x₂, -5 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ -1 → f k x₁ > f k x₂) →
  k ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_k_bound_l2363_236360


namespace NUMINAMATH_CALUDE_largest_common_term_l2363_236368

def is_in_arithmetic_sequence (a : ℕ) (first d : ℕ) : Prop :=
  ∃ k : ℕ, a = first + k * d

theorem largest_common_term (a₁ d₁ a₂ d₂ : ℕ) (h₁ : a₁ = 3) (h₂ : d₁ = 8) (h₃ : a₂ = 5) (h₄ : d₂ = 9) :
  (∀ n : ℕ, n > 131 ∧ n ≤ 150 → ¬(is_in_arithmetic_sequence n a₁ d₁ ∧ is_in_arithmetic_sequence n a₂ d₂)) ∧
  (is_in_arithmetic_sequence 131 a₁ d₁ ∧ is_in_arithmetic_sequence 131 a₂ d₂) :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l2363_236368


namespace NUMINAMATH_CALUDE_weight_of_b_l2363_236348

/-- Given the average weights of different combinations of people, prove the weight of person b. -/
theorem weight_of_b (a b c d : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43)
  (h4 : (b + c + d) / 3 = 47) :
  b = 31 := by
  sorry


end NUMINAMATH_CALUDE_weight_of_b_l2363_236348


namespace NUMINAMATH_CALUDE_doberman_schnauzer_relationship_num_dobermans_proof_l2363_236321

/-- The number of Doberman puppies -/
def num_dobermans : ℝ := 37.5

/-- The number of Schnauzers -/
def num_schnauzers : ℕ := 55

/-- Theorem stating the relationship between Doberman puppies and Schnauzers -/
theorem doberman_schnauzer_relationship : 
  3 * num_dobermans - 5 + (num_dobermans - num_schnauzers) = 90 :=
by sorry

/-- Theorem proving the number of Doberman puppies -/
theorem num_dobermans_proof : num_dobermans = 37.5 :=
by sorry

end NUMINAMATH_CALUDE_doberman_schnauzer_relationship_num_dobermans_proof_l2363_236321


namespace NUMINAMATH_CALUDE_base7_sum_theorem_l2363_236357

/-- Represents a single digit in base 7 --/
def Base7Digit := Fin 7

/-- Converts a base 7 number to base 10 --/
def toBase10 (x : Base7Digit) : Nat := x.val

/-- The equation 5XY₇ + 32₇ = 62X₇ in base 7 --/
def base7Equation (X Y : Base7Digit) : Prop :=
  (5 * 7 + toBase10 X) * 7 + toBase10 Y + 32 = (6 * 7 + 2) * 7 + toBase10 X

/-- Theorem stating that if X and Y satisfy the base 7 equation, then X + Y = 10 in base 10 --/
theorem base7_sum_theorem (X Y : Base7Digit) : 
  base7Equation X Y → toBase10 X + toBase10 Y = 10 := by
  sorry

end NUMINAMATH_CALUDE_base7_sum_theorem_l2363_236357


namespace NUMINAMATH_CALUDE_eulers_formula_l2363_236303

/-- A convex polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure ConvexPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Euler's formula for convex polyhedra states that the number of faces minus the number of edges plus the number of vertices equals two. -/
theorem eulers_formula (p : ConvexPolyhedron) : p.faces - p.edges + p.vertices = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l2363_236303


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l2363_236310

theorem opposite_of_negative_six : -(-(6)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l2363_236310


namespace NUMINAMATH_CALUDE_three_digit_number_sum_l2363_236340

theorem three_digit_number_sum (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 → a ≠ 0 →
  (100 * a + 10 * c + b) + 
  (100 * b + 10 * c + a) + 
  (100 * b + 10 * a + c) + 
  (100 * c + 10 * a + b) + 
  (100 * c + 10 * b + a) = 3194 →
  100 * a + 10 * b + c = 358 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_sum_l2363_236340


namespace NUMINAMATH_CALUDE_max_product_sum_300_l2363_236386

theorem max_product_sum_300 : 
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧ 
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l2363_236386


namespace NUMINAMATH_CALUDE_fraction_division_problem_l2363_236392

theorem fraction_division_problem : (4 + 2 / 3) / (9 / 7) = 98 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_problem_l2363_236392


namespace NUMINAMATH_CALUDE_correlation_relationships_l2363_236349

/-- A relationship between two variables -/
inductive Relationship
| AgeWealth
| CurveCoordinates
| AppleProductionClimate
| TreeDiameterHeight
| StudentSchool

/-- Predicate to determine if a relationship involves correlation -/
def involves_correlation (r : Relationship) : Prop :=
  match r with
  | Relationship.AgeWealth => true
  | Relationship.CurveCoordinates => false
  | Relationship.AppleProductionClimate => true
  | Relationship.TreeDiameterHeight => true
  | Relationship.StudentSchool => false

/-- The set of all relationships -/
def all_relationships : Set Relationship :=
  {Relationship.AgeWealth, Relationship.CurveCoordinates, Relationship.AppleProductionClimate,
   Relationship.TreeDiameterHeight, Relationship.StudentSchool}

/-- The theorem stating which relationships involve correlation -/
theorem correlation_relationships :
  {r ∈ all_relationships | involves_correlation r} =
  {Relationship.AgeWealth, Relationship.AppleProductionClimate, Relationship.TreeDiameterHeight} :=
by sorry

end NUMINAMATH_CALUDE_correlation_relationships_l2363_236349


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2363_236390

/-- The number of ways to choose 2 items from 5 items -/
def choose_2_from_5 : ℕ := 10

/-- The number of rectangles in a 5x5 grid -/
def num_rectangles : ℕ := choose_2_from_5 * choose_2_from_5

theorem rectangles_in_5x5_grid :
  num_rectangles = 100 := by sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2363_236390


namespace NUMINAMATH_CALUDE_new_quad_inscribable_l2363_236343

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define the points on the circle
variable (A₁ A₂ B₁ B₂ C₁ C₂ D₁ D₂ : ℝ × ℝ)

-- Define the convex quadrilateral
variable (quad : Set (ℝ × ℝ))

-- Define the condition that the quadrilateral is inscribed in the circle
variable (quad_inscribed : quad ⊆ circle)

-- Define the condition that the extended sides intersect the circle at the given points
variable (extended_sides : 
  A₁ ∈ circle ∧ A₂ ∈ circle ∧ 
  B₁ ∈ circle ∧ B₂ ∈ circle ∧ 
  C₁ ∈ circle ∧ C₂ ∈ circle ∧ 
  D₁ ∈ circle ∧ D₂ ∈ circle)

-- Define the equality condition
variable (equality_condition : 
  dist A₁ B₂ = dist B₁ C₂ ∧ 
  dist B₁ C₂ = dist C₁ D₂ ∧ 
  dist C₁ D₂ = dist D₁ A₂)

-- Define the quadrilateral formed by the lines A₁A₂, B₁B₂, C₁C₂, D₁D₂
def new_quad : Set (ℝ × ℝ) := sorry

-- The theorem to be proved
theorem new_quad_inscribable :
  ∃ (new_circle : Set (ℝ × ℝ)), new_quad ⊆ new_circle :=
sorry

end NUMINAMATH_CALUDE_new_quad_inscribable_l2363_236343


namespace NUMINAMATH_CALUDE_positive_solution_form_l2363_236323

theorem positive_solution_form (x : ℝ) : 
  x^2 - 18*x = 80 → 
  x > 0 → 
  ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ x = Real.sqrt c - d ∧ c = 161 ∧ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_form_l2363_236323


namespace NUMINAMATH_CALUDE_solution_conditions_l2363_236347

-- Define the variables
variable (a b x y z : ℝ)

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  (a > 0) ∧ (abs b < a) ∧ (a < Real.sqrt 2 * abs b) ∧ ((3 * a^2 - b^2) * (3 * b^2 - a^2) > 0)

-- Define the equations
def equations (a b x y z : ℝ) : Prop :=
  (x + y + z = a) ∧ (x^2 + y^2 + z^2 = b^2) ∧ (x * y = z^2)

-- Define the property of distinct positive numbers
def distinct_positive (x y z : ℝ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z)

-- Theorem statement
theorem solution_conditions (a b x y z : ℝ) :
  equations a b x y z → (conditions a b ↔ distinct_positive x y z) := by
  sorry

end NUMINAMATH_CALUDE_solution_conditions_l2363_236347


namespace NUMINAMATH_CALUDE_f_properties_l2363_236302

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

theorem f_properties :
  let T := Real.pi
  let φ := 7 * Real.pi / 12
  (∀ x, f (x + T) = f x) ∧
  (∀ y, T ≤ y → (∀ x, f (x + y) = f x) → y = T) ∧
  (Real.pi / 2 < φ ∧ φ < Real.pi) ∧
  (∀ x, f (x + φ) = f (-x + φ)) ∧
  (∀ ψ, Real.pi / 2 < ψ ∧ ψ < Real.pi → (∀ x, f (x + ψ) = f (-x + ψ)) → ψ = φ) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2363_236302


namespace NUMINAMATH_CALUDE_series_sum_equals_half_l2363_236337

noncomputable def series_sum (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

theorem series_sum_equals_half :
  ∑' n, series_sum n = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_half_l2363_236337


namespace NUMINAMATH_CALUDE_first_place_beats_joe_by_two_l2363_236378

/-- Calculates the total points for a team based on their match results -/
def calculate_points (wins : ℕ) (ties : ℕ) : ℕ :=
  3 * wins + ties

/-- Represents the scoring system and match results for the soccer tournament -/
structure TournamentResults where
  win_points : ℕ := 3
  tie_points : ℕ := 1
  joe_wins : ℕ := 1
  joe_ties : ℕ := 3
  first_place_wins : ℕ := 2
  first_place_ties : ℕ := 2

/-- Theorem stating that the first-place team beat Joe's team by 2 points -/
theorem first_place_beats_joe_by_two (results : TournamentResults) :
  calculate_points results.first_place_wins results.first_place_ties -
  calculate_points results.joe_wins results.joe_ties = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_first_place_beats_joe_by_two_l2363_236378


namespace NUMINAMATH_CALUDE_dime_count_in_collection_l2363_236316

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25

/-- Represents a collection of coins --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of a coin collection in cents --/
def totalValue (c : CoinCollection) : ℕ :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime +
  c.quarters * coinValue CoinType.Quarter

/-- Calculates the total number of coins in a collection --/
def totalCoins (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes + c.quarters

theorem dime_count_in_collection (c : CoinCollection) :
  totalCoins c = 13 ∧
  totalValue c = 141 ∧
  c.pennies ≥ 2 ∧
  c.nickels ≥ 2 ∧
  c.dimes ≥ 2 ∧
  c.quarters ≥ 2 →
  c.dimes = 3 := by
  sorry

end NUMINAMATH_CALUDE_dime_count_in_collection_l2363_236316


namespace NUMINAMATH_CALUDE_not_order_preserving_isomorphic_Z_Q_l2363_236300

theorem not_order_preserving_isomorphic_Z_Q :
  ¬∃ f : ℤ → ℚ, (∀ q : ℚ, ∃ z : ℤ, f z = q) ∧
    (∀ z₁ z₂ : ℤ, z₁ < z₂ → f z₁ < f z₂) := by
  sorry

end NUMINAMATH_CALUDE_not_order_preserving_isomorphic_Z_Q_l2363_236300


namespace NUMINAMATH_CALUDE_equal_share_of_tea_l2363_236373

-- Define the total number of cups of tea
def total_cups : ℕ := 10

-- Define the number of people sharing the tea
def num_people : ℕ := 5

-- Define the number of cups each person receives
def cups_per_person : ℚ := total_cups / num_people

-- Theorem to prove
theorem equal_share_of_tea :
  cups_per_person = 2 := by sorry

end NUMINAMATH_CALUDE_equal_share_of_tea_l2363_236373


namespace NUMINAMATH_CALUDE_solve_for_t_l2363_236313

theorem solve_for_t (Q m h t : ℝ) (hQ : Q > 0) (hm : m ≠ 0) (hh : h > -2) :
  Q = m^2 / (2 + h)^t ↔ t = Real.log (m^2 / Q) / Real.log (2 + h) :=
sorry

end NUMINAMATH_CALUDE_solve_for_t_l2363_236313


namespace NUMINAMATH_CALUDE_horner_method_f_2_l2363_236320

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem horner_method_f_2 :
  f 2 = horner_eval [4, 0, -3, 2, 5, 1] 2 ∧ horner_eval [4, 0, -3, 2, 5, 1] 2 = 123 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_2_l2363_236320


namespace NUMINAMATH_CALUDE_sine_sum_problem_l2363_236377

theorem sine_sum_problem (α : ℝ) (h : Real.sin (π / 3 + α) + Real.sin α = (4 * Real.sqrt 3) / 5) :
  Real.sin (α + 7 * π / 6) = -4 / 5 := by sorry

end NUMINAMATH_CALUDE_sine_sum_problem_l2363_236377


namespace NUMINAMATH_CALUDE_smallest_divisible_by_first_five_primes_l2363_236341

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem smallest_divisible_by_first_five_primes :
  (∀ p ∈ first_five_primes, 2310 % p = 0) ∧
  (∀ n < 2310, ∃ p ∈ first_five_primes, n % p ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_first_five_primes_l2363_236341


namespace NUMINAMATH_CALUDE_circle_differences_l2363_236346

theorem circle_differences (n : ℕ) (a : ℕ → ℝ) 
  (h : ∀ i, |a i - a ((i + 1) % n)| ≥ 2 * |a i - a ((i + 2) % n)|) :
  ∀ i, |a i - a ((i + 3) % n)| ≥ |a i - a ((i + 2) % n)| :=
by sorry

end NUMINAMATH_CALUDE_circle_differences_l2363_236346


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l2363_236326

/-- The unknown rate of two blankets given specific conditions -/
theorem unknown_blanket_rate :
  let num_blankets_1 : ℕ := 5
  let price_1 : ℕ := 100
  let num_blankets_2 : ℕ := 5
  let price_2 : ℕ := 150
  let num_blankets_unknown : ℕ := 2
  let average_price : ℕ := 150
  let total_blankets : ℕ := num_blankets_1 + num_blankets_2 + num_blankets_unknown
  let known_cost : ℕ := num_blankets_1 * price_1 + num_blankets_2 * price_2
  ∃ (unknown_rate : ℕ),
    (known_cost + num_blankets_unknown * unknown_rate) / total_blankets = average_price ∧
    unknown_rate = 275 :=
by sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l2363_236326


namespace NUMINAMATH_CALUDE_train_average_speed_l2363_236301

/-- Calculates the average speed of a train journey with a stop -/
theorem train_average_speed 
  (distance1 : ℝ) 
  (time1 : ℝ) 
  (stop_time : ℝ) 
  (distance2 : ℝ) 
  (time2 : ℝ) 
  (h1 : distance1 = 240) 
  (h2 : time1 = 3) 
  (h3 : stop_time = 0.5) 
  (h4 : distance2 = 450) 
  (h5 : time2 = 5) :
  (distance1 + distance2) / (time1 + stop_time + time2) = (240 + 450) / (3 + 0.5 + 5) :=
by sorry

end NUMINAMATH_CALUDE_train_average_speed_l2363_236301


namespace NUMINAMATH_CALUDE_carries_strawberry_harvest_l2363_236314

/-- Represents the dimensions of Carrie's garden -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Represents the planting and yield information -/
structure PlantingInfo where
  plantsPerSquareFoot : ℕ
  strawberriesPerPlant : ℕ

/-- Calculates the expected strawberry harvest given garden dimensions and planting information -/
def expectedHarvest (garden : GardenDimensions) (info : PlantingInfo) : ℕ :=
  garden.length * garden.width * info.plantsPerSquareFoot * info.strawberriesPerPlant

/-- Theorem stating that Carrie's expected strawberry harvest is 3150 -/
theorem carries_strawberry_harvest :
  let garden := GardenDimensions.mk 7 9
  let info := PlantingInfo.mk 5 10
  expectedHarvest garden info = 3150 := by
  sorry

end NUMINAMATH_CALUDE_carries_strawberry_harvest_l2363_236314


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2363_236398

theorem quadratic_equation_solution (x : ℝ) (h1 : x^2 - 6*x = 0) (h2 : x ≠ 0) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2363_236398


namespace NUMINAMATH_CALUDE_brokerage_percentage_approx_l2363_236322

/-- Calculates the brokerage percentage given the cash realized and total amount --/
def brokerage_percentage (cash_realized : ℚ) (total_amount : ℚ) : ℚ :=
  ((cash_realized - total_amount) / total_amount) * 100

/-- Theorem stating that the brokerage percentage is approximately 0.24% --/
theorem brokerage_percentage_approx :
  let cash_realized : ℚ := 10425 / 100
  let total_amount : ℚ := 104
  abs (brokerage_percentage cash_realized total_amount - 24 / 100) < 1 / 1000 := by
  sorry

#eval brokerage_percentage (10425 / 100) 104

end NUMINAMATH_CALUDE_brokerage_percentage_approx_l2363_236322


namespace NUMINAMATH_CALUDE_point_transformation_l2363_236315

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Transformation from x-axis coordinates to y-axis coordinates -/
def transformToYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

/-- Theorem stating the transformation of point P -/
theorem point_transformation :
  ∃ (P : Point2D), P.x = 1 ∧ P.y = -2 → (transformToYAxis P).x = -1 ∧ (transformToYAxis P).y = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l2363_236315


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l2363_236331

/-- The total amount of oil leaked into the water -/
def total_oil_leaked (pre_repair_leak : ℕ) (during_repair_leak : ℕ) : ℕ :=
  pre_repair_leak + during_repair_leak

/-- Theorem stating the total amount of oil leaked -/
theorem oil_leak_calculation :
  total_oil_leaked 6522 5165 = 11687 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l2363_236331


namespace NUMINAMATH_CALUDE_equilateral_triangle_splitting_l2363_236317

/-- An equilateral triangle with side length 111 -/
def EquilateralTriangle : ℕ := 111

/-- The number of marked points in the triangle -/
def MarkedPoints : ℕ := 6216

/-- The number of linear sets -/
def LinearSets : ℕ := 111

/-- The number of ways to split the marked points into linear sets -/
def SplittingWays : ℕ := 2^4107

theorem equilateral_triangle_splitting (T : ℕ) (points : ℕ) (sets : ℕ) (ways : ℕ) :
  T = EquilateralTriangle →
  points = MarkedPoints →
  sets = LinearSets →
  ways = SplittingWays →
  ways = 2^(points / 3 * 2) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_splitting_l2363_236317


namespace NUMINAMATH_CALUDE_x_minus_y_equals_40_l2363_236344

theorem x_minus_y_equals_40 (x y : ℤ) (h1 : x + y = 24) (h2 : x = 32) : x - y = 40 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_40_l2363_236344


namespace NUMINAMATH_CALUDE_right_triangle_angle_calculation_l2363_236393

theorem right_triangle_angle_calculation (A B C : Real) 
  (h1 : A + B + C = 180) -- Sum of angles in a triangle is 180°
  (h2 : C = 90) -- Angle C is 90°
  (h3 : A = 35.5) -- Angle A is 35.5°
  : B = 54.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_calculation_l2363_236393


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2363_236375

/-- An isosceles triangle with side lengths 6 and 7 has a perimeter of either 19 or 20 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  (a = 6 ∧ b = 7) ∨ (a = 7 ∧ b = 6) →  -- Given side lengths
  ((a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)) →  -- Isosceles condition
  a + b + c = 19 ∨ a + b + c = 20 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2363_236375


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2363_236333

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  ((∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = -x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2363_236333


namespace NUMINAMATH_CALUDE_investment_difference_theorem_l2363_236319

/-- Calculates the difference in total amounts between two investment schemes after one year -/
def investment_difference (initial_a : ℝ) (initial_b : ℝ) (yield_a : ℝ) (yield_b : ℝ) : ℝ :=
  (initial_a * (1 + yield_a)) - (initial_b * (1 + yield_b))

/-- Theorem stating the difference in total amounts between schemes A and B after one year -/
theorem investment_difference_theorem :
  investment_difference 300 200 0.3 0.5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_theorem_l2363_236319
