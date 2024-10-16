import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_condition_l795_79563

theorem divisibility_condition (a b c : ℝ) :
  ∀ n : ℕ, (∃ k : ℝ, a^n * (b - c) + b^n * (c - a) + c^n * (a - b) = k * (a^2 + b^2 + c^2 + a*b + b*c + c*a)) ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l795_79563


namespace NUMINAMATH_CALUDE_dice_probability_l795_79504

def num_dice : ℕ := 6
def num_success : ℕ := 3
def prob_success : ℚ := 1/3
def prob_failure : ℚ := 2/3

theorem dice_probability : 
  (Nat.choose num_dice num_success : ℚ) * prob_success^num_success * prob_failure^(num_dice - num_success) = 160/729 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l795_79504


namespace NUMINAMATH_CALUDE_quadratic_minimum_l795_79599

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 10*x + 3 ≥ -22) ∧ (∃ x : ℝ, x^2 + 10*x + 3 = -22) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l795_79599


namespace NUMINAMATH_CALUDE_episode_length_l795_79551

theorem episode_length
  (num_episodes : ℕ)
  (watching_hours_per_day : ℕ)
  (total_days : ℕ)
  (h1 : num_episodes = 90)
  (h2 : watching_hours_per_day = 2)
  (h3 : total_days = 15) :
  (total_days * watching_hours_per_day * 60) / num_episodes = 20 :=
by sorry

end NUMINAMATH_CALUDE_episode_length_l795_79551


namespace NUMINAMATH_CALUDE_range_of_a_l795_79510

theorem range_of_a (a : ℝ) : 
  (∀ x > a, x * (x - 1) > 0) ↔ a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l795_79510


namespace NUMINAMATH_CALUDE_total_envelopes_l795_79552

def blue_envelopes : ℕ := 120
def yellow_envelopes : ℕ := blue_envelopes - 25
def green_envelopes : ℕ := 5 * yellow_envelopes

theorem total_envelopes : blue_envelopes + yellow_envelopes + green_envelopes = 690 := by
  sorry

end NUMINAMATH_CALUDE_total_envelopes_l795_79552


namespace NUMINAMATH_CALUDE_integral_of_power_function_l795_79536

theorem integral_of_power_function : 
  ∫ x in (0:ℝ)..2, (1 + 3*x)^4 = 1120.4 := by sorry

end NUMINAMATH_CALUDE_integral_of_power_function_l795_79536


namespace NUMINAMATH_CALUDE_kaili_circle_method_l795_79541

theorem kaili_circle_method (S : ℝ) (V : ℝ) (h : S = 4 * Real.pi / 9) :
  (2/3)^3 = 16 * V / 9 :=
sorry

end NUMINAMATH_CALUDE_kaili_circle_method_l795_79541


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l795_79595

/-- Given the equation (1+x)+(1+x)^2+...+(1+x)^5 = a₀+a₁(1-x)+a₂(1-x)^2+...+a₅(1-x)^5,
    prove that a₁+a₂+a₃+a₄+a₅ = -57 -/
theorem sum_of_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  ((1+x) + (1+x)^2 + (1+x)^3 + (1+x)^4 + (1+x)^5 = 
   a₀ + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + a₅*(1-x)^5) → 
  (a₁ + a₂ + a₃ + a₄ + a₅ = -57) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l795_79595


namespace NUMINAMATH_CALUDE_solution_set_inequality_proof_l795_79568

-- Define the function f(x) = |x-2|
def f (x : ℝ) : ℝ := |x - 2|

-- Part 1: Prove that the solution set of f(x) + f(x+1) ≤ 2 is [0.5, 2.5]
theorem solution_set (x : ℝ) : 
  (f x + f (x + 1) ≤ 2) ↔ (0.5 ≤ x ∧ x ≤ 2.5) := by sorry

-- Part 2: Prove that for all a < 0 and all x, f(ax) - af(x) ≥ f(2a)
theorem inequality_proof (a x : ℝ) (h : a < 0) : 
  f (a * x) - a * f x ≥ f (2 * a) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_proof_l795_79568


namespace NUMINAMATH_CALUDE_find_a_and_b_l795_79509

theorem find_a_and_b : ∃ a b : ℤ, 
  (a - b = 831) ∧ 
  (a = 21 * b + 11) ∧ 
  (a = 872) ∧ 
  (b = 41) := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l795_79509


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l795_79542

/-- Proves that the weight of a replaced person is 66 kg given the conditions of the problem -/
theorem weight_of_replaced_person
  (n : ℕ) -- number of people in the initial group
  (avg_increase : ℝ) -- increase in average weight
  (new_weight : ℝ) -- weight of the new person
  (h1 : n = 8) -- there are 8 persons initially
  (h2 : avg_increase = 2.5) -- the average weight increases by 2.5 kg
  (h3 : new_weight = 86) -- the weight of the new person is 86 kg
  : ∃ (replaced_weight : ℝ), replaced_weight = 66 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l795_79542


namespace NUMINAMATH_CALUDE_specific_trade_profit_l795_79538

/-- Represents a trading scenario for baseball cards -/
structure CardTrade where
  card_given_value : ℝ
  cards_given_count : ℕ
  card_received_value : ℝ

/-- Calculates the profit from a card trade -/
def trade_profit (trade : CardTrade) : ℝ :=
  trade.card_received_value - (trade.card_given_value * trade.cards_given_count)

/-- Theorem stating that the specific trade results in a $5 profit -/
theorem specific_trade_profit :
  let trade : CardTrade := {
    card_given_value := 8,
    cards_given_count := 2,
    card_received_value := 21
  }
  trade_profit trade = 5 := by sorry

end NUMINAMATH_CALUDE_specific_trade_profit_l795_79538


namespace NUMINAMATH_CALUDE_P_evaluation_l795_79576

/-- The polynomial P(x) = x^6 - 3x^3 - x^2 - x - 2 -/
def P (x : ℤ) : ℤ := x^6 - 3*x^3 - x^2 - x - 2

/-- P is irreducible over the integers -/
axiom P_irreducible : Irreducible P

theorem P_evaluation : P 3 = 634 := by
  sorry

end NUMINAMATH_CALUDE_P_evaluation_l795_79576


namespace NUMINAMATH_CALUDE_line_tangent_to_curve_l795_79597

/-- The line equation y = x + a is tangent to the curve y = x^3 - x^2 + 1 at the point (-1/3, 23/27) when a = 32/27 -/
theorem line_tangent_to_curve :
  let line (x : ℝ) := x + 32/27
  let curve (x : ℝ) := x^3 - x^2 + 1
  let tangent_point : ℝ × ℝ := (-1/3, 23/27)
  (∀ x, line x ≠ curve x ∨ x = tangent_point.1) ∧
  (line tangent_point.1 = curve tangent_point.1) ∧
  (HasDerivAt curve (line tangent_point.1) tangent_point.1) :=
by sorry


end NUMINAMATH_CALUDE_line_tangent_to_curve_l795_79597


namespace NUMINAMATH_CALUDE_integral_x_cos_x_over_sin_cubed_x_l795_79572

open Real

theorem integral_x_cos_x_over_sin_cubed_x (x : ℝ) :
  deriv (fun x => - (x + cos x * sin x) / (2 * sin x ^ 2)) x = 
    x * cos x / sin x ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_cos_x_over_sin_cubed_x_l795_79572


namespace NUMINAMATH_CALUDE_solve_for_x_l795_79575

theorem solve_for_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l795_79575


namespace NUMINAMATH_CALUDE_min_sum_product_2400_l795_79596

theorem min_sum_product_2400 (x y z : ℕ+) (h : x * y * z = 2400) :
  x + y + z ≥ 43 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_product_2400_l795_79596


namespace NUMINAMATH_CALUDE_triangle_side_length_l795_79569

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →  -- Convert 60° to radians
  a = Real.sqrt 31 →
  b = 6 →
  (c = 1 ∨ c = 5) →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l795_79569


namespace NUMINAMATH_CALUDE_bacteria_after_three_hours_l795_79564

/-- Represents the number of bacteria after a given number of half-hour periods. -/
def bacteria_population (half_hours : ℕ) : ℕ := 2^half_hours

/-- Theorem stating that after 3 hours (6 half-hour periods), the bacteria population will be 64. -/
theorem bacteria_after_three_hours : bacteria_population 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_after_three_hours_l795_79564


namespace NUMINAMATH_CALUDE_vector_equality_l795_79583

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def vector_b : Fin 2 → ℝ := ![1, -2]

theorem vector_equality (x : ℝ) : 
  ‖vector_a x + vector_b‖ = ‖vector_a x - vector_b‖ → x = 2 := by
sorry

end NUMINAMATH_CALUDE_vector_equality_l795_79583


namespace NUMINAMATH_CALUDE_no_circular_arrangement_with_conditions_l795_79589

theorem no_circular_arrangement_with_conditions : ¬ ∃ (a : Fin 9 → ℕ),
  (∀ i, a i ∈ Finset.range 9) ∧
  (∀ i, a i ≠ 0) ∧
  (∀ i j, i ≠ j → a i ≠ a j) ∧
  (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) % 3 = 0) ∧
  (∀ i, a i + a ((i + 1) % 9) + a ((i + 2) % 9) > 12) :=
by sorry

end NUMINAMATH_CALUDE_no_circular_arrangement_with_conditions_l795_79589


namespace NUMINAMATH_CALUDE_true_propositions_l795_79516

theorem true_propositions :
  (∃ x : ℝ, x^3 < 1) ∧
  (∃ x : ℝ, x^2 + 1 > 0) ∧
  ¬(∃ x : ℚ, x^2 = 2) ∧
  ¬(∃ x : ℕ, x^3 > x^2) :=
by sorry

end NUMINAMATH_CALUDE_true_propositions_l795_79516


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l795_79593

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := fun x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l795_79593


namespace NUMINAMATH_CALUDE_carbon_atoms_count_l795_79500

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Atomic weights of elements in atomic mass units (amu) -/
def atomic_weight : CompoundComposition → ℕ
  | ⟨c, h, o⟩ => 12 * c + 1 * h + 16 * o

/-- The compound has 1 Hydrogen and 1 Oxygen atom -/
def compound_constraints (comp : CompoundComposition) : Prop :=
  comp.hydrogen = 1 ∧ comp.oxygen = 1

/-- The molecular weight of the compound is 65 amu -/
def molecular_weight_constraint (comp : CompoundComposition) : Prop :=
  atomic_weight comp = 65

theorem carbon_atoms_count :
  ∀ comp : CompoundComposition,
    compound_constraints comp →
    molecular_weight_constraint comp →
    comp.carbon = 4 :=
by sorry

end NUMINAMATH_CALUDE_carbon_atoms_count_l795_79500


namespace NUMINAMATH_CALUDE_angle_measure_in_special_pentagon_l795_79573

/-- Given a pentagon PQRST where ∠P ≅ ∠R ≅ ∠T and ∠Q is supplementary to ∠S,
    the measure of ∠T is 120°. -/
theorem angle_measure_in_special_pentagon (P Q R S T : ℝ) : 
  P + Q + R + S + T = 540 →  -- Sum of angles in a pentagon
  Q + S = 180 →              -- ∠Q and ∠S are supplementary
  P = T ∧ R = T →            -- ∠P ≅ ∠R ≅ ∠T
  T = 120 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_pentagon_l795_79573


namespace NUMINAMATH_CALUDE_annie_distance_equals_22_l795_79523

def base_fare : ℚ := 2.5
def per_mile_rate : ℚ := 0.25
def mike_distance : ℚ := 42
def annie_toll : ℚ := 5

theorem annie_distance_equals_22 :
  ∃ (annie_distance : ℚ),
    base_fare + per_mile_rate * mike_distance =
    base_fare + annie_toll + per_mile_rate * annie_distance ∧
    annie_distance = 22 := by
  sorry

end NUMINAMATH_CALUDE_annie_distance_equals_22_l795_79523


namespace NUMINAMATH_CALUDE_sequence_2019th_term_l795_79528

theorem sequence_2019th_term (a : ℕ → ℤ) 
  (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n - 2) : 
  a 2019 = -4034 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2019th_term_l795_79528


namespace NUMINAMATH_CALUDE_jeff_cabinets_l795_79592

/-- The total number of cabinets after Jeff's installation --/
def total_cabinets (initial : ℕ) (counters : ℕ) (extra : ℕ) : ℕ :=
  initial + counters * (2 * initial) + extra

/-- Proof that Jeff has 26 cabinets after the installation --/
theorem jeff_cabinets : total_cabinets 3 3 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_jeff_cabinets_l795_79592


namespace NUMINAMATH_CALUDE_range_of_2alpha_minus_beta_l795_79520

theorem range_of_2alpha_minus_beta (α β : ℝ) 
  (h : -π/2 < α ∧ α < β ∧ β < π/2) : 
  -3*π/2 < 2*α - β ∧ 2*α - β < π/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2alpha_minus_beta_l795_79520


namespace NUMINAMATH_CALUDE_barts_earnings_l795_79534

/-- Represents the earnings for a single day --/
structure DayEarnings where
  rate : Rat
  questionsPerSurvey : Nat
  surveysCompleted : Nat

/-- Calculates the total earnings for a given day --/
def calculateDayEarnings (day : DayEarnings) : Rat :=
  day.rate * day.questionsPerSurvey * day.surveysCompleted

/-- Calculates the total earnings for three days --/
def calculateTotalEarnings (day1 day2 day3 : DayEarnings) : Rat :=
  calculateDayEarnings day1 + calculateDayEarnings day2 + calculateDayEarnings day3

/-- Theorem statement for Bart's earnings over three days --/
theorem barts_earnings :
  let monday : DayEarnings := { rate := 1/5, questionsPerSurvey := 10, surveysCompleted := 3 }
  let tuesday : DayEarnings := { rate := 1/4, questionsPerSurvey := 12, surveysCompleted := 4 }
  let wednesday : DayEarnings := { rate := 1/10, questionsPerSurvey := 15, surveysCompleted := 5 }
  calculateTotalEarnings monday tuesday wednesday = 51/2 := by
  sorry

end NUMINAMATH_CALUDE_barts_earnings_l795_79534


namespace NUMINAMATH_CALUDE_tribal_leadership_theorem_l795_79579

def tribal_leadership_arrangements (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) * Nat.choose (n - 4) 2 * Nat.choose (n - 6) 2 * Nat.choose (n - 8) 2

theorem tribal_leadership_theorem :
  tribal_leadership_arrangements 13 = 18604800 := by
  sorry

end NUMINAMATH_CALUDE_tribal_leadership_theorem_l795_79579


namespace NUMINAMATH_CALUDE_fair_haired_women_percentage_l795_79587

theorem fair_haired_women_percentage
  (total_employees : ℝ)
  (women_fair_hair_percentage : ℝ)
  (fair_hair_percentage : ℝ)
  (h1 : women_fair_hair_percentage = 28)
  (h2 : fair_hair_percentage = 70) :
  (women_fair_hair_percentage / fair_hair_percentage) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_fair_haired_women_percentage_l795_79587


namespace NUMINAMATH_CALUDE_rectangle_max_area_l795_79517

theorem rectangle_max_area (perimeter : ℝ) (h_perimeter : perimeter = 40) :
  let short_side := perimeter / 6
  let long_side := 2 * short_side
  let area := short_side * long_side
  area = 800 / 9 := by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l795_79517


namespace NUMINAMATH_CALUDE_student_count_l795_79522

theorem student_count (n : ℕ) (rank_top : ℕ) (rank_bottom : ℕ) 
  (h1 : rank_top = 30) 
  (h2 : rank_bottom = 30) : 
  n = 59 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l795_79522


namespace NUMINAMATH_CALUDE_base_number_proof_l795_79532

theorem base_number_proof (x n b : ℝ) 
  (h1 : n = x^(1/4))
  (h2 : n^b = 16)
  (h3 : b = 16.000000000000004) :
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_base_number_proof_l795_79532


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_is_integer_l795_79543

theorem right_triangle_hypotenuse_is_integer (n : ℤ) :
  let a : ℤ := 2 * n + 1
  let b : ℤ := 2 * n * (n + 1)
  let c : ℤ := 2 * n^2 + 2 * n + 1
  c^2 = a^2 + b^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_is_integer_l795_79543


namespace NUMINAMATH_CALUDE_thirteen_seventh_mod_nine_l795_79539

theorem thirteen_seventh_mod_nine (n : ℕ) : 
  13^7 % 9 = n ∧ 0 ≤ n ∧ n < 9 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_seventh_mod_nine_l795_79539


namespace NUMINAMATH_CALUDE_diamonds_formula_diamonds_G15_l795_79562

/-- The number of diamonds in figure G_n -/
def diamonds (n : ℕ+) : ℕ :=
  6 * n

/-- The theorem stating that the number of diamonds in G_n is 6n -/
theorem diamonds_formula (n : ℕ+) : diamonds n = 6 * n := by
  sorry

/-- Corollary: The number of diamonds in G_15 is 90 -/
theorem diamonds_G15 : diamonds 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_formula_diamonds_G15_l795_79562


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l795_79544

/-- The length of the path traveled by the center of a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 1 / Real.pi) : 
  let path_length := 2 * (r * Real.pi / 4) + r * Real.pi / 2
  path_length = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l795_79544


namespace NUMINAMATH_CALUDE_sequence_sum_l795_79526

theorem sequence_sum (seq : Fin 10 → ℝ) 
  (h1 : seq 2 = 5)
  (h2 : ∀ i : Fin 8, seq i + seq (i + 1) + seq (i + 2) = 25) :
  seq 0 + seq 9 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l795_79526


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l795_79525

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
  13 ∣ x →
  x = 728 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l795_79525


namespace NUMINAMATH_CALUDE_first_car_mpg_l795_79588

/-- Proves that the average miles per gallon of the first car is 27.5 given the conditions --/
theorem first_car_mpg (total_miles : ℝ) (total_gallons : ℝ) (second_car_mpg : ℝ) (first_car_gallons : ℝ) :
  total_miles = 1825 →
  total_gallons = 55 →
  second_car_mpg = 40 →
  first_car_gallons = 30 →
  (first_car_gallons * (total_miles - second_car_mpg * (total_gallons - first_car_gallons))) / 
    (first_car_gallons * total_miles) = 27.5 := by
sorry

end NUMINAMATH_CALUDE_first_car_mpg_l795_79588


namespace NUMINAMATH_CALUDE_partners_shares_correct_l795_79545

/-- Represents the investment ratio of partners A, B, and C -/
def investment_ratio : Fin 3 → ℕ
| 0 => 2  -- Partner A
| 1 => 3  -- Partner B
| 2 => 5  -- Partner C

/-- The total profit in rupees -/
def total_profit : ℕ := 22400

/-- Calculates a partner's share of the profit based on their investment ratio -/
def partner_share (i : Fin 3) : ℕ :=
  (investment_ratio i * total_profit) / (investment_ratio 0 + investment_ratio 1 + investment_ratio 2)

/-- Theorem stating that the partners' shares are correct -/
theorem partners_shares_correct :
  partner_share 0 = 4480 ∧
  partner_share 1 = 6720 ∧
  partner_share 2 = 11200 := by
  sorry


end NUMINAMATH_CALUDE_partners_shares_correct_l795_79545


namespace NUMINAMATH_CALUDE_golden_retriever_weight_at_8_years_l795_79561

/-- Calculates the weight of a golden retriever given its age and initial conditions -/
def goldenRetrieverWeight (initialWeight : ℕ) (firstYearGain : ℕ) (yearlyGain : ℕ) (yearlyLoss : ℕ) (age : ℕ) : ℕ :=
  let weightAfterFirstYear := initialWeight + firstYearGain
  let netYearlyGain := yearlyGain - yearlyLoss
  weightAfterFirstYear + (age - 1) * netYearlyGain

/-- Theorem stating the weight of a specific golden retriever at 8 years old -/
theorem golden_retriever_weight_at_8_years :
  goldenRetrieverWeight 3 15 11 3 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_golden_retriever_weight_at_8_years_l795_79561


namespace NUMINAMATH_CALUDE_linear_function_composition_l795_79550

theorem linear_function_composition (a b : ℝ) : 
  (∀ x : ℝ, (3 * (a * x + b) - 6) = 4 * x + 7) → 
  a + b = 17 / 3 := by
sorry

end NUMINAMATH_CALUDE_linear_function_composition_l795_79550


namespace NUMINAMATH_CALUDE_third_candidate_votes_l795_79560

theorem third_candidate_votes : 
  ∀ (total_votes : ℕ) (invalid_percentage : ℚ) (first_candidate_percentage : ℚ) (second_candidate_percentage : ℚ),
  total_votes = 10000 →
  invalid_percentage = 1/4 →
  first_candidate_percentage = 1/2 →
  second_candidate_percentage = 3/10 →
  ∃ (third_candidate_votes : ℕ),
    third_candidate_votes = total_votes * (1 - invalid_percentage) - 
      (total_votes * (1 - invalid_percentage) * first_candidate_percentage + 
       total_votes * (1 - invalid_percentage) * second_candidate_percentage) ∧
    third_candidate_votes = 1500 :=
by sorry

end NUMINAMATH_CALUDE_third_candidate_votes_l795_79560


namespace NUMINAMATH_CALUDE_fraction_simplification_l795_79584

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 3 + Real.sqrt 48) = (5 * Real.sqrt 3) / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l795_79584


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l795_79577

theorem complex_magnitude_theorem (z : ℂ) (h : z^2 - 2 * Complex.abs z + 3 = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l795_79577


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l795_79529

theorem max_value_x_plus_2y (x y : ℝ) (h : x^2 - 2*x + 4*y = 5) : 
  (∃ (z : ℝ), x + 2*y ≤ z) ∧ (∀ (w : ℝ), x + 2*y ≤ w → 9/2 ≤ w) :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l795_79529


namespace NUMINAMATH_CALUDE_x_value_from_equation_l795_79505

theorem x_value_from_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 6 * x^2 + 18 * x * y = 2 * x^3 + 3 * x^2 * y^2) : 
  x = (3 + Real.sqrt 153) / 4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_from_equation_l795_79505


namespace NUMINAMATH_CALUDE_triangle_circumradius_l795_79530

/-- Given a triangle ABC with area S = (1/2) * sin A * sin B * sin C, 
    the radius R of its circumcircle is equal to 1/2. -/
theorem triangle_circumradius (A B C : ℝ) (a b c : ℝ) (S : ℝ) (R : ℝ) : 
  S = (1/2) * Real.sin A * Real.sin B * Real.sin C →
  R = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l795_79530


namespace NUMINAMATH_CALUDE_sin_cube_identity_l795_79555

theorem sin_cube_identity (θ : ℝ) : 
  Real.sin θ ^ 3 = -(1/4) * Real.sin (3 * θ) + (3/4) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l795_79555


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l795_79598

theorem binomial_coefficient_two (n : ℕ+) : (n.val.choose 2) = n.val * (n.val - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l795_79598


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l795_79524

theorem polynomial_evaluation :
  let x : ℤ := -2
  x^4 + x^3 + x^2 + x + 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l795_79524


namespace NUMINAMATH_CALUDE_train_speed_l795_79574

/-- The speed of a train given its length and time to cross an electric pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 1000) (h2 : time = 200) :
  length / time = 5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l795_79574


namespace NUMINAMATH_CALUDE_dorothy_taxes_l795_79565

/-- Calculates the amount left after taxes given an annual income and tax rate. -/
def amountLeftAfterTaxes (annualIncome : ℝ) (taxRate : ℝ) : ℝ :=
  annualIncome * (1 - taxRate)

/-- Proves that given an annual income of $60,000 and a tax rate of 18%, 
    the amount left after taxes is $49,200. -/
theorem dorothy_taxes : 
  amountLeftAfterTaxes 60000 0.18 = 49200 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_taxes_l795_79565


namespace NUMINAMATH_CALUDE_subtraction_proof_l795_79546

theorem subtraction_proof : 6236 - 797 = 5439 := by sorry

end NUMINAMATH_CALUDE_subtraction_proof_l795_79546


namespace NUMINAMATH_CALUDE_solution_sets_l795_79537

def f (a x : ℝ) := x^2 - (a - 1) * x - a

theorem solution_sets (a : ℝ) :
  (a = 2 → {x : ℝ | f 2 x < 0} = {x : ℝ | -1 < x ∧ x < 2}) ∧
  (a > -1 → {x : ℝ | f a x > 0} = {x : ℝ | x < -1 ∨ x > a}) ∧
  (a = -1 → {x : ℝ | f (-1) x > 0} = {x : ℝ | x < -1 ∨ x > -1}) ∧
  (a < -1 → {x : ℝ | f a x > 0} = {x : ℝ | x < a ∨ x > -1}) :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_l795_79537


namespace NUMINAMATH_CALUDE_passengers_scientific_notation_l795_79518

/-- Represents the number of passengers in millions -/
def passengers : ℝ := 1.446

/-- Represents the scientific notation of the number of passengers -/
def scientific_notation : ℝ := 1.446 * (10 ^ 6)

/-- Theorem stating that the number of passengers in millions 
    is equal to its scientific notation representation -/
theorem passengers_scientific_notation : 
  passengers * 1000000 = scientific_notation := by sorry

end NUMINAMATH_CALUDE_passengers_scientific_notation_l795_79518


namespace NUMINAMATH_CALUDE_compound_ratio_l795_79503

theorem compound_ratio (total_weight : ℝ) (weight_B : ℝ) :
  total_weight = 108 →
  weight_B = 90 →
  let weight_A := total_weight - weight_B
  (weight_A / weight_B) = (1 / 5 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_compound_ratio_l795_79503


namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l795_79553

theorem sum_of_number_and_its_square : 
  let n : ℕ := 8
  (n + n^2) = 72 := by sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l795_79553


namespace NUMINAMATH_CALUDE_total_eggs_supplied_weekly_l795_79571

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens supplied to Store A daily -/
def dozens_to_store_A : ℕ := 5

/-- The number of eggs supplied to Store B daily -/
def eggs_to_store_B : ℕ := 30

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total number of eggs supplied to both stores in a week is 630 -/
theorem total_eggs_supplied_weekly : 
  (dozens_to_store_A * eggs_per_dozen + eggs_to_store_B) * days_in_week = 630 := by
sorry

end NUMINAMATH_CALUDE_total_eggs_supplied_weekly_l795_79571


namespace NUMINAMATH_CALUDE_rain_duration_l795_79578

/-- Calculates the number of minutes it rained to fill a tank given initial conditions -/
theorem rain_duration (initial_water : ℕ) (evaporated : ℕ) (drained : ℕ) (rain_rate : ℕ) (final_water : ℕ) : 
  initial_water = 6000 →
  evaporated = 2000 →
  drained = 3500 →
  rain_rate = 350 →
  final_water = 1550 →
  (final_water - (initial_water - evaporated - drained)) / (rain_rate / 10) * 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rain_duration_l795_79578


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l795_79515

theorem arithmetic_sequence_squares (m : ℤ) : 
  (∃ (a d : ℝ), 
    (16 + m : ℝ) = (a : ℝ) ^ 2 ∧ 
    (100 + m : ℝ) = (a + d) ^ 2 ∧ 
    (484 + m : ℝ) = (a + 2 * d) ^ 2) ↔ 
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l795_79515


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l795_79513

theorem absolute_value_inequality (x : ℝ) : |x - 3| ≥ |x| ↔ x ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l795_79513


namespace NUMINAMATH_CALUDE_class_size_l795_79507

theorem class_size :
  let both := 5  -- number of people who like both baseball and football
  let baseball_only := 2  -- number of people who only like baseball
  let football_only := 3  -- number of people who only like football
  let neither := 6  -- number of people who like neither baseball nor football
  both + baseball_only + football_only + neither = 16 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l795_79507


namespace NUMINAMATH_CALUDE_trapezoid_not_axisymmetric_l795_79582

-- Define the shapes
inductive Shape
  | Angle
  | Rectangle
  | Trapezoid
  | Rhombus

-- Define the property of being axisymmetric
def is_axisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Angle => True
  | Shape.Rectangle => True
  | Shape.Rhombus => True
  | Shape.Trapezoid => false

-- Theorem stating that trapezoid is the only shape not necessarily axisymmetric
theorem trapezoid_not_axisymmetric :
  ∀ (s : Shape), ¬is_axisymmetric s ↔ s = Shape.Trapezoid :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_not_axisymmetric_l795_79582


namespace NUMINAMATH_CALUDE_range_of_a_l795_79508

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ 
    Real.exp x * (y₁ - x) - a * Real.exp (2 * y₁ - x) = 0 ∧
    Real.exp x * (y₂ - x) - a * Real.exp (2 * y₂ - x) = 0) →
  0 < a ∧ a < 1 / (2 * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l795_79508


namespace NUMINAMATH_CALUDE_special_function_at_three_l795_79548

/-- An increasing function satisfying a specific functional equation -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ 
  (∀ x, f (f x - 2^x) = 3)

/-- The value of the special function at 3 is 9 -/
theorem special_function_at_three 
  (f : ℝ → ℝ) (hf : SpecialFunction f) : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_three_l795_79548


namespace NUMINAMATH_CALUDE_people_who_left_line_l795_79558

theorem people_who_left_line (initial_people final_people joined_people : ℕ) 
  (h1 : initial_people = 30)
  (h2 : final_people = 25)
  (h3 : joined_people = 5)
  (h4 : final_people = initial_people - (people_who_left : ℕ) + joined_people) :
  people_who_left = 10 := by
sorry

end NUMINAMATH_CALUDE_people_who_left_line_l795_79558


namespace NUMINAMATH_CALUDE_vertex_angle_is_40_l795_79535

-- Define an isosceles triangle
structure IsoscelesTriangle where
  vertexAngle : ℝ
  baseAngle : ℝ
  sum_of_angles : vertexAngle + 2 * baseAngle = 180
  base_angle_relation : baseAngle = vertexAngle + 30

-- Theorem statement
theorem vertex_angle_is_40 (t : IsoscelesTriangle) : t.vertexAngle = 40 :=
by sorry

end NUMINAMATH_CALUDE_vertex_angle_is_40_l795_79535


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l795_79567

def sum_of_range (a b : ℕ) : ℕ := 
  ((b - a + 1) * (a + b)) / 2

def count_even_in_range (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_theorem : 
  sum_of_range 50 60 + count_even_in_range 50 60 = 611 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l795_79567


namespace NUMINAMATH_CALUDE_ball_probabilities_l795_79531

/-- The total number of balls in the bag -/
def total_balls : ℕ := 12

/-- The number of red balls initially in the bag -/
def red_balls : ℕ := 4

/-- The number of black balls in the bag -/
def black_balls : ℕ := 8

/-- The probability of drawing a black ball after removing m red balls -/
def prob_black (m : ℕ) : ℚ :=
  black_balls / (total_balls - m)

/-- The probability of drawing a black ball after removing n red balls -/
def prob_black_n (n : ℕ) : ℚ :=
  black_balls / (total_balls - n)

theorem ball_probabilities :
  (prob_black 4 = 1) ∧
  (prob_black 2 > 0 ∧ prob_black 2 < 1) ∧
  (prob_black 3 > 0 ∧ prob_black 3 < 1) ∧
  (prob_black_n 3 = 8/9) :=
sorry

end NUMINAMATH_CALUDE_ball_probabilities_l795_79531


namespace NUMINAMATH_CALUDE_evaluate_expression_l795_79557

theorem evaluate_expression : (3025^2 : ℝ) / (305^2 - 295^2) = 1525.10417 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l795_79557


namespace NUMINAMATH_CALUDE_square_roots_of_four_l795_79533

theorem square_roots_of_four :
  {y : ℝ | y ^ 2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_roots_of_four_l795_79533


namespace NUMINAMATH_CALUDE_min_fourth_integer_l795_79501

theorem min_fourth_integer (A B C D : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 →
  A = 3 * B →
  B = C - 2 →
  (A + B + C + D) / 4 = 16 →
  D ≥ 52 :=
by sorry

end NUMINAMATH_CALUDE_min_fourth_integer_l795_79501


namespace NUMINAMATH_CALUDE_intersecting_circles_common_chord_l795_79581

/-- Two intersecting circles with given radii and distance between centers have a common chord of length 10 -/
theorem intersecting_circles_common_chord 
  (R : ℝ) 
  (r : ℝ) 
  (d : ℝ) 
  (h1 : R = 13) 
  (h2 : r = 5) 
  (h3 : d = 12) :
  ∃ (chord_length : ℝ), 
    chord_length = 10 ∧ 
    chord_length = 2 * R * Real.sqrt (1 - ((R^2 + d^2 - r^2) / (2 * R * d))^2) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_common_chord_l795_79581


namespace NUMINAMATH_CALUDE_integer_sum_difference_product_square_difference_l795_79554

theorem integer_sum_difference_product_square_difference 
  (a b : ℕ+) 
  (sum_eq : a + b = 40)
  (diff_eq : a - b = 8) : 
  a * b = 384 ∧ a^2 - b^2 = 320 := by
sorry

end NUMINAMATH_CALUDE_integer_sum_difference_product_square_difference_l795_79554


namespace NUMINAMATH_CALUDE_shuai_shuai_memorization_l795_79591

/-- The number of words memorized by Shuai Shuai over 7 days -/
def total_words : ℕ := 198

/-- The number of words memorized in the first 3 days -/
def first_three_days : ℕ := 44

/-- The number of words memorized on the fourth day -/
def fourth_day : ℕ := 10

/-- The number of words memorized in the last 3 days -/
def last_three_days : ℕ := 45

/-- Theorem stating the conditions and the result -/
theorem shuai_shuai_memorization :
  (first_three_days + fourth_day + last_three_days = total_words) ∧
  (first_three_days = (4 : ℚ) / 5 * (fourth_day + last_three_days)) ∧
  (first_three_days + fourth_day = (6 : ℚ) / 5 * last_three_days) ∧
  (total_words > 100) ∧
  (total_words < 200) :=
by sorry

end NUMINAMATH_CALUDE_shuai_shuai_memorization_l795_79591


namespace NUMINAMATH_CALUDE_function_properties_l795_79521

noncomputable section

variable (a : ℝ)
variable (f : ℝ → ℝ)

def has_extremum_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a b, ∀ y ∈ Set.Ioo a b, f x ≤ f y ∨ f x ≥ f y

def has_unique_zero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

theorem function_properties (h1 : a > 0) 
    (h2 : f = λ x => Real.exp (2*x) + 2 / Real.exp x - a*x) :
  (has_extremum_in f 0 1 → 0 < a ∧ a < 2 * Real.exp 2 - 2 / Real.exp 1) ∧
  (has_unique_zero f → ∃ x₀, f x₀ = 0 ∧ Real.log 2 < x₀ ∧ x₀ < 1) := by
  sorry

end

end NUMINAMATH_CALUDE_function_properties_l795_79521


namespace NUMINAMATH_CALUDE_pams_age_l795_79556

/-- Proves that Pam's current age is 5 years, given the conditions of the problem -/
theorem pams_age (pam_age rena_age : ℕ) 
  (h1 : pam_age = rena_age / 2)
  (h2 : rena_age + 10 = (pam_age + 10) + 5) : 
  pam_age = 5 := by sorry

end NUMINAMATH_CALUDE_pams_age_l795_79556


namespace NUMINAMATH_CALUDE_expression_evaluation_l795_79559

theorem expression_evaluation (b : ℚ) (h : b = 4/3) : 
  (7*b^2 - 15*b + 5) * (3*b - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l795_79559


namespace NUMINAMATH_CALUDE_product_of_roots_equation_l795_79502

theorem product_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, (a - 4) * (a - 2) + (a - 2) * (a - 6) = 0 ∧ 
               (b - 4) * (b - 2) + (b - 2) * (b - 6) = 0 ∧ 
               a ≠ b ∧ 
               a * b = 10) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_equation_l795_79502


namespace NUMINAMATH_CALUDE_chris_birthday_savings_l795_79549

/-- Chris's birthday savings problem -/
theorem chris_birthday_savings 
  (grandmother : ℕ) 
  (aunt_uncle : ℕ) 
  (parents : ℕ) 
  (total_now : ℕ) 
  (h1 : grandmother = 25)
  (h2 : aunt_uncle = 20)
  (h3 : parents = 75)
  (h4 : total_now = 279) :
  total_now - (grandmother + aunt_uncle + parents) = 159 := by
sorry

end NUMINAMATH_CALUDE_chris_birthday_savings_l795_79549


namespace NUMINAMATH_CALUDE_tim_nickels_count_l795_79547

/-- The number of nickels Tim had initially -/
def initial_nickels : ℕ := 9

/-- The number of nickels Tim received from his dad -/
def received_nickels : ℕ := 3

/-- The total number of nickels Tim has after receiving coins from his dad -/
def total_nickels : ℕ := initial_nickels + received_nickels

theorem tim_nickels_count : total_nickels = 12 := by
  sorry

end NUMINAMATH_CALUDE_tim_nickels_count_l795_79547


namespace NUMINAMATH_CALUDE_sean_purchase_cost_l795_79540

/-- The cost of items in Sean's purchase -/
def CostCalculation (soda_price : ℝ) : Prop :=
  let soup_price := 3 * soda_price
  let sandwich_price := 3 * soup_price
  (3 * soda_price) + (2 * soup_price) + sandwich_price = 18

/-- Theorem stating the total cost of Sean's purchase -/
theorem sean_purchase_cost :
  CostCalculation 1 := by
  sorry

end NUMINAMATH_CALUDE_sean_purchase_cost_l795_79540


namespace NUMINAMATH_CALUDE_mary_needs_four_cups_l795_79585

/-- The number of cups of flour Mary needs to add to her cake -/
def additional_flour (total_required : ℕ) (already_added : ℕ) : ℕ :=
  total_required - already_added

/-- Proof that Mary needs to add 4 more cups of flour -/
theorem mary_needs_four_cups : additional_flour 10 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mary_needs_four_cups_l795_79585


namespace NUMINAMATH_CALUDE_min_cos_for_valid_sqrt_l795_79586

theorem min_cos_for_valid_sqrt (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (2 * Real.cos x - 1)) ↔ Real.cos x ≥ (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_min_cos_for_valid_sqrt_l795_79586


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l795_79580

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x + 3 and y = (3k)x + 1 are parallel -/
theorem parallel_lines_k_value : 
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 1) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l795_79580


namespace NUMINAMATH_CALUDE_pedal_triangle_perimeter_and_area_l795_79527

/-- Given a triangle with circumradius R and angles α, β, and γ,
    this theorem states the formulas for the perimeter and twice the area of its pedal triangle. -/
theorem pedal_triangle_perimeter_and_area 
  (R : ℝ) (α β γ : ℝ) : 
  ∃ (k t : ℝ),
    k = 4 * R * Real.sin α * Real.sin β * Real.sin γ ∧ 
    2 * t = R^2 * Real.sin (2*α) * Real.sin (2*β) * Real.sin (2*γ) := by
  sorry

end NUMINAMATH_CALUDE_pedal_triangle_perimeter_and_area_l795_79527


namespace NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_triangle_l795_79590

def pascal_triangle (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def is_in_pascal_triangle (x : ℕ) : Prop :=
  ∃ n k, pascal_triangle n k = x

def is_four_digit (x : ℕ) : Prop :=
  1000 ≤ x ∧ x < 10000

theorem smallest_four_digit_in_pascal_triangle :
  (is_in_pascal_triangle 1000) ∧
  (∀ x, is_in_pascal_triangle x → is_four_digit x → 1000 ≤ x) :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_triangle_l795_79590


namespace NUMINAMATH_CALUDE_selling_price_theorem_l795_79566

/-- The selling price of an article that results in a loss, given the cost price and a selling price that results in a profit. -/
def selling_price_with_loss (cost_price profit_price : ℕ) : ℕ :=
  2 * cost_price - profit_price

theorem selling_price_theorem (cost_price profit_price : ℕ) 
  (h1 : cost_price = 64)
  (h2 : profit_price = 86)
  (h3 : profit_price > cost_price) :
  selling_price_with_loss cost_price profit_price = 42 := by
  sorry

#eval selling_price_with_loss 64 86  -- Should output 42

end NUMINAMATH_CALUDE_selling_price_theorem_l795_79566


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l795_79519

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_prod : a 4 * a 5 * a 6 = 27) :
  a 1 * a 9 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l795_79519


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l795_79594

theorem quadratic_inequality_condition (x : ℝ) : 
  (2 * x^2 - 5 * x - 3 < 0) ↔ (-1/2 < x ∧ x < 3) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l795_79594


namespace NUMINAMATH_CALUDE_six_point_triangle_l795_79506

/-- A point in a plane represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Calculates the angle between three points -/
noncomputable def angle (p q r : Point) : ℝ :=
  sorry -- Definition of angle calculation

/-- Theorem: Given six points on a plane where no three are collinear,
    there exists a subset of three points forming a triangle with at least
    one angle less than or equal to 30 degrees -/
theorem six_point_triangle (points : Fin 6 → Point)
    (h : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) :
    ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (angle (points i) (points j) (points k) ≤ 30 ∨
     angle (points j) (points k) (points i) ≤ 30 ∨
     angle (points k) (points i) (points j) ≤ 30) :=
  sorry

end NUMINAMATH_CALUDE_six_point_triangle_l795_79506


namespace NUMINAMATH_CALUDE_square_diff_fourth_power_l795_79514

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_fourth_power_l795_79514


namespace NUMINAMATH_CALUDE_philips_farm_animals_l795_79512

/-- The number of animals on Philip's farm --/
def total_animals (cows ducks pigs : ℕ) : ℕ := cows + ducks + pigs

/-- Theorem stating the total number of animals on Philip's farm --/
theorem philips_farm_animals :
  ∀ (cows ducks pigs : ℕ),
  cows = 20 →
  ducks = cows + cows / 2 →
  pigs = (cows + ducks) / 5 →
  total_animals cows ducks pigs = 60 := by
sorry

end NUMINAMATH_CALUDE_philips_farm_animals_l795_79512


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l795_79511

-- Define the types for lines and relationships
def Line : Type := ℝ → ℝ → Prop
def Perpendicular (l₁ l₂ : Line) : Prop := sorry
def Parallel (l₁ l₂ : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_necessary_not_sufficient 
  (a b c : Line) (h : Perpendicular a b) : 
  (∀ (a b c : Line), Parallel b c → Perpendicular a c) ∧ 
  (∃ (a b c : Line), Perpendicular a c ∧ ¬Parallel b c) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l795_79511


namespace NUMINAMATH_CALUDE_quadratic_root_range_l795_79570

theorem quadratic_root_range (a : ℝ) :
  (∃ x y : ℝ, x^2 + (a^2 - 1)*x + a - 2 = 0 ∧ x > 1 ∧ y < 1 ∧ y^2 + (a^2 - 1)*y + a - 2 = 0) →
  a ∈ Set.Ioo (-2 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l795_79570
