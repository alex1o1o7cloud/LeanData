import Mathlib

namespace NUMINAMATH_CALUDE_total_teaching_time_is_5160_l2399_239986

-- Define the number of classes and durations for Eduardo
def eduardo_math_classes : ℕ := 3
def eduardo_science_classes : ℕ := 4
def eduardo_history_classes : ℕ := 2
def eduardo_math_duration : ℕ := 60
def eduardo_science_duration : ℕ := 90
def eduardo_history_duration : ℕ := 120

-- Define Frankie's multiplier
def frankie_multiplier : ℕ := 2

-- Define Georgina's multiplier and class durations
def georgina_multiplier : ℕ := 3
def georgina_math_duration : ℕ := 80
def georgina_science_duration : ℕ := 100
def georgina_history_duration : ℕ := 150

-- Calculate total teaching time
def total_teaching_time : ℕ :=
  -- Eduardo's teaching time
  (eduardo_math_classes * eduardo_math_duration +
   eduardo_science_classes * eduardo_science_duration +
   eduardo_history_classes * eduardo_history_duration) +
  -- Frankie's teaching time
  (frankie_multiplier * eduardo_math_classes * eduardo_math_duration +
   frankie_multiplier * eduardo_science_classes * eduardo_science_duration +
   frankie_multiplier * eduardo_history_classes * eduardo_history_duration) +
  -- Georgina's teaching time
  (georgina_multiplier * eduardo_math_classes * georgina_math_duration +
   georgina_multiplier * eduardo_science_classes * georgina_science_duration +
   georgina_multiplier * eduardo_history_classes * georgina_history_duration)

-- Theorem statement
theorem total_teaching_time_is_5160 : total_teaching_time = 5160 := by
  sorry

end NUMINAMATH_CALUDE_total_teaching_time_is_5160_l2399_239986


namespace NUMINAMATH_CALUDE_six_color_theorem_l2399_239997

-- Define a Map as a structure with countries and their adjacencies
structure Map where
  countries : Set (Nat)
  adjacent : countries → countries → Prop

-- Define a Coloring as a function from countries to colors
def Coloring (m : Map) := m.countries → Fin 6

-- Define what it means for a coloring to be proper
def IsProperColoring (m : Map) (c : Coloring m) : Prop :=
  ∀ x y : m.countries, m.adjacent x y → c x ≠ c y

-- State the theorem
theorem six_color_theorem (m : Map) : 
  ∃ c : Coloring m, IsProperColoring m c :=
sorry

end NUMINAMATH_CALUDE_six_color_theorem_l2399_239997


namespace NUMINAMATH_CALUDE_wendy_ribbon_left_l2399_239951

/-- The amount of ribbon Wendy has left after using some for wrapping presents -/
def ribbon_left (initial : ℕ) (used : ℕ) : ℕ :=
  initial - used

/-- Theorem: Given Wendy bought 84 inches of ribbon and used 46 inches, 
    the amount of ribbon left is 38 inches -/
theorem wendy_ribbon_left : 
  ribbon_left 84 46 = 38 := by
  sorry

end NUMINAMATH_CALUDE_wendy_ribbon_left_l2399_239951


namespace NUMINAMATH_CALUDE_equation_solution_l2399_239934

theorem equation_solution : 
  ∃ x₁ x₂ : ℚ, x₁ = 8/3 ∧ x₂ = 2 ∧ 
  (∀ x : ℚ, x^2 - 6*x + 9 = (5 - 2*x)^2 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2399_239934


namespace NUMINAMATH_CALUDE_cereal_consumption_time_l2399_239959

/-- The time taken for two people to consume a given amount of cereal together,
    given their individual consumption rates. -/
def time_to_consume (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Mr. Fat's cereal consumption rate in pounds per minute -/
def fat_rate : ℚ := 1 / 25

/-- Mr. Thin's cereal consumption rate in pounds per minute -/
def thin_rate : ℚ := 1 / 35

/-- The amount of cereal to be consumed in pounds -/
def cereal_amount : ℚ := 5

theorem cereal_consumption_time :
  ∃ (t : ℚ), abs (t - time_to_consume fat_rate thin_rate cereal_amount) < 1 ∧
             t = 73 := by sorry

end NUMINAMATH_CALUDE_cereal_consumption_time_l2399_239959


namespace NUMINAMATH_CALUDE_correct_calculation_l2399_239955

theorem correct_calculation (a : ℝ) : 3 * a^2 - 2 * a^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2399_239955


namespace NUMINAMATH_CALUDE_prime_factors_equation_l2399_239925

theorem prime_factors_equation (x : ℕ) : 22 + x + 2 = 29 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_equation_l2399_239925


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2399_239954

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, if they are parallel, then m = -3 -/
theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (1 + m, 1 - m)
  parallel a b → m = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2399_239954


namespace NUMINAMATH_CALUDE_zoo_field_trip_zoo_field_trip_result_l2399_239992

/-- Calculates the number of individuals left at the zoo after a field trip -/
theorem zoo_field_trip (students_per_class : ℕ) (num_classes : ℕ) (parent_chaperones : ℕ) 
  (teachers : ℕ) (students_left : ℕ) (chaperones_left : ℕ) : ℕ :=
  let initial_total := students_per_class * num_classes + parent_chaperones + teachers
  let left_total := students_left + chaperones_left
  initial_total - left_total

/-- Proves that the number of individuals left at the zoo is 15 -/
theorem zoo_field_trip_result : 
  zoo_field_trip 10 2 5 2 10 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_zoo_field_trip_zoo_field_trip_result_l2399_239992


namespace NUMINAMATH_CALUDE_inequality_range_l2399_239988

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 3| - |x + 1| ≤ a^2 - 3*a) ↔ (a ≤ -1 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l2399_239988


namespace NUMINAMATH_CALUDE_det_of_matrix_l2399_239921

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![8, 5; -2, 3]

theorem det_of_matrix : Matrix.det matrix = 34 := by sorry

end NUMINAMATH_CALUDE_det_of_matrix_l2399_239921


namespace NUMINAMATH_CALUDE_crossed_out_digit_l2399_239956

theorem crossed_out_digit (N : Nat) (x : Nat) : 
  (N % 9 = 3) → 
  (x ≤ 9) →
  (∃ a b : Nat, N = a * 10 + x + b ∧ b < 10^9) →
  ((N - x) % 9 = 7) →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_crossed_out_digit_l2399_239956


namespace NUMINAMATH_CALUDE_no_cracked_seashells_l2399_239964

theorem no_cracked_seashells (tom_shells fred_shells total_shells : ℕ) 
  (h1 : tom_shells = 15)
  (h2 : fred_shells = 43)
  (h3 : total_shells = 58)
  (h4 : tom_shells + fred_shells = total_shells) :
  total_shells - (tom_shells + fred_shells) = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_cracked_seashells_l2399_239964


namespace NUMINAMATH_CALUDE_price_restoration_l2399_239995

theorem price_restoration (original_price : ℝ) (original_price_positive : original_price > 0) : 
  let reduced_price := original_price * (1 - 0.15)
  let restoration_factor := original_price / reduced_price
  let percentage_increase := (restoration_factor - 1) * 100
  ∃ ε > 0, abs (percentage_increase - 17.65) < ε ∧ ε < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_price_restoration_l2399_239995


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2399_239922

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 9 = 0 ∧ 
  x^4 - 3*x^3 - 9*x^2 + 27*x - 8 = (65 + 81 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2399_239922


namespace NUMINAMATH_CALUDE_percent_less_u_than_y_l2399_239989

theorem percent_less_u_than_y 
  (w u y z : ℝ) 
  (hw : w = 0.60 * u) 
  (hz1 : z = 0.54 * y) 
  (hz2 : z = 1.50 * w) : 
  u = 0.60 * y := by sorry

end NUMINAMATH_CALUDE_percent_less_u_than_y_l2399_239989


namespace NUMINAMATH_CALUDE_yellow_green_weight_difference_l2399_239918

/-- The weight difference between two blocks -/
def weight_difference (yellow_weight green_weight : Real) : Real :=
  yellow_weight - green_weight

/-- Theorem stating the weight difference between yellow and green blocks -/
theorem yellow_green_weight_difference :
  let yellow_weight : Real := 0.6
  let green_weight : Real := 0.4
  weight_difference yellow_weight green_weight = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_green_weight_difference_l2399_239918


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l2399_239910

theorem smallest_five_digit_congruent_to_3_mod_17 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 3 → n ≥ 10018 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l2399_239910


namespace NUMINAMATH_CALUDE_probability_at_least_one_die_shows_one_or_ten_l2399_239916

/-- The number of sides on each die -/
def num_sides : ℕ := 10

/-- The number of outcomes where a die doesn't show 1 or 10 -/
def favorable_outcomes_per_die : ℕ := num_sides - 2

/-- The total number of outcomes when rolling two dice -/
def total_outcomes : ℕ := num_sides * num_sides

/-- The number of outcomes where neither die shows 1 or 10 -/
def unfavorable_outcomes : ℕ := favorable_outcomes_per_die * favorable_outcomes_per_die

/-- The number of favorable outcomes (at least one die shows 1 or 10) -/
def favorable_outcomes : ℕ := total_outcomes - unfavorable_outcomes

/-- The probability of at least one die showing 1 or 10 -/
theorem probability_at_least_one_die_shows_one_or_ten :
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_die_shows_one_or_ten_l2399_239916


namespace NUMINAMATH_CALUDE_M_subset_N_l2399_239904

-- Define the sets M and N
def M : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 4}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 8 - 1 / 4}

-- Theorem to prove
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l2399_239904


namespace NUMINAMATH_CALUDE_twentieth_term_da_yan_l2399_239976

def da_yan_sequence (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 2
  else
    (n^2 - 1) / 2

theorem twentieth_term_da_yan (n : ℕ) : da_yan_sequence 20 = 200 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_da_yan_l2399_239976


namespace NUMINAMATH_CALUDE_number_difference_l2399_239979

theorem number_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2399_239979


namespace NUMINAMATH_CALUDE_product_equality_l2399_239994

theorem product_equality : 469111111 * 99999999 = 46911111053088889 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2399_239994


namespace NUMINAMATH_CALUDE_original_number_l2399_239985

theorem original_number : ∃ x : ℕ, 100 * x - x = 1980 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l2399_239985


namespace NUMINAMATH_CALUDE_fraction_simplification_l2399_239953

theorem fraction_simplification : (2 : ℚ) / (1 - 2 / 3) = 6 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2399_239953


namespace NUMINAMATH_CALUDE_power_nap_duration_l2399_239901

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

/-- Represents one fourth of an hour -/
def quarter_hour : ℚ := 1 / 4

theorem power_nap_duration :
  hours_to_minutes quarter_hour = 15 := by sorry

end NUMINAMATH_CALUDE_power_nap_duration_l2399_239901


namespace NUMINAMATH_CALUDE_winter_clothing_count_l2399_239983

theorem winter_clothing_count (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) :
  num_boxes = 6 →
  scarves_per_box = 5 →
  mittens_per_box = 5 →
  num_boxes * (scarves_per_box + mittens_per_box) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_winter_clothing_count_l2399_239983


namespace NUMINAMATH_CALUDE_no_goal_scored_l2399_239981

def football_play (play1 play2 play3 play4 : ℝ) : Prop :=
  play1 = -5 ∧ 
  play2 = 13 ∧ 
  play3 = -(play1^2) ∧ 
  play4 = -play3 / 2

def total_progress (play1 play2 play3 play4 : ℝ) : ℝ :=
  play1 + play2 + play3 + play4

def score_goal (progress : ℝ) : Prop :=
  progress ≥ 30

theorem no_goal_scored (play1 play2 play3 play4 : ℝ) :
  football_play play1 play2 play3 play4 →
  ¬(score_goal (total_progress play1 play2 play3 play4)) :=
by sorry

end NUMINAMATH_CALUDE_no_goal_scored_l2399_239981


namespace NUMINAMATH_CALUDE_calculate_expression_l2399_239933

theorem calculate_expression : 7 * (9 + 2/5) + 3 = 68.8 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2399_239933


namespace NUMINAMATH_CALUDE_solve_diamond_equation_l2399_239937

-- Define the binary operation ⋄
noncomputable def diamond (a b : ℝ) : ℝ := sorry

-- Axioms for the binary operation
axiom diamond_assoc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = (diamond a b) * c

axiom diamond_self (a : ℝ) (ha : a ≠ 0) : diamond a a = 1

-- Theorem statement
theorem solve_diamond_equation :
  ∃ x : ℝ, x ≠ 0 ∧ diamond 504 (diamond 12 x) = 50 → x = 25 / 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_diamond_equation_l2399_239937


namespace NUMINAMATH_CALUDE_upgraded_fraction_is_one_fourth_l2399_239975

/-- Represents a satellite with modular units and sensors -/
structure Satellite where
  units : ℕ
  non_upgraded_per_unit : ℕ
  upgraded_total : ℕ

/-- The fraction of upgraded sensors on a satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.upgraded_total / (s.units * s.non_upgraded_per_unit + s.upgraded_total)

/-- Theorem: The fraction of upgraded sensors is 1/4 under given conditions -/
theorem upgraded_fraction_is_one_fourth (s : Satellite) 
    (h1 : s.units = 24)
    (h2 : s.non_upgraded_per_unit = s.upgraded_total / 8) :
  upgraded_fraction s = 1/4 := by
  sorry

#eval upgraded_fraction { units := 24, non_upgraded_per_unit := 1, upgraded_total := 8 }

end NUMINAMATH_CALUDE_upgraded_fraction_is_one_fourth_l2399_239975


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_two_l2399_239950

-- Define the curve
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

-- Theorem statement
theorem tangent_line_at_point_one_two :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 3*x - 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_two_l2399_239950


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2399_239960

theorem sum_of_coefficients (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (x - a)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₅ = 56 →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 2^8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2399_239960


namespace NUMINAMATH_CALUDE_expansion_coefficient_l2399_239982

/-- The coefficient of x^3y^7 in the expansion of (2/3x - 3/4y)^10 -/
def coefficient_x3y7 : ℚ :=
  let a : ℚ := 2/3
  let b : ℚ := -3/4
  let n : ℕ := 10
  let k : ℕ := 7
  (n.choose k) * a^(n-k) * b^k

theorem expansion_coefficient :
  coefficient_x3y7 = -4374/921 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l2399_239982


namespace NUMINAMATH_CALUDE_probability_no_consecutive_ones_l2399_239952

/-- Sequence without consecutive ones -/
def SeqWithoutConsecutiveOnes (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n+2) => SeqWithoutConsecutiveOnes (n+1) + SeqWithoutConsecutiveOnes n

/-- Total number of possible sequences -/
def TotalSequences (n : ℕ) : ℕ := 2^n

theorem probability_no_consecutive_ones :
  (SeqWithoutConsecutiveOnes 12 : ℚ) / (TotalSequences 12) = 377 / 4096 := by
  sorry

#eval SeqWithoutConsecutiveOnes 12
#eval TotalSequences 12

end NUMINAMATH_CALUDE_probability_no_consecutive_ones_l2399_239952


namespace NUMINAMATH_CALUDE_math_pass_count_l2399_239927

/-- Represents the number of students in various categories -/
structure StudentCounts where
  english : ℕ
  math : ℕ
  bothSubjects : ℕ
  onlyEnglish : ℕ
  onlyMath : ℕ

/-- Theorem stating the number of students who pass in Math -/
theorem math_pass_count (s : StudentCounts) 
  (h1 : s.english = 30)
  (h2 : s.english = s.onlyEnglish + s.bothSubjects)
  (h3 : s.onlyEnglish = s.onlyMath + 10)
  (h4 : s.math = s.onlyMath + s.bothSubjects) :
  s.math = 20 := by
  sorry

end NUMINAMATH_CALUDE_math_pass_count_l2399_239927


namespace NUMINAMATH_CALUDE_regular_soda_count_l2399_239929

/-- The number of bottles of regular soda in a grocery store -/
def regular_soda : ℕ := sorry

/-- The number of bottles of diet soda in a grocery store -/
def diet_soda : ℕ := 26

/-- The number of bottles of lite soda in a grocery store -/
def lite_soda : ℕ := 27

/-- The total number of soda bottles in a grocery store -/
def total_bottles : ℕ := 110

/-- Theorem stating that the number of bottles of regular soda is 57 -/
theorem regular_soda_count : regular_soda = 57 := by
  sorry

end NUMINAMATH_CALUDE_regular_soda_count_l2399_239929


namespace NUMINAMATH_CALUDE_complex_calculation_l2399_239944

theorem complex_calculation (p q : ℂ) (hp : p = 3 + 2*I) (hq : q = 2 - 3*I) :
  3*p + 4*q = 17 - 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l2399_239944


namespace NUMINAMATH_CALUDE_root_implies_c_value_l2399_239949

theorem root_implies_c_value (b c : ℝ) :
  (∃ (x : ℂ), x^2 + b*x + c = 0 ∧ x = 1 - Complex.I * Real.sqrt 2) →
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_c_value_l2399_239949


namespace NUMINAMATH_CALUDE_log_exponent_sum_l2399_239931

theorem log_exponent_sum (a : ℝ) (h : a = Real.log 5 / Real.log 4) :
  2^a + 2^(-a) = 6 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_log_exponent_sum_l2399_239931


namespace NUMINAMATH_CALUDE_find_T_l2399_239962

theorem find_T : ∃ T : ℚ, (1/3 : ℚ) * (1/4 : ℚ) * T = (1/3 : ℚ) * (1/5 : ℚ) * 120 ∧ T = 96 := by
  sorry

end NUMINAMATH_CALUDE_find_T_l2399_239962


namespace NUMINAMATH_CALUDE_calculate_required_hours_johns_work_schedule_l2399_239947

/-- Calculates the required weekly work hours for a target income given previous work data --/
theorem calculate_required_hours (winter_hours_per_week : ℕ) (winter_weeks : ℕ) (winter_earnings : ℕ) 
  (target_weeks : ℕ) (target_earnings : ℕ) : ℕ :=
  let hourly_rate := winter_earnings / (winter_hours_per_week * winter_weeks)
  let total_hours := target_earnings / hourly_rate
  total_hours / target_weeks

/-- John's work schedule problem --/
theorem johns_work_schedule : 
  calculate_required_hours 40 8 3200 24 4800 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculate_required_hours_johns_work_schedule_l2399_239947


namespace NUMINAMATH_CALUDE_sara_quarters_theorem_l2399_239978

/-- The number of quarters Sara had initially -/
def initial_quarters : ℕ := 783

/-- The number of quarters Sara's dad gave her -/
def quarters_from_dad : ℕ := 271

/-- The total number of quarters Sara has now -/
def total_quarters : ℕ := 1054

/-- Theorem stating that the initial number of quarters plus the quarters from dad equals the total quarters -/
theorem sara_quarters_theorem : initial_quarters + quarters_from_dad = total_quarters := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_theorem_l2399_239978


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2399_239913

/-- Given a group of 8 people, prove that if replacing one person with a new person
    weighing 105 kg increases the average weight by 2.5 kg, then the weight of the
    replaced person is 85 kg. -/
theorem weight_of_replaced_person
  (initial_count : ℕ)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h_initial_count : initial_count = 8)
  (h_weight_increase : weight_increase = 2.5)
  (h_new_person_weight : new_person_weight = 105)
  : ∃ (replaced_weight : ℝ),
    replaced_weight = 85 ∧
    (initial_count : ℝ) * weight_increase = new_person_weight - replaced_weight :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2399_239913


namespace NUMINAMATH_CALUDE_star_calculation_l2399_239900

def star (x y : ℤ) : ℤ := x * y - 1

theorem star_calculation : (star (star 2 3) 4) = 19 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l2399_239900


namespace NUMINAMATH_CALUDE_problem_solution_problem_solution_2_l2399_239938

def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∀ x, p x → q x m) → m ∈ Set.Ici 4 :=
sorry

theorem problem_solution_2 :
  ∃ S : Set ℝ, S = Set.Icc (-4) (-1) ∪ Set.Ioc 5 6 ∧
  ∀ x, x ∈ S ↔ (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_problem_solution_2_l2399_239938


namespace NUMINAMATH_CALUDE_derivative_roots_in_triangle_l2399_239999

/-- A polynomial of degree three with complex roots -/
def cubic_polynomial (a b c : ℂ) (x : ℂ) : ℂ :=
  (x - a) * (x - b) * (x - c)

/-- The derivative of the cubic polynomial -/
def cubic_derivative (a b c : ℂ) (x : ℂ) : ℂ :=
  (x - a) * (x - b) + (x - a) * (x - c) + (x - b) * (x - c)

/-- The triangle formed by the roots of the cubic polynomial -/
def root_triangle (a b c : ℂ) : Set ℂ :=
  {z : ℂ | ∃ (t₁ t₂ t₃ : ℝ), t₁ + t₂ + t₃ = 1 ∧ t₁ ≥ 0 ∧ t₂ ≥ 0 ∧ t₃ ≥ 0 ∧ z = t₁ * a + t₂ * b + t₃ * c}

/-- Theorem stating that the roots of the derivative lie inside the triangle formed by the roots of the original polynomial -/
theorem derivative_roots_in_triangle (a b c : ℂ) :
  ∀ z : ℂ, cubic_derivative a b c z = 0 → z ∈ root_triangle a b c :=
sorry

end NUMINAMATH_CALUDE_derivative_roots_in_triangle_l2399_239999


namespace NUMINAMATH_CALUDE_binomial_coefficient_200_l2399_239965

theorem binomial_coefficient_200 :
  (Nat.choose 200 200 = 1) ∧ (Nat.choose 200 0 = 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_200_l2399_239965


namespace NUMINAMATH_CALUDE_canned_food_bins_l2399_239924

theorem canned_food_bins (soup : Real) (vegetables : Real) (pasta : Real)
  (h1 : soup = 0.12)
  (h2 : vegetables = 0.12)
  (h3 : pasta = 0.5) :
  soup + vegetables + pasta = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_canned_food_bins_l2399_239924


namespace NUMINAMATH_CALUDE_c_share_is_54_l2399_239946

/-- Represents the rental arrangement for a pasture --/
structure PastureRental where
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ
  total_rent : ℕ

/-- Calculates the share of rent for person c given a PastureRental arrangement --/
def calculate_c_share (rental : PastureRental) : ℚ :=
  let total_ox_months := rental.a_oxen * rental.a_months + 
                         rental.b_oxen * rental.b_months + 
                         rental.c_oxen * rental.c_months
  let rent_per_ox_month : ℚ := rental.total_rent / total_ox_months
  (rental.c_oxen * rental.c_months : ℚ) * rent_per_ox_month

/-- The main theorem stating that c's share of the rent is 54 --/
theorem c_share_is_54 (rental : PastureRental) 
  (h1 : rental.a_oxen = 10) (h2 : rental.a_months = 7)
  (h3 : rental.b_oxen = 12) (h4 : rental.b_months = 5)
  (h5 : rental.c_oxen = 15) (h6 : rental.c_months = 3)
  (h7 : rental.total_rent = 210) : 
  calculate_c_share rental = 54 := by
  sorry

end NUMINAMATH_CALUDE_c_share_is_54_l2399_239946


namespace NUMINAMATH_CALUDE_cross_to_square_l2399_239905

/-- Represents a cross made of five equal squares -/
structure Cross where
  side_length : ℝ
  num_squares : Nat
  h_num_squares : num_squares = 5

/-- Represents the square formed by reassembling the cross parts -/
structure ReassembledSquare where
  side_length : ℝ

/-- States that a cross can be cut into parts that form a square -/
def can_form_square (c : Cross) (s : ReassembledSquare) : Prop :=
  s.side_length = c.side_length * Real.sqrt 5

/-- Theorem stating that a cross of five equal squares can be cut to form a square -/
theorem cross_to_square (c : Cross) :
  ∃ s : ReassembledSquare, can_form_square c s :=
sorry

end NUMINAMATH_CALUDE_cross_to_square_l2399_239905


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2399_239908

theorem absolute_value_inequality (x y : ℝ) (h : x < y ∧ y < 0) :
  abs x > (abs (x + y)) / 2 ∧ (abs (x + y)) / 2 > abs y := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2399_239908


namespace NUMINAMATH_CALUDE_student_score_l2399_239996

def max_marks : ℕ := 400
def pass_percentage : ℚ := 30 / 100
def fail_margin : ℕ := 40

theorem student_score : 
  ∀ (student_marks : ℕ),
    (student_marks = max_marks * pass_percentage - fail_margin) →
    student_marks = 80 := by
  sorry

end NUMINAMATH_CALUDE_student_score_l2399_239996


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l2399_239919

/-- The curve y = x^3 + ax + 1 passes through the point (2, 3) -/
def curve_passes_through (a : ℝ) : Prop :=
  2^3 + a*2 + 1 = 3

/-- The derivative of the curve y = x^3 + ax + 1 -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ :=
  3*x^2 + a

/-- The line y = kx + b is tangent to the curve y = x^3 + ax + 1 at x = 2 -/
def line_tangent_to_curve (a k b : ℝ) : Prop :=
  k = curve_derivative a 2

/-- The line y = kx + b passes through the point (2, 3) -/
def line_passes_through (k b : ℝ) : Prop :=
  k*2 + b = 3

theorem tangent_line_b_value (a k b : ℝ) :
  curve_passes_through a →
  line_tangent_to_curve a k b →
  line_passes_through k b →
  b = -15 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l2399_239919


namespace NUMINAMATH_CALUDE_bran_remaining_payment_l2399_239993

def tuition_fee : ℝ := 90
def monthly_earnings : ℝ := 15
def scholarship_percentage : ℝ := 0.30
def payment_period : ℕ := 3

theorem bran_remaining_payment :
  tuition_fee * (1 - scholarship_percentage) - monthly_earnings * payment_period = 18 := by
  sorry

end NUMINAMATH_CALUDE_bran_remaining_payment_l2399_239993


namespace NUMINAMATH_CALUDE_large_data_logarithm_l2399_239945

theorem large_data_logarithm (m : ℝ) (n : ℕ+) :
  (1 < m) ∧ (m < 10) ∧
  (0.4771 < Real.log 3 / Real.log 10) ∧ (Real.log 3 / Real.log 10 < 0.4772) ∧
  (3 ^ 2000 : ℝ) = m * 10 ^ (n : ℝ) →
  n = 954 := by
  sorry

end NUMINAMATH_CALUDE_large_data_logarithm_l2399_239945


namespace NUMINAMATH_CALUDE_tshirt_production_l2399_239909

/-- The number of minutes in an hour -/
def minutesPerHour : ℕ := 60

/-- The rate of t-shirt production in the first hour (minutes per t-shirt) -/
def rateFirstHour : ℕ := 12

/-- The rate of t-shirt production in the second hour (minutes per t-shirt) -/
def rateSecondHour : ℕ := 6

/-- The total number of t-shirts produced in two hours -/
def totalTShirts : ℕ := minutesPerHour / rateFirstHour + minutesPerHour / rateSecondHour

theorem tshirt_production : totalTShirts = 15 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_production_l2399_239909


namespace NUMINAMATH_CALUDE_fourth_person_height_l2399_239935

/-- Heights of four people in increasing order -/
def Heights := Fin 4 → ℝ

/-- The common difference between the heights of the first three people -/
def common_difference (h : Heights) : ℝ := h 1 - h 0

theorem fourth_person_height (h : Heights) 
  (increasing : ∀ i j, i < j → h i < h j)
  (common_diff : h 2 - h 1 = h 1 - h 0)
  (last_diff : h 3 - h 2 = 6)
  (avg_height : (h 0 + h 1 + h 2 + h 3) / 4 = 77) :
  h 3 = h 0 + 2 * (common_difference h) + 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l2399_239935


namespace NUMINAMATH_CALUDE_triangle_inequality_l2399_239987

/-- Triangle inequality for sides and area -/
theorem triangle_inequality (a b c S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : S = Real.sqrt (((a + b + c) / 2) * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 + (a - b)^2 + (b - c)^2 + (c - a)^2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequality_l2399_239987


namespace NUMINAMATH_CALUDE_binomial_sum_l2399_239928

theorem binomial_sum (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + x)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 63 :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_l2399_239928


namespace NUMINAMATH_CALUDE_race_heartbeats_l2399_239936

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Proves that the total number of heartbeats during a 30-mile race is 21600 -/
theorem race_heartbeats :
  let heart_rate : ℕ := 120  -- beats per minute
  let pace : ℕ := 6          -- minutes per mile
  let distance : ℕ := 30     -- miles
  total_heartbeats heart_rate pace distance = 21600 := by
sorry

#eval total_heartbeats 120 6 30

end NUMINAMATH_CALUDE_race_heartbeats_l2399_239936


namespace NUMINAMATH_CALUDE_distance_is_7920_meters_l2399_239973

/-- The distance traveled by a man driving at constant speed from the site of a blast -/
def distance_traveled (speed_of_sound : ℝ) (time_between_blasts : ℝ) (time_heard_second_blast : ℝ) : ℝ :=
  speed_of_sound * (time_heard_second_blast - time_between_blasts)

/-- Theorem stating that the distance traveled is 7920 meters -/
theorem distance_is_7920_meters :
  let speed_of_sound : ℝ := 330
  let time_between_blasts : ℝ := 30 * 60  -- 30 minutes in seconds
  let time_heard_second_blast : ℝ := 30 * 60 + 24  -- 30 minutes and 24 seconds in seconds
  distance_traveled speed_of_sound time_between_blasts time_heard_second_blast = 7920 := by
  sorry


end NUMINAMATH_CALUDE_distance_is_7920_meters_l2399_239973


namespace NUMINAMATH_CALUDE_division_with_remainder_4032_98_l2399_239911

theorem division_with_remainder_4032_98 : ∃ (q r : ℤ), 4032 = 98 * q + r ∧ 0 ≤ r ∧ r < 98 ∧ r = 14 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_4032_98_l2399_239911


namespace NUMINAMATH_CALUDE_lettuce_plants_needed_l2399_239971

def min_salads : ℕ := 12
def loss_ratio : ℚ := 1/2
def salads_per_plant : ℕ := 3

theorem lettuce_plants_needed : ℕ := by
  -- Proof goes here
  sorry

#check lettuce_plants_needed = 8

end NUMINAMATH_CALUDE_lettuce_plants_needed_l2399_239971


namespace NUMINAMATH_CALUDE_unique_number_six_times_sum_of_digits_l2399_239948

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for numbers that are 6 times the sum of their digits -/
def is_six_times_sum_of_digits (n : ℕ) : Prop :=
  n = 6 * sum_of_digits n

theorem unique_number_six_times_sum_of_digits :
  ∃! n : ℕ, n < 1000 ∧ is_six_times_sum_of_digits n :=
sorry

end NUMINAMATH_CALUDE_unique_number_six_times_sum_of_digits_l2399_239948


namespace NUMINAMATH_CALUDE_dorokhov_vacation_cost_l2399_239940

/-- Represents a travel agency with its pricing structure -/
structure TravelAgency where
  name : String
  price_young : ℕ
  price_old : ℕ
  age_threshold : ℕ
  discount_rate : ℚ
  is_commission : Bool

/-- Calculates the total cost for a family's vacation package -/
def calculate_total_cost (agency : TravelAgency) (num_adults num_children : ℕ) (child_age : ℕ) : ℚ :=
  sorry

/-- The Dorokhov family's vacation cost theorem -/
theorem dorokhov_vacation_cost :
  let globus : TravelAgency := {
    name := "Globus",
    price_young := 11200,
    price_old := 25400,
    age_threshold := 5,
    discount_rate := -2/100,
    is_commission := false
  }
  let around_the_world : TravelAgency := {
    name := "Around the World",
    price_young := 11400,
    price_old := 23500,
    age_threshold := 6,
    discount_rate := 1/100,
    is_commission := true
  }
  let num_adults : ℕ := 2
  let num_children : ℕ := 1
  let child_age : ℕ := 5
  
  min (calculate_total_cost globus num_adults num_children child_age)
      (calculate_total_cost around_the_world num_adults num_children child_age) = 58984 := by
  sorry

end NUMINAMATH_CALUDE_dorokhov_vacation_cost_l2399_239940


namespace NUMINAMATH_CALUDE_double_angle_formulas_l2399_239914

open Real

theorem double_angle_formulas (α p q : ℝ) (h : tan α = p / q) :
  sin (2 * α) = (2 * p * q) / (p^2 + q^2) ∧
  cos (2 * α) = (q^2 - p^2) / (q^2 + p^2) ∧
  tan (2 * α) = (2 * p * q) / (q^2 - p^2) := by
  sorry

end NUMINAMATH_CALUDE_double_angle_formulas_l2399_239914


namespace NUMINAMATH_CALUDE_volunteer_count_l2399_239941

/-- Represents the number of volunteers selected from each school -/
structure Volunteers where
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ

/-- The ratio of students in schools A, B, and C -/
def schoolRatio : Fin 3 → ℕ
  | 0 => 2  -- School A
  | 1 => 3  -- School B
  | 2 => 5  -- School C

/-- The total ratio sum -/
def totalRatio : ℕ := (schoolRatio 0) + (schoolRatio 1) + (schoolRatio 2)

/-- Stratified sampling condition -/
def isStratifiedSample (v : Volunteers) : Prop :=
  (v.schoolA * schoolRatio 1 = v.schoolB * schoolRatio 0) ∧
  (v.schoolA * schoolRatio 2 = v.schoolC * schoolRatio 0)

/-- The main theorem -/
theorem volunteer_count (v : Volunteers) 
  (h_stratified : isStratifiedSample v) 
  (h_schoolA : v.schoolA = 6) : 
  v.schoolA + v.schoolB + v.schoolC = 30 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_count_l2399_239941


namespace NUMINAMATH_CALUDE_systematic_sampling_first_number_l2399_239926

/-- Systematic sampling function that returns the number drawn from the nth group -/
def systematicSample (firstNumber : ℕ) (groupNumber : ℕ) (interval : ℕ) : ℕ :=
  firstNumber + interval * (groupNumber - 1)

theorem systematic_sampling_first_number :
  ∀ (totalStudents : ℕ) (numGroups : ℕ) (firstNumber : ℕ),
    totalStudents = 160 →
    numGroups = 20 →
    systematicSample firstNumber 15 8 = 116 →
    firstNumber = 4 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_number_l2399_239926


namespace NUMINAMATH_CALUDE_job_selection_ways_l2399_239917

theorem job_selection_ways (method1_people : ℕ) (method2_people : ℕ) 
  (h1 : method1_people = 3) (h2 : method2_people = 5) : 
  method1_people + method2_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_job_selection_ways_l2399_239917


namespace NUMINAMATH_CALUDE_third_square_perimeter_l2399_239923

/-- Given two squares with perimeters 40 cm and 32 cm, prove that a third square
    whose area is equal to the difference of the areas of the first two squares
    has a perimeter of 24 cm. -/
theorem third_square_perimeter (square1 square2 square3 : Real → Real → Real) :
  (∀ s, square1 s s = s * s) →
  (∀ s, square2 s s = s * s) →
  (∀ s, square3 s s = s * s) →
  (4 * 10 = 40) →
  (4 * 8 = 32) →
  (square1 10 10 - square2 8 8 = square3 6 6) →
  (4 * 6 = 24) := by
sorry

end NUMINAMATH_CALUDE_third_square_perimeter_l2399_239923


namespace NUMINAMATH_CALUDE_divisor_problem_l2399_239969

/-- 
Given a dividend of 23, a quotient of 5, and a remainder of 3, 
prove that the divisor is 4.
-/
theorem divisor_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) : 
  dividend = 23 → quotient = 5 → remainder = 3 → 
  dividend = divisor * quotient + remainder → 
  divisor = 4 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l2399_239969


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2399_239990

theorem inequality_solution_set (x : ℝ) : 
  (((2 * x - 1) / 3) > ((3 * x - 2) / 2 - 1)) ↔ (x < 2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2399_239990


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2399_239939

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : x^2 + y^2 + x*y = 1)
  (h2 : y^2 + z^2 + y*z = 2)
  (h3 : x^2 + z^2 + x*z = 3) :
  x + y + z = Real.sqrt (3 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2399_239939


namespace NUMINAMATH_CALUDE_system_solution_square_difference_l2399_239932

theorem system_solution_square_difference (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 1) 
  (eq2 : x + y = 2) : 
  x^2 - 2 * y^2 = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_square_difference_l2399_239932


namespace NUMINAMATH_CALUDE_fox_can_eat_80_fox_cannot_eat_65_l2399_239984

/-- Represents a distribution of candies into three piles -/
structure CandyDistribution :=
  (pile1 pile2 pile3 : ℕ)
  (sum_eq_100 : pile1 + pile2 + pile3 = 100)

/-- Calculates the number of candies the fox eats given a distribution -/
def fox_eats (d : CandyDistribution) : ℕ :=
  let min_pile := min d.pile1 (min d.pile2 d.pile3)
  let max_pile := max d.pile1 (max d.pile2 d.pile3)
  if d.pile1 = d.pile2 ∨ d.pile2 = d.pile3 ∨ d.pile1 = d.pile3
  then max_pile
  else min_pile + (max_pile - min_pile) / 2

theorem fox_can_eat_80 : ∃ d : CandyDistribution, fox_eats d = 80 := by sorry

theorem fox_cannot_eat_65 : ¬ ∃ d : CandyDistribution, fox_eats d = 65 := by sorry

end NUMINAMATH_CALUDE_fox_can_eat_80_fox_cannot_eat_65_l2399_239984


namespace NUMINAMATH_CALUDE_exterior_angle_regular_pentagon_l2399_239942

theorem exterior_angle_regular_pentagon :
  ∀ (exterior_angle : ℝ),
  (exterior_angle = 180 - (540 / 5)) →
  exterior_angle = 72 := by
sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_pentagon_l2399_239942


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2399_239915

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 3}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem intersection_complement_equality : B ∩ (U \ A) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2399_239915


namespace NUMINAMATH_CALUDE_vacant_seats_l2399_239920

/-- Given a hall with 600 seats where 62% are filled, prove that 228 seats are vacant. -/
theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 62/100) : 
  (total_seats : ℚ) * (1 - filled_percentage) = 228 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l2399_239920


namespace NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l2399_239907

theorem halfway_between_one_eighth_and_one_third :
  (1/8 : ℚ) + ((1/3 : ℚ) - (1/8 : ℚ)) / 2 = 11/48 := by sorry

end NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l2399_239907


namespace NUMINAMATH_CALUDE_school_supplies_cost_l2399_239974

theorem school_supplies_cost :
  let pencil_cartons : ℕ := 20
  let pencil_boxes_per_carton : ℕ := 10
  let pencil_box_cost : ℕ := 2
  let marker_cartons : ℕ := 10
  let marker_boxes_per_carton : ℕ := 5
  let marker_box_cost : ℕ := 4
  
  pencil_cartons * pencil_boxes_per_carton * pencil_box_cost +
  marker_cartons * marker_boxes_per_carton * marker_box_cost = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l2399_239974


namespace NUMINAMATH_CALUDE_base_product_sum_theorem_l2399_239930

/-- Represents a number in a given base --/
structure BaseNumber (base : ℕ) where
  value : ℕ

/-- Converts a BaseNumber to its decimal representation --/
def toDecimal {base : ℕ} (n : BaseNumber base) : ℕ := sorry

/-- Converts a decimal number to a BaseNumber --/
def fromDecimal (base : ℕ) (n : ℕ) : BaseNumber base := sorry

/-- Multiplies two BaseNumbers --/
def mult {base : ℕ} (a b : BaseNumber base) : BaseNumber base := sorry

/-- Adds two BaseNumbers --/
def add {base : ℕ} (a b : BaseNumber base) : BaseNumber base := sorry

theorem base_product_sum_theorem :
  ∀ c : ℕ,
    c > 1 →
    let thirteen := fromDecimal c 13
    let seventeen := fromDecimal c 17
    let nineteen := fromDecimal c 19
    let product := mult thirteen (mult seventeen nineteen)
    let sum := add thirteen (add seventeen nineteen)
    toDecimal product = toDecimal (fromDecimal c 4375) →
    toDecimal sum = toDecimal (fromDecimal 8 53) := by
  sorry

end NUMINAMATH_CALUDE_base_product_sum_theorem_l2399_239930


namespace NUMINAMATH_CALUDE_sqrt_114_plus_44_sqrt_6_l2399_239943

theorem sqrt_114_plus_44_sqrt_6 :
  ∃ (x y z : ℤ), (x + y * Real.sqrt z : ℝ) = Real.sqrt (114 + 44 * Real.sqrt 6) ∧
  z > 0 ∧
  (∀ (w : ℤ), w ^ 2 ∣ z → w = 1 ∨ w = -1) ∧
  x = 5 ∧ y = 2 ∧ z = 6 :=
sorry

end NUMINAMATH_CALUDE_sqrt_114_plus_44_sqrt_6_l2399_239943


namespace NUMINAMATH_CALUDE_max_m_value_l2399_239966

theorem max_m_value (m : ℝ) : 
  (∀ x ∈ Set.Icc (-π/4 : ℝ) (π/4 : ℝ), m ≤ Real.tan x + 1) → 
  (∃ M : ℝ, (∀ x ∈ Set.Icc (-π/4 : ℝ) (π/4 : ℝ), M ≤ Real.tan x + 1) ∧ M = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l2399_239966


namespace NUMINAMATH_CALUDE_soup_feeding_theorem_l2399_239903

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : Nat
  children : Nat

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (totalCans : Nat) (canCapacity : SoupCan) (childrenFed : Nat) : Nat :=
  let cansForChildren := childrenFed / canCapacity.children
  let remainingCans := totalCans - cansForChildren
  remainingCans * canCapacity.adults

/-- Theorem stating that given the conditions, 20 adults can be fed with the remaining soup -/
theorem soup_feeding_theorem (totalCans : Nat) (canCapacity : SoupCan) (childrenFed : Nat) :
  totalCans = 10 →
  canCapacity.adults = 4 →
  canCapacity.children = 8 →
  childrenFed = 40 →
  remainingAdults totalCans canCapacity childrenFed = 20 := by
  sorry

end NUMINAMATH_CALUDE_soup_feeding_theorem_l2399_239903


namespace NUMINAMATH_CALUDE_vector_on_line_iff_k_eq_half_l2399_239958

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

/-- A line passing through points represented by vectors p and q -/
def line (p q : n) : Set n :=
  {x | ∃ t : ℝ, x = p + t • (q - p)}

/-- The vector that should lie on the line -/
def vector_on_line (p q : n) (k : ℝ) : n :=
  k • p + (1/2) • q

/-- Theorem stating that the vector lies on the line if and only if k = 1/2 -/
theorem vector_on_line_iff_k_eq_half (p q : n) :
  ∀ k : ℝ, vector_on_line p q k ∈ line p q ↔ k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_iff_k_eq_half_l2399_239958


namespace NUMINAMATH_CALUDE_compound_interest_initial_sum_l2399_239968

/-- Given an initial sum of money P and an annual compound interest rate r,
    if P(1 + r)² = 8880 and P(1 + r)³ = 9261, then P is approximately equal to 8160. -/
theorem compound_interest_initial_sum (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 8880)
  (h2 : P * (1 + r)^3 = 9261) :
  ∃ ε > 0, |P - 8160| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_interest_initial_sum_l2399_239968


namespace NUMINAMATH_CALUDE_inverse_cube_root_relation_l2399_239961

/-- Given that z varies inversely as the cube root of x, and z = 2 when x = 8,
    prove that x = 1 when z = 4. -/
theorem inverse_cube_root_relation (z x : ℝ) (h1 : z * x^(1/3) = 2 * 8^(1/3)) :
  z = 4 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_cube_root_relation_l2399_239961


namespace NUMINAMATH_CALUDE_inequality_proof_l2399_239912

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  (a / (b + c + 1)) + (b / (c + a + 1)) + (c / (a + b + 1)) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2399_239912


namespace NUMINAMATH_CALUDE_multiples_of_15_sequence_two_thousand_sixteen_position_l2399_239991

theorem multiples_of_15_sequence (n : ℕ) : ℕ → ℕ
  | 0 => 0
  | (k + 1) => 15 * (k + 1)

theorem two_thousand_sixteen_position :
  ∃ (n : ℕ), multiples_of_15_sequence n 134 < 2016 ∧ 
             2016 < multiples_of_15_sequence n 135 ∧ 
             multiples_of_15_sequence n 135 - 2016 = 9 := by
  sorry


end NUMINAMATH_CALUDE_multiples_of_15_sequence_two_thousand_sixteen_position_l2399_239991


namespace NUMINAMATH_CALUDE_third_iteration_interval_l2399_239998

def bisection_interval (a b : ℝ) (n : ℕ) : Set (ℝ × ℝ) :=
  match n with
  | 0 => {(a, b)}
  | n+1 => let m := (a + b) / 2
           (bisection_interval a m n) ∪ (bisection_interval m b n)

theorem third_iteration_interval (a b : ℝ) (h : (a, b) = (-2, 4)) :
  (-1/2, 1) ∈ bisection_interval a b 3 :=
sorry

end NUMINAMATH_CALUDE_third_iteration_interval_l2399_239998


namespace NUMINAMATH_CALUDE_cat_count_correct_l2399_239963

/-- The number of cats that can meow -/
def meow : ℕ := 70

/-- The number of cats that can jump -/
def jump : ℕ := 40

/-- The number of cats that can fetch -/
def fetch : ℕ := 30

/-- The number of cats that can roll -/
def roll : ℕ := 50

/-- The number of cats that can meow and jump -/
def meow_jump : ℕ := 25

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 15

/-- The number of cats that can fetch and roll -/
def fetch_roll : ℕ := 20

/-- The number of cats that can meow and roll -/
def meow_roll : ℕ := 28

/-- The number of cats that can meow, jump, and fetch -/
def meow_jump_fetch : ℕ := 5

/-- The number of cats that can jump, fetch, and roll -/
def jump_fetch_roll : ℕ := 10

/-- The number of cats that can fetch, roll, and meow -/
def fetch_roll_meow : ℕ := 12

/-- The number of cats that can do all four tricks -/
def all_four : ℕ := 8

/-- The number of cats that can do no tricks -/
def no_tricks : ℕ := 12

/-- The total number of cats in the studio -/
def total_cats : ℕ := 129

theorem cat_count_correct : 
  total_cats = meow + jump + fetch + roll - meow_jump - jump_fetch - fetch_roll - meow_roll + 
               meow_jump_fetch + jump_fetch_roll + fetch_roll_meow - 2 * all_four + no_tricks := by
  sorry

end NUMINAMATH_CALUDE_cat_count_correct_l2399_239963


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l2399_239977

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The kite formed by the intersections of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

/-- The area of a kite -/
def kite_area (k : Kite) : ℝ := sorry

/-- The theorem to be proved -/
theorem parabola_kite_sum (c d : ℝ) :
  let p1 : Parabola := ⟨c, 3⟩
  let p2 : Parabola := ⟨-d, 7⟩
  let k : Kite := ⟨p1, p2⟩
  kite_area k = 20 → c + d = 18/25 := by sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l2399_239977


namespace NUMINAMATH_CALUDE_lattice_paths_avoiding_point_l2399_239906

/-- Represents a point on the lattice -/
structure Point where
  x : Nat
  y : Nat

/-- Calculates the number of paths from (0,0) to a given point -/
def numPaths (p : Point) : Nat :=
  Nat.choose (p.x + p.y) p.x

/-- The theorem to be proved -/
theorem lattice_paths_avoiding_point :
  numPaths ⟨4, 4⟩ - numPaths ⟨2, 2⟩ * numPaths ⟨2, 2⟩ = 34 := by
  sorry

#eval numPaths ⟨4, 4⟩ - numPaths ⟨2, 2⟩ * numPaths ⟨2, 2⟩

end NUMINAMATH_CALUDE_lattice_paths_avoiding_point_l2399_239906


namespace NUMINAMATH_CALUDE_rational_equation_equality_l2399_239967

theorem rational_equation_equality (x : ℝ) (h : x ≠ -1) : 
  (1 / (x + 1)) + (1 / (x + 1)^2) + ((-x - 1) / (x^2 + x + 1)) = 
  1 / ((x + 1)^2 * (x^2 + x + 1)) := by sorry

end NUMINAMATH_CALUDE_rational_equation_equality_l2399_239967


namespace NUMINAMATH_CALUDE_nested_f_evaluation_l2399_239957

def f (x : ℝ) : ℝ := x^2 + 1

theorem nested_f_evaluation : f (f (f (-1))) = 26 := by sorry

end NUMINAMATH_CALUDE_nested_f_evaluation_l2399_239957


namespace NUMINAMATH_CALUDE_small_portion_visible_implies_intersection_l2399_239980

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is above a line -/
def isAboveLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c > 0

/-- Predicate to check if a circle intersects a line -/
def circleIntersectsLine (c : Circle) (l : Line) : Prop :=
  ∃ (p : ℝ × ℝ), (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
                 l.a * p.1 + l.b * p.2 + l.c = 0

/-- Predicate to check if a small portion of a circle is visible above a line -/
def smallPortionVisible (c : Circle) (l : Line) : Prop :=
  ∃ (p q : ℝ × ℝ), 
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 ∧
    isAboveLine p l ∧
    isAboveLine q l ∧
    ∀ (r : ℝ × ℝ), (r.1 - c.center.1)^2 + (r.2 - c.center.2)^2 = c.radius^2 →
                   isAboveLine r l →
                   (r.1 ≥ min p.1 q.1 ∧ r.1 ≤ max p.1 q.1) ∧
                   (r.2 ≥ min p.2 q.2 ∧ r.2 ≤ max p.2 q.2)

theorem small_portion_visible_implies_intersection (c : Circle) (l : Line) :
  smallPortionVisible c l → circleIntersectsLine c l :=
by sorry

end NUMINAMATH_CALUDE_small_portion_visible_implies_intersection_l2399_239980


namespace NUMINAMATH_CALUDE_Betty_wallet_contribution_ratio_l2399_239970

theorem Betty_wallet_contribution_ratio :
  let wallet_cost : ℚ := 100
  let initial_savings : ℚ := wallet_cost / 2
  let parents_contribution : ℚ := 15
  let remaining_need : ℚ := 5
  let grandparents_contribution : ℚ := wallet_cost - initial_savings - parents_contribution - remaining_need
  grandparents_contribution / parents_contribution = 2 := by
    sorry

end NUMINAMATH_CALUDE_Betty_wallet_contribution_ratio_l2399_239970


namespace NUMINAMATH_CALUDE_geometric_ratio_from_arithmetic_l2399_239902

/-- An arithmetic sequence with a non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, b (n + 1) = r * b n

/-- The theorem statement -/
theorem geometric_ratio_from_arithmetic (a : ℕ → ℝ) (d : ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a d →
  (∃ k, b k = a 1 ∧ b (k + 1) = a 3 ∧ b (k + 2) = a 7) →
  ∃ r, geometric_sequence b r ∧ r = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_ratio_from_arithmetic_l2399_239902


namespace NUMINAMATH_CALUDE_variable_value_l2399_239972

theorem variable_value (x y : ℤ) (h1 : 2 * x - y = 11) (h2 : 4 * x + y ≠ 17) : y = -9 := by
  sorry

end NUMINAMATH_CALUDE_variable_value_l2399_239972
