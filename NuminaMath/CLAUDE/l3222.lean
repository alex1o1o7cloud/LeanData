import Mathlib

namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3222_322235

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 36 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3222_322235


namespace NUMINAMATH_CALUDE_expression_evaluation_l3222_322287

theorem expression_evaluation : (4 + 6 + 2) / 3 - 2 / 3 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3222_322287


namespace NUMINAMATH_CALUDE_sin_3x_periodic_l3222_322206

/-- The function f(x) = sin(3x) is periodic with period 2π/3 -/
theorem sin_3x_periodic (x : ℝ) : Real.sin (3 * (x + 2 * Real.pi / 3)) = Real.sin (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_3x_periodic_l3222_322206


namespace NUMINAMATH_CALUDE_subset_implies_m_leq_two_l3222_322205

def A : Set ℝ := {x | x < 2}
def B (m : ℝ) : Set ℝ := {x | x < m}

theorem subset_implies_m_leq_two (m : ℝ) : B m ⊆ A → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_leq_two_l3222_322205


namespace NUMINAMATH_CALUDE_two_distinct_roots_condition_l3222_322201

theorem two_distinct_roots_condition (b : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ |2^x - 1| = b ∧ |2^y - 1| = b) ↔ 0 < b ∧ b < 1 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_roots_condition_l3222_322201


namespace NUMINAMATH_CALUDE_remaining_payment_l3222_322222

def part_payment : ℝ := 875
def payment_percentage : ℝ := 0.25

theorem remaining_payment :
  let total_cost := part_payment / payment_percentage
  let remaining := total_cost - part_payment
  remaining = 2625 := by sorry

end NUMINAMATH_CALUDE_remaining_payment_l3222_322222


namespace NUMINAMATH_CALUDE_temporary_worker_percentage_is_18_l3222_322232

/-- Represents the composition of workers in a factory -/
structure WorkerComposition where
  total : ℝ
  technician_ratio : ℝ
  non_technician_ratio : ℝ
  permanent_technician_ratio : ℝ
  permanent_non_technician_ratio : ℝ
  (total_positive : total > 0)
  (technician_ratio_valid : technician_ratio ≥ 0 ∧ technician_ratio ≤ 1)
  (non_technician_ratio_valid : non_technician_ratio ≥ 0 ∧ non_technician_ratio ≤ 1)
  (ratios_sum_to_one : technician_ratio + non_technician_ratio = 1)
  (permanent_technician_ratio_valid : permanent_technician_ratio ≥ 0 ∧ permanent_technician_ratio ≤ 1)
  (permanent_non_technician_ratio_valid : permanent_non_technician_ratio ≥ 0 ∧ permanent_non_technician_ratio ≤ 1)

/-- Calculates the percentage of temporary workers in the factory -/
def temporaryWorkerPercentage (w : WorkerComposition) : ℝ :=
  (1 - (w.technician_ratio * w.permanent_technician_ratio + w.non_technician_ratio * w.permanent_non_technician_ratio)) * 100

/-- Theorem stating that given the specific worker composition, the percentage of temporary workers is 18% -/
theorem temporary_worker_percentage_is_18 (w : WorkerComposition)
  (h1 : w.technician_ratio = 0.9)
  (h2 : w.non_technician_ratio = 0.1)
  (h3 : w.permanent_technician_ratio = 0.9)
  (h4 : w.permanent_non_technician_ratio = 0.1) :
  temporaryWorkerPercentage w = 18 := by
  sorry


end NUMINAMATH_CALUDE_temporary_worker_percentage_is_18_l3222_322232


namespace NUMINAMATH_CALUDE_candy_probability_l3222_322218

/-- The number of red candies initially in the jar -/
def red_candies : ℕ := 15

/-- The number of blue candies initially in the jar -/
def blue_candies : ℕ := 15

/-- The total number of candies initially in the jar -/
def total_candies : ℕ := red_candies + blue_candies

/-- The number of candies Terry picks -/
def terry_picks : ℕ := 3

/-- The number of candies Mary picks -/
def mary_picks : ℕ := 2

/-- The probability that Terry and Mary pick candies of the same color -/
def same_color_probability : ℚ := 8008 / 142221

theorem candy_probability : 
  same_color_probability = 
    (Nat.choose red_candies terry_picks * Nat.choose (red_candies - terry_picks) mary_picks + 
     Nat.choose blue_candies terry_picks * Nat.choose (blue_candies - terry_picks) mary_picks) / 
    (Nat.choose total_candies terry_picks * Nat.choose (total_candies - terry_picks) mary_picks) :=
sorry

end NUMINAMATH_CALUDE_candy_probability_l3222_322218


namespace NUMINAMATH_CALUDE_fish_given_by_sister_l3222_322263

/-- Given that Mrs. Sheridan initially had 22 fish and now has 69 fish
    after receiving fish from her sister, prove that the number of fish
    her sister gave her is 47. -/
theorem fish_given_by_sister
  (initial_fish : ℕ)
  (final_fish : ℕ)
  (h1 : initial_fish = 22)
  (h2 : final_fish = 69) :
  final_fish - initial_fish = 47 := by
  sorry

end NUMINAMATH_CALUDE_fish_given_by_sister_l3222_322263


namespace NUMINAMATH_CALUDE_geometric_sequence_with_unit_modulus_ratio_l3222_322225

theorem geometric_sequence_with_unit_modulus_ratio (α : ℝ) : 
  let a : ℕ → ℂ := λ n => Complex.cos (n * α) + Complex.I * Complex.sin (n * α)
  ∃ r : ℂ, (∀ n : ℕ, a (n + 1) = r * a n) ∧ Complex.abs r = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_with_unit_modulus_ratio_l3222_322225


namespace NUMINAMATH_CALUDE_max_value_squared_l3222_322249

theorem max_value_squared (a b x y : ℝ) : 
  a > 0 → b > 0 → a ≥ b → 
  0 ≤ x → x < a → 
  0 ≤ y → y < b → 
  a^2 + y^2 = b^2 + x^2 → 
  (a - x)^2 + (b + y)^2 = b^2 + x^2 →
  x = a - 2*b →
  y = b/2 →
  (∀ ρ : ℝ, (a/b)^2 ≤ ρ^2 → ρ^2 ≤ 4/9) :=
by sorry

end NUMINAMATH_CALUDE_max_value_squared_l3222_322249


namespace NUMINAMATH_CALUDE_xiaoqiang_games_l3222_322214

/-- Represents a participant in the chess tournament -/
inductive Participant
  | Jia
  | Yi
  | Bing
  | Ding
  | Xiaoqiang

/-- The number of games played by each participant -/
def games_played (p : Participant) : ℕ :=
  match p with
  | Participant.Jia => 4
  | Participant.Yi => 3
  | Participant.Bing => 2
  | Participant.Ding => 1
  | Participant.Xiaoqiang => 2  -- This is what we want to prove

/-- The total number of games played in the tournament -/
def total_games : ℕ := 10  -- (5 choose 2) = 10

theorem xiaoqiang_games :
  games_played Participant.Xiaoqiang = 2 :=
by sorry

end NUMINAMATH_CALUDE_xiaoqiang_games_l3222_322214


namespace NUMINAMATH_CALUDE_square_sum_minus_one_le_zero_l3222_322261

theorem square_sum_minus_one_le_zero (a b : ℝ) :
  a^2 + b^2 - 1 - a^2 * b^2 ≤ 0 ↔ (a^2 - 1) * (b^2 - 1) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_square_sum_minus_one_le_zero_l3222_322261


namespace NUMINAMATH_CALUDE_student_count_l3222_322244

theorem student_count : ∃ n : ℕ, n > 0 ∧ 
  (n / 2 : ℚ) + (n / 4 : ℚ) + (n / 8 : ℚ) + 3 = n ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l3222_322244


namespace NUMINAMATH_CALUDE_number_difference_l3222_322240

theorem number_difference (N : ℝ) (h : 0.25 * N = 100) : N - (3/4 * N) = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3222_322240


namespace NUMINAMATH_CALUDE_storm_rainfall_l3222_322293

theorem storm_rainfall (first_hour : ℝ) (second_hour : ℝ) : 
  second_hour = 2 * first_hour + 7 →
  first_hour + second_hour = 22 →
  first_hour = 5 := by
sorry

end NUMINAMATH_CALUDE_storm_rainfall_l3222_322293


namespace NUMINAMATH_CALUDE_no_prime_solutions_for_equation_l3222_322203

theorem no_prime_solutions_for_equation : ¬∃ (x y z : ℕ), Prime x ∧ Prime y ∧ Prime z ∧ x^2 + y^3 = z^4 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solutions_for_equation_l3222_322203


namespace NUMINAMATH_CALUDE_age_difference_l3222_322217

theorem age_difference (father_age : ℕ) (son_age_5_years_ago : ℕ) :
  father_age = 38 →
  son_age_5_years_ago = 14 →
  father_age - (son_age_5_years_ago + 5) = 19 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3222_322217


namespace NUMINAMATH_CALUDE_probability_3_successes_in_7_trials_value_l3222_322267

/-- The probability of getting exactly 3 successes in 7 independent trials,
    where the probability of success in each trial is 3/7. -/
def probability_3_successes_in_7_trials : ℚ :=
  (Nat.choose 7 3 : ℚ) * (3/7)^3 * (4/7)^4

/-- Theorem stating that the probability of getting exactly 3 successes
    in 7 independent trials, where the probability of success in each trial
    is 3/7, is equal to 242112/823543. -/
theorem probability_3_successes_in_7_trials_value :
  probability_3_successes_in_7_trials = 242112/823543 := by
  sorry

end NUMINAMATH_CALUDE_probability_3_successes_in_7_trials_value_l3222_322267


namespace NUMINAMATH_CALUDE_blueberries_count_l3222_322295

/-- The number of strawberries in each red box -/
def strawberries_per_red_box : ℕ := 100

/-- The difference between strawberries in a red box and blueberries in a blue box -/
def berry_difference : ℕ := 30

/-- The number of blueberries in each blue box -/
def blueberries_per_blue_box : ℕ := strawberries_per_red_box - berry_difference

theorem blueberries_count : blueberries_per_blue_box = 70 := by
  sorry

end NUMINAMATH_CALUDE_blueberries_count_l3222_322295


namespace NUMINAMATH_CALUDE_equation_solution_l3222_322238

theorem equation_solution (a : ℝ) (ha : a < 0) :
  ∃! x : ℝ, x * |x| + |x| - x - a = 0 ∧ x = -1 - Real.sqrt (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3222_322238


namespace NUMINAMATH_CALUDE_green_ducks_percentage_in_larger_pond_l3222_322247

/-- Represents the percentage of green ducks in the larger pond -/
def larger_pond_green_percentage : ℝ := 12

theorem green_ducks_percentage_in_larger_pond :
  let smaller_pond_ducks : ℕ := 30
  let larger_pond_ducks : ℕ := 50
  let smaller_pond_green_percentage : ℝ := 20
  let total_green_percentage : ℝ := 15
  (smaller_pond_green_percentage / 100 * smaller_pond_ducks +
   larger_pond_green_percentage / 100 * larger_pond_ducks) /
  (smaller_pond_ducks + larger_pond_ducks) * 100 = total_green_percentage :=
by sorry

end NUMINAMATH_CALUDE_green_ducks_percentage_in_larger_pond_l3222_322247


namespace NUMINAMATH_CALUDE_regular_polygon_with_40_degree_exterior_angle_has_9_sides_l3222_322236

/-- A regular polygon with an exterior angle of 40° has 9 sides. -/
theorem regular_polygon_with_40_degree_exterior_angle_has_9_sides :
  ∀ (n : ℕ), n > 0 →
  (360 : ℝ) / n = 40 →
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_40_degree_exterior_angle_has_9_sides_l3222_322236


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l3222_322229

/-- The number of students taking both Geometry and History -/
def both_geometry_history : ℕ := 15

/-- The total number of students taking Geometry -/
def total_geometry : ℕ := 30

/-- The number of students taking History only -/
def history_only : ℕ := 15

/-- The number of students taking both Geometry and Science -/
def both_geometry_science : ℕ := 8

/-- The number of students taking Science only -/
def science_only : ℕ := 10

/-- Theorem stating that the number of students taking only one subject is 32 -/
theorem students_taking_one_subject :
  (total_geometry - both_geometry_history - both_geometry_science) + history_only + science_only = 32 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l3222_322229


namespace NUMINAMATH_CALUDE_baker_usual_pastries_l3222_322256

/-- The number of pastries the baker usually sells -/
def usual_pastries : ℕ := sorry

/-- The number of loaves the baker usually sells -/
def usual_loaves : ℕ := 10

/-- The number of pastries sold today -/
def today_pastries : ℕ := 14

/-- The number of loaves sold today -/
def today_loaves : ℕ := 25

/-- The price of a pastry in dollars -/
def pastry_price : ℚ := 2

/-- The price of a loaf in dollars -/
def loaf_price : ℚ := 4

/-- The difference between today's sales and average sales in dollars -/
def sales_difference : ℚ := 48

theorem baker_usual_pastries : 
  usual_pastries = 20 :=
by sorry

end NUMINAMATH_CALUDE_baker_usual_pastries_l3222_322256


namespace NUMINAMATH_CALUDE_ninth_term_is_15_l3222_322231

/-- An arithmetic sequence with properties S3 = 3 and S6 = 24 -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n / 2) * (a 1 + a n)
  S3_eq_3 : S 3 = 3
  S6_eq_24 : S 6 = 24

/-- The 9th term of the arithmetic sequence is 15 -/
theorem ninth_term_is_15 (seq : ArithmeticSequence) : seq.a 9 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_15_l3222_322231


namespace NUMINAMATH_CALUDE_circle_y_is_eleven_l3222_322251

/-- Represents the configuration of numbers in the circles. -/
structure CircleConfig where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  x : ℤ
  y : ℤ

/-- The conditions given in the problem. -/
def satisfiesConditions (config : CircleConfig) : Prop :=
  config.a + config.b + config.x = 30 ∧
  config.c + config.d + config.y = 30 ∧
  config.a + config.b + config.c + config.d = 40 ∧
  config.x + config.y + config.c + config.b = 40 ∧
  config.x = 9

/-- The theorem stating that if the conditions are satisfied, Y must be 11. -/
theorem circle_y_is_eleven (config : CircleConfig) 
  (h : satisfiesConditions config) : config.y = 11 := by
  sorry


end NUMINAMATH_CALUDE_circle_y_is_eleven_l3222_322251


namespace NUMINAMATH_CALUDE_prize_plan_optimal_l3222_322216

/-- Represents the prices and quantities of prizes A and B -/
structure PrizePlan where
  priceA : ℕ
  priceB : ℕ
  quantityA : ℕ
  quantityB : ℕ

/-- Conditions for the prize plan -/
def validPrizePlan (p : PrizePlan) : Prop :=
  3 * p.priceA + 2 * p.priceB = 130 ∧
  5 * p.priceA + 4 * p.priceB = 230 ∧
  p.quantityA + p.quantityB = 20 ∧
  p.quantityA ≥ 2 * p.quantityB

/-- Total cost of the prize plan -/
def totalCost (p : PrizePlan) : ℕ :=
  p.priceA * p.quantityA + p.priceB * p.quantityB

/-- The theorem to be proved -/
theorem prize_plan_optimal (p : PrizePlan) (h : validPrizePlan p) :
  p.priceA = 30 ∧ p.priceB = 20 ∧ p.quantityA = 14 ∧ p.quantityB = 6 ∧ totalCost p = 560 := by
  sorry

end NUMINAMATH_CALUDE_prize_plan_optimal_l3222_322216


namespace NUMINAMATH_CALUDE_chi_square_greater_than_critical_expected_volleyball_recipients_correct_l3222_322271

-- Define the total number of students
def total_students : ℕ := 200

-- Define the number of male and female students
def male_students : ℕ := 100
def female_students : ℕ := 100

-- Define the number of students in Group A (volleyball)
def group_a_total : ℕ := 96

-- Define the number of male students in Group A
def group_a_male : ℕ := 36

-- Define the critical value for α = 0.001
def critical_value : ℚ := 10828 / 1000

-- Define the chi-square statistic
def chi_square : ℚ := 11538 / 1000

-- Define the number of male students selected for stratified sampling
def stratified_sample : ℕ := 25

-- Define the number of students selected for gifts
def gift_recipients : ℕ := 3

-- Define the expected number of volleyball players among gift recipients
def expected_volleyball_recipients : ℚ := 621 / 575

-- Theorem 1: The chi-square value is greater than the critical value
theorem chi_square_greater_than_critical : chi_square > critical_value := by sorry

-- Theorem 2: The expected number of volleyball players among gift recipients is correct
theorem expected_volleyball_recipients_correct : 
  expected_volleyball_recipients = 621 / 575 := by sorry

end NUMINAMATH_CALUDE_chi_square_greater_than_critical_expected_volleyball_recipients_correct_l3222_322271


namespace NUMINAMATH_CALUDE_negation_equivalence_l3222_322286

theorem negation_equivalence (m : ℝ) : 
  (¬ ∃ x < 0, x^2 + 2*x - m > 0) ↔ (∀ x < 0, x^2 + 2*x - m ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3222_322286


namespace NUMINAMATH_CALUDE_additive_implies_odd_l3222_322223

-- Define the property of the function
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem additive_implies_odd (f : ℝ → ℝ) (h : is_additive f) : is_odd f := by
  sorry

end NUMINAMATH_CALUDE_additive_implies_odd_l3222_322223


namespace NUMINAMATH_CALUDE_function_inequality_l3222_322237

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x : ℝ, (deriv^[2] f) x < 2 * f x) : 
  (Real.exp 4034 * f (-2017) > f 0) ∧ (f 2017 < Real.exp 4034 * f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3222_322237


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3222_322221

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3222_322221


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_4th_power_l3222_322276

theorem nearest_integer_to_3_plus_sqrt5_4th_power :
  ∃ n : ℤ, n = 752 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5) ^ 4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5) ^ 4 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_4th_power_l3222_322276


namespace NUMINAMATH_CALUDE_sucrose_concentration_in_mixture_l3222_322241

/-- Concentration of sucrose in a mixture of two solutions --/
theorem sucrose_concentration_in_mixture 
  (conc_A : ℝ) (conc_B : ℝ) (vol_A : ℝ) (vol_B : ℝ) 
  (h1 : conc_A = 15.3) 
  (h2 : conc_B = 27.8) 
  (h3 : vol_A = 45) 
  (h4 : vol_B = 75) : 
  (conc_A * vol_A + conc_B * vol_B) / (vol_A + vol_B) = 
  (15.3 * 45 + 27.8 * 75) / (45 + 75) :=
by sorry

#eval (15.3 * 45 + 27.8 * 75) / (45 + 75)

end NUMINAMATH_CALUDE_sucrose_concentration_in_mixture_l3222_322241


namespace NUMINAMATH_CALUDE_rhombus_diagonals_bisect_l3222_322219

-- Define the property of diagonals bisecting each other
def diagonals_bisect (shape : Type) : Prop := sorry

-- Define the relationship between rhombus and parallelogram
def rhombus_is_parallelogram : Prop := sorry

-- Theorem statement
theorem rhombus_diagonals_bisect :
  diagonals_bisect Parallelogram →
  rhombus_is_parallelogram →
  diagonals_bisect Rhombus := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_bisect_l3222_322219


namespace NUMINAMATH_CALUDE_competition_winners_l3222_322266

theorem competition_winners (total_winners : Nat) (total_score : Nat) 
  (first_place_score : Nat) (second_place_score : Nat) (third_place_score : Nat) :
  total_winners = 5 →
  total_score = 94 →
  first_place_score = 20 →
  second_place_score = 19 →
  third_place_score = 18 →
  ∃ (first_place_winners second_place_winners third_place_winners : Nat),
    first_place_winners = 1 ∧
    second_place_winners = 2 ∧
    third_place_winners = 2 ∧
    first_place_winners + second_place_winners + third_place_winners = total_winners ∧
    first_place_winners * first_place_score + 
    second_place_winners * second_place_score + 
    third_place_winners * third_place_score = total_score :=
by sorry

end NUMINAMATH_CALUDE_competition_winners_l3222_322266


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_sin_equality_l3222_322252

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_sin_equality : 
  (¬ ∃ x : ℝ, x = Real.sin x) ↔ (∀ x : ℝ, x ≠ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_sin_equality_l3222_322252


namespace NUMINAMATH_CALUDE_inscribed_circumscribed_ratio_l3222_322230

/-- An equilateral triangle with inscribed and circumscribed circles -/
structure EquilateralTriangleWithCircles where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The radius of the inscribed circle is positive -/
  r_pos : r > 0
  /-- The radius of the circumscribed circle is positive -/
  R_pos : R > 0

/-- The ratio of the inscribed circle radius to the circumscribed circle radius in an equilateral triangle is 1:2 -/
theorem inscribed_circumscribed_ratio (t : EquilateralTriangleWithCircles) : t.r / t.R = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circumscribed_ratio_l3222_322230


namespace NUMINAMATH_CALUDE_intersection_k_value_l3222_322298

/-- The intersection point of two lines -3x + y = k and 2x + y = 20 when x = -10 -/
def intersection_point : ℝ × ℝ := (-10, 40)

/-- The first line equation: -3x + y = k -/
def line1 (k : ℝ) (p : ℝ × ℝ) : Prop :=
  -3 * p.1 + p.2 = k

/-- The second line equation: 2x + y = 20 -/
def line2 (p : ℝ × ℝ) : Prop :=
  2 * p.1 + p.2 = 20

/-- Theorem: The value of k is 70 given that the lines -3x + y = k and 2x + y = 20 intersect when x = -10 -/
theorem intersection_k_value :
  line2 intersection_point →
  (∃ k, line1 k intersection_point) →
  (∃! k, line1 k intersection_point ∧ k = 70) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_k_value_l3222_322298


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3222_322211

theorem muffin_banana_price_ratio :
  ∀ (m b : ℝ),
  (3 * m + 5 * b > 0) →
  (4 * m + 10 * b = 3 * m + 5 * b + 12) →
  (m / b = 2) :=
by sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3222_322211


namespace NUMINAMATH_CALUDE_fraction_simplification_l3222_322280

theorem fraction_simplification : 
  ((3^2010)^2 - (3^2008)^2) / ((3^2009)^2 - (3^2007)^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3222_322280


namespace NUMINAMATH_CALUDE_intersection_when_a_is_three_possible_values_of_a_l3222_322274

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 3 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x < 1 ∨ x > 6}

-- Theorem 1
theorem intersection_when_a_is_three :
  A 3 ∩ (Set.univ \ B) = {x | 1 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem possible_values_of_a (a : ℝ) :
  a > 0 ∧ A a ∩ B = ∅ → 0 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_three_possible_values_of_a_l3222_322274


namespace NUMINAMATH_CALUDE_age_difference_l3222_322268

theorem age_difference (a b c : ℕ) : 
  b = 12 →
  b = 2 * c →
  a + b + c = 32 →
  a = b + 2 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3222_322268


namespace NUMINAMATH_CALUDE_janet_tulips_l3222_322292

/-- The number of tulips Janet picked -/
def T : ℕ := sorry

/-- The total number of flowers Janet picked -/
def total_flowers : ℕ := T + 11

/-- The number of flowers Janet used -/
def used_flowers : ℕ := 11

/-- The number of extra flowers Janet had -/
def extra_flowers : ℕ := 4

theorem janet_tulips : T = 4 := by sorry

end NUMINAMATH_CALUDE_janet_tulips_l3222_322292


namespace NUMINAMATH_CALUDE_azalea_profit_l3222_322200

/-- Calculates the shearer's payment based on the amount of wool produced -/
def shearer_payment (wool_amount : ℕ) : ℕ :=
  1000 + 
  (if wool_amount > 1000 then 1500 else 0) + 
  (if wool_amount > 2000 then (wool_amount - 2000) / 2 else 0)

/-- Calculates the revenue from wool sales based on quality distribution -/
def wool_revenue (total_wool : ℕ) : ℕ :=
  (total_wool / 2) * 30 +  -- High-quality
  (total_wool * 3 / 10) * 20 +  -- Medium-quality
  (total_wool / 5) * 10  -- Low-quality

theorem azalea_profit :
  let total_wool := 2400
  let revenue := wool_revenue total_wool
  let payment := shearer_payment total_wool
  revenue - payment = 52500 := by sorry

end NUMINAMATH_CALUDE_azalea_profit_l3222_322200


namespace NUMINAMATH_CALUDE_marbles_left_l3222_322259

def initial_marbles : ℕ := 350
def marbles_given : ℕ := 175

theorem marbles_left : initial_marbles - marbles_given = 175 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_l3222_322259


namespace NUMINAMATH_CALUDE_remainder_of_m_l3222_322294

theorem remainder_of_m (m : ℕ) (h1 : m^3 % 7 = 6) (h2 : m^4 % 7 = 4) : m % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_m_l3222_322294


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3222_322272

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 3) :
  let original := (3 / (x - 1) - x - 1) / ((x^2 - 4*x + 4) / (x - 1))
  let simplified := -(x + 2) / (x - 2)
  original = simplified ∧ simplified = -5 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3222_322272


namespace NUMINAMATH_CALUDE_sum_of_angles_with_given_tangents_l3222_322226

theorem sum_of_angles_with_given_tangents (A B C : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  Real.tan A = 1 →
  Real.tan B = 2 →
  Real.tan C = 3 →
  A + B + C = π := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_with_given_tangents_l3222_322226


namespace NUMINAMATH_CALUDE_logarithmic_best_fit_l3222_322202

-- Define the types of functions we're considering
inductive FunctionType
  | Linear
  | Quadratic
  | Exponential
  | Logarithmic

-- Define the characteristics we're looking for
structure GrowthCharacteristics where
  rapid_initial_growth : Bool
  slowing_growth_rate : Bool

-- Define a function that checks if a function type matches the desired characteristics
def matches_characteristics (f : FunctionType) (c : GrowthCharacteristics) : Prop :=
  match f with
  | FunctionType.Linear => false
  | FunctionType.Quadratic => false
  | FunctionType.Exponential => false
  | FunctionType.Logarithmic => c.rapid_initial_growth ∧ c.slowing_growth_rate

-- Theorem statement
theorem logarithmic_best_fit (c : GrowthCharacteristics) 
  (h1 : c.rapid_initial_growth = true) 
  (h2 : c.slowing_growth_rate = true) : 
  ∀ f : FunctionType, matches_characteristics f c ↔ f = FunctionType.Logarithmic :=
sorry

end NUMINAMATH_CALUDE_logarithmic_best_fit_l3222_322202


namespace NUMINAMATH_CALUDE_groupD_correct_l3222_322297

/-- Represents a group of Chinese words -/
structure WordGroup :=
  (words : List String)

/-- Checks if a word is correctly written -/
def isCorrectlyWritten (word : String) : Prop :=
  sorry -- Implementation details omitted

/-- Checks if all words in a group are correctly written -/
def allWordsCorrect (group : WordGroup) : Prop :=
  ∀ word ∈ group.words, isCorrectlyWritten word

/-- The four given groups of words -/
def groupA : WordGroup :=
  ⟨["萌孽", "青鸾", "契合", "苦思冥想", "情深意笃", "骇人听闻"]⟩

def groupB : WordGroup :=
  ⟨["斒斓", "彭觞", "木楔", "虚与委蛇", "肆无忌惮", "殒身不恤"]⟩

def groupC : WordGroup :=
  ⟨["青睐", "气概", "编辑", "呼天抢地", "轻歌慢舞", "长歌当哭"]⟩

def groupD : WordGroup :=
  ⟨["缧绁", "剌谬", "陷阱", "伶仃孤苦", "运筹帷幄", "作壁上观"]⟩

/-- Theorem stating that group D is the only group with all words correctly written -/
theorem groupD_correct :
  allWordsCorrect groupD ∧
  ¬allWordsCorrect groupA ∧
  ¬allWordsCorrect groupB ∧
  ¬allWordsCorrect groupC :=
sorry

end NUMINAMATH_CALUDE_groupD_correct_l3222_322297


namespace NUMINAMATH_CALUDE_alice_score_l3222_322204

theorem alice_score (total_score : ℝ) (other_players : ℕ) (avg_score : ℝ) : 
  total_score = 72 ∧ 
  other_players = 7 ∧ 
  avg_score = 4.7 → 
  total_score - (other_players : ℝ) * avg_score = 39.1 := by
sorry

end NUMINAMATH_CALUDE_alice_score_l3222_322204


namespace NUMINAMATH_CALUDE_exam_score_problem_l3222_322290

theorem exam_score_problem (correct_score : ℕ) (incorrect_score : ℤ) 
  (total_score : ℕ) (correct_answers : ℕ) :
  correct_score = 4 →
  incorrect_score = -1 →
  total_score = 150 →
  correct_answers = 42 →
  ∃ (total_questions : ℕ), 
    total_questions = correct_answers + (correct_score * correct_answers - total_score) := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3222_322290


namespace NUMINAMATH_CALUDE_max_value_of_a_min_value_of_expression_l3222_322277

-- Problem I
theorem max_value_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = |x - 5/2| + |x - a|)
  (h2 : ∀ x, f x ≥ a) :
  a ≤ 5/4 ∧ ∃ x, f x = 5/4 := by
sorry

-- Problem II
theorem min_value_of_expression (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + 2*y + 3*z = 1) :
  3/x + 2/y + 1/z ≥ 16 + 8*Real.sqrt 3 ∧ 
  ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x + 2*y + 3*z = 1 ∧ 
    3/x + 2/y + 1/z = 16 + 8*Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_a_min_value_of_expression_l3222_322277


namespace NUMINAMATH_CALUDE_f_two_zeros_implies_a_range_l3222_322260

/-- The function f(x) defined in terms of the parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * a * x + 3 * a - 5

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * (x^2 - 1)

/-- Theorem stating that if f has at least two zeros, then 1 ≤ a ≤ 5 -/
theorem f_two_zeros_implies_a_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) →
  1 ≤ a ∧ a ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_f_two_zeros_implies_a_range_l3222_322260


namespace NUMINAMATH_CALUDE_sum_of_first_four_terms_l3222_322264

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sum_of_first_four_terms 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 9) 
  (h_a5 : a 5 = 243) : 
  (a 1 + a 2 + a 3 + a 4 = 120) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_four_terms_l3222_322264


namespace NUMINAMATH_CALUDE_eighteen_times_thirtysix_minus_twentyseven_times_eighteen_l3222_322291

theorem eighteen_times_thirtysix_minus_twentyseven_times_eighteen : 
  18 * 36 - 27 * 18 = 162 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_times_thirtysix_minus_twentyseven_times_eighteen_l3222_322291


namespace NUMINAMATH_CALUDE_container_capacity_l3222_322248

theorem container_capacity : ∀ (C : ℝ), 
  (0.30 * C + 27 = 0.75 * C) → C = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l3222_322248


namespace NUMINAMATH_CALUDE_lesogoria_inhabitants_l3222_322243

-- Define the types of inhabitants
inductive Inhabitant
| Elf
| Dwarf

-- Define the types of statements
inductive Statement
| AboutGold
| AboutDwarf
| Other

-- Define a function to determine if a statement is true based on the speaker and the type of statement
def isTruthful (speaker : Inhabitant) (statement : Statement) : Prop :=
  match speaker, statement with
  | Inhabitant.Dwarf, Statement.AboutGold => false
  | Inhabitant.Elf, Statement.AboutDwarf => false
  | _, _ => true

-- Define the statements made by A and B
def statementA : Statement := Statement.AboutGold
def statementB : Statement := Statement.Other

-- Define the theorem
theorem lesogoria_inhabitants :
  ∃ (a b : Inhabitant),
    (isTruthful a statementA = false) ∧
    (isTruthful b statementB = true) ∧
    (a = Inhabitant.Dwarf) ∧
    (b = Inhabitant.Dwarf) :=
  sorry


end NUMINAMATH_CALUDE_lesogoria_inhabitants_l3222_322243


namespace NUMINAMATH_CALUDE_fourth_arrangement_follows_pattern_l3222_322273

/-- Represents the four possible positions in a 2x2 grid --/
inductive Position
| topLeft
| topRight
| bottomLeft
| bottomRight

/-- Represents the orientation of the line segment --/
inductive LineOrientation
| horizontal
| vertical

/-- Represents a geometric shape --/
inductive Shape
| circle
| triangle
| square
| line

/-- Represents the arrangement of shapes in a square --/
structure Arrangement where
  circlePos : Position
  trianglePos : Position
  squarePos : Position
  lineOrientation : LineOrientation

/-- The sequence of arrangements in the first three squares --/
def firstThreeArrangements : List Arrangement := [
  { circlePos := Position.topLeft, trianglePos := Position.bottomLeft, 
    squarePos := Position.topRight, lineOrientation := LineOrientation.horizontal },
  { circlePos := Position.bottomLeft, trianglePos := Position.bottomRight, 
    squarePos := Position.topRight, lineOrientation := LineOrientation.vertical },
  { circlePos := Position.bottomRight, trianglePos := Position.topRight, 
    squarePos := Position.bottomLeft, lineOrientation := LineOrientation.horizontal }
]

/-- The predicted arrangement for the fourth square --/
def predictedFourthArrangement : Arrangement :=
  { circlePos := Position.topRight, trianglePos := Position.topLeft, 
    squarePos := Position.bottomLeft, lineOrientation := LineOrientation.vertical }

/-- Theorem stating that the predicted fourth arrangement follows the pattern --/
theorem fourth_arrangement_follows_pattern :
  predictedFourthArrangement = 
    { circlePos := Position.topRight, trianglePos := Position.topLeft, 
      squarePos := Position.bottomLeft, lineOrientation := LineOrientation.vertical } :=
by sorry

end NUMINAMATH_CALUDE_fourth_arrangement_follows_pattern_l3222_322273


namespace NUMINAMATH_CALUDE_equation_two_roots_l3222_322262

-- Define the equation
def equation (x k : ℂ) : Prop :=
  x / (x + 3) + x / (x + 4) = k * x

-- Define the condition for having exactly two distinct roots
def has_two_distinct_roots (k : ℂ) : Prop :=
  ∃ x y : ℂ, x ≠ y ∧ equation x k ∧ equation y k ∧
  ∀ z : ℂ, equation z k → z = x ∨ z = y

-- State the theorem
theorem equation_two_roots :
  ∀ k : ℂ, has_two_distinct_roots k ↔ k = 7/12 ∨ k = 2*I ∨ k = -2*I :=
sorry

end NUMINAMATH_CALUDE_equation_two_roots_l3222_322262


namespace NUMINAMATH_CALUDE_determinant_equality_l3222_322215

theorem determinant_equality (x y z w : ℝ) : 
  x * w - y * z = 7 → (x + z) * w - (y + 2 * w) * z = 7 - w * z := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l3222_322215


namespace NUMINAMATH_CALUDE_rain_probability_both_locations_l3222_322255

theorem rain_probability_both_locations (p_no_rain_A p_no_rain_B : ℝ) 
  (h1 : p_no_rain_A = 0.3)
  (h2 : p_no_rain_B = 0.4)
  (h3 : 0 ≤ p_no_rain_A ∧ p_no_rain_A ≤ 1)
  (h4 : 0 ≤ p_no_rain_B ∧ p_no_rain_B ≤ 1) :
  (1 - p_no_rain_A) * (1 - p_no_rain_B) = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_both_locations_l3222_322255


namespace NUMINAMATH_CALUDE_hades_can_prevent_sisyphus_l3222_322253

/-- Represents the state of the mountain with stones -/
structure MountainState where
  steps : Nat
  stones : Nat
  stone_positions : Finset Nat

/-- Defines the game rules and initial state -/
def initial_state : MountainState :=
  { steps := 1001
  , stones := 500
  , stone_positions := Finset.range 500 }

/-- Sisyphus's move: Lifts a stone to the nearest free step above -/
def sisyphus_move (state : MountainState) : MountainState :=
  sorry

/-- Hades's move: Lowers a stone to the nearest free step below -/
def hades_move (state : MountainState) : MountainState :=
  sorry

/-- Represents a full round of the game (Sisyphus's move followed by Hades's move) -/
def game_round (state : MountainState) : MountainState :=
  hades_move (sisyphus_move state)

/-- Theorem stating that Hades can prevent Sisyphus from reaching the top step -/
theorem hades_can_prevent_sisyphus (state : MountainState := initial_state) :
  ∀ n : Nat, (game_round^[n] state).stone_positions.max < state.steps :=
  sorry

end NUMINAMATH_CALUDE_hades_can_prevent_sisyphus_l3222_322253


namespace NUMINAMATH_CALUDE_division_sum_theorem_l3222_322209

theorem division_sum_theorem (n d : ℕ) (h1 : n = 55) (h2 : d = 11) :
  n + d + (n / d) = 71 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l3222_322209


namespace NUMINAMATH_CALUDE_partition_uniqueness_l3222_322258

/-- An arithmetic progression is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticProgression (a : ℤ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℤ, a (n + 1) = a n + d

/-- A set of integers X can be partitioned into N disjoint increasing
    arithmetic progressions. -/
def CanBePartitioned (X : Set ℤ) (N : ℕ) : Prop :=
  ∃ (partitions : Fin N → Set ℤ),
    (∀ i : Fin N, ∃ a : ℤ → ℤ, ArithmeticProgression a ∧ partitions i = Set.range a) ∧
    (∀ i j : Fin N, i ≠ j → partitions i ∩ partitions j = ∅) ∧
    (⋃ i : Fin N, partitions i) = X

/-- X cannot be partitioned into fewer than N arithmetic progressions. -/
def MinimalPartition (X : Set ℤ) (N : ℕ) : Prop :=
  CanBePartitioned X N ∧ ∀ k < N, ¬CanBePartitioned X k

/-- The partition of X into N arithmetic progressions is unique. -/
def UniquePartition (X : Set ℤ) (N : ℕ) : Prop :=
  ∀ p₁ p₂ : Fin N → Set ℤ,
    (∀ i : Fin N, ∃ a : ℤ → ℤ, ArithmeticProgression a ∧ p₁ i = Set.range a) →
    (∀ i : Fin N, ∃ a : ℤ → ℤ, ArithmeticProgression a ∧ p₂ i = Set.range a) →
    (∀ i j : Fin N, i ≠ j → p₁ i ∩ p₁ j = ∅) →
    (∀ i j : Fin N, i ≠ j → p₂ i ∩ p₂ j = ∅) →
    (⋃ i : Fin N, p₁ i) = X →
    (⋃ i : Fin N, p₂ i) = X →
    ∀ i : Fin N, ∃ j : Fin N, p₁ i = p₂ j

theorem partition_uniqueness (X : Set ℤ) :
  (∀ N : ℕ, MinimalPartition X N → (N = 2 → UniquePartition X N) ∧ (N = 3 → ¬UniquePartition X N)) := by
  sorry

end NUMINAMATH_CALUDE_partition_uniqueness_l3222_322258


namespace NUMINAMATH_CALUDE_minimum_packages_to_breakeven_l3222_322281

def bike_cost : ℕ := 1200
def earning_per_package : ℕ := 15
def maintenance_cost : ℕ := 5

theorem minimum_packages_to_breakeven :
  ∃ n : ℕ, n * (earning_per_package - maintenance_cost) ≥ bike_cost ∧
  ∀ m : ℕ, m * (earning_per_package - maintenance_cost) ≥ bike_cost → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_minimum_packages_to_breakeven_l3222_322281


namespace NUMINAMATH_CALUDE_range_of_f_l3222_322288

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((1 - x) / (1 + x)) + Real.arctan (2 * x)

theorem range_of_f :
  Set.range f = Set.Ioo (-Real.pi / 2) (Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3222_322288


namespace NUMINAMATH_CALUDE_exactly_two_solutions_l3222_322234

/-- The number of ordered pairs (a, b) of positive integers satisfying the equation -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    p.1 > 0 ∧ p.2 > 0 ∧
    2 * p.1 * p.2 + 108 = 15 * Nat.lcm p.1 p.2 + 18 * Nat.gcd p.1 p.2
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The main theorem stating that there are exactly 2 solutions -/
theorem exactly_two_solutions : solution_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_solutions_l3222_322234


namespace NUMINAMATH_CALUDE_sqrt_inequality_square_sum_inequality_l3222_322246

-- Theorem 1
theorem sqrt_inequality (a : ℝ) (h : a > 1) : 
  Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

-- Theorem 2
theorem square_sum_inequality (a b : ℝ) : 
  a^2 + b^2 ≥ a*b + a + b - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_square_sum_inequality_l3222_322246


namespace NUMINAMATH_CALUDE_first_customer_headphones_l3222_322282

/-- The cost of one MP3 player -/
def mp3_cost : ℕ := 120

/-- The cost of one set of headphones -/
def headphone_cost : ℕ := 30

/-- The number of MP3 players bought by the first customer -/
def first_customer_mp3 : ℕ := 5

/-- The total cost of the first customer's purchase -/
def first_customer_total : ℕ := 840

/-- The number of MP3 players bought by the second customer -/
def second_customer_mp3 : ℕ := 3

/-- The number of headphone sets bought by the second customer -/
def second_customer_headphones : ℕ := 4

/-- The total cost of the second customer's purchase -/
def second_customer_total : ℕ := 480

theorem first_customer_headphones :
  ∃ h : ℕ, first_customer_mp3 * mp3_cost + h * headphone_cost = first_customer_total ∧
          h = 8 :=
by sorry

end NUMINAMATH_CALUDE_first_customer_headphones_l3222_322282


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3222_322299

theorem power_of_two_equality (K : ℕ) : 32^2 * 4^5 = 2^K ↔ K = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3222_322299


namespace NUMINAMATH_CALUDE_apples_per_pie_l3222_322208

theorem apples_per_pie (initial_apples : ℕ) (handed_out : ℕ) (num_pies : ℕ) 
  (h1 : initial_apples = 86)
  (h2 : handed_out = 30)
  (h3 : num_pies = 7) :
  (initial_apples - handed_out) / num_pies = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l3222_322208


namespace NUMINAMATH_CALUDE_possible_values_of_C_l3222_322254

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The first number in the problem -/
def number1 (A B : Digit) : ℕ := 9100000 + 10000 * A.val + 300 + 10 * B.val + 2

/-- The second number in the problem -/
def number2 (A B C : Digit) : ℕ := 6000000 + 100000 * A.val + 10000 * B.val + 400 + 50 + 10 * C.val + 2

/-- Theorem stating the possible values of C -/
theorem possible_values_of_C :
  ∀ (A B C : Digit),
    (∃ k : ℕ, number1 A B = 3 * k) →
    (∃ m : ℕ, number2 A B C = 5 * m) →
    C.val = 0 ∨ C.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_C_l3222_322254


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_equality_l3222_322207

theorem max_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) :
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ (Real.sqrt 2 - 1) / 2 := by
sorry

theorem max_value_equality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) :
  (2 * Real.sqrt (a * b) - 4 * a^2 - b^2 = (Real.sqrt 2 - 1) / 2) ↔ (a = 1/4 ∧ b = 1/2) := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_equality_l3222_322207


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l3222_322224

def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 3

theorem min_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 2 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) 2 → f y ≥ f x) ∧
  f x = -37 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l3222_322224


namespace NUMINAMATH_CALUDE_inverse_36_mod_47_l3222_322242

theorem inverse_36_mod_47 (h : (11⁻¹ : ZMod 47) = 43) : (36⁻¹ : ZMod 47) = 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_36_mod_47_l3222_322242


namespace NUMINAMATH_CALUDE_james_excess_calories_james_specific_excess_calories_l3222_322220

/-- Calculates the excess calories eaten by James after eating Cheezits and going for a run. -/
theorem james_excess_calories (bags : Nat) (ounces_per_bag : Nat) (calories_per_ounce : Nat) 
  (run_duration : Nat) (calories_burned_per_minute : Nat) : Nat :=
  let total_calories_consumed := bags * ounces_per_bag * calories_per_ounce
  let total_calories_burned := run_duration * calories_burned_per_minute
  total_calories_consumed - total_calories_burned

/-- Proves that James ate 420 excess calories given the specific conditions. -/
theorem james_specific_excess_calories :
  james_excess_calories 3 2 150 40 12 = 420 := by
  sorry

end NUMINAMATH_CALUDE_james_excess_calories_james_specific_excess_calories_l3222_322220


namespace NUMINAMATH_CALUDE_smallest_taxicab_number_is_smallest_l3222_322285

/-- The smallest positive integer that can be expressed as the sum of two cubes in two different ways -/
def smallest_taxicab_number : ℕ := 1729

/-- A function that checks if a number can be expressed as the sum of two cubes in two different ways -/
def is_taxicab (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a < c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a^3 + b^3 = n ∧ c^3 + d^3 = n ∧ (a, b) ≠ (c, d)

/-- Theorem stating that smallest_taxicab_number is indeed the smallest taxicab number -/
theorem smallest_taxicab_number_is_smallest :
  is_taxicab smallest_taxicab_number ∧
  ∀ m : ℕ, m < smallest_taxicab_number → ¬is_taxicab m :=
sorry

end NUMINAMATH_CALUDE_smallest_taxicab_number_is_smallest_l3222_322285


namespace NUMINAMATH_CALUDE_complex_fraction_magnitude_l3222_322250

theorem complex_fraction_magnitude (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_magnitude_l3222_322250


namespace NUMINAMATH_CALUDE_quilt_block_shaded_half_l3222_322227

/-- Represents a square quilt block divided into a 4x4 grid -/
structure QuiltBlock where
  grid_size : Nat
  full_shaded : Nat
  half_shaded : Nat

/-- Calculates the fraction of shaded area in a quilt block -/
def shaded_fraction (qb : QuiltBlock) : Rat :=
  (qb.full_shaded + qb.half_shaded / 2) / (qb.grid_size * qb.grid_size)

theorem quilt_block_shaded_half :
  ∀ qb : QuiltBlock,
    qb.grid_size = 4 →
    qb.full_shaded = 6 →
    qb.half_shaded = 4 →
    shaded_fraction qb = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quilt_block_shaded_half_l3222_322227


namespace NUMINAMATH_CALUDE_quaternary_2132_equals_septenary_314_l3222_322210

/-- Converts a quaternary (base 4) number represented as a list of digits to decimal (base 10) -/
def quaternary_to_decimal (digits : List Nat) : Nat := sorry

/-- Converts a decimal (base 10) number to septenary (base 7) represented as a list of digits -/
def decimal_to_septenary (n : Nat) : List Nat := sorry

/-- Theorem stating that the quaternary number 2132 is equal to the septenary number 314 -/
theorem quaternary_2132_equals_septenary_314 :
  decimal_to_septenary (quaternary_to_decimal [2, 1, 3, 2]) = [3, 1, 4] := by sorry

end NUMINAMATH_CALUDE_quaternary_2132_equals_septenary_314_l3222_322210


namespace NUMINAMATH_CALUDE_apple_pie_price_per_pound_l3222_322289

/-- Given the following conditions for an apple pie:
  - The pie serves 8 people
  - 2 pounds of apples are needed
  - Pre-made pie crust costs $2.00
  - Lemon costs $0.50
  - Butter costs $1.50
  - Each serving of pie costs $1
  Prove that the price per pound of apples is $2.00 -/
theorem apple_pie_price_per_pound (servings : ℕ) (apple_pounds : ℝ) 
  (crust_cost lemon_cost butter_cost serving_cost : ℝ) :
  servings = 8 → 
  apple_pounds = 2 → 
  crust_cost = 2 → 
  lemon_cost = 0.5 → 
  butter_cost = 1.5 → 
  serving_cost = 1 → 
  (servings * serving_cost - (crust_cost + lemon_cost + butter_cost)) / apple_pounds = 2 :=
by sorry


end NUMINAMATH_CALUDE_apple_pie_price_per_pound_l3222_322289


namespace NUMINAMATH_CALUDE_earliest_solution_l3222_322284

/-- The temperature model as a function of time -/
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 60

/-- The theorem stating that 12.5 is the earliest non-negative solution -/
theorem earliest_solution :
  ∀ t : ℝ, t ≥ 0 → temperature t = 85 → t ≥ 12.5 := by sorry

end NUMINAMATH_CALUDE_earliest_solution_l3222_322284


namespace NUMINAMATH_CALUDE_exists_valid_grid_l3222_322283

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if all elements in a list are equal -/
def allEqual (l : List ℕ) : Prop :=
  l.all (· = l.head!)

/-- Checks if a grid satisfies the required properties -/
def validGrid (g : Grid) : Prop :=
  -- 0 is in the central position
  g 1 1 = 0 ∧
  -- Digits 1-8 are used exactly once each in the remaining positions
  (∀ i : Fin 9, i ≠ 0 → ∃! (r c : Fin 3), g r c = i.val) ∧
  -- The sum of digits in each row and each column is the same
  allEqual [
    g 0 0 + g 0 1 + g 0 2,
    g 1 0 + g 1 1 + g 1 2,
    g 2 0 + g 2 1 + g 2 2,
    g 0 0 + g 1 0 + g 2 0,
    g 0 1 + g 1 1 + g 2 1,
    g 0 2 + g 1 2 + g 2 2
  ]

/-- There exists a grid satisfying the required properties -/
theorem exists_valid_grid : ∃ g : Grid, validGrid g := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_grid_l3222_322283


namespace NUMINAMATH_CALUDE_leadership_combinations_count_l3222_322228

def tribe_size : ℕ := 15
def num_supporting_chiefs : ℕ := 3
def num_inferior_officers_per_chief : ℕ := 2

def leadership_combinations : ℕ := 
  tribe_size * 
  (tribe_size - 1) * 
  (tribe_size - 2) * 
  (tribe_size - 3) * 
  Nat.choose (tribe_size - 4) num_inferior_officers_per_chief *
  Nat.choose (tribe_size - 6) num_inferior_officers_per_chief *
  Nat.choose (tribe_size - 8) num_inferior_officers_per_chief

theorem leadership_combinations_count : leadership_combinations = 19320300 := by
  sorry

end NUMINAMATH_CALUDE_leadership_combinations_count_l3222_322228


namespace NUMINAMATH_CALUDE_adult_ticket_price_l3222_322257

/-- Proves that the price of an adult ticket is $15 given the conditions of the problem -/
theorem adult_ticket_price (total_cost : ℕ) (child_ticket_price : ℕ) (num_children : ℕ) :
  total_cost = 720 →
  child_ticket_price = 8 →
  num_children = 15 →
  ∃ (adult_ticket_price : ℕ),
    adult_ticket_price * (num_children + 25) + child_ticket_price * num_children = total_cost ∧
    adult_ticket_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l3222_322257


namespace NUMINAMATH_CALUDE_unique_solution_exists_l3222_322279

/-- Given a > 0 and a ≠ 1, there exists a unique x such that a^x = log_(1/4) x -/
theorem unique_solution_exists (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  ∃! x : ℝ, a^x = Real.log x / Real.log (1/4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l3222_322279


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l3222_322233

theorem smallest_solution_quadratic : 
  let f : ℝ → ℝ := fun y ↦ 10 * y^2 - 47 * y + 49
  ∃ y : ℝ, f y = 0 ∧ (∀ z : ℝ, f z = 0 → y ≤ z) ∧ y = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l3222_322233


namespace NUMINAMATH_CALUDE_parabola_properties_l3222_322245

/-- A parabola with equation y^2 = 2px and focus at (1,0) -/
structure Parabola where
  p : ℝ
  focus_x : ℝ
  focus_y : ℝ
  h_focus : (focus_x, focus_y) = (1, 0)

/-- The value of p for the parabola -/
def p_value (par : Parabola) : ℝ := par.p

/-- The equation of the directrix for the parabola -/
def directrix_equation (par : Parabola) : ℝ → Prop := fun x ↦ x = -1

theorem parabola_properties (par : Parabola) :
  p_value par = 2 ∧ directrix_equation par = fun x ↦ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3222_322245


namespace NUMINAMATH_CALUDE_ratio_of_a_to_d_l3222_322239

theorem ratio_of_a_to_d (a b c d : ℚ) 
  (hab : a / b = 5 / 3)
  (hbc : b / c = 1 / 5)
  (hcd : c / d = 3 / 2) :
  a / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_d_l3222_322239


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_3456_l3222_322270

/-- Given that 3456 = 2^7 * 3^3, this function counts the number of its positive integer factors that are perfect squares. -/
def count_perfect_square_factors : ℕ :=
  let prime_factorization : List (ℕ × ℕ) := [(2, 7), (3, 3)]
  sorry

/-- The number of positive integer factors of 3456 that are perfect squares is 8. -/
theorem perfect_square_factors_of_3456 : count_perfect_square_factors = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_3456_l3222_322270


namespace NUMINAMATH_CALUDE_exterior_angle_theorem_l3222_322265

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)
  (sum_eq_180 : a + b + c = 180)

-- Define the problem
theorem exterior_angle_theorem (t : Triangle) 
  (h1 : t.a = 45)
  (h2 : t.b = 30) :
  180 - t.c = 75 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_theorem_l3222_322265


namespace NUMINAMATH_CALUDE_original_selling_price_l3222_322275

theorem original_selling_price 
  (P : ℝ) -- Original purchase price
  (S : ℝ) -- Original selling price
  (S_new : ℝ) -- New selling price
  (h1 : S = 1.1 * P) -- Original selling price is 110% of purchase price
  (h2 : S_new = 1.17 * P) -- New selling price based on 10% lower purchase and 30% profit
  (h3 : S_new - S = 63) -- Difference between new and original selling price is $63
  : S = 990 := by
sorry

end NUMINAMATH_CALUDE_original_selling_price_l3222_322275


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_neg_two_l3222_322212

theorem sqrt_meaningful_iff_geq_neg_two (a : ℝ) : 
  (∃ x : ℝ, x^2 = a + 2) ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_neg_two_l3222_322212


namespace NUMINAMATH_CALUDE_star_equation_solution_l3222_322278

/-- Definition of the star operation -/
def star (a b : ℝ) : ℝ := a * b + b - a - 1

/-- Theorem: If 3 star x = 20, then x = 6 -/
theorem star_equation_solution :
  (∃ x : ℝ, star 3 x = 20) → (∃ x : ℝ, star 3 x = 20 ∧ x = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l3222_322278


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_fiftyfive_l3222_322213

theorem largest_multiple_of_seven_below_negative_fiftyfive :
  ∀ n : ℤ, n % 7 = 0 ∧ n < -55 → n ≤ -56 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_fiftyfive_l3222_322213


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3222_322269

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬(253 * m ≡ 989 * m [ZMOD 15])) →
  (253 * n ≡ 989 * n [ZMOD 15]) →
  n = 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3222_322269


namespace NUMINAMATH_CALUDE_continued_fraction_equation_solution_l3222_322296

def continued_fraction (a : ℕ → ℕ) (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (a n : ℚ)⁻¹ + continued_fraction a (n-1)

def left_side (n : ℕ) : ℚ :=
  1 - continued_fraction (fun i => i + 1) n

def right_side (x : ℕ → ℕ) (n : ℕ) : ℚ :=
  continued_fraction x n

theorem continued_fraction_equation_solution (n : ℕ) (h : n ≥ 2) :
  ∃! x : ℕ → ℕ, left_side n = right_side x n ∧
    x 1 = 1 ∧ x 2 = 1 ∧ ∀ i, 3 ≤ i → i ≤ n → x i = i :=
sorry

end NUMINAMATH_CALUDE_continued_fraction_equation_solution_l3222_322296
