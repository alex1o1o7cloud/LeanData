import Mathlib

namespace NUMINAMATH_CALUDE_trim_length_calculation_oliver_trim_purchase_l2889_288943

theorem trim_length_calculation (table_area : Real) (pi_approx : Real) (extra_trim : Real) : Real :=
  let radius := Real.sqrt (table_area / pi_approx)
  let circumference := 2 * pi_approx * radius
  circumference + extra_trim

theorem oliver_trim_purchase :
  trim_length_calculation 616 (22/7) 5 = 93 :=
by sorry

end NUMINAMATH_CALUDE_trim_length_calculation_oliver_trim_purchase_l2889_288943


namespace NUMINAMATH_CALUDE_solution_set_characterization_l2889_288981

def solution_set (x y : ℝ) : Prop :=
  3 * x - 4 * y + 12 > 0 ∧ x + y - 2 < 0

theorem solution_set_characterization (x y : ℝ) :
  solution_set x y ↔ (3 * x - 4 * y + 12 > 0 ∧ x + y - 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l2889_288981


namespace NUMINAMATH_CALUDE_chord_length_line_circle_intersection_specific_chord_length_l2889_288925

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection 
  (line : ℝ → ℝ → Prop) 
  (circle : ℝ → ℝ → Prop) : ℝ :=
by
  sorry

/-- Main theorem: The length of the chord formed by the intersection of 
    x + √3 y - 2 = 0 and x² + y² = 4 is 2√3 -/
theorem specific_chord_length : 
  chord_length_line_circle_intersection 
    (fun x y => x + Real.sqrt 3 * y - 2 = 0) 
    (fun x y => x^2 + y^2 = 4) = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_chord_length_line_circle_intersection_specific_chord_length_l2889_288925


namespace NUMINAMATH_CALUDE_low_key_function_m_range_l2889_288942

def is_t_degree_low_key (f : ℝ → ℝ) (t : ℝ) (C : Set ℝ) : Prop :=
  ∀ x ∈ C, f (x + t) ≤ f x

def f (m : ℝ) (x : ℝ) : ℝ := -|m * x - 3|

theorem low_key_function_m_range :
  ∀ m : ℝ, (is_t_degree_low_key (f m) 6 (Set.Ici 0)) →
    (m ≤ 0 ∨ m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_low_key_function_m_range_l2889_288942


namespace NUMINAMATH_CALUDE_parameter_range_l2889_288941

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (a * x / 3) else 3 * Real.log x / x

/-- The maximum value of f(x) on [-3, 3] is 3/e -/
axiom max_value (a : ℝ) : ∀ x ∈ Set.Icc (-3) 3, f a x ≤ 3 / Real.exp 1

/-- The range of parameter a is [1 - ln(3), +∞) -/
theorem parameter_range :
  {a : ℝ | ∀ x ∈ Set.Icc (-3) 3, f a x ≤ 3 / Real.exp 1} = Set.Ici (1 - Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_parameter_range_l2889_288941


namespace NUMINAMATH_CALUDE_animal_survival_probability_l2889_288979

theorem animal_survival_probability (p_20 p_25 : ℝ) 
  (h1 : p_20 = 0.7) 
  (h2 : p_25 = 0.56) : 
  p_25 / p_20 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_animal_survival_probability_l2889_288979


namespace NUMINAMATH_CALUDE_min_gennadies_for_festival_l2889_288929

/-- Represents the number of people with a specific name -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat

/-- Calculates the minimum number of Gennadies required -/
def minGennadies (counts : NameCount) : Nat :=
  max 0 (counts.borises - 1 - counts.alexanders - counts.vasilies)

/-- Theorem stating the minimum number of Gennadies required for the given scenario -/
theorem min_gennadies_for_festival (counts : NameCount) 
  (h1 : counts.alexanders = 45)
  (h2 : counts.borises = 122)
  (h3 : counts.vasilies = 27) :
  minGennadies counts = 49 := by
  sorry

#eval minGennadies { alexanders := 45, borises := 122, vasilies := 27 }

end NUMINAMATH_CALUDE_min_gennadies_for_festival_l2889_288929


namespace NUMINAMATH_CALUDE_news_spread_time_correct_total_time_l2889_288994

/-- The number of people in the city -/
def city_population : ℕ := 3000000

/-- The time interval in minutes for each round of information spreading -/
def time_interval : ℕ := 10

/-- The number of new people informed by each person in one interval -/
def spread_rate : ℕ := 2

/-- The total number of people who know the news after k intervals -/
def people_informed (k : ℕ) : ℕ := 2^(k+1) - 1

/-- The minimum number of intervals needed to inform the entire city -/
def min_intervals : ℕ := 21

theorem news_spread_time :
  (people_informed min_intervals ≥ city_population) ∧
  (∀ k < min_intervals, people_informed k < city_population) :=
sorry

/-- The total time needed to inform the entire city in minutes -/
def total_time : ℕ := min_intervals * time_interval

theorem correct_total_time : total_time = 210 :=
sorry

end NUMINAMATH_CALUDE_news_spread_time_correct_total_time_l2889_288994


namespace NUMINAMATH_CALUDE_certain_number_proof_l2889_288974

theorem certain_number_proof (x : ℝ) : 
  (x / 3 = 248.14814814814815 / 100 * 162) → x = 1206 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2889_288974


namespace NUMINAMATH_CALUDE_equal_numbers_exist_l2889_288930

/-- A 10x10 grid of integers -/
def Grid := Fin 10 → Fin 10 → ℤ

/-- Two cells are adjacent if they differ by 1 in exactly one coordinate -/
def adjacent (i j i' j' : Fin 10) : Prop :=
  (i = i' ∧ j.val + 1 = j'.val) ∨
  (i = i' ∧ j'.val + 1 = j.val) ∨
  (j = j' ∧ i.val + 1 = i'.val) ∨
  (j = j' ∧ i'.val + 1 = i.val)

/-- The property that adjacent cells differ by at most 5 -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j i' j', adjacent i j i' j' → |g i j - g i' j'| ≤ 5

theorem equal_numbers_exist (g : Grid) (h : valid_grid g) :
  ∃ i j i' j', (i ≠ i' ∨ j ≠ j') ∧ g i j = g i' j' :=
sorry

end NUMINAMATH_CALUDE_equal_numbers_exist_l2889_288930


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l2889_288995

/-- Given six positive consecutive integers starting with c, their average d,
    prove that the average of 7 consecutive integers starting with d is c + 5.5 -/
theorem consecutive_integers_average (c : ℤ) (d : ℚ) : 
  (c > 0) →
  (d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5)) / 6) →
  ((d + (d+1) + (d+2) + (d+3) + (d+4) + (d+5) + (d+6)) / 7 = c + 5.5) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l2889_288995


namespace NUMINAMATH_CALUDE_amount_after_two_years_l2889_288984

/-- The amount after n years with a given initial value and annual increase rate -/
def amountAfterYears (initialValue : ℝ) (increaseRate : ℝ) (years : ℕ) : ℝ :=
  initialValue * (1 + increaseRate) ^ years

/-- Theorem: Given an initial value of 3200 and an annual increase rate of 1/8,
    the value after two years will be 4050 -/
theorem amount_after_two_years :
  amountAfterYears 3200 (1/8) 2 = 4050 := by
  sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l2889_288984


namespace NUMINAMATH_CALUDE_solutions_equality_l2889_288927

theorem solutions_equality (b c : ℝ) : 
  (∀ x : ℝ, (|x - 3| = 4) ↔ (x^2 + b*x + c = 0)) → 
  (b = -6 ∧ c = -7) := by
sorry

end NUMINAMATH_CALUDE_solutions_equality_l2889_288927


namespace NUMINAMATH_CALUDE_at_least_one_woman_probability_l2889_288977

def num_men : ℕ := 9
def num_women : ℕ := 6
def total_people : ℕ := num_men + num_women
def num_selected : ℕ := 4

theorem at_least_one_woman_probability :
  (1 : ℚ) - (Nat.choose num_men num_selected : ℚ) / (Nat.choose total_people num_selected : ℚ) = 13/15 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_woman_probability_l2889_288977


namespace NUMINAMATH_CALUDE_distinct_sequences_is_256_l2889_288917

/-- Represents the number of coin flips -/
def total_flips : ℕ := 10

/-- Represents the number of fixed heads at the start -/
def fixed_heads : ℕ := 2

/-- Represents the number of possible outcomes for each flip -/
def outcomes_per_flip : ℕ := 2

/-- Calculates the number of distinct sequences -/
def distinct_sequences : ℕ := outcomes_per_flip ^ (total_flips - fixed_heads)

/-- Theorem stating that the number of distinct sequences is 256 -/
theorem distinct_sequences_is_256 : distinct_sequences = 256 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sequences_is_256_l2889_288917


namespace NUMINAMATH_CALUDE_function_equality_l2889_288992

theorem function_equality (f : ℝ → ℝ) :
  (∀ x : ℝ, f (2 * x + 1) = 4 * x^2 + 14 * x + 7) →
  (∀ x : ℝ, f x = x^2 + 5 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2889_288992


namespace NUMINAMATH_CALUDE_laura_to_ken_ratio_l2889_288955

/-- The number of tiles Don can paint per minute -/
def D : ℕ := 3

/-- The number of tiles Ken can paint per minute -/
def K : ℕ := D + 2

/-- The number of tiles Laura can paint per minute -/
def L : ℕ := 10

/-- The number of tiles Kim can paint per minute -/
def Kim : ℕ := L - 3

/-- The total number of tiles painted by all four people in 15 minutes -/
def total_tiles : ℕ := 375

/-- The theorem stating that the ratio of Laura's painting rate to Ken's painting rate is 2:1 -/
theorem laura_to_ken_ratio :
  (L : ℚ) / K = 2 / 1 ∧ 15 * (D + K + L + Kim) = total_tiles :=
by sorry

end NUMINAMATH_CALUDE_laura_to_ken_ratio_l2889_288955


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2889_288944

theorem fractional_equation_solution :
  ∃ x : ℝ, (((1 - x) / (2 - x)) - 1 = ((2 * x - 5) / (x - 2))) ∧ x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2889_288944


namespace NUMINAMATH_CALUDE_tim_meditates_one_hour_per_day_l2889_288982

/-- Tim's weekly schedule -/
structure TimSchedule where
  reading_time_per_week : ℝ
  meditation_time_per_day : ℝ

/-- Tim's schedule satisfies the given conditions -/
def valid_schedule (s : TimSchedule) : Prop :=
  s.reading_time_per_week = 14 ∧
  s.reading_time_per_week = 2 * (7 * s.meditation_time_per_day)

/-- Theorem: Tim meditates 1 hour per day -/
theorem tim_meditates_one_hour_per_day (s : TimSchedule) (h : valid_schedule s) :
  s.meditation_time_per_day = 1 := by
  sorry

end NUMINAMATH_CALUDE_tim_meditates_one_hour_per_day_l2889_288982


namespace NUMINAMATH_CALUDE_sequence_2007th_term_l2889_288932

theorem sequence_2007th_term (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 0 = 1 →
  (∀ n, a (n + 2) = 6 * a n - a (n + 1)) →
  a 2007 = 2^2007 := by
sorry

end NUMINAMATH_CALUDE_sequence_2007th_term_l2889_288932


namespace NUMINAMATH_CALUDE_plant_mass_problem_l2889_288926

theorem plant_mass_problem (initial_mass : ℝ) : 
  (((initial_mass * 3 + 4) * 3 + 4) * 3 + 4 = 133) → initial_mass = 3 := by
sorry

end NUMINAMATH_CALUDE_plant_mass_problem_l2889_288926


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2889_288988

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y + 20) :
  x + y = 12 + 2 * Real.sqrt 6 ∨ x + y = 12 - 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2889_288988


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2889_288952

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2889_288952


namespace NUMINAMATH_CALUDE_integral_nonnegative_function_integral_positive_at_point_l2889_288947

open MeasureTheory
open Measure
open Set
open Interval

theorem integral_nonnegative_function
  {a b : ℝ} (hab : a ≤ b)
  {f : ℝ → ℝ} (hf : ContinuousOn f (Icc a b))
  (hfnonneg : ∀ x ∈ Icc a b, f x ≥ 0) :
  ∫ x in a..b, f x ≥ 0 :=
sorry

theorem integral_positive_at_point
  {a b : ℝ} (hab : a ≤ b)
  {f : ℝ → ℝ} (hf : ContinuousOn f (Icc a b))
  (hfnonneg : ∀ x ∈ Icc a b, f x ≥ 0)
  (x₀ : ℝ) (hx₀ : x₀ ∈ Icc a b) (hfx₀ : f x₀ > 0) :
  ∫ x in a..b, f x > 0 :=
sorry

end NUMINAMATH_CALUDE_integral_nonnegative_function_integral_positive_at_point_l2889_288947


namespace NUMINAMATH_CALUDE_no_ratio_p_squared_l2889_288993

theorem no_ratio_p_squared (p : ℕ) (hp : Prime p) :
  ∀ (x y l : ℕ), l ≥ 1 →
    (x * (x + 1)) / (y * (y + 1)) ≠ p^(2 * l) :=
by sorry

end NUMINAMATH_CALUDE_no_ratio_p_squared_l2889_288993


namespace NUMINAMATH_CALUDE_houses_visited_per_day_l2889_288948

-- Define the parameters
def buyerPercentage : Real := 0.2
def cheapKnivesPrice : Real := 50
def expensiveKnivesPrice : Real := 150
def weeklyRevenue : Real := 5000
def workDaysPerWeek : Nat := 5

-- Define the theorem
theorem houses_visited_per_day :
  ∃ (housesPerDay : Nat),
    (housesPerDay : Real) * buyerPercentage * 
    ((cheapKnivesPrice + expensiveKnivesPrice) / 2) * 
    (workDaysPerWeek : Real) = weeklyRevenue ∧
    housesPerDay = 50 := by
  sorry

end NUMINAMATH_CALUDE_houses_visited_per_day_l2889_288948


namespace NUMINAMATH_CALUDE_inequality_proof_l2889_288965

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) : 
  a / (a + 2*b)^(1/3) + b / (b + 2*c)^(1/3) + c / (c + 2*a)^(1/3) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2889_288965


namespace NUMINAMATH_CALUDE_group_total_sum_l2889_288973

/-- The total sum spent by a group of friends on dinner and a gift -/
def total_sum (num_friends : ℕ) (additional_payment : ℚ) (gift_cost : ℚ) : ℚ :=
  let dinner_cost := (num_friends - 1) * additional_payment * 10
  dinner_cost + gift_cost

/-- Proof of the total sum spent by the group -/
theorem group_total_sum :
  total_sum 10 3 15 = 285 := by
  sorry

end NUMINAMATH_CALUDE_group_total_sum_l2889_288973


namespace NUMINAMATH_CALUDE_min_sum_squares_of_roots_l2889_288903

theorem min_sum_squares_of_roots (m : ℝ) (α β : ℝ) : 
  (4 * α^2 - 4*m*α + m + 2 = 0) →
  (4 * β^2 - 4*m*β + m + 2 = 0) →
  (∀ m' α' β' : ℝ, (4 * α'^2 - 4*m'*α' + m' + 2 = 0) → 
                   (4 * β'^2 - 4*m'*β' + m' + 2 = 0) → 
                   α^2 + β^2 ≤ α'^2 + β'^2) →
  m = -1 ∧ α^2 + β^2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_of_roots_l2889_288903


namespace NUMINAMATH_CALUDE_parent_age_problem_l2889_288915

/-- Given the conditions about the relationship between a parent's age and their daughter's age,
    prove that the parent's current age is 40 years. -/
theorem parent_age_problem (Y D : ℕ) : 
  Y = 4 * D →                 -- You are 4 times your daughter's age today
  Y - 7 = 11 * (D - 7) →      -- 7 years earlier, you were 11 times her age
  Y = 40                      -- Your current age is 40
:= by sorry

end NUMINAMATH_CALUDE_parent_age_problem_l2889_288915


namespace NUMINAMATH_CALUDE_calculate_expression_l2889_288958

theorem calculate_expression : (1/2)⁻¹ + |3 - Real.sqrt 12| + (-1)^2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2889_288958


namespace NUMINAMATH_CALUDE_equation_linear_iff_k_eq_neg_two_l2889_288970

/-- The equation (k-2)x^(|k|-1) = k+1 is linear in x if and only if k = -2 -/
theorem equation_linear_iff_k_eq_neg_two :
  ∀ k : ℤ, (∃ a b : ℝ, ∀ x : ℝ, (k - 2) * x^(|k| - 1) = a * x + b) ↔ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_linear_iff_k_eq_neg_two_l2889_288970


namespace NUMINAMATH_CALUDE_unique_number_exists_l2889_288916

theorem unique_number_exists : ∃! N : ℚ, (1 / 3) * N = 8 ∧ (1 / 8) * N = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_exists_l2889_288916


namespace NUMINAMATH_CALUDE_pig_price_calculation_l2889_288909

/-- Given a total of 3 pigs and 10 hens costing Rs. 1200 in total,
    with hens costing an average of Rs. 30 each,
    prove that the average price of a pig is Rs. 300. -/
theorem pig_price_calculation (total_cost : ℕ) (num_pigs num_hens : ℕ) (avg_hen_price : ℕ) :
  total_cost = 1200 →
  num_pigs = 3 →
  num_hens = 10 →
  avg_hen_price = 30 →
  (total_cost - num_hens * avg_hen_price) / num_pigs = 300 := by
  sorry

end NUMINAMATH_CALUDE_pig_price_calculation_l2889_288909


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2889_288956

/-- A positive geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1)^2 - 10 * (a 1) + 16 = 0 →
  (a 19)^2 - 10 * (a 19) + 16 = 0 →
  a 8 * a 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2889_288956


namespace NUMINAMATH_CALUDE_inequality_proof_l2889_288991

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2889_288991


namespace NUMINAMATH_CALUDE_gcd_8675309_7654321_l2889_288964

theorem gcd_8675309_7654321 : Nat.gcd 8675309 7654321 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8675309_7654321_l2889_288964


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l2889_288961

theorem arctan_sum_equals_pi_over_four :
  ∃ (n : ℕ), n > 0 ∧ Real.arctan (1/3) + Real.arctan (1/5) + Real.arctan (1/7) + Real.arctan (1/n : ℝ) = π/4 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l2889_288961


namespace NUMINAMATH_CALUDE_only_one_true_iff_in_range_l2889_288919

/-- The proposition p: no solution for the quadratic inequality -/
def p (a : ℝ) : Prop := a > 0 ∧ ∀ x, x^2 + (a-1)*x + a^2 > 0

/-- The proposition q: probability condition -/
def q (a : ℝ) : Prop := a > 0 ∧ (min a 4 + 2) / 6 ≥ 5/6

/-- The main theorem -/
theorem only_one_true_iff_in_range (a : ℝ) :
  (p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ a > 1/3 ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_only_one_true_iff_in_range_l2889_288919


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2889_288969

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 8*x + 9 = 0) ↔ ((x - 4)^2 = 7) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2889_288969


namespace NUMINAMATH_CALUDE_anna_spending_l2889_288998

/-- Anna's spending problem -/
theorem anna_spending (original : ℚ) (left : ℚ) (h1 : original = 32) (h2 : left = 24) :
  (original - left) / original = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_anna_spending_l2889_288998


namespace NUMINAMATH_CALUDE_equation_solution_l2889_288978

theorem equation_solution : ∃ x : ℝ, 20 * 14 + x = 20 + 14 * x ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2889_288978


namespace NUMINAMATH_CALUDE_five_letter_words_count_l2889_288953

def alphabet_size : Nat := 26
def excluded_letter : Nat := 1

theorem five_letter_words_count :
  let available_letters := alphabet_size - excluded_letter
  (available_letters ^ 4 : Nat) = 390625 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l2889_288953


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2889_288997

theorem pure_imaginary_complex_number (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 - 3*x + 2) (x - 1)
  (z.re = 0 ∧ z.im ≠ 0) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2889_288997


namespace NUMINAMATH_CALUDE_a_3_value_l2889_288938

def sequence_a (n : ℕ+) : ℚ := 1 / (n.val * (n.val + 1))

theorem a_3_value : sequence_a 3 = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_a_3_value_l2889_288938


namespace NUMINAMATH_CALUDE_right_triangle_properties_l2889_288910

/-- Right triangle ABC with given properties -/
structure RightTriangleABC where
  -- AB is the hypotenuse
  AB : ℝ
  BC : ℝ
  angleC : ℝ
  hypotenuse_length : AB = 5
  leg_length : BC = 3
  right_angle : angleC = 90

/-- The length of the altitude to the hypotenuse in the right triangle ABC -/
def altitude_to_hypotenuse (t : RightTriangleABC) : ℝ := 2.4

/-- The length of the median to the hypotenuse in the right triangle ABC -/
def median_to_hypotenuse (t : RightTriangleABC) : ℝ := 2.5

/-- Theorem stating the properties of the right triangle ABC -/
theorem right_triangle_properties (t : RightTriangleABC) :
  altitude_to_hypotenuse t = 2.4 ∧ median_to_hypotenuse t = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l2889_288910


namespace NUMINAMATH_CALUDE_least_positive_integer_divisibility_l2889_288960

theorem least_positive_integer_divisibility : ∃ d : ℕ+, 
  (∀ k : ℕ+, k < d → ¬(13 ∣ (k^3 + 1000))) ∧ (13 ∣ (d^3 + 1000)) ∧ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisibility_l2889_288960


namespace NUMINAMATH_CALUDE_prime_4k_plus_1_sum_of_squares_l2889_288957

theorem prime_4k_plus_1_sum_of_squares (p : ℕ) (k : ℕ) :
  Prime p → p = 4 * k + 1 → ∃ x r : ℤ, p = x^2 + r^2 := by sorry

end NUMINAMATH_CALUDE_prime_4k_plus_1_sum_of_squares_l2889_288957


namespace NUMINAMATH_CALUDE_harry_work_hours_l2889_288999

/-- Given the payment conditions for Harry and James, prove that if James worked 41 hours
    and they were paid the same amount, then Harry worked 39 hours. -/
theorem harry_work_hours (x : ℝ) (h : ℝ) :
  let harry_pay := 24 * x + (h - 24) * 1.5 * x
  let james_pay := 24 * x + (41 - 24) * 2 * x
  harry_pay = james_pay →
  h = 39 := by
  sorry

end NUMINAMATH_CALUDE_harry_work_hours_l2889_288999


namespace NUMINAMATH_CALUDE_doughnuts_count_l2889_288976

/-- Given a ratio of doughnuts to muffins and the number of muffins, 
    calculate the number of doughnuts -/
def calculate_doughnuts (ratio_doughnuts : ℕ) (ratio_muffins : ℕ) (num_muffins : ℕ) : ℕ :=
  (ratio_doughnuts * num_muffins) / ratio_muffins

/-- Theorem stating that given a ratio of 5:1 for doughnuts to muffins 
    and 10 muffins, there are 50 doughnuts -/
theorem doughnuts_count : calculate_doughnuts 5 1 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_count_l2889_288976


namespace NUMINAMATH_CALUDE_inequality_solution_l2889_288924

theorem inequality_solution (x : ℝ) : 
  (x^3 - 4*x) / (x^2 - 4*x + 4) > 0 ↔ (x > -2 ∧ x < 0) ∨ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2889_288924


namespace NUMINAMATH_CALUDE_cruise_ship_theorem_l2889_288972

def cruise_ship_problem (min_capacity max_capacity current_passengers : ℕ) : ℕ :=
  if min_capacity ≤ current_passengers then 0
  else min_capacity - current_passengers

theorem cruise_ship_theorem :
  cruise_ship_problem 16 30 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cruise_ship_theorem_l2889_288972


namespace NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l2889_288966

/-- A quadrilateral with perpendicular diagonals and equal adjacent sides, but not all sides equal -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perpendicular_diagonals : Bool
  /-- Each pair of adjacent sides is equal -/
  adjacent_sides_equal : Bool
  /-- Not all sides are equal -/
  not_all_sides_equal : Bool

/-- Definition of a kite -/
def is_kite (q : SpecialQuadrilateral) : Prop :=
  q.perpendicular_diagonals ∧ q.adjacent_sides_equal ∧ q.not_all_sides_equal

/-- Theorem stating that the given quadrilateral is a kite -/
theorem special_quadrilateral_is_kite (q : SpecialQuadrilateral) 
  (h1 : q.perpendicular_diagonals = true) 
  (h2 : q.adjacent_sides_equal = true) 
  (h3 : q.not_all_sides_equal = true) : 
  is_kite q :=
sorry

end NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l2889_288966


namespace NUMINAMATH_CALUDE_cosine_sum_zero_l2889_288935

theorem cosine_sum_zero (n : ℤ) (h : n % 7 = 1 ∨ n % 7 = 3 ∨ n % 7 = 4) :
  Real.cos (n * π / 7 - 13 * π / 14) + 
  Real.cos (3 * n * π / 7 - 3 * π / 14) + 
  Real.cos (5 * n * π / 7 - 3 * π / 14) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_zero_l2889_288935


namespace NUMINAMATH_CALUDE_is_root_of_polynomial_l2889_288939

theorem is_root_of_polynomial (x : ℝ) : 
  x = 4 → x^3 - 5*x^2 + 7*x - 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_is_root_of_polynomial_l2889_288939


namespace NUMINAMATH_CALUDE_cylinder_height_increase_l2889_288902

theorem cylinder_height_increase (y : ℝ) : y > 0 →
  (π * (5 + 2)^2 * 4 = π * 5^2 * (4 + y)) → y = 96 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_increase_l2889_288902


namespace NUMINAMATH_CALUDE_chess_piece_loss_l2889_288914

theorem chess_piece_loss (total_pieces : ℕ) (arianna_lost : ℕ) : 
  total_pieces = 20 →
  arianna_lost = 3 →
  32 - total_pieces = arianna_lost + 9 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_piece_loss_l2889_288914


namespace NUMINAMATH_CALUDE_prank_combinations_eq_180_l2889_288975

/-- The number of different combinations for Tim's prank --/
def prank_combinations : ℕ :=
  let monday : ℕ := 1 -- Joe is chosen
  let tuesday : ℕ := 3 -- Ambie, John, or Liz
  let wednesday : ℕ := 5 -- 5 new people
  let thursday : ℕ := 6 -- 6 new people
  let friday : ℕ := 2 -- Tim or Sam
  monday * tuesday * wednesday * thursday * friday

/-- Theorem stating that the number of different combinations is 180 --/
theorem prank_combinations_eq_180 : prank_combinations = 180 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_eq_180_l2889_288975


namespace NUMINAMATH_CALUDE_spoon_set_count_l2889_288905

/-- 
Given a set of spoons costing $21, where 5 spoons would cost $15 if sold separately,
prove that the number of spoons in the set is 7.
-/
theorem spoon_set_count (total_cost : ℚ) (five_spoon_cost : ℚ) (spoon_count : ℕ) : 
  total_cost = 21 →
  five_spoon_cost = 15 →
  (5 : ℚ) * (total_cost / (spoon_count : ℚ)) = five_spoon_cost →
  spoon_count = 7 := by
  sorry

#check spoon_set_count

end NUMINAMATH_CALUDE_spoon_set_count_l2889_288905


namespace NUMINAMATH_CALUDE_gcd_12012_18018_l2889_288904

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12012_18018_l2889_288904


namespace NUMINAMATH_CALUDE_dz_dt_formula_l2889_288936

noncomputable def z (t : ℝ) := Real.arcsin ((2*t)^2 + (4*t^2)^2 + t^2)

theorem dz_dt_formula (t : ℝ) :
  deriv z t = (2*t*(1 + 4*t + 32*t^2)) / Real.sqrt (1 - ((2*t)^2 + (4*t^2)^2 + t^2)^2) :=
sorry

end NUMINAMATH_CALUDE_dz_dt_formula_l2889_288936


namespace NUMINAMATH_CALUDE_log_343_property_l2889_288912

theorem log_343_property (x : ℝ) (h : Real.log (343 : ℝ) / Real.log (3 * x) = x) :
  (∃ (a b : ℤ), x = (a : ℝ) / (b : ℝ)) ∧ 
  (∀ (n : ℕ), n ≥ 2 → ¬∃ (m : ℤ), x = (m : ℝ) ^ (1 / n : ℝ)) ∧
  (¬∃ (n : ℤ), x = (n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_log_343_property_l2889_288912


namespace NUMINAMATH_CALUDE_tan_ratio_inequality_l2889_288954

theorem tan_ratio_inequality (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) :
  (Real.tan α) / α < (Real.tan β) / β := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_inequality_l2889_288954


namespace NUMINAMATH_CALUDE_sum_of_sign_ratios_l2889_288967

theorem sum_of_sign_ratios (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a / |a| + b / |b| + (a * b) / |a * b| = 3 ∨ a / |a| + b / |b| + (a * b) / |a * b| = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sign_ratios_l2889_288967


namespace NUMINAMATH_CALUDE_cube_tetrahedrons_l2889_288949

/-- A cube is a three-dimensional shape with 8 vertices. -/
structure Cube where
  vertices : Finset (Fin 8)
  vertex_count : vertices.card = 8

/-- A tetrahedron is a three-dimensional shape with 4 vertices. -/
structure Tetrahedron where
  vertices : Finset (Fin 8)
  vertex_count : vertices.card = 4

/-- The number of ways to choose 4 vertices from 8 vertices. -/
def total_choices : ℕ := Nat.choose 8 4

/-- The number of sets of 4 vertices that cannot form a tetrahedron. -/
def invalid_choices : ℕ := 12

/-- The function that calculates the number of valid tetrahedrons. -/
def valid_tetrahedrons (c : Cube) : ℕ := total_choices - invalid_choices

/-- Theorem: The number of distinct tetrahedrons that can be formed using the vertices of a cube is 58. -/
theorem cube_tetrahedrons (c : Cube) : valid_tetrahedrons c = 58 := by
  sorry

end NUMINAMATH_CALUDE_cube_tetrahedrons_l2889_288949


namespace NUMINAMATH_CALUDE_uniform_transform_l2889_288907

-- Define a uniform random variable on an interval
def UniformRandom (a b : ℝ) := {X : ℝ → ℝ | ∀ x, a ≤ x ∧ x ≤ b → X x = (b - a)⁻¹}

theorem uniform_transform (b₁ b : ℝ → ℝ) :
  UniformRandom 0 1 b₁ →
  (∀ x, b x = 3 * (b₁ x - 2)) →
  UniformRandom (-6) (-3) b := by
sorry

end NUMINAMATH_CALUDE_uniform_transform_l2889_288907


namespace NUMINAMATH_CALUDE_solution_set_l2889_288990

theorem solution_set (y : ℝ) : 2 ≤ y / (3 * y - 4) ∧ y / (3 * y - 4) < 5 ↔ 10 / 7 < y ∧ y ≤ 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l2889_288990


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2889_288906

theorem quadratic_roots_relation (q : ℝ) : 
  let eq1 := fun x : ℂ => x^2 + 2*x + q
  let eq2 := fun x : ℂ => (1+q)*(x^2 + 2*x + q) - 2*(q-1)*(x^2 + 1)
  (∃ x y : ℝ, x ≠ y ∧ eq1 x = 0 ∧ eq1 y = 0) ↔ 
  (∀ z : ℂ, eq2 z = 0 → z.im ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2889_288906


namespace NUMINAMATH_CALUDE_cargo_realization_time_l2889_288931

/-- Represents the speed of a boat in still water -/
structure BoatSpeed where
  speed : ℝ
  positive : speed > 0

/-- Represents the current speed of the river -/
structure RiverCurrent where
  speed : ℝ

/-- Represents a boat on the river -/
structure Boat where
  speed : BoatSpeed
  position : ℝ
  direction : Bool  -- True for downstream, False for upstream

/-- The time it takes for Boat 1 to realize its cargo is missing -/
def timeToCargo (boat1 : Boat) (boat2 : Boat) (river : RiverCurrent) : ℝ :=
  sorry

/-- Theorem stating that the time taken for Boat 1 to realize its cargo is missing is 40 minutes -/
theorem cargo_realization_time
  (boat1 : Boat)
  (boat2 : Boat)
  (river : RiverCurrent)
  (h1 : boat1.speed.speed = 2 * boat2.speed.speed)
  (h2 : boat1.direction = false)  -- Boat 1 starts upstream
  (h3 : boat2.direction = true)   -- Boat 2 starts downstream
  (h4 : ∃ (t : ℝ), t > 0 ∧ t < timeToCargo boat1 boat2 river ∧ 
        boat1.position + t * (boat1.speed.speed + river.speed) = 
        boat2.position + t * (boat2.speed.speed - river.speed))  -- Boats meet before cargo realization
  (h5 : ∃ (t : ℝ), t = 20 ∧ 
        boat1.position + t * (boat1.speed.speed + river.speed) = 
        boat2.position + t * (boat2.speed.speed - river.speed))  -- Boats meet at 20 minutes
  : timeToCargo boat1 boat2 river = 40 := by
  sorry

end NUMINAMATH_CALUDE_cargo_realization_time_l2889_288931


namespace NUMINAMATH_CALUDE_smallest_advantageous_discount_l2889_288921

theorem smallest_advantageous_discount : ∃ n : ℕ,
  (n : ℝ) > 0 ∧
  (∀ m : ℕ, m < n →
    (1 - m / 100 : ℝ) > (1 - 0.20)^2 ∨
    (1 - m / 100 : ℝ) > (1 - 0.13)^3 ∨
    (1 - m / 100 : ℝ) > (1 - 0.30) * (1 - 0.10)) ∧
  (1 - n / 100 : ℝ) ≤ (1 - 0.20)^2 ∧
  (1 - n / 100 : ℝ) ≤ (1 - 0.13)^3 ∧
  (1 - n / 100 : ℝ) ≤ (1 - 0.30) * (1 - 0.10) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_advantageous_discount_l2889_288921


namespace NUMINAMATH_CALUDE_mark_sugar_intake_excess_l2889_288922

/-- Represents the calorie content and sugar information for a soft drink -/
structure SoftDrink where
  totalCalories : ℕ
  sugarPercentage : ℚ

/-- Represents the sugar content of a candy bar -/
structure CandyBar where
  sugarCalories : ℕ

theorem mark_sugar_intake_excess (drink : SoftDrink) (bar : CandyBar) 
    (h1 : drink.totalCalories = 2500)
    (h2 : drink.sugarPercentage = 5 / 100)
    (h3 : bar.sugarCalories = 25)
    (h4 : (drink.totalCalories : ℚ) * drink.sugarPercentage + 7 * bar.sugarCalories = 300)
    (h5 : (300 : ℚ) / 150 - 1 = 1) : 
    (300 : ℚ) / 150 - 1 = 1 := by sorry

end NUMINAMATH_CALUDE_mark_sugar_intake_excess_l2889_288922


namespace NUMINAMATH_CALUDE_e_is_largest_l2889_288968

-- Define the variables
variable (a b c d e : ℝ)

-- Define the given equation
def equation := (a - 2 = b + 3) ∧ (b + 3 = c - 4) ∧ (c - 4 = d + 5) ∧ (d + 5 = e - 6)

-- Theorem statement
theorem e_is_largest (h : equation a b c d e) : 
  e = max a (max b (max c d)) :=
sorry

end NUMINAMATH_CALUDE_e_is_largest_l2889_288968


namespace NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l2889_288940

theorem sum_of_special_primes_is_prime (A B : ℕ+) : 
  Nat.Prime A.val → 
  Nat.Prime B.val → 
  Nat.Prime (A.val - B.val) → 
  Nat.Prime (A.val - B.val - B.val) → 
  Nat.Prime (A.val + B.val + (A.val - B.val) + (A.val - B.val - B.val)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l2889_288940


namespace NUMINAMATH_CALUDE_basketball_highlight_film_l2889_288900

theorem basketball_highlight_film (point_guard : ℕ) (shooting_guard : ℕ) (small_forward : ℕ) (power_forward : ℕ) :
  point_guard = 130 →
  shooting_guard = 145 →
  small_forward = 85 →
  power_forward = 60 →
  ∃ (center : ℕ),
    center = 180 ∧
    (point_guard + shooting_guard + small_forward + power_forward + center) / 5 = 120 :=
by sorry

end NUMINAMATH_CALUDE_basketball_highlight_film_l2889_288900


namespace NUMINAMATH_CALUDE_probability_of_specific_draw_l2889_288959

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : Nat := 52

/-- Number of 5's in a standard deck -/
def NumberOfFives : Nat := 4

/-- Number of hearts in a standard deck -/
def NumberOfHearts : Nat := 13

/-- Number of Aces in a standard deck -/
def NumberOfAces : Nat := 4

/-- Probability of drawing a 5 as the first card, a heart as the second card, 
    and an Ace as the third card from a standard 52-card deck -/
def probabilityOfSpecificDraw : ℚ :=
  (NumberOfFives * NumberOfHearts * NumberOfAces) / 
  (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2))

theorem probability_of_specific_draw :
  probabilityOfSpecificDraw = 1 / 663 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_draw_l2889_288959


namespace NUMINAMATH_CALUDE_negative_seven_times_sum_l2889_288983

theorem negative_seven_times_sum : -7 * 45 + (-7) * 55 = -700 := by
  sorry

end NUMINAMATH_CALUDE_negative_seven_times_sum_l2889_288983


namespace NUMINAMATH_CALUDE_cube_root_of_64_l2889_288933

theorem cube_root_of_64 (n : ℕ) (t : ℕ) : t = n * (n - 1) * (n + 1) + n → t = 64 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l2889_288933


namespace NUMINAMATH_CALUDE_percentage_of_14_to_70_l2889_288950

theorem percentage_of_14_to_70 : ∀ (x : ℚ), x = 14 / 70 * 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_14_to_70_l2889_288950


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2889_288934

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) : 
  a + b = 5 → a^3 + b^3 = 35 → a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2889_288934


namespace NUMINAMATH_CALUDE_hyperbola_center_l2889_288920

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (5, 0) → f2 = (9, 4) → center = (7, 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_l2889_288920


namespace NUMINAMATH_CALUDE_trigonometric_inequalities_l2889_288945

theorem trigonometric_inequalities :
  (Real.tan (3 * π / 5) < Real.tan (π / 5)) ∧
  (Real.cos (-17 * π / 4) > Real.cos (-23 * π / 5)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequalities_l2889_288945


namespace NUMINAMATH_CALUDE_no_natural_solution_l2889_288996

theorem no_natural_solution : ∀ x : ℕ, 19 * x^2 + 97 * x ≠ 1997 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l2889_288996


namespace NUMINAMATH_CALUDE_largest_product_sum_of_digits_l2889_288901

def is_single_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_sum_of_digits :
  ∃ (n d e : ℕ),
    is_single_digit d ∧
    is_prime d ∧
    is_single_digit e ∧
    is_odd e ∧
    ¬is_prime e ∧
    n = d * e * (d^2 + e) ∧
    (∀ (m : ℕ), m = d' * e' * (d'^2 + e') →
      is_single_digit d' →
      is_prime d' →
      is_single_digit e' →
      is_odd e' →
      ¬is_prime e' →
      m ≤ n) ∧
    sum_of_digits n = 9 :=
  sorry

end NUMINAMATH_CALUDE_largest_product_sum_of_digits_l2889_288901


namespace NUMINAMATH_CALUDE_smallest_integers_for_720_square_and_cube_l2889_288980

theorem smallest_integers_for_720_square_and_cube (a b : ℕ+) : 
  (∀ x : ℕ+, x < a → ¬∃ y : ℕ, 720 * x = y * y) ∧
  (∀ x : ℕ+, x < b → ¬∃ y : ℕ, 720 * x = y * y * y) ∧
  (∃ y : ℕ, 720 * a = y * y) ∧
  (∃ y : ℕ, 720 * b = y * y * y) →
  a + b = 305 := by
sorry

end NUMINAMATH_CALUDE_smallest_integers_for_720_square_and_cube_l2889_288980


namespace NUMINAMATH_CALUDE_planar_graph_inequality_l2889_288937

/-- A planar graph is a graph that can be embedded in the plane without edge crossings. -/
structure PlanarGraph where
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces

/-- For any planar graph, twice the number of edges is greater than or equal to
    three times the number of faces. -/
theorem planar_graph_inequality (G : PlanarGraph) : 2 * G.E ≥ 3 * G.F := by
  sorry

end NUMINAMATH_CALUDE_planar_graph_inequality_l2889_288937


namespace NUMINAMATH_CALUDE_quadratic_transform_h_value_l2889_288987

/-- Given a quadratic equation ax^2 + bx + c that can be expressed as 3(x - 5)^2 + 15,
    prove that when 4ax^2 + 4bx + 4c is expressed as n(x - h)^2 + k, h equals 5. -/
theorem quadratic_transform_h_value
  (a b c : ℝ)
  (h : ∀ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 15) :
  ∃ (n k : ℝ), ∀ x, 4 * a * x^2 + 4 * b * x + 4 * c = n * (x - 5)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transform_h_value_l2889_288987


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l2889_288908

theorem radio_loss_percentage (original_price sold_price : ℚ) :
  original_price = 490 →
  sold_price = 465.50 →
  (original_price - sold_price) / original_price * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l2889_288908


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2889_288951

/-- The functional equation that f must satisfy for all real a, b, c -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, (f a - f b) * (f b - f c) * (f c - f a) = f (a*b^2 + b*c^2 + c*a^2) - f (a^2*b + b^2*c + c^2*a)

/-- The set of possible functions that satisfy the equation -/
def PossibleFunctions (f : ℝ → ℝ) : Prop :=
  (∃ α β : ℝ, α ∈ ({-1, 0, 1} : Set ℝ) ∧ (∀ x, f x = α * x + β)) ∨
  (∃ α β : ℝ, α ∈ ({-1, 0, 1} : Set ℝ) ∧ (∀ x, f x = α * x^3 + β))

/-- The main theorem stating that any function satisfying the equation must be of the specified form -/
theorem functional_equation_solution (f : ℝ → ℝ) :
  FunctionalEquation f → PossibleFunctions f := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2889_288951


namespace NUMINAMATH_CALUDE_pure_imaginary_z_l2889_288918

theorem pure_imaginary_z (z : ℂ) : 
  (∃ (a : ℝ), z = Complex.I * a) → 
  Complex.abs (z - 1) = Complex.abs (-1 + Complex.I) → 
  z = Complex.I ∨ z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_z_l2889_288918


namespace NUMINAMATH_CALUDE_max_value_of_g_l2889_288913

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (x : ℝ), x ∈ Set.Icc 0 2 ∧ 
  (∀ y ∈ Set.Icc 0 2, g y ≤ g x) ∧
  g x = 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l2889_288913


namespace NUMINAMATH_CALUDE_num_persons_is_nine_l2889_288971

/-- The number of persons who went to the hotel -/
def num_persons : ℕ := 9

/-- The amount spent by each of the first 8 persons -/
def amount_per_person : ℕ := 12

/-- The additional amount spent by the 9th person above the average -/
def additional_amount : ℕ := 8

/-- The total expenditure of all persons -/
def total_expenditure : ℕ := 117

/-- Theorem stating that the number of persons who went to the hotel is 9 -/
theorem num_persons_is_nine :
  (num_persons - 1) * amount_per_person + 
  ((num_persons - 1) * amount_per_person + additional_amount) / num_persons + additional_amount = 
  total_expenditure :=
sorry

end NUMINAMATH_CALUDE_num_persons_is_nine_l2889_288971


namespace NUMINAMATH_CALUDE_quadrilateral_area_product_not_1988_l2889_288986

/-- Represents a convex quadrilateral divided by its diagonals into four triangles -/
structure QuadrilateralWithDiagonals where
  S₁ : ℕ  -- Area of triangle AOB
  S₂ : ℕ  -- Area of triangle BOC
  S₃ : ℕ  -- Area of triangle COD
  S₄ : ℕ  -- Area of triangle DOA

/-- The product of the areas of the four triangles in a quadrilateral divided by its diagonals
    cannot end in 1988 -/
theorem quadrilateral_area_product_not_1988 (q : QuadrilateralWithDiagonals) :
  ∀ (n : ℕ), q.S₁ * q.S₂ * q.S₃ * q.S₄ ≠ 1988 + 10000 * n := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_product_not_1988_l2889_288986


namespace NUMINAMATH_CALUDE_rectangle_area_l2889_288946

theorem rectangle_area (l w : ℕ) : 
  l * l + w * w = 17 * 17 →  -- diagonal is 17 cm
  2 * l + 2 * w = 46 →       -- perimeter is 46 cm
  l * w = 120 :=             -- area is 120 cm²
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2889_288946


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l2889_288928

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 120) : 
  x^2 + y^2 = 11980 / 121 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l2889_288928


namespace NUMINAMATH_CALUDE_product_of_roots_l2889_288989

theorem product_of_roots (x₁ x₂ k m : ℝ) : 
  x₁ ≠ x₂ →
  5 * x₁^2 - k * x₁ = m →
  5 * x₂^2 - k * x₂ = m →
  x₁ * x₂ = -m / 5 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l2889_288989


namespace NUMINAMATH_CALUDE_original_price_calculation_l2889_288963

-- Define the original cost price as a real number
variable (P : ℝ)

-- Define the selling price
def selling_price : ℝ := 1800

-- Define the sequence of operations on the price
def price_after_operations (original_price : ℝ) : ℝ :=
  original_price * 0.90 * 1.05 * 1.12 * 0.85

-- Define the final selling price with profit
def final_price (original_price : ℝ) : ℝ :=
  price_after_operations original_price * 1.20

-- Theorem stating the relationship between original price and selling price
theorem original_price_calculation :
  final_price P = selling_price :=
sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2889_288963


namespace NUMINAMATH_CALUDE_probability_of_selection_for_given_sizes_l2889_288962

/-- Simple random sampling without replacement -/
structure SimpleRandomSampling where
  population_size : ℕ
  sample_size : ℕ
  sample_size_le_population : sample_size ≤ population_size

/-- The probability of an individual being selected in simple random sampling -/
def probability_of_selection (srs : SimpleRandomSampling) : ℚ :=
  srs.sample_size / srs.population_size

theorem probability_of_selection_for_given_sizes :
  ∀ (srs : SimpleRandomSampling),
    srs.population_size = 6 →
    srs.sample_size = 3 →
    probability_of_selection srs = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_selection_for_given_sizes_l2889_288962


namespace NUMINAMATH_CALUDE_quadratic_real_roots_m_range_l2889_288923

theorem quadratic_real_roots_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_m_range_l2889_288923


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2889_288911

theorem smallest_number_divisible (n : ℕ) : n = 1009 ↔ 
  (∀ m : ℕ, m < n → ¬(12 ∣ (m - 2) ∧ 16 ∣ (m - 2) ∧ 18 ∣ (m - 2) ∧ 21 ∣ (m - 2) ∧ 28 ∣ (m - 2))) ∧
  (12 ∣ (n - 2) ∧ 16 ∣ (n - 2) ∧ 18 ∣ (n - 2) ∧ 21 ∣ (n - 2) ∧ 28 ∣ (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2889_288911


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l2889_288985

theorem solve_fraction_equation :
  ∀ x : ℚ, (2 / 3 - 1 / 4 : ℚ) = 1 / x → x = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l2889_288985
