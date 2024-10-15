import Mathlib

namespace NUMINAMATH_CALUDE_log_3_base_5_l2869_286939

theorem log_3_base_5 (a : ℝ) (h : Real.log 45 / Real.log 5 = a) :
  Real.log 3 / Real.log 5 = (a - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_3_base_5_l2869_286939


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2869_286947

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 20 - 5 / 200 + 7 / 2000 = 0.1285 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2869_286947


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l2869_286992

theorem largest_n_divisibility : ∃ (n : ℕ), n = 302 ∧ 
  (∀ m : ℕ, m > 302 → ¬(m + 11 ∣ m^3 + 101)) ∧
  (302 + 11 ∣ 302^3 + 101) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l2869_286992


namespace NUMINAMATH_CALUDE_sqrt_identity_l2869_286997

theorem sqrt_identity (a b : ℝ) (h : a > Real.sqrt b) :
  Real.sqrt ((a + Real.sqrt (a^2 - b)) / 2) + Real.sqrt ((a - Real.sqrt (a^2 - b)) / 2) =
  Real.sqrt (a + Real.sqrt b) ∧
  Real.sqrt ((a + Real.sqrt (a^2 - b)) / 2) - Real.sqrt ((a - Real.sqrt (a^2 - b)) / 2) =
  Real.sqrt (a - Real.sqrt b) := by
sorry

end NUMINAMATH_CALUDE_sqrt_identity_l2869_286997


namespace NUMINAMATH_CALUDE_h_negative_two_equals_eleven_l2869_286925

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 1
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_negative_two_equals_eleven : h (-2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_h_negative_two_equals_eleven_l2869_286925


namespace NUMINAMATH_CALUDE_height_difference_petronas_empire_l2869_286917

/-- The height difference between two buildings -/
def height_difference (h1 h2 : ℝ) : ℝ := |h1 - h2|

/-- The Empire State Building is 443 m tall -/
def empire_state_height : ℝ := 443

/-- The Petronas Towers is 452 m tall -/
def petronas_towers_height : ℝ := 452

/-- Theorem: The height difference between the Petronas Towers and the Empire State Building is 9 meters -/
theorem height_difference_petronas_empire : 
  height_difference petronas_towers_height empire_state_height = 9 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_petronas_empire_l2869_286917


namespace NUMINAMATH_CALUDE_graveyard_bone_ratio_l2869_286976

theorem graveyard_bone_ratio :
  let total_skeletons : ℕ := 20
  let adult_women_skeletons : ℕ := total_skeletons / 2
  let remaining_skeletons : ℕ := total_skeletons - adult_women_skeletons
  let adult_men_skeletons : ℕ := remaining_skeletons / 2
  let children_skeletons : ℕ := remaining_skeletons / 2
  let adult_woman_bones : ℕ := 20
  let adult_man_bones : ℕ := adult_woman_bones + 5
  let total_bones : ℕ := 375
  let child_bones : ℕ := (total_bones - (adult_women_skeletons * adult_woman_bones + adult_men_skeletons * adult_man_bones)) / children_skeletons
  (child_bones : ℚ) / (adult_woman_bones : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_graveyard_bone_ratio_l2869_286976


namespace NUMINAMATH_CALUDE_volleyball_tournament_probabilities_l2869_286918

theorem volleyball_tournament_probabilities :
  -- Definition of probability of student team winning a match
  let p_student_win : ℝ := 1/2
  -- Definition of probability of teacher team winning a match
  let p_teacher_win : ℝ := 3/5
  -- Total number of teams
  let total_teams : ℕ := 21
  -- Number of student teams
  let student_teams : ℕ := 20
  -- Number of teams advancing directly to quarterfinals
  let direct_advance : ℕ := 5
  -- Number of teams selected by drawing
  let drawn_teams : ℕ := 2

  -- 1. Probability of a student team winning two consecutive matches
  (p_student_win * p_student_win = 1/4) ∧

  -- 2. Probability distribution of number of rounds teacher team participates
  (1 - p_teacher_win = 2/5) ∧
  (p_teacher_win * (1 - p_teacher_win) = 6/25) ∧
  (p_teacher_win * p_teacher_win = 9/25) ∧

  -- 3. Expectation of number of rounds teacher team participates
  (1 * (1 - p_teacher_win) + 2 * (p_teacher_win * (1 - p_teacher_win)) + 3 * (p_teacher_win * p_teacher_win) = 49/25) :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_probabilities_l2869_286918


namespace NUMINAMATH_CALUDE_quiz_percentage_correct_l2869_286928

theorem quiz_percentage_correct (x : ℕ) : 
  let total_questions : ℕ := 7 * x
  let missed_questions : ℕ := 2 * x
  let correct_questions : ℕ := total_questions - missed_questions
  let percentage_correct : ℚ := (correct_questions : ℚ) / (total_questions : ℚ) * 100
  percentage_correct = 500 / 7 :=
by sorry

end NUMINAMATH_CALUDE_quiz_percentage_correct_l2869_286928


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l2869_286906

/-- The number of ways to arrange plants in a row -/
def arrangePlants (basil tomato pepper : Nat) : Nat :=
  Nat.factorial 3 * Nat.factorial basil * Nat.factorial tomato * Nat.factorial pepper

theorem plant_arrangement_count :
  arrangePlants 4 4 3 = 20736 := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangement_count_l2869_286906


namespace NUMINAMATH_CALUDE_larry_road_trip_money_l2869_286913

theorem larry_road_trip_money (initial_money : ℝ) : 
  initial_money * (1 - 0.04 - 0.30) - 52 = 368 → 
  initial_money = 636.36 := by
sorry

end NUMINAMATH_CALUDE_larry_road_trip_money_l2869_286913


namespace NUMINAMATH_CALUDE_five_dollar_neg_one_eq_zero_l2869_286970

-- Define the $ operation
def dollar_op (a b : ℤ) : ℤ := a * (b + 2) + a * b

-- Theorem statement
theorem five_dollar_neg_one_eq_zero : dollar_op 5 (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_five_dollar_neg_one_eq_zero_l2869_286970


namespace NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l2869_286959

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^3 * (x + y - 2) = y^3 * (x + y - 2)

-- Define what it means for two lines to intersect
def intersecting_lines (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x

-- Theorem statement
theorem equation_represents_two_intersecting_lines :
  ∃ (f g : ℝ → ℝ), 
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) ∧
    intersecting_lines f g :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l2869_286959


namespace NUMINAMATH_CALUDE_divisibility_by_four_l2869_286953

theorem divisibility_by_four (n : ℕ+) :
  4 ∣ (n * Nat.choose (2 * n) n) ↔ ¬∃ k : ℕ, n = 2^k := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_four_l2869_286953


namespace NUMINAMATH_CALUDE_quadruple_equality_l2869_286966

theorem quadruple_equality (a b c d : ℝ) : 
  (∀ X : ℝ, X^2 + a*X + b = (X-a)*(X-c)) ∧
  (∀ X : ℝ, X^2 + c*X + d = (X-b)*(X-d)) →
  ((a = 1 ∧ b = 2 ∧ c = -2 ∧ d = 0) ∨ 
   (a = -1 ∧ b = -2 ∧ c = 2 ∧ d = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_equality_l2869_286966


namespace NUMINAMATH_CALUDE_special_remainder_property_l2869_286985

theorem special_remainder_property (n : ℕ) : 
  (∀ q : ℕ, q > 0 → n % q^2 < q^(q^2) / 2) ↔ (n = 1 ∨ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_special_remainder_property_l2869_286985


namespace NUMINAMATH_CALUDE_subtract_negative_two_l2869_286955

theorem subtract_negative_two : 0 - (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_subtract_negative_two_l2869_286955


namespace NUMINAMATH_CALUDE_circle_ray_no_intersection_l2869_286919

/-- Given a circle (x-a)^2 + y^2 = 4 and a ray y = √3x (x ≥ 0) with no common points,
    the range of values for the real number a is {a | a < -2 or a > (4/3)√3}. -/
theorem circle_ray_no_intersection (a : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + y^2 = 4 → y ≠ Real.sqrt 3 * x ∨ x < 0) ↔ 
  (a < -2 ∨ a > (4/3) * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_circle_ray_no_intersection_l2869_286919


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_groups_l2869_286974

theorem common_number_in_overlapping_groups (list : List ℝ) : 
  list.length = 9 →
  (list.take 5).sum / 5 = 7 →
  (list.drop 4).sum / 5 = 10 →
  list.sum / 9 = 74 / 9 →
  ∃ x ∈ list, x ∈ list.take 5 ∧ x ∈ list.drop 4 ∧ x = 11 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_groups_l2869_286974


namespace NUMINAMATH_CALUDE_yuko_wins_l2869_286964

/-- The minimum value of Yuko's last die to be ahead of Yuri -/
def min_value_to_win (yuri_dice : Fin 3 → Nat) (yuko_dice : Fin 2 → Nat) : Nat :=
  (yuri_dice 0 + yuri_dice 1 + yuri_dice 2) - (yuko_dice 0 + yuko_dice 1) + 1

theorem yuko_wins (yuri_dice : Fin 3 → Nat) (yuko_dice : Fin 2 → Nat) :
  yuri_dice 0 = 2 → yuri_dice 1 = 4 → yuri_dice 2 = 5 →
  yuko_dice 0 = 1 → yuko_dice 1 = 5 →
  min_value_to_win yuri_dice yuko_dice = 6 := by
  sorry

#eval min_value_to_win (![2, 4, 5]) (![1, 5])

end NUMINAMATH_CALUDE_yuko_wins_l2869_286964


namespace NUMINAMATH_CALUDE_inequality_proof_l2869_286916

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ 1) (h2 : b ≥ 1) (h3 : c ≥ 1) (h4 : a + b + c = 9) :
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2869_286916


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2869_286949

theorem unique_solution_exists : ∃! (x : ℝ), x > 0 ∧ (Int.floor x) * x + x^2 = 93 ∧ ∀ (ε : ℝ), ε > 0 → |x - 7.10| < ε := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2869_286949


namespace NUMINAMATH_CALUDE_number_divided_by_three_l2869_286920

theorem number_divided_by_three (x : ℝ) (h : x - 39 = 54) : x / 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l2869_286920


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l2869_286927

def U : Set ℕ := {x | x < 6}
def P : Set ℕ := {2, 4}
def Q : Set ℕ := {1, 3, 4, 6}

theorem complement_intersection_problem :
  (U \ P) ∩ Q = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l2869_286927


namespace NUMINAMATH_CALUDE_jellybean_probability_l2869_286945

def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 2
def yellow_jellybeans : ℕ := 5
def picked_jellybeans : ℕ := 4

theorem jellybean_probability : 
  (Nat.choose red_jellybeans 3 * Nat.choose (total_jellybeans - red_jellybeans) 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 14 / 99 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l2869_286945


namespace NUMINAMATH_CALUDE_existence_of_equal_function_values_l2869_286987

theorem existence_of_equal_function_values (n : ℕ) (h_n : n ≤ 44) 
  (f : ℕ+ × ℕ+ → Fin n) : 
  ∃ (i j l k m p : ℕ+), 
    f (i, j) = f (i, k) ∧ f (i, j) = f (l, j) ∧ f (i, j) = f (l, k) ∧
    1989 * m ≤ i ∧ i < l ∧ l < 1989 + 1989 * m ∧
    1989 * p ≤ j ∧ j < k ∧ k < 1989 + 1989 * p :=
by sorry

end NUMINAMATH_CALUDE_existence_of_equal_function_values_l2869_286987


namespace NUMINAMATH_CALUDE_intersection_characterization_l2869_286908

-- Define set A
def A : Set ℝ := {x | Real.log (2 * x) < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 3 * x + 2}

-- Define the intersection of A and B
def A_intersect_B : Set (ℝ × ℝ) := {p | p.1 ∈ A ∧ p.2 ∈ B ∧ p.2 = 3 * p.1 + 2}

-- Theorem statement
theorem intersection_characterization : 
  A_intersect_B = {p : ℝ × ℝ | 2 < p.2 ∧ p.2 < 14} := by
  sorry

end NUMINAMATH_CALUDE_intersection_characterization_l2869_286908


namespace NUMINAMATH_CALUDE_states_joined_fraction_l2869_286926

theorem states_joined_fraction :
  let total_states : ℕ := 30
  let states_1780_to_1789 : ℕ := 12
  let states_1790_to_1799 : ℕ := 5
  let states_1780_to_1799 : ℕ := states_1780_to_1789 + states_1790_to_1799
  (states_1780_to_1799 : ℚ) / total_states = 17 / 30 := by
  sorry

end NUMINAMATH_CALUDE_states_joined_fraction_l2869_286926


namespace NUMINAMATH_CALUDE_prob_boy_girl_twins_l2869_286902

/-- The probability of twins being born -/
def prob_twins : ℚ := 3 / 250

/-- The probability of twins being identical, given that they are twins -/
def prob_identical_given_twins : ℚ := 1 / 3

/-- The probability of twins being fraternal, given that they are twins -/
def prob_fraternal_given_twins : ℚ := 1 - prob_identical_given_twins

/-- The probability of fraternal twins being a boy and a girl -/
def prob_boy_girl_given_fraternal : ℚ := 1 / 2

/-- The theorem stating the probability of a pregnant woman giving birth to boy-girl twins -/
theorem prob_boy_girl_twins : 
  prob_twins * prob_fraternal_given_twins * prob_boy_girl_given_fraternal = 1 / 250 := by
  sorry

end NUMINAMATH_CALUDE_prob_boy_girl_twins_l2869_286902


namespace NUMINAMATH_CALUDE_cookies_problem_l2869_286951

theorem cookies_problem (glenn_cookies : ℕ) (h1 : glenn_cookies = 24) 
  (h2 : ∃ kenny_cookies : ℕ, glenn_cookies = 4 * kenny_cookies) 
  (h3 : ∃ chris_cookies : ℕ, chris_cookies * 2 = kenny_cookies) : 
  glenn_cookies + kenny_cookies + chris_cookies = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_problem_l2869_286951


namespace NUMINAMATH_CALUDE_largest_number_l2869_286912

/-- Represents a number with a finite or repeating decimal expansion -/
structure DecimalNumber where
  integerPart : ℕ
  finitePart : List ℕ
  repeatingPart : List ℕ

/-- The set of numbers to compare -/
def numberSet : Set DecimalNumber := {
  ⟨8, [1, 2, 3, 5], []⟩,
  ⟨8, [1, 2, 3], [5]⟩,
  ⟨8, [1, 2, 3], [4, 5]⟩,
  ⟨8, [1, 2], [3, 4, 5]⟩,
  ⟨8, [1], [2, 3, 4, 5]⟩
}

/-- Converts a DecimalNumber to a real number -/
def toReal (d : DecimalNumber) : ℝ :=
  sorry

/-- Compares two DecimalNumbers -/
def greaterThan (a b : DecimalNumber) : Prop :=
  toReal a > toReal b

/-- Theorem stating that 8.123̅5 is the largest number in the set -/
theorem largest_number (n : DecimalNumber) :
  n ∈ numberSet →
  greaterThan ⟨8, [1, 2, 3], [5]⟩ n ∨ n = ⟨8, [1, 2, 3], [5]⟩ :=
  sorry

end NUMINAMATH_CALUDE_largest_number_l2869_286912


namespace NUMINAMATH_CALUDE_midpoint_of_fractions_l2869_286900

theorem midpoint_of_fractions :
  let a := 1 / 7
  let b := 1 / 9
  let midpoint := (a + b) / 2
  midpoint = 8 / 63 := by sorry

end NUMINAMATH_CALUDE_midpoint_of_fractions_l2869_286900


namespace NUMINAMATH_CALUDE_not_greater_than_three_equiv_l2869_286975

theorem not_greater_than_three_equiv (a : ℝ) : (¬(a > 3)) ↔ (a ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_not_greater_than_three_equiv_l2869_286975


namespace NUMINAMATH_CALUDE_profit_starts_third_year_option1_more_cost_effective_l2869_286988

/-- Represents the financial state of a fishing company -/
structure FishingCompany where
  initialCost : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrease : ℕ
  annualIncome : ℕ

/-- Calculates the year when the company starts to make a profit -/
def yearOfFirstProfit (company : FishingCompany) : ℕ :=
  sorry

/-- Calculates the more cost-effective option between two selling strategies -/
def moreCostEffectiveOption (company : FishingCompany) (option1Value : ℕ) (option2Value : ℕ) : Bool :=
  sorry

/-- Theorem stating that the company starts to make a profit in the third year -/
theorem profit_starts_third_year (company : FishingCompany) 
  (h1 : company.initialCost = 980000)
  (h2 : company.firstYearExpenses = 120000)
  (h3 : company.annualExpenseIncrease = 40000)
  (h4 : company.annualIncome = 500000) :
  yearOfFirstProfit company = 3 :=
sorry

/-- Theorem stating that the first option (selling for 260,000) is more cost-effective -/
theorem option1_more_cost_effective (company : FishingCompany)
  (h1 : company.initialCost = 980000)
  (h2 : company.firstYearExpenses = 120000)
  (h3 : company.annualExpenseIncrease = 40000)
  (h4 : company.annualIncome = 500000) :
  moreCostEffectiveOption company 260000 80000 = true :=
sorry

end NUMINAMATH_CALUDE_profit_starts_third_year_option1_more_cost_effective_l2869_286988


namespace NUMINAMATH_CALUDE_triangle_inequality_fraction_l2869_286950

theorem triangle_inequality_fraction (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) : 
  (a + b) / (1 + a + b) > c / (1 + c) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_fraction_l2869_286950


namespace NUMINAMATH_CALUDE_coin_bag_total_amount_l2869_286984

theorem coin_bag_total_amount :
  ∃ (x : ℕ),
    let one_cent := x
    let ten_cent := 2 * x
    let twenty_five_cent := 3 * (2 * x)
    let total := one_cent * 1 + ten_cent * 10 + twenty_five_cent * 25
    total = 342 := by
  sorry

end NUMINAMATH_CALUDE_coin_bag_total_amount_l2869_286984


namespace NUMINAMATH_CALUDE_total_sleep_time_in_week_l2869_286937

/-- The number of hours a cougar sleeps per night -/
def cougar_sleep_hours : ℕ := 4

/-- The additional hours a zebra sleeps compared to a cougar -/
def zebra_extra_sleep : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total sleep time for a cougar and a zebra in one week is 70 hours -/
theorem total_sleep_time_in_week : 
  (cougar_sleep_hours * days_in_week) + 
  ((cougar_sleep_hours + zebra_extra_sleep) * days_in_week) = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_sleep_time_in_week_l2869_286937


namespace NUMINAMATH_CALUDE_bc_fraction_of_ad_l2869_286956

-- Define the points
variable (A B C D : ℝ)

-- Define the conditions
axiom on_line_segment : B ≤ A ∧ B ≤ D ∧ C ≤ A ∧ C ≤ D

-- Define the length relationships
axiom length_AB : A - B = 3 * (D - B)
axiom length_AC : A - C = 7 * (D - C)

-- Theorem to prove
theorem bc_fraction_of_ad : (C - B) = (1/8) * (A - D) := by sorry

end NUMINAMATH_CALUDE_bc_fraction_of_ad_l2869_286956


namespace NUMINAMATH_CALUDE_f_properties_l2869_286993

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log (1/2)
  else if x = 0 then 0
  else Real.log (-x) / Real.log (1/2)

-- State the theorem
theorem f_properties :
  (∀ x, f x = f (-x)) ∧  -- f is even
  f 0 = 0 ∧             -- f(0) = 0
  (∀ x > 0, f x = Real.log x / Real.log (1/2)) →  -- f(x) = log₍₁/₂₎(x) for x > 0
  f (-4) = -2 ∧         -- Part 1: f(-4) = -2
  (∀ x, f x = if x > 0 then Real.log x / Real.log (1/2)
              else if x = 0 then 0
              else Real.log (-x) / Real.log (1/2))  -- Part 2: Analytic expression of f
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l2869_286993


namespace NUMINAMATH_CALUDE_range_of_m_l2869_286963

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 1/x + 4/y = 1) :
  (∀ x y, x > 0 → y > 0 → 1/x + 4/y = 1 → x + y > m^2 + 8*m) ↔ -9 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2869_286963


namespace NUMINAMATH_CALUDE_corner_divisions_l2869_286901

/-- A corner made up of 3 squares -/
structure Corner :=
  (squares : Fin 3 → Square)

/-- Represents a division of the corner into equal parts -/
structure Division :=
  (parts : ℕ)
  (is_equal : Bool)

/-- Checks if a division of the corner into n parts is possible and equal -/
def is_valid_division (c : Corner) (n : ℕ) : Prop :=
  ∃ (d : Division), d.parts = n ∧ d.is_equal = true

/-- Theorem stating that the corner can be divided into 2, 3, and 4 equal parts -/
theorem corner_divisions (c : Corner) :
  (is_valid_division c 2) ∧ 
  (is_valid_division c 3) ∧ 
  (is_valid_division c 4) :=
sorry

end NUMINAMATH_CALUDE_corner_divisions_l2869_286901


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l2869_286969

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

theorem shaded_area_theorem :
  (U \ (A ∪ B)) ∪ (A ∩ B) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l2869_286969


namespace NUMINAMATH_CALUDE_hypotenuse_length_l2869_286958

theorem hypotenuse_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a^2 + b^2) * (a^2 + b^2 + 1) = 12 →
  a^2 + b^2 = c^2 →
  c = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l2869_286958


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2869_286910

theorem max_value_sqrt_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 2) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 ∧ Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z) ≤ m ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 2 ∧
  Real.sqrt x₀ + Real.sqrt (2 * y₀) + Real.sqrt (3 * z₀) = m :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2869_286910


namespace NUMINAMATH_CALUDE_simplify_expression_l2869_286929

theorem simplify_expression : 
  ∃ x : ℚ, (3/4 * 60) - (8/5 * 60) + x = 12 ∧ x = 63 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2869_286929


namespace NUMINAMATH_CALUDE_roots_product_l2869_286921

theorem roots_product (p q : ℝ) : 
  (p - 3) * (3 * p + 8) = p^2 - 17 * p + 56 →
  (q - 3) * (3 * q + 8) = q^2 - 17 * q + 56 →
  p ≠ q →
  (p + 2) * (q + 2) = -60 := by
sorry

end NUMINAMATH_CALUDE_roots_product_l2869_286921


namespace NUMINAMATH_CALUDE_circle_arc_angle_l2869_286903

theorem circle_arc_angle (E AB BC CD AD : ℝ) : 
  E = 40 →
  AB = BC →
  BC = CD →
  AB + BC + CD + AD = 360 →
  (AB - AD) / 2 = E →
  ∃ (ACD : ℝ), ACD = 15 := by
sorry

end NUMINAMATH_CALUDE_circle_arc_angle_l2869_286903


namespace NUMINAMATH_CALUDE_arcsin_one_equals_pi_half_l2869_286973

theorem arcsin_one_equals_pi_half : Real.arcsin 1 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_equals_pi_half_l2869_286973


namespace NUMINAMATH_CALUDE_sum_marked_sides_ge_one_l2869_286940

/-- A rectangle within a unit square --/
structure Rectangle where
  width : ℝ
  height : ℝ
  markedSide : ℝ
  width_pos : 0 < width
  height_pos : 0 < height
  in_unit_square : width ≤ 1 ∧ height ≤ 1
  marked_side_valid : markedSide = width ∨ markedSide = height

/-- A partition of the unit square into rectangles --/
def UnitSquarePartition := List Rectangle

/-- The sum of the marked sides in a partition --/
def sumMarkedSides (partition : UnitSquarePartition) : ℝ :=
  partition.map (·.markedSide) |>.sum

/-- The total area of rectangles in a partition --/
def totalArea (partition : UnitSquarePartition) : ℝ :=
  partition.map (λ r => r.width * r.height) |>.sum

/-- Theorem: The sum of marked sides in any valid partition is at least 1 --/
theorem sum_marked_sides_ge_one (partition : UnitSquarePartition) 
  (h_valid : totalArea partition = 1) : 
  sumMarkedSides partition ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_sum_marked_sides_ge_one_l2869_286940


namespace NUMINAMATH_CALUDE_slope_of_line_l2869_286909

theorem slope_of_line (x y : ℝ) : 3 * y + 2 * x = 12 → (y - 4) / x = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2869_286909


namespace NUMINAMATH_CALUDE_line_through_point_l2869_286996

/-- Given a line equation bx - (b+2)y = b - 3 passing through the point (3, -5), prove that b = -13/7 -/
theorem line_through_point (b : ℚ) : 
  (b * 3 - (b + 2) * (-5) = b - 3) → b = -13/7 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2869_286996


namespace NUMINAMATH_CALUDE_unique_spicy_pair_l2869_286930

/-- A three-digit number is spicy if it equals the sum of the cubes of its digits. -/
def IsSpicy (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n = a^3 + b^3 + c^3

/-- 370 is the unique three-digit number n such that both n and n+1 are spicy. -/
theorem unique_spicy_pair : ∀ n : ℕ, (IsSpicy n ∧ IsSpicy (n + 1)) ↔ n = 370 := by
  sorry

end NUMINAMATH_CALUDE_unique_spicy_pair_l2869_286930


namespace NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l2869_286995

def A : Set ℝ := {x | x^2 ≤ 3}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_of_B_and_complement_of_A : B ∩ (Set.univ \ A) = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l2869_286995


namespace NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l2869_286931

theorem square_difference_divided (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (305^2 - 275^2) / 30 = 580 :=
by sorry

end NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l2869_286931


namespace NUMINAMATH_CALUDE_two_in_S_l2869_286904

def S : Set ℕ := {0, 1, 2}

theorem two_in_S : 2 ∈ S := by sorry

end NUMINAMATH_CALUDE_two_in_S_l2869_286904


namespace NUMINAMATH_CALUDE_product_abc_l2869_286981

theorem product_abc (a b c : ℕ+) (h : a * b^3 = 180) : a * b * c = 60 * c := by
  sorry

end NUMINAMATH_CALUDE_product_abc_l2869_286981


namespace NUMINAMATH_CALUDE_cos_13_cos_17_minus_sin_17_sin_13_l2869_286989

theorem cos_13_cos_17_minus_sin_17_sin_13 :
  Real.cos (13 * π / 180) * Real.cos (17 * π / 180) - 
  Real.sin (17 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_13_cos_17_minus_sin_17_sin_13_l2869_286989


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l2869_286907

def num_pigs : ℕ := 5
def num_rabbits : ℕ := 3
def num_dogs : ℕ := 2
def num_chickens : ℕ := 6

def total_animals : ℕ := num_pigs + num_rabbits + num_dogs + num_chickens

def num_animal_types : ℕ := 4

theorem animal_arrangement_count :
  (Nat.factorial num_animal_types) *
  (Nat.factorial num_pigs) *
  (Nat.factorial num_rabbits) *
  (Nat.factorial num_dogs) *
  (Nat.factorial num_chickens) = 12441600 :=
by sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l2869_286907


namespace NUMINAMATH_CALUDE_smallest_integer_solution_smallest_integer_solution_exists_smallest_integer_solution_is_zero_l2869_286938

theorem smallest_integer_solution (x : ℤ) : (7 - 5*x < 12) → x ≥ 0 :=
by
  sorry

theorem smallest_integer_solution_exists : ∃ x : ℤ, (7 - 5*x < 12) ∧ (∀ y : ℤ, (7 - 5*y < 12) → y ≥ x) :=
by
  sorry

theorem smallest_integer_solution_is_zero : 
  ∃ x : ℤ, x = 0 ∧ (7 - 5*x < 12) ∧ (∀ y : ℤ, (7 - 5*y < 12) → y ≥ x) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_smallest_integer_solution_exists_smallest_integer_solution_is_zero_l2869_286938


namespace NUMINAMATH_CALUDE_inequality_proof_l2869_286983

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a / (2 * b + 2 * c)) + Real.sqrt (b / (2 * a + 2 * c)) + Real.sqrt (c / (2 * a + 2 * b)) > 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2869_286983


namespace NUMINAMATH_CALUDE_opponent_total_runs_is_67_l2869_286932

/-- Represents the scores of a baseball team in a series of games. -/
structure BaseballScores :=
  (scores : List Nat)
  (lostByTwoGames : Nat)
  (wonByTripleGames : Nat)

/-- Calculates the total runs scored by the opponents. -/
def opponentTotalRuns (bs : BaseballScores) : Nat :=
  sorry

/-- The theorem states that given the specific conditions of the baseball team's games,
    the total runs scored by their opponents is 67. -/
theorem opponent_total_runs_is_67 :
  let bs : BaseballScores := {
    scores := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    lostByTwoGames := 5,
    wonByTripleGames := 5
  }
  opponentTotalRuns bs = 67 := by sorry

end NUMINAMATH_CALUDE_opponent_total_runs_is_67_l2869_286932


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2869_286967

theorem largest_prime_divisor_to_test (n : ℕ) : 
  1000 ≤ n ∧ n ≤ 1100 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  n.Prime ∨ n = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2869_286967


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2869_286960

/-- Given a quadratic equation x² - 4x - 2 = 0, prove that the correct completion of the square is (x-2)² = 6 -/
theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 4*x - 2 = 0 → (x - 2)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2869_286960


namespace NUMINAMATH_CALUDE_crow_speed_l2869_286982

/-- Crow's flight speed calculation -/
theorem crow_speed (distance_to_ditch : ℝ) (num_trips : ℕ) (time_hours : ℝ) :
  distance_to_ditch = 400 →
  num_trips = 15 →
  time_hours = 1.5 →
  (2 * distance_to_ditch * num_trips) / (1000 * time_hours) = 8 := by
  sorry

end NUMINAMATH_CALUDE_crow_speed_l2869_286982


namespace NUMINAMATH_CALUDE_smallest_angle_CBD_l2869_286943

theorem smallest_angle_CBD (ABC : ℝ) (ABD : ℝ) (CBD : ℝ) 
  (h1 : ABC = 40)
  (h2 : ABD = 15)
  (h3 : CBD = ABC - ABD) :
  CBD = 25 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_CBD_l2869_286943


namespace NUMINAMATH_CALUDE_evaluate_expression_l2869_286991

theorem evaluate_expression : (5^5 * 5^3) / 3^6 * 2^5 = 12480000 / 729 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2869_286991


namespace NUMINAMATH_CALUDE_mary_jamison_weight_difference_l2869_286979

/-- Proves that Mary weighs 20 lbs less than Jamison given the conditions in the problem -/
theorem mary_jamison_weight_difference :
  ∀ (john mary jamison : ℝ),
    mary = 160 →
    john = mary + (1/4) * mary →
    john + mary + jamison = 540 →
    jamison - mary = 20 := by
  sorry

end NUMINAMATH_CALUDE_mary_jamison_weight_difference_l2869_286979


namespace NUMINAMATH_CALUDE_percentage_of_girls_who_want_to_be_doctors_l2869_286924

theorem percentage_of_girls_who_want_to_be_doctors
  (total_students : ℝ)
  (boys_ratio : ℝ)
  (boys_doctor_ratio : ℝ)
  (boys_doctor_all_doctor_ratio : ℝ)
  (h1 : boys_ratio = 3 / 5)
  (h2 : boys_doctor_ratio = 1 / 3)
  (h3 : boys_doctor_all_doctor_ratio = 2 / 5) :
  (((1 - boys_ratio) * total_students) / total_students - 
   ((1 - boys_doctor_all_doctor_ratio) * (boys_ratio * boys_doctor_ratio * total_students)) / 
   ((1 - boys_ratio) * total_students)) * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_girls_who_want_to_be_doctors_l2869_286924


namespace NUMINAMATH_CALUDE_absent_student_percentage_l2869_286986

theorem absent_student_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h1 : total_students = 150)
  (h2 : boys = 90)
  (h3 : girls = 60)
  (h4 : boys_absent_fraction = 1 / 6)
  (h5 : girls_absent_fraction = 1 / 4)
  (h6 : total_students = boys + girls) :
  (↑boys * boys_absent_fraction + ↑girls * girls_absent_fraction) / ↑total_students = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_absent_student_percentage_l2869_286986


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l2869_286965

/-- The equation of a line passing through (-1, 2) with a slope angle of 45° is x - y + 3 = 0 -/
theorem line_equation_through_point_with_angle (x y : ℝ) :
  (x + 1 = -1 ∧ y - 2 = 0) →  -- The line passes through (-1, 2)
  (Real.tan (45 * π / 180) = 1) →  -- The slope angle is 45°
  x - y + 3 = 0  -- The equation of the line
  := by sorry


end NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l2869_286965


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_9_l2869_286999

theorem largest_integer_less_than_100_with_remainder_5_mod_9 :
  ∀ n : ℕ, n < 100 → n % 9 = 5 → n ≤ 95 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_9_l2869_286999


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2869_286944

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x + y ≥ 4) ∧
  (∃ x y : ℝ, x + y ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2869_286944


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2869_286935

theorem polynomial_evaluation : 
  let x : ℝ := 6
  (3 * x^2 + 15 * x + 7) + (4 * x^3 + 8 * x^2 - 5 * x + 10) = 1337 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2869_286935


namespace NUMINAMATH_CALUDE_avery_egg_cartons_l2869_286942

/-- Calculates the number of full egg cartons given the number of chickens,
    eggs per chicken, and eggs per carton. -/
def full_egg_cartons (num_chickens : ℕ) (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) : ℕ :=
  (num_chickens * eggs_per_chicken) / eggs_per_carton

/-- Proves that Avery can fill 10 egg cartons with the given conditions. -/
theorem avery_egg_cartons :
  full_egg_cartons 20 6 12 = 10 := by
  sorry

#eval full_egg_cartons 20 6 12

end NUMINAMATH_CALUDE_avery_egg_cartons_l2869_286942


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l2869_286961

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem twelfth_term_of_specific_arithmetic_sequence :
  let a₁ : ℚ := 1
  let a₂ : ℚ := 3/2
  let d : ℚ := a₂ - a₁
  arithmeticSequenceTerm a₁ d 12 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l2869_286961


namespace NUMINAMATH_CALUDE_sqrt_square_eq_identity_power_zero_eq_one_l2869_286948

-- Option C
theorem sqrt_square_eq_identity (x : ℝ) (h : x ≥ -2) :
  (Real.sqrt (x + 2))^2 = x + 2 := by sorry

-- Option D
theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) :
  x^0 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_identity_power_zero_eq_one_l2869_286948


namespace NUMINAMATH_CALUDE_number_addition_problem_l2869_286977

theorem number_addition_problem (N : ℝ) (X : ℝ) : 
  N = 180 → 
  N + X = (1/15) * N → 
  X = -168 := by sorry

end NUMINAMATH_CALUDE_number_addition_problem_l2869_286977


namespace NUMINAMATH_CALUDE_negation_of_exists_leq_negation_of_proposition_l2869_286962

theorem negation_of_exists_leq (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ ∀ x, ¬(p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_leq_negation_of_proposition_l2869_286962


namespace NUMINAMATH_CALUDE_wind_power_scientific_notation_l2869_286941

/-- Proves that 56 million kilowatts is equal to 5.6 × 10^7 kilowatts in scientific notation -/
theorem wind_power_scientific_notation : 
  (56000000 : ℝ) = 5.6 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_wind_power_scientific_notation_l2869_286941


namespace NUMINAMATH_CALUDE_ratio_of_two_numbers_l2869_286968

theorem ratio_of_two_numbers (x y : ℝ) (h1 : x + y = 33) (h2 : x = 22) :
  y / x = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_of_two_numbers_l2869_286968


namespace NUMINAMATH_CALUDE_line_through_point_unique_l2869_286933

/-- A line passing through a point -/
def line_passes_through_point (k : ℝ) : Prop :=
  2 * k * (-1/2) - 3 = -7 * 3

/-- The value of k that satisfies the line equation -/
def k_value : ℝ := 18

/-- Theorem: k_value is the unique real number that satisfies the line equation -/
theorem line_through_point_unique : 
  line_passes_through_point k_value ∧ 
  ∀ k : ℝ, line_passes_through_point k → k = k_value :=
sorry

end NUMINAMATH_CALUDE_line_through_point_unique_l2869_286933


namespace NUMINAMATH_CALUDE_lg_expression_equals_two_l2869_286914

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_two :
  lg 4 + lg 5 * lg 20 + (lg 5)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_expression_equals_two_l2869_286914


namespace NUMINAMATH_CALUDE_multiplication_of_decimals_l2869_286994

theorem multiplication_of_decimals : 3.6 * 0.3 = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_decimals_l2869_286994


namespace NUMINAMATH_CALUDE_mike_picked_52_peaches_l2869_286980

/-- The number of peaches Mike picked at the orchard -/
def peaches_picked (initial : ℕ) (total : ℕ) : ℕ :=
  total - initial

/-- Theorem stating that Mike picked 52 peaches at the orchard -/
theorem mike_picked_52_peaches (initial : ℕ) (total : ℕ) 
  (h1 : initial = 34) 
  (h2 : total = 86) : 
  peaches_picked initial total = 52 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_52_peaches_l2869_286980


namespace NUMINAMATH_CALUDE_minutkin_bedtime_l2869_286934

/-- Represents the number of minutes in a day -/
def minutes_per_day : ℕ := 24 * 60

/-- Represents the time Minutkin winds his watch in the morning (8:30 AM) -/
def morning_wind_time : ℕ := 8 * 60 + 30

/-- Represents the number of full turns Minutkin makes in the morning -/
def morning_turns : ℕ := 9

/-- Represents the number of full turns Minutkin makes at night -/
def night_turns : ℕ := 11

/-- Represents the total number of turns in a day -/
def total_turns : ℕ := morning_turns + night_turns

/-- Theorem stating that Minutkin goes to bed at 9:42 PM -/
theorem minutkin_bedtime :
  ∃ (bedtime : ℕ),
    bedtime = (minutes_per_day + morning_wind_time - (morning_turns * minutes_per_day / total_turns)) % minutes_per_day ∧
    bedtime = 21 * 60 + 42 :=
by sorry

end NUMINAMATH_CALUDE_minutkin_bedtime_l2869_286934


namespace NUMINAMATH_CALUDE_min_X_value_l2869_286936

def F (X : ℤ) : List ℤ := [-4, -1, 0, 6, X]

def F_new (X : ℤ) : List ℤ := [2, 3, 0, 6, X]

def mean (l : List ℤ) : ℚ := (l.sum : ℚ) / l.length

theorem min_X_value : 
  ∀ X : ℤ, (mean (F_new X) ≥ 2 * mean (F X)) → X ≥ 9 ∧
  ∀ Y : ℤ, Y < 9 → mean (F_new Y) < 2 * mean (F Y) :=
sorry

end NUMINAMATH_CALUDE_min_X_value_l2869_286936


namespace NUMINAMATH_CALUDE_apollonius_circle_minimum_l2869_286946

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (2, 1)

-- Define the distance ratio condition
def distance_ratio (P : ℝ × ℝ) : Prop :=
  (P.1 + 1)^2 + P.2^2 = 2 * ((P.1 - 2)^2 + (P.2 - 1)^2)

-- Define the symmetry line
def symmetry_line (m n : ℝ) (P : ℝ × ℝ) : Prop :=
  m * P.1 + n * P.2 = 2

-- Main theorem
theorem apollonius_circle_minimum (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ P : ℝ × ℝ, distance_ratio P → ∃ P', distance_ratio P' ∧ symmetry_line m n ((P.1 + P'.1)/2, (P.2 + P'.2)/2)) →
  (2/m + 5/n) ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_apollonius_circle_minimum_l2869_286946


namespace NUMINAMATH_CALUDE_tim_payment_l2869_286990

/-- The total amount Tim paid for his and his cat's medical visits -/
def total_payment (doctor_visit_cost : ℝ) (doctor_insurance_coverage : ℝ) 
  (cat_visit_cost : ℝ) (cat_insurance_coverage : ℝ) : ℝ :=
  (doctor_visit_cost - doctor_visit_cost * doctor_insurance_coverage) +
  (cat_visit_cost - cat_insurance_coverage)

/-- Theorem stating that Tim paid $135 in total -/
theorem tim_payment : 
  total_payment 300 0.75 120 60 = 135 := by
  sorry

end NUMINAMATH_CALUDE_tim_payment_l2869_286990


namespace NUMINAMATH_CALUDE_max_k_for_quadratic_root_difference_l2869_286972

theorem max_k_for_quadratic_root_difference (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   x^2 + k*x + 10 = 0 ∧ 
   y^2 + k*y + 10 = 0 ∧ 
   |x - y| = Real.sqrt 81) →
  k ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_quadratic_root_difference_l2869_286972


namespace NUMINAMATH_CALUDE_bathroom_visits_time_calculation_l2869_286952

/-- Given that it takes 20 minutes for 8 bathroom visits, 
    prove that 6 visits will take 15 minutes. -/
theorem bathroom_visits_time_calculation 
  (total_time : ℝ) 
  (total_visits : ℕ) 
  (target_visits : ℕ) 
  (h1 : total_time = 20) 
  (h2 : total_visits = 8) 
  (h3 : target_visits = 6) : 
  (total_time / total_visits) * target_visits = 15 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_visits_time_calculation_l2869_286952


namespace NUMINAMATH_CALUDE_mother_extra_rides_l2869_286957

/-- The number of times Billy rode his bike -/
def billy_rides : ℕ := 17

/-- The number of times John rode his bike -/
def john_rides : ℕ := 2 * billy_rides

/-- The number of times the mother rode her bike -/
def mother_rides (x : ℕ) : ℕ := john_rides + x

/-- The total number of times they all rode their bikes -/
def total_rides : ℕ := 95

/-- Theorem stating that the mother rode her bike 10 times more than John -/
theorem mother_extra_rides : 
  ∃ x : ℕ, x = 10 ∧ mother_rides x = john_rides + x ∧ 
  billy_rides + john_rides + mother_rides x = total_rides :=
sorry

end NUMINAMATH_CALUDE_mother_extra_rides_l2869_286957


namespace NUMINAMATH_CALUDE_number_divided_by_004_equals_25_l2869_286922

theorem number_divided_by_004_equals_25 : ∃ x : ℝ, x / 0.04 = 25 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_004_equals_25_l2869_286922


namespace NUMINAMATH_CALUDE_min_sum_quadratic_coeff_l2869_286923

theorem min_sum_quadratic_coeff (a b c : ℕ+) 
  (root_condition : ∃ x₁ x₂ : ℝ, (a:ℝ) * x₁^2 + (b:ℝ) * x₁ + (c:ℝ) = 0 ∧ 
                                (a:ℝ) * x₂^2 + (b:ℝ) * x₂ + (c:ℝ) = 0 ∧
                                x₁ ≠ x₂ ∧ 
                                abs x₁ < (1:ℝ)/3 ∧ 
                                abs x₂ < (1:ℝ)/3) : 
  (a:ℕ) + b + c ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_quadratic_coeff_l2869_286923


namespace NUMINAMATH_CALUDE_inverse_g_at_167_l2869_286905

def g (x : ℝ) : ℝ := 5 * x^5 + 7

theorem inverse_g_at_167 : g⁻¹ 167 = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_g_at_167_l2869_286905


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l2869_286998

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem f_decreasing_on_interval :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l2869_286998


namespace NUMINAMATH_CALUDE_f_positive_implies_a_range_a_range_implies_f_positive_f_positive_iff_a_range_l2869_286915

/-- The function f(x) = ax^2 - 2x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 2

/-- Theorem: If f(x) > 0 for all x in (1, 4), then a > 1/2 -/
theorem f_positive_implies_a_range (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → f a x > 0) → a > 1/2 := by
  sorry

/-- Theorem: If a > 1/2, then f(x) > 0 for all x in (1, 4) -/
theorem a_range_implies_f_positive (a : ℝ) :
  a > 1/2 → (∀ x, 1 < x ∧ x < 4 → f a x > 0) := by
  sorry

/-- The main theorem combining both directions -/
theorem f_positive_iff_a_range (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → f a x > 0) ↔ a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_implies_a_range_a_range_implies_f_positive_f_positive_iff_a_range_l2869_286915


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2869_286971

/-- The circumference of the base of a right circular cone with volume 18π cubic centimeters and height 6 cm is 6π cm. -/
theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 18 * Real.pi ∧ h = 6 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2869_286971


namespace NUMINAMATH_CALUDE_johns_total_payment_l2869_286978

/-- Calculates the total amount John paid for his dog's vet appointments and insurance -/
def total_payment (num_appointments : ℕ) (appointment_cost : ℚ) (insurance_cost : ℚ) (insurance_coverage : ℚ) : ℚ :=
  let first_appointment_cost := appointment_cost
  let insurance_payment := insurance_cost
  let subsequent_appointments_cost := appointment_cost * (num_appointments - 1 : ℚ)
  let covered_amount := subsequent_appointments_cost * insurance_coverage
  let out_of_pocket := subsequent_appointments_cost - covered_amount
  first_appointment_cost + insurance_payment + out_of_pocket

/-- Theorem stating that John's total payment for his dog's vet appointments and insurance is $660 -/
theorem johns_total_payment :
  total_payment 3 400 100 0.8 = 660 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_payment_l2869_286978


namespace NUMINAMATH_CALUDE_smallest_three_digit_square_append_l2869_286954

theorem smallest_three_digit_square_append : ∃ (n : ℕ), 
  (n = 183) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n → ¬(∃ k : ℕ, 1000 * m + (m + 1) = k * k)) ∧
  (∃ k : ℕ, 1000 * n + (n + 1) = k * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_square_append_l2869_286954


namespace NUMINAMATH_CALUDE_multiplier_problem_l2869_286911

theorem multiplier_problem (n : ℝ) (m : ℝ) (h1 : n = 3) (h2 : m * n = 3 * n + 12) : m = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l2869_286911
