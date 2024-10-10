import Mathlib

namespace total_village_tax_l168_16825

/-- Represents the farm tax collected from a village -/
structure FarmTax where
  total_tax : ℝ
  willam_tax : ℝ
  willam_land_percentage : ℝ

/-- Theorem stating the total tax collected from the village -/
theorem total_village_tax (ft : FarmTax) 
  (h1 : ft.willam_tax = 480)
  (h2 : ft.willam_land_percentage = 25) :
  ft.total_tax = 1920 := by
  sorry

end total_village_tax_l168_16825


namespace modular_arithmetic_problem_l168_16874

theorem modular_arithmetic_problem (n : ℕ) : 
  n < 19 ∧ (5 * n) % 19 = 1 → ((3^n)^2 - 3) % 19 = 3 := by
  sorry

end modular_arithmetic_problem_l168_16874


namespace largest_angle_in_special_triangle_l168_16823

theorem largest_angle_in_special_triangle (A B C : Real) (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = π) (h3 : Real.sin A / Real.sin B = 3 / 5)
  (h4 : Real.sin B / Real.sin C = 5 / 7) :
  max A (max B C) = 2 * π / 3 := by
  sorry

end largest_angle_in_special_triangle_l168_16823


namespace squirrel_solution_l168_16854

/-- The number of walnuts the girl squirrel ate -/
def squirrel_problem (initial : ℕ) (boy_gathered : ℕ) (boy_dropped : ℕ) (girl_brought : ℕ) (final : ℕ) : ℕ :=
  initial + boy_gathered - boy_dropped + girl_brought - final

/-- Theorem stating the solution to the squirrel problem -/
theorem squirrel_solution : squirrel_problem 12 6 1 5 20 = 2 := by
  sorry

end squirrel_solution_l168_16854


namespace log_equation_solution_l168_16886

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  (Real.log x / Real.log 16) + (Real.log x / Real.log 4) + (Real.log x / Real.log 2) = 7 →
  x = 16 := by
  sorry

end log_equation_solution_l168_16886


namespace football_scoring_problem_l168_16865

/-- Represents the football scoring problem with Gina and Tom -/
theorem football_scoring_problem 
  (gina_day1 : ℕ) 
  (tom_day1 : ℕ) 
  (tom_day2 : ℕ) 
  (gina_day2 : ℕ) 
  (h1 : gina_day1 = 2)
  (h2 : tom_day1 = gina_day1 + 3)
  (h3 : tom_day2 = 6)
  (h4 : gina_day2 < tom_day2)
  (h5 : gina_day1 + tom_day1 + gina_day2 + tom_day2 = 17) :
  tom_day2 - gina_day2 = 2 := by
sorry

end football_scoring_problem_l168_16865


namespace petyas_friends_l168_16852

/-- The number of stickers Petya gives to each friend in the first scenario -/
def stickers_per_friend_scenario1 : ℕ := 5

/-- The number of stickers Petya has left in the first scenario -/
def stickers_left_scenario1 : ℕ := 8

/-- The number of stickers Petya gives to each friend in the second scenario -/
def stickers_per_friend_scenario2 : ℕ := 6

/-- The number of additional stickers Petya needs in the second scenario -/
def additional_stickers_needed : ℕ := 11

/-- Petya's number of friends -/
def number_of_friends : ℕ := 19

theorem petyas_friends :
  (stickers_per_friend_scenario1 * number_of_friends + stickers_left_scenario1 =
   stickers_per_friend_scenario2 * number_of_friends - additional_stickers_needed) ∧
  (number_of_friends = 19) :=
by sorry

end petyas_friends_l168_16852


namespace half_month_days_l168_16831

/-- Proves that given a 30-day month with specified mean profits, 
    the number of days in each half of the month is 15. -/
theorem half_month_days (total_days : ℕ) (mean_profit : ℚ) 
    (first_half_mean : ℚ) (second_half_mean : ℚ) : 
    total_days = 30 ∧ 
    mean_profit = 350 ∧ 
    first_half_mean = 225 ∧ 
    second_half_mean = 475 → 
    ∃ (first_half_days second_half_days : ℕ), 
      first_half_days = 15 ∧ 
      second_half_days = 15 ∧ 
      first_half_days + second_half_days = total_days ∧
      (first_half_mean * first_half_days + second_half_mean * second_half_days) / total_days = mean_profit :=
by sorry

end half_month_days_l168_16831


namespace common_root_values_l168_16851

theorem common_root_values (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^3 + b * k^2 + c * k + d = 0)
  (h2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end common_root_values_l168_16851


namespace stating_time_is_seven_thirty_two_l168_16869

/-- Represents the number of minutes in an hour -/
def minutesInHour : ℕ := 60

/-- Represents the time in minutes after 7:00 a.m. -/
def minutesAfterSeven (x : ℚ) : ℚ := 8 * x

/-- Represents the time in minutes before 8:00 a.m. -/
def minutesBeforeEight (x : ℚ) : ℚ := 7 * x

/-- 
Theorem stating that if a time is 8x minutes after 7:00 a.m. and 7x minutes before 8:00 a.m.,
then the time is 32 minutes after 7:00 a.m. (which is 7:32 a.m.)
-/
theorem time_is_seven_thirty_two (x : ℚ) :
  minutesAfterSeven x + minutesBeforeEight x = minutesInHour →
  minutesAfterSeven x = 32 :=
by sorry

end stating_time_is_seven_thirty_two_l168_16869


namespace letter_sum_equals_fifteen_l168_16817

/-- Given a mapping of letters to numbers where A = 0, B = 1, C = 2, ..., Z = 25,
    prove that the sum of A + B + M + C equals 15. -/
theorem letter_sum_equals_fifteen :
  let letter_to_num : Char → ℕ := fun c => c.toNat - 65
  letter_to_num 'A' + letter_to_num 'B' + letter_to_num 'M' + letter_to_num 'C' = 15 := by
  sorry

end letter_sum_equals_fifteen_l168_16817


namespace min_points_to_guarantee_win_no_smaller_guarantee_l168_16807

/-- Represents the points earned in a single race -/
inductive RaceResult
| First  : RaceResult
| Second : RaceResult
| Third  : RaceResult

/-- Converts a race result to points -/
def points (result : RaceResult) : Nat :=
  match result with
  | RaceResult.First  => 6
  | RaceResult.Second => 4
  | RaceResult.Third  => 2

/-- Calculates the total points from a list of race results -/
def totalPoints (results : List RaceResult) : Nat :=
  results.map points |>.sum

/-- Represents the results of three races -/
def ThreeRaces := (RaceResult × RaceResult × RaceResult)

/-- Theorem: 16 points is the minimum to guarantee winning -/
theorem min_points_to_guarantee_win :
  ∀ (other : ThreeRaces),
  ∃ (winner : ThreeRaces),
    totalPoints (winner.1 :: winner.2.1 :: [winner.2.2]) = 16 ∧
    totalPoints (winner.1 :: winner.2.1 :: [winner.2.2]) >
    totalPoints (other.1 :: other.2.1 :: [other.2.2]) :=
  sorry

/-- Theorem: No smaller number of points can guarantee winning -/
theorem no_smaller_guarantee :
  ∀ (n : Nat),
  n < 16 →
  ∃ (player1 player2 : ThreeRaces),
    totalPoints (player1.1 :: player1.2.1 :: [player1.2.2]) = n ∧
    totalPoints (player2.1 :: player2.2.1 :: [player2.2.2]) ≥ n :=
  sorry

end min_points_to_guarantee_win_no_smaller_guarantee_l168_16807


namespace initial_group_size_l168_16861

theorem initial_group_size (X : ℕ) : 
  X - 6 + 5 - 2 + 3 = 13 → X = 11 := by
  sorry

end initial_group_size_l168_16861


namespace wage_increase_l168_16806

theorem wage_increase (original_wage : ℝ) (increase_percentage : ℝ) (new_wage : ℝ) :
  original_wage = 20 →
  increase_percentage = 40 →
  new_wage = original_wage * (1 + increase_percentage / 100) →
  new_wage = 28 := by
sorry

end wage_increase_l168_16806


namespace pf_length_l168_16867

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) where
  right_angled : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0
  pq_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 3
  pr_length : Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) = 3 * Real.sqrt 3

-- Define the altitude PL and median RM
def altitude (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry
def median (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the intersection point F
def intersectionPoint (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem pf_length (P Q R : ℝ × ℝ) (t : Triangle P Q R) :
  let F := intersectionPoint P Q R
  Real.sqrt ((F.1 - P.1)^2 + (F.2 - P.2)^2) = 0.857 * Real.sqrt 3 := by sorry

end pf_length_l168_16867


namespace quadratic_root_implies_k_l168_16899

theorem quadratic_root_implies_k (k : ℝ) : 
  (2 * (5 : ℝ)^2 + 3 * 5 - k = 0) → k = 65 := by
  sorry

end quadratic_root_implies_k_l168_16899


namespace sufficient_not_necessary_condition_l168_16844

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → x * y > 0) ∧
  (∃ x y : ℝ, x * y > 0 ∧ ¬(x > 0 ∧ y > 0)) := by
  sorry

end sufficient_not_necessary_condition_l168_16844


namespace all_zeros_not_pronounced_l168_16836

/-- Represents a natural number in decimal notation --/
def DecimalNumber : Type := List Nat

/-- Rules for reading integers --/
structure ReadingRules where
  readHighestToLowest : Bool
  skipEndZeros : Bool
  readConsecutiveZerosAsOne : Bool

/-- Function to determine if a digit should be pronounced --/
def shouldPronounce (rules : ReadingRules) (num : DecimalNumber) (index : Nat) : Bool :=
  sorry

/-- The number 3,406,000 in decimal notation --/
def number : DecimalNumber := [3, 4, 0, 6, 0, 0, 0]

/-- The rules for reading integers as described in the problem --/
def integerReadingRules : ReadingRules := {
  readHighestToLowest := true,
  skipEndZeros := true,
  readConsecutiveZerosAsOne := true
}

/-- Theorem stating that all zeros in 3,406,000 are not pronounced --/
theorem all_zeros_not_pronounced : 
  ∀ i, i ∈ [2, 4, 5, 6] → ¬(shouldPronounce integerReadingRules number i) :=
sorry

end all_zeros_not_pronounced_l168_16836


namespace find_a_l168_16812

theorem find_a : ∃ a : ℤ, 
  (∃ x : ℤ, (2 * x - a = 3) ∧ 
    (∀ y : ℤ, (1 - (y - 2) / 2 : ℚ) < ((1 + y) / 3 : ℚ) → y ≥ x) ∧
    (1 - (x - 2) / 2 : ℚ) < ((1 + x) / 3 : ℚ)) →
  a = 3 := by
sorry

end find_a_l168_16812


namespace max_b_value_l168_16821

theorem max_b_value (a b c : ℕ) (h1 : a * b * c = 360) (h2 : 1 < c) (h3 : c < b) (h4 : b < a) :
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ b' = 12 :=
by sorry

end max_b_value_l168_16821


namespace f_increasing_l168_16802

def f (x : ℝ) : ℝ := x^2 + 4*x + 3

theorem f_increasing : ∀ x y, 0 < x → x < y → f x < f y := by sorry

end f_increasing_l168_16802


namespace isosceles_right_triangle_line_l168_16853

/-- A line that passes through the point (3, 2) and forms an isosceles right triangle with the coordinate axes has the equation x - y - 1 = 0 or x + y - 5 = 0 -/
theorem isosceles_right_triangle_line : 
  ∀ (l : Set (ℝ × ℝ)), 
  ((3, 2) ∈ l) → 
  (∃ (a b : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ a * x + b * y = 1) →
  (∃ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l ∧ p.1 = 0 ∧ q.2 = 0 ∧ p.2 = q.1 ∧ 
    (p.2 - 3)^2 + (q.1 - 3)^2 = (3 - 0)^2 + (2 - 0)^2) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (x - y = 1 ∨ x + y = 5)) := by
  sorry

end isosceles_right_triangle_line_l168_16853


namespace red_ball_probability_l168_16828

theorem red_ball_probability (w r : ℕ) : 
  r > w ∧ r < 2 * w ∧ 2 * w + 3 * r = 60 → 
  (r : ℚ) / (w + r : ℚ) = 14 / 23 := by
sorry

end red_ball_probability_l168_16828


namespace ron_siblings_product_l168_16829

/-- Represents a family structure --/
structure Family :=
  (sisters : ℕ)
  (brothers : ℕ)

/-- The problem setup --/
def problem_setup (harry_family : Family) (harriet : Family) (ron : Family) : Prop :=
  harry_family.sisters = 4 ∧
  harry_family.brothers = 6 ∧
  harriet.sisters = harry_family.sisters - 1 ∧
  harriet.brothers = harry_family.brothers ∧
  ron.sisters = harriet.sisters ∧
  ron.brothers = harriet.brothers + 2

/-- The main theorem --/
theorem ron_siblings_product (harry_family : Family) (harriet : Family) (ron : Family) 
  (h : problem_setup harry_family harriet ron) : 
  ron.sisters * ron.brothers = 32 := by
  sorry


end ron_siblings_product_l168_16829


namespace dryer_cost_l168_16863

theorem dryer_cost (washer dryer : ℕ) : 
  washer + dryer = 600 →
  washer = 3 * dryer →
  dryer = 150 := by
sorry

end dryer_cost_l168_16863


namespace largest_coefficient_binomial_expansion_l168_16892

theorem largest_coefficient_binomial_expansion :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 6 →
  (Nat.choose 6 k) * (2^k) ≤ (Nat.choose 6 4) * (2^4) :=
by sorry

end largest_coefficient_binomial_expansion_l168_16892


namespace total_non_defective_engines_l168_16835

/-- Represents a batch of engines with their total count and defect rate -/
structure Batch where
  total : ℕ
  defect_rate : ℚ
  defect_rate_valid : 0 ≤ defect_rate ∧ defect_rate ≤ 1

/-- Calculates the number of non-defective engines in a batch -/
def non_defective (b : Batch) : ℚ :=
  b.total * (1 - b.defect_rate)

/-- The list of batches with their respective data -/
def batches : List Batch := [
  ⟨140, 12/100, by norm_num⟩,
  ⟨150, 18/100, by norm_num⟩,
  ⟨170, 22/100, by norm_num⟩,
  ⟨180, 28/100, by norm_num⟩,
  ⟨190, 32/100, by norm_num⟩,
  ⟨210, 36/100, by norm_num⟩,
  ⟨220, 41/100, by norm_num⟩
]

/-- The theorem stating the total number of non-defective engines -/
theorem total_non_defective_engines :
  Int.floor (batches.map non_defective).sum = 902 := by
  sorry

end total_non_defective_engines_l168_16835


namespace ball_count_proof_l168_16808

theorem ball_count_proof (white green yellow red purple : ℕ)
  (h1 : white = 50)
  (h2 : green = 30)
  (h3 : yellow = 8)
  (h4 : red = 9)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 88/100) :
  white + green + yellow + red + purple = 100 := by
sorry

end ball_count_proof_l168_16808


namespace largest_divisor_five_consecutive_integers_l168_16889

theorem largest_divisor_five_consecutive_integers : 
  ∃ (k : ℕ), k = 60 ∧ 
  (∀ (n : ℤ), ∃ (m : ℤ), (n * (n+1) * (n+2) * (n+3) * (n+4)) = m * k) ∧
  (∀ (l : ℕ), l > k → ∃ (n : ℤ), ∀ (m : ℤ), (n * (n+1) * (n+2) * (n+3) * (n+4)) ≠ m * l) :=
by sorry

end largest_divisor_five_consecutive_integers_l168_16889


namespace number_problem_l168_16830

theorem number_problem : 
  ∃ x : ℝ, x - (3/5) * x = 50 → x = 125 := by
  sorry

end number_problem_l168_16830


namespace A_less_than_B_l168_16875

theorem A_less_than_B (x y : ℝ) : 
  let A := -y^2 + 4*x - 3
  let B := x^2 + 2*x + 2*y
  A < B := by sorry

end A_less_than_B_l168_16875


namespace triangle_properties_l168_16837

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.c^2 + t.a * t.b = t.c * (t.a * Real.cos t.B - t.b * Real.cos t.A) + 2 * t.b^2

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.c = 2 * Real.sqrt 3) : 
  t.C = π / 3 ∧ 
  ∃ (x : ℝ), -2 * Real.sqrt 3 < x ∧ x < 2 * Real.sqrt 3 ∧ x = 4 * Real.sin t.B - t.a :=
sorry

end triangle_properties_l168_16837


namespace smallest_m_for_partition_l168_16811

theorem smallest_m_for_partition (r : ℕ+) :
  (∃ (m : ℕ+), ∀ (A : Fin r → Set ℕ),
    (∀ (i j : Fin r), i ≠ j → A i ∩ A j = ∅) →
    (⋃ (i : Fin r), A i) = Finset.range m →
    (∃ (k : Fin r) (a b : ℕ), a ∈ A k ∧ b ∈ A k ∧ a ≠ 0 ∧ 1 ≤ b / a ∧ b / a ≤ 1 + 1 / 2022)) ∧
  (∀ (m : ℕ+), m < 2023 * r →
    ¬(∀ (A : Fin r → Set ℕ),
      (∀ (i j : Fin r), i ≠ j → A i ∩ A j = ∅) →
      (⋃ (i : Fin r), A i) = Finset.range m →
      (∃ (k : Fin r) (a b : ℕ), a ∈ A k ∧ b ∈ A k ∧ a ≠ 0 ∧ 1 ≤ b / a ∧ b / a ≤ 1 + 1 / 2022))) :=
by sorry

end smallest_m_for_partition_l168_16811


namespace hyperbolas_same_asymptotes_l168_16846

-- Define the two hyperbolas
def hyperbola1 (x y : ℝ) : Prop := x^2/16 - y^2/9 = 1
def hyperbola2 (x y : ℝ) : Prop := y^2/9 - x^2/16 = 1

-- Define the asymptotes for a hyperbola
def asymptotes (a b : ℝ) (x y : ℝ) : Prop := y = (b/a)*x ∨ y = -(b/a)*x

-- Theorem stating that both hyperbolas have the same asymptotes
theorem hyperbolas_same_asymptotes :
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    (hyperbola1 x y ∨ hyperbola2 x y) → asymptotes a b x y :=
sorry

end hyperbolas_same_asymptotes_l168_16846


namespace smallest_number_with_2020_divisors_l168_16813

def number_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_number_with_2020_divisors :
  ∀ m : ℕ, m < 2^100 * 3^4 * 5 * 7 →
    number_of_divisors m ≠ 2020 ∧
    number_of_divisors (2^100 * 3^4 * 5 * 7) = 2020 :=
by sorry

end smallest_number_with_2020_divisors_l168_16813


namespace isabel_weekly_distance_l168_16841

/-- Calculates the total distance run in a week given a circuit length, 
    number of morning and afternoon runs, and number of days in a week. -/
def total_weekly_distance (circuit_length : ℕ) (morning_runs : ℕ) (afternoon_runs : ℕ) (days_in_week : ℕ) : ℕ :=
  (circuit_length * (morning_runs + afternoon_runs) * days_in_week)

/-- Theorem stating that running a 365-meter circuit 7 times in the morning and 3 times in the afternoon
    for 7 days results in a total distance of 25550 meters. -/
theorem isabel_weekly_distance :
  total_weekly_distance 365 7 3 7 = 25550 := by
  sorry

end isabel_weekly_distance_l168_16841


namespace average_initial_price_is_54_l168_16840

/-- Represents the price and quantity of fruit. -/
structure FruitInfo where
  applePrice : ℕ
  orangePrice : ℕ
  totalFruit : ℕ
  orangesPutBack : ℕ
  avgPriceKept : ℕ

/-- Calculates the average price of initially selected fruit. -/
def averageInitialPrice (info : FruitInfo) : ℚ :=
  let apples := info.totalFruit - (info.totalFruit - info.orangesPutBack - 
    (info.avgPriceKept * (info.totalFruit - info.orangesPutBack) - 
    info.orangePrice * (info.totalFruit - info.orangesPutBack - info.orangesPutBack)) / 
    (info.applePrice - info.orangePrice))
  let oranges := info.totalFruit - apples
  (info.applePrice * apples + info.orangePrice * oranges) / info.totalFruit

/-- Theorem stating that the average initial price is 54 cents. -/
theorem average_initial_price_is_54 (info : FruitInfo) 
    (h1 : info.applePrice = 40)
    (h2 : info.orangePrice = 60)
    (h3 : info.totalFruit = 10)
    (h4 : info.orangesPutBack = 6)
    (h5 : info.avgPriceKept = 45) :
  averageInitialPrice info = 54 := by
  sorry

end average_initial_price_is_54_l168_16840


namespace graduates_parents_l168_16882

theorem graduates_parents (graduates : ℕ) (teachers : ℕ) (total_chairs : ℕ)
  (h_graduates : graduates = 50)
  (h_teachers : teachers = 20)
  (h_total_chairs : total_chairs = 180) :
  (total_chairs - (graduates + teachers + teachers / 2)) / graduates = 2 := by
  sorry

end graduates_parents_l168_16882


namespace geometric_series_sum_of_cubes_l168_16819

theorem geometric_series_sum_of_cubes 
  (a : ℝ) (r : ℝ) (hr : -1 < r ∧ r < 1) 
  (h1 : a / (1 - r) = 2) 
  (h2 : a^2 / (1 - r^2) = 6) : 
  a^3 / (1 - r^3) = 96/7 := by
sorry

end geometric_series_sum_of_cubes_l168_16819


namespace members_neither_subject_count_l168_16834

/-- The number of club members taking neither computer science nor robotics -/
def membersNeitherSubject (totalMembers csMembers roboticsMembers bothSubjects : ℕ) : ℕ :=
  totalMembers - (csMembers + roboticsMembers - bothSubjects)

/-- Theorem stating the number of club members taking neither subject -/
theorem members_neither_subject_count :
  membersNeitherSubject 150 80 70 20 = 20 := by
  sorry

end members_neither_subject_count_l168_16834


namespace bottom_row_bricks_l168_16870

/-- Represents a brick wall with decreasing number of bricks in each row -/
structure BrickWall where
  totalRows : Nat
  totalBricks : Nat
  bottomRowBricks : Nat
  decreaseRate : Nat
  rowsDecreasing : bottomRowBricks ≥ (totalRows - 1) * decreaseRate

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a1 : Nat) (an : Nat) (n : Nat) : Nat :=
  n * (a1 + an) / 2

/-- Theorem stating that a brick wall with given properties has 43 bricks in the bottom row -/
theorem bottom_row_bricks (wall : BrickWall)
  (h1 : wall.totalRows = 10)
  (h2 : wall.totalBricks = 385)
  (h3 : wall.decreaseRate = 1)
  : wall.bottomRowBricks = 43 := by
  sorry

end bottom_row_bricks_l168_16870


namespace sin_cos_identity_l168_16873

theorem sin_cos_identity : 
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) + 
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_identity_l168_16873


namespace roots_of_polynomial_l168_16879

/-- The polynomial function defined by x^3(x-3)^2(2+x) -/
def f (x : ℝ) : ℝ := x^3 * (x-3)^2 * (2+x)

/-- The set of roots of the polynomial equation x^3(x-3)^2(2+x) = 0 -/
def roots : Set ℝ := {x : ℝ | f x = 0}

/-- Theorem: The set of roots of the polynomial equation x^3(x-3)^2(2+x) = 0 is {0, 3, -2} -/
theorem roots_of_polynomial : roots = {0, 3, -2} := by
  sorry

end roots_of_polynomial_l168_16879


namespace discount_rate_proof_l168_16815

theorem discount_rate_proof (initial_price final_price : ℝ) 
  (h1 : initial_price = 7200)
  (h2 : final_price = 3528)
  (h3 : ∃ x : ℝ, initial_price * (1 - x)^2 = final_price) :
  ∃ x : ℝ, x = 0.3 ∧ initial_price * (1 - x)^2 = final_price := by
sorry

end discount_rate_proof_l168_16815


namespace operations_correct_l168_16848

-- Define the operations
def operation3 (x : ℝ) : Prop := x ≠ 0 → x^6 / x^3 = x^3
def operation4 (x : ℝ) : Prop := (x^3)^2 = x^6

-- Theorem stating that both operations are correct
theorem operations_correct : 
  (∀ x : ℝ, operation3 x) ∧ (∀ x : ℝ, operation4 x) := by sorry

end operations_correct_l168_16848


namespace min_value_theorem_l168_16842

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 4 ∧
  (∀ (c d : ℝ), c > 0 → d > 0 → c + d = 4 → 1 / (c + 1) + 1 / (d + 3) ≥ 1 / (x + 1) + 1 / (y + 3)) ∧
  1 / (x + 1) + 1 / (y + 3) = 1 / 2 := by
  sorry

end min_value_theorem_l168_16842


namespace largest_value_l168_16895

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def digits_85_9 : List Nat := [8, 5]
def digits_111111_2 : List Nat := [1, 1, 1, 1, 1, 1]
def digits_1000_4 : List Nat := [1, 0, 0, 0]
def digits_210_6 : List Nat := [2, 1, 0]

theorem largest_value :
  let a := to_decimal digits_85_9 9
  let b := to_decimal digits_111111_2 2
  let c := to_decimal digits_1000_4 4
  let d := to_decimal digits_210_6 6
  d > a ∧ d > b ∧ d > c := by sorry

end largest_value_l168_16895


namespace fraction_power_rule_l168_16876

theorem fraction_power_rule (a b : ℝ) (hb : b ≠ 0) : (a / b) ^ 4 = a ^ 4 / b ^ 4 := by
  sorry

end fraction_power_rule_l168_16876


namespace max_cos_sin_sum_l168_16809

open Real

theorem max_cos_sin_sum (α β γ : ℝ) (h1 : 0 < α ∧ α < π)
                                   (h2 : 0 < β ∧ β < π)
                                   (h3 : 0 < γ ∧ γ < π)
                                   (h4 : α + β + 2 * γ = π) :
  (∀ a b c, 0 < a ∧ a < π ∧ 0 < b ∧ b < π ∧ 0 < c ∧ c < π ∧ a + b + 2 * c = π →
    cos α + cos β + sin (2 * γ) ≥ cos a + cos b + sin (2 * c)) ∧
  cos α + cos β + sin (2 * γ) = 3 * sqrt 3 / 2 :=
sorry

end max_cos_sin_sum_l168_16809


namespace absolute_value_inequality_l168_16803

theorem absolute_value_inequality (x : ℝ) :
  (|x - 1| + |x + 2| ≤ 5) ↔ (x ∈ Set.Icc (-3) 2) :=
by sorry

end absolute_value_inequality_l168_16803


namespace sequence_sum_l168_16884

theorem sequence_sum (a b c d : ℕ) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧  -- increasing positive integers
  b - a = c - b ∧                  -- arithmetic progression
  c * c = b * d ∧                  -- geometric progression
  d - a = 42                       -- difference between first and fourth terms
  → a + b + c + d = 123 := by
sorry

end sequence_sum_l168_16884


namespace charity_ticket_sales_l168_16866

theorem charity_ticket_sales (full_price_tickets half_price_tickets : ℕ) 
  (full_price half_price : ℚ) : 
  full_price_tickets + half_price_tickets = 160 →
  full_price_tickets * full_price + half_price_tickets * half_price = 2400 →
  half_price = full_price / 2 →
  full_price_tickets * full_price = 960 := by
sorry

end charity_ticket_sales_l168_16866


namespace g_is_odd_l168_16818

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define g as f(x) - f(-x)
def g (x : ℝ) : ℝ := f x - f (-x)

-- Theorem: g is an odd function
theorem g_is_odd : ∀ x : ℝ, g f (-x) = -(g f x) := by
  sorry

end g_is_odd_l168_16818


namespace school_play_attendance_l168_16898

theorem school_play_attendance : 
  let num_girls : ℕ := 10
  let num_boys : ℕ := 12
  let family_members_per_kid : ℕ := 3
  let kids_with_stepparent : ℕ := 5
  let kids_with_grandparents : ℕ := 3
  let additional_grandparents_per_kid : ℕ := 2

  (num_girls + num_boys) * family_members_per_kid + 
  kids_with_stepparent + 
  kids_with_grandparents * additional_grandparents_per_kid = 77 :=
by sorry

end school_play_attendance_l168_16898


namespace angle_C_is_pi_third_max_area_when_a_b_equal_c_max_area_is_sqrt_three_l168_16896

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition √3(a - c cos B) = b sin C -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 3 * (t.a - t.c * Real.cos t.B) = t.b * Real.sin t.C

theorem angle_C_is_pi_third (t : Triangle) (h : condition t) : t.C = π / 3 := by
  sorry

theorem max_area_when_a_b_equal_c (t : Triangle) (h : condition t) (hc : t.c = 2) :
  (∀ t' : Triangle, condition t' → t'.c = 2 → t.a * t.b ≥ t'.a * t'.b) →
  t.a = 2 ∧ t.b = 2 := by
  sorry

theorem max_area_is_sqrt_three (t : Triangle) (h : condition t) (hc : t.c = 2)
  (hmax : t.a = 2 ∧ t.b = 2) :
  (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end angle_C_is_pi_third_max_area_when_a_b_equal_c_max_area_is_sqrt_three_l168_16896


namespace infinitely_many_composites_l168_16847

/-- A strictly increasing sequence of natural numbers where each number from
    the third one onwards is the sum of some two preceding numbers. -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ n ≥ 2, ∃ i j, i < n ∧ j < n ∧ a n = a i + a j)

/-- A number is composite if it's not prime and greater than 1. -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- The main theorem stating that there are infinitely many composite numbers
    in a special sequence. -/
theorem infinitely_many_composites (a : ℕ → ℕ) (h : SpecialSequence a) :
    ∀ N, ∃ n > N, IsComposite (a n) :=
  sorry

end infinitely_many_composites_l168_16847


namespace rachel_adam_weight_difference_l168_16856

/-- Given the weights of three people Rachel, Jimmy, and Adam, prove that Rachel weighs 15 pounds more than Adam. -/
theorem rachel_adam_weight_difference (R J A : ℝ) : 
  R = 75 →  -- Rachel weighs 75 pounds
  R = J - 6 →  -- Rachel weighs 6 pounds less than Jimmy
  R > A →  -- Rachel weighs more than Adam
  (R + J + A) / 3 = 72 →  -- The average weight of the three people is 72 pounds
  R - A = 15 :=  -- Rachel weighs 15 pounds more than Adam
by sorry

end rachel_adam_weight_difference_l168_16856


namespace system_of_equations_l168_16814

theorem system_of_equations (y : ℝ) :
  ∃ (x z : ℝ),
    (19 * (x + y) + 17 = 19 * (-x + y) - 21) ∧
    (5 * x - 3 * z = 11 * y - 7) ∧
    (x = -1) ∧
    (z = -11 * y / 3 + 2 / 3) := by
  sorry

end system_of_equations_l168_16814


namespace parabola_distance_l168_16890

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def point_B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance (A : ℝ × ℝ) :
  parabola A.1 A.2 →
  ‖A - focus‖ = ‖point_B - focus‖ →
  ‖A - point_B‖ = 2 * Real.sqrt 2 := by
  sorry

end parabola_distance_l168_16890


namespace least_multiple_25_with_digit_product_125_l168_16805

def is_multiple_of_25 (n : ℕ) : Prop := ∃ k : ℕ, n = 25 * k

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  let digits := n.digits 10
  digits.prod

theorem least_multiple_25_with_digit_product_125 :
  ∀ n : ℕ, n > 0 → is_multiple_of_25 n → digit_product n = 125 → n ≥ 555 :=
sorry

end least_multiple_25_with_digit_product_125_l168_16805


namespace zion_age_is_8_dad_age_relation_future_age_relation_l168_16850

/-- Zion's current age in years -/
def zion_age : ℕ := 8

/-- Zion's dad's current age in years -/
def dad_age : ℕ := 4 * zion_age + 3

theorem zion_age_is_8 : zion_age = 8 := by sorry

theorem dad_age_relation : dad_age = 4 * zion_age + 3 := by sorry

theorem future_age_relation : dad_age + 10 = (zion_age + 10) + 27 := by sorry

end zion_age_is_8_dad_age_relation_future_age_relation_l168_16850


namespace nancy_and_rose_bracelets_l168_16897

/-- The number of beads required for each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has -/
def nancy_pearl_beads : ℕ := nancy_metal_beads + 20

/-- The number of crystal beads Rose has -/
def rose_crystal_beads : ℕ := 20

/-- The number of stone beads Rose has -/
def rose_stone_beads : ℕ := 2 * rose_crystal_beads

/-- The total number of beads Nancy and Rose have -/
def total_beads : ℕ := nancy_metal_beads + nancy_pearl_beads + rose_crystal_beads + rose_stone_beads

/-- The number of bracelets Nancy and Rose can make -/
def bracelets_made : ℕ := total_beads / beads_per_bracelet

theorem nancy_and_rose_bracelets : bracelets_made = 20 := by
  sorry

end nancy_and_rose_bracelets_l168_16897


namespace line_intersects_x_axis_l168_16893

/-- The line equation 3y - 4x = 12 -/
def line_equation (x y : ℝ) : Prop := 3 * y - 4 * x = 12

/-- The x-axis equation y = 0 -/
def x_axis (y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (-3, 0)

theorem line_intersects_x_axis :
  let (x, y) := intersection_point
  line_equation x y ∧ x_axis y := by sorry

end line_intersects_x_axis_l168_16893


namespace cross_tangential_cubic_cross_tangential_sine_cross_tangential_tangent_l168_16826

-- Define the concept of cross-tangential intersection
def cross_tangential_intersection (l : ℝ → ℝ) (c : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := p
  -- Condition (i): The line l is tangent to the curve C at the point P(x₀, y₀)
  (deriv c x₀ = deriv l x₀) ∧
  -- Condition (ii): The curve C lies on both sides of the line l near point P
  (∃ δ > 0, ∀ x ∈ Set.Ioo (x₀ - δ) (x₀ + δ),
    (x < x₀ → c x < l x) ∧ (x > x₀ → c x > l x) ∨
    (x < x₀ → c x > l x) ∧ (x > x₀ → c x < l x))

-- Statement 1
theorem cross_tangential_cubic :
  cross_tangential_intersection (λ _ => 0) (λ x => x^3) (0, 0) :=
sorry

-- Statement 3
theorem cross_tangential_sine :
  cross_tangential_intersection (λ x => x) Real.sin (0, 0) :=
sorry

-- Statement 4
theorem cross_tangential_tangent :
  cross_tangential_intersection (λ x => x) Real.tan (0, 0) :=
sorry

end cross_tangential_cubic_cross_tangential_sine_cross_tangential_tangent_l168_16826


namespace second_term_of_geometric_series_l168_16883

/-- 
Given an infinite geometric series with common ratio 1/4 and sum 40,
the second term of the sequence is 7.5.
-/
theorem second_term_of_geometric_series (a : ℝ) : 
  (∑' n, a * (1/4)^n = 40) → a * (1/4) = 7.5 := by
  sorry

end second_term_of_geometric_series_l168_16883


namespace quadratic_function_max_value_l168_16880

-- Define the quadratic function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def in_band (y k l : ℝ) : Prop := k ≤ y ∧ y ≤ l

theorem quadratic_function_max_value
  (a b c : ℝ)  -- Coefficients of f(x) = ax^2 + bx + c
  (h1 : in_band (f a b c (-2) + 2) 0 4)
  (h2 : in_band (f a b c 0 + 2) 0 4)
  (h3 : in_band (f a b c 2 + 2) 0 4)
  (h4 : ∀ t : ℝ, in_band (t + 1) (-1) 3 → in_band (f a b c t) (-5/2) (5/2)) :
  ∃ t : ℝ, |f a b c t| = 5/2 ∧ ∀ s : ℝ, |f a b c s| ≤ 5/2 :=
sorry

end quadratic_function_max_value_l168_16880


namespace no_prime_factor_6k_plus_5_l168_16843

theorem no_prime_factor_6k_plus_5 (n k : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_factor : p ∣ n^2 - n + 1) : p ≠ 6 * k + 5 := by
  sorry

end no_prime_factor_6k_plus_5_l168_16843


namespace inequality_solution_set_l168_16849

theorem inequality_solution_set (x : ℝ) : x - 3 > 4*x ↔ x < -1 := by sorry

end inequality_solution_set_l168_16849


namespace smallest_integer_satisfying_conditions_l168_16881

theorem smallest_integer_satisfying_conditions : 
  ∃ n : ℤ, (∀ m : ℤ, (m + 15 ≥ 16 ∧ -5 * m < -10) → n ≤ m) ∧ 
           (n + 15 ≥ 16 ∧ -5 * n < -10) := by
  sorry

end smallest_integer_satisfying_conditions_l168_16881


namespace sum_of_specific_terms_l168_16859

def tangent_sequence (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, k > 0 → a (k + 1) = (3 / 2) * a k

theorem sum_of_specific_terms 
  (a : ℕ → ℝ) 
  (h1 : tangent_sequence a) 
  (h2 : a 1 = 16) : 
  a 1 + a 3 + a 5 = 133 := by
sorry

end sum_of_specific_terms_l168_16859


namespace perpendicular_bisector_b_value_l168_16871

/-- Given that the line x + y = b is a perpendicular bisector of the line segment from (2,4) to (6,8), prove that b = 10. -/
theorem perpendicular_bisector_b_value : 
  ∀ b : ℝ, 
  (∀ x y : ℝ, x + y = b ↔ 
    (x - 4)^2 + (y - 6)^2 = (2 - 4)^2 + (4 - 6)^2 ∧ 
    (x - 4)^2 + (y - 6)^2 = (6 - 4)^2 + (8 - 6)^2) → 
  b = 10 := by
  sorry

end perpendicular_bisector_b_value_l168_16871


namespace difference_of_squares_l168_16800

theorem difference_of_squares : 65^2 - 55^2 = 1200 := by
  sorry

end difference_of_squares_l168_16800


namespace true_discount_equals_bankers_gain_l168_16822

/-- Present worth of a sum due -/
def present_worth : ℝ := 576

/-- Banker's gain -/
def bankers_gain : ℝ := 16

/-- True discount -/
def true_discount : ℝ := bankers_gain

theorem true_discount_equals_bankers_gain :
  true_discount = bankers_gain :=
by sorry

end true_discount_equals_bankers_gain_l168_16822


namespace magician_reappeared_count_l168_16885

/-- Represents the magician's performance statistics --/
structure MagicianStats where
  total_shows : ℕ
  min_audience : ℕ
  max_audience : ℕ
  disappear_ratio : ℕ
  no_reappear_prob : ℚ
  double_reappear_prob : ℚ
  triple_reappear_prob : ℚ

/-- Calculates the total number of people who reappeared in the magician's performances --/
def total_reappeared (stats : MagicianStats) : ℕ :=
  sorry

/-- Theorem stating that given the magician's performance statistics, 
    the total number of people who reappeared is 640 --/
theorem magician_reappeared_count (stats : MagicianStats) 
  (h1 : stats.total_shows = 100)
  (h2 : stats.min_audience = 50)
  (h3 : stats.max_audience = 500)
  (h4 : stats.disappear_ratio = 50)
  (h5 : stats.no_reappear_prob = 1/10)
  (h6 : stats.double_reappear_prob = 1/5)
  (h7 : stats.triple_reappear_prob = 1/20) :
  total_reappeared stats = 640 :=
sorry

end magician_reappeared_count_l168_16885


namespace mrs_thompson_potatoes_cost_l168_16833

/-- Calculates the cost of potatoes given the number of chickens, cost per chicken, and total amount paid. -/
def cost_of_potatoes (num_chickens : ℕ) (cost_per_chicken : ℕ) (total_paid : ℕ) : ℕ :=
  total_paid - (num_chickens * cost_per_chicken)

/-- Proves that the cost of potatoes is 6 given the specific conditions of Mrs. Thompson's purchase. -/
theorem mrs_thompson_potatoes_cost :
  cost_of_potatoes 3 3 15 = 6 := by
  sorry

end mrs_thompson_potatoes_cost_l168_16833


namespace simplify_and_evaluate_l168_16801

theorem simplify_and_evaluate (x : ℝ) (h : x = 1) : 
  (4 / (x^2 - 4)) / (2 / (x - 2)) = 2 / 3 := by
  sorry

end simplify_and_evaluate_l168_16801


namespace sam_yellow_marbles_l168_16804

theorem sam_yellow_marbles (initial_yellow : ℝ) (received_yellow : ℝ) 
  (h1 : initial_yellow = 86.0) (h2 : received_yellow = 25.0) :
  initial_yellow + received_yellow = 111.0 := by
  sorry

end sam_yellow_marbles_l168_16804


namespace diophantine_equation_solution_l168_16894

theorem diophantine_equation_solution :
  ∃ (a b c d : ℕ+), 4^(a : ℕ) * 5^(b : ℕ) - 3^(c : ℕ) * 11^(d : ℕ) = 1 :=
by
  sorry

end diophantine_equation_solution_l168_16894


namespace floor_x_width_l168_16862

/-- Represents a rectangular floor with a length and width in feet. -/
structure RectangularFloor where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular floor. -/
def area (floor : RectangularFloor) : ℝ :=
  floor.length * floor.width

theorem floor_x_width
  (x y : RectangularFloor)
  (h1 : area x = area y)
  (h2 : x.length = 18)
  (h3 : y.width = 9)
  (h4 : y.length = 20) :
  x.width = 10 := by
  sorry

end floor_x_width_l168_16862


namespace mod_equivalence_problem_l168_16860

theorem mod_equivalence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -4378 [ZMOD 6] ∧ n = 2 := by
  sorry

end mod_equivalence_problem_l168_16860


namespace pig_profit_is_960_l168_16838

/-- Calculates the profit from selling pigs given the specified conditions -/
def calculate_pig_profit (num_piglets : ℕ) (sale_price : ℕ) (feeding_cost : ℕ) 
  (months_group1 : ℕ) (months_group2 : ℕ) : ℕ :=
  let revenue := num_piglets * sale_price
  let cost_group1 := (num_piglets / 2) * feeding_cost * months_group1
  let cost_group2 := (num_piglets / 2) * feeding_cost * months_group2
  let total_cost := cost_group1 + cost_group2
  revenue - total_cost

/-- The profit from selling pigs under the given conditions is $960 -/
theorem pig_profit_is_960 : 
  calculate_pig_profit 6 300 10 12 16 = 960 := by
  sorry

end pig_profit_is_960_l168_16838


namespace rectangular_solid_surface_area_l168_16872

/-- The total surface area of a rectangular solid -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 6 meters, width 5 meters, and depth 2 meters is 104 square meters -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 6 5 2 = 104 := by
  sorry

end rectangular_solid_surface_area_l168_16872


namespace average_of_9_15_N_l168_16858

theorem average_of_9_15_N (N : ℝ) (h1 : 12 < N) (h2 : N < 25) :
  let avg := (9 + 15 + N) / 3
  avg = 15 ∨ avg = 17 :=
by sorry

end average_of_9_15_N_l168_16858


namespace somu_father_age_ratio_l168_16857

/-- Represents the ratio of two integers as a pair of integers -/
def Ratio := ℤ × ℤ

/-- Somu's present age in years -/
def somu_age : ℕ := 10

/-- Calculates the father's age given Somu's age -/
def father_age (s : ℕ) : ℕ :=
  5 * (s - 5) + 5

/-- Simplifies a ratio by dividing both numbers by their greatest common divisor -/
def simplify_ratio (r : Ratio) : Ratio :=
  let gcd := r.1.gcd r.2
  (r.1 / gcd, r.2 / gcd)

theorem somu_father_age_ratio :
  simplify_ratio (somu_age, father_age somu_age) = (1, 3) := by
  sorry

end somu_father_age_ratio_l168_16857


namespace remainder_theorem_l168_16839

theorem remainder_theorem (r : ℤ) : (r^11 - 1) % (r - 2) = 2047 := by
  sorry

end remainder_theorem_l168_16839


namespace prob_at_most_one_red_l168_16820

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def red_balls : ℕ := 2
def drawn_balls : ℕ := 3

theorem prob_at_most_one_red :
  (1 : ℚ) - (Nat.choose white_balls 1 * Nat.choose red_balls 2 : ℚ) / Nat.choose total_balls drawn_balls = 7/10 := by
  sorry

end prob_at_most_one_red_l168_16820


namespace pauls_strawberries_l168_16888

/-- Given an initial count of strawberries and an additional number picked,
    calculate the total number of strawberries. -/
def total_strawberries (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem: Paul's total strawberries after picking more -/
theorem pauls_strawberries :
  total_strawberries 42 78 = 120 := by
  sorry

end pauls_strawberries_l168_16888


namespace temporary_employee_percentage_l168_16832

theorem temporary_employee_percentage 
  (total_workers : ℕ) 
  (technicians : ℕ) 
  (non_technicians : ℕ) 
  (permanent_technicians : ℕ) 
  (permanent_non_technicians : ℕ) 
  (h1 : technicians = total_workers / 2) 
  (h2 : non_technicians = total_workers / 2) 
  (h3 : permanent_technicians = technicians / 2) 
  (h4 : permanent_non_technicians = non_technicians / 2) :
  (total_workers - (permanent_technicians + permanent_non_technicians)) * 100 / total_workers = 50 := by
  sorry

end temporary_employee_percentage_l168_16832


namespace connected_vessels_equilibrium_l168_16845

/-- Represents the final levels of liquids in two connected vessels after equilibrium -/
def FinalLevels (H : ℝ) : ℝ × ℝ :=
  (0.69 * H, H)

/-- Proves that the given final levels are correct for the connected vessels problem -/
theorem connected_vessels_equilibrium 
  (H : ℝ) 
  (h_positive : H > 0) 
  (ρ_water : ℝ) 
  (ρ_gasoline : ℝ) 
  (h_water_density : ρ_water = 1000) 
  (h_gasoline_density : ρ_gasoline = 600) 
  (h_initial_level : ℝ) 
  (h_initial : h_initial = 0.9 * H) 
  (h_tap_height : ℝ) 
  (h_tap : h_tap_height = 0.2 * H) : 
  FinalLevels H = (0.69 * H, H) :=
sorry

#check connected_vessels_equilibrium

end connected_vessels_equilibrium_l168_16845


namespace complex_triplet_theorem_l168_16810

theorem complex_triplet_theorem (a b c : ℂ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  Complex.abs a = Complex.abs b → 
  Complex.abs b = Complex.abs c → 
  a / b + b / c + c / a = -1 → 
  ((a = b ∧ c = -a) ∨ (b = c ∧ a = -b) ∨ (c = a ∧ b = -c)) := by
sorry

end complex_triplet_theorem_l168_16810


namespace initial_shells_l168_16868

theorem initial_shells (initial_amount added_amount total_amount : ℕ) 
  (h1 : added_amount = 12)
  (h2 : total_amount = 17)
  (h3 : initial_amount + added_amount = total_amount) :
  initial_amount = 5 := by
  sorry

end initial_shells_l168_16868


namespace apples_removed_by_ricki_l168_16864

/-- The number of apples Ricki removed -/
def rickis_apples : ℕ := 14

/-- The initial number of apples in the basket -/
def initial_apples : ℕ := 74

/-- The final number of apples in the basket -/
def final_apples : ℕ := 32

/-- Samson removed twice as many apples as Ricki -/
def samsons_apples : ℕ := 2 * rickis_apples

theorem apples_removed_by_ricki :
  rickis_apples = 14 ∧
  initial_apples = final_apples + rickis_apples + samsons_apples :=
by sorry

end apples_removed_by_ricki_l168_16864


namespace coordinate_system_proofs_l168_16827

-- Define the points A, B, and C
def A (a : ℝ) : ℝ × ℝ := (-2, a + 1)
def B (a : ℝ) : ℝ × ℝ := (a - 1, 4)
def C (b : ℝ) : ℝ × ℝ := (b - 2, b)

-- Define the conditions and prove the statements
theorem coordinate_system_proofs :
  -- 1. When C is on the x-axis, its coordinates are (-2,0)
  (∃ b : ℝ, C b = (-2, 0) ∧ (C b).2 = 0) ∧
  -- 2. When C is on the y-axis, its coordinates are (0,2)
  (∃ b : ℝ, C b = (0, 2) ∧ (C b).1 = 0) ∧
  -- 3. When AB is parallel to the x-axis, the distance between A and B is 4
  (∃ a : ℝ, (A a).2 = (B a).2 ∧ Real.sqrt ((A a).1 - (B a).1)^2 = 4) ∧
  -- 4. When CD is perpendicular to the x-axis at point D and CD=1, 
  --    the coordinates of C are either (-1,1) or (-3,-1)
  (∃ b d : ℝ, (C b).1 = d ∧ Real.sqrt ((C b).1 - d)^2 + (C b).2^2 = 1 ∧
    ((C b = (-1, 1)) ∨ (C b = (-3, -1)))) :=
by sorry

end coordinate_system_proofs_l168_16827


namespace M_is_empty_l168_16887

def M : Set ℝ := {x | x^4 + 4*x^2 - 12*x + 8 = 0 ∧ x > 0}

theorem M_is_empty : M = ∅ := by
  sorry

end M_is_empty_l168_16887


namespace solve_equation_l168_16878

theorem solve_equation (x : ℝ) :
  (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) → x = -15 := by
sorry

end solve_equation_l168_16878


namespace wally_bear_cost_l168_16877

/-- Calculates the total cost of bears given the number of bears, initial price, and discount per bear. -/
def total_cost (num_bears : ℕ) (initial_price : ℚ) (discount : ℚ) : ℚ :=
  initial_price + (num_bears - 1 : ℚ) * (initial_price - discount)

/-- Theorem stating that the total cost for 101 bears is $354, given the specified pricing scheme. -/
theorem wally_bear_cost :
  total_cost 101 4 0.5 = 354 := by
  sorry

end wally_bear_cost_l168_16877


namespace xiao_ming_foot_length_l168_16816

/-- The relationship between a person's height and foot length -/
def height_foot_relation (h d : ℝ) : Prop := h = 7 * d

/-- Xiao Ming's height in cm -/
def xiao_ming_height : ℝ := 171.5

theorem xiao_ming_foot_length :
  ∃ d : ℝ, height_foot_relation xiao_ming_height d ∧ d = 24.5 := by
  sorry

end xiao_ming_foot_length_l168_16816


namespace derivative_f_at_one_l168_16824

def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

theorem derivative_f_at_one :
  deriv f 1 = 4 := by
  sorry

end derivative_f_at_one_l168_16824


namespace travel_time_difference_l168_16891

/-- Proves that the difference in travel time between a 400-mile trip and a 360-mile trip,
    when traveling at a constant speed of 40 miles per hour, is 60 minutes. -/
theorem travel_time_difference (speed : ℝ) (dist1 : ℝ) (dist2 : ℝ) :
  speed = 40 → dist1 = 400 → dist2 = 360 →
  (dist1 / speed - dist2 / speed) * 60 = 60 := by
  sorry

end travel_time_difference_l168_16891


namespace yo_yo_count_l168_16855

theorem yo_yo_count 
  (x y z w : ℕ) 
  (h1 : x + y + w = 80)
  (h2 : (3/5 : ℚ) * 300 + (1/5 : ℚ) * 300 = x + y + z + w + 15)
  (h3 : x + y + z + w = 300 - ((3/5 : ℚ) * 300 + (1/5 : ℚ) * 300)) :
  z = 145 := by
  sorry

#check yo_yo_count

end yo_yo_count_l168_16855
