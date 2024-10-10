import Mathlib

namespace quadratic_p_value_l558_55844

/-- The quadratic function passing through specific points -/
def quadratic_function (p : ℝ) (x : ℝ) : ℝ := p * x^2 + 5 * x + p

theorem quadratic_p_value :
  ∃ (p : ℝ), 
    (quadratic_function p 0 = -2) ∧ 
    (quadratic_function p (1/2) = 0) ∧ 
    (quadratic_function p 2 = 0) ∧
    (p = -2) := by
  sorry

end quadratic_p_value_l558_55844


namespace complement_of_A_in_U_l558_55817

-- Define the universal set U
def U : Set ℝ := {x | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 < x ∧ x < 3)} := by sorry

end complement_of_A_in_U_l558_55817


namespace triangle_angle_ratio_right_angle_l558_55884

theorem triangle_angle_ratio_right_angle (α β γ : ℝ) (h_sum : α + β + γ = π) 
  (h_ratio : α = 3 * γ ∧ β = 2 * γ) : α = π / 2 := by
  sorry

end triangle_angle_ratio_right_angle_l558_55884


namespace min_colors_condition_1_min_colors_condition_2_l558_55845

variable (n : ℕ)

/-- The set of all lattice points in n-dimensional space -/
def X : Set (Fin n → ℤ) := Set.univ

/-- Distance between two lattice points -/
def distance (A B : Fin n → ℤ) : ℕ :=
  (Finset.univ.sum fun i => (A i - B i).natAbs)

/-- A coloring of X is valid for Condition 1 if any two points of the same color have distance ≥ 2 -/
def valid_coloring_1 (c : (Fin n → ℤ) → Fin r) : Prop :=
  ∀ A B : Fin n → ℤ, c A = c B → distance n A B ≥ 2

/-- A coloring of X is valid for Condition 2 if any two points of the same color have distance ≥ 3 -/
def valid_coloring_2 (c : (Fin n → ℤ) → Fin r) : Prop :=
  ∀ A B : Fin n → ℤ, c A = c B → distance n A B ≥ 3

/-- The minimum number of colors needed to satisfy Condition 1 is 2 -/
theorem min_colors_condition_1 :
  (∃ c : (Fin n → ℤ) → Fin 2, valid_coloring_1 n c) ∧
  (∀ r < 2, ¬∃ c : (Fin n → ℤ) → Fin r, valid_coloring_1 n c) :=
sorry

/-- The minimum number of colors needed to satisfy Condition 2 is 2n + 1 -/
theorem min_colors_condition_2 :
  (∃ c : (Fin n → ℤ) → Fin (2 * n + 1), valid_coloring_2 n c) ∧
  (∀ r < 2 * n + 1, ¬∃ c : (Fin n → ℤ) → Fin r, valid_coloring_2 n c) :=
sorry

end min_colors_condition_1_min_colors_condition_2_l558_55845


namespace fred_dimes_remaining_l558_55859

theorem fred_dimes_remaining (initial_dimes borrowed_dimes : ℕ) :
  initial_dimes = 7 →
  borrowed_dimes = 3 →
  initial_dimes - borrowed_dimes = 4 := by
  sorry

end fred_dimes_remaining_l558_55859


namespace distance_traveled_correct_mrs_hilt_trip_distance_l558_55896

/-- Calculates the distance traveled given initial and final odometer readings -/
def distance_traveled (initial_reading final_reading : ℝ) : ℝ :=
  final_reading - initial_reading

/-- Theorem: The distance traveled is the difference between final and initial odometer readings -/
theorem distance_traveled_correct (initial_reading final_reading : ℝ) :
  distance_traveled initial_reading final_reading = final_reading - initial_reading :=
by sorry

/-- Mrs. Hilt's trip distance calculation -/
theorem mrs_hilt_trip_distance :
  distance_traveled 212.3 372 = 159.7 :=
by sorry

end distance_traveled_correct_mrs_hilt_trip_distance_l558_55896


namespace cube_sum_and_product_theorem_l558_55836

theorem cube_sum_and_product_theorem :
  ∃! (n : ℕ), ∃ (a b : ℕ+),
    a ^ 3 + b ^ 3 = 189 ∧
    a * b = 20 ∧
    n = 2 :=
sorry

end cube_sum_and_product_theorem_l558_55836


namespace largest_n_for_divisibility_l558_55802

theorem largest_n_for_divisibility (n p q : ℕ+) : 
  (n.val ^ 3 + p.val) % (n.val + q.val) = 0 → 
  n.val ≤ 3060 ∧ 
  (n.val = 3060 → p.val = 300 ∧ q.val = 15) :=
sorry

end largest_n_for_divisibility_l558_55802


namespace lu_daokui_scholarship_winners_l558_55809

theorem lu_daokui_scholarship_winners 
  (total_winners : ℕ) 
  (first_prize_amount : ℕ) 
  (second_prize_amount : ℕ) 
  (total_prize_money : ℕ) 
  (h1 : total_winners = 28)
  (h2 : first_prize_amount = 10000)
  (h3 : second_prize_amount = 2000)
  (h4 : total_prize_money = 80000) :
  ∃ (first_prize_winners second_prize_winners : ℕ),
    first_prize_winners + second_prize_winners = total_winners ∧
    first_prize_winners * first_prize_amount + second_prize_winners * second_prize_amount = total_prize_money ∧
    first_prize_winners = 3 ∧
    second_prize_winners = 25 := by
  sorry

end lu_daokui_scholarship_winners_l558_55809


namespace employee_pay_l558_55803

theorem employee_pay (total : ℝ) (x y : ℝ) (h1 : total = 616) (h2 : x = 1.2 * y) (h3 : total = x + y) : y = 280 := by
  sorry

end employee_pay_l558_55803


namespace inverse_variation_cube_fourth_l558_55822

/-- Given that a^3 varies inversely with b^4, prove that if a = 5 when b = 2, then a = 5/2 when b = 4 -/
theorem inverse_variation_cube_fourth (a b : ℝ) (h : ∃ k : ℝ, ∀ a b, a^3 * b^4 = k) :
  (5^3 * 2^4 = a^3 * 4^4) → a = 5/2 := by sorry

end inverse_variation_cube_fourth_l558_55822


namespace point_2_3_in_first_quadrant_l558_55821

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Theorem: The point (2, 3) is in the first quadrant -/
theorem point_2_3_in_first_quadrant :
  let p : Point := ⟨2, 3⟩
  isInFirstQuadrant p := by
  sorry

end point_2_3_in_first_quadrant_l558_55821


namespace geometric_sequence_product_range_l558_55878

theorem geometric_sequence_product_range (a₁ a₂ a₃ m q : ℝ) (hm : m > 0) (hq : q > 0) :
  (a₁ + a₂ + a₃ = 3 * m) →
  (a₂ = a₁ * q) →
  (a₃ = a₂ * q) →
  let t := a₁ * a₂ * a₃
  0 < t ∧ t ≤ m^3 :=
by sorry

end geometric_sequence_product_range_l558_55878


namespace f_composition_of_three_l558_55860

def f (n : ℕ) : ℕ :=
  if n ≤ 3 then n^2 + 1 else 4*n + 2

theorem f_composition_of_three : f (f (f 3)) = 170 := by
  sorry

end f_composition_of_three_l558_55860


namespace variance_fluctuation_relationship_l558_55814

/-- Definition of variance for a list of numbers -/
def variance (data : List ℝ) : ℝ := sorry

/-- Definition of fluctuation for a list of numbers -/
def fluctuation (data : List ℝ) : ℝ := sorry

/-- Theorem: If the variance of data set A is greater than the variance of data set B,
    then the fluctuation of A is greater than the fluctuation of B -/
theorem variance_fluctuation_relationship (A B : List ℝ) :
  variance A > variance B → fluctuation A > fluctuation B := by sorry

end variance_fluctuation_relationship_l558_55814


namespace parabola_properties_l558_55840

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus
def focus : ℝ × ℝ := (0, 4)

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define point R
def R : ℝ × ℝ := (4, 6)

-- Define the property that R is the midpoint of PQ
def R_is_midpoint (P Q : ℝ × ℝ) : Prop :=
  R.1 = (P.1 + Q.1) / 2 ∧ R.2 = (P.2 + Q.2) / 2

-- Define point A as the intersection of tangents
def A : ℝ × ℝ := (4, -2)

-- Theorem statement
theorem parabola_properties
  (P Q : ℝ × ℝ)
  (hP : parabola P.1 P.2)
  (hQ : parabola Q.1 Q.2)
  (hl_P : line_l P.1 P.2)
  (hl_Q : line_l Q.1 Q.2)
  (hR : R_is_midpoint P Q) :
  (∃ (AF : ℝ), AF ≠ 4 * Real.sqrt 2 ∧ AF = Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2)) ∧
  (∃ (PQ : ℝ), PQ = 12 ∧ PQ = Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
  (¬ ((P.2 - Q.2) * (A.1 - focus.1) = -(P.1 - Q.1) * (A.2 - focus.2))) ∧
  (∃ (center : ℝ × ℝ), 
    (center.1 - P.1)^2 + (center.2 - P.2)^2 = (center.1 - Q.1)^2 + (center.2 - Q.2)^2 ∧
    (center.1 - A.1)^2 + (center.2 - A.2)^2 = (center.1 - P.1)^2 + (center.2 - P.2)^2) := by
  sorry

end parabola_properties_l558_55840


namespace greatest_p_value_l558_55808

theorem greatest_p_value (x : ℝ) (p : ℝ) : 
  (∃ x, 2 * Real.cos (2 * Real.pi - Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2)) - 3 = 
        p - 2 * Real.sin (-Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2))) →
  p ≤ -2 :=
by sorry

end greatest_p_value_l558_55808


namespace johns_bonus_is_twenty_l558_55863

/-- Calculate the performance bonus for John's job --/
def performance_bonus (normal_wage : ℝ) (normal_hours : ℝ) (extra_hours : ℝ) (bonus_rate : ℝ) : ℝ :=
  (normal_hours + extra_hours) * bonus_rate - normal_wage

/-- Theorem stating that John's performance bonus is $20 per day --/
theorem johns_bonus_is_twenty :
  performance_bonus 80 8 2 10 = 20 := by
  sorry

end johns_bonus_is_twenty_l558_55863


namespace remainder_3_100_mod_7_l558_55828

theorem remainder_3_100_mod_7 : 3^100 % 7 = 4 := by
  sorry

end remainder_3_100_mod_7_l558_55828


namespace q_satisfies_conditions_l558_55854

/-- The quartic polynomial q(x) that satisfies given conditions -/
def q (x : ℚ) : ℚ := (1/6) * x^4 - (8/3) * x^3 - (14/3) * x^2 - (8/3) * x - 16/3

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q 1 = -8 ∧ q 2 = -18 ∧ q 3 = -40 ∧ q 4 = -80 ∧ q 5 = -140 := by
  sorry

end q_satisfies_conditions_l558_55854


namespace libby_quarters_l558_55851

/-- The number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- The cost of the dress in dollars -/
def dress_cost : ℕ := 35

/-- The number of quarters Libby has left after paying for the dress -/
def quarters_left : ℕ := 20

/-- The initial number of quarters Libby had -/
def initial_quarters : ℕ := dress_cost * quarters_per_dollar + quarters_left

theorem libby_quarters : initial_quarters = 160 := by
  sorry

end libby_quarters_l558_55851


namespace mehki_age_l558_55897

/-- Given the ages of Mehki, Jordyn, and Zrinka, prove Mehki's age -/
theorem mehki_age (zrinka jordyn mehki : ℕ) 
  (h1 : mehki = jordyn + 10)
  (h2 : jordyn = 2 * zrinka)
  (h3 : zrinka = 6) :
  mehki = 22 := by sorry

end mehki_age_l558_55897


namespace exists_number_divisible_by_power_of_two_l558_55830

theorem exists_number_divisible_by_power_of_two (n : ℕ) :
  ∃ k : ℕ, (∀ d : ℕ, d < n → (k / 10^d % 10 = 1 ∨ k / 10^d % 10 = 2)) ∧ k % 2^n = 0 := by
  sorry

end exists_number_divisible_by_power_of_two_l558_55830


namespace sum_of_roots_quadratic_l558_55891

theorem sum_of_roots_quadratic : ∃ (x₁ x₂ : ℝ), 
  x₁^2 - 7*x₁ + 10 = 0 ∧ 
  x₂^2 - 7*x₂ + 10 = 0 ∧ 
  x₁ + x₂ = 7 := by
sorry

end sum_of_roots_quadratic_l558_55891


namespace complement_of_A_in_U_l558_55849

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {3, 4, 5} := by sorry

end complement_of_A_in_U_l558_55849


namespace chocolate_division_l558_55842

theorem chocolate_division (total_chocolate : ℚ) (piles : ℕ) (friends : ℕ) : 
  total_chocolate = 60 / 7 →
  piles = 5 →
  friends = 3 →
  (total_chocolate / piles * (piles - 1)) / friends = 16 / 7 := by
  sorry

end chocolate_division_l558_55842


namespace five_sundays_april_implies_five_mondays_may_l558_55806

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific month -/
structure Month where
  numDays : Nat
  firstDay : DayOfWeek

/-- Given a day, returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the number of occurrences of a specific day in a month -/
def countDaysInMonth (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If April has five Sundays, then May has five Mondays -/
theorem five_sundays_april_implies_five_mondays_may :
  ∀ (april : Month) (may : Month),
    april.numDays = 30 →
    may.numDays = 31 →
    may.firstDay = nextDay april.firstDay →
    countDaysInMonth april DayOfWeek.Sunday = 5 →
    countDaysInMonth may DayOfWeek.Monday = 5 :=
  sorry

end five_sundays_april_implies_five_mondays_may_l558_55806


namespace max_value_sqrt_sum_l558_55881

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end max_value_sqrt_sum_l558_55881


namespace necklace_count_l558_55898

/-- The number of unique necklaces made from 5 red and 2 blue beads -/
def unique_necklaces : ℕ := 3

/-- The number of red beads in each necklace -/
def red_beads : ℕ := 5

/-- The number of blue beads in each necklace -/
def blue_beads : ℕ := 2

/-- The total number of beads in each necklace -/
def total_beads : ℕ := red_beads + blue_beads

/-- Theorem stating that the number of unique necklaces is 3 -/
theorem necklace_count : unique_necklaces = 3 := by sorry

end necklace_count_l558_55898


namespace regular_hexagon_perimeter_l558_55804

/-- The perimeter of a regular hexagon given the distance between opposite sides -/
theorem regular_hexagon_perimeter (d : ℝ) (h : d = 15) : 
  let s := 2 * d / Real.sqrt 3
  6 * s = 60 * Real.sqrt 3 := by sorry

end regular_hexagon_perimeter_l558_55804


namespace mother_age_four_times_yujeong_age_l558_55885

/-- Represents the age difference between the current year and the year in question -/
def yearDifference : ℕ := 2

/-- Yujeong's current age -/
def yujeongCurrentAge : ℕ := 12

/-- Yujeong's mother's current age -/
def motherCurrentAge : ℕ := 42

/-- Theorem stating that 2 years ago, Yujeong's mother's age was 4 times Yujeong's age -/
theorem mother_age_four_times_yujeong_age :
  (motherCurrentAge - yearDifference) = 4 * (yujeongCurrentAge - yearDifference) := by
  sorry

end mother_age_four_times_yujeong_age_l558_55885


namespace unripe_oranges_zero_l558_55869

/-- Represents the daily harvest of oranges -/
structure DailyHarvest where
  ripe : ℕ
  unripe : ℕ

/-- Represents the total harvest over a period of days -/
structure TotalHarvest where
  days : ℕ
  ripe : ℕ

/-- Proves that the number of unripe oranges harvested per day is zero -/
theorem unripe_oranges_zero 
  (daily : DailyHarvest) 
  (total : TotalHarvest) 
  (h1 : daily.ripe = 82)
  (h2 : total.days = 25)
  (h3 : total.ripe = 2050)
  (h4 : daily.ripe * total.days = total.ripe) :
  daily.unripe = 0 := by
  sorry

#check unripe_oranges_zero

end unripe_oranges_zero_l558_55869


namespace sum_of_specific_T_l558_55874

/-- Definition of T_n for n ≥ 2 -/
def T (n : ℕ) : ℤ :=
  if n < 2 then 0 else
  if n % 2 = 0 then -n / 2 else (n + 1) / 2

/-- Theorem stating that T₂₀ + T₃₆ + T₄₅ = -5 -/
theorem sum_of_specific_T : T 20 + T 36 + T 45 = -5 := by
  sorry

end sum_of_specific_T_l558_55874


namespace auction_result_l558_55876

/-- Calculates the total amount received from selling a TV and a phone at an auction -/
def auction_total (tv_cost phone_cost : ℚ) (tv_increase phone_increase : ℚ) : ℚ :=
  (tv_cost + tv_cost * tv_increase) + (phone_cost + phone_cost * phone_increase)

/-- Theorem stating the total amount received from the auction -/
theorem auction_result : 
  auction_total 500 400 (2/5) (40/100) = 1260 := by sorry

end auction_result_l558_55876


namespace isabel_country_albums_l558_55823

/-- Represents the number of songs in each album -/
def songs_per_album : ℕ := 8

/-- Represents the number of pop albums bought -/
def pop_albums : ℕ := 5

/-- Represents the total number of songs bought -/
def total_songs : ℕ := 72

/-- Represents the number of country albums bought -/
def country_albums : ℕ := (total_songs - pop_albums * songs_per_album) / songs_per_album

theorem isabel_country_albums : country_albums = 4 := by
  sorry

end isabel_country_albums_l558_55823


namespace total_markers_count_l558_55848

/-- The number of red markers Connie has -/
def red_markers : ℕ := 2315

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 1028

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers

/-- Theorem stating that the total number of markers is 3343 -/
theorem total_markers_count : total_markers = 3343 := by
  sorry

end total_markers_count_l558_55848


namespace erdos_szekeres_l558_55856

theorem erdos_szekeres (n : ℕ) (seq : Fin (n^2 + 1) → ℝ) :
  (∃ (subseq : Fin (n + 1) → Fin (n^2 + 1)), Monotone (seq ∘ subseq)) ∨
  (∃ (subseq : Fin (n + 1) → Fin (n^2 + 1)), StrictAnti (seq ∘ subseq)) :=
sorry

end erdos_szekeres_l558_55856


namespace sum_of_decimals_l558_55864

theorem sum_of_decimals : 2.75 + 0.003 + 0.158 = 2.911 := by
  sorry

end sum_of_decimals_l558_55864


namespace complex_modulus_problem_l558_55811

def i : ℂ := Complex.I

theorem complex_modulus_problem (a : ℝ) (z : ℂ) 
  (h1 : z = (2 - a * i) / i) 
  (h2 : z.re = 0) : 
  Complex.abs z = 2 := by
  sorry

end complex_modulus_problem_l558_55811


namespace overtake_at_eight_hours_l558_55852

/-- Represents the chase between a pirate ship and a trading vessel -/
structure ChaseScenario where
  initial_distance : ℝ
  pirate_initial_speed : ℝ
  trading_initial_speed : ℝ
  damage_time : ℝ
  pirate_damaged_distance : ℝ
  trading_damaged_distance : ℝ

/-- The time at which the pirate ship overtakes the trading vessel -/
def overtake_time (scenario : ChaseScenario) : ℝ :=
  sorry

/-- The specific chase scenario described in the problem -/
def given_scenario : ChaseScenario :=
  { initial_distance := 15
  , pirate_initial_speed := 14
  , trading_initial_speed := 10
  , damage_time := 3
  , pirate_damaged_distance := 18
  , trading_damaged_distance := 17 }

/-- Theorem stating that the overtake time for the given scenario is 8 hours -/
theorem overtake_at_eight_hours :
  overtake_time given_scenario = 8 :=
sorry

end overtake_at_eight_hours_l558_55852


namespace hiker_total_distance_l558_55870

-- Define the hiker's walking parameters
def day1_distance : ℕ := 18
def day1_speed : ℕ := 3
def day2_speed_increase : ℕ := 1
def day3_speed : ℕ := 5
def day3_hours : ℕ := 6

-- Theorem to prove
theorem hiker_total_distance :
  let day1_hours : ℕ := day1_distance / day1_speed
  let day2_hours : ℕ := day1_hours - 1
  let day2_speed : ℕ := day1_speed + day2_speed_increase
  let day2_distance : ℕ := day2_speed * day2_hours
  let day3_distance : ℕ := day3_speed * day3_hours
  day1_distance + day2_distance + day3_distance = 68 := by
  sorry


end hiker_total_distance_l558_55870


namespace trolley_passengers_third_stop_l558_55889

/-- Proves the number of people who got off on the third stop of a trolley ride --/
theorem trolley_passengers_third_stop 
  (initial_passengers : ℕ) 
  (second_stop_off : ℕ) 
  (second_stop_on_multiplier : ℕ) 
  (third_stop_on : ℕ) 
  (final_passengers : ℕ) 
  (h1 : initial_passengers = 10)
  (h2 : second_stop_off = 3)
  (h3 : second_stop_on_multiplier = 2)
  (h4 : third_stop_on = 2)
  (h5 : final_passengers = 12) :
  initial_passengers - second_stop_off + second_stop_on_multiplier * initial_passengers - 
  (initial_passengers - second_stop_off + second_stop_on_multiplier * initial_passengers + third_stop_on - final_passengers) = 17 := by
  sorry


end trolley_passengers_third_stop_l558_55889


namespace division_remainder_549547_by_7_l558_55831

theorem division_remainder_549547_by_7 : 
  549547 % 7 = 5 := by sorry

end division_remainder_549547_by_7_l558_55831


namespace exponent_multiplication_l558_55882

theorem exponent_multiplication (a : ℝ) : a^2 * a^6 = a^8 := by
  sorry

end exponent_multiplication_l558_55882


namespace probability_between_R_and_S_l558_55819

/-- Given points P, Q, R, and S on a line segment PQ where PQ = 4PS and PQ = 8QR,
    the probability that a randomly selected point on PQ is between R and S is 5/8. -/
theorem probability_between_R_and_S (P Q R S : ℝ) : 
  P < R ∧ R < S ∧ S < Q →  -- Points are in order on the line
  Q - P = 4 * (S - P) →    -- PQ = 4PS
  Q - P = 8 * (Q - R) →    -- PQ = 8QR
  (S - R) / (Q - P) = 5/8  -- Probability is length of RS divided by length of PQ
  := by sorry

end probability_between_R_and_S_l558_55819


namespace train_speed_calculation_l558_55825

/-- Proves that a train with given length, crossing a bridge of given length in a given time, has a specific speed in km/hr -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 140 ∧ 
  bridge_length = 132 ∧ 
  crossing_time = 13.598912087033037 →
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
  sorry

end train_speed_calculation_l558_55825


namespace polygon_sides_l558_55816

theorem polygon_sides (sum_interior_angles : ℝ) :
  sum_interior_angles = 1620 →
  ∃ n : ℕ, n = 11 ∧ sum_interior_angles = 180 * (n - 2) :=
by
  sorry

end polygon_sides_l558_55816


namespace max_d_is_two_l558_55872

def a (n : ℕ) : ℕ := 101 + (n + 1)^2 + 3*n

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_is_two : 
  (∃ (n : ℕ), d n = 2) ∧ (∀ (n : ℕ), d n ≤ 2) :=
sorry

end max_d_is_two_l558_55872


namespace total_chicken_pieces_l558_55887

-- Define the number of chicken pieces per order type
def chicken_pasta_pieces : ℕ := 2
def barbecue_chicken_pieces : ℕ := 3
def fried_chicken_dinner_pieces : ℕ := 8

-- Define the number of orders for each type
def fried_chicken_dinner_orders : ℕ := 2
def chicken_pasta_orders : ℕ := 6
def barbecue_chicken_orders : ℕ := 3

-- Theorem stating the total number of chicken pieces needed
theorem total_chicken_pieces :
  fried_chicken_dinner_orders * fried_chicken_dinner_pieces +
  chicken_pasta_orders * chicken_pasta_pieces +
  barbecue_chicken_orders * barbecue_chicken_pieces = 37 :=
by sorry

end total_chicken_pieces_l558_55887


namespace linear_function_through_origin_l558_55858

/-- A linear function y = nx + (n^2 - 7) passing through (0, 2) with negative slope has n = -3 -/
theorem linear_function_through_origin (n : ℝ) : 
  (2 = n^2 - 7) →  -- The graph passes through (0, 2)
  (n < 0) →        -- y decreases as x increases (negative slope)
  n = -3 := by
sorry

end linear_function_through_origin_l558_55858


namespace rounded_avg_mb_per_minute_is_one_l558_55834

/-- Represents the number of days of music in the library -/
def days_of_music : ℕ := 15

/-- Represents the total disk space occupied by the library in megabytes -/
def total_disk_space : ℕ := 20000

/-- Calculates the total number of minutes of music in the library -/
def total_minutes : ℕ := days_of_music * 24 * 60

/-- Calculates the average megabytes per minute of music -/
def avg_mb_per_minute : ℚ := total_disk_space / total_minutes

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- Theorem stating that the rounded average megabytes per minute is 1 -/
theorem rounded_avg_mb_per_minute_is_one :
  round_to_nearest avg_mb_per_minute = 1 := by sorry

end rounded_avg_mb_per_minute_is_one_l558_55834


namespace triangle_shape_l558_55820

theorem triangle_shape (A : Real) (h1 : 0 < A ∧ A < π) 
  (h2 : Real.sin A + Real.cos A = 7/12) : 
  ∃ (B C : Real), 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π ∧ π/2 < A := by
  sorry

end triangle_shape_l558_55820


namespace shop_prices_l558_55893

theorem shop_prices (x y : ℝ) 
  (sum_condition : x + y = 5)
  (retail_condition : 3 * (x + 1) + 2 * (2 * y - 1) = 19) :
  x = 2 ∧ y = 3 := by
  sorry

end shop_prices_l558_55893


namespace collinear_vectors_m_value_l558_55813

/-- Given two non-collinear vectors in a plane, prove that m = -2/3 when the given conditions are met. -/
theorem collinear_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a ≠ 0 ∧ b ≠ 0 ∧ ¬ ∃ (k : ℝ), a = k • b →  -- a and b are non-collinear
  ∃ (A B C : ℝ × ℝ),
    B - A = 2 • a + m • b ∧  -- AB = 2a + mb
    C - B = 3 • a - b ∧  -- BC = 3a - b
    ∃ (t : ℝ), C - A = t • (B - A) →  -- A, B, C are collinear
  m = -2/3 := by
sorry

end collinear_vectors_m_value_l558_55813


namespace crate_missing_dimension_l558_55879

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  width : Real
  length : Real
  height : Real

/-- Represents a cylindrical tank -/
structure CylindricalTank where
  radius : Real
  height : Real

def fits_in_crate (tank : CylindricalTank) (crate : CrateDimensions) : Prop :=
  2 * tank.radius ≤ min crate.width crate.length ∧
  tank.height ≤ crate.height

def is_max_volume (tank : CylindricalTank) (crate : CrateDimensions) : Prop :=
  fits_in_crate tank crate ∧
  ∀ other_tank : CylindricalTank,
    fits_in_crate other_tank crate →
    tank.radius * tank.radius * tank.height ≥ other_tank.radius * other_tank.radius * other_tank.height

theorem crate_missing_dimension
  (crate : CrateDimensions)
  (h_width : crate.width = 8)
  (h_length : crate.length = 12)
  (tank : CylindricalTank)
  (h_radius : tank.radius = 6)
  (h_max_volume : is_max_volume tank crate) :
  crate.height = 12 :=
sorry

end crate_missing_dimension_l558_55879


namespace jacob_age_l558_55862

/-- Given Rehana's current age, her age relative to Phoebe's in 5 years, and Jacob's age relative to Phoebe's, prove Jacob's current age. -/
theorem jacob_age (rehana_age : ℕ) (phoebe_age : ℕ) (jacob_age : ℕ) : 
  rehana_age = 25 →
  rehana_age + 5 = 3 * (phoebe_age + 5) →
  jacob_age = 3 * phoebe_age / 5 →
  jacob_age = 3 :=
by sorry

end jacob_age_l558_55862


namespace cow_count_is_83_l558_55800

/-- Calculates the final number of cows given the initial count and changes in the herd -/
def final_cow_count (initial : ℕ) (died : ℕ) (sold : ℕ) (increased : ℕ) (bought : ℕ) (gifted : ℕ) : ℕ :=
  initial - died - sold + increased + bought + gifted

/-- Theorem stating that given the specific changes in the herd, the final count is 83 -/
theorem cow_count_is_83 :
  final_cow_count 39 25 6 24 43 8 = 83 := by
  sorry

end cow_count_is_83_l558_55800


namespace only_zero_satisfies_equations_l558_55853

theorem only_zero_satisfies_equations (x y a : ℝ) 
  (eq1 : x + y = a) 
  (eq2 : x^3 + y^3 = a) 
  (eq3 : x^5 + y^5 = a) : 
  a = 0 :=
sorry

end only_zero_satisfies_equations_l558_55853


namespace fraction_value_l558_55866

theorem fraction_value : (2200 - 2096)^2 / 121 = 89 := by sorry

end fraction_value_l558_55866


namespace range_of_a_l558_55892

-- Define set A
def A : Set ℝ := {x | x * (4 - x) ≥ 3}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = A → a < 1 := by
  sorry

-- Note: The proof is omitted as per the instructions

end range_of_a_l558_55892


namespace bipartite_graph_completion_l558_55857

/-- A bipartite graph with n vertices in each partition -/
structure BipartiteGraph (n : ℕ) :=
  (A B : Finset (Fin n))
  (edges : Finset (Fin n × Fin n))
  (bipartite : ∀ (e : Fin n × Fin n), e ∈ edges → (e.1 ∈ A ∧ e.2 ∈ B) ∨ (e.1 ∈ B ∧ e.2 ∈ A))

/-- The degree of a vertex in a bipartite graph -/
def degree (G : BipartiteGraph n) (v : Fin n) : ℕ :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- The theorem statement -/
theorem bipartite_graph_completion
  (n d : ℕ) (h_pos : 0 < n ∧ 0 < d) (h_bound : d < n / 2)
  (G : BipartiteGraph n)
  (h_degree : ∀ v, degree G v ≤ d) :
  ∃ G' : BipartiteGraph n,
    (∀ e ∈ G.edges, e ∈ G'.edges) ∧
    (∀ v, degree G' v = 2 * d) :=
sorry

end bipartite_graph_completion_l558_55857


namespace fibonacci_unique_triple_l558_55886

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def satisfies_conditions (a b c : ℕ) : Prop :=
  b < a ∧ c < a ∧ ∀ n, (fibonacci n - n * b * c^n) % a = 0

theorem fibonacci_unique_triple :
  ∃! (triple : ℕ × ℕ × ℕ), 
    let (a, b, c) := triple
    satisfies_conditions a b c ∧ a = 5 ∧ b = 2 ∧ c = 3 :=
sorry

end fibonacci_unique_triple_l558_55886


namespace quadratic_sum_l558_55895

/-- Given a quadratic function g(x) = dx^2 + ex + f, 
    if g(0) = 8 and g(1) = 5, then d + e + 2f = 13 -/
theorem quadratic_sum (d e f : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ d * x^2 + e * x + f
  (g 0 = 8) → (g 1 = 5) → d + e + 2 * f = 13 := by
  sorry

end quadratic_sum_l558_55895


namespace max_sum_on_circle_l558_55865

theorem max_sum_on_circle (x y : ℝ) (h : (x - 3)^2 + (y - 3)^2 = 8) :
  ∃ (max : ℝ), ∀ (a b : ℝ), (a - 3)^2 + (b - 3)^2 = 8 → a + b ≤ max ∧ ∃ (u v : ℝ), (u - 3)^2 + (v - 3)^2 = 8 ∧ u + v = max :=
by
  sorry

end max_sum_on_circle_l558_55865


namespace locus_of_tangent_circles_l558_55837

/-- The equation of circle C1 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The equation of circle C3 -/
def C3 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

/-- A circle is externally tangent to C1 and internally tangent to C3 -/
def is_tangent_to_C1_C3 (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (5 - r)^2)

/-- The locus equation -/
def locus_equation (a b : ℝ) : Prop :=
  40 * a^2 + 49 * b^2 - 48 * a - 64 = 0

theorem locus_of_tangent_circles :
  ∀ a b : ℝ, (∃ r : ℝ, is_tangent_to_C1_C3 a b r) ↔ locus_equation a b :=
sorry

end locus_of_tangent_circles_l558_55837


namespace colored_disk_overlap_l558_55847

/-- Represents a disk with colored sectors -/
structure ColoredDisk :=
  (total_sectors : ℕ)
  (colored_sectors : ℕ)
  (h_total : total_sectors > 0)
  (h_colored : colored_sectors ≤ total_sectors)

/-- Counts the number of positions with at most k overlapping colored sectors -/
def count_low_overlap_positions (d1 d2 : ColoredDisk) (k : ℕ) : ℕ :=
  sorry

theorem colored_disk_overlap (d1 d2 : ColoredDisk) 
  (h1 : d1.total_sectors = 1985) (h2 : d2.total_sectors = 1985)
  (h3 : d1.colored_sectors = 200) (h4 : d2.colored_sectors = 200) :
  count_low_overlap_positions d1 d2 20 ≥ 80 := by
  sorry

end colored_disk_overlap_l558_55847


namespace inequality_solution_set_f_less_than_one_l558_55894

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem 1: Solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 2 - |x + 1|} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2/3} := by sorry

-- Theorem 2: Proof that f(x) < 1 under given conditions
theorem f_less_than_one (x y : ℝ) (h1 : |x - y - 1| ≤ 1/3) (h2 : |2*y + 1| ≤ 1/6) :
  f x < 1 := by sorry

end inequality_solution_set_f_less_than_one_l558_55894


namespace ancient_pi_approximation_l558_55873

theorem ancient_pi_approximation (V : ℝ) (r : ℝ) (d : ℝ) :
  V = (4 / 3) * Real.pi * r^3 →
  d = (16 / 9 * V)^(1/3) →
  (6 * 9) / 16 = 3.375 :=
by sorry

end ancient_pi_approximation_l558_55873


namespace function_inequality_l558_55818

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x + (1 + a) / x

theorem function_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f a x₀ ≤ 0) →
  (a ≥ (Real.exp 2 + 1) / (Real.exp 1 - 1) ∨ a ≤ -2) :=
by sorry

end function_inequality_l558_55818


namespace vector_at_t_5_l558_55805

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  /-- The vector on the line at parameter t -/
  vector_at : ℝ → ℝ × ℝ

/-- Theorem: Given a parameterized line with specific points, the vector at t=5 is (10, -11) -/
theorem vector_at_t_5 
  (line : ParameterizedLine) 
  (h1 : line.vector_at 1 = (2, 5)) 
  (h4 : line.vector_at 4 = (8, -7)) : 
  line.vector_at 5 = (10, -11) := by
sorry


end vector_at_t_5_l558_55805


namespace distance_between_points_l558_55812

/-- The distance between points (0, 12) and (5, 6) is √61 -/
theorem distance_between_points : Real.sqrt 61 = Real.sqrt ((5 - 0)^2 + (6 - 12)^2) := by
  sorry

end distance_between_points_l558_55812


namespace complex_equation_solution_l558_55807

theorem complex_equation_solution (z : ℂ) : (1 + 3*I)*z = I - 3 → z = I := by
  sorry

end complex_equation_solution_l558_55807


namespace exponent_property_l558_55835

theorem exponent_property (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := by
  sorry

end exponent_property_l558_55835


namespace trig_identity_l558_55871

theorem trig_identity : 
  (2 * Real.sin (46 * π / 180) - Real.sqrt 3 * Real.cos (74 * π / 180)) / Real.cos (16 * π / 180) = 1 := by
  sorry

end trig_identity_l558_55871


namespace sum_of_x_and_y_l558_55839

theorem sum_of_x_and_y (x y : ℝ) (hxy : x ≠ y)
  (det1 : Matrix.det ![![2, 5, 10], ![4, x, y], ![4, y, x]] = 0)
  (det2 : Matrix.det ![![x, y], ![y, x]] = 16) :
  x + y = 30 := by
sorry

end sum_of_x_and_y_l558_55839


namespace statue_selling_price_l558_55832

/-- The selling price of a statue given its original cost and profit percentage -/
def selling_price (original_cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  original_cost * (1 + profit_percentage)

/-- Theorem: The selling price of the statue is $660 -/
theorem statue_selling_price :
  let original_cost : ℝ := 550
  let profit_percentage : ℝ := 0.20
  selling_price original_cost profit_percentage = 660 := by
  sorry

end statue_selling_price_l558_55832


namespace noahs_yearly_call_cost_l558_55829

/-- The total cost of Noah's calls to his Grammy for a year -/
def total_cost (weeks_per_year : ℕ) (minutes_per_call : ℕ) (cost_per_minute : ℚ) : ℚ :=
  (weeks_per_year * minutes_per_call : ℕ) * cost_per_minute

/-- Theorem: Noah's yearly call cost to Grammy is $78 -/
theorem noahs_yearly_call_cost :
  total_cost 52 30 (5/100) = 78 := by
  sorry

end noahs_yearly_call_cost_l558_55829


namespace defective_firecracker_fraction_l558_55841

theorem defective_firecracker_fraction :
  ∀ (initial_firecrackers confiscated_firecrackers good_firecrackers_set_off : ℕ),
    initial_firecrackers = 48 →
    confiscated_firecrackers = 12 →
    good_firecrackers_set_off = 15 →
    good_firecrackers_set_off * 2 = initial_firecrackers - confiscated_firecrackers - 
      (initial_firecrackers - confiscated_firecrackers - good_firecrackers_set_off * 2) →
    (initial_firecrackers - confiscated_firecrackers - good_firecrackers_set_off * 2) / 
    (initial_firecrackers - confiscated_firecrackers) = 1 / 6 :=
by sorry

end defective_firecracker_fraction_l558_55841


namespace a_greater_than_b_squared_l558_55899

theorem a_greater_than_b_squared {a b : ℝ} (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 := by
  sorry

end a_greater_than_b_squared_l558_55899


namespace square_ratio_proof_l558_55827

theorem square_ratio_proof (area_ratio : ℚ) (a b c : ℕ) :
  area_ratio = 75 / 128 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  a = 5 ∧ b = 6 ∧ c = 16 →
  a + b + c = 27 :=
by sorry

end square_ratio_proof_l558_55827


namespace linear_system_sum_a_d_l558_55888

theorem linear_system_sum_a_d :
  ∀ (a b c d e : ℝ),
    a + b = 14 →
    b + c = 9 →
    c + d = 3 →
    d + e = 6 →
    a - 2 * e = 1 →
    a + d = 8 :=
by
  sorry

end linear_system_sum_a_d_l558_55888


namespace vector_equation_proof_l558_55850

/-- Prove that the given values of a and b satisfy the vector equation -/
theorem vector_equation_proof :
  let a : ℚ := -3/14
  let b : ℚ := 107/14
  let v1 : Fin 2 → ℚ := ![3, 4]
  let v2 : Fin 2 → ℚ := ![1, 6]
  let result : Fin 2 → ℚ := ![7, 45]
  (a • v1 + b • v2) = result :=
by sorry

end vector_equation_proof_l558_55850


namespace simplify_complex_fraction_l558_55861

theorem simplify_complex_fraction :
  1 / ((1 / (Real.sqrt 3 + 1)) + (2 / (Real.sqrt 5 - 1))) = 
  (Real.sqrt 3 + 2 * Real.sqrt 5 - 1) / (2 + 4 * Real.sqrt 5) := by
sorry

end simplify_complex_fraction_l558_55861


namespace volleyball_lineup_combinations_l558_55890

-- Define the total number of team members
def total_members : ℕ := 18

-- Define the number of players in the starting lineup
def lineup_size : ℕ := 8

-- Define the number of interchangeable positions
def interchangeable_positions : ℕ := 6

-- Theorem statement
theorem volleyball_lineup_combinations :
  (total_members) *
  (total_members - 1) *
  (Nat.choose (total_members - 2) interchangeable_positions) =
  2448272 := by sorry

end volleyball_lineup_combinations_l558_55890


namespace square_area_from_perimeter_l558_55824

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) :
  perimeter = 40 →
  area = (perimeter / 4)^2 →
  area = 100 :=
by sorry

end square_area_from_perimeter_l558_55824


namespace geometric_sequence_first_term_l558_55867

/-- A geometric sequence with positive common ratio -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_first_term
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_relation : a 3 * a 9 = 2 * (a 5)^2)
  (h_a2 : a 2 = 2) :
  a 1 = Real.sqrt 2 := by
sorry

end geometric_sequence_first_term_l558_55867


namespace smallest_sum_of_sequence_l558_55855

theorem smallest_sum_of_sequence (X Y Z W : ℕ) : 
  X > 0 → Y > 0 → Z > 0 → W > 0 →
  (∃ d : ℤ, Z - Y = Y - X ∧ Z - Y = d) →
  (∃ r : ℚ, Z = r * Y ∧ W = r * Z) →
  Z = (9 : ℚ) / 5 * Y →
  (∀ a b c d : ℕ, a > 0 → b > 0 → c > 0 → d > 0 →
    (∃ d' : ℤ, c - b = b - a ∧ c - b = d') →
    (∃ r' : ℚ, c = r' * b ∧ d = r' * c) →
    c = (9 : ℚ) / 5 * b →
    X + Y + Z + W ≤ a + b + c + d) →
  X + Y + Z + W = 156 :=
by sorry

end smallest_sum_of_sequence_l558_55855


namespace constant_term_expansion_l558_55875

/-- The constant term in the expansion of (3x^2 + 2/x)^8 -/
def constant_term : ℕ :=
  (Nat.choose 8 4) * (3^4) * (2^4)

/-- Theorem stating that the constant term in the expansion of (3x^2 + 2/x)^8 is 90720 -/
theorem constant_term_expansion :
  constant_term = 90720 := by
  sorry

end constant_term_expansion_l558_55875


namespace domino_probability_and_attempts_l558_55877

/-- The total number of domino tiles -/
def total_tiles : ℕ := 45

/-- The number of tiles drawn -/
def drawn_tiles : ℕ := 3

/-- The probability of the event occurring in a single attempt -/
def event_probability : ℚ := 54 / 473

/-- The minimum probability we want to achieve -/
def target_probability : ℝ := 0.9

/-- The minimum number of attempts needed -/
def min_attempts : ℕ := 19

/-- Theorem stating the probability of the event and the minimum number of attempts needed -/
theorem domino_probability_and_attempts :
  (event_probability : ℝ) = 54 / 473 ∧
  (1 - (1 - event_probability) ^ min_attempts : ℝ) ≥ target_probability ∧
  ∀ n : ℕ, n < min_attempts → (1 - (1 - event_probability) ^ n : ℝ) < target_probability :=
sorry


end domino_probability_and_attempts_l558_55877


namespace unique_triple_solution_l558_55843

theorem unique_triple_solution : 
  ∃! (p q r : ℕ), 
    p > 0 ∧ q > 0 ∧ r > 0 ∧
    Nat.Prime p ∧ Nat.Prime q ∧
    (r^2 - 5*q^2) / (p^2 - 1) = 2 ∧
    p = 3 ∧ q = 2 ∧ r = 6 := by
  sorry

end unique_triple_solution_l558_55843


namespace seven_digit_divisible_by_nine_l558_55810

theorem seven_digit_divisible_by_nine (m : ℕ) : 
  m < 10 →
  (746 * 1000000 + m * 10000 + 813) % 9 = 0 →
  m = 7 := by
sorry

end seven_digit_divisible_by_nine_l558_55810


namespace all_triangles_isosceles_l558_55826

-- Define a point on the grid
structure GridPoint where
  x : Int
  y : Int

-- Define a triangle on the grid
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : GridPoint) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : GridTriangle) : Prop :=
  let d12 := squaredDistance t.p1 t.p2
  let d13 := squaredDistance t.p1 t.p3
  let d23 := squaredDistance t.p2 t.p3
  d12 = d13 ∨ d12 = d23 ∨ d13 = d23

-- Define the four triangles
def triangle1 : GridTriangle := ⟨⟨2, 2⟩, ⟨5, 2⟩, ⟨2, 5⟩⟩
def triangle2 : GridTriangle := ⟨⟨1, 1⟩, ⟨4, 1⟩, ⟨1, 4⟩⟩
def triangle3 : GridTriangle := ⟨⟨3, 3⟩, ⟨6, 3⟩, ⟨6, 6⟩⟩
def triangle4 : GridTriangle := ⟨⟨0, 0⟩, ⟨3, 0⟩, ⟨3, 3⟩⟩

-- Theorem: All four triangles are isosceles
theorem all_triangles_isosceles :
  isIsosceles triangle1 ∧
  isIsosceles triangle2 ∧
  isIsosceles triangle3 ∧
  isIsosceles triangle4 := by
  sorry

end all_triangles_isosceles_l558_55826


namespace distance_between_centers_l558_55801

/-- An isosceles triangle with its circumcircle and inscribed circle -/
structure IsoscelesTriangleWithCircles where
  /-- The radius of the circumcircle -/
  R : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- R is positive -/
  R_pos : R > 0
  /-- r is positive -/
  r_pos : r > 0
  /-- r is less than R (as the inscribed circle must fit inside the circumcircle) -/
  r_lt_R : r < R

/-- The distance between the centers of the circumcircle and inscribed circle
    of an isosceles triangle is √(R(R - 2r)) -/
theorem distance_between_centers (t : IsoscelesTriangleWithCircles) :
  ∃ d : ℝ, d = Real.sqrt (t.R * (t.R - 2 * t.r)) ∧ d > 0 := by
  sorry

end distance_between_centers_l558_55801


namespace largest_divisor_fifth_largest_divisor_l558_55838

def n : ℕ := 1516000000

-- Define a function to get the kth largest divisor
def kthLargestDivisor (k : ℕ) : ℕ := sorry

-- The largest divisor of n is itself
theorem largest_divisor : kthLargestDivisor 1 = n := sorry

-- The fifth-largest divisor of n is 94,750,000
theorem fifth_largest_divisor : kthLargestDivisor 5 = 94750000 := sorry

end largest_divisor_fifth_largest_divisor_l558_55838


namespace apple_seed_average_l558_55880

theorem apple_seed_average (total_seeds : ℕ) (pear_avg : ℕ) (grape_avg : ℕ)
  (apple_count : ℕ) (pear_count : ℕ) (grape_count : ℕ) (seeds_needed : ℕ)
  (h1 : total_seeds = 60)
  (h2 : pear_avg = 2)
  (h3 : grape_avg = 3)
  (h4 : apple_count = 4)
  (h5 : pear_count = 3)
  (h6 : grape_count = 9)
  (h7 : seeds_needed = 3) :
  ∃ (apple_avg : ℕ), apple_avg = 6 ∧
    apple_count * apple_avg + pear_count * pear_avg + grape_count * grape_avg
    = total_seeds - seeds_needed :=
by sorry

end apple_seed_average_l558_55880


namespace missing_number_proof_l558_55868

theorem missing_number_proof (x : ℝ) : 
  (x + 42 + 78 + 104) / 4 = 62 ∧ 
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 →
  x = 74 := by
sorry

end missing_number_proof_l558_55868


namespace thousand_to_100_equals_googol_cubed_l558_55846

-- Define googol
def googol : ℕ := 10^100

-- Theorem statement
theorem thousand_to_100_equals_googol_cubed :
  1000^100 = googol^3 := by
  sorry

end thousand_to_100_equals_googol_cubed_l558_55846


namespace total_tickets_sold_l558_55833

/-- Represents the ticket sales for a theater performance --/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- The pricing and sales data for the theater --/
def theaterData : TicketSales → Prop := fun ts =>
  12 * ts.orchestra + 8 * ts.balcony = 3320 ∧
  ts.balcony = ts.orchestra + 240

/-- Theorem stating that the total number of tickets sold is 380 --/
theorem total_tickets_sold (ts : TicketSales) (h : theaterData ts) : 
  ts.orchestra + ts.balcony = 380 := by
  sorry

#check total_tickets_sold

end total_tickets_sold_l558_55833


namespace longest_frog_vs_shortest_grasshopper_l558_55815

def frog_jumps : List ℕ := [39, 45, 50]
def grasshopper_jumps : List ℕ := [17, 22, 28, 31]

theorem longest_frog_vs_shortest_grasshopper :
  (List.maximum frog_jumps).get! - (List.minimum grasshopper_jumps).get! = 33 := by
  sorry

end longest_frog_vs_shortest_grasshopper_l558_55815


namespace seconds_in_hours_l558_55883

theorem seconds_in_hours : 
  (∀ (hours : ℝ), hours * 60 * 60 = hours * 3600) →
  3.5 * 3600 = 12600 := by sorry

end seconds_in_hours_l558_55883
