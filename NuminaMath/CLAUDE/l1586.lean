import Mathlib

namespace NUMINAMATH_CALUDE_salary_expenses_l1586_158648

theorem salary_expenses (S : ℝ) 
  (h1 : S - (2/5)*S - (3/10)*S - (1/8)*S = 1400) :
  (3/10)*S + (1/8)*S = 3400 :=
by sorry

end NUMINAMATH_CALUDE_salary_expenses_l1586_158648


namespace NUMINAMATH_CALUDE_kelly_games_l1586_158627

theorem kelly_games (games_given_away : ℕ) (games_left : ℕ) : games_given_away = 91 → games_left = 92 → games_given_away + games_left = 183 :=
by
  sorry

end NUMINAMATH_CALUDE_kelly_games_l1586_158627


namespace NUMINAMATH_CALUDE_lip_gloss_coverage_l1586_158649

theorem lip_gloss_coverage 
  (num_tubs : ℕ) 
  (tubes_per_tub : ℕ) 
  (total_people : ℕ) 
  (h1 : num_tubs = 6) 
  (h2 : tubes_per_tub = 2) 
  (h3 : total_people = 36) : 
  total_people / (num_tubs * tubes_per_tub) = 3 := by
sorry

end NUMINAMATH_CALUDE_lip_gloss_coverage_l1586_158649


namespace NUMINAMATH_CALUDE_envelope_area_l1586_158698

/-- The area of a rectangular envelope with width and length both equal to 4 inches is 16 square inches. -/
theorem envelope_area (width : ℝ) (length : ℝ) (h_width : width = 4) (h_length : length = 4) :
  width * length = 16 := by
  sorry

end NUMINAMATH_CALUDE_envelope_area_l1586_158698


namespace NUMINAMATH_CALUDE_solve_for_a_l1586_158672

theorem solve_for_a (a : ℝ) : (3 * (-1) + 2 * a + 1 = 2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1586_158672


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1586_158626

theorem sqrt_sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (sum_eq : a + b = c + d) (ineq : a < c ∧ c ≤ d ∧ d < b) :
  Real.sqrt a + Real.sqrt b < Real.sqrt c + Real.sqrt d := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1586_158626


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minus_product_l1586_158685

theorem quadratic_roots_sum_squares_minus_product (m n : ℝ) : 
  m^2 - 5*m - 2 = 0 → n^2 - 5*n - 2 = 0 → m^2 + n^2 - m*n = 31 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minus_product_l1586_158685


namespace NUMINAMATH_CALUDE_power_multiplication_l1586_158659

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1586_158659


namespace NUMINAMATH_CALUDE_paul_weekly_spending_l1586_158606

theorem paul_weekly_spending 
  (lawn_money : ℕ) 
  (weed_money : ℕ) 
  (weeks : ℕ) 
  (h1 : lawn_money = 68) 
  (h2 : weed_money = 13) 
  (h3 : weeks = 9) : 
  (lawn_money + weed_money) / weeks = 9 := by
sorry

end NUMINAMATH_CALUDE_paul_weekly_spending_l1586_158606


namespace NUMINAMATH_CALUDE_price_adjustment_l1586_158692

theorem price_adjustment (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) * 0.9 = 0.77 * P →
  x = 38 := by
sorry

end NUMINAMATH_CALUDE_price_adjustment_l1586_158692


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l1586_158621

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersecting_lines_k_value 
  (x y : ℝ) -- The coordinates of the intersection point
  (h1 : y = 3 * x + 6) -- First line equation
  (h2 : y = -4 * x - 20) -- Second line equation
  (h3 : y = 2 * x + k) -- Third line equation
  : k = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l1586_158621


namespace NUMINAMATH_CALUDE_june_upload_ratio_l1586_158664

/-- Represents the video upload scenario for a YouTuber in June --/
structure VideoUpload where
  totalDays : Nat
  halfMonth : Nat
  firstHalfDailyHours : Nat
  totalHours : Nat

/-- Calculates the ratio of daily video hours in the second half to the first half of the month --/
def uploadRatio (v : VideoUpload) : Rat :=
  let firstHalfTotal := v.firstHalfDailyHours * v.halfMonth
  let secondHalfTotal := v.totalHours - firstHalfTotal
  let secondHalfDaily := secondHalfTotal / v.halfMonth
  secondHalfDaily / v.firstHalfDailyHours

/-- The main theorem stating the upload ratio for the given scenario --/
theorem june_upload_ratio (v : VideoUpload) 
    (h1 : v.totalDays = 30)
    (h2 : v.halfMonth = 15)
    (h3 : v.firstHalfDailyHours = 10)
    (h4 : v.totalHours = 450) :
  uploadRatio v = 2 := by
  sorry

#eval uploadRatio { totalDays := 30, halfMonth := 15, firstHalfDailyHours := 10, totalHours := 450 }

end NUMINAMATH_CALUDE_june_upload_ratio_l1586_158664


namespace NUMINAMATH_CALUDE_fifty_eighth_digit_of_one_seventeenth_l1586_158633

def decimal_representation (n : ℕ) : List ℕ := sorry

def is_periodic (l : List ℕ) : Prop := sorry

def nth_digit (l : List ℕ) (n : ℕ) : ℕ := sorry

theorem fifty_eighth_digit_of_one_seventeenth (h : is_periodic (decimal_representation 17)) :
  nth_digit (decimal_representation 17) 58 = 4 := by sorry

end NUMINAMATH_CALUDE_fifty_eighth_digit_of_one_seventeenth_l1586_158633


namespace NUMINAMATH_CALUDE_divisible_by_fifteen_l1586_158689

theorem divisible_by_fifteen (x : ℤ) : 
  (∃ k : ℤ, x^2 + 2*x + 6 = 15 * k) ↔ 
  (∃ t : ℤ, x = 15*t - 6 ∨ x = 15*t + 4) := by
sorry

end NUMINAMATH_CALUDE_divisible_by_fifteen_l1586_158689


namespace NUMINAMATH_CALUDE_multiply_divide_equation_l1586_158620

theorem multiply_divide_equation : ∃ x : ℝ, (3.242 * x) / 100 = 0.04863 ∧ abs (x - 1.5) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_equation_l1586_158620


namespace NUMINAMATH_CALUDE_charlie_spent_56250_l1586_158624

/-- The amount Charlie spent on acorns -/
def charlie_spent (alice_acorns bob_acorns charlie_acorns : ℕ) 
  (bob_total : ℚ) (alice_multiplier : ℕ) : ℚ :=
  let bob_price := bob_total / bob_acorns
  let alice_price := alice_multiplier * bob_price
  let average_price := (bob_price + alice_price) / 2
  charlie_acorns * average_price

/-- Theorem stating that Charlie spent $56,250 on acorns -/
theorem charlie_spent_56250 :
  charlie_spent 3600 2400 4500 6000 9 = 56250 := by
  sorry

end NUMINAMATH_CALUDE_charlie_spent_56250_l1586_158624


namespace NUMINAMATH_CALUDE_foreign_student_percentage_l1586_158603

theorem foreign_student_percentage 
  (total_students : ℕ) 
  (new_foreign_students : ℕ) 
  (future_foreign_students : ℕ) :
  total_students = 1800 →
  new_foreign_students = 200 →
  future_foreign_students = 740 →
  (↑(future_foreign_students - new_foreign_students) / ↑total_students : ℚ) = 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_foreign_student_percentage_l1586_158603


namespace NUMINAMATH_CALUDE_color_copies_comparison_l1586_158635

/-- The cost per color copy at print shop X -/
def cost_X : ℚ := 1.25

/-- The cost per color copy at print shop Y -/
def cost_Y : ℚ := 2.75

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℚ := 60

/-- The number of color copies being compared -/
def n : ℚ := 40

theorem color_copies_comparison :
  cost_Y * n = cost_X * n + additional_charge := by
  sorry

end NUMINAMATH_CALUDE_color_copies_comparison_l1586_158635


namespace NUMINAMATH_CALUDE_age_difference_l1586_158634

/-- Given that the sum of A's and B's ages is 18 years more than the sum of B's and C's ages,
    prove that A is 18 years older than C. -/
theorem age_difference (a b c : ℕ) (h : a + b = b + c + 18) : a = c + 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1586_158634


namespace NUMINAMATH_CALUDE_race_difference_l1586_158601

/-- In a race, given the total distance and the differences between runners,
    calculate the difference between two runners. -/
theorem race_difference (total_distance : ℕ) (a_beats_b b_beats_c a_beats_c : ℕ) :
  total_distance = 1000 →
  a_beats_b = 70 →
  a_beats_c = 163 →
  b_beats_c = 93 :=
by sorry

end NUMINAMATH_CALUDE_race_difference_l1586_158601


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l1586_158690

/-- Given a normal distribution with mean 51 and 3 standard deviations below the mean greater than 44,
    prove that the standard deviation is less than 2.33 -/
theorem normal_distribution_std_dev (σ : ℝ) (h1 : 51 - 3 * σ > 44) : σ < 2.33 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l1586_158690


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1586_158614

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2)  -- Pythagorean theorem (right triangle condition)
  (h2 : a = 3)            -- One non-hypotenuse side length
  (h3 : c = 5)            -- Hypotenuse length
  : b = 4 := by           -- Conclusion: other non-hypotenuse side length
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1586_158614


namespace NUMINAMATH_CALUDE_base_conversion_sum_l1586_158668

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [2, 5, 3]
def base1 : Nat := 8
def den1 : List Nat := [1, 3]
def base2 : Nat := 4
def num2 : List Nat := [1, 4, 4]
def base3 : Nat := 5
def den2 : List Nat := [2, 2]
def base4 : Nat := 3

-- State the theorem
theorem base_conversion_sum :
  (baseToDecimal num1 base1 : Rat) / (baseToDecimal den1 base2 : Rat) +
  (baseToDecimal num2 base3 : Rat) / (baseToDecimal den2 base4 : Rat) =
  30.125 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l1586_158668


namespace NUMINAMATH_CALUDE_right_triangle_area_l1586_158631

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) : (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1586_158631


namespace NUMINAMATH_CALUDE_garden_pool_perimeter_l1586_158678

/-- Represents a rectangular garden with square plots and a pool -/
structure Garden where
  plot_area : ℝ
  garden_length : ℝ
  num_plots : ℕ

/-- Calculates the perimeter of the pool in the garden -/
def pool_perimeter (g : Garden) : ℝ :=
  2 * g.garden_length

/-- Theorem stating the perimeter of the pool in the given garden configuration -/
theorem garden_pool_perimeter (g : Garden) 
  (h1 : g.plot_area = 20)
  (h2 : g.garden_length = 9)
  (h3 : g.num_plots = 4) : 
  pool_perimeter g = 18 := by
  sorry

#check garden_pool_perimeter

end NUMINAMATH_CALUDE_garden_pool_perimeter_l1586_158678


namespace NUMINAMATH_CALUDE_squirrels_and_nuts_l1586_158618

theorem squirrels_and_nuts (num_squirrels num_nuts : ℕ) 
  (h1 : num_squirrels = 4) 
  (h2 : num_nuts = 2) : 
  num_squirrels - num_nuts = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrels_and_nuts_l1586_158618


namespace NUMINAMATH_CALUDE_student_claim_impossible_l1586_158604

theorem student_claim_impossible (m n : ℤ) (hn : 0 < n) (hn_bound : n ≤ 100) :
  ¬ (167 / 1000 : ℚ) ≤ (m : ℚ) / (n : ℚ) ∧ (m : ℚ) / (n : ℚ) < (168 / 1000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_student_claim_impossible_l1586_158604


namespace NUMINAMATH_CALUDE_system_solution_and_simplification_l1586_158695

-- Define the system of equations
def system (x y m : ℝ) : Prop :=
  x + y = m + 2 ∧ 4 * x + 5 * y = 6 * m + 3

-- Define the positivity condition for x and y
def positive_solution (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

-- Theorem statement
theorem system_solution_and_simplification (m : ℝ) :
  (∃ x y, system x y m ∧ positive_solution x y) →
  (5/2 < m ∧ m < 7) ∧
  (|2*m - 5| - |m - 7| = 3*m - 12) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_and_simplification_l1586_158695


namespace NUMINAMATH_CALUDE_cross_ratio_equality_l1586_158655

theorem cross_ratio_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cross_ratio_equality_l1586_158655


namespace NUMINAMATH_CALUDE_smallest_n_for_2007_l1586_158675

theorem smallest_n_for_2007 : 
  (∃ (n : ℕ) (S : Finset ℕ), 
    n > 1 ∧ 
    S.card = n ∧ 
    (∀ x ∈ S, x > 0) ∧ 
    S.prod id = 2007 ∧ 
    S.sum id = 2007 ∧ 
    (∀ m : ℕ, m > 1 → 
      (∃ T : Finset ℕ, 
        T.card = m ∧ 
        (∀ x ∈ T, x > 0) ∧ 
        T.prod id = 2007 ∧ 
        T.sum id = 2007) → 
      n ≤ m)) ∧ 
  (∀ S : Finset ℕ, 
    S.card > 1 ∧ 
    (∀ x ∈ S, x > 0) ∧ 
    S.prod id = 2007 ∧ 
    S.sum id = 2007 → 
    S.card ≥ 1337) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_2007_l1586_158675


namespace NUMINAMATH_CALUDE_merchant_coin_problem_l1586_158605

/-- Represents the initial amount of gold coins each merchant has -/
structure MerchantCoins where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The amount of coins that equalizes Ierema and Yuliy when Foma gives to Ierema -/
def equalize_ierema_yuliy : ℕ := 70

/-- The amount of coins that equalizes Foma and Yuliy when Foma gives to Ierema -/
def equalize_foma_yuliy : ℕ := 40

/-- The solution amount: coins Foma should give Ierema to equalize their amounts -/
def solution_amount : ℕ := 55

/-- Theorem stating the solution to the merchant problem -/
theorem merchant_coin_problem (m : MerchantCoins) :
  m.foma - solution_amount = m.ierema + solution_amount ∧
  m.foma - equalize_ierema_yuliy = m.yuliy ∧
  m.ierema + equalize_ierema_yuliy = m.yuliy ∧
  m.foma - equalize_foma_yuliy = m.yuliy :=
sorry

end NUMINAMATH_CALUDE_merchant_coin_problem_l1586_158605


namespace NUMINAMATH_CALUDE_restaurant_group_composition_l1586_158667

theorem restaurant_group_composition (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) : 
  total_people = 11 → 
  adult_meal_cost = 8 → 
  total_cost = 72 → 
  ∃ (num_adults num_kids : ℕ), 
    num_adults + num_kids = total_people ∧ 
    num_adults * adult_meal_cost = total_cost ∧ 
    num_kids = 2 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_group_composition_l1586_158667


namespace NUMINAMATH_CALUDE_f_local_min_at_one_f_no_local_max_l1586_158615

/-- The function f(x) = (x^3 - 1)^2 + 1 -/
def f (x : ℝ) : ℝ := (x^3 - 1)^2 + 1

/-- f has a local minimum at x = 1 -/
theorem f_local_min_at_one : 
  ∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1 :=
sorry

/-- f has no local maximum points -/
theorem f_no_local_max : 
  ¬∃ a, ∃ δ > 0, ∀ x, |x - a| < δ → f x ≤ f a :=
sorry

end NUMINAMATH_CALUDE_f_local_min_at_one_f_no_local_max_l1586_158615


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_reversed_difference_l1586_158616

theorem smallest_prime_factor_of_reversed_difference (A B C : ℕ) 
  (h1 : A ≠ C) 
  (h2 : A ≤ 9) (h3 : B ≤ 9) (h4 : C ≤ 9) 
  (h5 : A ≠ 0) :
  let ABC := 100 * A + 10 * B + C
  let CBA := 100 * C + 10 * B + A
  ∃ (k : ℕ), ABC - CBA = 3 * k ∧ 
  ∀ (p : ℕ), p < 3 → ¬(∃ (m : ℕ), ABC - CBA = p * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_reversed_difference_l1586_158616


namespace NUMINAMATH_CALUDE_difference_largest_smallest_l1586_158688

/-- Represents a three-digit positive integer with no repeated digits -/
structure ThreeDigitNoRepeat where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : 
    1 ≤ hundreds ∧ hundreds ≤ 9 ∧
    0 ≤ tens ∧ tens ≤ 9 ∧
    0 ≤ ones ∧ ones ≤ 9 ∧
    hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

/-- Converts a ThreeDigitNoRepeat to its integer value -/
def ThreeDigitNoRepeat.toNat (n : ThreeDigitNoRepeat) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The largest three-digit positive integer with no repeated digits -/
def largest : ThreeDigitNoRepeat := {
  hundreds := 9
  tens := 8
  ones := 7
  is_valid := by sorry
}

/-- The smallest three-digit positive integer with no repeated digits -/
def smallest : ThreeDigitNoRepeat := {
  hundreds := 1
  tens := 0
  ones := 2
  is_valid := by sorry
}

/-- The main theorem -/
theorem difference_largest_smallest : 
  largest.toNat - smallest.toNat = 885 := by sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_l1586_158688


namespace NUMINAMATH_CALUDE_det_special_matrix_l1586_158670

/-- The determinant of the matrix [[y+2, y-1, y+1], [y+1, y+2, y-1], [y-1, y+1, y+2]] 
    is equal to 6y^2 + 23y + 14 for any real number y. -/
theorem det_special_matrix (y : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![y + 2, y - 1, y + 1],
    ![y + 1, y + 2, y - 1],
    ![y - 1, y + 1, y + 2]
  ]
  Matrix.det M = 6 * y^2 + 23 * y + 14 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l1586_158670


namespace NUMINAMATH_CALUDE_equation_solution_l1586_158682

theorem equation_solution (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1586_158682


namespace NUMINAMATH_CALUDE_max_stations_is_ten_exists_valid_network_with_ten_stations_max_stations_is_exactly_ten_l1586_158674

/-- Represents a surveillance network of stations -/
structure SurveillanceNetwork where
  stations : Finset ℕ
  connections : Finset (ℕ × ℕ)

/-- Checks if a station can communicate with all others directly or through one intermediary -/
def canCommunicateWithAll (net : SurveillanceNetwork) (s : ℕ) : Prop :=
  ∀ t ∈ net.stations, s ≠ t →
    (s, t) ∈ net.connections ∨ ∃ u ∈ net.stations, (s, u) ∈ net.connections ∧ (u, t) ∈ net.connections

/-- Checks if a station has at most three direct connections -/
def hasAtMostThreeConnections (net : SurveillanceNetwork) (s : ℕ) : Prop :=
  (net.connections.filter (λ p => p.1 = s ∨ p.2 = s)).card ≤ 3

/-- A valid surveillance network satisfies all conditions -/
def isValidNetwork (net : SurveillanceNetwork) : Prop :=
  ∀ s ∈ net.stations, canCommunicateWithAll net s ∧ hasAtMostThreeConnections net s

/-- The maximum number of stations in a valid surveillance network is 10 -/
theorem max_stations_is_ten :
  ∀ net : SurveillanceNetwork, isValidNetwork net → net.stations.card ≤ 10 :=
sorry

/-- There exists a valid surveillance network with 10 stations -/
theorem exists_valid_network_with_ten_stations :
  ∃ net : SurveillanceNetwork, isValidNetwork net ∧ net.stations.card = 10 :=
sorry

/-- The maximum number of stations in a valid surveillance network is exactly 10 -/
theorem max_stations_is_exactly_ten :
  (∃ net : SurveillanceNetwork, isValidNetwork net ∧ net.stations.card = 10) ∧
  (∀ net : SurveillanceNetwork, isValidNetwork net → net.stations.card ≤ 10) :=
sorry

end NUMINAMATH_CALUDE_max_stations_is_ten_exists_valid_network_with_ten_stations_max_stations_is_exactly_ten_l1586_158674


namespace NUMINAMATH_CALUDE_product_equals_difference_of_powers_l1586_158610

theorem product_equals_difference_of_powers : 
  (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) * 
  (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64) = 4^128 - 3^128 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_difference_of_powers_l1586_158610


namespace NUMINAMATH_CALUDE_distance_P_to_x_axis_l1586_158665

def point_to_x_axis_distance (p : ℝ × ℝ) : ℝ :=
  |p.2|

theorem distance_P_to_x_axis :
  let P : ℝ × ℝ := (-3, 2)
  point_to_x_axis_distance P = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_x_axis_l1586_158665


namespace NUMINAMATH_CALUDE_additional_money_needed_l1586_158607

def phone_cost : ℝ := 1300
def percentage_owned : ℝ := 40

theorem additional_money_needed : 
  phone_cost - (percentage_owned / 100) * phone_cost = 780 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l1586_158607


namespace NUMINAMATH_CALUDE_champion_determination_races_l1586_158661

/-- The number of races needed to determine a champion sprinter -/
def races_needed (initial_sprinters : ℕ) (sprinters_per_race : ℕ) (eliminated_per_race : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 50 races are needed for the given conditions -/
theorem champion_determination_races :
  races_needed 400 10 8 = 50 := by sorry

end NUMINAMATH_CALUDE_champion_determination_races_l1586_158661


namespace NUMINAMATH_CALUDE_parabola_equation_l1586_158613

/-- A parabola with vertex at the origin, axis of symmetry along a coordinate axis, 
    and passing through the point (√3, -2√3) has the equation y² = 4√3x or x² = -√3/2y -/
theorem parabola_equation (x y : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ ((y^2 = 2*p*x ∧ x = Real.sqrt 3 ∧ y = -2*Real.sqrt 3) ∨ 
                       (x^2 = -2*p*y ∧ x = Real.sqrt 3 ∧ y = -2*Real.sqrt 3))) → 
  (y^2 = 4*Real.sqrt 3*x ∨ x^2 = -(Real.sqrt 3/2)*y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1586_158613


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l1586_158653

theorem concentric_circles_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  π * b^2 - π * a^2 = 4 * (π * a^2) → a / b = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l1586_158653


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1586_158691

def z : ℂ := Complex.I + Complex.I^2

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1586_158691


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l1586_158637

/-- Represents the number of female students in a stratified sample -/
def female_in_sample (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ) : ℚ :=
  (female_students : ℚ) * (sample_size : ℚ) / (total_students : ℚ)

/-- Theorem: In a school with 2100 total students (900 female),
    a stratified sample of 70 students will contain 30 female students -/
theorem stratified_sample_theorem :
  female_in_sample 2100 900 70 = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l1586_158637


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1586_158651

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 3) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 59525 / 30964 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1586_158651


namespace NUMINAMATH_CALUDE_volunteer_selection_l1586_158662

theorem volunteer_selection (n_boys n_girls n_selected : ℕ) 
  (h_boys : n_boys = 4)
  (h_girls : n_girls = 3)
  (h_selected : n_selected = 3) : 
  (Nat.choose n_boys 2 * Nat.choose n_girls 1) + 
  (Nat.choose n_girls 2 * Nat.choose n_boys 1) = 30 :=
sorry

end NUMINAMATH_CALUDE_volunteer_selection_l1586_158662


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l1586_158646

theorem fraction_sum_equation (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 5 * b) / (b + 5 * a) = 2) : 
  a / b = 0.6 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l1586_158646


namespace NUMINAMATH_CALUDE_garden_playground_area_equality_l1586_158622

theorem garden_playground_area_equality (garden_width garden_length playground_width : ℝ) :
  garden_width = 8 →
  2 * (garden_width + garden_length) = 64 →
  garden_width * garden_length = 16 * playground_width :=
by
  sorry

end NUMINAMATH_CALUDE_garden_playground_area_equality_l1586_158622


namespace NUMINAMATH_CALUDE_mice_without_coins_l1586_158687

theorem mice_without_coins (total_mice : ℕ) (total_coins : ℕ) 
  (h1 : total_mice = 40)
  (h2 : total_coins = 40)
  (h3 : ∃ (y z : ℕ), 
    2 * 2 + 7 * y + 4 * z = total_coins ∧
    2 + y + z + (total_mice - (2 + y + z)) = total_mice) :
  total_mice - (2 + y + z) = 32 :=
by sorry

end NUMINAMATH_CALUDE_mice_without_coins_l1586_158687


namespace NUMINAMATH_CALUDE_cube_color_theorem_l1586_158656

theorem cube_color_theorem (n : ℕ) (h : n = 82) :
  ∀ (coloring : Fin n → Type),
    (∃ (cubes : Fin 10 → Fin n), (∀ i j, i ≠ j → coloring (cubes i) ≠ coloring (cubes j))) ∨
    (∃ (color : Type) (cubes : Fin 10 → Fin n), (∀ i, coloring (cubes i) = color)) :=
by sorry

end NUMINAMATH_CALUDE_cube_color_theorem_l1586_158656


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1586_158647

theorem arithmetic_sequence_product (a b c d : ℝ) (m n p : ℕ+) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  b - a = Real.sqrt 2 →
  c - b = Real.sqrt 2 →
  d - c = Real.sqrt 2 →
  a * b * c * d = 2021 →
  d = (m + Real.sqrt n) / Real.sqrt p →
  ∀ (q : ℕ+), q * q ∣ m → q = 1 →
  ∀ (q : ℕ+), q * q ∣ n → q = 1 →
  ∀ (q : ℕ+), q * q ∣ p → q = 1 →
  m + n + p = 100 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1586_158647


namespace NUMINAMATH_CALUDE_base_9_8_conversion_l1586_158602

/-- Represents a number in a given base -/
def BaseRepresentation (base : ℕ) (tens_digit : ℕ) (ones_digit : ℕ) : ℕ :=
  base * tens_digit + ones_digit

theorem base_9_8_conversion : 
  ∃ (n : ℕ) (C D : ℕ), 
    C < 9 ∧ D < 9 ∧ D < 8 ∧ 
    n = BaseRepresentation 9 C D ∧
    n = BaseRepresentation 8 D C ∧
    n = 71 := by
  sorry

end NUMINAMATH_CALUDE_base_9_8_conversion_l1586_158602


namespace NUMINAMATH_CALUDE_stones_kept_as_favorite_l1586_158699

theorem stones_kept_as_favorite (original_stones sent_away_stones : ℕ) 
  (h1 : original_stones = 78) 
  (h2 : sent_away_stones = 63) : 
  original_stones - sent_away_stones = 15 := by
  sorry

end NUMINAMATH_CALUDE_stones_kept_as_favorite_l1586_158699


namespace NUMINAMATH_CALUDE_father_steps_problem_l1586_158644

/-- Calculates the number of steps taken by Father given the step ratios and total steps of children -/
def father_steps (father_masha_ratio : ℚ) (masha_yasha_ratio : ℚ) (total_children_steps : ℕ) : ℕ :=
  sorry

/-- Theorem stating that Father takes 90 steps given the problem conditions -/
theorem father_steps_problem :
  let father_masha_ratio : ℚ := 3 / 5
  let masha_yasha_ratio : ℚ := 3 / 5
  let total_children_steps : ℕ := 400
  father_steps father_masha_ratio masha_yasha_ratio total_children_steps = 90 := by
  sorry

end NUMINAMATH_CALUDE_father_steps_problem_l1586_158644


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l1586_158650

theorem perfect_square_divisibility (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : 
  ∃ k : ℕ, a = k^2 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l1586_158650


namespace NUMINAMATH_CALUDE_decimal_to_binary_34_l1586_158669

theorem decimal_to_binary_34 : 
  (34 : ℕ) = (1 * 2^5 + 0 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_34_l1586_158669


namespace NUMINAMATH_CALUDE_banana_mush_proof_l1586_158679

theorem banana_mush_proof (flour_ratio : ℝ) (total_bananas : ℝ) (total_flour : ℝ)
  (h1 : flour_ratio = 3)
  (h2 : total_bananas = 20)
  (h3 : total_flour = 15) :
  (total_bananas * flour_ratio) / total_flour = 4 := by
  sorry

end NUMINAMATH_CALUDE_banana_mush_proof_l1586_158679


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1586_158642

theorem polynomial_simplification (x : ℝ) :
  (2*x^6 + 3*x^5 + 4*x^4 + x^3 + x^2 + x + 20) - (x^6 + 4*x^5 + 2*x^4 - x^3 + 2*x^2 + x + 5) =
  x^6 - x^5 + 2*x^4 + 2*x^3 - x^2 + 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1586_158642


namespace NUMINAMATH_CALUDE_product_of_fraction_is_111_l1586_158629

/-- The repeating decimal 0.009̄ as a real number -/
def repeating_decimal : ℚ := 1 / 111

/-- The product of numerator and denominator when the repeating decimal is expressed as a fraction in lowest terms -/
def product_of_fraction : ℕ := 111

/-- Theorem stating that the product of the numerator and denominator of the fraction representation of 0.009̄ in lowest terms is 111 -/
theorem product_of_fraction_is_111 : 
  ∃ (n d : ℕ), d ≠ 0 ∧ repeating_decimal = n / d ∧ Nat.gcd n d = 1 ∧ n * d = product_of_fraction :=
by sorry

end NUMINAMATH_CALUDE_product_of_fraction_is_111_l1586_158629


namespace NUMINAMATH_CALUDE_equation_solutions_l1586_158671

theorem equation_solutions :
  (∃ x : ℝ, 2 * (x - 2) = 3 * x - 7 ∧ x = 3) ∧
  (∃ x : ℝ, (x - 1) / 2 - (2 * x + 3) / 6 = 1 ∧ x = 12) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1586_158671


namespace NUMINAMATH_CALUDE_final_amount_correct_l1586_158654

def total_income : ℝ := 1000000

def children_share : ℝ := 0.2
def num_children : ℕ := 3
def wife_share : ℝ := 0.3
def orphan_donation_rate : ℝ := 0.05

def amount_left : ℝ := 
  total_income * (1 - children_share * num_children - wife_share) * (1 - orphan_donation_rate)

theorem final_amount_correct : amount_left = 95000 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_correct_l1586_158654


namespace NUMINAMATH_CALUDE_factorization_of_x2y_minus_4y_l1586_158608

theorem factorization_of_x2y_minus_4y (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x2y_minus_4y_l1586_158608


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l1586_158696

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) :
  Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l1586_158696


namespace NUMINAMATH_CALUDE_cinnamon_swirls_theorem_l1586_158625

/-- The number of people eating cinnamon swirls -/
def num_people : ℕ := 3

/-- The number of pieces Jane ate -/
def janes_pieces : ℕ := 4

/-- The total number of cinnamon swirl pieces prepared -/
def total_pieces : ℕ := num_people * janes_pieces

theorem cinnamon_swirls_theorem :
  total_pieces = 12 :=
sorry

end NUMINAMATH_CALUDE_cinnamon_swirls_theorem_l1586_158625


namespace NUMINAMATH_CALUDE_max_2012_gons_less_than_1006_l1586_158683

/-- The number of sides in each polygon -/
def n : ℕ := 2012

/-- The maximum number of different n-gons that can be drawn with all vertices shared 
    and no sides shared between any two polygons -/
def max_polygons (n : ℕ) : ℕ := (n - 1) / 2

/-- Theorem: The maximum number of different 2012-gons that can be drawn with all vertices shared 
    and no sides shared between any two polygons is less than 1006 -/
theorem max_2012_gons_less_than_1006 : max_polygons n < 1006 := by
  sorry

end NUMINAMATH_CALUDE_max_2012_gons_less_than_1006_l1586_158683


namespace NUMINAMATH_CALUDE_p_range_l1586_158645

theorem p_range (x : ℝ) (P : ℝ) (h1 : x^2 - 5*x + 6 < 0) (h2 : P = x^2 + 5*x + 6) : 
  20 < P ∧ P < 30 := by
sorry

end NUMINAMATH_CALUDE_p_range_l1586_158645


namespace NUMINAMATH_CALUDE_work_completion_l1586_158658

theorem work_completion (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) :
  initial_days = 17 →
  absent_men = 8 →
  final_days = 21 →
  ∃ (original_men : ℕ),
    original_men * initial_days = (original_men - absent_men) * final_days ∧
    original_men = 42 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_l1586_158658


namespace NUMINAMATH_CALUDE_samuel_apple_ratio_l1586_158640

/-- Prove that the ratio of apples Samuel ate to the total number of apples he bought is 1:2 -/
theorem samuel_apple_ratio :
  let bonnie_apples : ℕ := 8
  let samuel_extra_apples : ℕ := 20
  let samuel_total_apples : ℕ := bonnie_apples + samuel_extra_apples
  let samuel_pie_apples : ℕ := samuel_total_apples / 7
  let samuel_left_apples : ℕ := 10
  let samuel_eaten_apples : ℕ := samuel_total_apples - samuel_pie_apples - samuel_left_apples
  (samuel_eaten_apples : ℚ) / (samuel_total_apples : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_samuel_apple_ratio_l1586_158640


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_plus_one_l1586_158693

theorem power_two_greater_than_square_plus_one (n : ℕ) (h : n ≥ 5) :
  2^n > n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_plus_one_l1586_158693


namespace NUMINAMATH_CALUDE_union_of_sets_l1586_158638

def M : Set Int := {-1, 3, -5}
def N (a : Int) : Set Int := {a + 2, a^2 - 6}

theorem union_of_sets :
  ∃ a : Int, (M ∩ N a = {3}) → (M ∪ N a = {-5, -1, 3, 5}) := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1586_158638


namespace NUMINAMATH_CALUDE_third_turtle_lying_l1586_158600

-- Define the type for turtles
inductive Turtle : Type
  | T1 : Turtle
  | T2 : Turtle
  | T3 : Turtle

-- Define the relative position of turtles
inductive Position : Type
  | Front : Position
  | Behind : Position

-- Define a function to represent the statement of each turtle
def turtleStatement (t : Turtle) : List (Turtle × Position) :=
  match t with
  | Turtle.T1 => [(Turtle.T2, Position.Behind), (Turtle.T3, Position.Behind)]
  | Turtle.T2 => [(Turtle.T1, Position.Front), (Turtle.T3, Position.Behind)]
  | Turtle.T3 => [(Turtle.T1, Position.Front), (Turtle.T2, Position.Front), (Turtle.T3, Position.Behind)]

-- Define a function to check if a turtle's statement is consistent with its position
def isConsistent (t : Turtle) (position : Nat) : Prop :=
  match t, position with
  | Turtle.T1, 0 => true
  | Turtle.T2, 1 => true
  | Turtle.T3, 2 => false
  | _, _ => false

-- Theorem: The third turtle's statement is inconsistent
theorem third_turtle_lying :
  ¬ (isConsistent Turtle.T3 2) :=
  sorry


end NUMINAMATH_CALUDE_third_turtle_lying_l1586_158600


namespace NUMINAMATH_CALUDE_largest_prime_factor_l1586_158663

def expression : ℤ := 17^4 + 3 * 17^2 + 2 - 16^4

theorem largest_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression.natAbs ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ expression.natAbs → q ≤ p ∧
  p = 34087 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l1586_158663


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1586_158697

theorem complex_equation_solution : ∃ (z : ℂ), (Complex.I + 1) * z = Complex.abs (2 * Complex.I) ∧ z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1586_158697


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l1586_158619

theorem quadratic_inequality_problem (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  (a = -12 ∧ b = -2) ∧
  (∀ x : ℝ, (a*x + b) / (x - 2) ≥ 0 ↔ -1/6 ≤ x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l1586_158619


namespace NUMINAMATH_CALUDE_smallest_square_area_l1586_158643

-- Define the radius of the circle
def radius : ℝ := 5

-- Define the diameter of the circle
def diameter : ℝ := 2 * radius

-- Define the side length of the square
def side_length : ℝ := diameter

-- Theorem: The area of the smallest square that can completely enclose a circle with a radius of 5 is 100
theorem smallest_square_area : side_length ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l1586_158643


namespace NUMINAMATH_CALUDE_y_value_proof_l1586_158612

theorem y_value_proof (y : ℝ) (h_pos : y > 0) 
  (h_eq : Real.sqrt (12 * y) * Real.sqrt (5 * y) * Real.sqrt (7 * y) * Real.sqrt (21 * y) = 21) : 
  y = 1 / Real.rpow 20 (1/4) :=
sorry

end NUMINAMATH_CALUDE_y_value_proof_l1586_158612


namespace NUMINAMATH_CALUDE_pond_ducks_l1586_158666

/-- The number of ducks in the pond -/
def num_ducks : ℕ := 3

/-- The total number of bread pieces thrown in the pond -/
def total_bread : ℕ := 100

/-- The number of bread pieces left in the water -/
def left_bread : ℕ := 30

/-- The number of bread pieces eaten by the second duck -/
def second_duck_bread : ℕ := 13

/-- The number of bread pieces eaten by the third duck -/
def third_duck_bread : ℕ := 7

/-- Theorem stating that the number of ducks in the pond is 3 -/
theorem pond_ducks : 
  (total_bread / 2 + second_duck_bread + third_duck_bread = total_bread - left_bread) → 
  num_ducks = 3 := by
  sorry


end NUMINAMATH_CALUDE_pond_ducks_l1586_158666


namespace NUMINAMATH_CALUDE_smallest_sphere_and_largest_cylinder_radius_l1586_158617

/-- Three identical cylindrical surfaces with radius R and mutually perpendicular axes that touch each other in pairs -/
structure PerpendicularCylinders (R : ℝ) :=
  (radius : ℝ := R)
  (perpendicular_axes : Prop)
  (touch_in_pairs : Prop)

theorem smallest_sphere_and_largest_cylinder_radius 
  (R : ℝ) 
  (h : R > 0) 
  (cylinders : PerpendicularCylinders R) : 
  ∃ (smallest_sphere_radius largest_cylinder_radius : ℝ),
    smallest_sphere_radius = (Real.sqrt 2 - 1) * R ∧
    largest_cylinder_radius = (Real.sqrt 2 - 1) * R :=
by sorry

end NUMINAMATH_CALUDE_smallest_sphere_and_largest_cylinder_radius_l1586_158617


namespace NUMINAMATH_CALUDE_subset_relation_l1586_158680

universe u

theorem subset_relation (A B : Set α) :
  (∃ x, x ∈ B) →
  (∀ y, y ∈ A → y ∈ B) →
  B ⊆ A :=
by sorry

end NUMINAMATH_CALUDE_subset_relation_l1586_158680


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1586_158628

theorem simplify_and_rationalize (x : ℝ) :
  x = 8 / (Real.sqrt 75 + 3 * Real.sqrt 3 + Real.sqrt 48) →
  x = 2 * Real.sqrt 3 / 9 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1586_158628


namespace NUMINAMATH_CALUDE_sum_of_two_squares_l1586_158694

theorem sum_of_two_squares (P : ℤ) (a b : ℤ) (h : P = a^2 + b^2) :
  ∃ x y : ℤ, 2*P = x^2 + y^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_l1586_158694


namespace NUMINAMATH_CALUDE_fraction_simplification_l1586_158636

theorem fraction_simplification (a b d : ℝ) (h : a^2 + d^2 - b^2 + 2*a*d ≠ 0) :
  (a^2 + b^2 + d^2 + 2*b*d) / (a^2 + d^2 - b^2 + 2*a*d) = 
  (a^2 + (b+d)^2) / ((a+d)^2 + a^2 - b^2) := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1586_158636


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1586_158630

theorem sqrt_equation_solution (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt (4 + x) + 3 * Real.sqrt (4 - x) = 5 * Real.sqrt 6 →
  x = Real.sqrt 43 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1586_158630


namespace NUMINAMATH_CALUDE_video_votes_l1586_158632

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 120 ∧ 
  like_percentage = 72 / 100 →
  ∃ (total_votes : ℕ), 
    (like_percentage * total_votes : ℚ) - ((1 - like_percentage) * total_votes : ℚ) = score ∧
    total_votes = 273 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l1586_158632


namespace NUMINAMATH_CALUDE_connected_triangles_theorem_l1586_158641

/-- A sequence of three connected right-angled triangles -/
structure TriangleSequence where
  -- First triangle
  AE : ℝ
  BE : ℝ
  -- Second triangle
  CE : ℝ
  -- Angles
  angleAEB : Real
  angleBEC : Real
  angleCED : Real

/-- The theorem statement -/
theorem connected_triangles_theorem (t : TriangleSequence) : 
  t.AE = 20 ∧ 
  t.angleAEB = 45 ∧ 
  t.angleBEC = 45 ∧ 
  t.angleCED = 45 → 
  t.CE = 10 := by
  sorry


end NUMINAMATH_CALUDE_connected_triangles_theorem_l1586_158641


namespace NUMINAMATH_CALUDE_dvd_average_price_l1586_158639

theorem dvd_average_price (price1 price2 : ℚ) (count1 count2 : ℕ) 
  (h1 : price1 = 2)
  (h2 : price2 = 5)
  (h3 : count1 = 10)
  (h4 : count2 = 5) :
  (price1 * count1 + price2 * count2) / (count1 + count2) = 3 := by
sorry

end NUMINAMATH_CALUDE_dvd_average_price_l1586_158639


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l1586_158609

-- Define a regular decagon
structure RegularDecagon :=
  (vertices : Finset (ℕ × ℕ))
  (is_regular : vertices.card = 10)

-- Define a triangle formed by three vertices of the decagon
def Triangle (d : RegularDecagon) :=
  {t : Finset (ℕ × ℕ) // t ⊆ d.vertices ∧ t.card = 3}

-- Define a predicate for a triangle not sharing sides with the decagon
def NoSharedSides (d : RegularDecagon) (t : Triangle d) : Prop := sorry

-- Define the probability function
def Probability (d : RegularDecagon) : ℚ := sorry

-- State the theorem
theorem decagon_triangle_probability (d : RegularDecagon) :
  Probability d = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l1586_158609


namespace NUMINAMATH_CALUDE_jellybean_count_l1586_158686

theorem jellybean_count (remaining_ratio : ℝ) (days : ℕ) (final_count : ℕ) 
  (h1 : remaining_ratio = 0.75)
  (h2 : days = 3)
  (h3 : final_count = 27) :
  ∃ (original_count : ℕ), 
    (remaining_ratio ^ days) * (original_count : ℝ) = final_count ∧ 
    original_count = 64 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l1586_158686


namespace NUMINAMATH_CALUDE_parallel_condition_l1586_158684

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Main theorem
theorem parallel_condition 
  (α β : Plane) 
  (m : Line) 
  (h_distinct : α ≠ β) 
  (h_m_in_α : line_in_plane m α) :
  (∀ α β : Plane, plane_parallel α β → line_parallel_plane m β) ∧ 
  (∃ α β : Plane, line_parallel_plane m β ∧ ¬plane_parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l1586_158684


namespace NUMINAMATH_CALUDE_max_value_abc_l1586_158681

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  a^4 * b^2 * c ≤ 1024/117649 := by sorry

end NUMINAMATH_CALUDE_max_value_abc_l1586_158681


namespace NUMINAMATH_CALUDE_single_round_robin_games_planned_games_equation_l1586_158673

/-- Represents the number of games in a single round-robin tournament -/
def num_games (x : ℕ) : ℚ := (x * (x - 1)) / 2

/-- Theorem: In a single round-robin tournament with x teams, 
    the total number of games is given by (x * (x - 1)) / 2 -/
theorem single_round_robin_games (x : ℕ) : 
  num_games x = (x * (x - 1)) / 2 := by
  sorry

/-- Given 15 planned games, prove that the equation (x * (x - 1)) / 2 = 15 
    correctly represents the number of games in terms of x -/
theorem planned_games_equation (x : ℕ) : 
  num_games x = 15 ↔ (x * (x - 1)) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_single_round_robin_games_planned_games_equation_l1586_158673


namespace NUMINAMATH_CALUDE_line_intercept_sum_l1586_158652

/-- Given a line 3x + 5y + c = 0, if the sum of its x-intercept and y-intercept is 55/4, then c = 825/32 -/
theorem line_intercept_sum (c : ℚ) : 
  (∃ x y : ℚ, 3 * x + 5 * y + c = 0 ∧ x + y = 55 / 4) → c = 825 / 32 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l1586_158652


namespace NUMINAMATH_CALUDE_trigonometric_calculation_l1586_158676

theorem trigonometric_calculation : ((-2)^2 : ℝ) + 2 * Real.sin (π/3) - Real.tan (π/3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_calculation_l1586_158676


namespace NUMINAMATH_CALUDE_second_side_bisected_l1586_158657

/-- A nonagon circumscribed around a circle -/
structure CircumscribedNonagon where
  /-- The lengths of the sides of the nonagon -/
  sides : Fin 9 → ℕ
  /-- All sides have positive integer lengths -/
  all_positive : ∀ i, sides i > 0
  /-- The first and third sides have length 1 -/
  first_third_one : sides 0 = 1 ∧ sides 2 = 1

/-- The point of tangency divides the second side into two equal segments -/
theorem second_side_bisected (n : CircumscribedNonagon) :
  ∃ (x : ℚ), x = 1/2 ∧ x * n.sides 1 = (1 - x) * n.sides 1 :=
sorry

end NUMINAMATH_CALUDE_second_side_bisected_l1586_158657


namespace NUMINAMATH_CALUDE_chromosome_stability_processes_l1586_158660

-- Define the type for physiological processes
inductive PhysiologicalProcess
  | Mitosis
  | Amitosis
  | Meiosis
  | Fertilization

-- Define the set of all physiological processes
def allProcesses : Set PhysiologicalProcess :=
  {PhysiologicalProcess.Mitosis, PhysiologicalProcess.Amitosis, 
   PhysiologicalProcess.Meiosis, PhysiologicalProcess.Fertilization}

-- Define the property of maintaining chromosome stability and continuity
def maintainsChromosomeStability (p : PhysiologicalProcess) : Prop :=
  match p with
  | PhysiologicalProcess.Meiosis => true
  | PhysiologicalProcess.Fertilization => true
  | _ => false

-- Theorem: The set of processes that maintain chromosome stability
--          is equal to {Meiosis, Fertilization}
theorem chromosome_stability_processes :
  {p ∈ allProcesses | maintainsChromosomeStability p} = 
  {PhysiologicalProcess.Meiosis, PhysiologicalProcess.Fertilization} :=
by
  sorry


end NUMINAMATH_CALUDE_chromosome_stability_processes_l1586_158660


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1586_158677

theorem right_triangle_perimeter (a b c : ℝ) : 
  a = 10 ∧ b = 24 ∧ c = 26 →
  a + b > c ∧ a + c > b ∧ b + c > a →
  a^2 + b^2 = c^2 →
  a + b + c = 60 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1586_158677


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l1586_158611

/-- The total surface area of a cube with edge length 3 meters and square holes of edge length 1 meter drilled through each face. -/
def total_surface_area (cube_edge : ℝ) (hole_edge : ℝ) : ℝ :=
  let exterior_area := 6 * cube_edge^2 - 6 * hole_edge^2
  let interior_area := 24 * cube_edge * hole_edge
  exterior_area + interior_area

/-- Theorem stating that the total surface area of the described cube is 120 square meters. -/
theorem cube_with_holes_surface_area :
  total_surface_area 3 1 = 120 := by
  sorry

#eval total_surface_area 3 1

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l1586_158611


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1586_158623

theorem quadratic_factorization (x : ℝ) : 4 - 4*x + x^2 = (2 - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1586_158623
