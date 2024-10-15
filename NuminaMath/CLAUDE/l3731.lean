import Mathlib

namespace NUMINAMATH_CALUDE_rainy_days_count_l3731_373124

theorem rainy_days_count (n : ℤ) : 
  (∃ (R NR : ℤ),
    R + NR = 7 ∧ 
    n * R + 4 * NR = 26 ∧ 
    4 * NR - n * R = 14 ∧ 
    R ≥ 0 ∧ NR ≥ 0) → 
  (∃ (R : ℤ), R = 2 ∧ 
    (∃ (NR : ℤ), 
      R + NR = 7 ∧ 
      n * R + 4 * NR = 26 ∧ 
      4 * NR - n * R = 14 ∧ 
      R ≥ 0 ∧ NR ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_rainy_days_count_l3731_373124


namespace NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l3731_373131

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that no positive integer A satisfies the given conditions -/
theorem no_integer_satisfies_conditions : ¬ ∃ A : ℕ+, 
  (sumOfDigits A = 16) ∧ (sumOfDigits (2 * A) = 17) := by sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l3731_373131


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3731_373153

/-- Theorem: Sum of interior angles of a regular polygon with 20-degree exterior angles -/
theorem sum_interior_angles_regular_polygon (n : ℕ) (h1 : n > 2) 
  (h2 : (360 : ℝ) / n = 20) : (n - 2 : ℝ) * 180 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3731_373153


namespace NUMINAMATH_CALUDE_smallest_integer_with_divisibility_pattern_l3731_373183

def is_divisible (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

def consecutive_three (a b c : ℕ) : Prop := b = a + 1 ∧ c = b + 1

theorem smallest_integer_with_divisibility_pattern :
  ∃ (n : ℕ) (a : ℕ),
    n > 0 ∧
    a > 0 ∧
    a < 39 ∧
    consecutive_three a (a + 1) (a + 2) ∧
    (∀ (k : ℕ), k > 0 ∧ k ≤ 40 ∧ k ≠ a ∧ k ≠ (a + 1) ∧ k ≠ (a + 2) → is_divisible n k) ∧
    (¬ is_divisible n a ∧ ¬ is_divisible n (a + 1) ∧ ¬ is_divisible n (a + 2)) ∧
    n = 299576986419800 ∧
    (∀ (m : ℕ), m > 0 ∧ m < n →
      ¬(∃ (b : ℕ), b > 0 ∧ b < 39 ∧
        consecutive_three b (b + 1) (b + 2) ∧
        (∀ (k : ℕ), k > 0 ∧ k ≤ 40 ∧ k ≠ b ∧ k ≠ (b + 1) ∧ k ≠ (b + 2) → is_divisible m k) ∧
        (¬ is_divisible m b ∧ ¬ is_divisible m (b + 1) ∧ ¬ is_divisible m (b + 2))))
  := by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_divisibility_pattern_l3731_373183


namespace NUMINAMATH_CALUDE_line_intersects_circle_twice_l3731_373100

/-- The circle C with equation x^2 + y^2 - 2x - 6y - 15 = 0 -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y - 15 = 0

/-- The line l with equation (1+3k)x + (3-2k)y + 4k - 17 = 0 for any real k -/
def Line (k x y : ℝ) : Prop :=
  (1+3*k)*x + (3-2*k)*y + 4*k - 17 = 0

/-- The theorem stating that the line intersects the circle at exactly two points for any real k -/
theorem line_intersects_circle_twice :
  ∀ k : ℝ, ∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ 
    Circle p1.1 p1.2 ∧ Circle p2.1 p2.2 ∧
    Line k p1.1 p1.2 ∧ Line k p2.1 p2.2 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_twice_l3731_373100


namespace NUMINAMATH_CALUDE_train_stop_time_l3731_373135

/-- Proves that a train with given speeds stops for 20 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 48)
  (h2 : speed_with_stops = 32) :
  (1 - speed_with_stops / speed_without_stops) * 60 = 20 := by
  sorry

#check train_stop_time

end NUMINAMATH_CALUDE_train_stop_time_l3731_373135


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l3731_373103

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l3731_373103


namespace NUMINAMATH_CALUDE_det_specific_matrix_l3731_373178

theorem det_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 4, -1; 0, 3, 2; 5, -1, 3]
  Matrix.det A = 77 := by
sorry

end NUMINAMATH_CALUDE_det_specific_matrix_l3731_373178


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3731_373142

theorem negation_of_existence_proposition :
  ¬(∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔
  (∀ c : ℝ, c > 0 → ∀ x : ℝ, x^2 - x + c ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3731_373142


namespace NUMINAMATH_CALUDE_even_function_alpha_beta_values_l3731_373147

theorem even_function_alpha_beta_values (α β : Real) :
  let f : Real → Real := λ x => 
    if x < 0 then Real.sin (x + α) else Real.cos (x + β)
  (∀ x, f (-x) = f x) →
  α = π / 3 ∧ β = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_even_function_alpha_beta_values_l3731_373147


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l3731_373150

/-- The cost of one dozen pens given the cost of one pen and the ratio of pen to pencil cost -/
theorem cost_of_dozen_pens 
  (cost_of_one_pen : ℕ) 
  (ratio_pen_to_pencil : ℚ) 
  (h1 : cost_of_one_pen = 65) 
  (h2 : ratio_pen_to_pencil = 5 / 1) : 
  12 * cost_of_one_pen = 780 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l3731_373150


namespace NUMINAMATH_CALUDE_class_size_is_69_l3731_373164

/-- Represents the number of students in a class with given enrollment data for French and German courses -/
def total_students (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  (french + german - both) + neither

/-- Theorem stating that the total number of students in the class is 69 -/
theorem class_size_is_69 :
  total_students 41 22 9 15 = 69 := by
  sorry

end NUMINAMATH_CALUDE_class_size_is_69_l3731_373164


namespace NUMINAMATH_CALUDE_camping_trip_items_l3731_373159

theorem camping_trip_items (total_items : ℕ) 
  (tent_stakes : ℕ) (drink_mix : ℕ) (water_bottles : ℕ) : 
  total_items = 22 → 
  drink_mix = 3 * tent_stakes → 
  water_bottles = tent_stakes + 2 → 
  total_items = tent_stakes + drink_mix + water_bottles → 
  tent_stakes = 4 := by
sorry

end NUMINAMATH_CALUDE_camping_trip_items_l3731_373159


namespace NUMINAMATH_CALUDE_largest_common_value_l3731_373113

/-- The first arithmetic progression -/
def progression1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression -/
def progression2 (n : ℕ) : ℕ := 5 + 9 * n

/-- A common term of both progressions -/
def commonTerm (m : ℕ) : ℕ := 14 + 45 * m

theorem largest_common_value :
  (∃ n1 n2 : ℕ, progression1 n1 = 959 ∧ progression2 n2 = 959) ∧ 
  (∀ k : ℕ, k < 1000 → k > 959 → 
    (∀ n1 n2 : ℕ, progression1 n1 ≠ k ∨ progression2 n2 ≠ k)) :=
sorry

end NUMINAMATH_CALUDE_largest_common_value_l3731_373113


namespace NUMINAMATH_CALUDE_multiple_birth_statistics_l3731_373186

theorem multiple_birth_statistics (total_babies : ℕ) 
  (twins triplets quadruplets quintuplets : ℕ) : 
  total_babies = 1200 →
  quintuplets = 2 * quadruplets →
  quadruplets = 3 * triplets →
  triplets = 2 * twins →
  2 * twins + 3 * triplets + 4 * quadruplets + 5 * quintuplets = total_babies →
  5 * quintuplets = 18000 / 23 := by
sorry

end NUMINAMATH_CALUDE_multiple_birth_statistics_l3731_373186


namespace NUMINAMATH_CALUDE_team_handedness_ratio_l3731_373167

/-- A ball team with right-handed and left-handed players -/
structure BallTeam where
  right_handed : ℕ
  left_handed : ℕ

/-- Represents the attendance at practice -/
structure PracticeAttendance (team : BallTeam) where
  present_right : ℕ
  present_left : ℕ
  absent_right : ℕ
  absent_left : ℕ
  total_present : present_right + present_left = team.right_handed + team.left_handed - (absent_right + absent_left)
  all_accounted : present_right + absent_right = team.right_handed
  all_accounted_left : present_left + absent_left = team.left_handed

/-- The theorem representing the problem -/
theorem team_handedness_ratio (team : BallTeam) (attendance : PracticeAttendance team) :
  (2 : ℚ) / 3 * (team.right_handed + team.left_handed) = attendance.absent_right + attendance.absent_left →
  (2 : ℚ) / 3 * (attendance.present_right + attendance.present_left) = attendance.present_left →
  (attendance.absent_right : ℚ) / attendance.absent_left = 14 / 10 →
  (team.right_handed : ℚ) / team.left_handed = 14 / 10 := by
  sorry

end NUMINAMATH_CALUDE_team_handedness_ratio_l3731_373167


namespace NUMINAMATH_CALUDE_residue_13_2045_mod_19_l3731_373130

theorem residue_13_2045_mod_19 : (13 ^ 2045 : ℕ) % 19 = 9 := by sorry

end NUMINAMATH_CALUDE_residue_13_2045_mod_19_l3731_373130


namespace NUMINAMATH_CALUDE_gum_distribution_l3731_373197

/-- Given the number of gum pieces for each person and the total number of people,
    calculate the number of gum pieces each person will receive after equal distribution. -/
def distribute_gum (john_gum : ℕ) (cole_gum : ℕ) (aubrey_gum : ℕ) (num_people : ℕ) : ℕ :=
  (john_gum + cole_gum + aubrey_gum) / num_people

/-- Theorem stating that when 54 pieces of gum, 45 pieces of gum, and 0 pieces of gum
    are combined and divided equally among 3 people, each person will receive 33 pieces of gum. -/
theorem gum_distribution :
  distribute_gum 54 45 0 3 = 33 := by
  sorry

#eval distribute_gum 54 45 0 3

end NUMINAMATH_CALUDE_gum_distribution_l3731_373197


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_l3731_373157

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum_zero 
  (a b c : ℝ) 
  (h1 : quadratic a b c 1 = 0)
  (h2 : quadratic a b c 5 = 0)
  (h3 : ∃ (k : ℝ), ∀ (x : ℝ), quadratic a b c x ≥ 36 ∧ quadratic a b c k = 36) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_l3731_373157


namespace NUMINAMATH_CALUDE_m_range_for_inequality_l3731_373182

theorem m_range_for_inequality (m : ℝ) : 
  (∀ x : ℝ, x ≤ -1 → (m^2 - m) * 4^x - 2^x < 0) ↔ -1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_for_inequality_l3731_373182


namespace NUMINAMATH_CALUDE_f_properties_l3731_373180

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x^2

theorem f_properties :
  (∃ x : ℝ, x ≠ 0 ∧ f x = 0 ↔ x = 1) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ∀ y : ℝ, y ≠ 0 → f y ≤ f x) ∧
  (∀ x : ℝ, x ≠ 0 → f x ≤ f 2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3731_373180


namespace NUMINAMATH_CALUDE_income_comparison_l3731_373151

theorem income_comparison (juan tim mart : ℝ) 
  (h1 : tim = 0.6 * juan) 
  (h2 : mart = 0.78 * juan) : 
  (mart - tim) / tim * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l3731_373151


namespace NUMINAMATH_CALUDE_no_rain_probability_l3731_373189

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^4 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l3731_373189


namespace NUMINAMATH_CALUDE_linda_win_probability_is_two_thirty_first_l3731_373123

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a player in the game -/
inductive Player
| Sara
| Peter
| Linda

/-- The game state -/
structure GameState where
  currentPlayer : Player
  saraLastFlip : Option CoinFlip
  
/-- The result of a game round -/
inductive RoundResult
| Continue (newState : GameState)
| SaraWins
| LindaWins

/-- Simulates a single round of the game -/
def playRound (state : GameState) (flip : CoinFlip) : RoundResult := sorry

/-- Calculates the probability of Linda winning given the game rules -/
def lindaWinProbability : ℚ := sorry

/-- Theorem stating that the probability of Linda winning is 2/31 -/
theorem linda_win_probability_is_two_thirty_first :
  lindaWinProbability = 2 / 31 := by sorry

end NUMINAMATH_CALUDE_linda_win_probability_is_two_thirty_first_l3731_373123


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3731_373121

/-- A quadratic function f(x) = x^2 + bx + c with f(1) = 0 and f(3) = 0 satisfies f(-1) = 8 -/
theorem quadratic_function_property (b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + b*x + c) 
  (h2 : f 1 = 0) 
  (h3 : f 3 = 0) : 
  f (-1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3731_373121


namespace NUMINAMATH_CALUDE_reverse_digit_integers_l3731_373132

theorem reverse_digit_integers (q r : ℕ) : 
  (q ≥ 10 ∧ q < 100) →  -- q is a two-digit positive integer
  (r ≥ 10 ∧ r < 100) →  -- r is a two-digit positive integer
  (q.div 10 = r.mod 10 ∧ q.mod 10 = r.div 10) →  -- q and r have the same digits in reverse order
  (q > r → q - r < 20) →  -- positive difference is less than 20
  (r > q → r - q < 20) →  -- positive difference is less than 20
  (∀ a b : ℕ, (a ≥ 10 ∧ a < 100) → (b ≥ 10 ∧ b < 100) → 
    (a.div 10 = b.mod 10 ∧ a.mod 10 = b.div 10) → (a - b ≤ 18)) →  -- greatest possible difference is 18
  (q.div 10 = q.mod 10 + 2) →  -- tens digit is 2 more than units digit for q
  (r.div 10 + 2 = r.mod 10) -- tens digit is 2 more than units digit for r (reverse of q)
  := by sorry

end NUMINAMATH_CALUDE_reverse_digit_integers_l3731_373132


namespace NUMINAMATH_CALUDE_two_point_questions_count_l3731_373168

/-- A test with two types of questions -/
structure Test where
  total_points : ℕ
  total_questions : ℕ
  two_point_questions : ℕ
  four_point_questions : ℕ

/-- The test satisfies the given conditions -/
def valid_test (t : Test) : Prop :=
  t.total_points = 100 ∧
  t.total_questions = 40 ∧
  t.two_point_questions + t.four_point_questions = t.total_questions ∧
  2 * t.two_point_questions + 4 * t.four_point_questions = t.total_points

theorem two_point_questions_count (t : Test) (h : valid_test t) :
  t.two_point_questions = 30 :=
by sorry

end NUMINAMATH_CALUDE_two_point_questions_count_l3731_373168


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3731_373146

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3731_373146


namespace NUMINAMATH_CALUDE_circle_radius_zero_l3731_373199

/-- The radius of a circle defined by the equation 4x^2 - 8x + 4y^2 + 16y + 20 = 0 is 0 -/
theorem circle_radius_zero (x y : ℝ) : 
  (4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0) → 
  ∃ (h k : ℝ), ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l3731_373199


namespace NUMINAMATH_CALUDE_weekly_wage_problem_l3731_373192

/-- The weekly wage problem -/
theorem weekly_wage_problem (Rm Hm Rn Hn : ℝ) 
  (h1 : Rm * Hm + Rn * Hn = 770)
  (h2 : Rm * Hm = 1.3 * (Rn * Hn)) :
  Rn * Hn = 335 := by
  sorry

end NUMINAMATH_CALUDE_weekly_wage_problem_l3731_373192


namespace NUMINAMATH_CALUDE_max_value_product_sum_l3731_373173

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 := by
sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l3731_373173


namespace NUMINAMATH_CALUDE_partnership_gain_l3731_373174

/-- Represents the investment and profit structure of a partnership --/
structure Partnership where
  raman_investment : ℝ
  lakshmi_share : ℝ
  profit_ratio : ℝ → ℝ → ℝ → Prop

/-- Calculates the total annual gain of the partnership --/
def total_annual_gain (p : Partnership) : ℝ :=
  3 * p.lakshmi_share

/-- Theorem stating that the total annual gain of the partnership is 36000 --/
theorem partnership_gain (p : Partnership) 
  (h1 : p.profit_ratio (p.raman_investment * 12) (2 * p.raman_investment * 6) (3 * p.raman_investment * 4))
  (h2 : p.lakshmi_share = 12000) : 
  total_annual_gain p = 36000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_gain_l3731_373174


namespace NUMINAMATH_CALUDE_power_product_equals_128_l3731_373181

theorem power_product_equals_128 (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_128_l3731_373181


namespace NUMINAMATH_CALUDE_discount_difference_l3731_373152

theorem discount_difference (bill : ℝ) (single_discount : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  bill = 20000 ∧
  single_discount = 0.3 ∧
  first_discount = 0.25 ∧
  second_discount = 0.05 →
  bill * (1 - first_discount) * (1 - second_discount) - bill * (1 - single_discount) = 250 :=
by sorry

end NUMINAMATH_CALUDE_discount_difference_l3731_373152


namespace NUMINAMATH_CALUDE_triangle_side_length_range_l3731_373127

theorem triangle_side_length_range (b : ℝ) (B : ℝ) :
  b = 2 →
  B = π / 3 →
  ∃ (a : ℝ), 2 < a ∧ a < 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_range_l3731_373127


namespace NUMINAMATH_CALUDE_intersection_A_B_l3731_373162

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3731_373162


namespace NUMINAMATH_CALUDE_line_segment_point_sum_l3731_373125

/-- Given a line y = -2/3x + 6, prove that the sum of coordinates of point T
    satisfies r + s = 8.25, where T(r,s) is on PQ, P and Q are x and y intercepts,
    and area of POQ is 4 times area of TOP. -/
theorem line_segment_point_sum (x₁ y₁ r s : ℝ) : 
  y₁ = 6 ∧                        -- Q is (0, y₁)
  x₁ = 9 ∧                        -- P is (x₁, 0)
  s = -2/3 * r + 6 ∧              -- T(r,s) is on the line
  0 ≤ r ∧ r ≤ x₁ ∧                -- T is between P and Q
  1/2 * x₁ * y₁ = 4 * (1/2 * r * s) -- Area POQ = 4 * Area TOP
  → r + s = 8.25 := by
    sorry

end NUMINAMATH_CALUDE_line_segment_point_sum_l3731_373125


namespace NUMINAMATH_CALUDE_matrix_power_four_l3731_373158

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l3731_373158


namespace NUMINAMATH_CALUDE_train_passing_time_l3731_373141

/-- Calculates the time for two trains to clear each other --/
theorem train_passing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 160)
  (h2 : length2 = 280)
  (h3 : speed1 = 42)
  (h4 : speed2 = 30) : 
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600)) = 22 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l3731_373141


namespace NUMINAMATH_CALUDE_sydney_initial_rocks_l3731_373188

/-- Rock collecting contest between Sydney and Conner --/
def rock_contest (sydney_initial : ℕ) : Prop :=
  let conner_initial := 723
  let sydney_day1 := 4
  let conner_day1 := 8 * sydney_day1
  let sydney_day2 := 0
  let conner_day2 := 123
  let sydney_day3 := 2 * conner_day1
  let conner_day3 := 27

  let sydney_total := sydney_initial + sydney_day1 + sydney_day2 + sydney_day3
  let conner_total := conner_initial + conner_day1 + conner_day2 + conner_day3

  sydney_total ≤ conner_total ∧ sydney_initial = 837

theorem sydney_initial_rocks : rock_contest 837 := by
  sorry

end NUMINAMATH_CALUDE_sydney_initial_rocks_l3731_373188


namespace NUMINAMATH_CALUDE_remainder_11_pow_101_mod_7_l3731_373169

theorem remainder_11_pow_101_mod_7 : 11^101 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_pow_101_mod_7_l3731_373169


namespace NUMINAMATH_CALUDE_rental_cost_difference_theorem_l3731_373102

/-- Calculates the rental cost difference between a ski boat and a sailboat --/
def rental_cost_difference (
  sailboat_weekday_cost : ℕ)
  (skiboat_weekend_hourly_cost : ℕ)
  (sailboat_fuel_cost_per_hour : ℕ)
  (skiboat_fuel_cost_per_hour : ℕ)
  (rental_hours_per_day : ℕ)
  (rental_days : ℕ)
  (discount_percentage : ℕ) : ℕ :=
  let sailboat_day1_cost := sailboat_weekday_cost + sailboat_fuel_cost_per_hour * rental_hours_per_day
  let sailboat_day2_cost := (sailboat_weekday_cost * (100 - discount_percentage) / 100) + sailboat_fuel_cost_per_hour * rental_hours_per_day
  let sailboat_total_cost := sailboat_day1_cost + sailboat_day2_cost

  let skiboat_day1_cost := skiboat_weekend_hourly_cost * rental_hours_per_day + skiboat_fuel_cost_per_hour * rental_hours_per_day
  let skiboat_day2_cost := (skiboat_weekend_hourly_cost * rental_hours_per_day * (100 - discount_percentage) / 100) + skiboat_fuel_cost_per_hour * rental_hours_per_day
  let skiboat_total_cost := skiboat_day1_cost + skiboat_day2_cost

  skiboat_total_cost - sailboat_total_cost

theorem rental_cost_difference_theorem :
  rental_cost_difference 60 120 10 20 3 2 10 = 630 := by
  sorry

end NUMINAMATH_CALUDE_rental_cost_difference_theorem_l3731_373102


namespace NUMINAMATH_CALUDE_max_value_of_f_l3731_373122

/-- Definition of the sum of the first n terms of the geometric sequence -/
def S (n : ℕ) (k : ℝ) : ℝ := 2^(n-1) + k

/-- Definition of the function f -/
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x^2 - 2*x + 1

/-- Theorem stating the maximum value of f -/
theorem max_value_of_f (k : ℝ) : 
  (∃ (n : ℕ), ∀ (m : ℕ), S m k = 2^(m-1) + k) → 
  (∃ (x : ℝ), ∀ (y : ℝ), f k y ≤ f k x) ∧ 
  (∃ (x : ℝ), f k x = 5/2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3731_373122


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3731_373184

/-- A line is described by the equation y + 3 = -3(x + 5).
    This theorem proves that the sum of its x-intercept and y-intercept is -24. -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 3 = -3 * (x + 5)) → 
  (∃ x_int y_int : ℝ, (y_int + 3 = -3 * (x_int + 5)) ∧ 
                      (0 + 3 = -3 * (x_int + 5)) ∧ 
                      (y_int + 3 = -3 * (0 + 5)) ∧ 
                      (x_int + y_int = -24)) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3731_373184


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3731_373171

-- Define the arithmetic sequence an
def an (n : ℕ) : ℝ := 2 * 3^(n - 1)

-- Define the sequence bn
def bn (n : ℕ) : ℝ := an n - 2 * n

-- Define the sum of the first n terms of bn
def Tn (n : ℕ) : ℝ := 3^n - 1 - n^2 - n

theorem arithmetic_sequence_proof :
  (∀ n : ℕ, n ≥ 1 → an n = 2 * 3^(n - 1)) ∧
  (an 2 = 6) ∧
  (an 1 + an 2 + an 3 = 26) ∧
  (∀ n : ℕ, n ≥ 1 → Tn n = 3^n - 1 - n^2 - n) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3731_373171


namespace NUMINAMATH_CALUDE_first_movie_length_proof_l3731_373126

/-- Represents the length of the first movie in hours -/
def first_movie_length : ℝ := 3.5

/-- Represents the length of the second movie in hours -/
def second_movie_length : ℝ := 1.5

/-- Represents the total available time in hours -/
def total_time : ℝ := 8

/-- Represents the reading rate in words per minute -/
def reading_rate : ℝ := 10

/-- Represents the total number of words read -/
def total_words_read : ℝ := 1800

/-- Proves that given the conditions, the length of the first movie must be 3.5 hours -/
theorem first_movie_length_proof :
  first_movie_length + second_movie_length + (total_words_read / reading_rate / 60) = total_time :=
by sorry

end NUMINAMATH_CALUDE_first_movie_length_proof_l3731_373126


namespace NUMINAMATH_CALUDE_jacob_coin_problem_l3731_373185

theorem jacob_coin_problem :
  ∃ (p n d : ℕ),
    p + n + d = 50 ∧
    p + 5 * n + 10 * d = 220 ∧
    d = 18 := by
  sorry

end NUMINAMATH_CALUDE_jacob_coin_problem_l3731_373185


namespace NUMINAMATH_CALUDE_derivative_equals_one_l3731_373161

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else 2^x

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ :=
  if x > 0 then 1 / (x * Real.log 2)
  else 2^x * Real.log 2

-- Theorem statement
theorem derivative_equals_one (a : ℝ) :
  f_derivative a = 1 ↔ a = 1 / Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_derivative_equals_one_l3731_373161


namespace NUMINAMATH_CALUDE_anna_final_collection_l3731_373136

structure StampCollection :=
  (nature : ℕ)
  (architecture : ℕ)
  (animals : ℕ)

def initial_anna : StampCollection := ⟨10, 15, 12⟩
def initial_alison : StampCollection := ⟨8, 10, 10⟩
def initial_jeff : StampCollection := ⟨12, 9, 10⟩

def transaction1 (anna alison : StampCollection) : StampCollection :=
  ⟨anna.nature + alison.nature / 2, anna.architecture + alison.architecture / 2, anna.animals + alison.animals / 2⟩

def transaction2 (anna : StampCollection) : StampCollection :=
  ⟨anna.nature + 2, anna.architecture, anna.animals - 1⟩

def transaction3 (anna : StampCollection) : StampCollection :=
  ⟨anna.nature, anna.architecture + 3, anna.animals - 5⟩

def transaction4 (anna : StampCollection) : StampCollection :=
  ⟨anna.nature + 7, anna.architecture, anna.animals - 4⟩

def final_anna : StampCollection :=
  transaction4 (transaction3 (transaction2 (transaction1 initial_anna initial_alison)))

theorem anna_final_collection :
  final_anna = ⟨23, 23, 7⟩ := by sorry

end NUMINAMATH_CALUDE_anna_final_collection_l3731_373136


namespace NUMINAMATH_CALUDE_senate_committee_seating_l3731_373198

/-- The number of ways to arrange n distinguishable objects in a circle -/
def circularPermutations (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of politicians in the committee -/
def committeeSize : ℕ := 4 + 4 + 3

theorem senate_committee_seating :
  circularPermutations committeeSize = 3628800 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_seating_l3731_373198


namespace NUMINAMATH_CALUDE_inequality_solutions_l3731_373116

/-- The solution set of the inequality 2x^2 + x - 3 < 0 -/
def solution_set_1 : Set ℝ := { x | -3/2 < x ∧ x < 1 }

/-- The solution set of the inequality x(9 - x) > 0 -/
def solution_set_2 : Set ℝ := { x | 0 < x ∧ x < 9 }

theorem inequality_solutions :
  (∀ x : ℝ, x ∈ solution_set_1 ↔ 2*x^2 + x - 3 < 0) ∧
  (∀ x : ℝ, x ∈ solution_set_2 ↔ x*(9 - x) > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l3731_373116


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3731_373104

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 7 = -8)
  (h_a2 : a 2 = 2) :
  ∃ d : ℝ, d = -3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3731_373104


namespace NUMINAMATH_CALUDE_f_neg_one_eq_zero_f_is_even_x_range_l3731_373148

noncomputable section

variable (f : ℝ → ℝ)

-- Define the functional equation
axiom functional_eq : ∀ (x₁ x₂ : ℝ), x₁ ≠ 0 → x₂ ≠ 0 → f (x₁ * x₂) = f x₁ + f x₂

-- Define that f is increasing on (0, +∞)
axiom f_increasing : ∀ (x y : ℝ), 0 < x → x < y → f x < f y

-- Define the inequality condition
axiom f_inequality : ∀ (x : ℝ), f (2 * x - 1) < f x

-- Theorem 1: f(-1) = 0
theorem f_neg_one_eq_zero : f (-1) = 0 := by sorry

-- Theorem 2: f is an even function
theorem f_is_even : ∀ (x : ℝ), f (-x) = f x := by sorry

-- Theorem 3: Range of x
theorem x_range : ∀ (x : ℝ), (1/3 < x ∧ x < 1) ↔ (f (2*x - 1) < f x ∧ ∀ (y z : ℝ), 0 < y → y < z → f y < f z) := by sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_zero_f_is_even_x_range_l3731_373148


namespace NUMINAMATH_CALUDE_improper_integral_convergence_l3731_373139

open Real MeasureTheory

/-- The improper integral ∫[a to b] 1/(x-a)^α dx converges if and only if 0 < α < 1, given α > 0 and b > a -/
theorem improper_integral_convergence 
  (a b : ℝ) (α : ℝ) 
  (h1 : α > 0) 
  (h2 : b > a) : 
  (∃ (I : ℝ), ∫ x in a..b, 1 / (x - a) ^ α = I) ↔ 0 < α ∧ α < 1 :=
sorry

end NUMINAMATH_CALUDE_improper_integral_convergence_l3731_373139


namespace NUMINAMATH_CALUDE_roots_equation_value_l3731_373112

theorem roots_equation_value (α β : ℝ) : 
  α^2 - α - 1 = 0 → β^2 - β - 1 = 0 → α^4 + 3*β = 5 := by sorry

end NUMINAMATH_CALUDE_roots_equation_value_l3731_373112


namespace NUMINAMATH_CALUDE_pie_cost_satisfies_conditions_l3731_373120

/-- The cost of one pie in rubles -/
def pie_cost : ℚ := 20

/-- The total value of Masha's two-ruble coins -/
def two_ruble_coins : ℚ := 4 * pie_cost - 60

/-- The total value of Masha's five-ruble coins -/
def five_ruble_coins : ℚ := 5 * pie_cost - 60

/-- Theorem stating that the pie cost satisfies all given conditions -/
theorem pie_cost_satisfies_conditions :
  (4 * pie_cost = two_ruble_coins + 60) ∧
  (5 * pie_cost = five_ruble_coins + 60) ∧
  (6 * pie_cost = two_ruble_coins + five_ruble_coins + 60) :=
by sorry

#check pie_cost_satisfies_conditions

end NUMINAMATH_CALUDE_pie_cost_satisfies_conditions_l3731_373120


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3731_373193

theorem power_of_two_equality (x : ℤ) : (1 / 8 : ℚ) * (2 ^ 50) = 2 ^ x → x = 47 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3731_373193


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_l3731_373105

theorem opposite_numbers_sum (a b : ℝ) : a + b = 0 → 3*a + 3*b + 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_l3731_373105


namespace NUMINAMATH_CALUDE_valerie_stamps_l3731_373110

/-- The number of stamps Valerie needs for all her envelopes --/
def total_stamps : ℕ :=
  let thank_you_cards := 3
  let water_bill := 1
  let electric_bill := 2
  let internet_bill := 3
  let bills := water_bill + electric_bill + internet_bill
  let rebates := bills + 3
  let job_applications := 2 * rebates
  thank_you_cards + bills + 2 * rebates + job_applications

/-- Theorem stating that Valerie needs 33 stamps in total --/
theorem valerie_stamps : total_stamps = 33 := by
  sorry

end NUMINAMATH_CALUDE_valerie_stamps_l3731_373110


namespace NUMINAMATH_CALUDE_slope_angle_range_l3731_373160

noncomputable def slope_angle (α : Real) : Prop :=
  ∃ (x : Real), x ≠ 0 ∧ Real.tan α = (1/2) * (x + 1/x)

theorem slope_angle_range :
  ∀ α, slope_angle α → 
    (α ∈ Set.Icc (π/4) (π/2) ∪ Set.Ioc (π/2) (3*π/4)) := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_range_l3731_373160


namespace NUMINAMATH_CALUDE_sledding_time_difference_l3731_373115

/-- Given the conditions of Mary and Ann's sledding trip, prove that Ann's trip takes 13 minutes longer than Mary's. -/
theorem sledding_time_difference 
  (mary_hill_length : ℝ) 
  (mary_speed : ℝ) 
  (ann_hill_length : ℝ) 
  (ann_speed : ℝ) 
  (h1 : mary_hill_length = 630)
  (h2 : mary_speed = 90)
  (h3 : ann_hill_length = 800)
  (h4 : ann_speed = 40) :
  ann_hill_length / ann_speed - mary_hill_length / mary_speed = 13 := by
  sorry

end NUMINAMATH_CALUDE_sledding_time_difference_l3731_373115


namespace NUMINAMATH_CALUDE_jennifer_cards_left_l3731_373190

/-- Given that Jennifer has 72 cards initially and 61 cards are eaten,
    prove that she will have 11 cards left. -/
theorem jennifer_cards_left (initial_cards : ℕ) (eaten_cards : ℕ) 
  (h1 : initial_cards = 72) 
  (h2 : eaten_cards = 61) : 
  initial_cards - eaten_cards = 11 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_cards_left_l3731_373190


namespace NUMINAMATH_CALUDE_allison_total_supplies_l3731_373101

/-- Represents the number of craft supplies bought by a person -/
structure CraftSupplies where
  glueSticks : ℕ
  constructionPaper : ℕ

/-- The total number of craft supplies -/
def CraftSupplies.total (cs : CraftSupplies) : ℕ :=
  cs.glueSticks + cs.constructionPaper

/-- Given information about Marie's purchases -/
def marie : CraftSupplies :=
  { glueSticks := 15
    constructionPaper := 30 }

/-- Theorem stating the total number of craft supplies Allison bought -/
theorem allison_total_supplies : 
  ∃ (allison : CraftSupplies), 
    (allison.glueSticks = marie.glueSticks + 8) ∧ 
    (allison.constructionPaper * 6 = marie.constructionPaper) ∧ 
    (allison.total = 28) := by
  sorry

end NUMINAMATH_CALUDE_allison_total_supplies_l3731_373101


namespace NUMINAMATH_CALUDE_total_distance_walked_l3731_373170

/-- Calculates the total distance walked to various destinations in a school. -/
theorem total_distance_walked (water_fountain_dist : ℕ) (main_office_dist : ℕ) (teacher_lounge_dist : ℕ)
  (water_fountain_trips : ℕ) (main_office_trips : ℕ) (teacher_lounge_trips : ℕ)
  (h1 : water_fountain_dist = 30)
  (h2 : main_office_dist = 50)
  (h3 : teacher_lounge_dist = 35)
  (h4 : water_fountain_trips = 4)
  (h5 : main_office_trips = 2)
  (h6 : teacher_lounge_trips = 3) :
  water_fountain_dist * water_fountain_trips +
  main_office_dist * main_office_trips +
  teacher_lounge_dist * teacher_lounge_trips = 325 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l3731_373170


namespace NUMINAMATH_CALUDE_fraction_equality_l3731_373179

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x + y) / (x - 5 * y) = -3) : 
  (x + 5 * y) / (5 * x - y) = 27 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3731_373179


namespace NUMINAMATH_CALUDE_lcm_gcd_ratio_240_360_l3731_373143

theorem lcm_gcd_ratio_240_360 : (lcm 240 360) / (gcd 240 360) = 6 := by sorry

end NUMINAMATH_CALUDE_lcm_gcd_ratio_240_360_l3731_373143


namespace NUMINAMATH_CALUDE_blueberries_count_l3731_373106

/-- Represents the number of blueberries in each blue box -/
def blueberries : ℕ := sorry

/-- Represents the number of strawberries in each red box -/
def strawberries : ℕ := sorry

/-- The increase in total berries when replacing a blue box with a red box -/
def berry_increase : ℕ := 10

/-- The increase in the difference between strawberries and blueberries when replacing a blue box with a red box -/
def difference_increase : ℕ := 50

theorem blueberries_count : 
  (strawberries - blueberries = berry_increase) ∧ 
  (strawberries = difference_increase) → 
  blueberries = 40 := by sorry

end NUMINAMATH_CALUDE_blueberries_count_l3731_373106


namespace NUMINAMATH_CALUDE_parallel_vector_proof_l3731_373108

/-- Given a planar vector b parallel to a = (2, 1) with magnitude 2√5, prove b is either (4, 2) or (-4, -2) -/
theorem parallel_vector_proof (b : ℝ × ℝ) : 
  (∃ k : ℝ, b = (2*k, k)) → -- b is parallel to (2, 1)
  (b.1^2 + b.2^2 = 20) →    -- |b| = 2√5
  (b = (4, 2) ∨ b = (-4, -2)) := by
sorry

end NUMINAMATH_CALUDE_parallel_vector_proof_l3731_373108


namespace NUMINAMATH_CALUDE_product_correction_l3731_373166

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The problem statement -/
theorem product_correction (a b : ℕ) : 
  10 ≤ a ∧ a < 100 →  -- a is a two-digit number
  a > 0 →  -- a is positive
  b > 0 →  -- b is positive
  reverse_digits a * b = 284 →
  a * b = 68 := by
sorry

end NUMINAMATH_CALUDE_product_correction_l3731_373166


namespace NUMINAMATH_CALUDE_pentagon_3010th_position_l3731_373137

/-- Represents the possible positions of the pentagon --/
inductive PentagonPosition
  | ABCDE
  | EABCD
  | DCBAE
  | EDABC

/-- Represents the operations that can be performed on the pentagon --/
inductive Operation
  | Rotate
  | Reflect

/-- Applies an operation to a pentagon position --/
def applyOperation (pos : PentagonPosition) (op : Operation) : PentagonPosition :=
  match pos, op with
  | PentagonPosition.ABCDE, Operation.Rotate => PentagonPosition.EABCD
  | PentagonPosition.EABCD, Operation.Reflect => PentagonPosition.DCBAE
  | PentagonPosition.DCBAE, Operation.Rotate => PentagonPosition.EDABC
  | PentagonPosition.EDABC, Operation.Reflect => PentagonPosition.ABCDE
  | _, _ => pos  -- Default case to satisfy exhaustiveness

/-- Applies a sequence of alternating rotate and reflect operations --/
def applySequence (n : Nat) : PentagonPosition :=
  match n % 4 with
  | 0 => PentagonPosition.ABCDE
  | 1 => PentagonPosition.EABCD
  | 2 => PentagonPosition.DCBAE
  | _ => PentagonPosition.EDABC

theorem pentagon_3010th_position :
  applySequence 3010 = PentagonPosition.ABCDE :=
sorry


end NUMINAMATH_CALUDE_pentagon_3010th_position_l3731_373137


namespace NUMINAMATH_CALUDE_no_two_digit_product_concatenation_l3731_373196

theorem no_two_digit_product_concatenation : ¬∃ (a b c d : ℕ), 
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  (10 * a + b) * (10 * c + d) = 1000 * a + 100 * b + 10 * c + d :=
by sorry

end NUMINAMATH_CALUDE_no_two_digit_product_concatenation_l3731_373196


namespace NUMINAMATH_CALUDE_expression_evaluation_l3731_373144

theorem expression_evaluation :
  (4^1001 * 9^1002) / (6^1002 * 4^1000) = 3^1002 / 2^1000 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3731_373144


namespace NUMINAMATH_CALUDE_fraction_equality_l3731_373172

theorem fraction_equality (x y z w k : ℝ) : 
  (9 / (x + y + w) = k / (x + z + w)) ∧ 
  (k / (x + z + w) = 12 / (z - y)) → 
  k = 21 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3731_373172


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3731_373195

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The sum of two line segments perpendicular to the asymptotes
    and passing through one of the foci -/
def sum_perp_segments (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_sum : sum_perp_segments h = h.a) : 
  eccentricity h = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3731_373195


namespace NUMINAMATH_CALUDE_num_parallelepipeds_is_29_l3731_373128

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of four points in 3D space -/
def FourPoints := Fin 4 → Point3D

/-- Predicate to check if four points are non-coplanar -/
def NonCoplanar (points : FourPoints) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧
    ∀ (i : Fin 4), a * (points i).x + b * (points i).y + c * (points i).z + d = 0

/-- The number of distinct parallelepipeds that can be formed -/
def NumParallelepipeds (points : FourPoints) : ℕ := 29

/-- Theorem stating that the number of distinct parallelepipeds is 29 -/
theorem num_parallelepipeds_is_29 (points : FourPoints) (h : NonCoplanar points) :
  NumParallelepipeds points = 29 := by
  sorry

end NUMINAMATH_CALUDE_num_parallelepipeds_is_29_l3731_373128


namespace NUMINAMATH_CALUDE_divisible_by_five_count_is_correct_l3731_373177

/-- The number of different positive, seven-digit integers divisible by 5,
    formed using the digits 2 (three times), 5 (two times), and 9 (two times) -/
def divisible_by_five_count : ℕ :=
  let total_digits : ℕ := 7
  let two_count : ℕ := 3
  let five_count : ℕ := 2
  let nine_count : ℕ := 2
  60

theorem divisible_by_five_count_is_correct :
  divisible_by_five_count = 60 := by sorry

end NUMINAMATH_CALUDE_divisible_by_five_count_is_correct_l3731_373177


namespace NUMINAMATH_CALUDE_meghans_money_is_550_l3731_373187

/-- Represents the number of bills of a specific denomination --/
structure BillCount where
  count : Nat
  denomination : Nat

/-- Calculates the total value of bills given their count and denomination --/
def billValue (b : BillCount) : Nat := b.count * b.denomination

/-- Represents Meghan's money --/
structure MeghansMoney where
  hundreds : BillCount
  fifties : BillCount
  tens : BillCount

/-- Calculates the total value of Meghan's money --/
def totalValue (m : MeghansMoney) : Nat :=
  billValue m.hundreds + billValue m.fifties + billValue m.tens

/-- Theorem stating that Meghan's total money is $550 --/
theorem meghans_money_is_550 (m : MeghansMoney) 
  (h1 : m.hundreds = { count := 2, denomination := 100 })
  (h2 : m.fifties = { count := 5, denomination := 50 })
  (h3 : m.tens = { count := 10, denomination := 10 }) :
  totalValue m = 550 := by sorry

end NUMINAMATH_CALUDE_meghans_money_is_550_l3731_373187


namespace NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l3731_373156

-- Define a quadratic function
def quadratic (m n t : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + t

-- Define a fixed point
def is_fixed_point (m n t : ℝ) (x : ℝ) : Prop := quadratic m n t x = x

theorem fixed_points_of_specific_quadratic :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point 1 (-1) (-3) x1 ∧ is_fixed_point 1 (-1) (-3) x2 ∧ x1 = -1 ∧ x2 = 3 := by sorry

theorem min_value_of_ratio_sum :
  ∀ a : ℝ, a > 1 →
  ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧
  is_fixed_point 2 (-(3+a)) (a-1) x1 ∧
  is_fixed_point 2 (-(3+a)) (a-1) x2 →
  (x1 / x2 + x2 / x1 ≥ 8) ∧ (∃ a0 : ℝ, a0 > 1 ∧ ∃ x3 x4 : ℝ, x3 / x4 + x4 / x3 = 8) := by sorry

theorem range_of_a_for_always_fixed_point :
  ∀ a : ℝ, a ≠ 0 →
  (∀ b : ℝ, ∃ x : ℝ, is_fixed_point a (b+1) (b-1) x) ↔
  (a > 0 ∧ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l3731_373156


namespace NUMINAMATH_CALUDE_dress_sewing_time_l3731_373176

/-- The time Allison and Al worked together on sewing dresses -/
def timeWorkedTogether (allisonRate alRate : ℚ) (allisonAloneTime : ℚ) : ℚ :=
  (1 - allisonRate * allisonAloneTime) / (allisonRate + alRate)

theorem dress_sewing_time : 
  let allisonRate : ℚ := 1/9
  let alRate : ℚ := 1/12
  let allisonAloneTime : ℚ := 15/4
  timeWorkedTogether allisonRate alRate allisonAloneTime = 3 := by
sorry

end NUMINAMATH_CALUDE_dress_sewing_time_l3731_373176


namespace NUMINAMATH_CALUDE_millet_percentage_in_brand_A_l3731_373163

/-- The percentage of millet in Brand A -/
def millet_in_A : ℝ := 0.4

/-- The percentage of sunflower in Brand A -/
def sunflower_in_A : ℝ := 0.6

/-- The percentage of millet in Brand B -/
def millet_in_B : ℝ := 0.65

/-- The percentage of Brand A in the mix -/
def brand_A_in_mix : ℝ := 0.6

/-- The percentage of Brand B in the mix -/
def brand_B_in_mix : ℝ := 0.4

/-- The percentage of millet in the mix -/
def millet_in_mix : ℝ := 0.5

theorem millet_percentage_in_brand_A :
  millet_in_A * brand_A_in_mix + millet_in_B * brand_B_in_mix = millet_in_mix ∧
  millet_in_A + sunflower_in_A = 1 :=
by sorry

end NUMINAMATH_CALUDE_millet_percentage_in_brand_A_l3731_373163


namespace NUMINAMATH_CALUDE_sin_minus_cos_with_tan_one_third_l3731_373134

theorem sin_minus_cos_with_tan_one_third 
  (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_with_tan_one_third_l3731_373134


namespace NUMINAMATH_CALUDE_cars_between_black_and_white_l3731_373119

/-- Given a row of 20 cars, with a black car 16th from the right and a white car 11th from the left,
    the number of cars between the black and white cars is 5. -/
theorem cars_between_black_and_white :
  ∀ (total_cars : ℕ) (black_from_right : ℕ) (white_from_left : ℕ),
    total_cars = 20 →
    black_from_right = 16 →
    white_from_left = 11 →
    white_from_left - (total_cars - black_from_right + 1) - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cars_between_black_and_white_l3731_373119


namespace NUMINAMATH_CALUDE_cube_structure_ratio_l3731_373145

/-- A structure formed by joining unit cubes -/
structure CubeStructure where
  num_cubes : ℕ
  central_cube : Bool
  shared_faces : ℕ

/-- Calculate the volume of the cube structure -/
def volume (s : CubeStructure) : ℕ :=
  s.num_cubes

/-- Calculate the surface area of the cube structure -/
def surface_area (s : CubeStructure) : ℕ :=
  (s.num_cubes - 1) * 5

/-- The ratio of volume to surface area -/
def volume_to_surface_ratio (s : CubeStructure) : ℚ :=
  (volume s : ℚ) / (surface_area s : ℚ)

/-- Theorem stating the ratio of volume to surface area for the specific cube structure -/
theorem cube_structure_ratio :
  ∃ (s : CubeStructure),
    s.num_cubes = 8 ∧
    s.central_cube = true ∧
    s.shared_faces = 6 ∧
    volume_to_surface_ratio s = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_structure_ratio_l3731_373145


namespace NUMINAMATH_CALUDE_greatest_number_satisfying_conditions_l3731_373138

/-- A number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

/-- A number is composed of the square of two distinct prime factors -/
def is_product_of_two_distinct_prime_squares (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p^2 * q^2

/-- A number has an odd number of positive factors -/
def has_odd_number_of_factors (n : ℕ) : Prop :=
  Odd (Nat.card (Nat.divisors n))

/-- The main theorem -/
theorem greatest_number_satisfying_conditions : 
  (∀ n : ℕ, n < 200 → is_perfect_square n → 
    is_product_of_two_distinct_prime_squares n → 
    has_odd_number_of_factors n → n ≤ 196) ∧ 
  (196 < 200 ∧ is_perfect_square 196 ∧ 
    is_product_of_two_distinct_prime_squares 196 ∧ 
    has_odd_number_of_factors 196) := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_satisfying_conditions_l3731_373138


namespace NUMINAMATH_CALUDE_gcd_1043_2295_l3731_373165

theorem gcd_1043_2295 : Nat.gcd 1043 2295 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1043_2295_l3731_373165


namespace NUMINAMATH_CALUDE_inequality_proof_l3731_373107

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  (a^5 + b^5)/(a*b*(a+b)) + (b^5 + c^5)/(b*c*(b+c)) + (c^5 + a^5)/(c*a*(c+a)) ≥ 3*(a*b + b*c + c*a) - 2 ∧
  (a^5 + b^5)/(a*b*(a+b)) + (b^5 + c^5)/(b*c*(b+c)) + (c^5 + a^5)/(c*a*(c+a)) ≥ 6 - 5*(a*b + b*c + c*a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3731_373107


namespace NUMINAMATH_CALUDE_M_is_range_of_f_l3731_373194

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x^2}

-- Define the function f(x) = x^2
def f : ℝ → ℝ := λ x ↦ x^2

-- Theorem statement
theorem M_is_range_of_f : M = Set.range f := by sorry

end NUMINAMATH_CALUDE_M_is_range_of_f_l3731_373194


namespace NUMINAMATH_CALUDE_replacement_theorem_l3731_373118

/-- Calculates the percentage of chemicals in a solution after replacing part of it with a different solution -/
def resulting_solution_percentage (original_percentage : ℝ) (replacement_percentage : ℝ) (replaced_portion : ℝ) : ℝ :=
  let remaining_portion := 1 - replaced_portion
  let original_chemicals := original_percentage * remaining_portion
  let replacement_chemicals := replacement_percentage * replaced_portion
  (original_chemicals + replacement_chemicals) * 100

/-- Theorem stating that replacing half of an 80% solution with a 20% solution results in a 50% solution -/
theorem replacement_theorem :
  resulting_solution_percentage 0.8 0.2 0.5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_replacement_theorem_l3731_373118


namespace NUMINAMATH_CALUDE_sine_double_angle_special_l3731_373149

theorem sine_double_angle_special (α : Real) : 
  α ∈ Set.Ioo 0 (Real.pi / 2) → 
  Real.cos (α + Real.pi / 6) = 3 / 5 → 
  Real.sin (2 * α + Real.pi / 3) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sine_double_angle_special_l3731_373149


namespace NUMINAMATH_CALUDE_parabola_vertex_l3731_373191

/-- The vertex of the parabola y = (x+2)^2 - 1 is at the point (-2, -1) -/
theorem parabola_vertex (x y : ℝ) : 
  y = (x + 2)^2 - 1 → (∀ x' y', y' = (x' + 2)^2 - 1 → y ≤ y') → x = -2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3731_373191


namespace NUMINAMATH_CALUDE_find_m_l3731_373154

theorem find_m : ∃ m : ℕ, (1/5 : ℚ)^m * (1/4 : ℚ)^2 = 1/(10^4 : ℚ) ∧ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3731_373154


namespace NUMINAMATH_CALUDE_newton_family_mean_age_l3731_373114

theorem newton_family_mean_age :
  let ages : List ℝ := [6, 6, 9, 12]
  let mean := (ages.sum) / (ages.length)
  mean = 8.25 := by
sorry

end NUMINAMATH_CALUDE_newton_family_mean_age_l3731_373114


namespace NUMINAMATH_CALUDE_apple_juice_problem_l3731_373129

theorem apple_juice_problem (x y : ℝ) : 
  (x - 1 = y + 1) →  -- Equalizing condition
  (x + 9 = 30) →     -- First barrel full after transfer
  (y - 9 = 10) →     -- Second barrel one-third full after transfer
  (x = 21 ∧ y = 19 ∧ x + y = 40) := by
  sorry

end NUMINAMATH_CALUDE_apple_juice_problem_l3731_373129


namespace NUMINAMATH_CALUDE_inequality_solution_condition_l3731_373133

theorem inequality_solution_condition (a : ℝ) :
  (∃ x : ℝ, x ≥ a ∧ |x - a| + |2*x + 1| ≤ 2*a + x) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_condition_l3731_373133


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l3731_373175

theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 2 * (2 * r)
  (side_length ^ 2 : ℝ) = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l3731_373175


namespace NUMINAMATH_CALUDE_cost_not_proportional_cost_increases_linearly_l3731_373155

/-- Represents the cost of a telegram -/
def telegram_cost (a b n : ℝ) : ℝ := a + b * n

/-- The cost is not proportional to the number of words -/
theorem cost_not_proportional (a b : ℝ) (h : a ≠ 0) :
  ¬∃ k : ℝ, ∀ n : ℝ, telegram_cost a b n = k * n :=
sorry

/-- The cost increases linearly with the number of words -/
theorem cost_increases_linearly (a b : ℝ) (h : b > 0) :
  ∀ n₁ n₂ : ℝ, n₁ < n₂ → telegram_cost a b n₁ < telegram_cost a b n₂ :=
sorry

end NUMINAMATH_CALUDE_cost_not_proportional_cost_increases_linearly_l3731_373155


namespace NUMINAMATH_CALUDE_three_tribes_at_campfire_l3731_373111

/-- Represents a native at the campfire -/
structure Native where
  tribe : ℕ

/-- Represents the circle of natives around the campfire -/
def Campfire := Vector Native 7

/-- Check if a native tells the truth to their left neighbor -/
def tellsTruth (c : Campfire) (i : Fin 7) : Prop :=
  (c.get i).tribe = (c.get ((i + 1) % 7)).tribe →
    (∀ j : Fin 7, j ≠ i ∧ j ≠ ((i + 1) % 7) → (c.get j).tribe ≠ (c.get i).tribe)

/-- The main theorem: there are exactly 3 tribes represented at the campfire -/
theorem three_tribes_at_campfire (c : Campfire) 
  (h : ∀ i : Fin 7, tellsTruth c i) :
  ∃! n : ℕ, n = 3 ∧ (∀ t : ℕ, (∃ i : Fin 7, (c.get i).tribe = t) → t ≤ n) :=
sorry

end NUMINAMATH_CALUDE_three_tribes_at_campfire_l3731_373111


namespace NUMINAMATH_CALUDE_roof_ratio_l3731_373140

theorem roof_ratio (length width : ℝ) : 
  length * width = 784 →
  length - width = 42 →
  length / width = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_roof_ratio_l3731_373140


namespace NUMINAMATH_CALUDE_normal_commute_time_l3731_373109

/-- A worker's commute scenario -/
structure WorkerCommute where
  normal_speed : ℝ
  normal_distance : ℝ
  normal_time : ℝ
  inclined_speed : ℝ
  inclined_distance : ℝ
  inclined_time : ℝ

/-- The conditions of the worker's commute -/
def commute_conditions (w : WorkerCommute) : Prop :=
  w.inclined_speed = 3 / 4 * w.normal_speed ∧
  w.inclined_distance = 5 / 4 * w.normal_distance ∧
  w.inclined_time = w.normal_time + 20 ∧
  w.normal_distance = w.normal_speed * w.normal_time ∧
  w.inclined_distance = w.inclined_speed * w.inclined_time

/-- The theorem stating that under the given conditions, the normal commute time is 30 minutes -/
theorem normal_commute_time (w : WorkerCommute) 
  (h : commute_conditions w) : w.normal_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_normal_commute_time_l3731_373109


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l3731_373117

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) := by sorry

theorem negation_of_inequality :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l3731_373117
