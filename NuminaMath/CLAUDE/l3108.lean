import Mathlib

namespace marble_problem_l3108_310865

theorem marble_problem (x : ℝ) : 
  x + 3*x + 6*x + 24*x = 156 → x = 156/34 := by
  sorry

end marble_problem_l3108_310865


namespace remainder_equivalence_l3108_310895

theorem remainder_equivalence (x : ℤ) : x % 5 = 4 → x % 61 = 4 := by
  sorry

end remainder_equivalence_l3108_310895


namespace simplify_complex_fraction_l3108_310873

theorem simplify_complex_fraction (a b : ℝ) 
  (h1 : a + b ≠ 0) 
  (h2 : a - 2*b ≠ 0) 
  (h3 : a^2 - b^2 ≠ 0) 
  (h4 : a^2 - 4*a*b + 4*b^2 ≠ 0) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end simplify_complex_fraction_l3108_310873


namespace reptile_insect_consumption_l3108_310832

theorem reptile_insect_consumption :
  let num_geckos : ℕ := 5
  let num_lizards : ℕ := 3
  let num_chameleons : ℕ := 4
  let num_iguanas : ℕ := 2
  let gecko_consumption : ℕ := 6
  let lizard_consumption : ℝ := 2 * gecko_consumption
  let chameleon_consumption : ℝ := 3.5 * gecko_consumption
  let iguana_consumption : ℝ := 0.75 * gecko_consumption
  
  (num_geckos * gecko_consumption : ℝ) +
  (num_lizards : ℝ) * lizard_consumption +
  (num_chameleons : ℝ) * chameleon_consumption +
  (num_iguanas : ℝ) * iguana_consumption = 159
  := by sorry

end reptile_insect_consumption_l3108_310832


namespace same_color_probability_l3108_310835

/-- Represents the number of pairs of shoes -/
def num_pairs : ℕ := 5

/-- Represents the total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- Represents the number of shoes to select -/
def shoes_to_select : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
theorem same_color_probability : 
  (num_pairs : ℚ) / (total_shoes.choose shoes_to_select) = 1 / 9 := by
  sorry

end same_color_probability_l3108_310835


namespace platform_length_l3108_310810

/-- The length of a platform given train parameters -/
theorem platform_length
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (time_to_cross : ℝ)
  (h1 : train_length = 450)
  (h2 : train_speed_kmph = 126)
  (h3 : time_to_cross = 20)
  : ∃ (platform_length : ℝ), platform_length = 250 := by
  sorry

#check platform_length

end platform_length_l3108_310810


namespace scalar_multiplication_distributivity_l3108_310864

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem scalar_multiplication_distributivity
  (m n : ℝ) (a : V)
  (hm : m ≠ 0) (hn : n ≠ 0) (ha : a ≠ 0) :
  (m + n) • a = m • a + n • a :=
by sorry

end scalar_multiplication_distributivity_l3108_310864


namespace soccer_score_theorem_l3108_310870

/-- Represents the scores of a soccer player -/
structure SoccerScores where
  game6 : ℕ
  game7 : ℕ
  game8 : ℕ
  game9 : ℕ
  first6GamesTotal : ℕ
  game10 : ℕ

/-- The minimum number of points scored in the 10th game -/
def minGame10Score (s : SoccerScores) : Prop :=
  s.game10 = 13

/-- The given conditions of the problem -/
def problemConditions (s : SoccerScores) : Prop :=
  s.game6 = 18 ∧
  s.game7 = 25 ∧
  s.game8 = 15 ∧
  s.game9 = 22 ∧
  (s.first6GamesTotal + s.game6 + s.game7 + s.game8 + s.game9 + s.game10) / 10 >
    (s.first6GamesTotal + s.game6) / 6 ∧
  (s.first6GamesTotal + s.game6 + s.game7 + s.game8 + s.game9 + s.game10) / 10 > 20

theorem soccer_score_theorem (s : SoccerScores) :
  problemConditions s → minGame10Score s := by
  sorry

end soccer_score_theorem_l3108_310870


namespace geometric_mean_of_4_and_16_l3108_310877

theorem geometric_mean_of_4_and_16 (x : ℝ) :
  x ^ 2 = 4 * 16 → x = 8 ∨ x = -8 := by
  sorry

end geometric_mean_of_4_and_16_l3108_310877


namespace num_sequences_equals_binomial_remainder_of_m_mod_1000_l3108_310837

/-- The number of increasing sequences of 10 positive integers satisfying given conditions -/
def num_sequences : ℕ := sorry

/-- The upper bound for each term in the sequence -/
def upper_bound : ℕ := 2007

/-- The length of the sequence -/
def sequence_length : ℕ := 10

/-- Predicate to check if a sequence satisfies the required conditions -/
def valid_sequence (a : Fin sequence_length → ℕ) : Prop :=
  (∀ i j : Fin sequence_length, i ≤ j → a i ≤ a j) ∧
  (∀ i : Fin sequence_length, a i ≤ upper_bound) ∧
  (∀ i : Fin sequence_length, Even (a i - i.val))

theorem num_sequences_equals_binomial :
  num_sequences = Nat.choose 1008 sequence_length :=
sorry

theorem remainder_of_m_mod_1000 :
  1008 % 1000 = 8 :=
sorry

end num_sequences_equals_binomial_remainder_of_m_mod_1000_l3108_310837


namespace meet_at_starting_line_l3108_310894

theorem meet_at_starting_line (henry_time margo_time cameron_time : ℕ) 
  (henry_eq : henry_time = 7)
  (margo_eq : margo_time = 12)
  (cameron_eq : cameron_time = 9) :
  Nat.lcm (Nat.lcm henry_time margo_time) cameron_time = 252 := by
  sorry

end meet_at_starting_line_l3108_310894


namespace committee_formation_count_l3108_310890

theorem committee_formation_count (n : ℕ) (k : ℕ) : n = 30 → k = 5 →
  (n.choose 1) * ((n - 1).choose 1) * ((n - 2).choose (k - 2)) = 2850360 := by
  sorry

end committee_formation_count_l3108_310890


namespace right_triangle_test_l3108_310822

theorem right_triangle_test : 
  -- Option A
  (3 : ℝ)^2 + 4^2 = 5^2 ∧
  -- Option B
  (1 : ℝ)^2 + 2^2 = (Real.sqrt 5)^2 ∧
  -- Option C
  (2 : ℝ)^2 + (2 * Real.sqrt 3)^2 ≠ 3^2 ∧
  -- Option D
  (1 : ℝ)^2 + (Real.sqrt 3)^2 = 2^2 :=
by sorry

end right_triangle_test_l3108_310822


namespace part_1_part_2_part_3_l3108_310841

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Part 1
theorem part_1 (a : ℝ) (h : circle_C a (a+1)) : 
  let P : ℝ × ℝ := (a, a+1)
  ∃ (d s : ℝ), d = Real.sqrt 40 ∧ s = 1/3 ∧ 
    d^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ∧
    s = (P.2 - Q.2) / (P.1 - Q.1) := by sorry

-- Part 2
theorem part_2 (M : ℝ × ℝ) (h : circle_C M.1 M.2) :
  2 * Real.sqrt 2 ≤ Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ∧
  Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ≤ 6 * Real.sqrt 2 := by sorry

-- Part 3
theorem part_3 (m n : ℝ) (h : m^2 + n^2 - 4*m - 14*n + 45 = 0) :
  2 - Real.sqrt 3 ≤ (n - 3) / (m + 2) ∧ (n - 3) / (m + 2) ≤ 2 + Real.sqrt 3 := by sorry

end part_1_part_2_part_3_l3108_310841


namespace triangle_area_l3108_310887

/-- The area of a right triangle with vertices at (0, 0), (0, 10), and (-10, 0) is 50 square units,
    given that the points (-3, 7) and (-7, 3) lie on its hypotenuse. -/
theorem triangle_area : 
  let p1 : ℝ × ℝ := (-3, 7)
  let p2 : ℝ × ℝ := (-7, 3)
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (0, 10)
  let v3 : ℝ × ℝ := (-10, 0)
  (p1.1 - p2.1) / (p1.2 - p2.2) = 1 →  -- Slope of the line through p1 and p2 is 1
  (∃ t : ℝ, v2 = p1 + t • (1, 1)) →  -- v2 lies on the line through p1 with slope 1
  (∃ t : ℝ, v3 = p2 + t • (1, 1)) →  -- v3 lies on the line through p2 with slope 1
  (1/2) * (v2.2 - v1.2) * (v1.1 - v3.1) = 50 := by
sorry


end triangle_area_l3108_310887


namespace cube_root_of_four_sixth_powers_l3108_310875

theorem cube_root_of_four_sixth_powers (x : ℝ) :
  x = (4^6 + 4^6 + 4^6 + 4^6)^(1/3) → x = 16 * (4^(1/3)) := by
  sorry

end cube_root_of_four_sixth_powers_l3108_310875


namespace sum_of_greater_is_greater_l3108_310896

theorem sum_of_greater_is_greater (a b c d : ℝ) : a > b → c > d → a + c > b + d := by
  sorry

end sum_of_greater_is_greater_l3108_310896


namespace total_students_correct_l3108_310897

/-- The number of students who tried out for the school's trivia teams. -/
def total_students : ℕ := 64

/-- The number of students who didn't get picked for the team. -/
def not_picked : ℕ := 36

/-- The number of groups the picked students were divided into. -/
def num_groups : ℕ := 4

/-- The number of students in each group of picked students. -/
def students_per_group : ℕ := 7

/-- Theorem stating that the total number of students who tried out is correct. -/
theorem total_students_correct :
  total_students = not_picked + num_groups * students_per_group :=
by sorry

end total_students_correct_l3108_310897


namespace red_yellow_difference_l3108_310818

/-- Represents the number of marbles of each color in a bowl. -/
structure MarbleCount where
  total : ℕ
  yellow : ℕ
  blue : ℕ
  red : ℕ

/-- Represents the ratio of blue to red marbles. -/
structure BlueRedRatio where
  blue : ℕ
  red : ℕ

/-- Given the total number of marbles, the number of yellow marbles, and the ratio of blue to red marbles,
    proves that there are 3 more red marbles than yellow marbles. -/
theorem red_yellow_difference (m : MarbleCount) (ratio : BlueRedRatio) : 
  m.total = 19 → 
  m.yellow = 5 → 
  m.blue + m.red = m.total - m.yellow →
  m.blue * ratio.red = m.red * ratio.blue →
  ratio.blue = 3 →
  ratio.red = 4 →
  m.red - m.yellow = 3 := by
  sorry

end red_yellow_difference_l3108_310818


namespace b_initial_investment_l3108_310860

/-- Represents the business investment scenario --/
structure BusinessInvestment where
  a_initial : ℕ  -- A's initial investment
  b_initial : ℕ  -- B's initial investment
  initial_duration : ℕ  -- Initial duration in months
  a_withdrawal : ℕ  -- Amount A withdraws after initial duration
  b_addition : ℕ  -- Amount B adds after initial duration
  total_profit : ℕ  -- Total profit at the end of the year
  a_profit_share : ℕ  -- A's share of the profit

/-- Calculates the total investment of A --/
def total_investment_a (bi : BusinessInvestment) : ℕ :=
  bi.a_initial * bi.initial_duration + (bi.a_initial - bi.a_withdrawal) * (12 - bi.initial_duration)

/-- Calculates the total investment of B --/
def total_investment_b (bi : BusinessInvestment) : ℕ :=
  bi.b_initial * bi.initial_duration + (bi.b_initial + bi.b_addition) * (12 - bi.initial_duration)

/-- Theorem stating that given the conditions, B's initial investment was 4000 Rs --/
theorem b_initial_investment (bi : BusinessInvestment) 
  (h1 : bi.a_initial = 3000)
  (h2 : bi.initial_duration = 8)
  (h3 : bi.a_withdrawal = 1000)
  (h4 : bi.b_addition = 1000)
  (h5 : bi.total_profit = 840)
  (h6 : bi.a_profit_share = 320)
  : bi.b_initial = 4000 := by
  sorry

end b_initial_investment_l3108_310860


namespace visited_both_countries_l3108_310855

theorem visited_both_countries (total : ℕ) (iceland : ℕ) (norway : ℕ) (neither : ℕ) : 
  total = 100 → iceland = 55 → norway = 43 → neither = 63 → 
  (total - neither) = (iceland + norway - (iceland + norway - (total - neither))) := by
  sorry

end visited_both_countries_l3108_310855


namespace complex_fraction_simplification_l3108_310898

theorem complex_fraction_simplification :
  (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I :=
by sorry

end complex_fraction_simplification_l3108_310898


namespace product_divisible_by_twelve_l3108_310830

theorem product_divisible_by_twelve (a b c d : ℤ) : 
  ∃ k : ℤ, (b - a) * (c - a) * (d - a) * (b - c) * (d - c) * (d - b) = 12 * k := by
  sorry

end product_divisible_by_twelve_l3108_310830


namespace ladybugs_without_spots_l3108_310807

def total_ladybugs : ℕ := 67082
def spotted_ladybugs : ℕ := 12170

theorem ladybugs_without_spots : 
  total_ladybugs - spotted_ladybugs = 54912 :=
by sorry

end ladybugs_without_spots_l3108_310807


namespace two_roses_more_expensive_than_three_carnations_l3108_310861

/-- Price of a single rose in yuan -/
def rose_price : ℝ := sorry

/-- Price of a single carnation in yuan -/
def carnation_price : ℝ := sorry

/-- The total price of 6 roses and 3 carnations is greater than 24 yuan -/
axiom condition1 : 6 * rose_price + 3 * carnation_price > 24

/-- The total price of 4 roses and 5 carnations is less than 22 yuan -/
axiom condition2 : 4 * rose_price + 5 * carnation_price < 22

/-- Theorem: The price of 2 roses is higher than the price of 3 carnations -/
theorem two_roses_more_expensive_than_three_carnations :
  2 * rose_price > 3 * carnation_price := by sorry

end two_roses_more_expensive_than_three_carnations_l3108_310861


namespace simplify_fraction_l3108_310851

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (12 * x * y^3) / (9 * x^3 * y^2) = 16 / 27 := by
  sorry

end simplify_fraction_l3108_310851


namespace simon_lego_count_l3108_310805

theorem simon_lego_count (kent_legos : ℕ) (bruce_extra : ℕ) (simon_percentage : ℚ) :
  kent_legos = 40 →
  bruce_extra = 20 →
  simon_percentage = 1/5 →
  (kent_legos + bruce_extra) * (1 + simon_percentage) = 72 :=
by sorry

end simon_lego_count_l3108_310805


namespace negation_of_all_positive_square_plus_one_l3108_310808

theorem negation_of_all_positive_square_plus_one (q : Prop) : 
  (q ↔ ∀ x : ℝ, x^2 + 1 > 0) →
  (¬q ↔ ∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end negation_of_all_positive_square_plus_one_l3108_310808


namespace arccos_sin_three_l3108_310879

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - π / 2 := by
  sorry

end arccos_sin_three_l3108_310879


namespace harmonic_geometric_log_ratio_l3108_310869

/-- Given distinct positive real numbers a, b, c forming a harmonic sequence,
    and their logarithms forming a geometric sequence, 
    prove that the common ratio of the geometric sequence is a non-1 cube root of unity. -/
theorem harmonic_geometric_log_ratio 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hharmseq : (1 / b) = (1 / (2 : ℝ)) * ((1 / a) + (1 / c)))
  (hgeomseq : ∃ r : ℂ, (Complex.log b / Complex.log a) = r ∧ 
                       (Complex.log c / Complex.log b) = r ∧ 
                       (Complex.log a / Complex.log c) = r) :
  ∃ r : ℂ, r^3 = 1 ∧ r ≠ 1 ∧ 
    ((Complex.log b / Complex.log a) = r ∧ 
     (Complex.log c / Complex.log b) = r ∧ 
     (Complex.log a / Complex.log c) = r) := by
  sorry

end harmonic_geometric_log_ratio_l3108_310869


namespace existence_of_bounded_irreducible_factorization_l3108_310881

def is_irreducible (S : Set ℕ) (x : ℕ) : Prop :=
  x ∈ S ∧ ∀ y z : ℕ, y ∈ S → z ∈ S → x = y * z → (y = 1 ∨ z = 1)

theorem existence_of_bounded_irreducible_factorization 
  (a b : ℕ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_gcd : ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ Nat.gcd a b ∧ q ∣ Nat.gcd a b) :
  ∃ t : ℕ, ∀ x ∈ {n : ℕ | n > 0 ∧ n % b = a % b}, 
    ∃ (factors : List ℕ), 
      (∀ f ∈ factors, is_irreducible {n : ℕ | n > 0 ∧ n % b = a % b} f) ∧
      (factors.prod = x) ∧
      (factors.length ≤ t) :=
by sorry

end existence_of_bounded_irreducible_factorization_l3108_310881


namespace tomorrow_is_saturday_l3108_310847

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Returns the day n days after the given day -/
def addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (addDays d m)

/-- The main theorem -/
theorem tomorrow_is_saturday 
  (h : addDays (addDays DayOfWeek.Wednesday 2) 5 = DayOfWeek.Monday) : 
  nextDay (addDays DayOfWeek.Wednesday 2) = DayOfWeek.Saturday := by
  sorry


end tomorrow_is_saturday_l3108_310847


namespace smallest_constant_for_inequality_l3108_310827

/-- The smallest possible real constant C such that 
    |x^3 + y^3 + z^3 + 1| ≤ C|x^5 + y^5 + z^5 + 1| 
    holds for all real x, y, z satisfying x + y + z = -1 -/
theorem smallest_constant_for_inequality :
  ∃ (C : ℝ), C = 1539 / 1449 ∧
  (∀ (x y z : ℝ), x + y + z = -1 →
    |x^3 + y^3 + z^3 + 1| ≤ C * |x^5 + y^5 + z^5 + 1|) ∧
  (∀ (C' : ℝ), C' < C →
    ∃ (x y z : ℝ), x + y + z = -1 ∧
      |x^3 + y^3 + z^3 + 1| > C' * |x^5 + y^5 + z^5 + 1|) :=
by sorry

end smallest_constant_for_inequality_l3108_310827


namespace halfway_fraction_l3108_310842

theorem halfway_fraction (a b : ℚ) (ha : a = 1/6) (hb : b = 1/4) :
  (a + b) / 2 = 5/24 := by
  sorry

end halfway_fraction_l3108_310842


namespace neither_sufficient_nor_necessary_l3108_310811

theorem neither_sufficient_nor_necessary (p q : Prop) :
  ¬(((p ∨ q) → ¬(p ∧ q)) ∧ (¬(p ∧ q) → (p ∨ q))) :=
by sorry

end neither_sufficient_nor_necessary_l3108_310811


namespace cube_difference_positive_l3108_310833

theorem cube_difference_positive (a b : ℝ) : a > b → a^3 - b^3 > 0 := by
  sorry

end cube_difference_positive_l3108_310833


namespace symmetry_of_point_l3108_310862

/-- Given a point P in 3D space, this function returns the point symmetric to P with respect to the y-axis --/
def symmetry_y_axis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := P
  (-x, y, -z)

/-- Theorem stating that the point symmetric to (2, -3, -5) with respect to the y-axis is (-2, -3, 5) --/
theorem symmetry_of_point :
  symmetry_y_axis (2, -3, -5) = (-2, -3, 5) := by
  sorry

end symmetry_of_point_l3108_310862


namespace petya_vasya_meeting_l3108_310825

/-- Represents the meeting point of two people walking towards each other along a line of lamps -/
def meeting_point (total_lamps : ℕ) (p_start p_pos : ℕ) (v_start v_pos : ℕ) : ℕ :=
  let p_traveled := p_pos - p_start
  let v_traveled := v_start - v_pos
  let remaining_distance := v_pos - p_pos
  let total_intervals := total_lamps - 1
  let p_speed := p_traveled
  let v_speed := v_traveled
  p_pos + (remaining_distance * p_speed) / (p_speed + v_speed)

/-- Theorem stating that Petya and Vasya meet at lamp 64 -/
theorem petya_vasya_meeting :
  let total_lamps : ℕ := 100
  let petya_start : ℕ := 1
  let vasya_start : ℕ := 100
  let petya_position : ℕ := 22
  let vasya_position : ℕ := 88
  meeting_point total_lamps petya_start petya_position vasya_start vasya_position = 64 := by
  sorry

#eval meeting_point 100 1 22 100 88

end petya_vasya_meeting_l3108_310825


namespace equal_bisecting_diagonals_implies_rectangle_bisecting_diagonals_implies_parallelogram_rhombus_equal_diagonals_implies_square_l3108_310853

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def diagonals_bisect_each_other (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def diagonals_perpendicular (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem equal_bisecting_diagonals_implies_rectangle 
  (q : Quadrilateral) 
  (h1 : has_equal_diagonals q) 
  (h2 : diagonals_bisect_each_other q) : 
  is_rectangle q := by sorry

-- Theorem 2
theorem bisecting_diagonals_implies_parallelogram 
  (q : Quadrilateral) 
  (h : diagonals_bisect_each_other q) : 
  is_parallelogram q := by sorry

-- Theorem 3
theorem rhombus_equal_diagonals_implies_square 
  (q : Quadrilateral) 
  (h1 : is_rhombus q) 
  (h2 : has_equal_diagonals q) : 
  is_square q := by sorry

end equal_bisecting_diagonals_implies_rectangle_bisecting_diagonals_implies_parallelogram_rhombus_equal_diagonals_implies_square_l3108_310853


namespace minimize_reciprocal_sum_l3108_310886

theorem minimize_reciprocal_sum (a b : ℕ+) (h : 4 * a.val + b.val = 30) :
  (1 : ℚ) / a.val + (1 : ℚ) / b.val ≥ (1 : ℚ) / 5 + (1 : ℚ) / 10 :=
sorry

end minimize_reciprocal_sum_l3108_310886


namespace rose_bush_cost_l3108_310846

def total_cost : ℕ := 4100
def num_rose_bushes : ℕ := 20
def gardener_hourly_rate : ℕ := 30
def gardener_hours_per_day : ℕ := 5
def gardener_days : ℕ := 4
def soil_volume : ℕ := 100
def soil_cost_per_unit : ℕ := 5

theorem rose_bush_cost : 
  (total_cost - (gardener_hourly_rate * gardener_hours_per_day * gardener_days) - 
   (soil_volume * soil_cost_per_unit)) / num_rose_bushes = 150 := by
  sorry

end rose_bush_cost_l3108_310846


namespace complex_magnitude_example_l3108_310820

theorem complex_magnitude_example : Complex.abs (11 + 18 * Complex.I + 4 - 3 * Complex.I) = 15 * Real.sqrt 2 := by
  sorry

end complex_magnitude_example_l3108_310820


namespace worker_B_time_proof_l3108_310817

/-- The time taken by worker B to complete a task, given that worker A is five times as efficient and takes 15 days less than B. -/
def time_taken_by_B : ℝ := 18.75

/-- The efficiency ratio of worker A to worker B -/
def efficiency_ratio : ℝ := 5

/-- The difference in days between the time taken by B and A to complete the task -/
def time_difference : ℝ := 15

theorem worker_B_time_proof :
  ∀ (rate_B : ℝ) (time_B : ℝ),
    rate_B > 0 →
    time_B > 0 →
    efficiency_ratio * rate_B * (time_B - time_difference) = rate_B * time_B →
    time_B = time_taken_by_B :=
by sorry

end worker_B_time_proof_l3108_310817


namespace parallelogram_iff_opposite_sides_equal_l3108_310849

structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

def opposite_sides_equal (q : Quadrilateral) : Prop :=
  (q.vertices 0 = q.vertices 2) ∧ (q.vertices 1 = q.vertices 3)

def is_parallelogram (q : Quadrilateral) : Prop :=
  ∃ (v : ℝ × ℝ), (q.vertices 1 - q.vertices 0 = v) ∧ (q.vertices 2 - q.vertices 3 = v) ∧
                 (q.vertices 3 - q.vertices 0 = v) ∧ (q.vertices 2 - q.vertices 1 = v)

theorem parallelogram_iff_opposite_sides_equal (q : Quadrilateral) :
  is_parallelogram q ↔ opposite_sides_equal q := by sorry

end parallelogram_iff_opposite_sides_equal_l3108_310849


namespace suv_distance_theorem_l3108_310831

/-- Represents the maximum distance an SUV can travel on 24 gallons of gas -/
def max_distance (x : ℝ) : ℝ :=
  1.824 * x + 292.8 - 2.928 * x

/-- Theorem stating the maximum distance formula for the SUV -/
theorem suv_distance_theorem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 100) :
  max_distance x = 7.6 * (x / 100) * 24 + 12.2 * ((100 - x) / 100) * 24 :=
by sorry

end suv_distance_theorem_l3108_310831


namespace simplification_and_exponent_sum_l3108_310815

-- Define the expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^9 * z^14) ^ (1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z^4 * (10 * x^2 * z^2) ^ (1/3)

-- Theorem statement
theorem simplification_and_exponent_sum :
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 →
  (original_expression x y z = simplified_expression x y z) ∧
  (1 + 1 + 4 = 6) := by
  sorry

end simplification_and_exponent_sum_l3108_310815


namespace prime_difference_values_l3108_310819

theorem prime_difference_values (p q : ℕ) (n : ℕ+) 
  (h_p : Nat.Prime p) (h_q : Nat.Prime q) 
  (h_eq : (p : ℚ) / (p + 1) + (q + 1 : ℚ) / q = (2 * n) / (n + 2)) :
  q - p ∈ ({2, 3, 5} : Set ℕ) := by
  sorry

end prime_difference_values_l3108_310819


namespace exists_m_divisible_by_1988_l3108_310899

def f (x : ℕ) : ℕ := 3 * x + 2

theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ, (3^100 * m + (3^100 - 1)) % 1988 = 0 := by
sorry

end exists_m_divisible_by_1988_l3108_310899


namespace game_result_l3108_310814

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 5 = 0 then 7
  else if n % 3 = 0 then 3
  else 0

def charlie_rolls : List ℕ := [6, 2, 3, 5]
def dana_rolls : List ℕ := [5, 3, 1, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result : total_points charlie_rolls * total_points dana_rolls = 36 := by
  sorry

end game_result_l3108_310814


namespace identity_proof_l3108_310891

theorem identity_proof (a b c d x y z u : ℝ) :
  (a*x + b*y + c*z + d*u)^2 + (b*x + c*y + d*z + a*u)^2 + (c*x + d*y + a*z + b*u)^2 + (d*x + a*y + b*z + c*u)^2
  = (d*x + c*y + b*z + a*u)^2 + (c*x + b*y + a*z + d*u)^2 + (b*x + a*y + d*z + c*u)^2 + (a*x + d*y + c*z + b*u)^2 := by
  sorry

end identity_proof_l3108_310891


namespace distance_on_quadratic_curve_l3108_310871

/-- The distance between two points on a quadratic curve -/
theorem distance_on_quadratic_curve 
  (a b c p r : ℝ) : 
  let q := a * p^2 + b * p + c
  let s := a * r^2 + b * r + c
  Real.sqrt ((r - p)^2 + (s - q)^2) = |r - p| * Real.sqrt (1 + (a * (r + p) + b)^2) :=
by sorry

end distance_on_quadratic_curve_l3108_310871


namespace parabola_focus_l3108_310829

theorem parabola_focus (p : ℝ) :
  4 * p = 1/4 → (0, 1/(16 : ℝ)) = (0, p) := by sorry

end parabola_focus_l3108_310829


namespace geometric_sequence_sum_l3108_310824

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of a geometric sequence with first term 1/2 and common ratio 1/3 -/
theorem geometric_sequence_sum : 
  geometric_sum (1/2) (1/3) 8 = 4920/6561 := by
  sorry

#eval geometric_sum (1/2) (1/3) 8

end geometric_sequence_sum_l3108_310824


namespace max_value_3x_4y_l3108_310823

theorem max_value_3x_4y (x y : ℝ) : 
  x^2 + y^2 = 14*x + 6*y + 6 → (3*x + 4*y ≤ 73) := by
  sorry

end max_value_3x_4y_l3108_310823


namespace work_completion_time_l3108_310801

/-- Represents the time taken to complete a work when one worker is assisted by two others on alternate days -/
def time_to_complete (time_a time_b time_c : ℝ) : ℝ :=
  2 * 4

/-- Theorem stating that if A can do a work in 11 days, B in 20 days, and C in 55 days,
    and A is assisted by B and C on alternate days, then the work can be completed in 8 days -/
theorem work_completion_time (time_a time_b time_c : ℝ)
  (ha : time_a = 11)
  (hb : time_b = 20)
  (hc : time_c = 55) :
  time_to_complete time_a time_b time_c = 8 := by
  sorry

#eval time_to_complete 11 20 55

end work_completion_time_l3108_310801


namespace abs_diff_positive_iff_not_equal_l3108_310800

theorem abs_diff_positive_iff_not_equal (x : ℝ) : x ≠ 3 ↔ |x - 3| > 0 := by sorry

end abs_diff_positive_iff_not_equal_l3108_310800


namespace quadratic_equation_solution_l3108_310806

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = -7*x ↔ x = 0 ∨ x = -7 := by sorry

end quadratic_equation_solution_l3108_310806


namespace inheritance_calculation_l3108_310884

/-- The original inheritance amount -/
def inheritance : ℝ := sorry

/-- The state tax rate -/
def state_tax_rate : ℝ := 0.15

/-- The federal tax rate -/
def federal_tax_rate : ℝ := 0.25

/-- The total tax paid -/
def total_tax_paid : ℝ := 18000

theorem inheritance_calculation : 
  state_tax_rate * inheritance + 
  federal_tax_rate * (1 - state_tax_rate) * inheritance = 
  total_tax_paid := by sorry

end inheritance_calculation_l3108_310884


namespace max_distance_ellipse_circle_l3108_310848

/-- The maximum distance between a point on the ellipse x²/9 + y² = 1
    and a point on the circle (x-4)² + y² = 1 is 8 -/
theorem max_distance_ellipse_circle : 
  ∃ (max_dist : ℝ),
    max_dist = 8 ∧
    ∀ (P Q : ℝ × ℝ),
      (P.1^2 / 9 + P.2^2 = 1) →
      ((Q.1 - 4)^2 + Q.2^2 = 1) →
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ max_dist :=
by sorry

end max_distance_ellipse_circle_l3108_310848


namespace linear_equation_condition_l3108_310863

/-- A linear equation in two variables x and y of the form mx + 3y = 4x - 1 -/
def linear_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x + 3 * y = 4 * x - 1

/-- The condition for the equation to be linear in two variables -/
def is_linear_in_two_variables (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y, linear_equation m x y ↔ a * x + b * y = c

theorem linear_equation_condition (m : ℝ) :
  is_linear_in_two_variables m ↔ m ≠ 4 :=
sorry

end linear_equation_condition_l3108_310863


namespace greatest_integer_jo_l3108_310828

theorem greatest_integer_jo (n : ℕ) : 
  n > 0 ∧ 
  n < 150 ∧ 
  ∃ k : ℕ, n + 2 = 9 * k ∧ 
  ∃ l : ℕ, n + 4 = 8 * l →
  n ≤ 146 ∧ 
  ∃ m : ℕ, 146 > 0 ∧ 
  146 < 150 ∧ 
  146 + 2 = 9 * m ∧ 
  ∃ p : ℕ, 146 + 4 = 8 * p :=
by sorry

end greatest_integer_jo_l3108_310828


namespace ellipse_equation_and_max_slope_product_l3108_310834

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a line in the form x = my + 1 -/
structure Line where
  m : ℝ

/-- Calculates the product of slopes of the three sides of a triangle formed by
    two points on an ellipse and a fixed point -/
def slope_product (e : Ellipse) (l : Line) (p : ℝ × ℝ) : ℝ :=
  sorry

theorem ellipse_equation_and_max_slope_product 
  (e : Ellipse) (p : ℝ × ℝ) (h_p_on_ellipse : p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1)
  (h_p : p = (1, 3/2)) (h_eccentricity : Real.sqrt (e.a^2 - e.b^2) / e.a = 1/2) :
  (∃ (e' : Ellipse), e'.a = 2 ∧ e'.b = Real.sqrt 3 ∧
    (∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1 ↔ x^2 / e'.a^2 + y^2 / e'.b^2 = 1)) ∧
  (∃ (max_t : ℝ), max_t = 9/64 ∧
    ∀ (l : Line), slope_product e' l p ≤ max_t) :=
sorry

end ellipse_equation_and_max_slope_product_l3108_310834


namespace greatest_divisor_with_remainders_l3108_310840

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  (∃ (q₁ : ℕ), 4351 = q₁ * n + 8) ∧
  (∃ (q₂ : ℕ), 5161 = q₂ * n + 10) ∧
  (∃ (q₃ : ℕ), 6272 = q₃ * n + 12) ∧
  (∃ (q₄ : ℕ), 7383 = q₄ * n + 14) ∧
  ∀ (m : ℕ), m > 0 →
    (∃ (r₁ : ℕ), 4351 = r₁ * m + 8) →
    (∃ (r₂ : ℕ), 5161 = r₂ * m + 10) →
    (∃ (r₃ : ℕ), 6272 = r₃ * m + 12) →
    (∃ (r₄ : ℕ), 7383 = r₄ * m + 14) →
    m ≤ n :=
by sorry

end greatest_divisor_with_remainders_l3108_310840


namespace f_decreasing_l3108_310880

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin x else -Real.sin x

theorem f_decreasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (-(Real.pi / 2) + 2 * k * Real.pi) ((Real.pi / 2) + 2 * k * Real.pi)) :=
sorry

end f_decreasing_l3108_310880


namespace jeff_peanut_butter_amount_l3108_310845

/-- The amount of peanut butter in ounces for each jar size -/
def jar_sizes : List Nat := [16, 28, 40]

/-- The total number of jars Jeff has -/
def total_jars : Nat := 9

/-- Theorem stating that Jeff has 252 ounces of peanut butter -/
theorem jeff_peanut_butter_amount :
  (total_jars / jar_sizes.length) * (jar_sizes.sum) = 252 := by
  sorry

#check jeff_peanut_butter_amount

end jeff_peanut_butter_amount_l3108_310845


namespace vocational_students_form_valid_set_l3108_310858

-- Define the universe of discourse
def Student : Type := String

-- Define the properties
def isDefinite (s : Set Student) : Prop := sorry
def isDistinct (s : Set Student) : Prop := sorry
def isUnordered (s : Set Student) : Prop := sorry

-- Define the sets corresponding to each option
def tallStudents : Set Student := sorry
def vocationalStudents : Set Student := sorry
def goodStudents : Set Student := sorry
def lushTrees : Set Student := sorry

-- Define what makes a valid set
def isValidSet (s : Set Student) : Prop :=
  isDefinite s ∧ isDistinct s ∧ isUnordered s

-- Theorem statement
theorem vocational_students_form_valid_set :
  isValidSet vocationalStudents ∧
  ¬isValidSet tallStudents ∧
  ¬isValidSet goodStudents ∧
  ¬isValidSet lushTrees :=
sorry

end vocational_students_form_valid_set_l3108_310858


namespace gcd_8251_6105_l3108_310804

theorem gcd_8251_6105 : Int.gcd 8251 6105 = 39 := by
  sorry

end gcd_8251_6105_l3108_310804


namespace g_minus_one_eq_zero_l3108_310843

/-- The polynomial function g(x) -/
def g (s : ℝ) (x : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

/-- Theorem stating that g(-1) = 0 when s = -4 -/
theorem g_minus_one_eq_zero : g (-4) (-1) = 0 := by
  sorry

end g_minus_one_eq_zero_l3108_310843


namespace parabola_coefficient_l3108_310878

/-- Given a parabola y = ax^2 + bx + c with vertex (q/2, q/2) and y-intercept (0, -2q),
    where q ≠ 0, prove that b = 10 -/
theorem parabola_coefficient (a b c q : ℝ) (h_q : q ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (q/2, q/2) = (-(b / (2 * a)), a * (-(b / (2 * a)))^2 + b * (-(b / (2 * a))) + c) →
  c = -2 * q →
  b = 10 := by
sorry

end parabola_coefficient_l3108_310878


namespace product_remainder_zero_l3108_310857

theorem product_remainder_zero : 
  (1296 * 1444 * 1700 * 1875) % 7 = 0 := by
  sorry

end product_remainder_zero_l3108_310857


namespace third_height_less_than_30_l3108_310854

/-- Given a triangle with two heights of 12 and 20, prove that the third height is less than 30. -/
theorem third_height_less_than_30 
  (h_a h_b h_c : ℝ) 
  (triangle_exists : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (height_a : h_a = 12)
  (height_b : h_b = 20) :
  h_c < 30 := by
sorry

end third_height_less_than_30_l3108_310854


namespace nursery_paintable_area_l3108_310882

/-- Calculates the total paintable wall area for three identical rooms -/
def totalPaintableArea (length width height : ℝ) (unpaintableArea : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let paintableAreaPerRoom := wallArea - unpaintableArea
  3 * paintableAreaPerRoom

/-- Theorem stating that the total paintable area for three rooms with given dimensions is 1200 sq ft -/
theorem nursery_paintable_area :
  totalPaintableArea 14 11 9 50 = 1200 := by
  sorry

end nursery_paintable_area_l3108_310882


namespace sum_of_binomial_coeffs_l3108_310812

-- Define the binomial coefficient
def binomial_coeff (n m : ℕ) : ℕ := sorry

-- State the combinatorial identity
axiom combinatorial_identity (n m : ℕ) : 
  binomial_coeff (n + 1) m = binomial_coeff n (m - 1) + binomial_coeff n m

-- State the theorem to be proved
theorem sum_of_binomial_coeffs :
  binomial_coeff 7 4 + binomial_coeff 7 5 + binomial_coeff 8 6 = binomial_coeff 9 6 := by
  sorry

end sum_of_binomial_coeffs_l3108_310812


namespace coin_flips_l3108_310821

/-- The number of times a coin is flipped -/
def n : ℕ := sorry

/-- The probability of getting heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting heads on the first 4 flips and not heads on the last flip -/
def p_event : ℚ := 1/32

theorem coin_flips : 
  p_heads = 1/2 → 
  p_event = (p_heads ^ 4) * ((1 - p_heads) ^ 1) * (p_heads ^ (n - 5)) → 
  n = 9 := by sorry

end coin_flips_l3108_310821


namespace tomato_pick_ratio_l3108_310874

/-- Represents the number of tomatoes picked in each week and the remaining tomatoes -/
structure TomatoPicks where
  initial : ℕ
  first_week : ℕ
  second_week : ℕ
  third_week : ℕ
  remaining : ℕ

/-- Calculates the ratio of tomatoes picked in the third week to the second week -/
def pick_ratio (picks : TomatoPicks) : ℚ :=
  picks.third_week / picks.second_week

/-- Theorem stating the ratio of tomatoes picked in the third week to the second week -/
theorem tomato_pick_ratio : 
  ∀ (picks : TomatoPicks), 
  picks.initial = 100 ∧ 
  picks.first_week = picks.initial / 4 ∧
  picks.second_week = 20 ∧
  picks.remaining = 15 ∧
  picks.initial = picks.first_week + picks.second_week + picks.third_week + picks.remaining
  → pick_ratio picks = 2 := by
  sorry

#check tomato_pick_ratio

end tomato_pick_ratio_l3108_310874


namespace expression_simplification_l3108_310866

theorem expression_simplification :
  ((3 + 4 + 5 + 7) / 3) + ((3 * 6 + 9) / 4) = 157 / 12 := by
  sorry

end expression_simplification_l3108_310866


namespace find_f_2022_l3108_310836

/-- A function satisfying the given condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3

/-- The theorem to prove -/
theorem find_f_2022 (f : ℝ → ℝ) (h : special_function f) (h1 : f 1 = 5) (h4 : f 4 = 2) :
  f 2022 = -2016 := by
  sorry


end find_f_2022_l3108_310836


namespace max_power_of_five_equals_three_l3108_310839

/-- The number of divisors of a positive integer -/
noncomputable def num_divisors (n : ℕ+) : ℕ := sorry

/-- The greatest integer j such that 5^j divides n -/
noncomputable def max_power_of_five (n : ℕ+) : ℕ := sorry

theorem max_power_of_five_equals_three (n : ℕ+) 
  (h1 : num_divisors n = 72)
  (h2 : num_divisors (5 * n) = 90) :
  max_power_of_five n = 3 := by sorry

end max_power_of_five_equals_three_l3108_310839


namespace stream_speed_l3108_310885

/-- Given a boat traveling downstream, this theorem proves the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 24 →
  downstream_distance = 84 →
  downstream_time = 3 →
  ∃ stream_speed : ℝ, stream_speed = 4 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by sorry

end stream_speed_l3108_310885


namespace six_by_six_grid_squares_l3108_310892

/-- The number of squares of a given size in a 6x6 grid -/
def squares_of_size (n : Nat) : Nat :=
  (7 - n) * (7 - n)

/-- The total number of squares in a 6x6 grid -/
def total_squares : Nat :=
  (squares_of_size 1) + (squares_of_size 2) + (squares_of_size 3) + 
  (squares_of_size 4) + (squares_of_size 5)

/-- Theorem: The total number of squares in a 6x6 grid is 55 -/
theorem six_by_six_grid_squares : total_squares = 55 := by
  sorry

end six_by_six_grid_squares_l3108_310892


namespace meals_for_adults_l3108_310888

/-- The number of meals initially available for adults -/
def A : ℕ := 18

/-- The number of children that can be fed with all the meals -/
def C : ℕ := 90

/-- Theorem stating that A is the correct number of meals initially available for adults -/
theorem meals_for_adults : 
  (∀ x : ℕ, x * (C / A) = 72 → x = 14) ∧ 
  (A : ℚ) = C / (72 / 14) :=
sorry

end meals_for_adults_l3108_310888


namespace constant_distance_l3108_310809

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line (x y m : ℝ) : Prop := y = (1/2) * x + m

-- Define the constraint on m
def m_constraint (m : ℝ) : Prop := -Real.sqrt 2 < m ∧ m < Real.sqrt 2

-- Define the intersection points A and C
def intersection_points (xa ya xc yc m : ℝ) : Prop :=
  ellipse xa ya ∧ ellipse xc yc ∧ line xa ya m ∧ line xc yc m

-- Define the square ABCD
def square_ABCD (xa ya xb yb xc yc xd yd : ℝ) : Prop :=
  (xc - xa)^2 + (yc - ya)^2 = (xd - xb)^2 + (yd - yb)^2 ∧
  (xb - xa)^2 + (yb - ya)^2 = (xd - xc)^2 + (yd - yc)^2

-- Define point N
def point_N (xn m : ℝ) : Prop := xn = -2 * m

-- Main theorem
theorem constant_distance
  (m xa ya xb yb xc yc xd yd xn : ℝ)
  (h_m : m_constraint m)
  (h_int : intersection_points xa ya xc yc m)
  (h_square : square_ABCD xa ya xb yb xc yc xd yd)
  (h_N : point_N xn m) :
  (xb - xn)^2 + yb^2 = 5/2 :=
sorry

end constant_distance_l3108_310809


namespace mode_best_for_market_share_l3108_310852

/-- Represents different statistical measures -/
inductive StatisticalMeasure
  | Mean
  | Median
  | Mode
  | Variance

/-- Represents a shoe factory -/
structure ShoeFactory where
  survey_data : List Nat  -- List of shoe sizes from the survey

/-- Determines the most appropriate statistical measure for increasing market share -/
def best_measure_for_market_share (factory : ShoeFactory) : StatisticalMeasure :=
  StatisticalMeasure.Mode

/-- Theorem stating that the mode is the most appropriate measure for increasing market share -/
theorem mode_best_for_market_share (factory : ShoeFactory) :
  best_measure_for_market_share factory = StatisticalMeasure.Mode := by
  sorry


end mode_best_for_market_share_l3108_310852


namespace binomial_sum_27_mod_9_l3108_310859

def binomial_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => Nat.choose n k)

theorem binomial_sum_27_mod_9 :
  (binomial_sum 27 - Nat.choose 27 0) % 9 = 7 := by
  sorry

end binomial_sum_27_mod_9_l3108_310859


namespace P_investment_theorem_l3108_310844

-- Define the investments and profit ratio
def Q_investment : ℕ := 60000
def profit_ratio_P : ℕ := 2
def profit_ratio_Q : ℕ := 3

-- Theorem to prove P's investment
theorem P_investment_theorem :
  ∃ P_investment : ℕ,
    P_investment * profit_ratio_Q = Q_investment * profit_ratio_P ∧
    P_investment = 40000 := by
  sorry

end P_investment_theorem_l3108_310844


namespace M_greater_than_N_l3108_310868

theorem M_greater_than_N (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) : a * b > a + b - 1 := by
  sorry

end M_greater_than_N_l3108_310868


namespace fraction_difference_simplification_l3108_310867

theorem fraction_difference_simplification : 
  ∃ q : ℕ+, (2022 : ℚ) / 2021 - 2021 / 2022 = (4043 : ℚ) / q ∧ Nat.gcd 4043 q = 1 := by
  sorry

end fraction_difference_simplification_l3108_310867


namespace or_necessary_not_sufficient_for_and_l3108_310876

theorem or_necessary_not_sufficient_for_and (p q : Prop) : 
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by
  sorry

end or_necessary_not_sufficient_for_and_l3108_310876


namespace valid_arrangements_count_l3108_310826

/-- Represents a soccer team with 11 players -/
def SoccerTeam := Fin 11

/-- The number of ways to arrange players from two soccer teams in a line
    such that no two adjacent players are from the same team -/
def valid_arrangements : ℕ :=
  2 * (Nat.factorial 11) ^ 2

/-- Theorem stating that the number of valid arrangements is correct -/
theorem valid_arrangements_count :
  valid_arrangements = 2 * (Nat.factorial 11) ^ 2 := by sorry

end valid_arrangements_count_l3108_310826


namespace opposite_of_three_l3108_310813

theorem opposite_of_three : (-(3 : ℤ)) = -3 := by
  sorry

end opposite_of_three_l3108_310813


namespace complex_fraction_simplification_l3108_310889

theorem complex_fraction_simplification :
  (7 : ℂ) + 18 * I / (3 - 4 * I) = -(51 : ℚ) / 25 + (82 : ℚ) / 25 * I :=
by sorry

end complex_fraction_simplification_l3108_310889


namespace complex_sum_argument_l3108_310893

theorem complex_sum_argument : 
  let z : ℂ := Complex.exp (7 * π * I / 60) + Complex.exp (17 * π * I / 60) + 
                Complex.exp (27 * π * I / 60) + Complex.exp (37 * π * I / 60) + 
                Complex.exp (47 * π * I / 60)
  Complex.arg z = 9 * π / 20 := by
  sorry

end complex_sum_argument_l3108_310893


namespace sum_of_coefficients_l3108_310816

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 510 := by
sorry

end sum_of_coefficients_l3108_310816


namespace arithmetic_sequence_property_l3108_310850

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (is_arithmetic_sequence a → a 1 + a 3 = 2 * a 2) ∧
  (∃ a : ℕ → ℝ, a 1 + a 3 = 2 * a 2 ∧ ¬is_arithmetic_sequence a) :=
sorry

end arithmetic_sequence_property_l3108_310850


namespace tan_one_implies_sin_2a_minus_cos_sq_a_eq_half_l3108_310883

theorem tan_one_implies_sin_2a_minus_cos_sq_a_eq_half (α : Real) 
  (h : Real.tan α = 1) : Real.sin (2 * α) - Real.cos α ^ 2 = 1/2 := by
  sorry

end tan_one_implies_sin_2a_minus_cos_sq_a_eq_half_l3108_310883


namespace line_equation_l3108_310856

/-- Circle P: x^2 + y^2 - 4y = 0 -/
def circleP (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*y = 0

/-- Parabola S: y = x^2 / 8 -/
def parabolaS (x y : ℝ) : Prop :=
  y = x^2 / 8

/-- Line l: y = k*x + b -/
def lineL (k b x y : ℝ) : Prop :=
  y = k*x + b

/-- Center of circle P -/
def centerP : ℝ × ℝ := (0, 2)

/-- Line l passes through the center of circle P -/
def lineThroughCenter (k b : ℝ) : Prop :=
  lineL k b (centerP.1) (centerP.2)

/-- Four intersection points of line l with circle P and parabola S -/
structure IntersectionPoints (k b : ℝ) :=
  (A B C D : ℝ × ℝ)
  (intersectCircleP : circleP A.1 A.2 ∧ circleP B.1 B.2 ∧ circleP C.1 C.2 ∧ circleP D.1 D.2)
  (intersectParabolaS : parabolaS A.1 A.2 ∧ parabolaS B.1 B.2 ∧ parabolaS C.1 C.2 ∧ parabolaS D.1 D.2)
  (onLineL : lineL k b A.1 A.2 ∧ lineL k b B.1 B.2 ∧ lineL k b C.1 C.2 ∧ lineL k b D.1 D.2)
  (leftToRight : A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1)

/-- Lengths of segments AB, BC, CD form an arithmetic sequence -/
def arithmeticSequence (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  BC - AB = CD - BC

theorem line_equation :
  ∀ k b : ℝ,
  lineThroughCenter k b →
  (∃ pts : IntersectionPoints k b, arithmeticSequence pts.A pts.B pts.C pts.D) →
  (k = -Real.sqrt 2 / 2 ∨ k = Real.sqrt 2 / 2) ∧ b = 2 :=
sorry

end line_equation_l3108_310856


namespace game_lives_calculation_l3108_310803

/-- Given an initial number of players, additional players joining, and lives per player,
    calculate the total number of lives for all players. -/
def totalLives (initialPlayers additionalPlayers livesPerPlayer : ℕ) : ℕ :=
  (initialPlayers + additionalPlayers) * livesPerPlayer

/-- Prove that given 25 initial players, 10 additional players, and 15 lives per player,
    the total number of lives for all players is 525. -/
theorem game_lives_calculation :
  totalLives 25 10 15 = 525 := by
  sorry

end game_lives_calculation_l3108_310803


namespace equation_solution_l3108_310872

theorem equation_solution : 
  ∀ x : ℝ, (3*x + 2)*(x + 3) = x + 3 ↔ x = -3 ∨ x = -1/3 := by
sorry

end equation_solution_l3108_310872


namespace min_value_a_plus_4b_l3108_310802

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) :
  a + 4 * b ≥ 9 := by
  sorry

end min_value_a_plus_4b_l3108_310802


namespace complex_equation_solution_l3108_310838

theorem complex_equation_solution (z : ℂ) : z - 1 = (z + 1) * I → z = I := by
  sorry

end complex_equation_solution_l3108_310838
