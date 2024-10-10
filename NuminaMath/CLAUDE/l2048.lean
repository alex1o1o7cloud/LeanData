import Mathlib

namespace four_digit_number_problem_l2048_204871

theorem four_digit_number_problem (n : ℕ) : 
  (1000 ≤ n) ∧ (n < 10000) ∧  -- n is a four-digit number
  (n % 10 = 9) ∧              -- the ones digit of n is 9
  ((n - 3) + 57 = 1823)       -- the sum of the mistaken number and 57 is 1823
  → n = 1769 := by
sorry

end four_digit_number_problem_l2048_204871


namespace quadratic_vertex_l2048_204811

/-- A quadratic function passing through specific points has its vertex at x = 5 -/
theorem quadratic_vertex (a b c : ℝ) : 
  (4 = a * 2^2 + b * 2 + c) →
  (4 = a * 8^2 + b * 8 + c) →
  (13 = a * 10^2 + b * 10 + c) →
  (-b / (2 * a) = 5) := by
  sorry

end quadratic_vertex_l2048_204811


namespace percentage_relation_l2048_204874

theorem percentage_relation (x y : ℕ+) (h1 : y * x = 100 * 100) (h2 : y = 125) :
  (y : ℝ) / ((25 : ℝ) / 100 * x) * 100 = 625 := by
  sorry

end percentage_relation_l2048_204874


namespace quadratic_factorization_l2048_204865

theorem quadratic_factorization (C D : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 88 * x + 63 = (C * x - 21) * (D * x - 3)) →
  C * D + C = 21 := by
sorry

end quadratic_factorization_l2048_204865


namespace thousand_to_hundred_power_l2048_204813

theorem thousand_to_hundred_power (h : 1000 = 10^3) : 1000^100 = 10^300 := by
  sorry

end thousand_to_hundred_power_l2048_204813


namespace polynomial_factorization_l2048_204855

theorem polynomial_factorization (a b c : ℚ) : 
  b^2 - c^2 + a*(a + 2*b) = (a + b + c)*(a + b - c) := by
  sorry

end polynomial_factorization_l2048_204855


namespace greatest_five_digit_with_product_90_l2048_204800

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).prod

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem greatest_five_digit_with_product_90 :
  ∃ M : ℕ, is_five_digit M ∧
           digit_product M = 90 ∧
           (∀ n : ℕ, is_five_digit n ∧ digit_product n = 90 → n ≤ M) ∧
           digit_sum M = 18 :=
sorry

end greatest_five_digit_with_product_90_l2048_204800


namespace triangle_problem_l2048_204846

theorem triangle_problem (A B C : ℝ) (a b c S : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  S = (1/2) * a * b * Real.sin C ∧
  (Real.cos B) / (Real.cos C) = -b / (2*a + c) →
  (B = 2*π/3 ∧
   (a = 4 ∧ S = 5 * Real.sqrt 3 → b = Real.sqrt 61)) := by
sorry

end triangle_problem_l2048_204846


namespace factorization_example_l2048_204822

-- Define factorization
def is_factorization (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x ∧ ∃ (p q : ℝ → ℝ), g x = p x * q x

-- Define the left-hand side of the equation
def lhs (m : ℝ) : ℝ := m^2 - 4

-- Define the right-hand side of the equation
def rhs (m : ℝ) : ℝ := (m + 2) * (m - 2)

-- Theorem statement
theorem factorization_example : is_factorization lhs rhs := by sorry

end factorization_example_l2048_204822


namespace solution_set_of_f_geq_1_l2048_204806

def f (x : ℝ) : ℝ := |x - 1| - |x - 2|

theorem solution_set_of_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 2} := by sorry

end solution_set_of_f_geq_1_l2048_204806


namespace quadratic_inequality_range_l2048_204825

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + 3 * (m - 1) < 0) → m < -1 := by
  sorry

end quadratic_inequality_range_l2048_204825


namespace nth_term_equation_l2048_204895

theorem nth_term_equation (n : ℕ) : 
  Real.sqrt ((2 * n^2 : ℝ) / (2 * n + 1) - (n - 1)) = Real.sqrt ((n + 1) * (2 * n + 1)) / (2 * n + 1) := by
  sorry

end nth_term_equation_l2048_204895


namespace ball_distribution_and_fairness_l2048_204869

-- Define the total number of balls
def total_balls : ℕ := 4

-- Define the probabilities
def prob_red_or_yellow : ℚ := 3/4
def prob_yellow_or_blue : ℚ := 1/2

-- Define the number of balls of each color
def red_balls : ℕ := 2
def yellow_balls : ℕ := 1
def blue_balls : ℕ := 1

-- Define the probabilities of drawing same color and different colors
def prob_same_color : ℚ := 3/8
def prob_diff_color : ℚ := 5/8

theorem ball_distribution_and_fairness :
  (red_balls + yellow_balls + blue_balls = total_balls) ∧
  (red_balls : ℚ) / total_balls + (yellow_balls : ℚ) / total_balls = prob_red_or_yellow ∧
  (yellow_balls : ℚ) / total_balls + (blue_balls : ℚ) / total_balls = prob_yellow_or_blue ∧
  prob_diff_color > prob_same_color :=
sorry

end ball_distribution_and_fairness_l2048_204869


namespace anniversary_day_probability_probability_distribution_l2048_204883

def is_leap_year (year : ℕ) : Bool :=
  year % 4 = 0

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def days_between (start_year end_year : ℕ) : ℕ :=
  (List.range (end_year - start_year + 1)).foldl (λ acc y ↦ acc + days_in_year (start_year + y)) 0

theorem anniversary_day_probability (meeting_year : ℕ) 
  (h1 : meeting_year ≥ 1668 ∧ meeting_year ≤ 1671) :
  let total_days := days_between meeting_year (meeting_year + 11)
  let day_shift := total_days % 7
  (day_shift = 0 ∧ meeting_year ∈ [1668, 1670, 1671]) ∨
  (day_shift = 6 ∧ meeting_year = 1669) :=
sorry

theorem probability_distribution :
  let meeting_years := [1668, 1669, 1670, 1671]
  let friday_probability := (meeting_years.filter (λ y ↦ (days_between y (y + 11)) % 7 = 0)).length / meeting_years.length
  let thursday_probability := (meeting_years.filter (λ y ↦ (days_between y (y + 11)) % 7 = 6)).length / meeting_years.length
  friday_probability = 3/4 ∧ thursday_probability = 1/4 :=
sorry

end anniversary_day_probability_probability_distribution_l2048_204883


namespace amount_ratio_l2048_204853

def total_amount : ℕ := 7000
def r_amount : ℕ := 2800

theorem amount_ratio : 
  let pq_amount := total_amount - r_amount
  (r_amount : ℚ) / (pq_amount : ℚ) = 2 / 3 := by
  sorry

end amount_ratio_l2048_204853


namespace quadratic_two_distinct_roots_l2048_204862

/-- 
Given a quadratic equation ax^2 - 4x - 1 = 0, this theorem states the conditions
on 'a' for the equation to have two distinct real roots.
-/
theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    a * x^2 - 4 * x - 1 = 0 ∧ 
    a * y^2 - 4 * y - 1 = 0) ↔ 
  (a > -4 ∧ a ≠ 0) :=
sorry

end quadratic_two_distinct_roots_l2048_204862


namespace range_of_x_l2048_204820

theorem range_of_x (x : ℝ) : 
  (∃ m : ℝ, m ∈ Set.Icc 1 3 ∧ x + 3 * m + 5 > 0) → x > -14 := by
  sorry

end range_of_x_l2048_204820


namespace number_count_proof_l2048_204889

theorem number_count_proof (total_avg : ℝ) (pair1_avg pair2_avg pair3_avg : ℝ) :
  total_avg = 3.95 →
  pair1_avg = 3.4 →
  pair2_avg = 3.85 →
  pair3_avg = 4.600000000000001 →
  (2 * pair1_avg + 2 * pair2_avg + 2 * pair3_avg) / total_avg = 6 := by
  sorry

#check number_count_proof

end number_count_proof_l2048_204889


namespace min_distance_scaled_circle_to_line_l2048_204819

/-- The minimum distance from a point on the scaled circle to a line -/
theorem min_distance_scaled_circle_to_line :
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let l : Set (ℝ × ℝ) := {p | p.1 + Real.sqrt 3 * p.2 - 6 = 0}
  let C' : Set (ℝ × ℝ) := {p | (p.1^2 / 9) + p.2^2 = 1}
  ∃ (d : ℝ), d = 3 - Real.sqrt 3 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ C' → 
      ∀ (q : ℝ × ℝ), q ∈ l → 
        d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end min_distance_scaled_circle_to_line_l2048_204819


namespace max_min_f_on_interval_l2048_204847

def f (x : ℝ) := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  let a := -3
  let b := 0
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 1 ∧ f x_min = -17 :=
sorry

end max_min_f_on_interval_l2048_204847


namespace log_expression_evaluation_l2048_204830

theorem log_expression_evaluation :
  2 * Real.log 2 / Real.log 3 - Real.log (32/9) / Real.log 3 + Real.log 8 / Real.log 3 - (5 : ℝ) ^ (2 * Real.log 3 / Real.log 5) = -7 := by
  sorry

end log_expression_evaluation_l2048_204830


namespace hexagon_unit_triangles_l2048_204840

/-- The number of unit equilateral triangles in a regular hexagon -/
def num_unit_triangles_in_hexagon (side_length : ℕ) : ℕ :=
  6 * side_length^2

/-- Theorem: A regular hexagon with side length 5 contains 150 unit equilateral triangles -/
theorem hexagon_unit_triangles :
  num_unit_triangles_in_hexagon 5 = 150 := by
  sorry

#eval num_unit_triangles_in_hexagon 5

end hexagon_unit_triangles_l2048_204840


namespace train_speed_problem_l2048_204807

/-- Proves that given a train journey where the distance is covered in 276 minutes
    at speed S1, and the same distance can be covered in 69 minutes at 16 kmph,
    then S1 = 4 kmph -/
theorem train_speed_problem (S1 : ℝ) : 
  (276 : ℝ) * S1 = 69 * 16 → S1 = 4 := by sorry

end train_speed_problem_l2048_204807


namespace arccos_one_eq_zero_l2048_204823

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l2048_204823


namespace trig_ratios_on_line_l2048_204802

/-- Given an angle α whose terminal side lies on the line y = 2x, 
    prove its trigonometric ratios. -/
theorem trig_ratios_on_line (α : Real) : 
  (∃ k : Real, k ≠ 0 ∧ Real.cos α = k ∧ Real.sin α = 2 * k) → 
  (Real.sin α)^2 = 4/5 ∧ (Real.cos α)^2 = 1/5 ∧ Real.tan α = 2 := by
  sorry

end trig_ratios_on_line_l2048_204802


namespace square_fencing_cost_theorem_l2048_204861

/-- Represents the cost of fencing a square. -/
structure SquareFencingCost where
  totalCost : ℝ
  sideCost : ℝ

/-- The cost of fencing a square with equal side costs. -/
def fencingCost (s : SquareFencingCost) : Prop :=
  s.totalCost = 4 * s.sideCost

theorem square_fencing_cost_theorem (s : SquareFencingCost) :
  s.totalCost = 316 → fencingCost s → s.sideCost = 79 := by
  sorry

end square_fencing_cost_theorem_l2048_204861


namespace seokjin_drank_least_l2048_204872

def seokjin_milk : ℚ := 11/10
def jungkook_milk : ℚ := 13/10
def yoongi_milk : ℚ := 7/6

theorem seokjin_drank_least :
  seokjin_milk < jungkook_milk ∧ seokjin_milk < yoongi_milk :=
by sorry

end seokjin_drank_least_l2048_204872


namespace min_value_expression_l2048_204803

theorem min_value_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  4 * x^2 + 9 * y^2 + 16 / x^2 + 6 * y / x ≥ 2 * Real.sqrt 564 / 3 ∧
  ∃ x₀ y₀ : ℝ, x₀ ≠ 0 ∧ y₀ ≠ 0 ∧
    4 * x₀^2 + 9 * y₀^2 + 16 / x₀^2 + 6 * y₀ / x₀ = 2 * Real.sqrt 564 / 3 :=
by sorry

end min_value_expression_l2048_204803


namespace infinite_good_pairs_l2048_204804

/-- A number is "good" if every prime factor in its prime factorization appears with an exponent of at least 2 -/
def is_good (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (∃ k : ℕ, k ≥ 2 ∧ p ^ k ∣ n)

/-- The sequence of "good" numbers -/
def good_sequence : ℕ → ℕ
  | 0 => 8
  | n + 1 => 4 * good_sequence n * (good_sequence n + 1)

/-- Theorem stating the existence of infinitely many pairs of consecutive "good" numbers -/
theorem infinite_good_pairs :
  ∀ n : ℕ, is_good (good_sequence n) ∧ is_good (good_sequence n + 1) :=
by sorry

end infinite_good_pairs_l2048_204804


namespace tan_squared_sum_lower_bound_l2048_204837

theorem tan_squared_sum_lower_bound 
  (α β γ : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < γ) (h4 : γ < π / 2)
  (h5 : Real.sin α ^ 3 + Real.sin β ^ 3 + Real.sin γ ^ 3 = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 / (9 ^ (1/3) - 1) := by
  sorry

end tan_squared_sum_lower_bound_l2048_204837


namespace gcd_9011_2147_l2048_204829

theorem gcd_9011_2147 : Nat.gcd 9011 2147 = 1 := by
  sorry

end gcd_9011_2147_l2048_204829


namespace geometric_mean_a2_a8_l2048_204816

theorem geometric_mean_a2_a8 (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                     -- first term
  q = 2 →                       -- common ratio
  (a 2 * a 8).sqrt = 48 ∨ (a 2 * a 8).sqrt = -48 :=
by sorry

end geometric_mean_a2_a8_l2048_204816


namespace cake_division_l2048_204884

theorem cake_division (K : ℕ) (h_K : K = 1997) : 
  ∃ N : ℕ, 
    (N > 0) ∧ 
    (K ∣ N) ∧ 
    (K ∣ N^3) ∧ 
    (K ∣ 6*N^2) ∧ 
    (∀ M : ℕ, M < N → ¬(K ∣ M) ∨ ¬(K ∣ M^3) ∨ ¬(K ∣ 6*M^2)) :=
by sorry

end cake_division_l2048_204884


namespace surface_area_of_special_rectangular_solid_l2048_204833

/-- A function that checks if a number is prime or a square of a prime -/
def isPrimeOrSquareOfPrime (n : ℕ) : Prop :=
  Nat.Prime n ∨ ∃ p, Nat.Prime p ∧ n = p^2

/-- Definition of a rectangular solid with the given properties -/
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ
  length_valid : isPrimeOrSquareOfPrime length
  width_valid : isPrimeOrSquareOfPrime width
  height_valid : isPrimeOrSquareOfPrime height
  volume_is_1155 : length * width * height = 1155

/-- The theorem to be proved -/
theorem surface_area_of_special_rectangular_solid (r : RectangularSolid) :
  2 * (r.length * r.width + r.width * r.height + r.height * r.length) = 814 :=
sorry

end surface_area_of_special_rectangular_solid_l2048_204833


namespace tank_filling_time_l2048_204827

/-- Represents the time (in minutes) it takes for pipe A to fill the tank alone -/
def A : ℝ := 24

/-- Represents the time (in minutes) it takes for pipe B to fill the tank alone -/
def B : ℝ := 32

/-- Represents the time (in minutes) both pipes are open before pipe B is closed -/
def t_both : ℝ := 8

/-- Represents the total time (in minutes) to fill the tank using both pipes as described -/
def t_total : ℝ := 18

theorem tank_filling_time : 
  (t_both * (1 / A + 1 / B)) + ((t_total - t_both) * (1 / A)) = 1 ∧ 
  A = 24 := by
  sorry

#check tank_filling_time

end tank_filling_time_l2048_204827


namespace simplify_expression_l2048_204878

theorem simplify_expression : 
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) = 14/3 := by
sorry

end simplify_expression_l2048_204878


namespace complement_of_at_least_two_defective_l2048_204801

def total_products : ℕ := 10

-- Define the event A
def event_A (defective : ℕ) : Prop := defective ≥ 2 ∧ defective ≤ total_products

-- Define the complementary event of A
def complement_A (defective : ℕ) : Prop := defective ≤ 1

-- Theorem statement
theorem complement_of_at_least_two_defective :
  ∀ (defective : ℕ), defective ≤ total_products →
  (¬ event_A defective ↔ complement_A defective) :=
sorry

end complement_of_at_least_two_defective_l2048_204801


namespace convex_polygon_diagonal_inequality_l2048_204892

theorem convex_polygon_diagonal_inequality (n : ℕ) (d p : ℝ) (h1 : n ≥ 3) (h2 : d > 0) (h3 : p > 0) : 
  (n : ℝ) - 3 < 2 * d / p ∧ 2 * d / p < ↑(n / 2) * ↑((n + 1) / 2) - 2 := by
  sorry

end convex_polygon_diagonal_inequality_l2048_204892


namespace second_largest_divisor_sum_l2048_204890

theorem second_largest_divisor_sum (n : ℕ) : 
  n > 1 → 
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ n + n / p = 2013) → 
  n = 1342 := by
sorry

end second_largest_divisor_sum_l2048_204890


namespace ice_melting_volume_l2048_204831

theorem ice_melting_volume (ice_volume : ℝ) (h1 : ice_volume = 2) :
  let water_volume := ice_volume * (10/11)
  water_volume = 20/11 :=
by sorry

end ice_melting_volume_l2048_204831


namespace min_value_of_exponential_sum_l2048_204848

theorem min_value_of_exponential_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  ∀ z, 3^x + 9^y ≥ z → z ≤ 2 * Real.sqrt 3 :=
by sorry

end min_value_of_exponential_sum_l2048_204848


namespace cos_210_degrees_l2048_204815

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_degrees_l2048_204815


namespace football_team_size_l2048_204828

/-- Represents the composition of a football team -/
structure FootballTeam where
  total : ℕ
  throwers : ℕ
  rightHanded : ℕ
  leftHanded : ℕ

/-- The properties of our specific football team -/
def ourTeam : FootballTeam where
  total := 70
  throwers := 40
  rightHanded := 60
  leftHanded := 70 - 40 - (60 - 40)

theorem football_team_size : 
  ∀ (team : FootballTeam), 
  team.throwers = 40 ∧ 
  team.rightHanded = 60 ∧ 
  team.leftHanded = (team.total - team.throwers) / 3 ∧
  team.rightHanded = team.throwers + 2 * (team.total - team.throwers) / 3 →
  team.total = 70 := by
  sorry

#check football_team_size

end football_team_size_l2048_204828


namespace coupon_usage_theorem_l2048_204832

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

def isCouponDay (startDay : DayOfWeek) (n : Nat) : Prop :=
  ∃ k : Nat, k < 8 ∧ advanceDays startDay (7 * k) = DayOfWeek.Monday

theorem coupon_usage_theorem (startDay : DayOfWeek) :
  startDay = DayOfWeek.Sunday ↔
    ¬(isCouponDay startDay 8) ∧
    ∀ d : DayOfWeek, d ≠ DayOfWeek.Sunday → isCouponDay d 8 :=
by sorry

end coupon_usage_theorem_l2048_204832


namespace uphill_speed_calculation_l2048_204897

theorem uphill_speed_calculation (uphill_distance : ℝ) (downhill_distance : ℝ) 
  (downhill_speed : ℝ) (average_speed : ℝ) :
  uphill_distance = 100 →
  downhill_distance = 50 →
  downhill_speed = 40 →
  average_speed = 32.73 →
  ∃ uphill_speed : ℝ,
    uphill_speed = 30 ∧
    average_speed = (uphill_distance + downhill_distance) / 
      (uphill_distance / uphill_speed + downhill_distance / downhill_speed) :=
by
  sorry

end uphill_speed_calculation_l2048_204897


namespace room_rent_problem_l2048_204854

theorem room_rent_problem (total_rent_A total_rent_B : ℝ) 
  (rent_difference : ℝ) (h1 : total_rent_A = 4800) (h2 : total_rent_B = 4200) 
  (h3 : rent_difference = 30) :
  let rent_A := 240
  let rent_B := 210
  (total_rent_A / rent_A = total_rent_B / rent_B) ∧ 
  (rent_A = rent_B + rent_difference) := by
  sorry

end room_rent_problem_l2048_204854


namespace existence_of_divisible_power_sum_l2048_204824

theorem existence_of_divisible_power_sum (a b : ℕ) (h : b > 1) :
  ∃ n : ℕ, n < b^2 ∧ b ∣ (a^n + n) := by
  sorry

end existence_of_divisible_power_sum_l2048_204824


namespace sin_minus_cos_tan_one_third_l2048_204880

theorem sin_minus_cos_tan_one_third (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan θ = 1/3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end sin_minus_cos_tan_one_third_l2048_204880


namespace water_tank_capacity_l2048_204826

/-- Represents a cylindrical water tank. -/
structure WaterTank where
  capacity : ℝ
  initialFill : ℝ

/-- Condition that the tank is 1/6 full initially. -/
def isInitiallySixthFull (tank : WaterTank) : Prop :=
  tank.initialFill / tank.capacity = 1 / 6

/-- Condition that the tank becomes 1/3 full after adding 5 liters. -/
def isThirdFullAfterAddingFive (tank : WaterTank) : Prop :=
  (tank.initialFill + 5) / tank.capacity = 1 / 3

/-- Theorem stating that if a water tank satisfies the given conditions, its capacity is 30 liters. -/
theorem water_tank_capacity
    (tank : WaterTank)
    (h1 : isInitiallySixthFull tank)
    (h2 : isThirdFullAfterAddingFive tank) :
    tank.capacity = 30 := by
  sorry


end water_tank_capacity_l2048_204826


namespace multiplication_result_l2048_204896

theorem multiplication_result : 163861 * 454733 = 74505853393 := by
  sorry

end multiplication_result_l2048_204896


namespace books_from_first_shop_l2048_204863

theorem books_from_first_shop 
  (total_cost_first : ℝ) 
  (books_second : ℕ) 
  (cost_second : ℝ) 
  (avg_price : ℝ) 
  (h1 : total_cost_first = 1160)
  (h2 : books_second = 50)
  (h3 : cost_second = 920)
  (h4 : avg_price = 18.08695652173913)
  : ∃ (books_first : ℕ), books_first = 65 ∧ 
    (total_cost_first + cost_second) / (books_first + books_second : ℝ) = avg_price :=
by sorry

end books_from_first_shop_l2048_204863


namespace product_of_geometric_terms_l2048_204842

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem product_of_geometric_terms
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_sum : a 3 + a 11 = 8)
  (h_equal : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry

end product_of_geometric_terms_l2048_204842


namespace hyperbola_equation_l2048_204821

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    left focus F₁ and right focus F₂ on the x-axis,
    point P(3,4) on an asymptote, and |PF₁ + PF₂| = |F₁F₂|,
    prove that the equation of the hyperbola is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (F₁ F₂ : ℝ × ℝ) (hF : ∃ c : ℝ, F₁ = (-c, 0) ∧ F₂ = (c, 0))
  (P : ℝ × ℝ) (hP : P = (3, 4))
  (h_asymptote : ∃ k : ℝ, k * 3 = 4 ∧ (∀ x y : ℝ, y = k * x → x^2/a^2 - y^2/b^2 = 1))
  (h_vector_sum : ‖P - F₁ + (P - F₂)‖ = ‖F₂ - F₁‖) :
  ∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ x^2/a^2 - y^2/b^2 = 1 :=
sorry

end hyperbola_equation_l2048_204821


namespace cherry_pie_degrees_l2048_204876

theorem cherry_pie_degrees (total_students : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ) 
  (h1 : total_students = 36)
  (h2 : chocolate = 12)
  (h3 : apple = 8)
  (h4 : blueberry = 6)
  (h5 : (total_students - (chocolate + apple + blueberry)) % 2 = 0) :
  (((total_students - (chocolate + apple + blueberry)) / 2) : ℚ) / total_students * 360 = 50 := by
  sorry

end cherry_pie_degrees_l2048_204876


namespace fifth_pythagorean_triple_l2048_204852

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def is_consecutive (m n : ℕ) : Prop :=
  m + 1 = n

theorem fifth_pythagorean_triple (a b c : ℕ) :
  is_pythagorean_triple 3 4 5 ∧
  is_pythagorean_triple 5 12 13 ∧
  is_pythagorean_triple 7 24 25 ∧
  is_pythagorean_triple 9 40 41 ∧
  (∀ x y z : ℕ, is_pythagorean_triple x y z → Odd x) ∧
  (∀ x y z : ℕ, is_pythagorean_triple x y z → is_consecutive y z) ∧
  (∀ x y z : ℕ, is_pythagorean_triple x y z → x * x = y + z) →
  is_pythagorean_triple 11 60 61 :=
by sorry

end fifth_pythagorean_triple_l2048_204852


namespace dihedral_angle_segment_length_l2048_204860

/-- Given a dihedral angle of 120°, this theorem calculates the length of the segment
    connecting the ends of two perpendiculars drawn from the ends of a segment on the edge
    of the dihedral angle. -/
theorem dihedral_angle_segment_length 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  ∃ (length : ℝ), length = Real.sqrt (a^2 + b^2 + a*b + c^2) := by
sorry

end dihedral_angle_segment_length_l2048_204860


namespace yoongis_class_size_l2048_204858

theorem yoongis_class_size :
  ∀ (students_a students_b students_both : ℕ),
    students_a = 18 →
    students_b = 24 →
    students_both = 7 →
    students_a + students_b - students_both = 35 :=
by
  sorry

end yoongis_class_size_l2048_204858


namespace symmetric_point_wrt_origin_l2048_204886

/-- The symmetric point of M(2, -3, 1) with respect to the origin is (-2, 3, -1). -/
theorem symmetric_point_wrt_origin :
  let M : ℝ × ℝ × ℝ := (2, -3, 1)
  let symmetric_point : ℝ × ℝ × ℝ := (-2, 3, -1)
  ∀ (x y z : ℝ), (x, y, z) = M → (-x, -y, -z) = symmetric_point :=
by sorry

end symmetric_point_wrt_origin_l2048_204886


namespace xyz_value_l2048_204866

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 22) :
  x * y * z = 14 / 3 := by
  sorry

end xyz_value_l2048_204866


namespace hall_width_is_25_l2048_204881

/-- Represents the dimensions and cost parameters of a rectangular hall --/
structure HallParameters where
  length : ℝ
  height : ℝ
  cost_per_sqm : ℝ
  total_cost : ℝ

/-- Calculates the total area to be covered in the hall --/
def total_area (params : HallParameters) (width : ℝ) : ℝ :=
  params.length * width + 2 * (params.length * params.height) + 2 * (width * params.height)

/-- Theorem stating that the width of the hall is 25 meters given the specified parameters --/
theorem hall_width_is_25 (params : HallParameters) 
    (h1 : params.length = 20)
    (h2 : params.height = 5)
    (h3 : params.cost_per_sqm = 40)
    (h4 : params.total_cost = 38000) :
    ∃ w : ℝ, w = 25 ∧ total_area params w * params.cost_per_sqm = params.total_cost :=
  sorry

end hall_width_is_25_l2048_204881


namespace reeya_average_score_l2048_204891

theorem reeya_average_score : 
  let scores : List ℝ := [50, 60, 70, 80, 80]
  (scores.sum / scores.length : ℝ) = 68 := by
sorry

end reeya_average_score_l2048_204891


namespace range_of_a_l2048_204849

theorem range_of_a (a : ℝ) : 
  (∀ x, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ x, x ≥ a ∧ |x - 1| ≥ 1) → 
  a ≤ 0 := by
sorry


end range_of_a_l2048_204849


namespace post_office_distance_l2048_204867

/-- Proves that the distance of a round trip is 20 km given specific speeds and total time -/
theorem post_office_distance (outbound_speed inbound_speed : ℝ) (total_time : ℝ) 
  (h1 : outbound_speed = 25)
  (h2 : inbound_speed = 4)
  (h3 : total_time = 5.8) :
  let distance := (outbound_speed * inbound_speed * total_time) / (outbound_speed + inbound_speed)
  distance = 20 := by
  sorry

end post_office_distance_l2048_204867


namespace math_teacher_initial_amount_l2048_204873

theorem math_teacher_initial_amount :
  let basic_calculator_cost : ℕ := 8
  let scientific_calculator_cost : ℕ := 2 * basic_calculator_cost
  let graphing_calculator_cost : ℕ := 3 * scientific_calculator_cost
  let total_cost : ℕ := basic_calculator_cost + scientific_calculator_cost + graphing_calculator_cost
  let change : ℕ := 28
  let initial_amount : ℕ := total_cost + change
  initial_amount = 100
  := by sorry

end math_teacher_initial_amount_l2048_204873


namespace picture_books_count_l2048_204856

theorem picture_books_count (total : ℕ) (fiction : ℕ) : 
  total = 35 →
  fiction = 5 →
  let nonfiction := fiction + 4
  let autobiographies := 2 * fiction
  let other_books := fiction + nonfiction + autobiographies
  total - other_books = 11 :=
by
  sorry

end picture_books_count_l2048_204856


namespace systematic_sampling_l2048_204809

/-- Systematic sampling problem -/
theorem systematic_sampling 
  (total_items : ℕ) 
  (selected_items : ℕ) 
  (first_selected : ℕ) 
  (group_number : ℕ) :
  total_items = 3000 →
  selected_items = 150 →
  first_selected = 11 →
  group_number = 61 →
  (group_number - 1) * (total_items / selected_items) + first_selected = 1211 :=
by sorry

end systematic_sampling_l2048_204809


namespace quadratic_root_difference_l2048_204894

theorem quadratic_root_difference (x : ℝ) : 
  let a : ℝ := 1
  let b : ℝ := -9
  let c : ℝ := 4
  let r₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x^2 - 9*x + 4 = 0 → abs (r₁ - r₂) = Real.sqrt 65 :=
by sorry

end quadratic_root_difference_l2048_204894


namespace min_value_x_plus_2y_l2048_204882

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 4*y + x*y = 1) : 
  ∃ (m : ℝ), m = 2*Real.sqrt 6 - 4 ∧ x + 2*y ≥ m ∧ ∀ z, (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 4*b + a*b = 1 ∧ z = a + 2*b) → z ≥ m :=
sorry

end min_value_x_plus_2y_l2048_204882


namespace special_rectangle_area_l2048_204844

/-- Rectangle with a special circle configuration -/
structure SpecialRectangle where
  -- The radius of the inscribed circle
  r : ℝ
  -- The width of the rectangle (length of side AB)
  w : ℝ
  -- The height of the rectangle (length of side AD)
  h : ℝ
  -- The circle is tangent to sides AD and BC
  tangent_sides : h = 2 * r
  -- The circle is tangent internally to the semicircle with diameter AB
  tangent_semicircle : w = 6 * r
  -- The circle passes through the midpoint of AB
  passes_midpoint : w / 2 = 3 * r

/-- The area of the special rectangle is 12r^2 -/
theorem special_rectangle_area (rect : SpecialRectangle) :
  rect.w * rect.h = 12 * rect.r^2 := by
  sorry


end special_rectangle_area_l2048_204844


namespace power_sum_equality_l2048_204805

theorem power_sum_equality : (-2)^2009 + (-2)^2010 = 2^2009 := by
  sorry

end power_sum_equality_l2048_204805


namespace parabola_increasing_condition_l2048_204877

/-- A parabola defined by y = (a - 1)x^2 + 1 -/
def parabola (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + 1

/-- The parabola increases as x increases when x ≥ 0 -/
def increases_for_nonneg_x (a : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → parabola a x₁ < parabola a x₂

theorem parabola_increasing_condition (a : ℝ) :
  increases_for_nonneg_x a → a > 1 := by sorry

end parabola_increasing_condition_l2048_204877


namespace min_value_theorem_l2048_204875

theorem min_value_theorem (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : (b + c) / a + (a + c) / b = (a + b) / c + 1) :
  (∀ x y z, 0 < x ∧ 0 < y ∧ 0 < z ∧ (y + z) / x + (x + z) / y = (x + y) / z + 1 → (a + b) / c ≤ (x + y) / z) ∧
  (a + b) / c = 5 / 2 := by
sorry

end min_value_theorem_l2048_204875


namespace cards_lost_l2048_204843

theorem cards_lost (initial_cards : ℝ) (final_cards : ℕ) : 
  initial_cards = 47.0 → final_cards = 40 → initial_cards - final_cards = 7 := by
  sorry

end cards_lost_l2048_204843


namespace unique_solution_m_l2048_204887

theorem unique_solution_m (m : ℚ) : 
  (∃! x, (x - 3) / (m * x + 4) = 2 * x) ↔ m = 49 / 24 := by
  sorry

end unique_solution_m_l2048_204887


namespace skateboard_travel_distance_l2048_204834

/-- Represents the distance traveled by a skateboard in a given number of seconds -/
def skateboardDistance (initialDistance : ℕ) (firstAcceleration : ℕ) (secondAcceleration : ℕ) (totalSeconds : ℕ) : ℕ :=
  let firstPeriodDistance := (5 : ℕ) * (2 * initialDistance + 4 * firstAcceleration) / 2
  let secondPeriodInitialDistance := initialDistance + 5 * firstAcceleration
  let secondPeriodDistance := (5 : ℕ) * (2 * secondPeriodInitialDistance + 4 * secondAcceleration) / 2
  firstPeriodDistance + secondPeriodDistance

theorem skateboard_travel_distance :
  skateboardDistance 8 6 9 10 = 380 := by
  sorry

end skateboard_travel_distance_l2048_204834


namespace abs_diff_roots_sum_of_cubes_l2048_204879

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := 2 * x^2 + 7 * x - 4

-- Define the roots
def x₁ : ℝ := sorry
def x₂ : ℝ := sorry

-- Axioms for the roots
axiom root₁ : quadratic x₁ = 0
axiom root₂ : quadratic x₂ = 0

-- Theorems to prove
theorem abs_diff_roots : |x₁ - x₂| = 9/2 := sorry

theorem sum_of_cubes : x₁^3 + x₂^3 = -511/8 := sorry

end abs_diff_roots_sum_of_cubes_l2048_204879


namespace count_divisible_sum_l2048_204835

theorem count_divisible_sum : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ (n * (n + 1) / 2) ∣ (8 * n)) ∧ 
  (∀ n : ℕ, n > 0 ∧ (n * (n + 1) / 2) ∣ (8 * n) → n ∈ S) ∧ 
  Finset.card S = 4 := by
  sorry

end count_divisible_sum_l2048_204835


namespace quadratic_range_iff_a_values_l2048_204899

/-- The quadratic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2*a + 4

/-- The theorem stating the relationship between the range of f and the values of a -/
theorem quadratic_range_iff_a_values (a : ℝ) :
  (∀ y : ℝ, y ≥ 1 → ∃ x : ℝ, f a x = y) ∧ (∀ x : ℝ, f a x ≥ 1) ↔ a = -1 ∨ a = 3 :=
sorry

end quadratic_range_iff_a_values_l2048_204899


namespace linear_function_proof_l2048_204812

/-- A linear function passing through (0,5) and parallel to y=x -/
def f (x : ℝ) : ℝ := x + 5

theorem linear_function_proof :
  (f 0 = 5) ∧ 
  (∀ x y : ℝ, f (x + y) - f x = y) ∧
  (∀ x : ℝ, f x = x + 5) :=
by sorry

end linear_function_proof_l2048_204812


namespace total_pay_is_1980_l2048_204864

/-- Calculates the total monthly pay for Josh and Carl given their work hours and rates -/
def total_monthly_pay (josh_hours_per_day : ℕ) (work_days_per_week : ℕ) (weeks_per_month : ℕ)
  (carl_hours_less : ℕ) (josh_hourly_rate : ℚ) : ℚ :=
  let josh_monthly_hours := josh_hours_per_day * work_days_per_week * weeks_per_month
  let carl_monthly_hours := (josh_hours_per_day - carl_hours_less) * work_days_per_week * weeks_per_month
  let carl_hourly_rate := josh_hourly_rate / 2
  josh_monthly_hours * josh_hourly_rate + carl_monthly_hours * carl_hourly_rate

theorem total_pay_is_1980 :
  total_monthly_pay 8 5 4 2 9 = 1980 := by
  sorry

end total_pay_is_1980_l2048_204864


namespace prime_sum_divides_cube_diff_l2048_204818

theorem prime_sum_divides_cube_diff (p q : ℕ) : 
  Prime p → Prime q → (p + q) ∣ (p^3 - q^3) → p = q := by
  sorry

end prime_sum_divides_cube_diff_l2048_204818


namespace world_book_day_solution_l2048_204850

/-- Represents the number of books bought by each student -/
structure BookCount where
  a : ℕ
  b : ℕ

/-- The conditions of the World Book Day problem -/
def worldBookDayProblem (bc : BookCount) : Prop :=
  bc.a + bc.b = 22 ∧ bc.a = 2 * bc.b + 1

/-- The theorem stating the solution to the World Book Day problem -/
theorem world_book_day_solution :
  ∃ (bc : BookCount), worldBookDayProblem bc ∧ bc.a = 15 ∧ bc.b = 7 := by
  sorry

end world_book_day_solution_l2048_204850


namespace unique_sum_of_equation_l2048_204888

theorem unique_sum_of_equation (x y : ℤ) :
  (1 / x + 1 / y) * (1 / x^2 + 1 / y^2) = -2/3 * (1 / x^4 - 1 / y^4) →
  ∃! s : ℤ, s = x + y :=
by sorry

end unique_sum_of_equation_l2048_204888


namespace smallest_expressible_proof_l2048_204851

/-- Represents the number of marbles in each box type -/
def box_sizes : Finset ℕ := {13, 11, 7}

/-- Checks if a number can be expressed as a non-negative integer combination of box sizes -/
def is_expressible (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 13 * a + 11 * b + 7 * c

/-- The smallest number such that all larger numbers are expressible -/
def smallest_expressible : ℕ := 30

theorem smallest_expressible_proof :
  (∀ m : ℕ, m > smallest_expressible → is_expressible m) ∧
  (∀ k : ℕ, k < smallest_expressible → ∃ n : ℕ, n > k ∧ ¬is_expressible n) :=
sorry

end smallest_expressible_proof_l2048_204851


namespace max_non_zero_numbers_eq_sum_binary_digits_l2048_204870

/-- The sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- The game state -/
structure GameState where
  numbers : List ℕ

/-- The game move -/
inductive Move
  | Sum : ℕ → ℕ → Move
  | Diff : ℕ → ℕ → Move

/-- Apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState := sorry

/-- Check if the game is over -/
def isGameOver (state : GameState) : Bool := sorry

/-- The maximum number of non-zero numbers at the end of the game -/
def maxNonZeroNumbers (initialOnes : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem max_non_zero_numbers_eq_sum_binary_digits :
  maxNonZeroNumbers 2020 = sumBinaryDigits 2020 := by sorry

end max_non_zero_numbers_eq_sum_binary_digits_l2048_204870


namespace remaining_uncracked_seashells_l2048_204838

def tom_seashells : ℕ := 15
def fred_seashells : ℕ := 43
def cracked_seashells : ℕ := 29
def giveaway_percentage : ℚ := 40 / 100

theorem remaining_uncracked_seashells :
  let total_seashells := tom_seashells + fred_seashells
  let uncracked_seashells := total_seashells - cracked_seashells
  let seashells_to_giveaway := ⌊(giveaway_percentage * uncracked_seashells : ℚ)⌋
  uncracked_seashells - seashells_to_giveaway = 18 := by sorry

end remaining_uncracked_seashells_l2048_204838


namespace solve_for_y_l2048_204836

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 7 = y + 3) (h2 : x = -5) : y = 44 := by
  sorry

end solve_for_y_l2048_204836


namespace two_dice_sum_ten_max_digits_l2048_204859

theorem two_dice_sum_ten_max_digits : ∀ x y : ℕ,
  1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 → x + y = 10 → x < 10 ∧ y < 10 :=
by sorry

end two_dice_sum_ten_max_digits_l2048_204859


namespace option_c_not_algorithm_l2048_204898

-- Define what constitutes an algorithm
def is_algorithm (process : String) : Prop :=
  ∃ (steps : List String), steps.length > 0 ∧ steps.all (λ step => step.length > 0)

-- Define the options
def option_a : String := "The process of solving the equation 2x-6=0 involves moving terms and making the coefficient 1"
def option_b : String := "To get from Jinan to Vancouver, one must first take a train to Beijing, then transfer to a plane"
def option_c : String := "Solving the equation 2x^2+x-1=0"
def option_d : String := "Using the formula S=πr^2 to calculate the area of a circle with radius 3 involves computing π×3^2"

-- Theorem stating that option C is not an algorithm while others are
theorem option_c_not_algorithm :
  is_algorithm option_a ∧
  is_algorithm option_b ∧
  ¬is_algorithm option_c ∧
  is_algorithm option_d :=
sorry

end option_c_not_algorithm_l2048_204898


namespace cassandra_pies_l2048_204808

/-- Calculates the number of apple pies Cassandra made -/
def number_of_pies (apples_bought : ℕ) (slices_per_pie : ℕ) (apples_per_slice : ℕ) : ℕ :=
  (apples_bought / apples_per_slice) / slices_per_pie

theorem cassandra_pies :
  let apples_bought := 4 * 12 -- four dozen
  let slices_per_pie := 6
  let apples_per_slice := 2
  number_of_pies apples_bought slices_per_pie apples_per_slice = 4 := by
  sorry

#eval number_of_pies (4 * 12) 6 2

end cassandra_pies_l2048_204808


namespace volume_is_12pi_l2048_204810

/-- Represents a solid object with three views and dimensions -/
structure Solid where
  frontView : Real × Real
  sideView : Real × Real
  topView : Real × Real

/-- Calculates the volume of a solid based on its views and dimensions -/
def volumeOfSolid (s : Solid) : Real := sorry

/-- Theorem stating that the volume of the given solid is 12π cm³ -/
theorem volume_is_12pi (s : Solid) : volumeOfSolid s = 12 * Real.pi := by sorry

end volume_is_12pi_l2048_204810


namespace base2_to_base4_example_l2048_204868

/-- Converts a natural number from base 2 to base 4 -/
def base2ToBase4 (n : ℕ) : ℕ := sorry

theorem base2_to_base4_example : base2ToBase4 0b10111010000 = 0x11310 := by sorry

end base2_to_base4_example_l2048_204868


namespace triangle_angle_C_l2048_204845

theorem triangle_angle_C (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  3 * Real.sin A + 4 * Real.cos B = 6 ∧  -- Given condition
  4 * Real.sin B + 3 * Real.cos A = 1  -- Given condition
  → C = π / 6 := by
sorry

end triangle_angle_C_l2048_204845


namespace debby_water_bottles_l2048_204893

/-- The number of water bottles Debby drank per day -/
def bottles_per_day : ℕ := 109

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := 74

/-- The total number of bottles Debby bought -/
def total_bottles : ℕ := bottles_per_day * days_lasted

theorem debby_water_bottles : total_bottles = 8066 := by
  sorry

end debby_water_bottles_l2048_204893


namespace parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l2048_204841

/-- The axis of symmetry of a parabola y = ax² + bx + c is x = -b/(2a) -/
theorem parabola_axis_of_symmetry (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  ∃! x₀, ∀ x, f (x₀ + x) = f (x₀ - x) :=
by sorry

/-- The axis of symmetry of the parabola y = -1/2 x² + x - 5/2 is x = 1 -/
theorem specific_parabola_axis_of_symmetry :
  let f : ℝ → ℝ := λ x => -1/2 * x^2 + x - 5/2
  ∃! x₀, ∀ x, f (x₀ + x) = f (x₀ - x) ∧ x₀ = 1 :=
by sorry

end parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l2048_204841


namespace quadratic_equation_roots_l2048_204817

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k + 3)*x₁ + k^2 = 0 ∧ 
    x₂^2 + (2*k + 3)*x₂ + k^2 = 0 ∧
    1/x₁ + 1/x₂ = -1) → 
  k = 3 := by
sorry

end quadratic_equation_roots_l2048_204817


namespace geometric_sequence_problem_l2048_204857

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a where a₃ = 20 and a₆ = 5, prove that a₉ = 5/4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a) 
    (h_a3 : a 3 = 20) 
    (h_a6 : a 6 = 5) : 
  a 9 = 5/4 := by
  sorry

end geometric_sequence_problem_l2048_204857


namespace food_lasts_fifty_days_l2048_204885

/-- The number of days dog food will last given the number of dogs, meals per day, 
    food per meal, number of sacks, and weight per sack. -/
def days_food_lasts (num_dogs : ℕ) (meals_per_day : ℕ) (food_per_meal : ℕ) 
                    (num_sacks : ℕ) (weight_per_sack : ℕ) : ℕ :=
  (num_sacks * weight_per_sack * 1000) / (num_dogs * meals_per_day * food_per_meal)

/-- Proof that given the specific conditions, the food will last 50 days. -/
theorem food_lasts_fifty_days : 
  days_food_lasts 4 2 250 2 50 = 50 := by
  sorry

end food_lasts_fifty_days_l2048_204885


namespace lamp_sales_problem_l2048_204814

/-- Shopping mall lamp sales problem -/
theorem lamp_sales_problem
  (initial_price : ℝ)
  (cost_price : ℝ)
  (initial_sales : ℝ)
  (price_increase : ℝ)
  (sales_decrease_rate : ℝ)
  (h1 : initial_price = 40)
  (h2 : cost_price = 30)
  (h3 : initial_sales = 600)
  (h4 : 0 < price_increase ∧ price_increase < 20)
  (h5 : sales_decrease_rate = 10) :
  let new_sales := initial_sales - sales_decrease_rate * price_increase
  let new_price := initial_price + price_increase
  let profit := (new_price - cost_price) * new_sales
  ∃ (optimal_increase : ℝ) (max_profit_price : ℝ),
    (new_sales = 600 - 10 * price_increase) ∧
    (profit = 10000 → new_price = 50 ∧ new_sales = 500) ∧
    (max_profit_price = 59 ∧ ∀ x, 0 < x ∧ x < 20 → profit ≤ (59 - cost_price) * (initial_sales - sales_decrease_rate * (59 - initial_price))) :=
by sorry

end lamp_sales_problem_l2048_204814


namespace power_and_division_equality_l2048_204839

theorem power_and_division_equality : (12 : ℕ)^2 * 6^4 / 432 = 432 := by
  sorry

end power_and_division_equality_l2048_204839
