import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_powers_l1167_116781

theorem sum_of_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1167_116781


namespace NUMINAMATH_CALUDE_fifth_element_row_20_l1167_116775

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Pascal's triangle element at row n, position k -/
def pascal_triangle_element (n k : ℕ) : ℕ := binomial n (k - 1)

/-- The fifth element in Row 20 of Pascal's triangle is 4845 -/
theorem fifth_element_row_20 : pascal_triangle_element 20 5 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_l1167_116775


namespace NUMINAMATH_CALUDE_sum_to_n_432_l1167_116748

theorem sum_to_n_432 : ∃ n : ℕ, (∀ m : ℕ, m > n → (m * (m + 1)) / 2 > 432) ∧ (n * (n + 1)) / 2 ≤ 432 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_n_432_l1167_116748


namespace NUMINAMATH_CALUDE_accuracy_of_150_38_million_l1167_116794

/-- Represents a number in millions with two decimal places -/
structure MillionNumber where
  value : ℝ
  isMillions : value ≥ 0
  twoDecimalPlaces : ∃ n : ℕ, value = (n : ℝ) / 100

/-- Represents the accuracy of a number in terms of place value -/
inductive PlaceValue
  | Hundred
  | Thousand
  | TenThousand
  | HundredThousand
  | Million

/-- Given a MillionNumber, returns its accuracy in terms of PlaceValue -/
def getAccuracy (n : MillionNumber) : PlaceValue :=
  PlaceValue.Hundred

/-- Theorem stating that 150.38 million is accurate to the hundred place -/
theorem accuracy_of_150_38_million :
  let n : MillionNumber := ⟨150.38, by norm_num, ⟨15038, by norm_num⟩⟩
  getAccuracy n = PlaceValue.Hundred := by
  sorry

end NUMINAMATH_CALUDE_accuracy_of_150_38_million_l1167_116794


namespace NUMINAMATH_CALUDE_odd_not_div_by_three_square_plus_five_div_by_six_l1167_116702

theorem odd_not_div_by_three_square_plus_five_div_by_six (n : ℤ) 
  (h_odd : Odd n) (h_not_div_three : ¬(3 ∣ n)) : 
  6 ∣ (n^2 + 5) := by
  sorry

end NUMINAMATH_CALUDE_odd_not_div_by_three_square_plus_five_div_by_six_l1167_116702


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1167_116727

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (hp : selling_price = 1170)
  (hq : profit_percentage = 20) : 
  ∃ cost_price : ℝ, 
    cost_price * (1 + profit_percentage / 100) = selling_price ∧ 
    cost_price = 975 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1167_116727


namespace NUMINAMATH_CALUDE_black_balls_count_l1167_116789

theorem black_balls_count (total : ℕ) (red : ℕ) (white_prob : ℚ) 
  (h_total : total = 100)
  (h_red : red = 30)
  (h_white_prob : white_prob = 47/100)
  (h_sum : red + (white_prob * total).floor + (total - red - (white_prob * total).floor) = total) :
  total - red - (white_prob * total).floor = 23 := by
  sorry

end NUMINAMATH_CALUDE_black_balls_count_l1167_116789


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1167_116719

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the perimeter is 7 + √19 under the following conditions:
    1) a² - c² + 3b = 0
    2) The area of the triangle is 5√3/2
    3) Angle A = 60° -/
theorem triangle_perimeter (a b c : ℝ) (A : ℝ) (S : ℝ) : 
  a^2 - c^2 + 3*b = 0 → 
  S = (5 * Real.sqrt 3) / 2 →
  A = π / 3 →
  a + b + c = 7 + Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1167_116719


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l1167_116755

/-- The ratio of the total volume of two cones to the volume of a cylinder -/
theorem cone_cylinder_volume_ratio :
  let r : ℝ := 4 -- radius of cylinder and cones
  let h_cyl : ℝ := 18 -- height of cylinder
  let h_cone1 : ℝ := 6 -- height of first cone
  let h_cone2 : ℝ := 9 -- height of second cone
  let v_cyl := π * r^2 * h_cyl -- volume of cylinder
  let v_cone1 := (1/3) * π * r^2 * h_cone1 -- volume of first cone
  let v_cone2 := (1/3) * π * r^2 * h_cone2 -- volume of second cone
  let v_cones := v_cone1 + v_cone2 -- total volume of cones
  v_cones / v_cyl = 5 / 18 := by
sorry


end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l1167_116755


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1167_116752

/-- Given a quadratic function f(x) = x^2 + 2px + r, 
    if the minimum value of f(x) is 1, then r = p^2 + 1 -/
theorem quadratic_minimum (p r : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 + 2*p*x + r) ∧ 
   (∃ (m : ℝ), ∀ x, f x ≥ m ∧ (∃ y, f y = m)) ∧
   (∃ x, f x = 1)) →
  r = p^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1167_116752


namespace NUMINAMATH_CALUDE_solve_for_x_l1167_116722

def U (x : ℝ) : Set ℝ := {2, 3, x^2 + 2*x - 3}
def A (x : ℝ) : Set ℝ := {2, |x + 7|}

theorem solve_for_x : 
  ∃ x : ℝ, (U x \ A x = {5}) ∧ x = -4 :=
sorry

end NUMINAMATH_CALUDE_solve_for_x_l1167_116722


namespace NUMINAMATH_CALUDE_temperature_comparison_l1167_116790

theorem temperature_comparison : -3 < -0.3 := by
  sorry

end NUMINAMATH_CALUDE_temperature_comparison_l1167_116790


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1167_116798

def M : Set Int := {-1, 1}
def N : Set Int := {-1, 0, 2}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1167_116798


namespace NUMINAMATH_CALUDE_john_jane_difference_l1167_116778

-- Define the street width
def street_width : ℕ := 25

-- Define the block side length
def block_side : ℕ := 500

-- Define Jane's path length (same as block side)
def jane_path : ℕ := block_side

-- Define John's path length (block side + 2 * street width)
def john_path : ℕ := block_side + 2 * street_width

-- Theorem statement
theorem john_jane_difference : 
  4 * john_path - 4 * jane_path = 200 := by
  sorry

end NUMINAMATH_CALUDE_john_jane_difference_l1167_116778


namespace NUMINAMATH_CALUDE_car_distance_proof_l1167_116788

theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) (time_factor : ℝ) : 
  initial_time = 6 →
  speed = 32 →
  time_factor = 3 / 2 →
  speed * (time_factor * initial_time) = 288 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l1167_116788


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l1167_116711

/-- The probability of selecting either a blue or yellow jelly bean from a bag -/
theorem jelly_bean_probability : 
  let red : ℕ := 6
  let green : ℕ := 7
  let yellow : ℕ := 8
  let blue : ℕ := 9
  let total : ℕ := red + green + yellow + blue
  let target : ℕ := yellow + blue
  (target : ℚ) / total = 17 / 30 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l1167_116711


namespace NUMINAMATH_CALUDE_exponential_inequality_l1167_116707

theorem exponential_inequality (a b : ℝ) (h : a > b) : (0.9 : ℝ) ^ a < (0.9 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1167_116707


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_l1167_116787

theorem half_abs_diff_squares : (1 / 2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_l1167_116787


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1167_116750

theorem inequality_solution_set : 
  {x : ℝ | x + 5 > -1} = {x : ℝ | x > -6} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1167_116750


namespace NUMINAMATH_CALUDE_second_business_owner_donation_l1167_116799

/-- Given the fundraising conditions, prove the second business owner's donation per slice --/
theorem second_business_owner_donation
  (total_cakes : ℕ)
  (slices_per_cake : ℕ)
  (price_per_slice : ℚ)
  (first_donation_per_slice : ℚ)
  (total_raised : ℚ)
  (h1 : total_cakes = 10)
  (h2 : slices_per_cake = 8)
  (h3 : price_per_slice = 1)
  (h4 : first_donation_per_slice = 1/2)
  (h5 : total_raised = 140) :
  (total_raised - (total_cakes * slices_per_cake * (price_per_slice + first_donation_per_slice))) / (total_cakes * slices_per_cake) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_second_business_owner_donation_l1167_116799


namespace NUMINAMATH_CALUDE_update_year_is_ninth_l1167_116776

def maintenance_cost (n : ℕ) : ℚ :=
  if n ≤ 7 then 2 * n + 2 else 16 * (5/4)^(n-7)

def maintenance_sum (n : ℕ) : ℚ :=
  if n ≤ 7 then n^2 + 3*n else 80 * (5/4)^(n-7) - 10

def average_maintenance_cost (n : ℕ) : ℚ :=
  maintenance_sum n / n

theorem update_year_is_ninth :
  ∀ k, k < 9 → average_maintenance_cost k ≤ 12 ∧
  average_maintenance_cost 9 > 12 :=
sorry

end NUMINAMATH_CALUDE_update_year_is_ninth_l1167_116776


namespace NUMINAMATH_CALUDE_points_collinear_l1167_116737

/-- Prove that points A(-1, -2), B(2, -1), and C(8, 1) are collinear. -/
theorem points_collinear : 
  let A : ℝ × ℝ := (-1, -2)
  let B : ℝ × ℝ := (2, -1)
  let C : ℝ × ℝ := (8, 1)
  ∃ (t : ℝ), C - A = t • (B - A) :=
by sorry

end NUMINAMATH_CALUDE_points_collinear_l1167_116737


namespace NUMINAMATH_CALUDE_max_triangle_area_l1167_116742

/-- The maximum area of a triangle ABC with side AB = 13 and BC:AC ratio of 60:61 is 3634 -/
theorem max_triangle_area (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  AB = 13 ∧ BC / AC = 60 / 61 → area ≤ 3634 :=
by sorry


end NUMINAMATH_CALUDE_max_triangle_area_l1167_116742


namespace NUMINAMATH_CALUDE_cooler_capacity_l1167_116791

theorem cooler_capacity (c1 c2 c3 : ℝ) : 
  c1 = 100 → 
  c2 = c1 * 1.5 → 
  c3 = c2 / 2 → 
  c1 + c2 + c3 = 325 := by
sorry

end NUMINAMATH_CALUDE_cooler_capacity_l1167_116791


namespace NUMINAMATH_CALUDE_purchase_equation_l1167_116796

/-- 
Given a group of people jointly purchasing an item, where:
- Contributing 8 units per person results in an excess of 3 units
- Contributing 7 units per person results in a shortage of 4 units
Prove that the number of people satisfies the equation 8x - 3 = 7x + 4
-/
theorem purchase_equation (x : ℕ) 
  (h1 : 8 * x - 3 = (8 * x - 3)) 
  (h2 : 7 * x + 4 = (7 * x + 4)) : 
  8 * x - 3 = 7 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_purchase_equation_l1167_116796


namespace NUMINAMATH_CALUDE_marks_weekly_reading_time_l1167_116701

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Mark's current daily reading time in hours -/
def daily_reading_time : ℕ := 3

/-- Mark's planned weekly increase in reading time in hours -/
def weekly_increase : ℕ := 6

/-- Theorem: Mark's total weekly reading time after the increase will be 27 hours -/
theorem marks_weekly_reading_time :
  daily_reading_time * days_in_week + weekly_increase = 27 := by
  sorry

end NUMINAMATH_CALUDE_marks_weekly_reading_time_l1167_116701


namespace NUMINAMATH_CALUDE_part1_part2_l1167_116718

/-- The function f(x) = x³ - x² --/
def f (x : ℝ) : ℝ := x^3 - x^2

/-- Part 1: At least one of f(m) and f(n) is not less than zero --/
theorem part1 (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n > 1) :
  max (f m) (f n) ≥ 0 := by sorry

/-- Part 2: a + b < 4/3 --/
theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (heq : f a = f b) :
  a + b < 4/3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1167_116718


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1167_116712

/-- The breadth of a rectangular plot with specific conditions -/
theorem rectangular_plot_breadth :
  ∀ (b l : ℝ),
  (l * b + (1/2 * (b/2) * (l/3)) = 24 * b) →  -- Area condition
  (l - b = 10) →                              -- Length-breadth difference
  (b = 158/13) :=                             -- Breadth of the plot
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1167_116712


namespace NUMINAMATH_CALUDE_trig_sum_equality_l1167_116771

open Real

theorem trig_sum_equality (θ φ : ℝ) :
  (cos θ ^ 6 / cos φ ^ 2) + (sin θ ^ 6 / sin φ ^ 2) = 2 →
  (sin φ ^ 6 / sin θ ^ 2) + (cos φ ^ 6 / cos θ ^ 2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_trig_sum_equality_l1167_116771


namespace NUMINAMATH_CALUDE_balloon_difference_l1167_116747

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
sorry

end NUMINAMATH_CALUDE_balloon_difference_l1167_116747


namespace NUMINAMATH_CALUDE_sum_first_11_odd_numbers_l1167_116784

theorem sum_first_11_odd_numbers : 
  (Finset.range 11).sum (fun i => 2 * i + 1) = 121 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_11_odd_numbers_l1167_116784


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l1167_116783

def average_age_of_team : ℝ := 25

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (wicket_keeper_age : ℝ) 
  (remaining_players_average_age : ℝ) :
  team_size = 11 →
  wicket_keeper_age = average_age_of_team + 3 →
  remaining_players_average_age = average_age_of_team - 1 →
  average_age_of_team * team_size = 
    wicket_keeper_age + 
    (team_size - 2) * remaining_players_average_age + 
    (average_age_of_team * team_size - wicket_keeper_age - (team_size - 2) * remaining_players_average_age) →
  average_age_of_team = 25 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l1167_116783


namespace NUMINAMATH_CALUDE_cake_mix_distribution_l1167_116735

theorem cake_mix_distribution (tray1 tray2 : ℕ) : 
  tray2 = tray1 - 20 → 
  tray1 + tray2 = 500 → 
  tray1 = 260 := by
sorry

end NUMINAMATH_CALUDE_cake_mix_distribution_l1167_116735


namespace NUMINAMATH_CALUDE_frustum_cone_altitude_l1167_116723

theorem frustum_cone_altitude (h : ℝ) (A_lower A_upper : ℝ) :
  h = 24 →
  A_lower = 225 * Real.pi →
  A_upper = 25 * Real.pi →
  ∃ x : ℝ, x = 12 ∧ x = (1/3) * (3/2 * h) :=
by sorry

end NUMINAMATH_CALUDE_frustum_cone_altitude_l1167_116723


namespace NUMINAMATH_CALUDE_harvest_time_calculation_l1167_116703

theorem harvest_time_calculation (initial_harvesters initial_days initial_area final_harvesters final_area : ℕ) 
  (h1 : initial_harvesters = 2)
  (h2 : initial_days = 3)
  (h3 : initial_area = 450)
  (h4 : final_harvesters = 7)
  (h5 : final_area = 2100) :
  (initial_harvesters * initial_days * final_area) / (initial_area * final_harvesters) = 4 := by
  sorry

end NUMINAMATH_CALUDE_harvest_time_calculation_l1167_116703


namespace NUMINAMATH_CALUDE_white_mice_count_l1167_116724

theorem white_mice_count (total : ℕ) (white : ℕ) (brown : ℕ) : 
  (white = 2 * total / 3) →  -- 2/3 of the mice are white
  (brown = 7) →              -- There are 7 brown mice
  (total = white + brown) →  -- Total mice is the sum of white and brown mice
  (white > 0) →              -- There are some white mice
  (white = 14) :=            -- The number of white mice is 14
by
  sorry

end NUMINAMATH_CALUDE_white_mice_count_l1167_116724


namespace NUMINAMATH_CALUDE_subjectB_least_hours_subjectB_total_hours_l1167_116765

/-- Represents the study hours for each subject over a 15-week semester. -/
structure StudyHours where
  subjectA : ℕ
  subjectB : ℕ
  subjectC : ℕ
  subjectD : ℕ

/-- Calculates the total study hours for Subject A over 15 weeks. -/
def calculateSubjectA : ℕ := 3 * 5 * 15

/-- Calculates the total study hours for Subject B over 15 weeks. -/
def calculateSubjectB : ℕ := 2 * 3 * 15

/-- Calculates the total study hours for Subject C over 15 weeks. -/
def calculateSubjectC : ℕ := (4 + 3 + 3) * 15

/-- Calculates the total study hours for Subject D over 15 weeks. -/
def calculateSubjectD : ℕ := (1 * 5 + 5) * 15

/-- Creates a StudyHours structure with the calculated hours for each subject. -/
def parisStudyHours : StudyHours :=
  { subjectA := calculateSubjectA
  , subjectB := calculateSubjectB
  , subjectC := calculateSubjectC
  , subjectD := calculateSubjectD }

/-- Theorem: Subject B has the least study hours among all subjects. -/
theorem subjectB_least_hours (h : StudyHours) (h_eq : h = parisStudyHours) :
  h.subjectB ≤ h.subjectA ∧ h.subjectB ≤ h.subjectC ∧ h.subjectB ≤ h.subjectD :=
by sorry

/-- Theorem: The total study hours for Subject B is 90. -/
theorem subjectB_total_hours : parisStudyHours.subjectB = 90 :=
by sorry

end NUMINAMATH_CALUDE_subjectB_least_hours_subjectB_total_hours_l1167_116765


namespace NUMINAMATH_CALUDE_bread_flour_calculation_l1167_116743

theorem bread_flour_calculation (x : ℝ) : 
  x > 0 ∧ 
  x + 10 > 0 ∧ 
  x * (1 + x / 100) + (x + 10) * (1 + (x + 10) / 100) = 112.5 → 
  x = 35 :=
by sorry

end NUMINAMATH_CALUDE_bread_flour_calculation_l1167_116743


namespace NUMINAMATH_CALUDE_unique_digit_system_solution_l1167_116763

theorem unique_digit_system_solution (a b c t x y : ℕ) 
  (unique_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ t ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧
                    a ≠ b ∧ a ≠ c ∧ a ≠ t ∧ a ≠ x ∧ a ≠ y ∧
                    b ≠ c ∧ b ≠ t ∧ b ≠ x ∧ b ≠ y ∧
                    c ≠ t ∧ c ≠ x ∧ c ≠ y ∧
                    t ≠ x ∧ t ≠ y ∧
                    x ≠ y)
  (eq1 : a + b = x)
  (eq2 : x + c = t)
  (eq3 : t + a = y)
  (eq4 : b + c + y = 20) :
  t = 10 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_system_solution_l1167_116763


namespace NUMINAMATH_CALUDE_sam_walking_time_l1167_116793

/-- Given that Sam walks 0.75 miles in 15 minutes at a constant rate, 
    prove that it takes him 40 minutes to walk 2 miles. -/
theorem sam_walking_time (initial_distance : ℝ) (initial_time : ℝ) (target_distance : ℝ) :
  initial_distance = 0.75 ∧ 
  initial_time = 15 ∧ 
  target_distance = 2 →
  (target_distance / initial_distance) * initial_time = 40 := by
sorry

end NUMINAMATH_CALUDE_sam_walking_time_l1167_116793


namespace NUMINAMATH_CALUDE_count_squares_in_H_l1167_116745

/-- The set of points (x,y) with integer coordinates satisfying 2 ≤ |x| ≤ 8 and 2 ≤ |y| ≤ 8 -/
def H : Set (ℤ × ℤ) :=
  {p | 2 ≤ |p.1| ∧ |p.1| ≤ 8 ∧ 2 ≤ |p.2| ∧ |p.2| ≤ 8}

/-- A square with vertices in H -/
structure SquareInH where
  vertices : Fin 4 → ℤ × ℤ
  in_H : ∀ i, vertices i ∈ H
  is_square : ∃ (side : ℤ), side ≥ 5 ∧
    (vertices 1).1 - (vertices 0).1 = side ∧
    (vertices 2).1 - (vertices 1).1 = side ∧
    (vertices 3).1 - (vertices 2).1 = -side ∧
    (vertices 0).1 - (vertices 3).1 = -side ∧
    (vertices 1).2 - (vertices 0).2 = side ∧
    (vertices 2).2 - (vertices 1).2 = -side ∧
    (vertices 3).2 - (vertices 2).2 = -side ∧
    (vertices 0).2 - (vertices 3).2 = side

/-- The number of squares with side length at least 5 whose vertices are in H -/
def numSquaresInH : ℕ := sorry

theorem count_squares_in_H : numSquaresInH = 14 := by sorry

end NUMINAMATH_CALUDE_count_squares_in_H_l1167_116745


namespace NUMINAMATH_CALUDE_school_vote_total_l1167_116782

theorem school_vote_total (x : ℝ) : 
  (0.35 * x = 0.65 * x) ∧ 
  (0.45 * (x + 80) = 0.65 * x) →
  x + 80 = 260 := by
sorry

end NUMINAMATH_CALUDE_school_vote_total_l1167_116782


namespace NUMINAMATH_CALUDE_f_is_bitwise_or_l1167_116773

/-- Bitwise OR operation for positive integers -/
def bitwiseOr (a b : ℕ+) : ℕ+ := sorry

/-- The function f we want to prove is equal to bitwise OR -/
noncomputable def f : ℕ+ → ℕ+ → ℕ+ := sorry

/-- Condition (i): f(a,b) ≤ a + b for all a, b ∈ ℤ⁺ -/
axiom condition_i (a b : ℕ+) : f a b ≤ a + b

/-- Condition (ii): f(a,f(b,c)) = f(f(a,b),c) for all a, b, c ∈ ℤ⁺ -/
axiom condition_ii (a b c : ℕ+) : f a (f b c) = f (f a b) c

/-- Condition (iii): Both (f(a,b) choose a) and (f(a,b) choose b) are odd numbers for all a, b ∈ ℤ⁺ -/
axiom condition_iii (a b : ℕ+) : Odd (Nat.choose (f a b) a) ∧ Odd (Nat.choose (f a b) b)

/-- f is surjective -/
axiom f_surjective : Function.Surjective f

/-- The main theorem: f is equal to bitwise OR -/
theorem f_is_bitwise_or : ∀ (a b : ℕ+), f a b = bitwiseOr a b := by sorry

end NUMINAMATH_CALUDE_f_is_bitwise_or_l1167_116773


namespace NUMINAMATH_CALUDE_equality_of_polynomials_l1167_116708

theorem equality_of_polynomials (a b c : ℝ) :
  (∀ x : ℝ, (x^2 + a*x - 3)*(x + 1) = x^3 + b*x^2 + c*x - 3) →
  b - c = 4 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_polynomials_l1167_116708


namespace NUMINAMATH_CALUDE_sqrt_not_arithmetic_if_geometric_not_arithmetic_l1167_116756

theorem sqrt_not_arithmetic_if_geometric_not_arithmetic
  (a b c : ℝ)
  (positive_a : 0 < a)
  (positive_b : 0 < b)
  (positive_c : 0 < c)
  (geometric_sequence : b^2 = a * c)
  (not_arithmetic_sequence : ¬(a + c = 2 * b)) :
  ¬(Real.sqrt a + Real.sqrt c = 2 * Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_not_arithmetic_if_geometric_not_arithmetic_l1167_116756


namespace NUMINAMATH_CALUDE_invalid_paper_percentage_l1167_116715

theorem invalid_paper_percentage (total_papers : ℕ) (valid_papers : ℕ) 
  (h1 : total_papers = 400)
  (h2 : valid_papers = 240) :
  (total_papers - valid_papers) * 100 / total_papers = 40 := by
  sorry

end NUMINAMATH_CALUDE_invalid_paper_percentage_l1167_116715


namespace NUMINAMATH_CALUDE_strip_width_problem_l1167_116725

theorem strip_width_problem (width1 width2 : ℕ) 
  (h1 : width1 = 44) (h2 : width2 = 33) : 
  Nat.gcd width1 width2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_strip_width_problem_l1167_116725


namespace NUMINAMATH_CALUDE_fish_pond_population_l1167_116749

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 80 →
  second_catch = 80 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = initial_tagged / (initial_tagged * second_catch / tagged_in_second) :=
by
  sorry

#check fish_pond_population

end NUMINAMATH_CALUDE_fish_pond_population_l1167_116749


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1167_116767

/-- 
Given a toy store's revenue in three months (November, December, and January), 
prove that the ratio of January's revenue to November's revenue is 1/3.
-/
theorem toy_store_revenue_ratio 
  (revenue_nov revenue_dec revenue_jan : ℝ)
  (h1 : revenue_nov = (3/5) * revenue_dec)
  (h2 : revenue_dec = (5/2) * ((revenue_nov + revenue_jan) / 2)) :
  revenue_jan / revenue_nov = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1167_116767


namespace NUMINAMATH_CALUDE_three_digit_square_ends_with_itself_l1167_116758

theorem three_digit_square_ends_with_itself (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) → (A^2 ≡ A [ZMOD 1000]) ↔ (A = 376 ∨ A = 625) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_square_ends_with_itself_l1167_116758


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1167_116705

/-- Given a geometric sequence {a_n} where (a_5 - a_1) / (a_3 - a_1) = 3,
    prove that (a_10 - a_2) / (a_6 + a_2) = 3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : (a 5 - a 1) / (a 3 - a 1) = 3)
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) :
  (a 10 - a 2) / (a 6 + a 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1167_116705


namespace NUMINAMATH_CALUDE_inverse_g_84_l1167_116777

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem inverse_g_84 : g⁻¹ 84 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_84_l1167_116777


namespace NUMINAMATH_CALUDE_fraction_always_positive_l1167_116759

theorem fraction_always_positive (x : ℝ) : 3 / (x^2 + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_always_positive_l1167_116759


namespace NUMINAMATH_CALUDE_cafe_outdoor_tables_l1167_116751

/-- The number of indoor tables -/
def indoor_tables : ℕ := 9

/-- The number of chairs per indoor table -/
def chairs_per_indoor_table : ℕ := 10

/-- The number of chairs per outdoor table -/
def chairs_per_outdoor_table : ℕ := 3

/-- The total number of chairs -/
def total_chairs : ℕ := 123

/-- The number of outdoor tables -/
def outdoor_tables : ℕ := (total_chairs - indoor_tables * chairs_per_indoor_table) / chairs_per_outdoor_table

theorem cafe_outdoor_tables : outdoor_tables = 11 := by
  sorry

end NUMINAMATH_CALUDE_cafe_outdoor_tables_l1167_116751


namespace NUMINAMATH_CALUDE_range_of_a_open_interval_l1167_116772

theorem range_of_a_open_interval :
  (∃ a : ℝ, ∀ x : ℝ, x^2 - a*x + 1 > 0) ↔ ∃ a : ℝ, -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_open_interval_l1167_116772


namespace NUMINAMATH_CALUDE_concert_cost_theorem_l1167_116731

/-- Calculates the total cost for two people to attend a concert -/
def concert_cost (ticket_price : ℝ) (processing_fee_rate : ℝ) (parking_fee : ℝ) (entrance_fee : ℝ) : ℝ :=
  let total_ticket_cost := 2 * ticket_price
  let processing_fee := total_ticket_cost * processing_fee_rate
  let total_entrance_fee := 2 * entrance_fee
  total_ticket_cost + processing_fee + parking_fee + total_entrance_fee

/-- Theorem stating that the total cost for two people to attend the concert is $135.00 -/
theorem concert_cost_theorem : 
  concert_cost 50 0.15 10 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_concert_cost_theorem_l1167_116731


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l1167_116734

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l1167_116734


namespace NUMINAMATH_CALUDE_gcf_24_72_60_l1167_116760

theorem gcf_24_72_60 : Nat.gcd 24 (Nat.gcd 72 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_24_72_60_l1167_116760


namespace NUMINAMATH_CALUDE_trig_identity_l1167_116710

theorem trig_identity (θ : Real) (h1 : π/2 < θ ∧ θ < π) (h2 : Real.tan (θ + π/3) = 1/2) : 
  Real.sin θ + Real.sqrt 3 * Real.cos θ = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1167_116710


namespace NUMINAMATH_CALUDE_rhombus_area_l1167_116785

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 13) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1167_116785


namespace NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l1167_116795

/-- A fraction a/b is a terminating decimal if and only if b can be written as 2^m * 5^n for some non-negative integers m and n. -/
def is_terminating_decimal (a b : ℕ) : Prop :=
  ∃ (m n : ℕ), b = 2^m * 5^n

/-- 50 is the smallest positive integer n such that n/(n+150) is a terminating decimal. -/
theorem smallest_n_for_terminating_decimal :
  (∀ k : ℕ, 0 < k → k < 50 → ¬ is_terminating_decimal k (k + 150)) ∧
  is_terminating_decimal 50 200 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l1167_116795


namespace NUMINAMATH_CALUDE_function_positive_interval_implies_m_range_l1167_116729

theorem function_positive_interval_implies_m_range 
  (F : ℝ → ℝ) (m : ℝ) 
  (h_def : ∀ x, F x = -x^2 - m*x + 1) 
  (h_pos : ∀ x ∈ Set.Icc m (m+1), F x > 0) : 
  m > -Real.sqrt 2 / 2 ∧ m < 0 := by
sorry

end NUMINAMATH_CALUDE_function_positive_interval_implies_m_range_l1167_116729


namespace NUMINAMATH_CALUDE_no_zero_roots_l1167_116746

-- Define the equations
def equation1 (x : ℝ) : Prop := 5 * x^2 - 15 = 35
def equation2 (x : ℝ) : Prop := (3*x-2)^2 = (2*x)^2
def equation3 (x : ℝ) : Prop := x^2 + 3*x - 4 = 2*x + 3

-- Theorem statement
theorem no_zero_roots :
  (∀ x : ℝ, equation1 x → x ≠ 0) ∧
  (∀ x : ℝ, equation2 x → x ≠ 0) ∧
  (∀ x : ℝ, equation3 x → x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_no_zero_roots_l1167_116746


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l1167_116717

/-- A quadratic function passing through (1,0) and (5,0) with minimum value 36 -/
def quadratic (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient_sum (a b c : ℝ) :
  (quadratic a b c 1 = 0) →
  (quadratic a b c 5 = 0) →
  (∃ x, ∀ y, quadratic a b c y ≥ quadratic a b c x) →
  (∃ x, quadratic a b c x = 36) →
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l1167_116717


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1167_116766

theorem trigonometric_identity (α : ℝ) : 
  1 - Real.cos (3 * Real.pi / 2 - 3 * α) - Real.sin (3 * α / 2) ^ 2 + Real.cos (3 * α / 2) ^ 2 = 
  2 * Real.sqrt 2 * Real.cos (3 * α / 2) * Real.sin (3 * α / 2 + Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1167_116766


namespace NUMINAMATH_CALUDE_b_fourth_zero_implies_b_squared_zero_l1167_116744

theorem b_fourth_zero_implies_b_squared_zero (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_fourth_zero_implies_b_squared_zero_l1167_116744


namespace NUMINAMATH_CALUDE_equation_solution_l1167_116762

theorem equation_solution : 
  ∃ (y₁ y₂ : ℝ), y₁ = 10/3 ∧ y₂ = -10 ∧ 
  (∀ y : ℝ, (10 - y)^2 = 4*y^2 ↔ (y = y₁ ∨ y = y₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1167_116762


namespace NUMINAMATH_CALUDE_puppies_remaining_l1167_116797

def initial_puppies : ℕ := 12
def puppies_given_away : ℕ := 7

theorem puppies_remaining (initial : ℕ) (given_away : ℕ) :
  initial = initial_puppies →
  given_away = puppies_given_away →
  initial - given_away = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_puppies_remaining_l1167_116797


namespace NUMINAMATH_CALUDE_dye_arrangement_count_l1167_116720

/-- The number of ways to arrange 3 organic dyes, 2 inorganic dyes, and 2 additives -/
def total_arrangements : ℕ := sorry

/-- The condition that no two organic dyes are adjacent -/
def organic_not_adjacent (arrangement : List (Fin 7)) : Prop := sorry

/-- The number of valid arrangements where no two organic dyes are adjacent -/
def valid_arrangements : ℕ := sorry

theorem dye_arrangement_count :
  valid_arrangements = 1440 := by sorry

end NUMINAMATH_CALUDE_dye_arrangement_count_l1167_116720


namespace NUMINAMATH_CALUDE_sqrt_three_subtraction_l1167_116713

theorem sqrt_three_subtraction : 2 * Real.sqrt 3 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_subtraction_l1167_116713


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1167_116769

theorem complex_fraction_sum (a b : ℝ) (h : (2 : ℂ) / (1 - I) = a + b * I) : a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1167_116769


namespace NUMINAMATH_CALUDE_subtraction_of_large_numbers_l1167_116786

theorem subtraction_of_large_numbers : 
  2222222222222 - 1111111111111 = 1111111111111 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_large_numbers_l1167_116786


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l1167_116764

/-- The number of games played in a single-elimination tournament. -/
def gamesPlayed (n : ℕ) : ℕ :=
  n - 1

/-- Theorem: A single-elimination tournament with 21 teams requires 20 games. -/
theorem single_elimination_tournament_games :
  gamesPlayed 21 = 20 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l1167_116764


namespace NUMINAMATH_CALUDE_tetrahedron_volume_is_16_l1167_116706

/-- Represents a tetrahedron PQRS with given side lengths and base area -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ
  area_PQR : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the given tetrahedron is 16 -/
theorem tetrahedron_volume_is_16 (t : Tetrahedron) 
  (h1 : t.PQ = 6)
  (h2 : t.PR = 4)
  (h3 : t.PS = 5)
  (h4 : t.QR = 5)
  (h5 : t.QS = 6)
  (h6 : t.RS = 15/2)
  (h7 : t.area_PQR = 12) :
  tetrahedron_volume t = 16 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_is_16_l1167_116706


namespace NUMINAMATH_CALUDE_ramanujan_number_proof_l1167_116733

def hardy_number : ℂ := 4 + 6 * Complex.I

theorem ramanujan_number_proof (product : ℂ) (h : product = 40 - 24 * Complex.I) :
  ∃ (ramanujan_number : ℂ), 
    ramanujan_number * hardy_number = product ∧ 
    ramanujan_number = 76/13 - 36/13 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_ramanujan_number_proof_l1167_116733


namespace NUMINAMATH_CALUDE_cos_difference_of_complex_exponentials_l1167_116779

theorem cos_difference_of_complex_exponentials 
  (θ φ : ℝ) 
  (h1 : Complex.exp (Complex.I * θ) = 4/5 + 3/5 * Complex.I)
  (h2 : Complex.exp (Complex.I * φ) = 5/13 + 12/13 * Complex.I) : 
  Real.cos (θ - φ) = -16/65 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_of_complex_exponentials_l1167_116779


namespace NUMINAMATH_CALUDE_reflection_sum_l1167_116730

/-- Given a line y = mx + b, if the reflection of point (-3, -1) across this line is (5, 3), then m + b = 1 -/
theorem reflection_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    (x = ((-3) + 5) / 2 ∧ y = ((-1) + 3) / 2) ∧ 
    (y = m * x + b) ∧
    (m = -(5 - (-3)) / (3 - (-1))))
  → m + b = 1 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_l1167_116730


namespace NUMINAMATH_CALUDE_trains_at_initial_positions_l1167_116732

/-- Represents a metro line with a given number of stations -/
structure MetroLine where
  stations : ℕ
  roundTripTime : ℕ

/-- Theorem: After 2016 minutes, all trains are at their initial positions -/
theorem trains_at_initial_positions 
  (red : MetroLine) 
  (blue : MetroLine) 
  (green : MetroLine)
  (h_red : red.stations = 7 ∧ red.roundTripTime = 14)
  (h_blue : blue.stations = 8 ∧ blue.roundTripTime = 16)
  (h_green : green.stations = 9 ∧ green.roundTripTime = 18) :
  2016 % red.roundTripTime = 0 ∧ 
  2016 % blue.roundTripTime = 0 ∧ 
  2016 % green.roundTripTime = 0 := by
  sorry

#eval 2016 % 14  -- Should output 0
#eval 2016 % 16  -- Should output 0
#eval 2016 % 18  -- Should output 0

end NUMINAMATH_CALUDE_trains_at_initial_positions_l1167_116732


namespace NUMINAMATH_CALUDE_no_snow_probability_l1167_116721

/-- The probability of no snow for five consecutive days, given the probability of snow each day is 2/3 -/
theorem no_snow_probability (p : ℚ) (h : p = 2/3) : (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l1167_116721


namespace NUMINAMATH_CALUDE_six_distinct_objects_arrangements_l1167_116768

theorem six_distinct_objects_arrangements : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_six_distinct_objects_arrangements_l1167_116768


namespace NUMINAMATH_CALUDE_car_sales_prediction_l1167_116709

theorem car_sales_prediction (sports_cars : ℕ) (sedans : ℕ) (other_cars : ℕ) : 
  sports_cars = 35 →
  5 * sedans = 8 * sports_cars →
  sedans = 2 * other_cars →
  other_cars = 28 := by
sorry

end NUMINAMATH_CALUDE_car_sales_prediction_l1167_116709


namespace NUMINAMATH_CALUDE_solve_system_of_equations_solve_system_of_inequalities_l1167_116728

-- Part 1: System of Equations
def system_of_equations (x y : ℝ) : Prop :=
  (2 * x - y = 3) ∧ (x + y = 6)

theorem solve_system_of_equations :
  ∃ x y : ℝ, system_of_equations x y ∧ x = 3 ∧ y = 3 :=
sorry

-- Part 2: System of Inequalities
def system_of_inequalities (x : ℝ) : Prop :=
  (3 * x > x - 4) ∧ ((4 + x) / 3 > x + 2)

theorem solve_system_of_inequalities :
  ∀ x : ℝ, system_of_inequalities x ↔ -2 < x ∧ x < -1 :=
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_solve_system_of_inequalities_l1167_116728


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1167_116774

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / x) ↔ x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1167_116774


namespace NUMINAMATH_CALUDE_page_number_added_thrice_l1167_116761

/-- Given a book with n pages, if the sum of all page numbers plus twice a specific page number p equals 2046, then p = 15 -/
theorem page_number_added_thrice (n : ℕ) (p : ℕ) 
  (h : n > 0) 
  (h_sum : n * (n + 1) / 2 + 2 * p = 2046) : 
  p = 15 := by
sorry

end NUMINAMATH_CALUDE_page_number_added_thrice_l1167_116761


namespace NUMINAMATH_CALUDE_original_price_calculation_l1167_116757

theorem original_price_calculation (discount_percentage : ℝ) (discounted_price : ℝ) : 
  discount_percentage = 20 ∧ discounted_price = 96 → 
  ∃ (original_price : ℝ), original_price = 120 ∧ discounted_price = original_price * (1 - discount_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1167_116757


namespace NUMINAMATH_CALUDE_orange_harvest_theorem_l1167_116770

/-- Calculates the number of sacks of oranges after a harvest period -/
def sacks_after_harvest (sacks_harvested_per_day : ℕ) (sacks_discarded_per_day : ℕ) (harvest_days : ℕ) : ℕ :=
  (sacks_harvested_per_day - sacks_discarded_per_day) * harvest_days

/-- Proves that the number of sacks of oranges after 51 days of harvest is 153 -/
theorem orange_harvest_theorem :
  sacks_after_harvest 74 71 51 = 153 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_theorem_l1167_116770


namespace NUMINAMATH_CALUDE_fiftieth_ring_squares_l1167_116739

/-- The number of squares in the nth ring around a 3x3 centered square -/
def ring_squares (n : ℕ) : ℕ :=
  if n = 1 then 16
  else if n = 2 then 24
  else 33 + 24 * (n - 1)

/-- The 50th ring contains 1209 unit squares -/
theorem fiftieth_ring_squares :
  ring_squares 50 = 1209 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_ring_squares_l1167_116739


namespace NUMINAMATH_CALUDE_base7_subtraction_l1167_116738

/-- Converts a base-7 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 7 * acc) 0

/-- Converts a decimal number to its base-7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The first number in base 7 -/
def num1 : List Nat := [2, 4, 5, 6]

/-- The second number in base 7 -/
def num2 : List Nat := [1, 2, 3, 4]

/-- The expected difference in base 7 -/
def expected_diff : List Nat := [1, 2, 2, 2]

theorem base7_subtraction :
  toBase7 (toDecimal num1 - toDecimal num2) = expected_diff := by
  sorry

end NUMINAMATH_CALUDE_base7_subtraction_l1167_116738


namespace NUMINAMATH_CALUDE_complex_modulus_example_l1167_116726

theorem complex_modulus_example : Complex.abs (3 - 10 * Complex.I * Real.sqrt 3) = Real.sqrt 309 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l1167_116726


namespace NUMINAMATH_CALUDE_farmer_cages_solution_l1167_116716

/-- Represents the problem of determining the number of cages a farmer wants to fill -/
def farmer_cages_problem (initial_rabbits : ℕ) (additional_rabbits : ℕ) (total_rabbits : ℕ) : Prop :=
  ∃ (num_cages : ℕ) (rabbits_per_cage : ℕ),
    num_cages > 1 ∧
    initial_rabbits + additional_rabbits = total_rabbits ∧
    num_cages * rabbits_per_cage = total_rabbits

/-- The solution to the farmer's cage problem -/
theorem farmer_cages_solution :
  farmer_cages_problem 164 6 170 → ∃ (num_cages : ℕ), num_cages = 10 :=
by
  sorry

#check farmer_cages_solution

end NUMINAMATH_CALUDE_farmer_cages_solution_l1167_116716


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1167_116704

/-- Given vectors a and b where a is parallel to b, prove that |3a + 2b| = √5 -/
theorem parallel_vectors_magnitude (y : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![-2, y]
  (∃ (k : ℝ), a = k • b) →
  ‖(3 : ℝ) • a + (2 : ℝ) • b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1167_116704


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1167_116753

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1167_116753


namespace NUMINAMATH_CALUDE_even_number_induction_step_l1167_116714

theorem even_number_induction_step (P : ℕ → Prop) (k : ℕ) 
  (h_even : Even k) (h_ge_2 : k ≥ 2) (h_base : P 2) (h_k : P k) :
  (∀ n, Even n → n ≥ 2 → P n) ↔ 
  (P k → P (k + 2)) :=
sorry

end NUMINAMATH_CALUDE_even_number_induction_step_l1167_116714


namespace NUMINAMATH_CALUDE_max_radius_of_circle_l1167_116741

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the two given points
def point1 : ℝ × ℝ := (4, 0)
def point2 : ℝ × ℝ := (-4, 0)

-- Theorem statement
theorem max_radius_of_circle (C : ℝ × ℝ → ℝ → Set (ℝ × ℝ)) 
  (h1 : point1 ∈ C center radius) (h2 : point2 ∈ C center radius) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), radius ≤ 4 ∧ 
  (∀ (center' : ℝ × ℝ) (radius' : ℝ), 
    point1 ∈ C center' radius' → point2 ∈ C center' radius' → radius' ≤ radius) :=
sorry

end NUMINAMATH_CALUDE_max_radius_of_circle_l1167_116741


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_l1167_116780

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line3 (x y : ℝ) : Prop := 2 * x + 3 * y + 5 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define parallel lines
def parallel_lines (a b c d e f : ℝ) : Prop := a * e = b * d

-- Theorem statement
theorem intersection_and_parallel_line :
  ∃ (k : ℝ), ∀ (x y : ℝ),
    intersection_point x y →
    parallel_lines 2 3 k 2 3 5 →
    2 * x + 3 * y + k = 0 →
    k = -7 :=
sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_line_l1167_116780


namespace NUMINAMATH_CALUDE_correct_repetitions_per_bracelet_l1167_116740

/-- The number of pattern repetitions per bracelet -/
def repetitions_per_bracelet : ℕ := 3

/-- The number of green beads in one pattern -/
def green_beads : ℕ := 3

/-- The number of purple beads in one pattern -/
def purple_beads : ℕ := 5

/-- The number of red beads in one pattern -/
def red_beads : ℕ := 6

/-- The number of beads in one pattern -/
def beads_per_pattern : ℕ := green_beads + purple_beads + red_beads

/-- The number of pattern repetitions per necklace -/
def repetitions_per_necklace : ℕ := 5

/-- The number of necklaces -/
def number_of_necklaces : ℕ := 10

/-- The total number of beads for 1 bracelet and 10 necklaces -/
def total_beads : ℕ := 742

theorem correct_repetitions_per_bracelet :
  repetitions_per_bracelet * beads_per_pattern +
  number_of_necklaces * repetitions_per_necklace * beads_per_pattern = total_beads :=
by sorry

end NUMINAMATH_CALUDE_correct_repetitions_per_bracelet_l1167_116740


namespace NUMINAMATH_CALUDE_fraction_equality_l1167_116736

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 6) 
  (h2 : s / u = 7 / 15) : 
  (5 * p * s - 3 * q * u) / (6 * q * u - 5 * p * s) = -19 / 73 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1167_116736


namespace NUMINAMATH_CALUDE_complex_product_theorem_l1167_116700

theorem complex_product_theorem (y : ℂ) (h : y = Complex.exp (4 * Real.pi * Complex.I / 9)) :
  (3 * y^2 + y^4) * (3 * y^4 + y^8) * (3 * y^6 + y^12) * 
  (3 * y^8 + y^16) * (3 * y^10 + y^20) * (3 * y^12 + y^24) = -8 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l1167_116700


namespace NUMINAMATH_CALUDE_max_additional_spheres_is_two_l1167_116754

/-- Represents a truncated cone -/
structure TruncatedCone where
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Represents the configuration of spheres in the truncated cone -/
structure SphereConfiguration where
  cone : TruncatedCone
  O₁ : Sphere
  O₂ : Sphere

/-- Calculates the maximum number of additional spheres that can be placed in the cone -/
def maxAdditionalSpheres (config : SphereConfiguration) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of additional spheres -/
theorem max_additional_spheres_is_two (config : SphereConfiguration) :
  config.cone.height = 8 ∧
  config.O₁.radius = 2 ∧
  config.O₂.radius = 3 →
  maxAdditionalSpheres config = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_additional_spheres_is_two_l1167_116754


namespace NUMINAMATH_CALUDE_max_product_sum_l1167_116792

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 12) :
  (∀ A' M' C' : ℕ, A' + M' + C' = 12 → 
    A'*M'*C' + A'*M' + M'*C' + C'*A' ≤ A*M*C + A*M + M*C + C*A) →
  A*M*C + A*M + M*C + C*A = 112 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l1167_116792
