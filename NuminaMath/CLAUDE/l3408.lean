import Mathlib

namespace NUMINAMATH_CALUDE_smaller_number_problem_l3408_340865

theorem smaller_number_problem (x y : ℝ) (h_sum : x + y = 18) (h_product : x * y = 80) :
  min x y = 8 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3408_340865


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3408_340804

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distributionCount (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 5 ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : distributionCount 5 3 = 5 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3408_340804


namespace NUMINAMATH_CALUDE_jerrys_average_increase_l3408_340869

theorem jerrys_average_increase :
  ∀ (initial_average : ℝ) (fourth_test_score : ℝ),
    initial_average = 90 →
    fourth_test_score = 98 →
    (3 * initial_average + fourth_test_score) / 4 = initial_average + 2 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_average_increase_l3408_340869


namespace NUMINAMATH_CALUDE_train_length_calculation_l3408_340826

/-- Calculates the length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60 →
  man_speed = 6 →
  passing_time = 23.998080153587715 →
  ∃ (train_length : ℝ), abs (train_length - 440) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l3408_340826


namespace NUMINAMATH_CALUDE_planted_field_fraction_l3408_340892

theorem planted_field_fraction (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_leg1 : a = 5) (h_leg2 : b = 12) (s : ℝ) (h_distance : 3 / 5 = s / (s + 3)) :
  (a * b / 2 - s^2) / (a * b / 2) = 13 / 40 := by
  sorry

end NUMINAMATH_CALUDE_planted_field_fraction_l3408_340892


namespace NUMINAMATH_CALUDE_ellipse_to_circle_transformation_l3408_340842

/-- Proves that the given scaling transformation transforms the ellipse into the circle -/
theorem ellipse_to_circle_transformation (x y x' y' : ℝ) :
  (x'^2 / 10 + y'^2 / 8 = 1) →
  (x' = (Real.sqrt 10 / 5) * x ∧ y' = (Real.sqrt 2 / 2) * y) →
  (x^2 + y^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_to_circle_transformation_l3408_340842


namespace NUMINAMATH_CALUDE_road_repair_hours_proof_l3408_340885

/-- The number of hours the first group works per day to repair a road -/
def hours_per_day : ℕ := 5

/-- The number of people in the first group -/
def people_group1 : ℕ := 39

/-- The number of days the first group works -/
def days_group1 : ℕ := 12

/-- The number of people in the second group -/
def people_group2 : ℕ := 30

/-- The number of days the second group works -/
def days_group2 : ℕ := 13

/-- The number of hours per day the second group works -/
def hours_group2 : ℕ := 6

theorem road_repair_hours_proof :
  people_group1 * days_group1 * hours_per_day = people_group2 * days_group2 * hours_group2 :=
by sorry

end NUMINAMATH_CALUDE_road_repair_hours_proof_l3408_340885


namespace NUMINAMATH_CALUDE_dogwood_tree_planting_l3408_340880

/-- The number of dogwood trees planted today -/
def trees_planted_today : ℕ := 41

/-- The initial number of trees in the park -/
def initial_trees : ℕ := 39

/-- The number of trees to be planted tomorrow -/
def trees_planted_tomorrow : ℕ := 20

/-- The final number of trees in the park -/
def final_trees : ℕ := 100

theorem dogwood_tree_planting :
  initial_trees + trees_planted_today + trees_planted_tomorrow = final_trees :=
by sorry

end NUMINAMATH_CALUDE_dogwood_tree_planting_l3408_340880


namespace NUMINAMATH_CALUDE_duck_cow_problem_l3408_340815

/-- Proves that in a group of ducks and cows, if the total number of legs is 28 more than twice the number of heads, then the number of cows is 14. -/
theorem duck_cow_problem (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 28) → cows = 14 := by
  sorry


end NUMINAMATH_CALUDE_duck_cow_problem_l3408_340815


namespace NUMINAMATH_CALUDE_max_consecutive_sum_15_l3408_340888

/-- The sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- A sequence of n consecutive positive integers starting from k -/
def consecutive_sum (n k : ℕ) : ℕ := n * k + triangular_number n

theorem max_consecutive_sum_15 :
  (∃ (n : ℕ), n > 0 ∧ consecutive_sum n 1 = 15) ∧
  (∀ (m : ℕ), m > 5 → consecutive_sum m 1 > 15) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_15_l3408_340888


namespace NUMINAMATH_CALUDE_ribbon_division_l3408_340850

theorem ribbon_division (total_ribbon : ℚ) (num_boxes : ℕ) : 
  total_ribbon = 5 / 12 → num_boxes = 5 → total_ribbon / num_boxes = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_division_l3408_340850


namespace NUMINAMATH_CALUDE_average_price_per_pair_l3408_340830

/-- Given the total sales and number of pairs sold, prove the average price per pair -/
theorem average_price_per_pair (total_sales : ℝ) (pairs_sold : ℕ) (h1 : total_sales = 686) (h2 : pairs_sold = 70) :
  total_sales / pairs_sold = 9.80 := by
  sorry

end NUMINAMATH_CALUDE_average_price_per_pair_l3408_340830


namespace NUMINAMATH_CALUDE_min_value_implies_a_l3408_340867

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |2*x + a|

/-- The theorem stating the relationship between the minimum value of f and the value of a -/
theorem min_value_implies_a (a : ℝ) : (∀ x : ℝ, f a x ≥ 3) ∧ (∃ x : ℝ, f a x = 3) → a = -4 ∨ a = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l3408_340867


namespace NUMINAMATH_CALUDE_system_solution_l3408_340808

theorem system_solution (a b c d x y z : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : x + y + z = 1)
  (h2 : a * x + b * y + c * z = d)
  (h3 : a^2 * x + b^2 * y + c^2 * z = d^2) :
  x = (d - c) * (b - d) / ((a - c) * (b - a)) ∧
  y = (a - d) * (c - d) / ((b - a) * (b - c)) ∧
  z = (a - d) * (d - b) / ((a - c) * (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3408_340808


namespace NUMINAMATH_CALUDE_rug_inner_length_is_three_l3408_340841

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : RectDimensions) : ℝ := d.length * d.width

/-- Represents the rug with its three regions -/
structure Rug where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem rug_inner_length_is_three (r : Rug) :
  r.inner.width = 2 →
  r.middle.length = r.inner.length + 3 →
  r.middle.width = r.inner.width + 3 →
  r.outer.length = r.middle.length + 3 →
  r.outer.width = r.middle.width + 3 →
  isArithmeticProgression (area r.inner) (area r.middle) (area r.outer) →
  r.inner.length = 3 := by
  sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_three_l3408_340841


namespace NUMINAMATH_CALUDE_lamp_switching_ratio_l3408_340879

/-- 
Given:
- n and k are positive integers of the same parity
- k ≥ n
- There are 2n lamps, initially all off
- A sequence consists of k steps, each step switches one lamp
- N is the number of k-step sequences ending with lamps 1 to n on and n+1 to 2n off
- M is the number of k-step sequences ending in the same state but not touching lamps n+1 to 2n

Prove that N/M = 2^(k-n)
-/
theorem lamp_switching_ratio (n k : ℕ) (h1 : n > 0) (h2 : k > 0) 
  (h3 : k ≥ n) (h4 : Even (n + k)) : ∃ (N M : ℕ), 
  N / M = 2^(k - n) :=
sorry

end NUMINAMATH_CALUDE_lamp_switching_ratio_l3408_340879


namespace NUMINAMATH_CALUDE_total_deduction_is_137_5_l3408_340833

/-- Represents David's hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Represents the local tax rate as a decimal -/
def local_tax_rate : ℝ := 0.025

/-- Represents the retirement fund contribution rate as a decimal -/
def retirement_rate : ℝ := 0.03

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℝ) : ℝ := dollars * 100

/-- Calculates the total deduction in cents -/
def total_deduction : ℝ :=
  dollars_to_cents (hourly_wage * local_tax_rate + hourly_wage * retirement_rate)

/-- Theorem stating that the total deduction is 137.5 cents -/
theorem total_deduction_is_137_5 : total_deduction = 137.5 := by
  sorry


end NUMINAMATH_CALUDE_total_deduction_is_137_5_l3408_340833


namespace NUMINAMATH_CALUDE_mary_nickels_problem_l3408_340858

/-- The number of nickels Mary's dad gave her -/
def nickels_from_dad (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

theorem mary_nickels_problem :
  let initial_nickels : ℕ := 7
  let final_nickels : ℕ := 12
  nickels_from_dad initial_nickels final_nickels = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_problem_l3408_340858


namespace NUMINAMATH_CALUDE_three_cubes_sum_equals_three_to_fourth_l3408_340868

theorem three_cubes_sum_equals_three_to_fourth : 3^3 + 3^3 + 3^3 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_three_cubes_sum_equals_three_to_fourth_l3408_340868


namespace NUMINAMATH_CALUDE_school_time_problem_l3408_340831

/-- Given a boy who reaches school 6 minutes early when walking at 7/6 of his usual rate,
    his usual time to reach the school is 42 minutes. -/
theorem school_time_problem (usual_time : ℝ) (usual_rate : ℝ) : 
  (usual_rate / usual_time = (7/6 * usual_rate) / (usual_time - 6)) → 
  usual_time = 42 := by
  sorry

end NUMINAMATH_CALUDE_school_time_problem_l3408_340831


namespace NUMINAMATH_CALUDE_parcel_boxes_count_l3408_340848

/-- Represents the position of a parcel in a rectangular arrangement of boxes -/
structure ParcelPosition where
  left : Nat
  right : Nat
  front : Nat
  back : Nat

/-- Calculates the total number of parcel boxes given the position of a specific parcel -/
def totalParcelBoxes (pos : ParcelPosition) : Nat :=
  (pos.left + pos.right - 1) * (pos.front + pos.back - 1)

/-- Theorem stating that given the specific parcel position, the total number of boxes is 399 -/
theorem parcel_boxes_count (pos : ParcelPosition) 
  (h_left : pos.left = 7)
  (h_right : pos.right = 13)
  (h_front : pos.front = 8)
  (h_back : pos.back = 14) : 
  totalParcelBoxes pos = 399 := by
  sorry

#eval totalParcelBoxes ⟨7, 13, 8, 14⟩

end NUMINAMATH_CALUDE_parcel_boxes_count_l3408_340848


namespace NUMINAMATH_CALUDE_prime_counting_inequality_characterize_equality_cases_l3408_340874

-- Define π(x) as the prime counting function
def prime_counting (x : ℕ) : ℕ := sorry

-- Define φ(x) as Euler's totient function
def euler_totient (x : ℕ) : ℕ := sorry

theorem prime_counting_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  prime_counting m - prime_counting n ≤ ((m - 1) * euler_totient n) / n :=
sorry

def equality_cases : List (ℕ × ℕ) :=
  [(1, 1), (2, 1), (3, 1), (3, 2), (5, 2), (7, 2)]

theorem characterize_equality_cases (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (prime_counting m - prime_counting n = ((m - 1) * euler_totient n) / n) ↔
  (m, n) ∈ equality_cases :=
sorry

end NUMINAMATH_CALUDE_prime_counting_inequality_characterize_equality_cases_l3408_340874


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3408_340886

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 21 + 7 + 12 + y) / 6 = 15 → y = 27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3408_340886


namespace NUMINAMATH_CALUDE_unique_apartment_number_l3408_340852

def is_valid_apartment_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (100 * a + 10 * c + b) +
    (100 * b + 10 * a + c) +
    (100 * b + 10 * c + a) +
    (100 * c + 10 * a + b) +
    (100 * c + 10 * b + a) = 2017

theorem unique_apartment_number :
  ∃! n : ℕ, is_valid_apartment_number n ∧ n = 425 :=
sorry

end NUMINAMATH_CALUDE_unique_apartment_number_l3408_340852


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_l3408_340864

/-- Given a point P(4, -3) on the terminal side of angle α, prove that cos(α) = 4/5 -/
theorem cos_alpha_for_point (α : Real) : 
  (∃ (P : Real × Real), P = (4, -3) ∧ P.1 = 4 * Real.cos α ∧ P.2 = 4 * Real.sin α) →
  Real.cos α = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_l3408_340864


namespace NUMINAMATH_CALUDE_M_subset_N_l3408_340873

-- Define set M
def M : Set ℝ := {x | x^2 = x}

-- Define set N
def N : Set ℝ := {x | x ≤ 1}

-- Theorem to prove
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l3408_340873


namespace NUMINAMATH_CALUDE_bisecting_line_exists_unique_l3408_340840

/-- A triangle with sides of length 6, 8, and 10 units. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10

/-- A line that intersects two sides of the triangle. -/
structure BisectingLine (T : Triangle) where
  x : ℝ  -- Intersection point on side b
  y : ℝ  -- Intersection point on side c
  hx : 0 < x ∧ x < T.b
  hy : 0 < y ∧ y < T.c

/-- The bisecting line divides the perimeter in half. -/
def bisects_perimeter (T : Triangle) (L : BisectingLine T) : Prop :=
  L.x + L.y = (T.a + T.b + T.c) / 2

/-- The bisecting line divides the area in half. -/
def bisects_area (T : Triangle) (L : BisectingLine T) : Prop :=
  L.x * L.y = (T.a * T.b) / 4

/-- The main theorem: existence and uniqueness of the bisecting line. -/
theorem bisecting_line_exists_unique (T : Triangle) :
  ∃! L : BisectingLine T, bisects_perimeter T L ∧ bisects_area T L :=
sorry

end NUMINAMATH_CALUDE_bisecting_line_exists_unique_l3408_340840


namespace NUMINAMATH_CALUDE_problem_1_2_l3408_340870

theorem problem_1_2 :
  (2 * Real.sqrt 6 + 2 / 3) * Real.sqrt 3 - Real.sqrt 32 = 2 * Real.sqrt 2 + 2 * Real.sqrt 3 / 3 ∧
  (Real.sqrt 7 + 2) * (Real.sqrt 7 - 2) - (Real.sqrt 45 + Real.sqrt 20) / Real.sqrt 5 = -10 :=
by sorry

end NUMINAMATH_CALUDE_problem_1_2_l3408_340870


namespace NUMINAMATH_CALUDE_range_of_a_l3408_340828

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2*x + 1 - a^2 < 0) →
  (a < -3 ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3408_340828


namespace NUMINAMATH_CALUDE_rational_segment_existence_l3408_340817

theorem rational_segment_existence (f : ℚ → ℤ) :
  ∃ a b : ℚ, f a + f b ≤ 2 * f ((a + b) / 2) := by
  sorry

end NUMINAMATH_CALUDE_rational_segment_existence_l3408_340817


namespace NUMINAMATH_CALUDE_probability_at_least_two_same_l3408_340876

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of at least two dice showing the same number when rolling 5 fair 8-sided dice -/
theorem probability_at_least_two_same (numSides : ℕ) (numDice : ℕ) :
  numSides = 8 → numDice = 5 →
  (1 - (numSides.factorial / (numSides - numDice).factorial) / numSides ^ numDice : ℚ) = 6512 / 8192 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_same_l3408_340876


namespace NUMINAMATH_CALUDE_jason_toy_count_l3408_340820

/-- The number of toys each person has -/
structure ToyCount where
  rachel : ℝ
  john : ℝ
  jason : ℝ

/-- The conditions of the problem -/
def toy_problem (t : ToyCount) : Prop :=
  t.rachel = 1 ∧
  t.john = t.rachel + 6.5 ∧
  t.jason = 3 * t.john

/-- Theorem stating that Jason has 22.5 toys -/
theorem jason_toy_count (t : ToyCount) (h : toy_problem t) : t.jason = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_jason_toy_count_l3408_340820


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l3408_340812

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 ∧ y = -15 →
  Real.sqrt (x^2 + y^2) = 17 :=
by sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l3408_340812


namespace NUMINAMATH_CALUDE_min_value_expression_l3408_340881

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a ≥ Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3408_340881


namespace NUMINAMATH_CALUDE_meeting_probability_in_our_tournament_l3408_340836

/-- Represents a knockout tournament --/
structure KnockoutTournament where
  total_players : Nat
  num_rounds : Nat
  random_pairing : Bool
  equal_win_chance : Bool

/-- The probability of two specific players meeting in a tournament --/
def meeting_probability (t : KnockoutTournament) : Rat :=
  sorry

/-- Our specific tournament --/
def our_tournament : KnockoutTournament :=
  { total_players := 32
  , num_rounds := 5
  , random_pairing := true
  , equal_win_chance := true }

theorem meeting_probability_in_our_tournament :
  meeting_probability our_tournament = 11097 / 167040 := by
  sorry

end NUMINAMATH_CALUDE_meeting_probability_in_our_tournament_l3408_340836


namespace NUMINAMATH_CALUDE_average_velocity_first_30_seconds_l3408_340863

-- Define the velocity function
def v (t : ℝ) : ℝ := t^2 - 3*t + 8

-- Define the time interval
def t_start : ℝ := 0
def t_end : ℝ := 30

-- Theorem statement
theorem average_velocity_first_30_seconds :
  (∫ t in t_start..t_end, v t) / (t_end - t_start) = 263 := by
  sorry

end NUMINAMATH_CALUDE_average_velocity_first_30_seconds_l3408_340863


namespace NUMINAMATH_CALUDE_diagonal_increase_l3408_340835

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := sorry

theorem diagonal_increase (n : ℕ) :
  num_diagonals (n + 1) = num_diagonals n + n - 1 :=
by sorry

end NUMINAMATH_CALUDE_diagonal_increase_l3408_340835


namespace NUMINAMATH_CALUDE_triangle_medians_theorem_l3408_340856

/-- Given a triangle with side lengths a, b, and c, and orthogonal medians m_a and m_b -/
def Triangle (a b c : ℝ) (m_a m_b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  m_a = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2) ∧
  m_b = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2) ∧
  m_a * m_b = 0  -- orthogonality condition

theorem triangle_medians_theorem {a b c m_a m_b : ℝ} (h : Triangle a b c m_a m_b) :
  let m_c := (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)
  -- 1. The medians form a right-angled triangle
  m_a^2 + m_b^2 = m_c^2 ∧
  -- 2. The inequality holds
  5*(a^2 + b^2 - c^2) ≥ 8*a*b :=
by sorry

end NUMINAMATH_CALUDE_triangle_medians_theorem_l3408_340856


namespace NUMINAMATH_CALUDE_boyds_boy_friends_percentage_l3408_340849

theorem boyds_boy_friends_percentage 
  (julian_total_friends : ℕ)
  (julian_boys_percentage : ℚ)
  (julian_girls_percentage : ℚ)
  (boyd_total_friends : ℕ)
  (h1 : julian_total_friends = 80)
  (h2 : julian_boys_percentage = 60 / 100)
  (h3 : julian_girls_percentage = 40 / 100)
  (h4 : julian_boys_percentage + julian_girls_percentage = 1)
  (h5 : boyd_total_friends = 100)
  (h6 : (julian_girls_percentage * julian_total_friends : ℚ) * 2 = boyd_total_friends - (boyd_total_friends - (julian_girls_percentage * julian_total_friends : ℚ) * 2)) :
  (boyd_total_friends - (julian_girls_percentage * julian_total_friends : ℚ) * 2) / boyd_total_friends = 36 / 100 := by
  sorry

end NUMINAMATH_CALUDE_boyds_boy_friends_percentage_l3408_340849


namespace NUMINAMATH_CALUDE_octagon_heptagon_diagonal_difference_l3408_340838

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Number of sides in a heptagon -/
def heptagon_sides : ℕ := 7

/-- The difference between the number of diagonals in an octagon and a heptagon is 6 -/
theorem octagon_heptagon_diagonal_difference :
  num_diagonals octagon_sides - num_diagonals heptagon_sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_octagon_heptagon_diagonal_difference_l3408_340838


namespace NUMINAMATH_CALUDE_expression_equals_eighteen_l3408_340859

theorem expression_equals_eighteen (x : ℝ) (h : x + 1 = 4) :
  (-3)^3 + (-3)^2 + (-3*x)^1 + 3*x^1 + 3^2 + 3^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_eighteen_l3408_340859


namespace NUMINAMATH_CALUDE_arctan_two_tan_75_minus_three_tan_15_l3408_340851

theorem arctan_two_tan_75_minus_three_tan_15 :
  Real.arctan (2 * Real.tan (75 * π / 180) - 3 * Real.tan (15 * π / 180)) = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_two_tan_75_minus_three_tan_15_l3408_340851


namespace NUMINAMATH_CALUDE_fraction_not_on_time_is_one_eighth_l3408_340832

/-- Represents the fraction of attendees who did not arrive on time at a monthly meeting -/
def fraction_not_on_time (total : ℕ) (male : ℕ) (male_on_time : ℕ) (female_on_time : ℕ) : ℚ :=
  1 - (male_on_time + female_on_time : ℚ) / total

/-- Theorem stating the fraction of attendees who did not arrive on time -/
theorem fraction_not_on_time_is_one_eighth
  (total : ℕ) (male : ℕ) (male_on_time : ℕ) (female_on_time : ℕ)
  (h_total_pos : 0 < total)
  (h_male_ratio : male = (3 * total) / 5)
  (h_male_on_time : male_on_time = (7 * male) / 8)
  (h_female_on_time : female_on_time = (9 * (total - male)) / 10) :
  fraction_not_on_time total male male_on_time female_on_time = 1/8 := by
  sorry

#check fraction_not_on_time_is_one_eighth

end NUMINAMATH_CALUDE_fraction_not_on_time_is_one_eighth_l3408_340832


namespace NUMINAMATH_CALUDE_sum_of_parts_l3408_340872

theorem sum_of_parts (x y : ℤ) (h1 : x + y = 54) (h2 : y = 34) : 10 * x + 22 * y = 948 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_l3408_340872


namespace NUMINAMATH_CALUDE_coefficient_x4_is_negative_15_l3408_340897

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 5*(x^3 - 2*x^4) + 3*(x^2 - x^4 + 2*x^6) - (2*x^4 + 5*x^3)

/-- The coefficient of x^4 in the simplified expression -/
def coefficient_x4 : ℝ := -15

/-- Theorem stating that the coefficient of x^4 in the simplified expression is -15 -/
theorem coefficient_x4_is_negative_15 :
  ∃ (f : ℝ → ℝ), ∀ x, expression x = f x + coefficient_x4 * x^4 ∧ 
  ∀ n, n ≠ 4 → (∃ c, ∀ x, f x = c * x^n + (f x - c * x^n)) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_negative_15_l3408_340897


namespace NUMINAMATH_CALUDE_calendar_date_theorem_l3408_340891

/-- Represents a monthly calendar with dates behind letters --/
structure MonthlyCalendar where
  C : ℤ  -- Date behind C
  A : ℤ  -- Date behind A
  B : ℤ  -- Date behind B
  Q : ℤ  -- Date behind Q

/-- Theorem: The difference between dates behind C and Q equals the sum of dates behind A and B --/
theorem calendar_date_theorem (cal : MonthlyCalendar) 
  (hC : cal.C = x)
  (hA : cal.A = x + 2)
  (hB : cal.B = x + 14)
  (hQ : cal.Q = -x - 16)
  : cal.C - cal.Q = cal.A + cal.B :=
by sorry

end NUMINAMATH_CALUDE_calendar_date_theorem_l3408_340891


namespace NUMINAMATH_CALUDE_equation_solution_l3408_340890

theorem equation_solution (z : ℝ) (some_number : ℝ) :
  (14 * (-1 + z) + some_number = -14 * (1 - z) - 10) →
  some_number = -10 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3408_340890


namespace NUMINAMATH_CALUDE_progress_primary_grade3_students_l3408_340843

/-- The number of students in Grade 3 of Progress Primary School -/
def total_students (num_classes : ℕ) (special_class_size : ℕ) (regular_class_size : ℕ) : ℕ :=
  special_class_size + (num_classes - 1) * regular_class_size

/-- Theorem stating the total number of students in Grade 3 of Progress Primary School -/
theorem progress_primary_grade3_students :
  total_students 10 48 50 = 48 + 9 * 50 := by
  sorry

end NUMINAMATH_CALUDE_progress_primary_grade3_students_l3408_340843


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3408_340803

theorem inequality_solution_set (x : ℝ) : 
  (-4 * x - 8 > 0) ↔ (x < -2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3408_340803


namespace NUMINAMATH_CALUDE_meeting_handshakes_l3408_340896

theorem meeting_handshakes (total_handshakes : ℕ) 
  (h1 : total_handshakes = 159) : ∃ (people second_handshakes : ℕ),
  people * (people - 1) / 2 + second_handshakes = total_handshakes ∧
  people = 18 ∧ 
  second_handshakes = 6 := by
sorry

end NUMINAMATH_CALUDE_meeting_handshakes_l3408_340896


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_condition_l3408_340811

/-- The range of m for which a line y = kx + 1 and an ellipse x²/5 + y²/m = 1 always intersect -/
theorem line_ellipse_intersection_condition (k : ℝ) :
  ∃ (m : ℝ), (∀ x y : ℝ, y = k * x + 1 → x^2 / 5 + y^2 / m = 1 → 
    (m ≥ 1 ∧ m ≠ 5)) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_condition_l3408_340811


namespace NUMINAMATH_CALUDE_circle_point_perpendicular_l3408_340802

theorem circle_point_perpendicular (m : ℝ) : m > 0 →
  (∃ (P : ℝ × ℝ), P.1^2 + P.2^2 = 1 ∧ 
    ((P.1 + m) * (P.1 - m) + (P.2 - 2) * (P.2 - 2) = 0)) →
  (3 : ℝ) - 1 = 2 := by sorry

end NUMINAMATH_CALUDE_circle_point_perpendicular_l3408_340802


namespace NUMINAMATH_CALUDE_cubic_function_extremum_l3408_340857

/-- Given a cubic function f with a local extremum at x = -1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + b

theorem cubic_function_extremum (a b : ℝ) (h1 : a > 1) 
  (h2 : f a b (-1) = 0) (h3 : f' a b (-1) = 0) :
  a = 2 ∧ b = 9 ∧ 
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, 0 ≤ f a b x ∧ f a b x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) 0, f a b x = 0) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) 0, f a b x = 4) := by
  sorry

#check cubic_function_extremum

end NUMINAMATH_CALUDE_cubic_function_extremum_l3408_340857


namespace NUMINAMATH_CALUDE_area_under_curve_l3408_340807

/-- The area enclosed by the curve y = x^2 + 1, the coordinate axes, and the line x = 1 is 4/3 -/
theorem area_under_curve : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 1
  ∫ x in (0 : ℝ)..1, f x = 4/3 := by sorry

end NUMINAMATH_CALUDE_area_under_curve_l3408_340807


namespace NUMINAMATH_CALUDE_dartboard_central_angle_l3408_340875

/-- The measure of the central angle of one section in a circular dartboard -/
def central_angle_measure (num_sections : ℕ) (section_probability : ℚ) : ℚ :=
  360 * section_probability

/-- Theorem: The central angle measure for a circular dartboard with 8 equal sections
    and 1/8 probability of landing in each section is 45 degrees -/
theorem dartboard_central_angle :
  central_angle_measure 8 (1/8) = 45 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_central_angle_l3408_340875


namespace NUMINAMATH_CALUDE_system_solution_l3408_340845

theorem system_solution :
  let x : ℚ := 57 / 31
  let y : ℚ := 195 / 62
  (3 * x - 4 * y = -7) ∧ (4 * x + 5 * y = 23) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3408_340845


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l3408_340898

theorem cubic_roots_sum_of_squares (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 12*x^2 + 47*x - 30 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  a^2 + b^2 + c^2 = 50 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l3408_340898


namespace NUMINAMATH_CALUDE_tiffany_bags_theorem_l3408_340824

/-- The total number of bags Tiffany collected over three days -/
def total_bags (initial : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  initial + day2 + day3

/-- Theorem stating that Tiffany's total bags equals 20 given the initial conditions -/
theorem tiffany_bags_theorem :
  total_bags 10 3 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_theorem_l3408_340824


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3408_340834

/-- Given vectors a, b, and c in ℝ², prove that (a-b) is perpendicular to c -/
theorem vector_perpendicular (a b c : ℝ × ℝ) 
  (ha : a = (0, 5)) 
  (hb : b = (4, -3)) 
  (hc : c = (-2, -1)) : 
  (a.1 - b.1) * c.1 + (a.2 - b.2) * c.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3408_340834


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_112_l3408_340837

theorem alpha_plus_beta_equals_112 
  (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96*x + 2210) / (x^2 + 65*x - 3510)) : 
  α + β = 112 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_112_l3408_340837


namespace NUMINAMATH_CALUDE_arrangement_counts_l3408_340844

/-- Counts the number of valid arrangements of crosses and zeros -/
def countArrangements (n : ℕ) (zeros : ℕ) : ℕ :=
  sorry

theorem arrangement_counts :
  (countArrangements 29 14 = 15) ∧ (countArrangements 28 14 = 120) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_counts_l3408_340844


namespace NUMINAMATH_CALUDE_complex_equation_l3408_340822

theorem complex_equation (z : ℂ) (h : z = 1 + I) : z^2 + 2/z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l3408_340822


namespace NUMINAMATH_CALUDE_code_deciphering_probability_l3408_340860

theorem code_deciphering_probability 
  (prob_A : ℚ) 
  (prob_B : ℚ) 
  (h_A : prob_A = 2 / 3) 
  (h_B : prob_B = 3 / 5) : 
  1 - (1 - prob_A) * (1 - prob_B) = 13 / 15 := by
  sorry

end NUMINAMATH_CALUDE_code_deciphering_probability_l3408_340860


namespace NUMINAMATH_CALUDE_fruit_gift_set_pears_l3408_340839

theorem fruit_gift_set_pears (total : ℕ) (apples : ℕ) (pears : ℕ) : 
  apples = 10 →
  (2 : ℚ) / 9 * total = apples →
  (2 : ℚ) / 5 * total = pears →
  pears = 18 := by
sorry

end NUMINAMATH_CALUDE_fruit_gift_set_pears_l3408_340839


namespace NUMINAMATH_CALUDE_number_problem_l3408_340816

theorem number_problem (x : ℚ) (h : (7/8) * x = 28) : (x + 16) * (5/16) = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3408_340816


namespace NUMINAMATH_CALUDE_probability_three_heads_is_one_eighth_l3408_340829

/-- Represents the possible outcomes of a coin flip -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the set of five coins -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (halfDollar : CoinOutcome)

/-- The probability of the penny, dime, and quarter all coming up heads -/
def probabilityThreeHeads : ℚ := 1 / 8

/-- Theorem stating that the probability of the penny, dime, and quarter
    all coming up heads when flipping five coins is 1/8 -/
theorem probability_three_heads_is_one_eighth :
  probabilityThreeHeads = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_probability_three_heads_is_one_eighth_l3408_340829


namespace NUMINAMATH_CALUDE_houses_on_block_l3408_340894

/-- Given a block of houses where:
  * The total number of pieces of junk mail for the block is 24
  * Each house receives 4 pieces of junk mail
  This theorem proves that there are 6 houses on the block. -/
theorem houses_on_block (total_mail : ℕ) (mail_per_house : ℕ) 
  (h1 : total_mail = 24) 
  (h2 : mail_per_house = 4) : 
  total_mail / mail_per_house = 6 := by
  sorry

end NUMINAMATH_CALUDE_houses_on_block_l3408_340894


namespace NUMINAMATH_CALUDE_impossible_table_l3408_340805

/-- Represents a cell in the table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the table -/
def Table := Cell → Int

/-- Two cells are adjacent if they share a side -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col + 1 = c2.col ∨ c2.col + 1 = c1.col)) ∨
  (c1.col = c2.col ∧ (c1.row + 1 = c2.row ∨ c2.row + 1 = c1.row))

/-- The table satisfies the adjacency condition -/
def satisfies_adjacency (t : Table) : Prop :=
  ∀ c1 c2 : Cell, adjacent c1 c2 → |t c1 - t c2| ≤ 18

/-- The table contains different integers -/
def all_different (t : Table) : Prop :=
  ∀ c1 c2 : Cell, c1 ≠ c2 → t c1 ≠ t c2

/-- The main theorem -/
theorem impossible_table : ¬∃ t : Table, satisfies_adjacency t ∧ all_different t := by
  sorry

end NUMINAMATH_CALUDE_impossible_table_l3408_340805


namespace NUMINAMATH_CALUDE_greatest_integer_a_l3408_340855

theorem greatest_integer_a : ∀ a : ℤ,
  (∃ x : ℤ, (x - a) * (x - 7) + 3 = 0) →
  a ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_a_l3408_340855


namespace NUMINAMATH_CALUDE_patio_layout_change_l3408_340809

/-- Represents a rectangular patio layout --/
structure PatioLayout where
  rows : ℕ
  columns : ℕ
  total_tiles : ℕ
  is_rectangular : rows * columns = total_tiles

/-- The change in patio layout --/
def change_layout (initial : PatioLayout) (row_increase : ℕ) : PatioLayout :=
  { rows := initial.rows + row_increase,
    columns := initial.total_tiles / (initial.rows + row_increase),
    total_tiles := initial.total_tiles,
    is_rectangular := sorry }

theorem patio_layout_change (initial : PatioLayout) 
  (h1 : initial.total_tiles = 30)
  (h2 : initial.rows = 5) :
  let final := change_layout initial 4
  initial.columns - final.columns = 3 := by sorry

end NUMINAMATH_CALUDE_patio_layout_change_l3408_340809


namespace NUMINAMATH_CALUDE_tom_found_four_seashells_today_l3408_340889

/-- The number of seashells Tom found yesterday -/
def yesterdays_seashells : ℕ := 7

/-- The total number of seashells Tom found -/
def total_seashells : ℕ := 11

/-- The number of seashells Tom found today -/
def todays_seashells : ℕ := total_seashells - yesterdays_seashells

theorem tom_found_four_seashells_today : todays_seashells = 4 := by
  sorry

end NUMINAMATH_CALUDE_tom_found_four_seashells_today_l3408_340889


namespace NUMINAMATH_CALUDE_expression_evaluation_l3408_340862

theorem expression_evaluation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

#eval 14 + 19 + 29

end NUMINAMATH_CALUDE_expression_evaluation_l3408_340862


namespace NUMINAMATH_CALUDE_valid_routes_l3408_340819

/-- Represents the lengths of route segments between consecutive cities --/
structure RouteLengths where
  ab : ℕ
  bc : ℕ
  cd : ℕ
  de : ℕ
  ef : ℕ

/-- Checks if the given route lengths satisfy all conditions --/
def isValidRoute (r : RouteLengths) : Prop :=
  r.ab > r.bc ∧ r.bc > r.cd ∧ r.cd > r.de ∧ r.de > r.ef ∧
  r.ab = 2 * r.ef ∧
  r.ab + r.bc + r.cd + r.de + r.ef = 53

/-- The theorem stating that only three specific combinations of route lengths are valid --/
theorem valid_routes :
  ∀ r : RouteLengths, isValidRoute r →
    (r = ⟨14, 12, 11, 9, 7⟩ ∨ r = ⟨14, 13, 11, 8, 7⟩ ∨ r = ⟨14, 13, 10, 9, 7⟩) :=
by sorry

end NUMINAMATH_CALUDE_valid_routes_l3408_340819


namespace NUMINAMATH_CALUDE_dice_roll_sum_l3408_340877

theorem dice_roll_sum (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 360 →
  a + b + c + d ≠ 20 := by
sorry

end NUMINAMATH_CALUDE_dice_roll_sum_l3408_340877


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l3408_340866

/-- A line passing through two given points intersects the y-axis at a specific point -/
theorem line_y_axis_intersection (x₁ y₁ x₂ y₂ : ℚ) 
  (h₁ : x₁ = 3) (h₂ : y₁ = 20) (h₃ : x₂ = -9) (h₄ : y₂ = -6) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0, b) = (0, 27/2) :=
by sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l3408_340866


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l3408_340899

theorem no_solution_to_inequalities : ¬∃ x : ℝ, (x / 2 ≥ 1 + x) ∧ (3 + 2*x > -3 - 3*x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l3408_340899


namespace NUMINAMATH_CALUDE_prop_one_prop_two_prop_three_prop_four_l3408_340821

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define symmetry about x = 1
def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 + x) = f (1 - x)

-- Proposition 1
theorem prop_one (h : ∀ x : ℝ, f (1 + 2*x) = f (1 - 2*x)) : 
  symmetric_about_one f := sorry

-- Proposition 2
theorem prop_two : 
  (∀ x : ℝ, f (x - 1) = f (1 - x)) → symmetric_about_one f := sorry

-- Proposition 3
theorem prop_three (h1 : ∀ x : ℝ, f x = f (-x)) 
  (h2 : ∀ x : ℝ, f (1 + x) = -f x) : symmetric_about_one f := sorry

-- Proposition 4
theorem prop_four (h1 : ∀ x : ℝ, f x = -f (-x)) 
  (h2 : ∀ x : ℝ, f x = f (-x - 2)) : symmetric_about_one f := sorry

end NUMINAMATH_CALUDE_prop_one_prop_two_prop_three_prop_four_l3408_340821


namespace NUMINAMATH_CALUDE_robins_gum_packages_robins_gum_packages_solution_l3408_340810

theorem robins_gum_packages (candy_packages : ℕ) (pieces_per_package : ℕ) (additional_pieces : ℕ) : ℕ :=
  let total_pieces := candy_packages * pieces_per_package + additional_pieces
  total_pieces / pieces_per_package

theorem robins_gum_packages_solution :
  robins_gum_packages 14 6 7 = 15 := by sorry

end NUMINAMATH_CALUDE_robins_gum_packages_robins_gum_packages_solution_l3408_340810


namespace NUMINAMATH_CALUDE_alicia_final_collection_l3408_340800

def egyptian_mask_collection (initial : ℕ) : ℕ :=
  let after_guggenheim := initial - 51
  let after_metropolitan := after_guggenheim - (after_guggenheim / 3)
  let after_louvre := after_metropolitan - (after_metropolitan / 4)
  let after_damage := after_louvre - 30
  let after_british := after_damage - (after_damage * 2 / 5)
  after_british - (after_british / 8)

theorem alicia_final_collection :
  egyptian_mask_collection 600 = 129 := by sorry

end NUMINAMATH_CALUDE_alicia_final_collection_l3408_340800


namespace NUMINAMATH_CALUDE_seating_arrangements_l3408_340861

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 10 people in a row. -/
def totalArrangements : ℕ := factorial 10

/-- The number of arrangements with 3 specific people in consecutive seats. -/
def threeConsecutive : ℕ := factorial 8 * factorial 3

/-- The number of arrangements with 2 specific people next to each other. -/
def twoTogether : ℕ := factorial 9 * factorial 2

/-- The number of arrangements satisfying both conditions. -/
def bothConditions : ℕ := factorial 7 * factorial 3 * factorial 2

/-- The number of valid seating arrangements. -/
def validArrangements : ℕ := totalArrangements - threeConsecutive - twoTogether + bothConditions

theorem seating_arrangements :
  validArrangements = 2685600 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3408_340861


namespace NUMINAMATH_CALUDE_expression_evaluation_l3408_340801

theorem expression_evaluation : (75 / 1.5) * (500 / 25) - (300 / 0.03) + (125 * 4 / 0.1) = -4000 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3408_340801


namespace NUMINAMATH_CALUDE_negative_eight_interpretations_l3408_340854

theorem negative_eight_interpretations :
  (-(- 8) = -(-8)) ∧
  (-(- 8) = (-1) * (-8)) ∧
  (-(- 8) = |(-8)|) ∧
  (-(- 8) = 8) := by
  sorry

end NUMINAMATH_CALUDE_negative_eight_interpretations_l3408_340854


namespace NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_3_7_13_l3408_340853

theorem smallest_six_digit_divisible_by_3_7_13 : ∀ n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧ n % 3 = 0 ∧ n % 7 = 0 ∧ n % 13 = 0 →
  100191 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_3_7_13_l3408_340853


namespace NUMINAMATH_CALUDE_gcd_5280_12155_l3408_340847

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5280_12155_l3408_340847


namespace NUMINAMATH_CALUDE_A_intersect_B_l3408_340878

def A : Set ℤ := {-2, 0, 2}

def f (x : ℤ) : ℤ := Int.natAbs x

def B : Set ℤ := f '' A

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l3408_340878


namespace NUMINAMATH_CALUDE_parents_can_catch_kolya_l3408_340813

/-- Represents a point in the park --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a person in the park --/
structure Person :=
  (position : Point)
  (speed : ℝ)

/-- Represents the park with its alleys --/
structure Park :=
  (square_side : ℝ)
  (alley_length : ℝ)

/-- Checks if a point is on an alley --/
def is_on_alley (park : Park) (p : Point) : Prop :=
  (p.x = 0 ∨ p.x = park.square_side ∨ p.x = park.square_side / 2) ∨
  (p.y = 0 ∨ p.y = park.square_side ∨ p.y = park.square_side / 2)

/-- Represents the state of the chase --/
structure ChaseState :=
  (park : Park)
  (kolya : Person)
  (parent1 : Person)
  (parent2 : Person)

/-- Defines what it means for parents to catch Kolya --/
def parents_catch_kolya (state : ChaseState) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ 
  ∃ (final_kolya final_parent1 final_parent2 : Point),
    is_on_alley state.park final_kolya ∧
    is_on_alley state.park final_parent1 ∧
    is_on_alley state.park final_parent2 ∧
    (final_kolya = final_parent1 ∨ final_kolya = final_parent2)

/-- The main theorem stating that parents can catch Kolya --/
theorem parents_can_catch_kolya (initial_state : ChaseState) :
  initial_state.kolya.speed = 3 * initial_state.parent1.speed ∧
  initial_state.kolya.speed = 3 * initial_state.parent2.speed ∧
  initial_state.park.square_side > 0 ∧
  initial_state.park.alley_length > 0 →
  parents_catch_kolya initial_state :=
sorry

end NUMINAMATH_CALUDE_parents_can_catch_kolya_l3408_340813


namespace NUMINAMATH_CALUDE_angle_measure_possibilities_l3408_340827

theorem angle_measure_possibilities :
  ∃! X : ℕ+, 
    ∃ Y : ℕ+, 
      (X : ℝ) + Y = 180 ∧ 
      (X : ℝ) = 3 * Y := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_possibilities_l3408_340827


namespace NUMINAMATH_CALUDE_problem_solution_l3408_340806

theorem problem_solution (x y : ℝ) : 
  let A := 2 * x^2 - x + y - 3 * x * y
  let B := x^2 - 2 * x - y + x * y
  (A - 2 * B = 3 * x + 3 * y - 5 * x * y) ∧ 
  (x + y = 4 → x * y = -1/5 → A - 2 * B = 13) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3408_340806


namespace NUMINAMATH_CALUDE_probability_all_red_by_fourth_draw_specific_l3408_340871

/-- The probability of drawing all red balls exactly by the 4th draw -/
def probability_all_red_by_fourth_draw (white_balls red_balls : ℕ) : ℚ :=
  let total_balls := white_balls + red_balls
  let prob_white := white_balls / total_balls
  let prob_red := red_balls / total_balls
  prob_white^3 * prob_red

/-- Theorem stating the probability of drawing all red balls exactly by the 4th draw
    given 8 white balls and 2 red balls in a bag -/
theorem probability_all_red_by_fourth_draw_specific :
  probability_all_red_by_fourth_draw 8 2 = 217/5000 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_red_by_fourth_draw_specific_l3408_340871


namespace NUMINAMATH_CALUDE_transmission_time_is_three_minutes_l3408_340887

/-- The number of blocks to be sent -/
def num_blocks : ℕ := 150

/-- The number of chunks per block -/
def chunks_per_block : ℕ := 256

/-- The transmission rate in chunks per second -/
def transmission_rate : ℕ := 200

/-- The time it takes to send all blocks in minutes -/
def transmission_time : ℚ :=
  (num_blocks * chunks_per_block : ℚ) / transmission_rate / 60

theorem transmission_time_is_three_minutes :
  transmission_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_transmission_time_is_three_minutes_l3408_340887


namespace NUMINAMATH_CALUDE_fraction_sum_l3408_340884

theorem fraction_sum : (2 : ℚ) / 3 + 5 / 18 - 1 / 6 = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3408_340884


namespace NUMINAMATH_CALUDE_badminton_team_lineup_count_l3408_340814

theorem badminton_team_lineup_count :
  let total_players : ℕ := 18
  let quadruplets : ℕ := 4
  let starters : ℕ := 8
  let non_quadruplets : ℕ := total_players - quadruplets
  let lineups_without_quadruplets : ℕ := Nat.choose non_quadruplets starters
  let lineups_with_one_quadruplet : ℕ := quadruplets * Nat.choose non_quadruplets (starters - 1)
  lineups_without_quadruplets + lineups_with_one_quadruplet = 16731 :=
by sorry

end NUMINAMATH_CALUDE_badminton_team_lineup_count_l3408_340814


namespace NUMINAMATH_CALUDE_odd_primes_with_eight_factors_l3408_340895

theorem odd_primes_with_eight_factors (w y : ℕ) : 
  Nat.Prime w → 
  Nat.Prime y → 
  w < y → 
  Odd w → 
  Odd y → 
  (Finset.card (Nat.divisors (2 * w * y)) = 8) → 
  w = 3 := by
sorry

end NUMINAMATH_CALUDE_odd_primes_with_eight_factors_l3408_340895


namespace NUMINAMATH_CALUDE_fourth_grade_final_count_l3408_340846

/-- Calculates the final number of students in a class given the initial count and changes throughout the year. -/
def final_student_count (initial : ℕ) (left_first : ℕ) (joined_first : ℕ) (left_second : ℕ) (joined_second : ℕ) : ℕ :=
  initial - left_first + joined_first - left_second + joined_second

/-- Theorem stating that the final number of students in the fourth grade class is 37. -/
theorem fourth_grade_final_count : 
  final_student_count 35 6 4 3 7 = 37 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_final_count_l3408_340846


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l3408_340825

theorem mistaken_multiplication (x : ℝ) : 67 * x - 59 * x = 4828 → x = 603.5 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l3408_340825


namespace NUMINAMATH_CALUDE_problem_solution_l3408_340883

theorem problem_solution (x y : ℝ) 
  (eq1 : |x| + x + y = 15)
  (eq2 : x + |y| - y = 9)
  (eq3 : y = 3*x - 7) : 
  x + y = 53/5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3408_340883


namespace NUMINAMATH_CALUDE_yearly_pet_feeding_cost_l3408_340893

-- Define the number of each type of pet
def num_geckos : ℕ := 3
def num_iguanas : ℕ := 2
def num_snakes : ℕ := 4

-- Define the monthly feeding cost for each type of pet
def gecko_cost : ℕ := 15
def iguana_cost : ℕ := 5
def snake_cost : ℕ := 10

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Theorem statement
theorem yearly_pet_feeding_cost :
  (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * snake_cost) * months_per_year = 1140 :=
by sorry

end NUMINAMATH_CALUDE_yearly_pet_feeding_cost_l3408_340893


namespace NUMINAMATH_CALUDE_total_fish_l3408_340818

def micah_fish : ℕ := 7

def kenneth_fish (m : ℕ) : ℕ := 3 * m

def matthias_fish (k : ℕ) : ℕ := k - 15

theorem total_fish :
  micah_fish + kenneth_fish micah_fish + matthias_fish (kenneth_fish micah_fish) = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l3408_340818


namespace NUMINAMATH_CALUDE_average_income_P_Q_l3408_340823

/-- Given the monthly incomes of three people P, Q, and R, prove that the average monthly income of P and Q is 5050, given certain conditions. -/
theorem average_income_P_Q (P Q R : ℕ) : 
  (Q + R) / 2 = 6250 →  -- Average income of Q and R
  (P + R) / 2 = 5200 →  -- Average income of P and R
  P = 4000 →            -- Income of P
  (P + Q) / 2 = 5050 :=  -- Average income of P and Q
by sorry

end NUMINAMATH_CALUDE_average_income_P_Q_l3408_340823


namespace NUMINAMATH_CALUDE_complement_of_intersection_l3408_340882

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 2, 4}

-- Define set N
def N : Finset Nat := {3, 4, 5}

-- Theorem statement
theorem complement_of_intersection (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2, 4} ∧ N = {3, 4, 5}) :
  (U \ (M ∩ N)) = {1, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l3408_340882
