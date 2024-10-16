import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_ages_in_future_l2990_299084

-- Define Will's age 3 years ago
def will_age_3_years_ago : ℕ := 4

-- Define the current year (relative to the problem's frame)
def current_year : ℕ := 3

-- Define the future year we're interested in
def future_year : ℕ := 5

-- Define Will's current age
def will_current_age : ℕ := will_age_3_years_ago + current_year

-- Define Diane's current age
def diane_current_age : ℕ := 2 * will_current_age

-- Define Will's future age
def will_future_age : ℕ := will_current_age + future_year

-- Define Diane's future age
def diane_future_age : ℕ := diane_current_age + future_year

-- Theorem to prove
theorem sum_of_ages_in_future : will_future_age + diane_future_age = 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_in_future_l2990_299084


namespace NUMINAMATH_CALUDE_parallel_planes_line_parallel_l2990_299025

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_line_parallel (α β : Plane) (a : Line) :
  plane_parallel α β → line_subset_plane a β → line_parallel_plane a α :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_line_parallel_l2990_299025


namespace NUMINAMATH_CALUDE_sphere_ratio_theorem_l2990_299035

/-- Given two spheres with radii r₁ and r₂ where r₁ : r₂ = 1 : 3, 
    prove that their surface areas are in the ratio 1:9 
    and their volumes are in the ratio 1:27 -/
theorem sphere_ratio_theorem (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 3) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 9 ∧
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_ratio_theorem_l2990_299035


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l2990_299081

theorem unique_solution_quadratic_equation :
  ∀ a b : ℝ, a^2 + b^2 + 4*a - 2*b + 5 = 0 → a = -2 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l2990_299081


namespace NUMINAMATH_CALUDE_uninsured_employees_count_l2990_299008

theorem uninsured_employees_count 
  (total : ℕ) 
  (part_time : ℕ) 
  (uninsured_part_time_ratio : ℚ) 
  (neither_uninsured_nor_part_time_prob : ℚ) 
  (h1 : total = 340)
  (h2 : part_time = 54)
  (h3 : uninsured_part_time_ratio = 125 / 1000)
  (h4 : neither_uninsured_nor_part_time_prob = 5735294117647058 / 10000000000000000) :
  ∃ uninsured : ℕ, uninsured = 104 := by
  sorry


end NUMINAMATH_CALUDE_uninsured_employees_count_l2990_299008


namespace NUMINAMATH_CALUDE_student_D_most_stable_l2990_299050

/-- Represents a student in the long jump training --/
inductive Student
| A
| B
| C
| D

/-- Returns the variance of a student's performance --/
def variance (s : Student) : ℝ :=
  match s with
  | Student.A => 2.1
  | Student.B => 3.5
  | Student.C => 9
  | Student.D => 0.7

/-- Determines if a student has the most stable performance --/
def has_most_stable_performance (s : Student) : Prop :=
  ∀ t : Student, variance s ≤ variance t

/-- Theorem stating that student D has the most stable performance --/
theorem student_D_most_stable :
  has_most_stable_performance Student.D :=
sorry

end NUMINAMATH_CALUDE_student_D_most_stable_l2990_299050


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l2990_299097

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) 
                                  (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 30 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 23 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l2990_299097


namespace NUMINAMATH_CALUDE_inequality_implication_l2990_299046

theorem inequality_implication (a b : ℝ) : a < b → -a + 3 > -b + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2990_299046


namespace NUMINAMATH_CALUDE_students_in_both_competitions_l2990_299085

/-- The number of students who participated in both Go and Chess competitions -/
def both_competitions (total : ℕ) (go : ℕ) (chess : ℕ) : ℕ :=
  go + chess - total

/-- Theorem stating the number of students in both competitions -/
theorem students_in_both_competitions :
  both_competitions 32 18 23 = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_competitions_l2990_299085


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2990_299060

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    left vertex A, top vertex B, right focus F, and midpoint M of AB,
    prove that the eccentricity e is in the range (0, -1+√3] 
    if 2⋅MA⋅MF + |BF|² ≥ 0 -/
theorem ellipse_eccentricity_range (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * (a/2 * (c + a/2) + b/2 * (-b/2)) + (b^2 + c^2) ≥ 0) :
  let e := c / a
  ∃ (e : ℝ), 0 < e ∧ e ≤ -1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2990_299060


namespace NUMINAMATH_CALUDE_pounds_in_ton_l2990_299010

theorem pounds_in_ton (ounces_per_pound : ℕ) (num_packets : ℕ) (packet_weight_pounds : ℕ) 
  (packet_weight_ounces : ℕ) (bag_capacity_tons : ℕ) :
  ounces_per_pound = 16 →
  num_packets = 1680 →
  packet_weight_pounds = 16 →
  packet_weight_ounces = 4 →
  bag_capacity_tons = 13 →
  ∃ (pounds_per_ton : ℕ), pounds_per_ton = 2100 :=
by
  sorry

#check pounds_in_ton

end NUMINAMATH_CALUDE_pounds_in_ton_l2990_299010


namespace NUMINAMATH_CALUDE_work_time_proof_l2990_299019

theorem work_time_proof (a b c h : ℝ) : 
  (1 / a + 1 / b + 1 / c = 1 / (a - 6)) →
  (1 / a + 1 / b + 1 / c = 1 / (b - 1)) →
  (1 / a + 1 / b + 1 / c = 2 / c) →
  (1 / a + 1 / b = 1 / h) →
  (a > 0) → (b > 0) → (c > 0) → (h > 0) →
  h = 4/3 := by
sorry

end NUMINAMATH_CALUDE_work_time_proof_l2990_299019


namespace NUMINAMATH_CALUDE_team_a_win_probabilities_l2990_299057

/-- Probability of Team A winning a single game -/
def p_a : ℝ := 0.6

/-- Probability of Team B winning a single game -/
def p_b : ℝ := 0.4

/-- Sum of probabilities for a single game is 1 -/
axiom prob_sum : p_a + p_b = 1

/-- Probability of Team A winning in a best-of-three format -/
def p_a_bo3 : ℝ := p_a^2 + 2 * p_a^2 * p_b

/-- Probability of Team A winning in a best-of-five format -/
def p_a_bo5 : ℝ := p_a^3 + 3 * p_a^3 * p_b + 6 * p_a^3 * p_b^2

/-- Theorem: Probabilities of Team A winning in best-of-three and best-of-five formats -/
theorem team_a_win_probabilities : 
  p_a_bo3 = 0.648 ∧ p_a_bo5 = 0.68256 :=
sorry

end NUMINAMATH_CALUDE_team_a_win_probabilities_l2990_299057


namespace NUMINAMATH_CALUDE_negation_of_exists_square_greater_than_power_of_two_l2990_299099

theorem negation_of_exists_square_greater_than_power_of_two :
  (¬ ∃ (n : ℕ+), n.val ^ 2 > 2 ^ n.val) ↔ ∀ (n : ℕ+), n.val ^ 2 ≤ 2 ^ n.val :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_square_greater_than_power_of_two_l2990_299099


namespace NUMINAMATH_CALUDE_cube_equation_solution_l2990_299086

theorem cube_equation_solution (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 45 * 35) : x = 35 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l2990_299086


namespace NUMINAMATH_CALUDE_more_green_than_blue_l2990_299041

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : Nat
  ratio : Fin 3 → Nat
  sum_ratio : ratio 0 + ratio 1 + ratio 2 = 18

theorem more_green_than_blue (bag : DiskBag) 
  (h_total : bag.total = 54)
  (h_ratio : bag.ratio = ![3, 7, 8]) :
  (bag.total * bag.ratio 2) / 18 - (bag.total * bag.ratio 0) / 18 = 15 := by
  sorry

#check more_green_than_blue

end NUMINAMATH_CALUDE_more_green_than_blue_l2990_299041


namespace NUMINAMATH_CALUDE_typists_productivity_l2990_299024

/-- Given that 10 typists can type 20 letters in 20 minutes, 
    prove that 40 typists working at the same rate for 1 hour will complete 240 letters. -/
theorem typists_productivity 
  (base_typists : ℕ) 
  (base_letters : ℕ) 
  (base_minutes : ℕ) 
  (new_typists : ℕ) 
  (new_minutes : ℕ)
  (h1 : base_typists = 10)
  (h2 : base_letters = 20)
  (h3 : base_minutes = 20)
  (h4 : new_typists = 40)
  (h5 : new_minutes = 60) :
  (new_typists * new_minutes * base_letters) / (base_typists * base_minutes) = 240 :=
sorry

end NUMINAMATH_CALUDE_typists_productivity_l2990_299024


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2990_299098

def simple_interest_rate : ℚ := 8 / 100
def simple_interest_time : ℕ := 5
def compound_principal : ℕ := 8000
def compound_interest_rate : ℚ := 15 / 100
def compound_interest_time : ℕ := 2

def compound_interest (P : ℕ) (r : ℚ) (t : ℕ) : ℚ :=
  P * ((1 + r) ^ t - 1)

theorem simple_interest_principal :
  ∃ (P : ℕ), 
    (P : ℚ) * simple_interest_rate * simple_interest_time = 
    (1 / 2) * compound_interest compound_principal compound_interest_rate compound_interest_time ∧
    P = 3225 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2990_299098


namespace NUMINAMATH_CALUDE_largest_number_proof_l2990_299062

theorem largest_number_proof (a b c : ℝ) 
  (sum_eq : a + b + c = 100)
  (larger_diff : c - b = 8)
  (smaller_diff : b - a = 5) :
  c = 121 / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_proof_l2990_299062


namespace NUMINAMATH_CALUDE_cube_root_of_product_l2990_299051

theorem cube_root_of_product (a : ℕ) : a^3 = 21 * 35 * 45 * 35 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l2990_299051


namespace NUMINAMATH_CALUDE_borrowed_amount_l2990_299088

/-- Represents the financial transaction described in the problem -/
structure Transaction where
  amount : ℝ  -- The amount borrowed/lent
  borrowRate : ℝ  -- Borrowing interest rate (as a decimal)
  lendRate : ℝ  -- Lending interest rate (as a decimal)
  years : ℝ  -- Duration of the transaction in years
  yearlyGain : ℝ  -- Gain per year

/-- Calculates the total gain over the entire period -/
def totalGain (t : Transaction) : ℝ :=
  (t.lendRate - t.borrowRate) * t.amount * t.years

/-- The main theorem that proves the borrowed amount given the conditions -/
theorem borrowed_amount (t : Transaction) 
    (h1 : t.years = 2)
    (h2 : t.borrowRate = 0.04)
    (h3 : t.lendRate = 0.06)
    (h4 : t.yearlyGain = 80) :
    t.amount = 2000 := by
  sorry

#check borrowed_amount

end NUMINAMATH_CALUDE_borrowed_amount_l2990_299088


namespace NUMINAMATH_CALUDE_union_complement_when_a_is_one_subset_iff_a_in_range_l2990_299078

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2*x + a ∧ 2*x + a ≤ 3}
def B : Set ℝ := {x | -1/2 < x ∧ x < 2}

-- Theorem 1
theorem union_complement_when_a_is_one :
  (Set.univ \ B) ∪ (A 1) = {x | x ≤ 1 ∨ x ≥ 2} := by sorry

-- Theorem 2
theorem subset_iff_a_in_range :
  ∀ a : ℝ, A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_union_complement_when_a_is_one_subset_iff_a_in_range_l2990_299078


namespace NUMINAMATH_CALUDE_line_passes_through_134_iff_a_gt_third_l2990_299014

/-- A line passes through the first, third, and fourth quadrants if and only if its slope is positive -/
axiom passes_through_134_iff_positive_slope (m : ℝ) (b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) ↔ m > 0

/-- The main theorem: the line y = (3a-1)x - 1 passes through the first, third, and fourth quadrants
    if and only if a > 1/3 -/
theorem line_passes_through_134_iff_a_gt_third (a : ℝ) : 
  (∀ x y : ℝ, y = (3*a - 1) * x - 1 → 
    (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) ↔ a > 1/3 :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_134_iff_a_gt_third_l2990_299014


namespace NUMINAMATH_CALUDE_solution_set_1_correct_solution_set_2_correct_l2990_299012

-- Define the solution set for the first inequality
def solution_set_1 : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the solution set for the second inequality based on the value of a
def solution_set_2 (a : ℝ) : Set ℝ :=
  if a = -2 then Set.univ
  else if a > -2 then {x | x ≤ -2 ∨ x ≥ a}
  else {x | x ≤ a ∨ x ≥ -2}

-- Theorem for the first inequality
theorem solution_set_1_correct :
  ∀ x : ℝ, x ∈ solution_set_1 ↔ (2 * x) / (x + 1) < 1 :=
sorry

-- Theorem for the second inequality
theorem solution_set_2_correct :
  ∀ a x : ℝ, x ∈ solution_set_2 a ↔ x^2 + (2 - a) * x - 2 * a ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_1_correct_solution_set_2_correct_l2990_299012


namespace NUMINAMATH_CALUDE_mack_journal_pages_l2990_299054

/-- The number of pages Mack writes on Monday -/
def monday_pages : ℕ := 60 / 30

/-- The number of pages Mack writes on Tuesday -/
def tuesday_pages : ℕ := 45 / 15

/-- The number of pages Mack writes on Wednesday -/
def wednesday_pages : ℕ := 5

/-- The total number of pages Mack writes from Monday to Wednesday -/
def total_pages : ℕ := monday_pages + tuesday_pages + wednesday_pages

theorem mack_journal_pages : total_pages = 10 := by sorry

end NUMINAMATH_CALUDE_mack_journal_pages_l2990_299054


namespace NUMINAMATH_CALUDE_samson_sandwich_count_l2990_299029

/-- The number of sandwiches Samson ate at lunch on Monday -/
def lunch_sandwiches : ℕ := sorry

/-- The number of sandwiches Samson ate at dinner on Monday -/
def dinner_sandwiches : ℕ := 2 * lunch_sandwiches

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_breakfast : ℕ := 1

/-- The total number of sandwiches Samson ate on Monday -/
def monday_total : ℕ := lunch_sandwiches + dinner_sandwiches

/-- The total number of sandwiches Samson ate on Tuesday -/
def tuesday_total : ℕ := tuesday_breakfast

theorem samson_sandwich_count : lunch_sandwiches = 3 := by
  have h1 : monday_total = tuesday_total + 8 := by sorry
  sorry

end NUMINAMATH_CALUDE_samson_sandwich_count_l2990_299029


namespace NUMINAMATH_CALUDE_period_length_l2990_299028

theorem period_length 
  (total_duration : ℕ) 
  (num_periods : ℕ) 
  (break_duration : ℕ) 
  (num_breaks : ℕ) :
  total_duration = 220 →
  num_periods = 5 →
  break_duration = 5 →
  num_breaks = 4 →
  (total_duration - num_breaks * break_duration) / num_periods = 40 :=
by sorry

end NUMINAMATH_CALUDE_period_length_l2990_299028


namespace NUMINAMATH_CALUDE_range_of_f_real_l2990_299038

def f (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem range_of_f_real : Set.range f = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_real_l2990_299038


namespace NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angle_has_10_sides_l2990_299005

theorem regular_polygon_with_144_degree_angle_has_10_sides :
  ∀ n : ℕ,
  n ≥ 3 →
  (n - 2) * 180 / n = 144 →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angle_has_10_sides_l2990_299005


namespace NUMINAMATH_CALUDE_julies_savings_l2990_299066

/-- Represents the initial savings amount in each account -/
def P : ℝ := sorry

/-- Represents the annual interest rate (as a decimal) -/
def r : ℝ := sorry

/-- Theorem stating that given the conditions, Julie's initial total savings was $1000 -/
theorem julies_savings : 
  (P * r * 2 = 100) →  -- Simple interest earned after 2 years
  (P * ((1 + r)^2 - 1) = 105) →  -- Compound interest earned after 2 years
  (2 * P = 1000) :=  -- Total initial savings
by sorry

end NUMINAMATH_CALUDE_julies_savings_l2990_299066


namespace NUMINAMATH_CALUDE_rectangle_width_l2990_299073

/-- A rectangle with length twice its width and perimeter equal to its area has width 3. -/
theorem rectangle_width (w : ℝ) (h1 : w > 0) : 
  (6 * w = 2 * w ^ 2) → w = 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_l2990_299073


namespace NUMINAMATH_CALUDE_taxi_occupancy_l2990_299002

theorem taxi_occupancy (cars : Nat) (car_capacity : Nat) (vans : Nat) (van_capacity : Nat) 
  (taxis : Nat) (total_people : Nat) :
  cars = 3 → car_capacity = 4 → vans = 2 → van_capacity = 5 → taxis = 6 → total_people = 58 →
  ∃ (taxi_capacity : Nat), taxi_capacity = 6 ∧ 
    cars * car_capacity + vans * van_capacity + taxis * taxi_capacity = total_people :=
by sorry

end NUMINAMATH_CALUDE_taxi_occupancy_l2990_299002


namespace NUMINAMATH_CALUDE_trip_cost_is_1050_l2990_299047

-- Define the distances and costs
def distance_AB : ℝ := 4000
def distance_BC : ℝ := 3000
def bus_rate : ℝ := 0.15
def plane_rate : ℝ := 0.12
def plane_booking_fee : ℝ := 120

-- Define the total trip cost function
def total_trip_cost : ℝ :=
  (distance_AB * plane_rate + plane_booking_fee) + (distance_BC * bus_rate)

-- Theorem statement
theorem trip_cost_is_1050 : total_trip_cost = 1050 := by
  sorry

end NUMINAMATH_CALUDE_trip_cost_is_1050_l2990_299047


namespace NUMINAMATH_CALUDE_function_depends_on_one_arg_l2990_299044

/-- A function that depends only on one of its arguments -/
def DependsOnOneArg {α : Type*} {k : ℕ} (f : (Fin k → α) → α) : Prop :=
  ∃ i : Fin k, ∀ x y : Fin k → α, (x i = y i) → (f x = f y)

/-- The main theorem -/
theorem function_depends_on_one_arg
  {n : ℕ} (h_n : n ≥ 3) (k : ℕ) (f : (Fin k → Fin n) → Fin n)
  (h_f : ∀ x y : Fin k → Fin n, (∀ i, x i ≠ y i) → f x ≠ f y) :
  DependsOnOneArg f := by
  sorry

end NUMINAMATH_CALUDE_function_depends_on_one_arg_l2990_299044


namespace NUMINAMATH_CALUDE_new_car_distance_l2990_299004

theorem new_car_distance (old_car_speed : ℝ) (old_car_distance : ℝ) (speed_increase : ℝ) :
  old_car_distance = 150 →
  speed_increase = 0.3 →
  old_car_speed * (1 + speed_increase) * (old_car_distance / old_car_speed) = 195 :=
by sorry

end NUMINAMATH_CALUDE_new_car_distance_l2990_299004


namespace NUMINAMATH_CALUDE_modes_of_test_scores_l2990_299096

/-- Represents a frequency distribution of test scores -/
def FrequencyDistribution := List (Nat × Nat)

/-- Finds the modes (most frequent scores) in a frequency distribution -/
def findModes (scores : FrequencyDistribution) : List Nat :=
  sorry

/-- The actual frequency distribution of the test scores -/
def testScores : FrequencyDistribution := [
  (62, 1), (65, 2), (70, 1), (74, 2), (78, 1),
  (81, 2), (86, 1), (89, 1), (92, 1), (97, 3),
  (101, 4), (104, 4), (110, 3)
]

theorem modes_of_test_scores :
  findModes testScores = [101, 104] :=
sorry

end NUMINAMATH_CALUDE_modes_of_test_scores_l2990_299096


namespace NUMINAMATH_CALUDE_three_digit_squares_ending_with_self_l2990_299065

theorem three_digit_squares_ending_with_self (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) ∧ (A^2 ≡ A [ZMOD 1000]) ↔ A = 376 ∨ A = 625 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_squares_ending_with_self_l2990_299065


namespace NUMINAMATH_CALUDE_unique_valid_cube_configuration_l2990_299067

-- Define a cube face
inductive Face
| White
| Gray
| Mixed

-- Define a cube
structure Cube :=
(front back left right top bottom : Face)

-- Define the conditions
def oppositeFacesValid (c : Cube) : Prop :=
  (c.front = Face.White → c.back = Face.Gray) ∧
  (c.left = Face.White → c.right = Face.Gray) ∧
  (c.top = Face.White → c.bottom = Face.Gray)

def adjacentFacesValid (c : Cube) : Prop :=
  (c.front = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed ∧ 
                          c.left ≠ Face.Mixed ∧ c.right ≠ Face.Mixed) ∧
  (c.back = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed ∧ 
                         c.left ≠ Face.Mixed ∧ c.right ≠ Face.Mixed) ∧
  (c.left = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed) ∧
  (c.right = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed)

-- Theorem stating the uniqueness of the valid cube configuration
theorem unique_valid_cube_configuration :
  ∃! c : Cube, oppositeFacesValid c ∧ adjacentFacesValid c :=
sorry

end NUMINAMATH_CALUDE_unique_valid_cube_configuration_l2990_299067


namespace NUMINAMATH_CALUDE_prime_power_sum_l2990_299023

theorem prime_power_sum (p q r : ℕ) : 
  p.Prime → q.Prime → r.Prime → p^q + q^p = r → 
  ((p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17)) :=
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l2990_299023


namespace NUMINAMATH_CALUDE_fruit_shop_quantities_l2990_299053

/-- Represents the quantities and prices of fruits in a shop --/
structure FruitShop where
  apple_quantity : ℝ
  pear_quantity : ℝ
  apple_price : ℝ
  pear_price : ℝ
  apple_profit_rate : ℝ
  pear_price_ratio : ℝ

/-- Theorem stating the correct quantities of apples and pears purchased --/
theorem fruit_shop_quantities (shop : FruitShop) 
  (total_weight : shop.apple_quantity + shop.pear_quantity = 200)
  (apple_price : shop.apple_price = 15)
  (pear_price : shop.pear_price = 10)
  (apple_profit : shop.apple_profit_rate = 0.4)
  (pear_price_ratio : shop.pear_price_ratio = 2/3)
  (total_profit : 
    shop.apple_quantity * shop.apple_price * shop.apple_profit_rate + 
    shop.pear_quantity * (shop.apple_price * (1 + shop.apple_profit_rate) * shop.pear_price_ratio - shop.pear_price) = 1020) :
  shop.apple_quantity = 110 ∧ shop.pear_quantity = 90 := by
  sorry

end NUMINAMATH_CALUDE_fruit_shop_quantities_l2990_299053


namespace NUMINAMATH_CALUDE_room_length_l2990_299092

/-- Given a rectangular room with width 12 m, surrounded by a 2 m wide veranda on all sides,
    if the area of the veranda is 132 m², then the length of the room is 17 m. -/
theorem room_length (room_width : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_width = 12 →
  veranda_width = 2 →
  veranda_area = 132 →
  ∃ (room_length : ℝ),
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) -
    room_length * room_width = veranda_area ∧
    room_length = 17 :=
by sorry

end NUMINAMATH_CALUDE_room_length_l2990_299092


namespace NUMINAMATH_CALUDE_dogs_can_prevent_wolf_escape_l2990_299068

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square plot -/
structure Square where
  side : ℝ
  center : Point

/-- Represents an animal (wolf or dog) -/
structure Animal where
  position : Point
  speed : ℝ

/-- Represents the game state -/
structure GameState where
  square : Square
  wolf : Animal
  dogs : List Animal

/-- Checks if a point is inside or on the boundary of a square -/
def isInsideSquare (s : Square) (p : Point) : Prop :=
  abs (p.x - s.center.x) ≤ s.side / 2 ∧ abs (p.y - s.center.y) ≤ s.side / 2

/-- Checks if a point is on the boundary of a square -/
def isOnSquareBoundary (s : Square) (p : Point) : Prop :=
  (abs (p.x - s.center.x) = s.side / 2 ∧ abs (p.y - s.center.y) ≤ s.side / 2) ∨
  (abs (p.x - s.center.x) ≤ s.side / 2 ∧ abs (p.y - s.center.y) = s.side / 2)

/-- Theorem: Dogs can prevent the wolf from escaping -/
theorem dogs_can_prevent_wolf_escape (g : GameState) 
  (h1 : g.wolf.position = g.square.center) 
  (h2 : ∀ d ∈ g.dogs, isOnSquareBoundary g.square d.position)
  (h3 : ∀ d ∈ g.dogs, d.speed = 1.5 * g.wolf.speed)
  (h4 : g.dogs.length = 4) :
  ∀ t : ℝ, ∃ strategy : ℝ → List Point, 
    (∀ p ∈ strategy t, isOnSquareBoundary g.square p) ∧ 
    isInsideSquare g.square (g.wolf.position) :=
sorry

end NUMINAMATH_CALUDE_dogs_can_prevent_wolf_escape_l2990_299068


namespace NUMINAMATH_CALUDE_circle_properties_l2990_299007

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m+3)*x + 2*(1-4*m^2)*y + 16*m^4 + 9 = 0

-- Define the theorem
theorem circle_properties :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) →
  ((-1/7 < m ∧ m < 1) ∧
   (∃ r : ℝ, 0 < r ∧ r ≤ 4 * Real.sqrt 7 / 7 ∧
    ∀ x y : ℝ, circle_equation x y m → (x - (m+3))^2 + (y - (4*m^2-1))^2 = r^2) ∧
   (∀ y : ℝ, (∃ x : ℝ, circle_equation x y m) → y ≥ -1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2990_299007


namespace NUMINAMATH_CALUDE_jeans_price_increase_l2990_299033

theorem jeans_price_increase (C : ℝ) (C_pos : C > 0) : 
  let retailer_price := 1.40 * C
  let customer_price := 1.54 * C
  (customer_price - retailer_price) / retailer_price * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_jeans_price_increase_l2990_299033


namespace NUMINAMATH_CALUDE_existence_of_distinct_pairs_l2990_299013

theorem existence_of_distinct_pairs 
  (S T : Type) [Finite S] [Finite T] 
  (U : Set (S × T)) 
  (h1 : ∀ s : S, ∃ t : T, (s, t) ∉ U) 
  (h2 : ∀ t : T, ∃ s : S, (s, t) ∈ U) :
  ∃ (s₁ s₂ : S) (t₁ t₂ : T), 
    s₁ ≠ s₂ ∧ t₁ ≠ t₂ ∧ 
    (s₁, t₁) ∈ U ∧ (s₂, t₂) ∈ U ∧ 
    (s₁, t₂) ∉ U ∧ (s₂, t₁) ∉ U :=
by sorry

end NUMINAMATH_CALUDE_existence_of_distinct_pairs_l2990_299013


namespace NUMINAMATH_CALUDE_solution_pairs_l2990_299001

theorem solution_pairs : ∃! (s : Set (ℝ × ℝ)), 
  s = {(1 + Real.sqrt 2, 1 - Real.sqrt 2), (1 - Real.sqrt 2, 1 + Real.sqrt 2)} ∧
  ∀ (x y : ℝ), (x, y) ∈ s ↔ 
    (x^2 + y^2 = (6 - x^2) + (6 - y^2)) ∧ 
    (x^2 - y^2 = (x - 2)^2 + (y - 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l2990_299001


namespace NUMINAMATH_CALUDE_selling_price_loss_l2990_299076

/-- Represents the ratio of selling price to cost price -/
def price_ratio : ℚ := 2 / 5

/-- The loss percentage when selling price is less than cost price -/
def loss_percent (r : ℚ) : ℚ := (1 - r) * 100

theorem selling_price_loss :
  price_ratio = 2 / 5 →
  loss_percent price_ratio = 60 := by
sorry

end NUMINAMATH_CALUDE_selling_price_loss_l2990_299076


namespace NUMINAMATH_CALUDE_jackies_tree_climb_l2990_299080

theorem jackies_tree_climb (h : ℝ) : 
  h > 0 →                             -- Height is positive
  (h + h/2 + h/2 + (h + 200)) / 4 = 800 →  -- Average height condition
  h = 1000 := by
sorry

end NUMINAMATH_CALUDE_jackies_tree_climb_l2990_299080


namespace NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l2990_299090

theorem longest_segment_in_quarter_circle (d : ℝ) (h : d = 10) :
  let r := d / 2
  let m := r * Real.sqrt 2
  m ^ 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l2990_299090


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2990_299077

theorem gcd_of_specific_numbers : 
  let m : ℕ := 555555555
  let n : ℕ := 1111111111
  Nat.gcd m n = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2990_299077


namespace NUMINAMATH_CALUDE_total_profit_calculation_l2990_299040

/-- The total profit of a business partnership given investments and one partner's profit share -/
theorem total_profit_calculation (p_investment q_investment : ℚ) (q_profit_share : ℚ) : 
  p_investment = 54000 →
  q_investment = 36000 →
  q_profit_share = 6001.89 →
  (p_investment + q_investment) / q_investment * q_profit_share = 15004.725 :=
by
  sorry

#eval (54000 + 36000) / 36000 * 6001.89

end NUMINAMATH_CALUDE_total_profit_calculation_l2990_299040


namespace NUMINAMATH_CALUDE_square_area_ratio_l2990_299072

theorem square_area_ratio (x : ℝ) (hx : x > 0) :
  (x^2) / ((3*x)^2) = 1/9 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2990_299072


namespace NUMINAMATH_CALUDE_no_rational_solution_l2990_299083

theorem no_rational_solution : ¬∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
  (1 : ℚ) / (x - y)^2 + (1 : ℚ) / (y - z)^2 + (1 : ℚ) / (z - x)^2 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l2990_299083


namespace NUMINAMATH_CALUDE_house_length_calculation_l2990_299059

/-- Given a house with width 10 feet and a porch measuring 6 feet by 4.5 feet,
    if 232 square feet of shingles are needed to roof both the house and the porch,
    then the length of the house is 20.5 feet. -/
theorem house_length_calculation (house_width porch_length porch_width total_shingle_area : ℝ) :
  house_width = 10 →
  porch_length = 6 →
  porch_width = 4.5 →
  total_shingle_area = 232 →
  ∃ house_length : ℝ,
    house_length * house_width + porch_length * porch_width = total_shingle_area ∧
    house_length = 20.5 :=
by sorry

end NUMINAMATH_CALUDE_house_length_calculation_l2990_299059


namespace NUMINAMATH_CALUDE_not_A_inter_B_eq_open_closed_interval_l2990_299087

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | |x - 1| > 2}

def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem not_A_inter_B_eq_open_closed_interval : 
  (Aᶜ ∩ B) = {x | 2 < x ∧ x ≤ 3} :=
sorry

end NUMINAMATH_CALUDE_not_A_inter_B_eq_open_closed_interval_l2990_299087


namespace NUMINAMATH_CALUDE_log_weight_l2990_299058

theorem log_weight (log_length : ℕ) (weight_per_foot : ℕ) (cut_pieces : ℕ) : 
  log_length = 20 → 
  weight_per_foot = 150 → 
  cut_pieces = 2 → 
  (log_length / cut_pieces) * weight_per_foot = 1500 :=
by sorry

end NUMINAMATH_CALUDE_log_weight_l2990_299058


namespace NUMINAMATH_CALUDE_room_dimension_l2990_299032

theorem room_dimension (b h d : ℝ) (hb : b = 8) (hh : h = 9) (hd : d = 17) :
  ∃ l : ℝ, l = 12 ∧ d^2 = l^2 + b^2 + h^2 := by sorry

end NUMINAMATH_CALUDE_room_dimension_l2990_299032


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2990_299091

theorem arithmetic_calculation : 1323 + 150 / 50 * 3 - 223 = 1109 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2990_299091


namespace NUMINAMATH_CALUDE_even_function_implies_k_equals_one_l2990_299011

/-- A function f is even if f(-x) = f(x) for all x in its domain --/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = kx^2 + (k-1)x + 3 --/
def f (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + (k - 1) * x + 3

/-- If f(x) = kx^2 + (k-1)x + 3 is an even function, then k = 1 --/
theorem even_function_implies_k_equals_one :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_k_equals_one_l2990_299011


namespace NUMINAMATH_CALUDE_invitation_methods_count_l2990_299039

-- Define the total number of students
def total_students : ℕ := 10

-- Define the number of students to be invited
def invited_students : ℕ := 6

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem invitation_methods_count :
  combination total_students invited_students - combination (total_students - 2) (invited_students - 2) = 140 := by
  sorry

end NUMINAMATH_CALUDE_invitation_methods_count_l2990_299039


namespace NUMINAMATH_CALUDE_water_purifier_max_profit_l2990_299093

/-- Represents the cost and selling prices of water purifiers --/
structure WaterPurifier where
  costA : ℕ  -- Cost price of A
  costB : ℕ  -- Cost price of B
  sellA : ℕ  -- Selling price of A
  sellB : ℕ  -- Selling price of B

/-- Calculates the maximum profit for selling water purifiers --/
def maxProfit (w : WaterPurifier) (total : ℕ) : ℕ :=
  let profitA := w.sellA - w.costA
  let profitB := w.sellB - w.costB
  let numA := min (total / 2) (total - (total / 2))
  profitA * numA + profitB * (total - numA)

/-- Theorem stating the maximum profit for the given scenario --/
theorem water_purifier_max_profit :
  ∀ (w : WaterPurifier),
    w.costA = w.costB + 300 →
    40000 / w.costA = 30000 / w.costB →
    w.sellA = 1500 →
    w.sellB = 1100 →
    maxProfit w 400 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_water_purifier_max_profit_l2990_299093


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2990_299095

-- Problem 1
theorem problem_1 : (-1/3)⁻¹ - Real.sqrt 12 - (2 - Real.sqrt 3)^0 = -4 - 2 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 : (1 + 1/2) + (2^2 - 1)/2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2990_299095


namespace NUMINAMATH_CALUDE_train_length_l2990_299049

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (length : ℝ) : 
  speed_kmh = 108 → time_sec = 50 → length = (speed_kmh * (5/18)) * time_sec → length = 1500 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2990_299049


namespace NUMINAMATH_CALUDE_head_circumference_ratio_l2990_299074

theorem head_circumference_ratio :
  let jack_circumference : ℝ := 12
  let charlie_circumference : ℝ := 9 + (jack_circumference / 2)
  let bill_circumference : ℝ := 10
  bill_circumference / charlie_circumference = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_head_circumference_ratio_l2990_299074


namespace NUMINAMATH_CALUDE_digit_sum_problem_l2990_299036

theorem digit_sum_problem (P Q : ℕ) : 
  P < 10 → Q < 10 → 
  100 * P + 10 * Q + Q + 
  100 * P + 10 * P + Q + 
  100 * Q + 10 * Q + Q = 876 → 
  P + Q = 5 := by sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l2990_299036


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2990_299018

theorem complex_fraction_equality : (1 : ℂ) / (3 * I + 1) = (1 : ℂ) / 10 + (3 : ℂ) * I / 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2990_299018


namespace NUMINAMATH_CALUDE_circle_center_not_constructible_with_straightedge_l2990_299045

-- Define a circle on a plane
def Circle : Type := sorry

-- Define a straightedge
def Straightedge : Type := sorry

-- Define a point on a plane
def Point : Type := sorry

-- Define the concept of constructing a point using a straightedge
def constructible (p : Point) (s : Straightedge) : Prop := sorry

-- Define the center of a circle
def center (c : Circle) : Point := sorry

-- Theorem statement
theorem circle_center_not_constructible_with_straightedge (c : Circle) (s : Straightedge) :
  ¬(constructible (center c) s) := by sorry

end NUMINAMATH_CALUDE_circle_center_not_constructible_with_straightedge_l2990_299045


namespace NUMINAMATH_CALUDE_lukes_trays_l2990_299022

/-- Given that Luke can carry 4 trays at a time, made 9 trips, and picked up 16 trays from the second table,
    prove that he picked up 20 trays from the first table. -/
theorem lukes_trays (trays_per_trip : ℕ) (total_trips : ℕ) (trays_second_table : ℕ)
    (h1 : trays_per_trip = 4)
    (h2 : total_trips = 9)
    (h3 : trays_second_table = 16) :
    trays_per_trip * total_trips - trays_second_table = 20 :=
by sorry

end NUMINAMATH_CALUDE_lukes_trays_l2990_299022


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2990_299043

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is an infinite set -/
def InfiniteDomain (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, f x ≠ 0

/-- There exist infinitely many real numbers x in the domain such that f(-x) = f(x) -/
def InfinitelyManySymmetricPoints (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, f (-x) = f x

theorem necessary_not_sufficient_condition (f : ℝ → ℝ) :
  InfiniteDomain f →
  (IsEven f → InfinitelyManySymmetricPoints f) ∧
  ¬(InfinitelyManySymmetricPoints f → IsEven f) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2990_299043


namespace NUMINAMATH_CALUDE_cake_muffin_probability_l2990_299063

/-- The probability of selecting a buyer who purchases neither cake mix nor muffin mix -/
theorem cake_muffin_probability (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ)
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_both : both = 17) :
  (total - (cake + muffin - both)) / total = 27 / 100 :=
by sorry

end NUMINAMATH_CALUDE_cake_muffin_probability_l2990_299063


namespace NUMINAMATH_CALUDE_two_machines_copies_l2990_299021

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  rate : ℕ  -- copies per minute

/-- Calculates the number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Theorem: Two copy machines working together make 2550 copies in 30 minutes -/
theorem two_machines_copies : 
  let machine1 : CopyMachine := ⟨30⟩
  let machine2 : CopyMachine := ⟨55⟩
  let total_time : ℕ := 30
  copies_made machine1 total_time + copies_made machine2 total_time = 2550 := by
  sorry


end NUMINAMATH_CALUDE_two_machines_copies_l2990_299021


namespace NUMINAMATH_CALUDE_min_diagonal_rectangle_l2990_299042

/-- Given a rectangle ABCD with perimeter 30 inches and width w ≥ 6 inches,
    the minimum length of diagonal AC is 7.5√2 inches. -/
theorem min_diagonal_rectangle (l w : ℝ) (h1 : l + w = 15) (h2 : w ≥ 6) :
  ∃ (AC : ℝ), AC = 7.5 * Real.sqrt 2 ∧ ∀ (AC' : ℝ), AC' ≥ AC := by
  sorry

end NUMINAMATH_CALUDE_min_diagonal_rectangle_l2990_299042


namespace NUMINAMATH_CALUDE_max_min_product_l2990_299056

theorem max_min_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 12) (prod_sum_eq : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 9 ∧ 
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l2990_299056


namespace NUMINAMATH_CALUDE_christels_initial_dolls_l2990_299030

theorem christels_initial_dolls (debelyn_initial : ℕ) (debelyn_gave : ℕ) (christel_gave : ℕ) :
  debelyn_initial = 20 →
  debelyn_gave = 2 →
  christel_gave = 5 →
  ∃ (christel_initial : ℕ) (andrena_final : ℕ),
    andrena_final = debelyn_gave + christel_gave ∧
    andrena_final = (christel_initial - christel_gave) + 2 ∧
    andrena_final = (debelyn_initial - debelyn_gave) + 3 →
    christel_initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_christels_initial_dolls_l2990_299030


namespace NUMINAMATH_CALUDE_min_value_at_two_l2990_299027

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 1

-- State the theorem
theorem min_value_at_two :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_at_two_l2990_299027


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2990_299089

theorem inequality_solution_set (m n : ℝ) : 
  (∀ x, x^2 - m*x - 6*n < 0 ↔ -3 < x ∧ x < 6) → m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2990_299089


namespace NUMINAMATH_CALUDE_exact_sequence_2007_l2990_299034

/-- An exact sequence of integers. -/
def ExactSequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n > m → a n ^ 2 - a m ^ 2 = a (n - m) * a (n + m)

/-- The 2007th term of the exact sequence with given initial conditions. -/
theorem exact_sequence_2007 (a : ℕ → ℤ) 
    (h_exact : ExactSequence a) 
    (h_init1 : a 1 = 1) 
    (h_init2 : a 2 = 0) : 
  a 2007 = -1 := by
  sorry

end NUMINAMATH_CALUDE_exact_sequence_2007_l2990_299034


namespace NUMINAMATH_CALUDE_complex_midpoint_and_distance_l2990_299031

theorem complex_midpoint_and_distance (z₁ z₂ m : ℂ) (h₁ : z₁ = -7 + 5*I) (h₂ : z₂ = 9 - 11*I) 
  (h_m : m = (z₁ + z₂) / 2) : 
  m = 1 - 3*I ∧ Complex.abs (z₁ - m) = 8*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_midpoint_and_distance_l2990_299031


namespace NUMINAMATH_CALUDE_division_problem_l2990_299017

theorem division_problem : (64 : ℝ) / 0.08 = 800 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2990_299017


namespace NUMINAMATH_CALUDE_cosine_angle_between_vectors_l2990_299079

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (5, 12)

theorem cosine_angle_between_vectors :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / (magnitude_a * magnitude_b) = 63 / 65 := by
sorry

end NUMINAMATH_CALUDE_cosine_angle_between_vectors_l2990_299079


namespace NUMINAMATH_CALUDE_oregano_basil_difference_l2990_299000

theorem oregano_basil_difference (basil : ℕ) (total : ℕ) (oregano : ℕ) :
  basil = 5 →
  total = 17 →
  oregano > 2 * basil →
  total = basil + oregano →
  oregano - 2 * basil = 2 := by
  sorry

end NUMINAMATH_CALUDE_oregano_basil_difference_l2990_299000


namespace NUMINAMATH_CALUDE_line_through_point_l2990_299052

theorem line_through_point (k : ℚ) :
  (1 - k * 5 = -2 * (-4)) → k = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2990_299052


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l2990_299006

/-- Given a parallelogram with opposite vertices (2, -3) and (14, 9),
    the intersection point of its diagonals is (8, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (14, 9)
  let midpoint := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 3) := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l2990_299006


namespace NUMINAMATH_CALUDE_concurrent_circles_and_collinearity_l2990_299070

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Circle :=
  (center : Point) (radius : ℝ)

-- Define the given conditions
def D (triangle : Triangle) : Point := sorry
def E (triangle : Triangle) : Point := sorry
def F (triangle : Triangle) : Point := sorry

-- Define the circles
def circleAEF (triangle : Triangle) : Circle := sorry
def circleBFD (triangle : Triangle) : Circle := sorry
def circleCDE (triangle : Triangle) : Circle := sorry

-- Define concurrency
def areConcurrent (c1 c2 c3 : Circle) : Prop := sorry

-- Define collinearity
def areCollinear (p1 p2 p3 : Point) : Prop := sorry

-- Define if a point lies on a circle
def liesOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Define the circumcircle of a triangle
def circumcircle (triangle : Triangle) : Circle := sorry

-- The theorem to prove
theorem concurrent_circles_and_collinearity 
  (triangle : Triangle) : 
  areConcurrent (circleAEF triangle) (circleBFD triangle) (circleCDE triangle) ∧ 
  (∃ M : Point, 
    liesOnCircle M (circleAEF triangle) ∧ 
    liesOnCircle M (circleBFD triangle) ∧ 
    liesOnCircle M (circleCDE triangle) ∧
    (liesOnCircle M (circumcircle triangle) ↔ 
      areCollinear (D triangle) (E triangle) (F triangle))) := by
  sorry

end NUMINAMATH_CALUDE_concurrent_circles_and_collinearity_l2990_299070


namespace NUMINAMATH_CALUDE_jake_coffee_drop_probability_l2990_299061

theorem jake_coffee_drop_probability 
  (trip_probability : ℝ) 
  (not_drop_probability : ℝ) 
  (h1 : trip_probability = 0.4)
  (h2 : not_drop_probability = 0.9) :
  1 - not_drop_probability = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_jake_coffee_drop_probability_l2990_299061


namespace NUMINAMATH_CALUDE_walk_distance_before_rest_l2990_299069

theorem walk_distance_before_rest 
  (total_distance : ℝ) 
  (distance_after_rest : ℝ) 
  (h1 : total_distance = 1) 
  (h2 : distance_after_rest = 0.25) : 
  total_distance - distance_after_rest = 0.75 := by
sorry

end NUMINAMATH_CALUDE_walk_distance_before_rest_l2990_299069


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2990_299026

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_2 + a_4 + a_5 + a_6 + a_8 = 25, prove that a_2 + a_8 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 2 + a 4 + a 5 + a 6 + a 8 = 25) : 
  a 2 + a 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2990_299026


namespace NUMINAMATH_CALUDE_downstream_distance_l2990_299094

-- Define the given parameters
def boat_speed : ℝ := 22
def stream_speed : ℝ := 5
def travel_time : ℝ := 3

-- Define the theorem
theorem downstream_distance :
  boat_speed + stream_speed * travel_time = 81 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_l2990_299094


namespace NUMINAMATH_CALUDE_davids_age_l2990_299071

/-- Given the ages of Uncle Bob, Emily, and David, prove David's age --/
theorem davids_age (uncle_bob_age : ℕ) (emily_age : ℕ) (david_age : ℕ) 
  (h1 : uncle_bob_age = 60)
  (h2 : emily_age = 2 * uncle_bob_age / 3)
  (h3 : david_age = emily_age - 10) : 
  david_age = 30 := by
  sorry

#check davids_age

end NUMINAMATH_CALUDE_davids_age_l2990_299071


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2990_299055

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a ternary number represented as a list of trits to its decimal equivalent -/
def ternary_to_decimal (trits : List ℕ) : ℕ :=
  trits.foldr (fun t acc => 3 * acc + t) 0

/-- The binary representation of 1101₂ -/
def binary_num : List Bool := [true, true, false, true]

/-- The ternary representation of 211₃ -/
def ternary_num : List ℕ := [2, 1, 1]

theorem product_of_binary_and_ternary :
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 286 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2990_299055


namespace NUMINAMATH_CALUDE_smallest_b_for_non_range_l2990_299037

theorem smallest_b_for_non_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 10 ≠ -6) ↔ b ≤ -7 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_non_range_l2990_299037


namespace NUMINAMATH_CALUDE_more_cylindrical_sandcastles_l2990_299016

/-- Represents the sandbox and sandcastle properties -/
structure Sandbox :=
  (base_area : ℝ)
  (sand_height : ℝ)
  (bucket_height : ℝ)
  (cylinder_base_area : ℝ)
  (m : ℕ)  -- number of cylindrical sandcastles
  (n : ℕ)  -- number of conical sandcastles

/-- Theorem stating that Masha's cylindrical sandcastles are more numerous -/
theorem more_cylindrical_sandcastles (sb : Sandbox) 
  (h1 : sb.sand_height = 1)
  (h2 : sb.bucket_height = 2)
  (h3 : sb.base_area = sb.cylinder_base_area * (sb.m + sb.n))
  (h4 : sb.base_area * sb.sand_height = 
        sb.cylinder_base_area * sb.bucket_height * sb.m + 
        (1/3) * sb.cylinder_base_area * sb.bucket_height * sb.n) :
  sb.m > sb.n := by
  sorry

end NUMINAMATH_CALUDE_more_cylindrical_sandcastles_l2990_299016


namespace NUMINAMATH_CALUDE_dividend_calculation_l2990_299048

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (share_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : share_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.05) :
  let actual_share_price := share_value * (1 + premium_rate)
  let num_shares := investment / actual_share_price
  let dividend_per_share := share_value * dividend_rate
  dividend_per_share * num_shares = 600 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2990_299048


namespace NUMINAMATH_CALUDE_shopping_remainder_l2990_299064

def initial_amount : ℕ := 109
def shirt_cost : ℕ := 11
def num_shirts : ℕ := 2
def pants_cost : ℕ := 13

theorem shopping_remainder :
  initial_amount - (shirt_cost * num_shirts + pants_cost) = 74 := by
  sorry

end NUMINAMATH_CALUDE_shopping_remainder_l2990_299064


namespace NUMINAMATH_CALUDE_extended_parallelepiped_volume_l2990_299075

/-- The volume of a set described by a rectangular parallelepiped extended by unit radius cylinders and spheres -/
theorem extended_parallelepiped_volume :
  let l : ℝ := 2  -- length
  let w : ℝ := 3  -- width
  let h : ℝ := 6  -- height
  let r : ℝ := 1  -- radius of extension

  -- Volume of the original parallelepiped
  let v_box := l * w * h

  -- Volume of outward projecting parallelepipeds
  let v_out := 2 * (r * w * h + r * l * h + r * l * w)

  -- Volume of quarter-cylinders along edges
  let edge_length := 2 * (l + w + h)
  let v_cyl := (π * r^2 / 4) * edge_length

  -- Volume of eighth-spheres at vertices
  let v_sph := 8 * ((4 / 3) * π * r^3 / 8)

  -- Total volume
  let v_total := v_box + v_out + v_cyl + v_sph

  v_total = (324 + 70 * π) / 3 :=
by sorry

end NUMINAMATH_CALUDE_extended_parallelepiped_volume_l2990_299075


namespace NUMINAMATH_CALUDE_dans_age_l2990_299009

/-- Given two people, Ben and Dan, where Ben is younger than Dan, 
    their ages sum to 53, and Ben is 25 years old, 
    prove that Dan is 28 years old. -/
theorem dans_age (ben_age dan_age : ℕ) : 
  ben_age < dan_age →
  ben_age + dan_age = 53 →
  ben_age = 25 →
  dan_age = 28 := by
  sorry

end NUMINAMATH_CALUDE_dans_age_l2990_299009


namespace NUMINAMATH_CALUDE_floor_sum_2017_l2990_299015

theorem floor_sum_2017 : 
  let floor (x : ℚ) := ⌊x⌋
  ∀ (isPrime2017 : Nat.Prime 2017),
    (floor (2017 * 3 / 11) : ℤ) + 
    (floor (2017 * 4 / 11) : ℤ) + 
    (floor (2017 * 5 / 11) : ℤ) + 
    (floor (2017 * 6 / 11) : ℤ) + 
    (floor (2017 * 7 / 11) : ℤ) + 
    (floor (2017 * 8 / 11) : ℤ) = 6048 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_2017_l2990_299015


namespace NUMINAMATH_CALUDE_pyramid_x_value_l2990_299082

/-- Represents a row in the pyramid -/
structure PyramidRow :=
  (left : ℕ) (middle : ℕ) (right : ℕ)

/-- Represents the pyramid structure -/
structure Pyramid :=
  (row2 : PyramidRow)
  (row3 : PyramidRow)
  (bottom : ℕ)

/-- Check if a pyramid is valid according to the problem conditions -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.row2.left = 10 ∧
  p.row2.right = 15 ∧
  p.row3.left = 150 ∧
  p.row3.right = 225 ∧
  p.bottom = 1800 ∧
  p.row3.left = p.row2.left * p.row2.middle ∧
  p.row3.right = p.row2.middle * p.row2.right ∧
  p.bottom = p.row3.left * p.row3.middle * p.row3.right

theorem pyramid_x_value (p : Pyramid) :
  is_valid_pyramid p → p.row2.middle = 15 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_x_value_l2990_299082


namespace NUMINAMATH_CALUDE_sculpture_cost_in_cny_l2990_299020

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℚ := 5

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 160

/-- Converts Namibian dollars to Chinese yuan -/
def nad_to_cny (nad : ℚ) : ℚ :=
  nad * (usd_to_cny / usd_to_nad)

theorem sculpture_cost_in_cny :
  nad_to_cny sculpture_cost_nad = 100 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_cny_l2990_299020


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2990_299003

theorem sqrt_equation_solution : ∃ x : ℝ, x = 1225 / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2990_299003
