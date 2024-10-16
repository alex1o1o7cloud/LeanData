import Mathlib

namespace NUMINAMATH_CALUDE_wicket_keeper_age_difference_l4183_418398

theorem wicket_keeper_age_difference (team_size : ℕ) (team_avg_age : ℝ) (remaining_players : ℕ) (age_difference : ℝ) :
  team_size = 11 →
  team_avg_age = 21 →
  remaining_players = 9 →
  age_difference = 1 →
  let total_age := team_size * team_avg_age
  let remaining_avg_age := team_avg_age - age_difference
  let remaining_total_age := remaining_players * remaining_avg_age
  let wicket_keeper_age := total_age - (remaining_total_age + team_avg_age)
  wicket_keeper_age - team_avg_age = 9 := by
  sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_difference_l4183_418398


namespace NUMINAMATH_CALUDE_f_properties_l4183_418324

def f_property (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) + f (x - y) + f (2 * x) = 4 * f x * f ((x + y) / 2) * f ((y - x) / 2) - 1) ∧
  f 1 = 0

theorem f_properties (f : ℝ → ℝ) (h : f_property f) :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x : ℝ, f x = f (x + 4)) ∧
  (∀ n : ℤ, f n = if n % 4 = 0 then 1 else if n % 4 = 1 ∨ n % 4 = 3 then 0 else -1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4183_418324


namespace NUMINAMATH_CALUDE_f_triple_eq_f_solutions_bound_l4183_418397

noncomputable def f (x : ℝ) : ℝ := -3 * Real.sin (Real.pi * x)

theorem f_triple_eq_f_solutions_bound :
  ∃ (S : Finset ℝ), (∀ x ∈ S, -1 ≤ x ∧ x ≤ 1) ∧ 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f (f (f x)) = f x → x ∈ S) ∧
  Finset.card S ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_f_triple_eq_f_solutions_bound_l4183_418397


namespace NUMINAMATH_CALUDE_prob_B_given_A_value_l4183_418320

/-- Represents the number of balls in the box -/
def total_balls : ℕ := 10

/-- Represents the number of black balls initially in the box -/
def black_balls : ℕ := 8

/-- Represents the number of red balls initially in the box -/
def red_balls : ℕ := 2

/-- Represents the number of balls each player draws -/
def balls_drawn : ℕ := 2

/-- Calculates the probability of player B drawing 2 black balls given that player A has drawn 2 black balls -/
def prob_B_given_A : ℚ :=
  (Nat.choose (black_balls - balls_drawn) balls_drawn) / (Nat.choose total_balls balls_drawn)

theorem prob_B_given_A_value : prob_B_given_A = 15 / 28 := by
  sorry

end NUMINAMATH_CALUDE_prob_B_given_A_value_l4183_418320


namespace NUMINAMATH_CALUDE_order_of_roots_l4183_418323

theorem order_of_roots : 3^(2/3) < 2^(4/3) ∧ 2^(4/3) < 25^(1/3) := by sorry

end NUMINAMATH_CALUDE_order_of_roots_l4183_418323


namespace NUMINAMATH_CALUDE_negative_sqrt_six_squared_equals_six_l4183_418356

theorem negative_sqrt_six_squared_equals_six : (-Real.sqrt 6)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_six_squared_equals_six_l4183_418356


namespace NUMINAMATH_CALUDE_intersection_points_distance_squared_l4183_418338

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The squared distance between two points in 2D space -/
def squaredDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The squared distance between intersection points of two specific circles is 16 -/
theorem intersection_points_distance_squared
  (c1 : Circle)
  (c2 : Circle)
  (h1 : c1.center = (1, 3))
  (h2 : c1.radius = 3)
  (h3 : c2.center = (1, -4))
  (h4 : c2.radius = 6)
  : ∃ p1 p2 : ℝ × ℝ,
    squaredDistance p1 p2 = 16 ∧
    squaredDistance p1 c1.center = c1.radius^2 ∧
    squaredDistance p1 c2.center = c2.radius^2 ∧
    squaredDistance p2 c1.center = c1.radius^2 ∧
    squaredDistance p2 c2.center = c2.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_distance_squared_l4183_418338


namespace NUMINAMATH_CALUDE_stratified_sample_size_l4183_418385

/-- Represents the total number of schools of each type -/
structure SchoolCounts where
  universities : ℕ
  middleSchools : ℕ
  primarySchools : ℕ

/-- Calculates the total number of schools -/
def totalSchools (counts : SchoolCounts) : ℕ :=
  counts.universities + counts.middleSchools + counts.primarySchools

/-- Represents the sample size for middle schools -/
def middleSchoolSample : ℕ := 10

/-- Theorem: In a stratified sampling of schools, if 10 middle schools are sampled
    from a population with 20 universities, 200 middle schools, and 480 primary schools,
    then the total sample size is 35. -/
theorem stratified_sample_size 
  (counts : SchoolCounts) 
  (h1 : counts.universities = 20) 
  (h2 : counts.middleSchools = 200) 
  (h3 : counts.primarySchools = 480) :
  (middleSchoolSample : ℚ) / counts.middleSchools = 
  (35 : ℚ) / (totalSchools counts) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l4183_418385


namespace NUMINAMATH_CALUDE_janet_ride_count_l4183_418301

theorem janet_ride_count (roller_coaster_tickets : ℕ) (giant_slide_tickets : ℕ) 
  (roller_coaster_rides : ℕ) (total_tickets : ℕ) :
  roller_coaster_tickets = 5 →
  giant_slide_tickets = 3 →
  roller_coaster_rides = 7 →
  total_tickets = 47 →
  ∃ (giant_slide_rides : ℕ), 
    roller_coaster_tickets * roller_coaster_rides + giant_slide_tickets * giant_slide_rides = total_tickets ∧
    giant_slide_rides = 4 := by
  sorry

end NUMINAMATH_CALUDE_janet_ride_count_l4183_418301


namespace NUMINAMATH_CALUDE_power_mod_five_l4183_418358

theorem power_mod_five : 2^345 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_five_l4183_418358


namespace NUMINAMATH_CALUDE_total_interest_after_trebling_l4183_418375

/-- 
Given a principal amount and an interest rate, if the simple interest 
on the principal for 10 years is 700, and the principal is trebled after 5 years, 
then the total interest at the end of the tenth year is 1750.
-/
theorem total_interest_after_trebling (P R : ℝ) : 
  (P * R * 10) / 100 = 700 → 
  ((P * R * 5) / 100) + (((3 * P) * R * 5) / 100) = 1750 := by
  sorry

#check total_interest_after_trebling

end NUMINAMATH_CALUDE_total_interest_after_trebling_l4183_418375


namespace NUMINAMATH_CALUDE_stratified_sampling_sizes_l4183_418362

/-- Represents the income groups in the community -/
inductive IncomeGroup
  | High
  | Middle
  | Low

/-- Calculates the sample size for a given income group -/
def sampleSize (totalPopulation : ℕ) (groupPopulation : ℕ) (totalSample : ℕ) : ℕ :=
  (groupPopulation * totalSample) / totalPopulation

/-- Theorem stating the correct sample sizes for each income group -/
theorem stratified_sampling_sizes :
  let totalPopulation := 600
  let highIncome := 230
  let middleIncome := 290
  let lowIncome := 80
  let totalSample := 60
  (sampleSize totalPopulation highIncome totalSample = 23) ∧
  (sampleSize totalPopulation middleIncome totalSample = 29) ∧
  (sampleSize totalPopulation lowIncome totalSample = 8) :=
by
  sorry

#check stratified_sampling_sizes

end NUMINAMATH_CALUDE_stratified_sampling_sizes_l4183_418362


namespace NUMINAMATH_CALUDE_games_given_solution_l4183_418337

/-- The number of games Henry gave to Neil -/
def games_given : ℕ := sorry

/-- Henry's initial number of games -/
def henry_initial : ℕ := 58

/-- Neil's initial number of games -/
def neil_initial : ℕ := 7

theorem games_given_solution :
  (henry_initial - games_given = 4 * (neil_initial + games_given)) ∧
  games_given = 6 := by sorry

end NUMINAMATH_CALUDE_games_given_solution_l4183_418337


namespace NUMINAMATH_CALUDE_min_d_value_l4183_418327

theorem min_d_value (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (horder : a < b ∧ b < c ∧ c < d)
  (hunique : ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a| + |x - b| + |x - c| + |x - d|) :
  d ≥ 999 ∧ ∃ (a' b' c' : ℕ), a' < b' ∧ b' < c' ∧ c' < 999 ∧
    ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a'| + |x - b'| + |x - c'| + |x - 999| :=
by
  sorry


end NUMINAMATH_CALUDE_min_d_value_l4183_418327


namespace NUMINAMATH_CALUDE_lcm_of_48_and_180_l4183_418395

theorem lcm_of_48_and_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_48_and_180_l4183_418395


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l4183_418351

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 599 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l4183_418351


namespace NUMINAMATH_CALUDE_clothing_percentage_is_fifty_percent_l4183_418399

/-- Represents the shopping breakdown and tax rates for Jill's purchases --/
structure ShoppingBreakdown where
  clothing_percentage : ℝ
  food_percentage : ℝ
  other_percentage : ℝ
  clothing_tax_rate : ℝ
  food_tax_rate : ℝ
  other_tax_rate : ℝ
  total_tax_rate : ℝ

/-- Calculates the percentage spent on clothing given the shopping breakdown --/
def calculate_clothing_percentage (sb : ShoppingBreakdown) : ℝ :=
  sb.clothing_percentage

/-- Theorem stating that the percentage spent on clothing is 50% --/
theorem clothing_percentage_is_fifty_percent (sb : ShoppingBreakdown) 
  (h1 : sb.food_percentage = 0.25)
  (h2 : sb.other_percentage = 0.25)
  (h3 : sb.clothing_tax_rate = 0.10)
  (h4 : sb.food_tax_rate = 0)
  (h5 : sb.other_tax_rate = 0.20)
  (h6 : sb.total_tax_rate = 0.10)
  (h7 : sb.clothing_percentage + sb.food_percentage + sb.other_percentage = 1) :
  calculate_clothing_percentage sb = 0.5 := by
  sorry

#eval calculate_clothing_percentage { 
  clothing_percentage := 0.5,
  food_percentage := 0.25,
  other_percentage := 0.25,
  clothing_tax_rate := 0.10,
  food_tax_rate := 0,
  other_tax_rate := 0.20,
  total_tax_rate := 0.10
}

end NUMINAMATH_CALUDE_clothing_percentage_is_fifty_percent_l4183_418399


namespace NUMINAMATH_CALUDE_complex_product_real_implies_ratio_l4183_418347

theorem complex_product_real_implies_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (r : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = r) : p / q = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_implies_ratio_l4183_418347


namespace NUMINAMATH_CALUDE_regular_pentagon_diagonal_inequality_l4183_418368

/-- A regular pentagon -/
structure RegularPentagon where
  side_length : ℝ
  diagonal_short : ℝ
  diagonal_long : ℝ
  side_length_pos : 0 < side_length

/-- The longer diagonal is greater than the shorter diagonal in a regular pentagon -/
theorem regular_pentagon_diagonal_inequality (p : RegularPentagon) : 
  p.diagonal_long > p.diagonal_short := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_diagonal_inequality_l4183_418368


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l4183_418341

/-- The line passing through (-1, 0) and perpendicular to x + y = 0 has equation x - y + 1 = 0 -/
theorem perpendicular_line_equation : 
  let c : ℝ × ℝ := (-1, 0)
  let l₁ : Set (ℝ × ℝ) := {p | p.1 + p.2 = 0}
  let l₂ : Set (ℝ × ℝ) := {p | p.1 - p.2 + 1 = 0}
  (∀ p ∈ l₂, (p.1 - c.1) * (1 + 1) = -(p.2 - c.2)) ∧ 
  c ∈ l₂ :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l4183_418341


namespace NUMINAMATH_CALUDE_robotics_workshop_average_age_l4183_418384

theorem robotics_workshop_average_age (total_members : Nat) (overall_avg : Nat) 
  (num_girls num_boys num_adults : Nat) (avg_girls avg_boys : Nat) :
  total_members = 50 →
  overall_avg = 21 →
  num_girls = 25 →
  num_boys = 20 →
  num_adults = 5 →
  avg_girls = 18 →
  avg_boys = 20 →
  (total_members * overall_avg - num_girls * avg_girls - num_boys * avg_boys) / num_adults = 40 :=
by sorry

end NUMINAMATH_CALUDE_robotics_workshop_average_age_l4183_418384


namespace NUMINAMATH_CALUDE_P_intersect_Q_equals_Q_l4183_418314

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Theorem statement
theorem P_intersect_Q_equals_Q : P ∩ Q = Q := by
  sorry

end NUMINAMATH_CALUDE_P_intersect_Q_equals_Q_l4183_418314


namespace NUMINAMATH_CALUDE_divisibility_by_1897_l4183_418342

theorem divisibility_by_1897 (n : ℕ) : 
  (1897 : ℤ) ∣ (2903^n - 803^n - 464^n + 261^n) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_1897_l4183_418342


namespace NUMINAMATH_CALUDE_blue_lipstick_count_l4183_418391

theorem blue_lipstick_count (total_students : ℕ) 
  (h1 : total_students = 200)
  (h2 : ∃ lipstick_wearers : ℕ, lipstick_wearers = total_students / 2)
  (h3 : ∃ red_lipstick_wearers : ℕ, red_lipstick_wearers = lipstick_wearers / 4)
  (h4 : ∃ blue_lipstick_wearers : ℕ, blue_lipstick_wearers = red_lipstick_wearers / 5) :
  ∃ blue_lipstick_wearers : ℕ, blue_lipstick_wearers = 5 := by
sorry

end NUMINAMATH_CALUDE_blue_lipstick_count_l4183_418391


namespace NUMINAMATH_CALUDE_adjacent_book_left_of_middle_l4183_418319

theorem adjacent_book_left_of_middle (n : ℕ) (p : ℝ) : 
  n = 862 →
  (∀ i : ℕ, i < n → p + 2 * i ≥ 0) →
  (p + 2 * ((n - 1) / 2)) + (p + 2 * (((n - 1) / 2) - 1)) = p + 2 * (n - 1) →
  (p + 2 * ((n - 1) / 2)) + (p + 2 * (((n - 1) / 2) + 1)) ≠ p + 2 * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_adjacent_book_left_of_middle_l4183_418319


namespace NUMINAMATH_CALUDE_point_on_line_l4183_418372

/-- A line passing through point (1,3) with slope 2 -/
def line_l (b : ℝ) : ℝ → ℝ := λ x ↦ 2 * x + b

/-- The y-coordinate of point P -/
def point_p_y : ℝ := 3

/-- The x-coordinate of point P -/
def point_p_x : ℝ := 1

/-- The y-coordinate of point Q -/
def point_q_y : ℝ := 5

/-- The x-coordinate of point Q -/
def point_q_x : ℝ := 2

theorem point_on_line :
  ∃ b : ℝ, line_l b point_p_x = point_p_y ∧ line_l b point_q_x = point_q_y := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l4183_418372


namespace NUMINAMATH_CALUDE_february_2020_average_rainfall_l4183_418393

/-- Calculate the average rainfall per hour in February 2020 --/
theorem february_2020_average_rainfall
  (total_rainfall : ℝ)
  (february_days : ℕ)
  (hours_per_day : ℕ)
  (h1 : total_rainfall = 290)
  (h2 : february_days = 29)
  (h3 : hours_per_day = 24) :
  total_rainfall / (february_days * hours_per_day : ℝ) = 290 / 696 :=
by sorry

end NUMINAMATH_CALUDE_february_2020_average_rainfall_l4183_418393


namespace NUMINAMATH_CALUDE_solution_to_equation_l4183_418326

theorem solution_to_equation :
  ∃ x y : ℝ, 3 * x^2 - 12 * y^2 + 6 * x = 0 ∧ x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l4183_418326


namespace NUMINAMATH_CALUDE_parallel_postulate_l4183_418360

-- Define a line in a 2D Euclidean plane
structure Line where
  -- You might represent a line using two points or a point and a direction
  -- For simplicity, we'll leave the internal representation abstract
  dummy : Unit

-- Define a point in a 2D Euclidean plane
structure Point where
  -- You might represent a point using x and y coordinates
  -- For simplicity, we'll leave the internal representation abstract
  dummy : Unit

-- Define what it means for a point to not be on a line
def Point.notOn (p : Point) (l : Line) : Prop := sorry

-- Define what it means for two lines to be parallel
def Line.parallel (l1 l2 : Line) : Prop := sorry

-- Define what it means for a line to pass through a point
def Line.passesThroughPoint (l : Line) (p : Point) : Prop := sorry

-- The parallel postulate
theorem parallel_postulate (L : Line) (P : Point) (h : P.notOn L) :
  ∃! L' : Line, L'.parallel L ∧ L'.passesThroughPoint P := by sorry

end NUMINAMATH_CALUDE_parallel_postulate_l4183_418360


namespace NUMINAMATH_CALUDE_wrapping_paper_area_correct_l4183_418309

/-- Represents a box with square base -/
structure Box where
  w : ℝ  -- width of the base
  h : ℝ  -- height of the box

/-- Calculates the area of the wrapping paper for a given box -/
def wrappingPaperArea (box : Box) : ℝ :=
  6 * box.w * box.h + box.h^2

/-- Theorem stating that the area of the wrapping paper is correct -/
theorem wrapping_paper_area_correct (box : Box) :
  wrappingPaperArea box = 6 * box.w * box.h + box.h^2 :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_correct_l4183_418309


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l4183_418303

theorem part_to_whole_ratio 
  (N P : ℚ) 
  (h1 : (1/4) * (1/3) * P = 17) 
  (h2 : (2/5) * N = 204) : 
  P/N = 2/5 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l4183_418303


namespace NUMINAMATH_CALUDE_cubic_coefficient_b_is_zero_l4183_418382

-- Define the cubic function
def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem cubic_coefficient_b_is_zero
  (a b c d : ℝ) :
  (g a b c d (-2) = 0) →
  (g a b c d 0 = 0) →
  (g a b c d 2 = 0) →
  (g a b c d 1 = -1) →
  b = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_coefficient_b_is_zero_l4183_418382


namespace NUMINAMATH_CALUDE_apple_piles_l4183_418328

/-- Given two piles of apples, prove the original number in the second pile -/
theorem apple_piles (a : ℕ) : 
  (∃ b : ℕ, (a - 2) * 2 = b + 2) → 
  (∃ b : ℕ, b = 2 * a - 6) :=
by sorry

end NUMINAMATH_CALUDE_apple_piles_l4183_418328


namespace NUMINAMATH_CALUDE_intersection_segment_length_l4183_418322

-- Define the polar curve C₁
def C₁ (ρ θ : ℝ) : Prop := ρ^2 * Real.cos (2 * θ) = 8

-- Define the line l
def l (t x y : ℝ) : Prop := x = 1 + (Real.sqrt 3 / 2) * t ∧ y = (1 / 2) * t

-- Define the Cartesian form of C₁
def C₁_cartesian (x y : ℝ) : Prop := x^2 - y^2 = 8

-- Theorem statement
theorem intersection_segment_length :
  ∃ (t₁ t₂ : ℝ),
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      l t₁ x₁ y₁ ∧ l t₂ x₂ y₂ ∧
      C₁_cartesian x₁ y₁ ∧ C₁_cartesian x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = 68) :=
sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l4183_418322


namespace NUMINAMATH_CALUDE_quadratic_has_real_roots_root_condition_implies_value_minimum_value_of_y_l4183_418350

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 2)*x + 4*k

-- Part 1: Prove that the equation always has real roots
theorem quadratic_has_real_roots (k : ℝ) :
  ∃ x : ℝ, quadratic k x = 0 := by sorry

-- Part 2: Given the condition on roots, find the value of the expression
theorem root_condition_implies_value (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : quadratic k x₁ = 0) (h2 : quadratic k x₂ = 0)
  (h3 : x₂/x₁ + x₁/x₂ - 2 = 0) :
  (1 + 4/(k^2 - 4)) * ((k + 2)/k) = -1 := by sorry

-- Part 3: Find the minimum value of y
theorem minimum_value_of_y (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : quadratic k x₁ = 0) (h2 : quadratic k x₂ = 0)
  (h3 : x₁ > x₂) (h4 : k < 1/2) :
  ∃ y_min : ℝ, y_min = 3/4 ∧ ∀ y : ℝ, y ≥ y_min → ∃ x₂ : ℝ, y = x₂^2 - k*x₁ + 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_has_real_roots_root_condition_implies_value_minimum_value_of_y_l4183_418350


namespace NUMINAMATH_CALUDE_dividend_calculation_l4183_418316

theorem dividend_calculation (dividend divisor quotient : ℕ) 
  (h1 : dividend = 5 * divisor) 
  (h2 : divisor = 4 * quotient) : 
  dividend = 100 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l4183_418316


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4183_418312

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 → 
  (∀ x, x^2 - 2*a*x - 8*a^2 < 0 ↔ x₁ < x ∧ x < x₂) → 
  x₂ - x₁ = 15 → 
  a = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4183_418312


namespace NUMINAMATH_CALUDE_jade_savings_l4183_418357

/-- Calculates Jade's monthly savings based on her income and expenses --/
def calculate_savings (
  monthly_income : ℝ)
  (contribution_401k_rate : ℝ)
  (tax_deduction_rate : ℝ)
  (living_expenses_rate : ℝ)
  (insurance_rate : ℝ)
  (transportation_rate : ℝ)
  (utilities_rate : ℝ) : ℝ :=
  let contribution_401k := monthly_income * contribution_401k_rate
  let tax_deduction := monthly_income * tax_deduction_rate
  let post_deduction_income := monthly_income - contribution_401k - tax_deduction
  let total_expenses := post_deduction_income * (living_expenses_rate + insurance_rate + transportation_rate + utilities_rate)
  post_deduction_income - total_expenses

/-- Theorem stating Jade's monthly savings --/
theorem jade_savings :
  calculate_savings 2800 0.08 0.10 0.55 0.20 0.12 0.08 = 114.80 := by
  sorry


end NUMINAMATH_CALUDE_jade_savings_l4183_418357


namespace NUMINAMATH_CALUDE_snails_removed_l4183_418308

def original_snails : ℕ := 11760
def remaining_snails : ℕ := 8278

theorem snails_removed : original_snails - remaining_snails = 3482 := by
  sorry

end NUMINAMATH_CALUDE_snails_removed_l4183_418308


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l4183_418348

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points A, B, and M
def A : ℝ × ℝ := (-2, 2)
def B : ℝ × ℝ := (-5, 5)
def M : ℝ × ℝ := (-2, 9)

-- Define the line l: x + y + 3 = 0
def l (p : ℝ × ℝ) : Prop := p.1 + p.2 + 3 = 0

-- Theorem statement
theorem circle_and_tangent_lines :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    -- C lies on line l
    l C ∧
    -- Circle passes through A and B
    A ∈ Circle C r ∧ B ∈ Circle C r ∧
    -- Standard equation of the circle
    (∀ (x y : ℝ), (x, y) ∈ Circle C r ↔ (x + 5)^2 + (y - 2)^2 = 9) ∧
    -- Tangent lines through M
    (∀ (x y : ℝ),
      ((x = -2) ∨ (20 * x - 21 * y + 229 = 0)) ↔
      ((x, y) ∈ Circle C r → (x - M.1) * (x - C.1) + (y - M.2) * (y - C.2) = 0)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l4183_418348


namespace NUMINAMATH_CALUDE_max_regions_11_rays_l4183_418325

/-- The number of regions a plane is divided into by n rays -/
def num_regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem: The maximum number of regions a plane can be divided into by 11 rays is 67 -/
theorem max_regions_11_rays : num_regions 11 = 67 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_11_rays_l4183_418325


namespace NUMINAMATH_CALUDE_product_simplification_l4183_418379

theorem product_simplification (y : ℝ) (h : y ≠ 0) :
  (21 * y^3) * (9 * y^2) * (1 / (7 * y)^2) = 27 / 7 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l4183_418379


namespace NUMINAMATH_CALUDE_median_in_70_74_l4183_418306

/-- Represents a score interval with its lower bound and student count -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (student_count : ℕ)

/-- The list of score intervals -/
def score_intervals : List ScoreInterval :=
  [⟨85, 10⟩, ⟨80, 15⟩, ⟨75, 20⟩, ⟨70, 25⟩, ⟨65, 15⟩, ⟨60, 10⟩, ⟨55, 5⟩]

/-- The total number of students -/
def total_students : ℕ := 100

/-- Find the interval containing the median score -/
def median_interval (intervals : List ScoreInterval) (total : ℕ) : Option ScoreInterval :=
  sorry

/-- Theorem: The interval containing the median score is 70-74 -/
theorem median_in_70_74 :
  median_interval score_intervals total_students = some ⟨70, 25⟩ :=
sorry

end NUMINAMATH_CALUDE_median_in_70_74_l4183_418306


namespace NUMINAMATH_CALUDE_max_points_for_28_lines_l4183_418370

/-- The maximum number of lines that can be determined by n distinct points on a plane -/
def maxLines (n : ℕ) : ℕ :=
  if n ≤ 1 then 0
  else (n * (n - 1)) / 2

/-- Theorem: 8 is the maximum number of distinct points on a plane that can determine at most 28 lines -/
theorem max_points_for_28_lines :
  (∀ n : ℕ, n ≤ 8 → maxLines n ≤ 28) ∧
  (maxLines 8 = 28) :=
sorry

end NUMINAMATH_CALUDE_max_points_for_28_lines_l4183_418370


namespace NUMINAMATH_CALUDE_market_price_calculation_l4183_418361

/-- Given an initial sales tax rate, a reduced sales tax rate, and the difference in tax amount,
    proves that the market price of an article is 6600. -/
theorem market_price_calculation (initial_rate reduced_rate : ℚ) (tax_difference : ℝ) :
  initial_rate = 35 / 1000 →
  reduced_rate = 100 / 3000 →
  tax_difference = 10.999999999999991 →
  ∃ (price : ℕ), price = 6600 ∧ (initial_rate - reduced_rate) * price = tax_difference :=
sorry

end NUMINAMATH_CALUDE_market_price_calculation_l4183_418361


namespace NUMINAMATH_CALUDE_symmetric_points_l4183_418345

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to a horizontal line -/
def isSymmetricHorizontal (p q : Point2D) (y_line : ℝ) : Prop :=
  p.x = q.x ∧ (q.y - y_line) = (y_line - p.y)

/-- Theorem: The point Q(3, 4) is symmetric to P(3, -2) with respect to y = 1 -/
theorem symmetric_points : 
  let P : Point2D := ⟨3, -2⟩
  let Q : Point2D := ⟨3, 4⟩
  let y_line : ℝ := 1
  isSymmetricHorizontal P Q y_line := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_l4183_418345


namespace NUMINAMATH_CALUDE_diverse_dates_2013_l4183_418334

/-- A date in the format DD/MM/YY -/
structure Date where
  day : Nat
  month : Nat
  year : Nat

/-- Check if a date is valid (day between 1 and 31, month between 1 and 12) -/
def Date.isValid (d : Date) : Prop :=
  1 ≤ d.day ∧ d.day ≤ 31 ∧ 1 ≤ d.month ∧ d.month ≤ 12

/-- Convert a date to a list of digits -/
def Date.toDigits (d : Date) : List Nat :=
  (d.day / 10) :: (d.day % 10) :: (d.month / 10) :: (d.month % 10) :: (d.year / 10) :: [d.year % 10]

/-- Check if a date is diverse (contains all digits from 0 to 5 exactly once) -/
def Date.isDiverse (d : Date) : Prop :=
  let digits := d.toDigits
  ∀ n : Nat, n ≤ 5 → (digits.count n = 1)

/-- The main theorem: there are exactly 2 diverse dates in 2013 -/
theorem diverse_dates_2013 :
  ∃! (dates : List Date), 
    (∀ d ∈ dates, d.year = 13 ∧ d.isValid ∧ d.isDiverse) ∧ 
    dates.length = 2 := by
  sorry

end NUMINAMATH_CALUDE_diverse_dates_2013_l4183_418334


namespace NUMINAMATH_CALUDE_range_of_m_l4183_418315

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being decreasing on [-2, 2]
def isDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ 2 → f y < f x

-- State the theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : isDecreasingOn f) 
  (h2 : f (m - 1) < f (-m)) : 
  1/2 < m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l4183_418315


namespace NUMINAMATH_CALUDE_train_delivery_wood_cars_l4183_418388

/-- Represents the train's cargo and delivery parameters -/
structure TrainDelivery where
  coal_cars : ℕ
  iron_cars : ℕ
  station_distance : ℕ
  travel_time : ℕ
  max_coal_deposit : ℕ
  max_iron_deposit : ℕ
  max_wood_deposit : ℕ
  total_delivery_time : ℕ

/-- Calculates the initial number of wood cars -/
def initial_wood_cars (td : TrainDelivery) : ℕ :=
  (td.total_delivery_time / td.travel_time) * td.max_wood_deposit

/-- Theorem stating that given the problem conditions, the initial number of wood cars is 4 -/
theorem train_delivery_wood_cars :
  let td : TrainDelivery := {
    coal_cars := 6,
    iron_cars := 12,
    station_distance := 6,
    travel_time := 25,
    max_coal_deposit := 2,
    max_iron_deposit := 3,
    max_wood_deposit := 1,
    total_delivery_time := 100
  }
  initial_wood_cars td = 4 := by
  sorry


end NUMINAMATH_CALUDE_train_delivery_wood_cars_l4183_418388


namespace NUMINAMATH_CALUDE_vacation_cost_division_l4183_418363

theorem vacation_cost_division (total_cost : ℕ) (n : ℕ) : 
  total_cost = 1000 → 
  (total_cost / 5 + 50 = total_cost / n) → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l4183_418363


namespace NUMINAMATH_CALUDE_kendras_cookies_l4183_418304

/-- Kendra's cookie baking problem -/
theorem kendras_cookies :
  ∀ (cookies_per_batch : ℕ)
    (family_size : ℕ)
    (chips_per_cookie : ℕ)
    (chips_per_person : ℕ),
  cookies_per_batch = 12 →
  family_size = 4 →
  chips_per_cookie = 2 →
  chips_per_person = 18 →
  (chips_per_person / chips_per_cookie * family_size) / cookies_per_batch = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_kendras_cookies_l4183_418304


namespace NUMINAMATH_CALUDE_vector_parallelism_l4183_418389

theorem vector_parallelism (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, x]
  (∃ (k : ℝ), (a + b) = k • (a - b)) → x = -4 := by
sorry

end NUMINAMATH_CALUDE_vector_parallelism_l4183_418389


namespace NUMINAMATH_CALUDE_problem_solution_l4183_418336

theorem problem_solution (a b c d : ℤ) 
  (h1 : a = d)
  (h2 : b = c)
  (h3 : d + d = c * d)
  (h4 : b = d)
  (h5 : d + d = d * d)
  (h6 : c = 3) :
  a * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4183_418336


namespace NUMINAMATH_CALUDE_village_plots_count_l4183_418332

theorem village_plots_count (street_length : ℝ) (narrow_width wide_width : ℝ)
  (narrow_plot_diff : ℕ) (plot_area_diff : ℝ) :
  street_length = 1200 →
  narrow_width = 50 →
  wide_width = 60 →
  narrow_plot_diff = 5 →
  plot_area_diff = 1200 →
  ∃ (wide_plots narrow_plots : ℕ),
    narrow_plots = wide_plots + narrow_plot_diff ∧
    (narrow_plots : ℝ) * (street_length * narrow_width / narrow_plots) =
      (wide_plots : ℝ) * (street_length * wide_width / wide_plots - plot_area_diff) ∧
    wide_plots + narrow_plots = 45 :=
by sorry

end NUMINAMATH_CALUDE_village_plots_count_l4183_418332


namespace NUMINAMATH_CALUDE_extreme_value_conditions_l4183_418390

def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + a^2

def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem extreme_value_conditions (a b : ℝ) : 
  f a b (-1) = 8 ∧ f_derivative a b (-1) = 0 → a = 2 ∧ b = -7 := by sorry

end NUMINAMATH_CALUDE_extreme_value_conditions_l4183_418390


namespace NUMINAMATH_CALUDE_prob_C_gets_10000_equal_expected_values_l4183_418369

/-- Represents the bonus distribution problem for a work group of three people. -/
structure BonusDistribution where
  total_bonus : ℝ
  p₁ : ℝ  -- Probability of taking 10,000 yuan
  p₂ : ℝ  -- Probability of taking 20,000 yuan

/-- The total bonus is 40,000 yuan -/
def bonus_amount : ℝ := 40000

/-- The probability of A or B taking 10,000 yuan plus the probability of taking 20,000 yuan equals 1 -/
axiom prob_sum (bd : BonusDistribution) : bd.p₁ + bd.p₂ = 1

/-- Expected bonus for A or B -/
def expected_bonus_AB (bd : BonusDistribution) : ℝ := 10000 * bd.p₁ + 20000 * bd.p₂

/-- Expected bonus for C -/
def expected_bonus_C (bd : BonusDistribution) : ℝ := 
  20000 * bd.p₁^2 + 10000 * 2 * bd.p₁ * bd.p₂

/-- Theorem: When p₁ = p₂ = 1/2, the probability that C gets 10,000 yuan is 1/2 -/
theorem prob_C_gets_10000 (bd : BonusDistribution) 
  (h₁ : bd.p₁ = 1/2) (h₂ : bd.p₂ = 1/2) : 
  bd.p₁ * bd.p₂ + bd.p₁ * bd.p₂ = 1/2 := by sorry

/-- Theorem: When expected values are equal, p₁ = 2/3 and p₂ = 1/3 -/
theorem equal_expected_values (bd : BonusDistribution) 
  (h : expected_bonus_AB bd = expected_bonus_C bd) : 
  bd.p₁ = 2/3 ∧ bd.p₂ = 1/3 := by sorry

end NUMINAMATH_CALUDE_prob_C_gets_10000_equal_expected_values_l4183_418369


namespace NUMINAMATH_CALUDE_sold_shares_value_l4183_418300

/-- Calculates the value of sold shares given the total business value,
    the fraction of ownership, and the fraction of shares sold. -/
def value_of_sold_shares (total_value : ℝ) (ownership_fraction : ℝ) (sold_fraction : ℝ) : ℝ :=
  total_value * ownership_fraction * sold_fraction

/-- Proves that selling 3/5 of 1/3 ownership of a 10000 rs business yields 2000 rs -/
theorem sold_shares_value :
  value_of_sold_shares 10000 (1/3) (3/5) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_sold_shares_value_l4183_418300


namespace NUMINAMATH_CALUDE_matrix_operation_example_l4183_418318

def matrix_operation (a b c d : ℚ) : ℚ := a * d - b * c

theorem matrix_operation_example : matrix_operation 1 2 3 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_operation_example_l4183_418318


namespace NUMINAMATH_CALUDE_total_vessels_l4183_418377

theorem total_vessels (x y z w : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (hxy : x < y) (hyz : y < z) (hzw : z < w) :
  let cruise_ships := x
  let cargo_ships := y * x
  let sailboats := y * x + z
  let fishing_boats := (y * x + z) / w
  cruise_ships + cargo_ships + sailboats + fishing_boats = x * (2 * y + 1) + z * (1 + 1 / w) :=
by sorry

end NUMINAMATH_CALUDE_total_vessels_l4183_418377


namespace NUMINAMATH_CALUDE_decryption_theorem_l4183_418394

/-- Represents a character in the Russian alphabet --/
inductive RussianChar : Type
| A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z | AA | AB | AC | AD | AE | AF | AG

/-- Represents an encrypted message --/
def EncryptedMessage := List Char

/-- Represents a decrypted message --/
def DecryptedMessage := List RussianChar

/-- Converts a base-7 number to base-10 --/
def baseSevenToTen (n : Int) : Int :=
  sorry

/-- Applies Caesar cipher shift to a character --/
def applyCaesarShift (c : Char) (shift : Int) : RussianChar :=
  sorry

/-- Decrypts a message using Caesar cipher and base-7 to base-10 conversion --/
def decryptMessage (msg : EncryptedMessage) (shift : Int) : DecryptedMessage :=
  sorry

/-- Checks if a decrypted message is valid Russian text --/
def isValidRussianText (msg : DecryptedMessage) : Prop :=
  sorry

/-- The main theorem: decrypting the messages with shift 22 results in valid Russian text --/
theorem decryption_theorem (messages : List EncryptedMessage) :
  ∀ msg ∈ messages, isValidRussianText (decryptMessage msg 22) :=
  sorry

end NUMINAMATH_CALUDE_decryption_theorem_l4183_418394


namespace NUMINAMATH_CALUDE_most_accurate_reading_l4183_418310

def scale_start : ℝ := 10.25
def scale_end : ℝ := 10.5
def arrow_position : ℝ := 10.3  -- Approximate position based on the problem description

def options : List ℝ := [10.05, 10.15, 10.25, 10.3, 10.6]

theorem most_accurate_reading :
  scale_start < arrow_position ∧ 
  arrow_position < scale_end ∧
  |arrow_position - 10.3| < |arrow_position - ((scale_start + scale_end) / 2)| →
  (options.filter (λ x => x ≥ scale_start ∧ x ≤ scale_end)).argmin (λ x => |x - arrow_position|) = some 10.3 := by
  sorry

end NUMINAMATH_CALUDE_most_accurate_reading_l4183_418310


namespace NUMINAMATH_CALUDE_cost_price_of_cloth_l4183_418381

/-- Represents the cost price of cloth per meter -/
def cost_price_per_meter (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_meters) / total_meters

/-- Theorem: The cost price of one meter of cloth is 118 rupees -/
theorem cost_price_of_cloth (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ)
    (h1 : total_meters = 80)
    (h2 : selling_price = 10000)
    (h3 : profit_per_meter = 7) :
  cost_price_per_meter total_meters selling_price profit_per_meter = 118 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_cloth_l4183_418381


namespace NUMINAMATH_CALUDE_minutes_to_seconds_conversion_seconds_to_minutes_conversion_l4183_418392

-- Define the conversion factor
def seconds_per_minute : ℝ := 60

-- Define the number of minutes
def minutes : ℝ := 8.5

-- Theorem to prove
theorem minutes_to_seconds_conversion :
  minutes * seconds_per_minute = 510 := by
  sorry

-- Verification theorem
theorem seconds_to_minutes_conversion :
  510 / seconds_per_minute = minutes := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_seconds_conversion_seconds_to_minutes_conversion_l4183_418392


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4183_418302

theorem complex_equation_solution (z : ℂ) : (1 - I) * z = 2 * I → z = -1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4183_418302


namespace NUMINAMATH_CALUDE_arrangement_count_equals_factorial_l4183_418386

/-- The number of ways to arrange n distinct objects in n positions -/
def permutations (n : ℕ) : ℕ := n.factorial

/-- The number of boxes in the grid -/
def num_boxes : ℕ := 6

/-- The number of available digits -/
def num_digits : ℕ := 5

/-- The number of ways to arrange digits and an empty space in boxes -/
def arrangement_count : ℕ := permutations num_boxes

theorem arrangement_count_equals_factorial :
  arrangement_count = num_boxes.factorial :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_equals_factorial_l4183_418386


namespace NUMINAMATH_CALUDE_first_discount_percentage_l4183_418387

/-- Proves that given a list price of 70, a final price of 56.16 after two successive discounts,
    where the second discount is 10.857142857142863%, the first discount percentage is 10%. -/
theorem first_discount_percentage
  (list_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : list_price = 70)
  (h2 : final_price = 56.16)
  (h3 : second_discount = 10.857142857142863)
  (h4 : ∃ (first_discount : ℝ),
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100)) :
  ∃ (first_discount : ℝ), first_discount = 10 ∧
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l4183_418387


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l4183_418353

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l4183_418353


namespace NUMINAMATH_CALUDE_methane_moles_required_l4183_418383

-- Define the chemical species involved
structure ChemicalSpecies where
  methane : ℝ
  chlorine : ℝ
  chloromethane : ℝ
  hydrochloric_acid : ℝ

-- Define the reaction conditions
def reaction_conditions (reactants products : ChemicalSpecies) : Prop :=
  reactants.chlorine = 2 ∧
  products.chloromethane = 2 ∧
  products.hydrochloric_acid = 2

-- Define the stoichiometric relationship
def stoichiometric_relationship (reactants products : ChemicalSpecies) : Prop :=
  reactants.methane = products.chloromethane ∧
  reactants.methane = products.hydrochloric_acid

-- Theorem statement
theorem methane_moles_required 
  (reactants products : ChemicalSpecies) 
  (h_conditions : reaction_conditions reactants products) 
  (h_stoichiometry : stoichiometric_relationship reactants products) : 
  reactants.methane = 2 := by
  sorry

end NUMINAMATH_CALUDE_methane_moles_required_l4183_418383


namespace NUMINAMATH_CALUDE_square_sum_equals_four_l4183_418346

theorem square_sum_equals_four (x y : ℝ) (h1 : x + y = -4) (h2 : x = 6 / y) : x^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_four_l4183_418346


namespace NUMINAMATH_CALUDE_line_vector_proof_l4183_418373

def line_vector (t : ℚ) : ℚ × ℚ × ℚ := sorry

theorem line_vector_proof :
  (line_vector (-2) = (2, 6, 16)) ∧
  (line_vector 1 = (0, -1, -2)) ∧
  (line_vector 4 = (-2, -8, -18)) →
  (line_vector 0 = (2/3, 4/3, 4)) ∧
  (line_vector 5 = (-8, -19, -26)) := by sorry

end NUMINAMATH_CALUDE_line_vector_proof_l4183_418373


namespace NUMINAMATH_CALUDE_terminal_side_negative_pi_in_fourth_quadrant_l4183_418349

/-- The terminal side of -π radians lies in the fourth quadrant -/
theorem terminal_side_negative_pi_in_fourth_quadrant :
  let angle : ℝ := -π
  (angle > -2*π ∧ angle ≤ -3*π/2) ∨ (angle > 3*π/2 ∧ angle ≤ 2*π) :=
by sorry

end NUMINAMATH_CALUDE_terminal_side_negative_pi_in_fourth_quadrant_l4183_418349


namespace NUMINAMATH_CALUDE_total_yards_two_days_l4183_418321

-- Define the basic throw distance at 50°F
def base_distance : ℕ := 20

-- Define the temperature effect
def temp_effect (temp : ℕ) : ℕ → ℕ :=
  λ d => if temp = 80 then 2 * d else d

-- Define the wind effect
def wind_effect (wind_speed : ℤ) : ℕ → ℤ :=
  λ d => d + wind_speed * 5 / 10

-- Calculate total distance for a day
def total_distance (temp : ℕ) (wind_speed : ℤ) (throws : ℕ) : ℕ :=
  (wind_effect wind_speed (temp_effect temp base_distance)).toNat * throws

-- Theorem statement
theorem total_yards_two_days :
  total_distance 50 (-1) 20 + total_distance 80 3 30 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_yards_two_days_l4183_418321


namespace NUMINAMATH_CALUDE_f_of_f_zero_l4183_418364

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 + 2 * x - 1

-- State the theorem
theorem f_of_f_zero : f (f 0) = 1 := by sorry

end NUMINAMATH_CALUDE_f_of_f_zero_l4183_418364


namespace NUMINAMATH_CALUDE_factors_of_60_l4183_418367

/-- The number of positive factors of 60 -/
def num_factors_60 : ℕ := sorry

/-- Theorem stating that the number of positive factors of 60 is 12 -/
theorem factors_of_60 : num_factors_60 = 12 := by sorry

end NUMINAMATH_CALUDE_factors_of_60_l4183_418367


namespace NUMINAMATH_CALUDE_walker_speed_l4183_418378

-- Define the track properties
def track_A_width : ℝ := 6
def track_B_width : ℝ := 8
def track_A_time_diff : ℝ := 36
def track_B_time_diff : ℝ := 48

-- Define the theorem
theorem walker_speed (speed : ℝ) : 
  (2 * Real.pi * track_A_width = speed * track_A_time_diff) →
  (2 * Real.pi * track_B_width = speed * track_B_time_diff) →
  speed = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_walker_speed_l4183_418378


namespace NUMINAMATH_CALUDE_square_side_length_l4183_418317

theorem square_side_length (d : ℝ) (h : d = Real.sqrt 8) :
  ∃ s : ℝ, s > 0 ∧ s * Real.sqrt 2 = d ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l4183_418317


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l4183_418396

theorem complex_modulus_equation :
  ∃ (t : ℝ), t > 0 ∧ Complex.abs (9 + t * Complex.I) = 15 ∧ t = 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l4183_418396


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_3_l4183_418374

/-- Two lines in the x-y plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- The condition for two lines to be distinct (not coincident) -/
def distinct (l1 l2 : Line) : Prop := l1.intercept ≠ l2.intercept

theorem parallel_lines_a_equals_3 (a : ℝ) :
  let l1 : Line := { slope := a^2, intercept := 3*a - a^2 }
  let l2 : Line := { slope := 4*a - 3, intercept := 2 }
  parallel l1 l2 ∧ distinct l1 l2 → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_3_l4183_418374


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l4183_418344

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def SimilarTriangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_side_length 
  (P Q R S T U : ℝ × ℝ) 
  (h_similar : SimilarTriangles {P, Q, R} {S, T, U}) 
  (h_PQ : dist P Q = 10) 
  (h_QR : dist Q R = 15) 
  (h_ST : dist S T = 6) : 
  dist T U = 9 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l4183_418344


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l4183_418365

theorem binomial_expansion_coefficient (a b : ℝ) :
  (∃ x, (1 + a*x)^5 = 1 + 10*x + b*x^2 + a^3*x^3 + a^4*x^4 + a^5*x^5) →
  b = 40 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l4183_418365


namespace NUMINAMATH_CALUDE_power_fraction_equality_l4183_418330

theorem power_fraction_equality : (7^14 : ℕ) / (49^6 : ℕ) = 49 := by sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l4183_418330


namespace NUMINAMATH_CALUDE_expansion_coefficient_l4183_418340

/-- The coefficient of x^5 in the expansion of (2x-√x)^8 -/
def coefficient_x5 : ℕ := 112

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem expansion_coefficient :
  coefficient_x5 = (binomial 8 6) * 2^2 :=
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l4183_418340


namespace NUMINAMATH_CALUDE_max_band_members_l4183_418376

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (f : BandFormation) (totalMembers : ℕ) : Prop :=
  totalMembers < 100 ∧
  totalMembers = f.rows * f.membersPerRow + 3 ∧
  totalMembers = (f.rows - 3) * (f.membersPerRow + 2)

/-- Theorem stating the maximum number of band members --/
theorem max_band_members :
  ∃ (m : ℕ) (f : BandFormation),
    isValidFormation f m ∧
    ∀ (n : ℕ) (g : BandFormation), isValidFormation g n → n ≤ m :=
  by sorry

end NUMINAMATH_CALUDE_max_band_members_l4183_418376


namespace NUMINAMATH_CALUDE_dividend_calculation_l4183_418366

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 36)
  (h2 : quotient = 21)
  (h3 : remainder = 4) :
  divisor * quotient + remainder = 760 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l4183_418366


namespace NUMINAMATH_CALUDE_discount_savings_difference_l4183_418313

def shoe_price : ℕ := 50
def discount_a_percent : ℕ := 40
def discount_b_amount : ℕ := 15

def cost_with_discount_a : ℕ := shoe_price + (shoe_price - (shoe_price * discount_a_percent / 100))
def cost_with_discount_b : ℕ := shoe_price + (shoe_price - discount_b_amount)

theorem discount_savings_difference : 
  cost_with_discount_b - cost_with_discount_a = 5 := by
  sorry

end NUMINAMATH_CALUDE_discount_savings_difference_l4183_418313


namespace NUMINAMATH_CALUDE_ellipse_theorem_l4183_418335

/-- Ellipse C with given properties -/
structure Ellipse :=
  (center : ℝ × ℝ)
  (major_axis : ℝ)
  (point_on_ellipse : ℝ × ℝ)
  (h_center : center = (0, 0))
  (h_major_axis : major_axis = 4)
  (h_point : point_on_ellipse = (1, Real.sqrt 3 / 2))

/-- Line with slope 1/2 passing through a point -/
structure Line (P : ℝ × ℝ) :=
  (slope : ℝ)
  (h_slope : slope = 1/2)

/-- Theorem about the ellipse C and intersecting lines -/
theorem ellipse_theorem (C : Ellipse) :
  (∃ (eq : ℝ × ℝ → Prop), ∀ (x y : ℝ), eq (x, y) ↔ x^2/4 + y^2 = 1) ∧
  (∀ (P : ℝ × ℝ), P.2 = 0 → P.1 ∈ Set.Icc (-2 : ℝ) 2 →
    ∀ (l : Line P) (A B : ℝ × ℝ),
      (∃ (t : ℝ), A = (t, (t - P.1)/2) ∧ A.1^2/4 + A.2^2 = 1) →
      (∃ (t : ℝ), B = (t, (t - P.1)/2) ∧ B.1^2/4 + B.2^2 = 1) →
      (A.1 - P.1)^2 + A.2^2 + (B.1 - P.1)^2 + B.2^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l4183_418335


namespace NUMINAMATH_CALUDE_magic_square_sum_divisible_by_three_l4183_418331

/-- Represents a 3x3 magic square -/
def MagicSquare : Type := Fin 3 → Fin 3 → ℕ

/-- The sum of a row, column, or diagonal in a magic square -/
def magic_sum (square : MagicSquare) : ℕ := square 0 0 + square 0 1 + square 0 2

/-- Predicate to check if a square is a valid magic square -/
def is_magic_square (square : MagicSquare) : Prop :=
  (∀ i : Fin 3, square i 0 + square i 1 + square i 2 = magic_sum square) ∧
  (∀ j : Fin 3, square 0 j + square 1 j + square 2 j = magic_sum square) ∧
  (square 0 0 + square 1 1 + square 2 2 = magic_sum square) ∧
  (square 0 2 + square 1 1 + square 2 0 = magic_sum square)

theorem magic_square_sum_divisible_by_three (square : MagicSquare) 
  (h : is_magic_square square) : 
  ∃ k : ℕ, magic_sum square = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_divisible_by_three_l4183_418331


namespace NUMINAMATH_CALUDE_club_members_count_l4183_418371

theorem club_members_count (n : ℕ) (h : n > 2) :
  (2 : ℚ) / ((n : ℚ) - 1) = (1 : ℚ) / 5 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_club_members_count_l4183_418371


namespace NUMINAMATH_CALUDE_exam_candidates_count_l4183_418307

theorem exam_candidates_count :
  ∀ (T : ℕ),
    (T : ℚ) * (49 / 100) = T * (percent_failed_english : ℚ) →
    (T : ℚ) * (36 / 100) = T * (percent_failed_hindi : ℚ) →
    (T : ℚ) * (15 / 100) = T * (percent_failed_both : ℚ) →
    (T : ℚ) * ((51 / 100) - (15 / 100)) = 630 →
    T = 1750 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_count_l4183_418307


namespace NUMINAMATH_CALUDE_inscribed_sphere_in_cone_l4183_418339

theorem inscribed_sphere_in_cone (a b c : ℝ) : 
  let cone_base_radius : ℝ := 20
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := (120 * (Real.sqrt 13 - 10)) / 27
  sphere_radius = a * Real.sqrt c - b →
  a + b + c = 253 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_in_cone_l4183_418339


namespace NUMINAMATH_CALUDE_total_age_calculation_l4183_418343

def family_gathering (K : ℕ) : Prop :=
  let father_age : ℕ := 60
  let mother_age : ℕ := father_age - 2
  let brother_age : ℕ := father_age / 2
  let sister_age : ℕ := 40
  let elder_cousin_age : ℕ := brother_age + 2 * sister_age
  let younger_cousin_age : ℕ := elder_cousin_age / 2 + 3
  let grandmother_age : ℕ := 3 * mother_age - 5
  let T : ℕ := father_age + mother_age + brother_age + sister_age + 
               elder_cousin_age + younger_cousin_age + grandmother_age + K
  T = 525 + K

theorem total_age_calculation (K : ℕ) : family_gathering K :=
  sorry

end NUMINAMATH_CALUDE_total_age_calculation_l4183_418343


namespace NUMINAMATH_CALUDE_problem_statement_l4183_418354

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

noncomputable def g (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) - x)

noncomputable def F (x : ℝ) : ℝ := f x + g x

theorem problem_statement :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, g (-x) = -g x) ∧
  (∃ M m, (∀ x ∈ Set.Icc (-1) 1, F x ≤ M ∧ m ≤ F x) ∧ M + m = 0) ∧
  (Set.Ioi 1 = {a | F (2*a) + F (-1-a) < 0}) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l4183_418354


namespace NUMINAMATH_CALUDE_sum_of_ratios_bound_l4183_418305

theorem sum_of_ratios_bound (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ (3 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_ratios_bound_l4183_418305


namespace NUMINAMATH_CALUDE_initial_girls_count_l4183_418333

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 18) = b) →
  (4 * (b - 36) = g - 18) →
  g = 31 :=
by sorry

end NUMINAMATH_CALUDE_initial_girls_count_l4183_418333


namespace NUMINAMATH_CALUDE_smallest_largest_five_digit_reverse_multiple_of_four_l4183_418329

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a five-digit number -/
def isFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem smallest_largest_five_digit_reverse_multiple_of_four :
  ∀ n : ℕ, isFiveDigit n → (reverseDigits n % 4 = 0) →
    21001 ≤ n ∧ n ≤ 88999 ∧
    (∀ m : ℕ, isFiveDigit m → (reverseDigits m % 4 = 0) →
      (m < 21001 ∨ 88999 < m) → False) :=
sorry

end NUMINAMATH_CALUDE_smallest_largest_five_digit_reverse_multiple_of_four_l4183_418329


namespace NUMINAMATH_CALUDE_point_on_curve_with_perpendicular_tangent_l4183_418380

theorem point_on_curve_with_perpendicular_tangent :
  ∀ x y : ℝ,
  (y = x^4 - x) →                           -- Point P(x, y) is on the curve f(x) = x^4 - x
  (4*x^3 - 1) * 1 + 3 * (-1) = 0 →          -- Tangent line is perpendicular to x + 3y = 0
  (x = 1 ∧ y = 0) :=                        -- P has coordinates (1, 0)
by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_with_perpendicular_tangent_l4183_418380


namespace NUMINAMATH_CALUDE_additional_cost_for_new_requirements_l4183_418352

/-- The additional cost for Farmer Brown to meet his new requirements -/
theorem additional_cost_for_new_requirements
  (initial_bales : ℕ)
  (original_cost_per_bale : ℕ)
  (better_quality_cost_per_bale : ℕ)
  (h1 : initial_bales = 10)
  (h2 : original_cost_per_bale = 15)
  (h3 : better_quality_cost_per_bale = 18) :
  (2 * initial_bales * better_quality_cost_per_bale) - (initial_bales * original_cost_per_bale) = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_additional_cost_for_new_requirements_l4183_418352


namespace NUMINAMATH_CALUDE_restaurant_bill_tax_calculation_l4183_418355

/-- Calculates the tax amount for a restaurant bill given specific conditions. -/
theorem restaurant_bill_tax_calculation
  (cheeseburger_price : ℚ)
  (milkshake_price : ℚ)
  (coke_price : ℚ)
  (fries_price : ℚ)
  (cookie_price : ℚ)
  (toby_initial_amount : ℚ)
  (toby_change : ℚ)
  (h1 : cheeseburger_price = 365/100)
  (h2 : milkshake_price = 2)
  (h3 : coke_price = 1)
  (h4 : fries_price = 4)
  (h5 : cookie_price = 1/2)
  (h6 : toby_initial_amount = 15)
  (h7 : toby_change = 7) :
  let subtotal := 2 * cheeseburger_price + milkshake_price + coke_price + fries_price + 3 * cookie_price
  let toby_spent := toby_initial_amount - toby_change
  let total_paid := 2 * toby_spent
  let tax := total_paid - subtotal
  tax = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_tax_calculation_l4183_418355


namespace NUMINAMATH_CALUDE_initial_bales_count_l4183_418311

theorem initial_bales_count (added_bales current_total : ℕ) 
  (h1 : added_bales = 26)
  (h2 : current_total = 54)
  : current_total - added_bales = 28 := by
  sorry

end NUMINAMATH_CALUDE_initial_bales_count_l4183_418311


namespace NUMINAMATH_CALUDE_greater_number_proof_l4183_418359

theorem greater_number_proof (x y : ℝ) (h1 : 4 * y = 5 * x) (h2 : x + y = 26) : 
  y = 130 / 9 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l4183_418359
