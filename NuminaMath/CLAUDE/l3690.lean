import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_l3690_369096

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) 
  (h1 : a + b + c = 0) (h2 : 4*a - 2*b + c = 0) : 
  ∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ 
  (∀ z : ℝ, a*z^2 + b*z + c = 0 ↔ z = x ∨ z = y) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3690_369096


namespace NUMINAMATH_CALUDE_highest_power_of_two_and_three_l3690_369044

def n : ℤ := 15^4 - 11^4

theorem highest_power_of_two_and_three (n : ℤ) (h : n = 15^4 - 11^4) :
  (∃ m : ℕ, 2^4 * m = n ∧ ¬(∃ k : ℕ, 2^5 * k = n)) ∧
  (∃ m : ℕ, 3^0 * m = n ∧ ¬(∃ k : ℕ, 3^1 * k = n)) :=
sorry

end NUMINAMATH_CALUDE_highest_power_of_two_and_three_l3690_369044


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l3690_369055

/-- Represents the filling rate of pipe A in liters per minute -/
def pipe_A_rate : ℕ := 40

/-- Represents the filling rate of pipe B in liters per minute -/
def pipe_B_rate : ℕ := 30

/-- Represents the draining rate of pipe C in liters per minute -/
def pipe_C_rate : ℕ := 20

/-- Represents the time in minutes it takes to fill the tank -/
def fill_time : ℕ := 48

/-- Represents the capacity of the tank in liters -/
def tank_capacity : ℕ := 780

/-- Theorem stating that given the pipe rates and fill time, the tank capacity is 780 liters -/
theorem tank_capacity_proof :
  pipe_A_rate = 40 →
  pipe_B_rate = 30 →
  pipe_C_rate = 20 →
  fill_time = 48 →
  tank_capacity = 780 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l3690_369055


namespace NUMINAMATH_CALUDE_next_joint_work_day_is_360_l3690_369060

/-- Represents the work schedule of a staff member -/
structure WorkSchedule where
  cycle : Nat
  deriving Repr

/-- Represents the community center with its staff members -/
structure CommunityCenter where
  alan : WorkSchedule
  berta : WorkSchedule
  carlos : WorkSchedule
  dora : WorkSchedule

/-- Calculates the next day when all staff members work together -/
def nextJointWorkDay (center : CommunityCenter) : Nat :=
  Nat.lcm center.alan.cycle (Nat.lcm center.berta.cycle (Nat.lcm center.carlos.cycle center.dora.cycle))

/-- The main theorem: proving that the next joint work day is 360 days from today -/
theorem next_joint_work_day_is_360 (center : CommunityCenter) 
  (h1 : center.alan.cycle = 5)
  (h2 : center.berta.cycle = 6)
  (h3 : center.carlos.cycle = 8)
  (h4 : center.dora.cycle = 9) :
  nextJointWorkDay center = 360 := by
  sorry

end NUMINAMATH_CALUDE_next_joint_work_day_is_360_l3690_369060


namespace NUMINAMATH_CALUDE_min_female_participants_l3690_369098

/-- Proves the minimum number of female students participating in community work -/
theorem min_female_participants (male_students female_students : ℕ) 
  (total_participants : ℕ) (h1 : male_students = 22) (h2 : female_students = 18) 
  (h3 : total_participants = ((male_students + female_students) * 6) / 10) : 
  ∃ (female_participants : ℕ), 
    female_participants ≥ 2 ∧ 
    female_participants ≤ female_students ∧
    female_participants + male_students ≥ total_participants :=
by sorry

end NUMINAMATH_CALUDE_min_female_participants_l3690_369098


namespace NUMINAMATH_CALUDE_weaving_problem_l3690_369021

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

theorem weaving_problem (a₁ d : ℚ) (n : ℕ) 
  (h₁ : a₁ = 5)
  (h₂ : n = 30)
  (h₃ : sum_arithmetic_sequence a₁ d n = 390) :
  d = 16/29 := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l3690_369021


namespace NUMINAMATH_CALUDE_inequality_proof_l3690_369038

theorem inequality_proof (a b : ℝ) (h1 : b > a) (h2 : a > 0) : 2 * a + b / 2 ≥ 2 * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3690_369038


namespace NUMINAMATH_CALUDE_stating_work_completion_time_l3690_369064

/-- The time it takes for a man and his son to complete a piece of work together -/
def combined_time : ℝ := 6

/-- The time it takes for the son to complete the work alone -/
def son_time : ℝ := 10

/-- The time it takes for the man to complete the work alone -/
def man_time : ℝ := 15

/-- 
Theorem stating that if a man and his son can complete a piece of work together in 6 days, 
and the son can complete the work alone in 10 days, then the man can complete the work 
alone in 15 days.
-/
theorem work_completion_time : 
  (1 / combined_time) = (1 / man_time) + (1 / son_time) :=
sorry

end NUMINAMATH_CALUDE_stating_work_completion_time_l3690_369064


namespace NUMINAMATH_CALUDE_isabel_song_count_l3690_369090

/-- The number of songs Isabel bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Theorem stating that Isabel bought 72 songs -/
theorem isabel_song_count :
  total_songs 4 5 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_isabel_song_count_l3690_369090


namespace NUMINAMATH_CALUDE_remainder_problem_l3690_369034

theorem remainder_problem (N : ℕ) : 
  (∃ r : ℕ, N = 44 * 432 + r ∧ r < 44) → 
  (∃ q : ℕ, N = 30 * q + 18) → 
  N % 44 = 18 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l3690_369034


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3690_369068

open Complex

theorem pure_imaginary_condition (θ : ℝ) :
  let Z : ℂ := 1 / (sin θ + cos θ * I) - (1 : ℂ) / 2
  (∃ y : ℝ, Z = y * I) →
  (∃ k : ℤ, θ = π / 6 + 2 * k * π ∨ θ = 5 * π / 6 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3690_369068


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_80_l3690_369006

theorem thirty_percent_less_than_80 : ∃ x : ℝ, (80 - 0.3 * 80) = x + 0.25 * x ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_80_l3690_369006


namespace NUMINAMATH_CALUDE_max_ab_value_l3690_369065

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let f := fun x : ℝ => 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| → |h| < ε → f (1 + h) ≤ f 1) →
  (∀ c : ℝ, a * b ≤ c → c ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l3690_369065


namespace NUMINAMATH_CALUDE_negation_of_existence_l3690_369050

theorem negation_of_existence (l : ℝ) : 
  (¬ ∃ x : ℝ, x + l ≥ 0) ↔ (∀ x : ℝ, x + l < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l3690_369050


namespace NUMINAMATH_CALUDE_carols_invitations_l3690_369078

/-- Given that Carol bought packages of invitations, prove that the number of friends she can invite is equal to the product of invitations per package and the number of packages. -/
theorem carols_invitations (invitations_per_package : ℕ) (num_packages : ℕ) :
  invitations_per_package = 9 →
  num_packages = 5 →
  invitations_per_package * num_packages = 45 := by
  sorry

#check carols_invitations

end NUMINAMATH_CALUDE_carols_invitations_l3690_369078


namespace NUMINAMATH_CALUDE_successive_numbers_product_l3690_369051

theorem successive_numbers_product (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 4160 → n = 64 := by
  sorry

end NUMINAMATH_CALUDE_successive_numbers_product_l3690_369051


namespace NUMINAMATH_CALUDE_g_function_equality_l3690_369000

/-- Given that 4x^4 + 5x^2 - 2x + 7 + g(x) = 6x^3 - 4x^2 + 8x - 1,
    prove that g(x) = -4x^4 + 6x^3 - 9x^2 + 10x - 8 -/
theorem g_function_equality (x : ℝ) (g : ℝ → ℝ)
    (h : ∀ x, 4 * x^4 + 5 * x^2 - 2 * x + 7 + g x = 6 * x^3 - 4 * x^2 + 8 * x - 1) :
  g x = -4 * x^4 + 6 * x^3 - 9 * x^2 + 10 * x - 8 := by
  sorry

end NUMINAMATH_CALUDE_g_function_equality_l3690_369000


namespace NUMINAMATH_CALUDE_min_value_of_f_range_of_t_l3690_369043

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 4|

-- Theorem for the minimum value of f
theorem min_value_of_f : ∀ x : ℝ, f x ≥ 6 ∧ ∃ x₀ : ℝ, f x₀ = 6 := by sorry

-- Define the set A
def A (t : ℝ) : Set ℝ := {x | f x ≤ t^2 - t}

-- Define the set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}

-- Theorem for the range of t
theorem range_of_t : ∀ t : ℝ, (A t ∩ B).Nonempty ↔ t ≤ -2 ∨ t ≥ 3 := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_range_of_t_l3690_369043


namespace NUMINAMATH_CALUDE_line_through_point_l3690_369013

/-- 
Given a line equation 2 - kx = -4y that passes through the point (3, -2),
prove that k = -2.
-/
theorem line_through_point (k : ℝ) : 
  (2 - k * 3 = -4 * (-2)) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l3690_369013


namespace NUMINAMATH_CALUDE_tens_digit_of_6_to_19_l3690_369030

-- Define a function to calculate the tens digit
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

-- State the theorem
theorem tens_digit_of_6_to_19 :
  tens_digit (6^19) = 1 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_to_19_l3690_369030


namespace NUMINAMATH_CALUDE_simplify_fraction_l3690_369045

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 / (x^2 - 1)) / (1 / (x - 1)) = 2 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3690_369045


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l3690_369087

-- Define G_n
def G (n : ℕ) : ℕ := 3 * 2^(2^n) + 4

-- Theorem statement
theorem units_digit_G_1000 : G 1000 % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l3690_369087


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l3690_369095

theorem polar_to_rectangular (x y ρ : ℝ) :
  ρ = 2 → x^2 + y^2 = ρ^2 → x^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l3690_369095


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3690_369016

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Theorem stating that the given asymptote equation is correct for the hyperbola
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola x y → (∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ hyperbola x' y' ∧ asymptote x' y') :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3690_369016


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3690_369041

-- Problem 1
theorem problem_1 : (-3/7) + 1/5 + 2/7 + (-6/5) = -8/7 := by sorry

-- Problem 2
theorem problem_2 : -(-1) + 3^2 / (1-4) * 2 = -5 := by sorry

-- Problem 3
theorem problem_3 : (-1/6)^2 / ((1/2 - 1/3)^2) / |(-6)|^2 = 1/36 := by sorry

-- Problem 4
theorem problem_4 : (-1)^1000 - 2.45 * 8 + 2.55 * (-8) = -39 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3690_369041


namespace NUMINAMATH_CALUDE_gcd_18_30_l3690_369094

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l3690_369094


namespace NUMINAMATH_CALUDE_our_system_is_linear_l3690_369071

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  toFun : ℝ → ℝ → Prop := λ x y => a * x + b * y = c

/-- A system of two equations -/
structure SystemOfTwoEquations where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system we want to prove is linear -/
def ourSystem : SystemOfTwoEquations where
  eq1 := { a := 1, b := 1, c := 2 }
  eq2 := { a := 1, b := -1, c := 4 }

/-- A predicate to check if a system is linear -/
def isLinearSystem (s : SystemOfTwoEquations) : Prop :=
  s.eq1.a ≠ 0 ∨ s.eq1.b ≠ 0 ∧
  s.eq2.a ≠ 0 ∨ s.eq2.b ≠ 0

theorem our_system_is_linear : isLinearSystem ourSystem := by
  sorry

end NUMINAMATH_CALUDE_our_system_is_linear_l3690_369071


namespace NUMINAMATH_CALUDE_running_track_l3690_369069

/-- Given two concentric circles with radii r₁ and r₂, where the difference in their circumferences is 24π feet, prove the width of the track and the enclosed area. -/
theorem running_track (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 24 * Real.pi) :
  r₁ - r₂ = 12 ∧ Real.pi * (r₁^2 - r₂^2) = Real.pi * (24 * r₂ + 144) :=
by sorry

end NUMINAMATH_CALUDE_running_track_l3690_369069


namespace NUMINAMATH_CALUDE_equation_solution_l3690_369019

theorem equation_solution : 
  let f : ℝ → ℝ := fun y => y^2 - 3*y - 10 + (y + 2)*(y + 6)
  (f (-1/2) = 0 ∧ f (-2) = 0) ∧ 
  ∀ y : ℝ, f y = 0 → (y = -1/2 ∨ y = -2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3690_369019


namespace NUMINAMATH_CALUDE_prize_probability_after_addition_l3690_369082

/-- Given a box with prizes, this function calculates the probability of pulling a prize -/
def prizeProbability (favorable : ℕ) (unfavorable : ℕ) : ℚ :=
  (favorable : ℚ) / (favorable + unfavorable : ℚ)

theorem prize_probability_after_addition (initial_favorable : ℕ) (initial_unfavorable : ℕ) 
  (h_initial_odds : initial_favorable = 5 ∧ initial_unfavorable = 6) 
  (added_prizes : ℕ) (h_added_prizes : added_prizes = 2) :
  prizeProbability (initial_favorable + added_prizes) initial_unfavorable = 7 / 13 := by
  sorry

#check prize_probability_after_addition

end NUMINAMATH_CALUDE_prize_probability_after_addition_l3690_369082


namespace NUMINAMATH_CALUDE_ellipse_param_sum_l3690_369011

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to the foci -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  dist_sum : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, compute its parameters -/
def computeEllipseParams (e : Ellipse) : EllipseParams :=
  sorry

/-- The main theorem: for the given ellipse, the sum of its parameters is 18 -/
theorem ellipse_param_sum :
  let e := Ellipse.mk (4, 2) (10, 2) 10
  let params := computeEllipseParams e
  params.h + params.k + params.a + params.b = 18 :=
sorry

end NUMINAMATH_CALUDE_ellipse_param_sum_l3690_369011


namespace NUMINAMATH_CALUDE_lcm_of_36_and_12_l3690_369093

theorem lcm_of_36_and_12 (a b : ℕ+) (h1 : a = 36) (h2 : b = 12) (h3 : Nat.gcd a b = 8) :
  Nat.lcm a b = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_12_l3690_369093


namespace NUMINAMATH_CALUDE_ticket_sales_revenue_ticket_sales_problem_l3690_369089

theorem ticket_sales_revenue 
  (student_price : ℕ) 
  (general_price : ℕ) 
  (total_tickets : ℕ) 
  (general_tickets : ℕ) : ℕ :=
  let student_tickets := total_tickets - general_tickets
  let student_revenue := student_tickets * student_price
  let general_revenue := general_tickets * general_price
  student_revenue + general_revenue

theorem ticket_sales_problem : 
  ticket_sales_revenue 4 6 525 388 = 2876 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_revenue_ticket_sales_problem_l3690_369089


namespace NUMINAMATH_CALUDE_m_range_correct_l3690_369005

/-- Statement p: For all x in ℝ, x^2 + mx + m/2 + 2 ≥ 0 always holds true -/
def statement_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m*x + m/2 + 2 ≥ 0

/-- Statement q: The distance from the focus of the parabola y^2 = 2mx (where m > 0) to its directrix is greater than 1 -/
def statement_q (m : ℝ) : Prop :=
  m > 0 ∧ m/2 > 1

/-- The range of m that satisfies all conditions -/
def m_range : Set ℝ :=
  {m : ℝ | m > 4}

theorem m_range_correct :
  ∀ m : ℝ, (statement_p m ∨ statement_q m) ∧ ¬(statement_p m ∧ statement_q m) ↔ m ∈ m_range := by
  sorry

end NUMINAMATH_CALUDE_m_range_correct_l3690_369005


namespace NUMINAMATH_CALUDE_initial_cargo_calculation_l3690_369053

theorem initial_cargo_calculation (cargo_loaded : ℕ) (total_cargo : ℕ) 
  (h1 : cargo_loaded = 8723)
  (h2 : total_cargo = 14696) :
  total_cargo - cargo_loaded = 5973 := by
  sorry

end NUMINAMATH_CALUDE_initial_cargo_calculation_l3690_369053


namespace NUMINAMATH_CALUDE_max_average_annual_profit_l3690_369085

/-- Represents the total profit (in million yuan) for operating 4 buses for x years -/
def total_profit (x : ℕ+) : ℚ :=
  16 * (-2 * x^2 + 23 * x - 50)

/-- Represents the average annual profit (in million yuan) for operating 4 buses for x years -/
def average_annual_profit (x : ℕ+) : ℚ :=
  total_profit x / x

/-- Theorem stating that the average annual profit is maximized when x = 5 -/
theorem max_average_annual_profit :
  ∀ x : ℕ+, average_annual_profit 5 ≥ average_annual_profit x :=
sorry

end NUMINAMATH_CALUDE_max_average_annual_profit_l3690_369085


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3690_369036

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  (1/x) + (4/y) + (9/z) ≥ 12 ∧ 
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 
    a + b + c = 3 ∧ (1/a) + (4/b) + (9/c) = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3690_369036


namespace NUMINAMATH_CALUDE_math_city_intersections_l3690_369023

/-- Represents a city with a given number of streets -/
structure City where
  num_streets : ℕ
  no_parallel : Bool
  no_triple_intersections : Bool

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  if c.num_streets ≤ 1 then 0
  else (c.num_streets - 1) * (c.num_streets - 2) / 2

/-- Theorem stating that a city with 12 streets, no parallel streets, 
    and no triple intersections has 66 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 12 → c.no_parallel = true → 
  c.no_triple_intersections = true → num_intersections c = 66 :=
by
  sorry


end NUMINAMATH_CALUDE_math_city_intersections_l3690_369023


namespace NUMINAMATH_CALUDE_intersection_condition_l3690_369001

def A (a : ℝ) : Set ℝ := {3, Real.sqrt a}
def B (a : ℝ) : Set ℝ := {1, a}

theorem intersection_condition (a : ℝ) : A a ∩ B a = {a} → a = 0 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3690_369001


namespace NUMINAMATH_CALUDE_largest_divisor_of_10000_l3690_369029

theorem largest_divisor_of_10000 :
  ∀ n : ℕ, n ∣ 10000 ∧ ¬(n ∣ 9999) → n ≤ 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_10000_l3690_369029


namespace NUMINAMATH_CALUDE_base_of_second_term_l3690_369010

theorem base_of_second_term (e : ℕ) (base : ℚ) 
  (h1 : e = 35)
  (h2 : (1/5 : ℚ)^e * base^18 = 1/(2*(10^35))) :
  base = 1/4 := by sorry

end NUMINAMATH_CALUDE_base_of_second_term_l3690_369010


namespace NUMINAMATH_CALUDE_lcm_12_18_l3690_369083

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l3690_369083


namespace NUMINAMATH_CALUDE_tims_income_percentage_l3690_369070

theorem tims_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = tim * 1.6)
  (h2 : mary = juan * 0.6400000000000001) : 
  tim = juan * 0.4 := by
  sorry

end NUMINAMATH_CALUDE_tims_income_percentage_l3690_369070


namespace NUMINAMATH_CALUDE_polynomial_descending_order_x_l3690_369003

theorem polynomial_descending_order_x (x y : ℝ) :
  3 * x * y^2 - 2 * x^2 * y - x^3 * y^3 - 4 =
  -x^3 * y^3 - 2 * x^2 * y + 3 * x * y^2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_descending_order_x_l3690_369003


namespace NUMINAMATH_CALUDE_rogers_coin_piles_l3690_369015

theorem rogers_coin_piles (num_quarter_piles num_dime_piles coins_per_pile total_coins : ℕ) :
  num_quarter_piles = num_dime_piles →
  coins_per_pile = 7 →
  total_coins = 42 →
  num_quarter_piles * coins_per_pile + num_dime_piles * coins_per_pile = total_coins →
  num_quarter_piles = 3 := by
  sorry

end NUMINAMATH_CALUDE_rogers_coin_piles_l3690_369015


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3690_369075

theorem necessary_but_not_sufficient (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  ¬(∀ a b c : ℝ, a > b → a * c^2 > b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3690_369075


namespace NUMINAMATH_CALUDE_largest_n_for_negative_quadratic_five_satisfies_condition_six_does_not_satisfy_l3690_369040

theorem largest_n_for_negative_quadratic : 
  ∀ n : ℤ, n^2 - 9*n + 18 < 0 → n ≤ 5 :=
by sorry

theorem five_satisfies_condition : 
  (5 : ℤ)^2 - 9*5 + 18 < 0 :=
by sorry

theorem six_does_not_satisfy : 
  ¬((6 : ℤ)^2 - 9*6 + 18 < 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_negative_quadratic_five_satisfies_condition_six_does_not_satisfy_l3690_369040


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3690_369052

theorem arithmetic_sequence_sum (a₁ a_n d : ℚ) (n : ℕ) (h1 : a₁ = 2/7) (h2 : a_n = 20/7) (h3 : d = 2/7) (h4 : n = 10) :
  (n : ℚ) / 2 * (a₁ + a_n) = 110/7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3690_369052


namespace NUMINAMATH_CALUDE_average_increase_l3690_369032

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ := 1.6 * x + 2

-- Theorem statement
theorem average_increase (x : ℝ) : 
  linear_regression (x + 1) - linear_regression x = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_l3690_369032


namespace NUMINAMATH_CALUDE_no_entangled_numbers_l3690_369012

/-- A two-digit positive integer is entangled if it equals twice the sum of its nonzero tens digit and the cube of its units digit -/
def is_entangled (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ (a b : ℕ), a > 0 ∧ b < 10 ∧ n = 10 * a + b ∧ n = 2 * (a + b^3)

/-- There are no entangled two-digit positive integers -/
theorem no_entangled_numbers : ¬∃ (n : ℕ), is_entangled n := by
  sorry

end NUMINAMATH_CALUDE_no_entangled_numbers_l3690_369012


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_factorization_of_2025_l3690_369054

/-- The largest integer k such that 2025^k divides (2025!)^2 is 505. -/
theorem largest_power_dividing_factorial_squared : ∃ k : ℕ, k = 505 ∧ 
  (∀ m : ℕ, (2025 ^ m : ℕ) ∣ (Nat.factorial 2025)^2 → m ≤ k) ∧
  (2025 ^ k : ℕ) ∣ (Nat.factorial 2025)^2 := by
  sorry

/-- 2025 is equal to 3^4 * 5^2 -/
theorem factorization_of_2025 : 2025 = 3^4 * 5^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_factorization_of_2025_l3690_369054


namespace NUMINAMATH_CALUDE_function_equality_l3690_369086

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x => 2 * (x - 1)^2 + 1

-- State the theorem
theorem function_equality (x : ℝ) (h : x ≥ 1) : 
  f (1 + Real.sqrt x) = 2 * x + 1 ∧ f x = 2 * x^2 - 4 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l3690_369086


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3690_369002

/-- Given a boat that travels 15 km/hr along a stream and 5 km/hr against the same stream,
    its speed in still water is 10 km/hr. -/
theorem boat_speed_in_still_water
  (along_stream : ℝ)
  (against_stream : ℝ)
  (h_along : along_stream = 15)
  (h_against : against_stream = 5) :
  (along_stream + against_stream) / 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3690_369002


namespace NUMINAMATH_CALUDE_rowing_time_ratio_l3690_369062

/-- Given Ethan's rowing time and the total rowing time for both Ethan and Frank,
    prove that the ratio of Frank's rowing time to Ethan's rowing time is 2:1. -/
theorem rowing_time_ratio 
  (ethan_time : ℕ) 
  (total_time : ℕ) 
  (h1 : ethan_time = 25)
  (h2 : total_time = 75) :
  (total_time - ethan_time) / ethan_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_rowing_time_ratio_l3690_369062


namespace NUMINAMATH_CALUDE_julia_tag_kids_l3690_369017

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 7

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 13

/-- The total number of kids Julia played tag with -/
def total_kids : ℕ := monday_kids + tuesday_kids

theorem julia_tag_kids : total_kids = 20 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_kids_l3690_369017


namespace NUMINAMATH_CALUDE_cart_distance_theorem_l3690_369024

def cart_distance (initial_distance : ℕ) (first_increment : ℕ) (second_increment : ℕ) (total_time : ℕ) : ℕ :=
  let first_section := (total_time / 2) * (2 * initial_distance + (total_time / 2 - 1) * first_increment) / 2
  let final_first_speed := initial_distance + (total_time / 2 - 1) * first_increment
  let second_section := (total_time / 2) * (2 * final_first_speed + (total_time / 2 - 1) * second_increment) / 2
  first_section + second_section

theorem cart_distance_theorem :
  cart_distance 8 10 6 30 = 4020 := by
  sorry

end NUMINAMATH_CALUDE_cart_distance_theorem_l3690_369024


namespace NUMINAMATH_CALUDE_burger_orders_l3690_369066

theorem burger_orders (total : ℕ) (burger_ratio : ℕ) : 
  total = 45 → burger_ratio = 2 → 
  ∃ (hotdog : ℕ), 
    hotdog + burger_ratio * hotdog = total ∧
    burger_ratio * hotdog = 30 := by
  sorry

end NUMINAMATH_CALUDE_burger_orders_l3690_369066


namespace NUMINAMATH_CALUDE_katie_pastries_left_l3690_369031

/-- Represents the number of pastries Katie had left after the bake sale -/
def pastries_left (cupcakes cookies sold : ℕ) : ℕ :=
  cupcakes + cookies - sold

/-- Proves that Katie had 8 pastries left after the bake sale -/
theorem katie_pastries_left : pastries_left 7 5 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_katie_pastries_left_l3690_369031


namespace NUMINAMATH_CALUDE_alloy_combination_theorem_l3690_369008

/-- Represents the composition of an alloy --/
structure AlloyComposition where
  copper : ℝ
  zinc : ℝ

/-- The first alloy composition --/
def firstAlloy : AlloyComposition :=
  { copper := 2, zinc := 1 }

/-- The second alloy composition --/
def secondAlloy : AlloyComposition :=
  { copper := 1, zinc := 5 }

/-- Combines two alloys in a given ratio --/
def combineAlloys (a1 a2 : AlloyComposition) (r1 r2 : ℝ) : AlloyComposition :=
  { copper := r1 * a1.copper + r2 * a2.copper
  , zinc := r1 * a1.zinc + r2 * a2.zinc }

/-- The theorem to be proved --/
theorem alloy_combination_theorem :
  let combinedAlloy := combineAlloys firstAlloy secondAlloy 1 2
  combinedAlloy.zinc = 2 * combinedAlloy.copper := by
  sorry

end NUMINAMATH_CALUDE_alloy_combination_theorem_l3690_369008


namespace NUMINAMATH_CALUDE_modified_equation_product_l3690_369026

theorem modified_equation_product (a b x y : ℝ) (m n p : ℤ) : 
  (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) → 
  ((a^m*x - a^n)*(a^p*y - a^3) = a^5*b^5) → 
  m * n * p = 0 := by
sorry

end NUMINAMATH_CALUDE_modified_equation_product_l3690_369026


namespace NUMINAMATH_CALUDE_farm_animals_l3690_369049

theorem farm_animals (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 4 * initial_cows →
  (initial_horses - 15) / (initial_cows + 15) = 7 / 3 →
  initial_horses - 15 - (initial_cows + 15) = 60 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l3690_369049


namespace NUMINAMATH_CALUDE_min_value_fraction_l3690_369048

theorem min_value_fraction (x : ℝ) (h : x > 6) :
  x^2 / (x - 6) ≥ 24 ∧ (x^2 / (x - 6) = 24 ↔ x = 12) := by
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3690_369048


namespace NUMINAMATH_CALUDE_can_display_rows_l3690_369056

/-- Represents a display of cans arranged in rows. -/
structure CanDisplay where
  firstRowCans : ℕ  -- Number of cans in the first row
  rowIncrement : ℕ  -- Increment in number of cans for each subsequent row
  totalCans : ℕ     -- Total number of cans in the display

/-- Calculates the number of rows in a can display. -/
def numberOfRows (display : CanDisplay) : ℕ :=
  sorry

/-- Theorem stating that a display with 2 cans in the first row,
    incrementing by 3 cans each row, and totaling 120 cans has 9 rows. -/
theorem can_display_rows :
  let display : CanDisplay := {
    firstRowCans := 2,
    rowIncrement := 3,
    totalCans := 120
  }
  numberOfRows display = 9 := by sorry

end NUMINAMATH_CALUDE_can_display_rows_l3690_369056


namespace NUMINAMATH_CALUDE_remainder_17_pow_2047_mod_23_l3690_369028

theorem remainder_17_pow_2047_mod_23 :
  (17 : ℤ) ^ 2047 % 23 = 11 := by sorry

end NUMINAMATH_CALUDE_remainder_17_pow_2047_mod_23_l3690_369028


namespace NUMINAMATH_CALUDE_change_calculation_l3690_369072

def laptop_price : ℝ := 600
def smartphone_price : ℝ := 400
def tablet_price : ℝ := 250
def headphone_price : ℝ := 100
def discount_rate : ℝ := 0.1
def tax_rate : ℝ := 0.05
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def num_tablets : ℕ := 3
def num_headphones : ℕ := 5
def initial_amount : ℝ := 5000

theorem change_calculation : 
  let total_before_discount := 
    num_laptops * laptop_price + 
    num_smartphones * smartphone_price + 
    num_tablets * tablet_price + 
    num_headphones * headphone_price
  let discount := 
    discount_rate * (num_laptops * laptop_price + num_tablets * tablet_price)
  let total_after_discount := total_before_discount - discount
  let tax := tax_rate * total_after_discount
  let final_price := total_after_discount + tax
  initial_amount - final_price = 952.25 := by sorry

end NUMINAMATH_CALUDE_change_calculation_l3690_369072


namespace NUMINAMATH_CALUDE_max_b_value_l3690_369047

/-- The volume of the box -/
def volume : ℕ := 360

/-- Theorem: Given a box with volume 360 cubic units and dimensions a, b, and c,
    where a, b, and c are integers satisfying 1 < c < b < a,
    the maximum possible value of b is 12. -/
theorem max_b_value (a b c : ℕ) 
  (h_volume : a * b * c = volume)
  (h_order : 1 < c ∧ c < b ∧ b < a) :
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = volume ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ b' = 12 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l3690_369047


namespace NUMINAMATH_CALUDE_mairead_exercise_ratio_l3690_369091

/-- Proves the ratio of miles walked to miles jogged for Mairead's exercise routine -/
theorem mairead_exercise_ratio :
  let miles_ran : ℝ := 40
  let miles_walked_fraction : ℝ := 3 / 5 * miles_ran
  let total_distance : ℝ := 184
  let miles_walked_multiple : ℝ := total_distance - miles_ran - miles_walked_fraction
  let total_miles_walked : ℝ := miles_walked_fraction + miles_walked_multiple
  total_miles_walked / miles_ran = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_mairead_exercise_ratio_l3690_369091


namespace NUMINAMATH_CALUDE_range_of_a_l3690_369046

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → a < x + 1/x) → a < 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3690_369046


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l3690_369099

def total_candies : ℕ := 2 * 18
def weekday_candies_per_week : ℕ := 2 * 5
def total_weeks : ℕ := 3
def remaining_days_per_week : ℕ := 2

theorem bobby_candy_consumption :
  (total_candies - weekday_candies_per_week * total_weeks) / (remaining_days_per_week * total_weeks) = 1 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l3690_369099


namespace NUMINAMATH_CALUDE_log_sequence_l3690_369076

theorem log_sequence (a b c : ℝ) (ha : a = Real.log 3 / Real.log 4)
    (hb : b = Real.log 6 / Real.log 4) (hc : c = Real.log 12 / Real.log 4) :
  (b - a = c - b) ∧ ¬(b / a = c / b) := by
  sorry

end NUMINAMATH_CALUDE_log_sequence_l3690_369076


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_even_reverse_l3690_369092

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ Even (reverse_digits n)

theorem smallest_two_digit_prime_with_even_reverse : 
  satisfies_condition 23 ∧ ∀ n : ℕ, satisfies_condition n → 23 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_even_reverse_l3690_369092


namespace NUMINAMATH_CALUDE_marts_income_percentage_l3690_369042

/-- Given that Tim's income is 60 percent less than Juan's income
    and Mart's income is 64 percent of Juan's income,
    prove that Mart's income is 60 percent more than Tim's income. -/
theorem marts_income_percentage (juan tim mart : ℝ) 
  (h1 : tim = juan - 0.60 * juan)
  (h2 : mart = 0.64 * juan) :
  mart = tim + 0.60 * tim := by
  sorry

end NUMINAMATH_CALUDE_marts_income_percentage_l3690_369042


namespace NUMINAMATH_CALUDE_special_cone_volume_l3690_369025

/-- A cone with coinciding centers of inscribed and circumscribed spheres -/
structure SpecialCone where
  /-- The radius of the inscribed sphere -/
  inscribed_radius : ℝ
  /-- The centers of inscribed and circumscribed spheres coincide -/
  centers_coincide : Bool

/-- The volume of a SpecialCone -/
def volume (c : SpecialCone) : ℝ :=
  sorry

/-- Theorem: The volume of a SpecialCone with inscribed radius 1 is 3π -/
theorem special_cone_volume (c : SpecialCone) 
  (h1 : c.inscribed_radius = 1) 
  (h2 : c.centers_coincide = true) : 
  volume c = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_special_cone_volume_l3690_369025


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3690_369088

theorem average_speed_calculation (speed1 speed2 : ℝ) (time : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 30)
  (h3 : time = 2) :
  (speed1 + speed2) / time = 25 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3690_369088


namespace NUMINAMATH_CALUDE_pizza_order_theorem_l3690_369022

theorem pizza_order_theorem : 
  (1 : ℚ) / 2 + (1 : ℚ) / 3 + (1 : ℚ) / 6 = 1 := by sorry

end NUMINAMATH_CALUDE_pizza_order_theorem_l3690_369022


namespace NUMINAMATH_CALUDE_tantrix_impossibility_l3690_369074

/-- Represents a tile in the Tantrix Solitaire game -/
structure Tile where
  blue_lines : Nat
  red_lines : Nat

/-- Represents the game board -/
structure Board where
  tiles : List Tile
  blue_loop : Bool
  no_gaps : Bool
  red_intersections : Nat

/-- Checks if a board configuration is valid -/
def is_valid_board (b : Board) : Prop :=
  b.tiles.length = 13 ∧ b.blue_loop ∧ b.no_gaps ∧ b.red_intersections = 3

/-- Theorem stating the impossibility of arranging 13 tiles to form a valid board -/
theorem tantrix_impossibility : ¬ ∃ (b : Board), is_valid_board b := by
  sorry

end NUMINAMATH_CALUDE_tantrix_impossibility_l3690_369074


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3690_369057

def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}

def B : Set ℝ := {x | Real.exp (x * Real.log 3) > 9}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x : ℝ | x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3690_369057


namespace NUMINAMATH_CALUDE_tank_capacity_tank_capacity_proof_l3690_369073

/-- Given a tank where adding 130 gallons when it's 1/6 full makes it 3/5 full,
    prove that the tank's total capacity is 300 gallons. -/
theorem tank_capacity : ℝ → Prop :=
  fun capacity =>
    (capacity / 6 + 130 = 3 * capacity / 5) → capacity = 300

-- The proof is omitted
theorem tank_capacity_proof : ∃ capacity, tank_capacity capacity :=
  sorry

end NUMINAMATH_CALUDE_tank_capacity_tank_capacity_proof_l3690_369073


namespace NUMINAMATH_CALUDE_final_result_depends_on_blue_l3690_369037

/-- Represents the color of a sprite -/
inductive SpriteColor
| Red
| Blue

/-- Represents the state of the game -/
structure GameState where
  red : ℕ  -- number of red sprites
  blue : ℕ  -- number of blue sprites

/-- Represents the result of the game -/
def GameResult := SpriteColor

/-- The game rules for sprite collision -/
def collide (c1 c2 : SpriteColor) : SpriteColor :=
  match c1, c2 with
  | SpriteColor.Red, SpriteColor.Red => SpriteColor.Red
  | SpriteColor.Blue, SpriteColor.Blue => SpriteColor.Red
  | _, _ => SpriteColor.Blue

/-- The final result of the game -/
def finalResult (initial : GameState) : GameResult :=
  if initial.blue % 2 = 0 then SpriteColor.Red else SpriteColor.Blue

/-- The main theorem: the final result depends only on the initial number of blue sprites -/
theorem final_result_depends_on_blue (m n : ℕ) :
  finalResult { red := m, blue := n } = 
    if n % 2 = 0 then SpriteColor.Red else SpriteColor.Blue :=
by sorry

end NUMINAMATH_CALUDE_final_result_depends_on_blue_l3690_369037


namespace NUMINAMATH_CALUDE_odd_function_property_l3690_369059

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_prop : ∀ x, f (1 + x) = f (-x)) 
  (h_value : f (-1/3) = 1/3) : 
  f (5/3) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l3690_369059


namespace NUMINAMATH_CALUDE_helen_gas_usage_l3690_369033

-- Define the number of months with 2 cuts and 4 cuts
def months_with_two_cuts : ℕ := 4
def months_with_four_cuts : ℕ := 4

-- Define the number of cuts per month for each category
def cuts_per_month_low : ℕ := 2
def cuts_per_month_high : ℕ := 4

-- Define the gas usage
def gas_per_fourth_cut : ℕ := 2
def cuts_per_gas_usage : ℕ := 4

-- Theorem statement
theorem helen_gas_usage :
  let total_cuts := months_with_two_cuts * cuts_per_month_low + months_with_four_cuts * cuts_per_month_high
  let gas_fill_ups := total_cuts / cuts_per_gas_usage
  gas_fill_ups * gas_per_fourth_cut = 12 := by
  sorry

end NUMINAMATH_CALUDE_helen_gas_usage_l3690_369033


namespace NUMINAMATH_CALUDE_inequality_solution_and_bound_l3690_369063

def f (x : ℝ) := |x - 3|
def g (x : ℝ) := |x - 2|

theorem inequality_solution_and_bound :
  (∀ x, f x + g x < 2 ↔ x ∈ Set.Ioo (3/2) (7/2)) ∧
  (∀ x y, f x ≤ 1 → g y ≤ 1 → |x - 2*y + 1| ≤ 3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_bound_l3690_369063


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l3690_369027

theorem quadratic_root_implies_k (k : ℝ) : 
  (2 * (5 : ℝ)^2 + 3 * 5 - k = 0) → k = 65 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l3690_369027


namespace NUMINAMATH_CALUDE_complex_number_problem_l3690_369079

theorem complex_number_problem (α β : ℂ) :
  (α - β).re > 0 →
  (2 * Complex.I * (α + β)).re > 0 →
  β = 4 + Complex.I →
  α = -4 + Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3690_369079


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l3690_369018

theorem students_playing_neither_sport (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ)
  (h_total : total = 40)
  (h_football : football = 26)
  (h_tennis : tennis = 20)
  (h_both : both = 17) :
  total - (football + tennis - both) = 11 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l3690_369018


namespace NUMINAMATH_CALUDE_intersection_condition_l3690_369080

def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}

def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

theorem intersection_condition (a : ℝ) : 
  (A ∩ B a = B a) ↔ (a < -4 ∨ a ≥ 4 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l3690_369080


namespace NUMINAMATH_CALUDE_system_solution_l3690_369058

theorem system_solution (x y k : ℝ) : 
  (2 * x + 3 * y = k) → 
  (x + 4 * y = k - 16) → 
  (x + y = 8) → 
  k = 12 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3690_369058


namespace NUMINAMATH_CALUDE_projection_a_on_b_l3690_369004

def a : ℝ × ℝ := (-8, 1)
def b : ℝ × ℝ := (3, 4)

theorem projection_a_on_b : 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_projection_a_on_b_l3690_369004


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3690_369077

theorem sum_of_squares_zero_implies_sum (a b c : ℝ) :
  (a - 2)^2 + (b - 6)^2 + (c - 8)^2 = 0 → a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3690_369077


namespace NUMINAMATH_CALUDE_partition_displacement_is_one_sixth_length_l3690_369084

/-- Represents a cylindrical vessel with a movable partition -/
structure Vessel where
  length : ℝ
  initial_partition_position : ℝ
  final_partition_position : ℝ

/-- Calculates the displacement of the partition -/
def partition_displacement (v : Vessel) : ℝ :=
  v.initial_partition_position - v.final_partition_position

/-- Theorem stating the displacement of the partition -/
theorem partition_displacement_is_one_sixth_length (v : Vessel) 
  (h1 : v.length > 0)
  (h2 : v.initial_partition_position = 2 * v.length / 3)
  (h3 : v.final_partition_position = v.length / 2) :
  partition_displacement v = v.length / 6 := by
  sorry

#check partition_displacement_is_one_sixth_length

end NUMINAMATH_CALUDE_partition_displacement_is_one_sixth_length_l3690_369084


namespace NUMINAMATH_CALUDE_right_angled_triangle_l3690_369009

theorem right_angled_triangle (A B C : ℝ) (h : Real.sin A + Real.sin B = Real.sin C * (Real.cos A + Real.cos B)) :
  Real.cos C = 0 :=
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l3690_369009


namespace NUMINAMATH_CALUDE_percentage_of_defective_meters_l3690_369007

theorem percentage_of_defective_meters 
  (total_meters : ℕ) 
  (rejected_meters : ℕ) 
  (h1 : total_meters = 200) 
  (h2 : rejected_meters = 20) : 
  (rejected_meters : ℝ) / (total_meters : ℝ) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_defective_meters_l3690_369007


namespace NUMINAMATH_CALUDE_remaining_area_formula_l3690_369081

/-- The remaining area of a rectangle with a hole -/
def remaining_area (x : ℝ) : ℝ :=
  (2*x + 5) * (x + 8) - (3*x - 2) * (x + 1)

/-- Theorem: The remaining area is equal to -x^2 + 20x + 42 -/
theorem remaining_area_formula (x : ℝ) :
  remaining_area x = -x^2 + 20*x + 42 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_formula_l3690_369081


namespace NUMINAMATH_CALUDE_survey_is_sample_l3690_369039

/-- Represents the total number of students in the population -/
def population_size : ℕ := 32000

/-- Represents the number of students surveyed -/
def survey_size : ℕ := 1600

/-- Represents a student's weight -/
structure Weight where
  value : ℝ

/-- Represents the population of all students' weights -/
def population : Finset Weight := sorry

/-- Represents the surveyed students' weights -/
def survey : Finset Weight := sorry

/-- Theorem stating that the survey is a sample of the population -/
theorem survey_is_sample : survey ⊆ population ∧ survey.card = survey_size := by sorry

end NUMINAMATH_CALUDE_survey_is_sample_l3690_369039


namespace NUMINAMATH_CALUDE_d_neither_sufficient_nor_necessary_for_a_l3690_369061

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : (A → B) ∧ ¬(B → A))  -- A is sufficient but not necessary for B
variable (h2 : (B ↔ C))             -- B is necessary and sufficient for C
variable (h3 : (D → C) ∧ ¬(C → D))  -- C is necessary but not sufficient for D

-- Theorem to prove
theorem d_neither_sufficient_nor_necessary_for_a :
  ¬((D → A) ∧ (A → D)) :=
sorry

end NUMINAMATH_CALUDE_d_neither_sufficient_nor_necessary_for_a_l3690_369061


namespace NUMINAMATH_CALUDE_triangle_area_l3690_369097

/-- Given a triangle with perimeter 32 and inradius 2.5, its area is 40 -/
theorem triangle_area (p r a : ℝ) (h1 : p = 32) (h2 : r = 2.5) (h3 : a = p * r / 4) : a = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3690_369097


namespace NUMINAMATH_CALUDE_root_equation_l3690_369020

theorem root_equation (k : ℝ) : 
  ((-2 : ℝ)^2 + k*(-2) - 2 = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_l3690_369020


namespace NUMINAMATH_CALUDE_set_operations_l3690_369014

def A : Set ℝ := {x | x ≤ 5}
def B : Set ℝ := {x | 3 < x ∧ x ≤ 7}

theorem set_operations :
  (A ∩ B = {x | 3 < x ∧ x ≤ 5}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ 5 ∨ x > 7}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3690_369014


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3690_369035

theorem polynomial_simplification (p : ℝ) :
  (4 * p^4 + 2 * p^3 - 7 * p + 3) + (5 * p^3 - 8 * p^2 + 3 * p + 2) =
  4 * p^4 + 7 * p^3 - 8 * p^2 - 4 * p + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3690_369035


namespace NUMINAMATH_CALUDE_dodecagon_rectangle_area_equality_l3690_369067

/-- The area of a regular dodecagon inscribed in a circle of radius r -/
def area_inscribed_dodecagon (r : ℝ) : ℝ := 3 * r^2

/-- The area of a rectangle with sides r and 3r -/
def area_rectangle (r : ℝ) : ℝ := r * (3 * r)

theorem dodecagon_rectangle_area_equality (r : ℝ) :
  area_inscribed_dodecagon r = area_rectangle r :=
by sorry

end NUMINAMATH_CALUDE_dodecagon_rectangle_area_equality_l3690_369067
