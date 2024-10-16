import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l3110_311008

def equation1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y = -12
def equation2 (x z : ℝ) : Prop := x^2 + z^2 - 6*x - 2*z = -5
def equation3 (y z : ℝ) : Prop := y^2 + z^2 - 8*y - 2*z = -7

def is_solution (x y z : ℝ) : Prop :=
  equation1 x y ∧ equation2 x z ∧ equation3 y z

theorem system_solution :
  (∀ x y z : ℝ, is_solution x y z ↔
    ((x = 1 ∧ y = 1 ∧ z = 0) ∨
     (x = 1 ∧ y = 1 ∧ z = 2) ∨
     (x = 1 ∧ y = 7 ∧ z = 0) ∨
     (x = 1 ∧ y = 7 ∧ z = 2) ∨
     (x = 5 ∧ y = 1 ∧ z = 0) ∨
     (x = 5 ∧ y = 1 ∧ z = 2) ∨
     (x = 5 ∧ y = 7 ∧ z = 0) ∨
     (x = 5 ∧ y = 7 ∧ z = 2))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3110_311008


namespace NUMINAMATH_CALUDE_shaded_squares_formula_l3110_311064

/-- Represents a row of squares in the pattern -/
structure Row :=
  (number : ℕ)  -- The row number
  (total : ℕ)   -- Total number of squares in the row
  (unshaded : ℕ) -- Number of unshaded squares
  (shaded : ℕ)   -- Number of shaded squares

/-- The properties of the sequence of rows -/
def ValidSequence (rows : ℕ → Row) : Prop :=
  (rows 1).total = 1 ∧ 
  (rows 1).unshaded = 1 ∧
  (rows 1).shaded = 0 ∧
  (∀ n : ℕ, n > 0 → (rows n).number = n) ∧
  (∀ n : ℕ, n > 1 → (rows n).total = (rows (n-1)).total + 2) ∧
  (∀ n : ℕ, n > 0 → (rows n).unshaded = (rows n).total - (rows n).shaded) ∧
  (∀ n : ℕ, n > 0 → (rows n).unshaded = n)

theorem shaded_squares_formula (rows : ℕ → Row) 
  (h : ValidSequence rows) (n : ℕ) (hn : n > 0) : 
  (rows n).shaded = n - 1 :=
sorry

end NUMINAMATH_CALUDE_shaded_squares_formula_l3110_311064


namespace NUMINAMATH_CALUDE_tank_filling_time_l3110_311020

/-- The time taken to fill a tank with two pipes and a leak -/
theorem tank_filling_time (pipe1_time pipe2_time : ℝ) (leak_fraction : ℝ) : 
  pipe1_time = 20 → 
  pipe2_time = 30 → 
  leak_fraction = 1/3 → 
  (1 / ((1 / pipe1_time + 1 / pipe2_time) * (1 - leak_fraction))) = 18 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l3110_311020


namespace NUMINAMATH_CALUDE_initial_squares_step_increase_squares_after_five_steps_l3110_311067

/-- The number of squares after n steps in the square subdivision process -/
def num_squares (n : ℕ) : ℕ := 5 + 4 * n

/-- The square subdivision process starts with 5 squares -/
theorem initial_squares : num_squares 0 = 5 := by sorry

/-- Each step adds 4 new squares -/
theorem step_increase (n : ℕ) : num_squares (n + 1) = num_squares n + 4 := by sorry

/-- The number of squares after 5 steps is 25 -/
theorem squares_after_five_steps : num_squares 5 = 25 := by sorry

end NUMINAMATH_CALUDE_initial_squares_step_increase_squares_after_five_steps_l3110_311067


namespace NUMINAMATH_CALUDE_even_function_implies_a_plus_minus_one_l3110_311061

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem even_function_implies_a_plus_minus_one (a : ℝ) :
  EvenFunction (fun x => x^2 + (a^2 - 1)*x + (a - 1)) →
  a = 1 ∨ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_plus_minus_one_l3110_311061


namespace NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_abs_rational_l3110_311099

theorem smallest_positive_largest_negative_smallest_abs_rational 
  (a b : ℤ) (c : ℚ) 
  (ha : a = 1) 
  (hb : b = -1) 
  (hc : c = 0) : a - b - c = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_abs_rational_l3110_311099


namespace NUMINAMATH_CALUDE_broken_clock_theorem_l3110_311069

/-- Represents the time shown on a clock --/
structure ClockTime where
  hours : ℕ
  minutes : ℕ

/-- Calculates the time shown on the broken clock after a given number of real minutes --/
def brokenClockTime (startTime : ClockTime) (realMinutes : ℕ) : ClockTime :=
  let totalMinutes := startTime.hours * 60 + startTime.minutes + realMinutes * 5 / 4
  { hours := totalMinutes / 60
    minutes := totalMinutes % 60 }

theorem broken_clock_theorem :
  let startTime := ClockTime.mk 14 0
  let realMinutes := 40
  brokenClockTime startTime realMinutes = ClockTime.mk 14 50 :=
by sorry

end NUMINAMATH_CALUDE_broken_clock_theorem_l3110_311069


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l3110_311092

theorem quadratic_equivalence : 
  ∀ x y : ℝ, y = x^2 - 2*x + 3 ↔ y = (x - 1)^2 + 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l3110_311092


namespace NUMINAMATH_CALUDE_inequality_proof_l3110_311021

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3110_311021


namespace NUMINAMATH_CALUDE_louise_teddy_bears_louise_teddy_bears_correct_l3110_311070

theorem louise_teddy_bears (initial_toys : ℕ) (initial_toy_cost : ℕ) 
  (total_money : ℕ) (teddy_bear_cost : ℕ) : ℕ :=
  let remaining_money := total_money - initial_toys * initial_toy_cost
  remaining_money / teddy_bear_cost

theorem louise_teddy_bears_correct 
  (initial_toys : ℕ) (initial_toy_cost : ℕ) 
  (total_money : ℕ) (teddy_bear_cost : ℕ) :
  louise_teddy_bears initial_toys initial_toy_cost total_money teddy_bear_cost = 20 ∧
  initial_toys = 28 ∧
  initial_toy_cost = 10 ∧
  total_money = 580 ∧
  teddy_bear_cost = 15 ∧
  total_money = initial_toys * initial_toy_cost + 
    (louise_teddy_bears initial_toys initial_toy_cost total_money teddy_bear_cost) * teddy_bear_cost :=
by
  sorry

end NUMINAMATH_CALUDE_louise_teddy_bears_louise_teddy_bears_correct_l3110_311070


namespace NUMINAMATH_CALUDE_marco_painting_fraction_l3110_311036

theorem marco_painting_fraction (marco_rate carla_rate : ℚ) : 
  marco_rate = 1 / 60 →
  marco_rate + carla_rate = 1 / 40 →
  marco_rate * 32 = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_marco_painting_fraction_l3110_311036


namespace NUMINAMATH_CALUDE_second_group_factories_l3110_311031

theorem second_group_factories (total : ℕ) (first_group : ℕ) (unchecked : ℕ) :
  total = 169 → first_group = 69 → unchecked = 48 →
  total - (first_group + unchecked) = 52 := by
  sorry

end NUMINAMATH_CALUDE_second_group_factories_l3110_311031


namespace NUMINAMATH_CALUDE_max_product_of_ranged_functions_l3110_311040

/-- Given two functions f and g defined on ℝ with specific ranges, 
    prove that the maximum value of their product is -1 -/
theorem max_product_of_ranged_functions 
  (f g : ℝ → ℝ) 
  (hf : ∀ x, 1 ≤ f x ∧ f x ≤ 6) 
  (hg : ∀ x, -4 ≤ g x ∧ g x ≤ -1) : 
  (∀ x, f x * g x ≤ -1) ∧ (∃ x, f x * g x = -1) :=
sorry

end NUMINAMATH_CALUDE_max_product_of_ranged_functions_l3110_311040


namespace NUMINAMATH_CALUDE_prime_has_property_P_infinitely_many_composite_with_property_P_l3110_311091

-- Define property P
def has_property_P (n : ℕ) : Prop :=
  ∀ a : ℕ, a > 0 → (n ∣ a^n - 1) → (n^2 ∣ a^n - 1)

-- Theorem 1: Every prime number has property P
theorem prime_has_property_P :
  ∀ p : ℕ, Prime p → has_property_P p :=
sorry

-- Define a set of composite numbers with property P
def composite_with_property_P : Set ℕ :=
  {n : ℕ | ¬Prime n ∧ has_property_P n}

-- Theorem 2: There are infinitely many composite numbers with property P
theorem infinitely_many_composite_with_property_P :
  Set.Infinite composite_with_property_P :=
sorry

end NUMINAMATH_CALUDE_prime_has_property_P_infinitely_many_composite_with_property_P_l3110_311091


namespace NUMINAMATH_CALUDE_cube_root_monotone_l3110_311024

theorem cube_root_monotone (a b : ℝ) (h : a ≤ b) : a ^ (1/3) ≤ b ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_monotone_l3110_311024


namespace NUMINAMATH_CALUDE_sally_net_earnings_two_months_l3110_311081

-- Define the given values
def last_month_work_income : ℝ := 1000
def last_month_work_expenses : ℝ := 200
def last_month_side_hustle : ℝ := 150
def work_income_increase : ℝ := 0.1
def work_expenses_increase : ℝ := 0.15
def side_hustle_increase : ℝ := 0.2
def tax_rate : ℝ := 0.25

-- Define the calculation functions
def calculate_net_work_income (income : ℝ) (expenses : ℝ) : ℝ :=
  income - expenses - (tax_rate * income)

def calculate_total_net_earnings (work_income : ℝ) (side_hustle : ℝ) : ℝ :=
  calculate_net_work_income work_income last_month_work_expenses + side_hustle

-- Theorem statement
theorem sally_net_earnings_two_months :
  let last_month := calculate_total_net_earnings last_month_work_income last_month_side_hustle
  let this_month := calculate_total_net_earnings 
    (last_month_work_income * (1 + work_income_increase))
    (last_month_side_hustle * (1 + side_hustle_increase))
  last_month + this_month = 1475 := by sorry

end NUMINAMATH_CALUDE_sally_net_earnings_two_months_l3110_311081


namespace NUMINAMATH_CALUDE_divides_n_l3110_311022

def n : ℕ := sorry

theorem divides_n : 1980 ∣ n := by sorry

end NUMINAMATH_CALUDE_divides_n_l3110_311022


namespace NUMINAMATH_CALUDE_hypotenuse_product_equals_area_l3110_311057

/-- A right-angled triangle with an incircle -/
structure RightTriangleWithIncircle where
  /-- The area of the triangle -/
  area : ℝ
  /-- The radius of the incircle -/
  incircle_radius : ℝ
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The first part of the hypotenuse divided by the incircle's point of contact -/
  x : ℝ
  /-- The second part of the hypotenuse divided by the incircle's point of contact -/
  y : ℝ
  /-- The sum of x and y is equal to the hypotenuse -/
  hypotenuse_division : x + y = hypotenuse
  /-- All lengths are positive -/
  all_positive : 0 < area ∧ 0 < incircle_radius ∧ 0 < hypotenuse ∧ 0 < x ∧ 0 < y

/-- The theorem stating that the product of the two parts of the hypotenuse 
    is equal to the area of the right-angled triangle with an incircle -/
theorem hypotenuse_product_equals_area (t : RightTriangleWithIncircle) : t.x * t.y = t.area := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_product_equals_area_l3110_311057


namespace NUMINAMATH_CALUDE_no_prime_solution_l3110_311023

def base_p_to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.foldl (fun acc d => acc * p + d) 0

theorem no_prime_solution :
  ¬ ∃ (p : Nat), Prime p ∧ 
    (base_p_to_decimal [2,0,3,4] p + 
     base_p_to_decimal [4,0,5] p + 
     base_p_to_decimal [1,2] p + 
     base_p_to_decimal [2,1,2] p + 
     base_p_to_decimal [7] p = 
     base_p_to_decimal [1,3,1,5] p + 
     base_p_to_decimal [5,4,1] p + 
     base_p_to_decimal [2,2,2] p) :=
by
  sorry


end NUMINAMATH_CALUDE_no_prime_solution_l3110_311023


namespace NUMINAMATH_CALUDE_work_rate_equality_l3110_311089

/-- Represents the work rate of a person -/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers -/
structure WorkerGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work rate of a group -/
def totalWorkRate (g : WorkerGroup) (m w : WorkRate) : ℝ :=
  (g.men : ℝ) * m.rate + (g.women : ℝ) * w.rate

theorem work_rate_equality 
  (m w : WorkRate) 
  (group1 group2 group3 : WorkerGroup) :
  group1.men = 3 →
  group1.women = 8 →
  group2.women = 2 →
  group3.men = 3 →
  group3.women = 2 →
  totalWorkRate group1 m w = totalWorkRate group2 m w →
  totalWorkRate group3 m w = (4/7) * totalWorkRate group1 m w →
  group2.men = 6 := by
  sorry

#check work_rate_equality

end NUMINAMATH_CALUDE_work_rate_equality_l3110_311089


namespace NUMINAMATH_CALUDE_polynomial_transformation_l3110_311096

theorem polynomial_transformation (x : ℝ) (hx : x ≠ 0) :
  let z := x - 1 / x
  x^4 - 3*x^3 - 2*x^2 + 3*x + 1 = x^2 * (z^2 - 3*z) := by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l3110_311096


namespace NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l3110_311068

/-- AMC 12 scoring system and problem parameters -/
structure AMC12Params where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Rat
  incorrect_penalty : Rat
  unanswered_points : Rat
  target_score : Rat

/-- Calculate the score based on the number of correct answers -/
def calculate_score (params : AMC12Params) (correct : Nat) : Rat :=
  let incorrect := params.attempted_problems - correct
  let unanswered := params.total_problems - params.attempted_problems
  correct * params.correct_points + 
  incorrect * (-params.incorrect_penalty) + 
  unanswered * params.unanswered_points

/-- The main theorem to prove -/
theorem min_correct_answers_for_target_score 
  (params : AMC12Params)
  (h_total : params.total_problems = 25)
  (h_attempted : params.attempted_problems = 15)
  (h_correct_points : params.correct_points = 7.5)
  (h_incorrect_penalty : params.incorrect_penalty = 2)
  (h_unanswered_points : params.unanswered_points = 2)
  (h_target_score : params.target_score = 120) :
  (∀ k < 14, calculate_score params k < params.target_score) ∧ 
  calculate_score params 14 ≥ params.target_score := by
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l3110_311068


namespace NUMINAMATH_CALUDE_divide_and_add_problem_l3110_311087

theorem divide_and_add_problem (x : ℝ) : (48 / x) + 7 = 15 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_divide_and_add_problem_l3110_311087


namespace NUMINAMATH_CALUDE_block_run_difference_l3110_311077

/-- The difference in distance run around a square block between outer and inner paths -/
theorem block_run_difference (block_side : ℝ) (street_width : ℝ) : 
  block_side = 500 → street_width = 30 → 
  (4 * (block_side + street_width / 2) * π / 2) = 1030 * π := by sorry

end NUMINAMATH_CALUDE_block_run_difference_l3110_311077


namespace NUMINAMATH_CALUDE_water_usage_median_and_mode_l3110_311018

def water_usage : List ℝ := [7, 5, 6, 8, 9, 9, 10]

def median (l : List ℝ) : ℝ := sorry

def mode (l : List ℝ) : ℝ := sorry

theorem water_usage_median_and_mode :
  median water_usage = 8 ∧ mode water_usage = 9 := by sorry

end NUMINAMATH_CALUDE_water_usage_median_and_mode_l3110_311018


namespace NUMINAMATH_CALUDE_quadratic_function_extrema_l3110_311007

def f (x : ℝ) := 3 * x^2 + 6 * x - 5

theorem quadratic_function_extrema :
  ∃ (min max : ℝ), 
    (∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x₂ = max) ∧
    min = -8 ∧ max = 19 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_extrema_l3110_311007


namespace NUMINAMATH_CALUDE_smallest_x_plus_y_l3110_311041

theorem smallest_x_plus_y : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧ x < y ∧
  (100 : ℚ) + (x : ℚ) / y = 2 * ((100 : ℚ) * x / y) ∧
  ∀ (a b : ℕ), a > 0 → b > 0 → a < b → 
    (100 : ℚ) + (a : ℚ) / b = 2 * ((100 : ℚ) * a / b) →
    x + y ≤ a + b ∧
  x + y = 299 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_plus_y_l3110_311041


namespace NUMINAMATH_CALUDE_probability_different_digits_l3110_311028

def is_valid_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_different_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

def count_valid_numbers : ℕ := 999 - 100 + 1

def count_numbers_with_different_digits : ℕ := 9 * 9 * 8

theorem probability_different_digits :
  (count_numbers_with_different_digits : ℚ) / count_valid_numbers = 18 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_digits_l3110_311028


namespace NUMINAMATH_CALUDE_hdha_ratio_is_zero_l3110_311030

/-- A triangle with sides of lengths 8, 15, and 17 -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 8
  hb : b = 15
  hc : c = 17

/-- The orthocenter (intersection of altitudes) of the triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The foot of the altitude from A to side BC -/
def altitudeFoot (t : Triangle) : ℝ × ℝ := sorry

/-- The vertex A of the triangle -/
def vertexA (t : Triangle) : ℝ × ℝ := sorry

/-- The ratio of HD to HA, where H is the orthocenter and D is the foot of the altitude from A -/
def hdhaRatio (t : Triangle) : ℝ := sorry

theorem hdha_ratio_is_zero (t : Triangle) : hdhaRatio t = 0 := by
  sorry

end NUMINAMATH_CALUDE_hdha_ratio_is_zero_l3110_311030


namespace NUMINAMATH_CALUDE_percentage_problem_l3110_311003

/-- Given that 15% of 40 is greater than y% of 16 by 2, prove that y = 25 -/
theorem percentage_problem (y : ℝ) : 
  (0.15 * 40 = y / 100 * 16 + 2) → y = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3110_311003


namespace NUMINAMATH_CALUDE_homogeneous_polynomial_theorem_l3110_311066

variable {n : ℕ}

-- Define a homogeneous polynomial of degree n
def IsHomogeneousPolynomial (f : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (x y t : ℝ), f (t * x) (t * y) = t^n * f x y

theorem homogeneous_polynomial_theorem (f : ℝ → ℝ → ℝ) (n : ℕ) 
  (h1 : IsHomogeneousPolynomial f n)
  (h2 : f 1 0 = 1)
  (h3 : ∀ (a b c : ℝ), f (a + b) c + f (b + c) a + f (c + a) b = 0) :
  ∀ (x y : ℝ), f x y = (x - 2*y) * (x + y)^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_homogeneous_polynomial_theorem_l3110_311066


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3110_311048

/-- Given a point Q(a-1, a+2) that lies on the x-axis, prove that its coordinates are (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (∃ Q : ℝ × ℝ, Q.1 = a - 1 ∧ Q.2 = a + 2 ∧ Q.2 = 0) → 
  (∃ Q : ℝ × ℝ, Q = (-3, 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3110_311048


namespace NUMINAMATH_CALUDE_reunion_handshakes_l3110_311095

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a reunion of 11 boys where each boy shakes hands exactly once with each of the others, 
    the total number of handshakes is 55 -/
theorem reunion_handshakes : handshakes 11 = 55 := by
  sorry

end NUMINAMATH_CALUDE_reunion_handshakes_l3110_311095


namespace NUMINAMATH_CALUDE_circle_circumference_increase_l3110_311019

/-- Given two circles, where the diameter of the first increases by 2π,
    the proportional increase in the circumference of the second is 2π² -/
theorem circle_circumference_increase (d₁ d₂ : ℝ) : 
  let increase_diameter : ℝ := 2 * Real.pi
  let increase_circumference : ℝ → ℝ := λ x => Real.pi * x
  increase_circumference increase_diameter = 2 * Real.pi^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_increase_l3110_311019


namespace NUMINAMATH_CALUDE_hypotenuse_length_from_quadratic_roots_l3110_311005

theorem hypotenuse_length_from_quadratic_roots :
  ∀ a b c : ℝ,
  (a^2 - 6*a + 4 = 0) →
  (b^2 - 6*b + 4 = 0) →
  (a ≠ b) →
  (c^2 = a^2 + b^2) →
  c = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_length_from_quadratic_roots_l3110_311005


namespace NUMINAMATH_CALUDE_square_sum_value_l3110_311055

theorem square_sum_value (x y : ℝ) (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l3110_311055


namespace NUMINAMATH_CALUDE_race_speeds_l3110_311083

theorem race_speeds (b c : ℝ) (h₁ : b > 0) (h₂ : c > 0) :
  let x := (c + Real.sqrt (c^2 + 120*b*c)) / 2
  let y := (-c + Real.sqrt (c^2 + 120*b*c)) / 2
  b / y - b / x = 1 / 30 ∧ b / (x - c) - b / (y + c) = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_race_speeds_l3110_311083


namespace NUMINAMATH_CALUDE_min_cells_in_square_sheet_exists_min_square_sheet_l3110_311088

/-- Represents a rectangular shape with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square shape with side length -/
structure Square where
  side : ℕ

/-- The ship cut out from the paper -/
def ship : Rectangle :=
  { width := 10, height := 11 }

/-- Theorem: The minimum number of cells in the square sheet of paper is 121 -/
theorem min_cells_in_square_sheet : 
  ∀ (s : Square), s.side ≥ max ship.width ship.height → s.side * s.side ≥ 121 :=
by
  sorry

/-- Corollary: There exists a square sheet with exactly 121 cells that can fit the ship -/
theorem exists_min_square_sheet :
  ∃ (s : Square), s.side * s.side = 121 ∧ s.side ≥ max ship.width ship.height :=
by
  sorry

end NUMINAMATH_CALUDE_min_cells_in_square_sheet_exists_min_square_sheet_l3110_311088


namespace NUMINAMATH_CALUDE_min_orders_for_given_conditions_l3110_311045

/-- The minimum number of orders required to purchase a given number of items
    while minimizing the total cost under specific discount conditions. -/
def min_orders (original_price : ℚ) (total_items : ℕ) (discount_percent : ℚ) 
                (additional_discount_threshold : ℚ) (additional_discount : ℚ) : ℕ :=
  sorry

/-- The theorem stating that the minimum number of orders is 4 under the given conditions. -/
theorem min_orders_for_given_conditions : 
  min_orders 48 42 0.6 300 100 = 4 := by sorry

end NUMINAMATH_CALUDE_min_orders_for_given_conditions_l3110_311045


namespace NUMINAMATH_CALUDE_rational_function_value_l3110_311026

/-- A rational function with specific properties -/
def rational_function (p q : ℝ → ℝ) : Prop :=
  ∃ k : ℝ,
    (∀ x, q x = (x + 5) * (x - 1)) ∧
    (∀ x, p x = k * x) ∧
    (p 0 / q 0 = 0) ∧
    (p 4 / q 4 = -1/2)

/-- The main theorem -/
theorem rational_function_value (p q : ℝ → ℝ) 
  (h : rational_function p q) : p (-1) / q (-1) = 27/64 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l3110_311026


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3110_311001

/-- The eccentricity range of a hyperbola -/
theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  let c := Real.sqrt (a^2 + b^2)
  let d := a * b / c
  d ≥ Real.sqrt 2 / 3 * c →
  Real.sqrt 6 / 2 ≤ e ∧ e ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3110_311001


namespace NUMINAMATH_CALUDE_product_of_conjugates_l3110_311004

theorem product_of_conjugates (x p q : ℝ) :
  (x + p / 2 - Real.sqrt (p^2 / 4 - q)) * (x + p / 2 + Real.sqrt (p^2 / 4 - q)) = x^2 + p * x + q :=
by sorry

end NUMINAMATH_CALUDE_product_of_conjugates_l3110_311004


namespace NUMINAMATH_CALUDE_shift_right_result_l3110_311014

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Shifts a linear function horizontally -/
def shift_right (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b - f.m * shift }

theorem shift_right_result :
  let original := LinearFunction.mk 2 4
  let shifted := shift_right original 2
  shifted = LinearFunction.mk 2 0 := by sorry

end NUMINAMATH_CALUDE_shift_right_result_l3110_311014


namespace NUMINAMATH_CALUDE_volunteer_selection_count_l3110_311012

/-- The number of ways to select 3 volunteers from 5 boys and 2 girls, with at least 1 girl selected -/
def select_volunteers (num_boys : ℕ) (num_girls : ℕ) (total_selected : ℕ) : ℕ :=
  Nat.choose num_girls 1 * Nat.choose num_boys 2 +
  Nat.choose num_girls 2 * Nat.choose num_boys 1

/-- Theorem stating that the number of ways to select 3 volunteers from 5 boys and 2 girls, 
    with at least 1 girl selected, is equal to 25 -/
theorem volunteer_selection_count :
  select_volunteers 5 2 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_count_l3110_311012


namespace NUMINAMATH_CALUDE_absolute_difference_of_factors_l3110_311085

theorem absolute_difference_of_factors (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 7) : 
  |m - n| = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_factors_l3110_311085


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3110_311002

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  (∃ (p q : ℝ × ℝ), p = (1, -Real.sqrt 3) ∧ q = (Real.cos B, Real.sin B) ∧ p.1 * q.2 = p.2 * q.1) →
  b * Real.cos C + c * Real.cos B = 2 * a * Real.sin A →
  A + B + C = Real.pi →
  C = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3110_311002


namespace NUMINAMATH_CALUDE_unique_prime_pair_with_square_differences_l3110_311086

theorem unique_prime_pair_with_square_differences : 
  ∃! (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    ∃ (a b : ℕ), a^2 = p - q ∧ b^2 = p*q - q :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_with_square_differences_l3110_311086


namespace NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l3110_311079

theorem stratified_sampling_male_athletes 
  (total_athletes : ℕ) 
  (male_athletes : ℕ) 
  (selected_athletes : ℕ) 
  (h1 : total_athletes = 98) 
  (h2 : male_athletes = 56) 
  (h3 : selected_athletes = 28) :
  (male_athletes : ℚ) / total_athletes * selected_athletes = 16 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l3110_311079


namespace NUMINAMATH_CALUDE_milk_conversion_rate_l3110_311093

/-- The number of ounces in a gallon of milk -/
def ounces_per_gallon : ℕ := sorry

/-- The initial amount of milk in gallons -/
def initial_gallons : ℕ := 3

/-- The amount of milk consumed in ounces -/
def consumed_ounces : ℕ := 13

/-- The remaining amount of milk in ounces -/
def remaining_ounces : ℕ := 371

theorem milk_conversion_rate :
  ounces_per_gallon = 128 :=
by sorry

end NUMINAMATH_CALUDE_milk_conversion_rate_l3110_311093


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3110_311034

theorem constant_term_binomial_expansion :
  let n : ℕ := 8
  let a : ℚ := 1
  let b : ℚ := -2/3
  let r : ℕ := 6
  (Nat.choose n r) * a^(n-r) * b^r = 1792 := by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3110_311034


namespace NUMINAMATH_CALUDE_luke_trivia_score_l3110_311054

/-- Luke's trivia game score calculation -/
theorem luke_trivia_score (points_per_round : ℕ) (num_rounds : ℕ) :
  points_per_round = 146 →
  num_rounds = 157 →
  points_per_round * num_rounds = 22822 := by
  sorry

end NUMINAMATH_CALUDE_luke_trivia_score_l3110_311054


namespace NUMINAMATH_CALUDE_square_property_l3110_311046

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def remove_last_two_digits (n : ℕ) : ℕ := n / 100

theorem square_property (n : ℕ) :
  (n > 0 ∧ is_perfect_square (remove_last_two_digits (n^2))) ↔
  (∃ k : ℕ, k > 0 ∧ n = 10 * k) ∨
  (n ∈ ({11,12,13,14,21,22,31,41,1,2,3,4,5,6,7,8,9} : Finset ℕ)) :=
sorry

end NUMINAMATH_CALUDE_square_property_l3110_311046


namespace NUMINAMATH_CALUDE_distance_product_theorem_l3110_311090

theorem distance_product_theorem (a b : ℝ) (θ : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let point1 := (Real.sqrt (a^2 - b^2), 0)
  let point2 := (-Real.sqrt (a^2 - b^2), 0)
  let line := fun (x y : ℝ) ↦ x * Real.cos θ / a + y * Real.sin θ / b = 1
  let distance (p : ℝ × ℝ) := 
    abs (b * Real.cos θ * p.1 + a * Real.sin θ * p.2 - a * b) / 
    Real.sqrt ((b * Real.cos θ)^2 + (a * Real.sin θ)^2)
  (distance point1) * (distance point2) = b^2 := by
sorry

end NUMINAMATH_CALUDE_distance_product_theorem_l3110_311090


namespace NUMINAMATH_CALUDE_prime_seven_mod_eight_not_sum_three_squares_l3110_311016

theorem prime_seven_mod_eight_not_sum_three_squares (p : ℕ) (hp : Nat.Prime p) (hm : p % 8 = 7) :
  ¬ ∃ (a b c : ℤ), (a * a + b * b + c * c : ℤ) = p := by
  sorry

end NUMINAMATH_CALUDE_prime_seven_mod_eight_not_sum_three_squares_l3110_311016


namespace NUMINAMATH_CALUDE_translation_of_sine_to_cosine_l3110_311073

/-- Given a function f(x) = sin(2x + π/6), prove that translating it π/6 units to the left
    results in the function g(x) = cos(2x) -/
theorem translation_of_sine_to_cosine (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 6)
  let g : ℝ → ℝ := λ x => f (x + π / 6)
  g x = Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_translation_of_sine_to_cosine_l3110_311073


namespace NUMINAMATH_CALUDE_stephanie_silverware_l3110_311076

/-- The number of types of silverware Stephanie needs to buy -/
def numTypes : ℕ := 4

/-- The initial number of pieces Stephanie plans to buy for each type -/
def initialPlan : ℕ := 5 + 10

/-- The reduction in the number of spoons and butter knives -/
def reductionSpoonsButter : ℕ := 4

/-- The reduction in the number of steak knives -/
def reductionSteak : ℕ := 5

/-- The reduction in the number of forks -/
def reductionForks : ℕ := 3

/-- The total number of silverware pieces Stephanie will buy -/
def totalSilverware : ℕ := 
  (initialPlan - reductionSpoonsButter) + 
  (initialPlan - reductionSpoonsButter) + 
  (initialPlan - reductionSteak) + 
  (initialPlan - reductionForks)

theorem stephanie_silverware : totalSilverware = 44 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_silverware_l3110_311076


namespace NUMINAMATH_CALUDE_book_cost_price_l3110_311010

def cost_price : ℝ → Prop := λ c => 
  (c * 1.1 + 90 = c * 1.15) ∧ 
  (c > 0)

theorem book_cost_price : ∃ c, cost_price c ∧ c = 1800 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l3110_311010


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3110_311043

-- Define the sets P and Q
def P : Set ℝ := {x | Real.log x > 0}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the open interval (1, 2)
def open_interval_one_two : Set ℝ := {x | 1 < x ∧ x < 2}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = open_interval_one_two := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3110_311043


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3110_311039

/-- Given a function f : ℝ → ℝ with f(0) = 1 and f'(x) > f(x) for all x,
    the set of x where f(x) > e^x is (0, +∞) -/
theorem solution_set_of_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h0 : f 0 = 1) (h1 : ∀ x, deriv f x > f x) :
    {x : ℝ | f x > Real.exp x} = Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3110_311039


namespace NUMINAMATH_CALUDE_probability_two_absent_one_present_l3110_311097

/-- The probability of a student being absent on a given day -/
def p_absent : ℚ := 2 / 25

/-- The probability of a student being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of students chosen -/
def n_students : ℕ := 3

/-- The number of students that should be absent -/
def n_absent : ℕ := 2

-- Theorem statement
theorem probability_two_absent_one_present :
  (n_students.choose n_absent : ℚ) * p_absent ^ n_absent * p_present ^ (n_students - n_absent) = 276 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_absent_one_present_l3110_311097


namespace NUMINAMATH_CALUDE_a_in_set_a_b_l3110_311056

universe u

variables {a b : Type u}

/-- Prove that a is an element of the set {a, b}. -/
theorem a_in_set_a_b : a ∈ ({a, b} : Set (Type u)) := by
  sorry

end NUMINAMATH_CALUDE_a_in_set_a_b_l3110_311056


namespace NUMINAMATH_CALUDE_chessboard_coverage_l3110_311080

/-- An L-shaped piece covers exactly 3 squares -/
def L_shape_coverage : ℕ := 3

/-- A unit square piece covers exactly 1 square -/
def unit_square_coverage : ℕ := 1

/-- Predicate to determine if an n×n chessboard can be covered -/
def can_cover (n : ℕ) : Prop :=
  ∃ k : ℕ, n^2 = k * L_shape_coverage ∨ n^2 = k * L_shape_coverage + unit_square_coverage

theorem chessboard_coverage (n : ℕ) :
  ¬(can_cover n) ↔ n % 3 = 2 :=
sorry

end NUMINAMATH_CALUDE_chessboard_coverage_l3110_311080


namespace NUMINAMATH_CALUDE_odd_function_solution_set_l3110_311015

open Set

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def slope_condition (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 1

theorem odd_function_solution_set 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_slope : slope_condition f) 
  (h_f1 : f 1 = 1) :
  {x : ℝ | f x - x > 0} = Iio (-1) ∪ Ioo 0 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_solution_set_l3110_311015


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3110_311074

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 4, 6}

theorem complement_of_M_in_U : 
  (U \ M) = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3110_311074


namespace NUMINAMATH_CALUDE_function_always_positive_l3110_311032

theorem function_always_positive (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  (∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (∀ x : ℝ, x < 1 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_function_always_positive_l3110_311032


namespace NUMINAMATH_CALUDE_crayons_per_box_l3110_311075

theorem crayons_per_box (total_crayons : Float) (total_boxes : Float) 
  (h1 : total_crayons = 7.0)
  (h2 : total_boxes = 1.4) :
  total_crayons / total_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_box_l3110_311075


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3110_311047

theorem perfect_square_condition (n : ℕ) : 
  (∃ m : ℕ, n^4 + 4*n^3 + 5*n^2 + 6*n = m^2) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3110_311047


namespace NUMINAMATH_CALUDE_percentage_knives_after_trade_l3110_311044

/-- Represents Carolyn's silverware set -/
structure Silverware :=
  (knives : ℕ)
  (forks : ℕ)
  (spoons : ℕ)

/-- Calculates the total number of silverware pieces -/
def total_silverware (s : Silverware) : ℕ :=
  s.knives + s.forks + s.spoons

/-- Calculates the percentage of knives in the silverware set -/
def percentage_knives (s : Silverware) : ℚ :=
  (s.knives : ℚ) / (total_silverware s : ℚ) * 100

/-- The initial silverware set -/
def initial_set : Silverware :=
  { knives := 6
  , forks := 12
  , spoons := 6 * 3 }

/-- The silverware set after the trade -/
def after_trade_set : Silverware :=
  { knives := initial_set.knives + 10
  , forks := initial_set.forks
  , spoons := initial_set.spoons - 6 }

theorem percentage_knives_after_trade :
  percentage_knives after_trade_set = 40 := by
  sorry


end NUMINAMATH_CALUDE_percentage_knives_after_trade_l3110_311044


namespace NUMINAMATH_CALUDE_inequality_proof_l3110_311098

theorem inequality_proof (x y z : ℝ) (n : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) (hn : n > 0) : 
  (x^4 / (y * (1 - y^n))) + (y^4 / (z * (1 - z^n))) + (z^4 / (x * (1 - x^n))) ≥ 
  3^n / (3^(n+2) - 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3110_311098


namespace NUMINAMATH_CALUDE_train_length_calculation_l3110_311011

/-- Prove that given two trains of equal length running on parallel lines in the same direction,
    with the faster train moving at 46 km/hr and the slower train at 36 km/hr,
    if the faster train completely passes the slower train in 18 seconds,
    then the length of each train is 25 meters. -/
theorem train_length_calculation (faster_speed slower_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 18 →
  (faster_speed - slower_speed) * (5 / 18) * passing_time = 2 * train_length →
  train_length = 25 := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l3110_311011


namespace NUMINAMATH_CALUDE_sum_f_negative_l3110_311063

def f (x : ℝ) : ℝ := x + x^3 + x^5

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₂ + x₃ < 0) (h₃ : x₃ + x₁ < 0) :
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l3110_311063


namespace NUMINAMATH_CALUDE_table_leg_problem_l3110_311051

theorem table_leg_problem :
  ∀ (x y : ℕ),
    x ≥ 2 →
    y ≥ 2 →
    3 * x + 4 * y = 23 →
    x = 5 ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_table_leg_problem_l3110_311051


namespace NUMINAMATH_CALUDE_rose_apples_l3110_311078

/-- The number of friends Rose shares her apples with -/
def num_friends : ℕ := 3

/-- The number of apples each friend would get if Rose shares her apples -/
def apples_per_friend : ℕ := 3

/-- The total number of apples Rose has -/
def total_apples : ℕ := num_friends * apples_per_friend

theorem rose_apples : total_apples = 9 := by sorry

end NUMINAMATH_CALUDE_rose_apples_l3110_311078


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3110_311059

/-- Represents the number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3110_311059


namespace NUMINAMATH_CALUDE_matches_for_512_players_l3110_311060

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_players : ℕ
  matches_eliminate_one : Bool

/-- The number of matches needed to determine the winner in a single-elimination tournament -/
def matches_needed (t : Tournament) : ℕ :=
  t.num_players - 1

/-- Theorem stating the number of matches needed for a 512-player single-elimination tournament -/
theorem matches_for_512_players (t : Tournament) 
  (h1 : t.num_players = 512) 
  (h2 : t.matches_eliminate_one = true) : 
  matches_needed t = 511 := by
  sorry

end NUMINAMATH_CALUDE_matches_for_512_players_l3110_311060


namespace NUMINAMATH_CALUDE_sequence_max_ratio_l3110_311084

theorem sequence_max_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → S n = (n + 1) / 2 * a n) →
  (∃ M : ℝ, ∀ n : ℕ, n > 1 → a n / a (n - 1) ≤ M) ∧
  (∀ ε > 0, ∃ n : ℕ, n > 1 ∧ a n / a (n - 1) > 2 - ε) :=
by sorry

end NUMINAMATH_CALUDE_sequence_max_ratio_l3110_311084


namespace NUMINAMATH_CALUDE_three_sixes_probability_l3110_311027

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 6
def biased_die_2_prob_six : ℚ := 1 / 2
def biased_die_2_prob_other : ℚ := 1 / 10
def biased_die_3_prob_six : ℚ := 3 / 4
def biased_die_3_prob_other : ℚ := 1 / 5  -- (1 - 3/4) / 5

-- Define the probability of choosing each die
def choose_die_prob : ℚ := 1 / 3

-- Define the event of rolling two sixes
def two_sixes_prob (die_prob : ℚ) : ℚ := die_prob * die_prob

-- Define the theorem
theorem three_sixes_probability :
  let total_two_sixes := 
    choose_die_prob * two_sixes_prob fair_die_prob +
    choose_die_prob * two_sixes_prob biased_die_2_prob_six +
    choose_die_prob * two_sixes_prob biased_die_3_prob_six
  let prob_fair_given_two_sixes := 
    (choose_die_prob * two_sixes_prob fair_die_prob) / total_two_sixes
  let prob_biased_2_given_two_sixes := 
    (choose_die_prob * two_sixes_prob biased_die_2_prob_six) / total_two_sixes
  let prob_biased_3_given_two_sixes := 
    (choose_die_prob * two_sixes_prob biased_die_3_prob_six) / total_two_sixes
  (prob_fair_given_two_sixes * fair_die_prob + 
   prob_biased_2_given_two_sixes * biased_die_2_prob_six + 
   prob_biased_3_given_two_sixes * biased_die_3_prob_six) = 109 / 148 := by
  sorry

end NUMINAMATH_CALUDE_three_sixes_probability_l3110_311027


namespace NUMINAMATH_CALUDE_sqrt_24_plus_3_bounds_l3110_311062

theorem sqrt_24_plus_3_bounds :
  (4 < Real.sqrt 24) ∧ (Real.sqrt 24 < 5) →
  (7 < Real.sqrt 24 + 3) ∧ (Real.sqrt 24 + 3 < 8) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_24_plus_3_bounds_l3110_311062


namespace NUMINAMATH_CALUDE_no_poly3_satisfies_conditions_l3110_311006

/-- A polynomial function of degree exactly 3 -/
structure Poly3 where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  degree_three : a ≠ 0

/-- Evaluation of a Poly3 at a point -/
def Poly3.eval (p : Poly3) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The conditions that the polynomial must satisfy -/
def satisfies_conditions (p : Poly3) : Prop :=
  ∀ x, p.eval (x^2) = (p.eval x)^2 ∧
       p.eval (x^2) = p.eval (p.eval x) ∧
       p.eval 1 = 2

theorem no_poly3_satisfies_conditions :
  ¬∃ p : Poly3, satisfies_conditions p :=
sorry

end NUMINAMATH_CALUDE_no_poly3_satisfies_conditions_l3110_311006


namespace NUMINAMATH_CALUDE_inequality_range_l3110_311037

theorem inequality_range : 
  (∃ (a : ℝ), ∀ (x : ℝ), |x - 1| - |x + 1| ≤ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), |x - 1| - |x + 1| ≤ b) → b ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3110_311037


namespace NUMINAMATH_CALUDE_min_value_ab_l3110_311042

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + 9 * b + 7) :
  a * b ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l3110_311042


namespace NUMINAMATH_CALUDE_allocation_methods_l3110_311000

/-- Represents the number of students --/
def num_students : ℕ := 5

/-- Represents the number of villages --/
def num_villages : ℕ := 3

/-- Represents the number of entities to be allocated (treating A and B as one entity) --/
def num_entities : ℕ := 4

/-- The number of ways to divide num_entities into num_villages non-empty groups --/
def ways_to_divide : ℕ := Nat.choose num_entities (num_villages - 1)

/-- The number of ways to arrange num_villages groups into num_villages villages --/
def ways_to_arrange : ℕ := Nat.factorial num_villages

/-- Theorem stating the total number of allocation methods --/
theorem allocation_methods :
  ways_to_divide * ways_to_arrange = 36 := by sorry

end NUMINAMATH_CALUDE_allocation_methods_l3110_311000


namespace NUMINAMATH_CALUDE_isosceles_triangle_l3110_311033

theorem isosceles_triangle (A B C : ℝ) (h : x^2 - x * Real.cos A * Real.cos B - (Real.cos (C/2))^2 = 0) 
  (root : 1^2 - 1 * Real.cos A * Real.cos B - (Real.cos (C/2))^2 = 0) : A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l3110_311033


namespace NUMINAMATH_CALUDE_sixth_group_52_implies_m_7_l3110_311050

/-- Represents a systematic sampling scheme with the given conditions -/
structure SystematicSampling where
  population : ℕ
  groups : ℕ
  sample_size : ℕ
  first_group_range : Set ℕ
  offset_rule : ℕ → ℕ → ℕ

/-- The specific systematic sampling scheme from the problem -/
def problem_sampling : SystematicSampling :=
  { population := 100
  , groups := 10
  , sample_size := 10
  , first_group_range := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  , offset_rule := λ m k => if m + k < 11 then (m + k - 1) % 10 else (m + k - 11) % 10
  }

/-- The theorem to be proved -/
theorem sixth_group_52_implies_m_7 (s : SystematicSampling) (h : s = problem_sampling) :
  ∃ (m : ℕ), m ∈ s.first_group_range ∧ s.offset_rule m 6 = 2 → m = 7 :=
sorry

end NUMINAMATH_CALUDE_sixth_group_52_implies_m_7_l3110_311050


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l3110_311065

theorem largest_solution_of_equation (x : ℝ) :
  (((15 * x^2 - 40 * x + 16) / (4 * x - 3)) + 3 * x = 7 * x + 2) →
  x ≤ -14 + Real.sqrt 218 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l3110_311065


namespace NUMINAMATH_CALUDE_tournament_committee_count_l3110_311029

/-- The number of teams in the league -/
def num_teams : ℕ := 4

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The size of the tournament committee -/
def committee_size : ℕ := 10

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 2

/-- The number of possible tournament committees -/
def num_committees : ℕ := 6146560

theorem tournament_committee_count :
  (num_teams : ℕ) *
  (Nat.choose team_size host_selection) *
  (Nat.choose team_size non_host_selection ^ (num_teams - 1)) =
  num_committees := by sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l3110_311029


namespace NUMINAMATH_CALUDE_optimal_allocation_l3110_311072

/-- Represents the advertising problem for a company --/
structure AdvertisingProblem where
  totalTime : ℝ
  totalBudget : ℝ
  rateA : ℝ
  rateB : ℝ
  revenueA : ℝ
  revenueB : ℝ

/-- Represents an advertising allocation --/
structure Allocation where
  timeA : ℝ
  timeB : ℝ

/-- Calculates the total revenue for a given allocation --/
def totalRevenue (p : AdvertisingProblem) (a : Allocation) : ℝ :=
  p.revenueA * a.timeA + p.revenueB * a.timeB

/-- Checks if an allocation is valid given the problem constraints --/
def isValidAllocation (p : AdvertisingProblem) (a : Allocation) : Prop :=
  a.timeA ≥ 0 ∧ a.timeB ≥ 0 ∧
  a.timeA + a.timeB ≤ p.totalTime ∧
  p.rateA * a.timeA + p.rateB * a.timeB ≤ p.totalBudget

/-- The main theorem stating that the given allocation maximizes revenue --/
theorem optimal_allocation (p : AdvertisingProblem) 
  (h1 : p.totalTime = 300)
  (h2 : p.totalBudget = 90000)
  (h3 : p.rateA = 500)
  (h4 : p.rateB = 200)
  (h5 : p.revenueA = 0.3)
  (h6 : p.revenueB = 0.2) :
  ∃ (a : Allocation),
    isValidAllocation p a ∧
    totalRevenue p a = 70 ∧
    ∀ (b : Allocation), isValidAllocation p b → totalRevenue p b ≤ totalRevenue p a :=
by sorry

end NUMINAMATH_CALUDE_optimal_allocation_l3110_311072


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3110_311082

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + 2*x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + 2*x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3110_311082


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3110_311058

theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ := (8, 16)
  let p₂ : ℝ × ℝ := (-2, -8)
  let midpoint := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  (midpoint.1 + midpoint.2 : ℝ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3110_311058


namespace NUMINAMATH_CALUDE_min_difference_of_sine_bounds_l3110_311013

open Real

theorem min_difference_of_sine_bounds (a b : ℝ) :
  (∀ x ∈ Set.Ioo 0 (π / 2), a * x < sin x ∧ sin x < b * x) →
  1 - 2 / π ≤ b - a :=
by sorry

end NUMINAMATH_CALUDE_min_difference_of_sine_bounds_l3110_311013


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3110_311071

theorem simplify_fraction_product : (320 : ℚ) / 18 * 9 / 144 * 4 / 5 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3110_311071


namespace NUMINAMATH_CALUDE_f_local_minimum_at_2_l3110_311025

def f (x : ℝ) := x^3 - 3*x^2 + 1

theorem f_local_minimum_at_2 :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 :=
sorry

end NUMINAMATH_CALUDE_f_local_minimum_at_2_l3110_311025


namespace NUMINAMATH_CALUDE_short_students_fraction_l3110_311053

/-- Given a class with the following properties:
  * There are 400 total students
  * There are 90 tall students
  * There are 150 students with average height
  Prove that the fraction of short students to the total number of students is 2/5 -/
theorem short_students_fraction (total : ℕ) (tall : ℕ) (average : ℕ) 
  (h_total : total = 400)
  (h_tall : tall = 90)
  (h_average : average = 150) :
  (total - tall - average : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_short_students_fraction_l3110_311053


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3110_311094

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  a = (2, 1) →
  a • b = 10 →
  ‖a + b‖ = 5 * Real.sqrt 2 →
  ‖b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3110_311094


namespace NUMINAMATH_CALUDE_smallest_ef_minus_de_l3110_311049

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  de : ℕ
  ef : ℕ
  fd : ℕ

/-- Checks if the given side lengths form a valid triangle -/
def isValidTriangle (t : Triangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.de + t.fd > t.ef ∧ t.ef + t.fd > t.de

/-- Theorem: The smallest possible value of EF - DE is 1 for a triangle DEF 
    with integer side lengths, perimeter 2010, and DE < EF ≤ FD -/
theorem smallest_ef_minus_de :
  ∀ t : Triangle,
    t.de + t.ef + t.fd = 2010 →
    t.de < t.ef →
    t.ef ≤ t.fd →
    isValidTriangle t →
    (∀ t' : Triangle,
      t'.de + t'.ef + t'.fd = 2010 →
      t'.de < t'.ef →
      t'.ef ≤ t'.fd →
      isValidTriangle t' →
      t'.ef - t'.de ≥ t.ef - t.de) →
    t.ef - t.de = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_ef_minus_de_l3110_311049


namespace NUMINAMATH_CALUDE_final_price_is_25_92_l3110_311038

/-- The final price observed by the buyer online given the commission rate,
    product cost, and desired profit rate. -/
def final_price (commission_rate : ℝ) (product_cost : ℝ) (profit_rate : ℝ) : ℝ :=
  let profit := product_cost * profit_rate
  let distributor_price := product_cost + profit
  let commission := distributor_price * commission_rate
  distributor_price + commission

/-- Theorem stating that the final price is $25.92 given the specified conditions -/
theorem final_price_is_25_92 :
  final_price 0.2 18 0.2 = 25.92 := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_25_92_l3110_311038


namespace NUMINAMATH_CALUDE_jacob_has_six_marshmallows_l3110_311009

/-- Calculates the number of marshmallows Jacob currently has -/
def jacobs_marshmallows (graham_crackers : ℕ) (more_marshmallows_needed : ℕ) : ℕ :=
  (graham_crackers / 2) - more_marshmallows_needed

/-- Proves that Jacob has 6 marshmallows given the problem conditions -/
theorem jacob_has_six_marshmallows :
  jacobs_marshmallows 48 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jacob_has_six_marshmallows_l3110_311009


namespace NUMINAMATH_CALUDE_strawberry_milk_probability_l3110_311017

theorem strawberry_milk_probability :
  let n : ℕ := 7  -- number of days
  let k : ℕ := 5  -- number of successful days
  let p : ℚ := 3/5  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 20412/78125 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_milk_probability_l3110_311017


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_not_exceeding_500_l3110_311052

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by
  sorry

theorem largest_n_not_exceeding_500 : 
  ∀ k : ℕ, (k * (k + 1) ≤ 1000 → k ≤ 31) ∧ 
           (31 * 32 ≤ 1000) ∧ 
           (32 * 33 > 1000) := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_not_exceeding_500_l3110_311052


namespace NUMINAMATH_CALUDE_not_div_sum_if_div_sum_squares_l3110_311035

theorem not_div_sum_if_div_sum_squares (a b : ℤ) : 
  7 ∣ (a^2 + b^2 + 1) → ¬(7 ∣ (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_not_div_sum_if_div_sum_squares_l3110_311035
