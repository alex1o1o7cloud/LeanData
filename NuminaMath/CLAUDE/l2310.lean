import Mathlib

namespace NUMINAMATH_CALUDE_single_plane_division_two_planes_division_l2310_231002

-- Define a type for space
structure Space :=
  (points : Set Point)

-- Define a type for plane
structure Plane :=
  (equation : Point → Prop)

-- Define a function to count the number of parts a set of planes divides space into
def countParts (space : Space) (planes : Set Plane) : ℕ :=
  sorry

-- Theorem for a single plane
theorem single_plane_division (space : Space) (plane : Plane) :
  countParts space {plane} = 2 :=
sorry

-- Theorem for two planes
theorem two_planes_division (space : Space) (plane1 plane2 : Plane) :
  countParts space {plane1, plane2} = 3 ∨ countParts space {plane1, plane2} = 4 :=
sorry

end NUMINAMATH_CALUDE_single_plane_division_two_planes_division_l2310_231002


namespace NUMINAMATH_CALUDE_exchange_problem_l2310_231032

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Represents the exchange problem and proves the sum of digits -/
theorem exchange_problem (d : ℕ) : 
  (11 * d : ℚ) / 8 - 70 = d → sumOfDigits d = 16 := by
  sorry

#eval sumOfDigits 187  -- Expected output: 16

end NUMINAMATH_CALUDE_exchange_problem_l2310_231032


namespace NUMINAMATH_CALUDE_max_min_difference_l2310_231013

theorem max_min_difference (a b c d : ℕ+) 
  (h1 : a + b = 20)
  (h2 : a + c = 24)
  (h3 : a + d = 22) : 
  (Nat.max (a + b + c + d) (a + b + c + d) : ℤ) - 
  (Nat.min (a + b + c + d) (a + b + c + d) : ℤ) = 36 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_l2310_231013


namespace NUMINAMATH_CALUDE_no_solution_condition_l2310_231074

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (m * x - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l2310_231074


namespace NUMINAMATH_CALUDE_function_range_theorem_l2310_231079

def f (x : ℝ) := x^2 - 2*x - 3

theorem function_range_theorem (m : ℝ) (h_m : m > 0) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ -3) ∧
  (∃ x ∈ Set.Icc 0 m, f x = -3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ -4) ∧
  (∃ x ∈ Set.Icc 0 m, f x = -4) →
  m ∈ Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_function_range_theorem_l2310_231079


namespace NUMINAMATH_CALUDE_area_of_triangle_AEB_main_theorem_l2310_231098

/-- Rectangle ABCD with given dimensions and points -/
structure Rectangle :=
  (A B C D F G E : ℝ × ℝ)
  (ab_length : ℝ)
  (bc_length : ℝ)
  (df_length : ℝ)
  (gc_length : ℝ)

/-- Conditions for the rectangle -/
def rectangle_conditions (rect : Rectangle) : Prop :=
  rect.ab_length = 7 ∧
  rect.bc_length = 4 ∧
  rect.df_length = 2 ∧
  rect.gc_length = 1 ∧
  rect.F.2 = rect.C.2 ∧
  rect.G.2 = rect.C.2 ∧
  rect.A.1 = rect.D.1 ∧
  rect.B.1 = rect.C.1 ∧
  rect.A.2 = rect.B.2 ∧
  rect.C.2 = rect.D.2 ∧
  (rect.E.1 - rect.A.1) / (rect.B.1 - rect.A.1) = (rect.F.1 - rect.D.1) / (rect.C.1 - rect.D.1) ∧
  (rect.E.1 - rect.B.1) / (rect.A.1 - rect.B.1) = (rect.G.1 - rect.C.1) / (rect.D.1 - rect.C.1)

/-- Theorem: The area of triangle AEB is 22.4 -/
theorem area_of_triangle_AEB (rect : Rectangle) 
  (h : rectangle_conditions rect) : ℝ :=
  22.4

/-- Main theorem: If the rectangle satisfies the given conditions, 
    then the area of triangle AEB is 22.4 -/
theorem main_theorem (rect : Rectangle) 
  (h : rectangle_conditions rect) : 
  area_of_triangle_AEB rect h = 22.4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_AEB_main_theorem_l2310_231098


namespace NUMINAMATH_CALUDE_bakers_cakes_l2310_231064

/-- Baker's pastry and cake problem -/
theorem bakers_cakes (pastries_made : ℕ) (cakes_sold : ℕ) (pastries_sold : ℕ) (cakes_left : ℕ) :
  pastries_made = 61 →
  cakes_sold = 108 →
  pastries_sold = 44 →
  cakes_left = 59 →
  cakes_sold + cakes_left = 167 := by
  sorry


end NUMINAMATH_CALUDE_bakers_cakes_l2310_231064


namespace NUMINAMATH_CALUDE_time_until_800_l2310_231026

def minutes_since_730 : ℕ := 16

def current_time : ℕ := 7 * 60 + 30 + minutes_since_730

def target_time : ℕ := 8 * 60

theorem time_until_800 : target_time - current_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_time_until_800_l2310_231026


namespace NUMINAMATH_CALUDE_range_of_x_when_m_is_2_range_of_m_when_q_necessary_not_sufficient_l2310_231022

-- Define propositions p and q
def p (x m : ℝ) : Prop := x^2 - 5*m*x + 6*m^2 < 0

def q (x : ℝ) : Prop := (x - 5) / (x - 1) < 0

-- Theorem 1
theorem range_of_x_when_m_is_2 (x : ℝ) :
  (p x 2 ∨ q x) → 1 < x ∧ x < 6 := by sorry

-- Theorem 2
theorem range_of_m_when_q_necessary_not_sufficient (m : ℝ) :
  (m > 0 ∧ ∀ x, p x m → q x) ∧ (∃ x, q x ∧ ¬p x m) →
  1/2 ≤ m ∧ m ≤ 5/3 := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_m_is_2_range_of_m_when_q_necessary_not_sufficient_l2310_231022


namespace NUMINAMATH_CALUDE_square_quotient_theorem_l2310_231092

theorem square_quotient_theorem (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_div : (a * b + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, (a^2 + b^2) / (a * b + 1) = k^2 := by
sorry

end NUMINAMATH_CALUDE_square_quotient_theorem_l2310_231092


namespace NUMINAMATH_CALUDE_total_cookies_count_l2310_231018

def cookies_eaten : ℕ := 4
def cookies_to_brother : ℕ := 6
def friends_count : ℕ := 3
def cookies_per_friend : ℕ := 2
def team_members : ℕ := 10
def first_team_member_cookies : ℕ := 2
def team_cookie_difference : ℕ := 2

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem total_cookies_count :
  cookies_eaten +
  cookies_to_brother +
  (friends_count * cookies_per_friend) +
  arithmetic_sum first_team_member_cookies team_cookie_difference team_members =
  126 := by sorry

end NUMINAMATH_CALUDE_total_cookies_count_l2310_231018


namespace NUMINAMATH_CALUDE_probability_four_ones_in_six_rolls_l2310_231017

theorem probability_four_ones_in_six_rolls (n : ℕ) (p : ℚ) : 
  n = 10 → p = 1 / n → 
  (Nat.choose 6 4 : ℚ) * p^4 * (1 - p)^2 = 243 / 200000 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_ones_in_six_rolls_l2310_231017


namespace NUMINAMATH_CALUDE_prob_five_odd_in_seven_rolls_prob_five_odd_in_seven_rolls_proof_l2310_231084

/-- The probability of getting exactly 5 odd numbers in 7 rolls of a fair 6-sided die -/
theorem prob_five_odd_in_seven_rolls : ℚ :=
  21 / 128

/-- A fair 6-sided die has equal probability for each outcome -/
axiom fair_die : ∀ (outcome : Fin 6), ℚ

/-- The probability of rolling an odd number on a fair 6-sided die is 1/2 -/
axiom prob_odd : (fair_die 1 + fair_die 3 + fair_die 5 : ℚ) = 1 / 2

/-- The rolls are independent -/
axiom independent_rolls : ∀ (n : ℕ), ℚ

/-- The probability of exactly k successes in n independent Bernoulli trials 
    with success probability p is given by the binomial probability formula -/
axiom binomial_probability : 
  ∀ (n k : ℕ) (p : ℚ), 
  0 ≤ p ∧ p ≤ 1 → 
  independent_rolls n = (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem prob_five_odd_in_seven_rolls_proof : 
  prob_five_odd_in_seven_rolls = independent_rolls 7 :=
sorry

end NUMINAMATH_CALUDE_prob_five_odd_in_seven_rolls_prob_five_odd_in_seven_rolls_proof_l2310_231084


namespace NUMINAMATH_CALUDE_combined_rocket_height_l2310_231059

def first_rocket_height : ℝ := 500

theorem combined_rocket_height :
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 := by
  sorry

end NUMINAMATH_CALUDE_combined_rocket_height_l2310_231059


namespace NUMINAMATH_CALUDE_overlap_percentage_l2310_231010

theorem overlap_percentage (square_side : ℝ) (rect_length rect_width : ℝ) :
  square_side = 18 →
  rect_length = 20 →
  rect_width = 18 →
  (rect_length * rect_width - 2 * square_side * square_side + (2 * square_side - rect_length) * rect_width) / (rect_length * rect_width) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_percentage_l2310_231010


namespace NUMINAMATH_CALUDE_square_sum_equals_21_l2310_231005

theorem square_sum_equals_21 (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -6) :
  x^2 + y^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_21_l2310_231005


namespace NUMINAMATH_CALUDE_combined_boys_avg_is_67_l2310_231062

/-- Represents a high school with average scores for boys, girls, and combined --/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two schools --/
structure CombinedSchools where
  school1 : School
  school2 : School
  combined_girls_avg : ℝ

/-- Calculates the combined average score for boys given two schools --/
def combined_boys_avg (schools : CombinedSchools) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the combined average score for boys is 67 --/
theorem combined_boys_avg_is_67 (schools : CombinedSchools) 
  (h1 : schools.school1 = ⟨65, 75, 68⟩)
  (h2 : schools.school2 = ⟨70, 85, 75⟩)
  (h3 : schools.combined_girls_avg = 80) :
  combined_boys_avg schools = 67 := by
  sorry

end NUMINAMATH_CALUDE_combined_boys_avg_is_67_l2310_231062


namespace NUMINAMATH_CALUDE_sum_factorials_mod_20_l2310_231075

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_mod_20 : sum_factorials 50 % 20 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_factorials_mod_20_l2310_231075


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2310_231061

open Complex

theorem complex_equation_solution :
  let z : ℂ := (1 + I^2 + 3*(1-I)) / (2+I)
  ∀ (a b : ℝ), z^2 + a*z + b = 1 + I → a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2310_231061


namespace NUMINAMATH_CALUDE_minimum_balls_to_draw_thirty_eight_sufficient_l2310_231011

/-- Represents the number of balls of each color in the bag -/
structure BagContents :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (other : ℕ)

/-- Represents the configuration of drawn balls -/
structure DrawnBalls :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (other : ℕ)

/-- Check if a given configuration of drawn balls satisfies the condition -/
def satisfiesCondition (drawn : DrawnBalls) : Prop :=
  drawn.red ≥ 10 ∨ drawn.blue ≥ 10 ∨ drawn.yellow ≥ 10

/-- Check if it's possible to draw a given configuration from the bag -/
def canDraw (bag : BagContents) (drawn : DrawnBalls) : Prop :=
  drawn.red ≤ bag.red ∧
  drawn.blue ≤ bag.blue ∧
  drawn.yellow ≤ bag.yellow ∧
  drawn.other ≤ bag.other ∧
  drawn.red + drawn.blue + drawn.yellow + drawn.other ≤ bag.red + bag.blue + bag.yellow + bag.other

theorem minimum_balls_to_draw (bag : BagContents)
  (h1 : bag.red = 20)
  (h2 : bag.blue = 20)
  (h3 : bag.yellow = 20)
  (h4 : bag.other = 10) :
  ∀ n : ℕ, n < 38 →
    ∃ drawn : DrawnBalls, canDraw bag drawn ∧ ¬satisfiesCondition drawn ∧ drawn.red + drawn.blue + drawn.yellow + drawn.other = n :=
by sorry

theorem thirty_eight_sufficient (bag : BagContents)
  (h1 : bag.red = 20)
  (h2 : bag.blue = 20)
  (h3 : bag.yellow = 20)
  (h4 : bag.other = 10) :
  ∀ drawn : DrawnBalls, canDraw bag drawn → drawn.red + drawn.blue + drawn.yellow + drawn.other = 38 →
    satisfiesCondition drawn :=
by sorry

end NUMINAMATH_CALUDE_minimum_balls_to_draw_thirty_eight_sufficient_l2310_231011


namespace NUMINAMATH_CALUDE_solution_set_equivalence_a_range_when_f_nonnegative_l2310_231035

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Part 1
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
sorry

-- Part 2
theorem a_range_when_f_nonnegative :
  ∀ a, (∀ x, f a (3-a) x ≥ 0) → a ∈ Set.Icc (-6) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_a_range_when_f_nonnegative_l2310_231035


namespace NUMINAMATH_CALUDE_equation_solution_l2310_231068

theorem equation_solution : 
  let x : ℚ := -43/8
  7 * (4 * x + 3) - 5 = -3 * (2 - 8 * x) + 1/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2310_231068


namespace NUMINAMATH_CALUDE_drum_sticks_per_show_l2310_231082

/-- Proves that the number of drum stick sets used per show for playing is 5 --/
theorem drum_sticks_per_show 
  (total_shows : ℕ) 
  (tossed_per_show : ℕ) 
  (total_sets : ℕ) 
  (h1 : total_shows = 30) 
  (h2 : tossed_per_show = 6) 
  (h3 : total_sets = 330) : 
  (total_sets - total_shows * tossed_per_show) / total_shows = 5 := by
  sorry

#check drum_sticks_per_show

end NUMINAMATH_CALUDE_drum_sticks_per_show_l2310_231082


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l2310_231045

theorem quadratic_solution_property : ∀ a b : ℝ, 
  a^2 + 8*a - 209 = 0 → 
  b^2 + 8*b - 209 = 0 → 
  a ≠ b →
  (a * b) / (a + b) = 209 / 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l2310_231045


namespace NUMINAMATH_CALUDE_min_value_theorem_l2310_231069

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 3) :
  x + 2 * y ≥ 3 + 6 * Real.sqrt 2 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 2) + 1 / (y₀ + 2) = 1 / 3 ∧
    x₀ + 2 * y₀ = 3 + 6 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2310_231069


namespace NUMINAMATH_CALUDE_total_bronze_needed_l2310_231070

/-- The weight of the first bell in pounds -/
def first_bell_weight : ℕ := 50

/-- The weight of the second bell in pounds -/
def second_bell_weight : ℕ := 2 * first_bell_weight

/-- The weight of the third bell in pounds -/
def third_bell_weight : ℕ := 4 * second_bell_weight

/-- The total weight of bronze needed for all three bells -/
def total_bronze_weight : ℕ := first_bell_weight + second_bell_weight + third_bell_weight

theorem total_bronze_needed :
  total_bronze_weight = 550 := by sorry

end NUMINAMATH_CALUDE_total_bronze_needed_l2310_231070


namespace NUMINAMATH_CALUDE_sum_of_ages_proof_l2310_231041

/-- Proves that the sum of ages of a mother and daughter is 70 years,
    given the daughter's age and the age difference. -/
theorem sum_of_ages_proof (daughter_age mother_daughter_diff : ℕ) : 
  daughter_age = 19 →
  mother_daughter_diff = 32 →
  daughter_age + (daughter_age + mother_daughter_diff) = 70 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_proof_l2310_231041


namespace NUMINAMATH_CALUDE_min_distance_squared_l2310_231065

theorem min_distance_squared (a b c d : ℝ) :
  (a - 2 * Real.exp a) / b = 1 →
  (2 - c) / d = 1 →
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (x - 2 * Real.exp x) / y = 1 →
    (2 - x) / y = 1 →
    (a - x)^2 + (b - y)^2 ≥ m ∧
    m = 8 :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l2310_231065


namespace NUMINAMATH_CALUDE_jordan_no_quiz_probability_l2310_231019

theorem jordan_no_quiz_probability (p_quiz : ℚ) (h : p_quiz = 5/9) :
  1 - p_quiz = 4/9 := by
sorry

end NUMINAMATH_CALUDE_jordan_no_quiz_probability_l2310_231019


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2310_231001

open Real

theorem solution_set_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : ∀ x, f x > f' x + 1)
  (h3 : ∀ x, f x - 2024 = -(f (-x) - 2024)) :
  {x : ℝ | f x - 2023 * exp x < 1} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2310_231001


namespace NUMINAMATH_CALUDE_painted_cube_problem_l2310_231046

/-- Represents a painted cube cut into smaller cubes -/
structure PaintedCube where
  /-- The number of small cubes along each edge of the large cube -/
  edge_count : ℕ
  /-- The number of small cubes with both brown and orange colors -/
  dual_color_count : ℕ

/-- Theorem stating the properties of the painted cube problem -/
theorem painted_cube_problem (cube : PaintedCube) 
  (h1 : cube.dual_color_count = 16) : 
  cube.edge_count = 4 ∧ cube.edge_count ^ 3 = 64 := by
  sorry

#check painted_cube_problem

end NUMINAMATH_CALUDE_painted_cube_problem_l2310_231046


namespace NUMINAMATH_CALUDE_markup_constant_l2310_231025

theorem markup_constant (C S : ℝ) (k : ℝ) (hk : k > 0) (hC : C > 0) (hS : S > 0) : 
  (S = C + k * S) → (k * S = 0.25 * C) → k = 1/5 := by
sorry

end NUMINAMATH_CALUDE_markup_constant_l2310_231025


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_y_negative_l2310_231053

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

theorem point_in_fourth_quadrant_y_negative (y : ℝ) :
  in_fourth_quadrant (Point2D.mk 5 y) → y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_y_negative_l2310_231053


namespace NUMINAMATH_CALUDE_odd_function_property_l2310_231020

-- Define an odd function f on ℝ
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h1 : f 1 = 2) 
  (h2 : f 2 = 3) : 
  f (f (-1)) = -3 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_property_l2310_231020


namespace NUMINAMATH_CALUDE_first_year_rate_is_12_percent_l2310_231093

/-- Profit rate in the first year -/
def first_year_rate : ℝ := 0.12

/-- Initial investment in millions of yuan -/
def initial_investment : ℝ := 5

/-- Profit rate increase in the second year -/
def rate_increase : ℝ := 0.08

/-- Net profit in the second year in millions of yuan -/
def second_year_profit : ℝ := 1.12

theorem first_year_rate_is_12_percent :
  (initial_investment + initial_investment * first_year_rate) * 
  (first_year_rate + rate_increase) = second_year_profit := by sorry

end NUMINAMATH_CALUDE_first_year_rate_is_12_percent_l2310_231093


namespace NUMINAMATH_CALUDE_regular_polygon_area_condition_l2310_231063

/-- A regular polygon with n sides inscribed in a circle of radius 2R has an area of 6R^2 if and only if n = 12 -/
theorem regular_polygon_area_condition (n : ℕ) (R : ℝ) (h1 : R > 0) :
  2 * n * R^2 * Real.sin (2 * Real.pi / n) = 6 * R^2 ↔ n = 12 := by
  sorry


end NUMINAMATH_CALUDE_regular_polygon_area_condition_l2310_231063


namespace NUMINAMATH_CALUDE_marnie_bracelets_l2310_231034

/-- The number of bracelets that can be made from a given number of beads -/
def bracelets_from_beads (total_beads : ℕ) (beads_per_bracelet : ℕ) : ℕ :=
  total_beads / beads_per_bracelet

/-- The total number of beads from multiple bags -/
def total_beads_from_bags (bags_of_50 : ℕ) (bags_of_100 : ℕ) : ℕ :=
  bags_of_50 * 50 + bags_of_100 * 100

theorem marnie_bracelets : 
  let bags_of_50 : ℕ := 5
  let bags_of_100 : ℕ := 2
  let beads_per_bracelet : ℕ := 50
  let total_beads := total_beads_from_bags bags_of_50 bags_of_100
  bracelets_from_beads total_beads beads_per_bracelet = 9 := by
  sorry

end NUMINAMATH_CALUDE_marnie_bracelets_l2310_231034


namespace NUMINAMATH_CALUDE_partner_c_investment_l2310_231029

/-- Represents the investment and profit structure of a business partnership --/
structure BusinessPartnership where
  capital_a : ℝ
  capital_b : ℝ
  capital_c : ℝ
  profit_b : ℝ
  profit_diff_ac : ℝ

/-- Theorem stating that given the conditions of the business partnership,
    the investment of partner c is 40000 --/
theorem partner_c_investment (bp : BusinessPartnership)
  (h1 : bp.capital_a = 8000)
  (h2 : bp.capital_b = 10000)
  (h3 : bp.profit_b = 3500)
  (h4 : bp.profit_diff_ac = 1399.9999999999998)
  : bp.capital_c = 40000 := by
  sorry


end NUMINAMATH_CALUDE_partner_c_investment_l2310_231029


namespace NUMINAMATH_CALUDE_original_number_is_332_l2310_231076

/-- Given a three-digit number abc, returns the sum of abc, acb, bca, bac, cab, and cba -/
def sum_permutations (a b c : Nat) : Nat :=
  100 * a + 10 * b + c +
  100 * a + 10 * c + b +
  100 * b + 10 * c + a +
  100 * b + 10 * a + c +
  100 * c + 10 * a + b +
  100 * c + 10 * b + a

/-- The original number abc satisfies the given conditions -/
theorem original_number_is_332 : 
  ∃ (a b c : Nat), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ 
    sum_permutations a b c = 4332 ∧
    100 * a + 10 * b + c = 332 :=
by sorry

end NUMINAMATH_CALUDE_original_number_is_332_l2310_231076


namespace NUMINAMATH_CALUDE_rotation_result_l2310_231071

/-- Represents the four positions around a circle -/
inductive Position
| Top
| Right
| Bottom
| Left

/-- Represents the four figures on the circle -/
inductive Figure
| Triangle
| SmallerCircle
| Square
| Pentagon

/-- Initial configuration of figures on the circle -/
def initial_config : Figure → Position
| Figure.Triangle => Position.Top
| Figure.SmallerCircle => Position.Right
| Figure.Square => Position.Bottom
| Figure.Pentagon => Position.Left

/-- Rotates a position by 150 degrees clockwise -/
def rotate_150_clockwise : Position → Position
| Position.Top => Position.Left
| Position.Right => Position.Top
| Position.Bottom => Position.Right
| Position.Left => Position.Bottom

/-- Final configuration after 150 degree clockwise rotation -/
def final_config : Figure → Position :=
  λ f => rotate_150_clockwise (initial_config f)

/-- Theorem stating the final positions after rotation -/
theorem rotation_result :
  final_config Figure.Triangle = Position.Left ∧
  final_config Figure.SmallerCircle = Position.Top ∧
  final_config Figure.Square = Position.Right ∧
  final_config Figure.Pentagon = Position.Bottom :=
sorry

end NUMINAMATH_CALUDE_rotation_result_l2310_231071


namespace NUMINAMATH_CALUDE_invisibility_elixir_combinations_l2310_231072

/-- The number of magical herbs available for the invisibility elixir. -/
def num_herbs : ℕ := 4

/-- The number of enchanted gems available for the invisibility elixir. -/
def num_gems : ℕ := 6

/-- The number of herb-gem combinations that cancel each other's magic. -/
def num_cancelling_combinations : ℕ := 3

/-- The number of successful combinations for the invisibility elixir. -/
def num_successful_combinations : ℕ := num_herbs * num_gems - num_cancelling_combinations

theorem invisibility_elixir_combinations :
  num_successful_combinations = 21 := by sorry

end NUMINAMATH_CALUDE_invisibility_elixir_combinations_l2310_231072


namespace NUMINAMATH_CALUDE_jump_rope_time_difference_l2310_231008

-- Define the jump rope times for each person
def cindy_time : ℕ := 12
def betsy_time : ℕ := cindy_time / 2
def tina_time : ℕ := betsy_time * 3

-- Theorem to prove
theorem jump_rope_time_difference : tina_time - cindy_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_jump_rope_time_difference_l2310_231008


namespace NUMINAMATH_CALUDE_chessboard_uniquely_determined_l2310_231089

/-- Represents a cell on the chessboard --/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the chessboard configuration --/
def Chessboard := Cell → Fin 64

/-- Represents a 2-cell rectangle on the chessboard --/
structure Rectangle :=
  (cell1 : Cell)
  (cell2 : Cell)

/-- Function to get the sum of numbers in a 2-cell rectangle --/
def getRectangleSum (board : Chessboard) (rect : Rectangle) : Nat :=
  (board rect.cell1).val + 1 + (board rect.cell2).val + 1

/-- Predicate to check if two cells are on the same diagonal --/
def onSameDiagonal (c1 c2 : Cell) : Prop :=
  (c1.row.val + c1.col.val = c2.row.val + c2.col.val) ∨
  (c1.row.val - c1.col.val = c2.row.val - c2.col.val)

/-- The main theorem --/
theorem chessboard_uniquely_determined
  (board : Chessboard)
  (h1 : ∃ c1 c2 : Cell, (board c1 = 0) ∧ (board c2 = 63) ∧ onSameDiagonal c1 c2)
  (h2 : ∀ rect : Rectangle, ∃ s : Nat, getRectangleSum board rect = s) :
  ∀ c : Cell, ∃! n : Fin 64, board c = n :=
sorry

end NUMINAMATH_CALUDE_chessboard_uniquely_determined_l2310_231089


namespace NUMINAMATH_CALUDE_median_after_removal_l2310_231096

def room_sequence : List Nat := List.range 26

def remaining_rooms (seq : List Nat) : List Nat :=
  seq.filter (fun n => n ≠ 15 ∧ n ≠ 20 ∧ n ≠ 0)

theorem median_after_removal (seq : List Nat) (h : seq = room_sequence) :
  (remaining_rooms seq).get? ((remaining_rooms seq).length / 2) = some 12 := by
  sorry

end NUMINAMATH_CALUDE_median_after_removal_l2310_231096


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l2310_231015

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l2310_231015


namespace NUMINAMATH_CALUDE_christine_money_l2310_231054

theorem christine_money (total : ℕ) (difference : ℕ) : 
  total = 50 → difference = 30 → ∃ (christine siri : ℕ), 
    christine = siri + difference ∧ 
    christine + siri = total ∧ 
    christine = 40 := by sorry

end NUMINAMATH_CALUDE_christine_money_l2310_231054


namespace NUMINAMATH_CALUDE_largest_difference_l2310_231014

def A : ℕ := 3 * 2003^2002
def B : ℕ := 2003^2002
def C : ℕ := 2002 * 2003^2001
def D : ℕ := 3 * 2003^2001
def E : ℕ := 2003^2001
def F : ℕ := 2003^2000

theorem largest_difference (A B C D E F : ℕ) 
  (hA : A = 3 * 2003^2002)
  (hB : B = 2003^2002)
  (hC : C = 2002 * 2003^2001)
  (hD : D = 3 * 2003^2001)
  (hE : E = 2003^2001)
  (hF : F = 2003^2000) :
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_l2310_231014


namespace NUMINAMATH_CALUDE_system_solution_unique_l2310_231031

theorem system_solution_unique :
  ∃! (x y : ℚ), x = 2 * y ∧ 2 * x - y = 5 :=
by
  -- The unique solution is x = 10/3 and y = 5/3
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2310_231031


namespace NUMINAMATH_CALUDE_ceiling_negative_example_l2310_231036

theorem ceiling_negative_example : ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_example_l2310_231036


namespace NUMINAMATH_CALUDE_hexagon_rounding_exists_l2310_231004

/-- Represents a hexagon with numbers on its vertices and sums on its sides. -/
structure Hexagon where
  -- Vertex numbers
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  a₄ : ℝ
  a₅ : ℝ
  a₆ : ℝ
  -- Side sums
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  s₄ : ℝ
  s₅ : ℝ
  s₆ : ℝ
  -- Ensure side sums match vertex sums
  h₁ : s₁ = a₁ + a₂
  h₂ : s₂ = a₂ + a₃
  h₃ : s₃ = a₃ + a₄
  h₄ : s₄ = a₄ + a₅
  h₅ : s₅ = a₅ + a₆
  h₆ : s₆ = a₆ + a₁

/-- Represents a rounding strategy for the hexagon. -/
structure RoundedHexagon where
  -- Rounded vertex numbers
  r₁ : ℤ
  r₂ : ℤ
  r₃ : ℤ
  r₄ : ℤ
  r₅ : ℤ
  r₆ : ℤ
  -- Rounded side sums
  t₁ : ℤ
  t₂ : ℤ
  t₃ : ℤ
  t₄ : ℤ
  t₅ : ℤ
  t₆ : ℤ

/-- Theorem: For any hexagon, there exists a rounding strategy that maintains the sum property. -/
theorem hexagon_rounding_exists (h : Hexagon) : 
  ∃ (r : RoundedHexagon), 
    (r.t₁ = r.r₁ + r.r₂) ∧
    (r.t₂ = r.r₂ + r.r₃) ∧
    (r.t₃ = r.r₃ + r.r₄) ∧
    (r.t₄ = r.r₄ + r.r₅) ∧
    (r.t₅ = r.r₅ + r.r₆) ∧
    (r.t₆ = r.r₆ + r.r₁) :=
  sorry

end NUMINAMATH_CALUDE_hexagon_rounding_exists_l2310_231004


namespace NUMINAMATH_CALUDE_sum_of_digits_of_1996_digit_multiple_of_9_l2310_231023

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a 1996-digit integer -/
def is1996Digit (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_1996_digit_multiple_of_9 (n : ℕ) 
  (h1 : is1996Digit n) 
  (h2 : n % 9 = 0) : 
  let p := sumOfDigits n
  let q := sumOfDigits p
  let r := sumOfDigits q
  r = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_1996_digit_multiple_of_9_l2310_231023


namespace NUMINAMATH_CALUDE_min_cards_to_verify_statement_l2310_231078

/-- Represents the side of a card -/
inductive CardSide
| Color (c : String)
| Smiley (happy : Bool)

/-- Represents a card with two sides -/
structure Card :=
  (side1 side2 : CardSide)

/-- The statement to be verified -/
def statement (c : Card) : Prop :=
  match c.side1, c.side2 with
  | CardSide.Smiley true, CardSide.Color "yellow" => True
  | CardSide.Color "yellow", CardSide.Smiley true => True
  | _, _ => False

/-- The set of cards given in the problem -/
def cards : Finset Card := sorry

/-- The minimum number of cards to turn over -/
def min_cards_to_turn : ℕ := sorry

theorem min_cards_to_verify_statement :
  min_cards_to_turn = 2 ∧
  ∃ (c1 c2 : Card), c1 ∈ cards ∧ c2 ∈ cards ∧ c1 ≠ c2 ∧
    (∀ (c : Card), c ∈ cards → statement c ↔ (c = c1 ∨ c = c2)) :=
sorry

end NUMINAMATH_CALUDE_min_cards_to_verify_statement_l2310_231078


namespace NUMINAMATH_CALUDE_cat_grooming_time_l2310_231049

/-- Calculates the total grooming time for a cat -/
def total_grooming_time (
  claws_per_foot : ℕ
  ) (
  feet : ℕ
  ) (
  clip_time : ℕ
  ) (
  ear_clean_time : ℕ
  ) (
  shampoo_time : ℕ
  ) : ℕ :=
  let total_claws := claws_per_foot * feet
  let total_clip_time := total_claws * clip_time
  let total_ear_clean_time := 2 * ear_clean_time
  let total_shampoo_time := shampoo_time * 60
  total_clip_time + total_ear_clean_time + total_shampoo_time

theorem cat_grooming_time :
  total_grooming_time 4 4 10 90 5 = 640 := by
  sorry

end NUMINAMATH_CALUDE_cat_grooming_time_l2310_231049


namespace NUMINAMATH_CALUDE_constant_term_proof_l2310_231039

theorem constant_term_proof (a k n : ℤ) :
  (∀ x, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) →
  a - n + k = 7 →
  (3 * 0 + 2) * (2 * 0 - 3) = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_proof_l2310_231039


namespace NUMINAMATH_CALUDE_product_digit_sum_base7_l2310_231091

/-- Converts a base 7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number in base 7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The problem statement --/
theorem product_digit_sum_base7 :
  let a := 35
  let b := 42
  sumOfDigitsBase7 (toBase7 (toDecimal a * toDecimal b)) = 18 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_base7_l2310_231091


namespace NUMINAMATH_CALUDE_dave_tickets_l2310_231099

theorem dave_tickets (won lost used : ℕ) (h1 : won = 14) (h2 : lost = 2) (h3 : used = 10) :
  won - lost - used = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l2310_231099


namespace NUMINAMATH_CALUDE_bert_stamp_collection_l2310_231077

theorem bert_stamp_collection (stamps_bought : ℕ) (stamps_before : ℕ) : 
  stamps_bought = 300 →
  stamps_before = stamps_bought / 2 →
  stamps_before + stamps_bought = 450 := by
sorry

end NUMINAMATH_CALUDE_bert_stamp_collection_l2310_231077


namespace NUMINAMATH_CALUDE_even_fraction_integers_l2310_231007

theorem even_fraction_integers (a : ℤ) : 
  (∃ k : ℤ, a / (1011 - a) = 2 * k) ↔ 
  a ∈ ({1010, 1012, 1008, 1014, 674, 1348, 0, 2022} : Set ℤ) := by
sorry

end NUMINAMATH_CALUDE_even_fraction_integers_l2310_231007


namespace NUMINAMATH_CALUDE_network_connections_l2310_231028

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l2310_231028


namespace NUMINAMATH_CALUDE_no_tangent_point_largest_integer_a_l2310_231066

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (a / 2) * x^2

theorem no_tangent_point (a : ℝ) : ¬∃ x, f a x = 0 ∧ (deriv (f a)) x = 0 := sorry

theorem largest_integer_a :
  ∃ a : ℤ, (∀ x₁ x₂ : ℝ, x₂ > 0 → f a (x₁ + x₂) - f a (x₁ - x₂) > -2 * x₂) ∧
  (∀ b : ℤ, b > a → ∃ x₁ x₂ : ℝ, x₂ > 0 ∧ f b (x₁ + x₂) - f b (x₁ - x₂) ≤ -2 * x₂) ∧
  a = 3 := sorry

end NUMINAMATH_CALUDE_no_tangent_point_largest_integer_a_l2310_231066


namespace NUMINAMATH_CALUDE_grapes_purchased_l2310_231073

/-- Proves that the number of kg of grapes purchased is 8 -/
theorem grapes_purchased (grape_price : ℕ) (mango_price : ℕ) (mango_kg : ℕ) (total_paid : ℕ) : 
  grape_price = 70 → 
  mango_price = 55 → 
  mango_kg = 9 → 
  total_paid = 1055 → 
  ∃ (grape_kg : ℕ), grape_kg * grape_price + mango_kg * mango_price = total_paid ∧ grape_kg = 8 :=
by sorry

end NUMINAMATH_CALUDE_grapes_purchased_l2310_231073


namespace NUMINAMATH_CALUDE_det_dilation_matrix_5_l2310_231037

/-- A dilation matrix with scale factor k -/
def dilationMatrix (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => k)

/-- Theorem: The determinant of a 3x3 dilation matrix with scale factor 5 is 125 -/
theorem det_dilation_matrix_5 :
  let D := dilationMatrix 5
  Matrix.det D = 125 := by sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_5_l2310_231037


namespace NUMINAMATH_CALUDE_problem_solution_l2310_231058

theorem problem_solution (a b : ℚ) 
  (h1 : 5 + a = 7 - b) 
  (h2 : 3 + b = 8 + a) : 
  4 - a = 11/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2310_231058


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l2310_231003

theorem sum_of_roots_cubic_equation : 
  let f : ℝ → ℝ := fun x ↦ 3 * x^3 - 9 * x^2 - 72 * x + 6
  ∃ r p q : ℝ, (∀ x : ℝ, f x = 0 ↔ x = r ∨ x = p ∨ x = q) ∧ r + p + q = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l2310_231003


namespace NUMINAMATH_CALUDE_tangent_parallel_and_inequality_l2310_231090

/-- The function f(x) = x^3 - ax^2 + 3x + b -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + 3*x + b

/-- The derivative of f(x) -/
def f_deriv (a x : ℝ) : ℝ := 3*x^2 - 2*a*x + 3

theorem tangent_parallel_and_inequality (a b : ℝ) :
  (f_deriv a 1 = 0) →
  (∀ x ∈ Set.Icc (-1) 4, f a b x > f_deriv a x) →
  (a = 3 ∧ b > 19) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_and_inequality_l2310_231090


namespace NUMINAMATH_CALUDE_two_lines_properties_l2310_231033

/-- Two lines l₁ and l₂ in the xy-plane -/
structure TwoLines (m n : ℝ) :=
  (l₁ : ℝ → ℝ → Prop)
  (l₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ m * x + 8 * y + n = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 2 * x + m * y - 1 = 0)

/-- The lines intersect at point P(m, -1) -/
def intersect_at_P (l : TwoLines m n) : Prop :=
  l.l₁ m (-1) ∧ l.l₂ m (-1)

/-- The lines are parallel -/
def parallel (l : TwoLines m n) : Prop :=
  m / 2 = 8 / m ∧ m / 2 ≠ n / (-1)

/-- The lines are perpendicular -/
def perpendicular (l : TwoLines m n) : Prop :=
  m = 0 ∨ (m ≠ 0 ∧ (-m / 8) * (1 / m) = -1)

/-- Main theorem about the properties of the two lines -/
theorem two_lines_properties (m n : ℝ) (l : TwoLines m n) :
  (intersect_at_P l → m = 1 ∧ n = 7) ∧
  (parallel l → (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2)) ∧
  (perpendicular l → m = 0) :=
sorry

end NUMINAMATH_CALUDE_two_lines_properties_l2310_231033


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2310_231050

theorem quadratic_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 3 * x^2 + 8 * x - 16 = 0 :=
by
  use 4/3
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2310_231050


namespace NUMINAMATH_CALUDE_equation_solution_sum_l2310_231024

theorem equation_solution_sum (a : ℝ) (h : a ≥ 1) :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt (a - Real.sqrt (a + x)) = x ∧
  (∀ y : ℝ, y ≥ 0 ∧ Real.sqrt (a - Real.sqrt (a + y)) = y → y = x) ∧
  x = (Real.sqrt (4 * a - 3) - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_sum_l2310_231024


namespace NUMINAMATH_CALUDE_scout_weekend_earnings_l2310_231030

def base_pay : ℝ := 10.00
def tip_per_customer : ℝ := 5.00
def saturday_hours : ℝ := 4
def saturday_customers : ℕ := 5
def sunday_hours : ℝ := 5
def sunday_customers : ℕ := 8

theorem scout_weekend_earnings :
  let saturday_earnings := base_pay * saturday_hours + tip_per_customer * saturday_customers
  let sunday_earnings := base_pay * sunday_hours + tip_per_customer * sunday_customers
  saturday_earnings + sunday_earnings = 155.00 := by
  sorry

end NUMINAMATH_CALUDE_scout_weekend_earnings_l2310_231030


namespace NUMINAMATH_CALUDE_disprove_statement_l2310_231085

theorem disprove_statement : ∃ (a b c : ℤ), a > b ∧ b > c ∧ ¬(a + b > c) :=
  sorry

end NUMINAMATH_CALUDE_disprove_statement_l2310_231085


namespace NUMINAMATH_CALUDE_kikis_money_l2310_231042

theorem kikis_money (num_scarves : ℕ) (scarf_price : ℚ) (hat_ratio : ℚ) (hat_percentage : ℚ) :
  num_scarves = 18 →
  scarf_price = 2 →
  hat_ratio = 2 →
  hat_percentage = 60 / 100 →
  ∃ (total_money : ℚ), 
    total_money = 90 ∧
    (num_scarves : ℚ) * scarf_price = (1 - hat_percentage) * total_money ∧
    (hat_ratio * num_scarves : ℚ) * (hat_percentage * total_money / (hat_ratio * num_scarves)) = hat_percentage * total_money :=
by sorry

end NUMINAMATH_CALUDE_kikis_money_l2310_231042


namespace NUMINAMATH_CALUDE_roots_always_real_l2310_231048

/-- Given real numbers a, b, and c, the discriminant of the quadratic equation
    resulting from 1/(x+a) + 1/(x+b) + 1/(x+c) = 3/x is non-negative. -/
theorem roots_always_real (a b c : ℝ) : 
  2 * (a^2 * (b - c)^2 + b^2 * (c - a)^2 + c^2 * (a - b)^2) ≥ 0 := by
  sorry

#check roots_always_real

end NUMINAMATH_CALUDE_roots_always_real_l2310_231048


namespace NUMINAMATH_CALUDE_triple_cheese_ratio_undetermined_l2310_231016

/-- Represents the types of pizzas available --/
inductive PizzaType
| TripleCheese
| MeatLovers

/-- Represents the pricing structure for pizzas --/
structure PizzaPricing where
  standardPrice : ℕ
  meatLoversOffer : ℕ → ℕ  -- Function that takes number of pizzas and returns number to pay for
  tripleCheesePricing : ℕ → ℕ  -- Function for triple cheese pizzas (unknown specifics)

/-- Represents the order details --/
structure Order where
  tripleCheeseCount : ℕ
  meatLoversCount : ℕ

/-- Calculates the total cost of an order --/
def calculateTotalCost (pricing : PizzaPricing) (order : Order) : ℕ :=
  pricing.tripleCheesePricing order.tripleCheeseCount * pricing.standardPrice +
  pricing.meatLoversOffer order.meatLoversCount * pricing.standardPrice

/-- Theorem stating that the ratio for triple cheese pizzas cannot be determined --/
theorem triple_cheese_ratio_undetermined (pricing : PizzaPricing) (order : Order) :
  pricing.standardPrice = 5 ∧
  order.meatLoversCount = 9 ∧
  calculateTotalCost pricing order = 55 →
  ¬ ∃ (r : ℚ), r > 0 ∧ r < 1 ∧ ∀ (n : ℕ), pricing.tripleCheesePricing n = n - ⌊n * r⌋ :=
by sorry

end NUMINAMATH_CALUDE_triple_cheese_ratio_undetermined_l2310_231016


namespace NUMINAMATH_CALUDE_inverse_of_congruent_area_equal_l2310_231094

-- Define the types for triangles and areas
def Triangle : Type := sorry
def Area : Type := sorry

-- Define the congruence relation for triangles
def congruent : Triangle → Triangle → Prop := sorry

-- Define the equality of areas
def area_equal : Area → Area → Prop := sorry

-- Define the area function for triangles
def triangle_area : Triangle → Area := sorry

-- Define the original proposition
def original_proposition : Prop :=
  ∀ (t1 t2 : Triangle), congruent t1 t2 → area_equal (triangle_area t1) (triangle_area t2)

-- Define the inverse proposition
def inverse_proposition : Prop :=
  ∀ (t1 t2 : Triangle), area_equal (triangle_area t1) (triangle_area t2) → congruent t1 t2

-- Theorem stating that the inverse_proposition is the correct inverse of the original_proposition
theorem inverse_of_congruent_area_equal :
  inverse_proposition = (¬original_proposition → ¬(∀ (t1 t2 : Triangle), congruent t1 t2)) := by sorry

end NUMINAMATH_CALUDE_inverse_of_congruent_area_equal_l2310_231094


namespace NUMINAMATH_CALUDE_max_value_g_range_of_a_inequality_for_f_l2310_231051

noncomputable section

def f (x : ℝ) : ℝ := Real.log x

def g (x : ℝ) : ℝ := f (x + 1) - x

theorem max_value_g :
  ∀ x > -1, g x ≤ 0 ∧ ∃ x₀ > -1, g x₀ = 0 :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≤ a * x ∧ a * x ≤ x^2 + 1) →
  (1 / Real.exp 1 ≤ a ∧ a ≤ 2) :=
sorry

theorem inequality_for_f (x₁ x₂ : ℝ) (h : x₁ > x₂ ∧ x₂ > 0) :
  (f x₁ - f x₂) / (x₁ - x₂) > (2 * x₂) / (x₁^2 + x₂^2) :=
sorry

end NUMINAMATH_CALUDE_max_value_g_range_of_a_inequality_for_f_l2310_231051


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2310_231044

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x^2 + a*x + 4 < 0) → (a < -4 ∨ a > 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2310_231044


namespace NUMINAMATH_CALUDE_quadratic_function_unique_form_l2310_231006

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_unique_form 
  (f : ℝ → ℝ) 
  (h_vertex : f (-1) = 4 ∧ ∀ x, f x ≤ f (-1)) 
  (h_point : f 2 = -5) :
  ∃ a b c : ℝ, f = quadratic_function a b c ∧ a = -1 ∧ b = -2 ∧ c = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_form_l2310_231006


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l2310_231021

/-- The function f satisfying the given conditions -/
noncomputable def f (x : ℝ) : ℝ := -5 * (4^x - 5^x)

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions :
  (f 1 = 5) ∧
  (∀ x y : ℝ, f (x + y) = 4^y * f x + 5^x * f y) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l2310_231021


namespace NUMINAMATH_CALUDE_logarithm_equality_l2310_231097

theorem logarithm_equality (a b c x : ℝ) (p q r y : ℝ) :
  a > 0 → b > 0 → c > 0 → x > 0 → x ≠ 1 →
  (∀ (base : ℝ), base > 1 →
    (Real.log a / p = Real.log b / q) ∧
    (Real.log b / q = Real.log c / r) ∧
    (Real.log c / r = Real.log x)) →
  b^3 / (a^2 * c) = x^y →
  y = 3*q - 2*p - r :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equality_l2310_231097


namespace NUMINAMATH_CALUDE_rectangle_circle_ratio_l2310_231086

theorem rectangle_circle_ratio (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 1225)
  (h2 : rectangle_area = 140)
  (h3 : rectangle_breadth = 10) :
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_ratio_l2310_231086


namespace NUMINAMATH_CALUDE_frisbee_price_problem_l2310_231060

/-- The price of the other frisbees in a sporting goods store -/
theorem frisbee_price_problem :
  ∀ (F₃ F_x x : ℕ),
    F₃ + F_x = 64 →
    3 * F₃ + x * F_x = 200 →
    F_x ≥ 8 →
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_frisbee_price_problem_l2310_231060


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_transform_l2310_231057

-- Define the polynomial
def p (x : ℝ) : ℝ := 15 * x^3 - 35 * x^2 + 20 * x - 2

-- Theorem statement
theorem root_sum_reciprocal_transform (a b c : ℝ) :
  p a = 0 → p b = 0 → p c = 0 →  -- a, b, c are roots of p
  a ≠ b → b ≠ c → a ≠ c →        -- roots are distinct
  0 < a → a < 1 →                -- 0 < a < 1
  0 < b → b < 1 →                -- 0 < b < 1
  0 < c → c < 1 →                -- 0 < c < 1
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_transform_l2310_231057


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_zero_l2310_231009

theorem complex_arithmetic_expression_equals_zero :
  -6 * (1/3 - 1/2) - 3^2 / (-12) - |-7/4| = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_zero_l2310_231009


namespace NUMINAMATH_CALUDE_power_calculation_l2310_231027

theorem power_calculation : (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2310_231027


namespace NUMINAMATH_CALUDE_equation_solution_l2310_231087

theorem equation_solution : ∃ x : ℝ, 4 * x - 2 = 2 * (x + 2) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2310_231087


namespace NUMINAMATH_CALUDE_inequality_solutions_l2310_231083

theorem inequality_solutions : 
  ∃! (s : Finset Int), 
    (∀ y ∈ s, (2 * y ≤ -y + 4 ∧ 5 * y ≥ -10 ∧ 3 * y ≤ -2 * y + 20)) ∧ 
    s.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l2310_231083


namespace NUMINAMATH_CALUDE_quadratic_decreasing_condition_l2310_231052

/-- A quadratic function f(x) = x² - mx + c -/
def f (m c x : ℝ) : ℝ := x^2 - m*x + c

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 2*x - m

theorem quadratic_decreasing_condition (m c : ℝ) :
  (∀ x < 1, (f' m x) < 0) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_condition_l2310_231052


namespace NUMINAMATH_CALUDE_exponent_division_l2310_231080

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2310_231080


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l2310_231095

theorem a_gt_one_sufficient_not_necessary_for_a_squared_gt_one :
  (∀ a : ℝ, a > 1 → a^2 > 1) ∧
  (∃ a : ℝ, a^2 > 1 ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l2310_231095


namespace NUMINAMATH_CALUDE_hikers_meeting_point_l2310_231067

/-- Represents the distance between two hikers at any given time -/
structure HikerDistance where
  total : ℝ := 100
  from_a : ℝ
  from_b : ℝ

/-- Calculates the distance traveled by hiker A in t hours -/
def distance_a (t : ℝ) : ℝ := 5 * t

/-- Calculates the distance traveled by hiker B in t hours -/
def distance_b (t : ℝ) : ℝ := t * (4 + 0.125 * (t - 1))

/-- Represents the meeting point of the two hikers -/
def meeting_point (t : ℝ) : HikerDistance :=
  { total := 100
  , from_a := distance_a t
  , from_b := distance_b t }

/-- The time at which the hikers meet -/
def meeting_time : ℕ := 10

theorem hikers_meeting_point :
  let mp := meeting_point meeting_time
  mp.from_b - mp.from_a = 2.5 := by sorry

end NUMINAMATH_CALUDE_hikers_meeting_point_l2310_231067


namespace NUMINAMATH_CALUDE_summer_sales_is_seven_l2310_231088

/-- The number of million hamburgers sold in each season --/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The total annual sales of hamburgers in millions --/
def total_sales (s : SeasonalSales) : ℝ :=
  s.spring + s.summer + s.fall + s.winter

/-- Theorem stating that the number of million hamburgers sold in the summer is 7 --/
theorem summer_sales_is_seven (s : SeasonalSales) 
  (h1 : s.fall = 0.2 * total_sales s)
  (h2 : s.fall = 3)
  (h3 : s.spring = 2)
  (h4 : s.winter = 3) : 
  s.summer = 7 := by
  sorry

end NUMINAMATH_CALUDE_summer_sales_is_seven_l2310_231088


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_equality_l2310_231040

theorem min_value_expression (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (hx₁ : x₁ ≥ 0) (hx₂ : x₂ ≥ 0) (hx₃ : x₃ ≥ 0) 
  (hy₁ : y₁ ≥ 0) (hy₂ : y₂ ≥ 0) (hy₃ : y₃ ≥ 0) : 
  Real.sqrt ((2018 - y₁ - y₂ - y₃)^2 + x₃^2) + 
  Real.sqrt (y₃^2 + x₂^2) + 
  Real.sqrt (y₂^2 + x₁^2) + 
  Real.sqrt (y₁^2 + (x₁ + x₂ + x₃)^2) ≥ 2018 :=
by sorry

theorem min_value_equality (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) : 
  (x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ y₁ = 0 ∧ y₂ = 0 ∧ y₃ = 0) → 
  Real.sqrt ((2018 - y₁ - y₂ - y₃)^2 + x₃^2) + 
  Real.sqrt (y₃^2 + x₂^2) + 
  Real.sqrt (y₂^2 + x₁^2) + 
  Real.sqrt (y₁^2 + (x₁ + x₂ + x₃)^2) = 2018 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_equality_l2310_231040


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_one_l2310_231047

theorem sum_of_roots_equals_one (x : ℝ) :
  (x + 3) * (x - 4) = 24 → ∃ y z : ℝ, y + z = 1 ∧ (y + 3) * (y - 4) = 24 ∧ (z + 3) * (z - 4) = 24 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_one_l2310_231047


namespace NUMINAMATH_CALUDE_ice_skating_falls_ratio_l2310_231038

/-- Given the number of falls for Steven, Stephanie, and Sonya while ice skating,
    prove that the ratio of Sonya's falls to half of Stephanie's falls is 3:4. -/
theorem ice_skating_falls_ratio 
  (steven_falls : ℕ) 
  (stephanie_falls : ℕ) 
  (sonya_falls : ℕ) 
  (h1 : steven_falls = 3)
  (h2 : stephanie_falls = steven_falls + 13)
  (h3 : sonya_falls = 6) :
  (sonya_falls : ℚ) / ((stephanie_falls : ℚ) / 2) = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_ice_skating_falls_ratio_l2310_231038


namespace NUMINAMATH_CALUDE_coin_problem_l2310_231056

theorem coin_problem (n d h : ℕ) : 
  n + d + h = 150 →
  5*n + 10*d + 50*h = 1250 →
  ∃ (d_min d_max : ℕ), 
    (∃ (n' h' : ℕ), n' + d_min + h' = 150 ∧ 5*n' + 10*d_min + 50*h' = 1250) ∧
    (∃ (n'' h'' : ℕ), n'' + d_max + h'' = 150 ∧ 5*n'' + 10*d_max + 50*h'' = 1250) ∧
    d_max - d_min = 99 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l2310_231056


namespace NUMINAMATH_CALUDE_ron_book_picks_l2310_231012

/-- Represents a book club with its properties --/
structure BookClub where
  members : ℕ
  weekly_meetings : ℕ
  holiday_breaks : ℕ
  guest_picks : ℕ
  leap_year_extra_meeting : ℕ

/-- Calculates the number of times a member gets to pick a book --/
def picks_per_member (club : BookClub) (is_leap_year : Bool) : ℕ :=
  let total_meetings := club.weekly_meetings - club.holiday_breaks + (if is_leap_year then club.leap_year_extra_meeting else 0)
  let member_picks := total_meetings - club.guest_picks - (if is_leap_year then 1 else 0)
  member_picks / club.members

/-- Theorem stating that Ron gets to pick 3 books in both leap and non-leap years --/
theorem ron_book_picks (club : BookClub) 
    (h1 : club.members = 13)
    (h2 : club.weekly_meetings = 52)
    (h3 : club.holiday_breaks = 5)
    (h4 : club.guest_picks = 6)
    (h5 : club.leap_year_extra_meeting = 1) : 
    picks_per_member club false = 3 ∧ picks_per_member club true = 3 := by
  sorry

end NUMINAMATH_CALUDE_ron_book_picks_l2310_231012


namespace NUMINAMATH_CALUDE_trapezoid_area_l2310_231055

theorem trapezoid_area (c : ℝ) (hc : c > 0) :
  let b := Real.sqrt c
  let shorter_base := b - 3
  let altitude := b
  let longer_base := b + 3
  let area := ((shorter_base + longer_base) / 2) * altitude
  area = c := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2310_231055


namespace NUMINAMATH_CALUDE_soccer_ball_donation_l2310_231043

theorem soccer_ball_donation :
  let num_schools : ℕ := 2
  let elementary_classes_per_school : ℕ := 4
  let middle_classes_per_school : ℕ := 5
  let balls_per_class : ℕ := 5
  let total_classes : ℕ := num_schools * (elementary_classes_per_school + middle_classes_per_school)
  let total_balls : ℕ := total_classes * balls_per_class
  total_balls = 90 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_donation_l2310_231043


namespace NUMINAMATH_CALUDE_square_root_fourth_power_l2310_231081

theorem square_root_fourth_power (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fourth_power_l2310_231081


namespace NUMINAMATH_CALUDE_unique_friendly_pair_l2310_231000

/-- Euler's totient function -/
def φ (n : ℕ) : ℕ := sorry

/-- The smallest positive integer greater than n that is not coprime to n -/
def f (n : ℕ) : ℕ := sorry

/-- A friendly pair is a pair of positive integers (n, m) where f(n) = m and φ(m) = n -/
def is_friendly_pair (n m : ℕ) : Prop :=
  f n = m ∧ φ m = n

theorem unique_friendly_pair : 
  ∀ n m : ℕ, is_friendly_pair n m → n = 2 ∧ m = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_friendly_pair_l2310_231000
