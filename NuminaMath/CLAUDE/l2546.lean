import Mathlib

namespace NUMINAMATH_CALUDE_no_extremum_range_l2546_254655

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- Theorem stating the range of a for which f(x) has no extremum -/
theorem no_extremum_range (a : ℝ) : 
  (∀ x : ℝ, f_derivative a x ≥ 0) ↔ a ∈ Set.Icc (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_no_extremum_range_l2546_254655


namespace NUMINAMATH_CALUDE_expected_votes_for_a_l2546_254646

-- Define the total number of voters (for simplicity, we'll use 100 as in the solution)
def total_voters : ℝ := 100

-- Define the percentage of Democratic voters
def dem_percentage : ℝ := 0.7

-- Define the percentage of Republican voters
def rep_percentage : ℝ := 1 - dem_percentage

-- Define the percentage of Democratic voters voting for candidate A
def dem_vote_for_a : ℝ := 0.8

-- Define the percentage of Republican voters voting for candidate A
def rep_vote_for_a : ℝ := 0.3

-- Theorem to prove
theorem expected_votes_for_a :
  (dem_percentage * dem_vote_for_a + rep_percentage * rep_vote_for_a) * 100 = 65 := by
  sorry


end NUMINAMATH_CALUDE_expected_votes_for_a_l2546_254646


namespace NUMINAMATH_CALUDE_spider_web_paths_l2546_254606

/-- The number of paths in a grid where only right and up moves are allowed -/
def number_of_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: In a 7x3 grid, the number of paths from bottom-left to top-right
    moving only right and up is equal to (10 choose 7) -/
theorem spider_web_paths :
  number_of_paths 7 3 = Nat.choose 10 7 := by
  sorry

end NUMINAMATH_CALUDE_spider_web_paths_l2546_254606


namespace NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l2546_254609

theorem arctan_sum_of_cubic_roots (u v w : ℝ) : 
  u^3 - 10*u + 11 = 0 → 
  v^3 - 10*v + 11 = 0 → 
  w^3 - 10*w + 11 = 0 → 
  u + v + w = 0 →
  u*v + v*w + w*u = -10 →
  u*v*w = -11 →
  Real.arctan u + Real.arctan v + Real.arctan w = π/4 := by sorry

end NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l2546_254609


namespace NUMINAMATH_CALUDE_cos_sum_sevenths_pi_l2546_254635

theorem cos_sum_sevenths_pi : 
  Real.cos (π / 7) + Real.cos (2 * π / 7) + Real.cos (3 * π / 7) + 
  Real.cos (4 * π / 7) + Real.cos (5 * π / 7) + Real.cos (6 * π / 7) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_sevenths_pi_l2546_254635


namespace NUMINAMATH_CALUDE_boys_from_clay_middle_school_l2546_254656

/-- Represents the three schools in the problem -/
inductive School
| Jonas
| Clay
| Pine

/-- Represents the gender of students -/
inductive Gender
| Boy
| Girl

/-- The total number of students at the camp -/
def total_students : ℕ := 150

/-- The number of boys at the camp -/
def total_boys : ℕ := 80

/-- The number of girls at the camp -/
def total_girls : ℕ := 70

/-- The number of students from each school -/
def students_per_school (s : School) : ℕ :=
  match s with
  | School.Jonas => 50
  | School.Clay => 60
  | School.Pine => 40

/-- The number of girls from Jonas Middle School -/
def girls_from_jonas : ℕ := 30

/-- The number of boys from Pine Middle School -/
def boys_from_pine : ℕ := 15

/-- The main theorem to prove -/
theorem boys_from_clay_middle_school :
  (students_per_school School.Clay) -
  (students_per_school School.Clay - 
   (total_boys - boys_from_pine - (students_per_school School.Jonas - girls_from_jonas))) = 45 := by
  sorry

end NUMINAMATH_CALUDE_boys_from_clay_middle_school_l2546_254656


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l2546_254630

theorem angle_sum_is_pi_over_two (α β : Real)
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h4 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l2546_254630


namespace NUMINAMATH_CALUDE_deepak_age_l2546_254616

/-- Given the ratio between Rahul and Deepak's ages is 4:3, and that Rahul will be 26 years old after 6 years, prove that Deepak's present age is 15 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  rahul_age = 4 * (rahul_age / 4) → 
  deepak_age = 3 * (rahul_age / 4) → 
  rahul_age + 6 = 26 → 
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l2546_254616


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l2546_254664

theorem polynomial_division_quotient (x : ℝ) :
  (x^2 + 7*x + 17) * (x - 2) + 43 = x^3 + 5*x^2 + 3*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l2546_254664


namespace NUMINAMATH_CALUDE_log_comparison_l2546_254670

theorem log_comparison : Real.log 675 / Real.log 135 > Real.log 75 / Real.log 45 := by
  sorry

end NUMINAMATH_CALUDE_log_comparison_l2546_254670


namespace NUMINAMATH_CALUDE_quadratic_equation_single_solution_l2546_254622

theorem quadratic_equation_single_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 15 * x + 6 = 0) :
  ∃ x, b * x^2 + 15 * x + 6 = 0 ∧ x = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_single_solution_l2546_254622


namespace NUMINAMATH_CALUDE_cosine_angle_between_vectors_l2546_254647

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![1, 3]

theorem cosine_angle_between_vectors :
  let inner_product := (a 0) * (b 0) + (a 1) * (b 1)
  let magnitude_a := Real.sqrt ((a 0)^2 + (a 1)^2)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2)
  (inner_product / (magnitude_a * magnitude_b)) = (7 * Real.sqrt 2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_angle_between_vectors_l2546_254647


namespace NUMINAMATH_CALUDE_percentage_commutation_l2546_254662

theorem percentage_commutation (x : ℝ) (h : 0.3 * (0.4 * x) = 48) : 
  0.4 * (0.3 * x) = 48 := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l2546_254662


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2546_254645

theorem fraction_to_decimal : (5 : ℚ) / 16 = (3125 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2546_254645


namespace NUMINAMATH_CALUDE_cookie_circle_radius_l2546_254695

theorem cookie_circle_radius (x y : ℝ) :
  (∃ (h k r : ℝ), ∀ x y : ℝ, x^2 + y^2 - 12*x + 16*y + 64 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) →
  (∃ (r : ℝ), r = 6 ∧ ∀ x y : ℝ, x^2 + y^2 - 12*x + 16*y + 64 = 0 ↔ (x - 6)^2 + (y + 8)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_cookie_circle_radius_l2546_254695


namespace NUMINAMATH_CALUDE_units_digit_17_pow_2007_l2546_254637

theorem units_digit_17_pow_2007 : (17^2007 : ℕ) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_2007_l2546_254637


namespace NUMINAMATH_CALUDE_probability_of_all_even_sums_l2546_254659

/-- Represents a tile with a number from 1 to 10 -/
def Tile := Fin 10

/-- Represents a player's selection of 3 tiles -/
def PlayerSelection := Fin 3 → Tile

/-- The set of all possible distributions of tiles to three players -/
def AllDistributions := Fin 3 → PlayerSelection

/-- Checks if a player's selection sum is even -/
def isEvenSum (selection : PlayerSelection) : Prop :=
  (selection 0).val + (selection 1).val + (selection 2).val % 2 = 0

/-- Checks if all players have even sums in a distribution -/
def allEvenSums (distribution : AllDistributions) : Prop :=
  ∀ i : Fin 3, isEvenSum (distribution i)

/-- The number of distributions where all players have even sums -/
def favorableDistributions : ℕ := sorry

/-- The total number of possible distributions -/
def totalDistributions : ℕ := sorry

/-- The main theorem stating the probability -/
theorem probability_of_all_even_sums :
  (favorableDistributions : ℚ) / totalDistributions = 1 / 28 := sorry

end NUMINAMATH_CALUDE_probability_of_all_even_sums_l2546_254659


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2546_254672

/-- The speed of a boat in still water, given stream speed and downstream travel data -/
theorem boat_speed_in_still_water (stream_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) : 
  stream_speed = 4 →
  downstream_distance = 112 →
  downstream_time = 4 →
  (downstream_distance / downstream_time) - stream_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2546_254672


namespace NUMINAMATH_CALUDE_smallest_number_l2546_254699

theorem smallest_number : 
  let numbers := [-0.991, -0.981, -0.989, -0.9801, -0.9901]
  ∀ x ∈ numbers, -0.991 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2546_254699


namespace NUMINAMATH_CALUDE_four_row_grid_has_27_triangles_l2546_254631

/-- Represents a triangular grid with a given number of rows -/
structure TriangularGrid :=
  (rows : ℕ)

/-- Counts the number of small triangles in a triangular grid -/
def countSmallTriangles (grid : TriangularGrid) : ℕ :=
  (grid.rows * (grid.rows + 1)) / 2

/-- Counts the number of medium triangles in a triangular grid -/
def countMediumTriangles (grid : TriangularGrid) : ℕ :=
  ((grid.rows - 1) * grid.rows) / 2

/-- Counts the number of large triangles in a triangular grid -/
def countLargeTriangles (grid : TriangularGrid) : ℕ :=
  if grid.rows ≥ 3 then 1 else 0

/-- Counts the total number of triangles in a triangular grid -/
def countTotalTriangles (grid : TriangularGrid) : ℕ :=
  countSmallTriangles grid + countMediumTriangles grid + countLargeTriangles grid

/-- Theorem: A triangular grid with 4 rows contains 27 triangles in total -/
theorem four_row_grid_has_27_triangles :
  countTotalTriangles (TriangularGrid.mk 4) = 27 := by
  sorry

end NUMINAMATH_CALUDE_four_row_grid_has_27_triangles_l2546_254631


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2546_254642

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₂ * a₆ = 4, then a₄ = 2 or a₄ = -2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) (h_prod : a 2 * a 6 = 4) : 
  a 4 = 2 ∨ a 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2546_254642


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2546_254680

theorem smallest_integer_with_remainders :
  ∃ (x : ℕ), x > 0 ∧
  x % 6 = 5 ∧
  x % 7 = 6 ∧
  x % 8 = 7 ∧
  ∀ (y : ℕ), y > 0 →
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2546_254680


namespace NUMINAMATH_CALUDE_sandy_change_l2546_254661

/-- The change Sandy received from her purchase of toys -/
def change_received (football_price baseball_price paid : ℚ) : ℚ :=
  paid - (football_price + baseball_price)

/-- Theorem stating the correct change Sandy received -/
theorem sandy_change : change_received 9.14 6.81 20 = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_l2546_254661


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l2546_254603

/-- Given two employees A and B with a total weekly pay of 550 and B's pay of 220,
    prove that A's pay is 150% of B's pay. -/
theorem employee_pay_percentage (total_pay : ℝ) (b_pay : ℝ) (a_pay : ℝ)
  (h1 : total_pay = 550)
  (h2 : b_pay = 220)
  (h3 : a_pay + b_pay = total_pay) :
  a_pay / b_pay * 100 = 150 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l2546_254603


namespace NUMINAMATH_CALUDE_no_prime_solution_l2546_254674

def base_p_to_decimal (n : ℕ) (p : ℕ) : ℕ :=
  let digits := n.digits p
  (List.range digits.length).foldl (λ acc i => acc + digits[i]! * p ^ i) 0

theorem no_prime_solution :
  ∀ p : ℕ, p.Prime → p ≠ 2 → p ≠ 3 → p ≠ 5 → p ≠ 7 →
    base_p_to_decimal 1014 p + base_p_to_decimal 309 p + base_p_to_decimal 120 p +
    base_p_to_decimal 132 p + base_p_to_decimal 7 p ≠
    base_p_to_decimal 153 p + base_p_to_decimal 276 p + base_p_to_decimal 371 p :=
by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l2546_254674


namespace NUMINAMATH_CALUDE_johns_days_off_l2546_254643

/-- Calculates the number of days John takes off per week given his streaming schedule and earnings. -/
theorem johns_days_off (hours_per_session : ℕ) (hourly_rate : ℕ) (weekly_earnings : ℕ) (days_per_week : ℕ)
  (h1 : hours_per_session = 4)
  (h2 : hourly_rate = 10)
  (h3 : weekly_earnings = 160)
  (h4 : days_per_week = 7) :
  days_per_week - (weekly_earnings / hourly_rate / hours_per_session) = 3 :=
by sorry

end NUMINAMATH_CALUDE_johns_days_off_l2546_254643


namespace NUMINAMATH_CALUDE_meal_price_calculation_l2546_254667

theorem meal_price_calculation (beef_amount : ℝ) (pork_ratio : ℝ) (meat_per_meal : ℝ) (total_revenue : ℝ) :
  beef_amount = 20 →
  pork_ratio = 1 / 2 →
  meat_per_meal = 1.5 →
  total_revenue = 400 →
  (total_revenue / ((beef_amount + beef_amount * pork_ratio) / meat_per_meal) = 20) :=
by sorry

end NUMINAMATH_CALUDE_meal_price_calculation_l2546_254667


namespace NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l2546_254693

theorem probability_at_least_one_multiple_of_four :
  let total_numbers : ℕ := 100
  let multiples_of_four : ℕ := 25
  let non_multiples_of_four : ℕ := total_numbers - multiples_of_four
  let prob_non_multiple : ℚ := non_multiples_of_four / total_numbers
  let prob_both_non_multiples : ℚ := prob_non_multiple * prob_non_multiple
  let prob_at_least_one_multiple : ℚ := 1 - prob_both_non_multiples
  prob_at_least_one_multiple = 7 / 16 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l2546_254693


namespace NUMINAMATH_CALUDE_factor_expression_l2546_254634

theorem factor_expression (a b : ℝ) : 2*a^2*b - 4*a*b^2 + 2*b^3 = 2*b*(a-b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2546_254634


namespace NUMINAMATH_CALUDE_history_students_count_l2546_254689

def total_students : ℕ := 86
def math_students : ℕ := 17
def english_students : ℕ := 36
def all_three_classes : ℕ := 3
def exactly_two_classes : ℕ := 3

theorem history_students_count : 
  ∃ (history_students : ℕ), 
    history_students = total_students - math_students - english_students + all_three_classes := by
  sorry

end NUMINAMATH_CALUDE_history_students_count_l2546_254689


namespace NUMINAMATH_CALUDE_tangent_y_intercept_l2546_254658

/-- The function representing the curve y = x³ + 11 -/
def f (x : ℝ) : ℝ := x^3 + 11

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3 * x^2

/-- The point of tangency -/
def point_of_tangency : ℝ × ℝ := (1, 12)

/-- The slope of the tangent line at the point of tangency -/
def tangent_slope : ℝ := f' point_of_tangency.1

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := point_of_tangency.2 - tangent_slope * point_of_tangency.1

theorem tangent_y_intercept :
  y_intercept = 9 :=
sorry

end NUMINAMATH_CALUDE_tangent_y_intercept_l2546_254658


namespace NUMINAMATH_CALUDE_married_men_fraction_l2546_254626

theorem married_men_fraction (total_women : ℕ) (h_total_women_pos : total_women > 0) :
  let single_women := (3 : ℚ) / 7 * total_women
  let married_women := total_women - single_women
  let married_men := married_women
  let total_people := total_women + married_men
  married_men / total_people = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l2546_254626


namespace NUMINAMATH_CALUDE_triangle_properties_l2546_254652

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  b = a * Real.cos C + (Real.sqrt 3 / 3) * a * Real.sin C →
  a = 2 →
  b + c ≥ 4 →
  A = π / 3 ∧ (1 / 2) * a * b * Real.sin C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2546_254652


namespace NUMINAMATH_CALUDE_sum_of_coefficients_factorized_form_l2546_254605

theorem sum_of_coefficients_factorized_form (x y : ℝ) : 
  ∃ (a b c d e : ℤ), 
    27 * x^6 - 512 * y^6 = (a * x^2 + b * y^2) * (c * x^4 + d * x^2 * y^2 + e * y^4) ∧
    a + b + c + d + e = 92 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_factorized_form_l2546_254605


namespace NUMINAMATH_CALUDE_bells_synchronization_l2546_254660

def church_interval : ℕ := 18
def school_interval : ℕ := 24
def daycare_interval : ℕ := 30
def library_interval : ℕ := 35

def noon_hour : ℕ := 12

theorem bells_synchronization :
  let intervals := [church_interval, school_interval, daycare_interval, library_interval]
  let lcm_minutes := Nat.lcm (Nat.lcm (Nat.lcm church_interval school_interval) daycare_interval) library_interval
  let hours_after_noon := lcm_minutes / 60
  let next_sync_hour := (noon_hour + hours_after_noon) % 24
  next_sync_hour = 6 ∧ hours_after_noon = 42 := by sorry

end NUMINAMATH_CALUDE_bells_synchronization_l2546_254660


namespace NUMINAMATH_CALUDE_correct_rows_per_bus_l2546_254684

/-- Represents the number of rows in each bus -/
def rows_per_bus : ℕ := 10

/-- Represents the number of columns in each bus -/
def columns_per_bus : ℕ := 4

/-- Represents the total number of buses -/
def total_buses : ℕ := 6

/-- Represents the total number of students that can be accommodated -/
def total_students : ℕ := 240

/-- Theorem stating that the number of rows per bus is correct -/
theorem correct_rows_per_bus : 
  rows_per_bus * columns_per_bus * total_buses = total_students := by
  sorry

end NUMINAMATH_CALUDE_correct_rows_per_bus_l2546_254684


namespace NUMINAMATH_CALUDE_teds_age_l2546_254691

theorem teds_age (s t j : ℕ) : 
  t = 2 * s - 20 →
  j = s + 6 →
  t + s + j = 90 →
  t = 32 := by
  sorry

end NUMINAMATH_CALUDE_teds_age_l2546_254691


namespace NUMINAMATH_CALUDE_sine_phase_shift_specific_sine_phase_shift_l2546_254677

/-- The phase shift of a sine function y = a * sin(bx - c) is c/b to the right when c is positive. -/
theorem sine_phase_shift (a b c : ℝ) (h : c > 0) :
  let f := fun x => a * Real.sin (b * x - c)
  let phase_shift := c / b
  (∀ x, f (x + phase_shift) = a * Real.sin (b * x)) :=
sorry

/-- The phase shift of y = 3 * sin(3x - π/4) is π/12 to the right. -/
theorem specific_sine_phase_shift :
  let f := fun x => 3 * Real.sin (3 * x - π/4)
  let phase_shift := π/12
  (∀ x, f (x + phase_shift) = 3 * Real.sin (3 * x)) :=
sorry

end NUMINAMATH_CALUDE_sine_phase_shift_specific_sine_phase_shift_l2546_254677


namespace NUMINAMATH_CALUDE_triple_f_of_3_l2546_254632

def f (x : ℝ) : ℝ := 7 * x - 3

theorem triple_f_of_3 : f (f (f 3)) = 858 := by sorry

end NUMINAMATH_CALUDE_triple_f_of_3_l2546_254632


namespace NUMINAMATH_CALUDE_total_lollipops_eq_twelve_l2546_254683

/-- The number of lollipops Sushi's father brought -/
def total_lollipops : ℕ := sorry

/-- The number of lollipops eaten by the children -/
def eaten_lollipops : ℕ := 5

/-- The number of lollipops left -/
def remaining_lollipops : ℕ := 7

/-- Theorem stating that the total number of lollipops equals 12 -/
theorem total_lollipops_eq_twelve :
  total_lollipops = eaten_lollipops + remaining_lollipops ∧
  total_lollipops = 12 := by sorry

end NUMINAMATH_CALUDE_total_lollipops_eq_twelve_l2546_254683


namespace NUMINAMATH_CALUDE_inequality_proof_l2546_254679

theorem inequality_proof (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) :
  (a + b < b + c) ∧ (a / (a + b) < 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2546_254679


namespace NUMINAMATH_CALUDE_faulty_passed_ratio_is_one_to_eight_l2546_254669

/-- Represents the ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the circuit board inspection results -/
structure CircuitBoardInspection where
  total : ℕ
  failed : ℕ
  faulty : ℕ

def faultyPassedRatio (inspection : CircuitBoardInspection) : Ratio :=
  { numerator := inspection.faulty - inspection.failed,
    denominator := inspection.total - inspection.failed }

theorem faulty_passed_ratio_is_one_to_eight 
  (inspection : CircuitBoardInspection) 
  (h1 : inspection.total = 3200)
  (h2 : inspection.failed = 64)
  (h3 : inspection.faulty = 456) : 
  faultyPassedRatio inspection = { numerator := 1, denominator := 8 } := by
  sorry

#check faulty_passed_ratio_is_one_to_eight

end NUMINAMATH_CALUDE_faulty_passed_ratio_is_one_to_eight_l2546_254669


namespace NUMINAMATH_CALUDE_rationalize_sqrt_5_12_l2546_254619

theorem rationalize_sqrt_5_12 : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_5_12_l2546_254619


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2546_254690

theorem min_value_quadratic (x : ℝ) :
  ∃ (m : ℝ), m = 1438 ∧ ∀ x, 3 * x^2 - 12 * x + 1450 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2546_254690


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l2546_254675

theorem unique_solution_sqrt_equation :
  ∀ m n : ℕ+, 
    (m : ℝ)^2 = Real.sqrt (n : ℝ) + Real.sqrt ((2 * n + 1) : ℝ) → 
    m = 13 ∧ n = 4900 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l2546_254675


namespace NUMINAMATH_CALUDE_base_8_to_base_7_l2546_254682

def base_8_to_decimal (n : ℕ) : ℕ := n

def decimal_to_base_7 (n : ℕ) : ℕ := n

theorem base_8_to_base_7 :
  decimal_to_base_7 (base_8_to_decimal 536) = 1010 :=
sorry

end NUMINAMATH_CALUDE_base_8_to_base_7_l2546_254682


namespace NUMINAMATH_CALUDE_juliet_supporter_in_capulet_l2546_254686

-- Define the population distribution
def montague_pop : ℚ := 4/6
def capulet_pop : ℚ := 1/6
def verona_pop : ℚ := 1/6

-- Define the support percentages for Juliet
def montague_juliet : ℚ := 1/5  -- 20% support Juliet (100% - 80%)
def capulet_juliet : ℚ := 7/10
def verona_juliet : ℚ := 3/5

-- Theorem statement
theorem juliet_supporter_in_capulet :
  let total_juliet := montague_pop * montague_juliet + capulet_pop * capulet_juliet + verona_pop * verona_juliet
  (capulet_pop * capulet_juliet) / total_juliet = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_juliet_supporter_in_capulet_l2546_254686


namespace NUMINAMATH_CALUDE_both_save_800_l2546_254678

/-- Represents the financial situation of Anand and Balu -/
structure FinancialSituation where
  anand_income : ℕ
  balu_income : ℕ
  anand_expenditure : ℕ
  balu_expenditure : ℕ

/-- Checks if the given financial situation satisfies the problem conditions -/
def satisfies_conditions (fs : FinancialSituation) : Prop :=
  fs.anand_income * 4 = fs.balu_income * 5 ∧
  fs.anand_expenditure * 2 = fs.balu_expenditure * 3 ∧
  fs.anand_income = 2000

/-- Calculates the savings for a person given their income and expenditure -/
def savings (income : ℕ) (expenditure : ℕ) : ℕ :=
  income - expenditure

/-- Theorem stating that both Anand and Balu save 800 each -/
theorem both_save_800 (fs : FinancialSituation) (h : satisfies_conditions fs) :
  savings fs.anand_income fs.anand_expenditure = 800 ∧
  savings fs.balu_income fs.balu_expenditure = 800 := by
  sorry


end NUMINAMATH_CALUDE_both_save_800_l2546_254678


namespace NUMINAMATH_CALUDE_right_triangle_of_orthocenters_l2546_254657

-- Define the circle and points
def Circle : Type := ℂ → Prop
def on_circle (c : Circle) (p : ℂ) : Prop := c p

-- Define the orthocenter function
def orthocenter (a b c : ℂ) : ℂ := sorry

-- Main theorem
theorem right_triangle_of_orthocenters 
  (O A B C D E : ℂ) 
  (c : Circle)
  (on_circle_A : on_circle c A)
  (on_circle_B : on_circle c B)
  (on_circle_C : on_circle c C)
  (on_circle_D : on_circle c D)
  (on_circle_E : on_circle c E)
  (consecutive : sorry) -- Represent that A, B, C, D, E are consecutive
  (equal_chords : AC = BD ∧ BD = CE ∧ CE = DO)
  (H₁ : ℂ := orthocenter A C D)
  (H₂ : ℂ := orthocenter B C D)
  (H₃ : ℂ := orthocenter B C E) :
  ∃ (θ : ℝ), Complex.arg ((H₁ - H₂) / (H₁ - H₃)) = θ ∧ θ = π/2 := by sorry

#check right_triangle_of_orthocenters

end NUMINAMATH_CALUDE_right_triangle_of_orthocenters_l2546_254657


namespace NUMINAMATH_CALUDE_fraction_equality_l2546_254617

theorem fraction_equality (x y : ℝ) (h : (1 / x) - (1 / y) = 2) :
  (x + x*y - y) / (x - x*y - y) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2546_254617


namespace NUMINAMATH_CALUDE_ratio_problem_l2546_254612

theorem ratio_problem (a b x y : ℕ) : 
  a > b → 
  a - b = 5 → 
  a * 5 = b * 6 → 
  (a - x) * 4 = (b - x) * 5 → 
  (a + y) * 6 = (b + y) * 7 → 
  x = 5 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2546_254612


namespace NUMINAMATH_CALUDE_train_passing_jogger_l2546_254602

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) 
  (h1 : jogger_speed = 9 * (1000 / 3600))
  (h2 : train_speed = 45 * (1000 / 3600))
  (h3 : train_length = 120)
  (h4 : initial_distance = 200) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 32 := by
  sorry

#check train_passing_jogger

end NUMINAMATH_CALUDE_train_passing_jogger_l2546_254602


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l2546_254604

theorem smallest_five_digit_divisible_by_first_five_primes :
  let first_five_primes := [2, 3, 5, 7, 11]
  let is_five_digit (n : ℕ) := 10000 ≤ n ∧ n ≤ 99999
  let divisible_by_all (n : ℕ) := ∀ p ∈ first_five_primes, n % p = 0
  ∃ (n : ℕ), is_five_digit n ∧ divisible_by_all n ∧
    ∀ m, is_five_digit m ∧ divisible_by_all m → n ≤ m ∧ n = 11550 :=
by
  sorry

#eval 11550 % 2  -- Should output 0
#eval 11550 % 3  -- Should output 0
#eval 11550 % 5  -- Should output 0
#eval 11550 % 7  -- Should output 0
#eval 11550 % 11 -- Should output 0

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l2546_254604


namespace NUMINAMATH_CALUDE_fraction_simplification_l2546_254676

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  (1 - 2 / (x + 1)) / (x / (x + 1)) = (x - 1) / x := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2546_254676


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2546_254618

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, n^2 < 2^n) ↔ (∃ n₀ : ℕ, n₀^2 ≥ 2^n₀) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2546_254618


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2546_254651

theorem abs_neg_three_eq_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2546_254651


namespace NUMINAMATH_CALUDE_min_value_a_l2546_254688

theorem min_value_a (a : ℝ) : 
  (∀ x > a, 2 * x + 2 / (x - 1) ≥ 7) → a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l2546_254688


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2546_254627

theorem min_value_of_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (2*a*(-1) - b*2 + 2 = 0) → (1/a + 1/b ≥ 4) ∧ ∃ a b, (1/a + 1/b = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2546_254627


namespace NUMINAMATH_CALUDE_special_function_properties_l2546_254636

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) - f y = (x + 2*y + 2) * x) ∧ (f 2 = 12)

theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (f 0 = 4) ∧
  (Set.Icc (-1 : ℝ) 5 = {a | ∃ x₀ ∈ Set.Ioo 1 4, f x₀ - 8 = a * x₀}) :=
sorry

end NUMINAMATH_CALUDE_special_function_properties_l2546_254636


namespace NUMINAMATH_CALUDE_mean_temperature_l2546_254687

def temperatures : List ℝ := [-7, -4, -4, -5, 1, 3, 2, 4]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℝ) = -1.25 := by
sorry

end NUMINAMATH_CALUDE_mean_temperature_l2546_254687


namespace NUMINAMATH_CALUDE_octagon_area_theorem_l2546_254621

/-- The area of an octagon formed by the intersection of two unit squares with the same center -/
def octagon_area (side_length : ℚ) : ℚ :=
  8 * (side_length * (1 / 2) * (1 / 2))

/-- The theorem stating the area of the octagon given the side length -/
theorem octagon_area_theorem (h : octagon_area (43 / 99) = 86 / 99) : True := by
  sorry

#eval octagon_area (43 / 99)

end NUMINAMATH_CALUDE_octagon_area_theorem_l2546_254621


namespace NUMINAMATH_CALUDE_species_assignment_theorem_l2546_254698

/-- Represents the compatibility between species -/
def Compatibility := Fin 8 → Finset (Fin 8)

/-- Theorem stating that it's possible to assign 8 species to 4 cages
    given the compatibility constraints -/
theorem species_assignment_theorem (c : Compatibility)
  (h : ∀ s : Fin 8, (c s).card ≤ 4) :
  ∃ (assignment : Fin 8 → Fin 4),
    ∀ s₁ s₂ : Fin 8, assignment s₁ = assignment s₂ → s₂ ∈ c s₁ := by
  sorry

end NUMINAMATH_CALUDE_species_assignment_theorem_l2546_254698


namespace NUMINAMATH_CALUDE_octagon_arc_length_l2546_254663

/-- The length of the arc intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (r : ℝ) (h : r = 4) : 
  (2 * π * r) / 8 = π := by sorry

end NUMINAMATH_CALUDE_octagon_arc_length_l2546_254663


namespace NUMINAMATH_CALUDE_f_two_equals_two_l2546_254620

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x) = (x^2 - x)/2 * f x + 2 - x

-- Theorem statement
theorem f_two_equals_two (h : has_property f) : f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_two_equals_two_l2546_254620


namespace NUMINAMATH_CALUDE_condition_relationship_l2546_254610

open Set

def condition_p (x : ℝ) : Prop := |x - 1| < 2
def condition_q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

def set_p : Set ℝ := {x | -1 < x ∧ x < 3}
def set_q : Set ℝ := {x | -1 < x ∧ x < 6}

theorem condition_relationship :
  (∀ x, condition_p x → x ∈ set_p) ∧
  (∀ x, condition_q x → x ∈ set_q) ∧
  set_p ⊂ set_q :=
sorry

end NUMINAMATH_CALUDE_condition_relationship_l2546_254610


namespace NUMINAMATH_CALUDE_sine_inequality_l2546_254615

theorem sine_inequality (t : ℝ) (h1 : 0 < t) (h2 : t ≤ π / 2) :
  1 / (Real.sin t)^2 ≤ 1 / t^2 + 1 - 4 / π^2 := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l2546_254615


namespace NUMINAMATH_CALUDE_no_real_roots_for_specific_k_l2546_254692

theorem no_real_roots_for_specific_k : ∀ x : ℝ, x^2 + 2*x + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_specific_k_l2546_254692


namespace NUMINAMATH_CALUDE_linear_equation_result_l2546_254625

theorem linear_equation_result (x m : ℝ) : 
  (∃ a b : ℝ, x^(2*m-3) + 6 = a*x + b) → (x + 3)^2010 = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_result_l2546_254625


namespace NUMINAMATH_CALUDE_gcd_12012_18018_l2546_254650

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12012_18018_l2546_254650


namespace NUMINAMATH_CALUDE_dodecahedron_path_count_l2546_254666

/-- Represents a face of the dodecahedron --/
inductive Face
  | Top
  | Bottom
  | UpperRing (n : Fin 5)
  | LowerRing (n : Fin 5)

/-- Represents a valid path on the dodecahedron --/
def ValidPath : List Face → Prop :=
  sorry

/-- The number of valid paths from top to bottom face --/
def numValidPaths : Nat :=
  sorry

/-- Theorem stating that the number of valid paths is 810 --/
theorem dodecahedron_path_count :
  numValidPaths = 810 :=
sorry

end NUMINAMATH_CALUDE_dodecahedron_path_count_l2546_254666


namespace NUMINAMATH_CALUDE_line_equation_perpendicular_line_equation_opposite_intercepts_l2546_254611

-- Define the line l
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the perpendicular line
def perpendicularLine : Line := { a := 2, b := 1, c := 3 }

-- Define the condition for a line to pass through a point
def passesThrough (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Define the condition for two lines to be perpendicular
def isPerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define the condition for a line to have intercepts with opposite signs
def hasOppositeIntercepts (l : Line) : Prop :=
  (l.a * l.c < 0 ∧ l.b * l.c < 0) ∨ (l.a = 0 ∧ l.b ≠ 0) ∨ (l.a ≠ 0 ∧ l.b = 0)

theorem line_equation_perpendicular (l : Line) :
  passesThrough l P ∧ isPerpendicular l perpendicularLine →
  l = { a := 1, b := -2, c := -4 } :=
sorry

theorem line_equation_opposite_intercepts (l : Line) :
  passesThrough l P ∧ hasOppositeIntercepts l →
  (l = { a := 1, b := 2, c := 0 } ∨ l = { a := 1, b := -1, c := -3 }) :=
sorry

end NUMINAMATH_CALUDE_line_equation_perpendicular_line_equation_opposite_intercepts_l2546_254611


namespace NUMINAMATH_CALUDE_missing_sale_is_3920_l2546_254614

/-- Calculates the missing sale amount given the sales for 5 months and the desired average -/
def calculate_missing_sale (sales : List ℕ) (average : ℕ) : ℕ :=
  6 * average - sales.sum

/-- The list of known sales amounts -/
def known_sales : List ℕ := [3435, 3855, 4230, 3560, 2000]

/-- The desired average sale -/
def desired_average : ℕ := 3500

theorem missing_sale_is_3920 :
  calculate_missing_sale known_sales desired_average = 3920 := by
  sorry

#eval calculate_missing_sale known_sales desired_average

end NUMINAMATH_CALUDE_missing_sale_is_3920_l2546_254614


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l2546_254648

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n^2 % 2 = 0 → n^2 % 3 = 0 → n^2 % 5 = 0 → n^2 ≥ 225 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l2546_254648


namespace NUMINAMATH_CALUDE_c_rent_share_is_27_l2546_254654

/-- Represents the rental information for a person --/
structure RentalInfo where
  oxen : ℕ
  months : ℕ

/-- Calculates the total rent share for a person --/
def calculateRentShare (totalRent : ℚ) (totalOxenMonths : ℕ) (info : RentalInfo) : ℚ :=
  totalRent * (info.oxen * info.months : ℚ) / totalOxenMonths

theorem c_rent_share_is_27 
  (a b c : RentalInfo)
  (h_a : a = ⟨10, 7⟩)
  (h_b : b = ⟨12, 5⟩)
  (h_c : c = ⟨15, 3⟩)
  (h_total_rent : totalRent = 105)
  (h_total_oxen_months : totalOxenMonths = a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) :
  calculateRentShare totalRent totalOxenMonths c = 27 := by
  sorry


end NUMINAMATH_CALUDE_c_rent_share_is_27_l2546_254654


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l2546_254600

theorem square_rectangle_area_relation : 
  ∀ x : ℝ,
  let square_side := x - 4
  let rect_length := x - 2
  let rect_width := x + 6
  let square_area := square_side * square_side
  let rect_area := rect_length * rect_width
  rect_area = 3 * square_area →
  (∃ x₁ x₂ : ℝ, 
    (square_side = x₁ - 4 ∧ rect_length = x₁ - 2 ∧ rect_width = x₁ + 6 ∧
     square_side = x₂ - 4 ∧ rect_length = x₂ - 2 ∧ rect_width = x₂ + 6) ∧
    x₁ + x₂ = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l2546_254600


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2546_254623

theorem quadratic_equation_roots (k : ℝ) (h : k > 1) :
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  (2 * x₁^2 - (4*k + 1) * x₁ + 2*k^2 - 1 = 0) ∧
  (2 * x₂^2 - (4*k + 1) * x₂ + 2*k^2 - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2546_254623


namespace NUMINAMATH_CALUDE_laura_has_435_l2546_254607

/-- Calculates Laura's money given Darwin's money -/
def lauras_money (darwins_money : ℕ) : ℕ :=
  let mias_money := 2 * darwins_money + 20
  let combined_money := mias_money + darwins_money
  3 * combined_money - 30

/-- Proves that Laura has $435 given the conditions -/
theorem laura_has_435 : lauras_money 45 = 435 := by
  sorry

end NUMINAMATH_CALUDE_laura_has_435_l2546_254607


namespace NUMINAMATH_CALUDE_abs_neg_one_third_l2546_254697

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_l2546_254697


namespace NUMINAMATH_CALUDE_porche_homework_time_l2546_254639

/-- Proves that given a total time of 3 hours (180 minutes) and homework assignments
    taking 45, 30, 50, and 25 minutes respectively, the remaining time for a special project
    is 30 minutes. -/
theorem porche_homework_time (total_time : ℕ) (math_time english_time science_time history_time : ℕ) :
  total_time = 180 ∧
  math_time = 45 ∧
  english_time = 30 ∧
  science_time = 50 ∧
  history_time = 25 →
  total_time - (math_time + english_time + science_time + history_time) = 30 :=
by sorry

end NUMINAMATH_CALUDE_porche_homework_time_l2546_254639


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2546_254624

theorem fourteenth_root_of_unity : 
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (2 * n * π * Complex.I / 14) :=
by sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2546_254624


namespace NUMINAMATH_CALUDE_radical_simplification_l2546_254613

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (20 * p) * Real.sqrt (10 * p^3) * Real.sqrt (6 * p^4) * Real.sqrt (15 * p^5) = 20 * p^6 * Real.sqrt (15 * p) :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l2546_254613


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2546_254629

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 144 is 8√2 -/
theorem ellipse_foci_distance : 
  ∀ (x y : ℝ), 9 * x^2 + y^2 = 144 → 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2546_254629


namespace NUMINAMATH_CALUDE_kendra_age_l2546_254673

/-- Proves that Kendra's age is 18 given the conditions in the problem -/
theorem kendra_age :
  ∀ (k s t : ℕ), -- k: Kendra's age, s: Sam's age, t: Sue's age
  s = 2 * t →    -- Sam is twice as old as Sue
  k = 3 * s →    -- Kendra is 3 times as old as Sam
  (k + 3) + (s + 3) + (t + 3) = 36 → -- Their total age in 3 years will be 36
  k = 18 := by
    sorry -- Proof omitted

end NUMINAMATH_CALUDE_kendra_age_l2546_254673


namespace NUMINAMATH_CALUDE_no_strictly_monotonic_pair_l2546_254694

theorem no_strictly_monotonic_pair :
  ¬∃ (f g : ℕ → ℕ),
    (∀ x y, x < y → f x < f y) ∧
    (∀ x y, x < y → g x < g y) ∧
    (∀ n, f (g (g n)) < g (f n)) :=
by sorry

end NUMINAMATH_CALUDE_no_strictly_monotonic_pair_l2546_254694


namespace NUMINAMATH_CALUDE_solve_weeks_worked_problem_l2546_254681

/-- Represents the problem of calculating the number of weeks worked --/
def WeeksWorkedProblem (regular_days_per_week : ℕ) 
                       (hours_per_day : ℕ) 
                       (regular_pay_rate : ℚ) 
                       (overtime_pay_rate : ℚ) 
                       (total_earnings : ℚ) 
                       (total_hours : ℕ) : Prop :=
  let regular_hours_per_week := regular_days_per_week * hours_per_day
  ∃ (weeks_worked : ℕ),
    let regular_hours := weeks_worked * regular_hours_per_week
    let overtime_hours := total_hours - regular_hours
    regular_hours * regular_pay_rate + overtime_hours * overtime_pay_rate = total_earnings ∧
    weeks_worked = 4

/-- The main theorem stating the solution to the problem --/
theorem solve_weeks_worked_problem :
  WeeksWorkedProblem 6 10 (210/100) (420/100) 525 245 := by
  sorry

#check solve_weeks_worked_problem

end NUMINAMATH_CALUDE_solve_weeks_worked_problem_l2546_254681


namespace NUMINAMATH_CALUDE_factorial_division_l2546_254633

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l2546_254633


namespace NUMINAMATH_CALUDE_select_and_arrange_five_three_unique_descending_arrangement_select_three_from_five_descending_l2546_254696

/-- The number of ways to select and arrange 3 people from 5 in descending height order -/
def select_and_arrange (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem select_and_arrange_five_three :
  select_and_arrange 5 3 = Nat.choose 5 3 := by
  sorry

/-- The number of ways to arrange 3 people in descending height order -/
def arrange_descending (k : ℕ) : ℕ := 1

theorem unique_descending_arrangement (k : ℕ) :
  arrange_descending k = 1 := by
  sorry

/-- The main theorem: selecting and arranging 3 from 5 equals C(5,3) -/
theorem select_three_from_five_descending :
  select_and_arrange 5 3 = Nat.choose 5 3 := by
  sorry

end NUMINAMATH_CALUDE_select_and_arrange_five_three_unique_descending_arrangement_select_three_from_five_descending_l2546_254696


namespace NUMINAMATH_CALUDE_colby_remaining_mangoes_l2546_254640

def total_harvest : ℕ := 60
def sold_to_market : ℕ := 20
def mangoes_per_kg : ℕ := 8

def remaining_after_market : ℕ := total_harvest - sold_to_market

def sold_to_community : ℕ := remaining_after_market / 2

def remaining_kg : ℕ := remaining_after_market - sold_to_community

theorem colby_remaining_mangoes :
  remaining_kg * mangoes_per_kg = 160 := by sorry

end NUMINAMATH_CALUDE_colby_remaining_mangoes_l2546_254640


namespace NUMINAMATH_CALUDE_exactly_one_true_l2546_254608

def proposition1 : Prop := ∀ x : ℝ, x^4 > x^2

def proposition2 : Prop := ∀ p q : Prop, (¬(p ∧ q)) → (¬p ∧ ¬q)

def proposition3 : Prop := (¬∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0)

theorem exactly_one_true : 
  (proposition1 ∧ ¬proposition2 ∧ ¬proposition3) ∨
  (¬proposition1 ∧ proposition2 ∧ ¬proposition3) ∨
  (¬proposition1 ∧ ¬proposition2 ∧ proposition3) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_true_l2546_254608


namespace NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l2546_254628

theorem gcd_of_powers_minus_one : Nat.gcd (2^300 - 1) (2^315 - 1) = 2^15 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l2546_254628


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2546_254644

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2546_254644


namespace NUMINAMATH_CALUDE_range_of_fraction_l2546_254685

-- Define a monotonically decreasing function on ℝ
def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- Define symmetry of f(x-1) with respect to (1,0)
def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = -f x

-- Main theorem
theorem range_of_fraction (f : ℝ → ℝ) (h_decr : monotonically_decreasing f)
    (h_sym : symmetric_about_one f) :
    ∀ t : ℝ, f (t^2 - 2*t) + f (-3) > 0 → (t - 1) / (t - 3) < 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_fraction_l2546_254685


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l2546_254671

theorem coefficient_x_cubed_in_binomial_expansion :
  (Finset.range 7).sum (λ k => Nat.choose 6 k * 2^(6 - k) * if k = 3 then 1 else 0) = 160 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l2546_254671


namespace NUMINAMATH_CALUDE_find_a_value_l2546_254668

theorem find_a_value (a : ℕ) (h : a ^ 3 = 21 * 25 * 45 * 49) : a = 105 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l2546_254668


namespace NUMINAMATH_CALUDE_min_value_line_circle_l2546_254649

/-- Given a line ax + by + c - 1 = 0 (where b, c > 0) passing through the center of the circle x^2 + y^2 - 2y - 5 = 0, 
    the minimum value of 4/b + 1/c is 9. -/
theorem min_value_line_circle (a b c : ℝ) : 
  b > 0 → c > 0 → 
  (∃ x y : ℝ, a * x + b * y + c - 1 = 0 ∧ x^2 + y^2 - 2*y - 5 = 0) →
  (∀ b' c' : ℝ, b' > 0 → c' > 0 → 
    (∃ x y : ℝ, a * x + b' * y + c' - 1 = 0 ∧ x^2 + y^2 - 2*y - 5 = 0) →
    4/b + 1/c ≤ 4/b' + 1/c') →
  4/b + 1/c = 9 :=
by sorry


end NUMINAMATH_CALUDE_min_value_line_circle_l2546_254649


namespace NUMINAMATH_CALUDE_total_good_balls_eq_144_l2546_254653

/-- The total number of soccer balls -/
def total_soccer_balls : ℕ := 180

/-- The total number of basketballs -/
def total_basketballs : ℕ := 75

/-- The total number of tennis balls -/
def total_tennis_balls : ℕ := 90

/-- The total number of volleyballs -/
def total_volleyballs : ℕ := 50

/-- The number of soccer balls with holes -/
def soccer_balls_with_holes : ℕ := 125

/-- The number of basketballs with holes -/
def basketballs_with_holes : ℕ := 49

/-- The number of tennis balls with holes -/
def tennis_balls_with_holes : ℕ := 62

/-- The number of deflated volleyballs -/
def deflated_volleyballs : ℕ := 15

/-- The total number of balls without holes or deflation -/
def total_good_balls : ℕ := 
  (total_soccer_balls - soccer_balls_with_holes) +
  (total_basketballs - basketballs_with_holes) +
  (total_tennis_balls - tennis_balls_with_holes) +
  (total_volleyballs - deflated_volleyballs)

theorem total_good_balls_eq_144 : total_good_balls = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_good_balls_eq_144_l2546_254653


namespace NUMINAMATH_CALUDE_triangle_side_b_l2546_254638

theorem triangle_side_b (a : ℝ) (A B : ℝ) (h1 : a = 5) (h2 : A = π/6) (h3 : Real.tan B = 3/4) :
  ∃ (b : ℝ), b = 6 ∧ (b / Real.sin B = a / Real.sin A) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_l2546_254638


namespace NUMINAMATH_CALUDE_square_root_equality_l2546_254665

theorem square_root_equality (x : ℝ) (a : ℝ) 
  (h_pos : x > 0) 
  (h1 : Real.sqrt x = 2 * a - 3) 
  (h2 : Real.sqrt x = 5 - a) : 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_square_root_equality_l2546_254665


namespace NUMINAMATH_CALUDE_five_by_five_not_coverable_l2546_254601

/-- Represents a checkerboard with width and height -/
structure Checkerboard :=
  (width : ℕ)
  (height : ℕ)

/-- Checks if a checkerboard can be covered by dominos -/
def can_be_covered_by_dominos (board : Checkerboard) : Prop :=
  (board.width * board.height) % 2 = 0 ∧
  (board.width * board.height) / 2 = (board.width * board.height + 1) / 2

theorem five_by_five_not_coverable :
  ¬(can_be_covered_by_dominos ⟨5, 5⟩) :=
by sorry

end NUMINAMATH_CALUDE_five_by_five_not_coverable_l2546_254601


namespace NUMINAMATH_CALUDE_age_difference_l2546_254641

/-- Given the ages of Mandy, her brother, and her sister, prove the age difference between Mandy and her sister. -/
theorem age_difference (mandy_age brother_age sister_age : ℕ) 
  (h1 : mandy_age = 3)
  (h2 : brother_age = 4 * mandy_age)
  (h3 : sister_age = brother_age - 5) :
  sister_age - mandy_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2546_254641
