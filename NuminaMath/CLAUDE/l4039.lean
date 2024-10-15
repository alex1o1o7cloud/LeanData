import Mathlib

namespace NUMINAMATH_CALUDE_dave_deleted_eleven_apps_l4039_403939

/-- The number of apps Dave initially had on his phone -/
def initial_apps : ℕ := 16

/-- The number of apps Dave had left after deletion -/
def remaining_apps : ℕ := 5

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := initial_apps - remaining_apps

theorem dave_deleted_eleven_apps : deleted_apps = 11 := by
  sorry

end NUMINAMATH_CALUDE_dave_deleted_eleven_apps_l4039_403939


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l4039_403965

theorem max_value_expression (y : ℝ) (h : y > 0) :
  (y^2 + 3 - Real.sqrt (y^4 + 9)) / y ≤ 6 / (2 * Real.sqrt 3 + Real.sqrt 6) :=
sorry

theorem max_value_achievable :
  ∃ y : ℝ, y > 0 ∧ (y^2 + 3 - Real.sqrt (y^4 + 9)) / y = 6 / (2 * Real.sqrt 3 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l4039_403965


namespace NUMINAMATH_CALUDE_game_result_l4039_403953

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [2, 6, 3, 2, 5]
def carl_rolls : List ℕ := [1, 4, 3, 6, 6]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result : total_points allie_rolls * total_points carl_rolls = 594 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l4039_403953


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l4039_403990

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 171 → n = 19 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l4039_403990


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l4039_403996

theorem lcm_gcf_ratio : (Nat.lcm 240 630) / (Nat.gcd 240 630) = 168 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l4039_403996


namespace NUMINAMATH_CALUDE_min_distance_sum_l4039_403948

/-- A rectangle with sides 20 cm and 10 cm -/
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (is_rectangle : sorry)
  (AB_length : dist A B = 20)
  (BC_length : dist B C = 10)

/-- The sum of distances BM + MN -/
def distance_sum (rect : Rectangle) (M : ℝ × ℝ) (N : ℝ × ℝ) : ℝ :=
  dist rect.B M + dist M N

/-- M is on AC -/
def M_on_AC (rect : Rectangle) (M : ℝ × ℝ) : Prop :=
  sorry

/-- N is on AB -/
def N_on_AB (rect : Rectangle) (N : ℝ × ℝ) : Prop :=
  sorry

theorem min_distance_sum (rect : Rectangle) :
  ∃ (M N : ℝ × ℝ), M_on_AC rect M ∧ N_on_AB rect N ∧
    (∀ (M' N' : ℝ × ℝ), M_on_AC rect M' → N_on_AB rect N' →
      distance_sum rect M N ≤ distance_sum rect M' N') ∧
    distance_sum rect M N = 16 :=
  sorry

end NUMINAMATH_CALUDE_min_distance_sum_l4039_403948


namespace NUMINAMATH_CALUDE_intersection_implies_solution_l4039_403972

-- Define the linear function
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem intersection_implies_solution (k b : ℝ) :
  linear_function k b (-3) = 0 →
  (∃ x : ℝ, -k * x + b = 0) ∧
  (∀ x : ℝ, -k * x + b = 0 → x = 3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_solution_l4039_403972


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_l4039_403951

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 3/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 3/b = 1 → x/2 + y/3 ≤ a/2 + b/3 :=
by sorry

theorem min_value_is_four (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 3/y = 1) :
  x/2 + y/3 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_four_l4039_403951


namespace NUMINAMATH_CALUDE_equation_solutions_l4039_403912

theorem equation_solutions : 
  let f (x : ℝ) := 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5))
  ∀ x : ℝ, f x = 1/8 ↔ x = 7 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4039_403912


namespace NUMINAMATH_CALUDE_total_cost_for_cakes_l4039_403907

/-- The number of cakes Claire wants to make -/
def num_cakes : ℕ := 2

/-- The number of packages of flour required for one cake -/
def packages_per_cake : ℕ := 2

/-- The cost of one package of flour in dollars -/
def cost_per_package : ℕ := 3

/-- Theorem: The total cost of flour for making 2 cakes is $12 -/
theorem total_cost_for_cakes : num_cakes * packages_per_cake * cost_per_package = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_for_cakes_l4039_403907


namespace NUMINAMATH_CALUDE_book_purchase_theorem_l4039_403967

/-- The number of people who purchased only book A -/
def Z : ℕ := 1000

/-- The number of people who purchased only book B -/
def X : ℕ := 250

/-- The number of people who purchased both books A and B -/
def Y : ℕ := 500

/-- The total number of people who purchased book A -/
def A : ℕ := Z + Y

/-- The total number of people who purchased book B -/
def B : ℕ := X + Y

theorem book_purchase_theorem :
  (A = 2 * B) ∧             -- The number of people who purchased book A is twice the number of people who purchased book B
  (Y = 500) ∧               -- The number of people who purchased both books A and B is 500
  (Y = 2 * X) ∧             -- The number of people who purchased both books A and B is twice the number of people who purchased only book B
  (Z = 1000) :=             -- The number of people who purchased only book A is 1000
by sorry

end NUMINAMATH_CALUDE_book_purchase_theorem_l4039_403967


namespace NUMINAMATH_CALUDE_regular_polygon_150_degree_angles_l4039_403923

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides -/
theorem regular_polygon_150_degree_angles (n : ℕ) : 
  (n ≥ 3) →                          -- A polygon has at least 3 sides
  (∀ i : ℕ, i < n → 150 = (n - 2) * 180 / n) →  -- Each interior angle is 150 degrees
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degree_angles_l4039_403923


namespace NUMINAMATH_CALUDE_second_grade_survey_size_l4039_403969

/-- Represents a school with three grades and a stratified sampling plan. -/
structure School where
  total_students : ℕ
  grade_ratio : Fin 3 → ℕ
  survey_size : ℕ

/-- Calculates the number of students to be surveyed from a specific grade. -/
def students_surveyed_in_grade (school : School) (grade : Fin 3) : ℕ :=
  (school.survey_size * school.grade_ratio grade) / (school.grade_ratio 0 + school.grade_ratio 1 + school.grade_ratio 2)

/-- The main theorem stating that 50 second-grade students should be surveyed. -/
theorem second_grade_survey_size (school : School) 
  (h1 : school.total_students = 1500)
  (h2 : school.grade_ratio 0 = 4)
  (h3 : school.grade_ratio 1 = 5)
  (h4 : school.grade_ratio 2 = 6)
  (h5 : school.survey_size = 150) :
  students_surveyed_in_grade school 1 = 50 := by
  sorry


end NUMINAMATH_CALUDE_second_grade_survey_size_l4039_403969


namespace NUMINAMATH_CALUDE_unique_zero_addition_l4039_403986

theorem unique_zero_addition (x : ℤ) :
  (∀ n : ℤ, n + x = n) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_addition_l4039_403986


namespace NUMINAMATH_CALUDE_nicky_profit_l4039_403933

def card_value_traded : ℕ := 8
def num_cards_traded : ℕ := 2
def card_value_received : ℕ := 21

def profit : ℕ := card_value_received - (card_value_traded * num_cards_traded)

theorem nicky_profit :
  profit = 5 := by sorry

end NUMINAMATH_CALUDE_nicky_profit_l4039_403933


namespace NUMINAMATH_CALUDE_equation_solution_l4039_403988

theorem equation_solution (a b c : ℝ) (h : 1 / a - 1 / b = 2 / c) : c = a * b * (b - a) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4039_403988


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l4039_403924

/-- Given a point M(-5, 2), its symmetric point with respect to the y-axis has coordinates (5, 2) -/
theorem symmetric_point_y_axis :
  let M : ℝ × ℝ := (-5, 2)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
  symmetric_point M = (5, 2) := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l4039_403924


namespace NUMINAMATH_CALUDE_electronics_store_profit_l4039_403943

theorem electronics_store_profit (n : ℕ) (CA : ℝ) : 
  let CB := 2 * CA
  let SA := (2 / 3) * CA
  let SB := 1.2 * CB
  let total_cost := n * CA + n * CB
  let total_sales := n * SA + n * SB
  (total_sales - total_cost) / total_cost = 0.1
  := by sorry

end NUMINAMATH_CALUDE_electronics_store_profit_l4039_403943


namespace NUMINAMATH_CALUDE_mr_mcpherson_contribution_l4039_403932

/-- Calculates the amount Mr. McPherson needs to raise for rent -/
theorem mr_mcpherson_contribution (total_rent : ℝ) (mrs_mcpherson_percentage : ℝ) :
  total_rent = 1200 →
  mrs_mcpherson_percentage = 30 →
  total_rent - (mrs_mcpherson_percentage / 100 * total_rent) = 840 := by
sorry

end NUMINAMATH_CALUDE_mr_mcpherson_contribution_l4039_403932


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l4039_403929

theorem rectangle_dimensions : ∃ (a b : ℝ), 
  b = a + 3 ∧ 
  2*a + 2*b + a = a*b ∧ 
  a = 3 ∧ 
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l4039_403929


namespace NUMINAMATH_CALUDE_cubic_equation_value_l4039_403991

theorem cubic_equation_value (x : ℝ) (h : x^2 + x - 2 = 0) :
  x^3 + 2*x^2 - x + 2021 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l4039_403991


namespace NUMINAMATH_CALUDE_same_color_probability_l4039_403956

/-- The probability of drawing two balls of the same color from a bag with 2 red and 2 white balls, with replacement -/
theorem same_color_probability (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 2 →
  white_balls = 2 →
  (red_balls : ℚ) / total_balls * (red_balls : ℚ) / total_balls +
  (white_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l4039_403956


namespace NUMINAMATH_CALUDE_fraction_simplification_l4039_403908

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4039_403908


namespace NUMINAMATH_CALUDE_mothers_day_rose_ratio_l4039_403968

/-- The number of roses Kyle picked last year -/
def last_year_roses : ℕ := 12

/-- The cost of one rose at the grocery store in dollars -/
def rose_cost : ℕ := 3

/-- The total amount Kyle spent on roses at the grocery store in dollars -/
def total_spent : ℕ := 54

/-- The ratio of roses in this year's bouquet to roses picked last year -/
def rose_ratio : Rat := 3 / 2

theorem mothers_day_rose_ratio :
  (total_spent / rose_cost : ℚ) / last_year_roses = rose_ratio :=
sorry

end NUMINAMATH_CALUDE_mothers_day_rose_ratio_l4039_403968


namespace NUMINAMATH_CALUDE_f_symmetry_l4039_403928

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 2

-- State the theorem
theorem f_symmetry (a b : ℝ) :
  f a b 2017 = 7 → f a b (-2017) = -11 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l4039_403928


namespace NUMINAMATH_CALUDE_hiker_final_distance_l4039_403998

-- Define the hiker's movements
def east_distance : ℝ := 15
def south_distance : ℝ := 20
def west_distance : ℝ := 15
def north_distance : ℝ := 5

-- Define the net horizontal and vertical movements
def net_horizontal : ℝ := east_distance - west_distance
def net_vertical : ℝ := south_distance - north_distance

-- Theorem to prove
theorem hiker_final_distance :
  Real.sqrt (net_horizontal ^ 2 + net_vertical ^ 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_hiker_final_distance_l4039_403998


namespace NUMINAMATH_CALUDE_integral_x_cos_x_plus_cube_root_x_squared_l4039_403978

open Real
open MeasureTheory
open Interval

theorem integral_x_cos_x_plus_cube_root_x_squared : 
  ∫ x in (-1)..1, (x * cos x + (x^2)^(1/3)) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_cos_x_plus_cube_root_x_squared_l4039_403978


namespace NUMINAMATH_CALUDE_delphine_chocolates_day1_l4039_403918

/-- Represents the number of chocolates Delphine ate on each day -/
structure ChocolatesEaten where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Theorem stating the number of chocolates Delphine ate on the first day -/
theorem delphine_chocolates_day1 (c : ChocolatesEaten) : c.day1 = 4 :=
  by
  have h1 : c.day2 = 2 * c.day1 - 3 := sorry
  have h2 : c.day3 = c.day1 - 2 := sorry
  have h3 : c.day4 = c.day3 - 1 := sorry
  have h4 : c.day1 + c.day2 + c.day3 + c.day4 + 12 = 24 := sorry
  sorry

#check delphine_chocolates_day1

end NUMINAMATH_CALUDE_delphine_chocolates_day1_l4039_403918


namespace NUMINAMATH_CALUDE_power_function_m_value_l4039_403942

/-- A function f(x) is a power function if it can be written in the form f(x) = ax^n, where a and n are constants and a ≠ 0. -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x ^ n

/-- Given that y = (m^2 - 3)x^(2m) is a power function, m equals ±2. -/
theorem power_function_m_value (m : ℝ) :
  IsPowerFunction (fun x => (m^2 - 3) * x^(2*m)) → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_m_value_l4039_403942


namespace NUMINAMATH_CALUDE_walter_age_2005_l4039_403944

theorem walter_age_2005 (walter_age_2000 : ℕ) (grandmother_age_2000 : ℕ) : 
  walter_age_2000 = grandmother_age_2000 / 3 →
  (2000 - walter_age_2000) + (2000 - grandmother_age_2000) = 3896 →
  walter_age_2000 + 5 = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_walter_age_2005_l4039_403944


namespace NUMINAMATH_CALUDE_power_of_negative_square_l4039_403904

theorem power_of_negative_square (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_square_l4039_403904


namespace NUMINAMATH_CALUDE_right_triangle_abc_area_l4039_403925

/-- A right triangle ABC in the xy-plane with specific properties -/
structure RightTriangleABC where
  -- Point A
  a : ℝ × ℝ
  -- Point B
  b : ℝ × ℝ
  -- Point C (right angle)
  c : ℝ × ℝ
  -- Hypotenuse length
  ab_length : ℝ
  -- Median through A equation
  median_a_slope : ℝ
  median_a_intercept : ℝ
  -- Median through B equation
  median_b_slope : ℝ
  median_b_intercept : ℝ
  -- Conditions
  right_angle_at_c : (a.1 - c.1) * (b.1 - c.1) + (a.2 - c.2) * (b.2 - c.2) = 0
  hypotenuse_length : (a.1 - b.1)^2 + (a.2 - b.2)^2 = ab_length^2
  median_a_equation : ∀ x y, y = median_a_slope * x + median_a_intercept → 
    2 * x = a.1 + c.1 ∧ 2 * y = a.2 + c.2
  median_b_equation : ∀ x y, y = median_b_slope * x + median_b_intercept → 
    2 * x = b.1 + c.1 ∧ 2 * y = b.2 + c.2

/-- The area of the right triangle ABC with given properties is 175 -/
theorem right_triangle_abc_area 
  (t : RightTriangleABC) 
  (h1 : t.ab_length = 50) 
  (h2 : t.median_a_slope = 1 ∧ t.median_a_intercept = 5)
  (h3 : t.median_b_slope = 2 ∧ t.median_b_intercept = 6) :
  abs ((t.a.1 * t.b.2 - t.b.1 * t.a.2) / 2) = 175 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_abc_area_l4039_403925


namespace NUMINAMATH_CALUDE_transaction_result_l4039_403974

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  carValue : Int
  hascar : Bool

/-- Represents a car transaction between two people -/
def carTransaction (buyer seller : FinancialState) (price : Int) : FinancialState × FinancialState :=
  let newBuyer : FinancialState := {
    cash := buyer.cash - price,
    carValue := seller.carValue,
    hascar := true
  }
  let newSeller : FinancialState := {
    cash := seller.cash + price,
    carValue := 0,
    hascar := false
  }
  (newBuyer, newSeller)

/-- Calculates the net worth of a person -/
def netWorth (state : FinancialState) : Int :=
  state.cash + (if state.hascar then state.carValue else 0)

theorem transaction_result (initialCarValue : Int) :
  let mrAInitial : FinancialState := { cash := 8000, carValue := initialCarValue, hascar := true }
  let mrBInitial : FinancialState := { cash := 9000, carValue := 0, hascar := false }
  let (mrBAfterFirst, mrAAfterFirst) := carTransaction mrBInitial mrAInitial 10000
  let (mrAFinal, mrBFinal) := carTransaction mrAAfterFirst mrBAfterFirst 7000
  (netWorth mrAFinal - netWorth mrAInitial = 3000) ∧
  (netWorth mrBFinal - netWorth mrBInitial = -3000) :=
by
  sorry


end NUMINAMATH_CALUDE_transaction_result_l4039_403974


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_A_subset_C_implies_a_greater_than_seven_l4039_403927

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem for question 1
theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

-- Theorem for question 2
theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by sorry

-- Theorem for question 3
theorem A_subset_C_implies_a_greater_than_seven (a : ℝ) : 
  A ⊆ C a → a > 7 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_A_subset_C_implies_a_greater_than_seven_l4039_403927


namespace NUMINAMATH_CALUDE_ratio_fifth_to_first_l4039_403915

/-- An arithmetic sequence with a non-zero common difference where a₁, a₂, and a₅ form a geometric sequence. -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : (a 2) ^ 2 = a 1 * a 5

/-- The ratio of the fifth term to the first term in the special arithmetic sequence is 9. -/
theorem ratio_fifth_to_first (seq : ArithmeticSequence) : seq.a 5 / seq.a 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fifth_to_first_l4039_403915


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l4039_403935

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 0) :
  Complex.abs (a + b + c) = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l4039_403935


namespace NUMINAMATH_CALUDE_exists_quadrilateral_equal_angle_tangents_l4039_403920

/-- A planar quadrilateral is represented by its four interior angles -/
structure PlanarQuadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 360

/-- Theorem: There exists a planar quadrilateral where the tangents of all its interior angles are equal -/
theorem exists_quadrilateral_equal_angle_tangents : 
  ∃ q : PlanarQuadrilateral, Real.tan q.α = Real.tan q.β ∧ Real.tan q.β = Real.tan q.γ ∧ Real.tan q.γ = Real.tan q.δ :=
sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_equal_angle_tangents_l4039_403920


namespace NUMINAMATH_CALUDE_three_integer_solutions_quadratic_inequality_l4039_403916

theorem three_integer_solutions_quadratic_inequality (b : ℤ) : 
  (∃! n : ℕ, n = 2 ∧ 
    (∃ s : Finset ℤ, s.card = n ∧ 
      (∀ b' ∈ s, (∃! t : Finset ℤ, t.card = 3 ∧ 
        (∀ x ∈ t, x^2 + b' * x + 6 ≤ 0) ∧ 
        (∀ x : ℤ, x^2 + b' * x + 6 ≤ 0 → x ∈ t))))) :=
sorry

end NUMINAMATH_CALUDE_three_integer_solutions_quadratic_inequality_l4039_403916


namespace NUMINAMATH_CALUDE_distance_from_origin_l4039_403995

theorem distance_from_origin (x y : ℝ) (h1 : |x| = 8) 
  (h2 : Real.sqrt ((x - 7)^2 + (y - 3)^2) = 8) (h3 : y > 3) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (136 + 6 * Real.sqrt 63) :=
by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l4039_403995


namespace NUMINAMATH_CALUDE_system_solution_l4039_403980

theorem system_solution :
  ∃ (m n : ℚ), m / 3 + n / 2 = 1 ∧ m - 2 * n = 2 ∧ m = 18 / 7 ∧ n = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4039_403980


namespace NUMINAMATH_CALUDE_three_prime_pairs_sum_52_l4039_403938

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given number -/
def count_prime_pairs (sum : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (sum - p)) (Finset.range (sum / 2 + 1))).card / 2

/-- Theorem stating that there are exactly 3 unordered pairs of prime numbers that sum to 52 -/
theorem three_prime_pairs_sum_52 : count_prime_pairs 52 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_prime_pairs_sum_52_l4039_403938


namespace NUMINAMATH_CALUDE_sams_calculation_l4039_403913

theorem sams_calculation (x y : ℝ) : 
  x + 2 * 2 + y = x * 2 + 2 + y → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sams_calculation_l4039_403913


namespace NUMINAMATH_CALUDE_bono_jelly_beans_l4039_403984

/-- Given the number of jelly beans for Alida, Bono, and Cate, prove that Bono has 4t - 1 jelly beans. -/
theorem bono_jelly_beans (t : ℕ) (A B C : ℕ) : 
  A + B = 6 * t + 3 →
  A + C = 4 * t + 5 →
  B + C = 6 * t →
  B = 4 * t - 1 := by
  sorry

end NUMINAMATH_CALUDE_bono_jelly_beans_l4039_403984


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l4039_403955

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - x + m^2 - 4 = 0 ∧ x = 1) → (m = 2 ∨ m = -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l4039_403955


namespace NUMINAMATH_CALUDE_sum_of_squares_l4039_403952

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 4)
  (eq2 : y^2 - 5*z = 5)
  (eq3 : z^2 - 7*x = -8) :
  x^2 + y^2 + z^2 = 83/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l4039_403952


namespace NUMINAMATH_CALUDE_no_valid_labeling_l4039_403945

-- Define a type for the vertices of a tetrahedron
inductive Vertex : Type
| A : Vertex
| B : Vertex
| C : Vertex
| D : Vertex

-- Define a type for the faces of a tetrahedron
inductive Face : Type
| ABC : Face
| ABD : Face
| ACD : Face
| BCD : Face

-- Define a labeling function
def Labeling := Vertex → Fin 4

-- Define a function to get the sum of a face given a labeling
def faceSum (l : Labeling) (f : Face) : Nat :=
  match f with
  | Face.ABC => (l Vertex.A).val + (l Vertex.B).val + (l Vertex.C).val
  | Face.ABD => (l Vertex.A).val + (l Vertex.B).val + (l Vertex.D).val
  | Face.ACD => (l Vertex.A).val + (l Vertex.C).val + (l Vertex.D).val
  | Face.BCD => (l Vertex.B).val + (l Vertex.C).val + (l Vertex.D).val

-- Define a predicate for a valid labeling
def isValidLabeling (l : Labeling) : Prop :=
  (∀ (v1 v2 : Vertex), v1 ≠ v2 → l v1 ≠ l v2) ∧
  (∀ (f1 f2 : Face), faceSum l f1 = faceSum l f2)

-- Theorem: There are no valid labelings
theorem no_valid_labeling : ¬∃ (l : Labeling), isValidLabeling l := by
  sorry


end NUMINAMATH_CALUDE_no_valid_labeling_l4039_403945


namespace NUMINAMATH_CALUDE_dana_marcus_pencil_difference_l4039_403934

/-- Given that Dana has 15 more pencils than Jayden, Jayden has twice as many pencils as Marcus,
    and Jayden has 20 pencils, prove that Dana has 25 more pencils than Marcus. -/
theorem dana_marcus_pencil_difference :
  ∀ (dana jayden marcus : ℕ),
  dana = jayden + 15 →
  jayden = 2 * marcus →
  jayden = 20 →
  dana - marcus = 25 := by
sorry

end NUMINAMATH_CALUDE_dana_marcus_pencil_difference_l4039_403934


namespace NUMINAMATH_CALUDE_range_of_x_when_m_is_one_range_of_m_for_sufficient_condition_l4039_403903

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 - 10*x + 16 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 ≤ 0

-- Theorem for part (1)
theorem range_of_x_when_m_is_one (x : ℝ) :
  (∃ m : ℝ, m = 1 ∧ m > 0 ∧ (p x ∨ q x m)) → x ∈ Set.Icc 1 8 :=
sorry

-- Theorem for part (2)
theorem range_of_m_for_sufficient_condition (m : ℝ) :
  (m > 0 ∧ (∀ x : ℝ, q x m → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x m)) →
  m ∈ Set.Icc 2 (8/3) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_m_is_one_range_of_m_for_sufficient_condition_l4039_403903


namespace NUMINAMATH_CALUDE_jacket_selling_price_l4039_403930

/-- Calculates the total selling price of a jacket given the original price,
    discount rate, tax rate, and processing fee. -/
def total_selling_price (original_price discount_rate tax_rate processing_fee : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_tax := discounted_price * (1 + tax_rate)
  price_with_tax + processing_fee

/-- Theorem stating that the total selling price of the jacket is $95.72 -/
theorem jacket_selling_price :
  total_selling_price 120 0.30 0.08 5 = 95.72 := by
  sorry

#eval total_selling_price 120 0.30 0.08 5

end NUMINAMATH_CALUDE_jacket_selling_price_l4039_403930


namespace NUMINAMATH_CALUDE_semicircle_radius_l4039_403992

theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 162) :
  ∃ (radius : ℝ), perimeter = radius * (Real.pi + 2) ∧ radius = 162 / (Real.pi + 2) := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l4039_403992


namespace NUMINAMATH_CALUDE_randy_lunch_cost_l4039_403982

theorem randy_lunch_cost (initial_amount : ℝ) (ice_cream_cost : ℝ) : 
  initial_amount = 30 →
  ice_cream_cost = 5 →
  ∃ (lunch_cost : ℝ),
    lunch_cost = 10 ∧
    (1/4) * (initial_amount - lunch_cost) = ice_cream_cost :=
by
  sorry

end NUMINAMATH_CALUDE_randy_lunch_cost_l4039_403982


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l4039_403999

theorem gcd_digits_bound (a b : ℕ) (ha : a < 100000) (hb : b < 100000)
  (hlcm : 10000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 100000000) :
  Nat.gcd a b < 1000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l4039_403999


namespace NUMINAMATH_CALUDE_smoothie_price_l4039_403946

theorem smoothie_price (cake_price : ℚ) (smoothies_sold : ℕ) (cakes_sold : ℕ) (total_revenue : ℚ) :
  cake_price = 2 →
  smoothies_sold = 40 →
  cakes_sold = 18 →
  total_revenue = 156 →
  ∃ (smoothie_price : ℚ), smoothie_price * smoothies_sold + cake_price * cakes_sold = total_revenue ∧ smoothie_price = 3 :=
by sorry

end NUMINAMATH_CALUDE_smoothie_price_l4039_403946


namespace NUMINAMATH_CALUDE_divide_by_fraction_main_proof_l4039_403957

theorem divide_by_fraction (a b c : ℝ) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b :=
by sorry

theorem main_proof : (5 : ℝ) / ((7 : ℝ) / 3) = 15 / 7 :=
by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_main_proof_l4039_403957


namespace NUMINAMATH_CALUDE_sin_five_half_pi_plus_alpha_l4039_403994

theorem sin_five_half_pi_plus_alpha (α : ℝ) : 
  Real.sin ((5 / 2) * Real.pi + α) = Real.cos α := by sorry

end NUMINAMATH_CALUDE_sin_five_half_pi_plus_alpha_l4039_403994


namespace NUMINAMATH_CALUDE_a_grazing_months_l4039_403976

/-- Represents the number of months 'a' put his oxen for grazing -/
def a_months : ℕ := sorry

/-- Represents the number of oxen 'a' put for grazing -/
def a_oxen : ℕ := 10

/-- Represents the number of oxen 'b' put for grazing -/
def b_oxen : ℕ := 12

/-- Represents the number of months 'b' put his oxen for grazing -/
def b_months : ℕ := 5

/-- Represents the number of oxen 'c' put for grazing -/
def c_oxen : ℕ := 15

/-- Represents the number of months 'c' put his oxen for grazing -/
def c_months : ℕ := 3

/-- Represents the total rent of the pasture in Rs. -/
def total_rent : ℕ := 105

/-- Represents 'c's share of the rent in Rs. -/
def c_share : ℕ := 27

/-- Theorem stating that 'a' put his oxen for grazing for 7 months -/
theorem a_grazing_months : a_months = 7 := by sorry

end NUMINAMATH_CALUDE_a_grazing_months_l4039_403976


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4039_403947

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₄ = 5 and a₉ = 17, then a₁₄ = 29. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arithmetic : IsArithmeticSequence a)
    (h_a4 : a 4 = 5)
    (h_a9 : a 9 = 17) : 
  a 14 = 29 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4039_403947


namespace NUMINAMATH_CALUDE_problem_solution_l4039_403977

noncomputable def θ : ℝ := sorry

-- The terminal side of angle θ lies on the ray y = 2x (x ≥ 0)
axiom h : ∀ x : ℝ, x ≥ 0 → Real.tan θ * x = 2 * x

theorem problem_solution :
  (Real.tan θ = 2) ∧
  ((2 * Real.cos θ + 3 * Real.sin θ) / (Real.cos θ - 3 * Real.sin θ) + Real.sin θ * Real.cos θ = -6/5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l4039_403977


namespace NUMINAMATH_CALUDE_arkos_population_2070_l4039_403959

def population_growth (initial_population : ℕ) (growth_factor : ℕ) (years : ℕ) : ℕ :=
  initial_population * growth_factor ^ (years / 10)

theorem arkos_population_2070 :
  let initial_population := 250
  let years := 50
  let growth_factor := 2
  population_growth initial_population growth_factor years = 8000 := by
sorry

end NUMINAMATH_CALUDE_arkos_population_2070_l4039_403959


namespace NUMINAMATH_CALUDE_x_in_terms_of_y_l4039_403914

theorem x_in_terms_of_y (x y : ℝ) (h : x / (x - 3) = (y^2 + 3*y + 1) / (y^2 + 3*y - 4)) :
  x = (3*y^2 + 9*y + 3) / 5 := by
sorry

end NUMINAMATH_CALUDE_x_in_terms_of_y_l4039_403914


namespace NUMINAMATH_CALUDE_percentage_difference_l4039_403987

theorem percentage_difference (x y : ℝ) (h : x = 8 * y) :
  (x - y) / x * 100 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l4039_403987


namespace NUMINAMATH_CALUDE_n_value_is_six_l4039_403983

/-- The cost of a water bottle in cents -/
def water_cost : ℕ := 50

/-- The cost of a fruit in cents -/
def fruit_cost : ℕ := 25

/-- The cost of a snack in cents -/
def snack_cost : ℕ := 100

/-- The number of water bottles in a bundle -/
def water_in_bundle : ℕ := 1

/-- The number of snacks in a bundle -/
def snacks_in_bundle : ℕ := 3

/-- The number of fruits in a bundle -/
def fruits_in_bundle : ℕ := 2

/-- The regular selling price of a bundle in cents -/
def bundle_price : ℕ := 460

/-- The special price for every nth bundle in cents -/
def special_price : ℕ := 200

/-- The function to calculate the cost of a regular bundle in cents -/
def bundle_cost : ℕ := 
  water_cost * water_in_bundle + 
  snack_cost * snacks_in_bundle + 
  fruit_cost * fruits_in_bundle

/-- The function to calculate the profit from a regular bundle in cents -/
def bundle_profit : ℕ := bundle_price - bundle_cost

/-- The function to calculate the cost of a special bundle in cents -/
def special_bundle_cost : ℕ := bundle_cost + snack_cost

/-- The function to calculate the loss from a special bundle in cents -/
def special_bundle_loss : ℕ := special_bundle_cost - special_price

/-- Theorem stating that the value of n is 6 -/
theorem n_value_is_six : 
  ∃ n : ℕ, n > 0 ∧ (n - 1) * bundle_profit = special_bundle_loss ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_n_value_is_six_l4039_403983


namespace NUMINAMATH_CALUDE_speed_gain_per_week_l4039_403937

def initial_speed : ℝ := 80
def training_weeks : ℕ := 16
def speed_increase_percentage : ℝ := 0.20

theorem speed_gain_per_week :
  let final_speed := initial_speed * (1 + speed_increase_percentage)
  let total_speed_gain := final_speed - initial_speed
  let speed_gain_per_week := total_speed_gain / training_weeks
  speed_gain_per_week = 1 := by
  sorry

end NUMINAMATH_CALUDE_speed_gain_per_week_l4039_403937


namespace NUMINAMATH_CALUDE_cubic_root_odd_and_increasing_l4039_403973

-- Define the function
def f (x : ℝ) : ℝ := x^(1/3)

-- State the theorem
theorem cubic_root_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_odd_and_increasing_l4039_403973


namespace NUMINAMATH_CALUDE_integers_between_neg_sqrt2_and_sqrt2_l4039_403921

theorem integers_between_neg_sqrt2_and_sqrt2 :
  {x : ℤ | -Real.sqrt 2 < x ∧ x < Real.sqrt 2} = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_integers_between_neg_sqrt2_and_sqrt2_l4039_403921


namespace NUMINAMATH_CALUDE_miriam_flowers_per_day_l4039_403902

/-- The number of flowers Miriam can take care of in 6 days -/
def total_flowers : ℕ := 360

/-- The number of days Miriam works -/
def work_days : ℕ := 6

/-- The number of flowers Miriam can take care of in one day -/
def flowers_per_day : ℕ := total_flowers / work_days

theorem miriam_flowers_per_day : flowers_per_day = 60 := by
  sorry

end NUMINAMATH_CALUDE_miriam_flowers_per_day_l4039_403902


namespace NUMINAMATH_CALUDE_yellow_balls_count_l4039_403931

theorem yellow_balls_count (red_balls : ℕ) (probability_red : ℚ) (yellow_balls : ℕ) : 
  red_balls = 10 →
  probability_red = 2/5 →
  (red_balls : ℚ) / ((red_balls : ℚ) + (yellow_balls : ℚ)) = probability_red →
  yellow_balls = 15 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l4039_403931


namespace NUMINAMATH_CALUDE_imaginary_part_of_3_minus_4i_l4039_403964

theorem imaginary_part_of_3_minus_4i :
  Complex.im (3 - 4 * Complex.I) = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_3_minus_4i_l4039_403964


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l4039_403911

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 + 3 / (a - 2)) / ((a^2 + 2*a + 1) / (a - 2)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l4039_403911


namespace NUMINAMATH_CALUDE_circles_covering_path_implies_odd_l4039_403961

/-- A configuration of n circles on a plane. -/
structure CircleConfiguration (n : ℕ) where
  /-- The set of circles. -/
  circles : Fin n → Set (ℝ × ℝ)
  /-- Any two circles intersect at exactly two points. -/
  two_intersections : ∀ (i j : Fin n), i ≠ j → ∃! (p q : ℝ × ℝ), p ≠ q ∧ p ∈ circles i ∧ p ∈ circles j ∧ q ∈ circles i ∧ q ∈ circles j
  /-- No three circles have a common point. -/
  no_triple_intersection : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ¬∃ (p : ℝ × ℝ), p ∈ circles i ∧ p ∈ circles j ∧ p ∈ circles k

/-- A path that covers all circles in the configuration. -/
def CoveringPath (n : ℕ) (config : CircleConfiguration n) :=
  ∃ (path : ℕ → Fin n), ∀ (i : Fin n), ∃ (k : ℕ), path k = i

/-- The main theorem: if there exists a covering path for n circles satisfying the given conditions,
    then n must be odd. -/
theorem circles_covering_path_implies_odd (n : ℕ) (config : CircleConfiguration n) :
  CoveringPath n config → Odd n :=
sorry

end NUMINAMATH_CALUDE_circles_covering_path_implies_odd_l4039_403961


namespace NUMINAMATH_CALUDE_sequence_equals_primes_l4039_403993

theorem sequence_equals_primes (a p : ℕ → ℕ) :
  (∀ n, 0 < a n) →
  (∀ n k, n < k → a n < a k) →
  (∀ n, Nat.Prime (p n)) →
  (∀ n, p n ∣ a n) →
  (∀ n k, a n - a k = p n - p k) →
  ∀ n, a n = p n :=
by sorry

end NUMINAMATH_CALUDE_sequence_equals_primes_l4039_403993


namespace NUMINAMATH_CALUDE_number_wall_m_equals_one_l4039_403975

/-- Represents a simplified version of the number wall structure -/
structure NumberWall where
  m : ℤ
  top : ℤ
  left : ℤ
  right : ℤ

/-- The number wall satisfies the given conditions -/
def valid_wall (w : NumberWall) : Prop :=
  w.top = w.left + w.right ∧ w.left = w.m + 22 ∧ w.right = 35 ∧ w.top = 58

/-- Theorem: In the given number wall structure, m = 1 -/
theorem number_wall_m_equals_one (w : NumberWall) (h : valid_wall w) : w.m = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_m_equals_one_l4039_403975


namespace NUMINAMATH_CALUDE_valid_assignment_example_l4039_403940

def is_variable (s : String) : Prop := s.length > 0 ∧ s.all Char.isAlpha

def is_expression (s : String) : Prop := s.length > 0

def is_valid_assignment (s : String) : Prop :=
  ∃ (lhs rhs : String),
    s = lhs ++ " = " ++ rhs ∧
    is_variable lhs ∧
    is_expression rhs

theorem valid_assignment_example :
  is_valid_assignment "A = A*A + A - 2" := by sorry

end NUMINAMATH_CALUDE_valid_assignment_example_l4039_403940


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l4039_403960

theorem bowling_ball_weight (kayak_weight : ℝ) (ball_weight : ℝ) :
  kayak_weight = 36 →
  9 * ball_weight = 2 * kayak_weight →
  ball_weight = 8 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l4039_403960


namespace NUMINAMATH_CALUDE_construct_octagon_from_square_l4039_403910

/-- A square sheet of paper --/
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

/-- A regular octagon --/
structure RegularOctagon :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Represents the ability to fold paper --/
def can_fold : Prop := True

/-- Represents the ability to cut along creases --/
def can_cut_along_creases : Prop := True

/-- Represents the prohibition of using a compass --/
def no_compass : Prop := True

/-- Represents the prohibition of using a ruler --/
def no_ruler : Prop := True

/-- Theorem stating that a regular octagon can be constructed from a square sheet of paper --/
theorem construct_octagon_from_square 
  (s : Square) 
  (fold : can_fold) 
  (cut : can_cut_along_creases) 
  (no_compass : no_compass) 
  (no_ruler : no_ruler) : 
  ∃ (o : RegularOctagon), True :=
sorry

end NUMINAMATH_CALUDE_construct_octagon_from_square_l4039_403910


namespace NUMINAMATH_CALUDE_range_of_t_t_value_for_diameter_6_l4039_403901

-- Define the equation of the circle
def circle_equation (x y t : ℝ) : Prop :=
  x^2 + y^2 + (Real.sqrt 3 * t + 1) * x + t * y + t^2 - 2 = 0

-- Theorem for the range of t
theorem range_of_t :
  ∀ t : ℝ, (∃ x y : ℝ, circle_equation x y t) → t > -(3 * Real.sqrt 3) / 2 :=
sorry

-- Theorem for the value of t when diameter is 6
theorem t_value_for_diameter_6 :
  ∃! t : ℝ, (∃ x y : ℝ, circle_equation x y t) ∧ 
  (∃ x₁ y₁ x₂ y₂ : ℝ, circle_equation x₁ y₁ t ∧ circle_equation x₂ y₂ t ∧ 
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6) ∧
  t = (9 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_t_value_for_diameter_6_l4039_403901


namespace NUMINAMATH_CALUDE_baseball_cards_equality_l4039_403936

theorem baseball_cards_equality (J M C : ℕ) : 
  C = 20 → 
  M = C - 6 → 
  J + M + C = 48 → 
  J = M := by sorry

end NUMINAMATH_CALUDE_baseball_cards_equality_l4039_403936


namespace NUMINAMATH_CALUDE_inequality_chain_l4039_403954

theorem inequality_chain (a b x : ℝ) (h1 : 0 < b) (h2 : b < x) (h3 : x < a) :
  b * x < x^2 ∧ x^2 < a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l4039_403954


namespace NUMINAMATH_CALUDE_floor_expression_l4039_403900

theorem floor_expression (n : ℕ) (h : n = 2009) : 
  ⌊((n + 1)^3 / ((n - 1) * n : ℝ) - (n - 1)^3 / (n * (n + 1) : ℝ))⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_l4039_403900


namespace NUMINAMATH_CALUDE_tank_capacity_l4039_403997

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 1/4 →
  final_fraction = 2/3 →
  added_amount = 180 →
  (final_fraction - initial_fraction) * (added_amount / (final_fraction - initial_fraction)) = 432 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l4039_403997


namespace NUMINAMATH_CALUDE_largest_fraction_l4039_403970

theorem largest_fraction (a b c d : ℝ) 
  (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d)
  (h5 : a = 2) (h6 : b = 3) (h7 : c = 5) (h8 : d = 8) :
  (c + d) / (a + b) = max ((a + b) / (c + d)) 
                         (max ((a + d) / (b + c)) 
                              (max ((b + c) / (a + d)) 
                                   ((b + d) / (a + c)))) := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l4039_403970


namespace NUMINAMATH_CALUDE_range_of_f_when_k_4_range_of_k_for_monotone_f_l4039_403950

/-- The function f(x) = (k-2)x^2 + 2kx - 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 + 2 * k * x - 3

/-- The range of f(x) when k = 4 in the interval (-4, 1) -/
theorem range_of_f_when_k_4 :
  Set.Icc (-11 : ℝ) 7 = Set.image (f 4) (Set.Ioo (-4 : ℝ) 1) := by sorry

/-- The range of k for which f(x) is monotonically increasing in [1, 2] -/
theorem range_of_k_for_monotone_f :
  ∀ k : ℝ, (∀ x y : ℝ, x ∈ Set.Icc (1 : ℝ) 2 → y ∈ Set.Icc (1 : ℝ) 2 → x ≤ y → f k x ≤ f k y) ↔
  k ∈ Set.Ici (4/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_f_when_k_4_range_of_k_for_monotone_f_l4039_403950


namespace NUMINAMATH_CALUDE_problem_solution_l4039_403971

theorem problem_solution (m : ℝ) (h : m + 1/m = 10) : 
  m^2 + 1/m^2 + m + 1/m = 108 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4039_403971


namespace NUMINAMATH_CALUDE_boat_trip_distance_l4039_403949

/-- Proves that given a boat with speed 9 kmph in standing water, a stream with speed 1.5 kmph,
    and a round trip time of 48 hours, the distance to the destination is 210 km. -/
theorem boat_trip_distance (boat_speed : ℝ) (stream_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  boat_speed = 9 →
  stream_speed = 1.5 →
  total_time = 48 →
  distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time →
  distance = 210 := by
  sorry

end NUMINAMATH_CALUDE_boat_trip_distance_l4039_403949


namespace NUMINAMATH_CALUDE_unique_positive_solution_l4039_403966

theorem unique_positive_solution : 
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 - 7 * x - 6 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l4039_403966


namespace NUMINAMATH_CALUDE_no_integer_roots_l4039_403989

theorem no_integer_roots : ∀ x : ℤ, x^2 + 2^2018 * x + 2^2019 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l4039_403989


namespace NUMINAMATH_CALUDE_propositions_truth_values_l4039_403958

theorem propositions_truth_values :
  (∃ a b : ℝ, a + b < 2 * Real.sqrt (a * b)) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (1/x + 9/y = 1) ∧ (x + y < 16)) ∧
  (∀ x : ℝ, x^2 + 4/x^2 ≥ 4) ∧
  (∀ a b : ℝ, (a * b > 0) → (b/a + a/b ≥ 2)) :=
by sorry


end NUMINAMATH_CALUDE_propositions_truth_values_l4039_403958


namespace NUMINAMATH_CALUDE_hyperbola_foci_intersection_l4039_403922

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the circle with diameter equal to the distance between its foci
    intersects one of its asymptotes at the point (3,4),
    then a = 3 and b = 4. -/
theorem hyperbola_foci_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2) →
  (∃ (x y : ℝ), x^2 + y^2 = c^2 ∧ y/x = b/a ∧ x = 3 ∧ y = 4) →
  a = 3 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_intersection_l4039_403922


namespace NUMINAMATH_CALUDE_kylie_made_five_bracelets_l4039_403917

/-- The number of beaded bracelets Kylie made on Wednesday -/
def bracelets_made_wednesday (
  monday_necklaces : ℕ)
  (tuesday_necklaces : ℕ)
  (wednesday_earrings : ℕ)
  (beads_per_necklace : ℕ)
  (beads_per_bracelet : ℕ)
  (beads_per_earring : ℕ)
  (total_beads_used : ℕ) : ℕ :=
  (total_beads_used - 
   (monday_necklaces + tuesday_necklaces) * beads_per_necklace - 
   wednesday_earrings * beads_per_earring) / 
  beads_per_bracelet

/-- Theorem stating that Kylie made 5 beaded bracelets on Wednesday -/
theorem kylie_made_five_bracelets : 
  bracelets_made_wednesday 10 2 7 20 10 5 325 = 5 := by
  sorry

end NUMINAMATH_CALUDE_kylie_made_five_bracelets_l4039_403917


namespace NUMINAMATH_CALUDE_correct_calculation_l4039_403941

/-- Represents the loan and investment scenario -/
structure LoanInvestment where
  loan_amount : ℝ
  interest_paid : ℝ
  business_profit_rate : ℝ
  total_profit : ℝ

/-- Calculates the interest rate and investment amount -/
def calculate_rate_and_investment (scenario : LoanInvestment) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correctness of the calculation -/
theorem correct_calculation (scenario : LoanInvestment) :
  scenario.loan_amount = 150000 ∧
  scenario.interest_paid = 42000 ∧
  scenario.business_profit_rate = 0.1 ∧
  scenario.total_profit = 25000 →
  let (rate, investment) := calculate_rate_and_investment scenario
  rate = 0.05 ∧ investment = 50000 :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l4039_403941


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l4039_403985

/-- For a normal distribution with mean 16.5 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 13.5. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (h_μ : μ = 16.5) (h_σ : σ = 1.5) :
  μ - 2 * σ = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l4039_403985


namespace NUMINAMATH_CALUDE_smallest_b_value_l4039_403926

theorem smallest_b_value (k a b : ℝ) (h1 : k > 1) (h2 : k < a) (h3 : a < b)
  (h4 : k + a ≤ b) (h5 : 1/a + 1/b ≤ 1/k) : b ≥ 2*k := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_l4039_403926


namespace NUMINAMATH_CALUDE_specific_box_volume_l4039_403909

/-- The volume of an open box constructed from a rectangular sheet --/
def box_volume (sheet_length sheet_width y : ℝ) : ℝ :=
  (sheet_length - 2 * y) * (sheet_width - 2 * y) * y

/-- Theorem stating the volume of the specific box described in the problem --/
theorem specific_box_volume (y : ℝ) :
  box_volume 15 12 y = 180 * y - 54 * y^2 + 4 * y^3 :=
by sorry

end NUMINAMATH_CALUDE_specific_box_volume_l4039_403909


namespace NUMINAMATH_CALUDE_f_is_even_l4039_403919

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Theorem: f is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l4039_403919


namespace NUMINAMATH_CALUDE_trapezoid_sides_l4039_403963

/-- Proves that a trapezoid with given area, height, and difference between parallel sides has specific lengths for its parallel sides -/
theorem trapezoid_sides (area : ℝ) (height : ℝ) (side_diff : ℝ) 
  (h_area : area = 594) 
  (h_height : height = 22) 
  (h_side_diff : side_diff = 6) :
  ∃ (a b : ℝ), 
    (a + b) * height / 2 = area ∧ 
    a - b = side_diff ∧ 
    a = 30 ∧ 
    b = 24 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_sides_l4039_403963


namespace NUMINAMATH_CALUDE_system_solution_l4039_403906

theorem system_solution (x y z t : ℝ) : 
  (x * y - t^2 = 9 ∧ x^2 + y^2 + z^2 = 18) → 
  ((x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ (x = -3 ∧ y = -3 ∧ z = 0 ∧ t = 0)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4039_403906


namespace NUMINAMATH_CALUDE_michael_digging_time_l4039_403981

/-- Given the conditions of Michael and his father's digging, prove that Michael will take 700 hours to dig his hole. -/
theorem michael_digging_time (father_rate : ℝ) (father_time : ℝ) (depth_difference : ℝ) :
  father_rate = 4 →
  father_time = 400 →
  depth_difference = 400 →
  let father_depth := father_rate * father_time
  let michael_depth := 2 * father_depth - depth_difference
  michael_depth / father_rate = 700 := by
  sorry

end NUMINAMATH_CALUDE_michael_digging_time_l4039_403981


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l4039_403962

theorem cube_volume_from_surface_area :
  ∀ (surface_area : ℝ) (volume : ℝ),
    surface_area = 384 →
    volume = (surface_area / 6) ^ (3/2) →
    volume = 512 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l4039_403962


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l4039_403905

/-- Calculates the area of a square sheet of wrapping paper needed to wrap a rectangular box -/
theorem wrapping_paper_area (box_length box_width box_height extra_fold : ℝ) :
  box_length = 10 ∧ box_width = 10 ∧ box_height = 5 ∧ extra_fold = 2 →
  (box_width / 2 + box_height + extra_fold) ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l4039_403905


namespace NUMINAMATH_CALUDE_three_numbers_sum_l4039_403979

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + b + c = 60 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l4039_403979
