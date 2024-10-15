import Mathlib

namespace NUMINAMATH_CALUDE_division_vs_multiplication_error_l1947_194710

theorem division_vs_multiplication_error (x : ℝ) (h : x > 0) :
  ∃ (ε : ℝ), abs (ε - 98) < 1 ∧
  (abs ((8 * x) - (x / 8)) / (8 * x)) * 100 = ε :=
sorry

end NUMINAMATH_CALUDE_division_vs_multiplication_error_l1947_194710


namespace NUMINAMATH_CALUDE_problem_solution_l1947_194700

theorem problem_solution (x y : ℝ) 
  (h1 : x = 153) 
  (h2 : x^3*y - 4*x^2*y + 4*x*y = 350064) : 
  y = 40/3967 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1947_194700


namespace NUMINAMATH_CALUDE_problem_solution_l1947_194732

-- Define the propositions
def p : Prop := ∀ x > 0, 3^x > 1
def q : Prop := ∀ a, a < -2 → (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) ∧
                    ¬(∀ a, (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) → a < -2)

-- Theorem statement
theorem problem_solution :
  (¬p ↔ ∃ x > 0, 3^x ≤ 1) ∧
  ¬p ∧
  q :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1947_194732


namespace NUMINAMATH_CALUDE_min_rotation_regular_pentagon_l1947_194764

/-- The angle of rotation for a regular pentagon to overlap with itself -/
def pentagon_rotation_angle : ℝ := 72

/-- A regular pentagon has 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The minimum angle of rotation for a regular pentagon to overlap with itself is 72 degrees -/
theorem min_rotation_regular_pentagon :
  pentagon_rotation_angle = 360 / pentagon_sides :=
sorry

end NUMINAMATH_CALUDE_min_rotation_regular_pentagon_l1947_194764


namespace NUMINAMATH_CALUDE_log_sum_cubes_l1947_194752

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_sum_cubes (h : lg 2 + lg 5 = 1) :
  (lg 2)^3 + 3*(lg 2)*(lg 5) + (lg 5)^3 = 1 := by sorry

end NUMINAMATH_CALUDE_log_sum_cubes_l1947_194752


namespace NUMINAMATH_CALUDE_G_fraction_difference_l1947_194753

/-- G is defined as the infinite repeating decimal 0.871871871... -/
def G : ℚ := 871 / 999

/-- The difference between the denominator and numerator when G is expressed as a fraction in lowest terms -/
def denominator_numerator_difference : ℕ := 999 - 871

theorem G_fraction_difference : denominator_numerator_difference = 128 := by
  sorry

end NUMINAMATH_CALUDE_G_fraction_difference_l1947_194753


namespace NUMINAMATH_CALUDE_boat_travel_time_l1947_194746

/-- Proves that a boat traveling upstream for 1.5 hours will take 1 hour to travel the same distance downstream, given the boat's speed in still water and the stream's speed. -/
theorem boat_travel_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (upstream_time : ℝ) 
  (h1 : boat_speed = 15) 
  (h2 : stream_speed = 3) 
  (h3 : upstream_time = 1.5) : 
  (boat_speed - stream_speed) * upstream_time / (boat_speed + stream_speed) = 1 := by
  sorry

#check boat_travel_time

end NUMINAMATH_CALUDE_boat_travel_time_l1947_194746


namespace NUMINAMATH_CALUDE_equation_solutions_l1947_194748

def equation (x y : ℝ) : Prop :=
  x^2 + x*y + y^2 + 2*x - 3*y - 3 = 0

def solution_set : Set (ℝ × ℝ) :=
  {(1, 2), (1, 0), (-5, 2), (-5, 6), (-3, 0)}

theorem equation_solutions :
  (∀ (x y : ℝ), (x, y) ∈ solution_set ↔ equation x y) ∧
  equation 1 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1947_194748


namespace NUMINAMATH_CALUDE_square_perimeter_l1947_194776

theorem square_perimeter (total_area overlap_area circle_area : ℝ) : 
  total_area = 2018 →
  overlap_area = 137 →
  circle_area = 1371 →
  ∃ (square_side : ℝ), 
    square_side > 0 ∧ 
    square_side^2 = total_area - (circle_area - overlap_area) ∧
    4 * square_side = 112 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_l1947_194776


namespace NUMINAMATH_CALUDE_root_equation_property_l1947_194782

theorem root_equation_property (α β : ℝ) : 
  (α^2 + α - 1 = 0) → 
  (β^2 + β - 1 = 0) → 
  α^2 + 2*β^2 + β = 4 := by
sorry

end NUMINAMATH_CALUDE_root_equation_property_l1947_194782


namespace NUMINAMATH_CALUDE_six_star_nine_l1947_194744

-- Define the star operation
def star (a b : ℕ) : ℚ :=
  (a * b : ℚ) / (a + b - 3 : ℚ)

-- Theorem statement
theorem six_star_nine :
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ a + b > 3) →
  star 6 9 = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_six_star_nine_l1947_194744


namespace NUMINAMATH_CALUDE_platform_height_l1947_194702

/-- Given two configurations of identical rectangular prisms on a platform,
    prove that the platform height is 37 inches. -/
theorem platform_height (l w : ℝ) : 
  l + 37 - w = 40 → w + 37 - l = 34 → 37 = 37 := by
  sorry

end NUMINAMATH_CALUDE_platform_height_l1947_194702


namespace NUMINAMATH_CALUDE_equal_share_money_l1947_194756

theorem equal_share_money (total_amount : ℚ) (num_people : ℕ) 
  (h1 : total_amount = 3.75)
  (h2 : num_people = 3) : 
  total_amount / num_people = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_money_l1947_194756


namespace NUMINAMATH_CALUDE_min_of_three_exists_l1947_194784

theorem min_of_three_exists : ∃ (f : ℝ → ℝ → ℝ → ℝ), 
  ∀ (a b c : ℝ), f a b c ≤ a ∧ f a b c ≤ b ∧ f a b c ≤ c ∧ 
  (∀ (m : ℝ), m ≤ a ∧ m ≤ b ∧ m ≤ c → f a b c ≥ m) :=
sorry

end NUMINAMATH_CALUDE_min_of_three_exists_l1947_194784


namespace NUMINAMATH_CALUDE_tshirt_purchase_cost_l1947_194717

theorem tshirt_purchase_cost : 
  let num_fandoms : ℕ := 4
  let shirts_per_fandom : ℕ := 5
  let original_price : ℚ := 15
  let initial_discount : ℚ := 0.2
  let additional_discount : ℚ := 0.1
  let seasonal_discount : ℚ := 0.25
  let seasonal_discount_portion : ℚ := 0.5
  let tax_rate : ℚ := 0.1

  let total_shirts := num_fandoms * shirts_per_fandom
  let original_total := total_shirts * original_price
  let after_initial_discount := original_total * (1 - initial_discount)
  let after_additional_discount := after_initial_discount * (1 - additional_discount)
  let seasonal_discount_amount := (original_total * seasonal_discount_portion) * seasonal_discount
  let after_all_discounts := after_additional_discount - seasonal_discount_amount
  let final_cost := after_all_discounts * (1 + tax_rate)

  final_cost = 196.35 := by sorry

end NUMINAMATH_CALUDE_tshirt_purchase_cost_l1947_194717


namespace NUMINAMATH_CALUDE_correct_raisin_distribution_l1947_194787

/-- The number of raisins received by each person -/
structure RaisinDistribution where
  bryce : ℕ
  carter : ℕ
  alice : ℕ

/-- The conditions of the raisin distribution problem -/
def valid_distribution (d : RaisinDistribution) : Prop :=
  d.bryce = d.carter + 10 ∧
  d.carter = d.bryce / 2 ∧
  d.alice = 2 * d.carter

/-- The theorem stating the correct raisin distribution -/
theorem correct_raisin_distribution :
  ∃ (d : RaisinDistribution), valid_distribution d ∧ d.bryce = 20 ∧ d.carter = 10 ∧ d.alice = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_raisin_distribution_l1947_194787


namespace NUMINAMATH_CALUDE_hot_pepper_percentage_is_twenty_percent_l1947_194734

/-- Represents the total number of peppers picked by Joel over 7 days -/
def total_peppers : ℕ := 80

/-- Represents the number of non-hot peppers picked by Joel -/
def non_hot_peppers : ℕ := 64

/-- Calculates the percentage of hot peppers in Joel's garden -/
def hot_pepper_percentage : ℚ :=
  (total_peppers - non_hot_peppers : ℚ) / total_peppers * 100

/-- Proves that the percentage of hot peppers in Joel's garden is 20% -/
theorem hot_pepper_percentage_is_twenty_percent :
  hot_pepper_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_hot_pepper_percentage_is_twenty_percent_l1947_194734


namespace NUMINAMATH_CALUDE_red_peaches_count_l1947_194729

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := sorry

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 11

/-- The difference between green and red peaches -/
def difference : ℕ := 6

/-- Theorem stating that the number of red peaches is 5 -/
theorem red_peaches_count : red_peaches = 5 := by
  sorry

/-- The relationship between green and red peaches -/
axiom green_red_relation : green_peaches = red_peaches + difference


end NUMINAMATH_CALUDE_red_peaches_count_l1947_194729


namespace NUMINAMATH_CALUDE_composite_sum_l1947_194791

theorem composite_sum (x y : ℕ) (h1 : x > 1) (h2 : y > 1) 
  (h3 : ∃ k : ℕ, x^2 + x*y - y = k^2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ x + y + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_sum_l1947_194791


namespace NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l1947_194774

theorem half_plus_five_equals_fifteen (n : ℝ) : (1/2) * n + 5 = 15 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l1947_194774


namespace NUMINAMATH_CALUDE_train_length_l1947_194741

/-- The length of a train given its speed, the speed of a man moving in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) : 
  train_speed = 25 →
  man_speed = 2 →
  crossing_time = 44 →
  (train_speed + man_speed) * crossing_time * (1000 / 3600) = 330 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1947_194741


namespace NUMINAMATH_CALUDE_performance_selection_ways_l1947_194771

/-- The number of students who can sing -/
def num_singers : ℕ := 3

/-- The number of students who can dance -/
def num_dancers : ℕ := 2

/-- The number of students who can both sing and dance -/
def num_both : ℕ := 1

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of students to be selected for singing -/
def singers_to_select : ℕ := 2

/-- The number of students to be selected for dancing -/
def dancers_to_select : ℕ := 1

/-- The number of ways to select the required students for the performance -/
def num_ways : ℕ := Nat.choose (num_singers + num_both) singers_to_select * num_dancers - 1

theorem performance_selection_ways :
  num_ways = Nat.choose (num_singers + num_both) singers_to_select * num_dancers - 1 :=
by sorry

end NUMINAMATH_CALUDE_performance_selection_ways_l1947_194771


namespace NUMINAMATH_CALUDE_expand_and_subtract_l1947_194735

theorem expand_and_subtract (x : ℝ) : (x + 3) * (2 * x - 5) - (2 * x + 1) = 2 * x^2 - x - 16 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_subtract_l1947_194735


namespace NUMINAMATH_CALUDE_fourth_month_sales_l1947_194722

def sales_1 : ℕ := 2500
def sales_2 : ℕ := 6500
def sales_3 : ℕ := 9855
def sales_5 : ℕ := 7000
def sales_6 : ℕ := 11915
def average_sale : ℕ := 7500
def num_months : ℕ := 6

theorem fourth_month_sales (sales_4 : ℕ) : 
  (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale → 
  sales_4 = 14230 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l1947_194722


namespace NUMINAMATH_CALUDE_tax_deduction_percentage_l1947_194760

theorem tax_deduction_percentage (weekly_income : ℝ) (water_bill : ℝ) (tithe_percentage : ℝ) (remaining_amount : ℝ)
  (h1 : weekly_income = 500)
  (h2 : water_bill = 55)
  (h3 : tithe_percentage = 10)
  (h4 : remaining_amount = 345)
  (h5 : remaining_amount = weekly_income - (weekly_income * (tithe_percentage / 100)) - water_bill - (weekly_income * (tax_percentage / 100))) :
  tax_percentage = 10 := by
  sorry


end NUMINAMATH_CALUDE_tax_deduction_percentage_l1947_194760


namespace NUMINAMATH_CALUDE_triangle_side_difference_l1947_194714

/-- Given a triangle ABC with side lengths satisfying specific conditions, prove that b - a = 0 --/
theorem triangle_side_difference (a b : ℤ) : 
  a > 1 → 
  b > 1 → 
  ∃ (AB BC CA : ℝ), 
    AB = b^2 - 1 ∧ 
    BC = a^2 ∧ 
    CA = 2*a ∧ 
    AB + BC > CA ∧ 
    BC + CA > AB ∧ 
    CA + AB > BC → 
    b - a = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l1947_194714


namespace NUMINAMATH_CALUDE_line_and_circle_problem_l1947_194751

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 = 0

-- Define the line m
def line_m (k : ℝ) (x y : ℝ) : Prop := x - k * y + 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define parallel lines
def parallel (k : ℝ) : Prop := ∃ (c : ℝ), ∀ x y, line_l k x y ↔ line_m k (x + c) (y + c * k)

-- Define tangent line to circle
def tangent (k : ℝ) : Prop := ∃! (x y : ℝ), line_l k x y ∧ circle_C x y

theorem line_and_circle_problem :
  (∀ k, parallel k → (k = 1 ∨ k = -1)) ∧
  (∀ k, tangent k → k = 1) :=
sorry

end NUMINAMATH_CALUDE_line_and_circle_problem_l1947_194751


namespace NUMINAMATH_CALUDE_triangle_area_range_l1947_194739

-- Define the triangle ABC
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

-- Define the variable points P and Q
structure VariablePoints :=
  (P : ℝ) -- distance AP
  (Q : ℝ) -- distance AQ

-- Define the perpendiculars x and y
structure Perpendiculars :=
  (x : ℝ)
  (y : ℝ)

-- Define the main theorem
theorem triangle_area_range (ABC : Triangle) (PQ : VariablePoints) (perp : Perpendiculars) :
  ABC.AB = 4 ∧ ABC.BC = 5 ∧ ABC.CA = 3 →
  0 < PQ.P ∧ PQ.P ≤ ABC.AB →
  0 < PQ.Q ∧ PQ.Q ≤ ABC.CA →
  perp.x = PQ.Q / 2 →
  perp.y = PQ.P / 2 →
  PQ.P * PQ.Q = 6 →
  6 ≤ 2 * perp.y + 3 * perp.x ∧ 2 * perp.y + 3 * perp.x ≤ 6.5 :=
by sorry


end NUMINAMATH_CALUDE_triangle_area_range_l1947_194739


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1947_194763

theorem inequality_system_solution :
  ∀ x : ℝ, (x + 2 < 3 * x ∧ (5 - x) / 2 + 1 < 0) ↔ x > 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1947_194763


namespace NUMINAMATH_CALUDE_cubic_factorization_l1947_194767

theorem cubic_factorization (t : ℝ) : t^3 - 125 = (t - 5) * (t^2 + 5*t + 25) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1947_194767


namespace NUMINAMATH_CALUDE_cyclists_max_daily_distance_l1947_194728

theorem cyclists_max_daily_distance (distance_to_boston distance_to_atlanta : ℕ) 
  (h1 : distance_to_boston = 840) 
  (h2 : distance_to_atlanta = 440) : 
  (Nat.gcd distance_to_boston distance_to_atlanta) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_max_daily_distance_l1947_194728


namespace NUMINAMATH_CALUDE_incorrect_expression_l1947_194775

theorem incorrect_expression (a b : ℝ) : 
  2 * ((a^2 + b^2) - a*b) ≠ (a + b)^2 - 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l1947_194775


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1947_194770

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (a^2 - 2*a) + (a - 2)*Complex.I
  (∀ x : ℝ, z = x*Complex.I) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1947_194770


namespace NUMINAMATH_CALUDE_quadratic_single_solution_l1947_194727

theorem quadratic_single_solution (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 - 7 * x + m = 0) ↔ m = 49 / 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_single_solution_l1947_194727


namespace NUMINAMATH_CALUDE_positive_trig_expressions_l1947_194783

theorem positive_trig_expressions :
  (Real.sin (305 * π / 180) * Real.cos (460 * π / 180) > 0) ∧
  (Real.cos (378 * π / 180) * Real.sin (1100 * π / 180) > 0) ∧
  (Real.tan (188 * π / 180) * Real.cos (158 * π / 180) ≤ 0) ∧
  (Real.tan (400 * π / 180) * Real.tan (470 * π / 180) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_positive_trig_expressions_l1947_194783


namespace NUMINAMATH_CALUDE_cubic_function_property_l1947_194750

/-- Given a cubic function f(x) = ax³ + bx + 2 where f(-12) = 3, prove that f(12) = 1 -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x + 2)
  (h2 : f (-12) = 3) : 
  f 12 = 1 := by sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1947_194750


namespace NUMINAMATH_CALUDE_parabola_chord_length_l1947_194708

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop := ∃ t : ℝ, x = 1 + t ∧ y = t

-- Define the intersection points
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y
  on_line : line_through_focus x y

-- Theorem statement
theorem parabola_chord_length 
  (A B : IntersectionPoint) 
  (sum_condition : A.x + B.x = 6) : 
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l1947_194708


namespace NUMINAMATH_CALUDE_xiao_yu_better_l1947_194798

/-- The number of optional questions -/
def total_questions : ℕ := 8

/-- The number of questions randomly selected -/
def selected_questions : ℕ := 4

/-- The probability of Xiao Ming correctly answering a single question -/
def xiao_ming_prob : ℚ := 3/4

/-- The number of questions Xiao Yu can correctly complete -/
def xiao_yu_correct : ℕ := 6

/-- The number of questions Xiao Yu cannot complete -/
def xiao_yu_incorrect : ℕ := 2

/-- The probability of Xiao Ming correctly completing at least 3 questions -/
def xiao_ming_at_least_three : ℚ :=
  Nat.choose selected_questions 3 * xiao_ming_prob^3 * (1 - xiao_ming_prob) +
  Nat.choose selected_questions 4 * xiao_ming_prob^4

/-- The probability of Xiao Yu correctly completing at least 3 questions -/
def xiao_yu_at_least_three : ℚ :=
  (Nat.choose xiao_yu_correct 3 * Nat.choose xiao_yu_incorrect 1 +
   Nat.choose xiao_yu_correct 4 * Nat.choose xiao_yu_incorrect 0) /
  Nat.choose total_questions selected_questions

/-- Theorem stating that Xiao Yu has a higher probability of correctly completing at least 3 questions -/
theorem xiao_yu_better : xiao_yu_at_least_three > xiao_ming_at_least_three := by
  sorry

end NUMINAMATH_CALUDE_xiao_yu_better_l1947_194798


namespace NUMINAMATH_CALUDE_expression_evaluation_l1947_194785

theorem expression_evaluation (x : ℝ) (h : x = 2) : 
  (2*x - 1)^2 + (x + 3)*(x - 3) - 4*(x - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1947_194785


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1947_194704

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 15 = 48 →
  a 3 + 3 * a 8 + a 13 = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1947_194704


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l1947_194706

theorem power_tower_mod_500 : 5^(5^(5^2)) ≡ 25 [ZMOD 500] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l1947_194706


namespace NUMINAMATH_CALUDE_binomial_coefficient_x_squared_l1947_194777

theorem binomial_coefficient_x_squared (x : ℝ) : 
  (Finset.range 11).sum (fun k => Nat.choose 10 k * x^(10 - k) * (1/x)^k) = 
  210 * x^2 + (Finset.range 11).sum (fun k => if k ≠ 4 then Nat.choose 10 k * x^(10 - k) * (1/x)^k else 0) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x_squared_l1947_194777


namespace NUMINAMATH_CALUDE_function_value_at_two_l1947_194724

/-- Given a function f: ℝ → ℝ satisfying f(x) + 2f(1/x) = 3x for all x ∈ ℝ, prove that f(2) = -3/2 -/
theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + 2 * f (1/x) = 3 * x) : f 2 = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l1947_194724


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1947_194745

/-- Given an equilateral triangle where the area is twice the length of one of its sides,
    prove that its perimeter is 8√3 units. -/
theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1947_194745


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1947_194711

/-- A rectangle with area 120 square feet and shorter sides of 8 feet has a perimeter of 46 feet -/
theorem rectangle_perimeter (area : ℝ) (short_side : ℝ) (long_side : ℝ) (perimeter : ℝ) : 
  area = 120 →
  short_side = 8 →
  area = long_side * short_side →
  perimeter = 2 * long_side + 2 * short_side →
  perimeter = 46 := by
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l1947_194711


namespace NUMINAMATH_CALUDE_angle_equivalence_l1947_194769

theorem angle_equivalence :
  ∃ (α : ℝ) (k : ℤ), -27/4 * π = α + 2*k*π ∧ 0 ≤ α ∧ α < 2*π ∧ α = 5*π/4 ∧ k = -8 :=
by sorry

end NUMINAMATH_CALUDE_angle_equivalence_l1947_194769


namespace NUMINAMATH_CALUDE_highDiveVelocity_l1947_194730

/-- The height function for a high-dive swimmer -/
def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

/-- The instantaneous velocity of the high-dive swimmer at t=1s -/
theorem highDiveVelocity : 
  (deriv h) 1 = -3.3 := by sorry

end NUMINAMATH_CALUDE_highDiveVelocity_l1947_194730


namespace NUMINAMATH_CALUDE_angle_range_l1947_194716

theorem angle_range (α : Real) :
  (|Real.sin (4 * Real.pi - α)| = Real.sin (Real.pi + α)) →
  ∃ k : ℤ, 2 * k * Real.pi - Real.pi ≤ α ∧ α ≤ 2 * k * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_angle_range_l1947_194716


namespace NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l1947_194786

noncomputable def f (x : ℝ) : ℝ := 
  (15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5) / (5 * x^5 + 3 * x^3 + 9 * x^2 + 2 * x + 4)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x, x > N → |f x| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l1947_194786


namespace NUMINAMATH_CALUDE_a_closed_form_l1947_194779

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 5 * a (n + 1) - 6 * a n + 4^(n + 1)

theorem a_closed_form (n : ℕ) :
  a n = 2^(n + 1) - 3^(n + 1) + 2 * 4^n :=
by sorry

end NUMINAMATH_CALUDE_a_closed_form_l1947_194779


namespace NUMINAMATH_CALUDE_f_properties_l1947_194794

def f (a x : ℝ) : ℝ := x^2 + |x - a| - 1

theorem f_properties (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 0 ∧
  (∀ x, f a x ≥ -a - 5/4) ∧ (a ≤ -1/2 → ∃ x, f a x = -a - 5/4) ∧
  (∀ x, f a x ≥ a^2 - 1) ∧ (-1/2 < a → a ≤ 1/2 → ∃ x, f a x = a^2 - 1) ∧
  (∀ x, f a x ≥ a - 5/4) ∧ (1/2 < a → ∃ x, f a x = a - 5/4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1947_194794


namespace NUMINAMATH_CALUDE_tv_show_production_cost_l1947_194712

/-- Calculates the total cost of producing all episodes of a TV show with the given conditions -/
theorem tv_show_production_cost :
  let num_seasons : ℕ := 5
  let first_season_cost : ℕ := 100000
  let other_season_cost : ℕ := 2 * first_season_cost
  let first_season_episodes : ℕ := 12
  let other_season_episodes : ℕ := first_season_episodes + (first_season_episodes / 2)
  let last_season_episodes : ℕ := 24
  
  let first_season_total : ℕ := first_season_episodes * first_season_cost
  let other_seasons_episodes : ℕ := other_season_episodes * (num_seasons - 2) + last_season_episodes
  let other_seasons_total : ℕ := other_seasons_episodes * other_season_cost
  
  first_season_total + other_seasons_total = 16800000 :=
by sorry

end NUMINAMATH_CALUDE_tv_show_production_cost_l1947_194712


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1947_194765

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : ∀ m : ℝ, 2 ≤ m ∧ m ≤ 3 → a + b ≤ m^2 - 2*m + 6) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    (∀ m : ℝ, 2 ≤ m ∧ m ≤ 3 → x + y ≤ m^2 - 2*m + 6) ∧
    Real.sqrt (x + 1) + Real.sqrt (y + 1) = 4 ∧
    (∀ c d : ℝ, c > 0 → d > 0 →
      (∀ m : ℝ, 2 ≤ m ∧ m ≤ 3 → c + d ≤ m^2 - 2*m + 6) →
      Real.sqrt (c + 1) + Real.sqrt (d + 1) ≤ 4) := by
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1947_194765


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1947_194778

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) ∧
  C = π/6 ∧
  a = 1 ∧
  b = Real.sqrt 3 →
  B = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1947_194778


namespace NUMINAMATH_CALUDE_even_function_implies_even_g_l1947_194799

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_implies_even_g
  (f g : ℝ → ℝ)
  (h1 : ∀ x, f x - x^2 = g x)
  (h2 : IsEven f) :
  IsEven g := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_even_g_l1947_194799


namespace NUMINAMATH_CALUDE_snacks_expenditure_l1947_194762

theorem snacks_expenditure (total : ℝ) (movies books music ice_cream snacks : ℝ) :
  total = 50 ∧
  movies = (1/4) * total ∧
  books = (1/8) * total ∧
  music = (1/4) * total ∧
  ice_cream = (1/5) * total ∧
  snacks = total - (movies + books + music + ice_cream) →
  snacks = 8.75 := by sorry

end NUMINAMATH_CALUDE_snacks_expenditure_l1947_194762


namespace NUMINAMATH_CALUDE_volume_removed_percentage_l1947_194796

/-- Proves that removing six 4 cm cubes from a 20 cm × 15 cm × 10 cm box removes 12.8% of its volume -/
theorem volume_removed_percentage (box_length box_width box_height cube_side : ℝ) 
  (num_cubes_removed : ℕ) : 
  box_length = 20 → 
  box_width = 15 → 
  box_height = 10 → 
  cube_side = 4 → 
  num_cubes_removed = 6 → 
  (num_cubes_removed * cube_side^3) / (box_length * box_width * box_height) * 100 = 12.8 := by
  sorry

end NUMINAMATH_CALUDE_volume_removed_percentage_l1947_194796


namespace NUMINAMATH_CALUDE_smallest_student_group_l1947_194755

theorem smallest_student_group (n : ℕ) : 
  (n % 6 = 3) ∧ 
  (n % 7 = 4) ∧ 
  (n % 8 = 5) ∧ 
  (n % 9 = 2) ∧ 
  (∀ m : ℕ, m < n → ¬(m % 6 = 3 ∧ m % 7 = 4 ∧ m % 8 = 5 ∧ m % 9 = 2)) → 
  n = 765 := by
sorry

end NUMINAMATH_CALUDE_smallest_student_group_l1947_194755


namespace NUMINAMATH_CALUDE_minimum_force_to_submerge_cube_l1947_194761

-- Define constants
def cube_volume : Real := 10e-6  -- 10 cm³ converted to m³
def cube_density : Real := 400   -- kg/m³
def water_density : Real := 1000 -- kg/m³
def gravity : Real := 10         -- m/s²

-- Define the minimum force function
def minimum_submerge_force (v : Real) (ρ_cube : Real) (ρ_water : Real) (g : Real) : Real :=
  (ρ_water - ρ_cube) * v * g

-- Theorem statement
theorem minimum_force_to_submerge_cube :
  minimum_submerge_force cube_volume cube_density water_density gravity = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_minimum_force_to_submerge_cube_l1947_194761


namespace NUMINAMATH_CALUDE_single_elimination_256_players_l1947_194733

/-- A single-elimination tournament structure -/
structure Tournament :=
  (num_players : ℕ)
  (is_single_elimination : Bool)

/-- The number of games needed to determine a champion in a single-elimination tournament -/
def games_to_champion (t : Tournament) : ℕ :=
  t.num_players - 1

/-- Theorem: In a single-elimination tournament with 256 players, 255 games are needed to determine the champion -/
theorem single_elimination_256_players :
  ∀ t : Tournament, t.num_players = 256 → t.is_single_elimination = true →
  games_to_champion t = 255 :=
by
  sorry

end NUMINAMATH_CALUDE_single_elimination_256_players_l1947_194733


namespace NUMINAMATH_CALUDE_edward_tickets_l1947_194719

theorem edward_tickets (booth_tickets : ℕ) (ride_cost : ℕ) (num_rides : ℕ) : 
  booth_tickets = 23 → ride_cost = 7 → num_rides = 8 →
  ∃ total_tickets : ℕ, total_tickets = booth_tickets + ride_cost * num_rides :=
by
  sorry

end NUMINAMATH_CALUDE_edward_tickets_l1947_194719


namespace NUMINAMATH_CALUDE_complex_product_real_imag_parts_l1947_194747

/-- If z = (2i-1)/i is a complex number with real part a and imaginary part b, then ab = 2 -/
theorem complex_product_real_imag_parts : 
  let z : ℂ := (2 * Complex.I - 1) / Complex.I
  let a : ℝ := z.re
  let b : ℝ := z.im
  a * b = 2 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_parts_l1947_194747


namespace NUMINAMATH_CALUDE_triangle_area_l1947_194766

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is (18 + 8√3) / 25 when a = √3, c = 8/5, and A = π/3 -/
theorem triangle_area (a b c A B C : ℝ) : 
  a = Real.sqrt 3 →
  c = 8 / 5 →
  A = π / 3 →
  (1 / 2) * a * c * Real.sin B = (18 + 8 * Real.sqrt 3) / 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1947_194766


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1947_194754

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (h1 : leg = 15) (h2 : angle = 45) :
  let hypotenuse := leg * Real.sqrt 2
  hypotenuse = 15 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1947_194754


namespace NUMINAMATH_CALUDE_garden_dimensions_l1947_194768

/-- Represents a rectangular garden with given perimeter and length-to-breadth ratio -/
structure RectangularGarden where
  perimeter : ℝ
  length_breadth_ratio : ℝ
  length : ℝ
  breadth : ℝ
  diagonal : ℝ
  perimeter_eq : perimeter = 2 * (length + breadth)
  ratio_eq : length = length_breadth_ratio * breadth

/-- Theorem about the dimensions of a specific rectangular garden -/
theorem garden_dimensions (g : RectangularGarden) 
  (h_perimeter : g.perimeter = 500)
  (h_ratio : g.length_breadth_ratio = 3/2) :
  g.length = 150 ∧ g.diagonal = Real.sqrt 32500 := by
  sorry

#check garden_dimensions

end NUMINAMATH_CALUDE_garden_dimensions_l1947_194768


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1947_194757

theorem fraction_equals_zero (x : ℝ) (h : 5 * x ≠ 0) :
  (x - 6) / (5 * x) = 0 ↔ x = 6 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1947_194757


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l1947_194726

/-- Represents the speed of a person rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

/-- Theorem stating that given the specified upstream and still water speeds, 
    the downstream speed is 53 kmph -/
theorem downstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.upstream = 37) 
  (h2 : s.stillWater = 45) : 
  downstreamSpeed s = 53 := by
  sorry

#eval downstreamSpeed { upstream := 37, stillWater := 45 }

end NUMINAMATH_CALUDE_downstream_speed_calculation_l1947_194726


namespace NUMINAMATH_CALUDE_min_value_expression_l1947_194737

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y < 27) :
  (Real.sqrt x + Real.sqrt y) / Real.sqrt (x * y) + 1 / Real.sqrt (27 - x - y) ≥ 1 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b < 27 ∧
    (Real.sqrt a + Real.sqrt b) / Real.sqrt (a * b) + 1 / Real.sqrt (27 - a - b) = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1947_194737


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1947_194772

theorem smallest_n_congruence (n : ℕ) : n = 3 ↔ (
  n > 0 ∧
  17 * n ≡ 136 [ZMOD 5] ∧
  ∀ m : ℕ, m > 0 → m < n → ¬(17 * m ≡ 136 [ZMOD 5])
) := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1947_194772


namespace NUMINAMATH_CALUDE_find_number_l1947_194723

theorem find_number : ∃ N : ℚ, (5/6 * N) - (5/16 * N) = 250 ∧ N = 480 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1947_194723


namespace NUMINAMATH_CALUDE_hypotenuse_length_l1947_194740

/-- A right triangle with specific properties -/
structure RightTriangle where
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The side opposite to the 30° angle -/
  opposite_30 : ℝ
  /-- Condition: The triangle has a right angle -/
  right_angle : True
  /-- Condition: One angle is 30° -/
  angle_30 : True
  /-- Condition: The perimeter is 120 units -/
  perimeter_120 : perimeter = 120
  /-- Condition: The side opposite to 30° is half the hypotenuse -/
  opposite_half_hypotenuse : opposite_30 = hypotenuse / 2

/-- Theorem: The hypotenuse of the specified right triangle is 40(3 - √3) -/
theorem hypotenuse_length (t : RightTriangle) : t.hypotenuse = 40 * (3 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l1947_194740


namespace NUMINAMATH_CALUDE_product_of_numbers_l1947_194705

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 58) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1947_194705


namespace NUMINAMATH_CALUDE_element_n3_l1947_194709

/-- Represents a right triangular number array where each column forms an arithmetic sequence
    and each row (starting from the third row) forms a geometric sequence with a constant common ratio. -/
structure TriangularArray where
  -- a[i][j] represents the element in the i-th row and j-th column
  a : Nat → Nat → Rat
  -- Each column forms an arithmetic sequence
  column_arithmetic : ∀ i j k, i ≥ j → k ≥ j → a (i+1) j - a i j = a (k+1) j - a k j
  -- Each row forms a geometric sequence (starting from the third row)
  row_geometric : ∀ i j, i ≥ 3 → j < i → a i (j+1) / a i j = a i (j+2) / a i (j+1)

/-- The element a_{n3} in the n-th row and 3rd column is equal to n/16 -/
theorem element_n3 (arr : TriangularArray) (n : Nat) :
  arr.a n 3 = n / 16 := by
  sorry

end NUMINAMATH_CALUDE_element_n3_l1947_194709


namespace NUMINAMATH_CALUDE_min_value_of_f_range_of_m_l1947_194789

noncomputable section

open Real MeasureTheory

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := log x + a / x - 1
def g (x : ℝ) : ℝ := x + 1 / x

-- Part 1
theorem min_value_of_f :
  (∀ x > 0, f 2 x ≥ log 2) ∧ (∃ x > 0, f 2 x = log 2) := by sorry

-- Part 2
theorem range_of_m :
  let f' := f (-1)
  {m : ℝ | ∃ x ∈ Set.Icc 1 (Real.exp 1), g x < m * (f' x + 1)} =
  Set.Ioi ((Real.exp 2 + 1) / (Real.exp 1 - 1)) ∪ Set.Iio (-2) := by sorry

end

end NUMINAMATH_CALUDE_min_value_of_f_range_of_m_l1947_194789


namespace NUMINAMATH_CALUDE_orangeade_price_day1_l1947_194736

/-- Represents the price of orangeade per glass on a given day -/
structure OrangeadePrice where
  price : ℚ
  day : ℕ

/-- Represents the amount of orangeade made on a given day -/
structure OrangeadeAmount where
  amount : ℚ
  day : ℕ

/-- Represents the revenue from selling orangeade on a given day -/
def revenue (price : OrangeadePrice) (amount : OrangeadeAmount) : ℚ :=
  price.price * amount.amount

theorem orangeade_price_day1 (juice : ℚ) 
  (amount_day1 : OrangeadeAmount) 
  (amount_day2 : OrangeadeAmount)
  (price_day1 : OrangeadePrice)
  (price_day2 : OrangeadePrice) :
  amount_day1.amount = 2 * juice →
  amount_day2.amount = 3 * juice →
  amount_day1.day = 1 →
  amount_day2.day = 2 →
  price_day1.day = 1 →
  price_day2.day = 2 →
  price_day2.price = 2/5 →
  revenue price_day1 amount_day1 = revenue price_day2 amount_day2 →
  price_day1.price = 3/5 := by
  sorry

#eval (3 : ℚ) / 5  -- Should output 0.6

end NUMINAMATH_CALUDE_orangeade_price_day1_l1947_194736


namespace NUMINAMATH_CALUDE_min_disks_required_l1947_194792

def disk_capacity : ℝ := 1.44

def file_count : ℕ := 40

def file_sizes : List ℝ := [0.95, 0.95, 0.95, 0.95, 0.95] ++ 
                           List.replicate 15 0.65 ++ 
                           List.replicate 20 0.45

def total_file_size : ℝ := file_sizes.sum

theorem min_disks_required : 
  ∀ (arrangement : List (List ℝ)),
    (arrangement.length < 17 → 
     ∃ (disk : List ℝ), disk ∈ arrangement ∧ disk.sum > disk_capacity) ∧
    (∃ (valid_arrangement : List (List ℝ)), 
      valid_arrangement.length = 17 ∧
      valid_arrangement.join.sum = total_file_size ∧
      ∀ (disk : List ℝ), disk ∈ valid_arrangement → disk.sum ≤ disk_capacity) :=
by sorry

end NUMINAMATH_CALUDE_min_disks_required_l1947_194792


namespace NUMINAMATH_CALUDE_value_range_of_sum_product_l1947_194793

theorem value_range_of_sum_product (x : ℝ) : 
  ∃ (a b c : ℝ), a + b + c = 1 ∧ a^2 * b + b^2 * c + c^2 * a = x :=
sorry

end NUMINAMATH_CALUDE_value_range_of_sum_product_l1947_194793


namespace NUMINAMATH_CALUDE_intersection_complement_problem_l1947_194720

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

theorem intersection_complement_problem :
  N ∩ (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_problem_l1947_194720


namespace NUMINAMATH_CALUDE_match_duration_l1947_194790

theorem match_duration (goals_per_interval : ℝ) (interval_duration : ℝ) (total_goals : ℝ) :
  goals_per_interval = 2 →
  interval_duration = 15 →
  total_goals = 16 →
  (total_goals / goals_per_interval) * interval_duration = 120 :=
by
  sorry

#check match_duration

end NUMINAMATH_CALUDE_match_duration_l1947_194790


namespace NUMINAMATH_CALUDE_exists_bound_factorial_digit_sum_l1947_194731

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number b such that for all natural numbers n > b,
    the sum of the digits of n! is greater than or equal to 10^100 -/
theorem exists_bound_factorial_digit_sum :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (n.factorial) ≥ 10^100 := by sorry

end NUMINAMATH_CALUDE_exists_bound_factorial_digit_sum_l1947_194731


namespace NUMINAMATH_CALUDE_max_interval_length_l1947_194701

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem max_interval_length 
  (a b : ℝ) 
  (h1 : a ≤ b) 
  (h2 : ∀ x ∈ Set.Icc a b, -3 ≤ f x ∧ f x ≤ 1) 
  (h3 : ∃ x ∈ Set.Icc a b, f x = -3) 
  (h4 : ∃ x ∈ Set.Icc a b, f x = 1) :
  b - a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_interval_length_l1947_194701


namespace NUMINAMATH_CALUDE_increments_of_z_l1947_194738

noncomputable def z (x y : ℝ) : ℝ := x^2 * y

theorem increments_of_z (x y Δx Δy : ℝ) 
  (hx : x = 1) (hy : y = 2) (hΔx : Δx = 0.1) (hΔy : Δy = -0.2) :
  let Δx_z := z (x + Δx) y - z x y
  let Δy_z := z x (y + Δy) - z x y
  let Δz := z (x + Δx) (y + Δy) - z x y
  (Δx_z = 0.42) ∧ (Δy_z = -0.2) ∧ (Δz = 0.178) := by
  sorry

end NUMINAMATH_CALUDE_increments_of_z_l1947_194738


namespace NUMINAMATH_CALUDE_xyz_value_l1947_194725

theorem xyz_value (x y z : ℝ) 
  (h1 : 2 * x + 3 * y + z = 13) 
  (h2 : 4 * x^2 + 9 * y^2 + z^2 - 2 * x + 15 * y + 3 * z = 82) : 
  x * y * z = 12 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1947_194725


namespace NUMINAMATH_CALUDE_negation_equivalence_l1947_194780

/-- Represents the property of being an honor student -/
def is_honor_student (x : Type) : Prop := sorry

/-- Represents the property of receiving a scholarship -/
def receives_scholarship (x : Type) : Prop := sorry

/-- The original statement: All honor students receive scholarships -/
def all_honor_students_receive_scholarships : Prop :=
  ∀ x, is_honor_student x → receives_scholarship x

/-- The negation of the original statement -/
def negation_of_statement : Prop :=
  ¬(∀ x, is_honor_student x → receives_scholarship x)

/-- The proposed equivalent negation: Some honor students do not receive scholarships -/
def some_honor_students_dont_receive_scholarships : Prop :=
  ∃ x, is_honor_student x ∧ ¬receives_scholarship x

/-- Theorem stating that the negation of the original statement is equivalent to 
    "Some honor students do not receive scholarships" -/
theorem negation_equivalence :
  negation_of_statement ↔ some_honor_students_dont_receive_scholarships := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1947_194780


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_eq_choose_l1947_194773

/-- The number of interior intersection points of diagonals in a regular decagon -/
def decagon_diagonal_intersections : ℕ :=
  Nat.choose 10 4

/-- Theorem: The number of interior intersection points of diagonals in a regular decagon
    is equal to the number of ways to choose 4 vertices out of 10 -/
theorem decagon_diagonal_intersections_eq_choose :
  decagon_diagonal_intersections = Nat.choose 10 4 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_eq_choose_l1947_194773


namespace NUMINAMATH_CALUDE_set_operations_l1947_194788

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x - 4 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Define the universe set U
def U : Set ℝ := Set.univ

-- Theorem statements
theorem set_operations :
  (A ∩ B = {x | 0 < x ∧ x < 2}) ∧
  (Set.compl A = {x | x ≥ 2}) ∧
  (Set.compl A ∩ B = {x | 2 ≤ x ∧ x < 5}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1947_194788


namespace NUMINAMATH_CALUDE_M_factor_count_l1947_194718

def M : ℕ := 2^6 * 3^5 * 5^3 * 7^4 * 11^1

def count_factors (n : ℕ) : ℕ := sorry

theorem M_factor_count : count_factors M = 1680 := by sorry

end NUMINAMATH_CALUDE_M_factor_count_l1947_194718


namespace NUMINAMATH_CALUDE_solve_equation_l1947_194707

theorem solve_equation (x : ℝ) : 11 + Real.sqrt (-4 + 6 * x / 3) = 13 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1947_194707


namespace NUMINAMATH_CALUDE_emma_sister_age_relationship_l1947_194797

/-- Emma's current age -/
def emma_age : ℕ := 7

/-- Age difference between Emma and her sister -/
def age_difference : ℕ := 9

/-- Emma's age when her sister is 56 -/
def emma_future_age : ℕ := 47

/-- Emma's sister's age when Emma is 47 -/
def sister_future_age : ℕ := 56

/-- Theorem stating the relationship between Emma's age and her sister's age -/
theorem emma_sister_age_relationship (x : ℕ) :
  x ≥ age_difference →
  emma_future_age = sister_future_age - age_difference →
  x - age_difference = emma_age + (x - sister_future_age) :=
by
  sorry

end NUMINAMATH_CALUDE_emma_sister_age_relationship_l1947_194797


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1947_194743

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 207)
  (h2 : profit_percentage = 0.15) : 
  ∃ (cost_price : ℝ), cost_price = 180 ∧ selling_price = cost_price * (1 + profit_percentage) :=
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1947_194743


namespace NUMINAMATH_CALUDE_cannot_form_triangle_l1947_194759

/-- Triangle Inequality Theorem: A triangle can be formed if the sum of the lengths 
    of any two sides is greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Proof that the set of line segments (5, 8, 2) cannot form a triangle -/
theorem cannot_form_triangle : ¬ triangle_inequality 5 8 2 := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_triangle_l1947_194759


namespace NUMINAMATH_CALUDE_age_sum_proof_l1947_194715

theorem age_sum_proof (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_proof_l1947_194715


namespace NUMINAMATH_CALUDE_average_age_combined_l1947_194703

theorem average_age_combined (n_students : Nat) (n_parents : Nat)
  (avg_age_students : ℚ) (avg_age_parents : ℚ)
  (h1 : n_students = 50)
  (h2 : n_parents = 75)
  (h3 : avg_age_students = 10)
  (h4 : avg_age_parents = 40) :
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents : ℚ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l1947_194703


namespace NUMINAMATH_CALUDE_travel_distance_ratio_l1947_194781

/-- Proves that if a person travels 400 km every odd month and x times 400 km every even month,
    and the total distance traveled in 24 months is 14400 km, then x = 2. -/
theorem travel_distance_ratio (x : ℝ) : 
  (12 * 400 + 12 * (400 * x) = 14400) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_travel_distance_ratio_l1947_194781


namespace NUMINAMATH_CALUDE_carnations_ordered_l1947_194758

/-- Proves that given the specified conditions, the number of carnations ordered is 375 -/
theorem carnations_ordered (tulips : ℕ) (roses : ℕ) (price_per_flower : ℕ) (total_expenses : ℕ) : 
  tulips = 250 → roses = 320 → price_per_flower = 2 → total_expenses = 1890 →
  ∃ carnations : ℕ, carnations = 375 ∧ 
    price_per_flower * (tulips + roses + carnations) = total_expenses := by
  sorry

#check carnations_ordered

end NUMINAMATH_CALUDE_carnations_ordered_l1947_194758


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1947_194742

theorem inequality_equivalence (x : ℝ) : (x - 1) / 2 ≤ -1 ↔ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1947_194742


namespace NUMINAMATH_CALUDE_large_pizza_slices_l1947_194713

def large_pizza_cost : ℝ := 10
def first_topping_cost : ℝ := 2
def next_two_toppings_cost : ℝ := 1
def remaining_toppings_cost : ℝ := 0.5
def num_toppings : ℕ := 7
def cost_per_slice : ℝ := 2

def total_pizza_cost : ℝ :=
  large_pizza_cost + first_topping_cost + 2 * next_two_toppings_cost + 
  (num_toppings - 3 : ℝ) * remaining_toppings_cost

theorem large_pizza_slices :
  (total_pizza_cost / cost_per_slice : ℝ) = 8 :=
sorry

end NUMINAMATH_CALUDE_large_pizza_slices_l1947_194713


namespace NUMINAMATH_CALUDE_ellipse_equation_from_line_l1947_194749

/-- The standard equation of an ellipse -/
structure EllipseEquation where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- A line passing through a focus and vertex of an ellipse -/
structure EllipseLine where
  slope : ℝ
  intercept : ℝ

/-- The theorem statement -/
theorem ellipse_equation_from_line (l : EllipseLine) 
  (h1 : l.slope = 1/2 ∧ l.intercept = -1) 
  (h2 : ∃ (f v : ℝ × ℝ), l.slope * f.1 + l.intercept = f.2 ∧ 
                          l.slope * v.1 + l.intercept = v.2) 
  (h3 : l.slope * 0 + l.intercept = 1) :
  ∃ (e1 e2 : EllipseEquation), 
    (e1.a^2 = 5 ∧ e1.b^2 = 1) ∨ 
    (e2.a^2 = 5 ∧ e2.b^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_line_l1947_194749


namespace NUMINAMATH_CALUDE_field_area_change_l1947_194795

theorem field_area_change (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let new_length := 1.35 * a
  let new_width := 0.86 * b
  let initial_area := a * b
  let new_area := new_length * new_width
  (new_area - initial_area) / initial_area = 0.161 := by
sorry

end NUMINAMATH_CALUDE_field_area_change_l1947_194795


namespace NUMINAMATH_CALUDE_expression_simplification_l1947_194721

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 2) :
  (a / (a^2 - 4*a + 4) + (a + 2) / (2*a - a^2)) / (2 / (a^2 - 2*a)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1947_194721
