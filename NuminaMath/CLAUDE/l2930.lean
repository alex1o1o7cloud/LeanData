import Mathlib

namespace initial_list_size_l2930_293017

theorem initial_list_size (l : List Int) (m : ℚ) : 
  (((l.sum + 20) / (l.length + 1) : ℚ) = m + 3) →
  (((l.sum + 25) / (l.length + 2) : ℚ) = m + 1) →
  l.length = 3 := by
sorry

end initial_list_size_l2930_293017


namespace factorial_200_less_than_100_pow_200_l2930_293050

-- Define factorial
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Theorem statement
theorem factorial_200_less_than_100_pow_200 :
  factorial 200 < 100^200 := by
  sorry

end factorial_200_less_than_100_pow_200_l2930_293050


namespace m_range_l2930_293030

/-- A function f parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 3 * x - m - 2

/-- The property that f has exactly one root in (0, 1) -/
def has_one_root_in_unit_interval (m : ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 1 ∧ f m x = 0

/-- The main theorem stating the range of m -/
theorem m_range :
  ∀ m : ℝ, has_one_root_in_unit_interval m ↔ m > -2 :=
sorry

end m_range_l2930_293030


namespace unique_solution_system_l2930_293010

theorem unique_solution_system (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃! (x y z : ℝ), x + a * y + a^2 * z = 0 ∧
                   x + b * y + b^2 * z = 0 ∧
                   x + c * y + c^2 * z = 0 :=
by
  sorry

end unique_solution_system_l2930_293010


namespace quadratic_roots_sum_squares_minimum_l2930_293070

theorem quadratic_roots_sum_squares_minimum (m : ℝ) :
  let a : ℝ := 6
  let b : ℝ := -8
  let c : ℝ := m
  let discriminant := b^2 - 4*a*c
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  let sum_of_squares := sum_of_roots^2 - 2*product_of_roots
  discriminant > 0 →
  (∀ m' : ℝ, discriminant > 0 → sum_of_squares ≤ ((-b/a)^2 - 2*(m'/a))) →
  m = 8/3 ∧ sum_of_squares = 8/9 :=
by sorry

end quadratic_roots_sum_squares_minimum_l2930_293070


namespace henry_bicycle_improvement_l2930_293056

/-- Henry's bicycle ride improvement --/
theorem henry_bicycle_improvement (initial_laps initial_time current_laps current_time : ℚ) 
  (h1 : initial_laps = 15)
  (h2 : initial_time = 45)
  (h3 : current_laps = 18)
  (h4 : current_time = 42) :
  (initial_time / initial_laps) - (current_time / current_laps) = 2/3 := by
  sorry

#eval (45 : ℚ) / 15 - (42 : ℚ) / 18

end henry_bicycle_improvement_l2930_293056


namespace sum_of_x_and_y_x_is_smallest_y_is_smallest_x_makes_square_y_makes_cube_l2930_293012

/-- The smallest positive integer x for which 420x is a square -/
def x : ℕ := 735

/-- The smallest positive integer y for which 420y is a cube -/
def y : ℕ := 22050

theorem sum_of_x_and_y : x + y = 22785 := by sorry

theorem x_is_smallest :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, 420 * n = m ^ 2) → n ≥ x := by sorry

theorem y_is_smallest :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, 420 * n = m ^ 3) → n ≥ y := by sorry

theorem x_makes_square : ∃ m : ℕ, 420 * x = m ^ 2 := by sorry

theorem y_makes_cube : ∃ m : ℕ, 420 * y = m ^ 3 := by sorry

end sum_of_x_and_y_x_is_smallest_y_is_smallest_x_makes_square_y_makes_cube_l2930_293012


namespace polynomial_equality_sum_l2930_293013

theorem polynomial_equality_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end polynomial_equality_sum_l2930_293013


namespace well_digging_cost_l2930_293093

/-- The cost of digging a cylindrical well -/
theorem well_digging_cost (depth : ℝ) (diameter : ℝ) (cost_per_cubic_meter : ℝ) :
  depth = 14 ∧ diameter = 3 ∧ cost_per_cubic_meter = 18 →
  ∃ (total_cost : ℝ), (abs (total_cost - 1782) < 1) ∧
  total_cost = (Real.pi * (diameter / 2)^2 * depth) * cost_per_cubic_meter :=
by sorry

end well_digging_cost_l2930_293093


namespace negative_rational_identification_l2930_293072

theorem negative_rational_identification :
  let a := -(-2010)
  let b := -|-2010|
  let c := (-2011)^2010
  let d := -2010 / -2011
  (¬ (a < 0 ∧ ∃ (p q : ℤ), a = p / q ∧ q ≠ 0)) ∧
  (b < 0 ∧ ∃ (p q : ℤ), b = p / q ∧ q ≠ 0) ∧
  (¬ (c < 0 ∧ ∃ (p q : ℤ), c = p / q ∧ q ≠ 0)) ∧
  (¬ (d < 0 ∧ ∃ (p q : ℤ), d = p / q ∧ q ≠ 0)) :=
by sorry


end negative_rational_identification_l2930_293072


namespace tensor_identity_implies_unit_vector_l2930_293038

def Vector2D := ℝ × ℝ

def tensor_product (m n : Vector2D) : Vector2D :=
  let (a, b) := m
  let (c, d) := n
  (a * c + b * d, a * d + b * c)

theorem tensor_identity_implies_unit_vector (p : Vector2D) :
  (∀ m : Vector2D, tensor_product m p = m) → p = (1, 0) := by
  sorry

end tensor_identity_implies_unit_vector_l2930_293038


namespace three_digit_divisible_by_21_ending_in_3_l2930_293062

theorem three_digit_divisible_by_21_ending_in_3 :
  ∃! (s : Finset Nat), 
    s.card = 3 ∧
    (∀ n ∈ s, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ n % 21 = 0) ∧
    (∀ n, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ n % 21 = 0 → n ∈ s) :=
by sorry

end three_digit_divisible_by_21_ending_in_3_l2930_293062


namespace perimeter_area_ratio_not_always_equal_l2930_293084

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  leg : ℝ
  perimeter : ℝ
  area : ℝ

/-- The theorem states that the ratio of perimeters is not always equal to the ratio of areas for two different isosceles triangles -/
theorem perimeter_area_ratio_not_always_equal
  (triangle1 triangle2 : IsoscelesTriangle)
  (h_base_neq : triangle1.base ≠ triangle2.base)
  (h_leg_neq : triangle1.leg ≠ triangle2.leg) :
  ¬ ∀ (triangle1 triangle2 : IsoscelesTriangle),
    triangle1.perimeter / triangle2.perimeter = triangle1.area / triangle2.area :=
by sorry

end perimeter_area_ratio_not_always_equal_l2930_293084


namespace smallest_number_in_set_l2930_293027

theorem smallest_number_in_set (a b c d : ℕ+) : 
  (a + b + c + d : ℝ) / 4 = 30 →
  b = 28 →
  b < c →
  c < d →
  d = b + 7 →
  a < b →
  a = 27 := by
sorry

end smallest_number_in_set_l2930_293027


namespace weight_sum_proof_l2930_293063

/-- Given the weights of four people and their pairwise sums, 
    prove that the sum of the weights of the first and last person is 295 pounds. -/
theorem weight_sum_proof (a b c d : ℝ) 
  (h1 : a + b = 270)
  (h2 : b + c = 255)
  (h3 : c + d = 280)
  (h4 : a + b + c + d = 480) :
  a + d = 295 := by
  sorry

end weight_sum_proof_l2930_293063


namespace greatest_common_divisor_420_90_under_60_l2930_293035

theorem greatest_common_divisor_420_90_under_60 : 
  ∃ (n : ℕ), n ∣ 420 ∧ n ∣ 90 ∧ n < 60 ∧ 
  ∀ (m : ℕ), m ∣ 420 ∧ m ∣ 90 ∧ m < 60 → m ≤ n :=
by
  -- The proof would go here
  sorry

end greatest_common_divisor_420_90_under_60_l2930_293035


namespace chess_tournament_score_difference_l2930_293067

-- Define the number of players
def num_players : ℕ := 12

-- Define the scoring system
def win_points : ℚ := 1
def draw_points : ℚ := 1/2
def loss_points : ℚ := 0

-- Define the total number of games
def total_games : ℕ := num_players * (num_players - 1) / 2

-- Define Vasya's score (minimum possible given the conditions)
def vasya_score : ℚ := loss_points + (num_players - 2) * draw_points

-- Define the minimum score for other players to be higher than Vasya
def min_other_score : ℚ := vasya_score + 1/2

-- Define Petya's score (maximum possible)
def petya_score : ℚ := (num_players - 1) * win_points

-- Theorem statement
theorem chess_tournament_score_difference :
  petya_score - vasya_score = 1 := by sorry

end chess_tournament_score_difference_l2930_293067


namespace hyperbola_eccentricity_l2930_293089

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let parabola (x y : ℝ) := y^2 = 20*x
  let hyperbola (x y : ℝ) := x^2/a^2 - y^2/b^2 = 1
  let focus_parabola : ℝ × ℝ := (5, 0)
  let asymptote (x y : ℝ) := b*x + a*y = 0
  let distance_focus_asymptote := 4
  let eccentricity := (Real.sqrt (a^2 + b^2)) / a
  (∀ x y, parabola x y → hyperbola x y) →
  (distance_focus_asymptote = 4) →
  eccentricity = 5/3 :=
by sorry

end hyperbola_eccentricity_l2930_293089


namespace prob_same_color_diff_foot_value_l2930_293088

def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 3

def total_shoes : ℕ := total_pairs * 2

def prob_same_color_diff_foot : ℚ :=
  (black_pairs * 2 * black_pairs) / (total_shoes * (total_shoes - 1)) +
  (brown_pairs * 2 * brown_pairs) / (total_shoes * (total_shoes - 1)) +
  (gray_pairs * 2 * gray_pairs) / (total_shoes * (total_shoes - 1))

theorem prob_same_color_diff_foot_value :
  prob_same_color_diff_foot = 89 / 435 := by
  sorry

end prob_same_color_diff_foot_value_l2930_293088


namespace shellys_total_money_l2930_293071

/-- Calculates the total amount of money Shelly has given her bill and coin counts. -/
def shellys_money (ten_dollar_bills : ℕ) : ℕ :=
  let five_dollar_bills := ten_dollar_bills - 12
  let twenty_dollar_bills := ten_dollar_bills / 2
  let one_dollar_coins := five_dollar_bills * 2
  10 * ten_dollar_bills + 5 * five_dollar_bills + 20 * twenty_dollar_bills + one_dollar_coins

/-- Proves that Shelly has $726 given the conditions in the problem. -/
theorem shellys_total_money : shellys_money 30 = 726 := by
  sorry

end shellys_total_money_l2930_293071


namespace cone_volume_l2930_293058

/-- Given a cone with base circumference 2π and lateral area 2π, its volume is (√3 * π) / 3 -/
theorem cone_volume (r h l : ℝ) (h1 : 2 * π = 2 * π * r) (h2 : 2 * π = π * r * l) :
  (1 / 3) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end cone_volume_l2930_293058


namespace right_triangle_area_l2930_293025

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 15) (h_side : a = 12) : (1/2) * a * b = 54 :=
by
  sorry

end right_triangle_area_l2930_293025


namespace square_circle_area_ratio_l2930_293099

theorem square_circle_area_ratio (r : ℝ) (h : r > 0) :
  let s := 2 * r
  (s^2) / (π * r^2) = 4 / π := by
sorry

end square_circle_area_ratio_l2930_293099


namespace mary_book_count_l2930_293026

/-- Represents the number of books Mary has at different stages --/
structure BookCount where
  initial : Nat
  afterReturningUnhelpful : Nat
  afterFirstCheckout : Nat
  beforeSecondCheckout : Nat
  final : Nat

/-- Represents the number of books Mary checks out or returns --/
structure BookTransactions where
  firstReturn : Nat
  firstCheckout : Nat
  secondReturn : Nat
  secondCheckout : Nat

theorem mary_book_count (b : BookCount) (t : BookTransactions) :
  b.initial = 5 →
  t.firstReturn = 3 →
  b.afterReturningUnhelpful = b.initial - t.firstReturn →
  b.afterFirstCheckout = b.afterReturningUnhelpful + t.firstCheckout →
  b.beforeSecondCheckout = b.afterFirstCheckout - t.secondReturn →
  t.secondReturn = 2 →
  t.secondCheckout = 7 →
  b.final = b.beforeSecondCheckout + t.secondCheckout →
  b.final = 12 →
  t.firstCheckout = 5 := by
sorry

end mary_book_count_l2930_293026


namespace seokgi_paper_usage_l2930_293051

theorem seokgi_paper_usage (total : ℕ) (used : ℕ) (remaining : ℕ) : 
  total = 82 ∧ 
  remaining = total - used ∧ 
  remaining = used - 6 → 
  used = 44 := by sorry

end seokgi_paper_usage_l2930_293051


namespace num_a_animals_l2930_293061

def total_animals : ℕ := 17
def num_b_animals : ℕ := 8

theorem num_a_animals : total_animals - num_b_animals = 9 := by
  sorry

end num_a_animals_l2930_293061


namespace intersection_M_N_l2930_293008

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_M_N_l2930_293008


namespace parabola_shift_left_two_l2930_293045

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x + h) }

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola :=
  { f := fun x => x^2 }

theorem parabola_shift_left_two :
  (shift_parabola standard_parabola 2).f = fun x => (x + 2)^2 := by
  sorry

end parabola_shift_left_two_l2930_293045


namespace least_multiple_and_digit_sum_l2930_293019

def least_multiple_of_17_gt_500 : ℕ := 510

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem least_multiple_and_digit_sum :
  (least_multiple_of_17_gt_500 % 17 = 0) ∧
  (least_multiple_of_17_gt_500 > 500) ∧
  (∀ m : ℕ, m % 17 = 0 ∧ m > 500 → m ≥ least_multiple_of_17_gt_500) ∧
  (sum_of_digits least_multiple_of_17_gt_500 = 6) :=
by sorry

end least_multiple_and_digit_sum_l2930_293019


namespace solution_set_proof_l2930_293036

theorem solution_set_proof (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h0 : f 0 = 2) (h1 : ∀ x : ℝ, f x + (deriv f) x > 1) :
  {x : ℝ | Real.exp x * f x > Real.exp x + 1} = {x : ℝ | x > 0} := by
  sorry

end solution_set_proof_l2930_293036


namespace hamburgers_left_over_l2930_293095

theorem hamburgers_left_over (hamburgers_made : ℕ) (hamburgers_served : ℕ) : 
  hamburgers_made = 15 → hamburgers_served = 8 → hamburgers_made - hamburgers_served = 7 := by
  sorry

#check hamburgers_left_over

end hamburgers_left_over_l2930_293095


namespace equation_solution_l2930_293040

theorem equation_solution (x : ℝ) : (6 : ℝ) / (x + 1) = (3 : ℝ) / 2 → x = 3 := by
  sorry

end equation_solution_l2930_293040


namespace f_neg_two_equals_nineteen_l2930_293082

/-- Given a function f(x) = 2x^2 - 4x + 3, prove that f(-2) = 19 -/
theorem f_neg_two_equals_nineteen : 
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4 * x + 3
  f (-2) = 19 := by
  sorry

end f_neg_two_equals_nineteen_l2930_293082


namespace mildred_weight_l2930_293044

/-- Given that Carol weighs 9 pounds and Mildred is 50 pounds heavier than Carol,
    prove that Mildred weighs 59 pounds. -/
theorem mildred_weight (carol_weight : ℕ) (weight_difference : ℕ) :
  carol_weight = 9 →
  weight_difference = 50 →
  carol_weight + weight_difference = 59 :=
by sorry

end mildred_weight_l2930_293044


namespace distinct_integer_roots_l2930_293077

theorem distinct_integer_roots (a : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + 2*a*x = 8*a ∧ y^2 + 2*a*y = 8*a) ↔ 
  a ∈ ({4.5, 1, -12.5, -9} : Set ℝ) :=
by sorry

end distinct_integer_roots_l2930_293077


namespace instantaneous_velocity_at_2_l2930_293086

-- Define the displacement function
def s (t : ℝ) : ℝ := 100 * t - 5 * t^2

-- Define the instantaneous velocity function
def v (t : ℝ) : ℝ := 100 - 10 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 :
  ∀ t : ℝ, 0 < t → t < 20 → v 2 = 80 := by
  sorry

end instantaneous_velocity_at_2_l2930_293086


namespace distribute_five_balls_four_boxes_l2930_293039

/-- Represents the number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 4 distinguishable boxes is 68 -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 68 := by sorry

end distribute_five_balls_four_boxes_l2930_293039


namespace min_value_quadratic_l2930_293034

theorem min_value_quadratic (x : ℝ) :
  let z := 4 * x^2 + 8 * x + 16
  ∀ y : ℝ, z ≤ y → 12 ≤ y :=
by sorry

end min_value_quadratic_l2930_293034


namespace set_equality_from_union_intersection_equality_l2930_293003

theorem set_equality_from_union_intersection_equality {α : Type*} (A B : Set α) :
  A ∪ B = A ∩ B → A = B := by sorry

end set_equality_from_union_intersection_equality_l2930_293003


namespace mans_upstream_rate_l2930_293054

/-- Given a man's rowing rates and current speed, calculate his upstream rate -/
theorem mans_upstream_rate
  (downstream_rate : ℝ)
  (still_water_rate : ℝ)
  (current_rate : ℝ)
  (h1 : downstream_rate = 24)
  (h2 : still_water_rate = 15.5)
  (h3 : current_rate = 8.5) :
  still_water_rate - current_rate = 7 := by
  sorry

end mans_upstream_rate_l2930_293054


namespace inequality_range_l2930_293090

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 2 < 0) ↔ -8 < m ∧ m ≤ 0 := by sorry

end inequality_range_l2930_293090


namespace largest_solution_l2930_293069

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ := x - floor x

/-- The equation from the problem -/
def equation (x : ℝ) : Prop := floor x = 6 + 50 * frac x

/-- The theorem stating the largest solution -/
theorem largest_solution :
  ∃ (x : ℝ), equation x ∧ ∀ (y : ℝ), equation y → y ≤ x :=
sorry

end largest_solution_l2930_293069


namespace lunchroom_tables_l2930_293028

theorem lunchroom_tables (students_per_table : ℕ) (total_students : ℕ) 
  (h1 : students_per_table = 6)
  (h2 : total_students = 204)
  (h3 : total_students % students_per_table = 0) :
  total_students / students_per_table = 34 := by
sorry

end lunchroom_tables_l2930_293028


namespace absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_30_l2930_293087

theorem absolute_value_equation_solution_difference : ℝ → Prop :=
  fun d => ∃ x₁ x₂ : ℝ,
    (|x₁ - 3| = 15) ∧
    (|x₂ - 3| = 15) ∧
    (x₁ ≠ x₂) ∧
    (d = |x₁ - x₂|) ∧
    (d = 30)

-- The proof is omitted
theorem absolute_value_equation_solution_difference_is_30 :
  absolute_value_equation_solution_difference 30 := by sorry

end absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_30_l2930_293087


namespace forty_nine_squared_equals_seven_to_zero_l2930_293021

theorem forty_nine_squared_equals_seven_to_zero : 49 * 49 = 7^0 := by
  sorry

end forty_nine_squared_equals_seven_to_zero_l2930_293021


namespace solution_set_f_less_than_4_range_of_a_for_solutions_f_min_value_f_min_condition_l2930_293059

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for part I
theorem solution_set_f_less_than_4 :
  {x : ℝ | f x < 4} = Set.Ioo (-2) 2 := by sorry

-- Theorem for part II
theorem range_of_a_for_solutions :
  {a : ℝ | ∃ x, f x - |a - 1| < 0} = Set.Iio (-1) ∪ Set.Ioi 3 := by sorry

-- Helper theorem: Minimum value of f is 2
theorem f_min_value :
  ∀ x : ℝ, f x ≥ 2 := by sorry

-- Helper theorem: Condition for f to achieve its minimum value
theorem f_min_condition (x : ℝ) :
  f x = 2 ↔ (x + 1) * (x - 1) ≤ 0 := by sorry

end solution_set_f_less_than_4_range_of_a_for_solutions_f_min_value_f_min_condition_l2930_293059


namespace square_root_divided_by_19_l2930_293018

theorem square_root_divided_by_19 : 
  Real.sqrt 5776 / 19 = 4 := by sorry

end square_root_divided_by_19_l2930_293018


namespace parallel_lines_k_value_l2930_293031

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 7) → k = 5 / 3 :=
by sorry

end parallel_lines_k_value_l2930_293031


namespace roots_sum_reciprocals_l2930_293060

theorem roots_sum_reciprocals (a b : ℝ) : 
  (a^2 - 3*a - 5 = 0) → 
  (b^2 - 3*b - 5 = 0) → 
  (a ≠ 0) →
  (b ≠ 0) →
  (1/a + 1/b = -3/5) := by
sorry

end roots_sum_reciprocals_l2930_293060


namespace congruence_problem_l2930_293097

theorem congruence_problem (x : ℤ) : (5 * x + 8) % 19 = 3 → (5 * x + 9) % 19 = 4 := by
  sorry

end congruence_problem_l2930_293097


namespace smallest_satisfying_number_l2930_293057

def satisfies_conditions (n : ℕ) : Prop :=
  ∀ d : ℕ, 2 ≤ d → d ≤ 10 → n % d = d - 1

theorem smallest_satisfying_number : 
  satisfies_conditions 2519 ∧ 
  ∀ m : ℕ, m < 2519 → ¬(satisfies_conditions m) :=
sorry

end smallest_satisfying_number_l2930_293057


namespace zoo_animal_ratio_l2930_293049

/-- Proves that the ratio of cheetahs to snakes is 7:10 given the zoo animal counts --/
theorem zoo_animal_ratio : 
  ∀ (snakes arctic_foxes leopards bee_eaters alligators cheetahs total : ℕ),
  snakes = 100 →
  arctic_foxes = 80 →
  leopards = 20 →
  bee_eaters = 10 * leopards →
  alligators = 2 * (arctic_foxes + leopards) →
  total = 670 →
  total = snakes + arctic_foxes + leopards + bee_eaters + alligators + cheetahs →
  (cheetahs : ℚ) / snakes = 7 / 10 :=
by
  sorry

end zoo_animal_ratio_l2930_293049


namespace isosceles_triangle_side_length_l2930_293046

/-- An isosceles triangle with one side length of 3 and perimeter of 7 has equal sides of length 3 or 2 -/
theorem isosceles_triangle_side_length (a b c : ℝ) : 
  a + b + c = 7 →  -- perimeter is 7
  (a = 3 ∨ b = 3 ∨ c = 3) →  -- one side length is 3
  ((a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)) →  -- isosceles condition
  (a = 3 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (b = 3 ∧ c = 3) ∨ (b = 2 ∧ c = 2) ∨ (a = 3 ∧ c = 3) ∨ (a = 2 ∧ c = 2) := by
sorry


end isosceles_triangle_side_length_l2930_293046


namespace tangent_perpendicular_line_l2930_293006

-- Define the curve
def C (x : ℝ) : ℝ := x^2 + x

-- Define the derivative of the curve
def C_derivative (x : ℝ) : ℝ := 2*x + 1

-- Define the slope of the tangent line at x = 1
def tangent_slope : ℝ := C_derivative 1

-- Define the condition for perpendicularity
def perpendicular_condition (a : ℝ) : Prop :=
  tangent_slope * a = -1

-- The theorem to prove
theorem tangent_perpendicular_line : 
  ∃ (a : ℝ), perpendicular_condition a ∧ a = -1/3 := by
  sorry

end tangent_perpendicular_line_l2930_293006


namespace ellipse_equation_equivalence_l2930_293020

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt (x^2 + (y + 3)^2) + Real.sqrt (x^2 + (y - 3)^2) = 10) ↔
  (x^2 / 25 + y^2 / 16 = 1) :=
by sorry

end ellipse_equation_equivalence_l2930_293020


namespace no_prime_solution_l2930_293016

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


end no_prime_solution_l2930_293016


namespace first_row_chairs_l2930_293004

/-- Given a sequence of chair counts in rows, prove that the first row has 14 chairs. -/
theorem first_row_chairs (chairs : ℕ → ℕ) : 
  chairs 2 = 23 →                    -- Second row has 23 chairs
  (∀ n ≥ 2, chairs (n + 1) = chairs n + 9) →  -- Each subsequent row increases by 9
  chairs 6 = 59 →                    -- Sixth row has 59 chairs
  chairs 1 = 14 :=                   -- First row has 14 chairs
by sorry

end first_row_chairs_l2930_293004


namespace tan_alpha_value_l2930_293000

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin (2 * α) + 2 * Real.cos (2 * α) = 2) : 
  Real.tan α = 1/2 := by sorry

end tan_alpha_value_l2930_293000


namespace half_plus_five_equals_eleven_l2930_293007

theorem half_plus_five_equals_eleven : 
  (12 / 2 : ℚ) + 5 = 11 := by sorry

end half_plus_five_equals_eleven_l2930_293007


namespace platform_length_l2930_293081

-- Define the train's properties
variable (l : ℝ) -- length of the train
variable (t : ℝ) -- time to pass a pole
variable (v : ℝ) -- velocity of the train

-- Define the platform
variable (p : ℝ) -- length of the platform

-- State the theorem
theorem platform_length 
  (h1 : v = l / t) -- velocity when passing the pole
  (h2 : v = (l + p) / (5 * t)) -- velocity when passing the platform
  : p = 4 * l := by
  sorry

end platform_length_l2930_293081


namespace equation_solution_l2930_293096

theorem equation_solution :
  ∃ x : ℚ, x - 1 ≠ 0 ∧ 1 - 1 / (x - 1) = 2 * x / (1 - x) ∧ x = 2 / 3 := by
  sorry

end equation_solution_l2930_293096


namespace negation_of_existence_l2930_293014

theorem negation_of_existence (x : ℝ) : 
  ¬(∃ x ≥ 0, x^2 - 2*x - 3 = 0) ↔ ∀ x ≥ 0, x^2 - 2*x - 3 ≠ 0 :=
by sorry

end negation_of_existence_l2930_293014


namespace sum_of_digits_of_large_number_l2930_293083

def large_number : ℕ := 3 * 10^500 - 2022 * 10^497 - 2022

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_large_number : sum_of_digits large_number = 4491 := by sorry

end sum_of_digits_of_large_number_l2930_293083


namespace rectangle_dimension_change_l2930_293048

theorem rectangle_dimension_change (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let new_L := 1.25 * L
  let new_W := W * (1 / 1.25)
  new_L * new_W = L * W ∧ (1 - new_W / W) * 100 = 20 := by
sorry

end rectangle_dimension_change_l2930_293048


namespace metallic_sheet_dimension_l2930_293065

/-- Given a rectangular metallic sheet with one dimension of 52 meters,
    if squares of 8 meters are cut from each corner to form an open box
    with a volume of 5760 cubic meters, then the length of the second
    dimension of the metallic sheet is 36 meters. -/
theorem metallic_sheet_dimension (w : ℝ) :
  w > 0 →
  (w - 2 * 8) * (52 - 2 * 8) * 8 = 5760 →
  w = 36 := by
  sorry

end metallic_sheet_dimension_l2930_293065


namespace building_height_average_l2930_293037

def measurements : List ℝ := [79.4, 80.6, 80.8, 79.1, 80, 79.6, 80.5]

theorem building_height_average : 
  (measurements.sum / measurements.length : ℝ) = 80 := by sorry

end building_height_average_l2930_293037


namespace purple_jellybeans_count_l2930_293001

theorem purple_jellybeans_count (total : ℕ) (blue : ℕ) (orange : ℕ) (red : ℕ) 
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_orange : orange = 40)
  (h_red : red = 120) :
  total - (blue + orange + red) = 26 := by
  sorry

end purple_jellybeans_count_l2930_293001


namespace function_existence_l2930_293009

theorem function_existence (A B : Type) [Fintype A] [Fintype B]
  (hA : Fintype.card A = 2011^2) (hB : Fintype.card B = 2010) :
  ∃ f : A × A → B,
    (∀ x y : A, f (x, y) = f (y, x)) ∧
    (∀ g : A → B, ∃ a₁ a₂ : A, a₁ ≠ a₂ ∧ g a₁ = f (a₁, a₂) ∧ f (a₁, a₂) = g a₂) := by
  sorry

end function_existence_l2930_293009


namespace trail_mix_pouches_per_pack_l2930_293098

theorem trail_mix_pouches_per_pack 
  (team_members : ℕ) 
  (coaches : ℕ) 
  (helpers : ℕ) 
  (total_packs : ℕ) 
  (h1 : team_members = 13)
  (h2 : coaches = 3)
  (h3 : helpers = 2)
  (h4 : total_packs = 3)
  : (team_members + coaches + helpers) / total_packs = 6 := by
  sorry

end trail_mix_pouches_per_pack_l2930_293098


namespace well_digging_time_l2930_293079

theorem well_digging_time 
  (combined_time : ℝ) 
  (paul_time : ℝ) 
  (hari_time : ℝ) 
  (h1 : combined_time = 8)
  (h2 : paul_time = 24)
  (h3 : hari_time = 48) : 
  ∃ jake_time : ℝ, 
    jake_time = 16 ∧ 
    1 / combined_time = 1 / jake_time + 1 / paul_time + 1 / hari_time :=
by sorry

end well_digging_time_l2930_293079


namespace final_dog_count_l2930_293022

/-- Calculates the number of dogs remaining in the rescue center at the end of the month -/
def dogsRemaining (initial : ℕ) (arrivals : List ℕ) (adoptions : List ℕ) (returned : ℕ) : ℕ :=
  let weeklyChanges := List.zipWith (λ a b => a - b) arrivals adoptions
  initial + weeklyChanges.sum - returned

theorem final_dog_count :
  let initial : ℕ := 200
  let arrivals : List ℕ := [30, 40, 30]
  let adoptions : List ℕ := [40, 50, 30, 70]
  let returned : ℕ := 20
  dogsRemaining initial arrivals adoptions returned = 90 := by
  sorry

end final_dog_count_l2930_293022


namespace negation_of_proposition_negation_of_specific_proposition_l2930_293042

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x > 1, p x) ↔ (∃ x > 1, ¬ p x) :=
by sorry

theorem negation_of_specific_proposition : 
  (¬ ∀ x > 1, x^3 + 16 > 8*x) ↔ (∃ x > 1, x^3 + 16 ≤ 8*x) :=
by sorry

end negation_of_proposition_negation_of_specific_proposition_l2930_293042


namespace staircase_markups_l2930_293011

/-- Represents the number of different markups for a staircase with n cells -/
def L (n : ℕ) : ℕ := n + 1

/-- Theorem stating that the number of different markups for a staircase with n cells is n + 1 -/
theorem staircase_markups (n : ℕ) : L n = n + 1 := by
  sorry

end staircase_markups_l2930_293011


namespace order_of_expressions_l2930_293052

theorem order_of_expressions : 3^(1/2) > Real.log (1/2) / Real.log (1/3) ∧ 
  Real.log (1/2) / Real.log (1/3) > Real.log (1/3) / Real.log 2 := by
  sorry

end order_of_expressions_l2930_293052


namespace triangle_perimeter_increase_l2930_293085

/-- Given an initial equilateral triangle and four subsequent triangles with increasing side lengths,
    calculate the percent increase in perimeter from the first to the fifth triangle. -/
theorem triangle_perimeter_increase (initial_side : ℝ) (scale_factor : ℝ) (num_triangles : ℕ) :
  initial_side = 3 →
  scale_factor = 2 →
  num_triangles = 5 →
  let first_perimeter := 3 * initial_side
  let last_side := initial_side * scale_factor ^ (num_triangles - 1)
  let last_perimeter := 3 * last_side
  (last_perimeter - first_perimeter) / first_perimeter * 100 = 1500 := by
  sorry

end triangle_perimeter_increase_l2930_293085


namespace cubic_root_sum_cubes_l2930_293078

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (9 * a ^ 3 + 14 * a ^ 2 + 2047 * a + 3024 = 0) →
  (9 * b ^ 3 + 14 * b ^ 2 + 2047 * b + 3024 = 0) →
  (9 * c ^ 3 + 14 * c ^ 2 + 2047 * c + 3024 = 0) →
  (a + b) ^ 3 + (b + c) ^ 3 + (c + a) ^ 3 = -58198 / 729 := by
sorry

end cubic_root_sum_cubes_l2930_293078


namespace sally_quarters_l2930_293064

/-- The number of quarters Sally has after her purchases -/
def remaining_quarters (initial : ℕ) (purchase1 : ℕ) (purchase2 : ℕ) : ℕ :=
  initial - purchase1 - purchase2

/-- Theorem stating that Sally has 150 quarters left after her purchases -/
theorem sally_quarters : remaining_quarters 760 418 192 = 150 := by
  sorry

end sally_quarters_l2930_293064


namespace greatest_possible_award_l2930_293080

theorem greatest_possible_award (total_prize : ℕ) (num_winners : ℕ) (min_award : ℕ) 
  (h1 : total_prize = 800)
  (h2 : num_winners = 20)
  (h3 : min_award = 20)
  (h4 : (2 : ℚ) / 5 * total_prize = (3 : ℚ) / 5 * num_winners * min_award) :
  ∃ (max_award : ℕ), max_award = 420 ∧ 
    (∀ (award : ℕ), award > max_award → 
      ¬(∃ (awards : List ℕ), awards.length = num_winners ∧ 
        awards.sum = total_prize ∧ 
        (∀ x ∈ awards, x ≥ min_award) ∧
        award ∈ awards)) :=
by sorry

end greatest_possible_award_l2930_293080


namespace class_size_calculation_l2930_293068

theorem class_size_calculation (incorrect_mark : ℕ) (correct_mark : ℕ) (average_increase : ℚ) : 
  incorrect_mark = 67 → 
  correct_mark = 45 → 
  average_increase = 1/2 →
  (incorrect_mark - correct_mark : ℚ) / (2 * average_increase) = 44 :=
by sorry

end class_size_calculation_l2930_293068


namespace opposite_of_negative_three_l2930_293024

theorem opposite_of_negative_three : -(- 3) = 3 := by sorry

end opposite_of_negative_three_l2930_293024


namespace money_difference_l2930_293055

/-- Proves that Hoseok has 170,000 won more than Min-young after they both earn additional money -/
theorem money_difference (initial_amount : ℕ) (minyoung_earnings hoseok_earnings : ℕ) :
  initial_amount = 1500000 →
  minyoung_earnings = 320000 →
  hoseok_earnings = 490000 →
  (initial_amount + hoseok_earnings) - (initial_amount + minyoung_earnings) = 170000 :=
by
  sorry

end money_difference_l2930_293055


namespace distance_to_line_l2930_293002

/-- Represents a line in 2D space using parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Calculates the distance from a point to a line given in parametric form --/
def distanceToParametricLine (px py : ℝ) (line : ParametricLine) : ℝ :=
  sorry

/-- The problem statement --/
theorem distance_to_line : 
  let l : ParametricLine := { x := λ t => 1 + t, y := λ t => -1 + t }
  let p : (ℝ × ℝ) := (4, 0)
  distanceToParametricLine p.1 p.2 l = Real.sqrt 2 := by
  sorry

end distance_to_line_l2930_293002


namespace expression_value_at_three_l2930_293076

theorem expression_value_at_three :
  let x : ℕ := 3
  x + x * (x ^ x) + x ^ 3 = 111 := by sorry

end expression_value_at_three_l2930_293076


namespace product_of_repeating_third_and_nine_l2930_293074

/-- The repeating decimal 0.333... -/
def repeating_third : ℚ := 1/3

theorem product_of_repeating_third_and_nine :
  repeating_third * 9 = 3 := by sorry

end product_of_repeating_third_and_nine_l2930_293074


namespace max_n_for_consecutive_product_l2930_293073

theorem max_n_for_consecutive_product : ∃ (n_max : ℕ), ∀ (n : ℕ), 
  (∃ (k : ℕ), 9*n^2 + 5*n + 26 = k * (k+1)) → n ≤ n_max :=
sorry

end max_n_for_consecutive_product_l2930_293073


namespace complex_simplification_l2930_293041

theorem complex_simplification :
  (4 - 3*Complex.I) - (7 + 5*Complex.I) + 2*(1 - 2*Complex.I) = -1 - 12*Complex.I :=
by sorry

end complex_simplification_l2930_293041


namespace log_5_12_equals_fraction_l2930_293032

-- Define the common logarithm (base 10) function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the logarithm with base 5
noncomputable def log_5 (x : ℝ) := Real.log x / Real.log 5

theorem log_5_12_equals_fraction (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  log_5 12 = (2 * a + b) / (1 - a) := by
  sorry

end log_5_12_equals_fraction_l2930_293032


namespace president_vp_committee_selection_l2930_293033

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem president_vp_committee_selection (n : ℕ) (h : n = 10) : 
  n * (n - 1) * choose (n - 2) 2 = 2520 := by
  sorry

end president_vp_committee_selection_l2930_293033


namespace tangent_line_and_inequalities_l2930_293047

noncomputable def f (x : ℝ) := x - x^2 + 3 * Real.log x

theorem tangent_line_and_inequalities :
  (∃ x₀ : ℝ, x₀ > 0 ∧ (∀ x > 0, f x ≤ 2 * x - 2) ∧
   (∀ k < 2, ∃ x₁ > 1, ∀ x ∈ Set.Ioo 1 x₁, f x ≥ k * (x - 1))) ∧
  (∃ a b : ℝ, ∀ x > 0, f x = 2 * x - 2 → x = a ∧ f x = b) :=
by sorry

end tangent_line_and_inequalities_l2930_293047


namespace constant_term_binomial_expansion_l2930_293066

/-- The constant term in the expansion of (x^2 - 2/x)^6 is 240 -/
theorem constant_term_binomial_expansion :
  let f (x : ℝ) := (x^2 - 2/x)^6
  ∃ c : ℝ, (∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ c = 240 := by
  sorry

end constant_term_binomial_expansion_l2930_293066


namespace abc_inequality_l2930_293091

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1)
  (eq_a : a = 2022 * Real.exp (a - 2022))
  (eq_b : b = 2023 * Real.exp (b - 2023))
  (eq_c : c = 2024 * Real.exp (c - 2024)) :
  c < b ∧ b < a := by sorry

end abc_inequality_l2930_293091


namespace divides_n_l2930_293015

def n : ℕ := sorry

theorem divides_n : 1980 ∣ n := by sorry

end divides_n_l2930_293015


namespace unique_prime_seventh_power_l2930_293053

theorem unique_prime_seventh_power (p : ℕ) : 
  Prime p ∧ ∃ q : ℕ, Prime q ∧ p + 25 = q^7 ↔ p = 103 :=
by sorry

end unique_prime_seventh_power_l2930_293053


namespace max_intersection_area_l2930_293092

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

theorem max_intersection_area :
  ∀ (r1 r2 : Rectangle),
    r1.height < r1.width →
    r2.height > r2.width →
    r1.area = 2015 →
    r2.area = 2016 →
    (∀ r : Rectangle,
      r.width ≤ min r1.width r2.width ∧
      r.height ≤ min r1.height r2.height →
      r.area ≤ 1302) ∧
    (∃ r : Rectangle,
      r.width ≤ min r1.width r2.width ∧
      r.height ≤ min r1.height r2.height ∧
      r.area = 1302) := by
  sorry

end max_intersection_area_l2930_293092


namespace three_greater_than_sqrt_seven_l2930_293094

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end three_greater_than_sqrt_seven_l2930_293094


namespace james_training_hours_l2930_293043

/-- Represents James' training schedule and conditions --/
structure TrainingSchedule where
  daysInYear : Nat
  weekdayTrainingHours : Nat
  vacationWeeks : Nat
  injuryDays : Nat
  competitionDays : Nat

/-- Calculates the total training hours for James in a non-leap year --/
def calculateTrainingHours (schedule : TrainingSchedule) : Nat :=
  let weekdays := schedule.daysInYear - (52 * 2)
  let trainingDays := weekdays - (schedule.vacationWeeks * 5) - schedule.injuryDays - schedule.competitionDays
  let trainingWeeks := trainingDays / 5
  trainingWeeks * (5 * schedule.weekdayTrainingHours)

/-- Theorem stating that James' total training hours in a non-leap year is 1904 --/
theorem james_training_hours :
  let schedule : TrainingSchedule := {
    daysInYear := 365,
    weekdayTrainingHours := 8,
    vacationWeeks := 2,
    injuryDays := 5,
    competitionDays := 8
  }
  calculateTrainingHours schedule = 1904 := by
  sorry

end james_training_hours_l2930_293043


namespace three_numbers_sum_l2930_293005

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 8 → 
  (a + b + c) / 3 = a + 12 →
  (a + b + c) / 3 = c - 20 →
  a + b + c = 48 := by
sorry

end three_numbers_sum_l2930_293005


namespace problem_statement_l2930_293023

theorem problem_statement (a b : ℕ+) (h : 8 * (a : ℝ)^(a : ℝ) * (b : ℝ)^(b : ℝ) = 27 * (a : ℝ)^(b : ℝ) * (b : ℝ)^(a : ℝ)) : 
  (a : ℝ)^2 + (b : ℝ)^2 = 117 := by
  sorry

end problem_statement_l2930_293023


namespace triangle_side_sum_range_l2930_293029

theorem triangle_side_sum_range (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- positive side lengths
  (a + b > c ∧ b + c > a ∧ a + c > b) →  -- triangle inequality
  (∃ x : ℝ, x^2 - (a + b)*x + a*b = 0) →  -- a and b are roots of the quadratic equation
  (a < b) →  -- given condition
  (7/8 < a + b - c ∧ a + b - c < Real.sqrt 5 - 1) := by
sorry

end triangle_side_sum_range_l2930_293029


namespace fibFactorial_characterization_l2930_293075

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Set of positive integers n for which n! is the product of two Fibonacci numbers -/
def fibFactorialSet : Set ℕ :=
  {n : ℕ | n > 0 ∧ ∃ k m : ℕ, n.factorial = fib k * fib m}

/-- Theorem stating that fibFactorialSet contains exactly 1, 2, 3, 4, and 6 -/
theorem fibFactorial_characterization :
    fibFactorialSet = {1, 2, 3, 4, 6} := by
  sorry

end fibFactorial_characterization_l2930_293075
