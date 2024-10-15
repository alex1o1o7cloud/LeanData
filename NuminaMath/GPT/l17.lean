import Mathlib

namespace NUMINAMATH_GPT_john_weekly_earnings_l17_1784

/-- John takes 3 days off of streaming per week. 
    John streams for 4 hours at a time on the days he does stream.
    John makes $10 an hour.
    Prove that John makes $160 a week. -/

theorem john_weekly_earnings (days_off : ℕ) (hours_per_day : ℕ) (wage_per_hour : ℕ) 
  (h_days_off : days_off = 3) (h_hours_per_day : hours_per_day = 4) 
  (h_wage_per_hour : wage_per_hour = 10) : 
  7 - days_off * hours_per_day * wage_per_hour = 160 := by
  sorry

end NUMINAMATH_GPT_john_weekly_earnings_l17_1784


namespace NUMINAMATH_GPT_circumference_greater_than_100_l17_1783

def running_conditions (A B : ℝ) (C : ℝ) (P : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ A ≠ B ∧ P = 0 ∧ C > 0

theorem circumference_greater_than_100 (A B C P : ℝ) (h : running_conditions A B C P):
  C > 100 :=
by
  sorry

end NUMINAMATH_GPT_circumference_greater_than_100_l17_1783


namespace NUMINAMATH_GPT_exists_parallelogram_marked_cells_l17_1750

theorem exists_parallelogram_marked_cells (n : ℕ) (marked : Finset (Fin n × Fin n)) (h_marked : marked.card = 2 * n) :
  ∃ (a b c d : Fin n × Fin n), a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧ d ∈ marked ∧ 
  ((a.1 = b.1) ∧ (c.1 = d.1) ∧ (a.2 = c.2) ∧ (b.2 = d.2)) :=
sorry

end NUMINAMATH_GPT_exists_parallelogram_marked_cells_l17_1750


namespace NUMINAMATH_GPT_clara_boxes_l17_1790

theorem clara_boxes (x : ℕ)
  (h1 : 12 * x + 20 * 80 + 16 * 70 = 3320) : x = 50 := by
  sorry

end NUMINAMATH_GPT_clara_boxes_l17_1790


namespace NUMINAMATH_GPT_no_flippy_numbers_divisible_by_11_and_6_l17_1715

def is_flippy (n : ℕ) : Prop :=
  let d1 := n / 10000
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  (d1 = d3 ∧ d3 = d5 ∧ d2 = d4 ∧ d1 ≠ d2) ∨ 
  (d2 = d4 ∧ d4 = d5 ∧ d1 = d3 ∧ d1 ≠ d2)

def is_divisible_by_11 (n : ℕ) : Prop :=
  (n % 11) = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10000) + (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

def sum_divisible_by_6 (n : ℕ) : Prop :=
  (sum_of_digits n) % 6 = 0

theorem no_flippy_numbers_divisible_by_11_and_6 :
  ∀ n, (10000 ≤ n ∧ n < 100000) → is_flippy n → is_divisible_by_11 n → sum_divisible_by_6 n → false :=
by
  intros n h_range h_flippy h_div11 h_sum6
  sorry

end NUMINAMATH_GPT_no_flippy_numbers_divisible_by_11_and_6_l17_1715


namespace NUMINAMATH_GPT_total_revenue_correct_l17_1732

-- Defining the basic parameters
def ticket_price : ℝ := 20
def first_discount_percentage : ℝ := 0.40
def next_discount_percentage : ℝ := 0.15
def first_people : ℕ := 10
def next_people : ℕ := 20
def total_people : ℕ := 48

-- Calculate the discounted prices based on the given percentages
def discounted_price_first : ℝ := ticket_price * (1 - first_discount_percentage)
def discounted_price_next : ℝ := ticket_price * (1 - next_discount_percentage)

-- Calculate the total revenue
def revenue_first : ℝ := first_people * discounted_price_first
def revenue_next : ℝ := next_people * discounted_price_next
def remaining_people : ℕ := total_people - first_people - next_people
def revenue_remaining : ℝ := remaining_people * ticket_price

def total_revenue : ℝ := revenue_first + revenue_next + revenue_remaining

-- The statement to be proved
theorem total_revenue_correct : total_revenue = 820 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l17_1732


namespace NUMINAMATH_GPT_no_n_for_equal_sums_l17_1794

theorem no_n_for_equal_sums (n : ℕ) (h : n ≠ 0) :
  let s1 := (3 * n^2 + 7 * n) / 2
  let s2 := (3 * n^2 + 37 * n) / 2
  s1 ≠ s2 :=
by
  let s1 := (3 * n^2 + 7 * n) / 2
  let s2 := (3 * n^2 + 37 * n) / 2
  sorry

end NUMINAMATH_GPT_no_n_for_equal_sums_l17_1794


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l17_1787

theorem sufficient_but_not_necessary_condition (x y m : ℝ) (h: x^2 + y^2 - 4 * x + 2 * y + m = 0):
  (m = 0) → (5 > m) ∧ ((5 > m) → (m ≠ 0)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l17_1787


namespace NUMINAMATH_GPT_sincos_terminal_side_l17_1719

noncomputable def sincos_expr (α : ℝ) :=
  let P : ℝ × ℝ := (-4, 3)
  let r := Real.sqrt (P.1 ^ 2 + P.2 ^ 2)
  let sinα := P.2 / r
  let cosα := P.1 / r
  sinα + 2 * cosα = -1

theorem sincos_terminal_side :
  sincos_expr α :=
by
  sorry

end NUMINAMATH_GPT_sincos_terminal_side_l17_1719


namespace NUMINAMATH_GPT_base_measurement_zions_house_l17_1700

-- Given conditions
def height_zion_house : ℝ := 20
def total_area_three_houses : ℝ := 1200
def num_houses : ℝ := 3

-- Correct answer
def base_zion_house : ℝ := 40

-- Proof statement (question translated to lean statement)
theorem base_measurement_zions_house :
  ∃ base : ℝ, (height_zion_house = 20 ∧ total_area_three_houses = 1200 ∧ num_houses = 3) →
  base = base_zion_house :=
by
  sorry

end NUMINAMATH_GPT_base_measurement_zions_house_l17_1700


namespace NUMINAMATH_GPT_course_selection_schemes_count_l17_1751

-- Define the total number of courses
def total_courses : ℕ := 8

-- Define the number of courses to choose
def courses_to_choose : ℕ := 5

-- Define the two specific courses, Course A and Course B
def courseA := 1
def courseB := 2

-- Define the combination function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the count when neither Course A nor Course B is selected
def case1 : ℕ := C 6 5

-- Define the count when exactly one of Course A or Course B is selected
def case2 : ℕ := C 2 1 * C 6 4

-- Combining both cases
theorem course_selection_schemes_count : case1 + case2 = 36 :=
by
  -- These would be replaced with actual combination calculations.
  sorry

end NUMINAMATH_GPT_course_selection_schemes_count_l17_1751


namespace NUMINAMATH_GPT_factor_difference_of_squares_l17_1730

theorem factor_difference_of_squares (a b p q : ℝ) :
  (∃ c d : ℝ, -a ^ 2 + 9 = c ^ 2 - d ^ 2) ∧
  (¬(∃ c d : ℝ, -a ^ 2 - b ^ 2 = c ^ 2 - d ^ 2)) ∧
  (¬(∃ c d : ℝ, p ^ 2 - (-q ^ 2) = c ^ 2 - d ^ 2)) ∧
  (¬(∃ c d : ℝ, a ^ 2 - b ^ 3 = c ^ 2 - d ^ 2)) := 
  by 
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l17_1730


namespace NUMINAMATH_GPT_smallest_m_l17_1735

theorem smallest_m (m : ℕ) (h1 : m > 0) (h2 : 3 ^ ((m + m ^ 2) / 4) > 500) : m = 5 := 
by sorry

end NUMINAMATH_GPT_smallest_m_l17_1735


namespace NUMINAMATH_GPT_solve_for_x_l17_1769

theorem solve_for_x (x : ℚ) : x^2 + 125 = (x - 15)^2 → x = 10 / 3 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l17_1769


namespace NUMINAMATH_GPT_eval_expression_l17_1723

theorem eval_expression : 4 * (8 - 3) - 7 = 13 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l17_1723


namespace NUMINAMATH_GPT_chord_central_angle_l17_1726

-- Given that a chord divides the circumference of a circle in the ratio 5:7
-- Prove that the central angle opposite this chord can be either 75° or 105°
theorem chord_central_angle (x : ℝ) (h : 5 * x + 7 * x = 180) :
  5 * x = 75 ∨ 7 * x = 105 :=
sorry

end NUMINAMATH_GPT_chord_central_angle_l17_1726


namespace NUMINAMATH_GPT_find_range_of_values_l17_1776

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem find_range_of_values (f : ℝ → ℝ) (h_even : is_even f)
  (h_increasing : is_increasing_on_nonneg f) (h_f1_zero : f 1 = 0) :
  { x : ℝ | f (Real.log x / Real.log (1/2)) > 0 } = 
  { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | x > 2 } :=
by 
  sorry

end NUMINAMATH_GPT_find_range_of_values_l17_1776


namespace NUMINAMATH_GPT_correct_equation_l17_1718

/-- Definitions and conditions used in the problem -/
def jan_revenue := 250
def feb_revenue (x : ℝ) := jan_revenue * (1 + x)
def mar_revenue (x : ℝ) := jan_revenue * (1 + x)^2
def first_quarter_target := 900

/-- Proof problem statement -/
theorem correct_equation (x : ℝ) : 
  jan_revenue + feb_revenue x + mar_revenue x = first_quarter_target := 
by
  sorry

end NUMINAMATH_GPT_correct_equation_l17_1718


namespace NUMINAMATH_GPT_point_in_third_quadrant_l17_1762

noncomputable def is_second_quadrant (a b : ℝ) : Prop :=
a < 0 ∧ b > 0

noncomputable def is_third_quadrant (a b : ℝ) : Prop :=
a < 0 ∧ b < 0

theorem point_in_third_quadrant (a b : ℝ) (h : is_second_quadrant a b) : is_third_quadrant a (-b) :=
by
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l17_1762


namespace NUMINAMATH_GPT_cream_butterfat_percentage_l17_1746

theorem cream_butterfat_percentage (x : ℝ) (h1 : 1 * (x / 100) + 3 * (5.5 / 100) = 4 * (6.5 / 100)) : 
  x = 9.5 :=
by
  sorry

end NUMINAMATH_GPT_cream_butterfat_percentage_l17_1746


namespace NUMINAMATH_GPT_find_brick_length_l17_1725

-- Conditions as given in the problem.
def wall_length : ℝ := 8
def wall_width : ℝ := 6
def wall_height : ℝ := 22.5
def number_of_bricks : ℕ := 6400
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- The volume of the wall in cubic centimeters.
def wall_volume_cm_cube : ℝ := (wall_length * 100) * (wall_width * 100) * (wall_height * 100)

-- Define the volume of one brick based on the unknown length L.
def brick_volume (L : ℝ) : ℝ := L * brick_width * brick_height

-- Define an equivalence for the total volume of the bricks to the volume of the wall.
theorem find_brick_length : 
  ∃ (L : ℝ), wall_volume_cm_cube = brick_volume L * number_of_bricks ∧ L = 2500 := 
by
  sorry

end NUMINAMATH_GPT_find_brick_length_l17_1725


namespace NUMINAMATH_GPT_ocean_depth_at_base_of_cone_l17_1786

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

noncomputable def submerged_height_fraction (total_height volume_fraction : ℝ) : ℝ :=
  total_height * (volume_fraction)^(1/3)

theorem ocean_depth_at_base_of_cone (total_height radius : ℝ) 
  (above_water_volume_fraction : ℝ) : ℝ :=
  let above_water_height := submerged_height_fraction total_height above_water_volume_fraction
  total_height - above_water_height

example : ocean_depth_at_base_of_cone 10000 2000 (3 / 5) = 1566 := by
  sorry

end NUMINAMATH_GPT_ocean_depth_at_base_of_cone_l17_1786


namespace NUMINAMATH_GPT_walking_west_10_neg_l17_1720

-- Define the condition that walking east for 20 meters is +20 meters
def walking_east_20 := 20

-- Assert that walking west for 10 meters is -10 meters given the east direction definition
theorem walking_west_10_neg : walking_east_20 = 20 → (-10 = -10) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_walking_west_10_neg_l17_1720


namespace NUMINAMATH_GPT_weight_range_correct_l17_1799

noncomputable def combined_weight : ℕ := 158
noncomputable def tracy_weight : ℕ := 52
noncomputable def jake_weight : ℕ := tracy_weight + 8
noncomputable def john_weight : ℕ := combined_weight - (tracy_weight + jake_weight)
noncomputable def weight_range : ℕ := jake_weight - john_weight

theorem weight_range_correct : weight_range = 14 := 
by
  sorry

end NUMINAMATH_GPT_weight_range_correct_l17_1799


namespace NUMINAMATH_GPT_smallest_positive_integer_divisible_by_10_13_14_l17_1706

theorem smallest_positive_integer_divisible_by_10_13_14 : ∃ n : ℕ, n > 0 ∧ (10 ∣ n) ∧ (13 ∣ n) ∧ (14 ∣ n) ∧ n = 910 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_positive_integer_divisible_by_10_13_14_l17_1706


namespace NUMINAMATH_GPT_exists_real_number_lt_neg_one_l17_1729

theorem exists_real_number_lt_neg_one : ∃ (x : ℝ), x < -1 := by
  sorry

end NUMINAMATH_GPT_exists_real_number_lt_neg_one_l17_1729


namespace NUMINAMATH_GPT_central_angle_radian_l17_1755

-- Define the context of the sector and conditions
def sector (r θ : ℝ) :=
  θ = r * 6 ∧ 1/2 * r^2 * θ = 6

-- Define the radian measure of the central angle
theorem central_angle_radian (r : ℝ) (θ : ℝ) (h : sector r θ) : θ = 3 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_radian_l17_1755


namespace NUMINAMATH_GPT_sum_of_possible_N_values_l17_1789

theorem sum_of_possible_N_values (a b c N : ℕ) (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = 2 * (a + b)) :
  ∃ sum_N : ℕ, sum_N = 672 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_N_values_l17_1789


namespace NUMINAMATH_GPT_upstream_distance_l17_1749

-- Define the conditions
def velocity_current : ℝ := 1.5
def distance_downstream : ℝ := 32
def time : ℝ := 6

-- Define the speed of the man in still water
noncomputable def speed_in_still_water : ℝ := (distance_downstream / time) - velocity_current

-- Define the distance rowed upstream
noncomputable def distance_upstream : ℝ := (speed_in_still_water - velocity_current) * time

-- The theorem statement to be proved
theorem upstream_distance (v c d : ℝ) (h1 : c = 1.5) (h2 : (v + c) * 6 = 32) (h3 : (v - c) * 6 = d) : d = 14 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_upstream_distance_l17_1749


namespace NUMINAMATH_GPT_odd_function_f_neg_one_l17_1771

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 2 then 2^x else 0 -- Placeholder; actual implementation skipped for simplicity

theorem odd_function_f_neg_one :
  (∀ x, f (-x) = -f x) ∧ (∀ x, (0 < x ∧ x < 2) → f x = 2^x) → 
  f (-1) = -2 :=
by
  intros h
  let odd_property := h.1
  let condition_in_range := h.2
  sorry

end NUMINAMATH_GPT_odd_function_f_neg_one_l17_1771


namespace NUMINAMATH_GPT_div_1947_l17_1782

theorem div_1947 (n : ℕ) (hn : n % 2 = 1) : 1947 ∣ (46^n + 296 * 13^n) :=
by
  sorry

end NUMINAMATH_GPT_div_1947_l17_1782


namespace NUMINAMATH_GPT_melinda_payment_l17_1745

theorem melinda_payment
  (D C : ℝ)
  (h1 : 3 * D + 4 * C = 4.91)
  (h2 : D = 0.45) :
  5 * D + 6 * C = 7.59 := 
by 
-- proof steps go here
sorry

end NUMINAMATH_GPT_melinda_payment_l17_1745


namespace NUMINAMATH_GPT_quilt_patch_cost_l17_1775

-- Definitions of the conditions
def length : ℕ := 16
def width : ℕ := 20
def patch_area : ℕ := 4
def cost_first_10 : ℕ := 10
def cost_after_10 : ℕ := 5
def num_first_patches : ℕ := 10

-- Define the calculations based on the problem conditions
def quilt_area : ℕ := length * width
def total_patches : ℕ := quilt_area / patch_area
def cost_first : ℕ := num_first_patches * cost_first_10
def remaining_patches : ℕ := total_patches - num_first_patches
def cost_remaining : ℕ := remaining_patches * cost_after_10
def total_cost : ℕ := cost_first + cost_remaining

-- Statement of the proof problem
theorem quilt_patch_cost : total_cost = 450 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_quilt_patch_cost_l17_1775


namespace NUMINAMATH_GPT_probability_single_draws_probability_two_different_colors_l17_1765

-- Define probabilities for black, yellow and green as events A, B, and C respectively.
variables (A B C : ℝ)

-- Conditions based on the problem statement
axiom h1 : A + B = 5/9
axiom h2 : B + C = 2/3
axiom h3 : A + B + C = 1

-- Here is the statement to prove the calculated probabilities of single draws
theorem probability_single_draws : 
  A = 1/3 ∧ B = 2/9 ∧ C = 4/9 :=
sorry

-- Define the event of drawing two balls of the same color
variables (black yellow green : ℕ)
axiom balls_count : black + yellow + green = 9
axiom black_component : A = black / 9
axiom yellow_component : B = yellow / 9
axiom green_component : C = green / 9

-- Using the counts to infer the probability of drawing two balls of different colors
axiom h4 : black = 3
axiom h5 : yellow = 2
axiom h6 : green = 4

theorem probability_two_different_colors :
  (1 - (3/36 + 1/36 + 6/36)) = 13/18 :=
sorry

end NUMINAMATH_GPT_probability_single_draws_probability_two_different_colors_l17_1765


namespace NUMINAMATH_GPT_rectangle_width_l17_1756

theorem rectangle_width (w : ℝ) 
  (h1 : ∃ w : ℝ, w > 0 ∧ (2 * w + 2 * (w - 2)) = 16) 
  (h2 : ∀ w, w > 0 → 2 * w + 2 * (w - 2) = 16 → w = 5) : 
  w = 5 := 
sorry

end NUMINAMATH_GPT_rectangle_width_l17_1756


namespace NUMINAMATH_GPT_smallest_non_lucky_multiple_of_8_l17_1738

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky_integer (n : ℕ) : Prop :=
  n % (sum_of_digits n) = 0

theorem smallest_non_lucky_multiple_of_8 : ∃ (m : ℕ), (m > 0) ∧ (m % 8 = 0) ∧ ¬ is_lucky_integer m ∧ m = 16 := sorry

end NUMINAMATH_GPT_smallest_non_lucky_multiple_of_8_l17_1738


namespace NUMINAMATH_GPT_triangle_PQR_not_right_l17_1747

-- Definitions based on conditions
def isIsosceles (a b c : ℝ) (angle1 angle2 : ℝ) : Prop := (angle1 = angle2) ∧ (a = c)

def perimeter (a b c : ℝ) : ℝ := a + b + c

def isRightTriangle (a b c : ℝ) : Prop := a * a = b * b + c * c

-- Given conditions
def PQR : ℝ := 10
def PRQ : ℝ := 10
def QR : ℝ := 6
def angle_PQR : ℝ := 1
def angle_PRQ : ℝ := 1

-- Lean statement for the proof problem
theorem triangle_PQR_not_right 
  (h1 : isIsosceles PQR QR PRQ angle_PQR angle_PRQ)
  (h2 : QR = 6)
  (h3 : PRQ = 10):
  ¬ isRightTriangle PQR QR PRQ ∧ perimeter PQR QR PRQ = 26 :=
by {
    sorry
}

end NUMINAMATH_GPT_triangle_PQR_not_right_l17_1747


namespace NUMINAMATH_GPT_complement_intersection_l17_1741

open Set

-- Definitions of U, A, and B
def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- Proof statement
theorem complement_intersection : 
  ((U \ A) ∩ (U \ B)) = ({0, 2, 4} : Set ℕ) :=
by sorry

end NUMINAMATH_GPT_complement_intersection_l17_1741


namespace NUMINAMATH_GPT_delta_value_l17_1774

theorem delta_value (Δ : ℤ) (h : 4 * (-3) = Δ - 3) : Δ = -9 :=
by {
  sorry
}

end NUMINAMATH_GPT_delta_value_l17_1774


namespace NUMINAMATH_GPT_max_colored_nodes_without_cycle_in_convex_polygon_l17_1780

def convex_polygon (n : ℕ) : Prop := n ≥ 3

def valid_diagonals (n : ℕ) : Prop := n = 2019

def no_three_diagonals_intersect_at_single_point (x : Type*) : Prop :=
  sorry -- You can provide a formal definition here based on combinatorial geometry.

def no_loops (n : ℕ) (k : ℕ) : Prop :=
  k ≤ (n * (n - 3)) / 2 - 1

theorem max_colored_nodes_without_cycle_in_convex_polygon :
  convex_polygon 2019 →
  valid_diagonals 2019 →
  no_three_diagonals_intersect_at_single_point ℝ →
  ∃ k, k = 2035151 ∧ no_loops 2019 k := 
by
  -- The proof would be constructed here.
  sorry

end NUMINAMATH_GPT_max_colored_nodes_without_cycle_in_convex_polygon_l17_1780


namespace NUMINAMATH_GPT_cos_alpha_is_negative_four_fifths_l17_1778

variable (α : ℝ)
variable (H1 : Real.sin α = 3 / 5)
variable (H2 : π / 2 < α ∧ α < π)

theorem cos_alpha_is_negative_four_fifths (H1 : Real.sin α = 3 / 5) (H2 : π / 2 < α ∧ α < π) :
  Real.cos α = -4 / 5 :=
sorry

end NUMINAMATH_GPT_cos_alpha_is_negative_four_fifths_l17_1778


namespace NUMINAMATH_GPT_std_dev_of_normal_distribution_l17_1716

theorem std_dev_of_normal_distribution (μ σ : ℝ) (h1: μ = 14.5) (h2: μ - 2 * σ = 11.5) : σ = 1.5 := 
by 
  sorry

end NUMINAMATH_GPT_std_dev_of_normal_distribution_l17_1716


namespace NUMINAMATH_GPT_general_term_of_A_inter_B_l17_1727

def setA : Set ℕ := { n*n + n | n : ℕ }
def setB : Set ℕ := { 3*m - 1 | m : ℕ }

theorem general_term_of_A_inter_B (k : ℕ) :
  let a_k := 9*k^2 - 9*k + 2
  a_k ∈ setA ∩ setB ∧ ∀ n ∈ setA ∩ setB, n = a_k :=
sorry

end NUMINAMATH_GPT_general_term_of_A_inter_B_l17_1727


namespace NUMINAMATH_GPT_intersection_points_l17_1733

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 15) / (3*x - 6)

noncomputable def g (x : ℝ) : ℝ := (-3*x^2 - 6*x + 115) / (x - 2)

theorem intersection_points:
  ∃ (x1 x2 : ℝ), x1 ≠ -3 ∧ x2 ≠ -3 ∧ (f x1 = g x1) ∧ (f x2 = g x2) ∧ 
  (x1 = -11 ∧ f x1 = -2) ∧ (x2 = 3 ∧ f x2 = -2) := 
sorry

end NUMINAMATH_GPT_intersection_points_l17_1733


namespace NUMINAMATH_GPT_cows_in_group_l17_1777

theorem cows_in_group (c h : ℕ) (h_condition : 4 * c + 2 * h = 2 * (c + h) + 16) : c = 8 :=
sorry

end NUMINAMATH_GPT_cows_in_group_l17_1777


namespace NUMINAMATH_GPT_exists_subset_with_property_l17_1795

theorem exists_subset_with_property :
  ∃ X : Set Int, ∀ n : Int, ∃ (a b : X), a + 2 * b = n ∧ ∀ (a' b' : X), (a + 2 * b = n ∧ a' + 2 * b' = n) → (a = a' ∧ b = b') :=
sorry

end NUMINAMATH_GPT_exists_subset_with_property_l17_1795


namespace NUMINAMATH_GPT_hannah_quarters_l17_1792

theorem hannah_quarters :
  ∃ n : ℕ, 40 < n ∧ n < 400 ∧
  n % 6 = 3 ∧ n % 7 = 3 ∧ n % 8 = 3 ∧ 
  (n = 171 ∨ n = 339) :=
by
  sorry

end NUMINAMATH_GPT_hannah_quarters_l17_1792


namespace NUMINAMATH_GPT_national_education_fund_expenditure_l17_1781

theorem national_education_fund_expenditure (gdp_2012 : ℝ) (h : gdp_2012 = 43.5 * 10^12) : 
  (0.04 * gdp_2012) = 1.74 * 10^13 := 
by sorry

end NUMINAMATH_GPT_national_education_fund_expenditure_l17_1781


namespace NUMINAMATH_GPT_rhombus_perimeter_l17_1739

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 52 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_l17_1739


namespace NUMINAMATH_GPT_not_all_x_heart_x_eq_0_l17_1743

def heartsuit (x y : ℝ) : ℝ := abs (x + y)

theorem not_all_x_heart_x_eq_0 :
  ¬ (∀ x : ℝ, heartsuit x x = 0) :=
by sorry

end NUMINAMATH_GPT_not_all_x_heart_x_eq_0_l17_1743


namespace NUMINAMATH_GPT_bottom_left_square_side_length_l17_1753

theorem bottom_left_square_side_length (x y : ℕ) 
  (h1 : 1 + (x - 1) = 1) 
  (h2 : 2 * x - 1 = (x - 2) + (x - 3) + y) :
  y = 4 :=
sorry

end NUMINAMATH_GPT_bottom_left_square_side_length_l17_1753


namespace NUMINAMATH_GPT_chromium_percentage_is_correct_l17_1752

noncomputable def chromium_percentage_new_alloy (chr_percent1 chr_percent2 weight1 weight2 : ℝ) : ℝ :=
  (chr_percent1 * weight1 + chr_percent2 * weight2) / (weight1 + weight2) * 100

theorem chromium_percentage_is_correct :
  chromium_percentage_new_alloy 0.10 0.06 15 35 = 7.2 :=
by
  sorry

end NUMINAMATH_GPT_chromium_percentage_is_correct_l17_1752


namespace NUMINAMATH_GPT_count_f_compositions_l17_1767

noncomputable def count_special_functions : Nat :=
  let A := Finset.range 6
  let f := (Set.univ : Set (A → A))
  sorry

theorem count_f_compositions (f : Fin 6 → Fin 6) 
  (h : ∀ x : Fin 6, (f ∘ f ∘ f) x = x) :
  count_special_functions = 81 :=
sorry

end NUMINAMATH_GPT_count_f_compositions_l17_1767


namespace NUMINAMATH_GPT_minimum_value_of_angle_l17_1766

theorem minimum_value_of_angle
  (α : ℝ)
  (h : ∃ x y : ℝ, (x, y) = (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))) :
  α = 11 * Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_angle_l17_1766


namespace NUMINAMATH_GPT_fred_allowance_is_16_l17_1764

def fred_weekly_allowance (A : ℕ) : Prop :=
  (A / 2) + 6 = 14

theorem fred_allowance_is_16 : ∃ A : ℕ, fred_weekly_allowance A ∧ A = 16 := 
by
  -- Proof can be filled here
  sorry

end NUMINAMATH_GPT_fred_allowance_is_16_l17_1764


namespace NUMINAMATH_GPT_tangent_line_eq_l17_1797

noncomputable def f (x : ℝ) := x / (2 * x - 1)

def tangentLineAtPoint (x : ℝ) : ℝ := -x + 2

theorem tangent_line_eq {x y : ℝ} (hxy : y = f 1) (f_deriv : deriv f 1 = -1) :
  y = 1 → tangentLineAtPoint x = -x + 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l17_1797


namespace NUMINAMATH_GPT_remainder_when_sum_is_divided_l17_1701

theorem remainder_when_sum_is_divided (n : ℤ) : ((8 - n) + (n + 5)) % 9 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_when_sum_is_divided_l17_1701


namespace NUMINAMATH_GPT_ellipse_sum_a_k_l17_1770

theorem ellipse_sum_a_k {a b h k : ℝ}
  (foci1 foci2 : ℝ × ℝ)
  (point_on_ellipse : ℝ × ℝ)
  (h_center : h = (foci1.1 + foci2.1) / 2)
  (k_center : k = (foci1.2 + foci2.2) / 2)
  (distance1 : ℝ := Real.sqrt ((point_on_ellipse.1 - foci1.1)^2 + (point_on_ellipse.2 - foci1.2)^2))
  (distance2 : ℝ := Real.sqrt ((point_on_ellipse.1 - foci2.1)^2 + (point_on_ellipse.2 - foci2.2)^2))
  (major_axis_length : ℝ := distance1 + distance2)
  (h_a : a = major_axis_length / 2)
  (c := Real.sqrt ((foci2.1 - foci1.1)^2 + (foci2.2 - foci1.2)^2) / 2)
  (h_b : b^2 = a^2 - c^2) :
  a + k = (7 + Real.sqrt 13) / 2 := 
by
  sorry

end NUMINAMATH_GPT_ellipse_sum_a_k_l17_1770


namespace NUMINAMATH_GPT_cups_of_sugar_l17_1757

theorem cups_of_sugar (flour_total flour_added sugar : ℕ) (h₁ : flour_total = 10) (h₂ : flour_added = 7) (h₃ : flour_total - flour_added = sugar + 1) :
  sugar = 2 :=
by
  sorry

end NUMINAMATH_GPT_cups_of_sugar_l17_1757


namespace NUMINAMATH_GPT_geometric_sequence_sum_eight_l17_1748

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_eight_l17_1748


namespace NUMINAMATH_GPT_find_minimum_width_l17_1731

-- Definitions based on the problem conditions
def length_from_width (w : ℝ) : ℝ := w + 12

def minimum_fence_area (w : ℝ) : Prop := w * length_from_width w ≥ 144

-- Proof statement
theorem find_minimum_width : ∃ w : ℝ, w ≥ 6 ∧ minimum_fence_area w :=
sorry

end NUMINAMATH_GPT_find_minimum_width_l17_1731


namespace NUMINAMATH_GPT_remaining_water_after_45_days_l17_1744

def initial_water : ℝ := 500
def daily_loss : ℝ := 1.2
def days : ℝ := 45

theorem remaining_water_after_45_days :
  initial_water - daily_loss * days = 446 := by
  sorry

end NUMINAMATH_GPT_remaining_water_after_45_days_l17_1744


namespace NUMINAMATH_GPT_transformation_thinking_reflected_in_solution_of_quadratic_l17_1759

theorem transformation_thinking_reflected_in_solution_of_quadratic :
  ∀ (x : ℝ), (x - 3)^2 - 5 * (x - 3) = 0 → (x = 3 ∨ x = 8) →
  transformation_thinking :=
by
  intros x h_eq h_solutions
  sorry

end NUMINAMATH_GPT_transformation_thinking_reflected_in_solution_of_quadratic_l17_1759


namespace NUMINAMATH_GPT_triangle_side_length_difference_l17_1717

theorem triangle_side_length_difference (a b c : ℕ) (hb : b = 8) (hc : c = 3)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  let min_a := 6
  let max_a := 10
  max_a - min_a = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_side_length_difference_l17_1717


namespace NUMINAMATH_GPT_moon_speed_conversion_l17_1791

theorem moon_speed_conversion
  (speed_kps : ℝ)
  (seconds_per_hour : ℝ)
  (h1 : speed_kps = 0.2)
  (h2 : seconds_per_hour = 3600) :
  speed_kps * seconds_per_hour = 720 := by
  sorry

end NUMINAMATH_GPT_moon_speed_conversion_l17_1791


namespace NUMINAMATH_GPT_sequence_properties_l17_1760

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 3 - 2^n

-- Prove the statements
theorem sequence_properties (n : ℕ) :
  (a (2 * n) = 3 - 4^n) ∧ (a 2 / a 3 = 1 / 5) :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l17_1760


namespace NUMINAMATH_GPT_find_divisor_l17_1728

theorem find_divisor (x : ℕ) (h : 180 % x = 0) (h_eq : 70 + 5 * 12 / (180 / x) = 71) : x = 3 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_divisor_l17_1728


namespace NUMINAMATH_GPT_Al_initial_portion_l17_1707

theorem Al_initial_portion (a b c : ℕ) 
  (h1 : a + b + c = 1200) 
  (h2 : a - 150 + 2 * b + 3 * c = 1800) 
  (h3 : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a = 550 :=
by {
  sorry
}

end NUMINAMATH_GPT_Al_initial_portion_l17_1707


namespace NUMINAMATH_GPT_average_income_BC_l17_1785

theorem average_income_BC {A_income B_income C_income : ℝ}
  (hAB : (A_income + B_income) / 2 = 4050)
  (hAC : (A_income + C_income) / 2 = 4200)
  (hA : A_income = 3000) :
  (B_income + C_income) / 2 = 5250 :=
by sorry

end NUMINAMATH_GPT_average_income_BC_l17_1785


namespace NUMINAMATH_GPT_range_of_theta_div_4_l17_1724

noncomputable def theta_third_quadrant (k : ℤ) (θ : ℝ) : Prop :=
  (2 * k * Real.pi + Real.pi < θ) ∧ (θ < 2 * k * Real.pi + 3 * Real.pi / 2)

noncomputable def sin_lt_cos (θ : ℝ) : Prop :=
  Real.sin (θ / 4) < Real.cos (θ / 4)

theorem range_of_theta_div_4 (k : ℤ) (θ : ℝ) :
  theta_third_quadrant k θ →
  sin_lt_cos θ →
  (2 * k * Real.pi + 5 * Real.pi / 4 < θ / 4 ∧ θ / 4 < 2 * k * Real.pi + 11 * Real.pi / 8) ∨
  (2 * k * Real.pi + 7 * Real.pi / 4 < θ / 4 ∧ θ / 4 < 2 * k * Real.pi + 15 * Real.pi / 8) := 
  by
    sorry

end NUMINAMATH_GPT_range_of_theta_div_4_l17_1724


namespace NUMINAMATH_GPT_highest_power_of_5_dividing_S_l17_1758

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def f (n : ℕ) : ℤ :=
  if sum_of_digits n % 2 = 0 then n ^ 100 else -n ^ 100

def S : ℤ :=
  (Finset.range (10 ^ 100)).sum (λ n => f n)

theorem highest_power_of_5_dividing_S :
  ∃ m : ℕ, 5 ^ m ∣ S ∧ ∀ k : ℕ, 5 ^ (k + 1) ∣ S → k < 24 :=
by
  sorry

end NUMINAMATH_GPT_highest_power_of_5_dividing_S_l17_1758


namespace NUMINAMATH_GPT_unique_pairs_of_socks_l17_1722

-- Defining the problem conditions
def pairs_socks : Nat := 3

-- The main proof statement
theorem unique_pairs_of_socks : ∃ (n : Nat), n = 3 ∧ 
  (∀ (p q : Fin 6), (p / 2 ≠ q / 2) → p ≠ q) →
  (n = (pairs_socks * (pairs_socks - 1)) / 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_pairs_of_socks_l17_1722


namespace NUMINAMATH_GPT_sum_m_n_eq_zero_l17_1779

theorem sum_m_n_eq_zero (m n p : ℝ) (h1 : m * n + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 := 
  sorry

end NUMINAMATH_GPT_sum_m_n_eq_zero_l17_1779


namespace NUMINAMATH_GPT_mod_equivalence_l17_1711

theorem mod_equivalence (n : ℤ) (hn₁ : 0 ≤ n) (hn₂ : n < 23) (hmod : -250 % 23 = n % 23) : n = 3 := by
  sorry

end NUMINAMATH_GPT_mod_equivalence_l17_1711


namespace NUMINAMATH_GPT_dried_grapes_weight_l17_1740

def fresh_grapes_weight : ℝ := 30
def fresh_grapes_water_percentage : ℝ := 0.60
def dried_grapes_water_percentage : ℝ := 0.20

theorem dried_grapes_weight :
  let non_water_content := fresh_grapes_weight * (1 - fresh_grapes_water_percentage)
  let dried_grapes := non_water_content / (1 - dried_grapes_water_percentage)
  dried_grapes = 15 :=
by
  let non_water_content := fresh_grapes_weight * (1 - fresh_grapes_water_percentage)
  let dried_grapes := non_water_content / (1 - dried_grapes_water_percentage)
  show dried_grapes = 15
  sorry

end NUMINAMATH_GPT_dried_grapes_weight_l17_1740


namespace NUMINAMATH_GPT_find_certain_number_l17_1705

theorem find_certain_number (h1 : 213 * 16 = 3408) (x : ℝ) (h2 : x * 2.13 = 0.03408) : x = 0.016 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l17_1705


namespace NUMINAMATH_GPT_area_percent_less_l17_1709

theorem area_percent_less 
  (r1 r2 : ℝ)
  (h : r1 / r2 = 3 / 10) 
  : 1 - (π * (r1:ℝ)^2 / (π * (r2:ℝ)^2)) = 0.91 := 
by 
  sorry

end NUMINAMATH_GPT_area_percent_less_l17_1709


namespace NUMINAMATH_GPT_smallest_cube_volume_l17_1793

noncomputable def sculpture_height : ℝ := 15
noncomputable def sculpture_base_radius : ℝ := 8
noncomputable def cube_side_length : ℝ := 16

theorem smallest_cube_volume :
  ∀ (h r s : ℝ), 
    h = sculpture_height ∧
    r = sculpture_base_radius ∧
    s = cube_side_length →
    s ^ 3 = 4096 :=
by
  intros h r s 
  intro h_def
  sorry

end NUMINAMATH_GPT_smallest_cube_volume_l17_1793


namespace NUMINAMATH_GPT_solve_for_a_l17_1761
-- Additional imports might be necessary depending on specifics of the proof

theorem solve_for_a (a x y : ℝ) (h1 : ax - y = 3) (h2 : x = 1) (h3 : y = 2) : a = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l17_1761


namespace NUMINAMATH_GPT_intersection_A_and_B_l17_1710

-- Define the sets based on the conditions
def setA : Set ℤ := {x : ℤ | x^2 - 2 * x - 8 ≤ 0}
def setB : Set ℤ := {x : ℤ | 1 < Real.log x / Real.log 2}

-- State the theorem (Note: The logarithmic condition should translate the values to integers)
theorem intersection_A_and_B : setA ∩ setB = {3, 4} :=
sorry

end NUMINAMATH_GPT_intersection_A_and_B_l17_1710


namespace NUMINAMATH_GPT_max_ab_squared_l17_1708

theorem max_ab_squared (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 2) :
  ∃ x, 0 < x ∧ x < 2 ∧ a = 2 - x ∧ ab^2 = x * (2 - x)^2 :=
sorry

end NUMINAMATH_GPT_max_ab_squared_l17_1708


namespace NUMINAMATH_GPT_sum_zero_of_distinct_and_ratio_l17_1703

noncomputable def distinct (a b c d : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

theorem sum_zero_of_distinct_and_ratio (x y u v : ℝ) 
  (h_distinct : distinct x y u v)
  (h_ratio : (x + u) / (x + v) = (y + v) / (y + u)) : 
  x + y + u + v = 0 := 
sorry

end NUMINAMATH_GPT_sum_zero_of_distinct_and_ratio_l17_1703


namespace NUMINAMATH_GPT_rectangular_prism_volume_l17_1714

theorem rectangular_prism_volume :
  ∀ (l w h : ℕ), 
  l = 2 * w → 
  w = 2 * h → 
  4 * (l + w + h) = 56 → 
  l * w * h = 64 := 
by
  intros l w h h_l_eq_2w h_w_eq_2h h_edge_len_eq_56
  sorry -- proof not provided

end NUMINAMATH_GPT_rectangular_prism_volume_l17_1714


namespace NUMINAMATH_GPT_proposition_2_proposition_3_l17_1773

theorem proposition_2 (a b : ℝ) (h: a > |b|) : a^2 > b^2 := 
sorry

theorem proposition_3 (a b : ℝ) (h: a > b) : a^3 > b^3 := 
sorry

end NUMINAMATH_GPT_proposition_2_proposition_3_l17_1773


namespace NUMINAMATH_GPT_find_time_for_products_maximize_salary_l17_1702

-- Assume the conditions and definitions based on the given problem
variables (x y a : ℝ)

-- Condition 1: Time to produce 6 type A and 4 type B products is 170 minutes
axiom cond1 : 6 * x + 4 * y = 170

-- Condition 2: Time to produce 10 type A and 10 type B products is 350 minutes
axiom cond2 : 10 * x + 10 * y = 350


-- Question 1: Validating the time to produce one type A product and one type B product
theorem find_time_for_products : 
  x = 15 ∧ y = 20 := by
  sorry

-- Variables for calculation of Zhang's daily salary
variables (m : ℕ) (base_salary : ℝ := 100) (daily_work: ℝ := 480)

-- Conditions for the piece-rate wages
variables (a_condition: 2 < a ∧ a < 3) 
variables (num_products: m + (28 - m) = 28)

-- Question 2: Finding optimal production plan to maximize daily salary
theorem maximize_salary :
  (2 < a ∧ a < 2.5) → m = 16 ∨ 
  (a = 2.5) → true ∨
  (2.5 < a ∧ a < 3) → m = 28 := by
  sorry

end NUMINAMATH_GPT_find_time_for_products_maximize_salary_l17_1702


namespace NUMINAMATH_GPT_equilateral_triangle_surface_area_correct_l17_1734

noncomputable def equilateral_triangle_surface_area : ℝ :=
  let side_length := 2
  let A := (0, 0, 0)
  let B := (side_length, 0, 0)
  let C := (side_length / 2, (side_length * (Real.sqrt 3)) / 2, 0)
  let D := (side_length / 2, (side_length * (Real.sqrt 3)) / 6, 0)
  let folded_angle := 90
  let diagonal_length := Real.sqrt (1 + 1 + 3)
  let radius := diagonal_length / 2
  let surface_area := 4 * Real.pi * radius^2
  5 * Real.pi

theorem equilateral_triangle_surface_area_correct :
  equilateral_triangle_surface_area = 5 * Real.pi :=
by
  unfold equilateral_triangle_surface_area
  sorry -- proof omitted

end NUMINAMATH_GPT_equilateral_triangle_surface_area_correct_l17_1734


namespace NUMINAMATH_GPT_compute_fraction_l17_1736

theorem compute_fraction :
  ((5 * 4) + 6) / 10 = 2.6 :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_l17_1736


namespace NUMINAMATH_GPT_well_defined_interval_l17_1788

def is_well_defined (x : ℝ) : Prop :=
  (5 - x > 0) ∧ (x ≠ 2)

theorem well_defined_interval : 
  ∀ x : ℝ, (is_well_defined x) ↔ (x < 5 ∧ x ≠ 2) :=
by 
  sorry

end NUMINAMATH_GPT_well_defined_interval_l17_1788


namespace NUMINAMATH_GPT_coco_hours_used_l17_1742

noncomputable def electricity_price : ℝ := 0.10
noncomputable def consumption_rate : ℝ := 2.4
noncomputable def total_cost : ℝ := 6.0

theorem coco_hours_used (hours_used : ℝ) : hours_used = total_cost / (consumption_rate * electricity_price) :=
by
  sorry

end NUMINAMATH_GPT_coco_hours_used_l17_1742


namespace NUMINAMATH_GPT_profit_percentage_l17_1737

/-- If the cost price is 81% of the selling price, then the profit percentage is approximately 23.46%. -/
theorem profit_percentage (SP CP: ℝ) (h : CP = 0.81 * SP) : 
  (SP - CP) / CP * 100 = 23.46 := 
sorry

end NUMINAMATH_GPT_profit_percentage_l17_1737


namespace NUMINAMATH_GPT_problem_solution_l17_1772

theorem problem_solution (n : Real) (h : 0.04 * n + 0.1 * (30 + n) = 15.2) : n = 89.09 := 
sorry

end NUMINAMATH_GPT_problem_solution_l17_1772


namespace NUMINAMATH_GPT_cos_B_and_area_of_triangle_l17_1768

theorem cos_B_and_area_of_triangle (A B C : ℝ) (a b c : ℝ)
  (h_sin_A : Real.sin A = Real.sin (2 * B))
  (h_a : a = 4) (h_b : b = 6) :
  Real.cos B = 1 / 3 ∧ ∃ (area : ℝ), area = 8 * Real.sqrt 2 :=
by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_cos_B_and_area_of_triangle_l17_1768


namespace NUMINAMATH_GPT_olivia_dad_spent_l17_1721

def cost_per_meal : ℕ := 7
def number_of_meals : ℕ := 3
def total_cost : ℕ := 21

theorem olivia_dad_spent :
  cost_per_meal * number_of_meals = total_cost :=
by
  sorry

end NUMINAMATH_GPT_olivia_dad_spent_l17_1721


namespace NUMINAMATH_GPT_vector_magnitude_sub_l17_1704

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (theta : ℝ) (h_theta : theta = Real.pi / 3)

/-- Given vectors a and b with magnitudes 2 and 3 respectively, and the angle between them is 60 degrees,
    we need to prove that the magnitude of the vector a - b is sqrt(7). -/
theorem vector_magnitude_sub : ‖a - b‖ = Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_vector_magnitude_sub_l17_1704


namespace NUMINAMATH_GPT_inequality_proof_l17_1713

theorem inequality_proof 
  (x1 x2 y1 y2 z1 z2 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : 0 < x2)
  (hxy1 : x1 * y1 > z1 ^ 2)
  (hxy2 : x2 * y2 > z2 ^ 2) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) ≤
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l17_1713


namespace NUMINAMATH_GPT_Pima_investment_value_at_week6_l17_1796

noncomputable def Pima_initial_investment : ℝ := 400
noncomputable def Pima_week1_gain : ℝ := 0.25
noncomputable def Pima_week1_addition : ℝ := 200
noncomputable def Pima_week2_gain : ℝ := 0.50
noncomputable def Pima_week2_withdrawal : ℝ := 150
noncomputable def Pima_week3_loss : ℝ := 0.10
noncomputable def Pima_week4_gain : ℝ := 0.20
noncomputable def Pima_week4_addition : ℝ := 100
noncomputable def Pima_week5_gain : ℝ := 0.05
noncomputable def Pima_week6_loss : ℝ := 0.15
noncomputable def Pima_week6_withdrawal : ℝ := 250
noncomputable def weekly_interest_rate : ℝ := 0.02

noncomputable def calculate_investment_value : ℝ :=
  let week0 := Pima_initial_investment
  let week1 := (week0 * (1 + Pima_week1_gain) * (1 + weekly_interest_rate)) + Pima_week1_addition
  let week2 := ((week1 * (1 + Pima_week2_gain) * (1 + weekly_interest_rate)) - Pima_week2_withdrawal)
  let week3 := (week2 * (1 - Pima_week3_loss) * (1 + weekly_interest_rate))
  let week4 := ((week3 * (1 + Pima_week4_gain) * (1 + weekly_interest_rate)) + Pima_week4_addition)
  let week5 := (week4 * (1 + Pima_week5_gain) * (1 + weekly_interest_rate))
  let week6 := ((week5 * (1 - Pima_week6_loss) * (1 + weekly_interest_rate)) - Pima_week6_withdrawal)
  week6

theorem Pima_investment_value_at_week6 : calculate_investment_value = 819.74 := 
  by
  sorry

end NUMINAMATH_GPT_Pima_investment_value_at_week6_l17_1796


namespace NUMINAMATH_GPT_regular_18gon_symmetries_l17_1763

theorem regular_18gon_symmetries :
  let L := 18
  let R := 20
  L + R = 38 := by
sorry

end NUMINAMATH_GPT_regular_18gon_symmetries_l17_1763


namespace NUMINAMATH_GPT_unique_solution_t_interval_l17_1798

theorem unique_solution_t_interval (x y z v t : ℝ) :
  (x + y + z + v = 0) →
  ((x * y + y * z + z * v) + t * (x * z + x * v + y * v) = 0) →
  (t > (3 - Real.sqrt 5) / 2) ∧ (t < (3 + Real.sqrt 5) / 2) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_unique_solution_t_interval_l17_1798


namespace NUMINAMATH_GPT_initial_cookies_count_l17_1754

theorem initial_cookies_count (x : ℕ) (h_ate : ℕ) (h_left : ℕ) :
  h_ate = 2 → h_left = 5 → (x - h_ate = h_left) → x = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_cookies_count_l17_1754


namespace NUMINAMATH_GPT_largest_divisible_by_two_power_l17_1712
-- Import the necessary Lean library

open scoped BigOperators

-- Prime and Multiples calculation based conditions
def primes_count : ℕ := 25
def multiples_of_four_count : ℕ := 25

-- Number of subsets of {1, 2, 3, ..., 100} with more primes than multiples of 4
def N : ℕ :=
  let pow := 2^50
  pow * (pow / 2 - (∑ k in Finset.range 26, Nat.choose 25 k ^ 2))

-- Theorem stating that the largest integer k such that 2^k divides N is 52
theorem largest_divisible_by_two_power :
  ∃ (k : ℕ), (2^k ∣ N) ∧ (∀ m : ℕ, 2^m ∣ N → m ≤ 52) :=
sorry

end NUMINAMATH_GPT_largest_divisible_by_two_power_l17_1712
