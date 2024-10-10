import Mathlib

namespace g_composition_value_l3400_340054

def g (y : ℝ) : ℝ := y^3 - 3*y + 1

theorem g_composition_value : g (g (g (-1))) = 6803 := by
  sorry

end g_composition_value_l3400_340054


namespace multiple_problem_l3400_340069

theorem multiple_problem (n m : ℝ) : n = 5 → n + m * n = 20 → m = 3 := by sorry

end multiple_problem_l3400_340069


namespace weekend_study_hours_per_day_l3400_340017

-- Define the given conditions
def weekday_study_hours_per_night : ℕ := 2
def weekday_study_nights_per_week : ℕ := 5
def weeks_until_exam : ℕ := 6
def total_study_hours : ℕ := 96

-- Define the number of days in a weekend
def days_per_weekend : ℕ := 2

-- Define the theorem
theorem weekend_study_hours_per_day :
  (total_study_hours - weekday_study_hours_per_night * weekday_study_nights_per_week * weeks_until_exam) / (days_per_weekend * weeks_until_exam) = 3 := by
  sorry

end weekend_study_hours_per_day_l3400_340017


namespace equation_solution_l3400_340071

theorem equation_solution :
  ∃ m : ℝ, (m - 5) ^ 3 = (1 / 16)⁻¹ ∧ m = 5 + 2 ^ (4 / 3) := by
  sorry

end equation_solution_l3400_340071


namespace pascals_triangle_56th_row_second_element_l3400_340055

theorem pascals_triangle_56th_row_second_element :
  let n : ℕ := 56  -- The row number (0-indexed) with 57 elements
  let k : ℕ := 1   -- The position of the second element (0-indexed)
  Nat.choose n k = 56 := by
sorry

end pascals_triangle_56th_row_second_element_l3400_340055


namespace no_two_common_tangents_l3400_340022

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Number of common tangents between two circles --/
def commonTangents (c1 c2 : Circle) : ℕ := sorry

theorem no_two_common_tangents (c1 c2 : Circle) (h : c1.radius ≠ c2.radius) :
  commonTangents c1 c2 ≠ 2 := by sorry

end no_two_common_tangents_l3400_340022


namespace intersection_of_lines_l3400_340056

/-- The intersection point of two lines -/
def intersection_point (m1 a1 m2 a2 : ℚ) : ℚ × ℚ :=
  let x := (a2 - a1) / (m1 - m2)
  let y := m1 * x + a1
  (x, y)

/-- First line: y = 3x -/
def line1 (x : ℚ) : ℚ := 3 * x

/-- Second line: y + 6 = -9x, or y = -9x - 6 -/
def line2 (x : ℚ) : ℚ := -9 * x - 6

theorem intersection_of_lines :
  intersection_point 3 0 (-9) (-6) = (-1/2, -3/2) ∧
  line1 (-1/2) = -3/2 ∧
  line2 (-1/2) = -3/2 :=
sorry

end intersection_of_lines_l3400_340056


namespace alice_notebook_savings_l3400_340060

/-- The amount Alice saves when buying notebooks during a sale -/
theorem alice_notebook_savings (number_of_notebooks : ℕ) (original_price : ℚ) (discount_rate : ℚ) :
  number_of_notebooks = 8 →
  original_price = 375/100 →
  discount_rate = 25/100 →
  (number_of_notebooks * original_price) - (number_of_notebooks * (original_price * (1 - discount_rate))) = 75/10 := by
  sorry

end alice_notebook_savings_l3400_340060


namespace min_lines_for_31_segments_l3400_340084

/-- A broken line represented by its number of segments -/
structure BrokenLine where
  segments : ℕ
  no_self_intersections : Bool
  distinct_endpoints : Bool

/-- The minimum number of straight lines formed by extending all segments of a broken line -/
def min_straight_lines (bl : BrokenLine) : ℕ :=
  (bl.segments + 1) / 2

/-- Theorem stating the minimum number of straight lines for a specific broken line -/
theorem min_lines_for_31_segments :
  ∀ (bl : BrokenLine),
    bl.segments = 31 →
    bl.no_self_intersections = true →
    bl.distinct_endpoints = true →
    min_straight_lines bl = 16 := by
  sorry

#eval min_straight_lines { segments := 31, no_self_intersections := true, distinct_endpoints := true }

end min_lines_for_31_segments_l3400_340084


namespace book_arrangement_count_l3400_340028

def num_arabic : ℕ := 2
def num_german : ℕ := 3
def num_spanish : ℕ := 4
def total_books : ℕ := num_arabic + num_german + num_spanish

def arrangement_count : ℕ := sorry

theorem book_arrangement_count :
  arrangement_count = 3456 := by sorry

end book_arrangement_count_l3400_340028


namespace normal_to_curve_l3400_340086

-- Define the curve
def curve (x y a : ℝ) : Prop := x^(2/3) + y^(2/3) = a^(2/3)

-- Define the normal equation
def normal_equation (x y a θ : ℝ) : Prop := y * Real.cos θ - x * Real.sin θ = a * Real.cos (2 * θ)

-- Theorem statement
theorem normal_to_curve (x y a θ : ℝ) :
  curve x y a →
  (∃ (p q : ℝ), curve p q a ∧ 
    -- The point (p, q) is on the curve and the normal at this point makes an angle θ with the X-axis
    (y - q) * Real.cos θ = (x - p) * Real.sin θ) →
  normal_equation x y a θ :=
by sorry

end normal_to_curve_l3400_340086


namespace franks_work_days_l3400_340033

/-- Frank's work schedule problem -/
theorem franks_work_days 
  (hours_per_day : ℕ) 
  (total_hours : ℕ) 
  (h1 : hours_per_day = 8) 
  (h2 : total_hours = 32) : 
  total_hours / hours_per_day = 4 := by
  sorry

end franks_work_days_l3400_340033


namespace bob_eats_one_more_than_george_l3400_340098

/-- Represents the number of slices in different pizza sizes and quantities purchased --/
structure PizzaOrder where
  small_slices : ℕ := 4
  large_slices : ℕ := 8
  small_count : ℕ := 3
  large_count : ℕ := 2

/-- Represents the pizza consumption of different people --/
structure PizzaConsumption where
  george : ℕ := 3
  bill : ℕ := 3
  fred : ℕ := 3
  mark : ℕ := 3
  leftover : ℕ := 10

/-- Theorem stating that Bob eats one more slice than George --/
theorem bob_eats_one_more_than_george (order : PizzaOrder) (consumption : PizzaConsumption) : 
  ∃ (bob : ℕ) (susie : ℕ), 
    susie = bob / 2 ∧ 
    bob = consumption.george + 1 ∧
    order.small_slices * order.small_count + order.large_slices * order.large_count = 
      consumption.george + bob + susie + consumption.bill + consumption.fred + consumption.mark + consumption.leftover :=
by
  sorry

end bob_eats_one_more_than_george_l3400_340098


namespace max_value_sum_l3400_340040

theorem max_value_sum (a b c d e : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
  (sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 504) : 
  ∃ (N a_N b_N c_N d_N e_N : ℝ),
    (∀ (x y z w v : ℝ), x * z + 3 * y * z + 4 * z * w + 8 * z * v ≤ N) ∧
    (N = a_N * c_N + 3 * b_N * c_N + 4 * c_N * d_N + 8 * c_N * e_N) ∧
    (a_N^2 + b_N^2 + c_N^2 + d_N^2 + e_N^2 = 504) ∧
    (N + a_N + b_N + c_N + d_N + e_N = 32 + 1512 * Real.sqrt 10 + 6 * Real.sqrt 7) :=
by sorry

end max_value_sum_l3400_340040


namespace inequality_minimum_a_l3400_340083

theorem inequality_minimum_a (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → x * y ≤ a * x^2 + 2 * y^2) → 
  a ≥ -1/2 := by
  sorry

end inequality_minimum_a_l3400_340083


namespace digit_105_of_7_19th_l3400_340008

/-- The decimal representation of 7/19 has a repeating cycle of length 18 -/
def decimal_cycle_length : ℕ := 18

/-- The repeating decimal representation of 7/19 -/
def decimal_rep : List ℕ := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]

/-- The 105th digit after the decimal point in the decimal representation of 7/19 is 7 -/
theorem digit_105_of_7_19th : decimal_rep[(105 - 1) % decimal_cycle_length] = 7 := by
  sorry

end digit_105_of_7_19th_l3400_340008


namespace round_trip_average_speed_l3400_340091

/-- Calculates the average speed of a round trip given the outbound speed and the fact that the return journey takes twice as long. -/
theorem round_trip_average_speed (outbound_speed : ℝ) 
  (h : outbound_speed = 45) : 
  let return_time := 2 * (1 / outbound_speed)
  let total_time := 1 / outbound_speed + return_time
  let total_distance := 2
  (total_distance / total_time) = 30 := by sorry

end round_trip_average_speed_l3400_340091


namespace complex_number_quadrant_l3400_340000

theorem complex_number_quadrant : 
  let z : ℂ := (2 + Complex.I) * Complex.I
  (z.re < 0 ∧ z.im > 0) := by sorry

end complex_number_quadrant_l3400_340000


namespace g_composition_two_roots_l3400_340073

def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + d

theorem g_composition_two_roots (d : ℝ) : 
  (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ ∀ x, g d (g d x) = 0 ↔ x = r₁ ∨ x = r₂) ↔ d = 8 := by
  sorry

end g_composition_two_roots_l3400_340073


namespace probability_not_spade_first_draw_l3400_340012

theorem probability_not_spade_first_draw (total_cards : ℕ) (spade_cards : ℕ) 
  (h1 : total_cards = 52) (h2 : spade_cards = 13) :
  (total_cards - spade_cards : ℚ) / total_cards = 3 / 4 := by
  sorry

end probability_not_spade_first_draw_l3400_340012


namespace box_volume_problem_l3400_340016

theorem box_volume_problem :
  ∃! (x : ℕ), x > 3 ∧ (x + 3) * (x - 3) * (x^2 + 9) < 500 := by sorry

end box_volume_problem_l3400_340016


namespace equal_intercept_line_theorem_tangent_circle_theorem_l3400_340085

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define point P
def point_P : ℝ × ℝ := (1, 2)

-- Define the line with equal intercepts passing through P
def equal_intercept_line (x y : ℝ) : Prop := x + y = 3

-- Define the circle
def tangent_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Theorem for the line with equal intercepts
theorem equal_intercept_line_theorem :
  ∃ (a : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, equal_intercept_line x y ↔ x / a + y / a = 1) ∧
  equal_intercept_line point_P.1 point_P.2 :=
sorry

-- Theorem for the tangent circle
theorem tangent_circle_theorem :
  ∃ A B : ℝ × ℝ,
  (line_l A.1 A.2 ∧ A.2 = 0) ∧
  (line_l B.1 B.2 ∧ B.1 = 0) ∧
  (∀ x y : ℝ, tangent_circle x y →
    (x = 0 ∨ y = 0 ∨ line_l x y)) :=
sorry

end equal_intercept_line_theorem_tangent_circle_theorem_l3400_340085


namespace difference_of_squares_262_258_l3400_340087

theorem difference_of_squares_262_258 : 262^2 - 258^2 = 2080 := by
  sorry

end difference_of_squares_262_258_l3400_340087


namespace initial_men_count_l3400_340058

/-- Given a group of men where:
  * The average age increases by 2 years when two women replace two men
  * The replaced men are 10 and 12 years old
  * The average age of the women is 21 years
  Prove that the initial number of men in the group is 10 -/
theorem initial_men_count (M : ℕ) (A : ℚ) : 
  (M * A - 22 + 42 = M * (A + 2)) → M = 10 := by
  sorry

end initial_men_count_l3400_340058


namespace min_value_cube_sum_squared_l3400_340021

theorem min_value_cube_sum_squared (a b c : ℝ) :
  (∃ (α β γ : ℤ), α ∈ ({-1, 1} : Set ℤ) ∧ β ∈ ({-1, 1} : Set ℤ) ∧ γ ∈ ({-1, 1} : Set ℤ) ∧ a * α + b * β + c * γ = 0) →
  ((a^3 + b^3 + c^3) / (a * b * c))^2 ≥ 9 :=
by sorry

end min_value_cube_sum_squared_l3400_340021


namespace quadratic_rewrite_l3400_340074

theorem quadratic_rewrite (x : ℝ) : 
  ∃ (a b c : ℤ), 16 * x^2 - 40 * x + 18 = (a * x + b)^2 + c ∧ a * b = -20 := by
  sorry

end quadratic_rewrite_l3400_340074


namespace find_y_l3400_340023

theorem find_y (x : ℕ) (y : ℕ) (h1 : 2^x - 2^(x-2) = 3 * 2^y) (h2 : x = 12) : y = 10 := by
  sorry

end find_y_l3400_340023


namespace function_symmetry_l3400_340094

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the conditions
def passes_through_point (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y

def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Theorem statement
theorem function_symmetry 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : passes_through_point (log a) 2 (-1)) 
  (f : ℝ → ℝ) 
  (h4 : symmetric_wrt_y_eq_x f (log a)) : 
  f = fun x ↦ (1/2)^x := by sorry

end function_symmetry_l3400_340094


namespace fan_weight_l3400_340081

/-- Given a box with fans, calculate the weight of a single fan. -/
theorem fan_weight (total_weight : ℝ) (num_fans : ℕ) (empty_box_weight : ℝ) 
  (h1 : total_weight = 11.14)
  (h2 : num_fans = 14)
  (h3 : empty_box_weight = 0.5) :
  (total_weight - empty_box_weight) / num_fans = 0.76 := by
  sorry

#check fan_weight

end fan_weight_l3400_340081


namespace carousel_attendance_l3400_340010

/-- The number of children attending a carousel, given:
  * 4 clowns also attend
  * The candy seller initially had 700 candies
  * Each clown and child receives 20 candies
  * The candy seller has 20 candies left after selling
-/
def num_children : ℕ := 30

theorem carousel_attendance : num_children = 30 := by
  sorry

end carousel_attendance_l3400_340010


namespace andy_profit_per_cake_l3400_340041

/-- Andy's cake business model -/
structure CakeBusiness where
  ingredient_cost_two_cakes : ℕ
  packaging_cost_per_cake : ℕ
  selling_price_per_cake : ℕ

/-- Calculate the profit per cake -/
def profit_per_cake (b : CakeBusiness) : ℕ :=
  b.selling_price_per_cake - (b.ingredient_cost_two_cakes / 2 + b.packaging_cost_per_cake)

/-- Theorem: Andy's profit per cake is $8 -/
theorem andy_profit_per_cake :
  ∃ (b : CakeBusiness),
    b.ingredient_cost_two_cakes = 12 ∧
    b.packaging_cost_per_cake = 1 ∧
    b.selling_price_per_cake = 15 ∧
    profit_per_cake b = 8 := by
  sorry

end andy_profit_per_cake_l3400_340041


namespace coprime_power_minus_one_divisible_l3400_340053

theorem coprime_power_minus_one_divisible
  (N₁ N₂ : ℕ+) (k : ℕ) 
  (h_coprime : Nat.Coprime N₁ N₂)
  (h_k : k = Nat.totient N₂) :
  N₂ ∣ (N₁^k - 1) :=
by sorry

end coprime_power_minus_one_divisible_l3400_340053


namespace regular_100gon_rectangle_two_colors_l3400_340019

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A coloring of the vertices of a polygon -/
def Coloring (n : ℕ) (k : ℕ) := Fin n → Fin k

/-- Four vertices form a rectangle in a regular polygon -/
def IsRectangle (p : RegularPolygon 100) (v1 v2 v3 v4 : Fin 100) : Prop :=
  sorry

/-- The number of distinct colors used for given vertices -/
def NumColors (c : Coloring 100 10) (vs : List (Fin 100)) : ℕ :=
  sorry

theorem regular_100gon_rectangle_two_colors :
  ∀ (p : RegularPolygon 100) (c : Coloring 100 10),
  ∃ (v1 v2 v3 v4 : Fin 100),
    IsRectangle p v1 v2 v3 v4 ∧ NumColors c [v1, v2, v3, v4] ≤ 2 :=
sorry

end regular_100gon_rectangle_two_colors_l3400_340019


namespace number_problem_l3400_340045

theorem number_problem (x : ℝ) : 42 + 3 * x - 10 = 65 → x = 11 := by
  sorry

end number_problem_l3400_340045


namespace ferris_wheel_cost_is_five_l3400_340070

/-- The cost of a Ferris wheel ride that satisfies the given conditions -/
def ferris_wheel_cost : ℕ → Prop := fun cost =>
  ∃ (total_children ferris_children : ℕ),
    total_children = 5 ∧
    ferris_children = 3 ∧
    total_children * (2 * 8 + 3) + ferris_children * cost = 110

/-- The cost of the Ferris wheel ride is $5 per child -/
theorem ferris_wheel_cost_is_five : ferris_wheel_cost 5 := by
  sorry

end ferris_wheel_cost_is_five_l3400_340070


namespace shortest_segment_length_l3400_340031

/-- Represents the paper strip and folding operations -/
structure PaperStrip where
  length : Real
  red_dot_position : Real
  yellow_dot_position : Real

/-- Calculates the position of the yellow dot after the first fold -/
def calculate_yellow_dot_position (strip : PaperStrip) : Real :=
  strip.length - strip.red_dot_position

/-- Calculates the length of the segment between red and yellow dots -/
def calculate_middle_segment (strip : PaperStrip) : Real :=
  strip.length - 2 * strip.yellow_dot_position

/-- Calculates the length of the shortest segment after all folds and cuts -/
def calculate_shortest_segment (strip : PaperStrip) : Real :=
  strip.red_dot_position - 2 * (strip.red_dot_position - strip.yellow_dot_position)

/-- Theorem stating that the shortest segment is 0.146 meters long -/
theorem shortest_segment_length :
  let initial_strip : PaperStrip := {
    length := 1,
    red_dot_position := 0.618,
    yellow_dot_position := calculate_yellow_dot_position { length := 1, red_dot_position := 0.618, yellow_dot_position := 0 }
  }
  calculate_shortest_segment initial_strip = 0.146 := by
  sorry

end shortest_segment_length_l3400_340031


namespace pure_imaginary_real_part_zero_l3400_340063

/-- A complex number z is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_real_part_zero (a : ℝ) :
  isPureImaginary (Complex.mk a 1) → a = 0 := by
  sorry

end pure_imaginary_real_part_zero_l3400_340063


namespace seashells_left_l3400_340062

def initial_seashells : ℕ := 62
def seashells_given : ℕ := 49

theorem seashells_left : initial_seashells - seashells_given = 13 := by
  sorry

end seashells_left_l3400_340062


namespace fraction_to_decimal_l3400_340059

theorem fraction_to_decimal : (2 : ℚ) / 25 = 0.08 := by sorry

end fraction_to_decimal_l3400_340059


namespace sequence_negative_term_l3400_340030

theorem sequence_negative_term
  (k : ℝ) (h_k : 0 < k ∧ k < 1)
  (a : ℕ → ℝ)
  (h_a : ∀ n : ℕ, n ≥ 1 → a (n + 1) ≤ (1 + k / n) * a n - 1) :
  ∃ t : ℕ, a t < 0 := by
sorry

end sequence_negative_term_l3400_340030


namespace max_value_of_f_l3400_340034

/-- The quadratic function f(x) = -2x^2 + 4x + 3 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 3

/-- The maximum value of f(x) for x ∈ ℝ is 5 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end max_value_of_f_l3400_340034


namespace no_prime_root_solution_l3400_340061

/-- A quadratic equation x^2 - 67x + k = 0 with prime roots -/
def has_prime_roots (k : ℤ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ 
  (p : ℤ) + q = 67 ∧ (p : ℤ) * q = k

/-- There are no integer values of k for which the equation x^2 - 67x + k = 0 has two prime roots -/
theorem no_prime_root_solution : ¬∃ k : ℤ, has_prime_roots k := by
  sorry

end no_prime_root_solution_l3400_340061


namespace opposite_of_2023_l3400_340020

theorem opposite_of_2023 : 
  ∀ x : ℤ, (x + 2023 = 0) ↔ (x = -2023) := by
  sorry

end opposite_of_2023_l3400_340020


namespace sqrt_mixed_number_simplification_l3400_340050

theorem sqrt_mixed_number_simplification :
  Real.sqrt (7 + 9 / 16) = 11 / 4 := by sorry

end sqrt_mixed_number_simplification_l3400_340050


namespace consecutive_composite_numbers_l3400_340006

theorem consecutive_composite_numbers (k k' : ℕ) :
  (∀ i ∈ Finset.range 7, ¬ Nat.Prime (210 * k + 1 + i + 1)) ∧
  (∀ i ∈ Finset.range 15, ¬ Nat.Prime (30030 * k' + 1 + i + 1)) := by
  sorry

end consecutive_composite_numbers_l3400_340006


namespace tourist_walking_speed_l3400_340057

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hMinutesValid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Represents the problem scenario -/
structure TouristProblem where
  scheduledArrival : Time
  actualArrival : Time
  busSpeed : ℝ
  earlyArrival : ℕ

/-- Calculates the tourists' walking speed -/
noncomputable def touristSpeed (problem : TouristProblem) : ℝ :=
  let walkingTime := timeDifference problem.actualArrival problem.scheduledArrival - problem.earlyArrival
  let distance := problem.busSpeed * (problem.earlyArrival / 2) / 60
  distance / (walkingTime / 60)

/-- The main theorem to prove -/
theorem tourist_walking_speed (problem : TouristProblem) 
  (hScheduledArrival : problem.scheduledArrival = ⟨5, 0, by norm_num⟩)
  (hActualArrival : problem.actualArrival = ⟨3, 10, by norm_num⟩)
  (hBusSpeed : problem.busSpeed = 60)
  (hEarlyArrival : problem.earlyArrival = 20) :
  touristSpeed problem = 6 := by
  sorry

end tourist_walking_speed_l3400_340057


namespace tangent_r_values_l3400_340011

-- Define the curves C and C1
def C (x y : ℝ) : Prop := (x - 0)^2 + (y - 2)^2 = 4

def C1 (x y r : ℝ) : Prop := ∃ α, x = 3 + r * Real.cos α ∧ y = -2 + r * Real.sin α

-- Define the tangency condition
def are_tangent (r : ℝ) : Prop :=
  ∃ x y, C x y ∧ C1 x y r

-- Theorem statement
theorem tangent_r_values :
  ∀ r : ℝ, are_tangent r ↔ r = 3 ∨ r = -3 ∨ r = 7 ∨ r = -7 :=
sorry

end tangent_r_values_l3400_340011


namespace squares_below_line_l3400_340080

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of integer points strictly below a line in the first quadrant --/
def countPointsBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line from the problem --/
def problemLine : Line :=
  { a := 12, b := 180, c := 2160 }

/-- The theorem statement --/
theorem squares_below_line :
  countPointsBelowLine problemLine = 984 := by
  sorry

end squares_below_line_l3400_340080


namespace min_sum_absolute_values_l3400_340044

theorem min_sum_absolute_values :
  ∀ x : ℝ, |x + 3| + |x + 5| + |x + 6| ≥ 5 ∧ ∃ x : ℝ, |x + 3| + |x + 5| + |x + 6| = 5 := by
  sorry

end min_sum_absolute_values_l3400_340044


namespace grumpy_not_orange_l3400_340077

structure Lizard where
  orange : Prop
  grumpy : Prop
  can_swim : Prop
  can_jump : Prop

def Cathys_lizards : Set Lizard := sorry

theorem grumpy_not_orange :
  ∀ (total : ℕ) (orange_count : ℕ) (grumpy_count : ℕ),
  total = 15 →
  orange_count = 6 →
  grumpy_count = 7 →
  (∀ l : Lizard, l ∈ Cathys_lizards → l.grumpy → l.can_swim) →
  (∀ l : Lizard, l ∈ Cathys_lizards → l.orange → ¬l.can_jump) →
  (∀ l : Lizard, l ∈ Cathys_lizards → ¬l.can_jump → ¬l.can_swim) →
  (∀ l : Lizard, l ∈ Cathys_lizards → ¬(l.grumpy ∧ l.orange)) :=
by sorry

end grumpy_not_orange_l3400_340077


namespace cans_collected_l3400_340046

/-- Proves that the number of cans collected is 144 given the recycling rates and total money received -/
theorem cans_collected (can_rate : ℚ) (newspaper_rate : ℚ) (newspaper_collected : ℚ) (total_money : ℚ) :
  can_rate = 1/24 →
  newspaper_rate = 3/10 →
  newspaper_collected = 20 →
  total_money = 12 →
  ∃ (cans : ℚ), cans * can_rate + newspaper_collected * newspaper_rate = total_money ∧ cans = 144 := by
  sorry

end cans_collected_l3400_340046


namespace geometric_arithmetic_sequence_common_difference_l3400_340099

/-- Given a geometric sequence {a_n} where a_1+1, a_3+4, a_5+7 form an arithmetic sequence,
    the common difference of this arithmetic sequence is 3. -/
theorem geometric_arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d) :
  ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ d = 3 :=
by sorry

end geometric_arithmetic_sequence_common_difference_l3400_340099


namespace quadratic_solution_difference_l3400_340052

theorem quadratic_solution_difference : 
  let f : ℝ → ℝ := λ x => x^2 + 5*x - 4 - (x + 66)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 2 * Real.sqrt 74 :=
by sorry

end quadratic_solution_difference_l3400_340052


namespace proportion_solution_l3400_340097

theorem proportion_solution (x : ℝ) : (0.60 / x = 6 / 4) → x = 0.4 := by
  sorry

end proportion_solution_l3400_340097


namespace limit_example_l3400_340049

/-- The limit of (5x^2 - 4x - 1)/(x - 1) as x approaches 1 is 6 -/
theorem limit_example : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 1| → |x - 1| < δ → 
    |(5*x^2 - 4*x - 1)/(x - 1) - 6| < ε := by
  sorry

end limit_example_l3400_340049


namespace man_speed_against_current_and_headwind_l3400_340038

/-- The speed of a man rowing in a river with current and headwind -/
def man_speed (downstream_speed current_speed headwind_reduction : ℝ) : ℝ :=
  downstream_speed - current_speed - current_speed - headwind_reduction

/-- Theorem stating the man's speed against current and headwind -/
theorem man_speed_against_current_and_headwind 
  (downstream_speed : ℝ) 
  (current_speed : ℝ) 
  (headwind_reduction : ℝ) 
  (h1 : downstream_speed = 22) 
  (h2 : current_speed = 4.5) 
  (h3 : headwind_reduction = 1.5) : 
  man_speed downstream_speed current_speed headwind_reduction = 11.5 := by
  sorry

#eval man_speed 22 4.5 1.5

end man_speed_against_current_and_headwind_l3400_340038


namespace max_lateral_surface_area_cylinder_in_sphere_l3400_340042

/-- The maximum lateral surface area of a cylinder inscribed in a sphere -/
theorem max_lateral_surface_area_cylinder_in_sphere :
  ∀ (R r l : ℝ),
  R > 0 →
  r > 0 →
  l > 0 →
  (4 / 3) * Real.pi * R^3 = (32 / 3) * Real.pi →
  r^2 + (l / 2)^2 = R^2 →
  2 * Real.pi * r * l ≤ 8 * Real.pi :=
by sorry

end max_lateral_surface_area_cylinder_in_sphere_l3400_340042


namespace square_of_negative_l3400_340066

theorem square_of_negative (a : ℝ) : (-a)^2 = a^2 := by
  sorry

end square_of_negative_l3400_340066


namespace johns_max_correct_answers_l3400_340068

/-- Represents an exam with a given number of questions and scoring system. -/
structure Exam where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  total_score : ℤ

/-- Checks if an exam result is valid according to the exam rules. -/
def is_valid_result (result : ExamResult) : Prop :=
  result.correct + result.incorrect + result.unanswered = result.exam.total_questions ∧
  result.total_score = result.correct * result.exam.correct_score + result.incorrect * result.exam.incorrect_score

/-- Theorem: The maximum number of correctly answered questions for John's exam is 12. -/
theorem johns_max_correct_answers (john_exam : Exam) (john_result : ExamResult) :
  john_exam.total_questions = 20 ∧
  john_exam.correct_score = 5 ∧
  john_exam.incorrect_score = -2 ∧
  john_result.exam = john_exam ∧
  john_result.total_score = 48 ∧
  is_valid_result john_result →
  ∀ (other_result : ExamResult),
    is_valid_result other_result ∧
    other_result.exam = john_exam ∧
    other_result.total_score = 48 →
    other_result.correct ≤ 12 :=
by sorry

end johns_max_correct_answers_l3400_340068


namespace product_of_complex_numbers_l3400_340076

/-- Represents a complex number in polar form -/
structure PolarComplex where
  r : ℝ
  θ : ℝ
  h_r_pos : r > 0
  h_θ_range : 0 ≤ θ ∧ θ < 2 * Real.pi

/-- Multiplies two complex numbers in polar form -/
def polar_multiply (z₁ z₂ : PolarComplex) : PolarComplex :=
  { r := z₁.r * z₂.r,
    θ := z₁.θ + z₂.θ,
    h_r_pos := by sorry,
    h_θ_range := by sorry }

theorem product_of_complex_numbers :
  let z₁ : PolarComplex := ⟨5, 30 * Real.pi / 180, by sorry, by sorry⟩
  let z₂ : PolarComplex := ⟨4, 140 * Real.pi / 180, by sorry, by sorry⟩
  let result := polar_multiply z₁ z₂
  result.r = 20 ∧ result.θ = 170 * Real.pi / 180 := by sorry

end product_of_complex_numbers_l3400_340076


namespace midpoint_coordinate_sum_l3400_340005

/-- Given that M(-3, 2) is the midpoint of AB and A(-8, 5) is one endpoint,
    prove that the sum of coordinates of B is 1. -/
theorem midpoint_coordinate_sum :
  let M : ℝ × ℝ := (-3, 2)
  let A : ℝ × ℝ := (-8, 5)
  ∀ B : ℝ × ℝ,
  (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) →
  B.1 + B.2 = 1 :=
by sorry

end midpoint_coordinate_sum_l3400_340005


namespace fourth_sphere_radius_l3400_340096

/-- Given four spheres on a table, where each sphere touches the table and the other three spheres,
    and three of the spheres have radius R, the radius of the fourth sphere is 4R/3. -/
theorem fourth_sphere_radius (R : ℝ) (R_pos : R > 0) : ∃ r : ℝ,
  (∀ (i j : Fin 4), i ≠ j → ∃ (x y z : ℝ),
    (i.val < 3 → norm (⟨x, y, z⟩ : ℝ × ℝ × ℝ) = R) ∧
    (i.val = 3 → norm (⟨x, y, z⟩ : ℝ × ℝ × ℝ) = r) ∧
    (j.val < 3 → ∃ (x' y' z' : ℝ), norm (⟨x - x', y - y', z - z'⟩ : ℝ × ℝ × ℝ) = R + R) ∧
    (j.val = 3 → ∃ (x' y' z' : ℝ), norm (⟨x - x', y - y', z - z'⟩ : ℝ × ℝ × ℝ) = R + r) ∧
    z ≥ R ∧ z' ≥ R) ∧
  r = 4 * R / 3 :=
by
  sorry

end fourth_sphere_radius_l3400_340096


namespace rtl_grouping_equivalence_l3400_340013

/-- Right-to-left grouping evaluation function -/
noncomputable def rtlEval (a b c d e : ℝ) : ℝ := a / (b * c - (d + e))

/-- Standard algebraic notation representation -/
noncomputable def standardNotation (a b c d e : ℝ) : ℝ := a / (b * c - d - e)

/-- Theorem stating the equivalence of right-to-left grouping and standard notation -/
theorem rtl_grouping_equivalence (a b c d e : ℝ) :
  rtlEval a b c d e = standardNotation a b c d e :=
sorry

end rtl_grouping_equivalence_l3400_340013


namespace max_value_of_a_l3400_340067

theorem max_value_of_a : 
  (∃ (a : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → Real.sqrt x - Real.sqrt (4 - x) ≥ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → Real.sqrt x - Real.sqrt (4 - x) ≥ b) → b ≤ -2) :=
sorry

end max_value_of_a_l3400_340067


namespace cubic_repeated_root_condition_l3400_340072

/-- A cubic polynomial with a repeated root -/
def has_repeated_root (b : ℝ) : Prop :=
  ∃ x : ℝ, (b * x^3 + 15 * x^2 + 9 * x + 2 = 0) ∧
           (3 * b * x^2 + 30 * x + 9 = 0)

/-- Theorem stating that if a nonzero b makes the cubic have a repeated root, then b = 100 -/
theorem cubic_repeated_root_condition (b : ℝ) (hb : b ≠ 0) :
  has_repeated_root b → b = 100 := by
  sorry

end cubic_repeated_root_condition_l3400_340072


namespace next_simultaneous_ring_l3400_340075

def library_interval : ℕ := 18
def fire_station_interval : ℕ := 24
def hospital_interval : ℕ := 30

theorem next_simultaneous_ring : 
  Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval = 360 := by
  sorry

end next_simultaneous_ring_l3400_340075


namespace largest_quantity_l3400_340002

theorem largest_quantity : 
  let A := (2010 : ℚ) / 2009 + 2010 / 2011
  let B := (2010 : ℚ) / 2011 + 2012 / 2011
  let C := (2011 : ℚ) / 2010 + 2011 / 2012
  A > B ∧ A > C := by sorry

end largest_quantity_l3400_340002


namespace janet_monday_wednesday_hours_l3400_340014

/-- Janet's weekly gym schedule -/
structure GymSchedule where
  total_hours : ℝ
  monday_hours : ℝ
  tuesday_hours : ℝ
  wednesday_hours : ℝ
  friday_hours : ℝ

/-- Janet's gym schedule satisfies the given conditions -/
def janet_schedule (s : GymSchedule) : Prop :=
  s.total_hours = 5 ∧
  s.tuesday_hours = s.friday_hours ∧
  s.friday_hours = 1 ∧
  s.monday_hours = s.wednesday_hours

/-- Theorem: Janet spends 1.5 hours at the gym on Monday and Wednesday each -/
theorem janet_monday_wednesday_hours (s : GymSchedule) 
  (h : janet_schedule s) : s.monday_hours = 1.5 ∧ s.wednesday_hours = 1.5 := by
  sorry


end janet_monday_wednesday_hours_l3400_340014


namespace smallest_positive_integer_satisfying_condition_l3400_340035

theorem smallest_positive_integer_satisfying_condition : 
  ∃ (x : ℕ+), (x : ℝ) + 1000 > 1000 * x ∧ 
  ∀ (y : ℕ+), ((y : ℝ) + 1000 > 1000 * y → x ≤ y) :=
sorry

end smallest_positive_integer_satisfying_condition_l3400_340035


namespace money_distribution_l3400_340092

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 350)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 350) :
  c = 200 := by
sorry

end money_distribution_l3400_340092


namespace group_morphism_identity_or_inverse_l3400_340003

variable {G : Type*} [Group G]

theorem group_morphism_identity_or_inverse
  (no_order_4 : ∀ g : G, g^4 = 1 → g = 1)
  (f : G → G)
  (f_hom : ∀ x y : G, f (x * y) = f x * f y)
  (f_property : ∀ x : G, f x = x ∨ f x = x⁻¹) :
  (∀ x : G, f x = x) ∨ (∀ x : G, f x = x⁻¹) := by
  sorry

end group_morphism_identity_or_inverse_l3400_340003


namespace distance_is_600_l3400_340082

/-- The distance between two points A and B, given specific train travel conditions. -/
def distance_between_points : ℝ :=
  let forward_speed : ℝ := 200
  let return_speed : ℝ := 100
  let time_difference : ℝ := 3
  600

/-- Theorem stating that the distance between points A and B is 600 km under given conditions. -/
theorem distance_is_600 (forward_speed return_speed time_difference : ℝ)
  (h1 : forward_speed = 200)
  (h2 : return_speed = 100)
  (h3 : time_difference = 3)
  : distance_between_points = 600 :=
by sorry

end distance_is_600_l3400_340082


namespace fraction_to_decimal_l3400_340001

theorem fraction_to_decimal : (21 : ℚ) / 40 = 0.525 := by
  sorry

end fraction_to_decimal_l3400_340001


namespace fraction_is_integer_l3400_340064

theorem fraction_is_integer (b t : ℤ) (h : b ≠ 1) :
  ∃ k : ℤ, (t^5 - 5*b + 4) / (b^2 - 2*b + 1) = k :=
sorry

end fraction_is_integer_l3400_340064


namespace quadratic_equation_roots_l3400_340048

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   x^2 - 2*(m+1)*x + m^2 + 2 = 0 ∧ 
   y^2 - 2*(m+1)*y + m^2 + 2 = 0 ∧ 
   (1/x + 1/y = 1)) → 
  m = 2 := by
sorry

end quadratic_equation_roots_l3400_340048


namespace sphere_surface_volume_relation_l3400_340088

theorem sphere_surface_volume_relation :
  ∀ (r R : ℝ),
  r > 0 →
  R > 0 →
  (4 * Real.pi * R^2) = (4 * (4 * Real.pi * r^2)) →
  ((4/3) * Real.pi * R^3) = (8 * ((4/3) * Real.pi * r^3)) :=
by sorry

end sphere_surface_volume_relation_l3400_340088


namespace sandbag_weight_l3400_340043

/-- Calculates the weight of a partially filled sandbag with a heavier filling material -/
theorem sandbag_weight (bag_capacity : ℝ) (fill_percentage : ℝ) (material_weight_increase : ℝ) : 
  bag_capacity > 0 → 
  fill_percentage > 0 → 
  fill_percentage ≤ 1 → 
  material_weight_increase ≥ 0 →
  let sand_weight := bag_capacity * fill_percentage
  let material_weight := sand_weight * (1 + material_weight_increase)
  bag_capacity + material_weight = 530 :=
by
  sorry

#check sandbag_weight 250 0.8 0.4

end sandbag_weight_l3400_340043


namespace degree_to_radian_300_l3400_340004

theorem degree_to_radian_300 : 
  (300 : ℝ) * (π / 180) = (5 * π) / 3 := by sorry

end degree_to_radian_300_l3400_340004


namespace ron_pick_frequency_l3400_340093

/-- Represents a book club with a given number of members -/
structure BookClub where
  members : ℕ

/-- Calculates how many times a member gets to pick a book in a year -/
def pickFrequency (club : BookClub) (weeksInYear : ℕ) : ℕ :=
  weeksInYear / club.members

theorem ron_pick_frequency :
  let couples := 3
  let singlePeople := 5
  let ronAndWife := 2
  let weeksInYear := 52
  let club := BookClub.mk (couples * 2 + singlePeople + ronAndWife)
  pickFrequency club weeksInYear = 4 := by
  sorry

end ron_pick_frequency_l3400_340093


namespace intersection_implies_sum_l3400_340027

def M (p : ℝ) := {x : ℝ | x^2 - p*x + 6 = 0}
def N (q : ℝ) := {x : ℝ | x^2 + 6*x - q = 0}

theorem intersection_implies_sum (p q : ℝ) :
  M p ∩ N q = {2} → p + q = 21 := by
  sorry

end intersection_implies_sum_l3400_340027


namespace circles_tangency_l3400_340095

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O2 (x y a : ℝ) : Prop := (x + 4)^2 + (y - a)^2 = 25

-- Define the condition of having exactly one common point
def have_one_common_point (a : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, circle_O1 p.1 p.2 ∧ circle_O2 p.1 p.2 a

-- State the theorem
theorem circles_tangency (a : ℝ) :
  have_one_common_point a → a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 ∨ a = 0 :=
by
  sorry

end circles_tangency_l3400_340095


namespace absolute_value_inequality_solution_set_l3400_340025

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 - x| ≥ 1} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by
  sorry

end absolute_value_inequality_solution_set_l3400_340025


namespace triangle_problem_l3400_340078

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b^2 = a^2 + c^2 - Real.sqrt 3 * a * c →
  B = π/6 ∧
  Real.sqrt 3 / 2 < Real.cos A + Real.sin C ∧ 
  Real.cos A + Real.sin C < 3/2 := by
sorry

end triangle_problem_l3400_340078


namespace speed_in_still_water_l3400_340018

/-- The speed of a man in still water given his upstream and downstream speeds -/
theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) :
  upstream_speed = 20 →
  downstream_speed = 80 →
  (upstream_speed + downstream_speed) / 2 = 50 := by
  sorry

#check speed_in_still_water

end speed_in_still_water_l3400_340018


namespace total_items_eq_137256_l3400_340032

/-- The number of old women going to Rome -/
def num_women : ℕ := 7

/-- The number of mules each woman has -/
def mules_per_woman : ℕ := 7

/-- The number of bags each mule carries -/
def bags_per_mule : ℕ := 7

/-- The number of loaves each bag contains -/
def loaves_per_bag : ℕ := 7

/-- The number of knives each loaf contains -/
def knives_per_loaf : ℕ := 7

/-- The number of sheaths each knife is in -/
def sheaths_per_knife : ℕ := 7

/-- The total number of items -/
def total_items : ℕ := 
  num_women +
  (num_women * mules_per_woman) +
  (num_women * mules_per_woman * bags_per_mule) +
  (num_women * mules_per_woman * bags_per_mule * loaves_per_bag) +
  (num_women * mules_per_woman * bags_per_mule * loaves_per_bag * knives_per_loaf) +
  (num_women * mules_per_woman * bags_per_mule * loaves_per_bag * knives_per_loaf * sheaths_per_knife)

theorem total_items_eq_137256 : total_items = 137256 := by
  sorry

end total_items_eq_137256_l3400_340032


namespace nathan_blanket_warmth_l3400_340079

theorem nathan_blanket_warmth (total_blankets : ℕ) (warmth_per_blanket : ℕ) (fraction_used : ℚ) : 
  total_blankets = 14 → 
  warmth_per_blanket = 3 → 
  fraction_used = 1/2 →
  (↑total_blankets * fraction_used : ℚ).floor * warmth_per_blanket = 21 := by
sorry

end nathan_blanket_warmth_l3400_340079


namespace price_decrease_percentage_l3400_340047

theorem price_decrease_percentage (initial_price : ℝ) (h : initial_price > 0) : 
  let increased_price := initial_price * (1 + 0.25)
  let decrease_percentage := (increased_price - initial_price) / increased_price * 100
  decrease_percentage = 20 := by
sorry

end price_decrease_percentage_l3400_340047


namespace f_derivative_at_zero_l3400_340065

def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem f_derivative_at_zero : 
  deriv f 0 = -120 := by sorry

end f_derivative_at_zero_l3400_340065


namespace derivative_y_l3400_340090

noncomputable def y (x : ℝ) : ℝ :=
  (1/2) * Real.tanh x + (1/(4*Real.sqrt 2)) * Real.log ((1 + Real.sqrt 2 * Real.tanh x) / (1 - Real.sqrt 2 * Real.tanh x))

theorem derivative_y (x : ℝ) :
  deriv y x = 1 / (Real.cosh x ^ 2 * (1 - Real.sinh x ^ 2)) :=
by sorry

end derivative_y_l3400_340090


namespace polynomial_equality_l3400_340029

theorem polynomial_equality (a b c d : ℝ) : 
  (∀ x : ℝ, x^4 + 4*x^3 + 3*x^2 + 2*x + 1 = 
    (x+1)^4 + a*(x+1)^3 + b*(x+1)^2 + c*(x+1) + d) → 
  (a = 0 ∧ b = -3 ∧ c = 4 ∧ d = -1) := by
sorry

end polynomial_equality_l3400_340029


namespace circle_through_points_l3400_340037

/-- A circle passing through three points -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Check if a point lies on the circle -/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + c.D * x + c.E * y + c.F = 0

/-- The specific circle we're interested in -/
def our_circle : Circle := { D := -4, E := -6, F := 0 }

theorem circle_through_points : 
  (our_circle.contains 0 0) ∧ 
  (our_circle.contains 4 0) ∧ 
  (our_circle.contains (-1) 1) :=
by sorry

#check circle_through_points

end circle_through_points_l3400_340037


namespace cosine_increasing_interval_l3400_340009

theorem cosine_increasing_interval (a : Real) : 
  (∀ x₁ x₂ : Real, -π ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a → Real.cos x₁ < Real.cos x₂) → 
  a ∈ Set.Ioc (-π) 0 := by
sorry

end cosine_increasing_interval_l3400_340009


namespace banana_box_cost_l3400_340007

/-- Calculates the total cost of bananas after discount -/
def totalCostAfterDiscount (
  bunches8 : ℕ)  -- Number of bunches with 8 bananas
  (price8 : ℚ)   -- Price of each bunch with 8 bananas
  (bunches7 : ℕ)  -- Number of bunches with 7 bananas
  (price7 : ℚ)   -- Price of each bunch with 7 bananas
  (discount : ℚ)  -- Discount as a decimal
  : ℚ :=
  let totalCost := bunches8 * price8 + bunches7 * price7
  totalCost * (1 - discount)

/-- Proves that the total cost after discount for the given conditions is $23.40 -/
theorem banana_box_cost :
  totalCostAfterDiscount 6 2.5 5 2.2 0.1 = 23.4 := by
  sorry

end banana_box_cost_l3400_340007


namespace gifts_sent_calculation_l3400_340039

/-- The number of gifts sent to the orphanage given the initial number of gifts and the number of gifts left -/
def gifts_sent_to_orphanage (initial_gifts : ℕ) (gifts_left : ℕ) : ℕ :=
  initial_gifts - gifts_left

/-- Theorem stating that for the given scenario, 66 gifts were sent to the orphanage -/
theorem gifts_sent_calculation :
  gifts_sent_to_orphanage 77 11 = 66 := by
  sorry

end gifts_sent_calculation_l3400_340039


namespace total_bathing_suits_l3400_340036

theorem total_bathing_suits (men_suits women_suits : ℕ) 
  (h1 : men_suits = 14797) 
  (h2 : women_suits = 4969) : 
  men_suits + women_suits = 19766 := by
  sorry

end total_bathing_suits_l3400_340036


namespace ashok_subjects_l3400_340026

theorem ashok_subjects (total_average : ℝ) (five_subjects_average : ℝ) (sixth_subject_mark : ℝ) 
  (h1 : total_average = 70)
  (h2 : five_subjects_average = 74)
  (h3 : sixth_subject_mark = 50) :
  ∃ (n : ℕ), n = 6 ∧ n * total_average = 5 * five_subjects_average + sixth_subject_mark :=
by
  sorry

end ashok_subjects_l3400_340026


namespace jungkook_balls_count_l3400_340089

/-- The number of boxes Jungkook has -/
def num_boxes : ℕ := 3

/-- The number of balls in each box -/
def balls_per_box : ℕ := 2

/-- The total number of balls Jungkook has -/
def total_balls : ℕ := num_boxes * balls_per_box

theorem jungkook_balls_count : total_balls = 6 := by
  sorry

end jungkook_balls_count_l3400_340089


namespace min_intercept_line_l3400_340024

/-- A line that passes through a point and intersects the positive halves of the coordinate axes -/
structure InterceptLine where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : 1 / a + 9 / b = 1

/-- The sum of intercepts of an InterceptLine -/
def sum_of_intercepts (l : InterceptLine) : ℝ := l.a + l.b

/-- The equation of the line with minimum sum of intercepts -/
def min_intercept_line_eq (x y : ℝ) : Prop := 3 * x + y - 12 = 0

theorem min_intercept_line :
  ∃ (l : InterceptLine), ∀ (l' : InterceptLine), 
    sum_of_intercepts l ≤ sum_of_intercepts l' ∧
    min_intercept_line_eq l.a l.b := by sorry

end min_intercept_line_l3400_340024


namespace short_sleeve_students_l3400_340015

/-- Proves the number of students wearing short sleeves in a class with given conditions -/
theorem short_sleeve_students (total : ℕ) (difference : ℕ) (short : ℕ) (long : ℕ) : 
  total = 36 →
  long - short = difference →
  difference = 24 →
  short + long = total →
  short = 6 := by sorry

end short_sleeve_students_l3400_340015


namespace complex_roots_of_unity_real_sixth_power_l3400_340051

theorem complex_roots_of_unity_real_sixth_power :
  ∃! (S : Finset ℂ), 
    (∀ z ∈ S, z^24 = 1 ∧ (∃ r : ℝ, z^6 = r)) ∧ 
    Finset.card S = 12 := by
  sorry

end complex_roots_of_unity_real_sixth_power_l3400_340051
