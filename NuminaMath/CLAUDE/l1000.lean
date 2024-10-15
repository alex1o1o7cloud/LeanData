import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_sides_triangle_l1000_100036

theorem min_sum_sides_triangle (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  ((a + b)^2 - c^2 = 4) →
  (C = Real.pi / 3) →
  (c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) →
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = a + b ∧ x * y = 4 / 3) →
  (a + b ≥ 4 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_sides_triangle_l1000_100036


namespace NUMINAMATH_CALUDE_a_8_equals_14_l1000_100085

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℕ := n^2 - n

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℕ := S n - S (n-1)

theorem a_8_equals_14 : a 8 = 14 := by sorry

end NUMINAMATH_CALUDE_a_8_equals_14_l1000_100085


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1000_100067

-- Define the slopes of two lines
def slope1 (a : ℝ) := -a
def slope2 : ℝ := 3

-- Define the perpendicular condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_a_value :
  ∀ a : ℝ, perpendicular (slope1 a) slope2 → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1000_100067


namespace NUMINAMATH_CALUDE_second_part_speed_l1000_100083

/-- Proves that given a trip of 70 kilometers, where the first 35 kilometers are traveled at 48 km/h
    and the average speed of the entire trip is 32 km/h, the speed of the second part of the trip is 24 km/h. -/
theorem second_part_speed (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (average_speed : ℝ)
    (h1 : total_distance = 70)
    (h2 : first_part_distance = 35)
    (h3 : first_part_speed = 48)
    (h4 : average_speed = 32) :
    let second_part_distance := total_distance - first_part_distance
    let total_time := total_distance / average_speed
    let first_part_time := first_part_distance / first_part_speed
    let second_part_time := total_time - first_part_time
    let second_part_speed := second_part_distance / second_part_time
    second_part_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_second_part_speed_l1000_100083


namespace NUMINAMATH_CALUDE_birth_interval_is_3_7_l1000_100075

/-- Represents the ages of 5 children -/
structure ChildrenAges where
  ages : Fin 5 → ℕ
  sum_65 : ages 0 + ages 1 + ages 2 + ages 3 + ages 4 = 65
  youngest_7 : ages 0 = 7

/-- The interval between births, assuming equal spacing -/
def birthInterval (c : ChildrenAges) : ℚ :=
  ((c.ages 4 - c.ages 0) : ℚ) / 4

/-- Theorem stating the birth interval is 3.7 years -/
theorem birth_interval_is_3_7 (c : ChildrenAges) : birthInterval c = 37/10 := by
  sorry

end NUMINAMATH_CALUDE_birth_interval_is_3_7_l1000_100075


namespace NUMINAMATH_CALUDE_smallest_six_digit_number_divisible_by_3_4_5_l1000_100023

def is_divisible_by (n m : Nat) : Prop := n % m = 0

theorem smallest_six_digit_number_divisible_by_3_4_5 :
  ∀ n : Nat,
    325000 ≤ n ∧ n < 326000 →
    is_divisible_by n 3 ∧ is_divisible_by n 4 ∧ is_divisible_by n 5 →
    325020 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_six_digit_number_divisible_by_3_4_5_l1000_100023


namespace NUMINAMATH_CALUDE_natalia_crates_l1000_100030

/-- Calculates the number of crates needed for a given number of items and crate capacity -/
def crates_needed (items : ℕ) (capacity : ℕ) : ℕ :=
  (items + capacity - 1) / capacity

/-- The total number of crates needed for Natalia's items -/
def total_crates : ℕ :=
  crates_needed 145 12 + crates_needed 271 8 + crates_needed 419 10 + crates_needed 209 14

theorem natalia_crates :
  total_crates = 104 := by
  sorry

end NUMINAMATH_CALUDE_natalia_crates_l1000_100030


namespace NUMINAMATH_CALUDE_sonnys_cookies_l1000_100011

/-- Given an initial number of cookie boxes and the number of boxes given to brother, sister, and cousin,
    calculate the number of boxes left for Sonny. -/
def cookies_left (initial : ℕ) (to_brother : ℕ) (to_sister : ℕ) (to_cousin : ℕ) : ℕ :=
  initial - (to_brother + to_sister + to_cousin)

/-- Theorem stating that given 45 initial boxes of cookies, after giving away 12 to brother,
    9 to sister, and 7 to cousin, the number of boxes left for Sonny is 17. -/
theorem sonnys_cookies : cookies_left 45 12 9 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sonnys_cookies_l1000_100011


namespace NUMINAMATH_CALUDE_cyclist_distance_difference_l1000_100081

/-- The difference in distance traveled between two cyclists over a given time period -/
def distance_difference (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) : ℝ :=
  (rate1 * time) - (rate2 * time)

/-- Theorem: The difference in distance traveled between two cyclists, 
    one traveling at 12 miles per hour and the other at 10 miles per hour, 
    over a period of 6 hours, is 12 miles. -/
theorem cyclist_distance_difference :
  distance_difference 12 10 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_distance_difference_l1000_100081


namespace NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l1000_100054

/-- Given a bag with 3 red balls and 2 white balls, the probability of drawing
    at least one white ball when 3 balls are randomly drawn is 9/10. -/
theorem probability_at_least_one_white_ball
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (drawn_balls : ℕ)
  (h1 : total_balls = red_balls + white_balls)
  (h2 : red_balls = 3)
  (h3 : white_balls = 2)
  (h4 : drawn_balls = 3) :
  (1 - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ)) = 9/10 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l1000_100054


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1000_100072

/-- Given a hyperbola with foci on the y-axis, real axis length of 6,
    and asymptotes y = ± 3/2 x, its standard equation is y²/9 - x²/4 = 1 -/
theorem hyperbola_standard_equation
  (foci_on_y_axis : Bool)
  (real_axis_length : ℝ)
  (asymptote_slope : ℝ)
  (h_real_axis : real_axis_length = 6)
  (h_asymptote : asymptote_slope = 3/2) :
  ∃ (a b : ℝ),
    a = real_axis_length / 2 ∧
    b = a / asymptote_slope ∧
    (λ (x y : ℝ) => y^2 / a^2 - x^2 / b^2 = 1) =
    (λ (x y : ℝ) => y^2 / 9 - x^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1000_100072


namespace NUMINAMATH_CALUDE_catch_up_time_l1000_100013

-- Define the velocities of objects A and B
def v_A (t : ℝ) : ℝ := 3 * t^2 + 1
def v_B (t : ℝ) : ℝ := 10 * t

-- Define the distances traveled by objects A and B
def d_A (t : ℝ) : ℝ := t^3 + t
def d_B (t : ℝ) : ℝ := 5 * t^2 + 5

-- Theorem: Object A catches up with object B at t = 5 seconds
theorem catch_up_time : 
  ∃ t : ℝ, t = 5 ∧ d_A t = d_B t :=
sorry

end NUMINAMATH_CALUDE_catch_up_time_l1000_100013


namespace NUMINAMATH_CALUDE_minimum_selling_price_for_profit_margin_l1000_100094

/-- The minimum selling price for a small refrigerator to maintain a 20% profit margin --/
theorem minimum_selling_price_for_profit_margin
  (average_sales : ℕ)
  (refrigerator_cost : ℝ)
  (shipping_fee : ℝ)
  (storefront_fee : ℝ)
  (repair_cost : ℝ)
  (profit_margin : ℝ)
  (h_average_sales : average_sales = 50)
  (h_refrigerator_cost : refrigerator_cost = 1200)
  (h_shipping_fee : shipping_fee = 20)
  (h_storefront_fee : storefront_fee = 10000)
  (h_repair_cost : repair_cost = 5000)
  (h_profit_margin : profit_margin = 0.2)
  : ∃ (x : ℝ), x ≥ 1824 ∧
    (average_sales : ℝ) * x - (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) ≥
    (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) * profit_margin ∧
    ∀ (y : ℝ), y < x →
      (average_sales : ℝ) * y - (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) <
      (average_sales * (refrigerator_cost + shipping_fee) + storefront_fee + repair_cost) * profit_margin :=
by sorry

end NUMINAMATH_CALUDE_minimum_selling_price_for_profit_margin_l1000_100094


namespace NUMINAMATH_CALUDE_log_sum_abs_l1000_100099

theorem log_sum_abs (x : ℝ) (θ : ℝ) (h : Real.log x / Real.log 3 = 1 + Real.sin θ) :
  |x - 1| + |x - 9| = 8 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_abs_l1000_100099


namespace NUMINAMATH_CALUDE_ball_fall_height_l1000_100064

/-- Given a ball falling from a certain height, this theorem calculates its final height from the ground. -/
theorem ball_fall_height (initial_height : ℝ) (fall_time : ℝ) (fall_speed : ℝ) :
  initial_height = 120 →
  fall_time = 20 →
  fall_speed = 4 →
  initial_height - fall_time * fall_speed = 40 := by
sorry

end NUMINAMATH_CALUDE_ball_fall_height_l1000_100064


namespace NUMINAMATH_CALUDE_plot_length_l1000_100096

/-- Given a rectangular plot with the specified conditions, prove that its length is 70 meters. -/
theorem plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 40 →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  perimeter = 2 * (length + breadth) →
  total_cost = cost_per_meter * perimeter →
  length = 70 := by sorry

end NUMINAMATH_CALUDE_plot_length_l1000_100096


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l1000_100053

def P : Set ℕ := {1, 2, 3, 4}

def Q : Set ℕ := {x : ℕ | 3 ≤ x ∧ x < 7}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l1000_100053


namespace NUMINAMATH_CALUDE_binomial_29_5_l1000_100071

theorem binomial_29_5 (h1 : Nat.choose 27 3 = 2925)
                      (h2 : Nat.choose 27 4 = 17550)
                      (h3 : Nat.choose 27 5 = 80730) :
  Nat.choose 29 5 = 118755 := by
  sorry

end NUMINAMATH_CALUDE_binomial_29_5_l1000_100071


namespace NUMINAMATH_CALUDE_dima_lives_on_seventh_floor_l1000_100069

/-- Represents the floor where Dima lives -/
def dimas_floor : ℕ := 7

/-- Represents the highest floor button Dima can reach -/
def max_reachable_floor : ℕ := 6

/-- The number of stories in the building -/
def building_stories : ℕ := 9

/-- Time (in seconds) it takes to descend from Dima's floor to the first floor -/
def descent_time : ℕ := 60

/-- Total time (in seconds) for the upward journey -/
def ascent_time : ℕ := 70

/-- Proposition stating that Dima lives on the 7th floor given the conditions -/
theorem dima_lives_on_seventh_floor :
  dimas_floor = 7 ∧
  max_reachable_floor = 6 ∧
  building_stories = 9 ∧
  descent_time = 60 ∧
  ascent_time = 70 ∧
  (5 * dimas_floor = 6 * max_reachable_floor + 1) :=
by sorry

end NUMINAMATH_CALUDE_dima_lives_on_seventh_floor_l1000_100069


namespace NUMINAMATH_CALUDE_candles_per_box_l1000_100035

/-- Given Kerry's birthday celebration scenario, prove the number of candles in a box. -/
theorem candles_per_box (kerry_age : ℕ) (num_cakes : ℕ) (total_cost : ℚ) (box_cost : ℚ) 
  (h1 : kerry_age = 8)
  (h2 : num_cakes = 3)
  (h3 : total_cost = 5)
  (h4 : box_cost = 5/2) :
  (kerry_age * num_cakes) / (total_cost / box_cost) = 12 := by
  sorry

end NUMINAMATH_CALUDE_candles_per_box_l1000_100035


namespace NUMINAMATH_CALUDE_rectangle_length_equality_l1000_100090

/-- Given two rectangles with equal area, where one rectangle measures 15 inches by 24 inches
    and the other is 45 inches wide, the length of the second rectangle is 8 inches. -/
theorem rectangle_length_equality (carol_length carol_width jordan_width : ℕ) 
    (jordan_length : ℚ) : 
  carol_length = 15 ∧ 
  carol_width = 24 ∧ 
  jordan_width = 45 ∧ 
  carol_length * carol_width = jordan_length * jordan_width →
  jordan_length = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_equality_l1000_100090


namespace NUMINAMATH_CALUDE_perimeter_of_shaded_region_l1000_100040

/-- The perimeter of the shaded region formed by three touching circles -/
theorem perimeter_of_shaded_region (circle_circumference : ℝ) :
  circle_circumference = 36 →
  (3 : ℝ) * (circle_circumference / 6) = 18 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_shaded_region_l1000_100040


namespace NUMINAMATH_CALUDE_square_root_of_256_l1000_100033

theorem square_root_of_256 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 256) : y = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_256_l1000_100033


namespace NUMINAMATH_CALUDE_mailman_problem_l1000_100019

theorem mailman_problem (total_junk_mail : ℕ) (white_mailboxes : ℕ) (red_mailboxes : ℕ) (mail_per_house : ℕ) :
  total_junk_mail = 48 →
  white_mailboxes = 2 →
  red_mailboxes = 3 →
  mail_per_house = 6 →
  white_mailboxes + red_mailboxes + (total_junk_mail - (white_mailboxes + red_mailboxes) * mail_per_house) / mail_per_house = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_mailman_problem_l1000_100019


namespace NUMINAMATH_CALUDE_crayon_count_prove_crayon_count_l1000_100056

theorem crayon_count : ℕ → Prop :=
  fun red_count =>
    let blue_count := red_count + 5
    let yellow_count := 2 * blue_count - 6
    yellow_count = 32 → red_count = 14

/-- Proof of the crayon count theorem -/
theorem prove_crayon_count : ∃ (red_count : ℕ), crayon_count red_count :=
  sorry

end NUMINAMATH_CALUDE_crayon_count_prove_crayon_count_l1000_100056


namespace NUMINAMATH_CALUDE_father_ate_chocolates_father_ate_two_chocolates_l1000_100031

theorem father_ate_chocolates (total_chocolates : ℕ) (num_sisters : ℕ) (given_to_mother : ℕ) (father_left : ℕ) : ℕ :=
  let num_people := num_sisters + 1
  let chocolates_per_person := total_chocolates / num_people
  let given_to_father := num_people * (chocolates_per_person / 2)
  let father_initial := given_to_father - given_to_mother
  father_initial - father_left

theorem father_ate_two_chocolates :
  father_ate_chocolates 20 4 3 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_father_ate_chocolates_father_ate_two_chocolates_l1000_100031


namespace NUMINAMATH_CALUDE_square_side_length_l1000_100059

/-- Given a square with diagonal length 4, prove that its side length is 2√2. -/
theorem square_side_length (d : ℝ) (h : d = 4) : 
  ∃ s : ℝ, s > 0 ∧ s * s * 2 = d * d ∧ s = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1000_100059


namespace NUMINAMATH_CALUDE_correct_calculation_l1000_100076

theorem correct_calculation (square : ℕ) (h : (325 - square) * 5 = 1500) : 
  325 - square * 5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1000_100076


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_same_plane_are_parallel_l1000_100037

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields/axioms for a plane

/-- A line in 3D space -/
structure Line where
  -- Add necessary fields/axioms for a line

/-- Two planes are perpendicular -/
def perpendicular (p1 p2 : Plane) : Prop :=
  sorry

/-- Two planes are parallel -/
def parallel (p1 p2 : Plane) : Prop :=
  sorry

/-- Two lines are parallel -/
def parallel_lines (l1 l2 : Line) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_plane (l : Line) (p : Plane) : Prop :=
  sorry

theorem planes_perpendicular_to_same_plane_are_parallel 
  (α β γ : Plane) : perpendicular α γ → perpendicular β γ → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_same_plane_are_parallel_l1000_100037


namespace NUMINAMATH_CALUDE_system_unique_solution_l1000_100093

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  Real.arccos ((4 - y) / 4) = Real.arccos ((a + x) / 2) ∧
  x^2 + y^2 + 2*x - 8*y = b

-- Define the condition for a
def a_condition (a : ℝ) : Prop :=
  a ≤ -9 ∨ a ≥ 11

-- Theorem statement
theorem system_unique_solution (a : ℝ) :
  (∀ b : ℝ, (∃! p : ℝ × ℝ, system a b p.1 p.2) ∨ (¬ ∃ p : ℝ × ℝ, system a b p.1 p.2)) ↔
  a_condition a :=
sorry

end NUMINAMATH_CALUDE_system_unique_solution_l1000_100093


namespace NUMINAMATH_CALUDE_value_of_expression_l1000_100078

theorem value_of_expression (h k : ℤ) : 
  (∃ a : ℤ, 3 * X^3 - h * X - k = a * (X - 3) * (X + 1)) →
  |3 * h - 2 * k| = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1000_100078


namespace NUMINAMATH_CALUDE_boxes_filled_in_five_minutes_l1000_100046

/-- Given a machine that fills boxes at a constant rate, 
    this theorem proves how many boxes it can fill in 5 minutes. -/
theorem boxes_filled_in_five_minutes 
  (boxes_per_hour : ℚ) 
  (h1 : boxes_per_hour = 24 / 60) : 
  boxes_per_hour * 5 = 2 := by
  sorry

#check boxes_filled_in_five_minutes

end NUMINAMATH_CALUDE_boxes_filled_in_five_minutes_l1000_100046


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_ge_5_not_p_sufficient_not_necessary_implies_a_le_2_l1000_100009

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | x ≥ 1 + a ∨ x ≤ 1 - a}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- Theorem 1: If A ∩ B = ∅, then a ≥ 5
theorem intersection_empty_implies_a_ge_5 (a : ℝ) (h : a > 0) :
  A ∩ B a = ∅ → a ≥ 5 := by sorry

-- Theorem 2: If ¬p is a sufficient but not necessary condition for q, then 0 < a ≤ 2
theorem not_p_sufficient_not_necessary_implies_a_le_2 (a : ℝ) (h : a > 0) :
  (∀ x, ¬p x → q a x) ∧ (∃ x, q a x ∧ p x) → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_ge_5_not_p_sufficient_not_necessary_implies_a_le_2_l1000_100009


namespace NUMINAMATH_CALUDE_room_width_calculation_l1000_100024

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  area : ℝ
  length : ℝ
  width : ℝ

/-- Theorem: Given a room with area 12.0 sq ft and length 1.5 ft, its width is 8.0 ft -/
theorem room_width_calculation (room : RoomDimensions) 
  (h_area : room.area = 12.0) 
  (h_length : room.length = 1.5) : 
  room.width = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1000_100024


namespace NUMINAMATH_CALUDE_triangle_side_length_l1000_100082

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the length function
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the median function
def median (P M : ℝ × ℝ) (Q R : ℝ × ℝ) : Prop :=
  length P M = 3.5 ∧ M = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2)

theorem triangle_side_length (P Q R : ℝ × ℝ) :
  Triangle P Q R →
  length P Q = 4 →
  length P R = 7 →
  median P M Q R →
  length Q R = 9 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1000_100082


namespace NUMINAMATH_CALUDE_b_value_range_l1000_100091

theorem b_value_range (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 18) : 
  ∃ (b_min b_max : ℝ), 
    (∀ b', (∃ a' c', a' + b' + c' = 3 ∧ a'^2 + b'^2 + c'^2 = 18) → b_min ≤ b' ∧ b' ≤ b_max) ∧
    b_max - b_min = 2 * Real.sqrt (45/4) :=
sorry

end NUMINAMATH_CALUDE_b_value_range_l1000_100091


namespace NUMINAMATH_CALUDE_reciprocal_F_location_l1000_100070

/-- A complex number in the first quadrant outside the unit circle -/
def F : ℂ :=
  sorry

/-- Theorem: The reciprocal of F is in the fourth quadrant inside the unit circle -/
theorem reciprocal_F_location :
  let z := F⁻¹
  0 < z.re ∧ z.im < 0 ∧ Complex.abs z < 1 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_F_location_l1000_100070


namespace NUMINAMATH_CALUDE_largest_quantity_l1000_100063

theorem largest_quantity (A B C : ℚ) : 
  A = 3003 / 3002 + 3003 / 3004 →
  B = 2 / 1 + 4 / 2 + 3005 / 3004 →
  C = 3004 / 3003 + 3004 / 3005 →
  B > A ∧ B > C := by
sorry


end NUMINAMATH_CALUDE_largest_quantity_l1000_100063


namespace NUMINAMATH_CALUDE_motorcycle_price_increase_l1000_100089

/-- Represents the price increase of a motorcycle model --/
def price_increase (original_price : ℝ) (new_price : ℝ) : ℝ :=
  new_price - original_price

/-- Theorem stating the price increase given the problem conditions --/
theorem motorcycle_price_increase :
  ∀ (original_price : ℝ) (original_quantity : ℕ) (new_quantity : ℕ) (revenue_increase : ℝ),
    original_quantity = new_quantity + 8 →
    new_quantity = 63 →
    revenue_increase = 26000 →
    original_price * original_quantity = 594000 - revenue_increase →
    (original_price + price_increase original_price (original_price + price_increase original_price original_price)) * new_quantity = 594000 →
    price_increase original_price (original_price + price_increase original_price original_price) = 1428.57 := by
  sorry


end NUMINAMATH_CALUDE_motorcycle_price_increase_l1000_100089


namespace NUMINAMATH_CALUDE_same_terminal_side_l1000_100018

theorem same_terminal_side (k : ℤ) : 
  -330 = k * 360 + 30 :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_l1000_100018


namespace NUMINAMATH_CALUDE_three_fish_thrown_back_l1000_100016

/-- Represents the number of fish caught by each family member and the total number of filets --/
structure FishingTrip where
  ben : Nat
  judy : Nat
  billy : Nat
  jim : Nat
  susie : Nat
  total_filets : Nat

/-- Calculates the number of fish thrown back given a fishing trip --/
def fish_thrown_back (trip : FishingTrip) : Nat :=
  let total_caught := trip.ben + trip.judy + trip.billy + trip.jim + trip.susie
  let kept := trip.total_filets / 2
  total_caught - kept

/-- Theorem stating that for the given fishing trip, 3 fish were thrown back --/
theorem three_fish_thrown_back : 
  let trip := FishingTrip.mk 4 1 3 2 5 24
  fish_thrown_back trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_fish_thrown_back_l1000_100016


namespace NUMINAMATH_CALUDE_liquid_distribution_l1000_100002

theorem liquid_distribution (n : ℕ) (a : ℝ) (h : n ≥ 2) :
  ∃ (x : ℕ → ℝ),
    (∀ k, 1 ≤ k ∧ k ≤ n → x k > 0) ∧
    (∀ k, 2 ≤ k ∧ k ≤ n → (1 - 1/n) * x k + (1/n) * x (k-1) = a) ∧
    ((1 - 1/n) * x 1 + (1/n) * x n = a) ∧
    (x 1 = a * n * (n-2) / (n-1)^2) ∧
    (x 2 = a * (n^2 - 2*n + 2) / (n-1)^2) ∧
    (∀ k, 3 ≤ k ∧ k ≤ n → x k = a) :=
by
  sorry

#check liquid_distribution

end NUMINAMATH_CALUDE_liquid_distribution_l1000_100002


namespace NUMINAMATH_CALUDE_min_sum_inequality_l1000_100027

theorem min_sum_inequality (a b μ : ℝ) (ha : a > 0) (hb : b > 0) (hμ : μ > 0) 
  (h : 1/a + 9/b = 1) : 
  (∀ a b, a > 0 → b > 0 → 1/a + 9/b = 1 → a + b ≥ μ) ↔ μ ∈ Set.Ioo 0 16 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_inequality_l1000_100027


namespace NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l1000_100038

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon is a polygon with 6 sides. -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720 degrees. -/
theorem sum_interior_angles_hexagon :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l1000_100038


namespace NUMINAMATH_CALUDE_natasha_dimes_l1000_100048

theorem natasha_dimes : ∃ n : ℕ, 
  10 < n ∧ n < 100 ∧ 
  n % 3 = 1 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 1 ∧ 
  n = 61 := by
  sorry

end NUMINAMATH_CALUDE_natasha_dimes_l1000_100048


namespace NUMINAMATH_CALUDE_train_length_proof_l1000_100050

/-- Given a train and a platform with equal length, if the train crosses the platform
    in 60 seconds at a speed of 30 m/s, then the length of the train is 900 meters. -/
theorem train_length_proof (train_length platform_length : ℝ) 
  (speed : ℝ) (time : ℝ) (h1 : train_length = platform_length) 
  (h2 : speed = 30) (h3 : time = 60) :
  train_length = 900 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l1000_100050


namespace NUMINAMATH_CALUDE_intersection_point_implies_m_equals_six_l1000_100005

theorem intersection_point_implies_m_equals_six (m : ℕ+) 
  (h : ∃ (x y : ℤ), 13 * x + 11 * y = 700 ∧ y = m * x - 1) : m = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_implies_m_equals_six_l1000_100005


namespace NUMINAMATH_CALUDE_number_division_problem_l1000_100092

theorem number_division_problem (x y : ℚ) 
  (h1 : (x - 5) / 7 = 7) 
  (h2 : (x - 6) / y = 6) : 
  y = 8 := by sorry

end NUMINAMATH_CALUDE_number_division_problem_l1000_100092


namespace NUMINAMATH_CALUDE_caterpillar_problem_l1000_100068

/-- The number of caterpillars remaining on a tree after population changes. -/
def caterpillarsRemaining (initial : ℕ) (hatched : ℕ) (left : ℕ) : ℕ :=
  initial + hatched - left

/-- Theorem stating that given the specific numbers in the problem, 
    the result is 10 caterpillars. -/
theorem caterpillar_problem : 
  caterpillarsRemaining 14 4 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_caterpillar_problem_l1000_100068


namespace NUMINAMATH_CALUDE_range_of_a_l1000_100021

-- Define the function f(x) = |x+2| - |x-3|
def f (x : ℝ) : ℝ := |x + 2| - |x - 3|

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∃ x, f x ≤ a) → a ∈ Set.Ici (-5) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1000_100021


namespace NUMINAMATH_CALUDE_f_extrema_a_range_l1000_100025

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x - 9

-- Theorem for part 1
theorem f_extrema :
  (∀ x ∈ Set.Icc 0 2, f x ≥ -4) ∧
  (∃ x ∈ Set.Icc 0 2, f x = -4) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≤ -3) ∧
  (∃ x ∈ Set.Icc 0 2, f x = -3) :=
sorry

-- Theorem for part 2
theorem a_range :
  ∀ a < 0,
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a^2 ≥ 1 + Real.cos x) →
  a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_f_extrema_a_range_l1000_100025


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1000_100012

open Real

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Condition 1: sin C + 2sin C cos B = sin A
  sin C + 2 * sin C * cos B = sin A →
  -- Condition 2: C ∈ (0, π/2)
  0 < C ∧ C < π / 2 →
  -- Condition 3: a = √6
  a = Real.sqrt 6 →
  -- Condition 4: cos B = 1/3
  cos B = 1 / 3 →
  -- Conclusion: b = 12/5
  b = 12 / 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1000_100012


namespace NUMINAMATH_CALUDE_two_books_different_genres_l1000_100000

/-- The number of ways to choose two books of different genres -/
def choose_two_books (mystery fantasy biography : ℕ) : ℕ :=
  mystery * fantasy + mystery * biography + fantasy * biography

/-- Theorem: Given 4 mystery novels, 3 fantasy novels, and 3 biographies,
    the number of ways to choose two books of different genres is 33 -/
theorem two_books_different_genres :
  choose_two_books 4 3 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_two_books_different_genres_l1000_100000


namespace NUMINAMATH_CALUDE_tank_emptying_time_l1000_100058

-- Define constants
def tank_volume_cubic_feet : ℝ := 20
def inlet_rate : ℝ := 5
def outlet_rate_1 : ℝ := 9
def outlet_rate_2 : ℝ := 8
def inches_per_foot : ℝ := 12

-- Theorem statement
theorem tank_emptying_time :
  let tank_volume_cubic_inches := tank_volume_cubic_feet * (inches_per_foot ^ 3)
  let net_emptying_rate := outlet_rate_1 + outlet_rate_2 - inlet_rate
  tank_volume_cubic_inches / net_emptying_rate = 2880 := by
  sorry

end NUMINAMATH_CALUDE_tank_emptying_time_l1000_100058


namespace NUMINAMATH_CALUDE_tax_percentage_proof_l1000_100015

def tax_problem (net_income : ℝ) (gross_income : ℝ) (untaxed_amount : ℝ) : Prop :=
  let taxable_income := gross_income - untaxed_amount
  let tax_rate := (gross_income - net_income) / taxable_income
  tax_rate = 0.10

theorem tax_percentage_proof :
  tax_problem 12000 13000 3000 := by
  sorry

end NUMINAMATH_CALUDE_tax_percentage_proof_l1000_100015


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_37_l1000_100066

theorem modular_inverse_of_3_mod_37 : ∃ x : ℤ, 
  (x * 3) % 37 = 1 ∧ 
  0 ≤ x ∧ 
  x ≤ 36 ∧ 
  x = 25 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_37_l1000_100066


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l1000_100003

/-- Given an investment with the following properties:
  * Initial investment: $8000
  * Interest rate: y% per annum
  * Time period: 2 years
  * Simple interest earned: $800
Prove that the compound interest earned is $820 -/
theorem compound_interest_calculation (initial_investment : ℝ) (y : ℝ) (time : ℝ) 
  (simple_interest : ℝ) (h1 : initial_investment = 8000)
  (h2 : time = 2) (h3 : simple_interest = 800) 
  (h4 : simple_interest = initial_investment * y * time / 100) :
  initial_investment * ((1 + y / 100) ^ time - 1) = 820 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l1000_100003


namespace NUMINAMATH_CALUDE_positive_number_square_plus_twice_l1000_100043

theorem positive_number_square_plus_twice : ∃ n : ℝ, n > 0 ∧ n^2 + 2*n = 210 ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_square_plus_twice_l1000_100043


namespace NUMINAMATH_CALUDE_binomial_consecutive_terms_ratio_l1000_100010

theorem binomial_consecutive_terms_ratio (n k : ℕ) : 
  (∃ (a b c : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a / b = 1 / 3 ∧ b / c = 3 / 5 ∧
    a / b = Nat.choose n k / Nat.choose n (k + 1) ∧
    b / c = Nat.choose n (k + 1) / Nat.choose n (k + 2)) →
  n + k = 19 :=
by sorry

end NUMINAMATH_CALUDE_binomial_consecutive_terms_ratio_l1000_100010


namespace NUMINAMATH_CALUDE_oates_reunion_attendees_l1000_100022

/-- The number of people attending the Oates reunion -/
def oates_attendees : ℕ := 50

/-- The number of people attending the Hall reunion -/
def hall_attendees : ℕ := 62

/-- The number of people attending both reunions -/
def both_attendees : ℕ := 12

/-- The total number of guests at the hotel -/
def total_guests : ℕ := 100

theorem oates_reunion_attendees :
  oates_attendees + hall_attendees - both_attendees = total_guests :=
by sorry

end NUMINAMATH_CALUDE_oates_reunion_attendees_l1000_100022


namespace NUMINAMATH_CALUDE_sin_half_range_l1000_100097

theorem sin_half_range (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α/2) > Real.cos (α/2)) :
  ∃ x, Real.sqrt 2 / 2 < x ∧ x < 1 ∧ x = Real.sin (α/2) :=
sorry

end NUMINAMATH_CALUDE_sin_half_range_l1000_100097


namespace NUMINAMATH_CALUDE_total_cost_trick_decks_l1000_100086

/-- Calculates the cost of decks based on tiered pricing and promotion --/
def calculate_cost (num_decks : ℕ) : ℚ :=
  let base_price := if num_decks ≤ 3 then 8
                    else if num_decks ≤ 6 then 7
                    else 6
  let full_price_decks := num_decks / 2
  let discounted_decks := num_decks - full_price_decks
  (full_price_decks * base_price + discounted_decks * base_price / 2 : ℚ)

/-- The total cost of trick decks for Victor and his friend --/
theorem total_cost_trick_decks : 
  calculate_cost 6 + calculate_cost 2 = 43.5 := by
  sorry

#eval calculate_cost 6 + calculate_cost 2

end NUMINAMATH_CALUDE_total_cost_trick_decks_l1000_100086


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1000_100060

theorem profit_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 60 → selling_price = 63 → 
  (selling_price - cost_price) / cost_price * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1000_100060


namespace NUMINAMATH_CALUDE_car_distance_proof_l1000_100042

theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) : 
  initial_time = 6 →
  speed = 80 →
  (initial_time * 3 / 2) * speed = 720 :=
by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l1000_100042


namespace NUMINAMATH_CALUDE_function_composition_equality_l1000_100028

/-- Given two functions f and g, where f is quadratic and g is linear,
    if f(g(x)) = g(f(x)) for all x, then certain conditions on their coefficients must hold. -/
theorem function_composition_equality
  (a b c d e : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∀ x, g x = d * x + e)
  (h_eq : ∀ x, f (g x) = g (f x)) :
  a * (d - 1) = 0 ∧ a * e = 0 ∧ c - e = a * e^2 :=
by sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1000_100028


namespace NUMINAMATH_CALUDE_perimeter_of_larger_square_l1000_100077

-- Define the side lengths of the small squares
def small_squares : List ℕ := [1, 1, 2, 3, 5, 8, 13]

-- Define the property that these squares form a larger square
def forms_larger_square (squares : List ℕ) : Prop := sorry

-- Define the perimeter calculation function
def calculate_perimeter (squares : List ℕ) : ℕ := sorry

-- Theorem statement
theorem perimeter_of_larger_square :
  forms_larger_square small_squares →
  calculate_perimeter small_squares = 68 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_larger_square_l1000_100077


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1000_100073

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem stating the total wet surface area of the given cistern -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 4 8 1.25 = 62 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1000_100073


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1000_100095

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + a * b = 3) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y + x * y = 3 → 2 * a + b ≤ 2 * x + y :=
by sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + a * b = 3) :
  2 * a + b = 4 * Real.sqrt 2 - 3 ↔ a = Real.sqrt 2 - 1 ∧ b = 2 * Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1000_100095


namespace NUMINAMATH_CALUDE_division_problem_l1000_100047

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
    (h1 : dividend = 2944)
    (h2 : divisor = 72)
    (h3 : remainder = 64)
    (h4 : dividend = divisor * quotient + remainder) :
  quotient = 40 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1000_100047


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_correct_l1000_100049

/-- The coefficient of x^2 in the binomial expansion of (x^2/2 - 1/√x)^6 -/
def coefficient_x_squared : ℚ :=
  let expression := (fun x => x^2/2 - 1/Real.sqrt x)^6
  -- We don't actually compute the coefficient here, just define it
  15/4

/-- Theorem stating that the coefficient of x^2 in the binomial expansion of (x^2/2 - 1/√x)^6 is 15/4 -/
theorem coefficient_x_squared_is_correct :
  coefficient_x_squared = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_correct_l1000_100049


namespace NUMINAMATH_CALUDE_train_length_l1000_100020

/-- Given a train that crosses a platform in 39 seconds and a signal pole in 18 seconds,
    where the platform is 350 meters long, prove that the length of the train is 300 meters. -/
theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
  (h1 : platform_crossing_time = 39)
  (h2 : pole_crossing_time = 18)
  (h3 : platform_length = 350) :
  let train_length := (platform_length * pole_crossing_time) / (platform_crossing_time - pole_crossing_time)
  train_length = 300 := by sorry

end NUMINAMATH_CALUDE_train_length_l1000_100020


namespace NUMINAMATH_CALUDE_complex_square_root_l1000_100006

theorem complex_square_root (z : ℂ) : z^2 = 3 - 4*I → z = 1 - 2*I ∨ z = -1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_l1000_100006


namespace NUMINAMATH_CALUDE_solution_to_system_l1000_100004

theorem solution_to_system :
  ∃ (x y : ℝ), x + y = (6 - x) + (6 - y) ∧ x - y = (x - 2) + (y - 2) ∧ x = 2 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l1000_100004


namespace NUMINAMATH_CALUDE_power_function_through_point_l1000_100044

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x ^ α) →
  f 2 = Real.sqrt 2 →
  α = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1000_100044


namespace NUMINAMATH_CALUDE_bulls_and_heat_games_l1000_100084

/-- Given that the Chicago Bulls won 70 games and the Miami Heat won 5 more games than the Bulls,
    prove that the total number of games won by both teams together is 145. -/
theorem bulls_and_heat_games (bulls_games : ℕ) (heat_games : ℕ) : 
  bulls_games = 70 → 
  heat_games = bulls_games + 5 → 
  bulls_games + heat_games = 145 := by
sorry

end NUMINAMATH_CALUDE_bulls_and_heat_games_l1000_100084


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l1000_100039

theorem product_of_cubic_fractions : 
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 4) * (f 5) * (f 6) * (f 7) * (f 8) = 73 / 312 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l1000_100039


namespace NUMINAMATH_CALUDE_complement_of_A_in_I_l1000_100029

def I : Set ℕ := {1,2,3,4,5,6,7}
def A : Set ℕ := {2,4,6,7}

theorem complement_of_A_in_I :
  I \ A = {1,3,5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_I_l1000_100029


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l1000_100055

theorem divisibility_by_seven (a b : ℕ) (h : 7 ∣ (a + b)) : 7 ∣ (101 * a + 10 * b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l1000_100055


namespace NUMINAMATH_CALUDE_company_employees_count_l1000_100080

theorem company_employees_count :
  let total_employees : ℝ := 140
  let prefer_x : ℝ := 0.6
  let prefer_y : ℝ := 0.4
  let max_satisfied : ℝ := 140
  prefer_x + prefer_y = 1 →
  prefer_x * total_employees + prefer_y * total_employees = max_satisfied →
  total_employees = 140 :=
by sorry

end NUMINAMATH_CALUDE_company_employees_count_l1000_100080


namespace NUMINAMATH_CALUDE_airplane_travel_time_l1000_100001

/-- Proves that an airplane traveling 3600 km against the wind in 5 hours,
    with a still air speed of 810 km/h, takes 4 hours to travel the same distance with the wind. -/
theorem airplane_travel_time
  (distance : ℝ)
  (time_against : ℝ)
  (speed_still : ℝ)
  (h_distance : distance = 3600)
  (h_time_against : time_against = 5)
  (h_speed_still : speed_still = 810)
  : ∃ (wind_speed : ℝ),
    (distance / (speed_still - wind_speed) = time_against) ∧
    (distance / (speed_still + wind_speed) = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_airplane_travel_time_l1000_100001


namespace NUMINAMATH_CALUDE_five_touching_circles_exist_l1000_100065

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_positive : radius > 0

/-- Two circles touch if the distance between their centers is equal to the sum or difference of their radii --/
def circles_touch (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2 ∨
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius - c2.radius)^2

/-- Theorem: There exists a configuration of five circles such that any two of them touch each other --/
theorem five_touching_circles_exist : ∃ (c1 c2 c3 c4 c5 : Circle),
  circles_touch c1 c2 ∧ circles_touch c1 c3 ∧ circles_touch c1 c4 ∧ circles_touch c1 c5 ∧
  circles_touch c2 c3 ∧ circles_touch c2 c4 ∧ circles_touch c2 c5 ∧
  circles_touch c3 c4 ∧ circles_touch c3 c5 ∧
  circles_touch c4 c5 :=
sorry

end NUMINAMATH_CALUDE_five_touching_circles_exist_l1000_100065


namespace NUMINAMATH_CALUDE_eve_distance_difference_l1000_100062

/-- Eve's running and walking distances problem -/
theorem eve_distance_difference :
  let run_distance : ℝ := 0.7
  let walk_distance : ℝ := 0.6
  run_distance - walk_distance = 0.1 := by sorry

end NUMINAMATH_CALUDE_eve_distance_difference_l1000_100062


namespace NUMINAMATH_CALUDE_parking_probability_theorem_l1000_100008

/-- Represents the parking fee structure and probabilities for a business district parking lot. -/
structure ParkingLot where
  base_fee : ℕ := 6  -- Base fee for first hour
  hourly_fee : ℕ := 8  -- Fee for each additional hour
  max_hours : ℕ := 4  -- Maximum parking duration
  prob_A_1to2 : ℚ := 1/3  -- Probability A parks between 1-2 hours
  prob_A_over14 : ℚ := 5/12  -- Probability A pays over 14 yuan

/-- Calculates the probability of various parking scenarios. -/
def parking_probabilities (lot : ParkingLot) : ℚ × ℚ :=
  let prob_A_6yuan := 1 - (lot.prob_A_1to2 + lot.prob_A_over14)
  let prob_total_36yuan := 1/4  -- Given equal probability for each time interval
  (prob_A_6yuan, prob_total_36yuan)

/-- Theorem stating the probabilities of specific parking scenarios. -/
theorem parking_probability_theorem (lot : ParkingLot) :
  parking_probabilities lot = (1/4, 1/4) := by sorry

/-- Verifies that the calculated probabilities match the expected values. -/
example (lot : ParkingLot) : 
  parking_probabilities lot = (1/4, 1/4) := by sorry

end NUMINAMATH_CALUDE_parking_probability_theorem_l1000_100008


namespace NUMINAMATH_CALUDE_ice_cream_choices_l1000_100074

/-- The number of ways to choose n items from k types with repetition -/
def choose_with_repetition (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to choose 5 scoops from 14 flavors with repetition -/
theorem ice_cream_choices : choose_with_repetition 5 14 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_choices_l1000_100074


namespace NUMINAMATH_CALUDE_beaver_count_l1000_100017

theorem beaver_count (initial_beavers additional_beaver : ℝ) 
  (h1 : initial_beavers = 2.0) 
  (h2 : additional_beaver = 1.0) : 
  initial_beavers + additional_beaver = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_beaver_count_l1000_100017


namespace NUMINAMATH_CALUDE_system_solution_l1000_100057

/-- Given a system of equations, prove that x and y have specific values. -/
theorem system_solution (a x y : ℝ) (h1 : Real.log (x^2 + y^2) / Real.log (Real.sqrt 10) = 2 * Real.log (2*a) / Real.log 10 + 2 * Real.log (x^2 - y^2) / Real.log 100) (h2 : x * y = a^2) :
  (x = a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = a * Real.sqrt (Real.sqrt 2 - 1)) ∨
  (x = -a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = -a * Real.sqrt (Real.sqrt 2 - 1)) ∨
  (x = a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = -a * Real.sqrt (Real.sqrt 2 - 1)) ∨
  (x = -a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = a * Real.sqrt (Real.sqrt 2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1000_100057


namespace NUMINAMATH_CALUDE_N_not_cube_l1000_100087

/-- Represents a number of the form 10...050...01 with 100 zeros in each group -/
def N : ℕ := 10^201 + 5 * 10^100 + 1

/-- Theorem stating that N is not a perfect cube -/
theorem N_not_cube : ¬ ∃ (m : ℕ), m^3 = N := by
  sorry

end NUMINAMATH_CALUDE_N_not_cube_l1000_100087


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l1000_100051

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l1000_100051


namespace NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l1000_100098

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : Nat
  cols : Nat
  horizontal_lines : Nat
  vertical_lines : Nat

/-- Calculates the number of squares on a checkerboard -/
def count_squares (board : Checkerboard) : Nat :=
  sorry

/-- Calculates the number of rectangles on a checkerboard -/
def count_rectangles (board : Checkerboard) : Nat :=
  sorry

/-- The main theorem stating the ratio of squares to rectangles on a 6x6 checkerboard -/
theorem squares_to_rectangles_ratio (board : Checkerboard) :
  board.rows = 6 ∧ board.cols = 6 ∧ board.horizontal_lines = 5 ∧ board.vertical_lines = 5 →
  (count_squares board : Rat) / (count_rectangles board : Rat) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l1000_100098


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l1000_100007

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of green balls in the bag -/
def num_green_balls : ℕ := 5

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red_balls + num_green_balls

/-- The probability of drawing a red ball from the bag -/
def prob_red_ball : ℚ := num_red_balls / total_balls

theorem probability_of_red_ball :
  prob_red_ball = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l1000_100007


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1000_100032

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (1 / a + 4 / b) ≥ 9 / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1000_100032


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l1000_100088

theorem unique_solution_trigonometric_equation :
  ∃! (n : ℕ+), (Real.sin (π / (3 * n.val)) + Real.cos (π / (3 * n.val)) = Real.sqrt (3 * n.val) / 3) ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l1000_100088


namespace NUMINAMATH_CALUDE_flour_needed_for_cake_l1000_100061

/-- Given a recipe that requires a certain amount of flour and some flour already added,
    calculate the remaining amount of flour needed. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- The problem statement -/
theorem flour_needed_for_cake : remaining_flour 7 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_for_cake_l1000_100061


namespace NUMINAMATH_CALUDE_tan_plus_4sin_20_deg_equals_sqrt3_l1000_100079

theorem tan_plus_4sin_20_deg_equals_sqrt3 :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_4sin_20_deg_equals_sqrt3_l1000_100079


namespace NUMINAMATH_CALUDE_square_difference_l1000_100041

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1000_100041


namespace NUMINAMATH_CALUDE_coefficient_of_b_fourth_l1000_100045

theorem coefficient_of_b_fourth (b : ℝ) : 
  (∃ b : ℝ, b^4 - 41*b^2 + 100 = 0) ∧ 
  (∃ b₁ b₂ : ℝ, b₁ ≥ b₂ ∧ b₂ ≥ 0 ∧ b₁ + b₂ = 4.5 ∧ 
    b₁^4 - 41*b₁^2 + 100 = 0 ∧ b₂^4 - 41*b₂^2 + 100 = 0) →
  (∃ a : ℝ, ∀ b : ℝ, a*b^4 - 41*b^2 + 100 = 0 → a = 1) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_b_fourth_l1000_100045


namespace NUMINAMATH_CALUDE_prime_natural_equation_solutions_l1000_100052

theorem prime_natural_equation_solutions :
  ∀ p n : ℕ,
    Prime p →
    p^2 + n^2 = 3*p*n + 1 →
    ((p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8)) :=
by sorry

end NUMINAMATH_CALUDE_prime_natural_equation_solutions_l1000_100052


namespace NUMINAMATH_CALUDE_julias_mean_score_l1000_100034

def scores : List ℝ := [88, 90, 92, 94, 95, 97, 98, 99]

def henry_mean : ℝ := 94

theorem julias_mean_score (h1 : scores.length = 8)
                          (h2 : ∃ henry_scores julia_scores : List ℝ,
                                henry_scores.length = 4 ∧
                                julia_scores.length = 4 ∧
                                henry_scores ++ julia_scores = scores)
                          (h3 : ∃ henry_scores : List ℝ,
                                henry_scores.length = 4 ∧
                                henry_scores.sum / 4 = henry_mean) :
  ∃ julia_scores : List ℝ,
    julia_scores.length = 4 ∧
    julia_scores.sum / 4 = 94.25 :=
sorry

end NUMINAMATH_CALUDE_julias_mean_score_l1000_100034


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1000_100026

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (∀ x : ℤ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = 121 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1000_100026


namespace NUMINAMATH_CALUDE_fraction_simplification_l1000_100014

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  (x^2 - x*y) / ((x - y)^2) = x / (x - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1000_100014
