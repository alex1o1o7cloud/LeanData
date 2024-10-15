import Mathlib

namespace NUMINAMATH_CALUDE_bread_slices_per_loaf_l3819_381919

theorem bread_slices_per_loaf :
  ∀ (num_loaves : ℕ) (payment : ℕ) (change : ℕ) (slice_cost : ℚ),
    num_loaves = 3 →
    payment = 40 →
    change = 16 →
    slice_cost = 2/5 →
    (((payment - change : ℚ) / slice_cost) / num_loaves : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_per_loaf_l3819_381919


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3819_381916

theorem inequality_and_equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x + 1/x)^2 + (y + 1/y)^2 ≥ 25/2 ∧
  ((x + 1/x)^2 + (y + 1/y)^2 = 25/2 ↔ x = 1/2 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3819_381916


namespace NUMINAMATH_CALUDE_chord_length_l3819_381947

/-- Circle C with equation x^2 + y^2 - 4x - 4y + 4 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 4 = 0

/-- Line l passing through points (4,0) and (0,2) -/
def line_l (x y : ℝ) : Prop := x + 2*y = 4

/-- The length of the chord cut by line l on circle C is 8√5/5 -/
theorem chord_length : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
  line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2) = (8*Real.sqrt 5/5)^2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l3819_381947


namespace NUMINAMATH_CALUDE_rhombus_perimeter_from_diagonals_l3819_381959

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter_from_diagonals (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 16 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_from_diagonals_l3819_381959


namespace NUMINAMATH_CALUDE_fraction_cube_sum_l3819_381997

theorem fraction_cube_sum : 
  (10 / 11) ^ 3 * (1 / 3) ^ 3 + (1 / 2) ^ 3 = 5492 / 35937 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cube_sum_l3819_381997


namespace NUMINAMATH_CALUDE_c_share_is_64_l3819_381935

/-- Given a total sum of money divided among three parties a, b, and c,
    where b's share is 65% of a's and c's share is 40% of a's,
    prove that c's share is 64 when the total sum is 328. -/
theorem c_share_is_64 (total : ℝ) (a b c : ℝ) :
  total = 328 →
  b = 0.65 * a →
  c = 0.40 * a →
  total = a + b + c →
  c = 64 := by
  sorry

end NUMINAMATH_CALUDE_c_share_is_64_l3819_381935


namespace NUMINAMATH_CALUDE_dog_reachable_area_l3819_381960

/-- The area a dog can reach when tethered to a vertex of a regular octagonal doghouse -/
theorem dog_reachable_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 →
  rope_length = 3 →
  ∃ (area : ℝ), area = 6.5 * Real.pi ∧ 
  area = (rope_length^2 * Real.pi * (240 / 360)) + (2 * (side_length^2 * Real.pi * (45 / 360))) :=
sorry

end NUMINAMATH_CALUDE_dog_reachable_area_l3819_381960


namespace NUMINAMATH_CALUDE_landscape_breadth_l3819_381987

/-- Given a rectangular landscape with a playground, proves that the breadth is 420 meters -/
theorem landscape_breadth (length breadth : ℝ) (playground_area : ℝ) : 
  breadth = 6 * length →
  playground_area = 4200 →
  playground_area = (1 / 7) * (length * breadth) →
  breadth = 420 := by
sorry

end NUMINAMATH_CALUDE_landscape_breadth_l3819_381987


namespace NUMINAMATH_CALUDE_f_zero_values_l3819_381917

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x y : ℝ, f (x + y) = f x * f y)
variable (h3 : deriv f 0 = 2)

-- Theorem statement
theorem f_zero_values : f 0 = 0 ∨ f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_values_l3819_381917


namespace NUMINAMATH_CALUDE_sqrt_200_simplified_l3819_381932

theorem sqrt_200_simplified : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplified_l3819_381932


namespace NUMINAMATH_CALUDE_ball_probabilities_l3819_381958

def total_balls : ℕ := 6
def white_balls : ℕ := 2
def red_balls : ℕ := 2
def yellow_balls : ℕ := 2

def prob_two_red : ℚ := 1 / 15
def prob_same_color : ℚ := 1 / 5
def prob_one_white : ℚ := 2 / 3

theorem ball_probabilities :
  (total_balls = white_balls + red_balls + yellow_balls) →
  (prob_two_red = (red_balls.choose 2) / (total_balls.choose 2)) ∧
  (prob_same_color = (white_balls.choose 2 + red_balls.choose 2 + yellow_balls.choose 2) / (total_balls.choose 2)) ∧
  (prob_one_white = (white_balls * (total_balls - white_balls)) / (total_balls * (total_balls - 1))) :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l3819_381958


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3819_381976

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ (∃ x : ℝ, Real.exp x ≤ x^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3819_381976


namespace NUMINAMATH_CALUDE_angle_between_points_l3819_381910

/-- The angle between two points on a spherical Earth given their coordinates -/
def angleOnSphere (latA longA latB longB : Real) : Real :=
  360 - longA - longB

/-- Point A's coordinates -/
def pointA : (Real × Real) := (0, 100)

/-- Point B's coordinates -/
def pointB : (Real × Real) := (45, -115)

theorem angle_between_points :
  angleOnSphere pointA.1 pointA.2 pointB.1 pointB.2 = 145 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_points_l3819_381910


namespace NUMINAMATH_CALUDE_log_equation_solution_l3819_381946

-- Define the logarithm function
noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- Define the property of being a non-square
def is_non_square (x : ℚ) : Prop := ∀ n : ℕ, x ≠ (n : ℚ)^2

-- Define the property of being a non-cube
def is_non_cube (x : ℚ) : Prop := ∀ n : ℕ, x ≠ (n : ℚ)^3

-- Define the property of being non-integral
def is_non_integral (x : ℚ) : Prop := ∀ n : ℤ, x ≠ n

-- Main theorem
theorem log_equation_solution :
  ∃ x : ℝ, 
    log_base (3 * x) 343 = x ∧ 
    x = 4 / 3 ∧
    (∃ q : ℚ, x = q) ∧
    is_non_square (4 / 3) ∧
    is_non_cube (4 / 3) ∧
    is_non_integral (4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3819_381946


namespace NUMINAMATH_CALUDE_northwest_molded_break_even_price_l3819_381906

/-- Calculate the break-even price per handle for Northwest Molded -/
theorem northwest_molded_break_even_price 
  (variable_cost : ℝ) 
  (fixed_cost : ℝ) 
  (break_even_quantity : ℝ) :
  variable_cost = 0.60 →
  fixed_cost = 7640 →
  break_even_quantity = 1910 →
  (fixed_cost + variable_cost * break_even_quantity) / break_even_quantity = 4.60 :=
by sorry

end NUMINAMATH_CALUDE_northwest_molded_break_even_price_l3819_381906


namespace NUMINAMATH_CALUDE_logarithm_properties_l3819_381993

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem logarithm_properties (a b x : ℝ) (ha : a > 0 ∧ a ≠ 1) (hb : b > 0 ∧ b ≠ 1) (hx : x > 0) :
  (log a x = (log b x) / (log b a)) ∧ (log a b = 1 / (log b a)) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_properties_l3819_381993


namespace NUMINAMATH_CALUDE_smallest_part_in_ratio_l3819_381953

/-- Given a total amount of (3000 + b) divided in the ratio 5:6:8, where the smallest part is c, then c = 100 -/
theorem smallest_part_in_ratio (b : ℝ) (c : ℝ) : 
  (c = (5 : ℝ) / (5 + 6 + 8 : ℝ) * (3000 + b)) → c = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_part_in_ratio_l3819_381953


namespace NUMINAMATH_CALUDE_remainder_1234567_div_256_l3819_381995

theorem remainder_1234567_div_256 : 1234567 % 256 = 57 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567_div_256_l3819_381995


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3819_381927

theorem quadratic_minimum (x : ℝ) : x^2 + 10*x ≥ -25 ∧ ∃ y : ℝ, y^2 + 10*y = -25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3819_381927


namespace NUMINAMATH_CALUDE_square_sum_xy_l3819_381967

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 40)
  (h2 : y * (x + y) = 90)
  (h3 : x = (2/3) * y) : 
  (x + y)^2 = 100 := by
sorry

end NUMINAMATH_CALUDE_square_sum_xy_l3819_381967


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3819_381996

/-- Proves that the rate of interest is 8% given the specified loan conditions -/
theorem interest_rate_calculation (principal : ℝ) (interest : ℝ) 
  (h1 : principal = 1100)
  (h2 : interest = 704)
  (h3 : ∀ r t, interest = principal * r * t / 100 → r = t) :
  ∃ r : ℝ, r = 8 ∧ interest = principal * r * r / 100 := by
  sorry


end NUMINAMATH_CALUDE_interest_rate_calculation_l3819_381996


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3819_381943

theorem complex_modulus_problem (z : ℂ) : z = (Complex.I : ℂ) / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3819_381943


namespace NUMINAMATH_CALUDE_volunteers_next_meeting_l3819_381914

def alison_schedule := 5
def ben_schedule := 3
def carla_schedule := 9
def dave_schedule := 8

theorem volunteers_next_meeting :
  Nat.lcm alison_schedule (Nat.lcm ben_schedule (Nat.lcm carla_schedule dave_schedule)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_next_meeting_l3819_381914


namespace NUMINAMATH_CALUDE_book_pages_sum_l3819_381952

theorem book_pages_sum (chapter1 chapter2 chapter3 : ℕ) 
  (h1 : chapter1 = 66)
  (h2 : chapter2 = 35)
  (h3 : chapter3 = 24) :
  chapter1 + chapter2 + chapter3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_sum_l3819_381952


namespace NUMINAMATH_CALUDE_greatest_power_of_five_l3819_381999

/-- The number of divisors function -/
noncomputable def num_divisors (n : ℕ) : ℕ := sorry

theorem greatest_power_of_five (n : ℕ) 
  (h1 : n > 0)
  (h2 : num_divisors n = 72)
  (h3 : num_divisors (5 * n) = 90) :
  ∃ (k : ℕ) (m : ℕ), n = 5^k * m ∧ m % 5 ≠ 0 ∧ k = 3 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_five_l3819_381999


namespace NUMINAMATH_CALUDE_product_inequality_l3819_381905

theorem product_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧ 
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

#check product_inequality

end NUMINAMATH_CALUDE_product_inequality_l3819_381905


namespace NUMINAMATH_CALUDE_total_pencils_l3819_381998

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of people who bought the color box -/
def num_people : ℕ := 3

/-- Theorem: The total number of pencils Serenity and her two friends have -/
theorem total_pencils : rainbow_colors * num_people = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3819_381998


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l3819_381934

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : (4/3 * π * r₁^3) / (4/3 * π * r₂^3) = 8/27) :
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4/9 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l3819_381934


namespace NUMINAMATH_CALUDE_minus_one_circle_plus_minus_four_l3819_381963

-- Define the ⊕ operation
def circle_plus (a b : ℝ) : ℝ := a + b - a * b

-- Theorem statement
theorem minus_one_circle_plus_minus_four :
  circle_plus (-1) (-4) = -9 := by
  sorry

end NUMINAMATH_CALUDE_minus_one_circle_plus_minus_four_l3819_381963


namespace NUMINAMATH_CALUDE_building_height_ratio_l3819_381938

/-- Given three buildings with specific height relationships, prove the ratio of the second to the first building's height. -/
theorem building_height_ratio :
  ∀ (h₁ h₂ h₃ : ℝ),
  h₁ = 600 →
  h₃ = 3 * (h₁ + h₂) →
  h₁ + h₂ + h₃ = 7200 →
  h₂ / h₁ = 2 := by
sorry

end NUMINAMATH_CALUDE_building_height_ratio_l3819_381938


namespace NUMINAMATH_CALUDE_h_ratio_theorem_l3819_381904

/-- Sum of even integers from 2 to n, inclusive, for even n -/
def h (n : ℕ) : ℚ :=
  if n % 2 = 0 then (n / 2) * (n + 2) / 4 else 0

theorem h_ratio_theorem (m k n : ℕ) (h_even : Even n) :
  h (m * n) / h (k * n) = (m : ℚ) / k * (m / k + 1) := by
  sorry

end NUMINAMATH_CALUDE_h_ratio_theorem_l3819_381904


namespace NUMINAMATH_CALUDE_triangle_area_l3819_381989

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  B = 60 * π / 180 →
  c = 3 →
  b = Real.sqrt 7 →
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) ∨
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l3819_381989


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3819_381979

theorem quadratic_roots_range (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + 1 = -2*k ∧ y^2 - 4*y + 1 = -2*k) → k < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3819_381979


namespace NUMINAMATH_CALUDE_pie_slices_theorem_l3819_381901

/-- Given the total number of pie slices sold and the number sold yesterday,
    calculate the number of slices served today. -/
def slices_served_today (total : ℕ) (yesterday : ℕ) : ℕ :=
  total - yesterday

theorem pie_slices_theorem :
  slices_served_today 7 5 = 2 :=
by sorry

end NUMINAMATH_CALUDE_pie_slices_theorem_l3819_381901


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_implies_k_nonnegative_l3819_381990

/-- A line is defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate to check if a point (x, y) is in the third quadrant -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Predicate to check if a line passes through the third quadrant -/
def passes_through_third_quadrant (l : Line) : Prop :=
  ∃ x y : ℝ, y = l.slope * x + l.intercept ∧ in_third_quadrant x y

/-- Theorem: If a line with slope -3 and y-intercept k does not pass through the third quadrant, then k ≥ 0 -/
theorem line_not_in_third_quadrant_implies_k_nonnegative :
  ∀ k : ℝ, ¬passes_through_third_quadrant ⟨-3, k⟩ → k ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_implies_k_nonnegative_l3819_381990


namespace NUMINAMATH_CALUDE_horner_method_operations_l3819_381918

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Count operations in Horner's method -/
def horner_count (coeffs : List ℝ) : Nat × Nat :=
  (coeffs.length - 1, coeffs.length - 1)

/-- The polynomial f(x) = 6x^6 + 4x^5 - 2x^4 + 5x^3 - 7x^2 - 2x + 5 -/
def f : List ℝ := [6, 4, -2, 5, -7, -2, 5]

theorem horner_method_operations :
  horner_count f = (6, 3) ∧
  horner_eval f 2 = f.foldl (fun acc a => acc * 2 + a) 0 :=
sorry

end NUMINAMATH_CALUDE_horner_method_operations_l3819_381918


namespace NUMINAMATH_CALUDE_statement_b_not_always_true_l3819_381951

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Intersection of two planes -/
def plane_intersection (p1 p2 : Plane3D) : Line3D :=
  sorry

/-- Statement B is not always true -/
theorem statement_b_not_always_true :
  ∃ (a : Line3D) (α β : Plane3D),
    parallel_line_plane a α ∧
    plane_intersection α β = b ∧
    ¬ parallel_lines a b :=
  sorry

end NUMINAMATH_CALUDE_statement_b_not_always_true_l3819_381951


namespace NUMINAMATH_CALUDE_units_digit_product_l3819_381928

theorem units_digit_product (a b c : ℕ) : 
  (3^1004 * 7^1003 * 17^1002) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l3819_381928


namespace NUMINAMATH_CALUDE_marbles_difference_l3819_381945

/-- Given information about Josh's marble collection -/
structure MarbleCollection where
  initial : ℕ
  found : ℕ
  lost : ℕ

/-- Theorem stating the difference between lost and found marbles -/
theorem marbles_difference (josh : MarbleCollection)
  (h1 : josh.initial = 15)
  (h2 : josh.found = 9)
  (h3 : josh.lost = 23) :
  josh.lost - josh.found = 14 := by
  sorry

end NUMINAMATH_CALUDE_marbles_difference_l3819_381945


namespace NUMINAMATH_CALUDE_cube_face_perimeter_l3819_381956

-- Define the volume of the cube
def cube_volume : ℝ := 343

-- Theorem statement
theorem cube_face_perimeter :
  let side_length := (cube_volume ^ (1/3 : ℝ))
  (4 : ℝ) * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_perimeter_l3819_381956


namespace NUMINAMATH_CALUDE_canoe_kayak_ratio_l3819_381911

theorem canoe_kayak_ratio : 
  ∀ (C K : ℕ),
  (9 * C + 12 * K = 432) →  -- Total revenue
  (C = K + 6) →             -- 6 more canoes than kayaks
  (∃ (n : ℕ), C = 3 * n * K) →  -- Canoes are a multiple of 3 times kayaks
  (C : ℚ) / K = 4 / 3 :=    -- Ratio of canoes to kayaks is 4:3
by
  sorry

#check canoe_kayak_ratio

end NUMINAMATH_CALUDE_canoe_kayak_ratio_l3819_381911


namespace NUMINAMATH_CALUDE_system_solution_implies_2a_minus_3b_equals_6_l3819_381921

theorem system_solution_implies_2a_minus_3b_equals_6
  (a b : ℝ)
  (eq1 : a * 2 - b * 1 = 4)
  (eq2 : a * 2 + b * 1 = 2) :
  2 * a - 3 * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_implies_2a_minus_3b_equals_6_l3819_381921


namespace NUMINAMATH_CALUDE_place_mat_length_l3819_381950

theorem place_mat_length (r : ℝ) (n : ℕ) (y : ℝ) : 
  r = 5 → n = 8 → y = 2 * r * Real.sin (π / (2 * n)) → y = 5 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

#check place_mat_length

end NUMINAMATH_CALUDE_place_mat_length_l3819_381950


namespace NUMINAMATH_CALUDE_root_zero_implies_m_six_l3819_381937

theorem root_zero_implies_m_six (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x + m - 6 = 0 ∧ x = 0) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_zero_implies_m_six_l3819_381937


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3819_381948

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 306) :
  ∃ n : ℕ, n > 0 ∧ n * (n - 1) * 2 = total_games ∧ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l3819_381948


namespace NUMINAMATH_CALUDE_hyperbola_semi_focal_distance_l3819_381977

/-- Given a hyperbola with equation x²/20 - y²/5 = 1, its semi-focal distance is 5 -/
theorem hyperbola_semi_focal_distance :
  ∀ (x y : ℝ), x^2 / 20 - y^2 / 5 = 1 → ∃ (c : ℝ), c = 5 ∧ c^2 = 20 + 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_semi_focal_distance_l3819_381977


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3819_381936

theorem arithmetic_sequence_problem :
  ∃ (a b c : ℝ), 
    (a > b ∧ b > c) ∧  -- Monotonically decreasing
    (b - a = c - b) ∧  -- Arithmetic sequence
    (a + b + c = 12) ∧ -- Sum is 12
    (a * b * c = 48) ∧ -- Product is 48
    (a = 6 ∧ b = 4 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3819_381936


namespace NUMINAMATH_CALUDE_complement_of_M_l3819_381973

def U : Set ℕ := {1,2,3,4,5,6}
def M : Set ℕ := {1,2,4}

theorem complement_of_M : Mᶜ = {3,5,6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l3819_381973


namespace NUMINAMATH_CALUDE_direct_proportion_relationship_l3819_381930

theorem direct_proportion_relationship (x y : ℝ) :
  (∃ k : ℝ, ∀ x, y - 2 = k * x) →  -- y-2 is directly proportional to x
  (1 = 1 ∧ y = -6) →              -- when x=1, y=-6
  y = -8 * x + 2 :=                -- relationship between y and x
by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_relationship_l3819_381930


namespace NUMINAMATH_CALUDE_number_square_relationship_l3819_381984

theorem number_square_relationship (n : ℝ) (h1 : n ≠ 0) (h2 : (n + n^2) / 2 = 5 * n) (h3 : n = 9) :
  (n + n^2) / 2 = 5 * n :=
by sorry

end NUMINAMATH_CALUDE_number_square_relationship_l3819_381984


namespace NUMINAMATH_CALUDE_emilys_journey_l3819_381971

theorem emilys_journey (total : ℝ) 
  (h1 : total / 5 + 30 + total / 3 + total / 6 = total) : total = 100 := by
  sorry

end NUMINAMATH_CALUDE_emilys_journey_l3819_381971


namespace NUMINAMATH_CALUDE_robot_returns_to_start_l3819_381922

/-- Represents a robot's movement pattern -/
structure RobotMovement where
  turn_interval : ℕ  -- Time in seconds between turns
  turn_angle : ℕ     -- Angle of turn in degrees

/-- Represents the state of the robot -/
structure RobotState where
  position : ℤ × ℤ   -- (x, y) coordinates
  direction : ℕ      -- 0: North, 1: East, 2: South, 3: West

/-- Calculates the new position after one movement -/
def move (state : RobotState) : RobotState :=
  match state.direction with
  | 0 => { state with position := (state.position.1, state.position.2 + 1) }
  | 1 => { state with position := (state.position.1 + 1, state.position.2) }
  | 2 => { state with position := (state.position.1, state.position.2 - 1) }
  | 3 => { state with position := (state.position.1 - 1, state.position.2) }
  | _ => state

/-- Calculates the new direction after turning -/
def turn (state : RobotState) : RobotState :=
  { state with direction := (state.direction + 1) % 4 }

/-- Simulates the robot's movement for a given number of seconds -/
def simulate (movement : RobotMovement) (initial_state : RobotState) (time : ℕ) : RobotState :=
  if time = 0 then initial_state
  else
    let new_state := if time % movement.turn_interval = 0 
                     then turn (move initial_state)
                     else move initial_state
    simulate movement new_state (time - 1)

/-- Theorem: The robot returns to its starting point after 6 minutes -/
theorem robot_returns_to_start (movement : RobotMovement) 
  (h1 : movement.turn_interval = 15)
  (h2 : movement.turn_angle = 90) :
  let initial_state : RobotState := ⟨(0, 0), 0⟩
  let final_state := simulate movement initial_state (6 * 60)
  final_state.position = initial_state.position :=
by sorry


end NUMINAMATH_CALUDE_robot_returns_to_start_l3819_381922


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3819_381949

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem rationalize_denominator :
  (1 : ℝ) / (cubeRoot 2 + cubeRoot 16) = cubeRoot 4 / 6 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3819_381949


namespace NUMINAMATH_CALUDE_savings_percentage_increase_l3819_381974

theorem savings_percentage_increase (initial_salary : ℝ) : 
  let last_year_savings := 0.10 * initial_salary
  let this_year_salary := 1.10 * initial_salary
  let this_year_savings := 0.15 * this_year_salary
  (this_year_savings / last_year_savings) * 100 = 165 := by
sorry

end NUMINAMATH_CALUDE_savings_percentage_increase_l3819_381974


namespace NUMINAMATH_CALUDE_interior_angle_measure_l3819_381907

/-- Given a triangle with an interior angle, if the measures of the three triangle angles are known,
    then the measure of the interior angle can be determined. -/
theorem interior_angle_measure (m1 m2 m3 m4 : ℝ) : 
  m1 = 62 → m2 = 36 → m3 = 24 → 
  m1 + m2 + m3 + m4 < 360 →
  m4 = 122 := by
  sorry

#check interior_angle_measure

end NUMINAMATH_CALUDE_interior_angle_measure_l3819_381907


namespace NUMINAMATH_CALUDE_equal_distribution_of_items_l3819_381981

theorem equal_distribution_of_items (pencils erasers friends : ℕ) 
  (h1 : pencils = 35) 
  (h2 : erasers = 5) 
  (h3 : friends = 5) : 
  (pencils + erasers) / friends = 8 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_items_l3819_381981


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3819_381978

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  a_1_eq_1 : a 1 = 1
  is_arithmetic : ∃ d ≠ 0, ∀ n : ℕ+, a (n + 1) = a n + d
  is_geometric : (a 2)^2 = a 1 * a 5

/-- The b_n sequence derived from the arithmetic sequence -/
def b (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  1 / (seq.a n * seq.a (n + 1))

/-- The sum of the first n terms of the b sequence -/
def T (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (Finset.range n).sum (λ i => b seq ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ+, seq.a n = 2 * n - 1) ∧
  (∀ n : ℕ+, T seq n = n / (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3819_381978


namespace NUMINAMATH_CALUDE_cupcake_package_size_l3819_381933

theorem cupcake_package_size :
  ∀ (small_package_size : ℕ) (total_cupcakes : ℕ) (small_packages : ℕ) (larger_package_size : ℕ),
    small_package_size = 10 →
    total_cupcakes = 100 →
    small_packages = 4 →
    total_cupcakes = small_package_size * small_packages + larger_package_size →
    larger_package_size = 60 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_package_size_l3819_381933


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3819_381900

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 2/7
  let a₂ : ℚ := 10/49
  let a₃ : ℚ := 50/343
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → a₁ * r^(n-1) = (2/7) * (5/7)^(n-1)) →
  r = 5/7 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3819_381900


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3819_381944

theorem quadratic_inequality_solution_range (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 6*x + c < 0) ↔ (c > 0 ∧ c < 9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3819_381944


namespace NUMINAMATH_CALUDE_wiener_age_theorem_l3819_381968

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

theorem wiener_age_theorem :
  ∃! a : ℕ, 
    is_four_digit (a^3) ∧ 
    is_six_digit (a^4) ∧ 
    (digits (a^3) ++ digits (a^4)).Nodup ∧
    (digits (a^3) ++ digits (a^4)).length = 10 ∧
    a = 18 := by
  sorry

end NUMINAMATH_CALUDE_wiener_age_theorem_l3819_381968


namespace NUMINAMATH_CALUDE_ball_problem_l3819_381913

/-- The number of red balls is one more than the number of yellow balls -/
def num_red (a : ℕ) : ℕ := a + 1

/-- The number of yellow balls -/
def num_yellow (a : ℕ) : ℕ := a

/-- The number of blue balls is always 1 -/
def num_blue : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls (a : ℕ) : ℕ := num_red a + num_yellow a + num_blue

/-- The score for drawing a red ball -/
def score_red : ℕ := 1

/-- The score for drawing a yellow ball -/
def score_yellow : ℕ := 2

/-- The score for drawing a blue ball -/
def score_blue : ℕ := 3

/-- The expected value of drawing a ball -/
def expected_value (a : ℕ) : ℚ :=
  (score_red * num_red a + score_yellow * num_yellow a + score_blue * num_blue) / total_balls a

/-- The theorem to be proved -/
theorem ball_problem (a : ℕ) (h1 : a > 0) (h2 : expected_value a = 5/3) :
  a = 2 ∧ (3 : ℚ)/10 = (Nat.choose (num_red 2) 1 * Nat.choose (num_yellow 2) 2 + 
                         Nat.choose (num_red 2) 2 * Nat.choose num_blue 1) / Nat.choose (total_balls 2) 3 :=
by sorry

end NUMINAMATH_CALUDE_ball_problem_l3819_381913


namespace NUMINAMATH_CALUDE_special_blend_probability_l3819_381985

theorem special_blend_probability : 
  let n : ℕ := 6  -- Total number of visits
  let k : ℕ := 5  -- Number of times the special blend is served
  let p : ℚ := 3/4  -- Probability of serving the special blend each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 1458/4096 := by
sorry

end NUMINAMATH_CALUDE_special_blend_probability_l3819_381985


namespace NUMINAMATH_CALUDE_probability_divisible_by_15_l3819_381915

/-- The set of integers between 1 and 2^30 with exactly two 1s in their binary expansions -/
def T : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2^30 ∧ (∃ j k : ℕ, j < k ∧ k < 30 ∧ n = 2^j + 2^k)}

/-- The number of elements in T -/
def T_count : ℕ := 435

/-- The number of elements in T divisible by 15 -/
def T_div15_count : ℕ := 28

theorem probability_divisible_by_15 :
  (T_div15_count : ℚ) / T_count = 28 / 435 ∧ 28 + 435 = 463 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_15_l3819_381915


namespace NUMINAMATH_CALUDE_gcd_n_squared_plus_four_n_plus_three_l3819_381909

theorem gcd_n_squared_plus_four_n_plus_three (n : ℕ) (h : n > 4) :
  Nat.gcd (n^2 + 4) (n + 3) = if (n + 3) % 13 = 0 then 13 else 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_squared_plus_four_n_plus_three_l3819_381909


namespace NUMINAMATH_CALUDE_lunchroom_students_l3819_381920

/-- The number of students sitting at each table -/
def students_per_table : ℕ := 6

/-- The number of tables in the lunchroom -/
def number_of_tables : ℕ := 34

/-- The total number of students in the lunchroom -/
def total_students : ℕ := students_per_table * number_of_tables

theorem lunchroom_students : total_students = 204 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l3819_381920


namespace NUMINAMATH_CALUDE_distance_roja_pooja_distance_sooraj_pole_angle_roja_pooja_l3819_381926

-- Define the speeds and time
def roja_speed : ℝ := 5
def pooja_speed : ℝ := 3
def sooraj_speed : ℝ := 4
def time : ℝ := 4

-- Define the distances traveled
def roja_distance : ℝ := roja_speed * time
def pooja_distance : ℝ := pooja_speed * time
def sooraj_distance : ℝ := sooraj_speed * time

-- Theorem for the distance between Roja and Pooja
theorem distance_roja_pooja : 
  Real.sqrt (roja_distance ^ 2 + pooja_distance ^ 2) = Real.sqrt 544 :=
sorry

-- Theorem for the distance between Sooraj and the pole
theorem distance_sooraj_pole : sooraj_distance = 16 :=
sorry

-- Theorem for the angle between Roja and Pooja's directions
theorem angle_roja_pooja : ∃ (angle : ℝ), angle = 90 :=
sorry

end NUMINAMATH_CALUDE_distance_roja_pooja_distance_sooraj_pole_angle_roja_pooja_l3819_381926


namespace NUMINAMATH_CALUDE_multiple_inequalities_l3819_381983

theorem multiple_inequalities :
  (∃ a b : ℝ, a + b < 2 * Real.sqrt (a * b)) ∧
  (∃ a : ℝ, a + 1 / a ≤ 2) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → b / a + a / b ≥ 2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → 2 / x + 1 / y ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_multiple_inequalities_l3819_381983


namespace NUMINAMATH_CALUDE_trees_on_road_l3819_381986

theorem trees_on_road (road_length : ℕ) (interval : ℕ) (trees : ℕ) : 
  road_length = 156 ∧ 
  interval = 6 ∧ 
  trees = road_length / interval + 1 →
  trees = 27 :=
by sorry

end NUMINAMATH_CALUDE_trees_on_road_l3819_381986


namespace NUMINAMATH_CALUDE_book_cd_price_difference_l3819_381955

/-- Proves that the difference between book price and CD price is $4 -/
theorem book_cd_price_difference :
  let album_price : ℝ := 20
  let cd_price : ℝ := 0.7 * album_price
  let book_price : ℝ := 18
  book_price - cd_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_book_cd_price_difference_l3819_381955


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3819_381966

theorem right_triangle_sides : ∃ (a b c : ℝ), 
  (a = 1 ∧ b = Real.sqrt 3 ∧ c = 2) ∧ 
  (a^2 + b^2 = c^2) ∧
  ¬(3^2 + 4^2 = 6^2) ∧
  ¬(5^2 + 12^2 = 14^2) ∧
  ¬((Real.sqrt 2)^2 + (Real.sqrt 3)^2 = 2^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3819_381966


namespace NUMINAMATH_CALUDE_cricket_average_l3819_381941

theorem cricket_average (innings : Nat) (next_runs : Nat) (increase : Nat) (current_average : Nat) : 
  innings = 10 →
  next_runs = 84 →
  increase = 4 →
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase →
  current_average = 40 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l3819_381941


namespace NUMINAMATH_CALUDE_lcm_1806_1230_l3819_381903

theorem lcm_1806_1230 : Nat.lcm 1806 1230 = 247230 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1806_1230_l3819_381903


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_count_l3819_381980

/-- 
Given an arithmetic sequence with:
- First term: a = -48
- Last term: l = 72
- Common difference: d = 6

Prove that the number of terms in the sequence is 21.
-/
theorem arithmetic_sequence_terms_count : 
  ∀ (a l d : ℤ) (n : ℕ),
  a = -48 →
  l = 72 →
  d = 6 →
  l = a + (n - 1) * d →
  n = 21 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_terms_count_l3819_381980


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3819_381962

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a b : Line) (α : Plane) 
  (h1 : a ≠ b)
  (h2 : parallel a α) 
  (h3 : perpendicular b α) : 
  perpendicularLines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3819_381962


namespace NUMINAMATH_CALUDE_megan_seashells_l3819_381965

/-- Given that Megan has 19 seashells and wants to have 25 seashells in total,
    prove that she needs to find 6 more seashells. -/
theorem megan_seashells (current : ℕ) (target : ℕ) (h1 : current = 19) (h2 : target = 25) :
  target - current = 6 := by
  sorry

end NUMINAMATH_CALUDE_megan_seashells_l3819_381965


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3819_381964

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 5 ∧ x^2 + a*x + 4 < 0) → a < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3819_381964


namespace NUMINAMATH_CALUDE_system_solution_l3819_381954

theorem system_solution : ∃! (x y : ℝ), 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - y = (x - 2) + (y - 2)) ∧ 
  x = 3 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3819_381954


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3819_381972

/-- Given a > 0 and f(x) = ax² + bx + c, with x₀ satisfying 2ax + b = 0,
    prove that f(x) ≥ f(x₀) for all x ∈ ℝ -/
theorem quadratic_minimum (a b c : ℝ) (ha : a > 0) :
  let f := fun x => a * x^2 + b * x + c
  let x₀ := -b / (2 * a)
  ∀ x, f x ≥ f x₀ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3819_381972


namespace NUMINAMATH_CALUDE_expression_value_l3819_381991

theorem expression_value (m n a b x : ℝ) 
  (h1 : m = -n)  -- m and n are opposites
  (h2 : a * b = -1)  -- a and b are negative reciprocals
  (h3 : |x| = 3)  -- absolute value of x equals 3
  : x^3 - (1 + m + n + a*b) * x^2 + (m + n) * x^2004 + (a*b)^2005 = 26 ∨ 
    x^3 - (1 + m + n + a*b) * x^2 + (m + n) * x^2004 + (a*b)^2005 = -28 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3819_381991


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l3819_381931

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequalities :
  ∀ a b c : ℝ,
  -- Part 1
  (∀ x : ℝ, -3 < x → x < 4 → f a b c x > 0) →
  (∀ x : ℝ, -3 < x → x < 5 → b * x^2 + 2 * a * x - (c + 3 * b) < 0) ∧
  -- Part 2
  (b = 2 → a > c → (∀ x : ℝ, f a b c x ≥ 0) → (∃ x₀ : ℝ, f a b c x₀ = 0) →
    ∃ min : ℝ, min = 2 * Real.sqrt 2 ∧ ∀ x : ℝ, (a^2 + c^2) / (a - c) ≥ min) ∧
  -- Part 3
  (a < b → (∀ x : ℝ, f a b c x ≥ 0) →
    ∃ min : ℝ, min = 8 ∧ ∀ x : ℝ, (a + 2 * b + 4 * c) / (b - a) ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l3819_381931


namespace NUMINAMATH_CALUDE_angle_measure_l3819_381969

theorem angle_measure (A B : ℝ) (h1 : A + B = 180) (h2 : A = 7 * B) : A = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3819_381969


namespace NUMINAMATH_CALUDE_dog_ratio_proof_l3819_381902

/-- Proves that for 12 dogs with 36 paws on the ground, split equally between those on back legs and all fours, the ratio of dogs on back legs to all fours is 1:1 -/
theorem dog_ratio_proof (total_dogs : ℕ) (total_paws : ℕ) 
  (h1 : total_dogs = 12) 
  (h2 : total_paws = 36) 
  (h3 : ∃ x y : ℕ, x + y = total_dogs ∧ x = y ∧ 2*x + 4*y = total_paws) : 
  ∃ x y : ℕ, x + y = total_dogs ∧ x = y ∧ x / y = 1 := by
  sorry

#check dog_ratio_proof

end NUMINAMATH_CALUDE_dog_ratio_proof_l3819_381902


namespace NUMINAMATH_CALUDE_irrational_sqrt_N_l3819_381994

def N (n : ℕ) : ℚ :=
  (10^n - 1) / 9 * 10^(2*n) + 4 * (10^(2*n) - 1) / 9

theorem irrational_sqrt_N (n : ℕ) (h : n > 1) :
  Irrational (Real.sqrt (N n)) :=
sorry

end NUMINAMATH_CALUDE_irrational_sqrt_N_l3819_381994


namespace NUMINAMATH_CALUDE_log_function_not_in_fourth_quadrant_l3819_381982

-- Define the logarithm function
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function y = log_a(x+b)
noncomputable def f (a b x : ℝ) : ℝ := log_base a (x + b)

-- Theorem statement
theorem log_function_not_in_fourth_quadrant (a b : ℝ) 
  (ha : a > 1) (hb : b < -1) :
  ∀ x y : ℝ, f a b x = y → ¬(x > 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_log_function_not_in_fourth_quadrant_l3819_381982


namespace NUMINAMATH_CALUDE_birthday_savings_growth_l3819_381975

/-- Calculates the final amount in a bank account after one year, given an initial amount and an annual interest rate. -/
def final_amount (initial_amount : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_amount * (1 + interest_rate)

/-- Theorem: Given an initial amount of $90 and an annual interest rate of 10%, 
    the final amount after 1 year with no withdrawals is $99. -/
theorem birthday_savings_growth : final_amount 90 0.1 = 99 := by
  sorry

end NUMINAMATH_CALUDE_birthday_savings_growth_l3819_381975


namespace NUMINAMATH_CALUDE_thermos_capacity_is_16_l3819_381939

/-- The capacity of a coffee thermos -/
def thermos_capacity (fills_per_day : ℕ) (days_per_week : ℕ) (current_consumption : ℚ) (normal_consumption_ratio : ℚ) : ℚ :=
  (current_consumption / normal_consumption_ratio) / (fills_per_day * days_per_week)

/-- Proof that the thermos capacity is 16 ounces -/
theorem thermos_capacity_is_16 :
  thermos_capacity 2 5 40 (1/4) = 16 := by
  sorry

end NUMINAMATH_CALUDE_thermos_capacity_is_16_l3819_381939


namespace NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l3819_381924

theorem choose_three_cooks_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 3 → Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_cooks_from_ten_l3819_381924


namespace NUMINAMATH_CALUDE_marble_count_l3819_381925

theorem marble_count (r g b : ℕ) : 
  g + b = 6 →
  r + b = 8 →
  r + g = 4 →
  r + g + b = 9 := by sorry

end NUMINAMATH_CALUDE_marble_count_l3819_381925


namespace NUMINAMATH_CALUDE_actual_distance_towns_distance_proof_l3819_381940

/-- Calculates the actual distance between two towns given the map distance and scale. -/
theorem actual_distance (map_distance : ℝ) (scale_distance : ℝ) (scale_miles : ℝ) : ℝ :=
  let miles_per_inch := scale_miles / scale_distance
  map_distance * miles_per_inch

/-- Proves that the actual distance between two towns is 400 miles given the specified conditions. -/
theorem towns_distance_proof :
  actual_distance 20 0.5 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_towns_distance_proof_l3819_381940


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3819_381992

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2 * Real.sqrt 5 - 1
  (1 / (x^2 + 2*x + 1)) * (1 + 3 / (x - 1)) / ((x + 2) / (x^2 - 1)) = Real.sqrt 5 / 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3819_381992


namespace NUMINAMATH_CALUDE_tangent_line_circle_product_range_l3819_381942

theorem tangent_line_circle_product_range (a b : ℝ) :
  a > 0 →
  b > 0 →
  (∃ x y : ℝ, x + y = 1 ∧ (x - a)^2 + (y - b)^2 = 2) →
  (∀ x y : ℝ, x + y = 1 → (x - a)^2 + (y - b)^2 ≥ 2) →
  0 < a * b ∧ a * b ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_circle_product_range_l3819_381942


namespace NUMINAMATH_CALUDE_positive_real_inequality_l3819_381912

theorem positive_real_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^4 * b^b * c^c ≥ min a (min b c) * min b (min a c) * min c (min a b) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l3819_381912


namespace NUMINAMATH_CALUDE_percentage_of_indian_women_l3819_381970

theorem percentage_of_indian_women (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (percent_indian_men : ℝ) (percent_indian_children : ℝ) (percent_not_indian : ℝ) :
  total_men = 500 →
  total_women = 300 →
  total_children = 500 →
  percent_indian_men = 10 →
  percent_indian_children = 70 →
  percent_not_indian = 55.38461538461539 →
  ∃ (percent_indian_women : ℝ),
    percent_indian_women = 60 ∧
    (percent_indian_men / 100 * total_men + percent_indian_women / 100 * total_women + percent_indian_children / 100 * total_children) /
    (total_men + total_women + total_children : ℝ) = 1 - percent_not_indian / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_indian_women_l3819_381970


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3819_381988

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (carbon_atoms hydrogen_atoms oxygen_atoms : ℕ) 
  (carbon_weight hydrogen_weight oxygen_weight : ℝ) : ℝ :=
  (carbon_atoms : ℝ) * carbon_weight + 
  (hydrogen_atoms : ℝ) * hydrogen_weight + 
  (oxygen_atoms : ℝ) * oxygen_weight

/-- The molecular weight of a compound with 7 Carbon, 6 Hydrogen, and 2 Oxygen atoms is approximately 122.118 g/mol -/
theorem compound_molecular_weight : 
  ∀ (ε : ℝ), ε > 0 → 
  |molecular_weight 7 6 2 12.01 1.008 16.00 - 122.118| < ε :=
by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3819_381988


namespace NUMINAMATH_CALUDE_square_difference_division_problem_solution_l3819_381923

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by
  sorry

theorem problem_solution : (315^2 - 285^2) / 30 = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_square_difference_division_problem_solution_l3819_381923


namespace NUMINAMATH_CALUDE_city_H_highest_increase_l3819_381929

structure City where
  name : String
  pop1990 : ℕ
  pop2000 : ℕ
  event_factor : ℚ

def effective_increase (c : City) : ℚ :=
  (c.pop2000 * c.event_factor - c.pop1990) / c.pop1990

def cities : List City := [
  ⟨"F", 90000, 120000, 11/10⟩,
  ⟨"G", 80000, 110000, 19/20⟩,
  ⟨"H", 70000, 115000, 11/10⟩,
  ⟨"I", 65000, 100000, 49/50⟩,
  ⟨"J", 95000, 145000, 1⟩
]

theorem city_H_highest_increase :
  ∃ c ∈ cities, c.name = "H" ∧
    ∀ c' ∈ cities, effective_increase c ≥ effective_increase c' := by
  sorry

end NUMINAMATH_CALUDE_city_H_highest_increase_l3819_381929


namespace NUMINAMATH_CALUDE_new_ratio_after_addition_l3819_381961

theorem new_ratio_after_addition (x : ℤ) : 
  (x : ℚ) / (4 * x : ℚ) = 1 / 4 →
  4 * x = 24 →
  (x + 6 : ℚ) / (4 * x : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_new_ratio_after_addition_l3819_381961


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3819_381908

theorem least_positive_integer_with_remainders (M : ℕ) : 
  (M % 11 = 10 ∧ M % 12 = 11 ∧ M % 13 = 12 ∧ M % 14 = 13) → 
  (∀ n : ℕ, n > 0 ∧ n % 11 = 10 ∧ n % 12 = 11 ∧ n % 13 = 12 ∧ n % 14 = 13 → M ≤ n) → 
  M = 30029 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3819_381908


namespace NUMINAMATH_CALUDE_football_practice_kicks_l3819_381957

/-- The number of penalty kicks in a football practice session. -/
def penalty_kicks (total_players : ℕ) (goalkeepers : ℕ) : ℕ :=
  goalkeepers * (total_players - 1)

/-- Theorem: In a football club with 22 players including 4 goalkeepers,
    where each outfield player shoots once against each goalkeeper,
    the total number of penalty kicks is 84. -/
theorem football_practice_kicks :
  penalty_kicks 22 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_football_practice_kicks_l3819_381957
