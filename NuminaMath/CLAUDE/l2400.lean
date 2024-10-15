import Mathlib

namespace NUMINAMATH_CALUDE_picture_book_shelves_l2400_240022

/-- Given a bookcase with the following properties:
  * Each shelf contains exactly 6 books
  * There are 5 shelves of mystery books
  * The total number of books is 54
  Prove that the number of shelves of picture books is 4 -/
theorem picture_book_shelves :
  ∀ (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ),
    books_per_shelf = 6 →
    mystery_shelves = 5 →
    total_books = 54 →
    (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 4 :=
by sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l2400_240022


namespace NUMINAMATH_CALUDE_raviraj_cycling_journey_l2400_240050

/-- Raviraj's cycling journey --/
theorem raviraj_cycling_journey (initial_south distance_west_1 distance_north distance_west_2 distance_to_home : ℝ) :
  distance_west_1 = 10 ∧
  distance_north = 20 ∧
  distance_west_2 = 20 ∧
  distance_to_home = 30 ∧
  distance_west_1 + distance_west_2 = distance_to_home ∧
  initial_south + distance_north = distance_to_home →
  initial_south = 10 := by sorry

end NUMINAMATH_CALUDE_raviraj_cycling_journey_l2400_240050


namespace NUMINAMATH_CALUDE_garland_arrangement_count_l2400_240007

def blue_bulbs : ℕ := 5
def red_bulbs : ℕ := 6
def white_bulbs : ℕ := 7

def total_non_white_bulbs : ℕ := blue_bulbs + red_bulbs
def total_spaces : ℕ := total_non_white_bulbs + 1

theorem garland_arrangement_count :
  (Nat.choose total_non_white_bulbs blue_bulbs) * (Nat.choose total_spaces white_bulbs) = 365904 :=
sorry

end NUMINAMATH_CALUDE_garland_arrangement_count_l2400_240007


namespace NUMINAMATH_CALUDE_triangle_theorem_l2400_240011

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle ABC -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.cos t.C + Real.cos t.A * Real.cos t.B = Real.sqrt 3 * Real.sin t.A * Real.cos t.B)
  (h2 : t.a + t.c = 1)
  (h3 : 0 < t.B)
  (h4 : t.B < Real.pi)
  (h5 : 0 < t.a)
  (h6 : t.a < 1) :
  Real.cos t.B = 1/2 ∧ 1/2 ≤ t.b ∧ t.b < 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2400_240011


namespace NUMINAMATH_CALUDE_max_value_fraction_l2400_240003

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y - x*y = 0) :
  (4 / (x + y)) ≤ 4/9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 4*y - x*y = 0 ∧ 4 / (x + y) = 4/9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2400_240003


namespace NUMINAMATH_CALUDE_symmetric_line_l2400_240083

/-- Given a line L1 with equation x - 4y + 2 = 0 and an axis of symmetry x = -2,
    the symmetric line L2 has the equation x + 4y + 2 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (∃ L1 : Set (ℝ × ℝ), L1 = {(x, y) | x - 4*y + 2 = 0}) →
  (∃ axis : Set (ℝ × ℝ), axis = {(x, y) | x = -2}) →
  (∃ L2 : Set (ℝ × ℝ), L2 = {(x, y) | x + 4*y + 2 = 0}) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_l2400_240083


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2400_240058

-- Define the universal set U
def U : Finset ℕ := {0, 1, 2, 3, 4}

-- Define set M
def M : Finset ℕ := {1, 2, 4}

-- Define set N
def N : Finset ℕ := {2, 3}

-- Theorem statement
theorem complement_union_theorem : 
  (U \ M) ∪ N = {0, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2400_240058


namespace NUMINAMATH_CALUDE_A_inter_B_eq_l2400_240040

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 < 0}

theorem A_inter_B_eq : A ∩ B = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_A_inter_B_eq_l2400_240040


namespace NUMINAMATH_CALUDE_increasing_not_always_unbounded_and_decreasing_not_always_unbounded_l2400_240031

-- Define a constantly increasing function
def constantlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define a constantly decreasing function
def constantlyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define a function that is bounded above
def boundedAbove (f : ℝ → ℝ) : Prop :=
  ∃ M, ∀ x, f x ≤ M

-- Define a function that is bounded below
def boundedBelow (f : ℝ → ℝ) : Prop :=
  ∃ m, ∀ x, f x ≥ m

-- Theorem statement
theorem increasing_not_always_unbounded_and_decreasing_not_always_unbounded :
  (∃ f : ℝ → ℝ, constantlyIncreasing f ∧ boundedAbove f) ∧
  (∃ g : ℝ → ℝ, constantlyDecreasing g ∧ boundedBelow g) :=
sorry

end NUMINAMATH_CALUDE_increasing_not_always_unbounded_and_decreasing_not_always_unbounded_l2400_240031


namespace NUMINAMATH_CALUDE_four_Y_three_equals_49_l2400_240025

-- Define the new operation Y
def Y (a b : ℝ) : ℝ := (a + b)^2

-- State the theorem
theorem four_Y_three_equals_49 : Y 4 3 = 49 := by
  sorry

end NUMINAMATH_CALUDE_four_Y_three_equals_49_l2400_240025


namespace NUMINAMATH_CALUDE_cos_shift_l2400_240085

theorem cos_shift (x : ℝ) : 
  Real.cos (x / 2 - π / 3) = Real.cos ((x - 2 * π / 3) / 2) := by sorry

end NUMINAMATH_CALUDE_cos_shift_l2400_240085


namespace NUMINAMATH_CALUDE_ratio_calculation_l2400_240078

theorem ratio_calculation (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.3 * P) 
  (hN : N = 0.5 * P) : 
  M / N = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l2400_240078


namespace NUMINAMATH_CALUDE_square_root_expression_value_l2400_240016

theorem square_root_expression_value :
  let x : ℝ := Real.sqrt 6 - Real.sqrt 2
  2 * x^2 + 4 * Real.sqrt 2 * x = 8 := by
sorry

end NUMINAMATH_CALUDE_square_root_expression_value_l2400_240016


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2400_240005

/-- The trajectory of the midpoint of a segment between a fixed point and a point on a circle -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ (px py : ℝ), px^2 + py^2 = 16 ∧ x = (px + 12) / 2 ∧ y = py / 2) → 
  (x - 6)^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2400_240005


namespace NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l2400_240068

/-- An angle is in the second quadrant if it's between 90° and 180° or between -270° and -180° --/
def is_second_quadrant_angle (α : Real) : Prop :=
  (90 < α ∧ α ≤ 180) ∨ (-270 < α ∧ α ≤ -180)

/-- An angle is obtuse if it's between 90° and 180° --/
def is_obtuse_angle (α : Real) : Prop :=
  90 < α ∧ α < 180

/-- Theorem stating that "α is a second quadrant angle" is a necessary but not sufficient condition for "α is an obtuse angle" --/
theorem second_quadrant_necessary_not_sufficient_for_obtuse :
  (∀ α : Real, is_obtuse_angle α → is_second_quadrant_angle α) ∧
  (∃ α : Real, is_second_quadrant_angle α ∧ ¬is_obtuse_angle α) :=
by sorry

end NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l2400_240068


namespace NUMINAMATH_CALUDE_book_profit_percentage_l2400_240061

/-- Calculates the profit percentage on the cost price for a book sale --/
theorem book_profit_percentage 
  (cost_price : ℝ) 
  (marked_price : ℝ) 
  (discount_rate : ℝ) 
  (h1 : cost_price = 47.50)
  (h2 : marked_price = 69.85)
  (h3 : discount_rate = 0.15) : 
  ∃ (profit_percentage : ℝ), 
    abs (profit_percentage - 24.99) < 0.01 ∧ 
    profit_percentage = (marked_price * (1 - discount_rate) - cost_price) / cost_price * 100 := by
  sorry


end NUMINAMATH_CALUDE_book_profit_percentage_l2400_240061


namespace NUMINAMATH_CALUDE_monday_water_usage_l2400_240074

/-- Represents the relationship between rainfall and water usage -/
structure RainfallWaterUsage where
  rainfall : ℝ
  water_used : ℝ

/-- The constant of inverse proportionality between rainfall and water usage -/
def inverse_proportionality_constant (day : RainfallWaterUsage) : ℝ :=
  day.rainfall * day.water_used

theorem monday_water_usage 
  (sunday : RainfallWaterUsage)
  (monday_rainfall : ℝ)
  (h_sunday_rainfall : sunday.rainfall = 3)
  (h_sunday_water : sunday.water_used = 10)
  (h_monday_rainfall : monday_rainfall = 5)
  (h_inverse_prop : ∀ (day1 day2 : RainfallWaterUsage), 
    inverse_proportionality_constant day1 = inverse_proportionality_constant day2) :
  ∃ (monday : RainfallWaterUsage), 
    monday.rainfall = monday_rainfall ∧ 
    monday.water_used = 6 :=
sorry

end NUMINAMATH_CALUDE_monday_water_usage_l2400_240074


namespace NUMINAMATH_CALUDE_pens_given_to_friends_l2400_240033

def initial_pens : ℕ := 56
def remaining_pens : ℕ := 34

theorem pens_given_to_friends :
  initial_pens - remaining_pens = 22 := by
  sorry

end NUMINAMATH_CALUDE_pens_given_to_friends_l2400_240033


namespace NUMINAMATH_CALUDE_problem_solution_l2400_240096

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f x < 0}

theorem problem_solution (m : ℝ) (α : ℝ) :
  (∀ x, determinant (x + m) 2 1 x < 0 ↔ x ∈ Set.Ioo (-1) 2) →
  m * Real.cos α + 2 * Real.sin α = 0 →
  m = -1 ∧ Real.tan (2 * α - Real.pi / 4) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2400_240096


namespace NUMINAMATH_CALUDE_greatest_a_value_l2400_240037

theorem greatest_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = 18) ∧ (a > 0) → a ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_greatest_a_value_l2400_240037


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2400_240017

theorem intersection_of_sets : 
  let A : Set ℤ := {-1, 0, 1, 2}
  let B : Set ℤ := {-2, 1, 2}
  A ∩ B = {1, 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2400_240017


namespace NUMINAMATH_CALUDE_enhanced_computer_price_difference_l2400_240026

/-- The price difference between an enhanced computer and a basic computer -/
def price_difference (total_basic : ℝ) (price_basic : ℝ) : ℝ :=
  let price_printer := total_basic - price_basic
  let price_enhanced := 6 * price_printer
  price_enhanced - price_basic

/-- Theorem stating the price difference between enhanced and basic computers -/
theorem enhanced_computer_price_difference :
  price_difference 2500 2000 = 500 := by
  sorry

end NUMINAMATH_CALUDE_enhanced_computer_price_difference_l2400_240026


namespace NUMINAMATH_CALUDE_vector_calculation_l2400_240000

/-- Given vectors AB and BC in 2D space, prove that -1/2 * AC equals the specified vector -/
theorem vector_calculation (AB BC : Fin 2 → ℝ) 
  (h1 : AB = ![3, 7])
  (h2 : BC = ![-2, 3]) :
  (-1/2 : ℝ) • (AB + BC) = ![-1/2, -5] := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l2400_240000


namespace NUMINAMATH_CALUDE_gala_trees_l2400_240076

/-- Represents the orchard with Fuji and Gala apple trees -/
structure Orchard where
  total : ℕ
  fuji : ℕ
  gala : ℕ
  crossPollinated : ℕ

/-- Conditions of the orchard -/
def validOrchard (o : Orchard) : Prop :=
  o.crossPollinated = o.total / 10 ∧
  o.fuji + o.crossPollinated = 170 ∧
  o.fuji = 3 * o.total / 4 ∧
  o.total = o.fuji + o.gala + o.crossPollinated

theorem gala_trees (o : Orchard) (h : validOrchard o) : o.gala = 50 := by
  sorry

end NUMINAMATH_CALUDE_gala_trees_l2400_240076


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2400_240048

-- Define the regular quadrilateral pyramid
structure RegularQuadPyramid where
  a : ℝ  -- base edge
  h : ℝ  -- height
  lateral_edge : ℝ
  lateral_edge_eq : lateral_edge = (5/2) * a

-- Define the cylinder
structure Cylinder (P : RegularQuadPyramid) where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cylinder

-- Theorem statement
theorem cylinder_lateral_surface_area 
  (P : RegularQuadPyramid) 
  (C : Cylinder P) :
  ∃ (S : ℝ), S = (π * P.a^2 * Real.sqrt 46) / 9 ∧ 
  S = 2 * π * C.r * C.h :=
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2400_240048


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2400_240073

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i^2 = -1 → Complex.im (2 / (2 + i)) = -2/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2400_240073


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2400_240063

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 4 * y^2 - 5 * y - 21 = (4 * y + a) * (y + b)) →
  a - b = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2400_240063


namespace NUMINAMATH_CALUDE_water_depth_in_cistern_l2400_240069

theorem water_depth_in_cistern (length width total_wet_area : ℝ) 
  (h_length : length = 7)
  (h_width : width = 4)
  (h_total_wet_area : total_wet_area = 55.5)
  : ∃ depth : ℝ, 
    depth = 1.25 ∧ 
    total_wet_area = length * width + 2 * length * depth + 2 * width * depth :=
by sorry

end NUMINAMATH_CALUDE_water_depth_in_cistern_l2400_240069


namespace NUMINAMATH_CALUDE_line_properties_l2400_240038

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := x - m * y + 2 = 0

-- Theorem statement
theorem line_properties (m : ℝ) :
  (∀ y, line_equation (-2) y m) ∧
  (∃ x, x ≠ 0 ∧ line_equation x 0 m) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l2400_240038


namespace NUMINAMATH_CALUDE_prime_power_sum_implies_power_of_three_l2400_240039

theorem prime_power_sum_implies_power_of_three (n : ℕ) :
  Nat.Prime (1 + 2^n + 4^n) → ∃ k : ℕ, n = 3^k :=
by sorry

end NUMINAMATH_CALUDE_prime_power_sum_implies_power_of_three_l2400_240039


namespace NUMINAMATH_CALUDE_special_integers_property_l2400_240060

/-- A function that reverses the hundreds and units digits of a three-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  units * 100 + tens * 10 + hundreds

/-- The theorem stating the property of the 90 special integers -/
theorem special_integers_property :
  ∃ (S : Finset ℕ), 
    Finset.card S = 90 ∧ 
    (∀ n ∈ S, 100 < n ∧ n < 1100) ∧
    (∀ n ∈ S, reverseDigits n = n + 99) := by
  sorry

#check special_integers_property

end NUMINAMATH_CALUDE_special_integers_property_l2400_240060


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2400_240066

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2400_240066


namespace NUMINAMATH_CALUDE_tan_seven_pi_sixths_l2400_240034

theorem tan_seven_pi_sixths : Real.tan (7 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_sixths_l2400_240034


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_l2400_240071

theorem cube_volume_from_diagonal (diagonal : ℝ) (h : diagonal = 6 * Real.sqrt 2) :
  ∃ (side : ℝ), side > 0 ∧ side^3 = 48 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_l2400_240071


namespace NUMINAMATH_CALUDE_cordelia_hair_coloring_time_l2400_240008

/-- Represents the hair coloring process -/
structure HairColoring where
  bleaching_time : ℝ
  dyeing_time : ℝ

/-- The total time for the hair coloring process -/
def total_time (hc : HairColoring) : ℝ :=
  hc.bleaching_time + hc.dyeing_time

/-- Theorem stating the total time for Cordelia's hair coloring process -/
theorem cordelia_hair_coloring_time :
  ∃ (hc : HairColoring),
    hc.bleaching_time = 3 ∧
    hc.dyeing_time = 2 * hc.bleaching_time ∧
    total_time hc = 9 := by
  sorry

end NUMINAMATH_CALUDE_cordelia_hair_coloring_time_l2400_240008


namespace NUMINAMATH_CALUDE_extreme_value_at_zero_tangent_line_equation_decreasing_condition_l2400_240024

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (3 * x^2 + a * x) / Real.exp x

def f_prime (a : ℝ) (x : ℝ) : ℝ := (-3 * x^2 + (6 - a) * x + a) / Real.exp x

theorem extreme_value_at_zero (a : ℝ) :
  f_prime a 0 = 0 → a = 0 := by sorry

theorem tangent_line_equation (a : ℝ) :
  a = 0 → ∀ x y : ℝ, y = f a x → (3 * x - Real.exp 1 * y = 0 ↔ x = 1) := by sorry

theorem decreasing_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 3 → f_prime a x ≤ 0) ↔ a ≥ -9/2 := by sorry

end NUMINAMATH_CALUDE_extreme_value_at_zero_tangent_line_equation_decreasing_condition_l2400_240024


namespace NUMINAMATH_CALUDE_triangle_side_difference_triangle_side_difference_is_12_l2400_240020

theorem triangle_side_difference : ℕ → Prop :=
  fun d =>
    ∃ (x_min x_max : ℤ),
      (∀ x : ℤ, (x > x_min ∧ x < x_max) → (x + 7 > 10 ∧ x + 10 > 7 ∧ 7 + 10 > x)) ∧
      (∀ x : ℤ, (x ≤ x_min ∨ x ≥ x_max) → ¬(x + 7 > 10 ∧ x + 10 > 7 ∧ 7 + 10 > x)) ∧
      (x_max - x_min = d + 1)

theorem triangle_side_difference_is_12 : triangle_side_difference 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_difference_triangle_side_difference_is_12_l2400_240020


namespace NUMINAMATH_CALUDE_star_properties_l2400_240070

-- Define the * operation for rational numbers
def star (a b : ℚ) : ℚ := (a + b) - abs (b - a)

-- Theorem statement
theorem star_properties :
  (star (-3) 2 = -6) ∧ (star (star 4 3) (-5) = -10) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l2400_240070


namespace NUMINAMATH_CALUDE_loss_percent_calculation_l2400_240095

theorem loss_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 600)
  (h2 : selling_price = 450) :
  (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_loss_percent_calculation_l2400_240095


namespace NUMINAMATH_CALUDE_greg_marbles_l2400_240051

/-- The number of marbles Adam has -/
def adam_marbles : ℕ := 29

/-- The number of additional marbles Greg has compared to Adam -/
def greg_additional_marbles : ℕ := 14

/-- Theorem: Greg has 43 marbles -/
theorem greg_marbles : adam_marbles + greg_additional_marbles = 43 := by
  sorry

end NUMINAMATH_CALUDE_greg_marbles_l2400_240051


namespace NUMINAMATH_CALUDE_minimum_buses_required_l2400_240091

theorem minimum_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 325) (h2 : bus_capacity = 45) : 
  ∃ (n : ℕ), n * bus_capacity ≥ total_students ∧ ∀ (m : ℕ), m * bus_capacity ≥ total_students → m ≥ n ∧ n = 8 :=
by sorry

end NUMINAMATH_CALUDE_minimum_buses_required_l2400_240091


namespace NUMINAMATH_CALUDE_min_value_of_ab_min_value_is_6_plus_4sqrt2_l2400_240036

theorem min_value_of_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a * b + 2 = 2 * (a + b)) :
  ∀ x y : ℝ, x > 1 → y > 1 → x * y + 2 = 2 * (x + y) → a * b ≤ x * y :=
by sorry

theorem min_value_is_6_plus_4sqrt2 (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a * b + 2 = 2 * (a + b)) :
  a * b = 6 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_ab_min_value_is_6_plus_4sqrt2_l2400_240036


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_distance_l2400_240084

/-- The distance between intersection points of an ellipse and a parabola -/
theorem ellipse_parabola_intersection_distance : 
  ∀ (ellipse : (ℝ × ℝ) → Prop) (parabola : (ℝ × ℝ) → Prop) 
    (focus : ℝ × ℝ) (directrix : ℝ → ℝ × ℝ),
  (∀ x y, ellipse (x, y) ↔ x^2 / 16 + y^2 / 36 = 1) →
  (∃ c, ∀ x, directrix x = (c, x)) →
  (∃ x₁ y₁ x₂ y₂, ellipse (x₁, y₁) ∧ parabola (x₁, y₁) ∧
                   ellipse (x₂, y₂) ∧ parabola (x₂, y₂) ∧
                   (x₁, y₁) ≠ (x₂, y₂)) →
  (∃ x y, focus = (x, y) ∧ parabola (x, y)) →
  (∃ x y, focus = (x, y) ∧ ellipse (x, y)) →
  ∃ x₁ y₁ x₂ y₂, 
    ellipse (x₁, y₁) ∧ parabola (x₁, y₁) ∧
    ellipse (x₂, y₂) ∧ parabola (x₂, y₂) ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 24 * Real.sqrt 5 / Real.sqrt (9 + 5 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_distance_l2400_240084


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2400_240006

theorem inequality_solution_set (x : ℝ) : 
  (2 * x - 4 < 6) ↔ (x < 5) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2400_240006


namespace NUMINAMATH_CALUDE_pablo_book_pages_l2400_240010

/-- The number of books Pablo reads -/
def num_books : ℕ := 12

/-- The total amount of money Pablo earned in cents -/
def total_earned : ℕ := 1800

/-- The number of pages in each book -/
def pages_per_book : ℕ := total_earned / num_books

theorem pablo_book_pages :
  pages_per_book = 150 :=
by sorry

end NUMINAMATH_CALUDE_pablo_book_pages_l2400_240010


namespace NUMINAMATH_CALUDE_k_bound_l2400_240047

/-- A sequence a_n defined as n^2 - kn for positive integers n -/
def a (k : ℝ) (n : ℕ) : ℝ := n^2 - k * n

/-- The property that a sequence is monotonically increasing -/
def MonotonicallyIncreasing (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, f (n + 1) > f n

/-- Theorem: If the sequence a_n is monotonically increasing, then k < 3 -/
theorem k_bound (k : ℝ) (h : MonotonicallyIncreasing (a k)) : k < 3 := by
  sorry

end NUMINAMATH_CALUDE_k_bound_l2400_240047


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2400_240072

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) ∧ 
  (b^3 - 2*b^2 + 3*b - 4 = 0) ∧ 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2400_240072


namespace NUMINAMATH_CALUDE_second_student_speed_l2400_240042

/-- Given two students walking in opposite directions, this theorem proves
    the speed of the second student given the conditions of the problem. -/
theorem second_student_speed
  (time : ℝ)
  (distance : ℝ)
  (speed1 : ℝ)
  (h1 : time = 4)
  (h2 : distance = 60)
  (h3 : speed1 = 6)
  (h4 : distance = (speed1 + speed2) * time) :
  speed2 = 9 :=
by
  sorry

#check second_student_speed

end NUMINAMATH_CALUDE_second_student_speed_l2400_240042


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2400_240092

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Number of terms in an arithmetic sequence -/
def num_terms (a₁ : ℚ) (d : ℚ) (l : ℚ) : ℕ :=
  Nat.floor ((l - a₁) / d + 1)

theorem arithmetic_sequence_ratio :
  let n₁ := num_terms 4 2 40
  let n₂ := num_terms 5 5 75
  let sum₁ := arithmetic_sum 4 2 n₁
  let sum₂ := arithmetic_sum 5 5 n₂
  sum₁ / sum₂ = 209 / 300 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2400_240092


namespace NUMINAMATH_CALUDE_average_rate_of_change_l2400_240015

def f (x : ℝ) : ℝ := 2 * x - 1

theorem average_rate_of_change (a b : ℝ) (h : a < b) :
  (f b - f a) / (b - a) = 2 :=
by sorry

end NUMINAMATH_CALUDE_average_rate_of_change_l2400_240015


namespace NUMINAMATH_CALUDE_one_chief_physician_probability_l2400_240012

theorem one_chief_physician_probability 
  (total_male_doctors : ℕ) 
  (total_female_doctors : ℕ) 
  (male_chief_physicians : ℕ) 
  (female_chief_physicians : ℕ) 
  (selected_male_doctors : ℕ) 
  (selected_female_doctors : ℕ) :
  total_male_doctors = 4 →
  total_female_doctors = 5 →
  male_chief_physicians = 1 →
  female_chief_physicians = 1 →
  selected_male_doctors = 3 →
  selected_female_doctors = 2 →
  (Nat.choose total_male_doctors selected_male_doctors *
   Nat.choose total_female_doctors selected_female_doctors -
   Nat.choose (total_male_doctors - male_chief_physicians) selected_male_doctors *
   Nat.choose (total_female_doctors - female_chief_physicians) selected_female_doctors -
   Nat.choose (total_male_doctors - male_chief_physicians) (selected_male_doctors - 1) *
   Nat.choose (total_female_doctors - female_chief_physicians) selected_female_doctors -
   Nat.choose total_male_doctors selected_male_doctors *
   Nat.choose (total_female_doctors - female_chief_physicians) (selected_female_doctors - 1)) /
  (Nat.choose total_male_doctors selected_male_doctors *
   Nat.choose total_female_doctors selected_female_doctors) = 6 / 17 := by
  sorry

end NUMINAMATH_CALUDE_one_chief_physician_probability_l2400_240012


namespace NUMINAMATH_CALUDE_determinant_2x2_matrix_l2400_240027

theorem determinant_2x2_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, 3; -1, 2]
  Matrix.det A = 17 := by
sorry

end NUMINAMATH_CALUDE_determinant_2x2_matrix_l2400_240027


namespace NUMINAMATH_CALUDE_x_squared_plus_7x_plus_12_bounds_l2400_240045

theorem x_squared_plus_7x_plus_12_bounds (x : ℝ) (h : x^2 - 7*x + 12 < 0) :
  42 < x^2 + 7*x + 12 ∧ x^2 + 7*x + 12 < 56 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_7x_plus_12_bounds_l2400_240045


namespace NUMINAMATH_CALUDE_cube_in_pyramid_volume_l2400_240052

/-- A pyramid with a square base and isosceles triangular lateral faces -/
structure Pyramid where
  base_side : ℝ
  lateral_height : ℝ

/-- A cube placed inside the pyramid -/
structure InsideCube where
  side_length : ℝ

/-- The volume of a cube -/
def cube_volume (c : InsideCube) : ℝ := c.side_length ^ 3

theorem cube_in_pyramid_volume 
  (p : Pyramid) 
  (c : InsideCube) 
  (h1 : p.base_side = 2) 
  (h2 : p.lateral_height = 4) 
  (h3 : c.side_length * 2 = p.lateral_height) : 
  cube_volume c = 8 := by
  sorry

#check cube_in_pyramid_volume

end NUMINAMATH_CALUDE_cube_in_pyramid_volume_l2400_240052


namespace NUMINAMATH_CALUDE_additional_water_for_two_tanks_l2400_240056

/-- Calculates the additional water needed to fill two tanks with equal capacity -/
theorem additional_water_for_two_tanks
  (capacity : ℝ)  -- Capacity of each tank
  (filled1 : ℝ)   -- Amount of water in the first tank
  (filled2 : ℝ)   -- Amount of water in the second tank
  (h1 : filled1 = 300)  -- First tank has 300 liters
  (h2 : filled2 = 450)  -- Second tank has 450 liters
  (h3 : filled2 / capacity = 0.45)  -- Second tank is 45% filled
  : capacity - filled1 + capacity - filled2 = 1250 :=
by sorry

end NUMINAMATH_CALUDE_additional_water_for_two_tanks_l2400_240056


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2400_240054

/-- Given that x and y are always positive, x^3 and y vary inversely, 
    and y = 8 when x = 2, prove that x = 1 / (13.5^(1/3)) when y = 1728 -/
theorem inverse_variation_problem (x y : ℝ) 
  (h_positive : x > 0 ∧ y > 0)
  (h_inverse : ∃ k : ℝ, k > 0 ∧ ∀ x y, x^3 * y = k)
  (h_initial : 2^3 * 8 = (h_inverse.choose)^3)
  (h_final : y = 1728) :
  x = 1 / (13.5^(1/3)) :=
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2400_240054


namespace NUMINAMATH_CALUDE_mod_equivalence_2023_l2400_240065

theorem mod_equivalence_2023 :
  ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -2023 [ZMOD 8] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_2023_l2400_240065


namespace NUMINAMATH_CALUDE_chemical_equilibrium_and_precipitate_l2400_240029

-- Define the chemical reaction parameters
def initial_BaCl2_concentration : ℝ := 10
def equilibrium_constant : ℝ := 5 * 10^6
def initial_volume : ℝ := 1

-- Define the molar mass of BaSO4
def molar_mass_BaSO4 : ℝ := 233.40

-- Define the theorem
theorem chemical_equilibrium_and_precipitate :
  ∃ (equilibrium_BaSO4_concentration : ℝ) (mass_BaSO4_precipitate : ℝ),
    (abs (equilibrium_BaSO4_concentration - 10) < 0.01) ∧
    (abs (mass_BaSO4_precipitate - 2334) < 0.1) :=
sorry

end NUMINAMATH_CALUDE_chemical_equilibrium_and_precipitate_l2400_240029


namespace NUMINAMATH_CALUDE_T4_championship_probability_l2400_240098

/-- Represents a team in the playoffs -/
inductive Team : Type
| T1 : Team
| T2 : Team
| T3 : Team
| T4 : Team

/-- The probability of team i winning against team j -/
def winProbability (i j : Team) : ℚ :=
  match i, j with
  | Team.T1, Team.T2 => 1/3
  | Team.T1, Team.T3 => 1/4
  | Team.T1, Team.T4 => 1/5
  | Team.T2, Team.T1 => 2/3
  | Team.T2, Team.T3 => 2/5
  | Team.T2, Team.T4 => 1/3
  | Team.T3, Team.T1 => 3/4
  | Team.T3, Team.T2 => 3/5
  | Team.T3, Team.T4 => 3/7
  | Team.T4, Team.T1 => 4/5
  | Team.T4, Team.T2 => 2/3
  | Team.T4, Team.T3 => 4/7
  | _, _ => 1/2  -- This case should never occur in our scenario

/-- The probability of T4 winning the championship -/
def T4ChampionshipProbability : ℚ :=
  (winProbability Team.T4 Team.T1) * 
  ((winProbability Team.T3 Team.T2) * (winProbability Team.T4 Team.T3) +
   (winProbability Team.T2 Team.T3) * (winProbability Team.T4 Team.T2))

theorem T4_championship_probability :
  T4ChampionshipProbability = 256/525 := by
  sorry

#eval T4ChampionshipProbability

end NUMINAMATH_CALUDE_T4_championship_probability_l2400_240098


namespace NUMINAMATH_CALUDE_larger_number_proof_l2400_240002

theorem larger_number_proof (x y : ℝ) (h1 : y > x) (h2 : 4 * y = 5 * x) (h3 : y - x = 10) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2400_240002


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2400_240090

theorem complex_equation_solution (z : ℂ) : z * (2 - I) = 11 + 7 * I → z = 3 + 5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2400_240090


namespace NUMINAMATH_CALUDE_consecutive_color_probability_value_l2400_240081

/-- Represents the number of green chips in the bag -/
def green_chips : ℕ := 4

/-- Represents the number of orange chips in the bag -/
def orange_chips : ℕ := 3

/-- Represents the number of blue chips in the bag -/
def blue_chips : ℕ := 5

/-- Represents the total number of chips in the bag -/
def total_chips : ℕ := green_chips + orange_chips + blue_chips

/-- The probability of drawing all chips such that each color group is drawn consecutively -/
def consecutive_color_probability : ℚ :=
  (Nat.factorial 3 * Nat.factorial green_chips * Nat.factorial orange_chips * Nat.factorial blue_chips) /
  Nat.factorial total_chips

theorem consecutive_color_probability_value :
  consecutive_color_probability = 1 / 4620 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_color_probability_value_l2400_240081


namespace NUMINAMATH_CALUDE_combined_fuel_efficiency_l2400_240023

/-- The combined fuel efficiency of three cars -/
theorem combined_fuel_efficiency 
  (m : ℝ) -- distance driven by each car
  (h1 : m > 0) -- ensure distance is positive
  (efficiency1 efficiency2 efficiency3 : ℝ) -- individual car efficiencies
  (h2 : efficiency1 = 35) -- Ray's car efficiency
  (h3 : efficiency2 = 25) -- Tom's car efficiency
  (h4 : efficiency3 = 20) -- Alice's car efficiency
  : (3 * m) / (m / efficiency1 + m / efficiency2 + m / efficiency3) = 2100 / 83 := by
  sorry

#eval (2100 : ℚ) / 83 -- To show the decimal approximation

end NUMINAMATH_CALUDE_combined_fuel_efficiency_l2400_240023


namespace NUMINAMATH_CALUDE_distance_on_line_l2400_240030

/-- The distance between two points on a line y = mx + k -/
theorem distance_on_line (m k a b c d : ℝ) 
  (h1 : b = m * a + k) 
  (h2 : d = m * c + k) : 
  Real.sqrt ((a - c)^2 + (b - d)^2) = |a - c| * Real.sqrt (1 + m^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l2400_240030


namespace NUMINAMATH_CALUDE_no_repeating_subsequence_l2400_240044

/-- Count the number of 1's in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Define the sequence a_n based on the parity of the number of 1's in the binary representation -/
def a (n : ℕ) : ℕ := 
  if countOnes n % 2 = 0 then 0 else 1

/-- The main theorem stating that there are no positive integers k and m satisfying the condition -/
theorem no_repeating_subsequence : 
  ¬ ∃ (k m : ℕ+), ∀ (j : ℕ), j < m → 
    a (k + j) = a (k + m + j) ∧ a (k + j) = a (k + 2*m + j) := by
  sorry

end NUMINAMATH_CALUDE_no_repeating_subsequence_l2400_240044


namespace NUMINAMATH_CALUDE_grocery_cost_l2400_240009

/-- The cost of groceries problem -/
theorem grocery_cost (mango_cost rice_cost flour_cost : ℝ) : 
  (10 * mango_cost = 24 * rice_cost) →
  (flour_cost = 2 * rice_cost) →
  (flour_cost = 24) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 271.2) := by
sorry

end NUMINAMATH_CALUDE_grocery_cost_l2400_240009


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2400_240057

theorem complex_equation_solution (i z : ℂ) (h1 : i * i = -1) (h2 : i * z = 1 - i) :
  z = -i - 1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2400_240057


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_of_primes_l2400_240064

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem least_positive_linear_combination_of_primes :
  ∃ (x y z w : ℕ), 
    is_prime x ∧ is_prime y ∧ is_prime z ∧ is_prime w ∧
    24*x + 16*y - 7*z + 5*w = 13 ∧
    (∀ (a b c d : ℕ), is_prime a → is_prime b → is_prime c → is_prime d →
      24*a + 16*b - 7*c + 5*d > 0 → 24*a + 16*b - 7*c + 5*d ≥ 13) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_of_primes_l2400_240064


namespace NUMINAMATH_CALUDE_solution_interval_l2400_240088

theorem solution_interval (x₀ : ℝ) : 
  (1/2:ℝ)^x₀ = x₀^(1/3) → 1/3 < x₀ ∧ x₀ < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l2400_240088


namespace NUMINAMATH_CALUDE_function_with_given_derivative_l2400_240094

/-- Given a differentiable function f on ℝ with f'(x) = 1 + sin x,
    prove that there exists a constant C such that f(x) = x - cos x + C. -/
theorem function_with_given_derivative
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) (hf' : ∀ x, deriv f x = 1 + Real.sin x) :
  ∃ C, ∀ x, f x = x - Real.cos x + C := by
  sorry

end NUMINAMATH_CALUDE_function_with_given_derivative_l2400_240094


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2400_240035

/-- The parabola function f(x) = x^2 - 10x + 20 --/
def f (x : ℝ) : ℝ := x^2 - 10*x + 20

/-- A square inscribed between a parabola and the x-axis --/
structure InscribedSquare where
  center : ℝ × ℝ
  side_length : ℝ
  h1 : center.1 = 5 -- The x-coordinate of the center is at the vertex of the parabola
  h2 : center.2 = side_length / 2 -- The y-coordinate of the center is half the side length
  h3 : f (center.1 + side_length / 2) = side_length -- The top right corner lies on the parabola

/-- The theorem stating that the area of the inscribed square is 400 --/
theorem inscribed_square_area (s : InscribedSquare) : s.side_length^2 = 400 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_square_area_l2400_240035


namespace NUMINAMATH_CALUDE_arlettes_age_l2400_240014

/-- Given the ages of three people: Omi, Kimiko, and Arlette, prove that Arlette's age is 21 years. -/
theorem arlettes_age (omi kimiko arlette : ℕ) : 
  omi = 2 * kimiko →  -- Omi's age is twice Kimiko's age
  kimiko = 28 →       -- Kimiko's age is 28 years
  (omi + kimiko + arlette) / 3 = 35 →  -- The average age of the three is 35 years
  arlette = 21 := by
sorry

end NUMINAMATH_CALUDE_arlettes_age_l2400_240014


namespace NUMINAMATH_CALUDE_number_between_fractions_l2400_240097

theorem number_between_fractions : 0.2012 > (1 : ℚ) / 5 ∧ 0.2012 < (1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_number_between_fractions_l2400_240097


namespace NUMINAMATH_CALUDE_square_value_l2400_240082

theorem square_value (square : ℚ) : 8/12 = square/3 → square = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l2400_240082


namespace NUMINAMATH_CALUDE_extremum_values_of_e_l2400_240013

theorem extremum_values_of_e (a b c d e : ℝ) 
  (h1 : 3*a + 2*b - c + 4*d + Real.sqrt 133 * e = Real.sqrt 133)
  (h2 : 2*a^2 + 3*b^2 + 3*c^2 + d^2 + 6*e^2 = 60) :
  ∃ (e_min e_max : ℝ), 
    e_min = (1 - Real.sqrt 19) / 2 ∧ 
    e_max = (1 + Real.sqrt 19) / 2 ∧
    e_min ≤ e ∧ e ≤ e_max ∧
    (e = e_min ∨ e = e_max → 
      ∃ (k : ℝ), a = 3*k/8 ∧ b = k/6 ∧ c = -k/12 ∧ d = k) :=
by sorry

end NUMINAMATH_CALUDE_extremum_values_of_e_l2400_240013


namespace NUMINAMATH_CALUDE_daves_old_cards_l2400_240093

/-- Given Dave's baseball card organization, prove the number of old cards --/
theorem daves_old_cards
  (cards_per_page : ℕ)
  (new_cards : ℕ)
  (pages_used : ℕ)
  (h1 : cards_per_page = 8)
  (h2 : new_cards = 3)
  (h3 : pages_used = 2) :
  pages_used * cards_per_page - new_cards = 13 := by
  sorry

end NUMINAMATH_CALUDE_daves_old_cards_l2400_240093


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2400_240062

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c ≥ a ∧ c ≥ b) : 
  a + b ≤ c * Real.sqrt 2 ∧ (a + b = c * Real.sqrt 2 ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2400_240062


namespace NUMINAMATH_CALUDE_area_of_triangle_OAB_is_one_l2400_240077

/-- Given vector a and b in ℝ², prove that the area of triangle OAB is 1 -/
theorem area_of_triangle_OAB_is_one 
  (a b : ℝ × ℝ)
  (h_a : a = (-1/2, Real.sqrt 3/2))
  (h_OA : (a.1 - b.1, a.2 - b.2) = (a.1 - b.1, a.2 - b.2))
  (h_OB : (a.1 + b.1, a.2 + b.2) = (a.1 + b.1, a.2 + b.2))
  (h_isosceles : ‖(a.1 - b.1, a.2 - b.2)‖ = ‖(a.1 + b.1, a.2 + b.2)‖)
  (h_right_angle : (a.1 - b.1, a.2 - b.2) • (a.1 + b.1, a.2 + b.2) = 0) :
  (1/2) * ‖(a.1 - b.1, a.2 - b.2)‖ * ‖(a.1 + b.1, a.2 + b.2)‖ = 1 :=
by sorry


end NUMINAMATH_CALUDE_area_of_triangle_OAB_is_one_l2400_240077


namespace NUMINAMATH_CALUDE_rental_fee_minimization_l2400_240053

/-- Represents the total number of buses to be rented -/
def total_buses : ℕ := 6

/-- Represents the rental fee for a Type A bus -/
def type_a_fee : ℕ := 450

/-- Represents the rental fee for a Type B bus -/
def type_b_fee : ℕ := 300

/-- Calculates the total rental fee based on the number of Type B buses -/
def rental_fee (x : ℕ) : ℕ := total_buses * type_a_fee - (type_a_fee - type_b_fee) * x

theorem rental_fee_minimization :
  ∀ x : ℕ, 0 < x → x < total_buses → x < total_buses - x →
  (∀ y : ℕ, 0 < y → y < total_buses → y < total_buses - y →
    rental_fee x ≤ rental_fee y) →
  x = 2 ∧ rental_fee x = 2400 := by sorry

end NUMINAMATH_CALUDE_rental_fee_minimization_l2400_240053


namespace NUMINAMATH_CALUDE_max_salary_is_368000_l2400_240018

/-- Represents a soccer team with salary constraints -/
structure SoccerTeam where
  num_players : ℕ
  min_salary : ℕ
  total_salary_cap : ℕ

/-- Calculates the maximum possible salary for a single player in a soccer team -/
def max_player_salary (team : SoccerTeam) : ℕ :=
  team.total_salary_cap - (team.num_players - 1) * team.min_salary

/-- Theorem: The maximum possible salary for a single player in the given conditions is 368000 -/
theorem max_salary_is_368000 :
  let team : SoccerTeam := ⟨25, 18000, 800000⟩
  max_player_salary team = 368000 := by
  sorry

#eval max_player_salary ⟨25, 18000, 800000⟩

end NUMINAMATH_CALUDE_max_salary_is_368000_l2400_240018


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l2400_240067

-- Define the function f(x) = |x+1|
def f (x : ℝ) : ℝ := |x + 1|

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  ∀ x y : ℝ, x ≥ -1 → y ≥ -1 → x ≤ y → f x ≤ f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l2400_240067


namespace NUMINAMATH_CALUDE_children_playing_both_sports_l2400_240041

/-- Given a class of children with the following properties:
  * The total number of children is 38
  * 19 children play tennis
  * 21 children play squash
  * 10 children play neither sport
  Then, the number of children who play both sports is 12 -/
theorem children_playing_both_sports
  (total : ℕ) (tennis : ℕ) (squash : ℕ) (neither : ℕ) (both : ℕ)
  (h1 : total = 38)
  (h2 : tennis = 19)
  (h3 : squash = 21)
  (h4 : neither = 10)
  (h5 : total = tennis + squash - both + neither) :
  both = 12 := by
sorry

end NUMINAMATH_CALUDE_children_playing_both_sports_l2400_240041


namespace NUMINAMATH_CALUDE_proposition_p_equivalence_l2400_240032

theorem proposition_p_equivalence (m : ℝ) :
  (∃ x : ℝ, x^2 + 2*x - m - 1 < 0) ↔ m > -2 := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_equivalence_l2400_240032


namespace NUMINAMATH_CALUDE_equation_solution_l2400_240086

theorem equation_solution (y : ℝ) : 
  (y / 6) / 3 = 9 / (y / 3) → y = 3 * Real.sqrt 54 ∨ y = -3 * Real.sqrt 54 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2400_240086


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2400_240089

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt x + Real.sqrt (x + 4) = 8 → x = 225 / 16 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2400_240089


namespace NUMINAMATH_CALUDE_function_properties_l2400_240087

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -Real.sqrt a / (a^x + Real.sqrt a)

theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, f a x + f a (1 - x) = -1) ∧
  (f a (-2) + f a (-1) + f a 0 + f a 1 + f a 2 + f a 3 = -3) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2400_240087


namespace NUMINAMATH_CALUDE_number_of_refills_l2400_240049

/-- Proves that the number of refills is 4 given the total spent and cost per refill -/
theorem number_of_refills (total_spent : ℕ) (cost_per_refill : ℕ) 
  (h1 : total_spent = 40) 
  (h2 : cost_per_refill = 10) : 
  total_spent / cost_per_refill = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_refills_l2400_240049


namespace NUMINAMATH_CALUDE_factorial_division_l2400_240043

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l2400_240043


namespace NUMINAMATH_CALUDE_unique_ambiguous_product_l2400_240004

def numbers : Finset Nat := {1, 2, 3, 4, 5, 6, 7}

def is_valid_product (p : Nat) : Prop :=
  ∃ (s : Finset Nat), s ⊆ numbers ∧ s.card = 5 ∧ s.prod id = p

def parity_ambiguous (p : Nat) : Prop :=
  ∃ (s1 s2 : Finset Nat), s1 ≠ s2 ∧
    s1 ⊆ numbers ∧ s2 ⊆ numbers ∧
    s1.card = 5 ∧ s2.card = 5 ∧
    s1.prod id = p ∧ s2.prod id = p ∧
    s1.sum id % 2 ≠ s2.sum id % 2

theorem unique_ambiguous_product :
  ∃! p, is_valid_product p ∧ parity_ambiguous p ∧ p = 420 := by sorry

end NUMINAMATH_CALUDE_unique_ambiguous_product_l2400_240004


namespace NUMINAMATH_CALUDE_equidistant_line_equations_l2400_240001

/-- A line passing through (1, 2) and equidistant from (0, 0) and (3, 1) -/
structure EquidistantLine where
  -- Coefficients of the line equation ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through (1, 2)
  passes_through : a + 2 * b + c = 0
  -- The line is equidistant from (0, 0) and (3, 1)
  equidistant : (c^2) / (a^2 + b^2) = (3*a + b + c)^2 / (a^2 + b^2)

/-- Theorem stating the two possible equations of the equidistant line -/
theorem equidistant_line_equations : 
  ∀ (l : EquidistantLine), (l.a = 1 ∧ l.b = -3 ∧ l.c = 5) ∨ (l.a = 3 ∧ l.b = 1 ∧ l.c = -5) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_line_equations_l2400_240001


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2400_240075

theorem greatest_of_three_consecutive_integers (x y z : ℤ) : 
  (y = x + 1) → (z = y + 1) → (x + y + z = 39) → max x (max y z) = 14 := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2400_240075


namespace NUMINAMATH_CALUDE_log_product_equation_l2400_240059

theorem log_product_equation (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 4 → x = 2401 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equation_l2400_240059


namespace NUMINAMATH_CALUDE_bus_ride_cost_l2400_240079

-- Define the cost of bus and train rides
def bus_cost : ℝ := sorry
def train_cost : ℝ := sorry

-- State the theorem
theorem bus_ride_cost :
  (train_cost = bus_cost + 6.85) →
  (train_cost + bus_cost = 9.65) →
  (bus_cost = 1.40) := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l2400_240079


namespace NUMINAMATH_CALUDE_scores_mode_l2400_240021

def scores : List ℕ := [61, 62, 71, 78, 85, 85, 92, 96]

def mode (l : List ℕ) : ℕ := sorry

theorem scores_mode : mode scores = 85 := by sorry

end NUMINAMATH_CALUDE_scores_mode_l2400_240021


namespace NUMINAMATH_CALUDE_pages_copied_for_30_dollars_l2400_240046

/-- Given a cost per page in cents and a budget in dollars, 
    calculate the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Theorem: With a cost of 3 cents per page and a budget of $30, 
    the maximum number of pages that can be copied is 1000. -/
theorem pages_copied_for_30_dollars : 
  max_pages_copied 3 30 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_for_30_dollars_l2400_240046


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2400_240080

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2400_240080


namespace NUMINAMATH_CALUDE_blanket_average_price_l2400_240055

/-- Given the following conditions:
    - A man purchased 8 blankets in total
    - 1 blanket costs Rs. 100
    - 5 blankets cost Rs. 150 each
    - 2 blankets cost Rs. 650 in total
    Prove that the average price of all blankets is Rs. 187.50 -/
theorem blanket_average_price :
  let total_blankets : ℕ := 8
  let price_of_one : ℕ := 100
  let price_of_five : ℕ := 150
  let price_of_two : ℕ := 650
  let total_cost : ℕ := price_of_one + 5 * price_of_five + price_of_two
  (total_cost : ℚ) / total_blankets = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_blanket_average_price_l2400_240055


namespace NUMINAMATH_CALUDE_car_speed_problem_l2400_240099

theorem car_speed_problem (D : ℝ) (V : ℝ) : 
  D > 0 →
  (D / ((D/3)/60 + (D/3)/24 + (D/3)/V)) = 37.89473684210527 →
  V = 48 := by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2400_240099


namespace NUMINAMATH_CALUDE_consecutive_non_primes_l2400_240019

theorem consecutive_non_primes (n : ℕ) : ∃ (k : ℕ), ∀ (i : ℕ), i < n → ¬ Nat.Prime (k + i) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_primes_l2400_240019


namespace NUMINAMATH_CALUDE_inequality_proof_l2400_240028

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 1) : 
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2400_240028
