import Mathlib

namespace NUMINAMATH_CALUDE_m_not_greater_than_one_l2322_232284

theorem m_not_greater_than_one (m : ℝ) (h : |m - 1| + m = 1) : m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_m_not_greater_than_one_l2322_232284


namespace NUMINAMATH_CALUDE_circle_radius_l2322_232213

theorem circle_radius (A C : ℝ) (h : A / C = 25) : 
  ∃ r : ℝ, r > 0 ∧ A = π * r^2 ∧ C = 2 * π * r ∧ r = 50 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2322_232213


namespace NUMINAMATH_CALUDE_tank_capacity_is_40_l2322_232266

/-- Represents the total capacity of a water tank in gallons. -/
def tank_capacity : ℝ := sorry

/-- The tank is initially 3/4 full of water. -/
axiom initial_fill : (3 / 4 : ℝ) * tank_capacity = tank_capacity - 5

/-- Adding 5 gallons of water makes the tank 7/8 full. -/
axiom after_adding : (7 / 8 : ℝ) * tank_capacity = tank_capacity

/-- The tank's total capacity is 40 gallons. -/
theorem tank_capacity_is_40 : tank_capacity = 40 := by sorry

end NUMINAMATH_CALUDE_tank_capacity_is_40_l2322_232266


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l2322_232283

theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l2322_232283


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2322_232269

theorem solution_set_of_inequality (x : ℝ) :
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2322_232269


namespace NUMINAMATH_CALUDE_triangle_five_sixths_nine_fourths_l2322_232270

/-- The triangle operation ∆ defined for fractions -/
def triangle (m n p q : ℚ) : ℚ := m^2 * p * (q / n)

/-- Theorem stating that (5/6) ∆ (9/4) = 150 -/
theorem triangle_five_sixths_nine_fourths : 
  triangle (5/6) (5/6) (9/4) (9/4) = 150 := by sorry

end NUMINAMATH_CALUDE_triangle_five_sixths_nine_fourths_l2322_232270


namespace NUMINAMATH_CALUDE_polygon_with_120_degree_interior_angles_has_6_sides_l2322_232293

theorem polygon_with_120_degree_interior_angles_has_6_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 120 →
    (n - 2) * 180 = n * interior_angle →
    n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_120_degree_interior_angles_has_6_sides_l2322_232293


namespace NUMINAMATH_CALUDE_complement_union_M_N_l2322_232288

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 - 3 = p.1 - 2 ∧ p ≠ (2, 3)}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 + 1}

-- Theorem statement
theorem complement_union_M_N : 
  (M ∪ N)ᶜ = {(2, 3)} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l2322_232288


namespace NUMINAMATH_CALUDE_board_cut_theorem_l2322_232267

theorem board_cut_theorem (total_length : ℝ) (short_length : ℝ) : 
  total_length = 6 →
  short_length + 2 * short_length = total_length →
  short_length = 2 := by
sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l2322_232267


namespace NUMINAMATH_CALUDE_ten_steps_climb_ways_l2322_232220

/-- Number of ways to climb n steps when one can move to the next step or skip one step -/
def climbWays : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => climbWays (n + 1) + climbWays n

/-- The number of ways to climb 10 steps is 89 -/
theorem ten_steps_climb_ways : climbWays 10 = 89 := by sorry

end NUMINAMATH_CALUDE_ten_steps_climb_ways_l2322_232220


namespace NUMINAMATH_CALUDE_logarithmic_equality_l2322_232203

noncomputable def log_expr1 (x : ℝ) : ℝ := Real.log ((7 * x / 2) - (17 / 4)) / Real.log ((x / 2 + 1)^2)

noncomputable def log_expr2 (x : ℝ) : ℝ := Real.log ((3 * x / 2) - 6)^2 / Real.log (((7 * x / 2) - (17 / 4))^(1/2))

noncomputable def log_expr3 (x : ℝ) : ℝ := Real.log (x / 2 + 1) / Real.log (((3 * x / 2) - 6)^(1/2))

theorem logarithmic_equality (x : ℝ) :
  (log_expr1 x = log_expr2 x ∧ log_expr1 x = log_expr3 x + 1) ∨
  (log_expr2 x = log_expr3 x ∧ log_expr2 x = log_expr1 x + 1) ∨
  (log_expr3 x = log_expr1 x ∧ log_expr3 x = log_expr2 x + 1) ↔
  x = 7 :=
sorry

end NUMINAMATH_CALUDE_logarithmic_equality_l2322_232203


namespace NUMINAMATH_CALUDE_square_diagonal_from_rectangle_area_l2322_232228

theorem square_diagonal_from_rectangle_area (length width : ℝ) (h1 : length = 90) (h2 : width = 80) :
  let rectangle_area := length * width
  let square_side := (rectangle_area : ℝ).sqrt
  let square_diagonal := (2 * square_side ^ 2).sqrt
  square_diagonal = 120 := by sorry

end NUMINAMATH_CALUDE_square_diagonal_from_rectangle_area_l2322_232228


namespace NUMINAMATH_CALUDE_fraction_inequality_l2322_232219

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2322_232219


namespace NUMINAMATH_CALUDE_at_least_one_red_certain_l2322_232285

/-- Represents the number of red balls in the pocket -/
def num_red_balls : ℕ := 2

/-- Represents the number of white balls in the pocket -/
def num_white_balls : ℕ := 1

/-- Represents the total number of balls in the pocket -/
def total_balls : ℕ := num_red_balls + num_white_balls

/-- Represents the number of balls drawn from the pocket -/
def num_drawn : ℕ := 2

/-- Theorem stating that drawing at least one red ball when drawing 2 balls
    from a pocket containing 2 red balls and 1 white ball is a certain event -/
theorem at_least_one_red_certain :
  (num_red_balls.choose num_drawn + num_red_balls.choose (num_drawn - 1) * num_white_balls.choose 1) / total_balls.choose num_drawn = 1 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_red_certain_l2322_232285


namespace NUMINAMATH_CALUDE_binomial_2057_1_l2322_232214

theorem binomial_2057_1 : Nat.choose 2057 1 = 2057 := by
  sorry

end NUMINAMATH_CALUDE_binomial_2057_1_l2322_232214


namespace NUMINAMATH_CALUDE_peters_pizza_fraction_l2322_232233

theorem peters_pizza_fraction (total_slices : ℕ) (peters_solo_slices : ℕ) 
  (shared_with_paul : ℕ) (shared_with_mary : ℕ) : 
  total_slices = 18 → 
  peters_solo_slices = 3 → 
  shared_with_paul = 2 → 
  shared_with_mary = 1 → 
  (peters_solo_slices : ℚ) / total_slices + 
  (shared_with_paul : ℚ) / (2 * total_slices) + 
  (shared_with_mary : ℚ) / (2 * total_slices) = 11 / 36 := by
sorry

end NUMINAMATH_CALUDE_peters_pizza_fraction_l2322_232233


namespace NUMINAMATH_CALUDE_barry_cycling_time_difference_barry_cycling_proof_l2322_232246

theorem barry_cycling_time_difference : ℝ → Prop :=
  λ time_diff : ℝ =>
    let total_distance : ℝ := 4 * 3
    let time_at_varying_speeds : ℝ := 2 * (3 / 6) + 1 * (3 / 3) + 1 * (3 / 5)
    let time_at_constant_speed : ℝ := total_distance / 5
    let time_diff_hours : ℝ := time_at_varying_speeds - time_at_constant_speed
    time_diff = time_diff_hours * 60 ∧ time_diff = 42

theorem barry_cycling_proof : barry_cycling_time_difference 42 := by
  sorry

end NUMINAMATH_CALUDE_barry_cycling_time_difference_barry_cycling_proof_l2322_232246


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2322_232297

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a - 2) * a * (a + 2) = a^3 - 8 → 
  a^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2322_232297


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2322_232298

/-- Given two trains running on parallel rails in the same direction, this theorem
    calculates the speed of the second train based on the given conditions. -/
theorem train_speed_calculation
  (length1 : ℝ) (length2 : ℝ) (speed1 : ℝ) (crossing_time : ℝ)
  (h1 : length1 = 200) -- Length of first train in meters
  (h2 : length2 = 180) -- Length of second train in meters
  (h3 : speed1 = 45) -- Speed of first train in km/h
  (h4 : crossing_time = 273.6) -- Time to cross in seconds
  : ∃ (speed2 : ℝ), speed2 = 40 ∧ 
    (speed1 - speed2) * (crossing_time / 3600) = (length1 + length2) / 1000 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2322_232298


namespace NUMINAMATH_CALUDE_cement_calculation_l2322_232290

theorem cement_calculation (initial bought total : ℕ) 
  (h1 : initial = 98)
  (h2 : bought = 215)
  (h3 : total = 450) :
  total - (initial + bought) = 137 := by
  sorry

end NUMINAMATH_CALUDE_cement_calculation_l2322_232290


namespace NUMINAMATH_CALUDE_amys_birthday_money_l2322_232287

theorem amys_birthday_money (initial : ℕ) (chore_money : ℕ) (final_total : ℕ) : 
  initial = 2 → chore_money = 13 → final_total = 18 → 
  final_total - (initial + chore_money) = 3 := by
  sorry

end NUMINAMATH_CALUDE_amys_birthday_money_l2322_232287


namespace NUMINAMATH_CALUDE_sector_central_angle_l2322_232209

-- Define the sector
structure Sector where
  perimeter : ℝ
  area : ℝ

-- Define the theorem
theorem sector_central_angle (s : Sector) (h1 : s.perimeter = 12) (h2 : s.area = 8) :
  ∃ (r l : ℝ), r > 0 ∧ l > 0 ∧ 2 * r + l = s.perimeter ∧ 1/2 * r * l = s.area ∧
  (l / r = 1 ∨ l / r = 4) := by
  sorry

#check sector_central_angle

end NUMINAMATH_CALUDE_sector_central_angle_l2322_232209


namespace NUMINAMATH_CALUDE_sum_integers_minus15_to_5_l2322_232251

def sum_integers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_minus15_to_5 :
  sum_integers (-15) 5 = -105 := by
  sorry

end NUMINAMATH_CALUDE_sum_integers_minus15_to_5_l2322_232251


namespace NUMINAMATH_CALUDE_ball_max_height_l2322_232212

/-- The height of the ball as a function of time -/
def f (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 25

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), f s ≤ f t ∧ f t = 45 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l2322_232212


namespace NUMINAMATH_CALUDE_nancy_grew_two_onions_l2322_232211

/-- The number of onions grown by Nancy, given the number of onions grown by Dan and Mike,
    and the total number of onions grown by all three. -/
def nancys_onions (dans_onions mikes_onions total_onions : ℕ) : ℕ :=
  total_onions - dans_onions - mikes_onions

/-- Theorem stating that Nancy grew 2 onions given the conditions in the problem. -/
theorem nancy_grew_two_onions :
  nancys_onions 9 4 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_nancy_grew_two_onions_l2322_232211


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l2322_232239

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₄ = 17 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l2322_232239


namespace NUMINAMATH_CALUDE_xy_equals_four_l2322_232278

theorem xy_equals_four (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdistinct : x ≠ y) 
  (h_eq : x + 4 / x = y + 4 / y) : x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_four_l2322_232278


namespace NUMINAMATH_CALUDE_sixtieth_element_is_2064_l2322_232223

/-- The set of sums of powers of 2 with natural number exponents where the first exponent is less than the second -/
def PowerSumSet : Set ℕ :=
  {n | ∃ (x y : ℕ), x < y ∧ n = 2^x + 2^y}

/-- The 60th element in the ascending order of PowerSumSet -/
def sixtieth_element : ℕ := sorry

/-- Theorem stating that the 60th element of PowerSumSet is 2064 -/
theorem sixtieth_element_is_2064 : sixtieth_element = 2064 := by sorry

end NUMINAMATH_CALUDE_sixtieth_element_is_2064_l2322_232223


namespace NUMINAMATH_CALUDE_paige_team_total_points_l2322_232234

def team_size : ℕ := 5
def paige_points : ℕ := 11
def other_player_points : ℕ := 6

theorem paige_team_total_points :
  (paige_points + (team_size - 1) * other_player_points) = 35 := by
  sorry

end NUMINAMATH_CALUDE_paige_team_total_points_l2322_232234


namespace NUMINAMATH_CALUDE_symmetry_about_59_l2322_232291

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the two functions y₁ and y₂
def y₁ (x : ℝ) : ℝ := f (x - 19)
def y₂ (x : ℝ) : ℝ := f (99 - x)

-- Theorem stating that y₁ and y₂ are symmetric about x = 59
theorem symmetry_about_59 :
  ∀ (x : ℝ), y₁ f (118 - x) = y₂ f x :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_59_l2322_232291


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l2322_232272

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∀ (l : Line2D),
    pointOnLine { x := 1, y := 2 } l →
    equalIntercepts l →
    (l.a = 2 ∧ l.b = -1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -3) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l2322_232272


namespace NUMINAMATH_CALUDE_james_toy_cars_l2322_232230

/-- Proves that James buys 20 toy cars given the problem conditions -/
theorem james_toy_cars : 
  ∀ (cars soldiers : ℕ),
  soldiers = 2 * cars →
  cars + soldiers = 60 →
  cars = 20 := by
sorry

end NUMINAMATH_CALUDE_james_toy_cars_l2322_232230


namespace NUMINAMATH_CALUDE_green_jelly_bean_probability_l2322_232235

/-- Represents the count of jelly beans for each color -/
structure JellyBeanCount where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  black : ℕ

/-- Calculates the total number of jelly beans -/
def totalJellyBeans (count : JellyBeanCount) : ℕ :=
  count.red + count.green + count.yellow + count.blue + count.black

/-- Calculates the probability of selecting a green jelly bean -/
def probabilityGreen (count : JellyBeanCount) : ℚ :=
  count.green / (totalJellyBeans count)

/-- Theorem: The probability of selecting a green jelly bean from the given bag is 5/22 -/
theorem green_jelly_bean_probability :
  let bag := JellyBeanCount.mk 8 10 9 12 5
  probabilityGreen bag = 5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_green_jelly_bean_probability_l2322_232235


namespace NUMINAMATH_CALUDE_average_sale_calculation_l2322_232210

theorem average_sale_calculation (sale1 sale2 sale3 sale4 : ℕ) :
  sale1 = 2500 →
  sale2 = 4000 →
  sale3 = 3540 →
  sale4 = 1520 →
  (sale1 + sale2 + sale3 + sale4) / 4 = 2890 := by
sorry

end NUMINAMATH_CALUDE_average_sale_calculation_l2322_232210


namespace NUMINAMATH_CALUDE_households_without_car_or_bike_l2322_232254

theorem households_without_car_or_bike
  (total : ℕ)
  (both : ℕ)
  (car : ℕ)
  (bike_only : ℕ)
  (h_total : total = 90)
  (h_both : both = 14)
  (h_car : car = 44)
  (h_bike_only : bike_only = 35) :
  total - (car + bike_only + both) = 11 :=
by sorry

end NUMINAMATH_CALUDE_households_without_car_or_bike_l2322_232254


namespace NUMINAMATH_CALUDE_robins_hair_length_l2322_232248

/-- Given Robin's initial hair length and the amount cut, calculate the remaining length -/
theorem robins_hair_length (initial_length cut_length : ℕ) : 
  initial_length = 17 → cut_length = 4 → initial_length - cut_length = 13 := by
  sorry

end NUMINAMATH_CALUDE_robins_hair_length_l2322_232248


namespace NUMINAMATH_CALUDE_side_length_6_sufficient_not_necessary_l2322_232259

structure IsoscelesTriangle where
  x : ℝ
  y : ℝ
  perimeter_eq : 2 * x + y = 16
  base_eq : y = x + 1

def has_side_length_6 (t : IsoscelesTriangle) : Prop :=
  t.x = 6 ∨ t.y = 6

theorem side_length_6_sufficient_not_necessary (t : IsoscelesTriangle) :
  (∃ (t' : IsoscelesTriangle), has_side_length_6 t') ∧
  ¬(∀ (t' : IsoscelesTriangle), has_side_length_6 t') :=
sorry

end NUMINAMATH_CALUDE_side_length_6_sufficient_not_necessary_l2322_232259


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2322_232226

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2322_232226


namespace NUMINAMATH_CALUDE_cos_sin_cos_bounds_l2322_232268

theorem cos_sin_cos_bounds (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2 + Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_cos_bounds_l2322_232268


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_l2322_232247

theorem smallest_five_digit_multiple : ∃ (n : ℕ), 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  (15 ∣ n) ∧ (45 ∣ n) ∧ (54 ∣ n) ∧ 
  (∃ (k : ℕ), n = 2^k * (n / 2^k)) ∧
  (∀ (m : ℕ), m < n → 
    ¬((m ≥ 10000 ∧ m < 100000) ∧ 
      (15 ∣ m) ∧ (45 ∣ m) ∧ (54 ∣ m) ∧ 
      (∃ (j : ℕ), m = 2^j * (m / 2^j)))) ∧
  n = 69120 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_l2322_232247


namespace NUMINAMATH_CALUDE_defective_item_testing_methods_l2322_232218

theorem defective_item_testing_methods :
  let genuine_items : ℕ := 6
  let defective_items : ℕ := 4
  let total_tests : ℕ := 5
  let last_test_defective : ℕ := 1
  let genuine_in_first_four : ℕ := 1
  let defective_in_first_four : ℕ := 3

  (Nat.choose defective_items last_test_defective) *
  (Nat.choose genuine_items genuine_in_first_four) *
  (Nat.choose defective_in_first_four defective_in_first_four) *
  (Nat.factorial defective_in_first_four) = 576 :=
by
  sorry

end NUMINAMATH_CALUDE_defective_item_testing_methods_l2322_232218


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_max_l2322_232241

/-- An arithmetic sequence -/
def ArithmeticSequence := ℕ+ → ℝ

/-- Sum of the first n terms of an arithmetic sequence -/
def SumOfTerms (a : ArithmeticSequence) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_sum_max (a : ArithmeticSequence) :
  (SumOfTerms a 10 > 0) →
  (SumOfTerms a 11 = 0) →
  (∀ n : ℕ+, ∃ k : ℕ+, SumOfTerms a n ≤ SumOfTerms a k) →
  ∃ k : ℕ+, (k = 5 ∨ k = 6) ∧ 
    (∀ n : ℕ+, SumOfTerms a n ≤ SumOfTerms a k) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_max_l2322_232241


namespace NUMINAMATH_CALUDE_arctan_arcsin_arccos_sum_l2322_232279

theorem arctan_arcsin_arccos_sum : Real.arctan (Real.sqrt 3 / 3) + Real.arcsin (-1/2) + Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arctan_arcsin_arccos_sum_l2322_232279


namespace NUMINAMATH_CALUDE_parallel_line_to_hyperbola_asymptote_l2322_232286

/-- Given a hyperbola x²/16 - y²/9 = 1 and a line y = kx - 1 parallel to one of its asymptotes, 
    prove that k = 3/4 -/
theorem parallel_line_to_hyperbola_asymptote (k : ℝ) 
  (h1 : k > 0)
  (h2 : ∃ (x y : ℝ), y = k * x - 1 ∧ (x^2 / 16 - y^2 / 9 = 1) ∧ 
        (∀ (x' y' : ℝ), x'^2 / 16 - y'^2 / 9 = 1 → 
          (y - y') / (x - x') = k ∨ (y - y') / (x - x') = -k)) : 
  k = 3/4 := by sorry

end NUMINAMATH_CALUDE_parallel_line_to_hyperbola_asymptote_l2322_232286


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_max_value_f_min_value_ab_l2322_232225

-- Define the function f
def f (x : ℝ) : ℝ := |x + 5| - |x - 1|

-- Theorem for the solution set of f(x) ≤ x
theorem solution_set_f_leq_x :
  {x : ℝ | f x ≤ x} = {x : ℝ | -6 ≤ x ∧ x ≤ -4 ∨ x ≥ 6} :=
sorry

-- Theorem for the maximum value of f(x)
theorem max_value_f : 
  ∀ x : ℝ, f x ≤ 6 :=
sorry

-- Theorem for the minimum value of ab
theorem min_value_ab (a b : ℝ) (h : Real.log a + Real.log (2 * b) = Real.log (a + 4 * b + 6)) :
  a * b ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_max_value_f_min_value_ab_l2322_232225


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_some_but_not_all_l2322_232224

/-- A function that checks if a number is divisible by some but not all integers from 1 to 10 -/
def isDivisibleBySomeButNotAll (m : ℕ) : Prop :=
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ m % k = 0) ∧
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ m % k ≠ 0)

/-- The main theorem stating that 3 is the least positive integer satisfying the condition -/
theorem least_positive_integer_divisible_by_some_but_not_all :
  (∀ n : ℕ, 0 < n ∧ n < 3 → ¬isDivisibleBySomeButNotAll (n^2 - n)) ∧
  isDivisibleBySomeButNotAll (3^2 - 3) := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_some_but_not_all_l2322_232224


namespace NUMINAMATH_CALUDE_enthalpy_change_proof_l2322_232242

-- Define the sum of standard formation enthalpies for products
def sum_enthalpy_products : ℝ := -286.0 - 297.0

-- Define the sum of standard formation enthalpies for reactants
def sum_enthalpy_reactants : ℝ := -20.17

-- Define Hess's Law
def hess_law (products reactants : ℝ) : ℝ := products - reactants

-- Theorem statement
theorem enthalpy_change_proof :
  hess_law sum_enthalpy_products sum_enthalpy_reactants = -1125.66 := by
  sorry

end NUMINAMATH_CALUDE_enthalpy_change_proof_l2322_232242


namespace NUMINAMATH_CALUDE_festival_ferry_total_l2322_232252

/-- Represents the ferry schedule and passenger count --/
structure FerrySchedule where
  startTime : Nat  -- Start time in minutes after midnight
  endTime : Nat    -- End time in minutes after midnight
  interval : Nat   -- Interval between trips in minutes
  initialPassengers : Nat  -- Number of passengers on the first trip
  passengerDecrease : Nat  -- Decrease in passengers per trip

/-- Calculates the total number of people ferried --/
def totalPeopleFerried (schedule : FerrySchedule) : Nat :=
  let numTrips := (schedule.endTime - schedule.startTime) / schedule.interval + 1
  let lastTripPassengers := schedule.initialPassengers - (numTrips - 1) * schedule.passengerDecrease
  (numTrips * (schedule.initialPassengers + lastTripPassengers)) / 2

/-- The ferry schedule for the festival --/
def festivalFerry : FerrySchedule :=
  { startTime := 9 * 60  -- 9 AM in minutes
    endTime := 16 * 60   -- 4 PM in minutes
    interval := 30
    initialPassengers := 120
    passengerDecrease := 2 }

/-- Theorem stating the total number of people ferried to the festival --/
theorem festival_ferry_total : totalPeopleFerried festivalFerry = 1590 := by
  sorry


end NUMINAMATH_CALUDE_festival_ferry_total_l2322_232252


namespace NUMINAMATH_CALUDE_cat_catches_rat_l2322_232208

/-- The time it takes for a cat to catch a rat given their speeds and a head start. -/
theorem cat_catches_rat (cat_speed rat_speed : ℝ) (head_start : ℝ) (catch_time : ℝ) : 
  cat_speed = 90 →
  rat_speed = 36 →
  head_start = 6 →
  catch_time * (cat_speed - rat_speed) = head_start * rat_speed →
  catch_time = 4 :=
by sorry

end NUMINAMATH_CALUDE_cat_catches_rat_l2322_232208


namespace NUMINAMATH_CALUDE_malcolm_brushing_time_l2322_232299

/-- The number of days Malcolm brushes his teeth -/
def days : ℕ := 30

/-- The number of times Malcolm brushes his teeth per day -/
def brushings_per_day : ℕ := 3

/-- The total time Malcolm spends brushing his teeth in hours -/
def total_hours : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem malcolm_brushing_time :
  (total_hours * minutes_per_hour) / (days * brushings_per_day) = 2 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_brushing_time_l2322_232299


namespace NUMINAMATH_CALUDE_rectangle_hall_length_l2322_232296

theorem rectangle_hall_length :
  ∀ (length breadth : ℝ),
    length = breadth + 5 →
    length * breadth = 750 →
    length = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_hall_length_l2322_232296


namespace NUMINAMATH_CALUDE_divisible_numbers_in_range_l2322_232206

theorem divisible_numbers_in_range : ∃! n : ℕ, 
  1000 < n ∧ n < 2500 ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_divisible_numbers_in_range_l2322_232206


namespace NUMINAMATH_CALUDE_rotate_A_180_origin_l2322_232205

def rotate_180_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), -(p.2))

theorem rotate_A_180_origin :
  let A : ℝ × ℝ := (1, 2)
  rotate_180_origin A = (-1, -2) := by sorry

end NUMINAMATH_CALUDE_rotate_A_180_origin_l2322_232205


namespace NUMINAMATH_CALUDE_f_range_l2322_232250

noncomputable def f (x : ℝ) : ℝ :=
  (1/2) * Real.sin (2*x) * Real.tan x + 2 * Real.sin x * Real.tan (x/2)

theorem f_range :
  Set.range f = Set.Icc 0 3 ∪ Set.Ioo 3 4 :=
sorry

end NUMINAMATH_CALUDE_f_range_l2322_232250


namespace NUMINAMATH_CALUDE_children_percentage_l2322_232262

def total_passengers : ℕ := 60
def adult_passengers : ℕ := 45

theorem children_percentage : 
  (total_passengers - adult_passengers) / total_passengers * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_children_percentage_l2322_232262


namespace NUMINAMATH_CALUDE_time_A_is_120_l2322_232207

/-- The time it takes for B to fill the tank alone (in minutes) -/
def time_B : ℝ := 40

/-- The total time to fill the tank when B is used for half the time and A and B fill it together for the other half (in minutes) -/
def total_time : ℝ := 29.999999999999993

/-- The time it takes for A to fill the tank alone (in minutes) -/
def time_A : ℝ := 120

/-- Theorem stating that the time for A to fill the tank alone is 120 minutes -/
theorem time_A_is_120 : time_A = 120 := by sorry

end NUMINAMATH_CALUDE_time_A_is_120_l2322_232207


namespace NUMINAMATH_CALUDE_rectangular_prism_area_volume_relation_l2322_232221

theorem rectangular_prism_area_volume_relation 
  (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) :
  (x * y) * (y * z) * (z * x) = (x * y * z)^3 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_area_volume_relation_l2322_232221


namespace NUMINAMATH_CALUDE_mary_farm_animals_l2322_232276

-- Define the initial state and transactions
def initial_lambs : ℕ := 18
def initial_alpacas : ℕ := 5
def lamb_babies : ℕ := 7 * 4
def traded_lambs : ℕ := 8
def traded_alpacas : ℕ := 2
def gained_goats : ℕ := 3
def gained_chickens : ℕ := 10
def alpacas_from_chickens : ℕ := 2
def additional_lambs : ℕ := 20
def additional_alpacas : ℕ := 6

-- Define the theorem
theorem mary_farm_animals :
  let lambs := initial_lambs + lamb_babies - traded_lambs + additional_lambs
  let alpacas := initial_alpacas - traded_alpacas + alpacas_from_chickens + additional_alpacas
  let goats := gained_goats
  let chickens := gained_chickens / 2
  (lambs = 58 ∧ alpacas = 11 ∧ goats = 3 ∧ chickens = 5) := by
  sorry

end NUMINAMATH_CALUDE_mary_farm_animals_l2322_232276


namespace NUMINAMATH_CALUDE_puppy_food_bags_puppy_food_bags_proof_l2322_232204

/-- Calculates the number of bags of special dog food needed for a puppy's first year --/
theorem puppy_food_bags : ℕ :=
  let days_in_year : ℕ := 365
  let first_period : ℕ := 60
  let second_period : ℕ := days_in_year - first_period
  let first_period_consumption : ℕ := first_period * 2
  let second_period_consumption : ℕ := second_period * 4
  let total_consumption : ℕ := first_period_consumption + second_period_consumption
  let ounces_per_pound : ℕ := 16
  let pounds_per_bag : ℕ := 5
  let ounces_per_bag : ℕ := ounces_per_pound * pounds_per_bag
  let bags_needed : ℕ := (total_consumption + ounces_per_bag - 1) / ounces_per_bag
  17

/-- Proof that the number of bags needed is 17 --/
theorem puppy_food_bags_proof : puppy_food_bags = 17 := by
  sorry

end NUMINAMATH_CALUDE_puppy_food_bags_puppy_food_bags_proof_l2322_232204


namespace NUMINAMATH_CALUDE_inequality_constraint_l2322_232275

theorem inequality_constraint (a b : ℝ) : 
  (∀ x : ℝ, |a * Real.sin x + b * Real.sin (2 * x)| ≤ 1) →
  |a| + |b| ≥ 2 / Real.sqrt 3 →
  ((a = 4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = 4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3))) :=
by sorry


end NUMINAMATH_CALUDE_inequality_constraint_l2322_232275


namespace NUMINAMATH_CALUDE_min_sum_products_l2322_232244

theorem min_sum_products (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (h : x + y + z = 3 * x * y * z) : 
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 3 * a * b * c → 
    x * y + y * z + x * z ≤ a * b + b * c + a * c :=
by sorry

end NUMINAMATH_CALUDE_min_sum_products_l2322_232244


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2322_232281

theorem inequality_solution_set (a b : ℝ) (h : a ≠ b) :
  {x : ℝ | a^2 * x + b^2 * (1 - x) ≥ (a * x + b * (1 - x))^2} = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2322_232281


namespace NUMINAMATH_CALUDE_samantha_born_1986_l2322_232243

/-- The year of the first Math Kangaroo contest -/
def first_math_kangaroo_year : ℕ := 1991

/-- The age of Samantha when she took the tenth Math Kangaroo -/
def samantha_age_tenth_kangaroo : ℕ := 14

/-- Function to calculate the year of the nth Math Kangaroo contest -/
def math_kangaroo_year (n : ℕ) : ℕ := first_math_kangaroo_year + n - 1

/-- Samantha's birth year -/
def samantha_birth_year : ℕ := math_kangaroo_year 10 - samantha_age_tenth_kangaroo

theorem samantha_born_1986 : samantha_birth_year = 1986 := by
  sorry

end NUMINAMATH_CALUDE_samantha_born_1986_l2322_232243


namespace NUMINAMATH_CALUDE_prism_24_edges_has_10_faces_l2322_232202

/-- A prism is a polyhedron with two congruent parallel faces (bases) and rectangular lateral faces. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges -/
def num_faces (p : Prism) : ℕ :=
  let base_edges := p.edges / 3
  base_edges + 2

theorem prism_24_edges_has_10_faces (p : Prism) (h : p.edges = 24) : num_faces p = 10 := by
  sorry

end NUMINAMATH_CALUDE_prism_24_edges_has_10_faces_l2322_232202


namespace NUMINAMATH_CALUDE_passengers_count_l2322_232282

/-- Calculates the number of passengers on a bus after three stops --/
def passengers_after_three_stops : ℕ :=
  let initial := 0
  let after_first_stop := initial + 7
  let after_second_stop := after_first_stop - 3 + 5
  let after_third_stop := after_second_stop - 2 + 4
  after_third_stop

/-- Theorem stating that the number of passengers after three stops is 11 --/
theorem passengers_count : passengers_after_three_stops = 11 := by
  sorry

end NUMINAMATH_CALUDE_passengers_count_l2322_232282


namespace NUMINAMATH_CALUDE_committee_arrangement_l2322_232236

theorem committee_arrangement (n m : ℕ) (hn : n = 7) (hm : m = 3) :
  (Nat.choose (n + m) m) = 120 := by
  sorry

end NUMINAMATH_CALUDE_committee_arrangement_l2322_232236


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l2322_232245

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 27 / 8) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l2322_232245


namespace NUMINAMATH_CALUDE_distance_to_place_l2322_232229

/-- The distance to the place -/
def distance : ℝ := 48

/-- The rowing speed in still water (km/h) -/
def rowing_speed : ℝ := 10

/-- The current velocity (km/h) -/
def current_velocity : ℝ := 2

/-- The wind speed (km/h) -/
def wind_speed : ℝ := 4

/-- The total time for the round trip (hours) -/
def total_time : ℝ := 15

/-- The effective speed towards the place (km/h) -/
def speed_to_place : ℝ := rowing_speed - wind_speed - current_velocity

/-- The effective speed returning from the place (km/h) -/
def speed_from_place : ℝ := rowing_speed + wind_speed + current_velocity

theorem distance_to_place : 
  distance = (total_time * speed_to_place * speed_from_place) / (speed_to_place + speed_from_place) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_place_l2322_232229


namespace NUMINAMATH_CALUDE_least_integer_in_ratio_l2322_232280

theorem least_integer_in_ratio (a b c : ℕ+) : 
  (a : ℝ) + (b : ℝ) + (c : ℝ) = 90 →
  (b : ℝ) = 3 * (a : ℝ) →
  (c : ℝ) = 5 * (a : ℝ) →
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_least_integer_in_ratio_l2322_232280


namespace NUMINAMATH_CALUDE_solve_for_y_l2322_232261

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 2 = y - 4) (h2 : x = -3) : y = 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2322_232261


namespace NUMINAMATH_CALUDE_train_length_l2322_232217

/-- The length of a train given its speed and the time it takes to pass a bridge -/
theorem train_length (bridge_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) :
  bridge_length = 180 →
  train_speed_kmh = 65 →
  passing_time = 21.04615384615385 →
  ∃ train_length : ℝ, abs (train_length - 200) < 0.00001 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l2322_232217


namespace NUMINAMATH_CALUDE_james_total_socks_l2322_232271

/-- Calculates the total number of socks James has -/
def total_socks (red_pairs : ℕ) : ℕ :=
  let red := red_pairs * 2
  let black := red / 2
  let white := (red + black) * 2
  red + black + white

/-- Proves that James has 180 socks in total -/
theorem james_total_socks : total_socks 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_james_total_socks_l2322_232271


namespace NUMINAMATH_CALUDE_joes_fast_food_cost_l2322_232253

/-- Calculates the total cost of a meal at Joe's Fast Food --/
def total_cost (sandwich_price : ℚ) (soda_price : ℚ) (fries_price : ℚ) 
                (sandwich_qty : ℕ) (soda_qty : ℕ) (fries_qty : ℕ) 
                (discount : ℚ) : ℚ :=
  sandwich_price * sandwich_qty + soda_price * soda_qty + fries_price * fries_qty - discount

/-- Theorem stating the total cost of the specified meal --/
theorem joes_fast_food_cost : 
  total_cost 4 (3/2) (5/2) 4 6 3 5 = 55/2 := by
  sorry

end NUMINAMATH_CALUDE_joes_fast_food_cost_l2322_232253


namespace NUMINAMATH_CALUDE_solve_equation_l2322_232215

theorem solve_equation :
  ∃ y : ℚ, (2 * y + 3 * y = 500 - (4 * y + 5 * y)) ∧ y = 250 / 7 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l2322_232215


namespace NUMINAMATH_CALUDE_ticket_price_reduction_l2322_232249

theorem ticket_price_reduction (original_price : ℚ) 
  (h1 : original_price = 50)
  (h2 : ∃ (x : ℚ), x > 0 ∧ 
    (4/3 * x) * (original_price - 25/2) = (5/4) * (x * original_price)) :
  original_price - 25/2 = 46.875 := by
sorry

end NUMINAMATH_CALUDE_ticket_price_reduction_l2322_232249


namespace NUMINAMATH_CALUDE_not_prime_5n_plus_1_l2322_232240

theorem not_prime_5n_plus_1 (n : ℕ) (x y : ℕ) 
  (h1 : x^2 = 2*n + 1) (h2 : y^2 = 3*n + 1) : 
  ¬ Nat.Prime (5*n + 1) := by
sorry

end NUMINAMATH_CALUDE_not_prime_5n_plus_1_l2322_232240


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_length_l2322_232231

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_line : ℝ

/-- The property that the line joining the midpoints of the diagonals is half the difference of the bases -/
def midpoint_line_property (t : Trapezoid) : Prop :=
  t.midpoint_line = (t.long_base - t.short_base) / 2

/-- Theorem: In a trapezoid where the line joining the midpoints of the diagonals has length 4
    and the longer base is 100, the shorter base has length 92 -/
theorem trapezoid_shorter_base_length :
  ∀ t : Trapezoid,
    t.long_base = 100 →
    t.midpoint_line = 4 →
    midpoint_line_property t →
    t.short_base = 92 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_shorter_base_length_l2322_232231


namespace NUMINAMATH_CALUDE_income_difference_l2322_232258

-- Define the incomes of A and B
def A (B : ℝ) : ℝ := 0.75 * B

-- Theorem statement
theorem income_difference (B : ℝ) (h : B > 0) : 
  (B - A B) / (A B) = 1/3 := by sorry

end NUMINAMATH_CALUDE_income_difference_l2322_232258


namespace NUMINAMATH_CALUDE_abs_difference_of_roots_l2322_232277

theorem abs_difference_of_roots (α β : ℝ) (h1 : α + β = 17) (h2 : α * β = 70) : 
  |α - β| = 3 := by
sorry

end NUMINAMATH_CALUDE_abs_difference_of_roots_l2322_232277


namespace NUMINAMATH_CALUDE_min_containers_correct_l2322_232201

/-- Calculates the minimum number of containers needed to transport boxes with weight restrictions. -/
def min_containers (total_boxes : ℕ) (main_box_weight : ℕ) (light_boxes : ℕ) (light_box_weight : ℕ) (max_container_weight : ℕ) : ℕ :=
  let total_weight := (total_boxes - light_boxes) * main_box_weight + light_boxes * light_box_weight
  let boxes_per_container := max_container_weight * 1000 / main_box_weight
  (total_boxes + boxes_per_container - 1) / boxes_per_container

theorem min_containers_correct :
  min_containers 90000 3300 5000 200 100 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_min_containers_correct_l2322_232201


namespace NUMINAMATH_CALUDE_division_by_less_than_one_multiplication_by_greater_than_one_multiply_by_100_equals_divide_by_001_l2322_232216

-- Statement 1
theorem division_by_less_than_one (x y : ℝ) (hx : x > 0) (hy : 0 < y) (hy1 : y < 1) :
  x / y > x :=
sorry

-- Statement 2
theorem multiplication_by_greater_than_one (x y : ℝ) (hy : y > 1) :
  x * y > x :=
sorry

-- Statement 3
theorem multiply_by_100_equals_divide_by_001 (x : ℝ) :
  x * 100 = x / 0.01 :=
sorry

end NUMINAMATH_CALUDE_division_by_less_than_one_multiplication_by_greater_than_one_multiply_by_100_equals_divide_by_001_l2322_232216


namespace NUMINAMATH_CALUDE_valid_partition_exists_l2322_232294

/-- Represents a person with their country and position on the circle -/
structure Person where
  country : Nat
  position : Nat

/-- Represents a partition of people into two groups -/
def Partition := Person → Bool

/-- The total number of people -/
def total_people : Nat := 100

/-- The total number of countries -/
def total_countries : Nat := 50

/-- Checks if two people are from the same country -/
def same_country (p1 p2 : Person) : Prop := p1.country = p2.country

/-- Checks if two people are adjacent on the circle -/
def adjacent (p1 p2 : Person) : Prop :=
  (p1.position + 1) % total_people = p2.position ∨
  (p2.position + 1) % total_people = p1.position

/-- Main theorem: There exists a valid partition -/
theorem valid_partition_exists : ∃ (partition : Partition),
  (∀ p1 p2 : Person, same_country p1 p2 → partition p1 ≠ partition p2) ∧
  (∀ p1 p2 p3 : Person, adjacent p1 p2 ∧ adjacent p2 p3 →
    ¬(partition p1 = partition p2 ∧ partition p2 = partition p3)) :=
  sorry

end NUMINAMATH_CALUDE_valid_partition_exists_l2322_232294


namespace NUMINAMATH_CALUDE_root_product_value_l2322_232289

theorem root_product_value : 
  ∀ (a b c d : ℝ), 
  (a^2 + 2000*a + 1 = 0) → 
  (b^2 + 2000*b + 1 = 0) → 
  (c^2 - 2008*c + 1 = 0) → 
  (d^2 - 2008*d + 1 = 0) → 
  (a+c)*(b+c)*(a-d)*(b-d) = 32064 := by
sorry

end NUMINAMATH_CALUDE_root_product_value_l2322_232289


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l2322_232238

-- Define the probability of having a boy or a girl
def prob_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- Theorem statement
theorem prob_at_least_one_boy_one_girl :
  (1 : ℚ) - (prob_boy_or_girl ^ num_children + prob_boy_or_girl ^ num_children) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l2322_232238


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_l2322_232292

theorem sqrt_six_times_sqrt_three : Real.sqrt 6 * Real.sqrt 3 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_l2322_232292


namespace NUMINAMATH_CALUDE_symmetric_point_on_x_axis_l2322_232232

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the form y = kx -/
structure Line where
  k : ℝ

/-- Predicate to check if a point is on the x-axis -/
def onXAxis (p : Point) : Prop :=
  p.y = 0

/-- Predicate to check if two points are symmetric about a line -/
def areSymmetric (p1 p2 : Point) (l : Line) : Prop :=
  (p1.y + p2.y) / 2 = l.k * ((p1.x + p2.x) / 2) ∧
  (p1.y - p2.y) / (p1.x - p2.x) * l.k = -1

theorem symmetric_point_on_x_axis (A : Point) (l : Line) :
  A.x = 3 ∧ A.y = 5 →
  ∃ (B : Point), areSymmetric A B l ∧ onXAxis B →
  l.k = (-3 + Real.sqrt 34) / 5 ∨ l.k = (-3 - Real.sqrt 34) / 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_on_x_axis_l2322_232232


namespace NUMINAMATH_CALUDE_terms_are_not_like_l2322_232274

/-- Two algebraic terms are considered like terms if they have the same variables raised to the same powers. -/
def are_like_terms (term1 term2 : Type) : Prop := sorry

/-- The first term in the problem -/
def term1 : Type := sorry

/-- The second term in the problem -/
def term2 : Type := sorry

/-- Theorem stating that the two terms are not like terms -/
theorem terms_are_not_like : ¬(are_like_terms term1 term2) := by sorry

end NUMINAMATH_CALUDE_terms_are_not_like_l2322_232274


namespace NUMINAMATH_CALUDE_product_of_roots_l2322_232265

theorem product_of_roots (b : ℝ) : 
  (∃ x₁ x₂ : ℝ, 2020 * x₁^2 + b * x₁ + 2021 = 0 ∧ 2020 * x₂^2 + b * x₂ + 2021 = 0 ∧ x₁ ≠ x₂) →
  (∃ y₁ y₂ : ℝ, 2019 * y₁^2 + b * y₁ + 2020 = 0 ∧ 2019 * y₂^2 + b * y₂ + 2020 = 0 ∧ y₁ ≠ y₂) →
  (∃ z₁ z₂ : ℝ, z₁^2 + b * z₁ + 2019 = 0 ∧ z₂^2 + b * z₂ + 2019 = 0 ∧ z₁ ≠ z₂) →
  (2021 / 2020) * (2020 / 2019) * 2019 = 2021 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2322_232265


namespace NUMINAMATH_CALUDE_workshop_worker_count_l2322_232295

/-- Represents the workshop scenario with workers, salaries, and technicians. -/
structure Workshop where
  totalWorkers : ℕ
  averageSalary : ℕ
  technicianCount : ℕ
  technicianAvgSalary : ℕ
  nonTechnicianAvgSalary : ℕ

/-- Theorem stating that given the conditions, the total number of workers is 18. -/
theorem workshop_worker_count (w : Workshop)
  (h1 : w.averageSalary = 8000)
  (h2 : w.technicianCount = 6)
  (h3 : w.technicianAvgSalary = 12000)
  (h4 : w.nonTechnicianAvgSalary = 6000)
  (h5 : w.totalWorkers * w.averageSalary = 
        w.technicianCount * w.technicianAvgSalary + 
        (w.totalWorkers - w.technicianCount) * w.nonTechnicianAvgSalary) :
  w.totalWorkers = 18 := by
  sorry


end NUMINAMATH_CALUDE_workshop_worker_count_l2322_232295


namespace NUMINAMATH_CALUDE_smallest_x_value_l2322_232227

theorem smallest_x_value (x : ℚ) : 
  (6 * (9 * x^2 + 9 * x + 10) = x * (9 * x - 45)) → x ≥ -4/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2322_232227


namespace NUMINAMATH_CALUDE_auction_sale_total_l2322_232257

/-- Calculate the total amount received from selling a TV and a phone at an auction -/
theorem auction_sale_total (tv_initial_cost phone_initial_cost : ℚ) 
  (tv_price_increase phone_price_increase : ℚ) : ℚ :=
  by
  -- Define the initial costs and price increases
  have h1 : tv_initial_cost = 500 := by sorry
  have h2 : tv_price_increase = 2 / 5 := by sorry
  have h3 : phone_initial_cost = 400 := by sorry
  have h4 : phone_price_increase = 40 / 100 := by sorry

  -- Calculate the final prices
  let tv_final_price := tv_initial_cost + tv_initial_cost * tv_price_increase
  let phone_final_price := phone_initial_cost + phone_initial_cost * phone_price_increase

  -- Calculate the total amount received
  let total_amount := tv_final_price + phone_final_price

  -- Prove that the total amount is equal to 1260
  sorry

end NUMINAMATH_CALUDE_auction_sale_total_l2322_232257


namespace NUMINAMATH_CALUDE_ashley_wedding_guests_l2322_232263

/-- Calculates the number of wedding guests based on champagne requirements. -/
def wedding_guests (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) (bottles_needed : ℕ) : ℕ :=
  (servings_per_bottle / glasses_per_guest) * bottles_needed

/-- Theorem stating that Ashley has 120 wedding guests. -/
theorem ashley_wedding_guests :
  wedding_guests 2 6 40 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ashley_wedding_guests_l2322_232263


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l2322_232273

theorem polynomial_identity_sum (b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄ : ℝ) :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃) * (x^2 + b₄*x + c₄)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ + b₄*c₄ = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l2322_232273


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2322_232260

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2322_232260


namespace NUMINAMATH_CALUDE_geometric_jump_sequence_ratio_l2322_232222

/-- A sequence is a jump sequence if (a_i - a_i+2)(a_i+2 - a_i+1) > 0 for any three consecutive terms -/
def is_jump_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, (a i - a (i + 2)) * (a (i + 2) - a (i + 1)) > 0

/-- A sequence is geometric with ratio q if a_(n+1) = q * a_n for all n -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_jump_sequence_ratio {a : ℕ → ℝ} {q : ℝ} 
  (h_geometric : is_geometric_sequence a q)
  (h_jump : is_jump_sequence a) :
  -1 < q ∧ q < 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_jump_sequence_ratio_l2322_232222


namespace NUMINAMATH_CALUDE_f_of_3_equals_10_l2322_232200

def f (x : ℝ) : ℝ := 3 * x + 1

theorem f_of_3_equals_10 : f 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_10_l2322_232200


namespace NUMINAMATH_CALUDE_music_festival_audience_count_l2322_232255

/-- Represents the distribution of audience for a band -/
structure BandDistribution where
  underThirtyMale : ℝ
  underThirtyFemale : ℝ
  thirtyToFiftyMale : ℝ
  thirtyToFiftyFemale : ℝ
  overFiftyMale : ℝ
  overFiftyFemale : ℝ

/-- The music festival with its audience distribution -/
def MusicFestival : List BandDistribution :=
  [
    { underThirtyMale := 0.04, underThirtyFemale := 0.0266667, thirtyToFiftyMale := 0.0375, thirtyToFiftyFemale := 0.0458333, overFiftyMale := 0.00833333, overFiftyFemale := 0.00833333 },
    { underThirtyMale := 0.03, underThirtyFemale := 0.07, thirtyToFiftyMale := 0.02, thirtyToFiftyFemale := 0.03, overFiftyMale := 0.00833333, overFiftyFemale := 0.00833333 },
    { underThirtyMale := 0.02, underThirtyFemale := 0.03, thirtyToFiftyMale := 0.0416667, thirtyToFiftyFemale := 0.0416667, overFiftyMale := 0.0133333, overFiftyFemale := 0.02 },
    { underThirtyMale := 0.0458333, underThirtyFemale := 0.0375, thirtyToFiftyMale := 0.03, thirtyToFiftyFemale := 0.0366667, overFiftyMale := 0.01, overFiftyFemale := 0.00666667 },
    { underThirtyMale := 0.015, underThirtyFemale := 0.0183333, thirtyToFiftyMale := 0.0333333, thirtyToFiftyFemale := 0.0333333, overFiftyMale := 0.03, overFiftyFemale := 0.0366667 },
    { underThirtyMale := 0.0583333, underThirtyFemale := 0.025, thirtyToFiftyMale := 0.03, thirtyToFiftyFemale := 0.0366667, overFiftyMale := 0.00916667, overFiftyFemale := 0.00750 }
  ]

theorem music_festival_audience_count : 
  let totalMaleUnder30 := (MusicFestival.map (λ b => b.underThirtyMale)).sum
  ∃ n : ℕ, n ≥ 431 ∧ n < 432 ∧ (90 : ℝ) / totalMaleUnder30 = n := by
  sorry

end NUMINAMATH_CALUDE_music_festival_audience_count_l2322_232255


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equations_l2322_232256

theorem smallest_x_satisfying_equations : 
  ∃ x : ℝ, x = -12 ∧ 
    abs (x - 3) = 15 ∧ 
    abs (x + 2) = 10 ∧ 
    ∀ y : ℝ, (abs (y - 3) = 15 ∧ abs (y + 2) = 10) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equations_l2322_232256


namespace NUMINAMATH_CALUDE_total_coins_l2322_232264

theorem total_coins (quarters_piles : Nat) (quarters_per_pile : Nat)
                    (dimes_piles : Nat) (dimes_per_pile : Nat)
                    (nickels_piles : Nat) (nickels_per_pile : Nat)
                    (pennies_piles : Nat) (pennies_per_pile : Nat)
                    (h1 : quarters_piles = 8) (h2 : quarters_per_pile = 5)
                    (h3 : dimes_piles = 6) (h4 : dimes_per_pile = 7)
                    (h5 : nickels_piles = 4) (h6 : nickels_per_pile = 4)
                    (h7 : pennies_piles = 3) (h8 : pennies_per_pile = 6) :
  quarters_piles * quarters_per_pile +
  dimes_piles * dimes_per_pile +
  nickels_piles * nickels_per_pile +
  pennies_piles * pennies_per_pile = 116 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_l2322_232264


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2322_232237

theorem right_triangle_third_side (a b x : ℝ) : 
  a = 3 → b = 4 → (a^2 + b^2 = x^2 ∨ a^2 + x^2 = b^2) → x = 5 ∨ x = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2322_232237
