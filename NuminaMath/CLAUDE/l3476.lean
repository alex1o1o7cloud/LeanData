import Mathlib

namespace NUMINAMATH_CALUDE_complement_of_P_in_U_l3476_347634

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x | x^2 < 1}

-- State the theorem
theorem complement_of_P_in_U :
  (U \ P) = {x | x ≤ -1 ∨ x ≥ 1} :=
sorry

end NUMINAMATH_CALUDE_complement_of_P_in_U_l3476_347634


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l3476_347636

theorem max_value_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  ∃ (M : ℝ), M = 12 ∧ ∀ z : ℂ, Complex.abs z = 2 →
    Complex.abs ((z - 2)^2 * (z + 2)) ≤ M ∧
    ∃ w₀ : ℂ, Complex.abs w₀ = 2 ∧ Complex.abs ((w₀ - 2)^2 * (w₀ + 2)) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l3476_347636


namespace NUMINAMATH_CALUDE_monomial_count_l3476_347679

-- Define what a monomial is
def is_monomial (expr : String) : Bool := sorry

-- Define the set of expressions
def expressions : List String := ["(x+a)/2", "-2", "2x^2y", "b", "7x^2+8x-1"]

-- State the theorem
theorem monomial_count : 
  (expressions.filter is_monomial).length = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_count_l3476_347679


namespace NUMINAMATH_CALUDE_total_interest_is_1380_l3476_347601

def total_investment : ℝ := 17000
def low_rate_investment : ℝ := 12000
def low_rate : ℝ := 0.04
def high_rate : ℝ := 0.18

def calculate_total_interest : ℝ := 
  let high_rate_investment := total_investment - low_rate_investment
  let low_rate_interest := low_rate_investment * low_rate
  let high_rate_interest := high_rate_investment * high_rate
  low_rate_interest + high_rate_interest

theorem total_interest_is_1380 : 
  calculate_total_interest = 1380 := by sorry

end NUMINAMATH_CALUDE_total_interest_is_1380_l3476_347601


namespace NUMINAMATH_CALUDE_selling_price_calculation_l3476_347640

def calculate_selling_price (initial_price maintenance_cost repair_cost transportation_cost : ℝ)
  (tax_rate currency_loss_rate depreciation_rate profit_margin : ℝ) : ℝ :=
  let total_expenses := initial_price + maintenance_cost + repair_cost + transportation_cost
  let after_tax := total_expenses * (1 + tax_rate)
  let after_currency_loss := after_tax * (1 - currency_loss_rate)
  let after_depreciation := after_currency_loss * (1 - depreciation_rate)
  after_depreciation * (1 + profit_margin)

theorem selling_price_calculation :
  calculate_selling_price 10000 2000 5000 1000 0.1 0.05 0.15 0.5 = 23982.75 :=
by sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l3476_347640


namespace NUMINAMATH_CALUDE_polygon_sides_l3476_347693

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →  -- Ensure it's a valid polygon
  ((n - 2) * 180 = 3 * 360) → -- Interior angles sum is 3 times exterior angles sum
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l3476_347693


namespace NUMINAMATH_CALUDE_age_order_l3476_347615

structure Person where
  name : String
  age : ℕ

def age_relationship (sergei sasha tolia : Person) : Prop :=
  sergei.age = 2 * (sergei.age + tolia.age - sergei.age)

theorem age_order (sergei sasha tolia : Person) 
  (h : age_relationship sergei sasha tolia) : 
  sergei.age > tolia.age ∧ tolia.age > sasha.age :=
by
  sorry

#check age_order

end NUMINAMATH_CALUDE_age_order_l3476_347615


namespace NUMINAMATH_CALUDE_quadratic_completion_l3476_347691

theorem quadratic_completion (x : ℝ) : x^2 + 16*x + 72 = (x + 8)^2 + 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l3476_347691


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3476_347665

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, 1 < x ∧ x < 2 → x > 0) ∧
  ¬(∀ x : ℝ, x > 0 → 1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3476_347665


namespace NUMINAMATH_CALUDE_square_factor_l3476_347689

theorem square_factor (a b : ℝ) (square : ℝ) :
  square * (3 * a * b) = 3 * a^2 * b → square = a := by
  sorry

end NUMINAMATH_CALUDE_square_factor_l3476_347689


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l3476_347659

theorem opposite_of_negative_one_third :
  -((-1 : ℚ) / 3) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l3476_347659


namespace NUMINAMATH_CALUDE_tammy_mountain_climb_l3476_347618

/-- Proves that given the conditions of Tammy's mountain climb, her speed on the second day was 4 km/h -/
theorem tammy_mountain_climb (total_time : ℝ) (total_distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ)
  (h1 : total_time = 14)
  (h2 : speed_increase = 0.5)
  (h3 : time_decrease = 2)
  (h4 : total_distance = 52) :
  ∃ (speed1 : ℝ) (time1 : ℝ),
    speed1 > 0 ∧
    time1 > 0 ∧
    time1 + (time1 - time_decrease) = total_time ∧
    speed1 * time1 + (speed1 + speed_increase) * (time1 - time_decrease) = total_distance ∧
    speed1 + speed_increase = 4 := by
  sorry

end NUMINAMATH_CALUDE_tammy_mountain_climb_l3476_347618


namespace NUMINAMATH_CALUDE_orange_bows_count_l3476_347627

theorem orange_bows_count (total : ℕ) (black : ℕ) : 
  black = 40 →
  (1 : ℚ) / 4 + (1 : ℚ) / 3 + (1 : ℚ) / 12 + (black : ℚ) / total = 1 →
  (1 : ℚ) / 12 * total = 10 :=
by sorry

end NUMINAMATH_CALUDE_orange_bows_count_l3476_347627


namespace NUMINAMATH_CALUDE_final_painting_width_l3476_347622

theorem final_painting_width (total_paintings : Nat) (total_area : Nat) 
  (small_paintings : Nat) (small_painting_side : Nat)
  (large_painting_width large_painting_height : Nat)
  (final_painting_height : Nat) :
  total_paintings = 5 →
  total_area = 200 →
  small_paintings = 3 →
  small_painting_side = 5 →
  large_painting_width = 10 →
  large_painting_height = 8 →
  final_painting_height = 5 →
  ∃ (final_painting_width : Nat),
    final_painting_width = 9 ∧
    total_area = 
      small_paintings * small_painting_side * small_painting_side +
      large_painting_width * large_painting_height +
      final_painting_height * final_painting_width :=
by sorry

end NUMINAMATH_CALUDE_final_painting_width_l3476_347622


namespace NUMINAMATH_CALUDE_triangle_side_count_is_35_l3476_347626

/-- The number of integer values for the third side of a triangle with two sides of length 18 and 45 -/
def triangle_side_count : ℕ :=
  let possible_x := Finset.filter (fun x : ℕ =>
    x > 27 ∧ x < 63 ∧ x + 18 > 45 ∧ x + 45 > 18 ∧ 18 + 45 > x) (Finset.range 100)
  possible_x.card

theorem triangle_side_count_is_35 : triangle_side_count = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_count_is_35_l3476_347626


namespace NUMINAMATH_CALUDE_tom_game_sale_amount_l3476_347623

/-- Calculates the amount received from selling a portion of an asset that has increased in value -/
def sellPartOfAppreciatedAsset (initialValue : ℝ) (appreciationFactor : ℝ) (portionSold : ℝ) : ℝ :=
  initialValue * appreciationFactor * portionSold

/-- Proves that Tom sold his games for $240 -/
theorem tom_game_sale_amount : 
  sellPartOfAppreciatedAsset 200 3 0.4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_tom_game_sale_amount_l3476_347623


namespace NUMINAMATH_CALUDE_xy_value_given_equation_l3476_347631

theorem xy_value_given_equation :
  ∀ x y : ℝ, 2*x^2 + 2*x*y + y^2 - 6*x + 9 = 0 → x^y = 1/27 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_given_equation_l3476_347631


namespace NUMINAMATH_CALUDE_raisins_amount_l3476_347669

/-- Represents the mixture of raisins and nuts -/
structure Mixture where
  raisins : ℝ
  nuts : ℝ
  raisin_cost : ℝ
  nut_cost : ℝ

/-- The conditions of the problem -/
def problem_conditions (m : Mixture) : Prop :=
  m.nuts = 4 ∧
  m.nut_cost = 3 * m.raisin_cost ∧
  m.raisins * m.raisin_cost = 0.25 * (m.raisins * m.raisin_cost + m.nuts * m.nut_cost)

/-- The theorem stating that under the given conditions, 
    the amount of raisins in the mixture is 4 pounds -/
theorem raisins_amount (m : Mixture) : 
  problem_conditions m → m.raisins = 4 := by sorry

end NUMINAMATH_CALUDE_raisins_amount_l3476_347669


namespace NUMINAMATH_CALUDE_excess_value_proof_l3476_347613

def two_digit_number : ℕ := 57

def tens_digit (n : ℕ) : ℕ := n / 10
def ones_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ := tens_digit n + ones_digit n

def reversed_number (n : ℕ) : ℕ := 10 * (ones_digit n) + tens_digit n

theorem excess_value_proof :
  ∃ (v : ℕ), two_digit_number = 4 * (sum_of_digits two_digit_number) + v ∧
  two_digit_number + 18 = reversed_number two_digit_number ∧
  v = 9 := by
  sorry

end NUMINAMATH_CALUDE_excess_value_proof_l3476_347613


namespace NUMINAMATH_CALUDE_smallest_integer_ending_in_9_divisible_by_13_l3476_347649

theorem smallest_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 9 ∧ n % 13 = 0 ∧
  ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 13 = 0 → m ≥ n :=
by
  use 39
  sorry

end NUMINAMATH_CALUDE_smallest_integer_ending_in_9_divisible_by_13_l3476_347649


namespace NUMINAMATH_CALUDE_proportional_function_m_value_l3476_347663

/-- A proportional function passing through a specific point -/
def proportional_function_through_point (k m : ℝ) : Prop :=
  4 * 2 = 3 - m

/-- Theorem: If the proportional function y = 4x passes through (2, 3-m), then m = -5 -/
theorem proportional_function_m_value (m : ℝ) :
  proportional_function_through_point 4 m → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_m_value_l3476_347663


namespace NUMINAMATH_CALUDE_count_descending_digit_numbers_l3476_347695

/-- The number of natural numbers with 2 or more digits where each subsequent digit is less than the previous one -/
def descending_digit_numbers : ℕ :=
  (Finset.range 9).sum (fun k => Nat.choose 10 (k + 2))

/-- Theorem stating that the number of natural numbers with 2 or more digits 
    where each subsequent digit is less than the previous one is 1013 -/
theorem count_descending_digit_numbers : descending_digit_numbers = 1013 := by
  sorry

end NUMINAMATH_CALUDE_count_descending_digit_numbers_l3476_347695


namespace NUMINAMATH_CALUDE_B_3_2_eq_4_l3476_347606

def B : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => B m 2
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem B_3_2_eq_4 : B 3 2 = 4 := by sorry

end NUMINAMATH_CALUDE_B_3_2_eq_4_l3476_347606


namespace NUMINAMATH_CALUDE_parabola_directrix_l3476_347637

/-- Given a parabola with equation x = (1/8)y^2, its directrix has equation x = -2 -/
theorem parabola_directrix (x y : ℝ) :
  (x = (1/8) * y^2) → (∃ (p : ℝ), p > 0 ∧ x = (1/(4*p)) * y^2 ∧ -p = -2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3476_347637


namespace NUMINAMATH_CALUDE_required_third_subject_score_l3476_347630

def average_score_two_subjects : ℝ := 88
def target_average_three_subjects : ℝ := 90
def number_of_subjects : ℕ := 3

theorem required_third_subject_score :
  let total_score_two_subjects := average_score_two_subjects * 2
  let total_score_three_subjects := target_average_three_subjects * number_of_subjects
  total_score_three_subjects - total_score_two_subjects = 94 := by
  sorry

end NUMINAMATH_CALUDE_required_third_subject_score_l3476_347630


namespace NUMINAMATH_CALUDE_initial_distance_between_trucks_l3476_347657

/-- Theorem: Initial distance between two trucks
Given:
- Two trucks X and Y traveling in the same direction
- Truck X's speed is 47 mph
- Truck Y's speed is 53 mph
- It takes 3 hours for Truck Y to overtake and be 5 miles ahead of Truck X
Prove: The initial distance between Truck X and Truck Y is 23 miles
-/
theorem initial_distance_between_trucks
  (speed_x : ℝ)
  (speed_y : ℝ)
  (overtake_time : ℝ)
  (ahead_distance : ℝ)
  (h1 : speed_x = 47)
  (h2 : speed_y = 53)
  (h3 : overtake_time = 3)
  (h4 : ahead_distance = 5)
  : ∃ (initial_distance : ℝ),
    initial_distance = (speed_y - speed_x) * overtake_time + ahead_distance :=
by
  sorry

end NUMINAMATH_CALUDE_initial_distance_between_trucks_l3476_347657


namespace NUMINAMATH_CALUDE_line_parameterization_l3476_347625

/-- Given a line y = 2x - 40 parameterized by (x, y) = (g(t), 20t - 14),
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) :
  (∀ x y, y = 2 * x - 40 ↔ ∃ t, x = g t ∧ y = 20 * t - 14) →
  ∀ t, g t = 10 * t + 13 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3476_347625


namespace NUMINAMATH_CALUDE_largest_touching_sphere_radius_l3476_347675

/-- A regular tetrahedron inscribed in a unit sphere -/
structure InscribedTetrahedron where
  /-- The tetrahedron is regular -/
  isRegular : Bool
  /-- The tetrahedron is inscribed in a unit sphere -/
  isInscribed : Bool

/-- A sphere touching the unit sphere internally and the tetrahedron externally -/
structure TouchingSphere where
  /-- The radius of the sphere -/
  radius : ℝ
  /-- The sphere touches the unit sphere internally -/
  touchesUnitSphereInternally : Bool
  /-- The sphere touches the tetrahedron externally -/
  touchesTetrahedronExternally : Bool

/-- The theorem stating the radius of the largest touching sphere -/
theorem largest_touching_sphere_radius 
  (t : InscribedTetrahedron) 
  (s : TouchingSphere) 
  (h1 : t.isRegular = true) 
  (h2 : t.isInscribed = true)
  (h3 : s.touchesUnitSphereInternally = true)
  (h4 : s.touchesTetrahedronExternally = true) :
  s.radius = 1/3 :=
sorry

end NUMINAMATH_CALUDE_largest_touching_sphere_radius_l3476_347675


namespace NUMINAMATH_CALUDE_pyramid_base_edge_length_l3476_347660

/-- A square pyramid with a hemisphere resting on top -/
structure PyramidWithHemisphere where
  /-- Height of the pyramid -/
  pyramidHeight : ℝ
  /-- Radius of the hemisphere -/
  hemisphereRadius : ℝ
  /-- The hemisphere is tangent to each of the four lateral faces of the pyramid -/
  isTangent : Bool

/-- Calculates the edge-length of the square base of the pyramid -/
def baseEdgeLength (p : PyramidWithHemisphere) : ℝ :=
  sorry

/-- Theorem stating that for a pyramid of height 9 cm and a hemisphere of radius 3 cm,
    the edge-length of the square base is 4.5 cm -/
theorem pyramid_base_edge_length :
  let p : PyramidWithHemisphere := {
    pyramidHeight := 9,
    hemisphereRadius := 3,
    isTangent := true
  }
  baseEdgeLength p = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_edge_length_l3476_347660


namespace NUMINAMATH_CALUDE_max_alpha_squared_l3476_347648

theorem max_alpha_squared (a b x y : ℝ) : 
  a > 0 → 
  b > 0 → 
  a = 2 * b → 
  0 ≤ x → 
  x < a → 
  0 ≤ y → 
  y < b → 
  a^2 + y^2 = b^2 + x^2 → 
  b^2 + x^2 = (a - x)^2 + (b - y)^2 → 
  (a / b)^2 ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_alpha_squared_l3476_347648


namespace NUMINAMATH_CALUDE_equation_system_solution_l3476_347676

/-- A system of 1000 equations where each x_i^2 = a * x_{i+1} + 1, with x_1000 wrapping back to x_1 -/
def EquationSystem (a : ℝ) (x : Fin 1000 → ℝ) : Prop :=
  ∀ i : Fin 1000, x i ^ 2 = a * x (i.succ) + 1

/-- The solutions to the equation system -/
def Solutions (a : ℝ) : Set ℝ :=
  {x | x = (a + Real.sqrt (a^2 + 4)) / 2 ∨ x = (a - Real.sqrt (a^2 + 4)) / 2}

theorem equation_system_solution (a : ℝ) (ha : |a| > 1) :
  ∀ x : Fin 1000 → ℝ, EquationSystem a x ↔ (∀ i, x i ∈ Solutions a) := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3476_347676


namespace NUMINAMATH_CALUDE_trapezoid_longer_side_length_l3476_347680

/-- Given a square with side length 1, divided into one triangle and three trapezoids
    by joining the center to points on each side, where these points divide each side
    into segments of length 1/4 and 3/4, and each section has equal area,
    prove that the length of the longer parallel side of the trapezoids is 3/4. -/
theorem trapezoid_longer_side_length (square_side : ℝ) (segment_short : ℝ) (segment_long : ℝ)
  (h_square_side : square_side = 1)
  (h_segment_short : segment_short = 1/4)
  (h_segment_long : segment_long = 3/4)
  (h_segments_sum : segment_short + segment_long = square_side)
  (h_equal_areas : ∀ section_area : ℝ, section_area = (square_side^2) / 4) :
  ∃ x : ℝ, x = 3/4 ∧ x = segment_long :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_longer_side_length_l3476_347680


namespace NUMINAMATH_CALUDE_min_value_expression_l3476_347692

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2023) + (y + 1/x) * (y + 1/x - 2023) ≥ -2050513 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3476_347692


namespace NUMINAMATH_CALUDE_mandy_gets_fifteen_l3476_347632

def chocolate_bar : ℕ := 60

def michael_share (total : ℕ) : ℕ := total / 2

def paige_share (remaining : ℕ) : ℕ := remaining / 2

def mandy_share (total : ℕ) : ℕ :=
  let after_michael := total - michael_share total
  after_michael - paige_share after_michael

theorem mandy_gets_fifteen :
  mandy_share chocolate_bar = 15 := by
  sorry

end NUMINAMATH_CALUDE_mandy_gets_fifteen_l3476_347632


namespace NUMINAMATH_CALUDE_no_perfect_square_sum_l3476_347602

theorem no_perfect_square_sum (n : ℕ) : n ≥ 1 → ¬∃ (m : ℕ), 2^n + 12^n + 2014^n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_sum_l3476_347602


namespace NUMINAMATH_CALUDE_selling_price_ratio_l3476_347635

theorem selling_price_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l3476_347635


namespace NUMINAMATH_CALUDE_max_trees_cut_2001_l3476_347604

/-- Represents a square grid of trees -/
structure TreeGrid where
  size : Nat
  is_square : size * size = size * size

/-- Represents the maximum number of trees that can be cut down -/
def max_trees_cut (grid : TreeGrid) : Nat :=
  (grid.size / 2) * (grid.size / 2) + 1

/-- The theorem to be proved -/
theorem max_trees_cut_2001 :
  ∀ (grid : TreeGrid),
    grid.size = 2001 →
    max_trees_cut grid = 1001001 := by
  sorry

end NUMINAMATH_CALUDE_max_trees_cut_2001_l3476_347604


namespace NUMINAMATH_CALUDE_robotics_club_mentor_age_l3476_347656

theorem robotics_club_mentor_age (total_members : ℕ) (avg_age : ℕ) 
  (num_boys num_girls num_mentors : ℕ) (avg_age_boys avg_age_girls : ℕ) :
  total_members = 50 →
  avg_age = 20 →
  num_boys = 25 →
  num_girls = 20 →
  num_mentors = 5 →
  avg_age_boys = 18 →
  avg_age_girls = 19 →
  (total_members * avg_age - num_boys * avg_age_boys - num_girls * avg_age_girls) / num_mentors = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_robotics_club_mentor_age_l3476_347656


namespace NUMINAMATH_CALUDE_increasing_interval_of_f_l3476_347619

-- Define the function
def f (x : ℝ) : ℝ := -(x - 2) * x

-- State the theorem
theorem increasing_interval_of_f :
  ∀ x y : ℝ, x < y ∧ x < 1 ∧ y < 1 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_increasing_interval_of_f_l3476_347619


namespace NUMINAMATH_CALUDE_frame_width_l3476_347698

/-- Given a frame with three square photo openings, this theorem proves that
    the width of the frame is 5 cm under the specified conditions. -/
theorem frame_width (s : ℝ) (d : ℝ) : 
  s > 0 →  -- side length of square opening is positive
  d > 0 →  -- frame width is positive
  4 * s = 60 →  -- perimeter of one photo opening
  2 * ((3 * s + 4 * d) + (s + 2 * d)) = 180 →  -- total perimeter of the frame
  d = 5 := by
  sorry

end NUMINAMATH_CALUDE_frame_width_l3476_347698


namespace NUMINAMATH_CALUDE_sapling_planting_equation_l3476_347642

theorem sapling_planting_equation (x : ℤ) : 
  (∀ (total : ℤ), (5 * x + 3 = total) ↔ (6 * x = total + 4)) :=
by sorry

end NUMINAMATH_CALUDE_sapling_planting_equation_l3476_347642


namespace NUMINAMATH_CALUDE_yoongi_subtraction_l3476_347651

theorem yoongi_subtraction (A B C : Nat) (h1 : A ≥ 1) (h2 : A ≤ 9) (h3 : B ≤ 9) (h4 : C ≤ 9) :
  (1000 * A + 100 * B + 10 * C + 6) - 57 = 1819 →
  (1000 * A + 100 * B + 10 * C + 9) - 57 = 1822 := by
sorry

end NUMINAMATH_CALUDE_yoongi_subtraction_l3476_347651


namespace NUMINAMATH_CALUDE_unique_composite_with_bounded_divisors_l3476_347628

def isComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def isProperDivisor (d n : ℕ) : Prop :=
  1 < d ∧ d < n ∧ n % d = 0

theorem unique_composite_with_bounded_divisors :
  ∃! n : ℕ, isComposite n ∧
    (∀ d : ℕ, isProperDivisor d n → n - 12 ≥ d ∧ d ≥ n - 20) ∧
    n = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_composite_with_bounded_divisors_l3476_347628


namespace NUMINAMATH_CALUDE_line_direction_vector_value_l3476_347638

def point := ℝ × ℝ

def direction_vector (a : ℝ) : point := (a, -2)

def line_passes_through (p1 p2 : point) (v : point) : Prop :=
  ∃ t : ℝ, p2 = (p1.1 + t * v.1, p1.2 + t * v.2)

theorem line_direction_vector_value :
  ∀ a : ℝ,
  line_passes_through (-3, 6) (2, -1) (direction_vector a) →
  a = 10/7 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_value_l3476_347638


namespace NUMINAMATH_CALUDE_poverty_education_relationship_l3476_347684

/-- Regression line for poverty and education data -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Define a point on the regression line -/
structure Point where
  x : ℝ
  y : ℝ

/-- The regression line equation -/
def on_regression_line (line : RegressionLine) (p : Point) : Prop :=
  p.y = line.slope * p.x + line.intercept

theorem poverty_education_relationship (line : RegressionLine) 
    (h_slope : line.slope = 0.8) (h_intercept : line.intercept = 4.6)
    (p1 p2 : Point) (h_on_line1 : on_regression_line line p1) 
    (h_on_line2 : on_regression_line line p2) (h_x_diff : p2.x - p1.x = 1) :
    p2.y - p1.y = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_poverty_education_relationship_l3476_347684


namespace NUMINAMATH_CALUDE_sqrt_of_four_l3476_347612

theorem sqrt_of_four : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_l3476_347612


namespace NUMINAMATH_CALUDE_always_two_real_roots_specific_case_l3476_347658

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := x^2 + (2-m)*x + (1-m)

-- Theorem stating that the equation always has two real roots
theorem always_two_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0 :=
sorry

-- Theorem for the specific case when m < 0 and the difference between roots is 4
theorem specific_case (m : ℝ) (h₁ : m < 0) :
  (∃ (x₁ x₂ : ℝ), x₁ - x₂ = 4 ∧ quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0) →
  m = -4 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_specific_case_l3476_347658


namespace NUMINAMATH_CALUDE_expression_simplification_l3476_347672

theorem expression_simplification 
  (a c d x y : ℝ) 
  (h : c * x + d * y ≠ 0) : 
  (c * x * (a^2 * x^2 + 3 * a^2 * y^2 + c^2 * y^2) + 
   d * y * (a^2 * x^2 + 3 * c^2 * x^2 + c^2 * y^2)) / 
  (c * x + d * y) = 
  a^2 * x^2 + 3 * a * c * x * y + c^2 * y^2 := by
sorry


end NUMINAMATH_CALUDE_expression_simplification_l3476_347672


namespace NUMINAMATH_CALUDE_betty_herb_garden_total_l3476_347662

/-- The number of basil plants in Betty's herb garden -/
def basil : ℕ := 5

/-- The number of oregano plants in Betty's herb garden -/
def oregano : ℕ := 2 * basil + 2

/-- The total number of plants in Betty's herb garden -/
def total_plants : ℕ := basil + oregano

theorem betty_herb_garden_total : total_plants = 17 := by
  sorry

end NUMINAMATH_CALUDE_betty_herb_garden_total_l3476_347662


namespace NUMINAMATH_CALUDE_binary_linear_equation_sum_l3476_347650

/-- A binary linear equation is an equation where the exponents of all variables are 1. -/
def IsBinaryLinearEquation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y, f x y = a * x + b * y + c

/-- Given that x^(3m-3) - 2y^(n-1) = 5 is a binary linear equation, prove that m + n = 10/3 -/
theorem binary_linear_equation_sum (m n : ℝ) :
  IsBinaryLinearEquation (fun x y => x^(3*m-3) - 2*y^(n-1) - 5) →
  m + n = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_binary_linear_equation_sum_l3476_347650


namespace NUMINAMATH_CALUDE_room_length_calculation_l3476_347671

theorem room_length_calculation (area : ℝ) (width : ℝ) (length : ℝ) :
  area = 12.0 ∧ width = 8.0 ∧ area = width * length → length = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l3476_347671


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_geometric_sequence_general_term_l3476_347677

/-- Geometric sequence -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

theorem geometric_sequence_sixth_term :
  let a₁ := 3
  let q := -2
  geometric_sequence a₁ q 6 = -96 := by sorry

theorem geometric_sequence_general_term :
  let a₃ := 20
  let a₆ := 160
  ∃ q : ℝ, ∀ n : ℕ, geometric_sequence (a₃ / q^2) q n = 5 * 2^(n - 1) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_geometric_sequence_general_term_l3476_347677


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3476_347647

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem largest_power_dividing_factorial :
  (∀ m : ℕ, m > 7 → ¬(divides (18^m) (factorial 30))) ∧
  (divides (18^7) (factorial 30)) := by
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3476_347647


namespace NUMINAMATH_CALUDE_angle_at_point_l3476_347621

theorem angle_at_point (x : ℝ) : 
  x > 0 ∧ x + x + 140 = 360 → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_angle_at_point_l3476_347621


namespace NUMINAMATH_CALUDE_cos_12_cos_18_minus_sin_12_sin_18_l3476_347668

theorem cos_12_cos_18_minus_sin_12_sin_18 :
  Real.cos (12 * π / 180) * Real.cos (18 * π / 180) - 
  Real.sin (12 * π / 180) * Real.sin (18 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_12_cos_18_minus_sin_12_sin_18_l3476_347668


namespace NUMINAMATH_CALUDE_one_pair_three_different_probability_l3476_347654

def total_socks : ℕ := 12
def socks_per_color : ℕ := 3
def num_colors : ℕ := 4
def drawn_socks : ℕ := 5

def probability_one_pair_three_different : ℚ :=
  27 / 66

theorem one_pair_three_different_probability :
  (total_socks = socks_per_color * num_colors) →
  (probability_one_pair_three_different =
    (num_colors * (socks_per_color.choose 2) *
     (socks_per_color ^ (num_colors - 1))) /
    (total_socks.choose drawn_socks)) :=
by sorry

end NUMINAMATH_CALUDE_one_pair_three_different_probability_l3476_347654


namespace NUMINAMATH_CALUDE_min_value_sum_product_l3476_347609

theorem min_value_sum_product (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l3476_347609


namespace NUMINAMATH_CALUDE_barrel_filling_time_l3476_347644

theorem barrel_filling_time (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  y - x = 1/4 ∧ 
  66/y - 40/x = 3 →
  40/x = 5 ∨ 40/x = 96 :=
by sorry

end NUMINAMATH_CALUDE_barrel_filling_time_l3476_347644


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l3476_347688

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l3476_347688


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_equals_twice_perimeter_l3476_347620

/-- Given a triangle with area A, perimeter p, semiperimeter s, and inradius r -/
theorem inscribed_circle_radius_when_area_equals_twice_perimeter 
  (A : ℝ) (p : ℝ) (s : ℝ) (r : ℝ) 
  (h1 : A = 2 * p)  -- Area is twice the perimeter
  (h2 : p = 2 * s)  -- Perimeter is twice the semiperimeter
  (h3 : A = r * s)  -- Area formula for a triangle
  (h4 : s ≠ 0)      -- Semiperimeter is non-zero
  : r = 4 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_equals_twice_perimeter_l3476_347620


namespace NUMINAMATH_CALUDE_family_size_l3476_347607

theorem family_size (purification_cost : ℚ) (water_per_person : ℚ) (family_cost : ℚ) :
  purification_cost = 1 →
  water_per_person = 1/2 →
  family_cost = 3 →
  (family_cost / (purification_cost * water_per_person) : ℚ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_family_size_l3476_347607


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3476_347646

theorem negation_of_proposition (p : Prop) : 
  (¬(∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0)) ↔ 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3476_347646


namespace NUMINAMATH_CALUDE_extraction_of_geometric_from_arithmetic_l3476_347687

-- Define the arithmetic progression
def arithmeticProgression (a b : ℤ) (k : ℤ) : ℤ := a + b * k

-- Define the geometric progression
def geometricProgression (a b : ℤ) (k : ℕ) : ℤ := a * (b + 1)^k

theorem extraction_of_geometric_from_arithmetic (a b : ℤ) :
  ∃ (f : ℕ → ℤ), (∀ k : ℕ, ∃ l : ℤ, geometricProgression a b k = arithmeticProgression a b l) :=
sorry

end NUMINAMATH_CALUDE_extraction_of_geometric_from_arithmetic_l3476_347687


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l3476_347690

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 5}

theorem intersection_complement_equals : A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l3476_347690


namespace NUMINAMATH_CALUDE_bell_ringing_problem_l3476_347664

theorem bell_ringing_problem (S B : ℕ) : 
  S = (1/3 : ℚ) * B + 4 →
  B = 36 →
  S + B = 52 := by sorry

end NUMINAMATH_CALUDE_bell_ringing_problem_l3476_347664


namespace NUMINAMATH_CALUDE_matrix_cube_proof_l3476_347645

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_cube_proof : A ^ 3 = !![(-4), 2; (-2), 1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_cube_proof_l3476_347645


namespace NUMINAMATH_CALUDE_decimal_division_l3476_347641

theorem decimal_division (x y : ℚ) (hx : x = 0.45) (hy : y = 0.005) : x / y = 90 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l3476_347641


namespace NUMINAMATH_CALUDE_spells_base7_to_base10_l3476_347643

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The number of spells in base 7 --/
def spellsBase7 : Nat := 653

/-- Theorem: The number of spells in base 7 (653) is equal to 332 in base 10 --/
theorem spells_base7_to_base10 :
  base7ToBase10 (spellsBase7 / 100) ((spellsBase7 / 10) % 10) (spellsBase7 % 10) = 332 := by
  sorry

end NUMINAMATH_CALUDE_spells_base7_to_base10_l3476_347643


namespace NUMINAMATH_CALUDE_ball_weights_l3476_347600

/-- The weight of a red ball in grams -/
def red_weight : ℝ := sorry

/-- The weight of a yellow ball in grams -/
def yellow_weight : ℝ := sorry

/-- The total weight of 5 red balls and 3 yellow balls in grams -/
def total_weight_1 : ℝ := 5 * red_weight + 3 * yellow_weight

/-- The total weight of 5 yellow balls and 3 red balls in grams -/
def total_weight_2 : ℝ := 5 * yellow_weight + 3 * red_weight

theorem ball_weights :
  total_weight_1 = 42 ∧ total_weight_2 = 38 → red_weight = 6 ∧ yellow_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_ball_weights_l3476_347600


namespace NUMINAMATH_CALUDE_number_subtraction_l3476_347696

theorem number_subtraction (x : ℤ) : x + 30 = 55 → x - 23 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_l3476_347696


namespace NUMINAMATH_CALUDE_prob_b_leads_2to1_expected_score_b_l3476_347673

/-- Represents a table tennis game between player A and player B -/
structure TableTennisGame where
  /-- Probability of the server scoring a point -/
  serverWinProb : ℝ
  /-- Player A serves first -/
  aServesFirst : Bool

/-- Calculates the probability of player B leading 2-1 at the start of the fourth serve -/
def probBLeads2to1 (game : TableTennisGame) : ℝ := sorry

/-- Calculates the expected score of player B at the start of the fourth serve -/
def expectedScoreB (game : TableTennisGame) : ℝ := sorry

/-- Theorem stating the probability of player B leading 2-1 at the start of the fourth serve -/
theorem prob_b_leads_2to1 (game : TableTennisGame) 
  (h1 : game.serverWinProb = 0.6) 
  (h2 : game.aServesFirst = true) : 
  probBLeads2to1 game = 0.352 := by sorry

/-- Theorem stating the expected score of player B at the start of the fourth serve -/
theorem expected_score_b (game : TableTennisGame) 
  (h1 : game.serverWinProb = 0.6) 
  (h2 : game.aServesFirst = true) : 
  expectedScoreB game = 1.400 := by sorry

end NUMINAMATH_CALUDE_prob_b_leads_2to1_expected_score_b_l3476_347673


namespace NUMINAMATH_CALUDE_square_root_of_nine_l3476_347674

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l3476_347674


namespace NUMINAMATH_CALUDE_flag_distribution_l3476_347699

theorem flag_distribution (F : ℕ) (blue_flags red_flags : ℕ) :
  F % 2 = 0 →
  F = blue_flags + red_flags →
  blue_flags ≥ (3 * F) / 10 →
  red_flags ≥ F / 4 →
  (F / 2 - (3 * F) / 10 - F / 4) / (F / 2) = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_flag_distribution_l3476_347699


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3476_347682

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Theorem statement
theorem circle_center_and_radius :
  ∃ (center_x center_y radius : ℝ),
    (center_x = 2 ∧ center_y = 0 ∧ radius = 2) ∧
    (∀ (x y : ℝ), circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3476_347682


namespace NUMINAMATH_CALUDE_bird_count_theorem_l3476_347629

/-- The number of birds on a fence after a series of additions and removals --/
def final_bird_count (initial : ℕ) (first_add : ℕ) (first_remove : ℕ) (second_add : ℕ) (third_add : ℚ) : ℚ :=
  let T : ℕ := initial + first_add
  let W : ℕ := T - first_remove + second_add
  (W : ℚ) / 2 + third_add

/-- Theorem stating the final number of birds on the fence --/
theorem bird_count_theorem : 
  final_bird_count 12 8 5 3 (5/2) = 23/2 := by sorry

end NUMINAMATH_CALUDE_bird_count_theorem_l3476_347629


namespace NUMINAMATH_CALUDE_area_between_derivative_and_x_axis_l3476_347666

open Real

noncomputable def f (x : ℝ) : ℝ := log x + x^2 - 3*x

theorem area_between_derivative_and_x_axis : 
  ∫ (x : ℝ) in (1/2)..1, (1/x + 2*x - 3) = -(3/4 - log 2) :=
sorry

end NUMINAMATH_CALUDE_area_between_derivative_and_x_axis_l3476_347666


namespace NUMINAMATH_CALUDE_find_missing_number_l3476_347685

/-- Given two sets of numbers with known means, find the missing number in the second set. -/
theorem find_missing_number (x : ℝ) (missing : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (128 + 255 + 511 + missing + x) / 5 = 398.2 →
  missing = 1023 := by
sorry

end NUMINAMATH_CALUDE_find_missing_number_l3476_347685


namespace NUMINAMATH_CALUDE_triangle_inequality_l3476_347697

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) (h1 : C ≥ π / 3) :
  let s := (a + b + c) / 2
  (a + b) * (1 / a + 1 / b + 1 / c) ≥ 4 + 1 / Real.sin (C / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3476_347697


namespace NUMINAMATH_CALUDE_sampling_methods_correct_l3476_347617

/-- Represents a sampling method --/
inductive SamplingMethod
  | Systematic
  | SimpleRandom

/-- Represents a scenario for sampling --/
structure SamplingScenario where
  description : String
  interval : Option ℕ
  sampleSize : ℕ
  populationSize : ℕ

/-- Determines the appropriate sampling method for a given scenario --/
def determineSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- The milk production line scenario --/
def milkProductionScenario : SamplingScenario :=
  { description := "Milk production line inspection"
  , interval := some 30
  , sampleSize := 1
  , populationSize := 0 }

/-- The math enthusiasts scenario --/
def mathEnthusiastsScenario : SamplingScenario :=
  { description := "Math enthusiasts study load"
  , interval := none
  , sampleSize := 3
  , populationSize := 30 }

theorem sampling_methods_correct :
  determineSamplingMethod milkProductionScenario = SamplingMethod.Systematic ∧
  determineSamplingMethod mathEnthusiastsScenario = SamplingMethod.SimpleRandom :=
  sorry

end NUMINAMATH_CALUDE_sampling_methods_correct_l3476_347617


namespace NUMINAMATH_CALUDE_children_still_hiding_l3476_347694

theorem children_still_hiding (total : ℕ) (found : ℕ) (seeker : ℕ) : 
  total = 16 → found = 6 → seeker = 1 → total - found - seeker = 9 := by
sorry

end NUMINAMATH_CALUDE_children_still_hiding_l3476_347694


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l3476_347681

theorem mystery_book_shelves (books_per_shelf : ℕ) (picture_book_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 6 →
  picture_book_shelves = 4 →
  total_books = 54 →
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 5 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l3476_347681


namespace NUMINAMATH_CALUDE_road_repair_hours_l3476_347639

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 33)
  (h2 : days1 = 12)
  (h3 : people2 = 30)
  (h4 : days2 = 11)
  (h5 : hours2 = 6)
  (h6 : people1 * days1 * (people1 * days1).lcm (people2 * days2 * hours2) / (people1 * days1) = 
        people2 * days2 * hours2 * (people1 * days1).lcm (people2 * days2 * hours2) / (people2 * days2 * hours2)) :
  (people1 * days1).lcm (people2 * days2 * hours2) / (people1 * days1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_hours_l3476_347639


namespace NUMINAMATH_CALUDE_can_repair_propeller_l3476_347614

/-- The cost of a blade in tugriks -/
def blade_cost : ℕ := 120

/-- The cost of a screw in tugriks -/
def screw_cost : ℕ := 9

/-- The discount rate applied after spending 250 tugriks -/
def discount_rate : ℚ := 0.2

/-- The threshold for applying the discount -/
def discount_threshold : ℕ := 250

/-- Karlsson's budget in tugriks -/
def budget : ℕ := 360

/-- The number of blades needed -/
def blades_needed : ℕ := 3

/-- The number of screws needed -/
def screws_needed : ℕ := 1

/-- Function to calculate the total cost with discount -/
def total_cost_with_discount (blade_cost screw_cost : ℕ) (discount_rate : ℚ) 
  (discount_threshold blades_needed screws_needed : ℕ) : ℚ :=
  let initial_purchase := 2 * blade_cost + 2 * screw_cost
  let remaining_purchase := blade_cost
  if initial_purchase ≥ discount_threshold
  then initial_purchase + remaining_purchase * (1 - discount_rate)
  else initial_purchase + remaining_purchase

/-- Theorem stating that Karlsson can afford to repair his propeller -/
theorem can_repair_propeller : 
  total_cost_with_discount blade_cost screw_cost discount_rate 
    discount_threshold blades_needed screws_needed ≤ budget := by
  sorry

end NUMINAMATH_CALUDE_can_repair_propeller_l3476_347614


namespace NUMINAMATH_CALUDE_frog_jumps_l3476_347611

/-- A jump sequence represents the frog's movements, where
    true represents a jump to the right and false represents a jump to the left. -/
def JumpSequence := List Bool

/-- The position after following a jump sequence -/
def position (p q : ℕ) (jumps : JumpSequence) : ℤ :=
  jumps.foldl (λ acc jump => if jump then acc + p else acc - q) 0

/-- A jump sequence is valid if it starts and ends at 0 -/
def is_valid_sequence (p q : ℕ) (jumps : JumpSequence) : Prop :=
  position p q jumps = 0

theorem frog_jumps (p q : ℕ) (jumps : JumpSequence) (d : ℕ) :
  Nat.Coprime p q →
  is_valid_sequence p q jumps →
  d < p + q →
  ∃ (i j : ℕ), i < jumps.length ∧ j < jumps.length ∧
    abs (position p q (jumps.take i) - position p q (jumps.take j)) = d :=
sorry

end NUMINAMATH_CALUDE_frog_jumps_l3476_347611


namespace NUMINAMATH_CALUDE_plane_equation_proof_l3476_347670

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by parametric equations -/
structure Line3D where
  t : ℝ → Point3D

/-- A plane in 3D space defined by Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def lineInPlane (plane : Plane) (line : Line3D) : Prop :=
  ∀ t : ℝ, pointOnPlane plane (line.t t)

/-- The given point (1,4,-3) -/
def givenPoint : Point3D := ⟨1, 4, -3⟩

/-- The given line (x - 2)/4 = (y + 1)/(-1) = (z - 3)/2 -/
def givenLine : Line3D :=
  ⟨λ t : ℝ => ⟨4*t + 2, -t - 1, 2*t + 3⟩⟩

/-- The plane we want to prove -/
def targetPlane : Plane := ⟨5, -22, -21, 41⟩

theorem plane_equation_proof :
  (pointOnPlane targetPlane givenPoint) ∧
  (lineInPlane targetPlane givenLine) ∧
  (targetPlane.A > 0) ∧
  (Nat.gcd (Nat.gcd (Int.natAbs targetPlane.A) (Int.natAbs targetPlane.B))
           (Nat.gcd (Int.natAbs targetPlane.C) (Int.natAbs targetPlane.D)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l3476_347670


namespace NUMINAMATH_CALUDE_dodecagon_side_length_l3476_347661

theorem dodecagon_side_length (d : ℝ) (a₁₂ : ℝ) (h : d > 0) :
  a₁₂ = (d / 2) * Real.sqrt (2 - Real.sqrt 3) ↔
  a₁₂^2 = ((d / 2) - Real.sqrt ((d / 2)^2 - (d / 4)^2))^2 + (d / 4)^2 :=
by sorry

end NUMINAMATH_CALUDE_dodecagon_side_length_l3476_347661


namespace NUMINAMATH_CALUDE_triangle_side_length_l3476_347653

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Given conditions
  a = Real.sqrt 3 →
  2 * (Real.cos ((A + C) / 2))^2 = (Real.sqrt 2 - 1) * Real.cos B →
  A = π / 3 →
  -- Conclusion
  c = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3476_347653


namespace NUMINAMATH_CALUDE_average_age_of_women_l3476_347667

theorem average_age_of_women (A : ℝ) (n : ℕ) : 
  n = 9 → 
  n * (A + 4) - (n * A - (36 + 32) + 104) = 0 → 
  104 / 2 = 52 :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_women_l3476_347667


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3476_347616

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem fifteenth_term_of_sequence (a₁ a₂ : ℤ) (h : a₂ = a₁ + 1) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 53 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3476_347616


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3476_347683

/-- Two real numbers are inversely proportional -/
def InverselyProportional (a b : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a * b = k

theorem inverse_proportion_problem (a₁ a₂ b₁ b₂ : ℝ) 
  (h_inverse : InverselyProportional a₁ b₁) 
  (h_initial : a₁ = 40 ∧ b₁ = 8) 
  (h_final : b₂ = 10) : 
  a₂ = 32 ∧ InverselyProportional a₂ b₂ :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3476_347683


namespace NUMINAMATH_CALUDE_quartic_roots_product_l3476_347605

theorem quartic_roots_product (x y z w : ℝ) 
  (sum_zero : x + y + z + w = 0)
  (sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quartic_roots_product_l3476_347605


namespace NUMINAMATH_CALUDE_battle_station_staffing_l3476_347624

/-- The number of ways to assign n distinct objects to k distinct positions,
    where each position must be filled by exactly one object. -/
def permutations (n k : ℕ) : ℕ := Nat.descFactorial n k

/-- The number of job openings -/
def num_jobs : ℕ := 6

/-- The number of suitable candidates -/
def num_candidates : ℕ := 15

theorem battle_station_staffing :
  permutations num_candidates num_jobs = 3276000 := by sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l3476_347624


namespace NUMINAMATH_CALUDE_tabletennis_arrangements_eq_252_l3476_347610

/-- The number of ways to arrange 5 players from a team of 10, 
    where 3 specific players must occupy positions 1, 3, and 5, 
    and 2 players from the remaining 7 must occupy positions 2 and 4 -/
def tabletennis_arrangements (total_players : ℕ) (main_players : ℕ) 
    (players_to_send : ℕ) (remaining_players : ℕ) : ℕ := 
  Nat.factorial main_players * (remaining_players * (remaining_players - 1))

theorem tabletennis_arrangements_eq_252 : 
  tabletennis_arrangements 10 3 5 7 = 252 := by
  sorry

#eval tabletennis_arrangements 10 3 5 7

end NUMINAMATH_CALUDE_tabletennis_arrangements_eq_252_l3476_347610


namespace NUMINAMATH_CALUDE_sugar_flour_difference_l3476_347633

-- Define constants based on the problem conditions
def flour_recipe : Real := 2.25  -- kg
def sugar_recipe : Real := 5.5   -- lb
def flour_added : Real := 1      -- kg
def kg_to_lb : Real := 2.205     -- 1 kg = 2.205 lb
def kg_to_g : Real := 1000       -- 1 kg = 1000 g

-- Theorem statement
theorem sugar_flour_difference :
  let flour_remaining := (flour_recipe - flour_added) * kg_to_g
  let sugar_needed := (sugar_recipe / kg_to_lb) * kg_to_g
  ∃ ε > 0, abs (sugar_needed - flour_remaining - 1244.8) < ε :=
by sorry

end NUMINAMATH_CALUDE_sugar_flour_difference_l3476_347633


namespace NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l3476_347686

theorem greatest_power_of_three_in_factorial :
  (∃ n : ℕ, n = 6 ∧ 
   ∀ k : ℕ, 3^k ∣ Nat.factorial 16 → k ≤ n) ∧
   3^6 ∣ Nat.factorial 16 :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l3476_347686


namespace NUMINAMATH_CALUDE_prime_pairs_perfect_square_l3476_347652

theorem prime_pairs_perfect_square :
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (∃ a : ℕ, p^2 + p*q + q^2 = a^2) → 
    ((p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_perfect_square_l3476_347652


namespace NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l3476_347678

def range_start : ℕ := 1
def range_end : ℕ := 60
def multiples_of_four : ℕ := 15

theorem probability_at_least_one_multiple_of_four :
  let total_numbers := range_end - range_start + 1
  let non_multiples := total_numbers - multiples_of_four
  let prob_neither_multiple := (non_multiples / total_numbers) ^ 2
  1 - prob_neither_multiple = 7 / 16 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l3476_347678


namespace NUMINAMATH_CALUDE_light_flash_duration_l3476_347608

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The interval between flashes in seconds -/
def flash_interval : ℕ := 20

/-- The number of flashes -/
def num_flashes : ℕ := 180

/-- Theorem: The time it takes for 180 flashes of a light that flashes every 20 seconds is equal to 1 hour -/
theorem light_flash_duration : 
  (flash_interval * num_flashes) = seconds_per_hour := by sorry

end NUMINAMATH_CALUDE_light_flash_duration_l3476_347608


namespace NUMINAMATH_CALUDE_orphanage_donation_percentage_l3476_347603

def total_income : ℝ := 200000
def children_percentage : ℝ := 0.15
def num_children : ℕ := 3
def wife_percentage : ℝ := 0.30
def final_amount : ℝ := 40000

theorem orphanage_donation_percentage :
  let children_total_percentage := children_percentage * num_children
  let total_given_percentage := children_total_percentage + wife_percentage
  let remaining_amount := total_income * (1 - total_given_percentage)
  let donated_amount := remaining_amount - final_amount
  donated_amount / remaining_amount = 0.20 := by
sorry

end NUMINAMATH_CALUDE_orphanage_donation_percentage_l3476_347603


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3476_347655

theorem cube_surface_area_from_volume (volume : ℝ) (side_length : ℝ) (surface_area : ℝ) : 
  volume = 729 → 
  volume = side_length ^ 3 → 
  surface_area = 6 * side_length ^ 2 → 
  surface_area = 486 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3476_347655
