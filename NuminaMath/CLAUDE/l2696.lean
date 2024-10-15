import Mathlib

namespace NUMINAMATH_CALUDE_parallel_reasoning_is_deductive_l2696_269671

-- Define a type for lines
structure Line : Type :=
  (id : ℕ)

-- Define a parallel relation between lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the property of transitivity for parallel lines
axiom parallel_transitive : ∀ (x y z : Line), parallel x y → parallel y z → parallel x z

-- Given lines a, b, and c
variable (a b c : Line)

-- Given that a is parallel to b, and b is parallel to c
axiom a_parallel_b : parallel a b
axiom b_parallel_c : parallel b c

-- Define deductive reasoning
def is_deductive_reasoning (conclusion : Prop) : Prop := sorry

-- Theorem to prove
theorem parallel_reasoning_is_deductive : 
  is_deductive_reasoning (parallel a c) := sorry

end NUMINAMATH_CALUDE_parallel_reasoning_is_deductive_l2696_269671


namespace NUMINAMATH_CALUDE_gummies_cost_gummies_cost_proof_l2696_269669

theorem gummies_cost (lollipop_count : ℕ) (lollipop_price : ℚ) 
                      (gummies_count : ℕ) (initial_amount : ℚ) 
                      (remaining_amount : ℚ) : ℚ :=
  let total_spent := initial_amount - remaining_amount
  let lollipop_total := ↑lollipop_count * lollipop_price
  let gummies_total := total_spent - lollipop_total
  gummies_total / ↑gummies_count

#check gummies_cost 4 (3/2) 2 15 5 = 2

theorem gummies_cost_proof :
  gummies_cost 4 (3/2) 2 15 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gummies_cost_gummies_cost_proof_l2696_269669


namespace NUMINAMATH_CALUDE_volume_of_specific_cuboid_l2696_269655

/-- The volume of a cuboid with given edge lengths. -/
def cuboid_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a cuboid with edges 2 cm, 5 cm, and 3 cm is 30 cubic centimeters. -/
theorem volume_of_specific_cuboid : 
  cuboid_volume 2 5 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_cuboid_l2696_269655


namespace NUMINAMATH_CALUDE_max_boxes_of_paint_A_l2696_269662

/-- The maximum number of boxes of paint A that can be purchased given the conditions -/
theorem max_boxes_of_paint_A : ℕ :=
  let price_A : ℕ := 24  -- Price of paint A in yuan
  let price_B : ℕ := 16  -- Price of paint B in yuan
  let total_boxes : ℕ := 200  -- Total number of boxes to be purchased
  let max_cost : ℕ := 3920  -- Maximum total cost in yuan
  let max_A : ℕ := 90  -- Maximum number of boxes of paint A (to be proved)

  have h1 : price_A + 2 * price_B = 56 := by sorry
  have h2 : 2 * price_A + price_B = 64 := by sorry
  have h3 : ∀ m : ℕ, m ≤ total_boxes → 
    price_A * m + price_B * (total_boxes - m) ≤ max_cost → 
    m ≤ max_A := by sorry

  max_A

end NUMINAMATH_CALUDE_max_boxes_of_paint_A_l2696_269662


namespace NUMINAMATH_CALUDE_tangent_line_parabola_l2696_269647

/-- The value of d for which the line y = 3x + d is tangent to the parabola y^2 = 12x -/
theorem tangent_line_parabola : 
  ∃ d : ℝ, (∀ x y : ℝ, y = 3*x + d ∧ y^2 = 12*x → 
    ∃! x₀ : ℝ, 3*x₀ + d = (12*x₀).sqrt ∧ 
    ∀ x : ℝ, x ≠ x₀ → 3*x + d ≠ (12*x).sqrt) → 
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_parabola_l2696_269647


namespace NUMINAMATH_CALUDE_special_polygon_properties_l2696_269609

/-- A polygon where the sum of interior angles is twice the sum of exterior angles -/
structure SpecialPolygon where
  sides : ℕ
  sum_interior_angles : ℝ
  sum_exterior_angles : ℝ
  interior_exterior_relation : sum_interior_angles = 2 * sum_exterior_angles

theorem special_polygon_properties (p : SpecialPolygon) :
  p.sum_interior_angles = 720 ∧ p.sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_properties_l2696_269609


namespace NUMINAMATH_CALUDE_charlie_calculator_problem_l2696_269683

theorem charlie_calculator_problem :
  let original_factor1 : ℚ := 75 / 10000
  let original_factor2 : ℚ := 256 / 10
  let incorrect_result : ℕ := 19200
  (original_factor1 * original_factor2 = 192 / 1000) ∧
  (75 * 256 = incorrect_result) := by
  sorry

end NUMINAMATH_CALUDE_charlie_calculator_problem_l2696_269683


namespace NUMINAMATH_CALUDE_certain_seconds_proof_l2696_269601

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes given in the problem -/
def given_minutes : ℕ := 6

/-- The first ratio number given in the problem -/
def ratio_1 : ℕ := 18

/-- The second ratio number given in the problem -/
def ratio_2 : ℕ := 9

/-- The certain number of seconds we need to find -/
def certain_seconds : ℕ := 720

theorem certain_seconds_proof : 
  (ratio_1 : ℚ) / certain_seconds = ratio_2 / (given_minutes * seconds_per_minute) :=
sorry

end NUMINAMATH_CALUDE_certain_seconds_proof_l2696_269601


namespace NUMINAMATH_CALUDE_same_solution_implies_value_l2696_269621

theorem same_solution_implies_value (a b : ℝ) :
  (∃ x y : ℝ, 5 * x + y = 3 ∧ a * x + 5 * y = 4 ∧ x - 2 * y = 5 ∧ 5 * x + b * y = 1) →
  1/2 * a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_value_l2696_269621


namespace NUMINAMATH_CALUDE_cost_per_person_l2696_269604

theorem cost_per_person (num_friends : ℕ) (total_cost : ℚ) (cost_per_person : ℚ) : 
  num_friends = 15 → 
  total_cost = 13500 → 
  cost_per_person = total_cost / num_friends → 
  cost_per_person = 900 := by
sorry

end NUMINAMATH_CALUDE_cost_per_person_l2696_269604


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2696_269617

/-- A line in the form kx - y - k + 1 = 0 passes through the point (1, 1) for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * 1 - 1 - k + 1 = 0) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2696_269617


namespace NUMINAMATH_CALUDE_line_through_circle_center_l2696_269625

theorem line_through_circle_center (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0 ∧ 
   ∀ cx cy : ℝ, (cx - x)^2 + (cy - y)^2 ≤ (cx + 1)^2 + (cy - 2)^2) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l2696_269625


namespace NUMINAMATH_CALUDE_sin_160_equals_sin_20_l2696_269635

theorem sin_160_equals_sin_20 : Real.sin (160 * π / 180) = Real.sin (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_160_equals_sin_20_l2696_269635


namespace NUMINAMATH_CALUDE_nonagon_diagonal_count_l2696_269689

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonal_count : nonagon_diagonals = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_count_l2696_269689


namespace NUMINAMATH_CALUDE_triangle_inequality_l2696_269646

theorem triangle_inequality (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_area : (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) = 1/4)
  (h_circumradius : (a * b * c) / (4 * (1/4)) = 1) : 
  (1/a + 1/b + 1/c) > (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

#check triangle_inequality

end NUMINAMATH_CALUDE_triangle_inequality_l2696_269646


namespace NUMINAMATH_CALUDE_count_words_to_1000_l2696_269652

def word_count_1_to_99 : Nat := 171

def word_count_100_to_999 : Nat := 486 + 1944

def word_count_1000 : Nat := 37

theorem count_words_to_1000 :
  word_count_1_to_99 + word_count_100_to_999 + word_count_1000 = 2611 :=
by sorry

end NUMINAMATH_CALUDE_count_words_to_1000_l2696_269652


namespace NUMINAMATH_CALUDE_triangle_side_length_l2696_269628

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  a = 1 ∧
  A = π / 6 ∧
  B = π / 3 →
  b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2696_269628


namespace NUMINAMATH_CALUDE_right_triangle_3_4_5_l2696_269615

theorem right_triangle_3_4_5 : ∃ (a b c : ℕ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_3_4_5_l2696_269615


namespace NUMINAMATH_CALUDE_division_remainder_and_primality_l2696_269607

theorem division_remainder_and_primality : 
  let dividend := 5432109
  let divisor := 125
  let remainder := dividend % divisor
  (remainder = 84) ∧ ¬(Nat.Prime remainder) := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_and_primality_l2696_269607


namespace NUMINAMATH_CALUDE_f_six_of_two_l2696_269696

def f (x : ℝ) : ℝ := 3 * x - 1

def f_iter (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => f (f_iter n x)

theorem f_six_of_two : f_iter 6 2 = 1094 := by sorry

end NUMINAMATH_CALUDE_f_six_of_two_l2696_269696


namespace NUMINAMATH_CALUDE_larger_circle_radius_is_32_l2696_269665

/-- Two concentric circles with chord properties -/
structure ConcentricCircles where
  r : ℝ  -- radius of the smaller circle
  AB : ℝ  -- length of AB
  h_ratio : r > 0  -- radius is positive
  h_AB : AB = 16  -- given length of AB

/-- The radius of the larger circle in the concentric circles setup -/
def larger_circle_radius (c : ConcentricCircles) : ℝ := 4 * c.r

theorem larger_circle_radius_is_32 (c : ConcentricCircles) : 
  larger_circle_radius c = 32 := by
  sorry

#check larger_circle_radius_is_32

end NUMINAMATH_CALUDE_larger_circle_radius_is_32_l2696_269665


namespace NUMINAMATH_CALUDE_function_property_l2696_269688

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_property (h1 : ∀ x y : ℝ, f (x + y) = f x + f y) (h2 : f 6 = 3) : f 7 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2696_269688


namespace NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l2696_269629

/-- Proves that for a rectangular hall with width being half the length and area 288 sq. m, 
    the difference between length and width is 12 meters -/
theorem rectangular_hall_dimension_difference 
  (length width : ℝ) 
  (h1 : width = length / 2) 
  (h2 : length * width = 288) : 
  length - width = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l2696_269629


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2696_269680

theorem purely_imaginary_complex_number (a : ℝ) : 
  (a^2 - a - 2 = 0) ∧ (|a - 1| - 1 ≠ 0) → a = -1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2696_269680


namespace NUMINAMATH_CALUDE_younger_person_age_is_29_l2696_269613

/-- The age difference between Brittany and the other person -/
def age_difference : ℕ := 3

/-- The duration of Brittany's vacation -/
def vacation_duration : ℕ := 4

/-- Brittany's age when she returns from vacation -/
def brittany_age_after_vacation : ℕ := 32

/-- The age of the person who is younger than Brittany -/
def younger_person_age : ℕ := brittany_age_after_vacation - vacation_duration - age_difference

theorem younger_person_age_is_29 : younger_person_age = 29 := by
  sorry

end NUMINAMATH_CALUDE_younger_person_age_is_29_l2696_269613


namespace NUMINAMATH_CALUDE_fraction_problem_l2696_269694

theorem fraction_problem (p q : ℚ) : 
  p = 4 → 
  (1 : ℚ)/7 + (2*q - p)/(2*q + p) = 0.5714285714285714 → 
  q = 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2696_269694


namespace NUMINAMATH_CALUDE_complex_repair_cost_is_50_l2696_269687

/-- Represents Jim's bike shop financials for a month -/
structure BikeShop where
  tire_repair_price : ℕ
  tire_repair_cost : ℕ
  tire_repairs_count : ℕ
  complex_repair_price : ℕ
  complex_repairs_count : ℕ
  retail_profit : ℕ
  fixed_expenses : ℕ
  total_profit : ℕ

/-- Calculates the cost of parts for each complex repair -/
def complex_repair_cost (shop : BikeShop) : ℕ :=
  let tire_repair_profit := (shop.tire_repair_price - shop.tire_repair_cost) * shop.tire_repairs_count
  let complex_repairs_revenue := shop.complex_repair_price * shop.complex_repairs_count
  let total_revenue := tire_repair_profit + shop.retail_profit + complex_repairs_revenue
  let profit_before_complex_costs := total_revenue - shop.fixed_expenses
  let complex_repairs_profit := shop.total_profit - (profit_before_complex_costs - complex_repairs_revenue)
  (complex_repairs_revenue - complex_repairs_profit) / shop.complex_repairs_count

theorem complex_repair_cost_is_50 (shop : BikeShop)
  (h1 : shop.tire_repair_price = 20)
  (h2 : shop.tire_repair_cost = 5)
  (h3 : shop.tire_repairs_count = 300)
  (h4 : shop.complex_repair_price = 300)
  (h5 : shop.complex_repairs_count = 2)
  (h6 : shop.retail_profit = 2000)
  (h7 : shop.fixed_expenses = 4000)
  (h8 : shop.total_profit = 3000) :
  complex_repair_cost shop = 50 := by
  sorry

end NUMINAMATH_CALUDE_complex_repair_cost_is_50_l2696_269687


namespace NUMINAMATH_CALUDE_faster_train_length_l2696_269639

/-- Calculates the length of a faster train given the speeds of two trains and the time it takes for the faster train to pass a man in the slower train. -/
theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : faster_speed = 72)
  (h2 : slower_speed = 36)
  (h3 : passing_time = 12)
  (h4 : faster_speed > slower_speed) :
  let relative_speed := faster_speed - slower_speed
  let speed_ms := relative_speed * (5 / 18)
  let train_length := speed_ms * passing_time
  train_length = 120 := by sorry

end NUMINAMATH_CALUDE_faster_train_length_l2696_269639


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_squared_l2696_269659

theorem min_value_of_quadratic_squared (x : ℝ) : 
  ∃ (y : ℝ), (x^2 + 6*x + 2)^2 ≥ 0 ∧ (y^2 + 6*y + 2)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_squared_l2696_269659


namespace NUMINAMATH_CALUDE_pencil_distribution_l2696_269650

/-- Given a total number of pencils and pencils per row, calculate the number of rows -/
def calculate_rows (total_pencils : ℕ) (pencils_per_row : ℕ) : ℕ :=
  total_pencils / pencils_per_row

/-- Theorem: Given 6 pencils distributed equally into rows of 3 pencils each, 
    the number of rows created is 2 -/
theorem pencil_distribution :
  calculate_rows 6 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2696_269650


namespace NUMINAMATH_CALUDE_f_geq_one_range_f_lt_a_plus_two_l2696_269632

/-- The quadratic function f(x) = ax² + (2-a)x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 - a) * x + a

/-- Theorem stating the range of a for which f(x) ≥ 1 holds for all real x -/
theorem f_geq_one_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≥ 2 * Real.sqrt 3 / 3 := by sorry

/-- Helper function to describe the solution set of f(x) < a+2 -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x < 1}
  else if a > 0 then {x | -2/a < x ∧ x < 1}
  else if -2 < a ∧ a < 0 then {x | x < 1 ∨ x > -2/a}
  else if a = -2 then Set.univ
  else {x | x < -2/a ∨ x > 1}

/-- Theorem stating the solution set of f(x) < a+2 -/
theorem f_lt_a_plus_two (a : ℝ) (x : ℝ) :
  f a x < a + 2 ↔ x ∈ solution_set a := by sorry

end NUMINAMATH_CALUDE_f_geq_one_range_f_lt_a_plus_two_l2696_269632


namespace NUMINAMATH_CALUDE_circular_sign_diameter_ratio_l2696_269685

theorem circular_sign_diameter_ratio (d₁ d₂ : ℝ) (h : d₁ > 0 ∧ d₂ > 0) :
  (π * (d₂ / 2)^2) = 49 * (π * (d₁ / 2)^2) → d₂ = 7 * d₁ := by
  sorry

end NUMINAMATH_CALUDE_circular_sign_diameter_ratio_l2696_269685


namespace NUMINAMATH_CALUDE_point_p_final_position_point_q_initial_position_l2696_269697

-- Define the movement of point P
def point_p_movement : ℝ := 2

-- Define the movement of point Q
def point_q_movement : ℝ := 3

-- Theorem for point P's final position
theorem point_p_final_position :
  point_p_movement = 2 → 0 + point_p_movement = 2 :=
by sorry

-- Theorem for point Q's initial position
theorem point_q_initial_position :
  point_q_movement = 3 →
  (0 + point_q_movement = 3 ∨ 0 - point_q_movement = -3) :=
by sorry

end NUMINAMATH_CALUDE_point_p_final_position_point_q_initial_position_l2696_269697


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2696_269600

/-- The probability of drawing a white ball from a bag of red and white balls -/
theorem probability_of_white_ball (total : ℕ) (red : ℕ) (white : ℕ) :
  total = red + white →
  white > 0 →
  total > 0 →
  (white : ℚ) / (total : ℚ) = 4 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l2696_269600


namespace NUMINAMATH_CALUDE_serenity_new_shoes_l2696_269682

theorem serenity_new_shoes (pairs_bought : ℕ) (shoes_per_pair : ℕ) :
  pairs_bought = 3 →
  shoes_per_pair = 2 →
  pairs_bought * shoes_per_pair = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_serenity_new_shoes_l2696_269682


namespace NUMINAMATH_CALUDE_projection_matrix_values_l2696_269602

/-- A 2x2 matrix is a projection matrix if and only if Q^2 = Q -/
def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  Q * Q = Q

/-- The specific matrix we're working with -/
def Q (x y : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![x, 21/49], ![y, 35/49]]

/-- The theorem stating the values of x and y that make Q a projection matrix -/
theorem projection_matrix_values :
  ∃ (x y : ℚ), is_projection_matrix (Q x y) ∧ x = 666/2401 ∧ y = (49 * 2401) / 1891 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l2696_269602


namespace NUMINAMATH_CALUDE_probability_letter_in_mathematics_l2696_269648

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

def unique_letters (s : String) : Finset Char :=
  s.toList.toFinset

theorem probability_letter_in_mathematics :
  (unique_letters mathematics).card / alphabet.card = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_letter_in_mathematics_l2696_269648


namespace NUMINAMATH_CALUDE_eight_integer_pairs_satisfy_equation_l2696_269627

theorem eight_integer_pairs_satisfy_equation :
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ s ↔ x > 0 ∧ y > 0 ∧ Real.sqrt (x * y) - 71 * Real.sqrt x + 30 = 0) ∧
    s.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_integer_pairs_satisfy_equation_l2696_269627


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2696_269664

-- Define the polynomial
def f (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30

-- Define the divisor
def g (x : ℝ) : ℝ := 4 * x - 8

-- Theorem statement
theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, f x = g x * q x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2696_269664


namespace NUMINAMATH_CALUDE_arc_length_240_degrees_l2696_269618

theorem arc_length_240_degrees (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 10 → θ = 240 → l = (θ * π * r) / 180 → l = (40 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_arc_length_240_degrees_l2696_269618


namespace NUMINAMATH_CALUDE_faye_age_l2696_269642

/-- Represents the ages of the people in the problem -/
structure Ages where
  diana : ℕ
  eduardo : ℕ
  chad : ℕ
  faye : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.diana = ages.eduardo - 4 ∧
  ages.eduardo = ages.chad + 5 ∧
  ages.faye = ages.chad + 4 ∧
  ages.diana = 18

/-- The theorem stating that under the given conditions, Faye is 21 years old -/
theorem faye_age (ages : Ages) : problem_conditions ages → ages.faye = 21 := by
  sorry

end NUMINAMATH_CALUDE_faye_age_l2696_269642


namespace NUMINAMATH_CALUDE_population_function_time_to_reach_1_2_million_max_growth_rate_20_years_l2696_269658

-- Define the initial population and growth rate
def initial_population : ℝ := 1000000
def annual_growth_rate : ℝ := 0.012

-- Define the population function
def population (years : ℕ) : ℝ := initial_population * (1 + annual_growth_rate) ^ years

-- Theorem 1: Population function
theorem population_function (years : ℕ) : 
  population years = 100 * (1.012 ^ years) * 10000 := by sorry

-- Theorem 2: Time to reach 1.2 million
theorem time_to_reach_1_2_million : 
  ∃ y : ℕ, y ≥ 16 ∧ y < 17 ∧ population y ≥ 1200000 ∧ population (y-1) < 1200000 := by sorry

-- Theorem 3: Maximum growth rate for 20 years
theorem max_growth_rate_20_years (max_rate : ℝ) : 
  (∀ rate : ℝ, rate ≤ max_rate → initial_population * (1 + rate) ^ 20 ≤ 1200000) ↔ 
  max_rate ≤ 0.009 := by sorry

end NUMINAMATH_CALUDE_population_function_time_to_reach_1_2_million_max_growth_rate_20_years_l2696_269658


namespace NUMINAMATH_CALUDE_ab_value_l2696_269641

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 33) : a * b = 18 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2696_269641


namespace NUMINAMATH_CALUDE_water_fountain_build_time_l2696_269640

/-- Represents the work rate for building water fountains -/
def work_rate (men : ℕ) (length : ℕ) (days : ℕ) : ℚ :=
  length / (men * days)

/-- Theorem stating the relationship between different teams building water fountains -/
theorem water_fountain_build_time 
  (men1 : ℕ) (length1 : ℕ) (days1 : ℕ)
  (men2 : ℕ) (length2 : ℕ) (days2 : ℕ)
  (h1 : men1 = 20) (h2 : length1 = 56) (h3 : days1 = 7)
  (h4 : men2 = 35) (h5 : length2 = 42) (h6 : days2 = 3) :
  work_rate men1 length1 days1 = work_rate men2 length2 days2 :=
by sorry

#check water_fountain_build_time

end NUMINAMATH_CALUDE_water_fountain_build_time_l2696_269640


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2696_269626

theorem arithmetic_equality : 8 / 2 - 5 + 3^2 * 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2696_269626


namespace NUMINAMATH_CALUDE_taller_tree_height_l2696_269633

theorem taller_tree_height (h_taller h_shorter : ℝ) : 
  h_taller - h_shorter = 18 →
  h_shorter / h_taller = 5 / 6 →
  h_taller = 108 := by
sorry

end NUMINAMATH_CALUDE_taller_tree_height_l2696_269633


namespace NUMINAMATH_CALUDE_stratified_sample_sum_eq_six_l2696_269624

/-- Represents the number of varieties in each food category -/
def food_categories : List Nat := [40, 10, 30, 20]

/-- The total number of food varieties -/
def total_varieties : Nat := food_categories.sum

/-- The sample size for food safety inspection -/
def sample_size : Nat := 20

/-- Calculates the number of samples for a given category size -/
def stratified_sample (category_size : Nat) : Nat :=
  (sample_size * category_size) / total_varieties

/-- Theorem: The sum of stratified samples from the second and fourth categories is 6 -/
theorem stratified_sample_sum_eq_six :
  stratified_sample (food_categories[1]) + stratified_sample (food_categories[3]) = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_sum_eq_six_l2696_269624


namespace NUMINAMATH_CALUDE_sunny_gave_away_two_cakes_l2696_269620

/-- The number of cakes Sunny initially baked -/
def initial_cakes : ℕ := 8

/-- The number of candles Sunny puts on each remaining cake -/
def candles_per_cake : ℕ := 6

/-- The total number of candles Sunny uses -/
def total_candles : ℕ := 36

/-- The number of cakes Sunny gave away -/
def cakes_given_away : ℕ := initial_cakes - (total_candles / candles_per_cake)

theorem sunny_gave_away_two_cakes : cakes_given_away = 2 := by
  sorry

end NUMINAMATH_CALUDE_sunny_gave_away_two_cakes_l2696_269620


namespace NUMINAMATH_CALUDE_min_value_of_abs_sum_l2696_269623

theorem min_value_of_abs_sum (x : ℝ) : 
  |x - 4| + |x + 2| + |x - 5| ≥ -1 ∧ ∃ y : ℝ, |y - 4| + |y + 2| + |y - 5| = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_abs_sum_l2696_269623


namespace NUMINAMATH_CALUDE_girls_in_senior_year_l2696_269672

theorem girls_in_senior_year 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (girls_boys_diff : ℕ) 
  (h1 : total_students = 1200)
  (h2 : sample_size = 100)
  (h3 : girls_boys_diff = 20) :
  let boys_in_sample := (sample_size + girls_boys_diff) / 2
  let girls_in_sample := sample_size - boys_in_sample
  let sampling_ratio := sample_size / total_students
  (girls_in_sample * (total_students / sample_size) : ℚ) = 480 := by
sorry

end NUMINAMATH_CALUDE_girls_in_senior_year_l2696_269672


namespace NUMINAMATH_CALUDE_min_PQ_ratio_approaches_infinity_l2696_269677

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define points
variable (X Y M P Q : V)

-- Define conditions
variable (h1 : M = (X + Y) / 2)
variable (h2 : ∃ (t k : ℝ) (d : V), P = Y + t • d ∧ Q = Y - k • d ∧ t > 0 ∧ k > 0)
variable (h3 : ‖X - Q‖ = 2 * ‖M - P‖)
variable (h4 : ‖X - Y‖ / 2 < ‖M - P‖ ∧ ‖M - P‖ < 3 * ‖X - Y‖ / 2)

-- Theorem statement
theorem min_PQ_ratio_approaches_infinity :
  ∀ ε > 0, ∃ δ > 0, ∀ P' Q' : V,
    ‖P' - Q'‖ < ‖P - Q‖ + δ →
    ‖P' - Y‖ / ‖Q' - Y‖ > 1 / ε :=
sorry

end NUMINAMATH_CALUDE_min_PQ_ratio_approaches_infinity_l2696_269677


namespace NUMINAMATH_CALUDE_total_shaded_area_l2696_269606

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a right triangle with two equal legs -/
structure RightTriangle where
  leg : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  r.width * r.height

/-- Calculates the area of a right triangle -/
def rightTriangleArea (t : RightTriangle) : ℝ :=
  0.5 * t.leg * t.leg

/-- Represents the overlap between rectangles -/
def rectangleOverlap : ℝ := 20

/-- Represents the fraction of triangle overlap with rectangles -/
def triangleOverlapFraction : ℝ := 0.5

/-- Theorem stating the total shaded area -/
theorem total_shaded_area (r1 r2 : Rectangle) (t : RightTriangle) :
  let totalArea := rectangleArea r1 + rectangleArea r2 - rectangleOverlap
  let triangleCorrection := triangleOverlapFraction * rightTriangleArea t
  totalArea - triangleCorrection = 70.75 :=
by
  sorry

#check total_shaded_area (Rectangle.mk 4 12) (Rectangle.mk 5 9) (RightTriangle.mk 3)

end NUMINAMATH_CALUDE_total_shaded_area_l2696_269606


namespace NUMINAMATH_CALUDE_solve_equation_l2696_269656

theorem solve_equation (y : ℝ) (h : Real.sqrt (3 / y + 3) = 5 / 3) : y = -27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2696_269656


namespace NUMINAMATH_CALUDE_smallest_valid_number_proof_l2696_269653

/-- Checks if a natural number contains all digits from 0 to 9 --/
def containsAllDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, n / 10^k % 10 = d

/-- The smallest 12-digit number divisible by 36 containing all digits --/
def smallestValidNumber : ℕ := 100023457896

theorem smallest_valid_number_proof :
  (smallestValidNumber ≥ 10^11) ∧ 
  (smallestValidNumber < 10^12) ∧
  (smallestValidNumber % 36 = 0) ∧
  containsAllDigits smallestValidNumber ∧
  ∀ m : ℕ, m ≥ 10^11 ∧ m < 10^12 ∧ m % 36 = 0 ∧ containsAllDigits m → m ≥ smallestValidNumber :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_proof_l2696_269653


namespace NUMINAMATH_CALUDE_parallel_vectors_problem_l2696_269670

/-- Given two vectors a and b in R², where a is parallel to (2a + b), prove that the second component of b is 4 and m = 2. -/
theorem parallel_vectors_problem (a b : ℝ × ℝ) (m : ℝ) :
  a = (1, 2) →
  b = (m, 4) →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • (2 • a + b) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_problem_l2696_269670


namespace NUMINAMATH_CALUDE_probability_theorem_l2696_269614

/-- Represents the enrollment data for language classes --/
structure LanguageEnrollment where
  total : ℕ
  french : ℕ
  spanish : ℕ
  german : ℕ
  french_and_spanish : ℕ
  spanish_and_german : ℕ
  french_and_german : ℕ
  all_three : ℕ

/-- Calculates the probability of selecting two students that cover all three languages --/
def probability_all_languages (e : LanguageEnrollment) : ℚ :=
  1 - (132 : ℚ) / (435 : ℚ)

/-- Theorem stating the probability of selecting two students covering all three languages --/
theorem probability_theorem (e : LanguageEnrollment) 
  (h1 : e.total = 30)
  (h2 : e.french = 20)
  (h3 : e.spanish = 18)
  (h4 : e.german = 10)
  (h5 : e.french_and_spanish = 12)
  (h6 : e.spanish_and_german = 5)
  (h7 : e.french_and_german = 4)
  (h8 : e.all_three = 3) :
  probability_all_languages e = 101 / 145 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2696_269614


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l2696_269686

theorem parallel_vectors_sum (m n : ℝ) : 
  let a : Fin 3 → ℝ := ![(-2 : ℝ), 3, -1]
  let b : Fin 3 → ℝ := ![4, m, n]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) →
  m + n = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l2696_269686


namespace NUMINAMATH_CALUDE_x_value_l2696_269649

theorem x_value (x y : ℝ) (h : x / (x - 2) = (y^2 + 3*y + 1) / (y^2 + 3*y - 1)) : 
  x = 2*y^2 + 6*y + 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2696_269649


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2696_269630

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (-3 + I) / (2 + I) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2696_269630


namespace NUMINAMATH_CALUDE_merck_hourly_rate_l2696_269645

/-- Represents the babysitting data for Layla --/
structure BabysittingData where
  donaldson_hours : ℕ
  merck_hours : ℕ
  hille_hours : ℕ
  total_earnings : ℚ

/-- Calculates the hourly rate for babysitting --/
def hourly_rate (data : BabysittingData) : ℚ :=
  data.total_earnings / (data.donaldson_hours + data.merck_hours + data.hille_hours)

/-- Theorem stating that the hourly rate for the Merck family is $17.0625 --/
theorem merck_hourly_rate (data : BabysittingData) 
  (h1 : data.donaldson_hours = 7)
  (h2 : data.merck_hours = 6)
  (h3 : data.hille_hours = 3)
  (h4 : data.total_earnings = 273) :
  hourly_rate data = 17.0625 := by
  sorry

end NUMINAMATH_CALUDE_merck_hourly_rate_l2696_269645


namespace NUMINAMATH_CALUDE_cross_arrangement_sum_l2696_269667

/-- A type representing digits from 0 to 9 -/
inductive Digit : Type
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Convert a Digit to its natural number value -/
def digitToNat (d : Digit) : Nat :=
  match d with
  | Digit.zero => 0
  | Digit.one => 1
  | Digit.two => 2
  | Digit.three => 3
  | Digit.four => 4
  | Digit.five => 5
  | Digit.six => 6
  | Digit.seven => 7
  | Digit.eight => 8
  | Digit.nine => 9

/-- The cross shape arrangement of digits -/
structure CrossArrangement :=
  (a b c d e f g : Digit)
  (all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
                   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
                   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
                   e ≠ f ∧ e ≠ g ∧
                   f ≠ g)
  (vertical_sum : digitToNat a + digitToNat b + digitToNat c = 25)
  (horizontal_sum : digitToNat d + digitToNat e + digitToNat f + digitToNat g = 17)

theorem cross_arrangement_sum (arr : CrossArrangement) :
  digitToNat arr.a + digitToNat arr.b + digitToNat arr.c +
  digitToNat arr.d + digitToNat arr.e + digitToNat arr.f + digitToNat arr.g = 33 :=
by sorry

end NUMINAMATH_CALUDE_cross_arrangement_sum_l2696_269667


namespace NUMINAMATH_CALUDE_special_gp_ratio_equation_special_gp_ratio_value_l2696_269643

/-- A geometric progression with positive terms where any term is equal to the square of the sum of the next two following terms -/
structure SpecialGeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  special_property : ∀ n : ℕ, a * r^n = (a * r^(n+1) + a * r^(n+2))^2

/-- The common ratio of a special geometric progression satisfies a specific equation -/
theorem special_gp_ratio_equation (gp : SpecialGeometricProgression) :
  gp.r^4 + 2 * gp.r^3 + gp.r^2 - 1 = 0 :=
sorry

/-- The positive solution to the equation r^4 + 2r^3 + r^2 - 1 = 0 is approximately 0.618 -/
theorem special_gp_ratio_value :
  ∃ r : ℝ, r > 0 ∧ r^4 + 2 * r^3 + r^2 - 1 = 0 ∧ abs (r - 0.618) < 0.001 :=
sorry

end NUMINAMATH_CALUDE_special_gp_ratio_equation_special_gp_ratio_value_l2696_269643


namespace NUMINAMATH_CALUDE_total_books_is_91_l2696_269675

/-- Calculates the total number of books sold over three days given the conditions -/
def total_books_sold (tuesday_sales : ℕ) : ℕ :=
  let wednesday_sales := 3 * tuesday_sales
  let thursday_sales := 3 * wednesday_sales
  tuesday_sales + wednesday_sales + thursday_sales

/-- Theorem stating that the total number of books sold over three days is 91 -/
theorem total_books_is_91 : total_books_sold 7 = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_91_l2696_269675


namespace NUMINAMATH_CALUDE_max_gcd_triangular_number_l2696_269651

def triangular_number (n : ℕ+) : ℕ := (n * (n + 1)) / 2

theorem max_gcd_triangular_number :
  ∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (6 * triangular_number n) (n - 2) ≤ 12 ∧
  Nat.gcd (6 * triangular_number k) (k - 2) = 12 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_triangular_number_l2696_269651


namespace NUMINAMATH_CALUDE_complex_division_pure_imaginary_l2696_269674

theorem complex_division_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 3 * Complex.I
  let z₂ : ℂ := 3 - 4 * Complex.I
  (∃ (b : ℝ), z₁ / z₂ = b * Complex.I) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_division_pure_imaginary_l2696_269674


namespace NUMINAMATH_CALUDE_nathan_ate_twenty_gumballs_l2696_269679

/-- The number of gumballs in each package -/
def gumballs_per_package : ℕ := 5

/-- The number of whole boxes Nathan consumed -/
def boxes_consumed : ℕ := 4

/-- The total number of gumballs Nathan ate -/
def gumballs_eaten : ℕ := gumballs_per_package * boxes_consumed

theorem nathan_ate_twenty_gumballs : gumballs_eaten = 20 := by
  sorry

end NUMINAMATH_CALUDE_nathan_ate_twenty_gumballs_l2696_269679


namespace NUMINAMATH_CALUDE_eliminate_denominator_l2696_269608

theorem eliminate_denominator (x : ℝ) : 
  (x + 1) / 3 - 3 = 2 * x + 7 → (x + 1) - 9 = 3 * (2 * x + 7) := by
  sorry

end NUMINAMATH_CALUDE_eliminate_denominator_l2696_269608


namespace NUMINAMATH_CALUDE_sneakers_discount_proof_l2696_269698

/-- Calculates the membership discount percentage given the original price,
    coupon discount, and final price after both discounts are applied. -/
def membership_discount_percentage (original_price coupon_discount final_price : ℚ) : ℚ :=
  let price_after_coupon := original_price - coupon_discount
  let discount_amount := price_after_coupon - final_price
  (discount_amount / price_after_coupon) * 100

/-- Proves that the membership discount percentage is 10% for the given scenario. -/
theorem sneakers_discount_proof :
  membership_discount_percentage 120 10 99 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sneakers_discount_proof_l2696_269698


namespace NUMINAMATH_CALUDE_class_size_l2696_269654

/-- Represents the number of students in a class with English and German courses -/
structure ClassEnrollment where
  total : ℕ
  english : ℕ
  german : ℕ
  both : ℕ
  onlyEnglish : ℕ

/-- Theorem stating the total number of students given the enrollment conditions -/
theorem class_size (c : ClassEnrollment)
  (h1 : c.both = 12)
  (h2 : c.german = 22)
  (h3 : c.onlyEnglish = 23)
  (h4 : c.total = c.english + c.german - c.both)
  (h5 : c.english = c.onlyEnglish + c.both) :
  c.total = 45 := by
  sorry

#check class_size

end NUMINAMATH_CALUDE_class_size_l2696_269654


namespace NUMINAMATH_CALUDE_trip_time_is_ten_weeks_l2696_269690

/-- Calculates the total time spent on a trip visiting three countries -/
def totalTripTime (firstStay : ℕ) (otherStaysMultiplier : ℕ) : ℕ :=
  firstStay + 2 * otherStaysMultiplier * firstStay

/-- Proves that the total trip time is 10 weeks given the specified conditions -/
theorem trip_time_is_ten_weeks :
  totalTripTime 2 2 = 10 := by
  sorry

#eval totalTripTime 2 2

end NUMINAMATH_CALUDE_trip_time_is_ten_weeks_l2696_269690


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l2696_269605

/-- The focus of the parabola x = -8y^2 has coordinates (-1/32, 0) -/
theorem parabola_focus_coordinates :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ x + 8 * y^2
  ∃! p : ℝ × ℝ, p = (-1/32, 0) ∧ 
    (∀ q : ℝ × ℝ, f q = 0 → (q.1 - p.1)^2 + (q.2 - p.2)^2 = (q.2 - 0)^2 + (1/16)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l2696_269605


namespace NUMINAMATH_CALUDE_triangle_distance_sum_l2696_269616

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a point is inside a triangle
def isInside (t : Triangle) (p : ℝ × ℝ) : Prop := sorry

-- Define a function to calculate the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_distance_sum (t : Triangle) (M : ℝ × ℝ) :
  isInside t M →
  distance M t.A + distance M t.B + distance M t.C > perimeter t / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_distance_sum_l2696_269616


namespace NUMINAMATH_CALUDE_fraction_simplification_l2696_269678

theorem fraction_simplification :
  (21 : ℚ) / 25 * 35 / 45 * 75 / 63 = 35 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2696_269678


namespace NUMINAMATH_CALUDE_harolds_books_ratio_l2696_269636

theorem harolds_books_ratio (h m : ℝ) : 
  h > 0 ∧ m > 0 → 
  (1/3 : ℝ) * h + (1/2 : ℝ) * m = (5/6 : ℝ) * m → 
  h / m = 1 := by
sorry

end NUMINAMATH_CALUDE_harolds_books_ratio_l2696_269636


namespace NUMINAMATH_CALUDE_handshakes_for_four_and_n_l2696_269603

/-- Number of handshakes for n people when every two people shake hands once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

theorem handshakes_for_four_and_n :
  (handshakes 4 = 6) ∧
  (∀ n : ℕ, handshakes n = n * (n - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_handshakes_for_four_and_n_l2696_269603


namespace NUMINAMATH_CALUDE_cloth_profit_theorem_l2696_269668

/-- Calculates the profit per meter of cloth (rounded to the nearest rupee) -/
def profit_per_meter (meters : ℕ) (total_selling_price : ℚ) (cost_price_per_meter : ℚ) : ℕ :=
  let total_cost_price := meters * cost_price_per_meter
  let total_profit := total_selling_price - total_cost_price
  let profit_per_meter := total_profit / meters
  (profit_per_meter + 1/2).floor.toNat

/-- The profit per meter of cloth is 29 rupees -/
theorem cloth_profit_theorem :
  profit_per_meter 78 6788 (58.02564102564102) = 29 := by
  sorry

end NUMINAMATH_CALUDE_cloth_profit_theorem_l2696_269668


namespace NUMINAMATH_CALUDE_complex_sum_zero_l2696_269692

theorem complex_sum_zero : 
  let x : ℂ := 2 * Complex.I / (1 - Complex.I)
  let n : ℕ := 2016
  (Finset.sum (Finset.range n) (fun k => Nat.choose n (k + 1) * x ^ (k + 1))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l2696_269692


namespace NUMINAMATH_CALUDE_range_of_f_l2696_269619

def f (x : Int) : Int := x + 1

def domain : Set Int := {-1, 1, 2}

theorem range_of_f :
  {y : Int | ∃ x ∈ domain, f x = y} = {0, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2696_269619


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2696_269673

/-- 
Given an arithmetic sequence where the first term is 3^2 and the third term is 3^4,
prove that the second term is 45.
-/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
  (a 0 = 3^2) → 
  (a 2 = 3^4) → 
  (∀ i j k, i < j → j < k → a j - a i = a k - a j) → 
  (a 1 = 45) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2696_269673


namespace NUMINAMATH_CALUDE_angle_equation_solutions_l2696_269691

theorem angle_equation_solutions (θ : Real) : 
  0 ≤ θ ∧ θ ≤ π ∧ Real.sqrt 2 * (Real.cos (2 * θ)) = Real.cos θ + Real.sin θ → 
  θ = π / 12 ∨ θ = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_equation_solutions_l2696_269691


namespace NUMINAMATH_CALUDE_binary_101101_equals_octal_55_l2696_269693

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- The binary representation of 101101 -/
def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_equals_octal_55 :
  decimal_to_octal (binary_to_decimal binary_101101) = [5, 5] := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_octal_55_l2696_269693


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2696_269634

/-- The intersection of a line and a circle with specific properties implies a unique value for the parameter a. -/
theorem line_circle_intersection (a : ℝ) (h_a_pos : a > 0) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = A.1 + 2*a ∧ B.2 = B.1 + 2*a) ∧ 
    (A.1^2 + A.2^2 - 2*a*A.2 - 2 = 0 ∧ B.1^2 + B.2^2 - 2*a*B.2 - 2 = 0) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 12)) →
  a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2696_269634


namespace NUMINAMATH_CALUDE_mistaken_polynomial_calculation_l2696_269699

/-- Given a polynomial P such that P + (x^2 - 3x + 5) = 5x^2 - 2x + 4,
    prove that P = 4x^2 + x - 1 and P - (x^2 - 3x + 5) = 3x^2 + 4x - 6 -/
theorem mistaken_polynomial_calculation (P : ℝ → ℝ) 
  (h : ∀ x, P x + (x^2 - 3*x + 5) = 5*x^2 - 2*x + 4) : 
  (∀ x, P x = 4*x^2 + x - 1) ∧ 
  (∀ x, P x - (x^2 - 3*x + 5) = 3*x^2 + 4*x - 6) := by
  sorry

end NUMINAMATH_CALUDE_mistaken_polynomial_calculation_l2696_269699


namespace NUMINAMATH_CALUDE_petyas_fruits_l2696_269644

theorem petyas_fruits (total : ℕ) (apples oranges tangerines : ℕ) : 
  total = 20 →
  apples = 6 * tangerines →
  apples > oranges →
  apples + oranges + tangerines = total →
  oranges = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_petyas_fruits_l2696_269644


namespace NUMINAMATH_CALUDE_two_tshirts_per_package_l2696_269638

/-- Given a number of packages and a total number of t-shirts, 
    calculate the number of t-shirts per package -/
def tshirts_per_package (num_packages : ℕ) (total_tshirts : ℕ) : ℕ :=
  total_tshirts / num_packages

/-- Theorem: Given 28 packages and 56 total t-shirts, 
    each package contains 2 t-shirts -/
theorem two_tshirts_per_package :
  tshirts_per_package 28 56 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_tshirts_per_package_l2696_269638


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2696_269663

theorem quadratic_equation_result (a : ℝ) (h : a^2 - 4*a - 12 = 0) : 
  -2*a^2 + 8*a + 40 = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2696_269663


namespace NUMINAMATH_CALUDE_find_divisor_l2696_269622

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 144 →
  quotient = 13 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2696_269622


namespace NUMINAMATH_CALUDE_teacher_pay_per_period_l2696_269660

/-- Calculates the pay per period for a teacher given their work schedule and total earnings --/
theorem teacher_pay_per_period 
  (periods_per_day : ℕ)
  (days_per_month : ℕ)
  (months_worked : ℕ)
  (total_earnings : ℕ)
  (h1 : periods_per_day = 5)
  (h2 : days_per_month = 24)
  (h3 : months_worked = 6)
  (h4 : total_earnings = 3600) :
  total_earnings / (periods_per_day * days_per_month * months_worked) = 5 := by
  sorry

#eval 3600 / (5 * 24 * 6)  -- This should output 5

end NUMINAMATH_CALUDE_teacher_pay_per_period_l2696_269660


namespace NUMINAMATH_CALUDE_cube_root_plus_square_root_l2696_269681

theorem cube_root_plus_square_root : 
  ∃ (x : ℝ), (x = 4 ∨ x = -8) ∧ x = ((-64 : ℝ)^(1/2))^(1/3) + (36 : ℝ)^(1/2) :=
sorry

end NUMINAMATH_CALUDE_cube_root_plus_square_root_l2696_269681


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2696_269610

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2696_269610


namespace NUMINAMATH_CALUDE_sin_cos_45_degrees_l2696_269657

theorem sin_cos_45_degrees : 
  Real.sin (π / 4) = 1 / Real.sqrt 2 ∧ Real.cos (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_45_degrees_l2696_269657


namespace NUMINAMATH_CALUDE_angle_conversion_correct_l2696_269684

/-- The number of clerts in a full circle on Mars -/
def mars_full_circle : ℕ := 400

/-- The number of degrees in a full circle on Earth -/
def earth_full_circle : ℕ := 360

/-- The number of degrees in the angle we're converting -/
def angle_to_convert : ℕ := 45

/-- The number of clerts corresponding to the given angle on Earth -/
def clerts_in_angle : ℕ := 50

theorem angle_conversion_correct : 
  (angle_to_convert : ℚ) / earth_full_circle * mars_full_circle = clerts_in_angle :=
sorry

end NUMINAMATH_CALUDE_angle_conversion_correct_l2696_269684


namespace NUMINAMATH_CALUDE_maximal_ratio_of_primes_l2696_269612

theorem maximal_ratio_of_primes (p q : ℕ) : 
  Prime p → Prime q → p > q → ¬(240 ∣ p^4 - q^4) → 
  (∃ (r : ℚ), r = q / p ∧ r ≤ 2/3 ∧ ∀ (s : ℚ), s = q / p → s ≤ r) :=
sorry

end NUMINAMATH_CALUDE_maximal_ratio_of_primes_l2696_269612


namespace NUMINAMATH_CALUDE_soccer_camp_ratio_l2696_269661

theorem soccer_camp_ratio (total_kids : ℕ) (afternoon_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : afternoon_kids = 750)
  (h3 : ∃ (morning_kids : ℕ), 4 * morning_kids = total_soccer_kids - afternoon_kids) :
  ∃ (total_soccer_kids : ℕ), 
    2 * total_soccer_kids = total_kids ∧ 
    4 * afternoon_kids = 3 * total_soccer_kids := by
sorry


end NUMINAMATH_CALUDE_soccer_camp_ratio_l2696_269661


namespace NUMINAMATH_CALUDE_star_three_five_l2696_269666

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

-- State the theorem
theorem star_three_five : star 3 5 = 64 := by sorry

end NUMINAMATH_CALUDE_star_three_five_l2696_269666


namespace NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l2696_269676

theorem cube_sum_divisible_by_nine (n : ℕ+) :
  ∃ k : ℤ, (n : ℤ)^3 + (n + 1 : ℤ)^3 + (n + 2 : ℤ)^3 = 9 * k :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l2696_269676


namespace NUMINAMATH_CALUDE_f_positive_at_one_l2696_269631

def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

theorem f_positive_at_one (a : ℝ) :
  f a 1 > 0 ↔ 3 - 2 * Real.sqrt 3 < a ∧ a < 3 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_f_positive_at_one_l2696_269631


namespace NUMINAMATH_CALUDE_window_savings_theorem_l2696_269637

/-- Represents the savings when purchasing windows together vs separately --/
def windowSavings (windowPrice : ℕ) (daveWindows : ℕ) (dougWindows : ℕ) : ℕ :=
  let batchSize := 10
  let freeWindows := 2
  let separateCost := 
    (((daveWindows + batchSize - 1) / batchSize * batchSize - freeWindows) * windowPrice)
    + (((dougWindows + batchSize - 1) / batchSize * batchSize - freeWindows) * windowPrice)
  let jointWindows := daveWindows + dougWindows
  let jointCost := ((jointWindows + batchSize - 1) / batchSize * batchSize - freeWindows * (jointWindows / batchSize)) * windowPrice
  separateCost - jointCost

/-- Theorem stating the savings when Dave and Doug purchase windows together --/
theorem window_savings_theorem : 
  windowSavings 120 9 11 = 120 := by
  sorry

end NUMINAMATH_CALUDE_window_savings_theorem_l2696_269637


namespace NUMINAMATH_CALUDE_students_left_after_dropout_l2696_269611

/-- Calculates the number of students left after some drop out -/
def studentsLeft (initialBoys initialGirls boysDropped girlsDropped : ℕ) : ℕ :=
  (initialBoys - boysDropped) + (initialGirls - girlsDropped)

/-- Theorem: Given 14 boys and 10 girls initially, if 4 boys and 3 girls drop out, 17 students are left -/
theorem students_left_after_dropout : studentsLeft 14 10 4 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_students_left_after_dropout_l2696_269611


namespace NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l2696_269695

theorem right_triangle_area_and_perimeter : 
  ∀ (triangle : Set ℝ) (leg1 leg2 hypotenuse : ℝ),
  -- Conditions
  leg1 = 30 →
  leg2 = 45 →
  hypotenuse^2 = leg1^2 + leg2^2 →
  -- Definitions
  let area := (1/2) * leg1 * leg2
  let perimeter := leg1 + leg2 + hypotenuse
  -- Theorem
  area = 675 ∧ perimeter = 129 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l2696_269695
