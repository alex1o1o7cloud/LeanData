import Mathlib

namespace distinct_collections_l480_48071

/-- Represents the count of each letter in MATHEMATICIAN --/
def letterCounts : Finset (Char × ℕ) := 
  {('M', 1), ('A', 3), ('T', 2), ('H', 1), ('E', 1), ('I', 3), ('C', 1), ('N', 1)}

/-- Represents the set of vowels in MATHEMATICIAN --/
def vowels : Finset Char := {'A', 'I', 'E'}

/-- Represents the set of consonants in MATHEMATICIAN --/
def consonants : Finset Char := {'M', 'T', 'H', 'C', 'N'}

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of distinct vowel selections --/
def vowelSelections : ℕ := 
  choose 3 3 + 3 * choose 3 2 + 3 * choose 3 1 + choose 3 0

/-- Calculates the number of distinct consonant selections --/
def consonantSelections : ℕ := 
  choose 4 3 + 2 * choose 4 2 + choose 4 1

/-- The main theorem stating the total number of distinct collections --/
theorem distinct_collections : 
  vowelSelections * consonantSelections = 112 := by sorry

end distinct_collections_l480_48071


namespace partial_fraction_decomposition_product_l480_48026

theorem partial_fraction_decomposition_product : 
  let f (x : ℝ) := (x^2 + 5*x - 14) / (x^3 + x^2 - 11*x - 13)
  let g (x : ℝ) (A B C : ℝ) := A / (x - 1) + B / (x + 1) + C / (x + 13)
  ∀ A B C : ℝ, (∀ x : ℝ, f x = g x A B C) → A * B * C = -360 / 343 := by
  sorry

end partial_fraction_decomposition_product_l480_48026


namespace blue_paint_cans_l480_48011

/-- Given a paint mixture with blue to yellow ratio of 7:3 and 50 total cans, 
    prove that 35 cans contain blue paint. -/
theorem blue_paint_cans (total_cans : ℕ) (blue_ratio yellow_ratio : ℕ) : 
  total_cans = 50 → 
  blue_ratio = 7 → 
  yellow_ratio = 3 → 
  (blue_ratio * total_cans) / (blue_ratio + yellow_ratio) = 35 := by
sorry

end blue_paint_cans_l480_48011


namespace total_pens_bought_l480_48039

theorem total_pens_bought (pen_cost : ℕ) (masha_spent : ℕ) (olya_spent : ℕ) : 
  pen_cost > 10 ∧ 
  masha_spent = 357 ∧ 
  olya_spent = 441 ∧
  masha_spent % pen_cost = 0 ∧ 
  olya_spent % pen_cost = 0 →
  masha_spent / pen_cost + olya_spent / pen_cost = 38 := by
sorry

end total_pens_bought_l480_48039


namespace power_sum_equals_three_l480_48083

theorem power_sum_equals_three (a b c : ℝ) 
  (sum_condition : a + b + c = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 = 3) :
  a^2008 + b^2008 + c^2008 = 3 := by
  sorry

end power_sum_equals_three_l480_48083


namespace smallest_k_and_exponent_l480_48032

theorem smallest_k_and_exponent (k : ℕ) (h : k = 7) :
  64^k > 4^20 ∧ 64^k ≤ 4^21 :=
by sorry

end smallest_k_and_exponent_l480_48032


namespace weight_comparison_l480_48054

def weights : List ℝ := [4, 4, 5, 7, 9, 120]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem weight_comparison (h : weights = [4, 4, 5, 7, 9, 120]) : 
  mean weights - median weights = 19 := by sorry

end weight_comparison_l480_48054


namespace gray_area_between_circles_l480_48031

theorem gray_area_between_circles (r : ℝ) (R : ℝ) : 
  r > 0 → 
  R = 3 * r → 
  2 * r = 4 → 
  R^2 * π - r^2 * π = 32 * π := by
sorry

end gray_area_between_circles_l480_48031


namespace cost_calculation_l480_48091

-- Define the number of caramel apples and ice cream cones
def caramel_apples : ℕ := 3
def ice_cream_cones : ℕ := 4

-- Define the price difference between a caramel apple and an ice cream cone
def price_difference : ℚ := 25 / 100

-- Define the total amount spent
def total_spent : ℚ := 2

-- Define the cost of an ice cream cone
def ice_cream_cost : ℚ := 125 / 700

-- Define the cost of a caramel apple
def caramel_apple_cost : ℚ := ice_cream_cost + price_difference

-- Theorem statement
theorem cost_calculation :
  (caramel_apples : ℚ) * caramel_apple_cost + (ice_cream_cones : ℚ) * ice_cream_cost = total_spent :=
sorry

end cost_calculation_l480_48091


namespace maxwell_walking_speed_l480_48020

-- Define the given constants
def total_distance : ℝ := 36
def brad_speed : ℝ := 4
def maxwell_distance : ℝ := 12

-- Define Maxwell's speed as a variable
def maxwell_speed : ℝ := sorry

-- Theorem statement
theorem maxwell_walking_speed :
  maxwell_speed = 8 :=
by
  -- The proof would go here, but we're using sorry to skip it
  sorry

end maxwell_walking_speed_l480_48020


namespace evaluate_expression_l480_48015

theorem evaluate_expression (a b : ℝ) (ha : a = 3) (hb : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := by
  sorry

end evaluate_expression_l480_48015


namespace sine_intersection_theorem_l480_48025

def M : Set ℝ := { y | ∃ x, y = Real.sin x }
def N : Set ℝ := {0, 1, 2}

theorem sine_intersection_theorem : M ∩ N = {0, 1} := by
  sorry

end sine_intersection_theorem_l480_48025


namespace profit_maximization_l480_48073

/-- Represents the sales volume as a function of price -/
def sales_volume (x : ℝ) : ℝ := -x + 40

/-- Represents the profit as a function of price -/
def profit (x : ℝ) : ℝ := (x - 10) * (sales_volume x)

/-- The optimal price that maximizes profit -/
def optimal_price : ℝ := 25

/-- The maximum profit achieved at the optimal price -/
def max_profit : ℝ := 225

theorem profit_maximization :
  (∀ x : ℝ, profit x ≤ profit optimal_price) ∧
  profit optimal_price = max_profit :=
sorry

end profit_maximization_l480_48073


namespace power_division_equals_729_l480_48081

theorem power_division_equals_729 : 3^12 / 27^2 = 729 :=
by
  -- Define 27 as 3^3
  have h1 : 27 = 3^3 := by sorry
  
  -- Prove that 3^12 / 27^2 = 729
  sorry

end power_division_equals_729_l480_48081


namespace remainder_problem_l480_48046

theorem remainder_problem (x y : ℕ) 
  (h1 : 1059 % x = y)
  (h2 : 1417 % x = y)
  (h3 : 2312 % x = y) :
  x - y = 15 := by sorry

end remainder_problem_l480_48046


namespace worker_overtime_hours_l480_48018

/-- A worker's pay calculation --/
theorem worker_overtime_hours (regular_rate : ℚ) (regular_hours : ℚ) (overtime_rate : ℚ) (total_pay : ℚ) : 
  regular_rate = 3 →
  regular_hours = 40 →
  overtime_rate = 2 * regular_rate →
  total_pay = 180 →
  (total_pay - regular_rate * regular_hours) / overtime_rate = 10 := by
sorry

end worker_overtime_hours_l480_48018


namespace pear_sales_problem_l480_48006

theorem pear_sales_problem (morning_sales afternoon_sales total_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  afternoon_sales = 320 →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 480 :=
by sorry

end pear_sales_problem_l480_48006


namespace inequality_proof_l480_48000

theorem inequality_proof (a b : ℝ) (h : a > b) : 3 - 2*a < 3 - 2*b := by
  sorry

end inequality_proof_l480_48000


namespace complex_equation_circle_l480_48067

/-- The set of complex numbers z satisfying |z|^2 + |z| = 2 forms a circle in the complex plane. -/
theorem complex_equation_circle : 
  {z : ℂ | Complex.abs z ^ 2 + Complex.abs z = 2} = {z : ℂ | Complex.abs z = 1} := by
  sorry

end complex_equation_circle_l480_48067


namespace sequence_a_10_l480_48022

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  ∀ m n : ℕ+, a (m + n) = a m * a n

theorem sequence_a_10 (a : ℕ+ → ℝ) (h1 : sequence_property a) (h2 : a 3 = 8) :
  a 10 = 1024 := by sorry

end sequence_a_10_l480_48022


namespace complex_roots_condition_l480_48028

theorem complex_roots_condition (p : ℝ) :
  (∀ x : ℝ, x^2 + p*x + 1 ≠ 0) →
  p < 2 ∧
  ¬(p < 2 → ∀ x : ℝ, x^2 + p*x + 1 ≠ 0) :=
by sorry

end complex_roots_condition_l480_48028


namespace unique_grouping_l480_48017

def numbers : List ℕ := [12, 30, 42, 44, 57, 91, 95, 143]

def is_valid_grouping (group1 group2 : List ℕ) : Prop :=
  group1.prod = group2.prod ∧
  (group1 ++ group2).toFinset = numbers.toFinset ∧
  group1.toFinset ∩ group2.toFinset = ∅

theorem unique_grouping :
  ∀ (group1 group2 : List ℕ),
    is_valid_grouping group1 group2 →
    ((group1.toFinset = {12, 42, 95, 143} ∧ group2.toFinset = {30, 44, 57, 91}) ∨
     (group2.toFinset = {12, 42, 95, 143} ∧ group1.toFinset = {30, 44, 57, 91})) :=
by sorry

end unique_grouping_l480_48017


namespace function_satisfying_inequality_is_constant_l480_48029

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by sorry

end function_satisfying_inequality_is_constant_l480_48029


namespace cube_edge_15cm_l480_48095

/-- The edge length of a cube that displaces a specific volume of water -/
def cube_edge_length (base_length : ℝ) (base_width : ℝ) (water_rise : ℝ) : ℝ :=
  (base_length * base_width * water_rise) ^ (1/3)

/-- Theorem stating that a cube with the given properties has an edge length of 15 cm -/
theorem cube_edge_15cm :
  cube_edge_length 20 15 11.25 = 15 := by
  sorry

end cube_edge_15cm_l480_48095


namespace right_triangle_area_floor_l480_48053

theorem right_triangle_area_floor (perimeter : ℝ) (inscribed_circle_area : ℝ) : 
  perimeter = 2008 →
  inscribed_circle_area = 100 * Real.pi ^ 3 →
  ⌊(perimeter / 2) * (inscribed_circle_area / Real.pi) ^ (1/2)⌋ = 31541 := by
  sorry

end right_triangle_area_floor_l480_48053


namespace exists_equal_boundary_interior_rectangle_l480_48097

/-- Represents a rectangle in a triangular lattice grid -/
structure TriLatticeRectangle where
  m : Nat  -- horizontal side length in lattice units
  n : Nat  -- vertical side length in lattice units

/-- Calculates the number of lattice points on the boundary of the rectangle -/
def boundaryPoints (rect : TriLatticeRectangle) : Nat :=
  2 * (rect.m + rect.n)

/-- Calculates the number of lattice points inside the rectangle -/
def interiorPoints (rect : TriLatticeRectangle) : Nat :=
  2 * rect.m * rect.n - rect.m - rect.n + 1

/-- Theorem stating the existence of a rectangle with equal boundary and interior points -/
theorem exists_equal_boundary_interior_rectangle :
  ∃ (rect : TriLatticeRectangle), boundaryPoints rect = interiorPoints rect :=
sorry

end exists_equal_boundary_interior_rectangle_l480_48097


namespace total_animals_l480_48098

theorem total_animals (L C P R Q : ℕ) : 
  L = 10 → 
  C = 2 * L + 4 → 
  ∃ G : ℕ, G = 2 * (L + 3) + Q → 
  (L + C + P) + ((L + 3) + R * (L + 3) + G) = 73 + P + R * 13 + Q :=
by sorry

end total_animals_l480_48098


namespace max_value_S_l480_48016

/-- The maximum value of S given the conditions -/
theorem max_value_S (a b : ℝ) (ha : a > 0) (hb : b > 0) (hline : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 
    2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≥ 2 * Real.sqrt (x * y) - 4 * x^2 - y^2) →
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 = (Real.sqrt 2 - 1) / 2 := by
sorry

end max_value_S_l480_48016


namespace circle_passes_through_points_l480_48084

/-- A circle passing through three given points -/
def CircleThroughPoints (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 + p.2^2 - 7*p.1 - 3*p.2 + 2) = 0}

/-- Theorem stating that the circle passes through the given points -/
theorem circle_passes_through_points :
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (1, 4)
  let C : ℝ × ℝ := (4, -2)
  A ∈ CircleThroughPoints A B C ∧
  B ∈ CircleThroughPoints A B C ∧
  C ∈ CircleThroughPoints A B C := by
  sorry

#check circle_passes_through_points

end circle_passes_through_points_l480_48084


namespace saline_solution_concentration_l480_48079

theorem saline_solution_concentration (x : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (a * x / 100 = (a + b) * 20 / 100) ∧ 
    ((a + b) * (1 - 20 / 100) = (a + 2*b) * (1 - 30 / 100))) →
  x = 70 / 3 := by
  sorry

end saline_solution_concentration_l480_48079


namespace centipede_sock_shoe_arrangements_l480_48063

def num_legs : ℕ := 10

def total_items : ℕ := 2 * num_legs

def valid_arrangements : ℕ := Nat.factorial total_items / (2^num_legs)

theorem centipede_sock_shoe_arrangements :
  valid_arrangements = Nat.factorial total_items / (2^num_legs) :=
by sorry

end centipede_sock_shoe_arrangements_l480_48063


namespace eight_couples_handshakes_l480_48045

/-- The number of handshakes in a gathering of married couples --/
def handshakes (n : ℕ) : ℕ :=
  (n * (2 * n - 1)) / 2 - n

/-- Theorem: In a gathering of 8 married couples, the total number of handshakes is 112 --/
theorem eight_couples_handshakes :
  handshakes 8 = 112 := by
  sorry

end eight_couples_handshakes_l480_48045


namespace students_not_in_biology_l480_48077

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) : 
  total_students = 880 →
  biology_percentage = 35 / 100 →
  total_students - (biology_percentage * total_students).floor = 572 := by
  sorry

end students_not_in_biology_l480_48077


namespace union_equals_one_two_three_l480_48036

def M : Set ℤ := {1, 3}
def N (a : ℤ) : Set ℤ := {1 - a, 3}

theorem union_equals_one_two_three (a : ℤ) : 
  M ∪ N a = {1, 2, 3} → a = -1 := by
  sorry

end union_equals_one_two_three_l480_48036


namespace sin_2theta_problem_l480_48050

theorem sin_2theta_problem (θ : Real) (h1 : π/2 < θ ∧ θ < π) (h2 : Real.cos (π/2 - θ) = 3/5) :
  Real.sin (2 * θ) = -24/25 := by
  sorry

end sin_2theta_problem_l480_48050


namespace sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l480_48021

theorem sum_of_solutions_quadratic (a b c d e : ℝ) :
  (∀ x, a * x^2 + b * x + c = d * x + e) →
  (a ≠ 0) →
  (∃ x y, a * x^2 + b * x + c = d * x + e ∧ 
          a * y^2 + b * y + c = d * y + e ∧ 
          x ≠ y) →
  (x + y = -(b - d) / a) :=
by sorry

theorem sum_of_solutions_specific_quadratic :
  let a : ℝ := 2
  let b : ℝ := -8
  let c : ℝ := 5
  let d : ℝ := 4
  let e : ℝ := -20
  (∀ x, a * x^2 + b * x + c = d * x + e) →
  (∃ x y, a * x^2 + b * x + c = d * x + e ∧ 
          a * y^2 + b * y + c = d * y + e ∧ 
          x ≠ y) →
  (x + y = 6) :=
by sorry

end sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l480_48021


namespace max_digits_product_5digit_4digit_l480_48094

theorem max_digits_product_5digit_4digit :
  ∀ (a b : ℕ), 
    10000 ≤ a ∧ a < 100000 →
    1000 ≤ b ∧ b < 10000 →
    a * b < 10000000000 :=
by sorry

end max_digits_product_5digit_4digit_l480_48094


namespace ceiling_floor_difference_l480_48096

theorem ceiling_floor_difference : ⌈(16 : ℝ) / 5 * (-34 : ℝ) / 4⌉ - ⌊(16 : ℝ) / 5 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 := by
  sorry

end ceiling_floor_difference_l480_48096


namespace shaded_area_regular_octagon_l480_48049

/-- The area of the shaded region in a regular octagon --/
theorem shaded_area_regular_octagon (s : ℝ) (h : s = 8) :
  let R := s / (2 * Real.sin (π / 8))
  let shaded_area := (R / 2) ^ 2
  shaded_area = 32 + 16 * Real.sqrt 2 := by
  sorry

end shaded_area_regular_octagon_l480_48049


namespace jim_bakes_two_loaves_l480_48051

/-- The amount of flour Jim can bake into loaves -/
def jim_loaves (cupboard kitchen_counter pantry loaf_requirement : ℕ) : ℕ :=
  (cupboard + kitchen_counter + pantry) / loaf_requirement

/-- Theorem: Jim can bake 2 loaves of bread -/
theorem jim_bakes_two_loaves :
  jim_loaves 200 100 100 200 = 2 := by
  sorry

end jim_bakes_two_loaves_l480_48051


namespace tower_heights_count_l480_48090

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of distinct tower heights achievable -/
def distinctTowerHeights (num_bricks : ℕ) (brick_dims : BrickDimensions) : ℕ :=
  sorry

/-- Theorem stating the number of distinct tower heights for the given problem -/
theorem tower_heights_count :
  let brick_dims : BrickDimensions := ⟨20, 10, 6⟩
  distinctTowerHeights 100 brick_dims = 701 := by
  sorry

end tower_heights_count_l480_48090


namespace sufficient_not_necessary_l480_48086

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, x > 0 → x > -1) ∧ (∃ x, x > -1 ∧ ¬(x > 0)) := by sorry

end sufficient_not_necessary_l480_48086


namespace unique_factor_pair_l480_48070

theorem unique_factor_pair : ∃! (x y : ℕ), 
  x > 0 ∧ y > 0 ∧ y ≥ x ∧ x + y ≤ 20 ∧ 
  (∃ (a b : ℕ), a ≠ x ∧ b ≠ y ∧ a * b = x * y) ∧
  (∀ (a b : ℕ), a > 0 → b > 0 → b ≥ a → a + b ≤ 20 → a * b ≠ x * y ∨ a + b ≠ 13) ∧
  x = 2 ∧ y = 11 := by
sorry

end unique_factor_pair_l480_48070


namespace max_value_theorem_l480_48048

theorem max_value_theorem (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) : 
  (a * b) / (2 * (a + b)) + (a * c) / (2 * (a + c)) + (b * c) / (2 * (b + c)) ≤ 1 / 2 := by
  sorry

end max_value_theorem_l480_48048


namespace integer_solutions_of_inequalities_l480_48074

theorem integer_solutions_of_inequalities (x : ℤ) :
  -1 < x ∧ x ≤ 1 ∧ 4*(2*x-1) ≤ 3*x+1 ∧ 2*x > (x-3)/2 → x = 0 ∨ x = 1 := by
  sorry

end integer_solutions_of_inequalities_l480_48074


namespace max_isosceles_triangles_correct_l480_48010

/-- Represents a set of points on a line and a point not on the line -/
structure PointConfiguration where
  n : ℕ  -- number of points on the line
  h : n = 100

/-- The maximum number of isosceles triangles that can be formed -/
def max_isosceles_triangles (config : PointConfiguration) : ℕ := 150

/-- Theorem stating the maximum number of isosceles triangles -/
theorem max_isosceles_triangles_correct (config : PointConfiguration) :
  max_isosceles_triangles config = 150 := by
  sorry

end max_isosceles_triangles_correct_l480_48010


namespace right_triangle_sum_of_squares_l480_48078

theorem right_triangle_sum_of_squares (A B C : ℝ × ℝ) :
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 1 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - C.1)^2 + (B.2 - C.2)^2 = 2 := by
sorry

end right_triangle_sum_of_squares_l480_48078


namespace rectangle_area_error_percentage_l480_48085

/-- Given a rectangle with actual length L and width W, if the measured length is 1.06L
    and the measured width is 0.95W, then the error percentage in the calculated area is 0.7%. -/
theorem rectangle_area_error_percentage (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let actual_area := L * W
  let calculated_area := measured_length * measured_width
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 0.7 := by
sorry

end rectangle_area_error_percentage_l480_48085


namespace max_value_z_l480_48013

theorem max_value_z (x y : ℝ) (h1 : x - y ≥ 0) (h2 : x + y ≤ 2) (h3 : y ≥ 0) :
  ∃ (z : ℝ), z = 3*x - y ∧ z ≤ 6 ∧ ∃ (x' y' : ℝ), x' - y' ≥ 0 ∧ x' + y' ≤ 2 ∧ y' ≥ 0 ∧ 3*x' - y' = 6 :=
by sorry

end max_value_z_l480_48013


namespace sum_must_be_odd_l480_48005

theorem sum_must_be_odd (x y : ℤ) (h : 7 * x + 5 * y = 11111) : 
  ¬(Even (x + y)) := by
  sorry

end sum_must_be_odd_l480_48005


namespace first_player_win_probability_l480_48034

-- Define the probability of winning in one roll
def prob_win_one_roll : ℚ := 21 / 36

-- Define the probability of not winning in one roll
def prob_not_win_one_roll : ℚ := 1 - prob_win_one_roll

-- Define the game
def dice_game_probability : ℚ :=
  prob_win_one_roll / (1 - prob_not_win_one_roll ^ 2)

-- Theorem statement
theorem first_player_win_probability :
  dice_game_probability = 12 / 17 := by sorry

end first_player_win_probability_l480_48034


namespace triangle_property_l480_48099

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition from the problem -/
def satisfies_condition (t : Triangle) : Prop :=
  1 - 2 * Real.sin t.B * Real.sin t.C = Real.cos (2 * t.B) + Real.cos (2 * t.C) - Real.cos (2 * t.A)

theorem triangle_property (t : Triangle) (h : satisfies_condition t) :
  t.A = Real.pi / 3 ∧ ∃ (x : ℝ), x ≤ Real.pi ∧ ∀ (y : ℝ), Real.sin t.B + Real.sin t.C ≤ Real.sin y := by
  sorry

end triangle_property_l480_48099


namespace rainfall_ratio_l480_48024

/-- Given the total rainfall over two weeks and the rainfall in the second week,
    calculate the ratio of the second week's rainfall to the first week's rainfall. -/
theorem rainfall_ratio (total : ℝ) (second_week : ℝ) :
  total = 30 →
  second_week = 18 →
  (second_week / (total - second_week) = 3 / 2) :=
by
  sorry

end rainfall_ratio_l480_48024


namespace xy_values_l480_48057

theorem xy_values (x y : ℝ) : (x + y + 2) * (x + y - 1) = 0 → x + y = -2 ∨ x + y = 1 := by
  sorry

end xy_values_l480_48057


namespace count_valid_arrangements_l480_48088

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The set of digits used to form the numbers. -/
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The number of digits in each formed number. -/
def number_length : ℕ := 7

/-- The digit that must be at the last position. -/
def last_digit : ℕ := 3

/-- The theorem stating the number of valid arrangements. -/
theorem count_valid_arrangements :
  (factorial (number_length - 1) / 2 : ℕ) = 360 := by sorry

end count_valid_arrangements_l480_48088


namespace workbook_problems_l480_48002

theorem workbook_problems (T : ℕ) : 
  (T : ℚ) / 2 + T / 4 + T / 6 + 20 = T → T = 240 := by
  sorry

end workbook_problems_l480_48002


namespace equation_solution_l480_48041

theorem equation_solution : ∃ x : ℝ, (2 / 7) * (1 / 8) * x = 12 ∧ x = 336 := by
  sorry

end equation_solution_l480_48041


namespace power_sum_six_l480_48092

theorem power_sum_six (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12098 := by
  sorry

end power_sum_six_l480_48092


namespace carnival_tickets_billy_carnival_tickets_l480_48004

/-- Calculate the total number of tickets used at a carnival --/
theorem carnival_tickets (ferris_wheel_rides bumper_car_rides ferris_wheel_cost bumper_car_cost : ℕ) :
  ferris_wheel_rides * ferris_wheel_cost + bumper_car_rides * bumper_car_cost =
  (ferris_wheel_rides * ferris_wheel_cost) + (bumper_car_rides * bumper_car_cost) := by
  sorry

/-- Billy's carnival ticket usage --/
theorem billy_carnival_tickets :
  let ferris_wheel_rides : ℕ := 7
  let bumper_car_rides : ℕ := 3
  let ferris_wheel_cost : ℕ := 6
  let bumper_car_cost : ℕ := 4
  ferris_wheel_rides * ferris_wheel_cost + bumper_car_rides * bumper_car_cost = 54 := by
  sorry

end carnival_tickets_billy_carnival_tickets_l480_48004


namespace b_investment_is_10000_l480_48075

/-- Represents the capital and profit distribution in a business partnership --/
structure BusinessPartnership where
  capitalA : ℝ
  capitalB : ℝ
  capitalC : ℝ
  profitShareB : ℝ
  profitShareDiffAC : ℝ

/-- Theorem stating that under given conditions, B's investment is 10000 --/
theorem b_investment_is_10000 (bp : BusinessPartnership)
  (h1 : bp.capitalA = 8000)
  (h2 : bp.capitalC = 12000)
  (h3 : bp.profitShareB = 1900)
  (h4 : bp.profitShareDiffAC = 760) :
  bp.capitalB = 10000 := by
  sorry

#check b_investment_is_10000

end b_investment_is_10000_l480_48075


namespace range_of_c_l480_48027

-- Define the propositions P and Q
def P (c : ℝ) : Prop := ∀ x : ℝ, Monotone (fun x => (c^2 - 5*c + 7)^x)

def Q (c : ℝ) : Prop := ∀ x : ℝ, |x - 1| + |x - 2*c| > 1

-- Define the theorem
theorem range_of_c :
  (∃! c : ℝ, P c ∨ Q c) →
  {c : ℝ | c ∈ Set.Icc 0 1 ∪ Set.Icc 2 3} = {c : ℝ | P c ∨ Q c} :=
sorry

end range_of_c_l480_48027


namespace intersection_point_l480_48056

/-- A quadratic function of the form y = x^2 + px + q where 3p + q = 2023 -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  h : 3 * p + q = 2023

/-- The point (3, 2032) lies on all quadratic functions satisfying the given condition -/
theorem intersection_point (f : QuadraticFunction) : 
  3^2 + f.p * 3 + f.q = 2032 := by sorry

end intersection_point_l480_48056


namespace triangle_properties_l480_48014

/-- Given a triangle ABC where b = 2√3 and 2a - c = 2b cos C, prove that B = π/3 and the maximum value of 3a + 2c is 4√19 -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  b = 2 * Real.sqrt 3 →
  2 * a - c = 2 * b * Real.cos C →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (B = π / 3 ∧ ∃ (x : ℝ), 3 * a + 2 * c ≤ 4 * Real.sqrt 19 ∧ 
    ∃ (A' B' C' a' b' c' : ℝ), 
      b' = 2 * Real.sqrt 3 ∧
      2 * a' - c' = 2 * b' * Real.cos C' ∧
      0 < A' ∧ A' < π ∧
      0 < B' ∧ B' < π ∧
      0 < C' ∧ C' < π ∧
      A' + B' + C' = π ∧
      3 * a' + 2 * c' = x) :=
by
  sorry


end triangle_properties_l480_48014


namespace cone_radii_sum_l480_48033

/-- Given a circle with radius 5 divided into three sectors with area ratios 1:2:3,
    when these sectors are used as lateral surfaces of three cones,
    the sum of the base radii of these cones equals 5. -/
theorem cone_radii_sum (r₁ r₂ r₃ : ℝ) : r₁ + r₂ + r₃ = 5 :=
  sorry

end cone_radii_sum_l480_48033


namespace min_value_problem_l480_48072

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3/x + 1/y = 1) :
  3*x + 4*y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3/x₀ + 1/y₀ = 1 ∧ 3*x₀ + 4*y₀ = 25 :=
sorry

end min_value_problem_l480_48072


namespace total_days_2010_to_2013_l480_48082

/-- A year is a leap year if it's divisible by 4, except for century years,
    which must be divisible by 400 to be a leap year. -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- The number of days in a given year -/
def daysInYear (year : ℕ) : ℕ :=
  if isLeapYear year then 366 else 365

/-- The range of years we're considering -/
def yearRange : List ℕ := [2010, 2011, 2012, 2013]

theorem total_days_2010_to_2013 :
  (yearRange.map daysInYear).sum = 1461 := by
  sorry

end total_days_2010_to_2013_l480_48082


namespace mans_age_to_sons_age_ratio_l480_48080

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1 -/
theorem mans_age_to_sons_age_ratio :
  ∀ (man_age son_age : ℕ),
  man_age = son_age + 18 →
  son_age = 16 →
  ∃ (k : ℕ), (man_age + 2) = k * (son_age + 2) →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end mans_age_to_sons_age_ratio_l480_48080


namespace currency_notes_problem_l480_48003

theorem currency_notes_problem :
  ∃ (D : ℕ+) (x y : ℕ),
    x + y = 100 ∧
    70 * x + D * y = 5000 :=
by sorry

end currency_notes_problem_l480_48003


namespace tile_coverage_l480_48055

theorem tile_coverage (original_count : ℕ) (original_side : ℝ) (new_side : ℝ) :
  original_count = 96 →
  original_side = 3 →
  new_side = 2 →
  (original_count * original_side * original_side) / (new_side * new_side) = 216 := by
  sorry

end tile_coverage_l480_48055


namespace quadratic_function_proof_l480_48061

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 11

theorem quadratic_function_proof :
  (∀ x : ℝ, f x ≤ 13) ∧  -- maximum value is 13
  f 3 = 5 ∧              -- f(3) = 5
  f (-1) = 5 ∧           -- f(-1) = 5
  (∀ x : ℝ, f x = -2 * x^2 + 4 * x + 11) -- explicit formula
  :=
by sorry

end quadratic_function_proof_l480_48061


namespace root_in_interval_l480_48069

def f (x : ℝ) := x^3 + 2*x - 1

theorem root_in_interval :
  (f 0 < 0) →
  (f 1 > 0) →
  ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f x = 0 :=
by
  sorry

end root_in_interval_l480_48069


namespace ice_cream_cone_ratio_l480_48066

def sugar_cones : ℕ := 45
def waffle_cones : ℕ := 36

theorem ice_cream_cone_ratio : 
  ∃ (a b : ℕ), a = 5 ∧ b = 4 ∧ sugar_cones * b = waffle_cones * a :=
by sorry

end ice_cream_cone_ratio_l480_48066


namespace percentage_problem_l480_48038

theorem percentage_problem (x : ℝ) : (35 / 100) * x = 126 → x = 360 := by
  sorry

end percentage_problem_l480_48038


namespace min_probability_theorem_l480_48065

def closest_integer (m : ℤ) (k : ℤ) : ℤ := 
  sorry

def P (k : ℤ) : ℚ :=
  sorry

theorem min_probability_theorem :
  ∀ k : ℤ, k % 2 = 1 → 1 ≤ k → k ≤ 99 →
    P k ≥ 34/67 ∧ 
    ∃ k₀ : ℤ, k₀ % 2 = 1 ∧ 1 ≤ k₀ ∧ k₀ ≤ 99 ∧ P k₀ = 34/67 :=
  sorry

end min_probability_theorem_l480_48065


namespace complement_of_60_18_l480_48030

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- Calculates the complement of an angle -/
def complement (α : Angle) : Angle :=
  let total_minutes := 90 * 60 - (α.degrees * 60 + α.minutes)
  ⟨total_minutes / 60, total_minutes % 60⟩

theorem complement_of_60_18 :
  let α : Angle := ⟨60, 18⟩
  complement α = ⟨29, 42⟩ := by
  sorry

end complement_of_60_18_l480_48030


namespace arithmetic_geometric_sum_l480_48019

theorem arithmetic_geometric_sum (a₁ d : ℚ) (g₁ r : ℚ) (n : ℕ) 
  (h₁ : a₁ = 15)
  (h₂ : d = 0.2)
  (h₃ : g₁ = 15)
  (h₄ : r = 2)
  (h₅ : n = 101) :
  (n : ℚ) * (a₁ + (a₁ + (n - 1) * d)) / 2 + g₁ * (r^n - 1) / (r - 1) = 15 * (2^101 - 1) + 2525 := by
  sorry

end arithmetic_geometric_sum_l480_48019


namespace factorization_equality_l480_48059

theorem factorization_equality (x : ℝ) : 90 * x^2 + 60 * x + 30 = 30 * (3 * x^2 + 2 * x + 1) := by
  sorry

end factorization_equality_l480_48059


namespace president_and_committee_from_eight_l480_48008

/-- The number of ways to choose a president and a 2-person committee from a group of people. -/
def choose_president_and_committee (n : ℕ) : ℕ :=
  n * (n - 1).choose 2

/-- The theorem stating that choosing a president and a 2-person committee from 8 people results in 168 ways. -/
theorem president_and_committee_from_eight :
  choose_president_and_committee 8 = 168 := by
  sorry

#eval choose_president_and_committee 8

end president_and_committee_from_eight_l480_48008


namespace w_range_l480_48040

-- Define the function w(x)
def w (x : ℝ) : ℝ := x^4 - 6*x^2 + 9

-- Theorem stating the range of w(x)
theorem w_range :
  Set.range w = Set.Ici (0 : ℝ) := by sorry

end w_range_l480_48040


namespace optimal_feed_consumption_l480_48001

/-- Represents the nutritional content and cost of animal feeds -/
structure Feed where
  nutrientA : ℝ
  nutrientB : ℝ
  cost : ℝ

/-- Represents the daily nutritional requirements for an animal -/
structure Requirements where
  minNutrientA : ℝ
  minNutrientB : ℝ

/-- Represents the daily consumption of feeds -/
structure Consumption where
  feedI : ℝ
  feedII : ℝ

/-- Calculates the total cost of a given consumption -/
def totalCost (c : Consumption) : ℝ := c.feedI + c.feedII

/-- Checks if a given consumption meets the nutritional requirements -/
def meetsRequirements (f1 f2 : Feed) (r : Requirements) (c : Consumption) : Prop :=
  c.feedI * f1.nutrientA + c.feedII * f2.nutrientA ≥ r.minNutrientA ∧
  c.feedI * f1.nutrientB + c.feedII * f2.nutrientB ≥ r.minNutrientB

/-- Theorem stating the optimal solution for the animal feed problem -/
theorem optimal_feed_consumption 
  (feedI feedII : Feed)
  (req : Requirements)
  (h1 : feedI.nutrientA = 5 ∧ feedI.nutrientB = 2.5 ∧ feedI.cost = 1)
  (h2 : feedII.nutrientA = 3 ∧ feedII.nutrientB = 3 ∧ feedII.cost = 1)
  (h3 : req.minNutrientA = 30 ∧ req.minNutrientB = 22.5) :
  ∃ (c : Consumption), 
    meetsRequirements feedI feedII req c ∧ 
    totalCost c = 8 ∧
    ∀ (c' : Consumption), meetsRequirements feedI feedII req c' → totalCost c' ≥ totalCost c :=
by sorry

end optimal_feed_consumption_l480_48001


namespace two_n_is_good_pair_exists_good_pair_greater_than_two_l480_48044

/-- A pair (m,n) is good if, when erasing every m-th and then every n-th number, 
    and separately erasing every n-th and then every m-th number, 
    any number k that occurs in both resulting lists appears at the same position in both lists -/
def is_good_pair (m n : ℕ) : Prop :=
  ∀ k : ℕ, 
    let pos1 := (k - k / n) - (k - k / n) / m
    let pos2 := k / m - (k / m) / n
    (pos1 ≠ 0 ∧ pos2 ≠ 0) → pos1 = pos2

/-- For any positive integer n, (2,n) is a good pair -/
theorem two_n_is_good_pair : ∀ n : ℕ, n > 0 → is_good_pair 2 n := by sorry

/-- There exists a pair of positive integers (m,n) such that 2 < m < n and (m,n) is a good pair -/
theorem exists_good_pair_greater_than_two : 
  ∃ m n : ℕ, 2 < m ∧ m < n ∧ is_good_pair m n := by sorry

end two_n_is_good_pair_exists_good_pair_greater_than_two_l480_48044


namespace ln_inequality_and_range_l480_48052

open Real

theorem ln_inequality_and_range (x : ℝ) (hx : x > 0) :
  (∀ x > 0, Real.log x ≤ x - 1) ∧
  (∀ a : ℝ, (∀ x > 0, Real.log x ≤ a * x + (a - 1) / x - 1) ↔ a ≥ 1) :=
by sorry

end ln_inequality_and_range_l480_48052


namespace bakery_sales_theorem_l480_48087

/-- Represents the bakery sales scenario -/
structure BakerySales where
  pumpkin_slices_per_pie : ℕ
  custard_slices_per_pie : ℕ
  pumpkin_price_per_slice : ℕ
  custard_price_per_slice : ℕ
  pumpkin_pies_sold : ℕ
  custard_pies_sold : ℕ

/-- Calculates the total sales from the bakery -/
def total_sales (s : BakerySales) : ℕ :=
  (s.pumpkin_slices_per_pie * s.pumpkin_pies_sold * s.pumpkin_price_per_slice) +
  (s.custard_slices_per_pie * s.custard_pies_sold * s.custard_price_per_slice)

/-- Theorem stating that given the specific conditions, the total sales equal $340 -/
theorem bakery_sales_theorem (s : BakerySales) 
  (h1 : s.pumpkin_slices_per_pie = 8)
  (h2 : s.custard_slices_per_pie = 6)
  (h3 : s.pumpkin_price_per_slice = 5)
  (h4 : s.custard_price_per_slice = 6)
  (h5 : s.pumpkin_pies_sold = 4)
  (h6 : s.custard_pies_sold = 5) :
  total_sales s = 340 := by
  sorry

#eval total_sales {
  pumpkin_slices_per_pie := 8,
  custard_slices_per_pie := 6,
  pumpkin_price_per_slice := 5,
  custard_price_per_slice := 6,
  pumpkin_pies_sold := 4,
  custard_pies_sold := 5
}

end bakery_sales_theorem_l480_48087


namespace discount_percentage_proof_l480_48007

theorem discount_percentage_proof (wholesale_price retail_price : ℝ) 
  (profit_percentage : ℝ) (h1 : wholesale_price = 81) 
  (h2 : retail_price = 108) (h3 : profit_percentage = 0.2) : 
  (retail_price - (wholesale_price + wholesale_price * profit_percentage)) / retail_price = 0.1 := by
  sorry

end discount_percentage_proof_l480_48007


namespace sequence_problem_l480_48042

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define a geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom : is_geometric_sequence b)
  (h_eq : 2 * a 5 - (a 8)^2 + 2 * a 11 = 0)
  (h_b8 : b 8 = a 8) :
  b 7 * b 9 = 4 := by sorry

end sequence_problem_l480_48042


namespace ngon_construction_l480_48023

/-- A line in 2D space -/
structure Line where
  -- Define a line using two points
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

/-- An n-gon in 2D space -/
structure Polygon where
  -- List of vertices
  vertices : List (ℝ × ℝ)

/-- Function to check if a line is a perpendicular bisector of a polygon side -/
def isPerpBisector (l : Line) (p : Polygon) : Prop :=
  sorry

/-- Function to check if a line is an angle bisector of a polygon vertex -/
def isAngleBisector (l : Line) (p : Polygon) : Prop :=
  sorry

/-- Main theorem: Given n lines, there exists an n-gon such that these lines
    are either perpendicular bisectors of its sides or angle bisectors -/
theorem ngon_construction (n : ℕ) (lines : List Line) :
  (lines.length = n) →
  ∃ (p : Polygon),
    (p.vertices.length = n) ∧
    (∀ l ∈ lines, isPerpBisector l p ∨ isAngleBisector l p) :=
by sorry

end ngon_construction_l480_48023


namespace star_operations_l480_48068

-- Define the new operation
def star (x y : ℚ) : ℚ := x * y + |x - y| - 2

-- Theorem statement
theorem star_operations :
  (star 3 (-2) = -3) ∧ (star (star 2 5) (-4) = -31) := by
  sorry

end star_operations_l480_48068


namespace sum_of_reciprocals_l480_48058

theorem sum_of_reciprocals (a b c : ℝ) 
  (sum_condition : a + b + c = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 = 10)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 36) :
  1/a + 1/b + 1/c = 13/18 := by
  sorry

end sum_of_reciprocals_l480_48058


namespace diagonal_angle_in_rectangular_parallelepiped_l480_48037

/-- Given a rectangular parallelepiped with two non-intersecting diagonals of adjacent faces
    inclined at angles α and β to the plane of the base, the angle γ between these diagonals
    is equal to arccos(sin α * sin β). -/
theorem diagonal_angle_in_rectangular_parallelepiped
  (α β : Real)
  (h_α : 0 < α ∧ α < π / 2)
  (h_β : 0 < β ∧ β < π / 2) :
  ∃ γ : Real, γ = Real.arccos (Real.sin α * Real.sin β) ∧
    0 ≤ γ ∧ γ ≤ π := by
  sorry

end diagonal_angle_in_rectangular_parallelepiped_l480_48037


namespace no_real_solutions_iff_k_in_range_l480_48062

theorem no_real_solutions_iff_k_in_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 + Real.sqrt 2 * k * x + 2 ≥ 0) ↔ k ∈ Set.Icc 0 4 :=
by sorry

end no_real_solutions_iff_k_in_range_l480_48062


namespace point_in_region_l480_48076

theorem point_in_region (m : ℝ) :
  (m^2 - 3*m + 2 > 0) ↔ (m < 1 ∨ m > 2) :=
by sorry

end point_in_region_l480_48076


namespace hyperbola_foci_distance_l480_48043

/-- The distance between the foci of a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 is 2√(a^2 + b^2) -/
theorem hyperbola_foci_distance (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let distance := 2 * Real.sqrt (a^2 + b^2)
  distance = 2 * Real.sqrt 34 ↔ a^2 = 25 ∧ b^2 = 9 :=
by sorry

end hyperbola_foci_distance_l480_48043


namespace solve_candy_problem_l480_48093

def candy_problem (total : ℕ) (snickers : ℕ) (mars : ℕ) : Prop :=
  ∃ butterfingers : ℕ, 
    total = snickers + mars + butterfingers ∧ 
    butterfingers = 7

theorem solve_candy_problem : candy_problem 12 3 2 := by
  sorry

end solve_candy_problem_l480_48093


namespace division_remainder_problem_l480_48035

theorem division_remainder_problem (L S R : ℝ) : 
  L - S = 1356 →
  S = 268.2 →
  L = 6 * S + R →
  R = 15 := by
sorry

end division_remainder_problem_l480_48035


namespace gcd_of_37500_and_61250_l480_48012

theorem gcd_of_37500_and_61250 : Nat.gcd 37500 61250 = 1250 := by
  sorry

end gcd_of_37500_and_61250_l480_48012


namespace Q_equals_G_l480_48047

-- Define the sets
def P : Set ℝ := {y | ∃ x, y = x^2 + 1}
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def E : Set ℝ := {x | ∃ y, y = x^2 + 1}
def F : Set (ℝ × ℝ) := {(x, y) | y = x^2 + 1}
def G : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_equals_G : Q = G := by sorry

end Q_equals_G_l480_48047


namespace ice_cream_melt_height_l480_48009

/-- The height of a cylinder with radius 9 inches, having the same volume as a sphere with radius 3 inches, is 4/9 inches. -/
theorem ice_cream_melt_height : 
  let sphere_radius : ℝ := 3
  let cylinder_radius : ℝ := 9
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius ^ 3
  let cylinder_volume (h : ℝ) := Real.pi * cylinder_radius ^ 2 * h
  ∃ h : ℝ, cylinder_volume h = sphere_volume ∧ h = 4 / 9 :=
by sorry

end ice_cream_melt_height_l480_48009


namespace min_abs_z_l480_48089

/-- Given a complex number z satisfying |z - 10| + |z + 3i| = 15, the minimum value of |z| is 2. -/
theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 10) + Complex.abs (z + 3*I) = 15) : 
  ∃ (w : ℂ), Complex.abs (z - 10) + Complex.abs (z + 3*I) = 15 ∧ Complex.abs w = 2 ∧ 
  ∀ (v : ℂ), Complex.abs (v - 10) + Complex.abs (v + 3*I) = 15 → Complex.abs w ≤ Complex.abs v :=
sorry

end min_abs_z_l480_48089


namespace a_fourth_plus_inverse_a_fourth_l480_48060

theorem a_fourth_plus_inverse_a_fourth (a : ℝ) (h : (a + 1/a)^3 = 7) :
  a^4 + 1/a^4 = 1519/81 := by
  sorry

end a_fourth_plus_inverse_a_fourth_l480_48060


namespace cornelia_countries_l480_48064

/-- The number of countries Cornelia visited in Europe -/
def europe_countries : ℕ := 20

/-- The number of countries Cornelia visited in South America -/
def south_america_countries : ℕ := 10

/-- The number of countries Cornelia visited in Asia -/
def asia_countries : ℕ := 6

/-- The total number of countries Cornelia visited -/
def total_countries : ℕ := europe_countries + south_america_countries + 2 * asia_countries

theorem cornelia_countries : total_countries = 42 := by
  sorry

end cornelia_countries_l480_48064
