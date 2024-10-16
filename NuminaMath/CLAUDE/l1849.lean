import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l1849_184968

theorem problem_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^3 / y = 2)
  (h2 : y^3 / z = 4)
  (h3 : z^3 / x = 8) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1849_184968


namespace NUMINAMATH_CALUDE_impossible_tiling_l1849_184982

/-- Represents a T-tetromino placement on a checkerboard -/
structure TTetromino where
  blackMajor : ℕ  -- number of T-tetrominoes with 3 black squares
  whiteMajor : ℕ  -- number of T-tetrominoes with 3 white squares

/-- The size of the grid -/
def gridSize : ℕ := 10

/-- The total number of squares in the grid -/
def totalSquares : ℕ := gridSize * gridSize

/-- Represents the coloring constraint of the checkerboard pattern -/
def colorConstraint (t : TTetromino) : Prop :=
  3 * t.blackMajor + t.whiteMajor = totalSquares / 2 ∧
  t.blackMajor + 3 * t.whiteMajor = totalSquares / 2

/-- Theorem stating the impossibility of tiling the grid -/
theorem impossible_tiling : ¬ ∃ t : TTetromino, colorConstraint t := by
  sorry

end NUMINAMATH_CALUDE_impossible_tiling_l1849_184982


namespace NUMINAMATH_CALUDE_tree_height_reaches_29_feet_in_15_years_l1849_184947

/-- Calculates the height of the tree after a given number of years -/
def tree_height (years : ℕ) : ℕ :=
  let initial_height := 4
  let first_year_growth := 5
  let second_year_growth := 4
  let min_growth := 1
  let rec height_after (n : ℕ) (current_height : ℕ) (current_growth : ℕ) : ℕ :=
    if n = 0 then
      current_height
    else if n = 1 then
      height_after (n - 1) (current_height + first_year_growth) second_year_growth
    else if current_growth > min_growth then
      height_after (n - 1) (current_height + current_growth) (current_growth - 1)
    else
      height_after (n - 1) (current_height + min_growth) min_growth
  height_after years initial_height first_year_growth

/-- Theorem stating that it takes 15 years for the tree to reach or exceed 29 feet -/
theorem tree_height_reaches_29_feet_in_15_years :
  tree_height 15 ≥ 29 ∧ ∀ y : ℕ, y < 15 → tree_height y < 29 :=
by sorry

end NUMINAMATH_CALUDE_tree_height_reaches_29_feet_in_15_years_l1849_184947


namespace NUMINAMATH_CALUDE_cookie_count_l1849_184932

/-- Given a set of bags filled with cookies, where each bag contains 3 cookies
    and there are 25 bags in total, the total number of cookies is 75. -/
theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) : 
  bags = 25 → cookies_per_bag = 3 → bags * cookies_per_bag = 75 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l1849_184932


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_of_coefficients_l1849_184912

theorem cubic_polynomial_sum_of_coefficients 
  (A B C : ℝ) (v : ℂ) :
  let Q : ℂ → ℂ := λ z ↦ z^3 + A*z^2 + B*z + C
  (∀ z : ℂ, Q z = 0 ↔ z = v - 2*I ∨ z = v + 7*I ∨ z = 3*v + 5) →
  A + B + C = Q 1 - 1 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_of_coefficients_l1849_184912


namespace NUMINAMATH_CALUDE_exists_decreasing_function_always_ge_one_l1849_184967

theorem exists_decreasing_function_always_ge_one :
  ∃ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f x > f y) ∧ (∀ x : ℝ, f x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_decreasing_function_always_ge_one_l1849_184967


namespace NUMINAMATH_CALUDE_purchase_combinations_eq_545_l1849_184913

/-- Represents the number of oreo flavors available -/
def oreo_flavors : ℕ := 6

/-- Represents the number of milk flavors available -/
def milk_flavors : ℕ := 4

/-- Represents the total number of products purchased -/
def total_products : ℕ := 3

/-- Represents the number of flavors Alpha can choose from (excluding chocolate) -/
def alpha_flavors : ℕ := oreo_flavors - 1 + milk_flavors

/-- Function to calculate the number of ways Alpha and Beta can purchase products -/
def purchase_combinations : ℕ := sorry

/-- Theorem stating the correct number of purchase combinations -/
theorem purchase_combinations_eq_545 : purchase_combinations = 545 := by sorry

end NUMINAMATH_CALUDE_purchase_combinations_eq_545_l1849_184913


namespace NUMINAMATH_CALUDE_dot_product_om_on_l1849_184953

/-- Given two points M and N on the line x + y - 2 = 0, where M(1,1) and |MN| = √2,
    prove that the dot product of OM and ON equals 2 -/
theorem dot_product_om_on (N : ℝ × ℝ) : 
  N.1 + N.2 = 2 →  -- N is on the line x + y - 2 = 0
  (N.1 - 1)^2 + (N.2 - 1)^2 = 2 →  -- |MN| = √2
  (1 * N.1 + 1 * N.2 : ℝ) = 2 := by  -- OM · ON = 2
sorry

end NUMINAMATH_CALUDE_dot_product_om_on_l1849_184953


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1849_184984

theorem simplify_and_evaluate (m n : ℤ) (h1 : m = 2) (h2 : n = -1^2023) :
  (2*m + n) * (2*m - n) - (2*m - n)^2 + 2*n*(m + n) = -12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1849_184984


namespace NUMINAMATH_CALUDE_train_b_start_time_l1849_184992

/-- The time when trains meet, in hours after midnight -/
def meeting_time : ℝ := 12

/-- The time when train A starts, in hours after midnight -/
def train_a_start : ℝ := 8

/-- The distance between city A and city B in kilometers -/
def total_distance : ℝ := 465

/-- The speed of train A in km/hr -/
def train_a_speed : ℝ := 60

/-- The speed of train B in km/hr -/
def train_b_speed : ℝ := 75

/-- The theorem stating that the train from city B starts at 9 a.m. -/
theorem train_b_start_time :
  ∃ (t : ℝ),
    t = 9 ∧
    (meeting_time - train_a_start) * train_a_speed +
      (meeting_time - t) * train_b_speed = total_distance :=
by sorry

end NUMINAMATH_CALUDE_train_b_start_time_l1849_184992


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l1849_184996

/-- Represents a cube with square holes cut through each face. -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the total surface area of a cube with square holes, including inside surfaces. -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length ^ 2
  let hole_area := 6 * cube.hole_side_length ^ 2
  let new_exposed_area := 6 * 4 * cube.hole_side_length ^ 2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem stating that a cube with edge length 4 and hole side length 2 has a total surface area of 168. -/
theorem cube_with_holes_surface_area :
  let cube := CubeWithHoles.mk 4 2
  total_surface_area cube = 168 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l1849_184996


namespace NUMINAMATH_CALUDE_salary_calculation_l1849_184914

theorem salary_calculation (salary : ℝ) : 
  salary > 0 →
  salary / 5 + salary / 10 + 3 * salary / 5 + 15000 = salary →
  salary = 150000 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l1849_184914


namespace NUMINAMATH_CALUDE_find_taco_order_l1849_184910

/-- Represents the number of tacos and enchiladas in an order -/
structure Order where
  tacos : ℕ
  enchiladas : ℕ

/-- Represents the cost of an order in dollars -/
def cost (order : Order) (taco_price enchilada_price : ℝ) : ℝ :=
  taco_price * order.tacos + enchilada_price * order.enchiladas

theorem find_taco_order : ∃ (my_order : Order) (enchilada_price : ℝ),
  my_order.enchiladas = 3 ∧
  cost my_order 0.9 enchilada_price = 7.8 ∧
  cost (Order.mk 3 5) 0.9 enchilada_price = 12.7 ∧
  my_order.tacos = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_taco_order_l1849_184910


namespace NUMINAMATH_CALUDE_divisible_by_five_l1849_184907

theorem divisible_by_five (B : Nat) : 
  B < 10 → (476 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l1849_184907


namespace NUMINAMATH_CALUDE_mass_CaSO4_formed_l1849_184927

-- Define the molar masses
def molar_mass_Ca : ℝ := 40.08
def molar_mass_S : ℝ := 32.06
def molar_mass_O : ℝ := 16.00

-- Define the molar mass of CaSO₄
def molar_mass_CaSO4 : ℝ := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

-- Define the number of moles of Ca(OH)₂
def moles_CaOH2 : ℝ := 12

-- Theorem statement
theorem mass_CaSO4_formed (excess_H2SO4 : Prop) (neutralization_reaction : Prop) :
  moles_CaOH2 * molar_mass_CaSO4 = 1633.68 := by
  sorry


end NUMINAMATH_CALUDE_mass_CaSO4_formed_l1849_184927


namespace NUMINAMATH_CALUDE_vehicle_value_fraction_l1849_184944

def vehicle_value_this_year : ℚ := 16000
def vehicle_value_last_year : ℚ := 20000

theorem vehicle_value_fraction :
  vehicle_value_this_year / vehicle_value_last_year = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_fraction_l1849_184944


namespace NUMINAMATH_CALUDE_factorization_equality_l1849_184954

theorem factorization_equality (a x y : ℝ) : 
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1849_184954


namespace NUMINAMATH_CALUDE_largest_number_l1849_184933

theorem largest_number (S : Set ℝ := {-5, 0, 3, 1/3}) : 
  ∃ m ∈ S, ∀ x ∈ S, x ≤ m ∧ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1849_184933


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_quadratic_equation_rational_roots_l1849_184977

/-- The discriminant of the quadratic equation 3x^2(a+b+c) + 4x(ab+bc+ca) + 4abc = 0 -/
def discriminant (a b c : ℝ) : ℝ :=
  16 * (a * b + b * c + c * a)^2 - 48 * (a + b + c) * a * b * c

theorem quadratic_equation_real_roots (a b c : ℝ) :
  discriminant a b c ≥ 0 := by sorry

theorem quadratic_equation_rational_roots (a b c : ℝ) :
  (b = c ∨ b * c = a * (b + c - a)) →
  ∃ (k : ℝ), discriminant a b c = k^2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_quadratic_equation_rational_roots_l1849_184977


namespace NUMINAMATH_CALUDE_complex_power_210_deg_60_l1849_184971

theorem complex_power_210_deg_60 :
  (Complex.exp (210 * π / 180 * I)) ^ 60 = -1/2 + Complex.I * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_210_deg_60_l1849_184971


namespace NUMINAMATH_CALUDE_circle_area_in_square_l1849_184943

theorem circle_area_in_square (square_area : Real) (circle_area : Real) : 
  square_area = 400 →
  circle_area = Real.pi * (Real.sqrt square_area / 2)^2 →
  circle_area = 100 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_area_in_square_l1849_184943


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_quadratic_roots_l1849_184903

theorem sum_reciprocals_of_quadratic_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b)
  (eq_a : a^2 + a - 2007 = 0) (eq_b : b^2 + b - 2007 = 0) :
  1/a + 1/b = 1/2007 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_quadratic_roots_l1849_184903


namespace NUMINAMATH_CALUDE_brothers_ticket_cost_l1849_184963

/-- Proves that each brother's ticket costs $10 given the problem conditions -/
theorem brothers_ticket_cost (isabelle_ticket_cost : ℕ) 
  (total_savings : ℕ) (weeks_worked : ℕ) (wage_per_week : ℕ) :
  isabelle_ticket_cost = 20 →
  total_savings = 10 →
  weeks_worked = 10 →
  wage_per_week = 3 →
  let total_money := total_savings + weeks_worked * wage_per_week
  let remaining_money := total_money - isabelle_ticket_cost
  remaining_money / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ticket_cost_l1849_184963


namespace NUMINAMATH_CALUDE_least_distance_is_one_thirtyfifth_l1849_184983

-- Define the unit segment [0, 1]
def unit_segment : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Define the division points for fifths
def fifth_points : Set ℝ := {x : ℝ | ∃ n : ℕ, 0 ≤ n ∧ n ≤ 5 ∧ x = n / 5}

-- Define the division points for sevenths
def seventh_points : Set ℝ := {x : ℝ | ∃ n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ x = n / 7}

-- Define all division points
def all_points : Set ℝ := fifth_points ∪ seventh_points

-- Define the distance between two points
def distance (x y : ℝ) : ℝ := |x - y|

-- Theorem statement
theorem least_distance_is_one_thirtyfifth :
  ∃ x y : ℝ, x ∈ all_points ∧ y ∈ all_points ∧ x ≠ y ∧
  distance x y = 1 / 35 ∧
  ∀ a b : ℝ, a ∈ all_points → b ∈ all_points → a ≠ b →
  distance a b ≥ 1 / 35 :=
sorry

end NUMINAMATH_CALUDE_least_distance_is_one_thirtyfifth_l1849_184983


namespace NUMINAMATH_CALUDE_constant_function_l1849_184923

theorem constant_function (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2011 * x) = 2011) :
  ∀ x : ℝ, f (3 * x) = 2011 := by
sorry

end NUMINAMATH_CALUDE_constant_function_l1849_184923


namespace NUMINAMATH_CALUDE_triangle_max_area_l1849_184970

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a * Real.cos C - c / 2 = b →
  a = 2 * Real.sqrt 3 →
  (∃ (S : ℝ), S = (1 / 2) * b * c * Real.sin A ∧
    ∀ (S' : ℝ), S' = (1 / 2) * b * c * Real.sin A → S' ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1849_184970


namespace NUMINAMATH_CALUDE_biker_journey_l1849_184918

/-- Proves that given a biker's journey conditions, the distance between towns is 140 km and initial speed is 28 km/hr -/
theorem biker_journey (total_distance : ℝ) (initial_speed : ℝ) : 
  (total_distance / 2 = initial_speed * 2.5) →
  (total_distance / 2 = (initial_speed + 2) * 2.333) →
  (total_distance = 140 ∧ initial_speed = 28) := by
  sorry

end NUMINAMATH_CALUDE_biker_journey_l1849_184918


namespace NUMINAMATH_CALUDE_sin_cos_shift_l1849_184902

theorem sin_cos_shift (x : ℝ) : Real.sin (x/2) = Real.cos ((x-π)/2 - π/4) := by sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l1849_184902


namespace NUMINAMATH_CALUDE_tan_half_product_l1849_184924

theorem tan_half_product (a b : Real) :
  7 * (Real.cos a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 5 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -5) := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_l1849_184924


namespace NUMINAMATH_CALUDE_orange_bin_problem_l1849_184978

theorem orange_bin_problem (initial_oranges : ℕ) (thrown_away : ℕ) (final_oranges : ℕ)
  (h1 : initial_oranges = 34)
  (h2 : thrown_away = 20)
  (h3 : final_oranges = 27) :
  final_oranges - (initial_oranges - thrown_away) = 13 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_problem_l1849_184978


namespace NUMINAMATH_CALUDE_marble_remainder_l1849_184928

theorem marble_remainder (r p : ℤ) 
  (hr : r % 8 = 5) 
  (hp : p % 8 = 6) : 
  (r + p) % 8 = 3 := by sorry

end NUMINAMATH_CALUDE_marble_remainder_l1849_184928


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1849_184921

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- Statement to prove
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1849_184921


namespace NUMINAMATH_CALUDE_minimum_sales_increase_l1849_184945

theorem minimum_sales_increase (x : ℝ) : 
  let jan_to_may : ℝ := 38.6
  let june : ℝ := 5
  let july : ℝ := june * (1 + x / 100)
  let august : ℝ := july * (1 + x / 100)
  let sep_oct : ℝ := july + august
  let total : ℝ := jan_to_may + june + july + august + sep_oct
  (total ≥ 70 ∧ ∀ y, y < x → (
    let july_y : ℝ := june * (1 + y / 100)
    let august_y : ℝ := july_y * (1 + y / 100)
    let sep_oct_y : ℝ := july_y + august_y
    let total_y : ℝ := jan_to_may + june + july_y + august_y + sep_oct_y
    total_y < 70
  )) → x = 20 := by
sorry

end NUMINAMATH_CALUDE_minimum_sales_increase_l1849_184945


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l1849_184995

/-- Given a geometric sequence with positive terms and common ratio not equal to 1,
    prove that the sum of the first and fourth terms is greater than
    the sum of the second and third terms. -/
theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ)  -- The geometric sequence
  (q : ℝ)      -- The common ratio
  (h1 : ∀ n, a n > 0)  -- All terms are positive
  (h2 : q ≠ 1)  -- Common ratio is not 1
  (h3 : ∀ n, a (n + 1) = a n * q)  -- Definition of geometric sequence
  : a 1 + a 4 > a 2 + a 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l1849_184995


namespace NUMINAMATH_CALUDE_function_value_implies_b_equals_negative_one_l1849_184988

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x - b else 2^x

theorem function_value_implies_b_equals_negative_one (b : ℝ) :
  (f b (f b (1/2)) = 4) → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_implies_b_equals_negative_one_l1849_184988


namespace NUMINAMATH_CALUDE_proposition_and_converse_l1849_184976

theorem proposition_and_converse : 
  (∀ a b : ℝ, a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ 
  ¬(∀ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_converse_l1849_184976


namespace NUMINAMATH_CALUDE_boy_age_multiple_l1849_184946

theorem boy_age_multiple : 
  let present_age : ℕ := 16
  let age_six_years_ago : ℕ := present_age - 6
  let age_four_years_hence : ℕ := present_age + 4
  (age_four_years_hence : ℚ) / (age_six_years_ago : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_boy_age_multiple_l1849_184946


namespace NUMINAMATH_CALUDE_society_member_property_l1849_184937

theorem society_member_property (n : ℕ) (h : n = 1978) :
  ∀ (f : Fin n → Fin 6),
  ∃ (i j k : Fin n),
    f i = f j ∧ f i = f k ∧
    (i.val = j.val + k.val ∨ i.val = 2 * j.val) :=
by sorry

end NUMINAMATH_CALUDE_society_member_property_l1849_184937


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l1849_184940

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (angle_cos : ℝ) 
  (h1 : a = 4) 
  (h2 : b = 5) 
  (h3 : 2 * angle_cos^2 + 3 * angle_cos - 2 = 0) 
  (h4 : c^2 = a^2 + b^2 - 2*a*b*angle_cos) : 
  c = Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l1849_184940


namespace NUMINAMATH_CALUDE_thirty_switch_network_connections_l1849_184975

/-- A network of switches with direct connections between them. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- The total number of connections in a switch network. -/
def total_connections (network : SwitchNetwork) : ℕ :=
  network.num_switches * network.connections_per_switch / 2

/-- Theorem: In a network of 30 switches, where each switch is directly
    connected to exactly 4 other switches, the total number of connections is 60. -/
theorem thirty_switch_network_connections :
  let network : SwitchNetwork := { num_switches := 30, connections_per_switch := 4 }
  total_connections network = 60 := by
  sorry


end NUMINAMATH_CALUDE_thirty_switch_network_connections_l1849_184975


namespace NUMINAMATH_CALUDE_set_operations_l1849_184908

def U : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def A : Finset Nat := {1, 3, 5, 8, 9}
def B : Finset Nat := {2, 5, 6, 8, 10}

theorem set_operations :
  (A ∪ B = {1, 2, 3, 5, 6, 8, 9, 10}) ∧
  ((U \ A) ∩ (U \ B) = {4, 7}) ∧
  (A \ B = {1, 3, 9}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l1849_184908


namespace NUMINAMATH_CALUDE_jennifer_sweets_distribution_l1849_184969

theorem jennifer_sweets_distribution (green blue yellow : ℕ) 
  (h1 : green = 212)
  (h2 : blue = 310)
  (h3 : yellow = 502)
  (friends : ℕ)
  (h4 : friends = 3) :
  (green + blue + yellow) / (friends + 1) = 256 := by
sorry

end NUMINAMATH_CALUDE_jennifer_sweets_distribution_l1849_184969


namespace NUMINAMATH_CALUDE_frank_has_three_cookies_l1849_184930

/-- Given the number of cookies Millie has -/
def millies_cookies : ℕ := 4

/-- Mike's cookies in terms of Millie's -/
def mikes_cookies (m : ℕ) : ℕ := 3 * m

/-- Frank's cookies in terms of Mike's -/
def franks_cookies (m : ℕ) : ℕ := m / 2 - 3

/-- Theorem stating that Frank has 3 cookies given the conditions -/
theorem frank_has_three_cookies :
  franks_cookies (mikes_cookies millies_cookies) = 3 := by
  sorry

end NUMINAMATH_CALUDE_frank_has_three_cookies_l1849_184930


namespace NUMINAMATH_CALUDE_binomial_coefficient_7_choose_5_l1849_184929

theorem binomial_coefficient_7_choose_5 : Nat.choose 7 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_7_choose_5_l1849_184929


namespace NUMINAMATH_CALUDE_swim_team_capacity_l1849_184922

/-- Represents the number of additional people that could have ridden with the swim team --/
def additional_capacity (num_cars num_vans : ℕ) 
                        (people_per_car people_per_van : ℕ)
                        (max_car_capacity max_van_capacity : ℕ) : ℕ :=
  let total_capacity := num_cars * max_car_capacity + num_vans * max_van_capacity
  let actual_passengers := num_cars * people_per_car + num_vans * people_per_van
  total_capacity - actual_passengers

/-- Theorem stating that given the problem conditions, 
    the additional capacity is 17 people --/
theorem swim_team_capacity : 
  additional_capacity 2 3 5 3 6 8 = 17 := by
  sorry

end NUMINAMATH_CALUDE_swim_team_capacity_l1849_184922


namespace NUMINAMATH_CALUDE_cube_root_of_three_twos_to_seven_l1849_184901

theorem cube_root_of_three_twos_to_seven (x : ℝ) :
  x = (2^7 + 2^7 + 2^7)^(1/3) → x = 4 * (2^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_three_twos_to_seven_l1849_184901


namespace NUMINAMATH_CALUDE_floor_neg_seven_fourths_l1849_184973

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_seven_fourths_l1849_184973


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l1849_184906

/-- Tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of tetrahedron PQRS with given edge lengths is 99/2 -/
theorem tetrahedron_volume :
  ∀ t : Tetrahedron,
  t.PQ = 6 ∧
  t.PR = 4 ∧
  t.PS = 5 ∧
  t.QR = 5 ∧
  t.QS = 4 ∧
  t.RS = 15 / 5 * Real.sqrt 2 →
  volume t = 99 / 2 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l1849_184906


namespace NUMINAMATH_CALUDE_tv_selection_problem_l1849_184917

theorem tv_selection_problem (type_a : ℕ) (type_b : ℕ) (total_select : ℕ) : 
  type_a = 4 → type_b = 5 → total_select = 3 →
  (Nat.choose type_a 1 * Nat.choose type_b 2 + Nat.choose type_a 2 * Nat.choose type_b 1) = 70 :=
by sorry

end NUMINAMATH_CALUDE_tv_selection_problem_l1849_184917


namespace NUMINAMATH_CALUDE_certain_number_problem_l1849_184957

theorem certain_number_problem (x : ℝ) (n : ℝ) : 
  (9 - 4 / x = 7 + n / x) → (x = 6) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1849_184957


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1849_184972

/-- Quadratic function y = x^2 - 2tx + 3 -/
def f (t x : ℝ) : ℝ := x^2 - 2*t*x + 3

theorem quadratic_function_properties (t : ℝ) (h_t : t > 0) :
  (f t 2 = 1 → t = 3/2) ∧
  (∃ (x_min : ℝ), x_min ∈ Set.Icc 0 3 ∧ 
    (∀ x, x ∈ Set.Icc 0 3 → f t x ≥ f t x_min) ∧ 
    f t x_min = -2 → t = Real.sqrt 5) ∧
  (∀ (m a b : ℝ), 
    f t (m-2) = a ∧ f t 4 = b ∧ f t m = a ∧ a < b ∧ b < 3 → 
    (3 < m ∧ m < 4) ∨ m > 6) := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1849_184972


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_two_l1849_184905

/-- A function satisfying the given inequality for all real x and y is constant and equal to 2. -/
theorem function_satisfying_inequality_is_constant_two 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, 2 * f x + 2 * f y - f x * f y ≥ 4) : 
  ∀ x : ℝ, f x = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_two_l1849_184905


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1849_184991

/-- A geometric sequence with common ratio q > 1 and positive first term -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ a 1 > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) 
    (h : GeometricSequence a q) 
    (eq : a 2 * a 4 + a 4 * a 10 - a 4 * a 6 - a 5 * a 5 = 9) :
  a 3 - a 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1849_184991


namespace NUMINAMATH_CALUDE_vector_magnitude_condition_l1849_184990

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_condition (a b : V) :
  ¬(∀ a b : V, ‖a‖ = ‖b‖ → ‖a + b‖ = ‖a - b‖) ∧
  ¬(∀ a b : V, ‖a + b‖ = ‖a - b‖ → ‖a‖ = ‖b‖) :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_condition_l1849_184990


namespace NUMINAMATH_CALUDE_expression_evaluation_l1849_184948

theorem expression_evaluation (d : ℕ) (h : d = 2) : 
  (d^d + d*(d+1)^d)^d = 484 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1849_184948


namespace NUMINAMATH_CALUDE_annika_return_time_l1849_184994

/-- Represents Annika's hiking scenario -/
structure HikingScenario where
  rate : ℝ  -- Hiking rate in minutes per kilometer
  initialDistance : ℝ  -- Initial distance hiked east in kilometers
  totalDistance : ℝ  -- Total distance to hike east in kilometers

/-- Calculates the time needed to return to the start of the trail -/
def timeToReturn (scenario : HikingScenario) : ℝ :=
  let remainingDistance := scenario.totalDistance - scenario.initialDistance
  let timeToCompleteEast := remainingDistance * scenario.rate
  let timeToReturnWest := scenario.totalDistance * scenario.rate
  timeToCompleteEast + timeToReturnWest

/-- Theorem stating that Annika needs 35 minutes to return to the start -/
theorem annika_return_time :
  let scenario : HikingScenario := {
    rate := 10,
    initialDistance := 2.5,
    totalDistance := 3
  }
  timeToReturn scenario = 35 := by sorry

end NUMINAMATH_CALUDE_annika_return_time_l1849_184994


namespace NUMINAMATH_CALUDE_absolute_sum_zero_implies_sum_l1849_184950

theorem absolute_sum_zero_implies_sum (x y : ℝ) :
  |x - 1| + |y + 3| = 0 → x + y = -2 := by
sorry

end NUMINAMATH_CALUDE_absolute_sum_zero_implies_sum_l1849_184950


namespace NUMINAMATH_CALUDE_age_difference_l1849_184916

/-- Given the age ratios and sum of ages, prove the difference between Patrick's and Monica's ages --/
theorem age_difference (patrick_age michael_age monica_age : ℕ) : 
  patrick_age * 5 = michael_age * 3 →
  michael_age * 5 = monica_age * 3 →
  patrick_age + michael_age + monica_age = 147 →
  monica_age - patrick_age = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1849_184916


namespace NUMINAMATH_CALUDE_difference_divisible_by_99_l1849_184998

/-- Given a natural number, returns the number of its digits -/
def numDigits (n : ℕ) : ℕ := sorry

/-- Given a natural number, returns the number formed by reversing its digits -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Theorem: For any natural number with an odd number of digits, 
    the difference between the number and its reverse is divisible by 99 -/
theorem difference_divisible_by_99 (n : ℕ) (h : Odd (numDigits n)) :
  99 ∣ (n - reverseDigits n) := by sorry

end NUMINAMATH_CALUDE_difference_divisible_by_99_l1849_184998


namespace NUMINAMATH_CALUDE_trivia_team_distribution_l1849_184980

theorem trivia_team_distribution (total : ℕ) (not_picked : ℕ) (groups : ℕ) 
  (h1 : total = 58) 
  (h2 : not_picked = 10) 
  (h3 : groups = 8) :
  (total - not_picked) / groups = 6 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_distribution_l1849_184980


namespace NUMINAMATH_CALUDE_ribbon_fraction_per_gift_l1849_184939

theorem ribbon_fraction_per_gift 
  (total_fraction : ℚ) 
  (num_gifts : ℕ) 
  (h1 : total_fraction = 4 / 15) 
  (h2 : num_gifts = 5) : 
  total_fraction / num_gifts = 4 / 75 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_fraction_per_gift_l1849_184939


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1849_184926

theorem solution_set_inequality (x : ℝ) (h : x ≠ 0) :
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio 0 ∪ Set.Ici (1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1849_184926


namespace NUMINAMATH_CALUDE_exactly_two_numbers_satisfy_l1849_184956

/-- A function that returns true if a number satisfies the given property --/
def satisfies_property (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧  -- n is a two-digit number
  ∃ (a b : ℕ),
    n = 10 * a + b ∧  -- n is represented as 10a + b
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧  -- a and b are single digits
    (n - (a + b) / 2) % 10 = 4  -- the property holds

/-- The theorem stating that exactly two numbers satisfy the property --/
theorem exactly_two_numbers_satisfy :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ n, n ∈ s ↔ satisfies_property n :=
sorry

end NUMINAMATH_CALUDE_exactly_two_numbers_satisfy_l1849_184956


namespace NUMINAMATH_CALUDE_eight_valid_arrangements_l1849_184925

/-- A type representing the possible positions of a card -/
inductive Position
  | Original
  | Left
  | Right

/-- A type representing a card arrangement -/
def Arrangement := Fin 5 → Position

/-- A function to check if an arrangement is valid -/
def is_valid (arr : Arrangement) : Prop :=
  ∀ i : Fin 5, arr i = Position.Original ∨ arr i = Position.Left ∨ arr i = Position.Right

/-- The number of valid arrangements -/
def num_valid_arrangements : ℕ := sorry

/-- The main theorem: there are 8 valid arrangements -/
theorem eight_valid_arrangements : num_valid_arrangements = 8 := by sorry

end NUMINAMATH_CALUDE_eight_valid_arrangements_l1849_184925


namespace NUMINAMATH_CALUDE_last_locker_opened_l1849_184938

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the locker-opening process -/
def lockerProcess (n : Nat) : Nat → Nat :=
  sorry

/-- The number of lockers in the hall -/
def totalLockers : Nat := 512

/-- Theorem stating that the last locker opened is number 509 -/
theorem last_locker_opened :
  ∃ (process : Nat → Nat → LockerState),
    (∀ k, k ≤ totalLockers → process totalLockers k = LockerState.Open) ∧
    (∀ k, k < 509 → ∃ m, m < 509 ∧ process totalLockers m = LockerState.Open ∧ m > k) ∧
    process totalLockers 509 = LockerState.Open :=
  sorry

end NUMINAMATH_CALUDE_last_locker_opened_l1849_184938


namespace NUMINAMATH_CALUDE_f_range_l1849_184965

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+2) - 3

theorem f_range : Set.range f = Set.Ici (-7) := by sorry

end NUMINAMATH_CALUDE_f_range_l1849_184965


namespace NUMINAMATH_CALUDE_one_equals_a_l1849_184942

theorem one_equals_a (x y z a : ℝ) 
  (sum_eq : x + y + z = a) 
  (inv_sum_eq : 1/x + 1/y + 1/z = 1/a) : 
  x = a ∨ y = a ∨ z = a := by
sorry

end NUMINAMATH_CALUDE_one_equals_a_l1849_184942


namespace NUMINAMATH_CALUDE_product_equals_eight_l1849_184981

theorem product_equals_eight :
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eight_l1849_184981


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1849_184959

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - 2

-- State the theorem
theorem function_passes_through_point :
  f (-1) = -5 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1849_184959


namespace NUMINAMATH_CALUDE_sum_of_coordinates_zero_l1849_184934

/-- For all points (x, y) in the real plane where x + y = 0, prove that y = -x -/
theorem sum_of_coordinates_zero (x y : ℝ) (h : x + y = 0) : y = -x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_zero_l1849_184934


namespace NUMINAMATH_CALUDE_average_growth_rate_proof_l1849_184962

def initial_sales : ℝ := 50000
def final_sales : ℝ := 72000
def time_period : ℝ := 2

theorem average_growth_rate_proof :
  (final_sales / initial_sales) ^ (1 / time_period) - 1 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_average_growth_rate_proof_l1849_184962


namespace NUMINAMATH_CALUDE_solve_score_problem_l1849_184952

def score_problem (s1 s3 s4 : ℕ) (avg : ℚ) : Prop :=
  s1 ≤ 100 ∧ s3 ≤ 100 ∧ s4 ≤ 100 ∧
  s1 = 65 ∧ s3 = 82 ∧ s4 = 85 ∧
  avg = 75 ∧
  ∃ (s2 : ℕ), s2 ≤ 100 ∧ (s1 + s2 + s3 + s4 : ℚ) / 4 = avg ∧ s2 = 68

theorem solve_score_problem (s1 s3 s4 : ℕ) (avg : ℚ) 
  (h : score_problem s1 s3 s4 avg) : 
  ∃ (s2 : ℕ), s2 = 68 ∧ (s1 + s2 + s3 + s4 : ℚ) / 4 = avg :=
by sorry

end NUMINAMATH_CALUDE_solve_score_problem_l1849_184952


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_radius_relation_l1849_184987

/-- Given a tetrahedron with congruent triangular faces, an inscribed sphere,
    and a triangular face with known angles and circumradius,
    prove the relationship between the inscribed sphere radius and the face properties. -/
theorem tetrahedron_sphere_radius_relation
  (r : ℝ) -- radius of inscribed sphere
  (R : ℝ) -- radius of circumscribed circle of a face
  (α β γ : ℝ) -- angles of a triangular face
  (h_positive_r : r > 0)
  (h_positive_R : R > 0)
  (h_angle_sum : α + β + γ = π)
  (h_positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_congruent_faces : True) -- placeholder for the condition of congruent faces
  : r = R * Real.sqrt (Real.cos α * Real.cos β * Real.cos γ) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_radius_relation_l1849_184987


namespace NUMINAMATH_CALUDE_formula_satisfies_table_l1849_184979

def table : List (ℕ × ℕ) := [(1, 1), (2, 8), (3, 27), (4, 64), (5, 125)]

theorem formula_satisfies_table : ∀ (pair : ℕ × ℕ), pair ∈ table → (pair.2 : ℚ) = (pair.1 : ℚ) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_formula_satisfies_table_l1849_184979


namespace NUMINAMATH_CALUDE_equation_transformation_l1849_184911

theorem equation_transformation (m : ℝ) : 2 * m - 1 = 3 → 2 * m = 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l1849_184911


namespace NUMINAMATH_CALUDE_cars_with_no_features_l1849_184920

theorem cars_with_no_features (total : ℕ) (airbags : ℕ) (power_windows : ℕ) (sunroofs : ℕ)
  (airbags_power : ℕ) (airbags_sunroofs : ℕ) (power_sunroofs : ℕ) (all_features : ℕ) :
  total = 80 →
  airbags = 45 →
  power_windows = 40 →
  sunroofs = 25 →
  airbags_power = 20 →
  airbags_sunroofs = 15 →
  power_sunroofs = 10 →
  all_features = 8 →
  total - (airbags + power_windows + sunroofs - airbags_power - airbags_sunroofs - power_sunroofs + all_features) = 7 :=
by sorry

end NUMINAMATH_CALUDE_cars_with_no_features_l1849_184920


namespace NUMINAMATH_CALUDE_landscape_ratio_l1849_184993

/-- Given a rectangular landscape with the following properties:
  - length is 120 meters
  - contains a playground of 1200 square meters
  - playground occupies 1/3 of the total landscape area
  Prove that the ratio of length to breadth is 4:1 -/
theorem landscape_ratio (length : ℝ) (playground_area : ℝ) (breadth : ℝ) : 
  length = 120 →
  playground_area = 1200 →
  playground_area = (1/3) * (length * breadth) →
  length / breadth = 4 := by
sorry


end NUMINAMATH_CALUDE_landscape_ratio_l1849_184993


namespace NUMINAMATH_CALUDE_waitress_income_fraction_from_tips_l1849_184964

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- Theorem: Given a waitress's income where tips are 9/4 of salary,
    the fraction of income from tips is 9/13 -/
theorem waitress_income_fraction_from_tips 
  (income : WaitressIncome) 
  (h : income.tips = (9 : ℚ) / 4 * income.salary) : 
  income.tips / (income.salary + income.tips) = (9 : ℚ) / 13 := by
  sorry


end NUMINAMATH_CALUDE_waitress_income_fraction_from_tips_l1849_184964


namespace NUMINAMATH_CALUDE_prom_color_assignment_l1849_184931

-- Define the colors
inductive Color
| White
| Red
| Blue

-- Define a person's outfit
structure Outfit :=
  (dress : Color)
  (shoes : Color)

-- Define the problem statement
theorem prom_color_assignment :
  ∀ (tamara valya lida : Outfit),
    -- Only Tamara's dress and shoes were the same color
    (tamara.dress = tamara.shoes) ∧
    (valya.dress ≠ valya.shoes ∨ lida.dress ≠ lida.shoes) →
    -- Valya was in white shoes
    (valya.shoes = Color.White) →
    -- Neither Lida's dress nor her shoes were red
    (lida.dress ≠ Color.Red ∧ lida.shoes ≠ Color.Red) →
    -- All colors are used exactly once for dresses
    (tamara.dress ≠ valya.dress ∧ tamara.dress ≠ lida.dress ∧ valya.dress ≠ lida.dress) →
    -- All colors are used exactly once for shoes
    (tamara.shoes ≠ valya.shoes ∧ tamara.shoes ≠ lida.shoes ∧ valya.shoes ≠ lida.shoes) →
    -- The only valid assignment is:
    (tamara = ⟨Color.Red, Color.Red⟩ ∧
     valya = ⟨Color.Blue, Color.White⟩ ∧
     lida = ⟨Color.White, Color.Blue⟩) :=
by sorry

end NUMINAMATH_CALUDE_prom_color_assignment_l1849_184931


namespace NUMINAMATH_CALUDE_melanie_picked_seven_plums_l1849_184955

/-- The number of plums Melanie picked from the orchard -/
def plums_picked : ℕ := sorry

/-- The number of plums Sam gave to Melanie -/
def plums_from_sam : ℕ := 3

/-- The total number of plums Melanie has now -/
def total_plums : ℕ := 10

/-- Theorem stating that Melanie picked 7 plums from the orchard -/
theorem melanie_picked_seven_plums :
  plums_picked = 7 ∧ plums_picked + plums_from_sam = total_plums :=
sorry

end NUMINAMATH_CALUDE_melanie_picked_seven_plums_l1849_184955


namespace NUMINAMATH_CALUDE_percentage_difference_l1849_184985

theorem percentage_difference (a b : ℝ) (h : b ≠ 0) :
  (a - b) / b * 100 = 25 → a = 100 ∧ b = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1849_184985


namespace NUMINAMATH_CALUDE_part_one_part_two_l1849_184900

-- Define the propositions p and q
def p (a x : ℝ) : Prop := a < x ∧ x < 3 * a

def q (x : ℝ) : Prop := x^2 - 5*x + 6 < 0

-- Part 1
theorem part_one (x : ℝ) : 
  p 1 x ∧ q x → 2 < x ∧ x < 3 :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  (a > 0) → (∀ x : ℝ, q x → p a x) → 1 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1849_184900


namespace NUMINAMATH_CALUDE_point_set_characterization_l1849_184989

theorem point_set_characterization (x y : ℝ) : 
  (∀ t : ℝ, -1 ≤ t ∧ t ≤ 1 → t^2 + y*t + x ≥ 0) ↔ 
  (y ∈ Set.Icc (-2) 2 → x ≥ y^2/4) ∧ 
  (y < -2 → x ≥ -y - 1) ∧ 
  (y > 2 → x ≥ y - 1) := by
sorry

end NUMINAMATH_CALUDE_point_set_characterization_l1849_184989


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1849_184999

theorem trigonometric_identities :
  (Real.sin (75 * π / 180))^2 - (Real.cos (75 * π / 180))^2 = Real.sqrt 3 / 2 ∧
  Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1849_184999


namespace NUMINAMATH_CALUDE_sock_pair_count_l1849_184909

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_pairs (white brown blue green : ℕ) : ℕ :=
  white * brown + white * blue + white * green +
  brown * blue + brown * green +
  blue * green

/-- Theorem: There are 81 ways to choose a pair of socks of different colors
    from a drawer containing 5 white, 5 brown, 3 blue, and 2 green socks -/
theorem sock_pair_count :
  different_color_pairs 5 5 3 2 = 81 := by sorry

end NUMINAMATH_CALUDE_sock_pair_count_l1849_184909


namespace NUMINAMATH_CALUDE_count_negative_numbers_l1849_184936

theorem count_negative_numbers : 
  let expr1 := -(-3)
  let expr2 := -|(-3)|
  let expr3 := -(3^2)
  (if expr1 < 0 then 1 else 0) + 
  (if expr2 < 0 then 1 else 0) + 
  (if expr3 < 0 then 1 else 0) = 2 := by
sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l1849_184936


namespace NUMINAMATH_CALUDE_sin_plus_cos_theorem_l1849_184935

theorem sin_plus_cos_theorem (θ : Real) (a b : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_sin_2θ : Real.sin (2 * θ) = a)
  (h_cos_2θ : Real.cos (2 * θ) = b) :
  Real.sin θ + Real.cos θ = Real.sqrt (1 + a) := by
sorry

end NUMINAMATH_CALUDE_sin_plus_cos_theorem_l1849_184935


namespace NUMINAMATH_CALUDE_jason_after_school_rate_l1849_184997

/-- Calculates Jason's hourly rate for after-school work --/
def after_school_rate (total_earnings weekly_hours saturday_hours saturday_rate : ℚ) : ℚ :=
  let saturday_earnings := saturday_hours * saturday_rate
  let after_school_earnings := total_earnings - saturday_earnings
  let after_school_hours := weekly_hours - saturday_hours
  after_school_earnings / after_school_hours

/-- Theorem stating Jason's after-school hourly rate --/
theorem jason_after_school_rate :
  after_school_rate 88 18 8 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_after_school_rate_l1849_184997


namespace NUMINAMATH_CALUDE_min_value_of_m_range_of_x_l1849_184974

-- Define the conditions
def conditions (a b m : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a^2 + b^2 = 9/2 ∧ a + b ≤ m

-- Part I: Minimum value of m
theorem min_value_of_m (a b m : ℝ) (h : conditions a b m) :
  m ≥ 3 :=
sorry

-- Part II: Range of x
theorem range_of_x (x : ℝ) :
  (∀ a b m, conditions a b m → 2*|x-1| + |x| ≥ a + b) →
  x ≤ -1/3 ∨ x ≥ 5/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_m_range_of_x_l1849_184974


namespace NUMINAMATH_CALUDE_complex_equality_l1849_184949

theorem complex_equality (a b : ℝ) : (1 + Complex.I) + (2 - 3 * Complex.I) = a + b * Complex.I → a = 3 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l1849_184949


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1849_184986

def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometricSum a r n = 455/1365 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1849_184986


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1849_184915

/-- Given a quadratic equation x^2 - 6x + k = 0 with one root x₁ = 2,
    prove that k = 8 and the other root x₂ = 4 -/
theorem quadratic_equation_roots (k : ℝ) (x₁ x₂ : ℝ) 
    (h1 : x₁^2 - 6*x₁ + k = 0)
    (h2 : x₁ = 2) :
    k = 8 ∧ x₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1849_184915


namespace NUMINAMATH_CALUDE_vins_school_distance_l1849_184951

/-- The distance Vins rides to school -/
def distance_to_school : ℝ := sorry

/-- The distance Vins rides back home -/
def distance_back_home : ℝ := 7

/-- The number of round trips Vins made this week -/
def number_of_trips : ℕ := 5

/-- The total distance Vins rode this week -/
def total_distance : ℝ := 65

theorem vins_school_distance :
  distance_to_school = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_vins_school_distance_l1849_184951


namespace NUMINAMATH_CALUDE_boric_acid_solution_percentage_l1849_184966

/-- Proves that the percentage of boric acid in the first solution must be 1% 
    to create a 3% boric acid solution under the given conditions -/
theorem boric_acid_solution_percentage 
  (total_volume : ℝ) 
  (final_concentration : ℝ) 
  (volume1 : ℝ) 
  (volume2 : ℝ) 
  (concentration2 : ℝ) 
  (h1 : total_volume = 30)
  (h2 : final_concentration = 0.03)
  (h3 : volume1 = 15)
  (h4 : volume2 = 15)
  (h5 : concentration2 = 0.05)
  (h6 : volume1 + volume2 = total_volume)
  : ∃ (concentration1 : ℝ), 
    concentration1 = 0.01 ∧ 
    concentration1 * volume1 + concentration2 * volume2 = final_concentration * total_volume :=
by sorry

end NUMINAMATH_CALUDE_boric_acid_solution_percentage_l1849_184966


namespace NUMINAMATH_CALUDE_george_marbles_l1849_184961

/-- The number of red marbles in George's collection --/
def red_marbles (total : ℕ) (white : ℕ) (yellow : ℕ) (green : ℕ) : ℕ :=
  total - (white + yellow + green)

/-- Theorem stating the number of red marbles in George's collection --/
theorem george_marbles :
  let total := 50
  let white := total / 2
  let yellow := 12
  let green := yellow - yellow / 2
  red_marbles total white yellow green = 7 := by
  sorry

end NUMINAMATH_CALUDE_george_marbles_l1849_184961


namespace NUMINAMATH_CALUDE_base14_remainder_theorem_l1849_184958

-- Define a function to convert a base-14 integer to decimal
def base14ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (14 ^ i)) 0

-- Define our specific base-14 number
def ourNumber : List Nat := [1, 4, 6, 2]

-- Theorem statement
theorem base14_remainder_theorem :
  (base14ToDecimal ourNumber) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base14_remainder_theorem_l1849_184958


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1849_184960

theorem tan_alpha_plus_pi_fourth (α β : Real) 
  (h1 : Real.tan (α + β) = 3/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  Real.tan (α + π/4) = 7/23 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1849_184960


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l1849_184919

theorem integer_solutions_of_equation :
  ∀ x m : ℤ, (|x^2 - 1| + |x^2 - 4| = m * x) ↔ ((x = -1 ∧ m = 3) ∨ (x = 1 ∧ m = 3)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l1849_184919


namespace NUMINAMATH_CALUDE_cube_units_digits_eq_all_digits_l1849_184904

/-- The set of all single digits -/
def AllDigits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The set of units digits of integral perfect cubes -/
def CubeUnitsDigits : Set Nat :=
  {d | ∃ n : Nat, d = (n^3) % 10}

/-- Theorem: The set of units digits of integral perfect cubes
    is equal to the set of all single digits -/
theorem cube_units_digits_eq_all_digits :
  CubeUnitsDigits = AllDigits := by sorry

end NUMINAMATH_CALUDE_cube_units_digits_eq_all_digits_l1849_184904


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l1849_184941

theorem adult_ticket_cost (student_ticket_cost : ℝ) (total_tickets : ℕ) (total_revenue : ℝ) (adult_tickets : ℕ) (student_tickets : ℕ) :
  student_ticket_cost = 3 →
  total_tickets = 846 →
  total_revenue = 3846 →
  adult_tickets = 410 →
  student_tickets = 436 →
  ∃ (adult_ticket_cost : ℝ), adult_ticket_cost = 6.19 ∧
    adult_ticket_cost * adult_tickets + student_ticket_cost * student_tickets = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l1849_184941
